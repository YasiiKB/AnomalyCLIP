import json
import os
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from torchmetrics import (
    AUROC,
    ROC,
    AveragePrecision,
    ConfusionMatrix,
    F1Score,
    MeanMetric,
    PrecisionRecallCurve,
)
from torchmetrics.classification import Accuracy, MulticlassAUROC, Precision

from src import utils
from src.models.components.loss import ComputeLoss

log = utils.get_pylogger(__name__)


class AnomalyCLIPModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss: ComputeLoss,
        **kwargs,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # loss function
        self.criterion = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        # freezing backbone
        for p in self.net.image_encoder.parameters():
            p.requires_grad = False
        for p in self.net.text_encoder.parameters():
            p.requires_grad = False
        self.net.text_encoder.text_projection.requires_grad = True
        for p in self.net.token_embedding.parameters():
            p.requires_grad = False

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.dir_abn_loss = MeanMetric()
        self.dir_nor_loss = MeanMetric()
        self.topk_abn_loss = MeanMetric()
        self.bottomk_abn_loss = MeanMetric()
        self.topk_nor_loss = MeanMetric()
        self.smooth_loss = MeanMetric()
        self.sparse_loss = MeanMetric()

        self.roc = ROC(task="binary")
        self.auroc = AUROC(task="binary")
        self.pr_curve = PrecisionRecallCurve(task="binary")
        self.average_precision = AveragePrecision(task="binary")
        self.f1 = F1Score(task="binary")
        self.confmat = ConfusionMatrix(
            task="multiclass", num_classes=self.hparams.num_classes, normalize="true"
        )
        self.pr = Precision(task="binary")
        self.top1_accuracy = Accuracy(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            top_k=1,
            average=None,
        )
        self.top5_accuracy = Accuracy(
            task="multiclass",
            num_classes=self.hparams.num_classes,
            top_k=5,
            average=None,
        )
        self.mc_auroc = MulticlassAUROC(
            num_classes=self.hparams.num_classes, average=None, thresholds=None
        )
        self.mc_aupr = AveragePrecision(
            task="multiclass", num_classes=self.hparams.num_classes, average=None
        )

        self.labels = []
        self.abnormal_scores = []
        self.class_probs = []
    
    # forward pass of this model which is the same as in anomaly_clip.py
    def forward(
        self,
        image_features: torch.Tensor,
        labels,
        ncentroid: torch.Tensor,
        segment_size: int = 1,
        test_mode: bool = False,
    ):
        return self.net(
            image_features,
            labels,
            ncentroid,
            segment_size,
            test_mode,
        )

    def on_train_start(self):
        '''
        This fuction computes or loads the average embedding of the normal class (ncentroid) before traning starts
        
        Output:
            ncentroid (torch.Tensor): torch.Size([512]) = feature_size
        '''
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        save_dir = Path(self.hparams.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        ncentroid_file = Path(save_dir / "ncentroid.pt")

        if ncentroid_file.is_file():
            # file exists, load
            self.ncentroid = torch.load(ncentroid_file)
            print("Loaded ncentroid!")
        else:
            print(f"ncentroid file NOT found! Computing ncentroid from scratch...")
            with torch.no_grad(): # no need to compute gradients (freeze the model)

                # Data loader for the normal class
                loader = self.trainer.datamodule.train_dataloader_test_mode()

                # Initialize variables to accumulate the sum of embeddings and the total count
                embedding_sum = torch.zeros(self.net.embedding_dim).to(self.device)
                count = 0

                # if features are loaded from dataloader, use them directly
                if self.trainer.datamodule.hparams.load_from_features:

                    # extract features and labels in each batch
                    for nimage_features, nlabels, _, _ in loader:

                        # collaps all dimensions except the last one
                        nimage_features = nimage_features.view(-1, nimage_features.shape[-1])

                        # slice the features to match the number of labels
                        nimage_features = nimage_features[: len(nlabels.squeeze())]

                        # move to device
                        nimage_features = nimage_features.to(self.device)

                        # Accumulate the sum of embeddings of all batches
                        embedding_sum += nimage_features.sum(dim=0)
                        count += nimage_features.shape[0] # count the number of feature vectors

                # if raw images are loaded, first encode them using the image encoder
                else:
                    for nimages, nlabels, _, _ in loader:
                        b, t, c, h, w = nimages.size()
                        nimages = nimages.view(-1, c, h, w)
                        nimages = nimages[: len(nlabels.squeeze())]
                        nimages = nimages.to(self.device)
                        nimage_features = self.net.image_encoder(nimages)
                        
                        # Accumulate the sum of embeddings
                        embedding_sum += nimage_features.sum(dim=0)
                        count += nimage_features.shape[0]
                    
                # Normalize the average embedding  
                self.ncentroid = embedding_sum / count

                # Save the ncentroid  
                torch.save(self.ncentroid, ncentroid_file)


    def model_step(self, batch: Any):
        '''
        This function is used to compute the forward pass of the model
        Using AnomalyCLIP model from anomaly_clip.py 

        Args:
            batch (Any): batch of data from dataloader: batch = (image_features, label)

        Returns:
            logits (torch.Tensor): torch.Size([batch_size (64)* num_segments (32) * seg_length (16) , num_classes (17 for ShanghaiTech, 14 for UCF_Crime)])
                                 - output of SelectorModel (named 'similarity' in training_step)

            logits_topk (torch.Tensor): torch.Size([batch_size, num_classes])

            labels (torch.Tensor): torch.Size([batch_size]) -- do not go through the model but are used to compute the loss

            scores (torch.Tensor): torch.Size([batch_size * num_segments * seg_length])
                                 - btw 0 and 1 scores (output of TemporalModel after going into ClassificationHead)

            idx_topk_abn (torch.Tensor): torch.Size([num_segments, k=3])

            idx_topk_nor (torch.Tensor): torch.Size([num_segments, k=3])
            
            idx_bottomk_abn (torch.Tensor): torch.Size([num_segments, k=3])
            
            ? image_features (torch.Tensor): torch.Size([ batch_size (64), 1 ,num_segments (32) * seg_length (16), 512]) -- do not go through the model but are used in training_step
        '''
        # Load from dataloader in Train_mode (not Test_mode) --> batch = (image_features, label)
        nbatch, abatch = batch # ground truth (devide into two parts: normal and abnormal)
        nimage_features, nlabel = nbatch
        aimage_features, alabel = abatch
        image_features = torch.cat((aimage_features, nimage_features), 0) # [64,1, 512, 512] = batch_size (64), 1 ,num_segments (32) * seg_length (16), 512
        labels = torch.cat((alabel, nlabel), 0) # batch_size (64)
        
        # Normal index
        normal_idx = self.trainer.datamodule.hparams.normal_id

        # Load centroids  #?
        save_dir = Path(self.hparams.save_dir)
        centroids_file = Path(save_dir / "centroids.pt")
        ncentroid_file = Path(save_dir / "ncentroid.pt")
        
        if centroids_file.is_file():
            # file exists, load
            centroids = torch.load(centroids_file)
            for label in labels:
                if label.item() in centroids:
                    # print(f'Returning ncentroid for {label.item()}')
                    ncentroid = centroids[label.item()]
                else: 
                    print(f'ncentroid for {label.item()} not found! Returning general ncentroid...')
                    ncentroid = centroids[normal_idx]

        elif ncentroid_file.is_file():
            ncentroid = torch.load(ncentroid_file)
        
        else: # never happens because ncentroid is calculated/loaded automatically in on_train_start
            raise FileNotFoundError(f"centroids file {centroids_file} or ncentroid file {ncentroid_file} not found")


        # forward from anomaly_clip.py 
        (
            logits,
            logits_topk,
            scores,
            idx_topk_abn,
            idx_topk_nor,
            idx_bottomk_abn,
        ) = self.forward(
            image_features,
            labels,
            ncentroid,
        )  

        
        return (
            logits,
            logits_topk,
            labels,
            scores,
            idx_topk_abn,
            idx_topk_nor,
            idx_bottomk_abn,
            image_features
        )

    def training_step(self, batch: Any, batch_idx: int):
        '''
        This is where traning happens. This function performs the feed forward pass of the model (from model_step) 
        then computes and logs different losses 
        
        Args:
            batch (Any): batch of data from dataloader: batch = (image_features, label)
            batch_idx (int): index of the batch in the dataloader
        
        Logs: 
            train_loss(loss)
            dir_abn_loss(ldir_abn)
            dir_nor_loss(ldir_nor)
            topk_abn_loss(ltopk_abn)
            bottomk_abn_loss(lbottomk_abn)
            topk_nor_loss(ltopk_nor)
            smooth_loss(lsmooth)
            sparse_loss(lsparse)

        Returns:
            {"loss": loss} (dict): loss value for backpropagation
        '''
        
        # Forward pass from model_step (anomaly_clip.py)
        (
            similarity,
            similarity_topk,
            labels,
            scores,
            idx_topk_abn,
            idx_topk_nor,
            idx_bottomk_abn,
            image_features
        ) = self.model_step(batch)

        # Normal index
        normal_idx = self.trainer.datamodule.hparams.normal_id

        # Threshold for normality
        threshold = 0.4
        print(f"Threshold: {threshold}")

        # Get Dimensions
        batch_size = self.trainer.datamodule.hparams.batch_size # 64
        num_segments = self.trainer.datamodule.hparams.num_segments # 32
        seg_length = self.trainer.datamodule.hparams.seg_length # 16
        d = self.net.embedding_dim # 512

        # Change the shape of the image features [64,1,512,512] --> [512]
        image_features = torch.squeeze(image_features)  # batch_size, num_segments * seg_length, 512
        image_features = image_features.view(-1, d) # 512
        
        # Change the shape of the scores [64*32*16] --> [64, 32, 16] 
        new_scores = scores.view(batch_size, num_segments, seg_length)

        # Initialize a dictionary to store centroids for each label
        centroids = {}

        # Iterate over each label and extract the corresponding image feature
        for i, label in enumerate(labels):
            
            # Extract the image feature for the current label
            image_feature_for_label = image_features[i]
            
            # Extract the score for the current label
            score_for_label = new_scores[i]  #?

            # Check if the score is less than threshold (consider it normal)
            if score_for_label.mean() <= threshold:
                if label.item() not in centroids:
                    centroids[label.item()] = {
                        # initialize with the original ncentroid
                        "embedding_sum": torch.zeros_like(image_feature_for_label),
                        # "embedding_sum": torch.zeros(self.net.embedding_dim).to(self.device),
                        "count": 0 #1
                    }
                centroids[label.item()]["embedding_sum"] += image_feature_for_label
                centroids[label.item()]["count"] += 1

        # Calculate the centroid for each label
        for label in centroids:
            centroids[label] = centroids[label]["embedding_sum"] / centroids[label]["count"]

        # Replace the calculated centroid for normal videos with the original one
        centroids[normal_idx] = self.ncentroid  

        # Save the centroids
        save_dir = Path(self.hparams.save_dir)
        centroids_file = Path(save_dir / f"centroids.pt")
        torch.save(centroids, centroids_file)
        
        # for test
        # centroid_test_file = Path(save_dir / f"centroids_{self.current_epoch}_{batch_idx}.pt")
        # torch.save(centroids, centroid_test_file)

        # Compute loss
        (
            loss,
            ldir_abn,
            ldir_nor,
            ltopk_abn,
            lbottomk_abn,
            ltopk_nor,
            lsmooth,
            lsparse,
        ) = self.criterion(
            similarity,
            similarity_topk,
            labels,
            scores,
            idx_topk_abn,
            idx_topk_nor,
            idx_bottomk_abn,
        )

        # update and log metrics
        self.train_loss(loss)
        self.dir_abn_loss(ldir_abn)
        self.dir_nor_loss(ldir_nor)
        self.topk_abn_loss(ltopk_abn)
        self.bottomk_abn_loss(lbottomk_abn)
        self.topk_nor_loss(ltopk_nor)
        self.smooth_loss(lsmooth)
        self.sparse_loss(lsparse)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/dir_abn_loss",
            self.dir_abn_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/dir_nor_loss",
            self.dir_nor_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/topk_abn_loss",
            self.topk_abn_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/bottomk_abn_loss",
            self.bottomk_abn_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/topk_nor_loss",
            self.topk_nor_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/smooth_loss",
            self.smooth_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/sparse_loss",
            self.sparse_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # return loss or backpropagation will fail
        return {"loss": loss}

    def on_train_epoch_end(self):
        '''
        This function is called at the end of traning and validation of each epoch
        
        Args:
            ONLY self (no other arguments)
        '''     
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        '''
        This fuction loads the batch and ncentroid, performs the forward pass (from init which is the same as anomaly_clip.py in Test_mode)
        It computes conditional probabilities (softmax_similarity) and joint probabilities (class_probs) which is used in on_validation_epoch_end
        '''
        # Loading data in Test mode --> batch = (image_features, labels, label, segment_size)
        image_features, labels, label, segment_size = batch # labels for each frame (torch.size = num of frames), label for the entire video (torch.size = 1)
        image_features = image_features.to(self.device)
        labels = labels.squeeze(0).to(self.device) 

        # save_dir = Path(self.hparams.save_dir)
        # ncentroid_file = Path(save_dir / "ncentroid.pt")
        # if ncentroid_file.is_file():
        #     # file exists, load
        #     self.ncentroid = torch.load(ncentroid_file) #TO DO: need to load the new ones 
        # else:
        #     raise FileNotFoundError(f"ncentroid file {ncentroid_file} not found")
        
        # Normal index
        normal_idx = self.trainer.datamodule.hparams.normal_id

        # Load centroids  #?
        save_dir = Path(self.hparams.save_dir)
        centroids_file = Path(save_dir / "centroids.pt")
        ncentroid_file = Path(save_dir / "ncentroid.pt")
        
        if centroids_file.is_file():
            # file exists, load
            centroids = torch.load(centroids_file)
            # for label in labels
            if label.item() in centroids:
                # print(f'Returning ncentroid for {label.item()}')
                ncentroid = centroids[label.item()] 
            else:
                print(f'ncentroid for {label.item()} not found! Returning general ncentroid...')
                ncentroid = centroids[normal_idx]

        elif ncentroid_file.is_file():
            ncentroid = torch.load(ncentroid_file)

        else: 
            raise FileNotFoundError(f"centroids file {centroids_file} or ncentroid file {ncentroid_file} not found")

        # Forward pass from __init__ which is the same as anomaly_clip.py in Test_mode
        similarity, abnormal_scores = self.forward(
            image_features,
            labels,
            ncentroid,
            segment_size,
            test_mode=True,
        )

        # Compute conditional probabilities
        softmax_similarity = torch.softmax(similarity, dim=1)

        # Compute joint probabilities
        class_probs = softmax_similarity * abnormal_scores.unsqueeze(1)

        # Remove padded frames
        num_labels = labels.shape[0]
        class_probs = class_probs[:num_labels]
        abnormal_scores = abnormal_scores[:num_labels]

        self.labels.extend(labels)
        self.class_probs.extend(class_probs)
        self.abnormal_scores.extend(abnormal_scores)

    def on_validation_epoch_end(self):
        '''
        This function is called at the end of validation of each epoch.
        It uses joint probabilities (class_probs) and abnormal_scores to compute evaluation metrics such as 
            AUC, AUPR, mean_mc_auroc, mean_mc_aupr and log their results
        '''
        labels = torch.stack(self.labels)
        class_probs = torch.stack(self.class_probs)
        abnormal_scores = torch.stack(self.abnormal_scores)

        num_classes = self.trainer.datamodule.num_classes
        normal_idx = self.trainer.datamodule.hparams.normal_id

        # add normal probability to the class probabilities
        normal_probs = 1 - abnormal_scores
        normal_probs = normal_probs.unsqueeze(1)  # Add a new dimension to match class_probs shape
        class_probs = torch.cat(
            (
                class_probs[:, :normal_idx],
                normal_probs,
                class_probs[:, normal_idx:],
            ),
            dim=1,
        )

        labels_binary = torch.where(labels == normal_idx, 0, 1)

        fpr, tpr, thresholds = self.roc(abnormal_scores, labels_binary)
        auc_roc = self.auroc(abnormal_scores, labels_binary)

        optimal_idx = np.argmax(tpr.cpu().data.numpy() - fpr.cpu().data.numpy())
        optimal_threshold = thresholds[optimal_idx]

        precision, recall, thresholds = self.pr_curve(abnormal_scores, labels_binary)
        auc_pr = self.average_precision(abnormal_scores, labels_binary)

        mc_auroc = self.mc_auroc(class_probs, labels)
        mc_aupr = self.mc_aupr(class_probs, labels)

        mc_auroc_without_normal = torch.cat((mc_auroc[:normal_idx], mc_auroc[normal_idx + 1 :]))
        mc_auroc_without_normal[mc_auroc_without_normal == 0] = torch.nan
        mean_mc_auroc = torch.nanmean(mc_auroc_without_normal)

        mc_aupr_without_normal = torch.cat((mc_aupr[:normal_idx], mc_aupr[normal_idx + 1 :]))
        mc_aupr_without_normal[mc_aupr_without_normal == 0] = torch.nan
        mean_mc_aupr = torch.nanmean(mc_aupr_without_normal)

        self.log("test/AUC", auc_roc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/AP", auc_pr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mAUC", mean_mc_auroc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mAP", mean_mc_aupr, on_step=False, on_epoch=True, prog_bar=True)

        metrics = {
            "epoch": self.trainer.current_epoch,
            "auc_roc": auc_roc.item(),
            "auc_pr": auc_pr.item(),
            "mean_mc_auroc": mean_mc_auroc.item(),
            "mean_mc_aupr": mean_mc_aupr.item(),
            "mc_auroc": mc_auroc.tolist(),
            "mc_aupr": mc_aupr.tolist(),
            "optimal_threshold": optimal_threshold.item(),
        }

        save_dir = Path(self.hparams.save_dir)

        # Save metrics of each epoch (change number of epochs in configs/experiment -- trainer)
        with open(save_dir / f"metrics_{self.trainer.current_epoch}.json", "w") as fp:
            json.dump(metrics, fp, indent=4, sort_keys=True)

        # Clear lists 
        self.labels.clear() # to avoid dimensianality issues for each epoch
        self.class_probs.clear()
        self.abnormal_scores.clear()

    def on_test_start(self):
        # ckpt_path = Path(self.trainer.ckpt_path)
        # save_dir = os.path.normpath(ckpt_path.parent).split(os.path.sep)[-1]
        # save_dir = Path(os.path.join("src/app/logs/train/runs", str(save_dir))) # new save_dir for test results
        # if not save_dir.is_dir():
        #     save_dir.mkdir(parents=True, exist_ok=True)

        # ncentroid_file = Path(save_dir / "ncentroid.pt")  #? # no need to load the new centroids for test?

        # if ncentroid_file.is_file():
        #     # file exists, load
        #     self.ncentroid = torch.load(ncentroid_file)
        
        # Load centroids  #?
        save_dir = Path(self.hparams.save_dir)
        centroids_file = Path(save_dir / "centroids.pt")
        ncentroid_file = Path(save_dir / "ncentroid.pt")
        
        if centroids_file.is_file():
            # file exists, load
            centroids = torch.load(centroids_file)

        elif ncentroid_file.is_file():
            ncentroid = torch.load(ncentroid_file)

        # Recalculating the centroids on TRAINING data (not test) just to have it in the new save_dir
        else:
            with torch.no_grad():
                loader = self.trainer.datamodule.train_dataloader_test_mode()

                # Initialize variables to accumulate the sum of embeddings and the total count
                embedding_sum = torch.zeros(self.net.embedding_dim).to(self.device)
                count = 0

                if self.trainer.datamodule.hparams.load_from_features:
                    for nimage_features, nlabels, _, _ in loader:
                        nimage_features = nimage_features.view(-1, nimage_features.shape[-1])
                        nimage_features = nimage_features[: len(nlabels.squeeze())]
                        nimage_features = nimage_features.to(self.device)
                        embedding_sum += nimage_features.sum(dim=0)
                        count += nimage_features.shape[0]
                else:
                    for nimages, nlabels, _, _ in loader:
                        b, t, c, h, w = nimages.size()
                        nimages = nimages.view(-1, c, h, w)
                        nimages = nimages[: len(nlabels.squeeze())]
                        nimages = nimages.to(self.device)
                        nimage_features = self.net.image_encoder(nimages)
                        embedding_sum += nimage_features.sum(dim=0)
                        count += nimage_features.shape[0]

            # Compute and save the average embedding
            ncentroid = embedding_sum / count
            torch.save(ncentroid, ncentroid_file)

    @rank_zero_only 
    def test_step(self, batch: Any, batch_idx: int):
        image_features, labels, label, segment_size = batch
        image_features = image_features.to(self.device)
        labels = labels.squeeze(0).to(self.device)

        # Normal index
        normal_idx = self.trainer.datamodule.hparams.normal_id

        # Load centroids  #?
        save_dir = Path(self.hparams.save_dir)
        centroids_file = Path(save_dir / "centroids.pt")
        ncentroid_file = Path(save_dir / "ncentroid.pt")
        
        if centroids_file.is_file():
            # file exists, load
            centroids = torch.load(centroids_file)
            for label in labels:
                if label.item() in centroids:
                    # print(f'Returning ncentroid for {label.item()}')
                    ncentroid = centroids[label.item()]
                else: 
                    print(f'ncentroid for {label.item()} not found! Returning general ncentroid...')
                    ncentroid = centroids[normal_idx]

        elif ncentroid_file.is_file():
            ncentroid = torch.load(ncentroid_file)
        
        else: # never happens because ncentroid is calculated/loaded automatically in on_train_start
            raise FileNotFoundError(f"centroids file {centroids_file} or ncentroid file {ncentroid_file} not found")

        # Forward pass
        similarity, abnormal_scores = self.forward(
            image_features,
            labels,
            ncentroid,
            segment_size,
            test_mode=True,
        )

        # Compute conditional probabilities
        softmax_similarity = torch.softmax(similarity, dim=1)

        # Compute joint probabilities
        class_probs = softmax_similarity * abnormal_scores.unsqueeze(1)

        # Remove padded frames
        num_labels = labels.shape[0]
        class_probs = class_probs[:num_labels]
        abnormal_scores = abnormal_scores[:num_labels]

        return {
            "abnormal_scores": abnormal_scores,
            "labels": labels,
            "class_probs": class_probs,
        }

    @rank_zero_only
    def test_epoch_end(self, outputs: List[Any]):
        abnormal_scores = torch.cat([o["abnormal_scores"] for o in outputs])
        labels = torch.cat([o["labels"] for o in outputs])
        class_probs = torch.cat([o["class_probs"] for o in outputs])

        num_classes = self.trainer.datamodule.num_classes
        normal_idx = self.trainer.datamodule.hparams.normal_id

        # add normal probability to the class probabilities
        normal_probs = 1 - abnormal_scores
        normal_probs = normal_probs.unsqueeze(1)  # Add a new dimension to match class_probs shape
        class_probs = torch.cat(
            (
                class_probs[:, :normal_idx],
                normal_probs,
                class_probs[:, normal_idx:],
            ),
            dim=1,
        )

        labels_binary = torch.where(labels == normal_idx, 0, 1)

        fpr, tpr, thresholds = self.roc(abnormal_scores, labels_binary)
        auc_roc = self.auroc(abnormal_scores, labels_binary)

        optimal_idx = np.argmax(tpr.cpu().data.numpy() - fpr.cpu().data.numpy())
        optimal_threshold = thresholds[optimal_idx]

        precision, recall, thresholds = self.pr_curve(abnormal_scores, labels_binary)
        auc_pr = self.average_precision(abnormal_scores, labels_binary)

        class_probs_without_normal = torch.cat(
            (class_probs[:, :normal_idx], class_probs[:, normal_idx + 1 :]),
            dim=1,
        )

        # select predictions based on abnormal score and class probabilities
        y_pred = []
        for i in range(len(abnormal_scores)):
            if abnormal_scores[i] < optimal_threshold:
                y_pred.append(normal_idx)
            else:
                pred = torch.argmax(class_probs_without_normal[i])
                if pred >= normal_idx:
                    pred += 1
                y_pred.append(pred)
        y_pred = torch.tensor(y_pred).to(self.device)

        # compute top1, top5, and auc roc for each class
        top1_accuracy = torch.zeros(num_classes)
        top5_accuracy = torch.zeros(num_classes)

        top1_preds = torch.max(class_probs_without_normal, dim=1)[1]
        top1_preds = torch.where(top1_preds >= normal_idx, top1_preds + 1, top1_preds)
        top1_preds = torch.where(y_pred == normal_idx, normal_idx, top1_preds)
        top5_preds = torch.topk(class_probs_without_normal, k=5, dim=1)[1]
        top5_preds = torch.where(top5_preds >= normal_idx, top5_preds + 1, top5_preds)
        # if y_pred == normal_idx, then top5_preds = [normal_idx, top5_preds[0], top5_preds[1], top5_preds[2], top5_preds[3]], else top5_preds = top5_preds
        top5_preds = torch.where(
            y_pred.unsqueeze(1) == normal_idx,
            torch.cat(
                (
                    torch.tensor([normal_idx])
                    .unsqueeze(0)
                    .expand(top5_preds.shape[0], -1)
                    .to(self.device),
                    top5_preds[:, :4],
                ),
                dim=1,
            ),
            top5_preds,
        )

        for class_idx in range(num_classes):
            class_mask = (labels == class_idx).bool()
            class_preds = top1_preds[class_mask]
            class_labels = labels[class_mask]
            top1_accuracy[class_idx] = (class_preds == class_labels).float().mean()
            top5_accuracy[class_idx] = (
                (top5_preds[class_mask] == class_labels.view(-1, 1)).any(dim=1).float().mean()
            )

        mc_auroc = self.mc_auroc(class_probs, labels)
        mc_aupr = self.mc_aupr(class_probs, labels)

        mc_auroc_without_normal = torch.cat((mc_auroc[:normal_idx], mc_auroc[normal_idx + 1 :]))
        mc_auroc_without_normal[mc_auroc_without_normal == 0] = torch.nan
        mean_mc_auroc = torch.nanmean(mc_auroc_without_normal)

        mc_aupr_without_normal = torch.cat((mc_aupr[:normal_idx], mc_aupr[normal_idx + 1 :]))
        mc_aupr_without_normal[mc_aupr_without_normal == 0] = torch.nan
        mean_mc_aupr = torch.nanmean(mc_aupr_without_normal)

        ckpt_path = Path(self.trainer.ckpt_path)
        save_dir = os.path.normpath(ckpt_path.parent).split(os.path.sep)[-1]
        save_dir = Path(os.path.join("src/app/logs/eval/runs", str(save_dir))) # changed path
        if not save_dir.is_dir():
            save_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving results to {save_dir}")

        labels_df = pd.read_csv(self.trainer.datamodule.hparams.labels_file)
        classes = labels_df["id"].tolist()
        class_names = labels_df["name"].tolist()

        metrics = {
            "epoch": self.trainer.current_epoch,
            "auc_roc": auc_roc.item(),
            "auc_pr": auc_pr.item(),
            "mean_mc_auroc": mean_mc_auroc.item(),
            "mean_mc_aupr": mean_mc_aupr.item(),
            "mc_auroc": mc_auroc.tolist(),
            "mc_aupr": mc_aupr.tolist(),
            "top1_accuracy": top1_accuracy.tolist(),
            "top5_accuracy": top5_accuracy.tolist(),
            "optimal_threshold": optimal_threshold.item(),
        }

        with open(save_dir / "metrics.json", "w") as fp:
            json.dump(metrics, fp, indent=4, sort_keys=True)

        f1_scores = {}
        far = {}
        for i in range(10):
            thresh = (i + 1) / 10
            y_pred_binary = torch.where(abnormal_scores < thresh, 0, 1)
            f1_scores[thresh] = self.f1(y_pred_binary, labels_binary)

        # PR-Curve plot
        recall = recall.cpu().data.numpy()
        precision = precision.cpu().data.numpy()
        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        plt.style.use("ggplot")
        plt.ylim(0, 1.1)
        plt.plot(recall, precision, color="red")
        plt.title(f"PR Curve: {auc_pr*100:.2f}")
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        fig_file = save_dir / "PR.png"
        plt.savefig(fig_file)
        plt.close()

        # ROC-Curve plot
        fpr = fpr.cpu().data.numpy()
        tpr = tpr.cpu().data.numpy()
        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        plt.style.use("ggplot")
        plt.ylim(0, 1.1)
        plt.plot(fpr, tpr, color="blue")
        plt.title(f"ROC Curve: {auc_roc*100:.2f}")
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        fig_file = save_dir / "ROC.png"
        plt.savefig(fig_file)
        plt.close()

        # F1 score curve
        x = [(i + 1) / 10 for i in range(10)]
        y = [f1_scores[xx].cpu().data.numpy() for xx in x]
        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        plt.style.use("ggplot")
        # plt.ylim(0, 1.1)
        plt.plot(x, y, color="blue")
        plt.title(f"F1@0.5: {f1_scores[0.5]*100:.2f}")
        plt.ylabel("F1")
        plt.xlabel("threshold")
        fig_file = save_dir / "F1.png"
        plt.savefig(fig_file)
        plt.close()

        confmat = self.confmat(y_pred, labels)
        fig = plt.figure(figsize=(20, 18))
        ax = plt.subplot()  # /np.sum(cm)
        f = sns.heatmap(confmat.cpu().data.numpy(), annot=True, ax=ax, fmt=".2%", cmap="Blues")

        # labels, title and ticks
        ax.set_xlabel("Predicted", fontsize=20)
        ax.xaxis.set_label_position("bottom")
        plt.xticks(rotation=90)
        ax.xaxis.set_ticklabels(class_names, fontsize=15)
        ax.xaxis.tick_bottom()
        #
        ax.set_ylabel("True", fontsize=20)
        ax.yaxis.set_ticklabels(class_names, fontsize=15)
        plt.yticks(rotation=0)

        fig_file = save_dir / "confusion_matrix.png"
        plt.savefig(fig_file)
        plt.close()


    # def calculate_centroids(self, image_features, label, labels, test_mode):
    #     '''
    #     This function loads/calculates the average normal embedding for each class (ncentroid).
        
    #     Args:
    #         image_features (torch.Tensor): torch.Size([batch_size, feature_size])
    #         label (torch.Tensor): torch.Size([1]) --> label for the entire video
    #         labels (torch.Tensor): torch.Size([num of frames]) --> labels for each frame
    #         test_mode (bool): True if in test mode, False if in validation mode

    #     Returns:
    #         ncentroid_dict[video_class] (torch.Tensor): torch.Size([feature_size])

    #     RESULTS:
    #         This implementation of ncentroid calculation is reducing model performance! 
    #         Because: 
    #             1. it uses frame-level labels to calculate the ncentroid for each video class which are not accurate.
    #             2. it is called in each validation step which doesn't influence the model's weights.
    #     SOLUTIONS:
    #         1. To find actual normal/abnormal frames, I need to use the model's prediction. (IDEA: after a few warm-up epochs, on a pre-trained model)
    #         2. To influence the model's weights, I need to call this function in the training loop (training_step) and update the ncentroid_dict after each epoch.
    #     '''

    #     # Normal index
    #     normal_idx = self.trainer.datamodule.hparams.normal_id

    #     # Paths for saving/loading ncentroids
    #     if test_mode == False: # Validation
    #         save_dir = Path(self.hparams.save_dir)
    #         ncentroid_file_dict = Path(save_dir / "ncentroid_dict.pt")
    #         ncentroid_file = Path(save_dir / "ncentroid.pt")
    #     else: # Test
    #         ckpt_path = Path(self.trainer.ckpt_path)
    #         save_dir = os.path.normpath(ckpt_path.parent).split(os.path.sep)[-1]
    #         save_dir = Path(os.path.join("src/app/logs/train/runs", str(save_dir))) 
    #         if not save_dir.is_dir():
    #             save_dir.mkdir(parents=True, exist_ok=True)
    
    #         ncentroid_file = Path(save_dir / "ncentroid.pt")
    #         ncentroid_file_dict = Path(save_dir / "ncentroid_dict.pt")

    #     # Load the ncentroid dictionary
    #     if ncentroid_file_dict.is_file():
    #         # dictionary file exists, load
    #         ncentroid_dict = torch.load(ncentroid_file_dict)
    #         # print("Loaded ncentroid_dict!")
    #         # # print keys of the dictionary
    #         # keys_list = sorted(ncentroid_dict.keys())
    #         # print(keys_list)

    #     # Load the ncentroid file
    #     elif ncentroid_file.is_file():
    #         # dictionary file does not exist, initialize it by loading the ncentroid file
    #         ncentroid = torch.load(ncentroid_file)
    #         ncentroid_dict = {normal_idx: ncentroid} # 8 for ShanghaiTech dataset, 7 for UCF-Crime dataset, 4 for xd dataset

    #     # in Test mode, if ncentroid file is not found, calculate it from scratch
    #     elif test_mode == True: 
    #         with torch.no_grad():
    #             loader = self.trainer.datamodule.train_dataloader_test_mode()

    #             # Initialize variables to accumulate the sum of embeddings and the total count
    #             embedding_sum = torch.zeros(self.net.embedding_dim).to(self.device)
    #             count = 0

    #             if self.trainer.datamodule.hparams.load_from_features:
    #                 for nimage_features, nlabels, _, _ in loader:
    #                     nimage_features = nimage_features.view(-1, nimage_features.shape[-1])
    #                     nimage_features = nimage_features[: len(nlabels.squeeze())]
    #                     nimage_features = nimage_features.to(self.device)
    #                     embedding_sum += nimage_features.sum(dim=0)
    #                     count += nimage_features.shape[0]
    #             else:
    #                 for nimages, nlabels, _, _ in loader:
    #                     b, t, c, h, w = nimages.size()
    #                     nimages = nimages.view(-1, c, h, w)
    #                     nimages = nimages[: len(nlabels.squeeze())]
    #                     nimages = nimages.to(self.device)
    #                     nimage_features = self.net.image_encoder(nimages)
    #                     embedding_sum += nimage_features.sum(dim=0)
    #                     count += nimage_features.shape[0]

    #         # Compute and save the average embedding
    #         self.ncentroid = embedding_sum / count
    #         torch.save(self.ncentroid, ncentroid_file)

    #     # in Validation mode, raise error if ncentroid file is not found
    #     else:
    #         raise FileNotFoundError(f"ncentroid file or dictionary not found")
        
    #     # Get the anomaly class label
    #     video_class = label[0].item()
    
    #     # Check if the video class centroid needs to be recalculated
    #     if video_class not in ncentroid_dict:
    #         # print("Recalculating Centroids...")
    #         # print('Class:', video_class)

    #         # Initialize variables to accumulate the sum of embeddings and the total count
    #         embedding_sums = torch.zeros(self.net.embedding_dim).to(self.device)
    #         counts = 0

    #         # Iterate over the labels and image features to accumulate sums for the normal label
    #         for lbl, img_feat in zip(labels, image_features):
    #             if lbl.item() == normal_idx:
    #                 img_feat = img_feat.view(-1, img_feat.shape[-1]).to(self.device)
    #                 embedding_sums += img_feat.sum(dim=0)
    #                 counts += img_feat.shape[0]
            
    #         # Calculate the new centroid for the video class
    #         if counts > 0: # to avoid division by zero (NaN)
    #             new_centroid = embedding_sums / counts
    #             ncentroid_dict[video_class] = new_centroid
        
    #         # Save the updated centroids
    #         torch.save(ncentroid_dict, ncentroid_file_dict)
                    
    #     # Return the ncentroid for the specified label (video class)
    #     if video_class == label[0].item():
    #         try:
    #             # print(f'Returning ncentroid for {video_class}')
    #             return ncentroid_dict[video_class]
    #         except KeyError:
    #             # print(f'ncentroid for {video_class} not found! Returning general ncentroid...')
    #             return ncentroid_dict[normal_idx]
    
    # def calculate_centroids(self, image_features, labels, scores):

    #     # Normal index
    #     normal_idx = self.trainer.datamodule.hparams.normal_id

    #     # Threshold for normality
    #     threshold = 0.5

    #     # Get Dimensions
    #     batch_size = self.trainer.datamodule.hparams.batch_size # 64
    #     num_segments = self.trainer.datamodule.hparams.num_segments # 32
    #     seg_length = self.trainer.datamodule.hparams.seg_length # 16
    #     d = self.net.embedding_dim # 512

    #     # Change the shape of the image features [64,1,512,512] --> [512]
    #     image_features = torch.squeeze(image_features)  # batch_size, num_segments * seg_length, 512
    #     image_features = image_features.view(-1, d) # 512
        
    #     # Change the shape of the scores [64*32*16] --> [64, 32, 16] 
    #     new_scores = scores.view(batch_size, num_segments, seg_length)

    #     # Initialize a dictionary to store centroids for each label
    #     centroids = {}

    #     # Iterate over each label and extract the corresponding image feature
    #     for i, label in enumerate(labels):
            
    #         # Extract the image feature for the current label
    #         image_feature_for_label = image_features[i]
            
    #         # Extract the score for the current label
    #         score_for_label = new_scores[i]

    #         # Check if the score is less than threshold (consider it normal)
    #         if score_for_label.mean() < threshold:
    #             if label.item() not in centroids:
    #                 centroids[label.item()] = {
    #                     "embedding_sum": torch.zeros_like(image_feature_for_label),
    #                     # "embedding_sum": torch.zeros(self.net.embedding_dim).to(self.device),
    #                     "count": 0
    #                 }
    #             centroids[label.item()]["embedding_sum"] += image_feature_for_label
    #             centroids[label.item()]["count"] += 1

    #     # Calculate the centroid for each label
    #     for label in centroids:
    #         centroids[label] = centroids[label]["embedding_sum"] / centroids[label]["count"]

    #     # Replace the calculated centroid for normal videos with the original one
    #     centroids[normal_idx] = self.ncentroid  

    #     return centroids

    #     # # Save the centroids
    #     # save_dir = Path(self.hparams.save_dir)
    #     # centroids_file = Path(save_dir / f"centroids.pt")
    #     # torch.save(centroids, centroids_file)


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        param_list = []

        param_list.append(
            {
                "params": self.net.selector_model.parameters(),
                "lr": self.hparams.solver.lr * self.hparams.solver.selector_model_ratio,
                "name": "selector_model",
            }
        )
        param_list.append(
            {
                "params": self.net.temporal_model.parameters(),
                "lr": self.hparams.solver.lr * self.hparams.solver.temporal_model_ratio,
                "name": "temporal_model",
            }
        )
        param_list.append(
            {
                "params": self.net.prompt_learner.parameters(),
                "lr": self.hparams.solver.lr * self.hparams.solver.prompt_learner_ratio,
                "name": "prompt_learner",
            }
        )
        param_list.append(
            {
                "params": self.net.text_encoder.text_projection,
                "lr": self.hparams.solver.lr * self.hparams.solver.text_projection_ratio,
                "name": "text_projection",
            }
        )

        optimizer = self.optimizer(params=param_list)
        if self.scheduler is not None:
            successor = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(self.trainer.max_epochs)
            )
            scheduler = self.scheduler(optimizer=optimizer, successor=successor)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = AnomalyCLIPModule(None, None, None, None)
