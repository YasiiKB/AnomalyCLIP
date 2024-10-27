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

        # threshold for normality 
        self.nthreshold = 1

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
        self.centroids = {}
    
    
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

        # Path
        ncentroid_file = Path(save_dir / "ncentroid.pt")
        centroids_file = Path(save_dir / f"centroids_{self.nthreshold}.pt")

        # ncentroid file exists, load
        if ncentroid_file.is_file():
            self.ncentroid = torch.load(ncentroid_file)

        # ncentroid file does not exist, compute from scratch
        else:
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

        # centroids file exists, load
        if centroids_file.is_file():
            self.centroids = torch.load(centroids_file) # dictionary of tensors

        # centroids file does not exist, intialize with ncentroid
        else:
            # Initialize centroids dict with ncentroid
            normal_idx = self.trainer.datamodule.hparams.normal_id
            self.centroids[normal_idx] = self.ncentroid 
            torch.save(self.centroids, centroids_file)

    def model_step(self, batch: Any):
        '''
        This function is used to compute the forward pass of the model
        Using AnomalyCLIP model from anomaly_clip.py 

        Args:
            batch (Any): batch of data from dataloader: batch = (image_features, label)

        Returns:
            logits (torch.Tensor): torch.Size([batch_size (64)* num_segments (32) * seg_length (16) , num_classes (17 for ShanghaiTech, 14 for UCF_Crime)]) -- output of SelectorModel (named 'similarity' in training_step)
            logits_topk (torch.Tensor): torch.Size([batch_size, num_classes])
            labels (torch.Tensor): torch.Size([batch_size]) -- do not go through the model but are used to compute the loss
            scores (torch.Tensor): torch.Size([batch_size * num_segments * seg_length]) -- btw 0 and 1 scores (output of TemporalModel after going into ClassificationHead)
            idx_topk_abn (torch.Tensor): torch.Size([num_segments, k=3])
            idx_topk_nor (torch.Tensor): torch.Size([num_segments, k=3])
            idx_bottomk_abn (torch.Tensor): torch.Size([num_segments, k=3])
            image_features (torch.Tensor): torch.Size([ batch_size (64), 1 ,num_segments (32) * seg_length (16), 512]) -- do not go through the model but are used in training_step
        '''
        # Load from dataloader in Train_mode (not Test_mode) --> batch = (image_features, label)
        nbatch, abatch = batch # ground truth (devide into two parts: normal and abnormal)
        nimage_features, nlabel = nbatch
        aimage_features, alabel = abatch
        image_features = torch.cat((aimage_features, nimage_features), 0) # [64,1, 512, 512] = batch_size (64), 1 ,num_segments (32) * seg_length (16), 512
        labels = torch.cat((alabel, nlabel), 0) # batch_size (64)
        
        # Normal index
        normal_idx = self.trainer.datamodule.hparams.normal_id

        # Load centroids from on_train_start
        for label in labels:
            if label.item() in self.centroids:
                self.ncentroid = self.centroids[label.item()]
            else: 
                self.ncentroid = self.centroids[normal_idx]

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
            ncentroid=self.ncentroid,
        )  

        
        return (
            logits,
            logits_topk,
            labels,
            scores,
            idx_topk_abn,
            idx_topk_nor,
            idx_bottomk_abn,
            image_features,
        )

    def training_step(self, batch: Any, batch_idx: int):
        '''
        This is where traning happens. This function performs the feed forward pass of the model (from model_step) 
        then computes and logs different losses 
        
        Args:
            batch (Any): batch of data from dataloader: batch = (image_features, label)
            batch_idx (int): index of the batch in the dataloader

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
            image_features,
        ) = self.model_step(batch)

        # Normal index
        normal_idx = self.trainer.datamodule.hparams.normal_id

        # Get Dimensions
        batch_size = self.trainer.datamodule.hparams.batch_size # 64
        num_segments = self.trainer.datamodule.hparams.num_segments # 32
        seg_length = self.trainer.datamodule.hparams.seg_length # 16
        d = self.net.embedding_dim # 512

        # Change the shape of the image features [64,1,512,512] --> [512]
        image_features = torch.squeeze(image_features)  # batch_size, num_segments * seg_length, 512
        image_features = image_features.view(-1, d) # 512
        # image_features = image_features.to(self.device)

        # Change the shape of the scores [64*32*16] --> [64, 32* 16] 
        new_scores = scores.view(batch_size, num_segments* seg_length) 

        # Initialize a dictionary to store centroids for each video (Label)
        updates = {}

        # Initialize lists to store the indices of most normal/abnormal frames for each videos in the batch
        abatch_topk_indices = []

        # Iterate over each video in the batch
        for i, label in enumerate(labels):
  
            # Initialize lists to store the indices of abnormal frames for the current video (label)
            avideo_topk_indices= []
            
            # Extract the scores for the current label (video) and add a new dimension
            scores_for_label = new_scores[i].unsqueeze(-1)  # num_segments* seg_length, 1 (32*16, 1)

            # Calculate max score ans std of scores for the current label (scaler)
            max_score = torch.max(scores_for_label).item()
            std_score = torch.std(scores_for_label).item()

            # Iterate over each frame in the video
            for idx, score in enumerate(scores_for_label):

                # Consider frames whose scores are less than or equal to the threshold normal
                if score.item() <= self.nthreshold:
                    if label.item() not in updates: #otherwise, all centroids will be the same 
                        updates[label.item()] = {
                            # initialize with the original ncentroid
                            "embedding_sum": torch.clone(self.ncentroid),
                            "count": 1
                        }
                    # Get the corresponding frame index (idx) and its image_features (512)
                    corresponding_image_feature = image_features[idx]

                    updates[label.item()]["embedding_sum"] += corresponding_image_feature
                    updates[label.item()]["count"] += 1 # not image_feature_for_label.shape[0] because we're initializing with ncentroid #?
                    # updates[label.item()]["count"] += image_feature_for_label.shape[0] #very small amounts for centroids -- too far from ncentroid

                # Get the indices of abnormal frames whose scores are btw max and std (most abnormal frames)
                if label.item() != normal_idx and score.item() > (max_score - std_score): 
                    avideo_topk_indices.append(idx)         # different length for each video
    
            # Make a list of avideo_topk_indices 
            if len(avideo_topk_indices) != 0:
                abatch_topk_indices.append(avideo_topk_indices) # a list, with len = 32, of lists with len(video_topk_indices)
        
        # ---- Flexible abnormal Topk---- (Incomplete)
        # Get the max lenght of the topk indices
        max_length = max(len(inner_list) for inner_list in abatch_topk_indices)

        # Pad the inner lists to have the same length
        padded_abatch_topk_indices = [
            inner_list + [0] * (max_length - len(inner_list)) for inner_list in abatch_topk_indices]

        # Convert the list of lists to a tensor
        abatch_topk_indices_tensor = torch.tensor(padded_abatch_topk_indices).to(self.device) #32, max_length

        # Update idx_topk_abn to abatch_topk_indices
        self.net.num_topk = max_length
        # idx_topk_abn = abatch_topk_indices_tensor 

        # ---- Multi Centroid ----
        # Calculate the centroid for each label
        for label in updates:
            updates[label] = updates[label]["embedding_sum"] / updates[label]["count"]

        # Replace the calculated centroid for normal videos with the original one (try without this)
        updates[normal_idx] = self.ncentroid  

        # Update the centroids
        self.centroids.update(updates)

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
        This function is called at the end of traning and validation of each epoch.
        '''
        # UNCOMMENT to Save the final centroids at the end of the last epoch
        # save_dir = Path(self.hparams.save_dir)
        # centroids_file = Path(save_dir / f"centroids_{self.nthreshold}.pt")
        # torch.save(self.centroids, centroids_file)
        # print(f'Centroids_{self.nthreshold} saved')

    def validation_step(self, batch: Any, batch_idx: int):
        '''
        This fuction loads the batch and ncentroid, performs the forward pass (from init which is the same as anomaly_clip.py in Test_mode)
        It computes conditional probabilities (softmax_similarity) and joint probabilities (class_probs) which is used in on_validation_epoch_end
        '''
        # Loading data in Test mode --> batch = (image_features, labels, label, segment_size)
        image_features, labels, label, segment_size = batch # labels for each frame (torch.size = num of frames), label for the entire video (torch.size = 1)
        image_features = image_features.to(self.device)
        labels = labels.squeeze(0).to(self.device) 
        
        # Normal index
        normal_idx = self.trainer.datamodule.hparams.normal_id
        
        # for each label in the batch, get the corresponding centroid (if found) or ncentroid
        for label in labels:
            if label.item() in self.centroids:
                self.ncentroid = self.centroids[label.item()]
            else: 
                self.ncentroid = self.centroids[normal_idx]

        # Forward pass from __init__ which is the same as anomaly_clip.py in Test_mode
        similarity, abnormal_scores = self.forward(
            image_features,
            labels,
            self.ncentroid,
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

        # Path 
        # save_dir = Path('logs/train/runs/ucfcrime') # for testing Helmond dataset
        save_dir = Path('logs/train/runs/shanghaitech') # for testing Helmond dataset
        # save_dir = Path('logs/train/runs/xdviolence') # for testing Helmond dataset

        ncentroid_file = Path(save_dir / "ncentroid.pt")
        centroids_file = Path(save_dir / f"centroids_{self.nthreshold}.pt")

        # ncentroid file exists, load
        if ncentroid_file.is_file():
            self.ncentroid = torch.load(ncentroid_file)
            
        # ncentroid file does not exist, compute from scratch on TRAINING data (not test)
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
            self.ncentroid = embedding_sum / count
            torch.save(self.ncentroid, ncentroid_file)
    
        # centroids file exists, load
        if centroids_file.is_file():
            self.centroids = torch.load(centroids_file) # dictionary of tensors

        # centroids file does not exist, intialize with ncentroid
        else:
            normal_idx = self.trainer.datamodule.hparams.normal_id
            self.centroids[normal_idx] = self.ncentroid 
            torch.save(self.centroids, centroids_file)


    @rank_zero_only 
    def test_step(self, batch: Any, batch_idx: int):
        image_features, labels, label, segment_size = batch
        image_features = image_features.to(self.device)
        labels = labels.squeeze(0).to(self.device)

        # Initialize lists to store the results of the forward pass for each centroid
        all_similarities = []
        all_abnormal_scores = []

        # Iterate through each centroid in the dictionary
        for centroid_key, centroid in self.centroids.items():
            print(f"Testing centroid {centroid_key}")
            # Forward pass for the current centroid
            similarity, abnormal_scores = self.forward(
                image_features,
                labels,
                centroid,
                segment_size,
                test_mode=True,
            )

            # Store the results
            all_similarities.append(similarity)
            all_abnormal_scores.append(abnormal_scores)

        # Concatenate the results along the appropriate dimension
        all_similarities = torch.cat(all_similarities, dim=0)
        all_abnormal_scores = torch.cat(all_abnormal_scores, dim=0)

        # Compute conditional probabilities
        softmax_similarity = torch.softmax(all_similarities, dim=1)

        # Compute joint probabilities
        class_probs = softmax_similarity * all_abnormal_scores.unsqueeze(1)

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
        fig_file = save_dir / f"PR_{self.nthreshold}.png"
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
        fig_file = save_dir / f"ROC_{self.nthreshold}.png"
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
        fig_file = save_dir / f"F1_{self.nthreshold}.png"
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

        ## For Helmond dataset on UCF_Crime and ShanghaiTech models
        # ax.xaxis.set_major_locator(plt.FixedLocator(range(len(class_names[:7]))))
        # ax.xaxis.set_ticklabels(class_names[:7], fontsize=15)
        
        ax.xaxis.set_ticklabels(class_names, fontsize=15)
        ax.xaxis.tick_bottom()
        
        ax.set_ylabel("True", fontsize=20)

        ## For Helmond dataset on UCF_Crime and ShanghaiTech models
        # ax.yaxis.set_major_locator(plt.FixedLocator(range(len(class_names[:7]))))
        # ax.yaxis.set_ticklabels(class_names[:7], fontsize=15)

        ax.yaxis.set_ticklabels(class_names, fontsize=15)
        plt.yticks(rotation=0)

        fig_file = save_dir / f"confusion_matrix_{self.nthreshold}.png"
        plt.savefig(fig_file)
        plt.close()


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
