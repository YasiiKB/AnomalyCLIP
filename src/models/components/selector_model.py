import torch
from torch import nn


class SelectorModel(nn.Module):
    '''
    Selector Model 

    takes image and abnormal text features, re-centers and normalizes them, applies scalar projection, 
    and if not in test mode, selects top-k and bottom-k segments 

    Args:
        classnames: list of class names
        normal_id: int, index of the normal class
        logit_scale: nn.Parameter, scale of the logits
        num_segments: int, number of segments
        seg_length: int, length of the segment
        select_idx_dropout_topk: float, dropout rate for top-k selection
        select_idx_dropout_bottomk: float, dropout rate for bottom-k selection
        num_topk: int, number of top-k segments
        num_bottomk: int, number of bottom-k segments
        
    Returns:
        logits: torch.Tensor of shape (batch, n_cls), scaler projection of image and text features
        logits_topk: torch.Tensor of shape (batch * k, n_cls), top-k logits
        logits_bottomk: torch.Tensor of shape (batch * k, n_cls), bottom-k logits
        idx_topk_abn: torch.Tensor of shape (batch/2, k), indices of top-k abnormal segments
        idx_topk_nor: torch.Tensor of shape (batch/2, k), indices of top-k normal segments
        idx_bottomk_abn: torch.Tensor of shape (batch/2, k), indices of bottom-k abnormal segments
    '''
    def __init__(
        self,
        classnames: list,
        normal_id: int,
        logit_scale: nn.Parameter,
        num_segments: int,
        seg_length: int,
        select_idx_dropout_topk: float,
        select_idx_dropout_bottomk: float,
        num_topk: int,
        num_bottomk: int,
    ):
        super().__init__()

        self.classnames = classnames
        self.normal_id = normal_id
        self.logit_scale = logit_scale
        self.num_segments = num_segments
        self.seg_length = seg_length
        self.select_idx_dropout_topk = select_idx_dropout_topk
        self.select_idx_dropout_bottomk = select_idx_dropout_bottomk
        self.num_topk = num_topk
        self.num_bottomk = num_bottomk

        self.bn_layer = nn.BatchNorm1d(len(classnames) - 1, affine=False)

    def forward(
        self,
        image_features,
        text_features,
        labels,
        ncentroid,
        test_mode,
    ):
        image_features = torch.reshape(
            image_features, (-1, image_features.shape[-1])
        )  # (ncrops * ncrops * t // 16, n_cls)

        text_features_except_normal = torch.cat(
            (
                text_features[: self.normal_id],
                text_features[(self.normal_id + 1) :],
            ),
            dim=0,
        )

        # Re-centering transformation
        print('labels in Selector_model', labels)
        text_features = text_features_except_normal - ncentroid  # num_classes - 1, 512
        image_features = image_features - ncentroid

        # Normalization
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )  # num_classes - 1, num_classes - 1

        # Scalar projection
        logits = image_features @ text_features.T

        # Normalization
        logits = self.bn_layer(logits)
        logits = logits.view(-1, logits.shape[-1])

        if test_mode:
            return logits
        else:
            logits = logits.view(
                -1, self.num_segments * self.seg_length, logits.shape[-1]
            )  # (batch, num_segments*seg_length, n_cls)

            topk_mask, bottomk_mask = self.generate_mask(logits)
            topk_mask = topk_mask.to(image_features.device)
            bottomk_mask = bottomk_mask.to(image_features.device)

            logits_topk, idx_topk = self.select_topk(logits, labels, topk_mask)
            idx_topk_abn, idx_topk_nor = (
                idx_topk[: idx_topk.shape[0] // 2],
                idx_topk[idx_topk.shape[0] // 2 :],
            )

            logits_bottomk, idx_bottomk = self.select_bottomk(logits, labels, bottomk_mask)
            idx_bottomk_abn = idx_bottomk[: idx_bottomk.shape[0] // 2]

            logits = logits.view(-1, logits.shape[-1])
            logits_topk = logits_topk.view(-1, logits_topk.shape[-1])
            logits_bottomk = logits_bottomk.view(-1, logits_bottomk.shape[-1])

            return (
                logits,
                logits_topk,
                logits_bottomk,
                idx_topk_abn,
                idx_topk_nor,
                idx_bottomk_abn,
            )

    def generate_mask(self, logits):
        '''
        Generate a mask with the desired percentage of zeros for each row
        
        Args: 
            logits: torch.Tensor of shape (batch, num_segments, n_cls)
        
        Returns:
            topk_mask: torch.Tensor of shape (batch, num_segments, n_cls)
            bottomk_mask: torch.Tensor of shape (batch, num_segments, n_cls)
        '''
        # Initialize a tensor of ones with shape (batch, num_segments)
        select_idx = torch.ones((logits.shape[0], self.num_segments))

        topk_select_idx = select_idx * (1 - self.select_idx_dropout_topk) # substract from 1 to ensure resulting tensors have values < 1
        bottomk_select_idx = select_idx * (1 - self.select_idx_dropout_bottomk)

        # Generate binary masks (tensor of 0s and 1s)
        topk_mask = (
            torch.bernoulli(topk_select_idx).unsqueeze(2).expand([-1, -1, logits.shape[-1]])
        )  # expand 2D mask (batch, num_segments) ---> 3D mask: (batch, num_segments, n_cls)

        bottomk_mask = (
            torch.bernoulli(bottomk_select_idx).unsqueeze(2).expand([-1, -1, logits.shape[-1]])
        )  # expand 2D mask (batch, num_segments) ---> 3D mask: (batch, num_segments, n_cls)

        if self.select_idx_dropout_topk == self.select_idx_dropout_bottomk:
            topk_mask = bottomk_mask

        return topk_mask, bottomk_mask

    def select_topk_idx(self, logits, labels, mask):
        '''
        Function to select the top-k indices of segments from a batch of video data based on their logits

        Args:
            logits: torch.Tensor of shape (batch, num_segments, n_cls)
            labels: torch.Tensor of shape (batch, 1)
            mask: torch.Tensor of shape (batch, num_segments, n_cls)
        
        Returns:
            idx_topk_abn: torch.Tensor of shape (batch/2, k_abn)
        '''

        b, t, num_classes = logits.shape  # b = batch size, t = num_segments, num_classes = n_cls

        # Reshape the logits tensor to have shape (batch, num_segments, seg_length, n_cls)
        logits_sum = logits.view(
            -1, self.num_segments, self.seg_length, num_classes
        ) 

        # Sum logits across the seg_length (2) dimension
        logits_sum = torch.sum(logits_sum, dim=2) 

        # Set the values to a low value where the mask is zero and leave the values unchanged where the mask is one
        min_value = -1e6
        logits_drop = torch.where(
            mask == 0, torch.ones_like(logits_sum) * min_value, logits_sum
        )  # (batch, num_segments, n_cls)

        # ------- Abnormal videos (assumed to be the first half of the batch) ------- 

        # Split the logits tensor into two halves: one for abnormal videos and the other for normal videos
        alogits_drop = logits_drop[: b // 2]

        # For each abnormal video, it returns the indices of the k_abn snippet most similar to their textual prompt
        alabels = labels[: b // 2]  # (batch//2, 1)
        alabels = torch.where(alabels > self.normal_id, alabels - 1, alabels)

        idx_topk_abn = []
        for alogits_i, alabels_i in zip(alogits_drop, alabels):
            # Get the indices [1] of the k_abn most abnormal snippets
            idx_topk_abn_i = torch.topk(alogits_i[:, alabels_i], self.num_topk, dim=0, largest=True,)[1]  # (k_abn) 

            idx_topk_abn_i = idx_topk_abn_i.unsqueeze(0)  # (1, num_topk)
            # collect indices in a list
            idx_topk_abn.append(idx_topk_abn_i)
        
        # Concatenate the list of indices to get a tensor
        idx_topk_abn = torch.cat(idx_topk_abn)  # (batch / 2, k_abn)

        # ------- Normal videos (assumed to be the second half of the batch) ------- 

        # Second half of the logits tensor for normal videos
        nlogits_drop = logits_drop[logits_drop.shape[0] // 2 :]
        nlogits_drop = torch.sum(nlogits_drop, dim=2)  # (batch, num_segments)

        # Get the indices of the k_abn most abnormal snippets (among the normal)
        idx_topk_nor = torch.topk(nlogits_drop, k=self.num_topk, dim=1, largest=True)[1]  # (batch / 2, k_abn)

        return idx_topk_abn, idx_topk_nor

    def select_topk(self, logits, labels, mask):
        '''
        Function to select the top-k segments from a batch of video data based on their logits

        Args:
            logits: torch.Tensor of shape (batch, num_segments, n_cls)
            labels: torch.Tensor of shape (batch, 1)
            mask: torch.Tensor of shape (batch, num_segments, n_cls)
        
        Returns:
            total_select_logits: torch.Tensor of shape (batch * k, seg_length, n_cls)
            idx_logits: torch.Tensor of shape (batch, k)
        '''
        b, t, num_classes = logits.shape # b = batch size, t = num_segments, num_classes = n_cls

        # Get the indices of the top-k segments for abnormal and normal videos
        idx_topk_abn, idx_topk_nor = self.select_topk_idx(logits, labels, mask)


        # ------- Abnormal videos -------

        idx_topk_abn_logits = idx_topk_abn.unsqueeze(2).expand(
            [-1, -1, num_classes]
        )  # (batch/2, k_abn, num_classes)

        alogits = logits[: logits.shape[0] // 2]
        alogits = alogits.view(
            -1, self.num_segments, self.seg_length, num_classes
        )  # (batch/2, num_segments, seg_length, n_cls)

        # expand indices of idx_most_abn_logits to (batch/2, num_topk, seg_length, n_cls)
        idx_topk_abn_logits = idx_topk_abn_logits.unsqueeze(2).expand(
            [-1, -1, self.seg_length, -1]
        )  # (batch/2, num_topk, seg_length, n_cls)

        total_select_abn_logits = []

        # loop through abnormal logits and their indices
        for a_logit, a_topk_idx in zip(alogits, idx_topk_abn_logits):
            # a_logit (num_segments, d)
            # a_topk_idx (k_abn, d)
            # Gathers values of a_logit along axis 0 with indices a_idx_most
            logit_topk_abn = torch.gather(
                a_logit, 0, a_topk_idx
            )  # 3 most abnormal snippets in abnormal bag  (k_abn, n_cls)
            total_select_abn_logits.append(logit_topk_abn)
        total_select_abn_logits = torch.cat(
            total_select_abn_logits
        )  # (batch/2*k_abn, seg_length, n_cls)


        # ------- Normal videos -------

        idx_topk_nor_logits = idx_topk_nor.unsqueeze(2).expand(
            [-1, -1, num_classes]
        )  # (batch/2, num_topk, 13)

        nlogits = logits[
            logits.shape[0] // 2 :
        ]  # normal feature logits (batch//2, num_segments * seg_length, n_cls) (batch_size//2, 32 , 14)
        nlogits = nlogits.view(
            -1, self.num_segments, self.seg_length, num_classes
        )  # (batch/2, num_segments, seg_length, n_cls)

        # expand indices of idx_most_abn_logits to (batch/2, k_abn, seg_length, n_cls)
        idx_topk_nor_logits = idx_topk_nor_logits.unsqueeze(2).expand(
            [-1, -1, self.seg_length, -1]
        )  # (batch/2, k_abn, seg_length, n_cls)

        total_select_nor_logits = []
        for n_logit, n_topk_idx in zip(nlogits, idx_topk_nor_logits):
            # n_logit (num_segments, d)
            # n_idx_least (k_abn, d)
            # Gathers values of n_logit along axis 0 with indices n_idx_least
            logit_topk_nor = torch.gather(
                n_logit, 0, n_topk_idx
            )  # 3 most abnormal snippets in normal bag  (k_abn, n_cls)
            total_select_nor_logits.append(logit_topk_nor)
        total_select_nor_logits = torch.cat(
            total_select_nor_logits
        )  # (batch/2*k_abn, seg_length, n_cls)

        # Concatenate the abnormal and normal logits to get the total logits selected as most abnormal for the bag
        total_select_logits = torch.cat(
            (total_select_abn_logits, total_select_nor_logits)
        )  # (batch * k, seg_length, n_cls)

        # Concatenate the indices of the abnormal and normal logits
        idx_logits = torch.cat((idx_topk_abn, idx_topk_nor), dim=0)

        return total_select_logits, idx_logits

    def select_bottomk_idx(self, logits, labels, mask):
        b, t, num_classes = logits.shape

        logits_sum = logits.view(
            -1, self.num_segments, self.seg_length, num_classes
        )  # (batch, num_segments, seg_length, n_cls)
        logits_sum = torch.sum(logits_sum, dim=2)

        # Set the values to a high value where the mask is zero and leave the values unchanged where the mask is one
        max_value = 1e6
        logits_drop = torch.where(
            mask == 0, torch.ones_like(logits_sum) * max_value, logits_sum
        )  # (batch, num_segments, n_cls)

        alogits_drop = logits_drop[: b // 2]

        # For each abnormal video, it returns the indices of the k_abn snippet most similar to their textual prompt
        alabels = labels[: b // 2]  # (batch//2, 1)
        alabels = torch.where(alabels > self.normal_id, alabels - 1, alabels)

        idx_bottomk_abn = []
        for alogits_i, alabels_i in zip(alogits_drop, alabels):
            idx_bottomk_abn_i = torch.topk(
                alogits_i[:, alabels_i],
                self.num_bottomk,
                dim=0,
                largest=False,
            )[
                1
            ]  # (num_bottomk)
            idx_bottomk_abn_i = idx_bottomk_abn_i.unsqueeze(0)  # (1, num_topk)
            idx_bottomk_abn.append(idx_bottomk_abn_i)
        idx_bottomk_abn = torch.cat(idx_bottomk_abn)  # (batch / 2, num_topk)

        nlogits_drop = logits_drop[b // 2 :]
        nlogits_drop = torch.sum(nlogits_drop, dim=2)  # (batch, num_segments)
        idx_bottomk_nor = torch.topk(nlogits_drop, k=self.num_bottomk, dim=1, largest=False)[
            1
        ]  # (batch / 2, num_bottomk)

        return idx_bottomk_abn, idx_bottomk_nor

    def select_bottomk(self, logits, labels, mask):
        b, t, num_classes = logits.shape

        idx_bottomk_abn, idx_bottomk_nor = self.select_bottomk_idx(logits, labels, mask)

        idx_bottomk_abn_logits = idx_bottomk_abn.unsqueeze(2).expand(
            [-1, -1, num_classes]
        )  # (batch/2, k_abn, num_classes)

        alogits = logits[: logits.shape[0] // 2]
        alogits = alogits.view(
            -1, self.num_segments, self.seg_length, num_classes
        )  # (batch/2, num_segments, seg_length, n_cls)
        # expand indices of idx_most_abn_logits to (batch/2, num_topk, seg_length, n_cls)
        idx_bottomk_abn_logits = idx_bottomk_abn_logits.unsqueeze(2).expand(
            [-1, -1, self.seg_length, -1]
        )  # (batch/2, num_topk, seg_length, n_cls)

        total_select_abn_logits = []
        for a_logit, a_bottomk_idx in zip(alogits, idx_bottomk_abn_logits):
            # a_logit (num_segments, d)
            # a_bottomk_idx (num_bottomk, d)
            # Gathers values of a_logit along axis 0 with indices a_idx_most
            logit_bottomk_abn = torch.gather(
                a_logit, 0, a_bottomk_idx
            )  # 3 most abnormal snippets in abnormal bag  (k_abn, n_cls)
            total_select_abn_logits.append(logit_bottomk_abn)
        total_select_abn_logits = torch.cat(
            total_select_abn_logits
        )  # (batch/2*k_abn, seg_length, n_cls)

        idx_bottomk_nor_logits = idx_bottomk_nor.unsqueeze(2).expand(
            [-1, -1, num_classes]
        )  # (batch/2, num_topk, 13)

        nlogits = logits[
            logits.shape[0] // 2 :
        ]  # normal feature logits (batch//2, num_segments * seg_length, n_cls) (batch_size//2, 32 , 14)
        nlogits = nlogits.view(
            -1, self.num_segments, self.seg_length, num_classes
        )  # (batch/2, num_segments, seg_length, n_cls)
        # expand indices of idx_most_abn_logits to (batch/2, num_bottomk, seg_length, n_cls)
        idx_bottomk_nor_logits = idx_bottomk_nor_logits.unsqueeze(2).expand(
            [-1, -1, self.seg_length, -1]
        )  # (batch/2, num_bottomk, seg_length, n_cls)

        total_select_nor_logits = []
        for n_logit, n_bottomk_idx in zip(nlogits, idx_bottomk_nor_logits):
            # n_logit (num_segments, d)
            # n_bottomk_idx (num_bottomk, d)
            # Gathers values of n_logit along axis 0 with indices n_idx_least
            logit_bottomk_nor = torch.gather(
                n_logit, 0, n_bottomk_idx
            )  # 3 most abnormal snippets in normal bag  (num_bottomk, n_cls)
            total_select_nor_logits.append(logit_bottomk_nor)
        total_select_nor_logits = torch.cat(
            total_select_nor_logits
        )  # (batch/2*num_bottomk, seg_length, n_cls)

        total_select_logits = torch.cat(
            (total_select_abn_logits, total_select_nor_logits)
        )  # (batch * k, seg_length, n_cls)

        idx_logits = torch.cat((idx_bottomk_abn, idx_bottomk_nor), dim=0)

        return total_select_logits, idx_logits
