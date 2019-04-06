"""handles joint training of entire system"""

from typing import Tuple, Sequence

import torch
from torch import Tensor
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset
from torch.optim import SGD
import numpy as np
from tensorboardX import SummaryWriter
from ml_utils.data import get_subset_lengths
from ml_utils.prediction_filtering import PredictionFilterPipeline

from .data import ImageInstance
from .data.encoding import (
    AnchorEncoder,
    RegionEncoder,
    frcnn_box_decode,
    track_encode
)
from .loss import RPNLoss, RCNNLoss, TrackLoss
from .models import DetectTrackModule, ResNetFeatures
from .utils import tensor_to_ndarray, make_input_transform


class DetectTrackTrainer:
    """approximate joint training for two stage object detectors.
    ignores detector head loss wrt to region proposal.
    this can be (but is not currently) addressed by substituting the
    ROIPooling layer for a ROIWarping layer.
    see https://arxiv.org/abs/1506.01497.

    Args:
        model:
        trn_set: training set.
        val_set: validation set.
        split_size: number of training examples to train on before
            validating and reporting.
        net_input_hw: height and width of network input tensor.
        anchor_encoder: assigns ground-truth labels and bounding boxes
            anchorwise.
        region_encoder: assigns ground-truth labels and bounding boxes
            regionwise.
        region_filter: given a set of region proposals, returns a higher
            confidence subset of proposals.
        alpha: loss alpha balancing factor.
        gamma: loss focusing factor.
        loss_coefs: leading coefficient for each element of joint loss.
            gradients are backpropagated from dot(loss_coefs, losses)
        sgd_kwargs: parameters for stochastic gradient descent.
        tboard_writer: tensorboard logger.
    """
    def __init__(
            self,
            model: DetectTrackModule,
            trn_set: Dataset,
            val_set: Dataset,
            split_size: int,
            net_input_hw: int,
            anchor_encoder: AnchorEncoder,
            region_encoder: RegionEncoder,
            region_filter: PredictionFilterPipeline,
            alpha: float,
            gamma: float,
            loss_coefs: Sequence[float],
            sgd_kwargs: dict,
            tboard_writer: SummaryWriter
    ) -> None:
        ### models
        self._im_to_x = make_input_transform(net_input_hw)
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        self.model = model.cuda()

        ### datasets
        self.trn_set = trn_set
        self.val_set = val_set
        self._subset_lens = get_subset_lengths(len(self.trn_set), split_size)

        ### ground-truth label encoding
        self._anchor_encoder = anchor_encoder  # anchorwise label assignment
        self._region_encoder = region_encoder  # regionwise label assignment
        self._region_filter = region_filter  # filters rois before rcnn

        ### loss functions
        self._rpn_loss_func = RPNLoss(alpha, gamma)
        self._rcnn_loss_func = RCNNLoss(alpha, gamma)
        self._track_loss_func = TrackLoss()

        ### optimizers
        self._loss_coefs = loss_coefs
        self._optim = SGD(self.model.parameters(), **sgd_kwargs)

        self.tboard_writer = tboard_writer

        self.n_iters = 0

    def _forward_loss(
            self, instance: Tuple[ImageInstance, ImageInstance]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """compute joint loss for a single instance.

        Args:
            instance: (image, labels) tuple for time t, t+tau.

        Returns:
            o_loss_rpn: RPN binary classification loss.
            b_loss_rpn: RPN bounding box regression loss.
            c_loss_rcnn: RCNN multiclass classification loss.
            b_loss_rcnn: RCNN bounding box regression loss.
            t_loss: cross-frame tracking loss.
        """
        inst_0, inst_1 = instance

        ### extract feature maps.
        x0 = self._im_to_x(inst_0.im)  # (3, H, W)
        x1 = self._im_to_x(inst_1.im)  # (3, H, W)
        x = torch.stack([x0, x1])  # (2, H, W)
        x = x.cuda()
        fmaps = self.model.backbone(x)  # pyramid of feature maps 3*(2, ...)

        ### compute losses for RPN
        ###   - inputs are feature maps
        ###   - supervision from ground-truth labels
        # RPN label encoding.
        lw0_rpn, c0_star_rpn, b0_star_rpn = self._anchor_encoder(inst_0.labels)
        lw1_rpn, c1_star_rpn, b1_star_rpn = self._anchor_encoder(inst_1.labels)
        lw_rpn = np.stack([lw0_rpn, lw1_rpn])  # (2, |A|)
        c_star_rpn = np.stack([c0_star_rpn, c1_star_rpn])  # (2, |A|)
        b_star_rpn = np.stack([b0_star_rpn, b1_star_rpn])  # (2, |A|, 4)
        # RPN predictions.
        o_hat_rpn, b_hat_rpn, fm_reg = self.model.rpn(fmaps.c4)
        # RPN loss.
        lw_rpn = torch.as_tensor(lw_rpn).cuda()
        c_star_rpn = torch.as_tensor(c_star_rpn).cuda()
        b_star_rpn = torch.as_tensor(b_star_rpn).cuda()
        o_loss_rpn, b_loss_rpn = self._rpn_loss_func(
            lw_rpn, o_hat_rpn, c_star_rpn, b_hat_rpn, b_star_rpn
        )

        ### compute losses for RCNN
        ###   - inputs are feature maps and RPN output
        ###   - supervision from ground-truth labels and regions from RPN output
        # acquire filtered regions for ROI pooling.
        o0_hat_rpn, o1_hat_rpn = [
            tensor_to_ndarray(confs)
            for confs in o_hat_rpn[:, :, 1]  # confidence for "object" class
        ]  # 2 * (|A|,)
        regions_0, regions_1 = [
            frcnn_box_decode(
                self._anchor_encoder.anchors,  # (|A|, 4)
                tensor_to_ndarray(offsets)  # (|A|, 4)
            )  # (|A|, 4)
            for offsets in b_hat_rpn  # (2, |A|, 4)
        ]  # 2*(|A|, 4)
        regions_0 = self._region_filter(o0_hat_rpn, regions_0)  # (|R0|, 4)
        regions_1 = self._region_filter(o1_hat_rpn, regions_1)  # (|R1|, 4)
        # would prefer to have encoding details abstracted away by a dataset
        # object, but the 2-stage structure complicates this. the main issue
        # is that the (unencoded) ground truth labels are required again once
        # we have obtained the region proposals in order to encode the labels
        # for the rcnn.
        c0_star_rcnn, b0_star_rcnn = self._region_encoder(
            regions_0, inst_0.labels
        )  # (|R0|,), (|R0|, 4)
        c1_star_rcnn, b1_star_rcnn = self._region_encoder(
            regions_1, inst_1.labels
        )  # (|R1|,), (|R1|, 4)
        c_star_rcnn = np.concatenate([c0_star_rcnn, c1_star_rcnn])  # (|R0 u R1|,)
        b_star_rcnn = np.concatenate([b0_star_rcnn, b1_star_rcnn])  # (|R0 u R1|, 4)
        # RCNN predictions.
        c5_0, c5_1 = fmaps.c5  # 2*(C', H', W')
        regions_0 = torch.as_tensor(regions_0).cuda()  # (|R0|, 4)
        regions_1 = torch.as_tensor(regions_1).cuda()  # (|R1|, 4)
        c0_hat_rcnn, b0_hat_rcnn = self.model.rcnn(c5_0, regions_0)  # (|R0|, ...)
        c1_hat_rcnn, b1_hat_rcnn = self.model.rcnn(c5_1, regions_1)  # (|R1|, ...)
        c_hat_rcnn = torch.cat([c0_hat_rcnn, c1_hat_rcnn])  # (|R0 u R1|, n_classes)
        b_hat_rcnn = torch.cat([b0_hat_rcnn, b1_hat_rcnn])  # (|R0 u R1|, 4)
        # RCNN loss.
        c_star_rcnn = torch.as_tensor(c_star_rcnn).cuda()  # (|R0 u R1|,)
        b_star_rcnn = torch.as_tensor(b_star_rcnn).cuda()  # (|R0 u R1|, 4)
        c_loss_rcnn, b_loss_rcnn = self._rcnn_loss_func(
            c_hat_rcnn, c_star_rcnn, b_hat_rcnn, b_star_rcnn
        )

        ### compute losses for correlation trackers
        ###   - inputs are feature maps from each time step
        ###   - supervision from ground-truth labels from each time step
        # CT label encoding.
        track_rois, t_star = track_encode(inst_0.labels, inst_1.labels)  # 2 * (|R0 n R1|, 4)
        # CT predictions.
        # start by unzipping features from each time step
        c3_0, c3_1 = fmaps.c3  # 2 * (C, H, W)
        c4_0, c4_1 = fmaps.c4  # 2 * (C, H', W')
        c5_0, c5_1 = fmaps.c5  # 2 * (C, H', W')
        fm_pyr0 = ResNetFeatures(c3=c3_0, c4=c4_0, c5=c5_0)
        fm_pyr1 = ResNetFeatures(c3=c3_1, c4=c4_1, c5=c5_1)
        fm_reg0, fm_reg1 = fm_reg  # 2 * (Cr, Hr, Wr) RPN feature maps
        track_rois = torch.as_tensor(track_rois).cuda()  # (|R0 n R1|, 4)
        t_hat = self.model.c_tracker(
            fm_pyr0, fm_pyr1, fm_reg0, fm_reg1, track_rois
        )  # (|R0 n R1|, 4)
        # CT loss.
        t_loss = self._track_loss_func(t_hat, t_star)

        return o_loss_rpn, b_loss_rpn, c_loss_rcnn, b_loss_rcnn, t_loss

    def train_on_subset(self, subset: Dataset) -> float:
        """train on subset and return average loss."""
        raise NotImplementedError

    @torch.no_grad()
    def validate(self) -> float:
        """run on validation set and return average loss"""
        raise NotImplementedError

    def train_epoch(self):
        """do one full pass through the entire dataset"""
        raise NotImplementedError
