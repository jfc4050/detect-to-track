"""handles joint training of entire system"""

from typing import Tuple, Sequence

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset
from torch.optim import SGD
from tensorboardX import SummaryWriter
from ml_utils.data import get_subset_lengths
from ml_utils.prediction_filtering import PredictionFilterPipeline

from .data import ImageInstance
from .data.encoding import (
    AnchorEncoder, RegionEncoder, frcnn_box_decode, make_input_transform
)


class DetectTrackTrainer:
    """approximate joint training for two stage object detectors.
    ignores detector head loss wrt to region proposal.
    this can be (but is not currently) addressed by substituting the
    ROIPooling layer for a ROIWarping layer.
    see https://arxiv.org/abs/1506.01497.

    Args:
        backbone: extracts feature map from image.
        rpn: region proposal network: feature map -> region proposals
        rcnn: region based convolutional neural network:
            region proposals -> predictions
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
        rpn_loss_func: loss function for RPN.
        rcnn_loss_func: loss function for RCNN.
        track_loss_func: loss function for tracking module.
        loss_coefs: leading coefficient for each element of joint loss.
            gradients are backpropagated from dot(loss_coefs, losses)
        sgd_kwargs: parameters for stochastic gradient descent.
        tboard_writer: tensorboard logger.
    """
    def __init__(
            self,
            backbone: Module,
            rpn: Module,
            rcnn: Module,
            trn_set: Dataset,
            val_set: Dataset,
            split_size: int,
            net_input_hw: int,
            anchor_encoder: AnchorEncoder,
            region_encoder: RegionEncoder,
            region_filter: PredictionFilterPipeline,
            rpn_loss_func: Module,
            rcnn_loss_func: Module,
            track_loss_func: Module,
            loss_coefs: Sequence[float],
            sgd_kwargs: dict,
            tboard_writer: SummaryWriter
    ) -> None:
        if torch.cuda.device_count() > 1:
            backbone = DataParallel(backbone)
            rpn = DataParallel(rpn)
            rcnn = DataParallel(rcnn)
        self.backbone = backbone.cuda()
        self.rpn = rpn.cuda()
        self.rcnn = rcnn.cuda()

        self.trn_set = trn_set
        self.val_set = val_set
        self._subset_lens = get_subset_lengths(len(self.trn_set), split_size)

        self._im_to_x = make_input_transform(net_input_hw)
        self._anchor_encoder = anchor_encoder  # anchorwise label assignment
        self._region_encoder = region_encoder  # regionwise label assignment
        self._region_filter = region_filter

        self._rpn_loss_func = rpn_loss_func
        self._rcnn_loss_func = rcnn_loss_func
        self._track_loss_func = track_loss_func

        self._loss_coefs = loss_coefs
        self._optimizers = tuple([
            SGD(self.backbone.parameters(), **sgd_kwargs),
            SGD(self.rpn.parameters(), **sgd_kwargs),
            SGD(self.rcnn.parameters() **sgd_kwargs)
        ])

        self.tboard_writer = tboard_writer

        self.n_iters = 0

    def _forward_loss(
            self, instance: ImageInstance
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """compute joint loss for a single instance.

        Args:
            instance: image, labels tuple.

        Returns:
            obj_loss_rpn: RPN binary classification loss.
            box_loss_rpn: RPN bounding box regression loss.
            cls_loss_rcnn: RCNN multiclass classification loss.
            box_loss_rcnn: RCNN bounding box regression loss.
        """
        x = self._im_to_x(instance.im)
        x = x.unsqueeze(0).cuda()  # (1, 3, H, W)
        fm = self.backbone(x)  # (1, C', H', W')

        # compute losses for RPN.
        # RPN input is dependent on backbone output.
        lw_rpn, c_star_rpn, b_star_rpn = self._anchor_encoder(instance.labels)

        lw_rpn = lw_rpn.unsqueeze(0).cuda()  # (1, |A|)
        c_star_rpn = c_star_rpn.unsqueeze(0).cuda()  # (1, |A|)
        b_star_rpn = b_star_rpn.unsqueeze(0).cuda()  # (1, |A|, 4)

        o_hat_rpn, b_hat_rpn = self.rpn(fm)  # (1, |A|), (1, |A|, 4)

        obj_loss_rpn, box_loss_rpn = self._rpn_loss_func(
            lw_rpn, o_hat_rpn, c_star_rpn, b_hat_rpn, b_star_rpn
        )  # scalar losses

        # compute losses for RCNN.
        # RCNN input is dependent on backbone and RPN output.
        regions = frcnn_box_decode(
            self._anchor_encoder.anchors,  # (|A|, 4)
            b_hat_rpn.squeeze(0).detach().cpu().numpy()  # (|A|, 4)
        )  # (|A|, 4) anchor offsets -> actual coordinates
        regions = self._region_filter(o_hat_rpn, regions)  # (|R|, 4)
        # would prefer to have encoding details abstracted away by a dataset
        # object, but the 2-stage structure complicates this. the main issue
        # is that the (unencoded) ground truth labels are required again once
        # we have obtained the region proposals in order to encode the labels
        # for the rcnn.
        c_star_rcnn, b_star_rcnn = self._region_encoder(
            regions, instance.labels
        )  # (|R|,), (|R|, 4)

        c_star_rcnn = c_star_rcnn.cuda()  # (|R|,)
        b_star_rcnn = b_star_rcnn.cuda()  # (|R|, 4)

        c_hat_rcnn, b_hat_rcnn = self.rcnn(fm, regions)  # (|R|, n_classes+1), (|R|, 4)

        cls_loss_rcnn, box_loss_rcnn = self._rcnn_loss_func(
            c_hat_rcnn, c_star_rcnn, b_hat_rcnn, b_star_rcnn
        )  # scalar losses

        return obj_loss_rpn, box_loss_rpn, cls_loss_rcnn, box_loss_rcnn

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
