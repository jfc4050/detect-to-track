"""detect to track module"""

from typing import Optional

from torch.nn import Module

from .resnet import resnet
from .rpn import RPN
from .rfcn import RFCN
from .correlation_tracker import CorrelationTracker


class DetectTrackModule(Module):
    """groups individual modules of detect-to-track so that operations
    like .train() or .cuda() can be applied to all of them at once,
    and also so that they can all share the same state_dict

    training forward pass logic is kept in the detect to track trainer
    class, and evaluation forward pass logic is kept in the detect to track
    detector class because:
        1) there is a large difference in required behavior for forward
        passes, especially for multi-step training schemes.
        2) there is also a wide variety of training schemes used for two-stage
        networks (see faster-rcnn paper for a few), each requiring different
        forward pass behavior.
        3) want to make sure all modules follow a consistent Tensor -> Tensor
        interface, and remain unaware of pre/postprocessing steps
        (image conversion to tensors, nms, confidence thresholding,
        encoding/decoding, etc.)

    Attributes:
        backbone: image tensor -> feature map.
        rpn: feature map -> region proposals.
        rcnn: feature map, region proposals -> classifications and offsets.
        c_tracker: feature maps from each time step -> tracks across frames.
    """
    def __init__(
            self,
            backbone: Optional[Module] = None,
            rpn: Optional[Module] = None,
            rcnn: Optional[Module] = None,
            c_tracker: Optional[Module] = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.rcnn = rcnn
        self.c_tracker = c_tracker

    def build_backbone(
            self,
            depth: int,
            zero_init_residual: bool = False,
            first_trainable_stage: int = 3
    ) -> None:
        self.backbone = resnet(
            depth,
            zero_init_residual=zero_init_residual,
            first_trainable_stage=first_trainable_stage,
            pretrained=True
        )

    def build_rpn(self, n_anchors: int) -> None:
        if self.backbone is None:
            raise ValueError('build backbone first')
        self.rpn = RPN(self.backbone.stage4.out_channels, n_anchors)

    def build_rcnn(self, n_classes: int, k: int) -> None:
        if self.backbone is None:
            raise ValueError('build backbone first')
        self.rcnn = RFCN(self.backbone.stage5.out_channels, n_classes, k)

    def build_c_tracker(self, d_max: int, r_hw: int) -> None:
        if self.rpn is None:
            raise ValueError('build rpn first')
        self.c_tracker = CorrelationTracker(
            d_max, r_hw, self.rpn.conv.out_channels
        )

    def forward(self):
        raise AttributeError(
            'this module doesnt know how to perform its own forward passes'
        )
