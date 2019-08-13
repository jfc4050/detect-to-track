"""detect to track module"""

from torch.nn import Module

from .resnet import resnet_backbone
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

    # TODO - compute these from parameters instead of hardcoding.
    stage3_outchannels = 512
    stage4_outchannels = 1024
    stage5_outchannels = 2048

    def __init__(
        self,
        backbone_arch: str,
        first_trainable_stage: int,
        n_anchors: int,
        n_classes: int,
        k: int,
        d_max: int,
        r_hw: int,
    ) -> None:
        super().__init__()
        self.backbone = resnet_backbone(backbone_arch, first_trainable_stage)
        self.rpn = RPN(self.stage4_outchannels, n_anchors)
        self.rcnn = RFCN(self.stage5_outchannels, n_classes, k)
        self.c_tracker = CorrelationTracker(d_max, r_hw, self.rpn.conv.out_channels)

    def forward(self):
        raise NotImplementedError(
            "this module doesnt know how to perform its own forward passes. "
            "forward passes are implemented by trainer/detector classes"
        )
