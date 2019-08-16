"""stride-reduced resnet with dilated convolutions"""

import re

from torch.nn import Module, BatchNorm2d, Sequential
from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from ml_utils.torch.modules import Normalizer


def resnet_backbone(backbone_arch: str, first_trainable_stage: int) -> Module:
    """return resnet backbone."""
    if backbone_arch.startswith("resnext"):
        # TODO - figure out why FrozenBatchNorm2d causes issues
        norm_layer = BatchNorm2d
    else:
        norm_layer = FrozenBatchNorm2d
    backbone = resnet.__dict__[backbone_arch](
        pretrained=True,
        norm_layer=norm_layer,
        replace_stride_with_dilation=(False, False, 2),
    )
    backbone.eval()

    # freeze layers
    for name, parameter in backbone.named_parameters():
        match = re.search(r"layer(\d)", name)
        if not (match and int(match.group(1)) >= first_trainable_stage):
            parameter.requires_grad_(False)

    return_layers = {"layer2": "c3", "layer3": "c4", "layer4": "c5"}

    normalizer = Normalizer()
    backbone = IntermediateLayerGetter(backbone, return_layers)

    backbone = Sequential(normalizer, backbone)

    return backbone
