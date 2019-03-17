"""stride-reduced resnet with dilated convolutions"""

import re
from typing import NamedTuple, Optional, Sequence

from torch import nn, Tensor
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls, Bottleneck, ResNet


__all__ = ['resnet', 'ResNetFeatures']


class ResNetFeatures(NamedTuple):
    """resnet intermediate representations corresponding to convolutional
    blocks 3, 4, and 5"""
    c3: Tensor
    c4: Tensor
    c5: Tensor


class _DilatedBottleneck(Bottleneck):
    """Bottleneck with dilated convolutions.

    Args:
        inplanes: see superclass.
        planes: see superclass.
        stride: see superclass.
        downsample: see superclass.
    """
    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None
    ):
        super().__init__(inplanes, planes, stride, downsample)
        self.conv2 = nn.Conv2d(  # overwrite
            planes, planes,
            kernel_size=3, stride=stride, dilation=2, padding=2, bias=False
        )


class _SRResNetBase(ResNet):
    """stride-reduced ResNet base with final convolutional layers replaced
    with dilated convolutions"""
    inplanes: int = 64

    def __init__(
            self,
            layers: Sequence[int],
            zero_init_residual: bool = False,
            first_trainable_layer: int = 4
    ):
        nn.Module.__init__(self)
        self.first_trainable_layer = first_trainable_layer

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(_DilatedBottleneck, 512, layers[3], stride=1)
        self.layer5 = nn.Conv2d(2048, 512, kernel_size=3, dilation=6, padding=6)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual
        # block behaves like an identity. This improves the model by 0.2~0.3%
        # according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

        # freeze all batch norm, then unfreeze batch norm in top layers,
        # leaving bottom layers frozen
        self.eval()
        self.train()

    def forward(self, x: Tensor) -> ResNetFeatures:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c0 = self.maxpool(x)

        c1 = self.layer1(c0)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)  # channel reduce

        return ResNetFeatures(c3=c3, c4=c4, c5=c5)

    def _should_freeze_layer(self, layer_name: str) -> bool:
        """given a layer name, return whether or not it should be frozen."""
        if layer_name.startswith(('conv1', 'bn1', 'relu', 'maxpool')):
            layer_num = 0
        else:
            layer_num = int(re.search(r'^layer(\d)', layer_name).group(1))

        should_freeze = layer_num < self.first_trainable_layer

        return should_freeze

    def load_state_dict(self, state_dict: dict) -> None:
        """load state dict then freeze everything that should_freeze_layer
        tells us to freeze."""
        super().load_state_dict(state_dict, strict=False)
        for param_name, param in self.named_parameters():
            if self._should_freeze_layer(param_name):
                param.requires_grad_(False)

    def train(self, mode: bool = True) -> None:
        """batch norm and dropout remain frozen in early layers"""
        for child_name, child in self.named_children():
            if not self._should_freeze_layer(child_name):
                child.train(mode)

    def eval(self) -> None:
        """freeze everthing indiscriminately."""
        super().train(False)


def resnet(depth: int, pretrained: bool = True, **kwargs) -> nn.Module:
    """constructs a resnet<depth> model.

    Args:
        depth: desired resnet depth.
        pretrained: load pretrained weights if true.

    Returns:
        resnet: model.
    """
    model_menu = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }
    model = _SRResNetBase(model_menu[depth], **kwargs)

    if pretrained:
        model.load_state_dict(
            model_zoo.load_url(model_urls[f'resnet{depth}'])
        )

    return model
