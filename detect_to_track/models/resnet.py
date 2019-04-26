"""stride-reduced resnet with dilated convolutions"""

import re
from typing import NamedTuple, Optional

from torch import nn, Tensor
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls, Bottleneck, ResNet


__all__ = ['SRResNet', 'resnet', 'ResNetFeatures']


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


class SRResNet(ResNet):
    """stride-reduced ResNet base with final convolutional layers replaced
    with dilated convolutions"""
    def __init__(
            self,
            depth: int,
            zero_init_residual: bool = False,
            first_trainable_stage: int = 3
    ):
        nn.Module.__init__(self)

        layers_2, layers_3, layers_4, layers_5 = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }[depth]

        self.inplanes = 64

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage2 = self._make_layer(Bottleneck, 64, layers_2)
        self.stage3 = self._make_layer(Bottleneck, 128, layers_3, stride=2)
        self.stage4 = self._make_layer(Bottleneck, 256, layers_4, stride=2)
        self.stage5 = self._make_layer(_DilatedBottleneck, 512, layers_5)

        # TODO - compute from parameters
        self.stage3.out_channels = 512
        self.stage4.out_channels = 1024
        self.stage5.out_channels = 2048

        self.first_trainable_stage = first_trainable_stage

        self._init_weights(zero_init_residual)

        # freeze all batch norm, then unfreeze batch norm in top layers,
        # leaving bottom layers frozen
        self.eval()
        self.train()

    def _init_weights(self, zero_init_residual: bool) -> None:
        """initialize weights."""
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

    def _should_freeze_layer(self, layer_name: str) -> bool:
        """given a layer name, return whether or not it should be frozen."""
        stage_num = int(re.search(r'^stage(\d)', layer_name).group(1))
        should_freeze = stage_num < self.first_trainable_stage

        return should_freeze

    def forward(self, x: Tensor) -> ResNetFeatures:
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        return ResNetFeatures(c3=c3, c4=c4, c5=c5)

    def freeze(self, indiscriminate: bool = False) -> None:
        """freeze parameters"""
        for param_name, param in self.named_parameters():
            if self._should_freeze_layer(param_name) or indiscriminate:
                param.requires_grad_(False)

    def train(self, mode: bool = True) -> None:
        """batch norm and dropout remain frozen in early layers"""
        for child_name, child in self.named_children():
            if not self._should_freeze_layer(child_name):
                child.train(mode)

    def eval(self) -> None:
        """freeze all batch norm + dropout indiscriminately."""
        super().train(False)


def resnet(
        depth: int,
        zero_init_residual: bool = True,
        first_trainable_stage: int = 3,
        pretrained: bool = True,
) -> nn.Module:
    """constructs a resnet<depth> model.

    Args:
        depth: desired resnet depth.
        zero_init_residual:
        first_trainable_stage:
        pretrained: load pretrained weights if true.

    Returns:
        resnet: model.
    """
    model = SRResNet(depth, zero_init_residual, first_trainable_stage)

    if pretrained:  # download state dict, rename params, then load.
        raw_state_dict = model_zoo.load_url(model_urls[f'resnet{depth}'])

        state_dict = dict()
        for param_name, param in raw_state_dict.items():
            if param_name.startswith('conv1'):
                param_name = re.sub(r'conv1\.', 'stage1.0.', param_name)
            elif param_name.startswith('bn1'):
                param_name = re.sub(r'bn1\.', 'stage1.1.', param_name)
            elif param_name.startswith('relu'):
                param_name = re.sub(r'relu\.', 'stage1.2.', param_name)
            elif param_name.startswith('maxpool'):
                param_name = re.sub(r'maxpool\.', 'stage1.3', param_name)
            elif param_name.startswith('layer'):
                layer_num = int(re.search(r'^layer(\d)\.', param_name).group(1))
                param_name = re.sub(
                    r'^layer\d\.', f'stage{layer_num+1}.', param_name
                )
            elif param_name.startswith(('avgpool', 'fc')):
                continue
            else:
                raise ValueError(f'unrecognized param name {param_name}')

            state_dict[param_name] = param

        model.load_state_dict(state_dict, strict=True)
        model.freeze()

    return model
