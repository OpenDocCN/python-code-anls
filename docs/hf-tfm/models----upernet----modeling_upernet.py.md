# `.\models\upernet\modeling_upernet.py`

```py
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch UperNet model. Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation."""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ...modeling_outputs import SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...utils.backbone_utils import load_backbone
from .configuration_upernet import UperNetConfig


UPERNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openmmlab/upernet-convnext-tiny",
    # See all UperNet models at https://huggingface.co/models?filter=upernet
]

# General docstring
_CONFIG_FOR_DOC = "UperNetConfig"


class UperNetConvModule(nn.Module):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int], str] = 0,
        bias: bool = False,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        # Initialize convolutional layer with specified parameters
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        # Batch normalization layer to normalize the output of convolution
        self.batch_norm = nn.BatchNorm2d(out_channels)
        # ReLU activation function to introduce non-linearity
        self.activation = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Perform convolution operation
        output = self.conv(input)
        # Apply batch normalization
        output = self.batch_norm(output)
        # Apply ReLU activation
        output = self.activation(output)

        return output


class UperNetPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        super().__init__()
        # Define layers for pyramid pooling block: adaptive average pooling and convolution module
        self.layers = [
            nn.AdaptiveAvgPool2d(pool_scale),  # Adaptive average pooling with specified scale
            UperNetConvModule(in_channels, channels, kernel_size=1),  # Convolution module
        ]
        # Add each layer to the module
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)
    # 定义神经网络前向传播方法，接受输入张量 input，并返回处理后的张量
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 将输入张量作为初始隐藏状态
        hidden_state = input
        # 遍历神经网络的每一层，并依次对隐藏状态进行处理
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        # 返回处理后的最终隐藏状态
        return hidden_state
class UperNetPyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (`Tuple[int]`):
            Pooling scales used in Pooling Pyramid Module.
        in_channels (`int`):
            Input channels.
        channels (`int`):
            Channels after modules, before conv_seg.
        align_corners (`bool`):
            align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales: Tuple[int, ...], in_channels: int, channels: int, align_corners: bool) -> None:
        super().__init__()
        # 存储传入的参数
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = []
        # 根据给定的每个尺度创建对应的 UperNetPyramidPoolingBlock 并添加到 blocks 列表中
        for i, pool_scale in enumerate(pool_scales):
            block = UperNetPyramidPoolingBlock(pool_scale=pool_scale, in_channels=in_channels, channels=channels)
            self.blocks.append(block)
            self.add_module(str(i), block)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ppm_outs = []
        # 遍历每个 PyramidPoolingBlock 执行前向传播
        for ppm in self.blocks:
            ppm_out = ppm(x)
            # 使用双线性插值上采样到原始大小
            upsampled_ppm_out = nn.functional.interpolate(
                ppm_out, size=x.size()[2:], mode="bilinear", align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class UperNetHead(nn.Module):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).
    """
    # 初始化函数，接受配置对象和输入通道数作为参数
    def __init__(self, config, in_channels):
        # 调用父类的初始化方法
        super().__init__()

        # 保存配置对象和池化尺度
        self.config = config
        self.pool_scales = config.pool_scales  # e.g. (1, 2, 3, 6)
        # 保存输入通道数和隐藏层大小
        self.in_channels = in_channels
        self.channels = config.hidden_size
        # 设置插值参数为False
        self.align_corners = False
        # 创建一个卷积层分类器，输出通道数为config.num_labels，核大小为1
        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

        # PSP模块
        self.psp_modules = UperNetPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],  # 取输入通道数的最后一个值
            self.channels,
            align_corners=self.align_corners,
        )
        # 创建一个UperNetConvModule作为瓶颈层，输入为最后一个输入通道数和池化尺度数乘以隐藏层大小
        self.bottleneck = UperNetConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        
        # FPN模块
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        # 遍历除了最后一层的所有输入通道
        for in_channels in self.in_channels[:-1]:  # 跳过顶层
            # 创建UperNetConvModule作为侧边卷积层，输入通道数到隐藏层大小的转换，核大小为1
            l_conv = UperNetConvModule(in_channels, self.channels, kernel_size=1)
            # 创建UperNetConvModule作为FPN卷积层，输入隐藏层大小到隐藏层大小的转换，核大小为3
            fpn_conv = UperNetConvModule(self.channels, self.channels, kernel_size=3, padding=1)
            # 将侧边卷积层和FPN卷积层添加到对应的列表中
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # 创建一个UperNetConvModule作为FPN瓶颈层，输入为所有输入通道数乘以隐藏层大小
        self.fpn_bottleneck = UperNetConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )

    # 初始化权重的方法
    def init_weights(self):
        # 调用apply方法，应用_init_weights方法初始化权重
        self.apply(self._init_weights)

    # 初始化权重的具体实现方法
    def _init_weights(self, module):
        # 如果是Conv2d类型的模块
        if isinstance(module, nn.Conv2d):
            # 从正态分布中初始化权重，均值为0，标准差为配置对象的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，则将偏置初始化为0
            if module.bias is not None:
                module.bias.data.zero_()

    # PSP模块的前向传播方法
    def psp_forward(self, inputs):
        # 取最后一个输入作为x
        x = inputs[-1]
        # 将x作为初始输出
        psp_outs = [x]
        # 使用PSP模块处理x，将处理后的结果扩展到psp_outs列表中
        psp_outs.extend(self.psp_modules(x))
        # 在通道维度上连接psp_outs列表中的所有张量
        psp_outs = torch.cat(psp_outs, dim=1)
        # 将连接后的结果作为输入，通过瓶颈层进行处理，并返回处理后的输出
        output = self.bottleneck(psp_outs)

        return output
    # 前向传播函数，接收编码器隐藏状态作为输入，并返回一个张量作为输出
    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # 构建侧边连接
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # 将 PSP 模块的输出添加到侧边连接列表中
        laterals.append(self.psp_forward(encoder_hidden_states))

        # 构建自顶向下的路径
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # 将当前层的特征图与上一层的特征图进行相加，并使用双线性插值调整大小
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )

        # 构建输出
        # 对侧边连接中的每一层应用 FPN 卷积层
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # 将 PSP 模块的特征图追加到 FPN 输出列表中
        fpn_outs.append(laterals[-1])

        # 对每一层 FPN 输出进行自顶向下的插值调整大小
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
        
        # 在通道维度上连接所有 FPN 输出
        fpn_outs = torch.cat(fpn_outs, dim=1)
        
        # 使用 FPN 瓶颈层处理连接后的特征图
        output = self.fpn_bottleneck(fpn_outs)
        
        # 使用分类器处理最终的输出特征图
        output = self.classifier(output)

        # 返回处理后的输出张量
        return output
class UperNetFCNHead(nn.Module):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is the implementation of
    [FCNNet](https://arxiv.org/abs/1411.4038>).

    Args:
        config:
            Configuration.
        in_channels (int):
            Number of input channels.
        kernel_size (int):
            The kernel size for convs in the head. Default: 3.
        dilation (int):
            The dilation rate for convs in the head. Default: 1.
    """

    def __init__(
        self, config, in_index: int = 2, kernel_size: int = 3, dilation: Union[int, Tuple[int, int]] = 1
    ) -> None:
        super().__init__()

        self.config = config  # 保存配置信息
        self.in_channels = config.auxiliary_in_channels  # 输入通道数
        self.channels = config.auxiliary_channels  # 通道数
        self.num_convs = config.auxiliary_num_convs  # 卷积层数
        self.concat_input = config.auxiliary_concat_input  # 是否连接输入
        self.in_index = in_index  # 输入索引

        conv_padding = (kernel_size // 2) * dilation  # 计算卷积的填充大小
        convs = []
        convs.append(
            UperNetConvModule(
                self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
            )  # 添加第一个卷积模块
        )
        for i in range(self.num_convs - 1):
            convs.append(
                UperNetConvModule(
                    self.channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                )  # 根据配置添加更多卷积模块
            )
        if self.num_convs == 0:
            self.convs = nn.Identity()  # 如果没有卷积层，使用恒等映射
        else:
            self.convs = nn.Sequential(*convs)  # 将卷积模块序列化
        if self.concat_input:
            self.conv_cat = UperNetConvModule(
                self.in_channels + self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2
            )  # 如果连接输入，则添加一个连接卷积模块

        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)  # 分类器卷积层

    def init_weights(self):
        self.apply(self._init_weights)  # 初始化权重

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)  # 初始化卷积层权重
            if module.bias is not None:
                module.bias.data.zero_()  # 初始化卷积层偏置

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # just take the relevant feature maps
        hidden_states = encoder_hidden_states[self.in_index]  # 根据输入索引选择隐藏状态
        output = self.convs(hidden_states)  # 通过卷积模块处理隐藏状态
        if self.concat_input:
            output = self.conv_cat(torch.cat([hidden_states, output], dim=1))  # 如果连接输入，则进行连接操作
        output = self.classifier(output)  # 分类器处理输出
        return output
    # 定义一个方法用于初始化模型的权重，接受一个模块作为参数
    def _init_weights(self, module):
        # 检查传入的模块是否是 UperNetPreTrainedModel 的实例
        if isinstance(module, UperNetPreTrainedModel):
            # 初始化模块的主干网络的权重
            module.backbone.init_weights()
            # 初始化模块的解码头部的权重
            module.decode_head.init_weights()
            # 如果模块有辅助头部，则初始化辅助头部的权重
            if module.auxiliary_head is not None:
                module.auxiliary_head.init_weights()

    # 定义一个方法用于初始化整个模型的权重
    def init_weights(self):
        """Initialize the weights"""
        # 初始化模型的主干网络的权重
        self.backbone.init_weights()
        # 初始化模型的解码头部的权重
        self.decode_head.init_weights()
        # 如果模型有辅助头部，则初始化辅助头部的权重
        if self.auxiliary_head is not None:
            self.auxiliary_head.init_weights()
# UperNetForSemanticSegmentation 类的文档字符串，描述了 UperNet 框架及其使用的模型参数和配置信息
UPERNET_START_DOCSTRING = r"""
    Parameters:
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
        config ([`UperNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# UPERNET_INPUTS_DOCSTRING 为 UperNetForSemanticSegmentation 类的 forward 方法提供的输入参数文档字符串
UPERNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`SegformerImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers in case the backbone has them. See
            `attentions` under returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers of the backbone. See `hidden_states` under
            returned tensors for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# UperNetForSemanticSegmentation 类的定义，继承自 UperNetPreTrainedModel 类
@add_start_docstrings(
    """UperNet framework leveraging any vision backbone e.g. for ADE20k, CityScapes.""",
    UPERNET_START_DOCSTRING,
)
class UperNetForSemanticSegmentation(UperNetPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 加载指定的视觉骨干网络，并赋值给 self.backbone
        self.backbone = load_backbone(config)

        # 语义分割头部模块，使用 UperNetHead 类进行初始化
        self.decode_head = UperNetHead(config, in_channels=self.backbone.channels)
        # 如果配置中指定使用辅助头部，则初始化 UperNetFCNHead 类作为辅助头部
        self.auxiliary_head = UperNetFCNHead(config) if config.use_auxiliary_head else None

        # 初始化权重并进行最终处理
        self.post_init()

    # forward 方法的文档字符串，描述了 forward 方法的输入参数及其作用
    @add_start_docstrings_to_model_forward(UPERNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
```