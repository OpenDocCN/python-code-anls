# `.\models\regnet\modeling_regnet.py`

```py
# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch RegNet model."""

from typing import Optional

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_regnet import RegNetConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "RegNetConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/regnet-y-040"
_EXPECTED_OUTPUT_SHAPE = [1, 1088, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "facebook/regnet-y-040"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

REGNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/regnet-y-040",
    # See all regnet models at https://huggingface.co/models?filter=regnet
]


class RegNetConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: Optional[str] = "relu",
    ):
        super().__init__()
        # 定义卷积层，设置卷积核大小、步长、填充方式、分组数和是否使用偏置
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        # 定义批归一化层
        self.normalization = nn.BatchNorm2d(out_channels)
        # 根据激活函数名称选择激活函数，或者使用恒等映射
        self.activation = ACT2FN[activation] if activation is not None else nn.Identity()

    def forward(self, hidden_state):
        # 执行卷积操作
        hidden_state = self.convolution(hidden_state)
        # 执行批归一化操作
        hidden_state = self.normalization(hidden_state)
        # 执行激活函数操作
        hidden_state = self.activation(hidden_state)
        return hidden_state


class RegNetEmbeddings(nn.Module):
    """
    RegNet Embedddings (stem) composed of a single aggressive convolution.
    """
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config: RegNetConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个 RegNetConvLayer 实例作为 embedder 属性，配置如下参数：
        # - config.num_channels: 输入通道数
        # - config.embedding_size: 嵌入向量的大小
        # - kernel_size=3: 卷积核大小为 3x3
        # - stride=2: 步长为 2
        # - activation=config.hidden_act: 激活函数由配置对象中的 hidden_act 决定
        self.embedder = RegNetConvLayer(
            config.num_channels, config.embedding_size, kernel_size=3, stride=2, activation=config.hidden_act
        )
        # 将配置对象中的 num_channels 属性赋值给实例的 num_channels 属性
        self.num_channels = config.num_channels

    # 前向传播函数，接受像素值作为输入
    def forward(self, pixel_values):
        # 获取像素值的通道数
        num_channels = pixel_values.shape[1]
        # 如果像素值的通道数与实例属性中的 num_channels 不匹配，抛出 ValueError
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 将像素值传递给 embedder 进行处理，得到隐藏状态 hidden_state
        hidden_state = self.embedder(pixel_values)
        # 返回隐藏状态 hidden_state
        return hidden_state
# 从transformers.models.resnet.modeling_resnet.ResNetShortCut复制并修改为RegNetShortCut
class RegNetShortCut(nn.Module):
    """
    RegNet的shortcut，用于将残差特征投影到正确的大小。如果需要，还用于使用`stride=2`对输入进行下采样。
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        # 使用1x1的卷积层进行投影，并设置步长和无偏置
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        # 添加批归一化层
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, input: Tensor) -> Tensor:
        # 对输入进行1x1卷积操作
        hidden_state = self.convolution(input)
        # 对卷积结果进行批归一化
        hidden_state = self.normalization(hidden_state)
        return hidden_state


class RegNetSELayer(nn.Module):
    """
    压缩与激发层(SE)，在[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)中提出。
    """

    def __init__(self, in_channels: int, reduced_channels: int):
        super().__init__()
        # 自适应平均池化层，将输入大小池化为(1, 1)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        # SE结构，包括两个1x1卷积层，ReLU激活函数和Sigmoid激活函数
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_state):
        # 输入为b c h w，将其池化为b c 1 1
        pooled = self.pooler(hidden_state)
        # 使用SE结构计算注意力权重
        attention = self.attention(pooled)
        # 使用注意力权重加权输入特征
        hidden_state = hidden_state * attention
        return hidden_state


class RegNetXLayer(nn.Module):
    """
    RegNet的层，由三个3x3的卷积组成，与ResNet的瓶颈层相同，但reduction=1。
    """

    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # 确定是否应用shortcut，以及设置groups参数
        should_apply_shortcut = in_channels != out_channels or stride != 1
        groups = max(1, out_channels // config.groups_width)
        # 设置shortcut连接，如果需要则使用RegNetShortCut，否则使用身份映射
        self.shortcut = (
            RegNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        )
        # 设计层的顺序：第一层1x1卷积，第二层3x3卷积（可能使用分组卷积），第三层1x1卷积
        self.layer = nn.Sequential(
            RegNetConvLayer(in_channels, out_channels, kernel_size=1, activation=config.hidden_act),
            RegNetConvLayer(out_channels, out_channels, stride=stride, groups=groups, activation=config.hidden_act),
            RegNetConvLayer(out_channels, out_channels, kernel_size=1, activation=None),
        )
        # 设定激活函数
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        # 保留输入的残差连接
        residual = hidden_state
        # 执行层内的卷积操作
        hidden_state = self.layer(hidden_state)
        # 应用shortcut连接
        residual = self.shortcut(residual)
        # 将残差添加到层的输出中
        hidden_state += residual
        # 应用激活函数
        hidden_state = self.activation(hidden_state)
        return hidden_state


class RegNetYLayer(nn.Module):
    """
    RegNet的Y层：一个带有Squeeze和Excitation的X层。
    """
    # 初始化函数，用于初始化一个网络模块
    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = 1):
        # 调用父类的初始化函数
        super().__init__()
        # 根据输入输出通道数和步长判断是否需要应用快捷连接
        should_apply_shortcut = in_channels != out_channels or stride != 1
        # 计算分组卷积的分组数，确保至少有一个分组
        groups = max(1, out_channels // config.groups_width)
        # 如果需要应用快捷连接，则创建RegNetShortCut对象；否则创建nn.Identity对象
        self.shortcut = (
            RegNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        )
        # 创建一个包含多个子模块的序列模块
        self.layer = nn.Sequential(
            # 第一个卷积层：输入通道数到输出通道数的卷积，卷积核大小为1，激活函数为config中指定的隐藏层激活函数
            RegNetConvLayer(in_channels, out_channels, kernel_size=1, activation=config.hidden_act),
            # 第二个卷积层：输出通道数到输出通道数的卷积，卷积核大小为3（由步长决定），分组卷积数为groups，激活函数为config中指定的隐藏层激活函数
            RegNetConvLayer(out_channels, out_channels, stride=stride, groups=groups, activation=config.hidden_act),
            # Squeeze-and-Excitation(SE)模块：对输出通道数进行SE操作，减少通道数为输入通道数的四分之一
            RegNetSELayer(out_channels, reduced_channels=int(round(in_channels / 4))),
            # 第三个卷积层：输出通道数到输出通道数的卷积，卷积核大小为1，无激活函数
            RegNetConvLayer(out_channels, out_channels, kernel_size=1, activation=None),
        )
        # 激活函数，从配置中选择合适的激活函数
        self.activation = ACT2FN[config.hidden_act]

    # 前向传播函数，用于定义数据从输入到输出的流程
    def forward(self, hidden_state):
        # 保存输入作为残差
        residual = hidden_state
        # 将输入通过序列模块进行前向传播
        hidden_state = self.layer(hidden_state)
        # 使用快捷连接模块对残差进行转换
        residual = self.shortcut(residual)
        # 将前向传播结果与转换后的残差相加
        hidden_state += residual
        # 对相加后的结果应用激活函数
        hidden_state = self.activation(hidden_state)
        # 返回处理后的输出结果
        return hidden_state
class RegNetStage(nn.Module):
    """
    A RegNet stage composed by stacked layers.
    """

    def __init__(
        self,
        config: RegNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        depth: int = 2,
    ):
        super().__init__()

        # 根据配置选择不同类型的层
        layer = RegNetXLayer if config.layer_type == "x" else RegNetYLayer

        # 使用 nn.Sequential 定义层的序列
        self.layers = nn.Sequential(
            # 第一层进行下采样，步幅为2
            layer(
                config,
                in_channels,
                out_channels,
                stride=stride,
            ),
            *[layer(config, out_channels, out_channels) for _ in range(depth - 1)],
        )

    def forward(self, hidden_state):
        # 前向传播函数，依次通过每一层
        hidden_state = self.layers(hidden_state)
        return hidden_state


class RegNetEncoder(nn.Module):
    def __init__(self, config: RegNetConfig):
        super().__init__()
        self.stages = nn.ModuleList([])

        # 根据配置决定是否在第一阶段的第一层进行输入下采样
        self.stages.append(
            RegNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
            )
        )

        # 逐阶段定义 RegNetStage，并连接起来
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            self.stages.append(RegNetStage(config, in_channels, out_channels, depth=depth))

    def forward(
        self, hidden_state: Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> BaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        # 逐阶段通过 RegNetStage 进行前向传播
        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        # 根据 return_dict 返回不同的输出格式
        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)


class RegNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RegNetConfig
    base_model_prefix = "regnet"
    main_input_name = "pixel_values"

    # 从 transformers.models.resnet.modeling_resnet.ResNetPreTrainedModel._init_weights 复制而来的初始化权重函数
    # 定义一个方法 `_init_weights`，用于初始化神经网络模块的权重
    def _init_weights(self, module):
        # 如果传入的模块是 nn.Conv2d 类型，则使用 Kaiming 正态分布初始化权重
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        # 如果传入的模块是 nn.BatchNorm2d 或 nn.GroupNorm 类型，则初始化权重为常数 1，偏置为常数 0
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
REGNET_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
    behavior.

    Parameters:
        config ([`RegNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

REGNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`ConvNextImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare RegNet model outputting raw features without any specific head on top.",
    REGNET_START_DOCSTRING,
)
# Copied from transformers.models.resnet.modeling_resnet.ResNetModel with RESNET->REGNET,ResNet->RegNet
class RegNetModel(RegNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedder = RegNetEmbeddings(config)
        self.encoder = RegNetEncoder(config)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ):
        """
        Perform the forward pass of the RegNet model.

        Args:
            pixel_values (torch.FloatTensor): Pixel values of shape `(batch_size, num_channels, height, width)`.
                These values are obtained using an `AutoImageProcessor`.

            output_hidden_states (bool, optional): Whether or not to return hidden states of all layers.
                Refer to `hidden_states` in the returned tensors for details.

            return_dict (bool, optional): Whether to return a `ModelOutput` instead of a tuple.

        Returns:
            Depending on `return_dict`, either a `ModelOutput` or a tuple of outputs from the model.
        """
        # Forward pass logic goes here
        pass
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        # 函数声明，指定返回类型为BaseModelOutputWithPoolingAndNoAttention

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果输出隐藏状态参数不为空，则使用该参数；否则使用self.config.output_hidden_states

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果返回字典参数不为空，则使用该参数；否则使用self.config.use_return_dict

        embedding_output = self.embedder(pixel_values)
        # 将像素值传入嵌入器(embedder)，获取嵌入输出

        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        # 使用编码器(encoder)处理嵌入输出，可以选择输出隐藏状态和是否返回字典

        last_hidden_state = encoder_outputs[0]
        # 获取编码器输出的最后隐藏状态

        pooled_output = self.pooler(last_hidden_state)
        # 使用池化器(pooler)对最后隐藏状态进行池化操作，得到池化输出

        if not return_dict:
            # 如果不返回字典
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
            # 返回最后隐藏状态、池化输出，以及编码器输出的其余部分

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
        # 如果返回字典，则使用BaseModelOutputWithPoolingAndNoAttention类创建并返回一个对象，包括最后隐藏状态、池化输出和所有隐藏状态
@add_start_docstrings(
    """
    RegNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    REGNET_START_DOCSTRING,
)
# 定义 RegNetForImageClassification 类，继承自 RegNetPreTrainedModel 类
class RegNetForImageClassification(RegNetPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置分类标签数量
        self.num_labels = config.num_labels
        # 初始化 RegNetModel，并赋值给 self.regnet
        self.regnet = RegNetModel(config)
        # 定义分类器，使用 nn.Sequential 定义层序列
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 将输入展平
            # 如果配置中有标签数量大于零，则添加全连接层；否则使用恒等映射
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity(),
        )
        # 执行初始化权重和最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 重写 forward 方法，接受像素值、标签等参数，返回模型输出
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 输入参数详细文档字符串已添加

        # 在此处输入参数详细文档字符串已添加
        ):
        # 正文函数方法
    ) -> ImageClassifierOutputWithNoAttention:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用给定的 return_dict；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 regnet 方法进行图像处理，返回输出结果
        outputs = self.regnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 如果 return_dict 为 True，则使用 outputs.pooler_output 作为 pooled_output；否则使用 outputs 的第二个元素
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 使用 classifier 模型对 pooled_output 进行分类得到 logits
        logits = self.classifier(pooled_output)

        # 初始化 loss 为 None
        loss = None

        # 如果 labels 不为 None，则计算损失函数
        if labels is not None:
            # 如果 self.config.problem_type 为 None，则根据条件设置 problem_type
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据 problem_type 计算相应的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 对于单标签回归任务，计算 logits.squeeze() 和 labels.squeeze() 的均方误差损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签回归任务，计算 logits 和 labels 的均方误差损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类任务，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类任务，使用带 logits 的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果 return_dict 为 False，则返回 logits 和 outputs 的其他部分
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        # 返回 ImageClassifierOutputWithNoAttention 对象，包括 loss、logits 和 hidden_states
        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
```