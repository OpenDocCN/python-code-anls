# `.\models\efficientnet\modeling_efficientnet.py`

```
# coding=utf-8
# Copyright 2023 Google Research, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch EfficientNet model."""


import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_efficientnet import EfficientNetConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "EfficientNetConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/efficientnet-b7"
_EXPECTED_OUTPUT_SHAPE = [1, 768, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/efficientnet-b7"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

EFFICIENTNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/efficientnet-b7",
    # See all EfficientNet models at https://huggingface.co/models?filter=efficientnet
]


EFFICIENTNET_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`EfficientNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

EFFICIENTNET_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`~transformers.EfficientNetTokenizer`.
            See :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
            for more details.

        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, optional):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not** masked,
            - 0 for tokens that are **masked**.

        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, optional):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, optional):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.

        output_attentions (:obj:`bool`, optional):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
            returned tensors for more detail.

        output_hidden_states (:obj:`bool`, optional):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
            for more detail.

        return_dict (:obj:`bool`, optional):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            When set to ``True``, the output will be a :class:`~transformers.file_utils.ModelOutput` object.

    Returns:
        :class:`~transformers.file_utils.ModelOutput` or tuple:
        Example of output for a model with 12 hidden layers and a vocabulary size of 30522.

        Args:
            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
            hidden_states (:obj:`tuple(torch.FloatTensor)`, optional, returned when ``output_hidden_states=True`` is passed or when ``return_dict=True`` is passed):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, optional, returned when ``output_attentions=True`` is passed or when ``return_dict=True`` is passed):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

        Examples::

            from transformers import EfficientNetTokenizer, EfficientNetModel
            import torch

            tokenizer = EfficientNetTokenizer.from_pretrained('efficientnet')
            model = EfficientNetModel.from_pretrained('efficientnet')

            inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            outputs = model(**inputs)
"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 输入的像素值张量，形状为 `(batch_size, num_channels, height, width)`
            # 像素值可以使用 `AutoImageProcessor` 获得。详见 [`AutoImageProcessor.__call__`]。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。查看返回的张量中的 `hidden_states` 以获取更多细节。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
# 定义一个函数，根据 EfficientNet 的配置和通道数，调整滤波器的数量
def round_filters(config: EfficientNetConfig, num_channels: int):
    # 获取深度除数
    divisor = config.depth_divisor
    # 根据宽度系数调整通道数
    num_channels *= config.width_coefficient
    # 计算新的维度，确保是 divisor 的倍数且接近最接近的整数
    new_dim = max(divisor, int(num_channels + divisor / 2) // divisor * divisor)

    # 确保下取整不会低于原始通道数的 90%
    if new_dim < 0.9 * num_channels:
        new_dim += divisor

    return int(new_dim)


# 定义一个函数，用于计算深度可分离卷积的填充值的实用工具函数
def correct_pad(kernel_size: Union[int, Tuple], adjust: bool = True):
    # 如果 kernel_size 是整数，则转换成元组
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    # 计算正确的填充值
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    if adjust:
        return (correct[1] - 1, correct[1], correct[0] - 1, correct[0])
    else:
        return (correct[1], correct[1], correct[0], correct[0])


class EfficientNetEmbeddings(nn.Module):
    r"""
    EfficientNet 的嵌入模块，对应原始工作中的 stem 模块。
    """

    def __init__(self, config: EfficientNetConfig):
        super().__init__()

        # 计算输出维度
        self.out_dim = round_filters(config, 32)
        # 添加零填充层
        self.padding = nn.ZeroPad2d(padding=(0, 1, 0, 1))
        # 定义卷积层
        self.convolution = nn.Conv2d(
            config.num_channels, self.out_dim, kernel_size=3, stride=2, padding="valid", bias=False
        )
        # 批归一化层
        self.batchnorm = nn.BatchNorm2d(self.out_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)
        # 激活函数
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 对输入进行填充
        features = self.padding(pixel_values)
        # 进行卷积操作
        features = self.convolution(features)
        # 执行批归一化
        features = self.batchnorm(features)
        # 应用激活函数
        features = self.activation(features)

        return features


class EfficientNetDepthwiseConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        depth_multiplier=1,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
    ):
        # 计算输出通道数
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # 设置分组卷积数为输入通道数
            bias=bias,
            padding_mode=padding_mode,
        )


class EfficientNetExpansionLayer(nn.Module):
    r"""
    这个类对应原始实现中每个块的扩展阶段。
    """
    # 初始化函数，用于创建一个扩展层对象
    def __init__(self, config: EfficientNetConfig, in_dim: int, out_dim: int, stride: int):
        super().__init__()  # 调用父类构造函数

        # 创建1x1卷积层，用于通道数扩展
        self.expand_conv = nn.Conv2d(
            in_channels=in_dim,          # 输入通道数
            out_channels=out_dim,        # 输出通道数
            kernel_size=1,               # 卷积核大小为1x1
            padding="same",              # 使用与原图大小相同的填充方式
            bias=False,                  # 不使用偏置项
        )
        
        # 创建批归一化层，用于标准化输出
        self.expand_bn = nn.BatchNorm2d(num_features=out_dim, eps=config.batch_norm_eps)
        
        # 选择激活函数，根据配置文件中的隐藏层激活函数选择
        self.expand_act = ACT2FN[config.hidden_act]

    # 前向传播函数，实现扩展阶段的处理过程
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 执行扩展卷积操作
        hidden_states = self.expand_conv(hidden_states)
        # 执行批归一化操作
        hidden_states = self.expand_bn(hidden_states)
        # 执行激活函数操作
        hidden_states = self.expand_act(hidden_states)

        # 返回处理后的结果张量
        return hidden_states
# 定义 EfficientNet 模型的深度可分离卷积层
class EfficientNetDepthwiseLayer(nn.Module):
    r"""
    This corresponds to the depthwise convolution phase of each block in the original implementation.
    """

    def __init__(
        self,
        config: EfficientNetConfig,
        in_dim: int,
        stride: int,
        kernel_size: int,
        adjust_padding: bool,
    ):
        super().__init__()
        self.stride = stride
        # 根据步长选择是否使用 valid 或 same 填充方式
        conv_pad = "valid" if self.stride == 2 else "same"
        # 计算正确的填充量
        padding = correct_pad(kernel_size, adjust=adjust_padding)

        # 创建深度可分离卷积的零填充层
        self.depthwise_conv_pad = nn.ZeroPad2d(padding=padding)
        # 创建深度可分离卷积层
        self.depthwise_conv = EfficientNetDepthwiseConv2d(
            in_dim, kernel_size=kernel_size, stride=stride, padding=conv_pad, bias=False
        )
        # 创建批归一化层
        self.depthwise_norm = nn.BatchNorm2d(
            num_features=in_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum
        )
        # 选择激活函数
        self.depthwise_act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 深度可分离卷积
        if self.stride == 2:
            hidden_states = self.depthwise_conv_pad(hidden_states)

        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_norm(hidden_states)
        hidden_states = self.depthwise_act(hidden_states)

        return hidden_states


# 定义 EfficientNet 模型的 Squeeze and Excite 层
class EfficientNetSqueezeExciteLayer(nn.Module):
    r"""
    This corresponds to the Squeeze and Excitement phase of each block in the original implementation.
    """

    def __init__(self, config: EfficientNetConfig, in_dim: int, expand_dim: int, expand: bool = False):
        super().__init__()
        self.dim = expand_dim if expand else in_dim
        # 计算 Squeeze and Excite 的维度
        self.dim_se = max(1, int(in_dim * config.squeeze_expansion_ratio))

        # 创建全局平均池化层
        self.squeeze = nn.AdaptiveAvgPool2d(output_size=1)
        # 创建 Squeeze 层的卷积操作
        self.reduce = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim_se,
            kernel_size=1,
            padding="same",
        )
        # 创建 Excite 层的卷积操作
        self.expand = nn.Conv2d(
            in_channels=self.dim_se,
            out_channels=self.dim,
            kernel_size=1,
            padding="same",
        )
        # 选择 Squeeze 层的激活函数
        self.act_reduce = ACT2FN[config.hidden_act]
        # 创建 Excite 层的激活函数
        self.act_expand = nn.Sigmoid()

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        inputs = hidden_states
        hidden_states = self.squeeze(hidden_states)
        hidden_states = self.reduce(hidden_states)
        hidden_states = self.act_reduce(hidden_states)

        hidden_states = self.expand(hidden_states)
        hidden_states = self.act_expand(hidden_states)
        hidden_states = torch.mul(inputs, hidden_states)

        return hidden_states


# 定义 EfficientNet 模型的最终阶段的块
class EfficientNetFinalBlockLayer(nn.Module):
    r"""
    This corresponds to the final phase of each block in the original implementation.
    """
    # 初始化函数，用于构建一个 EfficientNetBlock 对象
    def __init__(
        self, config: EfficientNetConfig, in_dim: int, out_dim: int, stride: int, drop_rate: float, id_skip: bool
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 根据条件确定是否应用 dropout
        self.apply_dropout = stride == 1 and not id_skip
        # 创建 1x1 的卷积层，用于调整输入通道数和输出通道数
        self.project_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            padding="same",  # 注意：此处应为 "same"，可能是个错误，通常 "same" 用于填充不应该在此使用
            bias=False,  # 不使用偏置项
        )
        # 创建批归一化层，对输出特征图进行归一化
        self.project_bn = nn.BatchNorm2d(
            num_features=out_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum
        )
        # 创建 dropout 层，用于在训练过程中随机丢弃部分特征
        self.dropout = nn.Dropout(p=drop_rate)

    # 前向传播函数，定义了 EfficientNetBlock 的前向计算过程
    def forward(self, embeddings: torch.FloatTensor, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 使用 1x1 卷积层对输入的隐藏状态进行通道数的调整
        hidden_states = self.project_conv(hidden_states)
        # 对调整后的隐藏状态进行批归一化处理
        hidden_states = self.project_bn(hidden_states)

        # 如果应用了 dropout，则在隐藏状态上进行 dropout 操作，并将嵌入向量添加到 dropout 后的结果中
        if self.apply_dropout:
            hidden_states = self.dropout(hidden_states)
            hidden_states = hidden_states + embeddings  # 将嵌入向量添加到 dropout 后的结果中

        # 返回处理后的隐藏状态作为最终的输出
        return hidden_states
# 定义 EfficientNet 模型的一个块，对应原始实现中每个块的扩展和深度卷积阶段
class EfficientNetBlock(nn.Module):
    r"""
    This corresponds to the expansion and depthwise convolution phase of each block in the original implementation.

    Args:
        config ([`EfficientNetConfig`]):
            Model configuration class.
        in_dim (`int`):
            Number of input channels.
        out_dim (`int`):
            Number of output channels.
        stride (`int`):
            Stride size to be used in convolution layers.
        expand_ratio (`int`):
            Expand ratio to set the output dimensions for the expansion and squeeze-excite layers.
        kernel_size (`int`):
            Kernel size for the depthwise convolution layer.
        drop_rate (`float`):
            Dropout rate to be used in the final phase of each block.
        id_skip (`bool`):
            Whether to apply dropout and sum the final hidden states with the input embeddings during the final phase
            of each block. Set to `True` for the first block of each stage.
        adjust_padding (`bool`):
            Whether to apply padding to only right and bottom side of the input kernel before the depthwise convolution
            operation, set to `True` for inputs with odd input sizes.
    """

    def __init__(
        self,
        config: EfficientNetConfig,
        in_dim: int,
        out_dim: int,
        stride: int,
        expand_ratio: int,
        kernel_size: int,
        drop_rate: float,
        id_skip: bool,
        adjust_padding: bool,
    ):
        super().__init__()
        # 设置扩展比例
        self.expand_ratio = expand_ratio
        # 检查是否需要进行扩展
        self.expand = True if self.expand_ratio != 1 else False
        # 计算扩展后的输入维度
        expand_in_dim = in_dim * expand_ratio

        # 如果需要扩展，则使用 EfficientNetExpansionLayer 执行扩展
        if self.expand:
            self.expansion = EfficientNetExpansionLayer(
                config=config, in_dim=in_dim, out_dim=expand_in_dim, stride=stride
            )

        # 使用 EfficientNetDepthwiseLayer 执行深度卷积
        self.depthwise_conv = EfficientNetDepthwiseLayer(
            config=config,
            in_dim=expand_in_dim if self.expand else in_dim,
            stride=stride,
            kernel_size=kernel_size,
            adjust_padding=adjust_padding,
        )

        # 使用 EfficientNetSqueezeExciteLayer 执行 Squeeze-Excite 操作
        self.squeeze_excite = EfficientNetSqueezeExciteLayer(
            config=config, in_dim=in_dim, expand_dim=expand_in_dim, expand=self.expand
        )

        # 使用 EfficientNetFinalBlockLayer 执行最终的投影和残差连接
        self.projection = EfficientNetFinalBlockLayer(
            config=config,
            in_dim=expand_in_dim if self.expand else in_dim,
            out_dim=out_dim,
            stride=stride,
            drop_rate=drop_rate,
            id_skip=id_skip,
        )
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        embeddings = hidden_states  # 将输入的隐藏状态保存到变量embeddings中
        # Expansion and depthwise convolution phase
        if self.expand_ratio != 1:
            hidden_states = self.expansion(hidden_states)  # 如果扩展比例不为1，通过self.expansion扩展隐藏状态

        hidden_states = self.depthwise_conv(hidden_states)  # 使用深度可分离卷积处理隐藏状态

        # Squeeze and excite phase
        hidden_states = self.squeeze_excite(hidden_states)  # 使用squeeze-and-excite模块处理隐藏状态
        hidden_states = self.projection(embeddings, hidden_states)  # 使用投影层处理原始输入和处理后的隐藏状态

        return hidden_states  # 返回处理后的隐藏状态作为输出
    r"""
    Forward propogates the embeddings through each EfficientNet block.

    Args:
        config ([`EfficientNetConfig`]):
            Model configuration class.
    """
    
    def __init__(self, config: EfficientNetConfig):
        super().__init__()
        self.config = config
        self.depth_coefficient = config.depth_coefficient

        def round_repeats(repeats):
            # 根据深度系数向上取整，确定块的重复次数
            return int(math.ceil(self.depth_coefficient * repeats))

        num_base_blocks = len(config.in_channels)
        num_blocks = sum(round_repeats(n) for n in config.num_block_repeats)

        curr_block_num = 0
        blocks = []
        for i in range(num_base_blocks):
            in_dim = round_filters(config, config.in_channels[i])
            out_dim = round_filters(config, config.out_channels[i])
            stride = config.strides[i]
            kernel_size = config.kernel_sizes[i]
            expand_ratio = config.expand_ratios[i]

            for j in range(round_repeats(config.num_block_repeats[i])):
                id_skip = True if j == 0 else False
                stride = 1 if j > 0 else stride
                in_dim = out_dim if j > 0 else in_dim
                adjust_padding = False if curr_block_num in config.depthwise_padding else True
                drop_rate = config.drop_connect_rate * curr_block_num / num_blocks

                # 创建 EfficientNetBlock 对象并添加到 blocks 列表中
                block = EfficientNetBlock(
                    config=config,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    stride=stride,
                    kernel_size=kernel_size,
                    expand_ratio=expand_ratio,
                    drop_rate=drop_rate,
                    id_skip=id_skip,
                    adjust_padding=adjust_padding,
                )
                blocks.append(block)
                curr_block_num += 1

        # 将所有块组成的列表转换为 ModuleList，以便能够在 PyTorch 中进行管理
        self.blocks = nn.ModuleList(blocks)

        # 添加顶部的卷积层，1x1 卷积，输出通道数为 round_filters(config, 1280)
        self.top_conv = nn.Conv2d(
            in_channels=out_dim,
            out_channels=round_filters(config, 1280),
            kernel_size=1,
            padding="same",  # 使用相同的填充方式
            bias=False,  # 不使用偏置
        )

        # 添加顶部的 Batch Normalization 层
        self.top_bn = nn.BatchNorm2d(
            num_features=config.hidden_dim,  # 输入特征的数量为 config.hidden_dim
            eps=config.batch_norm_eps,  # BN 层的 epsilon 值
            momentum=config.batch_norm_momentum  # BN 层的动量
        )

        # 添加顶部的激活函数，使用 EfficientNetConfig 中指定的激活函数
        self.top_activation = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        # 前向传播函数，接收隐藏状态作为输入，可选地返回隐藏状态字典或单个张量
        ) -> BaseModelOutputWithNoAttention:
        # 函数签名，指定返回类型为BaseModelOutputWithNoAttention

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        # 如果需要输出所有隐藏状态，则初始化一个元组，包含当前隐藏状态；否则初始化为None

        for block in self.blocks:
            # 遍历模型中的每一个块
            hidden_states = block(hidden_states)
            # 将当前隐藏状态传入块中进行处理

            if output_hidden_states:
                # 如果需要输出所有隐藏状态
                all_hidden_states += (hidden_states,)
                # 将当前处理后的隐藏状态添加到所有隐藏状态元组中

        hidden_states = self.top_conv(hidden_states)
        # 将当前隐藏状态通过顶层卷积层处理

        hidden_states = self.top_bn(hidden_states)
        # 将处理后的隐藏状态通过顶层批归一化层处理

        hidden_states = self.top_activation(hidden_states)
        # 将处理后的隐藏状态通过顶层激活函数处理

        if not return_dict:
            # 如果不需要返回字典形式的输出
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
            # 返回一个元组，包含所有非None的隐藏状态和所有隐藏状态，作为输出

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )
        # 返回一个BaseModelOutputWithNoAttention对象，包含最终的隐藏状态和所有隐藏状态
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    
    # 配置类，用于EfficientNet的配置
    config_class = EfficientNetConfig
    # 基础模型前缀，用于标识EfficientNet模型
    base_model_prefix = "efficientnet"
    # 主输入名称，代表模型的像素值输入
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层或卷积层，使用正态分布初始化权重
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 与TensorFlow版本略有不同，这里使用正态分布而不是截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是LayerNorm层，初始化偏置为零，权重为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@add_start_docstrings(
    "The bare EfficientNet model outputting raw features without any specific head on top.",
    EFFICIENTNET_START_DOCSTRING,
)
class EfficientNetModel(EfficientNetPreTrainedModel):
    def __init__(self, config: EfficientNetConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 保存配置对象
        self.config = config
        # 创建EfficientNet模型的嵌入层和编码器
        self.embeddings = EfficientNetEmbeddings(config)
        self.encoder = EfficientNetEncoder(config)

        # 根据配置选择最终的池化层
        if config.pooling_type == "mean":
            self.pooler = nn.AvgPool2d(config.hidden_dim, ceil_mode=True)
        elif config.pooling_type == "max":
            self.pooler = nn.MaxPool2d(config.hidden_dim, ceil_mode=True)
        else:
            # 抛出错误，要求配置中的池化类型必须是'mean'或'max'
            raise ValueError(f"config.pooling must be one of ['mean', 'max'] got {config.pooling}")

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(EFFICIENTNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        # 设置是否输出隐藏状态，默认为模型配置中的设定
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典形式的输出，默认为模型配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供像素值，抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值通过嵌入层处理，得到嵌入输出
        embedding_output = self.embeddings(pixel_values)

        # 使用编码器处理嵌入输出，根据需要返回隐藏状态或字典形式的输出
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 应用池化操作，从编码器输出的第一个元素中获取最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state)

        # 将池化输出的形状从 (batch_size, 1280, 1, 1) 调整为 (batch_size, 1280)
        pooled_output = pooled_output.reshape(pooled_output.shape[:2])

        # 如果不需要以字典形式返回结果，则返回元组形式的结果
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果需要以特定输出类型返回结果，则创建该类型的对象并返回
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
"""
EfficientNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g.
for ImageNet.
"""
# 继承自预训练模型基类 EfficientNetPreTrainedModel，用于图像分类任务
class EfficientNetForImageClassification(EfficientNetPreTrainedModel):
    
    def __init__(self, config):
        # 调用父类构造函数初始化
        super().__init__(config)
        
        # 从配置中获取标签数目
        self.num_labels = config.num_labels
        self.config = config
        
        # 创建 EfficientNet 模型实例
        self.efficientnet = EfficientNetModel(config)
        
        # 分类器头部
        self.dropout = nn.Dropout(p=config.dropout_rate)  # Dropout 层，用于减少过拟合
        self.classifier = nn.Linear(config.hidden_dim, self.num_labels) if self.num_labels > 0 else nn.Identity()
        # 线性层作为分类器，根据是否有标签数目来决定使用 nn.Linear 还是 nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(EFFICIENTNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 覆盖父类的 forward 方法，添加文档字符串和示例代码文档
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用 return_dict；否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 efficientnet 模型进行推断
        outputs = self.efficientnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 如果 return_dict 为 True，则使用 outputs 的 pooler_output；否则使用 outputs 的第二个元素
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 对 pooled_output 进行 dropout 处理
        pooled_output = self.dropout(pooled_output)

        # 使用分类器计算 logits
        logits = self.classifier(pooled_output)

        # 初始化 loss 为 None
        loss = None

        # 如果 labels 不为 None，则计算损失函数
        if labels is not None:
            # 如果问题类型未定义，则根据条件自动设定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 对单个标签的回归问题应用损失函数
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对多标签的回归问题应用损失函数
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对单标签分类问题应用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对多标签分类问题应用二进制交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果 return_dict 为 False，则返回包含 logits 和额外输出的元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 ImageClassifierOutputWithNoAttention 类的实例
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
```