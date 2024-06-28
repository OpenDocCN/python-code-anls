# `.\models\mobilevit\modeling_mobilevit.py`

```
# coding=utf-8
# Copyright 2022 Apple Inc. and The HuggingFace Inc. team. All rights reserved.
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
#
# Original license: https://github.com/apple/ml-cvnets/blob/main/LICENSE
""" PyTorch MobileViT model."""

# 导入数学库
import math
# 导入类型提示
from typing import Dict, Optional, Set, Tuple, Union

# 导入 PyTorch 库
import torch
# 导入 PyTorch 中的模型定义相关模块
import torch.utils.checkpoint
from torch import nn
# 导入损失函数
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入激活函数映射
from ...activations import ACT2FN
# 导入模型输出定义
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
    SemanticSegmenterOutput,
)
# 导入模型工具函数
from ...modeling_utils import PreTrainedModel
# 导入 PyTorch 工具函数
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
# 导入通用工具函数
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# 获取日志记录器
logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "MobileViTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "apple/mobilevit-small"
_EXPECTED_OUTPUT_SHAPE = [1, 640, 8, 8]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "apple/mobilevit-small"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


# MobileViT 预训练模型列表
MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "apple/mobilevit-small",
    "apple/mobilevit-x-small",
    "apple/mobilevit-xx-small",
    "apple/deeplabv3-mobilevit-small",
    "apple/deeplabv3-mobilevit-x-small",
    "apple/deeplabv3-mobilevit-xx-small",
    # See all MobileViT models at https://huggingface.co/models?filter=mobilevit
]


def make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    Ensure that all layers have a channel count that is divisible by `divisor`. This function is taken from the
    original TensorFlow repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    # 如果未指定最小值，则设为除数
    if min_value is None:
        min_value = divisor
    # 计算新的值，确保可被除数整除
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # 确保向下舍入不会低于原来的值的 90%
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class MobileViTConvLayer(nn.Module):
    # 初始化函数，用于设置卷积层、标准化层和激活函数
    def __init__(
        self,
        config: MobileViTConfig,            # MobileViT 模型的配置对象
        in_channels: int,                   # 输入特征的通道数
        out_channels: int,                  # 输出特征的通道数
        kernel_size: int,                   # 卷积核大小
        stride: int = 1,                    # 卷积步长，默认为 1
        groups: int = 1,                    # 分组卷积中的组数，默认为 1
        bias: bool = False,                 # 是否包含偏置项，默认为 False
        dilation: int = 1,                  # 卷积核元素之间的间隔，默认为 1
        use_normalization: bool = True,     # 是否使用标准化层，默认为 True
        use_activation: Union[bool, str] = True,  # 是否使用激活函数，可以是布尔值或激活函数名称
    ) -> None:
        super().__init__()  # 调用父类的初始化函数

        padding = int((kernel_size - 1) / 2) * dilation  # 计算填充大小

        if in_channels % groups != 0:
            raise ValueError(f"Input channels ({in_channels}) are not divisible by {groups} groups.")
        if out_channels % groups != 0:
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

        # 创建卷积层对象
        self.convolution = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
        )

        # 根据 use_normalization 参数决定是否创建标准化层对象
        if use_normalization:
            self.normalization = nn.BatchNorm2d(
                num_features=out_channels,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            )
        else:
            self.normalization = None

        # 根据 use_activation 参数决定是否创建激活函数对象
        if use_activation:
            if isinstance(use_activation, str):
                self.activation = ACT2FN[use_activation]  # 根据配置的名称选择激活函数
            elif isinstance(config.hidden_act, str):
                self.activation = ACT2FN[config.hidden_act]  # 根据配置中的隐藏层激活函数选择
            else:
                self.activation = config.hidden_act  # 使用默认的激活函数配置
        else:
            self.activation = None  # 不使用激活函数

    # 前向传播函数，接受输入特征并进行卷积、标准化和激活操作，返回处理后的特征
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.convolution(features)  # 卷积操作
        if self.normalization is not None:
            features = self.normalization(features)  # 标准化操作（若标准化层存在）
        if self.activation is not None:
            features = self.activation(features)  # 激活操作（若激活函数存在）
        return features  # 返回处理后的特征张量
# 定义 MobileViTInvertedResidual 类，实现 MobileNetv2 中的反向残差块
class MobileViTInvertedResidual(nn.Module):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """

    def __init__(
        self, config: MobileViTConfig, in_channels: int, out_channels: int, stride: int, dilation: int = 1
    ) -> None:
        super().__init__()
        # 根据配置计算扩展后的通道数，确保为 8 的倍数
        expanded_channels = make_divisible(int(round(in_channels * config.expand_ratio)), 8)

        # 检查步幅是否为合法值
        if stride not in [1, 2]:
            raise ValueError(f"Invalid stride {stride}.")

        # 决定是否使用残差连接，条件为步幅为 1 且输入输出通道数相同
        self.use_residual = (stride == 1) and (in_channels == out_channels)

        # 1x1 卷积扩展层
        self.expand_1x1 = MobileViTConvLayer(
            config, in_channels=in_channels, out_channels=expanded_channels, kernel_size=1
        )

        # 3x3 卷积层
        self.conv_3x3 = MobileViTConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=stride,
            groups=expanded_channels,  # 使用组卷积，组数等于扩展后的通道数
            dilation=dilation,
        )

        # 1x1 卷积降维层
        self.reduce_1x1 = MobileViTConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,  # 不使用激活函数
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features  # 保留输入特征作为残差连接的一部分

        # 执行前向传播：扩展层、3x3 卷积层、1x1 卷积降维层
        features = self.expand_1x1(features)
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)

        # 如果使用残差连接，则将残差与处理后的特征相加
        return residual + features if self.use_residual else features


# 定义 MobileViTMobileNetLayer 类，用于堆叠多个 MobileViTInvertedResidual 模块
class MobileViTMobileNetLayer(nn.Module):
    def __init__(
        self, config: MobileViTConfig, in_channels: int, out_channels: int, stride: int = 1, num_stages: int = 1
    ) -> None:
        super().__init__()

        self.layer = nn.ModuleList()  # 创建模块列表用于存放堆叠的 MobileViTInvertedResidual 模块
        for i in range(num_stages):
            # 根据给定参数创建 MobileViTInvertedResidual 模块并添加到模块列表中
            layer = MobileViTInvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if i == 0 else 1,  # 只在第一层使用指定的步幅
            )
            self.layer.append(layer)
            in_channels = out_channels  # 更新输入通道数为输出通道数

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 逐层对输入特征执行前向传播
        for layer_module in self.layer:
            features = layer_module(features)
        return features


# 定义 MobileViTSelfAttention 类，未完成的类定义，暂无代码
class MobileViTSelfAttention(nn.Module):
    pass  # 占位符，待完成
    # 初始化函数，用于初始化一个 MobileViTAttention 对象
    def __init__(self, config: MobileViTConfig, hidden_size: int) -> None:
        # 调用父类的初始化方法
        super().__init__()

        # 检查隐藏层大小是否是注意力头数的整数倍，如果不是则抛出数值错误异常
        if hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键、值的线性映射层，并指定是否包含偏置
        self.query = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 定义注意力概率的 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入张量 x 进行维度转换，以便进行注意力计算
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 获取新的张量形状，将最后两个维度替换为注意力头数和每个头的大小
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 调整张量的形状
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，用于计算给定隐藏状态的上下文张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 计算混合查询层，通过查询线性映射器
        mixed_query_layer = self.query(hidden_states)

        # 计算转置后的键和值层，以便进行注意力计算
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算查询和键之间的点积，得到原始注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 将注意力分数除以缩放因子，以提升计算稳定性
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 对注意力分数进行归一化处理，转换为注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力概率应用 dropout 操作，随机屏蔽整个 token
        attention_probs = self.dropout(attention_probs)

        # 计算上下文张量，通过注意力概率加权值层
        context_layer = torch.matmul(attention_probs, value_layer)

        # 调整上下文张量的维度顺序，并确保其连续性
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # 调整上下文张量的形状，将头维度合并到一起
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
# 定义 MobileViTSelfOutput 类，继承自 nn.Module
class MobileViTSelfOutput(nn.Module):
    # 初始化方法，接受 MobileViTConfig 类型的 config 对象和整数 hidden_size
    def __init__(self, config: MobileViTConfig, hidden_size: int) -> None:
        super().__init__()
        # 创建一个线性层，输入和输出大小都为 hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        # 创建一个 Dropout 层，使用配置对象中的隐藏层 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接受输入 hidden_states：torch.Tensor，返回输出 torch.Tensor
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入 hidden_states 经过线性层 dense 处理
        hidden_states = self.dense(hidden_states)
        # 经过 Dropout 层处理
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states


# 定义 MobileViTAttention 类，继承自 nn.Module
class MobileViTAttention(nn.Module):
    # 初始化方法，接受 MobileViTConfig 类型的 config 对象和整数 hidden_size
    def __init__(self, config: MobileViTConfig, hidden_size: int) -> None:
        super().__init__()
        # 创建一个 MobileViTSelfAttention 对象，使用给定的 config 和 hidden_size
        self.attention = MobileViTSelfAttention(config, hidden_size)
        # 创建一个 MobileViTSelfOutput 对象，使用给定的 config 和 hidden_size
        self.output = MobileViTSelfOutput(config, hidden_size)
        # 初始化一个空集合，用于存储要修剪的注意力头
        self.pruned_heads = set()

    # 头修剪方法，接受一个整数集合 heads
    def prune_heads(self, heads: Set[int]) -> None:
        # 如果 heads 集合为空，直接返回
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 方法获取要修剪的头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 对注意力层的查询、键、值和输出进行线性层修剪
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪后的头部
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播方法，接受输入 hidden_states：torch.Tensor，返回输出 torch.Tensor
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用 self.attention 对象处理 hidden_states，得到 self_outputs
        self_outputs = self.attention(hidden_states)
        # 使用 self.output 处理 self_outputs，得到 attention_output
        attention_output = self.output(self_outputs)
        # 返回 attention_output
        return attention_output


# 定义 MobileViTIntermediate 类，继承自 nn.Module
class MobileViTIntermediate(nn.Module):
    # 初始化方法，接受 MobileViTConfig 类型的 config 对象，整数 hidden_size 和 intermediate_size
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        # 创建一个线性层，输入大小为 hidden_size，输出大小为 intermediate_size
        self.dense = nn.Linear(hidden_size, intermediate_size)
        # 如果 config.hidden_act 是字符串类型，使用 ACT2FN 字典获取对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则，直接使用 config.hidden_act 作为激活函数
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法，接受输入 hidden_states：torch.Tensor，返回输出 torch.Tensor
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入 hidden_states 经过线性层 dense 处理
        hidden_states = self.dense(hidden_states)
        # 经过激活函数 intermediate_act_fn 处理
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states


# 定义 MobileViTOutput 类，继承自 nn.Module
class MobileViTOutput(nn.Module):
    # 初始化方法，接受 MobileViTConfig 类型的 config 对象，整数 hidden_size 和 intermediate_size
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        # 创建一个线性层，输入大小为 intermediate_size，输出大小为 hidden_size
        self.dense = nn.Linear(intermediate_size, hidden_size)
        # 创建一个 Dropout 层，使用配置对象中的隐藏层 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    # 定义前向传播方法，接受隐藏状态和输入张量作为参数，并返回张量作为输出
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态应用丢弃操作
        hidden_states = self.dropout(hidden_states)
        # 将丢弃后的隐藏状态与输入张量相加，实现残差连接
        hidden_states = hidden_states + input_tensor
        # 返回最终的隐藏状态张量作为输出
        return hidden_states
# 定义 MobileViTTransformerLayer 类，继承自 nn.Module
class MobileViTTransformerLayer(nn.Module):
    # 初始化方法，接收 MobileViTConfig 对象、隐藏层大小和中间层大小作为参数
    def __init__(self, config: MobileViTConfig, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        # 创建注意力层对象，使用 MobileViTAttention 类
        self.attention = MobileViTAttention(config, hidden_size)
        # 创建中间层对象，使用 MobileViTIntermediate 类
        self.intermediate = MobileViTIntermediate(config, hidden_size, intermediate_size)
        # 创建输出层对象，使用 MobileViTOutput 类
        self.output = MobileViTOutput(config, hidden_size, intermediate_size)
        # 创建 LayerNorm 层，用于在注意力层之前和之后进行层归一化
        self.layernorm_before = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    # 前向传播方法，接收隐藏状态张量作为输入，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对输入的隐藏状态进行层归一化，并通过注意力层处理得到注意力输出
        attention_output = self.attention(self.layernorm_before(hidden_states))
        # 将注意力输出与输入的隐藏状态进行残差连接
        hidden_states = attention_output + hidden_states

        # 对残差连接后的隐藏状态再进行层归一化
        layer_output = self.layernorm_after(hidden_states)
        # 通过中间层处理得到中间层的输出
        layer_output = self.intermediate(layer_output)
        # 最后通过输出层处理得到最终的层输出
        layer_output = self.output(layer_output, hidden_states)
        return layer_output


# 定义 MobileViTTransformer 类，继承自 nn.Module
class MobileViTTransformer(nn.Module):
    # 初始化方法，接收 MobileViTConfig 对象、隐藏层大小和阶段数量作为参数
    def __init__(self, config: MobileViTConfig, hidden_size: int, num_stages: int) -> None:
        super().__init__()

        # 创建 nn.ModuleList 对象，用于存储多个 Transformer 层
        self.layer = nn.ModuleList()
        # 根据指定的阶段数量循环创建 TransformerLayer 对象并添加到 nn.ModuleList 中
        for _ in range(num_stages):
            transformer_layer = MobileViTTransformerLayer(
                config,
                hidden_size=hidden_size,
                intermediate_size=int(hidden_size * config.mlp_ratio),
            )
            self.layer.append(transformer_layer)

    # 前向传播方法，接收隐藏状态张量作为输入，通过多个 TransformerLayer 处理后返回最终的张量输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 遍历存储在 nn.ModuleList 中的每个 TransformerLayer，并依次对隐藏状态进行处理
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states


# 定义 MobileViTLayer 类，继承自 nn.Module
class MobileViTLayer(nn.Module):
    """
    MobileViT block: https://arxiv.org/abs/2110.02178
    """

    # 初始化方法，接收 MobileViTConfig 对象、输入通道数、输出通道数、步幅、隐藏层大小、阶段数量和扩展率（默认为1）作为参数
    def __init__(
        self,
        config: MobileViTConfig,
        in_channels: int,
        out_channels: int,
        stride: int,
        hidden_size: int,
        num_stages: int,
        dilation: int = 1,
    ) -> None:
        # 调用父类的构造函数，初始化对象
        super().__init__()
        # 设置补丁的宽度和高度为配置文件中定义的补丁大小
        self.patch_width = config.patch_size
        self.patch_height = config.patch_size

        # 如果步长为2，创建一个下采样层对象
        if stride == 2:
            self.downsampling_layer = MobileViTInvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if dilation == 1 else 1,
                dilation=dilation // 2 if dilation > 1 else 1,
            )
            # 更新输入通道数为输出通道数，以便后续层次使用
            in_channels = out_channels
        else:
            # 如果步长不为2，则不创建下采样层
            self.downsampling_layer = None

        # 创建一个卷积层对象，使用 MobileViTConvLayer 类定义
        self.conv_kxk = MobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=config.conv_kernel_size,
        )

        # 创建另一个卷积层对象，用于变换过程中的特征变换
        self.conv_1x1 = MobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
        )

        # 创建一个 MobileViTTransformer 对象，用于进行变换器层处理
        self.transformer = MobileViTTransformer(
            config,
            hidden_size=hidden_size,
            num_stages=num_stages,
        )

        # 创建一个 LayerNorm 层对象，用于归一化处理
        self.layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

        # 创建一个卷积层对象，用于最终的特征映射
        self.conv_projection = MobileViTConvLayer(
            config, in_channels=hidden_size, out_channels=in_channels, kernel_size=1
        )

        # 创建一个融合层的卷积对象，用于特征融合
        self.fusion = MobileViTConvLayer(
            config, in_channels=2 * in_channels, out_channels=in_channels, kernel_size=config.conv_kernel_size
        )
    # 定义一个方法用于将特征张量展开成补丁（patches）形式，并返回补丁张量及相关信息字典
    def unfolding(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # 获取补丁的宽度和高度
        patch_width, patch_height = self.patch_width, self.patch_height
        # 计算补丁的面积
        patch_area = int(patch_width * patch_height)

        # 获取特征张量的批大小、通道数、原始高度和宽度
        batch_size, channels, orig_height, orig_width = features.shape

        # 计算调整后的新高度和宽度，确保能够划分整数个补丁
        new_height = int(math.ceil(orig_height / patch_height) * patch_height)
        new_width = int(math.ceil(orig_width / patch_width) * patch_width)

        interpolate = False
        # 如果新的宽度或高度与原始不同，则进行插值处理
        if new_width != orig_width or new_height != orig_height:
            # 注意：可以进行填充处理，但需要在注意力函数中处理
            features = nn.functional.interpolate(
                features, size=(new_height, new_width), mode="bilinear", align_corners=False
            )
            interpolate = True

        # 计算沿宽度和高度的补丁数
        num_patch_width = new_width // patch_width
        num_patch_height = new_height // patch_height
        num_patches = num_patch_height * num_patch_width

        # 将特征张量重塑为 (batch_size * patch_area, num_patches, channels) 的形状
        patches = features.reshape(
            batch_size * channels * num_patch_height, patch_height, num_patch_width, patch_width
        )
        # 调整张量的维度顺序
        patches = patches.transpose(1, 2)
        patches = patches.reshape(batch_size, channels, num_patches, patch_area)
        patches = patches.transpose(1, 3)
        patches = patches.reshape(batch_size * patch_area, num_patches, -1)

        # 构建包含相关信息的字典
        info_dict = {
            "orig_size": (orig_height, orig_width),
            "batch_size": batch_size,
            "channels": channels,
            "interpolate": interpolate,
            "num_patches": num_patches,
            "num_patches_width": num_patch_width,
            "num_patches_height": num_patch_height,
        }
        # 返回补丁张量和信息字典
        return patches, info_dict
    def folding(self, patches: torch.Tensor, info_dict: Dict) -> torch.Tensor:
        patch_width, patch_height = self.patch_width, self.patch_height
        patch_area = int(patch_width * patch_height)

        batch_size = info_dict["batch_size"]  # 从信息字典中获取批大小
        channels = info_dict["channels"]  # 从信息字典中获取通道数
        num_patches = info_dict["num_patches"]  # 从信息字典中获取补丁数量
        num_patch_height = info_dict["num_patches_height"]  # 从信息字典中获取补丁高度
        num_patch_width = info_dict["num_patches_width"]  # 从信息字典中获取补丁宽度

        # 将张量重塑为(batch_size, channels, orig_height, orig_width)
        # 形状从(batch_size * patch_area, num_patches, channels)转换回来
        features = patches.contiguous().view(batch_size, patch_area, num_patches, -1)
        features = features.transpose(1, 3)
        features = features.reshape(
            batch_size * channels * num_patch_height, num_patch_width, patch_height, patch_width
        )
        features = features.transpose(1, 2)
        features = features.reshape(
            batch_size, channels, num_patch_height * patch_height, num_patch_width * patch_width
        )

        if info_dict["interpolate"]:
            # 如果需要插值，则使用双线性插值将特征映射插值为原始大小
            features = nn.functional.interpolate(
                features, size=info_dict["orig_size"], mode="bilinear", align_corners=False
            )

        return features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 如果需要降低空间维度，则使用下采样层对特征进行降维
        if self.downsampling_layer:
            features = self.downsampling_layer(features)

        residual = features  # 保留残差连接的特征

        # 局部表示
        features = self.conv_kxk(features)  # 使用kxk卷积处理特征
        features = self.conv_1x1(features)  # 使用1x1卷积处理特征

        # 将特征图转换为补丁
        patches, info_dict = self.unfolding(features)

        # 学习全局表示
        patches = self.transformer(patches)  # 使用transformer处理补丁
        patches = self.layernorm(patches)  # 对处理后的补丁进行层归一化

        # 将补丁重新转换为特征图
        features = self.folding(patches, info_dict)

        features = self.conv_projection(features)  # 使用投影卷积处理特征
        features = self.fusion(torch.cat((residual, features), dim=1))  # 使用融合操作融合残差和处理后的特征
        return features
# 定义一个名为 MobileViTEncoder 的神经网络模块，继承自 nn.Module 类
class MobileViTEncoder(nn.Module):
    # 初始化方法，接收一个 MobileViTConfig 类型的参数 config
    def __init__(self, config: MobileViTConfig) -> None:
        super().__init__()
        # 将传入的配置参数保存到当前对象的 config 属性中
        self.config = config

        # 初始化一个空的神经网络层列表
        self.layer = nn.ModuleList()
        # 设定梯度检查点标志为 False
        self.gradient_checkpointing = False

        # 根据配置参数中的 output_stride 值，设置两个布尔变量来控制网络结构
        dilate_layer_4 = dilate_layer_5 = False
        if config.output_stride == 8:
            dilate_layer_4 = True
            dilate_layer_5 = True
        elif config.output_stride == 16:
            dilate_layer_5 = True

        # 初始化 dilation 参数为 1
        dilation = 1

        # 创建第一个 MobileViTMobileNetLayer 层，并添加到 self.layer 列表中
        layer_1 = MobileViTMobileNetLayer(
            config,
            in_channels=config.neck_hidden_sizes[0],
            out_channels=config.neck_hidden_sizes[1],
            stride=1,
            num_stages=1,
        )
        self.layer.append(layer_1)

        # 创建第二个 MobileViTMobileNetLayer 层，并添加到 self.layer 列表中
        layer_2 = MobileViTMobileNetLayer(
            config,
            in_channels=config.neck_hidden_sizes[1],
            out_channels=config.neck_hidden_sizes[2],
            stride=2,
            num_stages=3,
        )
        self.layer.append(layer_2)

        # 创建第三个 MobileViTLayer 层，并添加到 self.layer 列表中
        layer_3 = MobileViTLayer(
            config,
            in_channels=config.neck_hidden_sizes[2],
            out_channels=config.neck_hidden_sizes[3],
            stride=2,
            hidden_size=config.hidden_sizes[0],
            num_stages=2,
        )
        self.layer.append(layer_3)

        # 如果 dilate_layer_4 为 True，则将 dilation 增加为原来的两倍
        if dilate_layer_4:
            dilation *= 2

        # 创建第四个 MobileViTLayer 层，并添加到 self.layer 列表中
        layer_4 = MobileViTLayer(
            config,
            in_channels=config.neck_hidden_sizes[3],
            out_channels=config.neck_hidden_sizes[4],
            stride=2,
            hidden_size=config.hidden_sizes[1],
            num_stages=4,
            dilation=dilation,
        )
        self.layer.append(layer_4)

        # 如果 dilate_layer_5 为 True，则再次将 dilation 增加为原来的两倍
        if dilate_layer_5:
            dilation *= 2

        # 创建第五个 MobileViTLayer 层，并添加到 self.layer 列表中
        layer_5 = MobileViTLayer(
            config,
            in_channels=config.neck_hidden_sizes[4],
            out_channels=config.neck_hidden_sizes[5],
            stride=2,
            hidden_size=config.hidden_sizes[2],
            num_stages=3,
            dilation=dilation,
        )
        self.layer.append(layer_5)

    # 前向传播方法，接收输入的隐藏状态张量 hidden_states 和额外的参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        # 继续接收其他参数（未完全显示）
    # 函数签名，声明函数的返回类型为元组或BaseModelOutputWithNoAttention类型
    ) -> Union[tuple, BaseModelOutputWithNoAttention]:
        # 如果不输出所有隐藏状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个层次的模块
        for i, layer_module in enumerate(self.layer):
            # 如果启用了梯度检查点且处于训练模式下，则使用梯度检查点函数
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer_module.__call__,  # 调用当前层模块的__call__方法
                    hidden_states,  # 当前隐藏状态
                )
            else:
                hidden_states = layer_module(hidden_states)  # 调用当前层模块处理当前隐藏状态

            # 如果需要输出所有隐藏状态，则将当前隐藏状态加入到所有隐藏状态的元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要以字典形式返回结果，则返回一个元组，过滤掉值为None的项
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        # 以BaseModelOutputWithNoAttention类型返回结果，包含最终隐藏状态和所有隐藏状态
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)
class MobileViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 MobileViTConfig 作为配置类
    config_class = MobileViTConfig
    # 模型的基础名称前缀
    base_model_prefix = "mobilevit"
    # 主要输入的名称
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 对于线性层和卷积层，使用正态分布初始化权重，标准差为配置中的初始化范围
            # 这与 TF 版本稍有不同，后者使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果存在偏置，则将其初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对于 LayerNorm 层，初始化偏置为零，初始化权重为全1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


MOBILEVIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MobileViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MOBILEVIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`MobileViTImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MobileViT model outputting raw hidden-states without any specific head on top.",
    MOBILEVIT_START_DOCSTRING,
)
class MobileViTModel(MobileViTPreTrainedModel):
    """
    MobileViTModel extends MobileViTPreTrainedModel to include specific functionalities for the MobileViT model.

    Inherits from:
        `MobileViTPreTrainedModel`: Provides general initialization and weights handling functionalities.

    Docstring from `add_start_docstrings` decorator:
        "The bare MobileViT model outputting raw hidden-states without any specific head on top."
        MOBILEVIT_START_DOCSTRING: Detailed documentation regarding model usage and configuration parameters.
    """
    def __init__(self, config: MobileViTConfig, expand_output: bool = True):
        super().__init__(config)  # 调用父类的初始化方法，传入配置参数
        self.config = config  # 存储模型配置对象
        self.expand_output = expand_output  # 是否扩展输出的标志

        self.conv_stem = MobileViTConvLayer(
            config,
            in_channels=config.num_channels,  # 输入通道数
            out_channels=config.neck_hidden_sizes[0],  # 输出通道数
            kernel_size=3,  # 卷积核大小
            stride=2,  # 步幅
        )

        self.encoder = MobileViTEncoder(config)  # 创建 MobileViT 编码器对象

        if self.expand_output:
            self.conv_1x1_exp = MobileViTConvLayer(
                config,
                in_channels=config.neck_hidden_sizes[5],  # 输入通道数
                out_channels=config.neck_hidden_sizes[6],  # 输出通道数
                kernel_size=1,  # 卷积核大小
            )

        # 执行后续的初始化和权重设置
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base class PreTrainedModel
        """
        for layer_index, heads in heads_to_prune.items():
            mobilevit_layer = self.encoder.layer[layer_index]  # 获取指定层的 MobileViT 层对象
            if isinstance(mobilevit_layer, MobileViTLayer):  # 如果是 MobileViTLayer 类型的层
                for transformer_layer in mobilevit_layer.transformer.layer:
                    transformer_layer.attention.prune_heads(heads)  # 对注意力机制的头部进行修剪操作

    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,  # 输入像素值的张量，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可选

        self,
        pixel_values: Optional[torch.Tensor] = None,  # 输入像素值的张量，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可选
    ) -> Union[tuple, BaseModelOutputWithPoolingAndNoAttention]:
        # 如果没有指定output_hidden_states，则使用self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有指定return_dict，则使用self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果pixel_values为None，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将pixel_values输入卷积层stem，得到嵌入输出
        embedding_output = self.conv_stem(pixel_values)

        # 将embedding_output作为输入，调用编码器encoder
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果设置了expand_output标志位，则对编码器输出进行额外处理
        if self.expand_output:
            # 对编码器输出的第一个元素进行1x1卷积处理
            last_hidden_state = self.conv_1x1_exp(encoder_outputs[0])

            # 全局平均池化：(batch_size, channels, height, width) -> (batch_size, channels)
            pooled_output = torch.mean(last_hidden_state, dim=[-2, -1], keepdim=False)
        else:
            # 否则直接使用编码器的第一个输出作为最终隐藏状态
            last_hidden_state = encoder_outputs[0]
            pooled_output = None

        # 如果return_dict为False，则返回元组形式的输出
        if not return_dict:
            output = (last_hidden_state, pooled_output) if pooled_output is not None else (last_hidden_state,)
            # 返回编码器输出的所有元素，除了第一个（因为它已经在output中）
            return output + encoder_outputs[1:]

        # 如果return_dict为True，则创建BaseModelOutputWithPoolingAndNoAttention对象进行返回
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
# 使用装饰器添加类的文档字符串，描述了此类是在 MobileViTPreTrainedModel 的基础上添加了一个图像分类头部（线性层）的 MobileViT 模型
@add_start_docstrings(
    """
    MobileViT model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    MOBILEVIT_START_DOCSTRING,  # 引用了 MOBILEVIT_START_DOCSTRING 的文档字符串
)
class MobileViTForImageClassification(MobileViTPreTrainedModel):
    def __init__(self, config: MobileViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels  # 设置类别数目
        self.mobilevit = MobileViTModel(config)  # 初始化 MobileViT 模型

        # 分类器头部
        self.dropout = nn.Dropout(config.classifier_dropout_prob, inplace=True)  # Dropout 层，使用指定的 dropout 概率
        self.classifier = (
            nn.Linear(config.neck_hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
            # 如果有类别数目大于 0，则使用线性层作为分类器；否则使用恒等映射
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用装饰器添加模型前向方法的文档字符串，描述了输入输出的格式及模型的样例
    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,  # 指定检查点
        output_type=ImageClassifierOutputWithNoAttention,  # 输出类型
        config_class=_CONFIG_FOR_DOC,  # 配置类
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,  # 预期输出
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,  # 像素值，可选的 PyTorch 张量
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        labels: Optional[torch.Tensor] = None,  # 标签，可选的 PyTorch 张量
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可选的布尔值
        # 方法未完全展示
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 根据需要设置返回字典，若未指定则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 MobileViT 模型，传入像素值并指定是否返回隐藏状态和是否使用返回字典
        outputs = self.mobilevit(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 如果需要返回字典，则从输出中获取池化后的特征向量；否则直接获取第二个输出（即池化后的特征向量）
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 将池化后的特征向量输入分类器，并施加 dropout
        logits = self.classifier(self.dropout(pooled_output))

        # 初始化损失为 None
        loss = None
        # 如果给定了标签
        if labels is not None:
            # 如果问题类型未定义，则根据标签类型进行推断
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不需要返回字典，则输出结果包括 logits 和可能的隐藏状态
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则构造一个 ImageClassifierOutputWithNoAttention 对象
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
class MobileViTASPPPooling(nn.Module):
    def __init__(self, config: MobileViTConfig, in_channels: int, out_channels: int) -> None:
        super().__init__()

        # 使用全局平均池化层，将输入特征图池化成大小为 1x1 的输出
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 1x1 卷积层，用于通道变换和特征维度的调整
        self.conv_1x1 = MobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_normalization=True,
            use_activation="relu",
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        spatial_size = features.shape[-2:]  # 记录输入特征图的空间维度
        features = self.global_pool(features)  # 对输入特征图进行全局平均池化
        features = self.conv_1x1(features)    # 将池化后的特征图通过1x1卷积层处理
        # 使用双线性插值方法将特征图的大小调整为原始空间大小
        features = nn.functional.interpolate(features, size=spatial_size, mode="bilinear", align_corners=False)
        return features


class MobileViTASPP(nn.Module):
    """
    ASPP module defined in DeepLab papers: https://arxiv.org/abs/1606.00915, https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTConfig) -> None:
        super().__init__()

        # 获取输入通道数和输出通道数
        in_channels = config.neck_hidden_sizes[-2]
        out_channels = config.aspp_out_channels

        if len(config.atrous_rates) != 3:
            raise ValueError("Expected 3 values for atrous_rates")

        # 初始化卷积层列表
        self.convs = nn.ModuleList()

        # 第一个卷积层，使用1x1卷积核进行通道变换和特征映射
        in_projection = MobileViTConvLayer(
            config,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation="relu",
        )
        self.convs.append(in_projection)

        # 使用不同的扩张率构建多个卷积层，以捕捉不同尺度上的信息
        self.convs.extend(
            [
                MobileViTConvLayer(
                    config,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    dilation=rate,
                    use_activation="relu",
                )
                for rate in config.atrous_rates
            ]
        )

        # ASPP池化层，用于捕捉全局信息
        pool_layer = MobileViTASPPPooling(config, in_channels, out_channels)
        self.convs.append(pool_layer)

        # 最终的投影层，用于将多个卷积层的输出特征连接并减少特征维度
        self.project = MobileViTConvLayer(
            config, in_channels=5 * out_channels, out_channels=out_channels, kernel_size=1, use_activation="relu"
        )

        # Dropout 层，用于随机断开神经元连接，防止过拟合
        self.dropout = nn.Dropout(p=config.aspp_dropout_prob)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pyramid = []
        for conv in self.convs:
            pyramid.append(conv(features))
        pyramid = torch.cat(pyramid, dim=1)  # 沿通道维度拼接多个卷积层的输出特征

        # 将拼接后的特征图通过投影层进一步处理
        pooled_features = self.project(pyramid)
        pooled_features = self.dropout(pooled_features)  # 对投影层输出进行 Dropout 处理
        return pooled_features


class MobileViTDeepLabV3(nn.Module):
    """
    DeepLabv3 architecture: https://arxiv.org/abs/1706.05587
    """
    # 初始化函数，用于创建一个 MobileViT 模型对象
    def __init__(self, config: MobileViTConfig) -> None:
        # 调用父类构造函数进行初始化
        super().__init__()
        
        # 创建一个 MobileViTASPP 实例，使用给定的配置信息
        self.aspp = MobileViTASPP(config)
        
        # 创建一个二维 Dropout 层，用于在训练时随机丢弃特征图中的一部分数据
        self.dropout = nn.Dropout2d(config.classifier_dropout_prob)
        
        # 创建一个 MobileViTConvLayer 实例作为分类器，用于将特征映射转换为预测标签
        self.classifier = MobileViTConvLayer(
            config,
            in_channels=config.aspp_out_channels,
            out_channels=config.num_labels,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
            bias=True,
        )

    # 前向传播函数，处理输入的隐藏状态并返回预测输出的特征张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态经过 ASPP 模块得到特征表示
        features = self.aspp(hidden_states[-1])
        
        # 对特征表示进行 Dropout 操作，以减少过拟合风险
        features = self.dropout(features)
        
        # 使用分类器模块对处理后的特征表示进行分类预测
        features = self.classifier(features)
        
        # 返回最终的特征表示，用于后续的分类或其他任务
        return features
@add_start_docstrings(
    """
    MobileViT model with a semantic segmentation head on top, e.g. for Pascal VOC.
    """,
    MOBILEVIT_START_DOCSTRING,
)
class MobileViTForSemanticSegmentation(MobileViTPreTrainedModel):
    def __init__(self, config: MobileViTConfig) -> None:
        super().__init__(config)

        # 设置分类数目
        self.num_labels = config.num_labels
        # 创建 MobileViT 模型，关闭扩展输出
        self.mobilevit = MobileViTModel(config, expand_output=False)
        # 创建语义分割头部模型
        self.segmentation_head = MobileViTDeepLabV3(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MOBILEVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilevit(
            pixel_values,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

获取是否输出隐藏状态和返回类型的设定，若未指定则使用模型配置中的默认设定。


        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

根据返回类型决定使用模型输出的隐藏状态或者第二个元素作为编码器的隐藏状态。


        logits = self.segmentation_head(encoder_hidden_states)

使用编码器隐藏状态作为输入，通过分割头部生成预测的logits。


        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                # upsample logits to the images' original size
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)

如果提供了标签，根据标签的形状和配置中的忽略索引，使用交叉熵损失函数计算损失值。


        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

根据返回类型和是否输出隐藏状态，构建输出元组并返回。如果有损失值，则将其作为第一个元素返回。


        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )

以自定义的输出对象形式返回结果，包括损失、logits、隐藏状态（如果需要）和注意力机制（目前为None）。```python
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilevit(
            pixel_values,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

设置是否输出隐藏状态和返回类型的选择，如果未指定则使用模型配置中的默认设置。


        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

根据返回类型决定使用模型输出的隐藏状态或者第二个元素作为编码器的隐藏状态。


        logits = self.segmentation_head(encoder_hidden_states)

使用编码器隐藏状态作为输入，通过分割头部生成预测的logits。


        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                # upsample logits to the images' original size
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)

如果提供了标签，根据标签的形状和配置中的忽略索引，使用交叉熵损失函数计算损失值。


        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

根据返回类型和是否输出隐藏状态，构建输出元组并返回。如果有损失值，则将其作为第一个元素返回。


        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )

以自定义的输出对象形式返回结果，包括损失、logits、隐藏状态（如果需要）和注意力机制（目前为None）。
```