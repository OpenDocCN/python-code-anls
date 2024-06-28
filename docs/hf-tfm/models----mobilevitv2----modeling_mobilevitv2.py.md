# `.\models\mobilevitv2\modeling_mobilevitv2.py`

```
# coding=utf-8
# Copyright 2023 Apple Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch MobileViTV2 model."""


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
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_mobilevitv2 import MobileViTV2Config


logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "MobileViTV2Config"

# Base docstring
_CHECKPOINT_FOR_DOC = "apple/mobilevitv2-1.0-imagenet1k-256"
_EXPECTED_OUTPUT_SHAPE = [1, 512, 8, 8]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "apple/mobilevitv2-1.0-imagenet1k-256"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


MOBILEVITV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "apple/mobilevitv2-1.0-imagenet1k-256"
    # See all MobileViTV2 models at https://huggingface.co/models?filter=mobilevitv2
]


# Copied from transformers.models.mobilevit.modeling_mobilevit.make_divisible
def make_divisible(value: int, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    Ensure that all layers have a channel count that is divisible by `divisor`. This function is taken from the
    original TensorFlow repo. It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def clip(value: float, min_val: float = float("-inf"), max_val: float = float("inf")) -> float:
    """
    Clip the input `value` to ensure it falls within the specified range [`min_val`, `max_val`].
    """
    return max(min_val, min(max_val, value))


# Copied from transformers.models.mobilevit.modeling_mobilevit.MobileViTConvLayer with MobileViT->MobileViTV2
class MobileViTV2ConvLayer(nn.Module):
    """
    MobileViTV2 convolutional layer module that extends nn.Module.
    """
    # 初始化函数，用于初始化一个卷积层模块
    def __init__(
        self,
        config: MobileViTV2Config,  # 接收配置对象，指定模型的参数和行为
        in_channels: int,  # 输入通道数
        out_channels: int,  # 输出通道数
        kernel_size: int,  # 卷积核大小
        stride: int = 1,  # 卷积步长，默认为1
        groups: int = 1,  # 分组卷积中的组数，默认为1
        bias: bool = False,  # 是否使用偏置，默认不使用
        dilation: int = 1,  # 空洞卷积的扩张率，默认为1
        use_normalization: bool = True,  # 是否使用归一化，默认使用
        use_activation: Union[bool, str] = True,  # 是否使用激活函数，或指定激活函数类型，默认使用
    ) -> None:
        super().__init__()  # 调用父类的初始化函数

        padding = int((kernel_size - 1) / 2) * dilation  # 计算卷积的填充大小

        # 检查输入通道数是否能被组数整除，否则抛出数值错误
        if in_channels % groups != 0:
            raise ValueError(f"Input channels ({in_channels}) are not divisible by {groups} groups.")
        # 检查输出通道数是否能被组数整除，否则抛出数值错误
        if out_channels % groups != 0:
            raise ValueError(f"Output channels ({out_channels}) are not divisible by {groups} groups.")

        # 创建卷积层对象
        self.convolution = nn.Conv2d(
            in_channels=in_channels,  # 输入通道数
            out_channels=out_channels,  # 输出通道数
            kernel_size=kernel_size,  # 卷积核大小
            stride=stride,  # 卷积步长
            padding=padding,  # 填充大小
            dilation=dilation,  # 空洞卷积的扩张率
            groups=groups,  # 分组卷积的组数
            bias=bias,  # 是否使用偏置
            padding_mode="zeros",  # 填充模式为零填充
        )

        # 根据是否使用归一化，创建归一化层对象或设置为None
        if use_normalization:
            self.normalization = nn.BatchNorm2d(
                num_features=out_channels,  # 归一化的特征数，即输出通道数
                eps=1e-5,  # 用于数值稳定性的小值
                momentum=0.1,  # 动量参数，用于计算移动平均
                affine=True,  # 是否学习仿射参数
                track_running_stats=True,  # 是否跟踪运行时统计信息
            )
        else:
            self.normalization = None  # 不使用归一化

        # 根据是否使用激活函数，选择合适的激活函数对象
        if use_activation:
            if isinstance(use_activation, str):
                self.activation = ACT2FN[use_activation]  # 根据字符串映射到对应的激活函数
            elif isinstance(config.hidden_act, str):
                self.activation = ACT2FN[config.hidden_act]  # 根据配置中的隐藏层激活函数映射到对应的激活函数
            else:
                self.activation = config.hidden_act  # 使用配置中指定的激活函数
        else:
            self.activation = None  # 不使用激活函数

    # 前向传播函数，接收输入特征并输出处理后的特征
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.convolution(features)  # 进行卷积操作
        if self.normalization is not None:
            features = self.normalization(features)  # 如果有归一化层，进行归一化操作
        if self.activation is not None:
            features = self.activation(features)  # 如果有激活函数，应用激活函数
        return features  # 返回处理后的特征
# 从MobileViT模型中复制的MobileViTV2InvertedResidual类，用于MobileViTV2模型
class MobileViTV2InvertedResidual(nn.Module):
    """
    反向残差块（MobileNetv2）：https://arxiv.org/abs/1801.04381
    """

    def __init__(
        self, config: MobileViTV2Config, in_channels: int, out_channels: int, stride: int, dilation: int = 1
    ) -> None:
        super().__init__()
        # 根据配置参数计算扩展后的通道数，确保是8的倍数
        expanded_channels = make_divisible(int(round(in_channels * config.expand_ratio)), 8)

        # 检查步幅是否合法
        if stride not in [1, 2]:
            raise ValueError(f"Invalid stride {stride}.")

        # 判断是否使用残差连接
        self.use_residual = (stride == 1) and (in_channels == out_channels)

        # 第一个1x1扩展卷积层
        self.expand_1x1 = MobileViTV2ConvLayer(
            config, in_channels=in_channels, out_channels=expanded_channels, kernel_size=1
        )

        # 3x3深度可分离卷积层
        self.conv_3x3 = MobileViTV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=expanded_channels,
            kernel_size=3,
            stride=stride,
            groups=expanded_channels,
            dilation=dilation,
        )

        # 第二个1x1卷积层，用于减少通道数
        self.reduce_1x1 = MobileViTV2ConvLayer(
            config,
            in_channels=expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation=False,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features

        # 执行前向传播：扩展卷积、深度可分离卷积、通道减少卷积
        features = self.expand_1x1(features)
        features = self.conv_3x3(features)
        features = self.reduce_1x1(features)

        # 如果使用残差连接，则加上原始特征
        return residual + features if self.use_residual else features


# 从MobileViT模型中复制的MobileViTV2MobileNetLayer类，用于MobileViTV2模型
class MobileViTV2MobileNetLayer(nn.Module):
    def __init__(
        self, config: MobileViTV2Config, in_channels: int, out_channels: int, stride: int = 1, num_stages: int = 1
    ) -> None:
        super().__init__()

        # 创建模型层列表
        self.layer = nn.ModuleList()
        for i in range(num_stages):
            # 创建MobileViTV2InvertedResidual块并添加到层列表中
            layer = MobileViTV2InvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if i == 0 else 1,
            )
            self.layer.append(layer)
            in_channels = out_channels

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 逐层执行前向传播
        for layer_module in self.layer:
            features = layer_module(features)
        return features


class MobileViTV2LinearSelfAttention(nn.Module):
    """
    这一层应用了MobileViTV2论文中描述的线性复杂度的自注意力机制：
    https://arxiv.org/abs/2206.02680

    Args:
        config (`MobileVitv2Config`):
             模型配置对象
        embed_dim (`int`):
            预期输入的通道数，尺寸为(batch_size, input_channels, height, width)
    """
    def __init__(self, config: MobileViTV2Config, embed_dim: int) -> None:
        super().__init__()

        # 初始化查询/键/值投影层
        self.qkv_proj = MobileViTV2ConvLayer(
            config=config,
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=True,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
        )

        # Dropout 用于注意力权重
        self.attn_dropout = nn.Dropout(p=config.attn_dropout)

        # 初始化输出投影层
        self.out_proj = MobileViTV2ConvLayer(
            config=config,
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=True,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
        )

        # 保存嵌入维度
        self.embed_dim = embed_dim

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用查询/键/值投影层处理隐藏状态
        # qkv 的形状从 (batch_size, embed_dim, num_pixels_in_patch, num_patches) 变为 (batch_size, 1+2*embed_dim, num_pixels_in_patch, num_patches)
        qkv = self.qkv_proj(hidden_states)

        # 将 qkv 张量分解为查询、键和值
        # query 的形状为 [batch_size, 1, num_pixels_in_patch, num_patches]
        # key 和 value 的形状为 [batch_size, embed_dim, num_pixels_in_patch, num_patches]
        query, key, value = torch.split(qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1)

        # 在 num_patches 维度上应用 softmax
        context_scores = torch.nn.functional.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)

        # 计算上下文向量
        # context_vector 的形状为 [batch_size, embed_dim, num_pixels_in_patch, 1]
        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # 将上下文向量与值结合起来
        # out 的形状为 [batch_size, embed_dim, num_pixels_in_patch, num_patches]
        out = torch.nn.functional.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out
# 定义一个 MobileViTV2FFN 类，继承自 nn.Module
class MobileViTV2FFN(nn.Module):
    # 初始化方法，接受 MobileViTV2Config 对象、嵌入维度、FFN 潜在维度、以及可选的 FFN dropout 率
    def __init__(
        self,
        config: MobileViTV2Config,
        embed_dim: int,
        ffn_latent_dim: int,
        ffn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # 第一个卷积层，使用 MobileViTV2ConvLayer 初始化
        self.conv1 = MobileViTV2ConvLayer(
            config=config,
            in_channels=embed_dim,
            out_channels=ffn_latent_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            use_normalization=False,
            use_activation=True,
        )
        # 第一个 dropout 层
        self.dropout1 = nn.Dropout(ffn_dropout)

        # 第二个卷积层，使用 MobileViTV2ConvLayer 初始化
        self.conv2 = MobileViTV2ConvLayer(
            config=config,
            in_channels=ffn_latent_dim,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            bias=True,
            use_normalization=False,
            use_activation=False,
        )
        # 第二个 dropout 层
        self.dropout2 = nn.Dropout(ffn_dropout)

    # 前向传播方法，接受输入张量 hidden_states，返回输出张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 第一层卷积操作和激活函数
        hidden_states = self.conv1(hidden_states)
        # 第一个 dropout 操作
        hidden_states = self.dropout1(hidden_states)
        # 第二层卷积操作（无激活函数）
        hidden_states = self.conv2(hidden_states)
        # 第二个 dropout 操作
        hidden_states = self.dropout2(hidden_states)
        # 返回最终输出张量
        return hidden_states


# 定义一个 MobileViTV2TransformerLayer 类，继承自 nn.Module
class MobileViTV2TransformerLayer(nn.Module):
    # 初始化方法，接受 MobileViTV2Config 对象、嵌入维度、FFN 潜在维度、以及可选的 dropout 率
    def __init__(
        self,
        config: MobileViTV2Config,
        embed_dim: int,
        ffn_latent_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # LayerNorm 操作，用于输入前
        self.layernorm_before = nn.GroupNorm(num_groups=1, num_channels=embed_dim, eps=config.layer_norm_eps)
        # 线性自注意力层，使用 MobileViTV2LinearSelfAttention 初始化
        self.attention = MobileViTV2LinearSelfAttention(config, embed_dim)
        # 第一个 dropout 层
        self.dropout1 = nn.Dropout(p=dropout)
        # LayerNorm 操作，用于输入后
        self.layernorm_after = nn.GroupNorm(num_groups=1, num_channels=embed_dim, eps=config.layer_norm_eps)
        # Feed Forward Network (FFN)，使用 MobileViTV2FFN 初始化
        self.ffn = MobileViTV2FFN(config, embed_dim, ffn_latent_dim, config.ffn_dropout)

    # 前向传播方法，接受输入张量 hidden_states，返回输出张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # LayerNorm 操作，输入前
        layernorm_1_out = self.layernorm_before(hidden_states)
        # 自注意力层操作
        attention_output = self.attention(layernorm_1_out)
        # 残差连接和 LayerNorm 操作，输入后
        hidden_states = attention_output + hidden_states

        # LayerNorm 操作，输入后
        layer_output = self.layernorm_after(hidden_states)
        # Feed Forward Network 操作
        layer_output = self.ffn(layer_output)

        # 残差连接
        layer_output = layer_output + hidden_states
        # 返回最终输出张量
        return layer_output


# 定义一个 MobileViTV2Transformer 类，继承自 nn.Module
class MobileViTV2Transformer(nn.Module):
    # 初始化方法，接受 MobileViTV2Config 对象、层数、模型维度
    def __init__(self, config: MobileViTV2Config, n_layers: int, d_model: int) -> None:
        super().__init__()

        # FFN 维度的倍增器
        ffn_multiplier = config.ffn_multiplier

        # 构建 FFN 各层的维度列表，确保维度是 16 的倍数
        ffn_dims = [ffn_multiplier * d_model] * n_layers
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        # 用于存储 Transformer 层的列表
        self.layer = nn.ModuleList()
        # 循环创建并添加 Transformer 层到列表中
        for block_idx in range(n_layers):
            transformer_layer = MobileViTV2TransformerLayer(
                config, embed_dim=d_model, ffn_latent_dim=ffn_dims[block_idx]
            )
            self.layer.append(transformer_layer)
    # 定义一个前向传播函数，接受隐藏状态作为输入并返回处理后的隐藏状态张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 遍历神经网络模型中的每个层模块
        for layer_module in self.layer:
            # 将隐藏状态张量输入当前层模块，得到处理后的输出隐藏状态张量
            hidden_states = layer_module(hidden_states)
        # 返回最终处理后的隐藏状态张量
        return hidden_states
class MobileViTV2Layer(nn.Module):
    """
    MobileViTV2 layer: https://arxiv.org/abs/2206.02680
    """

    def __init__(
        self,
        config: MobileViTV2Config,
        in_channels: int,
        out_channels: int,
        attn_unit_dim: int,
        n_attn_blocks: int = 2,
        dilation: int = 1,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.patch_width = config.patch_size  # 设置实例变量 patch_width 为配置中的 patch_size
        self.patch_height = config.patch_size  # 设置实例变量 patch_height 为配置中的 patch_size

        cnn_out_dim = attn_unit_dim  # 将注意力单元维度赋给 cnn_out_dim 变量

        if stride == 2:
            # 如果步长为 2，则创建下采样层对象
            self.downsampling_layer = MobileViTV2InvertedResidual(
                config,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride if dilation == 1 else 1,
                dilation=dilation // 2 if dilation > 1 else 1,
            )
            in_channels = out_channels  # 更新输入通道数为输出通道数
        else:
            self.downsampling_layer = None  # 如果步长不为 2，则下采样层设为 None

        # 创建局部表示的卷积层
        self.conv_kxk = MobileViTV2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=config.conv_kernel_size,
            groups=in_channels,
        )

        # 创建局部表示的 1x1 卷积层
        self.conv_1x1 = MobileViTV2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
        )

        # 创建全局表示的变换器层
        self.transformer = MobileViTV2Transformer(config, d_model=attn_unit_dim, n_layers=n_attn_blocks)

        # 创建层归一化对象，使用 GroupNorm 形式
        self.layernorm = nn.GroupNorm(num_groups=1, num_channels=attn_unit_dim, eps=config.layer_norm_eps)

        # 创建融合用的投影卷积层
        self.conv_projection = MobileViTV2ConvLayer(
            config,
            in_channels=cnn_out_dim,
            out_channels=in_channels,
            kernel_size=1,
            use_normalization=True,
            use_activation=False,
        )

    def unfolding(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        batch_size, in_channels, img_height, img_width = feature_map.shape
        # 对特征图进行展开，生成图像块，步长为指定的 patch 尺寸
        patches = nn.functional.unfold(
            feature_map,
            kernel_size=(self.patch_height, self.patch_width),
            stride=(self.patch_height, self.patch_width),
        )
        patches = patches.reshape(batch_size, in_channels, self.patch_height * self.patch_width, -1)

        return patches, (img_height, img_width)
    # 定义一个方法用于将 patches 转换回特征图
    def folding(self, patches: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        # 获取 patches 的维度信息
        batch_size, in_dim, patch_size, n_patches = patches.shape
        # 将 patches 重新整形为 [batch_size, in_dim * patch_size, n_patches]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        # 使用 PyTorch 的 fold 函数将 patches 折叠回特征图
        feature_map = nn.functional.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_height, self.patch_width),
            stride=(self.patch_height, self.patch_width),
        )

        return feature_map

    # 定义模型的前向传播方法
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 如果定义了下采样层，则对输入特征进行空间维度缩减
        if self.downsampling_layer:
            features = self.downsampling_layer(features)

        # 进行局部表示学习
        features = self.conv_kxk(features)  # 应用 kxk 卷积
        features = self.conv_1x1(features)  # 应用 1x1 卷积

        # 将特征图转换为 patches 和输出大小信息
        patches, output_size = self.unfolding(features)

        # 学习全局表示
        patches = self.transformer(patches)  # 使用 transformer 对 patches 进行处理
        patches = self.layernorm(patches)    # 对处理后的 patches 进行 layer normalization

        # 将 patches 转换回特征图
        # [batch_size, patch_height, patch_width, input_dim] --> [batch_size, input_dim, patch_height, patch_width]
        features = self.folding(patches, output_size)  # 调用 folding 方法将 patches 折叠为特征图

        features = self.conv_projection(features)  # 应用卷积投影层将特征图投影到最终输出维度
        return features  # 返回最终的特征图作为模型的输出
class MobileViTV2Encoder(nn.Module):
    # MobileViTV2 编码器的定义，继承自 nn.Module
    def __init__(self, config: MobileViTV2Config) -> None:
        super().__init__()
        self.config = config

        # 初始化一个空的模块列表，用于存储各层模块
        self.layer = nn.ModuleList()
        # 梯度检查点默认为 False
        self.gradient_checkpointing = False

        # 根据配置调整输出步幅，适用于 DeepLab 和 PSPNet 这类分割架构修改分类骨干网的步幅
        dilate_layer_4 = dilate_layer_5 = False
        if config.output_stride == 8:
            dilate_layer_4 = True
            dilate_layer_5 = True
        elif config.output_stride == 16:
            dilate_layer_5 = True

        dilation = 1

        # 计算各层的维度，使其可分割
        layer_0_dim = make_divisible(
            clip(value=32 * config.width_multiplier, min_val=16, max_val=64), divisor=8, min_value=16
        )

        layer_1_dim = make_divisible(64 * config.width_multiplier, divisor=16)
        layer_2_dim = make_divisible(128 * config.width_multiplier, divisor=8)
        layer_3_dim = make_divisible(256 * config.width_multiplier, divisor=8)
        layer_4_dim = make_divisible(384 * config.width_multiplier, divisor=8)
        layer_5_dim = make_divisible(512 * config.width_multiplier, divisor=8)

        # 创建 MobileViTV2MobileNetLayer 层，并添加到模块列表
        layer_1 = MobileViTV2MobileNetLayer(
            config,
            in_channels=layer_0_dim,
            out_channels=layer_1_dim,
            stride=1,
            num_stages=1,
        )
        self.layer.append(layer_1)

        # 创建 MobileViTV2MobileNetLayer 层，并添加到模块列表
        layer_2 = MobileViTV2MobileNetLayer(
            config,
            in_channels=layer_1_dim,
            out_channels=layer_2_dim,
            stride=2,
            num_stages=2,
        )
        self.layer.append(layer_2)

        # 创建 MobileViTV2Layer 层，并添加到模块列表
        layer_3 = MobileViTV2Layer(
            config,
            in_channels=layer_2_dim,
            out_channels=layer_3_dim,
            attn_unit_dim=make_divisible(config.base_attn_unit_dims[0] * config.width_multiplier, divisor=8),
            n_attn_blocks=config.n_attn_blocks[0],
        )
        self.layer.append(layer_3)

        # 如果需要扩展 layer_4 的空洞卷积
        if dilate_layer_4:
            dilation *= 2

        # 创建 MobileViTV2Layer 层，并添加到模块列表
        layer_4 = MobileViTV2Layer(
            config,
            in_channels=layer_3_dim,
            out_channels=layer_4_dim,
            attn_unit_dim=make_divisible(config.base_attn_unit_dims[1] * config.width_multiplier, divisor=8),
            n_attn_blocks=config.n_attn_blocks[1],
            dilation=dilation,
        )
        self.layer.append(layer_4)

        # 如果需要扩展 layer_5 的空洞卷积
        if dilate_layer_5:
            dilation *= 2

        # 创建 MobileViTV2Layer 层，并添加到模块列表
        layer_5 = MobileViTV2Layer(
            config,
            in_channels=layer_4_dim,
            out_channels=layer_5_dim,
            attn_unit_dim=make_divisible(config.base_attn_unit_dims[2] * config.width_multiplier, divisor=8),
            n_attn_blocks=config.n_attn_blocks[2],
            dilation=dilation,
        )
        self.layer.append(layer_5)

    # 前向传播函数，接收隐藏状态张量和是否输出隐藏状态的标志，并返回一个字典
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutputWithNoAttention]:
        # 如果不需要输出隐藏状态，初始化空元组；否则置为 None
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个 Transformer 层
        for i, layer_module in enumerate(self.layer):
            # 如果开启了梯度检查点且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数来计算当前层的隐藏状态
                hidden_states = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                )
            else:
                # 否则直接调用当前层获取隐藏状态
                hidden_states = layer_module(hidden_states)

            # 如果需要输出隐藏状态，则将当前层的隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，将隐藏状态和所有隐藏层状态以元组形式返回
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        # 返回字典形式的结果，包括最终的隐藏状态和所有隐藏层状态
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)
# 定义 MobileViTV2PreTrainedModel 类，继承自 PreTrainedModel
class MobileViTV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 MobileViTV2Config
    config_class = MobileViTV2Config
    # 模型的基础名称前缀为 "mobilevitv2"
    base_model_prefix = "mobilevitv2"
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化模型权重的方法
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        # 如果是 Linear 或 Conv2d 模块
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用正态分布初始化权重，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置，初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 LayerNorm 模块
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置为零，权重为全 1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# MOBILEVITV2_START_DOCSTRING 文档字符串
MOBILEVITV2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MobileViTV2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# MOBILEVITV2_INPUTS_DOCSTRING 输入参数文档字符串
MOBILEVITV2_INPUTS_DOCSTRING = r"""
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

# 添加文档字符串注释到 MobileViTV2Model 类
@add_start_docstrings(
    "The bare MobileViTV2 model outputting raw hidden-states without any specific head on top.",
    MOBILEVITV2_START_DOCSTRING,
)
class MobileViTV2Model(MobileViTV2PreTrainedModel):
    pass  # 类主体为空，只继承 MobileViTV2PreTrainedModel
    def __init__(self, config: MobileViTV2Config, expand_output: bool = True):
        super().__init__(config)
        self.config = config  # 存储传入的配置对象
        self.expand_output = expand_output  # 是否扩展输出的标志

        # 计算第一个卷积层的输出维度
        layer_0_dim = make_divisible(
            clip(value=32 * config.width_multiplier, min_val=16, max_val=64), divisor=8, min_value=16
        )

        # 创建第一个卷积层对象
        self.conv_stem = MobileViTV2ConvLayer(
            config,
            in_channels=config.num_channels,
            out_channels=layer_0_dim,
            kernel_size=3,
            stride=2,
            use_normalization=True,
            use_activation=True,
        )
        # 创建 MobileViTV2Encoder 实例
        self.encoder = MobileViTV2Encoder(config)

        # 初始化权重并进行最终处理
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base class PreTrainedModel
        """
        # 遍历需要修剪的层和头部信息
        for layer_index, heads in heads_to_prune.items():
            mobilevitv2_layer = self.encoder.layer[layer_index]
            # 确保层类型为 MobileViTV2Layer
            if isinstance(mobilevitv2_layer, MobileViTV2Layer):
                # 遍历每个 transformer 层的注意力头部，修剪指定的头部
                for transformer_layer in mobilevitv2_layer.transformer.layer:
                    transformer_layer.attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MOBILEVITV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Performs forward pass of the MobileViTV2 model.
        pixel_values: Optional[torch.Tensor], input pixel values of shape (batch_size, num_channels, height, width)
        output_hidden_states: Optional[bool], whether to output hidden states
        return_dict: Optional[bool], whether to return a dictionary as output
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有显式指定 output_hidden_states，则使用模型配置中的默认设置

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果没有显式指定 return_dict，则使用模型配置中的默认设置

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        # 如果未提供 pixel_values 参数，则抛出数值错误异常

        embedding_output = self.conv_stem(pixel_values)
        # 将像素值输入卷积层，得到嵌入输出

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 将嵌入输出传入编码器进行编码，返回编码器的输出

        if self.expand_output:
            last_hidden_state = encoder_outputs[0]
            # 如果指定了扩展输出，取编码器的最后隐藏状态

            # 全局平均池化: (batch_size, channels, height, width) -> (batch_size, channels)
            pooled_output = torch.mean(last_hidden_state, dim=[-2, -1], keepdim=False)
            # 对最后隐藏状态进行全局平均池化，得到池化输出
        else:
            last_hidden_state = encoder_outputs[0]
            pooled_output = None
            # 否则，只取编码器的最后隐藏状态，并且池化输出为空

        if not return_dict:
            output = (last_hidden_state, pooled_output) if pooled_output is not None else (last_hidden_state,)
            # 如果不要求返回字典格式，则返回最后隐藏状态和池化输出（如果有），否则只返回最后隐藏状态
            return output + encoder_outputs[1:]
            # 返回编码器的其他输出

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
        # 如果要求返回字典格式，则创建并返回包含最后隐藏状态、池化输出和所有隐藏状态的 BaseModelOutputWithPoolingAndNoAttention 对象
# 定义 MobileViTV2 图像分类模型，其在 MobileViTV2PreTrainedModel 基础上增加了一个图像分类头部（即在池化特征之上的线性层），例如适用于 ImageNet。
@add_start_docstrings(
    """
    MobileViTV2 model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    MOBILEVITV2_START_DOCSTRING,
)
class MobileViTV2ForImageClassification(MobileViTV2PreTrainedModel):
    
    def __init__(self, config: MobileViTV2Config) -> None:
        super().__init__(config)

        # 从配置中获取类别数目
        self.num_labels = config.num_labels
        # 初始化 MobileViTV2 模型
        self.mobilevitv2 = MobileViTV2Model(config)

        # 计算第五层的输出维度，并确保是8的倍数
        out_channels = make_divisible(512 * config.width_multiplier, divisor=8)  # layer 5 output dimension
        
        # 分类器头部
        self.classifier = (
            nn.Linear(in_features=out_channels, out_features=config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MOBILEVITV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        # 以下省略部分 forward 方法参数注释
    ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss). If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 参数不为 None，则使用参数值；否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 MobileViTV2 模型进行前向传播
        outputs = self.mobilevitv2(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 如果 return_dict 为 True，则使用 pooler_output 作为 pooled_output；否则使用 outputs 的第二个元素作为 pooled_output
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 将 pooled_output 输入分类器（全连接层），得到 logits
        logits = self.classifier(pooled_output)

        # 初始化 loss 为 None
        loss = None
        # 如果 labels 不为 None，则计算损失函数
        if labels is not None:
            # 如果问题类型未定义，则根据 num_labels 和 labels 的数据类型确定问题类型
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
                    # 对于单标签回归问题，计算均方误差损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签回归问题，计算均方误差损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类问题，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类问题，使用带 logits 的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果 return_dict 为 False，则返回 logits 和额外的输出 hidden_states
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 ImageClassifierOutputWithNoAttention 类型的对象
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
# 从 transformers.models.mobilevit.modeling_mobilevit.MobileViTASPPPooling 复制而来，名称更改为 MobileViTV2ASPPPooling
class MobileViTV2ASPPPooling(nn.Module):
    def __init__(self, config: MobileViTV2Config, in_channels: int, out_channels: int) -> None:
        super().__init__()

        # 创建一个全局平均池化层，将输入特征图池化到输出大小为 1x1
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 创建一个 1x1 卷积层，用于通道变换和特征提取
        self.conv_1x1 = MobileViTV2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_normalization=True,
            use_activation="relu",
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 获取输入特征图的空间大小（高度和宽度）
        spatial_size = features.shape[-2:]
        # 对输入特征图进行全局平均池化，将特征图池化到大小为 1x1
        features = self.global_pool(features)
        # 通过 1x1 卷积层处理池化后的特征图，进行通道变换和特征提取
        features = self.conv_1x1(features)
        # 使用双线性插值将特征图的大小插值回原来的空间大小
        features = nn.functional.interpolate(features, size=spatial_size, mode="bilinear", align_corners=False)
        return features


class MobileViTV2ASPP(nn.Module):
    """
    ASPP 模块，由 DeepLab 论文定义：https://arxiv.org/abs/1606.00915, https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTV2Config) -> None:
        super().__init__()

        # 计算编码器输出通道数，并确保可被 8 整除，作为输入通道数
        encoder_out_channels = make_divisible(512 * config.width_multiplier, divisor=8)  # 第 5 层输出维度
        in_channels = encoder_out_channels
        out_channels = config.aspp_out_channels

        # 如果空洞卷积的扩张率不是 3 个值，抛出异常
        if len(config.atrous_rates) != 3:
            raise ValueError("Expected 3 values for atrous_rates")

        # 创建一个包含多个卷积层的模块列表
        self.convs = nn.ModuleList()

        # 创建输入投影层，使用 1x1 卷积进行通道变换和特征提取
        in_projection = MobileViTV2ConvLayer(
            config,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_activation="relu",
        )
        self.convs.append(in_projection)

        # 使用不同的空洞率创建多个卷积层，并加入到模块列表中
        self.convs.extend(
            [
                MobileViTV2ConvLayer(
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

        # 创建 ASPP 池化层，使用全局平均池化和 1x1 卷积
        pool_layer = MobileViTV2ASPPPooling(config, in_channels, out_channels)
        self.convs.append(pool_layer)

        # 创建投影层，将多个卷积层的输出连接起来，并通过 1x1 卷积进行通道变换和特征提取
        self.project = MobileViTV2ConvLayer(
            config, in_channels=5 * out_channels, out_channels=out_channels, kernel_size=1, use_activation="relu"
        )

        # 创建 Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(p=config.aspp_dropout_prob)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 创建一个空列表，用于存储多个卷积层处理后的特征图
        pyramid = []
        # 遍历模块列表中的每个卷积层，对输入特征图进行处理，并将处理后的结果添加到列表中
        for conv in self.convs:
            pyramid.append(conv(features))
        # 将列表中所有处理后的特征图沿着通道维度拼接起来
        pyramid = torch.cat(pyramid, dim=1)

        # 使用投影层处理拼接后的特征图，进行通道变换和特征提取
        pooled_features = self.project(pyramid)
        # 对投影后的特征图进行 Dropout 操作，以减少过拟合
        pooled_features = self.dropout(pooled_features)
        return pooled_features
# 从 transformers.models.mobilevit.modeling_mobilevit.MobileViTDeepLabV3 复制而来，将 MobileViT 改为 MobileViTV2
class MobileViTV2DeepLabV3(nn.Module):
    """
    DeepLabv3 架构：https://arxiv.org/abs/1706.05587
    """

    def __init__(self, config: MobileViTV2Config) -> None:
        super().__init__()
        # 初始化 ASPP 模块，用于多尺度特征处理
        self.aspp = MobileViTV2ASPP(config)

        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout2d(config.classifier_dropout_prob)

        # 分类器模块，将 ASPP 输出转换为最终的语义分割结果
        self.classifier = MobileViTV2ConvLayer(
            config,
            in_channels=config.aspp_out_channels,
            out_channels=config.num_labels,
            kernel_size=1,
            use_normalization=False,
            use_activation=False,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 前向传播函数，处理输入的隐藏状态张量并生成语义分割的特征张量
        features = self.aspp(hidden_states[-1])  # 使用 ASPP 模块处理最后一层隐藏状态
        features = self.dropout(features)  # 对特征张量进行 dropout 处理
        features = self.classifier(features)  # 使用分类器模块生成最终的语义分割特征
        return features


@add_start_docstrings(
    """
    MobileViTV2 模型，顶部带有语义分割头，例如用于 Pascal VOC 数据集。
    """,
    MOBILEVITV2_START_DOCSTRING,
)
class MobileViTV2ForSemanticSegmentation(MobileViTV2PreTrainedModel):
    def __init__(self, config: MobileViTV2Config) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.mobilevitv2 = MobileViTV2Model(config, expand_output=False)  # MobileViTV2 主干模型
        self.segmentation_head = MobileViTV2DeepLabV3(config)  # 深度解析 v3 分割头模型

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MOBILEVITV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[SemanticSegmenterOutput, Tuple[torch.Tensor, ...]]:
        # 前向传播函数，接受像素值、标签等输入，返回语义分割输出
        return self.mobilevitv2(
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilevitv2(
            pixel_values,
            output_hidden_states=True,  # 指定需要中间隐藏状态作为输出
            return_dict=return_dict,  # 指定是否返回字典形式的输出
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        # 如果返回字典形式，则使用 `outputs.hidden_states`，否则使用 `outputs[1]`

        logits = self.segmentation_head(encoder_hidden_states)
        # 使用编码器的隐藏状态生成分割头部的 logits

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                # 将 logits 插值到原始图像大小
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
                loss = loss_fct(upsampled_logits, labels)
                # 计算交叉熵损失，忽略指定索引处的标签

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
                # 如果不返回字典，且需要隐藏状态，则输出 logits 和隐藏状态
            else:
                output = (logits,) + outputs[2:]
                # 如果不返回字典，不需要隐藏状态，则输出 logits 和其他输出

            return ((loss,) + output) if loss is not None else output
            # 如果有损失，则输出损失和其它输出；否则只输出其它输出

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )
        # 返回语义分割器的输出对象，包括损失、logits、隐藏状态和注意力信息（注意力暂未提供）
```