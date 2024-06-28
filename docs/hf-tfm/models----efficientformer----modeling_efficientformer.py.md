# `.\models\efficientformer\modeling_efficientformer.py`

```
# coding=utf-8
# 版权 2022 年 Snapchat 研究团队和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可;
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按“原样”分发，
# 不提供任何明示或暗示的担保或条件。
# 请参阅许可证了解特定语言下的权限和限制。

""" PyTorch EfficientFormer 模型。"""

import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_efficientformer import EfficientFormerConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "EfficientFormerConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "snap-research/efficientformer-l1-300"
_EXPECTED_OUTPUT_SHAPE = [1, 49, 448]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "snap-research/efficientformer-l1-300"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"


EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "snap-research/efficientformer-l1-300",
    # See all EfficientFormer models at https://huggingface.co/models?filter=efficientformer
]


class EfficientFormerPatchEmbeddings(nn.Module):
    """
    此类在两个阶段之间执行下采样。对于形状为 [batch_size, num_channels, height, width] 的输入张量，
    它生成形状为 [batch_size, num_channels, height/stride, width/stride] 的输出张量。
    """

    def __init__(self, config: EfficientFormerConfig, num_channels: int, embed_dim: int, apply_norm: bool = True):
        super().__init__()
        self.num_channels = num_channels

        # 使用 nn.Conv2d 定义投影层，用于下采样操作
        self.projection = nn.Conv2d(
            num_channels,
            embed_dim,
            kernel_size=config.downsample_patch_size,
            stride=config.downsample_stride,
            padding=config.downsample_pad,
        )
        # 根据 apply_norm 参数选择是否添加批标准化层或恒等映射
        self.norm = nn.BatchNorm2d(embed_dim, eps=config.batch_norm_eps) if apply_norm else nn.Identity()
    # 定义前向传播方法，接受像素值张量作为输入，并返回处理后的张量
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 获取输入张量的批处理大小、通道数、高度和宽度
        batch_size, num_channels, height, width = pixel_values.shape
        
        # 检查通道数是否与模型配置中设置的通道数一致，如果不一致则抛出数值错误异常
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        
        # 将输入张量投影到嵌入空间中
        embeddings = self.projection(pixel_values)
        
        # 对投影后的张量进行规范化处理
        embeddings = self.norm(embeddings)

        # 返回处理后的嵌入张量作为前向传播的输出
        return embeddings
class EfficientFormerSelfAttention(nn.Module):
    def __init__(self, dim: int, key_dim: int, num_heads: int, attention_ratio: int, resolution: int):
        super().__init__()

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention_ratio = attention_ratio
        self.scale = key_dim**-0.5
        self.total_key_dim = key_dim * num_heads
        self.expanded_key_dim = int(attention_ratio * key_dim)
        self.total_expanded_key_dim = int(self.expanded_key_dim * num_heads)
        
        # Calculate the hidden size based on key dimensions and attention ratios
        hidden_size = self.total_expanded_key_dim + self.total_key_dim * 2
        
        # Linear transformation for Q, K, V inputs
        self.qkv = nn.Linear(dim, hidden_size)
        
        # Linear projection for output
        self.projection = nn.Linear(self.total_expanded_key_dim, dim)
        
        # Generate all possible pairs of points in the resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        num_points = len(points)
        
        # Create unique offsets and assign indices to them
        attention_offsets = {}
        idxs = []
        for point_1 in points:
            for point_2 in points:
                offset = (abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        
        # Define attention biases as a parameter
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        
        # Register buffer for storing attention bias indices
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(num_points, num_points))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        
        # Delete existing attention biases if training mode is enabled
        if mode and hasattr(self, "ab"):
            del self.ab
        else:
            # Store attention biases sliced by precomputed indices
            self.ab = self.attention_biases[:, self.attention_bias_idxs]
    # 定义前向传播函数，接受隐藏状态张量和是否输出注意力权重的标志，返回元组类型的张量
    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False) -> Tuple[torch.Tensor]:
        # 获取隐藏状态张量的批量大小、序列长度和通道数
        batch_size, sequence_length, num_channels = hidden_states.shape
        
        # 使用 self.qkv 对象处理隐藏状态张量，得到查询、键、值张量
        qkv = self.qkv(hidden_states)
        
        # 将处理后的 qkv 张量重塑并分割成查询层、键层、值层
        query_layer, key_layer, value_layer = qkv.reshape(batch_size, sequence_length, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.expanded_key_dim], dim=3
        )
        
        # 对查询层、键层、值层的维度进行重新排列
        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)

        # 如果不处于训练状态，则将 self.ab 张量移动到与 attention_biases 的设备相同
        if not self.training:
            self.ab = self.ab.to(self.attention_biases.device)
        
        # 计算注意力概率，考虑缩放因子和注意力偏置
        attention_probs = (torch.matmul(query_layer, key_layer.transpose(-2, -1))) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        )
        
        # 对注意力概率进行 softmax 归一化
        attention_probs = attention_probs.softmax(dim=-1)

        # 计算上下文层，将注意力概率应用于值层，再进行维度转置
        context_layer = torch.matmul(attention_probs, value_layer).transpose(1, 2)
        
        # 将上下文层重塑为(batch_size, sequence_length, total_expanded_key_dim)
        context_layer = context_layer.reshape(batch_size, sequence_length, self.total_expanded_key_dim)
        
        # 使用投影层处理上下文层，得到最终输出的上下文层
        context_layer = self.projection(context_layer)

        # 如果输出注意力权重，则将其加入输出元组
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # 返回最终输出的元组
        return outputs
# 定义一个 EfficientFormerConvStem 类，继承自 nn.Module 类
class EfficientFormerConvStem(nn.Module):
    # 初始化函数，接受 EfficientFormerConfig 类型的 config 参数和一个整数 out_channels 参数
    def __init__(self, config: EfficientFormerConfig, out_channels: int):
        # 调用父类 nn.Module 的初始化函数
        super().__init__()

        # 创建一个 2D 卷积层，输入通道数为 config.num_channels，输出通道数为 out_channels // 2，卷积核大小为 3x3，步幅为 2，填充为 1
        self.convolution1 = nn.Conv2d(config.num_channels, out_channels // 2, kernel_size=3, stride=2, padding=1)
        # 创建一个批标准化层，输入通道数为 out_channels // 2，epsilon 参数为 config.batch_norm_eps
        self.batchnorm_before = nn.BatchNorm2d(out_channels // 2, eps=config.batch_norm_eps)

        # 创建第二个 2D 卷积层，输入通道数为 out_channels // 2，输出通道数为 out_channels，卷积核大小为 3x3，步幅为 2，填充为 1
        self.convolution2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1)
        # 创建第二个批标准化层，输入通道数为 out_channels，epsilon 参数为 config.batch_norm_eps
        self.batchnorm_after = nn.BatchNorm2d(out_channels, eps=config.batch_norm_eps)

        # 创建一个 ReLU 激活函数
        self.activation = nn.ReLU()

    # 前向传播函数，接受一个名为 pixel_values 的 torch.Tensor 输入，返回一个 torch.Tensor 输出
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 对输入 pixel_values 进行第一次卷积和批标准化，然后应用激活函数
        features = self.batchnorm_before(self.convolution1(pixel_values))
        features = self.activation(features)
        # 对前一步得到的特征再进行一次卷积和批标准化，然后再应用激活函数
        features = self.batchnorm_after(self.convolution2(features))
        features = self.activation(features)

        # 返回处理后的特征
        return features


# 定义一个 EfficientFormerPooling 类，继承自 nn.Module 类
class EfficientFormerPooling(nn.Module):
    # 初始化函数，接受一个整数 pool_size 参数
    def __init__(self, pool_size: int):
        # 调用父类 nn.Module 的初始化函数
        super().__init__()
        # 创建一个平均池化层，池化大小为 pool_size x pool_size，步幅为 1，填充大小为 pool_size // 2，不包括填充部分到计算中
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    # 前向传播函数，接受一个名为 hidden_states 的 torch.Tensor 输入，返回一个 torch.Tensor 输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对输入 hidden_states 执行平均池化操作，然后从原始 hidden_states 中减去池化结果
        output = self.pool(hidden_states) - hidden_states
        # 返回处理后的输出
        return output


# 定义一个 EfficientFormerDenseMlp 类，继承自 nn.Module 类
class EfficientFormerDenseMlp(nn.Module):
    # 初始化函数，接受一个 EfficientFormerConfig 类型的 config 参数，整数 in_features、hidden_features 和 out_features 参数
    def __init__(
        self,
        config: EfficientFormerConfig,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        # 调用父类 nn.Module 的初始化函数
        super().__init__()
        # 如果未提供 out_features，则设置为输入的 in_features
        out_features = out_features or in_features
        # 如果未提供 hidden_features，则设置为输入的 in_features
        hidden_features = hidden_features or in_features

        # 创建一个线性层，输入特征数为 in_features，输出特征数为 hidden_features
        self.linear_in = nn.Linear(in_features, hidden_features)
        # 根据配置中的 hidden_act 属性选择相应的激活函数
        self.activation = ACT2FN[config.hidden_act]
        # 创建一个以 config.hidden_dropout_prob 为概率丢弃部分神经元的 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个线性层，输入特征数为 hidden_features，输出特征数为 out_features
        self.linear_out = nn.Linear(hidden_features, out_features)

    # 前向传播函数，接受一个名为 hidden_states 的 torch.Tensor 输入，返回一个 torch.Tensor 输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对输入 hidden_states 进行线性变换
        hidden_states = self.linear_in(hidden_states)
        # 应用预先选择的激活函数
        hidden_states = self.activation(hidden_states)
        # 对激活后的结果应用 dropout
        hidden_states = self.dropout(hidden_states)
        # 再次进行线性变换
        hidden_states = self.linear_out(hidden_states)
        # 再次应用 dropout
        hidden_states = self.dropout(hidden_states)

        # 返回处理后的输出
        return hidden_states


# 定义一个 EfficientFormerConvMlp 类，继承自 nn.Module 类
class EfficientFormerConvMlp(nn.Module):
    # 初始化函数，接受一个 EfficientFormerConfig 类型的 config 参数，整数 in_features、hidden_features 和 out_features 参数，以及一个浮点数 drop 参数，默认为 0.0
    def __init__(
        self,
        config: EfficientFormerConfig,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        # 调用父类 nn.Module 的初始化函数
        super().__init__()
        # 如果未提供 out_features，则设置为输入的 in_features
        out_features = out_features or in_features
        # 如果未提供 hidden_features，则设置为输入的 in_features
        hidden_features = hidden_features or in_features
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 如果未指定输出特征数，则默认与输入特征数相同
        out_features = out_features or in_features
        # 如果未指定隐藏层特征数，则默认与输入特征数相同
        hidden_features = hidden_features or in_features

        # 定义第一个卷积层，输入特征数为in_features，输出特征数为hidden_features，卷积核大小为1x1
        self.convolution1 = nn.Conv2d(in_features, hidden_features, 1)
        # 根据配置选择激活函数
        self.activation = ACT2FN[config.hidden_act]
        # 定义第二个卷积层，输入特征数为hidden_features，输出特征数为out_features，卷积核大小为1x1
        self.convolution2 = nn.Conv2d(hidden_features, out_features, 1)
        # 定义一个Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(drop)

        # 定义第一个批归一化层，对hidden_features个通道的特征进行归一化，epsilon设为config中的batch_norm_eps
        self.batchnorm_before = nn.BatchNorm2d(hidden_features, eps=config.batch_norm_eps)
        # 定义第二个批归一化层，对out_features个通道的特征进行归一化，epsilon设为config中的batch_norm_eps
        self.batchnorm_after = nn.BatchNorm2d(out_features, eps=config.batch_norm_eps)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # 第一层卷积操作，将hidden_state作为输入
        hidden_state = self.convolution1(hidden_state)
        # 第一层卷积后进行批归一化操作
        hidden_state = self.batchnorm_before(hidden_state)

        # 使用指定的激活函数对特征进行非线性变换
        hidden_state = self.activation(hidden_state)
        # 对特征进行Dropout操作，以减少过拟合风险
        hidden_state = self.dropout(hidden_state)
        # 第二层卷积操作
        hidden_state = self.convolution2(hidden_state)

        # 第二层卷积后进行批归一化操作
        hidden_state = self.batchnorm_after(hidden_state)
        # 再次对特征进行Dropout操作
        hidden_state = self.dropout(hidden_state)

        # 返回最终的特征表示
        return hidden_state
# Copied from transformers.models.convnext.modeling_convnext.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    # 如果 dropout 概率为 0 或者不处于训练模式，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留的概率
    keep_prob = 1 - drop_prob
    # 生成与输入形状相同的随机张量，用于随机丢弃路径
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 将随机张量二值化
    # 计算输出，将输入按照保留概率进行缩放，并且乘以随机张量
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath with Beit->EfficientFormer
class EfficientFormerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用上面定义的 drop_path 函数来实现 drop path 功能
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        # 返回描述对象的字符串，包括 drop_prob 参数的信息
        return "p={}".format(self.drop_prob)


class EfficientFormerFlat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        # 将输入张量展平，并且交换维度以适应特定的输出格式
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states


class EfficientFormerMeta3D(nn.Module):
    # 这里是定义的一个类，暂未提供具体实现
    # 初始化函数，用于创建 EfficientFormer 类的实例，接收配置对象、维度和可选的 drop_path 参数
    def __init__(self, config: EfficientFormerConfig, dim: int, drop_path: float = 0.0):
        # 调用父类的初始化方法
        super().__init__()
    
        # 创建 EfficientFormerSelfAttention 实例，用于处理 token_mixer 的自注意力机制
        self.token_mixer = EfficientFormerSelfAttention(
            dim=config.dim,                     # 设置维度参数
            key_dim=config.key_dim,             # 设置键的维度
            num_heads=config.num_attention_heads,  # 设置注意力头的数量
            attention_ratio=config.attention_ratio,  # 设置注意力机制的比率
            resolution=config.resolution,       # 设置分辨率参数
        )
    
        # 创建 LayerNorm 层，用于第一层的归一化
        self.layernorm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建 LayerNorm 层，用于第二层的归一化
        self.layernorm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
    
        # 计算 MLP 隐藏层的维度
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        # 创建 EfficientFormerDenseMlp 实例，用于多层感知机操作
        self.mlp = EfficientFormerDenseMlp(config, in_features=dim, hidden_features=mlp_hidden_dim)
    
        # 如果 drop_path 大于 0，则创建 EfficientFormerDropPath 实例，否则创建单位函数（Identity）
        self.drop_path = EfficientFormerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
        # 检查是否使用层缩放
        self.use_layer_scale = config.use_layer_scale
        if config.use_layer_scale:
            # 创建可学习的参数，初始化为 config.layer_scale_init_value 倍的 dim 维度张量，用于第一层的缩放
            self.layer_scale_1 = nn.Parameter(config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            # 创建可学习的参数，初始化为 config.layer_scale_init_value 倍的 dim 维度张量，用于第二层的缩放
            self.layer_scale_2 = nn.Parameter(config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)
    
    # 前向传播函数，接收隐藏状态张量和输出注意力权重的标志，返回元组类型的张量
    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False) -> Tuple[torch.Tensor]:
        # 进行 token_mixer 的自注意力计算，并应用第一层的 LayerNorm
        self_attention_outputs = self.token_mixer(self.layernorm1(hidden_states), output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力权重信息到 outputs 中
    
        # 如果使用层缩放，则按照层缩放因子进行加权和操作，否则直接应用 drop_path 和 MLP
        if self.use_layer_scale:
            layer_output = hidden_states + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0) * attention_output
            )
            layer_output = layer_output + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(self.layernorm2(layer_output))
            )
        else:
            layer_output = hidden_states + self.drop_path(attention_output)
            layer_output = layer_output + self.drop_path(self.mlp(self.layernorm2(layer_output)))
    
        # 将最终的层输出添加到 outputs 中，并返回
        outputs = (layer_output,) + outputs
    
        return outputs
# 定义一个 EfficientFormerMeta4DLayers 类，继承自 nn.Module 类
class EfficientFormerMeta4DLayers(nn.Module):
    # 初始化方法，接收一个 EfficientFormerConfig 类型的 config 参数
    def __init__(self, config: EfficientFormerConfig):
        # 调用父类的初始化方法
        super().__init__()
        
        # 计算每个块的 drop path 值列表
        drop_paths = [
            config.drop_path_rate * (block_idx + sum(config.depths[:-1]))
            for block_idx in range(config.num_meta4d_blocks)
        ]
        
        # 使用列表推导式创建一个 nn.ModuleList，包含多个 EfficientFormerMeta4D 实例化对象
        self.blocks = nn.ModuleList(
            [EfficientFormerMeta4D(config, config.hidden_sizes[-1], drop_path=drop_path) for drop_path in drop_paths]
        )

    # 前向传播方法，接收输入的 hidden_states 张量和一个布尔类型的 output_attentions 参数，返回一个元组
    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False) -> Tuple[torch.Tensor]:
        # 如果 output_attentions 为 False，则初始化一个空元组 all_attention_outputs
        all_attention_outputs = () if output_attentions else None

        # 遍历 self.blocks 中的每个 layer_module
        for layer_module in self.blocks:
            # 如果 hidden_states 是元组，则取其第一个元素
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            # 调用当前层的 layer_module 进行前向传播，更新 hidden_states
            hidden_states = layer_module(hidden_states)

            # 如果 output_attentions 为 True，则将当前层的注意力输出加入 all_attention_outputs 元组中
            if output_attentions:
                all_attention_outputs = all_attention_outputs + (hidden_states[1],)

        # 如果 output_attentions 为 True，则构造输出元组 outputs
        if output_attentions:
            outputs = (hidden_states[0],) + all_attention_outputs
            return outputs

        # 返回最终的 hidden_states
        return hidden_states
    def __init__(self, config: EfficientFormerConfig, stage_idx: int):
        # 调用父类的初始化方法
        super().__init__()
        # 根据给定阶段索引获取层的数量
        num_layers = (
            config.depths[stage_idx] if stage_idx != -1 else config.depths[stage_idx] - config.num_meta3d_blocks
        )
        # 计算每个块的丢弃路径率并存储在列表中
        drop_paths = [
            config.drop_path_rate * (block_idx + sum(config.depths[:stage_idx])) for block_idx in range(num_layers)
        ]

        # 创建包含各个块的模块列表
        self.blocks = nn.ModuleList(
            [
                EfficientFormerMeta4D(config, config.hidden_sizes[stage_idx], drop_path=drop_path)
                for drop_path in drop_paths
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        # 遍历每个块模块并对输入的隐藏状态进行处理
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
class EfficientFormerIntermediateStage(nn.Module):
    def __init__(self, config: EfficientFormerConfig, index: int):
        super().__init__()
        # 创建 EfficientFormerMeta4DLayers 实例作为中间层处理器
        self.meta4D_layers = EfficientFormerMeta4DLayers(config, index)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        # 调用中间层处理器处理隐藏状态张量
        hidden_states = self.meta4D_layers(hidden_states)
        return hidden_states


class EfficientFormerLastStage(nn.Module):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__()
        # 创建 EfficientFormerMeta4DLayers 实例作为最后阶段处理器
        self.meta4D_layers = EfficientFormerMeta4DLayers(config, -1)
        # 创建 EfficientFormerFlat 实例用于扁平化处理
        self.flat = EfficientFormerFlat()
        # 创建 EfficientFormerMeta3DLayers 实例作为三维层处理器
        self.meta3D_layers = EfficientFormerMeta3DLayers(config)

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False) -> Tuple[torch.Tensor]:
        # 调用最后阶段处理器处理隐藏状态张量
        hidden_states = self.meta4D_layers(hidden_states)
        # 调用扁平化处理器处理隐藏状态张量
        hidden_states = self.flat(hidden_states)
        # 调用三维层处理器处理隐藏状态张量和注意力输出标志
        hidden_states = self.meta3D_layers(hidden_states, output_attentions)
        return hidden_states


class EfficientFormerEncoder(nn.Module):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__()
        self.config = config
        num_intermediate_stages = len(config.depths) - 1
        # 根据配置计算是否需要降采样
        downsamples = [
            config.downsamples[i] or config.hidden_sizes[i] != config.hidden_sizes[i + 1]
            for i in range(num_intermediate_stages)
        ]
        intermediate_stages = []

        # 构建中间阶段模块列表
        for i in range(num_intermediate_stages):
            # 添加 EfficientFormerIntermediateStage 实例到中间阶段列表
            intermediate_stages.append(EfficientFormerIntermediateStage(config, i))
            # 如果需要降采样，添加 EfficientFormerPatchEmbeddings 实例到中间阶段列表
            if downsamples[i]:
                intermediate_stages.append(
                    EfficientFormerPatchEmbeddings(config, config.hidden_sizes[i], config.hidden_sizes[i + 1])
                )

        # 使用 nn.ModuleList 封装中间阶段模块列表
        self.intermediate_stages = nn.ModuleList(intermediate_stages)
        # 创建 EfficientFormerLastStage 实例作为最后阶段处理器
        self.last_stage = EfficientFormerLastStage(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 如果输出隐藏状态，初始化一个空元组用于存储所有隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 遍历中间层模块并逐层计算隐藏状态
        for layer_module in self.intermediate_stages:
            hidden_states = layer_module(hidden_states)
            # 如果输出隐藏状态，将当前层的隐藏状态加入到存储中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # 调用最后一个阶段模块计算最终输出
        layer_output = self.last_stage(hidden_states, output_attentions=output_attentions)

        # 如果输出注意力权重，将当前层的注意力权重加入到存储中
        if output_attentions:
            all_self_attentions = all_self_attentions + layer_output[1:]

        # 如果输出隐藏状态，将最后一层的隐藏状态加入到存储中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (layer_output[0],)

        # 如果不返回字典形式的结果，将各部分非空的结果组成元组返回
        if not return_dict:
            return tuple(v for v in [layer_output[0], all_hidden_states, all_self_attentions] if v is not None)

        # 返回以字典形式封装的 BaseModelOutput 对象
        return BaseModelOutput(
            last_hidden_state=layer_output[0],
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
@add_start_docstrings(
    "The bare EfficientFormer Model transformer outputting raw hidden-states without any specific head on top.",
    EFFICIENTFORMER_START_DOCSTRING,
)
class EfficientFormerModel(EfficientFormerPreTrainedModel):
    """
    EfficientFormerModel extends EfficientFormerPreTrainedModel and represents a transformer model architecture 
    without specific task heads.

    Args:
        config (EfficientFormerConfig): The configuration class for initializing the model.

    Attributes:
        patch_embed (EfficientFormerConvStem): Patch embedding layer.
        encoder (EfficientFormerEncoder): Transformer encoder.
        layernorm (nn.LayerNorm): Layer normalization for the final output.

    Methods:
        forward: Implements the forward pass of the model.

    Inherits from:
        EfficientFormerPreTrainedModel: Handles weights initialization and pretrained model loading interface.
    """

    def __init__(self, config: EfficientFormerConfig):
        super().__init__(config)
        self.config = config

        # Initialize patch embedding layer
        self.patch_embed = EfficientFormerConvStem(config, config.hidden_sizes[0])
        # Initialize transformer encoder
        self.encoder = EfficientFormerEncoder(config)
        # Layer normalization for the final output
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    def forward(self, pixel_values, output_attentions=False, output_hidden_states=False, return_dict=True):
        """
        Defines the forward pass of EfficientFormerModel.

        Args:
            pixel_values (torch.FloatTensor): Input pixel values of shape (batch_size, num_channels, height, width).
            output_attentions (bool, optional): Whether to return attention tensors of all layers.
            output_hidden_states (bool, optional): Whether to return hidden states of all layers.
            return_dict (bool, optional): Whether to return a ModelOutput instead of a tuple.

        Returns:
            ModelOutput or tuple:
                Depending on `return_dict`, either:
                - ModelOutput if `return_dict=True` (default),
                - A tuple of torch.FloatTensor otherwise.
        """
        pass  # Placeholder for the actual implementation of the forward method
    # 使用 @add_code_sample_docstrings 装饰器添加文档字符串，用于代码示例的文档化
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 指定文档化的检查点（checkpoint）
        output_type=BaseModelOutputWithPooling,  # 指定输出类型为包含汇总的基础模型输出
        config_class=_CONFIG_FOR_DOC,  # 指定用于文档化的配置类
        modality="vision",  # 指定模态性（此处为视觉）
        expected_output=_EXPECTED_OUTPUT_SHAPE,  # 指定预期输出的形状
    )
    # 定义前向传播方法，接收输入的像素值、是否输出注意力、是否输出隐藏状态、是否返回字典等参数，返回联合类型的结果或基础模型输出
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,  # 输入参数：像素值，默认为空
        output_attentions: Optional[bool] = None,  # 输入参数：是否输出注意力，默认为空
        output_hidden_states: Optional[bool] = None,  # 输入参数：是否输出隐藏状态，默认为空
        return_dict: Optional[bool] = None,  # 输入参数：是否返回字典，默认为空
    ) -> Union[tuple, BaseModelOutput]:
        # 如果未提供像素值，则抛出数值错误
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据输入或配置设定是否输出注意力
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据输入或配置设定是否输出隐藏状态
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据输入或配置设定是否使用返回字典

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        # 如果像素值为空，则引发数值错误异常

        # 将像素值传递给 patch_embed 方法进行嵌入
        embedding_output = self.patch_embed(pixel_values)
        # 使用编码器处理嵌入输出，根据参数设定是否输出注意力和隐藏状态
        encoder_outputs = self.encoder(
            embedding_output, output_attentions=output_attentions, output_hidden_states=output_hidden_states
        )

        # 获取编码器的序列输出
        sequence_output = encoder_outputs[0]
        # 对序列输出进行层归一化处理
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            # 如果不要求返回字典，则返回元组形式的头部输出和编码器其他输出状态
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # 否则，返回基础模型输出对象，包含最终隐藏状态、所有隐藏状态和注意力
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 使用自定义的文档字符串描述 EfficientFormer 模型，这是一个在顶部增加了图像分类头部的转换器模型，例如用于 ImageNet 数据集的场景。
@add_start_docstrings(
    """
    EfficientFormer Model transformer with an image classification head on top (a linear layer on top of the final
    hidden state of the [CLS] token) e.g. for ImageNet.
    """,
    EFFICIENTFORMER_START_DOCSTRING,
)
class EfficientFormerForImageClassification(EfficientFormerPreTrainedModel):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__(config)

        # 初始化模型的标签数量
        self.num_labels = config.num_labels
        # 初始化 EfficientFormer 模型
        self.efficientformer = EfficientFormerModel(config)

        # 分类器头部
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 此方法用于模型的前向传播，接受输入参数如像素值、标签等，并返回模型输出
        # 具体文档化细节参见 add_start_docstrings_to_model_forward 和 add_code_sample_docstrings 装饰器
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用给定的值，否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Efficientformer 处理输入的像素值，根据参数设置输出注意力和隐藏状态，并返回相应的对象或字典
        outputs = self.efficientformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 Efficientformer 的输出中获取序列输出
        sequence_output = outputs[0]

        # 将序列输出传入分类器，计算 logits
        logits = self.classifier(sequence_output.mean(-2))

        # 初始化 loss 为 None
        loss = None
        if labels is not None:
            # 确定问题类型（回归或分类）
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数和计算损失
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

        # 如果 return_dict 为 False，则按照一定格式返回输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则构造 ImageClassifierOutput 对象并返回
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 用于存储图像分类模型输出的数据类，继承自`ModelOutput`
@dataclass
class EfficientFormerForImageClassificationWithTeacherOutput(ModelOutput):
    """
    [`EfficientFormerForImageClassificationWithTeacher`] 的输出类型。

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            预测分数，是 `cls_logits` 和 `distillation_logits` 的平均值。
        cls_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类头部的预测分数（即最终隐藏状态的类标记之上的线性层）。
        distillation_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            蒸馏头部的预测分数（即最终隐藏状态的蒸馏标记之上的线性层）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当 `output_hidden_states=True` 传递或 `config.output_hidden_states=True` 时返回):
            `torch.FloatTensor` 元组（一个用于嵌入的输出 + 每层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。
            模型在每层输出的隐藏状态加上初始嵌入的输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当 `output_attentions=True` 传递或 `config.output_attentions=True` 时返回):
            `torch.FloatTensor` 元组（每层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            在注意力 softmax 之后的注意力权重，用于计算自注意力头中的加权平均值。
    """


# 基于 `EfficientFormerPreTrainedModel` 的图像分类头部模型变换器，包含两个线性层（一个在 [CLS] 标记的最终隐藏状态之上，一个在蒸馏标记的最终隐藏状态之上），例如用于 ImageNet。
@add_start_docstrings(
    """
    `EfficientFormer` 模型变换器，其顶部包含图像分类头部（一个在 [CLS] 标记的最终隐藏状态之上的线性层，一个在蒸馏标记的最终隐藏状态之上的线性层），
    例如用于 ImageNet。

    <Tip warning={true}>

           此模型仅支持推断。目前不支持使用蒸馏进行微调（即带有教师模型）。

    </Tip>
    """,
    EFFICIENTFORMER_START_DOCSTRING,
)
class EfficientFormerForImageClassificationWithTeacher(EfficientFormerPreTrainedModel):
    # 初始化函数，接受一个 EfficientFormerConfig 类型的参数 config
    def __init__(self, config: EfficientFormerConfig):
        # 调用父类的初始化方法，传入 config 参数
        super().__init__(config)

        # 将配置中的 num_labels 属性赋值给当前对象的 num_labels 属性
        self.num_labels = config.num_labels
        # 根据配置创建一个 EfficientFormerModel 对象，并赋值给当前对象的 efficientformer 属性
        self.efficientformer = EfficientFormerModel(config)

        # 分类器头部，根据配置中的 hidden_size 和 num_labels 创建线性分类器或者恒等映射
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        # 蒸馏头部，根据配置中的 hidden_size 和 num_labels 创建线性分类器或者恒等映射
        self.distillation_classifier = (
            nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 调用初始化权重和应用最终处理的函数
        self.post_init()

    # 前向传播函数，接受多个参数，返回一个包含预测输出的对象或元组
    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=EfficientFormerForImageClassificationWithTeacherOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, EfficientFormerForImageClassificationWithTeacherOutput]:
        # 如果 return_dict 参数为 None，则使用配置中的 use_return_dict 属性
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 调用 EfficientFormerModel 的前向传播方法，传入相应参数，并获取输出
        outputs = self.efficientformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从输出中获取序列输出（通常是模型最后一层的输出）
        sequence_output = outputs[0]

        # 对序列输出进行均值池化，并通过分类器头部获取分类器预测结果
        cls_logits = self.classifier(sequence_output.mean(-2))
        # 对序列输出进行均值池化，并通过蒸馏头部获取蒸馏预测结果
        distillation_logits = self.distillation_classifier(sequence_output.mean(-2))

        # 在推断过程中，返回两个分类器预测结果的平均值作为最终预测值
        logits = (cls_logits + distillation_logits) / 2

        # 如果 return_dict 为 False，则返回一个包含所有输出和预测结果的元组
        if not return_dict:
            output = (logits, cls_logits, distillation_logits) + outputs[1:]
            return output

        # 如果 return_dict 为 True，则返回一个包含输出对象及相关属性的 EfficientFormerForImageClassificationWithTeacherOutput 对象
        return EfficientFormerForImageClassificationWithTeacherOutput(
            logits=logits,
            cls_logits=cls_logits,
            distillation_logits=distillation_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```