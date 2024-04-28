# `.\models\efficientformer\modeling_efficientformer.py`

```py
# coding=utf-8
# 设置文件编码为UTF-8

# 2022年Snapchat Research和The HuggingFace Inc.团队版权所有。
# 根据Apache许可证第2.0版("许可证")授权;
# 您不得使用此文件，除非符合许可证的规定。
# 您可以在以下地址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或写入协议，否则不得分发软件
# 根据许可证的规定，在"原样"的基础上分发
# 没有任何明示或暗示的保证或条件
# 有关特定语言的权限和限制，请参阅许可证

""" PyTorch EfficientFormer model."""

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

# 获取Logger实例
logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "EfficientFormerConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "snap-research/efficientformer-l1-300"
_EXPECTED_OUTPUT_SHAPE = [1, 49, 448]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "snap-research/efficientformer-l1-300"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"

# EfficientFormer预训练模型列表
EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "snap-research/efficientformer-l1-300",
    # 查看所有EfficientFormer模型 https://huggingface.co/models?filter=efficientformer
]


class EfficientFormerPatchEmbeddings(nn.Module):
    """
    This class performs downsampling between two stages. For the input tensor with the shape [batch_size, num_channels,
    height, width] it produces output tensor with the shape [batch_size, num_channels, height/stride, width/stride]
    """
    # EfficientFormerPatchEmbeddings类，用于在两个阶段之间进行下采样，对于形状为[batch_size, num_channels, height, width]的输入张量，
    # 它产生形状为[batch_size, num_channels, height/stride, width/stride]的输出张量
    def __init__(self, config: EfficientFormerConfig, num_channels: int, embed_dim: int, apply_norm: bool = True):
        # 初始化函数
        super().__init__()
        self.num_channels = num_channels

        # 创建卷积层，用于投影
        self.projection = nn.Conv2d(
            num_channels,
            embed_dim,
            kernel_size=config.downsample_patch_size,
            stride=config.downsample_stride,
            padding=config.downsample_pad,
        )
        # 如果apply_norm为真，则创建批归一化层，否则创建单位矩阵层
        self.norm = nn.BatchNorm2d(embed_dim, eps=config.batch_norm_eps) if apply_norm else nn.Identity()
    # 定义一个前向传播函数，接收像素数值作为输入，返回处理后的数据
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 获取输入数据的批处理大小、通道数、高度和宽度
        batch_size, num_channels, height, width = pixel_values.shape
        # 如果输入数据的通道数与预设的通道数不匹配，抛出数值错误异常
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        # 通过投影层处理输入数据，得到嵌入向量
        embeddings = self.projection(pixel_values)
        # 对嵌入向量进行归一化处理
        embeddings = self.norm(embeddings)

        # 返回处理后的嵌入向量
        return embeddings
class EfficientFormerSelfAttention(nn.Module):
    def __init__(self, dim: int, key_dim: int, num_heads: int, attention_ratio: int, resolution: int):
        super().__init__()

        self.num_heads = num_heads  # 存储注意力头的数量
        self.key_dim = key_dim  # 存储键的维度
        self.attention_ratio = attention_ratio  # 存储注意力扩展比率
        self.scale = key_dim**-0.5  # 缩放参数
        self.total_key_dim = key_dim * num_heads  # 总键的维度
        self.expanded_key_dim = int(attention_ratio * key_dim)  # 扩展后的键的维度
        self.total_expanded_key_dim = int(self.expanded_key_dim * num_heads)  # 总扩展键的维度
        hidden_size = self.total_expanded_key_dim + self.total_key_dim * 2  # 隐藏层的尺寸
        self.qkv = nn.Linear(dim, hidden_size)  # 将输入维度映射到隐藏层
        self.projection = nn.Linear(self.total_expanded_key_dim, dim)  # 投影操作将扩展键的维度映射回原始维度
        points = list(itertools.product(range(resolution), range(resolution)))  # 生成坐标点列表
        num_points = len(points)  # 坐标点的数量
        attention_offsets = {}  # 存储注意力偏移量的字典
        idxs = []  # 存储偏移量索引的列表
        for point_1 in points:  # 遍历所有坐标点
            for point_2 in points:  # 再次遍历所有坐标点
                offset = (abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))  # 计算偏移量
                if offset not in attention_offsets:  # 如果偏移量不在字典中
                    attention_offsets[offset] = len(attention_offsets)  # 添加到字典中
                idxs.append(attention_offsets[offset])  # 将偏移量索引添加到列表中
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))  # 定义注意力偏置参数
        self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(num_points, num_points))  # 注册注意力偏置索引作为缓冲区

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)  # 调用父类的训练方法
        if mode and hasattr(self, "ab"):  # 如果模式为True且self对象具有属性"ab"
            del self.ab  # 删除"ab"属性
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]  # 计算并存储注意力偏置
    # 前向传播函数，用于执行自注意力机制
    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False) -> Tuple[torch.Tensor]:
        # 获取输入张量的维度信息
        batch_size, sequence_length, num_channels = hidden_states.shape
        # 使用全连接层进行查询、键、值的映射
        qkv = self.qkv(hidden_states)
        # 将输出张量按照指定维度进行分割，得到查询、键、值张量
        query_layer, key_layer, value_layer = qkv.reshape(batch_size, sequence_length, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.expanded_key_dim], dim=3
        )
        # 调整张量的维度顺序，使得通道维度在前，以便进行矩阵乘法
        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = key_layer.permute(0, 2, 1, 3)
        value_layer = value_layer.permute(0, 2, 1, 3)

        # 如果模型处于推理状态，则将存储在设备上的注意力偏置张量与计算的注意力偏置相结合
        if not self.training:
            self.ab = self.ab.to(self.attention_biases.device)
        # 计算注意力得分，包括矩阵乘法、缩放和注意力偏置
        attention_probs = (torch.matmul(query_layer, key_layer.transpose(-2, -1))) * self.scale + (
            self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab
        )

        # 对注意力得分进行 softmax 操作，得到注意力权重
        attention_probs = attention_probs.softmax(dim=-1)

        # 使用注意力权重对值张量进行加权求和，得到上下文张量
        context_layer = torch.matmul(attention_probs, value_layer).transpose(1, 2)
        # 调整上下文张量的维度，以便后续的投影操作
        context_layer = context_layer.reshape(batch_size, sequence_length, self.total_expanded_key_dim)
        # 使用投影层进行维度转换
        context_layer = self.projection(context_layer)

        # 根据需要是否返回注意力权重
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # 返回输出结果
        return outputs
# 定义一个 EfficientFormerConvStem 类，用于构建 EfficientFormer 模型的卷积块
class EfficientFormerConvStem(nn.Module):
    def __init__(self, config: EfficientFormerConfig, out_channels: int):
        super().__init__()

        # 定义第一个卷积层，输入通道数为 config.num_channels，输出通道数为 out_channels // 2，卷积核大小为 3x3，步长为 2，填充为 1
        self.convolution1 = nn.Conv2d(config.num_channels, out_channels // 2, kernel_size=3, stride=2, padding=1)
        # 定义第一个批归一化层，通道数为 out_channels // 2，epsilon 为 config.batch_norm_eps
        self.batchnorm_before = nn.BatchNorm2d(out_channels // 2, eps=config.batch_norm_eps)

        # 定义第二个卷积层，输入通道数为 out_channels // 2，输出通道数为 out_channels，卷积核大小为 3x3，步长为 2，填充为 1
        self.convolution2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1)
        # 定义第二个批归一化层，通道数为 out_channels，epsilon 为 config.batch_norm_eps
        self.batchnorm_after = nn.BatchNorm2d(out_channels, eps=config.batch_norm_eps)

        # 定义激活函数为 ReLU
        self.activation = nn.ReLU()

    # 前向传播方法
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 经过第一个卷积层和批归一化层后的特征
        features = self.batchnorm_before(self.convolution1(pixel_values))
        # 经过激活函数
        features = self.activation(features)
        # 经过第二个卷积层和批归一化层后的特征
        features = self.batchnorm_after(self.convolution2(features))
        # 经过激活函数
        features = self.activation(features)

        return features


# 定义一个 EfficientFormerPooling 类，用于构建 EfficientFormer 模型的池化层
class EfficientFormerPooling(nn.Module):
    def __init__(self, pool_size: int):
        super().__init__()
        # 定义平均池化层，池化核大小为 pool_size x pool_size，步长为 1，填充大小为池化核大小的一半，不包含填充
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对隐藏状态进行池化操作
        output = self.pool(hidden_states) - hidden_states
        return output


# 定义一个 EfficientFormerDenseMlp 类，用于构建 EfficientFormer 模型的全连接层
class EfficientFormerDenseMlp(nn.Module):
    def __init__(
        self,
        config: EfficientFormerConfig,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        # 如果未指定隐藏层特征数，则设置为输入特征数
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # 输入全连接层，输入特征数为 in_features，输出特征数为 hidden_features
        self.linear_in = nn.Linear(in_features, hidden_features)
        # 激活函数
        self.activation = ACT2FN[config.hidden_act]
        # Dropout 层，丢弃概率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 输出全连接层，输入特征数为 hidden_features，输出特征数为 out_features
        self.linear_out = nn.Linear(hidden_features, out_features)

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入全连接层
        hidden_states = self.linear_in(hidden_states)
        # 激活函数
        hidden_states = self.activation(hidden_states)
        # Dropout 层
        hidden_states = self.dropout(hidden_states)
        # 输出全连接层
        hidden_states = self.linear_out(hidden_states)
        # Dropout 层
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# 定义一个 EfficientFormerConvMlp 类，用于构建 EfficientFormer 模型的卷积全连接混合层
class EfficientFormerConvMlp(nn.Module):
    def __init__(
        self,
        config: EfficientFormerConfig,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    # 定义一个类，继承自 nn.Module 类
    ):
        # 调用父类的构造函数
        super().__init__()
        # 如果输出特征数未指定，则与输入特征数相同
        out_features = out_features or in_features
        # 如果隐藏特征数未指定，则与输入特征数相同
        hidden_features = hidden_features or in_features

        # 第一个卷积层，输入特征数为 in_features，输出特征数为 hidden_features，卷积核大小为 1x1
        self.convolution1 = nn.Conv2d(in_features, hidden_features, 1)
        # 激活函数根据配置文件中的隐藏激活函数选择对应的激活函数
        self.activation = ACT2FN[config.hidden_act]
        # 第二个卷积层，输入特征数为 hidden_features，输出特征数为 out_features，卷积核大小为 1x1
        self.convolution2 = nn.Conv2d(hidden_features, out_features, 1)
        # Dropout 层，概率为 drop
        self.dropout = nn.Dropout(drop)

        # 第一个批标准化层，输入特征数为 hidden_features，epsilon 为配置文件中的批标准化 epsilon 值
        self.batchnorm_before = nn.BatchNorm2d(hidden_features, eps=config.batch_norm_eps)
        # 第二个批标准化层，输入特征数为 out_features，epsilon 为配置文件中的批标准化 epsilon 值
        self.batchnorm_after = nn.BatchNorm2d(out_features, eps=config.batch_norm_eps)

    # 前向传播函数，接收隐藏状态作为输入，返回处理后的隐藏状态
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # 第一个卷积层处理隐藏状态
        hidden_state = self.convolution1(hidden_state)
        # 第一个批标准化层处理卷积结果
        hidden_state = self.batchnorm_before(hidden_state)

        # 使用激活函数处理批标准化后的结果
        hidden_state = self.activation(hidden_state)
        # Dropout 层处理激活后的结果
        hidden_state = self.dropout(hidden_state)
        # 第二个卷积层处理dropout后的结果
        hidden_state = self.convolution2(hidden_state)

        # 第二个批标准化层处理卷积后的结果
        hidden_state = self.batchnorm_after(hidden_state)
        # Dropout 层处理批标准化后的结果
        hidden_state = self.dropout(hidden_state)

        # 返回处理后的隐藏状态
        return hidden_state
# 从transformers.models.convnext.modeling_convnext.drop_path中复制了drop_path函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    为每个样本丢弃路径（随机深度），当应用于残差块的主路径时。

    Ross Wightman的评论：这与我为EfficientNet等网络创建的DropConnect实现相同，
    然而，原始名称具有误导性，因为'Drop Connect'是在另一篇论文中描述的不同形式的丢失连接...
    参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    我选择更改层和参数名称为'drop path'，而不是将DropConnect作为层名称混合使用，并将'survival rate'用作参数。
    """
    if drop_prob == 0.0 or not training:
        # 如果丢失概率为0或者不处于训练模式，则直接返回输入
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度张量，而不仅仅是2D卷积网络
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 二值化
    output = input.div(keep_prob) * random_tensor
    return output


# 从transformers.models.beit.modeling_beit.BeitDropPath中复制了EfficientFormerDropPath类，将Beit->EfficientFormer
class EfficientFormerDropPath(nn.Module):
    """每个样本（当应用于残差块的主路径时）丢弃路径（随机深度）。"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class EfficientFormerFlat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        # 将隐藏状态展平为2维并转置
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states


class EfficientFormerMeta3D(nn.Module):
    # 初始化方法
    def __init__(self, config: EfficientFormerConfig, dim: int, drop_path: float = 0.0):
        # 调用父类的初始化方法
        super().__init__()

        # 创建 EfficientFormerSelfAttention 对象，token_mixer 用于处理令牌混合
        self.token_mixer = EfficientFormerSelfAttention(
            dim=config.dim,
            key_dim=config.key_dim,
            num_heads=config.num_attention_heads,
            attention_ratio=config.attention_ratio,
            resolution=config.resolution,
        )

        # 创建 LayerNorm 层，layernorm1 用于对输入进行归一化
        self.layernorm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)

        # 计算 MLP 隐藏层的维度
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        # 创建 EfficientFormerDenseMlp 对象，mlp 用于多层感知机的计算
        self.mlp = EfficientFormerDenseMlp(config, in_features=dim, hidden_features=mlp_hidden_dim)

        # 创建 EfficientFormerDropPath 对象，用于应用 DropPath（一种正则化方法）
        self.drop_path = EfficientFormerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # 检查是否使用层标度
        self.use_layer_scale = config.use_layer_scale
        if config.use_layer_scale:
            # 创建用于层标度的参数
            self.layer_scale_1 = nn.Parameter(config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False) -> Tuple[torch.Tensor]:
        # 使用 token_mixer 对输入的 hidden_states 进行自注意力计算，得到 self_attention_outputs
        self_attention_outputs = self.token_mixer(self.layernorm1(hidden_states), output_attentions)
        # 取 self_attention_outputs 的第一个值作为 attention_output
        attention_output = self_attention_outputs[0]
        # 存储 self_attention_outputs 中除了第一个值外的其它值到 outputs 中，如果需要输出注意力权重，则包含注意力权重
        outputs = self_attention_outputs[1:]

        # 如果使用层标度
        if self.use_layer_scale:
            # 计算 layer_output，并应用 drop_path 和层标度
            layer_output = hidden_states + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0) * attention_output
            )
            # 再次计算 layer_output，并应用 drop_path 和层标度，同时使用 mlp 对结果进行计算
            layer_output = layer_output + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(self.layernorm2(layer_output))
            )
        else:
            # 计算 layer_output，并应用 drop_path
            layer_output = hidden_states + self.drop_path(attention_output)
            # 再次计算 layer_output，并应用 drop_path，同时使用 mlp 对结果进行计算
            layer_output = layer_output + self.drop_path(self.mlp(self.layernorm2(layer_output)))

        # 将 layer_output 添加到 outputs 中
        outputs = (layer_output,) + outputs

        # 返回 outputs
        return outputs
class EfficientFormerMeta3DLayers(nn.Module):
    # EfficientFormerMeta3DLayers 类的初始化函数
    def __init__(self, config: EfficientFormerConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 计算每个块的丢弃路径
        drop_paths = [
            config.drop_path_rate * (block_idx + sum(config.depths[:-1]))
            for block_idx in range(config.num_meta3d_blocks)
        ]
        # 创建一个包含多个 EfficientFormerMeta3D 实例的模块列表
        self.blocks = nn.ModuleList(
            [EfficientFormerMeta3D(config, config.hidden_sizes[-1], drop_path=drop_path) for drop_path in drop_paths]
        )

    # EfficientFormerMeta3DLayers 类的前向传播函数
    def forward(self, hidden_states: torch.Tensor, output_attentions: bool = False) -> Tuple[torch.Tensor]:
        # 如果要输出注意力权重，则初始化一个空元组用于存储所有的注意力权重
        all_attention_outputs = () if output_attentions else None

        # 遍历每个块
        for layer_module in self.blocks:
            # 如果 hidden_states 是元组，则取第一个元素
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            # 调用当前块的前向传播函数
            hidden_states = layer_module(hidden_states, output_attentions)

            # 如果要输出注意力权重，则将当前块的注意力权重加入到 all_attention_outputs 中
            if output_attentions:
                all_attention_outputs = all_attention_outputs + (hidden_states[1],)

        # 如果要输出注意力权重，则返回隐藏状态和所有注意力权重
        if output_attentions:
            outputs = (hidden_states[0],) + all_attention_outputs
            return outputs

        # 否则，只返回隐藏状态
        return hidden_states


class EfficientFormerMeta4D(nn.Module):
    # EfficientFormerMeta4D 类的初始化函数
    def __init__(self, config: EfficientFormerConfig, dim: int, drop_path: float = 0.0):
        # 调用父类的初始化函数
        super().__init__()
        # 如果未指定池化大小，则使用默认值 3
        pool_size = config.pool_size if config.pool_size is not None else 3
        # 创建一个 EfficientFormerPooling 实例用于 token 混合
        self.token_mixer = EfficientFormerPooling(pool_size=pool_size)
        # 计算 MLP 隐藏层维度
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        # 创建一个 EfficientFormerConvMlp 实例用于 MLP 操作
        self.mlp = EfficientFormerConvMlp(
            config, in_features=dim, hidden_features=mlp_hidden_dim, drop=config.hidden_dropout_prob
        )
        # 如果丢弃路径大于 0，则创建一个 EfficientFormerDropPath 实例，否则使用恒等映射
        self.drop_path = EfficientFormerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # 是否使用层缩放
        self.use_layer_scale = config.use_layer_scale
        # 如果使用层缩放，则创建两个可训练参数来进行缩放
        if config.use_layer_scale:
            self.layer_scale_1 = nn.Parameter(config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(config.layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    # EfficientFormerMeta4D 类的前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        # 对输入进行 token 混合
        outputs = self.token_mixer(hidden_states)

        # 如果使用层缩放
        if self.use_layer_scale:
            # 第一次层缩放和丢弃路径后的输出
            layer_output = hidden_states + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * outputs)
            # 第二次层缩放和丢弃路径后的输出
            layer_output = layer_output + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(layer_output)
            )
        else:
            # 使用丢弃路径后的输出
            layer_output = hidden_states + self.drop_path(outputs)
            # 使用 MLP 输出再次进行丢弃路径后的输出
            layer_output = layer_output + self.drop_path(self.mlp(layer_output))

        # 返回层输出
        return layer_output


class EfficientFormerMeta4DLayers(nn.Module):
    # EfficientFormerMeta4DLayers 类的初始化函数
    def __init__(self, config: EfficientFormerConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 省略了这个类的初始化代码，因为在提供的代码中缺失
    # 初始化方法，接收配置和阶段索引作为参数
    def __init__(self, config: EfficientFormerConfig, stage_idx: int):
        # 调用父类的初始化方法
        super().__init__()
        # 计算当前阶段的层数
        num_layers = (
            config.depths[stage_idx] if stage_idx != -1 else config.depths[stage_idx] - config.num_meta3d_blocks
        )
        # 计算每个 block 的 drop_path
        drop_paths = [
            config.drop_path_rate * (block_idx + sum(config.depths[:stage_idx])) for block_idx in range(num_layers)
        ]

        # 创建包含多个 EfficientFormerMeta4D 对象的模块列表
        self.blocks = nn.ModuleList(
            [
                EfficientFormerMeta4D(config, config.hidden_sizes[stage_idx], drop_path=drop_path)
                for drop_path in drop_paths
            ]
        )

    # 前向传播方法，接收隐藏状态的张量作为参数，返回多个张量组成的元组
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        # 遍历每个 block 模块进行前向传播
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states)
        # 返回更新后的隐藏状态
        return hidden_states
# EfficientFormerEncoder 类: 实现 EfficientFormerEncoder 模块，用于编码输入的隐藏状态
class EfficientFormerEncoder(nn.Module):
    # 初始化函数，接收 EfficientFormerConfig 类实例作为配置参数
    def __init__(self, config: EfficientFormerConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 将配置参数保存到类中
        self.config = config
        # 计算中间阶段的数量，即 depths 列表长度减 1
        num_intermediate_stages = len(config.depths) - 1
        # 计算 downsamples 列表，用于确定是否需要下采样
        downsamples = [
            config.downsamples[i] or config.hidden_sizes[i] != config.hidden_sizes[i + 1]
            for i in range(num_intermediate_stages)
        ]
        # 创建空的 intermediate_stages 列表
        intermediate_stages = []

        # 循环创建中间阶段的模块
        for i in range(num_intermediate_stages):
            # 将 EfficientFormerIntermediateStage 实例添加到 intermediate_stages 列表中
            intermediate_stages.append(EfficientFormerIntermediateStage(config, i))
            # 如果当前阶段需要下采样，则将 EfficientFormerPatchEmbeddings 实例添加到 intermediate_stages 列表中
            if downsamples[i]:
                intermediate_stages.append(
                    EfficientFormerPatchEmbeddings(config, config.hidden_sizes[i], config.hidden_sizes[i + 1])
                )

        # 将 intermediate_stages 列表转换为 ModuleList 类实例，保存到类中
        self.intermediate_stages = nn.ModuleList(intermediate_stages)
        # 创建 EfficientFormerLastStage 实例，保存到类中
        self.last_stage = EfficientFormerLastStage(config)

    # 前向传播函数，接收输入的隐藏状态等参数，返回编码后的隐藏状态
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
        ):
        # 循环遍历中间阶段的模块，并依次对输入的隐藏状态进行编码
        for stage in self.intermediate_stages:
            hidden_states = stage(hidden_states)
        # 对最后一个阶段的模块进行编码
        hidden_states = self.last_stage(hidden_states)

        # 返回编码后的隐藏状态
        return hidden_states
        ) -> BaseModelOutput:
        # 初始化变量 all_hidden_states 和 all_self_attentions，根据输出设置是否为空元组
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 如果输出隐藏层状态，则将当前隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 遍历中间层，并对隐藏状态进行处理
        for layer_module in self.intermediate_stages:
            hidden_states = layer_module(hidden_states)
            # 如果输出隐藏层状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # 处理最后一层，并获取输出结果
        layer_output = self.last_stage(hidden_states, output_attentions=output_attentions)

        # 如果输出注意力权重，则将当前层的注意力权重添加到 all_self_attentions 中
        if output_attentions:
            all_self_attentions = all_self_attentions + layer_output[1:]

        # 如果输出隐藏层状态，则将当前层的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (layer_output[0],)

        # 如果不返回字典形式的结果，则返回元组
        if not return_dict:
            return tuple(v for v in [layer_output[0], all_hidden_states, all_self_attentions] if v is not None)
        
        # 返回字典形式的结果
        return BaseModelOutput(
            last_hidden_state=layer_output[0],
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
```  
class EfficientFormerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 EfficientFormerConfig
    config_class = EfficientFormerConfig
    # 基础模型前缀为 "efficientformer"
    base_model_prefix = "efficientformer"
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 不支持梯度检查点
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module):
        """Initialize the weights"""
        # 如果模块为线性层或二维卷积层，使用正态分布初始化权重
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置，初始化为 0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块为 LayerNormalization 层，初始化偏置为 0，权重为 1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


EFFICIENTFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [nn.Module](https://pytorch.org/docs/stable/nn.html#nn.Module) subclass. Use it as a
    regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

    Parameters:
        config ([`EfficientFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

EFFICIENTFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`ViTImageProcessor`]. See
            [`ViTImageProcessor.preprocess`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare EfficientFormer Model transformer outputting raw hidden-states without any specific head on top.",
    EFFICIENTFORMER_START_DOCSTRING,
)
class EfficientFormerModel(EfficientFormerPreTrainedModel):
    def __init__(self, config: EfficientFormerConfig):
        super().__init__(config)
        self.config = config

        # 初始化 patch_embed
        self.patch_embed = EfficientFormerConvStem(config, config.hidden_sizes[0])
        # 初始化 encoder
        self.encoder = EfficientFormerEncoder(config)
        # 初始化 layernorm
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(EFFICIENTFORMER_INPUTS_DOCSTRING)
    # 使用 add_code_sample_docstrings 装饰器添加文档字符串，包括模型 checkpoint、输出类型、配置类、模态、预期输出形状等信息
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 定义前向传播方法，接受像素值张量、是否输出注意力、是否输出隐藏状态、是否返回字典等参数
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutput]:
        # 如果未提供像素值张量，抛出数值错误
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 使用 patch_embed 方法对像素值张量进行处理得到嵌入输出
        embedding_output = self.patch_embed(pixel_values)
        # 将嵌入输出传入编码器得到编码器输出
        encoder_outputs = self.encoder(
            embedding_output, output_attentions=output_attentions, output_hidden_states=output_hidden_states
        )

        # 获取编码器输出的序列部分
        sequence_output = encoder_outputs[0]
        # 对序列部分进行 layernorm 处理
        sequence_output = self.layernorm(sequence_output)

        # 如果不需要返回字典形式的输出
        if not return_dict:
            # 将序列部分作为头部输出，加上编码器的其他输出（隐藏状态和注意力）
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # 如果需要返回字典形式的输出，构造 BaseModelOutput 对象
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 在 EfficientFormer 模型的基础上，添加一个图片分类头部（即在 [CLS] 标记的最终隐藏状态上面的一个线性层）来进行图像分类
class EfficientFormerForImageClassification(EfficientFormerPreTrainedModel):
    def __init__(self, config: EfficientFormerConfig):
        # 继承 EfficientFormerPreTrainedModel 的初始化
        super().__init__(config)

        # 获取标签数量
        self.num_labels = config.num_labels
        # 创建 EfficientFormer 模型
        self.efficientformer = EfficientFormerModel(config)

        # 分类器头部
        # 如果标签数量大于 0，创建线性层；否则创建 Identity 层
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 模型前向传播
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 设置返回字典，如果未提供则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 EfficientFormer 进行前向传播
        outputs = self.efficientformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取序列输出
        sequence_output = outputs[0]

        # 通过分类器获取 logits
        logits = self.classifier(sequence_output.mean(-2))

        # 初始化损失为 None
        loss = None
        # 如果存在标签
        if labels is not None:
            # 如果问题类型未指定
            if self.config.problem_type is None:
                # 根据标签数量确定问题类型
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

        # 如果不要求返回字典，则将输出组装成元组返回
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 ImageClassifierOutput 类的实例，包括损失、logits、隐藏状态和注意力权重
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 dataclass 装饰器定义类 EfficientFormerForImageClassificationWithTeacherOutput，继承自 ModelOutput
@dataclass
class EfficientFormerForImageClassificationWithTeacherOutput(ModelOutput):
    """
    Output type of [`EfficientFormerForImageClassificationWithTeacher`].

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores as the average of the cls_logits and distillation logits.
        cls_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
            class token).
        distillation_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
            distillation token).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: torch.FloatTensor = None
    cls_logits: torch.FloatTensor = None
    distillation_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 使用 add_start_docstrings 装饰器为 EfficientFormerForImageClassificationWithTeacher 类添加文档字符串
@add_start_docstrings(
    """
    EfficientFormer Model transformer with image classification heads on top (a linear layer on top of the final hidden
    state of the [CLS] token and a linear layer on top of the final hidden state of the distillation token) e.g. for
    ImageNet.

    <Tip warning={true}>

           This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
           supported.

    </Tip>
    """,
    EFFICIENTFORMER_START_DOCSTRING,
)
# 定义类 EfficientFormerForImageClassificationWithTeacher，继承自 EfficientFormerPreTrainedModel
class EfficientFormerForImageClassificationWithTeacher(EfficientFormerPreTrainedModel):
    # 初始化方法，接受一个 EfficientFormerConfig 对象作为配置参数
    def __init__(self, config: EfficientFormerConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 保存配置中的标签数量
        self.num_labels = config.num_labels
        # 创建 EfficientFormerModel 模型
        self.efficientformer = EfficientFormerModel(config)

        # 分类器头部，根据配置中的标签数量决定是创建线性层还是 Identity 层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        # 蒸馏头，根据配置中的标签数量决定是创建线性层还是 Identity 层
        self.distillation_classifier = (
            nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法，接受输入像素值、是否输出注意力、是否输出隐藏状态、是否返回字典类型的参数，返回模型输出
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
        # 如果返回字典类型参数为 None，则使用配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 调用 EfficientFormerModel 的前向传播方法，获取模型输出
        outputs = self.efficientformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的序列部分
        sequence_output = outputs[0]

        # 使用分类器对序列输出求均值得到分类预测
        cls_logits = self.classifier(sequence_output.mean(-2))
        # 使用蒸馏分类器对序列输出求均值得到蒸馏预测
        distillation_logits = self.distillation_classifier(sequence_output.mean(-2))

        # 在推理过程中，返回两个分类器预测的平均值
        logits = (cls_logits + distillation_logits) / 2

        # 如果不返回字典类型参数
        if not return_dict:
            # 返回模型输出
            output = (logits, cls_logits, distillation_logits) + outputs[1:]
            return output

        # 返回 EfficientFormerForImageClassificationWithTeacherOutput 类型的模型输出
        return EfficientFormerForImageClassificationWithTeacherOutput(
            logits=logits,
            cls_logits=cls_logits,
            distillation_logits=distillation_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```