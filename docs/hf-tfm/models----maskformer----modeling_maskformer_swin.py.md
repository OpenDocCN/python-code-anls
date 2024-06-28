# `.\models\maskformer\modeling_maskformer_swin.py`

```
# coding=utf-8
# 声明代码文件使用 UTF-8 编码

# 版权声明及许可协议，这里使用 Apache License 2.0
# 详细说明了使用条件，允许了如何使用和分发代码
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可协议的副本

"""MaskFormer Swin Transformer. The reason Swin Transformer is implemented here is because MaskFormer uses the hidden
states before downsampling, which is different from the default Swin Transformer."""

import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

# 导入自定义的激活函数映射表
from ...activations import ACT2FN
# 导入文件工具函数
from ...file_utils import ModelOutput
# 导入模型输出类，用于承载模型输出结果
from ...modeling_outputs import BackboneOutput
# 导入预训练模型基类
from ...modeling_utils import PreTrainedModel
# 导入 PyTorch 工具函数，如头部剪枝，网格操作等
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
# 导入支撑函数，BackboneMixin 类，支持 Swin Transformer 模型
from ...utils.backbone_utils import BackboneMixin
# 导入 MaskFormer Swin 的配置类
from .configuration_maskformer_swin import MaskFormerSwinConfig


@dataclass
# 继承自 ModelOutput 类，增加了包含隐藏状态空间维度的输出类
class MaskFormerSwinModelOutputWithPooling(ModelOutput):
    """
    Class for MaskFormerSwinModel's outputs that also contains the spatial dimensions of the hidden states.
    """
    """
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列。
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            经过平均池化操作后的最后一层隐藏状态。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含每一层的隐藏状态，形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每一层的输出隐藏状态，以及初始嵌入输出。
        hidden_states_spatial_dimensions (`tuple(tuple(int, int))`, *optional*):
            包含每个隐藏状态的空间维度元组，用于将 `hidden_states` 重塑为 `batch, channels, height, width` 的形式。
            由于填充存在，无法在 `forward` 方法之前推断它们的空间大小。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            一个元组，包含每一层的注意力权重，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_spatial_dimensions: Tuple[Tuple[int, int]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 数据类装饰器，定义了一个输出模型的基类
@dataclass
class MaskFormerSwinBaseModelOutput(ModelOutput):
    """
    SwinEncoder模型输出的类。

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            一个元组的 `torch.FloatTensor`（对应每层的输出和初始嵌入输出），
            形状为 `(batch_size, sequence_length, hidden_size)`。

            模型每一层的隐藏状态加上初始嵌入输出。
        hidden_states_spatial_dimensions (`tuple(tuple(int, int))`, *optional*):
            包含每个 `hidden_state` 的空间维度的元组，用于将 `hidden_states` 重塑为 `batch, channels, height, width`。
            由于填充，它们的空间大小在 `forward` 方法之前无法推断。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            一个元组的 `torch.FloatTensor`（每层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            经过注意力 softmax 后的注意力权重，用于在自注意力头中计算加权平均值。
    """

    last_hidden_state: torch.FloatTensor = None  # 最后一层的隐藏状态
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 每层的隐藏状态
    hidden_states_spatial_dimensions: Tuple[Tuple[int, int]] = None  # 隐藏状态的空间维度
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 注意力权重


# 从transformers.models.swin.modeling_swin.window_partition复制过来的函数
def window_partition(input_feature, window_size):
    """
    将给定输入分割为窗口。
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


# 从transformers.models.swin.modeling_swin.window_reverse复制过来的函数
def window_reverse(windows, window_size, height, width):
    """
    合并窗口以产生更高分辨率的特征。
    """
    num_channels = windows.shape[-1]
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    return windows


# 从transformers.models.swin.modeling_swin.drop_path复制过来的函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    实现丢弃路径(drop path)操作。

    Args:
        input (torch.Tensor): 输入张量。
        drop_prob (float, optional): 丢弃概率。默认为0.0。
        training (bool, optional): 是否处于训练模式。默认为False。

    Returns:
        torch.Tensor: 处理后的张量。
    """
    # 略
    # 如果 drop_prob 等于 0.0 或者不处于训练状态，直接返回输入，不进行 Drop Path 操作
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留概率
    keep_prob = 1 - drop_prob
    # 计算输出张量的形状，适用于各种维度的张量，而不仅仅是二维卷积神经网络
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    # 生成随机张量，与输入张量相同形状，用于二值化
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 对随机张量进行二值化处理
    # 计算输出，将输入张量除以保留概率，再乘以二值化后的随机张量
    output = input.div(keep_prob) * random_tensor
    # 返回处理后的输出张量
    return output
# 定义一个名为 MaskFormerSwinEmbeddings 的 PyTorch 模块，用于构建补丁和位置嵌入。
class MaskFormerSwinEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings.
    """

    # 初始化方法，接收一个 config 对象作为参数。
    def __init__(self, config):
        super().__init__()

        # 使用 MaskFormerSwinPatchEmbeddings 类创建补丁嵌入对象。
        self.patch_embeddings = MaskFormerSwinPatchEmbeddings(config)
        # 获取补丁数量
        num_patches = self.patch_embeddings.num_patches
        # 获取补丁网格大小
        self.patch_grid = self.patch_embeddings.grid_size

        # 根据配置选择是否使用绝对位置嵌入
        if config.use_absolute_embeddings:
            # 如果使用绝对位置嵌入，则创建一个全零的可学习参数张量
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        else:
            # 否则位置嵌入设为 None
            self.position_embeddings = None

        # LayerNorm 层，用于归一化嵌入向量
        self.norm = nn.LayerNorm(config.embed_dim)
        # Dropout 层，用于随机失活以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接收像素值作为输入
    def forward(self, pixel_values):
        # 使用补丁嵌入对象处理像素值，得到嵌入张量和输出维度信息
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        # 对嵌入张量进行归一化
        embeddings = self.norm(embeddings)

        # 如果位置嵌入不为 None，则将位置嵌入加到嵌入张量上
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        # 对嵌入张量进行随机失活
        embeddings = self.dropout(embeddings)

        # 返回处理后的嵌入张量和输出维度信息
        return embeddings, output_dimensions


# 从 transformers.models.swin.modeling_swin.SwinPatchEmbeddings 复制而来的类
class MaskFormerSwinPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    # 初始化方法，接收一个 config 对象作为参数
    def __init__(self, config):
        super().__init__()
        # 从配置中获取图像大小和补丁大小
        image_size, patch_size = config.image_size, config.patch_size
        # 从配置中获取通道数和嵌入维度大小
        num_channels, hidden_size = config.num_channels, config.embed_dim
        # 如果图像大小和补丁大小不是可迭代对象，则转换为元组形式
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算补丁数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        # 将初始化的图像大小、补丁大小、通道数、嵌入维度等保存为类属性
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        # 使用卷积层将输入的像素值转换为补丁嵌入的隐藏状态
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    # 可能的填充方法，用于在图像尺寸不是补丁的整数倍时进行填充
    def maybe_pad(self, pixel_values, height, width):
        # 如果宽度不是补丁大小的整数倍，则在宽度方向进行填充
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        # 如果高度不是补丁大小的整数倍，则在高度方向进行填充
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        # 返回填充后的像素值张量
        return pixel_values
    # 定义前向传播函数，接受像素值作为输入，返回嵌入向量和输出尺寸元组
    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        # 获取像素值张量的形状信息，包括通道数、高度和宽度
        _, num_channels, height, width = pixel_values.shape
        
        # 检查通道数是否与配置中设置的通道数相匹配，如果不匹配则抛出错误
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        
        # 如果需要，对输入进行填充，使其能够被 self.patch_size 整除
        pixel_values = self.maybe_pad(pixel_values, height, width)
        
        # 将像素值投影到嵌入空间
        embeddings = self.projection(pixel_values)
        
        # 获取投影后嵌入张量的形状信息，包括通道数、高度和宽度
        _, _, height, width = embeddings.shape
        
        # 计算最终输出的高度和宽度，并存储为元组
        output_dimensions = (height, width)
        
        # 将嵌入张量按第二维展平，并交换第一和第二维的顺序
        embeddings = embeddings.flatten(2).transpose(1, 2)

        # 返回处理后的嵌入向量和输出尺寸元组
        return embeddings, output_dimensions
# Copied from transformers.models.swin.modeling_swin.SwinPatchMerging
class MaskFormerSwinPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.input_resolution = input_resolution  # 保存输入特征的分辨率信息
        self.dim = dim  # 输入通道数
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 线性变换层，用于特征维度的变换
        self.norm = norm_layer(4 * dim)  # 标准化层，对输入特征进行标准化处理

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)
        return input_feature  # 可能对输入特征进行填充操作，使其尺寸符合要求

    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
        height, width = input_dimensions
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width, num_channels)  # 将输入特征重塑为四维张量
        input_feature = self.maybe_pad(input_feature, height, width)  # 可能对输入特征进行填充操作
        input_feature_0 = input_feature[:, 0::2, 0::2, :]  # 提取输入特征的子区块
        input_feature_1 = input_feature[:, 1::2, 0::2, :]  # 提取输入特征的子区块
        input_feature_2 = input_feature[:, 0::2, 1::2, :]  # 提取输入特征的子区块
        input_feature_3 = input_feature[:, 1::2, 1::2, :]  # 提取输入特征的子区块
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)  # 按最后一个维度拼接特征
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)  # 将特征重塑为三维张量

        input_feature = self.norm(input_feature)  # 对特征进行标准化处理
        input_feature = self.reduction(input_feature)  # 对特征进行线性变换

        return input_feature


# Copied from transformers.models.swin.modeling_swin.SwinDropPath with Swin->MaskFormerSwin
class MaskFormerSwinDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob  # 初始化丢弃概率

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)  # 调用外部函数 drop_path 进行随机深度丢弃操作

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)  # 返回描述实例状态的字符串
# 从 transformers.models.swin.modeling_swin.SwinSelfAttention 复制而来，修改为 MaskFormerSwinSelfAttention
class MaskFormerSwinSelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )

        # 创建相对位置偏置表的可学习参数
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        # 计算窗口内每个位置对之间的相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # 定义查询、键、值的线性变换层
        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)

        # 定义 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入张量重塑为注意力分数的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，接受隐藏状态、注意力掩码、头部掩码和是否输出注意力分数作为输入
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor]:  # 定义函数签名，指定返回类型为包含单个张量的元组
        batch_size, dim, num_channels = hidden_states.shape  # 获取隐藏状态的形状信息
        mixed_query_layer = self.query(hidden_states)  # 使用查询函数处理隐藏状态

        key_layer = self.transpose_for_scores(self.key(hidden_states))  # 使用键函数处理隐藏状态并转置
        value_layer = self.transpose_for_scores(self.value(hidden_states))  # 使用值函数处理隐藏状态并转置
        query_layer = self.transpose_for_scores(mixed_query_layer)  # 处理混合查询层并转置

        # 计算注意力分数，即查询和键的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # 对注意力分数进行缩放

        # 获取相对位置偏置并调整形状
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # 调整相对位置偏置的维度顺序
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)  # 添加相对位置偏置到注意力分数中

        if attention_mask is not None:
            # 应用预先计算好的注意力掩码（适用于MaskFormerSwinModel forward()函数的所有层）
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # 将注意力分数归一化为注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用dropout进行注意力概率的处理
        attention_probs = self.dropout(attention_probs)

        # 如果指定了头部掩码，则应用头部掩码
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)  # 使用注意力概率加权值层得到上下文层
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # 调整上下文层的维度顺序
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)  # 调整上下文层的形状

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)  # 返回模型输出

        return outputs  # 返回上下文层和注意力概率的元组
# Copied from transformers.models.swin.modeling_swin.SwinSelfOutput with Swin->MaskFormerSwin
class MaskFormerSwinSelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个全连接层，输入维度为dim，输出维度为dim
        self.dense = nn.Linear(dim, dim)
        # 创建一个dropout层，使用config中指定的dropout概率
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 对输入的hidden_states进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的结果进行dropout
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的hidden_states
        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinAttention with Swin->MaskFormerSwin
class MaskFormerSwinAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        # 创建MaskFormerSwinSelfAttention对象
        self.self = MaskFormerSwinSelfAttention(config, dim, num_heads, window_size)
        # 创建MaskFormerSwinSelfOutput对象
        self.output = MaskFormerSwinSelfOutput(config, dim)
        # 初始化一个空集合，用于存储被剪枝的注意力头索引
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 查找可剪枝的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头索引
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 执行自注意力机制，并返回self_outputs
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        # 使用self_outputs[0]和hidden_states作为输入，执行输出层操作
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力信息，则将其添加到outputs中
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出注意力信息，将其添加到outputs中
        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinIntermediate with Swin->MaskFormerSwin
class MaskFormerSwinIntermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个全连接层，输入维度为dim，输出维度为config.mlp_ratio * dim
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        # 如果config.hidden_act是字符串类型，则使用ACT2FN字典中对应的激活函数，否则直接使用config.hidden_act作为激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 定义一个方法 `forward`，接受一个名为 `hidden_states` 的张量作为输入，并返回一个张量作为输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入张量 `hidden_states` 传递给 `self.dense` 层，执行线性变换
        hidden_states = self.dense(hidden_states)
        # 将经过线性变换后的张量 `hidden_states` 应用激活函数 `self.intermediate_act_fn`
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回经过激活函数处理后的张量 `hidden_states`
        return hidden_states
# 从 transformers.models.swin.modeling_swin.SwinOutput 复制的类，将 Swin 替换为 MaskFormerSwin
class MaskFormerSwinOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个线性层，输入维度为 config.mlp_ratio * dim，输出维度为 dim
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        # 定义一个 Dropout 层，使用 config.hidden_dropout_prob 作为丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 前向传播函数，先通过线性层处理 hidden_states
        hidden_states = self.dense(hidden_states)
        # 然后对处理后的结果进行 Dropout
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class MaskFormerSwinLayer(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0):
        super().__init__()
        # 初始化 MaskFormerSwinLayer 类，设置一些初始参数
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution
        # 添加 LayerNorm 层，对输入进行归一化，eps 参数为 config.layer_norm_eps
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 定义 MaskFormerSwinAttention 层，处理注意力相关计算
        self.attention = MaskFormerSwinAttention(config, dim, num_heads, self.window_size)
        # 如果 config.drop_path_rate 大于 0.0，则添加 MaskFormerSwinDropPath 层，否则添加一个恒等映射
        self.drop_path = (
            MaskFormerSwinDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        )
        # 添加 LayerNorm 层，对输入进行归一化，eps 参数为 config.layer_norm_eps
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 定义 MaskFormerSwinIntermediate 层，处理中间过渡层的计算
        self.intermediate = MaskFormerSwinIntermediate(config, dim)
        # 定义 MaskFormerSwinOutput 层，处理最终输出层的计算
        self.output = MaskFormerSwinOutput(config, dim)

    def get_attn_mask(self, input_resolution):
        if self.shift_size > 0:
            # 如果 shift_size 大于 0，则计算用于 SW-MSA 的注意力掩码
            height, width = input_resolution
            # 创建一个全零张量作为图像掩码，维度为 (1, height, width, 1)
            img_mask = torch.zeros((1, height, width, 1))
            # 定义高度和宽度的切片区域，根据 window_size 和 shift_size 的值生成不同的切片
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            # 填充图像掩码张量的不同区域
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            # 将图像掩码切分成窗口，并展平为二维张量
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            # 计算注意力掩码，使对角线元素为 0，其余元素分别填充为 -100.0 或 0.0
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask
    # 定义一个方法用于可能的填充操作，用于保证输入张量的高度和宽度能被窗口大小整除
    def maybe_pad(self, hidden_states, height, width):
        # 计算左边和顶部需要填充的像素数，默认为0
        pad_left = pad_top = 0
        # 计算右边需要填充的像素数，确保能够被窗口大小整除
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        # 计算底部需要填充的像素数，确保能够被窗口大小整除
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        # 组装填充的数值，顺序为 (前填充高度, 后填充高度, 左填充宽度, 右填充宽度, 顶部填充高度, 底部填充高度)
        pad_values = (0, 0, pad_left, pad_right, pad_top, pad_bottom)
        # 对隐藏状态张量进行填充操作，使用给定的填充数值
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        # 返回填充后的隐藏状态张量以及填充数值，用于后续可能的反填充操作
        return hidden_states, pad_values
    def forward(self, hidden_states, input_dimensions, head_mask=None, output_attentions=False):
        # 解构输入维度元组
        height, width = input_dimensions
        # 获取隐藏状态张量的批大小、维度和通道数
        batch_size, dim, channels = hidden_states.size()
        # 保存原始隐藏状态张量
        shortcut = hidden_states

        # Layer normalization 在注意力机制之前应用于隐藏状态张量
        hidden_states = self.layernorm_before(hidden_states)
        # 将隐藏状态张量重新排列为四维张量（批大小、高度、宽度、通道）
        hidden_states = hidden_states.view(batch_size, height, width, channels)
        # 可能需要对隐藏状态张量进行填充，使其大小成为窗口大小的倍数
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        # 获取填充后张量的维度信息
        _, height_pad, width_pad, _ = hidden_states.shape
        # 如果设置了 cyclic shift
        if self.shift_size > 0:
            # 在指定维度上对隐藏状态进行循环移位
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # 将隐藏状态分割成窗口
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        # 将分割后的窗口张量重新视图为二维张量
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        # 获取注意力掩码
        attn_mask = self.get_attn_mask((height_pad, width_pad))
        # 如果存在注意力掩码，则将其转移到与隐藏状态窗口相同的设备上
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        # 执行自注意力机制
        self_attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        # 获取自注意力机制的输出
        attention_output = self_attention_outputs[0]

        # 如果需要输出注意力权重，则将其添加到输出中
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 将注意力输出视图为四维张量
        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        # 反转窗口分割，恢复到原始大小
        shifted_windows = window_reverse(
            attention_windows, self.window_size, height_pad, width_pad
        )  # B height' width' C

        # 如果设置了 cyclic shift，将注意力窗口进行反向循环移位
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        # 如果存在填充，截取注意力窗口以移除填充部分
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        # 将注意力窗口重新视图为三维张量
        attention_windows = attention_windows.view(batch_size, height * width, channels)

        # 将原始隐藏状态张量与注意力窗口加上 drop path 结果相加
        hidden_states = shortcut + self.drop_path(attention_windows)

        # 在注意力机制之后应用 layer normalization
        layer_output = self.layernorm_after(hidden_states)
        # 中间层处理
        layer_output = self.intermediate(layer_output)
        # 在隐藏状态上添加输出层结果
        layer_output = hidden_states + self.output(layer_output)

        # 将层输出添加到总体输出中
        outputs = (layer_output,) + outputs

        # 返回所有输出
        return outputs
# 基于 transformers.models.swin.modeling_swin.SwinStage.__init__ 复制而来的 MaskFormerSwinStage 类
class MaskFormerSwinStage(nn.Module):
    # 初始化函数，接收配置参数 config，特征维度 dim，输入分辨率 input_resolution，深度 depth，注意力头数目 num_heads，丢弃路径 drop_path，降采样 downsample
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        super().__init__()
        # 保存传入的配置参数
        self.config = config
        # 保存特征维度
        self.dim = dim
        # 创建包含多个 MaskFormerSwinLayer 模块的模块列表 blocks
        self.blocks = nn.ModuleList(
            [
                MaskFormerSwinLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                )
                for i in range(depth)
            ]
        )

        # 如果有降采样函数，创建降采样层 self.downsample，否则设为 None
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        # 初始时设置 pointing 属性为 False
        self.pointing = False

    # 前向传播函数，接收隐藏状态 hidden_states，输入维度 input_dimensions，头部掩码 head_mask，是否输出注意力 output_attentions，是否输出隐藏状态 output_hidden_states
    def forward(
        self, hidden_states, input_dimensions, head_mask=None, output_attentions=False, output_hidden_states=False
    ):
        # 如果需要输出隐藏状态，则初始化空的元组 all_hidden_states 用于存储所有隐藏状态
        all_hidden_states = () if output_hidden_states else None

        # 获取输入维度的高度和宽度
        height, width = input_dimensions
        # 遍历所有 blocks 中的模块
        for i, block_module in enumerate(self.blocks):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码，如果没有传入头部掩码则为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 调用当前 block_module 的前向传播函数，计算该模块的隐藏状态
            block_hidden_states = block_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前模块计算得到的隐藏状态的第一个元素
            hidden_states = block_hidden_states[0]

            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        # 如果存在降采样层 self.downsample
        if self.downsample is not None:
            # 计算降采样后的高度和宽度
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            # 计算输出维度，包括原始和降采样后的尺寸
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            # 调用降采样层的前向传播函数，对隐藏状态进行降采样处理
            hidden_states = self.downsample(hidden_states, input_dimensions)
        else:
            # 如果不存在降采样层，则输出维度与输入维度相同
            output_dimensions = (height, width, height, width)

        # 返回最终的隐藏状态、输出维度以及所有的隐藏状态（如果需要输出）
        return hidden_states, output_dimensions, all_hidden_states


# 基于 transformers.models.swin.modeling_swin.SwinEncoder.__init__ 复制而来的 MaskFormerSwinEncoder 类
class MaskFormerSwinEncoder(nn.Module):
    pass  # 这里暂时没有任何代码，仅为占位符，具体实现可能会在后续添加
    # 初始化函数，接受配置和网格大小作为参数
    def __init__(self, config, grid_size):
        # 调用父类的初始化方法
        super().__init__()
        # 计算网络层数
        self.num_layers = len(config.depths)
        # 保存配置信息
        self.config = config
        # 根据配置中的 drop_path_rate 参数生成一个线性空间的列表，转换为 Python 列表类型
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        # 创建一个 nn.ModuleList，包含多个 MaskFormerSwinStage 模块
        self.layers = nn.ModuleList(
            [
                MaskFormerSwinStage(
                    config=config,
                    # 计算当前层的嵌入维度
                    dim=int(config.embed_dim * 2**i_layer),
                    # 计算当前层的输入分辨率
                    input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                    # 当前层的深度
                    depth=config.depths[i_layer],
                    # 当前层的注意力头数
                    num_heads=config.num_heads[i_layer],
                    # 当前层的 drop_path 参数，根据当前层的深度切片生成
                    drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    # 是否进行下采样，最后一层不进行下采样
                    downsample=MaskFormerSwinPatchMerging if (i_layer < self.num_layers - 1) else None,
                )
                for i_layer in range(self.num_layers)  # 循环创建每一层的 MaskFormerSwinStage 模块
            ]
        )

        # 梯度检查点设置为 False
        self.gradient_checkpointing = False

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        input_dimensions,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):
            # 如果不输出隐藏状态，则初始化为空元组；否则设为 None
            all_hidden_states = () if output_hidden_states else None
            # 初始化所有输入维度为空元组
            all_input_dimensions = ()
            # 如果不输出注意力，则初始化为 None
            all_self_attentions = () if output_attentions else None

            # 如果需要输出隐藏状态
            if output_hidden_states:
                # 将当前层的隐藏状态添加到 all_hidden_states 中
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 遍历所有的层，并获取每层的模块和屏蔽头掩码
            for i, layer_module in enumerate(self.layers):
                layer_head_mask = head_mask[i] if head_mask is not None else None

                # 如果启用了梯度检查点且处于训练模式
                if self.gradient_checkpointing and self.training:
                    # 使用梯度检查点函数执行当前层的调用，并获取隐藏状态、输出维度和所有隐藏状态
                    layer_hidden_states, output_dimensions, layer_all_hidden_states = self._gradient_checkpointing_func(
                        layer_module.__call__,
                        hidden_states,
                        layer_head_mask,
                        output_attentions,
                    )
                else:
                    # 否则，直接调用当前层模块，并获取隐藏状态、输出维度和所有隐藏状态
                    layer_hidden_states, output_dimensions, layer_all_hidden_states = layer_module(
                        hidden_states,
                        input_dimensions,
                        layer_head_mask,
                        output_attentions,
                        output_hidden_states,
                    )

                # 更新输入维度为当前输出维度的最后两个维度
                input_dimensions = (output_dimensions[-2], output_dimensions[-1])
                # 将当前输入维度添加到 all_input_dimensions 中
                all_input_dimensions += (input_dimensions,)
                # 如果需要输出隐藏状态，则将当前层的所有隐藏状态添加到 all_hidden_states 中
                if output_hidden_states:
                    all_hidden_states += (layer_all_hidden_states,)

                # 更新隐藏状态为当前层的隐藏状态
                hidden_states = layer_hidden_states

                # 如果需要输出注意力，则将当前层的第二个隐藏状态添加到 all_self_attentions 中
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_all_hidden_states[1],)

            # 如果不返回字典，则返回所有非空的结果元组
            if not return_dict:
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

            # 否则，返回 MaskFormerSwinBaseModelOutput 对象，包含最后的隐藏状态、所有隐藏状态、空间维度和注意力
            return MaskFormerSwinBaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                hidden_states_spatial_dimensions=all_input_dimensions,
                attentions=all_self_attentions,
            )
# 从 transformers.models.swin.modeling_swin.SwinPreTrainedModel 复制代码，修改为 MaskFormerSwinPreTrainedModel，类用于 MaskFormerSwin 模型
class MaskFormerSwinPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 MaskFormerSwinConfig 作为配置类
    config_class = MaskFormerSwinConfig
    # 基础模型的前缀为 "model"
    base_model_prefix = "model"
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 对于线性层和卷积层，使用正态分布初始化权重，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果存在偏置项，则将其初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对于 LayerNorm 层，将偏置项初始化为零，权重初始化为 1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class MaskFormerSwinModel(MaskFormerSwinPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        # 初始化 MaskFormerSwin 模型的嵌入层
        self.embeddings = MaskFormerSwinEmbeddings(config)
        # 初始化 MaskFormerSwin 模型的编码器
        self.encoder = MaskFormerSwinEncoder(config, self.embeddings.patch_grid)

        # 初始化层归一化层，输入特征数为 self.num_features，epsilon 为 config.layer_norm_eps
        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        # 如果设置了 add_pooling_layer 为 True，则初始化自适应平均池化层
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

    def get_input_embeddings(self):
        # 返回输入嵌入层的 patch_embeddings
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历要修剪的头部 heads_to_prune 字典
        for layer, heads in heads_to_prune.items():
            # 在编码器的每一层中修剪指定的注意力头部
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):
        # 设置输出注意力矩阵选项，默认使用配置文件中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态选项，默认使用配置文件中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典的选项，默认使用配置文件中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供像素值，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 准备头部掩码（如果需要）
        # 在 head_mask 中为 1.0 表示保留对应的注意力头
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 形状为 [num_heads] 或者 [num_hidden_layers x num_heads]
        # 将 head_mask 转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))

        # 对像素值进行嵌入操作
        embedding_output, input_dimensions = self.embeddings(pixel_values)

        # 编码器处理阶段
        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果 return_dict 为 True，则使用字典返回
        sequence_output = encoder_outputs.last_hidden_state if return_dict else encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        # 如果存在池化器，则进行池化操作
        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)

        # 如果 return_dict 为 False，则返回元组形式的输出
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 计算隐藏状态的空间维度
        hidden_states_spatial_dimensions = (input_dimensions,) + encoder_outputs.hidden_states_spatial_dimensions

        # 使用 MaskFormerSwinModelOutputWithPooling 类封装返回结果
        return MaskFormerSwinModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            hidden_states_spatial_dimensions=hidden_states_spatial_dimensions,
            attentions=encoder_outputs.attentions,
        )
        # MaskFormerSwinBackbone 类定义，继承自 MaskFormerSwinPreTrainedModel 和 BackboneMixin
        """
        MaskFormerSwin backbone, designed especially for the MaskFormer framework.

        This classes reshapes `hidden_states` from (`batch_size, sequence_length, hidden_size)` to (`batch_size,
        num_channels, height, width)`). It also adds additional layernorms after each stage.

        Args:
            config (`MaskFormerSwinConfig`):
                The configuration used by [`MaskFormerSwinModel`].
        """
        # 初始化方法，接收 MaskFormerSwinConfig 类型的参数 config
        def __init__(self, config: MaskFormerSwinConfig):
            # 调用父类 MaskFormerSwinPreTrainedModel 的初始化方法
            super().__init__(config)
            # 调用父类 BackboneMixin 的初始化方法
            super()._init_backbone(config)

            # 创建 MaskFormerSwinModel 的实例，并赋值给 self.model
            self.model = MaskFormerSwinModel(config)
            # 检查是否在 out_features 中包含 'stem'，若包含则抛出 ValueError 异常
            if "stem" in self.out_features:
                raise ValueError("This backbone does not support 'stem' in the `out_features`.")
            
            # 计算特征图的通道数列表，根据 config 中的 embed_dim 和 depths 参数计算
            self.num_features = [config.embed_dim] + [int(config.embed_dim * 2**i) for i in range(len(config.depths))]
            
            # 创建包含各层规范化操作的 nn.ModuleList，每层规范化操作的输入通道数对应 num_features 中的后续元素
            self.hidden_states_norms = nn.ModuleList(
                [nn.LayerNorm(num_channels) for num_channels in self.num_features[1:]]
            )

            # 调用 post_init 方法进行权重初始化和最终处理
            self.post_init()

        # 前向传播方法，接收输入 pixel_values 和可选的输出控制参数
        def forward(
            self,
            pixel_values: Tensor,
            output_hidden_states: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> BackboneOutput:
        # 确定是否返回字典类型的结果，若未指定则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 确定是否输出隐藏状态，若未指定则使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否输出注意力权重，若未指定则使用配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # 使用模型进行前向传播，指定输出隐藏状态和注意力权重，并以字典类型返回结果
        outputs = self.model(
            pixel_values, output_hidden_states=True, output_attentions=output_attentions, return_dict=True
        )

        # 跳过模型的stem部分，即第一个隐藏状态
        hidden_states = outputs.hidden_states[1:]

        # 将隐藏状态重塑回原始的空间维度
        # 空间维度包含每个阶段的所有高度和宽度，包括嵌入后的维度
        spatial_dimensions: Tuple[Tuple[int, int]] = outputs.hidden_states_spatial_dimensions
        feature_maps = ()
        for i, (hidden_state, stage, (height, width)) in enumerate(
            zip(hidden_states, self.stage_names[1:], spatial_dimensions)
        ):
            norm = self.hidden_states_norms[i]
            # 获取经过最后一个块输出但未经过补丁合并的隐藏状态
            hidden_state_unpolled = hidden_state[-1]
            # 对隐藏状态进行归一化处理
            hidden_state_norm = norm(hidden_state_unpolled)
            # 像素解码器（FPN）需要3D张量（特征）
            batch_size, _, hidden_size = hidden_state_norm.shape
            # 重塑张量形状为 "b (h w) d -> b d h w"
            hidden_state_permuted = (
                hidden_state_norm.permute(0, 2, 1).view((batch_size, hidden_size, height, width)).contiguous()
            )
            if stage in self.out_features:
                feature_maps += (hidden_state_permuted,)

        # 如果不返回字典类型的结果，则构造输出元组
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            if output_attentions:
                output += (outputs.attentions,)
            return output

        # 返回BackboneOutput对象，包含特征图、隐藏状态（如果输出）、注意力权重（如果输出）
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
```