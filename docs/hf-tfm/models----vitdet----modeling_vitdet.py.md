# `.\models\vitdet\modeling_vitdet.py`

```py
# coding=utf-8
# 定义了 UTF-8 编码格式

# 版权声明，版权归 Meta AI 和 The HuggingFace Inc. 团队所有
# 根据 Apache License, Version 2.0 许可，除非符合许可协议，否则禁止使用本文件
# 可以在以下网址获取许可协议的副本：http://www.apache.org/licenses/LICENSE-2.0

""" PyTorch ViTDet backbone."""
# 导入必要的库和模块
import collections.abc  # 导入 collections.abc 模块
import math  # 导入 math 模块
from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 工具
from torch import nn  # 导入 PyTorch 的 nn 模块

# 导入其他相关模块和函数
from ...activations import ACT2FN  # 从特定路径导入 ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput  # 从特定路径导入 BackboneOutput 和 BaseModelOutput 类
from ...modeling_utils import PreTrainedModel  # 从特定路径导入 PreTrainedModel 类
from ...utils import (  # 从特定路径导入多个函数和类
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import BackboneMixin  # 从特定路径导入 BackboneMixin 类
from .configuration_vitdet import VitDetConfig  # 从当前路径导入 VitDetConfig 类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# General docstring
_CONFIG_FOR_DOC = "VitDetConfig"  # 设置文档中的配置说明为 "VitDetConfig"

# 定义预训练模型的存档列表
VITDET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/vit-det-base",
    # 查看所有 ViTDet 模型的列表网址 https://huggingface.co/models?filter=vitdet
]


class VitDetEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) to be consumed by a Transformer.
    """

    def __init__(self, config):
        super().__init__()  # 调用父类的构造方法

        # 从配置中获取相关参数
        image_size, patch_size = config.pretrain_image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 处理图像大小和补丁大小的数据类型
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

        # 计算补丁的数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        # 设置类的属性
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 如果配置指定使用绝对位置嵌入，则初始化绝对位置嵌入
        if config.use_absolute_position_embeddings:
            num_positions = num_patches + 1
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_positions, config.hidden_size))
        else:
            self.position_embeddings = None

        # 图像通道到隐藏大小的投影
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
    def get_absolute_positions(self, abs_pos_embeddings, has_cls_token, height, width):
        """
        Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token dimension for the
        original embeddings.

        Args:
            abs_pos_embeddings (`torch.Tensor`):
                Absolute positional embeddings with (1, num_position, num_channels).
            has_cls_token (`bool`):
                If true, has 1 embedding in abs_pos_embeddings for cls token.
            height (`int`):
                Height of input image tokens.
            width (`int`):
                Width of input image tokens.

        Returns:
            Absolute positional embeddings after processing with shape (1, height, width, num_channels)
        """
        # If the input has cls_token, remove the first embedding dimension
        if has_cls_token:
            abs_pos_embeddings = abs_pos_embeddings[:, 1:]

        # Calculate the number of position embeddings
        num_position = abs_pos_embeddings.shape[1]

        # Determine the size of the square matrix from the number of position embeddings
        size = int(math.sqrt(num_position))
        if size * size != num_position:
            raise ValueError("Absolute position embeddings must be a square number.")

        # If the size of embeddings does not match input height or width, resize them
        if size != height or size != width:
            new_abs_pos_embeddings = nn.functional.interpolate(
                abs_pos_embeddings.reshape(1, size, size, -1).permute(0, 3, 1, 2),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )
            # Rearrange dimensions to match the expected output shape
            return new_abs_pos_embeddings.permute(0, 2, 3, 1)
        else:
            # Reshape embeddings to the expected output shape
            return abs_pos_embeddings.reshape(1, height, width, -1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Check if the number of channels in pixel_values matches the expected configuration
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )

        # Project pixel values to obtain embeddings
        embeddings = self.projection(pixel_values)

        # If position embeddings are provided, incorporate them into the embeddings
        if self.position_embeddings is not None:
            # Rearrange dimensions of embeddings to (batch_size, height, width, num_channels)
            embeddings = embeddings.permute(0, 2, 3, 1)
            
            # Add absolute positional embeddings to the embeddings
            embeddings = embeddings + self.get_absolute_positions(
                self.position_embeddings, True, embeddings.shape[1], embeddings.shape[2]
            )
            
            # Rearrange dimensions back to (batch_size, num_channels, height, width)
            embeddings = embeddings.permute(0, 3, 1, 2)

        # Return the processed embeddings
        return embeddings
def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of query and key sizes.

    Args:
        q_size (`int`):
            Size of query q.
        k_size (`int`):
            Size of key k.
        rel_pos (`torch.Tensor`):
            Relative position embeddings (num_embeddings, num_channels).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    # 计算相对位置的最大距离
    max_rel_dist = int(2 * max(q_size, k_size) - 1)

    # 如果 rel_pos 的第一个维度不等于 max_rel_dist，则进行插值
    if rel_pos.shape[0] != max_rel_dist:
        # 插值处理相对位置嵌入
        rel_pos_resized = nn.functional.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # 根据 q 和 k 的形状差异，对坐标进行缩放处理
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_relative_positions(attn, queries, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings as introduced in
    [MViT2](https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py).

    Args:
        attn (`torch.Tensor`):
            Attention map.
        queries (`torch.Tensor`):
            Query q in the attention layer with shape (batch_size, queries_height * queries_width, num_channels).
        rel_pos_h (`torch.Tensor`):
            Relative position embeddings (Lh, num_channels) for height axis.
        rel_pos_w (`torch.Tensor`):
            Relative position embeddings (Lw, num_channels) for width axis.
        q_size (`Tuple[int]`):
            Spatial sequence size of query q with (queries_height, queries_width).
        k_size (`Tuple[int]`]):
            Spatial sequence size of key k with (keys_height, keys_width).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    # 获取 queries 和 keys 的高度和宽度
    queries_height, queries_width = q_size
    keys_height, keys_width = k_size

    # 获取高度和宽度方向上的相对位置嵌入
    relative_height = get_rel_pos(queries_height, keys_height, rel_pos_h)
    relative_width = get_rel_pos(queries_width, keys_width, rel_pos_w)

    batch_size, _, dim = queries.shape
    r_q = queries.reshape(batch_size, queries_height, queries_width, dim)

    # 使用 Einstein 求和符号计算相对高度和宽度的加权值
    relative_height = torch.einsum("bhwc,hkc->bhwk", r_q, relative_height)
    relative_weight = torch.einsum("bhwc,wkc->bhwk", r_q, relative_width)
    # 将注意力矩阵重新形状为五维张量，用于计算注意力分数
    attn = (
        attn.view(batch_size, queries_height, queries_width, keys_height, keys_width)  # 将注意力矩阵重新形状为五维张量
        + relative_height[:, :, :, :, None]  # 添加相对高度信息到张量的对应维度
        + relative_weight[:, :, :, None, :]  # 添加相对权重信息到张量的对应维度
    ).view(batch_size, queries_height * queries_width, keys_height * keys_width)  # 将张量重新展平为二维形状
    
    # 返回处理后的注意力矩阵
    return attn
class VitDetAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, config, input_size=None):
        """
        Args:
            config (`VitDetConfig`):
                Model configuration.
            input_size (`Tuple[int]`, *optional*):
                Input resolution, only required in case relative position embeddings are added.
        """
        super().__init__()

        dim = config.hidden_size  # 从配置中获取隐藏层大小
        num_heads = config.num_attention_heads  # 从配置中获取注意力头的数量

        self.num_heads = num_heads  # 存储注意力头的数量
        head_dim = dim // num_heads  # 计算每个注意力头的维度
        self.scale = head_dim**-0.5  # 缩放因子，用于缩放注意力权重

        self.qkv = nn.Linear(dim, dim * 3, bias=config.qkv_bias)  # 定义线性层 qkv，用于查询、键、值的线性变换
        self.proj = nn.Linear(dim, dim)  # 定义线性层 proj，用于最终的投影

        self.use_relative_position_embeddings = config.use_relative_position_embeddings  # 是否使用相对位置编码
        if self.use_relative_position_embeddings:
            # 初始化相对位置编码的参数
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, hidden_state, output_attentions=False):
        batch_size, height, width, _ = hidden_state.shape  # 获取隐藏状态的形状信息
        # 执行 qkv 线性变换并重新排列形状以便后续处理
        qkv = self.qkv(hidden_state).reshape(batch_size, height * width, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # 拆分为查询、键、值，并重新组织形状
        queries, keys, values = qkv.reshape(3, batch_size * self.num_heads, height * width, -1).unbind(0)

        # 计算注意力分数
        attention_scores = (queries * self.scale) @ keys.transpose(-2, -1)

        if self.use_relative_position_embeddings:
            # 使用相对位置编码来调整注意力分数
            attention_scores = add_decomposed_relative_positions(
                attention_scores, queries, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

        attention_probs = attention_scores.softmax(dim=-1)  # 对注意力分数进行 softmax 操作

        hidden_state = attention_probs @ values  # 计算加权后的值
        hidden_state = hidden_state.view(batch_size, self.num_heads, height, width, -1)  # 调整形状
        hidden_state = hidden_state.permute(0, 2, 3, 1, 4)  # 重新排列维度顺序
        hidden_state = hidden_state.reshape(batch_size, height, width, -1)  # 再次调整形状
        hidden_state = self.proj(hidden_state)  # 应用最终的投影变换

        if output_attentions:
            attention_probs = attention_probs.reshape(
                batch_size, self.num_heads, attention_probs.shape[-2], attention_probs.shape[-1]
            )
            outputs = (hidden_state, attention_probs)  # 如果需要输出注意力权重，则存储在 outputs 中
        else:
            outputs = (hidden_state,)  # 否则只存储隐藏状态输出

        return outputs


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    # 如果不是训练阶段或者 drop_prob 为 0，则直接返回输入
    if not training or drop_prob == 0.0:
        return input

    keep_prob = 1.0 - drop_prob  # 计算保留的概率
    mask = torch.rand(input.shape[0], 1, 1, 1, device=input.device) < keep_prob  # 创建掩码张量
    output = input / keep_prob * mask  # 应用掩码并进行缩放
    return output  # 返回处理后的张量
    # 如果 dropout 概率为 0 或者不处于训练模式，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留概率
    keep_prob = 1 - drop_prob
    # 创建与输入张量相同形状的随机张量，用于随机保留节点
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度的张量，而不仅限于二维卷积网络
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 对随机张量进行取整操作，实现二值化
    # 计算输出张量，通过随机张量实现节点随机保留的效果
    output = input.div(keep_prob) * random_tensor
    # 返回处理后的输出张量
    return output
# Copied from transformers.models.beit.modeling_beit.BeitDropPath
# 定义一个类 VitDetDropPath，用于实现样本级别的随机深度（Drop Path），应用于残差块的主路径中
class VitDetDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob  # 初始化 drop_prob 属性

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)  # 调用 drop_path 函数进行前向传播

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)  # 返回描述对象的额外信息字符串


class VitDetLayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and variance normalization over the
    channel dimension for inputs that have shape (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 初始化可学习的权重参数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # 初始化可学习的偏置参数
        self.eps = eps  # 设置 epsilon 参数
        self.normalized_shape = (normalized_shape,)  # 记录规范化形状元组

    def forward(self, x):
        u = x.mean(1, keepdim=True)  # 计算输入张量在通道维度上的均值
        s = (x - u).pow(2).mean(1, keepdim=True)  # 计算输入张量在通道维度上的方差
        x = (x - u) / torch.sqrt(s + self.eps)  # 应用 LayerNorm 公式进行标准化
        x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 应用可学习的权重和偏置进行缩放和平移
        return x


class VitDetResBottleneckBlock(nn.Module):
    """
    The standard bottleneck residual block without the last activation layer. It contains 3 conv layers with kernels
    1x1, 3x3, 1x1.
    """

    def __init__(self, config, in_channels, out_channels, bottleneck_channels):
        """
        Args:
            config (`VitDetConfig`):
                Model configuration.
            in_channels (`int`):
                Number of input channels.
            out_channels (`int`):
                Number of output channels.
            bottleneck_channels (`int`):
                Number of output channels for the 3x3 "bottleneck" conv layers.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)  # 第一个卷积层，1x1卷积
        self.norm1 = VitDetLayerNorm(bottleneck_channels)  # 第一个 LayerNorm 层
        self.act1 = ACT2FN[config.hidden_act]  # 第一个激活函数

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False)  # 第二个卷积层，3x3卷积
        self.norm2 = VitDetLayerNorm(bottleneck_channels)  # 第二个 LayerNorm 层
        self.act2 = ACT2FN[config.hidden_act]  # 第二个激活函数

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)  # 第三个卷积层，1x1卷积
        self.norm3 = VitDetLayerNorm(out_channels)  # 第三个 LayerNorm 层

    def forward(self, x):
        out = x
        for layer in self.children():  # 遍历模块的所有子层（conv, norm, act）
            out = layer(out)  # 依次对输入应用各层操作

        out = x + out  # 残差连接
        return out


class VitDetMlp(nn.Module):
    # 初始化函数，用于初始化神经网络的结构和参数
    def __init__(self, config, in_features: int, hidden_features: int) -> None:
        # 调用父类的初始化方法，确保正确初始化
        super().__init__()
        # 第一个全连接层，输入特征数为in_features，输出特征数为hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数，根据配置文件中的隐藏层激活函数名称选择对应的激活函数
        self.act = ACT2FN[config.hidden_act]
        # 第二个全连接层，输入特征数为hidden_features，输出特征数为in_features
        self.fc2 = nn.Linear(hidden_features, in_features)
        # Dropout层，使用config中指定的丢弃概率
        self.drop = nn.Dropout(config.dropout_prob)

    # 前向传播函数，定义了数据从输入到输出的流程
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一层全连接层的前向传播，将输入x变换为隐藏特征空间
        x = self.fc1(x)
        # 应用激活函数将线性变换后的结果进行非线性映射
        x = self.act(x)
        # 对映射后的结果进行Dropout操作，以防止过拟合
        x = self.drop(x)
        # 第二层全连接层的前向传播，将隐藏特征空间映射回原始特征空间
        x = self.fc2(x)
        # 再次对映射后的结果进行Dropout操作
        x = self.drop(x)

        # 返回前向传播的结果，这里没有应用激活函数，通常用于回归问题
        return x
def window_partition(hidden_state, window_size):
    """
    Partition into non-overlapping windows with padding if needed.

    Args:
        hidden_state (`torch.Tensor`):
            Input tokens with [batch_size, height, width, num_channels].
        window_size (`int`):
            Window size.

    Returns:
        `tuple(torch.FloatTensor)` comprising various elements:
        - windows: windows after partition with [batch_size * num_windows, window_size, window_size, num_channels].
        - (patch_height, patch_width): padded height and width before partition
    """
    # 获取输入张量的维度信息
    batch_size, height, width, num_channels = hidden_state.shape

    # 计算需要填充的高度和宽度
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size

    # 如果存在高度或宽度的填充需求，则进行填充
    if pad_height > 0 or pad_width > 0:
        hidden_state = nn.functional.pad(hidden_state, (0, 0, 0, pad_width, 0, pad_height))

    # 计算填充后的图像尺寸
    patch_height, patch_width = height + pad_height, width + pad_width

    # 将填充后的张量重塑为窗口视图
    hidden_state = hidden_state.view(
        batch_size, patch_height // window_size, window_size, patch_width // window_size, window_size, num_channels
    )

    # 对窗口视图进行维度置换和连续化操作，以生成最终的窗口
    windows = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)

    # 返回窗口和填充前的高度宽度信息
    return windows, (patch_height, patch_width)


def window_unpartition(windows, window_size, pad_height_width, height_width):
    """
    Window unpartition into original sequences and removing padding.

    Args:
        windows (`torch.Tensor`):
            Input tokens with [batch_size * num_windows, window_size, window_size, num_channels].
        window_size (`int`):
            Window size.
        pad_height_width (`Tuple[int]`):
            Padded height and width (patch_height, patch_width).
        height_width (`Tuple[int]`):
            Original height and width before padding.

    Returns:
        hidden_state: unpartitioned sequences with [batch_size, height, width, num_channels].
    """
    # 获取填充前后的高度和宽度信息
    patch_height, patch_width = pad_height_width
    height, width = height_width

    # 计算批量大小
    batch_size = windows.shape[0] // (patch_height * patch_width // window_size // window_size)

    # 将窗口张量视图还原为原始序列
    hidden_state = windows.view(
        batch_size, patch_height // window_size, patch_width // window_size, window_size, window_size, -1
    )

    # 对还原后的张量进行维度置换和连续化操作，以得到最终的隐藏状态
    hidden_state = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, patch_height, patch_width, -1)

    # 如果存在填充前的高度或宽度超过原始尺寸，则进行裁剪
    if patch_height > height or patch_width > width:
        hidden_state = hidden_state[:, :height, :width, :].contiguous()

    # 返回最终的隐藏状态
    return hidden_state


class VitDetLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(
        self, config: VitDetConfig, drop_path_rate: float = 0, window_size: int = 0, use_residual_block: bool = False
    ):
        super().__init__()
        # 初始化 VIT 检测层，可以接收 VIT 检测配置、下降路径率、窗口大小和是否使用残差块作为参数
        self.config = config
        self.drop_path_rate = drop_path_rate
        self.window_size = window_size
        self.use_residual_block = use_residual_block
    ) -> None:
        super().__init__()  # 调用父类的构造函数，初始化父类的属性和方法

        dim = config.hidden_size  # 从配置中获取隐藏层的大小
        input_size = (config.image_size // config.patch_size, config.image_size // config.patch_size)  # 计算输入大小

        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)  # 初始化第一个 LayerNorm 层
        self.attention = VitDetAttention(
            config, input_size=input_size if window_size == 0 else (window_size, window_size)
        )  # 初始化 VitDetAttention 模块，根据窗口大小选择输入大小

        self.drop_path = VitDetDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()  # 初始化 DropPath 层或者 Identity 层
        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)  # 初始化第二个 LayerNorm 层
        self.mlp = VitDetMlp(config=config, in_features=dim, hidden_features=int(dim * config.mlp_ratio))  # 初始化 MLP 模块

        self.window_size = window_size  # 设置窗口大小

        self.use_residual_block = use_residual_block  # 设置是否使用残差块
        if self.use_residual_block:
            # 如果使用残差块，则初始化 VitDetResBottleneckBlock
            self.residual = VitDetResBottleneckBlock(
                config=config,
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        hidden_states = hidden_states.permute(0, 2, 3, 1)  # 调整输入张量的维度顺序

        shortcut = hidden_states  # 将输入张量保存为 shortcut 变量

        hidden_states = self.norm1(hidden_states)  # 在第一个 LayerNorm 层中归一化隐藏状态

        # 窗口分区
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, pad_height_width = window_partition(hidden_states, self.window_size)

        self_attention_outputs = self.attention(
            hidden_states,
            output_attentions=output_attentions,
        )  # 使用注意力模块处理隐藏状态

        hidden_states = self_attention_outputs[0]  # 更新隐藏状态
        outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重，则保存在 outputs 中

        # 反向窗口分区
        if self.window_size > 0:
            hidden_states = window_unpartition(hidden_states, self.window_size, pad_height_width, (height, width))

        # 第一个残差连接
        hidden_states = shortcut + self.drop_path(hidden_states)

        hidden_states = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))  # 在第二个 LayerNorm 层后应用 MLP 模块和 DropPath

        hidden_states = hidden_states.permute(0, 3, 1, 2)  # 恢复输出张量的维度顺序

        if self.use_residual_block:
            hidden_states = self.residual(hidden_states)  # 如果使用残差块，则应用残差块

        outputs = (hidden_states,) + outputs  # 将处理后的隐藏状态与可能的注意力权重输出组合成输出元组

        return outputs  # 返回输出元组
# 定义一个 VitDetEncoder 类，继承自 nn.Module
class VitDetEncoder(nn.Module):
    def __init__(self, config: VitDetConfig) -> None:
        super().__init__()
        self.config = config
        depth = config.num_hidden_layers

        # stochastic depth decay rule
        # 根据深度生成随机深度衰减率列表
        drop_path_rate = [x.item() for x in torch.linspace(0, config.drop_path_rate, depth)]

        layers = []
        # 根据深度创建 VitDetLayer 层
        for i in range(depth):
            layers.append(
                VitDetLayer(
                    config,
                    drop_path_rate=drop_path_rate[i],
                    window_size=config.window_size if i in config.window_block_indices else 0,
                    use_residual_block=i in config.residual_block_indices,
                )
            )

        # 将所有层组成 nn.ModuleList
        self.layer = nn.ModuleList(layers)
        self.gradient_checkpointing = False

    # 定义前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 初始化隐藏状态和注意力张量的存储变量
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 遍历每个层进行前向传播
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到存储变量中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用了梯度检查点并且处于训练阶段，则使用梯度检查点函数执行当前层的调用
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层进行前向传播
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力张量，则将当前层的注意力张量添加到存储变量中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到存储变量中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要以字典形式返回结果，则按需返回隐藏状态、隐藏状态列表和注意力张量列表
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则以 BaseModelOutput 类型返回结果，包含最终隐藏状态、隐藏状态列表和注意力张量列表
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# 定义函数 caffe2_msra_fill，用于初始化 module 的权重和偏置
def caffe2_msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2. Also initializes `module.bias` to 0.

    Source: https://detectron2.readthedocs.io/en/latest/_modules/fvcore/nn/weight_init.html.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # 使用 kaiming_normal_ 初始化权重，非线性函数为 relu
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    # 如果存在偏置，则初始化为常数 0
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


# 定义 VitDetPreTrainedModel 类，继承自 PreTrainedModel
class VitDetPreTrainedModel(PreTrainedModel):
    """
    Placeholder for a pre-trained model class.
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为 VitDetConfig
    config_class = VitDetConfig
    # 基础模型前缀为 "vitdet"
    base_model_prefix = "vitdet"
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要分割的模块列表为空
    _no_split_modules = []

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        # 如果模块是线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 对权重进行截断正态分布初始化，避免在 half 精度下 `trunc_normal_cpu` 未实现的问题
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            # 如果有偏置，初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置为零，权重为全1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # 如果模块是 VitDetEmbeddings 类型
        elif isinstance(module, VitDetEmbeddings):
            # 对位置嵌入进行截断正态分布初始化
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

        # 如果模块是 VitDetAttention 类型并且配置使用相对位置嵌入
        elif isinstance(module, VitDetAttention) and self.config.use_relative_position_embeddings:
            # 对相对位置编码的水平偏移和垂直偏移进行截断正态分布初始化
            module.rel_pos_h.data = nn.init.trunc_normal_(
                module.rel_pos_h.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            )
            module.rel_pos_w.data = nn.init.trunc_normal_(
                module.rel_pos_w.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            )

        # 如果模块是 VitDetResBottleneckBlock 类型
        elif isinstance(module, VitDetResBottleneckBlock):
            # 对模块内的卷积层进行 MSRA 填充
            for layer in [module.conv1, module.conv2, module.conv3]:
                caffe2_msra_fill(layer)
            # 对归一化层的权重初始化为1，偏置初始化为0
            for layer in [module.norm1, module.norm2]:
                layer.weight.data.fill_(1.0)
                layer.bias.data.zero_()
            # 最后一个归一化层初始化权重和偏置为0
            module.norm3.weight.data.zero_()
            module.norm3.bias.data.zero_()
"""
The bare VitDet Transformer model outputting raw hidden-states without any specific head on top.
This model is a PyTorch torch.nn.Module subclass. Use it as a regular PyTorch Module and refer to the PyTorch 
documentation for all matter related to general usage and behavior.

Parameters:
    config ([`VitDetConfig`]): Model configuration class with all the parameters of the model.
        Initializing with a config file does not load the weights associated with the model, only the
        configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
@add_start_docstrings(
    "The bare VitDet Transformer model outputting raw hidden-states without any specific head on top.",
    VITDET_START_DOCSTRING,
)
class VitDetModel(VitDetPreTrainedModel):
    """
    VitDetModel class represents the Vision Transformer based model for detection tasks.

    Args:
        config (VitDetConfig): The configuration object that holds all the model hyperparameters.

    Attributes:
        embeddings (VitDetEmbeddings): Instance of the embedding layer for this model.
        encoder (VitDetEncoder): Instance of the transformer encoder for this model.
        config (VitDetConfig): The configuration object that holds all the model hyperparameters.
    """
    def __init__(self, config: VitDetConfig):
        super().__init__(config)
        self.config = config

        # Initialize embeddings and encoder based on the provided configuration
        self.embeddings = VitDetEmbeddings(config)
        self.encoder = VitDetEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> VitDetEmbeddings:
        """
        Returns the input embeddings of the model.

        Returns:
            VitDetEmbeddings: The embedding layer used for input embeddings.
        """
        return self.embeddings.projection

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel

        Args:
            heads_to_prune (Dict[int, List[int]]): Dictionary mapping layer numbers to lists of head indices to prune.
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VITDET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        前向传播函数，用于模型推理阶段或者训练阶段的前向计算。

        Returns:
        返回一个元组或者BaseModelOutput对象，取决于return_dict参数。

        Examples:
        演示如何使用该forward函数进行模型推理：

        ```
        >>> from transformers import VitDetConfig, VitDetModel
        >>> import torch

        >>> config = VitDetConfig()
        >>> model = VitDetModel(config)

        >>> pixel_values = torch.randn(1, 3, 224, 224)

        >>> with torch.no_grad():
        ...     outputs = model(pixel_values)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 768, 14, 14]
        ```
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 准备头部遮罩（如果需要）
        # 在head_mask中为1.0表示保留该头部
        # attention_probs的形状为bsz x n_heads x N x N
        # 输入的head_mask形状为[num_heads]或[num_hidden_layers x num_heads]
        # head_mask被转换为形状[num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
@add_start_docstrings(
    """
    ViTDet backbone, to be used with frameworks like Mask R-CNN.
    """,
    VITDET_START_DOCSTRING,
)
class VitDetBackbone(VitDetPreTrainedModel, BackboneMixin):
    def __init__(self, config):
        super().__init__(config)
        super()._init_backbone(config)

        # 初始化嵌入层和编码器
        self.embeddings = VitDetEmbeddings(config)
        self.encoder = VitDetEncoder(config)
        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> VitDetEmbeddings:
        # 返回嵌入层的投影
        return self.embeddings.projection

    @add_start_docstrings_to_model_forward(VITDET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```
        >>> from transformers import VitDetConfig, VitDetBackbone
        >>> import torch

        >>> config = VitDetConfig()
        >>> model = VitDetBackbone(config)

        >>> pixel_values = torch.randn(1, 3, 224, 224)

        >>> with torch.no_grad():
        ...     outputs = model(pixel_values)

        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 14, 14]
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # 对输入像素值进行嵌入处理
        embedding_output = self.embeddings(pixel_values)

        # 将嵌入输出传递给编码器，获取输出
        outputs = self.encoder(
            embedding_output,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        # 根据是否返回字典决定使用隐藏状态或者元组的第二个元素作为隐藏状态
        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                feature_maps += (hidden_state,)

        if not return_dict:
            if output_hidden_states:
                output = (feature_maps,) + outputs[1:]
            else:
                output = (feature_maps,) + outputs[2:]
            return output

        # 返回 BackboneOutput 对象，包括特征图、隐藏状态和注意力信息（如果有）
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
```