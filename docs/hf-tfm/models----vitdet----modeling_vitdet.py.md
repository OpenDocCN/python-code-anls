# `.\transformers\models\vitdet\modeling_vitdet.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 版权所有 2023 年 Meta AI 和 The HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，
# 没有任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言的权限和限制。
""" PyTorch ViTDet backbone."""

# 导入所需的库
import collections.abc
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# 导入自定义的模块
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import BackboneMixin
from .configuration_vitdet import VitDetConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 通用文档字符串
_CONFIG_FOR_DOC = "VitDetConfig"

# 预训练模型存档列表
VITDET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/vit-det-base",
    # 查看所有 ViTDet 模型 https://huggingface.co/models?filter=vitdet
]

# 定义 VitDetEmbeddings 类
class VitDetEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) to be consumed by a Transformer.
    """

    def __init__(self, config):
        super().__init__()
        # 从配置中获取参数
        image_size, patch_size = config.pretrain_image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 处理图像大小和补丁大小
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        if config.use_absolute_position_embeddings:
            # 如果使用绝对位置嵌入，则初始化绝对位置嵌入
            num_positions = num_patches + 1
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_positions, config.hidden_size))
        else:
            self.position_embeddings = None

        # 使用卷积层进行投影
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
        # 如果有 cls token，则移除第一个 embedding
        if has_cls_token:
            abs_pos_embeddings = abs_pos_embeddings[:, 1:]
        # 获取绝对位置 embedding 的数量
        num_position = abs_pos_embeddings.shape[1]
        # 计算 size，即 num_position 的平方根
        size = int(math.sqrt(num_position))
        # 如果 num_position 不是完全平方数，则抛出异常
        if size * size != num_position:
            raise ValueError("Absolute position embeddings must be a square number.")

        # 如果 size 不等于 height 或 width，则进行插值操作
        if size != height or size != width:
            # 将绝对位置 embedding 转换为 (1, size, size, -1) 的形状，然后进行插值
            new_abs_pos_embeddings = nn.functional.interpolate(
                abs_pos_embeddings.reshape(1, size, size, -1).permute(0, 3, 1, 2),
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            )

            # 将插值后的结果转换为 (1, height, width, num_channels) 的形状
            return new_abs_pos_embeddings.permute(0, 2, 3, 1)
        else:
            # 如果 size 等于 height 和 width，则直接返回原始形状
            return abs_pos_embeddings.reshape(1, height, width, -1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 获取输入张量的通道数
        num_channels = pixel_values.shape[1]
        # 如果通道数不等于配置中设置的通道数，则抛出异常
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        # 对输入张量进行投影
        embeddings = self.projection(pixel_values)

        # 如果存在位置 embedding
        if self.position_embeddings is not None:
            # 将张量形状从 (batch_size, num_channels, height, width) 转换为 (batch_size, height, width, num_channels)
            embeddings = embeddings.permute(0, 2, 3, 1)
            # 添加位置 embedding
            embeddings = embeddings + self.get_absolute_positions(
                self.position_embeddings, True, embeddings.shape[1], embeddings.shape[2]
            )
            # 将张量形状从 (batch_size, height, width, num_channels) 转换为 (batch_size, num_channels, height, width)
            embeddings = embeddings.permute(0, 3, 1, 2)

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
    # Calculate the maximum relative distance based on query and key sizes
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    
    # Interpolate rel pos if needed
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate relative position embeddings to match max_rel_dist
        rel_pos_resized = nn.functional.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coordinates with short length if shapes for q and k are different
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
    queries_height, queries_width = q_size
    keys_height, keys_width = k_size
    
    # Get relative positional embeddings for height and width
    relative_height = get_rel_pos(queries_height, keys_height, rel_pos_h)
    relative_width = get_rel_pos(queries_width, keys_width, rel_pos_w)

    batch_size, _, dim = queries.shape
    r_q = queries.reshape(batch_size, queries_height, queries_width, dim)
    
    # Calculate relative height and width weights
    relative_height = torch.einsum("bhwc,hkc->bhwk", r_q, relative_height)
    relative_weight = torch.einsum("bhwc,wkc->bhwk", r_q, relative_width)
    # 将注意力矩阵进行形状变换，将其视图调整为(batch_size, queries_height, queries_width, keys_height, keys_width)
    attn = (
        attn.view(batch_size, queries_height, queries_width, keys_height, keys_width)
        # 添加相对高度信息
        + relative_height[:, :, :, :, None]
        # 添加相对权重信息
        + relative_weight[:, :, :, None, :]
    )
    # 再次将注意力矩阵进行形状变换，将其视图调整为(batch_size, queries_height * queries_width, keys_height * keys_width)
    attn = attn.view(batch_size, queries_height * queries_width, keys_height * keys_width)

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
        # 初始化 VitDetAttention 类
        super().__init__()

        dim = config.hidden_size
        num_heads = config.num_attention_heads

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # 线性变换层，用于计算 Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=config.qkv_bias)
        # 线性变换层，用于投影
        self.proj = nn.Linear(dim, dim)

        self.use_relative_position_embeddings = config.use_relative_position_embeddings
        if self.use_relative_position_embeddings:
            # 初始化相对位置编码
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, hidden_state, output_attentions=False):
        batch_size, height, width, _ = hidden_state.shape
        # 计算 Q, K, V，并重塑形状
        qkv = self.qkv(hidden_state).reshape(batch_size, height * width, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # 将 Q, K, V 拆分
        queries, keys, values = qkv.reshape(3, batch_size * self.num_heads, height * width, -1).unbind(0)

        # 计算注意力分数
        attention_scores = (queries * self.scale) @ keys.transpose(-2, -1)

        if self.use_relative_position_embeddings:
            # 添加相对位置编码
            attention_scores = add_decomposed_relative_positions(
                attention_scores, queries, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

        # 计算注意力概率
        attention_probs = attention_scores.softmax(dim=-1)

        # 计算加权后的值
        hidden_state = attention_probs @ values
        hidden_state = hidden_state.view(batch_size, self.num_heads, height, width, -1)
        hidden_state = hidden_state.permute(0, 2, 3, 1, 4)
        hidden_state = hidden_state.reshape(batch_size, height, width, -1)
        hidden_state = self.proj(hidden_state)

        if output_attentions:
            # 重塑注意力概率的形状
            attention_probs = attention_probs.reshape(
                batch_size, self.num_heads, attention_probs.shape[-2], attention_probs.shape[-1]
            )
            outputs = (hidden_state, attention_probs)
        else:
            outputs = (hidden_state,)

        return outputs


# 从 transformers.models.beit.modeling_beit.drop_path 复制而来
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    # 如果 dropout 概率为 0 或者不处于训练状态，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留概率
    keep_prob = 1 - drop_prob
    # 计算新的 shape，适用于不同维度的张量，而不仅仅是 2D ConvNets
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    # 生成随机张量，用于二值化
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 对随机张量进行二值化
    # 计算输出，通过随机张量进行 dropout 操作
    output = input.div(keep_prob) * random_tensor
    # 返回输出
    return output
# 从 transformers.models.beit.modeling_beit.BeitDropPath 复制了代码，定义了一个继承自 nn.Module 的类 VitDetDropPath
class VitDetDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        # 初始化 VitDetDropPath 类，设置 drop_prob 参数
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 在模型训练过程中，对输入的 hidden_states 进行 Stochastic Depth（随机深度）的操作
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        # 返回描述对象的额外信息，这里返回了 VitDetDropPath 类的 drop_prob 参数值
        return "p={}".format(self.drop_prob)


class VitDetLayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and variance normalization over the
    channel dimension for inputs that have shape (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        # 初始化 VitDetLayerNorm 类，设置权重和偏置参数
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # 对输入的张量 x 进行 LayerNorm 操作
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
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
        # 初始化 VitDetResBottleneckBlock 类，设置卷积层和归一化层
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = VitDetLayerNorm(bottleneck_channels)
        self.act1 = ACT2FN[config.hidden_act]

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False)
        self.norm2 = VitDetLayerNorm(bottleneck_channels)
        self.act2 = ACT2FN[config.hidden_act]

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = VitDetLayerNorm(out_channels)

    def forward(self, x):
        # 对输入的张量 x 进行前向传播
        out = x
        for layer in self.children():
            out = layer(out)

        # 将输入张量与处理后的张量相加，构成残差连接
        out = x + out
        return out


class VitDetMlp(nn.Module):
    # 初始化神经网络模型
    def __init__(self, config, in_features: int, hidden_features: int) -> None:
        # 调用父类初始化方法
        super().__init__()
        # 第一个全连接层，输入特征数为in_features，输出特征数为hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 获取激活函数类型
        self.act = ACT2FN[config.hidden_act]
        # 第二个全连接层，输入特征数为hidden_features，输出特征数为in_features
        self.fc2 = nn.Linear(hidden_features, in_features)
        # 添加Dropout层，用于防止过拟合
        self.drop = nn.Dropout(config.dropout_prob)

    # 前向传播函数
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一层全连接层
        x = self.fc1(x)
        # 使用激活函数
        x = self.act(x)
        # Dropout层
        x = self.drop(x)
        # 第二层全连接层
        x = self.fc2(x)
        # 再次应用Dropout层
        x = self.drop(x)

        # 返回结果
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
    # 获取输入 hidden_state 的维度信息
    batch_size, height, width, num_channels = hidden_state.shape

    # 计算需要进行的垂直和水平方向的填充数量
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size

    # 如果存在填充，则在 hidden_state 上进行填充
    if pad_height > 0 or pad_width > 0:
        hidden_state = nn.functional.pad(hidden_state, (0, 0, 0, pad_width, 0, pad_height))
    # 计算填充后的 patch_height 和 patch_width
    patch_height, patch_width = height + pad_height, width + pad_width

    # 将填充后的 hidden_state 视图重塑为分区后的窗口
    hidden_state = hidden_state.view(
        batch_size, patch_height // window_size, window_size, patch_width // window_size, window_size, num_channels
    )
    # 调整维度顺序以得到最终的窗口表示
    windows = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    # 返回分区后的窗口以及填充前的高度和宽度
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
    # 获取填充后的高度和宽度以及原始高度和宽度
    patch_height, patch_width = pad_height_width
    height, width = height_width
    # 计算 batch_size
    batch_size = windows.shape[0] // (patch_height * patch_width // window_size // window_size)
    # 将 windows 视图重塑为未分区的隐藏状态
    hidden_state = windows.view(
        batch_size, patch_height // window_size, patch_width // window_size, window_size, window_size, -1
    )
    # 调整维度顺序以还原原始序列
    hidden_state = hidden_state.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, patch_height, patch_width, -1)

    # 如果填充后的高度或宽度大于原始高度或宽度，则进行裁剪
    if patch_height > height or patch_width > width:
        hidden_state = hidden_state[:, :height, :width, :].contiguous()
    # 返回未分区的隐藏状态
    return hidden_state


class VitDetLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(
        self, config: VitDetConfig, drop_path_rate: float = 0, window_size: int = 0, use_residual_block: bool = False
    ):
        # 初始化 VitDetLayer 类
        super(VitDetLayer, self).__init__()
        # 配置 VitDetLayer 类的参数
        self.config = config
        self.drop_path_rate = drop_path_rate
        self.window_size = window_size
        self.use_residual_block = use_residual_block
    # 定义一个继承自nn.Module的类VitDetAttention，表示ViT-Det的Attention模块
    class VitDetAttention(nn.Module):
        # 构造函数，用于初始化实例的属性
        def __init__(self, config, input_size):
            # 调用nn.Module的构造函数，继承父类的属性
            super().__init__()
    
            # 计算隐藏状态的维度
            dim = config.hidden_size
    
            # 针对dim进行Layer Normalization
            self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
    
            # 创建Attention计算的实例，传入config和输入维度进行初始化
            self.attention = Attention(
                config,
                input_size=input_size if window_size == 0 else (window_size, window_size)
            )
    
            # 如果存在drop_path_rate非零，则创建DropPath实例进行添加随机Dropout
            self.drop_path = VitDetDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
    
            # 再次针对dim进行Layer Normalization
            self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)
    
            # 创建Mlp实例，传入config和输入维度和隐藏维度
            self.mlp = Mlp(config=config, in_features=dim, hidden_features=int(dim * config.mlp_ratio))
    
            # 设置窗口大小
            self.window_size = window_size
    
            # 设置是否使用残差块
            self.use_residual_block = use_residual_block
            if self.use_residual_block:
                # 使用含有瓶颈通道的残差块，输入输出和瓶颈通道数量都是dim // 2
                self.residual = VitDetResBottleneckBlock(
                    config=config,
                    in_channels=dim,
                    out_channels=dim,
                    bottleneck_channels=dim // 2,
                )
    
        # 前向传播函数，计算模型的输出，并返回
        def forward(
            self,
            hidden_states: torch.Tensor,  # 输入的隐藏状态，维度为(batch_size, hidden_size, height, width)
            head_mask: Optional[torch.Tensor] = None,  # 头部遮罩，用于指定哪些头部需要屏蔽
            output_attentions: bool = False,  # 是否输出Attention map
        ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
            # 将输入的隐藏状态维度转换为(batch_size, height, width, hidden_size)
            hidden_states = hidden_states.permute(0, 2, 3, 1)
    
            # 将输入的隐藏状态进行Layer Normalization
            # 确保其均值为0，方差为1
            hidden_states = self.norm1(hidden_states)
    
            # 如果窗口大小大于0，则对输入的隐藏状态进行窗口划分
            if self.window_size > 0:
                height, width = hidden_states.shape[1], hidden_states.shape[2]
                hidden_states, pad_height_width = window_partition(hidden_states, self.window_size)
    
            # 接收Attention计算的输出
            self_attention_outputs = self.attention(
                hidden_states,
                output_attentions=output_attentions,
            )
    
            # 取出Attention计算的输出中的隐藏状态
            hidden_states = self_attention_outputs[0]
    
            # 如果输出Attention权重，则也包括在输出中
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
    
            # 如果窗口大小大于0，则对隐藏状态进行窗口联合
            if self.window_size > 0:
                hidden_states = window_unpartition(hidden_states, self.window_size, pad_height_width, (height, width))
    
            # 第一次残差连接
            hidden_states = shortcut + self.drop_path(hidden_states)
    
            # 第二次残差连接
            hidden_states = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))
    
            # 将隐藏状态维度再次转换为(batch_size, hidden_size, height, width)
            hidden_states = hidden_states.permute(0, 3, 1, 2)
    
            # 如果使用残差块，则进一步对隐藏状态进行残差连接
            if self.use_residual_block:
                hidden_states = self.residual(hidden_states)
    
            # 将隐藏状态和输出组成元组作为最终的输出结果
            outputs = (hidden_states,) + outputs
    
            # 返回输出结果
            return outputs
# 创建 VitDetEncoder 模型类，继承自 nn.Module
class VitDetEncoder(nn.Module):
    def __init__(self, config: VitDetConfig) -> None:
        super().__init__()
        self.config = config
        depth = config.num_hidden_layers

        # 使用线性空间生成随机深度衰减率
        drop_path_rate = [x.item() for x in torch.linspace(0, config.drop_path_rate, depth)]

        # 创建 VitDetLayer 层，并根据配置参数进行设置
        layers = []
        for i in range(depth):
            layers.append(
                VitDetLayer(
                    config,
                    drop_path_rate=drop_path_rate[i],
                    window_size=config.window_size if i in config.window_block_indices else 0,
                    use_residual_block=i in config.residual_block_indices,
                )
            )

        self.layer = nn.ModuleList(layers)
        self.gradient_checkpointing = False

    # VitDetEncoder 的前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 遍历 VitDetLayer 层进行前向传播
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 使用梯度检查点功能（仅在训练阶段）或直接调用 VitDetLayer 层进行前向传播
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则保存最后的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不要求返回字典，则返回隐藏状态、所有隐藏状态和注意力权重
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# 使用 Caffe2 中的 "MSRAFill" 方法初始化模型权重，偏置项初始化为 0
def caffe2_msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2. Also initializes `module.bias` to 0.

    Source: https://detectron2.readthedocs.io/en/latest/_modules/fvcore/nn/weight_init.html.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


# 创建 VitDetPreTrainedModel 模型类，继承自 PreTrainedModel
class VitDetPreTrainedModel(PreTrainedModel):
    """
    Placeholder for VitDetPreTrainedModel class
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    
    # 初始化权重的基类，处理权重初始化和预训练模型的简单接口
    config_class = VitDetConfig
    # 基础模型前缀
    base_model_prefix = "vitdet"
    # 主要输入名称
    main_input_name = "pixel_values"
    # 支持渐变检查点
    supports_gradient_checkpointing = True
    # 不需要分割的模块
    _no_split_modules = []

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        # 如果模块是线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 将输入升级为`fp32`，并将其转换为所需的`dtype`，以避免在`half`模式下出现`trunc_normal_cpu`未实现的问题
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            # 如果有偏置，则初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        elif isinstance(module, VitDetEmbeddings):
            # 初始化位置嵌入
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

        elif isinstance(module, VitDetAttention) and self.config.use_relative_position_embeddings:
            # 初始化相对位置嵌入
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

        elif isinstance(module, VitDetResBottleneckBlock):
            # 针对每个层初始化权重和偏置
            for layer in [module.conv1, module.conv2, module.conv3]:
                # 使用 caffe2_msra_fill 函数填充层
                caffe2_msra_fill(layer)
            for layer in [module.norm1, module.norm2]:
                # 将权重初始化为1，偏置初始化为0
                layer.weight.data.fill_(1.0)
                layer.bias.data.zero_()
            # 最后一个规范层初���化为零
            module.norm3.weight.data.zero_()
            module.norm3.bias.data.zero_()
# VitDet 模型的起始文档字符串
VITDET_START_DOCSTRING = r"""
    该模型是一个PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类。将其用作常规的 PyTorch 模块,并参考 PyTorch 文档以了解有关一般用法和行为的所有信息。

    参数:
        config ([`VitDetConfig`]): 带有模型所有参数的模型配置类。
            通过配置文件初始化并不会加载与模型相关联的权重,只会加载配置。
            查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

# VitDet 模型输入的文档字符串  
VITDET_INPUTS_DOCSTRING = r"""
    参数:
        pixel_values (`torch.FloatTensor` 形状为 `(batch_size, num_channels, height, width)`):
            像素值。可以使用 [`AutoImageProcessor`] 获取像素值。详见 [`ViTImageProcessor.__call__`]。

        head_mask (`torch.FloatTensor` 形状为 `(num_heads,)` 或 `(num_layers, num_heads)`, *可选*):
            用于遮蔽所选注意力模块中的头部的掩码。选择 mask 值 `[0, 1]`:

            - 1 表示头部 **未被遮蔽**,
            - 0 表示头部 **被遮蔽**。

        output_attentions (`bool`, *可选*):
            是否返回所有注意力层的注意力张量。更多详情见返回张量中的 `attentions`。
        output_hidden_states (`bool`, *可选*):
            是否返回所有层的隐藏状态。更多详情见返回张量中的 `hidden_states`。
        return_dict (`bool`, *可选*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# VitDet 模型类
@add_start_docstrings(
    "The bare VitDet Transformer model outputting raw hidden-states without any specific head on top.",
    VITDET_START_DOCSTRING,
)
class VitDetModel(VitDetPreTrainedModel):
    def __init__(self, config: VitDetConfig):
        super().__init__(config)
        self.config = config

        # 创建嵌入层
        self.embeddings = VitDetEmbeddings(config)
        # 创建编码器
        self.encoder = VitDetEncoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self) -> VitDetEmbeddings:
        return self.embeddings.projection

    # 剪枝模型中的注意力头
    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        剪枝模型中的注意力头。heads_to_prune: 字典, 键为层索引, 值为该层要剪枝的头索引列表。
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(VITDET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,  # 定义输入参数pixel_values，类型为torch张量，可选
        head_mask: Optional[torch.Tensor] = None,  # 定义输入参数head_mask，类型为torch张量，可选
        output_attentions: Optional[bool] = None,  # 定义输入参数output_attentions，类型为布尔值，可选
        output_hidden_states: Optional[bool] = None,  # 定义输入参数output_hidden_states，类型为布尔值，可选
        return_dict: Optional[bool] = None,  # 定义输入参数return_dict，类型为布尔值，可选
    ) -> Union[Tuple, BaseModelOutput]:  # 返回值为一个元组或BaseModelOutput类型的联合类型
        """
        Returns:  # 返回值说明

        Examples:  # 示例说明

        ```python  # Python代码示例
        >>> from transformers import VitDetConfig, VitDetModel  # 导入所需类
        >>> import torch  # 导入torch库

        >>> config = VitDetConfig()  # 创建VitDetConfig配置对象
        >>> model = VitDetModel(config)  # 创建VitDetModel模型对象

        >>> pixel_values = torch.randn(1, 3, 224, 224)  # 创建输入的像素值张量

        >>> with torch.no_grad():  # 使用torch.no_grad()上下文管理器，不进行梯度计算
        ...     outputs = model(pixel_values)  # 将像素值输入模型得到输出结果

        >>> last_hidden_states = outputs.last_hidden_state  # 获取最后一层隐藏状态
        >>> list(last_hidden_states.shape)  # 打印最后一层隐藏状态的形状
        [1, 768, 14, 14]  # 最后一层隐藏状态的形状为[1, 768, 14, 14]
        ```py"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  # 如果output_attentions不为空则使用，否则使用self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )  # 如果output_hidden_states不为空则使用，否则使用self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 如果return_dict不为空则使用，否则使用self.config.use_return_dict

        if pixel_values is None:  # 如果pixel_values为空
            raise ValueError("You have to specify pixel_values")  # 抛出数值错误，提示需要指定pixel_values

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)  # 准备头部遮盖（如果需要），获取头部遮盖
                                                                            # head_mask中的1.0表示保留头部
                                                                            # attention_probs具有形状bsz x n_heads x N x N
                                                                            # 输入头部遮盖的形状为[num_heads]或[num_hidden_layers x num_heads]
                                                                            # head_mask转换为形状[num_hidden_layers x batch x num_heads x seq_length x seq_length]

        embedding_output = self.embeddings(pixel_values)  # 使用pixel_values生成嵌入输出

        encoder_outputs = self.encoder(  # 使用编码器模型对嵌入输出进行编码
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]  # 获取编码器输出的序列输出

        if not return_dict:  # 如果不需要返回字典
            return (sequence_output,) + encoder_outputs[1:]  # 返回包括序列输出和其他编码器输出的元组

        return BaseModelOutput(  # 返回BaseModelOutput类型的对象
            last_hidden_state=sequence_output,  # 最后一层隐藏状态为序列输出
            hidden_states=encoder_outputs.hidden_states,  # 隐藏状态为编码器输出的隐藏状态
            attentions=encoder_outputs.attentions,  # 注意力为编码器输出的注意力
        )
@add_start_docstrings(
    """
    ViTDet backbone, to be used with frameworks like Mask R-CNN.
    """,
    VITDET_START_DOCSTRING,
)
class VitDetBackbone(VitDetPreTrainedModel, BackboneMixin):
    def __init__(self, config):
        # 调用父类构造函数初始化模型
        super().__init__(config)
        # 调用父类方法初始化骨干网络
        super()._init_backbone(config)

        # 初始化图像嵌入层
        self.embeddings = VitDetEmbeddings(config)
        # 初始化编码器
        self.encoder = VitDetEncoder(config)
        # 记录每个层的特征维度
        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> VitDetEmbeddings:
        # 获取输入嵌入层
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

        ```python
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
        ```py"""
        # 如果没有指定返回字典，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果没有指定是否输出隐藏状态，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有指定是否输出注意力，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # 将输入的像素值传递给图像嵌入层
        embedding_output = self.embeddings(pixel_values)

        # 将嵌入输出传递给编码器
        outputs = self.encoder(
            embedding_output,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        # 如果返回字典，则从中获取隐藏状态；否则，从输出中获取
        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        feature_maps = ()
        # 将每个阶段的隐藏状态添加到特征映射中
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                feature_maps += (hidden_state,)

        if not return_dict:
            # 如果不返回字典，则根据输出是否包含隐藏状态构建输出
            if output_hidden_states:
                output = (feature_maps,) + outputs[1:]
            else:
                output = (feature_maps,) + outputs[2:]
            return output

        # 返回包含特征映射、隐藏状态和注意力权重的字典形式输出
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
```