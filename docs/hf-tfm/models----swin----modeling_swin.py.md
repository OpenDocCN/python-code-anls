# `.\transformers\models\swin\modeling_swin.py`

```
# 设置文件编码格式为 UTF-8
# 版权声明
# Apache License, Version 2.0 权利声明
# 如果没有符合许可证的相关法律或书面同意，则不得使用此文件
# 可以在下面网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 未经授权，按原样提供软件，无论有无担保或条件，默认情况下都按原样提供
# 请参阅特定语言的许可证以获取代办事项和限制
""" PyTorch Swin Transformer model."""


导入必要的库和模块
# 如果非必要，请不要删除这个代码注释
import collections.abc
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

从相关库和模块中导入一些特定函数
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import BackboneMixin
from .configuration_swin import SwinConfig

导入日志记录模块
logger = logging.get_logger(__name__)

# 通用文档字符串
_CONFIG_FOR_DOC = "SwinConfig"

# 基础文档字符串
_CHECKPOINT_FOR_DOC = "microsoft/swin-tiny-patch4-window7-224"
_EXPECTED_OUTPUT_SHAPE = [1, 49, 768]

# 图像分类文档字符串
_IMAGE_CLASS_CHECKPOINT = "microsoft/swin-tiny-patch4-window7-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"


SWIN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/swin-tiny-patch4-window7-224",
    # See all Swin models at https://huggingface.co/models?filter=swin
]

# drop_path, SwinPatchEmbeddings, SwinPatchMerging and SwinDropPath are from the timm library.


@dataclass
class SwinEncoderOutput(ModelOutput):
    """
    Swin encoder's outputs, with potential hidden states and attentions.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            # 这是一个参数说明，表示最后一层模型输出的隐藏状态序列，其形状为(batch_size, sequence_length, hidden_size)。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            # 这是一个参数说明，表示隐藏状态的元组，包括嵌入层的输出和每个阶段的输出，其形状为(batch_size, sequence_length, hidden_size)。
        
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            # 模型在每一层输出的隐藏状态，以及初始嵌入输出。

        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            # 这是一个参数说明，表示注意力权重的元组，包括每个阶段的注意力权重，其形状为(batch_size, num_heads, sequence_length, sequence_length)。
            
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            # 经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。

        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            # 这是一个参数说明，表示重塑后的隐藏状态的元组，包括每个阶段的隐藏状态和初始嵌入输出的重塑，其形状为(batch_size, hidden_size, height, width)。
            
            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
            # 模型在每一层输出的隐藏状态，以及初始嵌入输出的重塑，包括空间维度。

    """

    last_hidden_state: torch.FloatTensor = None
    # 最后的隐藏状态，类型为 torch.FloatTensor，初始值为 None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 隐藏状态，类型为 torch.FloatTensor 的元组，可选的，初始值为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重，类型为 torch.FloatTensor 的元组，可选的，初始值为 None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 重塑后的隐藏状态，类型为 torch.FloatTensor 的元组，可选的，初始值为 None
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.modeling_outputs import ModelOutput

# Swin 模型输出的数据类，包含最后隐藏状态的汇聚
@dataclass
class SwinModelOutput(ModelOutput):
    """
    Swin model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed):
            Average pooling of the last layer hidden-state.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """

    # 最后的隐藏状态
    last_hidden_state: torch.FloatTensor = None
    # 汇聚层的输出
    pooler_output: Optional[torch.FloatTensor] = None
    # 隐藏状态，每个层的隐藏状态组成的元组
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重，每个阶段的注意力权重组成的元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 重塑后的隐藏状态，每个层的隐藏状态组成的元组
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SwinMaskedImageModelingOutput(ModelOutput):
    """
    Swin masked image model outputs.
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Masked image modeling (MLM) loss.  # 定义了 loss 参数，表示被遮盖的图像建模的损失，是一个可选参数，当提供了 `bool_masked_pos` 时返回
        reconstruction (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Reconstructed pixel values.  # 重建后的像素数值，包含了 batch_size、通道数、高度和宽度的形状信息
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.  # 隐藏状态的元组，当传递了 `output_hidden_states=True` 或者 `config.output_hidden_states=True` 时返回，每个元素包含了嵌入输出和每个阶段的输出
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.  # 注意力权重的元组，当传递了 `output_attentions=True` 或者 `config.output_attentions=True` 时返回，每个元素包含了每个阶段的注意力权重
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.  # 调整形状后的隐藏状态元组，当传递了 `output_hidden_states=True` 或者 `config.output_hidden_states=True` 时返回，每个元素包含了每个层的输出和形状维度
    """

    loss: Optional[torch.FloatTensor] = None  # 初始化 loss 为 None
    reconstruction: torch.FloatTensor = None  # 初始化 reconstruction 为 None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 初始化 hidden_states 为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 初始化 attentions 为 None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 初始化 reshaped_hidden_states 为 None

    @property
    def logits(self):
        warnings.warn(
            "logits attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the reconstruction attribute to retrieve the final output instead.",
            FutureWarning,
        )
        return self.reconstruction  # 返回 reconstruction 属性
@dataclass
class SwinImageClassifierOutput(ModelOutput):
    """
    Swin outputs for image classification.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.

    Args:
        input_feature (`torch.Tensor`): Input feature tensor of shape `(batch_size, height, width, num_channels)`.
        window_size (int): Size of the window for partitioning.

    Returns:
        `torch.Tensor`: Partitioned windows tensor.
    """
    batch_size, height, width, num_channels = input_feature.shape
    # Reshape input_feature into windows of specified window_size
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    # Permute dimensions for contiguous view and reshape into final windows tensor
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.

    Args:
        windows (`torch.Tensor`): Input windows tensor of shape `(num_windows, window_size, window_size, num_channels)`.
        window_size (int): Size of the window used for partitioning.
        height (int): Height of the original image.
        width (int): Width of the original image.

    Returns:
        `torch.Tensor`: Merged tensor representing higher resolution features.
    """
    num_channels = windows.shape[-1]
    # Reshape windows tensor into intermediate form
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    # Permute dimensions for contiguous view and reshape into final higher resolution tensor
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    return windows
    # 返回 windows 变量的值
    return windows
class SwinEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        # 初始化 SwinEmbeddings 类
        super().__init__()

        # 创建 SwinPatchEmbeddings 实例
        self.patch_embeddings = SwinPatchEmbeddings(config)
        # 获取 patch 数量
        num_patches = self.patch_embeddings.num_patches
        # 获取 patch 网格大小
        self.patch_grid = self.patch_embeddings.grid_size
        # 创建或不创建 mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None

        # 创建或不创建位置 embeddings
        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        else:
            self.position_embeddings = None

        # LayerNorm 层
        self.norm = nn.LayerNorm(config.embed_dim)
        # Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, pixel_values: Optional[torch.FloatTensor], bool_masked_pos: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor]:
        # 获取 patch embeddings 和输出维度
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        # LayerNorm 层
        embeddings = self.norm(embeddings)
        # 获取 batch 大小和序列长度
        batch_size, seq_len, _ = embeddings.size()

        # 如果存在 mask token，则将其扩展成与 embeddings 相同大小的 tensor
        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # 将被 mask 的可视 token 替换为 mask token
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 如果存在位置 embeddings，则加上它们
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        # Dropout 层
        embeddings = self.dropout(embeddings)

        return embeddings, output_dimensions


class SwinPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        # 初始化 SwinPatchEmbeddings 类
        super().__init__()
        # 获取配置参数
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        # 如果 image_size 和 patch_size 不是 iterable，则转换成 iterable
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算 patch 数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        # 使用卷积来进行 projection
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
    # 如果输入的宽度不能被patch_size[1]整除，则需要填充像素值
    if width % self.patch_size[1] != 0:
        # 计算填充的值，使得宽度可以被patch_size[1]整除
        pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
        # 对输入进行填充操作
        pixel_values = nn.functional.pad(pixel_values, pad_values)
    # 如果输入的高度不能被patch_size[0]整除，则需要填充像素值
    if height % self.patch_size[0] != 0:
        # 计算填充的值，使得高度可以被patch_size[0]整除
        pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
        # 对输入进行填充操作
        pixel_values = nn.functional.pad(pixel_values, pad_values)
    # 返回填充后的像素值
    return pixel_values

# 定义模型的前向传播方法
def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
    # 获取输入张量的形状信息
    _, num_channels, height, width = pixel_values.shape
    # 检查通道数是否与配置中设置的通道数一致
    if num_channels != self.num_channels:
        raise ValueError(
            "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
        )
    # 调用maybe_pad方法，对输入进行填充操作，使得其可以被patch_size整除
    pixel_values = self.maybe_pad(pixel_values, height, width)
    # 将填充后的像素值通过投影层projection映射为嵌入向量
    embeddings = self.projection(pixel_values)
    # 获取嵌入向量的形状信息
    _, _, height, width = embeddings.shape
    # 记录输出的高度和宽度信息
    output_dimensions = (height, width)
    # 对嵌入向量进行展平和转置操作，以便后续处理
    embeddings = embeddings.flatten(2).transpose(1, 2)
    # 返回嵌入向量和输出的尺寸信息
    return embeddings, output_dimensions
class SwinPatchMerging(nn.Module):
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
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def maybe_pad(self, input_feature, height, width):
        # Check if padding is needed to make height and width even
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)

        return input_feature

    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
        height, width = input_dimensions
        # Calculate the batch size, dimension, and number of channels of the input feature
        batch_size, dim, num_channels = input_feature.shape

        # Reshape the input feature
        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # Pad input to be divisible by width and height, if needed
        input_feature = self.maybe_pad(input_feature, height, width)
        # Extract sub-features based on stride 2 in height and width
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # Concatenate the sub-features
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)  # Reshape the input feature

        input_feature = self.norm(input_feature)
        input_feature = self.reduction(input_feature)

        return input_feature


def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    name of this impl to `drop_path` to clear up any confusion.
    """
```  
    def drop_path(input, drop_prob, training):
        """
        如果 drop_prob 为 0 或者不处于训练状态，则直接返回输入
        """
        if drop_prob == 0.0 or not training:
            return input
        # 计算保留的概率
        keep_prob = 1 - drop_prob
        # 确定输出形状，以便适用于不同维度的张量，而不仅限于 2D ConvNets
        shape = (input.shape[0],) + (1,) * (input.ndim - 1)
        # 生成与输入形状相同的随机张量，使用 keep_prob 作为基数
        random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
        # 将随机张量二值化，以实现随机丢弃
        random_tensor.floor_()
        # 计算输出，除以 keep_prob 并乘以随机张量
        output = input.div(keep_prob) * random_tensor
        # 返回输出
        return output
# 从transformers.models.beit.modeling_beit.BeitDropPath复制，并将Beit->Swin
class SwinDropPath(nn.Module):
    """针对每个样本进行丢弃路径（随机深度），应用于残差块的主路径中。"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        # 设置丢弃概率
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用drop_path函数，用于处理隐藏状态、丢弃概率和训练模式
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class SwinSelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"隐藏大小({dim})不是注意力头数({num_heads})的倍数"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )

        # 创建用于存储相对位置偏置的可学习参数，维度为(2 * window_size[0] - 1) * (2 * window_size[1] - 1) * num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        # 获取窗口内每个标记的配对相对位置索引
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
        # 将相对位置索引作为缓冲区注册
        self.register_buffer("relative_position_index", relative_position_index)

        # 初始化查询、键、值的线性变换层
        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)

        # 初始化dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 将张量重塑为(batch_size, num_heads, seq_len, head_dim)形状
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        # 定义函数的输入参数和返回类型为一个包含 torch.Tensor 类型的元组
        ) -> Tuple[torch.Tensor]:
        # 获取隐藏状态张量的批量大小、维度和通道数
        batch_size, dim, num_channels = hidden_states.shape
        # 使用 self.query 函数对隐藏状态进行查询
        mixed_query_layer = self.query(hidden_states)

        # 使用 self.key 函数生成密钥层，并通过 self.transpose_for_scores 函数进行格式转换
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 使用 self.value 函数生成值层，并通过 self.transpose_for_scores 函数进行格式转换
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 使用 mixed_query_layer 生成查询层，并通过 self.transpose_for_scores 函数进行格式转换
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算"查询"和"密钥"之间的点积，得到原始的注意力得分
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力得分除以 sqrt(注意力头大小) 进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 从相对位置偏置表获取相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )

        # 调整相对位置偏置的维度顺序
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # 将获取的相对位置偏置添加到注意力得分中
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        # 如果存在注意力遮罩，应用该遮罩
        if attention_mask is not None:
            # 获取遮罩的形状信息
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # 将注意力得分归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 通过 dropout 函数对注意力概率进行dropout操作
        attention_probs = self.dropout(attention_probs)

        # 如果有头部遮罩，将影响应用到注意力概率上
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，通过注意力概率和值层的矩阵乘积得到
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据是否输出注意力信息来返回相应的结果
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 定义一个自定义的模块 SwinSelfOutput，包括初始化函数和前向传播函数
class SwinSelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个线性层对象，输入和输出维度均为 dim
        self.dense = nn.Linear(dim, dim)
        # 创建一个 Dropout 层对象，概率设为 config.attention_probs_dropout_prob
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用线性层处理 hidden_states
        hidden_states = self.dense(hidden_states)
        # 使用 Dropout 层处理 hidden_states
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# 定义一个自定义的模块 SwinAttention，包括初始化函数和 prune_heads、forward 两个方法
class SwinAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        # 创建 SwinSelfAttention 对象
        self.self = SwinSelfAttention(config, dim, num_heads, window_size)
        # 创建 SwinSelfOutput 对象
        self.output = SwinSelfOutput(config, dim)
        # 初始化一个空集合 pruned_heads
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 方法找到可剪枝的头和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头
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
        # 调用 SwinSelfAttention 的前向传播方法
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        # 调用 SwinSelfOutput 的前向传播方法
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出了注意力权重，则添加到输出中
        return outputs


# 定义一个自定义的模块 SwinIntermediate，包括初始化函数和前向传播函数
class SwinIntermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个线性层对象，输入维度为 dim，输出维度为 config.mlp_ratio * dim
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        # 根据 config 中的 hidden_act 字符串或函数，选择相应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用线性层处理 hidden_states
        hidden_states = self.dense(hidden_states)
        # 使用选定的激活函数处理 hidden_states
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 定义一个自定义的模块 SwinOutput，包括初始化函数
class SwinOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个线性层对象，输入维度为 config.mlp_ratio * dim，输出维度为 dim
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        # 创建一个 Dropout 层对象，概率设为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    # 前向传播函数，接受隐藏状态作为输入张量，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过全连接层处理隐藏状态张量，输出经过线性变换的结果
        hidden_states = self.dense(hidden_states)
        # 对处理后的张量进行 dropout 操作，以减少过拟合风险
        hidden_states = self.dropout(hidden_states)
        # 返回经过全连接层和 dropout 处理后的张量
        return hidden_states
# 定义SwinLayer类，继承自nn.Module
class SwinLayer(nn.Module):
    # 初始化方法，接收config, dim, input_resolution, num_heads, shift_size等参数
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0):
        # 调用父类的初始化方法
        super().__init__()
        # 设置chunk_size_feed_forward属性
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置shift_size属性
        self.shift_size = shift_size
        # 设置window_size属性
        self.window_size = config.window_size
        # 设置input_resolution属性
        self.input_resolution = input_resolution
        # 对输入进行layer normalization
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建注意力层对象
        self.attention = SwinAttention(config, dim, num_heads, window_size=self.window_size)
        # 随机丢弃路径
        self.drop_path = SwinDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        # 对输出进行layer normalization
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建SwinIntermediate对象
        self.intermediate = SwinIntermediate(config, dim)
        # 创建SwinOutput对象
        self.output = SwinOutput(config, dim)

    # 设置shift_size和window_size方法，接收input_resolution参数
    def set_shift_and_window_size(self, input_resolution):
        # 如果输入分辨率的最小值小于等于窗口尺寸，即输入尺寸小于等于窗口尺寸
        if min(input_resolution) <= self.window_size:
            # 将shift_size置为0
            self.shift_size = 0
            # 将窗口尺寸置为输入分辨率的最小值
            self.window_size = min(input_resolution)

    # 获取注意力掩码方法，接收height, width, dtype参数
    def get_attn_mask(self, height, width, dtype):
        # 如果shift_size大于0
        if self.shift_size > 0:
            # 计算SW-MSA的注意力掩码
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
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
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        # 返回获取到的注意力掩码
        return attn_mask

    # 对隐藏状态进行填充方法，接收hidden_states, height, width参数
    def maybe_pad(self, hidden_states, height, width):
        # 计算右侧和底部所需填充的长度
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        # 构建填充值
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        # 对隐藏状态进行填充
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        # 返回填充后的隐藏状态和填充值
        return hidden_states, pad_values

    # 前向传播方法，接收hidden_states, input_dimensions, head_mask, output_attentions, always_partition等参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    # 定义函数的返回类型为一个包含两个 torch.Tensor 类型元素的元组
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 如果不总是分区，则根据输入维度设置偏移和窗口大小
        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)
        else:
            pass
        # 获取输入维度的高度和宽度
        height, width = input_dimensions
        # 获取隐藏状态的批大小、通道数和通道数
        batch_size, _, channels = hidden_states.size()
        # 备份隐藏状态
        shortcut = hidden_states

        # LayerNorm 操作，应用于隐藏状态
        hidden_states = self.layernorm_before(hidden_states)

        # 将隐藏状态调整形状为(batch_size, height, width, channels)
        hidden_states = hidden_states.view(batch_size, height, width, channels)

        # 将 hidden_states 填充为窗口大小的倍数
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        # 获取pad后的高度、宽度
        _, height_pad, width_pad, _ = hidden_states.shape
        # 循环移位
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # 划分窗口
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        # 注意力输出
        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        attention_output = attention_outputs[0]

        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)

        # 反向循环移位
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        # 如果进行了填充，则截断 attention_windows
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)

        # 与shortcut相加并应用 drop_path 操作
        hidden_states = shortcut + self.drop_path(attention_windows)

        # LayerNorm 操作，应用于隐藏状态
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.output(layer_output)

        # 如果需要输出注意力，则返回 layer_output 和 attention 输出的元组，否则只返回 layer_output
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs
```  
class SwinStage(nn.Module):
    # 定义SwinStage类，用于实现Swin Transformer的一个阶段
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        # 初始化方法，接受参数config、dim、input_resolution、depth、num_heads、drop_path、downsample
        super().__init__()
        self.config = config
        self.dim = dim
        # 使用ModuleList创建一个模块列表，包含depth个SwinLayer实例
        self.blocks = nn.ModuleList(
            [
                SwinLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            # 如果downsample不为None，则创建downsample实例
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None
        self.pointing = False
        # 初始化pointing标志为False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 定义前向传播方法，接受输入hidden_states、input_dimensions以及多个可选参数
        height, width = input_dimensions
        # 从input_dimensions中获取height和width
        for i, layer_module in enumerate(self.blocks):
            # 遍历每个SwinLayer模块
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition
            )
            # 调用SwinLayer模块进行前向传播

            hidden_states = layer_outputs[0]
            # 将当前层的输出作为下一层的输入

        hidden_states_before_downsampling = hidden_states
        # 保存下采样之前的hidden_states
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            # 计算下采样后的高度和宽度
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)
            # 进行下采样操作
        else:
            output_dimensions = (height, width, height, width)
        # 更新output_dimensions

        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)
        # 将计算结果作为输出

        if output_attentions:
            stage_outputs += layer_outputs[1:]
        # 如果需要输出注意力矩阵，则将其加入到输出中
        return stage_outputs
        # 返回阶段的输出


class SwinEncoder(nn.Module):
    # 定义SwinEncoder类，用于实现Swin Transformer的编码器
    # 初始化函数，接受配置和网格大小参数
    def __init__(self, config, grid_size):
        # 调用父类的初始化方法
        super().__init__()
        # 计算网络层数
        self.num_layers = len(config.depths)
        # 保存配置信息
        self.config = config
        # 生成一组按比例递增的随机数，用于 DropPath（Dropout + 路径）操作
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        # 创建一个模块列表，每个元素是一个 SwinStage 模块
        self.layers = nn.ModuleList(
            [
                SwinStage(
                    # 配置信息
                    config=config,
                    # 当前层的维度，随着层数的增加而增大
                    dim=int(config.embed_dim * 2**i_layer),
                    # 输入分辨率，随着层数的增加而减小
                    input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                    # 当前层的深度
                    depth=config.depths[i_layer],
                    # 当前层的头数
                    num_heads=config.num_heads[i_layer],
                    # 当前层的 DropPath 概率列表，对应该层中每个 Transformer 模块
                    drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    # 下采样操作，如果不是最后一层，则使用 SwinPatchMerging 类进行下采样
                    downsample=SwinPatchMerging if (i_layer < self.num_layers - 1) else None,
                )
                # 对每一层进行循环
                for i_layer in range(self.num_layers)
            ]
        )
        # 是否启用梯度检查点
        self.gradient_checkpointing = False

    # 前向传播函数
    def forward(
        self,
        # 输入隐藏状态张量
        hidden_states: torch.Tensor,
        # 输入图像的维度
        input_dimensions: Tuple[int, int],
        # 头部掩码，用于控制哪些头部的注意力会被屏蔽
        head_mask: Optional[torch.FloatTensor] = None,
        # 是否输出注意力分布
        output_attentions: Optional[bool] = False,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = False,
        # 是否在下采样之前输出隐藏状态
        output_hidden_states_before_downsampling: Optional[bool] = False,
        # 是否总是对输入进行分区
        always_partition: Optional[bool] = False,
        # 是否以字典形式返回结果
        return_dict: Optional[bool] = True,
class SwinPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置 Swin 模型的配置类
    config_class = SwinConfig
    # Swin 模型的主要输入名称
    base_model_prefix = "swin"
    # Swin 模型的主要输入名称
    main_input_name = "pixel_values"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 初始化权重
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 对于线性层和卷积层，使用正态分布初始化权重
            # 与 TF 版本略有不同，TF 版本使用截断正态分布初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，则将偏置初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对于 LayerNorm 层，将偏置初始化为零，将权重初始化为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


SWIN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SwinConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SWIN_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

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
    "The bare Swin Model transformer outputting raw hidden-states without any specific head on top.",
    SWIN_START_DOCSTRING,
    # 这个代码块定义了两个可选参数:
    # 1. add_pooling_layer: 一个布尔类型的参数,默认为 True,用于决定是否应用池化层。
    # 2. use_mask_token: 一个布尔类型的参数,默认为 False,用于决定是否在嵌入层中创建和应用掩码令牌。
    """
        add_pooling_layer (`bool`, *optional*, defaults to `True`):
                Whether or not to apply pooling layer.
        use_mask_token (`bool`, *optional*, defaults to `False`):
                Whether or not to create and apply mask tokens in the embedding layer.
    """
# 定义了一个 SwinModel 类，继承自 SwinPreTrainedModel 类
class SwinModel(SwinPreTrainedModel):
    # 初始化函数，接受配置对象、是否添加池化层标志和是否使用掩码标记作为参数
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        # 调用父类的初始化函数
        super().__init__(config)
        # 将配置对象保存在实例中
        self.config = config
        # 计算特征维度
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        # 创建 SwinEmbeddings 实例
        self.embeddings = SwinEmbeddings(config, use_mask_token=use_mask_token)
        # 创建 SwinEncoder 实例
        self.encoder = SwinEncoder(config, self.embeddings.patch_grid)

        # 创建 LayerNorm 层，用于规范化
        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        # 如果需要添加池化层，则创建 AdaptiveAvgPool1d 池化层
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 对模型的注意力头进行剪枝
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # 获取指定层的注意力头并进行剪枝
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播方法，接受像素值、布尔掩码位置、头掩码、输出注意力、输出隐藏状态和返回字典标志作为参数
    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SwinModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, SwinModelOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # 初始化 output_attentions 变量，如果未提供则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 初始化 output_hidden_states 变量，如果未提供则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 初始化 return_dict 变量，如果未提供则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供像素值，则引发数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 准备头部掩码（head_mask）（如果需要）
        # head_mask 中的 1.0 表示要保留该头部
        # attention_probs 的形状是 bsz x n_heads x N x N
        # 输入的头部掩码（head_mask）的形状是 [num_heads] 或 [num_hidden_layers x num_heads]
        # 并且将 head_mask 转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))

        # 使用嵌入层将像素值转换成嵌入输出，同时传入布尔掩码位置信息
        embedding_output, input_dimensions = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        # 使用编码器对嵌入输出进行编码
        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = encoder_outputs[0]
        # 对序列输出进行 Layernorm 处理
        sequence_output = self.layernorm(sequence_output)

        pooled_output = None
        # 如果存在池化器对象，则对序列输出进行池化操作
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)

        # 如果不需要返回字典，则返回输出元组
        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]

            return output

        # 如果需要返回字典形式的输出，返回 SwinModelOutput 对象
        return SwinModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )
# 添加文档字符串描述 SwinForMaskedImageModeling 类，提供一些关于该类的信息和提示
class SwinForMaskedImageModeling(SwinPreTrainedModel):
    # 初始化函数，接收配置参数，并调用父类的初始化函数
    def __init__(self, config):
        super().__init__(config)

        # 创建 Swin 模型对象，关闭池化层，使用掩膜标记
        self.swin = SwinModel(config, add_pooling_layer=False, use_mask_token=True)

        # 计算特征数
        num_features = int(config.embed_dim * 2 ** (config.num_layers - 1))
        
        # 创建解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=num_features, out_channels=config.encoder_stride**2 * config.num_channels, kernel_size=1
            ),
            nn.PixelShuffle(config.encoder_stride),
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 添加文档字符串描述 forward 函数的输入参数和输出类型
    # 替换返回文档字符串格式为 SwinMaskedImageModelingOutput，模型配置为 _CONFIG_FOR_DOC
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

# 添加文档字符串描述 SwinForImageClassification 类，提供一些关于该类的信息
class SwinForImageClassification(SwinPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        # 获取标签数目和创建 Swin 模型对象
        self.num_labels = config.num_labels
        self.swin = SwinModel(config)

        # 分类器头部
        self.classifier = (
            nn.Linear(self.swin.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 添加文档字符串描述 forward 函数的输入参数和输出类型
    # 添加代码示例文档字符串，包括模型预训练检查点、输出类型、配置类和预期输出
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SwinImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确保 return_dict 参数不为 None，如果为 None，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入数据传递给 Swin Transformer 模型进行处理
        outputs = self.swin(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取汇总后的特征向量
        pooled_output = outputs[1]

        # 将汇总后的特征向量传递给分类器，获得分类器的 logits
        logits = self.classifier(pooled_output)

        loss = None
        # 如果标签不为 None，则计算损失
        if labels is not None:
            # 确定问题类型（回归、单标签分类、多标签分类）
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数，并计算损失值
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

        # 如果 return_dict 为 False，则返回不包含损失值的输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回一个 SwinImageClassifierOutput 对象
        return SwinImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )
# SwinBackbone 类是一个实现了 Swin 基本功能的类，可以用于 DETR 和 MaskFormer 这样的框架
#@add_start_docstrings 是一个装饰器，用于向类提供额外的文档字符串
class SwinBackbone(SwinPreTrainedModel, BackboneMixin):
    def __init__(self, config: SwinConfig):
        # 调用父类的 __init__() 方法来初始化 SwinBackbone 类
        super().__init__(config)
        # 调用父类的 _init_backbone() 方法来初始化骨干网络
        super()._init_backbone(config)

        # 保存每个层的特征维度
        self.num_features = [config.embed_dim] + [int(config.embed_dim * 2**i) for i in range(len(config.depths))]
        # 创建 SwinEmbeddings 对象，使用给定的配置
        self.embeddings = SwinEmbeddings(config)
        # 创建 SwinEncoder 对象，使用给定的配置和 patchs
        self.encoder = SwinEncoder(config, self.embeddings.patch_grid)

        # 创建隐藏状态的 LayerNorm，用于最后的输出特征
        hidden_states_norms = {}
        for stage, num_channels in zip(self._out_features, self.channels):
            hidden_states_norms[stage] = nn.LayerNorm(num_channels)
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)

        # 初始化权重并进行最后的处理
        self.post_init()

    # 返回嵌入图像的嵌入层
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 前向传播函数
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> BackboneOutput:
        """
        返回输出包含特征映射、隐藏状态和注意力
        示例：
        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("shi-labs/nat-mini-in1k-224")
        >>> model = AutoBackbone.from_pretrained(
        ...     "microsoft/swin-tiny-patch4-window7-224", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 7, 7]
        ```"""
        确定返回字典是否为None，如果不是则使用配置中的返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        确定是否输出所有隐藏状态，如果不是则使用配置中的输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        确定是否输出所有注意力，如果不是则使用配置中的输出注意力
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        使用嵌入层处理像素值
        embedding_output, input_dimensions = self.embeddings(pixel_values)

        使用编码器处理嵌入输出
        outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            always_partition=True,
            return_dict=True,
        )

        获取隐藏状态
        hidden_states = outputs.reshaped_hidden_states

        初始化特征映射列表
        feature_maps = ()
        遍历阶段和对应隐藏状态，处理后添加到特征映射列表中
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                batch_size, num_channels, height, width = hidden_state.shape
                hidden_state = hidden_state.permute(0, 2, 3, 1).contiguous()
                hidden_state = hidden_state.view(batch_size, height * width, num_channels)
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                hidden_state = hidden_state.view(batch_size, height, width, num_channels)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps += (hidden_state,)

        如果不需要返回字典，则返回特征映射和隐藏状态
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        返回由特征映射、隐藏状态和注意力组成的BackboneOutput对象
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
```