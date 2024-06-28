# `.\models\swin\modeling_swin.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 Microsoft Research 和 HuggingFace Inc. 团队所有

# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下链接获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0

# 除非适用法律要求或书面同意，否则本软件根据“原样”分发，不附带任何形式的保证或条件
# 详见许可证，以获取具体的语言表达和限制条件

""" PyTorch Swin Transformer model."""

# 导入必要的模块和库
import collections.abc  # 导入 collections.abc 模块
import math  # 导入 math 模块
import warnings  # 导入 warnings 模块
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from typing import Optional, Tuple, Union  # 导入类型提示所需的类和元组

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 模块
from torch import nn  # 从 torch 导入 nn 模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 从 torch.nn 导入损失函数

# 导入自定义模块和函数
from ...activations import ACT2FN  # 从上级目录导入 ACT2FN 函数
from ...modeling_outputs import BackboneOutput  # 从上级目录导入 BackboneOutput 类
from ...modeling_utils import PreTrainedModel  # 从上级目录导入 PreTrainedModel 类
from ...pytorch_utils import (  # 从上级目录导入多个函数和类
    find_pruneable_heads_and_indices,
    meshgrid,
    prune_linear_layer
)
from ...utils import (  # 从上级目录导入多个实用函数和类
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings
)
from ...utils.backbone_utils import BackboneMixin  # 从上级目录导入 BackboneMixin 类
from .configuration_swin import SwinConfig  # 从当前目录导入 SwinConfig 类

# 获取 logger 对象
logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "SwinConfig"  # _CONFIG_FOR_DOC 变量的文档说明

# Base docstring
_CHECKPOINT_FOR_DOC = "microsoft/swin-tiny-patch4-window7-224"  # _CHECKPOINT_FOR_DOC 变量的文档说明
_EXPECTED_OUTPUT_SHAPE = [1, 49, 768]  # _EXPECTED_OUTPUT_SHAPE 变量的文档说明

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "microsoft/swin-tiny-patch4-window7-224"  # _IMAGE_CLASS_CHECKPOINT 变量的文档说明
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"  # _IMAGE_CLASS_EXPECTED_OUTPUT 变量的文档说明

# Swin 预训练模型的存档列表
SWIN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/swin-tiny-patch4-window7-224",
    # 查看所有 Swin 模型：https://huggingface.co/models?filter=swin
]

# drop_path, SwinPatchEmbeddings, SwinPatchMerging 和 SwinDropPath 来自 timm 库
    # 定义函数参数和返回值的类型注解，使用了 Torch 库中的数据类型说明
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列，形状为 `(batch_size, sequence_length, hidden_size)`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含了模型每一层的隐藏状态，形状为 `(batch_size, sequence_length, hidden_size)`.
    
            模型每一层的隐藏状态以及初始嵌入输出的列表。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            一个元组，包含了每个阶段的注意力权重，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`.
    
            经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含了模型每一层的隐藏状态，形状为 `(batch_size, hidden_size, height, width)`.
    
            模型每一层的隐藏状态以及初始嵌入输出，重塑以包含空间维度。
    """
    
    # 定义变量并初始化为 None，用来存储模型输出的各种信息
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
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

    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class SwinMaskedImageModelingOutput(ModelOutput):
    """
    Swin masked image model outputs.

    This class is a data structure to hold outputs from a Swin Transformer model applied to masked image inputs.
    It extends the `ModelOutput` class.

    It includes the following attributes:

    """
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Masked image modeling (MLM) loss.
            图像掩码建模（MLM）损失，当提供 `bool_masked_pos` 时返回。
        reconstruction (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Reconstructed pixel values.
            重建的像素值，形状为 `(batch_size, num_channels, height, width)`。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            模型每层输出的隐藏状态，包括初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均。
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
            每层输出的隐藏状态，包括重塑以包含空间维度的初始嵌入输出。

    """

    # Optional 类型的变量定义，用于存储损失值，默认为 None
    loss: Optional[torch.FloatTensor] = None
    # 定义变量，存储重建后的像素值，类型为 torch.FloatTensor
    reconstruction: torch.FloatTensor = None
    # Optional 类型的变量定义，存储隐藏状态的元组，默认为 None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # Optional 类型的变量定义，存储注意力权重的元组，默认为 None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # Optional 类型的变量定义，存储重塑后的隐藏状态的元组，默认为 None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None

    @property
    def logits(self):
        # 发出警告信息，提示 logits 属性即将在 Transformers 的第 5 版中被移除
        warnings.warn(
            "logits attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the reconstruction attribute to retrieve the final output instead.",
            FutureWarning,
        )
        # 返回 reconstruction 属性作为最终输出
        return self.reconstruction
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

    loss: Optional[torch.FloatTensor] = None  # 分类（或回归，如果config.num_labels==1）的损失值
    logits: torch.FloatTensor = None  # 分类（或回归，如果config.num_labels==1）的得分（SoftMax之前）
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 模型每层输出的隐藏状态及初始嵌入输出的元组
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 注意力权重的元组，用于计算自注意力头中的加权平均值
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 每层输出的隐藏状态及初始嵌入输出的元组，包括空间维度

def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    batch_size, height, width, num_channels = input_feature.shape  # 获取输入特征的形状信息
    input_feature = input_feature.view(  # 将输入特征重塑为窗口大小的块
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(  # 对重塑后的特征进行维度置换和重新排序，得到窗口列表
        -1, window_size, window_size, num_channels)
    return windows  # 返回分区后的窗口列表

def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]  # 获取窗口的通道数
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)  # 将窗口重新排列以恢复高分辨率特征
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(  # 对重排的窗口进行维度置换和重新排序
        -1, height, width, num_channels)
    # 返回 windows 变量的值作为函数的结果
    return windows
class SwinEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.patch_embeddings = SwinPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches  # 获取patch嵌入的patch数目
        self.patch_grid = self.patch_embeddings.grid_size  # 获取patch网格大小
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None  # 如果使用mask token，则创建一个可训练的零张量

        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))  # 如果使用绝对位置嵌入，则创建一个可训练的零张量
        else:
            self.position_embeddings = None  # 否则位置嵌入设为None

        self.norm = nn.LayerNorm(config.embed_dim)  # 创建LayerNorm层，用于标准化嵌入
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 创建Dropout层，用于随机dropout

    def forward(
        self, pixel_values: Optional[torch.FloatTensor], bool_masked_pos: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)  # 获得patch嵌入和输出维度
        embeddings = self.norm(embeddings)  # 对嵌入进行标准化
        batch_size, seq_len, _ = embeddings.size()  # 获取批量大小、序列长度和嵌入维度

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)  # 将mask token扩展到与嵌入张量相同的维度
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)  # 将布尔掩码转换为与mask token相同类型的张量
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask  # 根据掩码替换被遮蔽的视觉token为mask token

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings  # 如果存在位置嵌入，则加上位置嵌入

        embeddings = self.dropout(embeddings)  # 对嵌入进行随机dropout

        return embeddings, output_dimensions


class SwinPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])  # 计算图像分割成patch后的数量
        self.image_size = image_size  # 图像大小
        self.patch_size = patch_size  # patch大小
        self.num_channels = num_channels  # 输入通道数
        self.num_patches = num_patches  # patch数目
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])  # patch的网格大小

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        # 创建2D卷积层，用于将输入的像素值转换为patch嵌入
    # 可能对输入进行填充，使其能被 self.patch_size 整除（如果需要）
    def maybe_pad(self, pixel_values, height, width):
        if width % self.patch_size[1] != 0:
            # 计算需要填充的值，使得宽度能够被 patch_size[1] 整除
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            # 使用 nn.functional.pad 函数进行填充操作
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            # 计算需要填充的值，使得高度能够被 patch_size[0] 整除
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            # 使用 nn.functional.pad 函数进行填充操作
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    # 前向传播函数，接受像素值作为输入，返回嵌入张量和输出尺寸元组
    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        # 获取输入张量的形状信息
        _, num_channels, height, width = pixel_values.shape
        # 检查通道维度是否与配置中设置的通道数匹配，如果不匹配则引发 ValueError
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 可能对输入进行填充操作，使其能被 patch_size 整除
        pixel_values = self.maybe_pad(pixel_values, height, width)
        # 将填充后的输入传递给投影层，生成嵌入张量
        embeddings = self.projection(pixel_values)
        # 获取嵌入张量的形状信息
        _, _, height, width = embeddings.shape
        # 计算输出的高度和宽度尺寸
        output_dimensions = (height, width)
        # 将嵌入张量展平并转置，以便输出
        embeddings = embeddings.flatten(2).transpose(1, 2)

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
        self.input_resolution = input_resolution  # 存储输入特征的分辨率
        self.dim = dim  # 存储输入通道数
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 线性变换，用于特征维度的降维
        self.norm = norm_layer(4 * dim)  # 使用指定的归一化层对特征进行归一化

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)  # 如果需要，对输入特征进行填充

        return input_feature

    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
        height, width = input_dimensions
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width, num_channels)  # 重新组织输入特征的形状
        input_feature = self.maybe_pad(input_feature, height, width)  # 调用填充函数进行可能的填充操作

        # 分割输入特征并组合成新的特征图
        input_feature_0 = input_feature[:, 0::2, 0::2, :]  # 取出每隔一个像素的子图
        input_feature_1 = input_feature[:, 1::2, 0::2, :]  # 取出每隔一个像素的子图
        input_feature_2 = input_feature[:, 0::2, 1::2, :]  # 取出每隔一个像素的子图
        input_feature_3 = input_feature[:, 1::2, 1::2, :]  # 取出每隔一个像素的子图

        # 将分割的子图按通道方向连接起来
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)

        # 重新调整形状，合并后的特征图
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)

        input_feature = self.norm(input_feature)  # 对连接后的特征进行归一化
        input_feature = self.reduction(input_feature)  # 使用线性变换减少特征维度

        return input_feature


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    """
    # 如果 drop_prob 为 0 或者不处于训练状态，则直接返回输入，无需执行 Dropout
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留概率
    keep_prob = 1 - drop_prob
    # 创建形状与输入相同的随机张量，用于决定每个神经元的保留情况
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度张量，不仅限于2D卷积网络
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 将随机张量二值化为0或1
    # 应用 Dropout，乘以随机张量并除以保留概率
    output = input.div(keep_prob) * random_tensor
    # 返回 Dropout 后的输出
    return output
# 从 transformers.models.swin.modeling_swin.SwinDropPath 复制，并将 Beit->Swin
class SwinDropPath(nn.Module):
    """每个样本应用于残差块主路径的丢弃路径（随机深度）。"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class SwinSelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"隐藏大小 ({dim}) 不是注意力头数 ({num_heads}) 的倍数"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        # 获取每个窗口内每个标记的成对相对位置索引
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

        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor]:  # 函数定义，返回一个元组，包含一个 torch.Tensor
        batch_size, dim, num_channels = hidden_states.shape  # 获取隐藏状态的批量大小、维度和通道数
        mixed_query_layer = self.query(hidden_states)  # 使用查询函数处理隐藏状态得到混合查询层

        key_layer = self.transpose_for_scores(self.key(hidden_states))  # 使用键函数处理隐藏状态并转置得到键层
        value_layer = self.transpose_for_scores(self.value(hidden_states))  # 使用值函数处理隐藏状态并转置得到值层
        query_layer = self.transpose_for_scores(mixed_query_layer)  # 转置处理混合查询层得到查询层

        # 计算查询与键的点积，得到原始注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # 根据头部大小对注意力分数进行缩放

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)  # 添加相对位置偏置到注意力分数中

        if attention_mask is not None:
            # 如果存在注意力掩码，则应用它（在 SwinModel 的 forward() 函数中预先计算）
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # 将注意力分数归一化为注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用 dropout 随机丢弃整个 token 的注意力概率
        attention_probs = self.dropout(attention_probs)

        # 如果有头部掩码，则应用它
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)  # 计算上下文层
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)  # 准备输出结果

        return outputs  # 返回计算后的输出
# 定义一个名为 SwinSelfOutput 的类，继承自 nn.Module
class SwinSelfOutput(nn.Module):
    # 初始化方法，接收 config 和 dim 参数
    def __init__(self, config, dim):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入和输出维度均为 dim
        self.dense = nn.Linear(dim, dim)
        # 创建一个 Dropout 层，使用 config 中的注意力概率参数
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 前向传播方法，接收 hidden_states 和 input_tensor 两个 Tensor 作为输入，返回一个 Tensor
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将 hidden_states 输入到 self.dense 中进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的结果使用 Dropout 层
        hidden_states = self.dropout(hidden_states)

        # 返回处理后的 hidden_states
        return hidden_states


# 定义一个名为 SwinAttention 的类，继承自 nn.Module
class SwinAttention(nn.Module):
    # 初始化方法，接收 config、dim、num_heads 和 window_size 参数
    def __init__(self, config, dim, num_heads, window_size):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个 SwinSelfAttention 对象，传入 config、dim、num_heads 和 window_size 参数
        self.self = SwinSelfAttention(config, dim, num_heads, window_size)
        # 创建一个 SwinSelfOutput 对象，传入 config 和 dim 参数
        self.output = SwinSelfOutput(config, dim)
        # 创建一个空集合，用于存储需要剪枝的注意力头索引
        self.pruned_heads = set()

    # 剪枝注意力头的方法，接收 heads 参数
    def prune_heads(self, heads):
        # 如果 heads 集合为空，则直接返回
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数，获取可剪枝的头部索引和相关信息
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 对 self.self 中的 query、key、value 和 output.dense 层进行剪枝
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头部索引
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播方法，接收 hidden_states、attention_mask、head_mask 和 output_attentions 四个参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 调用 self.self 的前向传播方法，计算自注意力输出
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        # 将 self_outputs[0] 和 hidden_states 输入到 self.output 中，计算注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 将注意力输出和可能的额外注意力返回作为元组 outputs 的一部分
        outputs = (attention_output,) + self_outputs[1:]  # 如果有额外的注意力输出，添加到 outputs 中
        return outputs


# 定义一个名为 SwinIntermediate 的类，继承自 nn.Module
class SwinIntermediate(nn.Module):
    # 初始化方法，接收 config 和 dim 参数
    def __init__(self, config, dim):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，将 dim 维度映射到 config.mlp_ratio * dim 维度
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        # 如果 config.hidden_act 是字符串类型，则使用 ACT2FN 字典中对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法，接收 hidden_states 作为输入，返回处理后的 hidden_states
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将 hidden_states 输入到 self.dense 中进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的结果使用 intermediate_act_fn 激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states


# 定义一个名为 SwinOutput 的类，继承自 nn.Module
class SwinOutput(nn.Module):
    # 初始化方法，接收 config 和 dim 参数
    def __init__(self, config, dim):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，将 config.mlp_ratio * dim 维度映射到 dim 维度
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        # 创建一个 Dropout 层，使用 config 中的隐藏层 Dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    # 定义前向传播方法，接受隐藏状态张量作为输入，并返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入张量通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的张量应用丢弃（dropout）操作，以防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 返回经过全连接层和丢弃操作后的张量作为输出
        return hidden_states
    # SwinLayer 类定义，继承自 nn.Module
    class SwinLayer(nn.Module):
        # 初始化函数，接受配置、维度、输入分辨率、头数、偏移大小等参数
        def __init__(self, config, dim, input_resolution, num_heads, shift_size=0):
            super().__init__()
            # 设置块大小用于前馈传播
            self.chunk_size_feed_forward = config.chunk_size_feed_forward
            # 设置偏移大小
            self.shift_size = shift_size
            # 窗口大小
            self.window_size = config.window_size
            # 输入分辨率
            self.input_resolution = input_resolution
            # 在注意力操作前使用 LayerNorm
            self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
            # 使用 SwinAttention 类进行注意力计算
            self.attention = SwinAttention(config, dim, num_heads, window_size=self.window_size)
            # 如果存在丢弃路径率，则应用 SwinDropPath；否则使用恒等映射
            self.drop_path = SwinDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
            # 在注意力操作后使用 LayerNorm
            self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
            # SwinIntermediate 类，处理注意力后的中间层输出
            self.intermediate = SwinIntermediate(config, dim)
            # SwinOutput 类，生成最终的输出
            self.output = SwinOutput(config, dim)

        # 设置偏移和窗口大小的方法，根据输入分辨率调整
        def set_shift_and_window_size(self, input_resolution):
            if min(input_resolution) <= self.window_size:
                # 如果窗口大小大于输入分辨率，则不分割窗口
                self.shift_size = 0
                self.window_size = min(input_resolution)

        # 生成注意力掩码的方法，根据高度、宽度和数据类型生成不同形状的掩码
        def get_attn_mask(self, height, width, dtype):
            if self.shift_size > 0:
                # 如果设置了偏移大小，则计算 SW-MSA 的注意力掩码
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
            return attn_mask

        # 在输入状态可能需要填充时进行填充的方法
        def maybe_pad(self, hidden_states, height, width):
            pad_right = (self.window_size - width % self.window_size) % self.window_size
            pad_bottom = (self.window_size - height % self.window_size) % self.window_size
            pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
            hidden_states = nn.functional.pad(hidden_states, pad_values)
            return hidden_states, pad_values

        # 前向传播方法，接受输入状态张量、输入尺寸、头部掩码、输出注意力和是否总是分割的布尔参数
        def forward(
            self,
            hidden_states: torch.Tensor,
            input_dimensions: Tuple[int, int],
            head_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = False,
            always_partition: Optional[bool] = False,
        ):
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 如果不是始终分区，则根据输入维度设置位移和窗口大小
        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)
        else:
            # 如果始终分区，不执行任何操作
            pass
        # 解包输入维度
        height, width = input_dimensions
        # 解包隐藏状态的批量大小、高度、宽度和通道数
        batch_size, _, channels = hidden_states.size()
        # 备份隐藏状态
        shortcut = hidden_states

        # 在层归一化之前应用层归一化
        hidden_states = self.layernorm_before(hidden_states)

        # 将隐藏状态重塑为 [batch_size, height, width, channels] 的形状
        hidden_states = hidden_states.view(batch_size, height, width, channels)

        # 对隐藏状态进行填充，使其大小为窗口大小的倍数
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        # 获取填充后的隐藏状态的维度信息
        _, height_pad, width_pad, _ = hidden_states.shape

        # 如果位移大小大于 0，则对隐藏状态进行循环移位操作
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # 分区窗口
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        # 将分区后的窗口重塑为 [batch_size * num_windows, window_size * window_size, channels] 的形状
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)

        # 获取注意力掩码
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        # 应用注意力机制
        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        # 获取注意力输出
        attention_output = attention_outputs[0]

        # 将注意力输出重塑为 [batch_size * num_windows, window_size, window_size, channels] 的形状
        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)

        # 反转窗口分区操作
        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)

        # 如果位移大小大于 0，则反转循环位移操作
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        # 检查是否进行了填充
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            # 如果进行了填充，则截取注意力窗口以匹配原始输入尺寸
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        # 将注意力窗口重塑为 [batch_size, height * width, channels] 的形状
        attention_windows = attention_windows.view(batch_size, height * width, channels)

        # 将捷径添加到注意力窗口上，并应用丢弃路径
        hidden_states = shortcut + self.drop_path(attention_windows)

        # 在层归一化之后应用层归一化
        layer_output = self.layernorm_after(hidden_states)

        # 应用中间层
        layer_output = self.intermediate(layer_output)

        # 应用输出层
        layer_output = hidden_states + self.output(layer_output)

        # 返回层输出及注意力信息（如果需要）
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs
# 定义 SwinStage 类，继承自 nn.Module，用于实现一个 Swin Transformer 的阶段
class SwinStage(nn.Module):
    # 初始化方法，接受多个参数来配置 SwinStage 实例
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        super().__init__()
        # 将传入的配置参数保存到实例中
        self.config = config
        # 设置维度参数
        self.dim = dim
        # 创建 SwinLayer 组成的模块列表
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

        # 如果 downsample 参数不为 None，则创建一个下采样层
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        # 初始化指向性为 False
        self.pointing = False

    # 前向传播方法，接受多个输入参数并返回多个输出参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 解包输入维度
        height, width = input_dimensions
        # 遍历 SwinLayer 模块列表
        for i, layer_module in enumerate(self.blocks):
            # 如果 head_mask 不为 None，则取出当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 调用当前层模块的 forward 方法计算输出
            layer_outputs = layer_module(
                hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition
            )

            # 更新 hidden_states 为当前层的输出
            hidden_states = layer_outputs[0]

        # 在进行下采样之前保存当前 hidden_states
        hidden_states_before_downsampling = hidden_states
        # 如果存在下采样层，则进行下采样操作
        if self.downsample is not None:
            # 计算下采样后的高度和宽度
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            # 更新输出维度信息
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            # 调用下采样层的 forward 方法进行下采样
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)
        else:
            # 没有下采样时，输出维度信息保持不变
            output_dimensions = (height, width, height, width)

        # 组装阶段的所有输出结果
        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)

        # 如果开启了输出注意力权重，则将每层的注意力权重输出加入到 stage_outputs 中
        if output_attentions:
            stage_outputs += layer_outputs[1:]
        
        # 返回阶段的所有输出结果作为元组
        return stage_outputs
    # 初始化函数，接受配置和网格大小作为参数
    def __init__(self, config, grid_size):
        # 调用父类的初始化方法
        super().__init__()
        # 计算网络层数
        self.num_layers = len(config.depths)
        # 存储配置信息
        self.config = config
        # 根据 drop_path_rate 参数生成一组 drop path 概率列表
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        # 创建网络层的 ModuleList，每层是一个 SwinStage 实例
        self.layers = nn.ModuleList(
            [
                SwinStage(
                    config=config,
                    # 计算当前层的嵌入维度
                    dim=int(config.embed_dim * 2**i_layer),
                    # 计算输入分辨率
                    input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                    # 设置当前层的深度
                    depth=config.depths[i_layer],
                    # 设置当前层的注意力头数
                    num_heads=config.num_heads[i_layer],
                    # 提取当前层的 drop path 概率列表
                    drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    # 如果不是最后一层，则设置 downsample 为 SwinPatchMerging 类
                    downsample=SwinPatchMerging if (i_layer < self.num_layers - 1) else None,
                )
                # 遍历所有网络层
                for i_layer in range(self.num_layers)
            ]
        )

        # 默认关闭梯度检查点
        self.gradient_checkpointing = False

    # 前向传播函数，接受输入的隐藏状态、输入维度和可选的掩码、输出配置等参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        # 后续可能还有其他参数
class SwinPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 SwinConfig
    config_class = SwinConfig
    # 基础模型前缀为 "swin"
    base_model_prefix = "swin"
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层或卷积层，使用正态分布初始化权重
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 与 TensorFlow 版本稍有不同，PyTorch 使用正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 LayerNorm 层，初始化偏置为零，权重为全1
        elif isinstance(module, nn.LayerNorm):
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
    """
    add_pooling_layer (`bool`, *optional*, defaults to `True`):
            是否应用池化层。
    use_mask_token (`bool`, *optional*, defaults to `False`):
            是否在嵌入层中创建和应用掩码标记。
    """
)
# 定义一个名为 SwinModel 的类，继承自 SwinPreTrainedModel
class SwinModel(SwinPreTrainedModel):
    # 初始化方法，接受配置参数 config、是否添加池化层 add_pooling_layer 和是否使用掩码令牌 use_mask_token
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将参数 config 存储为对象的配置
        self.config = config
        # 计算层数和特征数
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        # 创建 SwinEmbeddings 对象并存储在 self.embeddings 中
        self.embeddings = SwinEmbeddings(config, use_mask_token=use_mask_token)
        # 创建 SwinEncoder 对象并存储在 self.encoder 中，传入 patch_grid 参数
        self.encoder = SwinEncoder(config, self.embeddings.patch_grid)

        # 使用 nn.LayerNorm 初始化 layernorm 层，特征数为 self.num_features，epsilon 为 config.layer_norm_eps
        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        # 如果 add_pooling_layer 为 True，则创建 AdaptiveAvgPool1d 池化层，存储在 self.pooler 中；否则为 None
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

        # 调用 post_init 方法完成权重初始化和最终处理
        self.post_init()

    # 获取输入嵌入的方法，返回 patch_embeddings 属性
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 头部剪枝方法，用于剪枝模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历 heads_to_prune 字典，对每一层指定需要剪枝的注意力头
        for layer, heads in heads_to_prune.items():
            # 调用 encoder 层的每一层的 attention.prune_heads 方法，剪枝指定的注意力头
            self.encoder.layer[layer].attention.prune_heads(heads)

    # forward 方法，模型的前向传播
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
        # 根据需要设置输出注意力矩阵，默认使用配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据需要设置输出隐藏状态，默认使用配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据需要设置返回类型，默认使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            # 如果未提供像素值，抛出数值错误
            raise ValueError("You have to specify pixel_values")

        # 如果需要，准备头部掩码
        # head_mask 中的 1.0 表示保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 形状为 [num_heads] 或者 [num_hidden_layers x num_heads]
        # 将 head_mask 转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))

        # 将像素值和布尔掩码位置传递给嵌入层进行处理
        embedding_output, input_dimensions = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        # 使用编码器处理嵌入的输出
        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的序列输出并进行层归一化
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        # 初始化池化输出为 None
        pooled_output = None
        if self.pooler is not None:
            # 如果存在池化层，使用池化层对序列输出进行池化，并展平
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)

        if not return_dict:
            # 如果不要求返回字典形式的输出，构造并返回输出元组
            output = (sequence_output, pooled_output) + encoder_outputs[1:]

            return output

        # 如果需要返回字典形式的输出，构造并返回 SwinModelOutput 对象
        return SwinModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )
@add_start_docstrings(
    """
    Swin Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>
    """,
    SWIN_START_DOCSTRING,
)
class SwinForMaskedImageModeling(SwinPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.swin = SwinModel(config, add_pooling_layer=False, use_mask_token=True)

        num_features = int(config.embed_dim * 2 ** (config.num_layers - 1))
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=num_features, out_channels=config.encoder_stride**2 * config.num_channels, kernel_size=1
            ),
            nn.PixelShuffle(config.encoder_stride),
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SwinMaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Performs forward pass of the Swin model for masked image modeling.

        Args:
            pixel_values (torch.FloatTensor, optional): Tensor of pixel values of shape (batch_size, num_channels, height, width).
            bool_masked_pos (torch.BoolTensor, optional): Boolean mask indicating positions in pixel_values to be masked.
            head_mask (torch.FloatTensor, optional): Mask for attention heads.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dictionary instead of a tuple.

        Returns:
            SwinMaskedImageModelingOutput: Output object containing model outputs.
        """
        # Forward pass implementation is handled in the superclass and decorators

@add_start_docstrings(
    """
    Swin Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    SWIN_START_DOCSTRING,
)
class SwinForImageClassification(SwinPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.swin = SwinModel(config)

        # Classifier head
        self.classifier = (
            nn.Linear(self.swin.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=SwinImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Performs forward pass of the Swin model for image classification.

        Args:
            pixel_values (torch.FloatTensor, optional): Tensor of pixel values of shape (batch_size, num_channels, height, width).
            head_mask (torch.FloatTensor, optional): Mask for attention heads.
            labels (torch.LongTensor, optional): Labels for computing the classification loss.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dictionary instead of a tuple.

        Returns:
            SwinImageClassifierOutput: Output object containing model outputs.
        """
        # Forward pass implementation is handled in the superclass and decorators
        ) -> Union[Tuple, SwinImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确保 return_dict 变量不为 None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 Swin Transformer 模型进行前向传播
        outputs = self.swin(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中取得池化后的特征表示
        pooled_output = outputs[1]

        # 将池化后的特征表示传入分类器以得到 logits
        logits = self.classifier(pooled_output)

        # 初始化损失为 None
        loss = None
        if labels is not None:
            # 确定问题类型，如果尚未确定
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
                    # 对于单标签回归问题，使用均方误差损失函数
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签回归问题，同样使用均方误差损失函数
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类问题，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类问题，使用带 Logits 的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果 return_dict 为 False，则返回不包含损失的输出元组
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
@add_start_docstrings(
    """
    Swin backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    SWIN_START_DOCSTRING,
)
class SwinBackbone(SwinPreTrainedModel, BackboneMixin):
    """
    SwinTransformer的主干网络，可用于DETR和MaskFormer等框架。
    继承自SwinPreTrainedModel和BackboneMixin。
    """

    def __init__(self, config: SwinConfig):
        """
        初始化函数，接收一个SwinConfig类型的参数config。
        """
        super().__init__(config)
        # 调用父类SwinPreTrainedModel的初始化方法
        super()._init_backbone(config)

        # 计算每个阶段的特征维度
        self.num_features = [config.embed_dim] + [int(config.embed_dim * 2**i) for i in range(len(config.depths))]
        
        # 创建SwinEmbeddings对象
        self.embeddings = SwinEmbeddings(config)
        
        # 创建SwinEncoder对象，使用patch_grid参数
        self.encoder = SwinEncoder(config, self.embeddings.patch_grid)

        # 为输出特征的隐藏状态添加层归一化层
        hidden_states_norms = {}
        for stage, num_channels in zip(self._out_features, self.channels):
            hidden_states_norms[stage] = nn.LayerNorm(num_channels)
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        """
        获取输入嵌入的patch_embeddings。
        """
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        """
        前向传播函数，接收以下参数：
        - pixel_values: 输入的像素值张量
        - output_hidden_states: 是否输出隐藏状态，默认为None
        - output_attentions: 是否输出注意力权重，默认为None
        - return_dict: 是否返回字典格式的输出，默认为None
        """
        # 省略部分前向传播代码...
        """
        返回BackboneOutput对象。

        返回：
            返回BackboneOutput对象，其中包含特征图、隐藏状态和注意力分数（如果有的话）。

        Examples:
        示例代码块，展示了如何使用该函数从图像提取特征图。
        """

        # 确定是否返回字典形式的结果，如果未指定，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 确定是否返回隐藏状态，如果未指定，则使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否返回注意力分数，如果未指定，则使用配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # 使用输入的像素值进行嵌入处理，获取嵌入输出和输入维度信息
        embedding_output, input_dimensions = self.embeddings(pixel_values)

        # 使用编码器对嵌入输出进行编码，获取编码器的输出
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

        # 获取重塑后的隐藏状态
        hidden_states = outputs.reshaped_hidden_states

        # 初始化特征图空元组
        feature_maps = ()
        # 遍历阶段名称和对应的隐藏状态，生成特征图
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                # 获取隐藏状态的形状信息
                batch_size, num_channels, height, width = hidden_state.shape
                # 重新排列维度，便于后续处理
                hidden_state = hidden_state.permute(0, 2, 3, 1).contiguous()
                # 调整形状以应用规范化
                hidden_state = hidden_state.view(batch_size, height * width, num_channels)
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                hidden_state = hidden_state.view(batch_size, height, width, num_channels)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                # 将处理后的隐藏状态添加到特征图中
                feature_maps += (hidden_state,)

        # 如果不返回字典形式的结果，则返回特征图和可能的其他隐藏状态
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        # 返回BackboneOutput对象，包括特征图、隐藏状态和注意力分数（如果有的话）
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
```