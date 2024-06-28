# `.\models\swinv2\modeling_swinv2.py`

```py
# 设置文件编码为 UTF-8
# 版权声明：2022 年由 Microsoft Research 和 The HuggingFace Inc. 团队保留所有权利
#
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依据“原样”提供，不提供任何明示或暗示的保证或条件
# 请参阅许可证获取详细信息
""" PyTorch Swinv2 Transformer model."""

import collections.abc  # 导入集合抽象基类，用于类型检查
import math  # 导入数学库，用于数学计算
import warnings  # 导入警告模块，用于处理警告
from dataclasses import dataclass  # 导入 dataclass 装饰器，用于创建数据类
from typing import Optional, Tuple, Union  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 模块，用于实现模型的内存优化
from torch import Tensor, nn  # 导入 PyTorch 的张量和神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入损失函数

from ...activations import ACT2FN  # 导入激活函数映射
from ...modeling_outputs import BackboneOutput  # 导入模型输出类
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer  # 导入模型工具函数
from ...utils import (
    ModelOutput,  # 导入模型输出基类
    add_code_sample_docstrings,  # 导入用于添加代码示例文档字符串的函数
    add_start_docstrings,  # 导入用于添加起始文档字符串的函数
    add_start_docstrings_to_model_forward,  # 导入用于模型前向方法的起始文档字符串函数
    logging,  # 导入日志模块
    replace_return_docstrings,  # 导入用于替换返回文档字符串的函数
)
from ...utils.backbone_utils import BackboneMixin  # 导入骨干网络相关的工具函数
from .configuration_swinv2 import Swinv2Config  # 导入 Swinv2 模型的配置类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 用于文档的配置文件名
_CONFIG_FOR_DOC = "Swinv2Config"

# 用于文档的检查点信息
_CHECKPOINT_FOR_DOC = "microsoft/swinv2-tiny-patch4-window8-256"

# 预期输出形状的说明
_EXPECTED_OUTPUT_SHAPE = [1, 64, 768]

# 图像分类检查点信息
_IMAGE_CLASS_CHECKPOINT = "microsoft/swinv2-tiny-patch4-window8-256"

# 图像分类预期输出的示例
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"

# Swinv2 预训练模型的存档列表
SWINV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/swinv2-tiny-patch4-window8-256",
    # 可在 https://huggingface.co/models?filter=swinv2 查看所有 Swinv2 模型
]

# 以下定义部分来自 https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer_v2.py.

@dataclass
# 从 transformers.models.swin.modeling_swin.SwinEncoderOutput 复制并将 Swin->Swinv2
class Swinv2EncoderOutput(ModelOutput):
    """
    Swinv2 编码器的输出，可能包含隐藏状态和注意力权重。
    # 最后一层模型的隐藏状态，形状为(batch_size, sequence_length, hidden_size)
    last_hidden_state: torch.FloatTensor = None
    # 模型每一层的隐藏状态的元组，形状为(batch_size, sequence_length, hidden_size)，可选项，当`output_hidden_states=True`时返回
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 注意力权重的元组，形状为(batch_size, num_heads, sequence_length, sequence_length)，可选项，当`output_attentions=True`时返回
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 模型每一层的隐藏状态的元组，形状为(batch_size, hidden_size, height, width)，包括空间维度，可选项，当`output_hidden_states=True`且输出被重塑时返回
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
# 使用 dataclass 装饰器定义 Swinv2ModelOutput 类，它继承自 ModelOutput 类
# ModelOutput 是一个基础类，可能在 transformers 库中定义
@dataclass
# 从 transformers.models.swin.modeling_swin.SwinModelOutput 复制的类定义，将 Swin 替换为 Swinv2
class Swinv2ModelOutput(ModelOutput):
    """
    Swinv2 模型的输出，同时包含最后隐藏状态的池化结果。

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列输出。
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*, 当 `add_pooling_layer=True` 时返回):
            最后一层隐藏状态的平均池化结果。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            包含模型每一层隐藏状态的元组，以及初始嵌入输出。
            形状为 `(batch_size, sequence_length, hidden_size)`。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            自注意力机制 softmax 后的注意力权重，用于计算自注意力头的加权平均值。
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            包含模型每一层隐藏状态的元组，以及初始嵌入输出，重塑为包含空间维度的形状。
            形状为 `(batch_size, hidden_size, height, width)`。

"""


@dataclass
# 从 transformers.models.swin.modeling_swin.SwinMaskedImageModelingOutput 复制的类定义，将 Swin 替换为 Swinv2
class Swinv2MaskedImageModelingOutput(ModelOutput):
    """
    Swinv2 掩码图像模型的输出。

    这个类定义可能还需要填充完整，以匹配 Swinv2 模型的具体输出内容和结构。
    通常来说，这些数据类定义了模型输出的结构，包括各个部分的详细说明。
    你可以根据实际的 Swinv2 模型输出来进一步补充这个类的内容。
    
    例如，可以包括类似于上面 Swinv2ModelOutput 类的参数说明，描述具体的模型输出内容和形状。
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Masked image modeling (MLM) loss.
            图像模型的掩码损失（MLM损失）。
        reconstruction (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Reconstructed pixel values.
            重建的像素数值。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            模型在每一层输出的隐藏状态，包括初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力权重经过注意力softmax后的结果，用于计算自注意力头中的加权平均。
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
            模型在每一层输出的隐藏状态，包括重塑以包括空间维度的初始嵌入输出。

    """

    # 定义属性loss，类型为Optional[torch.FloatTensor]，默认值为None
    loss: Optional[torch.FloatTensor] = None
    # 定义属性reconstruction，类型为torch.FloatTensor，默认值为None
    reconstruction: torch.FloatTensor = None
    # 定义属性hidden_states，类型为Optional[Tuple[torch.FloatTensor, ...]]，默认值为None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 定义属性attentions，类型为Optional[Tuple[torch.FloatTensor, ...]]，默认值为None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 定义属性reshaped_hidden_states，类型为Optional[Tuple[torch.FloatTensor, ...]]，默认值为None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None

    @property
    def logits(self):
        # 警告信息，提醒logits属性在Transformers的版本5中将被移除，建议使用reconstruction属性获取最终输出。
        warnings.warn(
            "logits attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the reconstruction attribute to retrieve the final output instead.",
            FutureWarning,
        )
        # 返回属性reconstruction的值作为logits属性的输出
        return self.reconstruction
@dataclass
# 从transformers.models.swin.modeling_swin.SwinImageClassifierOutput复制到Swinv2ImageClassifierOutput
class Swinv2ImageClassifierOutput(ModelOutput):
    """
    Swinv2图像分类的输出。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, 当提供`labels`时返回):
            分类（如果config.num_labels==1则是回归）损失。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类（如果config.num_labels==1则是回归）得分（SoftMax之前）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当`output_hidden_states=True`时返回或者当`config.output_hidden_states=True`时返回):
            包含每层输出的`torch.FloatTensor`元组，形状为`(batch_size, sequence_length, hidden_size)`。

            每个层的模型隐藏状态加上初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当`output_attentions=True`时返回或者当`config.output_attentions=True`时返回):
            包含每个阶段`torch.FloatTensor`元组，形状为`(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力softmax后的注意力权重，用于计算自注意力头的加权平均值。
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当`output_hidden_states=True`时返回或者当`config.output_hidden_states=True`时返回):
            包含每层输出的`torch.FloatTensor`元组，形状为`(batch_size, hidden_size, height, width)`。

            每个层的模型隐藏状态加上初始嵌入输出，重塑以包含空间维度。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


# 从transformers.models.swin.modeling_swin.window_partition复制
def window_partition(input_feature, window_size):
    """
    将给定输入分区为窗口。
    """
    batch_size, height, width, num_channels = input_feature.shape
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


# 从transformers.models.swin.modeling_swin.window_reverse复制
def window_reverse(windows, window_size, height, width):
    """
    合并窗口以产生更高分辨率的特征。
    """
    # 获取窗口数组的通道数量
    num_channels = windows.shape[-1]
    # 将窗口数组重塑为指定窗口大小的网格结构
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    # 对重塑后的窗口数组进行维度置换，以便重新排列窗口的顺序
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
    # 再次将重排后的窗口数组展平为原始形状
    windows = windows.view(-1, height, width, num_channels)
    # 返回重新排列和重塑后的窗口数组
    return windows
# Copied from transformers.models.swin.modeling_swin.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    # 如果 drop_prob 为 0 或者不处于训练模式，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留概率
    keep_prob = 1 - drop_prob
    # 确定随机张量的形状
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # 生成均匀分布的随机张量，并进行二值化处理
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    # 对输入进行按元素除法，并应用二值化的随机张量
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.swin.modeling_swin.SwinDropPath with Swin->Swinv2
class Swinv2DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用 drop_path 函数，传递当前实例的 drop_prob 属性和训练模式
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# Copied from transformers.models.swin.modeling_swin.SwinEmbeddings with Swin->Swinv2
class Swinv2Embeddings(nn.Module):
    """
    Construct the patch and position embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__()

        # 初始化 Swinv2PatchEmbeddings 实例
        self.patch_embeddings = Swinv2PatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        # 如果 use_mask_token 为真，则初始化一个用于掩码的张量参数
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None

        # 根据配置决定是否初始化位置编码张量参数
        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        else:
            self.position_embeddings = None

        # 初始化 LayerNorm 层和 Dropout 层
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, pixel_values: Optional[torch.FloatTensor], bool_masked_pos: Optional[torch.BoolTensor] = None
    ):
        # 省略了 forward 方法的其余部分，用于构造图像块和位置编码的嵌入
        pass
    ) -> Tuple[torch.Tensor]:
        # 获取图像块的嵌入表示和输出维度信息
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        # 对嵌入表示进行归一化处理
        embeddings = self.norm(embeddings)
        # 获取批处理大小、序列长度以及嵌入表示的最后一个维度大小
        batch_size, seq_len, _ = embeddings.size()

        # 如果存在掩码位置信息
        if bool_masked_pos is not None:
            # 使用mask_token在整个批次上扩展以替换掩码的视觉标记
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # 创建掩码，使其类型与mask_tokens一致，并在嵌入表示中应用
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 如果存在位置嵌入，则将其加到嵌入表示中
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        # 对嵌入表示进行dropout处理
        embeddings = self.dropout(embeddings)

        # 返回处理后的嵌入表示和输出维度信息
        return embeddings, output_dimensions
# Copied from transformers.models.swin.modeling_swin.SwinPatchEmbeddings with Swin->Swinv2
class Swinv2PatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        # Extract configuration parameters
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        # Ensure image_size and patch_size are iterable
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # Calculate number of patches
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        # Initialize instance variables
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        # Projection layer: Conv2d for patch embedding
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def maybe_pad(self, pixel_values, height, width):
        # Pad pixel_values if height or width is not divisible by patch_size
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        # Retrieve dimensions of pixel_values
        _, num_channels, height, width = pixel_values.shape
        # Check if number of channels matches self.num_channels
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # Pad input pixel_values to ensure divisibility by patch_size
        pixel_values = self.maybe_pad(pixel_values, height, width)
        # Project pixel_values into patch embeddings
        embeddings = self.projection(pixel_values)
        # Retrieve dimensions of embeddings after projection
        _, _, height, width = embeddings.shape
        # Flatten embeddings and transpose dimensions for further processing
        embeddings = embeddings.flatten(2).transpose(1, 2)

        # Return embeddings and output dimensions
        return embeddings, (height, width)


class Swinv2PatchMerging(nn.Module):
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
        self.input_resolution = input_resolution  # 设置输入分辨率
        self.dim = dim  # 设置维度
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 初始化线性变换层，减少维度
        self.norm = norm_layer(2 * dim)  # 初始化规范化层

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)  # 检查是否需要填充
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)  # 计算填充值
            input_feature = nn.functional.pad(input_feature, pad_values)  # 执行填充操作

        return input_feature

    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
        height, width = input_dimensions  # 解析输入尺寸
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape  # 获取输入特征的形状信息

        input_feature = input_feature.view(batch_size, height, width, num_channels)  # 重新组织输入特征的形状
        # pad input to be disible by width and height, if needed
        input_feature = self.maybe_pad(input_feature, height, width)  # 调用填充函数，确保特征是宽高可整除的
        # [batch_size, height/2, width/2, num_channels]
        input_feature_0 = input_feature[:, 0::2, 0::2, :]  # 提取特征的子块1
        # [batch_size, height/2, width/2, num_channels]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]  # 提取特征的子块2
        # [batch_size, height/2, width/2, num_channels]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]  # 提取特征的子块3
        # [batch_size, height/2, width/2, num_channels]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]  # 提取特征的子块4
        # [batch_size, height/2 * width/2, 4*num_channels]
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)  # 将四个子块合并
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)  # 重新组织合并后的特征形状

        input_feature = self.reduction(input_feature)  # 执行线性变换
        input_feature = self.norm(input_feature)  # 执行规范化操作

        return input_feature  # 返回处理后的特征
# 定义一个名为Swinv2SelfAttention的自定义神经网络模块类
class Swinv2SelfAttention(nn.Module):
    # 定义一个用于将输入张量x转换为注意力分数形状的方法
    def transpose_for_scores(self, x):
        # 计算新的张量形状，保留除了最后一维外的所有维度，并增加注意力头数和每个头的大小
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 重新调整张量的形状
        x = x.view(new_x_shape)
        # 对调张量的维度顺序，将第0和第2个维度互换，第1和第3个维度互换
        return x.permute(0, 2, 1, 3)

    # 定义前向传播方法，接受隐藏状态张量、注意力掩码、头部掩码和输出注意力的可选参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 获取输入张量的维度信息
        batch_size, dim, num_channels = hidden_states.shape
        # 使用 self.query 对隐藏状态进行查询操作，生成混合的查询层
        mixed_query_layer = self.query(hidden_states)

        # 使用 self.key 对隐藏状态进行键操作，并转置以便计算注意力分数
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 使用 self.value 对隐藏状态进行值操作，并转置以便后续计算上下文层
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 对混合的查询层也进行转置以便计算注意力分数
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 使用余弦相似度计算注意力分数
        attention_scores = nn.functional.normalize(query_layer, dim=-1) @ nn.functional.normalize(
            key_layer, dim=-1
        ).transpose(-2, -1)

        # 对注意力分数进行缩放，使用 torch.clamp 限制缩放因子的最大值
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attention_scores = attention_scores * logit_scale

        # 使用 MLP 模块计算相对位置偏置，并重新组织形状以匹配注意力分数
        relative_position_bias_table = self.continuous_position_bias_mlp(self.relative_coords_table).view(
            -1, self.num_attention_heads
        )
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        # 如果存在注意力遮罩，则将其应用于注意力分数
        if attention_mask is not None:
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            ) + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # 将注意力分数归一化为注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力概率进行 dropout
        attention_probs = self.dropout(attention_probs)

        # 如果存在头部掩码，则将其应用于注意力概率
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算最终的上下文层，将注意力概率与值层相乘
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # 调整上下文层的形状以符合输出的预期形状
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据输出设置，构造最终输出结果
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# Copied from transformers.models.swin.modeling_swin.SwinSelfOutput with Swin->Swinv2
class Swinv2SelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 定义一个线性层，输入和输出维度都为 dim
        self.dense = nn.Linear(dim, dim)
        # 定义一个 Dropout 层，使用配置中的注意力概率作为丢弃概率
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过线性层进行变换
        hidden_states = self.dense(hidden_states)
        # 应用 Dropout 进行随机丢弃
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class Swinv2Attention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size, pretrained_window_size=0):
        super().__init__()
        # 初始化自注意力层对象，传入配置、维度、头数、窗口大小等参数
        self.self = Swinv2SelfAttention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            pretrained_window_size=pretrained_window_size
            if isinstance(pretrained_window_size, collections.abc.Iterable)
            else (pretrained_window_size, pretrained_window_size),
        )
        # 初始化自注意力输出层对象，传入配置和维度参数
        self.output = Swinv2SelfOutput(config, dim)
        # 初始化被修剪的注意力头集合为空集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 寻找可修剪的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪后的注意力头
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
        # 调用自注意力层的前向传播函数
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        # 将自注意力输出作为输入，通过自注意力输出层进行变换
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果输出注意力权重，将它们添加到输出元组中
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinIntermediate with Swin->Swinv2
class Swinv2Intermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 定义一个线性层，输入维度为 dim，输出维度为 config.mlp_ratio * dim
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        # 如果配置中的隐藏层激活函数是字符串，使用对应的激活函数；否则使用配置中的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 定义神经网络的前向传播函数，接受隐藏状态作为输入张量，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态应用激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的隐藏状态张量作为输出
        return hidden_states
# 从 transformers.models.swin.modeling_swin.SwinOutput 复制代码，并将类名中的 Swin 改为 Swinv2
class Swinv2Output(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个全连接层，将输入的特征维度缩放为 config.mlp_ratio * dim
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        # 定义一个 dropout 层，用于随机丢弃神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 全连接层计算
        hidden_states = self.dense(hidden_states)
        # dropout 操作
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Swinv2Layer 类定义
class Swinv2Layer(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0, pretrained_window_size=0):
        super().__init__()
        # 计算窗口大小和位移大小，确保它们不超过输入分辨率
        window_size, shift_size = self._compute_window_shift(
            (config.window_size, config.window_size), (shift_size, shift_size)
        )
        # 设置当前层的窗口大小和位移大小
        self.window_size = window_size[0]
        self.shift_size = shift_size[0]
        
        # 创建 Swinv2Attention 层，用于执行注意力机制
        self.attention = Swinv2Attention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            pretrained_window_size=pretrained_window_size
                if isinstance(pretrained_window_size, collections.abc.Iterable)
                else (pretrained_window_size, pretrained_window_size),
        )
        
        # LayerNorm 层，用于归一化输入数据
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        
        # 如果设置了 drop path rate，则创建 Swinv2DropPath 层；否则创建一个恒等映射（Identity）层
        self.drop_path = Swinv2DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        
        # Swinv2Intermediate 类，用于中间层的计算
        self.intermediate = Swinv2Intermediate(config, dim)
        
        # Swinv2Output 类，用于最终输出的全连接层
        self.output = Swinv2Output(config, dim)
        
        # LayerNorm 层，用于归一化输出数据
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)

    def _compute_window_shift(self, target_window_size, target_shift_size) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        # 计算适应于输入分辨率的窗口大小和位移大小
        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
        return window_size, shift_size
    # 返回注意力掩码
    def get_attn_mask(self, height, width, dtype):
        if self.shift_size > 0:
            # 为了实现窗口移位的多头自注意力机制，计算注意力掩码
            # 创建一个全零张量作为初始掩码
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
            # 定义高度和宽度的切片，用于生成窗口之外的掩码
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
            # 遍历高度和宽度切片，为每个窗口分配唯一的计数值
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            # 将整个图像分成窗口，并展平为二维数组
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            # 创建注意力掩码，基于窗口计数之间的差异
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # 将非零值的位置设为-100.0，零值位置设为0.0
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            # 如果不需要窗口移位，返回空的注意力掩码
            attn_mask = None
        return attn_mask

    # 对隐藏状态进行可能的填充，使其高度和宽度可被窗口大小整除
    def maybe_pad(self, hidden_states, height, width):
        # 计算需要在右侧和底部填充的像素数，以确保整数倍窗口大小
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        # 定义填充的数值（左、右、上、下）
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        # 使用 PyTorch 的函数进行填充操作
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    # 前向传播函数，接受隐藏状态张量和输入维度，可选的头部掩码和输出注意力
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 解析输入维度
        height, width = input_dimensions
        # 解析隐藏状态的批处理大小、高度、宽度、通道数
        batch_size, _, channels = hidden_states.size()
        # 保存隐藏状态的快捷方式
        shortcut = hidden_states

        # 将隐藏状态重新形状为(batch_size, height, width, channels)
        hidden_states = hidden_states.view(batch_size, height, width, channels)
        # 可能对隐藏状态进行填充以使其成为窗口大小的倍数，并获取填充的值
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)
        # 获取填充后的高度和宽度
        _, height_pad, width_pad, _ = hidden_states.shape

        # 如果设定了shift_size，则进行循环移位操作
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # 将移位后的隐藏状态分割成窗口
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        # 将窗口重新形状为(-1, self.window_size * self.window_size, channels)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)

        # 获取注意力掩码
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        # 如果注意力掩码存在，则将其移动到hidden_states_windows的设备上
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        # 应用注意力机制
        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        # 获取注意力输出
        attention_output = attention_outputs[0]

        # 将注意力输出重新形状为(-1, self.window_size, self.window_size, channels)
        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)

        # 将窗口反转恢复为原始形状
        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)

        # 如果设定了shift_size，则反转循环移位操作
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        # 检查是否进行了填充，如果是，则截取有效部分
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        # 将窗口形状重新调整为(batch_size, height * width, channels)
        attention_windows = attention_windows.view(batch_size, height * width, channels)

        # 通过layernorm进行前处理
        hidden_states = self.layernorm_before(attention_windows)
        # 添加shortcut并进行drop_path操作
        hidden_states = shortcut + self.drop_path(hidden_states)

        # 应用中间层操作
        layer_output = self.intermediate(hidden_states)
        # 应用输出层操作
        layer_output = self.output(layer_output)
        # 应用layernorm后处理并添加drop_path
        layer_output = hidden_states + self.drop_path(self.layernorm_after(layer_output))

        # 如果需要输出注意力信息，则返回注意力输出
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        # 返回层输出
        return layer_outputs
# 定义 Swinv2Stage 类，作为 Swin Transformer V2 模型的一个阶段
class Swinv2Stage(nn.Module):
    # 初始化方法
    def __init__(
        self, config, dim, input_resolution, depth, num_heads, drop_path, downsample, pretrained_window_size=0
    ):
        super().__init__()
        self.config = config  # 保存配置参数
        self.dim = dim  # 特征维度
        blocks = []
        # 循环创建指定数量的 Swinv2Layer 块
        for i in range(depth):
            # 创建 Swinv2Layer 块并添加到 blocks 列表中
            block = Swinv2Layer(
                config=config,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                pretrained_window_size=pretrained_window_size,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)  # 将 blocks 转为 nn.ModuleList

        # 如果有下采样层，则初始化下采样方法
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        self.pointing = False  # 初始化指向状态为 False

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        height, width = input_dimensions  # 获取输入图片的高度和宽度
        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None  # 获取当前层的注意力掩码

            # 调用每个 Swinv2Layer 块的 forward 方法进行前向传播
            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                layer_head_mask,
                output_attentions,
            )

            hidden_states = layer_outputs[0]  # 更新隐藏状态为当前层的输出

        hidden_states_before_downsampling = hidden_states  # 保存下采样前的隐藏状态
        if self.downsample is not None:
            # 计算下采样后的图片尺寸
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)  # 输出尺寸信息
            # 调用下采样方法对隐藏状态进行下采样处理
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)
        else:
            output_dimensions = (height, width, height, width)  # 如果没有下采样，输出尺寸不变

        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)  # 阶段输出信息

        if output_attentions:
            stage_outputs += layer_outputs[1:]  # 如果需要输出注意力信息，则将其添加到输出中
        return stage_outputs  # 返回阶段的输出结果
    # 初始化函数，用于创建一个 Swin Transformer 模型
    def __init__(self, config, grid_size, pretrained_window_sizes=(0, 0, 0, 0)):
        # 调用父类的初始化方法
        super().__init__()
        # 计算模型的层数
        self.num_layers = len(config.depths)
        # 将配置信息保存到对象中
        self.config = config
        # 如果配置中指定了预训练窗口大小，则使用配置中的值
        if self.config.pretrained_window_sizes is not None:
            pretrained_window_sizes = config.pretrained_window_sizes
        # 生成一个按照 config.drop_path_rate 线性分布的列表，并转换成 Python 列表
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        # 初始化一个空列表用于保存每个阶段的 Swin Transformer 层
        layers = []
        # 遍历每个层
        for i_layer in range(self.num_layers):
            # 创建一个 Swin Transformer 的阶段（stage）
            stage = Swinv2Stage(
                config=config,
                # 设置当前层的维度大小为 config.embed_dim * 2^i_layer
                dim=int(config.embed_dim * 2**i_layer),
                # 设置输入分辨率为原始网格大小除以 2^i_layer
                input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                # 设置当前层的深度为 config.depths[i_layer]
                depth=config.depths[i_layer],
                # 设置当前层的注意力头数为 config.num_heads[i_layer]
                num_heads=config.num_heads[i_layer],
                # 设置当前层的 drop path 策略
                drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                # 如果不是最后一层，则进行下采样
                downsample=Swinv2PatchMerging if (i_layer < self.num_layers - 1) else None,
                # 设置当前层的预训练窗口大小
                pretrained_window_size=pretrained_window_sizes[i_layer],
            )
            # 将当前创建的阶段加入到层列表中
            layers.append(stage)
        # 将所有的阶段组成的层列表转换为 nn.ModuleList 类型，并保存到对象的 layers 属性中
        self.layers = nn.ModuleList(layers)

        # 默认关闭梯度检查点
        self.gradient_checkpointing = False

    # 前向传播函数，定义了模型的前向计算逻辑
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        return_dict: Optional[bool] = True,
# 从transformers.models.swin.modeling_swin.SwinPreTrainedModel复制的代码，并将Swin->Swinv2,swin->swinv2
class Swinv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定该模型使用的配置类
    config_class = Swinv2Config
    # 基础模型的前缀名称
    base_model_prefix = "swinv2"
    # 主要输入的名称
    main_input_name = "pixel_values"
    # 支持梯度检查点的标志
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用正态分布初始化权重，平均值为0，标准差为self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是LayerNorm层
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置项为零
            module.bias.data.zero_()
            # 初始化权重为1
            module.weight.data.fill_(1.0)


# SWINV2_START_DOCSTRING文档字符串
SWINV2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Swinv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# SWINV2_INPUTS_DOCSTRING文档字符串
SWINV2_INPUTS_DOCSTRING = r"""
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
    "The bare Swinv2 Model transformer outputting raw hidden-states without any specific head on top.",
    SWINV2_START_DOCSTRING,
)
# 从transformers.models.swin.modeling_swin.SwinModel复制并修改为Swinv2Model，SWIN->SWINV2，Swin->Swinv2
class Swinv2Model(Swinv2PreTrainedModel):
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        super().__init__(config)
        self.config = config
        # 计算模型的层数和特征维度
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        # 初始化嵌入层和编码器
        self.embeddings = Swinv2Embeddings(config, use_mask_token=use_mask_token)
        self.encoder = Swinv2Encoder(config, self.embeddings.patch_grid)

        # 初始化层归一化和池化层（如果指定添加池化层）
        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回输入嵌入的Patch嵌入层
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        对模型的注意力头进行剪枝。
        heads_to_prune: {layer_num: 需要在该层剪枝的头列表} 参见基类PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(SWINV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Swinv2ModelOutput,
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
        # 输入参数详见SWINV2_INPUTS_DOCSTRING，传入像素值、布尔掩码位置、头掩码等信息
        # 返回Swinv2ModelOutput类型的预期输出
        ) -> Union[Tuple, Swinv2ModelOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # 根据需要设定是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据需要设定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据需要设定是否使用返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            # 如果未提供像素值，则抛出数值错误
            raise ValueError("You have to specify pixel_values")

        # 准备头部遮罩（如果需要）
        # head_mask 中的 1.0 表示保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # 将 head_mask 转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))

        # 嵌入层输出和输入尺寸
        embedding_output, input_dimensions = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        # 编码器输出
        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 序列输出
        sequence_output = encoder_outputs[0]
        # 序列输出进行 LayerNorm 处理
        sequence_output = self.layernorm(sequence_output)

        pooled_output = None
        if self.pooler is not None:
            # 如果存在池化器，则计算池化输出
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)

        if not return_dict:
            # 如果不使用返回字典形式，则返回元组形式的输出
            output = (sequence_output, pooled_output) + encoder_outputs[1:]
            return output

        # 使用 Swinv2ModelOutput 类构建返回字典形式的输出
        return Swinv2ModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )
@add_start_docstrings(
    """
    Swinv2 Model with a decoder on top for masked image modeling, as proposed in
    [SimMIM](https://arxiv.org/abs/2111.09886).

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>
    """,
    SWINV2_START_DOCSTRING,
)
# 定义 Swinv2ForMaskedImageModeling 类，用于进行面向掩膜图像建模的解码器模型
# 该类基于 Swinv2PreTrainedModel，并包含了 Swinv2 模型和一个解码器
class Swinv2ForMaskedImageModeling(Swinv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 Swinv2 模型，设置不添加池化层并使用掩膜令牌
        self.swinv2 = Swinv2Model(config, add_pooling_layer=False, use_mask_token=True)

        # 计算特征数量用于解码器
        num_features = int(config.embed_dim * 2 ** (config.num_layers - 1))
        
        # 定义解码器的结构
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=num_features, out_channels=config.encoder_stride**2 * config.num_channels, kernel_size=1
            ),
            nn.PixelShuffle(config.encoder_stride),
        )

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(SWINV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Swinv2MaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    # 重写 forward 方法，接收输入并返回 Swinv2MaskedImageModelingOutput 类型的输出
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 输入参数详见 SWINV2_INPUTS_DOCSTRING



@add_start_docstrings(
    """
    Swinv2 Model transformer with an image classification head on top (a linear layer on top of the final hidden state
    of the [CLS] token) e.g. for ImageNet.
    """,
    SWINV2_START_DOCSTRING,
)
# 定义 Swinv2ForImageClassification 类，用于图像分类的 Swinv2 模型
# 该类基于 Swinv2PreTrainedModel，并包含了 Swinv2 模型和分类器头部
class Swinv2ForImageClassification(Swinv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 设置类别数量
        self.num_labels = config.num_labels

        # 初始化 Swinv2 模型
        self.swinv2 = Swinv2Model(config)

        # 分类器头部
        self.classifier = (
            nn.Linear(self.swinv2.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(SWINV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=Swinv2ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 重写 forward 方法，接收输入并返回 Swinv2ImageClassifierOutput 类型的输出
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,  # 输入的像素值张量，可选
        head_mask: Optional[torch.FloatTensor] = None,     # 头部掩码张量，可选
        labels: Optional[torch.LongTensor] = None,          # 图像分类/回归的标签张量，可选
        output_attentions: Optional[bool] = None,           # 是否输出注意力张量，可选
        output_hidden_states: Optional[bool] = None,        # 是否输出隐藏状态张量，可选
        return_dict: Optional[bool] = None,                 # 是否返回字典形式的输出，可选
    ) -> Union[Tuple, Swinv2ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确定是否使用返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用Swin Transformer模型进行前向传播
        outputs = self.swinv2(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取池化后的输出
        pooled_output = outputs[1]

        # 使用分类器对池化后的输出进行分类得到logits
        logits = self.classifier(pooled_output)

        # 初始化损失值为None
        loss = None

        # 如果存在标签
        if labels is not None:
            # 如果问题类型未定义，则根据标签数据类型和类别数目设置问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失函数
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

        # 如果不使用返回字典形式的输出，则将logits与其他输出合并返回
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 使用Swinv2ImageClassifierOutput类封装输出并返回
        return Swinv2ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )
@add_start_docstrings(
    """
    Swinv2 backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    SWINV2_START_DOCSTRING,
)
class Swinv2Backbone(Swinv2PreTrainedModel, BackboneMixin):
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)
        # 调用父类的_backbone初始化方法
        super()._init_backbone(config)

        # 计算特征维度列表，从config.embed_dim开始，按2的幂级增加，直到config.depths的长度
        self.num_features = [config.embed_dim] + [int(config.embed_dim * 2**i) for i in range(len(config.depths))]
        
        # 初始化Swinv2的嵌入层
        self.embeddings = Swinv2Embeddings(config)
        
        # 初始化Swinv2的编码器，传入嵌入层的patch grid
        self.encoder = Swinv2Encoder(config, self.embeddings.patch_grid)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回patch embeddings作为输入嵌入
        return self.embeddings.patch_embeddings

    @add_start_docstrings_to_model_forward(SWINV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        """
        根据给定的参数返回 BackboneOutput 对象。

        参数:
            return_dict (bool, optional): 是否返回字典形式的输出，默认为使用配置中的设定。
            output_hidden_states (bool, optional): 是否输出隐藏状态，默认为使用配置中的设定。
            output_attentions (bool, optional): 是否输出注意力权重，默认为使用配置中的设定。

        返回:
            BackboneOutput: 包含特征图、隐藏状态和注意力权重的对象。

        示例:

        ```
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        >>> model = AutoBackbone.from_pretrained(
        ...     "microsoft/swinv2-tiny-patch4-window8-256", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 2048, 7, 7]
        ```
        """
        # 如果 return_dict 为 None，则使用配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 output_hidden_states 为 None，则使用配置中的设定
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 output_attentions 为 None，则使用配置中的设定
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # 调用 self.embeddings 处理输入像素值，获得嵌入输出和输入尺寸
        embedding_output, input_dimensions = self.embeddings(pixel_values)

        # 调用 self.encoder 处理嵌入输出和输入尺寸，返回输出结果
        outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            return_dict=return_dict,
        )

        # 根据是否返回字典决定取得隐藏状态的方式
        hidden_states = outputs.reshaped_hidden_states if return_dict else outputs[-1]

        # 初始化空元组用于存储特征图
        feature_maps = ()
        # 遍历阶段名称和隐藏状态，如果阶段在输出特征列表中，则加入特征图元组
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                feature_maps += (hidden_state,)

        # 如果不返回字典形式的结果，则按照指定顺序组装输出元组
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs[1],)
            if output_attentions:
                output += (outputs[2],)
            return output

        # 返回 BackboneOutput 对象，包含特征图、隐藏状态和注意力权重
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
```