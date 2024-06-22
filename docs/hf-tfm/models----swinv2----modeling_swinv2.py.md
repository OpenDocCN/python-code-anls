# `.\transformers\models\swinv2\modeling_swinv2.py`

```py
# 导入所需的模块和类
import collections.abc
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

# 导入 PyTorch 库
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入激活函数
from ...activations import ACT2FN

# 导入模型输出类
from ...modeling_outputs import BackboneOutput

# 导入预训练模型基类
from ...modeling_utils import PreTrainedModel

# 导入 PyTorch 相关工具函数
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer

# 导入辅助函数和类
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# 导入 Swinv2 配置类
from .configuration_swinv2 import Swinv2Config

# 获取日志记录器
logger = logging.get_logger(__name__)

# General docstring
# 用于文档的 Swinv2 配置说明
_CONFIG_FOR_DOC = "Swinv2Config"

# Base docstring
# 用于文档的检查点说明
_CHECKPOINT_FOR_DOC = "microsoft/swinv2-tiny-patch4-window8-256"
# 期望的输出形状
_EXPECTED_OUTPUT_SHAPE = [1, 64, 768]

# Image classification docstring
# 用于文档的图像分类检查点说明
_IMAGE_CLASS_CHECKPOINT = "microsoft/swinv2-tiny-patch4-window8-256"
# 预期的图像分类输出
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"

# Swinv2 预训练模型存档列表
SWINV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/swinv2-tiny-patch4-window8-256",
    # 查看所有 Swinv2 模型：https://huggingface.co/models?filter=swinv2
]

# drop_path, Swinv2PatchEmbeddings, Swinv2PatchMerging 和 Swinv2DropPath 源自 https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer_v2.py.


@dataclass
# 从 transformers.models.swin.modeling_swin.SwinEncoderOutput 复制而来，将 Swin 替换为 Swinv2
class Swinv2EncoderOutput(ModelOutput):
    """
    Swinv2 编码器的输出，可能包含隐藏状态和注意力权重。
```  
    # 定义函数的输入参数，包括最后一层模型输出的隐藏状态、模型各层隐藏状态的元组、注意力权重、重塑后的隐藏状态
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            最后一层模型输出的隐藏状态序列
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型各层隐藏状态的元组，包括嵌入层的输出和每个阶段的输出
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            注意力权重的元组，包括每个阶段的注意力权重
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            重塑后的隐藏状态的元组，包括每个阶段的隐藏状态并包括空间维度
    """
    
    # 初始化函数的返回值
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
from dataclasses import dataclass
# 导入模型输出基类
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Tuple
# 从 torch 库导入必要的数据类型
import torch

@dataclass
# 从 transformers.models.swin.modeling_swin.SwinModelOutput 复制并修改为 Swinv2ModelOutput
class Swinv2ModelOutput(ModelOutput):
    """
    Swinv2 模型的输出，还包含最后隐藏状态的汇集。

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列。
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*, 当传入 `add_pooling_layer=True` 时返回):
            最后一层隐藏状态的平均池化。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当传入 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            `torch.FloatTensor` 元组（一个用于嵌入输出 + 一个用于每个阶段的输出），形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每一层输出的隐藏状态以及初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当传入 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            `torch.FloatTensor` 元组（每个阶段一个）形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当传入 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            `torch.FloatTensor` 元组（一个用于嵌入输出 + 一个用于每个阶段的输出），形状为 `(batch_size, hidden_size, height, width)`。

            模型在每一层输出的隐藏状态以及初始嵌入输出，重塑以包括空间维度。
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# 从 transformers.models.swin.modeling_swin.SwinMaskedImageModelingOutput 复制并修改为 Swinv2MaskedImageModelingOutput
class Swinv2MaskedImageModelingOutput(ModelOutput):
    """
    Swinv2 掩膜图像模型输出。

    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Masked image modeling (MLM) loss.  # 损失函数，用于图像建模
        reconstruction (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Reconstructed pixel values.  # 重构的像素值
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.  # 模型在每层输出的隐藏状态和初始嵌入输出的元组
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.  # 注意力权重，用于计算自注意头部的加权平均值
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.  # 模型在每层输出的隐藏状态和初始嵌入输出的元组，重塑为包含空间维度
    """
    
    loss: Optional[torch.FloatTensor] = None
    reconstruction: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
   
   # 返回重构后的像素值作为属性
    @property
    def logits(self):
        warnings.warn(
            "logits attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the reconstruction attribute to retrieve the final output instead.",
            FutureWarning,
        )
        return self.reconstruction
# 定义 Swinv2ImageClassifierOutput 类，用于表示 Swin V2 模型的图像分类输出
@dataclass
# 从 transformers.models.swin.modeling_swin.SwinImageClassifierOutput 复制并修改名称为 Swinv2ImageClassifierOutput
class Swinv2ImageClassifierOutput(ModelOutput):
    """
    Swinv2 图像分类的输出。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（或回归，如果 config.num_labels==1）损失。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类（或回归，如果 config.num_labels==1）分数（SoftMax 之前）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor` 的元组（一个用于嵌入的输出 + 一个用于每个阶段的输出），形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每个层输出的隐藏状态，加上初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `torch.FloatTensor` 的元组（每个阶段一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于在自注意力头中计算加权平均值。
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor` 的元组（一个用于嵌入的输出 + 一个用于每个阶段的输出），形状为 `(batch_size, hidden_size, height, width)`。

            模型在每个层输出的隐藏状态，加上初始嵌入输出重塑以包括空间维度。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# 从 transformers.models.swin.modeling_swin.window_partition 复制的窗口分割函数
def window_partition(input_feature, window_size):
    """
    将给定输入分割为窗口。
    """
    # 获取输入特征的形状信息
    batch_size, height, width, num_channels = input_feature.shape
    # 重塑输入特征以形成窗口
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    # 重新排列窗口以适应模型要求的形状
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


# 从 transformers.models.swin.modeling_swin.window_reverse 复制的窗口反转函数
def window_reverse(windows, window_size, height, width):
    """
    合并窗口以产生更高分辨率的特征。
    """
    # 获取窗口数组的通道数
    num_channels = windows.shape[-1]
    # 将窗口数组重塑为指定形状
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    # 调整窗口数组的维度顺序
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    # 返回处理后的窗口数组
    return windows
# 从transformers.models.swin.modeling_swin.drop_path中复制得到的drop_path函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    对每个样本进行路径丢弃（随机深度），当应用在残差块的主路径时。

    Ross Wightman的评论：这与我为EfficientNet等网络创建的DropConnect实现相同，但原来的名称是误导性的，因为'Drop Connect'是另一篇论文中的不同形式的dropout...
    参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... 我选择将图层和参数名称更改为'drop path'，而不是将DropConnect作为图层名称，并使用'survival rate'作为参数。
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度的张量，而不仅仅是2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 二值化
    output = input.div(keep_prob) * random_tensor
    return output


# 从transformers.models.swin.modeling_swin.SwinDropPath复制得到，将Swin->Swinv2
class Swinv2DropPath(nn.Module):
    """对每个样本进行路径丢弃（随机深度），当应用在残差块的主路径时。"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# 从transformers.models.swin.modeling_swin.SwinEmbeddings复制得到，将Swin->Swinv2
class Swinv2Embeddings(nn.Module):
    """
    构建补丁和位置嵌入。可选地，也包括掩码令牌。
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.patch_embeddings = Swinv2PatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None

        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        else:
            self.position_embeddings = None

        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, pixel_values: Optional[torch.FloatTensor], bool_masked_pos: Optional[torch.BoolTensor] = None
    # 定义一个方法，接受像素值作为输入，并返回嵌入向量和输出维度的元组
    ) -> Tuple[torch.Tensor]:
        # 使用像素值获取补丁嵌入和输出维度
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        # 对嵌入向量进行标准化处理
        embeddings = self.norm(embeddings)
        # 获取批处理大小、序列长度和嵌入向量维度
        batch_size, seq_len, _ = embeddings.size()

        # 如果存在布尔掩码位置
        if bool_masked_pos is not None:
            # 创建一个与嵌入向量形状相同的掩码词，填充内容为掩码词
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # 用mask_tokens替换掩码的视觉标记
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 如果存在位置嵌入
        if self.position_embeddings is not None:
            # 将位置嵌入添加到嵌入向量上
            embeddings = embeddings + self.position_embeddings

        # 对嵌入向量进行丢弃操作
        embeddings = self.dropout(embeddings)

        # 返回嵌入向量和输出维度
        return embeddings, output_dimensions
# 这个类将输入图像的像素值转换为Transformer的输入隐藏状态(patch embeddings)
class Swinv2PatchEmbeddings(nn.Module):
    """
    这个类将 `pixel_values` 形状为 `(batch_size, num_channels, height, width)` 的输入转换为
    形状为 `(batch_size, seq_length, hidden_size)` 的初始 `hidden_states`(patch embeddings),
    以供Transformer使用。
    """

    def __init__(self, config):
        super().__init__()
        # 从配置中读取图像尺寸、patch尺寸、通道数和隐藏层尺寸
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        # 确保图像和patch的尺寸是可迭代的
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算patch的数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        # 保存一些重要的参数
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        # 使用卷积层将图像转换为patch embeddings
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    # 如果输入图像的尺寸不能被patch尺寸整除,则进行填充
    def maybe_pad(self, pixel_values, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    # 前向传播,将输入图像转换为patch embeddings
    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        # 获取输入图像的形状
        _, num_channels, height, width = pixel_values.shape
        # 检查输入通道数是否与配置中的一致
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 如有必要,对输入图像进行填充
        pixel_values = self.maybe_pad(pixel_values, height, width)
        # 使用卷积层将图像转换为patch embeddings
        embeddings = self.projection(pixel_values)
        # 获取patch embeddings的新尺寸
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        # 将patch embeddings展平并转置
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, output_dimensions


# Patch Merging Layer
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
    # 初始化函数，设置输入分辨率、维度和规范化层
    def __init__(self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 保存输入分辨率和维度
        self.input_resolution = input_resolution
        self.dim = dim
        # 创建一个线性层作为压缩层，将 4*dim 的输入压缩到 2*dim，不使用偏置
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # 使用给定的规范化层对 2*dim 进行规范化
        self.norm = norm_layer(2 * dim)
    
    # 对输入特征进行可能的填充，使其高度和宽度可被整除
    def maybe_pad(self, input_feature, height, width):
        # 判断是否需要填充
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            # 计算填充值
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            # 对输入特征进行填充
            input_feature = nn.functional.pad(input_feature, pad_values)
        return input_feature
    
    # 前向传播函数，将输入特征映射并进行压缩和规范化
    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
        height, width = input_dimensions
        # 获取输入特征的形状信息
        batch_size, dim, num_channels = input_feature.shape
        # 将输入特征进行形状重塑
        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # 对输入特征进行填充，使其高度和宽度可被整除
        input_feature = self.maybe_pad(input_feature, height, width)
        # 提取四个子网格的特征
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # 将四个子网格的特征连接成一个新的特征
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)  # 重新调整形状
        # 使用压缩层进行特征压缩
        input_feature = self.reduction(input_feature)
        # 使用规范化层进行特征规范化
        input_feature = self.norm(input_feature)
        # 返回处理后的特征
        return input_feature
# 创建一个名为Swinv2SelfAttention的类，该类继承自nn.Module
class Swinv2SelfAttention(nn.Module):
    # 定义一个名为transpose_for_scores的方法，用于转换张量的形状
    def transpose_for_scores(self, x):
        # 将输入张量的形状转换为新的形状，其中头的数量和注意力头的尺寸作为新形状的一部分
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 使用新的形状对输入张量进行重新视图，并将结果赋值给x
        x = x.view(new_x_shape)
        # 将x的维度重新排列，以满足注意力矩阵的计算需求
        return x.permute(0, 2, 1, 3)

    # 定义一个名为forward的方法，用于实现自注意力机制的前向传播
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ):
    ) -> Tuple[torch.Tensor]:
        # 获取隐藏状态张量的批量大小、维度和通道数
        batch_size, dim, num_channels = hidden_states.shape
        # 使用self.query对隐藏状态进行查询
        mixed_query_layer = self.query(hidden_states)

        # 使用self.key对隐藏状态进行键转换
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 使用self.value对隐藏状态进行值转换
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 对混合的查询层进行转置
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算余弦注意力得分
        attention_scores = nn.functional.normalize(query_layer, dim=-1) @ nn.functional.normalize(
            key_layer, dim=-1
        ).transpose(-2, -1)
        # 对得分进行缩放
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attention_scores = attention_scores * logit_scale
        # 计算相对位置偏置表
        relative_position_bias_table = self.continuous_position_bias_mlp(self.relative_coords_table).view(
            -1, self.num_attention_heads
        )
        # 重新排列相对位置偏置
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            # 应用注意力掩码
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            ) + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # 将注意力得分规范化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # Dropout注意力概率
        attention_probs = self.dropout(attention_probs)

        # 如果存在head_mask，则应用头部遮罩
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 输出上下文层和注意力概率（如果设置了输出注意力）
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 定义一个名为Swinv2SelfOutput的类，继承自nn.Module类，用于模型自注意力机制输出层
class Swinv2SelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个线性层，输入维度为dim，输出维度为dim
        self.dense = nn.Linear(dim, dim)
        # 创建一个dropout层，使用config中的attention_probs_dropout_prob作为dropout概率
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的hidden_states经过dense层得到输出
        hidden_states = self.dense(hidden_states)
        # 对输出进行dropout操作
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states

# 定义一个名为Swinv2Attention的类，继承自nn.Module类，用于模型的注意力机制
class Swinv2Attention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size, pretrained_window_size=0):
        super().__init__()
        # 初始化self属性为自注意力机制实例Swinv2SelfAttention
        self.self = Swinv2SelfAttention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            pretrained_window_size=pretrained_window_size
            if isinstance(pretrained_window_size, collections.abc.Iterable)
            else (pretrained_window_size, pretrained_window_size),
        )
        # 初始化output属性为Swinv2SelfOutput实例
        self.output = Swinv2SelfOutput(config, dim)
        # 初始化pruned_heads属性为空集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 使用index对线性层进行剪枝
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头信息
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
        # 使用self.self进行自注意力计算，得到self_outputs
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        # 使用output对self_outputs和hidden_states进行操作得到attention_output
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果有需要输出注意力信息，将其加入outputs
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

# 定义一个名为Swinv2Intermediate的类，继承自nn.Module类
class Swinv2Intermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个线性层，输入维度为dim，输出维度为config.mlp_ratio * dim
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        # 根据config中的hidden_act初始化intermediate_act_fn
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
```py 
    # 前向传播函数，接受隐藏状态张量作为输入，返回处理后的隐藏状态张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态张量通过全连接层处理
        hidden_states = self.dense(hidden_states)
        # 使用激活函数对处理后的隐藏状态张量进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的隐藏状态张量
        return hidden_states
# 从transformers.models.swin.modeling_swin.SwinOutput复制代码，将Swin改为Swinv2
class Swinv2Output(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)  # 创建线性层，输入为mlp_ratio*dim，输出为dim
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 创建dropout层，使用配置中的隐藏层dropout概率

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)  # 将隐藏状态传入线性层
        hidden_states = self.dropout(hidden_states)  # 将结果传入dropout层
        return hidden_states  # 返回处理后的隐藏状态


class Swinv2Layer(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0, pretrained_window_size=0):
        super().__init__()
        self.input_resolution = input_resolution
        # 计算窗口大小和移动大小
        window_size, shift_size = self._compute_window_shift(
            (config.window_size, config.window_size), (shift_size, shift_size)
        )
        self.window_size = window_size[0]  # 窗口大小
        self.shift_size = shift_size[0]  # 移动大小
        # 创建Swinv2Attention层
        self.attention = Swinv2Attention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            pretrained_window_size=pretrained_window_size
            if isinstance(pretrained_window_size, collections.abc.Iterable)
            else (pretrained_window_size, pretrained_window_size),
        )
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)  # 创建LayerNorm层
        self.drop_path = Swinv2DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()  # 创建DropPath层
        self.intermediate = Swinv2Intermediate(config, dim)  # 创建Swinv2Intermediate层
        self.output = Swinv2Output(config, dim)  # 创建Swinv2Output层
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)  # 创建LayerNorm层

    # 计算窗口大小和移动大小的方法
    def _compute_window_shift(self, target_window_size, target_shift_size) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]  # 计算窗口大小
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]  # 计算移动大小
        return window_size, shift_size  # 返回计算结果
    # 定义一个方法用于生成注意力遮罩（attention mask），用于自注意力机制
    def get_attn_mask(self, height, width, dtype):
        # 如果设置了窗口偏移量，则计算用于偏移窗口多头自注意力的注意力遮罩
        if self.shift_size > 0:
            # 创建一个全零张量作为图像遮罩，形状为 (1, height, width, 1)，指定数据类型为 dtype
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
            # 定义高度切片和宽度切片用于遮罩窗口的生成
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
            # 初始化遮罩计数
            count = 0
            # 使用双重循环填充图像遮罩
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            # 将图像遮罩分割为窗口，并展平为二维张量
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            # 计算注意力遮罩，表示窗口之间的偏移
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # 将非零值的位置设为负无穷大，将零值位置设为零
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            # 如果未设置窗口偏移量，则返回空的注意力遮罩
            attn_mask = None
        # 返回生成的注意力遮罩
        return attn_mask

    # 方法用于可能的填充（padding），使输入张量尺寸符合窗口大小的整数倍
    def maybe_pad(self, hidden_states, height, width):
        # 计算右边和底部需要填充的大小，使宽度和高度能够被窗口大小整除
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        # 定义填充值，左、右、上、下各方向的填充值
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        # 对隐藏状态进行填充操作，使用 nn.functional.pad() 函数
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        # 返回填充后的隐藏状态及填充参数
        return hidden_states, pad_values

    # 定义前向传播方法，接收隐藏状态张量、输入尺寸、头部遮罩和是否输出注意力信息等参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 获取输入维度
        height, width = input_dimensions
        # 获取隐藏状态的维度
        batch_size, _, channels = hidden_states.size()
        # 对隐藏状态进行备份
        shortcut = hidden_states

        # 将隐藏状态的形状变为(batch_size, height, width, channels)
        hidden_states = hidden_states.view(batch_size, height, width, channels)
        # 对隐藏状态进行填充，使其维度是窗口大小的倍数，并返回填充的值
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)
        # 获取填充后隐藏状态的形状
        _, height_pad, width_pad, _ = hidden_states.shape
        # 如果shift_size大于0，进行循环移动操作（向左上角移动shift_size个位置）
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # 划分窗口
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        # 将窗口reshape为(batch_size * num_windows, window_size * window_size, channels)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        # 获取attention mask
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        # 进行attention计算
        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        attention_output = attention_outputs[0]

        # 将注意力输出的形状转换为(batch_size, height, width, channels)
        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        # 对注意力窗口进行逆转操作，返回(batch_size, height_pad, width_pad, channels)
        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)

        # 如果shift_size大于0，对逆转后的注意力窗口进行循环逆转操作（向右下角移动shift_size个位置）
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        # 判断是否进行了填充
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            # 如果进行了填充，将注意力窗口截取为(batch_size, height, width, channels)
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        # 将注意力窗口reshape为(batch_size, height * width, channels)
        attention_windows = attention_windows.view(batch_size, height * width, channels)
        # 在attention之前进行layer normalization
        hidden_states = self.layernorm_before(attention_windows)
        # 使用残差连接
        hidden_states = shortcut + self.drop_path(hidden_states)

        # 经过一个全连接层
        layer_output = self.intermediate(hidden_states)
        layer_output = self.output(layer_output)
        # 在attention之后进行layer normalization
        layer_output = hidden_states + self.drop_path(self.layernorm_after(layer_output))

        # 返回层输出，并可能返回注意力权重
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs
# 定义 Swinv2Stage 类，继承自 nn.Module
class Swinv2Stage(nn.Module):
    # 初始化函数，定义了 Swinv2Stage 类的属性和参数
    def __init__(
        self, config, dim, input_resolution, depth, num_heads, drop_path, downsample, pretrained_window_size=0
    ):
        super().__init__()  # 调用父类的初始化函数
        self.config = config  # 保存 config 参数到实例属性
        self.dim = dim  # 保存 dim 参数到实例属性
        blocks = []  # 创建空列表用于存储 Swinv2Layer 实例
        # 循环创建 Swinv2Layer 实例，并加入到 blocks 列表中
        for i in range(depth):
            block = Swinv2Layer(
                config=config,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                pretrained_window_size=pretrained_window_size,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)  # 将 blocks 转换为 nn.ModuleList 类型的实例，并保存到实例属性中

        # patch merging layer
        # 如果 downsample 参数不为 None，则创建下采样层实例，保存到实例属性中；否则置为 None
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        self.pointing = False  # 初始化实例属性 pointing 为 False

    # 前向传播函数，定义了 Swinv2Stage 类的前向计算逻辑
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        height, width = input_dimensions  # 获取输入维度信息
        # 遍历 blocks 列表，对每个 Swinv2Layer 实例进行前向计算
        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None  # 获取当前层的注意力头部掩码
            layer_outputs = layer_module(
                hidden_states,
                input_dimensions,
                layer_head_mask,
                output_attentions,
            )  # 调用 Swinv2Layer 实例的前向方法进行计算

            hidden_states = layer_outputs[0]  # 更新隐藏状态

        hidden_states_before_downsampling = hidden_states  # 保存下采样前的隐藏状态
        # 如果存在下采样层，计算下采样后的维度信息，并进行下采样操作
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)  # 输出维度信息
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)  # 执行下采样操作
        else:
            output_dimensions = (height, width, height, width)  # 输出维度信息不变

        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)  # 存储阶段输出信息

        # 如果需要输出注意力信息，则将注意力信息加入输出结果中
        if output_attentions:
            stage_outputs += layer_outputs[1:]

        return stage_outputs  # 返回阶段输出信息
    # 初始化函数，接受配置、网格大小和预训练窗口尺寸参数
    def __init__(self, config, grid_size, pretrained_window_sizes=(0, 0, 0, 0)):
        # 调用父类初始化函数
        super().__init__()
        # 计算层数
        self.num_layers = len(config.depths)
        # 存储配置信息
        self.config = config
        # 如果配置中指定了预训练窗口尺寸，则使用配置中的值
        if self.config.pretrained_window_sizes is not None:
            pretrained_window_sizes = config.pretrained_window_sizes
        # 计算 drop path 的概率值
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        # 存储所有层的列表
        layers = []
        # 遍历每一层
        for i_layer in range(self.num_layers):
            # 创建 Swin Transformer 的每个阶段
            stage = Swinv2Stage(
                config=config,
                # 设置输入的维度
                dim=int(config.embed_dim * 2**i_layer),
                # 计算输入分辨率
                input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                # 设置阶段的深度
                depth=config.depths[i_layer],
                # 设置注意力头的数量
                num_heads=config.num_heads[i_layer],
                # 设置 drop path 的值
                drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                # 如果不是最后一层，则进行下采样，否则为 None
                downsample=Swinv2PatchMerging if (i_layer < self.num_layers - 1) else None,
                # 设置预训练窗口尺寸
                pretrained_window_size=pretrained_window_sizes[i_layer],
            )
            # 将当前阶段添加到列表中
            layers.append(stage)
        # 使用所有阶段创建 nn.ModuleList
        self.layers = nn.ModuleList(layers)

        # 设置梯度检查点为 False
        self.gradient_checkpointing = False

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        return_dict: Optional[bool] = True,
# 从transformers.models.swin.modeling_swin.SwinPreTrainedModel中复制代码，并将Swin->Swinv2,swin->swinv2
class Swinv2PreTrainedModel(PreTrainedModel):
    """
    处理权重初始化和一个简单接口用于下载和加载预训练模型的抽象类。
    """

    config_class = Swinv2Config
    base_model_prefix = "swinv2"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 和 TF 版本稍有不同，TF 版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


SWINV2_START_DOCSTRING = r"""
    这个模型是一个 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 子类。将其用作常规的 PyTorch 模块，并参考 PyTorch 文档以了解与一般使用和行为相关的所有事项。

    参数:
        config ([`Swinv2Config`]): 包含模型所有参数的模型配置类。
            使用配置文件进行初始化不会加载与模型关联的权重，只会加载配置信息。查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

SWINV2_INPUTS_DOCSTRING = r"""
    参数:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。可以使用 [`AutoImageProcessor`] 获取像素值。查看 [`ViTImageProcessor.__call__`] 以获取详细信息。
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            用于使自注意力模块的特定头无效的掩码。掩码值选在 `[0, 1]`：

            - 1 表示头 **不被掩蔽**，
            - 0 表示头 **被掩蔽**。

        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。查看返回张量中的 `attentions` 以获取更多详细信息。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。查看返回张量中的 `hidden_states` 以获取更多详细信息。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是纯元组。
"""


@add_start_docstrings(
    "The bare Swinv2 Model transformer outputting raw hidden-states without any specific head on top.",
    SWINV2_START_DOCSTRING,
)
# 从transformers.models.swin.modeling_swin.SwinModel复制而来，将SWIN改为SWINV2，将Swin改为Swinv2
class Swinv2Model(Swinv2PreTrainedModel):
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        # 调用父类的初始化方法
        super().__init__(config)
        self.config = config
        # 计算层数和特征数量
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        # 初始化嵌入层和编码器
        self.embeddings = Swinv2Embeddings(config, use_mask_token=use_mask_token)
        self.encoder = Swinv2Encoder(config, self.embeddings.patch_grid)

        # 初始化 LayerNorm 和池化层
        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 剪枝模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播函数，接受各种输入并返回模型输出
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



# 此处需要使用 python 和 markdown 语言标记展示注释
    # 定义函数，接收输入参数 pixel_values 和 bool_masked_pos，并返回一个元组或者 Swinv2ModelOutput 对象
    def forward(
        pixel_values: torch.Tensor, 
        bool_masked_pos: Optional[torch.BoolTensor] = None
    ) -> Union[Tuple, Swinv2ModelOutput]:
        """
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # 如果 output_attentions 参数不为空，则使用它；否则使用 self.config.output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 参数不为空，则使用它；否则使用 self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 参数不为空，则使用它；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 如果 pixel_values 为空，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
    
        # 准备头部掩码，如果需要的话
        # head_mask 中的 1.0 表示保留头部
        # attention_probs 的形状是 bsz x n_heads x N x N
        # 输入的 head_mask 的形状是 [num_heads] 或 [num_hidden_layers x num_heads]
        # 并且 head_mask 被转换成形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))
    
        # 通过 embeddings 函数得到嵌入输出和输入维度
        embedding_output, input_dimensions = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        
        # 使用 encoder 处理嵌入输出，得到编码器输出
        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 获取序列输出并进行 layernorm 处理
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
    
        # 如果存在池化器，则进行池化操作
        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)
    
        # 如果 return_dict 为 False，则返回输出元组
        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]
            return output
    
        # 返回 Swinv2ModelOutput 对象，包括最后的隐藏状态��汇聚输出、隐藏状态和注意力值
        return Swinv2ModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )
# 为 Swinv2 模型添加描述，并包含了一个用于遮罩图像建模的解码器，正如 SimMIM 所提出的
# 提示用户可以在示例目录中使用脚本来在自定义数据上预训练此模型
class Swinv2ForMaskedImageModeling(Swinv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 Swinv2 模型
        self.swinv2 = Swinv2Model(config, add_pooling_layer=False, use_mask_token=True)

        # 计算特征数
        num_features = int(config.embed_dim * 2 ** (config.num_layers - 1))
        # 定义解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=num_features, out_channels=config.encoder_stride**2 * config.num_channels, kernel_size=1
            ),
            nn.PixelShuffle(config.encoder_stride),
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # Swinv2 图像建模的前向传播函数
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



# 为 Swinv2 模型添加描述，并包含了一个用于图像分类的线性层的头部（在 [CLS] token 最终隐藏状态之上）
class Swinv2ForImageClassification(Swinv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.swinv2 = Swinv2Model(config)

        # 分类器头部
        self.classifier = (
            nn.Linear(self.swinv2.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # Swinv2 图像分类的前向传播函数
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,  # 输入像素值，类型为可选的浮点型张量
        head_mask: Optional[torch.FloatTensor] = None,  # 头部遮罩，类型为可选的浮点型张量
        labels: Optional[torch.LongTensor] = None,  # 标签，类型为可选的长整型张量
        output_attentions: Optional[bool] = None,  # 是否输出注意力信息，类型为可选的布尔型
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为可选的布尔型
        return_dict: Optional[bool] = None,  # 是否返回结果字典，类型为可选的布尔型
    ) -> Union[Tuple, Swinv2ImageClassifierOutput]:  # 方法返回值的类型注释
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 设置返回结果字典

        outputs = self.swinv2(  # 使用Swin Transformer进行前向传播
            pixel_values,  # 输入像素值
            head_mask=head_mask,  # 头部遮罩
            output_attentions=output_attentions,  # 是否输出注意力信息
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回结果字典
        )

        pooled_output = outputs[1]  # 获取Swin Transformer的输出

        logits = self.classifier(pooled_output)  # 使用分类器对输出进行分类/回归

        loss = None  # 初始化损失值
        if labels is not None:  # 如果存在标签
            if self.config.problem_type is None:  # 如果问题类型未指定
                if self.num_labels == 1:  # 如果标签数为1
                    self.config.problem_type = "regression"  # 设置问题类型为回归
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):  # 如果标签数大于1且标签类型为长整型或整型
                    self.config.problem_type = "single_label_classification"  # 设置问题类型为单标签分类
                else:  # 否则
                    self.config.problem_type = "multi_label_classification"  # 设置问题类型为多标签分类

            if self.config.problem_type == "regression":  # 如果问题类型为回归
                loss_fct = MSELoss()  # 使用均方误差损失
                if self.num_labels == 1:  # 如果标签数为1
                    loss = loss_fct(logits.squeeze(), labels.squeeze())  # 计算损失
                else:  # 否则
                    loss = loss_fct(logits, labels)  # 计算损失
            elif self.config.problem_type == "single_label_classification":  # 如果问题类型为单标签分类
                loss_fct = CrossEntropyLoss()  # 使用交叉熵损失
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 计算损失
            elif self.config.problem_type == "multi_label_classification":  # 如果问题类型为多标签分类
                loss_fct = BCEWithLogitsLoss()  # 使用二元交叉熵损失
                loss = loss_fct(logits, labels)  # 计算损失

        if not return_dict:  # 如果不返回结果字典
            output = (logits,) + outputs[2:]  # 输出结果为logits和Swin Transformer的其他输出
            return ((loss,) + output) if loss is not None else output  # 返回结果加上损失值（如果存在）

        return Swinv2ImageClassifierOutput(  # 返回Swin Transformer图像分类器的输出
            loss=loss,  # 损失值
            logits=logits,  # 分类器的输出
            hidden_states=outputs.hidden_states,  # 隐藏状态
            attentions=outputs.attentions,  # 注意力信息
            reshaped_hidden_states=outputs.reshaped_hidden_states,  # 重塑后的隐藏状态
        )
@add_start_docstrings(
    """
    Swinv2 backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    SWINV2_START_DOCSTRING,
)
# Swinv2Backbone 类，继承自 Swinv2PreTrainedModel 和 BackboneMixin
class Swinv2Backbone(Swinv2PreTrainedModel, BackboneMixin):
    # 初始化函数
    def __init__(self, config):
        # 调用父类 Swinv2PreTrainedModel 的初始化函数
        super().__init__(config)
        # 调用父类 BackboneMixin 的初始化函数
        super()._init_backbone(config)

        # 计算特征维度列表，从 patch_embeddings 的维度开始，以 config.depths 的长度逐步扩大
        self.num_features = [config.embed_dim] + [int(config.embed_dim * 2**i) for i in range(len(config.depths))]
        # 创建 Swinv2Embeddings 对象
        self.embeddings = Swinv2Embeddings(config)
        # 创建 Swinv2Encoder 对象，传入 patch_grid 参数
        self.encoder = Swinv2Encoder(config, self.embeddings.patch_grid)

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入的函数
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 前向传播函数
    @add_start_docstrings_to_model_forward(SWINV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义函数返回的类型是 BackboneOutput
    ) -> BackboneOutput:
        """
        # 这段注释提供了函数的返回信息和示例代码
        Returns:
    
        Examples:
    
        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests
    
        # 下载示例图像，并使用 PIL 打开
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
    
        # 使用预训练的处理器和骨干网络模型
        >>> processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
        >>> model = AutoBackbone.from_pretrained(
        ...     "microsoft/swinv2-tiny-patch4-window8-256", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )
    
        # 将图像处理为张量形式
        >>> inputs = processor(image, return_tensors="pt")
    
        # 通过模型生成特征图
        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        # 打印输出特征图的形状
        >>> list(feature_maps[-1].shape)
        [1, 2048, 7, 7]
        ```py"""
        
        # 如果没有提供 return_dict 参数，使用默认配置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果没有提供 output_hidden_states 参数，使用默认配置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有提供 output_attentions 参数，使用默认配置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    
        # 调用 self.embeddings() 方法，获取嵌入和输入的维度
        embedding_output, input_dimensions = self.embeddings(pixel_values)
    
        # 使用嵌入的输出和其他配置参数进行编码
        outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            return_dict=return_dict,
        )
    
        # 根据 return_dict 的值来获取隐藏状态
        hidden_states = outputs.reshaped_hidden_states if return_dict else outputs[-1]
    
        # 初始化空的特征图元组
        feature_maps = ()
        # 遍历所有阶段的名称和对应的隐藏状态
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            # 如果当前阶段在 out_features 中，则将其加入特征图
            if stage in self.out_features:
                feature_maps += (hidden_state,)
    
        # 如果 return_dict 为 False，创建返回值元组
        if not return_dict:
            output = (feature_maps,)
            # 如果需要输出隐藏状态，则将其加入输出
            if output_hidden_states:
                output += (outputs[1],)
            # 如果需要输出注意力机制结果，则将其加入输出
            if output_attentions:
                output += (outputs[2],)
            return output
    
        # 否则，返回 BackboneOutput 对象，包含特征图、隐藏状态和注意力机制输出
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
```