# `.\transformers\models\maskformer\modeling_maskformer_swin.py`

```py
# 引入 Python 标准库中的 collections.abc 模块，用于处理集合类数据类型的抽象基类
import collections.abc
# 引入数学库中的 math 模块，用于执行数学计算
import math
# 引入 dataclasses 模块中的 dataclass 装饰器，用于创建不可变的数据类
from dataclasses import dataclass
# 引入 typing 模块，用于类型提示
from typing import Optional, Tuple

# 引入 PyTorch 库中的 Tensor 和 nn 模块
import torch
from torch import Tensor, nn

# 引入 HuggingFace 库中的激活函数映射模块 ACT2FN、文件工具模块和模型输出模块
from ...activations import ACT2FN
from ...file_utils import ModelOutput
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils.backbone_utils import BackboneMixin
# 引入 MaskFormer Swin Transformer 配置类
from .configuration_maskformer_swin import MaskFormerSwinConfig

# 使用 dataclass 装饰器定义一个数据类，用于表示 MaskFormer Swin 模型的输出，包含隐藏状态的空间维度信息
@dataclass
class MaskFormerSwinModelOutputWithPooling(ModelOutput):
    """
    Class for MaskFormerSwinModel's outputs that also contains the spatial dimensions of the hidden states.
    """
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state after a mean pooling operation.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        hidden_states_spatial_dimensions (`tuple(tuple(int, int))`, *optional*):
            A tuple containing the spatial dimension of each `hidden_state` needed to reshape the `hidden_states` to
            `batch, channels, height, width`. Due to padding, their spatial size cannot be inferred before the
            `forward` method.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义变量并初始化为 None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_spatial_dimensions: Tuple[Tuple[int, int]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类，用于表示SwinEncoder的输出结果
@dataclass
class MaskFormerSwinBaseModelOutput(ModelOutput):
    """
    Class for SwinEncoder's outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        hidden_states_spatial_dimensions (`tuple(tuple(int, int))`, *optional*):
            A tuple containing the spatial dimension of each `hidden_state` needed to reshape the `hidden_states` to
            `batch, channels, height, width`. Due to padding, their spatial size cannot inferred before the `forward`
            method.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_spatial_dimensions: Tuple[Tuple[int, int]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 从transformers.models.swin.modeling_swin.window_partition中复制的函数
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    # 获取输入特征的形状信息
    batch_size, height, width, num_channels = input_feature.shape
    # 将输入特征按照窗口大小进行划分
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    # 调整窗口顺序
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


# 从transformers.models.swin.modeling_swin.window_reverse中复制的函数
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    # 获取窗口的��道数
    num_channels = windows.shape[-1]
    # 将窗口合并为更高分辨率的特征
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    return windows


# 从transformers.models.swin.modeling_swin.drop_path中复制的函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop path implementation.
    """
    # 对每个样本进行路径丢弃（随机深度），应用于残差块的主路径中
    # 作者 Ross Wightman 的注释：这与我为 EfficientNet 等网络创建的 DropConnect 实现相同，
    # 但原始名称具有误导性，因为“Drop Connect”是另一篇论文中不同形式的丢失连接...
    # 参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... 我选择将层和参数名称更改为“drop path”，而不是混合使用 DropConnect 作为层名称，并将“survival rate”用作参数。
    """
    # 如果丢失概率为 0.0 或者不处于训练状态，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留概率
    keep_prob = 1 - drop_prob
    # 计算形状，适用于不同维度的张量，而不仅仅是 2D ConvNets
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    # 生成随机张量，用于二值化
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 向下取整，二值化
    # 计算输出，通过随机张量进行路径丢弃
    output = input.div(keep_prob) * random_tensor
    return output
class MaskFormerSwinEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings.
    """

    def __init__(self, config):
        super().__init__()

        # 初始化 PatchEmbeddings 对象
        self.patch_embeddings = MaskFormerSwinPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size

        # 根据配置选择是否使用绝对位置编码
        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        else:
            self.position_embeddings = None

        # 初始化 LayerNorm 和 Dropout 层
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values):
        # 获取 PatchEmbeddings 的输出和维度
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)

        # 如果使用绝对位置编码，则加上位置编码
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings, output_dimensions


# Copied from transformers.models.swin.modeling_swin.SwinPatchEmbeddings
class MaskFormerSwinPatchEmbeddings(nn.Module):
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
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        # 使用卷积层将像素值转换为 patch embeddings
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def maybe_pad(self, pixel_values, height, width):
        # 如果宽度不能整除 patch 大小，则进行填充
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        # 如果高度不能整除 patch 大小，则进行填充
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values
    # 前向传播函数，接受像素值作为输入，返回嵌入向量和输出维度
    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        # 获取像素值的形状信息
        _, num_channels, height, width = pixel_values.shape
        # 检查通道数是否与配置中设置的通道数匹配
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 如果需要，对输入进行填充，使其能够被 self.patch_size 整除
        pixel_values = self.maybe_pad(pixel_values, height, width)
        # 将像素值投影到嵌入空间
        embeddings = self.projection(pixel_values)
        # 获取嵌入向量的形状信息
        _, _, height, width = embeddings.shape
        # 记录输出的高度和宽度
        output_dimensions = (height, width)
        # 将嵌入向量展平并转置
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, output_dimensions
# 定义一个名为 MaskFormerSwinPatchMerging 的类，继承自 nn.Module
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
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    # 检查是否需要对输入进行填充
    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)

        return input_feature

    # 前向传播函数
    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
        height, width = input_dimensions
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # 对输入进行填充，使其可以被宽度和高度整除
        input_feature = self.maybe_pad(input_feature, height, width)
        # 提取四个子块的特征
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # 将四个子块的特征拼接在一起
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)  # batch_size height/2*width/2 4*C

        input_feature = self.norm(input_feature)
        input_feature = self.reduction(input_feature)

        return input_feature


# 定义一个名为 MaskFormerSwinDropPath 的类，继承自 nn.Module
class MaskFormerSwinDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    # ��向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    # 返回额外的描述信息
    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)
# 从transformers.models.swin.modeling_swin.SwinSelfAttention复制代码，并将Swin->MaskFormerSwin
class MaskFormerSwinSelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        # 如果隐藏大小（dim）不能被注意力头数（num_heads）整除，则引发错误
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )

        # 初始化相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        # 获取窗口内每个标记的成对相对位置索引
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

        # 初始化查询、键、值的线性层
        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)

        # 初始化dropout层
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
        # 定义函数的输入和输出类型
        ) -> Tuple[torch.Tensor]:
        # 获取隐藏状态的批量大小、维度和通道数
        batch_size, dim, num_channels = hidden_states.shape
        # 使用查询网络处理隐藏状态，得到混合的查询层
        mixed_query_layer = self.query(hidden_states)

        # 使用键网络处理隐藏状态，然后转置以便计算注意力分数
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 使用值网络处理隐藏状态，然后转置以便计算注意力分数
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 转置混合的查询层以便计算注意力分数
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始的注意力分数，即查询和键的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 获取相对位置偏置表，并根据相对位置索引获取相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )

        # 调整相对位置偏置的维度顺序
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # 将相对位置偏置添加到注意力分数中
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            # 应用预先计算的注意力掩码
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用dropout进行注意力概率的处理
        attention_probs = self.dropout(attention_probs)

        # 如果有头部掩码，则将其应用到注意力概率中
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，即注意力概率与值层的乘积
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据输出注意力的需求返回不同的结果
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 从transformers.models.swin.modeling_swin.SwinSelfOutput复制代码，并将Swin->MaskFormerSwin
class MaskFormerSwinSelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个线性层，输入和输出维度都为dim
        self.dense = nn.Linear(dim, dim)
        # 创建一个dropout层，使用config中的attention_probs_dropout_prob作为dropout概率
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将hidden_states传入线性层
        hidden_states = self.dense(hidden_states)
        # 对线性层的输出进行dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# 从transformers.models.swin.modeling_swin.SwinAttention复制代码，并将Swin->MaskFormerSwin
class MaskFormerSwinAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        # 创建MaskFormerSwinSelfAttention对象
        self.self = MaskFormerSwinSelfAttention(config, dim, num_heads, window_size)
        # 创建MaskFormerSwinSelfOutput对象
        self.output = MaskFormerSwinSelfOutput(config, dim)
        # 初始化一个空集合，用于存储被剪枝的头
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可剪枝的头和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数��存储被剪枝的头
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
        # 调用self的forward方法，传入参数并返回结果
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        # 将self的输出传入output层
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则将其添加到outputs中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# 从transformers.models.swin.modeling_swin.SwinIntermediate复制代码，并将Swin->MaskFormerSwin
class MaskFormerSwinIntermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个线性层，输入维度为dim，输出维度为config.mlp_ratio * dim
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        # 如果config.hidden_act是字符串，则使用ACT2FN字典中对应的激活函数，否则直接使用config.hidden_act作为激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 定义一个前向传播函数，接受隐藏状态作为输入，并返回处理后的隐藏状态
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 使用激活函数对线性变换后的隐藏状态进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.swin.modeling_swin.SwinOutput复制代码，并将Swin->MaskFormerSwin
class MaskFormerSwinOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个线性层，输入维度为config.mlp_ratio * dim，输出维度为dim
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        # 创建一个Dropout层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入hidden_states传入线性层
        hidden_states = self.dense(hidden_states)
        # 将线性层的输出传入Dropout层
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的hidden_states
        return hidden_states


class MaskFormerSwinLayer(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0):
        super().__init__()
        # 初始化MaskFormerSwinLayer，设置shift_size、window_size、input_resolution等属性
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution
        # 创建LayerNorm层，输入维度为dim，eps为config.layer_norm_eps
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建MaskFormerSwinAttention层
        self.attention = MaskFormerSwinAttention(config, dim, num_heads, self.window_size)
        # 创建DropPath层，如果drop_path_rate大于0，则使用MaskFormerSwinDropPath，否则使用nn.Identity()
        self.drop_path = (
            MaskFormerSwinDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        )
        # 创建LayerNorm层，输入维度为dim，eps为config.layer_norm_eps
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建MaskFormerSwinIntermediate层
        self.intermediate = MaskFormerSwinIntermediate(config, dim)
        # 创建MaskFormerSwinOutput层
        self.output = MaskFormerSwinOutput(config, dim)

    def get_attn_mask(self, input_resolution):
        if self.shift_size > 0:
            # 计算SW-MSA的注意力掩码
            height, width = input_resolution
            img_mask = torch.zeros((1, height, width, 1))
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
    # 在需要的情况下对隐藏状态进行填充，使其高度和宽度符合指定的窗口大小
    def maybe_pad(self, hidden_states, height, width):
        # 初始化左侧和顶部填充值为0
        pad_left = pad_top = 0
        # 计算右侧填充值，确保宽度是窗口大小的整数倍
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        # 计算底部填充值，确保高度是窗口大小的整数倍
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        # 组合填充值，格式为(top, bottom, left, right)
        pad_values = (0, 0, pad_left, pad_right, pad_top, pad_bottom)
        # 使用填充值对隐藏状态进行填充
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        # 返回填充后的隐藏状态和填充值
        return hidden_states, pad_values
    # 前向传播函数，接收隐藏状态、输入维度、头部掩码和是否输出注意力权重
    def forward(self, hidden_states, input_dimensions, head_mask=None, output_attentions=False):
        # 解包输入维度
        height, width = input_dimensions
        # 获取隐藏状态的批量大小、维度和通道数
        batch_size, dim, channels = hidden_states.size()
        # 保存隐藏状态的快捷方式
        shortcut = hidden_states

        # 对隐藏状态进行 LayerNormalization 处理
        hidden_states = self.layernorm_before(hidden_states)
        # 将隐藏状态重塑为(batch_size, height, width, channels)的形状
        hidden_states = hidden_states.view(batch_size, height, width, channels)
        # 对隐藏状态进行填充，使其大小为窗口大小的倍数
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        _, height_pad, width_pad, _ = hidden_states.shape
        # 如果存在循环移位
        if self.shift_size > 0:
            # 对隐藏状态进行循环移位
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # 划分窗口
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        # 获取注意力掩码
        attn_mask = self.get_attn_mask((height_pad, width_pad))
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        # 进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力

        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        # 反转窗口
        shifted_windows = window_reverse(
            attention_windows, self.window_size, height_pad, width_pad
        )  # B height' width' C

        # 如果存在循环移位
        if self.shift_size > 0:
            # 对注意力窗口进行反向循环移位
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        # 判断是否进行了填充
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            # 如果进行了填充，则截取注意力窗口
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)

        # 将快捷方式和注意力窗口相加，并进行 DropPath 操作
        hidden_states = shortcut + self.drop_path(attention_windows)

        # LayerNormalization
        layer_output = self.layernorm_after(hidden_states)
        # 进行中间层计算
        layer_output = self.intermediate(layer_output)
        # 添加残差连接和输出层计算
        layer_output = hidden_states + self.output(layer_output)

        outputs = (layer_output,) + outputs

        return outputs
# 定义 MaskFormerSwinStage 类，继承自 nn.Module
class MaskFormerSwinStage(nn.Module):
    # 初始化函数，接受配置、维度、输入分辨率、深度、头数、drop_path 和 downsample 参数
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        # 调用父类的初始化函数
        super().__init__()
        # 保存配置、维度等参数
        self.config = config
        self.dim = dim
        # 创建 nn.ModuleList，包含多个 MaskFormerSwinLayer 对象
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

        # 如果 downsample 不为 None，则创建 downsample 层
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        # 初始化 pointing 属性为 False

    # 前向传播函数，接受隐藏状态、输入维度、头部掩码等参数
    def forward(
        self, hidden_states, input_dimensions, head_mask=None, output_attentions=False, output_hidden_states=False
    ):
        # 如果需要输出隐藏状态，则初始化 all_hidden_states 为空元组
        all_hidden_states = () if output_hidden_states else None

        # 获取输入维度的高度和宽度
        height, width = input_dimensions
        # 遍历每个 block 模块
        for i, block_module in enumerate(self.blocks):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 调用 block_module 进行前向传播
            block_hidden_states = block_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)

            # 更新隐藏状态为 block_hidden_states 的第一个元素
            hidden_states = block_hidden_states[0]

            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        # 如果存在 downsample 层
        if self.downsample is not None:
            # 计算下采样后的高度和宽度
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            # 对隐藏状态进行下采样操作
            hidden_states = self.downsample(hidden_states, input_dimensions)
        else:
            output_dimensions = (height, width, height, width)

        # 返回隐藏状态、输出维度和所有隐藏状态
        return hidden_states, output_dimensions, all_hidden_states


# 定义 MaskFormerSwinEncoder 类，暂时未提供具体实现
class MaskFormerSwinEncoder(nn.Module):
    # 未提供具体实现
    # 初始化函数，接受配置和网格大小作为参数
    def __init__(self, config, grid_size):
        # 调用父类的初始化函数
        super().__init__()
        # 计算层数
        self.num_layers = len(config.depths)
        # 保存配置信息
        self.config = config
        # 生成一系列的 drop path rate
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        # 创建多层 MaskFormerSwinStage 模块
        self.layers = nn.ModuleList(
            [
                MaskFormerSwinStage(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    downsample=MaskFormerSwinPatchMerging if (i_layer < self.num_layers - 1) else None,
                )
                for i_layer in range(self.num_layers)
            ]
        )

        # 是否启用梯度检查点
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
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出隐藏状态，则初始化一个空元组，否则设为None
        all_input_dimensions = ()
        # 初始化一个空元组，用于存储输入维度信息
        all_self_attentions = () if output_attentions else None
        # 如果需要输出注意力矩阵，则初始化一个空元组，否则设为None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        # 如果需要输出隐藏状态，则将当前隐藏状态添加到all_hidden_states元组中

        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的头部掩码，如果没有则设为None

            if self.gradient_checkpointing and self.training:
                layer_hidden_states, output_dimensions, layer_all_hidden_states = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_hidden_states, output_dimensions, layer_all_hidden_states = layer_module(
                    hidden_states,
                    input_dimensions,
                    layer_head_mask,
                    output_attentions,
                    output_hidden_states,
                )
            # 根据是否启用梯度检查点和训练状态，调用不同的函数计算当前层的隐藏状态、输出维度和所有隐藏状态

            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            all_input_dimensions += (input_dimensions,)
            # 更新输入维度信息

            if output_hidden_states:
                all_hidden_states += (layer_all_hidden_states,)
            # 如果需要输出隐藏状态，则将当前层的所有隐藏状态添加到all_hidden_states元组中

            hidden_states = layer_hidden_states
            # 更新当前隐藏状态

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_all_hidden_states[1],)
            # 如果需要输出注意力矩阵，则将当前层的注意力矩阵添加到all_self_attentions元组中

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 如果不需要返回字典形式的结果，则返回包含非空值的元组

        return MaskFormerSwinBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            hidden_states_spatial_dimensions=all_input_dimensions,
            attentions=all_self_attentions,
        )
        # 返回MaskFormerSwinBaseModelOutput对象，包含最终隐藏状态、所有隐藏状态、输入维度信息和注意力矩阵
# 从transformers.models.swin.modeling_swin.SwinPreTrainedModel复制而来，将Swin->MaskFormerSwin，swin->model
class MaskFormerSwinPreTrainedModel(PreTrainedModel):
    """
    一个处理权重初始化和简单接口用于下载和加载预训练模型的抽象类。
    """

    config_class = MaskFormerSwinConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 与TF版本略有不同，TF版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class MaskFormerSwinModel(MaskFormerSwinPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        self.embeddings = MaskFormerSwinEmbeddings(config)
        self.encoder = MaskFormerSwinEncoder(config, self.embeddings.patch_grid)

        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        剪枝模型的注意力头。heads_to_prune: {layer_num: 要在该层剪枝的头列表} 参见基类PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):
            # 如果未指定output_attentions，则使用配置中的output_attentions
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # 如果未指定output_hidden_states，则使用配置中的output_hidden_states
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 如果未指定return_dict，则使用配置中的use_return_dict
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if pixel_values is None:
                # 如果未指定pixel_values，则抛出数值错误
                raise ValueError("You have to specify pixel_values")

            # 准备头部掩码（head mask）如果需要
            # head_mask中的1.0表示保留该头部
            # attention_probs的形状为bsz x n_heads x N x N
            # 输入的head_mask的形状为[num_heads]或[num_hidden_layers x num_heads]
            # 并且head_mask被转换为形状[num_hidden_layers x batch x num_heads x seq_length x seq_length]
            head_mask = self.get_head_mask(head_mask, len(self.config.depths))

            # 将像素值嵌入到模型中
            embedding_output, input_dimensions = self.embeddings(pixel_values)

            # 编码器的输出
            encoder_outputs = self.encoder(
                embedding_output,
                input_dimensions,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            # 如果return_dict为False，则使用最后一个隐藏状态作为序列输出
            sequence_output = encoder_outputs.last_hidden_state if return_dict else encoder_outputs[0]
            # 对序列输出进行layernorm
            sequence_output = self.layernorm(sequence_output)

            pooled_output = None
            if self.pooler is not None:
                # 如果存在pooler，则对序列输出进行池化
                pooled_output = self.pooler(sequence_output.transpose(1, 2))
                pooled_output = torch.flatten(pooled_output, 1)

            if not return_dict:
                # 如果return_dict为False，则返回序列输出、池化输出以及其他编码器输出
                return (sequence_output, pooled_output) + encoder_outputs[1:]

            # 计算隐藏状态的空间维度
            hidden_states_spatial_dimensions = (input_dimensions,) + encoder_outputs.hidden_states_spatial_dimensions

            return MaskFormerSwinModelOutputWithPooling(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                hidden_states_spatial_dimensions=hidden_states_spatial_dimensions,
                attentions=encoder_outputs.attentions,
            )
class MaskFormerSwinBackbone(MaskFormerSwinPreTrainedModel, BackboneMixin):
    """
    MaskFormerSwin backbone, designed especially for the MaskFormer framework.

    This classes reshapes `hidden_states` from (`batch_size, sequence_length, hidden_size)` to (`batch_size,
    num_channels, height, width)`). It also adds additional layernorms after each stage.

    Args:
        config (`MaskFormerSwinConfig`):
            The configuration used by [`MaskFormerSwinModel`].
    """

    def __init__(self, config: MaskFormerSwinConfig):
        # 调用父类的构造函数，传入配置参数
        super().__init__(config)
        # 初始化骨干网络
        super()._init_backbone(config)

        # 创建 MaskFormerSwinModel 模型
        self.model = MaskFormerSwinModel(config)
        # 检查是否支持 'stem' 在 `out_features` 中
        if "stem" in self.out_features:
            raise ValueError("This backbone does not support 'stem' in the `out_features`.")
        # 计算特征维度
        self.num_features = [config.embed_dim] + [int(config.embed_dim * 2**i) for i in range(len(config.depths)]
        # 创建多个 layernorm 模块
        self.hidden_states_norms = nn.ModuleList(
            [nn.LayerNorm(num_channels) for num_channels in self.num_features[1:]]
        )

        # 初始化权重并应用最终处理
        self.post_init()

    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> BackboneOutput:
            # 设置返回字典，如果未指定则使用配置中的默认值
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            # 设置是否输出隐藏状态，如果未指定则使用配置中的默认值
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 设置是否输出注意力，如果未指定则使用配置中的默认值
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

            # 使用模型进行推理
            outputs = self.model(
                pixel_values, output_hidden_states=True, output_attentions=output_attentions, return_dict=True
            )

            # 跳过 stem 部分
            hidden_states = outputs.hidden_states[1:]

            # 需要将隐藏状态重塑为原始空间维度
            # 空间维度包含每个阶段的所有高度和宽度，包括嵌入后
            spatial_dimensions: Tuple[Tuple[int, int]] = outputs.hidden_states_spatial_dimensions
            feature_maps = ()
            for i, (hidden_state, stage, (height, width)) in enumerate(
                zip(hidden_states, self.stage_names[1:], spatial_dimensions)
            ):
                norm = self.hidden_states_norms[i]
                # 最后一个元素对应于层的最后一个块输出，但在合并补丁之前
                hidden_state_unpolled = hidden_state[-1]
                hidden_state_norm = norm(hidden_state_unpolled)
                # 像素解码器（FPN）期望 3D 张量（特征）
                batch_size, _, hidden_size = hidden_state_norm.shape
                # 重塑 "b (h w) d -> b d h w"
                hidden_state_permuted = (
                    hidden_state_norm.permute(0, 2, 1).view((batch_size, hidden_size, height, width)).contiguous()
                )
                if stage in self.out_features:
                    feature_maps += (hidden_state_permuted,)

            # 如果不返回字典，则返回元组
            if not return_dict:
                output = (feature_maps,)
                if output_hidden_states:
                    output += (outputs.hidden_states,)
                if output_attentions:
                    output += (outputs.attentions,)
                return output

            # 返回 BackboneOutput 对象
            return BackboneOutput(
                feature_maps=feature_maps,
                hidden_states=outputs.hidden_states if output_hidden_states else None,
                attentions=outputs.attentions,
            )
```