# `.\models\donut\modeling_donut_swin.py`

```
# 指定字符编码为 utf-8
# 版权声明
# 基于 Apache License, Version 2.0 授权
# 详细可参考 http://www.apache.org/licenses/LICENSE-2.0
# 软件分发基于 "AS IS" 的基础上，没有任何形式的明示或默示的担保或条件
# 请查看 License 以了解具体语言的规定和限制
""" PyTorch Donut Swin Transformer model.
# 引入需要的模块
import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
# 引入 HuggingFace 通用模块
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_donut_swin import DonutSwinConfig
# 获取 logger
logger = logging.get_logger(__name__)
# 通用文档字符串
_CONFIG_FOR_DOC = "DonutSwinConfig"
# 基本文档字符串
_CHECKPOINT_FOR_DOC = "https://huggingface.co/naver-clova-ix/donut-base"
_EXPECTED_OUTPUT_SHAPE = [1, 49, 768]
# Donut Swin 预训练模型列表
DONUT_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "naver-clova-ix/donut-base",
    # 查看所有的 Donut Swin 模型 https://huggingface.co/models?filter=donut
]
@dataclass
# DonutSwin encoder 输出，包含潜在的隐藏状态和注意力
class DonutSwinEncoderOutput(ModelOutput):
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            模型最后一层的输出的隐藏状态序列。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.
            模型每一层输出的隐藏状态的元组，包括嵌入层输出和每个阶段的输出。

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            模型在每一层输出的隐藏状态，加上初始嵌入层的输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            每个阶段的注意力权重的元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.
            模型每一层输出的隐藏状态的元组，包括嵌入层输出和每个阶段的输出，形状为 `(batch_size, hidden_size, height, width)`。

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
            模型在每一层输出的隐藏状态，加上初始嵌入层的输出，重塑以包括空间维度。
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
from dataclasses import dataclass
# 引入 ModelOutput 类
from transformers.modeling_outputs import ModelOutput
# 引入 torch 库
import torch
# 引入 Optional 类型
from typing import Optional, Tuple

# 使用 @dataclass 装饰器声明一个数据类，用于表示 DonutSwin 模型的输出
@dataclass
# 从 transformers.models.swin.modeling_swin.SwinModelOutput 复制并修改为 DonutSwin
class DonutSwinModelOutput(ModelOutput):
    """
    DonutSwin model's outputs that also contains a pooling of the last hidden states.

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

    # 定义模型输出的属性
    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# 从 transformers.models.swin.modeling_swin.window_partition 复制
# 定义函数用于将输入特征分割成窗口
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    # 获取输入特征的形状信息
    batch_size, height, width, num_channels = input_feature.shape
    # 重新组织输入特征，将其划分成窗口
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    # 转置和重塑以获得窗口
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    # 返回窗口
    return windows


# 从 transformers.models.swin.modeling_swin.window_reverse 复制
# 定义函数用于反转窗口操作
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    # 获取窗口的通道数
    num_channels = windows.shape[-1]
    # 调整窗口形状，将其变为四维张量，其中前两维表示窗口的高度和宽度除以窗口大小，最后一维表示通道数
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    # 对窗口进行维度置换，将原本的高度、宽度和窗口大小的顺序调整为样本、高度、窗口大小、宽度、窗口大小、通道数
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    # 返回调整后的窗口
    return windows
# 从transformers.models.swin.modeling_swin.SwinEmbeddings复制并替换为DonutSwinEmbeddings
class DonutSwinEmbeddings(nn.Module):
    """
    构建补丁和位置嵌入。 可选择是否包含蒙版令牌。
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.patch_embeddings = DonutSwinPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches  # 获取补丁嵌入的数量
        self.patch_grid = self.patch_embeddings.grid_size  # 获取补丁网格的大小
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None  # 根据条件创建蒙版令牌

        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))  # 如果使用绝对嵌入，创建位置嵌入参数
        else:
            self.position_embeddings = None  # 否则位置嵌入参数为空

        self.norm = nn.LayerNorm(config.embed_dim)  # 创建 LayerNorm 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 创建 Dropout 层

    def forward(
        self, pixel_values: Optional[torch.FloatTensor], bool_masked_pos: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)  # 获取补丁嵌入
        embeddings = self.norm(embeddings)  # 归一化嵌入
        batch_size, seq_len, _ = embeddings.size()  # 获取批次大小和序列长度

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)  # 根据条件创建蒙版令牌
            # 用蒙版令牌替换被屏蔽的视觉令牌
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        if self.position_embeddings is not None:  # 如果位置嵌入参数不为空
            embeddings = embeddings + self.position_embeddings  # 添加位置嵌入到嵌入中

        embeddings = self.dropout(embeddings)  # 使用 Dropout 层

        return embeddings, output_dimensions  # 返回嵌入和输出维度


# 从transformers.models.swin.modeling_swin.SwinPatchEmbeddings复制并替换为DonutSwinPatchEmbeddings
class DonutSwinPatchEmbeddings(nn.Module):
    """
    此类将形状为`(batch_size, num_channels, height, width)`的`pixel_values`转换为形状为`(batch_size, seq_length, hidden_size)`的初始`hidden_states`（补丁嵌入），以供Transformer使用。
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])  # 计算补丁数量
        self.image_size = image_size  # 设置图像大小
        self.patch_size = patch_size  # 设置补丁大小
        self.num_channels = num_channels  # 设置通道数量
        self.num_patches = num_patches  # 设置补丁数量
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])  # 设置网格大小

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)  # 创建卷积层
    # 根据给定的像素值和高度、宽度，对像素值进行填充，使其能够被 self.patch_size 整除
    def maybe_pad(self, pixel_values, height, width):
        # 如果宽度不能被 self.patch_size[1] 整除，计算需要填充的值，并对像素值进行填充
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        # 如果高度不能被 self.patch_size[0] 整除，计算需要填充的值，并对像素值进行填充
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        # 返回经过填充后的像素值
        return pixel_values

    # 前向传播函数，接受像素值作为输入，返回嵌入值和输出维度
    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        # 获取像素值的形状，并检查通道数是否匹配配置中的通道数
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 对输入进行填充，使其能够被 self.patch_size 整除
        pixel_values = self.maybe_pad(pixel_values, height, width)
        # 通过 projection 对像素值进行投影，得到嵌入值
        embeddings = self.projection(pixel_values)
        # 获取嵌入值的形状，并对其进行扁平化和转置
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        # 返回嵌入值和输出维度
        return embeddings, output_dimensions
# 从transformers.models.swin.modeling_swin.SwinPatchMerging获取
class DonutSwinPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            输入特征的分辨率。
        dim (`int`):
            输入通道数。
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            标准化层类。
    """

    def __init__(self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 将输入特征维度减少的线性层
        self.norm = norm_layer(4 * dim)  # 参数维度的标准化层

    def maybe_pad(self, input_feature, height, width):  # 如果需要的话对输入特征进行填充
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = nn.functional.pad(input_feature, pad_values)

        return input_feature

    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:  # 前向传播
        height, width = input_dimensions  # 输入特征的高和宽
        # `dim` 是高度*宽度
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # 如果需要的话对输入进行填充
        input_feature = self.maybe_pad(input_feature, height, width)
        # [batch_size, height/2, width/2, num_channels]
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # batch_size height/2 width/2 4*num_channels
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)  # 连接四个输入特征
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)  # batch_size height/2*width/2 4*C

        input_feature = self.norm(input_feature)  # 对输入特征进行标准化
        input_feature = self.reduction(input_feature)  # 对输入特征进行减少维度

        return input_feature


# 从transformers.models.beit.modeling_beit.drop_path获取
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    为每个样本丢弃路径（随机深度）（当应用于残差块的主路径时）。

    Ross Wightman的评论：这与我为EfficientNet等网络创建的Drop Connect实现相同，但原始名称是具有误导性的，因为'Drop Connect'是另一篇论文中的一种不同形式的退出...
    请参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... 我选择更改
    # 如果 dropout 概率为 0 或者不处于训练模式，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留节点的概率
    keep_prob = 1 - drop_prob
    # 确定输出张量的形状，支持不同维度的张量，不仅限于二维卷积神经网络
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    # 生成与输入张量相同形状的随机张量，用于决定每个节点的存活与否
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    # 将随机张量二值化，即保留节点的概率转换为二值化概率
    random_tensor.floor_()
    # 计算输出，将输入张量除以保留节点的概率，再乘以随机张量
    output = input.div(keep_prob) * random_tensor
    # 返回处理后的输出
    return output
# 定义 DonutSwinDropPath 类，用于每个样本的路径上应用随机深度（Stochastic Depth）（当应用在残差块的主路径上时）
class DonutSwinDropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用 drop_path 函数对隐藏状态进行处理，传入的参数为 hidden_states、self.drop_prob 和 self.training
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# 定义 DonutSwinSelfAttention 类，继承自 nn.Module 类
class DonutSwinSelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        if dim % num_heads != 0:
            # 如果隐藏大小（dim）不能被注意力头数（num_heads）整除，则抛出错误
            raise ValueError(f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})")

        # 初始化各个属性
        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )

        # 创建一个参数化的相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        # 获取窗口内每个标记的两两相对位置索引
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

        # 创建 query、key 和 value 的线性映射层
        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)

        # 创建 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # 对输入张量进行形状变换，将最后两个维度分别转换为 self.num_attention_heads 和 self.attention_head_size
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    # 定义一个名为 forward 的方法，接受多个参数并返回一个元组
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        # 获取 hidden_states 的形状信息
        batch_size, dim, num_channels = hidden_states.shape
        # 使用 self.query 对 hidden_states 进行处理得到 mixed_query_layer
        mixed_query_layer = self.query(hidden_states)
        # 使用 self.key 对 hidden_states 进行处理得到 key_layer
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 使用 self.value 对 hidden_states 进行处理得到 value_layer
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 使用 self.transpose_for_scores 对 mixed_query_layer 进行处理得到 query_layer
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算 "query" 和 "key" 之间的点积，得到原始的注意力得分
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 除以 sqrt(注意力头的大小) 进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 获取相对位置偏差表中的偏差值
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        # 重新调整 relative_position_bias 的形状
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        # 转置 relative_position_bias 为合适的形状
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # 将相对位置偏差加入到注意力得分中
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        # 如果存在注意力遮罩
        if attention_mask is not None:
            # 调整注意力得分和遮罩的形状
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # 对注意力得分进行 softmax 处理，得到注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力概率进行 dropout
        attention_probs = self.dropout(attention_probs)

        # 如果存在头遮罩
        if head_mask is not None:
            # 对注意力概率应用头遮罩
            attention_probs = attention_probs * head_mask

        # 计算上下文向量
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 如果需要输出注意力信息，则包含在输出结果中
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # 返回输出结果
        return outputs
# 定义一个名为 DonutSwinSelfOutput 的类，继承自 nn.Module
class DonutSwinSelfOutput(nn.Module):
    # 初始化方法，接受 config 和 dim 参数
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个线性层，输入维度为 dim，输出维度为 dim
        self.dense = nn.Linear(dim, dim)
        # 创建一个丢弃层，丢弃概率为 config.attention_probs_dropout_prob
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 前向传播方法，接受 hidden_states 和 input_tensor 两个参数，返回 torch.Tensor
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过线性层处理 hidden_states
        hidden_states = self.dense(hidden_states)
        # 通过丢弃层处理 hidden_states
        hidden_states = self.dropout(hidden_states)

        return hidden_states

# 定义一个名为 DonutSwinAttention 的类，继承自 nn.Module
class DonutSwinAttention(nn.Module):
    # 初始化方法，接受 config、dim、num_heads 和 window_size 参数
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        # 创建一个 DonutSwinSelfAttention 对象
        self.self = DonutSwinSelfAttention(config, dim, num_heads, window_size)
        # 创建一个 DonutSwinSelfOutput 对象
        self.output = DonutSwinSelfOutput(config, dim)
        # 创建一个空集合 pruned_heads
        self.pruned_heads = set()

    # 头部剪枝方法，接受 heads 参数
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 在 self.self 中找到可剪枝的头部和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播方法，接受多个参数，返回元组
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 调用 self.self 的前向传播方法
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        # 获取注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 将结果存储到 outputs 中并返回
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出注意力，则添加到结果中
        return outputs

# 定义一个名为 DonutSwinIntermediate 的类，继承自 nn.Module
class DonutSwinIntermediate(nn.Module):
    # 初始化方法，接受 config 和 dim 参数
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个线性层，输入维度为 dim，输出维度为 config.mlp_ratio * dim
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        # 判断 config.hidden_act 是否为字符串，选择相应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法，接受 hidden_states 参数，返回 torch.Tensor
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过线性层处理 hidden_states
        hidden_states = self.dense(hidden_states)
        # 通过激活函数处理 hidden_states
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
class DonutSwinOutput(nn.Module):
    # 定义一个名为 "DonutSwinOutput" 的类，继承自 nn.Module
    def __init__(self, config, dim):
        # 构造函数，接受参数 config 和 dim
        super().__init__()
        # 调用父类的构造函数
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        # 创建一个全连接层，输入维度为 int(config.mlp_ratio * dim)，输出维度为 dim
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个丢弃层，参数为 config.hidden_dropout_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 前向传播函数，接受参数 hidden_states，返回类型为 torch.Tensor
        hidden_states = self.dense(hidden_states)
        # 经过全连接层处理
        hidden_states = self.dropout(hidden_states)
        # 经过丢弃层处理
        return hidden_states
        # 返回处理后的 hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinLayer with Swin->DonutSwin
# 从transformers.models.swin.modeling_swin.SwinLayer复制过来，并将Swin->DonutSwin
class DonutSwinLayer(nn.Module):
    # 定义一个名为 "DonutSwinLayer" 的类，继承自 nn.Module
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0):
        # 构造函数，接受参数 config, dim, input_resolution, num_heads, shift_size
        super().__init__()
        # 调用父类的构造函数
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 初始化 chunk_size_feed_forward 为 config 的 chunk_size_feed_forward
        self.shift_size = shift_size
        # 初始化 shift_size 为传入的 shift_size
        self.window_size = config.window_size
        # 初始化 window_size 为 config 的 window_size
        self.input_resolution = input_resolution
        # 初始化 input_resolution 为传入的 input_resolution
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建一个 LayerNorm 层，输入维度为 dim，eps 为 config 的 layer_norm_eps
        self.attention = DonutSwinAttention(config, dim, num_heads, window_size=self.window_size)
        # 创建一个 DonutSwinAttention
        self.drop_path = DonutSwinDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        # 创建一个 DonutSwinDropPath 或者 nn.Identity()，条件为 config 的 drop_path_rate 大于 0.0
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建一个 LayerNorm 层，输入维度为 dim，eps 为 config 的 layer_norm_eps
        self.intermediate = DonutSwinIntermediate(config, dim)
        # 创建一个 DonutSwinIntermediate
        self.output = DonutSwinOutput(config, dim)
        # 创建一个 DonutSwinOutput

    def set_shift_and_window_size(self, input_resolution):
        # 定义一个函数，接受参数 input_resolution
        if min(input_resolution) <= self.window_size:
            # 如果 input_resolution 中的最小值小于等于 window_size
            self.shift_size = 0
            # 则将 shift_size 设为 0
            self.window_size = min(input_resolution)
            # 将 window_size 设为 input_resolution 中的最小值

    def get_attn_mask(self, height, width, dtype):
        # 定义一个函数，接受参数 height, width, dtype
        if self.shift_size > 0:
            # 如果 shift_size 大于 0
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
            # 创建一个全零张量，形状为 (1, height, width, 1)，数据类型为 dtype
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            # 定义 height_slices
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            # 定义 width_slices
            count = 0
            # 初始化 count 为 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    # 对 img_mask 进行索引和赋值
                    count += 1
                    # count 自增1
            mask_windows = window_partition(img_mask, self.window_size)
            # 划分窗口
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            # 重新reshape成二维张量
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # 计算得到注意力掩码
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            # 根据条件填充掩码值
        else:
            attn_mask = None
            # 否则，attn_mask 为 None
        return attn_mask
        # 返回计算得到的注意力掩码
    # 对输入的 hidden_states 进行可能的填充操作，使其高度和宽度均为 window_size 的倍数
    def maybe_pad(self, hidden_states, height, width):
        # 计算宽度方向需要填充的值
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        # 计算高度方向需要填充的值
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        # 构造填充值的元组 (top, bottom, left, right, 0, 0)
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        # 使用 nn.functional.pad 对 hidden_states 进行填充操作
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        # 返回填充后的 hidden_states 和填充值的元组
        return hidden_states, pad_values

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    # 定义函数，接受隐藏状态(hidden_states)和总体输入维度(input_dimensions)作为输入，返回两个张量组成的元组
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 如果不总是进行分区
        if not always_partition:
            # 设置位移和窗口大小，具体实现在另一个函数中
            self.set_shift_and_window_size(input_dimensions)
        # 如果总是进行分区，什么也不做
        else:
            pass
        # 获取输入维度的高度和宽度
        height, width = input_dimensions
        # 获取隐藏状态的批量大小、高度、宽度和通道数
        batch_size, _, channels = hidden_states.size()
        # 将隐藏状态赋值给shortcut，用于后续的加法操作
        shortcut = hidden_states

        # 在LayerNorm之前对隐藏状态进行LayerNorm操作
        hidden_states = self.layernorm_before(hidden_states)

        # 将隐藏状态转换为形状为[batch_size, height, width, channels]的张量
        hidden_states = hidden_states.view(batch_size, height, width, channels)

        # 将隐藏状态填充到窗口大小的倍数，返回填充后的张量和填充值
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        # 获取填充后的张量的形状
        _, height_pad, width_pad, _ = hidden_states.shape
        # 如果存在位移量
        if self.shift_size > 0:
            # 对隐藏状态进行循环位移操作
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        # 如果不存在位移量，直接赋值
        else:
            shifted_hidden_states = hidden_states

        # 将位移后的隐藏状态划分为窗口，返回划分后的窗口张量
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        # 将窗口张量重新调整形状为[batch_size * num_windows, window_size * window_size, channels]
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        # 获取注意力掩码，形状为[1, 1, height_pad, width_pad]，在后续的注意力计算中使用
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        # 如果注意力掩码不为空
        if attn_mask is not None:
            # 将注意力掩码移动到与窗口张量相同的设备上
            attn_mask = attn_mask.to(hidden_states_windows.device)

        # 使用注意力计算器进行注意力计算，返回注意力计算结果和可能的输出注意力
        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        # 获取注意力计算的输出结果
        attention_output = attention_outputs[0]

        # 将注意力输出结果重新调整形状为[batch_size * num_windows, window_size, window_size, channels]
        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        # 将注意力窗口反转，返回反转后的张量
        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)

        # 如果存在位移量
        if self.shift_size > 0:
            # 对反转后的注意力窗口进行循环反转操作，还原为原始顺序
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        # 如果不存在位移量，直接赋值
        else:
            attention_windows = shifted_windows

        # 检查是否进行了填充
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        # 如果进行了填充
        if was_padded:
            # 截取注意力窗口张量，使其与输入维度相匹配
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        # 将注意力窗口张量重新调整形状为[batch_size, height * width, channels]
        attention_windows = attention_windows.view(batch_size, height * width, channels)

        # 对shortcut和注意力窗口张量进行加法操作，并加上DropPath操作
        hidden_states = shortcut + self.drop_path(attention_windows)

        # 对加和后的结果进行LayerNorm操作
        layer_output = self.layernorm_after(hidden_states)
        # 进行中间层计算
        layer_output = self.intermediate(layer_output)
        # 对中间层结果进行输出计算
        layer_output = hidden_states + self.output(layer_output)

        # 如果需要输出注意力信息
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        # 返回层级输出结果
        return layer_outputs
# 从transformers.models.swin.modeling_swin.SwinStage复制过来的，将Swin替换为DonutSwin
class DonutSwinStage(nn.Module):
    # 初始化函数
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        # 调用父类初始化函数
        super().__init__()
        # 保存传入的参数值
        self.config = config
        self.dim = dim
        # 创建一个nn.ModuleList，包含多个DonutSwinLayer对象
        self.blocks = nn.ModuleList(
            [
                DonutSwinLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                )
                for i in range(depth)
            ]
        )

        # 如果downsample不为None，创建downsample对象，否则为None
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        # 初始化pointing为False
        self.pointing = False

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 获取输入的高度和宽度
        height, width = input_dimensions
        # 遍历blocks中的每一层，求得输出hidden_states
        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 调用每一层的前向传播函数
            layer_outputs = layer_module(
                hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition
            )
            # 更新hidden_states
            hidden_states = layer_outputs[0]

        # 保存下采样前的hidden_states
        hidden_states_before_downsampling = hidden_states
        # 如果downsample不为None，对hidden_states进行下采样
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)
        else:
            output_dimensions = (height, width, height, width)

        # 保存stage的输出信息
        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)

        # 如果需要输出attentions，则将attentions信息一并保存
        if output_attentions:
            stage_outputs += layer_outputs[1:]
        # 返回stage的输出信息
        return stage_outputs


# 从transformers.models.swin.modeling_swin.SwinEncoder复制过来的，将Swin替换为DonutSwin
class DonutSwinEncoder(nn.Module):
    # 初始化函数，接受配置和网格大小参数
    def __init__(self, config, grid_size):
        # 调用父类的初始化函数
        super().__init__()
        # 获取深度列表的长度作为层数
        self.num_layers = len(config.depths)
        # 保存配置参数
        self.config = config
        # 生成 drop path rate 列表
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        # 创建模块列表，每个模块都是 DonutSwinStage 实例
        self.layers = nn.ModuleList(
            [
                DonutSwinStage(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    downsample=DonutSwinPatchMerging if (i_layer < self.num_layers - 1) else None,
                )
                for i_layer in range(self.num_layers)
            ]
        )

        # 是否启用梯度检查点
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
        always_partition: Optional[bool] = False,
        return_dict: Optional[bool] = True,
# 定义DonutSwinPreTrainedModel类，继承PreTrainedModel类
class DonutSwinPreTrainedModel(PreTrainedModel):
    
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
    """

    # 配置类
    config_class = DonutSwinConfig
    # 模型前缀
    base_model_prefix = "swin"
    # 主要输入名称
    main_input_name = "pixel_values"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化模型权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 初始化权重
            # 与TF版本稍有不同，TF版本使用截断正态分布进行初始化
            # 参考https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，置为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是LayerNorm层，设置偏置为0，权重为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
# 输入文档字符串
SWIN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`DonutSwinConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 输入文档字符串
SWIN_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`DonutImageProcessor.__call__`] for details.
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

# 添加文档字符串
@add_start_docstrings(
    "The bare Donut Swin Model transformer outputting raw hidden-states without any specific head on top.",
    SWIN_START_DOCSTRING,
)
# 定义DonutSwinModel类，继承DonutSwinPreTrainedModel类
class DonutSwinModel(DonutSwinPreTrainedModel):
    # 初始化函数，接受配置参数、是否添加池化层和是否使用掩码标记作为参数
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        # 调用父类的初始化函数
        super().__init__(config)
        # 保存配置参数
        self.config = config
        # 计算编码器层数
        self.num_layers = len(config.depths)
        # 计算特征数量
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))

        # 创建嵌入层对象
        self.embeddings = DonutSwinEmbeddings(config, use_mask_token=use_mask_token)
        # 创建编码器对象
        self.encoder = DonutSwinEncoder(config, self.embeddings.patch_grid)

        # 如果需要添加池化层，则创建自适应平均池化层对象；否则设置为None
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层的函数
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 剪枝模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要剪枝的层和对应的注意力头
        for layer, heads in heads_to_prune.items():
            # 剪枝对应层的注意力头
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 模型前向传播函数
    @add_start_docstrings_to_model_forward(SWIN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=DonutSwinModelOutput,
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
        ) -> Union[Tuple, DonutSwinModelOutput]:
            r"""
            定义函数的输入参数和返回类型，本例中表示输入参数为一个元组或者 DonutSwinModelOutput 类型的对象
            bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
                表示布尔类型的掩码位置张量，形状为(batch_size, num_patches)，用来指示哪些补丁被遮盖（1）而哪些没有（0）
            """
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            获取输出注意力的标志，如果未指定，则使用配置中的输出注意力标志
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            获取输出隐藏状态的标志，如果未指定，则使用配置中的输出隐藏状态标志
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            获取返回字典的标志，如果未指定，则使用配置中的使用返回字典标志
    
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")
            如果像素值为None，则触发值错误，提示需要指定像素值
    
            # 准备头部掩码
            # head_mask中的1.0表示我们保留该头部
            # attention_probs的形状为bsz x n_heads x N x N
            # 输入的head_mask的形状为[num_heads]或[num_hidden_layers x num_heads]
            # head_mask被转换为形状[num_hidden_layers x batch x num_heads x seq_length x seq_length]
            head_mask = self.get_head_mask(head_mask, len(self.config.depths))
            根据头部掩码的长度，获取头部掩码
    
            embedding_output, input_dimensions = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
            将像素值和bool_masked_pos传入embeddings函数中进行嵌入并获取嵌入输出和输入维度
    
            encoder_outputs = self.encoder(
                embedding_output,
                input_dimensions,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            将嵌入输出、输入维度、头部掩码等传入encoder函数中进行编码得到编码器输出
    
            sequence_output = encoder_outputs[0]
            获取编码器输出的序列输出
    
            pooled_output = None
            如果存在池化器
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)
            对编码器序列输出进行池化处理并将其展平
    
            if not return_dict:
                output = (sequence_output, pooled_output) + encoder_outputs[1:]
                如果不返回字典，则将序列输出、池化输出和编码器输出的其他部分放入元组中返回
    
                return output
                返回输出
    
            return DonutSwinModelOutput(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
                reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
            )
            否则返回一个DonutSwinModelOutput对象，包含最后隐藏的状态、池化输出、隐藏状态和注意力等信息
```