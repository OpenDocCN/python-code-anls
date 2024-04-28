# `.\models\data2vec\modeling_data2vec_vision.py`

```
# coding=utf-8
# 版权声明
# 本文件中的版权归 Meta Platforms 和 HuggingFace Inc. 团队所有，保留所有权利。

# 根据 Apache 许可证，版本 2.0 进行许可；
# 除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获得许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或经书面同意，否则按"原样"分发软件；
# 没有任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言管理权限和限制。
""" PyTorch Data2VecVision model."""

import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 获取 logger 对象
logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "Data2VecVisionConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/data2vec-vision-base"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "facebook/data2vec-vision-base-ft1k"
_IMAGE_CLASS_EXPECTED_OUTPUT = "remote control, remote"

# Data2VecVision 预训练模型列表
DATA2VEC_VISION_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/data2vec-vision-base-ft1k",
    # 在 https://huggingface.co/models?filter=data2vec-vision 查看所有 Data2VecVision 模型
]

@dataclass
# 从 transformers.models.beit.modeling_beit.BeitModelOutputWithPooling 复制并将 Beit 替换成 Data2VecVision
class Data2VecVisionModelOutputWithPooling(BaseModelOutputWithPooling):
    """
    Class for outputs of [`Data2VecVisionModel`].
    # 参数说明：
    # - last_hidden_state: 模型最后一层的隐藏状态，形状为(batch_size, sequence_length, hidden_size)
    # - pooler_output: 如果config.use_mean_pooling为True，则为除了[CLS]标记之外路径标记的最后一层隐藏状态的平均值；
    #                  如果为False，则返回[CLS]标记的最终隐藏状态。
    # - hidden_states: 一个元组，包含每一层输出的隐藏状态。形状为(batch_size, sequence_length, hidden_size)。当output_hidden_states=True时返回。
    # - attentions: 一个元组，包含每一层 self-attention 后的注意力权重。形状为(batch_size, num_heads, sequence_length, sequence_length)。当output_attentions=True时返回。
# 从transformers.models.beit.modeling_beit.drop_path中复制得到的函数，用于在模型中实现Drop Path（随机深度）
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    # 如果dropout概率为0或者不处于训练模式，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留的概率
    keep_prob = 1 - drop_prob
    # 确定形状，以便处理不同维度的张量，而不仅仅是2D的ConvNets
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    # 生成一个与input相同形状的随机张量，并使用设备和数据类型与input相同
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    # 将随机张量二值化，使其成为0或1
    random_tensor.floor_()
    # 将input除以保留的概率，然后乘以随机张量
    output = input.div(keep_prob) * random_tensor
    # 返回处理后的张量
    return output


# 从transformers.models.beit.modeling_beit.BeitDropPath中复制得到的类，用于在模型中实现Drop Path（随机深度）
class Data2VecVisionDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        # 设置drop的概率
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用前面定义的drop_path函数来处理输入张量
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        # 返回额外的表示，用于打印类的信息
        return "p={}".format(self.drop_prob)


# 从transformers.models.beit.modeling_beit.BeitEmbeddings中复制得到的类，用于构建数据到向量（Data2Vec）视觉任务的嵌入层
class Data2VecVisionEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.

    """

    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__()
        # 定义CLS token作为模型的第一个token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 如果配置中设置了使用mask token，则定义mask token
        if config.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        else:
            self.mask_token = None
        # 定义patch的嵌入层
        self.patch_embeddings = Data2VecVisionPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        # 如果配置中设置了使用绝对位置嵌入，则定义绝对位置嵌入
        if config.use_absolute_position_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        else:
            self.position_embeddings = None
        # 定义dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    # 定义一个前向传播方法，接受像素值作为输入，并可选地接受布尔类型的遮罩位置信息，返回嵌入的张量
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        # 使用 patch_embeddings 方法处理像素值，得到嵌入张量和图像分块的高度和宽度信息
        embeddings, (patch_height, patch_width) = self.patch_embeddings(
            pixel_values, self.position_embeddings[:, 1:, :] if self.position_embeddings is not None else None
        )
        # 获取嵌入张量的批量大小、序列长度和嵌入维度
        batch_size, seq_len, _ = embeddings.size()

        # 如果存在遮罩位置信息
        if bool_masked_pos is not None:
            # 使用 mask_token 扩展遮罩令牌以匹配嵌入张量的形状
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # 将受遮罩的可见令牌替换为 mask_tokens
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - w) + mask_tokens * w

        # 使用 cls_token 扩展类别令牌以匹配嵌入张量的形状
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # 如果存在位置嵌入，则将类别令牌与其相对应的位置嵌入相加
        if self.position_embeddings is not None:
            cls_tokens = cls_tokens + self.position_embeddings[:, :1, :]

        # 在嵌入张量的开头添加类别令牌，得到完整的嵌入张量
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 对嵌入张量应用 dropout
        embeddings = self.dropout(embeddings)

        # 返回嵌入张量及图像分块的高度和宽度信息
        return embeddings, (patch_height, patch_width)
class Data2VecVisionPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # Convert single values to tuples if necessary
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # Calculate number of patches and patch shape
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        # Store relevant configuration parameters
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        # Projection layer to convert pixel values to patch embeddings
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, position_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Get dimensions of input pixel values
        batch_size, num_channels, height, width = pixel_values.shape
        # Check if number of channels matches the configured number of channels
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        # Project pixel values into patch embeddings
        embeddings = self.projection(pixel_values)
        # Get height and width of the patches
        patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]

        # Add position embedding if provided
        if position_embedding is not None:
            # Interpolate the position embedding to match the size of the patches
            position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1).permute(
                0, 3, 1, 2
            )
            position_embedding = nn.functional.interpolate(
                position_embedding, size=(patch_height, patch_width), mode="bicubic"
            )
            # Add the position embedding to the patch embeddings
            embeddings = embeddings + position_embedding

        # Flatten the embeddings and transpose dimensions for transformer input
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, (patch_height, patch_width)


class Data2VecVisionSelfAttention(nn.Module):
    # 初始化方法，接收配置和窗口大小作为参数
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None) -> None:
        # 调用父类初始化方法
        super().__init__()
        # 如果隐藏层大小不是注意力头数的倍数，并且配置对象没有嵌入大小属性，则抛出数值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化线性层，用于计算查询、键和值
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化丢弃层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 如果有窗口大小，则初始化相对位置偏置对象，否则设置为None
        if window_size:
            self.relative_position_bias = Data2VecVisionRelativePositionBias(config, window_size=window_size)
        else:
            self.relative_position_bias = None

    # 将输入张量重塑为注意力分数所需的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播方法，接收隐藏状态、注意力头蒙版、输出注意力指数、相对位置偏置等作为参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional["Data2VecVisionRelativePositionBias"] = None,
    # 定义函数，接收输入信息和头部掩蔽，并返回上下文层和注意力概率
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        # 对隐藏状态进行查询操作
        mixed_query_layer = self.query(hidden_states)

        # 调用self.key方法，获取键层
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 调用self.value方法，获取值层
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 对混合的查询层进行转置，以备接下来的矩阵乘法
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 对"查询"和"键"进行点积，得到原始的注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 如果存在相对位置偏置，则添加进注意力分数中
        if self.relative_position_bias is not None:
            attention_scores = attention_scores + self.relative_position_bias().unsqueeze(0)

        # 如果提供了共享的相对位置偏置，则添加进注意力分数中
        if relative_position_bias is not None:
            attention_scores = attention_scores + relative_position_bias

        # 将注意力分数规范化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力概率进行Dropout操作
        attention_probs = self.dropout(attention_probs)

        # 如果存在头部掩蔽，则对注意力概率进行掩蔽
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，将注意力概率和值层进行矩阵乘法
        context_layer = torch.matmul(attention_probs, value_layer)

        # 将上下文层进行维度变换
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 根据输出的设置，返回上下文层和注意力概率，或者仅返回上下文层
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # 返回输出结果
        return outputs
# 从transformers.models.beit.modeling_beit.BeitSelfOutput复制代码并将Beit->Data2VecVision
class Data2VecVisionSelfOutput(nn.Module):
    """
    The residual connection is defined in Data2VecVisionLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """
    
    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 创建一个全连接层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 创建一个dropout层

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, gamma=None) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)  # 对输入的hidden_states使用全连接层
        hidden_states = self.dropout(hidden_states)  # 对全连接层的输出使用dropout层

        return hidden_states


# 从transformers.models.beit.modeling_beit.BeitAttention复制代码并将Beit->Data2VecVision
class Data2VecVisionAttention(nn.Module):
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None) -> None:
        super().__init__()
        self.attention = Data2VecVisionSelfAttention(config, window_size=window_size)  # 创建Data2VecVisionSelfAttention对象
        self.output = Data2VecVisionSelfOutput(config)  # 创建Data2VecVisionSelfOutput对象
        self.pruned_heads = set()  # 创建一个空的set用于存储被prune掉的head

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)  # 对attention中的query进行prune
        self.attention.key = prune_linear_layer(self.attention.key, index)  # 对attention中的key进行prune
        self.attention.value = prune_linear_layer(self.attention.value, index)  # 对attention中的value进行prune
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)  # 对output中的dense层进行prune

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)  # 更新num_attention_heads
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads  # 更新all_head_size
        self.pruned_heads = self.pruned_heads.union(heads)  # 将被prune的head添加到pruned_heads中

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional["Data2VecVisionRelativePositionBias"] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions, relative_position_bias)  # 调用attention层

        attention_output = self.output(self_outputs[0], hidden_states)  # 将attention输出作为SelfOutput的输入

        outputs = (attention_output,) + self_outputs[1:]  # 构建输出元组
        return outputs


# 从transformers.models.beit.modeling_beit.BeitIntermediate复制代码并将Beit->Data2VecVision
class Data2VecVisionIntermediate(nn.Module):
    # 初始化函数，接受一个Data2VecVisionConfig类型的参数config
    def __init__(self, config: Data2VecVisionConfig) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个nn.Linear对象，将输入维度设置为config.hidden_size，输出维度设置为config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 检查config.hidden_act是否为字符串类型，如果是则使用ACT2FN字典对应的函数，否则直接使用config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数，接受一个torch.Tensor类型的参数hidden_states，返回一个torch.Tensor类型的结果
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入hidden_states通过self.dense进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将经过线性变换后的hidden_states通过self.intermediate_act_fn进行激活函数处理
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states
# 定义了一个继承自 nn.Module 的 Data2VecVisionOutput 类
class Data2VecVisionOutput(nn.Module):
    # 初始化函数
    def __init__(self, config: Data2VecVisionConfig) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个线性层，输入为 intermediate_size 维，输出为 hidden_size 维
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 Dropout 层，丢弃比例为 hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入通过线性层
        hidden_states = self.dense(hidden_states)
        # 将输出通过 Dropout 层
        hidden_states = self.dropout(hidden_states)
        # 返回最终的输出
        return hidden_states


# 定义了一个继承自 nn.Module 的 Data2VecVisionLayer 类
class Data2VecVisionLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    # 初始化函数
    def __init__(
        self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None, drop_path_rate: float = 0.0
    ) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 设置 chunk_size_feed_forward 和 seq_len_dim 属性
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 创建 Data2VecVisionAttention 模块
        self.attention = Data2VecVisionAttention(config, window_size=window_size)
        # 创建 Data2VecVisionIntermediate 模块
        self.intermediate = Data2VecVisionIntermediate(config)
        # 创建 Data2VecVisionOutput 模块
        self.output = Data2VecVisionOutput(config)
        # 创建两个 LayerNorm 层
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 DropPath 模块或 Identity 模块
        self.drop_path = Data2VecVisionDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        # 根据 layer_scale_init_value 参数创建两个可学习参数
        init_values = config.layer_scale_init_value
        if init_values > 0:
            self.lambda_1 = nn.Parameter(init_values * torch.ones((config.hidden_size)), requires_grad=True)
            self.lambda_2 = nn.Parameter(init_values * torch.ones((config.hidden_size)), requires_grad=True)
        else:
            self.lambda_1, self.lambda_2 = None, None

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional["Data2VecVisionRelativePositionBias"] = None,
    ):
        # 省略后续代码
        ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        # 使用 self-attention 模块处理隐藏状态，传入 layernorm 处理后的隐藏状态
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在 Data2VecVision 中，self-attention 前会应用 layernorm
            head_mask,
            output_attentions=output_attentions,
            relative_position_bias=relative_position_bias,
        )
        # 获取 self-attention 的输出
        attention_output = self_attention_outputs[0]
        # 如果输出注意权重，则将注意权重添加到 outputs 中
        outputs = self_attention_outputs[1:]

        # 如果存在 lambda_1，则应用 lambda_1
        if self.lambda_1 is not None:
            attention_output = self.lambda_1 * attention_output

        # 应用第一个残差连接
        hidden_states = self.drop_path(attention_output) + hidden_states

        # 在 Data2VecVision 中，self-attention 后也应用 layernorm
        layer_output = self.layernorm_after(hidden_states)

        # 应用 intermediate 层
        layer_output = self.intermediate(layer_output)
        # 应用 output 层
        layer_output = self.output(layer_output)

        # 如果存在 lambda_2，则应用 lambda_2
        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # 应用第二个残差连接
        layer_output = self.drop_path(layer_output) + hidden_states

        # 将 layer_output 添加到输出中
        outputs = (layer_output,) + outputs

        return outputs
# 从transformers.models.beit.modeling_beit.BeitRelativePositionBias 复制而来，将Beit替换为Data2VecVision
class Data2VecVisionRelativePositionBias(nn.Module):
    def __init__(self, config: Data2VecVisionConfig, window_size: tuple) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        # 初始化相对位置偏置表为所有元素为0的张量，形状为(num_relative_distance, num_attention_heads)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, config.num_attention_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # 获取在窗口内每个令牌的配对相对位置索引
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # 计算令牌间的相对坐标
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # 从0开始偏移
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        # 将相对位置索引设置为buffer，并不会随训练而更新
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def forward(self) -> torch.Tensor:
        # 获取对应相对位置偏置，reshape为(nH, Wh*Ww, Wh*Ww)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1
        )  # Wh*Ww,Wh*Ww,nH

        # 调整维度顺序并返回
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


# 从transformers.models.beit.modeling_beit.BeitEncoder 复制而来，将Beit替换为Data2VecVision
class Data2VecVisionEncoder(nn.Module):
    # 初始化函数，接受配置和窗口大小参数
    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple] = None) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 保存配置信息
        self.config = config
        # 根据配置信息判断是否使用共享的相对位置偏置
        if config.use_shared_relative_position_bias:
            # 如果使用，则创建一个 Data2VecVisionRelativePositionBias 对象
            self.relative_position_bias = Data2VecVisionRelativePositionBias(config, window_size=window_size)
        else:
            # 否则设置为 None
            self.relative_position_bias = None

        # 使用随机深度衰减规则，生成一个从 0 到 drop_path_rate 的数列
        # 并创建一个包含多个 Data2VecVisionLayer 对象的列表
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layer = nn.ModuleList(
            [
                Data2VecVisionLayer(
                    config,
                    window_size=window_size if config.use_relative_position_bias else None,
                    drop_path_rate=dpr[i],
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        # 关闭渐变检查点技术
        self.gradient_checkpointing = False

    # 正向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 初始化存储所有隐藏状态和自注意力的空元组
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 遍历每个 Data2VecVisionLayer 对象
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果开启渐变检查点技术并且正在训练，则使用 gradient_checkpointing_func 函数
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数获取当前层的输出
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则，获取相对位置偏置，如果存在的话
                relative_position_bias = (
                    self.relative_position_bias() if self.relative_position_bias is not None else None
                )
                # 调用当前 Data2VecVisionLayer 的前向传播函数
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, relative_position_bias)

            # 更新当前隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出自注意力权重，则将当前层的自注意力权重添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要以字典形式返回结果，则返回隐藏状态、所有隐藏状态和自注意力权重
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则以 BaseModelOutput 对象的形式返回结果
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 创建 Data2VecVisionPreTrainedModel 类，继承自 PreTrainedModel
class Data2VecVisionPreTrainedModel(PreTrainedModel):
    """
    用于处理权重初始化和下载预训练模型的抽象类。

    config_class：Data2VecVisionConfig 的类
    base_model_prefix：模型的基本前缀为 "data2vec_vision"
    main_input_name：主要输入名称为 "pixel_values"
    supports_gradient_checkpointing：支持梯度检查点
    """

    # 初始化模型的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # 稍有不同于 TF 版本，使用标准正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# Data2VecVision 模型的文档起始字符串
DATA2VEC_VISION_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Data2VecVisionConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# Data2VecVision 模型的输入文档字符串
DATA2VEC_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素数值。可以使用 [`AutoImageProcessor`] 获取像素值。查看 [`BeitImageProcessor.__call__`] 获取更多细节。

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块的特定头部置零的掩码。掩码值范围在 `[0, 1]` 之间：

            - 1 表示头部是 **未屏蔽** 的，
            - 0 表示头部是 **屏蔽** 的。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。查看返回的张量中的 `attentions` 获取更多细节。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。查看返回的张量中的 `hidden_states` 获取更多细节。

        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是一个普通的元组。
# 添加引言文档字符串到Data2VecVision Model transformer的基本输出，输出原始的隐藏状态，没有特定的输出头
# 添加Data2VecVision模型的起始文档字符串
# 从transformers.models.beit.modeling_beit.BeitModel复制，将BEIT->DATA2VEC_VISION，Beit->Data2VecVision,True->False
class Data2VecVisionModel(Data2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig, add_pooling_layer: bool = False) -> None:
        # 调用父类初始化方法，传入配置
        super().__init__(config)
        # 设置配置属性
        self.config = config

        # 初始化嵌入层
        self.embeddings = Data2VecVisionEmbeddings(config)
        # 初始化编码器
        self.encoder = Data2VecVisionEncoder(config, window_size=self.embeddings.patch_embeddings.patch_shape)

        # 如果配置使用均值池化，则使用Identity层，否则使用LayerNorm层
        self.layernorm = (
            nn.Identity() if config.use_mean_pooling else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        # 如果需要添加池化层，则初始化池化层，否则设置为None
        self.pooler = Data2VecVisionPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 剪枝模型的头部，heads_to_prune: {layer_num: 要剪枝的头部列表}
    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 添加模型前向方法的文档字符串
    # 添加代码示例文档字符串
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

… （以下省略）
        ) -> Union[tuple, Data2VecVisionModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # 如果未指定output_attentions，则使用配置中的output_attentions值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定output_hidden_states，则使用配置中的output_hidden_states值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定return_dict，则使用配置中的use_return_dict值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果pixel_values为None，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 准备头部掩码（head_mask）如果需要
        # 在head_mask中，1.0表示保留该头部
        # attention_probs的形状为bsz x n_heads x N x N
        # 输入的head_mask形状为[num_heads]或[num_hidden_layers x num_heads]
        # 并且head_mask转换为形状[num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 嵌入层的输出和（patch_height, patch_width）维度
        embedding_output, (patch_height, patch_width) = self.embeddings(pixel_values, bool_masked_pos)

        # 编码器的输出
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 序列输出
        sequence_output = encoder_outputs[0]
        # 序列归一化
        sequence_output = self.layernorm(sequence_output)
        # 如果pooler存在，则生成池化输出；否则为None
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果不使用return_dict，则返回一组头部输出
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # 使用return_dict时，返回自定义的数据类Data2VecVisionModelOutputWithPooling的实例
        return Data2VecVisionModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 从transformers.models.beit.modeling_beit导入Data2VecVisionPooler类，并将Beit改为Data2VecVision
class Data2VecVisionPooler(nn.Module):
    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__()
        # 如果配置config参数中使用均值池化，则创建LayerNorm层
        self.layernorm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if config.use_mean_pooling else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.layernorm is not None:
            # 对补丁令牌(patch tokens)的最终隐藏状态进行均值池化
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = self.layernorm(patch_tokens.mean(1))
        else:
            # 只需取[CLS]标记的最终隐藏状态进行池化
            pooled_output = hidden_states[:, 0]

        return pooled_output


# 添加起始文档字符串描述
@add_start_docstrings(
    """
    Data2VecVision Model transformer with an image classification head on top (a linear layer on top of the average of
    the final hidden states of the patch tokens) e.g. for ImageNet.
    """,
    DATA2VEC_VISION_START_DOCSTRING,
)
# 从transformers.models.beit.modeling_beit导入Data2VecVisionForImageClassification类，并将BEIT->DATA2VEC_VISION,Beit->Data2VecVision,beit->data2vec_vision
class Data2VecVisionForImageClassification(Data2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__(config)

        # 配置分类标签的数量
        self.num_labels = config.num_labels
        # 创建Data2VecVisionModel模型，同时添加池化层
        self.data2vec_vision = Data2VecVisionModel(config, add_pooling_layer=True)

        # 分类器头部
        # 如果num_labels大于0，则创建线性层用于分类，否则创建一个恒等映射
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # 初始化权重并进行最终处理
        self.post_init()

    # 添加起始文档字符串到模型的forward方法
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    # 添加代码示例文档字符串
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 运行数据转换和向量化处理
        outputs = self.data2vec_vision(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果return_dict为False，则pooled_output为outputs的第二部分，否则为outputs的pooler_output部分
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 通过分类器计算logits
        logits = self.classifier(pooled_output)

        # 初始化loss为空
        loss = None
        # 如果labels不为空
        if labels is not None:
            # 如果self.config.problem_type为None
            if self.config.problem_type is None:
                # 如果num_labels为1，将problem_type设置为"regression"
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                # 如果num_labels大于1且labels的数据类型为torch.long或torch.int，将problem_type设置为"single_label_classification"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                # 否则将problem_type设置为"multi_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 如果problem_type为"regression"
            if self.config.problem_type == "regression":
                # 使用MSELoss计算损失
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 如果num_labels为1，使用squeezed logits和labels计算损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 否则直接使用logits和labels计算损失
                    loss = loss_fct(logits, labels)
            # 如果problem_type为"single_label_classification"
            elif self.config.problem_type == "single_label_classification":
                # 使用CrossEntropyLoss计算损失
                loss_fct = CrossEntropyLoss()
                # 将logits和labels展平后计算损失
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # 如果problem_type为"multi_label_classification"
            elif self.config.problem_type == "multi_label_classification":
                # 使用BCEWithLogitsLoss计算损失
                loss_fct = BCEWithLogitsLoss()
                # 直接使用logits和labels计算损失
                loss = loss_fct(logits, labels)
        # 如果return_dict为False
        if not return_dict:
            # 返回带有损失的output
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回包含损失、logits、hidden_states和attentions的ImageClassifierOutput
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个名为Data2VecVisionConvModule的类，继承自nn.Module类，封装了卷积块的操作
class Data2VecVisionConvModule(nn.Module):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    
    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int], str] = 0,
        bias: bool = False,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        # 创建卷积层对象
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        # 创建批量归一化层对象
        self.bn = nn.BatchNorm2d(out_channels)
        # 创建激活函数层对象
        self.activation = nn.ReLU()

    # 前向传播函数，对输入数据进行卷积、归一化和激活操作
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 执行卷积操作
        output = self.conv(input)
        # 执行归一化操作
        output = self.bn(output)
        # 执行激活操作
        output = self.activation(output)

        return output


# 定义一个名为Data2VecVisionPyramidPoolingBlock的类，继承自nn.Module类，用于执行池化操作
class Data2VecVisionPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        super().__init__()
        # 创建池化块层
        self.layers = [
            nn.AdaptiveAvgPool2d(pool_scale),
            Data2VecVisionConvModule(in_channels, channels, kernel_size=1),
        ]
        for i, layer in enumerate(self.layers):
            # 添加层到模块中
            self.add_module(str(i), layer)

    # 前向传播函数，执行多层池化和卷积操作
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


# 定义一个名为Data2VecVisionPyramidPoolingModule的类，继承自nn.Module类，用于执行金字塔池化操作
class Data2VecVisionPyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """
    # 初始化函数，接受池化比例、输入通道数、输出通道数和是否对齐边界作为参数
    def __init__(self, pool_scales: Tuple[int, ...], in_channels: int, channels: int, align_corners: bool) -> None:
        # 调用父类初始化函数
        super().__init__()
        # 设置池化比例
        self.pool_scales = pool_scales
        # 是否对齐边界
        self.align_corners = align_corners
        # 输入通道数
        self.in_channels = in_channels
        # 输出通道数
        self.channels = channels
        # 初始化模块列表
        self.blocks = []
        # 遍历池化比例
        for i, pool_scale in enumerate(pool_scales):
            # 创建并初始化 Data2VecVisionPyramidPoolingBlock 模块
            block = Data2VecVisionPyramidPoolingBlock(
                pool_scale=pool_scale, in_channels=in_channels, channels=channels
            )
            # 将创建的模块添加到模块列表中
            self.blocks.append(block)
            # 为每个模块添加命名属性，使其成为子模块
            self.add_module(str(i), block)

    # 前向传播函数，接受输入张量 x 作为参数，返回上采样后的特征列表
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # 初始化用于存储每个模块输出的列表
        ppm_outs = []
        # 遍历模块列表
        for ppm in self.blocks:
            # 将输入张量传递给模块，获取输出
            ppm_out = ppm(x)
            # 对输出进行双线性插值上采样
            upsampled_ppm_out = nn.functional.interpolate(
                ppm_out, size=x.size()[2:], mode="bilinear", align_corners=self.align_corners
            )
            # 将上采样后的结果添加到输出列表中
            ppm_outs.append(upsampled_ppm_out)
        # 返回上采样后的特征列表
        return ppm_outs
# 定
    def
# 定义一个名为 Data2VecVisionFCNHead 的类，继承自 nn.Module 类。这个类实现了全卷积网络用于语义分割的头部。
class Data2VecVisionFCNHead(nn.Module):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is implemented of
    [FCNNet](https://arxiv.org/abs/1411.4038>).

    Args:
        config (Data2VecVisionConfig): Configuration.
        in_channels
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        dilation (int): The dilation rate for convs in the head. Default: 1.


    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    # 构造函数，初始化对象
    def __init__(
        self,
        config: Data2VecVisionConfig,
        in_index: int = 2,
        kernel_size: int = 3,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        # 获取配置参数
        self.in_channels = config.hidden_size
        self.channels = config.auxiliary_channels
        self.num_convs = config.auxiliary_num_convs
        self.concat_input = config.auxiliary_concat_input
        self.in_index = in_index

        # 计算卷积的填充
        conv_padding = (kernel_size // 2) * dilation
        convs = []
        # 添加卷积层到列表
        convs.append(
            Data2VecVisionConvModule(
                self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
            )
        )
        for i in range(self.num_convs - 1):
            convs.append(
                Data2VecVisionConvModule(
                    self.channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                )
            )
        # 如果没有卷积层，则使用 nn.Identity()
        if self.num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        # 如果设置了输入拼接，则添加另一个卷积层
        if self.concat_input:
            self.conv_cat = Data2VecVisionConvModule(
                self.in_channels + self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2
            )

        # 分类器层
        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

    # 前向传播方法
    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # 从encoder_hidden_states中取出相关特征图
        hidden_states = encoder_hidden_states[self.in_index]
        # 对特征图进行卷积操作
        output = self.convs(hidden_states)
        # 如果需要拼接输入，则进行拼接并使用conv_cat层
        if self.concat_input:
            output = self.conv_cat(torch.cat([hidden_states, output], dim=1))
        # 使用分类器层
        output = self.classifier(output)
        return output


@add_start_docstrings(
    """
    Data2VecVision Model transformer with a semantic segmentation head on top e.g. for ADE20k, CityScapes.
    """,
    DATA2VEC_VISION_START_DOCSTRING,
)
# 定义一个名为 Data2VecVisionForSemanticSegmentation 的类，继承自 Data2VecVisionPreTrainedModel 类。
# 这个类用于实现在 Data2VecVision 模型之上带有语义分割头部的转换器模型，例如用于 ADE20k、CityScapes 等场景。
# 此代码块为注释块结束，不包含代码注释
class Data2VecVisionForSemanticSegmentation(Data2VecVisionPreTrainedModel):
    # 初始化函数，接收 Data2VecVisionConfig 类型的参数并调用父类的初始化函数
    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__(config)

        # 设置类属性 num_labels 和 data2vec_vision
        self.num_labels = config.num_labels
        self.data2vec_vision = Data2VecVisionModel(config, add_pooling_layer=False)

        # 创建 FPNs
        if len(self.config.out_indices) != 4:
            # 如果 config.out_indices 不是含有四个整数的列表，抛出数值错误
            raise ValueError(
                "Data2VecVisionForSemanticSegmentation requires config.out_indices to be a list of 4 integers, "
                "specifying which features to use from the backbone. One can use [3, 5, 7, 11] in case of "
                "a base-sized architecture."
            )
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
            nn.BatchNorm2d(config.hidden_size),
            nn.GELU(),
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 创建 Semantic segmentation head(s)
        self.decode_head = Data2VecVisionUperHead(config)
        self.auxiliary_head = Data2VecVisionFCNHead(config) if config.use_auxiliary_head else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 计算损失函数
    def compute_loss(self, logits, auxiliary_logits, labels):
        # 将 logits 上采样到原始图像大小
        upsampled_logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        if auxiliary_logits is not None:
            upsampled_auxiliary_logits = nn.functional.interpolate(
                auxiliary_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
        # 计算加权损失
        loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
        main_loss = loss_fct(upsampled_logits, labels)
        loss = main_loss
        if auxiliary_logits is not None:
            auxiliary_loss = loss_fct(upsampled_auxiliary_logits, labels)
            loss += self.config.auxiliary_loss_weight * auxiliary_loss

        return loss

    # 根据输入参数进行前向传播
    @add_start_docstrings_to_model_forward(DATA2VEC_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```