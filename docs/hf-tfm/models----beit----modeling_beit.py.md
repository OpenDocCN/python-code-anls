# `.\models\beit\modeling_beit.py`

```py
# 设置文件编码为 UTF-8
# 版权声明 2021 Microsoft Research 和 HuggingFace Inc. 团队，保留所有权利
#
# 根据 Apache 许可证 2.0 版本使用本文件，除非符合许可证的条款，否则不得使用本文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依据"原样"分发本软件，
# 没有任何形式的担保或条件，包括但不限于对适销性或特定用途的适用性的保证。
# 有关详细信息，请参阅许可证。
""" PyTorch BEiT 模型。"""

# 导入必要的库和模块
import collections.abc  # 引入 collections.abc 模块
import math  # 引入 math 模块
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from typing import List, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 模块
from torch import Tensor, nn  # 从 PyTorch 导入 Tensor 和 nn（神经网络）模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 从 nn 模块导入损失函数

# 导入其他需要的类和函数
from ...activations import ACT2FN  # 从 activations 模块导入 ACT2FN 激活函数
from ...modeling_outputs import (  # 导入模型输出相关的类
    BackboneOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedLMOutput,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel  # 从 modeling_utils 模块导入 PreTrainedModel 类
from ...pytorch_utils import (  # 导入 PyTorch 工具函数
    find_pruneable_heads_and_indices,
    meshgrid,
    prune_linear_layer,
)
from ...utils import (  # 导入工具函数和类
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import BackboneMixin  # 从 backbone_utils 模块导入 BackboneMixin 类
from .configuration_beit import BeitConfig  # 导入 BEiT 模型的配置类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 概述文件的一般用途
_CONFIG_FOR_DOC = "BeitConfig"

# 基础说明文档
_CHECKPOINT_FOR_DOC = "microsoft/beit-base-patch16-224-pt22k"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# 图像分类的说明文档
_IMAGE_CLASS_CHECKPOINT = "microsoft/beit-base-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

BEIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/beit-base-patch16-224",
    # 查看所有 BEiT 模型的列表 https://huggingface.co/models?filter=beit
]

@dataclass
class BeitModelOutputWithPooling(BaseModelOutputWithPooling):
    """
    [`BeitModel`] 的输出类。
    """
    pass  # 占位符，表示类目前不包含额外的属性或方法，继承自 BaseModelOutputWithPooling 类
    # 接收模型最后一层的隐藏状态，形状为 `(batch_size, sequence_length, hidden_size)` 的张量
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
    
    # 如果 *config.use_mean_pooling* 设置为 True，则返回补丁标记的最后一层隐藏状态的平均值（不包括 *[CLS]* 标记）。
    # 如果设置为 False，则返回 *[CLS]* 标记的最终隐藏状态。
    pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
    
    # 可选参数，当 `output_hidden_states=True` 时返回，或者当 `config.output_hidden_states=True` 时返回。
    # 是一个元组，包含 `torch.FloatTensor` 类型的张量：
    #   - 一个是嵌入层的输出
    #   - 其余每一层的输出，形状为 `(batch_size, sequence_length, hidden_size)`
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
    
    # 可选参数，当 `output_attentions=True` 时返回，或者当 `config.output_attentions=True` 时返回。
    # 是一个元组，包含 `torch.FloatTensor` 类型的张量：
    #   - 每一层的注意力权重，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`
    #   这些权重经过注意力 softmax 后得到，用于计算自注意力头中的加权平均值。
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
# 定义一个函数，用于在模型训练时对输入的张量进行路径丢弃（随机深度），通常应用于残差块的主路径。
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        # 如果丢弃概率为0或者当前不处于训练状态，直接返回输入张量
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度张量，而不仅仅是2D卷积网络
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 将随机张量二值化
    output = input.div(keep_prob) * random_tensor
    return output


class BeitDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用上面定义的drop_path函数来处理输入的隐藏状态张量
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        # 返回当前DropPath模块的额外表示，包括当前的丢弃概率
        return "p={}".format(self.drop_prob)


# 基于timm实现，可以在此找到：
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class BeitEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.

    """

    def __init__(self, config: BeitConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        if config.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        else:
            self.mask_token = None
        self.patch_embeddings = BeitPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        if config.use_absolute_position_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        else:
            self.position_embeddings = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    # 定义前向传播方法，接受像素值张量和可选的掩码位置张量，返回处理后的张量
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        # 使用 patch_embeddings 方法处理像素值张量，得到嵌入向量和嵌入坐标
        embeddings, (patch_height, patch_width) = self.patch_embeddings(
            pixel_values, self.position_embeddings[:, 1:, :] if self.position_embeddings is not None else None
        )
        # 获取批次大小、序列长度和嵌入向量的维度
        batch_size, seq_len, _ = embeddings.size()

        # 如果存在掩码位置张量
        if bool_masked_pos is not None:
            # 将掩码位置标记的视觉标记替换为掩码标记
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - w) + mask_tokens * w

        # 将 cls_token 扩展到与批次大小和嵌入向量维度相匹配
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # 如果存在位置嵌入，则将其加到 cls_token 上
        if self.position_embeddings is not None:
            cls_tokens = cls_tokens + self.position_embeddings[:, :1, :]

        # 在序列的开头连接 cls_token 和 embeddings
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 对 embeddings 应用 dropout
        embeddings = self.dropout(embeddings)

        # 返回处理后的 embeddings 和 patch 的高度、宽度信息
        return embeddings, (patch_height, patch_width)
# 定义一个用于将像素值转换成初始隐藏状态（即补丁嵌入）的类，以便Transformer模型使用。
class BeitPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        # 从配置中获取图像大小和补丁大小
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 确保图像大小和补丁大小是可迭代的对象，如果不是则转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        
        # 计算补丁数量和补丁形状
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        
        # 初始化对象的属性
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        # 使用卷积层进行投影，将输入的通道数转换为隐藏状态的大小
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, position_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 获取输入像素值的维度信息
        batch_size, num_channels, height, width = pixel_values.shape
        # 检查输入的通道数是否与配置中设置的通道数一致
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        # 使用投影层对输入像素值进行投影，得到嵌入表示
        embeddings = self.projection(pixel_values)
        # 获取投影后的补丁高度和宽度
        patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]

        if position_embedding is not None:
            # 插值位置嵌入到相应的大小
            position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1).permute(
                0, 3, 1, 2
            )
            position_embedding = nn.functional.interpolate(
                position_embedding, size=(patch_height, patch_width), mode="bicubic"
            )
            # 将位置嵌入加到投影后的嵌入中
            embeddings = embeddings + position_embedding

        # 将嵌入表示展平，并交换维度顺序以符合Transformer的输入格式
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, (patch_height, patch_width)
    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None) -> None:
        super().__init__()
        # 检查隐藏层大小是否能被注意力头数整除，同时没有嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 初始化注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化用于随机失活的 Dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 如果指定了窗口大小，初始化相对位置偏置层
        if window_size:
            self.relative_position_bias = BeitRelativePositionBias(config, window_size=window_size)
        else:
            self.relative_position_bias = None

    def transpose_for_scores(self, x):
        # 调整张量形状以便进行多头注意力计算
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional["BeitRelativePositionBias"] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        # 通过调用 self.query 方法生成混合查询向量 mixed_query_layer
        mixed_query_layer = self.query(hidden_states)

        # 使用 self.key 方法生成键向量 key_layer，并通过 transpose_for_scores 方法转置以备注意力计算使用
        key_layer = self.transpose_for_scores(self.key(hidden_states))

        # 使用 self.value 方法生成值向量 value_layer，并通过 transpose_for_scores 方法转置以备注意力计算使用
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 再次调用 transpose_for_scores 方法转置 mixed_query_layer 以备注意力计算使用
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算注意力分数，采用 query_layer 和 key_layer 的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 缩放注意力分数，除以 sqrt(attention_head_size)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 如果存在相对位置偏置，将其加入注意力分数中
        if self.relative_position_bias is not None:
            attention_scores = attention_scores + self.relative_position_bias().unsqueeze(0)

        # 如果给定了 shared relative position bias，也将其加入注意力分数中
        if relative_position_bias is not None:
            attention_scores = attention_scores + relative_position_bias

        # 将注意力分数归一化为概率分布
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用 dropout 方法对注意力概率进行随机失活处理
        attention_probs = self.dropout(attention_probs)

        # 如果给定了 head_mask，将其应用到 attention_probs 上
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算加权后的值向量，得到上下文向量 context_layer
        context_layer = torch.matmul(attention_probs, value_layer)

        # 对 context_layer 进行维度重排，以符合后续计算要求
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 根据输出设置返回结果，包括 context_layer 和 attention_probs（如果需要）
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
class BeitSelfOutput(nn.Module):
    """
    The residual connection is defined in BeitLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        # Linear transformation for the output of self-attention
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, gamma=None) -> torch.Tensor:
        # Linear transformation
        hidden_states = self.dense(hidden_states)
        # Apply dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class BeitAttention(nn.Module):
    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None) -> None:
        super().__init__()
        # Self-attention mechanism
        self.attention = BeitSelfAttention(config, window_size=window_size)
        # Output layer after attention
        self.output = BeitSelfOutput(config)
        # Set of pruned attention heads
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # Find and prune attention heads based on indices
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers for attention components
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update number of attention heads and related sizes, and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional["BeitRelativePositionBias"] = None,
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        # Perform self-attention
        self_outputs = self.attention(hidden_states, head_mask, output_attentions, relative_position_bias)

        # Output of attention passed through output layer
        attention_output = self.output(self_outputs[0], hidden_states)

        # Collect outputs, including attention matrices if requested
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BeitIntermediate(nn.Module):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        # Intermediate dense layer
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # Activation function for intermediate layer
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 定义一个方法 `forward`，用于前向传播计算
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态 `hidden_states` 经过全连接层 `dense` 处理
        hidden_states = self.dense(hidden_states)
        # 对处理后的隐藏状态应用激活函数 `intermediate_act_fn`
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的隐藏状态作为输出
        return hidden_states
class BeitOutput(nn.Module):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        # 创建一个全连接层，将输入特征的维度缩放为隐藏层大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 定义一个dropout层，用于随机置零输入张量的一些元素，以减少过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入张量传入全连接层，执行线性变换
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出执行dropout操作
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class BeitLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        # 设置用于分块feed forward的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度
        self.seq_len_dim = 1
        # 使用给定配置和窗口大小创建注意力机制
        self.attention = BeitAttention(config, window_size=window_size)
        # 创建中间层对象，将输入特征映射到隐藏层大小
        self.intermediate = BeitIntermediate(config)
        # 创建输出层对象，将中间层的输出映射到最终的隐藏层大小
        self.output = BeitOutput(config)
        # 应用LayerNorm在隐藏层上，以归一化特征
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 根据dropout路径率初始化drop path对象，如果路径率大于0
        self.drop_path = BeitDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        # 再次应用LayerNorm在隐藏层上，以归一化特征
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化lambda参数，如果初始值大于0，则创建可学习的参数张量
        init_values = config.layer_scale_init_value
        if init_values > 0:
            self.lambda_1 = nn.Parameter(init_values * torch.ones((config.hidden_size)), requires_grad=True)
            self.lambda_2 = nn.Parameter(init_values * torch.ones((config.hidden_size)), requires_grad=True)
        else:
            self.lambda_1, self.lambda_2 = None, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        relative_position_bias: Optional["BeitRelativePositionBias"] = None,
        ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        # 使用 self.attention 对 hidden_states 应用自注意力机制
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在 BEiT 模型中，先对 hidden_states 应用 layernorm
            head_mask,
            output_attentions=output_attentions,
            relative_position_bias=relative_position_bias,
        )
        # 获取自注意力输出
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重，则添加自注意力结果

        # 如果定义了 lambda_1，则对 attention_output 应用缩放
        if self.lambda_1 is not None:
            attention_output = self.lambda_1 * attention_output

        # 第一个残差连接
        hidden_states = self.drop_path(attention_output) + hidden_states

        # 在 BEiT 中，还会在自注意力后应用 layernorm
        layer_output = self.layernorm_after(hidden_states)

        # 经过中间层和输出层
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output)

        # 如果定义了 lambda_2，则对 layer_output 应用缩放
        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # 第二个残差连接
        layer_output = self.drop_path(layer_output) + hidden_states

        # 整合最终输出
        outputs = (layer_output,) + outputs

        return outputs
class BeitRelativePositionBias(nn.Module):
    def __init__(self, config: BeitConfig, window_size: tuple) -> None:
        super().__init__()
        self.window_size = window_size
        # 计算相对位置偏置表的大小
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        # 创建一个可学习的参数，用于存储相对位置偏置表，大小为 num_relative_distance x num_attention_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, config.num_attention_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # 用于描述cls到token、token到cls、cls到cls之间的相对位置关系

        # 获取每个窗口内每个token之间的pair-wise相对位置索引
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # 将坐标向左移动
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        # 将相对位置索引注册为非参数化缓冲区
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def forward(self) -> torch.Tensor:
        # 根据相对位置索引从相对位置偏置表中获取相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1
        )  # Wh*Ww,Wh*Ww,nH

        # 返回维度变换后的相对位置偏置，维度顺序为 nH, Wh*Ww, Wh*Ww
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    def __init__(self, config: BeitConfig, window_size: Optional[tuple] = None) -> None:
        super().__init__()
        self.config = config
        # 如果配置中使用共享的相对位置偏置，则创建相对位置偏置对象
        if config.use_shared_relative_position_bias:
            self.relative_position_bias = BeitRelativePositionBias(config, window_size=window_size)
        else:
            self.relative_position_bias = None

        # 根据随机深度衰减规则生成每个层的衰减率列表
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        # 创建神经网络层的列表，每层使用不同的衰减率和配置
        self.layer = nn.ModuleList(
            [
                BeitLayer(
                    config,
                    window_size=window_size if config.use_relative_position_bias else None,
                    drop_path_rate=dpr[i],
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        # 梯度检查点功能默认关闭
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 初始化空元组以保存所有隐藏状态和注意力分数
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 遍历每个神经网络层进行前向传播
        for i, layer_module in enumerate(self.layer):
            # 如果需要记录隐藏状态，则将当前隐藏状态添加到列表中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用梯度检查点且在训练模式下，则使用梯度检查点函数调用当前层
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 获取相对位置偏置（如果存在）并传递给当前层
                relative_position_bias = (
                    self.relative_position_bias() if self.relative_position_bias is not None else None
                )
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, relative_position_bias)

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力分数，则将当前层的注意力分数添加到列表中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要记录隐藏状态，则将最终隐藏状态添加到列表中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据返回类型决定输出格式
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
BEIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`BeitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
    # "The bare Beit Model transformer outputting raw hidden-states without any specific head on top."
    # BEIT_START_DOCSTRING,
    )
class BeitModel(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig, add_pooling_layer: bool = True) -> None:
        super().__init__(config)
        self.config = config

        # 初始化嵌入层和编码器
        self.embeddings = BeitEmbeddings(config)
        self.encoder = BeitEncoder(config, window_size=self.embeddings.patch_embeddings.patch_shape)

        # 根据配置选择性地添加层归一化或池化层
        self.layernorm = (
            nn.Identity() if config.use_mean_pooling else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.pooler = BeitPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回嵌入层的补丁嵌入
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要修剪的层和头部，并在注意力机制中执行修剪
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BeitModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BeitModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # Determine whether to return attentions, hidden states, etc., based on input or default config
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Validate input: pixel_values must be specified
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicates that the head is kept active during attention
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # head_mask is reshaped to [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # Embedding process: computes embeddings from pixel values and masked positions
        embedding_output, (patch_height, patch_width) = self.embeddings(pixel_values, bool_masked_pos)

        # Encoder block: applies transformer encoding to embedding output
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Extract sequence output from encoder output and normalize using layer normalization
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        # Pooler layer: computes pooled output if pooler is defined
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # Return different outputs based on return_dict flag
        if not return_dict:
            # Return tuple of sequence output and pooled output (if available) along with other encoder outputs
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # Return structured output using BeitModelOutputWithPooling
        return BeitModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 定义一个名为 `BeitPooler` 的神经网络模块，用于对 BEiT 模型的隐藏状态进行池化操作
class BeitPooler(nn.Module):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        # 如果配置要求使用均值池化，则使用 LayerNorm 对隐藏状态进行归一化
        self.layernorm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if config.use_mean_pooling else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.layernorm is not None:
            # 如果存在 LayerNorm 对象，则对补丁令牌的最终隐藏状态进行均值池化
            patch_tokens = hidden_states[:, 1:, :]  # 选择除了第一个令牌外的所有令牌的隐藏状态
            pooled_output = self.layernorm(patch_tokens.mean(1))  # 对补丁令牌的隐藏状态进行均值池化并归一化
        else:
            # 否则，通过简单地使用 [CLS] 令牌的最终隐藏状态进行池化
            pooled_output = hidden_states[:, 0]  # 选择 [CLS] 令牌的最终隐藏状态作为池化输出

        return pooled_output


@add_start_docstrings(
    """Beit Model transformer with a 'language' modeling head on top. BEiT does masked image modeling by predicting
    visual tokens of a Vector-Quantize Variational Autoencoder (VQ-VAE), whereas other vision models like ViT and DeiT
    predict RGB pixel values. As a result, this class is incompatible with [`AutoModelForMaskedImageModeling`], so you
    will need to use [`BeitForMaskedImageModeling`] directly if you wish to do masked image modeling with BEiT.""",
    BEIT_START_DOCSTRING,
)
# 定义一个带有语言建模头部的 BEiT 模型变压器。BEiT 通过预测矢量量化变分自动编码器（VQ-VAE）的视觉令牌来进行遮罩图像建模，而像 ViT 和 DeiT 这样的其他视觉模型预测 RGB 像素值。因此，此类与 [`AutoModelForMaskedImageModeling`] 不兼容，如果要使用 BEiT 进行遮罩图像建模，您需要直接使用 [`BeitForMaskedImageModeling`]。
class BeitForMaskedImageModeling(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=False)  # 初始化 BEiT 模型，不添加池化层

        # 分类器头部
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 使用 LayerNorm 对隐藏状态进行归一化
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)  # 线性层用于语言模型的预测

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    # 重写 `forward` 方法，用于模型的前向传播
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,  # 像素值，可选输入
        bool_masked_pos: Optional[torch.BoolTensor] = None,  # 遮罩位置的布尔张量，可选输入
        head_mask: Optional[torch.Tensor] = None,  # 头部遮罩，可选输入
        labels: Optional[torch.Tensor] = None,  # 标签，可选输入
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选输入
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选输入
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可选输入
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 return_dict 不为 None，则使用其值；否则使用 self.config.use_return_dict 的值

        outputs = self.beit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 调用 self.beit 方法，传入像素数值 pixel_values 和其他参数，根据 return_dict 是否为真决定是否返回字典形式的输出

        sequence_output = outputs[0]
        # 从模型输出中取得序列输出

        sequence_output = self.layernorm(sequence_output)
        # 对序列输出进行 layer normalization 处理

        prediction_scores = self.lm_head(sequence_output[:, 1:])
        # 使用 lm_head 对序列输出的部分进行预测评分，通常是用来生成模型的输出结果

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 定义交叉熵损失函数，用于计算损失
            masked_lm_loss = loss_fct(prediction_scores[bool_masked_pos], labels)
            # 如果给定了标签 labels，则计算被遮蔽位置 bool_masked_pos 的预测结果与标签之间的损失

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            # 如果不要求返回字典形式的输出，则返回预测分数和其他附加输出

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 如果需要返回字典形式的输出，则返回一个 MaskedLMOutput 对象，包含损失、预测分数、隐藏状态和注意力权重信息
@add_start_docstrings(
    """
    Beit Model transformer with an image classification head on top (a linear layer on top of the average of the final
    hidden states of the patch tokens) e.g. for ImageNet.
    """,
    BEIT_START_DOCSTRING,
)
class BeitForImageClassification(BeitPreTrainedModel):
    def __init__(self, config: BeitConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        # 使用配置初始化 BeitModel，添加池化层以便用于分类任务
        self.beit = BeitModel(config, add_pooling_layer=True)

        # 分类器头部，根据配置的隐藏层大小和标签数量初始化线性层或者恒等映射
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
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
    ):
        # 模型前向传播方法，接收像素值、头部掩码、标签等参数，返回模型输出
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 根据需要决定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 使用 BEiT 模型进行推理
        outputs = self.beit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 根据返回值是否为字典形式，选择 pooled_output
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 使用分类器对 pooled_output 进行分类得到 logits
        logits = self.classifier(pooled_output)

        # 初始化损失为 None
        loss = None
        # 如果有标签输入
        if labels is not None:
            # 如果问题类型未指定，则根据标签类型和数量确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择损失函数和计算损失
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

        # 如果不使用字典形式返回结果，则组装输出并返回
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果使用字典形式返回结果，则创建 ImageClassifierOutput 对象并返回
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class BeitConvModule(nn.Module):
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
        # 定义卷积层，设置输入输出通道数、核大小、填充、是否有偏置、扩张率等参数
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        # 定义批归一化层，设置输出通道数
        self.bn = nn.BatchNorm2d(out_channels)
        # 定义激活函数层为ReLU
        self.activation = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 前向传播函数，依次经过卷积、批归一化和ReLU激活
        output = self.conv(input)
        output = self.bn(output)
        output = self.activation(output)

        return output


class BeitPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        super().__init__()
        # 创建自适应平均池化层和BeitConvModule卷积模块，并将其作为列表存储在self.layers中
        self.layers = [
            nn.AdaptiveAvgPool2d(pool_scale),
            BeitConvModule(in_channels, channels, kernel_size=1),
        ]
        # 将每个层添加为模块
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_state = input
        # 依次对输入数据应用self.layers中的每个层，并返回最终的隐藏状态
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class BeitPyramidPoolingModule(nn.Module):
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

    def __init__(self, pool_scales: Tuple[int, ...], in_channels: int, channels: int, align_corners: bool) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = []
        # 根据给定的pool_scales创建多个BeitPyramidPoolingBlock模块，并添加为子模块
        for i, pool_scale in enumerate(pool_scales):
            block = BeitPyramidPoolingBlock(pool_scale=pool_scale, in_channels=in_channels, channels=channels)
            self.blocks.append(block)
            self.add_module(str(i), block)
    # 定义前向传播方法，接受一个张量 x 作为输入，并返回一个张量列表
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # 初始化一个空列表，用于存储各个 PPM 模块的输出张量
        ppm_outs = []
        # 遍历 self.blocks 中的每个 PPM 模块
        for ppm in self.blocks:
            # 对输入 x 应用当前的 PPM 模块，得到该模块的输出张量 ppm_out
            ppm_out = ppm(x)
            # 使用双线性插值方法将 ppm_out 上采样到与输入 x 相同的大小
            upsampled_ppm_out = nn.functional.interpolate(
                ppm_out, size=x.size()[2:], mode="bilinear", align_corners=self.align_corners
            )
            # 将上采样后的 ppm_out 添加到 ppm_outs 列表中
            ppm_outs.append(upsampled_ppm_out)
        # 返回所有 PPM 模块的输出张量组成的列表 ppm_outs
        return ppm_outs
# 定义一个名为 `BeitUperHead` 的类，继承自 `nn.Module`，用于实现场景理解的统一感知解析。
class BeitUperHead(nn.Module):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    # 初始化方法，接收一个 `BeitConfig` 类型的配置参数 `config`
    def __init__(self, config: BeitConfig) -> None:
        super().__init__()

        # 设置池化尺度，例如 (1, 2, 3, 6)
        self.pool_scales = config.pool_scales  # e.g. (1, 2, 3, 6)
        # 设置输入通道数列表，全为 `config.hidden_size`，例如 [768, 768, 768, 768]
        self.in_channels = [config.hidden_size] * 4  # e.g. [768, 768, 768, 768]
        # 设置通道数为 `config.hidden_size`
        self.channels = config.hidden_size
        # 是否对齐角点，默认为 False
        self.align_corners = False
        # 分类器，使用 1x1 卷积将通道数从 `self.channels` 转换为 `config.num_labels`
        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

        # PSP Module，使用 `BeitPyramidPoolingModule` 初始化池化模块
        self.psp_modules = BeitPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],  # 最后一个输入通道数
            self.channels,
            align_corners=self.align_corners,
        )
        # 瓶颈模块，使用 `BeitConvModule` 初始化卷积模块
        self.bottleneck = BeitConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        
        # FPN Module，构建特征金字塔网络模块
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        # 遍历除了顶层之外的所有输入通道数
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            # 使用 `BeitConvModule` 初始化侧边卷积模块
            l_conv = BeitConvModule(in_channels, self.channels, kernel_size=1)
            # 使用 `BeitConvModule` 初始化金字塔卷积模块
            fpn_conv = BeitConvModule(self.channels, self.channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)  # 添加到侧边卷积模块列表
            self.fpn_convs.append(fpn_conv)    # 添加到金字塔卷积模块列表

        # FPN 瓶颈模块，使用 `BeitConvModule` 初始化卷积模块
        self.fpn_bottleneck = BeitConvModule(
            len(self.in_channels) * self.channels,  # 所有输入通道数的总和
            self.channels,
            kernel_size=3,
            padding=1,
        )

    # PSP 前向传播方法，接收输入 `inputs`，返回处理后的输出
    def psp_forward(self, inputs):
        x = inputs[-1]  # 取输入列表的最后一个元素作为输入
        psp_outs = [x]  # 初始化 PSP 输出列表
        psp_outs.extend(self.psp_modules(x))  # 将 PSP 模块的输出扩展到 PSP 输出列表
        psp_outs = torch.cat(psp_outs, dim=1)  # 在通道维度上连接 PSP 输出
        output = self.bottleneck(psp_outs)  # 使用瓶颈模块处理 PSP 输出

        return output  # 返回处理后的输出
    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # 构建侧边连接
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # 将PSP模块的输出添加到侧边连接列表中
        laterals.append(self.psp_forward(encoder_hidden_states))

        # 构建自顶向下路径
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )

        # 构建FPN输出
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        
        # 将PSP特征追加到FPN输出列表中
        fpn_outs.append(laterals[-1])

        # 对FPN输出进行上采样
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )

        # 在通道维度上连接所有FPN输出
        fpn_outs = torch.cat(fpn_outs, dim=1)

        # 经过FPN瓶颈层处理
        output = self.fpn_bottleneck(fpn_outs)

        # 使用分类器处理最终输出
        output = self.classifier(output)

        # 返回最终结果
        return output
# 定义一个用于语义分割的头部模块，基于 Fully Convolution Networks（FCN）的设计。
# 详见论文 [FCNNet](https://arxiv.org/abs/1411.4038>)。
class BeitFCNHead(nn.Module):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is implemented of
    [FCNNet](https://arxiv.org/abs/1411.4038>).

    Args:
        config (BeitConfig): Configuration.
        in_index (int): Index of the encoder hidden state to use as input. Default: 2.
        kernel_size (int): The kernel size for convolutions in the head. Default: 3.
        dilation (int or tuple): The dilation rate for convolutions in the head. Default: 1.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(
        self, config: BeitConfig, in_index: int = 2, kernel_size: int = 3, dilation: Union[int, Tuple[int, int]] = 1
    ) -> None:
        super().__init__()
        # 初始化头部模块的参数
        self.in_channels = config.hidden_size  # 输入通道数等于隐藏状态的大小
        self.channels = config.auxiliary_channels  # 辅助通道数
        self.num_convs = config.auxiliary_num_convs  # 卷积层的数量
        self.concat_input = config.auxiliary_concat_input  # 是否将输入与卷积输出拼接的标志
        self.in_index = in_index  # 输入隐藏状态的索引位置

        # 计算卷积层的填充大小
        conv_padding = (kernel_size // 2) * dilation
        convs = []
        
        # 添加第一个卷积模块
        convs.append(
            BeitConvModule(
                self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
            )
        )
        
        # 添加剩余的卷积模块
        for i in range(self.num_convs - 1):
            convs.append(
                BeitConvModule(
                    self.channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                )
            )
        
        # 如果没有卷积层，则使用 nn.Identity 作为卷积层
        if self.num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        
        # 如果设置了拼接输入标志，则创建用于拼接的卷积模块
        if self.concat_input:
            self.conv_cat = BeitConvModule(
                self.in_channels + self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2
            )
        
        # 分类器，最终输出的通道数为配置中的标签数量
        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # 从编码器隐藏状态中取出指定的特征图
        hidden_states = encoder_hidden_states[self.in_index]
        
        # 经过卷积层处理
        output = self.convs(hidden_states)
        
        # 如果设置了拼接输入标志，则将原始输入与卷积输出拼接后再进行处理
        if self.concat_input:
            output = self.conv_cat(torch.cat([hidden_states, output], dim=1))
        
        # 最后经过分类器输出结果
        output = self.classifier(output)
        return output


@add_start_docstrings(
    """
    Beit Model transformer with a semantic segmentation head on top e.g. for ADE20k, CityScapes.
    """,
    BEIT_START_DOCSTRING,
)
class BeitForSemanticSegmentation(BeitPreTrainedModel):
    """
    Beit Model transformer with a semantic segmentation head for tasks like ADE20k, CityScapes.

    Inherits from BeitPreTrainedModel, which is the base class for all Beit models.
    """
    def __init__(self, config: BeitConfig) -> None:
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)

        # 从配置对象中获取标签数量
        self.num_labels = config.num_labels
        # 创建一个 BEiT 模型对象，不添加池化层
        self.beit = BeitModel(config, add_pooling_layer=False)

        # FPNs
        # 检查配置中的输出索引是否为四个整数，否则抛出数值错误异常
        if len(self.config.out_indices) != 4:
            raise ValueError(
                "BeitForSemanticSegmentation requires config.out_indices to be a list of 4 integers, "
                "specifying which features to use from the backbone. One can use [3, 5, 7, 11] in case of "
                "a base-sized architecture."
            )
        # 定义语义分割头部网络的几个转置卷积操作序列
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
            nn.BatchNorm2d(config.hidden_size),
            nn.GELU(),
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Identity()  # 直接返回输入的恒等映射
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化操作，核大小为2x2

        # 语义分割头部网络
        self.decode_head = BeitUperHead(config)  # 创建解码头部对象
        self.auxiliary_head = BeitFCNHead(config) if config.use_auxiliary_head else None  # 如果配置中启用辅助头部，则创建辅助头部对象，否则为 None

        # 初始化权重并应用最终处理
        self.post_init()

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

    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用 add_start_docstrings 装饰器添加 BEiT 的背景说明文档，用于与 DETR 和 MaskFormer 等框架集成
@add_start_docstrings(
    """
    BEiT backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    BEIT_START_DOCSTRING,
)
# 定义 BEiT 的骨干网络类，继承自 BeitPreTrainedModel 和 BackboneMixin
class BeitBackbone(BeitPreTrainedModel, BackboneMixin):
    # 初始化函数，接受一个 config 对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 调用 BackboneMixin 类的初始化方法，初始化骨干网络
        super()._init_backbone(config)

        # 根据配置设置特征的数量为隐藏层大小的列表
        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        # 初始化嵌入层
        self.embeddings = BeitEmbeddings(config)
        # 初始化编码器，并传递嵌入层的窗口大小作为参数
        self.encoder = BeitEncoder(config, window_size=self.embeddings.patch_embeddings.patch_shape)

        # 如果配置中指定要添加 FPN
        if config.add_fpn:
            # 检查配置中的输出索引列表是否包含四个整数
            if len(self.config.out_indices) != 4:
                # 如果不是，则抛出数值错误异常
                raise ValueError(
                    "BeitBackbone requires config.out_indices to be a list of 4 integers, "
                    "specifying which features to use from the backbone. One can use [3, 5, 7, 11] in case of "
                    "a base-sized architecture."
                )
            # 获取隐藏层大小
            hidden_size = config.hidden_size
            # 初始化 FPN1，包括两个转置卷积层和批归一化层，使用 GELU 激活函数
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2),
                nn.BatchNorm2d(hidden_size, eps=config.batch_norm_eps),
                nn.GELU(),
                nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2),
            )

            # 初始化 FPN2，包括一个转置卷积层
            self.fpn2 = nn.Sequential(nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=2, stride=2))
            # 初始化 FPN3，为恒等映射层
            self.fpn3 = nn.Identity()
            # 初始化 FPN4，为最大池化层
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 重写 forward 方法，接受像素值张量和可选的输出隐藏状态、注意力和返回字典作为参数
    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        """
        如果 return_dict 不为 None，则使用其值；否则使用 self.config.use_return_dict 的值作为返回结果的字典选择
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        """
        如果 output_hidden_states 不为 None，则使用其值；否则使用 self.config.output_hidden_states 的值作为输出隐藏状态的选择
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        """
        如果 output_attentions 不为 None，则使用其值；否则使用 self.config.output_attentions 的值作为输出注意力的选择
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        """
        获取输入像素值的批次大小
        """
        batch_size = pixel_values.shape[0]
        """
        使用 self.embeddings 处理像素值，得到嵌入输出和每个补丁的高度和宽度
        """
        embedding_output, (patch_height, patch_width) = self.embeddings(pixel_values)

        """
        使用 self.encoder 处理嵌入输出，设置输出隐藏状态和注意力的选择，根据 return_dict 决定是否返回字典
        """
        outputs = self.encoder(
            embedding_output, output_hidden_states=True, output_attentions=output_attentions, return_dict=return_dict
        )

        """
        如果 return_dict 为 True，则将隐藏状态存储在 outputs.hidden_states 中；否则在 outputs[1] 中
        """
        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        """
        初始化空的特征映射元组
        """
        feature_maps = ()
        """
        遍历阶段名称和隐藏状态，根据设定的输出特征名称收集对应的隐藏状态
        """
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                """
                如果 self.config.reshape_hidden_states 为 True，则对隐藏状态进行形状调整
                """
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, 1:, :]  # 移除CLS标记
                    hidden_state = hidden_state.permute(0, 2, 1)  # 调整维度顺序
                    hidden_state = hidden_state.reshape(batch_size, -1, patch_height, patch_width)  # 重塑形状

                """
                将符合条件的隐藏状态添加到特征映射中
                """
                feature_maps += (hidden_state,)

        """
        如果配置中添加了特征金字塔网络（FPN），则对特征映射进行相应的处理
        """
        if self.config.add_fpn:
            feature_maps = [
                self.fpn1(feature_maps[0]),
                self.fpn2(feature_maps[1]),
                self.fpn3(feature_maps[2]),
                self.fpn4(feature_maps[3]),
            ]
            feature_maps = tuple(feature_maps)

        """
        如果不返回字典，则根据输出隐藏状态的设置返回输出元组
        """
        if not return_dict:
            if output_hidden_states:
                output = (feature_maps,) + outputs[1:]
            else:
                output = (feature_maps,) + outputs[2:]
            return output

        """
        返回 BackboneOutput 对象，包含特征映射、隐藏状态和注意力
        """
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
```