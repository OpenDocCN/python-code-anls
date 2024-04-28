# `.\transformers\models\vilt\modeling_vilt.py`

```py
# 设置文件编码为 utf-8
# 版权声明
# 根据Apache License，Version 2.0授权，除非遵守许可证，否则不得使用本文件
# 可以从以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 如适用法律或书面协议要求，按"原样"分发的软件，没有任何保证或条件，无论是明示的还是默示的
# 请查看许可证以了解特定语言的权限和限制
""" PyTorch ViLT 模型。"""

# 导入所需库
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import (
    find_pruneable_heads_and_indices,
    meshgrid,
    prune_linear_layer,
)
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vilt import ViltConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档中使用的配置和检查点
_CONFIG_FOR_DOC = "ViltConfig"
_CHECKPOINT_FOR_DOC = "dandelin/vilt-b32-mlm"

# 预训练模型列表
VILT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "dandelin/vilt-b32-mlm",
    # 查看所有 ViLT 模型: https://huggingface.co/models?filter=vilt
]


@dataclass
class ViltForImagesAndTextClassificationOutput(ModelOutput):
    """
    Class for outputs of [`ViltForImagesAndTextClassification`].
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            # 损失值，如果提供了`labels`，则返回分类（或回归，如果config.num_labels==1）的损失值。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            # 分类（或回归，如果config.num_labels==1）得分（SoftMax之前）。
        hidden_states (`List[tuple(torch.FloatTensor)]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            # 隐藏状态列表，包含每个图像-文本对应的元组的`torch.FloatTensor`（每个元组包含嵌入输出和每一层的输出），形状为`(batch_size, sequence_length, hidden_size)`。
            # 模型在每一层输出的隐藏状态以及初始嵌入输出。
        attentions (`List[tuple(torch.FloatTensor)]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            # 注意力列表，包含每个图像-文本对应的元组的`torch.FloatTensor`（每个元组包含形状为`(batch_size, num_heads, sequence_length, sequence_length)`的注意力权重）。
            # 注意力softmax后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    # 损失值，默认为None
    logits: torch.FloatTensor = None
    # 分类得分，默认为None
    hidden_states: Optional[List[Tuple[torch.FloatTensor]]] = None
    # 隐藏状态，默认为None
    attentions: Optional[List[Tuple[torch.FloatTensor]]] = None
    # 注意力，默认为None
class ViltEmbeddings(nn.Module):
    """
    Construct the text and patch embeddings.

    Text embeddings are equivalent to BERT embeddings.

    Patch embeddings are equivalent to ViT embeddings.
    """

    def __init__(self, config):
        super().__init__()

        # text embeddings
        self.text_embeddings = TextEmbeddings(config)
        # patch embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ViltPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        # modality type (text/patch) embeddings
        self.token_type_embeddings = nn.Embedding(config.modality_type_vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        pixel_values,
        pixel_mask,
        inputs_embeds,
        image_embeds,
        image_token_type_idx=1,
    ):
        # PART 1: text embeddings
        text_embeds = self.text_embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # PART 2: patch embeddings (with interpolated position encodings)
        if image_embeds is None:
            image_embeds, image_masks, patch_index = self.visual_embed(
                pixel_values, pixel_mask, max_image_length=self.config.max_image_length
            )
        else:
            image_masks = pixel_mask.flatten(1)

        # PART 3: add modality type embeddings
        # 0 indicates text, 1 indicates image, 2 is optionally used when a second image is provided (NLVR2)
        if image_token_type_idx is None:
            image_token_type_idx = 1
        text_embeds = text_embeds + self.token_type_embeddings(
            torch.zeros_like(attention_mask, dtype=torch.long, device=text_embeds.device)
        )
        image_embeds = image_embeds + self.token_type_embeddings(
            torch.full_like(image_masks, image_token_type_idx, dtype=torch.long, device=text_embeds.device)
        )

        # PART 4: concatenate
        embeddings = torch.cat([text_embeds, image_embeds], dim=1)
        masks = torch.cat([attention_mask, image_masks], dim=1)

        return embeddings, masks


class TextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__()
        # 创建词嵌入层，vocab_size表示词汇表大小，hidden_size表示隐藏层大小，padding_idx表示填充标记的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，max_position_embeddings表示位置嵌入的最大位置数，hidden_size表示隐藏层大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建标记类型嵌入层，type_vocab_size表示标记类型的数量，hidden_size表示隐藏层大小
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 使用 TensorFlow 模型变量名称以及能够加载任何 TensorFlow 检查点文件的方式命名 self.LayerNorm
        # self.LayerNorm不使用蛇形命名，以保持与 TensorFlow 模型变量名称一致，并能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 dropout 层，使用指定的 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 设置位置嵌入类型，默认为"absolute"，即绝对位置嵌入
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册缓冲区，用于存储位置标记的索引，序列化时会导出 position_ids (1, len position emb) 是内存中的连续值
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册缓冲区，用于存储标记类型的索引，初始值全为零
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    # 前向传播函数
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # 如果传入了 input_ids，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果未提供位置标记，则使用事先注册的位置标记
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果未提供标记类型，则设置为全零
        if token_type_ids is None:
            # 如果已经定义了 token_type_ids，则使用事先注册的 token_type_ids
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            # 否则创建全零的 token_type_ids
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供输入嵌入，则使用词嵌入层生成输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取标记类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将词嵌入和标记类型嵌入相加得到总的嵌入
        embeddings = inputs_embeds + token_type_embeddings
        # 如果使用绝对位置嵌入，则添加位置嵌入
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # LayerNorm 正则化
        embeddings = self.LayerNorm(embeddings)
        # 使用 dropout
        embeddings = self.dropout(embeddings)
        # 返回嵌入
        return embeddings
# 图像到补丁嵌入层
class ViltPatchEmbeddings(nn.Module):
    """
    图像到补丁嵌入。
    """

    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
        # 获取配置中的图像尺寸和补丁尺寸
        image_size, patch_size = config.image_size, config.patch_size
        # 获取配置中的通道数和隐藏层大小
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 确保图像尺寸和补丁尺寸为可迭代的
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算补丁数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        # 保存属性
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 定义卷积层进行图像到补丁的映射
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        # 获取输入的批量大小、通道数、高度和宽度
        batch_size, num_channels, height, width = pixel_values.shape
        # 检查输入通道数是否与配置一致
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 使用卷积层进行图像到补丁的映射
        x = self.projection(pixel_values)
        return x


# 多头自注意力模块
class ViltSelfAttention(nn.Module):
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
        # 检查隐藏层大小是否能被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 保存注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键和值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 定义注意力权重dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # 将输入 x 重塑为多头注意力的形状
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # 将通道维度移动到第二个维度
        return x.permute(0, 2, 1, 3)
    # 定义一个前向传播函数，接收隐藏状态、注意力掩码、头掩码、是否输出注意力矩阵作为参数
    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        # 通过查询网络计算混合查询层
        mixed_query_layer = self.query(hidden_states)

        # 将键层进行转置以便进行计算
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 将值层进行转置以便进行计算
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 将查询层进行转置以便进行计算
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始的注意力分数，通过"查询"和"键"的点积来得到
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 对得到的注意力分数进行归一化处理
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # 如果有注意力掩码，则应用它
            attention_scores = attention_scores + attention_mask

        # 将注意力分数归一化为概率值
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # 通过dropout函数进行整个token的dropout
        attention_probs = self.dropout(attention_probs)

        # 如果需要，对头进行掩码操作
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，通过注意力概率和值层的矩阵相乘
        context_layer = torch.matmul(attention_probs, value_layer)

        # 对上下文层进行维度置换和重塑
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 根据是否输出注意力矩阵来返回相应结果
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 从transformers.models.vit.modeling_vit.ViTSelfOutput复制而来，将ViT替换为Vilt
class ViltSelfOutput(nn.Module):
    """
    The residual connection is defined in ViltLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViltConfig) -> None:
        super().__init__()
        # 定义线性层，用于变换隐藏状态的维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义Dropout层，用于随机置零部分输入单元
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过线性层变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态进行随机置零
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViltAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义自注意力层
        self.attention = ViltSelfAttention(config)
        # 定义自注意力输出层
        self.output = ViltSelfOutput(config)
        # 用于存储被剪枝的注意力头索引的集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到需要被剪枝的注意力头的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头索引
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        # 通过自注意力层进行前向传播
        self_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions)

        # 将自注意力层的输出通过自注意力输出层处理
        attention_output = self.output(self_outputs[0], hidden_states)

        # 如果需要输出注意力权重，则将其加入到输出中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# 从transformers.models.vit.modeling_vit.ViTIntermediate复制而来，将ViT替换为Vilt
class ViltIntermediate(nn.Module):
    def __init__(self, config: ViltConfig) -> None:
        super().__init__()
        # 定义线性层，用于变换隐藏状态的维度
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果配置中隐藏激活函数为字符串，则使用相应的激活函数，否则使用配置中指定的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过线性层变换
        hidden_states = self.dense(hidden_states)
        # 将变换后的隐藏状态通过激活函数处理
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states
# 从transformers.models.vit.modeling_vit.ViTOutput复制代码，并将ViT更改为Vilt
class ViltOutput(nn.Module):
    def __init__(self, config: ViltConfig) -> None:
        super().__init__()
        # 创建一个线性层，将输入维度转换为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个dropout层，用于随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过线性层转换hidden_states的维度
        hidden_states = self.dense(hidden_states)
        # 对转换后的hidden_states进行dropout操作
        hidden_states = self.dropout(hidden_states)

        # 将转换后的hidden_states与输入的input_tensor相加
        hidden_states = hidden_states + input_tensor

        return hidden_states


class ViltLayer(nn.Module):
    """这对应于timm实现中的Block类。"""

    def __init__(self, config):
        super().__init__()
        # 初始化ViltLayer类的属性
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViltAttention(config)
        self.intermediate = ViltIntermediate(config)
        self.output = ViltOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        # 对hidden_states应用layernorm，然后传入self-attention模块
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加self attentions

        # 第一个残差连接
        hidden_states = attention_output + hidden_states.to(attention_output.device)

        # 在ViLT中，也在self-attention之后应用layernorm
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接在这里完成
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class ViltEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建一个包含config.num_hidden_layers个ViltLayer实例的ModuleList
        self.layer = nn.ModuleList([ViltLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):
        # 如果不输出隐藏状态，则初始化一个空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力权重，则初始化一个空元组
        all_self_attentions = () if output_attentions else None

        # 遍历每个 Transformer 层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用了梯度检查点并且处于训练模式，则使用梯度检查点函数
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的前向传播函数
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，则返回包含隐藏状态、所有隐藏状态和所有注意力权重的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回包含最终隐藏状态、所有隐藏状态和所有注意力权重的 BaseModelOutput 对象
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class ViltPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # ViltPreTrainedModel 类继承自 PreTrainedModel 类，用于处理权重初始化和预训练模型的下载和加载
    config_class = ViltConfig
    # 配置类为 ViltConfig
    base_model_prefix = "vilt"
    # 基础模型前缀为 "vilt"
    supports_gradient_checkpointing = True
    # 支持梯度检查点
    _no_split_modules = ["ViltEmbeddings", "ViltSelfAttention"]
    # 不分割的模块列表包括 "ViltEmbeddings" 和 "ViltSelfAttention"

    def _init_weights(self, module):
        """Initialize the weights"""
        # 初始化权重的函数
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 如果模块是线性层或卷积层
            # 与 TF 版本略有不同，使用正态分布初始化权重
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 初始化权重数据为正态分布
            if module.bias is not None:
                module.bias.data.zero_()
                # 如果存在偏置，则初始化为零
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果模块是嵌入层，则初始化权重数据为正态分布
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                # 如果存在填充索引，则将对应位置的权重初始化为零
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            # 如果模块是 LayerNorm 层，则初始化偏置为零
            module.weight.data.fill_(1.0)
            # 初始化权重为全 1

VILT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ subclass. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViltConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
# VILT_START_DOCSTRING 为模型的文档字符串，介绍了模型是 PyTorch 的 torch.nn.Module 子类，如何使用以及参数说明

VILT_INPUTS_DOCSTRING = r"""
"""
# VILT_INPUTS_DOCSTRING 为输入文档字符串

VILT_IMAGES_AND_TEXT_CLASSIFICATION_INPUTS_DOCSTRING = r"""
"""
# VILT_IMAGES_AND_TEXT_CLASSIFICATION_INPUTS_DOCSTRING 为图像和文本分类输入文档字符串

@add_start_docstrings(
    "The bare ViLT Model transformer outputting raw hidden-states without any specific head on top.",
    VILT_START_DOCSTRING,
)
# 添加文档字符串，描述裸 ViLT 模型变换器输出原始隐藏状态，没有特定的输出头

class ViltModel(ViltPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        # 调用父类的初始化方法
        self.config = config

        self.embeddings = ViltEmbeddings(config)
        # 初始化嵌入层
        self.encoder = ViltEncoder(config)
        # 初始化编码器

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 LayerNorm 层
        self.pooler = ViltPooler(config) if add_pooling_layer else None
        # 如果需要添加池化层，则初始化池化层，否则为 None

        # Initialize weights and apply final processing
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.text_embeddings.word_embeddings
        # 获取输入嵌入层的词嵌入

    def set_input_embeddings(self, value):
        self.embeddings.text_embeddings.word_embeddings = value
        # 设置输入嵌入层的词嵌入
    # 修剪模型的注意力头部，根据给定的字典指定每个层需要修剪的注意力头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要修剪的层和对应的注意力头部
        for layer, heads in heads_to_prune.items():
            # 获取指定层的注意力对象，然后修剪指定的注意力头部
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 重写父类 PreTrainedModel 的 forward 方法，并添加模型输入的文档字符串
    @add_start_docstrings_to_model_forward(VILT_INPUTS_DOCSTRING)
    # 替换输出的文档字符串，指定输出的类型和配置类
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        image_token_type_idx: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class ViltPooler(nn.Module):
    def __init__(self, config):
        # 初始化 ViltPooler 类
        super().__init__()
        # 创建一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个激活函数 Tanh()
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 在 forward 方法中，通过取第一个 token 对应的隐藏状态来"池化"模型
        first_token_tensor = hidden_states[:, 0]
        # 将第一个 token 的隐藏状态传入全连接层
        pooled_output = self.dense(first_token_tensor)
        # 将全连接层的输出传入激活函数
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出
        return pooled_output


@add_start_docstrings(
    """
    ViLT Model with a language modeling head on top as done during pretraining.
    """,
    VILT_START_DOCSTRING,
)
class ViltForMaskedLM(ViltPreTrainedModel):
    _tied_weights_keys = ["mlm_score.decoder.weight", "mlm_score.decoder.bias"]

    def __init__(self, config):
        # 初始化 ViltForMaskedLM 类
        super().__init__(config)

        # 创建 ViltModel 和 ViltMLMHead 实例
        self.vilt = ViltModel(config)
        self.mlm_score = ViltMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 返回 mlm_score.decoder 的权重
        return self.mlm_score.decoder

    def set_output_embeddings(self, new_embeddings):
        # 设置 mlm_score.decoder 的权重为新的 embeddings
        self.mlm_score.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(VILT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class ViltPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        # 初始化 ViltPredictionHeadTransform 类
        super().__init__()
        # 创建一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据配置中的激活函数类型选择对应的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 创建 LayerNorm 层，输入维度为 config.hidden_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # 将隐藏状态传入全连接层
        hidden_states = self.dense(hidden_states)
        # 经过激活函数处理
        hidden_states = self.transform_act_fn(hidden_states)
        # 经过 LayerNorm 处理
        hidden_states = self.LayerNorm(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


class ViltMLMHead(nn.Module):
    # 初始化函数，接受配置和权重参数
    def __init__(self, config, weight=None):
        # 调用父类的初始化函数
        super().__init__()
        # 保存配置参数
        self.config = config
        # 创建 ViltPredictionHeadTransform 实例
        self.transform = ViltPredictionHeadTransform(config)
        # 创建线性层，将隐藏层的输出映射到词汇表大小的向量
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 创建偏置参数
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 如果传入了权重参数，则使用传入的权重
        if weight is not None:
            self.decoder.weight = weight

        # 需要一个链接来确保偏置参数能够正确地随着 `resize_token_embeddings` 被调整大小
        self.decoder.bias = self.bias

    # 前向传播函数
    def forward(self, x):
        # 对输入进行变换
        x = self.transform(x)
        # 使用线性层进行映射
        x = self.decoder(x)
        # 返回结果
        return x
# 使用 Vilt 模型在视觉问答任务上添加一个分类器头部（在 [CLS] 标记的最终隐藏状态之上的线性层），例如用于 VQAv2
@add_start_docstrings(
    """
    Vilt Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the [CLS]
    token) for visual question answering, e.g. for VQAv2.
    """,
    VILT_START_DOCSTRING,
)
class ViltForQuestionAnswering(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 获取标签数量
        self.num_labels = config.num_labels
        # 初始化 Vilt 模型
        self.vilt = ViltModel(config)

        # 分类器头部
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.num_labels),
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(VILT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        r"""
        labels (`torch.FloatTensor` of shape `(batch_size, num_labels)`, *optional`):
            Labels for computing the visual question answering loss. This tensor must be either a one-hot encoding of
            all answers that are applicable for a given example in the batch, or a soft encoding indicating which
            answers are applicable, where 1.0 is the highest score.

        Returns:

        Examples:

        ```python
        >>> from transformers import ViltProcessor, ViltForQuestionAnswering
        >>> import requests
        >>> from PIL import Image

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "How many cats are there?"

        >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        >>> model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

        >>> # prepare inputs
        >>> encoding = processor(image, text, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**encoding)
        >>> logits = outputs.logits
        >>> idx = logits.argmax(-1).item()
        >>> print("Predicted answer:", model.config.id2label[idx])
        Predicted answer: 2
        ```py"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Vilt 模型进行前向传播
        outputs = self.vilt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取池化层输出
        pooler_output = outputs.pooler_output if return_dict else outputs[1]

        # 使用分类器对池化层输出进行分类
        logits = self.classifier(pooler_output)

        loss = None
        if labels is not None:
            # 将标签移动到正确的设备以启用后处理
            labels = labels.to(logits.device)
            # 计算二元交叉熵损失
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels) * labels.shape[1]
            # 参考链接：https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回分类器输出
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 Vilt 模型进行图像到文本或文本到图像的检索，例如 MSCOCO 和 F30K，顶部有一个分类器头（线性层在[CLS]标记的最终隐藏状态之上）
class ViltForImageAndTextRetrieval(ViltPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 Vilt 模型
        self.vilt = ViltModel(config)

        # 分类器头部
        self.rank_output = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels are currently not supported.

        Returns:

        Examples:

        ```python
        >>> from transformers import ViltProcessor, ViltForImageAndTextRetrieval
        >>> import requests
        >>> from PIL import Image

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

        >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
        >>> model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")

        >>> # forward pass
        >>> scores = dict()
        >>> for text in texts:
        ...     # prepare inputs
        ...     encoding = processor(image, text, return_tensors="pt")
        ...     outputs = model(**encoding)
        ...     scores[text] = outputs.logits[0, :].item()
        ```py"""
        # 设置返回字典为传入值或者使用配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 VILT 模型进行前向传播
        outputs = self.vilt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果 return_dict 为真，则使用 outputs.pooler_output，否则使用 outputs[1]
        pooler_output = outputs.pooler_output if return_dict else outputs[1]

        # 使用 rank_output 方法对 pooler_output 进行处理得到 logits
        logits = self.rank_output(pooler_output)

        loss = None
        if labels is not None:
            # 将标签移动到正确的设备以启用 PP
            labels = labels.to(logits.device)
            raise NotImplementedError("Training is not yet supported.")

        # 如果 return_dict 为假，则返回 logits 和 outputs[2:]，否则返回 loss 和 output
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 SequenceClassifierOutput 对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 Vilt 模型进行自然语言视觉推理的分类器头部，例如 NLVR2
@add_start_docstrings(
    """
    Vilt Model transformer with a classifier head on top for natural language visual reasoning, e.g. NLVR2.
    """,
    VILT_IMAGES_AND_TEXT_CLASSIFICATION_INPUTS_DOCSTRING,
)
class ViltForImagesAndTextClassification(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vilt = ViltModel(config)

        # 分类器头部
        num_images = config.num_images
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * num_images, config.hidden_size * num_images),
            nn.LayerNorm(config.hidden_size * num_images),
            nn.GELU(),
            nn.Linear(config.hidden_size * num_images, config.num_labels),
        )

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(VILT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ViltForImagesAndTextClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

# 使用 ViLT 模型进行文本标记头部的标记分类，例如用于命名实体识别（NER）任务
@add_start_docstrings(
    """
    ViLT Model with a token classification head on top (a linear layer on top of the final hidden-states of the text
    tokens) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    VILT_START_DOCSTRING,
)
class ViltForTokenClassification(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vilt = ViltModel(config, add_pooling_layer=False)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(VILT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个前向传播函数，接受多个输入参数并返回预测结果
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token ID
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码
        token_type_ids: Optional[torch.LongTensor] = None,  # token 类型 ID
        pixel_values: Optional[torch.FloatTensor] = None,  # 图像像素值
        pixel_mask: Optional[torch.LongTensor] = None,  # 图像掩码
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入
        image_embeds: Optional[torch.FloatTensor] = None,  # 图像嵌入
        labels: Optional[torch.LongTensor] = None,  # 标签
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果
    ) -> Union[TokenClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:
        """

        # 如果 return_dict 为 None，则使用配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 VILT 模型进行前向传播
        outputs = self.vilt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 获取文本输入大小
        text_input_size = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 对序列输出进行 dropout
        sequence_output = self.dropout(sequence_output)
        # 获取分类器的 logits
        logits = self.classifier(sequence_output[:, :text_input_size])

        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 将标签移动到正确的设备以启用后处理
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不返回字典形式的结果，则返回元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```