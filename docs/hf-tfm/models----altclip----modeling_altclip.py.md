# `.\transformers\models\altclip\modeling_altclip.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，对该文件的使用受限
# 只有在遵守许可证的情况下才能使用该文件
# 可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" PyTorch AltCLIP 模型。"""
# 导入所需的库
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

# 导入其他模块
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
    BaseModelOutputWithPoolingAndProjection,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_altclip import AltCLIPConfig, AltCLIPTextConfig, AltCLIPVisionConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "BAAI/AltCLIP"
_CONFIG_FOR_DOC = "AltCLIPConfig"

# 预训练模型存档列表
ALTCLIP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "BAAI/AltCLIP",
    # 查看所有 AltCLIP 模型 https://huggingface.co/models?filter=altclip
]

# AltCLIP 模型的起始文档字符串
ALTCLIP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`CLIPConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# AltCLIP 文本输入的文档字符串
ALTCLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。默认情况下，将忽略填充。
            # 可以使用 AutoTokenizer 获取索引。参见 PreTrainedTokenizer.encode 和 PreTrainedTokenizer.__call__。
            # 什么是输入 ID？
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。掩码值选在 [0, 1] 范围内：
            # - 1 表示**未被掩码**的标记，
            # - 0 表示**被掩码**的标记。
            # 什么是注意力掩码？
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择范围为 [0, config.max_position_embeddings - 1]。
            # 什么是位置 ID？
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多细节，请参见返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多细节，请参见返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回一个 `~utils.ModelOutput` 而不是一个普通元组。
"""

ALTCLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。默认情况下，如果提供了填充值，将被忽略。像素值可以使用 [`AutoImageProcessor`] 获得。有关详细信息，请参见 [`CLIPImageProcessor.__call__`]。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关详细信息，请参见返回的张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关详细信息，请参见返回的张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

ALTCLIP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。默认情况下，如果提供了填充值，将被忽略。

            索引可以使用 [`AutoTokenizer`] 获得。有关详细信息，请参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。
            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            避免在填充标记索引上执行注意力的掩码。掩码值选择在 `[0, 1]`：

            - 对于**未被掩蔽**的标记，为 1，
            - 对于**被掩蔽**的标记，为 0。
            [什么是注意力掩码？](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            每个输入序列标记在位置嵌入中的位置索引。选择在范围 `[0, config.max_position_embeddings - 1]` 内。

            [什么是位置 ID？](../glossary#position-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。默认情况下，如果提供了填充值，将被忽略。像素值可以使用 [`AutoImageProcessor`] 获得。有关详细信息，请参见 [`CLIPImageProcessor.__call__`]。
        return_loss (`bool`, *optional*):
            是否返回对比损失。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关详细信息，请参见返回的张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关详细信息，请参见返回的张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
# 对比损失函数，从给定链接中适配而来
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    # 使用交叉熵损失计算对比损失
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    # 计算文本和图像之间的对比损失
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    # 返回文本和图像对比损失的平均值
    return (caption_loss + image_loss) / 2.0


@dataclass
# 从transformers.models.clip.modeling_clip.CLIPOutput复制并更改为AltCLIPOutput
class AltCLIPOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            图像-文本相似性的对比损失。
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            `image_embeds`和`text_embeds`之间的缩放点积分数。这代表了图像-文本的相似性分数。
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            `text_embeds`和`image_embeds`之间的缩放点积分数。这代表了文本-图像的相似性分数。
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            通过将[`AltCLIPTextModel`]的池化输出应用于投影层获得的文本嵌入。
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            通过将[`AltCLIPVisionModel`]的池化输出应用于投影层获得的图像嵌入。
        text_model_output(`BaseModelOutputWithPooling`):
            [`AltCLIPTextModel`]的输出。
        vision_model_output(`BaseModelOutputWithPooling`):
            [`AltCLIPVisionModel`]的输出。
    """

    loss: Optional[torch.FloatTensor] = None
    logits_per_image: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# 从transformers.models.roberta.modeling_roberta.RobertaEmbeddings复制并更改为AltRobertaEmbeddings
class AltRobertaEmbeddings(nn.Module):
    """
    与BertEmbeddings相同，但对于位���嵌入的索引有微小调整。
    """

    # 从transformers.models.bert.modeling_bert.BertEmbeddings.__init__复制
    # 初始化函数，用于创建一个新的实例
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建词嵌入层，将词汇映射到隐藏层，使用了config中的参数进行设置
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，将位置信息映射到隐藏层，使用了config中的参数进行设置
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建类型嵌入层，将类型信息映射到隐藏层，使用了config中的参数进行设置
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 将LayerNorm命名为LayerNorm，以便与TensorFlow模型变量名保持一致，并能够加载任何TensorFlow检查点文件
        # 创建LayerNorm层，对隐藏层进行归一化，使用了config中的参数进行设置
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建Dropout层，对隐藏层进行随机失活，使用了config中的参数进行设置
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 位置嵌入类型，默认为"absolute"
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册缓冲区，包含位置ID信息
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册缓冲区，包含token类型ID信息，默认为全零
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        # 填充ID，用于在输入序列中标识填充的token
        self.padding_idx = config.pad_token_id
        # 创建新的位置嵌入层，将位置信息映射到隐藏层，使用了config中的参数进行设置
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    # 前向传播函数，用于定义模型的前向计算逻辑
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
        ):
        # 如果位置 id 为 None
        if position_ids is None:
            # 如果输入 id 不为 None
            if input_ids is not None:
                # 从输入的标记 id 创建位置 id。任何填充的标记仍然保持填充状态。
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                # 从输入的嵌入创建位置 id
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果输入 id 不为 None
        if input_ids is not None:
            # 获取输入 id 的形状
            input_shape = input_ids.size()
        else:
            # 获取输入嵌入的形状，去掉最后一个维度
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 将 token_type_ids 设置为构造函数中注册的缓冲区，其中所有值都为零，通常在自动生成时发生，
        # 注册的缓冲区有助于用户在不传递 token_type_ids 的情况下跟踪模型，解决问题 #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果输入嵌入为 None
        if inputs_embeds is None:
            # 使用 word_embeddings 获取输入 id 的嵌入
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取 token_type_ids 的嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入和 token_type_ids 的嵌入相加
        embeddings = inputs_embeds + token_type_embeddings
        # 如果位置嵌入类型为 "absolute"
        if self.position_embedding_type == "absolute":
            # 获取位置嵌入
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 对 embeddings 进行 LayerNorm
        embeddings = self.LayerNorm(embeddings)
        # 对 embeddings 进行 dropout
        embeddings = self.dropout(embeddings)
        # 返回 embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入嵌入的形状，去掉最后一个维度
        input_shape = inputs_embeds.size()[:-1]
        # 获取序列长度
        sequence_length = input_shape[1]

        # 生成顺序位置 id
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
# 从transformers.models.roberta.modeling_roberta.RobertaSelfAttention复制代码，并将Roberta->AltRoberta
class AltRobertaSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏大小是否是注意力头数的倍数，如果不是则引发错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对键或相对键查询，则初始化距离嵌入
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    # 将输入张量转换为分数张量
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
# 从transformers.models.roberta.modeling_roberta.RobertaSelfOutput复制代码
class AltRobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化全连接层、LayerNorm和dropout层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
# 从 transformers.models.roberta.modeling_roberta.RobertaAttention 复制而来，将 Roberta 替换为 AltRoberta
class AltRobertaAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 创建 AltRobertaSelfAttention 对象
        self.self = AltRobertaSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建 AltRobertaSelfOutput 对象
        self.output = AltRobertaSelfOutput(config)
        # 存储被修剪的注意力头的索引
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到要修剪的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪的注意力头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 调用 AltRobertaSelfAttention 对象进行前向传播
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 调用 AltRobertaSelfOutput 对象进行前向传播，得到注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果有输出注意力，将其添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# 从 transformers.models.roberta.modeling_roberta.RobertaIntermediate 复制而来，将 Roberta 替换为 AltRoberta
class AltRobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建线性层，用于中间层变换
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 判断配置中隐藏层激活函数是否是字符串，若是，则获取相应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过线性层进行变换
        hidden_states = self.dense(hidden_states)
        # 应用中间层激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.roberta.modeling_roberta.RobertaOutput 复制而来
class AltRobertaOutput(nn.Module):
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入大小为config.intermediate_size，输出大小为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，输入大小为config.hidden_size，eps为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # 前向传播函数，接受两个张量参数，返回一个张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将hidden_states传入全连接层，得到输出
        hidden_states = self.dense(hidden_states)
        # 对输出进行Dropout操作
        hidden_states = self.dropout(hidden_states)
        # 将Dropout后的输出与input_tensor相加，然后传入LayerNorm层
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回LayerNorm层的输出
        return hidden_states
# 从transformers.models.roberta.modeling_roberta.RobertaLayer复制而来，将Roberta替换为AltRoberta
class AltRobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义前向传播时用到的参数
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度
        self.seq_len_dim = 1
        # 定义注意力机制
        self.attention = AltRobertaAttention(config)
        # 是否为解码器
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了交叉注意力，必须是解码器模型
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 交叉注意力机制
            self.crossattention = AltRobertaAttention(config, position_embedding_type="absolute")
        # 中间层
        self.intermediate = AltRobertaIntermediate(config)
        # 输出层
        self.output = AltRobertaOutput(config)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 如果有过去的键/值缓存，则解码器的自注意力缓存在位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 执行自注意力机制
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力机制的输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果是解码器并且提供了编码器的隐藏状态，则需要进行交叉注意力
            if not hasattr(self, "crossattention"):
                # 如果提供了`encoder_hidden_states`，则必须通过设置`config.add_cross_attention=True`来实例化具有交叉注意力层的`self`
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 交叉注意力的缓存键/值元组位于past_key_value元组的位置3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 执行交叉注意力机制
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力机制的输出
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加交叉注意力

            # 将交叉注意力缓存添加到present_key_value元组的位置3,4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 应用前向分块到前向网络
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将前向网络的输出添加到outputs元组中
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 前向分块函数，包括了中间层和输出层的计算
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从 transformers.models.roberta.modeling_roberta.RobertaEncoder 复制而来，修改为 AltRoberta
class AltRobertaEncoder(nn.Module):
    def __init__(self, config):
        # 调用父类构造函数
        super().__init__()
        # 保存配置信息
        self.config = config
        # 创建层列表，包含 config.num_hidden_layers 个 AltRobertaLayer 层对象
        self.layer = nn.ModuleList([AltRobertaLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点默认关闭
        self.gradient_checkpointing = False

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    # 定义函数，接受输入参数并返回输出结果，输出结果为 torch.Tensor 或包含额外属性的 BaseModelOutputWithPastAndCrossAttentions 类型的元组
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果设置了输出隐藏状态，则初始化空元组，否则为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果设置了输出注意力权重，则初始化空元组，否则为 None
        all_self_attentions = () if output_attentions else None
        # 如果设置了输出交叉注意力权重并且配置中添加了交叉注意力，则初始化空元组，否则为 None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用了梯度检查点并且处于训练模式下，则执行以下代码
        if self.gradient_checkpointing and self.training:
            # 如果设置了使用缓存，则发出警告，并将 use_cache 设置为 False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果设置了使用缓存，则初始化空元组，否则为 None
        next_decoder_cache = () if use_cache else None
        # 遍历每个解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果设置了输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果设置了头部遮罩，则获取当前层的头部遮罩，否则为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果传入了过去的键值对，则获取当前层的过去键值对，否则为 None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点并且处于训练模式下，则使用梯度检查点函数执行当前层的前向传播
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                # 否则直接执行当前层的前向传播
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新当前隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的缓存添加到 next_decoder_cache 中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果设置了输出注意力权重，则将当前层的注意力权重添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果配置中添加了交叉注意力，则将当前层的交叉注意力权重添加到 all_cross_attentions 中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果设置了输出隐藏状态，则将最后一层的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典格式的结果，则返回一个元组，包含 hidden_states、next_decoder_cache、all_hidden_states、all_self_attentions、all_cross_attentions 中非 None 的部分
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # 否则，返回一个 BaseModelOutputWithPastAndCrossAttentions 类型的对象，包含最终的隐藏状态、过去键值对、所有隐藏状态、所有自注意力权重、所有交叉注意力权重
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从transformers.models.roberta.modeling_roberta.RobertaPooler复制过来的自定义RobertaPooler类
class AltRobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数使用双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过取第一个标记对应的隐藏状态来“汇聚”模型
        first_token_tensor = hidden_states[:, 0]
        # 将第一个标记对应的隐藏状态传入全连接层
        pooled_output = self.dense(first_token_tensor)
        # 使用激活函数激活全连接层的输出
        pooled_output = self.activation(pooled_output)
        # 返回汇聚后的输出
        return pooled_output


# 从transformers.models.clip.modeling_clip.CLIPAttention复制过来的自定义AltCLIPAttention类
class AltCLIPAttention(nn.Module):
    """来自“Attention Is All You Need”论文的多头注意力"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        # 检查是否可以将embed_dim均分成num_heads份
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim必须可以被num_heads整除（得到`embed_dim`：{self.embed_dim}和`num_heads`："
                f" {self.num_heads}）。"
            )
        # 缩放因子，用于缩放注意力分数
        self.scale = self.head_dim**-0.5
        # 注意力的dropout比例
        self.dropout = config.attention_dropout

        # 分别定义Q、K、V、输出的线性变换层
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 调整张量形状以适应多头注意力的需求
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
):
# 从transformers.models.clip.modeling_clip.CLIPMLP复制过来的自定义AltCLIPMLP类
class AltCLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 激活函数由配置文件中指定
        self.activation_fn = ACT2FN[config.hidden_act]
        # 第一个全连接层，输入维度为config.hidden_size，输出维度为config.intermediate_size
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 第二个全连接层，输入和输出维度都为config.hidden_size
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过第一个全连接层
        hidden_states = self.fc1(hidden_states)
        # 使用激活函数激活输出
        hidden_states = self.activation_fn(hidden_states)
        # 通过第二个全连接层
        hidden_states = self.fc2(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 从transformers.models.clip.modeling_clip.CLIPEncoderLayer复制过来的自定义AltCLIPEncoderLayer类
class AltCLIPEncoderLayer(nn.Module):
    # 初始化函数，接受一个 AltCLIPConfig 类型的参数 config
    def __init__(self, config: AltCLIPConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置嵌入维度为 config 中的 hidden_size
        self.embed_dim = config.hidden_size
        # 创建 AltCLIPAttention 对象并赋值给 self.self_attn
        self.self_attn = AltCLIPAttention(config)
        # 创建 LayerNorm 对象并赋值给 self.layer_norm1，设置归一化维度和 epsilon
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 创建 AltCLIPMLP 对象并赋值给 self.mlp
        self.mlp = AltCLIPMLP(config)
        # 创建 LayerNorm 对象并赋值给 self.layer_norm2，设置归一化维度和 epsilon

    # 前向传播函数，接受输入 hidden_states、attention_mask、causal_attention_mask 和 output_attentions 参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # 保存输入 hidden_states 作为残差连接的基准
        residual = hidden_states

        # 对输入 hidden_states 进行 LayerNorm 归一化
        hidden_states = self.layer_norm1(hidden_states)
        # 调用 self.self_attn 进行自注意力计算
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        # 将残差连接和自注意力计算结果相加
        hidden_states = residual + hidden_states

        # 保存当前状态作为下一层的残差连接基准
        residual = hidden_states
        # 对当前状态进行 LayerNorm 归一化
        hidden_states = self.layer_norm2(hidden_states)
        # 调用 self.mlp 进行多层��知机计算
        hidden_states = self.mlp(hidden_states)
        # 将残差连接和多层感知机计算结果相加
        hidden_states = residual + hidden_states

        # 将最终结果保存在 outputs 中
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重添加到 outputs 中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回 outputs
        return outputs
# 从 transformers.models.clip.modeling_clip.CLIPEncoder 复制而来，将 CLIP->AltCLIP
class AltCLIPEncoder(nn.Module):
    """
    由 `config.num_hidden_layers` 个自注意力层组成的 Transformer 编码器。每一层都是一个 [`AltCLIPEncoderLayer`]。

    Args:
        config: AltCLIPConfig
    """

    def __init__(self, config: AltCLIPConfig):
        super().__init__()
        self.config = config
        # 创建一个包含 `config.num_hidden_layers` 个 AltCLIPEncoderLayer 的模块列表
        self.layers = nn.ModuleList([AltCLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否启用梯度检查点
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

# 从 transformers.models.clip.modeling_clip.CLIPVisionEmbeddings 复制而来，将 CLIP->AltCLIP
class AltCLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: AltCLIPVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # 类别嵌入
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        # 补丁嵌入
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        # 计算嵌入位置
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        # 对像素值进行补丁嵌入
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 类别嵌入扩展到每个样本
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        # 拼接类别嵌入和补丁嵌入
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # 加上位置嵌入
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class AltCLIPPreTrainedModel(PreTrainedModel):
    """
    一个处理权重初始化以及下载和加载预训练模型的简单接口的抽象类。
    """

    config_class = AltCLIPConfig
    base_model_prefix = "altclip"
    supports_gradient_checkpointing = True
```  
    def _init_weights(self, module):
        """初始化模型参数的权重"""
        # 获取初始化因子
        factor = self.config.initializer_factor
        # 如果模块是 AltCLIPVisionEmbeddings 类型
        if isinstance(module, AltCLIPVisionEmbeddings):
            # 重新获取初始化因子
            factor = self.config.initializer_factor
            # 对类别嵌入进行标准正态分布初始化
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            # 对路径嵌入进行标准正态分布初始化
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            # 对位置嵌入进行标准正态分布初始化
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        # 如果模块是 AltCLIPAttention 类型
        elif isinstance(module, AltCLIPAttention):
            # 重新获取初始化因子
            factor = self.config.initializer_factor
            # 计算输入投影的标准差
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            # 计算输出投影的标准差
            out_proj_std = (module.embed_dim**-0.5) * factor
            # 对查询投影的权重进行标准正态分布初始化
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            # 对键投影的权重进行标准正态分布初始化
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            # 对值投影的权重进行标准正态分布初始化
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            # 对输出投影的权重进行标准正态分布初始化
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        # 如果模块是 AltCLIPMLP 类型
        elif isinstance(module, AltCLIPMLP):
            # 重新获取初始化因子
            factor = self.config.initializer_factor
            # 计算输入投影的标准差
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            # 计算全连接层的标准差
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            # 对第一个全连接层的权重进行标准正态分布初始化
            nn.init.normal_(module.fc1.weight, std=fc_std)
            # 对第二个全连接层的权重进行标准正态分布初始化
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        # 如果模块是 AltCLIPModel 类型
        elif isinstance(module, AltCLIPModel):
            # 对文本投影的权重进行标准正态分布初始化
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            # 设置文本投影已经初始化的标志
            module.text_projection._is_hf_initialized = True
            # 对视觉投影的权重进行标准正态分布初始化
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )
            # 设置视觉投影已经初始化的标志
            module.visual_projection._is_hf_initialized = True
        # 如果模块是 nn.LayerNorm 类型
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
        # 如果模块是 nn.Linear 类型
        elif isinstance(module, nn.Linear):
            # 对权重进行标准正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_factor)
            # 如果存在偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 对权重进行标准正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_factor)
            # 如果存在填充索引，将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
# 从transformers.models.clip.modeling_clip.CLIPVisionTransformer复制代码，并进行修改以创建AltCLIPVisionTransformer
class AltCLIPVisionTransformer(nn.Module):
    def __init__(self, config: AltCLIPVisionConfig):
        super().__init__()
        # 设置配置属性
        self.config = config
        # 提取配置中的隐藏尺寸作为嵌入维度
        embed_dim = config.hidden_size

        # 实例化嵌入层对象
        self.embeddings = AltCLIPVisionEmbeddings(config)
        # 添加预层标准化层
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        # 实例化编码器对象
        self.encoder = AltCLIPEncoder(config)
        # 添加后层标准化层
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    # 添加前向传播方法的说明文档，使用装饰器
    @add_start_docstrings_to_model_forward(ALTCLIP_VISION_INPUTS_DOCSTRING)
    # 替换返回文档的说明文档，使用装饰器
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=AltCLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        返回:
        """

        # 如果未提供像素值，则引发错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值传递给嵌入层进行处理
        hidden_states = self.embeddings(pixel_values)
        # 对嵌入层输出进行预层标准化
        hidden_states = self.pre_layrnorm(hidden_states)

        # 使用编码器处理嵌入层输出
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取编码器输出中的最后隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 从最后隐藏状态中提取池化输出
        pooled_output = last_hidden_state[:, 0, :]
        # 对池化输出进行后层标准化
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不返回字典，则返回元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 返回字典形式的模型输出
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class AltCLIPVisionModel(AltCLIPPreTrainedModel):
    # 设置配置类
    config_class = AltCLIPVisionConfig
    # 设置主要输入名称为"pixel_values"
    main_input_name = "pixel_values"

    def __init__(self, config: AltCLIPVisionConfig):
        super().__init__(config)
        # 创建视觉模型对象
        self.vision_model = AltCLIPVisionTransformer(config)
        # 初始化权重并应用最终处理
        self.post_init()
    # 获取输入的嵌入模块，这里是视觉模型的嵌入模块
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    # 将额外的文档字符串添加到模型前向方法中
    @add_start_docstrings_to_model_forward(ALTCLIP_VISION_INPUTS_DOCSTRING)
    # 替换前向方法的返回文档字符串
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=AltCLIPVisionConfig)
    # 前向传播方法
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,  # 输入的像素值，默认为空
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，默认为空
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为空
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为空
    ) -> Union[Tuple, BaseModelOutputWithPooling]:  # 返回值为元组或带池化的基本模型输出
        r"""
        Returns:  # 返回模型输出

        Examples:  # 示例代码

        ```py
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AltCLIPVisionModel

        >>> model = AltCLIPVisionModel.from_pretrained("BAAI/AltCLIP")
        >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        # 如果 return_dict 为 None，则使用模型配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用视觉模型的前向方法
        return self.vision_model(
            pixel_values=pixel_values,  # 像素值
            output_attentions=output_attentions,  # 输出注意力权重
            output_hidden_states=output_hidden_states,  # 输出隐藏状态
            return_dict=return_dict,  # 返回字典
        )
```py  
class AltRobertaModel(AltCLIPPreTrainedModel):
    """
    
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    config_class = AltCLIPTextConfig

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->AltRoberta
    def __init__(self, config, add_pooling_layer=True):
        # 调用父类构造函数
        super().__init__(config)
        # 设置当前类的配置
        self.config = config

        # 初始化嵌入层
        self.embeddings = AltRobertaEmbeddings(config)
        # 初始化编码器
        self.encoder = AltRobertaEncoder(config)

        # 如果需要添加池化层，则初始化池化层
        self.pooler = AltRobertaPooler(config) if add_pooling_layer else None

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # 调用编码器中每一层的注意力头剪枝函数
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    # 前向传播函数，接受各种输入和参数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class AltCLIPTextModel(AltCLIPPreTrainedModel):
    config_class = AltCLIPTextConfig
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        # 初始化一个 AltRobertaModel 模型，关闭额外的池化层
        self.roberta = AltRobertaModel(config, add_pooling_layer=False)
        # 初始化一个线性变换层，将隐藏层的尺寸转换为配置对象中指定的项目维度
        self.transformation = nn.Linear(config.hidden_size, config.project_dim)
        # 初始化一个预层归一化层，使用配置对象中指定的 epsilon 值
        self.pre_LN = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 调用后期初始化方法
        self.post_init()

    # 获取输入嵌入的方法，返回 Roberta 模型的词嵌入层
    def get_input_embeddings(self) -> nn.Module:
        return self.roberta.embeddings.word_embeddings

    # 设置输入嵌入的方法，接受一个嵌入层对象作为参数，并将其设置为 Roberta 模型的词嵌入层
    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.roberta.embeddings.word_embeddings = value

    # 调整标记嵌入的方法，接受一个新的标记数量作为可选参数，并返回调整后的标记嵌入层对象
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        return super().resize_token_embeddings(new_num_tokens)

    # 前向传播方法，接受一系列输入，并根据模型进行计算，返回输出结果
    @add_start_docstrings_to_model_forward(ALTCLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndProjection, config_class=AltCLIPTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndProjection]:
        r"""
        Returns:

        Examples:

        """


        # 如果返回字典未指定，使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 RoBERTa 模型进行前向传播
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取 RoBERTa 模型输出的序列表示
        sequence_output = outputs[0]

        # 将序列表示传入预处理层进行处理
        sequence_output = self.pre_LN(sequence_output)

        # 使用转换层对序列表示进行投影
        projection_state = self.transformation(sequence_output)
        
        # 从投影状态中获取池化输出，只使用第一个位置的表示
        pooler_output = projection_state[:, 0]

        # 如果不需要返回字典，则返回投影状态、池化输出以及额外的隐藏状态和注意力
        if not return_dict:
            return (projection_state, pooler_output) + outputs[2:4]

        # 如果需要返回字典，则构建输出对象并返回
        return BaseModelOutputWithPoolingAndProjection(
            last_hidden_state=projection_state,
            pooler_output=pooler_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class AltCLIPModel(AltCLIPPreTrainedModel):
    # 设置配置类为AltCLIPConfig
    config_class = AltCLIPConfig

    def __init__(self, config: AltCLIPConfig):
        # 调用父类的构造函数
        super().__init__(config)

        # 检查config.vision_config是否为AltCLIPVisionConfig类型，如果不是则抛出数值错误
        if not isinstance(config.vision_config, AltCLIPVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type AltCLIPVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )
        # 检查config.text_config是否为AltCLIPTextConfig类型，如果不是则抛出数值错误
        if not isinstance(config.text_config, AltCLIPTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type AltCLIPTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 获取text_config和vision_config
        text_config = config.text_config
        vision_config = config.vision_config

        # 设置投影维度、文本嵌入维度和视觉嵌入维度
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.project_dim
        self.vision_embed_dim = vision_config.hidden_size

        # 初始化文本模型和视觉模型
        self.text_model = AltCLIPTextModel(text_config)
        self.vision_model = AltCLIPVisionTransformer(vision_config)

        # 创建视觉投影和文本投影层
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ALTCLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`AltCLIPTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AltCLIPModel

        >>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")
        >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")
        >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```py"""
        # 使用 AltCLIP 模型的配置来覆盖一些字段（如果已指定），而不是使用视觉和文本组件的配置。
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定输出隐藏状态，则使用配置中的默认值。
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定返回字典，则使用配置中的默认值。
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取文本输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 从文本输出中获取池化输出
        pooled_output = text_outputs[1]
        # 通过文本投影层对池化输出进行投影，得到文本特征
        text_features = self.text_projection(pooled_output)

        # 返回文本特征
        return text_features

    @add_start_docstrings_to_model_forward(ALTCLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`AltCLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image  # 导入图像处理库中的 Image 模块
        >>> import requests  # 导入用于发送 HTTP 请求的 requests 模块
        >>> from transformers import AutoProcessor, AltCLIPModel  # 导入 Hugging Face 的 transformers 库中的 AutoProcessor 和 AltCLIPModel 类

        >>> model = AltCLIPModel.from_pretrained("BAAI/AltCLIP")  # 从预训练模型加载 AltCLIP 模型
        >>> processor = AutoProcessor.from_pretrained("BAAI/AltCLIP")  # 从预训练模型加载 AltCLIP 的 processor
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 定义图像 URL
        >>> image = Image.open(requests.get(url, stream=True).raw)  # 使用 requests 库下载图像，并用 PIL 库打开
        >>> inputs = processor(images=image, return_tensors="pt")  # 使用 processor 处理图像，并返回 PyTorch 张量
        >>> image_features = model.get_image_features(**inputs)  # 获取图像特征
        ```py"""
        # Use AltCLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  # 使用 AltCLIP 模型的配置而不是视觉和文本组件的配置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )  # 使用 AltCLIP 模型的配置而不是视觉和文本组件的配置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 使用 AltCLIP 模型的配置而不是视觉和文本组件的配置

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,  # 图像像素值
            output_attentions=output_attentions,  # 是否返回注意力权重
            output_hidden_states=output_hidden_states,  # 是否返回隐藏状态
            return_dict=return_dict,  # 是否返回字典
        )

        pooled_output = vision_outputs[1]  # 获取池化后的输出
        image_features = self.visual_projection(pooled_output)  # 将池化后的输出投影到特征空间

        return image_features  # 返回图像特征

    @add_start_docstrings_to_model_forward(ALTCLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=AltCLIPOutput, config_class=AltCLIPConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入 token 的 ID
        pixel_values: Optional[torch.FloatTensor] = None,  # 图像像素值
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩
        position_ids: Optional[torch.LongTensor] = None,  # 位置 ID
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 ID
        return_loss: Optional[bool] = None,  # 是否返回损失值
        output_attentions: Optional[bool] = None,  # 是否返回注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否返回隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
# 从输入的 input_ids 中创建位置编号，替换非填充符号为它们的位置数字。位置数字从 padding_idx+1 开始计数。填充符号将被忽略。
# 这个函数是从 fairseq 的 `utils.make_positions` 修改而来。

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    # 创建一个 mask，将 input_ids 中非填充符号转换为 1，填充符号转换为 0
    mask = input_ids.ne(padding_idx).int()
    # 计算每个位置的累积非填充符号数量，加上过去键值长度 past_key_values_length，并将结果转换为与 mask 相同的数据类型
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 将 incremental_indices 转换为长整型，并加上 padding_idx 得到最终的位置编号
    return incremental_indices.long() + padding_idx
```