# `.\models\layoutlm\modeling_layoutlm.py`

```py
# coding=utf-8
# 版权归 Microsoft Research Asia LayoutLM Team 作者和 HuggingFace Inc. 团队所有。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）授权;
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发的软件，
# 没有任何明示或暗示的保证或条件。
# 请查阅许可证获取具体的法律授权和限制。
""" PyTorch LayoutLM 模型。"""


import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_layoutlm import LayoutLMConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LayoutLMConfig"
_CHECKPOINT_FOR_DOC = "microsoft/layoutlm-base-uncased"

LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "layoutlm-base-uncased",
    "layoutlm-large-uncased",
]


LayoutLMLayerNorm = nn.LayerNorm


class LayoutLMEmbeddings(nn.Module):
    """从词嵌入、位置嵌入和标记类型嵌入构建嵌入。"""

    def __init__(self, config):
        super(LayoutLMEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = LayoutLMLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
    # 定义前向传播方法，接收多个输入参数
    def forward(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        # 如果传入了 input_ids 参数，获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则，获取 inputs_embeds 的形状，排除最后一维
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 确定设备，如果有 input_ids 使用 input_ids 的设备，否则使用 inputs_embeds 的设备
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果 position_ids 为 None，则使用 self.position_ids 的前 seq_length 列
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果 token_type_ids 为 None，则初始化为全零张量
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 如果 inputs_embeds 为 None，则使用 self.word_embeddings 对 input_ids 进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 获取单词嵌入向量
        words_embeddings = inputs_embeds
        # 获取位置嵌入向量
        position_embeddings = self.position_embeddings(position_ids)

        # 尝试获取左上右下四个方向的位置嵌入向量，并处理 IndexError 异常
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            # 如果 IndexError 发生，抛出异常并提供更具体的错误信息
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        # 获取高度和宽度的位置嵌入向量
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        # 获取 token_type 的嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 计算最终的嵌入向量，包括单词、位置、各方向位置、高度、宽度和 token_type 的嵌入向量
        embeddings = (
            words_embeddings
            + position_embeddings
            + left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
            + token_type_embeddings
        )

        # 对嵌入向量进行 LayerNorm
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入向量应用 dropout
        embeddings = self.dropout(embeddings)

        # 返回最终的嵌入向量
        return embeddings
# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->LayoutLM
class LayoutLMSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏大小是否能被注意力头数整除，若不能则引发值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 设置位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对键或相对键-查询，则初始化距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 判断是否是解码器模式
        self.is_decoder = config.is_decoder

    # 将张量变换为注意力分数的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数定义
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        # 此处省略部分前向传播的具体实现
        pass


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->LayoutLM
class LayoutLMSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化全连接层、LayerNorm和dropout
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数定义
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层变换
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # LayerNorm层和残差连接
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->LayoutLM
# LayoutLMAttention 类，用于 LayoutLM 模型中的注意力机制部分
class LayoutLMAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化自注意力层和输出层
        self.self = LayoutLMSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = LayoutLMSelfOutput(config)
        self.pruned_heads = set()

    # 剪枝注意力头部
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可剪枝的注意力头部和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝后的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

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
        # 执行自注意力层的前向传播
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将自注意力层的输出经过输出层
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # 如果有需要，添加注意力权重
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
# LayoutLMIntermediate 类，用于 LayoutLM 模型中的中间层
class LayoutLMIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化线性层和激活函数
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # 应用中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->LayoutLM
# LayoutLMOutput 类，用于 LayoutLM 模型中的输出层
class LayoutLMOutput(nn.Module):
    # 初始化函数，用于创建一个新的神经网络层
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个线性层，输入维度为 config.intermediate_size，输出维度为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，输入维度为 config.hidden_size，设置 epsilon 参数为 config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，用于在训练过程中随机丢弃部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，定义了神经网络的计算流程
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用线性层进行计算，将 hidden_states 映射到新的表示空间
        hidden_states = self.dense(hidden_states)
        # 对计算结果进行 dropout 操作，随机丢弃部分神经元的输出
        hidden_states = self.dropout(hidden_states)
        # 将 dropout 后的结果与 input_tensor 相加，并进行 LayerNorm 操作，得到最终的隐藏状态表示
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回最终的隐藏状态表示作为本层的输出
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertLayer复制代码，将Bert替换为LayoutLM
class LayoutLMLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化LayoutLMLayer类，设置前馈块的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度设置为1
        self.seq_len_dim = 1
        # 创建LayoutLMAttention对象并赋给self.attention
        self.attention = LayoutLMAttention(config)
        # 检查是否为解码器模型
        self.is_decoder = config.is_decoder
        # 检查是否添加跨注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加跨注意力但不是解码器模型，则引发错误
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 使用绝对位置嵌入类型创建LayoutLMAttention对象并赋给self.crossattention
            self.crossattention = LayoutLMAttention(config, position_embedding_type="absolute")
        # 创建LayoutLMIntermediate对象并赋给self.intermediate
        self.intermediate = LayoutLMIntermediate(config)
        # 创建LayoutLMOutput对象并赋给self.output
        self.output = LayoutLMOutput(config)

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
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用当前的 self_attn_past_key_value 执行 self-attention 计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            # 从 self_attention_outputs 中提取除了最后一个元素之外的所有元素作为 outputs
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果不是 decoder，将 self_attention_outputs 中除了第一个元素之外的所有元素作为 outputs
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                # 如果没有设置 cross-attention 层，则抛出 ValueError
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            # 从 past_key_value 中提取出后两个元素作为 cross_attn_past_key_value
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            # 将 cross_attention_outputs 中除了最后一个元素之外的所有元素添加到 outputs 中
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            # 将 cross_attention_outputs 中的最后一个元素添加到 present_key_value 中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 对 attention_output 应用 chunking 策略来处理长序列
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        # 如果是 decoder，将 present_key_value 作为最后一个输出添加到 outputs 中
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 将 attention_output 应用 feed-forward 网络的中间层和输出层
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从transformers.models.bert.modeling_bert.BertEncoder复制并修改为LayoutLMEncoder类
class LayoutLMEncoder(nn.Module):
    # 初始化函数，接受一个配置参数config
    def __init__(self, config):
        super().__init__()
        # 将传入的配置参数保存到self.config中
        self.config = config
        # 创建一个由多个LayoutLMLayer对象组成的层列表，数量为config.num_hidden_layers
        self.layer = nn.ModuleList([LayoutLMLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点标志，初始设置为False
        self.gradient_checkpointing = False

    # 前向传播函数，接受多个输入参数
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
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 初始化空元组，根据参数设置是否输出隐藏状态、注意力权重和交叉注意力权重
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果开启了梯度检查点且在训练模式下，处理缓存使用情况
        if self.gradient_checkpointing and self.training:
            if use_cache:
                # 如果同时使用缓存和梯度检查点，给出警告并强制将 use_cache 设置为 False
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 初始化下一个解码器缓存的元组，根据 use_cache 参数决定是否为空
        next_decoder_cache = () if use_cache else None
        # 遍历所有的层模块
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态加入到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 根据 head_mask 参数决定当前层的注意力头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 根据 past_key_values 参数决定过去的键值对
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果开启了梯度检查点且在训练模式下，调用梯度检查点函数处理当前层的计算
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
                # 否则直接调用当前层模块进行前向传播计算
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新隐藏状态为当前层模块的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果使用缓存，将当前层的输出的最后一个元素加入到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，将当前层模块的输出的第二个元素加入到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果模型配置中添加了交叉注意力，将当前层模块的输出的第三个元素加入到 all_cross_attentions 中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 最后一层计算完成后，如果需要输出隐藏状态，将最终的隐藏状态加入到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据 return_dict 参数决定返回类型，返回相应的结果
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
        # 返回包含最终结果的 BaseModelOutputWithPastAndCrossAttentions 对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# Copied from transformers.models.bert.modeling_bert.BertPooler
class LayoutLMPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数使用双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 从隐藏状态中取出第一个 token 的隐藏状态作为池化输出
        first_token_tensor = hidden_states[:, 0]
        # 将第一个 token 的隐藏状态通过全连接层进行线性变换
        pooled_output = self.dense(first_token_tensor)
        # 将线性变换后的输出应用激活函数
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->LayoutLM
class LayoutLMPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据配置文件选择激活函数，如果是字符串，则使用预定义的激活函数；否则直接使用给定的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # Layer normalization，对隐藏状态进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 首先将隐藏状态通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 然后将线性变换后的结果应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 最后对处理后的隐藏状态进行 Layer normalization
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->LayoutLM
class LayoutLMLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个预测头变换层，用于将隐藏状态映射到预测值
        self.transform = LayoutLMPredictionHeadTransform(config)

        # 输出权重与输入嵌入层相同，但每个标记都有一个仅输出的偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 输出偏置，用于每个标记的预测
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要连接两个变量，以便在 `resize_token_embeddings` 时正确调整偏置大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 首先通过变换层处理隐藏状态
        hidden_states = self.transform(hidden_states)
        # 然后通过线性层进行预测
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->LayoutLM
class LayoutLMOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化预测层，用于MLM任务的预测
        self.predictions = LayoutLMLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 对序列输出进行预测得分计算
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class LayoutLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定使用的配置类为LayoutLMConfig
    config_class = LayoutLMConfig
    # 使用预训练模型的存档映射列表作为初始值
    pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST
    # 定义基础模型的前缀字符串
    base_model_prefix = "layoutlm"
    # 模型支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化模型权重"""
        # 如果模块是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，标准差为配置文件中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，标准差为配置文件中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，将其对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是自定义的 LayoutLMLayerNorm 层
        elif isinstance(module, LayoutLMLayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全 1
            module.weight.data.fill_(1.0)
"""
LAYOUTLM_START_DOCSTRING = r"""
    The LayoutLM model was proposed in [LayoutLM: Pre-training of Text and Layout for Document Image
    Understanding](https://arxiv.org/abs/1912.13318) by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and
    Ming Zhou.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LayoutLMConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LAYOUTLM_INPUTS_DOCSTRING = r"""
    Args:
        batch_size (int): The batch size of the input data.
        sequence_length (int): The length of the input sequences.

    This method returns the LayoutLM Model's outputs with the specified input parameters.
"""

@add_start_docstrings(
    "The bare LayoutLM Model transformer outputting raw hidden-states without any specific head on top.",
    LAYOUTLM_START_DOCSTRING,
)
class LayoutLMModel(LayoutLMPreTrainedModel):
    def __init__(self, config):
        super(LayoutLMModel, self).__init__(config)
        self.config = config

        self.embeddings = LayoutLMEmbeddings(config)
        self.encoder = LayoutLMEncoder(config)
        self.pooler = LayoutLMPooler(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns:
            torch.nn.Embedding: The word embedding layer of the LayoutLM Model.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Args:
            value (torch.Tensor): The new input embeddings to be set for the LayoutLM Model.
        """
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model.

        Args:
            heads_to_prune (dict): Dictionary of {layer_num: list of heads to prune in this layer}.
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutputWithPoolingAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Args:
            input_ids (torch.LongTensor, optional): The input IDs of the tokens.
            bbox (torch.LongTensor, optional): The bounding boxes of each token in the input.
            attention_mask (torch.FloatTensor, optional): The attention mask for the input.
            token_type_ids (torch.LongTensor, optional): The token type IDs for the input.
            position_ids (torch.LongTensor, optional): The position IDs for positional embeddings.
            head_mask (torch.FloatTensor, optional): The mask for heads in the multi-head attention mechanism.
            inputs_embeds (torch.FloatTensor, optional): The embedded input sequences.
            encoder_hidden_states (torch.FloatTensor, optional): The hidden states from the encoder.
            encoder_attention_mask (torch.FloatTensor, optional): The attention mask for encoder hidden states.
            output_attentions (bool, optional): Whether to output attentions.
            output_hidden_states (bool, optional): Whether to output hidden states.
            return_dict (bool, optional): Whether to return a dictionary as output.

        Returns:
            BaseModelOutputWithPoolingAndCrossAttentions or torch.Tensor:
                The model outputs with additional pooling and cross-attention information if configured.
        """
        return super().forward(
            input_ids=input_ids,
            bbox=bbox,
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
# 创建一个 LayoutLMForMaskedLM 类，继承自 LayoutLMPreTrainedModel 类
class LayoutLMForMaskedLM(LayoutLMPreTrainedModel):
    # 定义一个包含需要共享权重的 key 的列表
    _tied_weights_keys = ["cls.predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # 定义初始化方法，接受一个 config 对象参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建一个 LayoutLMModel 对象
        self.layoutlm = LayoutLMModel(config)
        # 创建一个 LayoutLMOnlyMLMHead 对象
        self.cls = LayoutLMOnlyMLMHead(config)

        # 调用自定义的 post_init 方法
        self.post_init()

    # 定义方法，返回 layoutlm.embeddings.word_embeddings 对象
    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    # 定义方法，返回 cls.predictions.decoder 对象
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 定义方法，设置 cls.predictions.decoder 对象的值为 new_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 使用装饰器添加模型前向传播方法的文档注释
    # 使用装饰器替换返回文档注释
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 添加模型前向传播的文档注释
        # 使用布尔值参数指定是否返回字典类型输出


# 创建一个 LayoutLMForSequenceClassification 类，继承自 LayoutLMPreTrainedModel 类
class LayoutLMForSequenceClassification(LayoutLMPreTrainedModel):
    # 定义初始化方法，接受一个 config 对象参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置类别数量为 config 的 num_labels 属性
        self.num_labels = config.num_labels
        # 创建一个 LayoutLMModel 对象
        self.layoutlm = LayoutLMModel(config)
        # 创建一个 nn.Dropout 对象，用于屏蔽部分神经元
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个 nn.Linear 对象，用于线性变换
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 调用自定义的 post_init 方法
        self.post_init()

    # 定义方法，返回 layoutlm.embeddings.word_embeddings 对象
    def get_input_embeddings(self):
        return self.layoutlm.embeddings.word_embeddings

    # 使用装饰器添加模型前向传播方法的文档注释
    # 使用装饰器替换返回文档注释
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 添加模型前向传播的文档注释
    # 定义一个方法 `forward`，用于模型的前向传播计算
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    """
    LayoutLM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    sequence labeling (information extraction) tasks such as the [FUNSD](https://guillaumejaume.github.io/FUNSD/)
    dataset and the [SROIE](https://rrc.cvc.uab.es/?ch=13) dataset.
    """,
    LAYOUTLM_START_DOCSTRING,
)
class LayoutLMForTokenClassification(LayoutLMPreTrainedModel):
    """
    LayoutLM 模型，顶部带有一个标记分类头部（在隐藏状态输出之上的线性层），例如用于序列标记（信息提取）任务，如 FUNSD 和 SROIE 数据集。
    继承自 LayoutLMPreTrainedModel。
    """

    def __init__(self, config):
        """
        初始化方法，配置模型参数和各层组件。
        
        Args:
            config (LayoutLMConfig): 包含模型配置的对象实例。
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlm = LayoutLMModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        返回模型的输入嵌入层，这里是 layoutlm.embeddings.word_embeddings。
        
        Returns:
            nn.Embedding: 输入嵌入层对象。
        """
        return self.layoutlm.embeddings.word_embeddings

    @add_start_docstrings_to_model_forward(LAYOUTLM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        前向传播方法，接受一系列输入参数，执行模型的前向计算。
        
        Args:
            input_ids (torch.LongTensor, optional): 输入 token IDs，形状为 [batch_size, sequence_length]。
            bbox (torch.LongTensor, optional): Bounding box 数据，形状为 [batch_size, sequence_length, 4]。
            attention_mask (torch.FloatTensor, optional): 注意力掩码，形状为 [batch_size, sequence_length]。
            token_type_ids (torch.LongTensor, optional): Token 类型 IDs，形状为 [batch_size, sequence_length]。
            position_ids (torch.LongTensor, optional): 位置 IDs，形状为 [batch_size, sequence_length]。
            head_mask (torch.FloatTensor, optional): 头部掩码，形状为 [num_heads] 或 [num_hidden_layers x num_heads]。
            inputs_embeds (torch.FloatTensor, optional): 嵌入输入，形状为 [batch_size, sequence_length, embedding_size]。
            labels (torch.LongTensor, optional): 标签数据，形状为 [batch_size, sequence_length]。
            output_attentions (bool, optional): 是否输出注意力权重。
            output_hidden_states (bool, optional): 是否输出隐藏状态。
            return_dict (bool, optional): 是否返回字典格式的输出。
        
        Returns:
            TokenClassifierOutput: 根据配置返回不同的输出对象，包含损失和预测等信息。
        """
        # 省略部分代码...
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 return_dict 不为 None，则使用它；否则使用 self.config.use_return_dict

        outputs = self.layoutlm(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 调用 layoutlm 模型进行前向传播，传入各种输入参数，并根据 return_dict 参数返回不同的输出形式

        sequence_output = outputs[0]
        # 从模型输出中获取序列输出

        sequence_output = self.dropout(sequence_output)
        # 对序列输出进行 dropout 操作

        logits = self.classifier(sequence_output)
        # 使用分类器对序列输出进行分类，得到 logits（对数概率）

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 定义交叉熵损失函数

            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # 计算预测 logits 和真实标签 labels 之间的交叉熵损失

        if not return_dict:
            output = (logits,) + outputs[2:]
            # 如果 return_dict 是 False，构建输出元组，包括 logits 和额外的输出信息

            return ((loss,) + output) if loss is not None else output
            # 如果有损失，则返回损失和输出元组；否则只返回输出元组

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 返回 TokenClassifierOutput 对象，包含损失、logits、隐藏状态和注意力信息
@add_start_docstrings(
    """
    LayoutLM Model with a span classification head on top for extractive question-answering tasks such as
    [DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the final hidden-states output to compute `span
    start logits` and `span end logits`).
    """,
    LAYOUTLM_START_DOCSTRING,
)
class LayoutLMForQuestionAnswering(LayoutLMPreTrainedModel):
    """
    LayoutLMForQuestionAnswering extends LayoutLMPreTrainedModel for question answering tasks.
    It includes a span classification head on top for tasks like DocVQA.
    """

    def __init__(self, config, has_visual_segment_embedding=True):
        """
        Initializes LayoutLMForQuestionAnswering model with the provided configuration.

        Args:
            config: The configuration object that defines the model architecture.
            has_visual_segment_embedding: Whether to include visual segment embeddings. Default is True.
        """
        super().__init__(config)
        self.num_labels = config.num_labels

        # Initialize LayoutLM model backbone
        self.layoutlm = LayoutLMModel(config)
        
        # Linear layer for question answering outputs
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Returns the word embeddings from the LayoutLM model.

        Returns:
            torch.Tensor: Word embeddings.
        """
        return self.layoutlm.embeddings.word_embeddings

    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```