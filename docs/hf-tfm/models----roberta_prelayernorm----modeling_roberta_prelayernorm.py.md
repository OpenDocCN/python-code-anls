# `.\models\roberta_prelayernorm\modeling_roberta_prelayernorm.py`

```py
# coding=utf-8
# 版权所有 2022 年 Google AI 语言团队和 HuggingFace Inc. 团队
# 版权所有 2018 年 NVIDIA CORPORATION。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可;
# 除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于“原样”分发的，
# 不附带任何明示或暗示的保证或条件。
# 有关详细信息，请参阅许可证。

"""PyTorch RoBERTa-PreLayerNorm 模型。"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN, gelu
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_roberta_prelayernorm import RobertaPreLayerNormConfig

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "andreasmadsen/efficient_mlm_m0.40"
_CONFIG_FOR_DOC = "RobertaPreLayerNormConfig"

ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "andreasmadsen/efficient_mlm_m0.15",
    "andreasmadsen/efficient_mlm_m0.20",
    "andreasmadsen/efficient_mlm_m0.30",
    "andreasmadsen/efficient_mlm_m0.40",
    "andreasmadsen/efficient_mlm_m0.50",
    "andreasmadsen/efficient_mlm_m0.60",
    "andreasmadsen/efficient_mlm_m0.70",
    "andreasmadsen/efficient_mlm_m0.80",
    # 查看所有 RoBERTaWithPreLayerNorm 模型，请访问 https://huggingface.co/models?filter=roberta_with_prelayernorm
]


# 从 transformers.models.roberta.modeling_roberta.RobertaEmbeddings 复制并修改为 RobertaPreLayerNormEmbeddings
class RobertaPreLayerNormEmbeddings(nn.Module):
    """
    与 BertEmbeddings 相同，稍作调整以适应位置嵌入的索引。
    """
    # 初始化函数，用于创建模型对象
    def __init__(self, config):
        # 调用父类构造函数初始化
        super().__init__()
        # 创建词嵌入层，用于将词索引映射为隐藏表示，支持填充标记
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，用于将位置索引映射为隐藏表示
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建类型嵌入层，用于将类型索引映射为隐藏表示
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 使用 TensorFlow 模型变量名的方式命名 LayerNorm，以便能够加载 TensorFlow 的检查点文件
        # self.LayerNorm 不使用蛇形命名法，保持与 TensorFlow 模型变量名一致
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机丢弃输入的一部分数据，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册缓冲区 position_ids，用于保存位置嵌入的索引
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册缓冲区 token_type_ids，用于保存类型嵌入的索引，初始化为零张量
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        # 填充标记索引，用于指示输入中的填充标记
        self.padding_idx = config.pad_token_id
        # 创建位置嵌入层，再次定义，支持填充标记
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    # 前向传播函数，定义了模型的计算流程
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        # 如果未提供位置 ID，则根据输入的 token ids 创建位置 IDs。任何填充的 token 保持填充状态。
        if position_ids is None:
            if input_ids is not None:
                # 从输入的 token ids 创建位置 IDs，使用 padding_idx 和 past_key_values_length 参数
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                # 如果未提供 input_ids，则从 inputs_embeds 创建位置 IDs
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            # 获取 input_ids 的形状
            input_shape = input_ids.size()
        else:
            # 获取 inputs_embeds 的形状，去除最后一个维度（通常是 batch 维度）
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # 将 token_type_ids 设置为在构造函数中注册的缓冲区，通常情况下全为零。这有助于用户在不传递 token_type_ids 的情况下追踪模型，解决了问题 #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                # 从注册的缓冲区中获取 token_type_ids，截取到与序列长度相同的部分，并扩展为与输入形状相同
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 如果模型没有 token_type_ids 属性，则创建全零的 token_type_ids 张量，与输入形状相同
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # 如果未提供 inputs_embeds，则根据 input_ids 获取词嵌入
            inputs_embeds = self.word_embeddings(input_ids)
        # 根据 token_type_ids 获取 token 类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将词嵌入与 token 类型嵌入相加得到总的嵌入向量
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            # 如果位置嵌入类型是绝对位置，则获取位置嵌入并加到总的嵌入向量上
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 应用 LayerNormalization 到嵌入向量
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入向量进行 dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回处理后的嵌入向量
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        直接提供了嵌入向量，无法推断哪些是填充的，因此只生成顺序的位置 IDs。

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 根据输入嵌入向量的设备生成顺序的位置 IDs
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 扩展为与输入形状相同的张量并返回
        return position_ids.unsqueeze(0).expand(input_shape)
# 从transformers.models.bert.modeling_bert.BertSelfAttention复制代码，将Bert->RobertaPreLayerNorm
class RobertaPreLayerNormSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 如果隐藏层大小不是注意力头数的整数倍，并且配置中没有embedding_size属性，则引发错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 位置嵌入类型，默认为absolute
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型为relative_key或relative_key_query，则创建距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 是否为解码器
        self.is_decoder = config.is_decoder

    # 调整形状以便计算分数
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，接收隐藏状态、注意力掩码、头掩码、编码器隐藏状态、编码器注意力掩码等参数
class RobertaPreLayerNormSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 全连接层和dropout层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接收隐藏状态和输入张量，并返回处理后的隐藏状态
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


class RobertaPreLayerNormAttention(nn.Module):
    # 初始化函数，接受配置和位置嵌入类型作为参数
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化方法
        super().__init__()
        # 创建自注意力层对象，并传入配置和位置嵌入类型参数
        self.self = RobertaPreLayerNormSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建自注意力输出层对象，并传入配置参数
        self.output = RobertaPreLayerNormSelfOutput(config)
        # 创建 LayerNorm 层对象，对隐藏状态进行归一化，使用给定的 epsilon 值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化被修剪头部的集合为空集
        self.pruned_heads = set()

    # 从 transformers 库中复制的函数：用于修剪自注意力层的头部
    def prune_heads(self, heads):
        # 如果待修剪的头部集合为空，则直接返回
        if len(heads) == 0:
            return
        # 调用帮助函数，找到可修剪的头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层：查询、键、值、输出密集层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被修剪的头部信息
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
        # 对隐藏状态进行 LayerNorm 归一化
        hidden_states_pre_layer_norm = self.LayerNorm(hidden_states)
        # 使用自注意力层处理归一化后的隐藏状态
        self_outputs = self.self(
            hidden_states_pre_layer_norm,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将自注意力层的输出传递给输出层，并与原始隐藏状态合并
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则在输出中添加注意力权重信息
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要，添加注意力权重信息
        return outputs
# 定义一个名为 RobetaPreLayerNormLayer 的新的 PyTorch 模块，继承自 nn.Module
class RobertaPreLayerNormLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置用于层归一化的 LayerNorm 层，根据配置参数设置隐藏大小和 epsilon 值
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 在维度 1（即第二维）上进行分块前馈的大小设置
        self.seq_len_dim = 1
        # 初始化自注意力模块，使用 RobertaPreLayerNormAttention 类
        self.attention = RobertaPreLayerNormAttention(config)
        # 是否作为解码器模型使用
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力机制的标志
        self.add_cross_attention = config.add_cross_attention
        # 如果设置了添加交叉注意力，需要在解码器模型中使用，否则引发错误
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化交叉注意力模块，使用 RobertaPreLayerNormAttention 类，并设置位置嵌入类型为 "absolute"
            self.crossattention = RobertaPreLayerNormAttention(config, position_embedding_type="absolute")
        # 初始化中间层模块，使用 RobertaPreLayerNormIntermediate 类
        self.intermediate = RobertaPreLayerNormIntermediate(config)
        # 初始化输出层模块，使用 RobertaPreLayerNormOutput 类
        self.output = RobertaPreLayerNormOutput(config)

    # 定义前向传播方法
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
        # 应用层归一化到输入的隐藏状态
        hidden_states = self.LayerNorm(hidden_states)
        # 通过全连接层处理隐藏状态，调整其维度为中间大小
        hidden_states = self.dense(hidden_states)
        # 应用激活函数到中间隐藏状态
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的隐藏状态作为输出
        return hidden_states
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # Perform self-attention using the stored key/value pairs if available
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # Extract the attention output tensor
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            # Exclude the first and last element (self-attention outputs) for decoder outputs
            outputs = self_attention_outputs[1:-1]
            # Retrieve the present key/value for attention caching
            present_key_value = self_attention_outputs[-1]
        else:
            # Include self-attention outputs in outputs for non-decoder case
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # Perform cross-attention using the stored key/value pairs if available
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # Extract the cross-attention output tensor
            attention_output = cross_attention_outputs[0]
            # Add cross-attention outputs to outputs
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # Apply chunking mechanism to feed forward layer for memory efficiency
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # Append the layer output to outputs
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            # Append present key/value for attention caching as the last element of outputs
            outputs = outputs + (present_key_value,)

        # Return all computed outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        # Pass attention output through intermediate layer
        intermediate_output = self.intermediate(attention_output)
        # Pass through final output layer to get the final layer output
        layer_output = self.output(intermediate_output, attention_output)
        # Return the final layer output
        return layer_output
# 从 transformers.models.bert.modeling_bert.BertEncoder 复制并修改为使用 RobertaPreLayerNormEncoder
class RobertaPreLayerNormEncoder(nn.Module):
    # 初始化方法，接收一个配置对象 config
    def __init__(self, config):
        super().__init__()
        # 将配置对象保存到实例变量中
        self.config = config
        # 创建一个 nn.ModuleList，其中包含 config.num_hidden_layers 个 RobertaPreLayerNormLayer 对象
        self.layer = nn.ModuleList([RobertaPreLayerNormLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点标志，默认为 False
        self.gradient_checkpointing = False

    # 前向传播方法
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
    # 定义函数的返回类型为一个元组，包含了 torch.Tensor 或者 BaseModelOutputWithPastAndCrossAttentions 类型
    -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果不输出隐藏状态，则初始化为空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力权重，则初始化为空元组
        all_self_attentions = () if output_attentions else None
        # 如果不输出交叉注意力权重且配置要求添加交叉注意力，则初始化为空元组
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果开启了梯度检查点且处于训练模式
        if self.gradient_checkpointing and self.training:
            # 如果设置了 use_cache=True，与梯度检查点不兼容，发出警告并设置 use_cache=False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果 use_cache=True，则初始化下一个解码器缓存为空元组
        next_decoder_cache = () if use_cache else None

        # 遍历每一个解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果存在头部掩码，则使用对应层的头部掩码，否则为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果存在过去的键值对，则使用对应层的过去键值对，否则为 None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果开启了梯度检查点且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数执行当前层模块的调用
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

            # 更新隐藏状态为当前层模块计算后的输出的第一个元素（隐藏状态）
            hidden_states = layer_outputs[0]
            # 如果 use_cache=True，则将当前层模块计算后的输出的最后一个元素（缓存）添加到 next_decoder_cache 中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果输出注意力权重，则将当前层模块计算后的输出的第二个元素（自注意力权重）添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果配置要求添加交叉注意力，且当前层模块计算后的输出中有第三个元素（交叉注意力权重），则添加到 all_cross_attentions 中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果输出隐藏状态，则将最终隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，则返回一个包含非空元素的元组
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
        # 否则返回一个 BaseModelOutputWithPastAndCrossAttentions 对象，包含各类输出结果
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从transformers库中的BertPooler类复制而来，用于RoBERTa模型的预处理层归一化池化
class RobertaPreLayerNormPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 密集连接层，输入和输出维度均为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数使用双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 池化操作通过选择第一个标记对应的隐藏状态来实现
        first_token_tensor = hidden_states[:, 0]
        # 将选择的隐藏状态传递给密集连接层
        pooled_output = self.dense(first_token_tensor)
        # 应用激活函数到输出
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出张量
        return pooled_output


# 从transformers库中的RobertaPreTrainedModel类复制而来，用于RoBERTa模型的预处理层归一化预训练模型
class RobertaPreLayerNormPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，处理权重初始化和下载预训练模型的简单接口。
    """

    # 配置类为RobertaPreLayerNormConfig
    config_class = RobertaPreLayerNormConfig
    # 基础模型前缀为"roberta_prelayernorm"
    base_model_prefix = "roberta_prelayernorm"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不分割的模块列表
    _no_split_modules = ["RobertaPreLayerNormEmbeddings", "RobertaPreLayerNormSelfAttention"]

    # 从transformers库中的BertPreTrainedModel类复制而来的方法，用于初始化权重
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 与TensorFlow版本稍有不同，使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果有偏置项，则初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 初始化嵌入层权重为正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果有填充索引，则将对应位置的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 初始化LayerNorm层的偏置项为零
            module.bias.data.zero_()
            # 初始化LayerNorm层的权重为1
            module.weight.data.fill_(1.0)


ROBERTA_PRELAYERNORM_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
        Parameters:
            config ([`RobertaPreLayerNormConfig`]): Model configuration class with all the parameters of the
                model. Initializing with a config file does not load the weights associated with the model, only the
                configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
# 用于定义 RoBERTa-PreLayerNorm 模型，输出原始隐藏状态而不带特定的输出头
ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            This parameter can only be used when the model is initialized with `type_vocab_size` parameter with value
            >= 2. All the value in this tensor should be always < type_vocab_size.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class RobertaPreLayerNormModel(RobertaPreLayerNormPreTrainedModel):
    """
    这个模型可以作为一个编码器（仅自注意力）或者解码器来使用。如果作为解码器使用，则在自注意力层之间添加交叉注意力层，
    这遵循了*Attention is all you need*一书中描述的架构，作者为Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser和Illia Polosukhin。

    要作为解码器使用，需要将`is_decoder`参数设置为`True`。要用于Seq2Seq模型，需要将`is_decoder`参数和`add_cross_attention`
    参数都设置为`True`；此时需要一个`encoder_hidden_states`作为前向传递的输入。

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = RobertaPreLayerNormEmbeddings(config)
        self.encoder = RobertaPreLayerNormEncoder(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.pooler = RobertaPreLayerNormPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        获取输入嵌入层的方法
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        设置输入嵌入层的方法
        """
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        剪枝模型的注意力头方法。heads_to_prune: 字典，格式为{层号: 需要在该层中剪枝的头列表}，参见基类PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
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
    ):
        """
        前向传递方法，详细参数见函数上方的注释说明。
        """
@add_start_docstrings(
    """
    RoBERTa-PreLayerNorm Model with a `language modeling` head on top for CLM fine-tuning.
    """
    # 将 RoBERTa-PreLayerNorm 模型与 CLM 微调的语言建模头部组合在一起
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
# Copied from transformers.models.roberta.modeling_roberta.RobertaForCausalLM with modifications to support a different model configuration
class RobertaPreLayerNormForCausalLM(RobertaPreLayerNormPreTrainedModel):
    # Define keys for tied weights in the model
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        # Warn if the model is not configured as a decoder
        if not config.is_decoder:
            logger.warning(
                "If you want to use `RobertaPreLayerNormLMHeadModel` as a standalone, add `is_decoder=True.`"
            )

        # Initialize the Roberta model with pre-layer normalization and without pooling layer
        self.roberta_prelayernorm = RobertaPreLayerNormModel(config, add_pooling_layer=False)
        # Initialize the language model head for pre-layer normalization
        self.lm_head = RobertaPreLayerNormLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        # Return the decoder part of the language model head
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        # Set new embeddings for the decoder part of the language model head
        self.lm_head.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 为生成准备输入数据，在生成过程中使用的方法
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 获取输入张量的形状
        input_shape = input_ids.shape
        
        # 如果没有提供注意力遮罩，则创建全为1的遮罩张量，与输入张量形状相同
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果传入了过去的键值（past_key_values），则根据过去的长度调整输入的ID
        if past_key_values is not None:
            # 获取过去状态的长度（通常是过去的输入序列长度）
            past_length = past_key_values[0][0].shape[2]

            # 如果当前输入ID的长度大于过去的长度，则截取掉前面部分的输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则，默认行为是保留最后一个输入ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # 返回包含输入ID、注意力遮罩和过去键值的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

    # 重新排序缓存数据，以适应束搜索生成时的顺序
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        # 对每一层的过去状态进行重新排序
        for layer_past in past_key_values:
            reordered_past += (
                # 使用束索引（beam_idx）将过去状态重新排序，并转移到相同设备上
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去状态元组
        return reordered_past
# 定义 RoBERTa-PreLayerNorm 模型，带有在顶部的语言建模头部
@add_start_docstrings(
    """RoBERTa-PreLayerNorm Model with a `language modeling` head on top.""", ROBERTA_PRELAYERNORM_START_DOCSTRING
)
class RobertaPreLayerNormForMaskedLM(RobertaPreLayerNormPreTrainedModel):
    # 共享权重的键列表
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    # 从 transformers.models.roberta.modeling_roberta.RobertaForMaskedLM.__init__ 复制而来，对应的 ROBERTA->ROBERTA_PRELAYERNORM,Roberta->RobertaPreLayerNorm,roberta->roberta_prelayernorm
    def __init__(self, config):
        super().__init__(config)

        # 如果配置为解码器，发出警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaPreLayerNormForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化 RoBERTa-PreLayerNorm 模型和 LM 头部
        self.roberta_prelayernorm = RobertaPreLayerNormModel(config, add_pooling_layer=False)
        self.lm_head = RobertaPreLayerNormLMHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 返回 LM 头部的输出嵌入
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置 LM 头部的输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 从 transformers.models.roberta.modeling_roberta.RobertaForMaskedLM.forward 复制而来，对应的 ROBERTA->ROBERTA_PRELAYERNORM,Roberta->RobertaPreLayerNorm,roberta->roberta_prelayernorm
    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
        expected_output="' Paris'",
        expected_loss=0.69,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 函数参数文档字符串，指定输入参数的格式和功能
        **kwargs,
    ):
        # 函数的具体实现在后续的代码中
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # Decide whether to return a dictionary format based on the provided `return_dict` parameter or the model's configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through the Roberta model with specified inputs and optional arguments
        outputs = self.roberta_prelayernorm(
            input_ids,
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
        # Extract the sequence output from the model's outputs
        sequence_output = outputs[0]
        # Generate prediction scores using the language modeling head
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        # Calculate masked language modeling loss if labels are provided
        if labels is not None:
            # Move labels tensor to the device of prediction_scores to enable parallel computation if using model parallelism
            labels = labels.to(prediction_scores.device)
            # Define CrossEntropyLoss function
            loss_fct = CrossEntropyLoss()
            # Compute masked LM loss based on prediction scores and labels
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # If return_dict is False, construct output tuple excluding masked_lm_loss if it's None
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # Return MaskedLMOutput object containing loss, logits, hidden states, and attentions
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# Copied from transformers.models.roberta.modeling_roberta.RobertaLMHead with Roberta->RobertaPreLayerNorm
class RobertaPreLayerNormLMHead(nn.Module):
    """RobertaPreLayerNorm Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，用于将隐藏状态映射到相同大小的空间
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化一个 LayerNorm 层，用于归一化隐藏状态的分布
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化一个全连接层，用于最终的分类，输出的大小是词汇表的大小
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # 初始化一个偏置项参数，用于输出层的偏置
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        # 将输入特征通过全连接层映射
        x = self.dense(features)
        # 应用 GELU 激活函数
        x = gelu(x)
        # 应用 LayerNorm 层，归一化输出
        x = self.layer_norm(x)

        # 通过输出层映射到词汇表大小的空间，加上偏置
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # 如果输出层的偏置设备类型为 "meta"，则将输出层的偏置与模型的偏置绑定
        # 用于加速兼容性，并且不破坏向后兼容性
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


@add_start_docstrings(
    """
    RoBERTa-PreLayerNorm Model transformer with a sequence classification/regression head on top (a linear layer on top
    of the pooled output) e.g. for GLUE tasks.
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class RobertaPreLayerNormForSequenceClassification(RobertaPreLayerNormPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # 使用 RoBERTaPreLayerNormModel 创建一个 RoBERTa-PreLayerNorm 模型，不添加池化层
        self.roberta_prelayernorm = RobertaPreLayerNormModel(config, add_pooling_layer=False)
        # 创建一个 RoBERTa-PreLayerNorm 分类头部
        self.classifier = RobertaPreLayerNormClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # Copied from transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification.forward with roberta->roberta_prelayernorm
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确保返回的字典对象不为空，根据配置决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用预处理后的 RoBERTa 模型获取输出
        outputs = self.roberta_prelayernorm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 从 RoBERTa 输出中获取序列输出
        sequence_output = outputs[0]
        # 将序列输出传入分类器获取 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        if labels is not None:
            # 将标签移到正确的设备以启用模型并行计算
            labels = labels.to(logits.device)
            # 根据配置确定问题类型（回归、单标签分类或多标签分类）
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
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不使用返回字典，则构建输出元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 使用返回字典的情况下，返回 SequenceClassifierOutput 对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加文档字符串，描述了此类的作用是基于 RobustPreLayerNorm 模型的多选分类器
@add_start_docstrings(
    """
    RobertaPreLayerNorm Model with a multiple choice classification head on top (a linear layer on top of the pooled
    output and a softmax) e.g. for RocStories/SWAG tasks.
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,  # 包含 ROBERTA_PRELAYERNORM_START_DOCSTRING 的文档字符串
)
# 基于 RobertaPreLayerNormPreTrainedModel 创建的类，用于多选任务
class RobertaPreLayerNormForMultipleChoice(RobertaPreLayerNormPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 RobertaPreLayerNormModel 对象
        self.roberta_prelayernorm = RobertaPreLayerNormModel(config)
        # 添加 dropout 层，使用配置中的 hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加分类器线性层，将隐藏状态映射到单一输出维度
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(
        ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 包含 _CHECKPOINT_FOR_DOC 的代码示例文档字符串
        output_type=MultipleChoiceModelOutput,  # 指定输出类型为 MultipleChoiceModelOutput
        config_class=_CONFIG_FOR_DOC,  # 包含 _CONFIG_FOR_DOC 的配置类文档字符串
    )
    # 前向传播方法，接收多个输入和配置参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 模型前向传播的输入文档字符串
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确保返回字典不为空，若为空则使用模型配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算选择项的数量，即第二维度的大小
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入张量展平为二维张量，便于模型处理
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 将展平后的输入传递给模型的前处理层
        outputs = self.roberta_prelayernorm(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 提取汇总的输出
        pooled_output = outputs[1]

        # 对汇总输出应用 dropout 操作
        pooled_output = self.dropout(pooled_output)
        # 使用分类器对汇总后的输出进行分类预测
        logits = self.classifier(pooled_output)
        # 调整 logits 的形状以便计算损失
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            # 将标签移到正确的设备上以启用模型的并行计算
            labels = labels.to(reshaped_logits.device)
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算损失
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            # 如果不返回字典，则按顺序返回损失和模型输出
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回多项选择模型的输出，包括损失、调整后的 logits、隐藏状态和注意力
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用带有标记分类头部的 RobertaPreLayerNorm 模型，用于例如命名实体识别（NER）任务
# 继承自 RobertaPreLayerNormPreTrainedModel 类
class RobertaPreLayerNormForTokenClassification(RobertaPreLayerNormPreTrainedModel):
    
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels
        
        # 使用 RobertaPreLayerNormModel 创建 RobertaPreLayerNorm 模型实例，不添加池化层
        self.roberta_prelayernorm = RobertaPreLayerNormModel(config, add_pooling_layer=False)
        
        # 确定分类器的 dropout 比例
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 定义线性分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 将模型向前传播的方法
    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 从 transformers.models.roberta.modeling_roberta.RobertaForTokenClassification.forward 复制，将 roberta 替换为 roberta_prelayernorm
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
    
        # Decide whether to use the provided return_dict or default to the model's configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # Pass the inputs through the Roberta model layers
        outputs = self.roberta_prelayernorm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # Extract the sequence output from the model's outputs
        sequence_output = outputs[0]
    
        # Apply dropout to the sequence output
        sequence_output = self.dropout(sequence_output)
    
        # Feed the sequence output into the classifier to obtain logits
        logits = self.classifier(sequence_output)
    
        # Initialize loss as None
        loss = None
    
        # Compute the loss if labels are provided
        if labels is not None:
            # Move labels to the correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Use CrossEntropyLoss to compute the classification loss
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
        # If return_dict is False, return output tuple without loss
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        # If return_dict is True, return TokenClassifierOutput object
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从transformers.models.roberta.modeling_roberta.RobertaClassificationHead复制而来，将Roberta改为RobertaPreLayerNorm
class RobertaPreLayerNormClassificationHead(nn.Module):
    """用于句子级分类任务的头部模块。"""

    def __init__(self, config):
        super().__init__()
        # 线性层，输入和输出大小都为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 分类器的dropout率，如果未指定则使用config.hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # Dropout层
        self.dropout = nn.Dropout(classifier_dropout)
        # 输出投影层，将隐藏状态映射到config.num_labels维度
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 获取特征的第一个token的隐藏状态（相当于[CLS] token）
        x = features[:, 0, :]
        x = self.dropout(x)  # 应用dropout
        x = self.dense(x)  # 全连接层
        x = torch.tanh(x)  # tanh激活函数
        x = self.dropout(x)  # 再次应用dropout
        x = self.out_proj(x)  # 输出投影层
        return x


@add_start_docstrings(
    """
    基于RoBERTaPreLayerNorm模型的用于抽取式问答（例如SQuAD）的跨度分类头部模块。
    在隐藏状态输出之上添加了线性层，用于计算'起始跨度logits'和'结束跨度logits'。
    """,
    ROBERTA_PRELAYERNORM_START_DOCSTRING,
)
class RobertaPreLayerNormForQuestionAnswering(RobertaPreLayerNormPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # RoBERTa模型的预层标准化版本，不添加池化层
        self.roberta_prelayernorm = RobertaPreLayerNormModel(config, add_pooling_layer=False)
        # 线性层，将隐藏状态映射到config.num_labels维度
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_PRELAYERNORM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 从transformers.models.roberta.modeling_roberta.RobertaForQuestionAnswering.forward复制而来，将roberta->roberta_prelayernorm
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 默认情况下，如果 return_dict 为 None，则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Roberta 的前处理层处理输入，获取模型输出
        outputs = self.roberta_prelayernorm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取序列输出
        sequence_output = outputs[0]

        # 将序列输出传入 QA 输出层获取 logits
        logits = self.qa_outputs(sequence_output)

        # 将 logits 按最后一个维度分割为起始位置和结束位置的 logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # 去除多余的维度并保证连续性
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果 start_positions 或 end_positions 是多维的，压缩至一维
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # 忽略超出模型输入长度的位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，忽略指定索引的位置
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果 return_dict 为 False，则输出 tuple
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 如果 return_dict 为 True，则输出 QuestionAnsweringModelOutput 对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 根据输入的 `input_ids` 和 `padding_idx` 创建位置 ID 列表，忽略填充符号并替换为其位置数字。
# 位置数字从 `padding_idx + 1` 开始计数。此函数修改自 fairseq 的 `utils.make_positions`。

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: 输入的 torch.Tensor，包含输入序列的标识符
        padding_idx: 填充符号的索引，要被忽略的位置
        past_key_values_length: 过去的键值对长度，用于增量索引计算

    Returns:
        torch.Tensor: 包含位置 ID 的张量
    """

    # 创建一个掩码张量，标记非填充符号的位置为1，填充符号位置为0
    mask = input_ids.ne(padding_idx).int()

    # 计算增量索引，累加非填充符号的数量，并加上过去的键值对长度
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask

    # 将增量索引转换为长整型，并加上填充索引，以获得最终的位置 ID
    return incremental_indices.long() + padding_idx
```