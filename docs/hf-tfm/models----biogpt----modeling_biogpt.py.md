# `.\models\biogpt\modeling_biogpt.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Team and Microsoft Research AI4Science All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BioGPT model."""


import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_biogpt import BioGptConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "microsoft/biogpt"
_CONFIG_FOR_DOC = "BioGptConfig"


BIOGPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/biogpt",
    "microsoft/BioGPT-Large",
    # See all BioGPT models at https://huggingface.co/models?filter=biogpt
]


# Copied from transformers.models.opt.modeling_opt.OPTLearnedPositionalEmbedding with OPT->BioGpt
class BioGptLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # BioGpt is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->BioGpt
class BioGptAttention(nn.Module):
    """
    Placeholder for the BioGPT Attention module.
    This class will define the attention mechanism for BioGPT.
    Actual implementation details will be filled in later.
    """
    
    # Placeholder for attention module, actual implementation details pending.
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # 初始化函数，定义多头注意力模型的参数和层
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[BioGptConfig] = None,
    ):
        super().__init__()  # 调用父类的初始化函数
        self.embed_dim = embed_dim  # 设置嵌入维度
        self.num_heads = num_heads  # 设置注意力头的数量
        self.dropout = dropout  # 设置dropout比例
        self.head_dim = embed_dim // num_heads  # 计算每个头的维度
        self.config = config  # 设置配置参数

        if (self.head_dim * num_heads) != self.embed_dim:
            # 检查嵌入维度是否能被注意力头数整除
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子，用于调整注意力分数的大小
        self.is_decoder = is_decoder  # 是否为解码器
        self.is_causal = is_causal  # 是否使用因果注意力

        # 初始化四个线性投影层，用于对输入进行线性变换
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 将输入张量重塑为适合多头注意力计算的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数，实现多头注意力的计算过程
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
# 定义一个名为 BioGptDecoderLayer 的自定义神经网络层，继承自 nn.Module
class BioGptDecoderLayer(nn.Module):
    # 初始化函数，接受一个名为 config 的 BioGptConfig 类型参数
    def __init__(self, config: BioGptConfig):
        # 调用父类 nn.Module 的初始化函数
        super().__init__()
        # 设置隐藏层大小为配置中的 hidden_size
        self.embed_dim = config.hidden_size

        # 创建一个名为 self_attn 的 BioGptAttention 实例
        self.self_attn = BioGptAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            is_decoder=True,
        )

        # 设置隐藏层的 dropout 概率为配置中的 hidden_dropout_prob
        self.dropout = config.hidden_dropout_prob
        # 根据配置中的 hidden_act 选择激活函数，并赋值给 activation_fn
        self.activation_fn = ACT2FN[config.hidden_act]
        # 设置激活函数的 dropout 概率为配置中的 activation_dropout
        self.activation_dropout = config.activation_dropout

        # 创建一个具有 LayerNorm 的自注意力层，输入维度为 embed_dim
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 创建一个线性层，输入维度为 embed_dim，输出维度为 intermediate_size
        self.fc1 = nn.Linear(self.embed_dim, config.intermediate_size)
        # 创建一个线性层，输入维度为 intermediate_size，输出维度为 embed_dim
        self.fc2 = nn.Linear(config.intermediate_size, self.embed_dim)
        # 创建一个具有 LayerNorm 的最终层，输入维度为 embed_dim
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播函数，接受多个输入参数，并返回计算结果
    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态的张量输入
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码张量输入
        layer_head_mask: Optional[torch.Tensor] = None,  # 可选的层头掩码张量输入
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 可选的过去键值元组输入
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重，默认为 False
        use_cache: Optional[bool] = True,  # 是否使用缓存，默认为 True
        ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        # 保留输入的原始状态，用于残差连接
        residual = hidden_states

        # 对输入的 hidden_states 进行 layer normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # 如果有过去的 key/value 缓存，则提取前两个位置的缓存，否则为 None
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 执行 self-attention 操作，返回更新后的 hidden_states、self attention 权重和当前的 key/value 缓存
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对更新后的 hidden_states 应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差与更新后的 hidden_states 相加，实现残差连接
        hidden_states = residual + hidden_states

        # Fully Connected
        # 保留输入的原始状态，用于残差连接
        residual = hidden_states
        # 对输入的 hidden_states 进行 layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        # 执行第一个全连接层的操作
        hidden_states = self.fc1(hidden_states)
        # 应用激活函数
        hidden_states = self.activation_fn(hidden_states)
        # 对更新后的 hidden_states 应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 执行第二个全连接层的操作
        hidden_states = self.fc2(hidden_states)
        # 对更新后的 hidden_states 应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差与更新后的 hidden_states 相加，实现残差连接
        hidden_states = residual + hidden_states

        # 构建输出元组
        outputs = (hidden_states,)

        # 如果需要输出 attentions，则将 self attention 的权重添加到输出中
        if output_attentions:
            outputs += (self_attn_weights,)

        # 如果需要使用缓存，则将当前的 key/value 缓存添加到输出中
        if use_cache:
            outputs += (present_key_value,)

        # 返回最终的输出元组
        return outputs
class BioGptPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类
    config_class = BioGptConfig
    # 模型名前缀
    base_model_prefix = "biogpt"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 如果是线性层，使用正态分布初始化权重，偏置置零
            # 与 TF 版本稍有不同，TF 使用截断正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 如果是嵌入层，使用正态分布初始化权重，特定位置索引处权重置零
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 如果是层归一化层，偏置置零，权重置为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


BIOGPT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~BioGptConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BIOGPT_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare BioGPT Model transformer outputting raw hidden-states without any specific head on top.",
    BIOGPT_START_DOCSTRING,
)
class BioGptModel(BioGptPreTrainedModel):
    def __init__(self, config: BioGptConfig):
        super().__init__(config)
        self.config = config
        self.layerdrop = config.layerdrop
        self.dropout = config.hidden_dropout_prob
        self.embed_dim = config.hidden_size
        self.padding_idx = config.pad_token_id
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0

        # 嵌入层：词汇量大小为 config.vocab_size，嵌入维度为 self.embed_dim，使用 padding_idx 进行填充
        self.embed_tokens = nn.Embedding(config.vocab_size, self.embed_dim, self.padding_idx)
        # 学习到的位置嵌入：最大位置嵌入数为 config.max_position_embeddings，嵌入维度为 self.embed_dim
        self.embed_positions = BioGptLearnedPositionalEmbedding(config.max_position_embeddings, self.embed_dim)

        # 层列表：包含 config.num_hidden_layers 个 BioGptDecoderLayer 层
        self.layers = nn.ModuleList([BioGptDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 层归一化层：输入维度为 self.embed_dim
        self.layer_norm = nn.LayerNorm(self.embed_dim)

        self.gradient_checkpointing = False
        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    # 使用装饰器将下面的函数添加文档字符串，文档字符串包含有关输入参数的信息
    @add_start_docstrings_to_model_forward(BIOGPT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器添加代码示例文档字符串，指定模型的检查点、输出类型、配置类等信息
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs 张量，可以为 None
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩张量，可以为 None
        head_mask: Optional[torch.FloatTensor] = None,  # 头部遮罩张量，可以为 None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 嵌入输入张量，可以为 None
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 过去的键值对，可以为 None
        use_cache: Optional[bool] = None,  # 是否使用缓存，可以为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可以为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可以为 None
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可以为 None
# 为 BioGPT 模型添加文档字符串，说明其具有顶部的语言建模头用于 CLM 微调
@add_start_docstrings(
    """BioGPT Model with a `language modeling` head on top for CLM fine-tuning.""", BIOGPT_START_DOCSTRING
)
# 定义 BioGptForCausalLM 类，继承自 BioGptPreTrainedModel 类
class BioGptForCausalLM(BioGptPreTrainedModel):
    # 定义权重绑定的键值列表
    _tied_weights_keys = ["output_projection.weight"]

    # 初始化函数，接收一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建一个 BioGptModel 实例，并赋值给 self.biogpt
        self.biogpt = BioGptModel(config)
        # 创建一个线性层，用于将隐藏状态映射到词汇表大小的输出
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出词嵌入的方法
    def get_output_embeddings(self):
        return self.output_projection

    # 设置输出词嵌入的方法
    def set_output_embeddings(self, new_embeddings):
        self.output_projection = new_embeddings

    # 为 forward 方法添加文档字符串，描述输入参数的作用，使用给定的模板
    @add_start_docstrings_to_model_forward(BIOGPT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档字符串，指定检查点、输出类型和配置类
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播方法，接收多个可选的输入参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 根据需要确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给预训练模型并获取输出
        outputs = self.biogpt(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中提取序列输出（即预测的序列）
        sequence_output = outputs[0]
        # 将序列输出投影到预测分数（logits）空间
        prediction_scores = self.output_projection(sequence_output)

        lm_loss = None
        if labels is not None:
            # 如果有提供标签，计算语言建模的损失
            # 预测的分数向左移动一位，以便进行下一个标记的预测
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            # 使用交叉熵损失函数计算损失
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            # 如果不需要返回字典，构建输出元组
            output = (prediction_scores,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        # 如果需要返回字典，构建包含附加信息的对象
        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )
        # 如果 past_key_values 参数不为 None，则根据定义的行为保留输入的最后几个 token
        if past_key_values is not None:
            # 获取第一个 past_key_values 的第一个元素的形状的第三个维度长度
            past_length = past_key_values[0][0].shape[2]

            # 如果输入的 input_ids 长度大于 past_length，则保留最后 past_length 个 token
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则，默认保留最后一个 token
                remove_prefix_length = input_ids.shape[1] - 1

            # 更新 input_ids，仅保留所需的部分
            input_ids = input_ids[:, remove_prefix_length:]

        # 如果 inputs_embeds 不为 None 且 past_key_values 为 None，则将其作为模型输入
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # 否则，默认使用 input_ids 作为模型输入
            model_inputs = {"input_ids": input_ids}

        # 更新 model_inputs 字典，添加 attention_mask、past_key_values 和 use_cache 参数
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        # 返回组装好的模型输入
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 重新排序 past_key_values 中的数据，根据给定的 beam_idx
        reordered_past = ()
        for layer_past in past_key_values:
            # 对每层的 past_state 执行重新排序操作，根据 beam_idx
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的 past_key_values
        return reordered_past
# 使用装饰器为类添加文档字符串，描述了这是一个在 BioGPT 模型基础上增加了标记分类头的模型，用于命名实体识别（NER）任务
@add_start_docstrings(
    """
    BioGPT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BIOGPT_START_DOCSTRING,
)
# 定义 BioGptForTokenClassification 类，继承自 BioGptPreTrainedModel
class BioGptForTokenClassification(BioGptPreTrainedModel):
    # 初始化函数，接受一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置类别数量为配置中的 num_labels
        self.num_labels = config.num_labels

        # 创建一个 BioGptModel 对象，并将其赋值给 self.biogpt
        self.biogpt = BioGptModel(config)

        # 检查配置中是否有 classifier_dropout 属性，并根据其值设置分类器的 dropout
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        else:
            # 否则使用配置中的 hidden_dropout_prob 作为 dropout
            classifier_dropout = config.hidden_dropout_prob
        # 创建一个 Dropout 层，用于模型训练中的随机失活
        self.dropout = nn.Dropout(classifier_dropout)

        # 创建一个全连接层，将隐藏状态的输出映射到 num_labels 大小的向量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 调用模型初始化后的处理函数，用于进一步初始化工作
        self.post_init()

    # 使用装饰器为 forward 方法添加文档字符串，描述了其输入参数和输出
    @add_start_docstrings_to_model_forward(BIOGPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播方法定义，接受多个输入参数和返回值
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        # 前向传播方法的具体实现部分尚未提供，在此处省略
        pass


这段代码定义了一个 `BioGptForTokenClassification` 类，它是在 `BioGptPreTrainedModel` 基础上构建的，用于处理标记分类任务，例如命名实体识别（NER）。类中的 `forward` 方法尚未具体实现前向传播逻辑，但通过装饰器和注释详细描述了输入参数和输出，以及一些样例和模型配置的文档。
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不是 None，则使用传入的 return_dict 值；否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 biogpt 模型进行推断
        transformer_outputs = self.biogpt(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的 hidden states
        hidden_states = transformer_outputs[0]
        # 对 hidden states 进行 dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 使用分类器得到 logits
        logits = self.classifier(hidden_states)

        loss = None
        # 如果提供了 labels，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 只保留 attention_mask 中激活部分的损失
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                # 使用 active_loss 选择性地处理 labels
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                # 计算损失
                loss = loss_fct(active_logits, active_labels)
            else:
                # 计算整体的损失
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不需要返回字典形式的输出，则返回一个元组
        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 对象，包括损失、logits、hidden states 和 attentions
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
# 使用装饰器为类添加文档字符串，描述了 BioGptForSequenceClassification 模型的作用和工作原理
@add_start_docstrings(
    """
    The BioGpt Model transformer with a sequence classification head on top (linear layer).

    [`BioGptForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it is required to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    BIOGPT_START_DOCSTRING,
)
class BioGptForSequenceClassification(BioGptPreTrainedModel):
    def __init__(self, config: BioGptConfig):
        # 调用父类构造函数初始化模型配置
        super().__init__(config)
        # 从配置中获取类别数量
        self.num_labels = config.num_labels
        # 初始化 BioGptModel 模型
        self.biogpt = BioGptModel(config)
        # 使用线性层进行分类，输出维度为隐藏层大小到类别数量的映射
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    # 使用装饰器为 forward 方法添加文档字符串，描述了输入参数和输出的详细说明
    @add_start_docstrings_to_model_forward(BIOGPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 实现模型的前向传播逻辑，具体细节通过装饰器文档字符串提供

    # 获取输入的嵌入层（embeddings）
    def get_input_embeddings(self):
        return self.biogpt.embed_tokens

    # 设置输入的嵌入层（embeddings）
    def set_input_embeddings(self, value):
        self.biogpt.embed_tokens = value
```