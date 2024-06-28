# `.\models\opt\modeling_opt.py`

```py
# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
"""
PyTorch OPT model.
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_opt import OPTConfig

# Check if flash attention 2 is available and import necessary functions
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# Get logger instance for the current module
logger = logging.get_logger(__name__)

# Documented variables for model documentation
_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Expected output shape for the base model
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# Checkpoint and expected outputs for sequence classification
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/opt-350m-dummy-sc"
_SEQ_CLASS_EXPECTED_LOSS = 1.71
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"

# List of pretrained model archives for OPT models
OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]

# Function to get unpad data from attention mask
# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    """
    Get indices, cumulative sequence lengths, and maximum sequence length from attention mask.

    Args:
        attention_mask (torch.Tensor): Attention mask tensor.

    Returns:
        Tuple: Tuple containing:
            - indices (torch.Tensor): Indices of attention mask where True.
            - cu_seqlens (torch.Tensor): Cumulative sequence lengths.
            - max_seqlen_in_batch (int): Maximum sequence length in the batch.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch

class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        # 初始化函数，用于初始化一个嵌入层对象。
        # 如果设置了 padding_idx，偏移嵌入 ID 2 个单位，并相应调整 num_embeddings。
        # 其他模型没有这种特殊处理。
        self.offset = 2
        # 调用父类的初始化方法，将 num_embeddings 加上偏移量 self.offset 传递给父类
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        # 将 attention_mask 转换为 long 类型
        attention_mask = attention_mask.long()

        # 根据 attention_mask 创建位置编码
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # 如果 past_key_values_length 大于 0，则截取位置编码
        positions = positions[:, past_key_values_length:]

        # 调用父类的 forward 方法，传递调整后的位置编码 positions + self.offset
        return super().forward(positions + self.offset)
    class OptFlashAttention2(OPTAttention):
        """
        OPT flash attention module. This module inherits from `OPTAttention` as the weights of the module stays untouched.
        The only required change would be on the forward pass where it needs to correctly call the public API of flash
        attention and deal with padding tokens in case the input contains any of them.
        """

        # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
            # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
            # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
            self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
    # 定义一个方法，用于执行前向传播操作，接受多个参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward 复制并引用
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        # Determine if causal masking should be applied
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # Temporary check for specific condition until version 2.1
            # of Flash Attention for RoCm; see LlamaFlashAttention2 __init__ comment
            causal = self.is_causal and query_length != 1

        # Check if there are any padding tokens in the input sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            # Unpad the input based on the attention mask
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            # Extract sequence lengths after unpadding
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # Perform variable length Flash Attention computation
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            # Pad the attention output based on the unpadding indices
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # Perform standard Flash Attention computation without masking
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        # Return the computed attention output
        return attn_output
    # 在内部方法中处理输入数据，用于构建查询、键和值的层
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取未填充数据的索引、当前序列长度和批次中最大的序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # 获取键值对层的形状信息
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 根据未填充数据的索引重新排列键值对层，以便处理未填充的数据
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 根据未填充数据的索引重新排列值对层，以便处理未填充的数据
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        
        # 根据查询长度调整查询层的处理方式
        if query_length == kv_seq_len:
            # 如果查询长度等于键值序列长度，按照未填充数据的索引重新排列查询层
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            # 如果查询长度为1，生成一个序列长度为批次大小的序列，用于查询层的处理
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个memcpy操作，这样做效率很低。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 对于其他查询长度，假设存在左填充情况，根据注意力掩码和查询层进行未填充处理
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的查询层、键层、值层以及相关的索引和序列长度信息
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
OPT_ATTENTION_CLASSES = {
    "eager": OPTAttention,  # 定义了不同的注意力类，根据配置选择不同的实现方式
    "flash_attention_2": OptFlashAttention2,
}


class OPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size  # 设置嵌入维度为隐藏大小

        self.self_attn = OPT_ATTENTION_CLASSES[config._attn_implementation](config=config, is_decoder=True)
        # 初始化自注意力层，根据配置选择相应的注意力实现类

        self.do_layer_norm_before = config.do_layer_norm_before  # 标志是否在层归一化之前执行
        self.dropout = config.dropout  # 设置dropout比率
        self.activation_fn = ACT2FN[config.activation_function]  # 获取激活函数

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        # 自注意力层的归一化层

        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        # 第一个全连接层，将嵌入维度映射到FFN维度

        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        # 第二个全连接层，将FFN维度映射回嵌入维度

        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)
        # 最终归一化层

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        # 前向传播方法，接收隐藏状态和可选的掩码、层头掩码等参数进行处理
        pass


OPT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OPTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTPreTrainedModel(PreTrainedModel):
    config_class = OPTConfig  # 设置配置类
    base_model_prefix = "model"  # 基础模型前缀
    supports_gradient_checkpointing = True  # 支持梯度检查点
    _no_split_modules = ["OPTDecoderLayer"]  # 不拆分的模块列表
    _supports_flash_attn_2 = True  # 支持闪光注意力2
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        # 从配置中获取初始化的标准差
        std = self.config.init_std
        
        # 如果当前模块是一个线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有偏置项，将偏置项数据初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        
        # 如果当前模块是一个嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果设置了填充索引，将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
# 定义一个空的文档字符串，通常用于类或函数的说明文档
OPT_INPUTS_DOCSTRING = r"""
"""


class OPTDecoder(OPTPreTrainedModel):
    """
    OPT 解码器，由 config.num_hidden_layers 层组成。每一层都是一个 OPTDecoderLayer 对象。

    Args:
        config: OPTConfig
    """

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout  # 初始化 dropout
        self.layerdrop = config.layerdrop  # 初始化层级 dropout
        self.padding_idx = config.pad_token_id  # 初始化填充 token 的索引
        self.max_target_positions = config.max_position_embeddings  # 最大目标位置
        self.vocab_size = config.vocab_size  # 词汇表大小

        # 词嵌入层，将词汇表中的词转换为 word_embed_proj_dim 维度的向量，支持填充 token
        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        # 学习到的位置嵌入
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            # 如果词嵌入维度与隐藏层大小不同，定义一个线性层用于投影输出
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            # 如果词嵌入维度与隐藏层大小不同，定义一个线性层用于投影输入
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # 根据配置初始化最终的层归一化层，用于处理最后输出
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        # 创建包含 config.num_hidden_layers 个 OPTDecoderLayer 对象的层列表
        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 根据配置选择是否使用 Flash Attention 2 实现
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 添加文档字符串的装饰器，指示此方法输出原始隐藏状态，没有特定的输出头
        @add_start_docstrings(
            "The bare OPT Model outputting raw hidden-states without any specific head on top.",
            OPT_START_DOCSTRING,
        )
        class OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        # 调用父类的初始化方法，传入配置参数 config
        super().__init__(config)
        # 创建一个 OPTDecoder 类的实例，并将其赋值给 self.decoder
        self.decoder = OPTDecoder(config)
        # 调用类内部方法 post_init，用于初始化权重和应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回 self.decoder 的 embed_tokens 属性，通常用于获取输入的嵌入表示
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        # 设置 self.decoder 的 embed_tokens 属性为给定的 value
        self.decoder.embed_tokens = value

    def get_decoder(self):
        # 返回 self.decoder 对象，通常用于获取解码器的实例
        return self.decoder

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # 如果 output_attentions 不为 None，则使用其值；否则使用 self.config.output_attentions 的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 不为 None，则使用其值；否则使用 self.config.output_hidden_states 的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 use_cache 不为 None，则使用其值；否则使用 self.config.use_cache 的值
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 如果 return_dict 不为 None，则使用其值；否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 self.decoder 的前向传播方法，传入各种参数，并将结果赋值给 decoder_outputs
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果 return_dict 为 False，则直接返回 decoder_outputs
        if not return_dict:
            return decoder_outputs

        # 如果 return_dict 为 True，则构造一个 BaseModelOutputWithPast 对象并返回
        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )
    class OPTForCausalLM(OPTPreTrainedModel):
        # 定义权重共享的键名列表
        _tied_weights_keys = ["lm_head.weight"]

        def __init__(self, config):
            # 调用父类的初始化方法
            super().__init__(config)
            # 根据配置信息创建OPTModel模型
            self.model = OPTModel(config)

            # 初始化 lm_head，将输入维度投影到词汇表大小的线性层，无偏置
            self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

            # 执行后续的初始化操作
            self.post_init()

        def get_input_embeddings(self):
            # 返回模型解码器的嵌入词汇表
            return self.model.decoder.embed_tokens

        def set_input_embeddings(self, value):
            # 设置模型解码器的嵌入词汇表
            self.model.decoder.embed_tokens = value

        def get_output_embeddings(self):
            # 返回 lm_head 作为输出的嵌入层
            return self.lm_head

        def set_output_embeddings(self, new_embeddings):
            # 设置 lm_head 作为输出的新嵌入层
            self.lm_head = new_embeddings

        def set_decoder(self, decoder):
            # 设置模型的解码器
            self.model.decoder = decoder

        def get_decoder(self):
            # 返回模型的解码器
            return self.model.decoder

        @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            # 模型的前向传播函数，接收多种输入参数并返回相应的输出

        def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
        ):
            if past_key_values is not None:
                # 计算过去键值对的长度
                past_length = past_key_values[0][0].shape[2]

                # 某些生成方法已经仅传递最后一个输入 ID
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # 默认情况下保留仅最后一个 ID
                    remove_prefix_length = input_ids.shape[1] - 1

                # 仅保留从 remove_prefix_length 开始的 input_ids
                input_ids = input_ids[:, remove_prefix_length:]

            # 如果传入了 `inputs_embeds`，我们只在第一个生成步骤中使用它们
            if inputs_embeds is not None and past_key_values is None:
                model_inputs = {"inputs_embeds": inputs_embeds}
            else:
                model_inputs = {"input_ids": input_ids}

            # 更新模型输入参数字典
            model_inputs.update(
                {
                    "past_key_values": past_key_values,
                    "use_cache": kwargs.get("use_cache"),
                    "attention_mask": attention_mask,
                }
            )
            return model_inputs

        @staticmethod
    # 定义一个函数 `_reorder_cache`，用于重排序缓存数据 `past_key_values`，以适应新的束搜索索引 `beam_idx`。
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化一个空的重排序后的缓存元组
        reordered_past = ()
        # 遍历 `past_key_values` 中的每一层的过去状态
        for layer_past in past_key_values:
            # 对于每一层的过去状态，根据 `beam_idx` 将状态重新排序并转移到相同的设备上
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重排序后的缓存元组
        return reordered_past
"""
The OPT Model transformer with a sequence classification head on top (linear layer).

[`OPTForSequenceClassification`] uses the last token in order to do the classification, as other causal models
(e.g. GPT-2) do.

Since it does classification on the last token, it requires to know the position of the last token. If a
`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
each row of the batch).
"""
@add_start_docstrings(OPT_START_DOCSTRING)
class OPTForSequenceClassification(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = OPTModel(config)
        self.score = nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
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
        """
        Forward pass of the OPTForSequenceClassification model.
        """
        # Implementation details are encapsulated in the model's architecture.

    def get_input_embeddings(self):
        """
        Retrieve the input embeddings from the model's decoder.
        """
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set new input embeddings for the model's decoder.
        """
        self.model.decoder.embed_tokens = value


"""
The OPT Model transformer with a span classification head on top for extractive question-answering tasks like SQuAD
(a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
"""
@add_start_docstrings(OPT_START_DOCSTRING)
class OPTForQuestionAnswering(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.model = OPTModel(config)
        self.qa_outputs = nn.Linear(config.word_embed_proj_dim, 2)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    # 使用装饰器替换返回文档字符串，指定输出类型为QuestionAnsweringModelOutput，配置类为_CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token IDs，可以为空
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码，可以为空
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码，可以为空
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 过去的键值对，可以为空
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量，可以为空
        start_positions: Optional[torch.LongTensor] = None,  # 起始位置，可以为空
        end_positions: Optional[torch.LongTensor] = None,  # 结束位置，可以为空
        use_cache: Optional[bool] = None,  # 是否使用缓存，可以为空
        output_attentions: Optional[bool] = None,  # 是否输出注意力，可以为空
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可以为空
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果，可以为空
    ):
        # 返回模型输入嵌入的对象
        def get_input_embeddings(self):
            return self.model.decoder.embed_tokens

        # 设置模型输入嵌入的值
        def set_input_embeddings(self, value):
            self.model.decoder.embed_tokens = value
```