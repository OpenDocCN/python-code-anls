# `.\models\gpt_neox_japanese\modeling_gpt_neox_japanese.py`

```
# coding=utf-8
# Copyright 2022 ABEJA, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch GPTNeoX model."""

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_gpt_neox_japanese import GPTNeoXJapaneseConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "abeja/gpt-neox-japanese-2.7b"
_CONFIG_FOR_DOC = "GPTNeoXJapaneseConfig"

GPT_NEOX_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST = {
    "https://huggingface.co/abeja/gpt-neox-japanese-2.7b/resolve/main/config.json",
    # See all GPTNeoXJapanese models at https://huggingface.co/models?filter=gpt_neox_japanese
}


class GPTNeoXJapanesePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTNeoXJapaneseConfig
    base_model_prefix = "gpt_neox_japanese"
    _no_split_modules = ["GPTNeoXJapaneseLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是线性层，则使用正态分布初始化权重和零初始化偏置
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层，则使用正态分布初始化权重，对于指定的填充索引，将其权重初始化为零
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是层归一化层，则初始化偏置为零，权重为全1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class GPTNeoXJapaneseAttention(nn.Module):
    # 初始化函数，用于初始化一个注意力机制的模型
    def __init__(self, config, use_bias=False):
        super().__init__()
        # 设置注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 设置隐藏层的大小
        self.hidden_size = config.hidden_size
        # 计算每个注意力头的大小
        self.head_size = self.hidden_size // self.num_attention_heads

        # 计算旋转嵌入的维度，基于头大小和配置中的旋转百分比
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        # 创建旋转嵌入对象，用于位置编码
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims, config.max_position_embeddings, base=config.rotary_emb_base
        )
        # 设置最大位置编码数
        self.max_positions = config.max_position_embeddings
        # 设置注意力的dropout层
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        # 计算归一化因子，用于注意力计算中
        self.norm_factor = torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype())

        # 创建查询、键、值的线性层
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        # 创建输出密集层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # 如果是最后一层，则激活偏置项
        self.use_bias = use_bias
        # 如果使用偏置，则创建偏置参数
        self.dense_bias = nn.Parameter(torch.zeros(config.hidden_size)) if use_bias else None

    # 前向传播函数，用于计算模型的输出
    def forward(
        self,
        hidden_states,
        attention_mask,
        head_mask=None,
        layer_past=None,
        use_cache=False,
        output_attentions=False,
        ):
            # 检查是否存在先前的层信息，并且该信息的元素数大于0
            has_layer_past = layer_past is not None and layer_past[0].numel() > 0

            # 计算 QKV
            # 注意力头 [batch, seq_len, hidden_size]
            #   --> [batch, seq_len, (np * 3 * head_size)]
            qkv = self.query_key_value(hidden_states)

            # [batch, seq_len, (num_heads * 3 * head_size)]
            #   --> [batch, seq_len, num_heads, 3 * head_size]
            new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
            qkv = qkv.view(*new_qkv_shape)

            # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
            query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
            key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
            value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

            # 计算旋转嵌入在旋转维度上的应用
            query_rot = query[..., : self.rotary_ndims]
            query_pass = query[..., self.rotary_ndims :]
            key_rot = key[..., : self.rotary_ndims]
            key_pass = key[..., self.rotary_ndims :]

            # 计算旋转嵌入的令牌偏移量（在解码时）
            seq_len = key.shape[-2]
            offset = 0
            if has_layer_past:
                offset = layer_past[0].shape[-2]
                seq_len += offset
            cos, sin = self.rotary_emb(value, seq_len=seq_len)
            query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, offset=offset)
            query = torch.cat((query, query_pass), dim=-1)
            key = torch.cat((key, key_pass), dim=-1)

            # 缓存 QKV 值
            if has_layer_past:
                past_key = layer_past[0]
                past_value = layer_past[1]
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)
            present = (key, value) if use_cache else None

            # 计算注意力
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

            # 重塑输出
            attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
            attn_output = self.dense(attn_output)

            outputs = (attn_output, present)
            if output_attentions:
                outputs += (attn_weights,)

            return outputs, self.dense_bias

        @classmethod
        def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
            """
            将隐藏维度分割为 attn_head_size 和 num_attention_heads
            """
            # tensor: [bs, seq_len, hidden_size]
            new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
            # -> [bs, seq_len, num_attention_heads, attn_head_size]
            tensor = tensor.view(new_shape)
            # -> [bs, num_attention_heads, seq_len, attn_head_size]
            tensor = tensor.permute(0, 2, 1, 3)
            return tensor
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # 将张量进行维度置换，调整为 [bs, seq_len, num_attention_heads, attn_head_size] 的格式
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # 将多头注意力机制和注意力头尺寸的维度合并成隐藏层维度，得到 [bs, seq_len, hidden_size] 的张量
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # 返回合并后的张量
        return tensor

    def _create_causal_mask(self, key_length, query_length):
        # 创建一个因果遮蔽（causal mask）张量，用于自注意力机制中
        causal_mask = torch.tril(
            torch.ones((self.max_positions, self.max_positions), dtype=torch.bool).view(
                1, 1, self.max_positions, self.max_positions
            )
        )
        # 从因果遮蔽张量中选择出需要的部分，形成 [1, 1, key_length - query_length, key_length] 的子张量
        return causal_mask[:, :, key_length - query_length : key_length, :key_length]
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # query, key, value 的维度说明：[批大小, 注意力头数, 序列长度, 每个注意力头的大小]

        # 获取 query 的维度信息
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        # 获取 key 的序列长度信息
        key_length = key.size(-2)

        # 创建因果遮罩，基于 key 的长度和 query 的长度
        causal_mask = self._create_causal_mask(key_length, query_length)

        # 将 query 和 key 重塑成适合计算的形状
        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)

        # 初始化注意力分数矩阵为零
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )

        # 计算注意力分数
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=(torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor),
        )

        # 将注意力分数重塑回原来的形状
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        # 生成用于掩码的最小值
        mask_value = torch.finfo(attn_scores.dtype).min
        # 将最小值转换为张量，并移到相同的设备上
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)
        causal_mask = causal_mask.to(attn_scores.device)

        # 应用因果遮罩
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        # 如果提供了注意力掩码，则应用它
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # 对注意力分数进行 softmax 归一化
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)

        # 应用注意力 dropout
        attn_weights = self.attention_dropout(attn_weights)

        # 将注意力权重转换为与 value 相同的数据类型
        attn_weights = attn_weights.to(value.dtype)

        # 如果提供了头部掩码，则应用它
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, value)

        # 返回注意力输出和注意力权重
        return attn_output, attn_weights
# 从 transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXRotaryEmbedding 复制的类，现命名为 RotaryEmbedding
class RotaryEmbedding(nn.Module):
    # 从 transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding.__init__ 复制的构造函数
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        # 初始化旋转嵌入层的参数
        self.dim = dim  # 嵌入维度
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入数
        self.base = base  # 基数
        # 计算频率倒数，并在设备上注册为缓冲区
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 为了使 `torch.jit.trace` 正常工作，构建余弦和正弦缓存
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 设置余弦和正弦缓存
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # 根据论文使用的不同排列方式，构建嵌入张量
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len=None):
        # 前向传播函数
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 返回余弦和正弦缓存
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x):
    """将输入的一半隐藏维度旋转"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    """应用旋转位置嵌入到查询和键中"""
    cos = cos[..., offset : q.shape[-2] + offset, :]
    sin = sin[..., offset : q.shape[-2] + offset, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def bias_dropout_add(x: Tensor, bias: Tensor, residual: Optional[Tensor], prob: float, training: bool) -> Tensor:
    """为输入添加偏置，应用 dropout 和残差连接

    Args:
        x (Tensor): 主路径的输出
        bias (Tensor): 最后一个注意力层的 attn_bias 或者 None
        residual (Optional[Tensor]): 残差值
        prob (float): dropout 概率
        training (bool): 是否处于训练模式

    Returns:
        Tensor: dropout(x + bias) + residual
    """
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out


class GPTNeoXJapaneseMLP(nn.Module):
    # 初始化方法，接收一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 根据配置计算中间层的大小，这里使用了隐藏层大小乘以一个倍数来确定中间层的大小
        intermediate_size = int(config.hidden_size * config.intermediate_multiple_size)
        # 创建一个线性层，将隐藏状态映射到四倍隐藏状态大小的维度，不使用偏置
        self.dense_h_to_4h = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        # 创建一个线性层，将四倍隐藏状态大小映射回隐藏状态大小的维度，不使用偏置
        # 这一步是将映射的结果投影回原始隐藏状态的维度
        self.dense_4h_to_h = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        # 从配置中获取激活函数的选择，并赋值给类的实例变量
        self.act = ACT2FN[config.hidden_act]

    # 前向传播方法，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 使用第一个线性层将隐藏状态映射到四倍隐藏状态大小的维度
        intermediate = self.dense_h_to_4h(hidden_states)
        # 对映射结果应用激活函数
        intermediate = self.act(intermediate)
        # 使用第二个线性层将映射后的结果投影回原始隐藏状态大小的维度
        output = self.dense_4h_to_h(intermediate)
        # 返回最终的输出结果
        return output
# 定义一个名为 GPTNeoXJapaneseLayer 的新类，继承自 nn.Module
class GPTNeoXJapaneseLayer(nn.Module):
    # 初始化方法，接受 config 和 layer_number 两个参数
    def __init__(self, config, layer_number):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 将输入的层编号存储到实例变量中
        self.layer_number = layer_number
        # 初始化输入层归一化层，使用 config 中定义的隐藏层大小和层归一化的 epsilon 值
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化注意力后归一化层，使用 config 中定义的隐藏层大小和层归一化的 epsilon 值
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 如果是最后一层，激活注意力层的偏置
        self.attention = GPTNeoXJapaneseAttention(config=config, use_bias=layer_number == config.num_hidden_layers - 1)
        # 初始化多层感知机层
        self.mlp = GPTNeoXJapaneseMLP(config)
        # 隐藏层的 dropout 概率
        self.hidden_dropout = config.hidden_dropout

    # 前向传播方法
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        layer_past=None,
        output_attentions=False,
    ):
        # 将隐藏状态保存为残差连接的基础
        residual = hidden_states
        # 对输入层进行归一化处理
        ln_out = self.input_layernorm(hidden_states)
        # 使用注意力层进行计算，得到注意力层的输出和注意力偏置
        attention_layer_outputs, attn_bias = self.attention(
            ln_out,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 获取注意力层的输出，索引为 0 的元素
        attn_output = attention_layer_outputs[0]  # output_attn: a, present, (attentions)
        # 将其余的输出保存到 outputs 变量中
        outputs = attention_layer_outputs[1:]

        # 使用 bias_dropout_add 函数将注意力输出与偏置和残差相加，并应用 dropout
        attn_output = bias_dropout_add(
            attn_output,
            bias=attn_bias.expand_as(residual) if attn_bias is not None else attn_bias,
            residual=residual,
            prob=self.hidden_dropout,
            training=self.training,
        )
        # 对注意力输出进行 MLP 处理，并应用注意力后归一化
        mlp_output = self.mlp(self.post_attention_layernorm(attn_output))

        # 再次使用 bias_dropout_add 函数将 MLP 输出与残差和注意力输出相加，并应用 dropout
        attn_output = bias_dropout_add(
            mlp_output, bias=None, residual=attn_output, prob=self.hidden_dropout, training=self.training
        )

        # 如果 use_cache 为真，将 attn_output 添加到 outputs 中；否则，只保留 outputs 中的第一个元素
        if use_cache:
            outputs = (attn_output,) + outputs
        else:
            outputs = (attn_output,) + outputs[1:]

        # 返回处理后的输出，包括隐藏状态、present 和（如果设置）注意力信息
        return outputs  # hidden_states, present, (attentions)
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获得这些索引。

        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮罩，用于避免在填充的标记索引上执行注意力操作。
            # 遮罩值选在 `[0, 1]` 之间：
            # - 1 表示**不被遮罩**的标记，
            # - 0 表示**被遮罩**的标记。

        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段标记索引，用于指示输入的第一部分和第二部分。
            # 索引选在 `[0, 1]` 之间：
            # - 0 对应于*句子 A*的标记，
            # - 1 对应于*句子 B*的标记。

        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。
            # 选择范围为 `[0, config.max_position_embeddings - 1]`。

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于屏蔽自注意力模块中选择的头部的遮罩。
            # 遮罩值选在 `[0, 1]` 之间：
            # - 1 表示头部**未被遮罩**，
            # - 0 表示头部**被遮罩**。

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选参数，代替传递 `input_ids`，直接传递嵌入表示。
            # 如果希望更精细地控制如何将 *input_ids* 索引转换为关联向量，比模型内部的嵌入查找矩阵更有用。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详细信息见返回的张量中的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详细信息见返回的张量中的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~file_utils.ModelOutput`] 而不是普通元组。
"""
@add_start_docstrings(
    "The bare GPTNeoXJapaneseForCausalLM Model transformer with a causal language modeling head on top.",
    GPT_NEOX_JAPANESE_START_DOCSTRING,
)
"""
class GPTNeoXJapaneseForCausalLM(GPTNeoXJapanesePreTrainedModel):
    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        """
        Initialize the GPTNeoXJapaneseForCausalLM model.

        Args:
            config (GPTNeoXJapaneseConfig): Configuration class for the model.
        """
        super().__init__(config)
        self.config = config

        # Initialize the base GPTNeoXJapaneseModel
        self.gpt_neox_japanese = GPTNeoXJapaneseModel(config)
        
        # Initialize the output linear layer for language modeling
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Retrieve the output embeddings.

        Returns:
            nn.Linear: The output embeddings layer.
        """
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        """
        Set new output embeddings.

        Args:
            new_embeddings (nn.Linear): New embeddings to be set.
        """
        self.embed_out = new_embeddings

    @add_start_docstrings_to_model_forward(GPT_NEOX_JAPANESE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Perform forward pass of the GPTNeoXJapaneseForCausalLM model.

        Args:
            input_ids (torch.LongTensor, optional): Input token IDs. Default: None
            attention_mask (torch.FloatTensor, optional): Attention mask. Default: None
            head_mask (torch.FloatTensor, optional): Head mask. Default: None
            inputs_embeds (torch.FloatTensor, optional): Embedded inputs. Default: None
            past_key_values (Tuple[Tuple[torch.FloatTensor]], optional): Past key values for autoregressive generation. Default: None
            use_cache (bool, optional): Whether to use cache for autoregressive generation. Default: None
            output_attentions (bool, optional): Whether to output attentions weights. Default: None
            output_hidden_states (bool, optional): Whether to output hidden states. Default: None
            return_dict (bool, optional): Whether to return a dictionary as output. Default: None

        Returns:
            output (CausalLMOutputWithPast): Model output for language modeling.
        """
        # Perform forward pass through the GPTNeoXJapaneseModel
        return self.gpt_neox_japanese(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    # 定义一个方法 `forward`，接收多个输入参数，并返回一个字典或张量
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 定义一个方法 `prepare_inputs_for_generation`，准备用于生成的输入
        def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
            # 获取输入 `input_ids` 的形状
            input_shape = input_ids.shape

            # 如果没有提供注意力掩码，则创建一个全为1的张量作为注意力掩码
            if attention_mask is None:
                attention_mask = input_ids.new_ones(input_shape)

            # 如果使用过去的键值对 `past_key_values`，则截取 `input_ids`
            if past_key_values and past_key_values[0] is not None:
                input_ids = input_ids[:, -1:]

            # 返回一个包含 `input_ids`, `attention_mask`, `past_key_values` 的字典
            return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}

        # 定义一个方法 `_reorder_cache`，重新排序缓存 `past_key_values` 中的内容
        def _reorder_cache(self, past_key_values, beam_idx):
            reordered_past = ()
            # 对每一层的过去键值对执行重新排序
            for layer_past in past_key_values:
                reordered_past += (
                    # 对过去状态进行索引选择，并根据 `beam_idx` 和设备类型调整顺序
                    tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                    # 保持其余部分不变
                    + layer_past[2:],
                )
            # 返回重新排序后的过去键值对
            return reordered_past
```