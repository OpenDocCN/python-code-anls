# `.\models\gptj\modeling_gptj.py`

```py
# coding=utf-8
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
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
""" PyTorch GPT-J model."""

import warnings
from typing import Optional, Tuple, Union

import torch
import torch.fx
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
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
    is_torch_fx_proxy,
    logging,
)
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_gptj import GPTJConfig

# Check if flash attention v2 is available
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# Get logger instance for this module
logger = logging.get_logger(__name__)

# Constants used for documentation and testing
_CHECKPOINT_FOR_DOC = "hf-internal-testing/tiny-random-gptj"
_REAL_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-j-6B"
_CONFIG_FOR_DOC = "GPTJConfig"

# List of pretrained model archives for GPT-J
GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EleutherAI/gpt-j-6B",
    # See all GPT-J models at https://huggingface.co/models?filter=gptj
]

# Function to get unpad data based on attention mask
# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

# Function to create sinusoidal positions based on number of positions and dimension
def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    # Calculate inverse frequency for sinusoidal function
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64) / dim))
    # Generate sinusoidal inputs
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=torch.int64).float(), inv_freq).float()
    # Concatenate sine and cosine of sinusoidal inputs along dimension 1
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)

# Wrap the following function in TorchFX framework for symbolic tracing
@torch.fx.wrap
def get_embed_positions(embed_positions, position_ids):
    # 将嵌入位置张量转移到与位置 ID 张量相同的设备上，并重复多次以匹配位置 ID 张量的形状
    return embed_positions.to(position_ids.device).repeat(position_ids.shape[0], 1, 1)
# 定义一个函数，用于将输入张量的每个位置的偶数索引位置的数据提取出来
def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, :, :, ::2]  # 提取偶数索引位置的数据
    x2 = x[:, :, :, 1::2]  # 提取奇数索引位置的数据
    x = torch.stack((-x2, x1), dim=-1)  # 将奇偶索引位置的数据组成新的张量，并进行堆叠
    return x.flatten(-2)  # 将最后两个维度展平，即将每对奇偶索引位置的数据合并成单个维度


# 定义一个函数，用于在给定的张量上应用旋转位置编码
def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)  # 在第3维上重复插入sin值，用于奇数索引位置
    cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)  # 在第3维上重复插入cos值，用于偶数索引位置
    return (tensor * cos) + (rotate_every_two(tensor) * sin)  # 应用旋转位置编码公式：tensor * cos + rotate_every_two(tensor) * sin


# 定义一个自注意力模块类，用于处理注意力机制相关的操作
class GPTJAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings
        # 创建一个下三角矩阵作为偏置，用于掩码操作
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)  # 创建一个掩码偏置

        self.attn_dropout = nn.Dropout(config.attn_pdrop)  # 注意力权重的dropout
        self.resid_dropout = nn.Dropout(config.resid_pdrop)  # 残差连接的dropout

        self.is_causal = True  # 是否是因果关系（用于自回归模型）

        self.embed_dim = config.hidden_size  # 嵌入维度大小
        self.num_attention_heads = config.num_attention_heads  # 注意力头的数量
        self.head_dim = self.embed_dim // self.num_attention_heads  # 每个注意力头的维度
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

        # 定义四个线性映射层，分别用于计算查询、键、值和输出
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.rotary_dim = config.rotary_dim  # 旋转位置编码的维度
        pos_embd_dim = self.rotary_dim or self.embed_dim  # 位置编码的维度，默认为嵌入维度
        self.embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)  # 创建正弦位置编码
       
    # 将输入张量进行分头处理
    def _split_heads(self, tensor, num_attention_heads, attn_head_size, rotary):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)  # 重塑张量的形状，分成多个头
        tensor = tensor.view(new_shape)
        if rotary:
            return tensor  # 如果使用旋转位置编码，直接返回分头后的张量
        if len(tensor.shape) == 5:
            return tensor.permute(0, 1, 3, 2, 4)  # 调整维度顺序，适用于5维张量
        elif len(tensor.shape) == 4:
            return tensor.permute(0, 2, 1, 3)  # 调整维度顺序，适用于4维张量
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")  # 抛出维度错误异常
    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # 如果输入张量维度为5，则交换维度顺序使得 attn_head_size 和 num_attention_heads 维度合并到隐藏层维度
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        # 如果输入张量维度为4，则交换维度顺序使得 attn_head_size 和 num_attention_heads 维度合并到隐藏层维度
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            # 抛出异常，如果张量维度既不是4也不是5
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        # 计算新的张量形状，将 attn_head_size 和 num_attention_heads 合并到最后两个维度
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        # 从 causal_mask buffer 计算 causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        # 将 query 和 key 张量类型转换为 float32，以避免溢出问题
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        # 计算注意力权重
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        # 设置 mask_value 为最小的浮点数，与 attn_weights 张量相同类型
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        # 根据 causal_mask，将不需要的位置设置为 mask_value
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        # 根据缩放因子缩放注意力权重
        attn_weights = attn_weights / self.scale_attn

        if attention_mask is not None:
            # 应用额外的注意力掩码
            attn_weights = attn_weights + attention_mask

        # 使用 softmax 计算最终的注意力权重
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # 将注意力权重转换为与 value 张量相同的数据类型
        attn_weights = attn_weights.to(value.dtype)
        # 应用注意力 dropout
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            # 如果需要，对注意力权重进行头部掩码操作
            attn_weights = attn_weights * head_mask

        # 计算最终的注意力输出
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _get_embed_positions(self, position_ids):
        embed_positions = self.embed_positions
        # 如果 embed_positions 的设备与 position_ids 不同，则将其移到 position_ids 的设备上
        if embed_positions.device != position_ids.device:
            embed_positions = embed_positions.to(position_ids.device)
            self.embed_positions = embed_positions
        # 将 embed_positions 扩展到与 position_ids 的第一个维度相同
        return embed_positions.repeat(position_ids.shape[0], 1, 1)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:
        # 对隐藏状态进行投影以获得查询、键和值
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # 将查询、键、值按注意力头数和头维度拆分
        query = self._split_heads(query, self.num_attention_heads, self.head_dim, True)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, True)
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, False)

        if is_torch_fx_proxy(position_ids) or torch.jit.is_tracing():
            # 在 torch.fx 框架中或正在追踪时，无法跟踪条件复制到 GPU 的逻辑，因此每次在 torch.fx 框架中执行此操作
            embed_positions = get_embed_positions(self.embed_positions, position_ids)
        else:
            # 获取嵌入位置
            embed_positions = self._get_embed_positions(position_ids)

        # 重复位置ID以匹配嵌入位置的形状
        repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        sincos = torch.gather(embed_positions, 1, repeated_position_ids)
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

        if self.rotary_dim is not None:
            # 如果存在旋转维度，则将键和查询分为旋转部分和传递部分，并应用旋转位置编码
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
            q_rot = apply_rotary_pos_emb(q_rot, sin, cos)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            # 否则，直接应用旋转位置编码到键和查询
            key = apply_rotary_pos_emb(key, sin, cos)
            query = apply_rotary_pos_emb(query, sin, cos)

        # 将键和查询的维度进行转置
        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        if layer_past is not None:
            # 如果存在过去的层状态，则将过去的键和值与当前的键和值连接起来
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            # 如果使用缓存，则返回带有浮点数类型的键和值，参考自 GitHub 上的实现
            present = (key.to(hidden_states.dtype), value)
        else:
            # 否则，不返回任何状态
            present = None

        # 计算自注意力输出和注意力权重
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 合并注意力头并进行输出投影
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            # 如果需要输出注意力权重，则添加到输出中
            outputs += (attn_weights,)

        return outputs  # 返回注意力输出、状态以及（如果需要）注意力权重
class GPTJFlashAttention2(GPTJAttention):
    """
    GPTJ flash attention module. This module inherits from `GPTJAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        # Flag to determine if flash attention uses top-left aligned mask
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:
        """
        Forward pass of the GPTJFlashAttention2 module.
        
        Args:
        - hidden_states: Input tensor of shape (batch_size, seq_length, hidden_size).
        - layer_past: Tuple of past key-value states.
        - attention_mask: Optional tensor with attention mask of shape (batch_size, seq_length).
        - position_ids: Optional tensor with position ids of shape (batch_size, seq_length).
        - head_mask: Optional tensor with mask for attention heads of shape (num_heads,).
        - use_cache: Optional boolean flag indicating whether to use caching.
        - output_attentions: Optional boolean flag indicating whether to output attention weights.
        
        Returns:
        - Tuple of output tensor and updated layer past.
        """
        # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
        def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
        ):
            """
            Internal function to perform forward pass of flash attention.

            Args:
            - query_states: Query tensor of shape (batch_size, query_length, hidden_size).
            - key_states: Key tensor of shape (batch_size, key_length, hidden_size).
            - value_states: Value tensor of shape (batch_size, key_length, hidden_size).
            - attention_mask: Attention mask tensor of shape (batch_size, query_length, key_length).
            - query_length: Length of the query sequence.
            - dropout: Optional dropout rate.
            - softmax_scale: Optional scaling factor for softmax.

            Returns:
            - Tuple of output tensor and updated attention weights.
            """
            # Implementation of flash attention forward pass
            pass
    ):
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
        # Determine if causal masking is needed based on configuration
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal  # Set causal to self.is_causal if not using top-left mask
        else:
            # Special case for RoCm compatibility: adjust causal based on query length
            # TODO: Remove this check after upgrading Flash Attention to version 2.1
            causal = self.is_causal and query_length != 1

        # Check if there are padding tokens in the input sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]  # Get the batch size from query states
            # Unpad the input sequences based on the attention mask
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            # Extract sequence lengths after unpadding
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # Perform variable-length Flash Attention computation
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

            # Pad the attention output back to the original sequence length
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # If no padding mask is provided, perform standard Flash Attention
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input with num_heads->num_attention_heads
    # 定义一个方法 `_upad_input`，接收以下参数：query_layer, key_layer, value_layer, attention_mask, query_length
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 调用 `_get_unpad_data` 方法获取解压后的数据的索引、cu_seqlens_k 和 max_seqlen_in_batch_k
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        
        # 获取 key_layer 的形状信息
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 将 key_layer 重塑成适合索引操作的形状，并按照 indices_k 进行索引
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        
        # 将 value_layer 重塑成适合索引操作的形状，并按照 indices_k 进行索引
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        
        # 如果 query_length 等于 kv_seq_len，则将 query_layer 重塑成适合索引操作的形状，并按照 indices_k 进行索引
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_attention_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        # 如果 query_length 等于 1，则设置 max_seqlen_in_batch_q 为 1，cu_seqlens_q 为一个序列，然后将 query_layer 压缩
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 对 attention_mask 进行切片，保留最后 query_length 列，然后调用 unpad_input 函数解压输入
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回更新后的 query_layer, key_layer, value_layer, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_in_batch_q, max_seqlen_in_batch_k)
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
GPTJ_ATTENTION_CLASSES = {
    "eager": GPTJAttention,
    "flash_attention_2": GPTJFlashAttention2,
}

class GPTJMLP(nn.Module):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()
        embed_dim = config.n_embd

        # 初始化输入层和输出层的线性变换
        self.fc_in = nn.Linear(embed_dim, intermediate_size)
        self.fc_out = nn.Linear(intermediate_size, embed_dim)

        # 选择激活函数
        self.act = ACT2FN[config.activation_function]
        # 添加dropout层，以减少过拟合风险
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[torch.FloatTensor]) -> torch.FloatTensor:
        # 输入数据进行线性变换和激活函数处理
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        # 对输出结果进行dropout处理
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPTJBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化层归一化和注意力机制类
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPTJ_ATTENTION_CLASSES[config._attn_implementation](config)
        self.mlp = GPTJMLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        # 残差连接
        residual = hidden_states
        # 应用层归一化
        hidden_states = self.ln_1(hidden_states)
        # 执行注意力机制
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 获取注意力机制的输出
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        # 执行MLP前向传播
        feed_forward_hidden_states = self.mlp(hidden_states)
        # 结合注意力机制输出、MLP输出和残差连接结果
        hidden_states = attn_output + feed_forward_hidden_states + residual

        # 根据使用缓存选项决定输出
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)


class GPTJPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置模型的类
    config_class = GPTJConfig
    # 基础模型的前缀
    base_model_prefix = "transformer"
    # 模型支持并行处理
    is_parallelizable = True
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要分割的模块列表
    _no_split_modules = ["GPTJBlock"]
    # 定义类变量，用于指定在设备放置时跳过的键名
    _skip_keys_device_placement = "past_key_values"
    # 定义类变量，表示是否支持闪存注意力机制2
    _supports_flash_attn_2 = True
    
    # 初始化方法，接受任意位置参数和关键字参数，并调用父类的初始化方法
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
    
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        """Initialize the weights."""
        # 如果模块是线性层
        if isinstance(module, (nn.Linear,)):
            # 使用正态分布初始化权重，均值为0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果指定了填充索引，则将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是层归一化层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
"""
    This string defines the documentation (docstring) for describing the model class `GPTJ`, which is a subclass of
    `torch.nn.Module` in PyTorch. Users should consult the PyTorch documentation for general usage and behavior of
    `torch.nn.Module`.

    Parameters:
        config (`GPTJConfig`): This parameter is expected to be an instance of `GPTJConfig`, which holds all the
            configuration parameters for the model. It does not load the model weights, only the configuration. To
            load the weights, users should refer to the `from_pretrained` method of `PreTrainedModel`.

    Note:
        - The docstring explains the purpose and usage of the `GPTJ` model class.
        - It provides guidance on the `config` parameter and where to load model weights from.
"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获取这些索引。详见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮盖掩码，用于在填充的标记索引上避免进行注意力计算。
            # 掩码值在 `[0, 1]` 范围内：
            # - 1 表示**未被遮盖**的标记，
            # - 0 表示**被遮盖**的标记。
            # [什么是注意力遮盖？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 分段标记索引，用于指示输入的第一部分和第二部分。
            # 索引在 `[0, 1]` 范围内：
            # - 0 对应 *句子 A* 的标记，
            # - 1 对应 *句子 B* 的标记。
            # [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 输入序列中每个标记的位置索引，用于位置嵌入。
            # 索引选择范围是 `[0, config.n_positions - 1]`。
            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_attention_heads,)` or `(n_layer, num_attention_heads)`, *optional*):
            # 自注意力模块中用于屏蔽选定头部的掩码。
            # 掩码值在 `[0, 1]` 范围内：
            # - 1 表示头部**未被屏蔽**，
            # - 0 表示头部**被屏蔽**。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_dim)`, *optional*):
            # 可选参数，可以直接传递嵌入表示而不是 `input_ids`。
            # 如果需要更多控制如何将 *input_ids* 索引转换为相关联的向量，而不是使用模型的内部嵌入查找矩阵，这将非常有用。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。
            # 查看返回张量中的 `attentions` 以获取更多细节。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。
            # 查看返回张量中的 `hidden_states` 以获取更多细节。
        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是一个简单的元组。
# 并行化功能的文档字符串，描述了该功能的实验性质以及如何使用设备映射来分配模型的注意力模块到多个设备上
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice. Uses a device map to distribute
    attention modules of the model across several devices. If no device map is given, it will evenly distribute blocks
    across all devices.

    Args:
        device_map (`Dict[int, list]`, optional, defaults to None):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the GPT-J models have the
            following number of attention modules:

                - gpt-j-6B: 28

    Example:

    ```
    # Here is an example of a device map on a machine with 4 GPUs using gpt-j-6B, which has a total of 28 attention modules:
    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6],
        1: [7, 8, 9, 10, 11, 12, 13],
        2: [14, 15, 16, 17, 18, 19, 20],
        3: [21, 22, 23, 24, 25, 26, 27],
    }
    model.parallelize(device_map)
    ```
"""

# 反并行化功能的文档字符串，描述了将模型从模型并行状态移回 CPU 的过程
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to CPU from a model parallel state.

    Example:

    ```
    # On a 4 GPU machine with gpt-j-6B:
    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    device_map = {
        0: [0, 1, 2, 3, 4, 5, 6],
        1: [7, 8, 9, 10, 11, 12, 13],
        2: [14, 15, 16, 17, 18, 19, 20],
        3: [21, 22, 23, 24, 25, 26, 27],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""

# GPT-J 模型的类定义，继承自 GPTJPreTrainedModel
@add_start_docstrings(
    "The bare GPT-J Model transformer outputting raw hidden-states without any specific head on top.",
    GPTJ_START_DOCSTRING,
)
class GPTJModel(GPTJPreTrainedModel):
    
    # 初始化方法，接受一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化模型的一些基本属性
        self.embed_dim = config.n_embd  # 嵌入维度
        self.vocab_size = config.vocab_size  # 词汇表大小
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)  # 词嵌入模块
        self.drop = nn.Dropout(config.embd_pdrop)  # Dropout 模块
        self.h = nn.ModuleList([GPTJBlock(config) for _ in range(config.n_layer)])  # 多层 GPTJBlock 模块列表
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)  # 最后一层的 LayerNorm 模块

        # 模型并行化相关的属性
        self.model_parallel = False  # 是否启用模型并行化，默认为 False
        self.device_map = None  # 设备映射，默认为 None
        self.gradient_checkpointing = False  # 是否启用梯度检查点，默认为 False

        # 初始化权重并应用最终处理
        self.post_init()

        # 根据配置决定是否使用 flash_attention_2 实现
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    # 使用 PARALLELIZE_DOCSTRING 文档字符串装饰该方法
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    # 使用装饰器添加文档字符串，文档字符串内容来自 DEPARALLELIZE_DOCSTRING
    def deparallelize(self):
        # 发出警告，指出 `deparallelize` 方法即将在 Transformers v5 中移除
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 禁用模型并设置相关属性
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        # 将输入嵌入层（self.wte）移动到 CPU
        self.wte = self.wte.to("cpu")
        # 将所有隐藏层（self.h）移动到 CPU
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        # 将最后一层归一化层（self.ln_f）移动到 CPU
        self.ln_f = self.ln_f.to("cpu")
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        # 返回输入嵌入层（self.wte）
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        # 设置新的输入嵌入层（self.wte）
        self.wte = new_embeddings

    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用装饰器添加文档字符串描述 GPT-J 模型，这是一个在语言建模头部之上的变压器模型
@add_start_docstrings(
    """
    The GPT-J Model transformer with a language modeling head on top.
    """,
    GPTJ_START_DOCSTRING,
)
# 定义 GPTJForCausalLM 类，继承自 GPTJPreTrainedModel
class GPTJForCausalLM(GPTJPreTrainedModel):
    # 定义一个列表，指定需要共享权重的键
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化方法，接收一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建一个 GPTJModel 实例，使用给定的配置
        self.transformer = GPTJModel(config)
        # 创建一个线性层，将 GPTJ 模型的隐藏状态映射到词汇表大小的输出
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # Model parallel
        # 是否开启模型并行计算，默认为 False
        self.model_parallel = False
        # 设备映射，默认为 None
        self.device_map = None

        # 调用 post_init 方法，初始化权重并进行最终处理
        self.post_init()

    # 使用装饰器添加并行化方法的文档字符串描述
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # 发出警告，表明此方法即将在后续版本中移除
        warnings.warn(
            "`GPTJForCausalLM.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'transformer.h.0':"
            " 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        # 如果 device_map 为 None，则使用 get_device_map 方法生成一个设备映射
        # 该映射将模型层分配到不同的 GPU 设备上
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # 断言设备映射的正确性，确保每个模型层都被正确映射到对应设备
        assert_device_map(self.device_map, len(self.transformer.h))
        # 调用 GPTJModel 类的 parallelize 方法，根据设备映射进行模型并行化
        self.transformer.parallelize(self.device_map)
        # 将 lm_head 层移到 transformer 的第一个设备上
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        # 标记模型已经进行了模型并行化
        self.model_parallel = True

    # 使用装饰器添加取消并行化方法的文档字符串描述
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        # 发出警告，表明此方法即将在后续版本中移除
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 调用 GPTJModel 类的 deparallelize 方法，取消模型的并行化
        self.transformer.deparallelize()
        # 将 transformer 和 lm_head 层移回 CPU 上
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        # 标记模型未进行模型并行化
        self.model_parallel = False
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()

    # 返回 lm_head 层，用于获取输出的词嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置 lm_head 层的新词嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # 如果存在 past_key_values 参数，则忽略已经被包含在其中的输入 ID
        if past_key_values:
            # 计算 past_key_values 的长度，即历史输入的数量
            past_length = past_key_values[0][0].shape[2]

            # 检查输入 ID 的长度是否大于历史输入长度
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length  # 移除的前缀长度为历史输入长度
            else:
                # 默认行为：仅保留最后一个输入 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 移除前缀长度对应的部分输入 ID
            input_ids = input_ids[:, remove_prefix_length:]
            # 如果存在 token_type_ids，则也相应地截取
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # 在批量生成时动态创建 position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                # 如果存在 past_key_values，则仅保留对应的 position_ids
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果传入了 inputs_embeds，则仅在第一个生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 更新 model_inputs 字典，包含所有可能的模型输入参数
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        return model_inputs

    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
    )
    # 此方法用于模型的前向传播，接受多个可能的输入参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 根据需要决定是否返回字典类型的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用Transformer模型进行前向传播
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # 如果使用模型并行化，则设置隐藏状态所在的设备
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # 确保在fp16下采样工作正常，并使用fp32计算损失以匹配mesh-tf版本
        # 参考链接: https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        lm_logits = self.lm_head(hidden_states).to(torch.float32)

        loss = None
        if labels is not None:
            # 将标签移动到正确的设备以启用模型并行化
            labels = labels.to(lm_logits.device)
            # 将logits向左移动一位，以便对比预测下一个token
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 展平tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss = loss.to(hidden_states.dtype)

        # 如果不返回字典类型的结果，则组装输出
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回包含过去键值的CausalLMOutputWithPast对象
        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """

        # 返回一个元组，其中每个元素也是一个元组，表示重新排序后的 past_key_values
        return tuple(
            # 对于 past_key_values 中的每个 layer_past，进行如下操作
            tuple(
                # 对于 layer_past 中的每个 past_state，通过 index_select 方法按照 beam_idx 的索引重新排序，
                # 并将结果移到 past_state 的设备上
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            # 遍历整个 past_key_values，对每个 layer_past 执行上述操作
            for layer_past in past_key_values
        )
"""
定义一个 GPT-J 模型，其顶部有一个序列分类头（线性层）。

[`GPTJForSequenceClassification`] 使用最后一个 token 来进行分类，与其他因果模型（如 GPT、GPT-2、GPT-Neo）类似。

由于它在最后一个 token 上进行分类，因此需要知道最后一个 token 的位置。如果配置中定义了 `pad_token_id`，则在每行中找到不是填充 token 的最后一个 token。如果没有定义 `pad_token_id`，则简单地取每个批次行的最后一个值。当传递 `inputs_embeds` 而不是 `input_ids` 时，由于无法猜测填充 token，它也采用相同的方式（取每行批次的最后一个值）。
"""
@add_start_docstrings(
    """
    The GPT-J Model transformer with a sequence classification head on top (linear layer).

    [`GPTJForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT, GPT-2, GPT-Neo) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GPTJ_START_DOCSTRING,
)
class GPTJForSequenceClassification(GPTJPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPTJModel(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="ydshieh/tiny-random-gptj-for-sequence-classification",
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # Forward 方法，接受多种输入参数，用于执行模型的前向推理。
@add_start_docstrings(
    """
    The GPT-J Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    GPTJ_START_DOCSTRING,
)
class GPTJForQuestionAnswering(GPTJPreTrainedModel):
"""
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        # 设置类属性 num_labels，从配置对象中获取标签数量
        self.num_labels = config.num_labels
        # 创建 GPTJModel 实例并将其赋给类属性 transformer，传入配置对象作为参数
        self.transformer = GPTJModel(config)
        # 创建一个线性层 nn.Linear，将隐藏层大小调整为标签数量，赋给类属性 qa_outputs
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Model parallel
        # 设置模型并行为 False
        self.model_parallel = False
        # 设备映射设为 None
        self.device_map = None

        # 调用自定义的初始化方法 post_init
        # 初始化权重并进行最终处理
        self.post_init()

    # 前向传播函数，接受多个可选的张量作为输入
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
    )
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
        ) -> Union[Tuple, QuestionAnsweringModelOutput]:
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
        # 初始化返回字典，若未提供则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用transformer模型处理输入的各种参数，得到输出
        outputs = self.transformer(
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

        # 从transformer模型的输出中取得序列输出
        sequence_output = outputs[0]

        # 使用qa_outputs模型处理序列输出，得到开始和结束的logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果使用多GPU，则扩展维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)
            # 忽略超出模型输入范围的起始/结束位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，忽略指定的索引
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不需要返回字典形式的结果，则返回元组
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 如果需要返回字典形式的结果，则返回QuestionAnsweringModelOutput对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```