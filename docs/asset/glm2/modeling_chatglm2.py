""" PyTorch ChatGLM model. """

import math
import copy
import warnings
import re
import sys

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput

from .configuration_chatglm2 import ChatGLMConfig

# flags required to enable jit fusion kernels

if sys.platform != 'darwin':
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "THUDM/ChatGLM2-6B"
_CONFIG_FOR_DOC = "ChatGLM6BConfig"

CHATGLM_6B_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "THUDM/chatglm2-6b",
    # See all ChatGLM models at https://huggingface.co/models?filter=chatglm
]


def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # KVSize = LayerCount * HeadSize * 2GroupCount
            kv_size = config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            # Emb: [PrefLen, KVSize]
            self.embedding = torch.nn.Embedding(config.pre_seq_len, kv_size)
            # LL1: [KVSize, HidSize]
            # LL2: [HidSize, KVSize]
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(kv_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, kv_size)
            )
        else:
            # Emb: [PrefLen, KVSize]
            self.embedding = torch.nn.Embedding(config.pre_seq_len,
                                                config.num_layers * config.kv_channels * config.multi_query_group_num * 2)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            # 输入 -> Emb -> LL1 -> tanh -> LL2 -> 输出
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            # 输入 -> Emb -> 输出
            past_key_values = self.embedding(prefix)
        return past_key_values


def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )


@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).to(input_dtype)


class CoreAttention(torch.nn.Module):
    def __init__(self, config: ChatGLMConfig, layer_number):
        super(CoreAttention, self).__init__()

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        pytorch_major_version = int(torch.__version__.split('.')[0])
        if pytorch_major_version >= 2:
            query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
            if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                 is_causal=True)
            else:
                if attention_mask is not None:
                    attention_mask = ~attention_mask
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                 attention_mask)
            context_layer = context_layer.permute(2, 0, 1, 3)
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.reshape(*new_context_layer_shape)
        else:
            # Raw attention scores

            # [b, np, sq, sk]
            output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

            # [sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

            # preallocting input tensor: [b * np, sq, sk]
            matmul_input_buffer = torch.empty(
                output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype,
                device=query_layer.device
            )

            # Raw attention scores. [b * np, sq, sk]
            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / self.norm_factor),
            )

            # change view to [b, np, sq, sk]
            attention_scores = matmul_result.view(*output_size)

            # ===========================
            # Attention probs and dropout
            # ===========================

            # attention scores and attention mask [b, np, sq, sk]
            if self.attention_softmax_in_fp32:
                attention_scores = attention_scores.float()
            if self.coeff is not None:
                attention_scores = attention_scores * self.coeff
            if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
                attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                            device=attention_scores.device, dtype=torch.bool)
                attention_mask.tril_()
                attention_mask = ~attention_mask
            if attention_mask is not None:
                attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = attention_probs.type_as(value_layer)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.attention_dropout(attention_probs)
            # =========================
            # Context layer. [sq, b, hp]
            # =========================

            # value_layer -> context layer.
            # [sk, b, np, hn] --> [b, np, sq, hn]

            # context layer shape: [b, np, sq, hn]
            output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))
            # change view [sk, b * np, hn]
            value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
            # change view [b * np, sq, sk]
            attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
            # matmul: [b * np, sq, hn]
            context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
            # change view [b, np, sq, hn]
            context_layer = context_layer.view(*output_size)
            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)
        # PS = HeadSize * HC
        # 注意这个 PS 并不是嵌入向量的大小，每层输入经过这个投影来压缩，所以叫投影大小
        self.projection_size = config.kv_channels * config.num_attention_heads

        # `hidden_size_per_attention_head`其实就是上面的`kv_channels`，统一记作 HeadSize
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        # 如果不启用 MQA，QKVS 是 QKV 连起来的最后一维大小，所以等于 3PS
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            # 如果启用了 MQA，QKVS = PS + 2 * HeadSize * GroupCount
            # 也就是把 QK 的 HC 换成了 GroupCount
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        # LLQKV 的权重 Wqkv，尺寸为 [HidSize, QKVS]，实际上是 Wq、Wk、Wv 按最后一维连起来
        self.query_key_value = nn.Linear(config.hidden_size, self.qkv_hidden_size,
                                         bias=config.add_bias_linear or config.add_qkv_bias,
                                         device=device, **_config_to_kwargs(config)
                                         )

        self.core_attention = CoreAttention(config, self.layer_number)

        # LLO，权重为 Wo，尺寸 [PS, HidSize]，用于乘上核心注意力的输出
        self.dense = nn.Linear(self.projection_size, config.hidden_size, bias=config.add_bias_linear,
                               device=device, **_config_to_kwargs(config)
                               )

    def _allocate_memory(self, inference_max_sequence_len, batch_size, device=None, dtype=None):
        if self.multi_query_attention:
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            num_attention_heads = self.num_attention_heads_per_partition
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
    ):
        # `hidden_states`尺寸为 [SeqLen, BatchSize, HidSize]
        # 话说一般 BatchSize 都是数据集第一维，这样好不习惯


        # 将输入 X 传给 LLQKV，得到 QKV 的连接，尺寸为 [SeqLen, BatchSize, QKVS]
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            # 如果启用了 MQA，那么 QKVS = PS + 2 * HeadSize * GroupCount
            # 沿最后一维拆出 Q、K、V
            # Q 的尺寸是 [SeqLen, BatchSize, PS]
            # K 和 V 都是 [SeqLen, BatchSize, HeadSize * GroupCount]
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            # 将每个头的 Q、K、V 拆出来
            # Q 转型为 [SeqLen, BatchSize, HC, HeadSize]
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            # K 和 V 转型为 [SeqLen, BatchSize, GroupCount, HeadSize]
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            # 如果没有启用 MQA，那么 QKVS = 3 * PS
            # 把 QKV 转型成 [SeqLen, BatchSize, HC, 3HS]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # 沿最后一维等分三份，将 Q、K、V 拆出来，每个尺寸都是 [SeqLen, BatchSize, HC, HeadSize]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # 应用位置编码 RPE
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # 将 KV 缓存添加到 KV 的前面（也就是单词那一维
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=0)
            value_layer = torch.cat((cache_v, value_layer), dim=0)
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention:
            # [SeqLen, BatchSize, GroupCount, HeadSize] => [SeqLen, BatchSize, GroupCount, 1, HeadSize]
            key_layer = key_layer.unsqueeze(-2)
            # GS = HC // GroupCount，每个组的头部数量
            # [SeqLen, BatchSize, GroupCount, 1 => GS, HeadSize]
            # 注意每个组的所有头的 K 和 V 都是共享的
            key_layer = key_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            # [SeqLen, BatchSize, HC, HeadSize]
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            # 下同
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )

        # 将 Q K V 传给核心注意力层，输出尺寸为 [SeqLen, BatchSize, PS]
        context_layer = self.core_attention(query_layer, key_layer, value_layer, attention_mask)

        # 核心注意力层的输出经过 LLO，得到最终输出，尺寸为 [SeqLen, BatchSize, HidSize]
        output = self.dense(context_layer)

        return output, kv_cache


def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    return common_kwargs


class MLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: ChatGLMConfig, device=None):
        super(MLP, self).__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        # LL1，最后一维 HidSize => 4ES
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

        # LL2，最后一维 4ES => HidSize
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

    def forward(self, hidden_states):
        # 输入 -> LL1 -> swiglu -> LL2 -> 输出
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        return output

# GLM 块包括注意力层、FFN层和之间的残差
class GLMBlock(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(GLMBlock, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection
        # 判断使用 RMS 还是 LN
        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # LN1
        self.input_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                             dtype=config.torch_dtype)

        # 注意力层
        self.self_attention = SelfAttention(config, layer_number, device=device)
        # Dropout
        self.hidden_dropout = config.hidden_dropout

        # LLN2
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                      dtype=config.torch_dtype)

        # FFN
        self.mlp = MLP(config, device=device)

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
    ):
        # hidden_states: [s, b, h]

        # 输入 -> LN1 -> 注意力层 -> ...
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache
        )

        # 判断残差是否在LN1后面
        # 如果为真，那么：
        # 输入 -> LN1 -> 注意力 -> Dropout -> ⊕ -> ...
        #  |                                  ↑
        #  +----------------------------------+
        # 否则：
        # 输入 -> LN1 -> 注意力 -> Dropout -> ⊕ -> ...
        #          |                          ↑
        #          +--------------------------+
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        
        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # ... -> LN2 -> FFN -> ...
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        mlp_output = self.mlp(layernorm_output)

        # 判断残差是否在LN1后面
        # 如果为真，那么：
        # ... -> LN2 -> FFN -> Dropout -> ⊕ -> 输出
        #  |                               ↑
        #  +-------------------------------+
        # 否则：
        # ... -> LN2 -> FFN -> Dropout -> ⊕ -> 输出
        #         |                        ↑
        #         +------------------------+
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        return output, kv_cache

# 编码器模块，包含所有 GLM 块
class GLMTransformer(torch.nn.Module):
    """Transformer class."""

    def __init__(self, config: ChatGLMConfig, device=None):
        super(GLMTransformer, self).__init__()

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # LayerCount
        self.num_layers = config.num_layers

        # TFBlock 层
        def build_layer(layer_number):
            return GLMBlock(config, layer_number, device=device)

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        # 如果最后添加 LN，初始化 LN 层
        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                 dtype=config.torch_dtype)

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):
        # 如果没有提供 KV 缓存，将其初始化为 [None] * LayerCount 保持代码统一
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        # `presents`保存每一层的 KV 的缓存
        presents = () if use_cache else None
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_self_attentions = None
        # `all_hidden_states`保存输入和所有层的输出
        all_hidden_states = () if output_hidden_states else None
        
        # 输入 -> TFBlock1 -> TFBlock2 -> ... TFBLockN -> LN? -> 输出
        for index in range(self.num_layers):
            # 将当前一层的输入存入`all_hidden_states`
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前一层，将输入扔进去，得到输出和 KV 缓存
            layer = self._get_layer(index)
            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache
                )
            else:
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache
                )
            # 将输出作为新的输入
            hidden_states, kv_cache = layer_ret
            # 保存当前一层的 KV 缓存
            if use_cache:
                presents = presents + (kv_cache,)

        # 将最后一层的输出存入`all_hidden_states`
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 将最后一层的输出传给 LN 得到 GLM 输出
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        # 返回 GLM 输出，所有层的 KV 缓存，所有层的输出，以及所有层的注意力矩阵（None）
        return hidden_states, presents, all_hidden_states, all_self_attentions


class ChatGLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    is_parallelizable = False
    supports_gradient_checkpointing = True
    config_class = ChatGLMConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["GLMBlock"]

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        return

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        batch_size, seq_length = input_ids.shape
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device)
        full_attention_mask.tril_()
        past_length = 0
        if past_key_values:
            past_length = past_key_values[0][0].shape[0]
        if past_length:
            full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length,
                                                        device=input_ids.device), full_attention_mask), dim=-1)
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask

    def get_position_ids(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        return position_ids

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GLMTransformer):
            module.gradient_checkpointing = value

# 词嵌入层
class Embedding(torch.nn.Module):
    """Language model embeddings."""

    def __init__(self, config: ChatGLMConfig, device=None):
        super(Embedding, self).__init__()

        self.hidden_size = config.hidden_size
        # 真正的嵌入层 [VocabSize, HidSize]
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size,
            self.hidden_size,
            dtype=config.torch_dtype,
            device=device
        )
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        # 单词 ID 传给嵌入层得到词向量
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # [BatchSize, SeqLen, HidSize] => [SeqLen, BatchSize, HidSize]
        embeddings = embeddings.transpose(0, 1).contiguous()
        # 如果 FP32 标志开启的话，转成 FP32
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings

# 完整的 GLM 模型，包括嵌入层、编码器、输出层
class ChatGLMModel(ChatGLMPreTrainedModel):
    def __init__(self, config: ChatGLMConfig, device=None, empty_init=True):
        super().__init__(config)
        # 如果设置了`empty_init`，创建任何 PyTorch 模块时，不初始化参数
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device
        # 单词嵌入层
        self.embedding = init_method(Embedding, config, **init_kwargs)
        # LayerCount
        self.num_layers = config.num_layers
        # GroupCount
        self.multi_query_group_num = config.multi_query_group_num
        # HeadSize
        self.kv_channels = config.kv_channels

        # SeqLen
        self.seq_length = config.seq_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        # 位置嵌入（PE）
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, original_impl=config.original_rope, device=device,
                                              dtype=config.torch_dtype)
        # GLM 编码器
        self.encoder = init_method(GLMTransformer, config, **init_kwargs)
        # 输出层
        self.output_layer = init_method(nn.Linear, config.hidden_size, config.padded_vocab_size, bias=False,
                                        dtype=config.torch_dtype, **init_kwargs)
        self.pre_seq_len = config.pre_seq_len
        self.prefix_projection = config.prefix_projection
        if self.pre_seq_len is not None:
            # 如果设置了前缀序列长度（PrefLen）
            # 关闭所有参数的自动梯度
            for param in self.parameters():
                param.requires_grad = False
            # [0, 1, ..., PrefLen - 1]
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            # 初始化前缀编码层和 Dropout
            self.prefix_encoder = PrefixEncoder(config)
            self.dropout = torch.nn.Dropout(0.1)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def get_prompt(self, batch_size, device, dtype=torch.half):
        # prefix_tokens = [0, 1, ..., PrefLen - 1]
        # [PrefLen] => [1, PrefLen] => [BatchSize, PrefLen]
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        # [BatchSize, PrefLen, KVSize=LayerCount * HeadSize * 2GroupCount]
        past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)
        # [BatchSize, PrefLen, KVSize=LayerCount * HeadSize * 2GroupCount] => [BatchSize, PrefLen, 2LayerCount, GroupCount, HeadSize]
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_layers * 2,
            self.multi_query_group_num,
            self.kv_channels
        )
        
        past_key_values = self.dropout(past_key_values)
        # [BatchSize, PrefLen, 2LayerCount, GroupCount, HeadSize] => [2LayerCount, PrefLen, BatchSize, GroupCount, HeadSize] => LayerCount * [2, PrefLen, BatchSize, GroupCount, HeadSize]
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
        return past_key_values

    def forward(
            self,
            input_ids,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 输入是单词 ID，的形状为 [BatchSize, SeqLen]
        batch_size, seq_length = input_ids.shape
        # 将单词 ID 传递给词嵌入层得到嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        # 如果设置了 PrefLen
        if self.pre_seq_len is not None:
            # 如果没有提供 KV 缓存，初始化为前 PrefLen 个前缀的词嵌入
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)),
                                            attention_mask], dim=-1)

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        # 计算 PE
        # 初始化位置编码层
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        # 如果提供了位置 ID 就是用它检索位置嵌入矩阵
        # 如果没有，就返回嵌入矩阵的前 SeqLen 个向量
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        # [BatchSize, SeqLen, HidSize] => [SeqLen, BatchSize, HidSize]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # 将词嵌入和位置嵌入传给编码器得到编码器输出
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
        )

        # 返回 GLM 输出，每层的 KV 缓存和每层的输出
        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def quantize(self, weight_bit_width: int):
        from .quantization import quantize
        quantize(self.encoder, weight_bit_width)
        return self


class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel):
    def __init__(self, config: ChatGLMConfig, empty_init=True, device=None):
        super().__init__(config)

        self.max_sequence_length = config.max_length
        self.transformer = ChatGLMModel(config, empty_init=empty_init, device=device)
        self.config = config
        self.quantized = False

        if self.config.quantization_bit:
            self.quantize(self.config.quantization_bit, empty_init=True)

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        model_kwargs["is_first_forward"] = False
        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            is_first_forward: bool = True,
            **kwargs
    ) -> dict:
        # only last token for input_ids if past is not None
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids, device=input_ids.device)
        if not is_first_forward:
            position_ids = position_ids[..., -1:]
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "return_last_logit": True
        }

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        # `return_last_logit`表示只保留最后一个单词的
        if return_last_logit:
            hidden_states = hidden_states[-1:]
        # 将编码器输出传入输出层得到单词概率
        lm_logits = self.transformer.output_layer(hidden_states)
        # [SeqLen, BatchSize, ...] => [BatchSize, SeqLen, ...]
        lm_logits = lm_logits.transpose(0, 1).contiguous()

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # 让第 i 个词前面的单词预测第 i 个词
            # 假如原文是 [A, B, C, D, E]
            # logits = [A, B, C, D]，labels = [B, C, D, E]
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 单词 Logits 变形为 [BatchSize * (SeqLen - 1), VocabSize]
            # 标签变形为 [BatchSize * (SeqLen - 1)]
            # 计算交叉熵
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        # 返回损失、单词 Logits、KV 缓存、编码器输出、以及编码器注意力矩阵
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(1, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(1, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )

    def process_response(self, response):
        #  裁剪空白，替换训练时间
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        return response

    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        '''
        将历史问答和当前提问组装成整个输入
        In [1]: tokenizer.build_prompt('Q3', [('Q1', 'A1'),('Q2', 'A2')])
        Out[1]: '[Round 1]\n\n问：Q1\n\n答：A1\n\n[Round 2]\n\n问：Q2\n\n答：A2\n\n[Round 3]\n\n问：Q3\n\n答：'
        '''
        prompt = tokenizer.build_prompt(query, history=history)
        '''
        整个提问传给分词器得到单词ID
        In [2]: tokenizer(['你好'], return_tensors="pt")
        Out[2]: {
           'input_ids': tensor([[64790, 64792, 36474, 54591]]), 
           'attention_mask': tensor([[1, 1, 1, 1]]), 
           'position_ids': tensor([[0, 1, 2, 3]])
        }
        '''
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    def build_stream_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        # PKV 不为空的时候调用这个函数，使用当前问题构建输入
        if history:
            # 历史不为空，只使用最后一轮的提问构建输入
            # 为了和之前的问答历史衔接，需要添加换行符
            # query = '你好', prompt = "\n\n[Round x]\n\n问：你好\n\n答："
            prompt = "\n\n[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            '''
            将 prompt 转成单词 ID，去掉开头的 ID64790、ID64792
            In [147]: tokenizer.encode('\n\n你好', add_special_tokens=False)
            Out[147]: [30910, 13, 13, 39701]
            In [149]: tokenizer.encode('\n\n你好')
            Out[149]: [64790, 64792, 30910, 13, 13, 39701]
            '''
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            # 去掉开头的 ID30910 
            input_ids = input_ids[1:]
            '''
            为 input_ids 生成相应的 attention_mask 和 position_ids
            In [151]: tokenizer.batch_encode_plus(
                [([13,13,39701], None)], 
                return_tensors="pt", 
                add_special_tokens=False
            )
            Out[151]: {
                'input_ids': tensor([[   13,    13, 39701]]), 
                'attention_mask': tensor([[1, 1, 1]]), 
                'position_ids': tensor([[0, 1, 2]])
            }
            '''
            inputs = tokenizer.batch_encode_plus([(input_ids, None)], return_tensors="pt", add_special_tokens=False)
        else:
            # 历史为空，仅仅使用第一轮的提问构建输入
            prompt = "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs


    '''
    In [1]: q = '你好'

    In [2]: r, his = model.chat(tokenizer, q)

    In [3]: r
    Out[3]: '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'

    In [4]: his
    Out[4]: [('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。')]
    
    In [5]: q = '你可以做什么？'

    In [6]: r, his = model.chat(tokenizer, q, his)

    In [7]: r
    Out[7]: '我是一个大型语言模型，可以进行自然语言处理和生成。具体来说，我可以：\n\n1.  回答问题：像人类一样回答您的问题，或者提供 相关信息。\n\n2.  提供建议：根据您的问题提供一些建议，或者提供一些参考信息。\n\n3.  进行翻译：将一种语言翻译成另一种语言，或者将一种语言的文本翻译成另一种语言的文本。\n\n4.  生成文本：根据您的问题生成一些文本，比如文章、故事、新闻报道等。\n\n5.  自动文本摘要：自动概括文本的内容，并生成摘要。\n\n6.  情感分析：判断文本中情感的程度，并返回相应的情感信息。\n\n7.  智能对话：进行智能对话，与人类交流并完成任务。\n\n请注意，我是一个机器，我的回答可能不够准确，也可能会有所误导。'

    In [8]: his
    Out[8]:
    [('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'),
     ('你可以做什么？',
      '我是一个大型语言模型，可以进行自然语言处理和生成。具体来说，我可以：\n\n1.  回答问题：像人类一样回答您的问题，或者提供相关信息 。\n\n2.  提供建议：根据您的问题提供一些建议，或者提供一些参考信息。\n\n3.  进行翻译：将一种语言翻译成另一种语言，或者将一种语言的文本翻译成另一种语言的文本。\n\n4.  生成文本：根据您的问题生成一些文本，比如文章、故事、新闻报道等。\n\n5.  自动文本摘要：自动概括文本的内容，并生成摘要。\n\n6.  情感分析：判断文本中情感的程度，并返回相应的情感信息。\n\n7.  智能对话：进行智能对话，与人类交流并完成任务。\n\n请注意，我是一个机器，我的回答可能不够准确，也可能会有所误导。')]
    '''

    @torch.inference_mode()
    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 8192, num_beams=1,
             do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        # 组织模型配置项
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        # 将历史问答和当前提问组成整个提问，然后传给分词器得到单词ID
        inputs = self.build_inputs(tokenizer, query, history=history)
        # 提问的单词 ID 输入模型得到回答的单词概率
        outputs = self.generate(**inputs, **gen_kwargs)
        # 取第一个回答，并截断回答中的提问部分
        '''
        prompt: '你好, output: tensor([[64790, 64792,   790, 30951,   517, 30910, 30939, 30996,    13,    13,
         54761, 31211, 39701,    13,    13, 55437, 31211, 36474, 54591,   243,
           162,   148,   142, 31404, 33030, 34797, 42481, 22011, 10461, 30944,
         30943, 30941, 30978, 30949, 31123, 48895, 35214, 54622, 31123, 32616,
         39905, 31901, 31639, 31155,     2]], device='cuda:0')
        tokenizer.decode(output[0]): '[Round 1]\n\n问：你好\n\n答： 你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'
        '''
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        # 单词概率解码得到单词
        response = tokenizer.decode(outputs)
        # 裁剪空白，替换训练时间
        response = self.process_response(response)
        # 记录历史问答
        history = history + [(query, response)]
        return response, history

    '''
    In [133]: q = '你好'

    In [134]: it = model.stream_chat(tokenizer, q)

    In [135]: for r, his in it: print(r); print(his)
    你
    [('你好', '你')]
    你好
    [('你好', '你好')]
    你好👋
    [('你好', '你好👋')]
    ...
    你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题
    [('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题')]
    你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
    [('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。')]
    你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。
    [('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。')]

    In [136]: q = '你可以做什么？'

    In [137]: it = model.stream_chat(tokenizer, q, his)

    In [138]: for r, his in it: print(r); print(his)
    我
    [('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'), ('你可以做什么？', '我')]
    我是一款
    [('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'), ('你可以做什么？', '我是一款')]
    我是一款大型
    [('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'), ('你可以做什么？', '我是一款大型')]
    ...
    我是一款大型语言模型，可以进行自然语言处理和生成，以及提供各种服务和咨询。我的目标是帮助人们更方便、高效地获取信息、解决问题和交流沟通
    [('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'), ('你可以做什么？', '我是一款大型语言模型，可 以进行自然语言处理和生成，以及提供各种服务和咨询。我的目标是帮助人们更方便、高效地获取信息、解决问题和交流沟通')]
    我是一款大型语言模型，可以进行自然语言处理和生成，以及提供各种服务和咨询。我的目标是帮助人们更方便、高效地获取信息、解决问题和交流沟通。
    [('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'), ('你可以做什么？', '我是一款大型语言模型，可 以进行自然语言处理和生成，以及提供各种服务和咨询。我的目标是帮助人们更方便、高效地获取信息、解决问题和交流沟通。')]
    我是一款大型语言模型，可以进行自然语言处理和生成，以及提供各种服务和咨询。我的目标是帮助人们更方便、高效地获取信息、解决问题和交流沟通。
    [('你好', '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'), ('你可以做什么？', '我是一款大型语言模型，可 以进行自然语言处理和生成，以及提供各种服务和咨询。我的目标是帮助人们更方便、高效地获取信息、解决问题和交流沟通。')]

    '''

    @torch.inference_mode()
    def stream_chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, past_key_values=None,
                    max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
                    return_past_key_values=False, **kwargs):
        # 为历史和 logit 处理器设置默认值
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        if past_key_values is None and not return_past_key_values:
            # 如果 PKV 为空，就需要使用完整的历史对话记录构建模型输入
            inputs = self.build_inputs(tokenizer, query, history=history)
        else:
            # 如果 PKV 不为空，它是历史对话记录的 KV 缓存，
            # 只需要使用当前问题构建模型输入
            inputs = self.build_stream_inputs(tokenizer, query, history=history)
        if past_key_values is not None:
            # 得到之前输入的长度
            past_length = past_key_values[0][0].shape[0]
            # 如果有PSL， 从中减去
            if self.transformer.pre_seq_len is not None:
                past_length -= self.transformer.pre_seq_len
            # 位置 ID 都后移指定长度
            inputs.position_ids += past_length
            # attention_mask 前面添加 PL 个 1
            attention_mask = inputs.attention_mask
            attention_mask = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=1)
            inputs['attention_mask'] = attention_mask
        for outputs in self.stream_generate(**inputs, past_key_values=past_key_values,
                                            return_past_key_values=return_past_key_values, **gen_kwargs):
            if return_past_key_values:
                outputs, past_key_values = outputs
            # 取第一个回答，并截断回答中的提问部分
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
            '''
            q: '你好'
            iter1 response: '你'
            iter2 response: '你好'
            ...
            iterN response: '你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。'
            '''
            response = tokenizer.decode(outputs)
            # 如果回答最后一个字不是终止符
            if response and response[-1] != "�":
                # 处理时间
                response = self.process_response(response)
                # 将问题和当前回答加入历史
                new_history = history + [(query, response)]
                if return_past_key_values:
                    yield response, new_history, past_key_values
                else:
                    yield response, new_history

    @torch.inference_mode()
    def stream_generate(
            self,
            input_ids,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            return_past_key_values=False,
            **kwargs,
    ):
        batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            )
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            if not has_default_max_length:
                logger.warn(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                    UserWarning,
                )

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        logits_warper = self._get_logits_warper(generation_config)

        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        scores = None
        while True:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            if generation_config.do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())
            if return_past_key_values:
                yield input_ids, outputs.past_key_values
            else:
                yield input_ids
            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

    def quantize(self, bits: int, empty_init=False, device=None, **kwargs):
        if bits == 0:
            return

        from .quantization import quantize

        if self.quantized:
            logger.info("Already quantized.")
            return self

        self.quantized = True

        self.config.quantization_bit = bits

        self.transformer.encoder = quantize(self.transformer.encoder, bits, empty_init=empty_init, device=device,
                                            **kwargs)
        return self
