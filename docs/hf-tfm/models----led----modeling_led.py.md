# `.\models\led\modeling_led.py`

```py
# coding=utf-8
# Copyright 2021 Iz Beltagy, Matthew E. Peters, Arman Cohan and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch LED model."""


import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_led import LEDConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "allenai/led-base-16384"
_CONFIG_FOR_DOC = "LEDConfig"


LED_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "allenai/led-base-16384",
    # See all LED models at https://huggingface.co/models?filter=led
]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    # 创建一个新的张量，形状与输入相同，用于存放右移后的输入ids
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将输入ids除了第一个位置外的所有位置向右移动一位
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # 将第一个位置设置为decoder起始token的id
    shifted_input_ids[:, 0] = decoder_start_token_id

    # 如果pad_token_id为None，抛出值错误异常
    if pad_token_id is None:
        raise ValueError("config.pad_token_id has to be defined.")
    # 将labels中可能存在的-100值替换为pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _prepare_4d_attention_mask_inverted(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    # 获取输入mask的形状信息
    bsz, src_len = mask.size()
    # 如果未指定tgt_len，默认设为src_len
    tgt_len = tgt_len if tgt_len is not None else src_len

    # 将mask从[bsz, seq_len]扩展为[bsz, 1, tgt_seq_len, src_seq_len]
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    # 创建反转的mask，用于全局attention
    inverted_mask = 1.0 - expanded_mask
    # 将反转后的mask中的True值用极小的负数填充，以便后续处理
    expanded_attention_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

    # 确保全局attention_mask为正数
    # 将扩展后的注意力掩码与反转掩码逐元素相乘，实现对应位置的逻辑运算
    expanded_attention_mask = expanded_attention_mask * inverted_mask

    # 返回经过处理后的扩展注意力掩码
    return expanded_attention_mask
class LEDLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        # 获取输入的批量大小（batch size）和序列长度（sequence length）
        bsz, seq_len = input_ids_shape[:2]
        # 根据序列长度和历史键值长度计算出位置信息
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        # 调用父类的forward方法，传入计算得到的位置信息
        return super().forward(positions)


# Copied from transformers.models.longformer.modeling_longformer.LongformerSelfAttention with Longformer->LEDEncoder
class LEDEncoderSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        # 检查隐藏大小是否可以整除注意力头数
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        # 设置注意力头数、每头的维度和嵌入维度
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        # 为查询、键和值设置线性映射层
        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        # 为全局注意力的查询、键和值设置线性映射层
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

        # 设置注意力概率的dropout率
        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        # 检查并设置单向注意力窗口大小
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2

        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        # 此处应该是从Longformer模型复制过来的代码，但未完成，需要补充完整
        pass
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """
        对最后两个维度进行填充和转置操作。

        Args:
            hidden_states_padded (torch.Tensor): 填充后的隐藏状态张量
            padding (tuple): 填充值，实际数值并不重要，因为它将被覆盖

        Returns:
            torch.Tensor: 填充和转置后的隐藏状态张量
        """
        hidden_states_padded = nn.functional.pad(
            hidden_states_padded, padding
        )  # 使用 padding 对 hidden_states_padded 进行填充，填充值并不重要，因为后续会被覆盖
        hidden_states_padded = hidden_states_padded.view(
            *hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2)
        )  # 转置最后两个维度
        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        将每一行向右移动一步，将列转换为对角线。

        Args:
            chunked_hidden_states (torch.Tensor): 分块的隐藏状态张量

        Returns:
            torch.Tensor: 填充和对角化后的隐藏状态张量
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()
        chunked_hidden_states = nn.functional.pad(
            chunked_hidden_states, (0, window_overlap + 1)
        )  # 对 chunked_hidden_states 进行填充，第一个维度不填充，第二个维度填充 window_overlap + 1
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, -1
        )  # 将张量视图重塑为 total_num_heads x num_chunks x (window_overlap*hidden_dim + window_overlap + 1)
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # 截取最后一个维度的部分，得到 total_num_heads x num_chunks x (window_overlap*hidden_dim)
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )  # 将张量重塑为 total_num_heads x num_chunks x window_overlap x (window_overlap + hidden_dim)
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]  # 去除最后一个维度的最后一个元素
        return chunked_hidden_states
    def _chunk(hidden_states, window_overlap, onnx_export: bool = False):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        if not onnx_export:
            # 非 ONNX 导出模式下，创建大小为 2w 的非重叠块
            hidden_states = hidden_states.view(
                hidden_states.size(0),
                torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
                window_overlap * 2,
                hidden_states.size(2),
            )
            # 使用 `as_strided` 实现重叠块，重叠大小为 window_overlap
            chunk_size = list(hidden_states.size())
            chunk_size[1] = chunk_size[1] * 2 - 1

            chunk_stride = list(hidden_states.stride())
            chunk_stride[1] = chunk_stride[1] // 2
            return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)

        # 当导出到 ONNX 时，使用单独的逻辑
        # 因为 ONNX 导出不支持 `as_strided`、`unfold` 和二维张量索引，所以需要使用较慢的实现方法

        # TODO 替换以下代码为
        # > return hidden_states.unfold(dimension=1, size=window_overlap * 2, step=window_overlap).transpose(2, 3)
        # 一旦 `unfold` 得到支持
        # 当 hidden_states.size(1) == window_overlap * 2 时，也可以简单返回 hidden_states.unsqueeze(1)，但这是控制流

        chunk_size = [
            hidden_states.size(0),
            torch.div(hidden_states.size(1), window_overlap, rounding_mode="trunc") - 1,
            window_overlap * 2,
            hidden_states.size(2),
        ]

        # 创建一个与 hidden_states 形状相同的张量用于存储重叠块
        overlapping_chunks = torch.empty(chunk_size, device=hidden_states.device)
        for chunk in range(chunk_size[1]):
            # 将重叠块存储到 overlapping_chunks 中
            overlapping_chunks[:, chunk, :, :] = hidden_states[
                :, chunk * window_overlap : chunk * window_overlap + 2 * window_overlap, :
            ]
        return overlapping_chunks

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
        # 创建一个二维矩阵，用于掩盖无效位置
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        
        # 控制起始部分的输入张量，使其无效位置被掩盖
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
            beginning_input, -float("inf")
        ).where(beginning_mask.bool(), beginning_input)
        
        # 控制结束部分的输入张量，使其无效位置被掩盖
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
            ending_input, -float("inf")
        ).where(ending_mask.bool(), ending_input)
    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        # 获取 value 的形状信息：batch_size 是批大小，seq_len 是序列长度，num_heads 是注意力头数，head_dim 是每个头的维度
        batch_size, seq_len, num_heads, head_dim = value.size()

        # 断言确保 seq_len 能被 2 * window_overlap 整除，以支持滑动窗口的操作
        assert seq_len % (window_overlap * 2) == 0
        # 断言确保 attn_probs 和 value 的前三个维度匹配，即 batch_size、seq_len 和 num_heads
        assert attn_probs.size()[:3] == value.size()[:3]
        # 断言确保 attn_probs 的第四个维度是 2 * window_overlap + 1，与滑动窗口的大小匹配
        assert attn_probs.size(3) == 2 * window_overlap + 1

        # 计算 chunks_count，即分块的数量，每个块大小为 window_overlap
        chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1

        # 将 attn_probs 转置并重塑成新的形状，以便进行滑动窗口操作
        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads,
            torch.div(seq_len, window_overlap, rounding_mode="trunc"),
            window_overlap,
            2 * window_overlap + 1,
        )

        # 将 value 转置并重塑成新的形状，以便进行滑动窗口操作
        value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        # 在序列的开头和结尾各填充 window_overlap 个值，以支持滑动窗口操作
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # 根据滑动窗口的大小和重叠，将 padded_value 进行分块处理
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        # 对 chunked_attn_probs 进行填充和对角化处理
        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        # 使用 Einstein Summation 计算 context 向量，用于最终的输出
        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))

        # 将 context 向量重塑成最终的形状，并进行维度转置，以匹配期望的输出形状
        return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    def _get_global_attn_indices(is_index_global_attn):
        """计算在整个前向传递中需要使用的全局注意力索引"""
        # 计算每个样本中全局注意力索引的数量
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)

        # 批次中全局注意力索引的最大数量
        max_num_global_attn_indices = num_global_attn_indices.max()

        # 获取全局注意力索引的位置
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)

        # 辅助变量，用于标识是否为全局注意力索引
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)

        # 非零值位置的全局注意力索引
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)

        # 非全局注意力索引的零值位置
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)
        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _concat_with_global_key_attn_probs(
        self,
        key_vectors,
        query_vectors,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    ):
        batch_size = key_vectors.shape[0]

        # 创建仅包含全局键向量的张量
        key_vectors_only_global = key_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )

        # 将全局注意力索引对应的键向量复制到新张量中
        key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[is_index_global_attn_nonzero]

        # 使用 Einstein Summation 表示计算注意力概率
        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = torch.einsum("blhd,bshd->blhs", (query_vectors, key_vectors_only_global))

        # 由于 ONNX 导出仅支持连续索引，需要转置操作
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)

        # 将非全局注意力索引位置的值置为一个极小的数，用于遮盖
        attn_probs_from_global_key[
            is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :
        ] = torch.finfo(attn_probs_from_global_key.dtype).min

        # 再次进行转置，使形状与原始张量保持一致
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)

        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(
        self,
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    ):
    ):
        batch_size = attn_probs.shape[0]  # 获取批量大小

        # cut local attn probs to global only
        attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
        # get value vectors for global only
        value_vectors_only_global = value_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )
        # 将全局注意力的位置上的值向量复制到新的张量中
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[is_index_global_attn_nonzero]

        # use `matmul` because `einsum` crashes sometimes with fp16
        # 使用`matmul`进行矩阵乘法计算，因为在fp16模式下，`einsum`有时会崩溃
        attn_output_only_global = torch.matmul(
            attn_probs_only_global.transpose(1, 2).clone(), value_vectors_only_global.transpose(1, 2).clone()
        ).transpose(1, 2)

        # reshape attn probs
        # 重新整形非全局注意力概率
        attn_probs_without_global = attn_probs.narrow(
            -1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices
        ).contiguous()

        # compute attn output with global
        # 使用滑动窗口方法计算包含全局注意力的注意力输出
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors, self.one_sided_attn_window_size
        )
        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(
        self,
        hidden_states,
        max_num_global_attn_indices,
        layer_head_mask,
        is_local_index_global_attn_nonzero,
        is_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
        is_index_masked,
class LEDEncoderAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        # 初始化自注意力机制模块，使用给定的配置和层编号
        self.longformer_self_attn = LEDEncoderSelfAttention(config, layer_id=layer_id)
        # 输出层，线性变换到配置中指定的模型维度
        self.output = nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        is_index_masked: Optional[torch.Tensor] = None,
        is_index_global_attn: Optional[torch.Tensor] = None,
        is_global_attn: Optional[bool] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        
        # 调用长形式自注意力模块进行前向传播
        self_outputs = self.longformer_self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )

        # 将自注意力模块的输出进行线性变换
        attn_output = self.output(self_outputs[0])
        # 组装并返回输出，包括注意力输出（如果有）、额外的注意力权重等
        outputs = (attn_output,) + self_outputs[1:]

        return outputs


class LEDDecoderAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        # 初始化解码器注意力模块，指定嵌入维度、头数、dropout率和是否为解码器
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        # 缩放因子，根据头维度进行设置
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 初始化键、值、查询和输出的线性投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将张量重塑为适合多头注意力机制的形状
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """Forward pass of the decoder attention module."""
        # 省略部分详细的前向传播说明，可根据需要进一步添加
    def __init__(self, config: LEDConfig, layer_id: int):
        super().__init__()
        # 初始化 LED 层的配置
        self.embed_dim = config.d_model
        # 创建自注意力机制对象
        self.self_attn = LEDEncoderAttention(config, layer_id)
        # 创建自注意力层的 LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设定 dropout 概率
        self.dropout = config.dropout
        # 获取激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设定激活函数的 dropout 概率
        self.activation_dropout = config.activation_dropout
        # 第一个全连接层，线性映射到 config.encoder_ffn_dim 维度
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 第二个全连接层，线性映射回 self.embed_dim 维度
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 创建最终的 LayerNorm 层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape *(batch, seq_len, embed_dim)*
            attention_mask (`torch.FloatTensor`): attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                *(encoder_attention_heads,)*.
        """
        # 保存残差连接
        residual = hidden_states
        # 执行自注意力计算
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        # 更新 hidden_states 为自注意力输出
        hidden_states = attn_outputs[0]
        # 应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 执行自注意力层的 LayerNorm
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存残差连接
        residual = hidden_states
        # 应用激活函数并进行第一个全连接层映射
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 应用第二个全连接层映射
        hidden_states = self.fc2(hidden_states)
        # 应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 添加残差连接
        hidden_states = residual + hidden_states
        # 执行最终的 LayerNorm
        hidden_states = self.final_layer_norm(hidden_states)

        # 处理浮点数异常（如果存在）
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        # 返回结果，包括 hidden_states 和可能的注意力输出
        return (hidden_states,) + attn_outputs[1:]
class LEDDecoderLayer(nn.Module):
    def __init__(self, config: LEDConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 初始化 embed_dim 属性为配置中的 d_model

        # 初始化自注意力机制
        self.self_attn = LEDDecoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )

        self.dropout = config.dropout  # 初始化 dropout 属性为配置中的 dropout
        self.activation_fn = ACT2FN[config.activation_function]  # 根据配置选择激活函数
        self.activation_dropout = config.activation_dropout  # 初始化激活函数的 dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 初始化自注意力层的 LayerNorm
        # 初始化编码器注意力机制
        self.encoder_attn = LEDDecoderAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 初始化编码器注意力层的 LayerNorm
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)  # 第二个全连接层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最终输出的 LayerNorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        # 省略具体的前向传播逻辑，用于处理给定的参数和层，并返回结果
        pass


class LEDClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)  # 全连接层
        self.dropout = nn.Dropout(p=pooler_dropout)  # Dropout 层
        self.out_proj = nn.Linear(inner_dim, num_classes)  # 输出分类的全连接层

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)  # 应用 Dropout
        hidden_states = self.dense(hidden_states)  # 全连接层
        hidden_states = torch.tanh(hidden_states)  # Tanh 激活函数
        hidden_states = self.dropout(hidden_states)  # 再次应用 Dropout
        hidden_states = self.out_proj(hidden_states)  # 输出分类的全连接层
        return hidden_states  # 返回分类结果


class LEDPreTrainedModel(PreTrainedModel):
    config_class = LEDConfig  # 配置类为 LEDConfig
    base_model_prefix = "led"  # 基础模型前缀为 "led"
    supports_gradient_checkpointing = True  # 支持梯度检查点

    def _init_weights(self, module):
        std = self.config.init_std  # 从配置中获取初始化标准差
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)  # 初始化线性层的权重
            if module.bias is not None:
                module.bias.data.zero_()  # 初始化偏置为零
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)  # 初始化嵌入层的权重
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()  # 如果有填充索引，将其初始化为零

    @property
    def dummy_property(self):
        # 这是一个虚拟的属性示例，通常用于占位或者模型中的特定设置
        pass
    # 定义一个方法用于生成虚拟的输入数据
    def dummy_inputs(self):
        # 获取配置中的填充标记 ID
        pad_token = self.config.pad_token_id
        # 创建包含两个示例输入序列的张量，使用 PyTorch 库生成
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        # 构建虚拟输入的字典，包含注意力掩码和输入序列
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),  # 生成注意力掩码，用于指示哪些位置是真实输入
            "input_ids": input_ids,  # 将输入序列添加到字典中
        }
        # 返回生成的虚拟输入字典
        return dummy_inputs
# 使用 @dataclass 装饰器声明一个数据类，用于定义 LEDEncoderBaseModelOutput 类
@dataclass
# 从 transformers.models.longformer.modeling_longformer.LongformerBaseModelOutput 复制代码，并将 Longformer 替换为 LEDEncoder
# LEDEncoderBaseModelOutput 类是 LEDEncoder 模型输出的基类，可能包含隐藏状态、局部和全局注意力等信息
class LEDEncoderBaseModelOutput(ModelOutput):
    """
    LEDEncoder 的输出基类，可能包含隐藏状态、局部和全局注意力等信息。
    """
    # 定义函数参数：最后一层模型隐藏状态
    last_hidden_state: torch.FloatTensor
    # 定义函数参数：可选项，隐藏状态的元组，包括每层模型的隐藏状态和初始嵌入输出
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 定义一个名为 attentions 的可选元组，用于存储 torch.FloatTensor 类型的数据，初始值为 None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 定义一个名为 global_attentions 的可选元组，用于存储 torch.FloatTensor 类型的数据，初始值为 None
    global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
@dataclass
class LEDSeq2SeqModelOutput(ModelOutput):
    """
    LEDSeq2SeqModelOutput 类，继承自 ModelOutput，用于表示模型编码器的输出，
    同时包含预先计算的隐藏状态，可加快顺序解码过程。
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class LEDSeq2SeqLMOutput(ModelOutput):
    """
    LEDSeq2SeqLMOutput 类，继承自 ModelOutput，用于表示序列到序列语言模型的输出。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class LEDSeq2SeqSequenceClassifierOutput(ModelOutput):
    """
    LEDSeq2SeqSequenceClassifierOutput 类，继承自 ModelOutput，用于表示序列到序列句子分类模型的输出。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class LEDSeq2SeqQuestionAnsweringModelOutput(ModelOutput):
    """
    LEDSeq2SeqQuestionAnsweringModelOutput 类，继承自 ModelOutput，用于表示序列到序列问答模型的输出。
    """

    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 定义一个可选的变量，用于存储编码器的最后隐藏状态，初始化为None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 定义一个可选的元组变量，用于存储编码器的所有隐藏状态，初始化为None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 定义一个可选的元组变量，用于存储编码器的所有注意力权重，初始化为None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 定义一个可选的元组变量，用于存储编码器的所有全局注意力权重，初始化为None
    encoder_global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# LED_START_DOCSTRING 是一个长字符串，包含有关 LED 模型的文档说明。它继承自 PreTrainedModel，并列出了模型通用方法和使用方式。
LED_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. See the superclass documentation for the generic methods the library
    implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for general usage and behavior.

    Parameters:
        config ([`LEDConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# LED_GENERATION_EXAMPLE 是一个包含摘要示例的字符串，展示了如何使用 LED 模型进行摘要生成。
LED_GENERATION_EXAMPLE = r"""
    Summarization example:

    ```
    >>> import torch
    >>> from transformers import AutoTokenizer, LEDForConditionalGeneration

    >>> model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv")
    >>> tokenizer = AutoTokenizer.from_pretrained("allenai/led-large-16384-arxiv")

    >>> ARTICLE_TO_SUMMARIZE = '''Transformers (Vaswani et al., 2017) have achieved state-of-the-art
    ...     results in a wide range of natural language tasks including generative language modeling
    ...     (Dai et al., 2019; Radford et al., 2019) and discriminative ... language understanding (Devlin et al., 2019).
    ...     This success is partly due to the self-attention component which enables the network to capture contextual
    ...     information from the entire sequence. While powerful, the memory and computational requirements of
    ...     self-attention grow quadratically with sequence length, making it infeasible (or very expensive) to
    ...     process long sequences. To address this limitation, we present Longformer, a modified Transformer
    ...     architecture with a self-attention operation that scales linearly with the sequence length, making it
    ...     versatile for processing long documents (Fig 1). This is an advantage for natural language tasks such as
    ...     long document classification, question answering (QA), and coreference resolution, where existing approaches
    ...     partition or shorten the long context into smaller sequences that fall within the typical 512 token limit
    ...     of BERT-style pretrained models. Such partitioning could potentially result in loss of important
    ...     cross-partition information, and to mitigate this problem, existing methods often rely on complex
    ...     architectures to address such interactions. On the other hand, our proposed Longformer is able to build
    ...     contextual representations of the entire context using multiple layers of attention, reducing the need for
    ...     task-specific architectures.'''
    >>> inputs = tokenizer.encode(ARTICLE_TO_SUMMARIZE, return_tensors="pt")
    # 使用给定的文章进行分词编码，返回PyTorch张量表示的输入
    
    >>> global_attention_mask = torch.zeros_like(inputs)
    # 创建一个与输入张量相同大小的全零张量，用于全局注意力掩码
    
    >>> global_attention_mask[:, 0] = 1
    # 将全局注意力掩码的第一个位置设置为1，以指示模型应该全局关注输入的第一个token
    
    >>> summary_ids = model.generate(inputs, global_attention_mask=global_attention_mask, num_beams=3, max_length=32)
    # 使用模型生成摘要，传入输入张量、全局注意力掩码、束搜索数量和最大长度限制
    
    >>> print(tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
    # 解码生成的摘要，跳过特殊标记并清理标记化空格后打印输出
"""
LED_INPUTS_DOCSTRING = r"""
"""


class LEDEncoder(LEDPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self-attention layers. Each layer is a
    [`LEDEncoderLayer`].

    Args:
        config: LEDConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: LEDConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout  # 从配置中获取 dropout 概率
        self.layerdrop = config.encoder_layerdrop  # 从配置中获取层之间的 dropout 概率

        embed_dim = config.d_model  # 从配置中获取嵌入维度
        self.padding_idx = config.pad_token_id  # 从配置中获取填充标记的索引
        self.max_source_positions = config.max_encoder_position_embeddings  # 从配置中获取最大源位置编码数

        if isinstance(config.attention_window, int):
            if config.attention_window % 2 != 0:
                raise ValueError("`config.attention_window` has to be an even value")  # 如果注意窗口大小为奇数则报错
            if config.attention_window <= 0:
                raise ValueError("`config.attention_window` has to be positive")  # 如果注意窗口大小为非正数则报错
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # 每层使用相同的注意窗口大小
        else:
            if len(config.attention_window) != config.num_hidden_layers:
                raise ValueError(
                    "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                    f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
                )  # 如果注意窗口大小列表长度不匹配层数则报错

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens  # 如果提供了嵌入标记，则使用它
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)  # 否则创建新的嵌入层

        self.embed_positions = LEDLearnedPositionalEmbedding(
            self.max_source_positions,
            embed_dim,
        )  # 创建学习的位置编码嵌入层

        self.layers = nn.ModuleList([LEDEncoderLayer(config, i) for i in range(config.encoder_layers)])  # 创建多层编码器层
        self.layernorm_embedding = nn.LayerNorm(embed_dim)  # 创建归一化层用于嵌入层

        self.gradient_checkpointing = False  # 是否使用梯度检查点，默认为 False

        # Initialize weights and apply final processing
        self.post_init()

    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        # longformer self-attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)  # 合并局部和全局注意力掩码
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1  # 如果没有给定注意力掩码，则使用全局注意力掩码

        return attention_mask

    def _pad_to_window_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        pad_token_id: int,
    ):
        """
        A helper function to pad tokens and mask to work with implementation of Longformer self-attention.
        """
        # padding
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        if attention_window % 2 != 0:
            raise ValueError(f"`attention_window` should be an even value. Given {attention_window}")
        
        # Determine the shape of the input tensor (either input_ids or inputs_embeds)
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        # Calculate the padding length required to make seq_len a multiple of attention_window
        padding_len = (attention_window - seq_len % attention_window) % attention_window
        
        # Warn and pad input_ids or inputs_embeds if padding is necessary
        if padding_len > 0:
            logger.warning_once(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )
            if input_ids is not None:
                # Pad input_ids tensor with pad_token_id
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                # Create padding tensor for inputs_embeds and concatenate with original inputs_embeds
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.pad_token_id,
                    dtype=torch.long,
                )
                inputs_embeds_padding = self.embed_tokens(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            # Pad attention_mask tensor to match the new input_ids shape
            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=False
            )  # no attention on the padding tokens

        # Return the padding length and updated tensors: input_ids, attention_mask, inputs_embeds
        return padding_len, input_ids, attention_mask, inputs_embeds

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
class LEDDecoder(LEDPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`LEDDecoderLayer`]

    Args:
        config: LEDConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: LEDConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout  # 从配置中获取dropout比率
        self.layerdrop = config.decoder_layerdrop  # 从配置中获取decoder层的layerdrop比率
        self.padding_idx = config.pad_token_id  # 从配置中获取填充token的索引
        self.max_target_positions = config.max_decoder_position_embeddings  # 获取最大解码位置

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens  # 如果提供了embed_tokens，则使用给定的嵌入
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)  # 否则创建新的嵌入层

        self.embed_positions = LEDLearnedPositionalEmbedding(
            self.max_target_positions,
            config.d_model,
        )  # 创建学习的位置嵌入

        self.layers = nn.ModuleList([LEDDecoderLayer(config) for _ in range(config.decoder_layers)])  # 创建多层decoder层
        self.layernorm_embedding = nn.LayerNorm(config.d_model)  # 创建嵌入层的LayerNorm

        self.gradient_checkpointing = False  # 是否使用梯度检查点优化，默认为False

        # Initialize weights and apply final processing
        self.post_init()  # 执行后初始化步骤



    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass of the LEDDecoder model.

        Args:
            input_ids (torch.LongTensor, optional): Input token IDs, shape (batch_size, seq_length).
            attention_mask (torch.Tensor, optional): Attention mask for input_ids, shape (batch_size, seq_length).
            global_attention_mask (torch.Tensor, optional): Global attention mask, shape (batch_size, seq_length).
            encoder_hidden_states (torch.FloatTensor, optional): Hidden states from the encoder.
            encoder_attention_mask (torch.FloatTensor, optional): Attention mask for encoder_hidden_states.
            head_mask (torch.FloatTensor, optional): Mask to nullify heads, shape (num_heads).
            cross_attn_head_mask (torch.FloatTensor, optional): Mask for cross-attention heads, shape (num_decoder_heads, num_encoder_heads).
            past_key_values (tuple, optional): Cached key/values for faster autoregressive decoding.
            inputs_embeds (torch.FloatTensor, optional): Embedded inputs if input_ids is not provided, shape (batch_size, seq_length, embed_dim).
            use_cache (bool, optional): Whether to use cached key/values for autoregressive decoding.
            output_attentions (bool, optional): Whether to return attentions weights.
            output_hidden_states (bool, optional): Whether to return hidden states.
            return_dict (bool, optional): Whether to return a dictionary as output.

        Returns:
            Sequence of output tensors depending on flags (return_dict, output_attentions, output_hidden_states).
        """
        # Forward pass logic of LEDDecoder
        # (具体的前向传播逻辑由各个方法和层实现，这里是参数和返回值的说明)



@add_start_docstrings(
    "The bare LED Model outputting raw hidden-states without any specific head on top.",
    LED_START_DOCSTRING,
)
class LEDModel(LEDPreTrainedModel):
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]

    def __init__(self, config: LEDConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = LEDEncoder(config, self.shared)  # 创建LED编码器
        self.decoder = LEDDecoder(config, self.shared)  # 创建LED解码器

        # Initialize weights and apply final processing
        self.post_init()  # 执行后初始化步骤

    def get_input_embeddings(self):
        return self.shared  # 返回共享的嵌入层

    def set_input_embeddings(self, value):
        self.shared = value  # 设置新的嵌入层
        self.encoder.embed_tokens = self.shared  # 更新编码器的嵌入层
        self.decoder.embed_tokens = self.shared  # 更新解码器的嵌入层

    def get_encoder(self):
        return self.encoder  # 返回编码器对象

    def get_decoder(self):
        return self.decoder  # 返回解码器对象

    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass of the LEDModel.

        Args:
            input_ids (torch.LongTensor, optional): Input token IDs, shape (batch_size, seq_length).
            attention_mask (torch.Tensor, optional): Attention mask for input_ids, shape (batch_size, seq_length).
            global_attention_mask (torch.Tensor, optional): Global attention mask, shape (batch_size, seq_length).
            encoder_outputs (tuple, optional): Outputs of the encoder.
            decoder_input_ids (torch.LongTensor, optional): Decoder input token IDs, shape (batch_size, seq_length).
            decoder_attention_mask (torch.Tensor, optional): Attention mask for decoder_input_ids, shape (batch_size, seq_length).
            decoder_past_key_values (tuple, optional): Cached key/values for faster autoregressive decoding.
            use_cache (bool, optional): Whether to use cached key/values for autoregressive decoding.
            output_attentions (bool, optional): Whether to return attentions weights.
            output_hidden_states (bool, optional): Whether to return hidden states.
            return_dict (bool, optional): Whether to return a dictionary as output.

        Returns:
            Sequence of output tensors depending on flags (return_dict, output_attentions, output_hidden_states).
        """
        # Forward pass logic of LEDModel
        # (具体的前向传播逻辑由各个方法和层实现，这里是参数和返回值的说明)
    # 定义 Transformer 模型的前向传播方法，处理输入和输出
    def forward(
        self,
        # 输入序列的 token IDs，可选的长整型张量
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力掩码，指示哪些元素是填充的，可选的张量
        attention_mask: Optional[torch.Tensor] = None,
        # 解码器输入的 token IDs，可选的长整型张量
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器的注意力掩码，指示哪些元素是填充的，可选的长整型张量
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 头部掩码，用于指定哪些注意力头部应该被保留，可选的张量
        head_mask: Optional[torch.Tensor] = None,
        # 解码器头部掩码，用于指定哪些解码器的注意力头部应该被保留，可选的张量
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头部掩码，用于指定哪些交叉注意力头部应该被保留，可选的张量
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出的元组，包含每一层的输出，可选的张量
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 全局注意力掩码，指示哪些元素是填充的，可选的浮点数张量
        global_attention_mask: Optional[torch.FloatTensor] = None,
        # 过去键值对，用于缓存的元组，包含每一层的键值对张量
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 输入的嵌入张量，可选的浮点数张量
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 解码器输入的嵌入张量，可选的浮点数张量
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # 是否使用缓存，可选的布尔值
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，可选的布尔值
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选的布尔值
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的结果，可选的布尔值
        return_dict: Optional[bool] = None,
# 使用装饰器添加文档字符串，描述此类是一个带有语言建模头的 LED 模型，可以用于摘要生成
@add_start_docstrings(
    "The LED Model with a language modeling head. Can be used for summarization.", LED_START_DOCSTRING
)
# 定义 LEDForConditionalGeneration 类，继承自 LEDPreTrainedModel
class LEDForConditionalGeneration(LEDPreTrainedModel):
    # 模型的基础名称前缀为 "led"
    base_model_prefix = "led"
    # 在加载模型时忽略的键名列表，缺失时不加载 "final_logits_bias"
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    # 共享权重的键名列表
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化方法，接收 LEDConfig 类型的 config 参数
    def __init__(self, config: LEDConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 LEDModel 对象，使用给定的 config
        self.led = LEDModel(config)
        # 注册一个缓冲区 "final_logits_bias"，大小为 (1, self.led.shared.num_embeddings)，初始化为零张量
        self.register_buffer("final_logits_bias", torch.zeros((1, self.led.shared.num_embeddings)))
        # 创建一个线性层 lm_head，将输入大小 config.d_model 映射到 self.led.shared.num_embeddings，不使用偏置
        self.lm_head = nn.Linear(config.d_model, self.led.shared.num_embeddings, bias=False)

        # 执行初始化权重和应用最终处理
        self.post_init()

    # 获取编码器的方法
    def get_encoder(self):
        return self.led.get_encoder()

    # 获取解码器的方法
    def get_decoder(self):
        return self.led.get_decoder()

    # 调整 token embeddings 的大小，返回调整后的新的 nn.Embedding 对象
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        # 调用父类的 resize_token_embeddings 方法，获取新的 embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 调用 _resize_final_logits_bias 方法，调整 final_logits_bias 的大小
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        # 返回新的 embeddings
        return new_embeddings

    # 调整 final_logits_bias 的大小，确保与新的 token 数量匹配
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        # 获取当前 final_logits_bias 的 token 数量
        old_num_tokens = self.final_logits_bias.shape[-1]
        # 如果新的 token 数量小于等于当前的数量，截取部分旧的 final_logits_bias
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        # 如果新的 token 数量大于当前的数量，扩展 final_logits_bias，并填充零张量
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        # 注册更新后的 final_logits_bias
        self.register_buffer("final_logits_bias", new_bias)

    # 获取输出 embeddings 的方法
    def get_output_embeddings(self):
        return self.lm_head

    # 设置新的输出 embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 使用装饰器添加文档字符串，描述模型前向传播的输入
    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
    # 替换返回的文档字符串，指定输出类型为 Seq2SeqLMOutput，使用 _CONFIG_FOR_DOC 配置类
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 添加末尾的文档字符串，提供 LED 生成的示例
    @add_end_docstrings(LED_GENERATION_EXAMPLE)
    # 定义模型的前向传播方法，用于生成模型输出
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        global_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 如果使用过去的键值（past_key_values），则仅保留decoder_input_ids的最后一个标记
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回一个字典，包含用于模型生成的输入和相关掩码信息
        return {
            "input_ids": None,  # encoder_outputs已定义，不需要input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 将此项更改为避免缓存（可能用于调试）
        }

    # 准备生成过程中的输入，用于生成decoder_input_ids
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用过去的键值（past_key_values），则截断decoder_input_ids
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        # 返回一个字典，包含用于生成的输入数据
        return {
            "input_ids": None,  # encoder_outputs已定义，input_ids不需要
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 将此项更改为避免缓存（可能用于调试）
        }

    # 根据标签生成decoder_input_ids，用于模型解码
    @staticmethod
    def prepare_decoder_input_ids_from_labels(labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    # 重新排列缓存中的键值对，以便与beam search结果对应
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态不需要重新排序 -> 它们始终相同
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
@add_start_docstrings(
    """
    LED model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    LED_START_DOCSTRING,
)
class LEDForSequenceClassification(LEDPreTrainedModel):
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]

    def __init__(self, config: LEDConfig, **kwargs):
        # 发出警告信息，表明此类将在 Transformers 版本 5 中被移除
        warnings.warn(
            "The `transformers.LEDForSequenceClassification` class is deprecated and will be removed in version 5 of"
            " Transformers. No actual method were provided in the original paper on how to perfom"
            " sequence classification.",
            FutureWarning,
        )
        # 调用父类构造函数初始化模型配置
        super().__init__(config, **kwargs)
        # 创建 LEDModel 实例
        self.led = LEDModel(config)
        # 创建用于分类任务的分类头部
        self.classification_head = LEDClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        global_attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



@add_start_docstrings(
    """
    LED Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer
    on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LED_START_DOCSTRING,
)
class LEDForQuestionAnswering(LEDPreTrainedModel):
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]
    # 初始化函数，接受一个配置参数config
    def __init__(self, config):
        # 调用父类的初始化函数，传入配置参数config
        super().__init__(config)

        # 设置模型的分类标签数为2
        config.num_labels = 2
        # 将分类标签数保存到实例变量self.num_labels中
        self.num_labels = config.num_labels

        # 创建LEDModel对象，传入配置参数config，并保存到self.led中
        self.led = LEDModel(config)
        
        # 创建一个线性层，输入维度为config.hidden_size，输出维度为config.num_labels，并保存到self.qa_outputs中
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 调用模型的后初始化方法
        # 在这个方法里进行权重的初始化和最终的处理
        self.post_init()

    # 前向传播函数，定义模型的输入输出及其处理过程
    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        global_attention_mask: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```