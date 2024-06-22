# `.\transformers\models\led\modeling_led.py`

```py
# coding=utf-8
# 版权所有 2021 Iz Beltagy，Matthew E. Peters，Arman Cohan 和 The HuggingFace Inc. 团队。保留所有权利。
# 根据 Apache 许可证 2.0（“许可证”）进行许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获得许可证副本：
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，
# 没有任何形式的保证或条件，无论是明示的还是暗示的。
# 有关许可证的限制，请参阅许可证。

"""PyTorch LED 模型。"""


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

# 获取 logger
logger = logging.get_logger(__name__)

# 文档中的模型文件和配置文件
_CHECKPOINT_FOR_DOC = "allenai/led-base-16384"
_CONFIG_FOR_DOC = "LEDConfig"

# LED 预训练模型库
LED_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "allenai/led-base-16384",
    # 查看所有 LED 模型：https://huggingface.co/models?filter=led
]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    将输入的 token 向右移动一个位置。
    """
    # 创建一个与 input_ids 形状相同的新 Tensor，并用 0 填充
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将 input_ids 的数据从第二个 token 开始复制到 shifted_input_ids 中
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # 将 decoder_start_token_id（解码器的起始 token）放入 shifted_input_ids 的首位位置

    # 若未定义 pad_token_id，则抛出 ValueError
    if pad_token_id is None:
        raise ValueError("config.pad_token_id has to be defined.")
    # 将 shifted_input_ids 中可能的 -100 值替换为 pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _prepare_4d_attention_mask_inverted(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    将一个 `[bsz, seq_len]` 的 attention_mask 扩展到 `[bsz, 1, tgt_seq_len, src_seq_len]`。
    """
    # 获取 mask 的 shape
    bsz, src_len = mask.size()
    # 若未指定 tgt_len，则默认为 src_len
    tgt_len = tgt_len if tgt_len is not None else src_len

    # 创建与 mask 形状相同的新 Tensor 并扩展维度
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    # 计算 inverted_mask 和 expanded_attention_mask 的值
    inverted_mask = 1.0 - expanded_mask
    expanded_attention_mask = inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

    # 确保 global_attn_mask 为正数
    # 将扩展后的注意力掩码与反转的掩码逐元素相乘，以实现对掩码的部分设置
    expanded_attention_mask = expanded_attention_mask * inverted_mask
    
    # 返回经过处理的扩展后的注意力掩码
    return expanded_attention_mask
class LEDLearnedPositionalEmbedding(nn.Embedding):
    """
    这个模块学习固定最大尺寸的位置嵌入。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape`预期为[bsz x seqlen]。"""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions)


# 从transformers.models.longformer.modeling_longformer.LongformerSelfAttention复制并更改为LEDEncoder
class LEDEncoderSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"隐藏大小({config.hidden_size})不是注意力头数({config.num_attention_heads})的倍数"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        # 为全局注意力的token单独设置投影层
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"第{self.layer_id}层的attention_window必须是偶数。给定{attention_window}"
        assert (
            attention_window > 0
        ), f"第{self.layer_id}层的attention_window必须是正数。给定{attention_window}"

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
    @staticmethod
    # 对隐藏状态进行填充并交换最后两个维度
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """pads rows and then flips rows and columns"""
        # 使用 nn.functional.pad 对隐藏状态进行填充
        hidden_states_padded = nn.functional.pad(
            hidden_states_padded, padding
        )  # padding value is not important because it will be overwritten
        # 交换隐藏状态的最后两个维度
        hidden_states_padded = hidden_states_padded.view(
            *hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2)
        )
        # 返回填充和转置后的隐藏状态
        return hidden_states_padded

    @staticmethod
    # 对隐藏状态进行填充和对角化
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        shift every row 1 step right, converting columns into diagonals.

        Example:
        ...

        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()
        # 使用 nn.functional.pad 对分块的隐藏状态进行填充
        chunked_hidden_states = nn.functional.pad(
            chunked_hidden_states, (0, window_overlap + 1)
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        # 将填充后的隐藏状态进行维度变换
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, -1
        )  # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        # 截取不需要的填充部分
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # total_num_heads x num_chunks x window_overlap*window_overlap
        # 进行最终的维度变换
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]  # 移除最后一个维度
        # 返回填充和对角化后的隐藏状态
        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap, onnx_export: bool = False):
        """将隐藏状态转换为重叠的块。块大小= 2w，重叠大小= w"""
        if not onnx_export:
            # 将大小为2w的非重叠块转换为视图
            hidden_states = hidden_states.view(
                hidden_states.size(0),
                torch.div(hidden_states.size(1), (window_overlap * 2), rounding_mode="trunc"),
                window_overlap * 2,
                hidden_states.size(2),
            )
            # 使用`as_strided`使块与重叠大小= window_overlap
            chunk_size = list(hidden_states.size())
            chunk_size[1] = chunk_size[1] * 2 - 1

            chunk_stride = list(hidden_states.stride())
            chunk_stride[1] = chunk_stride[1] // 2
            return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)

        # 在导出到ONNX时，使用这个单独的逻辑
        # 因为在ONNX导出中，`as_strided`、`unfold` 和 二维张量索引不受支持，所以必须使用较慢的实现

        # TODO 用以下代码替换
        # > return hidden_states.unfold(dimension=1, size=window_overlap * 2, step=window_overlap).transpose(2, 3)
        # 一旦`unfold`得到支持
        # 当hidden_states.size(1) == window_overlap * 2时，也可以简单地返回hidden_states.unsqueeze(1)，但那是控制流

        chunk_size = [
            hidden_states.size(0),
            torch.div(hidden_states.size(1), window_overlap, rounding_mode="trunc") - 1,
            window_overlap * 2,
            hidden_states.size(2),
        ]

        overlapping_chunks = torch.empty(chunk_size, device=hidden_states.device)
        for chunk in range(chunk_size[1]):
            overlapping_chunks[:, chunk, :, :] = hidden_states[
                :, chunk * window_overlap : chunk * window_overlap + 2 * window_overlap, :
            ]
        return overlapping_chunks

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
            beginning_input, -float("inf")
        ).where(beginning_mask.bool(), beginning_input)
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
            ending_input, -float("inf")
        ).where(ending_mask.bool(), ending_input)
    # 使用滑动窗口的方式对注意力概率和数值进行矩阵乘法，返回的张量形状与 attn_probs 相同
    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        # 获取 batch_size, seq_len, num_heads, head_dim 的值
        batch_size, seq_len, num_heads, head_dim = value.size()

        # 断言，确保 seq_len 可以被 2*window_overlap 整除
        assert seq_len % (window_overlap * 2) == 0
        # 断言，确保 attn_probs 和 value 的前三个维度相等
        assert attn_probs.size()[:3] == value.size()[:3]
        # 断言，确保 attn_probs 的第四个维度为 2*window_overlap+1
        assert attn_probs.size(3) == 2 * window_overlap + 1
        # 计算 chunk 的数量
        chunks_count = torch.div(seq_len, window_overlap, rounding_mode="trunc") - 1
        # 将 batch_size 和 num_heads 维度合并，然后将 seq_len 划分为大小为 2*window_overlap 的 chunk

        # 转置和重塑 attn_probs
        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads,
            torch.div(seq_len, window_overlap, rounding_mode="trunc"),
            window_overlap,
            2 * window_overlap + 1,
        )

        # 合并 batch_size 和 num_heads 维度
        value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        # 在序列的开头和结尾各填充一个窗口重叠大小的值
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # 将填充后的值分成大小为 3*window_overlap 和重叠大小为 window_overlap 的 chunk
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

        # 使用 Einstein Summation 对 chunked_attn_probs 和 chunked_value 进行矩阵乘法
        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        # 调整形状，然后对结果进行维度转置
        return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """计算在整个前向传播过程中需要的全局注意力索引"""
        # 计算每个样本中全局注意力索引的数量
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)

        # 批次中全局注意力索引的最大数量
        max_num_global_attn_indices = num_global_attn_indices.max()

        # 全局注意力索引的索引位置
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)

        # 辅助变量
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)

        # 全局注意力索引中非填充值的位置
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)

        # 全局注意力索引中填充值的位置
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

        # 仅创建全局键向量
        key_vectors_only_global = key_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )

        key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[is_index_global_attn_nonzero]

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = torch.einsum("blhd,bshd->blhs", (query_vectors, key_vectors_only_global))

        # 需要转置，因为 ONNX 导出仅支持连续索引: https://pytorch.org/docs/stable/onnx.html#writes-sets
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)
        attn_probs_from_global_key[
            is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :
        ] = torch.finfo(attn_probs_from_global_key.dtype).min
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)

        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(
        self,
        value_vectors,
        attn_probs,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
    # 计算批处理大小
    batch_size = attn_probs.shape[0]
    
    # 仅保留全局注意力概率
    attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
    # 获取全局注意力的值向量
    value_vectors_only_global = value_vectors.new_zeros(
        batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
    )
    value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[is_index_global_attn_nonzero]
    
    # 使用`matmul`，因为`einsum`有时会在fp16时崩溃
    # 仅计算全局注意力输出
    attn_output_only_global = torch.matmul(
        attn_probs_only_global.transpose(1, 2).clone(), value_vectors_only_global.transpose(1, 2).clone()
    ).transpose(1, 2)
    
    # 重新整形注意力概率
    attn_probs_without_global = attn_probs.narrow(
        -1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices
    ).contiguous()
    
    # 计算带有全局注意力的注意力输出
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
    # 定义LEDEncoderAttention类，继承自nn.Module类
    def __init__(self, config, layer_id):
        # 初始化方法，包含config配置和layer_id参数
        super().__init__()
        # 调用父类的初始化方法
        self.longformer_self_attn = LEDEncoderSelfAttention(config, layer_id=layer_id)
        # 初始化self.longformer_self_attn为LEDEncoderSelfAttention类的实例，传入config和layer_id参数
        self.output = nn.Linear(config.d_model, config.d_model)
        # 初始化self.output为Linear层，将输入维度和输出维度都设为config.d_model

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
        # forward方法用于前向传播计算
        self_outputs = self.longformer_self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        # 调用longformer_self_attn进行自注意力计算

        attn_output = self.output(self_outputs[0])
        # 将self_outputs第一个元素经过self.output线性层得到attention的输出
        outputs = (attn_output,) + self_outputs[1:]
        # 将attention输出和self_outputs的其他部分组合成元组outputs并返回

        return outputs


class LEDDecoderAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 定义LEDDecoderAttention类，实现'Attention Is All You Need'论文中的多头注意力机制

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        # 初始化方法，包含嵌入维度、头数、dropout率、是否为解码器以及是否存在偏置等参数
        super().__init__()
        # 调用父类的初始化方法
        self.embed_dim = embed_dim
        # 初始化嵌入维度
        self.num_heads = num_heads
        # 初始化头数
        self.dropout = dropout
        # 初始化dropout率
        self.head_dim = embed_dim // num_heads
        # 计算头维度
        if self.head_dim * num_heads != self.embed_dim:
            # 如果嵌入维度不能整除头数，抛出数值错误
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        # 计算缩放系数
        self.is_decoder = is_decoder
        # 设置是否为解码器标志

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 设置k映射线性层
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 设置v映射线性层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 设置q映射线性层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 设置输出映射线性层

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 定义私有方法_shape，用于整理张量形状
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # 调整张量形状并转置维度以维护连续性

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,


class LEDEncoderLayer(nn.Module):


        ```
    # 初始化方法，接受配置和层编号作为参数
    def __init__(self, config: LEDConfig, layer_id: int):
        # 调用父类的初始化方法
        super().__init__()
        # 设置嵌入维度为配置中的模型维度
        self.embed_dim = config.d_model
        # 创建自注意力层对象
        self.self_attn = LEDEncoderAttention(config, layer_id)
        # 创建自注意力层后的 LayerNorm 层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设置丢弃概率为配置中的丢弃概率
        self.dropout = config.dropout
        # 获取激活函数类型，并设置为当前激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数的丢弃概率为配置中的激活函数丢弃概率
        self.activation_dropout = config.activation_dropout
        # 创建全连接层1，输入维度为嵌入维度，输出维度为配置中的编码器前馈神经网络维度
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 创建全连接层2，输入维度为配置中的编码器前馈神经网络维度，输出维度为嵌入维度
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 创建最终的 LayerNorm 层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 前向传播方法，接受隐藏状态、注意力掩码、层头掩码等参数
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
        # 备份隐藏状态
        residual = hidden_states
        # 调用自注意力层进行前向传播
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        # 更新隐藏状态为自注意力层的输出
        hidden_states = attn_outputs[0]
        # 使用丢弃概率进行丢弃
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接到当前隐藏状态
        hidden_states = residual + hidden_states
        # 对当前隐藏状态进行 LayerNorm
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 备份当前隐藏状态
        residual = hidden_states
        # 使用激活函数对全连接层1进行计算
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 使用激活函数的丢弃概率进行丢弃
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 使用全连接层2进行计算
        hidden_states = self.fc2(hidden_states)
        # 使用丢弃概率进行丢弃
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接到当前隐藏状态
        hidden_states = residual + hidden_states
        # 对当前隐藏状态进行最终的 LayerNorm
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果当前隐藏状态的数据类型为 torch.float16，并且包含 inf 或 nan，则进行截断处理
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
        # 返回隐藏状态及注意力输出（如果需要）
        return (hidden_states,) + attn_outputs[1:]
class LEDDecoderLayer(nn.Module):
    # LED 解码器层的类定义
    def __init__(self, config: LEDConfig):
        # 初始化函数，接受一个 LEDConfig 类型的参数
        super().__init__()
        # 调用父类的初始化函数
        self.embed_dim = config.d_model
        # 从配置中获取嵌入维度

        self.self_attn = LEDDecoderAttention(
            # 定义自注意力机制
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        # 设置丢弃率
        self.activation_fn = ACT2FN[config.activation_function]
        # 获取激活函数
        self.activation_dropout = config.activation_dropout
        # 获取激活函数的丢弃率

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 对自注意力机制进行层归一化
        self.encoder_attn = LEDDecoderAttention(
            # 定义编码器注意力机制
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 对编码器注意力机制进行层归一化
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 第一个全连接层
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 第二个全连接层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        # 最终输出进行层归一化

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
    # 定义前向传播函数，包括各种输入参数
        ...
        # 其余代码省略
    # 创建一个方法用于生成虚拟输入数据
    def dummy_inputs(self):
        # 获取配置中的填充标记ID
        pad_token = self.config.pad_token_id
        # 创建一个张量，包含两个子列表，每个子列表表示一个句子的token ID
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        # 创建一个虚拟的输入字典，包含注意力掩码和输入token ID
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),  # 使用逻辑运算符生成注意力掩码
            "input_ids": input_ids,  # 将输入token ID添加到输入字典
        }
        return dummy_inputs  # 返回生成的虚拟输入字典
# 使用 dataclass 装饰器定义一个数据类
@dataclass
# 从 transformers.models.longformer.modeling_longformer.LongformerBaseModelOutput 拷贝过来，将其中的 Longformer 替换为 LEDEncoder
# 表示定义了一个名为 LEDEncoderBaseModelOutput 的基类，用于表示 LEDEncoder 的输出，包括潜在的隐藏状态、局部注意力和全局注意力。
class LEDEncoderBaseModelOutput(ModelOutput):
    """
    Base class for LEDEncoder's outputs, with potential hidden states, local and global attentions.
    """


代码：


@dataclass
class LEDEncoderOutput(ModelOutput):
    """
    Base class for LEDEncoder's outputs, with potential hidden states, local and global attentions.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None
    compressed_attentions: Optional[Tuple[torch.FloatTensor]] = None
    compressed_global_attentions: Optional[Tuple[torch.FloatTensor]] = None




注释：
    # 接收模型最后一层的隐藏状态作为输入
    last_hidden_state: torch.FloatTensor
    # 隐藏状态是一个元组，包含了从模型每一层得到的隐藏状态，以及初始嵌入的输出
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力是一个元组，包含了每一层的本地注意力权重，用于计算自注意力头中的加权平均值。
    # 这些是从序列中每个标记到具有全局注意力的每个标记之间的注意力权重，并且对于具有全局注意力的标记的所有其他标记的注意力权重设置为0，值应该从`global_attentions`中访问。
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 全局注意力是一个元组，包含了每一层的全局注意力权重，用于计算自注意力头中的加权平均值。
    # 这些是从具有全局注意力的每个标记到序列中的每个标记之间的注意力权重。
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义了两个可选的 Torch 浮点张量元组变量，分别为 attentions 和 global_attentions，初始值为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class LEDSeq2SeqModelOutput(ModelOutput):
    """
    LEDSeq2SeqModelOutput 类，继承自 ModelOutput 类，用于存储模型编码器的输出，同时包含了预计算的隐藏状态以加速顺序解码。
    """

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_global_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class LEDSeq2SeqLMOutput(ModelOutput):
    """
    LEDSeq2SeqLMOutput 类，用于存储序列到序列语言模型的输出。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_global_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class LEDSeq2SeqSequenceClassifierOutput(ModelOutput):
    """
    LEDSeq2SeqSequenceClassifierOutput 类，用于存储序列到序列句子分类模型的输出。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_global_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class LEDSeq2SeqQuestionAnsweringModelOutput(ModelOutput):
    """
    LEDSeq2SeqQuestionAnsweringModelOutput 类，用于存储序列到序列问答模型的输出。
    """

    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 定义一个可选的元组，存储编码器的隐藏状态，默认为None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个可选的元组，存储编码器的注意力，默认为None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个可选的元组，存储编码器的全局注意力，默认为None
    encoder_global_attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义LED_START_DOCSTRING，用来存储LED模型的文档字符串
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
# 定义LED_GENERATION_EXAMPLE，用来存储LED模型的生成示例
LED_GENERATION_EXAMPLE = r"""
    Summarization example:

    ```py
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
    # 使用 tokenizer 对文章进行编码，返回 PyTorch 张量
    inputs = tokenizer.encode(ARTICLE_TO_SUMMARIZE, return_tensors="pt")
    
    # 创建与 inputs 形状相同的全零张量，用于指定全局注意力
    global_attention_mask = torch.zeros_like(inputs)
    # 将全局注意力集中在第一个 token 上
    global_attention_mask[:, 0] = 1
    
    # 生成摘要
    summary_ids = model.generate(inputs, global_attention_mask=global_attention_mask, num_beams=3, max_length=32)
    # 打印摘要内容，跳过特殊 token 并清理 token 化空格
    print(tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
"""
LED_INPUTS_DOCSTRING = r"""
"""

# LEDEncoder 类的定义，继承自 LEDPreTrainedModel
class LEDEncoder(LEDPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self-attention layers. Each layer is a
    [`LEDEncoderLayer`].

    Args:
        config: LEDConfig  LED 模型的配置对象
        embed_tokens (nn.Embedding): output embedding 输出嵌入层对象
    """

    # LEDEncoder 类的初始化方法
    def __init__(self, config: LEDConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的初始化方法
        super().__init__(config)

        # 配置类中的属性赋值
        self.dropout = config.dropout  # dropout 参数
        self.layerdrop = config.encoder_layerdrop  # encoder_layerdrop 参数

        # 获取词嵌入的维度
        embed_dim = config.d_model
        # 获取填充标记的索引
        self.padding_idx = config.pad_token_id
        # 获取最大源序列长度
        self.max_source_positions = config.max_encoder_position_embeddings

        # 如果 config.attention_window 是 int 类型
        if isinstance(config.attention_window, int):
            # 如果窗口大小不是偶数，则抛出 ValueError
            if config.attention_window % 2 != 0:
                raise ValueError("`config.attention_window` has to be an even value")
            # 如果窗口大小小于等于0，则抛出 ValueError
            if config.attention_window <= 0:
                raise ValueError("`config.attention_window` has to be positive")
            # 为每一层的注意力窗口赋值，使其与隐藏层数相同
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        # 如果 config.attention_window 是 list 类型
        else:
            # 如果 config.attention_window 的长度与隐藏层数不相等，则抛出 ValueError
            if len(config.attention_window) != config.num_hidden_layers:
                raise ValueError(
                    "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                    f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
                )

        # 如果 embed_tokens 不为 None，则直接使用传入的嵌入层对象
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        # 如果 embed_tokens 为 None，则创建一个词嵌入层对象
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # 创建一个学习到的位置嵌入对象
        self.embed_positions = LEDLearnedPositionalEmbedding(
            self.max_source_positions,
            embed_dim,
        )
        # 创建多层 LEDEncoderLayer 组成的列表
        self.layers = nn.ModuleList([LEDEncoderLayer(config, i) for i in range(config.encoder_layers)])
        # 创建一个 LayerNorm 层，用于归一化词嵌入
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        # 初始化梯度检查点参数为 False
        self.gradient_checkpointing = False
        # 初始化权重并进行最终处理
        self.post_init()

    # 将局部和全局注意力融合到注意力掩码中
    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        # longformer 的自注意力期望注意掩码具有 0（无注意力），1（局部注意力），2（全局注意力）
        # （global_attention_mask + 1）=> 1 代表局部注意力，2 代表全局注意力
        # 最终的 attention_mask => 0 表示无注意力，1 表示局部注意力，2 表示全局注意力
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # 如果没有提供 attention_mask，则直接使用 global_attention_mask
            attention_mask = global_attention_mask + 1
        return attention_mask

    # 将输入填充到窗口大小
    def _pad_to_window_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        pad_token_id: int,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        # 计算注意力窗口大小，如果是整数则直接使用，如果是列表取最大值
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        # 如果注意力窗口大小为奇数，则抛出异常
        if attention_window % 2 != 0:
            raise ValueError(f"`attention_window` should be an even value. Given {attention_window}")
        
        # 获取输入的形状
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        # 计算需要填充的长度，使得序列长度能够被 attention_window 整除
        padding_len = (attention_window - seq_len % attention_window) % attention_window
        
        # 如果需要填充，则进行相应的处理
        if padding_len > 0:
            logger.warning_once(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )
            
            # 如果输入的是 token ids，则进行填充
            if input_ids is not None:
                input_ids = nn.functional.pad(input_ids, (0, padding_len), value=pad_token_id)
            
            # 如果输入的是嵌入向量，则进行填充
            if inputs_embeds is not None:
                # 生成填充的 token ids
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.pad_token_id,
                    dtype=torch.long,
                )
                # 根据填充的 token ids 生成填充的嵌入向量
                inputs_embeds_padding = self.embed_tokens(input_ids_padding)
                # 将填充的嵌入向量拼接到原始嵌入向量后面
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            # 填充注意力掩码，填充部分的注意力为 False
            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=False
            )  # no attention on the padding tokens

        # 返回填充的长度、填充后的输入 token ids、填充后的注意力掩码、填充后的输入嵌入向量
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
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置丢弃率
        self.dropout = config.dropout
        # 设置层丢弃率
        self.layerdrop = config.decoder_layerdrop
        # 设置填充索引
        self.padding_idx = config.pad_token_id
        # 设置最大目标位置
        self.max_target_positions = config.max_decoder_position_embeddings

        # 如果传入了输出嵌入，则使用传入的，否则创建一个新的输出嵌入
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # 创建学习到的位置编码嵌入
        self.embed_positions = LEDLearnedPositionalEmbedding(
            self.max_target_positions,
            config.d_model,
        )
        # 创建多个解码层
        self.layers = nn.ModuleList([LEDDecoderLayer(config) for _ in range(config.decoder_layers)])
        # 创建嵌入层的 LayerNorm
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        # 初始化梯度检查点为 False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

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
        # 此处应该是方法签名，缺失
        pass


@add_start_docstrings(
    "The bare LED Model outputting raw hidden-states without any specific head on top.",
    LED_START_DOCSTRING,
)
class LEDModel(LEDPreTrainedModel):
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]

    def __init__(self, config: LEDConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # 创建共享嵌入层
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # 创建编码器和解码器
        self.encoder = LEDEncoder(config, self.shared)
        self.decoder = LEDDecoder(config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回输入嵌入层
        return self.shared

    def set_input_embeddings(self, value):
        # 设置输入嵌入层
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        # 返回编码器
        return self.encoder

    def get_decoder(self):
        # 返回解码器
        return self.decoder

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
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 此处应该是方法签名，缺失
        pass
```  
    # 定义前向传播函数，接受多个输入参数，包括输入的编码器词 id，注意力 mask，解码器输入词 id，解码器注意力 mask，编码器头 mask，解码器头 mask，
    # 交叉注意力头 mask，编码器输出，全局注意力 mask，过去的 key-value 对，输入嵌入，解码器输入嵌入，是否使用缓存，是否输出注意力权重，是否输出隐藏状态，是否返回字典对象
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        # 输入编码器词 id，默认为 None
        attention_mask: Optional[torch.Tensor] = None,
        # 注意力 mask，默认为 None
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器输入词 id，默认为 None
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 解码器注意力 mask，默认为 None
        head_mask: Optional[torch.Tensor] = None,
        # 编码器头 mask，默认为 None
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 解码器头 mask，默认为 None
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头 mask，默认为 None
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 编码器输出，默认为 None
        global_attention_mask: Optional[torch.FloatTensor] = None,
        # 全局注意力 mask，默认为 None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 过去的 key-value 对，默认为 None
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 输入嵌入，默认为 None
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # 解码器输入嵌入，默认为 None
        use_cache: Optional[bool] = None,
        # 是否使用缓存，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出注意力权重，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否输出隐藏状态，默认为 None
        return_dict: Optional[bool] = None,
        # 是否返回字典对象，默认为 None
# 添加开始文档字符串到LED模型，用于生成摘要
@add_start_docstrings(
    "The LED Model with a language modeling head. Can be used for summarization.", LED_START_DOCSTRING
)
# 定义LEDForConditionalGeneration类，继承自LEDPreTrainedModel
class LEDForConditionalGeneration(LEDPreTrainedModel):
    # 设置基础模型前缀为"led"
    base_model_prefix = "led"
    # 在加载时忽略缺失的键
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]
    # 定义权重共享的键
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化方法，接受LEDConfig类型的配置
    def __init__(self, config: LEDConfig):
        # 调用父类初始化方法
        super().__init__(config)
        # 创建LEDModel对象
        self.led = LEDModel(config)
        # 注册final_logits_bias的缓冲区，初始化为全零向量
        self.register_buffer("final_logits_bias", torch.zeros((1, self.led.shared.num_embeddings)))
        # 创建线性层对象lm_head，设定输入维度为config.d_model，输出维度为num_embeddings，无偏置项
        self.lm_head = nn.Linear(config.d_model, self.led.shared.num_embeddings, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取编码器
    def get_encoder(self):
        return self.led.get_encoder()

    # 获取解码器
    def get_decoder(self):
        return self.led.get_decoder()

    # 调整token嵌入层的大小
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 调整final_logits_bias
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    # 调整final_logits_bias的大小
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        # 获取旧token数
        old_num_tokens = self.final_logits_bias.shape[-1]
        # 根据新旧token数调整final_logits_bias
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 添加文档字符串到模型前向方法
    @add_start_docstrings_to_model_forward(LED_INPUTS_DOCSTRING)
    # 替换返回文档字符串为Seq2SeqLMOutput类型
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 添加结束文档字符串LED_GENERATION_EXAMPLE
    @add_end_docstrings(LED_GENERATION_EXAMPLE)
    # 定义了一个名为 forward 的方法，用于模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，可选参数，默认为 None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选参数，默认为 None
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器的输入 token IDs，可选参数，默认为 None
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器的注意力掩码，可选参数，默认为 None
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，可选参数，默认为 None
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器头部掩码，可选参数，默认为 None
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 跨注意力头部掩码，可选参数，默认为 None
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 编码器输出，可选参数，默认为 None
        global_attention_mask: Optional[torch.FloatTensor] = None,  # 全局注意力掩码，可选参数，默认为 None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 过去的键值对，可选参数，默认为 None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入，可选参数，默认为 None
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入的嵌入，可选参数，默认为 None
        labels: Optional[torch.LongTensor] = None,  # 标签，可选参数，默认为 None
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选参数，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选参数，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选参数，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果，可选参数，默认为 None
    # 定义了一个名为 prepare_inputs_for_generation 的方法，用于为生成准备输入
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,  # 解码器输入的 token IDs
        past_key_values=None,  # 过去的键值对，默认为 None
        attention_mask=None,  # 注意力掩码，默认为 None
        global_attention_mask=None,  # 全局注意力掩码，默认为 None
        head_mask=None,  # 头部掩码，默认为 None
        decoder_head_mask=None,  # 解码器头部掩码，默认为 None
        cross_attn_head_mask=None,  # 跨注意力头部掩码，默认为 None
        use_cache=None,  # 是否使用缓存，默认为 None
        encoder_outputs=None,  # 编码器输出，默认为 None
        **kwargs,  # 其它关键字参数
    ):
        # 如果使用了过去的键值对，截取解码器输入的最后一个 token
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        # 返回一个字典，包含为生成所需的所有输入
        return {
            "input_ids": None,  # encoder_outputs 已经定义，不需要 input_ids
            "encoder_outputs": encoder_outputs,  # 编码器输出
            "past_key_values": past_key_values,  # 过去的键值对
            "decoder_input_ids": decoder_input_ids,  # 解码器输入的 token IDs
            "attention_mask": attention_mask,  # 注意力掩码
            "global_attention_mask": global_attention_mask,  # 全局注意力掩码
            "head_mask": head_mask,  # 头部掩码
            "decoder_head_mask": decoder_head_mask,  # 解码器头部掩码
            "cross_attn_head_mask": cross_attn_head_mask,  # 跨注意力头部掩码
            "use_cache": use_cache,  # 是否使用缓存，这里将其改为避免缓存（可能用于调试）
        }
    # 定义了一个静态方法 _reorder_cache，用于重新排序缓存
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的跨注意力状态不需要重新排序，它们始终相同
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],  # 不需要重新排序的部分
            )
        return reordered_past
# 为序列分类任务设计的 LED 模型，包含一个序列分类头（在汇总输出的顶部是一个线性层），例如用于 GLUE 任务
@add_start_docstrings(
    """
    LED model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    LED_START_DOCSTRING,
)
class LEDForSequenceClassification(LEDPreTrainedModel):
    # 被绑定权重的键列表
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]

    def __init__(self, config: LEDConfig, **kwargs):
        # 警告：transformers.LEDForSequenceClassification 类已弃用，将在 Transformers 的第 5 版中移除。原始论文中未提供
        # 如何执行序列分类的实际方法。
        warnings.warn(
            "The `transformers.LEDForSequenceClassification` class is deprecated and will be removed in version 5 of"
            " Transformers. No actual method were provided in the original paper on how to perfom"
            " sequence classification.",
            FutureWarning,
        )
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)
        # 创建 LED 模型
        self.led = LEDModel(config)
        # 创建分类头
        self.classification_head = LEDClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 重写 forward 方法，添加了文档字符串和代码示例文档字符串
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
```py 


# 为抽取式问答任务设计的 LED 模型，包含一个跨度分类头（在隐藏状态输出的顶部是一个线性层），例如用于 SQuAD 任务
@add_start_docstrings(
    """
    LED Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer
    on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    LED_START_DOCSTRING,
)
class LEDForQuestionAnswering(LEDPreTrainedModel):
    # 被绑定权重的键列表
    _tied_weights_keys = ["decoder.embed_tokens.weight", "encoder.embed_tokens.weight"]
``` 
    # 初始化 Seq2SeqQuestionAnsweringModel 类
    def __init__(self, config):
        # 调用基类的初始化方法
        super().__init__(config)

        # 设置标签数量为2
        config.num_labels = 2
        self.num_labels = config.num_labels

        # 创建 LEDModel 对象
        self.led = LEDModel(config)
        # 创建一个线性层，用于输出问题答案
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接收若干输入参数
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