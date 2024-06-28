# `.\models\bigbird_pegasus\modeling_bigbird_pegasus.py`

```py
# coding=utf-8
# Copyright 2021 Google Research The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch BigBirdPegasus model."""

import copy  # 导入 copy 模块用于复制对象
import math  # 导入 math 模块用于数学运算
from typing import List, Optional, Tuple, Union  # 导入类型提示相关的类和函数

import numpy as np  # 导入 NumPy 库用于数值计算
import torch  # 导入 PyTorch 库
from torch import nn  # 导入 PyTorch 的神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入损失函数

from ...activations import ACT2FN  # 导入激活函数映射
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask  # 导入辅助注意力掩码函数
from ...modeling_outputs import (  # 导入模型输出相关类
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型的基类
from ...utils import (  # 导入工具函数
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_bigbird_pegasus import BigBirdPegasusConfig  # 导入配置类

logger = logging.get_logger(__name__)  # 获取日志记录器

_CHECKPOINT_FOR_DOC = "google/bigbird-pegasus-large-arxiv"  # 文档中使用的预训练模型检查点名称
_CONFIG_FOR_DOC = "BigBirdPegasusConfig"  # 文档中使用的配置类名称
_EXPECTED_OUTPUT_SHAPE = [1, 7, 1024]  # 预期输出的形状

BIGBIRD_PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 预训练模型的列表
    "google/bigbird-pegasus-large-arxiv",
    "google/bigbird-pegasus-large-pubmed",
    "google/bigbird-pegasus-large-bigpatent",
    # See all BigBirdPegasus models at https://huggingface.co/models?filter=bigbird_pegasus
]


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)  # 创建与 input_ids 形状相同的零张量
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()  # 将 input_ids 向右移动一位
    shifted_input_ids[:, 0] = decoder_start_token_id  # 在第一列填充 decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")  # 如果 pad_token_id 未定义则引发 ValueError
    # 将 shifted_input_ids 中可能的 -100 值替换为 pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids  # 返回右移后的 input_ids


class BigBirdPegasusLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)  # 调用父类 nn.Embedding 的构造方法，初始化位置嵌入层
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        # 从输入参数 `input_ids_shape` 中获取 batch size 和 sequence length
        bsz, seq_len = input_ids_shape[:2]
        # 生成一个序列，表示位置编码，起始位置从 `past_key_values_length` 到 `past_key_values_length + seq_len`
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        # 调用父类的 `forward` 方法，传入位置编码 `positions`，并返回结果
        return super().forward(positions)
# Copied from transformers.models.big_bird.modeling_big_bird.BigBirdSelfAttention with BigBird->BigBirdPegasus
# 定义了 BigBirdPegasusSelfAttention 类，继承自 nn.Module
class BigBirdPegasusSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 检查隐藏层大小是否是注意力头数的整数倍，如果不是则抛出 ValueError 异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义用于查询、键、值的线性变换层，输入大小为隐藏大小，输出大小为所有头的大小
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)

        # 定义用于dropout的层，以及是否作为解码器的标志
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.is_decoder = config.is_decoder

    # 将输入张量转换为注意力分数的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，接收隐藏状态等多个参数，执行自注意力机制
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):



# Copied from transformers.models.big_bird.modeling_big_bird.BigBirdBlockSparseAttention with BigBird->BigBirdPegasus
# 定义了 BigBirdPegasusBlockSparseAttention 类，继承自 nn.Module
class BigBirdPegasusBlockSparseAttention(nn.Module):
    def __init__(self, config, seed=None):
        super().__init__()

        self.max_seqlen = config.max_position_embeddings
        self.seed = seed

        # 检查隐藏大小是否是注意力头数的整数倍，否则抛出 ValueError 异常
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.num_random_blocks = config.num_random_blocks
        self.block_size = config.block_size

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义用于查询、键、值的线性变换层，输入大小为隐藏大小，输出大小为所有头的大小
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
    def transpose_for_scores(self, x):
        # 计算转置后张量的新形状，以便用于注意力计算
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 对输入张量进行形状变换
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
        output_attentions=None,
    ):
        # 当前这个类不能用于解码器

        # 获取隐藏状态张量的维度信息
        batch_size, seqlen, _ = hidden_states.size()
        to_seq_length = from_seq_length = seqlen
        from_block_size = to_block_size = self.block_size

        # 检查查询侧序列长度是否是块大小的倍数
        if from_seq_length % from_block_size != 0:
            raise ValueError("Query sided sequence length must be multiple of block size")

        # 检查键/值侧序列长度是否是块大小的倍数
        if to_seq_length % to_block_size != 0:
            raise ValueError("Key/Value sided sequence length must be multiple of block size")

        # 对查询、键、值进行转置，以便进行注意力计算
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 调用自定义的大鸟块稀疏注意力机制
        context_layer, attention_probs = self.bigbird_block_sparse_attention(
            query_layer,
            key_layer,
            value_layer,
            band_mask,
            from_mask,
            to_mask,
            from_blocked_mask,
            to_blocked_mask,
            self.num_attention_heads,
            self.num_random_blocks,
            self.attention_head_size,
            from_block_size,
            to_block_size,
            batch_size,
            from_seq_length,
            to_seq_length,
            seed=self.seed,
            plan_from_length=None,
            plan_num_rand_blocks=None,
            output_attentions=output_attentions,
        )

        # 将上下文层重新变形为原始形状
        context_layer = context_layer.contiguous().view(batch_size, from_seq_length, -1)

        # 如果需要输出注意力权重，将其包含在输出中
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    @staticmethod
    def torch_bmm_nd(inp_1, inp_2, ndim=None):
        """快速的多维矩阵乘法"""
        # 使用torch.bmm更快地实现torch.einsum ("bhqk,bhkd->bhqd")的功能
        return torch.bmm(inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:])).view(
            inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 1])
        )

    @staticmethod
    def torch_bmm_nd_transpose(inp_1, inp_2, ndim=None):
        """带转置的快速多维矩阵乘法"""
        # 使用torch.bmm更快地实现torch.einsum ("bhqd,bhkd->bhqk")的功能
        return torch.bmm(
            inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:]).transpose(1, 2)
        ).view(inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 2]))
    @staticmethod
    def bigbird_block_sparse_attention(
        self,
        query_layer,
        key_layer,
        value_layer,
        band_mask,
        from_mask,
        to_mask,
        from_blocked_mask,
        to_blocked_mask,
        n_heads,
        n_rand_blocks,
        attention_head_size,
        from_block_size,
        to_block_size,
        batch_size,
        from_seq_len,
        to_seq_len,
        seed,
        plan_from_length,
        plan_num_rand_blocks,
        output_attentions,
    ):
        # 实现BigBird模型中的稀疏块注意力机制
        # 参数说明:
        # - query_layer, key_layer, value_layer: 查询、键、值的张量
        # - band_mask: 带状掩码，限制注意力只在一定带状范围内
        # - from_mask, to_mask: 来源和目标的掩码，限制注意力的有效范围
        # - from_blocked_mask, to_blocked_mask: 分块的掩码，用于分块注意力机制
        # - n_heads: 注意力头的数量
        # - n_rand_blocks: 随机块的数量
        # - attention_head_size: 注意力头的尺寸
        # - from_block_size, to_block_size: 来源和目标块的尺寸
        # - batch_size: 批次大小
        # - from_seq_len, to_seq_len: 来源和目标序列的长度
        # - seed: 随机种子
        # - plan_from_length: 计划的来源长度
        # - plan_num_rand_blocks: 计划的随机块数量
        # - output_attentions: 是否输出注意力权重

        # 实现tf.gather类似的torch版本的功能，当batch_dims=2时
    @staticmethod
    def torch_gather_b2(params, indices):
        # 此操作相当于tf.gather，当batch_dims=2时

        if params.shape[:2] != indices.shape[:2]:
            raise ValueError(
                "Make sure that the first two dimensions of params and indices are identical, but"
                f" they are params: {params.shape[:2]} vs. indices: {indices.shape[:2]}"
            )
        num_indices_to_gather = indices.shape[-2] * indices.shape[-1]
        num_indices_to_pick_from = params.shape[2]

        shift = torch.arange(indices.shape[0] * indices.shape[1] * num_indices_to_gather, device=indices.device)
        indices_shift = torch.div(shift, num_indices_to_gather, rounding_mode="floor") * num_indices_to_pick_from

        flattened_indices = indices.view(-1) + indices_shift
        flattened_params = params.reshape(-1, params.shape[-2], params.shape[-1])

        out_flattened = flattened_params.index_select(0, flattened_indices)

        out = out_flattened.reshape(params.shape[:2] + (num_indices_to_gather,) + params.shape[3:])
        return out

    @staticmethod
    def _create_rand_mask_from_inputs(
        from_blocked_mask,
        to_blocked_mask,
        rand_attn,
        num_attention_heads,
        num_rand_blocks,
        batch_size,
        from_seq_length,
        from_block_size,
    ):
        """
        Create 3D attention mask from a 2D tensor mask.

        Args:
            from_blocked_mask: 2D Tensor of shape [batch_size,
                from_seq_length//from_block_size, from_block_size].
                输入的来自的序列被块化后的掩码，形状为 [batch_size, from_seq_length//from_block_size, from_block_size]。
            to_blocked_mask: int32 Tensor of shape [batch_size,
                to_seq_length//to_block_size, to_block_size].
                输入的目标序列被块化后的掩码，形状为 [batch_size, to_seq_length//to_block_size, to_block_size]。
            rand_attn: [batch_size, num_attention_heads,
                from_seq_length//from_block_size-2, num_rand_blocks]
                随机注意力的掩码，形状为 [batch_size, num_attention_heads, from_seq_length//from_block_size-2, num_rand_blocks]。
            num_attention_heads: int. Number of attention heads.
                注意力头的数量。
            num_rand_blocks: int. Number of random chunks per row.
                每行的随机块数。
            batch_size: int. Batch size for computation.
                计算的批次大小。
            from_seq_length: int. length of from sequence.
                输入序列的长度。
            from_block_size: int. size of block in from sequence.
                输入序列中的块大小。

        Returns:
            float Tensor of shape [batch_size, num_attention_heads, from_seq_length//from_block_size-2,
                from_block_size, num_rand_blocks*to_block_size].
            返回形状为 [batch_size, num_attention_heads, from_seq_length//from_block_size-2,
                from_block_size, num_rand_blocks*to_block_size] 的浮点数张量。
        """
        num_windows = from_seq_length // from_block_size - 2
        # 根据输入序列的块大小计算窗口数
        rand_mask = torch.stack([p1[i1.flatten()] for p1, i1 in zip(to_blocked_mask, rand_attn)])
        # 使用目标序列的掩码和随机注意力创建随机掩码
        rand_mask = rand_mask.view(batch_size, num_attention_heads, num_windows, num_rand_blocks * from_block_size)
        # 通过 einsum 操作组合来自序列的掩码和随机掩码
        rand_mask = torch.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1], rand_mask)
        return rand_mask

    @staticmethod
    def _get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
        """
        Gives the plan of where to put random attention.

        Args:
            from_seq_length: int. length of from sequence.
                输入序列的长度。
            from_block_size: int. size of block in from sequence.
                输入序列中的块大小。
            num_rand_blocks: int. Number of random chunks per row.
                每行的随机块数。

        Returns:
            plan_from_length: ending location of from block plan_num_rand_blocks: number of random ending location for
                each block
            返回计划的输入序列块结束位置和每个块的随机结束位置的计划。
        """
        plan_from_length = []
        plan_num_rand_blocks = []
        if (2 * num_rand_blocks + 5) < (from_seq_length // from_block_size):
            plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(0)
        elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
            plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
            plan_num_rand_blocks.append(num_rand_blocks // 2)
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks // 2))
        else:
            plan_from_length.append(from_seq_length)
            plan_num_rand_blocks.append(num_rand_blocks)

        return plan_from_length, plan_num_rand_blocks

    def _bigbird_block_rand_mask(
        self, from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, last_idx=-1
    ):
        """
        Create adjacency list of random attention.

        Args:
            from_seq_length: int. length of from sequence.
            to_seq_length: int. length of to sequence.
            from_block_size: int. size of block in from sequence.
            to_block_size: int. size of block in to sequence.
            num_rand_blocks: int. Number of random chunks per row.
            last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
            if positive then num_rand_blocks blocks chosen only up to last_idx.

        Returns:
            adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
        """
        # using this method when from_seq_length in [1024, 3072, 4096]

        # 检查是否 from_seq_length 和 to_seq_length 的块数相等，否则抛出异常
        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError("Error the number of blocks needs to be same!")

        # 创建一个全零数组，表示随机注意力的邻接列表
        rand_attn = np.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)

        # 推理阶段（非训练状态），直接返回全零的随机注意力邻接列表
        if not self.training:
            return rand_attn

        # 创建中间序列，用于生成随机块索引
        middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
        last = to_seq_length // to_block_size - 1

        # 根据 last_idx 的值确定最后一个块的索引范围
        if last_idx > (2 * to_block_size):
            last = (last_idx // to_block_size) - 1

        r = num_rand_blocks  # 缩写 r 表示 num_rand_blocks

        # 循环创建每行的随机注意力邻接列表
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i

            if i == 1:
                # 对第一行进行随机排列选择中间序列中的块索引
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
            elif i == 2:
                # 对第二行进行随机排列选择中间序列中的块索引
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
            elif i == from_seq_length // from_block_size - 3:
                # 对倒数第三行进行随机排列选择中间序列中的块索引
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            elif i == from_seq_length // from_block_size - 2:
                # 对倒数第二行进行随机排列选择中间序列中的块索引
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            else:
                if start > last:
                    start = last
                    # 如果起始大于最后一个块的索引，则选择中间序列中的前 start 个块索引进行随机排列
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                elif (end + 1) == last:
                    # 如果结束索引的下一个等于最后一个块的索引，则选择中间序列中的前 start 个块索引进行随机排列
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                else:
                    # 否则，选择中间序列中除了指定的 start 和 end 块索引外的其余块索引进行随机排列
                    rand_attn[i - 1, :] = np.random.permutation(
                        np.concatenate((middle_seq[:start], middle_seq[end + 1 : last]))
                    )[:r]

        # 返回生成的随机注意力邻接列表
        return rand_attn
    def _bigbird_block_rand_mask_with_head(
        self,
        from_seq_length,
        to_seq_length,
        from_block_size,
        to_block_size,
        num_heads,
        plan_from_length,
        plan_num_rand_blocks,
        window_block_left=1,
        window_block_right=1,
        global_block_top=1,
        global_block_bottom=1,
        global_block_left=1,
        global_block_right=1,
    ):
        """
        Generates a random mask for BigBird attention with head.

        Args:
            from_seq_length: int. Length of the source sequence.
            to_seq_length: int. Length of the target sequence.
            from_block_size: int. Block size of the source sequence.
            to_block_size: int. Block size of the target sequence.
            num_heads: int. Number of attention heads.
            plan_from_length: int. Planned length of the source sequence.
            plan_num_rand_blocks: int. Planned number of random blocks.
            window_block_left: int. Number of blocks of window to the left of a block.
            window_block_right: int. Number of blocks of window to the right of a block.
            global_block_top: int. Number of blocks globally used at the top.
            global_block_bottom: int. Number of blocks globally used at the bottom.
            global_block_left: int. Number of blocks globally used to the left.
            global_block_right: int. Number of blocks globally used to the right.

        Returns:
            Random mask with head for BigBird attention.
        """
        # Implementation of random mask generation for BigBird attention
        pass


    @staticmethod
    def _get_single_block_row_attention(
        block_id,
        to_start_block_id,
        to_end_block_id,
        num_rand_blocks,
        window_block_left=1,
        window_block_right=1,
        global_block_left=1,
        global_block_right=1,
    ):
        """
        For a single row block, get random row attention.

        Args:
            block_id: int. Block ID of the row.
            to_start_block_id: int. Start ID of the target blocks for random attention.
            to_end_block_id: int. End ID of the target blocks for random attention.
            num_rand_blocks: int. Number of random blocks to be selected.
            window_block_left: int. Number of blocks of window to the left of a block.
            window_block_right: int. Number of blocks of window to the right of a block.
            global_block_left: int. Number of blocks globally used to the left.
            global_block_right: int. Number of blocks globally used to the right.

        Returns:
            Array containing the selected random attention vector of size num_rand_blocks.
        """
        # List of to_blocks from which to choose random attention
        to_block_list = np.arange(to_start_block_id, to_end_block_id, dtype=np.int32)
        # Permute the blocks
        perm_block = np.random.permutation(to_block_list)

        # Illegal blocks for the current block id, using window
        illegal_blocks = list(range(block_id - window_block_left, block_id + window_block_right + 1))

        # Add blocks at the start and at the end
        illegal_blocks.extend(list(range(global_block_left)))
        illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))

        # The second from_block cannot choose random attention on second last to_block
        if block_id == 1:
            illegal_blocks.append(to_end_block_id - 2)

        # The second last from_block cannot choose random attention on second to_block
        if block_id == to_end_block_id - 2:
            illegal_blocks.append(1)

        selected_random_blocks = []

        for i in range(to_end_block_id - to_start_block_id):
            if perm_block[i] not in illegal_blocks:
                selected_random_blocks.append(perm_block[i])
            if len(selected_random_blocks) == num_rand_blocks:
                break
        return np.array(selected_random_blocks, dtype=np.int32)
# 定义 BigBirdPegasusEncoderAttention 类，继承自 nn.Module，用于编码器部分的注意力机制
class BigBirdPegasusEncoderAttention(nn.Module):
    # 初始化方法，接受配置参数 config 和种子参数 seed（可选）
    def __init__(self, config, seed=None):
        super().__init__()
        # 将配置参数 config 和种子参数 seed 存储在实例中
        self.config = config
        self.seed = seed

        # 从配置中获取注意力类型并存储在实例变量中
        self.attention_type = config.attention_type

        # 根据不同的注意力类型选择对应的注意力模块
        if self.attention_type == "original_full":
            self.self = BigBirdPegasusSelfAttention(config)
        elif self.attention_type == "block_sparse":
            self.self = BigBirdPegasusBlockSparseAttention(config, seed)
        else:
            # 如果注意力类型不是预期的值，抛出 ValueError 异常
            raise ValueError(
                f"attention_type can either be original_full or block_sparse, but is {self.config.attention_type}"
            )

        # 定义输出层，将隐藏状态映射回原始维度
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)

    # 设置注意力类型的方法，接受字符串类型的 value 参数
    def set_attention_type(self, value: str):
        # 如果 value 不在允许的类型列表中，则抛出 ValueError 异常
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        
        # 如果 value 和当前注意力类型相同，则不做任何操作直接返回
        if value == self.attention_type:
            return

        # 将实例的 attention_type 设置为新的 value
        self.attention_type = value
        
        # 根据新的 attention_type 重新设置 self.self 对象
        if value == "original_full":
            # 复制所有权重到新的完全注意力类
            attn_weights = BigBirdPegasusSelfAttention(self.config)
        else:
            # 复制所有权重到新的稀疏注意力类
            attn_weights = BigBirdPegasusBlockSparseAttention(self.config, self.seed)

        # 将当前 self.self 的 query、value、key 属性复制到新的 attn_weights 对象
        attn_weights.query = self.self.query
        attn_weights.value = self.self.value
        attn_weights.key = self.self.key
        
        # 更新实例的 self.self 为新的 attn_weights
        self.self = attn_weights
        
        # 同时更新实例的 attention_type
        self.attention_type = value

        # 如果不处于训练模式，则将 self.self 设为评估状态
        if not self.training:
            self.self.eval()

    # 前向传播方法，接受多个输入参数并返回输出
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_value=None,
        output_attentions=False,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
    ):
        # 如果 head_mask 不为 None，则将其扩展一个维度以便在自注意力模块中进行乘法操作
        head_mask = head_mask.reshape(1, -1, 1, 1) if head_mask is not None else None

        # 根据配置中的 attention_type 选择不同的 self.self 模块进行计算
        if self.config.attention_type == "original_full":
            # 使用完全注意力模块进行计算
            self_outputs = self.self(
                hidden_states,
                attention_mask,
                head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
        else:
            # 使用稀疏注意力模块进行计算
            self_outputs = self.self(
                hidden_states, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, output_attentions
            )

        # 将注意力输出通过输出层映射回原始维度
        attention_output = self.output(self_outputs[0])
        
        # 如果需要输出注意力矩阵，则在输出元组中包含它们
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要，添加注意力矩阵到输出元组中
        return outputs

# 从 transformers.models.bart.modeling_bart.BartAttention 复制并修改为 BigBirdPegasusDecoderConfig->BigBirdPegasusConfig, Bart->BigBirdPegasusDecoder
class BigBirdPegasusDecoderAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[BigBirdPegasusConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim  # 设置注意力机制的嵌入维度
        self.num_heads = num_heads  # 设置注意力头的数量
        self.dropout = dropout  # 设置dropout率
        self.head_dim = embed_dim // num_heads  # 计算每个注意力头的维度
        self.config = config  # 设置配置参数

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5  # 缩放因子，用于调整注意力分布
        self.is_decoder = is_decoder  # 是否为解码器
        self.is_causal = is_causal  # 是否是因果的

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 线性变换k
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 线性变换v
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 线性变换q
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 输出变换

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # 将输入张量reshape为(batch_size, seq_len, num_heads, head_dim)的形状，并进行维度转置和连续化处理

    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        key_value_states: Optional[torch.Tensor] = None,  # 键值对状态张量（可选）
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # 过去的键值对（可选）
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码（可选）
        layer_head_mask: Optional[torch.Tensor] = None,  # 层级头掩码（可选）
        output_attentions: bool = False,  # 是否输出注意力权重
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states  # 保存输入 hidden_states 作为残差连接的基础

        hidden_states = self.self_attn_layer_norm(hidden_states)  # 对输入 hidden_states 进行层归一化

        # 使用 self-attention 模块处理归一化后的 hidden_states
        self_attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=layer_head_mask,
            output_attentions=output_attentions,
            band_mask=band_mask,
            from_mask=from_mask,
            to_mask=to_mask,
            from_blocked_mask=from_blocked_mask,
            to_blocked_mask=to_blocked_mask,
        )
        hidden_states = self_attention_outputs[0]  # 更新 hidden_states 为 self-attention 的输出结果

        # 对 hidden_states 进行 dropout 处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # 残差连接：将残差加回到处理后的 hidden_states
        hidden_states = residual + hidden_states

        residual = hidden_states  # 更新残差连接的基础为当前的 hidden_states

        hidden_states = self.final_layer_norm(hidden_states)  # 对 hidden_states 进行最终的层归一化
        hidden_states = self.activation_fn(self.fc1(hidden_states))  # 经过激活函数和第一个全连接层处理

        hidden_states = self.fc2(hidden_states)  # 第二个全连接层处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # dropout 处理

        # 残差连接：将残差加回到处理后的 hidden_states
        hidden_states = residual + hidden_states

        # 如果 hidden_states 的数据类型为 torch.float16，并且包含无穷大或 NaN 值，则进行数值截断处理
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)  # 将最终的 hidden_states 打包为输出元组

        if output_attentions:
            outputs += (self_attention_outputs[1],)  # 如果需要返回 attentions，则将 attentions 加入输出元组

        return outputs  # 返回最终输出元组，包含处理后的 hidden_states 和可能的 attentions

    def set_attention_type(self, value: str):
        if value not in ["original_full", "block_sparse"]:  # 检查输入值是否合法
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # 如果 attention_type 已经正确设置，则直接返回，无需修改
        if value == self.attention_type:
            return
        self.attention_type = value  # 更新 attention_type 为新的值
        self.self_attn.set_attention_type(value)  # 更新 self-attention 模块的 attention_type
# 定义 BigBirdPegasusDecoderLayer 类，继承自 nn.Module
class BigBirdPegasusDecoderLayer(nn.Module):
    
    # 初始化方法，接受一个 BigBirdPegasusConfig 类型的参数 config
    def __init__(self, config: BigBirdPegasusConfig):
        super().__init__()
        
        # 设置 embed_dim 属性为 config.d_model
        self.embed_dim = config.d_model
        
        # 创建 BigBirdPegasusDecoderAttention 对象并赋给 self.self_attn 属性
        self.self_attn = BigBirdPegasusDecoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.use_bias,
        )
        
        # 设置 dropout 属性为 config.dropout
        self.dropout = config.dropout
        
        # 根据配置中的激活函数名称获取相应的激活函数，并赋给 self.activation_fn 属性
        self.activation_fn = ACT2FN[config.activation_function]
        
        # 设置 activation_dropout 属性为 config.activation_dropout
        self.activation_dropout = config.activation_dropout

        # 创建 nn.LayerNorm 对象并赋给 self.self_attn_layer_norm 属性
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # 创建 BigBirdPegasusDecoderAttention 对象并赋给 self.encoder_attn 属性
        self.encoder_attn = BigBirdPegasusDecoderAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.use_bias,
        )
        
        # 创建 nn.LayerNorm 对象并赋给 self.encoder_attn_layer_norm 属性
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # 创建 nn.Linear 对象并赋给 self.fc1 属性，用于第一个全连接层
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        
        # 创建 nn.Linear 对象并赋给 self.fc2 属性，用于第二个全连接层
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        
        # 创建 nn.LayerNorm 对象并赋给 self.final_layer_norm 属性
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 定义 forward 方法，执行模型的前向传播
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
        # 略
        pass


# 定义 BigBirdPegasusClassificationHead 类，继承自 nn.Module
class BigBirdPegasusClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    # 初始化方法，接受 input_dim、inner_dim、num_classes、pooler_dropout 四个参数
    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        
        # 创建 nn.Linear 对象并赋给 self.dense 属性，用于密集连接层
        self.dense = nn.Linear(input_dim, inner_dim)
        
        # 创建 nn.Dropout 对象并赋给 self.dropout 属性，用于 dropout 操作
        self.dropout = nn.Dropout(p=pooler_dropout)
        
        # 创建 nn.Linear 对象并赋给 self.out_proj 属性，用于最终的线性变换
        self.out_proj = nn.Linear(inner_dim, num_classes)

    # 定义 forward 方法，执行模型的前向传播
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 应用 dropout 操作到 hidden_states
        hidden_states = self.dropout(hidden_states)
        
        # 通过全连接层 self.dense 进行线性变换
        hidden_states = self.dense(hidden_states)
        
        # 应用 tanh 激活函数
        hidden_states = torch.tanh(hidden_states)
        
        # 再次应用 dropout 操作
        hidden_states = self.dropout(hidden_states)
        
        # 通过最终的线性变换 self.out_proj 得到最终输出
        hidden_states = self.out_proj(hidden_states)
        
        # 返回最终的输出张量
        return hidden_states


# 定义 BigBirdPegasusPreTrainedModel 类，继承自 PreTrainedModel
class BigBirdPegasusPreTrainedModel(PreTrainedModel):
    
    # 设置 config_class 属性为 BigBirdPegasusConfig 类
    config_class = BigBirdPegasusConfig
    
    # 设置 base_model_prefix 属性为 "model"
    base_model_prefix = "model"
    
    # 设置 supports_gradient_checkpointing 属性为 True
    supports_gradient_checkpointing = True
    
    # 设置 _no_split_modules 属性为 ["BigBirdPegasusEncoderLayer", "BigBirdPegasusDecoderLayer"]
    _no_split_modules = ["BigBirdPegasusEncoderLayer", "BigBirdPegasusDecoderLayer"]
    
    # 设置 _skip_keys_device_placement 属性为 "past_key_values"
    _skip_keys_device_placement = "past_key_values"
    # 初始化模块的权重，根据模块类型设置不同的初始化标准差
    def _init_weights(self, module):
        std = self.config.init_std
        # 如果是线性层模块
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层模块
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在填充索引，则将其对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    # 返回虚拟的输入数据，用于模型测试
    def dummy_inputs(self):
        # 获取填充标记的 ID
        pad_token = self.config.pad_token_id
        # 创建输入 ID 张量，包含两个示例句子的 ID 序列
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        # 构建虚拟输入字典，包括注意力掩码和输入 ID
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),  # 注意力掩码表示哪些位置是填充的
            "input_ids": input_ids,  # 实际输入的 ID 序列
        }
        # 返回虚拟输入字典
        return dummy_inputs
BIGBIRD_PEGASUS_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BigBirdPegasusConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BIGBIRD_PEGASUS_GENERATION_EXAMPLE = r"""
    Summarization example:

    ```
    >>> from transformers import AutoTokenizer, BigBirdPegasusForConditionalGeneration

    >>> model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-arxiv")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv")

    >>> ARTICLE_TO_SUMMARIZE = (
    ...     "The dominant sequence transduction models are based on complex recurrent or convolutional neural "
    ...     "networks in an encoder-decoder configuration. The best performing models also connect the encoder "
    ...     "and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, "
    ...     "based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. "
    ...     "Experiments on two machine translation tasks show these models to be superior in quality "
    ...     "while being more parallelizable and requiring significantly less time to train."
    ... )
    >>> inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=4096, return_tensors="pt", truncation=True)

    >>> # Generate Summary
    >>> summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=15)
    >>> tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    'dominant sequence models are based on recurrent or convolutional neural networks .'
    ```
"""

BIGBIRD_PEGASUS_INPUTS_DOCSTRING = r"""
    Placeholder for documenting inputs for BigBirdPegasus models.
"""

BIGBIRD_PEGASUS_STANDALONE_INPUTS_DOCSTRING = r"""
    Placeholder for documenting standalone inputs for BigBirdPegasus models.
"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。默认情况下，忽略填充标记。

            # 可以使用 `ProphetNetTokenizer` 来获取这些索引。详见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`。

            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免对填充标记进行注意力计算的掩码张量。掩码值在 `[0, 1]` 范围内：

            # - 1 表示 **未被掩码** 的标记，
            # - 0 表示 **被掩码** 的标记。

            # [什么是注意力掩码？](../glossary#attention-mask)
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详见返回的张量中的 `attentions` 获取更多细节。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详见返回的张量中的 `hidden_states` 获取更多细节。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BigBirdPegasusEncoderLayer`].

    Args:
        config: BigBirdPegasusConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BigBirdPegasusConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.attention_type = config.attention_type  # 从配置中获取注意力类型
        self.block_size = config.block_size  # 从配置中获取块大小

        self.dropout = config.dropout  # 从配置中获取 dropout 率
        self.layerdrop = config.encoder_layerdrop  # 从配置中获取层级 dropout 率

        embed_dim = config.d_model  # 从配置中获取嵌入维度
        self.padding_idx = config.pad_token_id  # 从配置中获取填充标识符
        self.max_source_positions = config.max_position_embeddings  # 从配置中获取最大位置嵌入
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0  # 根据配置设置嵌入缩放因子

        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)  # 初始化嵌入层

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight  # 如果提供了外部嵌入，则使用其权重

        self.embed_positions = BigBirdPegasusLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )  # 初始化位置嵌入

        self.layers = nn.ModuleList([BigBirdPegasusEncoderLayer(config, seed=i) for i in range(config.encoder_layers)])
        # 创建多层编码器层，并存储在模块列表中

        self.layernorm_embedding = nn.LayerNorm(embed_dim)  # 初始化嵌入层归一化层

        self.gradient_checkpointing = False  # 梯度检查点设置为 False
        # 初始化权重并应用最终处理
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        pass  # 此处为前向传播函数的占位符，实际执行模型推理过程

    def set_attention_type(self, value: str):
        if value not in ["original_full", "block_sparse"]:  # 检查传入的注意力类型是否合法
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # 如果注意力类型已经正确设置，则直接返回
        if value == self.attention_type:
            return
        self.attention_type = value  # 更新注意力类型为新值
        for layer in self.layers:
            layer.set_attention_type(value)  # 更新每个编码器层的注意力类型

    @staticmethod  # 静态方法，用于生成块稀疏注意力的掩码，从 Transformers 源代码复制而来
    # transformers.models.big_bird.modeling_big_bird.BigBirdModel.create_masks_for_block_sparse_attn
    def create_masks_for_block_sparse_attn(attention_mask: torch.Tensor, block_size: int):
        batch_size, seq_length = attention_mask.size()
        # 检查序列长度是否是块大小的倍数，如果不是则抛出异常
        if seq_length % block_size != 0:
            raise ValueError(
                f"Sequence length must be multiple of block size, but sequence length is {seq_length}, while block"
                f" size is {block_size}."
            )

        def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
            """
            从二维张量掩码创建三维注意力掩码。

            Args:
                from_blocked_mask: 形状为 [batch_size, from_seq_length//from_block_size, from_block_size] 的二维张量掩码。
                to_blocked_mask: 形状为 [batch_size, to_seq_length//to_block_size, to_block_size] 的整数张量掩码。

            Returns:
                形状为 [batch_size, 1, from_seq_length//from_block_size-4, from_block_size, 3*to_block_size] 的浮点张量。
            """
            # 构造用于填充的扩展阻塞掩码
            exp_blocked_to_pad = torch.cat(
                [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], dim=2
            )
            # 使用 Einstein Summation Notation 创建带状掩码
            band_mask = torch.einsum("blq,blk->blqk", from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
            band_mask.unsqueeze_(1)
            return band_mask

        # 将注意力掩码视图重新形状为块大小的块编码器掩码
        blocked_encoder_mask = attention_mask.view(batch_size, seq_length // block_size, block_size)
        # 创建带状掩码
        band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)

        # 创建来自掩码和去掩码
        from_mask = attention_mask.view(batch_size, 1, seq_length, 1)
        to_mask = attention_mask.view(batch_size, 1, 1, seq_length)

        return blocked_encoder_mask, band_mask, from_mask, to_mask

    def _pad_to_block_size(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        """A helper function to pad tokens and mask to work with implementation of BigBird block-sparse attention."""
        # 填充函数，用于与 BigBird 块稀疏注意力实现一起工作的辅助函数
        # 填充
        block_size = self.config.block_size
        batch_size, seq_len = hidden_states.shape[:2]

        padding_len = (block_size - seq_len % block_size) % block_size
        if padding_len > 0:
            # 如果需要填充，警告并自动填充输入 ID 和嵌入到块大小的倍数
            logger.warning_once(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.block_size`: {block_size}"
            )
            pad_id = self.config.pad_token_id
            device = hidden_states.device
            input_ids_padding = torch.ones((batch_size, padding_len), dtype=torch.long, device=device) * pad_id
            inputs_embeds_padding = self.embed_tokens(input_ids_padding)
            hidden_states = torch.cat([hidden_states, inputs_embeds_padding], dim=-2)

            # 使用 nn.functional.pad 对注意力掩码进行填充，填充部分的注意力为0
            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=0
            )  # no attention on the padding tokens

        return padding_len, hidden_states, attention_mask
class BigBirdPegasusDecoder(BigBirdPegasusPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BigBirdPegasusDecoderLayer`]

    Args:
        config: BigBirdPegasusConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BigBirdPegasusConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout  # 从配置中获取 dropout 概率
        self.layerdrop = config.decoder_layerdrop  # 从配置中获取层级 dropout 概率
        self.padding_idx = config.pad_token_id  # 从配置中获取填充标记的索引
        self.max_target_positions = config.max_position_embeddings  # 从配置中获取最大目标位置数
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0  # 根据配置计算嵌入尺度

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)  # 初始化词嵌入层

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight  # 如果提供了预训练的嵌入层，则使用它

        self.embed_positions = BigBirdPegasusLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )  # 初始化位置编码器

        self.layers = nn.ModuleList([BigBirdPegasusDecoderLayer(config) for _ in range(config.decoder_layers)])  # 创建多层解码器层
        self.layernorm_embedding = nn.LayerNorm(config.d_model)  # 应用层归一化到嵌入层

        self.gradient_checkpointing = False  # 初始化梯度检查点

        # Initialize weights and apply final processing
        self.post_init()  # 执行初始化权重和最终处理



@add_start_docstrings(
    "The bare BigBirdPegasus Model outputting raw hidden-states without any specific head on top.",
    BIGBIRD_PEGASUS_START_DOCSTRING,
)
class BigBirdPegasusModel(BigBirdPegasusPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BigBirdPegasusConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)  # 初始化共享的嵌入层

        self.encoder = BigBirdPegasusEncoder(config, self.shared)  # 创建BigBirdPegasus编码器，使用共享嵌入
        self.decoder = BigBirdPegasusDecoder(config, self.shared)  # 创建BigBirdPegasus解码器，使用共享嵌入

        # Initialize weights and apply final processing
        self.post_init()  # 执行初始化权重和最终处理
    # 返回输入的共享输入嵌入
    def get_input_embeddings(self):
        return self.shared

    # 设置共享输入嵌入，并更新编码器和解码器的嵌入
    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # 如果配置要求词嵌入共享，则绑定编码器和解码器的嵌入权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 返回编码器对象
    def get_encoder(self):
        return self.encoder

    # 返回解码器对象
    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(BIGBIRD_PEGASUS_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 从 transformers.models.bart.modeling_bart.BartModel.forward 复制的代码，并将 Bart 替换为 BigBirdPegasus
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用装饰器为类添加文档字符串，描述了 BigBirdPegasusForConditionalGeneration 模型的用途和摘要功能
@add_start_docstrings(
    "The BigBirdPegasus Model with a language modeling head. Can be used for summarization.",
    BIGBIRD_PEGASUS_START_DOCSTRING,
)
# 从 transformers.models.bart.modeling_bart.BartForConditionalGeneration 复制代码，并将 Bart 替换为 BigBirdPegasus，BART 替换为 BIGBIRD_PEGASUS
class BigBirdPegasusForConditionalGeneration(BigBirdPegasusPreTrainedModel):
    # 设置模型主体的前缀为 "model"
    base_model_prefix = "model"
    # 定义在加载过程中需要忽略的键名列表，这些键名对应缺失时不会引发警告
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    # 定义加载时忽略的关键字列表，指定不会加载的额外逻辑
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]

    # 初始化函数，接受 BigBirdPegasusConfig 类型的配置对象
    def __init__(self, config: BigBirdPegasusConfig):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        # 创建 BigBirdPegasusModel 实例并赋值给 self.model
        self.model = BigBirdPegasusModel(config)
        # 注册一个缓冲区，初始化为全零向量，维度是 (1, self.model.shared.num_embeddings)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # 创建一个线性层，作为语言模型头，输入大小为 config.d_model，输出大小为 self.model.shared.num_embeddings，不使用偏置
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取编码器部分的方法，返回 self.model 的编码器
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器部分的方法，返回 self.model 的解码器
    def get_decoder(self):
        return self.model.get_decoder()

    # 调整 token embeddings 大小的方法，返回新的嵌入层对象
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        # 调用父类的 resize_token_embeddings 方法，返回新的嵌入层对象
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 调整 final_logits_bias 的大小以匹配新的 token 数量
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    # 调整 final_logits_bias 大小的私有方法，不返回任何内容
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        # 获取旧的 token 数量
        old_num_tokens = self.final_logits_bias.shape[-1]
        # 如果新的 token 数量小于等于旧的 token 数量，则截取 final_logits_bias
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        # 如果新的 token 数量大于旧的 token 数量，则在最后增加零偏置
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        # 注册新的 final_logits_bias 缓冲区
        self.register_buffer("final_logits_bias", new_bias)

    # 获取输出嵌入层的方法，返回 self.lm_head
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层的方法，接受新的嵌入层作为参数
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 使用装饰器添加文档字符串，描述了 model_forward 方法的输入和输出
    @add_start_docstrings_to_model_forward(BIGBIRD_PEGASUS_INPUTS_DOCSTRING)
    # 替换返回值的文档字符串，指定输出类型为 Seq2SeqLMOutput，配置类为 _CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 添加结尾的文档字符串，提供 BIGBIRD_PEGASUS_GENERATION_EXAMPLE 的生成示例
    @add_end_docstrings(BIGBIRD_PEGASUS_GENERATION_EXAMPLE)
    # 定义模型的前向传播方法，用于执行推断或训练过程中的正向计算
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入序列的token IDs，类型为长整型张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选的张量类型
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入的token IDs，可选的长整型张量
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器注意力掩码，可选的长整型张量
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，可选的张量
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器头部掩码，可选的张量
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 跨注意力头部掩码，可选的张量
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,  # 编码器输出的列表，包含浮点张量
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对列表，包含浮点张量
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入，可选的浮点张量
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入嵌入，可选的浮点张量
        labels: Optional[torch.LongTensor] = None,  # 标签，可选的长整型张量
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果，可选的布尔值
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Return type annotation indicating the function returns either a tuple or `Seq2SeqLMOutput`.

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # If labels are provided, adjust `use_cache` behavior and prepare `decoder_input_ids`
        if labels is not None:
            if use_cache:
                # Warn if `use_cache` is `True` because `labels` are provided; set `use_cache` to `False`
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            # If `decoder_input_ids` are not provided, shift `labels` for decoder inputs
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Pass the input arguments to the model for processing
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Generate logits for language modeling head and adjust with final bias
        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        # Compute masked language modeling loss if labels are provided
        if labels is not None:
            labels = labels.to(lm_logits.device)  # Ensure labels are on the same device as logits
            loss_fct = CrossEntropyLoss()  # Define the loss function
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # If `return_dict` is `False`, return outputs as a tuple
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # If `return_dict` is `True`, return structured `Seq2SeqLMOutput`
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用了过去的键值（past_key_values），则根据其长度修剪 decoder_input_ids
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 有些生成方法已经只传递了最后一个输入 ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认旧行为：仅保留最后一个 ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回准备好的输入字典，用于生成
        return {
            "input_ids": None,  # encoder_outputs 已定义，input_ids 不需要
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 更改此处以避免缓存（推测是为了调试）
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 将标签右移一个位置，以准备解码器的输入
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态无需重新排序 -> 它们始终相同
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        # 返回重新排序后的过去键值
        return reordered_past
@add_start_docstrings(
    """
    BigBirdPegasus model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g.
    for GLUE tasks.
    """,
    BIGBIRD_PEGASUS_START_DOCSTRING,
)
class BigBirdPegasusForSequenceClassification(BigBirdPegasusPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BigBirdPegasusConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BigBirdPegasusModel(config)
        self.classification_head = BigBirdPegasusClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BIGBIRD_PEGASUS_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # Copied from transformers.models.bart.modeling_bart.BartForSequenceClassification.forward
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass for the BigBirdPegasusForSequenceClassification model.

        Args:
            input_ids (torch.LongTensor, optional): Input token IDs. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            decoder_input_ids (torch.LongTensor, optional): Decoder input token IDs. Defaults to None.
            decoder_attention_mask (torch.LongTensor, optional): Decoder attention mask. Defaults to None.
            head_mask (torch.Tensor, optional): Head mask. Defaults to None.
            decoder_head_mask (torch.Tensor, optional): Decoder head mask. Defaults to None.
            cross_attn_head_mask (torch.Tensor, optional): Cross-attention head mask. Defaults to None.
            encoder_outputs (List[torch.FloatTensor], optional): Encoder outputs. Defaults to None.
            inputs_embeds (torch.FloatTensor, optional): Embedded inputs. Defaults to None.
            decoder_inputs_embeds (torch.FloatTensor, optional): Embedded decoder inputs. Defaults to None.
            labels (torch.LongTensor, optional): Labels for classification. Defaults to None.
            use_cache (bool, optional): Whether to use cache. Defaults to None.
            output_attentions (bool, optional): Whether to output attentions. Defaults to None.
            output_hidden_states (bool, optional): Whether to output hidden states. Defaults to None.
            return_dict (bool, optional): Whether to return a dictionary. Defaults to None.

        Returns:
            Seq2SeqSequenceClassifierOutput or dict: Sequence classification output.
        """
        # Actual implementation of the forward pass follows in the code of the function.
        pass


@add_start_docstrings(
    """
    BigBirdPegasus Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BIGBIRD_PEGASUS_START_DOCSTRING,
)
class BigBirdPegasusForQuestionAnswering(BigBirdPegasusPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config):
        super().__init__(config)

        # Set number of output labels to 2 for question answering (start and end positions)
        config.num_labels = 2
        self.num_labels = config.num_labels

        self.model = BigBirdPegasusModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BIGBIRD_PEGASUS_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 使用装饰器添加代码示例的文档字符串，指定相关的检查点、输出类型和配置类

    # 以下内容是从 transformers.models.bart.modeling_bart.BartForQuestionAnswering.forward 复制而来

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



        # 前向传播函数，接受多个参数用于模型推断
        # input_ids: 输入序列的 token IDs
        # attention_mask: 注意力掩码，指定哪些位置是填充的
        # decoder_input_ids: 解码器输入的 token IDs
        # decoder_attention_mask: 解码器的注意力掩码
        # head_mask: 多头注意力机制的掩码
        # decoder_head_mask: 解码器多头注意力的掩码
        # cross_attn_head_mask: 跨注意力头的掩码
        # encoder_outputs: 编码器输出的列表
        # start_positions: 答案开始位置的 token IDs
        # end_positions: 答案结束位置的 token IDs
        # inputs_embeds: 嵌入式输入的张量
        # decoder_inputs_embeds: 解码器输入的嵌入式张量
        # use_cache: 是否使用缓存
        # output_attentions: 是否输出注意力权重
        # output_hidden_states: 是否输出隐藏状态
        # return_dict: 是否返回字典形式的输出
# 从transformers.models.pegasus.modeling_pegasus.PegasusDecoderWrapper复制代码，并将Pegasus更改为BigBirdPegasus
class BigBirdPegasusDecoderWrapper(BigBirdPegasusPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    def __init__(self, config):
        super().__init__(config)
        # 初始化BigBirdPegasusDecoder对象作为decoder
        self.decoder = BigBirdPegasusDecoder(config)

    def forward(self, *args, **kwargs):
        # 调用decoder的forward方法，并将参数传递下去
        return self.decoder(*args, **kwargs)


class BigBirdPegasusForCausalLM(BigBirdPegasusPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 深度拷贝config对象，设定为decoder模式，并关闭encoder-decoder模式
        config = copy.deepcopy(config)
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        # 初始化BigBirdPegasusDecoderWrapper对象作为model
        self.model = BigBirdPegasusDecoderWrapper(config)

        # 初始化线性层，作为lm_head，用于生成输出
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回decoder的embed_tokens作为输入的嵌入层
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        # 设置decoder的embed_tokens为新的值
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        # 返回lm_head作为输出的嵌入层
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置lm_head为新的输出嵌入层
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        # 设置decoder模型为给定的decoder对象
        self.model.decoder = decoder

    def get_decoder(self):
        # 返回当前的decoder对象
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass for the BigBirdPegasusForCausalLM model.
        """
        # 实现模型的前向传播，接受多种输入参数，并返回输出结果
        ...

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        """
        Prepare inputs for generation based on the BigBirdPegasusForCausalLM model.
        """
        # 准备生成模型输入的方法，接受多种参数，并返回适用于生成的输入
        ...
    ):
        # 如果模型被用作编码器-解码器模型中的解码器，注意力遮罩会即时创建
        if attention_mask is None:
            # 如果注意力遮罩为空，则创建一个与输入张量形状相同的全为1的张量作为注意力遮罩
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            # 如果有过去的键值状态，则只保留输入张量的最后一个位置作为当前输入
            input_ids = input_ids[:, -1:]
        # 返回一个包含各种输出和状态的字典
        return {
            "input_ids": input_ids,  # encoder_outputs 已经定义，不再需要 input_ids
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 对每一层的过去状态按照 beam_idx 重新排序，并转移到正确的设备上
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去状态
        return reordered_past
```