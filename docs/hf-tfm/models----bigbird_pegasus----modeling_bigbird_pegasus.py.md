# `.\transformers\models\bigbird_pegasus\modeling_bigbird_pegasus.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 Google Research 团队和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，不附带任何担保或条件
# 查看许可证以获取详细信息
""" PyTorch BigBirdPegasus 模型。"""

# 导入所需的模块
import copy
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义模块和类
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_bigbird_pegasus import BigBirdPegasusConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档中用到的变量
_CHECKPOINT_FOR_DOC = "google/bigbird-pegasus-large-arxiv"
_CONFIG_FOR_DOC = "BigBirdPegasusConfig"
_EXPECTED_OUTPUT_SHAPE = [1, 7, 1024]

# BigBirdPegasus 预训练模型列表
BIGBIRD_PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/bigbird-pegasus-large-arxiv",
    "google/bigbird-pegasus-large-pubmed",
    "google/bigbird-pegasus-large-bigpatent",
    # 更多 BigBirdPegasus 模型请查看：https://huggingface.co/models?filter=bigbird_pegasus
]

# 将输入的 token 向右移动一个位置
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    将输入的 token 向右移动一个位置。
    """
    # 创建一个新的张量，形状与输入相同，并填充为零
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # 将输入的 token 向右移动一个位置
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # 将解码器起始 token 放在第一个位置
    shifted_input_ids[:, 0] = decoder_start_token_id

    # 如果未定义 pad_token_id，则抛出 ValueError
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将标签中可能存在的 -100 值替换为 pad_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


# BigBirdPegasus 学习到的位置嵌入类
class BigBirdPegasusLearnedPositionalEmbedding(nn.Embedding):
    """
    这个模块学习位置嵌入，最大尺寸固定。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
    # 定义一个方法，用于计算位置编码
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        # 获取输入的 batch size 和序列长度
        bsz, seq_len = input_ids_shape[:2]
        # 生成位置编码的序列，范围从 past_key_values_length 到 past_key_values_length + seq_len
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        # 调用父类的 forward 方法，传入位置编码序列
        return super().forward(positions)
# 从transformers.models.big_bird.modeling_big_bird.BigBirdSelfAttention复制代码，并将BigBird替换为BigBirdPegasus
class BigBirdPegasusSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 检查隐藏大小是否是注意力头数的倍数，如果不是则引发错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)

        # 创建dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.is_decoder = config.is_decoder

    # 将输入张量重塑为注意力分数计算所需的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
# 从transformers.models.big_bird.modeling_big_bird.BigBirdBlockSparseAttention复制代码，并将BigBird替换为BigBirdPegasus
class BigBirdPegasusBlockSparseAttention(nn.Module):
    def __init__(self, config, seed=None):
        super().__init__()

        self.max_seqlen = config.max_position_embeddings
        self.seed = seed

        # 检查隐藏大小是否是注意力头数的倍数，如果不是则引发错误
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

        # 创建查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.use_bias)
    def transpose_for_scores(self, x):
        # 计算新的张量形状，将最后两个维度调整为多头注意力的形状
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 调整张量形状
        x = x.view(*new_x_shape)
        # 对张量进行维度置换，以符合注意力计算的要求
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
        # 当前此“类”不能在解码器中使用。

        # 获取隐藏状态张量的大小
        batch_size, seqlen, _ = hidden_states.size()
        to_seq_length = from_seq_length = seqlen
        from_block_size = to_block_size = self.block_size

        # 如果查询序列长度不是块大小的倍数，则引发错误
        if from_seq_length % from_block_size != 0:
            raise ValueError("Query sided sequence length must be multiple of block size")

        # 如果键/值序列长度不是块大小的倍数，则引发错误
        if to_seq_length % to_block_size != 0:
            raise ValueError("Key/Value sided sequence length must be multiple of block size")

        # 对隐藏状态进行查询、键、值的线性变换并转置，以准备进行注意力计算
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 执行 BigBird 块稀疏注意力机制
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

        # 重新构造上下文张量的形状
        context_layer = context_layer.contiguous().view(batch_size, from_seq_length, -1)

        # 如果需要输出注意力矩阵，则将其包含在输出中
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    @staticmethod
    def torch_bmm_nd(inp_1, inp_2, ndim=None):
        """Fast nd matrix multiplication"""
        # 快速的 n 维矩阵乘法
        # 替代 torch.einsum 的更快方法 ("bhqk,bhkd->bhqd")
        return torch.bmm(inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:])).view(
            inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 1])
        )

    @staticmethod
    def torch_bmm_nd_transpose(inp_1, inp_2, ndim=None):
        """Fast nd matrix multiplication with transpose"""
        # 快速的带转置的 n 维矩阵乘法
        # 替代 torch.einsum 的更快方法 (bhqd,bhkd->bhqk)
        return torch.bmm(
            inp_1.reshape((-1,) + inp_1.shape[-2:]), inp_2.reshape((-1,) + inp_2.shape[-2:]).transpose(1, 2)
        ).view(inp_1.shape[: ndim - 2] + (inp_1.shape[ndim - 2], inp_2.shape[ndim - 2]))
    # 定义一个方法，用于执行大型稀疏注意力计算
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
    
    # 定义一个静态方法，用于在 batch_dims=2 时执行 torch.gather 操作
    @staticmethod
    def torch_gather_b2(params, indices):
        # 这个操作等同于 tf.gather，当 batch_dims=2 时

        # 检查 params 和 indices 的前两个维度是否相同
        if params.shape[:2] != indices.shape[:2]:
            raise ValueError(
                "Make sure that the first two dimensions of params and indices are identical,                 but"
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

    # 静态方法，用于从输入创建随机掩码
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
        从一个2D张量掩码创建3D注意力掩码。

        Args:
            from_blocked_mask: 形状为[batch_size, from_seq_length//from_block_size, from_block_size]的2D张量掩码。
            to_blocked_mask: 形状为[batch_size, to_seq_length//to_block_size, to_block_size]的int32张量掩码。
            rand_attn: 形状为[batch_size, num_attention_heads, from_seq_length//from_block_size-2, num_rand_blocks]的张量。
            num_attention_heads: int。注意力头的数量。
            num_rand_blocks: int。每行的随机块数。
            batch_size: int。计算的批量大小。
            from_seq_length: int。来自序列的长度。
            from_block_size: int。来自序列中块的大小。

        Returns:
            形状为[batch_size, num_attention_heads, from_seq_length//from_block_size-2, from_block_size, num_rand_blocks*to_block_size]的float张量。
        """
        num_windows = from_seq_length // from_block_size - 2
        rand_mask = torch.stack([p1[i1.flatten()] for p1, i1 in zip(to_blocked_mask, rand_attn)])
        rand_mask = rand_mask.view(batch_size, num_attention_heads, num_windows, num_rand_blocks * from_block_size)
        rand_mask = torch.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1], rand_mask)
        return rand_mask

    @staticmethod
    def _get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
        """
        给出随机注意力放置计划。

        Args:
            from_seq_length: int。来自序列的长度。
            from_block_size: int。来自序列中块的大小。
            num_rand_blocks: int。每行的随机块数。

        Returns:
            plan_from_length: 来自块计划的结束位置。
            plan_num_rand_blocks: 每个块的随机结束位置数量。
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

        # 如果 from_seq_length 除以 from_block_size 不等于 to_seq_length 除以 to_block_size，则抛出数值错误
        if from_seq_length // from_block_size != to_seq_length // to_block_size:
            raise ValueError("Error the number of blocks needs to be same!")

        # 创建一个大小为 from_seq_length//from_block_size-2 by num_rand_blocks 的全零矩阵
        rand_attn = np.zeros((from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
        # 在推断（评估）过程中不使用随机性，直接返回全零矩阵
        if not self.training:
            return rand_attn
        # 创建中间序列，范围从1到to_seq_length // to_block_size - 2
        middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
        last = to_seq_length // to_block_size - 1
        # 如果 last_idx 大于 (2 * to_block_size)，则将 last 限制为 (last_idx // to_block_size) - 1
        if last_idx > (2 * to_block_size):
            last = (last_idx // to_block_size) - 1

        r = num_rand_blocks  # shorthand
        # 遍历 from_seq_length // from_block_size - 1 次，从第二个块开始到倒数第二个块
        for i in range(1, from_seq_length // from_block_size - 1):
            start = i - 2
            end = i
            # 如果当前是第一个块
            if i == 1:
                # 从中间序列中的第三个元素到最后一个元素中随机选择 r 个元素
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
            # 如果当前是第二个块
            elif i == 2:
                # 从中间序列中的第四个元素到最后一个元素中随机选择 r 个元素
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
            # 如果当前是倒数第三个块
            elif i == from_seq_length // from_block_size - 3:
                # 从中间序列中的第一个元素到最后一个元素中随机选择 r 个元素
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # 如果当前是倒数第二个块
            elif i == from_seq_length // from_block_size - 2:
                # 从中间序列中的第一个元素到最后一个元素中随机选择 r 个元素
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            else:
                # 如果起始索引大于最后一个索引，将起始索引设置为最后一个索引，并从中间序列的开头选择 r 个元素
                if start > last:
                    start = last
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                # 如果结束索引加一等于最后一个索引，从中间序列的开头选择 r 个元素
                elif (end + 1) == last:
                    rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
                # 否则，从连接起始索引和结束索引加一之后到最后一个索引之间的元素中随机选择 r 个元素
                else:
                    rand_attn[i - 1, :] = np.random.permutation(
                        np.concatenate((middle_seq[:start], middle_seq[end + 1 : last]))
                    )[:r]
        # 返回随机注意力的邻接列表
        return rand_attn
    # 定义一个方法，用于生成带有头部的大鸟块随机掩码
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
    @staticmethod
    # 定义一个静态方法，用于获取单个块的行注意力
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
        对于单个行块，获取随机行注意力。

        Args:
            block_id: int. 行的块ID。
            to_start_block_id: int. 随机注意力列的起始ID。
            to_end_block_id: int. 随机注意力列的结束ID。
            num_rand_blocks: int. 要选择的随机块数。
            window_block_left: int. 块左侧窗口中的块数。
            window_block_right: int. 块右侧窗口中的块数。
            global_block_left: int. 左侧全局使用的块数。
            global_block_right: int. 右侧全局使用的块数。

        Returns:
            包含大小为num_rand_blocks的随机注意力向量的行。
        """
        # 生成要选择随机注意力的to_blocks列表
        to_block_list = np.arange(to_start_block_id, to_end_block_id, dtype=np.int32)
        # 对块进行排列
        perm_block = np.random.permutation(to_block_list)

        # 当前块ID的非法块，使用窗口
        illegal_blocks = list(range(block_id - window_block_left, block_id + window_block_right + 1))

        # 在开头和结尾添加块
        illegal_blocks.extend(list(range(global_block_left)))
        illegal_blocks.extend(list(range(to_end_block_id - global_block_right, to_end_block_id)))

        # 第二个from_block不能在倒数第二个to_block上选择随机注意力
        if block_id == 1:
            illegal_blocks.append(to_end_block_id - 2)

        # 倒数第二个from_block不能在第二个to_block上选择随机注意力
        if block_id == to_end_block_id - 2:
            illegal_blocks.append(1)

        selected_random_blokcs = []

        for i in range(to_end_block_id - to_start_block_id):
            if perm_block[i] not in illegal_blocks:
                selected_random_blokcs.append(perm_block[i])
            if len(selected_random_blokcs) == num_rand_blocks:
                break
        return np.array(selected_random_blokcs, dtype=np.int32)
class BigBirdPegasusEncoderAttention(nn.Module):
    def __init__(self, config, seed=None):
        super().__init__()
        self.config = config  # 初始化模型配置
        self.seed = seed  # 设定种子用于随机数生成

        self.attention_type = config.attention_type  # 从配置中获取注意力类型

        if self.attention_type == "original_full":
            self.self = BigBirdPegasusSelfAttention(config)  # 如果是原始的全局注意力，使用对应的自注意力模块
        elif self.attention_type == "block_sparse":
            self.self = BigBirdPegasusBlockSparseAttention(config, seed)  # 如果是块稀疏注意力，使用对应的自注意力模块
        else:
            raise ValueError(
                f"attention_type can either be original_full or block_sparse, but is {self.config.attention_type}"
            )  # 如果注意力类型不合法，抛出数值错误异常

        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)  # 线性变换层用于输出

    def set_attention_type(self, value: str):
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )  # 如果设置的注意力类型不合法，抛出数值错误异常
        # 如果注意力类型已经正确设置，则直接返回
        if value == self.attention_type:
            return

        self.attention_type = value  # 更新注意力类型
        if value == "original_full":
            # 将所有权重复制到新的全局注意力类中
            attn_weights = BigBirdPegasusSelfAttention(self.config)
        else:
            # 将所有权重复制到新的稀疏注意力类中
            attn_weights = BigBirdPegasusBlockSparseAttention(self.config, self.seed)

        attn_weights.query = self.self.query  # 复制查询权重
        attn_weights.value = self.self.value  # 复制值权重
        attn_weights.key = self.self.key  # 复制键权重
        self.self = attn_weights  # 更新自注意力模块
        self.attention_type = value  # 更新注意力类型

        if not self.training:
            self.self.eval()  # 如果不处于训练模式，将自注意力模块设置为评估模式

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
        # 扩展维度以在自注意力模块中进行乘法操作
        head_mask = head_mask.reshape(1, -1, 1, 1) if head_mask is not None else None

        if self.config.attention_type == "original_full":
            self_outputs = self.self(
                hidden_states,
                attention_mask,
                head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )  # 如果是原始的全局注意力，调用对应的自注意力模块
        else:
            self_outputs = self.self(
                hidden_states, band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask, output_attentions
            )  # 如果是块稀疏注意力，调用对应的自注意力模块

        attention_output = self.output(self_outputs[0])  # 输出层处理注意力输出
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力权重，则添加到输出中
        return outputs


# 从transformers.models.bart.modeling_bart.BartAttention复制而来，将BartConfig->BigBirdPegasusConfig，Bart->BigBirdPegasusDecoder
```  
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
        # 初始化注意力机制的参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 确保头的维度整除嵌入维度
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放因子
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 初始化线性映射层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将张量重塑为多头形式
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
):
        # 注意力机制的前向传播
        pass

class BigBirdPegasusEncoderLayer(nn.Module):
    def __init__(self, config: BigBirdPegasusConfig, seed=None):
        super().__init__()
        # 初始化编码器层的参数
        self.attention_type = config.attention_type
        self.embed_dim = config.d_model
        self.self_attn = BigBirdPegasusEncoderAttention(config, seed=seed)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        band_mask=None,
        from_mask=None,
        to_mask=None,
        from_blocked_mask=None,
        to_blocked_mask=None,
        output_attentions: bool = False,
):
        # 编码器层的前向传播
        pass
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): 形状为 `(batch, seq_len, embed_dim)` 的层的输入
            attention_mask (`torch.FloatTensor`): 大小为 `(batch, 1, tgt_len, src_len)` 的注意力掩码，
                其中填充元素由非常大的负值表示。
            output_attentions (`bool`, *可选*):
                是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量下的`attentions`。
        """
        # 保存残差连接
        residual = hidden_states
        # 对输入进行自注意力层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 自注意力计算及其输出
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
        # 更新隐藏状态
        hidden_states = self_attention_outputs[0]

        # 使用丢弃操作进行正则化
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 更新隐藏状态
        hidden_states = residual + hidden_states

        # 保存残差连接
        residual = hidden_states
        # 对输入进行最终层归一化
        hidden_states = self.final_layer_norm(hidden_states)
        # 应用激活函数
        hidden_states = self.activation_fn(self.fc1(hidden_states))

        # 第二个全连接层
        hidden_states = self.fc2(hidden_states)
        # 使用丢弃操作进行正则化
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 更新隐藏状态
        hidden_states = residual + hidden_states

        # 如果隐藏状态的数据类型为 torch.float16 并且存在 inf 或 nan
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            # 将隐藏状态限制在一个较小的范围内
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 输出包含隐藏状态
        outputs = (hidden_states,)

        # 如果需要输出注意力，将注意力张量添加到输出中
        if output_attentions:
            outputs += (self_attention_outputs[1],)

        return outputs

    def set_attention_type(self, value: str):
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # 如果给定的值与当前的注意力类型一致，则直接返回
        if value == self.attention_type:
            return
        # 否则更新注意力类型
        self.attention_type = value
        self.self_attn.set_attention_type(value)
class BigBirdPegasusDecoderLayer(nn.Module):
    def __init__(self, config: BigBirdPegasusConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BigBirdPegasusDecoderAttention(
            embed_dim=self.embed_dim,  # 设置自注意力机制的嵌入维度
            num_heads=config.decoder_attention_heads,  # 自注意力机制的头数
            dropout=config.attention_dropout,  # 自注意力机制的dropout比率
            is_decoder=True,  # 表明这是解码器层
            bias=config.use_bias,  # 是否使用偏置
        )
        self.dropout = config.dropout  # dropout比率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数
        self.activation_dropout = config.activation_dropout  # 激活函数的dropout比率

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 自注意力机制后的LayerNorm
        self.encoder_attn = BigBirdPegasusDecoderAttention(
            self.embed_dim,  # 编码器注意力机制的嵌入维度
            config.decoder_attention_heads,  # 编码器注意力机制的头数
            dropout=config.attention_dropout,  # 编码器注意力机制的dropout比率
            is_decoder=True,  # 表明这是解码器层
            bias=config.use_bias,  # 是否使用偏置
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 编码器注意力机制后的LayerNorm
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)  # 全连接层1
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)  # 全连接层2
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最终的LayerNorm

    # 从transformers.models.mbart.modeling_mbart.MBartDecoderLayer.forward复制而来
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
# 从transformers.models.bart.modeling_bart.BartClassificationHead 复制而来，将Bart改为BigBirdPegasus
class BigBirdPegasusClassificationHead(nn.Module):
    """用于句子级分类任务的头部模块。"""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)  # 全连接层
        self.dropout = nn.Dropout(p=pooler_dropout)  # dropout层
        self.out_proj = nn.Linear(inner_dim, num_classes)  # 输出层

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)  # 应用dropout
        hidden_states = self.dense(hidden_states)  # 全连接层
        hidden_states = torch.tanh(hidden_states)  # Tanh激活函数
        hidden_states = self.dropout(hidden_states)  # 应用dropout
        hidden_states = self.out_proj(hidden_states)  # 输出层
        return hidden_states


class BigBirdPegasusPreTrainedModel(PreTrainedModel):
    config_class = BigBirdPegasusConfig  # 使用的配置类
    base_model_prefix = "model"  # 基础模型的前缀
    supports_gradient_checkpointing = True  # 支持梯度检查点
    _no_split_modules = ["BigBirdPegasusEncoderLayer", "BigBirdPegasusDecoderLayer"]  # 不分割的模块列表
    _skip_keys_device_placement = "past_key_values"  # 跳过的键设备放置
    # 初始化模型参数的权重
    def _init_weights(self, module):
        # 获取初始化标准差
        std = self.config.init_std
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在填充索引，将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # 返回虚拟输入，用于模型推理
    @property
    def dummy_inputs(self):
        # 获取填充标记
        pad_token = self.config.pad_token_id
        # 创建虚拟输入张量
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        # 构建虚拟输入字典
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        # 返回虚拟输入字典
        return dummy_inputs
# 定义 BigBirdPegasus 模型的起始文档字符串，包含模型的继承关系、参数说明等信息
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

# 定义 BigBirdPegasus 模型的生成示例文档字符串，包含摘要示例代码
BIGBIRD_PEGASUS_GENERATION_EXAMPLE = r"""
    Summarization example:

    ```py
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

# 定义 BigBirdPegasus 模型的输入文档字符串
BIGBIRD_PEGASUS_INPUTS_DOCSTRING = r"""
"""

# 定义 BigBirdPegasus 模型的独立���入文档字符串
BIGBIRD_PEGASUS_STANDALONE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。默认情况下会忽略填充。
            # 使用 [`ProphetNetTokenizer`] 可以获得这些索引。有关详情，请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。掩码值在 `[0, 1]` 之间：

            # - 对于 **未掩码** 的标记，掩码值为 1，
            # - 对于 **掩码** 的标记，掩码值为 0。

            # [什么是注意力掩码？](../glossary#attention-mask)
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是一个普通的元组。
```py  
# 定义一个 BigBirdPegasusEncoder 类，继承自 BigBirdPegasusPreTrainedModel
class BigBirdPegasusEncoder(BigBirdPegasusPreTrainedModel):
    """
    Transformer 编码器，由 config.encoder_layers 个自注意力层组成。每个层都是一个 BigBirdPegasusEncoderLayer。

    Args:
        config: BigBirdPegasusConfig
        embed_tokens (nn.Embedding): 输出嵌入
    """

    def __init__(self, config: BigBirdPegasusConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类构造函数
        super().__init__(config)

        # 初始化各种参数
        self.attention_type = config.attention_type
        self.block_size = config.block_size

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # 初始化嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # 如果传入了 embed_tokens，则使用传入的权重
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        # 初始化位置编码
        self.embed_positions = BigBirdPegasusLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        # 初始化编码器层
        self.layers = nn.ModuleList([BigBirdPegasusEncoderLayer(config, seed=i) for i in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def set_attention_type(self, value: str):
        # 设置注意力类型
        if value not in ["original_full", "block_sparse"]:
            raise ValueError(
                f"attention_type can only be set to either 'original_full' or 'block_sparse', but is {value}"
            )
        # 如果注意力类型已经正确设置，则直接返回
        if value == self.attention_type:
            return
        self.attention_type = value
        # 设置每个编码器层的注意力类型
        for layer in self.layers:
            layer.set_attention_type(value)

    @staticmethod  # 从 transformers.models.big_bird.modeling_big_bird.BigBirdModel.create_masks_for_block_sparse_attn 复制
    def create_masks_for_block_sparse_attn(attention_mask: torch.Tensor, block_size: int):
        # 获取 batch_size 和 seq_length
        batch_size, seq_length = attention_mask.size()
        # 检查 seq_length 是否是 block_size 的倍数
        if seq_length % block_size != 0:
            # 如果不是，抛出数值错误
            raise ValueError(
                f"Sequence length must be multiple of block size, but sequence length is {seq_length}, while block"
                f" size is {block_size}."
            )

        def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
            """
            Create 3D attention mask from a 2D tensor mask.

            Args:
                from_blocked_mask: 2D Tensor of shape [batch_size,
                from_seq_length//from_block_size, from_block_size].
                to_blocked_mask: int32 Tensor of shape [batch_size,
                to_seq_length//to_block_size, to_block_size].

            Returns:
                float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4, from_block_size,
                3*to_block_size].
            """
            # 拼接 to_blocked_mask 的部分
            exp_blocked_to_pad = torch.cat(
                [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]], dim=2
            )
            # 使用 einsum 创建 band_mask
            band_mask = torch.einsum("blq,blk->blqk", from_blocked_mask[:, 2:-2], exp_blocked_to_pad)
            band_mask.unsqueeze_(1)
            return band_mask

        # 将 attention_mask 转换为 block 形式
        blocked_encoder_mask = attention_mask.view(batch_size, seq_length // block_size, block_size)
        # 创建 band_mask
        band_mask = create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)

        # 创建 from_mask 和 to_mask
        from_mask = attention_mask.view(batch_size, 1, seq_length, 1)
        to_mask = attention_mask.view(batch_size, 1, 1, seq_length)

        return blocked_encoder_mask, band_mask, from_mask, to_mask

    def _pad_to_block_size(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        """A helper function to pad tokens and mask to work with implementation of BigBird block-sparse attention."""
        # 填充 tokens 和 mask 以适应 BigBird block-sparse attention 的实现
        # 获取 block_size
        block_size = self.config.block_size
        batch_size, seq_len = hidden_states.shape[:2]

        # 计算需要填充的长度
        padding_len = (block_size - seq_len % block_size) % block_size
        if padding_len > 0:
            # 如果需要填充，警告并进行填充
            logger.warning_once(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.block_size`: {block_size}"
            )
            pad_id = self.config.pad_token_id
            device = hidden_states.device
            input_ids_padding = torch.ones((batch_size, padding_len), dtype=torch.long, device=device) * pad_id
            inputs_embeds_padding = self.embed_tokens(input_ids_padding)
            hidden_states = torch.cat([hidden_states, inputs_embeds_padding], dim=-2)

            # 对 attention_mask 进行填充
            attention_mask = nn.functional.pad(
                attention_mask, (0, padding_len), value=0
            )  # no attention on the padding tokens

        return padding_len, hidden_states, attention_mask
class BigBirdPegasusDecoder(BigBirdPegasusPreTrainedModel):
    """
    Transformer 解码器，由 *config.decoder_layers* 层组成。每一层都是一个 [`BigBirdPegasusDecoderLayer`]

    Args:
        config: BigBirdPegasusConfig
        embed_tokens (nn.Embedding): 输出的嵌入
    """

    def __init__(self, config: BigBirdPegasusConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置丢弃率
        self.dropout = config.dropout
        # 设置层丢弃率
        self.layerdrop = config.decoder_layerdrop
        # 设置填充索引
        self.padding_idx = config.pad_token_id
        # 设置最大目标位置
        self.max_target_positions = config.max_position_embeddings
        # 设置嵌入尺度
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 初始化嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # 如果提供了外部嵌入，则使用外部嵌入的权重
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        # 初始化位置嵌入
        self.embed_positions = BigBirdPegasusLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        # 初始化解码器层
        self.layers = nn.ModuleList([BigBirdPegasusDecoderLayer(config) for _ in range(config.decoder_layers)])
        # 初始化嵌入层的 LayerNorm
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        # 是否使用梯度检查点
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



@add_start_docstrings(
    "The bare BigBirdPegasus Model outputting raw hidden-states without any specific head on top.",
    BIGBIRD_PEGASUS_START_DOCSTRING,
)
class BigBirdPegasusModel(BigBirdPegasusPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BigBirdPegasusConfig):
        super().__init__(config)

        # 获取填充索引和词汇量大小
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # 共享嵌入层
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # 初始化编码器和解码器
        self.encoder = BigBirdPegasusEncoder(config, self.shared)
        self.decoder = BigBirdPegasusDecoder(config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()
    # 返回输入嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # 绑定权重
    def _tie_weights(self):
        # 如果配置中设置了词嵌入共享，则将编码器和解码器的嵌入权重绑定到共享的嵌入层
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 使用装饰器为模型的前向传播方法添加文档字符串
    # 使用装饰器添加代码示例文档字符串
    # 从transformers.models.bart.modeling_bart.BartModel.forward复制而来，将Bart->BigBirdPegasus
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
# 添加模型文档字符串，说明这是一个带有语言建模头的BigBirdPegasus模型，可用于摘要生成
# 从transformers.models.bart.modeling_bart.BartForConditionalGeneration复制代码，并将Bart->BigBirdPegasus, BART->BIGBIRD_PEGASUS
class BigBirdPegasusForConditionalGeneration(BigBirdPegasusPreTrainedModel):
    # 指定基础模型前缀为"model"
    base_model_prefix = "model"
    # 指定需要共享权重的键
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    # 指定加载时忽略的键
    _keys_to_ignore_on_load_missing = ["final_logits_bias"]

    def __init__(self, config: BigBirdPegasusConfig):
        super().__init__(config)
        # 创建BigBirdPegasusModel模型
        self.model = BigBirdPegasusModel(config)
        # 注册缓冲区"final_logits_bias"，初始化为全零向量
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # 创建线性层lm_head，用于语言建模
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取编码器
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器
    def get_decoder(self):
        return self.model.get_decoder()

    # 调整token嵌入的大小
    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 调整final_logits_bias的大小以匹配新的嵌入大小
        self._resize_final_logits_bias(new_embeddings.weight.shape[0])
        return new_embeddings

    # 调整final_logits_bias的大小
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 添加模型前向方法的文档字符串
    # 替换返回文档字符串，指定输出类型为Seq2SeqLMOutput，配置类为_CONFIG_FOR_DOC
    # 添加生成示例的结束文档字符串
    # 定义一个前向传播函数，接受多个输入参数
    def forward(
        # 输入的 token IDs，数据类型为 LongTensor
        input_ids: torch.LongTensor = None,
        # 注意力掩码，数据类型为可选的 Tensor
        attention_mask: Optional[torch.Tensor] = None,
        # 解码器的输入 token IDs，数据类型为可选的 LongTensor
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器的注意力掩码，数据类型为可选的 LongTensor
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 头部掩码，数据类型为可选的 Tensor
        head_mask: Optional[torch.Tensor] = None,
        # 解码器的头部掩码，数据类型为可选的 Tensor
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头部掩码，数据类型为可选的 Tensor
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出，数据类型为可选的 FloatTensor 列表
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        # 过去的键值对，数据类型为可选的 FloatTensor 列表
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 输入的嵌入向量，数据类型为可选的 FloatTensor
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 解码器的输入嵌入向量，数据类型为可选的 FloatTensor
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # 标签，数据类型为可选的 LongTensor
        labels: Optional[torch.LongTensor] = None,
        # 是否使用缓存，数据类型为可选的布尔值
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，数据类型为可选的布尔值
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，数据类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典，数据类型为可选的布尔值
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        返回值类型可以是元组或Seq2SeqLMOutput对象
        
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 确定是否返回字典类型的结果，若未指定，则根据配置决定

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            # 如果提供了标签，且use_cache为True，则给出警告并将use_cache设为False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
            # 如果没有提供解码器输入ID和解码器输入嵌入，并且提供了标签，则创建解码器输入ID，将标签右移一位作为输入

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
        # 使用模型进行前向传播，根据参数传递相应的输入和掩码，并根据需要返回字典形式的输出

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        # 计算语言模型的logits，加上最终的logits偏置

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            # 如果提供了标签，则计算masked语言建模损失

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            # 如果不返回字典，则构造输出元组，包含logits和其他输出信息

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
        # 返回Seq2SeqLMOutput对象，包含损失、logits和其他输出信息
    # 为生成准备输入的函数，用于生成模型的推理过程
    def prepare_inputs_for_generation(
        self,
        # 解码器的输入 ID
        decoder_input_ids,
        # 过去的键值对（用于记忆），默认为 None
        past_key_values=None,
        # 注意力遮罩，指示模型关注哪些位置的输入
        attention_mask=None,
        # 解码器的注意力遮罩，指示解码器关注哪些位置的输入
        decoder_attention_mask=None,
        # 头部遮罩，用于控制多头注意力中的头部的掩码
        head_mask=None,
        # 解码器头部遮罩，用于控制解码器中的多头注意力中的头部的掩码
        decoder_head_mask=None,
        # 交叉注意力头部遮罩，用于控制编码器-解码器注意力中的多头注意力中的头部的掩码
        cross_attn_head_mask=None,
        # 是否使用缓存，用于控制是否缓存中间计算结果
        use_cache=None,
        # 编码器输出，用于生成
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用过去的键值对（用于记忆）
        if past_key_values is not None:
            # 获取过去键值对中的过去长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法可能已经只传递最后一个输入 ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认为旧的行为：仅保留最后一个 ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 从解码器输入 ID 中删除前缀长度
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回生成所需的输入
        return {
            "input_ids": None,  # 编码器输出已定义，不需要输入 ID
            "encoder_outputs": encoder_outputs,  # 编码器输出
            "past_key_values": past_key_values,  # 过去的键值对
            "decoder_input_ids": decoder_input_ids,  # 解码器输入 ID
            "attention_mask": attention_mask,  # 注意力遮罩
            "decoder_attention_mask": decoder_attention_mask,  # 解码器注意力遮罩
            "head_mask": head_mask,  # 头部遮罩
            "decoder_head_mask": decoder_head_mask,  # 解码器头部遮罩
            "cross_attn_head_mask": cross_attn_head_mask,  # 交叉注意力头部遮罩
            "use_cache": use_cache,  # 更改此选项以避免缓存（可能用于调试）
        }

    # 从标签准备解码器输入 ID 的函数
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 将标签向右移动一位以准备解码器的输入 ID
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    # 重新排列缓存的函数，用于重新排序缓存中的键值对
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 重新排序过的过去键值对
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态不需要重新排序->它们始终相同
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
# 定义一个带有序列分类/头部的 BigBirdPegasus 模型，例如用于 GLUE 任务
class BigBirdPegasusForSequenceClassification(BigBirdPegasusPreTrainedModel):
    # 定义共享权重的键值对
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: BigBirdPegasusConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, **kwargs)
        # 创建 BigBirdPegasus 模型
        self.model = BigBirdPegasusModel(config)
        # 创建分类头部
        self.classification_head = BigBirdPegasusClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播方法
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
        # 前向传播逻辑

# 定义一个带有问题回答的 BigBirdPegasus 模型，用于提取性问题回答任务，如 SQuAD
class BigBirdPegasusForQuestionAnswering(BigBirdPegasusPreTrainedModel):
    # 定义共享权重的键值对
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置标签数量为 2
        config.num_labels = 2
        self.num_labels = config.num_labels

        # 创建 BigBirdPegasus 模型
        self.model = BigBirdPegasusModel(config)
        # 创建问题回答输出层
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播方法
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
        # 前向传播逻辑
        # 返回字典，包含模型的前向传播输出
        return_dict: Optional[bool] = None,
# 从 transformers.models.pegasus.modeling_pegasus.PegasusDecoderWrapper 复制代码，将 Pegasus 替换为 BigBirdPegasus
class BigBirdPegasusDecoderWrapper(BigBirdPegasusPreTrainedModel):
    """
    This wrapper class is a helper class to correctly load pretrained checkpoints when the causal language model is
    used in combination with the [`EncoderDecoderModel`] framework.
    """

    # 初始化方法，接受配置参数并调用父类的初始化方法
    def __init__(self, config):
        super().__init__(config)
        # 创建 BigBirdPegasusDecoder 实例并赋值给 self.decoder
        self.decoder = BigBirdPegasusDecoder(config)

    # 前向传播方法，调用 self.decoder 的前向传播方法，并返回结果
    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


class BigBirdPegasusForCausalLM(BigBirdPegasusPreTrainedModel):
    # 定义与权重共享相关的键名
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化方法，接受配置参数并调用父类的初始化方法
    def __init__(self, config):
        # 对配置进行深拷贝
        config = copy.deepcopy(config)
        # 设置为解码器
        config.is_decoder = True
        # 设置为非编码器解码器
        config.is_encoder_decoder = False
        super().__init__(config)
        # 创建 BigBirdPegasusDecoderWrapper 实例并赋值给 self.model
        self.model = BigBirdPegasusDecoderWrapper(config)

        # 创建线性层以用作语言模型头部，输入大小为隐藏层大小，输出大小为词汇表大小，无偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 设置解码器
    def set_decoder(self, decoder):
        self.model.decoder = decoder

    # 获取解码器
    def get_decoder(self):
        return self.model.decoder

    # 前向传播方法，调用 self.model.decoder 的前向传播方法，并返回结果，使用参数替换文档字符串
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
    # 准备用于生成的输入方法，将参数透传到 BigBirdPegasusDecoderWrapper 实例的同名方法
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
        # 如果模型作为编码器-解码器模型的解码器使用，则动态创建解码器注意力掩码
        if attention_mask is None:
            # 如果没有提供注意力掩码，则创建一个全为1的掩码，与输入张量的形状相同
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            # 如果提供了过去的键值对，则只取输入张量的最后一个标记
            input_ids = input_ids[:, -1:]
        # 第一步，decoder_cached_states为空
        return {
            "input_ids": input_ids,  # encoder_outputs已定义，不需要input_ids
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                # 重新排序过去的键值对，根据beam索引
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```