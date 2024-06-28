# `.\models\gpt_neox\modeling_gpt_neox.py`

```
# coding=utf-8
# 版权 2022 EleutherAI 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可 2.0 版本（“许可证”）许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”分发，
# 没有任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
""" PyTorch GPTNeoX 模型。"""

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from ...activations import ACT2FN
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10, logging
from .configuration_gpt_neox import GPTNeoXConfig

# 如果 flash attention 2 可用，则导入相关函数
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置信息
_CHECKPOINT_FOR_DOC = "trl-internal-testing/tiny-random-GPTNeoXForCausalLM"
_REAL_CHECKPOINT_FOR_DOC = "EleutherAI/gpt-neox-20b"
_CONFIG_FOR_DOC = "GPTNeoXConfig"

# GPTNeoX 预训练模型的存档列表
GPT_NEOX_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "EleutherAI/gpt-neox-20b",
    # 查看所有 GPTNeoX 模型：https://huggingface.co/models?filter=gpt_neox
]

# 从 transformers.models.llama.modeling_llama._get_unpad_data 复制而来的函数
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

# GPTNeoX 预训练模型的基类，用于处理权重初始化和预训练模型的下载加载接口
class GPTNeoXPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，用于处理权重初始化和简单的预训练模型下载加载接口。
    """

    config_class = GPTNeoXConfig  # 配置类
    base_model_prefix = "gpt_neox"  # 基础模型前缀
    supports_gradient_checkpointing = True  # 支持梯度检查点
    _no_split_modules = ["GPTNeoXLayer"]  # 不需要拆分的模块列表
    _skip_keys_device_placement = "past_key_values"  # 跳过设备放置的键名
    _supports_flash_attn_2 = True  # 支持 flash attention 2 特性
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层（全连接层）
        if isinstance(module, nn.Linear):
            # 初始化权重为正态分布，均值为0，标准差为配置文件中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 初始化权重为正态分布，均值为0，标准差为配置文件中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果定义了填充索引，将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是层归一化层
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置项为零
            module.bias.data.zero_()
            # 初始化权重为1
            module.weight.data.fill_(1.0)
# 定义一个名为 GPTNeoXAttention 的类，继承自 nn.Module
class GPTNeoXAttention(nn.Module):
    # 初始化方法，接收一个名为 config 的参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的 config 参数保存在实例变量 self.config 中
        self.config = config
        # 从 config 中获取并保存注意力头数
        self.num_attention_heads = config.num_attention_heads
        # 从 config 中获取并保存隐藏层大小
        self.hidden_size = config.hidden_size
        # 检查隐藏层大小是否可以被注意力头数整除，否则抛出异常
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them"
            )
        # 计算每个注意力头的大小
        self.head_size = self.hidden_size // self.num_attention_heads
        # 计算旋转嵌入的维度数量
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        # 初始化偏置参数
        self._init_bias(config.max_position_embeddings)

        # 注册一个名为 masked_bias 的缓冲区，并设置其为固定的 torch.tensor(-1e9)
        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)
        # 初始化 RoPE（Rotary Positional Embeddings）
        self._init_rope()

        # 计算规范化因子，用于注意力计算中
        self.norm_factor = self.head_size ** -0.5
        # 定义一个线性层，用于生成查询、键、值向量
        self.query_key_value = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        # 定义一个线性层，用于最终输出
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        # 定义一个 Dropout 层，用于注意力机制中的随机失活
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        # 设定是否为因果关系（自回归任务中使用）
        self.is_causal = True

    # 初始化偏置方法，接收最大位置嵌入数和设备参数（可选）
    def _init_bias(self, max_positions, device=None):
        # 创建一个下三角矩阵，用于自注意力机制中的掩码
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        # 如果设备参数不为空，则将 bias 缓冲区移到指定设备上
        if device is not None:
            self.bias = self.bias.to(device)

    # 初始化 RoPE 方法
    def _init_rope(self):
        # 如果配置中未指定 RoPE 缩放类型，则使用基本的 GPTNeoXRotaryEmbedding
        if self.config.rope_scaling is None:
            self.rotary_emb = GPTNeoXRotaryEmbedding(
                self.rotary_ndims, self.config.max_position_embeddings, base=self.config.rotary_emb_base
            )
        else:
            # 否则，根据配置中的 RoPE 缩放类型选择不同的 RoPE 实现
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = GPTNeoXLinearScalingRotaryEmbedding(
                    self.rotary_ndims,
                    self.config.max_position_embeddings,
                    base=self.config.rotary_emb_base,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = GPTNeoXDynamicNTKScalingRotaryEmbedding(
                    self.rotary_ndims,
                    self.config.max_position_embeddings,
                    base=self.config.rotary_emb_base,
                    scaling_factor=scaling_factor,
                )
            else:
                # 如果配置中指定了未知的 RoPE 缩放类型，则抛出异常
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        # 检查是否存在先前的层级过去信息
        has_layer_past = layer_past is not None

        # 计算查询、键和值的QKV
        qkv = self.query_key_value(hidden_states)

        # 重塑QKV的形状，以便分离头部
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # 分离查询、键和值，并重新排列维度以适应多头注意力计算
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # 对旋转维度的查询和键应用旋转嵌入
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]
        
        # 计算旋转嵌入的余弦和正弦值
        seq_len = key.shape[-2]
        if has_layer_past:
            seq_len += layer_past[0].shape[-2]
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # 如果存在先前的层级过去信息，将当前的键和值与先前的拼接起来
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # 计算注意力输出和注意力权重
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 将多头注意力的输出重新合并
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)

        # 准备输出元组
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    @classmethod
    # 将隐藏维度拆分为多头注意力的大小和数量
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        # tensor: [bs, seq_len, hidden_size]
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(new_shape)
        # -> [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    @classmethod
    # 将注意力头的大小和数量合并为隐藏维度
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        # tensor [bs, num_attention_heads, seq_len, attn_head_size]
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # -> [bs, seq_len, num_attention_heads, attn_head_size]
        tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        # -> [bs, seq_len, hidden_size]
        return tensor
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # query, key, value分别表示查询、键、值的张量，维度为[批大小, 注意力头数, 序列长度, 每个注意力头大小]

        # 获取查询张量的维度信息
        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        # 获取键张量的长度信息
        key_length = key.size(-2)

        # 根据需要动态增加因果掩码的长度
        if key_length > self.bias.shape[-1]:
            self._init_bias(key_length, device=key.device)
        # 从预先存储的偏置中获取因果掩码
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        # 将查询、键张量重塑为二维矩阵以进行注意力计算
        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)

        # 初始化注意力分数张量，全零初始化
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
            alpha=self.norm_factor,
        )

        # 将注意力分数张量重新塑造为四维张量
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        # 创建一个最小浮点数的张量，用于掩码操作
        mask_value = torch.finfo(attn_scores.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype).to(attn_scores.device)

        # 根据因果掩码进行注意力分数的掩码处理
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        # 如果提供了注意力掩码，则应用该掩码
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # 对注意力分数进行 softmax 操作，以获取注意力权重
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.to(value.dtype)

        # 如果提供了头掩码，则对注意力权重进行掩码处理
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 对注意力权重应用注意力丢弃（dropout）
        attn_weights = self.attention_dropout(attn_weights)

        # 计算最终的注意力输出，使用注意力权重加权值张量
        attn_output = torch.matmul(attn_weights, value)

        # 返回注意力输出和注意力权重张量
        return attn_output, attn_weights
class GPTNeoXFlashAttention2(GPTNeoXAttention):
    """
    GPTNeoX flash attention module. This module inherits from `GPTNeoXAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        # 标记是否使用旧版 Flash Attention 的顶部左对齐掩码
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
        # 执行注意力机制的前向传播，处理输入的隐藏状态、注意力掩码和位置 ID，还有一些可选参数如头部掩码、历史信息、缓存和是否输出注意力权重
        def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
        ):
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
        # Determine if causal masking is needed based on `_flash_attn_uses_top_left_mask` and `query_length`
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # Temporary check for specific conditions related to Flash Attention for RoCm version 2.1
            causal = self.is_causal and query_length != 1

        # Check if there are any padding tokens in the input sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            # Unpad the input based on the attention mask
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # Perform Flash Attention on the unpadded inputs
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

            # Pad the attention output back according to the original indices
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # Perform regular Flash Attention without masking
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input with num_heads->num_attention_heads
    # 定义一个私有方法 `_upad_input`，用于处理注意力机制中的输入数据
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取不需要填充的数据的索引、当前序列长度和批次中的最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        
        # 获取批次大小、键值对序列长度、键值头数以及头维度
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
        
        # 重塑键层和值层，以便进行索引操作
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        
        # 根据查询长度的不同情况处理查询层
        if query_length == kv_seq_len:
            # 如果查询长度等于键值对序列长度，则直接索引查询层
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_attention_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            # 如果查询长度为1，则直接处理成适应的形状
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个 memcpy 操作，效率很差。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 否则，根据注意力掩码对查询层进行处理
            # -query_length: 切片假定左填充
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)
        
        # 返回处理后的查询层、键层、值层、查询层索引、当前序列长度元组和最大序列长度元组
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# 定义一个函数，用于处理注意力分数和左到右掩码
def attention_mask_func(attention_scores, ltor_mask):
    # 将注意力分数中掩码为假（False）的位置替换为一个极小的值，以此实现屏蔽效果
    attention_scores.masked_fill_(~ltor_mask, torch.finfo(attention_scores.dtype).min)
    return attention_scores

# 定义一个新的神经网络模块，用于实现旋转嵌入
class GPTNeoXRotaryEmbedding(nn.Module):
    # 从transformers库中的MistralRotaryEmbedding类的__init__方法复制而来
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        # 初始化旋转嵌入模块的参数
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 计算频率倒数，并将其作为缓冲区（buffer）注册到模块中
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 为了让`torch.jit.trace`正常工作，在这里构建cos和sin的缓存
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    # 设置cos和sin的缓存，用于旋转嵌入的计算
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        # 计算频率与位置的外积，生成用于旋转嵌入的cos和sin缓存
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # 按最后一个维度拼接cos和sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    # 前向传播函数，用于应用旋转嵌入到输入张量上
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]

        # 如果指定的序列长度大于当前缓存的最大序列长度，重新设置cos和sin的缓存
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # 返回当前缓存中的cos和sin，截取到指定的序列长度
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


# 从transformers库中的LlamaLinearScalingRotaryEmbedding类的__init__方法复制而来
class GPTNeoXLinearScalingRotaryEmbedding(GPTNeoXRotaryEmbedding):
    """GPTNeoXRotaryEmbedding扩展，添加了线性缩放功能。鸣谢Reddit用户/u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        # 初始化线性缩放旋转嵌入模块的参数
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    # 重写父类方法，设置缩放后的cos和sin的缓存
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor  # 应用缩放因子到位置编码中的时间步长

        # 计算缩放后的频率与位置的外积，生成用于旋转嵌入的cos和sin缓存
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # 按最后一个维度拼接cos和sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
class GPTNeoXDynamicNTKScalingRotaryEmbedding(GPTNeoXRotaryEmbedding):
    """GPTNeoXRotaryEmbedding扩展，增加了动态NTK缩放功能。由Reddit用户/u/bloc97和/u/emozilla贡献"""

    # 从transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding.__init__复制而来
    # TODO @gante 现在不再从那里复制
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        # 初始化函数，接收维度(dim)、最大位置嵌入数(max_position_embeddings)、基础值(base)、设备(device)和缩放因子(scaling_factor)
        self.scaling_factor = scaling_factor
        # 调用父类的初始化函数，传入维度、最大位置嵌入数、基础值和设备
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 设置余弦和正弦缓存，用于后续的旋转位置嵌入计算
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            # 如果序列长度超过最大位置嵌入数，则计算基础值
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            # 计算逆频率向量
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
            # 注册逆频率向量为缓存
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # 不同于论文中的实现，但使用不同的排列以达到相同的计算效果
        emb = torch.cat((freqs, freqs), dim=-1)
        # 注册余弦缓存
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        # 注册正弦缓存
        self.register_buffer("sin_cached", emb.sin(), persistent=False)


def rotate_half(x):
    """将输入的一半隐藏维度进行旋转。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# 从transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb复制而来
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """将旋转位置嵌入应用到查询和键张量上。"""
    # 使用给定的位置索引从余弦部分提取位置编码，并在指定维度上进行unsqueeze操作，以便与q和k进行广播匹配
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    
    # 使用给定的位置索引从正弦部分提取位置编码，并在指定维度上进行unsqueeze操作，以便与q和k进行广播匹配
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    
    # 将查询张量q与cos位置编码相乘，并将其与查询张量q经过rotate_half函数后的结果与sin位置编码相乘的结果相加，得到旋转后的查询张量
    q_embed = (q * cos) + (rotate_half(q) * sin)
    
    # 将键张量k与cos位置编码相乘，并将其与键张量k经过rotate_half函数后的结果与sin位置编码相乘的结果相加，得到旋转后的键张量
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    # 返回旋转后的查询张量和键张量作为元组
    return q_embed, k_embed
# 定义了一个名为 GPTNeoXMLP 的新神经网络模块，继承自 nn.Module 类
class GPTNeoXMLP(nn.Module):
    # 初始化方法，接收一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，将输入大小为 config.hidden_size，输出大小为 config.intermediate_size
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size)
        # 创建另一个线性层，将输入大小为 config.intermediate_size，输出大小为 config.hidden_size
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size)
        # 选择激活函数，根据配置中的 hidden_act 参数，从预定义的字典 ACT2FN 中获取对应的激活函数
        self.act = ACT2FN[config.hidden_act]

    # 前向传播方法，接收 hidden_states 作为输入
    def forward(self, hidden_states):
        # 输入 hidden_states 经过第一个线性层 dense_h_to_4h
        hidden_states = self.dense_h_to_4h(hidden_states)
        # 经过激活函数 act 处理后的 hidden_states
        hidden_states = self.act(hidden_states)
        # 再经过第二个线性层 dense_4h_to_h，最终输出
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


# 定义了一个名为 GPTNeoXLayer 的新神经网络模块，继承自 nn.Module 类
class GPTNeoXLayer(nn.Module):
    # 初始化方法，接收一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 根据配置参数设置是否使用并行残差连接
        self.use_parallel_residual = config.use_parallel_residual
        # 输入层的 Layer Normalization，输入大小为 config.hidden_size
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 经过注意力机制后的 Layer Normalization，输入大小同样为 config.hidden_size
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 注意力机制后的 Dropout 层，丢弃率为 config.hidden_dropout
        self.post_attention_dropout = nn.Dropout(config.hidden_dropout)
        # MLP（多层感知机）后的 Dropout 层，丢弃率同样为 config.hidden_dropout
        self.post_mlp_dropout = nn.Dropout(config.hidden_dropout)
        # 根据配置中的 _attn_implementation 参数选择相应的注意力机制类，并初始化
        self.attention = GPT_NEOX_ATTENTION_CLASSES[config._attn_implementation](config)
        # 创建一个 GPTNeoXMLP 对象，用于处理 MLP 部分
        self.mlp = GPTNeoXMLP(config)

    # 前向传播方法，接收多个输入参数
    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        ):
            # 使用 self.attention 方法处理输入的 hidden_states，应用 layer normalization
            # 这里包括了 attention_mask、position_ids、layer_past、head_mask 和 use_cache 等参数
            attention_layer_outputs = self.attention(
                self.input_layernorm(hidden_states),
                attention_mask=attention_mask,
                position_ids=position_ids,
                layer_past=layer_past,
                head_mask=head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # 获取注意力层的输出作为 attn_output
            attn_output = attention_layer_outputs[0]  # output_attn: attn_output, present, (attn_weights)
            # 对 attn_output 应用后续的 dropout
            attn_output = self.post_attention_dropout(attn_output)
            # outputs 包含 attention_layer_outputs 的其余部分
            outputs = attention_layer_outputs[1:]

            if self.use_parallel_residual:
                # 如果使用并行残差连接（parallel residual connection）
                # 通过 MLP 处理经过注意力后的层标准化（ln1），并应用 dropout
                mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
                mlp_output = self.post_mlp_dropout(mlp_output)
                # 更新 hidden_states 为 mlp_output、attn_output 和原始 hidden_states 的和
                hidden_states = mlp_output + attn_output + hidden_states
            else:
                # 如果不使用并行残差连接
                # 先将 attn_output 加到 hidden_states 上
                attn_output = attn_output + hidden_states
                # 然后通过 MLP 处理经过注意力后的层标准化（ln1），并应用 dropout
                mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
                mlp_output = self.post_mlp_dropout(mlp_output)
                # 更新 hidden_states 为 mlp_output 和 attn_output 的和
                hidden_states = mlp_output + attn_output

            if use_cache:
                # 如果 use_cache 为真，则在输出中包含 hidden_states
                outputs = (hidden_states,) + outputs  # hidden_states, present, (attn_weights)
            else:
                # 如果 use_cache 为假，则在输出中不包含 present
                outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

            # 返回最终的 outputs
            return outputs
GPT_NEOX_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~GPTNeoXConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GPT_NEOX_INPUTS_DOCSTRING = r"""
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
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    """Add model-specific documentation to the provided function or class.

    Args:
        docstring (`str`): The docstring to add to the function or class.

    Returns:
        Callable: A decorator function that adds the specified docstring to the decorated function or class.
    """
)
    # 定义一个字符串，描述GPTNeoX模型输出原始隐藏状态而不带任何特定的顶层头部
    "The bare GPTNeoX Model transformer outputting raw hidden-states without any specific head on top.",
    # GPT_NEOX_START_DOCSTRING用于标记GPTNeoX模型文档字符串的起始位置
    GPT_NEOX_START_DOCSTRING,
# 定义一个名为 GPTNeoXModel 的类，继承自 GPTNeoXPreTrainedModel
class GPTNeoXModel(GPTNeoXPreTrainedModel):
    # 初始化方法，接收一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将传入的 config 参数保存在实例变量 self.config 中
        self.config = config

        # 创建一个词嵌入层，参数为词汇表大小和隐藏层大小
        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        # 创建一个 dropout 层，参数为隐藏层的 dropout 概率
        self.emb_dropout = nn.Dropout(config.hidden_dropout)
        
        # 创建一个由多个 GPTNeoXLayer 组成的层列表，列表长度为 config.num_hidden_layers
        self.layers = nn.ModuleList([GPTNeoXLayer(config) for _ in range(config.num_hidden_layers)])
        
        # 创建一个 LayerNorm 层，用于最终的归一化处理，参数为隐藏层大小和 eps 值
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 根据配置文件判断是否使用 flash_attention_2
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # 梯度检查点默认关闭
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入词嵌入层的方法
    def get_input_embeddings(self):
        return self.embed_in

    # 设置输入词嵌入层的方法
    def set_input_embeddings(self, value):
        self.embed_in = value

    # 前向传播方法，接收多个输入参数
    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



# 添加描述模型的开始文档字符串，说明这是一个用于 CLM fine-tuning 的 GPTNeoX 模型
@add_start_docstrings(
    """GPTNeoX Model with a `language modeling` head on top for CLM fine-tuning.""", GPT_NEOX_START_DOCSTRING
)
# 定义一个名为 GPTNeoXForCausalLM 的类，继承自 GPTNeoXPreTrainedModel
class GPTNeoXForCausalLM(GPTNeoXPreTrainedModel):
    # 静态变量，指定与嵌入权重相关联的键名
    _tied_weights_keys = ["embed_out.weight"]

    # 初始化方法，接收一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建一个 GPTNeoXModel 类的实例，传入 config 参数
        self.gpt_neox = GPTNeoXModel(config)
        
        # 创建一个线性层，将隐藏层的输出映射到词汇表大小的输出空间，不使用偏置项
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出词嵌入层的方法
    def get_output_embeddings(self):
        return self.embed_out

    # 设置输出词嵌入层的方法
    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    # 前向传播方法，接收多个输入参数
    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Model forward method for transformer-like models.

        Args:
            input_ids (Optional[torch.LongTensor]): Input token IDs.
            attention_mask (Optional[torch.FloatTensor]): Mask to avoid performing attention on padding tokens.
            position_ids (Optional[torch.LongTensor]): Position IDs for positional embeddings.
            inputs_embeds (Optional[torch.FloatTensor]): Optional input embeddings directly provided instead of input_ids.
            head_mask (Optional[torch.FloatTensor]): Mask for attention heads.
            past_key_values (Optional[Tuple[Tuple[torch.FloatTensor]]]): Cached key-value states for fast autoregressive decoding.
            labels (Optional[torch.LongTensor]): Target labels for training.
            use_cache (Optional[bool]): Whether to use the cached past key-values for generation.
            output_attentions (Optional[bool]): Whether to return attention weights.
            output_hidden_states (Optional[bool]): Whether to return hidden states.
            return_dict (Optional[bool]): Whether to return a dictionary.

        Returns:
            model_inputs (Dict[str, torch.Tensor]): Dictionary containing model inputs.
        """
        # Implementation details of model forward pass goes here
        pass

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        """
        Prepares inputs for generation by adjusting input_ids and other necessary tensors.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            past_key_values (Optional[Tuple[Tuple[torch.FloatTensor]]]): Cached key-value states from previous decoding steps.
            attention_mask (torch.Tensor): Mask to avoid attending to padding tokens.
            inputs_embeds (torch.Tensor): Optional input embeddings.
            **kwargs: Additional keyword arguments.

        Returns:
            model_inputs (Dict[str, torch.Tensor]): Dictionary containing prepared model inputs.
        """
        input_shape = input_ids.shape

        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
        )

        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Reorders cached past key-values according to beam search index.

        Args:
            past_key_values (Tuple[Tuple[torch.FloatTensor]]): Cached key-value states.
            beam_idx (torch.Tensor): Index tensor for reordering.

        Returns:
            reordered_past (Tuple[Tuple[torch.FloatTensor]]): Reordered past key-values.
        """
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
"""
The GPTNeoX Model transformer with a sequence classification head on top (linear layer).

[`GPTNeoXForSequenceClassification`] uses the last token in order to do the classification, as other causal models
(e.g. GPT-1) do.

Since it does classification on the last token, it requires to know the position of the last token. If a
`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
each row of the batch).
"""
@add_start_docstrings(
    """
    The GPTNeoX Model transformer with a token classification head on top (linear layer).

    This model uses the GPTNeoX architecture and adds a linear layer on top for token classification tasks.

    It includes dropout and a linear classifier for the token classification layer.

    Since it performs classification on each token independently, it requires the position information for each token.
    If `pad_token_id` is defined in the configuration, it identifies tokens that are not padding tokens in each sequence.
    If `pad_token_id` is not defined, it uses the last token in each sequence. When using `inputs_embeds` instead of
    `input_ids`, the model assumes the last token in each sequence for classification.

    Note that the configuration checkpoint for this model can be found at "LarsJonasson/pythia-410m-deduped-sft-swedish".
    """,
    GPT_NEOX_START_DOCSTRING,
)
class GPTNeoXForTokenClassification(GPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Initialize GPTNeoX model
        self.gpt_neox = GPTNeoXModel(config)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(config.classifier_dropout)
        # Linear layer for classification
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint="LarsJonasson/pythia-410m-deduped-sft-swedish",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_loss=0.25,
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
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确定是否返回字典格式的输出，若未指定则使用模型配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 GPT-NeoX 模型，并获取输出
        outputs = self.gpt_neox(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取模型的隐藏状态并应用 dropout
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        
        # 将处理后的隐藏状态传递给分类器，得到分类器的 logits
        logits = self.classifier(hidden_states)

        # 计算损失（如果有提供标签）
        loss = None
        if labels is not None:
            # 将标签转移到与 logits 相同的设备上
            labels = labels.to(logits.device)
            # 使用交叉熵损失函数计算损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不要求返回字典格式的输出，则组装并返回输出元组
        if not return_dict:
            output = (logits,) + outputs[2:]  # 输出包括 logits 和可能的附加信息
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典格式的输出，则使用 TokenClassifierOutput 类构建输出
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
    The GPT-NeoX Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """
    # 使用 GPT-NeoX 模型变换器，顶部带有用于抽取式问答任务的跨度分类头，例如 SQuAD
    # （在隐藏状态输出的顶部添加线性层来计算“起始位置对数”和“结束位置对数”）
    @add_start_docstrings_to_model_forward(GPT_NEOX_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 将模型的前向传播方法添加描述性文档字符串，指定输入的文档字符串格式
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        real_checkpoint=_REAL_CHECKPOINT_FOR_DOC,
    )
    # 将模型的前向传播方法添加代码示例的文档字符串，包括用于文档的检查点、输出类型、配置类和真实检查点
    class GPTNeoXForQuestionAnswering(GPTNeoXPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.gpt_neox = GPTNeoXModel(config)
            self.qa_outputs = nn.Linear(config.hidden_size, 2)

            # 初始化权重并应用最终处理
            self.post_init()

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
        ):
            # 前向传播函数，接受多种输入，并根据需要返回字典或单个张量
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
        # 确保 return_dict 不为 None 则使用其值，否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 GPT-NeoX 模型进行推理
        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取序列输出
        sequence_output = outputs[0]

        # 使用序列输出计算问答模型的 logits
        logits = self.qa_outputs(sequence_output)

        # 将 logits 拆分为 start_logits 和 end_logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # 去除多余的维度，并确保连续性
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果 start_positions 或 end_positions 的维度大于 1，则压缩至一维，并转移到相应设备
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1).to(start_logits.device)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1).to(end_logits.device)

            # 忽略超出模型输入范围的位置索引
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，忽略指定的索引位置
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            # 计算总损失
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # 如果不返回字典，则返回元组形式的输出
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回 QuestionAnsweringModelOutput 类型的对象，包含损失、logits、隐藏状态和注意力权重
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```