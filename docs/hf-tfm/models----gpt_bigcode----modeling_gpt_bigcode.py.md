# `.\models\gpt_bigcode\modeling_gpt_bigcode.py`

```py
# 导入所需的模块和库
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入特定功能模块和函数
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_2_2
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)

# 如果系统支持 Flash Attention 2.0 及以上版本，导入相关函数和模块
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置信息
_CHECKPOINT_FOR_DOC = "bigcode/gpt_bigcode-santacoder"
_CONFIG_FOR_DOC = "GPTBigCodeConfig"

# 预训练模型存档列表
GPT_BIGCODE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bigcode/gpt_bigcode-santacoder",
    # 更多模型可以在 https://huggingface.co/models?filter=gpt_bigcode 查看
]

# 下面是一些使用 Torch JIT 脚本定义的函数，用于在 GPU 上优化计算效率

# 对输入进行 softmax 计算，并支持按条件屏蔽某些位置
@torch.jit.script
def upcast_masked_softmax(
    x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor, scale: float, softmax_dtype: torch.dtype
):
    # 将输入张量转换为指定的数据类型以提升计算效率
    input_dtype = x.dtype
    x = x.to(softmax_dtype) * scale
    # 根据掩码条件，将无效位置的值替换为指定的掩码值
    x = torch.where(mask, x, mask_value)
    # 在指定维度上进行 softmax 计算，并将结果转回原始输入张量的数据类型
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x

# 对输入进行 softmax 计算，并支持指定数据类型以提升计算效率
@torch.jit.script
def upcast_softmax(x: torch.Tensor, scale: float, softmax_dtype: torch.dtype):
    # 将输入张量转换为指定的数据类型以提升计算效率
    input_dtype = x.dtype
    x = x.to(softmax_dtype) * scale
    # 在指定维度上进行 softmax 计算，并将结果转回原始输入张量的数据类型
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x

# 对输入进行 softmax 计算，并支持按条件屏蔽某些位置
@torch.jit.script
def masked_softmax(x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor):
    # 根据掩码条件，将无效位置的值替换为指定的掩码值
    x = torch.where(mask, x, mask_value)
    # 使用 PyTorch 的 nn.functional.softmax 函数对张量 x 进行 softmax 操作，指定在最后一个维度上进行计算
    x = torch.nn.functional.softmax(x, dim=-1)
    # 返回经过 softmax 操作后的张量 x
    return x
# 定义一个函数 `_get_unpad_data`，用于处理注意力掩码。
def _get_unpad_data(attention_mask):
    # 计算每个样本序列的长度之和，结果为一个整数张量
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 找出所有非零元素的索引并展平，返回的是一维张量
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 计算批次中最大的序列长度，将其转换为 Python 整数
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算累积序列长度，使用零填充以保持形状
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # 返回处理后的结果元组
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# 定义一个名为 GPTBigCodeAttention 的类，继承自 nn.Module
class GPTBigCodeAttention(nn.Module):
    # 初始化方法，接收 config、is_cross_attention 和 layer_idx 参数
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        # 调用父类的初始化方法
        super().__init__()
        # 将 config 参数保存在实例变量中
        self.config = config

        # 初始化一些实例变量
        self.mask_value = None
        self.multi_query = config.multi_query
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.kv_heads = 1 if self.multi_query else self.num_heads
        self.kv_dim = self.kv_heads * self.head_dim
        self.split_size = self.embed_dim
        self.is_causal = True

        # 检查是否满足 embed_dim 能被 num_heads 整除的条件
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # 保存一些配置参数
        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        self.layer_idx = layer_idx
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = (
            config.scale_attention_softmax_in_fp32 and config.attention_softmax_in_fp32
        )
        self.attn_pdrop = config.attn_pdrop

        # 如果是交叉注意力模式
        if self.is_cross_attention:
            # 如果使用多查询，抛出未实现错误
            if self.multi_query:
                raise NotImplementedError("Multi-Query Attention not supported for cross_attention")

            # 创建一个线性层，用于跨注意力的内容注意力
            self.c_attn = nn.Linear(self.embed_dim, 2 * self.embed_dim)
            # 创建一个线性层，用于跨注意力的查询注意力
            self.q_attn = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            # 创建一个线性层，用于自注意力的内容和键值对注意力
            self.c_attn = nn.Linear(self.embed_dim, self.embed_dim + 2 * self.kv_dim)

        # 创建一个线性层，用于计算注意力后的投影
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # 创建一个注意力丢弃层
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        # 创建一个残差连接丢弃层
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    # 辅助方法，获取掩码值
    def _get_mask_value(self, device, dtype):
        # torch.where 函数期望一个张量，为了避免每次重新创建，使用缓存
        if self.mask_value is None or self.mask_value.dtype != dtype or self.mask_value.device != device:
            self.mask_value = torch.full([], torch.finfo(dtype).min, dtype=dtype, device=device)
        # 返回缓存的掩码值
        return self.mask_value
    # 定义前向传播函数，用于Transformer模型的自注意力机制或者交叉注意力机制
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]],
    ]:
        # 如果存在编码器隐藏状态，则执行交叉注意力机制的相关逻辑
        if encoder_hidden_states is not None:
            # 如果当前对象没有属性 "q_attn" 或者不是交叉注意力模式，则抛出值错误异常
            if not hasattr(self, "q_attn") or not self.is_cross_attention:
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPTBigCodeAttention(..., is_cross_attention=True)`."
                )

            # 使用当前对象的自注意力权重函数对隐藏状态进行查询
            query = self.q_attn(hidden_states)
            # 使用当前对象的交叉注意力权重函数对编码器隐藏状态进行键值映射
            key_value = self.c_attn(encoder_hidden_states)
            # 更新注意力掩码为编码器的注意力掩码
            attention_mask = encoder_attention_mask
        # 如果是多查询模式
        elif self.multi_query:
            # 使用当前对象的注意力权重函数对隐藏状态进行键值映射并分割为查询和键值对
            query, key_value = self.c_attn(hidden_states).split((self.embed_dim, 2 * self.kv_dim), dim=2)
        else:
            # 注意：我们将维度分割为 (self.num_heads, 3, self.head_dim) 而不是 (3, self.num_heads, self.head_dim)，
            # 即，内存布局与GPT2不同。
            # 这样可以更有效地与过去的键值对连接。
            query, key_value = (
                self.c_attn(hidden_states)
                .view(*hidden_states.shape[:2], self.num_heads, 3 * self.head_dim)
                .transpose(1, 2)
                .split((self.head_dim, 2 * self.head_dim), dim=3)
            )

        # 如果存在过去的层键值对，则将其与当前键值对连接起来
        if layer_past is not None:
            key_value = torch.cat((layer_past, key_value), dim=-2)
        # 如果使用缓存，则将当前键值对设置为输出的 "present"
        present = key_value if use_cache else None

        # 将键值对分割为键和值
        key, value = key_value.split((self.head_dim, self.head_dim), dim=-1)

        # 执行注意力计算，得到注意力输出和注意力权重
        attn_output, attn_weights = self._attn(query, key.transpose(-1, -2), value, attention_mask, head_mask)

        # 如果不是多查询模式，则转置注意力输出并重新整形为与隐藏状态相同的形状
        if not self.multi_query:
            attn_output = attn_output.transpose(1, 2).reshape(hidden_states.shape)
        # 使用当前对象的投影函数对注意力输出进行变换
        attn_output = self.c_proj(attn_output)
        # 对注意力输出进行残差连接的dropout处理
        attn_output = self.resid_dropout(attn_output)

        # 将注意力输出和 "present" 放入输出元组
        outputs = (attn_output, present)
        # 如果需要输出注意力权重，则将其添加到输出元组中
        if output_attentions:
            if self.multi_query:
                # 转置以返回通常格式的注意力权重 (batch_size, num_heads, query_length, key_length)
                attn_weights = attn_weights.transpose(1, 2)
            outputs += (attn_weights,)

        return outputs  # 返回注意力输出，"present"，(注意力权重)
# 定义了一个名为 GPTBigCodeFlashAttention2 的类，继承自 GPTBigCodeAttention 类。该模块用于处理 flash attention，保持权重不变。唯一需要修改的是前向传播，在其中正确调用 flash attention 的公共 API，并处理可能存在的填充标记。
class GPTBigCodeFlashAttention2(GPTBigCodeAttention):
    """
    GPTBigCode flash attention module. This module inherits from `GPTBigCodeAttention` as the weights of the module
    stays untouched. The only required change would be on the forward pass where it needs to correctly call the public
    API of flash attention and deal with padding tokens in case the input contains any of them.
    """

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__ 复制而来
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Once Flash Attention for RoCm is bumped to 2.1, this should be removed.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment,
        # that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        # Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # 注意，flash_attn<2.1 在 q_seqlen != k_seqlen 时（除非 q_seqlen == 1），生成的是错误的掩码（左上角）。
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    # 定义前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]],
    ]:
        # 前向传播方法的输入参数和返回值类型注释

        # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward 复制而来
        def _flash_attention_forward(
            self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
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
        if not self._flash_attn_uses_top_left_mask:
            # Determine if causal masking is required based on the model's configuration
            causal = self.is_causal
        else:
            # Temporary workaround until Flash Attention for RoCm version 2.1
            # Remove this condition when the issue is resolved, see LlamaFlashAttention2 __init__ for details
            causal = self.is_causal and query_length != 1

        # Check if there are any padding tokens in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            # Unpad the input sequences based on the attention mask
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            # Extract sequence lengths after unpadding
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # Compute attention scores for the unpad inputs
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

            # Pad the attention output back to original sequence length
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # Compute attention scores without considering padding (fallback case)
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
    # 定义一个方法用于处理输入数据，用于注意力机制
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取未填充数据的索引、当前序列长度及批次内最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # 获取批次大小、键值对序列长度、键值头数、头维度
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 重新组织键层数据，按未填充数据的索引进行索引
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 重新组织值层数据，按未填充数据的索引进行索引
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )

        # 根据查询长度调整查询层数据
        if query_length == kv_seq_len:
            # 当查询长度等于键值对序列长度时，按未填充数据的索引重新组织查询层数据
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            # 当查询长度为1时，处理查询层数据
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个内存复制操作，性能较差。
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 否则，根据查询长度和注意力掩码进行未填充数据处理
            # -query_length: 切片表示左填充操作
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的查询层、键层、值层、查询索引、序列长度信息元组
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
class GPTBigCodeSdpaAttention(GPTBigCodeAttention):
    # 继承自GPTBigCodeAttention类的SDPA注意力机制的实现
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]],
    ]:
        # SDPA注意力机制的前向传播函数，接受多个参数并返回输出张量和可能的额外输出
        pass  # 实际实现未提供，暂未实现具体的前向传播逻辑


class GPTBigCodeMLP(nn.Module):
    # 基于配置的GPT大型代码模型的多层感知机（MLP）实现
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        # 第一个线性层：将输入维度转换为中间维度
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        # 第二个线性层：将中间维度转换回原始嵌入维度
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        # 激活函数，根据配置选择
        self.act = ACT2FN[config.activation_function]
        # Dropout层，根据配置设置丢弃概率
        self.dropout = nn.Dropout(config.resid_pdrop)

    # 从transformers.models.gpt2.modeling_gpt2.GPT2MLP.forward复制而来
    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        # 多层感知机的前向传播逻辑
        hidden_states = self.c_fc(hidden_states)  # 线性变换
        hidden_states = self.act(hidden_states)   # 激活函数
        hidden_states = self.c_proj(hidden_states)  # 第二个线性变换
        hidden_states = self.dropout(hidden_states)  # Dropout
        return hidden_states


GPTBIGCODE_ATTENTION_CLASSES = {
    "eager": GPTBigCodeAttention,
    "flash_attention_2": GPTBigCodeFlashAttention2,
    "sdpa": GPTBigCodeSdpaAttention,  # SDPA注意力机制类
}


class GPTBigCodeBlock(nn.Module):
    # GPT大型代码模型的块，根据配置初始化层归一化、注意力、MLP等组件
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        # 内部维度，如果配置未指定，则为4倍隐藏层大小
        self.inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        # 第一层归一化层
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # 根据配置选择的注意力机制实现
        self.attn = GPTBIGCODE_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)

        # 第二层归一化层
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # 如果配置指定添加交叉注意力
        if config.add_cross_attention:
            if config.multi_query:
                raise NotImplementedError("Cross-attention not implemented for MQA")

            # 初始化交叉注意力
            self.crossattention = GPTBIGCODE_ATTENTION_CLASSES[config._attn_implementation](
                config, is_cross_attention=True, layer_idx=layer_idx
            )

            # 交叉注意力后的归一化层
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # 多层感知机模块
        self.mlp = GPTBigCodeMLP(self.inner_dim, config)
    # 定义模型的前向传播函数，用于处理输入的隐藏状态和一些可选参数，返回不同组合的输出元组
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.Tensor]],  # 输入的隐藏状态，可以是一个张量元组，可选
        layer_past: Optional[torch.Tensor] = None,  # 先前层的状态，可选，默认为 None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选，默认为 None
        head_mask: Optional[torch.Tensor] = None,  # 注意力头的掩码，可选，默认为 None
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态，可选，默认为 None
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器的注意力掩码，可选，默认为 None
        use_cache: Optional[bool] = False,  # 是否使用缓存，可选，默认为 False
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重，可选，默认为 False
    ) -> Union[
        Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        # 保存输入的隐藏状态作为残差连接的基准
        residual = hidden_states
        # 应用 Layer Normalization 到隐藏状态
        hidden_states = self.ln_1(hidden_states)
        # 调用注意力层处理隐藏状态
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 获取注意力层的输出
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        # 获取额外的输出（如果有的话）
        outputs = attn_outputs[1:]
        # 执行残差连接
        hidden_states = attn_output + residual

        # 如果存在编码器的隐藏状态
        if encoder_hidden_states is not None:
            # 添加一个用于交叉注意力的自注意力块
            if not hasattr(self, "crossattention"):
                # 如果未配置交叉注意力层，则抛出错误
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            # 保存当前隐藏状态作为残差连接的基准
            residual = hidden_states
            # 应用 Layer Normalization 到交叉注意力层的隐藏状态
            hidden_states = self.ln_cross_attn(hidden_states)
            # 调用交叉注意力层处理隐藏状态
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            # 获取交叉注意力层的输出
            attn_output = cross_attn_outputs[0]
            # 执行残差连接
            hidden_states = residual + attn_output
            # 添加交叉注意力权重到输出中（如果需要输出注意力权重）
            outputs = outputs + cross_attn_outputs[2:]

        # 保存当前隐藏状态作为残差连接的基准
        residual = hidden_states
        # 应用 Layer Normalization 到隐藏状态
        hidden_states = self.ln_2(hidden_states)
        # 应用 MLP 层处理隐藏状态
        feed_forward_hidden_states = self.mlp(hidden_states)
        # 执行残差连接
        hidden_states = residual + feed_forward_hidden_states

        # 如果需要使用缓存，则将当前隐藏状态添加到输出中
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            # 否则，只保留除了隐藏状态以外的输出部分
            outputs = (hidden_states,) + outputs[1:]

        # 返回最终的输出元组
        return outputs  # hidden_states, present, (attentions, cross_attentions)
class GPTBigCodePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 GPTBigCodeConfig 作为配置类
    config_class = GPTBigCodeConfig
    # 指定基础模型的前缀名称
    base_model_prefix = "transformer"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要拆分的模块名称列表
    _no_split_modules = ["GPTBigCodeBlock"]
    # 跳过设备放置的键名
    _skip_keys_device_placement = "past_key_values"
    # 支持 Flash Attention 2
    _supports_flash_attn_2 = True
    # 支持 Self-Dual-Path Attention (SDPA)
    _supports_sdpa = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (GPTBigCodeMLP, GPTBigCodeAttention)):
            # 根据 OpenAI GPT-2 论文中的方案重新初始化选定的权重：
            #   > 使用修改后的初始化，考虑模型深度上残差路径的累积。在初始化时，通过 1/√N 缩放残差层的权重，
            #   > 其中 N 是残差层的数量。
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # 参考 (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            module.c_proj.weight.data.normal_(
                mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer))
            )
            module.c_proj._is_hf_initialized = True
        elif isinstance(module, nn.Linear):
            # 与 TF 版本略有不同，TF 版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


GPT_BIGCODE_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.


"""
    Parameters:
        config ([`GPTBigCodeConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
GPT_BIGCODE_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare GPT_BIGCODE Model transformer outputting raw hidden-states without any specific head on top.",
    GPT_BIGCODE_START_DOCSTRING,
)
class GPTBigCodeModel(GPTBigCodePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.multi_query = config.multi_query  # 从配置中获取多查询选项
        self.embed_dim = config.hidden_size  # 从配置中获取嵌入维度大小

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)  # 词嵌入层，根据词汇表大小和嵌入维度创建
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)  # 位置嵌入层，根据最大位置嵌入大小和嵌入维度创建

        self.drop = nn.Dropout(config.embd_pdrop)  # Dropout层，根据配置中的嵌入dropout概率创建
        self.h = nn.ModuleList([GPTBigCodeBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])  # 多层GPTBigCodeBlock组成的层列表
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)  # Layer normalization层，根据嵌入维度和配置中的epsilon创建

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias", torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)), persistent=False
        )  # 创建一个下三角矩阵作为偏置，类型为bool，注册为模型的缓冲区

        self.gradient_checkpointing = False  # 梯度检查点开关，默认关闭

        self._use_sdpa = config._attn_implementation == "sdpa"  # 根据配置中的注意力实现类型判断是否使用sdpa
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"  # 根据配置中的注意力实现类型判断是否使用flash_attention_2

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.wte  # 返回输入嵌入层对象

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings  # 设置新的输入嵌入层对象

    @add_start_docstrings_to_model_forward(GPT_BIGCODE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 此处应该包含模型前向传播的详细文档字符串和示例代码注释，但根据示例我们只输出类的注释部分
        pass


@add_start_docstrings(
    """
    The GPT_BIGCODE Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT_BIGCODE_START_DOCSTRING,
)
class GPTBigCodeForCausalLM(GPTBigCodePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]  # 定义与输入嵌入层权重相关联的权重键
    def __init__(self, config):
        # 调用父类的初始化方法，传递配置参数
        super().__init__(config)
        # 使用给定配置创建 GPTBigCodeModel 模型
        self.transformer = GPTBigCodeModel(config)
        # 创建线性层用于语言模型的输出
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 初始化模型权重并进行最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 返回语言模型头部，即线性层
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置新的输出嵌入
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        
        # 如果存在过去的键值（past_key_values），则移除已覆盖的token
        if past_key_values:
            if self.config.multi_query:
                past_length = past_key_values[0].shape[1]
            else:
                past_length = past_key_values[0].shape[2]

            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1

            # 移除已覆盖的token
            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        # 如果存在attention_mask但不存在position_ids，则动态创建position_ids用于批处理生成
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # 如果传入了inputs_embeds，只在第一次生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 更新模型输入参数
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

    @add_start_docstrings_to_model_forward(GPT_BIGCODE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义 Transformer 模型的前向传播方法，用于推断或训练过程中的正向计算
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的 token IDs，可以为 None
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 用于存储过去的键值对，可以为 None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，用于指定哪些位置的 token 需要被忽略
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 IDs，如用于区分句子 A 和句子 B
        position_ids: Optional[torch.Tensor] = None,  # 位置 IDs，标识每个 token 的位置信息
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，用于指定哪些注意力头部需要被忽略
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入表示，代替 input_ids 使用
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态，用于编码器-解码器结构
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器的注意力掩码
        labels: Optional[torch.Tensor] = None,  # 预测的标签，用于计算损失
        use_cache: Optional[bool] = None,  # 是否使用缓存，用于存储中间计算结果以加速解码
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回一个字典作为输出
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 如果 return_dict 不是 None，则使用其值；否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 transformer 处理输入数据，获取 transformer 的输出结果
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取 transformer 的隐藏状态
        hidden_states = transformer_outputs[0]

        # 使用 lm_head 对隐藏状态进行预测，得到语言模型的 logits
        lm_logits = self.lm_head(hidden_states)

        # 初始化损失值
        loss = None
        # 如果存在 labels，则计算损失
        if labels is not None:
            # 将 logits 向左移动一个位置，以便于标签预测下一个位置的 token
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            # 将预测的 token 和标签展开，计算交叉熵损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 如果 return_dict 为 False，则返回输出的元组
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回带有交叉注意力的 CausalLMOutputWithCrossAttentions 对象
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        # 根据 beam_idx 重新排序 past_key_values 的缓存，以匹配每个生成步骤的正确 beam_idx
        return tuple(layer_past.index_select(0, beam_idx.to(layer_past.device)) for layer_past in past_key_values)
"""
The GPTBigCode Model transformer with a sequence classification head on top (linear layer).

[`GPTBigCodeForSequenceClassification`] uses the last token in order to do the classification, as other causal
models (e.g. GPT-1) do.

Since it does classification on the last token, it requires to know the position of the last token. If a
`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
each row of the batch).
"""
@add_start_docstrings(
    """
    GPT_BIGCODE Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    GPT_BIGCODE_START_DOCSTRING,
)
class GPTBigCodeForTokenClassification(GPTBigCodePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = GPTBigCodeModel(config)
        # Determine the dropout rate for the classifier based on the configuration
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        # Define the linear classifier layer for token classification
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
    @add_start_docstrings_to_model_forward(GPT_BIGCODE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 若未指定 return_dict，则使用模型配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 transformer 的 forward 方法，传递所有参数
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

        # 获取 transformer 输出的隐藏状态
        hidden_states = transformer_outputs[0]
        # 对隐藏状态应用 dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 将处理后的隐藏状态输入分类器得到 logits
        logits = self.classifier(hidden_states)

        # 初始化损失值为 None
        loss = None
        # 如果提供了标签，则计算损失值
        if labels is not None:
            # 使用交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 将 logits 和标签视图转换为合适的形状并计算损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).to(logits.device))

        # 如果不要求返回字典格式的输出，则按需返回元组形式的输出
        if not return_dict:
            output = (logits,) + transformer_outputs[2:]  # 按顺序拼接输出元组
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 对象，其中包含损失、logits、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
```