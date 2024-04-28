# `.\models\gpt_bigcode\modeling_gpt_bigcode.py`

```py
# 设置文件编码格式为utf-8

# 导入所需的库
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入其他相关模块
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from .configuration_gpt_bigcode import GPTBigCodeConfig

# 检查是否已安装flash_attn，并导入相应的函数和模块
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 获取logger
logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bigcode/gpt_bigcode-santacoder"  # 用于生成文档的检查点
_CONFIG_FOR_DOC = "GPTBigCodeConfig"  # 用于生成文档的配置

# 预训练模型列表
GPT_BIGCODE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bigcode/gpt_bigcode-santacoder",
    # 查看所有GPTBigCode模型：https://huggingface.co/models?filter=gpt_bigcode
]

# 融合的内核
# 对于每种情况使用单独的函数，因为条件语句限制了内核融合
# TODO：根据缩放、丢弃和头遮罩，可能有更好的融合内核
# 是否可以在不编写32个函数的情况下完成？
@torch.jit.script
def upcast_masked_softmax(
    x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor, scale: float, softmax_dtype: torch.dtype
):
    input_dtype = x.dtype
    # 将输入向量升级为指定类型，然后乘以缩放因子，并在需要的位置填充掩码值
    x = x.to(softmax_dtype) * scale
    x = torch.where(mask, x, mask_value)
    # 对向量进行softmax计算，然后将结果转换回原始类型
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x

@torch.jit.script
def upcast_softmax(x: torch.Tensor, scale: float, softmax_dtype: torch.dtype):
    input_dtype = x.dtype
    # 将输入向量升级为指定类型，然后乘以缩放因子
    x = x.to(softmax_dtype) * scale
    # 对向量进行softmax计算，然后将结果转换回原始类型
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x

@torch.jit.script
def masked_softmax(x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor):
    # 在需要的位置填充掩码值，然后对向量进行softmax计算
    x = torch.where(mask, x, mask_value)
    x = torch.nn.functional.softmax(x, dim=-1)
    return x
# 从transformers.models.llama.modeling_llama._get_unpad_data中复制过来的函数
def _get_unpad_data(attention_mask):
    # 计算每个序列在批次中的长度
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 获取非零元素的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 获取批次中最大的序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 对序列长度进行累积求和并在前面填充一个0
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

# GPTBigCodeAttention类定义
class GPTBigCodeAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config

        self.mask_value = None
        self.multi_query = config.multi_query
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.kv_heads = 1 if self.multi_query else self.num_heads
        self.kv_dim = self.kv_heads * self.head_dim
        self.split_size = self.embed_dim
        self.is_causal = True

        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        self.layer_idx = layer_idx
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = (
            config.scale_attention_softmax_in_fp32 and config.attention_softmax_in_fp32
        )
        self.attn_pdrop = config.attn_pdrop

        # 如果是跨注意力，执行以下操作
        if self.is_cross_attention:
            if self.multi_query:
                raise NotImplementedError("Multi-Query Attention not supported for cross_attention")

            # 针对交叉注意力的输入线性变换
            self.c_attn = nn.Linear(self.embed_dim, 2 * self.embed_dim)
            self.q_attn = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            # 针对自注意力的输入线性变换
            self.c_attn = nn.Linear(self.embed_dim, self.embed_dim + 2 * self.kv_dim)

        # 结合查询和值的线性变换
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # 注意力层的dropout
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        # 残差连接的dropout
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    # 获取遮蔽值
    def _get_mask_value(self, device, dtype):
        # torch.where需要一个张量。我们使用缓存来避免每次重新创建
        if self.mask_value is None or self.mask_value.dtype != dtype or self.mask_value.device != device:
            self.mask_value = torch.full([], torch.finfo(dtype).min, dtype=dtype, device=device)
        return self.mask_value
    # 定义前向传播函数，接受输入隐藏状态，先前的层缓存，注意力掩码，头屏蔽，编码器隐藏状态，编码器注意力掩码，是否使用缓存，是否输出注意力分布
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
        # 如果存在编码器的隐藏状态
        if encoder_hidden_states is not None:
            # 如果self中没有属性"q_attn"或者不是交叉注意力模式，抛出值错误
            if not hasattr(self, "q_attn") or not self.is_cross_attention:
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPTBigCodeAttention(..., is_cross_attention=True)`."
                )

            # 使用self.q_attn对隐藏状态进行查询
            query = self.q_attn(hidden_states)
            # 使用self.c_attn对编码器隐藏状态进行键值对计算
            key_value = self.c_attn(encoder_hidden_states)
            # 更新注意力掩码为编码器的注意力掩码
            attention_mask = encoder_attention_mask
        # 如果是多个查询条件
        elif self.multi_query:
            # 使用self.c_attn计算隐藏状态的键值对，并拆分为查询和键值
            query, key_value = self.c_attn(hidden_states).split((self.embed_dim, 2 * self.kv_dim), dim=2)
        else:
            # 对隐藏状态进行键值对计算，按(self.num_heads, 3, self.head_dim)的方式拆分，用于更高效地与过去的键值对进行连接
            query, key_value = (
                self.c_attn(hidden_states)
                .view(*hidden_states.shape[:2], self.num_heads, 3 * self.head_dim)
                .transpose(1, 2)
                .split((self.head_dim, 2 * self.head_dim), dim=3)
            )

        # 如果存在先前的层缓存
        if layer_past is not None:
            # 将过去的键值对与现在的键值对进行连接
            key_value = torch.cat((layer_past, key_value), dim=-2)
        # 如果使用缓存，则更新现在的键值对
        present = key_value if use_cache else None

        # 对键值对拆分为键和值
        key, value = key_value.split((self.head_dim, self.head_dim), dim=-1)

        # 使用自定义的_attn函数进行自注意力计算
        attn_output, attn_weights = self._attn(query, key.transpose(-1, -2), value, attention_mask, head_mask)

        # 如果不是多个查询条件，则转置输出维度并重塑形状
        if not self.multi_query:
            attn_output = attn_output.transpose(1, 2).reshape(hidden_states.shape)
        # 将注意力输出通过映射层和残差dropout
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        # 如果输出注意力权重
        if output_attentions:
            if self.multi_query:
                # 转置以返回常规格式的注意力权重 (batch_size, num_heads, query_length, key_length)
                attn_weights = attn_weights.transpose(1, 2)
            outputs += (attn_weights,)

        # 返回输出元组：attn_output, present, (attentions)
        return outputs
class GPTBigCodeFlashAttention2(GPTBigCodeAttention):
    """
    GPTBigCode flash attention module. This module inherits from `GPTBigCodeAttention` as the weights of the module
    stays untouched. The only required change would be on the forward pass where it needs to correctly call the public
    API of flash attention and deal with padding tokens in case the input contains any of them.
    """

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__ 复制而来
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化函数
        super().__init__(*args, **kwargs)

        # TODO: 一旦 RoCm 上的 Flash Attention 升级到 2.1，应该删除此处的内容。
        # flash_attn
    ):
        """
        调用 Flash Attention 的前向方法 - 如果输入隐藏状态至少包含一个填充标记，则首先取消填充输入，然后计算注意力分数并填充最终的注意力分数。

        Args:
            query_states (`torch.Tensor`):
                要传递给 Flash Attention API 的输入查询状态
            key_states (`torch.Tensor`):
                要传递给 Flash Attention API 的输入键状态
            value_states (`torch.Tensor`):
                要传递给 Flash Attention API 的输入值状态
            attention_mask (`torch.Tensor`):
                填充遮罩 - 对应于大小为 `(batch_size, seq_len)` 的张量，其中 0 表示填充标记的位置，1 表示非填充标记的位置。
            dropout (`int`, *optional*):
                注意力丢弃率
            softmax_scale (`float`, *optional*):
                应用 softmax 前的 QK^T 缩放。默认为 1 / sqrt(head_dim)
        """
        如果 self._flash_attn_uses_top_left_mask 为 False，则 causal 设置为 is_causal
        否则，当 query_length 不等于 1 时，将 causal 设置为 is_causal，待移除 `query_length != 1` 检查一旦 Flash Attention for RoCm 升级到 2.1 时。详情请参阅 LlamaFlashAttention2 __init__ 中的注释。

        # 序列中至少包含一个填充标记
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            将输入的查询状态、键状态、值状态、注意力遮罩、查询长度传递给 _upad_input 方法，以取消填充输入
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

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

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            使用 flash_attn_func 计算注意力
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        返回注意力输出结果

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input 复制过来的
``` 
    # 定义一个私有方法用于处理注意力机制的输入数据
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取未填充数据的索引、当前序列长度和批次中的最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # 获取批次大小、键值序列长度、键值头的数量和头维度
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # 根据索引重排键层和值层，以便与查询层对应
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )

        # 如果查询长度等于键值序列长度，则直接使用索引后的查询层和相应的长度和索引
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        # 如果查询长度为1，则使用特定的处理方式
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            # 生成一个序列长度为批次大小+1的张量，用于描述未填充数据的位置
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            # 剔除最后一个元素，得到有效的索引
            indices_q = cu_seqlens_q[:-1]
            # 压缩查询层的第一个维度
            query_layer = query_layer.squeeze(1)
        else:
            # 对于非常规的查询长度，采用unpad_input方法处理
            # 假设-q_len:切片表示左填充
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        # 返回处理后的查询层、键层、值层、查询层索引、序列长度元组和最大序列长度元组
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
class GPTBigCodeSdpaAttention(GPTBigCodeAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        layer_past: Optional[torch.Tensor] = None,  # 过去层的状态张量，默认为None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩张量，默认为None
        head_mask: Optional[torch.Tensor] = None,  # 头部遮罩张量，默认为None
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态张量，默认为None
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器的注意力遮罩张量，默认为None
        use_cache: Optional[bool] = False,  # 是否使用缓存，默认为False
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重，默认为False
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],  # 返回隐藏状态张量和可选的过去层状态张量的元组
        Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]],  # 返回隐藏状态张量、可选的过去层状态张量和注意力权重元组的元组
class GPTBigCodeMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size  # 获取嵌入维度
        self.c_fc = nn.Linear(embed_dim, intermediate_size)  # 全连接层，输入维度为嵌入维度，输出维度为中间维度
        self.c_proj = nn.Linear(intermediate_size, embed_dim)  # 全连接层，输入维度为中间维度，输出维度为嵌入维度
        self.act = ACT2FN[config.activation_function]  # 激活函数，根据配置选择相应的激活函数
        self.dropout = nn.Dropout(config.resid_pdrop)  # Dropout层，以指定的概率丢弃输入

    # 从transformers.models.gpt2.modeling_gpt2.GPT2MLP.forward复制而来
    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)  # 全连接层
        hidden_states = self.act(hidden_states)  # 激活函数
        hidden_states = self.c_proj(hidden_states)  # 全连接层
        hidden_states = self.dropout(hidden_states)  # Dropout层
        return hidden_states  # 返回处理后的隐藏状态


GPTBIGCODE_ATTENTION_CLASSES = {
    "eager": GPTBigCodeAttention,  # “eager”对应的注意力机制类为GPTBigCodeAttention
    "flash_attention_2": GPTBigCodeFlashAttention2,  # “flash_attention_2”对应的注意力机制类为GPTBigCodeFlashAttention2
    "sdpa": GPTBigCodeSdpaAttention,  # “sdpa”对应的注意力机制类为GPTBigCodeSdpaAttention
}


class GPTBigCodeBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size  # 获取隐藏状态的维度
        self.inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size  # 内部维度为配置中指定的内部维度，若未指定则为隐藏状态维度的四倍

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)  # 第一个LayerNorm层，对隐藏状态进行归一化

        self.attn = GPTBIGCODE_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)  # 根据配置选择相应的注意力机制类实例化

        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)  # 第二个LayerNorm层，对隐藏状态进行归一化

        if config.add_cross_attention:  # 如果配置中包含交叉注意力
            if config.multi_query:  # 如果配置中包含多查询（multi_query）
                raise NotImplementedError("Cross-attention not implemented for MQA")  # 抛出未实现的错误信息

            self.crossattention = GPTBIGCODE_ATTENTION_CLASSES[config._attn_implementation](  # 实例化交叉注意力机制类
                config, is_cross_attention=True, layer_idx=layer_idx
            )

            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)  # 交叉注意力的LayerNorm层，对隐藏状态进行归一化

        self.mlp = GPTBigCodeMLP(self.inner_dim, config)  # 实例化多层感知机
    # 定义 Transformer 模型的前向传播方法
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.Tensor]],  # 输入的隐藏状态，可选
        layer_past: Optional[torch.Tensor] = None,  # 上一层的输出，可选
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩，可选
        head_mask: Optional[torch.Tensor] = None,  # 头部遮罩，可选
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态，可选
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器的注意力遮罩，可选
        use_cache: Optional[bool] = False,  # 是否使用缓存，可选
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重，可选
    ) -> Union[
        Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        # 保留原始输入作为残差
        residual = hidden_states
        # LayerNorm 层，对隐藏状态进行归一化
        hidden_states = self.ln_1(hidden_states)
        # 多头自注意力机制
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # 自注意力输出结果
        outputs = attn_outputs[1:]  # 其它输出结果（可能包括 present, (attentions)）

        # 残差连接
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # 添加一个自注意力块，用于交叉注意力
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            # 保留残差
            residual = hidden_states
            # LayerNorm 层，对隐藏状态进行归一化
            hidden_states = self.ln_cross_attn(hidden_states)
            # 交叉注意力机制
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]  # 交叉注意力输出结果
            # 残差连接
            hidden_states = residual + attn_output
            # 添加交叉注意力的输出结果
            outputs = outputs + cross_attn_outputs[2:]  # 如果需要输出注意力权重，添加交叉注意力的注意力权重

        # 保留残差
        residual = hidden_states
        # LayerNorm 层，对隐藏状态进行归一化
        hidden_states = self.ln_2(hidden_states)
        # MLP 层
        feed_forward_hidden_states = self.mlp(hidden_states)
        # 残差连接
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        # 返回结果（隐藏状态，present, (attentions, cross_attentions)）
        return outputs
class GPTBigCodePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 GPTBigCodeConfig
    config_class = GPTBigCodeConfig
    # 定义模型中与权重相关的模块前缀为 "transformer"
    base_model_prefix = "transformer"
    # 表明支持梯度检查点
    supports_gradient_checkpointing = True
    # 定义不需要拆分的模块名称列表
    _no_split_modules = ["GPTBigCodeBlock"]
    # 定义跳过设备放置的键值
    _skip_keys_device_placement = "past_key_values"
    # 表明支持 Flash Attention 2
    _supports_flash_attn_2 = True
    # 表明支持 SDPA（Scaled Dot-Product Attention）
    _supports_sdpa = True

    def __init__(self, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        # 如果模块是 GPTBigCodeMLP 或 GPTBigCodeAttention 类的实例
        if isinstance(module, (GPTBigCodeMLP, GPTBigCodeAttention)):
            # 重新初始化选定的权重，遵循 OpenAI GPT-2 论文方案：
            #   > 使用改进的初始化，考虑到残差路径随着模型深度的累积。在初始化时，通过因子 1/√N 对残差层的权重进行缩放，
            #   > 其中 N 是残差层数。
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # 参考（Megatron-LM）：https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            module.c_proj.weight.data.normal_(
                mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer))
            )
            # 标记权重已由 Hugging Face 初始化
            module.c_proj._is_hf_initialized = True
        # 如果模块是 nn.Linear 类的实例
        elif isinstance(module, nn.Linear):
            # 与 TF 版本略有不同，TF 版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，则初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 nn.Embedding 类的实例
        elif isinstance(module, nn.Embedding):
            # 初始化权重为正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，则将相应位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是 nn.LayerNorm 类的实例
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置为零
            module.bias.data.zero_()
            # 初始化权重为1
            module.weight.data.fill_(1.0)


# 模型文档字符串的起始部分
GPT_BIGCODE_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.


"""
    Parameters:
        config ([`GPTBigCodeConfig`]): 模型配置类，包含模型的所有参数。
            通过配置文件初始化不会加载与模型关联的权重，只会加载配置。
            可以使用 [`~PreTrainedModel.from_pretrained`] 方法加载模型权重。
"""
GPT_BIGCODE_INPUTS_DOCSTRING = r"""
"""
"""

# 定义 GPTBigCodeModel 类，这个类是 Transformer 模型的通用类，输出原始隐藏状态而不是特定的输出层
@add_start_docstrings(
    "The bare GPT_BIGCODE Model transformer outputting raw hidden-states without any specific head on top.",
    GPT_BIGCODE_START_DOCSTRING,
)
class GPTBigCodeModel(GPTBigCodePreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取配置信息
        self.multi_query = config.multi_query
        self.embed_dim = config.hidden_size

        # 定义词嵌入层和位置嵌入层
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        # Dropout 层
        self.drop = nn.Dropout(config.embd_pdrop)
        # 多层 Transformer 模块
        self.h = nn.ModuleList([GPTBigCodeBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        # Layer normalization
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # 初始化一个下三角矩阵作为偏置
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias", torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)), persistent=False
        )

        self.gradient_checkpointing = False

        # 根据不同配置选择使用的注意力机制
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入词嵌入
    def get_input_embeddings(self):
        return self.wte

    # 设置输入词嵌入
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    # 定义前向传播方法
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


# 定义 GPTBigCodeForCausalLM 类，这个类是在 Transformer 模型的基础上增加了语言建模头部的类
@add_start_docstrings(
    """
    The GPT_BIGCODE Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT_BIGCODE_START_DOCSTRING,
)
class GPTBigCodeForCausalLM(GPTBigCodePreTrainedModel):
    # 定义被绑定权重的键名
    _tied_weights_keys = ["lm_head.weight"]
    # 初始化方法，传入配置参数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)
        # 创建一个 GPTBigCodeModel 的实例并赋值给 transformer 属性
        self.transformer = GPTBigCodeModel(config)
        # 创建一个线性层并赋值给 lm_head 属性，用于模型输出的最终处理
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 调用后续初始化方法
        self.post_init()

    # 返回 lm_head 属性的值，即模型输出处理的线性层
    def get_output_embeddings(self):
        return self.lm_head

    # 将新的嵌入层赋值给 lm_head 属性
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 为生成准备输入，处理输入数据
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # 获取额外的标记类型信息
        token_type_ids = kwargs.get("token_type_ids", None)
        
        # 通过 past_key_values 检查需要忽略的标记
        if past_key_values:
            # 根据模型配置的查询方式，获取已知标记的长度
            if self.config.multi_query:
                past_length = past_key_values[0].shape[1]
            else:
                past_length = past_key_values[0].shape[2]

            # 判断输入 ID 是否超出已知标记的长度，进行覆盖处理
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
            
            # 如果存在标记类型信息，同样进行相应的截取
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        # 获取额外的注意力掩码和位置 ID 信息
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        # 如果存在注意力掩码但不存在位置 ID，则动态生成位置 ID 用于批量生成
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # 如果存在 past_key_values，则进行位置 ID 的截取
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None
        
        # 如果存在 inputs_embeds，并且不存在 past_key_values，则仅在第一次生成时使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 更新 model_inputs 字典中的各种输入信息
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        # 返回处理后的输入信息
        return model_inputs

    # 为模型前向方法添加文档注释和代码示例注释
    @add_start_docstrings_to_model_forward(GPT_BIGCODE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义一个前向传播函数，接受多个参数，类型为可选的 torch.Tensor 类型
    # input_ids: 输入的 token 序列
    # past_key_values: 用于存储过去的 attention key 和 value 的元组
    # attention_mask: 注意力遮罩，指示哪些位置需要被注意，哪些位置不需要
    # token_type_ids: token 类型标识符，用于区分不同句子的 tokens
    # position_ids: 位置标识符，用于指示每个 token 的位置
    # head_mask: 指定哪些头应该被遮蔽
    # inputs_embeds: 要直接提供的嵌入向量，而不是使用输入 token 下的自动嵌入查找
    # encoder_hidden_states: 编码器的隐藏状态
    # encoder_attention_mask: 编码器的注意力遮罩
    # labels: 损失函数的标签
    # use_cache: 是否使用缓存
    # output_attentions: 是否输出注意力权重
    # output_hidden_states: 是否输出隐藏状态
    # return_dict: 是否返回字典形式的结果
        ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 设置是否返回字典结果，默认为配置参数中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 对 transformer 进行前向传播得到输出
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
        # 获取模型输出的隐藏状态
        hidden_states = transformer_outputs[0]

        # 根据隐藏状态计算语言模型的logits
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # 由于模型内部对 labels 进行了位移，此处也做相应的位移操作
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            # 展平 tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有交叉注意力的 CausalLMOutputWithCrossAttentions 类
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
        # 用于重新排序 `past_key_values` 缓存，当调用 `PreTrainedModel.beam_search` 或 `PreTrainedModel.beam_sample` 时使用，
        # 这是为了在每个生成步骤中将 `past_key_values` 与正确的 beam_idx 匹配
        return tuple(layer_past.index_select(0, beam_idx.to(layer_past.device)) for layer_past in past_key_values)
# 为GPTBigCodeForSequenceClassification添加文档字符串，说明这个模型是在GPTBigCode模型之上增加了一个用于序列分类的头部（线性层）
# 在进行分类时，使用最后一个标记来执行分类，与其他因果模型（如GPT-1）类似
# 当最后一个标记无法确定时，会根据配置中的pad_token_id找到每一行不是填充标记的最后一个标记，如果没有定义pad_token_id，会取批次中每一行的最后一个值
# 当传入inputs_embeds而不是input_ids时，会执行相同的操作（取批次中每一行的最后一个值）
@add_start_docstrings(
    """
    The GPTBigCode Model transformer with a sequence classification head on top (linear layer).

    [`GPTBigCodeForSequenceClassification`] uses the last token in order to do the classification, as other causal
    models (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GPT_BIGCODE_START_DOCSTRING,
)
class GPTBigCodeForSequenceClassification(GPTBigCodePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPTBigCodeModel(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

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
    )

# 为GPTBigCodeForTokenClassification添加文档字符串，说明这个模型是在GPTBigCode模型之上增加了一个用于标记分类的头部（隐藏状态输出之上的线性层），例如用于命名实体识别（NER）任务
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
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
    @add_start_docstrings_to_model_forward(GPT_BIGCODE_INPUTS_DOCSTRING)
    # 为模型的 forward 方法添加文档字符串
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
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 检查是否应该返回字典形式的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 Transformer 模型的 forward 方法
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

        # 从 Transformer 输出中获取隐藏状态
        hidden_states = transformer_outputs[0]
        # 对隐藏状态应用 dropout
        hidden_states = self.dropout(hidden_states)
        # 使用分类器计算 logits
        logits = self.classifier(hidden_states)

        loss = None
        # 如果有标签，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).to(logits.device))

        # 如果不需要返回字典形式的结果
        if not return_dict:
            # 组装输出元组
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
```