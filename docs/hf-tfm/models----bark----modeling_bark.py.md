# `.\transformers\models\bark\modeling_bark.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权使用此文件
# 除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取有关权限和限制的详细信息
""" PyTorch BARK model."""
# 导入所需的库
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# 导入自定义的模块
from ...generation.logits_process import (
    AlternatingCodebooksLogitsProcessor,
    BarkEosPrioritizerLogitsProcessor,
    SuppressTokensLogitsProcessor,
)
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput
from ...modeling_utils import PreTrainedModel, get_parameter_device
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_accelerate_available,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from ..auto import AutoModel
from .configuration_bark import (
    BarkCoarseConfig,
    BarkConfig,
    BarkFineConfig,
    BarkSemanticConfig,
    BarkSubModelConfig,
)
from .generation_configuration_bark import (
    BarkCoarseGenerationConfig,
    BarkFineGenerationConfig,
    BarkSemanticGenerationConfig,
)

# 检查是否可用 Flash Attention 2
if is_flash_attn_2_available():
    # 导入 Flash Attention 相关函数和模块
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "suno/bark-small"
_CONFIG_FOR_DOC = "BarkConfig"

# 预训练模型存档列表
BARK_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "suno/bark-small",
    "suno/bark",
    # 查看所有 Bark 模型 https://huggingface.co/models?filter=bark
]

# 从 transformers.models.llama.modeling_llama._get_unpad_data 复制的函数
def _get_unpad_data(attention_mask):
    # 计算每个序列的长度
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 找到非零元素的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 获取批次中最大的序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算累积序列长度
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

# 定义 BarkSelfAttention 类
class BarkSelfAttention(nn.Module):
    # 从 GPTNeoSelfAttention 和 Bark 代码调整而来
    # BarkSelfAttention 可以有两种注意力类型，即全局注意力或因果注意力
    # 初始化方法，接受配置参数和是否使用因果注意力的标志
    def __init__(self, config, is_causal=False):
        # 调用父类的初始化方法
        super().__init__()

        # 正则化
        # 设置丢弃率
        self.dropout = config.dropout
        # 创建用于注意力丢弃的Dropout层
        self.attn_dropout = nn.Dropout(config.dropout)
        # 创建用于残差连接的Dropout层
        self.resid_dropout = nn.Dropout(config.dropout)

        # 嵌入维度
        self.embed_dim = config.hidden_size
        # 注意力头的数量
        self.num_heads = config.num_heads
        # 每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_heads

        # 检查是否可以整除
        if config.hidden_size % config.num_heads != 0:
            # 若不能整除，抛出数值错误
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # 用于所有注意力头的键、查询和值的投影
        self.att_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.bias)
        # 输出投影
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)

        # 是否使用因果注意力
        self.is_causal = is_causal
        if is_causal:
            # 如果是因果注意力，创建一个因果偏置张量
            block_size = config.block_size
            bias = torch.tril(torch.ones((block_size, block_size), dtype=bool)).view(1, 1, block_size, block_size)
            self.register_buffer("bias", bias)

    # 从transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention._split_heads复制的方法
    # 将隐藏维度拆分为注意力头大小和头数
    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        # 新的张量形状
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        # 重塑张量形状
        tensor = tensor.view(new_shape)
        # 转置张量以适应多头注意力的形状(batch, head, seq_length, head_features)
        return tensor.permute(0, 2, 1, 3)

    # 合并注意力头和头数的方法
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """

        # 重新组装所有头部的输出并排在一起
        # (batch, num_heads, seq_len, attn_head_size) -> (batch, seq_len, num_heads*attn_head_size)
        tensor = tensor.transpose(1, 2).contiguous()
        # 重塑张量形状以适应合并的形状
        tensor = tensor.view(tensor.size()[:-2] + (num_heads * attn_head_size,))

        return tensor
    # 定义注意力计算函数，接受查询、键、值、注意力掩码和头部掩码作为输入
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # 计算注意力权重，与查询和键的转置相乘，同时除以查询和键的维度的平方根
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * (1.0 / math.sqrt(self.head_dim))

        # 如果是因果注意力，则需要进行特殊处理
        if self.is_causal:
            query_length, key_length = query.size(-2), key.size(-2)

            # 将注意力权重的左上部分填充为负无穷
            attn_weights = attn_weights.masked_fill(
                self.bias[:, :, key_length - query_length : key_length, :key_length] == 0,
                torch.finfo(attn_weights.dtype).min,
            )

        # 如果有注意力掩码，则应用它
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 对注意力权重进行 softmax 归一化
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        # 对注意力权重进行 dropout
        attn_weights = self.attn_dropout(attn_weights)

        # 如果需要，对头部进行掩码
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 计算注意力输出，将注意力权重与值相乘
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    # 定义模型前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        # 计算所有头部的查询、键和值，并将头部维度移至批次维度
        query, key, value = self.att_proj(hidden_states).split(self.embed_dim, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # 如果过去的键值不为空，则将当前键值与过去的键值连接起来
        if past_key_values is not None:
            past_key = past_key_values[0]
            past_value = past_key_values[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        # 如果需要缓存，则存储当前键值
        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # 使用注意力计算函数计算注意力输出和注意力权重
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 合并头部，并将输出投影到指定维度
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        # 对输出进行残差 dropout
        attn_output = self.resid_dropout(attn_output)

        # 组装模型输出
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
class BarkSelfFlashAttention2(BarkSelfAttention):
    """
    Bark flash attention module. This module inherits from `BarkSelfAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        # 调用父类的构造函数
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        # 设置属性以处理 Flash Attention 版本差异
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        # 将 hidden_size 维度拆分成 attn_head_size 和 num_heads
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        # Flash attention 要求输入具有形状 batch_size x seq_length x head_dim x hidden_dim - (batch, seq_length, head, head_features)
        return tensor

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        # 重新组装所有头部的输出并排在一起
        # (batch, seq_len, num_heads, attn_head_size) -> (batch, seq_len, num_heads*attn_head_size)
        tensor = tensor.view(tensor.size()[:-2] + (num_heads * attn_head_size,))
        return tensor

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        # 获取隐藏状态的批量大小、查询长度和特征维度
        batch_size, query_len, _ = hidden_states.size()

        # 使用注意力投影层将隐藏状态投影为查询、键和值，并在特征维度上分割成多个头
        query, key, value = self.att_proj(hidden_states).split(self.embed_dim, dim=2)

        # 将查询、键和值分割成多个头
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # 如果已有过去的键值对，则将它们与当前的键和值连接起来
        if past_key_values is not None:
            # 将过去的键和值转置，并在序列长度维度上连接当前的键和值
            past_key = past_key_values[0].transpose(1, 2)
            past_value = past_key_values[1].transpose(1, 2)
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)

        # 如果使用缓存，则生成当前的键和值对
        if use_cache is True:
            # 将键和值转置，使得头维度和序列长度维度交换
            present = (key.transpose(1, 2), value.transpose(1, 2))
        else:
            present = None

        # 进行闪电注意力前向传播
        attn_output = self._flash_attention_forward(query, key, value, attention_mask, query_len, dropout=self.dropout)

        # 合并多个头的输出
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        # 输出投影
        attn_output = self.out_proj(attn_output)
        # 应用残差连接的 dropout
        attn_output = self.resid_dropout(attn_output)

        # 输出包括注意力输出和可选的当前键值对
        outputs = (attn_output, present)
        if output_attentions:
            # 如果需要输出注意力权重，则将其初始化为空
            attn_weights = None
            outputs += (attn_weights,)

        return outputs

    # 从transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward中复制
    # 闪电注意力前向传播
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        调用 Flash Attention 的前向方法 - 如果输入的隐藏状态包含至少一个填充标记，则首先取消填充输入，然后计算注意力分数并填充最终的注意力分数。

        Args:
            query_states (`torch.Tensor`):
                要传递给 Flash Attention API 的输入查询状态
            key_states (`torch.Tensor`):
                要传递给 Flash Attention API 的输入键状态
            value_states (`torch.Tensor`):
                要传递给 Flash Attention API 的输入值状态
            attention_mask (`torch.Tensor`):
                填充掩码 - 对应于大小为 `(batch_size, seq_len)` 的张量，其中 0 表示填充标记的位置，1 表示非填充标记的位置。
            dropout (`int`, *optional*):
                注意力丢弃率
            softmax_scale (`float`, *optional*):
                在应用 softmax 之前的 QK^T 的缩放。默认为 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: 一旦 Flash Attention for RoCm 升级到 2.1，删除 `query_length != 1` 检查。有关详细信息，请参阅 LlamaFlashAttention2 __init__ 中的注释。
            causal = self.is_causal and query_length != 1

        # 序列中至少包含一个填充标记
        if attention_mask is not None:
            batch_size = query_states.shape[0]
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
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    # 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input 复制而来
    # 从输入中获取非填充数据的索引、当前序列长度和批次中的最大序列长度
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    # 获取 key_layer 的形状信息
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    # 根据非填充数据的索引重新排列 key_layer 和 value_layer，以去除填充
    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    
    # 处理查询序列的情况
    if query_length == kv_seq_len:
        # 如果查询长度等于键值对序列长度，则直接重新排列查询层
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        # 如果查询长度为1，则直接处理
        max_seqlen_in_batch_q = 1
        # 创建一个序列长度数组，长度为 batch_size + 1，设备为 query_layer 的设备
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        # 截取索引，去除最后一个索引
        indices_q = cu_seqlens_q[:-1]
        # 压缩查询层的第一个维度
        query_layer = query_layer.squeeze(1)
    else:
        # 处理一般情况，即查询长度不等于键值对序列长度的情况
        # 用 -query_length: 切片假设左填充
        attention_mask = attention_mask[:, -query_length:]
        # 处理查询序列的非填充输入
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

    # 返回处理后的结果
    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )
# 定义一个字典，将字符串映射到对应的自定义注意力类
BARK_ATTENTION_CLASSES = {
    "eager": BarkSelfAttention,
    "flash_attention_2": BarkSelfFlashAttention2,
}

# 定义一个自定义的 LayerNorm 类，支持可选的偏置项
class BarkLayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False."""

    def __init__(self, hidden_size, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps=1e-5)

# 定义一个自定义的 MLP 类
class BarkMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_proj = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=config.bias)
        self.out_proj = nn.Linear(4 * config.hidden_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.in_proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

# 定义一个自定义的 Block 类
class BarkBlock(nn.Module):
    def __init__(self, config, is_causal=False):
        super().__init__()

        if is_causal:
            # 如果是因果的，使用手动实现的 LayerNorm，以支持可选的偏置项
            # 这个手动实现的 LayerNorm 用于与 Bark 的选择保持一致，即在自回归模型中保留可选的偏置项
            self.layernorm_1 = BarkLayerNorm(config.hidden_size, bias=config.bias)
            self.layernorm_2 = BarkLayerNorm(config.hidden_size, bias=config.bias)
        else:
            self.layernorm_1 = nn.LayerNorm(config.hidden_size)
            self.layernorm_2 = nn.LayerNorm(config.hidden_size)

        # 根据配置选择不同的注意力实现类
        self.attn = BARK_ATTENTION_CLASSES[config._attn_implementation](config, is_causal=is_causal)

        self.mlp = BarkMLP(config)

    def forward(
        self,
        hidden_states,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        intermediary_hidden_states = self.layernorm_1(hidden_states)
        # 对隐藏状态进行 LayerNormalization 处理，得到中间隐藏状态

        attn_outputs = self.attn(
            intermediary_hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 利用注意力机制处理中间隐藏状态，得到注意力输出

        attn_output = attn_outputs[0]  # output_attn: output, present_key_values, (attn_weights)
        # 提取注意力输出中的第一个元素，即注意力值

        outputs = attn_outputs[1:]
        # 提取注意力输出中除了第一个元素外的所有元素，即除了注意力值外的其它输出信息

        intermediary_hidden_states = hidden_states + attn_output
        # 将原始隐藏状态与注意力输出相加，得到新的中间隐藏状态

        intermediary_hidden_states = intermediary_hidden_states + self.mlp(
            self.layernorm_2(intermediary_hidden_states)
        )
        # 对新的中间隐藏状态进行 LayerNormalization 和 MLP 处理，得到最终的中间隐藏状态

        if use_cache:
            outputs = (intermediary_hidden_states,) + outputs
            # 如果使用缓存，将最终的中间隐藏状态加入输出
        else:
            outputs = (intermediary_hidden_states,) + outputs[1:]
            # 如果不使用缓存，将最终的中间隐藏状态加入输出，并移除原始的隐藏状态

        return outputs  # hidden_states, ((present), attentions)
        # 返回最终输出，包括隐藏状态和可能的 present 状态以及注意力信息
class BarkPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为BarkConfig
    config_class = BarkConfig
    # 不支持梯度检查点
    supports_gradient_checkpointing = False
    # 支持flash注意力机制的第二版本
    _supports_flash_attn_2 = True

    # 初始化权重
    def _init_weights(self, module):
        """Initialize the weights."""
        # 如果是线性层
        if isinstance(module, (nn.Linear,)):
            # 使用正态分布初始化权重，均值为0，标准差为配置文件中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置，则将偏置初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为配置文件中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，则将填充索引处的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是LayerNorm层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置初始化为0
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)

    # 构造函数
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    # 获取设备信息
    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """

        # 如果没有'_hf_hook'属性，说明未被转移到其他设备上，直接获取参数所在设备
        if not hasattr(self, "_hf_hook"):
            return get_parameter_device(self)
        # 遍历所有模块
        for module in self.modules():
            # 如果模块具有'_hf_hook'属性，并且'_hf_hook'属性具有'execution_device'属性
            # 则返回模块执行的设备
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)

        # 如果没有找到具有'_hf_hook'属性的模块，则返回参数所在设备
        return get_parameter_device(self)


BARK_MODEL_START_DOCSTRING = """
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`{config}`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


BARK_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
``` 
    # 该代码段是对模型库中所有模型的实现进行说明，包括下载或保存模型、调整输入嵌入的大小、修剪模型头等等。

    # 该模型也是 PyTorch 的 torch.nn.Module 的子类。可以像常规的 PyTorch 模块一样使用它，并参考 PyTorch 文档了解与常规使用和行为相关的所有事项。

    # 参数:
    #     config ([`BarkConfig`]):
    #         包含模型所有参数的模型配置类。使用配置文件初始化不会加载模型关联的权重，只会加载配置。请查看
    #         [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
# 定义了一个原因型的Bark模型，继承自BarkPreTrainedModel
class BarkCausalModel(BarkPreTrainedModel):
    # 配置类为BarkSubModelConfig
    config_class = BarkSubModelConfig
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        # 将配置对象保存到当前实例中
        self.config = config

        # 初始化输入词嵌入层，将输入词汇映射到隐藏层大小的向量空间
        self.input_embeds_layer = nn.Embedding(config.input_vocab_size, config.hidden_size)
        # 初始化位置嵌入层，用于表示输入的位置信息
        self.position_embeds_layer = nn.Embedding(config.block_size, config.hidden_size)

        # 初始化 Dropout 层，用于在训练过程中进行随机失活
        self.drop = nn.Dropout(config.dropout)

        # 初始化多层 Transformer Block，每个 Block 包含一个 BarkBlock 实例，is_causal=True 表示使用自回归机制
        self.layers = nn.ModuleList([BarkBlock(config, is_causal=True) for _ in range(config.num_layers)])
        # 检查是否使用了第二个版本的 Flash Attention 实现
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # 初始化最后的层归一化层，用于在模型的输出之前进行归一化处理
        self.layernorm_final = BarkLayerNorm(config.hidden_size, bias=config.bias)

        # 初始化语言模型头，用于将隐藏状态映射到输出词汇空间
        self.lm_head = nn.Linear(config.hidden_size, config.output_vocab_size, bias=False)
        # 设置梯度检查点为 False，表示不使用梯度检查点
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入词嵌入层
    def get_input_embeddings(self):
        return self.input_embeds_layer

    # 设置输入词嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.input_embeds_layer = new_embeddings
    # 准备生成的输入
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # 获取输入的嵌入向量，如果没有则为 None
        input_embeds = kwargs.get("input_embeds", None)

        # 获取注意力掩码，如果没有则为 None
        attention_mask = kwargs.get("attention_mask", None)
        # 获取位置编码，如果没有则为 None
        position_ids = kwargs.get("position_ids", None)

        # 如果过去的键值不为空
        if past_key_values is not None:
            # 忽略已被过去键值覆盖的标记
            seq_len = input_ids.shape[1]
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认旧行为：只保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

            # input_embeds 已经被使用，不再需要
            input_embeds = None
        else:
            # 如果 input_embeds 不为空且使用缓存
            if input_embeds is not None and kwargs.get("use_cache"):
                seq_len = input_embeds.shape[1]
            else:
                seq_len = input_ids.shape[1]

        # 确保 attention_mask 和 position_ids 的形状与在第一次前向传递时减少序列长度的奇怪的 Bark 修复方法对齐
        if attention_mask is not None:
            attention_mask = attention_mask[:, :seq_len]
        if position_ids is not None:
            position_ids = position_ids[:, :seq_len]

        # 如果 attention_mask 不为空且 position_ids 为空
        if attention_mask is not None and position_ids is None:
            # 为批量生成动态创建 position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # 如果 input_embeds 不为空且使用缓存
        if input_embeds is not None and kwargs.get("use_cache"):
            return {
                "input_ids": None,
                "input_embeds": input_embeds,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            }
        # 返回准备好的输入
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    # 添加模型前向传递的开始文档字符串
    @add_start_docstrings_to_model_forward(BARK_CAUSAL_MODEL_INPUTS_DOCSTRING)
    # 定义一个方法用于模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的 token ID
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,  # 用于存储过去的 key 和 value
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        position_ids: Optional[torch.Tensor] = None,  # 位置 ID
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码
        labels: Optional[torch.LongTensor] = None,  # 标签
        input_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入向量
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果
    @staticmethod
    # 重新排序缓存，用于 beam search
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        # 必要的用于 beam_search
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
# 添加文档字符串，描述BarkSemanticModel的语义（或文本）模型，与粗略模型共享相同的架构，是一个类似于GPT-2的自回归模型，顶部有一个语言建模头
@add_start_docstrings(
    """Bark semantic (or text) model. It shares the same architecture as the coarse model.
    It is a GPT-2 like autoregressive model with a language modeling head on top.""",
    BARK_MODEL_START_DOCSTRING.format(config="BarkSemanticConfig"),
)
class BarkSemanticModel(BarkCausalModel):
    # 设置基础模型前缀为"semantic"
    base_model_prefix = "semantic"
    # 设置配置类为BarkSemanticConfig
    config_class = BarkSemanticConfig

    # 定义生成方法
    def generate(
        self,
        input_ids: torch.Tensor,
        semantic_generation_config: BarkSemanticGenerationConfig = None,
        history_prompt: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
# 添加文档字符串，描述BarkCoarseModel的粗略声学模型，与语义（或文本）模型共享相同的架构，是一个类似于GPT-2的自回归模型，顶部有一个语言建模头
@add_start_docstrings(
    """Bark coarse acoustics model.
    It shares the same architecture as the semantic (or text) model. It is a GPT-2 like autoregressive model with a
    language modeling head on top.""",
    BARK_MODEL_START_DOCSTRING.format(config="BarkCoarseConfig"),
)
class BarkCoarseModel(BarkCausalModel):
    # 设置基础模型前缀为"coarse_acoustics"
    base_model_prefix = "coarse_acoustics"
    # 设置配置类为BarkCoarseConfig
    config_class = BarkCoarseConfig

    # 定义预处理历史方法
    def preprocess_histories(
        self,
        max_coarse_history: int,
        semantic_to_coarse_ratio: int,
        batch_size: int,
        semantic_generation_config: int,
        codebook_size: int,
        history_prompt: Optional[Dict[str, torch.Tensor]] = None,
    # 定义生成方法
    def generate(
        self,
        semantic_output: torch.Tensor,
        semantic_generation_config: BarkSemanticGenerationConfig = None,
        coarse_generation_config: BarkCoarseGenerationConfig = None,
        codebook_size: int = 1024,
        history_prompt: Optional[Dict[str, torch.Tensor]] = None,
        return_output_lengths: Optional[bool] = None,
        **kwargs,
# 添加文档字符串，描述BarkFineModel的细粒度声学模型，是一个非因果GPT-like模型，具有`config.n_codes_total`嵌入层和语言建模头，每个码书一个
@add_start_docstrings(
    """Bark fine acoustics model. It is a non-causal GPT-like model with `config.n_codes_total` embedding layers and
    language modeling heads, one for each codebook.""",
    BARK_MODEL_START_DOCSTRING.format(config="BarkFineConfig"),
)
class BarkFineModel(BarkPreTrainedModel):
    # 设置基础模型前缀为"fine_acoustics"
    base_model_prefix = "fine_acoustics"
    # 设置配置类为BarkFineConfig
    config_class = BarkFineConfig
    # 设置主输入名称为"codebook_idx"
    main_input_name = "codebook_idx"
    # 初始化模型，包括非因果 GPT 类型模型，每个 Encodec 代码本都有一个嵌入层和一个 lm_head
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 保存配置信息
        self.config = config

        # 初始化修改后的非因果 GPT 类型模型
        # 每个 Encodec 代码本都有一个嵌入层和一个 lm_head
        self.input_embeds_layers = nn.ModuleList(
            [nn.Embedding(config.input_vocab_size, config.hidden_size) for _ in range(config.n_codes_total)]
        )
        self.position_embeds_layer = nn.Embedding(config.block_size, config.hidden_size)

        self.drop = nn.Dropout(config.dropout)

        # 创建多个 BarkBlock 层，用于构建模型的层
        self.layers = nn.ModuleList([BarkBlock(config, is_causal=False) for _ in range(config.num_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.layernorm_final = nn.LayerNorm(config.hidden_size)

        # 创建多个 lm_head 层，用于模型输出
        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.output_vocab_size, bias=False)
                for _ in range(config.n_codes_given, config.n_codes_total)
            ]
        )
        self.gradient_checkpointing = False
        self.n_codes_total = config.n_codes_total

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回每个代码本的嵌入层
        return self.input_embeds_layers

    def set_input_embeddings(self, new_embeddings):
        # 设置每个代码本的嵌入层
        self.input_embeds_layers = new_embeddings

    def get_output_embeddings(self):
        # 返回每个代码本的 lm_head
        return self.lm_heads

    def set_output_embeddings(self, new_output_embeddings):
        # 设置每个代码本的 lm_head
        self.lm_heads = new_output_embeddings

    def _resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None):
        old_embeddings_list = self.get_input_embeddings()
        new_embeddings_list = nn.ModuleList(
            [
                self._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of)
                for old_embeddings in old_embeddings_list
            ]
        )
        self.set_input_embeddings(new_embeddings_list)
        new_num_tokens = new_embeddings_list[0].weight.shape[0]

        # 如果单词嵌入没有绑定，确保 lm_head 也被调整大小
        if self.get_output_embeddings() is not None and not self.config.tie_word_embeddings:
            old_lm_head_list = self.get_output_embeddings()
            new_lm_head_list = nn.ModuleList(
                [self._get_resized_lm_head(old_lm_head, new_num_tokens) for old_lm_head in old_lm_head_list]
            )
            self.set_output_embeddings(new_lm_head_list)

        return self.get_input_embeddings()
    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        """
        调整模型的输入token嵌入矩阵的大小，如果`new_num_tokens != config.vocab_size`。

        在模型类具有`tie_weights()`方法的情况下，之后会处理绑定权重嵌入。

        参数:
            new_num_tokens (`int`, *optional*):
                嵌入矩阵中的新token数量。增加大小将在末尾添加新初始化的向量。减小大小将从末尾移除向量。
                如果未提供或为 `None`，则只返回模型的输入tokens `torch.nn.Embedding` 模块的指针，不执行任何操作。
            pad_to_multiple_of (`int`, *optional*):
                如果设置，将嵌入矩阵填充为提供的值的倍数。

                这在启用Tensor Core（在NVIDIA硬件上，计算能力 `>= 7.5`（Volta））或者在需要序列长度为128的倍数时非常有用（例如TPU）。
                有关此内容的更多详细信息，或者关于选择正确的调整大小值的帮助，请参阅此指南：
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        返回:
            `torch.nn.Embedding`: 模型的输入tokens Embeddings模块的指针。
        """
        # 调整token嵌入
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 如果未提供新token数量和填充到倍数，则直接返回调整后的嵌入
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # 更新基本模型和当前模型配置
        self.config.output_vocab_size = model_embeds[0].weight.shape[0]
        self.config.vocab_size = model_embeds[0].weight.shape[0]
        self.output_vocab_size = model_embeds[0].weight.shape[0]
        self.vocab_size = model_embeds[0].weight.shape[0]

        # 如果需要，重新绑定权重
        self.tie_weights()

        return model_embeds
    # 将输入嵌入列表和输出嵌入列表之间的权重相互绑定或克隆

    def tie_weights(self):
        """
        Tie the weights between the input embeddings list and the output embeddings list.

        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
        weights instead.
        """
        # 如果配置中设置了 `tie_word_embeddings` 标志为真，则执行权重绑定或克隆操作
        if getattr(self.config, "tie_word_embeddings", True):
            # 初始化存储绑定权重键的列表
            self._tied_weights_keys = []
            # 获取输出嵌入列表
            output_embeddings = self.get_output_embeddings()
            # 获取输入嵌入列表
            input_embeddings = self.get_input_embeddings()

            # 遍历从输入到输出的所有嵌入层，将它们的权重相互绑定或克隆
            for i in range(self.config.n_codes_total - self.config.n_codes_given):
                # 将当前输出嵌入层的权重与下一个输入嵌入层的权重相绑定或克隆
                self._tie_or_clone_weights(output_embeddings[i], input_embeddings[i + 1])
                # 将绑定权重键添加到列表中，用于记录
                self._tied_weights_keys.append(f"lm_heads.{i}.weight")

        # 遍历模型中的每个模块，执行可能存在的进一步权重绑定操作
        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()

    @add_start_docstrings_to_model_forward(BARK_FINE_INPUTS_DOCSTRING)
    # 定义模型的前向传播方法，接受一系列输入参数，返回模型的输出结果
    def forward(
        self,
        codebook_idx: int,  # an additionnal idx corresponding to the id of the codebook that will be predicted
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义模型的生成方法，接受粗略输出、语义生成配置、粗略生成配置、细化生成配置、码书大小、历史提示等参数
    def generate(
        self,
        coarse_output: torch.Tensor,
        semantic_generation_config: BarkSemanticGenerationConfig = None,
        coarse_generation_config: BarkCoarseGenerationConfig = None,
        fine_generation_config: BarkFineGenerationConfig = None,
        codebook_size: int = 1024,
        history_prompt: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
# 引入自定义的文档字符串装饰器，并为 BarkModel 添加详细的文档字符串说明其组成部分
@add_start_docstrings(
    """
    The full Bark model, a text-to-speech model composed of 4 sub-models:
    - [`BarkSemanticModel`] (also referred to as the 'text' model): a causal auto-regressive transformer model that
      takes
    as input tokenized text, and predicts semantic text tokens that capture the meaning of the text.
    - [`BarkCoarseModel`] (also refered to as the 'coarse acoustics' model), also a causal autoregressive transformer,
    that takes into input the results of the last model. It aims at regressing the first two audio codebooks necessary
    to `encodec`.
    - [`BarkFineModel`] (the 'fine acoustics' model), this time a non-causal autoencoder transformer, which iteratively
    predicts the last codebooks based on the sum of the previous codebooks embeddings.
    - having predicted all the codebook channels from the [`EncodecModel`], Bark uses it to decode the output audio
      array.

    It should be noted that each of the first three modules can support conditional speaker embeddings to condition the
    output sound according to specific predefined voice.
    """,
    BARK_START_DOCSTRING,  # 添加自定义文档字符串的起始标记
)
# 定义 BarkModel 类，继承自 BarkPreTrainedModel
class BarkModel(BarkPreTrainedModel):
    # 使用 BarkConfig 作为配置类
    config_class = BarkConfig

    # 初始化方法，接收一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化语义模型（BarkSemanticModel），传入配置的语义配置
        self.semantic = BarkSemanticModel(config.semantic_config)
        # 初始化粗声学模型（BarkCoarseModel），传入配置的粗声学配置
        self.coarse_acoustics = BarkCoarseModel(config.coarse_acoustics_config)
        # 初始化细声学模型（BarkFineModel），传入配置的细声学配置
        self.fine_acoustics = BarkFineModel(config.fine_acoustics_config)

        # 初始化编解码模型（AutoModel），传入编解码配置
        self.codec_model = AutoModel.from_config(config.codec_config)

        # 设置当前类的配置属性
        self.config = config

    # 定义 device 属性，返回模型所在的设备
    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        # 对于 bark_model，设备必须在其子模型中进行验证
        # 如果有 _hf_hook，已经被卸载，因此必须在钩子中找到设备
        if not hasattr(self.semantic, "_hf_hook"):
            # 如果语义模型没有 _hf_hook 属性，则返回模型参数所在的设备
            return get_parameter_device(self)
        # 遍历语义模型的所有模块
        for module in self.semantic.modules():
            # 如果当前模块具有 _hf_hook 属性，并且具有执行设备，并且执行设备不为 None
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                # 返回当前模块的执行设备
                return torch.device(module._hf_hook.execution_device)
    def enable_cpu_offload(self, gpu_id: Optional[int] = 0):
        r"""
        Offloads all sub-models to CPU using accelerate, reducing memory usage with a low impact on performance. This
        method moves one whole sub-model at a time to the GPU when it is used, and the sub-model remains in GPU until
        the next sub-model runs.

        Args:
            gpu_id (`int`, *optional*, defaults to 0):
                GPU id on which the sub-models will be loaded and offloaded.
        """
        # 检查是否安装了 accelerate 库
        if is_accelerate_available():
            from accelerate import cpu_offload_with_hook
        else:
            # 如果没有安装 accelerate 库，则抛出 ImportError
            raise ImportError("`enable_model_cpu_offload` requires `accelerate`.")

        # 根据给定的 GPU ID 创建设备对象
        device = torch.device(f"cuda:{gpu_id}")

        # 如果当前模型不在 CPU 上，则将其移动到 CPU
        if self.device.type != "cpu":
            self.to("cpu")
            torch.cuda.empty_cache()  # 释放 GPU 缓存，以便看到内存节省（尽管可能存在）

        # 将 semantic 模型的 input_embeds_layer 移动到 CPU
        self.semantic.input_embeds_layer, _ = cpu_offload_with_hook(self.semantic.input_embeds_layer, device)

        hook = None
        # 遍历需要移动到 CPU 的模型列表，并使用 cpu_offload_with_hook 方法进行移动
        for cpu_offloaded_model in [
            self.semantic,
            self.coarse_acoustics,
            self.fine_acoustics,
        ]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        self.fine_acoustics_hook = hook

        # 将 codec_model 模型移动到 CPU
        _, hook = cpu_offload_with_hook(self.codec_model, device, prev_module_hook=hook)

        # 手动移动最后一个模型到 CPU
        self.codec_model_hook = hook

    def codec_decode(self, fine_output, output_lengths=None):
        """Turn quantized audio codes into audio array using encodec."""

        # 转置 fine_output 张量的维度
        fine_output = fine_output.transpose(0, 1)
        # 使用 codec_model 模型的 quantizer 解码 fine_output
        emb = self.codec_model.quantizer.decode(fine_output)

        if output_lengths is not None:
            # 对于每个样本，根据 output_lengths 截取 emb 张量，并解码得到音频数组
            out = [sample[:, :l].unsqueeze(0) for (sample, l) in zip(emb, output_lengths)]
            audio_arr = [self.codec_model.decoder(sample).squeeze() for sample in out]
        else:
            # 解码整个 emb 张量，并得到音频数组
            out = self.codec_model.decoder(emb)
            audio_arr = out.squeeze(1)  # 压缩 codebook 维度

        return audio_arr

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        history_prompt: Optional[Dict[str, torch.Tensor]] = None,
        return_output_lengths: Optional[bool] = None,
        **kwargs,
    @classmethod
    def _check_and_enable_flash_attn_2(
        cls,
        config,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]] = None,
        hard_check_only: bool = False,
    ):
        """
        `_check_and_enable_flash_attn_2` originally don't expand flash attention enabling to the model
        sub-configurations. We override the original method to make sure that Bark sub-models are using Flash Attention
        if necessary.

        If you don't know about Flash Attention, check out the official repository of flash attention:
        https://github.com/Dao-AILab/flash-attention

        For using Flash Attention 1.0 you can do it directly via the `BetterTransformer` API, have a look at this
        specific section of the documentation to learn more about it:
        https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#decoder-models

        The method checks if the current setup is compatible with Flash Attention as it requires the model to be in
        half precision and not ran on CPU.

        If all checks pass and `hard_check_only` is False, the method will set the config attribute `_attn_implementation` to "flash_attention_2" so that the model
        can initialize the correct attention module
        """
        # 调用父类方法来检查并激活 Flash Attention 2
        config = super()._check_and_enable_flash_attn_2(
            config, torch_dtype, device_map, hard_check_only=hard_check_only
        )

        # 设置语义模型配置的注意力实现
        config.semantic_config._attn_implementation = config._attn_implementation
        # 设置粗声学模型配置的注意力实现
        config.coarse_acoustics_config._attn_implementation = config._attn_implementation
        # 设置细声学模型配置的注意力实现
        config.fine_acoustics_config._attn_implementation = config._attn_implementation
        # 返回配置
        return config
```