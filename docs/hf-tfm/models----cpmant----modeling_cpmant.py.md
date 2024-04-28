# `.\models\cpmant\modeling_cpmant.py`

```py
# 设置文件编码为utf-8
# 版权声明
# 根据Apache 2.0许可证授权使用该文件
# 仅在符合许可证的情况下使用此文件
# 可以获取许可协议的拷贝
# 详见：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则以"AS IS"基础发布的软件
# 没有任何明示或暗示的保证或条件
# 查看许可证以了解特定语言控制权限和限制
""" PyTorch CPMAnt"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_cpmant import CpmAntConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "openbmb/cpm-ant-10b"
_CONFIG_FOR_DOC = "CpmAntConfig"

# CPMAnt预训练模型归档列表
CPMANT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openbmb/cpm-ant-10b",
    # 查看所有CPMAnt模型：https://huggingface.co/models?filter=cpmant
]


class CpmAntLayerNorm(nn.Module):
    """
    使用Root Mean Square (RMS)层归一化，请参阅 https://arxiv.org/abs/1910.07467 获取更多细节。
    """

    def __init__(self, config: CpmAntConfig):
        super().__init__()

        self.eps = config.eps
        self.dim_norm = config.hidden_size
        self.weight = nn.Parameter(torch.empty(config.hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (`torch.Tensor`的形状为`(batch, seq_len, dim_in)`)
        """
        # 断言：如果hidden_states的最后一个维度不等于dim_norm，则引发AssertionError异常
        if hidden_states.size(-1) != self.dim_norm:
            raise AssertionError("hidden_states.size(-1) != self.dim_norm")
        old_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        hidden_states = (hidden_states * torch.rsqrt(variance + self.eps)).to(old_dtype) * self.weight
        return hidden_states


class CpmAntAttention(nn.Module):
    # 初始化函数，接受一个配置参数对象
    def __init__(self, config: CpmAntConfig):
        # 调用父类构造函数
        super().__init__()
        # 从配置对象中获取隐藏层尺寸作为维度模型
        self.dim_model = config.hidden_size
        # 从配置对象中获取注意力头的数量
        self.num_heads = config.num_attention_heads
        # 从配置对象中获取注意力头的维度
        self.dim_head = config.dim_head

        # 线性层将输入维度映射到头的数量乘以头的维度，无偏置
        self.project_q = nn.Linear(self.dim_model, self.num_heads * self.dim_head, bias=False)
        self.project_k = nn.Linear(self.dim_model, self.num_heads * self.dim_head, bias=False)
        self.project_v = nn.Linear(self.dim_model, self.num_heads * self.dim_head, bias=False)

        # 线性层将头的数量乘以头的维度映射回原始维度，无偏置
        self.attention_out = nn.Linear(self.num_heads * self.dim_head, self.dim_model, bias=False)

        # Softmax 函数，对最后一个维度进行操作
        self.softmax = torch.nn.Softmax(dim=-1)

        # 如果配置中有定义丢弃率，则使用丢弃层
        if config.dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=config.dropout_p)
        else:
            self.dropout = None

    # 前向传播函数
    def forward(
        self,
        hidden_q: torch.Tensor,  # 查询张量
        hidden_kv: torch.Tensor,  # 键值张量
        attention_mask: torch.BoolTensor,  # 注意力掩码
        position_bias: torch.Tensor,  # 位置偏置
        output_attentions: Optional[bool] = False,  # 是否返回注意力值，默认为 False
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # 先前的键值对，可选
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选
class CpmAntSelfAttentionBlock(nn.Module):
    # 初始化自注意力模块
    def __init__(self, config: CpmAntConfig):
        super().__init__()
        # 前层归一化
        self.layernorm_before_attention = CpmAntLayerNorm(config)
        # 自注意力机制
        self.self_attention = CpmAntAttention(config)
        # 如果有dropout，则使用torch的Dropout
        if config.dropout_p:
            self.dropout = torch.nn.Dropout(config.dropout_p)
        else:
            self.dropout = None

    # 自注意力模块的前向传播
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, len_seq, dim_model)`):
                Input of transformer block(self-attention block). It can be the raw embedding of a batch of sequences.
            attention_mask (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Avoid invalid areas to participate in the calculation of self-attention.
            position_bias (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                Provide positional information to self-attention block.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*):
                Cached past key and value projection states.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        outputs = self.layernorm_before_attention(hidden_states)
        outputs = self.self_attention(
            outputs, outputs, attention_mask, position_bias, output_attentions, past_key_values, use_cache
        )

        outputs, attn_weights, current_key_value = outputs

        if self.dropout is not None:
            outputs = self.dropout(outputs)
        hidden_states = hidden_states + outputs

        return hidden_states, attn_weights, current_key_value


class CpmAntDenseGatedACT(nn.Module):
    # 初始化稠密门控激活函数模块
    def __init__(self, config: CpmAntConfig):
        super().__init__()
        # 使用线性变换
        self.w_0 = nn.Linear(config.hidden_size, config.dim_ff, bias=False)
        self.w_1 = nn.Linear(config.hidden_size, config.dim_ff, bias=False)
        # 使用GELU激活函数
        self.act = torch.nn.GELU()

    # 稠密门控激活函数的前向传播
    def forward(self, hidden_states: torch.Tensor):
        """Transform an input tensor from one feature space to another via a nonlinear operation
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, seq_len, dim_in)`)
        """
        gate_score = self.act(self.w_0(hidden_states))
        hidden_states = self.w_1(hidden_states)
        hidden_states = gate_score * hidden_states
        return hidden_states
class CpmAntFeedForward(nn.Module):
    # 定义一个带有前馈网络的模块，参数为CpmAntConfig对象
    def __init__(self, config: CpmAntConfig):
        super().__init__()
        # 初始化输入权重
        self.w_in = CpmAntDenseGatedACT(config)
        # 如果有设置dropout概率，则创建Dropout层，否则为None
        if config.dropout_p is not None:
            self.dropout = torch.nn.Dropout(config.dropout_p)
        else:
            self.dropout = None

        # 初始化输出层线性变换
        self.w_out = nn.Linear(config.dim_ff, config.hidden_size, bias=False)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, seq_len, dim_in)`)
        """
        # 通过输入权重进行转换
        hidden_states = self.w_in(hidden_states)

        # 如果存在dropout层，则应用dropout
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)

        # 通过输出层进行线性变换
        hidden_states = self.w_out(hidden_states)

        return hidden_states


class CpmAntFFNBlock(nn.Module):
    # 定义一个带有Feed Forward网络块的模块，参数为CpmAntConfig对象
    def __init__(self, config: CpmAntConfig):
        super().__init__()
        # 初始化层归一化
        self.layernorm_before_ffn = CpmAntLayerNorm(config)
        # 初始化前馈网络
        self.ffn = CpmAntFeedForward(config)
        # 如果配置有dropout概率，则创建Dropout层，否则为None
        if config.dropout_p:
            self.dropout = torch.nn.Dropout(config.dropout_p)
        else:
            self.dropout = None

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
    ):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, len_seq, dim_model)`):
                Hidden states before feed forward layer.
        """
        # 通过层归一化对输入进行归一化
        ln_outputs = self.layernorm_before_ffn(hidden_states)
        # 经过前馈网络处理
        outputs = self.ffn(ln_outputs)
        # 如果存在dropout层，则应用dropout
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        # 将处理后的结果与原始输入相加
        hidden_states = hidden_states + outputs
        return hidden_states


class CpmAntTransformerBlock(nn.Module):
    # 定义一个带有Transformer Block的模块，参数为CpmAntConfig对象
    def __init__(self, config: CpmAntConfig):
        super().__init__()
        # 初始化自注意力块
        self.self_att = CpmAntSelfAttentionBlock(config)
        # 初始化Feed Forward网络��
        self.ffn = CpmAntFFNBlock(config)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        """
        Args:
            hidden_states (`torch.Tensor`):
                输入层的形状为 `(batch, seq_len, dim_model)` 的张量
            attention_mask (`torch.Tensor`):
                避免无效区域参与形状为 `(batch, seq_len, seq_len)` 的计算
            position_bias (`torch.Tensor`):
                提供位置信息给注意力机制的形状为 `(num_heads, seq_len, seq_len)` 的张量
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量
            past_key_values (`Tuple[torch.Tensor, torch.Tensor])`, *optional*):
                缓存的过去的键和值投影状态
            use_cache (`bool`, *optional*):
                如果设置为 `True`，则返回 `past_key_values` 键值状态，并可用于加速解码
                (参见 `past_key_values`)
        """
        # 使用 self-attention 层处理输入隐藏状态
        hidden_states = self.self_att(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # 获取处理后的隐藏状态、注意力权重和当前键值
        hidden_states, attn_weights, current_key_value = hidden_states

        # 使用 feed-forward 网络处理隐藏状态
        hidden_states = self.ffn(hidden_states)

        # 返回处理后的隐藏状态、注意力权重和当前键值
        return hidden_states, attn_weights, current_key_value
class CpmAntEncoder(nn.Module):
    def __init__(self, config: CpmAntConfig):
        # 初始化 CpmAntEncoder 类
        super().__init__()
        # 获取隐藏层的数量
        self.num_layers = config.num_hidden_layers
        # 创建由多个 CpmAntTransformerBlock 组成的列表
        self.layers = nn.ModuleList([CpmAntTransformerBlock(config) for ith in range(self.num_layers)])

        # 创建输出层的 LayerNorm 模块
        self.output_layernorm = CpmAntLayerNorm(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_bias: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
    ):
        """
        Args:
            hidden_states (`torch.Tensor`):
                输入层的张量，形状为 `(batch, seq_len, dim_model)`
            attention_mask (`torch.Tensor`):
                避免无效区域参与计算的掩码张量，形状为 `(batch, seq_len, seq_len)`
            position_bias (`torch.Tensor`):
                提供位置信息给注意力机制的张量，形状为 `(num_heads, seq_len, seq_len)`
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量
            output_hidden_states (`bool`, *optional*):
                是否返回所有层的隐藏状态
            past_key_values (`Tuple[torch.Tensor, torch.Tensor])`, *optional*):
                已缓存的过去 key 和 value 投影状态
            use_cache (`bool`, *optional*):
                若为 `True`，则返回 `past_key_values` 的 key 和 value 状态，并可以用于加速解码过程
        """
        # 初始化需要保存的信息列表
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        current_key_values = () if use_cache else None

        # 遍历每个 transformer block
        for i, layer in enumerate(self.layers):
            # 如果需要输出隐藏状态，则记录当前层的隐藏状态
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            # 调用当前层的 forward 方法进行正向传播
            layer_outputs = layer(
                hidden_states,
                attention_mask,
                position_bias,
                output_attentions=output_attentions,
                past_key_values=past_key_values[i] if past_key_values else None,
                use_cache=use_cache,
            )
            # 获取当前层的输出
            hidden_states, attn_weights, current_key_value = layer_outputs
            # 如果需要输出注意力权重，则记录当前层的注意力权重
            if output_attentions:
                all_self_attns += (attn_weights,)
            # 如果当前 key value 不为空，则记录
            if current_key_value is not None:
                current_key_values = current_key_values + (current_key_value,)

        # 对输出进行 LayerNorm 处理
        hidden_states = self.output_layernorm(hidden_states)

        # 如果需要输出隐藏状态，则记录最后一层的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 返回隐藏状态、当前 key value、所有隐藏状态和所有注意力权重
        return hidden_states, current_key_values, all_hidden_states, all_self_attns
# 从transformers.models.bert.modeling_bert.BertIntermediate中复制代码，将Bert改为CPMAnt
class CpmAntIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，将输入维度从config.hidden_size变换为config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果config.hidden_act是字符串类型，则使用对应的激活函数，否则直接使用config.hidden_act中定义的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入hidden_states经过全连接层变换
        hidden_states = self.dense(hidden_states)
        # 经过激活函数处理
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的hidden_states
        return hidden_states


class CpmAntSegmentPositionEmbedding(nn.Module):
    def __init__(self, config: CpmAntConfig):
        super().__init__()
        # 初始化一些参数
        self.num_heads = config.num_attention_heads
        self.num_buckets = config.position_bias_num_buckets
        self.max_distance = config.position_bias_max_distance
        self.num_segments = config.segment_types
        # 定义一个相对注意力偏置，是一个可训练的参数
        self.relative_attention_bias = nn.Parameter(
            torch.empty(
                config.segment_types * config.segment_types + config.position_bias_num_buckets,
                config.num_attention_heads,
            )
        )

    def forward(
        self,
        key_pos: torch.Tensor,
        query_pos: torch.Tensor,
        key_segment: torch.Tensor,
        query_segment: torch.Tensor,
    ):
        # 禁止梯度计算
        with torch.no_grad():
            # 获取批次大小
            batch = key_pos.size(0)
            # 获取键序列长度
            keylen = key_pos.size(1)
            # 获取查询序列长度
            querylen = query_pos.size(1)

            # 检查键序列和查询序列的批次大小是否相等，若不等则引发断言错误
            if key_pos.size(0) != query_pos.size(0):
                raise AssertionError(
                    f"key_pos.size(0) should be equal to query_pos.size(0), but got {key_pos.size(0)} and {query_pos.size(0)}!"
                )
            # 检查键序列长度和键分段长度是否相等，若不等则引发断言错误
            if keylen != key_segment.size(1) or querylen != query_segment.size(1):
                raise AssertionError(
                    f"keylen should be equal to key_segment.size(1), but got {keylen} and {key_segment.size(1)}!"
                )
            # 检查查询序列长度和查询分段长度是否相等，若不等则引发断言错误
            if querylen != query_segment.size(1):
                raise AssertionError(
                    f"querylen should be equal to query_segment.size(1), but got {querylen} and {query_segment.szie(1)}!"
                )

            # 将键位置张量重塑为(batch, -1, keylen)
            key_pos = key_pos.view(batch, -1, keylen)
            # 将查询位置张量重塑为(batch, querylen, -1)
            query_pos = query_pos.view(batch, querylen, -1)
            # 将键分段张量重塑为(batch, -1, keylen)
            key_segment = key_segment.view(batch, -1, keylen)
            # 将查询分段张量重塑为(batch, querylen, -1)
            query_segment = query_segment.view(batch, querylen, -1)

            # 计算相对位置桶
            relative_position_bucket = self._segment_relative_position_bucket(query_segment, key_segment)
            # 增加偏移量
            relative_position_bucket = relative_position_bucket + self.num_buckets

            # 计算绝对位置桶
            absolute_position_bucket = self._position_bucket(
                torch.arange(keylen, dtype=torch.int32, device=relative_position_bucket.device)[None, :]
                - torch.arange(querylen, dtype=torch.int32, device=relative_position_bucket.device)[:, None],
                num_buckets=self.num_buckets,
                max_distance=self.max_distance,
            )
            # 若键分段与查询分段相等，则使用绝对位置桶
            relative_position_bucket = torch.where(
                (key_segment == query_segment),
                absolute_position_bucket[None, :, :],
                relative_position_bucket,
            )

        # 嵌入相对位置桶
        embeds = F.embedding(relative_position_bucket, self.relative_attention_bias)
        # 转置嵌入张量维度以便后续计算
        embeds = embeds.permute(0, 3, 1, 2).contiguous()
        # 返回嵌入张量
        return embeds

    # 计算分段相对位置桶
    def _segment_relative_position_bucket(self, query_segment, key_segment):
        return query_segment * self.num_segments + key_segment
    # 计算相对位置对应的桶号
    def _position_bucket(self, relative_position, num_buckets=32, max_distance=128):
        # 初始化相对桶号
        relative_buckets = 0
        # CPMAnt 始终是双向的
        num_buckets //= 2
        # 计算相对位置是否大于0，若是则赋值为 num_buckets
        relative_buckets = (relative_position > 0).to(torch.int32) * num_buckets
        # 取相对位置的绝对值
        relative_position = torch.abs(relative_position)
        # 计算最大精确值
        max_exact = num_buckets // 2
        # 判断相对位置是否小于最大精确值
        is_small = relative_position < max_exact
        # 如果相对位置大于最大精确值，计算相对位置对应的桶号
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.int32)
        # 如果相对位置对应的桶号大于等于num_buckets-1，则取num_buckets-1
        relative_postion_if_large = torch.min(
            relative_postion_if_large,
            torch.full_like(relative_postion_if_large, num_buckets - 1),
        )
        # 最终相对桶号为 is_small 为真时相对位置转为整数，否则取计算得到的相对位置
        relative_buckets += torch.where(is_small, relative_position.to(torch.int32), relative_postion_if_large)
        # 返回相对桶号
        return relative_buckets
# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->CPMAnt
class CpmAntOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，输入维度为config.intermediate_size，输出维度为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，输入维度为config.hidden_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，dropout概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用线性层处理hidden_states
        hidden_states = self.dense(hidden_states)
        # 使用dropout层处理hidden_states
        hidden_states = self.dropout(hidden_states)
        # 使用LayerNorm层处理hidden_states和input_tensor相加的结果
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的hidden_states
        return hidden_states


class CpmAntPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置config_class为CpmAntConfig，表明这个模型使用CpmAntConfig作为配置类
    config_class = CpmAntConfig
    # 设置base_model_prefix为"cpmant"，表示模型参数以"cpmant"开头
    base_model_prefix = "cpmant"

    def _init_weights(self, module):
        """Initialize the weights"""
        # 对module进行初始化权重操作
        if isinstance(module, nn.Linear):
            # 对nn.Linear层进行权重初始化，权重值从均值为0，标准差为self.config.init_std的正态分布中采样
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                # 如果存在偏置项，将其初始化为0
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对nn.Embedding层进行权重初始化，权重值从均值为0，标准差为self.config.init_std的正态分布中采样
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                # 如果存在padding_idx，将其对应的权重初始化为0
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 将nn.LayerNorm层的偏置初始化为0
            module.bias.data.zero_()
            # 将nn.LayerNorm层的权重初始化为1
            module.weight.data.fill_(1.0)
        elif isinstance(module, CpmAntLayerNorm):
            # 将自定义的CpmAntLayerNorm层的权重初始化为1
            module.weight.data.fill_(1.0)
        elif isinstance(module, CpmAntSegmentPositionEmbedding):
            # 将自定义的CpmAntSegmentPositionEmbedding层的relative_attention_bias的权重从均值为0，标准差为self.config.init_std的正态分布中采样初始化
            module.relative_attention_bias.data.normal_(mean=0.0, std=self.config.init_std)


CPMANT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters
        config ([`~CpmAntConfig`]): Model configuration class with all the parameters of the
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

CPMANT_INPUTS_DOCSTRING = r"""



注释：
    Args:
        input_ids (`torch.Tensor` of shape `(batch_size, seq_len)`):
            # 输入序列标记在词汇表中的索引。

            # 可以使用 [`CPMAntTokenizer`] 获取这些索引。详见 [`PreTrainedTokenizer.encode`] 和
            # [`PreTrainedTokenizer.__call__`] 的详情。

            # [什么是输入 ID？](../glossary#input-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            # 包含预先计算的隐藏状态（自注意力块和交叉注意力块中的键和值），可以用于加速序列解码。

            # 如果传递了 `use_cache=True` 或者 `config.use_cache=True`，则返回（详见 `past_key_values` 输入）。

        use_cache (`bool`, *optional*):
            # 如果设置为 `True`，则返回 `past_key_values` 键值状态，并可用于加速解码（详见 `past_key_values`）。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。
        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是一个普通的元组。
# 定义一个 CPMAnt Model 类，输出原始的隐藏状态，没有具体的头部结构
# 包含 CPMANT_START_DOCSTRING 的文档字符串
class CpmAntModel(CpmAntPreTrainedModel):
    def __init__(self, config: CpmAntConfig):
        # 初始化函数，接受一个 CpmAntConfig 类型的参数
        super().__init__(config)
        # 创建 CPM Ant 编码器
        self.encoder = CpmAntEncoder(config)
        # 创建分段嵌入，使用 nn.Embedding
        self.segment_embedding = nn.Embedding(config.segment_types, config.hidden_size)
        # 创建输入嵌入，使用 nn.Embedding
        self.input_embedding = nn.Embedding(
            config.vocab_size + config.prompt_types * config.prompt_length, config.hidden_size
        )
        # 创建位置偏差，使用 CpmAntSegmentPositionEmbedding
        self.position_bias = CpmAntSegmentPositionEmbedding(config)
        # 保存 prompt_length 和 vocab_size 到对象中
        self.prompt_length = config.prompt_length
        self.vocab_size = config.vocab_size

        # 调用 post_init 方法
        self.post_init()

    # 定义获取输入嵌入的方法
    def get_input_embeddings(self):
        return self.input_embedding

    # 定义设置输入嵌入的方法
    def set_input_embeddings(self, embeddings, **kwargs):
        self.input_embedding = embeddings

    # 准备注意力掩码的内部方法
    def _prepare_attention_mask(self, input_ids, span, context, length):
        # ... (具体实现略)

    # 前向传播方法，包括参数和返回值的文档字符串
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    # 包含 CPMANT_START_DOCSTRING 的文档字符串
    @add_start_docstrings(
        """
        The CPMAnt Model with a language modeling head on top (linear layer with weights tied to the input embeddings).
        """,
        CPMANT_START_DOCSTRING,
    )
    class CpmAntForCausalLM(CpmAntPreTrainedModel):
        # 定义 _tied_weights_keys 变量
        _tied_weights_keys = ["lm_head.weight"]
    # 初始化函数，接收一个 CpmAntConfig 对象作为参数
    def __init__(self, config: CpmAntConfig):
        # 调用父类的初始化函数
        super().__init__(config)
        # 创建一个 CpmAntModel 对象
        self.cpmant = CpmAntModel(config)

        # lm_head.weight 是与 cpmant.input_embedding.weight 相关联的
        # 创建一个线性层，用于对隐藏状态进行线性变换，不使用偏置项
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size + config.prompt_types * config.prompt_length, bias=False
        )
        # 调用类中的 post_init 函数
        self.post_init()

    # 前向传播函数，接收多个参数
    @add_start_docstrings_to_model_forward(CPMANT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,  # 用于文本生成管道的虚拟参数
        **kwargs,
    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.cpmant.input_embedding

    # 设置输入嵌入层
    def set_input_embeddings(self, embeddings):
        self.cpmant.input_embedding = embeddings

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 为生成准备输入数据
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        input_ids = input_ids.int()
        # 保存虚拟注意力掩码的内存使用
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = torch.zeros(1, 1)

        return {
            "input_ids": input_ids,
            "use_cache": kwargs["use_cache"],
            "past_key_values": kwargs.get("past_key_values", None),
        }

    # 重排缓存数据
    def _reorder_cache(self, past_key_values, beam_idx):
        past_key_values = [list(each) if each is not None else each for each in past_key_values]
        for key_value_layer in past_key_values:
            key_value_layer[0] = key_value_layer[0][beam_idx]
            key_value_layer[1] = key_value_layer[1][beam_idx]
        return past_key_values
```