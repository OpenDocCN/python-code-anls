# `.\models\cpmant\modeling_cpmant.py`

```
# 设置文件编码格式为UTF-8
# 版权声明，指明本代码文件的版权归属
# 根据Apache许可证2.0，除非符合许可证条件，否则不得使用此文件
# 可在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依据本软件的发布是在"按现状"的基础上，不附带任何明示或暗示的担保或条件
# 有关授权的详细信息，请参阅许可证。
""" PyTorch CPMAnt"""

import math  # 导入数学库，用于执行数学运算
from typing import List, Optional, Tuple, Union  # 引入类型提示的相关类

import torch  # 导入PyTorch深度学习库
import torch.nn.functional as F  # 导入PyTorch中的函数库
import torch.utils.checkpoint  # 导入PyTorch的checkpoint工具
from torch import nn  # 从PyTorch导入神经网络模块
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数

from ...activations import ACT2FN  # 从上层目录中导入激活函数
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast  # 从模型输出中导入基础模型输出和有过去上下文的因果语言建模输出
from ...modeling_utils import PreTrainedModel  # 从模型工具中导入预训练模型类
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging  # 从工具类中导入文档字符串添加工具和日志记录工具
from .configuration_cpmant import CpmAntConfig  # 从当前目录导入CPMAnt模型的配置类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CHECKPOINT_FOR_DOC = "openbmb/cpm-ant-10b"  # CPMAnt模型的预训练检查点路径
_CONFIG_FOR_DOC = "CpmAntConfig"  # CPMAnt模型的配置类名称

CPMANT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openbmb/cpm-ant-10b",  # CPMAnt模型的预训练模型存档列表中包含的路径
    # 可在此处查看所有CPMAnt模型：https://huggingface.co/models?filter=cpmant
]


class CpmAntLayerNorm(nn.Module):
    """
    We use Root Mean Square (RMS) Layer Normalization, please see https://arxiv.org/abs/1910.07467 for details."
    """

    def __init__(self, config: CpmAntConfig):
        super().__init__()  # 调用父类的初始化方法

        self.eps = config.eps  # 初始化层归一化时的epsilon值
        self.dim_norm = config.hidden_size  # 从配置中获取隐藏尺寸
        self.weight = nn.Parameter(torch.empty(config.hidden_size))  # 初始化权重参数

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, seq_len, dim_in)`)
        """
        if hidden_states.size(-1) != self.dim_norm:
            raise AssertionError("hidden_states.size(-1) != self.dim_norm")  # 如果隐藏状态的最后一个维度不等于预期的尺寸，则引发断言错误

        old_dtype = hidden_states.dtype  # 保存旧的数据类型
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)  # 计算方差
        hidden_states = (hidden_states * torch.rsqrt(variance + self.eps)).to(old_dtype) * self.weight  # 应用层归一化
        return hidden_states  # 返回归一化后的隐藏状态


class CpmAntAttention(nn.Module):
    # 初始化方法，接受一个配置对象 config: CpmAntConfig
    def __init__(self, config: CpmAntConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置模型的隐藏大小为配置中的隐藏大小
        self.dim_model = config.hidden_size
        # 设置注意力头的数量为配置中的注意力头数量
        self.num_heads = config.num_attention_heads
        # 设置每个注意力头的维度为配置中的注意力头维度
        self.dim_head = config.dim_head

        # 创建用于投影查询向量的线性层，输出维度为注意力头数乘以每个头的维度
        self.project_q = nn.Linear(self.dim_model, self.num_heads * self.dim_head, bias=False)
        # 创建用于投影键向量的线性层，输出维度为注意力头数乘以每个头的维度
        self.project_k = nn.Linear(self.dim_model, self.num_heads * self.dim_head, bias=False)
        # 创建用于投影值向量的线性层，输出维度为注意力头数乘以每个头的维度
        self.project_v = nn.Linear(self.dim_model, self.num_heads * self.dim_head, bias=False)

        # 创建用于输出注意力计算结果的线性层，输入为注意力头数乘以每个头的维度，输出为隐藏大小
        self.attention_out = nn.Linear(self.num_heads * self.dim_head, self.dim_model, bias=False)

        # 创建一个在最后一个维度上进行 softmax 操作的 Softmax 层
        self.softmax = torch.nn.Softmax(dim=-1)

        # 如果配置中指定了 dropout 概率，则创建一个 Dropout 层，否则设为 None
        if config.dropout_p is not None:
            self.dropout = torch.nn.Dropout(p=config.dropout_p)
        else:
            self.dropout = None
class CpmAntSelfAttentionBlock(nn.Module):
    def __init__(self, config: CpmAntConfig):
        super().__init__()
        # 初始化自注意力模块前的 LayerNormalization 层
        self.layernorm_before_attention = CpmAntLayerNorm(config)
        # 初始化自注意力机制模块
        self.self_attention = CpmAntAttention(config)
        # 如果配置中定义了 dropout 概率，则创建对应的 Dropout 层，否则设为 None
        if config.dropout_p:
            self.dropout = torch.nn.Dropout(config.dropout_p)
        else:
            self.dropout = None

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
                自注意力模块的输入，可以是一批序列的原始嵌入。
            attention_mask (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                遮罩矩阵，避免无效区域参与自注意力计算。
            position_bias (`torch.Tensor` of shape `(batch, len_seq, len_seq)`):
                位置偏置，提供给自注意力模块的位置信息。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。
            past_key_values (`Tuple(torch.FloatTensor)`, *optional*):
                缓存的过去键和值投影状态。
            use_cache (`bool`, *optional*):
                如果设置为 `True`，则返回 `past_key_values` 键值状态，可用于加速解码过程（参见 `past_key_values`）。
        """
        # 应用 LayerNormalization 到输入的 hidden_states
        outputs = self.layernorm_before_attention(hidden_states)
        # 调用自注意力模块进行计算
        outputs = self.self_attention(
            outputs, outputs, attention_mask, position_bias, output_attentions, past_key_values, use_cache
        )

        outputs, attn_weights, current_key_value = outputs

        # 如果存在 Dropout 层，则应用 Dropout
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        # 将输出与原始输入相加，作为最终输出
        hidden_states = hidden_states + outputs

        return hidden_states, attn_weights, current_key_value


class CpmAntDenseGatedACT(nn.Module):
    def __init__(self, config: CpmAntConfig):
        super().__init__()
        # 初始化线性变换 w_0 和 w_1
        self.w_0 = nn.Linear(config.hidden_size, config.dim_ff, bias=False)
        self.w_1 = nn.Linear(config.hidden_size, config.dim_ff, bias=False)
        # 初始化激活函数 GELU
        self.act = torch.nn.GELU()

    def forward(self, hidden_states: torch.Tensor):
        """通过非线性操作将输入张量从一个特征空间转换到另一个特征空间

        Args:
            hidden_states (`torch.Tensor` of shape `(batch, seq_len, dim_in)`)
        """
        # 计算门控分数
        gate_score = self.act(self.w_0(hidden_states))
        # 进行线性变换
        hidden_states = self.w_1(hidden_states)
        # 使用门控分数对 hidden_states 进行加权乘法
        hidden_states = gate_score * hidden_states
        return hidden_states
# 定义一个名为 CpmAntTransformerBlock 的类，继承自 nn.Module
class CpmAntTransformerBlock(nn.Module):
    # 初始化函数，接收一个 config 参数，类型为 CpmAntConfig
    def __init__(self, config: CpmAntConfig):
        super().__init__()
        # 创建 self_att 属性，使用 CpmAntSelfAttentionBlock 类初始化，传入 config 参数
        self.self_att = CpmAntSelfAttentionBlock(config)
        # 创建 ffn 属性，使用 CpmAntFFNBlock 类初始化，传入 config 参数
        self.ffn = CpmAntFFNBlock(config)

    # 前向传播函数定义
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
                输入的隐藏状态，形状为 (batch, len_seq, dim_model)。
            attention_mask (`torch.Tensor`):
                注意力掩码，形状可以根据具体应用而变化。
            position_bias (`Optional[torch.Tensor]`, optional):
                位置偏置张量，形状可以根据具体应用而变化，默认为 None。
            output_attentions (`Optional[bool]`, optional):
                是否输出注意力权重，默认为 False。
            past_key_values (`Optional[Tuple[torch.Tensor, torch.Tensor]]`, optional):
                过去的键-值对，用于缓存，形状为 (key, value)，默认为 None。
            use_cache (`Optional[bool]`, optional):
                是否使用缓存，默认为 None。

        Returns:
            `torch.Tensor`: 经过自注意力和前馈网络后的隐藏状态张量。
        """
        # 对输入的 hidden_states 进行自注意力操作，并将结果保存在 ln_outputs 中
        ln_outputs = self.self_att(hidden_states, attention_mask, position_bias,
                                   output_attentions, past_key_values, use_cache)
        # 将 ln_outputs 输入到前馈网络 ffn 中，得到输出并保存在 outputs 中
        outputs = self.ffn(ln_outputs)
        # 如果存在 dropout 层，则对输出进行 dropout 处理
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        # 将原始隐藏状态 hidden_states 与前馈网络的输出相加，得到最终的隐藏状态结果
        hidden_states = hidden_states + outputs
        # 返回最终的隐藏状态结果
        return hidden_states
    ):
        """
        Args:
            hidden_states (`torch.Tensor`):
                输入到层的张量，形状为 `(batch, seq_len, dim_model)`
            attention_mask (`torch.Tensor`):
                避免无效区域参与计算的张量，形状为 `(batch, seq_len, seq_len)`
            position_bias (`torch.Tensor`):
                提供位置信息给注意力机制的张量，形状为 `(num_heads, seq_len, seq_len)`
            output_attentions (`bool`, *可选*):
                是否返回所有注意力层的注意力张量。
            past_key_values (`Tuple[torch.Tensor, torch.Tensor]`, *可选*):
                缓存的过去键和值投影状态
            use_cache (`bool`, *可选*):
                如果设置为 `True`，则返回 `past_key_values` 键值状态，可用于加速解码 (参见 `past_key_values`)。
        """
        # 使用 self_att 层处理隐藏状态
        hidden_states = self.self_att(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # 解包处理后的隐藏状态、注意力权重和当前键值
        hidden_states, attn_weights, current_key_value = hidden_states

        # 使用 ffn 层处理隐藏状态
        hidden_states = self.ffn(hidden_states)

        # 返回处理后的隐藏状态、注意力权重和当前键值
        return hidden_states, attn_weights, current_key_value
class CpmAntEncoder(nn.Module):
    # CpmAntEncoder 类定义，继承自 nn.Module
    def __init__(self, config: CpmAntConfig):
        # 初始化方法，接受一个 CpmAntConfig 类型的参数 config
        super().__init__()
        # 调用父类的初始化方法
        self.num_layers = config.num_hidden_layers
        # 从 config 中获取隐藏层的数量
        self.layers = nn.ModuleList([CpmAntTransformerBlock(config) for ith in range(self.num_layers)])
        # 使用列表推导式创建一个包含多个 CpmAntTransformerBlock 实例的 ModuleList

        self.output_layernorm = CpmAntLayerNorm(config)
        # 初始化输出层的 LayerNorm

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
                输入的张量，形状为 `(batch, seq_len, dim_model)`
            attention_mask (`torch.Tensor`):
                注意力掩码张量，形状为 `(batch, seq_len, seq_len)`
            position_bias (`torch.Tensor`):
                位置偏置张量，提供位置信息给注意力机制，形状为 `(num_heads, seq_len, seq_len)`
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量
            output_hidden_states (`bool`, *optional*):
                是否返回所有层的隐藏状态
            past_key_values (`Tuple[torch.Tensor, torch.Tensor])`, *optional*):
                缓存的过去键和值投影状态
            use_cache (`bool`, *optional*):
                如果为 `True`，返回 `past_key_values` 键值状态以加速解码
        """
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出隐藏状态，则初始化空元组，否则为 None
        all_self_attns = () if output_attentions else None
        # 如果需要输出注意力张量，则初始化空元组，否则为 None
        current_key_values = () if use_cache else None
        # 如果使用缓存，则初始化空元组，否则为 None

        for i, layer in enumerate(self.layers):
            # 遍历所有 Transformer 层
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            layer_outputs = layer(
                hidden_states,
                attention_mask,
                position_bias,
                output_attentions=output_attentions,
                past_key_values=past_key_values[i] if past_key_values else None,
                use_cache=use_cache,
            )
            # 调用 Transformer 层的 forward 方法
            hidden_states, attn_weights, current_key_value = layer_outputs
            # 获取 Transformer 层的输出：隐藏状态、注意力权重、当前键值状态
            if output_attentions:
                all_self_attns += (attn_weights,)
                # 如果需要输出注意力张量，则将当前注意力权重添加到 all_self_attns 中
            if current_key_value is not None:
                current_key_values = current_key_values + (current_key_value,)
                # 如果当前键值状态不为 None，则添加到 current_key_values 中

        hidden_states = self.output_layernorm(hidden_states)
        # 对最终的隐藏状态进行 LayerNorm 处理

        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            # 如果需要输出隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中

        return hidden_states, current_key_values, all_hidden_states, all_self_attns
        # 返回最终的隐藏状态、当前键值状态、所有隐藏状态、所有注意力张量
# 从transformers.models.bert.modeling_bert.BertIntermediate复制而来，将Bert->CPMAnt
class CpmAntIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将输入特征大小转换为中间层特征大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择激活函数，可能是预定义的激活函数或者自定义的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用线性层进行特征转换
        hidden_states = self.dense(hidden_states)
        # 应用中间层的激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class CpmAntSegmentPositionEmbedding(nn.Module):
    def __init__(self, config: CpmAntConfig):
        super().__init__()

        # 设置注意力头数、位置偏置的桶数、最大距离和段落数
        self.num_heads = config.num_attention_heads
        self.num_buckets = config.position_bias_num_buckets
        self.max_distance = config.position_bias_max_distance
        self.num_segments = config.segment_types

        # 定义相对注意力偏置的参数，形状为 (段落数 * 段落数 + 桶数, 注意力头数)
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
            # 进入上下文管理器，禁用梯度计算
            with torch.no_grad():
                # 获取批量大小和键值位置序列的长度
                batch = key_pos.size(0)
                keylen = key_pos.size(1)
                querylen = query_pos.size(1)

                # 检查键值位置序列的批量大小是否与查询位置序列相同，若不同则引发断言错误
                if key_pos.size(0) != query_pos.size(0):
                    raise AssertionError(
                        f"key_pos.size(0) should be equal to query_pos.size(0), but got {key_pos.size(0)} and {query_pos.size(0)}!"
                    )
                # 检查键值长度和键段长度是否一致，若不一致则引发断言错误
                if keylen != key_segment.size(1) or querylen != query_segment.size(1):
                    raise AssertionError(
                        f"keylen should be equal to key_segment.size(1), but got {keylen} and {key_segment.size(1)}!"
                    )
                # 检查查询长度和查询段长度是否一致，若不一致则引发断言错误
                if querylen != query_segment.size(1):
                    raise AssertionError(
                        f"querylen should be equal to query_segment.size(1), but got {querylen} and {query_segment.szie(1)}!"
                    )

                # 对键值位置序列和查询位置序列进行形状重塑
                key_pos = key_pos.view(batch, -1, keylen)
                query_pos = query_pos.view(batch, querylen, -1)
                key_segment = key_segment.view(batch, -1, keylen)
                query_segment = query_segment.view(batch, querylen, -1)

                # 计算相对位置桶
                relative_position_bucket = self._segment_relative_position_bucket(query_segment, key_segment)
                relative_position_bucket = relative_position_bucket + self.num_buckets

                # (batch, len_q, len_k)
                # 计算绝对位置桶
                absolute_position_bucket = self._position_bucket(
                    torch.arange(keylen, dtype=torch.int32, device=relative_position_bucket.device)[None, :]
                    - torch.arange(querylen, dtype=torch.int32, device=relative_position_bucket.device)[:, None],
                    num_buckets=self.num_buckets,
                    max_distance=self.max_distance,
                )
                # 根据条件更新相对位置桶
                relative_position_bucket = torch.where(
                    (key_segment == query_segment),
                    absolute_position_bucket[None, :, :],
                    relative_position_bucket,
                )

            # (batch, len_q, len_k, num_heads)
            # 使用相对注意力偏置对相对位置桶进行嵌入
            embeds = F.embedding(relative_position_bucket, self.relative_attention_bias)
            # (batch, num_heads, len_q, len_k)
            # 重新排列张量维度以匹配注意力矩阵的期望格式
            embeds = embeds.permute(0, 3, 1, 2).contiguous()
            return embeds

    # 计算查询段和键段的相对位置桶
    def _segment_relative_position_bucket(self, query_segment, key_segment):
        return query_segment * self.num_segments + key_segment
    # 定义一个方法来计算相对位置对应的桶号
    def _position_bucket(self, relative_position, num_buckets=32, max_distance=128):
        # CPMAnt 算法中始终是双向的
        num_buckets //= 2
        # 根据相对位置是否大于零来确定相对桶号的基数
        relative_buckets = (relative_position > 0).to(torch.int32) * num_buckets
        # 计算相对位置的绝对值
        relative_position = torch.abs(relative_position)
        # 定义桶的最大精确值
        max_exact = num_buckets // 2
        # 判断相对位置是否属于小距离
        is_small = relative_position < max_exact
        # 如果是大距离，则计算大距离情况下的相对桶号
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.int32)
        # 确保相对桶号不超出桶的最大数量
        relative_postion_if_large = torch.min(
            relative_postion_if_large,
            torch.full_like(relative_postion_if_large, num_buckets - 1),
        )
        # 根据距离大小选择最终的相对桶号
        relative_buckets += torch.where(is_small, relative_position.to(torch.int32), relative_postion_if_large)
        # 返回计算得到的相对桶号
        return relative_buckets
# 从transformers.models.bert.modeling_bert.BertOutput复制并将Bert->CPMAnt
class CpmAntOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将中间尺寸的输出转换为隐藏尺寸
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，用于规范隐藏状态的输出
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，用于随机失活隐藏状态中的一部分单元
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态输入全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态进行随机失活
        hidden_states = self.dropout(hidden_states)
        # 将随机失活后的隐藏状态与输入张量相加，再输入LayerNorm层进行规范化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回规范化后的隐藏状态作为输出
        return hidden_states


class CpmAntPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定模型的配置类为CpmAntConfig
    config_class = CpmAntConfig
    # 模型参数的前缀设置为"cpmant"
    base_model_prefix = "cpmant"

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 对线性层的权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            # 如果存在偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对嵌入层的权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            # 如果设置了padding_idx，则将对应位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对LayerNorm层的偏置项初始化为零，权重初始化为1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, CpmAntLayerNorm):
            # 对自定义的CpmAntLayerNorm层的权重初始化为1.0
            module.weight.data.fill_(1.0)
        elif isinstance(module, CpmAntSegmentPositionEmbedding):
            # 对自定义的CpmAntSegmentPositionEmbedding层的相对注意力偏置项进行正态分布初始化
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
    Args:
        input_ids (`torch.Tensor` of shape `(batch_size, seq_len)`):
            # 输入序列标记在词汇表中的索引。

            # 可以使用 `CPMAntTokenizer` 获得这些索引。参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__` 获取更多细节。

            # [什么是输入ID？](../glossary#input-ids)

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            # 包含预先计算的隐藏状态（自注意力块和交叉注意力块中的键和值），可以用于加速序列解码。

            # 当 `use_cache=True` 或 `config.use_cache=True` 时返回。

        use_cache (`bool`, *optional*):
            # 如果设置为 `True`，则返回 `past_key_values` 中的键值状态，可用于加速解码。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。

        return_dict (`bool`, *optional*):
            # 是否返回一个 `~utils.ModelOutput` 而不是一个普通的元组。
"""
@add_start_docstrings(
    """
    The CPMAnt Model outputting raw hidden-states without any specific head on top.
    """,
    CPMANT_START_DOCSTRING,
)
"""
定义一个 CPMAnt 模型类，用于生成不带特定输出头的原始隐藏状态。

class CpmAntModel(CpmAntPreTrainedModel):
    """
    CPMAnt 模型类，继承自 CpmAntPreTrainedModel。
    """
    def __init__(self, config: CpmAntConfig):
        """
        初始化方法，接受一个 CpmAntConfig 对象作为参数。
        """
        super().__init__(config)
        # 初始化编码器
        self.encoder = CpmAntEncoder(config)
        # 初始化分段嵌入
        self.segment_embedding = nn.Embedding(config.segment_types, config.hidden_size)
        # 初始化输入嵌入
        self.input_embedding = nn.Embedding(
            config.vocab_size + config.prompt_types * config.prompt_length, config.hidden_size
        )
        # 初始化位置偏置
        self.position_bias = CpmAntSegmentPositionEmbedding(config)
        # 设置提示长度和词汇表大小
        self.prompt_length = config.prompt_length
        self.vocab_size = config.vocab_size

        # 执行初始化后的附加步骤
        self.post_init()

    def get_input_embeddings(self):
        """
        返回输入嵌入层。
        """
        return self.input_embedding

    def set_input_embeddings(self, embeddings, **kwargs):
        """
        设置输入嵌入层。
        """
        self.input_embedding = embeddings

    def _prepare_attention_mask(self, input_ids, span, context, length):
        """
        准备注意力掩码。
        """
        batch = input_ids.size(0)
        seqlen = input_ids.size(1)
        device = input_ids.device

        # 创建方向性掩码
        directional_mask_2d = torch.arange(seqlen, device=device) <= torch.arange(seqlen, device=device).view(-1, 1)
        attention_mask = context[:, None, :] | (
            context[:, :, None].logical_not() & directional_mask_2d.view(1, seqlen, seqlen)
        )
        attention_mask = attention_mask & (span[:, None, :] == span[:, :, None])

        # 创建左填充掩码
        mask_1d = (
            torch.tensor(list(range(seqlen - self.prompt_length))[::-1], device=device)[None, :].repeat(batch, 1)
            < length[:, None]
        )
        mask_1d = torch.cat((torch.ones(batch, self.prompt_length, device=device).bool(), mask_1d), dim=1)
        attention_mask = mask_1d.view(batch, seqlen, 1) & mask_1d.view(batch, 1, seqlen) & attention_mask

        return attention_mask

    @add_start_docstrings_to_model_forward(CPMANT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """
        CPMAnt 模型的前向传播方法，接受多个输入参数并返回输出。

        Args:
            input_ids (Optional[torch.Tensor], optional): 输入张量，默认为 None。
            output_attentions (Optional[bool], optional): 是否输出注意力，默认为 None。
            output_hidden_states (Optional[bool], optional): 是否输出隐藏状态，默认为 None。
            past_key_values (Optional[Tuple[Tuple[torch.Tensor]]], optional): 过去键值元组，默认为 None。
            use_cache (Optional[bool], optional): 是否使用缓存，默认为 None。
            return_dict (Optional[bool], optional): 是否返回字典，默认为 None。
            **kwargs: 其他关键字参数。

        Returns:
            模型的输出，包含过去的键值对。
        """
        # 实际前向传播逻辑在子类中实现
        pass


@add_start_docstrings(
    """
    The CPMAnt Model with a language modeling head on top (linear layer with weights tied to the input embeddings).
    """,
    CPMANT_START_DOCSTRING,
)
"""
定义一个带有语言建模头的 CPMAnt 模型类，使用输入嵌入层权重来绑定线性层。
class CpmAntForCausalLM(CpmAntPreTrainedModel):
    """
    CPMAnt 用于因果语言建模的模型类，继承自 CpmAntPreTrainedModel。
    """
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config: CpmAntConfig):
        # 调用父类的构造方法，传入配置参数
        super().__init__(config)
        # 使用给定的配置参数初始化 CpmAntModel 实例
        self.cpmant = CpmAntModel(config)

        # lm_head.weight 被绑定到 cpmant.input_embedding.weight
        # 初始化一个线性层，输入大小为 config.hidden_size，输出大小为 config.vocab_size + config.prompt_types * config.prompt_length，无偏置
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size + config.prompt_types * config.prompt_length, bias=False
        )
        # 执行初始化后续操作
        self.post_init()

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
        attention_mask: Optional[torch.Tensor] = None,  # 文本生成流程中的虚拟参数
        **kwargs,
    ):
        # 此处定义模型的前向传播逻辑，具体实现可能涉及多种输入和输出参数的处理，根据具体实现来理解其作用
        pass

    def get_input_embeddings(self):
        # 返回当前模型的输入嵌入层
        return self.cpmant.input_embedding

    def set_input_embeddings(self, embeddings):
        # 设置当前模型的输入嵌入层为给定的嵌入层
        self.cpmant.input_embedding = embeddings

    def get_output_embeddings(self):
        # 返回当前模型的输出嵌入层（即 lm_head 线性层）
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置当前模型的输出嵌入层为给定的新嵌入层
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # 将输入的 token IDs 转换为整数类型
        input_ids = input_ids.int()
        # 如果 kwargs 中包含 attention_mask，则将其设为一个全零的张量，用于节省内存使用
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = torch.zeros(1, 1)

        # 返回经过处理后的输入字典，包含 input_ids、use_cache 和可能的 past_key_values
        return {
            "input_ids": input_ids,
            "use_cache": kwargs["use_cache"],
            "past_key_values": kwargs.get("past_key_values", None),
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        # 对 past_key_values 中的每个 past_key_value 进行重排序，根据给定的 beam_idx
        past_key_values = [list(each) if each is not None else each for each in past_key_values]
        for key_value_layer in past_key_values:
            key_value_layer[0] = key_value_layer[0][beam_idx]
            key_value_layer[1] = key_value_layer[1][beam_idx]
        return past_key_values
```