# `.\models\umt5\modeling_umt5.py`

```py
# 设置文件编码为 UTF-8
# 版权声明和许可信息，指定代码使用的许可证为 Apache License, Version 2.0
# 不可使用此文件，除非符合 Apache License, Version 2.0 的规定。可以通过上述链接获取许可证副本。
# 根据适用法律或书面同意，本软件按“原样”分发，无任何担保或条件。
# 有关详细信息，请参阅许可证的特定语言，限制和条件
""" PyTorch UMT5 模型."""

import copy  # 导入 copy 模块
import math  # 导入 math 模块
from typing import List, Optional, Tuple, Union  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 导入神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 从 PyTorch 神经网络模块导入损失函数

from ...activations import ACT2FN  # 导入激活函数映射
from ...modeling_outputs import (  # 导入模型输出相关类
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...utils import (  # 导入辅助工具函数和常量
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from .configuration_umt5 import UMT5Config  # 导入 UMT5 模型的配置类


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CONFIG_FOR_DOC = "UMT5Config"  # 文档中显示的配置文件名称
_CHECKPOINT_FOR_DOC = "google/umt5-small"  # 文档中显示的检查点名称


# 从 transformers.models.t5.modeling_t5.T5LayerNorm 复制并改为 UMT5
class UMT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        构造一个 UMT5 风格的 LayerNorm 模块。无偏差和无平均值减法。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 初始化权重参数为全1张量
        self.variance_epsilon = eps  # 方差的小值偏置

    def forward(self, hidden_states):
        # UMT5 使用一个只进行缩放而不进行偏移的 LayerNorm，这也称为均方根层归一化
        # 因此，方差是在没有均值的情况下计算的，而且没有偏差。另外，我们要确保对于半精度输入的累积是在 fp32 中进行的

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)  # 计算方差
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)  # 归一化操作

        # 如果权重数据类型是半精度浮点数或 BF16，则将隐藏状态转换为相应的数据类型
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states  # 返回经归一化处理的隐藏状态乘以权重


# 从 transformers.models.t5.modeling_t5.T5DenseActDense 复制并改为 UMT5
class UMT5DenseActDense(nn.Module):
    # 初始化方法，接收一个UMT5Config类型的配置参数
    def __init__(self, config: UMT5Config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入维度为config.d_model，输出维度为config.d_ff，无偏置项
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 创建一个线性层，输入维度为config.d_ff，输出维度为config.d_model，无偏置项
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # 创建一个以config.dropout_rate为丢弃率的Dropout层
        self.dropout = nn.Dropout(config.dropout_rate)
        # 根据配置选择合适的激活函数，并赋值给self.act
        self.act = ACT2FN[config.dense_act_fn]

    # 前向传播方法，接收隐藏状态hidden_states作为输入
    def forward(self, hidden_states):
        # 输入hidden_states经过self.wi线性层
        hidden_states = self.wi(hidden_states)
        # 经过激活函数self.act
        hidden_states = self.act(hidden_states)
        # 经过丢弃层self.dropout
        hidden_states = self.dropout(hidden_states)
        # 如果self.wo.weight是torch.Tensor类型，且hidden_states的数据类型不等于self.wo.weight的数据类型，且self.wo.weight的数据类型不是torch.int8
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            # 将hidden_states转换为self.wo.weight的数据类型
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        # 输入hidden_states经过self.wo线性层
        hidden_states = self.wo(hidden_states)
        # 返回经过self.wo线性层后的hidden_states
        return hidden_states
# 从 transformers.models.t5.modeling_t5.T5DenseGatedActDense 复制代码，将 T5 替换为 UMT5
class UMT5DenseGatedActDense(nn.Module):
    def __init__(self, config: UMT5Config):
        super().__init__()
        # 定义一个线性层，将输入维度 config.d_model 映射到 config.d_ff，无偏置
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 定义另一个线性层，同样将输入维度 config.d_model 映射到 config.d_ff，无偏置
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 定义一个线性层，将输入维度 config.d_ff 映射回 config.d_model，无偏置
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # 定义一个 dropout 层，丢弃概率为 config.dropout_rate
        self.dropout = nn.Dropout(config.dropout_rate)
        # 根据配置选择激活函数 ACT2FN 中的一个作为 self.act
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        # 将 hidden_states 经过 self.wi_0 和激活函数 self.act 处理得到 hidden_gelu
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 将 hidden_states 经过 self.wi_1 处理得到 hidden_linear
        hidden_linear = self.wi_1(hidden_states)
        # 将 hidden_gelu 和 hidden_linear 逐元素相乘，得到新的 hidden_states
        hidden_states = hidden_gelu * hidden_linear
        # 对 hidden_states 进行 dropout 处理
        hidden_states = self.dropout(hidden_states)

        # 为了使得 8 位量化在 google/flan-t5-xxl 上工作，self.wo 保持为 float32 类型
        # 参考 https://github.com/huggingface/transformers/issues/20287
        # 同时确保权重不是 `int8` 类型，以防用户将 `_keep_in_fp32_modules` 强制设置为 `None`
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            # 将 hidden_states 转换为 self.wo.weight 的数据类型
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        # 将 hidden_states 经过 self.wo 处理得到最终的输出
        hidden_states = self.wo(hidden_states)
        return hidden_states


# 从 transformers.models.t5.modeling_t5.T5LayerFF 复制代码，将 T5 替换为 UMT5
class UMT5LayerFF(nn.Module):
    def __init__(self, config: UMT5Config):
        super().__init__()
        # 如果配置为使用 gated activation，则使用 UMT5DenseGatedActDense，否则使用 UMT5DenseActDense
        if config.is_gated_act:
            self.DenseReluDense = UMT5DenseGatedActDense(config)
        else:
            self.DenseReluDense = UMT5DenseActDense(config)

        # 定义层归一化层，输入维度为 config.d_model，epsilon 为 config.layer_norm_epsilon
        self.layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 定义 dropout 层，丢弃概率为 config.dropout_rate
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        # 对输入的 hidden_states 进行层归一化处理
        forwarded_states = self.layer_norm(hidden_states)
        # 将归一化后的 hidden_states 输入到 self.DenseReluDense 中进行处理
        forwarded_states = self.DenseReluDense(forwarded_states)
        # 将原始的 hidden_states 和经过 dropout 处理后的 forwarded_states 相加作为最终输出
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class UMT5Attention(nn.Module):
    """
    使用 relative_attention_bias 的 T5 注意力模块。
    """
    # 初始化函数，用于初始化一个注意力头部模型
    def __init__(self, config, has_relative_attention_bias=False):
        # 调用父类构造函数初始化
        super().__init__()
        # 根据配置设置是否为解码器
        self.is_decoder = config.is_decoder
        # 设置是否存在相对注意力偏置
        self.has_relative_attention_bias = has_relative_attention_bias
        # 相对注意力的桶数目
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        # 相对注意力的最大距离
        self.relative_attention_max_distance = config.relative_attention_max_distance
        # 模型的维度
        self.d_model = config.d_model
        # 键值映射的维度
        self.key_value_proj_dim = config.d_kv
        # 注意力头部的数量
        self.n_heads = config.num_heads
        # 丢弃率
        self.dropout = config.dropout_rate
        # 内部维度，等于头部数量乘以键值映射的维度
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 创建线性层，用于查询
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        # 创建线性层，用于键
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        # 创建线性层，用于值
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        # 创建线性层，用于输出
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # 如果存在相对注意力偏置，则创建相对注意力偏置的嵌入层
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        
        # 初始化剪枝的注意力头部集合为空集
        self.pruned_heads = set()

    # 重新形状函数，用于调整注意力头部的投影
    def _shape(self, projection: torch.Tensor) -> torch.Tensor:
        # 计算新的投影形状
        new_projection_shape = projection.size()[:-1] + (self.n_heads, self.key_value_proj_dim)
        # 调整投影的形状，将头部移动到第二个位置 (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        # 返回调整后的新投影
        return new_projection
    def _relative_position_bucket(self, relative_position):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0  # 初始化相对位置的桶号为0

        # 获取相对位置的桶数和最大距离
        num_buckets = self.relative_attention_num_buckets
        max_distance = self.relative_attention_max_distance

        # 如果不是解码器模式，调整桶数和相对位置
        if not self.is_decoder:
            num_buckets //= 2  # 桶数减半
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets  # 根据相对位置正负，选择桶号
            relative_position = torch.abs(relative_position)  # 取相对位置的绝对值
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))  # 如果是解码器模式，调整相对位置

        # 现在相对位置在区间[0, inf)

        # 将一半的桶用于精确增量位置
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact  # 判断相对位置是否小于最大精确值

        # 另一半桶用于对数增量位置，直到最大距离
        log_ratio = torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact)
        log_ratio = log_ratio * (num_buckets - max_exact)
        relative_position_if_large = max_exact + log_ratio.to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)  # 根据相对位置大小选择最终的桶号
        return relative_buckets  # 返回计算出的相对位置的桶号
    # 计算相对位置偏置
    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        # 如果设备未指定，则使用相对注意力偏置张量的设备
        if device is None:
            device = self.relative_attention_bias.weight.device
        # 创建上下文位置张量，包含长度为 query_length 的序列，dtype 为 long，设备为指定设备
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # 创建记忆位置张量，包含长度为 key_length 的序列，dtype 为 long，设备为指定设备
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        # 计算相对位置，形状为 (query_length, key_length)
        relative_position = memory_position - context_position
        # 将相对位置转换为桶索引
        relative_position_bucket = self._relative_position_bucket(relative_position)
        # 使用相对注意力偏置张量计算偏置值，形状为 (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # 调整维度顺序，形状变为 (1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        # 返回计算得到的偏置值张量
        return values

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
# UMT5 模型中的自注意力层定义，用于处理自注意力机制
class UMT5LayerSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化自注意力层，配置是否包含相对注意力偏置
        self.SelfAttention = UMT5Attention(config, has_relative_attention_bias=True)
        # 初始化层归一化（Layer Normalization），输入维度为 config.d_model，epsilon 设置为 config.layer_norm_epsilon
        self.layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化 dropout，丢弃率为 config.dropout_rate
        self.dropout = nn.Dropout(config.dropout_rate)

    # 定义前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        past_key_value=None,
    ):
        # 对输入的 hidden_states 进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 将归一化后的 hidden_states 输入到 SelfAttention 层中进行自注意力计算
        attention_output = self.SelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
        )
        # 将原始的 hidden_states 和经过 dropout 处理的 attention_output 相加，作为最终输出的 hidden_states
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 构建输出元组，包含更新后的 hidden_states 和可能的 attention 情况（如果有的话）
        outputs = (hidden_states,) + attention_output[1:]  # 如果有的话，添加 attention
        return outputs


# UMT5 模型中的编码-解码注意力层定义
class UMT5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化编码-解码注意力层，配置不包含相对注意力偏置
        self.EncDecAttention = UMT5Attention(config, has_relative_attention_bias=False)
        # 初始化层归一化（Layer Normalization），输入维度为 config.d_model，epsilon 设置为 config.layer_norm_epsilon
        self.layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化 dropout，丢弃率为 config.dropout_rate
        self.dropout = nn.Dropout(config.dropout_rate)

    # 定义前向传播函数
    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        layer_head_mask=None,
        past_key_value=None,
    ):
        # 对输入的 hidden_states 进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 将归一化后的 hidden_states 输入到 EncDecAttention 层中进行编码-解码注意力计算
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
        )
        # 将原始的 hidden_states 和经过 dropout 处理的 attention_output 相加，作为最终输出的 hidden_states
        layer_output = hidden_states + self.dropout(attention_output[0])
        # 构建输出元组，包含更新后的 hidden_states 和可能的 attention 情况（如果有的话）
        outputs = (layer_output,) + attention_output[1:]  # 如果有的话，添加 attention
        return outputs


# UMT5 模型中的单个块定义，包含自注意力层、可能的编码-解码注意力层和前馈神经网络层
class UMT5Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 标志是否为解码器
        self.is_decoder = config.is_decoder
        # 层列表，用于存放块内的各层
        self.layer = nn.ModuleList()
        # 添加自注意力层到层列表中
        self.layer.append(UMT5LayerSelfAttention(config))
        # 如果是解码器，添加编码-解码注意力层到层列表中
        if self.is_decoder:
            self.layer.append(UMT5LayerCrossAttention(config))
        # 添加前馈神经网络层到层列表中
        self.layer.append(UMT5LayerFF(config))

    # 定义前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        # 依次对层列表中的每一层进行前向传播
        for layer_module in self.layer:
            # 如果层为自注意力或编码-解码注意力层，传递相应参数进行计算
            if isinstance(layer_module, (UMT5LayerSelfAttention, UMT5LayerCrossAttention)):
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    layer_head_mask=cross_attn_layer_head_mask if isinstance(layer_module, UMT5LayerCrossAttention) else layer_head_mask,
                    past_key_value=past_key_value,
                )[0]
            else:
                # 否则，直接对隐藏状态进行前向传播
                layer_outputs = layer_module(hidden_states)
                hidden_states = layer_outputs[0]  # 更新隐藏状态为层的输出

        # 构建输出元组，包含最终更新后的 hidden_states 和可能的 attention 情况（如果有的话）
        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (layer_outputs[1],)  # 添加 attention 情况
        return outputs
        # Self Attention
        # 如果过去的键/值对不为 None，则取其前两个元素作为当前自注意力层的缓存键/值对
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        # 调用第一个层的自注意力机制，处理隐藏状态
        hidden_states, self_attn_weights, present_key_value = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
        )

        # 如果隐藏状态的数据类型为 torch.float16，则将无穷大的值 clamp 到一个较小的值，以支持 fp16 训练
        if hidden_states.dtype == torch.float16:
            max_dtype = torch.finfo(hidden_states.dtype).max
            clamp_value = torch.where(torch.isinf(hidden_states).any(), max_dtype - 1000, max_dtype)
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        # 如果模型是解码器且 encoder_hidden_states 不为 None，则进行交叉注意力计算
        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # 如果过去的键/值对不为 None，则取其后两个元素作为当前交叉注意力层的缓存键/值对
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 调用第二个层的交叉注意力机制，处理隐藏状态和编码器的隐藏状态
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.layer[1](
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
            )
            # 如果隐藏状态的数据类型为 torch.float16，则将无穷大的值 clamp 到一个较小的值，以支持 fp16 训练
            if hidden_states.dtype == torch.float16:
                max_dtype = torch.finfo(hidden_states.dtype).max
                clamp_value = torch.where(torch.isinf(hidden_states).any(), max_dtype - 1000, max_dtype)
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # 更新当前的键/值对，加上交叉注意力的结果
            present_key_value += cross_attn_present_key_value

        # 应用 Feed Forward 层
        hidden_states = self.layer[-1](hidden_states)

        # 如果隐藏状态的数据类型为 torch.float16，则将无穷大的值 clamp 到一个较小的值，以支持 fp16 训练
        if hidden_states.dtype == torch.float16:
            max_dtype = torch.finfo(hidden_states.dtype).max
            clamp_value = torch.where(torch.isinf(hidden_states).any(), max_dtype - 1000, max_dtype)
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 设置输出内容
        outputs = (
            hidden_states,  # 最终的隐藏状态
            present_key_value,  # 当前键/值对
        )

        # 如果需要输出注意力权重，则将自注意力和交叉注意力的权重也加入输出
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        # 返回最终输出结果
        return outputs
# Copied from transformers.models.t5.modeling_t5.T5ClassificationHead with T5->UMT5
# 在 T5ClassificationHead 的基础上复制并修改为 UMT5ClassificationHead

class UMT5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    # 用于句子级分类任务的头部模块

    def __init__(self, config: UMT5Config):
        super().__init__()
        # 调用父类构造函数初始化模块
        self.dense = nn.Linear(config.d_model, config.d_model)
        # 全连接层，输入和输出维度为 config.d_model
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        # Dropout 层，使用概率为 config.classifier_dropout 的概率丢弃神经元
        self.out_proj = nn.Linear(config.d_model, config.num_labels)
        # 全连接层，将维度为 config.d_model 的输入映射到 config.num_labels 的输出

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        # 对输入 hidden_states 进行 Dropout 处理
        hidden_states = self.dense(hidden_states)
        # 将经过 Dropout 处理的 hidden_states 输入全连接层 self.dense
        hidden_states = torch.tanh(hidden_states)
        # 对全连接层的输出应用 Tanh 激活函数
        hidden_states = self.dropout(hidden_states)
        # 再次对输出进行 Dropout 处理
        hidden_states = self.out_proj(hidden_states)
        # 将处理后的 hidden_states 输入全连接层 self.out_proj
        return hidden_states
        # 返回全连接层的输出作为模型的输出结果


class UMT5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 处理权重初始化以及预训练模型下载和加载的抽象类

    config_class = UMT5Config
    # 使用 UMT5Config 类配置模型参数
    base_model_prefix = "transformer"
    # 基础模型前缀名为 "transformer"
    supports_gradient_checkpointing = True
    # 支持梯度检查点

    _no_split_modules = ["UMT5Block"]
    # 不拆分的模块列表，包含 "UMT5Block"
    _keep_in_fp32_modules = ["wo"]
    # 在 FP32 精度下保持的模块列表，包含 "wo"

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs
        # 返回用于测试的虚拟输入数据字典 dummy_inputs

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In UMT5 it is usually set to the pad_token_id. "
                "See UMT5 docs for more information."
            )
        # 如果 decoder_start_token_id 未定义，则抛出 ValueError

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            # 对于代理对象，不支持原生的项目分配
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id
            # 将输入向右移动一位，并在开头插入 decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
            # 如果 pad_token_id 未定义，则抛出 ValueError
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        # 将标签中可能的 -100 值替换为 pad_token_id

        return shifted_input_ids
        # 返回右移后的输入张量


class UMT5Stack(UMT5PreTrainedModel):
    # 初始化方法，接受配置和嵌入标记作为参数
    def __init__(self, config, embed_tokens=None):
        # 调用父类初始化方法，传入配置
        super().__init__(config)
        # 设置嵌入标记属性
        self.embed_tokens = embed_tokens
        # 根据配置设置解码器标志
        self.is_decoder = config.is_decoder
        # 创建一个由多个UMT5Block组成的模块列表，列表长度为配置中指定的层数
        self.block = nn.ModuleList([UMT5Block(config) for i in range(config.num_layers)])
        # 创建一个最终层归一化对象，将模型维度和epsilon作为参数
        self.final_layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 创建一个dropout层，使用配置中的dropout率
        self.dropout = nn.Dropout(config.dropout_rate)

        # 初始化权重并应用最终处理
        # 设置梯度检查点为False
        self.gradient_checkpointing = False
        # 执行后初始化操作
        self.post_init()

    # 返回输入嵌入对象的方法
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置新的输入嵌入对象的方法
    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    # 前向传播方法，接收多个输入参数
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
UMT5_START_DOCSTRING = r"""
    UMT5 模型是由 Colin Raffel, Noam Shazeer, Adam Roberts 等人在文献 [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
    Transformer](https://arxiv.org/abs/1910.10683) 中提出的。它是一个编码解码转换器，在文本去噪生成任务中进行预训练。

    该模型继承自 [`PreTrainedModel`]。请查阅其超类文档以了解库实现的通用方法（如下载或保存模型、调整输入嵌入大小、修剪头等）。

    此模型也是 PyTorch 的 [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 子类。
    您可以像使用常规 PyTorch 模块一样使用它，并参考 PyTorch 文档以获取有关一般使用和行为的所有相关信息。

    参数:
        config ([`UMT5Config`]): 包含模型所有参数的配置类。
            使用配置文件进行初始化不会加载与模型相关的权重，只会加载配置信息。
            查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""

UMT5_INPUTS_DOCSTRING = r"""
    输入文档字符串未提供具体内容，暂无注释。
"""

UMT5_ENCODER_INPUTS_DOCSTRING = r"""
    编码器输入文档字符串未提供具体内容，暂无注释。
"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记的索引，形状为(batch_size, sequence_length)。
            # UMT5 是一个具有相对位置嵌入的模型，因此可以在输入的右侧和左侧进行填充。

            # 可以使用 `AutoTokenizer` 获取这些索引。详见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`。

            # 要了解如何为预训练准备 `input_ids`，请查看 [UMT5 Training](./umt5#training)。
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 遮盖掩码，避免在填充的标记索引上执行注意力操作。掩码值在 `[0, 1]` 之间选择：

            # - 1 表示 **未被遮盖** 的标记，
            # - 0 表示 **被遮盖** 的标记。

            # [什么是注意力遮盖？](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于取消选择自注意力模块中特定头部的掩码。掩码值在 `[0, 1]` 之间选择：

            # - 1 表示头部 **未被遮盖**，
            # - 0 表示头部 **被遮盖**。

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            # 可选参数，可以直接传递嵌入表示，而不是传递 `input_ids`。如果希望更多控制如何将 `input_ids` 索引转换为相关联向量，
            # 这非常有用，而不是使用模型内部的嵌入查找矩阵。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。查看返回的张量中的 `attentions` 以获取更多细节。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。查看返回的张量中的 `hidden_states` 以获取更多细节。

        return_dict (`bool`, *optional*):
            # 是否返回 `~utils.ModelOutput` 而不是普通的元组。
"""
The bare UMT5 Model transformer outputting raw hidden-states without any specific head on top.
"""
class UMT5Model(UMT5PreTrainedModel):
    r"""
    Examples:

    ```
    >>> from transformers import UMT5Model, AutoTokenizer

    >>> model = UMT5Model.from_pretrained("google/umt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
    >>> noisy_text = "UN Offizier sagt, dass weiter <extra_id_0> werden muss in Syrien."
    >>> label = "<extra_id_0> verhandelt"
    >>> inputs = tokenizer(inputs, return_tensors="pt")
    >>> labels = tokenizer(label=label, return_tensors="pt")

    >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
    >>> hidden_states = outputs.last_hidden_state
    ```

    Initializes UMT5 model with configuration parameters and shared embeddings.

    Args:
        config (UMT5Config): Configuration object defining model parameters.

    Attributes:
        model_type (str): Type of the model ("umt5").
        config_class (UMT5Config): Class defining model configuration settings.
        _tied_weights_keys (List[str]): List of keys for tied weights between encoder and decoder embeddings.
        shared (nn.Embedding): Shared embeddings across encoder and decoder.
        encoder (UMT5Stack): Encoder stack of the UMT5 model.
        decoder (UMT5Stack): Decoder stack of the UMT5 model.
    """
    
    model_type = "umt5"
    config_class = UMT5Config
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # Initialize encoder with modified configuration
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UMT5Stack(encoder_config, self.shared)

        # Initialize decoder with modified configuration
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = UMT5Stack(decoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.t5.modeling_t5.T5Model.get_input_embeddings
    def get_input_embeddings(self):
        """
        Returns the shared input embeddings used by the model.
        """
        return self.shared

    # Copied from transformers.models.t5.modeling_t5.T5Model.set_input_embeddings
    def set_input_embeddings(self, new_embeddings):
        """
        Sets new shared input embeddings for the model and propagates them to encoder and decoder.
        """
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # Copied from transformers.models.t5.modeling_t5.T5Model._tie_weights
    def _tie_weights(self):
        """
        Ties the weights between encoder and decoder embeddings if configured to do so.
        """
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # Copied from transformers.models.t5.modeling_t5.T5Model.get_encoder
    def get_encoder(self):
        """
        Returns the encoder stack of the model.
        """
        return self.encoder

    # Copied from transformers.models.t5.modeling_t5.T5Model.get_decoder
    def get_decoder(self):
        """
        Returns the decoder stack of the model.
        """
        return self.decoder

    # Copied from transformers.models.t5.modeling_t5.T5Model._prune_heads
    # 定义模型的前向传播函数，用于执行模型的前向计算过程
    @add_start_docstrings_to_model_forward(UMT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，类型为可选的长整型张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩，类型为可选的浮点张量
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器的输入 token IDs，类型为可选的长整型张量
        decoder_attention_mask: Optional[torch.BoolTensor] = None,  # 解码器的注意力遮罩，类型为可选的布尔张量
        head_mask: Optional[torch.FloatTensor] = None,  # 注意力头部的遮罩，类型为可选的浮点张量
        decoder_head_mask: Optional[torch.FloatTensor] = None,  # 解码器注意力头部的遮罩，类型为可选的浮点张量
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 跨注意力头部的遮罩，类型为可选的张量
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 编码器的输出，类型为可选的元组
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 过去的键值对，类型为可选的元组
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入向量，类型为可选的张量
        decoder_inputs_embeds: Optional[torch.Tensor] = None,  # 解码器输入的嵌入向量，类型为可选的张量
        use_cache: Optional[bool] = None,  # 是否使用缓存，类型为可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，类型为可选的布尔值
# 用于给 UMT5ForConditionalGeneration 类添加文档字符串，说明其在语言建模上的应用
@add_start_docstrings("""UMT5 Model with a `language modeling` head on top.""", UMT5_START_DOCSTRING)
class UMT5ForConditionalGeneration(UMT5PreTrainedModel):
    r"""
    Examples:

    ```
    >>> from transformers import UMT5ForConditionalGeneration, AutoTokenizer

    >>> model = UMT5ForConditionalGeneration.from_pretrained("google/umt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, text_target=summary, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> loss = outputs.loss
    ```"""

    # 模型类型标识符
    model_type = "umt5"
    # 被绑定权重的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化函数，接收一个配置对象并进行初始化
    def __init__(self, config):
        super().__init__(config)
        # 设置模型维度
        self.model_dim = config.d_model

        # 共享的嵌入层，使用 nn.Embedding 初始化
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制配置对象用于编码器
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 初始化编码器实例
        self.encoder = UMT5Stack(encoder_config, self.shared)

        # 复制配置对象用于解码器
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 初始化解码器实例
        self.decoder = UMT5Stack(decoder_config, self.shared)

        # 语言建模头部，线性层，将模型维度映射到词汇表大小
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration 中复制的方法
    # 返回共享的嵌入层对象
    def get_input_embeddings(self):
        return self.shared

    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration 中复制的方法
    # 设置共享的嵌入层对象
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration 中复制的方法
    # 如果配置要求，将权重绑定到共享的嵌入层上
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration 中复制的方法
    # 设置输出嵌入层对象
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration 中复制的方法
    # 返回输出嵌入层对象
    def get_output_embeddings(self):
        return self.lm_head

    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration 中复制的方法
    # 返回编码器对象
    def get_encoder(self):
        return self.encoder
    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_decoder
    def get_decoder(self):
        return self.decoder
    # 返回模型的解码器对象
    
    @add_start_docstrings_to_model_forward(UMT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
    # 从输入到输出的前向传播函数，接受多个参数用于控制模型行为和计算，返回输出结果
    
    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用过去的键值对，根据其长度截断输入的序列
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
    
            # 一些生成方法可能只传递最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认行为：保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1
    
            input_ids = input_ids[:, remove_prefix_length:]
    
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
    # 准备生成过程中需要的输入，根据传入的参数返回一个包含各种输入信息的字典
    
    # Copied from transformers.models.t5.modeling_t5.T5ForConditionalGeneration.prepare_decoder_input_ids_from_labels
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)
    # 根据标签生成解码器的输入序列，通过右移操作来实现
    # 定义一个函数 `_reorder_cache`，重新排列缓存中的历史键值
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化一个空元组，用于存储重新排列后的历史键值
        reordered_past = ()
        # 遍历每个层的历史键值
        for layer_past in past_key_values:
            # 对每个层的历史状态按照给定的索引 `beam_idx` 进行重新排序，并转移到对应的设备上
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排列后的历史键值
        return reordered_past
# 在 UMT5 模型的基础上定义了一个编码器模型 UMT5EncoderModel，用于输出编码器的原始隐藏状态，没有额外的特定头部结构。
# 继承自 UMT5PreTrainedModel，这是一个预训练模型基类。

@add_start_docstrings(
    "The bare UMT5 Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    UMT5_START_DOCSTRING,
)
class UMT5EncoderModel(UMT5PreTrainedModel):
    r"""
    Examples:

    ```
    >>> from transformers import UMT5EncoderModel, AutoTokenizer

    >>> model = UMT5EncoderModel.from_pretrained("google/umt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> input_ids = tokenizer(article, return_tensors="pt").input_ids
    >>> outputs = model(input_ids)
    >>> hidden_state = outputs.last_hidden_state
    ```"""

    model_type = "umt5"
    # config_class = UMT5Config
    _tied_weights_keys = ["encoder.embed_tokens.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 创建编码器配置的深层副本，确保不使用缓存，且不是编码器-解码器结构
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 初始化共享的嵌入层和编码器堆栈
        self.encoder = UMT5Stack(encoder_config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 transformers.models.t5.modeling_t5.T5EncoderModel.get_input_embeddings 复制过来
    def get_input_embeddings(self):
        return self.shared

    # 从 transformers.models.t5.modeling_t5.T5EncoderModel.set_input_embeddings 复制过来
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    # 从 transformers.models.t5.modeling_t5.T5EncoderModel._tie_weights 复制过来
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)

    # 从 transformers.models.t5.modeling_t5.T5EncoderModel.get_encoder 复制过来
    def get_encoder(self):
        return self.encoder

    # 从 transformers.models.t5.modeling_t5.T5EncoderModel._prune_heads 复制过来
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(UMT5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    # 从 transformers.models.t5.modeling_t5.T5EncoderModel.forward 复制过来，将 T5 替换为 UMT5，google-t5/t5-small 替换为 google/umt5-small
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token ID序列，可以为空
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩，用于指示模型应该关注哪些token
        head_mask: Optional[torch.FloatTensor] = None,  # 头部遮罩，控制每个注意力头的掩盖
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 可选的嵌入输入，代替输入ID
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否以字典形式返回输出
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns:
            返回值的类型可以是包含Tensor的元组，或者BaseModelOutput对象

        Example:

        ```
        >>> from transformers import AutoTokenizer, UMT5EncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
        >>> model = UMT5EncoderModel.from_pretrained("google/umt5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 确定是否使用配置中的返回字典选项

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 使用encoder处理输入，返回编码器的输出

        return encoder_outputs
"""
UMT5 model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
tasks.
"""
# UMT5 序列分类模型，顶部有一个序列分类头部（在汇总输出之上的线性层），例如用于 GLUE 任务。
@add_start_docstrings(
    """
    UMT5 Encoder Model with a token classification head on top (a linear layer on top of the hidden-states output)
    e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    UMT5_START_DOCSTRING,
)
# UMT5 编码器模型，顶部有一个标记分类头部（在隐藏状态输出之上的线性层），例如用于命名实体识别（NER）任务。
class UMT5ForTokenClassification(UMT5PreTrainedModel):
    # Keys to ignore when loading unexpected elements during model loading
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    # Keys indicating tied weights between encoder and decoder
    _tied_weights_keys = ["transformer.encoder.embed_tokens.weight"]

    # Copied from transformers.models.t5.modeling_t5.T5ForTokenClassification.__init__ with T5->UMT5
    def __init__(self, config: UMT5Config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # UMT5 编码器模型
        self.transformer = UMT5EncoderModel(config)
        # Dropout layer
        self.dropout = nn.Dropout(config.classifier_dropout)
        # Linear layer for classification
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(UMT5_INPUTS_DOCSTRING)
"""
UMT5 model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
tasks.
"""
# UMT5 序列分类模型，顶部有一个序列分类头部（在汇总输出之上的线性层），例如用于 GLUE 任务。
@add_start_docstrings(
    """
    UMT5 model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    UMT5_START_DOCSTRING,
)
class UMT5ForSequenceClassification(UMT5PreTrainedModel):
    # Keys to ignore when loading unexpected elements during model loading
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    # Keys indicating tied weights between encoder and decoder
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # Copied from transformers.models.t5.modeling_t5.T5ForSequenceClassification.__init__ with T5->UMT5
    def __init__(self, config: UMT5Config):
        super().__init__(config)
        # UMT5 模型的变换器
        self.transformer = UMT5Model(config)
        # UMT5 模型的分类头部
        self.classification_head = UMT5ClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallelism setting
        self.model_parallel = False

    @add_start_docstrings_to_model_forward(UMT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 替换返回值文档字符串，输出类型为 Seq2SeqSequenceClassifierOutput，配置类为 _CONFIG_FOR_DOC
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
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 使用装饰器替换返回值文档字符串，指定输出类型为TokenClassifierOutput，配置类为_CONFIG_FOR_DOC
    # 从transformers.models.t5.modeling_t5.T5ForTokenClassification.forward复制而来，将T5替换为UMT5
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        标签 (`torch.LongTensor`，形状为 `(batch_size, sequence_length)`，*可选*):
            用于计算标记分类损失的标签。索引应在 `[0, ..., config.num_labels - 1]` 范围内。
        返回:
        """
        # 如果 return_dict 不为 None，则使用给定值；否则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 Transformer 模型
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取隐藏状态并进行 dropout 处理
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # 将隐藏状态传递给分类器得到 logits
        logits = self.classifier(hidden_states)

        # 如果提供了标签，则计算损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不返回字典格式的结果，则返回元组格式的输出
        if not return_dict:
            output = (logits, outputs[2:-1])  # 排除最后一个元素
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 类型的结果，包括损失、logits、隐藏状态和注意力
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 为 UMT5ForQuestionAnswering 类添加文档字符串，描述其作为 UMT5 模型的问题回答器的用途和结构
@add_start_docstrings(
    """
    UMT5 Model with a span classification head on top for extractive question-answering tasks like SQuAD (linear layers
    on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    UMT5_START_DOCSTRING,
)
class UMT5ForQuestionAnswering(UMT5PreTrainedModel):
    # 定义一个列表，包含与权重绑定相关的键
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置模型维度为配置对象中的模型维度
        self.model_dim = config.d_model

        # 创建一个共享的嵌入层，用于共享词汇表和模型维度的嵌入
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制编码器配置，设置为非解码器模式，并禁用缓存，创建编码器对象
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UMT5Stack(encoder_config, self.shared)

        # 复制解码器配置，设置为解码器模式，并创建解码器对象
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = UMT5Stack(decoder_config, self.shared)

        # 设置模型输出标签数量和一个线性层用于问题回答任务的输出
        self.num_labels = config.num_labels
        self.qa_outputs = nn.Linear(config.d_model, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 transformers 库中 T5ForQuestionAnswering 类的方法复制，返回共享的嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 从 transformers 库中 T5ForQuestionAnswering 类的方法复制，设置新的输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        # 更新共享的嵌入层
        self.shared = new_embeddings
        # 更新编码器和解码器的输入嵌入层
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # 从 transformers 库中 T5ForQuestionAnswering 类的方法复制，用于绑定权重
    def _tie_weights(self):
        # 如果配置指定要绑定词嵌入权重，则将编码器和解码器的词嵌入权重绑定到共享的嵌入层上
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 从 transformers 库中 T5ForQuestionAnswering 类的方法复制，返回编码器对象
    def get_encoder(self):
        return self.encoder

    # 从 transformers 库中 T5ForQuestionAnswering 类的方法复制，返回解码器对象
    def get_decoder(self):
        return self.decoder

    # 使用装饰器添加模型前向方法的文档字符串，描述输入和输出的结构和用途
    @add_start_docstrings_to_model_forward(UMT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法用于模型的前向传播，接受多个可选参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入序列的token IDs，可以为None
        attention_mask: Optional[torch.FloatTensor] = None,  # 输入序列的注意力掩码，可以为None
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入序列的token IDs，可以为None
        decoder_attention_mask: Optional[torch.BoolTensor] = None,  # 解码器输入序列的注意力掩码，可以为None
        head_mask: Optional[torch.FloatTensor] = None,  # 多头注意力机制的掩码，可以为None
        decoder_head_mask: Optional[torch.FloatTensor] = None,  # 解码器的多头注意力机制的掩码，可以为None
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力的多头掩码，可以为None
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 编码器的输出，可以为None
        start_positions: Optional[torch.LongTensor] = None,  # 开始位置的token ID，可以为None
        end_positions: Optional[torch.LongTensor] = None,  # 结束位置的token ID，可以为None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 嵌入的输入张量，可以为None
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器的嵌入输入张量，可以为None
        use_cache: Optional[bool] = None,  # 是否使用缓存，可以为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可以为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可以为None
        return_dict: Optional[bool] = None,  # 是否以字典的形式返回结果，可以为None
```