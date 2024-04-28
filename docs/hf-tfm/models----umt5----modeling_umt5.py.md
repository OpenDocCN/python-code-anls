# `.\transformers\models\umt5\modeling_umt5.py`

```
# 设置文件编码为 utf-8
# 版权声明
#
# 根据 Apache 许可证，除非遵守许可证，否则不得使用此文件。您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则在“AS IS”基础上分发软件，没有任何担保或条件，无论是明示的还是隐含的。
# 有限制的语言权利限制特定的语言权利，并
# 根据许可证，对于特定语言的特定权利。
""" PyTorch UMT5 model."""

# 导入必要的库
import copy
import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入必要的组件和输出类
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from .configuration_umt5 import UMT5Config

# 获取logger对象
logger = logging.get_logger(__name__)

# 用于文档的配置字符串
_CONFIG_FOR_DOC = "UMT5Config"
# 用于文档的检查点字符串
_CHECKPOINT_FOR_DOC = "google/umt5-small"

# 从 transformers.models.t5.modeling_t5.T5LayerNorm 复制并将 T5 替换为 UMT5
class UMT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        UMT5 风格的 LayerNorm 模块。没有偏置，也没有减去均值。
        """
        super().__init__()
        # 初始化权重参数
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # UMT5 使用的是只进行缩放而不进行移位的 layer_norm，也称为 RMS Layer Normalization
        # 因此方差是在没有均值的情况下计算的，也没有偏差。此外，我们希望确保对于半精度输入，在 fp32 中进行累积

        # 计算方差
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 如果权重类型在 [torch.float16, torch.bfloat16] 中，则转换为该类型
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


# 从 transformers.models.t5.modeling_t5.T5DenseActDense 复制并将 T5 替换为 UMT5
class UMT5DenseActDense(nn.Module):
    # 初始化函数，接受一个UMT5Config类型的参数config
    def __init__(self, config: UMT5Config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性变换层，输入尺寸为config.d_model，输出尺寸为config.d_ff，没有偏置
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 创建一个线性变换层，输入尺寸为config.d_ff，输出尺寸为config.d_model，没有偏置
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # 创建一个以config.dropout_rate为丢弃概率的Dropout层
        self.dropout = nn.Dropout(config.dropout_rate)
        # 根据配置选择激活函数的类型，并赋值给self.act
        self.act = ACT2FN[config.dense_act_fn]
    
    # 前向传播函数，接受输入hidden_states
    def forward(self, hidden_states):
        # 使用wi进行线性变换
        hidden_states = self.wi(hidden_states)
        # 使用激活函数act
        hidden_states = self.act(hidden_states)
        # 使用dropout进行丢弃
        hidden_states = self.dropout(hidden_states)
        # 判断wo.weight的类型是否为torch.Tensor，并且hidden_states的dtype与wo.weight的dtype不同，且wo.weight的dtype不为torch.int8
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            # 将hidden_states转换为wo.weight的dtype
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        # 使用wo进行线性变换，并返回结果
        hidden_states = self.wo(hidden_states)
        return hidden_states
# 从transformers.models.t5.modeling_t5.T5DenseGatedActDense复制过来，将T5->UMT5
class UMT5DenseGatedActDense(nn.Module):
    def __init__(self, config: UMT5Config):
        super().__init__()
        # 定义一个线性变换层，将输入维度为config.d_model的向量映射成维度为config.d_ff的向量，不使用偏置
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 定义第二个线性变换层，同样将输入维度为config.d_model的向量映射成维度为config.d_ff的向量，不使用偏置
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 定义一个线性变换层，将维度为config.d_ff的向量映射回维度为config.d_model的向量，不使用偏置
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # 定义一个dropout层，按照config.dropout_rate的概率丢弃输入中的元素
        self.dropout = nn.Dropout(config.dropout_rate)
        # 从ACT2FN字典中选取对应的激活函数
        self.act = ACT2FN[config.dense_act_fn]

    # 前向传播函数
    def forward(self, hidden_states):
        # 经过激活函数后的隐藏状态
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 经过第二个线性变换层后的隐藏状态
        hidden_linear = self.wi_1(hidden_states)
        # 将两个隐藏状态相乘
        hidden_states = hidden_gelu * hidden_linear
        # 对相乘结果进行dropout操作
        hidden_states = self.dropout(hidden_states)

        # 为了使8位量化适用于google/flan-t5-xxl，self.wo保持为float32
        # 参见https://github.com/huggingface/transformers/issues/20287
        # 也确保权重不是`int8`，以防用户强制`_keep_in_fp32_modules`为`None`
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            # 将隐藏状态转换为与self.wo权重相同的数据类型
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        # 经过wo线性变换层后的隐藏状态
        hidden_states = self.wo(hidden_states)
        return hidden_states


# 从transformers.models.t5.modeling_t5.T5LayerFF复制过来，将T5->UMT5
class UMT5LayerFF(nn.Module):
    def __init__(self, config: UMT5Config):
        super().__init__()
        # 如果is_gated_act为真，则使用UMT5DenseGatedActDense，否则使用UMT5DenseActDense
        if config.is_gated_act:
            self.DenseReluDense = UMT5DenseGatedActDense(config)
        else:
            self.DenseReluDense = UMT5DenseActDense(config)

        # 初始化UMT5LayerNorm层，传入config中的隐藏层维度和layer_norm_epsilon
        self.layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化一个dropout层，按照config.dropout_rate的概率丢弃输入中的元素
        self.dropout = nn.Dropout(config.dropout_rate)

    # 前向传播函数
    def forward(self, hidden_states):
        # 对输入的隐藏状态进行Layer Norm处理
        forwarded_states = self.layer_norm(hidden_states)
        # 经过UMT5DenseGatedActDense或UMT5DenseActDense层后的结果
        forwarded_states = self.DenseReluDense(forwarded_states)
        # 将原始输入的隐藏状态和经过dropout后的结果相加
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


# 定义UMT5Attention类
class UMT5Attention(nn.Module):
    """
    T5's attention using relative_attention_bias.
    """
    # 初始化类，接受配置和是否包含相对注意力偏置标志
    def __init__(self, config, has_relative_attention_bias=False):
        # 调用父类初始化方法
        super().__init__()
        # 设置是否为解码器的标志
        self.is_decoder = config.is_decoder
        # 设置是否包含相对注意力偏置的标志
        self.has_relative_attention_bias = has_relative_attention_bias
        # 设置相对注意力桶的数量
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        # 设置相对注意力的最大距离
        self.relative_attention_max_distance = config.relative_attention_max_distance
        # 设置模型维度
        self.d_model = config.d_model
        # 设置键值投影维度
        self.key_value_proj_dim = config.d_kv
        # 设置注意力头的数量
        self.n_heads = config.num_heads
        # 设置丢弃率
        self.dropout = config.dropout_rate
        # 计算内部维度
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 使用线性变换定义查询、键、值和输出投影层
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # 如果包含相对注意力偏置，初始化相对注意力偏置
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        # 初始化剪枝的注意力头集合
        self.pruned_heads = set()

    # 重塑投影张量的形状
    def _shape(self, projection: torch.Tensor) -> torch.Tensor:
        # 计算新的投影张量形状
        new_projection_shape = projection.size()[:-1] + (self.n_heads, self.key_value_proj_dim)
        # 移动注意力头到第二个位置 (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
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
        relative_buckets = 0  # 初始化相对位置桶变量为0
        num_buckets = self.relative_attention_num_buckets  # 从对象属性获取总桶数
        max_distance = self.relative_attention_max_distance  # 从对象属性获取最大距离限制
        if not self.is_decoder:  # 如果当前模块不是解码器
            num_buckets //= 2  # 桶数量减半，因为仅使用一半桶用于正方向或负方向
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets  # 根据位置正负分配到不同的桶区间
            relative_position = torch.abs(relative_position)  # 将位置转换为绝对值，以用于后续计算
        else:  # 如果是解码器
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))  # 调整相对位置值，保证它非负
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2  # 确切位置桶的最大数量（一半的桶）
        is_small = relative_position < max_exact  # 检查哪些位置在确切位置桶的范围内

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        log_ratio = torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact)  # 计算对数桶比例
        log_ratio = log_ratio * (num_buckets - max_exact)  # 转换比例到适合的桶索引范围
        relative_position_if_large = max_exact + log_ratio.to(torch.long)  # 计算较大距离的相对位置所在的桶索引
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )  # 限制桶索引不超过最大值

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)  # 根据大小选择适当的桶索引
        return relative_buckets  # 返回计算的桶索引
    # 计算分箱的相对位置偏置
    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        # 如果设备为None，则使用权重的设备
        if device is None:
            device = self.relative_attention_bias.weight.device
        # 创建一个包含查询长度元素的长整型张量，设备为给定设备
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # 创建一个包含键长度元素的长整型张量，设备为给定设备
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        # 计算相对位置，形状为(query_length, key_length)
        relative_position = memory_position - context_position
        # 计算相对位置所属的桶，根据相对位置算出桶的索引
        relative_position_bucket = self._relative_position_bucket(relative_position)
        # 通过相对位置桶计算相对注意力偏置，形状为(query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # 转置values维度，形状变为(1, num_heads, query_length, key_length)，在最前面添加一个维度
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
# 定义 UMT5 自注意力层的模块
class UMT5LayerSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 UMT5 自注意力层对象，包含相对注意力偏置
        self.SelfAttention = UMT5Attention(config, has_relative_attention_bias=True)
        # 创建层归一化对象
        self.layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 创建 Dropout 层对象
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        past_key_value=None,
    ):
        # 对隐藏状态进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 通过自注意力层进行前向传播
        attention_output = self.SelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
        )
        # 将自注意力层的输出与原始隐藏状态进行残差连接，并应用 Dropout
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 构建输出元组，包含残差连接后的隐藏状态和可能的注意力信息
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# 定义 UMT5 跨注意力层的模块
class UMT5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 UMT5 跨注意力层对象，不包含相对注意力偏置
        self.EncDecAttention = UMT5Attention(config, has_relative_attention_bias=False)
        # 创建层归一化对象
        self.layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 创建 Dropout 层对象
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        layer_head_mask=None,
        past_key_value=None,
    ):
        # 对隐藏状态进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 通过跨注意力层进行前向传播
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
        )
        # 将跨注意力层的输出与原始隐藏状态进行残差连接，并应用 Dropout
        layer_output = hidden_states + self.dropout(attention_output[0])
        # 构建输出元组，包含残差连接后的隐藏状态和可能的注意力信息
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


# 定义 UMT5 块的模块
class UMT5Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 判断是否为解码器块
        self.is_decoder = config.is_decoder
        # 创建模块列表
        self.layer = nn.ModuleList()
        # 添加自注意力层到模块列表
        self.layer.append(UMT5LayerSelfAttention(config))
        # 如果是解码器块，添加跨注意力层到模块列表
        if self.is_decoder:
            self.layer.append(UMT5LayerCrossAttention(config))
        # 添加前馈网络层到模块列表
        self.layer.append(UMT5LayerFF(config))

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
    # 该函数是 Transformer 解码器层的前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        output_attentions=False,
        ):
        # 自注意力层的缓存的key和value元组
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    
        # 执行自注意力层的前向传播, 输出为隐藏状态、自注意力权重、当前key和value
        hidden_states, self_attn_weights, present_key_value = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
        )
    
        # 对隐藏状态进行截断, 避免溢出, 仅在float16精度下使用
        if hidden_states.dtype == torch.float16:
            max_dtype = torch.finfo(hidden_states.dtype).max
            clamp_value = torch.where(torch.isinf(hidden_states).any(), max_dtype - 1000, max_dtype)
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    
        # 跨注意力层的缓存的key和value元组
        cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
    
        # 仅在解码器且有encoder hidden states时执行跨注意力层
        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # 执行跨注意力层的前向传播, 输出为隐藏状态、跨注意力权重、当前key和value
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.layer[1](
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
            )
            # 对隐藏状态进行截断, 避免溢出, 仅在float16精度下使用
            if hidden_states.dtype == torch.float16:
                max_dtype = torch.finfo(hidden_states.dtype).max
                clamp_value = torch.where(torch.isinf(hidden_states).any(), max_dtype - 1000, max_dtype)
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
            # 更新当前的key和value
            present_key_value += cross_attn_present_key_value
    
        # 执行前馈网络层
        hidden_states = self.layer[-1](hidden_states)
    
        # 对隐藏状态进行截断, 避免溢出, 仅在float16精度下使用
        if hidden_states.dtype == torch.float16:
            max_dtype = torch.finfo(hidden_states.dtype).max
            clamp_value = torch.where(torch.isinf(hidden_states).any(), max_dtype - 1000, max_dtype)
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    
        # 返回结果
        outputs = (
            hidden_states,
            present_key_value,
        )
    
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
    
        return outputs
# 这是一个 UMT5 分类头的实现，用于句子级别的分类任务
class UMT5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: UMT5Config):
        super().__init__()
        # 初始化一个全连接层，输入维度为 config.d_model，输出维度也为 config.d_model
        self.dense = nn.Linear(config.d_model, config.d_model)
        # 初始化一个 dropout 层，丢弃率为 config.classifier_dropout
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        # 初始化一个全连接层，输入维度为 config.d_model，输出维度为 config.num_labels
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对输入的隐藏状态应用 dropout
        hidden_states = self.dropout(hidden_states)
        # 将隐藏状态通过第一个全连接层
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出应用 tanh 激活函数
        hidden_states = torch.tanh(hidden_states)
        # 对激活后的隐藏状态再次应用 dropout
        hidden_states = self.dropout(hidden_states)
        # 将最终的隐藏状态通过输出全连接层得到分类结果
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


# 这是 UMT5 预训练模型的抽象基类
class UMT5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = UMT5Config
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["UMT5Block"]
    _keep_in_fp32_modules = ["wo"]

    @property
    def dummy_inputs(self):
        # 定义一些dummy输入用于模型测试
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _shift_right(self, input_ids):
        # 将输入的 token ID 向右移动一位，并用 decoder_start_token_id 填充第一个位置
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In UMT5 it is usually set to the pad_token_id. "
                "See UMT5 docs for more information."
            )

        if is_torch_fx_proxy(input_ids):
            # 对于 torch.fx 代理对象，使用 torch.full 和 torch.cat 进行操作
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            # 对于普通的 tensor，使用 new_zeros 和 masked_fill 进行操作
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # 将 -100 值替换为 pad_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


# UMT5 模型的主体实现
class UMT5Stack(UMT5PreTrainedModel):
    # 初始化方法，接受配置和嵌入标记作为参数
    def __init__(self, config, embed_tokens=None):
        # 调用父类的初始化方法
        super().__init__(config)
        # 初始化嵌入标记
        self.embed_tokens = embed_tokens
        # 判断是否为解码器
        self.is_decoder = config.is_decoder
        # 创建包含多个UMT5Block对象的列表
        self.block = nn.ModuleList([UMT5Block(config) for i in range(config.num_layers)])
        # 创建最终层的 LayerNorm
        self.final_layer_norm = UMT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 创建丢弃层
        self.dropout = nn.Dropout(config.dropout_rate)

        # 初始化权重并应用最终处理
        self.gradient_checkpointing = False
        # 调用后期初始化方法
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    # 前向传播方法
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
# UMT5_START_DOCSTRING 的值是 UMT5 模型的文档字符串，包含了模型的提出背景、论文引用、继承关系、参数说明等信息
UMT5_START_DOCSTRING = r"""

    The UMT5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
    Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan
    Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a
    text-to-text denoising generative setting.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`UMT5Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# UMT5_INPUTS_DOCSTRING 和 UMT5_ENCODER_INPUTS_DOCSTRING 的值为空，可能用于存储模型输入的文档字符串，暂未填充内容
UMT5_INPUTS_DOCSTRING = r"""
"""

UMT5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。UMT5 是一个具有相对位置嵌入的模型，因此您应该能够在左右两侧对输入进行填充。

            # 可以使用 [`AutoTokenizer`] 获取索引。参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 获取详细信息。

            # 若要了解如何为预训练准备`input_ids`，请参阅 [UMT5 Training](./umt5#training)。
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。掩码值在 `[0, 1]` 中选择：

            # - 1 表示**未掩码**的标记，
            # - 0 表示**掩码**的标记。

            # [什么是注意力掩码?](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于抵消自注意力模块选择性头部的掩码。掩码值在 `[0, 1]` 中选择：

            # - 1 表示**未掩码**的头部，
            # - 0 表示**掩码**的头部。

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            # 可选择直接传递嵌入表示而不是传递`input_ids`。如果您想更好地控制如何将`input_ids`索引转换为关联向量，这将非常有用，而不是模型的内部嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            # 是否返回所有关注层的注意力张量。有关更多详细信息，请参见返回张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回一个[`~utils.ModelOutput`]而不是一个普通的元组。
# 导入必要的包
from transformers.modeling_t5 import T5PreTrainedModel, T5Stack, T5Config
from transformers.file_utils import add_start_docstrings
import copy
import torch.nn as nn

# 定义 UMT5 模型
@add_start_docstrings(
    "The bare UMT5 Model transformer outputting raw hidden-states without any specific head on top.",
    UMT5_START_DOCSTRING,
)
class UMT5Model(UMT5PreTrainedModel):
    r"""
    Examples:

    ```python
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

    初始化 UMT5Model 类
    """
    # 初始化 UMT5Model 类的必要属性
    model_type = "uumt5"
    config_class = UMT5Config
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # 初始化 UMT5Model 类
    def __init__(self, config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制配置并设置编码器
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UMT5Stack(encoder_config, self.shared)

        # 复制配置并设置解码器
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = UMT5Stack(decoder_config, self.shared)

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 剪枝头信息
    # 对模型的注意力头进行修剪
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要修剪的层和对应需要修剪的注意力头
        for layer, heads in heads_to_prune.items():
            # 在编码器中修剪对应层的注意力头
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 重写父类函数的注释并添加到模型的向前传播函数
    @add_start_docstrings_to_model_forward(UMT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        # 输入编码器的标记序列
        input_ids: Optional[torch.LongTensor] = None,
        # 编码器的注意力蒙版
        attention_mask: Optional[torch.FloatTensor] = None,
        # 解码器的输入标记序列
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器的注意力蒙版
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        # 编码器的注意力头蒙版
        head_mask: Optional[torch.FloatTensor] = None,
        # 解码器的注意力头蒙版
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        # 交叉注意力头的蒙版
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 过去的键值对
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 编码器的嵌入输入
        inputs_embeds: Optional[torch.Tensor] = None,
        # 解码器的嵌入输入
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        # 是否使用缓存
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的结果
        return_dict: Optional[bool] = None,
# 为 UMT5 模型添加文档字符串，说明其具有 language modeling head 功能
@add_start_docstrings("""UMT5 Model with a `language modeling` head on top.""", UMT5_START_DOCSTRING)
class UMT5ForConditionalGeneration(UMT5PreTrainedModel):
    r"""
    Examples:

    ```python
    >>> from transformers import UMT5ForConditionalGeneration, AutoTokenizer

    >>> model = UMT5ForConditionalGeneration.from_pretrained("google/umt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, text_target=summary, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> loss = outputs.loss
    ```"""

    # 设置模型类型为 "umt5"
    model_type = "umt5"
    # 设置共享权重的键
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化函数
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__(config)
        # 设置模型维度为 config 中的 d_model
        self.model_dim = config.d_model

        # 创建共享权重的嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制 config 创建编码器配置，并设置为非解码器
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建编码器
        self.encoder = UMT5Stack(encoder_config, self.shared)

        # 复制 config 创建解码器配置，并设置为解码器
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 创建解码器
        self.decoder = UMT5Stack(decoder_config, self.shared)

        # 创建语言模型头
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 获取编码器
    def get_encoder(self):
        return self.encoder
    # 从transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_decoder中复制方法
    def get_decoder(self):
        # 返回decoder对象
        return self.decoder
    
    # 添加model forward的文档字符串到前向传播
    # 替换返回文档字符串为Seq2SeqLMOutput类型，配置类为_CONFIG_FOR_DOC
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
    
    # 从transformers.models.t5.modeling_t5.T5ForConditionalGeneration.prepare_inputs_for_generation中复制方法
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
        # 如果past_key_values不为空，则剪切decoder_input_ids
        if past_key_values is not None:
            # 获取过去键值的长度
            past_length = past_key_values[0][0].shape[2]
    
            # 一些生成方法已经只传递了最后一个输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认使用旧的行为：只保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1
    
            # 将decoder_input_ids截取为剩余部分
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
    
    # 从transformers.models.t5.modeling_t5.T5ForConditionalGeneration.prepare_decoder_input_ids_from_labels中复制方法
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 调用_shift_right方法，将标签向右移动
        return self._shift_right(labels)
    
    @staticmethod
    # 重新排序缓存数据，以适应beam搜索
    def _reorder_cache(past_key_values, beam_idx):
        # 初始化一个空的重新排序后的缓存
        reordered_past = ()
        # 遍历过去键和值的列表
        for layer_past in past_key_values:
            # 对每一层的过去状态进行重新排序，并添加到重新排序后的缓存中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的缓存
        return reordered_past
# 添加起始文档字符串，描述UMT5模型的编码器的输出，没有特定的头部
# 引入UMT5_START_DOCSTRING
class UMT5EncoderModel(UMT5PreTrainedModel):
    r"""
    Examples:

    ```python
    >>> from transformers import UMT5EncoderModel, AutoTokenizer

    >>> model = UMT5EncoderModel.from_pretrained("google/umt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> input_ids = tokenizer(article, return_tensors="pt").input_ids
    >>> outputs = model(input_ids)
    >>> hidden_state = outputs.last_hidden_state
    ```"""

    # UMT5模型类型为"umt5"
    model_type = "umt5"
    # 定义用于共享权重的键
    _tied_weights_keys = ["encoder.embed_tokens.weight"]

    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 定义共享的嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制配置并进行一些调整
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 初始化UMT5编码器堆栈
        self.encoder = UMT5Stack(encoder_config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从transformers.models.t5.modeling_t5.T5EncoderModel中复制的函数
    def get_input_embeddings(self):
        return self.shared

    # 从transformers.models.t5.modeling_t5.T5EncoderModel中复制的函数
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    # 从transformers.models.t5.modeling_t5.T5EncoderModel中复制的函数
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)

    # 从transformers.models.t5.modeling_t5.T5EncoderModel中复制的函数
    def get_encoder(self):
        return self.encoder

    # 从transformers.models.t5.modeling_t5.T5EncoderModel中复制的函数
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(UMT5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    # 从transformers.models.t5.modeling_t5.T5EncoderModel.forward中复制的函数，将T5替换成UMT5，t5-small替换成google/umt5-small
    # 此方法用于模型的前向传播，接受输入参数并返回模型输出
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，类型为可选的长整型张量，默认为 None
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩张量，类型为可选的浮点数张量，默认为 None
        head_mask: Optional[torch.FloatTensor] = None,  # 头部遮罩张量，类型为可选的浮点数张量，默认为 None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入张量，类型为可选的浮点数张量，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力张量，类型为可选的布尔值，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态张量，类型为可选的布尔值，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，类型为可选的布尔值，默认为 None
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, UMT5EncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("google/umt5-small")
        >>> model = UMT5EncoderModel.from_pretrained("google/umt5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        # 如果 return_dict 不为 None，则使用参数中的值；否则，使用模型配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入参数传递给编码器模块，获取编码器的输出
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 返回编码器的输出
        return encoder_outputs
# 定义一个 UMT5 模型，用于序列分类任务，包含一个线性层用于汇总输出，在 GLUE 等任务中应用
@add_start_docstrings(
    """
    UMT5 model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    UMT5_START_DOCSTRING,
)
class UMT5ForSequenceClassification(UMT5PreTrainedModel):
    # 在加载过程中忽略的键值对应的键列表
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    # 共享权重的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # 根据配置初始化 UMT5ForSequenceClassification 类
    # 从 transformers.models.t5.modeling_t5.T5ForSequenceClassification.__init__ 复制并替换 T5 为 UMT5
    def __init__(self, config: UMT5Config):
        super().__init__(config)
        # 初始化 UMT5 模型和分类头
        self.transformer = UMT5Model(config)
        self.classification_head = UMT5ClassificationHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

        # 设置模型并行处理为 False
        self.model_parallel = False

    # 在前向传播时添加注释描述
    @add_start_docstrings_to_model_forward(UMT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        # 输入的 token ID
        input_ids: torch.LongTensor = None,
        # 注意力遮盖
        attention_mask: Optional[torch.Tensor] = None,
        # 解码器输入的 token ID
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器的注意力遮盖
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        # 头部遮盖
        head_mask: Optional[torch.Tensor] = None,
        # 解码器头部遮盖
        decoder_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力头部遮盖
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        # 嵌入的输入
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 解码器嵌入的输入
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # 标签
        labels: Optional[torch.LongTensor] = None,
        # 是否使用缓存
        use_cache: Optional[bool] = None,
        # 输出注意力
        output_attentions: Optional[bool] = None,
        # 输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 返回的字典类型
        return_dict: Optional[bool] = None,



# 定义一个 UMT5 模型，用于问答任务，包含���个线性层用于计算 `span start logits` 和 `span end logits` 的输出，在类似 SQuAD 的任务中应用
@add_start_docstrings(
    """
    UMT5 Model with a span classification head on top for extractive question-answering tasks like SQuAD (linear layers
    on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    UMT5_START_DOCSTRING,
)
class UMT5ForQuestionAnswering(UMT5PreTrainedModel):
    # 共享权重的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    # 初始化函数，接受配置参数，并设置模型维度
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 设置模型维度为配置参数中的 d_model
        self.model_dim = config.d_model

        # 创建一个共享的嵌入层，词汇量为配置参数中的 vocab_size，维度为 d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制配置参数，用于生成编码器实例的配置
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建编码器实例
        self.encoder = UMT5Stack(encoder_config, self.shared)

        # 复制配置参数，用于生成解码器实例的配置
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 创建解码器实例
        self.decoder = UMT5Stack(decoder_config, self.shared)

        # 设置标签数量为配置参数中的 num_labels
        self.num_labels = config.num_labels
        # 创建一个线性层，输入维度为 d_model，输出维度为 num_labels
        self.qa_outputs = nn.Linear(config.d_model, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入层的函数
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层的函数
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        # 设置编码器和解码器的输入嵌入层
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # 权重共享函数
    def _tie_weights(self):
        # 如果配置参数中设置了词嵌入共享
        if self.config.tie_word_embeddings:
            # 将编码器和解码器的嵌入层权重进行共享或克隆
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 获取编码器实例的函数
    def get_encoder(self):
        return self.encoder

    # 获取解码器实例的函数
    def get_decoder(self):
        return self.decoder

    # 根据模型前向推理的文档规范添加文档字符串注释，使用 Seq2SeqQuestionAnsweringModelOutput 进行替换
    @add_start_docstrings_to_model_forward(UMT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    # 这个函数是模型的前向传播函数
    def forward(
        self,
        # 输入序列的 ID，形状为 (batch_size, sequence_length)
        input_ids: Optional[torch.LongTensor] = None,
        # 输入序列的注意力掩码，形状为 (batch_size, sequence_length)
        attention_mask: Optional[torch.FloatTensor] = None,
        # 解码器的输入序列 ID，形状为 (batch_size, sequence_length)
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器的注意力掩码，形状为 (batch_size, sequence_length)
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        # 编码器注意力头的掩码，形状为 (num_heads, sequence_length, sequence_length)
        head_mask: Optional[torch.FloatTensor] = None,
        # 解码器注意力头的掩码，形状为 (num_heads, sequence_length, sequence_length)
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        # 跨注意力头的掩码，形状为 (num_heads, sequence_length, sequence_length)
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器的输出，一个元组的元组，包含隐藏状态和注意力权重
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        # 问答任务的起始位置，形状为 (batch_size,)
        start_positions: Optional[torch.LongTensor] = None,
        # 问答任务的结束位置，形状为 (batch_size,)
        end_positions: Optional[torch.LongTensor] = None,
        # 输入的 embedding 表示，形状为 (batch_size, sequence_length, embedding_dim)
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 解码器的 embedding 表示，形状为 (batch_size, sequence_length, embedding_dim)
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        # 是否使用缓存
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回 dict 格式的输出
        return_dict: Optional[bool] = None,
    ):
```