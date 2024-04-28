# `.\transformers\models\pop2piano\modeling_pop2piano.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 2.0 许可证，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按照“原样”分发软件
# 没有任何形式的担保或条件，无论是明示的还是隐含的
# 有关特定语言的权限和限制，请参阅许可证
# PyTorch Pop2Piano 模型

# 导入库
import copy
import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.generation import GenerationConfig

# 导入内部模块
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
# 导入 Pop2Piano 配置
from .configuration_pop2piano import Pop2PianoConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 初始化 Pop2PianoLayerNorm 模块加载标志
_load_pop2piano_layer_norm = True

# 尝试导入 apex.normalization 模块
try:
    from apex.normalization import FusedRMSNorm
    # 若成功导入，设置 Pop2PianoLayerNorm 模块加载标志为 False
    _load_pop2piano_layer_norm = False
    # 打印日志信息
    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of Pop2PianoLayerNorm")
except ImportError:
    # 若导入失败，继续使用普通的 Pop2PianoLayerNorm
    pass
except Exception:
    # 若导入出现异常，记录日志信息
    logger.warning("Discovered apex but it failed to load, falling back to Pop2PianoLayerNorm")
    pass

# 定义文档注释所需的配置和检查点
_CONFIG_FOR_DOC = "Pop2PianoConfig"
_CHECKPOINT_FOR_DOC = "sweetcocoa/pop2piano"

# Pop2Piano 预训练模型列表
POP2PIANO_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "sweetcocoa/pop2piano",
    # 查看所有 Pop2Piano 模型：https://huggingface.co/models?filter=pop2piano
]

# Pop2Piano 输入文档字符串
POP2PIANO_INPUTS_DOCSTRING = r"""
"""

# 以下代码是从 transformers.models.t5.modeling_t5.T5LayerNorm 复制并修改为 Pop2Piano
# Pop2Piano 的层归一化模块
class Pop2PianoLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        构建 Pop2Piano 风格的 layernorm 模块，无偏置和均值减法。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    # 前向传播函数，用于 Pop2Piano 模型
    def forward(self, hidden_states):
        # Pop2Piano 使用一个仅进行缩放而不进行偏移的层归一化，也称为均方根层归一化
        # https://arxiv.org/abs/1910.07467，因此方差是在不带均值的情况下计算的，
        # 并且没有偏差。此外，我们希望确保对于半精度输入，累积是以 fp32 进行的。

        # 计算方差，将隐藏状态转换为 float32，求平方，沿着最后一个维度计算平均值，保持维度不变
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # 使用均方根层归一化调整隐藏状态
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 如果权重的数据类型是 torch.float16 或 torch.bfloat16，则将隐藏状态转换为半精度
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        # 返回调整后的隐藏状态乘以权重
        return self.weight * hidden_states
# 如果未加载 Pop2PianoLayerNorm，则将 FusedRMSNorm 作为 Pop2PianoLayerNorm
if not _load_pop2piano_layer_norm:
    Pop2PianoLayerNorm = FusedRMSNorm  # noqa

# 将 Pop2PianoLayerNorm 添加到 ALL_LAYERNORM_LAYERS 列表
ALL_LAYERNORM_LAYERS.append(Pop2PianoLayerNorm)


# 复制自 transformers.models.t5.modeling_t5.T5DenseActDense，将 T5 替换为 Pop2Piano，t5 替换为 pop2piano
class Pop2PianoDenseActDense(nn.Module):
    def __init__(self, config: Pop2PianoConfig):
        super().__init__()
        # 创建两个线性层，分别用于输入和输出
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # 创建 dropout 层
        self.dropout = nn.Dropout(config.dropout_rate)
        # 根据配置创建激活函数
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        # 使用输入线性层进行线性变换
        hidden_states = self.wi(hidden_states)
        # 应用激活函数
        hidden_states = self.act(hidden_states)
        # 应用 dropout
        hidden_states = self.dropout(hidden_states)
        # 如果输出线性层权重数据类型与输入不同，并且不是 int8，则转换输入数据类型
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        # 使用输出线性层进行线性变换
        hidden_states = self.wo(hidden_states)
        return hidden_states


# 复制自 transformers.models.t5.modeling_t5.T5DenseGatedActDense，将 T5 替换为 Pop2Piano
class Pop2PianoDenseGatedActDense(nn.Module):
    def __init__(self, config: Pop2PianoConfig):
        super().__init__()
        # 创建三个线性层，分别用于两个输入和一个输出
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # 创建 dropout 层
        self.dropout = nn.Dropout(config.dropout_rate)
        # 根据配置创建激活函数
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        # 使用第一个输入线性层进行线性变换，并应用激活函数
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 使用第二个输入线性层进行线性变换
        hidden_linear = self.wi_1(hidden_states)
        # 将两个变换结果相乘
        hidden_states = hidden_gelu * hidden_linear
        # 应用 dropout
        hidden_states = self.dropout(hidden_states)

        # 如果输出线性层权重数据类型与输入不同，并且不是 int8，则转换输入数据类型
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        # 使用输出线性层进行线性变换
        hidden_states = self.wo(hidden_states)
        return hidden_states


# 复制自 transformers.models.t5.modeling_t5.T5LayerFF，将 T5 替换为 Pop2Piano
class Pop2PianoLayerFF(nn.Module):
    # 初始化方法，接收一个 Pop2PianoConfig 对象作为参数
    def __init__(self, config: Pop2PianoConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 根据配置选择是否使用门控激活函数创建 DenseReluDense 对象
        if config.is_gated_act:
            self.DenseReluDense = Pop2PianoDenseGatedActDense(config)
        # 否则创建普通激活函数的 DenseReluDense 对象
        else:
            self.DenseReluDense = Pop2PianoDenseActDense(config)
    
        # 创建一个 LayerNorm 层，参数为配置中的 d_model 和 layer_norm_epsilon
        self.layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 创建一个 Dropout 层，参数为配置中的 dropout_rate
        self.dropout = nn.Dropout(config.dropout_rate)
    
    # 前向传播方法，接收 hidden_states 作为输入
    def forward(self, hidden_states):
        # 对输入应用 LayerNorm 层
        forwarded_states = self.layer_norm(hidden_states)
        # 将应用 LayerNorm 后的状态输入 DenseReluDense 层
        forwarded_states = self.DenseReluDense(forwarded_states)
        # 使用 Dropout 层对输出进行dropout，并与输入相加
        hidden_states = hidden_states + self.dropout(forwarded_states)
        # 返回处理后的 hidden_states
        return hidden_states
# 从transformers.models.pop2piano.modeling_pop2piano.Pop2PianoAttention复制而来，用于Pop2Piano模型中的注意力机制
class Pop2PianoAttention(nn.Module):
    def __init__(self, config: Pop2PianoConfig, has_relative_attention_bias=False):
        # 初始化函数，接受Pop2PianoConfig配置和是否有相对注意力偏置的参数
        super().__init__()
        # 设置属性
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 使用Mesh TensorFlow初始化，避免在softmax之前进行缩放
        # 创建线性变换层
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            # 如果存在相对注意力偏置，创建嵌入层
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        # 剪枝头部
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # 剪枝线性层
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # 更新超参数
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    # 根据相对位置计算相对位置bucket编号的函数
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        # 这个函数是从Mesh Tensorflow项目中改编而来的
        # 它的作用是将相对位置转换成一个bucket编号，用于相对注意力计算
        # 相对位置被定义为memory_position - query_position，即查询位置到被关注位置的距离
        # 如果bidirectional为False，则正相对位置是无效的
        # 我们使用较小的bucket来表示小的绝对相对位置，较大的bucket来表示较大的绝对相对位置
        # 所有大于等于max_distance的相对位置都映射到同一个bucket
        # 所有小于等于-max_distance的相对位置都映射到同一个bucket
        # 这样可以更好地推广到比训练集更长的序列
    
        参数:
            relative_position: 一个int32 Tensor
            bidirectional: 一个布尔值，表示注意力是否双向
            num_buckets: 一个整数，表示bucket的数量
            max_distance: 一个整数，表示最大距离
    
        返回:
            一个与relative_position形状相同的Tensor，包含[0, num_buckets)范围内的int32值
        """
        relative_buckets = 0
        if bidirectional:
            # 如果是双向的，将bucket数量减半，正负位置分别计算
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            # 如果是单向的，将负相对位置映射到0及以上
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        
        # 现在relative_position的范围是[0, inf)
    
        # 一半的bucket用于精确的位置增量
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
    
        # 另一半bucket用于logarithmic bins，最大到max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
    
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        # 如果设备未指定，则使用相对注意力偏置张量的设备
        if device is None:
            device = self.relative_attention_bias.weight.device
        # 创建一个表示上下文位置的张量，其长度为查询长度
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # 创建一个表示记忆位置的张量，其长度为键长度
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        # 计算相对位置，形状为 (查询长度, 键长度)
        relative_position = memory_position - context_position
        # 将相对位置映射到桶中，考虑是否双向以及桶的数量和最大距离
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # 形状为 (查询长度, 键长度)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # 使用相对位置桶张量获取相对注意力偏置值，形状为 (查询长度, 键长度, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # 转置并添加批次维度，形状为 (1, num_heads, 查询长度, 键长度)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
# 重新定义 Pop2PianoLayerSelfAttention 类，继承自 nn.Module
# 这个类是从 transformers.models.t5.modeling_t5.T5LayerSelfAttention 复制过来的，将其中的 T5 变成 Pop2Piano
class Pop2PianoLayerSelfAttention(nn.Module):
    # 初始化函数，接受 config 和 has_relative_attention_bias 参数
    def __init__(self, config, has_relative_attention_bias=False):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个 Pop2PianoAttention 对象，传入 config 和 has_relative_attention_bias 参数
        self.SelfAttention = Pop2PianoAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        # 创建一个 Pop2PianoLayerNorm 对象，传入 config.d_model 和 config.layer_norm_epsilon 参数
        self.layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 创建一个 nn.Dropout 对象，传入 config.dropout_rate 参数
        self.dropout = nn.Dropout(config.dropout_rate)

    # 前向传播函数，接受多个参数
    def forward(
        self,
        hidden_states,  # 输入的隐藏状态
        attention_mask=None,  # 注意力掩码
        position_bias=None,  # 位置偏置
        layer_head_mask=None,  # 层和头的掩码
        past_key_value=None,  # 过去的键值对
        use_cache=False,  # 是否使用缓存
        output_attentions=False,  # 是否输出注意力权重
    ):
        # 对隐藏状态进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用 Pop2PianoAttention 对象进行自注意力计算，传入相应参数
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 将自注意力计算结果与原始隐藏状态相加并加上 dropout
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 将结果打包返回，包括 hidden_states 和 attention_output 中的其它元素（如果有的话）
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        # 返回结果
        return outputs


# 重新定义 Pop2PianoLayerCrossAttention 类，继承自 nn.Module
# 这个类是从 transformers.models.t5.modeling_t5.T5LayerCrossAttention 复制过来的，将其中的 T5 变成 Pop2Piano
class Pop2PianoLayerCrossAttention(nn.Module):
    # 初始化函数，接受 config 参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个 Pop2PianoAttention 对象，传入 config 和 has_relative_attention_bias 参数
        self.EncDecAttention = Pop2PianoAttention(config, has_relative_attention_bias=False)
        # 创建一个 Pop2PianoLayerNorm 对象，传入 config.d_model 和 config.layer_norm_epsilon 参数
        self.layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 创建一个 nn.Dropout 对象，传入 config.dropout_rate 参数
        self.dropout = nn.Dropout(config.dropout_rate)

    # 前向传播函数，接受多个参数
    def forward(
        self,
        hidden_states,  # 输入的隐藏状态
        key_value_states,  # 键值状态
        attention_mask=None,  # 注意力掩码
        position_bias=None,  # 位置偏置
        layer_head_mask=None,  # 层和头的掩码
        past_key_value=None,  # 过去的键值对
        use_cache=False,  # 是否使用缓存
        query_length=None,  # 查询长度
        output_attentions=False,  # 是否输出注意力权重
    ):
        # 对隐藏状态进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用 Pop2PianoAttention 对象进行编码-解码注意力计算，传入相应参数
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        # 将编码-解码注意力计算结果与原始隐藏状态相加并加上 dropout
        layer_output = hidden_states + self.dropout(attention_output[0])
        # 将结果打包返回，包括 layer_output 和 attention_output 中的其它元素（如果有的话）
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        # 返回结果
        return outputs


# 重新定义 Pop2PianoBlock 类，继承自 nn.Module
# 这个类是从 transformers.models.t5.modeling_t5.T5Block 复制过来的，将其中的 T5 变成 Pop2Piano
class Pop2PianoBlock(nn.Module):
    # 定义一个 Pop2PianoLayer 类，继承自 nn.Module
    def __init__(self, config, has_relative_attention_bias=False):
        # 调用父类的 __init__ 方法
        super().__init__()
        # 判断是否为解码器层
        self.is_decoder = config.is_decoder
        # 创建一个 ModuleList 列表存储多个子层
        self.layer = nn.ModuleList()
        # 添加一个自注意力层
        self.layer.append(Pop2PianoLayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        # 如果是解码器层，则添加一个交叉注意力层
        if self.is_decoder:
            self.layer.append(Pop2PianoLayerCrossAttention(config))
        # 添加一个前馈神经网络层
        self.layer.append(Pop2PianoLayerFF(config))
    
    # 定义前向传播方法
    def forward(
        self,
        hidden_states,  # 输入的隐藏状态
        attention_mask=None,  # 注意力掩码
        position_bias=None,  # 位置偏差
        encoder_hidden_states=None,  # 编码器输出的隐藏状态
        encoder_attention_mask=None,  # 编码器的注意力掩码
        encoder_decoder_position_bias=None,  # 编码器-解码器的位置偏差
        layer_head_mask=None,  # 层头掩码
        cross_attn_layer_head_mask=None,  # 交叉注意力层头掩码
        past_key_value=None,  # 上一时间步的键值对
        use_cache=False,  # 是否使用缓存
        output_attentions=False,  # 是否输出注意力权重
        return_dict=True  # 是否以字典的形式返回结果
    ):
        # 执行前向传播逻辑
        pass
# 定义一个名为 Pop2PianoPreTrainedModel 的类，继承自 PreTrainedModel 类
class Pop2PianoPreTrainedModel(PreTrainedModel):
    """
    一个用于处理权重初始化和提供下载和加载预训练模型的简单接口的抽象类。
    """

    # 指定配置类为 Pop2PianoConfig
    config_class = Pop2PianoConfig
    # 定义基础模型前缀为 "transformer"
    base_model_prefix = "transformer"
    # 指定模型不可并行化
    is_parallelizable = False
    # 指定模型支持梯度检查点
    supports_gradient_checkpointing = True
    # 指定不进行拆分的模块列表
    _no_split_modules = ["Pop2PianoBlock"]
    # 指定保持在 FP32 的模块列表
    _keep_in_fp32_modules = ["wo"]

    # 定义一个名为 _shift_right 的方法，用于将输入 ID 向右移动
    def _shift_right(self, input_ids):
        # 获取配置中的解码器起始 token ID
        decoder_start_token_id = self.config.decoder_start_token_id
        # 获取配置中的填充 token ID
        pad_token_id = self.config.pad_token_id

        # 如果解码器起始 token ID 未定义，抛出错误
        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id 必须被定义。 在 Pop2Piano 中通常设置为 pad_token_id。"
            )

        # 将输入 ID 向右移动
        if is_torch_fx_proxy(input_ids):
            # 对于代理对象，不支持原生的赋值操作
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            # 使用起始 token ID 和输入 ID 的切片拼接
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            # 创建新的移位输入 ID 张量
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            # 将输入 ID 切片移动到新的位置
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            # 将起始位置设为解码器起始 token ID
            shifted_input_ids[..., 0] = decoder_start_token_id

        # 如果填充 token ID 未定义，抛出错误
        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id 必须被定义。")
        # 将标签中可能的 -100 值替换为填充 token ID
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        # 返回向右移动的输入 ID
        return shifted_input_ids


# 定义一个名为 Pop2PianoStack 的类，继承自 Pop2PianoPreTrainedModel 类
class Pop2PianoStack(Pop2PianoPreTrainedModel):
    # 从 transformers.models.t5.modeling_t5.T5Stack.__init__ 复制，修改 T5 为 Pop2Piano，t5 为 pop2piano
    def __init__(self, config, embed_tokens=None):
        # 调用父类的初始化方法
        super().__init__(config)

        # 保存嵌入 tokens
        self.embed_tokens = embed_tokens
        # 指定模型是解码器还是编码器
        self.is_decoder = config.is_decoder

        # 创建 Pop2PianoBlock 的模块列表，包含指定数量的层
        self.block = nn.ModuleList(
            [Pop2PianoBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        # 创建最终层归一化层
        self.final_layer_norm = Pop2PianoLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 创建 dropout 层
        self.dropout = nn.Dropout(config.dropout_rate)

        # 初始化权重并应用最终处理
        self.post_init()
        # 指定模型并行性
        self.model_parallel = False
        # 保存设备映射
        self.device_map = None
        # 指定模型是否使用梯度检查点
        self.gradient_checkpointing = False

    # 从 transformers.models.t5.modeling_t5.T5Stack.get_input_embeddings 复制
    def get_input_embeddings(self):
        # 返回嵌入 tokens
        return self.embed_tokens

    # 从 transformers.models.t5.modeling_t5.T5Stack.set_input_embeddings 复制
    def set_input_embeddings(self, new_embeddings):
        # 设置新的嵌入 tokens
        self.embed_tokens = new_embeddings
    # 定义模型前向传播的方法
    def forward(
        self,
        # 输入序列的 token ID
        input_ids=None,
        # 输入序列的注意力掩码
        attention_mask=None,
        # 编码器隐藏状态
        encoder_hidden_states=None,
        # 编码器的注意力掩码
        encoder_attention_mask=None,
        # 输入序列的 embeddings
        inputs_embeds=None,
        # 头部掩码
        head_mask=None,
        # 交叉注意力头部掩码
        cross_attn_head_mask=None,
        # 过去的关键值对
        past_key_values=None,
        # 是否使用缓存
        use_cache=None,
        # 是否输出注意力权重
        output_attentions=None,
        # 是否输出所有隐藏状态
        output_hidden_states=None,
        # 是否返回字典格式
        return_dict=None,
    ):
class Pop2PianoConcatEmbeddingToMel(nn.Module):
    """将 `composer` 标记的嵌入矩阵连接到 Mel 编码器"""

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=config.composer_vocab_size, embedding_dim=config.d_model)

    def forward(self, feature, index_value, embedding_offset):
        index_shifted = index_value - embedding_offset  # 计算偏移后的索引值
        composer_embedding = self.embedding(index_shifted).unsqueeze(1)  # 获取特定索引的嵌入矩阵并增加维度
        inputs_embeds = torch.cat([composer_embedding, feature], dim=1)  # 在维度1上连接嵌入矩阵和特征
        return inputs_embeds


Pop2Piano_START_DOCSTRING = r"""
    该模型继承自 [`PreTrainedModel`]。查看超类文档以了解库实现的通用方法，如下载或保存、改变输入嵌入、修剪头部等。

    该模型也是一个 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类。
    将其视为常规的 PyTorch 模块，并参考 PyTorch 文档以了解一切与一般使用和行为相关的内容。

    参数:
        config ([`Pop2PianoConfig`]): 包含模型所有参数的模型配置类。
            用配置文件初始化不会加载与模型关联的权重，只会加载配置。查看 [`~PreTrainedModel.from_pretrained`] 方法
            以加载模型权重。
"""


@add_start_docstrings("""在顶部有 `language modeling` 头部的 Pop2Piano 模型。""", Pop2Piano_START_DOCSTRING)
class Pop2PianoForConditionalGeneration(Pop2PianoPreTrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: Pop2PianoConfig):
        super().__init__(config)
        self.config = config
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)  # 共享的词嵌入矩阵

        self.mel_conditioner = Pop2PianoConcatEmbeddingToMel(config)  # Mel 编码器的 composer 嵌入矩阵连接对象

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.encoder = Pop2PianoStack(encoder_config, self.shared)  # 编码器

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = Pop2PianoStack(decoder_config, self.shared)  # 解码器

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)  # 语言模型的线性层

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.shared  # 返回共享词嵌入矩阵

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings  # 设置新的词嵌入矩阵
        self.encoder.set_input_embeddings(new_embeddings)  # 设置编码器的输入嵌入矩阵
        self.decoder.set_input_embeddings(new_embeddings)  # 设置解码器的输入嵌入矩阵
    # 设置输出的嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 获取输出的嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 获取mel条件器输出
    def get_mel_conditioner_outputs(
        self,
        input_features: torch.FloatTensor,
        composer: str,
        generation_config: GenerationConfig,
        attention_mask: torch.FloatTensor = None,
    ):
        """
        This method is used to concatenate mel conditioner tokens at the front of the input_features in order to
        control the type of MIDI token generated by the model.

        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                input features extracted from the feature extractor.
            composer (`str`):
                composer token which determines the type of MIDI tokens to be generated.
            generation_config (`~generation.GenerationConfig`):
                The generation is used to get the composer-feature_token pair.
            attention_mask (``, *optional*):
                For batched generation `input_features` are padded to have the same shape across all examples.
                `attention_mask` helps to determine which areas were padded and which were not.
                - 1 for tokens that are **not padded**,
                - 0 for tokens that are **padded**.
        """
        # 获取composer_to_feature_token的映射
        composer_to_feature_token = generation_config.composer_to_feature_token
        # 如果composer不在composer_to_feature_token的键中，则引发值错误
        if composer not in composer_to_feature_token.keys():
            raise ValueError(
                f"Please choose a composer from {list(composer_to_feature_token.keys())}. Composer received - {composer}"
            )
        # 获取对应composer的值
        composer_value = composer_to_feature_token[composer]
        composer_value = torch.tensor(composer_value, device=self.device)
        composer_value = composer_value.repeat(input_features.shape[0])

        # 获取embedding_offset的最小值
        embedding_offset = min(composer_to_feature_token.values())

        # 使用mel_conditioner添加新数组到input_features的前面
        input_features = self.mel_conditioner(
            feature=input_features,
            index_value=composer_value,
            embedding_offset=embedding_offset,
        )
        # 如果存在attention_mask
        if attention_mask is not None:
            # 将attention_mask的值为0的位置对应的input_features设置为0.0
            input_features[~attention_mask[:, 0].bool()] = 0.0

            # 由于self.mel_conditioner在inputs_embeds的前面添加了一个新数组，我们需要为attention_mask做同样的操作保持形状相同
            attention_mask = torch.cat([attention_mask[:, 0].view(-1, 1), attention_mask], dim=1)
            return input_features, attention_mask

        return input_features, None

    @add_start_docstrings_to_model_forward(POP2PIANO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播函数，对模型进行推断
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的词嵌入 ID
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器的输入词嵌入 ID
        decoder_attention_mask: Optional[torch.BoolTensor] = None,  # 解码器的注意力掩码
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码
        decoder_head_mask: Optional[torch.FloatTensor] = None,  # 解码器的头部掩码
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力头部掩码
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 编码器输出
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 过去的键值
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量
        input_features: Optional[torch.FloatTensor] = None,  # 输入特征
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入嵌入向量
        labels: Optional[torch.LongTensor] = None,  # 标签
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
    @torch.no_grad()
    # 生成函数，执行模型的生成
    def generate(
        self,
        input_features,  # 输入特征
        attention_mask=None,  # 注意力掩码，默认为空
        composer="composer1",  # composer参数，默认为composer1
        generation_config=None,  # 生成的配置信息，默认为空
        **kwargs,  # 其他关键字参数
    # 为生成准备输入函数
    def prepare_inputs_for_generation(
        self,
        input_ids,  # 输入的 ID
        past_key_values=None,  # 过去的键值，默认为空
        attention_mask=None,  # 注意力掩码，默认为空
        head_mask=None,  # 头部掩码，默认为空
        decoder_head_mask=None,  # 解码器头部掩码，默认为空
        cross_attn_head_mask=None,  # 交叉注意力头部掩码，默认为空
        use_cache=None,  # 是否使用缓存，默认为空
        encoder_outputs=None,  # 编码器输出，默认为空
        **kwargs,  # 其他关键字参数
    ):
        # 如果使用过去的键值，则截取decoder_input_ids
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 返回准备好的输入字典
        return {
            "decoder_input_ids": input_ids,  # 解码器输入 ID
            "past_key_values": past_key_values,  # 过去的键值
            "encoder_outputs": encoder_outputs,  # 编码器输出
            "attention_mask": attention_mask,  # 注意力掩码
            "head_mask": head_mask,  # 头部掩码
            "decoder_head_mask": decoder_head_mask,  # 解码器头部掩码
            "cross_attn_head_mask": cross_attn_head_mask,  # 交叉注意力头部掩码
            "use_cache": use_cache,  # 是否使用缓存
        }

    # 从标签准备解码器输入 ID
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 将标签右移一位作为解码器输入
        return self._shift_right(labels)
    # 根据当前 beam 索引重新排序 past_key_values
    def _reorder_cache(self, past_key_values, beam_idx):
        # 如果 past_key_values 为 None，说明没有启用 use_cache，无需重新排序
        if past_key_values is None:
            # 提醒用户可以开启 use_cache 加快解码速度
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values
    
        # 创建一个新的元组，用于存储重新排序后的 past_key_values
        reordered_decoder_past = ()
        
        # 遍历每个 layer 的 past_key_values
        for layer_past_states in past_key_values:
            # 创建一个新的元组，用于存储重新排序后的当前 layer 的 past_key_values
            reordered_layer_past_states = ()
            
            # 遍历当前 layer 的每个 past_key_value
            for layer_past_state in layer_past_states:
                # 根据当前 beam 索引重新排序当前 past_key_value
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )
            
            # 检查重新排序后的 past_key_values 是否与原始的长度和形状一致
            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )
            
            # 将当前 layer 的重新排序后的 past_key_values 添加到新的元组中
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        
        # 返回重新排序后的 past_key_values
        return reordered_decoder_past
```