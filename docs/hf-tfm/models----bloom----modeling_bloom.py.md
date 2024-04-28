# `.\transformers\models\bloom\modeling_bloom.py`

```
# 设置 UTF-8 编码声明
# 版权声明和许可证信息
"""PyTorch BLOOM 模型。"""

# 导入必要的库
import math  # 导入数学库
import warnings  # 导入警告库
from typing import Optional, Tuple, Union  # 导入类型提示相关的类

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 检查点相关库
from torch import nn  # 导入神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss  # 导入损失函数相关类
from torch.nn import functional as F  # 导入 torch.nn.functional 库，并命名为 F

from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward  # 导入文件工具相关函数
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask  # 导入模型注意力掩码工具函数
from ...modeling_outputs import (  # 导入模型输出类
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...utils import logging  # 导入日志记录工具
from .configuration_bloom import BloomConfig  # 导入 BLOOM 模型配置类

logger = logging.get_logger(__name__)  # 获取日志记录器

_CHECKPOINT_FOR_DOC = "bigscience/bloom-560m"  # 文档中的检查点
_CONFIG_FOR_DOC = "BloomConfig"  # 文档中的配置

# 预训练模型的存档列表
BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bigscience/bigscience-small-testing",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    "bigscience/bloom",
]

def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """
    构建 Alibi 张量。

    链接到论文：https://arxiv.org/abs/2108.12409 Alibi 张量不是因果的，原始论文提到，它依赖于 softmax 的平移不变性，以进行快速实现：
    对于固定值 `softmax(l+a) = softmax(l)`。基于 https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 由于掩码策略的问题，这不太好用，因此掩码会略有不同。

    Args:
        attention_mask (`torch.Tensor`):
            每个标记的注意力掩码，形状应为 (batch_size, max_seq_len)。
        num_heads (`int`, *required*):
            注意力头的数量。
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            输出张量的数据类型。

    Returns:
        `torch.Tensor`: 形状为 (batch_size * num_heads, 1, max_seq_len) 的张量。
    """
    batch_size, seq_length = attention_mask.shape  # 获取注意力掩码的形状
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))  # 计算最接近 num_heads 的 2 的幂次方
``` 
    # 计算基础值，根据最接近的 2 的幂次方的对数计算
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    # 生成从 1 到最接近的 2 的幂次方的整数序列
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    # 计算斜率值
    slopes = torch.pow(base, powers)

    # 如果最接近的 2 的幂次方不等于头数，则需要额外计算
    if closest_power_of_2 != num_heads:
        # 计算额外的基础值
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        # 计算剩余头数
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        # 生成额外的整数序列
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        # 将额外计算的斜率值拼接到原有的斜率值中
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # 生成用于计算注意力偏置的 alibi 张量
    # 注意：alibi 将被添加到应用于注意力的查询-键乘积的偏置中
    # => 因此 alibi 必须具有形状 (batch_size, num_heads, query_length, key_length)
    # => 这里我们设置 (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => 然后 query_length 维度将被正确广播
    # 这与 T5 的相对位置偏置几乎相同
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    # 重塑 alibi 张量的形状并转换数据类型
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)
def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            residual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required`):
            training mode
    """
    # 对输入进行 dropout 操作
    out = F.dropout(x, p=prob, training=training)
    # 将 dropout 后的结果与残差相加
    out = residual + out
    return out


def bloom_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    # 自定义的 GELU 函数，用于推理模式
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def bloom_gelu_back(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    gradient of tanh approximation of gelu gradient of actual gelu is: 0.5 * (1. + torch.erf(x * 0.70710678)) +
    0.3989423 * x * torch.exp(-0.5 * x * x)

    Args:
        g (`torch.tensor`, *required*):
            gradient output tensor
        x (`torch.tensor`, *required*):
            input tensor
    """
    x = x[0]  # x is a tuple of 1 element, needs to unpack it first
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # 计算 GELU 的反向传播梯度
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g


class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return bloom_gelu_forward(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input = ctx.saved_tensors
        tmp = bloom_gelu_back(grad_output, input)
        return tmp


class BloomGelu(nn.Module):
    """
    BloomBiasGelu wrapper function that make use of the simple function on inference mode to make the model
    torchscriptable and use the autograd function in training mode to get the accurate results of the gradients Partly
    copied from Megatron-DeepSpeed code and adapted for our needs

    See here why autograd functions are not torchscriptable: https://github.com/pytorch/pytorch/issues/22329
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return GeLUFunction.apply(x)
        else:
            return bloom_gelu_forward(x)


class BloomAttention(nn.Module):
    # 初始化函数，接受一个 BloomConfig 类型的参数
    def __init__(self, config: BloomConfig):
        # 调用父类的初始化函数
        super().__init__()

        # 从配置中获取预训练类型和是否慢但准确的标志
        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact

        # 获取隐藏层大小、头数、头维度、分割大小和隐藏层dropout率
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        # 检查隐藏层大小是否可以被头数整除
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # 计算层间注意力的缩放因子和 beta 值
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        # 初始化查询、键、值的线性变换层和全连接层
        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    # 将融合的查询、键、值张量分割成查询、键、值三部分
    def _split_heads(self, fused_qkv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
        storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """
        # 获取输入张量的形状信息
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        # 重塑张量，将最后一维分割成 num_heads * 3 * head_dim 的形状
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        # 返回分割后的查询、键、值张量
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimension

        Args:
            x (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # 将最后一维的头合并在一起

        # 获取输入张量的形状
        batch_size_and_num_heads, seq_length, _ = x.shape
        # 计算每个批次中的样本数
        batch_size = batch_size_and_num_heads // self.num_heads

        # 首先重新排列张量以分解批次大小
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # 将维度重新排列以符合预期的形状
        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # 将头维度合并到一个维度中
        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
class BloomMLP(nn.Module):
    # 定义 BloomMLP 类，继承自 nn.Module
    def __init__(self, config: BloomConfig):
        # 初始化函数，接受一个 BloomConfig 类型的参数 config
        super().__init__()
        # 调用父类的初始化函数

        hidden_size = config.hidden_size
        # 从 config 中获取 hidden_size

        self.pretraining_tp = config.pretraining_tp
        # 从 config 中获取 pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        # 从 config 中获取 slow_but_exact
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
        # 创建一个全连接层，输入维度为 hidden_size，输出维度为 4 * hidden_size
        self.gelu_impl = BloomGelu()
        # 创建一个 BloomGelu 实例
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)
        # 创建一个全连接层，输入维度为 4 * hidden_size，输出维度为 hidden_size
        self.hidden_dropout = config.hidden_dropout
        # 从 config 中获取 hidden_dropout

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # 前向传播函数，接受两个 torch.Tensor 类型的参数 hidden_states 和 residual，返回一个 torch.Tensor
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))
        # 将 hidden_states 经过全连接层和激活函数 BloomGelu 处理

        if self.pretraining_tp > 1 and self.slow_but_exact:
            # 如果 pretraining_tp 大于 1 并且 slow_but_exact 为真
            intermediate_output = torch.zeros_like(residual)
            # 创建一个与 residual 相同形状的全零张量
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            # 计算每个切片的大小
            for i in range(self.pretraining_tp):
                # 遍历 pretraining_tp 次
                intermediate_output = intermediate_output + F.linear(
                    hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
                # 使用 F.linear 计算中间输出
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)
            # 否则直接使用全连接层处理 hidden_states

        output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)
        # 调用 dropout_add 函数，将中间输出和 residual 进行加权和处理

        return output
        # 返回输出结果


class BloomBlock(nn.Module):
    # 定义 BloomBlock 类，继承自 nn.Module
    def __init__(self, config: BloomConfig):
        # 初始化函数，接受一个 BloomConfig 类型的参数 config
        super().__init__()
        # 调用父类的初始化函数

        hidden_size = config.hidden_size
        # 从 config 中获取 hidden_size

        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 创建一个 LayerNorm 层，输入维度为 hidden_size
        self.num_heads = config.n_head
        # 从 config 中获取 n_head
        self.self_attention = BloomAttention(config)
        # 创建一个 BloomAttention 实例
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 创建一个 LayerNorm 层，输入维度为 hidden_size

        self.mlp = BloomMLP(config)
        # 创建一个 BloomMLP 实例

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        # 从 config 中获取 apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout
        # 从 config 中获取 hidden_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        # 在变换器层开始处进行层归一化
        layernorm_output = self.input_layernorm(hidden_states)

        # 在自注意力之后进行层归一化
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # 自注意力机制
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        # 在自注意力之后再次进行层归一化
        layernorm_output = self.post_attention_layernorm(attention_output)

        # 获取残差连接
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # 多层感知机
        output = self.mlp(layernorm_output, residual)

        # 如果使用缓存，则输出中包含输出；否则，输出不包含输入
        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions
class BloomPreTrainedModel(PreTrainedModel):
    # 设置配置类为BloomConfig
    config_class = BloomConfig
    # 设置基础模型前缀为"transformer"
    base_model_prefix = "transformer"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要拆分的模块为"BloomBlock"
    _no_split_modules = ["BloomBlock"]
    # 跳过键的设备放置为"past_key_values"
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # 如果是线性层，使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果有偏置项，初始化为0
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 如果是嵌入层，使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果有填充索引，将对应位置的权重初始化为0
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            # 如果是LayerNorm层，初始化偏置为0，权重为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @staticmethod
    def _convert_to_standard_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_size: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
        num_heads, ...]))
        """
        batch_size_times_num_heads, head_dim, seq_length = past_key_value[0][0].shape
        num_heads = batch_size_times_num_heads // batch_size
        # 将键值格式标准化，以匹配大多数实现，即tuple(tuple([batch_size, num_heads, ...]))
        # 将键值格式转换为[batch_size, num_heads, head_dim, seq_length]和[batch_size, num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size, num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size, num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )

    @staticmethod
    def _convert_to_bloom_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        将缓存转换为 Bloom 预期的格式，即元组（元组（[batch_size * num_heads, ...]））
        """
        # 从 past_key_value 中获取形状信息
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        # 计算 batch_size 乘以 num_heads，以便重塑张量
        batch_size_times_num_heads = batch_size * num_heads
        # 重塑 past_key_value 中的张量，以适应 Bloom 所需的形状
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size_times_num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim),
            )
            for layer_past in past_key_value
        )
# Bloom 模型的文档字符串，包含了继承关系、模型的基本参数说明以及在 PyTorch 中的使用方式
BLOOM_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BloomConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# Bloom 模型的输入文档字符串
BLOOM_INPUTS_DOCSTRING = r"""
"""

# BloomModel 类，继承自 BloomPreTrainedModel，用于定义 Bloom 模型的结构
@add_start_docstrings(
    "The bare Bloom Model transformer outputting raw hidden-states without any specific head on top.",
    BLOOM_START_DOCSTRING,
)
class BloomModel(BloomPreTrainedModel):
    # BloomModel 类的初始化方法
    def __init__(self, config: BloomConfig):
        super().__init__(config)

        # 获取配置中的隐藏层维度和注意力头数
        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        # 词嵌入层及 LayerNorm 层
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Transformer 块
        self.h = nn.ModuleList([BloomBlock(config) for _ in range(config.num_hidden_layers)])

        # 最终的 Layer Norm 层
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # 是否启用渐变检查点
        self.gradient_checkpointing = False

        # 初始化权重并进行最终处理
        self.post_init()

    # 构建 alibi 张量
    def build_alibi_tensor(self, attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
        return build_alibi_tensor(attention_mask, num_heads, dtype)

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    # BloomModel 类的前向传播方法，包含了详细的参数说明
    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
# 导入必要的库
@add_start_docstrings(
    """
    The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    BLOOM_START_DOCSTRING,
)
# 定义 BloomForCausalLM 类，继承自 BloomPreTrainedModel
class BloomForCausalLM(BloomPreTrainedModel):
    # 定义 tied_weights_keys 属性
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化方法，接受一个 BloomConfig 类型的参数
    def __init__(self, config: BloomConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 BloomModel 对象
        self.transformer = BloomModel(config)
        # 创建线性层，用于语言建模
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    # 为生成准备输入
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # 如果 past_key_values 不为 None，则只保留最后一个 token
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 如果 input_ids 的长度大于 past_length，则移除前缀长度为 past_length
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则，默认只保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

            # 如果 past_key_values 的格式需要转换，则转换为 Bloom 的格式
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

        # 如果传入了 inputs_embeds，并且 past_key_values 为 None，则只在第一个生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    # 添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    # 添加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # Transformer 模型的前向传播方法，用于生成模型的输出
    def forward(
        self,
        # 输入 token IDs，可选，表示输入的 token 序列
        input_ids: Optional[torch.LongTensor] = None,
        # 用于存储 Transformer 模型的过去键值的元组，可选
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        # 注意力遮罩，可选，用于指示哪些 token 是无效的
        attention_mask: Optional[torch.Tensor] = None,
        # 头部遮罩，可选，用于指示哪些注意力头是无效的
        head_mask: Optional[torch.Tensor] = None,
        # 输入嵌入，可选，用于直接输入嵌入表示而不是 token IDs
        inputs_embeds: Optional[torch.Tensor] = None,
        # 标签，可选，用于计算损失
        labels: Optional[torch.Tensor] = None,
        # 是否使用缓存，可选，用于控制是否存储 Transformer 模型的中间状态
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，可选，控制是否返回每层的注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选，控制是否返回每层的隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典类型的结果，可选，控制返回结果的格式
        return_dict: Optional[bool] = None,
        # 废弃的参数列表，允许传递额外的参数，但会被忽略
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 检查是否有过时的参数，并将其从字典中弹出
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` 可能是 `torch.Tensor` 或 `None`，所以默认将 pop 设置为 `False` 可以检测用户是否显式传递了 `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        # 如果存在未预期的参数，则引发 ValueError
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 传递输入到 Transformer 模型中，并获取输出
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取隐藏状态
        hidden_states = transformer_outputs[0]

        # 生成语言模型的 logits
        lm_logits = self.lm_head(hidden_states)

        # 计算损失
        loss = None
        if labels is not None:
            # 将标签移到正确的设备以启用模型并行处理
            labels = labels.to(lm_logits.device)
            # 移位，使得 < n 的 token 预测为 n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # 将 token 展平
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        # 如果不返回字典，则返回元组
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有交叉注意力的 CausalLMOutputWithCrossAttentions 对象
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
        self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        这个函数用于重新排列`past_key_values`缓存，如果调用了[`~PreTrainedModel.beam_search`]或
        [`~PreTrainedModel.beam_sample`]。这是为了在每个生成步骤中将`past_key_values`与正确的beam_idx匹配。

        输出与`past`共享相同的内存存储。
        """
        # 将`past`转换为标准缓存，以便匹配给定的`beam_idx`大小
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx))

        # 在需要这些索引的所有设备上获取`beam_idx`的副本。
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        # 重新排列`standardized_past`中的缓存，以匹配`beam_idx`
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        # 将重新排列后的缓存转换为布隆缓存并返回
        return self._convert_to_bloom_cache(reordered_past)
# 定义一个带有序列分类头部的 Bloom 模型转换器（线性层）
# BloomForSequenceClassification 使用最后一个标记进行分类，类似于其他因果模型（例如 GPT-1）的做法
# 由于它在最后一个标记上进行分类，因此需要知道最后一个标记的位置。如果配置中定义了 pad_token_id，则在每行中找到不是填充标记的最后一个标记。
# 如果未定义 pad_token_id，则简单地取每行批次中的最后一个值。当传递 inputs_embeds 而不是 input_ids 时，无法猜测填充标记，因此也是相同的做法（取每行批次中的最后一个值）。
class BloomForSequenceClassification(BloomPreTrainedModel):
    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = BloomModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
# 定义一个带有标记分类头部的 Bloom 模型（隐藏状态输出的顶部线性层），例如用于命名实体识别（NER）任务
class BloomForTokenClassification(BloomPreTrainedModel):
    # 初始化函数，接受一个 BloomConfig 类型的参数 config
    def __init__(self, config: BloomConfig):
        # 调用父类的初始化函数
        super().__init__(config)
        # 设置类属性 num_labels 为 config 中的 num_labels
        self.num_labels = config.num_labels

        # 创建一个 BloomModel 类的实例，传入参数为 config
        self.transformer = BloomModel(config)
        
        # 检查 config 是否具有 classifier_dropout 属性，并且其值不为 None
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            # 如果是，则将 classifier_dropout 设置为 config 中的 classifier_dropout
            classifier_dropout = config.classifier_dropout
        # 否则，检查 config 是否具有 hidden_dropout 属性，并且其值不为 None
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            # 如果是，则将 classifier_dropout 设置为 config 中的 hidden_dropout
            classifier_dropout = config.hidden_dropout
        else:
            # 否则，将 classifier_dropout 设置为 0.1
            classifier_dropout = 0.1
        
        # 创建一个 dropout 层，dropout 的概率为 classifier_dropout
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建一个线性层，输入大小为 config 中的 hidden_size，输出大小为 config 中的 num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 前向传播函数，接受多个输入参数
    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 检查是否有过时的参数`position_ids`，并弹出以查看是否为`False`，如果是，则意味着用户显式地传递了`None`
        if deprecated_arguments.pop("position_ids", False) is not False:
            # 发出警告，说明`position_ids`在BLOOM中没有功能，并将在v5.0.0中删除。您可以安全地忽略传递`position_ids`。
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        # 如果有未使用的参数，则引发值错误，将其打印出来
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        # 如果`return_dict`为`None`，则将其设置为`self.config.use_return_dict`
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给transformer模型进行处理
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取transformer输出的隐藏状态
        hidden_states = transformer_outputs[0]
        # 对隐藏状态进行dropout
        hidden_states = self.dropout(hidden_states)
        # 将dropout后的隐藏状态传递给分类器，得到logits
        logits = self.classifier(hidden_states)

        # 初始化损失为None
        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            # 将标签移到正确的设备以启用模型并行处理
            labels = labels.to(logits.device)
            # 获取标签的形状
            batch_size, seq_length = labels.shape
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算损失
            loss = loss_fct(
                logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
            )

        # 如果`return_dict`为False，则返回包含logits和其他transformer输出的元组
        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果`return_dict`为True，则返回TokenClassifierOutput对象，包含损失、logits、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
# 使用 BLOOM 模型进行抽取式问答任务的转换器，顶部有一个用于计算 `span start logits` 和 `span end logits` 的线性层
class BloomForQuestionAnswering(BloomPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 初始化 BLOOM 模型
        self.transformer = BloomModel(config)
        # 初始化用于问答任务的线性层
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
        # 设置返回字典，如果未指定则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用transformer处理输入数据
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 获取logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果在多GPU上，添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时开始/结束位置超出模型输入范围，忽略这些项
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```