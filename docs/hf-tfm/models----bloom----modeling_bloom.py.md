# `.\models\bloom\modeling_bloom.py`

```
# 设置文件编码格式为 UTF-8

# 导入必要的库和模块
import math  # 导入数学库，用于数学运算
import warnings  # 导入警告模块，用于处理警告信息
from typing import Optional, Tuple, Union  # 导入类型提示模块，用于定义函数参数和返回类型

import torch  # 导入 PyTorch 深度学习库
import torch.utils.checkpoint  # 导入 PyTorch 中用于支持 checkpoint 的工具函数
from torch import nn  # 导入 PyTorch 的神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss  # 导入 PyTorch 中的损失函数
from torch.nn import functional as F  # 导入 PyTorch 中的函数模块，使用别名 F

# 导入 Hugging Face 提供的一些工具和模块
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_bloom import BloomConfig  # 导入 BLOOM 模型的配置文件

logger = logging.get_logger(__name__)  # 获取日志记录器对象

_CHECKPOINT_FOR_DOC = "bigscience/bloom-560m"  # 模型检查点的示例名称
_CONFIG_FOR_DOC = "BloomConfig"  # Bloom 模型的配置文件示例名称

BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bigscience/bigscience-small-testing",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    "bigscience/bloom",
]  # 预训练模型的列表

def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """
    构建 Alibi 张量，参考文献：https://arxiv.org/abs/2108.12409。Alibi 张量不是因果性的，
    原始论文提到它依赖于 softmax 的平移不变性以进行快速实现：对于一个张量 l 和一个固定值 `softmax(l+a) = softmax(l)`。
    基于 https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 这并不完全适用于掩码策略，因此掩码稍有不同。

    Args:
        attention_mask (`torch.Tensor`):
            令牌级别的注意力掩码，应为形状 (batch_size, max_seq_len)。
        num_heads (`int`, *required*):
            多头注意力的数量。
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            输出张量的数据类型。

    Returns:
        torch.Tensor:
            形状为 (batch_size * num_heads, 1, max_seq_len) 的张量。
    """
    batch_size, seq_length = attention_mask.shape  # 获取注意力掩码的批量大小和序列长度
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))  # 计算最接近 num_heads 的 2 的幂次方
    # 使用最接近的 2 的幂次方的对数，计算指数，并将其转换为二进制后取倒数，再次取倒数，得到基数
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )

    # 创建一个张量，其中包含从 1 到最接近的 2 的幂次方的整数序列，设备和数据类型与 attention_mask 一致
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)

    # 根据 base 和 powers 计算斜率 slopes
    slopes = torch.pow(base, powers)

    # 如果最接近的 2 的幂次方不等于 num_heads，则需要添加额外的斜率
    if closest_power_of_2 != num_heads:
        # 计算额外基数，使用最接近的 2 倍的最接近的 2 的幂次方的对数
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        # 计算剩余的头数
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        # 创建一个张量，包含从 1 开始，以步长 2，到 2 倍剩余头数的整数序列
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        # 将额外的斜率拼接到现有的 slopes 张量中
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # 创建一个张量 arange_tensor，用于计算 alibi，形状为 (batch_size, num_heads, seq_length)
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]

    # 计算 alibi，将 slopes 乘以 arange_tensor，并添加一个维度以保持与返回的 alibi 形状一致
    alibi = slopes[..., None] * arange_tensor

    # 将 alibi 重塑为所需的形状，其中 batch_size 乘以 num_heads 作为第一维，seq_length 作为第三维，并转换为指定的数据类型
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
        training (`bool`, *required*):
            training mode
    """
    # Apply dropout to the input tensor `x` based on the provided probability `prob` and training mode
    out = F.dropout(x, p=prob, training=training)
    # Add the residual tensor to the dropout-applied tensor `out`
    out = residual + out
    return out


def bloom_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jittable.

    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    # Implement the Gaussian Error Linear Unit (GELU) activation function with a custom bias
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
    # Unpack the single element tuple `x` and calculate the derivative of the custom GELU function
    x = x[0]  # x is a tuple of 1 element, needs to unpack it first
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # Calculate the backward gradient for the GELU function
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g


class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the custom autograd function for GeLU.

        Args:
            ctx (`torch.autograd.function.Context`):
                context object to save tensors for backward pass
            input (`torch.tensor`, *required*):
                input tensor
        """
        # Save the input tensor `input` in the context for later use in backward pass
        ctx.save_for_backward(input)
        # Return the output of the custom GELU forward function
        return bloom_gelu_forward(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the custom autograd function for GeLU.

        Args:
            ctx (`torch.autograd.function.Context`):
                context object holding saved tensors from forward pass
            grad_output (`torch.tensor`, *required*):
                gradient of the output tensor
        """
        # Retrieve the saved input tensor from the context
        input = ctx.saved_tensors
        # Calculate the backward gradient using the custom GELU backward function
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
        """
        Forward pass of the BloomGelu module.

        Args:
            x (`torch.tensor`, *required*):
                input tensor
        """
        # Check if the model is in training mode; if yes, use custom autograd function GeLUFunction, otherwise use inference mode
        if self.training:
            return GeLUFunction.apply(x)
        else:
            return bloom_gelu_forward(x)


class BloomAttention(nn.Module):
    # This class definition is currently incomplete in the provided snippet
    pass
    def __init__(self, config: BloomConfig):
        # 调用父类构造函数初始化对象
        super().__init__()

        # 从配置对象中获取预训练的时间点和是否采用精确但较慢的模式
        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact

        # 从配置对象中获取隐藏层大小、头的数量和每个头的维度
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        # 检查隐藏层大小能否被头的数量整除
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # 计算用于注意力机制的缩放因子和初始值
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        # 初始化线性层，用于生成查询、键、值
        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        # 初始化线性层，用于注意力输出的全连接层
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        # 初始化注意力机制的dropout层
        self.attention_dropout = nn.Dropout(config.attention_dropout)

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
        # 获取输入张量的维度信息
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        # 将张量重新组织为 [batch_size, seq_length, num_heads, 3, head_dim]
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        # 返回查询、键、值张量，将最后一个维度的前三个用作查询、键、值
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads together over the last dimension

        Args:
            x (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # 获取输入张量的形状信息
        batch_size_and_num_heads, seq_length, _ = x.shape
        # 计算真实的 batch_size
        batch_size = batch_size_and_num_heads // self.num_heads

        # 将张量重塑为带有分解后的 batch size 的形状
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # 调整维度顺序，使得头部维度(num_heads)在中间，seq_length在前面
        x = x.permute(0, 2, 1, 3)

        # 将头部维度和 head_dim 合并成一个维度
        x = x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

        # 返回合并后的张量
        return x

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
# 定义一个名为 BloomMLP 的神经网络模块
class BloomMLP(nn.Module):
    # 初始化函数，接收一个 BloomConfig 类的配置对象作为参数
    def __init__(self, config: BloomConfig):
        super().__init__()
        hidden_size = config.hidden_size

        # 根据配置对象设置一些属性
        self.pretraining_tp = config.pretraining_tp  # 设定预训练的参数
        self.slow_but_exact = config.slow_but_exact  # 设定是否选择精确但慢的模式
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)  # 创建一个线性变换层，将隐藏层大小映射到4倍隐藏层大小
        self.gelu_impl = BloomGelu()  # 创建一个自定义的 GELU 激活函数对象
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)  # 创建一个线性变换层，将4倍隐藏层大小映射回隐藏层大小
        self.hidden_dropout = config.hidden_dropout  # 设定隐藏层的 dropout 概率

    # 前向传播函数，接收隐藏状态和残差张量作为输入，并返回一个张量
    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))  # 使用 GELU 激活函数处理线性变换后的隐藏状态

        # 根据预训练的参数和模式选择不同的计算方式
        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = torch.zeros_like(residual)  # 创建一个与残差张量相同大小的零张量
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp  # 计算切片的大小
            for i in range(self.pretraining_tp):
                # 执行线性变换操作，将隐藏状态的不同部分映射到目标维度
                intermediate_output = intermediate_output + F.linear(
                    hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)  # 直接使用线性变换将隐藏状态映射到目标维度

        # 调用外部的 dropout_add 函数，将中间输出与残差张量相加并应用 dropout
        output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)

        return output  # 返回处理后的输出张量


# 定义一个名为 BloomBlock 的神经网络模块
class BloomBlock(nn.Module):
    # 初始化函数，接收一个 BloomConfig 类的配置对象作为参数
    def __init__(self, config: BloomConfig):
        super().__init__()
        hidden_size = config.hidden_size

        # 创建输入层归一化层，使用给定的 epsilon 值初始化
        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head  # 设置头的数量
        self.self_attention = BloomAttention(config)  # 创建自注意力机制对象
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)  # 创建自注意力后的归一化层

        self.mlp = BloomMLP(config)  # 创建 MLP 模块对象

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm  # 是否应用残差连接后的归一化
        self.hidden_dropout = config.hidden_dropout  # 设置隐藏层的 dropout 概率

    # 前向传播函数，接收隐藏状态、辅助张量、注意力掩码等作为输入，并可选择性地返回注意力信息
    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,

        # 对输入隐藏状态应用层归一化
        hidden_states = self.input_layernorm(hidden_states)

        # 使用自注意力机制处理隐藏状态，返回处理后的注意力信息和新的隐藏状态
        attention_output = self.self_attention(
            hidden_states,
            attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        # 如果配置中设置了应用残差连接后的归一化，则将自注意力输出与输入隐藏状态相加并再次进行归一化
        if self.apply_residual_connection_post_layernorm:
            hidden_states = hidden_states + attention_output[0]  # 使用注意力输出更新隐藏状态
            hidden_states = self.post_attention_layernorm(hidden_states)  # 对更新后的隐藏状态再次进行归一化
        else:
            hidden_states = attention_output[0]  # 否则直接使用注意力输出作为隐藏状态的更新

        # 使用 MLP 模块处理更新后的隐藏状态和辅助张量，并返回处理后的结果
        output = self.mlp(hidden_states, alibi)

        return output  # 返回最终处理结果
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]
        # 定义一个方法，输入参数包含 hidden_states，表示隐藏状态的张量，形状为 [batch_size, seq_length, hidden_size]

        # Layer norm at the beginning of the transformer layer.
        # 在Transformer层的开始处进行层归一化处理
        layernorm_output = self.input_layernorm(hidden_states)
        # 使用 input_layernorm 对 hidden_states 进行层归一化处理，并将结果赋给 layernorm_output

        # Layer norm post the self attention.
        # 在自注意力机制之后进行层归一化处理
        if self.apply_residual_connection_post_layernorm:
            # 如果配置为在层归一化之后应用残差连接
            residual = layernorm_output
        else:
            # 否则，应用残差连接到原始的 hidden_states
            residual = hidden_states

        # Self attention.
        # 自注意力机制
        attn_outputs = self.self_attention(
            layernorm_output,  # 输入为归一化后的输出
            residual,           # 残差连接的输入
            layer_past=layer_past,              # 历史层的信息
            attention_mask=attention_mask,      # 注意力掩码
            alibi=alibi,                        # 辅助信息
            head_mask=head_mask,                # 注意力头的掩码
            use_cache=use_cache,                # 是否使用缓存
            output_attentions=output_attentions  # 是否输出注意力
        )

        attention_output = attn_outputs[0]  # 提取注意力输出

        outputs = attn_outputs[1:]  # 提取其它输出

        layernorm_output = self.post_attention_layernorm(attention_output)
        # 在注意力输出后进行层归一化处理，并将结果赋给 layernorm_output

        # Get residual
        # 获取残差连接
        if self.apply_residual_connection_post_layernorm:
            # 如果配置为在层归一化之后应用残差连接
            residual = layernorm_output
        else:
            # 否则，应用残差连接到注意力输出
            residual = attention_output

        # MLP.
        # 多层感知机（MLP）处理
        output = self.mlp(layernorm_output, residual)
        # 使用 MLP 处理层归一化的输出和残差连接结果，并将结果赋给 output

        if use_cache:
            # 如果使用缓存，则将 output 添加到输出元组中
            outputs = (output,) + outputs
        else:
            # 否则，仅将 output 添加到输出元组的第一个位置之后的元素中
            outputs = (output,) + outputs[1:]

        return outputs  # 返回结果元组，包含 hidden_states, present, attentions
    @staticmethod
    def _convert_to_bloom_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Convert the cache format to a custom format specific to Bloom.
        
        Args:
            past_key_value: Tuple of tuples containing tensors for keys and values from previous attention layers.
        
        Returns:
            Tuple of tuples containing tensors reshaped and restructured for Bloom model caching.
        """
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        # Convert to a custom cache format for Bloom
        # key: [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return tuple(
            (
                layer_past[0].view(batch_size * num_heads, head_dim, seq_length),
                layer_past[1].view(batch_size * num_heads, seq_length, head_dim),
                layer_past[0].view(batch_size * num_heads, head_dim, seq_length),  # Extra line added
            )
            for layer_past in past_key_value
        )
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the cache to the format expected by Bloom, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        # 解构 past_key_value 中第一个元素，获取其形状信息
        batch_size, num_heads, head_dim, seq_length = past_key_value[0][0].shape
        # 计算 batch_size * num_heads
        batch_size_times_num_heads = batch_size * num_heads
        # 将每个 layer_past 转换为 Bloom 预期的格式，并返回一个元组
        return tuple(
            (
                # 调整 key 的形状：[batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
                layer_past[0].view(batch_size_times_num_heads, head_dim, seq_length),
                # 调整 value 的形状：[batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
                layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim),
            )
            # 遍历 past_key_value 中的每个 layer_past
            for layer_past in past_key_value
        )
"""
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

BLOOM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
        past_key_values (:obj:`Tuple[Tuple[torch.Tensor, torch.Tensor], ...]`, `optional`):
            Tuple of length `config.num_hidden_layers`, containing tuples (`key`, `value`) for the cross-attention
            layers.
        attention_mask (:obj:`torch.Tensor`, `optional`):
            Mask to avoid performing attention on padding token indices. It is a tensor with shape
            `(batch_size, sequence_length)`, where each value is `0` for real tokens and `1` for padding tokens.
        head_mask (:obj:`torch.LongTensor`, `optional`):
            Mask to nullify selected heads of the self-attention modules. It is a tensor of shape
            `(num_heads,)`, where each value is either `0` or `1`. A `1` indicates the head is **not masked**, while a
            `0` indicates the head is masked.
        inputs_embeds (:obj:`torch.LongTensor`, `optional`):
            Embedded representation of the inputs. It is a tensor of shape `(batch_size, sequence_length,
            embedding_dim)`.
        use_cache (:obj:`bool`, `optional`):
            Whether or not to use the cached keys and values. If `False`, all intermediate keys and values are
            discarded and recomputed on-the-fly.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a dictionary instead of a tuple of outputs.

    Returns:
        :class:`~transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions`: A BaseModelOutputWithPastAndCrossAttentions
        object containing various elements depending on the configuration (e.g., hidden states, attentions, etc.).
"""

@add_start_docstrings(
    "The bare Bloom Model transformer outputting raw hidden-states without any specific head on top.",
    BLOOM_START_DOCSTRING,
)
class BloomModel(BloomPreTrainedModel):
    def __init__(self, config: BloomConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Transformer blocks
        self.h = nn.ModuleList([BloomBlock(config) for _ in range(config.num_hidden_layers)])

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def build_alibi_tensor(self, attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
        """
        Helper function to build a tensor with values based on attention_mask and number of heads.

        Args:
            attention_mask (:obj:`torch.Tensor`): Tensor indicating positions to ignore in attention computation.
            num_heads (:obj:`int`): Number of attention heads.
            dtype (:obj:`torch.dtype`): Data type of the tensor.

        Returns:
            :obj:`torch.Tensor`: A tensor with specific values based on input parameters.
        """
        return build_alibi_tensor(attention_mask, num_heads, dtype)

    def get_input_embeddings(self):
        """
        Retrieve the word embedding layer.

        Returns:
            :obj:`torch.nn.Embedding`: The word embedding layer.
        """
        return self.word_embeddings

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        """
        Set new word embeddings for the model.

        Args:
            new_embeddings (:obj:`torch.Tensor`): New word embeddings to be set.
        """
        self.word_embeddings = new_embeddings

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
    ):
        """
        Perform a forward pass of the BloomModel.

        Args:
            input_ids (:obj:`torch.LongTensor`, `optional`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (:obj:`Tuple[Tuple[torch.Tensor, torch.Tensor], ...]`, `optional`):
                Tuple of length `config.num_hidden_layers`, containing tuples (`key`, `value`) for the cross-attention
                layers.
            attention_mask (:obj:`torch.Tensor`, `optional`):
                Mask to avoid performing attention on padding token indices.
            head_mask (:obj:`torch.LongTensor`, `optional`):
                Mask to nullify selected heads of the self-attention modules.
            inputs_embeds (:obj:`torch.LongTensor`, `optional`):
                Embedded representation of the inputs.
            use_cache (:obj:`bool`, `optional`):
                Whether or not to use the cached keys and values.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a dictionary instead of a tuple of outputs.

        Returns:
            :class:`~transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions`: A BaseModelOutputWithPastAndCrossAttentions
            object containing various elements depending on the configuration.
        """
        # Implementation of forward pass is omitted for brevity in commenting.
        pass
@add_start_docstrings(
    """
    The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    BLOOM_START_DOCSTRING,
)
class BloomForCausalLM(BloomPreTrainedModel):
    # Define keys for tied weights
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BloomConfig):
        # Initialize the model with a configuration
        super().__init__(config)
        # Initialize the transformer model
        self.transformer = BloomModel(config)
        # Initialize the language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and perform any final processing
        self.post_init()

    def get_output_embeddings(self):
        # Return the language modeling head for output embeddings
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        # Set new embeddings for the language modeling head
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # Process inputs for generation
        
        # If past_key_values is provided, determine the length of past key values
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Adjust input_ids to keep only the last tokens if necessary
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

            # Convert past_key_values format if necessary for compatibility with Bloom cache
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

        # If inputs_embeds are provided and past_key_values is None, use them in the first generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Update model_inputs with additional parameters
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播函数，用于执行模型推理过程
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token ID序列，可以为空
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,  # 缓存的注意力机制的过去键值对，可以为空
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可以为空
        head_mask: Optional[torch.Tensor] = None,  # 多头注意力机制的掩码，可以为空
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入表示，可以为空
        labels: Optional[torch.Tensor] = None,  # 模型输出的标签，可以为空
        use_cache: Optional[bool] = None,  # 是否使用缓存，可以为空
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可以为空
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可以为空
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果，可以为空
        **deprecated_arguments,  # 其他已过时的参数，作为关键字参数传递
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 如果deprecated_arguments中包含"position_ids"键，并且其值不为False，则发出警告
        if deprecated_arguments.pop("position_ids", False) is not False:
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        # 如果deprecated_arguments中还有其他未预期的参数，则抛出异常
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        # 确定是否使用返回字典，如果未指定，则使用self.config.use_return_dict的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用transformer处理输入数据，获取transformer模型的输出
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
        # 从transformer模型输出中获取隐藏状态
        hidden_states = transformer_outputs[0]

        # 使用lm_head将隐藏状态转换为语言模型的logits
        lm_logits = self.lm_head(hidden_states)

        # 初始化损失为None
        loss = None
        # 如果提供了标签，则计算交叉熵损失
        if labels is not None:
            # 将标签移动到正确的设备以启用模型并行处理
            labels = labels.to(lm_logits.device)
            # 将logits向左移动一位，以便预测下一个标记
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # 展平标记
            loss_fct = CrossEntropyLoss()
            # 计算损失
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        # 如果不需要返回字典，则按照tuple形式返回输出
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则创建CausalLMOutputWithCrossAttentions对象
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
        self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
   `
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        # 将 past 转换为标准缓存格式，batch_size 设置为 beam_idx 的长度
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx))

        # 创建一个字典，将每个 layer_past 的设备映射到对应的 beam_idx
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        # 对每个 layer_past，使用 index_select 根据 device_to_beam_idx 重新排序，返回一个元组
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        # 将重新排序后的 past 转换为 bloom 缓存格式
        return self._convert_to_bloom_cache(reordered_past)
# Bloom 模型的序列分类器，使用线性层进行分类。

# 根据最后一个 token 进行分类，与其他因果模型（如 GPT-1）类似。

# 当进行最后一个 token 的分类时，需要知道最后一个 token 的位置。如果配置中定义了 `pad_token_id`，则在每一行中找到不是填充 token 的最后一个 token。
# 如果没有定义 `pad_token_id`，则在每个批次的每一行中取最后一个值。当传递 `inputs_embeds` 而不是 `input_ids` 时，无法猜测填充 token，因此也会取每行批次的最后一个值。

@add_start_docstrings(
    """
    Bloom 模型的令牌分类器，位于隐藏状态输出之上的线性层，例如用于命名实体识别（NER）任务。
    """,
    BLOOM_START_DOCSTRING,
)
class BloomForTokenClassification(BloomPreTrainedModel):
    # 初始化函数，接受一个BloomConfig类型的配置对象作为参数
    def __init__(self, config: BloomConfig):
        # 调用父类的初始化函数，将配置对象传递给父类
        super().__init__(config)
        # 从配置对象中获取num_labels属性，并赋值给当前对象的num_labels属性
        self.num_labels = config.num_labels

        # 使用BloomModel类根据配置对象初始化一个transformer模型
        self.transformer = BloomModel(config)

        # 根据配置对象中的classifier_dropout或hidden_dropout属性设置classifier_dropout变量
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        # 使用nn.Dropout类根据classifier_dropout变量初始化一个dropout层
        self.dropout = nn.Dropout(classifier_dropout)
        # 使用nn.Linear类初始化一个线性层，输入维度为配置对象中的hidden_size，输出维度为num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 执行额外的初始化和最终处理步骤
        self.post_init()

    # 将BLOOM_INPUTS_DOCSTRING和其他文档注释添加到模型的forward函数上
    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义前向传播函数，接受多个输入参数，并返回输出结果
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
        # 检查是否有被弃用的参数 `position_ids`，如果有则发出警告
        if deprecated_arguments.pop("position_ids", False) is not False:
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        # 检查是否有未预期的其他参数，如果有则引发 ValueError 异常
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        # 确定是否要返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Transformer 模型处理输入数据
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

        # 获取 Transformer 输出中的隐藏状态
        hidden_states = transformer_outputs[0]
        # 对隐藏状态应用 dropout 正则化
        hidden_states = self.dropout(hidden_states)
        # 使用分类器得到分类结果 logits
        logits = self.classifier(hidden_states)

        # 初始化损失值为 None
        loss = None
        # 如果提供了标签，则计算损失值
        if labels is not None:
            # 将标签移动到正确的设备以支持模型并行计算
            labels = labels.to(logits.device)
            batch_size, seq_length = labels.shape
            # 使用交叉熵损失函数计算损失值
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
            )

        # 如果不需要返回字典形式的输出，则组装返回结果
        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典形式的输出，则构建 TokenClassifierOutput 对象返回
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
# 使用自定义的文档字符串注释该类，描述其在问题回答任务中的用途和顶部的分类头部分
@add_start_docstrings(
    """
    The BLOOM Model transformer with a span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BLOOM_START_DOCSTRING,
)
class BloomForQuestionAnswering(BloomPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        # 初始化 BLOOM 模型部分
        self.transformer = BloomModel(config)
        # 初始化用于问题回答的输出线性层，输出大小为2（用于span的起始和结束logits）
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        # 根据 return_dict 参数确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Transformer 模型处理输入数据
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

        # 从 Transformer 输出中获取序列输出
        sequence_output = outputs[0]

        # 使用 QA 输出层处理序列输出，得到起始和结束位置的 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果 start_positions 或 end_positions 是多维的，在第一维上进行压缩
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 将超出模型输入长度的位置调整到有效范围内
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，忽略 ignore_index 处的预测
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            # 计算起始和结束位置损失的平均值作为总损失
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # 如果不要求返回字典，则返回一个元组
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 如果要求返回字典，则返回 QuestionAnsweringModelOutput 对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```