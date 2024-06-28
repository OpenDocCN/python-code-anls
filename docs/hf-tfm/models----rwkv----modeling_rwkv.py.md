# `.\models\rwkv\modeling_rwkv.py`

```
# 设置文件编码为 UTF-8
# 版权声明：2023 年 Bo Peng 和 HuggingFace 公司团队版权所有
# 版权声明：2018 年 NVIDIA 公司版权所有
#
# 根据 Apache 许可证 2.0 版本许可，除非符合许可协议，否则不得使用本文件
# 您可以在以下网址获取许可协议的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 不提供任何形式的担保或条件，无论是明示的还是默示的
# 有关详细信息，请参阅许可协议

"""PyTorch RWKV 模型."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 从模型工具中导入预训练模型类
from ...modeling_utils import PreTrainedModel
# 从工具中导入文档字符串生成函数和其它实用函数
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_bitsandbytes_available,
    is_ninja_available,
    is_torch_cuda_available,
    logging,
)
# 从相应模块导入 RWKV 配置类
from .configuration_rwkv import RwkvConfig

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置信息
_CHECKPOINT_FOR_DOC = "RWKV/rwkv-4-169m-pile"
_CONFIG_FOR_DOC = "RwkvConfig"

# 预训练模型归档列表
RWKV_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "RWKV/rwkv-4-169m-pile",
    "RWKV/rwkv-4-430m-pile",
    "RWKV/rwkv-4-1b5-pile",
    "RWKV/rwkv-4-3b-pile",
    "RWKV/rwkv-4-7b-pile",
    "RWKV/rwkv-4-14b-pile",
    "RWKV/rwkv-raven-1b5",
    "RWKV/rwkv-raven-3b",
    "RWKV/rwkv-raven-7b",
    "RWKV/rwkv-raven-14b",
    # 查看所有 RWKV 模型：https://huggingface.co/models?filter=rwkv
]

# RWKV CUDA 核心初始化为 None
rwkv_cuda_kernel = None


def load_wkv_cuda_kernel(context_length):
    # 从 torch.utils.cpp_extension 中加载 CUDA 核心
    from torch.utils.cpp_extension import load as load_kernel

    global rwkv_cuda_kernel

    # 获取 CUDA 核心文件夹路径
    kernel_folder = Path(__file__).resolve().parent.parent.parent / "kernels" / "rwkv"
    cuda_kernel_files = [kernel_folder / f for f in ["wkv_op.cpp", "wkv_cuda.cu", "wkv_cuda_bf16.cu"]]

    # 如果已加载的 CUDA 核心存在且上下文长度未更改，则直接返回
    if rwkv_cuda_kernel is not None and rwkv_cuda_kernel.max_seq_length == context_length:
        return

    # 记录加载 RWKV CUDA 核心的信息
    logger.info(f"Loading CUDA kernel for RWKV at context length of {context_length}.")

    # CUDA 编译标志
    flags = [
        "-res-usage",
        "--maxrregcount 60",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
        f"-DTmax={context_length}",
    ]
    # 加载 CUDA 核心
    rwkv_cuda_kernel = load_kernel(
        name=f"wkv_{context_length}",
        sources=cuda_kernel_files,
        verbose=(logging.get_verbosity() == logging.DEBUG),
        extra_cuda_cflags=flags,
    )
    rwkv_cuda_kernel.max_seq_length = context_length


class RwkvLinearAttention(torch.autograd.Function):
    @staticmethod
    # 定义一个静态方法 `forward`，接受多个参数和可选的状态信息，执行前向传播计算
    def forward(ctx, time_decay, time_first, key, value, state=None, return_state=False):
        # 获取输入张量的批量大小、序列长度和隐藏层大小
        batch_size, seq_len, hidden_size = key.size()
        # 如果序列长度超过最大允许长度，抛出异常
        if seq_len > rwkv_cuda_kernel.max_seq_length:
            raise ValueError(
                f"Cannot process a batch with {seq_len} tokens at the same time, use a maximum of "
                f"{rwkv_cuda_kernel.max_seq_length} with this model."
            )
        # 如果批量大小乘以隐藏层大小不能整除最小值（32），抛出异常
        if batch_size * hidden_size % min(hidden_size, 32) != 0:
            raise ValueError(
                f"The product of batch size ({batch_size}) and hidden size ({hidden_size}) needs to be a round "
                f"multiple of {min(hidden_size, 32)}."
            )

        # 设置上下文对象的输入数据类型为 key 的数据类型
        ctx.input_dtype = key.dtype

        # 检查时间衰减、时间优先、key 和 value 张量是否都在 CUDA 设备上，否则抛出异常
        if (
            time_decay.device.type != "cuda"
            or time_first.device.type != "cuda"
            or key.device.type != "cuda"
            or value.device.type != "cuda"
        ):
            raise ValueError("Calling the CUDA kernel for wkv attention requires all tensors to be on CUDA devices.")

        # 将时间衰减张量取负指数，转换为 float 类型并保证连续内存布局
        time_decay = -torch.exp(time_decay.float().contiguous())
        # 如果 key 的数据类型为 float16，将 time_first、key 和 value 转换为 float32 类型
        if key.dtype == torch.float16:
            time_first = time_first.float()
            key = key.float()
            value = value.float()
        # 确保 time_first、key 和 value 的连续内存布局
        time_first = time_first.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # 根据 key 的内存布局创建一个空的输出张量，保证其连续内存布局
        # CUDA 内核将填充这个张量
        output = torch.empty_like(key, memory_format=torch.contiguous_format)

        # 如果需要返回状态信息或者已提供状态信息
        if return_state or state is not None:
            # 如果未提供状态信息，则创建全零状态张量，并初始化最后一维度为 -1e38
            if state is None:
                state = torch.zeros(
                    batch_size,
                    hidden_size,
                    3,
                    dtype=torch.float32,
                    device=key.device,
                    memory_format=torch.contiguous_format,
                )
                state[:, :, 2] -= 1e38
            else:
                # 否则，将现有状态信息按最后一维度拼接，并保证连续内存布局
                state = torch.cat([s.unsqueeze(2) for s in state], dim=2).contiguous()
            # 根据 key 的数据类型选择前向传播函数，处理状态信息
            if key.dtype == torch.bfloat16:
                forward_func = rwkv_cuda_kernel.forward_with_state_bf16
            else:
                forward_func = rwkv_cuda_kernel.forward_with_state
            # 调用 CUDA 内核执行前向传播计算，并传递状态信息
            forward_func(time_decay, time_first, key, value, output, state)
        else:
            # 否则，根据 key 的数据类型选择相应的前向传播函数，不处理状态信息
            forward_func = rwkv_cuda_kernel.forward_bf16 if key.dtype == torch.bfloat16 else rwkv_cuda_kernel.forward
            # 调用 CUDA 内核执行前向传播计算，不传递状态信息
            forward_func(time_decay, time_first, key, value, output)

        # 将输入的关键数据和输出保存在上下文对象的备份中
        ctx.save_for_backward(time_decay, time_first, key, value, output)

        # 如果提供了状态信息，将其拆分并返回
        if state is not None:
            state = [s.squeeze(2) for s in torch.chunk(state, 3, dim=2)]

        # 返回计算结果的输出张量，并保证其数据类型与输入一致，同时返回状态信息
        return output.to(ctx.input_dtype), state

    @staticmethod
    # 静态方法的注释，g 代表梯度
    def backward(ctx, g_output, g_state=None):
        # 获取输入数据类型
        input_dtype = ctx.input_dtype

        # 从上下文中恢复保存的张量数据
        time_decay, time_first, key, value, output = ctx.saved_tensors
        # CUDA核心将填充这些张量。

        # 根据输入数据类型创建对应的梯度张量
        g_time_decay = torch.empty_like(
            time_decay,
            memory_format=torch.contiguous_format,
            dtype=torch.bfloat16 if input_dtype == torch.bfloat16 else torch.float32,
        )
        g_time_first = torch.empty_like(time_first, memory_format=torch.contiguous_format)
        g_key = torch.empty_like(key, memory_format=torch.contiguous_format)
        g_value = torch.empty_like(value, memory_format=torch.contiguous_format)

        # 如果输入数据类型是torch.float16，则将g_output转换为float类型
        if input_dtype == torch.float16:
            g_output = g_output.float()

        # 选择对应的CUDA函数进行反向传播计算
        backward_func = rwkv_cuda_kernel.backward_bf16 if input_dtype == torch.bfloat16 else rwkv_cuda_kernel.backward
        backward_func(
            time_decay,
            time_first,
            key,
            value,
            output,
            g_output.contiguous(),  # 获取g_output的连续内存视图
            g_time_decay,
            g_time_first,
            g_key,
            g_value,
        )

        # 将计算得到的梯度张量转换回输入数据类型并返回
        return (
            g_time_decay.to(input_dtype),
            g_time_first.to(input_dtype),
            g_key.to(input_dtype),
            g_value.to(input_dtype),
            None,
            None,
        )
# 使用线性键值注意力的 CPU 版本实现。如果不在 torch.no_grad 下执行，可能比自定义 CUDA 内核更慢且消耗更多内存。
def rwkv_linear_attention_cpu(time_decay, time_first, key, value, state=None, return_state=False):
    _, seq_length, _ = key.size()  # 获取键张量的序列长度
    output = torch.zeros_like(key)  # 初始化输出张量，与键张量相同形状

    if state is None:
        # 如果状态为空，初始化状态张量
        num_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
        den_state = torch.zeros_like(key[:, 0], dtype=torch.float32)
        max_state = torch.zeros_like(key[:, 0], dtype=torch.float32) - 1e38
    else:
        num_state, den_state, max_state = state  # 否则使用提供的状态张量

    # 对数值稳定性的考虑
    time_decay = -torch.exp(time_decay)

    # 迭代序列长度
    for current_index in range(seq_length):
        current_key = key[:, current_index].float()  # 当前时间步的键张量
        current_value = value[:, current_index]  # 当前时间步的值张量

        # 在时间步 t 计算线性键值注意力
        max_for_output = torch.maximum(max_state, current_key + time_first)
        e1 = torch.exp(max_state - max_for_output)
        e2 = torch.exp(current_key + time_first - max_for_output)
        numerator = e1 * num_state + e2 * current_value
        denominator = e1 * den_state + e2
        output[:, current_index] = (numerator / denominator).to(output.dtype)

        # 更新状态以备下一次迭代
        max_for_state = torch.maximum(max_state + time_decay, current_key)
        e1 = torch.exp(max_state + time_decay - max_for_state)
        e2 = torch.exp(current_key - max_for_state)
        num_state = e1 * num_state + e2 * current_value
        den_state = e1 * den_state + e2
        max_state = max_for_state

    # 如果需要返回状态或者状态不为空，则返回更新后的状态张量
    if return_state or state is not None:
        state = [num_state, den_state, max_state]

    return output, state


# 使用线性键值注意力的入口函数，根据硬件支持情况选择 CPU 或 CUDA 实现
def rwkv_linear_attention(time_decay, time_first, key, value, state=None, return_state=False):
    # 检查是否存在不支持 CUDA 的硬件，或者键张量的长度为 1
    no_cuda = any(t.device.type != "cuda" for t in [time_decay, time_first, key, value])
    one_token = key.size(1) == 1

    # 如果没有 CUDA 内核、不支持 CUDA 的硬件或者键张量的长度为 1，则调用 CPU 版本实现
    if rwkv_cuda_kernel is None or no_cuda or one_token:
        return rwkv_linear_attention_cpu(time_decay, time_first, key, value, state=state, return_state=return_state)
    else:
        # 否则调用 CUDA 版本实现
        return RwkvLinearAttention.apply(time_decay, time_first, key, value, state, return_state)
    # 初始化函数，用于初始化一个自定义的注意力层对象
    def __init__(self, config, layer_id=0):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置信息保存在对象属性中
        self.config = config
        # 检查是否已经加载了CUDA内核，并且内核支持的最大序列长度符合配置中的上下文长度
        kernel_loaded = rwkv_cuda_kernel is not None and rwkv_cuda_kernel.max_seq_length == config.context_length
        # 如果可以使用Ninja编译器、有可用的CUDA设备，并且尚未加载CUDA内核，则尝试加载自定义CUDA内核
        if is_ninja_available() and is_torch_cuda_available() and not kernel_loaded:
            try:
                load_wkv_cuda_kernel(config.context_length)
            except Exception:
                logger.info("Could not load the custom CUDA kernel for RWKV attention.")
        # 将层的ID保存在对象属性中
        self.layer_id = layer_id
        # 获取隐藏层的大小
        hidden_size = config.hidden_size
        # 获取注意力隐藏层的大小，如果未指定，则默认与隐藏层大小相同
        attention_hidden_size = (
            config.attention_hidden_size if config.attention_hidden_size is not None else hidden_size
        )
        # 将注意力隐藏层的大小保存在对象属性中
        self.attention_hidden_size = attention_hidden_size

        # 初始化时间衰减参数，用于注意力机制
        self.time_decay = nn.Parameter(torch.empty(attention_hidden_size))
        # 初始化时间首参数，用于注意力机制
        self.time_first = nn.Parameter(torch.empty(attention_hidden_size))

        # 初始化时间混合关键字参数，用于注意力机制
        self.time_mix_key = nn.Parameter(torch.empty(1, 1, hidden_size))
        # 初始化时间混合数值参数，用于注意力机制
        self.time_mix_value = nn.Parameter(torch.empty(1, 1, hidden_size))
        # 初始化时间混合接收参数，用于注意力机制
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, hidden_size))

        # 初始化时间偏移层，使用2D零填充，只在垂直方向（时间维度）上进行
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        # 初始化关键字线性层，将隐藏层映射到注意力隐藏层，无偏置
        self.key = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        # 初始化数值线性层，将隐藏层映射到注意力隐藏层，无偏置
        self.value = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        # 初始化接收线性层，将隐藏层映射到注意力隐藏层，无偏置
        self.receptance = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        # 初始化输出线性层，将注意力隐藏层映射回隐藏层大小，无偏置
        self.output = nn.Linear(attention_hidden_size, hidden_size, bias=False)

    # TODO: maybe jit, otherwise move inside forward
    # 提取关键字和数值，可能使用jit，否则将其移动到前向传播方法内
    def extract_key_value(self, hidden, state=None):
        # 将当前隐藏状态与上一时间步状态混合，生成关键字、数值、接收参数
        if hidden.size(1) == 1 and state is not None:
            # 如果隐藏状态的时间步为1且状态不为空，则从状态中提取上一时间步的值
            shifted = state[1][:, :, self.layer_id]
        else:
            # 否则，使用时间偏移层处理当前隐藏状态
            shifted = self.time_shift(hidden)
            # 如果状态不为空，则将上一时间步的值混合到当前时间步
            if state is not None:
                shifted[:, 0] = state[1][:, :, self.layer_id]
        # 使用时间混合关键字参数混合当前隐藏状态和上一时间步状态，生成关键字
        key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
        # 使用时间混合数值参数混合当前隐藏状态和上一时间步状态，生成数值
        value = hidden * self.time_mix_value + shifted * (1 - self.time_mix_value)
        # 使用时间混合接收参数混合当前隐藏状态和上一时间步状态，生成接收参数，并使用Sigmoid函数处理
        receptance = hidden * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance)

        # 将关键字、数值、接收参数分别通过线性层映射到注意力隐藏层
        key = self.key(key)
        value = self.value(value)
        receptance = torch.sigmoid(self.receptance(receptance))
        # 如果状态不为空，则更新状态中的上一时间步隐藏状态
        if state is not None:
            state[1][:, :, self.layer_id] = hidden[:, -1]
        # 返回接收参数、关键字、数值、状态
        return receptance, key, value, state
    # 前向传播函数，用于处理输入隐藏状态，可选地使用缓存
    def forward(self, hidden, state=None, use_cache=False):
        # 从隐藏状态中提取接受度、键和值，同时更新状态
        receptance, key, value, state = self.extract_key_value(hidden, state=state)
        
        # 如果存在状态，则从状态中提取当前层的状态信息
        layer_state = tuple(s[:, :, self.layer_id] for s in state[2:]) if state is not None else None
        
        # 使用 RWKV 线性注意力计算，考虑时间衰减和时间维度
        rwkv, layer_state = rwkv_linear_attention(
            self.time_decay,
            self.time_first,
            key,
            value,
            state=layer_state,
            return_state=use_cache,
        )

        # 如果存在层状态信息，则更新整体状态的当前层信息
        if layer_state is not None:
            state[2][:, :, self.layer_id] = layer_state[0]
            state[3][:, :, self.layer_id] = layer_state[1]
            state[4][:, :, self.layer_id] = layer_state[2]

        # 返回经过输出层处理后的结果以及更新后的状态
        return self.output(receptance * rwkv), state
# 定义一个名为 RwkvFeedForward 的新神经网络模块，继承自 nn.Module 类
class RwkvFeedForward(nn.Module):
    # 初始化函数，接受配置参数 config 和层编号 layer_id
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 保存配置信息和层编号到对象属性中
        self.config = config
        self.layer_id = layer_id
        # 从配置中获取隐藏层大小和中间层大小
        hidden_size = config.hidden_size
        intermediate_size = (
            config.intermediate_size if config.intermediate_size is not None else 4 * config.hidden_size
        )

        # 创建一个沿时间轴零填充的二维零填充层
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        # 创建一个时间混合关键字的可训练参数
        self.time_mix_key = nn.Parameter(torch.empty(1, 1, hidden_size))
        # 创建一个时间混合接受度的可训练参数
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, hidden_size))

        # 创建一个线性层对象，用于生成关键字
        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 创建一个线性层对象，用于生成接受度
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        # 创建一个线性层对象，用于生成值
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

    # 前向传播函数，接受隐藏层输入和状态信息
    def forward(self, hidden, state=None):
        # 如果隐藏层的第二维大小为1且状态不为空，则获取状态中的相应层次的偏移量
        if hidden.size(1) == 1 and state is not None:
            shifted = state[0][:, :, self.layer_id]
        else:
            # 否则，对隐藏层进行时间轴零填充操作，并根据状态调整填充结果
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[0][:, :, self.layer_id]

        # 计算关键字和接受度，根据时间混合参数和偏移量
        key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
        receptance = hidden * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance)

        # 对关键字进行非负整数平方操作，并使用 ReLU 激活函数
        key = torch.square(torch.relu(self.key(key)))
        # 将处理后的关键字输入值生成线性层，并输出值
        value = self.value(key)
        # 对接受度应用 sigmoid 激活函数
        receptance = torch.sigmoid(self.receptance(receptance))

        # 如果状态不为空，则更新状态中的隐藏层信息
        if state is not None:
            state[0][:, :, self.layer_id] = hidden[:, -1]

        # 返回接受度乘以值和更新后的状态
        return receptance * value, state


# 定义一个名为 RwkvBlock 的新神经网络模块，继承自 nn.Module 类
class RwkvBlock(nn.Module):
    # 初始化函数，接受配置参数 config 和层编号 layer_id
    def __init__(self, config, layer_id):
        super().__init__()
        # 保存配置信息和层编号到对象属性中
        self.config = config
        self.layer_id = layer_id

        # 如果层编号为0，则创建一个 LayerNorm 层对象，对隐藏层进行预处理
        if layer_id == 0:
            self.pre_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # 创建两个 LayerNorm 层对象，用于注意力机制前后的归一化处理
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # 创建 RwkvSelfAttention 和 RwkvFeedForward 的实例对象，用于注意力机制和前向传播
        self.attention = RwkvSelfAttention(config, layer_id)
        self.feed_forward = RwkvFeedForward(config, layer_id)

    # 前向传播函数，接受隐藏层输入、状态信息、是否使用缓存和是否输出注意力矩阵的参数
    def forward(self, hidden, state=None, use_cache=False, output_attentions=False):
        # 如果层编号为0，则对隐藏层进行预处理
        if self.layer_id == 0:
            hidden = self.pre_ln(hidden)

        # 将隐藏层输入传入注意力机制，获取注意力结果和更新后的状态
        attention, state = self.attention(self.ln1(hidden), state=state, use_cache=use_cache)
        # 将注意力结果加上原始隐藏层输入，得到新的隐藏层输出
        hidden = hidden + attention

        # 将新的隐藏层输入传入前向传播模块，获取前向传播结果和更新后的状态
        feed_forward, state = self.feed_forward(self.ln2(hidden), state=state)
        # 将前向传播结果加上原始隐藏层输入，得到最终的隐藏层输出
        hidden = hidden + feed_forward

        # 将隐藏层输出和状态信息作为元组返回
        outputs = (hidden, state)
        # 如果需要输出注意力矩阵，则将注意力矩阵加入返回的元组中
        if output_attentions:
            outputs += (attention,)
        else:
            outputs += (None,)

        # 返回最终的输出元组
        return outputs


# 定义一个名为 RwkvPreTrainedModel 的抽象神经网络模型类，继承自 PreTrainedModel
class RwkvPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，处理权重初始化和简单的预训练模型下载与加载接口。
    """

    # 类属性，配置类为 RwkvConfig
    config_class = RwkvConfig
    # 基础模型前缀为 "rwkv"
    base_model_prefix = "rwkv"
    # 不需要分割的模块名称列表中包含 "RwkvBlock"
    _no_split_modules = ["RwkvBlock"]
    # 定义需要保留在 FP32 模块中的模块名称列表
    _keep_in_fp32_modules = ["time_decay", "time_first"]
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化权重"""
        # 如果模块是 RwkvSelfAttention 类型
        if isinstance(module, RwkvSelfAttention):
            # 获取当前层的编号和总隐藏层数
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size
            attention_hidden_size = module.attention_hidden_size

            # 计算比率 0 到 1，表示当前层在所有隐藏层中的位置
            ratio_0_to_1 = layer_id / (num_hidden_layers - 1)  # 0 到 1
            # 计算比率 1 到 接近 0，表示当前层在所有隐藏层中的位置的反向比率
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 到 ~0

            # 创建时间权重张量，用于调整时间相关的关键字
            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.time_mix_key.dtype,
                device=module.time_mix_key.device,
            )
            time_weight = time_weight[None, None, :]

            # 计算时间衰减速度，根据注意力隐藏层大小和层位置动态调整
            decay_speed = [
                -5 + 8 * (h / (attention_hidden_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                for h in range(attention_hidden_size)
            ]
            decay_speed = torch.tensor(decay_speed, dtype=module.time_decay.dtype, device=module.time_decay.device)
            # 创建用于时间优先标记的波动
            zigzag = (
                torch.tensor(
                    [(i + 1) % 3 - 1 for i in range(attention_hidden_size)],
                    dtype=module.time_first.dtype,
                    device=module.time_first.device,
                )
                * 0.5
            )

            # 使用无梯度操作设置模块的时间衰减、时间优先和时间权重混合
            with torch.no_grad():
                module.time_decay.data = decay_speed
                module.time_first.data = torch.ones_like(module.time_first * math.log(0.3) + zigzag)

                module.time_mix_key.data = torch.pow(time_weight, ratio_1_to_almost0)
                module.time_mix_value.data = torch.pow(time_weight, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
                module.time_mix_receptance.data = torch.pow(time_weight, 0.5 * ratio_1_to_almost0)
        
        # 如果模块是 RwkvFeedForward 类型
        elif isinstance(module, RwkvFeedForward):
            # 获取当前层的编号和总隐藏层数
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size

            # 计算比率 1 到 接近 0，表示当前层在所有隐藏层中的位置的反向比率
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 到 ~0

            # 创建时间权重张量，用于调整时间相关的关键字
            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.time_mix_key.dtype,
                device=module.time_mix_key.device,
            )
            time_weight = time_weight[None, None, :]

            # 使用无梯度操作设置模块的时间权重混合和时间接受度
            with torch.no_grad():
                module.time_mix_key.data = torch.pow(time_weight, ratio_1_to_almost0)
                module.time_mix_receptance.data = torch.pow(time_weight, ratio_1_to_almost0)
# 使用 @dataclass 装饰器声明一个数据类，用于封装 RWKV 模型的输出结果
@dataclass
class RwkvOutput(ModelOutput):
    """
    Class for the RWKV model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义 RWKV 模型的输出属性
    last_hidden_state: torch.FloatTensor = None  # 最后一层模型的隐藏状态
    state: Optional[List[torch.FloatTensor]] = None  # 模型在最后时间步的状态
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None  # 每层模型的隐藏状态
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None  # 每层注意力权重
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        state (list of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 损失值，用于语言建模任务中的下一个标记预测，当提供了`labels`时返回
    loss: Optional[torch.FloatTensor] = None
    # 语言建模头部的预测分数，即在应用SoftMax之前每个词汇标记的分数，形状为`(batch_size, sequence_length, config.vocab_size)`
    logits: torch.FloatTensor = None
    # 模型在最后一个时间步的状态，可以在下一个`input_ids`的前向方法中使用，避免提供旧的`input_ids`
    state: Optional[List[torch.FloatTensor]] = None
    # 模型每一层的隐藏状态的元组，包括（如果存在）嵌入层的输出，形状为`(batch_size, sequence_length, hidden_size)`
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 注意力权重的元组，用于自注意力头部中的加权平均计算，形状为`(batch_size, num_heads, sequence_length, sequence_length)`
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# RWKV_START_DOCSTRING 定义了一个多行字符串，用于描述某个模型类的文档字符串。
# 文档字符串解释了该模型继承自 PreTrainedModel，列出了该库对所有模型实现的通用方法（如下载或保存模型、调整输入嵌入、剪枝头部等）。
# 这个模型也是 PyTorch 的 torch.nn.Module 的子类，可以像普通的 PyTorch 模块一样使用，所有与一般使用和行为相关的事项请参考 PyTorch 文档。

RWKV_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            # `input_ids` 是输入序列的 token 索引，在词汇表中进行查找得到。
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            # 用来避免对填充 token 索引执行注意力操作的掩码。掩码值选择在 `[0, 1]` 范围内：

            - 1 表示**未被掩码**的 token，
            - 0 表示**被掩码**的 token。

            This is currently not used by `RwkvModel`, but will be supported in the future.

            [What are attention masks?](../glossary#attention-mask)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            # 可选参数，代替 `input_ids` 直接传递嵌入表示。如果希望更好地控制如何将 `input_ids` 索引转换为关联向量，
            这是非常有用的，比如使用自定义的嵌入查找矩阵。

            This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        state (tuple of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`, *optional*):
            # 如果提供，模型将在所有块中使用先前状态（这将给出模型对提供的 `input_ids` 和 `state_input_ids` 作为上下文的输出）。

            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            # 如果设置为 `True`，则返回最后的状态，并且可以用于快速生成下一个 logits。

            If set to `True`, the last state is returned and can be used to quickly generate the next logits.
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详细信息请参见返回的张量中的 `attentions`。

            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详细信息请参见返回的张量中的 `hidden_states`。

            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通的元组。

            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
定义一个 RwkvModel 类，继承自 RwkvPreTrainedModel 类。

@add_start_docstrings(
    "The bare RWKV Model transformer outputting raw hidden-states without any specific head on top.",
    RWKV_START_DOCSTRING,
)
添加文档字符串，描述该模型是一个裸的 RWKV 模型，输出未经特定顶层处理的原始隐藏状态。

class RwkvModel(RwkvPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化嵌入层，使用给定的词汇量大小和隐藏层大小
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # 创建包含多个 RwkvBlock 的层列表，每个块的配置由传入的 config 控制
        self.blocks = nn.ModuleList([RwkvBlock(config, layer_id=idx) for idx in range(config.num_hidden_layers)])
        
        # 初始化 LayerNorm 层，对隐藏状态进行归一化处理
        self.ln_out = nn.LayerNorm(config.hidden_size)

        # 初始化标志：层是否被重新缩放
        self.layers_are_rescaled = False

        # 初始化标志：是否使用梯度检查点
        self.gradient_checkpointing = False

        # 执行额外的初始化操作
        # 这可能包括权重初始化和最终处理
        self.post_init()

    # 返回嵌入层
    def get_input_embeddings(self):
        return self.embeddings

    # 设置新的嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(RWKV_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=RwkvOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    定义 forward 方法，接收多个输入参数，执行模型的前向传播过程。

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def _rescale_layers(self):
        # Layers should be rescaled for inference only.
        if self.layers_are_rescaled == (not self.training):
            return
        # Check if rescaling interval is specified
        if self.config.rescale_every > 0:
            # Perform rescaling without gradient tracking
            with torch.no_grad():
                # Iterate over blocks in the model
                for block_id, block in enumerate(self.blocks):
                    if self.training:
                        # Scale weights during training
                        block.attention.output.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                        block.feed_forward.value.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                    else:
                        # Handle quantization statistics during inference
                        if hasattr(block.attention.output.weight, "SCB"):
                            block.attention.output.weight.SCB.div_(2 ** int(block_id // self.config.rescale_every))
                            block.feed_forward.value.weight.SCB.div_(2 ** int(block_id // self.config.rescale_every))
                        elif hasattr(block.attention.output.weight, "quant_state"):
                            # Perform 4-bit dequantization and rescaling
                            self._bnb_4bit_dequantize_and_rescale(block.attention.output, block_id)
                            self._bnb_4bit_dequantize_and_rescale(block.feed_forward.value, block_id)
                        else:
                            # Default case: rescale weights
                            block.attention.output.weight.div_(2 ** int(block_id // self.config.rescale_every))
                            block.feed_forward.value.weight.div_(2 ** int(block_id // self.config.rescale_every))

        # Update rescaling status
        self.layers_are_rescaled = not self.training

    def _bnb_4bit_dequantize_and_rescale(self, target_layer, block_id):
        r"""
        Perform the dequantization and rescaling of the weights of a given layer. After that operation the layer will
        be quantized again.
        """
        # Check if bitsandbytes library is available
        if not is_bitsandbytes_available():
            raise ImportError("Please install bitsandbytes to use this method.")
        import bitsandbytes as bnb

        # Dequantize 4-bit weights
        dequant_weights = bnb.functional.dequantize_4bit(target_layer.weight.data, target_layer.weight.quant_state)

        # Rescale weights
        dequant_weights.div_(2 ** int(block_id // self.config.rescale_every))

        # Re-quantize the weights
        # Move weights to CPU and back to device to handle quantization
        quant_weight = bnb.nn.Params4bit(dequant_weights.to("cpu"), requires_grad=False).to(dequant_weights.device)
        setattr(target_layer, "weight", quant_weight)
@add_start_docstrings(
    """
    The RWKV Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    RWKV_START_DOCSTRING,
)
class RwkvForCausalLM(RwkvPreTrainedModel):
    _tied_weights_keys = ["head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.rwkv = RwkvModel(config)  # 初始化 RWKV 模型
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 创建线性层作为语言建模的输出层

        # Initialize weights and apply final processing
        self.post_init()  # 执行初始化权重和最终处理

    def get_output_embeddings(self):
        return self.head  # 返回输出层的权重

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings  # 设置新的输出层权重

    def generate(self, *args, **kwargs):
        # Thin wrapper to raise exceptions when trying to generate with methods that manipulate `past_key_values`.
        # RWKV is one of the few models that don't have it (it has `state` instead, which has different properties and
        # usage).
        try:
            gen_output = super().generate(*args, **kwargs)  # 调用父类的 generate 方法
        except AttributeError as exc:
            # Expected exception: "AttributeError: '(object name)' object has no attribute 'past_key_values'"
            if "past_key_values" in str(exc):
                raise AttributeError(
                    "You tried to call `generate` with a decoding strategy that manipulates `past_key_values`. RWKV "
                    "doesn't have that attribute, try another generation strategy instead. For the available "
                    "generation strategies, check this doc: https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exc
        return gen_output

    def prepare_inputs_for_generation(self, input_ids, state=None, inputs_embeds=None, **kwargs):
        # only last token for inputs_ids if the state is passed along.
        if state is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)  # 只使用输入的最后一个标记作为生成输入

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and state is None:
            model_inputs = {"inputs_embeds": inputs_embeds}  # 如果传入了 inputs_embeds，则只在第一个生成步骤中使用它们
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["state"] = state  # 将状态信息添加到模型输入中
        return model_inputs

    @add_start_docstrings_to_model_forward(RWKV_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=RwkvCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,  # noqa
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, RwkvCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 如果 return_dict 为 None，则使用模型配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 rwkv 方法进行前向传播
        rwkv_outputs = self.rwkv(
            input_ids,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取 rwkv 输出中的隐藏状态
        hidden_states = rwkv_outputs[0]

        # 将隐藏状态传入头部模型计算 logits
        logits = self.head(hidden_states)

        # 初始化损失为 None
        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            # 将标签移动到与 logits 相同的设备上，以便进行模型并行计算
            labels = labels.to(logits.device)
            # 将 logits 向左移动一个位置，以对齐标签
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 使用交叉熵损失函数计算损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 如果 return_dict 为 False，则返回一个元组
        if not return_dict:
            output = (logits,) + rwkv_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 RwkvCausalLMOutput 对象
        return RwkvCausalLMOutput(
            loss=loss,
            logits=logits,
            state=rwkv_outputs.state,
            hidden_states=rwkv_outputs.hidden_states,
            attentions=rwkv_outputs.attentions,
        )
```