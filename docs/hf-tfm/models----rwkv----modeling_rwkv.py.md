# `.\transformers\models\rwkv\modeling_rwkv.py`

```py
# 设置文件编码为 utf-8
# 版权声明
# 版权声明
# 根据 Apache License, Version 2.0 使用该文件
# 除非有适用法律要求或书面同意，否则按 "AS IS" 基础分发该软件
# 没有任何明示或暗示的保证或条件
# 请查看特定语言的许可证以了解权限和限制
"""PyTorch RWKV model."""  # PyTorch RWKV 模型

# 导入所需模块
import math  # 导入 math 模块
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 类
from pathlib import Path  # 从 pathlib 模块导入 Path 类
from typing import List, Optional, Tuple, Union  # 从 typing 模块导入 List, Optional, Tuple, Union 等类型

import torch  # 导入 torch 模块
import torch.utils.checkpoint  # 导入 torch.utils.checkpoint 模块
from torch import nn  # 从 torch 导入 nn 模块
from torch.nn import CrossEntropyLoss  # 从 torch.nn 导入 CrossEntropyLoss 类

from ...modeling_utils import PreTrainedModel  # 从 ...modeling_utils 模块导入 PreTrainedModel 类
from ...utils import (  # 从 ...utils 模块导入各种函数
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_bitsandbytes_available,
    is_ninja_available,
    is_torch_cuda_available,
    logging,
)
from .configuration_rwkv import RwkvConfig  # 从 .configuration_rwkv 模块导入 RwkvConfig 类

# 获取日志记录器
logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "RWKV/rwkv-4-169m-pile"  # 用于文档的检查点
_CONFIG_FOR_DOC = "RwkvConfig"  # 用于文档的配置项

RWKV_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 预训练模型的归档列表
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
    # 查看所有 RWKV 模型 https://huggingface.co/models?filter=rwkv
]

rwkv_cuda_kernel = None  # 初始化 rwkv_cuda_kernel 为 None

# 加载 wkv_cuda_kernel
def load_wkv_cuda_kernel(context_length):
    from torch.utils.cpp_extension import load as load_kernel  # 从 torch.utils.cpp_extension 导入 load 函数

    global rwkv_cuda_kernel

    kernel_folder = Path(__file__).resolve().parent.parent.parent / "kernels" / "rwkv"  # 获取 CUDA 内核的文件夹路径
    cuda_kernel_files = [kernel_folder / f for f in ["wkv_op.cpp", "wkv_cuda.cu", "wkv_cuda_bf16.cu"]]  # 获取 CUDA 内核文件列表

    # 当 rwkv_cuda_kernel 为空或者上下文长度改变时才加载内核
    if rwkv_cuda_kernel is not None and rwkv_cuda_kernel.max_seq_length == context_length:
        return

    logger.info(f"Loading CUDA kernel for RWKV at context length of {context_length}.")  # 记录加载 CUDA 内核的信息

    flags = [  # CUDA 编译标志列表
        "-res-usage",
        "--maxrregcount 60",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
        f"-DTmax={context_length}",
    ]
    rwkv_cuda_kernel = load_kernel(  # 加载 CUDA 内核
        name=f"wkv_{context_length}",
        sources=cuda_kernel_files,
        verbose=(logging.get_verbosity() == logging.DEBUG),
        extra_cuda_cflags=flags,
    )
    rwkv_cuda_kernel.max_seq_length = context_length  # 设置最大序列长度到 rwkv_cuda_kernel 中

# RwkvLinearAttention 类
class RwkvLinearAttention(torch.autograd.Function):
    @staticmethod  # 静态方法
    # 定义前向传播函数
    def forward(ctx, time_decay, time_first, key, value, state=None, return_state=False):
        # 获取输入张量的批大小、序列长度和隐藏层大小
        batch_size, seq_len, hidden_size = key.size()
        # 如果序列长度大于最大允许的长度，抛出错误
        if seq_len > rwkv_cuda_kernel.max_seq_length:
            raise ValueError(
                f"Cannot process a batch with {seq_len} tokens at the same time, use a maximum of "
                f"{rwkv_cuda_kernel.max_seq_length} with this model."
            )
        # 如果批大小乘以隐藏层大小不是32的倍数，抛出错误
        if batch_size * hidden_size % min(hidden_size, 32) != 0:
            raise ValueError(
                f"The product of batch size ({batch_size}) and hidden size ({hidden_size}) needs to be a round "
                f"multiple of {min(hidden_size, 32)}."
            )
    
        # 保存输入张量的数据类型
        ctx.input_dtype = key.dtype
    
        # 检查所有张量是否都在 CUDA 设备上
        if (
            time_decay.device.type != "cuda"
            or time_first.device.type != "cuda"
            or key.device.type != "cuda"
            or value.device.type != "cuda"
        ):
            raise ValueError("Calling the CUDA kernel for wkv attention requires all tensors to be on CUDA devices.")
    
        # 处理 time_decay 和 time_first 张量
        time_decay = -torch.exp(time_decay.float().contiguous())
        if key.dtype == torch.float16:
            time_first = time_first.float()
            key = key.float()
            value = value.float()
        time_first = time_first.contiguous()
        key = key.contiguous()
        value = value.contiguous()
    
        # 创建输出张量
        output = torch.empty_like(key, memory_format=torch.contiguous_format)
    
        # 如果需要返回状态或提供状态
        if return_state or state is not None:
            if state is None:
                # 初始化状态张量
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
                # 整理状态张量
                state = torch.cat([s.unsqueeze(2) for s in state], dim=2).contiguous()
            # 根据数据类型选择不同的前向函数
            if key.dtype == torch.bfloat16:
                forward_func = rwkv_cuda_kernel.forward_with_state_bf16
            else:
                forward_func = rwkv_cuda_kernel.forward_with_state
            # 调用前向函数并返回输出和状态
            forward_func(time_decay, time_first, key, value, output, state)
        else:
            # 调用不带状态的前向函数
            forward_func = rwkv_cuda_kernel.forward_bf16 if key.dtype == torch.bfloat16 else rwkv_cuda_kernel.forward
            forward_func(time_decay, time_first, key, value, output)
    
        # 保存中间结果以供反向传播使用
        ctx.save_for_backward(time_decay, time_first, key, value, output)
    
        # 如果提供了状态，拆分状态张量并返回
        if state is not None:
            state = [s.squeeze(2) for s in torch.chunk(state, 3, dim=2)]
    
        return output.to(ctx.input_dtype), state
    
    # 定义反向传播静态方法
    @staticmethod
    # g stands for grad
    # 定义反向传播函数，接受计算图上下文、输出的梯度和状态的梯度（可选）
    def backward(ctx, g_output, g_state=None):
        # 获取输入的数据类型
        input_dtype = ctx.input_dtype

        # 从上下文中提取保存的张量
        time_decay, time_first, key, value, output = ctx.saved_tensors
        # CUDA内核将填充这些张量。

        # 创建与time_decay形状相同的张量，用于存储time_decay的梯度
        g_time_decay = torch.empty_like(
            time_decay,
            memory_format=torch.contiguous_format,
            dtype=torch.bfloat16 if input_dtype == torch.bfloat16 else torch.float32,
        )
        # 创建与time_first形状相同的张量，用于存储time_first的梯度
        g_time_first = torch.empty_like(time_first, memory_format=torch.contiguous_format)
        # 创建与key形状相同的张量，用于存储key的梯度
        g_key = torch.empty_like(key, memory_format=torch.contiguous_format)
        # 创建与value形状相同的张量，用于存储value的梯度
        g_value = torch.empty_like(value, memory_format=torch.contiguous_format)

        # 如果输入数据类型是torch.float16，则将g_output转换为float类型
        if input_dtype == torch.float16:
            g_output = g_output.float()
        # 根据输入数据类型选择对应的CUDA内核函数
        backward_func = rwkv_cuda_kernel.backward_bf16 if input_dtype == torch.bfloat16 else rwkv_cuda_kernel.backward
        # 调用CUDA内核函数进行反向传播计算
        backward_func(
            time_decay,
            time_first,
            key,
            value,
            output,
            g_output.contiguous(),  # 确保g_output是连续内存
            g_time_decay,
            g_time_first,
            g_key,
            g_value,
        )

        # 返回计算得到的梯度，全部转换为输入数据类型
        return (
            g_time_decay.to(input_dtype),
            g_time_first.to(input_dtype),
            g_key.to(input_dtype),
            g_value.to(input_dtype),
            None,
            None,
        )
# 定义一个函数，使用线性加权函数实现自注意力机制，当CPU性能不足时使用
# 参数 time_decay: 时间衰减值
# 参数 time_first: 时间加权值
# 参数 key: 关键字
# 参数 value: 值
# 参数 state: 状态变量，可选，默认为None
# 参数 return_state: 返回状态标识，可选，默认为False
def rwkv_linear_attention_cpu(time_decay, time_first, key, value, state=None, return_state=False):
    # 对于CPU回退。如果不在`torch.no_grad`范围内执行，将会比自定义的CUDA内核更慢，可能需要更多的内存
    _, seq_length, _ = key.size()  # 获取关键字的维度信息
    output = torch.zeros_like(key)  # 初始化输出张量，与关键字形状相同

    if state is None:  # 如果状态变量为空
        num_state = torch.zeros_like(key[:, 0], dtype=torch.float32)  # 初始化数值状态
        den_state = torch.zeros_like(key[:, 0], dtype=torch.float32)  # 初始化分母状态
        max_state = torch.zeros_like(key[:, 0], dtype=torch.float32) - 1e38  # 初始化最大状态
    else:  # 如果状态变量不为空
        num_state, den_state, max_state = state  # 获取状态变量
    # 为了数值稳定性
    #     real_numerator_state = num_state * torch.exp(max_state)
    #     real_denominator_state = den_state * torch.exp(max_state)

    time_decay = -torch.exp(time_decay)  # 计算时间衰减值的负指数

    for current_index in range(seq_length):  # 遍历序列长度
        current_key = key[:, current_index].float()  # 获取当前关键字
        current_value = value[:, current_index]  # 获取当前值

        # 在时间 t 计算 wkv
        max_for_output = torch.maximum(max_state, current_key + time_first)  # 计算最大输出值
        e1 = torch.exp(max_state - max_for_output)  # 计算指数项
        e2 = torch.exp(current_key + time_first - max_for_output)  # 计算指数项
        numerator = e1 * num_state + e2 * current_value  # 计算分子
        denominator = e1 * den_state + e2  # 计算分母
        output[:, current_index] = (numerator / denominator).to(output.dtype)  # 计算输出值

        # 更新下一次迭代的状态
        max_for_state = torch.maximum(max_state + time_decay, current_key)  # 计算最大状态
        e1 = torch.exp(max_state + time_decay - max_for_state)  # 计算指数项
        e2 = torch.exp(current_key - max_for_state)  # 计算指数项
        num_state = e1 * num_state + e2 * current_value  # 更新数值状态
        den_state = e1 * den_state + e2  # 更新分母状态
        max_state = max_for_state  # 更新最大状态

    if return_state or state is not None:  # 如果需要返回状态，或者状态不为空
        state = [num_state, den_state, max_state]  # 更新状态变量

    return output, state  # 返回��出和状态变量


def rwkv_linear_attention(time_decay, time_first, key, value, state=None, return_state=False):
    no_cuda = any(t.device.type != "cuda" for t in [time_decay, time_first, key, value])  # 检查是否存在CUDA设备
    # 仅对一个标记启动CUDA内核的情况下，实际上会更慢（在CPU版本中这种情况下没有for循环）
    one_token = key.size(1) == 1  # 检查是否仅有一个token
    if rwkv_cuda_kernel is None or no_cuda or one_token:  # 如果未找到CUDA内核，或者不存在CUDA设备，或者只有一个token
        return rwkv_linear_attention_cpu(time_decay, time_first, key, value, state=state, return_state=return_state)  # 调用CPU版本函数
    else:  # 如果存在CUDA内核，并且有CUDA设备，并且不只有一个token
        return RwkvLinearAttention.apply(time_decay, time_first, key, value, state, return_state)  # 调用CUDA内核


class RwkvSelfAttention(nn.Module):
    # 初始化函数，接受配置和层id作为参数
    def __init__(self, config, layer_id=0):
        # 调用父类初始化函数
        super().__init__()
        
        # 保存配置信息
        self.config = config
        
        # 检查是否已加载 CUDA 内核并且其最大序列长度与配置中的上下文长度相同
        kernel_loaded = rwkv_cuda_kernel is not None and rwkv_cuda_kernel.max_seq_length == config.context_length
        
        # 检查是否可用 Ninja 编译器以及是否可用 Torch CUDA 并且尚未加载内核
        if is_ninja_available() and is_torch_cuda_available() and not kernel_loaded:
            try:
                # 尝试加载 RWKV 注意力的自定义 CUDA 内核
                load_wkv_cuda_kernel(config.context_length)
            except Exception:
                logger.info("Could not load the custom CUDA kernel for RWKV attention.")
        
        # 保存层id
        self.layer_id = layer_id
        
        # 保存隐藏层大小和注意力隐藏层大小
        hidden_size = config.hidden_size
        attention_hidden_size = (
            config.attention_hidden_size if config.attention_hidden_size is not None else hidden_size
        )
        self.attention_hidden_size = attention_hidden_size
    
        # 初始化学习参数
        self.time_decay = nn.Parameter(torch.empty(attention_hidden_size))
        self.time_first = nn.Parameter(torch.empty(attention_hidden_size))
        self.time_mix_key = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_value = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, hidden_size))
        
        # 时间维度的移位操作
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        # 线性变换层，将隐藏层映射到注意力隐藏层
        self.key = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.output = nn.Linear(attention_hidden_size, hidden_size, bias=False)
    
    # 提取关键、数值、接受率函数，用于前向传播
    # TODO: 可能加入 jit 编译，否则将操作放入 forward 函数内
    def extract_key_value(self, hidden, state=None):
        # 将隐藏状态与上一时间步的隐藏状态混合以产生关键、数值、接受率
        if hidden.size(1) == 1 and state is not None:
            shifted = state[1][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[1][:, :, self.layer_id]
        
        key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
        value = hidden * self.time_mix_value + shifted * (1 - self.time_mix_value)
        receptance = hidden * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance)
        
        # 线性变换
        key = self.key(key)
        value = self.value(value)
        receptance = torch.sigmoid(self.receptance(receptance))
        
        # 更新状态
        if state is not None:
            state[1][:, :, self.layer_id] = hidden[:, -1]
        
        # 返回接受率、关键、数值、状态
        return receptance, key, value, state
    ```  
    # 前向传播函数，接收隐藏层状态和状态（默认为空），是否使用缓存
    def forward(self, hidden, state=None, use_cache=False):
        # 提取关键字和数值向量、键、值、状态
        receptance, key, value, state = self.extract_key_value(hidden, state=state)
        # 如果状态不为空，则取出隐藏状态中第二个元组中对应的当前层状态
        layer_state = tuple(s[:, :, self.layer_id] for s in state[2:]) if state is not None else None
        # 调用 rwkv_linear_attention 函数计算注意力权重和值，返回状态
        rwkv, layer_state = rwkv_linear_attention(
            self.time_decay,
            self.time_first,
            key,
            value,
            state=layer_state,
            return_state=use_cache,
        )

        # 如果获得了新的状态，则更新当前层的状态值
        if layer_state is not None:
            state[2][:, :, self.layer_id] = layer_state[0]
            state[3][:, :, self.layer_id] = layer_state[1]
            state[4][:, :, self.layer_id] = layer_state[2]

        # 返回输出结果和状态
        return self.output(receptance * rwkv), state
```py  
# 定义一个名为RwkvFeedForward的类，继承自nn.Module类
class RwkvFeedForward(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config  # 保存传入的config参数
        self.layer_id = layer_id  # 保存传入的layer_id参数
        hidden_size = config.hidden_size  # 从config参数中获取hidden_size
        intermediate_size = (
            config.intermediate_size if config.intermediate_size is not None else 4 * config.hidden_size
        )  # 根据config参数中的intermediate_size计算中间层的size，如果没有则默认为4倍的hidden_size

        # 创建用于时间处理的ZeroPad2d实例
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        # 创建时间mix的关键参数，用于更新隐藏状态
        self.time_mix_key = nn.Parameter(torch.empty(1, 1, hidden_size))
        # 创建时间mix的接受参数，用于更新隐藏状态
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, hidden_size))

        # 创建线性变换层，用于提取关键信息
        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 创建线性变换层，用于提取接受信息
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        # 创建线性变换层，用于提取值信息
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

    # 前向传播函数，接受hidden和state作为输入
    def forward(self, hidden, state=None):
        # 如果hidden的第二个维度为1且state不为空，则将shifted设置为state中的对应部分
        if hidden.size(1) == 1 and state is not None:
            shifted = state[0][:, :, self.layer_id]
        else:
            # 否则将shifted设置为hidden按时间维度进行shift后的结果
            shifted = self.time_shift(hidden)
            # 如果state不为空，则将shifted的第一维度设置为state中的对应部分
            if state is not None:
                shifted[:, 0] = state[0][:, :, self.layer_id]
        # 计算关键信息的加权和
        key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
        # 计算接受信息的加权和
        receptance = hidden * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance)

        # 对关键信息进行线性变换和ReLU激活
        key = torch.square(torch.relu(self.key(key)))
        # 对接受信息进行线性变换
        value = self.value(key)
        # 对接受信息进行Sigmoid激活
        receptance = torch.sigmoid(self.receptance(receptance))

        # 如果state不为空，则将hidden的最后一个时间步保存到state中的对应部分
        if state is not None:
            state[0][:, :, self.layer_id] = hidden[:, -1]

        # 返回经过权重处理后的value和receptance，以及更新后的state
        return receptance * value, state


# 定义一个名为RwkvBlock的类，继承自nn.Module类
class RwkvBlock(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config  # 保存传入的config参数
        self.layer_id = layer_id  # 保存传入的layer_id参数

        # 如果layer_id为0，则创建用于预处理的LayerNorm实例
        if layer_id == 0:
            self.pre_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # 创建两个用于隐藏状态的LayerNorm实例
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        # 创建自注意力机制的实例
        self.attention = RwkvSelfAttention(config, layer_id)
        # 创建前向传播的实例
        self.feed_forward = RwkvFeedForward(config, layer_id)

    # 前向传播函数，接受hidden、state、use_cache和output_attentions作为输入
    def forward(self, hidden, state=None, use_cache=False, output_attentions=False):
        # 如果layer_id为0，则对hidden进行预处理
        if self.layer_id == 0:
            hidden = self.pre_ln(hidden)

        # 调用自注意力机制计算attention和更新state
        attention, state = self.attention(self.ln1(hidden), state=state, use_cache=use_cache)
        hidden = hidden + attention  # 更新hidden

        # 调用前向传播计算feed_forward和更新state
        feed_forward, state = self.feed_forward(self.ln2(hidden), state=state)
        hidden = hidden + feed_forward  # 更新hidden

        outputs = (hidden, state)  # 将更新后的hidden和state组成outputs
        if output_attentions:
            outputs += (attention,)  # 如果output_attentions为True，则将attention也加入outputs中
        else:
            outputs += (None,)  # 否则将None加入outputs中

        return outputs  # 返回outputs


class RwkvPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = RwkvConfig  # 设置config_class为RwkvConfig
    base_model_prefix = "rwkv"  # 设置base_model_prefix为"rwkv"
    _no_split_modules = ["RwkvBlock"]  # 设置_no_split_modules为["RwkvBlock"]
    # 保留使用 FP32 模块列表
    _keep_in_fp32_modules = ["time_decay", "time_first"]
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    
    # 初始化模型权重
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, RwkvSelfAttention):
            # 获取层标识、隐藏层数量、隐藏层大小和注意力隐藏层大小
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size
            attention_hidden_size = module.attention_hidden_size
    
            # 计算用于时间权重的比率
            ratio_0_to_1 = layer_id / (num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0
    
            # 创建时间权重并转换为张量
            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.time_mix_key.dtype,
                device=module.time_mix_key.device,
            )
            time_weight = time_weight[None, None, :]
    
            # 计算衰减速度
            decay_speed = [
                -5 + 8 * (h / (attention_hidden_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                for h in range(attention_hidden_size)
            ]
            decay_speed = torch.tensor(decay_speed, dtype=module.time_decay.dtype, device=module.time_decay.device)
            zigzag = (
                torch.tensor(
                    [(i + 1) % 3 - 1 for i in range(attention_hidden_size)],
                    dtype=module.time_first.dtype,
                    device=module.time_first.device,
                )
                * 0.5
            )
    
            # 无梯度下更新模型权重
            with torch.no_grad():
                module.time_decay.data = decay_speed
                module.time_first.data = torch.ones_like(module.time_first * math.log(0.3) + zigzag)
    
                module.time_mix_key.data = torch.pow(time_weight, ratio_1_to_almost0)
                module.time_mix_value.data = torch.pow(time_weight, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
                module.time_mix_receptance.data = torch.pow(time_weight, 0.5 * ratio_1_to_almost0)
        elif isinstance(module, RwkvFeedForward):
            # 获取层标识、隐藏层数量和隐藏层大小
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size
    
            # 计算用于时间权重的比率
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0
    
            # 创建时间权重并转换为张量
            time_weight = torch.tensor(
                [i / hidden_size for i in range(hidden_size)],
                dtype=module.time_mix_key.dtype,
                device=module.time_mix_key.device,
            )
            time_weight = time_weight[None, None, :]
    
            # 无梯���下更新模型权重
            with torch.no_grad():
                module.time_mix_key.data = torch.pow(time_weight, ratio_1_to_almost0)
                module.time_mix_receptance.data = torch.pow(time_weight, ratio_1_to_almost0)
# 使用 dataclass 装饰器定义 RwkvOutput 类，用于表示 RWKV 模型的输出结果
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
    # 定义 last_hidden_state 属性，表示模型的最后一层的隐藏状态
    last_hidden_state: torch.FloatTensor = None
    # 定义 state 属性，表示模型在最后一个时间步的状态
    state: Optional[List[torch.FloatTensor]] = None
    # 定义 hidden_states 属性，表示模型在每个层的隐藏状态
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义 attentions 属性，表示模型的注意力权重
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 使用 dataclass 装饰器定义 RwkvCausalLMOutput 类，用于表示因果语言模型（或自回归模型）的输出结果
@dataclass
class RwkvCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.
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
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # Optional type annotation for language modeling loss
    loss: Optional[torch.FloatTensor] = None
    # Type annotation for prediction scores of the language modeling head
    logits: torch.FloatTensor = None
    # Optional type annotation for the state of the model at the last time step
    state: Optional[List[torch.FloatTensor]] = None
    # Optional type annotation for hidden states of the model
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # Optional type annotation for attention weights after the attention softmax
    attentions: Optional[Tuple[torch.FloatTensor]] = None
RWKV_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RwkvConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


RWKV_INPUTS_DOCSTRING = r"""

"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
            # 定义输入的token索引张量，用于表示输入序列中的token在词汇表中的位置

        attention_mask (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            This is currently not used by `RwkvModel`, but will be supported in the future.

            [What are attention masks?](../glossary#attention-mask)
            # 定义用于避免在填充token索引上进行注意力计算的mask张量

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
            # 可选参数，用于直接传递嵌入表示，而不是通过`input_ids`传递。当你想要更多控制如何将`input_ids`索引转换成相关向量时，这会很有用。

        state (tuple of five `torch.FloatTensor` of shape `(batch_size, hidden_size, num_hidden_layers)`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
            # 可选参数，传递之后，模型将在所有块中使用先前的状态，这将给出提供的`input_ids`的输出，就好像模型将`state_input_ids + input_ids` 作为上下文添加。

        use_cache (`bool`, *optional*):
            If set to `True`, the last state is returned and can be used to quickly generate the next logits.
            # 如果设置为`True`，则返回最后的状态，并可用来快速生成下一个logits。

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
            # 是否返回所有注意力层的注意力张量。更多细节请查看返回的张量中的`attentions`。

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
            # 是否返回所有层的隐藏状态。更多细节请查看返回的张量中的`hidden_states`。

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            # 是否返回一个包含`~utils.ModelOutput`的字典，而不是普通的元组。
# 定义一个 RWKV 模型类，继承自 RwkvPreTrainedModel
@add_start_docstrings(
    "The bare RWKV Model transformer outputting raw hidden-states without any specific head on top.",
    RWKV_START_DOCSTRING,
)
class RwkvModel(RwkvPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化词嵌入层
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # 初始化块列表，包含多个 RWKV 块
        self.blocks = nn.ModuleList([RwkvBlock(config, layer_id=idx) for idx in range(config.num_hidden_layers)])
        # 初始化输出层的 LayerNorm
        self.ln_out = nn.LayerNorm(config.hidden_size)

        # 初始化是否对层进行了重新缩放的标志
        self.layers_are_rescaled = False
        # 初始化梯度检查点标志
        self.gradient_checkpointing = False

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入词嵌入层
    def get_input_embeddings(self):
        return self.embeddings

    # 设置输入词嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    # 前向传播方法，接受多种参数，返回模型输出
    @add_start_docstrings_to_model_forward(RWKV_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=RwkvOutput,
        config_class=_CONFIG_FOR_DOC,
    )
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
    # 重新缩放神经网络中的层，仅适用于推断阶段
    def _rescale_layers(self):
        # 如果层已经被重新缩放，且当前不处于训练阶段，则直接返回
        if self.layers_are_rescaled == (not self.training):
            return
        # 每隔一定步骤重新缩放层权重
        if self.config.rescale_every > 0:
            # 在无需计算梯度的情况下进行操作
            with torch.no_grad():
                # 遍历神经网络中的每个块
                for block_id, block in enumerate(self.blocks):
                    # 如果处于训练阶段
                    if self.training:
                        # 放大注意力输出和前馈传播值的权重
                        block.attention.output.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                        block.feed_forward.value.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                    else:
                        # 处理量化统计信息
                        if hasattr(block.attention.output.weight, "SCB"):
                            block.attention.output.weight.SCB.div_(2 ** int(block_id // self.config.rescale_every))
                            block.feed_forward.value.weight.SCB.div_(2 ** int(block_id // self.config.rescale_every))
                        elif hasattr(block.attention.output.weight, "quant_state"):
                            # 反量化并重新缩放注意力输出和前馈传播值的权重
                            self._bnb_4bit_dequantize_and_rescale(block.attention.output, block_id)
                            self._bnb_4bit_dequantize_and_rescale(block.feed_forward.value, block_id)
                        else:
                            # 重新缩放注意力输出和前馈传播值的权重
                            block.attention.output.weight.div_(2 ** int(block_id // self.config.rescale_every))
                            block.feed_forward.value.weight.div_(2 ** int(block_id // self.config.rescale_every))

        # 更新层已重新缩放的标志
        self.layers_are_rescaled = not self.training

    # 反量化和重新缩放给定层的权重
    def _bnb_4bit_dequantize_and_rescale(self, target_layer, block_id):
        r"""
        执行给定层权重的反量化和重新缩放操作。操作后，层将再次被量化。
        """
        # 如果未安装 bitsandbytes 库，则抛出 ImportError
        if not is_bitsandbytes_available():
            raise ImportError("Please install bitsandbytes to use this method.")
        # 导入 bitsandbytes 库
        import bitsandbytes as bnb

        # 反量化 4 位权重数据
        dequant_weights = bnb.functional.dequantize_4bit(target_layer.weight.data, target_layer.weight.quant_state)

        # 重新缩放权重
        dequant_weights.div_(2 ** int(block_id // self.config.rescale_every))

        # 重新量化模型：
        # 首先将其转移到 CPU，然后再转回设备上
        # 这会导致性能开销 :/
        # 我们设置 requires_grad=False，因为无法在 4 位参数上计算梯度，也为了避免与 bnb 的 bug
        quant_weight = bnb.nn.Params4bit(dequant_weights.to("cpu"), requires_grad=False).to(dequant_weights.device)
        # 设置目标层的权重为重新量化后的权重
        setattr(target_layer, "weight", quant_weight)
# 使用 add_start_docstrings 装饰器添加模型文档字符串，并指定 RWKV 模型的开始文档字符串
@add_start_docstrings(
    """
    The RWKV Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    RWKV_START_DOCSTRING,
)
# 定义 RwkvForCausalLM 类，继承自 RwkvPreTrainedModel 类
class RwkvForCausalLM(RwkvPreTrainedModel):
    # 定义 _tied_weights_keys 属性，用于指定与输入嵌入相关联的权重键名
    _tied_weights_keys = ["head.weight"]

    # 定义初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 RwkvModel 实例
        self.rwkv = RwkvModel(config)
        # 创建线性层，用于语言建模的头部，其权重与输入嵌入相关联
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义获取输出嵌入的方法
    def get_output_embeddings(self):
        return self.head

    # 定义设置输出嵌入的方法
    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    # 定义为生成准备输入的方法
    def prepare_inputs_for_generation(self, input_ids, state=None, inputs_embeds=None, **kwargs):
        # 如果传递了状态，则仅使用输入 ID 的最后一个令牌
        if state is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        # 如果传递了 inputs_embeds，并且未传递状态，则仅在第一代步骤中使用它们
        if inputs_embeds is not None and state is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 将状态添加到模型输入中
        model_inputs["state"] = state
        return model_inputs

    # 使用 add_start_docstrings_to_model_forward 装饰器添加模型前向传播的文档字符串，并指定输入的文档字符串
    @add_start_docstrings_to_model_forward(RWKV_INPUTS_DOCSTRING)
    # 使用 add_code_sample_docstrings 装饰器添加代码示例的文档字符串，指定检查点、输出类型和配置类
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=RwkvCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义前向传播方法
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
        # 确定是否返回字典类型的输出，如果未指定，则使用模型配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用Rwkv模块处理输入，获取输出
        rwkv_outputs = self.rwkv(
            input_ids,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 从Rwkv模块输出中提取隐藏状态
        hidden_states = rwkv_outputs[0]

        # 使用头部模块处理隐藏状态，得到logits
        logits = self.head(hidden_states)

        loss = None
        if labels is not None:
            # 将标签移至正确的设备以启用模型并行计算
            labels = labels.to(logits.device)
            # 移位以使 < n 的标记预测 n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 将标记展平
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            # 如果不返回字典类型的输出，则返回元组
            output = (logits,) + rwkv_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回RwkvCausalLMOutput对象，其中包含损失、logits、状态、隐藏状态和注意力权重
        return RwkvCausalLMOutput(
            loss=loss,
            logits=logits,
            state=rwkv_outputs.state,
            hidden_states=rwkv_outputs.hidden_states,
            attentions=rwkv_outputs.attentions,
        )
```