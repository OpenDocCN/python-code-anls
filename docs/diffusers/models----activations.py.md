# `.\diffusers\models\activations.py`

```py
# coding=utf-8  # 指定文件编码为 UTF-8
# Copyright 2024 HuggingFace Inc.  # 版权声明
#
# Licensed under the Apache License, Version 2.0 (the "License");  # 许可声明
# you may not use this file except in compliance with the License.  # 使用文件的合规性声明
# You may obtain a copy of the License at  # 许可证获取说明
#
#     http://www.apache.org/licenses/LICENSE-2.0  # 许可证的 URL
#
# Unless required by applicable law or agreed to in writing, software  # 免责声明
# distributed under the License is distributed on an "AS IS" BASIS,  # 以“现状”基础分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何明示或暗示的保证
# See the License for the specific language governing permissions and  # 查看许可证了解权限
# limitations under the License.  # 许可证下的限制条款

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的功能性 API
from torch import nn  # 从 PyTorch 导入神经网络模块

from ..utils import deprecate  # 从 utils 导入 deprecate 方法
from ..utils.import_utils import is_torch_npu_available  # 从 utils 导入检查 NPU 可用性的函数


if is_torch_npu_available():  # 如果 NPU 可用
    import torch_npu  # 导入 NPU 库

# 定义一个字典，映射激活函数名称到相应的 PyTorch 激活函数
ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),  # Swish 激活函数
    "silu": nn.SiLU(),  # SiLU 激活函数
    "mish": nn.Mish(),  # Mish 激活函数
    "gelu": nn.GELU(),  # GELU 激活函数
    "relu": nn.ReLU(),  # ReLU 激活函数
}

# 获取激活函数的帮助函数
def get_activation(act_fn: str) -> nn.Module:  # 定义函数，接受激活函数名称
    """Helper function to get activation function from string.  # 文档字符串，说明功能

    Args:  # 参数说明
        act_fn (str): Name of activation function.  # 激活函数名称

    Returns:  # 返回值说明
        nn.Module: Activation function.  # 返回对应的激活函数模块
    """

    act_fn = act_fn.lower()  # 将激活函数名称转换为小写
    if act_fn in ACTIVATION_FUNCTIONS:  # 如果激活函数在字典中
        return ACTIVATION_FUNCTIONS[act_fn]  # 返回对应的激活函数
    else:  # 否则
        raise ValueError(f"Unsupported activation function: {act_fn}")  # 抛出不支持的激活函数错误


class FP32SiLU(nn.Module):  # 定义 FP32SiLU 类，继承自 nn.Module
    r"""  # 文档字符串，描述该类
    SiLU activation function with input upcasted to torch.float32.  # SiLU 激活函数，输入转换为 float32
    """

    def __init__(self):  # 初始化方法
        super().__init__()  # 调用父类构造函数

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # 定义前向传播方法
        return F.silu(inputs.float(), inplace=False).to(inputs.dtype)  # 将输入转换为 float32，计算 SiLU，返回原数据类型


class GELU(nn.Module):  # 定义 GELU 类，继承自 nn.Module
    r"""  # 文档字符串，描述该类
    GELU activation function with tanh approximation support with `approximate="tanh"`.  # GELU 激活函数，支持 tanh 近似

    Parameters:  # 参数说明
        dim_in (`int`): The number of channels in the input.  # 输入通道数
        dim_out (`int`): The number of channels in the output.  # 输出通道数
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.  # 是否使用 tanh 近似
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.  # 是否在线性层中使用偏置
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):  # 初始化方法
        super().__init__()  # 调用父类构造函数
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)  # 创建线性层
        self.approximate = approximate  # 设置近似参数

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:  # 定义 GELU 方法
        if gate.device.type != "mps":  # 如果设备不是 MPS
            return F.gelu(gate, approximate=self.approximate)  # 计算并返回 GELU
        # mps: gelu is not implemented for float16  # 对于 MPS，float16 不支持 GELU
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)  # 转换为 float32 计算 GELU，返回原数据类型

    def forward(self, hidden_states):  # 定义前向传播方法
        hidden_states = self.proj(hidden_states)  # 通过线性层处理隐藏状态
        hidden_states = self.gelu(hidden_states)  # 计算 GELU 激活
        return hidden_states  # 返回激活后的隐藏状态


class GEGLU(nn.Module):  # 定义 GEGLU 类，继承自 nn.Module
    r"""  # 文档字符串，描述该类
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.  # GEGLU 激活函数的变种
    # 参数说明
    Parameters:
        dim_in (`int`): 输入通道的数量。
        dim_out (`int`): 输出通道的数量。
        bias (`bool`, defaults to True): 是否在线性层中使用偏置。

    # 初始化方法，设置输入和输出通道，及偏置选项
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入通道为 dim_in，输出通道为 dim_out * 2
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    # GELU 激活函数的实现
    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        # 检查当前设备类型是否为 MPS
        if gate.device.type != "mps":
            # 如果不是 MPS，直接返回 GELU 的计算结果
            return F.gelu(gate)
        # 对于 MPS：GELU 未对 float16 实现
        # 将 gate 转换为 float32 计算 GELU，然后再转换回原始数据类型
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    # 前向传播方法
    def forward(self, hidden_states, *args, **kwargs):
        # 如果传入额外参数或 kwargs 中包含 scale，给出弃用提示
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            # 弃用提示信息，告知用户 scale 参数将被忽略
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # 调用弃用函数，显示警告
            deprecate("scale", "1.0.0", deprecation_message)
        # 将隐藏状态通过线性层进行变换
        hidden_states = self.proj(hidden_states)
        # 检查是否可用 NPU
        if is_torch_npu_available():
            # 使用 torch_npu.npu_geglu 可以在 NPU 上更快且节省内存
            return torch_npu.npu_geglu(hidden_states, dim=-1, approximate=1)[0]
        else:
            # 将隐藏状态分为两部分：hidden_states 和 gate
            hidden_states, gate = hidden_states.chunk(2, dim=-1)
            # 返回 hidden_states 与 gate 的 GELU 结果的乘积
            return hidden_states * self.gelu(gate)
# 定义一个名为 SwiGLU 的类，继承自 nn.Module
class SwiGLU(nn.Module):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function. It's similar to `GEGLU`
    but uses SiLU / Swish instead of GeLU.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    # 初始化方法，接受输入和输出的维度及偏置参数
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        # 调用父类构造函数
        super().__init__()
        # 定义一个线性层，将输入通道映射到输出通道的两倍
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
        # 使用 SiLU 激活函数
        self.activation = nn.SiLU()

    # 前向传播方法
    def forward(self, hidden_states):
        # 通过线性层处理输入
        hidden_states = self.proj(hidden_states)
        # 将处理后的输出拆分为两部分
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        # 返回激活后的输出和门控的乘积
        return hidden_states * self.activation(gate)


# 定义一个名为 ApproximateGELU 的类，继承自 nn.Module
class ApproximateGELU(nn.Module):
    r"""
    The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
    [paper](https://arxiv.org/abs/1606.08415).

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    # 初始化方法，接受输入和输出的维度及偏置参数
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        # 调用父类构造函数
        super().__init__()
        # 定义一个线性层，将输入通道映射到输出通道
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    # 前向传播方法
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通过线性层处理输入
        x = self.proj(x)
        # 返回经过 sigmoid 函数调节后的输出
        return x * torch.sigmoid(1.702 * x)
```