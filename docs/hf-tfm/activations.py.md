# `.\activations.py`

```
# 导入必要的库
import math
from collections import OrderedDict

# 导入 PyTorch 相关模块
import torch
from packaging import version
from torch import Tensor, nn

# 导入自定义的日志记录工具
from .utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义一个 PyTorch 模块，实现了一个高效的 GELU tanh 近似激活函数
class PytorchGELUTanh(nn.Module):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://arxiv.org/abs/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def __init__(self):
        super().__init__()
        # 检查所需的 PyTorch 版本是否满足要求
        if version.parse(torch.__version__) < version.parse("1.12.0"):
            raise ImportError(
                f"You are using torch=={torch.__version__}, but torch>=1.12.0 is required to use "
                "PytorchGELUTanh. Please upgrade torch."
            )

    def forward(self, input: Tensor) -> Tensor:
        # 使用 PyTorch 的内置函数实现 GELU tanh 近似激活
        return nn.functional.gelu(input, approximate="tanh")


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        # 实现 GELU 激活函数的计算公式
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        # 根据参数选择使用 Python 实现的 GELU 函数还是 PyTorch 内置的函数
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        # Python 实现的 GELU 函数
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        # 调用选择的 GELU 函数进行前向传播
        return self.act(input)


class FastGELUActivation(nn.Module):
    """
    Placeholder for a fast GELU activation function. Actual implementation is not provided here.
    """
    # 应用 GELU 近似函数，比 QuickGELU 更慢但更准确。参考：https://github.com/hendrycks/GELUs
    """

    # 前向传播函数，接收一个张量作为输入，返回处理后的张量
    def forward(self, input: Tensor) -> Tensor:
        # 使用 GELU 近似函数计算
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))
class QuickGELUActivation(nn.Module):
    """
    Applies a fast but approximate version of GELU activation.

    Reference: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        # Implementing GELU approximation using a sigmoid function
        return input * torch.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Module):
    """
    Applies GELU activation with output clipped to a specified range [min, max].

    This is useful for quantization purposes to handle negative values in the GELU spectrum.

    References:
    - https://arxiv.org/abs/2004.09602
    """

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        # Applying GELU activation and clipping the output
        return torch.clip(gelu(x), self.min, self.max)


class AccurateGELUActivation(nn.Module):
    """
    Applies a more accurate version of GELU activation compared to QuickGELU.

    Reference: https://github.com/hendrycks/GELUs

    Implemented in the context of MEGA (Moving Average Equipped Gated Attention).
    """

    def __init__(self):
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def forward(self, input: Tensor) -> Tensor:
        # Implementing the accurate GELU activation formula
        return 0.5 * input * (1 + torch.tanh(self.precomputed_constant * (input + 0.044715 * torch.pow(input, 3))))


class MishActivation(nn.Module):
    """
    Applies the Mish activation function, a self-regularized non-monotonic activation.

    Reference: https://arxiv.org/abs/1908.08681
    """

    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.9.0"):
            self.act = self._mish_python
        else:
            self.act = nn.functional.mish

    def _mish_python(self, input: Tensor) -> Tensor:
        # Implementing Mish activation using Python function
        return input * torch.tanh(nn.functional.softplus(input))

    def forward(self, input: Tensor) -> Tensor:
        # Applying Mish activation function
        return self.act(input)


class LinearActivation(nn.Module):
    """
    Applies the linear activation function, i.e., forwarding input directly to output.
    """

    def forward(self, input: Tensor) -> Tensor:
        # Identity function; returns input unchanged
        return input


class LaplaceActivation(nn.Module):
    """
    Applies an elementwise activation based on the Laplace function, introduced in MEGA for attention.

    This activation is inspired by squared ReLU but offers a bounded range and gradient for improved stability.

    Reference: https://arxiv.org/abs/2209.10655
    """
    """
    此方法用于计算正向传播过程中的操作，对输入进行标准化处理后，应用误差函数。
    :param input: 输入张量
    :param mu: 均值参数，默认为0.707107
    :param sigma: 标准差参数，默认为0.282095
    :return: 处理后的张量

    将输入张量标准化，减去均值 mu 并除以标准差乘以 sqrt(2.0)
    input = (input - mu).div(sigma * math.sqrt(2.0))
    应用误差函数，计算误差函数的正向传播结果，返回结果
    return 0.5 * (1.0 + torch.erf(input))
    """
# 定义一个自定义的激活函数 ReLUSquaredActivation，继承自 nn.Module
class ReLUSquaredActivation(nn.Module):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    # 定义前向传播方法，接受输入 input
    def forward(self, input):
        # 应用 ReLU 激活函数到输入
        relu_applied = nn.functional.relu(input)
        # 对经过 ReLU 激活后的结果进行平方操作
        squared = torch.square(relu_applied)
        # 返回平方后的结果作为输出
        return squared


# 定义一个名为 ClassInstantier 的类，继承自 OrderedDict
class ClassInstantier(OrderedDict):
    # 重写 __getitem__ 方法，接受键 key 作为输入
    def __getitem__(self, key):
        # 调用父类 OrderedDict 的 __getitem__ 方法获取键对应的值 content
        content = super().__getitem__(key)
        # 如果值 content 是一个元组，则将其解包为 cls 和 kwargs；否则将 cls 设为 content，kwargs 设为一个空字典
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        # 返回使用 cls 和 kwargs 创建的类实例
        return cls(**kwargs)


# 定义一个名为 ACT2CLS 的字典，将字符串映射到对应的激活函数类或者类与参数元组
ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": PytorchGELUTanh,
    "gelu_accurate": AccurateGELUActivation,
    "laplace": LaplaceActivation,
    "leaky_relu": nn.LeakyReLU,
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu2": ReLUSquaredActivation,  # 引用了之前定义的 ReLUSquaredActivation 激活函数类
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,  # SiLU 激活函数类，也称作 Swish
    "swish": nn.SiLU,  # 同上，SiLU 激活函数
    "tanh": nn.Tanh,
}

# 使用 ClassInstantier 类创建 ACT2FN 字典，将字符串映射为对应的激活函数类实例
ACT2FN = ClassInstantier(ACT2CLS)


# 定义一个函数 get_activation，接受一个激活函数字符串作为参数
def get_activation(activation_string):
    # 如果 activation_string 存在于 ACT2FN 字典中，则返回对应的激活函数类实例
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        # 否则抛出 KeyError，指示找不到对应的激活函数字符串
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")


# 创建几个全局变量，用于快速访问不同的激活函数实例
gelu_python = get_activation("gelu_python")
gelu_new = get_activation("gelu_new")
gelu = get_activation("gelu")
gelu_fast = get_activation("gelu_fast")
quick_gelu = get_activation("quick_gelu")
silu = get_activation("silu")
mish = get_activation("mish")
linear_act = get_activation("linear")
```