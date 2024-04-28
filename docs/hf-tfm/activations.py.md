# `.\transformers\activations.py`

```py
# 导入必要的模块和库
import math  # 导入数学模块
import warnings  # 导入警告模块
from collections import OrderedDict  # 导入有序字典模块

import torch  # 导入 PyTorch 库
from packaging import version  # 导入版本管理模块
from torch import Tensor, nn  # 从 PyTorch 中导入张量和神经网络模块

from .utils import logging  # 从当前包中导入日志工具模块


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


class PytorchGELUTanh(nn.Module):
    """
    一个快速的 C 实现的 GeLU 激活函数的 tanh 近似版本。参见 https://arxiv.org/abs/1606.08415。

    这个实现等价于 NewGELU 和 FastGELU，但速度要快得多。然而，由于舍入误差，它不是精确的数值匹配。
    """

    def __init__(self):
        super().__init__()
        # 检查 PyTorch 版本是否符合要求
        if version.parse(torch.__version__) < version.parse("1.12.0"):
            raise ImportError(
                f"You are using torch=={torch.__version__}, but torch>=1.12.0 is required to use "
                "PytorchGELUTanh. Please upgrade torch."
            )

    def forward(self, input: Tensor) -> Tensor:
        # 使用 tanh 近似实现的 GeLU 激活函数
        return nn.functional.gelu(input, approximate="tanh")


class NewGELUActivation(nn.Module):
    """
    实现 Google BERT 仓库中当前的 GELU 激活函数（与 OpenAI GPT 相同）。另请参阅高斯误差线性单元论文: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        # 新的 GELU 激活函数实现
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class GELUActivation(nn.Module):
    """
    在 Google BERT 仓库初始创建时的 GELU 激活函数的原始实现。对于信息: OpenAI GPT 的 GELU 稍有不同（并且给出略微不同的结果）:
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) 现在在 nn.functional 中用 C 写成了。
    也请参阅高斯误差线性单元论文: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        # 根据选择使用 Python 实现的 GELU 还是 PyTorch 提供的 C 实现
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        # Python 实现的 GELU 激活函数
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        # 使用选择的实现计算 GELU 激活函数
        return self.act(input)


class FastGELUActivation(nn.Module):
    """
    """
    # 应用比 QuickGELU 更准确但更慢的 GELU 近似方法。参考：https://github.com/hendrycks/GELUs
    """

    # 前向传播函数，接受一个张量输入，返回一个张量输出
    def forward(self, input: Tensor) -> Tensor:
        # 使用 GELU 近似方法计算输出
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))
class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        # 应用快速但略有不准确的 GELU 近似函数
        return input * torch.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Module):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """

    def __init__(self, min: float, max: float):
        # 初始化 ClippedGELUActivation 类，限制 GeLU 输出范围在 [min, max] 之间
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        # 返回限制在指定范围内的 GeLU 输出
        return torch.clip(gelu(x), self.min, self.max)


class AccurateGELUActivation(nn.Module):
    """
    Applies GELU approximation that is faster than default and more accurate than QuickGELU. See:
    https://github.com/hendrycks/GELUs

    Implemented along with MEGA (Moving Average Equipped Gated Attention)
    """

    def __init__(self):
        # 初始化 AccurateGELUActivation 类，应用比默认更快且比 QuickGELU 更准确的 GELU 近似函数
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def forward(self, input: Tensor) -> Tensor:
        # 返回更准确的 GELU 近似函数的输出
        return 0.5 * input * (1 + torch.tanh(self.precomputed_constant * (input + 0.044715 * torch.pow(input, 3)))


class SiLUActivation(nn.SiLU):
    def __init__(self, *args, **kwargs):
        # 警告：SiLUActivation 类已被弃用，并将在 v4.39 中移除，请使用 nn.SiLU 代替
        warnings.warn(
            "The SiLUActivation class has been deprecated and will be removed in v4.39. Please use nn.SiLU instead.",
        )
        super().__init__(*args, **kwargs)


class MishActivation(nn.Module):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """

    def __init__(self):
        # 初始化 MishActivation 类，应用 Mish 激活函数
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.9.0"):
            self.act = self._mish_python
        else:
            self.act = nn.functional.mish

    def _mish_python(self, input: Tensor) -> Tensor:
        # 使用 Python 实现的 Mish 激活函数
        return input * torch.tanh(nn.functional.softplus(input))

    def forward(self, input: Tensor) -> Tensor:
        # 返回使用 Mish 激活函数的输出
        return self.act(input)


class LinearActivation(nn.Module):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def forward(self, input: Tensor) -> Tensor:
        # 应用线性激活函数，即直接将输入转发到输出
        return input
class LaplaceActivation(nn.Module):
    """
    Applies elementwise activation based on Laplace function, introduced in MEGA as an attention activation. See
    https://arxiv.org/abs/2209.10655

    Inspired by squared relu, but with bounded range and gradient for better stability
    """

    def forward(self, input, mu=0.707107, sigma=0.282095):
        # 将输入标准化到均值为 mu，标准差为 sigma 的 Laplace 分布上
        input = (input - mu).div(sigma * math.sqrt(2.0))
        # 应用 Laplace 激活函数
        return 0.5 * (1.0 + torch.erf(input))


class ReLUSquaredActivation(nn.Module):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    def forward(self, input):
        # 应用 ReLU 激活函数
        relu_applied = nn.functional.relu(input)
        # 对 ReLU 激活后的结果取平方
        squared = torch.square(relu_applied)
        return squared


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        # 获取键对应的值
        content = super().__getitem__(key)
        # 如果值是元组，则第一个元素是类，第二个元素是关键字参数字典；否则，默认关键字参数为空字典
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        # 实例化类，传入关键字参数
        return cls(**kwargs)


# 定义激活函数名到对应激活函数类的映射
ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": PytorchGELUTanh,
    "gelu_accurate": AccurateGELUActivation,
    "laplace": LaplaceActivation,  # Laplace 激活函数
    "leaky_relu": nn.LeakyReLU,
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu2": ReLUSquaredActivation,  # ReLU 平方激活函数
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}
# 使用 ClassInstantier 类创建激活函数名到激活函数类的映射
ACT2FN = ClassInstantier(ACT2CLS)


def get_activation(activation_string):
    # 如果激活函数名存在于 ACT2FN 映射中，则返回对应的激活函数类
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        # 否则，引发 KeyError 异常
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")


# 为了向后兼容：from activations import gelu_python
# 获取 gelu_python 激活函数类
gelu_python = get_activation("gelu_python")
# 获取 gelu_new 激活函数类
gelu_new = get_activation("gelu_new")
# 获取 gelu 激活函数类
gelu = get_activation("gelu")
# 获取 gelu_fast 激活函数类
gelu_fast = get_activation("gelu_fast")
# 获取 quick_gelu 激活函数类
quick_gelu = get_activation("quick_gelu")
# 获取 silu 激活函数类
silu = get_activation("silu")
# 获取 mish 激活函数类
mish = get_activation("mish")
# 获取 linear 激活函数类
linear_act = get_activation("linear")
```