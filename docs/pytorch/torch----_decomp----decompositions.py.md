# `.\pytorch\torch\_decomp\decompositions.py`

```
# 设置允许未类型化的定义，在使用mypy进行类型检查时忽略未注释类型的函数
# 导入 functools 模块，用于高阶函数的支持
# 导入 numbers 模块，用于数值相关的操作
# 导入 operator 模块，用于函数式编程中的运算符操作
# 导入 sys 模块，提供对解释器相关的功能访问
# 导入 Enum 类，用于定义枚举类型
from enum import Enum
# 导入 functools 模块中的 partial 和 reduce 函数，用于创建部分应用函数和进行迭代操作的简化
from functools import partial, reduce
# 导入 itertools 模块中的 chain 和 product 函数，用于创建迭代器对象和迭代器的笛卡尔积
from itertools import chain, product
# 导入 typing 模块中的类型相关工具
from typing import Any, Callable, cast, Iterable, List, Optional, Tuple, Union

# 导入 PyTorch 深度学习库
import torch
# 导入 torch._meta_registrations 模块，用于元数据注册
import torch._meta_registrations
# 导入 torch._prims 模块，提供基本原语操作的接口
import torch._prims as prims
# 导入 torch._prims_common 模块，提供通用的原语操作函数
import torch._prims_common as utils
# 导入 torch.nn.functional 模块，提供神经网络中的函数操作
import torch.nn.functional as F
# 从 torch 模块中导入 sym_float 和 sym_int 符号化类型
from torch import sym_float, sym_int, Tensor
# 导入 torch._decomp 模块中的 register_decomposition 函数
from torch._decomp import register_decomposition
# 导入 torch._higher_order_ops.out_dtype 模块中的 out_dtype 函数
from torch._higher_order_ops.out_dtype import out_dtype
# 导入 torch._prims_common 模块中的一些数据类型和内存格式建议函数
from torch._prims_common import (
    IntLike,
    NumberType,
    suggest_memory_format,
    TensorLike,
    TensorSequenceType,
)
# 导入 torch._prims_common.wrappers 模块中的一些数据类型转换和封装函数
from torch._prims_common.wrappers import (
    _maybe_convert_to_dtype,
    _maybe_resize_out,
    _safe_copy_out,
    out_wrapper,
)
# 从 torch.utils 模块中导入 _pytree 模块，用于处理树状数据结构
from torch.utils import _pytree as pytree
# 从 torch.utils._pytree 模块中导入 tree_map 函数，用于对树状结构进行映射操作

# 定义 DispatchKey 类型为 torch._C.DispatchKey，用于表示分发键
DispatchKey = torch._C.DispatchKey  # type: ignore[attr-defined]

# 声明空列表 __all__，用于标识模块中公开的函数或对象
__all__: List[str] = []

# 使用 torch._ops.ops.aten 模块中的 aten 变量，表示基本张量操作的集合
aten = torch._ops.ops.aten

# 定义枚举类型 Reduction，包括 NONE、MEAN 和 SUM，用于表示归约操作的类型
class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


# type_casts 函数用于包装一个分解操作，并根据提供的策略进行类型推广逻辑
# 函数签名为 Callable 类型，接受一个函数 f、类型推广策略 type_promotion 和一个布尔值 compute_dtype_only
def type_casts(
    f: Callable,
    type_promotion: utils.ELEMENTWISE_TYPE_PROMOTION_KIND,
    compute_dtype_only: bool = False,
):
    # 内部函数 inner 对函数 f 进行包装
    @functools.wraps(f)
    def inner(*args, **kwargs):
        # 使用 pytree.arg_tree_leaves 函数获取参数和关键字参数中的所有 Tensor 对象
        flat_args = [
            x for x in pytree.arg_tree_leaves(*args, **kwargs) if isinstance(x, Tensor)
        ]
        # 调用 utils.elementwise_dtypes 函数获取计算和结果的数据类型
        computation_dtype, result_dtype = utils.elementwise_dtypes(
            *flat_args, type_promotion_kind=type_promotion
        )

        # 内部函数 increase_prec 和 decrease_prec 分别用于提升和降低精度
        def increase_prec(x):
            if isinstance(x, Tensor):
                return x.to(computation_dtype)
            else:
                return x

        def decrease_prec(x):
            if isinstance(x, Tensor):
                return x.to(result_dtype)
            else:
                return x

        # 调用原始函数 f，并根据需要进行精度处理
        r = f(*tree_map(increase_prec, args), **tree_map(increase_prec, kwargs))
        # 如果 compute_dtype_only 为 True，则返回计算的数据类型结果
        if compute_dtype_only:
            return r
        else:
            # 否则对结果进行精度降低处理后返回
            return tree_map(decrease_prec, r)

    return inner


# 创建 compute_only_pw_cast_for_opmath 函数，部分应用 type_casts 函数以支持运算操作的类型推广
compute_only_pw_cast_for_opmath = partial(
    type_casts,
    type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    compute_dtype_only=True,
)
# 创建 pw_cast_for_opmath 函数，部分应用 type_casts 函数以支持默认的类型推广
pw_cast_for_opmath = partial(
    type_casts, type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)
# 创建 pw_cast_for_int_to_real 函数，部分应用 type_casts 函数以支持整数到实数的类型推广
pw_cast_for_int_to_real = partial(
    type_casts, type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)


# _unsqueeze_to_dim 函数用于将输入张量 x 扩展到指定的维度 dim
# 参数 x 为输入张量，dim 为目标维度
# 返回值为扩展后的张量
def _unsqueeze_to_dim(x: Tensor, dim: int) -> Tensor:
    # 对于输入张量 x，扩展其维度，直到维度达到 dim
    for _ in range(dim - x.dim()):
        # 在最后一个维度上添加一个维度，使得 x 的维度增加
        x = x.unsqueeze(-1)
    # 返回处理后的张量 x
    return x
@register_decomposition(aten.tanh_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def tanh_backward(out_grad: Tensor, y: Tensor):
    return out_grad * (1 - y * y).conj_physical()


@register_decomposition(aten.sigmoid_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def sigmoid_backward(out_grad: Tensor, y: Tensor):
    return out_grad * (y * (1 - y)).conj_physical()


@register_decomposition(aten.softplus_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def softplus_backward(out_grad: Tensor, x: Tensor, beta: float, threshold: float):
    # 计算指数函数的值
    z = (x * beta).exp()
    # 使用条件语句处理超过阈值的情况
    return torch.where((x * beta) > threshold, out_grad, out_grad * z / (z + 1.0))


@register_decomposition(aten.elu_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def elu_backward(
    grad_output: Tensor,
    alpha: float,
    scale: float,
    input_scale: float,
    is_result: bool,
    self_or_result: Tensor,
):
    # 计算负系数、正系数和输入系数
    negcoef = alpha * scale
    poscoef = scale
    negiptcoef = input_scale
    # 根据条件返回不同的梯度计算方式
    if is_result:
        return torch.where(
            self_or_result <= 0,
            grad_output * negiptcoef * (self_or_result + negcoef),
            grad_output * poscoef,
        )
    else:
        return torch.where(
            self_or_result <= 0,
            grad_output * negiptcoef * negcoef * torch.exp(self_or_result * negiptcoef),
            grad_output * poscoef,
        )


@register_decomposition([aten.fill.Scalar])
def fill_scalar(self, value):
    # 返回与输入相同形状的全值张量
    return torch.full_like(self, value)


@register_decomposition([aten.fill.Tensor])
def fill_tensor(self, value: Tensor):
    # 检查值张量是否为零维，并进行相应的填充操作
    torch._check(
        value.dim() == 0,
        lambda: f"fill only supports 0-dimension value tensor but got tensor with {value.dim()} dimensions",
    )
    return aten.copy(self, value)


@register_decomposition(aten.hardsigmoid)
@out_wrapper()
@pw_cast_for_opmath
def hardsigmoid(self: Tensor) -> Tensor:
    # 使用硬切sigmoid函数
    return torch.clamp(torch.clamp(self + 3, min=0), max=6) / 6


@register_decomposition(aten.hardsigmoid_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def hardsigmoid_backward(grad_output: Tensor, self: Tensor):
    # 使用条件语句处理硬切sigmoid反向传播的情况
    return torch.where(
        (self > -3.0) & (self < 3.0),
        grad_output * (1.0 / 6.0),
        0.0,
    )


@register_decomposition(aten.hardtanh_backward)
@out_wrapper("grad_input")
def hardtanh_backward(
    grad_output: Tensor, self: Tensor, min_val: float, max_val: float
):
    # 使用条件语句处理hardtanh反向传播的情况
    return torch.where((self <= min_val) | (self >= max_val), 0.0, grad_output)


@register_decomposition(aten.hardswish)
@out_wrapper()
@pw_cast_for_opmath
def hardswish(self: Tensor) -> Tensor:
    # 使用硬切swish函数
    return self * torch.clamp(torch.clamp(self + 3, min=0), max=6) / 6


@register_decomposition(aten.hardswish_backward)
@out_wrapper()
@pw_cast_for_opmath
def hardswish_backward(grad_output: Tensor, self: Tensor) -> Tensor:
    # 使用条件语句处理硬切swish反向传播的情况
    return torch.where(
        self < -3,
        0.0,
        torch.where(self <= 3, grad_output * ((self / 3) + 0.5), grad_output),
    )
@register_decomposition(aten.threshold_backward)
@out_wrapper("grad_input")
# 定义一个函数，用于计算阈值反向传播的梯度
def threshold_backward(grad_output: Tensor, self: Tensor, threshold: float):
    # 使用 torch.where 函数根据条件返回不同的值，当 self <= threshold 时返回 0，否则返回 grad_output
    return torch.where(self <= threshold, 0, grad_output)


@register_decomposition(aten.leaky_relu_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
# 定义一个函数，用于计算泄漏整流线性单元（Leaky ReLU）的反向传播梯度
def leaky_relu_backward(
    grad_output: Tensor, self: Tensor, negative_slope: float, self_is_result: bool
):
    # 使用 torch.where 函数根据条件返回不同的值，当 self > 0 时返回 grad_output，否则返回 grad_output * negative_slope
    return torch.where(self > 0, grad_output, grad_output * negative_slope)


@register_decomposition(aten.gelu_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
# 定义一个函数，用于计算 GELU 激活函数的反向传播梯度
def gelu_backward(grad: Tensor, self: Tensor, approximate: str = "none"):
    M_SQRT2 = 1.41421356237309504880
    M_SQRT1_2 = 0.70710678118654752440
    M_2_SQRTPI = 1.12837916709551257390
    if approximate == "tanh":
        kBeta = M_SQRT2 * M_2_SQRTPI * 0.5
        kKappa = 0.044715
        x_sq = self * self
        x_cube = x_sq * self
        inner = kBeta * (self + kKappa * x_cube)
        tanh_inner = torch.tanh(inner)

        left = 0.5 * self
        right = 1 + tanh_inner

        left_derivative = 0.5 * right

        tanh_derivative = 1 - tanh_inner * tanh_inner
        inner_derivative = kBeta * (1 + 3 * kKappa * x_sq)
        right_derivative = left * tanh_derivative * inner_derivative

        # 返回梯度乘以左右导数之和
        return grad * (left_derivative + right_derivative)
    else:
        kAlpha = M_SQRT1_2
        kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5
        cdf = 0.5 * (1 + torch.erf(self * kAlpha))
        pdf = kBeta * torch.exp(self * self * -0.5)
        # 返回梯度乘以累积分布函数（CDF）和概率密度函数（PDF）之和
        return grad * (cdf + self * pdf)


@register_decomposition(aten.mish_backward)
@pw_cast_for_opmath
# 定义一个函数，用于计算 Mish 激活函数的反向传播梯度
def mish_backward(grad_output: Tensor, input: Tensor):
    input_tanh_softplus = torch.tanh(F.softplus(input))
    input_sigmoid = torch.sigmoid(input)
    out = input * input_sigmoid * (1 - input_tanh_softplus * input_tanh_softplus)
    # 返回梯度乘以 tanh_softplus 和 out 之和
    return grad_output * (input_tanh_softplus + out)


@register_decomposition(aten.silu)
@out_wrapper()
@pw_cast_for_opmath
# 定义一个函数，用于计算 SiLU 激活函数
def silu(self: Tensor) -> Tensor:
    # 返回输入与 sigmoid 函数作用的乘积
    return self * torch.sigmoid(self)


@register_decomposition(aten.silu_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
# 定义一个函数，用于计算 SiLU 激活函数的反向传播梯度
def silu_backward(grad_output: Tensor, self: Tensor) -> Tensor:
    sigmoid = 1 / (1 + torch.exp(-self))
    # 返回梯度乘以 sigmoid 函数、1 加上 self 与 sigmoid 乘积
    return grad_output * sigmoid * (1 + self * (1 - sigmoid))


@register_decomposition(aten._prelu_kernel)
# 定义一个函数，用于计算 PReLU 激活函数的内核
def _prelu_kernel(self: Tensor, weight: Tensor) -> Tensor:
    # 使用 torch.where 函数根据条件返回不同的值，当 self > 0 时返回 self，否则返回 weight * self
    return torch.where(self > 0, self, weight * self)


@register_decomposition(aten._prelu_kernel_backward)
# 定义一个函数，用于计算 PReLU 激活函数内核的反向传播梯度
def _prelu_kernel_backward(
    grad_output: Tensor,
    self: Tensor,
    weight: Tensor,
) -> Tuple[Tensor, Tensor]:
    # 使用 torch.where 函数根据条件返回不同的值，计算输入梯度和权重梯度
    input_grad = torch.where(self > 0, grad_output, weight * grad_output)
    weight_grad = torch.where(self > 0, 0.0, self * grad_output)
    return (input_grad, weight_grad)


@register_decomposition(aten.rrelu_with_noise)
@aten.rrelu_with_noise.default.py_impl(DispatchKey.AutogradCUDA)
@out_wrapper()
# 注册一个 RReLU 激活函数的实现
# 注册一个装饰器，用于将函数标记为 PyTorch 的 opmath 操作，并进行类型转换
@register_decomposition(aten.mse_loss)
# 为函数添加输出包装器，用于处理输出结果
@out_wrapper()
# 对函数应用数学运算的类型转换
@pw_cast_for_opmath
# 声明定义均方误差损失函数
def mse_loss(
    # 定义函数的参数：self 代表对象本身，通常在类方法中使用；target 是函数的目标张量参数；
    # reduction 是一个整数，默认值为 Reduction.MEAN.value，用于指定损失函数的计算方式。
# 定义一个函数，计算均方误差损失的反向传播
@register_decomposition(aten.mse_loss_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def mse_loss_backward(
    grad_output: Tensor, input: Tensor, target: Tensor, reduction: int
):
    # 根据减少操作的类型计算正则化系数
    norm = 2.0 / input.numel() if reduction == Reduction.MEAN.value else 2.0
    # 计算损失函数对输入的导数
    return norm * (input - target) * grad_output


# 注册一个函数，用于计算平滑 L1 损失的反向传播
@register_decomposition(aten.smooth_l1_loss_backward.default)
@pw_cast_for_opmath
def smooth_l1_loss_backward(
    grad_output: Tensor, self: Tensor, target: Tensor, reduction: int, beta: float
):
    # 根据减少操作的类型计算正则化系数
    norm = 1.0 / self.numel() if reduction == Reduction.MEAN.value else 1.0
    # 计算输入与目标之间的差异
    x = self - target
    abs_x = torch.abs(x)
    # 计算梯度的归一化值
    norm_grad = norm * grad_output
    # 返回根据条件选择的结果
    return torch.where(
        abs_x < beta,
        norm_grad * x / beta,
        norm_grad * torch.sign(x),
    )


# 注册一个函数，用于平滑 L1 损失的反向传播，并输出到指定的梯度输入张量中
@register_decomposition(aten.smooth_l1_loss_backward.grad_input)
@pw_cast_for_opmath
def smooth_l1_loss_backward_out(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int,
    beta: float,
    grad_input: Tensor,
):
    # 计算平滑 L1 损失的反向传播结果
    result = smooth_l1_loss_backward(grad_output, self, target, reduction, beta)
    # 确保输出张量的大小匹配
    _maybe_resize_out(grad_input, result.shape)
    # 将结果安全复制到输出张量中，并确保与指定的数据类型精确匹配
    return _safe_copy_out(copy_from=result, copy_to=grad_input, exact_dtype=True)


# 注册一个函数，用于 Huber 损失的反向传播
@register_decomposition(aten.huber_loss_backward.default)
@pw_cast_for_opmath
def huber_loss_backward(
    grad_output: Tensor, self: Tensor, target: Tensor, reduction: int, delta: float
):
    # 根据减少操作的类型计算正则化系数
    norm = 1.0 / self.numel() if reduction == Reduction.MEAN.value else 1.0
    # 计算输入与目标之间的差异
    x = self - target
    # 根据条件选择不同的结果返回
    return torch.where(
        x < -delta,
        -norm * grad_output * delta,
        torch.where(x > delta, norm * grad_output * delta, norm * x * grad_output),
    )


# 注册一个函数，用于 Huber 损失的反向传播，并输出到指定的梯度输入张量中
@register_decomposition(aten.huber_loss_backward.out)
@pw_cast_for_opmath
def huber_loss_backward_out(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int,
    delta: float,
    grad_input: Tensor,
):
    # 计算 Huber 损失的反向传播结果
    result = huber_loss_backward(grad_output, self, target, reduction, delta)
    # 确保输出张量的大小匹配
    _maybe_resize_out(grad_input, result.shape)
    # 将结果安全复制到输出张量中，并确保与指定的数据类型精确匹配
    return _safe_copy_out(copy_from=result, copy_to=grad_input, exact_dtype=True)


# 定义一个函数，计算负对数似然损失的反向传播
def _nll_loss_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
    total_weight: Tensor,
) -> Tensor:
    # 确定通道维度的索引
    channel_dim = 0 if self.dim() < 2 else 1
    # 如果使用的是均值减少模式，则对梯度输出进行归一化处理
    if reduction == Reduction.MEAN.value:
        grad_output = grad_output / total_weight

    # 在指定维度上增加一个维度，用于后续操作
    target = target.unsqueeze(channel_dim)
    # 将目标张量中等于ignore_index的值替换为0，其它值保持不变
    safe_target = torch.where(target != ignore_index, target, 0)
    # 创建一个与当前张量相同形状的零张量作为梯度输入
    grad_input = torch.zeros_like(self)
    # 根据safe_target在指定维度上对grad_input进行填充操作，填充值为-1.0
    grad_input = torch.scatter(grad_input, channel_dim, safe_target, -1.0)

    # 如果梯度输入的维度大于梯度输出的维度且梯度输出的维度大于0，则在指定维度上增加一个维度
    if grad_input.dim() > grad_output.dim() > 0:
        grad_output = grad_output.unsqueeze(channel_dim)

    # 如果权重不为None，则根据权重的形状调整梯度输出的形状，并对梯度输出进行相应的加权处理
    if weight is not None:
        new_shape = [1 for _ in range(self.dim())]
        new_shape[channel_dim] = weight.shape[0]
        weight = weight.reshape(new_shape)
        grad_output = grad_output * weight

    # 将梯度输出中等于ignore_index的位置的值替换为0，其它位置保持不变
    grad_output = torch.where(target != ignore_index, grad_output, 0)

    # 返回经过填充后的梯度输入与加权后的梯度输出的乘积作为最终的梯度
    return grad_input * grad_output
# 注册函数的装饰器，将aten.glu_backward函数注册到某个处理器
@register_decomposition(aten.glu_backward)
# 将输出参数名称设置为"grad_input"
@out_wrapper("grad_input")
# 应用于操作数数学的类型转换装饰器
@pw_cast_for_opmath
def glu_backward(grad_output: Tensor, self: Tensor, dim: int) -> Tensor:
    # 断言self的维度大于0，否则抛出异常
    assert self.dim() > 0, "glu does not support 0-dimensional tensors"
    # 规范化维度dim，并赋值给wrap_dim
    wrap_dim = utils.canonicalize_dim(self.dim(), dim)
    # 获取self在wrap_dim维度上的大小
    nIn = self.size(wrap_dim)
    # 断言nIn是偶数，否则抛出异常
    assert (
        nIn % 2 == 0
    ), f"Halving dimension must be even, but dimension {wrap_dim} is size {nIn}"
    # 计算输入的大小
    inputSize = nIn // 2
    # 将self在wrap_dim维度上切片，获取前半部分数据
    firstHalf = self.narrow(wrap_dim, 0, inputSize)
    # 将self在wrap_dim维度上切片，获取后半部分数据
    secondHalf = self.narrow(wrap_dim, inputSize, inputSize)
    # 计算第一部分的梯度输入，使用torch.sigmoid函数对secondHalf进行操作
    gradInputFirstHalf = torch.sigmoid(secondHalf)
    # 计算第二部分的梯度输入，使用torch.sigmoid函数计算的结果结合其他变量进行计算
    gradInputSecondHalf = (
        (1.0 - gradInputFirstHalf) * gradInputFirstHalf * firstHalf * grad_output
    )
    # 更新gradInputFirstHalf为其与grad_output的乘积
    gradInputFirstHalf = gradInputFirstHalf * grad_output
    # 在wrap_dim维度上拼接gradInputFirstHalf和gradInputSecondHalf，返回结果
    return torch.cat([gradInputFirstHalf, gradInputSecondHalf], dim=wrap_dim)


# 注册函数的装饰器，将aten.nll_loss_backward函数注册到某个处理器
@register_decomposition(aten.nll_loss_backward)
# 将输出参数名称设置为"grad_input"
@out_wrapper("grad_input")
def nll_loss_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
    total_weight: Tensor,
) -> Tensor:
    # 断言self的维度在0到2之间，否则抛出异常
    assert 0 <= self.dim() <= 2, "input tensor should be 1D or 2D"
    # 断言target的维度不大于1，否则抛出异常
    assert (
        target.dim() <= 1
    ), "0D or 1D target tensor expected, multi-target not supported"

    # 检查是否没有批次维度，并且self和target的形状匹配，否则抛出异常
    no_batch_dim = self.dim() == 1 and target.dim() == 0
    assert no_batch_dim or (
        self.shape[0] == target.shape[0]
    ), f"size mismatch (got input: {self.shape}, target: {target.shape})"
    # 断言total_weight是单元素张量，否则抛出异常
    assert total_weight.numel() == 1, (
        "expected total_weight to be a single element tensor, got: ",
        f"{total_weight.shape} ({total_weight.numel()} elements)",
    )

    # 断言weight为None或者weight的元素数量等于self的最后一个维度大小，否则抛出异常
    assert (
        weight is None or weight.numel() == self.shape[-1]
    ), "weight tensor should be defined either for all or no classes"

    # 如果reduction为Reduction.NONE.value并且self的维度为2，则进一步检查grad_output的维度和形状
    if reduction == Reduction.NONE.value and self.dim() == 2:
        assert grad_output.dim() == 1 and grad_output.shape[0] == self.shape[0], (
            f"Expected a tensor of dimension 1 and tensor.size[0] == {self.shape[0]} but "
            f"got: dimension {grad_output.dim()} and tensor.size[0] == {grad_output.shape[0]}"
        )
    else:
        # 否则，断言grad_output是单元素张量，否则抛出异常
        assert (
            grad_output.dim() <= 1 and grad_output.numel() == 1
        ), f"Expected a single element grad_output tensor, but got: {grad_output.shape}"

    # 调用内部函数_nll_loss_backward，并返回其结果
    return _nll_loss_backward(
        grad_output, self, target, weight, reduction, ignore_index, total_weight
    )


# 注册函数的装饰器，将aten.nll_loss2d_backward函数注册到某个处理器
@register_decomposition(aten.nll_loss2d_backward)
# 将输出参数名称设置为"grad_input"
@out_wrapper("grad_input")
def nll_loss2d_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
    total_weight: Tensor,
) -> Tensor:
    # 断言self的维度为4，否则抛出异常
    assert (
        self.dim() == 4
    ), f"only batches of spatial inputs supported (4D tensors), but got input of dimension: {self.dim()}"

    # 断言target的维度为3
    assert (
        target.dim() == 3
    # 检查目标张量是否为3维张量（空间目标），否则抛出错误
    ), f"only batches of spatial targets supported (3D tensors) but got targets of dimension: {target.dim()}"

    # 检查输入张量和目标张量的形状是否匹配，否则抛出错误
    assert (
        self.shape[0] == target.shape[0]
        and self.shape[2] == target.shape[1]
        and self.shape[3] == target.shape[2]
    ), f"size mismatch (got input: {self.shape}, target: {target.shape}"

    # 检查总权重张量是否只有一个元素，否则抛出错误
    assert total_weight.numel() == 1, (
        "expected total_weight to be a single element tensor, "
        f"got: {total_weight.shape} ( {total_weight.numel()}, elements)"
    )

    # 调用内部函数进行负对数似然损失的反向传播计算，并返回结果
    return _nll_loss_backward(
        grad_output, self, target, weight, reduction, ignore_index, total_weight
    )
@register_decomposition(aten.binary_cross_entropy)
@out_wrapper()
@pw_cast_for_opmath
def binary_cross_entropy(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: int = Reduction.MEAN.value,
) -> Tensor:
    # 定义二元交叉熵损失函数
    # 计算损失，根据公式：(target - 1) * log(1 - self) - target * log(self)
    loss = (target - 1) * torch.maximum(
        torch.log1p(-self), self.new_full((), -100)
    ) - target * torch.maximum(torch.log(self), self.new_full((), -100))
    # 如果提供了权重，则按权重调整损失
    if weight is not None:
        loss = loss * weight
    # 应用损失函数的缩减方式，并返回计算得到的损失
    return apply_loss_reduction(loss, reduction)


@register_decomposition(aten.binary_cross_entropy_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def binary_cross_entropy_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: int = Reduction.MEAN.value,
) -> Tensor:
    # 定义二元交叉熵损失函数的反向传播
    EPSILON = 1e-12
    # 计算损失梯度
    result = grad_output * (self - target) / torch.clamp(self * (1 - self), min=EPSILON)
    # 如果提供了权重，则按权重调整梯度
    if weight is not None:
        result = result * weight
    # 如果使用的是平均缩减方式，则将梯度除以元素数量
    if reduction == Reduction.MEAN.value:
        result = result / self.numel()
    # 返回计算得到的梯度
    return result


@register_decomposition(aten.soft_margin_loss)
@out_wrapper()
@pw_cast_for_opmath
def soft_margin_loss(
    input: Tensor,
    target: Tensor,
    reduction: int = Reduction.MEAN.value,
) -> Tensor:
    # 定义软间隔损失函数
    # 计算损失，根据公式：log(1 + exp(-input * target))
    loss = torch.log1p(torch.exp(-input * target))
    # 应用损失函数的缩减方式，并返回计算得到的损失
    return apply_loss_reduction(loss, reduction)


@register_decomposition(aten.soft_margin_loss_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def soft_margin_loss_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int = Reduction.MEAN.value,
) -> Tensor:
    # 定义软间隔损失函数的反向传播
    # 计算输入梯度
    grad_input = target * grad_output * (torch.sigmoid(target * self) - 1)
    # 如果使用的是平均缩减方式，则将梯度除以元素数量
    if reduction == Reduction.MEAN.value:
        grad_input = grad_input / self.numel()
    # 返回计算得到的输入梯度
    return grad_input


@register_decomposition(aten.dist)
@out_wrapper()
def dist(input: Tensor, other: Tensor, p: float = 2):
    # 计算输入张量与另一个张量之间的距离
    return aten.norm(input - other, p=p)


@register_decomposition(aten._euclidean_dist)
@out_wrapper()
def _euclidean_dist(x1: Tensor, x2: Tensor) -> Tensor:
    # 计算欧氏距离
    x1_norm = x1.pow(2).sum(-1, True)
    x1_pad = torch.ones_like(x1_norm, memory_format=torch.contiguous_format)
    x2_norm = x2.pow(2).sum(-1, True)
    x2_pad = torch.ones_like(x2_norm, memory_format=torch.contiguous_format)
    x1_ = torch.cat([x1.mul(-2), x1_norm, x1_pad], -1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], -1)
    # 计算结果矩阵并对其进行修剪、开平方
    result = x1_.matmul(x2_.mT)
    return result.clamp_min(0).sqrt()


@register_decomposition(aten.slice_backward)
@out_wrapper()
def slice_backward(
    grad_output: Tensor,
    input_sizes: List[int],
    dim: int,
    start: int,
    end: int,
    step: int,
):
    # 定义切片操作的反向传播
    # 创建与输入尺寸相同的零张量作为梯度输入
    grad_input = grad_output.new_zeros(input_sizes)
    # 返回计算得到的梯度输入
    return grad_input
    # 使用 Torch 提供的 slice_scatter 方法对梯度输入 grad_input 进行切片散列操作，
    # 使用梯度输出 grad_output 对切片后的结果进行填充
    # 在指定维度 dim 上，从 start 到 end（不包括 end），步长为 step
    return torch.slice_scatter(grad_input, grad_output, dim, start, end, step)
# 注册切片函数的分解器，用于处理torch.Tensor类型的对象
@register_decomposition(aten.slice.Tensor)
def slice_forward(
    # 定义切片操作的前向函数，接受自身Tensor对象，维度dim，默认起始和结束位置，步长为step
    self: Tensor,
    dim: int = 0,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
):
    from torch.fx.experimental.symbolic_shapes import (
        guard_size_oblivious,
        statically_known_true,
    )

    # 获取Tensor的维度数
    ndim = self.dim()
    # 如果Tensor为0维，则抛出运行时错误
    if ndim == 0:
        raise RuntimeError("slice() cannot be applied to a 0-dim tensor.")
    # 规范化dim参数，确保在有效范围内
    dim = utils.canonicalize_dim(self.dim(), dim)
    # 获取Tensor各维度的大小和步长
    sizes = list(self.size())
    strides = list(self.stride())

    # 若步长小于等于0，则抛出运行时错误
    if step <= 0:
        raise RuntimeError("slice step must be positive")

    # 初始化起始和结束位置的值
    start_val = start if start is not None else 0
    end_val = end if end is not None else sys.maxsize  # 2^63 - 1

    # 处理起始位置为负数的情况
    if start_val < 0:
        start_val += sizes[dim]

    # 处理结束位置为负数的情况
    if end_val < 0:
        end_val += sizes[dim]

    # 确保起始位置在有效范围内
    if start_val < 0:
        start_val = 0
    elif start_val > sizes[dim]:
        start_val = sizes[dim]

    # 确保结束位置在起始位置之后
    if end_val < start_val:
        end_val = start_val
    # 如果结束位置为最大值或超过当前维度的大小，则将其设置为当前维度的大小
    elif statically_known_true(end_val == sys.maxsize) or guard_size_oblivious(
        end_val > sizes[dim]
    ):
        end_val = sizes[dim]

    # 计算存储偏移量
    storage_offset = self.storage_offset() + start_val * strides[dim]
    # 计算切片后的长度
    len = end_val - start_val
    # 更新切片后的维度大小和步长
    sizes[dim] = (len + step - 1) // step
    strides[dim] *= step

    # 如果Tensor是量化的，则抛出未实现的错误
    if self.is_quantized:
        raise NotImplementedError(
            "Slice decomposition for quantized tensors aren't implemented"
        )
    else:
        # 否则，返回切片后的Tensor对象
        return self.as_strided(sizes, strides, storage_offset)


def _normalize_start_end(
    x: Tensor, dim: int, start: Optional[int], end: Optional[int]
) -> Tuple[int, int]:
    """
    Normalize start and end such that both are in the range
    [0, x.get_size()[dim]] and start <= end.
    """
    # 获取指定维度的大小
    dim_size = x.shape[dim]

    def clamp_wrap(val, lower, upper, default) -> int:
        # 辅助函数：将val限制在指定的下界和上界之间，如果为None，则返回默认值
        if val is None:
            return default
        if val < 0:
            val = val + dim_size
        return min(max(val, lower), upper)

    # 规范化起始和结束位置，确保它们在有效范围内，并且起始位置小于等于结束位置
    start = clamp_wrap(start, 0, dim_size, 0)
    end = clamp_wrap(end, start, dim_size, dim_size)
    return start, end


# 尽管torch._refs中没有，因为aten.index在aten._unsafe_masked_index中使用，没有分解
@register_decomposition(aten.slice_scatter)
@out_wrapper()
def slice_scatter(
    input: Tensor,
    src: Tensor,
    dim: int = 0,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
):
    # 规范化维度参数
    dim = utils.canonicalize_dim(input.ndim, dim)
    # 获取指定维度的大小
    dim_size = input.shape[dim]
    # 规范化起始和结束位置
    start, end = _normalize_start_end(input, dim, start, end)

    # 创建与输入Tensor相同形状的src Tensor
    src_size = list(input.shape)
    src_size[dim] = (end - start + (step - 1)) // step
    src = src.expand(src_size)

    # 如果起始为0且结束为维度大小且步长为1，则直接返回src的克隆
    if start == 0 and end == dim_size and step == 1:
        return src.clone()

    # 创建索引列表，并生成设备相关的索引
    indices = [None] * input.dim()
    idx = torch.arange(dim_size, device=input.device)
    # 计算在给定维度上的索引值，根据起始位置和步长计算
    indices[dim] = (idx - start) // step

    # 创建一个全为 True 的掩码张量，设备为 input 的设备，数据类型为布尔型
    mask = torch.ones(dim_size, device=input.device, dtype=torch.bool)

    # 如果起始位置不为 0，则更新掩码为当前索引大于等于起始位置的逻辑与
    if start != 0:
        mask = torch.logical_and(mask, idx >= start)

    # 如果结束位置不等于维度大小，则更新掩码为当前索引小于结束位置的逻辑与
    if end != dim_size:
        mask = torch.logical_and(mask, idx < end)

    # 如果步长不为 1，则更新掩码为当前索引满足从起始位置开始以步长递增的逻辑与
    if step != 1:
        mask = torch.logical_and(mask, (idx - start) % step == 0)

    # 创建一个与 input 张量相同维度的掩码形状，其中在指定维度上为 -1
    mask_shape = [1] * input.dim()
    mask_shape[dim] = -1
    mask = mask.view(mask_shape)

    # 使用掩码对输入张量进行条件索引，返回结果
    return aten.where(mask, aten._unsafe_masked_index(src, mask, indices, 0), input)
# 注册 im2col 函数的分解方法，将其注册到相应的框架中
@register_decomposition(aten.im2col)
# 使用 out_wrapper 装饰器修饰函数
@out_wrapper()
# 定义 im2col 函数，将输入张量转换为列矩阵
def im2col(
    input: Tensor,
    kernel_size: List[int],
    dilation: List[int],
    padding: List[int],
    stride: List[int],
) -> Tensor:
    # 检查 kernel_size 必须是长度为 2 的列表，因为只支持 2D 卷积核
    torch._check(len(kernel_size) == 2, lambda: "im2col(): only 2D kernel supported")
    # 检查 dilation 必须是长度为 2 的列表，因为只支持 2D 膨胀
    torch._check(len(dilation) == 2, lambda: "im2col(): only 2D dilation supported")



# 实用函数，用于实现 im2col 和 col2im 的转换
def _im2col_col2im_indices_along_dim(
    input_d, kernel_d, dilation_d, padding_d, stride_d, device
):
    """Utility function to implement im2col and col2im"""
    # 计算 blocks_d 的大小，用于确定输出矩阵的尺寸
    blocks_d = input_d + padding_d * 2 - dilation_d * (kernel_d - 1)

    # 创建一个指定设备的偏函数 arange_kw，用于生成整数序列
    arange_kw = partial(torch.arange, dtype=torch.int64, device=device)

    # 在维度 d 上按步长 stride_d 滑动卷积核，找到起始索引
    blocks_d_indices = arange_kw(0, blocks_d, stride_d).unsqueeze(0)

    # 对卷积核应用 dilation_d，找到沿维度 d 的索引
    kernel_grid = arange_kw(0, kernel_d * dilation_d, dilation_d).unsqueeze(-1)

    # 广播并将卷积核的起始位置索引与 kernel_grid 沿维度 d 相加，得到沿维度 d 的块索引
    return blocks_d_indices + kernel_grid



# 注册 _softmax_backward_data 函数的分解方法，将其注册到相应的框架中
@register_decomposition(aten._softmax_backward_data)
# 使用 out_wrapper 装饰器修饰函数
@out_wrapper("grad_input")
# 对于操作数学的计算，仅进行点对点类型转换
@compute_only_pw_cast_for_opmath
# _softmax_backward_data 函数的定义，用于计算 softmax 反向传播的数据梯度
def _softmax_backward_data(
    grad_output: Tensor, output: Tensor, dim: int, input_dtype: torch.dtype
):
    # 计算新的梯度输出，是梯度输出与输出的点积
    new_grad_output = grad_output * output
    # 计算梯度输入，是新的梯度输出减去输出乘以沿 dim 维度的总和
    grad_input = new_grad_output - output * torch.sum(
        new_grad_output, dim=dim, keepdim=True
    )

    # CPU 内核不尊重输入数据类型，但下面的检查对于元张量并不适用
    # if grad_output.device == torch.device("cpu"):
    #     return grad_input.contiguous()

    # 将梯度输入转换为输入数据类型后返回，保证连续性
    return _cast_grad_to_input_dtype(grad_output, grad_input, input_dtype).contiguous()



# 注册 _log_softmax_backward_data 函数的分解方法，将其注册到相应的框架中
@register_decomposition(aten._log_softmax_backward_data)
# 使用 out_wrapper 装饰器修饰函数
@out_wrapper()
# 对于操作数学的计算，仅进行点对点类型转换
@compute_only_pw_cast_for_opmath
# _log_softmax_backward_data 函数的定义，用于计算 log_softmax 反向传播的数据梯度
def _log_softmax_backward_data(
    grad_output: Tensor, output: Tensor, dim: int, input_dtype: torch.dtype
):
    # 计算梯度输入，是梯度输出减去输出的指数乘以沿 dim 维度的总和
    grad_input = grad_output - torch.exp(output) * torch.sum(
        grad_output, dim=dim, keepdim=True
    )
    # 将梯度输入转换为输入数据类型后返回
    return _cast_grad_to_input_dtype(grad_output, grad_input, input_dtype)
    # 检查 padding 是否为长度为 2 的元组
    torch._check(len(padding) == 2, lambda: "im2col(): only 2D padding supported")
    # 检查 stride 是否为长度为 2 的元组
    torch._check(len(stride) == 2, lambda: "im2col(): only 2D stride supported")

    # 定义检查参数正数的函数
    def check_positive(param, param_name, strict=True):
        # strict 为 True 时要求所有参数大于 0，否则大于等于 0
        cond = all(p > 0 for p in param) if strict else all(p >= 0 for p in param)
        torch._check(
            cond, lambda: f"{param_name} should be greater than zero, but got {param}"
        )

    # 检查 kernel_size 中所有元素是否大于 0
    check_positive(kernel_size, "kernel_size")
    # 检查 dilation 中所有元素是否大于 0
    check_positive(dilation, "dilation")
    # 检查 padding 中所有元素是否大于等于 0
    check_positive(dilation, "padding", strict=False)
    # 检查 stride 中所有元素是否大于 0
    check_positive(stride, "stride")

    # 获取输入张量的形状和维度
    shape = input.shape
    ndim = len(shape)
    # 检查输入张量的维度是否为 3 或 4，并且最后三个维度的值都不为 0
    torch._check(
        ndim in (3, 4) and all(d != 0 for d in shape[-3:]),
        lambda: "Expected 3D or 4D (batch mode) tensor for input with possible 0 batch size "
        f"and non-zero dimensions, but got: {tuple(shape)}",
    )

    # 计算输出张量的尺寸
    output_size = tuple(
        1 + (out + 2 * pad - dil * (ker - 1) - 1) // st
        for out, pad, dil, ker, st in zip(
            shape[-2:], padding, dilation, kernel_size, stride
        )
    )
    # 检查输出张量的尺寸是否每个维度都大于 0
    torch._check(
        all(c > 0 for c in output_size),
        lambda: f"Given an input with spacial size {tuple(shape[-2:])}, "
        f"kernel_size={kernel_size}, dilation={dilation}, "
        f"padding={padding}, stride={stride}, "
        "the calculated shape of the array of sliding blocks "
        f"is {output_size}, but its components must be at least one.",
    )

    # 检查是否为批处理输入，如果不是，则扩展维度
    batched_input = ndim == 4
    if not batched_input:
        input = input.unsqueeze(0)

    # 获取输入张量的批次维度、通道维度、以及高度和宽度维度
    batch_dim, channel_dim, input_h, input_w = input.shape

    # 解包 stride、padding、dilation 和 kernel_size
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    kernel_h, kernel_w = kernel_size

    # 计算行和列方向上的块的索引
    blocks_row_indices = _im2col_col2im_indices_along_dim(
        input_h, kernel_h, dilation_h, padding_h, stride_h, input.device
    )
    blocks_col_indices = _im2col_col2im_indices_along_dim(
        input_w, kernel_w, dilation_w, padding_w, stride_w, input.device
    )

    # 使用 F.pad 进行填充，注意参数顺序 (padding_left, padding_right, padding_top, padding_bottom)
    padded_input = F.pad(input, (padding_w, padding_w, padding_h, padding_h))

    # 增加维度以匹配 blocks_row_indices 和 blocks_col_indices 的维度
    blocks_row_indices = blocks_row_indices.unsqueeze(-1).unsqueeze(-1)
    # 利用 blocks_row_indices 和 blocks_col_indices 从 padded_input 中提取输出
    output = padded_input[:, :, blocks_row_indices, blocks_col_indices]
    # 调整输出的维度顺序
    output = output.permute(0, 1, 2, 4, 3, 5)
    # 计算输出张量的形状
    num_blocks_row = blocks_row_indices.size(1)
    num_blocks_col = blocks_col_indices.size(1)
    output = output.reshape(
        batch_dim, channel_dim * kernel_h * kernel_w, num_blocks_row * num_blocks_col
    )

    # 如果输入不是批处理的，则去除批次维度
    if not batched_input:
        output = output.squeeze(0)
    # 返回最终的输出张量
    return output
# 将函数注册为 aten.col2im 的分解函数
# 应用 out_wrapper 装饰器
# 应用 pw_cast_for_opmath 装饰器
def col2im(
    input: Tensor,
    output_size: List[int],
    kernel_size: List[int],
    dilation: List[int],
    padding: List[int],
    stride: List[int],
) -> Tensor:
    # 检查 output_size 是否为二维
    torch._check(len(output_size) == 2, lambda: "only 2D output_size supported")
    # 检查 kernel_size 是否为二维
    torch._check(len(kernel_size) == 2, lambda: "only 2D kernel supported")
    # 检查 dilation 是否为二维
    torch._check(len(dilation) == 2, lambda: "only 2D dilation supported")
    # 检查 padding 是否为二维
    torch._check(len(padding) == 2, lambda: "only 2D padding supported")
    # 检查 stride 是否为二维
    torch._check(len(stride) == 2, lambda: "only 2D stride supported")

    # 检查参数列表中的每个参数是否为正数
    def check_positive(param, param_name, strict=True):
        cond = all(p > 0 for p in param) if strict else all(p >= 0 for p in param)
        torch._check(
            cond, lambda: f"{param_name} should be greater than zero, but got {param}"
        )

    # 检查 kernel_size 是否为正数
    check_positive(kernel_size, "kernel_size")
    # 检查 dilation 是否为正数
    check_positive(dilation, "dilation")
    # 检查 padding 是否为非负数
    check_positive(padding, "padding", strict=False)
    # 检查 stride 是否为正数
    check_positive(stride, "stride")
    # 检查 output_size 是否为正数
    check_positive(output_size, "output_size")

    # 获取输入张量的形状和维度数
    shape = input.shape
    ndim = len(shape)
    # 检查输入张量的维度是否为 2 或 3，并且最后两个维度不为 0
    torch._check(
        ndim in (2, 3) and all(d != 0 for d in shape[-2:]),
        lambda: "Expected 2D or 3D (batch mode) tensor for input with possible 0 batch size "
        f"and non-zero dimensions, but got: {tuple(shape)}",
    )

    # 计算 kernel_size 的乘积
    prod_kernel_size = kernel_size[0] * kernel_size[1]
    # 检查输入张量的第一个非批量维度是否能被 kernel_size 的乘积整除
    torch._check(
        shape[-2] % prod_kernel_size == 0,
        lambda: "Expected size of input's first non-batch dimension to be divisible by the "
        f"product of kernel_size, but got input.shape[-2] = {shape[-2]} and "
        f"kernel_size={kernel_size}",
    )

    # 计算 col2im 的输出尺寸
    col = [
        1 + (out + 2 * pad - dil * (ker - 1) - 1) // st
        for out, pad, dil, ker, st in zip(
            output_size, padding, dilation, kernel_size, stride
        )
    ]
    L = col[0] * col[1]
    # 检查输入张量的最后一个维度是否等于 L
    torch._check(
        shape[-1] == L,
        lambda: f"Given output_size={output_size}, kernel_size={kernel_size}, "
        f"dilation={dilation}, padding={padding}, stride={stride}, "
        f"expected input.size(-1) to be {L} but got {shape[-1]}.",
    )
    # 检查 L 是否大于 0
    torch._check(
        L > 0,
        lambda: f"Given output_size={output_size}, kernel_size={kernel_size}, "
        f"dilation={dilation}, padding={padding}, stride={stride}, "
        f"expected input.size(-1) to be {L} but got {shape[-1]}.",
    )

    # 检查是否为批量输入，若不是则增加一个维度
    batched_input = ndim == 3
    if not batched_input:
        input = input.unsqueeze(0)

    shape = input.shape

    # 获取输出的高度和宽度以及步长、填充、膨胀和卷积核尺寸
    out_h, out_w = output_size
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    kernel_h, kernel_w = kernel_size

    # 将输入张量重塑为合适的形状
    input = input.reshape([shape[0], shape[1] // prod_kernel_size] + kernel_size + col)
    # 对输入张量进行维度置换
    input = input.permute(0, 1, 2, 4, 3, 5)
    # 使用 _im2col_col2im_indices_along_dim 函数计算行索引，用于构造输出张量
    indices_row = _im2col_col2im_indices_along_dim(
        out_h, kernel_h, dilation_h, padding_h, stride_h, input.device
    )
    # 将行索引张量在第四个维度上添加一个维度
    indices_row = _unsqueeze_to_dim(indices_row, 4)
    # 使用 _im2col_col2im_indices_along_dim 函数计算列索引，用于构造输出张量
    indices_col = _im2col_col2im_indices_along_dim(
        out_w, kernel_w, dilation_w, padding_w, stride_w, input.device
    )

    # 计算输出张量的填充后尺寸，考虑填充大小
    output_padded_size = [o + 2 * p for o, p in zip(output_size, padding)]
    # 创建一个与输入张量相同数据类型的零张量作为输出张量
    output = input.new_zeros(
        [shape[0], shape[1] // prod(kernel_size)] + output_padded_size
    )
    # 使用索引 idx 将输入张量 input 的值复制到输出张量 output 中
    idx = (None, None, indices_row, indices_col)
    output = aten._unsafe_index_put(output, idx, input, accumulate=True)
    # 对输出张量进行反向填充以去除填充的部分
    output = F.pad(output, (-padding_w, -padding_w, -padding_h, -padding_h))

    # 如果输入张量未进行批处理，则压缩输出张量的第一个维度
    if not batched_input:
        output = output.squeeze(0)
    # 返回输出张量作为函数的结果
    return output
@register_decomposition(aten.native_dropout_backward)
@out_wrapper()
def native_dropout_backward(grad_output: Tensor, mask: Tensor, scale: float):
    # 根据 CUDA 核心实现，我们应该进行此测试；
    # 但似乎测试失败了！
    # torch._check(mask.dtype == torch.bool, lambda: f"Mask should be Bool Scalar Type {mask.dtype}")

    # 模拟 CUDA 核心的行为以获取输出步幅：输出遵循输入的内存格式
    # 这与 TensorIterator 的行为不同
    r = (grad_output * (mask.type_as(grad_output) * scale)).clone(
        memory_format=utils.suggest_memory_format(grad_output)
    )
    return r


@register_decomposition(aten.unfold_backward)
@out_wrapper()
def unfold_backward(
    grad: Tensor, input_size: List[int], dimension: int, size: int, step: int
) -> Tensor:
    if len(input_size) == 0:
        return torch.squeeze_copy(grad, 0)
    dim = utils.canonicalize_dim(len(input_size), dimension)
    idx = torch.arange(input_size[dim], device=grad.device, dtype=torch.int32)
    idx = idx.unfold(0, size, step).flatten()
    grad = grad.movedim(-1, dim + 1).flatten(dim, dim + 1)
    # 注意：目前这在 triton 中生成两个核心
    # 可能可以融合为一个 scatter_reduce 调用，
    # 在 step <= size 的情况下提供 scatter_reduce 生成 1 个核心
    grad_input = grad.new_zeros(input_size)
    index = (None,) * dim + (idx,)
    return aten._unsafe_index_put(grad_input, index, grad, accumulate=True).contiguous()


@register_decomposition(aten.logit_backward.default)
@pw_cast_for_opmath
def logit_backward(
    grad_output: Tensor, self: Tensor, eps: Optional[float] = None
) -> Tensor:
    if eps is not None:
        lo = eps
        hi = 1.0 - lo
        return torch.where(
            torch.logical_and(self >= lo, self <= hi),
            grad_output / (self * (1.0 - self)),
            0.0,
        )
    else:
        return torch.where(
            torch.logical_and(self >= 0.0, self <= 1.0),
            grad_output / (self * (1.0 - self)),
            self.new_full((), float("nan")),
        )


@register_decomposition(aten.dropout)
@aten.dropout.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.dropout.default.py_impl(DispatchKey.Autograd)
def dropout(input: Tensor, p: float, train: Optional[bool]):
    if train and p != 0:
        return aten.native_dropout(input, p, train)[0]
    else:
        return input.clone()


@register_decomposition(aten.native_dropout)
@out_wrapper("out0", "out1")
def native_dropout(input: Tensor, p: float, train: Optional[bool]):
    # 此函数尚未实现具体代码逻辑，仅有函数定义
    # 如果训练为真并且概率 p 不为 0
    if train and p != 0:
        # 如果 p 等于 1
        if p == 1:
            # 返回一个和输入 input 形状相同的全零张量和一个和输入 input 形状相同的全零布尔张量
            return (torch.zeros_like(input), torch.zeros_like(input, dtype=torch.bool))
        # 如果输入 input 的数据类型不是浮点型
        if not input.dtype.is_floating_point:
            # 抛出运行时错误，指示结果类型 Float 无法转换为期望的输出类型 Long
            raise RuntimeError(
                "result type Float can't be cast to the desired output type Long"
            )
        # 生成一个和输入 input 形状相同的随机张量，如果大于 p，则为 True，否则为 False
        bool_mask = torch.rand_like(input) > p
        # 计算输出结果 res，使用布尔掩码 bool_mask 对输入 input 进行掩码，乘以系数 float(1.0 / (1.0 - p))
        res = bool_mask * input * float(1.0 / (1.0 - p))
        # 返回结果 res 和布尔掩码 bool_mask
        return (res, bool_mask)
    else:
        # 如果不处于训练模式或概率 p 为 0，直接返回输入 input 和一个和输入 input 形状相同的全一布尔张量
        return (input, torch.ones_like(input, dtype=torch.bool))
# 注册 _softmax 函数的分解方法，使用装饰器 @register_decomposition
# 同时应用 @out_wrapper 装饰器，以确保返回一个连续的张量
def _softmax(x: Tensor, dim: int, half_to_float: bool):
    # 使输入张量 x 变为连续张量
    x = x.contiguous()
    # 如果需要将半精度转换为单精度，则断言输入张量 x 的数据类型为 torch.half
    if half_to_float:
        assert x.dtype == torch.half
    # 计算元素操作的数据类型和结果数据类型，根据输入张量 x 的类型决定
    computation_dtype, result_dtype = utils.elementwise_dtypes(
        x, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    # 将输入张量 x 转换为计算操作的数据类型
    x = x.to(computation_dtype)
    # 如果输入张量 x 的元素数为 0，则返回其指数
    if x.numel() == 0:
        unnormalized = torch.exp(x)
    else:
        # 计算输入张量 x 指定维度上的最大值，并保持维度
        x_max = torch.amax(x, dim, keepdim=True)
        # 计算经过偏移的输入张量 x 的指数
        unnormalized = torch.exp(x - x_max)
    # 对未归一化的张量进行归一化处理，并保持维度
    result = unnormalized / torch.sum(unnormalized, dim, keepdim=True)
    # 如果不需要将半精度转换为单精度，则将结果张量转换为结果数据类型
    if not half_to_float:
        result = result.to(result_dtype)
    # 返回归一化结果
    return result


# 注册 _log_softmax 函数的分解方法，使用装饰器 @register_decomposition
# 同时应用 @out_wrapper 装饰器，以确保返回一个连续的张量
def _log_softmax(x: Tensor, dim: int, half_to_float: bool):
    # 使输入张量 x 变为连续张量
    x = x.contiguous()
    # 如果需要将半精度转换为单精度，则断言输入张量 x 的数据类型为 torch.half
    if half_to_float:
        assert x.dtype == torch.half
    # 计算元素操作的数据类型和结果数据类型，根据输入张量 x 的类型决定
    computation_dtype, result_dtype = utils.elementwise_dtypes(
        x, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    # 将输入张量 x 转换为计算操作的数据类型
    x = x.to(computation_dtype)
    # 如果输入张量 x 的元素数为 0，则返回其本身
    if x.numel() == 0:
        shifted = x
    else:
        # 计算输入张量 x 指定维度上的最大值，并保持维度
        x_max = torch.amax(x, dim, keepdim=True)
        # 对输入张量 x 进行偏移处理
        shifted = x - x_max
    # 计算经过偏移的张量的对数和
    shifted_logsumexp = torch.log(torch.sum(torch.exp(shifted), dim, keepdim=True))
    # 计算经过偏移后的结果张量
    result = shifted - shifted_logsumexp
    # 如果不需要将半精度转换为单精度，则将结果张量转换为结果数据类型
    if not half_to_float:
        result = result.to(result_dtype)
    # 返回结果张量
    return result


# 注册 embedding 函数的分解方法，使用装饰器 @register_decomposition
# 同时应用 @out_wrapper 装饰器，以确保返回一个连续的张量
def embedding(
    weight: Tensor,
    indices: Tensor,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    # 断言权重张量 weight 的维度为 2
    assert weight.dim() == 2, "'weight' must be 2-D"
    # 注意：scale_grad_by_freq 在前向传播中未使用
    if indices.ndim <= 1:
        # 如果索引张量 indices 的维度小于等于 1，则使用索引选择功能获取部分权重张量
        # 在这些情况下，weight[indices] 调用 item()
        out = weight.index_select(0, indices)
        # 如果索引张量 indices 的维度为 0，则压缩输出张量的维度
        if indices.ndim == 0:
            out = out.squeeze(0)
        # 返回部分权重张量
        return out
    else:
        # 如果索引张量 indices 的维度大于 1，则直接返回相应的权重张量
        return weight[indices]


# 注册 embedding_dense_backward 函数的分解方法，使用装饰器 @register_decomposition
# 同时应用 @out_wrapper 装饰器，以确保返回一个连续的张量
def embedding_dense_backward(
    grad_output: Tensor,
    indices: Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
):
    # 计算元素操作的数据类型和结果数据类型，根据输入张量 grad_output 的类型决定
    computation_dtype, result_dtype = utils.elementwise_dtypes(
        grad_output, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    # 将梯度输出张量 grad_output 转换为计算操作的数据类型
    grad_output = grad_output.to(computation_dtype)
    # 将索引张量 indices 转换为 torch.long 类型，忽略类型检查
    indices = _maybe_convert_to_dtype(indices, torch.long)  # type: ignore[assignment]
    # 如果设置了按频率缩放梯度
    if scale_grad_by_freq:
        # 创建一个新的零张量，形状与indices相同
        counts = indices.new_zeros((num_weights,))
        # 创建一个与indices相同形状的全1张量
        ones = torch.ones_like(indices)
        # 在counts张量上进行不安全的索引放置操作，将ones放置到indices指定位置，累加计算
        counts = aten._unsafe_index_put(counts, [indices], ones, accumulate=True)
        # 根据indices获取对应的counts值作为梯度权重缩放因子
        grad_weights_scale = counts[indices]
        # 将grad_output按照grad_weights_scale在最后一个维度上扩展，并对grad_output进行除法操作
        grad_output = grad_output / grad_weights_scale.unsqueeze(-1)

    # 创建一个mask张量，用来标记indices中等于padding_idx的位置
    mask = _unsqueeze_to_dim(indices == padding_idx, grad_output.ndim)
    # 根据mask，对grad_output进行填充操作，将padding位置的梯度置为0
    grad = grad_output.masked_fill(mask, 0)
    # 创建一个新的零张量，形状为(num_weights, grad_output.shape[indices.ndim :])
    grad_weight = grad_output.new_zeros(
        (num_weights,) + grad_output.shape[indices.ndim :]
    )
    # 在grad_weight张量上进行不安全的索引放置操作，将grad放置到indices指定位置，累加计算，并转换为result_dtype类型
    return aten._unsafe_index_put(grad_weight, [indices], grad, accumulate=True).to(
        result_dtype
    )
# 定义一个函数 prod，计算列表 x 中所有元素的乘积，并返回结果
def prod(x: List[int]):
    r = 1  # 初始化乘积为 1
    for i in x:
        r *= i  # 逐个将列表中的元素乘到 r 中
    return r  # 返回最终的乘积结果


# 定义一个函数 _pad_chunk，对输入的张量列表进行填充操作，并返回填充后的张量列表
def _pad_chunk(
    tensors: List[Tensor],  # 输入的张量列表
    dim: int,  # 填充的维度
    num_chunks: int,  # 填充的块数
) -> List[Tensor]:  # 返回填充后的张量列表
    padded_tensors = []  # 初始化空列表，用于存放填充后的张量
    for tensor in tensors:
        tensor_size = tensor.size()  # 获取张量的大小
        # 计算沿指定维度 dim 的填充量，使得张量能被 num_chunks 均匀分割
        pad_along_dim = (tensor_size[dim] + num_chunks - 1) // num_chunks * num_chunks
        if pad_along_dim != tensor_size[dim]:  # 如果需要进行填充
            # 构造填充的参数列表，用于调用 aten.constant_pad_nd 函数
            pad = [0] * 2 * (tensor.ndim - dim - 1) + [
                0,
                pad_along_dim - tensor_size[dim],
            ]
            tensor = aten.constant_pad_nd(tensor, pad, 0)  # 进行张量的填充操作
        view_size = tensor_size[:dim] + torch.Size([num_chunks, -1])
        padded_tensors.append(tensor.view(view_size))  # 将填充后的张量视图添加到列表中
    return padded_tensors  # 返回填充后的张量列表


# 定义一个函数 have_same_ndims，检查输入张量列表中的所有张量是否具有相同的维度数，返回布尔值
def have_same_ndims(tensors: List[Tensor]):
    ndim = tensors[0].ndim  # 获取第一个张量的维度数
    for tensor in tensors:
        if tensor.ndim != ndim:  # 如果有任何一个张量的维度数不同于第一个张量
            return False  # 返回 False
    return True  # 所有张量的维度数均相同，返回 True


# 定义一个函数 leading_dimension_matches，检查输入张量列表中的所有张量是否在指定维度 dim 上具有相同的大小
def leading_dimension_matches(tensors: List[Tensor], dim: int):
    leading_dim_sizes = tensors[0].size()[:dim]  # 获取第一个张量在前 dim 维度上的大小
    for tensor in tensors:
        # 检查每个张量在前 dim 维度上的大小是否与第一个张量相同
        torch._check(
            tensor.size()[:dim] == leading_dim_sizes,
            lambda: "_chunk_cat expects same sizes of 0,...,dim-1 dimensions for all tensors",
        )


# 定义一个函数 _preprocess_chunk_cat_inputs，对输入的张量列表及参数进行预处理，并返回处理后的维度 dim
def _preprocess_chunk_cat_inputs(
    tensors: List[Tensor],  # 输入的张量列表
    dim: int,  # 要连接的维度
    num_chunks: int,  # 分块数目
):
    torch._check(num_chunks >= 1, lambda: "_chunk_cat expects positive num_chunks")  # 检查 num_chunks 是否大于等于 1
    torch._check(
        len(tensors) > 0, lambda: "_chunk_cat expects a non-empty input tensor list"
    )  # 检查张量列表是否为空
    expected_dtype = tensors[0].dtype  # 获取第一个张量的数据类型
    expected_device = tensors[0].device  # 获取第一个张量的设备
    for tensor in tensors:
        torch._check(tensor.numel() > 0, lambda: "_chunk_cat expects non-empty tensor")  # 检查张量是否非空
        torch._check(
            tensor.dtype == expected_dtype,
            lambda: "_chunk_cat expects all input tensors with the same dtype",
        )  # 检查张量的数据类型是否一致
        torch._check(
            tensor.device == expected_device,
            lambda: "_chunk_cat expects all inputs tensors on the same device",
        )  # 检查张量是否在相同的设备上
    if have_same_ndims(tensors):  # 如果所有张量具有相同的维度数
        dim = utils.canonicalize_dim(tensors[0].dim(), dim)  # 规范化维度 dim
    else:
        torch._check(
            dim >= 0,
            lambda: "_chunk_cat expects non-negative dim when input tensors have different ndims",
        )  # 检查 dim 是否为非负数
        for tensor in tensors:
            torch._check(
                dim < tensor.ndim,
                lambda: "_chunk_cat expects dim < ndim for all input tensors",
            )  # 检查 dim 是否小于所有张量的维度数
    leading_dimension_matches(tensors, dim)  # 检查所有张量在指定维度 dim 上的大小是否相同
    return dim  # 返回预处理后的维度 dim


# 注册 _chunk_cat 函数的装饰器，指定它可以处理的函数签名
@register_decomposition([aten._chunk_cat.default, aten._chunk_cat.out])
def _chunk_cat(
    tensors: List[Tensor],  # 输入的张量列表
    dim: int,  # 要连接的维度
    num_chunks: int,  # 分块数目
    out: Optional[Tensor] = None,  # 输出张量（可选）
) -> Tensor:  # 返回连接后的张量
    dim = _preprocess_chunk_cat_inputs(tensors, dim, num_chunks)  # 预处理输入的张量列表和参数
    padded_tensors = _pad_chunk(tensors, dim, num_chunks)  # 对输入的张量列表进行填充操作
    # 如果输出张量 out 为 None，则使用 torch.cat 函数将 padded_tensors 列表中的张量沿着 dim + 1 维度拼接起来
    if out is None:
        return torch.cat(padded_tensors, dim + 1)
    # 如果输出张量 out 不为 None，则使用 torch.cat 函数将 padded_tensors 列表中的张量沿着 dim + 1 维度拼接起来，并将结果存储到 out 中
    else:
        torch.cat(padded_tensors, dim + 1, out=out)
        # 返回存储结果的张量 out
        return out
# 注册一个自定义分解函数用于 torch.aten.split_with_sizes 函数
@register_decomposition(aten.split_with_sizes)
def split_with_sizes(
    self: Tensor, split_sizes: List[int], dim: int = 0
) -> List[Tensor]:
    # 注意：首先执行 check_is_size 测试，以防 sum 测试尝试替换操作
    for i in range(len(split_sizes)):
        torch._check_is_size(
            split_sizes[i],
            lambda: "split_with_sizes expects split_sizes have only non-negative entries",
        )
    # 确保分割大小的总和等于张量在给定维度上的大小
    torch._check_with(
        ValueError,
        sum(split_sizes) == self.shape[dim],
        lambda: f"Split sizes add up to {sum(split_sizes)} but got the tensor's size of {self.shape[dim]}",
    )
    # 获取分割数目
    num_splits = len(split_sizes)
    splits = []
    start_idx = 0

    # 避免在模块级别导入 sympy
    from torch.fx.experimental.symbolic_shapes import expect_true

    for i in range(num_splits):
        length = split_sizes[i]
        # 由于前面的 sum 测试，我们知道这是真实的，但此断言有助于我们的内部推理
        expect_true(start_idx + length <= self.shape[dim])
        # 执行分割操作并将结果添加到 splits 列表中
        splits.append(self.narrow(dim, start_idx, length))
        start_idx += length
    # 返回分割后的张量列表
    return splits


# out_wrapper 目前不允许可选输出
@register_decomposition(
    [aten.split_with_sizes_copy.default, aten.split_with_sizes_copy.out]
)
def split_with_sizes_copy(
    self: Tensor,
    split_sizes: List[int],
    dim: int = 0,
    out: Optional[List[Tensor]] = None,
) -> Optional[List[Tensor]]:
    # 调用 split_with_sizes 函数获取分割后的张量列表 splits
    splits = split_with_sizes(self, split_sizes, dim=dim)
    if out is None:
        # 如果输出列表 out 为空，则对 splits 中的每个张量进行克隆操作
        return [s.clone(memory_format=torch.contiguous_format) for s in splits]
    else:
        # 否则，对 out 和 splits 中的每个张量进行逐一复制和调整大小的操作
        for output, split in zip(out, splits):
            _maybe_resize_out(output, split.shape)
            _safe_copy_out(copy_from=split, copy_to=output, exact_dtype=True)
        # 返回空值表示操作已完成
        return None


# 注册一个自定义分解函数用于 torch.aten.unsafe_split.Tensor 函数
@register_decomposition(aten.unsafe_split.Tensor)
def unsafe_split(input: Tensor, split_size: int, dim: int = 0) -> Tuple[Tensor, ...]:
    # 直接调用 torch.aten.split.Tensor 函数来实现不安全的分割操作
    return aten.split.Tensor(input, split_size, dim)


# 注册一个自定义分解函数用于 torch.aten.unsafe_split_with_sizes.default 函数
@register_decomposition(aten.unsafe_split_with_sizes.default)
def unsafe_split_with_sizes(
    input: Tensor, split_sizes: List[int], dim: int = 0
) -> Tuple[Tensor, ...]:
    # 直接调用 torch.aten.split_with_sizes.default 函数来实现不安全的带尺寸分割操作
    return aten.split_with_sizes.default(input, split_sizes, dim)


# 注册一个自定义分解函数用于 torch.aten.split.Tensor 函数
@register_decomposition(aten.split.Tensor)
def split(self: Tensor, split_size: int, dim: int = 0) -> Tuple[Tensor, ...]:
    # 获取输入张量的尺寸信息
    input_sizes = self.shape
    dim_size = input_sizes[dim]
    if split_size == 0:
        # 特殊情况：当分割尺寸为零时，直接返回原始张量的元组形式
        assert dim_size == 0
        return (self,)
    # 计算分块数目
    chunks = (dim_size + split_size - 1) // split_size

    # 避免在模块级别导入 sympy
    from torch.fx.experimental.symbolic_shapes import guard_int

    # 对 chunks 进行类型保护，确保其为整数
    chunks = guard_int(chunks)
    # 创建一个分割尺寸列表，每个分块尺寸为 split_size，除了最后一个分块可能会有不同的大小
    split_sizes = [split_size for i in range(chunks)]
    split_sizes[-1] = split_size - (split_size * chunks - dim_size)
    # 调用 torch.split 函数来实现张量的分割操作
    return torch.split(self, split_sizes, dim)


# 注册一个自定义分解函数用于 torch.aten.tensor_split.tensor_indices_or_sections.py_impl 函数
@aten.tensor_split.tensor_indices_or_sections.py_impl(
    DispatchKey.CompositeImplicitAutograd
@register_decomposition(aten.addmv)
@out_wrapper()
@pw_cast_for_opmath
# 注册 ATen 函数 `addmv` 的分解实现，用于张量操作数学函数的输出包装和类型转换
def addmv(
    self: Tensor,
    mat: Tensor,
    vec: Tensor,
    beta: int = 1,
    alpha: int = 1,
    out: Optional[Tensor] = None,
):
    if not self.is_floating_point() and not self.is_complex():
        beta = int(beta)
        alpha = int(alpha)
    out = alpha * torch.mv(mat, vec)
    if beta == 0:
        return out

    # The output of aten.addmv is contiguous, we need to match this behavior in the decomposition.
    # The original implementation 'beta * self + out' would return a strided tensor if `self` is strided.
    # We thus use `out`, the output of torch.mv, which is always contiguous, as the first argument for addition.
    # This is relying on TensorIterator's behavior that it takes higher precedence on the stride of first input.
    # Alternative, we can write `(beta * self + out).contiguous()`, but it introduces another copy in some cases.
    # This implementation is not ideal, and we should revisit this when we have a better solution.
    return out + beta * self
def addmv(self: Tensor, mat1: Tensor, vec: Tensor, beta: int = 1, alpha: int = 1):
    # 检查 self 是否为浮点数或复数，如果不是，则将 beta 和 alpha 转换为整数
    if not self.is_floating_point() and not self.is_complex():
        beta = int(beta)
        alpha = int(alpha)
    
    # 计算 alpha * mat1 @ vec，其中 @ 表示矩阵乘以向量的操作
    out = alpha * torch.mv(mat1, vec)
    
    # 如果 beta 等于 0，则直接返回 out
    if beta == 0:
        return out
    
    # 否则返回 out + beta * self
    return out + beta * self


@register_decomposition(aten.native_group_norm_backward.default)
@pw_cast_for_opmath
def native_group_norm_backward(
    grad_output: Tensor,
    input: Tensor,
    mean: Tensor,
    rstd: Tensor,
    gamma: Optional[Tensor],
    N: int,
    C: int,
    HxW: int,
    group: int,
    output_mask: List[bool],
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    # 检查所有输入张量的设备是否相同，不允许使用 CPU 标量张量
    utils.check_same_device(
        grad_output, input, mean, rstd, allow_cpu_scalar_tensors=False
    )
    
    # 检查输入张量和梯度张量的形状是否相同，不允许使用 CPU 标量张量
    utils.check_same_shape(input, grad_output, allow_cpu_scalar_tensors=False)
    
    # 检查均值张量和标准差的形状是否相同，不允许使用 CPU 标量张量
    utils.check_same_shape(mean, rstd, allow_cpu_scalar_tensors=False)
    
    # 检查输入张量的元素数是否等于 N * C * HxW
    torch._check(
        input.numel() == N * C * HxW,
        lambda: f"Expect input to have {N * C * HxW} elements",
    )
    
    # 检查均值张量的形状是否为 (N, group)
    torch._check(
        mean.shape == (N, group),
        lambda: f"Expect mean to have shape ({N}, {group}, but got {mean.shape}",
    )
    
    # 检查 gamma 张量是否为 None 或元素数是否为 C
    torch._check(
        gamma is None or gamma.numel() == C,
        lambda: f"Expect gamma to have {C} elements but got {gamma.numel() if gamma is not None else -1}",
    )

    # 计算每个组的通道数
    cpg, _rem = divmod(C, group)
    
    # 检查通道数是否能被组数整除
    torch._check(
        _rem == 0,
        lambda: f"Expect number of channels {C} to be evenly-divisible by number of groups {group}",
    )

    # 计算内部梯度 ds 和 db
    ds = torch.mul(grad_output, input).view(N, C, HxW).sum(dim=[2])
    db = grad_output.view(N, C, HxW).sum(dim=[2])

    d_input: Optional[Tensor] = None
    d_gamma: Optional[Tensor] = None
    d_bias: Optional[Tensor] = None

    # 如果输出掩码为真
    if output_mask[0]:
        s = 1.0 / (HxW * cpg)
        
        # 根据是否有 gamma 计算 ds_val 和 db_val
        if gamma is not None:
            ds_val = torch.mul(ds, gamma.unsqueeze(0)).reshape(N, group, cpg).sum(2)
            db_val = torch.mul(db, gamma.unsqueeze(0)).reshape(N, group, cpg).sum(2)
            c1 = torch.mul(
                rstd.unsqueeze(-1),
                gamma.reshape(1, group, cpg),
            )
        else:
            ds_val = ds.reshape(N, group, cpg).sum(2)
            db_val = db.reshape(N, group, cpg).sum(2)
            c1 = torch.mul(
                rstd.unsqueeze(-1),
                torch.ones((1, group, cpg), device=rstd.device),
            )
        
        # 计算 c2 和 c3
        c2 = (db_val * mean - ds_val) * rstd * rstd * rstd * s
        c3 = -c2 * mean - db_val * rstd * s

        # 在适当的维度上扩展 c1, c2 和 c3
        c1 = c1.unsqueeze(-1)
        c2 = _unsqueeze_to_dim(c2, 4)
        c3 = _unsqueeze_to_dim(c3, 4)
        
        # 计算输入梯度 d_input
        d_input = (
            torch.mul(grad_output.reshape(N, group, cpg, HxW), c1)
            + torch.mul(input.reshape(N, group, cpg, HxW), c2)
            + c3
        )
        
        # 将 d_input 转换为与 input 相同的数据类型和形状
        d_input = d_input.reshape(input.shape).to(input.dtype)
    # 如果输出掩码中第二位为真，则计算关于 gamma 的梯度 d_gamma
    if output_mask[1]:
        # 计算 gamma 的梯度 d_gamma
        d_gamma = (
            (
                # 计算在 Batch Normalization 中使用的梯度部分，
                # ds 是标准化后的输入，db 是标准化参数中的偏移量
                (ds.view(N, group, cpg) - db.view(N, group, cpg) * mean.unsqueeze(-1))
                * rstd.unsqueeze(-1)  # 乘以标准差的倒数
            )
            .sum(dim=[0])  # 沿指定维度求和，得到 C 个通道的和
            .reshape(C)  # 重新整形成 C 个通道
        )
    
    # 如果输出掩码中第三位为真，则计算关于 bias 的梯度 d_bias
    if output_mask[2]:
        # 计算 bias 的梯度 d_bias，直接对 db 在指定维度求和
        d_bias = db.sum(dim=[0])

    # 返回计算得到的梯度：输入的梯度 d_input，gamma 的梯度 d_gamma，bias 的梯度 d_bias
    return (d_input, d_gamma, d_bias)
# 注册一个函数作为 `aten.native_group_norm_backward.out` 的分解函数
@register_decomposition(aten.native_group_norm_backward.out)
def native_group_norm_backward_out(
    grad_output: Tensor,                              # 梯度输出张量
    input: Tensor,                                    # 输入张量
    mean: Tensor,                                     # 均值张量
    rstd: Tensor,                                     # 标准差倒数张量
    gamma: Optional[Tensor],                          # 可选的 gamma 参数张量
    N: int,                                           # 批次大小
    C: int,                                           # 通道数
    HxW: int,                                         # 高度乘以宽度
    group: int,                                       # 组数
    output_mask: List[bool],                          # 输出掩码列表
    *,
    out0: torch.Tensor,                               # 输出张量 0
    out1: torch.Tensor,                               # 输出张量 1
    out2: torch.Tensor,                               # 输出张量 2
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    result = native_group_norm_backward(
        grad_output, input, mean, rstd, gamma, N, C, HxW, group, output_mask
    )
    grad_input = (out0, out1, out2)                   # 梯度输入的元组
    for i, r in enumerate(result):
        if r is not None:
            _maybe_resize_out(grad_input[i], r.shape)  # 可能调整输出的大小
            _safe_copy_out(copy_from=r, copy_to=grad_input[i], exact_dtype=True)  # 安全复制张量数据到 grad_input[i]

    return grad_input                                  # 返回梯度输入张量的元组


# 将张量 x 转换为指定的数据类型，如果 x 不为 None
def _maybe_cast(x: Optional[Tensor], dtype) -> Optional[Tensor]:
    if x is not None:
        return x.to(dtype)
    return x


# TODO: 仔细查看类型提升的语义
# 注册一个函数作为 `aten.native_layer_norm_backward.default` 的分解函数
def native_layer_norm_backward(
    grad_out: Tensor,                                 # 梯度输出张量
    input: Tensor,                                    # 输入张量
    normalized_shape: List[int],                      # 归一化形状的列表
    mean: Tensor,                                     # 均值张量
    rstd: Tensor,                                     # 标准差倒数张量
    weight: Optional[Tensor],                         # 可选的权重张量
    bias: Optional[Tensor],                           # 可选的偏置张量
    output_mask: List[bool],                          # 输出掩码列表
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    input_shape = input.shape                          # 输入张量的形状
    input_ndim = input.dim()                          # 输入张量的维度数
    computation_dtype = utils.get_computation_dtype(input.dtype)  # 计算数据类型
    grad_out_cast, input_cast, weight_cast, bias_cast = (
        x.to(computation_dtype).contiguous() if x is not None else x  # 将输入张量、权重张量、偏置张量转换为指定数据类型并保证内存连续性，如果它们不为 None
        for x in (grad_out, input, weight, bias)
    )
    assert grad_out_cast is not None                    # 断言梯度输出张量不为 None

    axis = input_ndim - len(normalized_shape)            # 轴数
    inner_dims = input_shape[axis:]                     # 内部维度
    outer_dims = input_shape[:axis]                     # 外部维度
    inner_dim_indices: List[int] = []
    outer_dim_indices: List[int] = []
    for i in range(input_ndim):
        if i >= axis:
            inner_dim_indices.append(i)
        else:
            outer_dim_indices.append(i)

    N = prod(inner_dims)  # type: ignore[arg-type]       # 内部维度乘积
    M = prod(outer_dims)  # type: ignore[arg-type]       # 外部维度乘积
    if M <= 0 or N <= 0:
        return (
            input.new_zeros(input_shape) if output_mask[0] else None,  # 根据输出掩码创建全零张量或 None
            input.new_zeros(input_shape[axis:]) if output_mask[1] else None,  # 根据输出掩码创建全零张量（部分）或 None
            input.new_zeros(input_shape[axis:]) if output_mask[2] else None,  # 根据输出掩码创建全零张量（部分）或 None
        )
    mean = _unsqueeze_to_dim(mean, input_cast.dim())  # type: ignore[union-attr]  # 将均值张量扩展到指定的维度
    rstd = _unsqueeze_to_dim(rstd, input_cast.dim())  # type: ignore[union-attr]  # 将标准差倒数张量扩展到指定的维度
    x_hat = (input_cast - mean) * rstd                 # 归一化输入
    if weight_cast is not None:
        grad_x_hat = grad_out_cast * weight_cast       # 加权梯度
    else:
        grad_x_hat = grad_out_cast                     # 没有权重的梯度
    a = grad_x_hat * N                                 # 计算 a
    b = torch.sum(grad_x_hat, inner_dim_indices, True)  # 计算 b
    c1 = torch.mul(grad_x_hat, x_hat)                  # 计算 c1
    c2 = torch.sum(c1, inner_dim_indices, True)        # 计算 c2
    # 计算 x_hat 和 c2 的逐元素乘积
    c3 = torch.mul(x_hat, c2)

    # 计算 inner = a - b - c3
    inner = a - b - c3
    
    # 初始化梯度变量，用于存储反向传播的梯度
    d_input: Optional[Tensor] = None
    d_weight: Optional[Tensor] = None
    d_bias: Optional[Tensor] = None
    
    # 如果 output_mask 的第一个元素为真，则计算 d_input
    if output_mask[0]:
        d_input = (rstd / N) * inner

    # 如果 output_mask 的第二个元素为真，并且 weight_cast 不为 None，则计算 d_weight
    if output_mask[1] and weight_cast is not None:
        # 如果 outer_dim_indices 的长度大于 0，则按照指定维度求和
        if len(outer_dim_indices) > 0:
            d_weight = torch.sum(grad_out_cast * x_hat, outer_dim_indices, False)
        else:
            d_weight = grad_out_cast * x_hat

    # 如果 output_mask 的第三个元素为真，并且 bias_cast 不为 None，则计算 d_bias
    if output_mask[2] and bias_cast is not None:
        # 如果 outer_dim_indices 的长度大于 0，则按照指定维度求和
        if len(outer_dim_indices) > 0:
            d_bias = torch.sum(grad_out_cast, outer_dim_indices, False)
        else:
            d_bias = grad_out_cast.clone()

    # 返回计算得到的梯度，通过 _maybe_cast 函数进行可能的类型转换
    return (
        _maybe_cast(d_input, input.dtype),
        _maybe_cast(d_weight, input.dtype),
        _maybe_cast(d_bias, input.dtype),
    )
# 注册特定函数到解构器中，用于 ATen 原生层归一化反向传播
@register_decomposition(aten.native_layer_norm_backward.out)
def native_layer_norm_backward_out(
    grad_out: Tensor,                        # 梯度输出张量
    input: Tensor,                           # 输入张量
    normalized_shape: List[int],             # 归一化形状
    mean: Tensor,                            # 均值张量
    rstd: Tensor,                            # 标准差的倒数张量
    weight: Optional[Tensor],                # 权重张量（可选）
    bias: Optional[Tensor],                  # 偏置张量（可选）
    output_mask: List[bool],                 # 输出掩码列表
    *,
    out0: torch.Tensor,                      # 输出张量0
    out1: torch.Tensor,                      # 输出张量1
    out2: torch.Tensor,                      # 输出张量2
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    result = native_layer_norm_backward(     # 调用原生层归一化反向传播函数
        grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask
    )
    grad_input = (out0, out1, out2)          # 设置梯度输入元组
    for i, r in enumerate(result):           # 遍历结果列表
        if r is not None:
            _maybe_resize_out(grad_input[i], r.shape)  # 如果结果不为空，则可能调整输出大小
            _safe_copy_out(copy_from=r, copy_to=grad_input[i], exact_dtype=True)  # 安全复制结果到梯度输入

    return grad_input                        # 返回梯度输入元组


def native_batch_norm_helper(
    input: Tensor,                           # 输入张量
    weight: Optional[Tensor],                # 权重张量（可选）
    bias: Optional[Tensor],                  # 偏置张量（可选）
    running_mean: Optional[Tensor],          # 运行时均值张量（可选）
    running_var: Optional[Tensor],           # 运行时方差张量（可选）
    training: bool,                          # 是否训练模式
    momentum: float,                         # 动量参数
    eps: float,                              # 用于数值稳定性的小常数
    functional: bool,                        # 是否功能模式
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    reduction_dims = [0] + list(range(2, input.dim()))  # 减少维度的列表
    computation_dtype = utils.get_computation_dtype(input.dtype)  # 获取计算数据类型
    new_running_mean = running_mean          # 新的运行时均值初始化为当前值
    new_running_var = running_var            # 新的运行时方差初始化为当前值
    if training:                             # 如果是训练模式
        computation_dtype = utils.get_computation_dtype(input.dtype)  # 再次获取计算数据类型
        input_acc = input.to(dtype=computation_dtype)  # 将输入张量转换为指定的计算数据类型
        biased_var, mean = torch.var_mean(
            input_acc, dim=reduction_dims, correction=0, keepdim=True
        )                                   # 计算偏差方差和均值
        rstd = torch.rsqrt(biased_var + eps)  # 计算倒数的平方根

        output = (input - mean) * rstd        # 计算输出张量

        save_mean = torch.squeeze(mean, reduction_dims)  # 保存压缩后的均值
        save_rstd = torch.squeeze(rstd, reduction_dims)  # 保存压缩后的倒数的平方根
        if running_mean is not None:          # 如果运行时均值不为空
            new_running_mean = momentum * save_mean + (1 - momentum) * running_mean  # 更新新的运行时均值
            if not functional:                # 如果不是功能模式
                running_mean.copy_(new_running_mean)  # 复制新的运行时均值到原始的运行时均值
        if running_var is not None:           # 如果运行时方差不为空
            n = input.numel() / input.shape[1]
            # 这里不严格匹配 eager 模式的数值计算，它累积方差和直接应用修正
            # 但...这需要在一个几乎不影响数值计算的张量上重新实现 var
            squeezed_var = torch.squeeze(biased_var, reduction_dims)
            unbiased_var = squeezed_var * (n / (n - 1))
            new_running_var = momentum * unbiased_var + (1 - momentum) * running_var  # 更新新的运行时方差
            if not functional:                # 如果不是功能模式
                running_var.copy_(new_running_var)  # 复制新的运行时方差到原始的运行时方差
    else:
        assert running_mean is not None and running_var is not None
        # 将运行均值和方差转换为指定计算类型的张量，并进行复制
        running_mean = running_mean.to(dtype=computation_dtype, copy=True)
        new_running_mean = running_mean
        running_var = running_var.to(dtype=computation_dtype, copy=True)
        new_running_var = running_var
        mean = running_mean
        # 计算标准差的倒数，用于归一化输入数据
        invstd = 1 / (torch.sqrt(running_var + eps))
        # 在 CPU 和 CUDA 之间存在非常令人恼火的不一致性，导致形状不同
        if input.device.type != "cpu":
            # 在 GPU 上保存均值和归一化标准差
            save_mean = running_mean
            save_rstd = invstd
        else:
            # 在 CPU 上创建形状为空的张量来保存均值和归一化标准差
            save_mean = input.new_zeros((0,))
            save_rstd = input.new_zeros((0,))
        # 将均值增加一个维度，以匹配输入张量的维度
        mean = _unsqueeze_to_dim(mean, input.dim() - 1)
        # 将标准差的倒数增加一个维度，以匹配输入张量的维度
        invstd = _unsqueeze_to_dim(invstd, input.dim() - 1)
        # 对输入数据进行归一化操作
        output = (input - mean) * invstd

    if weight is not None:
        # 将权重展平为一维张量
        weight = weight.flatten()
        # 将权重增加一个维度，以匹配输入张量的维度
        weight = _unsqueeze_to_dim(weight, input.dim() - 1)
        # 对输出进行加权处理
        output = output * weight

    if bias is not None:
        # 将偏置项展平为一维张量
        bias = bias.flatten()
        # 将偏置项增加一个维度，以匹配输入张量的维度
        bias = _unsqueeze_to_dim(bias, input.dim() - 1)
        # 将偏置项加到输出上
        output = output + bias

    if input.device.type == "cpu":
        # 将在 GPU 上计算得到的保存均值和归一化标准差转换为输入数据类型的张量
        save_mean = save_mean.to(dtype=input.dtype)
        save_rstd = save_rstd.to(dtype=input.dtype)
    # 返回输出张量、保存的均值、保存的归一化标准差、更新后的运行均值、更新后的运行方差
    return (
        output.to(dtype=input.dtype),
        save_mean,
        save_rstd,
        new_running_mean,
        new_running_var,
    )
# 注册 native_batch_norm 函数的装饰器，用于函数注册分解
@register_decomposition(aten.native_batch_norm)
# 将输出包装为元组的装饰器，输出参数为 "out", "save_mean", "save_invstd"
@out_wrapper("out", "save_mean", "save_invstd")
# 定义 native_batch_norm 函数，接受多个参数并返回一个包含输出、保存均值和保存标准差的元组
def native_batch_norm(
    input: Tensor,  # 输入张量
    weight: Optional[Tensor],  # 权重张量（可选）
    bias: Optional[Tensor],  # 偏置张量（可选）
    running_mean: Optional[Tensor],  # 移动平均均值张量（可选）
    running_var: Optional[Tensor],  # 移动平均方差张量（可选）
    training: bool,  # 训练标志，指示是否在训练模式下
    momentum: float,  # 动量参数
    eps: float,  # 防止除零的小数值
) -> Tuple[Tensor, Tensor, Tensor]:  # 返回类型为包含三个张量的元组

    # 调用辅助函数 native_batch_norm_helper，获取输出、保存的均值和标准差、以及其他参数
    output, save_mean, save_rstd, _, _ = native_batch_norm_helper(
        input, weight, bias, running_mean, running_var, training, momentum, eps, False
    )

    # 返回计算得到的输出、保存的均值和标准差的元组
    return output, save_mean, save_rstd


# TODO: 这个分解不是长久之计。我们更倾向于用新的正确 schema 的 _native_batch_norm_legit 及其变种来替换 native_batch_norm，
# 但是由于目前不能立即在 C++ 中这样做，因为这会导致某些移动端使用情况不兼容。
#
# 由于这个变化对 aot autograd/functionalization 影响最大，我们简单地在 Python 分发器的 Autograd 键上注册这个分解
# （目前只被 aot autograd/functionalization 使用，实际上其他人不怎么用）。
# 大约两周后，我们应该移除这个分解，并逐步淘汰当前的 native_batch_norm，改为 _native_batch_norm_legit，并使用正确的 schema
# （表明有输入的变异）。
@aten.native_batch_norm.default.py_impl(DispatchKey.Autograd)
@aten.native_batch_norm.default.py_impl(DispatchKey.CompositeImplicitAutograd)
# 定义 native_batch_norm_decomposition 函数，接受多个参数并返回一个包含输出、保存均值和保存标准差的元组
def native_batch_norm_decomposition(
    input: Tensor,  # 输入张量
    weight: Optional[Tensor],  # 权重张量（可选）
    bias: Optional[Tensor],  # 偏置张量（可选）
    running_mean: Optional[Tensor],  # 移动平均均值张量（可选）
    running_var: Optional[Tensor],  # 移动平均方差张量（可选）
    training: bool,  # 训练标志，指示是否在训练模式下
    momentum: float,  # 动量参数
    eps: float,  # 防止除零的小数值
) -> Tuple[Tensor, Tensor, Tensor]:  # 返回类型为包含三个张量的元组

    # 如果 running_mean 和 running_var 都为 None，则调用 _native_batch_norm_legit 函数
    if running_mean is None and running_var is None:
        return aten._native_batch_norm_legit(
            input, weight, bias, training, momentum, eps
        )
    # 如果 running_mean 为 None 而 running_var 不为 None，则抛出运行时错误
    if running_mean is None:
        raise RuntimeError(
            "running_mean is None, but running_var is provided. "
            "They should both be None or both be provided."
        )
    # 如果 running_var 为 None 而 running_mean 不为 None，则抛出运行时错误
    if running_var is None:
        raise RuntimeError(
            "running_var is None, but running_mean is provided. "
            "They should both be None or both be provided."
        )
    # 如果处于训练模式，则调用 _native_batch_norm_legit 函数
    if training:
        # HACK: batch norm consolidation should clean this up so this op doesn't take in a training arg.
        return aten._native_batch_norm_legit(
            input, weight, bias, running_mean, running_var, training, momentum, eps
        )
    else:
        # 否则，调用 _native_batch_norm_legit_no_training 函数
        return aten._native_batch_norm_legit_no_training(
            input, weight, bias, running_mean, running_var, momentum, eps
        )


# 注册 unsafe_chunk 函数的装饰器，用于函数注册分解
@aten.unsafe_chunk.default.py_impl(DispatchKey.CompositeImplicitAutograd)
# 定义 unsafe_chunk_py_impl 函数，接受张量、块数和维度（默认为 0），返回张量列表
def unsafe_chunk_py_impl(tensor, chunks, dim=0) -> List[Tensor]:
    # 获取指定维度的尺寸
    dim_size = tensor.size(dim)
    # 计算每个块的大小
    split_size = (dim_size + chunks - 1) // chunks

    # 返回切分后的张量列表
    return tensor.chunk(chunks, dim)
    # 如果分割大小和维度大小都为0，则进行特殊处理
    if split_size == 0 and dim_size == 0:
        # 创建一个包含与 chunks 数量相同的分割大小列表，并调整最后一个分割大小，确保总和等于维度大小
        split_sizes = [split_size for _ in chunks]
        split_sizes[chunks - 1] = split_size - (split_size * chunks - dim_size)
        # 使用 ATen 库中的 unsafe_split_with_sizes.default 方法来执行不安全的张量分割
        return torch.ops.aten.unsafe_split_with_sizes.default(tensor, split_sizes, dim)
    # 如果不满足上述条件，则使用 ATen 库中的 unsafe_split.Tensor 方法来执行不安全的张量分割
    return torch.ops.aten.unsafe_split.Tensor(tensor, split_size, dim)
# 注册函数 `_native_batch_norm_legit_no_training` 到 `aten._native_batch_norm_legit_no_training.default` 的分解函数
@register_decomposition(aten._native_batch_norm_legit_no_training.default)
def _native_batch_norm_legit_no_training(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    # 调用 `aten._native_batch_norm_legit.default` 函数进行批量归一化操作，返回结果
    return aten._native_batch_norm_legit.default(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        False,  # training 参数设为 False 表示不处于训练模式
        momentum,
        eps,
    )


# 注册函数 `_native_batch_norm_legit` 到 `aten._native_batch_norm_legit.default` 的分解函数
@register_decomposition(aten._native_batch_norm_legit.default)
def _native_batch_norm_legit(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    # 调用 `native_batch_norm_helper` 辅助函数进行批量归一化操作，返回输出及相关统计信息
    output, save_mean, save_rstd, _, _ = native_batch_norm_helper(
        input, weight, bias, running_mean, running_var, training, momentum, eps, False
    )
    return output, save_mean, save_rstd


# 注册函数 `_native_batch_norm_legit_no_stats` 到 `aten._native_batch_norm_legit.no_stats` 的分解函数
@register_decomposition(aten._native_batch_norm_legit.no_stats)
def _native_batch_norm_legit_no_stats(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    # 调用 `native_batch_norm_helper` 辅助函数进行批量归一化操作，不使用统计信息，返回输出及相关统计信息
    output, save_mean, save_rstd, _, _ = native_batch_norm_helper(
        input, weight, bias, None, None, training, momentum, eps, False
    )
    return output, save_mean, save_rstd


# 注册函数 `_native_batch_norm_legit_functional` 到 `aten._native_batch_norm_legit_functional.default` 的分解函数
@register_decomposition(aten._native_batch_norm_legit_functional.default)
def _native_batch_norm_legit_functional(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    # 调用 `native_batch_norm_helper` 辅助函数进行批量归一化操作，返回输出及相关统计信息以及更新后的均值和方差
    (
        output,
        save_mean,
        save_rstd,
        new_running_mean,
        new_running_var,
    ) = native_batch_norm_helper(
        input, weight, bias, running_mean, running_var, training, momentum, eps, True
    )
    # 断言确保新的运行均值和方差不为 None
    assert new_running_mean is not None, "new_running_mean should not be None"
    assert new_running_var is not None, "new_running_var should not be None"
    return output, save_mean, save_rstd, new_running_mean, new_running_var


def _get_batch_norm_reserve_tensor(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    eps: float,
    training: bool,
) -> Tensor:
    """
    为批量归一化准备一个保留张量，仅供 cudnn 在前向状态传递到后向传递时使用。
    这在 `_batch_norm_with_update` 和 `_batch_norm_no_update` 中是需要的，
    这些函数支持多种后端，包括 cudnn。我们在这里创建这个张量，以便在检测到将调用
    cudnn 内核时，可以获得正确的形状。
    """
    # 实现保留张量的功能，具体形状由跟踪图确定，避免实际生成该张量依靠 DCE 进行优化。
    pass
    # 调用 PyTorch 的内部函数选择批归一化的后端实现，并忽略类型检查
    backend = torch._C._select_batch_norm_backend(  # type: ignore[attr-defined]
        input, weight, bias, running_mean, running_var, True, eps
    )
    # 初始化变量 reserve_size 为 0
    reserve_size = 0
    # 如果选择的批归一化后端是 CUDNN，则获取 CUDNN 批归一化需要的保留空间大小
    if backend == torch._C._BatchNormBackend.Cudnn:  # type: ignore[attr-defined]
        reserve_size = torch._C._get_cudnn_batch_norm_reserve_space_size(input, training)  # type: ignore[attr-defined]
    # 返回一个空的 Tensor，用于分配指定大小的内存空间
    return torch.empty(
        reserve_size, dtype=torch.uint8, layout=input.layout, device=input.device
    )
@register_decomposition(aten._batch_norm_with_update.default)
# 注册函数 _batch_norm_with_update 为 aten._batch_norm_with_update.default 的分解函数
def _batch_norm_with_update(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # 调用 native_batch_norm_helper 函数进行批归一化操作
    output, save_mean, save_rstd, _, _ = native_batch_norm_helper(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        True,  # training
        momentum,
        eps,
        False,  # functional
    )
    # 调用 _get_batch_norm_reserve_tensor 函数获取批归一化的保留值
    reserve = _get_batch_norm_reserve_tensor(
        input, weight, bias, running_mean, running_var, eps, training=True
    )
    # 返回结果，包括输出、保存的均值、保存的标准差和保留值
    return output, save_mean, save_rstd, reserve


@register_decomposition(aten._batch_norm_with_update_functional.default)
# 注册函数 _batch_norm_with_update_functional 为 aten._batch_norm_with_update_functional.default 的分解函数
def _batch_norm_with_update_functional(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    # 调用 native_batch_norm_helper 函数进行函数式批归一化操作
    (
        output,
        save_mean,
        save_rstd,
        new_rm,
        new_rv,
    ) = native_batch_norm_helper(
        input, weight, bias, running_mean, running_var, True, momentum, eps, True
    )
    # 调用 _get_batch_norm_reserve_tensor 函数获取批归一化的保留值
    reserve = _get_batch_norm_reserve_tensor(
        input, weight, bias, running_mean, running_var, eps, training=True
    )
    # 断言新的均值和方差不为空
    assert new_rm is not None, "new_running_mean should not be None"
    assert new_rv is not None, "new_running_var should not be None"
    # 返回结果，包括输出、保存的均值、保存的标准差、保留值、新的均值和新的方差
    return (output, save_mean, save_rstd, reserve, new_rm, new_rv)


@register_decomposition(aten._batch_norm_no_update.default)
# 注册函数 _batch_norm_no_update 为 aten._batch_norm_no_update.default 的分解函数
def _batch_norm_no_update(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # 调用 native_batch_norm_helper 函数进行批归一化操作，但不更新统计信息
    output, save_mean, save_rstd, _, _ = native_batch_norm_helper(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        False,  # training
        momentum,
        eps,
        False,  # functional
    )
    # 调用 _get_batch_norm_reserve_tensor 函数获取批归一化的保留值
    reserve = _get_batch_norm_reserve_tensor(
        input, weight, bias, running_mean, running_var, eps, training=False
    )
    # 返回结果，包括输出、保存的均值、保存的标准差和保留值
    return output, save_mean, save_rstd, reserve


@register_decomposition(aten._fused_dropout)
# 注册函数 _fused_dropout_decomposition 为 aten._fused_dropout 的分解函数
@out_wrapper("out0", "out1")
@pw_cast_for_opmath
def _fused_dropout_decomposition(input, p, generator=None):
    # 断言 generator 为 None
    assert generator is None
    # 创建一个掩码，用于执行融合的 dropout 操作
    mask = (torch.rand_like(input) < p).to(dtype=torch.uint8)
    # 计算融合的 dropout 结果
    res = mask.type_as(input) * input * (1.0 / p)
    # 返回结果，包括融合的输出和掩码
    return (res, mask)


@register_decomposition(aten._to_copy)
# 注册函数 _to_copy 为 aten._to_copy 的分解函数
@out_wrapper()
def _to_copy(
    x: Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
    layout=None,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: Optional[torch.memory_format] = None,
):
    # 实现将张量复制到指定设备的功能
    # 如果布局不为 None 并且不是 torch.strided，抛出异常 "TODO"
    # 如果要求使用固定内存，则抛出异常 "TODO"
    if device is None and dtype is None and memory_format is None:
        # 如果没有指定设备、数据类型和内存格式，则返回输入张量的克隆副本
        return x.clone()
    
    dtype_converted = False
    
    if device is not None and device != x.device:
        # 如果指定了设备，并且输入张量不在该设备上
        # 避免在 CPU 上进行数据类型转换
        if dtype is not None and device.type == "cpu":
            # 将输入张量转换为指定的数据类型
            x = torch._prims.convert_element_type(x, dtype)
            dtype_converted = True
        # 将输入张量移到指定设备上
        x = torch._prims.device_put(x, device)
    
    if dtype is not None and not dtype_converted:
        # 如果指定了数据类型，并且尚未进行数据类型转换
        # 将输入张量转换为指定的数据类型
        x = torch._prims.convert_element_type(x, dtype)
        dtype_converted = True
    
    if memory_format is not None:
        # 如果指定了内存格式
        # 返回指定内存格式的输入张量的克隆副本
        return torch.clone(x, memory_format=memory_format)
    
    # 返回经过所有操作（如果有）后的输入张量
    return x
# 注册 nop_decomposition 函数为特定操作的解析函数，用于将操作转换为对应的别名操作
# 该装饰器为 nop_decomposition 函数添加了输出包装器，但不改变函数本身的返回结果
@register_decomposition([aten.detach, aten.lift, aten.lift_fresh])
@out_wrapper()
def nop_decomposition(x):
    # 返回输入张量的别名，即不做任何操作，直接返回输入
    return aten.alias(x)


# 为 cudnn_batch_norm 函数注册 Autograd 分发键，以便在 autograd 之上运行此解析函数
# native_batch_norm 需要在 autograd 之前进行解析成其他操作
@aten.cudnn_batch_norm.default.py_impl(DispatchKey.Autograd)
@register_decomposition(aten.cudnn_batch_norm)
# 为 cudnn_batch_norm 函数添加输出包装器，指定输出的命名
@out_wrapper("out0", "out1", "out2", "out3")
def cudnn_batch_norm(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    training: bool,
    exponential_average_factor: float,
    epsilon: float,
):
    # 调用 native_batch_norm 函数进行批归一化操作
    a, b, c = aten.native_batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
    )
    # 如果处于训练阶段，返回额外的零张量以表示运行时的平均值和方差
    if training:
        return (a, b, c, input.new_zeros((0,), dtype=torch.uint8))
    # 否则返回正常的输出值和零张量作为占位符
    return (
        a,
        weight.new_zeros((0,)),
        weight.new_zeros((0,)),
        input.new_zeros((0,), dtype=torch.uint8),
    )


# 定义 _broadcast_batch_norm_backward 函数，用于在特定轴上广播批归一化的反向传播梯度
def _broadcast_batch_norm_backward(x, broadcast_mask):
    # 遍历广播掩码，根据需要在轴上添加维度以匹配掩码
    for axis, mask in enumerate(broadcast_mask):
        if mask == 1 and not (axis < x.ndim and x.shape[axis] == mask):
            x = x.unsqueeze(axis)
    return x


# 为 batch_norm_backward 函数注册特定操作的解析函数，用于将批归一化反向传播转换为对应的原生批归一化反向传播
@register_decomposition(aten.batch_norm_backward.default)
def batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    save_mean: Optional[Tensor],
    save_invstd: Optional[Tensor],
    train: bool,
    eps: float,
    output_mask: List[bool],
    reserve: Tensor,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    # 调用 native_batch_norm_backward 函数执行原生批归一化的反向传播
    return native_batch_norm_backward(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        eps,
        output_mask,
    )


# 为 native_batch_norm_backward 函数注册特定操作的解析函数，用于将原生批归一化反向传播转换为对应的张量操作
@register_decomposition(aten.native_batch_norm_backward.default)
def native_batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    save_mean: Optional[Tensor],
    save_invstd: Optional[Tensor],
    train: bool,
    eps: float,
    output_mask: List[bool],
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    # 获取输入张量的数据类型
    input_dtype = input.dtype
    # 如果权重张量不为空，则获取其数据类型；否则使用输入张量的数据类型
    if weight is not None:
        weight_dtype = weight.dtype
    else:
        weight_dtype = input_dtype
    # 获取计算过程中使用的数据类型
    computation_dtype = utils.get_computation_dtype(input.dtype)
    (
        grad_out_cast,
        input_cast,
        weight_cast,
        running_mean_cast,
        running_var_cast,
        save_mean_cast,
        save_invstd_cast,
    ) = (
        x.to(computation_dtype) if x is not None else x
        for x in (
            grad_out,
            input,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
        )
    )
    # 将各个输入变量按需转换为计算数据类型，如果为 None 则保持为 None

    input_shape = input.shape
    # 获取输入张量的形状信息
    input_rank = input.dim()
    # 获取输入张量的维度数

    assert input_rank >= 2, "rank of the input must be at least 2"
    # 断言输入张量的维度数至少为2，否则抛出错误信息

    axis = 1
    # 指定进行批归一化的维度轴
    num_features = prod(list(input_shape)) / input_shape[axis]
    # 计算批次中的特征数量
    mean = save_mean_cast
    invstd = save_invstd_cast
    # 初始化均值和标准差的反数为保存的值

    if train:
        assert save_mean_cast is not None and save_invstd_cast is not None
        # 如果是训练模式，断言保存的均值和标准差的反数不为 None
    else:
        assert running_mean_cast is not None and running_var_cast is not None
        # 如果是推理模式，断言运行时的均值和方差不为 None
        mean = running_mean_cast
        invstd = torch.rsqrt(running_var_cast + eps)
        # 使用运行时的均值和标准差的反数来更新均值和标准差的反数

    broadcast_mask: List[int] = [1] * input_rank
    broadcast_mask[axis] = input_shape[axis]
    # 创建广播掩码，用于批次归一化的广播操作

    reduction_axes: List[int] = []
    for i in range(input_rank):
        if i != axis:
            reduction_axes.append(i)
    # 创建用于减少求和的轴列表，排除批次归一化的轴

    mean = _broadcast_batch_norm_backward(mean, broadcast_mask)  # type: ignore[arg-type]
    # 对均值进行批次归一化的反向传播
    norm = 1.0 / num_features
    # 计算归一化因子
    grad_output_sum = torch.sum(grad_out_cast, reduction_axes)  # type: ignore[arg-type]
    # 计算梯度输出的和

    dot_p = torch.sum(grad_out_cast * (input_cast - mean), reduction_axes)  # type: ignore[operator]
    # 计算梯度输出和输入与均值之间的点积

    grad_mean = _broadcast_batch_norm_backward(grad_output_sum * norm, broadcast_mask)
    # 计算均值的梯度
    proj_scale = _broadcast_batch_norm_backward(torch.mul(dot_p * norm, invstd * invstd), broadcast_mask)  # type: ignore[operator]
    # 计算投影尺度

    if weight_cast is None:
        grad_scale = _broadcast_batch_norm_backward(invstd, broadcast_mask) * 1.0  # type: ignore[arg-type]
        # 如果权重为 None，则梯度尺度为标准差的批次归一化的反向传播乘以1.0
    else:
        grad_scale = _broadcast_batch_norm_backward(
            invstd * weight_cast, broadcast_mask
        )
        # 否则，梯度尺度为标准差乘以权重的批次归一化的反向传播

    if train:
        proj = (input_cast - mean) * proj_scale  # type: ignore[operator]
        # 如果是训练模式，计算投影
        grad_input = ((grad_out_cast - proj) - grad_mean) * grad_scale
        # 计算输入的梯度
    else:
        grad_input = grad_out_cast * grad_scale
        # 否则，计算输入的梯度

    if output_mask[1]:
        grad_weight = dot_p * invstd
        # 如果输出掩码的第二个位置为真，则计算权重的梯度
    else:
        grad_weight = None  # "None" doesn't work with vjp, should use zeros for vjp
        # 否则，设置权重的梯度为 None

    if output_mask[2]:
        grad_bias = grad_output_sum
        # 如果输出掩码的第三个位置为真，则计算偏置的梯度
    else:
        grad_bias = None  # "None" doesn't work with vjp, should use zeros for vjp
        # 否则，设置偏置的梯度为 None

    return (
        grad_input.to(input_dtype),
        _maybe_cast(grad_weight, weight_dtype),
        _maybe_cast(grad_bias, weight_dtype),
    )
    # 返回输入的梯度，可能转换为权重数据类型的权重梯度和偏置梯度
# 注册函数，将给定的函数注册为特定操作的反向计算
@register_decomposition(aten.native_batch_norm_backward.out)
def native_batch_norm_backward_out(
    grad_out: Tensor,                      # 梯度输出张量
    input: Tensor,                         # 输入张量
    weight: Optional[Tensor],               # 权重张量（可选）
    running_mean: Optional[Tensor],         # 运行时均值（可选）
    running_var: Optional[Tensor],          # 运行时方差（可选）
    save_mean: Optional[Tensor],            # 保存的均值（可选）
    save_invstd: Optional[Tensor],          # 保存的标准差的倒数（可选）
    train: bool,                            # 训练模式标志
    eps: float,                             # epsilon 参数
    output_mask: List[bool],                # 输出掩码列表
    *,
    out0: torch.Tensor,                     # 输出张量 out0
    out1: torch.Tensor,                     # 输出张量 out1
    out2: torch.Tensor,                     # 输出张量 out2
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    # 调用原生批归一化反向计算函数
    result = native_batch_norm_backward(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        eps,
        output_mask,
    )
    # 构建梯度输入元组
    grad_input = (out0, out1, out2)
    # 遍历结果和梯度输入，根据需要调整大小并安全复制
    for i, r in enumerate(result):
        if r is not None:
            _maybe_resize_out(grad_input[i], r.shape)
            _safe_copy_out(copy_from=r, copy_to=grad_input[i], exact_dtype=True)

    # 返回梯度输入元组
    return grad_input


# 将 miopen 批归一化反向计算函数注册为特定操作的反向计算
@register_decomposition(aten.miopen_batch_norm_backward)
@out_wrapper("out0", "out1", "out2")
def miopen_batch_norm_backward(
    input: Tensor,                          # 输入张量
    grad_output: Tensor,                    # 梯度输出张量
    weight: Tensor,                         # 权重张量
    running_mean: Optional[Tensor],         # 运行时均值（可选）
    running_var: Optional[Tensor],          # 运行时方差（可选）
    save_mean: Optional[Tensor],            # 保存的均值（可选）
    save_var: Optional[Tensor],             # 保存的方差（可选）
    epsilon: float,                         # epsilon 参数
):
    # 调用原生批归一化反向计算函数
    return aten.native_batch_norm_backward(
        grad_output,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        True,
        epsilon,
        [True, True, True],
    )


# 将 cudnn 批归一化反向计算函数注册为特定操作的反向计算
@register_decomposition(aten.cudnn_batch_norm_backward)
@out_wrapper("out0", "out1", "out2")
def cudnn_batch_norm_backward(
    input: Tensor,                          # 输入张量
    grad_output: Tensor,                    # 梯度输出张量
    weight: Tensor,                         # 权重张量
    running_mean: Optional[Tensor],         # 运行时均值（可选）
    running_var: Optional[Tensor],          # 运行时方差（可选）
    save_mean: Optional[Tensor],            # 保存的均值（可选）
    save_var: Optional[Tensor],             # 保存的方差（可选）
    epsilon: float,                         # epsilon 参数
    reserveSpace: Tensor,                   # 保留空间张量
):
    # 调用原生批归一化反向计算函数
    return aten.native_batch_norm_backward(
        grad_output,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        True,
        epsilon,
        [True, True, True],
    )


# 将自适应平均池化函数注册为特定操作的反向计算
@register_decomposition(aten._adaptive_avg_pool2d)
@out_wrapper()
@pw_cast_for_opmath
def adaptive_avg_pool2d(input: Tensor, output_size: Tuple[int, int]):
    # 前置条件检查
    device = input.device                    # 获取输入张量的设备信息
    shape = input.shape                      # 获取输入张量的形状信息
    ndim = len(shape)                        # 计算输入张量的维度数
    torch._check(
        ndim in (3, 4),                      # 检查维度数是否为 3 或 4
        lambda: f"adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got {ndim}",
    )
    for d in input.shape[-2:]:
        torch._check(
            d != 0,                           # 检查非批处理维度是否具有非零大小
            lambda: "adaptive_avg_pool2d(): Expected input to have non-zero size for "
            f"non-batch dimensions, but input has shape {tuple(shape)}.",
        )

    # 优化点（我们也应该在核实现中执行此操作）
    # 检查输入形状的最后两维是否可以整除输出大小的对应维度
    if shape[-2] % output_size[-2] == 0 and shape[-1] % output_size[-1] == 0:
        # 计算池化操作的步长
        stride = tuple(i // o for i, o in zip(shape[-2:], output_size))
        # 计算池化核的大小
        kernel = tuple(
            i - (o - 1) * s for i, o, s in zip(shape[-2:], output_size, stride)
        )
        # 执行平均池化操作并返回结果
        return torch.nn.functional.avg_pool2d(input, kernel, stride)

    # 定义计算起始索引的函数
    def start_index(a, b, c):
        return torch.div(a * c, b, rounding_mode="trunc")

    # 定义计算结束索引的函数
    def end_index(a, b, c):
        return torch.div((a + 1) * c + b - 1, b, rounding_mode="trunc")

    # 定义计算索引范围和长度的函数
    def compute_idx(in_size, out_size):
        # 创建一个从0到out_size的整数张量orange
        orange = torch.arange(out_size, device=device, dtype=torch.int64)
        # 计算起始索引
        i0 = start_index(orange, out_size, in_size)
        # 计算最大长度maxlength
        maxlength = in_size // out_size + 1
        in_size_mod = in_size % out_size
        # 根据是否自适应调整maxlength
        adaptive = not (in_size_mod == 0 or out_size % in_size_mod == 0)
        if adaptive:
            maxlength += 1
        elif in_size_mod == 0:
            maxlength -= 1
        # 创建一个从0到maxlength的整数张量range_max
        range_max = torch.arange(maxlength, device=device, dtype=torch.int64)
        # 计算索引张量idx
        idx = i0.unsqueeze(-1) + range_max
        if adaptive:
            # 如果是自适应的，需要进行截断以避免访问越界内存
            maxval = torch.scalar_tensor(
                in_size - 1, dtype=idx.dtype, device=idx.device
            )
            idx = torch.minimum(idx, maxval)
            # 计算每个窗口的长度
            i1 = end_index(orange, out_size, in_size)
            length = i1 - i0
        else:
            length = maxlength
        return idx, length, range_max, adaptive

    # 计算水平方向的索引和长度
    idxh, length_h, range_max_h, adaptive_h = compute_idx(shape[-2], output_size[-2])
    # 计算垂直方向的索引和长度
    idxw, length_w, range_max_w, adaptive_w = compute_idx(shape[-1], output_size[-1])

    # 根据计算得到的索引idxh和idxw从输入中选择对应的值
    vals = input[..., _unsqueeze_to_dim(idxh, 4), idxw]

    # 如果水平和垂直方向都不是自适应的，返回平均值
    if not adaptive_h and not adaptive_w:
        return torch.mean(vals, dim=(-3, -1))

    # 定义可能对值进行掩码操作的函数
    def maybe_mask(vals, length, range_max, adaptive, dim):
        if isinstance(length, IntLike):
            return vals, length
        else:
            # 零化我们不想选择的部分
            assert dim < 0
            mask = range_max >= length.unsqueeze(-1)
            if dim == -2:
                mask = _unsqueeze_to_dim(mask, 4)
            vals = torch.masked_fill(vals, mask, 0.0)
            # 计算每个窗口的长度
            length = _unsqueeze_to_dim(length, -dim)
            return vals, length

    # 根据是否自适应对水平方向的vals和length_h进行掩码操作
    vals, length_h = maybe_mask(
        vals, length_h, range_max_h, adaptive=adaptive_h, dim=-2
    )
    # 调用 maybe_mask 函数处理输入的 vals 和 length_w，并返回处理后的结果和长度值 length_w
    vals, length_w = maybe_mask(
        vals, length_w, range_max_w, adaptive=adaptive_w, dim=-1
    )

    # 针对小内核情况，展开求和计算
    ret = None
    # 遍历 vals 张量的倒数第三和倒数第一维度上的索引 i 和 j
    for i, j in product(range(vals.shape[-3]), range(vals.shape[-1])):
        # 初始化 ret 变量为 vals 的部分切片
        if ret is None:
            ret = vals[..., i, :, j]
        else:
            # 累加 vals 的部分切片到 ret
            ret = ret + vals[..., i, :, j]
    # 返回 ret 除以长度值 length_h 和 length_w 的乘积，计算平均值
    return ret / (length_h * length_w)
# 将函数`index_add_`注册为`aten.index_add_`的分解函数
@register_decomposition(aten.index_add_)
def index_add_(
    x: TensorLike,
    dim: int,
    index: TensorLike,
    tensor: TensorLike,
    *,
    alpha: NumberType = 1,
):
    # 调用内部函数 `_index_add`，执行索引加法操作，将结果就地修改到张量 `x` 中
    return _index_add(x, dim, index, tensor, inplace=True, alpha=alpha)


# 将函数`index_add`注册为`aten.index_add`的分解函数，并对输出进行包装
@register_decomposition(aten.index_add)
@out_wrapper()
def index_add(
    x: TensorLike,
    dim: int,
    index: TensorLike,
    tensor: TensorLike,
    *,
    alpha: NumberType = 1,
):
    # 调用内部函数 `_index_add`，执行索引加法操作，返回新的张量而不修改输入的张量 `x`
    return _index_add(x, dim, index, tensor, inplace=False, alpha=alpha)


# 实际执行索引加法操作的内部函数
def _index_add(
    x: TensorLike,
    dim: int,
    index: TensorLike,
    tensor: TensorLike,
    *,
    inplace: bool,
    alpha: NumberType = 1,
):
    # 规范化维度 `dim`，确保在张量 `x` 的维度范围内
    dim = utils.canonicalize_dims(x.ndim, dim)
    # 检查索引 `index` 的维度是否为 1 或 0
    torch._check(
        index.ndim <= 1,
        lambda: f"Index should have dimension 1 or 0 (got {index.ndim})",
    )
    # 获取索引 `index` 的大小
    index_size = index.size(0) if index.ndim == 1 else 1
    # 获取张量 `tensor` 在指定维度 `dim` 上的大小
    tensor_size = tensor.size(dim) if tensor.ndim > 0 else 1
    # 检查索引数量与张量大小是否相等
    torch._check(
        tensor_size == index_size,
        lambda: f"Number of indices ({index_size}) should be equal to tensor.size(dim) ({tensor_size}), for {dim=}",
    )
    # 如果 `alpha` 不等于 1，则将张量 `tensor` 缩放
    if alpha != 1:
        # 将张量 `tensor` 缩放为 `alpha * tensor`
        python_type = utils.dtype_to_type(x.dtype)
        torch._check(
            python_type == bool
            or utils.is_weakly_lesser_type(type(alpha), python_type),
            lambda: f"alpha argument of type {type(alpha)} cannot be safely cast to type {python_type}!",
        )
        tensor = tensor * alpha
    # 处理零维张量，将其视为 \R^1 的元素
    zero_dim = x.ndim == 0
    x1 = x.unsqueeze(0) if zero_dim else x
    # 构建索引元组 `idx`，用于索引放置操作
    idx = (None,) * dim + (index,)
    # 根据是否就地操作选择 `index_put_` 或 `index_put` 函数
    index_put = aten.index_put_ if inplace else aten.index_put
    # 执行索引放置操作，并累加到 `x1` 上
    out = index_put(x1, idx, tensor, accumulate=True)
    # 如果是就地操作，则返回修改后的张量 `x`
    if inplace:
        return x
    else:
        # 如果不是就地操作，则返回结果 `out`，并在零维情况下去除额外的维度
        return out.squeeze(0) if zero_dim else out.contiguous()


# 将函数 `pad_sequence` 注册为 `aten.pad_sequence.default` 的分解函数
@register_decomposition(aten.pad_sequence.default)
@aten.pad_sequence.default.py_impl(DispatchKey.CompositeImplicitAutograd)
def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    # 检查输入序列 `sequences` 的长度是否大于 0
    torch._check(len(sequences) > 0, lambda: "received an empty list of sequences")
    # 获取序列的数量 `sequences_size`
    sequences_size = len(sequences)
    # 获取第一个序列的大小 `max_size`
    max_size = sequences[0].size()
    # 获取除了第一个维度外的所有维度 `trailing_dims`
    trailing_dims = max_size[1:]
    # 获取序列中最大长度 `max_len`
    max_len = max(x.size(0) for x in sequences)
    # 根据 `batch_first` 决定输出张量的维度 `out_dims`
    if batch_first:
        out_dims = (sequences_size, max_len)
    else:
        out_dims = (max_len, sequences_size)
    # 将 `trailing_dims` 添加到输出维度中
    out_dims = out_dims + trailing_dims
    # 创建填充值为 `padding_value` 的新张量 `out`
    out = sequences[0].new_full(out_dims, padding_value)
    # 构造用于零维填充的维度 `dim_paddings`
    dim_paddings = (0, 0) * len(trailing_dims)
    # 遍历序列 `sequences`
    for i in range(sequences_size):
        currseq = sequences[i]
        # 在零维填充的基础上进行常数填充 `row`
        row = aten.constant_pad_nd(
            currseq, dim_paddings + (0, max_len - currseq.size(0)), padding_value
        )
        # 根据 `batch_first` 决定在哪个维度上进行散射选择操作，并更新 `out`
        if batch_first:
            out = aten.select_scatter(out, row, dim=0, index=i)
        else:
            out = aten.select_scatter(out, row, dim=1, index=i)
    # 返回填充后的输出张量 `out`
    return out


# 将函数 `index_copy_` 注册为 `aten.index_copy_` 的分解函数
@register_decomposition(aten.index_copy_)
# 将指定索引处的张量替换为另一个张量，支持原地操作
def index_copy_(x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike):
    return _index_copy(x, dim, index, tensor, inplace=True)


# 注册 `aten.index_copy` 的分解函数，并输出结果
@out_wrapper()
@register_decomposition(aten.index_copy)
def index_copy(x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike):
    return _index_copy(x, dim, index, tensor, inplace=False)


# 执行索引复制的实际操作，根据 inplace 参数选择不同的处理方式
def _index_copy(
    x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike, *, inplace: bool
):
    # 规范化维度参数 dim
    dim = utils.canonicalize_dims(x.ndim, dim)
    # 检查索引的维度是否为 0 或 1
    torch._check(
        index.ndim <= 1,
        lambda: f"Index should have dimension 1 or 0 (got {index.ndim})",
    )
    # 如果 x 是标量，则将其视为 \R^1 中的元素
    zero_dim = x.ndim == 0
    x1 = x.unsqueeze(0) if zero_dim else x
    index = index.unsqueeze(0) if index.ndim == 0 else index
    # 构建索引元组
    idx = (None,) * dim + (index,)
    # 根据 inplace 参数选择不同的索引放置函数
    index_put = aten.index_put_ if inplace else aten.index_put
    # 执行索引放置操作
    out = index_put(x1, idx, tensor)
    # 如果是原地操作，则返回原始张量 x；否则返回处理后的结果 out
    if inplace:
        return x
    else:
        return out.squeeze(0) if zero_dim else out.contiguous()


# 注册 `aten.log_sigmoid_forward` 的分解函数，并对输出进行包装
@pw_cast_for_opmath
@out_wrapper("output", "buffer")
@register_decomposition(aten.log_sigmoid_forward)
def log_sigmoid_forward(self: Tensor) -> Tuple[Tensor, Tensor]:
    # 计算 log sigmoid 的前向传播
    min = torch.minimum(self.new_zeros(()), self)
    z = torch.exp(-torch.abs(self))
    # 根据是否在 CUDA 上执行，创建相应的 buffer
    if self.is_cuda:
        buffer = self.new_zeros((0,))
    else:
        buffer = z
    # 返回计算结果以及可能的缓冲区 buffer
    return min - torch.log1p(z), buffer


# 注册 `aten.uniform` 的分解函数，并输出结果
@out_wrapper()
@register_decomposition(aten.uniform)
def uniform(
    x: Tensor,
    low: Union[bool, int, float] = 0.0,
    high: Union[bool, int, float] = 1.0,
    generator: Optional[torch.Generator] = None,
):
    # 调用 `_uniform_helper` 辅助函数来生成均匀分布的张量
    return prims._uniform_helper(
        x.shape,
        low=sym_float(low),
        high=sym_float(high),
        dtype=x.dtype,
        device=x.device,
        generator=generator,
    )


# 注册 `aten.uniform_` 的函数，执行原地操作
@register_decomposition(aten.uniform_)
def uniform_(self, low=0, high=1, generator=None):
    # 调用 `uniform` 函数生成均匀分布的张量，并执行原地拷贝操作
    return self.copy_(uniform(self, low, high, generator))


# 在 `aten/src/ATen/native/UpSample.cpp` 的 `compute_output_size` 函数中计算输出尺寸
def upsample_compute_output_size(input_size, output_size, scale_factors):
    # 获取空间维度数
    spatial_dimensions = len(input_size) - 2
    if output_size is not None:
        # 检查输出尺寸和缩放因子不能同时指定
        torch._check(
            scale_factors is None,
            lambda: "Must specify exactly one of output_size and scale_factors",
        )
        # 检查输出尺寸的维度是否与空间维度数相符，然后返回输出尺寸
        torch._check(len(output_size) == spatial_dimensions, lambda: "")
        return output_size
    # 如果给定了缩放因子，则执行以下操作
    if scale_factors is not None:
        # 确保output_size参数未指定，并输出警告信息
        torch._check(
            output_size is None,
            lambda: "Must specify exactly one of output_size and scale_factors",
        )
        # 确保scale_factors的数量与空间维度数相同
        torch._check(len(scale_factors) == spatial_dimensions, lambda: "")
        
        # 初始化output_size列表
        output_size = []
        # 遍历每个维度的缩放因子
        for i, s in enumerate(scale_factors):
            # 如果缩放因子是整数，则直接计算输出尺寸
            if int(s) == s:
                output_size.append(input_size[i + 2] * int(s))
            else:
                # 如果缩放因子不是整数，则调用sym_int函数计算输出尺寸
                output_size.append(sym_int(input_size[i + 2] * s))
        
        # 返回计算得到的输出尺寸
        return output_size
    
    # 如果未给定缩放因子，则执行以下操作
    # 输出一个错误，指示必须精确指定output_size或scale_factors中的一个
    torch._check(
        False, lambda: "Must specify exactly one of output_size and scale_factors"
    )
# 定义函数_get_scale_value，用于从列表scales中获取指定索引idx处的值
def get_scale_value(scales, idx):
    # 如果scales为None，则返回None
    if scales is None:
        return None
    # 否则返回scales列表中索引为idx的值
    return scales[idx]


# 注册函数_decomposition为上采样函数的装饰器，处理最近邻插值的1D、2D、3D版本
@register_decomposition(aten.upsample_nearest1d.vec)
@register_decomposition(aten.upsample_nearest2d.vec)
@register_decomposition(aten.upsample_nearest3d.vec)
# 注册最近邻插值的1D、2D、3D版本为自动微分调度的Python实现
@aten.upsample_nearest1d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest1d.vec.py_impl(DispatchKey.Autograd)
@aten.upsample_nearest2d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest2d.vec.py_impl(DispatchKey.Autograd)
@aten.upsample_nearest3d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest3d.vec.py_impl(DispatchKey.Autograd)
# 定义最近邻插值的向量化函数_upsample_nearest_vec
def _upsample_nearest_vec(
    input: Tensor,
    output_size: Optional[List[int]],
    scale_factors: Optional[List[float]],
) -> Tensor:
    # 计算输出尺寸osize，根据输入尺寸、输出尺寸和尺度因子
    osize = upsample_compute_output_size(input.size(), output_size, scale_factors)
    # 如果尺度因子scale_factors存在，则直接赋值给scales，否则创建一个与osize长度相同的None列表
    scales = (
        scale_factors if scale_factors else [None] * len(osize)  # type: ignore[list-item]
    )
    # 调用_upsample_nearest函数进行最近邻上采样操作，并返回结果
    return _upsample_nearest(input, osize, scales)


# 注册函数_decomposition为精确最近邻插值函数的装饰器，处理精确的1D、2D、3D版本
@register_decomposition(aten._upsample_nearest_exact1d.vec)
@register_decomposition(aten._upsample_nearest_exact2d.vec)
@register_decomposition(aten._upsample_nearest_exact3d.vec)
# 注册精确最近邻插值的1D、2D、3D版本为自动微分调度的Python实现
@aten._upsample_nearest_exact1d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten._upsample_nearest_exact1d.vec.py_impl(DispatchKey.Autograd)
@aten._upsample_nearest_exact2d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten._upsample_nearest_exact2d.vec.py_impl(DispatchKey.Autograd)
@aten._upsample_nearest_exact3d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten._upsample_nearest_exact3d.vec.py_impl(DispatchKey.Autograd)
# 定义精确最近邻插值的向量化函数_upsample_nearest_exact_vec
def _upsample_nearest_exact_vec(
    input: Tensor,
    output_size: Optional[List[int]],
    scale_factors: Optional[List[float]],
) -> Tensor:
    # 计算输出尺寸osize，根据输入尺寸、输出尺寸和尺度因子
    osize = upsample_compute_output_size(input.size(), output_size, scale_factors)
    # 如果尺度因子scale_factors存在，则直接赋值给scales，否则创建一个与osize长度相同的None列表
    scales = (
        scale_factors if scale_factors else [None] * len(osize)  # type: ignore[list-item]
    )
    # 调用_upsample_nearest函数进行精确最近邻上采样操作，并返回结果
    return _upsample_nearest(input, osize, scales, exact=True)


# 定义_compute_upsample_nearest_indices函数，用于计算最近邻上采样的索引
def _compute_upsample_nearest_indices(input, output_size, scales, exact=False):
    # 对于每个维度在output_size中，计算用于生成上采样输出的输入索引集合
    indices = []
    # 空间维度的数量等于output_size的长度
    num_spatial_dims = len(output_size)
    # 如果exact为True，则offset为0.5，否则为0.0
    offset = 0.5 if exact else 0.0
    # 遍历空间维度的数量
    for d in range(num_spatial_dims):
        # 数学匹配于aten/src/ATen/native/cpu/UpSampleKernel.cpp
        #
        # 计算索引如下：
        # scale = isize / osize
        # Case: exact=False
        # input_index = floor(output_index * scale)
        # 与OpenCV中的INTER_NEAREST相同
        #
        # Case: exact=True
        # index_f32 = (output_index + 0.5) * scale - 0.5
        # input_index = round(index_f32)
        # 与Pillow以及Scikit-Image/Scipy中的ndi.zoom相同
        #
        # 获取输出尺寸
        osize = output_size[d]
        # 获取输入张量在当前空间维度上的大小
        isize = input.shape[-num_spatial_dims + d]
        # 计算缩放比例
        scale = isize / (isize * scales[d]) if scales[d] is not None else isize / osize

        # 创建一个包含从0到osize的浮点数张量，设备为输入张量的设备
        output_indices = torch.arange(osize, dtype=torch.float32, device=input.device)
        # 计算输入张量的索引
        input_indices = ((output_indices + offset) * scale).to(torch.int64)
        # 对于剩余的空间维度，扩展输入索引张量
        for _ in range(num_spatial_dims - 1 - d):
            input_indices = input_indices.unsqueeze(-1)
        # 将计算得到的输入索引添加到索引列表中
        indices.append(input_indices)
    
    # 返回索引列表
    return indices
# 注册上采样最近邻方法的默认实现和输出方法到分解列表中
@register_decomposition([aten.upsample_nearest1d.default, aten.upsample_nearest1d.out])
# 将默认实现与自动微分相关的分发键注释为上采样最近邻方法的实现
@aten.upsample_nearest1d.default.py_impl(DispatchKey.CompositeImplicitAutograd)
# 将默认实现与自动微分相关的分发键注释为上采样最近邻方法的实现
@aten.upsample_nearest1d.default.py_impl(DispatchKey.Autograd)
# 以指定参数保留内存格式和确切的数据类型进行输出包装
@out_wrapper(preserve_memory_format=True, exact_dtype=True)
# 定义上采样最近邻方法，接受输入张量、输出尺寸和可选的比例参数，返回张量
def upsample_nearest1d(
    input: Tensor,
    output_size: List[int],
    scales: Optional[float] = None,
) -> Tensor:
    # 调用内部函数 _upsample_nearest 实现上采样最近邻操作，将比例参数作为列表传递
    return _upsample_nearest(input, output_size, [scales])


# 注册上采样最近邻精确方法的默认实现和输出方法到分解列表中
@register_decomposition(
    [aten._upsample_nearest_exact1d.default, aten._upsample_nearest_exact1d.out]
)
# 将默认实现与自动微分相关的分发键注释为上采样最近邻精确方法的实现
@aten._upsample_nearest_exact1d.default.py_impl(DispatchKey.CompositeImplicitAutograd)
# 将默认实现与自动微分相关的分发键注释为上采样最近邻精确方法的实现
@aten._upsample_nearest_exact1d.default.py_impl(DispatchKey.Autograd)
# 以指定参数保留内存格式和确切的数据类型进行输出包装
@out_wrapper(preserve_memory_format=True, exact_dtype=True)
# 定义上采样最近邻精确方法，接受输入张量、输出尺寸和可选的比例参数，返回张量
def upsample_nearest_exact1d(
    input: Tensor,
    output_size: List[int],
    scales: Optional[float] = None,
) -> Tensor:
    # 调用内部函数 _upsample_nearest 实现上采样最近邻操作，将比例参数和 exact=True 作为参数传递
    return _upsample_nearest(input, output_size, [scales], exact=True)


# 注册上采样最近邻方法的默认实现和输出方法到分解列表中
@register_decomposition([aten.upsample_nearest2d.default, aten.upsample_nearest2d.out])
# 将默认实现与自动微分相关的分发键注释为上采样最近邻方法的实现
@aten.upsample_nearest2d.default.py_impl(DispatchKey.CompositeImplicitAutograd)
# 将默认实现与自动微分相关的分发键注释为上采样最近邻方法的实现
@aten.upsample_nearest2d.default.py_impl(DispatchKey.Autograd)
# 以指定参数保留内存格式和确切的数据类型进行输出包装
@out_wrapper(preserve_memory_format=True, exact_dtype=True)
# 定义上采样最近邻方法，接受输入张量、输出尺寸和可选的比例参数（分别对应高度和宽度），返回张量
def upsample_nearest2d(
    input: Tensor,
    output_size: List[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    # 调用内部函数 _upsample_nearest 实现上采样最近邻操作，将高度和宽度的比例参数作为列表传递
    return _upsample_nearest(input, output_size, [scales_h, scales_w])


# 注册上采样最近邻精确方法的默认实现和输出方法到分解列表中
@register_decomposition(
    [aten._upsample_nearest_exact2d.default, aten._upsample_nearest_exact2d.out]
)
# 将默认实现与自动微分相关的分发键注释为上采样最近邻精确方法的实现
@aten._upsample_nearest_exact2d.default.py_impl(DispatchKey.CompositeImplicitAutograd)
# 将默认实现与自动微分相关的分发键注释为上采样最近邻精确方法的实现
@aten._upsample_nearest_exact2d.default.py_impl(DispatchKey.Autograd)
# 以指定参数保留内存格式和确切的数据类型进行输出包装
@out_wrapper(preserve_memory_format=True, exact_dtype=True)
# 定义上采样最近邻精确方法，接受输入张量、输出尺寸和可选的比例参数（分别对应高度和宽度），返回张量
def _upsample_nearest_exact2d(
    input: Tensor,
    output_size: List[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    # 调用内部函数 _upsample_nearest 实现上采样最近邻操作，将高度和宽度的比例参数以及 exact=True 作为参数传递
    return _upsample_nearest(input, output_size, [scales_h, scales_w], exact=True)


# 注册上采样最近邻方法的默认实现和输出方法到分解列表中
@register_decomposition([aten.upsample_nearest3d.default, aten.upsample_nearest3d.out])
# 将默认实现与自动微分相关的分发键注释为上采样最近邻方法的实现
@aten.upsample_nearest3d.default.py_impl(DispatchKey.CompositeImplicitAutograd)
# 将默认实现与自动微分相关的分发键注释为上采样最近邻方法的实现
@aten.upsample_nearest3d.default.py_impl(DispatchKey.Autograd)
# 以指定参数保留内存格式和确切的数据类型进行输出包装
@out_wrapper(preserve_memory_format=True, exact_dtype=True)
# 定义上采样最近邻方法，接受输入张量、输出尺寸和可选的比例参数（分别对应深度、高度和宽度），返回张量
def upsample_nearest3d(
    input: Tensor,
    output_size: List[int],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    # 调用内部函数 _upsample_nearest 实现上采样最近邻操作，将深度、高度和宽度的比例参数作为列表传递
    return _upsample_nearest(input, output_size, [scales_d, scales_h, scales_w])


# 注册上采样最近邻精确方法的默认实现和输出方法到分解列表中
@register_decomposition(
    [aten._upsample_nearest_exact3d.default, aten._upsample_nearest_exact3d.out]
)
# 将默认实现与自动微分相关的分发键注释为上采样最近邻精确方法的实现
@aten._upsample_nearest_exact3d.default.py_impl(DispatchKey.CompositeImplicitAutograd)
# 将默认实现与自动微分相关的分发键注释为上采样最近邻精确方法的实现
@aten._upsample_nearest_exact3d.default.py_impl(DispatchKey.Autograd)
# 以指定参数保留内存格式和确切的数据类型进行输出包装
@out_wrapper(preserve_memory_format=True, exact_dtype=True)
# 定义一个函数，使用最近邻插值方法对输入的三维张量进行上采样，保持精确输出尺寸
def _upsample_nearest_exact3d(
    input: Tensor,
    output_size: List[int],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    # 调用_upsample_nearest函数进行最近邻上采样，传入精确标志为True
    return _upsample_nearest(
        input, output_size, [scales_d, scales_h, scales_w], exact=True
    )


# 用于最近邻插值上采样的函数
@pw_cast_for_opmath
def _upsample_nearest(
    input: Tensor,
    output_size: List[int],
    scales: List[Optional[float]],
    exact: bool = False,
) -> Tensor:
    # 计算最近邻上采样的空间索引
    spatial_indices = _compute_upsample_nearest_indices(
        input, output_size, scales, exact=exact
    )

    # 构造索引，前两个维度为空，后续维度为计算得到的空间索引
    indices = [None, None] + spatial_indices
    # 使用不安全的索引操作，从输入中获取结果
    result = aten._unsafe_index(input, indices)

    # 如果结果的维度为4，则根据输入推荐的内存格式进行转换
    if result.ndim == 4:
        # 推荐合适的内存格式
        memory_format = utils.suggest_memory_format(input)

        # 根据启发式规则，仅当channels_last路径比连续路径更快时才使用channels_last路径
        n_channels = input.shape[1]
        if input.device.type == "cuda" and n_channels < 4:
            memory_format = torch.contiguous_format

        # 转换结果为连续内存格式
        result = result.contiguous(memory_format=memory_format)
    return result


# 根据是否具有偏置和投影来分组参数
def gather_params(params, has_biases, has_projections):
    if has_biases and has_projections:
        group_size = 5
    elif has_biases:
        group_size = 4
    elif has_projections:
        group_size = 3
    else:
        group_size = 2

    # 断言参数的长度能够被group_size整除
    assert len(params) % group_size == 0, len(params)
    # 返回按照group_size分组的参数元组列表
    return [
        tuple(params[i : i + group_size]) for i in range(0, len(params), group_size)
    ]


# 根据是否双向来选择隐藏状态参数
def params_hiddens(params, hiddens, i, bidirectional):
    if bidirectional:
        cur_params, cur_hidden = params[2 * i], hiddens[2 * i]
        bidir_params, bidir_hidden = params[2 * i + 1], hiddens[2 * i + 1]
    else:
        cur_params, cur_hidden = params[i], hiddens[i]
        bidir_params, bidir_hidden = None, None

    return cur_params, cur_hidden, bidir_params, bidir_hidden


# 更新用于打包数据的隐藏状态
def update_hidden_for_packed(cur_hidden, last_batch_size, batch_size, hiddens):
    # 断言上一个批次大小大于当前批次大小
    assert last_batch_size > batch_size
    # 将当前隐藏状态的部分追加到隐藏状态列表中
    hiddens.append(cur_hidden.narrow(0, batch_size, last_batch_size - batch_size))
    # 返回截取后的当前隐藏状态，用于更新
    return cur_hidden.narrow(0, 0, batch_size)


# 更新用于反向打包数据的隐藏状态
def update_hidden_for_packed_reverse(
    cur_hidden, last_batch_size, batch_size, inp_hidden
):
    # 如果上一个批次大小等于当前批次大小，则直接返回当前隐藏状态
    if last_batch_size == batch_size:
        return cur_hidden
    # 断言上一个批次大小小于当前批次大小
    assert last_batch_size < batch_size
    # 拼接当前隐藏状态和输入隐藏状态的截取部分
    return torch.concat(
        (
            cur_hidden,
            inp_hidden.narrow(0, last_batch_size, batch_size - last_batch_size),
        )
    )


# 处理单层RNN的数据和隐藏状态
def one_layer_rnn_data(
    inp, hidden, params, has_biases, hidden_fn, batch_sizes, reverse=False
):
    # 获取输入到隐藏状态的权重
    ih_weight = params[0]
    hh_weight = params[1]
    # 如果有偏置，获取输入到隐藏状态和隐藏到隐藏状态的偏置
    ih_bias = params[2] if has_biases else None
    hh_bias = params[3] if has_biases else None

    # 步骤输出列表
    step_output = []
    # 隐藏状态列表
    hiddens: List[torch.Tensor] = []

    # 如果是反向处理，则获取最后一个批次大小，否则获取第一个批次大小
    last_batch_size = batch_sizes[-1] if reverse else batch_sizes[0]
    # 从 hidden 张量中取出前 last_batch_size 个元素，并赋值给 cur_hidden
    cur_hidden = hidden.narrow(0, 0, last_batch_size)
    # 根据 batch_sizes 列表将 inp 张量拆分成多个子张量组成的元组
    split_inp = torch.split(inp, list(batch_sizes))
    # 如果 reverse 标志为 True，则反转 split_inp 中的子张量顺序
    if reverse:
        split_inp = split_inp[::-1]
    # 遍历 split_inp 中的每个子张量 inp
    for inp in split_inp:
        # 获取当前子张量 inp 的长度 i
        i = inp.shape[0]

        # 如果当前子张量长度 i 等于 last_batch_size，则跳过更新 cur_hidden 的步骤
        if last_batch_size == i:
            pass  # 不更新 cur_hidden
        # 当 reverse=False 时执行，因为 batch_sizes 已按大小排序，这种情况只会发生一次
        elif reverse:
            # 调用函数 update_hidden_for_packed_reverse 更新 cur_hidden
            cur_hidden = update_hidden_for_packed_reverse(
                cur_hidden, last_batch_size, i, hidden
            )
        else:
            # 调用函数 update_hidden_for_packed 更新 cur_hidden
            cur_hidden = update_hidden_for_packed(
                cur_hidden, last_batch_size, i, hiddens
            )

        # 使用 hidden_fn 函数更新 cur_hidden
        cur_hidden = hidden_fn(inp, cur_hidden, ih_weight, ih_bias, hh_weight, hh_bias)
        # 更新 last_batch_size 为当前子张量 inp 的长度 i
        last_batch_size = i
        # 将当前 cur_hidden 加入到 step_output 列表中
        step_output.append(cur_hidden)

    # 如果 reverse=True，则反转 step_output 列表中的元素顺序
    if reverse:
        step_output.reverse()
    else:
        # 将最终的 cur_hidden 加入到 hiddens 列表中
        hiddens.append(cur_hidden)
        # 反转 hiddens 列表中的元素顺序
        hiddens.reverse()

    # 将 step_output 列表中的所有张量连接成一个张量 out
    out = torch.cat(step_output, 0)
    # 如果 reverse=True，则将 cur_hidden 作为 hidden_out 返回；否则将 hiddens 列表中的所有张量连接成一个张量作为 hidden_out 返回
    hidden_out = torch.cat(hiddens, 0) if not reverse else cur_hidden
    # 返回最终结果 out 和 hidden_out
    return out, hidden_out
# 定义一个RNN单元函数，根据给定的非线性函数返回一个内部函数
def rnn_cell(nonlinearity):
    def inner(i, cur_hidden, ih_weight, ih_bias, hh_weight, hh_bias):
        return nonlinearity(F.linear(cur_hidden, hh_weight, hh_bias) + i)
    return inner


# 定义一个RNN单元函数，将输入进行线性变换后再应用非线性函数，返回内部函数
def rnn_cell_data(nonlinearity):
    def inner(i, cur_hidden, ih_weight, ih_bias, hh_weight, hh_bias):
        i = F.linear(i, ih_weight, ih_bias)
        return nonlinearity(F.linear(cur_hidden, hh_weight, hh_bias) + i)
    return inner


# 定义一个单层RNN的函数，执行一系列计算，返回输出和最后一个隐藏状态
def one_layer_rnn(inp, hidden, params, has_biases, hidden_fn, reverse=False):
    ih_weight = params[0]
    hh_weight = params[1]
    ih_bias = params[2] if has_biases else None
    hh_bias = params[3] if has_biases else None

    # 对输入进行预计算的线性变换
    precomputed_input = F.linear(inp, ih_weight, ih_bias)
    precomputed_input = precomputed_input.flip(0) if reverse else precomputed_input
    cur_hidden = hidden.unsqueeze(0)
    step_output = []
    
    # 循环遍历预计算的输入，应用给定的隐藏函数，收集每一步的隐藏状态
    for i in precomputed_input:
        cur_hidden = hidden_fn(i, cur_hidden, ih_weight, ih_bias, hh_weight, hh_bias)
        step_output.append(cur_hidden)

    if reverse:
        step_output.reverse()  # 如果是反向遍历，将输出反转

    out = torch.cat(step_output, 0)  # 将所有步骤的输出连接起来

    return out, cur_hidden.squeeze(0)


# 定义一个使用MKLDNN优化的单层LSTM函数，执行MKLDNN的RNN操作，返回输出和最终的隐藏状态
def mkldnn_one_layer_lstm(inp, hidden, params, has_biases, reverse=False):
    w0 = params[0]
    w1 = params[1]
    if has_biases:
        w2 = params[2]
        w3 = params[3]
    else:
        w2 = torch.zeros(w0.size())
        w3 = torch.zeros(w1.size())

    hx = hidden[0].unsqueeze(0)
    cx = hidden[1].unsqueeze(0)

    batch_sizes: List[int] = []
    mode = 2  # 定义MKLDNN中LSTM模式的值为2

    hidden_size = hx.size(2)
    num_layers = 1

    # _rnn_helper函数已经处理了双向和batch_first，这里硬编码为False
    bidirectional = False
    batch_first = False

    train = False
    # 如果batch_first为True，则inp已在_rnn_helper中进行了排列。在此处进行连续化处理。
    inp = inp.contiguous()
    hx = hx.contiguous()
    cx = cx.contiguous()

    # 调用MKLDNN的RNN层操作函数，返回输出和最终的隐藏状态
    outputs = torch.ops.aten.mkldnn_rnn_layer.default(
        inp,
        w0,
        w1,
        w2,
        w3,
        hx,
        cx,
        reverse,
        batch_sizes,
        mode,
        hidden_size,
        num_layers,
        has_biases,
        bidirectional,
        batch_first,
        train,
    )
    y, hy, cy = outputs[0], outputs[1], outputs[2]
    return y, (hy.squeeze(0), cy.squeeze(0))


# 辅助函数，用于处理RNN操作，包括输入的转置和最终隐藏状态的处理
def _rnn_helper(
    input,
    hidden,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
    layer_fn,
):
    input = input.transpose(0, 1) if batch_first else input
    final_hiddens = []
    # 对神经网络模型的每一层进行迭代处理
    for i in range(num_layers):
        # 调用函数 params_hiddens 获取当前层的参数和隐藏状态
        cur_params, cur_hidden, bidir_params, bidir_hidden = params_hiddens(
            params, hidden, i, bidirectional
        )
        # 如果处于训练状态并且不是最后一层，则应用 dropout；否则 dropout 设为 0
        dropout = dropout if (train and num_layers < i - 1) else 0.0
        # 应用当前层的前向传播函数，计算前向输入和隐藏状态
        fwd_inp, fwd_hidden = layer_fn(input, cur_hidden, cur_params, has_biases)
        # 将当前层的隐藏状态添加到最终隐藏状态列表中
        final_hiddens.append(fwd_hidden)
    
        # 如果是双向模型，执行后向传播函数，并将后向隐藏状态添加到最终隐藏状态列表中
        if bidirectional:
            bwd_inp, bwd_hidden = layer_fn(
                input, bidir_hidden, bidir_params, has_biases, reverse=True
            )
            final_hiddens.append(bwd_hidden)
    
        # 如果是双向模型，将前向和后向输入张量连接起来
        if bidirectional:
            input = torch.cat([fwd_inp, bwd_inp], fwd_inp.dim() - 1)  # type: ignore[possibly-undefined]
        else:
            input = fwd_inp
    
        # 如果 dropout 率不为 0，并且处于训练状态且当前层不是最后一层，则应用 dropout
        if dropout != 0 and train and i < num_layers - 1:
            input = torch.dropout(input, dropout, train=True)
    
    # 如果设置了 batch_first，则转置输入张量的维度
    input = input.transpose(0, 1) if batch_first else input
    # 返回处理后的输入张量和最终的隐藏状态列表
    return input, final_hiddens
# 注册并定义了一个使用 Tanh 激活函数的 RNN 输入函数
@register_decomposition(aten.rnn_tanh.input)
@aten.rnn_tanh.input.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.rnn_tanh.input.py_impl(DispatchKey.Autograd)
def rnn_tanh_input(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    # 将初始隐藏状态拆解成单独的张量列表
    hidden = hx.unbind(0)
    # 根据参数收集相应的参数值和偏置
    params = gather_params(params, has_biases, False)
    # 调用帮助函数 `_rnn_helper` 来执行 RNN 操作，使用 Tanh 作为激活函数
    out, final_hiddens = _rnn_helper(
        input,
        hidden,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        partial(one_layer_rnn, hidden_fn=rnn_cell(torch.tanh)),
    )
    # 返回 RNN 输出和最终隐藏状态的堆叠张量
    return out, torch.stack(final_hiddens, 0)


# 注册并定义了一个使用 ReLU 激活函数的 RNN 输入函数
@register_decomposition(aten.rnn_relu.input)
@aten.rnn_relu.input.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.rnn_relu.input.py_impl(DispatchKey.Autograd)
def rnn_relu_input(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    # 将初始隐藏状态拆解成单独的张量列表
    hidden = hx.unbind(0)
    # 根据参数收集相应的参数值和偏置
    params = gather_params(params, has_biases, False)
    # 调用帮助函数 `_rnn_helper` 来执行 RNN 操作，使用 ReLU 作为激活函数
    out, final_hiddens = _rnn_helper(
        input,
        hidden,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        partial(one_layer_rnn, hidden_fn=rnn_cell(torch.relu)),
    )
    # 返回 RNN 输出和最终隐藏状态的堆叠张量
    return out, torch.stack(final_hiddens, 0)


# 注册并定义了一个使用 ReLU 激活函数的 RNN 数据处理函数
@register_decomposition(aten.rnn_relu.data)
@aten.rnn_relu.data.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.rnn_relu.data.py_impl(DispatchKey.Autograd)
def rnn_relu_data(
    data,
    batch_sizes,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
):
    # 将初始隐藏状态拆解成单独的张量列表
    hidden = hx.unbind(0)
    # 根据参数收集相应的参数值和偏置
    params = gather_params(params, has_biases, False)
    # 调用帮助函数 `_rnn_helper` 来执行 RNN 操作，使用 ReLU 作为激活函数
    out, final_hiddens = _rnn_helper(
        data,
        hidden,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        False,  # 不使用 batch_first
        partial(
            one_layer_rnn_data,
            batch_sizes=batch_sizes,
            hidden_fn=rnn_cell_data(torch.relu),
        ),
    )
    # 返回 RNN 输出和最终隐藏状态的堆叠张量
    return out, torch.stack(final_hiddens, 0)


# 注册并定义了一个使用 Tanh 激活函数的 RNN 数据处理函数
@register_decomposition(aten.rnn_tanh.data)
@aten.rnn_tanh.data.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.rnn_tanh.data.py_impl(DispatchKey.Autograd)
def rnn_tanh_data(
    data,
    batch_sizes,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
):
    # 将初始隐藏状态拆解成单独的张量列表
    hidden = hx.unbind(0)
    # 根据参数收集相应的参数值和偏置
    params = gather_params(params, has_biases, False)
    # 调用帮助函数 `_rnn_helper` 来执行 RNN 操作，使用 Tanh 作为激活函数
    out, final_hiddens = _rnn_helper(
        data,
        hidden,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        False,  # 不使用 batch_first
        partial(
            one_layer_rnn_data,
            batch_sizes=batch_sizes,
            hidden_fn=rnn_cell_data(torch.tanh),
        ),
    )
    # 返回 RNN 输出和最终隐藏状态的堆叠张量
    return out, torch.stack(final_hiddens, 0)
# 定义一个 LSTM 单元的函数，执行单步的 LSTM 运算
def lstm_cell(inp, hx, cx, hh_weight, hh_bias, hr_weight, chunk_dim):
    # 计算门控信息：输入门、遗忘门、细胞状态更新门、输出门
    gates = F.linear(hx, hh_weight, hh_bias) + inp
    # 将门控信息按照 chunk_dim 分块，通常为4块
    chunked_gates = gates.chunk(4, chunk_dim)
    # 计算输入门的 sigmoid 激活值
    in_gate = chunked_gates[0].sigmoid()
    # 计算遗忘门的 sigmoid 激活值
    forget_gate = chunked_gates[1].sigmoid()
    # 计算细胞状态的 tanh 激活值
    cell_gate = chunked_gates[2].tanh()
    # 计算输出门的 sigmoid 激活值
    out_gate = chunked_gates[3].sigmoid()
    # 根据门控信息更新细胞状态
    cy = forget_gate * cx + (in_gate * cell_gate)
    # 根据细胞状态和输出门的激活值更新隐藏状态
    hy = out_gate * cy.tanh()
    # 如果存在 hr_weight，则对隐藏状态进行线性变换
    hy = hy if hr_weight is None else F.linear(hy, hr_weight, None)

    return hy, cy


# 定义一个单层 LSTM 的函数，处理整个输入序列
def one_layer_lstm(inp, hidden, params, has_biases, reverse=False):
    # 提取参数
    ih_weight = params[0]
    hh_weight = params[1]
    ih_bias = params[2] if has_biases else None
    hh_bias = params[3] if has_biases else None
    hr_weight = (
        params[4] if len(params) == 5 else params[2] if len(params) == 3 else None
    )

    # 初始化隐藏状态和细胞状态
    hx = hidden[0].unsqueeze(0)
    cx = hidden[1].unsqueeze(0)

    # 对输入数据进行预计算
    precomputed_input = F.linear(inp, ih_weight, ih_bias)
    # 如果 reverse 为真，则翻转预计算的输入数据
    precomputed_input = precomputed_input.flip(0) if reverse else precomputed_input
    step_output = []

    # 逐步处理每个输入序列
    for inp in precomputed_input:
        # 执行 LSTM 单元运算，更新隐藏状态和细胞状态
        hx, cx = lstm_cell(inp, hx, cx, hh_weight, hh_bias, hr_weight, chunk_dim=2)
        # 将当前步骤的隐藏状态添加到输出列表中
        step_output.append(hx)

    # 如果 reverse 为真，则反转输出列表
    if reverse:
        step_output.reverse()

    # 将所有步骤的输出连接起来形成最终输出
    out = torch.cat(step_output, 0)

    return out, (hx.squeeze(1), cx.squeeze(1))


# 定义一个支持批处理的单层 LSTM 函数，处理变长输入序列
def one_layer_lstm_data(inp, hidden, params, has_biases, batch_sizes, reverse=False):
    # 提取参数
    ih_weight = params[0]
    hh_weight = params[1]
    ih_bias = params[2] if has_biases else None
    hh_bias = params[3] if has_biases else None
    hr_weight = (
        params[4] if len(params) == 5 else params[2] if len(params) == 3 else None
    )

    step_output = []
    hiddens = []

    # 确定最后一个批次的大小
    last_batch_size = batch_sizes[-1] if reverse else batch_sizes[0]
    # 根据 reverse 的值重新排列输入数据
    split_inp = torch.split(inp, list(batch_sizes))
    if reverse:
        split_inp = split_inp[::-1]

    # 提取原始的隐藏状态和细胞状态
    orig_hx = hidden[0]
    orig_cx = hidden[1]
    # 根据方向选择初始隐藏状态和细胞状态的一部分
    hx, cx = orig_hx.narrow(0, 0, last_batch_size), orig_cx.narrow(
        0, 0, last_batch_size
    )
    # 遍历分割后的输入序列中的每个元素
    for inp in split_inp:
        # 获取当前输入的批次大小
        i = inp.shape[0]
        # 对当前输入进行线性变换，使用ih_weight和ih_bias
        inp = F.linear(inp, ih_weight, ih_bias)

        # 当reverse=False时，只有在批次大小按降序排列时才会发生
        if i < last_batch_size:
            # 将当前时刻的隐藏状态和细胞状态部分切片存入hiddens列表中
            hiddens.append(
                (
                    hx.narrow(0, i, last_batch_size - i),
                    cx.narrow(0, i, last_batch_size - i),
                )
            )
            # 更新hx和cx，只保留前i个元素的部分
            hx, cx = hx.narrow(0, 0, i), cx.narrow(0, 0, i)

        # 当reverse=True时，只有在批次大小按升序排列时才会发生
        if i > last_batch_size:
            # 将原始hx的后面部分和cx的后面部分拼接到hx和cx中
            hx = torch.concat(
                (hx, orig_hx.narrow(0, last_batch_size, i - last_batch_size)), 0
            )
            cx = torch.concat(
                (cx, orig_cx.narrow(0, last_batch_size, i - last_batch_size)), 0
            )

        # 执行LSTM单元操作，更新hx和cx
        hx, cx = lstm_cell(inp, hx, cx, hh_weight, hh_bias, hr_weight, chunk_dim=1)
        # 更新最后一个批次大小为当前批次大小
        last_batch_size = i
        # 将当前时刻的hx添加到step_output列表中
        step_output.append(hx)

    # 如果reverse=True，反转step_output列表
    if reverse:
        step_output.reverse()
        # 将最终的hx和cx作为输出的隐藏状态
        hidden_out = (hx, cx)
    else:
        # 将最后一个时刻的hx和cx添加到hiddens列表中
        hiddens.append((hx, cx))
        # 反转hiddens列表
        hiddens.reverse()
        # 将hiddens列表拆分为hidden0和hidden1，然后拼接它们
        hidden0, hidden1 = zip(*hiddens)
        hidden_out = torch.cat(hidden0, 0), torch.cat(hidden1, 0)

    # 将step_output列表中的所有元素拼接成一个张量作为输出
    out = torch.cat(step_output, 0)
    # 返回最终的输出张量和隐藏状态
    return out, hidden_out
# 定义一个函数，用于检查是否可以使用 `mkldnn_rnn_layer` 来分解 LSTM 函数。
# 所有以下条件必须满足才能使用该函数：
# * `torch._C._get_mkldnn_enabled()` 返回 `True`。
# * 所有输入参数都位于 CPU 上。
# * 参数的数据类型是 `torch.float` 或 `torch.bfloat16`。
# * 处于推断模式。
# * `has_projections` 返回 `False`。

def select_one_layer_lstm_function(input, hx, params):
    r"""Check whether we could use decompose lstm with mkldnn_rnn_layer.
    All the below conditions need to be met:
        * ``torch._C._get_mkldnn_enabled()`` returns ``True``.
        * All the input args are on CPU.
        * The dtypes of args are either torch.float or torch.bfloat16.
        * Inference.
        * ``has_projections`` returns ``False``.

    Args:
        * input: the input sequence to LSTM
        * hx: a tuple of the input hidden state and cell state ``(h_0, c_0)`` to LSTM
        * params: the weight and bias tensors of LSTM
    """

    # 内部函数，用于检查是否可以使用 `mkldnn_rnn_layer`
    def use_mkldnn(input, hx, params):
        # 检查是否启用了 MKLDNN 加速
        if not torch._C._get_mkldnn_enabled():
            return False

        # 收集所有相关的张量，包括输入、隐藏状态和参数
        tensors = [input] + list(hx) + list(chain.from_iterable(params))
        # 检查所有张量是否在同一个设备上
        devices = {t.device for t in tensors}
        if len(devices) != 1:
            return False

        # 检查设备是否为 CPU
        device = devices.pop()
        if device != torch.device("cpu"):
            return False

        # 检查所有张量的数据类型是否为 torch.float 或 torch.bfloat16
        dtypes = {t.dtype for t in tensors}
        for dtype in dtypes:
            if dtype not in [torch.float, torch.bfloat16]:
                return False

        # 检查输入张量是否需要梯度计算
        if input.requires_grad:
            return False

        # 检查是否有投影层
        has_projections = hx[0].size(2) != hx[1].size(2)
        if has_projections:
            return False

        return True

    # 如果可以使用 `mkldnn_rnn_layer`，返回对应的函数
    if use_mkldnn(input, hx, params):
        return mkldnn_one_layer_lstm
    else:
        return one_layer_lstm


# 为 `lstm_impl` 函数注册不同的分解实现方法
@register_decomposition(aten.lstm.input)
@aten.lstm.input.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.lstm.input.py_impl(DispatchKey.Autograd)
def lstm_impl(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    # 断言隐藏状态的长度为 2，即 LSTM 需要两个隐藏状态
    assert len(hx) == 2, "lstm expects two hidden states"
    # 根据是否有投影层，收集相应的参数
    params = gather_params(params, has_biases, hx[0].size(2) != hx[1].size(2))
    # 将隐藏状态组合成列表
    hidden = list(zip(hx[0], hx[1]))
    # 选择要使用的单层 LSTM 函数
    layer_fn = select_one_layer_lstm_function(input, hx, params)
    # 调用辅助函数 `_rnn_helper` 执行 LSTM 操作
    out, final_hiddens = _rnn_helper(
        input,
        hidden,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        layer_fn,
    )
    # 将最终的隐藏状态转换成列表形式
    final_hiddens = list(zip(*final_hiddens))
    # 返回输出结果和最终的隐藏状态
    return out, torch.stack(final_hiddens[0], 0), torch.stack(final_hiddens[1], 0)


# 为 `lstm_data_impl` 函数注册不同的数据分解实现方法
@register_decomposition(aten.lstm.data)
@aten.lstm.data.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.lstm.data.py_impl(DispatchKey.Autograd)
def lstm_data_impl(
    data,
    batch_sizes,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
):
    # 断言隐藏状态的长度为 2，即 LSTM 需要两个隐藏状态
    assert len(hx) == 2, "lstm expects two hidden states"
    # 根据是否有投影层，收集相应的参数
    params = gather_params(params, has_biases, hx[0].size(2) != hx[1].size(2))
    # 将 hx[0] 和 hx[1] 组合成一个元组列表
    hidden = list(zip(hx[0], hx[1]))
    
    # 调用 _rnn_helper 函数进行循环神经网络的计算
    # data: 输入数据
    # hidden: 初始隐藏状态
    # params: RNN 参数
    # has_biases: 是否包含偏置项
    # num_layers: RNN 的层数
    # dropout: 丢弃率
    # train: 是否在训练模式下
    # bidirectional: 是否使用双向 RNN
    # False: 不使用反向传播计算
    # partial(one_layer_lstm_data, batch_sizes=batch_sizes): 部分应用函数，用于传递给 _rnn_helper
    out, final_hiddens = _rnn_helper(
        data,
        hidden,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        False,
        partial(one_layer_lstm_data, batch_sizes=batch_sizes),
    )
    
    # 将 final_hiddens 中的数据转置，并且转换成列表
    final_hiddens = list(zip(*final_hiddens))
    
    # 返回计算结果，包括输出 out 和转置后的隐藏状态堆叠起来的张量
    return out, torch.stack(final_hiddens[0], 0), torch.stack(final_hiddens[1], 0)
# 定义一个 GRU 单元函数，实现 GRU 单元的前向计算
def gru_cell(inp, cur_hidden, ih_weight, ih_bias, hh_weight, hh_bias):
    # 将输入 inp 按第一个维度分块成 3 个部分，分别表示重置门、更新门和新的候选值
    chunked_igates = inp.chunk(3, 1)
    # 使用当前隐藏状态 cur_hidden 与隐藏层权重 hh_weight 和偏置 hh_bias 计算线性变换，
    # 然后按第二个维度分块成 3 个部分，分别表示重置门、更新门和新的候选值
    chunked_hgates = F.linear(cur_hidden, hh_weight, hh_bias).chunk(3, 2)
    # 计算重置门，使用当前隐藏状态的重置门部分和输入的重置门部分，通过 sigmoid 函数激活
    reset_gate = (chunked_hgates[0] + chunked_igates[0]).sigmoid()
    # 计算更新门，使用当前隐藏状态的更新门部分和输入的更新门部分，通过 sigmoid 函数激活
    input_gate = (chunked_hgates[1] + chunked_igates[1]).sigmoid()
    # 计算新的候选值，使用输入的新的候选值部分和当前隐藏状态的重置门部分，
    # 然后通过 tanh 函数激活得到新的候选值
    new_gate = (chunked_igates[2] + (chunked_hgates[2] * reset_gate)).tanh()
    # 返回更新后的隐藏状态
    return (cur_hidden - new_gate) * input_gate + new_gate


# 定义一个 GRU 单元函数，输入数据已经与输入权重和偏置进行了线性变换
def gru_cell_data(inp, cur_hidden, ih_weight, ih_bias, hh_weight, hh_bias):
    # 将输入 inp 与输入权重 ih_weight 和偏置 ih_bias 计算线性变换后，按第一个维度分块成 3 个部分，
    # 分别表示重置门、更新门和新的候选值
    chunked_igates = F.linear(inp, ih_weight, ih_bias).chunk(3, 1)
    # 使用当前隐藏状态 cur_hidden 与隐藏层权重 hh_weight 和偏置 hh_bias 计算线性变换，
    # 然后按第一个维度分块成 3 个部分，分别表示重置门、更新门和新的候选值
    chunked_hgates = F.linear(cur_hidden, hh_weight, hh_bias).chunk(3, 1)
    # 计算重置门，使用当前隐藏状态的重置门部分和输入的重置门部分，通过 sigmoid 函数激活
    reset_gate = (chunked_hgates[0] + chunked_igates[0]).sigmoid()
    # 计算更新门，使用当前隐藏状态的更新门部分和输入的更新门部分，通过 sigmoid 函数激活
    input_gate = (chunked_hgates[1] + chunked_igates[1]).sigmoid()
    # 计算新的候选值，使用输入的新的候选值部分和当前隐藏状态的重置门部分，
    # 然后通过 tanh 函数激活得到新的候选值
    new_gate = (chunked_igates[2] + (chunked_hgates[2] * reset_gate)).tanh()
    # 返回更新后的隐藏状态
    return (cur_hidden - new_gate) * input_gate + new_gate


# 注册对数据输入进行 GRU 实现的分解，使用给定的分解器和实现方式
@register_decomposition(aten.gru.data)
@aten.gru.data.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.gru.data.py_impl(DispatchKey.Autograd)
def gru_impl_data(
    data,
    batch_sizes,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
):
    # 从参数中收集参数，根据是否有偏置选择是否为 GRU
    params = gather_params(params, has_biases, False)
    # 使用辅助函数 _rnn_helper 进行 RNN 的计算，
    # 使用 one_layer_rnn_data 作为内部单层 RNN 的处理函数
    out, final_hiddens = _rnn_helper(
        data,
        hx.unbind(0),  # 将隐藏状态按照第一个维度解绑
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        False,  # 不使用 batch_first
        partial(one_layer_rnn_data, batch_sizes=batch_sizes, hidden_fn=gru_cell_data),  # 使用 gru_cell_data 作为隐藏层函数
    )
    # 返回输出结果和最终隐藏状态堆叠后的张量
    return out, torch.stack(final_hiddens, 0)


# 注册对输入进行 GRU 实现的分解，使用给定的分解器和实现方式
@register_decomposition(aten.gru.input)
@aten.gru.input.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.gru.input.py_impl(DispatchKey.Autograd)
def gru_impl(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    # 从参数中收集参数，根据是否有偏置选择是否为 GRU
    params = gather_params(params, has_biases, False)
    # 使用辅助函数 _rnn_helper 进行 RNN 的计算，
    # 使用 one_layer_rnn 作为内部单层 RNN 的处理函数
    out, final_hiddens = _rnn_helper(
        input,
        hx.unbind(0),  # 将隐藏状态按照第一个维度解绑
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        partial(one_layer_rnn, hidden_fn=gru_cell),  # 使用 gru_cell 作为隐藏层函数
    )
    # 返回输出结果和最终隐藏状态堆叠后的张量
    return out, torch.stack(final_hiddens, 0)


# 注册对双线性 2D 上采样的实现，使用给定的输入、输出尺寸、对齐方式和尺度因子
@register_decomposition(aten._upsample_bilinear2d_aa.vec)
@aten._upsample_bilinear2d_aa.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten._upsample_bilinear2d_aa.vec.py_impl(DispatchKey.Autograd)
def upsample_bilinear2d_aa_vec(input, output_size, align_corners, scale_factors):
    # 计算输出尺寸
    osize = upsample_compute_output_size(input.size(), output_size, scale_factors)
    # 获取高度和宽度的尺度因子
    scale_h = get_scale_value(scale_factors, 0)
    scale_w = get_scale_value(scale_factors, 1)
    # 调用底层的 C++ 函数实现双线性 2D 上采样
    return torch.ops.aten._upsample_bilinear2d_aa(
        input, osize, align_corners, scale_h, scale_w
    )


# 注册对双立方 2D 上采样的实现，使用给定的输入、输出尺寸、对齐方式和尺度因子
@register_decomposition(aten._upsample_bicubic2d_aa.vec)
@aten._upsample_bicubic2d_aa.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
# 定义函数 upsample_bicubic2d_aa_vec，使用 bicubic 双线性插值方法对 2D 输入进行上采样
@aten._upsample_bicubic2d_aa.vec.py_impl(DispatchKey.Autograd)
def upsample_bicubic2d_aa_vec(input, output_size, align_corners, scale_factors):
    # 计算输出大小
    osize = upsample_compute_output_size(input.size(), output_size, scale_factors)
    # 获取高度和宽度的缩放因子
    scale_h = get_scale_value(scale_factors, 0)
    scale_w = get_scale_value(scale_factors, 1)
    # 调用 Torch 的 C++ 扩展函数实现 bicubic 双线性插值上采样
    return torch.ops.aten._upsample_bicubic2d_aa(
        input, osize, align_corners, scale_h, scale_w
    )


# 定义函数 _upsample_linear_vec，用于线性插值上采样操作
@register_decomposition(aten.upsample_bilinear2d.vec)
@register_decomposition(aten.upsample_trilinear3d.vec)
@aten.upsample_linear1d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_linear1d.vec.py_impl(DispatchKey.Autograd)
@aten.upsample_bilinear2d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_bilinear2d.vec.py_impl(DispatchKey.Autograd)
@aten.upsample_trilinear3d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_trilinear3d.vec.py_impl(DispatchKey.Autograd)
def _upsample_linear_vec(input, output_size, align_corners, scale_factors):
    # 计算输出大小
    osize = upsample_compute_output_size(input.size(), output_size, scale_factors)
    # 如果有指定缩放因子，将其作为 scales；否则用 None 填充
    scales = scale_factors if scale_factors else [None] * len(osize)
    # 调用 _upsample_linear 函数进行线性插值上采样
    return _upsample_linear(input, osize, align_corners, scales)


# 定义函数 upsample_linear1d，用于 1D 线性插值上采样操作
@register_decomposition(
    [aten.upsample_linear1d.default, aten.upsample_linear1d.out]
)
@out_wrapper()
def upsample_linear1d(
    input: Tensor,
    output_size: List[int],
    align_corners: bool,
    scales_w: Optional[float] = None,
) -> Tensor:
    # 调用 _upsample_linear 函数进行 1D 线性插值上采样
    return _upsample_linear(input, output_size, align_corners, [scales_w])


# 定义函数 upsample_bilinear2d，用于 2D 双线性插值上采样操作
@register_decomposition(
    [aten.upsample_bilinear2d.default, aten.upsample_bilinear2d.out]
)
@aten.upsample_bilinear2d.default.py_impl(DispatchKey.Autograd)
@out_wrapper()
def upsample_bilinear2d(
    input: Tensor,
    output_size: List[int],
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    # 调用 _upsample_linear 函数进行 2D 双线性插值上采样
    return _upsample_linear(input, output_size, align_corners, [scales_h, scales_w])


# 定义函数 upsample_trilinear3d，用于 3D 三线性插值上采样操作
@register_decomposition(
    [aten.upsample_trilinear3d.default, aten.upsample_trilinear3d.out]
)
@out_wrapper()
def upsample_trilinear3d(
    input: Tensor,
    output_size: List[int],
    align_corners: bool,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    # 调用 _upsample_linear 函数进行 3D 三线性插值上采样
    return _upsample_linear(
        input, output_size, align_corners, [scales_d, scales_h, scales_w]
    )


# 定义函数 _compute_scale，用于计算缩放比例
def _compute_scale(in_size, out_size, align_corners, scale=None):
    if align_corners:
        # 根据 align_corners 的情况计算缩放比例
        return (in_size - 1.0) / (out_size - 1.0) if out_size > 1 else 0
    else:
        # 如果不使用 align_corners，则根据指定的 scale 或输入输出尺寸计算比例
        return 1.0 / scale if scale is not None and scale > 0 else in_size / out_size


# 定义函数 _compute_source_index，用于计算源索引位置
def _compute_source_index(scale, dst_index, align_corners):
    if align_corners:
        # 如果使用 align_corners，直接按缩放比例计算源索引位置
        return scale * dst_index
    else:
        # 如果不使用 align_corners，计算中心化的源索引位置
        return scale * (dst_index + 0.5) - 0.5


# 定义函数 _sum_tensors_uint8，待实现，用于对多个 uint8 类型张量求和
    src: Iterable[Tensor], weights: Iterable[Tensor], weights_precision: Tensor


    # 接收参数 src、weights 和 weights_precision，它们的类型分别是 Iterable[Tensor]、Iterable[Tensor] 和 Tensor
@pw_cast_for_opmath
def _upsample_linear(
    input: Tensor,
    output_size: List[int],
    align_corners: bool,
    scales: List[Optional[float]],
) -> Tensor:
    # 获取原始图像的批次数和通道数
    n_batch, n_channels = input.shape[:2]
    # 获取输入图像的尺寸
    inp_sizes = input.shape[2:]
    # 获取输入数据类型
    _, dtype = utils.elementwise_dtypes(
        input,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )

    def get_values(inp_size, out_size, scales, nsqueeze):
        # 首先计算缩放因子
        scale_factor = _compute_scale(inp_size, out_size, align_corners, scales)
        # 创建具有 int64 数据类型的 arange，通过 .to 方法避免在诱导器中额外创建内核并导致性能下降
        i = torch.arange(out_size, device=input.device).to(dtype=dtype)

        # 计算源索引
        x_f32 = _compute_source_index(scale_factor, i, align_corners).clamp(min=0.0)
        # 将 x_f32 重塑为与输入数据尺寸匹配的形状
        x_f32 = x_f32.reshape(x_f32.shape[0], *[1] * (nsqueeze))
        # 将 x_f32 转换为 int64 数据类型
        x = x_f32.to(torch.int64)
        # 计算 x + 1，并限制其最大值为 inp_size - 1
        xp1 = (x + 1).clamp(max=inp_size - 1)
        return x_f32, x, xp1

    # 获取所有维度的值
    values = [
        get_values(inp_size, out_size, scales, n_dims - 1 - i)
        for i, (inp_size, out_size, scales) in enumerate(
            zip(inp_sizes, output_size, scales)
        )
    ]
    # 将返回的结果解压缩成 xs_f32, xs, xp1s
    xs_f32, xs, xp1s = list(zip(*values))

    vs = []
    # 对于每个维度的所有组合，执行以下操作
    for a in product(*[[0, 1]] * n_dims):
        # 根据索引创建切片
        idx = [None, None] + [xs[k] if a[k] == 0 else xp1s[k] for k in range(n_dims)]
        # 使用 _unsafe_index 获取张量的值
        v = aten._unsafe_index(input, idx)
        # 将 v 转换为指定的数据类型 dtype
        v = _maybe_convert_to_dtype(v, dtype)
        # 将结果添加到 vs 中
        vs.append(v)

    # 对维度进行反向遍历
    for i in reversed(range(n_dims)):
        # 计算 xscale，范围在 0.0 到 1.0 之间，并转换为指定的数据类型 dtype
        xscale = (xs_f32[i] - xs[i]).clamp(0.0, 1.0).to(dtype)
        # 执行线性插值操作
        vs = [
            # x1 * (1 - alpha) + x2 * alpha == x1 + (x2 - x1) * alpha
            v1 + torch.mul(v2 - v1, xscale)
            for v1, v2 in zip(vs[::2], vs[1::2])
        ]

    # 断言 vs 中只有一个元素
    assert len(vs) == 1
    # 获取最终结果
    result = vs[0]

    # 将输出转换为正确的内存格式（如果需要）
    memory_format = utils.suggest_memory_format(input)

    # 根据启发式规则选择内存格式路径
    if input.device.type == "cuda" and n_channels < 16:
        memory_format = torch.contiguous_format

    # 断言 result 是 torch.Tensor 类型
    assert isinstance(result, torch.Tensor)

    # 使用指定的内存格式对结果进行连续性处理
    result = result.contiguous(memory_format=memory_format)
    # 检查输入是否不是浮点数（即输入是整数）
    if not input.is_floating_point():
        # 如果输入不是浮点数，对结果进行四舍五入处理
        result = result.round()

    # 返回处理后的结果
    return result
# 在所有变换之后应用分解
@register_decomposition(aten.is_same_size.default)
def is_same_size(a: Tensor, b: Tensor) -> bool:
    # 检查两个张量的形状是否相同
    return a.shape == b.shape


@register_decomposition([aten._reshape_alias, aten._unsafe_view])
@out_wrapper()
def _reshape_alias(x, shape, *args):
    # 调用 ATen 库中的 view 函数对张量 x 进行形状重塑
    return aten.view(x, shape)


@register_decomposition([aten._unsafe_index])
def _unsafe_index(x, indices):
    # 使用 ATen 库中的 index 函数对张量 x 进行索引操作
    return aten.index(x, indices)


@register_decomposition([aten._unsafe_masked_index])
def _unsafe_masked_index(x, mask, indices, fill):
    # 检查 indices 张量的类型，确保为 long 或 int 类型
    for index in indices:
        if index is not None:
            torch._check(
                index.dtype in [torch.long, torch.int],
                lambda: "tensors used as indices must be long or int tensors",
            )

    # 检查 mask 张量的类型，确保为布尔类型
    torch._check(
        mask.dtype == torch.bool,
        lambda: "tensors used as masks must be bool tensors",
    )

    # 如果张量 x 的元素数为 0，则返回填充值为 fill 的元素张量
    if x.numel() == 0:
        meta_result = torch._meta_registrations.meta_index_Tensor(x, indices)
        return x.new_full(meta_result.shape, fill)

    # 对 indices 中的每个索引进行范围限制，确保不超出张量 x 的边界
    for i in range(len(indices)):
        index = indices[i]
        if index is not None:
            indices[i] = index.clamp(min=0, max=x.size(i) - 1)

    # 调用 ATen 库中的 _unsafe_index 函数进行索引操作，并根据 mask 进行填充操作
    return aten._unsafe_index(x, indices).masked_fill(~mask, fill)


@register_decomposition([aten._unsafe_masked_index_put_accumulate])
def _unsafe_masked_index_put_accumulate(x, mask, indices, values):
    # 检查 indices 张量的类型，确保为 long 或 int 类型
    for index in indices:
        if index is not None:
            torch._check(
                index.dtype in [torch.long, torch.int],
                lambda: "tensors used as indices must be long or int tensors",
            )

    # 检查 mask 张量的类型，确保为布尔类型
    torch._check(
        mask.dtype == torch.bool,
        lambda: "tensors used as masks must be bool tensors",
    )

    # 如果张量 x 的元素数为 0，则直接克隆该张量并返回
    if x.numel() == 0:
        return x.clone()

    # 对 indices 中的每个索引进行范围限制，确保在有效范围内
    for i in range(len(indices)):
        index = indices[i]
        if index is not None:
            indices[i] = index.clamp(min=-x.size(i), max=x.size(i) - 1)

    # 根据 mask 对 values 进行填充操作，并调用 ATen 库中的 _unsafe_index_put 函数进行累积更新
    masked_value = values.masked_fill(~mask, 0)
    return aten._unsafe_index_put(x, indices, masked_value, accumulate=True)


def _nll_loss_forward(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
) -> Tuple[Tensor, Tensor]:
    # self 可能是 [N, C] 或 [C]
    # target 可能是 [N] 或 []

    # 获取张量 self 的维度数
    n_dims = self.dim()
    channel_dim = 1
    if n_dims < 2:
        channel_dim = 0

    # 如果 weight 不为 None，则对 self 进行加权处理
    if weight is not None:
        if n_dims > 1:
            # 根据 weight 的形状对 self 进行广播
            shape = [
                1,
            ] * n_dims
            shape[channel_dim] = weight.shape[0]
            w = weight.view(shape)
        else:
            w = weight
        self = self * w

    # 对 target 进行安全处理，将 ignore_index 的位置替换为 0
    safe_target = torch.where(target != ignore_index, target, 0)
    safe_target_ = safe_target.unsqueeze(channel_dim)
    # target 可能是 [N, 1] 或 [1]

    # 使用 gather 函数从 self 中按 channel_dim 维度聚合 safe_target_ 所指示的索引值
    result = -torch.gather(self, channel_dim, safe_target_).squeeze(channel_dim)

    # 将 ignore_index 的位置替换为 0
    result = torch.where(target != ignore_index, result, 0)
    # 如果 reduction 的值为 NONE 且 n_dims 大于 1，则执行以下逻辑
    if reduction == Reduction.NONE.value and n_dims > 1:
        # 创建一个值为 0.0 的标量张量作为 total_weight
        total_weight = self.new_full((), 0.0)
        # 返回计算结果和 total_weight
        return result, total_weight

    # 如果 weight 参数不为空
    if weight is not None:
        # 将 w 扩展为与 self.shape 相同的形状
        w = w.expand(self.shape)
        # 使用 torch.gather 在 channel_dim 维度上收集 w 中的值到 safe_target_，然后在 channel_dim 维度上挤压
        wsum = torch.gather(w, channel_dim, safe_target_).squeeze(channel_dim)
        # 使用 torch.where 将 target 等于 ignore_index 的位置置为 0
        wsum = torch.where(target != ignore_index, wsum, 0)
        # 计算 wsum 中的所有元素之和，作为 total_weight
        total_weight = wsum.sum()
    else:
        # 计算 target 中不等于 ignore_index 的元素数量，并转换为当前对象的数据类型
        total_weight = (target != ignore_index).sum().to(self)

    # 根据 reduction 参数的不同值对 result 进行相应的汇总操作
    if reduction == Reduction.SUM.value:
        result = result.sum()
    elif reduction == Reduction.MEAN.value:
        # 将 result 求和后除以 total_weight，计算均值
        result = result.sum() / total_weight

    # 返回计算结果 result 和 total_weight
    return result, total_weight
# 注册 nll_loss_forward 函数的装饰器，指定使用 aten.nll_loss_forward 函数进行分解
# 同时将输出包装为 ("output", "total_weight") 形式的元组
@register_decomposition(aten.nll_loss_forward)
@out_wrapper("output", "total_weight")
def nll_loss_forward(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
) -> Tuple[Tensor, Tensor]:
    # 断言输入张量的维度在1到2之间，用于检查输入张量的正确性
    assert self.dim() > 0 and self.dim() <= 2, "input tensor should be 1D or 2D"
    # 断言目标张量的维度不大于1，仅支持0维或1维目标张量，不支持多目标张量
    assert (
        target.dim() <= 1
    ), "0D or 1D target tensor expected, multi-target not supported"

    # 判断是否没有批次维度，即输入张量是1维且目标张量是0维
    no_batch_dim = self.dim() == 1 and target.dim() == 0
    # 断言没有批次维度或者输入张量和目标张量的第一维大小相同
    assert no_batch_dim or (
        self.shape[0] == target.shape[0]
    ), f"size mismatch (got input: {self.shape}, target: {target.shape})"

    # 计算类别数，即输入张量的最后一维的大小
    n_classes = self.shape[-1]

    # 断言权重张量为None或者其维度为1且大小与类别数相同
    assert weight is None or (
        weight.dim() == 1 and weight.numel() == n_classes
    ), f"weight tensor should be defined either for all {n_classes} classes or no classes but got weight tensor of shape: {weight.shape}"  # noqa: B950

    # 调用 _nll_loss_forward 函数进行实际的负对数似然损失计算
    return _nll_loss_forward(self, target, weight, reduction, ignore_index)


# 注册 nll_loss2d_forward 函数的装饰器，指定使用 aten.nll_loss2d_forward 函数进行分解
# 同时将输出包装为 ("output", "total_weight") 形式的元组
@register_decomposition(aten.nll_loss2d_forward)
@out_wrapper("output", "total_weight")
def nll_loss2d_forward(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
) -> Tuple[Tensor, Tensor]:
    # 调用 _nll_loss_forward 函数进行实际的二维负对数似然损失计算
    return _nll_loss_forward(self, target, weight, reduction, ignore_index)


# 这些函数是从 aten/src/ATen/native/UpSample.h 改编而来，其基础是双三次插值的卷积算法
# 参考自 https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
def _upsample_cubic_convolution1(x: Tensor, A: float) -> Tensor:
    # 第一个双三次插值卷积函数的实现
    return ((A + 2) * x - (A + 3)) * x * x + 1


def _upsample_cubic_convolution2(x: Tensor, A: float) -> Tensor:
    # 第二个双三次插值卷积函数的实现
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A


def _upsample_get_cubic_coefficients(t: Tensor) -> TensorSequenceType:
    A = -0.75

    # 根据张量 t 的设备类型，选择不同的实现路径
    if t.device == torch.device("cpu"):
        # 在 CPU 上，堆叠 t 和 1-t，用于双三次插值卷积系数计算
        tt1 = torch.stack([t, 1.0 - t], dim=0)
        tt2 = torch.stack([t + 1.0, 2.0 - t], dim=0)
        # 计算双三次插值卷积系数
        w03 = _upsample_cubic_convolution2(tt2, A)
        w12 = _upsample_cubic_convolution1(tt1, A)
        w0, w3 = torch.unbind(w03, dim=0)
        w1, w2 = torch.unbind(w12, dim=0)
        return w0, w1, w2, w3
    else:
        # 在其他设备上，直接计算双三次插值卷积系数
        return (
            _upsample_cubic_convolution2(t + 1.0, A),
            _upsample_cubic_convolution1(t, A),
            _upsample_cubic_convolution1(1.0 - t, A),
            _upsample_cubic_convolution2(2.0 - t, A),
        )


def _upsample_cubic_interp1d(coeffs: TensorSequenceType, ts: Tensor) -> Tensor:
    # 获取双三次插值的系数
    coeffs2 = _upsample_get_cubic_coefficients(ts)
    # 对两组系数进行双三次插值计算
    return _sum_tensors(c1 * c2 for (c1, c2) in zip(coeffs, coeffs2))


# 用于替代 sum() 以满足 mypy 的要求，对输入的张量列表进行逐元素相加求和
def _sum_tensors(ts: Iterable[Tensor]) -> Tensor:
    return reduce(torch.add, ts)


# 从 -1 开始生成均匀间隔的线性空间
def _linspace_from_neg_one(
    num_steps: int, align_corners: bool, dtype: torch.dtype, device: torch.device
):
    # 如果步数小于等于1，返回一个在指定设备上指定类型的0张量
    if num_steps <= 1:
        return torch.tensor(0, device=device, dtype=dtype)
    # 根据 align_corners 参数决定 a 的值，如果 align_corners 为 False，则计算 ((num_steps - 1) / num_steps)，否则为 1
    a = ((num_steps - 1) / num_steps) if not align_corners else 1
    # 使用 torch.linspace 生成一个在 [-a, a] 范围内的均匀间隔的张量
    return torch.linspace(-a, a, steps=num_steps, device=device, dtype=dtype)
# 生成一个4维空间中的基础网格，用于仿射变换
def _make_base_grid_4d(theta: Tensor, h: int, w: int, align_corners: bool):
    dtype = theta.dtype  # 获取 theta 张量的数据类型
    device = theta.device  # 获取 theta 张量的设备信息

    # 创建一个水平方向从-1到1的均匀间隔网格，形状为(1, w, 1)
    grid_x = _linspace_from_neg_one(w, align_corners, dtype, device).view(1, w, 1)
    # 创建一个垂直方向从-1到1的均匀间隔网格，形状为(h, 1, 1)
    grid_y = _linspace_from_neg_one(h, align_corners, dtype, device).view(h, 1, 1)
    # 创建一个单元网格，值为1，形状为(1, 1, 1)
    grid_one = torch.ones((1, 1, 1), dtype=dtype, device=device)

    # 临时的填充操作，替代了使用 torch.stack 的方式，待 #104480 合并后应使用 torch.stack
    grid_x = torch.nn.functional.pad(grid_x, pad=(0, 2), mode="constant", value=0)
    grid_y = torch.nn.functional.pad(grid_y, pad=(1, 1), mode="constant", value=0)
    grid_one = torch.nn.functional.pad(grid_one, pad=(2, 0), mode="constant", value=0)

    return grid_x + grid_y + grid_one  # 返回组合后的网格


# 生成一个5维空间中的基础网格，用于仿射变换
def _make_base_grid_5d(theta: Tensor, d: int, h: int, w: int, align_corners: bool):
    dtype = theta.dtype  # 获取 theta 张量的数据类型
    device = theta.device  # 获取 theta 张量的设备信息

    # 创建一个在水平方向从-1到1均匀间隔的网格，形状为(1, 1, w, 1)
    grid_x = _linspace_from_neg_one(w, align_corners, dtype, device).view(1, 1, w, 1)
    # 创建一个在垂直方向从-1到1均匀间隔的网格，形状为(1, h, 1, 1)
    grid_y = _linspace_from_neg_one(h, align_corners, dtype, device).view(1, h, 1, 1)
    # 创建一个在深度方向从-1到1均匀间隔的网格，形状为(d, 1, 1, 1)
    grid_z = _linspace_from_neg_one(d, align_corners, dtype, device).view(d, 1, 1, 1)
    # 创建一个单位网格，值为1，形状为(1, 1, 1, 1)
    grid_one = torch.ones((1, 1, 1, 1), dtype=dtype, device=device)

    # 临时的填充操作，替代了使用 torch.stack 的方式，待 #104480 合并后应使用 torch.stack
    grid_x = torch.nn.functional.pad(grid_x, pad=(0, 3), mode="constant", value=0)
    grid_y = torch.nn.functional.pad(grid_y, pad=(1, 2), mode="constant", value=0)
    grid_z = torch.nn.functional.pad(grid_z, pad=(2, 1), mode="constant", value=0)
    grid_one = torch.nn.functional.pad(grid_one, pad=(3, 0), mode="constant", value=0)

    return grid_x + grid_y + grid_z + grid_one  # 返回组合后的网格


# 生成4维仿射变换的网格
def _affine_grid_generator_4d(theta: Tensor, size: List[int], align_corners: bool):
    n, _, h, w = size
    base_grid = _make_base_grid_4d(theta, h, w, align_corners=align_corners)
    # base_grid 的形状为 (h, w, 3)，theta 的形状为 (n, 2, 3)
    # 手动进行矩阵乘法运算，比使用 mm() 更快
    # (h * w, 3, 1) * (n, 1, 3, 2) -> (n, h * w, 2)
    grid = (base_grid.view(-1, 3, 1) * theta.mT.unsqueeze(1)).sum(-2)
    return grid.view(n, h, w, 2)  # 返回形状为 (n, h, w, 2) 的网格


# 生成5维仿射变换的网格
def _affine_grid_generator_5d(theta: Tensor, size: List[int], align_corners: bool):
    n, _, d, h, w = size
    base_grid = _make_base_grid_5d(theta, d, h, w, align_corners=align_corners)
    # base_grid 的形状为 (d, h, w, 4)，theta 的形状为 (n, 3, 4)
    # 手动进行矩阵乘法运算，比使用 mm() 更快
    # (d * h * w, 4, 1) * (n, 1, 4, 3) -> (n, d * h * w, 3)
    grid = (base_grid.view(-1, 4, 1) * theta.mT.unsqueeze(1)).sum(-2)
    return grid.view(n, d, h, w, 3)  # 返回形状为 (n, d, h, w, 3) 的网格


# 注册一个仿射网格生成器函数，用于生成仿射变换的网格
@register_decomposition(aten.affine_grid_generator)
@out_wrapper()
@pw_cast_for_opmath
def affine_grid_generator(theta: Tensor, size: List[int], align_corners: bool):
    # 检查输入的张量维度是否为4或5
    torch._check(
        len(size) in (4, 5),
        lambda: "affine_grid_generator needs 4d (spatial) or 5d (volumetric) inputs.",
    )
    # 如果张量维度为4，则调用4维仿射网格生成函数
    if len(size) == 4:
        return _affine_grid_generator_4d(theta, size, align_corners=align_corners)
    # 如果张量维度为5，则调用5维仿射网格生成函数
    else:
        return _affine_grid_generator_5d(theta, size, align_corners=align_corners)
def _grid_sampler_2d(
    a: Tensor,
    grid: Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
    _expand_grid: bool = True,
) -> Tensor:
    # 这个方法是 grid_sampler_2d 实现的一个副本，并引入了额外的参数 _expand_grid，
    # 可以选择性地扩展输入的网格，以提高性能。
    # 在本地实验中发现，如果我们将网格从 (N, H, W, 2) 扩展到 (N, C, H, W, 2)，
    # 对于双三次插值模式，编译后的 CUDA 代码加速约为 5 倍，
    # 对于双线性插值模式，CPU 加速约为 2 倍。
    # 但是在双线性插值模式下，channels first 情况下，这会导致大约 0.8 倍的减速。
    # 因此，我们应用这个技巧来避免为此情况扩展网格。

    torch._check(
        interpolation_mode in (0, 1, 2),
        lambda: f"Invalid interpolation mode {interpolation_mode}",
    )
    torch._check(
        padding_mode in (0, 1, 2), lambda: f"Invalid padding mode {padding_mode}"
    )

    def unnormalize(coords: Tensor, size: int) -> Tensor:
        # 将坐标从 [-1, 1] 反向缩放到:
        #   [0, size - 1]，如果 align_corners 是 True
        #   [-.5, size -.5]，如果 align_corners 是 False
        mul = (size * 0.5 - 0.5) if align_corners else (size * 0.5)
        ofs = size * 0.5 - 0.5
        return coords * mul + ofs

    # 反射坐标，直到它们落在 low 和 high 之间（包括边界）。
    # 边界以两倍的值传递，以便可以表示半整数值为整数。
    def reflect_coordinates(coords: Tensor, twice_low: int, twice_high: int) -> Tensor:
        if twice_low == twice_high:
            return torch.zeros_like(coords)
        coords_min = twice_low / 2
        coords_span = (twice_high - twice_low) / 2
        coords2 = (coords - coords_min).abs()
        extra = torch.fmod(coords2, coords_span)
        flips = (coords2 / coords_span).floor().to(dtype=torch.int8)
        return torch.where(
            flips & 1 == 0, extra + coords_min, coords_span + coords_min - extra
        )

    def compute_coordinates(coords: Tensor, size: int) -> Tensor:
        if padding_mode == 0:  # Zero
            return coords
        elif padding_mode == 1:  # Borders
            return torch.clamp(coords, 0, size - 1)
        else:  # padding_mode == 2, Reflection
            if align_corners:
                coords_reflected = reflect_coordinates(coords, 0, 2 * (size - 1))
            else:
                coords_reflected = reflect_coordinates(coords, -1, 2 * size - 1)
            return torch.clamp(coords_reflected, 0, size - 1)

    def compute_source_index(coords: Tensor, size: int) -> Tensor:
        coords_un = unnormalize(coords, size)
        return compute_coordinates(coords_un, size)

    N, C, iH, iW = a.shape
    _, oH, oW, two = grid.shape
    assert two == 2
    if _expand_grid:
        # 如果 _expand_grid 为真，则进行网格扩展到 [N, C, oH, oW, 2]
        # 这样可以生成一个单独的 Triton CUDA 内核而不是两个内核。
        # 两个内核是因为源索引，权重的形状为 (N, 1, oH, oW)，xnumel=N*oH*oW
        # 输出的形状为 (N, C, oH, oW)，xnumel=N*C*oH*oW
        # 将网格扩展到 (N, C, oH, oW, 2) 统一了 xnumel 为 N*C*oH*oW
        grid = grid.view(N, 1, oH, oW, 2).expand(N, C, oH, oW, 2)

    def in_bounds_cond(xs: Tensor, ys: Tensor) -> Tensor:
        # 检查坐标是否在有效范围内的条件
        return torch.logical_and(
            0 <= xs, torch.logical_and(xs < iW, torch.logical_and(0 <= ys, ys < iH))
        )

    N_idx = torch.arange(N, device=a.device).view(N, 1, 1, 1)
    C_idx = torch.arange(C, device=a.device).view(1, C, 1, 1)

    def clip(xs: Tensor, ys: Tensor, ws: Tensor) -> TensorSequenceType:
        # 对坐标进行裁剪到有效范围内，同时调整张量形状以便与 N_idx、C_idx 进行广播索引
        c = C if _expand_grid else 1
        return tuple(
            torch.where(cond, t, 0).view(N, c, oH, oW)
            for t in (xs.to(dtype=torch.int64), ys.to(dtype=torch.int64), ws)
        )

    def get_summand(ix: Tensor, iy: Tensor, w) -> Tensor:
        # 执行裁剪，索引输入张量，并乘以权重
        idx_x, idx_y, w_ = clip(ix, iy, w)
        return a[N_idx, C_idx, idx_y, idx_x] * w_

    x = grid[..., 0]
    y = grid[..., 1]

    if interpolation_mode == 0:  # Bilinear
        ix = compute_source_index(x, iW)
        iy = compute_source_index(y, iH)

        ix_nw, iy_nw = ix.floor(), iy.floor()
        ix_ne, iy_ne = ix_nw + 1, iy_nw
        ix_sw, iy_sw = ix_nw, iy_nw + 1
        ix_se, iy_se = ix_ne, iy_sw

        w_nw = (ix_se - ix) * (iy_se - iy)
        w_ne = (ix - ix_sw) * (iy_sw - iy)
        w_sw = (ix_ne - ix) * (iy - iy_ne)
        w_se = (ix - ix_nw) * (iy - iy_nw)

        # 返回四个角的插值结果的总和
        return _sum_tensors(
            get_summand(ix, iy, w)
            for (ix, iy, w) in (
                (ix_nw, iy_nw, w_nw),
                (ix_ne, iy_ne, w_ne),
                (ix_sw, iy_sw, w_sw),
                (ix_se, iy_se, w_se),
            )
        )
    elif interpolation_mode == 1:  # Nearest
        ix = compute_source_index(x, iW)
        iy = compute_source_index(y, iH)

        ix_nearest = ix.round()
        iy_nearest = iy.round()

        # 返回最近邻插值的结果
        return get_summand(ix_nearest, iy_nearest, 1)
    else:  # interpolation_mode == 2, Bicubic
        # 将 x 和 y 进行反归一化操作，获取原始图像中的坐标
        ix = unnormalize(x, iW)
        iy = unnormalize(y, iH)

        # 计算最近的整数左上角像素位置
        ix_nw = ix.floor()
        iy_nw = iy.floor()

        # 计算插值的偏移量
        tx = ix - ix_nw
        ty = iy - iy_nw

        # 如果不扩展网格，则将 tx 和 ty 转换为包含维度的张量
        if not _expand_grid:
            tx = tx.unsqueeze(1)
            ty = ty.unsqueeze(1)

        # 定义一个函数，用于获取在边界内的值
        def get_value_bounded(ix: Tensor, iy: Tensor) -> Tensor:
            x = compute_coordinates(ix, iW)
            y = compute_coordinates(iy, iH)
            return get_summand(x, y, 1)

        # 定义一个函数，用于获取插值系数
        def get_coeff(ofs: int) -> Tensor:
            # 计算当前偏移量下的 iy 值
            iy_ofs = iy_nw + (ofs - 1)
            # 获取四个在 x 方向上的值，并进行边界内的获取
            cs = (
                get_value_bounded(ix_nw - 1, iy_ofs),
                get_value_bounded(ix_nw, iy_ofs),
                get_value_bounded(ix_nw + 1, iy_ofs),
                get_value_bounded(ix_nw + 2, iy_ofs),
            )
            # 使用三次样条插值方法计算插值系数
            return _upsample_cubic_interp1d(cs, tx)

        # 获取所有偏移量下的插值系数，并在 y 方向上进行插值
        coeffs = tuple(get_coeff(ofs) for ofs in range(4))
        return _upsample_cubic_interp1d(coeffs, ty)
# 注册函数 `grid_sampler_2d` 为 `aten.grid_sampler_2d` 的分解函数，并应用装饰器 `out_wrapper` 和 `pw_cast_for_opmath`
@aten.grid_sampler_2d.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@out_wrapper()
@pw_cast_for_opmath
def grid_sampler_2d(
    a: Tensor,
    grid: Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
) -> Tensor:
    # 调用底层函数 `_grid_sampler_2d`，并返回结果
    return _grid_sampler_2d(
        a,
        grid=grid,
        interpolation_mode=interpolation_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


# 注册函数 `mv` 为 `aten.mv` 的分解函数，并应用装饰器 `out_wrapper` 和 `pw_cast_for_opmath`
@aten.mv.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@out_wrapper()
@pw_cast_for_opmath
def mv(self, vec):
    # 检查矩阵和向量的维度是否符合要求，否则抛出相应的错误消息
    torch._check(
        self.dim() == 2 and vec.dim() == 1,
        lambda: f"matrix @ vector expected, got {self.dim()}, {vec.dim()}",
    )
    torch._check(
        self.size(1) == vec.size(0),
        lambda: f"size mismatch, got input ({self.size(0)}x{self.size(1)}), vec ({vec.size(0)})",
    )
    # 计算矩阵向量乘积的和并返回
    return (self * vec).sum(dim=1)


# 注册函数 `binary_cross_entropy_with_logits` 为 `aten.binary_cross_entropy_with_logits` 的分解函数，并应用装饰器 `out_wrapper`
@aten.binary_cross_entropy_with_logits.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@out_wrapper()
def binary_cross_entropy_with_logits(
    self, target, weight=None, pos_weight=None, reduction=Reduction.MEAN.value
):
    if pos_weight is not None:
        # 根据正权重计算加权损失
        log_weight = (pos_weight - 1) * target + 1
        loss = (1 - target) * self - (log_weight * F.logsigmoid(self))
    else:
        # 普通的二分类交叉熵损失计算
        loss = (1 - target) * self - F.logsigmoid(self)

    if weight is not None:
        # 根据权重调整损失值
        loss = loss * weight

    # 应用损失缩减方法并返回最终的损失值
    return apply_loss_reduction(loss, reduction)


# 函数 `should_fold` 判断是否应该对输入的两个张量进行合并计算
def should_fold(tensor1: torch.Tensor, tensor2: torch.Tensor, is_out: bool) -> bool:
    # 如果张量1的维度大于等于3且张量2的维度小于等于2，则返回True
    t1, t2 = (tensor1, tensor2) if tensor1.ndim >= tensor2.ndim else (tensor2, tensor1)
    if not (t1.ndim >= 3 and t2.ndim <= 2):
        return False
    # 如果张量2需要梯度并且不是输出结果，则返回True
    if t2.requires_grad and not is_out:
        return True
    # 如果张量1的维度为2，则返回False
    if tensor1.ndim == 2:
        return False
    # 如果张量1的元素个数为0，返回True
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious
    if guard_size_oblivious(t1.numel() == 0):
        return True

    # 检查张量1的形状和步长，并根据条件判断是否应该进行合并计算
    t1_shape = t1.shape
    t1_stride = t1.stride()
    return all(
        st1 == st2 * s2
        for (st1, st2, s2) in zip(t1_stride[:-2], t1_stride[1:-1], t1_shape[1:-1])
    )


# 注册函数 `matmul` 为矩阵乘法的分解函数，并应用相应的装饰器 `out_wrapper` 和 `pass_is_out=True`
@aten.matmul.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.matmul.out.py_impl(DispatchKey.CompositeImplicitAutograd)
@out_wrapper(pass_is_out=True)
def matmul(tensor1, tensor2, *, is_out=False):
    # 检查张量1和张量2的维度，确保它们不是标量
    dim_tensor1 = tensor1.dim()
    dim_tensor2 = tensor2.dim()
    assert dim_tensor1 != 0 and dim_tensor2 != 0
    # 根据张量的维度情况选择相应的矩阵乘法操作并返回结果
    if dim_tensor1 == 1 and dim_tensor2 == 1:
        return torch.dot(tensor1, tensor2)
    elif dim_tensor1 == 2 and dim_tensor2 == 1:
        return torch.mv(tensor1, tensor2)
    elif dim_tensor1 == 1 and dim_tensor2 == 2:
        return torch.squeeze(torch.mm(torch.unsqueeze(tensor1, 0), tensor2), 0)
    elif dim_tensor1 == 2 and dim_tensor2 == 2:
        return torch.mm(tensor1, tensor2)
    # 如果满足条件：dim_tensor1 >=3 && (dim_tensor2 == 1 || dim_tensor2 == 2) ||
    #             dim_tensor2 >=3 && (dim_tensor1 == 1 || dim_tensor1 == 2)
    # 并且满足某些关于步长的条件

    # 优化：通过将较大张量的批次折叠到其主要矩阵维度，使用 mm 而不是 bmm
    transpose = dim_tensor2 > dim_tensor1
    t1 = tensor2.mT if transpose else tensor1
    t2 = (
        tensor2 if not transpose else (tensor1.t() if dim_tensor1 == 2 else tensor1)
    )

    # 不变性：t1.dim() >= 3 && (t2.dim() == 1 || t2.dim() == 2)
    #        并且 t1 和 t2 是矩阵乘法兼容的

    # 为什么不使用 t1.view(-1, sizes_1[-1])？
    # 如果最后一个维度是 0，则 view(-1, 0) 无法工作，因为 -1 变得模糊不清。
    # 例如，在 [3, 5, 0] @ [0, 0] 的情况下可能会发生这种情况。

    sizes_1 = t1.shape
    output_shape = list(sizes_1[:-1])
    folded_dim1 = reduce(operator.mul, output_shape)

    # 如果我们正在与一个矩阵相乘，则重新调整 output_shape
    t2_is_matrix = t2.dim() == 2
    if t2_is_matrix:
        output_shape.append(t2.shape[1])

    # 这几乎总是一个视图（view）操作。
    # 如果 t2.requires_grad()，则可能不是视图。请参阅 aten/ 中的 should_fold 函数的解释。
    t1_folded = t1.reshape(folded_dim1, sizes_1[-1])
    if t2_is_matrix:
        # 如果我们执行 2D @ 3D，并且第一个张量需要梯度，这将会复制。
        # 请参阅 native/LinearAlgebra.cpp 中的 should_fold 函数的原因。
        output = t1_folded.mm(t2).view(output_shape)
        return output.mT.contiguous() if transpose else output
    else:
        return t1_folded.mv(t2).view(output_shape)
    elif dim_tensor1 >= 1 and dim_tensor2 >= 1:
        # 如果 tensor1 和 tensor2 的维度至少为1
        # 我们正在将 b1 x n x m1 乘以 x2 x m2 x p（其中 b1 可以是一个列表）；
        # 我们单独跟踪 m1 和 m2，即使它们必须匹配以便能更好地显示错误消息

        # 获取 tensor1 的 n 和 m1 的尺寸
        n = tensor1.size(-2) if dim_tensor1 > 1 else 1
        m1 = tensor1.size(-1)
        # 获取 tensor1 的批处理部分的形状
        batch_tensor1 = tensor1.shape[:-2]
        # 获取 tensor2 的 m2 和 p 的尺寸
        m2 = tensor2.size(-2) if dim_tensor2 > 1 else tensor2.size(-1)
        p = tensor2.size(-1) if dim_tensor2 > 1 else 1

        # 初始化 batch_tensor2 为一个空列表
        batch_tensor2: List[int] = []
        # TODO: 处理切片的情况
        for i in range(dim_tensor2 - 2):
            batch_tensor2.append(tensor2.size(i))

        # 对梯度进行相同的优化，如 should_fold 中所示
        # 如果我们要广播，则强制进入 should_fold 分支
        if (
            dim_tensor1 == 3
            and dim_tensor2 == 3
            and batch_tensor1[0] != batch_tensor2[0]
        ):
            if batch_tensor1[0] == 1 and tensor1.requires_grad:
                return matmul(tensor1.squeeze(0), tensor2)
            if batch_tensor2[0] == 1 and tensor2.requires_grad:
                return matmul(tensor1, tensor2.squeeze(0))

        # 扩展批处理部分（即去除矩阵维度并扩展其余部分）
        expand_batch_portion = list(
            torch.broadcast_shapes(batch_tensor1, batch_tensor2)
        )

        # 计算 tensor1 扩展后的尺寸
        tensor1_expand_size = expand_batch_portion + [n, m1]

        # 计算扩展后的批处理乘积
        expand_batch_product = prod(expand_batch_portion)

        # HACK: 我们需要带有 symint 支持的 reshape
        tensor1_expanded = tensor1.expand(tensor1_expand_size).reshape(
            expand_batch_product, n, m1
        )

        # 判断 tensor2 是否是一个向量
        vector_rhs = dim_tensor2 == 1
        if vector_rhs:
            # 如果是向量，则计算 tensor2 扩展后的尺寸，并扩展并重塑 tensor2_expanded
            tensor2_expand_size = expand_batch_portion + [m2]
            tensor2_expanded = (
                tensor2.expand(tensor2_expand_size)
                .reshape(expand_batch_product, m2)
                .unsqueeze(2)
            )
        else:
            # 如果不是向量，则计算 tensor2 扩展后的尺寸，并扩展并重塑 tensor2_expanded
            tensor2_expand_size = expand_batch_portion + [m2, p]
            tensor2_expanded = tensor2.expand(tensor2_expand_size).reshape(
                expand_batch_product, m2, p
            )

        # 计算输出的形状
        output_shape = expand_batch_portion
        if dim_tensor1 > 1:
            output_shape.append(n)
        if dim_tensor2 > 1:
            output_shape.append(p)

        # 如果 tensor2 是向量，则返回经过 bmm 和 squeeze 处理后的结果
        if vector_rhs:
            return tensor1_expanded.bmm(tensor2_expanded).squeeze(-1).view(output_shape)
        else:
            # 如果 tensor2 不是向量，则返回经过 bmm 处理后的结果
            return tensor1_expanded.bmm(tensor2_expanded).view(output_shape)
    else:
        # 如果 tensor1 和 tensor2 的维度不足1，则抛出错误
        torch._check(False, lambda: "both arguments to matmul need to be at least 1D")
@register_decomposition([aten.upsample_bicubic2d.default, aten.upsample_bicubic2d.out])
@aten.upsample_bicubic2d.default.py_impl(DispatchKey.Autograd)
@out_wrapper()
@pw_cast_for_opmath
def upsample_bicubic2d_default(
    input: Tensor,
    output_size: Tuple[int, int],
    align_corners: bool,
    scale_h: Optional[float] = None,
    scale_w: Optional[float] = None,
) -> Tensor:
    # 获取原始图像的尺寸
    _, _, in_h, in_w = input.shape

    # 计算水平和垂直缩放因子
    h_scale_factor = _compute_scale(in_h, output_size[0], align_corners, scale_h)
    w_scale_factor = _compute_scale(in_w, output_size[1], align_corners, scale_w)

    # 确定元素类型
    _, dtype = utils.elementwise_dtypes(
        input, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )

    # 创建整数序列，避免额外的内核创建以提高性能
    i = torch.arange(output_size[0], device=input.device).to(dtype=dtype)
    j = torch.arange(output_size[1], device=input.device).to(dtype=dtype)

    # 计算源索引的浮点数值
    x_float = _compute_source_index(w_scale_factor, j, align_corners)
    y_float = _compute_source_index(h_scale_factor, i, align_corners)
    y_float = y_float.unsqueeze(-1)

    # 向下取整得到最近的整数索引
    x = x_float.floor()
    y = y_float.floor()

    # 对于插值，需要对xscale和yscale进行截断和限制
    # 参见 UpSample.h 中的 guard_index_and_lambda
    yscale = (y_float - y).clamp(0.0, 1.0)
    xscale = (x_float - x).clamp(0.0, 1.0)
    x = x.to(torch.int64)
    y = y.to(torch.int64)

    # 定义索引偏移
    iys_ofs = (y - 1, y, y + 1, y + 2)
    ixs_ofs = (x - 1, x, x + 1, x + 2)

    # 获取三次样条插值的系数
    weights_x = _upsample_get_cubic_coefficients(xscale)
    weights_y = _upsample_get_cubic_coefficients(yscale)

    # 如果输入是 uint8 类型，计算权重的精度
    weights_precision_x, weights_precision_y = None, None
    if input.dtype == torch.uint8:
        weights_precision_x = _compute_weight_precision(weights_x)
        weights_precision_y = _compute_weight_precision(weights_y)

        # 将权重缩放到整数表示，并应用四舍五入
        weights_x = [
            (w * (1 << weights_precision_x) + torch.sign(w) * 0.5).to(torch.int16)
            for w in weights_x
        ]
        weights_y = [
            (w * (1 << weights_precision_y) + torch.sign(w) * 0.5).to(torch.int16)
            for w in weights_y
        ]

    # 定义函数来加载边界限制后的值
    def load_bounded(ys, xs):
        y_idx = torch.clamp(ys, 0, in_h - 1)
        x_idx = torch.clamp(xs, 0, in_w - 1)
        v = aten._unsafe_index(input, [None, None, y_idx, x_idx])
        return v

    # 定义函数来获取 x 方向的插值
    def get_x_interp(y):
        src_x = tuple(load_bounded(y, x_ofs) for x_ofs in ixs_ofs)
        if input.dtype == torch.uint8:
            assert weights_precision_x is not None
            return _sum_tensors_uint8(src_x, weights_x, weights_precision_x)
        return _sum_tensors(c1 * c2 for (c1, c2) in zip(src_x, weights_x))

    # 获取 y 方向的插值
    src_y = tuple(get_x_interp(y_ofs) for y_ofs in iys_ofs)
    # 如果输入的数据类型是 torch.uint8
    if input.dtype == torch.uint8:
        # 确保 weights_precision_y 不为 None
        assert weights_precision_y is not None
        # 调用 _sum_tensors_uint8 函数对输入数据进行求和操作
        result = _sum_tensors_uint8(src_y, weights_y, weights_precision_y)
    else:
        # 否则，调用 _sum_tensors 函数对 src_y 和 weights_y 中的元素逐个相乘后求和
        result = _sum_tensors(c1 * c2 for (c1, c2) in zip(src_y, weights_y))

    # 将输出转换为正确的内存格式，如果有必要的话
    memory_format = utils.suggest_memory_format(input)
    result = result.contiguous(memory_format=memory_format)
    # 返回结果
    return result
# 注册函数的装饰器，将函数与特定的ATen操作绑定
@register_decomposition(aten.upsample_bicubic2d.vec)
# 为函数指定py_impl属性，关联CompositeImplicitAutograd调度键
@aten.upsample_bicubic2d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
# 为函数指定py_impl属性，关联Autograd调度键
@aten.upsample_bicubic2d.vec.py_impl(DispatchKey.Autograd)
# 将函数包装为输出包装器
@out_wrapper()
# 为操作数学操作进行类型转换
@pw_cast_for_opmath
# 定义函数，实现双线性插值上采样
def upsample_bicubic2d_vec(
    a: Tensor,
    output_size: Optional[Tuple[int, int]],
    align_corners: bool,
    scale_factors: Optional[Tuple[float, float]] = None,
) -> Tensor:
    # 检查是否指定了正确的参数组合
    torch._check(
        bool(output_size) + bool(scale_factors) == 1,
        lambda: "Must specify exactly one of output_size and scale_factors.",
    )
    # 如果未指定output_size，则根据scale_factors计算output_size
    if output_size is None:
        assert scale_factors is not None
        output_size = cast(
            Tuple[int, int],
            tuple(
                sym_int(sym_float(w) * scale)
                for w, scale in zip(a.shape[2:], scale_factors)
            ),
        )
    # 获取缩放因子
    scale_h, scale_w = scale_factors if scale_factors else (None, None)
    # 调用默认的双线性插值上采样函数
    return upsample_bicubic2d_default(a, output_size, align_corners, scale_h, scale_w)


# 注册函数的装饰器，与reflection_pad1d函数绑定
@register_decomposition(aten.reflection_pad1d)
# 注册函数的装饰器，与reflection_pad2d函数绑定
@register_decomposition(aten.reflection_pad2d)
# 注册函数的装饰器，与reflection_pad3d函数绑定
@register_decomposition(aten.reflection_pad3d)
# 为操作数学操作进行类型转换
@pw_cast_for_opmath
# 将函数包装为输出包装器
@out_wrapper()
# 实现反射填充操作
def _reflection_pad(a: Tensor, padding: Tuple[int, ...]) -> Tensor:
    # 定义索引函数，用于计算反射填充时的索引
    def idx(left, middle, right):
        dim_idx = torch.arange(-left, middle + right, device=a.device)
        return middle - 1 - (middle - 1 - dim_idx.abs()).abs()

    # 调用通用的反射或复制填充函数
    return _reflection_or_replication_pad(
        a,
        padding,
        idx,
    )


# 注册函数的装饰器，与replication_pad1d函数绑定
@register_decomposition(aten.replication_pad1d)
# 注册函数的装饰器，与replication_pad2d函数绑定
@register_decomposition(aten.replication_pad2d)
# 注册函数的装饰器，与replication_pad3d函数绑定
@register_decomposition(aten.replication_pad3d)
# 为操作数学操作进行类型转换
@pw_cast_for_opmath
# 将函数包装为输出包装器
@out_wrapper()
# 实现复制填充操作
def _replication_pad(a: Tensor, padding: Tuple[int, ...]) -> Tensor:
    # 定义索引函数，用于计算复制填充时的索引
    def idx(left, middle, right):
        dim_idx = torch.arange(-left, middle + right, device=a.device)
        return torch.clamp(dim_idx, 0, middle - 1)

    # 调用通用的反射或复制填充函数
    return _reflection_or_replication_pad(
        a,
        padding,
        idx,
    )


# 实现通用的反射或复制填充操作
def _reflection_or_replication_pad(
    a: Tensor,
    padding: Tuple[int, ...],
    idx_fn: Callable[[int, int, int], Tensor],
) -> Tensor:
    # 确定填充的维度
    dim = len(padding) // 2
    # 检查输入张量的维度是否正确
    torch._check(
        a.dim() in (dim + 1, dim + 2),
        lambda: f"reflection_pad{dim}d requires {dim + 1}D or {dim + 2}D input",
    )
    # 获取输入张量的形状
    inp_shape = a.shape[-dim:]
    # 获取非批处理维度
    nc_dim = a.dim() - dim

    # 计算填充的左侧和右侧边界
    padding_left = [padding[2 * (dim - 1 - i)] for i in range(dim)]
    padding_right = [padding[2 * (dim - 1 - i) + 1] for i in range(dim)]

    result = a
    # 遍历每个维度，进行填充操作
    for i in range(dim):
        idx: List[Any] = [None] * result.dim()
        idx[i + nc_dim] = idx_fn(padding_left[i], inp_shape[i], padding_right[i])
        result = aten._unsafe_index(result, idx)

    # 将输出转换为正确的内存格式（如果需要）
    memory_format = utils.suggest_memory_format(result)
    result = result.contiguous(memory_format=memory_format)
    return result


# 注册函数的装饰器，与aminmax函数绑定
@register_decomposition(aten.aminmax)
# 将函数包装为输出包装器，并指定输出的命名为min和max
@out_wrapper("min", "max")
# 定义一个方法 aminmax，计算张量在指定维度上的最小值和最大值
def aminmax(self, *, dim=None, keepdim=False):
    # 使用 torch.amin 函数计算张量在指定维度上的最小值
    amin = torch.amin(self, dim=dim, keepdim=keepdim)
    # 使用 torch.amax 函数计算张量在指定维度上的最大值
    amax = torch.amax(self, dim=dim, keepdim=keepdim)
    # 返回最小值和最大值的元组
    return amin, amax


# 注册函数 nansum 到 aten.nansum，使用 torch.sum 计算张量中非 NaN 元素的和
@register_decomposition(aten.nansum)
@out_wrapper()
def nansum(self, dim=None, keepdim=False, *, dtype=None):
    # 使用 torch.isnan 找出张量中的 NaN 元素，并用 0 替换，然后计算和
    return aten.sum(torch.where(torch.isnan(self), 0, self), dim, keepdim, dtype=dtype)


# 注册函数 arange_default 到 aten.arange.default 和 aten.arange.out，生成一个从 0 开始的整数序列
@register_decomposition([aten.arange.default, aten.arange.out])
@out_wrapper()
def arange_default(
    end: NumberType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
):
    # 调用 aten.arange.start_step 生成一个从 0 开始到 end-1 的整数序列
    return aten.arange.start_step(
        0, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


# 注册函数 arange_start 到 aten.arange.start，生成一个指定起始值的整数序列
@register_decomposition([aten.arange.start])
def arange_start(
    start: NumberType,
    end: NumberType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
):
    # 调用 aten.arange.start_step 生成一个从 start 开始到 end-1 的整数序列
    return aten.arange.start_step(
        start, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


# 注册函数 out_dtype_decomp 到 out_dtype，通过 out_dtype_dense 生成特定的输出数据类型
@register_decomposition(out_dtype)
def out_dtype_decomp(*args, **kwargs):
    from torch._higher_order_ops.out_dtype import out_dtype_dense

    # 调用 out_dtype_dense 函数返回特定的输出数据类型
    return out_dtype_dense(*args, **kwargs)


# 注册函数 multi_margin_loss 到 aten.multi_margin_loss，计算多边距损失
@register_decomposition(aten.multi_margin_loss)
@aten.multi_margin_loss.default.py_impl(DispatchKey.Autograd)
@out_wrapper()
def multi_margin_loss(
    input: Tensor,
    target: Tensor,
    p: NumberType = 1,
    margin: NumberType = 1,
    weight: Optional[Tensor] = None,
    reduction: int = Reduction.MEAN.value,
) -> Tensor:
    # 将 input 张量至少扩展为二维
    input = torch.atleast_2d(input)
    # 将 target 张量至少扩展为一维，并检查维度和元素个数
    target = torch.atleast_1d(target)
    nframe = input.shape[0]
    dim = input.shape[1]
    # 检查 p 是否为 1 或 2
    torch._check(p == 1 or p == 2, lambda: "only p == 1 and p == 2 supported")
    # 检查 input 的维度是否为 2 并且 dim 不为 0
    torch._check(
        input.ndim == 2 and dim != 0,
        lambda: f"Expected non-empty vector or matrix with optional 0-dim batch size, but got: {input.shape}",
    )
    # 检查 target 的维度是否为 1 并且元素个数与 nframe 相同
    torch._check(
        target.ndim == 1 and target.numel() == nframe,
        lambda: f"inconsistent target size, expected {nframe} but got {target.shape}",
    )
    # 如果 weight 不为空，则至少将其扩展为一维，并检查维度和元素个数
    if weight is not None:
        weight = torch.atleast_1d(weight)
        torch._check(
            weight.ndim == 1 and weight.numel() == dim,  # type: ignore[union-attr]
            lambda: f"inconsistent weight size, expected {dim} but got {weight.shape}",  # type: ignore[union-attr]
        )
    # 将 target 扩展为二维，并使用 gather 函数在指定维度上进行索引操作
    target = target.unsqueeze(1)
    u = torch.gather(input, dim=1, index=target)
    # 计算多边距损失函数的中间结果 z
    z = margin - u + input
    z = z.clamp_min(0)
    # 根据 p 的值进行相应的平方操作
    z = z if p == 1 else z * z
    # 如果 weight 不为空，则根据 target 的索引位置进行加权操作
    if weight is not None:
        z = z * weight[target]
    # 创建一个与 dim 相同的设备上的整数序列
    idx = torch.arange(dim, device=input.device)
    # 在 z 张量中使用 where 函数，将 target 外的索引位置置为 0
    z = torch.where(idx != target, z, 0)
    # 如果 reduction 为 Reduction.MEAN.value，则计算 z 的均值并返回
    if reduction == Reduction.MEAN.value:
        return z.mean()
    # 如果 reduction 参数指定为 SUM 值时执行以下操作
    elif reduction == Reduction.SUM.value:
        # 计算张量 z 沿列方向的和并返回每列的平均值
        return z.sum() / z.shape[1]
    # 如果 reduction 参数不是 SUM 值时执行以下操作
    else:
        # 计算张量 z 沿行方向的均值并返回结果
        return z.mean(dim=1)
# 注册 multilabel_margin_loss_forward 函数的自动分解装饰器
# 将 multilabel_margin_loss_forward 函数的默认 Python 实现关联到 Autograd 的分发键上
# 包装输出结果为 ("output", "is_target")
@register_decomposition(aten.multilabel_margin_loss_forward)
@aten.multilabel_margin_loss_forward.default.py_impl(DispatchKey.Autograd)
@out_wrapper("output", "is_target")
def multilabel_margin_loss_forward(
    input: Tensor,
    target: Tensor,
    reduction: int,
) -> Tuple[Tensor, Tensor]:
    # 记录输入的原始形状
    orig_input_shape = input.shape
    # 记录目标的原始形状
    orig_target_shape = target.shape
    # 确保 input 至少是二维的张量
    input = torch.atleast_2d(input)
    # 确保 target 至少是二维的张量
    target = torch.atleast_2d(target)
    # 获取 input 的第二个维度的大小
    dim = input.shape[1]
    # 检查输入形状是否符合预期条件
    torch._check(
        len(orig_input_shape) <= 2 and dim != 0,
        lambda: f"Expected non-empty vector or matrix with optional 0-dim batch size, but got: {orig_input_shape}",
    )
    # 检查目标形状是否与输入形状一致
    torch._check(
        len(orig_target_shape) <= 2 and orig_target_shape == orig_input_shape,
        lambda: f"inconsistent target size: {orig_target_shape} for input of size: {orig_input_shape}",
    )
    
    # 忽略第一个 -1 之后的标签，检测是否不存在 -1
    idx = torch.arange(dim, device=target.device)
    is_end = target == -1
    end_idx = torch.amin(torch.where(is_end, idx, dim), dim=-1, keepdim=True)
    # 目标索引
    target_mask = idx < end_idx
    # 将目标掩码化，以便使用 gather 函数，因为 gather 函数不允许 -1
    tidx0 = torch.where(target_mask, target, 0)
    u = torch.gather(input, dim=-1, index=tidx0)
    # is_target
    tidx1 = torch.where(target_mask, target, -1)
    is_target = torch.any(idx == tidx1.unsqueeze(dim=-1), dim=1)
    
    # 计算损失
    z = 1.0 - u.T.unsqueeze(dim=-1) + input
    z = z.clamp_min(0)
    z = z / dim
    
    # 对损失进行掩码处理
    z = torch.where(is_target, 0, z)
    
    # 根据 reduction 参数进行损失的降维处理
    if reduction == Reduction.MEAN.value:
        z = z.sum(dim=(0, -1)).mean()
    elif reduction == Reduction.SUM.value:
        z = z.sum()
    else:
        z = z.sum(dim=(0, -1))
    
    # 将 is_target 转换为与 input 相同的数据类型，并恢复其原始目标形状
    is_target = is_target.to(input.dtype).reshape(orig_target_shape)
    
    # 返回损失值和 is_target 标志
    return z, is_target


# 注册 _scaled_dot_product_flash_attention_for_cpu 函数的自动分解装饰器
# scaled_dot_product_attention 函数在导出路径中调用 _scaled_dot_product_flash_attention_for_cpu
# 由于此分解规则应该被排除在引导器之外，确保 scaled_dot_product_attention 仍按照以前的方式分解，
# 即通过 _scaled_dot_product_attention_math 函数
@register_decomposition(aten._scaled_dot_product_flash_attention_for_cpu.default)
def scaled_dot_product_flash_attention_for_cpu(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    attn_mask: Optional[Tensor] = None,
    scale: Optional[float] = None,
# 定义一个函数，接受三个张量（query, key, value），返回一个元组，包含两个张量。
def forward(
    query: Tensor, key: Tensor, value: Tensor
) -> Tuple[Tensor, Tensor]:
    # 获取 query 张量的数据类型
    dtype = query.dtype
    # 检查 query 张量是否为浮点数类型（FP32, FP64, BF16, FP16）
    torch._check(
        torch.is_floating_point(query),
        lambda: f"query must be FP32, FP64, BF16, FP16 but got {query.dtype}",
    )
    # 检查 query, key, value 张量是否都是四维张量
    torch._check(
        query.dim() == 4 and key.dim() == 4 and value.dim() == 4,
        lambda: f"q, k, v must be a 4 dimensional tensor, got {query.dim()}, {key.dim()}, {value.dim()}",
    )
    # 检查 dropout_p 是否为 0.0
    torch._check(
        dropout_p == 0.0, lambda: f"dropout probability must be zero, got {dropout_p}"
    )
    # 检查 query, key, value 张量的最后一个维度是否相同，应该具有相同的头大小
    torch._check(
        query.shape[3] == value.shape[3] and key.shape[3] == value.shape[3],
        lambda: "q, k, v should have the same head size",
    )

    # 调用 _scaled_dot_product_attention_math.default 方法进行注意力计算
    output, attn = aten._scaled_dot_product_attention_math.default(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        dropout_mask=None,
        scale=scale,
    )

    # 对 output 进行维度转置和连续化，以满足导出时的需求
    output = output.transpose(1, 2).contiguous(memory_format=torch.contiguous_format)
    
    # 返回转置后的 output 和 attn
    return (output.transpose(1, 2), attn)
    # 定义一个函数 inplace_op，接受任意位置参数和关键字参数
    def inplace_op(*args, **kwargs):
        # 调用 outplace_op 函数处理参数，并将结果保存到 out 变量中
        out = outplace_op(*args, **kwargs)
        # 调用 args[0] 对象的 copy_ 方法，将 out 的内容复制到 args[0] 中
        return args[0].copy_(out)

    # 返回 inplace_op 函数作为结果
    return inplace_op
# 将 baddbmm 函数注册为特定操作的分解函数，并装饰处理输出
@register_decomposition([aten.baddbmm])
@out_wrapper()
@pw_cast_for_opmath
def baddbmm(self, batch1, batch2, beta=1, alpha=1):
    # 如果张量不是浮点数且不是复数类型，则将 beta 和 alpha 转换为整数
    if not self.is_floating_point() and not self.is_complex():
        beta = int(beta)
        alpha = int(alpha)
    # 计算 batch1 和 batch2 的批量矩阵乘积
    result = torch.bmm(batch1, batch2)
    # 如果 alpha 不是数值类型或者不等于 1，则对结果乘以 alpha
    if not isinstance(alpha, numbers.Number) or alpha != 1:
        result = result * alpha
    # 如果 beta 等于 0，则直接返回结果
    if beta == 0:
        return result
    # 如果 beta 不是数值类型或者不等于 1，则将 self 乘以 beta
    if not isinstance(beta, numbers.Number) or beta != 1:
        self = self * beta
    # 返回 self 加上结果
    return self + result


# 将 floor_divide 函数注册为 floor_divide 操作的分解函数，并装饰处理输出
@register_decomposition(aten.floor_divide)
@out_wrapper()
def floor_divide(self, other):
    # 使用 floor 模式进行除法运算
    return torch.div(self, other, rounding_mode="floor")


# 将 sym_numel 函数注册为 sym_numel 操作的分解函数
@register_decomposition(aten.sym_numel)
def sym_numel(t):
    # 计算张量 t 的元素数量
    return functools.reduce(operator.mul, t.shape, 1)


# 将 sum_default 函数注册为 sum 操作的默认分解函数或带输出的分解函数
@register_decomposition([aten.sum.default, aten.sum.out])
def sum_default(
    self: Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> Tensor:
    # 如果输出 out 为 None，则调用 aten.sum.dim_IntList 函数进行求和
    if out is None:
        return aten.sum.dim_IntList(self, [], dtype=dtype)
    else:
        # 否则，调用 aten.sum.IntList_out 函数进行求和，结果输出到 out
        return aten.sum.IntList_out(self, [], dtype=dtype, out=out)


# 将 squeeze_default 函数注册为 squeeze 操作的默认分解函数或指定维度的分解函数
@register_decomposition([aten.squeeze.default, aten.squeeze.dim])
def squeeze_default(self: Tensor, dim: Optional[int] = None):
    # 如果维度 dim 为 None，则对所有维度进行挤压操作
    if dim is None:
        return aten.squeeze.dims(self, list(range(self.dim())))
    else:
        # 否则，对指定维度 dim 进行挤压操作
        return aten.squeeze.dims(self, [dim])


# 将 _weight_norm_interface 函数注册为 _weight_norm_interface 操作的分解函数
@register_decomposition(torch.ops.aten._weight_norm_interface)
def _weight_norm_interface(v, g, dim=0):
    # 根据 v 的形状计算要保留的维度，排除指定的 dim 维度
    keep_dim = tuple(i for i in range(len(v.shape)) if i != dim)
    # 如果 g 的类型是 torch.bfloat16，则将 norm 的类型设为 torch.float，以保持与 CUDA 的行为一致
    norm_dtype = torch.float if g.dtype == torch.bfloat16 else None
    # 计算 v 的二范数，并保持维度，根据需要使用 norm_dtype 指定的数据类型
    norm = v.norm(2, keep_dim, keepdim=True, dtype=norm_dtype)
    # 返回归一化后的 v 和计算得到的 norm
    return v * (g / norm.to(g.dtype)), norm


# 将 isin 函数注册为 isin 操作的分解函数，并装饰处理输出
@register_decomposition(aten.isin)
@out_wrapper()
def isin(elements, test_elements, *, assume_unique=False, invert=False):
    # 处理当 elements 或 test_elements 是标量时（它们不能同时为标量）
    if not isinstance(elements, torch.Tensor):
        elements = torch.tensor(elements, device=test_elements.device)
    if not isinstance(test_elements, torch.Tensor):
        test_elements = torch.tensor(test_elements, device=elements.device)

    # 根据 test_elements 的元素数量和 elements 的元素数量计算是否使用排序方法
    if test_elements.numel() < 10.0 * pow(elements.numel(), 0.145):
        return isin_default(elements, test_elements, invert=invert)
    else:
        return isin_sorting(
            elements, test_elements, assume_unique=assume_unique, invert=invert
        )


# isin 操作的默认分解函数，处理 elements 和 test_elements 之间的比较
def isin_default(elements, test_elements, *, invert=False):
    # 如果 elements 的元素数量为 0，则返回与 elements 形状相同的空布尔张量
    if elements.numel() == 0:
        return torch.empty_like(elements, dtype=torch.bool)

    # 将 elements 视图扩展为与 test_elements 相同的形状，并进行比较
    x = elements.view(*elements.shape, *((1,) * test_elements.ndim))
    # 如果 invert 为 False，则比较是否相等；否则，比较是否不相等
    if not invert:
        cmp = x == test_elements
    else:
        cmp = x != test_elements
    # 创建一个元组，表示对数组维度的反向索引范围，从最后一维的倒数第一到倒数第 test_elements.ndim + 1 维
    dim = tuple(range(-1, -test_elements.ndim - 1, -1))
    # 返回 cmp 对象的任何维度上的最大值
    return cmp.any(dim=dim)
# 定义一个函数，用于在元素集合中进行排序并检查元素是否在测试集合中
def isin_sorting(elements, test_elements, *, assume_unique=False, invert=False):
    # 将元素集合展平
    elements_flat = elements.flatten()
    # 将测试元素集合展平
    test_elements_flat = test_elements.flatten()
    
    # 如果 assume_unique 为 True，则假设元素集合和测试集合中的元素是唯一的
    if assume_unique:
        # 将元素集合和测试元素集合合并
        all_elements = torch.cat([elements_flat, test_elements_flat])
        # 对合并后的所有元素进行排序，返回排序后的元素及其原始索引
        sorted_elements, sorted_order = torch.sort(all_elements, stable=True)
        
        # 找出重复元素的掩码
        duplicate_mask = sorted_elements[1:] == sorted_elements[:-1]
        # 在掩码的末尾添加一个 False，保持与原始大小相同
        duplicate_mask = torch.constant_pad_nd(duplicate_mask, [0, 1], False)
        
        # 如果 invert 为 True，则取反重复掩码
        if invert:
            duplicate_mask = duplicate_mask.logical_not()
        
        # 根据排序后的顺序复制重复掩码，以保持原始元素集合顺序
        mask = torch.empty_like(duplicate_mask)
        mask = mask.index_copy(0, sorted_order, duplicate_mask)
        
        # 返回与元素集合相同大小的掩码
        return mask[0 : elements.numel()]
    
    # 如果 assume_unique 为 False，则不能假设元素是唯一的
    else:
        # 对测试元素集合进行排序
        sorted_test_elements, _ = torch.sort(test_elements_flat)
        # 在排序后的测试元素集合中搜索元素集合中的元素的位置索引
        idx = torch.searchsorted(sorted_test_elements, elements_flat)
        # 将搜索结果与测试元素集合的大小比较，避免越界
        test_idx = torch.where(idx < sorted_test_elements.numel(), idx, 0)
        # 比较测试元素与元素集合中的元素是否相等
        cmp = sorted_test_elements[test_idx] == elements_flat
        # 如果 invert 为 True，则取反比较结果
        cmp = cmp.logical_not() if invert else cmp
        # 将比较结果重新形状为与元素集合相同的形状
        return cmp.reshape(elements.shape)


# 注册一个函数的装饰器，用于对给定索引的张量进行扁平化，并返回该索引处的元素
@register_decomposition(aten.take)
@out_wrapper()
def take(self, index):
    # 将张量扁平化
    flattened = self.reshape(-1)
    # 返回给定索引处的元素
    return flattened[index]


# 注册一个函数，用于调整张量的形状与另一个张量相同，并选择性地指定内存格式
@register_decomposition(aten.resize_as)
def resize_as(self, other, memory_format=None):
    # 如果未指定内存格式，则使用默认的张量连续格式
    if memory_format is None:
        memory_format = torch.contiguous_format
    # 如果内存格式为保持格式，则根据建议选择内存格式
    if memory_format == torch.preserve_format:
        memory_format = suggest_memory_format(other)
    # 调整张量的形状与另一个张量相同，并使用指定的内存格式
    return aten.resize(self, other.shape, memory_format=memory_format)


# 注册一系列原地操作的函数，如张量的原地加法、原地逻辑运算等
register_inplace(aten.addbmm_, aten.addbmm)
register_inplace(aten.addmm_, aten.addmm)
register_inplace(aten.addmv_, aten.addmv)
register_inplace(aten.baddbmm_, aten.baddbmm)
register_inplace(aten.fill_, aten.fill)
register_inplace(aten.gelu_, aten.gelu)
register_inplace(aten.hardswish_, aten.hardswish)
register_inplace(aten.hardtanh_, aten.hardtanh)
register_inplace(aten.hardsigmoid_, aten.hardsigmoid)
register_inplace(aten.__iand__, aten.__and__)
register_inplace(aten.__ilshift__, aten.__lshift__)
register_inplace(aten.index_put_, aten.index_put)
register_inplace(aten.index_reduce_, aten.index_reduce)
register_inplace(aten.__ior__, aten.__or__)
register_inplace(aten.__irshift__, aten.__rshift__)
register_inplace(aten.__ixor__, aten.__xor__)
register_inplace(aten.leaky_relu_, aten.leaky_relu)
register_inplace(aten.logit_, aten.logit)
register_inplace(aten.relu_, aten.relu)
register_inplace(aten.renorm_, aten.renorm)
register_inplace(aten.round_, aten.round)
register_inplace(aten.scatter_, aten.scatter)
register_inplace(aten.scatter_add_, aten.scatter_add)
register_inplace(aten.scatter_reduce_, aten.scatter_reduce)
register_inplace(aten.silu_, aten.silu)
```