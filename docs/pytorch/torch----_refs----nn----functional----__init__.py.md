# `.\pytorch\torch\_refs\nn\functional\__init__.py`

```
# mypy: allow-untyped-defs
# 导入数学库
import math
# 导入装饰器相关库
from functools import wraps
# 导入类型相关库
from typing import Callable, Optional, Union

# 导入PyTorch库
import torch
# 导入PyTorch的内部运算模块
import torch._prims as prims
# 导入PyTorch的常用工具模块
import torch._prims_common as utils
# 导入PyTorch的引用模块
import torch._refs as refs
# 导入PyTorch的分解模块注册函数
from torch._decomp import register_decomposition
# 从PyTorch的常用工具模块中导入特定类型和常量
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    NumberType,
    ShapeType,
    TensorLike,
    TensorLikeType,
)
# 从PyTorch的常用工具模块中导入包装器函数
from torch._prims_common.wrappers import (
    elementwise_type_promotion_wrapper,
    elementwise_unary_scalar_wrapper,
    out_wrapper,
)
# 从PyTorch的引用模块中导入特定函数
from torch._refs import _make_inplace

# 声明__all__列表，指定公开的函数名
__all__ = [
    "alpha_dropout",
    "celu",
    "celu_",
    "dropout",
    "elu",
    "elu_",
    "gelu",
    "glu",
    "group_norm",
    "hardshrink",
    "hardtanh",
    "hinge_embedding_loss",
    "huber_loss",
    "l1_loss",
    "layer_norm",
    "leaky_relu",
    "log_softmax",
    "margin_ranking_loss",
    "mish",
    "mish_",
    "mse_loss",
    "nll_loss",
    "pairwise_distance",
    "pdist",
    "poisson_nll_loss",
    "prelu",
    "relu",
    "relu6",
    "selu",
    "selu_",
    "smooth_l1_loss",
    "softmax",
    "softmin",
    "softplus",
    "softshrink",
    "tanhshrink",
    "threshold",
    "threshold_",
    "triplet_margin_loss",
]

# 定义别名Tensor表示PyTorch张量类型
Tensor = torch.Tensor
# 定义aten作为PyTorch的运算操作模块
aten = torch._ops.ops.aten
# 定义DispatchKey表示PyTorch的分发键类型，类型不确定时忽略属性定义错误
DispatchKey = torch._C.DispatchKey  # type: ignore[attr-defined]


def _dropout_helper(
    self: TensorLikeType,
    val: float,
) -> TensorLikeType:
    """
    Helper function for all dropout-type operators. During training,
    some of the elements of the input tensor are randomly masked.

    Returns the masked tensor of the boolean values.
    """
    # 调用refs模块的_uniform_helper函数生成一个形状与self相同的随机张量，
    # 其中每个元素满足在[0.0, 1.0)之间，作为掩码
    return (
        refs._uniform_helper(
            self.shape, low=0.0, high=1.0, dtype=torch.float32, device=self.device
        )
        < val
    )


@register_decomposition(aten.alpha_dropout)
def alpha_dropout(
    self: TensorLikeType, p: float = 0.5, training: bool = False, inplace: bool = False
) -> TensorLikeType:
    """
    Applies alpha dropout to the input tensor.

    Args:
        self: Input tensor.
        p: Dropout probability, must be between 0 and 1.
        training: Whether the model is in training mode.
        inplace: Whether to perform the operation in-place (not supported).

    Returns:
        Tensor: Output tensor after applying alpha dropout.
    """
    # 如果要求就地操作，则抛出NotImplementedError
    if inplace:
        raise NotImplementedError

    # 如果不是在训练模式，则直接返回原始输入
    if not training:
        return self

    # 检查dropout概率p必须在0到1之间，否则抛出异常
    torch._check(
        p <= 1 and p >= 0,
        lambda: f"dropout probability has to be between 0 and 1, but got, {p}",
    )

    # 如果dropout概率为1，则返回与输入形状相同的全零张量
    if p == 1:
        return torch.zeros_like(self)

    # 如果dropout概率为0，则直接返回输入张量
    if p == 0:
        return self

    # 调用_dropout_helper函数获取dropout掩码
    dropout_mask = _dropout_helper(self, 1 - p)

    # 计算alpha值，用于alpha dropout公式
    # 根据论文《Self-Normalizing Neural Networks》中的推导，这里的alpha为固定值
    alpha = -1.7580993408473766

    # 计算alpha dropout的缩放系数a和偏置项b
    a = 1.0 / math.sqrt((alpha * alpha * p + 1) * (1 - p))
    b = torch.logical_not(dropout_mask)
    b = b * (alpha * a) + alpha * a * p
    dropout_mask = a * dropout_mask

    # 应用alpha dropout公式，返回处理后的张量
    return self * dropout_mask + b


def _inplace_wrapper(fn):
    """
    Given a nn.functional non-linearity, implements its `inplace: bool` argument
    """
    # 使用装饰器 wraps 将内部函数 _fn 的元数据（如文档字符串和函数名）复制到包装后的函数中
    @wraps(fn)
    # 定义一个内部函数 _fn，接受一个位置参数 a 和任意数量的位置参数 *args，还有一个关键字参数 inplace 和任意数量的关键字参数 **kwargs
    def _fn(a, *args, inplace=False, **kwargs):
        # 如果 inplace 参数为 True，则进行如下操作
        if inplace:
            # 使用 torch._check 函数检查 kwargs 中是否包含 "out" 参数，如果包含则抛出异常
            torch._check(
                "out" not in kwargs,
                lambda: "Cannot set inplace=True and pass out= at the same time",
            )
            # 调用原始函数 fn，传递参数 a, *args, inplace=False, out=a 和 **kwargs，并返回结果
            return fn(a, *args, inplace=False, out=a, **kwargs)
        else:
            # 如果 inplace 参数为 False，则调用原始函数 fn，传递参数 a, *args, inplace=False 和 **kwargs，并返回结果
            return fn(a, *args, inplace=False, **kwargs)
    
    # 返回内部函数 _fn 的引用作为装饰后的函数
    return _fn
# 注册函数的装饰器，将 torch.celu 函数注册为分解函数的一部分
# 注册函数的装饰器，标记为原地操作
# 标记输出包装器的装饰器
# 元素级类型提升的包装器，推广参数为 ("a",)
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
# 定义 celu 函数，实现 torch.nn.functional.celu 的参考实现
def celu(
    a: TensorLikeType, alpha: Optional[NumberType] = None, inplace: bool = False
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.celu
    """

    # 如果 inplace 参数为 True，则抛出未实现的错误
    if inplace:
        raise NotImplementedError

    # 右手边的变量声明为 TensorLikeType
    rhs: TensorLikeType
    # 如果 alpha 参数不为 None，则执行以下操作
    if alpha is not None:
        # 获取 a 的 Python 类型
        python_type = utils.dtype_to_type(a.dtype)
        # 如果 alpha 的类型不能安全地转换为 a 的类型，则抛出值错误
        if not utils.is_weakly_lesser_type(type(alpha), python_type):
            msg = f"alpha argument of type {type(alpha)} cannot be safely cast to type {python_type}!"
            raise ValueError(msg)
        # 计算 rhs 作为 alpha 乘以 torch.expm1(torch.true_divide(a, alpha)) 的结果
        rhs = alpha * torch.expm1(torch.true_divide(a, alpha))  # type: ignore[arg-type]
    else:
        # 否则，rhs 为 torch.expm1(a) 的结果
        rhs = torch.expm1(a)

    # 返回 torch.where(a > 0, a, rhs) 的结果
    return torch.where(a > 0, a, rhs)


# 原地操作的包装器函数
# 标记输出包装器的装饰器
def dropout(
    a: TensorLikeType, p: float = 0.5, training: bool = True, inplace: bool = False
) -> TensorLikeType:
    # 如果 inplace 参数为 True，则抛出未实现的错误
    if inplace:
        raise NotImplementedError

    # 如果不处于训练模式，则直接返回 a
    if not training:
        return a

    # 检查 dropout 概率 p 是否在 [0, 1] 之间，否则抛出错误
    torch._check(
        p <= 1 and p >= 0,
        lambda: f"dropout probability has to be between 0 and 1, but got, {p}",
    )

    # 如果 p 等于 1，则返回一个与 a 相同形状的全零张量
    if p == 1:
        return torch.zeros_like(a)

    # 如果 p 等于 0，则直接返回 a
    if p == 0:
        return a

    # 计算 scale 为 1 / (1 - p)
    scale = 1 / (1 - p)
    # 调用 _dropout_helper(a, 1 - p) 获取 dropout_mask
    dropout_mask = _dropout_helper(a, 1 - p)

    # 返回 a * dropout_mask * scale 的结果
    return a * dropout_mask * scale


# 注册函数的装饰器，将 torch.elu 函数注册为分解函数的一部分
# 注册函数的装饰器，标记为原地操作
# 标记输出包装器的装饰器
# 元素级类型提升的包装器，推广参数为 ("a",)
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
# 定义 elu 函数，实现 torch.nn.functional.elu 的参考实现
def elu(
    a: TensorLikeType,
    alpha: NumberType = 1.0,
    scale: NumberType = 1.0,
    input_scale: NumberType = 1.0,
    inplace: bool = False,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.elu
    """
    # 如果 inplace 参数为 True，则抛出未实现的错误
    if inplace:
        raise NotImplementedError

    # 获取 a 的 Python 类型
    python_type = utils.dtype_to_type(a.dtype)
    # 检查 input_scale 的类型是否可以安全地转换为 a 的类型，否则抛出错误
    torch._check(
        utils.is_weakly_lesser_type(type(input_scale), python_type),
        lambda: f"input_scale argument of type {type(input_scale)} cannot be safely cast to type {python_type}!",
    )
    # 检查 scale 的类型是否可以安全地转换为 a 的类型，否则抛出错误
    torch._check(
        utils.is_weakly_lesser_type(type(scale), python_type),
        lambda: f"scale argument of type {type(scale)} cannot be safely cast to type {python_type}!",
    )
    # 检查 alpha 的类型是否可以安全地转换为 a 的类型，否则抛出错误
    torch._check(
        utils.is_weakly_lesser_type(type(alpha), python_type),
        lambda: f"alpha argument of type {type(alpha)} cannot be safely cast to type {python_type}!",
    )

    # 返回 torch.where(a > 0, scale * a, (alpha * scale) * torch.expm1(a * input_scale)) 的结果
    return torch.where(a > 0, scale * a, (alpha * scale) * torch.expm1(a * input_scale))
    # 设定函数参数 `type_promoting_args`，指定为一个包含单个字符串 "a" 的元组
    type_promoting_args=("a",),
    # 设定函数参数 `type_promotion_kind`，指定为元素级别的类型提升种类，默认为 DEFAULT
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
# 注册 selu 函数的分解操作
@register_decomposition(aten.selu)
# 包装函数，用于处理 inplace 操作
@_inplace_wrapper
# 包装函数，用于输出
@out_wrapper()
# 元素级类型提升的包装器，指定要提升类型的参数
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
# SELU 激活函数的参考实现
def selu(
    a: TensorLikeType, inplace: bool = False
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.selu
    """

    # 如果 inplace 为 True，暂不支持
    if inplace:
        raise NotImplementedError
    # 计算 SELU 激活函数的输出
    return a * torch.selu(a)
# 使用元素类型推广包装器，指定要推广类型的参数为"a"
# 使用默认的元素类型推广方式
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
# 实现 torch.nn.functional.selu 的参考实现
def selu(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.selu
    """
    # 如果 inplace 参数为 True，则抛出未实现的错误
    if inplace:
        raise NotImplementedError

    # 设定 alpha 和 scale 的值
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    # 计算 selu 函数的右手边值
    rhs = alpha * torch.expm1(a)

    # 返回计算结果，根据条件选择 a 或 rhs
    return scale * torch.where(a > 0, a, rhs)


# 前向别名：函数变体不支持 out 参数
# CompositeImplicitAutograd - 不注册分解
def softmax(
    a: TensorLikeType,
    dim: Optional[int] = None,
    _stacklevel: int = 3,  # 兼容 TorchRefsMode(strict=True) 时使用
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    # 此错误是为了与常规 PyTorch 兼容，后者有此行为已弃用。对于 PrimTorch，放弃对已弃用行为的支持是可以的，因为它需要明确的选择。此错误用于通知用户如何更新其调用。
    torch._check(dim is not None, lambda: "implicit dim not supported, use dim=X")
    # 调用 torch.softmax 函数，返回结果
    return torch.softmax(a=a, dim=dim, dtype=dtype)  # type: ignore[call-overload]


# CompositeImplicitAutograd - 不注册分解
def softmin(
    a: TensorLikeType,
    dim: Optional[int] = None,
    _stacklevel: int = 3,  # 兼容 TorchRefsMode(strict=True) 时使用
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    # 此错误是为了与常规 PyTorch 兼容，后者有此行为已弃用。对于 PrimTorch，放弃对已弃用行为的支持是可以的，因为它需要明确的选择。此错误用于通知用户如何更新其调用。
    torch._check(dim is not None, lambda: "implicit dim not supported, use dim=X")
    # 调用 torch.softmax 函数，返回结果，输入 a 取反
    return torch.softmax(a=-a, dim=dim, dtype=dtype)  # type: ignore[call-overload]


# softplus 实现了特殊处理，因为它有 beta 和 threshold 参数
# 注册 aten.softplus 的分解
# _inplace_wrapper 装饰器用于原地操作
# out_wrapper 装饰器用于处理输出
# 使用元素类型推广包装器，指定要推广类型的参数为"a"
# 使用默认的元素类型推广方式
@register_decomposition(aten.softplus)
@_inplace_wrapper
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
# 实现 torch.nn.functional.softplus 的参考实现
def softplus(
    a: TensorLikeType,
    beta: Optional[NumberType] = None,
    threshold: NumberType = 20,
    inplace: bool = False,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.softplus
    """

    # 如果 inplace 参数为 True，则抛出未实现的错误
    if inplace:
        raise NotImplementedError

    # 如果 beta 不为 None，则进行类型安全检查
    rhs: TensorLikeType
    if beta is not None:
        python_type = utils.dtype_to_type(a.dtype)
        if not utils.is_weakly_lesser_type(type(beta), python_type):
            msg = f"beta argument of type {type(beta)} cannot be safely cast to type {python_type}!"
            raise ValueError(msg)
        scaled_input = a * beta
        rhs = torch.true_divide(torch.log1p(torch.exp(scaled_input)), beta)  # type: ignore[arg-type]
    # 如果条件不成立，则将输入张量a赋值给scaled_input
    else:
        scaled_input = a
        # 计算log(1 + exp(scaled_input))，用于数值稳定性，避免直接计算exp(scaled_input)可能造成的溢出
        rhs = torch.log1p(torch.exp(scaled_input))

    # 返回一个新的张量，根据条件选择scaled_input或rhs作为每个元素的值
    return torch.where(scaled_input > threshold, a, rhs)
# 使用 torch 的硬阈值函数实现，以默认的 PyTorch 自动微分调度键进行注册
@aten.hardshrink.default.py_impl(DispatchKey.Autograd)
# 注册到 aten.hardshrink 的分解
@register_decomposition(aten.hardshrink)
# 将输出包装在装饰器中
@out_wrapper()
# 定义硬阈值函数，输入参数为 a: 输入张量类型，lambd: 阈值，默认为 0.5
def hardshrink(a: TensorLikeType, lambd: float = 0.5):
    # 硬阈值函数的定义
    # hardshrink(x) = x if x > lambd
    #               = x if x < -lambd
    #               = 0 otherwise
    return torch.where(torch.abs(a) <= lambd, 0, a)


# 使用 torch 的软阈值函数实现，以默认的 PyTorch 自动微分调度键进行注册
@aten.softshrink.default.py_impl(DispatchKey.Autograd)
# 注册到 aten.softshrink 的分解
@register_decomposition(aten.softshrink)
# 将输出包装在装饰器中
@out_wrapper()
# 定义软阈值函数，输入参数为 a: 输入张量类型，lambd: 阈值，默认为 0.5
def softshrink(a: TensorLikeType, lambd: float = 0.5):
    # 软阈值函数的定义
    # softshrink(x) = x - lambd if x > lambd
    #               = x + lambd if x < -lambd
    #               = 0 otherwise
    # 检查 lambd 是否大于等于 0，否则引发异常
    torch._check(
        lambd >= 0,
        lambda: f"lambda must be greater or equal to 0, but found to be {lambd}",
    )
    # 实现软阈值函数的一种方式，利用 torch.where 在反向传播时生成更好的代码
    # 参见 https://github.com/pytorch/pytorch/pull/107052#discussion_r1293748211
    return torch.where(torch.abs(a) > lambd, a - torch.sign(a) * lambd, 0)


# 损失函数相关

# 将整数类型的减少方式转换为字符串
def _reduction_int_to_str(reduction: int) -> str:
    from torch._decomp.decompositions import Reduction

    if reduction == Reduction.NONE.value:
        return "none"
    elif reduction == Reduction.MEAN.value:
        return "mean"
    elif reduction == Reduction.SUM.value:
        return "sum"
    else:
        raise ValueError(f"{reduction} is not a valid value for reduction")


# 根据指定的减少方式应用损失减少操作
def _apply_loss_reduction(loss: TensorLikeType, reduction: str) -> TensorLikeType:
    if reduction == "sum":
        return torch.sum(loss)
    elif reduction == "mean":
        return torch.mean(loss)
    else:  # reduction == "none"
        return loss


# 检查减少值是否有效
def _check_reduction_value(reduction: str):
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"{reduction} is not a valid value for reduction")


# 该辅助函数将 "size_average" 和 "reduce" 参数映射到相应的 "reduction" 字符串参数
def _get_string_reduction_arg(
    *, size_average: Optional[bool], reduce: Optional[bool]
) -> str:
    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True
    if size_average and reduce:
        ret = "mean"
    elif reduce:
        ret = "sum"
    else:
        ret = "none"
    return ret


# CompositeImplicitAutograd - 不注册分解
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)
# L1 损失函数的参考实现，输入参数为 input: 输入张量类型，target: 目标张量类型，
# size_average: 是否对损失求均值，默认为 None，reduce: 是否对损失进行缩减，默认为 None，
# reduction: 减少方式，默认为 "mean"
def l1_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> TensorLikeType:
    """
    torch.nn.functional.l1_loss 的参考实现
    """
    # 如果 size_average 或 reduce 参数不为 None，则执行以下操作
    if size_average is not None or reduce is not None:
        # TODO: 抛出异常而不是转换数值。这仅适用于primTorch，因为它可以放弃对废弃参数的支持。
        # msg = "size_average and reduce args are deprecated, please use reduction argument."
        
        # 根据 size_average 和 reduce 参数获取对应的字符串形式的 reduction 参数
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)
    
    # 检查 reduction 参数的值是否合法
    _check_reduction_value(reduction)
    
    # 计算输入张量 input 和目标张量 target 的绝对误差
    loss = torch.abs(input - target)
    
    # 应用指定的 reduction 方式对损失值进行处理
    return _apply_loss_reduction(loss, reduction)
# 应用元素级类型提升装饰器，将输入和目标数据提升为浮点数
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)
# 定义平滑 L1 损失函数
def smooth_l1_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    beta: float = 1.0,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.smooth_l1_loss
    """
    # 如果 size_average 或 reduce 参数不为 None，则抛出异常，提示使用 reduction 参数
    if size_average is not None or reduce is not None:
        # TODO: Raise exception instead of converting value.  This is only for
        # primTorch since it can drop support for deprecated arguments.
        # msg = "size_average and reduce args are deprecated, please use reduction argument."
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)
    # 检查 reduction 参数值的有效性
    _check_reduction_value(reduction)

    # 如果 beta 等于 0.0，则返回 L1 损失的计算结果
    if beta == 0.0:
        return torch.nn.functional.l1_loss(
            input, target, size_average=size_average, reduce=reduce, reduction=reduction
        )
    else:
        # 计算平滑 L1 损失
        loss = torch.abs(input - target)
        loss = torch.where(loss < beta, 0.5 * loss**2 / beta, loss - 0.5 * beta)
        return _apply_loss_reduction(loss, reduction)


# Forwarding alias: the functional variant doesn't support the out kwarg
# CompositeImplicitAutograd - don't register decomp
# 对数 softmax 函数的前向别名，函数变体不支持 out 参数
def log_softmax(
    a: TensorLikeType,
    dim: Optional[int] = None,
    _stacklevel: int = 3,  # for compat when using TorchRefsMode(strict=True)
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    # 对于常规 PyTorch，如果使用了隐式的 dim 参数，将会发出错误信息，因为这种行为已经被弃用
    # 对于 PrimTorch，可以放弃对弃用行为的支持，因为它需要显式的选择。这个错误是为了通知用户如何更新其调用方式。
    torch._check(dim is not None, lambda: "implicit dim not supported, use dim=X")
    return torch.log_softmax(a=a, dim=dim, dtype=dtype)  # type: ignore[call-overload]


# 注册对 margin_ranking_loss 函数的分解
@register_decomposition(aten.margin_ranking_loss)
def margin_ranking_loss(
    input1: TensorLikeType,
    input2: TensorLikeType,
    target: TensorLikeType,
    margin: float = 0.0,
    reduction: str = "mean",
) -> TensorLikeType:
    # 如果 input1、input2 和 target 张量的维度不同，则抛出运行时错误
    if input1.ndim != input2.ndim or input1.ndim != target.ndim:
        raise RuntimeError(
            "margin_ranking_loss : All input tensors should have same dimension but got sizes: "
            f"input1: {input1.shape}, input2: {input2.shape}, target: {target.shape} "
        )
    # 检查 reduction 参数值的有效性
    _check_reduction_value(reduction)
    # 计算 margin ranking loss，使用 torch.clamp_min 来确保 loss 大于等于 0
    loss = torch.clamp_min(-target * (input1 - input2) + margin, 0)
    return _apply_loss_reduction(loss, reduction)


# 应用元素级类型提升装饰器，将输入和目标数据提升为浮点数
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)
# 定义均方误差损失函数
def mse_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> TensorLikeType:
    size_average: Optional[bool] = None,
    # size_average 参数，可选布尔值，默认为 None
    reduce: Optional[bool] = None,
    # reduce 参数，可选布尔值，默认为 None
    reduction: str = "mean",
    # reduction 参数，字符串类型，默认为 "mean"
# 定义函数签名，指定返回类型为 TensorLikeType
) -> TensorLikeType:
    # 如果 size_average 或 reduce 参数不为 None，则抛出异常而非转换值。
    # 这仅适用于 primTorch，因为它可以停止对废弃参数的支持。
    # msg = "size_average and reduce args are deprecated, please use reduction argument."
    # 根据 size_average 和 reduce 参数获取 reduction 参数的字符串表示
    reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)
    
# 检查 reduction 参数的有效性
_check_reduction_value(reduction)

# 计算平方误差损失
loss = torch.pow(input - target, 2)

# 应用指定的损失函数减少方式，返回处理后的损失值
return _apply_loss_reduction(loss, reduction)


# 使用 aten.hinge_embedding_loss 注册为分解函数的装饰器
@register_decomposition(aten.hinge_embedding_loss)
def hinge_embedding_loss(
    # 输入张量
    input: TensorLikeType,
    # 目标张量
    target: TensorLikeType,
    # 间隔值，默认为 1.0
    margin: float = 1.0,
    # 减少方式，默认为 "mean"
    reduction: str = "mean",
) -> TensorLikeType:
    # 检查 reduction 参数的有效性
    _check_reduction_value(reduction)
    
    # 计算 margin_clamp，使用 torch.clamp_min() 确保大于等于 0
    margin_clamp = torch.clamp_min(margin - input, 0)
    
    # 计算 output_margin，当 target 不为 1 时为 margin_clamp，否则为 0
    output_margin = torch.where(target != 1, margin_clamp, 0)
    
    # 计算 output_self，当 target 不为 -1 时为 input，否则为 0
    output_self = torch.where(target != -1, input, 0)
    
    # 计算总损失，为 output_margin 和 output_self 的和
    loss = output_margin + output_self
    
    # 应用指定的损失函数减少方式，返回处理后的损失值
    return _apply_loss_reduction(loss, reduction)


# 定义 _nll_loss_nd 函数
def _nll_loss_nd(
    # 输入张量
    input: TensorLikeType,
    # 目标张量
    target: TensorLikeType,
    # 权重张量，可选
    weight: Optional[TensorLikeType],
    # 减少方式
    reduction: str,
    # 忽略的索引
    ignore_index: int,
) -> TensorLikeType:
    # 检查输入张量维度是否在合法范围 [1, 2, 3]
    torch._check(
        input.ndim > 0 and input.ndim <= 3,
        lambda: f"Expected input dimension to be either [1, 2, 3] but received {input.ndim}.",
    )
    
    # 检查输入张量的批处理大小是否与目标张量的批处理大小匹配
    torch._check(
        (input.ndim == 1) or (input.shape[0] == target.shape[0]),
        lambda: f"Expected input batch size {input.shape[0]} to match target batch size {target.shape[0]}.",
    )
    
    # 检查 reduction 参数的有效性
    _check_reduction_value(reduction)
    
    # 将目标张量扁平化
    flat_target = torch.flatten(target)
    
    # 创建忽略类别的掩码
    ignore_classes_mask = torch.eq(flat_target, ignore_index)
    
    """
    # TODO: Enable data-dependent checks with debug mode
    # TODO: This check does not work with FakeTensor inputs; See Issue #85834
    # 显式转换为 bool 类型的 class_check；参见 Issue #78071
    from torch._subclasses.fake_tensor import FakeTensor
    num_classes = input.shape[1] if input.ndim > 1 else input.shape[0]
    valid_classes_mask = torch.logical_and(
        (flat_target >= 0), (flat_target < num_classes)
    )
    class_check = torch.all(torch.logical_or(ignore_classes_mask, valid_classes_mask))
    torch._check(
        isinstance(target, FakeTensor) or bool(class_check.item()),
        lambda: "A target class is out-of-bounds and not the ignore index.",
    )
    """
    
    # 初始化忽略类别权重为零
    ignore_class_weight = torch.scalar_tensor(0, dtype=input.dtype, device=input.device)
    
    # 根据权重张量和目标张量的扁平化结果选择类别权重
    class_weight = (
        torch.scalar_tensor(1, dtype=input.dtype, device=input.device)
        if weight is None
        else weight[flat_target]
    )
    
    # 根据忽略类别掩码选择当前权重
    current_weight = torch.where(
        ignore_classes_mask,
        ignore_class_weight,
        class_weight,
    )
    if input.ndim == 1:
        # 如果输入是一维的，隐式批处理大小为1
        # input (1 batch size, C classes)
        # 计算损失，针对单个目标类别，乘以当前权重
        loss = -input[target] * current_weight
    elif input.ndim == 2:
        # 如果输入是二维的
        # input (N batch size, C classes)
        # 获取批处理大小
        batch_size = input.shape[0]
        # 计算损失，对每个样本中的目标类别，乘以当前权重
        loss = -input[torch.arange(batch_size), target] * current_weight
    else:
        # 如果输入是三维的
        # 3D case (N batch size, C classes, K dimensions)
        # input (N batch size, C classes, K)
        # 获取批处理大小和每个样本的维度数
        batch_size = input.shape[0]
        extent = input.shape[2]
        numel = batch_size * extent
        # 创建用于索引的序列
        indices = torch.arange(numel)
        # 计算损失，对每个批次、平坦化目标、以及每个样本维度，乘以当前权重
        bdx = indices // extent
        kdx = indices % extent
        loss = -input[bdx, flat_target, kdx] * current_weight
    # 将损失重塑成与目标形状相同
    loss = torch.reshape(loss, target.shape)

    if reduction == "none":
        # 如果不进行减少操作，直接返回损失
        return loss
    elif reduction == "sum":
        # 如果进行求和减少操作，返回损失的总和
        return torch.sum(loss)
    else:
        # 如果进行加权平均损失函数的计算
        # 返回损失总和除以当前权重总和，得到加权平均值
        return torch.sum(loss) / torch.sum(current_weight)
@register_decomposition(aten.nll_loss)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
# 定义 nll_loss 函数，实现 torch.nn.functional.nll_loss 的参考实现
def nll_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    weight: Optional[TensorLikeType] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.nll_loss
    """

    # 检查输入张量的维度是否大于0
    torch._check(
        input.ndim > 0,
        lambda: f"Expected input tensor to have 1 or more dimensions (got {input.ndim})",
    )

    # TODO: raise exception instead of converting value
    # msg = "size_average and reduce args are deprecated, please use reduction argument."
    # 如果 size_average 或 reduce 不为 None，则使用 _get_string_reduction_arg 函数进行转换以保持与 eager 模式的一致性
    if size_average is not None or reduce is not None:
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)

    # 当输入和目标张量都没有元素时的预期行为：
    #   reduction = 'none' --- 返回空张量 tensor([])
    #   reduction = 'sum'  --- 返回空张量 tensor(0.)
    #   reduction = 'mean' --- 返回空张量 tensor(nan)
    # 空张量的均值计算会产生 NaN。参考 https://github.com/pytorch/pytorch/pull/64572#issuecomment-926504162
    if input.numel() == 0 and target.numel() == 0:
        if reduction == "none":
            return torch.zeros_like(target)
        elif reduction == "sum":
            return torch.empty_like(target)
        else:
            return torch.full_like(target, float("nan"))

    # _nll_loss_nd 辅助函数处理最常见的情况。
    # ndim == 1 (单个样本)
    #   => 批大小: 1, 输入: (C), 目标: ()
    # ndim == 2 (k = 1)
    #   => 批大小: N, 输入: (N, C), 目标: (N)
    # ndim == 3 (k > 1)
    #   => 批大小: N, 输入: (N, C, K), 目标: (N, K)
    if input.ndim <= 3:
        return _nll_loss_nd(input, target, weight, reduction, ignore_index)

    # 对于 ndim > 3 的情况，将输入和目标重塑为3维情况。
    # 输入 (N 批大小, C 类别数, k 维度)
    # 目标 (N 批大小, k 维度)
    torch._check(
        input.ndim > 0 and target.ndim > 0 and target.shape[1:] == input.shape[2:],
        lambda: (
            "Expected input and target to both have ndim > 0 and "
            "target.shape[1:] == input.shape[2:], but got "
            f"target.shape {target.shape} and input.shape {input.shape}"
        ),
    )

    # 计算批大小、类别数和输出尺寸
    batch_size = input.shape[0]
    num_classes = input.shape[1]
    out_size = [batch_size] + list(target.shape[1:])

    # 将输入和目标张量重塑为3维情况后再调用 _nll_loss_nd 函数，根据指定的 reduction 进行损失计算
    input = torch.reshape(input, [batch_size, num_classes, -1])
    target = torch.reshape(target, [batch_size, -1])
    if reduction != "none":
        return _nll_loss_nd(input, target, weight, reduction, ignore_index)
    else:
        # 如果条件不满足，则调用 _nll_loss_nd 函数计算损失
        result = _nll_loss_nd(input, target, weight, reduction, ignore_index)
        # 将结果张量按照指定的输出大小 out_size 进行重塑
        return torch.reshape(result, out_size)
# TODO: This ref supports int reduction and out kwarg to be compatible with ATen:
# https://github.com/pytorch/pytorch/issues/83931
# TODO: Could be rewritten to support complex:
# https://github.com/pytorch/pytorch/pull/85041

# 注册 huber_loss 函数的分解方法，用于自动微分
@register_decomposition(aten.huber_loss)
# 应用输出包装器，但此处未指定具体函数
@out_wrapper()
# 应用元素级类型提升包装器，将输入参数 "input" 和 "target" 的类型提升
# 类型提升种类为默认的 ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
# 定义 huber_loss 函数，接受张量类输入 input 和 target，以及额外的参数
def huber_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    reduction: Union[str, int] = "mean",
    delta: float = 1.0,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.huber_loss
    """
    # 如果 reduction 是整数类型，则将其转换为字符串类型
    if type(reduction) is int:
        reduction = _reduction_int_to_str(reduction)
    # 检查 reduction 值的有效性
    _check_reduction_value(reduction)  # type: ignore[arg-type]
    # 检查 delta 参数值必须大于 0，否则抛出异常
    torch._check(
        delta > 0,
        lambda: "huber_loss does not support non-positive values for delta.",
    )
    # 计算输入和目标之间的绝对误差
    z = (input - target).abs()
    # 根据 z 是否小于 delta 来计算损失
    loss = torch.where(z < delta, 0.5 * z * z, delta * (z - 0.5 * delta))
    # 应用损失函数的减少操作，根据指定的 reduction 类型
    return _apply_loss_reduction(loss, reduction)  # type: ignore[arg-type]


# tanhshrink 函数不使用 _make_elementwise_unary_reference，因为它不支持 out 参数
@elementwise_unary_scalar_wrapper
# 应用元素级类型提升包装器，将输入参数 "a" 的类型提升
# 类型提升种类为 INT_TO_FLOAT
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
# 定义 tanhshrink 函数，接受张量类输入 a，并返回处理后的张量
def tanhshrink(a: TensorLikeType) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.tanhshrink
    """
    # 如果 a 不是张量类，则抛出运行时异常
    if not isinstance(a, TensorLike):
        raise RuntimeError(
            "Expected a tensor input for an elementwise unary operation!"
        )
    # 返回 tanhshrink 函数的处理结果
    return a - torch.tanh(a)


# 注册 threshold 函数的分解方法，用于自动微分
@register_decomposition(aten.threshold)
# 应用 _inplace_wrapper 修饰器，表示该函数支持原地操作
@_inplace_wrapper
# 应用输出包装器，但此处未指定具体函数
@out_wrapper()
# 应用元素级类型提升包装器，将输入参数 "a" 的类型提升
# 类型提升种类为默认的 ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
# 定义 threshold 函数，接受张量类输入 a 和其他参数，返回处理后的张量
def threshold(
    a: TensorLikeType,
    threshold: NumberType,
    value: Union[bool, int, float],
    inplace: bool = False,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.threshold
    """

    # 如果 inplace 为 True，则抛出未实现的异常
    if inplace:
        raise NotImplementedError

    # 使用 torch.where 函数根据条件进行阈值处理
    return torch.where(a <= threshold, value, a)


# CompositeImplicitAutograd - 不注册分解方法
# 没有元素级类型提升 - 核心操作不明确支持类型提升
def triplet_margin_loss(
    anchor: TensorLikeType,
    positive: TensorLikeType,
    negative: TensorLikeType,
    margin: float = 1.0,
    p: float = 2,
    eps: float = 1e-6,
    swap: bool = False,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> TensorLikeType:
    # 三元组间隔损失函数的参考实现
    # 该函数没有注册分解方法，也没有元素级类型提升
    # 如果 size_average 或 reduce 任一不为 None，则执行以下操作：
    # TODO: 抛出异常而不是转换数值。这仅适用于 primTorch，因为它可能停止支持废弃的参数。
    # msg = "size_average and reduce args are deprecated, please use reduction argument."
    # 根据 size_average 和 reduce 参数获取 reduction 参数的字符串表示
    reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)

    # 如果 margin 小于等于 0，则抛出数值错误
    raise ValueError(f"margin must be greater than 0, got {margin}")

    # torch.nn.functional.triplet_margin_with_distance_loss 没有定义引用，
    # 因为它是纯 Python 实现。使用这个辅助函数代替。
    return _triplet_margin_with_distance_loss(
        anchor=anchor,
        positive=positive,
        negative=negative,
        # 定义距离函数，使用 torch.pairwise_distance 计算距离
        distance_function=lambda x, y: torch.pairwise_distance(x, y, p, eps),
        margin=margin,
        swap=swap,
        reduction=reduction,
    )
# Pure Python impl - don't register decomp and don't add a ref.  Defined as a
# helper here since triplet_margin_loss can be nicely implemented with it.
# 定义了一个纯Python实现的函数，不会注册decomp，也不会添加引用。
# 这里将其定义为一个辅助函数，因为triplet_margin_loss可以很好地使用它实现。

def _triplet_margin_with_distance_loss(
    anchor: TensorLikeType,
    positive: TensorLikeType,
    negative: TensorLikeType,
    *,
    distance_function: Optional[
        Callable[[TensorLikeType, TensorLikeType], TensorLikeType]
    ] = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean",
) -> TensorLikeType:
    """
    Computes triplet margin loss using the specified distance function and margin.

    Args:
        anchor: Tensor representing the anchor data point.
        positive: Tensor representing the positive data point.
        negative: Tensor representing the negative data point.
        distance_function: Optional distance function to compute distances between tensors.
        margin: Margin value for the triplet loss.
        swap: Boolean flag indicating whether to perform distance swap in loss computation.
        reduction: Specifies the reduction to apply to the computed loss.

    Returns:
        Tensor representing the computed triplet margin loss.
    """

    _check_reduction_value(reduction)

    # Check if anchor, positive, and negative tensors have the same number of dimensions
    a_dim = anchor.ndim
    p_dim = positive.ndim
    n_dim = negative.ndim
    torch._check(
        a_dim == p_dim and p_dim == n_dim,
        lambda: (
            f"The anchor, positive, and negative tensors are expected to have "
            f"the same number of dimensions, but got: anchor {a_dim}D, "
            f"positive {p_dim}D, and negative {n_dim}D inputs"
        ),
    )

    # Set default distance function if not provided
    if distance_function is None:
        distance_function = torch.pairwise_distance

    # Compute distances between anchor and positive, anchor and negative tensors
    dist_pos = distance_function(anchor, positive)
    dist_neg = distance_function(anchor, negative)

    # Perform distance swap if specified
    # This handles the scenario where the positive example is closer to the negative
    # example than the anchor, as described in the referenced paper.
    if swap:
        dist_swap = distance_function(positive, negative)
        dist_neg = torch.minimum(dist_neg, dist_swap)

    # Compute triplet margin loss using the specified margin and computed distances
    loss = torch.clamp_min(margin + dist_pos - dist_neg, 0)

    # Apply reduction to the computed loss
    return _apply_loss_reduction(loss, reduction)


@register_decomposition(aten.hardtanh)
@_inplace_wrapper
@out_wrapper()
@elementwise_unary_scalar_wrapper
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
def hardtanh(
    a: TensorLikeType,
    min_val: NumberType = -1,
    max_val: NumberType = 1,
    inplace: bool = False,
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.hardtanh.

    Args:
        a: Input tensor to which the hardtanh function is applied.
        min_val: Minimum value of the output tensor after applying hardtanh.
        max_val: Maximum value of the output tensor after applying hardtanh.
        inplace: Whether to apply hardtanh function in-place (not supported).

    Returns:
        Tensor: Output tensor after applying hardtanh function element-wise.
    """

    if inplace:
        raise NotImplementedError("Inplace operation is not supported for hardtanh")
    
    if utils.is_boolean_dtype(a.dtype):
        raise RuntimeError("Bool inputs not supported for hardtanh")

    # Preserve legacy behavior of boundaries not causing type promotion
    if utils.is_integer_dtype(a.dtype):
        min_val = int(min_val)  # type: ignore[arg-type]
        max_val = int(max_val)  # type: ignore[arg-type]
        if not (a.dtype != torch.uint8 or (min_val >= 0 and max_val >= 0)):
            raise RuntimeError(
                "Cannot do hardtanh on an unsigned type with negative limits"
            )

    # Check if min_val is greater than max_val
    if min_val > max_val:  # type: ignore[operator]
        raise ValueError("min_val cannot be greater than max_val")

    # Apply hardtanh function element-wise to tensor 'a' with specified min and max values
    return torch.clamp(a, min_val, max_val)  # type: ignore[arg-type]
# 将函数装饰为元素级一元标量包装器
@elementwise_unary_scalar_wrapper
# 将函数装饰为元素级类型提升包装器，指定类型提升参数和默认的类型提升种类
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
# 定义 GELU 激活函数
def gelu(a: TensorLikeType, approximate: str = "none") -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.gelu
    """
    # 如果 a 不是 TensorLike 类型，则引发运行时错误
    if not isinstance(a, TensorLike):
        raise RuntimeError(
            "Expected a tensor input for an elementwise unary operation!"
        )
    # 定义常数 M_SQRT2, M_SQRT1_2, M_2_SQRTPI
    M_SQRT2 = 1.41421356237309504880
    M_SQRT1_2 = 0.70710678118654752440
    M_2_SQRTPI = 1.12837916709551257390
    
    # 根据 approximate 参数选择不同的近似方式计算 GELU 函数
    if approximate == "tanh":
        kBeta = M_SQRT2 * M_2_SQRTPI * 0.5
        kKappa = 0.044715
        a_cube = a * a * a
        inner = kBeta * (a + kKappa * a_cube)
        return 0.5 * a * (1 + torch.tanh(inner))
    elif approximate == "none":
        kAlpha = M_SQRT1_2
        return a * 0.5 * (1 + torch.erf(a * kAlpha))
    else:
        # 如果 approximate 参数既不是 "none" 也不是 "tanh"，则引发运行时错误
        raise RuntimeError("approximate argument must be either none or tanh.")


# CompositeImplicitAutograd - don't register decomp
# 将函数装饰为元素级类型提升包装器，指定类型提升参数和整数到浮点数的类型提升种类
@elementwise_type_promotion_wrapper(
    type_promoting_args=("input", "target"),
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
# 定义 Poisson Negative Log-Likelihood 损失函数
def poisson_nll_loss(
    input: TensorLikeType,
    target: TensorLikeType,
    log_input: bool = True,
    full: bool = False,
    size_average: Optional[bool] = None,
    eps: float = 1e-8,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.poisson_nll_loss
    """
    # 如果 size_average 或 reduce 参数不是 None，则使用 reduction 参数的值，否则忽略
    if size_average is not None or reduce is not None:
        # TODO: Raise exception instead of converting value.  This is only for
        # primTorch since it can drop support for deprecated arguments.
        # msg = "size_average and reduce args are deprecated, please use reduction argument."
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)
    # 检查 reduction 参数的值是否有效
    _check_reduction_value(reduction)
    
    # 根据 log_input 参数计算损失
    if log_input:
        loss = torch.exp(input) - target * input
    else:
        loss = input - target * torch.log(input + eps)

    # 如果 full 参数为 True，则计算包含 Stirling 项的损失
    if full:
        stirling_term = (
            target * torch.log(target) - target + 0.5 * torch.log(2 * torch.pi * target)
        )
        # 避免原地操作添加
        loss = loss + stirling_term.masked_fill(target <= 1, 0)
    
    # 应用损失的减少操作，并返回结果
    return _apply_loss_reduction(loss, reduction)


# 注册 ATen PReLU 函数的分解
@register_decomposition(aten.prelu)
# 将函数装饰为元素级类型提升包装器，指定类型提升参数和默认的类型提升种类
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "weight"),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
# 定义 PReLU 激活函数
def prelu(a: TensorLikeType, weight: TensorLikeType) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.prelu
    """
    # 检查 a 是否为 TensorLike 类型，如果不是，则引发异常
    torch._check(
        isinstance(a, TensorLike),
        lambda: f"prelu: Expected `a` to be tensor, but got: {type(a)}",
    )
    # 检查 `weight` 是否为 TensorLike 类型，如果不是，抛出异常并显示错误消息
    torch._check(
        isinstance(weight, TensorLike),
        lambda: f"prelu: Expected `weight` to be tensor, but got: {type(weight)}",
    )

    # 如果 `weight` 张量元素个数不为1，执行以下检查和调整
    if weight.numel() != 1:
        # 检查输入张量 `a` 的维度是否大于0，如果不是，抛出不允许零维输入张量的异常
        torch._check(a.ndim > 0, lambda: "Not allow zero-dim input tensor.")
        
        # 计算输入张量 `a` 的通道大小（如果有的话），如果输入张量的维度大于等于2，默认为1
        channel_size = a.shape[1] if a.ndim >= 2 else 1
        
        # 检查 `weight` 张量的元素个数是否与输入通道大小匹配，如果不匹配，抛出异常
        torch._check(
            weight.numel() == channel_size,
            lambda: f"Mismatch of parameter numbers and input channel size. Found parameter numbers ="
            f" {weight.numel()} and channel size = {channel_size}.",
        )

    # 检查 `weight` 张量的维度是否为0或1，如果不是，抛出异常
    torch._check(
        weight.ndim == 0 or weight.ndim == 1,
        lambda: f"prelu: Expected `weight` to be a scalar or 1D tensor, but got: "
        f"ndim = {weight.ndim}",
    )

    # 如果输入张量 `a` 的维度为0，将 `weight` 转换为标量（如果 `weight` 为1维张量），否则保持不变
    if a.ndim == 0:
        weight = weight[0] if weight.ndim == 1 else weight
    else:
        # 在指定的维度上广播 `weight` 张量，以匹配输入张量 `a` 的形状
        weight = prims.broadcast_in_dim(
            weight, a.shape, tuple() if weight.ndim == 0 else (0 if a.ndim == 1 else 1,)
        )

    # 使用 torch.where 函数实现 PReLU 激活函数，大于0的部分保持不变，小于等于0的部分乘以 `weight`
    return torch.where(a > 0, a, a * weight)
# 注册函数的装饰器，用于 ATen 函数 relu6
@register_decomposition(aten.relu6)
# 对函数进行原地操作的包装器装饰
@_inplace_wrapper
# 输出结果的包装器装饰
@out_wrapper()
# 定义 relu6 函数，实现 torch.nn.functional.relu6 的参考实现
def relu6(a: TensorLikeType, inplace: bool = False) -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.relu6
    """
    # 如果 inplace 参数为 True，则抛出未实现错误
    if inplace:
        raise NotImplementedError

    # 使用 hardtanh 函数实现 relu6 的功能，详细信息参见链接
    # https://github.com/pytorch/pytorch/pull/81142#discussion_r918220126
    # 这里选择使用 hardtanh 而非 clamp，以复制现有实现的行为
    return torch.nn.functional.hardtanh(a, 0, 6)


# 注册函数的装饰器，用于 ATen 函数 glu
@register_decomposition(aten.glu)
# 输出结果的包装器装饰
@out_wrapper()
# 元素类型提升的包装器装饰
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
# 定义 glu 函数
def glu(a: TensorLikeType, dim: int = -1) -> TensorLikeType:
    # 规范化维度参数 dim
    dim = utils.canonicalize_dims(a.ndim, dim)
    # 检查维度是否为偶数，否则抛出错误
    torch._check(
        a.shape[dim] % 2 == 0,
        lambda: f"Halving dimension must be even, but dimension {dim} is size {a.shape[dim]}",
    )
    # 使用 torch.tensor_split 对张量 a 进行分割，分割成两部分 b 和 c
    b, c = torch.tensor_split(a, 2, dim)

    # 返回 b 乘以 sigmoid 函数对 c 进行处理的结果
    return b * torch.sigmoid(c)


# 注册函数的装饰器，用于 ATen 函数 pairwise_distance
@register_decomposition(aten.pairwise_distance)
# 输出结果的包装器装饰
@out_wrapper()
# 定义 pairwise_distance 函数
def pairwise_distance(
    x1: TensorLikeType,
    x2: TensorLikeType,
    p: NumberType = 2.0,
    eps: NumberType = 1e-6,
    keepdim=False,
) -> TensorLikeType:
    # 使用 torch.linalg.vector_norm 计算 x1 与 x2 之间的 p 范数
    return torch.linalg.vector_norm(x1 - x2 + eps, ord=p, dim=-1, keepdim=keepdim)


# 注册函数的装饰器，用于 ATen 函数 pdist
@register_decomposition(aten.pdist)
# 输出结果的包装器装饰
@out_wrapper()
# 元素类型提升的包装器装饰
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
)
# 定义 pdist 函数
def pdist(a: TensorLikeType, p: float = 2) -> TensorLikeType:
    # 检查张量 a 的维度是否为 2，否则抛出错误
    torch._check(a.ndim == 2, lambda: f"pdist only supports 2D tensors, got: {a.ndim}D")
    # 检查 p 是否为非负数，否则抛出错误
    torch._check(p >= 0, lambda: "pdist only supports non-negative p values")
    
    # 根据不同的 p 值选择不同的实现方式
    if p == 2:
        # 当 p 等于 2 时，使用高效的实现方法计算 p 范数
        aTa = torch.mm(a, a.T)
        aTa_diag = torch.diag(aTa)
        t = torch.sqrt(torch.clamp(aTa_diag + aTa_diag.unsqueeze(-1) - 2 * aTa, min=0))
    else:
        # 当 p 不等于 2 时，需要创建一个更大的中间张量
        t = torch.linalg.vector_norm(a.unsqueeze(1) - a, ord=p, dim=2)
    
    # 获取 t 张量上三角部分的索引，然后按照索引从 t 张量中选择值并展平
    i = torch.triu_indices(t.shape[0], t.shape[1], offset=1, device=a.device)
    return t.flatten().index_select(0, i[0] * t.shape[0] + i[1])


# 注册函数的装饰器，用于 ATen 函数 pixel_shuffle
@register_decomposition(aten.pixel_shuffle)
# 输出结果的包装器装饰
@out_wrapper()
# 定义 pixel_shuffle 函数
def pixel_shuffle(self: Tensor, upscale_factor: int):
    # 检查输入张量的维度是否至少为 3，否则抛出错误
    torch._check(
        self.dim() >= 3,
        lambda: f"pixel_shuffle expects input to have at least 3 dimensions, but got input with {self.dim} dimension(s)",
    )
    # 获取批量维度
    batch = self.shape[:-3]
    # 计算输出通道数 C_out
    C_out = self.shape[-3] // upscale_factor**2
    # 计算输出高宽尺寸 HW_out
    HW_out = (self.shape[-2] * upscale_factor, self.shape[-1] * upscale_factor)
    # 获取批量维度的长度
    n = len(batch)
    # 定义维度范围
    B_dims = range(n)
    C_dim, r1_dim, r2_dim, H_dim, W_dim = range(n, n + 5)
    # 将输入张量通过视图操作进行重塑，设置输出通道数和上采样因子
    return (
        # 调用视图操作，传入批处理大小、输出通道数、上采样因子和输入张量的高度和宽度
        self.view(
            *batch,             # 批处理维度
            C_out,              # 输出通道数
            upscale_factor,     # 上采样因子
            upscale_factor,     # 上采样因子（重复）
            self.shape[-2],     # 输入张量的高度
            self.shape[-1],     # 输入张量的宽度
        )
        # 执行视图操作后，对结果进行维度置换
        .permute(
            *B_dims,            # 批处理维度
            C_dim,              # 通道维度
            H_dim,              # 高度维度
            r1_dim,             # r1 维度
            W_dim,              # 宽度维度
            r2_dim              # r2 维度
        )
        # 维度置换后，将张量重新整形为指定形状
        .reshape(
            *batch,             # 批处理维度
            C_out,              # 输出通道数
            *HW_out             # 输出高度和宽度
        )
        # 使用建议的内存格式克隆张量，以提高内存使用效率
        .clone(memory_format=utils.suggest_memory_format(self))
    )
# 注册像素重排的分解函数，并应用输出包装器
@注册分解(aten.pixel_unshuffle)
@out_wrapper()
def pixel_unshuffle(self: Tensor, downscale_factor: int):
    # 检查输入张量的维度是否至少为3
    torch._check(
        self.dim() >= 3,
        lambda: f"pixel_unshuffle expects input to have at least 3 dimensions, but got input with {self.dim} dimension(s)",
    )
    # 提取批量维度
    batch = self.shape[:-3]
    # 计算输出通道数
    C_out = self.shape[-3] * downscale_factor**2
    # 计算输出空间维度
    HW_out = (self.shape[-2] // downscale_factor, self.shape[-1] // downscale_factor)
    # 获取批量大小
    n = len(batch)
    # 定义批量维度范围
    B_dims = range(n)
    # 定义通道维度、高度维度及相关重排后的维度
    C_dim, H_dim, r1_dim, W_dim, r2_dim = range(n, n + 5)
    # 返回经过重排和重构的张量副本，内存格式建议由工具函数确定
    return (
        self.view(
            *batch,
            self.shape[-3],
            HW_out[0],
            downscale_factor,
            HW_out[1],
            downscale_factor,
        )
        .permute(*B_dims, C_dim, r1_dim, r2_dim, H_dim, W_dim)
        .reshape(*batch, C_out, *HW_out)
        .clone(memory_format=utils.suggest_memory_format(self))
    )


# 即使这些函数没有 in-place 参数，也需要定义为 in-place 版本
celu_ = _make_inplace(celu)
elu_ = _make_inplace(elu)
mish_ = _make_inplace(mish)
selu_ = _make_inplace(selu)
threshold_ = _make_inplace(threshold)


这些注释详细解释了每行代码的作用，确保理解和维护代码的可靠性和可读性。
```