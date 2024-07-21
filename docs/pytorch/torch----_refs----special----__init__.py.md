# `.\pytorch\torch\_refs\special\__init__.py`

```py
# 导入必要的数学模块和类型提示模块
import math
from typing import Optional, Union

# 导入 PyTorch 相关模块
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs

# 导入特定的张量类型
from torch import Tensor

# 导入特定的分解函数注册模块
from torch._decomp import register_decomposition

# 导入共享的基本数学类型和张量类型
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    Number,
    NumberType,
    TensorLike,
    TensorLikeType,
)

# 导入特定的数学运算封装函数和输出包装器
from torch._prims_common.wrappers import elementwise_type_promotion_wrapper, out_wrapper

# 导入特定的引用模块和函数别名
from torch._refs import (
    _make_alias,
    _make_elementwise_binary_reference,
    _make_elementwise_unary_reference,
)

# 所有公开的函数名列表
__all__ = [
    "bessel_j0",
    "bessel_j1",
    "entr",
    "erfcx",
    "expit",
    "i0e",
    "i1",
    "i1e",
    "log_ndtr",
    "logit",
    "log_softmax",
    "multigammaln",
    "ndtr",
    "ndtri",
    "softmax",
    "spherical_bessel_j0",
    "xlog1py",
    "zeta",
]

# 引用 PyTorch 中的 aten 操作
aten = torch._ops.ops.aten


# 使用 _make_elementwise_unary_reference 装饰器定义 bessel_j0 函数
@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def bessel_j0(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_j0(a)


# 使用 _make_elementwise_unary_reference 装饰器定义 bessel_j1 函数
@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def bessel_j1(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_j1(a)


# 使用 @register_decomposition 注册分解函数 entr，并使用 @out_wrapper 包装输出
@register_decomposition(aten.special_entr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
# 定义 entr 函数，计算熵值
def entr(a: TensorLikeType) -> TensorLikeType:
    return torch.where(
        torch.isnan(a),
        a,
        torch.where(a > 0, -a * torch.log(a), torch.where(a == 0, 0, -torch.inf)),
    )


# 使用 @register_decomposition 注册分解函数 erfcx，并使用 @out_wrapper 包装输出
@register_decomposition(aten.special_erfcx)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
# 定义 erfcx 函数，计算修正的互补误差函数
def erfcx(a: TensorLikeType) -> TensorLikeType:
    return prims.erfcx(a)


# 定义 expit 函数，为 sigmoid 函数的别名
expit = _make_alias(torch.sigmoid, "expit")


# 使用 _make_elementwise_unary_reference 装饰器定义 i0e 函数
@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def i0e(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_i0e(a)


# 使用 _make_elementwise_unary_reference 装饰器定义 i1 函数
@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def i1(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_i1(a)


# 使用 _make_elementwise_unary_reference 装饰器定义 i1e 函数
@_make_elementwise_unary_reference(
    ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
def i1e(a: TensorLikeType) -> TensorLikeType:
    return prims.bessel_i1e(a)


# 使用 @register_decomposition 注册分解函数 log_ndtr，并使用 @out_wrapper 包装输出
@register_decomposition(aten.special_log_ndtr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
# 定义 log_ndtr 函数，计算标准正态分布的对数累积分布函数的导数
def log_ndtr(a: TensorLikeType) -> TensorLikeType:
    # 注意: M_SQRT1_2 是 1 / sqrt(2) 的值
    M_SQRT1_2 = 0.707106781186547524400844362104849039
    t = a * M_SQRT1_2
    # 使用 torch.where 条件函数根据条件选择不同的计算结果返回
    return torch.where(
        # 如果 a 小于 1.0，则执行以下计算
        a < 1.0,
        # 计算第一个分支：计算 torch.log(torch.special.erfcx(-t) / 2) - t * t
        torch.log(torch.special.erfcx(-t) / 2) - t * t,
        # 如果 a 不小于 1.0，则执行以下计算
        # 计算第二个分支：计算 torch.log1p(-torch.erfc(t) / 2)
        torch.log1p(-torch.erfc(t) / 2),
    )
@register_decomposition(aten.logit)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("self",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
# 定义 logit 函数，用于计算逻辑斯蒂函数的反函数
def logit(self: TensorLikeType, eps: Optional[float] = None) -> TensorLikeType:
    if eps is None:
        eps = -1.0
    # 设定下限为 eps
    lo = eps
    # 设定上限为 1 - eps
    hi = 1 - eps
    # 将 self 张量的值限制在 [lo, hi] 范围内
    self = torch.clamp(self, lo, hi)
    # 返回 self 的对数几率（log odds）
    return torch.log(torch.true_divide(self, torch.sub(1, self)))


@register_decomposition(aten.special_xlog1py)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a", "b"),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
# 定义 xlog1py 函数，用于计算 a * log1p(b) 的乘积
def xlog1py(a: Union[TensorLikeType, NumberType], b: Union[TensorLikeType, NumberType]):
    # 检查参数 a 或 b 是否为 TensorLike 类型，如果不是则抛出错误
    torch._check(
        isinstance(a, TensorLike) or isinstance(b, TensorLike),
        lambda: 'Expected either argument a or b to be a Tensor"',
    )

    # 如果 a 是 TensorLike 类型且 b 是数值类型，则将 b 转换为与 a 相同的数据类型和设备的标量张量
    if isinstance(a, TensorLike) and isinstance(b, Number):
        b = refs.scalar_tensor(b, dtype=a.dtype, device=a.device)
    # 如果 b 是 TensorLike 类型且 a 是数值类型，则将 a 转换为与 b 相同的数据类型和设备的标量张量
    elif isinstance(b, TensorLike) and isinstance(a, Number):
        a = refs.scalar_tensor(a, dtype=b.dtype, device=b.device)

    # 断言 a 和 b 都是 TensorLike 类型
    # mypy: expected "Tensor"
    assert isinstance(a, TensorLike)
    assert isinstance(b, TensorLike)
    
    # 计算 rhs = a * log1p(b)，在 a 为 0 时将 rhs 设为 0
    rhs = torch.where(torch.eq(a, 0), 0, torch.mul(a, torch.log1p(b)))
    # 如果 b 中有 NaN 值，则将 rhs 对应位置设为 NaN
    return torch.where(torch.isnan(b), float("nan"), rhs)


@register_decomposition(aten.mvlgamma)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
# 定义 multigammaln 函数，用于计算多变量 gamma 函数的自然对数
def multigammaln(a: TensorLikeType, p: int) -> TensorLikeType:
    # 计算常数 c
    c = 0.25 * p * (p - 1) * math.log(math.pi)
    # 构造等差数列 b
    b = 0.5 * torch.arange(start=(1 - p), end=1, step=1, dtype=a.dtype, device=a.device)
    # 返回 lgamma(a + b) 的和加上常数 c
    return torch.sum(torch.lgamma(a.unsqueeze(-1) + b), dim=-1) + c


@register_decomposition(aten.special_ndtr)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
# 定义 ndtr 函数，用于计算标准正态分布的累积分布函数
def ndtr(a: TensorLikeType) -> TensorLikeType:
    # 定义常数 M_SQRT1_2，即 1 / sqrt(2)
    M_SQRT1_2 = 0.707106781186547524400844362104849039
    # 计算 a * M_SQRT1_2
    a_sqrt_2 = a * M_SQRT1_2
    # 返回 (1 + erf(a_sqrt_2)) * 0.5
    return (1 + torch.erf(a_sqrt_2)) * 0.5


@register_decomposition(aten.special_ndtri)
@out_wrapper()
@elementwise_type_promotion_wrapper(
    type_promoting_args=("a",),
    type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
)
# 定义 ndtri 函数，用于计算标准正态分布的逆累积分布函数
def ndtri(a: TensorLikeType) -> TensorLikeType:
    # 调用 prims 模块中的 ndtri 函数，并返回其结果
    return prims.ndtri(a)


# Forwarding alias: the special variant doesn't support the out kwarg
# CompositeImplicitAutograd - don't register decomp
# 定义 log_softmax 函数，用于计算 softmax 函数的对数
def log_softmax(
    a: TensorLikeType,
    dim: int,
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    # 计算输入张量 a 在指定维度 dim 上的对数 softmax，返回对数 softmax 后的张量
    # type: ignore[call-overload] 表示忽略类型检查的重载调用警告
    return torch.log_softmax(a=a, dim=dim, dtype=dtype)
# 创建一个别名函数，用于调用 torch 库中的 softmax 函数，其中 a 是输入张量，dim 是指定的维度，dtype 是数据类型（可选）
# 这里的 type: ignore[call-overload] 是类型提示，用于忽略函数重载时的类型检查
def softmax(
    a: TensorLikeType,
    dim: int,
    dtype: Optional[torch.dtype] = None,
) -> TensorLikeType:
    return torch.softmax(a=a, dim=dim, dtype=dtype)  # 调用 torch 库中的 softmax 函数


# 创建一个元素级别的一元函数引用，执行 spherical_bessel_j0 的操作
# 使用 ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT 类型提升方式
def spherical_bessel_j0(a: TensorLikeType) -> TensorLikeType:
    return prims.spherical_bessel_j0(a)  # 调用 prims 库中的 spherical_bessel_j0 函数


# TODO: 添加文档字符串
# 创建一个元素级别的二元函数引用，执行 zeta 的操作
# 使用 ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT 类型提升方式
def zeta(a: TensorLikeType, b: TensorLikeType) -> TensorLikeType:
    return prims.zeta(a, b)  # 调用 prims 库中的 zeta 函数
```