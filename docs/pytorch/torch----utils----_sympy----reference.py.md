# `.\pytorch\torch\utils\_sympy\reference.py`

```
# mypy: allow-untyped-defs
# 引入 math 库，提供数学函数和常量
import math

# 引入 operator 库，提供 Python 中的操作符函数
import operator

# 引入 sympy 库，用于符号计算
import sympy

# 引入 torch 库，用于深度学习框架 PyTorch
import torch

# 从 torch.utils._sympy.functions 导入以下函数，用于将 sympy 对象转换为 PyTorch 对象
from torch.utils._sympy.functions import (
    _keep_float,         # 保持浮点数
    FloatPow,            # 浮点数幂运算
    FloatTrueDiv,        # 浮点数真除运算
    FloorDiv,            # 向下取整除法运算
    IntTrueDiv,          # 整数真除运算
    Mod,                 # 取模运算
    OpaqueUnaryFn_exp,   # 指数函数
    OpaqueUnaryFn_log,   # 对数函数
    OpaqueUnaryFn_sqrt,  # 平方根函数
    PowByNatural,        # 自然数幂运算
    RoundDecimal,        # 四舍五入到小数位运算
    RoundToInt,          # 四舍五入到整数运算
    ToFloat,             # 转换为浮点数
    TruncToInt,          # 截断为整数运算
)

# 对操作符的 sympy 解释。有时也可以适用于普通的 int/float，但如果进行某些操作，最终会得到一个 sympy.Basic 对象。
# 如果需要 Python/FX 可追踪的解释，请检查 PythonReferenceAnalysis。
class ReferenceAnalysis:
    @staticmethod
    # 返回一个 sympy.Basic 对象，表示常量 c
    def constant(c, dtype):
        return sympy.sympify(c)

    @staticmethod
    # 逻辑或运算
    def or_(a, b):
        return a | b

    @staticmethod
    # 逻辑与运算
    def and_(a, b):
        return a & b

    @staticmethod
    # 判断是否相等，如果其中一个参数是 sympy.Expr，则返回 sympy.Eq 对象
    def eq(a, b):
        if isinstance(a, sympy.Expr) or isinstance(b, sympy.Expr):
            return sympy.Eq(a, b)
        return a == b

    @classmethod
    # 判断是否不相等，调用 eq 方法判断相等后取反
    def ne(cls, a, b):
        return cls.not_(cls.eq(a, b))

    @staticmethod
    # 小于比较运算
    def lt(a, b):
        return a < b

    @staticmethod
    # 大于比较运算
    def gt(a, b):
        return a > b

    @staticmethod
    # 小于等于比较运算
    def le(a, b):
        return a <= b

    @staticmethod
    # 大于等于比较运算
    def ge(a, b):
        return a >= b

    @staticmethod
    # 逻辑非运算
    def not_(a):
        assert not isinstance(a, bool)
        return ~a

    @staticmethod
    # 返回浮点数 x 的倒数
    def reciprocal(x):
        return FloatTrueDiv(1.0, x)

    @staticmethod
    # 返回 x 的平方
    def square(x):
        return PowByNatural(x, 2)

    @staticmethod
    # 截断为整数运算
    def trunc_to_int(x, dtype):
        return TruncToInt(x)

    @staticmethod
    # 向上取整到整数运算
    def ceil_to_int(x, dtype):
        return sympy.ceiling(x)

    @staticmethod
    # 向下取整到整数运算
    def floor_to_int(x, dtype):
        return sympy.floor(x)

    @staticmethod
    # 返回 x 的向下取整结果
    def floor(x):
        return _keep_float(sympy.floor)(x)

    @staticmethod
    # 返回 x 的向上取整结果
    def ceil(x):
        return _keep_float(sympy.ceiling)(x)

    @staticmethod
    # 根据 dtype 类型将 x 转换为对应的 PyTorch 浮点数类型
    def to_dtype(x, dtype):
        if dtype == torch.float64:
            return ToFloat(x)
        raise NotImplementedError(f"to_dtype {dtype} NYI")

    @staticmethod
    # 返回 x 对 y 取模的结果
    def mod(x, y):
        return Mod(x, y)

    @staticmethod
    # 返回 x 的绝对值
    def abs(x):
        return abs(x)

    @staticmethod
    # 返回 x 的相反数
    def neg(x):
        return -x

    @staticmethod
    # 返回浮点数 a 除以 b 的结果
    def truediv(a, b):
        return FloatTrueDiv(a, b)

    @staticmethod
    # 返回整数 a 除以 b 的结果
    def int_truediv(a, b):
        return IntTrueDiv(a, b)

    @staticmethod
    # 返回 a 除以 b 的向下取整结果
    def floordiv(a, b):
        return FloorDiv(a, b)

    @staticmethod
    # 截断除法运算，未实现
    def truncdiv(a, b):
        raise NotImplementedError("TODO: truncdiv")

    @staticmethod
    # 返回 a 和 b 的加法结果
    def add(a, b):
        return _keep_float(operator.add)(a, b)

    @staticmethod
    # 返回 a 和 b 的乘法结果
    def mul(a, b):
        return _keep_float(operator.mul)(a, b)

    @staticmethod
    # 返回 a 和 b 的减法结果
    def sub(a, b):
        return _keep_float(operator.sub)(a, b)
    # 定义一个静态方法，计算 e 的 x 次方
    @staticmethod
    def exp(x):
        return OpaqueUnaryFn_exp(x)

    # 定义一个静态方法，计算 x 的自然对数
    @staticmethod
    def log(x):
        return OpaqueUnaryFn_log(x)

    # 定义一个静态方法，计算 x 的平方根
    @staticmethod
    def sqrt(x):
        return OpaqueUnaryFn_sqrt(x)

    # 定义一个静态方法，计算 a 的 b 次方，保持浮点数形式
    @staticmethod
    def pow(a, b):
        return _keep_float(FloatPow)(a, b)

    # 定义一个静态方法，使用自然对数计算 a 的 b 次方
    @staticmethod
    def pow_by_natural(a, b):
        return PowByNatural(a, b)

    # 定义一个静态方法，返回 a 和 b 中的较小值
    @staticmethod
    def minimum(a, b):
        return sympy.Min(a, b)

    # 定义一个静态方法，返回 a 和 b 中的较大值
    @staticmethod
    def maximum(a, b):
        return sympy.Max(a, b)

    # 定义一个静态方法，将浮点数 a 四舍五入为整数，返回整数类型
    @staticmethod
    def round_to_int(a, dtype):
        return RoundToInt(a)

    # 定义一个静态方法，将浮点数 a 四舍五入到指定小数位数 b
    @staticmethod
    def round_decimal(a, b):
        return RoundDecimal(a, b)
# PythonReferenceAnalysis 类继承自 ReferenceAnalysis，用于处理 Python 类型的参考分析。
# 不同于 ReferenceAnalysis，它不使用 sympy 化简，而是直接处理原始的 Python 类型，并且支持 FX 追踪。
# 继承关系在这里仅用于代码共享，未来可能会考虑将其拆分为 BaseReferenceAnalysis。

class PythonReferenceAnalysis(ReferenceAnalysis):

    # 将常数 c 转换为指定的 dtype 类型
    @staticmethod
    def constant(c, dtype):
        if dtype is torch.int64:
            return int(c)
        elif dtype is torch.double:
            return float(c)
        elif dtype is torch.bool:
            return bool(c)
        else:
            raise AssertionError(f"unrecognized dtype {dtype}")

    # 对 a 进行逻辑非操作
    @staticmethod
    def not_(a):
        return torch.sym_not(a)

    # 返回 a 除以 b 的整数部分
    @staticmethod
    def floordiv(a, b):
        return a // b

    # 返回 x 除以 y 的余数
    @staticmethod
    def mod(x, y):
        return x % y

    # 返回 a 除以 b 的结果
    @staticmethod
    def truncdiv(a, b):
        return a / b

    # 将 x 转换为指定的 dtype 类型，目前仅支持 torch.float64 类型
    @staticmethod
    def to_dtype(x, dtype):
        if dtype == torch.float64:
            return float(x)
        raise NotImplementedError(f"to_dtype {dtype} NYI")

    # 抛出异常，不支持 exp 函数用于形状 sympy 表达式
    @staticmethod
    def exp(x):
        raise AssertionError("exp is not valid shape sympy expr")

    # 抛出异常，不支持 log 函数用于形状 sympy 表达式
    @staticmethod
    def log(x):
        raise AssertionError("log is not valid shape sympy expr")

    # 返回 x 的平方根，使用 torch._sym_sqrt 进行计算
    @staticmethod
    def sqrt(x):
        return torch._sym_sqrt(x)  # type: ignore[attr-defined]

    # 返回 a 和 b 中的较小值
    @staticmethod
    def minimum(a, b):
        return torch.sym_min(a, b)

    # 返回 a 和 b 中的较大值
    @staticmethod
    def maximum(a, b):
        return torch.sym_max(a, b)

    # 将 x 向下取整后转换为整数
    @staticmethod
    def floor_to_int(x, dtype):
        return math.floor(x)

    # 将 x 向上取整后转换为整数
    @staticmethod
    def ceil_to_int(x, dtype):
        return math.ceil(x)

    # 返回 x 向下取整后的结果
    @staticmethod
    def floor(x):
        return float(math.floor(x))

    # 返回 x 向上取整后的结果
    @staticmethod
    def ceil(x):
        return float(math.ceil(x))

    # 返回 a 除以 b 的结果
    @staticmethod
    def truediv(a, b):
        return a / b

    # 返回 a 的 b 次方
    @staticmethod
    def pow(a, b):
        return a**b

    # 返回 a 的 b 次方，假设这里不会出现溢出问题
    @staticmethod
    def pow_by_natural(a, b):
        # 希望这里不需要 safe_pow，特别是在 VR 低/高范围中，溢出可能性较小
        return a**b

    # 将 a 四舍五入后转换为整数
    @staticmethod
    def round_to_int(a, dtype):
        return round(a)

    # 将 a 四舍五入保留 b 位小数
    @staticmethod
    def round_decimal(a, b):
        return round(a, ndigits=b)
```