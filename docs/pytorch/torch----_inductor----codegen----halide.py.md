# `.\pytorch\torch\_inductor\codegen\halide.py`

```py
# 引入mypy类型检查中允许未标记类型的定义
# from __future__ import annotations

# 引入必要的模块和库
import dataclasses               # 数据类装饰器，用于简化数据对象的创建和比较
import functools                 # 函数工具，提供了用于操作函数和调用的工具
import itertools                 # 提供了用于创建和操作迭代器的函数
import logging                   # Python标准库中用于记录日志的模块
import re                        # 正则表达式操作模块
from collections import defaultdict  # 默认字典，是字典的子类，提供了一个可以指定默认值的字典
from math import inf             # 无穷大的数学常量
from typing import (             # 引入类型提示，用于声明函数参数和返回值的类型
    Any,                         # 任意类型
    Callable,                    # 可调用对象
    Dict,                        # 字典类型
    List,                        # 列表类型
    Optional,                    # 可选类型
    Sequence,                    # 序列类型
    Set,                         # 集合类型
    Tuple,                       # 元组类型
    TYPE_CHECKING,               # 类型检查标志
    Union                        # 联合类型
)

import sympy                     # Python符号计算库

import torch                     # PyTorch深度学习库
import torch._logging            # PyTorch的内部日志模块
from ..._prims_common import is_integer_dtype  # 导入判断是否为整数类型的函数
from ...utils._sympy.functions import FloorDiv, ModularIndexing  # 导入符号计算相关的函数
from ...utils._sympy.symbol import symbol_is_type, SymT  # 导入符号计算相关的函数和符号类型
from ...utils._sympy.value_ranges import ValueRanges  # 导入值范围处理模块
from .. import config, ir         # 导入相关模块
from ..codecache import HalideCodeCache  # 导入Halide代码缓存模块
from ..metrics import is_metric_table_enabled, log_kernel_metadata  # 导入性能指标相关函数

from ..runtime.hints import HalideInputSpec, HalideMeta, ReductionHint  # 导入Halide运行时相关的类型提示
from ..utils import (             # 导入各种实用函数
    get_bounds_index_expr,        # 获取索引表达式边界
    get_kernel_metadata,          # 获取内核元数据
    parallel_num_threads,         # 并行线程数
    sympy_index_symbol,           # sympy符号计算
    sympy_subs                   # sympy符号替换
)
from ..virtualized import _ops as ops, OpsHandler, V  # 导入虚拟化操作相关模块和类
from .common import (             # 导入共用的类和函数
    BackendFeature,               # 后端特性
    CSEVariable,                  # 公共子表达式变量
    DeferredLine,                 # 延迟行
    IndentedBuffer,               # 缩进缓冲
    OpOverrides,                  # 操作重载
    PythonPrinter,                # Python打印器
    SizeArg,                      # 尺寸参数
    TensorArg                     # 张量参数
)
from .cpp import DTYPE_TO_CPP    # 导入C++数据类型到PyTorch数据类型的映射
from .cpp_utils import cexpr     # 导入C++表达式计算模块
from .simd import constant_repr, SIMDKernel, SIMDScheduling  # 导入SIMD相关模块和函数

# 如果类型检查开启，则导入更多类型相关的模块
if TYPE_CHECKING:
    from ..ops_handler import ReductionType, StoreMode  # 导入操作处理相关的类型定义

# 设置日志记录器
log = logging.getLogger(__name__)


def halide_constant(val):
    # 检查值是否为整数且超出32位整数的范围
    if isinstance(val, int) and not (-2147483648 <= val <= 2147483647):
        info = torch.iinfo(torch.int64)  # 获取torch中64位整数的信息
        # 如果值等于最小64位整数，则返回Halide表示的最小整数值
        if val == info.min:
            return "hl.Int(64).min()"
        # 如果值等于最大64位整数，则返回Halide表示的最大整数值
        if val == info.max:
            return "hl.Int(64).max()"
        return f"hl.i64({val!r})"  # 返回64位整数类型的Halide表示
    # 如果值为浮点数，则返回Halide表示的浮点数值
    if isinstance(val, float):
        return f"hl.f64({constant_repr(val)})"  # 返回64位浮点数类型的Halide表示
    return repr(val)  # 返回值的字符串表示


class Unsupported(RuntimeError):
    # 自定义异常类，用于表示Halide后端不支持的情况
    def __init__(self, thing):
        super().__init__(f"halide backend does not support: {thing}")  # 异常初始化方法


class HalidePrinter(PythonPrinter):
    # 继承自PythonPrinter类的Halide打印器类

    @staticmethod
    def cast_index(expr):
        # 对表达式进行索引类型的转换
        return f"hl.cast({V.kernel.index_dtype}, {expr})"

    @staticmethod
    def cast_float(expr):
        # 对表达式进行浮点数类型的转换
        return f"hl.cast(hl.Float(32), {expr})"

    def _print_Float(self, expr):
        # 打印浮点数表达式
        return f"hl.f32({expr})"

    def _print_floor(self, expr):
        # 打印向下取整表达式
        assert len(expr.args) == 1
        return self.cast_index(f"hl.floor({self._print(expr.args[0])})")

    def _print_Trunc(self, expr):
        # 打印截断表达式
        assert len(expr.args) == 1
        return self.cast_index(f"hl.trunc({self._print(expr.args[0])})")

    def _print_ceiling(self, expr):
        # 打印向上取整表达式
        assert len(expr.args) == 1
        return self.cast_index(f"hl.ceil({self._print(expr.args[0])})")

    def _helper_sqrt(self, expr):
        # 打印求平方根的辅助方法
        return f"hl.sqrt({self.cast_float(self._print(expr))})"

    def _print_Where(self, expr):
        # 打印Where条件表达式
        c = self.doprint(expr.args[0])
        p = self.doprint(expr.args[1])
        q = self.doprint(expr.args[2])
        return f"hl.select({c}, {p}, {q})"  # 返回Halide的条件选择函数调用
    # 打印 sympy.Min 函数的字符串表示
    def _print_Min(self, expr):
        # 如果参数列表长度为1，直接打印该参数的字符串表示
        if len(expr.args) == 1:
            return self._print(expr.args[0])

        # 计算参数列表的中间位置
        mid = len(expr.args) // 2
        # 递归打印 Min 函数左半部分的字符串表示
        a = self._print(sympy.Min(*expr.args[:mid]))
        # 递归打印 Min 函数右半部分的字符串表示
        b = self._print(sympy.Min(*expr.args[mid:]))
        # 返回组合后的字符串表示
        return f"hl.min({a}, {b})"

    # 打印 sympy.Max 函数的字符串表示
    def _print_Max(self, expr):
        # 如果参数列表长度为1，直接打印该参数的字符串表示
        if len(expr.args) == 1:
            return self._print(expr.args[0])

        # 计算参数列表的中间位置
        mid = len(expr.args) // 2
        # 递归打印 Max 函数左半部分的字符串表示
        a = self._print(sympy.Max(*expr.args[:mid]))
        # 递归打印 Max 函数右半部分的字符串表示
        b = self._print(sympy.Max(*expr.args[mid:]))
        # 返回组合后的字符串表示
        return f"hl.max({a}, {b})"

    # 打印 sympy.Abs 函数的字符串表示
    def _print_Abs(self, expr):
        # 断言参数列表长度为1
        assert len(expr.args) == 1
        # 打印 Abs 函数的字符串表示，并进行索引转换
        return self.cast_index(f"hl.abs({self._print(expr.args[0])})")

    # 打印 OpaqueUnaryFn_cos 函数的字符串表示
    def _print_OpaqueUnaryFn_cos(self, expr):
        # 断言参数列表长度为1
        assert len(expr.args) == 1
        # 返回 OpaqueUnaryFn_cos 函数的字符串表示
        return f"hl.cos(({self._print(expr.args[0])})"

    # 打印 OpaqueUnaryFn_cosh 函数的字符串表示
    def _print_OpaqueUnaryFn_cosh(self, expr):
        # 断言参数列表长度为1
        assert len(expr.args) == 1
        # 返回 OpaqueUnaryFn_cosh 函数的字符串表示
        return f"hl.cosh(({self._print(expr.args[0])})"

    # 打印 OpaqueUnaryFn_acos 函数的字符串表示
    def _print_OpaqueUnaryFn_acos(self, expr):
        # 断言参数列表长度为1
        assert len(expr.args) == 1
        # 返回 OpaqueUnaryFn_acos 函数的字符串表示
        return f"hl.acos(({self._print(expr.args[0])})"

    # 打印 OpaqueUnaryFn_sin 函数的字符串表示
    def _print_OpaqueUnaryFn_sin(self, expr):
        # 断言参数列表长度为1
        assert len(expr.args) == 1
        # 返回 OpaqueUnaryFn_sin 函数的字符串表示
        return f"hl.sin(({self._print(expr.args[0])})"

    # 打印 OpaqueUnaryFn_sinh 函数的字符串表示
    def _print_OpaqueUnaryFn_sinh(self, expr):
        # 断言参数列表长度为1
        assert len(expr.args) == 1
        # 返回 OpaqueUnaryFn_sinh 函数的字符串表示
        return f"hl.sinh(({self._print(expr.args[0])})"

    # 打印 OpaqueUnaryFn_asin 函数的字符串表示
    def _print_OpaqueUnaryFn_asin(self, expr):
        # 断言参数列表长度为1
        assert len(expr.args) == 1
        # 返回 OpaqueUnaryFn_asin 函数的字符串表示
        return f"hl.asin(({self._print(expr.args[0])})"

    # 打印 OpaqueUnaryFn_tan 函数的字符串表示
    def _print_OpaqueUnaryFn_tan(self, expr):
        # 断言参数列表长度为1
        assert len(expr.args) == 1
        # 返回 OpaqueUnaryFn_tan 函数的字符串表示
        return f"hl.tan(({self._print(expr.args[0])})"

    # 打印 OpaqueUnaryFn_tanh 函数的字符串表示
    def _print_OpaqueUnaryFn_tanh(self, expr):
        # 断言参数列表长度为1
        assert len(expr.args) == 1
        # 返回 OpaqueUnaryFn_tanh 函数的字符串表示
        return f"hl.tanh(({self._print(expr.args[0])})"

    # 打印 OpaqueUnaryFn_atan 函数的字符串表示
    def _print_OpaqueUnaryFn_atan(self, expr):
        # 断言参数列表长度为1
        assert len(expr.args) == 1
        # 返回 OpaqueUnaryFn_atan 函数的字符串表示
        return f"hl.atan(({self._print(expr.args[0])})"

    # 打印 FloorDiv 函数的字符串表示
    def _print_FloorDiv(self, expr):
        # 如果表达式是整数类型，则调用父类的打印函数
        if expr.is_integer:
            return super()._print_FloorDiv(expr)

        # 获取表达式的参数
        x, div = expr.args
        # 将 x 和 div 转换为浮点数并进行打印
        x = self.cast_float(self.paren(self.doprint(x)))
        div = self.cast_float(self.paren(self.doprint(div)))
        # 返回组合后的字符串表示
        return self.cast_index(f"hl.floor({x} / {div})")

    # 打印 Round 函数的字符串表示
    def _print_Round(self, expr):
        # 断言参数列表长度为1
        assert len(expr.args) == 1
        # 返回 Round 函数的字符串表示，并进行索引转换
        return self.cast_index(f"hl.round({self._print(expr.args[0])})")

    # _print_RoundToInt 函数与 _print_Round 函数相同
    _print_RoundToInt = _print_Round

    # 打印 IntTrueDiv 函数的字符串表示
    def _print_IntTrueDiv(self, expr):
        # 获取 IntTrueDiv 函数的参数
        a, b = expr.args
        # 强制将 a 转换为浮点数
        return f"({a}) / ({b}+hl.f32(0))"

    # 打印 RoundDecimal 函数的字符串表示
    def _print_RoundDecimal(self, expr):
        # 获取 RoundDecimal 函数的参数
        val, n = expr.args
        # 打印 val 的字符串表示
        val = self._print(val)
        # 将 n 转换为整数
        n = int(n)
        # 返回组合后的字符串表示
        return f"hl.f32({10.**(-n)!r})*hl.round(({val})*hl.f32({10.**n!r}))"
# Halide 打印器用于生成 Halide 表达式
texpr = HalidePrinter().doprint
# Python 打印器用于生成 Python 表达式
pexpr = PythonPrinter().doprint

# 定义了一个字典，将 Torch 张量类型映射到对应的 Halide 类型字符串
_halide_type = {
    torch.bool: "hl.Bool()",
    torch.bfloat16: "hl.BFloat(16)",
    torch.float16: "hl.Float(16)",
    torch.float32: "hl.Float(32)",
    torch.float64: "hl.Float(64)",
    torch.int8: "hl.Int(8)",
    torch.int16: "hl.Int(16)",
    torch.int32: "hl.Int(32)",
    torch.int64: "hl.Int(64)",
    torch.uint8: "hl.UInt(8)",
    torch.uint16: "hl.UInt(16)",
    torch.uint32: "hl.UInt(32)",
    torch.uint64: "hl.UInt(64)",
}

# 根据给定的 Torch 张量类型返回对应的 Halide 类型字符串
def halide_type(dtype):
    return _halide_type[dtype]

# 根据给定的 Torch 张量类型返回对应的 Halide 访问类型字符串
def halide_acc_type(dtype):
    # 如果是有符号整数类型且不是 int64，则转换为 int32
    if is_integer_dtype(dtype) and dtype.is_signed and dtype != torch.int64:
        dtype = torch.int32
    # 如果是 float16 或 bfloat16 类型，则转换为 float32
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    return halide_type(dtype)

# HalideOverrides 类继承自 OpOverrides 类，用于覆盖操作的行为
class HalideOverrides(OpOverrides):
    # 将张量 x 转换为指定的 dtype 类型的 Halide 表达式
    @staticmethod
    def to_dtype(x, dtype: torch.dtype, src_dtype: Optional[torch.dtype] = None):
        if dtype == torch.bool:
            return f"({x} != 0)"
        return f"hl.cast({halide_type(dtype)}, {x})"

    # 将张量 x 从 src_dtype 类型转换为 dtype 类型的位转换表达式
    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype, src_dtype: torch.dtype):
        # 如果原始类型是 float16 或 bfloat16，则将 x 转换为对应类型的 Halide 表达式
        if src_dtype in (torch.float16, torch.bfloat16):
            x = f"hl.cast({halide_type(src_dtype)}, {x})"  # body compute is upcast to fp32
        line = f"hl.reinterpret({halide_type(dtype)}, {x})"
        # 如果目标类型是 float16 或 bfloat16，则将表达式再转换为 float32 类型的表达式
        if dtype in (torch.float16, torch.bfloat16):
            line = f"hl.cast(hl.Float(32), {line})"
        return line

    # 返回常量值 value 的 dtype 类型的 Halide 表达式
    @classmethod
    def constant(cls, value, dtype):
        return cls.to_dtype(halide_constant(value), dtype)

    # 返回张量 x 的绝对值的 Halide 表达式
    @staticmethod
    def abs(x):
        return f"hl.abs({x})"

    # 返回张量 x 的指数函数的 Halide 表达式
    @staticmethod
    def exp(x):
        return f"hl.fast_exp(hl.cast(hl.Float(32), {x})) if {x.name}.type().bits() <= 32 else hl.exp({x})"

    # 返回 libdevice 中的指数函数的 Halide 表达式
    @staticmethod
    def libdevice_exp(x):
        return f"hl.exp({x})"  # higher precision that ops.exp

    # 返回张量 x 的平方根的 Halide 表达式
    @staticmethod
    def sqrt(x):
        return f"hl.sqrt({x})"

    # 返回张量 a 和 b 中的较小值的 Halide 表达式，处理 NaN 值
    @staticmethod
    def minimum(a, b):
        # 原实现存在 NaN 处理错误，这里修改为正确处理方式
        b = f"hl.cast({a.name}.type(), {b})"
        return f"hl.select(({a}<{b})|hl.is_nan({a}), {a}, {b}) if {a.name}.type().is_float() else hl.min({a}, {b})"

    # 返回张量 a 和 b 中的较大值的 Halide 表达式，处理 NaN 值
    @staticmethod
    def maximum(a, b):
        # 原实现存在 NaN 处理错误，这里修改为正确处理方式
        b = f"hl.cast({a.name}.type(), {b})"
        return f"hl.select(({a}>{b})|hl.is_nan({a}), {a}, {b}) if {a.name}.type().is_float() else hl.max({a}, {b})"

    # 返回根据条件 a 选择张量 b 或 c 的 Halide 表达式
    @staticmethod
    def where(a, b, c):
        return f"hl.select({a}, {b}, hl.cast({b.name}.type(), {c}))"

    # 返回张量 x 的余弦函数的 Halide 表达式
    @staticmethod
    def cos(x):
        return f"hl.cos({x})"

    # 返回张量 x 的正弦函数的 Halide 表达式
    @staticmethod
    def sin(x):
        return f"hl.sin({x})"

    # 抛出不支持的异常，因为不支持 lgamma 函数
    @staticmethod
    def lgamma(x):
        raise Unsupported("lgamma")

    # 返回张量 x 的误差函数的 Halide 表达式
    @staticmethod
    def erf(x):
        return f"hl.erf({x})"

    # 返回张量 x 的双曲余弦函数的 Halide 表达式
    @staticmethod
    def cosh(x):
        return f"hl.cosh({x})"
    def sinh(x):
        return f"hl.sinh({x})"
    # 返回双曲正弦函数的字符串表示，使用 Halide 的 hl.sinh 函数

    @staticmethod
    def acos(x):
        return f"hl.acos({x})"
    # 返回反余弦函数的字符串表示，使用 Halide 的 hl.acos 函数

    @staticmethod
    def acosh(x):
        return f"hl.acosh({x})"
    # 返回反双曲余弦函数的字符串表示，使用 Halide 的 hl.acosh 函数

    @staticmethod
    def asin(x):
        return f"hl.asin({x})"
    # 返回反正弦函数的字符串表示，使用 Halide 的 hl.asin 函数

    @staticmethod
    def asinh(x):
        return f"hl.asinh({x})"
    # 返回反双曲正弦函数的字符串表示，使用 Halide 的 hl.asinh 函数

    @staticmethod
    def atan2(x, y):
        return f"hl.atan2({x}, {y})"
    # 返回二参数反正切函数的字符串表示，使用 Halide 的 hl.atan2 函数

    @staticmethod
    def atan(x):
        return f"hl.atan({x})"
    # 返回反正切函数的字符串表示，使用 Halide 的 hl.atan 函数

    @staticmethod
    def atanh(x):
        return f"hl.atanh({x})"
    # 返回反双曲正切函数的字符串表示，使用 Halide 的 hl.atanh 函数

    @staticmethod
    def copysign(x, y):
        raise Unsupported("copysign")
    # 抛出不支持异常，因为 copysign 函数未实现

    @staticmethod
    def erfinv(x):
        raise Unsupported("erfinv")
    # 抛出不支持异常，因为 erfinv 函数未实现

    @staticmethod
    def hypot(x, y):
        return f"hl.hypot({x}, {y})"
    # 返回欧几里德范数函数的字符串表示，使用 Halide 的 hl.hypot 函数

    @staticmethod
    def nextafter(x, y):
        raise Unsupported("nextafter")
    # 抛出不支持异常，因为 nextafter 函数未实现

    @staticmethod
    def logical_and(a, b):
        return f"{a} & {b}"
    # 返回逻辑与运算的字符串表示，按位与操作

    @staticmethod
    def logical_not(a):
        return f"{a} == 0"
    # 返回逻辑非运算的字符串表示，判断是否等于零

    @staticmethod
    def logical_or(a, b):
        return f"{a} | {b}"
    # 返回逻辑或运算的字符串表示，按位或操作

    @staticmethod
    def logical_xor(a, b):
        return f"({a} ^ {b})"
    # 返回逻辑异或运算的字符串表示，按位异或操作

    @staticmethod
    def bitwise_and(a, b):
        return f"{a} & {b}"
    # 返回按位与运算的字符串表示

    @staticmethod
    def bitwise_not(a):
        return f"~{a}"
    # 返回按位取反运算的字符串表示

    @staticmethod
    def bitwise_or(a, b):
        return f"{a} | {b}"
    # 返回按位或运算的字符串表示

    @staticmethod
    def bitwise_xor(a, b):
        return f"{a} ^ {b}"
    # 返回按位异或运算的字符串表示

    @staticmethod
    def bitwise_left_shift(a, b):
        return f"{a} << {b}"
    # 返回按位左移运算的字符串表示

    @staticmethod
    def bitwise_right_shift(a, b):
        return f"{a} >> {b}"
    # 返回按位右移运算的字符串表示

    @staticmethod
    def rand(seed, offset):
        raise Unsupported("rand")
    # 抛出不支持异常，因为 rand 函数未实现

    @staticmethod
    def randn(seed, offset):
        raise Unsupported("rand")
    # 抛出不支持异常，因为 randn 函数未实现

    @staticmethod
    def randint64(seed, offset, low, high):
        raise Unsupported("rand")
    # 抛出不支持异常，因为 randint64 函数未实现

    @staticmethod
    def load_seed(name, offset):
        raise Unsupported("rand")
    # 抛出不支持异常，因为 load_seed 函数未实现

    @staticmethod
    def rsqrt(x):
        # return f"hl.fast_inverse_sqrt({x})"  <== accuracy issues
        return f"1./hl.sqrt({x})"
    # 返回反平方根函数的字符串表示，使用 1.0 除以 Halide 的 hl.sqrt 函数

    @staticmethod
    def tan(x):
        return f"hl.tan({x})"
    # 返回正切函数的字符串表示，使用 Halide 的 hl.tan 函数

    @staticmethod
    def tanh(x):
        return f"hl.tanh({x})"
    # 返回双曲正切函数的字符串表示，使用 Halide 的 hl.tanh 函数

    @staticmethod
    def signbit(x):
        return f"(hl.reinterpret(hl.UInt(32), hl.cast(hl.Float(32), {x})) >> 31) != 0"
    # 返回符号位判断函数的字符串表示，使用 Halide 的 hl.reinterpret 和 hl.cast 函数

    @staticmethod
    def fmod(a, b):
        # TODO(jansel): find a better way to do this, builtin % has wrong sign
        return f"{a} - hl.trunc({a}/{b})*{b}"
    # 返回取模运算的字符串表示，使用 Halide 的 hl.trunc 函数处理除法结果

    @staticmethod
    def pow(a, b):
        return f"hl.pow({a}, {b})"
    # 返回幂函数的字符串表示，使用 Halide 的 hl.pow 函数

    @staticmethod
    def log(x):
        return f"hl.log({x})"
    # 返回对数函数的字符串表示，使用 Halide 的 hl.log 函数

    @staticmethod
    def isinf(x):
        # workaround https://github.com/halide/Halide/issues/8309
        return f"hl.is_inf(hl.cast(hl.Float(32), {x}))"
    # 返回判断是否为无穷大的字符串表示，使用 Halide 的 hl.is_inf 和 hl.cast 函数
    def isnan(x):
        # 返回一个字符串，表示将 x 转换为 Float(32) 类型后，检查是否为 NaN
        return f"hl.is_nan(hl.cast(hl.Float(32), {x}))"

    @staticmethod
    def round(x):
        # 返回将 x 四舍五入的字符串表示
        return f"hl.round({x})"

    @staticmethod
    def floor(x):
        # 返回将 x 向下取整的字符串表示
        return f"hl.floor({x})"

    @staticmethod
    def int_truediv(a, b):
        # 返回 a 除以 b，确保 b 不为零，以避免浮点数除以零异常
        return f"({a}) / ({b} + hl.f32(0))"

    @staticmethod
    def floordiv(a, b):
        # 返回将 a 除以 b 并向下取整的字符串表示
        # 注意：目前未找到使用 triton.py 中 select-based 技巧的更好方法
        return (
            f"hl.floor(hl.cast(hl.Float(max(32, {a.name}.type().bits())), {a}) / {b})"
        )

    @classmethod
    def sign(cls, x):
        # 返回将 x 转换为整数后，计算其符号的字符串表示
        left = ops.to_dtype(ops.lt("0", x), torch.int8)
        right = ops.to_dtype(ops.lt(x, "0"), torch.int8)
        sub = ops.sub(left, right)
        return f"hl.cast({x.name}.type(), {sub})"

    @staticmethod
    def trunc(x):
        # 返回将 x 截断为整数的字符串表示
        return f"hl.trunc({x})"

    @staticmethod
    def truncdiv(a, b):
        # 返回将 a 除以 b 并截断为整数的字符串表示
        # 注意：这会导致浮点异常，见 test_div_zero_dim_cpu
        return (
            f"hl.trunc(hl.cast(hl.Float(max(32, {a.name}.type().bits())), {a}) / {b})"
        )

    @staticmethod
    def ceil(x):
        # 返回将 x 向上取整的字符串表示
        return f"hl.ceil({x})"

    @staticmethod
    def relu(x):
        # 返回对 x 进行 ReLU 操作的字符串表示
        return f"hl.max({x}, 0)"

    @classmethod
    def index_expr(cls, expr, dtype):
        # 准备索引表达式，返回相应的字符串表示
        index = V.kernel.prepare_indexing(expr)
        var = V.kernel.genfunc(
            V.kernel.index_to_str(index),
            V.kernel.used_dims_from_index(index),
            bounds=get_bounds_index_expr(expr),
        )
        if dtype not in {torch.int32, torch.int64}:
            return ops.to_dtype(var, dtype)
        return var

    @classmethod
    def indirect_indexing(cls, index_var, size, check=True):
        # 执行间接索引，将索引变量转换为 int32 类型并进行边界检查，返回符号表示
        # 注意：Halide 仅支持 32 位索引，可能会在溢出时报错
        index_var = ops.to_dtype(index_var, torch.int32)
        index_var = ops.halide_clamp(index_var, size, check)
        index_var.indirect_indexing_size = size
        return sympy_index_symbol(str(index_var))

    @classmethod
    def halide_clamp(cls, value, size, check):
        # 返回将 value 约束在 [0, size-1] 范围内的字符串表示
        end = V.kernel.kexpr(V.kernel.rename_indexing(size) - 1)
        if not isinstance(size, (int, sympy.Integer)):
            end = f"hl.cast({value.name}.type(), {end})"
        return f"hl.clamp({value}, 0, {end})"
    # 定义函数 `masked`，接受三个参数：mask、body 和 other
    def masked(mask, body, other):
        # 使用 V.kernel.mask_loads 方法加载 mask 和 other，并创建新的 mask 上下文 new_mask
        with V.kernel.mask_loads(mask, other) as new_mask:
            # 执行 body 函数，获取结果
            result = body()

        # 如果 result 的 bounds 是布尔类型
        if result.bounds.is_bool:
            # 将 other 转换为布尔类型
            other = bool(other)

        # 从 result 中获取 dtype，以防止意外的类型提升
        other = V.kernel.genfunc(
            f"hl.cast({result.name}.type(), {halide_constant(other)})",
            [],  # 空列表作为 genfunc 的第二个参数
            bounds=ValueRanges.wrap(other),  # 使用 other 创建 ValueRanges 的 bounds
        )
        # TODO(jansel): look into removing the where in the same places triton does
        # 使用 ops.where 函数根据 new_mask 条件返回 result 或者 other
        return ops.where(new_mask, result, other)
# 使用 mypy 检查 HalideOverrides 协议是否正确实现
def _typecheck_HalideOverrides(h: HalideOverrides) -> OpsHandler[str]:
    # 返回参数 h，确认其符合 OpsHandler[str] 类型
    return h


class HalideCSEVariable(CSEVariable):
    # 正则表达式，匹配未定义的变量名，形如 tmp\d+[\?]
    undefined_re = re.compile(r"\b(tmp\d+)\[\?\]")

    def __init__(self, name, bounds: ValueRanges[Any]):
        # 调用父类构造函数初始化变量名和边界
        super().__init__(name, bounds)
        # 初始化 used_dims 属性为 None
        self.used_dims: Optional[List[sympy.Symbol]] = None

    def update_on_args(self, name, args, kwargs):
        # 用于跟新 used_dims 属性，根据传入的参数 args 和 kwargs
        used = set(self.used_dims or ())
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, HalideCSEVariable):
                assert arg.used_dims is not None, (name, arg, args)
                used.update(arg.used_dims)
        # 将 used 转为已排序的 used_dims
        self.used_dims = V.kernel.sort_used_dims(used)

    def index_str(self, dims):
        # 根据维度 dims 构造变量的索引字符串
        if len(dims) == 0:
            return f"{self.name}[()]"
        # Halide 使用列优先存储，因此维度需反转
        return f"{self.name}[{', '.join(map(str, dims))}]"

    def __str__(self):
        if self.used_dims is None:
            # 如果 used_dims 为 None，则返回占位符字符串
            # 在 codegen_kernel() 中将重新计算并替换
            return f"{self.name}[?]"
        # 否则返回根据 used_dims 构造的索引字符串
        return self.index_str(self.used_dims)

    def subs_str(self, replacements):
        # 替换 used_dims 中的符号，返回更新后的索引字符串
        assert self.used_dims is not None and all(
            isinstance(x, sympy.Expr) for x in self.used_dims
        )
        return self.index_str([replacements.get(n, n) for n in self.used_dims])


@dataclasses.dataclass
class DimensionInfo:
    expr: Optional[sympy.Expr]
    size: sympy.Expr
    stride: sympy.Expr

    def __init__(self, expr, size, stride):
        super().__init__()
        # 如果 stride < 0，取其绝对值并反转 expr 的符号
        if V.graph.sizevars.statically_known_lt(stride, 0):
            stride = -stride
            expr = -expr
        # 初始化对象的 expr, size 和 stride 属性
        self.expr = expr
        self.size = size
        self.stride = stride

    def index_str(self, replacements=None, zero_vars=False):
        # 返回对象的索引字符串表示
        assert self.expr is not None
        expr = self.expr
        if zero_vars and expr == 0:
            return "hl.Var()"
        if replacements:
            replacements = {**replacements}
            for sym in expr.free_symbols:
                if symbol_is_type(sym, SymT.TMP):
                    assert isinstance(sym, sympy.Symbol)
                    var = V.kernel.lookup_cse_var(sym.name)
                    assert isinstance(var, HalideCSEVariable)
                    replacements[sym] = sympy_index_symbol(var.subs_str(replacements))
            expr = sympy_subs(expr, replacements)
        # 返回索引字符串表示
        return V.kernel.index_to_str(expr)


def eq(left, right):
    # 比较 left 和 right 是否相等
    if V.graph.sizevars.statically_known_equals(left, right):
        return True
    try:
        # 尝试获取 left 和 right 的大小提示
        a = V.graph.sizevars.size_hint(left)
        b = V.graph.sizevars.size_hint(right)
    except TypeError:  # unbacked symints
        return False
    if a == b:
        # 若大小提示相等，则记录 left 和 right 相等
        V.graph.sizevars.guard_equals(left, right)
    # 返回比较结果
    return a == b


def lt(left, right):
    # 比较 left 是否小于 right
    if V.graph.sizevars.statically_known_lt(left, right):
        return True
    # 尝试获取左侧和右侧变量的图形大小提示
    try:
        # 获取左侧变量的图形大小提示
        a = V.graph.sizevars.size_hint(left)
        # 获取右侧变量的图形大小提示
        b = V.graph.sizevars.size_hint(right)
    except TypeError:  # 如果出现类型错误，则表示未支持的符号整数
        # 计算左侧和右侧变量的最大公约数
        gcd = sympy.gcd(left, right)
        # 如果最大公约数等于左侧变量，则返回左侧变量是否不等于右侧变量
        if gcd == left:
            return left != right
        # 否则返回 False
        return False
    
    # 如果左侧变量的图形大小提示小于右侧变量的图形大小提示
    if a < b:
        # 对左侧和右侧变量进行图形大小提示的比较
        V.graph.sizevars.guard_lt(left, right)
    
    # 返回左侧变量的图形大小提示是否小于右侧变量的图形大小提示
    return a < b
class HalideKernel(SIMDKernel):
    overrides = HalideOverrides  # type: ignore[assignment]
    kexpr: Callable[[sympy.Expr], str] = texpr

    def __init__(
        self,
        *groups,
        index_dtype: str,
        mutations: Optional[Set[str]] = None,
        pid_cache=None,
        reduction_hint=ReductionHint.DEFAULT,
        override_persistent_reduction=None,
    ):
        super().__init__(
            *groups,
            index_dtype=index_dtype,
            mutations=mutations,
            reduction_hint=reduction_hint,
            pid_cache=pid_cache,
            override_persistent_reduction=override_persistent_reduction,
        )
        # For halide, we just write directly to the body
        self.compute = self.body
        self.loads = self.body
        self.stores = self.body
        self.indexing_code_dom = IndentedBuffer()
        self.needs_dom_indexing = self.inside_reduction
        self.has_reduction = self.inside_reduction
        self.buffer_dimensions: Dict[str, List[DimensionInfo]] = {}
        self.buffer_offsets: Dict[str, sympy.Expr] = {}
        # {h0: size1, h1: size2, ...}
        self.halide_vars: Dict[sympy.Symbol, sympy.Expr] = {}
        # {x0: h0, x1: h1+10*h2, ...}
        self.index_replacements: Dict[sympy.Expr, sympy.Expr] = {}
        # {h1: hr1, ...}
        self.reduction_renames: Dict[sympy.Symbol, sympy.Symbol] = {}
        # {"i": {h0: hi0}, "o": ...}
        self.dom_renames: Dict[str, Dict[sympy.Symbol, sympy.Symbol]] = {}
        # {"in_ptr0": ["in_ptr0_view0"], ...}
        self.buffer_aliases: Dict[str, List[str]] = defaultdict(list)

    def create_cse_var(self, name, bounds=None):
        # 创建一个新的 HalideCSEVariable 对象，并将其写入到 body 中
        self.body.writeline(f"{name} = hl.Func({name!r})")
        return HalideCSEVariable(name, bounds)

    def setup_dom_indexing(self):
        """RDom based indexing uses explicit iteration ranges for Func updates"""
        prefix = "i" if self.inside_reduction else "o"
        if prefix in self.dom_renames:
            return self.dom_renames[prefix]

        renames = {}
        for var in self.halide_vars.keys():
            if not self.inside_reduction and var in self.reduction_renames:
                continue
            m = re.match(r"^h(\d+)$", var.name)
            assert m
            renames[var] = sympy_index_symbol(f"h{prefix}{m.group(1)}")

        # 生成 RDom 相关的代码段，并将其写入到 indexing_code 中
        self.codegen_rdom(
            f"{prefix}dom", {rv: self.halide_vars[v] for v, rv in renames.items()}
        )

        self.dom_renames[prefix] = renames
        return renames

    def codegen_rdom(self, name, vars):
        # 生成 RDom 的代码，并将其写入到 indexing_code 中
        rsizes = [
            f"hl.Range(0, {self.kexpr(self.rename_indexing(size))})"
            for size in vars.values()
        ]
        self.indexing_code.writeline(f"{name} = hl.RDom([{', '.join(rsizes)}])")
        for i, rsym in enumerate(vars.keys()):
            self.indexing_code.writeline(f"{rsym} = {name}[{i}]")

    def prepare_indexing(
        self,
        index: sympy.Expr,
    ):
        # 调用父类方法准备索引，获取处理后的索引对象
        index = super().prepare_indexing(index)
        # 对索引对象进行符号替换，使用预定义的替换字典
        index = sympy_subs(index, self.index_replacements)
        # 使用 Halide 变量处理索引对象的大小变量，并简化范围
        return V.graph.sizevars.simplify_with_ranges(index, self.halide_vars)  # type: ignore[arg-type]

    def sym_size(self, sym):
        """获取索引符号的大小"""
        # 如果符号为临时类型，则返回对应的间接索引大小
        if symbol_is_type(sym, SymT.TMP):
            return self.lookup_cse_var(sym.name).indirect_indexing_size
        # 否则返回 Halide 变量中对应符号的大小
        return self.halide_vars[sym]

    def install_dims(self, var, dims, offset, is_store):
        """尝试设置 self.buffer_dimensions[var]，成功返回 True"""
        # 如果变量不在 buffer_dimensions 中，则设置其维度和偏移量，并返回成功
        if var not in self.buffer_dimensions:
            self.buffer_dimensions[var] = dims
            self.buffer_offsets[var] = offset
            return True
        # 如果变量已经存在，则检查偏移量和维度是否一致
        if self.buffer_offsets[var] != offset or len(
            self.buffer_dimensions[var]
        ) != len(dims):
            return False
        # 如果是存储操作，直接比较维度是否一致
        if is_store:
            return self.buffer_dimensions[var] == dims
        # 对比每个维度的步长和表达式是否一致，如果不一致则更新旧的维度信息
        for old, new in zip(self.buffer_dimensions[var], dims):
            if old.stride != new.stride:
                return False
            if old.size != new.size or old.expr != new.expr:
                old.size = V.graph.sizevars.evaluate_max(old.size, new.size)
                old.expr = None
        return True

    def apply_offset_to_dimension(self, dims, offset):
        """对维度列表应用偏移量"""
        # 如果偏移量为 0，则直接返回
        if offset == 0:
            return
        # 逆序遍历维度列表，根据步长和偏移量更新表达式
        for i in reversed(range(len(dims))):
            if dims[i].stride == 1 or V.graph.sizevars.statically_known_geq(
                offset, dims[i].stride
            ):
                part = FloorDiv(offset, dims[i].stride)
                offset -= part * dims[i].stride
                dims[i].expr += part
        # 断言偏移量已经完全应用到维度中
        assert offset == 0

    def used_dims_from_index(self, index: sympy.Expr):
        """检测哪些范围树用于填充 HalideCSEVariable.used_dims"""
        # 初始化使用的维度集合
        used_dims = set()
        # 遍历索引中的自由符号
        for sym in index.free_symbols:
            assert isinstance(sym, sympy.Symbol)
            # 如果符号是临时类型，则进行间接索引
            if symbol_is_type(sym, SymT.TMP):
                # 获取符号对应的 HalideCSEVariable 对象
                cse_var = self.lookup_cse_var(sym.name)
                assert (
                    isinstance(cse_var, HalideCSEVariable)
                    and cse_var.used_dims is not None
                )
                # 更新使用的维度集合
                used_dims.update(cse_var.used_dims)
            # 如果符号是 Halide 类型，则直接添加到使用的维度集合中
            elif symbol_is_type(sym, SymT.HALIDE):
                used_dims.add(sym)
            # 如果符号是以下类型之一，则不处理
            elif symbol_is_type(
                sym, (SymT.UNBACKED_INT, SymT.SIZE, SymT.PRECOMPUTED_SIZE, SymT.INDEX)
            ):
                pass
            # 否则抛出未实现的类型错误
            else:
                raise NotImplementedError(f"unhandled symbol {sym}")
        # 返回排序后的使用的维度集合
        return self.sort_used_dims(used_dims)
    # 对已使用的维度进行排序，确保所有元素都是 sympy.Expr 类型
    def sort_used_dims(self, used_dims):
        assert all(isinstance(x, sympy.Expr) for x in used_dims)
        # 按照指定顺序排序，包括 Halide 变量和重命名的约简变量
        ordered = [
            sym
            for sym in itertools.chain(
                self.halide_vars, self.reduction_renames.values()
            )
            if sym in used_dims
        ]
        # 确保排序后的列表长度与 used_dims 相等
        assert len(ordered) == len(used_dims)
        return ordered

    def load(self, name: str, index: sympy.Expr):
        """从 InputBuffer 中生成加载代码"""
        # 调用 args.input 方法获取变量 var
        var = self.args.input(name)
        # 准备索引操作，并返回处理后的索引值
        index = self.prepare_indexing(index)
        # 将 var 和 index 转换为维度信息 dims
        var, dims = self.indexing_to_dimensions(var, index, False)
        # 生成索引字符串
        index_str = ", ".join(d.index_str() for d in dims)
        # 生成加载数据的行代码，使用 workaround 处理尾随逗号问题
        line = f"{var}[{index_str},]"  # trailing comma workaround for https://github.com/halide/Halide/issues/8299
        # 获取变量 name 的数据类型 dtype
        dtype = V.graph.get_dtype(name)
        # 如果数据类型为 torch.float16 或 torch.bfloat16，则转换为 torch.float32
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
            line = f"hl.cast(hl.Float(32), {line})"

        # 如果存在加载掩码 _load_mask
        if self._load_mask:
            # 确保 _load_mask 是 HalideCSEVariable 类型且其 used_dims 不为空
            assert (
                isinstance(self._load_mask, HalideCSEVariable)
                and self._load_mask.used_dims is not None
            )
            # 计算使用的维度集合 used_dims，包括从索引中获取的维度和 _load_mask 的 used_dims
            used_dims = {*self.used_dims_from_index(index), *self._load_mask.used_dims}
            # 生成新的函数结果 result，根据排序后的 used_dims
            result = self.newfunc(self.sort_used_dims(used_dims))
            # 如果 result 的 used_dims 不为空
            if result.used_dims:
                # 写入生成函数体的一行，定义 _mask 变量为 hl.RDom([hl.Range(0, 1)])
                self.body.writeline(f"{result.name}_mask = hl.RDom([hl.Range(0, 1)])")
                # 将 _load_mask 应用到 _mask 变量上
                self.body.writeline(f"{result.name}_mask.where({self._load_mask})")
                # 计算 _load_other 或 0 的表达式并赋给 other 变量
                other = self.kexpr(self._load_other or 0)  # type: ignore[arg-type]
                # 将 other 转换为指定数据类型的代码
                self.body.writeline(
                    f"{result} = hl.cast({halide_type(dtype)}, {other})"
                )
                # 将 line 与 _mask 的数据类型转换后的结果相加，并赋给 result 变量
                self.body.writeline(
                    f"{result} = {line} + hl.cast({halide_type(dtype)}, {result.name}_mask)"
                )
            else:
                # 处理标量情况，如果 result.used_dims 为空
                self.body.writeline(
                    f"{result} = hl.select({self._load_mask}, {line}, hl.cast({halide_type(dtype)}, 0))"
                )
            return result
        else:
            # 如果不存在加载掩码 _load_mask，则调用 genfunc 生成函数并返回结果
            return self.genfunc(line, self.used_dims_from_index(index))

    # 查找并返回名称为 name 的常数传播表达式变量
    def lookup_cse_var(self, name: str):
        return self.cse.varname_map[re.sub(r"\[.*", "", name)]

    # 存储操作方法，将值 value 存储到名称为 name 的变量中，使用 index 进行索引，存储模式为 mode
    ) -> None:
        """
        Codegen a store to an OutputBuffer.
        """
        # 确保 value 是 HalideCSEVariable 类型
        assert isinstance(value, HalideCSEVariable)
        
        # 根据名称获取输出变量 var
        var = self.args.output(name)
        
        # 准备索引操作，将索引转换为维度
        index = self.prepare_indexing(index)
        var, dims = self.indexing_to_dimensions(var, index, True)
        
        # 如果索引是间接索引或者 mode 不为 None，则设置域索引
        if self.is_indirect_indexing(index) or mode is not None:
            replacements = self.setup_dom_indexing()
            # 构建索引字符串和值字符串
            index_str = ", ".join(d.index_str(replacements) for d in dims)
            value_str = value.subs_str(replacements)
            # 创建未定义维度字符串
            undef_dims = ", ".join(["hl.Var()"] * len(dims))
            # 将延迟行写入主体
            self.body.writeline(
                DeferredLine(name, f"{var}[{undef_dims}] = hl.undef({var}.type())")
            )
        else:
            # 构建索引字符串和值字符串
            index_str = ", ".join(d.index_str(zero_vars=True) for d in dims)
            value_str = str(value)

        # 获取数据类型
        dtype = V.graph.get_dtype(name)
        
        # 根据模式生成存储行
        if mode is None:
            line = f"{var}[{index_str},] = hl.cast({halide_type(dtype)}, {value_str})"
        elif mode == "atomic_add":
            line = f"{var}[{index_str},] += hl.cast({halide_type(dtype)}, {value_str})"
        else:
            # 抛出未实现的错误
            raise NotImplementedError(f"store mode={mode}")
        
        # 将延迟行写入主体
        self.body.writeline(DeferredLine(name, line))

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, Tuple[CSEVariable, ...]],
    ) -> Union[CSEVariable, Tuple[CSEVariable, ...]]:
        """定义一个函数，生成一个归约操作"""
        # 断言确保在归约操作内部
        assert self.inside_reduction
        # 断言确保没有加载掩码
        assert not self._load_mask

        # 构建缓存键
        cache_key = (src_dtype, reduction_type, value)
        # 如果缓存中已存在相同的键，则直接返回缓存中的结果
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]

        # 如果值是一个元组，则执行 Welford 合并操作
        if isinstance(value, tuple):
            assert reduction_type == "welford_combine"
            self.cse.reduction_cache[
                cache_key
            ] = result_tuple = self.welford_combine_impl(*value)
            return result_tuple

        # 断言值是 HalideCSEVariable 类型且其使用的维度不为 None
        assert isinstance(value, HalideCSEVariable) and value.used_dims is not None
        # 构建归约变量集合
        reduction_vars = {*self.reduction_renames}
        # 创建新的归约变量
        result_var = self.newfunc(
            [v for v in value.used_dims if v not in reduction_vars]
        )
        # 如果归约变量不在值的使用维度中，则重新生成值
        if reduction_vars - {*value.used_dims}:
            value = self.genfunc(
                f"{value}", self.sort_used_dims({*value.used_dims, *reduction_vars})
            )
        # 获取值的字符串表示，并进行变量替换
        value_str = value.subs_str(self.reduction_renames)
        # 获取默认的归约累加器
        default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
        # 获取 Halide 累加器类型
        acc_type = halide_acc_type(dtype)

        # 根据归约类型执行不同的操作
        if reduction_type in ("argmax", "argmin"):
            # 构建索引变量名
            index = f"{result_var.name}_{reduction_type}"
            # 在主体中写入 Halide 的 argmax 或 argmin 操作
            self.body.writeline(f"{index} = hl.{reduction_type}(rdom, {value_str})")
            # 将 N-D 的 argmax 索引转换为 1-D 的索引
            parts = []
            stride = 1
            for i, sym in enumerate(self.reduction_renames):
                parts.append(f"{index}[{i}]")
                if stride != 1:
                    parts[-1] += f"*{stride}"
                stride *= self.halide_vars[sym]
            self.body.writeline(f"{result_var} = {' + '.join(parts)}")
        elif reduction_type in ("sum", "prod", "min", "max", "any"):
            # 根据归约类型选择不同的函数名
            fn = {
                "sum": "sum",
                "prod": "product",
                "min": "minimum",
                "max": "maximum",
                "any": "maximum",
            }[reduction_type]
            # 在主体中写入 Halide 的归约操作
            self.body.writeline(f"{result_var} = hl.{fn}(rdom, {value_str})")
        elif reduction_type == "xor_sum":
            # 初始化结果变量
            result_var_init = result_var
            # 如果结果变量没有使用的维度，则需要一个虚拟维度
            if not result_var.used_dims:
                result_var_init = result_var.index_str([sympy.Symbol("hl.Var()")])
                result_var.used_dims = [sympy.Integer(0)]
            # 在主体中写入 Halide 的 cast 和异或操作
            self.body.writeline(
                f"{result_var_init} = hl.cast({acc_type}, {halide_constant(default)})"
            )
            self.body.writeline(f"{result_var} = {result_var} ^ {value_str}")
        elif reduction_type == "welford_reduce":
            # 如果是 welford_reduce 类型，则执行特定的回退操作
            # TODO: 实现没有回退的 welford_reduce
            result_var = self.welford_reduce_fallback(dtype, value)
        else:
            # 如果归约类型不被支持，则抛出异常
            raise Unsupported(reduction_type)

        # 将计算结果存入缓存
        self.cse.reduction_cache[cache_key] = result_var
        # 返回结果变量
        return result_var
    def welford_combine_impl(self, mean, m2, weight):
        # 断言确保mean, m2, weight是HalideCSEVariable类型，并且它们有被使用的维度信息
        assert isinstance(mean, HalideCSEVariable) and mean.used_dims is not None
        assert isinstance(m2, HalideCSEVariable) and m2.used_dims is not None
        assert isinstance(weight, HalideCSEVariable) and weight.used_dims is not None
        # 计算出所有被使用的维度，合并并去除已经在self.reduction_renames中重命名的维度
        used_dims = {*mean.used_dims, *m2.used_dims, *weight.used_dims} or {
            *self.halide_vars
        }
        used_dims -= {*self.reduction_renames}
        # 创建一个新的变量result_var，用排序后的used_dims初始化它
        result_var = self.newfunc(self.sort_used_dims(used_dims))
        # 默认值数组，用于创建一个包含默认值的Tuple
        default = [f"hl.cast({x.name}.type(), 0)" for x in (mean, m2, weight)]
        pfx = result_var.name
        # 在self.body中写入创建Tuple的语句
        self.body.writeline(f"{result_var} = hl.Tuple([{', '.join(default)}])")
        # 在self.body中写入分别提取mean, m2, weight的值的语句
        self.body.writeline(f"{pfx}_mean_1 = {result_var}[0]")
        self.body.writeline(f"{pfx}_m2_1 = {result_var}[1]")
        self.body.writeline(f"{pfx}_weight_1 = {result_var}[2]")
        # 在self.body中写入提取经过重命名后的mean, m2, weight值的语句
        self.body.writeline(f"{pfx}_mean_2 = {mean.subs_str(self.reduction_renames)}")
        self.body.writeline(f"{pfx}_m2_2 = {m2.subs_str(self.reduction_renames)}")
        self.body.writeline(
            f"{pfx}_weight_2 = {weight.subs_str(self.reduction_renames)}"
        )
        # 在self.body中写入计算delta, new_weight和w2_over_w的语句
        self.body.writeline(f"{pfx}_delta = {pfx}_mean_2 - {pfx}_mean_1")
        self.body.writeline(f"{pfx}_new_weight = {pfx}_weight_1 + {pfx}_weight_2")
        self.body.writeline(
            f"{pfx}_w2_over_w = hl.select({pfx}_new_weight == 0.0, 0.0, {pfx}_weight_2 / {pfx}_new_weight)"
        )
        # 更新值的表达式数组
        update = [
            f"{pfx}_mean_1 + {pfx}_delta * {pfx}_w2_over_w",
            f"{pfx}_m2_1 + {pfx}_m2_2 + {pfx}_delta * {pfx}_delta * {pfx}_weight_1 * {pfx}_w2_over_w",
            f"{pfx}_new_weight",
        ]
        # 在self.body中写入更新result_var的值的语句
        self.body.writeline(f"{result_var} = hl.Tuple([{', '.join(update)}])")

        # 解包更新后的值
        unpacked = []
        for i in range(3):
            # 创建一个新的变量，用于存储解包后的值
            unpacked.append(self.newfunc(result_var.used_dims))
            self.body.writeline(f"{unpacked[-1]} = {result_var}[{i}]")
        # 返回解包后的结果作为元组
        return tuple(unpacked)

    def genfunc(
        self, line, used_dims, *, bounds=ValueRanges.unknown()
    ) -> HalideCSEVariable:
        # 调用cse生成新的变量，确保它是HalideCSEVariable类型，并且设置它的used_dims属性
        var = self.cse.generate(self.body, line, bounds=bounds)
        assert isinstance(var, HalideCSEVariable)
        var.used_dims = used_dims
        return var

    def newfunc(self, used_dims) -> HalideCSEVariable:
        # 创建一个新的变量，确保它是HalideCSEVariable类型，并且设置它的used_dims属性
        var = self.cse.newvar()
        assert isinstance(var, HalideCSEVariable)
        var.used_dims = used_dims
        return var

    def halide_buffer_numel(self, name: str):
        """
        将所有张量映射到Halide中的1D缓冲区，因为Halide在表示某些PyTorch支持的步长时存在问题。
        如果底层布局中存在间隙，则传递给Halide的numel包括这些间隙，而PyTorch的numel则排除它们。
        """
        # 获取名称为name的缓冲区的布局，并返回其存储大小
        return V.graph.get_buffer(name).get_layout().storage_size()
    def halide_argdefs(self):
        """
        Halide requires scalar inputs before outputs, so need to reorder args.
        """
        # 定义参数排序函数，确保标量输入在输出之前
        def arg_order(arg_tuple):
            call_str, arg = arg_tuple
            if isinstance(arg, SizeArg):
                return 1  # 这个参数通常在最后，移动到中间
            elif "out_ptr" in arg.name:
                return 2  # 输出指针参数
            else:
                assert "in_ptr" in arg.name
                return 0  # 输入指针参数

        result = []
        _, a, b, _ = self.args.python_argdefs()
        # 对参数进行排序，根据定义的排序函数
        for call_str, arg in sorted(zip(a, b), key=arg_order):
            result.append((call_str, arg))
            if isinstance(arg, TensorArg):
                assert arg.offset == 0 and arg.alias_of is None
                # 处理缓冲区别名，确保在参数列表中包含别名
                for alias in self.buffer_aliases.get(arg.name, ()):
                    result.append(
                        (
                            None,
                            TensorArg(
                                alias,
                                arg.buffer,
                                arg.dtype,
                                arg.offset,
                                alias_of=arg.name,
                            ),
                        )
                    )
        return result

    @staticmethod
    def _autoscheduler_workarounds(n, dims):
        if (
            len(dims) == 1
            and config.halide.scheduler_cuda == "Anderson2021"
            and V.graph.scheduler.get_current_device_or_throw().type == "cuda"
        ):
            # 解决 https://github.com/halide/Halide/issues/8246 的问题
            n = max(2, n)  # 强制设定 n 的最小值为 2
        return n

    def call_kernel(self, name: str, node=None):
        """Codegen a call to this kernel"""
        wrapper = V.graph.wrapper_code
        # 生成调用此内核的代码
        call_args = [f"{n}" for n, arg in self.halide_argdefs() if arg.alias_of is None]
        current_device = V.graph.scheduler.get_current_device_or_throw()
        if current_device.type == "cuda":
            # 如果当前设备是 CUDA，则获取 CUDA 流的名称并添加到参数列表
            stream_name = wrapper.write_get_raw_stream(current_device.index, V.graph)
            call_args.append(stream_name)
        wrapper.generate_kernel_call(
            name,
            call_args,
            cuda=False,  # 网格/流在 Halide 内部处理
        )

    def generate_assert(self, check):
        return False  # TODO(jansel): 支持断言功能的实现

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ):
        pass  # TODO(jansel): 支持边界检查功能的实现
class HalideScheduling(SIMDScheduling):
    # 设置类属性，指定32位整数类型
    int32_type = "hl.Int(32)"
    # TODO(jansel): Halide实际上不支持64位索引...
    # 设置类属性，指定64位整数类型，用于注释提醒
    int64_type = "hl.Int(64)"
    # 设置类属性，指定内核类型为HalideKernel类
    kernel_type = HalideKernel

    @classmethod
    def get_backend_features(cls, device: torch.device):
        # 创建一个字典，键为BackendFeature枚举值，值为None
        result = dict.fromkeys(
            [
                BackendFeature.TUPLE_REDUCTION,
                BackendFeature.PREFER_STORE_LOOP_ORDER,
                BackendFeature.REDUCE_TO_SINGLE_ELEMENT,
            ]
        )
        return result

    def define_kernel(self, src_code, node_schedule, kernel):
        """Codegen kernel definition to go in output wrapper code"""
        # 获取图形对象的包装器代码
        wrapper = V.graph.wrapper_code
        # 如果源代码已经在包装器中有对应的内核名
        if src_code in wrapper.src_to_kernel:
            # 获取对应的内核名
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            # 否则，生成一个新的内核名
            kernel_name = f"halide_kernel_{wrapper.next_kernel_suffix()}"
            # 将源代码与新生成的内核名关联存储在包装器中
            wrapper.src_to_kernel[src_code] = kernel_name
            # 添加导入语句到包装器中，导入HalideMeta和HalideInputSpec
            wrapper.add_import_once(
                "from torch._inductor.runtime.hints import HalideMeta, HalideInputSpec"
            )

            # 创建缩进缓冲区用于编译包装器
            compile_wrapper = IndentedBuffer()
            # 写入异步编译Halide内核的调用代码
            compile_wrapper.writeline(
                f"async_compile.halide({kernel.halide_kernel_meta()!r}, '''"
            )
            # 插入源代码到编译包装器中，去除多余的空格
            compile_wrapper.splice(src_code, strip=True)
            # 写入结束标记
            compile_wrapper.writeline("''')")

            # 获取内核元数据和详细内核原点
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            # 构建包含原点信息的注释字符串
            metadata_comment = f"{origins}\n{detailed_origins}"
            # 在包装器中定义内核，包括内核名、编译包装器的内容和元数据注释
            wrapper.define_kernel(
                kernel_name, compile_wrapper.getvalue(), metadata_comment
            )
            # 如果启用了"kernel_metadata"的度量表，则记录内核元数据
            if is_metric_table_enabled("kernel_metadata"):
                log_kernel_metadata(kernel_name, "", src_code)

        # 返回内核名
        return kernel_name
```