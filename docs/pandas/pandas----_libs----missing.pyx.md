# `D:\src\scipysrc\pandas\pandas\_libs\missing.pyx`

```
from decimal import Decimal  # 导入 Decimal 类
import numbers  # 导入 numbers 模块
from sys import maxsize  # 从 sys 模块中导入 maxsize 常量

cimport cython  # 导入 Cython 的 cimport
from cpython.datetime cimport (  # 从 CPython 的 datetime 模块中导入以下类型
    date,  # date 类型
    time,  # time 类型
    timedelta,  # timedelta 类型
)
from cython cimport Py_ssize_t  # 从 Cython 中导入 Py_ssize_t 类型

import numpy as np  # 导入 NumPy 库

cimport numpy as cnp  # 导入 NumPy 的 C 扩展 cimport
from numpy cimport (  # 从 NumPy 的 C 扩展中导入以下类型
    flatiter,  # flatiter 类型
    float64_t,  # float64_t 类型
    int64_t,  # int64_t 类型
    ndarray,  # ndarray 类型
    uint8_t,  # uint8_t 类型
)

cnp.import_array()  # 调用 NumPy 的 import_array 函数

from pandas._libs cimport util  # 从 pandas._libs 中导入 util 模块
from pandas._libs.tslibs.nattype cimport (  # 从 pandas._libs.tslibs.nattype 中导入以下类型
    c_NaT as NaT,  # 将 c_NaT 别名为 NaT
    checknull_with_nat,  # checknull_with_nat 函数
    is_dt64nat,  # is_dt64nat 函数
    is_td64nat,  # is_td64nat 函数
)
from pandas._libs.tslibs.np_datetime cimport (  # 从 pandas._libs.tslibs.np_datetime 中导入以下类型
    get_datetime64_unit,  # get_datetime64_unit 函数
    import_pandas_datetime,  # import_pandas_datetime 函数
)

import_pandas_datetime()  # 调用 import_pandas_datetime 函数

from pandas._libs.ops_dispatch import maybe_dispatch_ufunc_to_dunder_op  # 从 pandas._libs.ops_dispatch 中导入 maybe_dispatch_ufunc_to_dunder_op 函数

cdef:
    float64_t INF = <float64_t>np.inf  # 定义 INF 常量为 NumPy 的正无穷
    float64_t NEGINF = -INF  # 定义 NEGINF 常量为负无穷

    int64_t NPY_NAT = util.get_nat()  # 将 NPY_NAT 常量设为 util 模块的 get_nat() 返回值

    bint is_32bit = maxsize <= 2 ** 32  # 判断当前系统是否为 32 位

    type cDecimal = Decimal  # 为了更快的 isinstance 检查，定义 cDecimal 类型为 Decimal

cpdef bint check_na_tuples_nonequal(object left, object right):
    """
    When we have NA in one of the tuples but not the other we have to check here,
    because our regular checks fail before with ambiguous boolean value.

    Parameters
    ----------
    left: Any
    right: Any

    Returns
    -------
    True if we are dealing with tuples that have NA on one side and non NA on
    the other side.
    """
    if not isinstance(left, tuple) or not isinstance(right, tuple):
        return False  # 如果 left 或 right 不是 tuple 类型，则返回 False

    if len(left) != len(right):
        return False  # 如果 left 和 right 的长度不相等，则返回 False

    for left_element, right_element in zip(left, right):
        if left_element is C_NA and right_element is not C_NA:
            return True  # 如果 left_element 是 C_NA 而 right_element 不是 C_NA，则返回 True
        elif right_element is C_NA and left_element is not C_NA:
            return True  # 如果 right_element 是 C_NA 而 left_element 不是 C_NA，则返回 True

    return False  # 如果以上条件都不满足，则返回 False


cpdef bint is_matching_na(object left, object right, bint nan_matches_none=False):
    """
    Check if two scalars are both NA of matching types.

    Parameters
    ----------
    left : Any
    right : Any
    nan_matches_none : bool, default False
        For backwards compatibility, consider NaN as matching None.

    Returns
    -------
    bool
    """
    if left is None:
        if nan_matches_none and util.is_nan(right):
            return True  # 如果 left 是 None，且 nan_matches_none 为 True，且 right 是 NaN，则返回 True
        return right is None  # 否则，返回 right 是否为 None
    elif left is C_NA:
        return right is C_NA  # 如果 left 是 C_NA，返回 right 是否也是 C_NA
    elif left is NaT:
        return right is NaT  # 如果 left 是 NaT，返回 right 是否也是 NaT
    elif util.is_float_object(left):
        if nan_matches_none and right is None and util.is_nan(left):
            return True  # 如果 left 是 float 对象且是 NaN，且 nan_matches_none 为 True，right 是 None，则返回 True
        return (
            util.is_nan(left)
            and util.is_float_object(right)
            and util.is_nan(right)
        )  # 否则，返回 left 和 right 是否都是 NaN
    elif util.is_complex_object(left):
        return (
            util.is_nan(left)
            and util.is_complex_object(right)
            and util.is_nan(right)
        )  # 如果 left 和 right 都是复数对象且都是 NaN，则返回 True
    elif cnp.is_datetime64_object(left):
        # 如果 left 是 datetime64 类型对象
        return (
            # 左边的值为 NaT（Not a Time）且 right 也是 datetime64 类型对象且值为 NaT
            cnp.get_datetime64_value(left) == NPY_NAT
            and cnp.is_datetime64_object(right)
            and cnp.get_datetime64_value(right) == NPY_NAT
            # 并且左右两边的 datetime64 单位相同
            and get_datetime64_unit(left) == get_datetime64_unit(right)
        )
    elif cnp.is_timedelta64_object(left):
        # 如果 left 是 timedelta64 类型对象
        return (
            # 左边的值为 NaT（Not a Time）且 right 也是 timedelta64 类型对象且值为 NaT
            cnp.get_timedelta64_value(left) == NPY_NAT
            and cnp.is_timedelta64_object(right)
            and cnp.get_timedelta64_value(right) == NPY_NAT
            # 并且左右两边的 timedelta64 单位相同
            and get_datetime64_unit(left) == get_datetime64_unit(right)
        )
    elif is_decimal_na(left):
        # 如果 left 是 decimal_na
        return is_decimal_na(right)
        # 返回 right 是否也是 decimal_na
    # 如果以上条件均不满足，则返回 False
    return False
# 定义一个 Cython 函数，用于检查输入值是否为 NA-like，包括以下情况：
# - None
# - nan
# - NaT
# - np.datetime64 表示的 NaT
# - np.timedelta64 表示的 NaT
# - NA
# - Decimal("NaN")
cpdef bint checknull(object val):

    if val is None or val is NaT or val is C_NA:
        # 如果值为 None 或 NaT 或 C_NA，则返回 True
        return True
    elif util.is_float_object(val) or util.is_complex_object(val):
        # 如果值为浮点数对象或复数对象，则进一步判断是否为 NaN
        if val != val:  # 检查是否为 NaN
            return True
        return False
    elif cnp.is_timedelta64_object(val):
        # 如果值为 timedelta64 对象，则检查其值是否为 NPY_NAT
        return cnp.get_timedelta64_value(val) == NPY_NAT
    elif cnp.is_datetime64_object(val):
        # 如果值为 datetime64 对象，则检查其值是否为 NPY_NAT
        return cnp.get_datetime64_value(val) == NPY_NAT
    else:
        # 否则调用 is_decimal_na 函数来检查是否为 Decimal("NaN")
        return is_decimal_na(val)


# 定义一个 Cython 的函数，用于判断输入值是否为 decimal.Decimal 对象且其值为 Decimal("NaN")
cdef bint is_decimal_na(object val):
    return isinstance(val, cDecimal) and val != val


# 使用 Cython 声明装饰器，禁用数组的负索引访问
@cython.wraparound(False)
# 使用 Cython 声明装饰器，禁用数组的边界检查
@cython.boundscheck(False)
cpdef ndarray[uint8_t] isnaobj(ndarray arr):
    """
    返回一个布尔掩码，指示一维数组中哪些元素是 NA-like 的，根据 `checknull` 函数的定义：
     - None
     - nan
     - NaT
     - np.datetime64 表示的 NaT
     - np.timedelta64 表示的 NaT
     - NA
     - Decimal("NaN")

    Parameters
    ----------
    arr : ndarray

    Returns
    -------
    result : ndarray (dtype=np.bool_)
    """
    cdef:
        Py_ssize_t i, n = arr.size  # 获取数组的大小
        object val  # 声明一个对象类型的变量 val
        bint is_null  # 声明一个布尔类型的变量 is_null
        # 创建一个与 arr 相同形状的无符号 8 位整数数组 result
        ndarray result = np.empty((<object>arr).shape, dtype=np.uint8)
        # 创建一个 arr 的迭代器 it
        flatiter it = cnp.PyArray_IterNew(arr)
        # 创建一个 result 的迭代器 it2
        flatiter it2 = cnp.PyArray_IterNew(result)

    for i in range(n):
        # 使用 PyArray_GETITEM 和 PyArray_ITER_NEXT 可以更快地实现 `val = values[i]`
        val = cnp.PyArray_GETITEM(arr, cnp.PyArray_ITER_DATA(it))
        cnp.PyArray_ITER_NEXT(it)
        # 检查 val 是否为 NA-like
        is_null = checknull(val)
        # 解引用指针（设置值）
        (<uint8_t *>(cnp.PyArray_ITER_DATA(it2)))[0] = <uint8_t>is_null
        cnp.PyArray_ITER_NEXT(it2)

    return result.view(np.bool_)


# 定义一个 Python 函数，用于判断输入的标量值是否为正无穷
def isposinf_scalar(val: object) -> bool:
    return util.is_float_object(val) and val == INF


# 定义一个 Python 函数，用于判断输入的标量值是否为负无穷
def isneginf_scalar(val: object) -> bool:
    return util.is_float_object(val) and val == NEGINF


# 使用 Cython 声明一个函数，用于判断输入的值是否为 null datetime64
cdef bint is_null_datetime64(v):
    # 确定是否为 null 的 datetime (或整数版本)，不包括 np.timedelta64('nat')
    if checknull_with_nat(v) or is_dt64nat(v):
        return True
    return False


# 使用 Cython 声明一个函数，用于判断输入的值是否为 null timedelta64
cdef bint is_null_timedelta64(v):
    # 确定是否为 null 的 timedelta (或整数版本)，不包括 np.datetime64('nat')
    if checknull_with_nat(v) or is_td64nat(v):
        return True
    return False


# 使用 Cython 声明一个函数，检查输入的对象是否为 NA-like 或 C_NA
cdef bint checknull_with_nat_and_na(object obj):
    # 参见 GH#32214
    return checknull_with_nat(obj) or obj is C_NA


# 使用 Cython 声明装饰器，禁用数组的负索引访问
@cython.wraparound(False)
@cython.boundscheck(False)
# 使用 Cython 的 boundscheck(False) 优化性能，关闭数组边界检查

def is_numeric_na(values: ndarray) -> ndarray:
    """
    Check for NA values consistent with IntegerArray/FloatingArray.
    
    Similar to a vectorized is_valid_na_for_dtype restricted to numeric dtypes.
    
    Returns
    -------
    ndarray[bool]
        Boolean array indicating NA values.
    """
    cdef:
        ndarray[uint8_t] result  # 用于存储结果的无符号字节型数组
        Py_ssize_t i, N  # Python 中的 ssize_t 和变量 i、N
        object val  # Python 对象类型变量 val

    N = len(values)  # 获取数组 values 的长度
    result = np.zeros(N, dtype=np.uint8)  # 初始化结果数组为零，数据类型为无符号字节型

    for i in range(N):
        val = values[i]  # 获取 values 数组中的每个元素
        if checknull(val):
            # 检查是否为 NA 值，包括 None、C_NA、NaN 和 Decimal 的 NA
            if val is None or val is C_NA or util.is_nan(val) or is_decimal_na(val):
                result[i] = True  # 如果是 NA 值，设置结果数组对应位置为 True
            else:
                raise TypeError(f"'values' contains non-numeric NA {val}")  # 抛出类型错误异常
    return result.view(bool)  # 返回结果数组，并将其视图转换为布尔类型


# -----------------------------------------------------------------------------
# Implementation of NA singleton

def _create_binary_propagating_op(name, is_divmod=False):
    is_cmp = name.strip("_") in ["eq", "ne", "le", "lt", "ge", "gt"]

    def method(self, other):
        if (other is C_NA or isinstance(other, (str, bytes))
                or isinstance(other, (numbers.Number, np.bool_))
                or util.is_array(other) and not other.shape):
            # 处理 NA 单例操作的二元传播运算符
            # 检查 other 是否为 C_NA、字符串、字节、数字、布尔值或空 NumPy 数组
            # 需要处理 NumPy 标量的 other.shape 条款
            if is_divmod:
                return NA, NA  # 如果是 divmod 操作，返回 NA 对象的元组
            else:
                return NA  # 否则返回 NA 对象

        elif util.is_array(other):
            out = np.empty(other.shape, dtype=object)  # 创建和 other 形状相同的空对象数组
            out[:] = NA  # 将 NA 填充到 out 数组中

            if is_divmod:
                return out, out.copy()  # 如果是 divmod 操作，返回两个填充的 out 数组副本
            else:
                return out  # 否则返回填充的 out 数组

        elif is_cmp and isinstance(other, (date, time, timedelta)):
            return NA  # 处理比较操作且 other 是日期、时间或时间间隔类型时，返回 NA

        elif isinstance(other, date):
            if name in ["__sub__", "__rsub__"]:
                return NA  # 处理日期类型的特定操作，返回 NA

        elif isinstance(other, timedelta):
            if name in ["__sub__", "__rsub__", "__add__", "__radd__"]:
                return NA  # 处理时间间隔类型的特定操作，返回 NA

        return NotImplemented  # 如果无法处理的情况，返回 NotImplemented


def _create_unary_propagating_op(name: str):
    def method(self):
        return NA  # 创建 NA 单例操作的一元传播运算符

    method.__name__ = name  # 设置方法名称
    return method


cdef class C_NAType:
    pass  # 定义 Cython 类 C_NAType


class NAType(C_NAType):
    """
    NA ("not available") missing value indicator.
    
    .. warning::
    
       Experimental: the behaviour of NA can still change without warning.
    
    The NA singleton is a missing value indicator defined by pandas. It is
    used in certain new extension dtypes (currently the "string" dtype).
    
    Examples
    --------
    >>> pd.NA
    <NA>
    
    >>> True | pd.NA
    True
    
    >>> True & pd.NA
    <NA>
    
    >>> pd.NA != pd.NA
    <NA>
    
    >>> pd.NA == pd.NA
    <NA>
    
    >>> True | pd.NA
    True
    """
    _instance = None  # 单例模式中的实例变量
    def __new__(cls, *args, **kwargs):
        # 如果 NAType._instance 为空，则创建一个新的 C_NAType 实例并赋值给 NAType._instance
        if NAType._instance is None:
            NAType._instance = C_NAType.__new__(cls, *args, **kwargs)
        # 返回 NAType 的单例实例
        return NAType._instance

    def __repr__(self) -> str:
        # 返回 NA 的字符串表示形式
        return "<NA>"

    def __format__(self, format_spec) -> str:
        try:
            # 格式化 NA 对象的字符串表示形式
            return self.__repr__().__format__(format_spec)
        except ValueError:
            # 处理格式化异常，返回 NA 的字符串表示形式
            return self.__repr__()

    def __bool__(self):
        # 抛出类型错误，NA 的布尔值不明确
        raise TypeError("boolean value of NA is ambiguous")

    def __hash__(self):
        # GH 30013: 确保哈希值足够大，避免与整数的哈希冲突
        exponent = 31 if is_32bit else 61
        # 返回一个大于整数范围的哈希值
        return 2 ** exponent - 1

    def __reduce__(self):
        # 序列化 NA 对象时返回字符串 "NA"
        return "NA"

    # 二元算术和比较操作 -> 传播

    __add__ = _create_binary_propagating_op("__add__")
    __radd__ = _create_binary_propagating_op("__radd__")
    __sub__ = _create_binary_propagating_op("__sub__")
    __rsub__ = _create_binary_propagating_op("__rsub__")
    __mul__ = _create_binary_propagating_op("__mul__")
    __rmul__ = _create_binary_propagating_op("__rmul__")
    __matmul__ = _create_binary_propagating_op("__matmul__")
    __rmatmul__ = _create_binary_propagating_op("__rmatmul__")
    __truediv__ = _create_binary_propagating_op("__truediv__")
    __rtruediv__ = _create_binary_propagating_op("__rtruediv__")
    __floordiv__ = _create_binary_propagating_op("__floordiv__")
    __rfloordiv__ = _create_binary_propagating_op("__rfloordiv__")
    __mod__ = _create_binary_propagating_op("__mod__")
    __rmod__ = _create_binary_propagating_op("__rmod__")
    __divmod__ = _create_binary_propagating_op("__divmod__", is_divmod=True)
    __rdivmod__ = _create_binary_propagating_op("__rdivmod__", is_divmod=True)

    # __lshift__ 和 __rshift__ 操作未实现

    __eq__ = _create_binary_propagating_op("__eq__")
    __ne__ = _create_binary_propagating_op("__ne__")
    __le__ = _create_binary_propagating_op("__le__")
    __lt__ = _create_binary_propagating_op("__lt__")
    __gt__ = _create_binary_propagating_op("__gt__")
    __ge__ = _create_binary_propagating_op("__ge__")

    # 一元操作

    __neg__ = _create_unary_propagating_op("__neg__")
    __pos__ = _create_unary_propagating_op("__pos__")
    __abs__ = _create_unary_propagating_op("__abs__")
    __invert__ = _create_unary_propagating_op("__invert__")

    # pow 有特殊处理

    def __pow__(self, other):
        # 如果 other 是 C_NA，则返回 NA
        if other is C_NA:
            return NA
        # 如果 other 是数字或布尔值，根据情况返回特定值或 NA
        elif isinstance(other, (numbers.Number, np.bool_)):
            if other == 0:
                # 对于 other == 0，返回正数
                return type(other)(1)
            else:
                return NA
        # 如果 other 是数组，根据条件返回特定值或 NA
        elif util.is_array(other):
            return np.where(other == 0, other.dtype.type(1), NA)

        # 其他情况返回 NotImplemented
        return NotImplemented
    # 定义特殊方法 __rpow__，处理右方的指数运算
    def __rpow__(self, other):
        # 如果右侧操作数是 C_NA（特定常量），返回 NA
        if other is C_NA:
            return NA
        # 如果右侧操作数是数字或者 NumPy 布尔值
        elif isinstance(other, (numbers.Number, np.bool_)):
            # 如果右侧操作数是 1，返回其本身
            if other == 1:
                return other
            # 否则返回 NA
            else:
                return NA
        # 如果右侧操作数是数组类型
        elif util.is_array(other):
            # 使用 NumPy 函数 np.where 处理数组，将等于 1 的元素保留，其余置为 NA
            return np.where(other == 1, other, NA)
        # 其他情况下返回 NotImplemented，表示不支持的操作
        return NotImplemented

    # 逻辑运算符使用 Kleene 逻辑实现

    # 定义特殊方法 __and__，处理与运算
    def __and__(self, other):
        # 如果右侧操作数是 False，返回 False
        if other is False:
            return False
        # 如果右侧操作数是 True 或者 C_NA（特定常量），返回 NA
        elif other is True or other is C_NA:
            return NA
        # 其他情况下返回 NotImplemented，表示不支持的操作
        return NotImplemented

    # __rand__ 是 __and__ 的反向运算，直接引用 __and__

    # 定义特殊方法 __or__，处理或运算
    def __or__(self, other):
        # 如果右侧操作数是 True，返回 True
        if other is True:
            return True
        # 如果右侧操作数是 False 或者 C_NA（特定常量），返回 NA
        elif other is False or other is C_NA:
            return NA
        # 其他情况下返回 NotImplemented，表示不支持的操作
        return NotImplemented

    # __ror__ 是 __or__ 的反向运算，直接引用 __or__

    # 定义特殊方法 __xor__，处理异或运算
    def __xor__(self, other):
        # 如果右侧操作数是 False、True 或者 C_NA（特定常量），返回 NA
        if other is False or other is True or other is C_NA:
            return NA
        # 其他情况下返回 NotImplemented，表示不支持的操作
        return NotImplemented

    # __rxor__ 是 __xor__ 的反向运算，直接引用 __xor__

    # 设置特殊属性 __array_priority__ 为 1000，用于 NumPy 数组操作的优先级
    __array_priority__ = 1000

    # 定义支持的数据类型列表 _HANDLED_TYPES
    _HANDLED_TYPES = (np.ndarray, numbers.Number, str, np.bool_)

    # 定义特殊方法 __array_ufunc__，处理 NumPy ufunc 操作
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # 支持的数据类型包括 _HANDLED_TYPES 和 NAType（特定常量）
        types = self._HANDLED_TYPES + (NAType,)
        # 检查输入是否都属于支持的数据类型
        for x in inputs:
            if not isinstance(x, types):
                return NotImplemented

        # 如果方法不是 "__call__"，抛出异常，因为 NA 不支持此类 ufunc 操作
        if method != "__call__":
            raise ValueError(f"ufunc method '{method}' not supported for NA")

        # 调用内部函数 maybe_dispatch_ufunc_to_dunder_op 处理 ufunc 操作
        result = maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )

        # 如果返回结果是 NotImplemented，处理特定情况下的广播
        if result is NotImplemented:
            # 找到输入中第一个 NA 的索引
            index = [i for i, x in enumerate(inputs) if x is NA][0]
            # 对输入进行广播，获取对应索引的结果
            result = np.broadcast_arrays(*inputs)[index]
            # 如果结果是标量，转换为标量值
            if result.ndim == 0:
                result = result.item()
            # 如果 ufunc 返回多个结果，将结果置为 NA
            if ufunc.nout > 1:
                result = (NA,) * ufunc.nout

        # 返回处理后的结果
        return result
# 创建一个新的NAType对象并赋值给C_NA变量，表示C语言中可见
C_NA = NAType()   # C-visible
# 将C_NA变量的引用赋给NA变量，使得在Python中可见
NA = C_NA         # Python-visible
```