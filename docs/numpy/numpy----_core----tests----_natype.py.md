# `.\numpy\numpy\_core\tests\_natype.py`

```
# Vendored implementation of pandas.NA, adapted from pandas/_libs/missing.pyx
#
# This is vendored to avoid adding pandas as a test dependency.

__all__ = ["pd_NA"]  # 指定了此模块中对外公开的对象列表，仅包含 pd_NA

import numbers  # 导入 numbers 模块，用于处理数字相关操作

import numpy as np  # 导入 numpy 库，并使用 np 别名

def _create_binary_propagating_op(name, is_divmod=False):
    is_cmp = name.strip("_") in ["eq", "ne", "le", "lt", "ge", "gt"]  # 检查操作符名称是否属于比较操作

    def method(self, other):
        # 根据不同类型的 other 参数返回相应的 NA 值或对象
        if (
            other is pd_NA
            or isinstance(other, (str, bytes))  # 字符串或字节类型
            or isinstance(other, (numbers.Number, np.bool))  # 数字类型或布尔类型
            or isinstance(other, np.ndarray) and not other.shape  # 空的 NumPy 数组
        ):
            # 对于特定类型的 other，返回 pd_NA
            if is_divmod:
                return pd_NA, pd_NA  # 如果是 divmod 操作，返回两个 pd_NA
            else:
                return pd_NA  # 否则返回单个 pd_NA

        elif isinstance(other, np.ndarray):
            out = np.empty(other.shape, dtype=object)  # 创建一个与 other 形状相同的空对象数组
            out[:] = pd_NA  # 将所有元素赋值为 pd_NA

            if is_divmod:
                return out, out.copy()  # 如果是 divmod 操作，返回两个 out 数组
            else:
                return out  # 否则返回 out 数组

        elif is_cmp and isinstance(other, (np.datetime64, np.timedelta64)):
            return pd_NA  # 如果是比较操作且 other 是日期时间或时间间隔，返回 pd_NA

        elif isinstance(other, np.datetime64):
            if name in ["__sub__", "__rsub__"]:
                return pd_NA  # 对于日期时间的加减法操作，返回 pd_NA

        elif isinstance(other, np.timedelta64):
            if name in ["__sub__", "__rsub__", "__add__", "__radd__"]:
                return pd_NA  # 对于时间间隔的加减法操作，返回 pd_NA

        return NotImplemented  # 其他情况返回 NotImplemented

    method.__name__ = name  # 设置方法名称
    return method  # 返回创建的方法


def _create_unary_propagating_op(name: str):
    def method(self):
        return pd_NA  # 返回 pd_NA

    method.__name__ = name  # 设置方法名称
    return method  # 返回创建的方法


class NAType:
    def __repr__(self) -> str:
        return "<NA>"  # 返回 NA 类型的字符串表示形式

    def __format__(self, format_spec) -> str:
        try:
            return self.__repr__().__format__(format_spec)  # 返回格式化后的 NA 类型字符串表示形式
        except ValueError:
            return self.__repr__()  # 如果格式化失败，返回 NA 类型的字符串表示形式

    def __bool__(self):
        raise TypeError("boolean value of NA is ambiguous")  # 抛出异常，NA 类型的布尔值不明确

    def __hash__(self):
        exponent = 31 if is_32bit else 61  # 根据平台位数选择指数值
        return 2**exponent - 1  # 返回 hash 值

    def __reduce__(self):
        return "pd_NA"  # 返回用于 pickle 时的表示 NA 的字符串

    # Binary arithmetic and comparison ops -> propagate

    __add__ = _create_binary_propagating_op("__add__")  # 创建二进制加法操作并返回
    __radd__ = _create_binary_propagating_op("__radd__")  # 创建反向二进制加法操作并返回
    __sub__ = _create_binary_propagating_op("__sub__")  # 创建二进制减法操作并返回
    __rsub__ = _create_binary_propagating_op("__rsub__")  # 创建反向二进制减法操作并返回
    __mul__ = _create_binary_propagating_op("__mul__")  # 创建二进制乘法操作并返回
    __rmul__ = _create_binary_propagating_op("__rmul__")  # 创建反向二进制乘法操作并返回
    __matmul__ = _create_binary_propagating_op("__matmul__")  # 创建矩阵乘法操作并返回
    __rmatmul__ = _create_binary_propagating_op("__rmatmul__")  # 创建反向矩阵乘法操作并返回
    __truediv__ = _create_binary_propagating_op("__truediv__")  # 创建真除操作并返回
    __rtruediv__ = _create_binary_propagating_op("__rtruediv__")  # 创建反向真除操作并返回
    __floordiv__ = _create_binary_propagating_op("__floordiv__")  # 创建整除操作并返回
    __rfloordiv__ = _create_binary_propagating_op("__rfloordiv__")  # 创建反向整除操作并返回
    # 创建二进制传播操作的方法 "__mod__"，并赋值给 __mod__
    __mod__ = _create_binary_propagating_op("__mod__")
    # 创建反向二进制传播操作的方法 "__rmod__"，并赋值给 __rmod__
    __rmod__ = _create_binary_propagating_op("__rmod__")
    # 创建二进制传播操作的方法 "__divmod__"，并赋值给 __divmod__，指定为 divmod 操作
    __divmod__ = _create_binary_propagating_op("__divmod__", is_divmod=True)
    # 创建反向二进制传播操作的方法 "__rdivmod__"，并赋值给 __rdivmod__，指定为 divmod 操作
    __rdivmod__ = _create_binary_propagating_op("__rdivmod__", is_divmod=True)
    
    # __lshift__ 和 __rshift__ 操作未实现
    
    # 创建二进制传播操作的方法 "__eq__"，并赋值给 __eq__
    __eq__ = _create_binary_propagating_op("__eq__")
    # 创建二进制传播操作的方法 "__ne__"，并赋值给 __ne__
    __ne__ = _create_binary_propagating_op("__ne__")
    # 创建二进制传播操作的方法 "__le__"，并赋值给 __le__
    __le__ = _create_binary_propagating_op("__le__")
    # 创建二进制传播操作的方法 "__lt__"，并赋值给 __lt__
    __lt__ = _create_binary_propagating_op("__lt__")
    # 创建二进制传播操作的方法 "__gt__"，并赋值给 __gt__
    __gt__ = _create_binary_propagating_op("__gt__")
    # 创建二进制传播操作的方法 "__ge__"，并赋值给 __ge__
    __ge__ = _create_binary_propagating_op("__ge__")
    
    # 一元操作
    
    # 创建一元传播操作的方法 "__neg__"，并赋值给 __neg__
    __neg__ = _create_unary_propagating_op("__neg__")
    # 创建一元传播操作的方法 "__pos__"，并赋值给 __pos__
    __pos__ = _create_unary_propagating_op("__pos__")
    # 创建一元传播操作的方法 "__abs__"，并赋值给 __abs__
    __abs__ = _create_unary_propagating_op("__abs__")
    # 创建一元传播操作的方法 "__invert__"，并赋值给 __invert__
    __invert__ = _create_unary_propagating_op("__invert__")
    
    # pow 操作有特殊处理
    
    # 定义 __pow__ 方法，根据 other 的类型进行不同的处理
    def __pow__(self, other):
        if other is pd_NA:
            return pd_NA
        elif isinstance(other, (numbers.Number, np.bool)):
            if other == 0:
                # 返回 1 是对 +/- 0 的正确处理
                return type(other)(1)
            else:
                return pd_NA
        elif util.is_array(other):
            return np.where(other == 0, other.dtype.type(1), pd_NA)
    
        return NotImplemented
    
    # 定义 __rpow__ 方法，根据 other 的类型进行不同的处理
    def __rpow__(self, other):
        if other is pd_NA:
            return pd_NA
        elif isinstance(other, (numbers.Number, np.bool)):
            if other == 1:
                return other
            else:
                return pd_NA
        elif util.is_array(other):
            return np.where(other == 1, other, pd_NA)
        return NotImplemented
    
    # 使用 Kleene 逻辑实现的逻辑操作
    
    # 定义 __and__ 方法，根据 other 的值进行不同的逻辑与操作
    def __and__(self, other):
        if other is False:
            return False
        elif other is True or other is pd_NA:
            return pd_NA
        return NotImplemented
    
    # __rand__ 属性与 __and__ 方法相同
    __rand__ = __and__
    
    # 定义 __or__ 方法，根据 other 的值进行不同的逻辑或操作
    def __or__(self, other):
        if other is True:
            return True
        elif other is False or other is pd_NA:
            return pd_NA
        return NotImplemented
    
    # __ror__ 属性与 __or__ 方法相同
    __ror__ = __or__
    
    # 定义 __xor__ 方法，根据 other 的值进行不同的逻辑异或操作
    def __xor__(self, other):
        if other is False or other is True or other is pd_NA:
            return pd_NA
        return NotImplemented
    
    # __rxor__ 属性与 __xor__ 方法相同
    __rxor__ = __xor__
    
    # 数组优先级设定为 1000
    __array_priority__ = 1000
    # 处理的类型定义为 np.ndarray, numbers.Number, str, np.bool
    _HANDLED_TYPES = (np.ndarray, numbers.Number, str, np.bool)
    # 定义一个特殊方法，用于处理数组对象的通用函数（ufunc）调用
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # 支持处理的数据类型，包括特殊的NA类型（NAType）
        types = self._HANDLED_TYPES + (NAType,)
        # 检查输入参数是否属于支持处理的数据类型，否则返回NotImplemented
        for x in inputs:
            if not isinstance(x, types):
                return NotImplemented

        # 如果ufunc调用方法不是 "__call__"，则抛出错误
        if method != "__call__":
            raise ValueError(f"ufunc method '{method}' not supported for NA")
        
        # 尝试将ufunc调度到对象的dunder方法（例如 __add__、__sub__等）
        result = maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        # 如果无法调度到dunder方法，处理NumPy ufunc的特殊情况
        if result is NotImplemented:
            # 找到inputs中值为pd_NA的第一个元素的索引
            index = [i for i, x in enumerate(inputs) if x is pd_NA][0]
            # 广播所有inputs，获取对应索引位置的结果
            result = np.broadcast_arrays(*inputs)[index]
            # 如果结果是0维，则转换为标量
            if result.ndim == 0:
                result = result.item()
            # 如果ufunc的输出数量大于1，则返回多个pd_NA
            if ufunc.nout > 1:
                result = (pd_NA,) * ufunc.nout

        # 返回处理后的结果
        return result
# 创建一个新的NA类型对象并将其赋值给pd_NA变量
pd_NA = NAType()
```