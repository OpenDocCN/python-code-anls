# `D:\src\scipysrc\pandas\pandas\tests\extension\decimal\array.py`

```
from __future__ import annotations
# 导入用于支持类型注释的特殊模块，确保支持类型注释的向前兼容性

import decimal
import numbers
import sys
from typing import TYPE_CHECKING
# 导入必要的标准库和模块

import numpy as np
# 导入 NumPy 库，用于数值计算

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
    is_dtype_equal,
    is_float,
    is_integer,
    pandas_dtype,
)
# 导入 Pandas 的扩展数据类型和相关工具函数

import pandas as pd
# 导入 Pandas 库

from pandas.api.extensions import (
    no_default,
    register_extension_dtype,
)
# 导入 Pandas 扩展 API 中的相关函数和装饰器

from pandas.api.types import (
    is_list_like,
    is_scalar,
)
# 导入 Pandas 的类型检查工具函数

from pandas.core import arraylike
# 导入 Pandas 核心模块中的数组相关工具

from pandas.core.algorithms import value_counts_internal as value_counts
# 导入 Pandas 核心算法模块中的内部值计数函数别名

from pandas.core.arraylike import OpsMixin
# 导入 Pandas 核心数组混合运算工具

from pandas.core.arrays import (
    ExtensionArray,
    ExtensionScalarOpsMixin,
)
# 导入 Pandas 核心数组和扩展标量混合运算工具

from pandas.core.indexers import check_array_indexer
# 导入 Pandas 核心索引器工具函数

if TYPE_CHECKING:
    from pandas._typing import type_t
# 如果是类型检查阶段，导入用于类型提示的类型定义

@register_extension_dtype
# 注册扩展的数据类型
class DecimalDtype(ExtensionDtype):
    type = decimal.Decimal
    name = "decimal"
    na_value = decimal.Decimal("NaN")
    _metadata = ("context",)

    def __init__(self, context=None) -> None:
        self.context = context or decimal.getcontext()
        # 初始化 DecimalDtype 类，设置上下文环境，默认为当前 Decimal 上下文

    def __repr__(self) -> str:
        return f"DecimalDtype(context={self.context})"
        # 返回 DecimalDtype 对象的字符串表示形式

    @classmethod
    def construct_array_type(cls) -> type_t[DecimalArray]:
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return DecimalArray
        # 返回与此数据类型关联的数组类型

    @property
    def _is_numeric(self) -> bool:
        return True
        # 返回此数据类型是否是数值类型的布尔值

class DecimalArray(OpsMixin, ExtensionScalarOpsMixin, ExtensionArray):
    __array_priority__ = 1000

    def __init__(self, values, dtype=None, copy=False, context=None) -> None:
        for i, val in enumerate(values):
            if is_float(val) or is_integer(val):
                if np.isnan(val):
                    values[i] = DecimalDtype.na_value
                    # 如果值是浮点数或整数且为 NaN，则设置为 DecimalDtype 的 na_value
                else:
                    values[i] = DecimalDtype.type(val)  # type: ignore[arg-type]
                    # 否则，将值转换为 Decimal 类型
            elif not isinstance(val, decimal.Decimal):
                raise TypeError("All values must be of type " + str(decimal.Decimal))
                # 如果值不是 Decimal 类型，则引发类型错误异常
        values = np.asarray(values, dtype=object)

        self._data = values
        # 将处理后的值存储在 _data 属性中
        self._items = self.data = self._data
        # 设置 _items、data、_data 别名以确保 Pandas 支持这些属性
        self._dtype = DecimalDtype(context)
        # 设置 _dtype 属性为 DecimalDtype 类的实例，使用给定的上下文环境

    @property
    def dtype(self):
        return self._dtype
        # 返回 DecimalArray 的数据类型

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        return cls(scalars)
        # 类方法，从标量序列创建 DecimalArray 的实例
    # 类方法，从字符串序列创建对象，使用给定的扩展数据类型
    def _from_sequence_of_strings(cls, strings, *, dtype: ExtensionDtype, copy=False):
        # 将输入的字符串列表转换为 Decimal 对象的列表，并使用指定的数据类型创建实例
        return cls._from_sequence(
            [decimal.Decimal(x) for x in strings], dtype=dtype, copy=copy
        )

    # 类方法，从因子化数据创建对象
    def _from_factorized(cls, values, original):
        # 直接使用给定的 values 创建实例
        return cls(values)

    # 类属性，包含被处理的数据类型列表
    _HANDLED_TYPES = (decimal.Decimal, numbers.Number, np.ndarray)

    # 将对象转换为 NumPy 数组
    def to_numpy(
        self,
        dtype=None,
        copy: bool = False,
        na_value: object = no_default,
        decimals=None,
    ) -> np.ndarray:
        # 将对象转换为 NumPy 数组
        result = np.asarray(self, dtype=dtype)
        # 如果指定了小数位数，则对结果进行四舍五入
        if decimals is not None:
            result = np.asarray([round(x, decimals) for x in result])
        return result

    # 处理 NumPy 的数组函数
    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        #
        if not all(
            isinstance(t, self._HANDLED_TYPES + (DecimalArray,)) for t in inputs
        ):
            return NotImplemented

        # 尝试调用对象的 dunder 操作方法
        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        # 如果成功调用，返回结果
        if result is not NotImplemented:
            # 例如：test_array_ufunc_series_scalar_other
            return result

        # 如果存在输出参数，则使用给定的输出参数进行计算
        if "out" in kwargs:
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        # 如果输入中包含 DecimalArray 对象，则将其转换为其内部数据
        inputs = tuple(x._data if isinstance(x, DecimalArray) else x for x in inputs)
        # 使用 NumPy 的数组函数计算结果
        result = getattr(ufunc, method)(*inputs, **kwargs)

        # 如果是 reduce 方法，调用对象的 reduce 处理方法
        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            # 如果处理成功，则返回结果
            if result is not NotImplemented:
                return result

        # 定义用于重建结果的函数
        def reconstruct(x):
            if isinstance(x, (decimal.Decimal, numbers.Number)):
                return x
            else:
                # 否则，从序列重建对象，使用当前对象的数据类型
                return type(self)._from_sequence(x, dtype=self.dtype)

        # 如果输出结果有多个元素，将每个元素重建后返回
        if ufunc.nout > 1:
            return tuple(reconstruct(x) for x in result)
        else:
            # 否则，重建单个结果并返回
            return reconstruct(result)

    # 获取对象的元素或切片
    def __getitem__(self, item):
        if isinstance(item, numbers.Integral):
            # 如果 item 是整数，则直接获取对应位置的数据
            return self._data[item]
        else:
            # 否则，使用 Pandas 的索引检查函数处理 item，并创建新对象返回
            item = pd.api.indexers.check_array_indexer(self, item)
            return type(self)(self._data[item])

    # 从数据中获取指定索引的元素
    def take(self, indexer, allow_fill=False, fill_value=None):
        from pandas.api.extensions import take

        # 获取对象的数据
        data = self._data
        # 如果允许填充且填充值未指定，则使用对象的缺失值
        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        # 使用 Pandas 的 take 函数从数据中获取指定索引的数据
        result = take(data, indexer, fill_value=fill_value, allow_fill=allow_fill)
        # 使用对象的 from_sequence 方法创建新对象并返回
        return self._from_sequence(result, dtype=self.dtype)

    # 复制对象
    def copy(self):
        # 使用当前对象的数据创建副本对象，并返回
        return type(self)(self._data.copy(), dtype=self.dtype)
    # 将数据对象转换为指定的数据类型，返回转换后的副本或者原对象
    def astype(self, dtype, copy=True):
        # 检查目标数据类型是否与当前数据对象的数据类型相同
        if is_dtype_equal(dtype, self._dtype):
            # 如果不需要复制且数据类型相同，直接返回原对象
            if not copy:
                return self
        # 将数据类型转换为 Pandas 数据类型
        dtype = pandas_dtype(dtype)
        # 如果目标数据类型与当前数据对象的数据类型相同，返回新创建的对象
        if isinstance(dtype, type(self.dtype)):
            return type(self)(self._data, copy=copy, context=dtype.context)

        return super().astype(dtype, copy=copy)

    # 设置对象的索引位置的值
    def __setitem__(self, key, value) -> None:
        # 检查值是否类似列表
        if is_list_like(value):
            # 如果索引是标量而值是序列，则引发错误
            if is_scalar(key):
                raise ValueError("setting an array element with a sequence.")
            # 将值列表转换为 Decimal 类型
            value = [decimal.Decimal(v) for v in value]
        else:
            # 将值转换为 Decimal 类型
            value = decimal.Decimal(value)

        # 检查并返回有效的数组索引
        key = check_array_indexer(self, key)
        # 更新数据对象的指定索引位置的值
        self._data[key] = value

    # 返回数据对象的长度
    def __len__(self) -> int:
        return len(self._data)

    # 检查数据对象是否包含指定的 Decimal 类型的值
    def __contains__(self, item) -> bool | np.bool_:
        # 如果 item 不是 Decimal 类型，返回 False
        if not isinstance(item, decimal.Decimal):
            return False
        # 如果 item 是 NaN，则检查数据对象是否有任何 NaN 值
        elif item.is_nan():
            return self.isna().any()
        else:
            # 调用父类方法检查是否包含 item
            return super().__contains__(item)

    # 返回数据对象占用的字节数
    @property
    def nbytes(self) -> int:
        # 计算数据对象的元素个数
        n = len(self)
        # 如果有元素，返回第一个元素所占的内存字节数的总和
        if n:
            return n * sys.getsizeof(self[0])
        # 没有元素则返回 0
        return 0

    # 检查数据对象中是否包含 NaN 值
    def isna(self):
        # 返回一个布尔数组，指示数据对象中每个元素是否为 NaN
        return np.array([x.is_nan() for x in self._data], dtype=bool)

    # 返回 NaN 的 Decimal 值
    @property
    def _na_value(self):
        return decimal.Decimal("NaN")

    # 返回适合打印显示的格式化函数
    def _formatter(self, boxed=False):
        # 如果 boxed 为 True，则返回带有 "Decimal: " 前缀的格式化函数
        if boxed:
            return "Decimal: {}".format
        # 否则返回默认的 repr 函数
        return repr

    # 类方法：将同一类型的对象连接成一个新对象
    @classmethod
    def _concat_same_type(cls, to_concat):
        # 将待连接对象的数据数组拼接起来，创建一个新的对象
        return cls(np.concatenate([x._data for x in to_concat]))

    # 对数据进行指定操作的归约处理
    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        # 如果需要跳过 NaN 并且数据对象包含 NaN 值
        if skipna and self.isna().any():
            # 过滤掉 NaN 值后，对剩余的数据执行相同的归约操作
            other = self[~self.isna()]
            result = other._reduce(name, **kwargs)
        # 如果是求和操作且数据对象长度为 0
        elif name == "sum" and len(self) == 0:
            # 避免在旧版本的 numpy 上返回 int 0 或 np.bool_(False)
            result = decimal.Decimal(0)
        else:
            try:
                # 获取数据对象的指定操作方法
                op = getattr(self.data, name)
            except AttributeError as err:
                # 如果方法不存在，抛出未实现的错误
                raise NotImplementedError(
                    f"decimal does not support the {name} operation"
                ) from err
            # 执行指定操作
            result = op(axis=0)

        # 如果 keepdims 为 True，则返回一个长度为 1 的对象
        if keepdims:
            return type(self)([result])
        # 否则返回归约的结果
        else:
            return result
    # 定义一个比较方法，用于混合运算
    def _cmp_method(self, other, op):
        # 内部函数，将参数转换为值列表，如果参数是扩展数组或类列表，则直接使用，否则假设是对象，复制参数直到与 self 长度相等
        def convert_values(param):
            if isinstance(param, ExtensionArray) or is_list_like(param):
                ovalues = param
            else:
                # 假设是对象，复制参数直到与 self 长度相等
                ovalues = [param] * len(self)
            return ovalues
        
        # 将当前对象 self 视为左值列表
        lvalues = self
        # 调用 convert_values 函数将 other 转换为右值列表
        rvalues = convert_values(other)
        
        # 使用 zip 将左值列表和右值列表逐对传递给操作符函数 op，并生成结果列表
        res = [op(a, b) for (a, b) in zip(lvalues, rvalues)]
        
        # 将结果列表转换为 numpy 数组，并指定数据类型为布尔型
        return np.asarray(res, dtype=bool)

    # 返回通过调用 to_numpy 方法后再调用 value_counts 函数的结果
    def value_counts(self, dropna: bool = True):
        return value_counts(self.to_numpy(), dropna=dropna)

    # 在此处覆盖 fillna 方法以模拟一个第三方扩展数组，该扩展数组尚未更新以包含 fillna 方法中的 "copy" 关键字参数
    def fillna(self, value=None, limit=None):
        return super().fillna(value=value, limit=limit, copy=True)
# 将输入的值转换为 DecimalArray 类型的对象，使用给定的上下文环境（如果提供）
def to_decimal(values, context=None):
    # 创建 DecimalArray 对象，其中每个值都被转换为 Decimal 类型
    return DecimalArray([decimal.Decimal(x) for x in values], context=context)


# 生成包含随机 Decimal 数值的列表
def make_data():
    # 使用 numpy 生成器创建一个包含 100 个随机数的列表，每个随机数被转换为 Decimal 类型
    return [decimal.Decimal(val) for val in np.random.default_rng(2).random(100)]


# 添加 DecimalArray 类的算术运算方法
DecimalArray._add_arithmetic_ops()
```