# `D:\src\scipysrc\pandas\pandas\core\computation\align.py`

```
"""
Core eval alignment algorithms.
"""

from __future__ import annotations  # 允许类型注解中使用当前模块的名称

from functools import (  # 导入 functools 模块中的 partial 和 wraps 函数
    partial,
    wraps,
)
from typing import TYPE_CHECKING  # 导入类型注解需要的 TYPE_CHECKING 类型

import warnings  # 导入警告模块

import numpy as np  # 导入 NumPy 库，并简写为 np

from pandas._config.config import get_option  # 导入 pandas 的配置模块中的 get_option 函数

from pandas.errors import PerformanceWarning  # 导入 pandas 的错误模块中的 PerformanceWarning 类
from pandas.util._exceptions import find_stack_level  # 导入 pandas 的异常处理模块中的 find_stack_level 函数

from pandas.core.dtypes.generic import (  # 导入 pandas 的通用数据类型模块中的特定类
    ABCDataFrame,
    ABCSeries,
)

from pandas.core.base import PandasObject  # 导入 pandas 的核心基类中的 PandasObject 类
import pandas.core.common as com  # 导入 pandas 的核心公共模块并简写为 com
from pandas.core.computation.common import result_type_many  # 导入 pandas 的计算模块中的 result_type_many 函数

if TYPE_CHECKING:  # 如果是类型检查阶段
    from collections.abc import (  # 从 collections.abc 模块导入 Callable 和 Sequence 类
        Callable,
        Sequence,
    )

    from pandas._typing import F  # 从 pandas 的类型注解模块导入 F 类型

    from pandas.core.generic import NDFrame  # 导入 pandas 的核心通用模块中的 NDFrame 类
    from pandas.core.indexes.api import Index  # 导入 pandas 的索引 API 中的 Index 类


def _align_core_single_unary_op(  # 定义函数 _align_core_single_unary_op，接受一个 term 参数
    term,
) -> tuple[partial | type[NDFrame], dict[str, Index] | None]:  # 函数返回一个元组，包含 partial 或 NDFrame 类型和可选的字典
    typ: partial | type[NDFrame]  # 声明 typ 变量，类型为 partial 或 NDFrame
    axes: dict[str, Index] | None = None  # 声明 axes 变量，类型为字典或 None，默认为 None

    if isinstance(term.value, np.ndarray):  # 如果 term.value 是 NumPy 数组
        typ = partial(np.asanyarray, dtype=term.value.dtype)  # 使用 np.asanyarray 函数转换为数组，指定 dtype
    else:  # 否则
        typ = type(term.value)  # typ 为 term.value 的类型
        if hasattr(term.value, "axes"):  # 如果 term.value 有 axes 属性
            axes = _zip_axes_from_type(typ, term.value.axes)  # 调用 _zip_axes_from_type 函数获取 axes 字典

    return typ, axes  # 返回 typ 和 axes


def _zip_axes_from_type(  # 定义函数 _zip_axes_from_type，接受一个 typ 和 new_axes 参数
    typ: type[NDFrame], new_axes: Sequence[Index]  # typ 类型为 NDFrame 的子类，new_axes 类型为 Index 类的序列
) -> dict[str, Index]:  # 函数返回一个字典，键为字符串，值为 Index 对象
    return {name: new_axes[i] for i, name in enumerate(typ._AXIS_ORDERS)}  # 返回根据 typ._AXIS_ORDERS 构建的字典


def _any_pandas_objects(terms) -> bool:  # 定义函数 _any_pandas_objects，接受 terms 参数
    """
    Check a sequence of terms for instances of PandasObject.
    """
    return any(isinstance(term.value, PandasObject) for term in terms)  # 返回是否存在 PandasObject 实例的布尔值


def _filter_special_cases(f) -> Callable[[F], F]:  # 定义函数 _filter_special_cases，接受一个函数 f 作为参数
    @wraps(f)  # 使用 functools.wraps 装饰器，保留原始函数的元数据
    def wrapper(terms):  # 定义内部函数 wrapper，接受 terms 参数
        # single unary operand
        if len(terms) == 1:  # 如果 terms 的长度为 1
            return _align_core_single_unary_op(terms[0])  # 调用 _align_core_single_unary_op 处理 terms 中的第一个元素

        term_values = (term.value for term in terms)  # 生成器表达式，获取 terms 中每个 term 的 value 属性

        # we don't have any pandas objects
        if not _any_pandas_objects(terms):  # 如果 terms 中没有 Pandas 对象
            return result_type_many(*term_values), None  # 返回 result_type_many 函数处理的结果和 None

        return f(terms)  # 调用函数 f 处理 terms

    return wrapper  # 返回 wrapper 函数


@_filter_special_cases  # 应用 _filter_special_cases 装饰器
def _align_core(terms):  # 定义函数 _align_core，接受 terms 参数
    term_index = [i for i, term in enumerate(terms) if hasattr(term.value, "axes")]  # 获取具有 axes 属性的 term 的索引列表
    term_dims = [terms[i].value.ndim for i in term_index]  # 获取具有 axes 属性的 term 的维度列表

    from pandas import Series  # 从 pandas 中导入 Series 类

    ndims = Series(dict(zip(term_index, term_dims)))  # 创建 Series 对象，键为 term_index，值为 term_dims

    # initial axes are the axes of the largest-axis'd term
    biggest = terms[ndims.idxmax()].value  # 获取维度最大的 term 的 value 属性
    typ = biggest._constructor  # 获取最大维度 term 的构造函数
    axes = biggest.axes  # 获取最大维度 term 的 axes
    naxes = len(axes)  # 获取 axes 的长度
    gt_than_one_axis = naxes > 1  # 判断是否大于 1

    for value in (terms[i].value for i in term_index):  # 遍历具有 axes 属性的 term 的 value 属性
        is_series = isinstance(value, ABCSeries)  # 判断 value 是否为 ABCSeries 的实例
        is_series_and_gt_one_axis = is_series and gt_than_one_axis  # 判断是否为 Series 并且 axes 大于 1

        for axis, items in enumerate(value.axes):  # 枚举 value 的 axes
            if is_series_and_gt_one_axis:  # 如果是 Series 并且 axes 大于 1
                ax, itm = naxes - 1, value.index  # 设置 ax 和 itm
            else:
                ax, itm = axis, items  # 设置 ax 和 itm

            if not axes[ax].is_(itm):  # 如果 axes[ax] 不等于 itm
                axes[ax] = axes[ax].union(itm)  # 合并 axes[ax] 和 itm
    # 遍历 ndims 字典中的每对键值对，其中 i 是索引，ndim 是维度
    for i, ndim in ndims.items():
        # 遍历 axes 列表中的每个元素，其中 axis 是索引，items 是元组或列表
        for axis, items in zip(range(ndim), axes):
            # 获取 terms[i] 对应的值 ti
            ti = terms[i].value
            
            # 检查 ti 是否具有 reindex 方法
            if hasattr(ti, "reindex"):
                # 判断 ti 是否为 ABCSeries 的实例且 naxes 大于 1，确定是否需要转置
                transpose = isinstance(ti, ABCSeries) and naxes > 1
                
                # 如果 transpose 为 True，reindexer 设为 axes[naxes - 1]，否则设为 items
                reindexer = axes[naxes - 1] if transpose else items
                
                # 获取 ti 在指定轴上的大小和 reindexer 的大小
                term_axis_size = len(ti.axes[axis])
                reindexer_size = len(reindexer)
                
                # 计算大小差异的对数
                ordm = np.log10(max(1, abs(reindexer_size - term_axis_size)))
                
                # 如果开启了性能警告选项，并且差异超过一个数量级且 reindexer_size 大于等于 10000
                if (
                    get_option("performance_warnings")
                    and ordm >= 1
                    and reindexer_size >= 10000
                ):
                    # 构造警告信息字符串
                    w = (
                        f"Alignment difference on axis {axis} is larger "
                        f"than an order of magnitude on term {terms[i].name!r}, "
                        f"by more than {ordm:.4g}; performance may suffer."
                    )
                    # 发出性能警告
                    warnings.warn(
                        w, category=PerformanceWarning, stacklevel=find_stack_level()
                    )
                
                # 对 ti 执行 reindex 操作，使用 reindexer 和指定的 axis
                obj = ti.reindex(reindexer, axis=axis)
                
                # 更新 terms[i] 对应的对象
                terms[i].update(obj)
        
        # 更新 terms[i] 的值为 terms[i].value 的值
        terms[i].update(terms[i].value.values)
    
    # 返回 typ 和通过 _zip_axes_from_type 函数生成的 _zip_axes_from_type(typ, axes) 结果
    return typ, _zip_axes_from_type(typ, axes)
# 定义函数 align_terms，用于对一组术语进行对齐处理
def align_terms(terms):
    """
    Align a set of terms.
    """
    try:
        # 尝试将解析树（实际上是一个嵌套列表）展平
        terms = list(com.flatten(terms))
    except TypeError:
        # 如果无法迭代，说明它可能是一个常量或单个变量
        # 如果 terms.value 是 ABCSeries 或 ABCDataFrame 的实例
        if isinstance(terms.value, (ABCSeries, ABCDataFrame)):
            typ = type(terms.value)  # 获取类型
            name = terms.value.name if isinstance(terms.value, ABCSeries) else None  # 获取名称（如果是 Series）
            return typ, _zip_axes_from_type(typ, terms.value.axes), name  # 返回类型、从类型中提取的轴、名称
        return np.result_type(terms.type), None, None  # 否则返回结果类型、空轴、空名称

    # 如果所有解析的变量都是数值标量
    if all(term.is_scalar for term in terms):
        return result_type_many(*(term.value for term in terms)).type, None, None  # 返回多个结果类型

    # 如果所有输入序列具有共同的名称，则传播它到返回的序列
    names = {term.value.name for term in terms if isinstance(term.value, ABCSeries)}  # 获取所有序列的名称
    name = names.pop() if len(names) == 1 else None  # 如果只有一个名称，则使用它

    # 执行主要的对齐操作
    typ, axes = _align_core(terms)  # 调用内部函数 _align_core 进行对齐
    return typ, axes, name  # 返回类型、轴、名称


# 定义函数 reconstruct_object，用于根据类型、原始值、可能为空的轴重建对象
def reconstruct_object(typ, obj, axes, dtype, name):
    """
    Reconstruct an object given its type, raw value, and possibly empty
    (None) axes.

    Parameters
    ----------
    typ : object
        A type
    obj : object
        The value to use in the type constructor
    axes : dict
        The axes to use to construct the resulting pandas object

    Returns
    -------
    ret : typ
        An object of type ``typ`` with the value `obj` and possible axes
        `axes`.
    """
    try:
        typ = typ.type  # 尝试获取 typ 的类型
    except AttributeError:
        pass

    res_t = np.result_type(obj.dtype, dtype)  # 获取结果类型

    # 如果 typ 不是 partial 的实例且是 PandasObject 的子类
    if not isinstance(typ, partial) and issubclass(typ, PandasObject):
        if name is None:
            return typ(obj, dtype=res_t, **axes)  # 如果名称为空，则构造 PandasObject 对象
        return typ(obj, dtype=res_t, name=name, **axes)  # 否则构造带有名称的 PandasObject 对象

    # 对于像 ~True/~False 这样的特殊情况
    if hasattr(res_t, "type") and typ == np.bool_ and res_t != np.bool_:
        ret_value = res_t.type(obj)  # 构造特定类型的对象
    else:
        ret_value = typ(obj).astype(res_t)  # 将 obj 转换为 typ 类型并转换为 res_t 类型
        # 下面的条件用于区分 0 维数组（标量返回）和 1 元素数组
        # 例如 np.array(0) 和 np.array([0])
        if (
            len(obj.shape) == 1
            and len(obj) == 1
            and not isinstance(ret_value, np.ndarray)
        ):
            ret_value = np.array([ret_value]).astype(res_t)  # 将标量转换为 1 元素数组并设置类型

    return ret_value  # 返回重建的对象
```