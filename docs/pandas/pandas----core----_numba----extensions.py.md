# `D:\src\scipysrc\pandas\pandas\core\_numba\extensions.py`

```
# Disable type checking for this module since numba's internals
# are not typed, and we use numba's internals via its extension API
# mypy: ignore-errors

"""
Utility classes/functions to let numba recognize
pandas Index/Series/DataFrame

Mostly vendored from https://github.com/numba/numba/blob/main/numba/tests/pdlike_usecase.py
"""

from __future__ import annotations

from contextlib import contextmanager  # 导入上下文管理器相关模块
import operator  # 导入操作符模块
from typing import TYPE_CHECKING  # 导入类型检查相关模块

import numba  # 导入 numba 库
from numba import types  # 导入 numba 的类型系统模块
from numba.core import cgutils  # 导入 numba 核心的代码生成模块
from numba.core.datamodel import models  # 导入 numba 核心的数据模型模块
from numba.core.extending import (  # 导入 numba 的扩展 API 相关功能
    NativeValue,
    box,
    lower_builtin,
    make_attribute_wrapper,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    type_callable,
    typeof_impl,
    unbox,
)
from numba.core.imputils import impl_ret_borrowed  # 导入 numba 核心的实现工具模块
import numpy as np  # 导入 numpy 库

from pandas._libs import lib  # 导入 pandas 库的 _libs 子模块

from pandas.core.indexes.base import Index  # 导入 pandas 的 Index 基类
from pandas.core.indexing import _iLocIndexer  # 导入 pandas 的 _iLocIndexer 模块
from pandas.core.internals import SingleBlockManager  # 导入 pandas 的 SingleBlockManager 模块
from pandas.core.series import Series  # 导入 pandas 的 Series 模块

if TYPE_CHECKING:
    from pandas._typing import Self  # 如果类型检查开启，导入 Self 类型别名


# Helper function to hack around fact that Index casts numpy string dtype to object
#
# Idea is to set an attribute on a Index called _numba_data
# that is the original data, or the object data casted to numpy string dtype,
# with a context manager that is unset afterwards
@contextmanager
def set_numba_data(index: Index):
    numba_data = index._data
    if numba_data.dtype == object:
        if not lib.is_string_array(numba_data):
            raise ValueError(
                "The numba engine only supports using string or numeric column names"
            )
        numba_data = numba_data.astype("U")
    try:
        index._numba_data = numba_data
        yield index
    finally:
        del index._numba_data


# TODO: Range index support
# (this currently lowers OK, but does not round-trip)
class IndexType(types.Type):
    """
    The type class for Index objects.
    """

    def __init__(self, dtype, layout, pyclass: any) -> None:
        self.pyclass = pyclass
        name = f"index({dtype}, {layout})"
        self.dtype = dtype
        self.layout = layout
        super().__init__(name)

    @property
    def key(self):
        return self.pyclass, self.dtype, self.layout

    @property
    def as_array(self):
        return types.Array(self.dtype, 1, self.layout)

    def copy(self, dtype=None, ndim: int = 1, layout=None) -> Self:
        assert ndim == 1
        if dtype is None:
            dtype = self.dtype
        layout = layout or self.layout
        return type(self)(dtype, layout, self.pyclass)


class SeriesType(types.Type):
    """
    The type class for Series objects.
    """
    # 定义初始化方法，用于创建一个新的 Series 对象
    def __init__(self, dtype, index, namety) -> None:
        # 断言确保 index 是 IndexType 的实例
        assert isinstance(index, IndexType)
        # 设置对象的数据类型
        self.dtype = dtype
        # 设置对象的索引
        self.index = index
        # 创建一个新的一维数组对象，用于存储 Series 的值
        self.values = types.Array(self.dtype, 1, "C")
        # 设置对象的 namety 属性
        self.namety = namety
        # 生成 Series 对象的名称字符串
        name = f"series({dtype}, {index}, {namety})"
        # 调用父类的初始化方法，传入名称
        super().__init__(name)

    @property
    # 定义 key 属性，返回包含 dtype、index 和 namety 的元组
    def key(self):
        return self.dtype, self.index, self.namety

    @property
    # 定义 as_array 属性，返回 Series 对象的 values 属性（一维数组）
    def as_array(self):
        return self.values

    # 定义复制方法，用于生成当前对象的副本
    def copy(self, dtype=None, ndim: int = 1, layout: str = "C") -> Self:
        # 断言确保 ndim 为 1
        assert ndim == 1
        # 断言确保 layout 为 "C"
        assert layout == "C"
        # 如果未指定 dtype，则使用当前对象的 dtype
        if dtype is None:
            dtype = self.dtype
        # 返回当前对象的类型的新实例，参数包括 dtype、index 和 namety
        return type(self)(dtype, self.index, self.namety)
# Register a type inference function for handling instances of `Index`.
@typeof_impl.register(Index)
def typeof_index(val, c) -> IndexType:
    """
    This will assume that only strings are in object dtype
    index.
    (you should check this before this gets lowered down to numba)
    """
    # Determine the type of `val._numba_data` using the type inference system
    arrty = typeof_impl(val._numba_data, c)
    # Ensure that the inferred type has a single dimension
    assert arrty.ndim == 1
    # Return an `IndexType` object encapsulating the dtype, layout, and class type of `val`
    return IndexType(arrty.dtype, arrty.layout, type(val))


# Register a type inference function for handling instances of `Series`.
@typeof_impl.register(Series)
def typeof_series(val, c) -> SeriesType:
    # Determine the type of `val.index` using the type inference system
    index = typeof_impl(val.index, c)
    # Determine the type of `val.values` using the type inference system
    arrty = typeof_impl(val.values, c)
    # Determine the type of `val.name` using the type inference system
    namety = typeof_impl(val.name, c)
    # Ensure that `val.values` has a single dimension and "C" layout
    assert arrty.ndim == 1
    assert arrty.layout == "C"
    # Return a `SeriesType` object encapsulating the dtype of `val.values`, index type, and name type
    return SeriesType(arrty.dtype, index, namety)


# Register a callable type constructor for `Series` objects.
@type_callable(Series)
def type_series_constructor(context):
    def typer(data, index, name=None):
        # Check if `index` is of type `IndexType` and `data` is of type `types.Array`
        if isinstance(index, IndexType) and isinstance(data, types.Array):
            # Ensure `data` has a single dimension
            assert data.ndim == 1
            # If `name` is None, default it to `types.intp`
            if name is None:
                name = types.intp
            # Return a `SeriesType` object with dtype of `data`, `index` type, and `name` type
            return SeriesType(data.dtype, index, name)

    return typer


# Register a callable type constructor for `Index` objects.
@type_callable(Index)
def type_index_constructor(context):
    def typer(data, hashmap=None):
        # Check if `data` is of type `types.Array`
        if isinstance(data, types.Array):
            # Ensure `data` has "C" layout and a single dimension
            assert data.layout == "C"
            assert data.ndim == 1
            # Ensure `hashmap` is either `None` or of type `types.DictType`
            assert hashmap is None or isinstance(hashmap, types.DictType)
            # Return an `IndexType` object with dtype of `data`, "C" layout, and class type `Index`
            return IndexType(data.dtype, layout=data.layout, pyclass=Index)

    return typer


# Register a model for `IndexType` to define its internal structure.
@register_model(IndexType)
class IndexModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None:
        # Define members of the `IndexType` struct
        members = [
            ("data", fe_type.as_array),
            # Define a dictionary type mapping from values in the index to their positions
            ("hashmap", types.DictType(fe_type.dtype, types.intp)),
            # Pointer to the `Index` object from which this `IndexType` was created or boxed
            ("parent", types.pyobject),
        ]
        # Initialize the `StructModel` with the defined members
        models.StructModel.__init__(self, dmm, fe_type, members)


# Register a model for `SeriesType` to define its internal structure.
@register_model(SeriesType)
class SeriesModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None:
        # Define members of the `SeriesType` struct
        members = [
            ("index", fe_type.index),
            ("values", fe_type.as_array),
            ("name", fe_type.namety),
        ]
        # Initialize the `StructModel` with the defined members
        models.StructModel.__init__(self, dmm, fe_type, members)


# Create attribute wrappers for `IndexType` to access its internal data members.
make_attribute_wrapper(IndexType, "data", "_data")
make_attribute_wrapper(IndexType, "hashmap", "hashmap")

# Create attribute wrappers for `SeriesType` to access its internal data members.
make_attribute_wrapper(SeriesType, "index", "index")
make_attribute_wrapper(SeriesType, "values", "values")
make_attribute_wrapper(SeriesType, "name", "name")


# Lowering function for constructing `Series` objects from `types.Array` and `IndexType`.
@lower_builtin(Series, types.Array, IndexType)
def pdseries_constructor(context, builder, sig, args):
    # This function is intended to lower the construction of `Series` objects in the LLVM IR.
    # 从参数 args 中解包得到 data 和 index
    data, index = args
    # 使用 cgutils.create_struct_proxy 创建一个结构体代理对象，返回的对象类型为 sig.return_type
    series = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    # 将 series 对象的 index 属性设置为传入的 index
    series.index = index
    # 将 series 对象的 values 属性设置为传入的 data
    series.values = data
    # 将 series 对象的 name 属性设置为常量 0，类型为 types.intp
    series.name = context.get_constant(types.intp, 0)
    # 使用 impl_ret_borrowed 函数返回一个借用的实现对象，其类型为 sig.return_type，值为 series 对象的内部值
    return impl_ret_borrowed(context, builder, sig.return_type, series._getvalue())
# 注册一个特定类型和参数的函数，用于创建带名称的 pandas Series 对象
@lower_builtin(Series, types.Array, IndexType, types.intp)
@lower_builtin(Series, types.Array, IndexType, types.float64)
@lower_builtin(Series, types.Array, IndexType, types.unicode_type)
def pdseries_constructor_with_name(context, builder, sig, args):
    # 解析传入的参数为 data、index 和 name
    data, index, name = args
    # 使用结构代理创建一个空的 Series 对象
    series = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    # 设置 Series 对象的 index、values 和 name 属性
    series.index = index
    series.values = data
    series.name = name
    # 返回一个从函数返回的借用对象
    return impl_ret_borrowed(context, builder, sig.return_type, series._getvalue())


# 注册一个特定类型和参数的函数，用于创建 Index 对象，接受两个参数
@lower_builtin(Index, types.Array, types.DictType, types.pyobject)
def index_constructor_2arg(context, builder, sig, args):
    # 解析传入的参数为 data、hashmap 和 parent
    (data, hashmap, parent) = args
    # 使用结构代理创建一个空的 Index 对象
    index = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    # 设置 Index 对象的 data、hashmap 和 parent 属性
    index.data = data
    index.hashmap = hashmap
    index.parent = parent
    # 返回一个从函数返回的借用对象
    return impl_ret_borrowed(context, builder, sig.return_type, index._getvalue())


# 注册一个特定类型和参数的函数，用于创建 Index 对象，接受两个参数，但没有 parent 参数
@lower_builtin(Index, types.Array, types.DictType)
def index_constructor_2arg_parent(context, builder, sig, args):
    # 解析传入的参数为 data 和 hashmap
    (data, hashmap) = args
    # 使用结构代理创建一个空的 Index 对象
    index = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    # 设置 Index 对象的 data 和 hashmap 属性
    index.data = data
    index.hashmap = hashmap
    # 返回一个从函数返回的借用对象
    return impl_ret_borrowed(context, builder, sig.return_type, index._getvalue())


# 注册一个特定类型和参数的函数，用于创建 Index 对象，接受一个参数
@lower_builtin(Index, types.Array)
def index_constructor_1arg(context, builder, sig, args):
    from numba.typed import Dict
    
    # 确定 key_type 和 value_type 的类型
    key_type = sig.return_type.dtype
    value_type = types.intp
    
    # 定义一个内部函数 index_impl，用于创建 Index 对象并返回
    def index_impl(data):
        return Index(data, Dict.empty(key_type, value_type))
    
    # 编译内部函数并返回结果
    return context.compile_internal(builder, index_impl, sig, args)


# 辅助函数，将 numpy 的 UnicodeCharSeq 转换为普通字符串
def maybe_cast_str(x):
    # numba 可以重载的虚拟函数
    pass


# 注册 maybe_cast_str 函数的重载，用于将 UnicodeCharSeq 转换为字符串类型
@overload(maybe_cast_str)
def maybe_cast_str_impl(x):
    """Converts numba UnicodeCharSeq (numpy string scalar) -> unicode type (string).
    Is a no-op for other types."""
    if isinstance(x, types.UnicodeCharSeq):
        return lambda x: str(x)
    else:
        return lambda x: x


# 将 Index 对象转换为本地结构的辅助函数
@unbox(IndexType)
def unbox_index(typ, obj, c):
    """
    Convert a Index object to a native structure.

    Note: Object dtype is not allowed here
    """
    # 获取 Index 对象的 _numba_data 属性
    data_obj = c.pyapi.object_getattr_string(obj, "_numba_data")
    # 使用结构代理创建一个空的 Index 对象
    index = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    # 如果是对象数组，则假定已验证只包含字符串，并进行转换
    index.data = c.unbox(typ.as_array, data_obj).value
    # 反序列化并创建一个空的 numba typed dict 对象
    typed_dict_obj = c.pyapi.unserialize(c.pyapi.serialize_object(numba.typed.Dict))
    # 创建一个空的 typed dict，用于作为索引的 hashmap
    # 相当于 numba.typed.Dict.empty(typ.dtype, types.intp)
    arr_type_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.dtype))
    # 使用 CPython API 将 Python 对象 types.intp 序列化为对象
    intp_type_obj = c.pyapi.unserialize(c.pyapi.serialize_object(types.intp))
    
    # 使用 CPython API 调用 typed_dict_obj 对象的 "empty" 方法，并传入 arr_type_obj 和 intp_type_obj 作为参数
    hashmap_obj = c.pyapi.call_method(
        typed_dict_obj, "empty", (arr_type_obj, intp_type_obj)
    )
    
    # 使用 CPython API 将 hashmap_obj 中的数据解包，并转换为 types.DictType(typ.dtype, types.intp) 类型，将其值赋给 index.hashmap
    index.hashmap = c.unbox(types.DictType(typ.dtype, types.intp), hashmap_obj).value
    
    # 将 obj 赋值给 index.parent，用于快速封箱（boxing）
    index.parent = obj

    # 递减引用计数，释放相关的 Python 对象资源
    c.pyapi.decref(data_obj)
    c.pyapi.decref(arr_type_obj)
    c.pyapi.decref(intp_type_obj)
    c.pyapi.decref(typed_dict_obj)

    # 返回 index._getvalue() 的 NativeValue 封装结果
    return NativeValue(index._getvalue())
# 将 Series 对象转换为本地结构的函数装饰器
@unbox(SeriesType)
def unbox_series(typ, obj, c):
    """
    Convert a Series object to a native structure.
    """
    # 获取 Series 对象的 index 属性
    index_obj = c.pyapi.object_getattr_string(obj, "index")
    # 获取 Series 对象的 values 属性
    values_obj = c.pyapi.object_getattr_string(obj, "values")
    # 获取 Series 对象的 name 属性
    name_obj = c.pyapi.object_getattr_string(obj, "name")

    # 使用 cgutils 创建结构代理对象 series
    series = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    # 将 Series 对象的 index 属性解封装为本地结构并赋值给 series.index
    series.index = c.unbox(typ.index, index_obj).value
    # 将 Series 对象的 values 属性解封装为本地结构并赋值给 series.values
    series.values = c.unbox(typ.values, values_obj).value
    # 将 Series 对象的 name 属性解封装为本地结构并赋值给 series.name
    series.name = c.unbox(typ.namety, name_obj).value

    # 减少引用计数
    c.pyapi.decref(index_obj)
    c.pyapi.decref(values_obj)
    c.pyapi.decref(name_obj)

    # 返回本地值对象 NativeValue(series._getvalue())
    return NativeValue(series._getvalue())


# 将本地索引结构转换为 Index 对象的函数装饰器
@box(IndexType)
def box_index(typ, val, c):
    """
    Convert a native index structure to a Index object.

    If our native index is of a numpy string dtype, we'll cast it to
    object.
    """
    # 使用 cgutils 创建结构代理对象 index
    index = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)

    # 分配一次性内存给 res
    res = cgutils.alloca_once_value(c.builder, index.parent)

    # 检查 parent 是否存在
    with c.builder.if_else(cgutils.is_not_null(c.builder, index.parent)) as (
        has_parent,
        otherwise,
    ):
        with has_parent:
            # 若 parent 存在，则增加其引用计数
            c.pyapi.incref(index.parent)
        with otherwise:
            # 若 parent 不存在，则需要重新构造 Index 对象

            # 反序列化并获取 Index 类的对象
            class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Index))
            # 将 index.data 封装为数组对象
            array_obj = c.box(typ.as_array, index.data)

            # 如果 dtype 是 UnicodeCharSeq 类型，则需要转换为 object 类型
            if isinstance(typ.dtype, types.UnicodeCharSeq):
                object_str_obj = c.pyapi.unserialize(c.pyapi.serialize_object("object"))
                array_obj = c.pyapi.call_method(array_obj, "astype", (object_str_obj,))
                c.pyapi.decref(object_str_obj)

            # 构造 Index 对象，相当于 Index._simple_new(array_obj, name_obj) 的 Python 代码
            index_obj = c.pyapi.call_method(class_obj, "_simple_new", (array_obj,))
            index.parent = index_obj
            c.builder.store(index_obj, res)

            # 减少引用计数
            c.pyapi.decref(class_obj)
            c.pyapi.decref(array_obj)

    # 返回加载后的 res
    return c.builder.load(res)


# 将本地 Series 结构转换为 Series 对象的函数装饰器
@box(SeriesType)
def box_series(typ, val, c):
    """
    Convert a native series structure to a Series object.
    """
    # 使用 cgutils 创建结构代理对象 series
    series = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    # 反序列化并获取 Series 类的 _from_mgr 方法对象
    series_const_obj = c.pyapi.unserialize(c.pyapi.serialize_object(Series._from_mgr))
    # 使用序列化对象函数将 SingleBlockManager.from_array 序列化，并用 c.pyapi.unserialize 解序列化
    mgr_const_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(SingleBlockManager.from_array)
    )
    # 使用 c.box 函数将 series.index 包装成对应类型的 Python 对象
    index_obj = c.box(typ.index, series.index)
    # 使用 c.box 函数将 series.values 包装成对应类型的 Python 对象
    array_obj = c.box(typ.as_array, series.values)
    # 使用 c.box 函数将 series.name 包装成对应类型的 Python 对象
    name_obj = c.box(typ.namety, series.name)

    # 调用 mgr_const_obj 函数构造 manager 对象，使用 array_obj 作为数据，index_obj 作为索引
    mgr_obj = c.pyapi.call_function_objargs(
        mgr_const_obj,
        (
            array_obj,
            index_obj,
        ),
    )
    # 从 mgr_obj 中获取 "axes" 属性对象
    mgr_axes_obj = c.pyapi.object_getattr_string(mgr_obj, "axes")

    # 使用 series_const_obj 函数构造 Series 对象，使用 mgr_obj 作为 manager，mgr_axes_obj 作为 axes
    series_obj = c.pyapi.call_function_objargs(
        series_const_obj, (mgr_obj, mgr_axes_obj)
    )
    # 将 name_obj 对象设置为 series_obj 的 "_name" 属性
    c.pyapi.object_setattr_string(series_obj, "_name", name_obj)

    # 释放对象引用，减少内存占用
    c.pyapi.decref(series_const_obj)
    c.pyapi.decref(mgr_axes_obj)
    c.pyapi.decref(mgr_obj)
    c.pyapi.decref(mgr_const_obj)
    c.pyapi.decref(index_obj)
    c.pyapi.decref(array_obj)
    c.pyapi.decref(name_obj)

    # 返回构造好的 Series 对象
    return series_obj
# 定义生成系列缩减函数的函数，支持常见的缩减操作（例如平均值、总和）
# 还支持常见的二元操作（例如加法、减法、乘法、除法）
def generate_series_reduction(ser_reduction, ser_method):
    # 通过装饰器定义对应于 SeriesType 的 ser_reduction 的函数重载
    @overload_method(SeriesType, ser_reduction)
    def series_reduction(series):
        # 实际的系列缩减实现函数，将操作应用到系列的值上
        def series_reduction_impl(series):
            return ser_method(series.values)

        return series_reduction_impl

    return series_reduction


# 定义生成系列二元操作函数的函数
def generate_series_binop(binop):
    # 通过装饰器定义对应于 binop 的函数重载
    @overload(binop)
    def series_binop(series1, value):
        # 如果 series1 是 SeriesType 类型
        if isinstance(series1, SeriesType):
            # 如果 value 也是 SeriesType 类型
            if isinstance(value, SeriesType):

                # 定义对应于两个系列对象的二元操作实现函数
                def series_binop_impl(series1, series2):
                    # TODO: 检查索引是否匹配？
                    return Series(
                        binop(series1.values, series2.values),
                        series1.index,
                        series1.name,
                    )

                return series_binop_impl
            else:

                # 定义对应于系列对象和单值的二元操作实现函数
                def series_binop_impl(series1, value):
                    return Series(
                        binop(series1.values, value), series1.index, series1.name
                    )

                return series_binop_impl

    return series_binop


# 预定义的系列缩减操作及其对应的方法
series_reductions = [
    ("sum", np.sum),
    ("mean", np.mean),
    # 由于 numba 的标准差和 pandas 的标准差存在差异而禁用
    # ("std", np.std),
    # ("var", np.var),
    ("min", np.min),
    ("max", np.max),
]
# 为每个系列缩减操作生成对应的函数
for reduction, reduction_method in series_reductions:
    generate_series_reduction(reduction, reduction_method)

# 预定义的系列二元操作
series_binops = [operator.add, operator.sub, operator.mul, operator.truediv]
# 为每个系列二元操作生成对应的函数
for ser_binop in series_binops:
    generate_series_binop(ser_binop)


# Index 类型的 get_loc 方法的重载
@overload_method(IndexType, "get_loc")
def index_get_loc(index, item):
    # 定义 index_get_loc 方法的实现函数
    def index_get_loc_impl(index, item):
        # 如果哈希表尚未初始化，则初始化它
        if len(index.hashmap) == 0:
            for i, val in enumerate(index._data):
                index.hashmap[val] = i
        # 返回索引项在哈希表中的位置
        return index.hashmap[item]

    return index_get_loc_impl


# Series/Index 的索引操作重载
@overload(operator.getitem)
def series_indexing(series, item):
    # 如果 series 是 SeriesType 类型
    if isinstance(series, SeriesType):

        # 定义 series_getitem 方法的实现函数
        def series_getitem(series, item):
            # 获取索引项在系列索引中的位置
            loc = series.index.get_loc(item)
            # 返回对应位置的系列值
            return series.iloc[loc]

        return series_getitem


# IndexType 类型的索引操作重载
@overload(operator.getitem)
def index_indexing(index, idx):
    # 如果 index 是 IndexType 类型
    if isinstance(index, IndexType):

        # 定义 index_getitem 方法的实现函数
        def index_getitem(index, idx):
            # 返回索引位置 idx 处的数据值
            return index._data[idx]

        return index_getitem


# IlocType 类型的定义，用于 iLoc 索引器
class IlocType(types.Type):
    def __init__(self, obj_type) -> None:
        self.obj_type = obj_type
        name = f"iLocIndexer({obj_type})"
        super().__init__(name=name)

    @property
    def key(self):
        return self.obj_type


# 对 _iLocIndexer 类型进行 typeof_impl 注册的实现
@typeof_impl.register(_iLocIndexer)
def typeof_iloc(val, c) -> IlocType:
    # 获取对象类型的类型信息，并返回 IlocType 类型
    objtype = typeof_impl(val.obj, c)
    return IlocType(objtype)
# 定义一个类型可调用函数，用于处理 _iLocIndexer 类型的对象
def type_iloc_constructor(context):
    # 定义一个内部函数 typer，用于根据输入对象返回相应的类型
    def typer(obj):
        # 如果输入对象是 SeriesType 类型，则返回 IlocType 类型
        if isinstance(obj, SeriesType):
            return IlocType(obj)

    return typer

# 为 _iLocIndexer 和 SeriesType 类型的对象定义下界函数
@lower_builtin(_iLocIndexer, SeriesType)
def iloc_constructor(context, builder, sig, args):
    # 解析参数
    (obj,) = args
    # 创建一个结构代理对象 iloc_indexer
    iloc_indexer = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    # 将输入对象赋值给 iloc_indexer 的 obj 属性
    iloc_indexer.obj = obj
    # 返回一个借用的实现结果
    return impl_ret_borrowed(
        context, builder, sig.return_type, iloc_indexer._getvalue()
    )

# 注册 IlocType 类型的模型
@register_model(IlocType)
class ILocModel(models.StructModel):
    def __init__(self, dmm, fe_type) -> None:
        # 定义结构成员
        members = [("obj", fe_type.obj_type)]
        # 初始化结构模型
        models.StructModel.__init__(self, dmm, fe_type, members)

# 创建 IlocType 类型的 obj 属性的属性包装器
make_attribute_wrapper(IlocType, "obj", "obj")

# 为 SeriesType 类型的对象重载属性 "iloc"
@overload_attribute(SeriesType, "iloc")
def series_iloc(series):
    # 定义获取属性的函数
    def get(series):
        # 返回一个 _iLocIndexer 对象
        return _iLocIndexer(series)

    return get

# 为 iloc_indexer 对象和索引值 i 重载操作符 "getitem"
@overload(operator.getitem)
def iloc_getitem(iloc_indexer, i):
    # 如果 iloc_indexer 是 IlocType 类型的对象
    if isinstance(iloc_indexer, IlocType):
        # 定义获取索引值的实现函数
        def getitem_impl(iloc_indexer, i):
            return iloc_indexer.obj.values[i]

        return getitem_impl
```