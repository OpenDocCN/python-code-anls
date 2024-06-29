# `D:\src\scipysrc\pandas\pandas\core\internals\construction.py`

```
"""
Functions for preparing various inputs passed to the DataFrame or Series
constructors before passing them to a BlockManager.
"""

from __future__ import annotations

from collections import abc
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
from numpy import ma

from pandas._config import using_pyarrow_string_dtype

from pandas._libs import lib

from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
    construct_1d_arraylike_from_scalar,
    dict_compat,
    maybe_cast_to_datetime,
    maybe_convert_platform,
    maybe_infer_to_datetimelike,
)
from pandas.core.dtypes.common import (
    is_1d_only_ea_dtype,
    is_integer_dtype,
    is_list_like,
    is_named_tuple,
    is_object_dtype,
    is_scalar,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna

from pandas.core import (
    algorithms,
    common as com,
)
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
    array as pd_array,
    extract_array,
    range_to_ndarray,
    sanitize_array,
)
from pandas.core.indexes.api import (
    DatetimeIndex,
    Index,
    TimedeltaIndex,
    default_index,
    ensure_index,
    get_objs_combined_axis,
    maybe_sequence_to_range,
    union_indexes,
)
from pandas.core.internals.blocks import (
    BlockPlacement,
    ensure_block_shape,
    new_block,
    new_block_2d,
)
from pandas.core.internals.managers import (
    create_block_manager_from_blocks,
    create_block_manager_from_column_arrays,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        DtypeObj,
        Manager,
        npt,
    )
# ---------------------------------------------------------------------
# BlockManager Interface


def arrays_to_mgr(
    arrays,
    columns: Index,
    index,
    *,
    dtype: DtypeObj | None = None,
    verify_integrity: bool = True,
    consolidate: bool = True,
) -> Manager:
    """
    Segregate Series based on type and coerce into matrices.

    Needs to handle a lot of exceptional cases.
    """
    if verify_integrity:
        # figure out the index, if necessary
        if index is None:
            index = _extract_index(arrays)
        else:
            index = ensure_index(index)

        # don't force copy because getting jammed in an ndarray anyway
        # 将输入的数组进行规范化，确保它们的长度与索引一致，并返回规范化后的数组和引用
        arrays, refs = _homogenize(arrays, index, dtype)
        # _homogenize 确保：
        #  - 所有的数组长度都等于索引的长度
        #  - 所有的数组都是一维的
        #  - 所有的数组要么是 np.ndarray 或 ExtensionArray 的实例
        #  - 所有的数组类型不是 NumpyExtensionArray
    else:
        # 确保索引的有效性
        index = ensure_index(index)
        # 提取每个数组，确保转换为 numpy 数组，而非 Series 对象
        arrays = [extract_array(x, extract_numpy=True) for x in arrays]
        # 对于从 DataFrame._from_arrays 调用的情况，这里进行最小的验证
        # _from_arrays 方法传入的数组不应该是 Series 对象
        refs = [None] * len(arrays)

        # 遍历所有数组进行验证
        for arr in arrays:
            if (
                not isinstance(arr, (np.ndarray, ExtensionArray))
                or arr.ndim != 1
                or len(arr) != len(index)
            ):
                raise ValueError(
                    "Arrays must be 1-dimensional np.ndarray or ExtensionArray "
                    "with length matching len(index)"
                )

    # 确保列索引的有效性
    columns = ensure_index(columns)
    # 检查传入的数组数量与列索引的数量是否匹配
    if len(columns) != len(arrays):
        raise ValueError("len(arrays) must match len(columns)")

    # 从 BlockManager 的角度来看
    axes = [columns, index]

    # 调用函数创建 BlockManager 对象，基于传入的列数组
    return create_block_manager_from_column_arrays(
        arrays, axes, consolidate=consolidate, refs=refs
    )
# 从一个带有掩码的记录数组中提取数据并创建管理器对象
def rec_array_to_mgr(
    data: np.rec.recarray | np.ndarray,  # 输入参数data可以是一个掩码记录数组或者普通的ndarray数组
    index,  # 索引对象，表示行索引
    columns,  # 列索引对象，表示列索引
    dtype: DtypeObj | None,  # 数据类型对象，可以是指定的数据类型或者None
    copy: bool,  # 是否复制数据
) -> Manager:  # 函数返回一个Manager对象

    """
    Extract from a masked rec array and create the manager.
    """
    # 从掩码记录数组中获取数据部分
    fdata = ma.getdata(data)
    
    # 如果索引为空，则使用默认的索引
    if index is None:
        index = default_index(len(fdata))
    else:
        index = ensure_index(index)

    # 如果列索引不为空，则确保其为索引对象
    if columns is not None:
        columns = ensure_index(columns)

    # 将数据转换为数组，并获取列名列表
    arrays, arr_columns = to_arrays(fdata, columns)

    # 重新排列数组以匹配指定的列索引和行数
    arrays, arr_columns = reorder_arrays(arrays, arr_columns, columns, len(index))
    
    # 如果未指定列索引，则使用arr_columns
    if columns is None:
        columns = arr_columns

    # 将数组转换为Manager对象
    mgr = arrays_to_mgr(arrays, columns, index, dtype=dtype)

    # 如果需要复制数据，则复制Manager对象
    if copy:
        mgr = mgr.copy()
    
    return mgr


# ---------------------------------------------------------------------
# DataFrame Constructor Interface


def ndarray_to_mgr(
    values,  # 输入的值，可以是ndarray、列表、Series、Index或ExtensionArray
    index,  # 索引对象，表示行索引
    columns,  # 列索引对象，表示列索引
    dtype: DtypeObj | None,  # 数据类型对象，可以是指定的数据类型或者None
    copy: bool,  # 是否复制数据
) -> Manager:  # 函数返回一个Manager对象

    # 用于DataFrame.__init__方法中
    # 输入值必须是ndarray、列表、Series、Index或ExtensionArray类型的对象
    infer_object = not isinstance(values, (ABCSeries, Index, ExtensionArray))

    # 如果输入值是ABCSeries类型，则处理相应的索引和列
    if isinstance(values, ABCSeries):
        if columns is None:
            if values.name is not None:
                columns = Index([values.name])
        if index is None:
            index = values.index
        else:
            values = values.reindex(index)

        # 处理长度为零的情况（GH #2234）
        if not len(values) and columns is not None and len(columns):
            values = np.empty((0, 1), dtype=object)

    # 获取输入值的dtype属性
    vdtype = getattr(values, "dtype", None)
    refs = None

    # 如果输入值或指定的dtype是一维扩展数组类型，执行以下逻辑
    if is_1d_only_ea_dtype(vdtype) or is_1d_only_ea_dtype(dtype):
        # GH#19157

        # 如果输入值是ndarray或ExtensionArray，并且是二维的情况下，拆分为多个扩展数组
        if isinstance(values, (np.ndarray, ExtensionArray)) and values.ndim > 1:
            # GH#12513 如果使用了二维数组和EA dtype，则将其拆分为多个扩展数组
            values = [
                values[:, n]  # type: ignore[call-overload]
                for n in range(values.shape[1])
            ]
        else:
            values = [values]

        # 如果未指定列索引，则使用默认的索引范围
        if columns is None:
            columns = Index(range(len(values)))
        else:
            columns = ensure_index(columns)

        # 将数据数组转换为Manager对象
        return arrays_to_mgr(values, columns, index, dtype=dtype)

    # 如果输入值的dtype是ExtensionDtype类型，则执行以下逻辑
    elif isinstance(vdtype, ExtensionDtype):
        # 即Datetime64TZ、PeriodDtype等情况；上面已经处理了is_1d_only_ea_dtype(vdtype)的情况
        # 提取数据数组，并转换为numpy数组形式
        values = extract_array(values, extract_numpy=True)
        if copy:
            values = values.copy()
        # 如果数据数组是一维的，则reshape为二维的形式
        if values.ndim == 1:
            values = values.reshape(-1, 1)
    elif isinstance(values, (ABCSeries, Index)):
        # 如果 values 是 pandas 的 Series 或者 Index 对象
        if not copy and (dtype is None or astype_is_view(values.dtype, dtype)):
            # 如果不需要复制数据且数据类型为 None 或者可以直接转换为视图，则保留引用关系
            refs = values._references

        if copy:
            # 如果需要复制数据，则进行深拷贝操作
            values = values._values.copy()
        else:
            # 否则直接获取数据的引用
            values = values._values

        # 确保数据是二维的
        values = _ensure_2d(values)

    elif isinstance(values, (np.ndarray, ExtensionArray)):
        # 如果 values 是 numpy 的 ndarray 或者扩展数组
        # 去除子类信息
        if copy and (dtype is None or astype_is_view(values.dtype, dtype)):
            # 只有在需要复制数据并且后续的 astype 不会导致复制时才进行深拷贝
            values = np.array(values, copy=True, order="F")
        else:
            # 否则只是创建数据的浅拷贝
            values = np.array(values, copy=False)

        # 确保数据是二维的
        values = _ensure_2d(values)

    else:
        # 否则，按定义应该是一个数组
        # 数据类型将被强制转换为单一的数据类型
        values = _prep_ndarraylike(values, copy=copy)

    if dtype is not None and values.dtype != dtype:
        # 如果指定了 dtype 并且 values 的数据类型与指定的 dtype 不同
        # 进行数组的数据清理和规范化
        values = sanitize_array(
            values,
            None,
            dtype=dtype,
            copy=copy,
            allow_2d=True,
        )

    # 确保此时 values 的维度是二维的
    index, columns = _get_axes(
        values.shape[0], values.shape[1], index=index, columns=columns
    )

    # 检查数据、索引和列的形状是否匹配
    _check_values_indices_shape_match(values, index, columns)

    # 对 values 进行转置操作
    values = values.T

    # 如果没有指定 dtype，则尝试在整个数据块上进行对象类型的转换
    # 这是为了处理在对象类型中嵌入日期时间的情况
    if dtype is None and infer_object and is_object_dtype(values.dtype):
        # 提取对象列
        obj_columns = list(values)
        # 对于可能是日期时间的对象列，进行类型推断转换
        maybe_datetime = [maybe_infer_to_datetimelike(x) for x in obj_columns]
        # 如果有任何列发生了类型推断变化，则创建新的数据块
        if any(x is not y for x, y in zip(obj_columns, maybe_datetime)):
            block_values = [
                new_block_2d(ensure_block_shape(dval, 2), placement=BlockPlacement(n))
                for n, dval in enumerate(maybe_datetime)
            ]
        else:
            # 否则直接创建新的数据块
            bp = BlockPlacement(slice(len(columns)))
            nb = new_block_2d(values, placement=bp, refs=refs)
            block_values = [nb]
    elif dtype is None and values.dtype.kind == "U" and using_pyarrow_string_dtype():
        # 如果没有指定 dtype，且 values 的数据类型为 Unicode 字符串，并且正在使用 pyarrow 的字符串类型
        dtype = StringDtype(storage="pyarrow_numpy")

        # 提取对象列
        obj_columns = list(values)
        # 根据 pyarrow 字符串类型构造新的数据块
        block_values = [
            new_block(
                dtype.construct_array_type()._from_sequence(data, dtype=dtype),
                BlockPlacement(slice(i, i + 1)),
                ndim=2,
            )
            for i, data in enumerate(obj_columns)
        ]

    else:
        # 否则，创建新的数据块
        bp = BlockPlacement(slice(len(columns)))
        nb = new_block_2d(values, placement=bp, refs=refs)
        block_values = [nb]
    # 如果 columns 列表长度为 0，则执行以下操作
    if len(columns) == 0:
        # TODO: 检查是否 values 列表长度也为 0？
        # 如果是，将 block_values 置为空列表
        block_values = []

    # 调用函数 create_block_manager_from_blocks，传入以下参数：
    # - block_values: 块数值，此处可能为空列表或具有数据块
    # - [columns, index]: 列和索引组成的列表，作为块管理器的构建参数之一
    # - verify_integrity=False: 禁用完整性验证
    return create_block_manager_from_blocks(
        block_values, [columns, index], verify_integrity=False
    )
# 检查传入的值、索引和列的形状是否匹配，如果不匹配则抛出异常
def _check_values_indices_shape_match(
    values: np.ndarray, index: Index, columns: Index
) -> None:
    """
    Check that the shape implied by our axes matches the actual shape of the
    data.
    """
    if values.shape[1] != len(columns) or values.shape[0] != len(index):
        # 如果传入的值的行数和列数与索引和列的长度不匹配，则抛出异常
        if values.shape[0] == 0 < len(index):
            raise ValueError("Empty data passed with indices specified.")

        passed = values.shape
        implied = (len(index), len(columns))
        raise ValueError(f"Shape of passed values is {passed}, indices imply {implied}")


def dict_to_mgr(
    data: dict,
    index,
    columns,
    *,
    dtype: DtypeObj | None = None,
    copy: bool = True,
) -> Manager:
    """
    Segregate Series based on type and coerce into matrices.
    Needs to handle a lot of exceptional cases.

    Used in DataFrame.__init__
    """
    arrays: Sequence[Any]

    if columns is not None:
        # 确保列是索引对象
        columns = ensure_index(columns)
        # 初始化数组列表
        arrays = [np.nan] * len(columns)
        # 中间索引集合
        midxs = set()
        # 确保数据键是索引对象，并将数据值转换为列表
        data_keys = ensure_index(data.keys())  # type: ignore[arg-type]
        data_values = list(data.values())

        for i, column in enumerate(columns):
            try:
                # 获取列在数据键中的位置索引
                idx = data_keys.get_loc(column)
            except KeyError:
                # 如果列不在数据键中，则加入到中间索引集合中
                midxs.add(i)
                continue
            # 获取列对应的数组
            array = data_values[idx]
            arrays[i] = array
            # 如果数组是标量且为缺失值，则加入到中间索引集合中
            if is_scalar(array) and isna(array):
                midxs.add(i)

        if index is None:
            # 如果索引为None，则根据数组中的非标量数据提取索引
            if midxs:
                index = _extract_index(
                    [array for i, array in enumerate(arrays) if i not in midxs]
                )
            else:
                index = _extract_index(arrays)
        else:
            # 确保索引是索引对象
            index = ensure_index(index)

        # 如果存在中间索引且dtype不是整数类型，则根据dtype构造一维数组
        if midxs and not is_integer_dtype(dtype):
            for i in midxs:
                arr = construct_1d_arraylike_from_scalar(
                    arrays[i],
                    len(index),
                    dtype if dtype is not None else np.dtype("object"),
                )
                arrays[i] = arr

    else:
        # 如果列为None，则将数据的键转换为范围内的序列，作为列索引
        keys = maybe_sequence_to_range(list(data.keys()))
        columns = Index(keys) if keys else default_index(0)
        arrays = [com.maybe_iterable_to_list(data[k]) for k in keys]
    # 如果需要复制数组（即 copy 参数为真），则进行以下操作：
    # 我们只需要复制那些不会被合并的数组，即只有扩展数组（EA arrays）需要复制
    arrays = [
        x.copy()  # 如果 x 是 ExtensionArray 类型，则进行浅复制
        if isinstance(x, ExtensionArray)
        else x.copy(deep=True)  # 如果 x 是 Index 或 ABCSeries 类型，并且其数据类型仅支持一维扩展数组，则进行深复制
        if (
            isinstance(x, Index)  # 如果 x 是 Index 类型
            or isinstance(x, ABCSeries)  # 或者 x 是 ABCSeries 类型
            and is_1d_only_ea_dtype(x.dtype)  # 并且其数据类型仅支持一维扩展数组
        )
        else x  # 否则直接使用 x
        for x in arrays  # 对 arrays 列表中的每个元素 x 执行上述操作
    ]

    # 将复制后的数组以及其他参数传递给 arrays_to_mgr 函数，生成管理器对象
    return arrays_to_mgr(arrays, columns, index, dtype=dtype, consolidate=copy)
def nested_data_to_arrays(
    data: Sequence,
    columns: Index | None,
    index: Index | None,
    dtype: DtypeObj | None,
) -> tuple[list[ArrayLike], Index, Index]:
    """
    Convert a single sequence of arrays to multiple arrays.
    """
    # By the time we get here we have already checked treat_as_nested(data)

    if is_named_tuple(data[0]) and columns is None:
        # 如果数据的第一个元素是命名元组并且未提供列名，则从元组的字段获取列名
        columns = ensure_index(data[0]._fields)

    # 将数据转换为多个数组，并确保列名是索引对象
    arrays, columns = to_arrays(data, columns, dtype=dtype)
    columns = ensure_index(columns)

    if index is None:
        if isinstance(data[0], ABCSeries):
            # 如果数据的第一个元素是 Pandas Series，则从索引中获取名称
            index = _get_names_from_index(data)
        else:
            # 否则使用默认索引
            index = default_index(len(data))

    return arrays, columns, index


def treat_as_nested(data) -> bool:
    """
    Check if we should use nested_data_to_arrays.
    """
    return (
        len(data) > 0
        and is_list_like(data[0])
        and getattr(data[0], "ndim", 1) == 1
        and not (isinstance(data, ExtensionArray) and data.ndim == 2)
    )


# ---------------------------------------------------------------------


def _prep_ndarraylike(values, copy: bool = True) -> np.ndarray:
    # values is specifically _not_ ndarray, EA, Index, or Series
    # We only get here with `not treat_as_nested(values)`

    if len(values) == 0:
        # 如果 values 长度为零，返回一个空的对象数组
        return np.empty((0, 0), dtype=object)
    elif isinstance(values, range):
        # 如果 values 是 range 类型，则转换为 ndarray，并添加一个新轴
        arr = range_to_ndarray(values)
        return arr[..., np.newaxis]

    def convert(v):
        if not is_list_like(v) or isinstance(v, ABCDataFrame):
            return v

        # 将可迭代对象转换为数组，并尝试转换为当前平台的数据类型
        v = extract_array(v, extract_numpy=True)
        res = maybe_convert_platform(v)
        # 这里不进行日期时间推断，因为将在后续处理中逐列进行
        return res

    # values 可能是一维或二维列表
    # 这里相当于 np.asarray，但会进行对象类型的转换和平台数据类型的保留
    # 不会像 np.asarray 那样将例如 [1, "a", True] 转换为 ["1", "a", "True"]
    if is_list_like(values[0]):
        values = np.array([convert(v) for v in values])
    elif isinstance(values[0], np.ndarray) and values[0].ndim == 0:
        # 对于一维 ndarray 的特殊处理，参考 GH#21861
        values = np.array([convert(v) for v in values])
    else:
        values = convert(values)

    return _ensure_2d(values)


def _ensure_2d(values: np.ndarray) -> np.ndarray:
    """
    Reshape 1D values, raise on anything else other than 2D.
    """
    # 如果 values 是一维数组，则重塑为二维数组，否则抛出 ValueError
    if values.ndim == 1:
        values = values.reshape((values.shape[0], 1))
    elif values.ndim != 2:
        raise ValueError(f"Must pass 2-d input. shape={values.shape}")
    return values


def _homogenize(
    data, index: Index, dtype: DtypeObj | None
) -> tuple[list[ArrayLike], list[Any]]:
    oindex = None
    homogenized = []
    # 如果 `data` 中的原始数组类似于 Series，则跟踪该 Series 的引用
    refs: list[Any] = []

    # 遍历数据中的每个元素
    for val in data:
        # 检查当前元素是否为 ABCSeries 或 Index 的实例
        if isinstance(val, (ABCSeries, Index)):
            # 如果指定了 dtype，则将当前元素转换为指定的数据类型
            if dtype is not None:
                val = val.astype(dtype)
            
            # 如果当前元素是 ABCSeries 并且其索引不是预期的索引，则强制重新索引
            if isinstance(val, ABCSeries) and val.index is not index:
                # 强制对齐。我们后面会将其放入 ndarray 中，所以不需要复制数据
                val = val.reindex(index)
            
            # 将当前元素的引用添加到 refs 列表中
            refs.append(val._references)
            # 取出当前元素的值部分
            val = val._values
        else:
            # 如果当前元素是字典类型
            if isinstance(val, dict):
                # GH#41785 这个操作应该等价于（但更快）val = Series(val, index=index)._values
                # 如果 oindex 为空，则将 index 转换为对象数组类型
                if oindex is None:
                    oindex = index.astype("O")
                
                # 如果索引是 DatetimeIndex 或 TimedeltaIndex 类型
                if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
                    # 参见 test_constructor_dict_datetime64_index
                    val = dict_compat(val)
                else:
                    # 参见 test_constructor_subclass_dict
                    val = dict(val)
                
                # 使用 lib.fast_multiget 方法快速获取数据，对于不存在的键，默认填充为 np.nan
                val = lib.fast_multiget(val, oindex._values, default=np.nan)

            # 对当前元素进行数组的清洗和标准化，确保其与索引的长度匹配，并且不进行复制
            val = sanitize_array(val, index, dtype=dtype, copy=False)
            # 检查当前元素与索引的长度是否匹配，如果不匹配则会引发异常
            com.require_length_match(val, index)
            # 将空引用添加到 refs 列表中
            refs.append(None)

        # 将处理过的元素添加到 homogenized 列表中
        homogenized.append(val)

    # 返回处理后的 homogenized 列表和 refs 列表
    return homogenized, refs
def _extract_index(data) -> Index:
    """
    Try to infer an Index from the passed data, raise ValueError on failure.
    """
    # 声明变量 index 为 Index 类型
    index: Index
    # 如果传入的数据长度为 0，则返回一个默认长度为 0 的索引
    if len(data) == 0:
        return default_index(0)

    # 用来存储各种类型数据的长度集合
    raw_lengths = set()
    # 用来存储索引或哈希化的数据的列表
    indexes: list[list[Hashable] | Index] = []

    # 标志变量：是否有原始数组
    have_raw_arrays = False
    # 标志变量：是否有 Pandas 的 Series 对象
    have_series = False
    # 标志变量：是否有字典对象
    have_dicts = False

    # 遍历传入的数据
    for val in data:
        # 如果是 Pandas 的 Series 对象
        if isinstance(val, ABCSeries):
            have_series = True
            # 将 Series 的索引加入索引列表
            indexes.append(val.index)
        # 如果是字典对象
        elif isinstance(val, dict):
            have_dicts = True
            # 将字典的键（即索引）加入索引列表
            indexes.append(list(val.keys()))
        # 如果是类似列表的对象，并且是一维的
        elif is_list_like(val) and getattr(val, "ndim", 1) == 1:
            have_raw_arrays = True
            # 将该数组的长度加入长度集合
            raw_lengths.add(len(val))
        # 如果是 NumPy 的 ndarray 对象，并且不是一维的
        elif isinstance(val, np.ndarray) and val.ndim > 1:
            # 抛出错误，每列的数组必须是一维的
            raise ValueError("Per-column arrays must each be 1-dimensional")

    # 如果索引列表和长度集合都为空，则抛出错误，如果只有标量值，则必须传入索引
    if not indexes and not raw_lengths:
        raise ValueError("If using all scalar values, you must pass an index")

    # 如果存在 Pandas 的 Series 对象
    if have_series:
        # 使用 union_indexes 函数合并所有的索引
        index = union_indexes(indexes)
    # 如果存在字典对象
    elif have_dicts:
        # 使用 union_indexes 函数合并所有的索引，但不排序
        index = union_indexes(indexes, sort=False)

    # 如果存在原始数组
    if have_raw_arrays:
        # 如果长度集合中的长度大于 1，则抛出错误，所有数组必须长度相同
        if len(raw_lengths) > 1:
            raise ValueError("All arrays must be of the same length")

        # 如果同时存在字典对象，则抛出错误，混合字典和非 Series 可能导致排序不明确
        if have_dicts:
            raise ValueError(
                "Mixing dicts with non-Series may lead to ambiguous ordering."
            )

        # 取出长度集合中的唯一长度
        raw_length = raw_lengths.pop()
        # 如果存在 Series 对象，并且数组长度与索引长度不一致，则抛出错误
        if have_series:
            if raw_length != len(index):
                msg = (
                    f"array length {raw_length} does not match index "
                    f"length {len(index)}"
                )
                raise ValueError(msg)
        # 否则，使用默认索引创建索引对象
        else:
            index = default_index(raw_length)

    # 确保返回的索引对象是有效的
    return ensure_index(index)


def reorder_arrays(
    arrays: list[ArrayLike], arr_columns: Index, columns: Index | None, length: int
) -> tuple[list[ArrayLike], Index]:
    """
    Pre-emptively (cheaply) reindex arrays with new columns.
    """
    # 根据列重新排序数组
    if columns is not None:
        # 如果 columns 不为空，并且与 arr_columns 不相等，则进行重新排序
        if not columns.equals(arr_columns):
            # 新数组列表
            new_arrays: list[ArrayLike] = []
            # 获取 arr_columns 到 columns 的索引映射
            indexer = arr_columns.get_indexer(columns)
            # 遍历索引映射
            for i, k in enumerate(indexer):
                # 如果映射为 -1，按照约定，默认填充全为 NaN 的对象 dtype 数组
                if k == -1:
                    arr = np.empty(length, dtype=object)
                    arr.fill(np.nan)
                else:
                    arr = arrays[k]
                # 将处理后的数组添加到新数组列表中
                new_arrays.append(arr)

            # 更新原数组为新数组列表
            arrays = new_arrays
            # 更新 arr_columns 为 columns
            arr_columns = columns

    # 返回更新后的数组和列索引
    return arrays, arr_columns


def _get_names_from_index(data) -> Index:
    # 检查数据中是否有对象具有名称属性
    has_some_name = any(getattr(s, "name", None) is not None for s in data)
    # 如果没有任何对象具有名称属性，则返回默认长度为数据长度的索引
    if not has_some_name:
        return default_index(len(data))

    # 创建一个包含数据长度范围的索引列表
    index: list[Hashable] = list(range(len(data)))
    # 初始化计数器
    count = 0
    # 遍历数据列表 `data` 中的每个元素 `s`，同时获取索引 `i`
    for i, s in enumerate(data):
        # 尝试从元素 `s` 中获取属性 `name` 的值 `n`
        n = getattr(s, "name", None)
        # 如果 `n` 不为 None，则将其赋给索引 `i` 的位置
        if n is not None:
            index[i] = n
        # 如果 `n` 为 None，则使用格式化字符串生成一个默认名称，并增加计数器 `count`
        else:
            index[i] = f"Unnamed {count}"
            count += 1

    # 返回用生成的索引字典 `index` 创建的 `Index` 对象
    return Index(index)
# 辅助函数，用于创建索引的轴
# 返回索引或默认值

def _get_axes(
    N: int, K: int, index: Index | None, columns: Index | None
) -> tuple[Index, Index]:
    # 如果未提供索引，则使用默认索引值创建
    if index is None:
        index = default_index(N)
    else:
        # 确保索引是有效的索引对象
        index = ensure_index(index)

    # 如果未提供列索引，则使用默认列索引值创建
    if columns is None:
        columns = default_index(K)
    else:
        # 确保列索引是有效的索引对象
        columns = ensure_index(columns)
    return index, columns


def dataclasses_to_dicts(data):
    """
    Converts a list of dataclass instances to a list of dictionaries.

    Parameters
    ----------
    data : List[Type[dataclass]]
        要转换的数据类实例列表

    Returns
    --------
    list_dict : List[dict]
        包含字典表示的数据的列表

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Point:
    ...     x: int
    ...     y: int

    >>> dataclasses_to_dicts([Point(1, 2), Point(2, 3)])
    [{'x': 1, 'y': 2}, {'x': 2, 'y': 3}]

    """
    from dataclasses import asdict

    return list(map(asdict, data))


# ---------------------------------------------------------------------
# 转换输入为数组


def to_arrays(
    data, columns: Index | None, dtype: DtypeObj | None = None
) -> tuple[list[ArrayLike], Index]:
    """
    Return list of arrays, columns.

    Returns
    -------
    list[ArrayLike]
        These will become columns in a DataFrame.
    Index
        This will become frame.columns.

    Notes
    -----
    Ensures that len(result_arrays) == len(result_index).
    """

    if not len(data):
        if isinstance(data, np.ndarray):
            if data.dtype.names is not None:
                # 如果是 numpy 结构化数组
                columns = ensure_index(data.dtype.names)
                arrays = [data[name] for name in columns]

                if len(data) == 0:
                    # GH#42456 上述索引操作导致 2D ndarray 的列表
                    # TODO: 这是否与 numpy 有关？
                    for i, arr in enumerate(arrays):
                        if arr.ndim == 2:
                            arrays[i] = arr[:, 0]

                return arrays, columns
        return [], ensure_index([])

    elif isinstance(data, np.ndarray) and data.dtype.names is not None:
        # 如果是 recarray
        columns = Index(list(data.dtype.names))
        arrays = [data[k] for k in columns]
        return arrays, columns

    if isinstance(data[0], (list, tuple)):
        arr = _list_to_arrays(data)
    elif isinstance(data[0], abc.Mapping):
        arr, columns = _list_of_dict_to_arrays(data, columns)
    elif isinstance(data[0], ABCSeries):
        arr, columns = _list_of_series_to_arrays(data, columns)
    else:
        # 最后的尝试
        data = [tuple(x) for x in data]
        arr = _list_to_arrays(data)

    content, columns = _finalize_columns_and_data(arr, columns, dtype)
    return content, columns


def _list_to_arrays(data: list[tuple | list]) -> np.ndarray:
    # 返回的 np.ndarray 具有 ndim = 2
    # 将列表转换为数组的私有函数
    # 如果数据的第一个元素是元组，则调用 to_object_array_tuples 函数转换数据格式
    if isinstance(data[0], tuple):
        content = lib.to_object_array_tuples(data)
    else:
        # 否则，假设数据是列表的列表形式，调用 to_object_array 函数转换数据格式
        content = lib.to_object_array(data)
    # 返回转换后的内容
    return content
# 将系列的列表转换为数组的元组返回
def _list_of_series_to_arrays(
    data: list,               # 输入的数据列表
    columns: Index | None,    # 列索引，可以为 None
) -> tuple[np.ndarray, Index]:
    # 返回的 np.ndarray 的维度为 2

    if columns is None:
        # 如果 columns 为空，则从 data 中筛选出非空的 pass_data，因为 data[0] 是 Series 类型
        pass_data = [x for x in data if isinstance(x, (ABCSeries, ABCDataFrame))]
        # 从 pass_data 中获取组合后的轴
        columns = get_objs_combined_axis(pass_data, sort=False)

    indexer_cache: dict[int, np.ndarray] = {}  # 索引器缓存，用于存储索引器

    aligned_values = []
    for s in data:
        index = getattr(s, "index", None)
        if index is None:
            index = default_index(len(s))  # 使用默认索引，长度为 s 的长度

        if id(index) in indexer_cache:
            indexer = indexer_cache[id(index)]
        else:
            indexer = indexer_cache[id(index)] = index.get_indexer(columns)

        # 提取数组，并根据索引器取值
        values = extract_array(s, extract_numpy=True)
        aligned_values.append(algorithms.take_nd(values, indexer))

    content = np.vstack(aligned_values)  # 垂直堆叠 aligned_values 形成 content
    return content, columns


# 将字典的列表转换为数组的元组返回
def _list_of_dict_to_arrays(
    data: list[dict],         # 字典的列表作为输入数据
    columns: Index | None,    # 列索引，可以为 None
) -> tuple[np.ndarray, Index]:
    """
    Convert list of dicts to numpy arrays

    if `columns` is not passed, column names are inferred from the records
    - for OrderedDict and dicts, the column names match
      the key insertion-order from the first record to the last.
    - For other kinds of dict-likes, the keys are lexically sorted.

    Parameters
    ----------
    data : iterable
        collection of records (OrderedDict, dict)
    columns: iterables or None

    Returns
    -------
    content : np.ndarray[object, ndim=2]
    columns : Index
    """
    if columns is None:
        # 从 data 的键列表中生成列名生成器
        gen = (list(x.keys()) for x in data)
        # 如果 data 中的元素不是 dict 类型，则进行排序
        sort = not any(isinstance(d, dict) for d in data)
        # 使用 lib.fast_unique_multiple_list_gen 生成预处理的列名列表
        pre_cols = lib.fast_unique_multiple_list_gen(gen, sort=sort)
        # 确保 columns 是 Index 类型
        columns = ensure_index(pre_cols)

    # 确保 data 中的元素都是基本的 dict 类型，而不是派生类
    data = [d if type(d) is dict else dict(d) for d in data]  # noqa: E721

    # 使用 lib.dicts_to_array 将 data 转换为数组，列名为 columns
    content = lib.dicts_to_array(data, list(columns))
    return content, columns


# 确保有有效的列名和数据，如果可能的话，将对象类型的数据转换为指定的数据类型
def _finalize_columns_and_data(
    content: np.ndarray,     # 输入数据，维度为 2
    columns: Index | None,   # 列索引，可以为 None
    dtype: DtypeObj | None,  # 数据类型，可以为 None
) -> tuple[list[ArrayLike], Index]:
    """
    Ensure we have valid columns, cast object dtypes if possible.
    """
    contents = list(content.T)  # 将 content 转置后转换为列表

    try:
        # 验证或将 columns 转换为 Index 类型
        columns = _validate_or_indexify_columns(contents, columns)
    except AssertionError as err:
        # 如果出现断言错误，不要向用户显示 AssertionError
        raise ValueError(err) from err

    # 如果 contents 非空且第一个元素的数据类型为对象类型，尝试转换为指定的数据类型
    if len(contents) and contents[0].dtype == np.object_:
        contents = convert_object_array(contents, dtype=dtype)

    return contents, columns


# 验证或将列名转换为 Index 类型
def _validate_or_indexify_columns(
    content: list[np.ndarray],  # 输入数据列表，每个元素是 np.ndarray
    columns: Index | None       # 列索引，可以为 None
) -> Index:
    """
    If columns is None, make numbers as column names; Otherwise, validate that
    columns have valid length.

    Parameters
    ----------
    content : list[np.ndarray]
        List of arrays representing data columns
    columns : Index or None
        Index of column names or None

    Returns
    -------
    Index
        Validated or indexified columns
    """
    # content : list of np.ndarrays
    # columns : Index or None

    # 如果 columns 参数为 None，则使用 default_index 函数为其赋值一个默认的位置索引
    # 返回一个 Index 对象作为列索引
    Returns
    -------
    Index
        If columns is None, assign positional column index value as columns.

    # 当以下情况发生时会引发 AssertionError：
    # - content 不是由列表组成的列表，并且 columns 的长度不等于 content 的长度
    # 引发 AssertionError
    Raises
    ------
    1. AssertionError when content is not composed of list of lists, and if
        length of columns is not equal to length of content.

    # 当以下情况发生时会引发 ValueError：
    # - content 是由列表组成的列表，但是每个子列表的长度不相等
    # - content 是由列表组成的列表，但子列表的长度与 content 的长度不相等
    # 引发 ValueError
    2. ValueError when content is list of lists, but length of each sub-list
        is not equal
    3. ValueError when content is list of lists, but length of sub-list is
        not equal to length of content
    """
    if columns is None:
        # 如果 columns 参数为 None，则使用 default_index 函数为其赋值一个默认的位置索引
        columns = default_index(len(content))
    else:
        # 如果 columns 是一个由列表组成的列表，检查每个元素是否也是列表
        is_mi_list = isinstance(columns, list) and all(
            isinstance(col, list) for col in columns
        )

        # 如果 columns 不是由列表组成的列表，并且其长度不等于 content 的长度
        if not is_mi_list and len(columns) != len(content):  # pragma: no cover
            # caller's responsibility to check for this...
            # 抛出 AssertionError 异常，说明传递的列数与数据列数不匹配
            raise AssertionError(
                f"{len(columns)} columns passed, passed data had "
                f"{len(content)} columns"
            )
        
        # 如果 columns 是一个由列表组成的列表
        if is_mi_list:
            # 检查每个子列表的长度是否相等
            if len({len(col) for col in columns}) > 1:
                # 如果不相等，则抛出 ValueError 异常，说明多级索引的列的长度不同
                raise ValueError(
                    "Length of columns passed for MultiIndex columns is different"
                )

            # 如果 columns 不为空，并且子列表的长度与 content 的长度不相等
            if columns and len(columns[0]) != len(content):
                # 抛出 ValueError 异常，说明多级索引的列与数据的列数不匹配
                raise ValueError(
                    f"{len(columns[0])} columns passed, passed data had "
                    f"{len(content)} columns"
                )
    # 返回处理后的 columns 列表
    return columns
# 定义函数 convert_object_array，用于将对象数组进行转换
def convert_object_array(
    content: list[npt.NDArray[np.object_]],  # 参数 content：包含 numpy 对象数组的列表
    dtype: DtypeObj | None,  # 参数 dtype：指定的 numpy 数据类型或扩展数据类型
    dtype_backend: str = "numpy",  # 参数 dtype_backend：控制是否返回可空/pyarrow数据类型
    coerce_float: bool = False,  # 参数 coerce_float：是否强制将浮点数转换为整数

) -> list[ArrayLike]:  # 函数返回一个列表，其中元素为 ArrayLike 类型的对象

    """
    Internal function to convert object array.

    Parameters
    ----------
    content: List[np.ndarray]
        包含要转换的 numpy 数组的列表
    dtype: np.dtype or ExtensionDtype
        指定的 numpy 数据类型或扩展数据类型
    dtype_backend: str, optional
        控制是否返回可空/pyarrow数据类型，默认为 "numpy"
    coerce_float: bool, optional
        是否强制将浮点数转换为整数，默认为 False

    Returns
    -------
    List[ArrayLike]
        转换后的数组列表
    """

    # provide soft conversion of object dtypes
    # 提供对象数据类型的软转换

    def convert(arr):
        # 如果 dtype 不是对象数据类型 "O"
        if dtype != np.dtype("O"):
            # 调用 lib.maybe_convert_objects 函数尝试转换对象数组
            arr = lib.maybe_convert_objects(
                arr,
                try_float=coerce_float,
                convert_to_nullable_dtype=dtype_backend != "numpy",
            )

            # Notes on cases that get here 2023-02-15
            # 1) we DO get here when arr is all Timestamps and dtype=None
            # 2) disabling this doesn't break the world, so this must be
            #    getting caught at a higher level
            # 3) passing convert_non_numeric to maybe_convert_objects get this right
            # 4) convert_non_numeric?

            # 如果 dtype 是 None
            if dtype is None:
                # 如果 arr 的数据类型是对象数据类型 "O"
                if arr.dtype == np.dtype("O"):
                    # 尝试推断是否为日期时间类型
                    arr = maybe_infer_to_datetimelike(arr)
                    # 如果 dtype_backend 不是 "numpy" 并且 arr 的数据类型仍然是对象数据类型 "O"
                    if dtype_backend != "numpy" and arr.dtype == np.dtype("O"):
                        # 创建新的字符串数据类型对象
                        new_dtype = StringDtype()
                        arr_cls = new_dtype.construct_array_type()
                        arr = arr_cls._from_sequence(arr, dtype=new_dtype)
                # 如果 dtype_backend 不是 "numpy" 并且 arr 是 numpy 数组
                elif dtype_backend != "numpy" and isinstance(arr, np.ndarray):
                    # 如果 arr 的数据类型的种类在 "iufb" 中
                    if arr.dtype.kind in "iufb":
                        # 调用 pd_array 函数将 arr 转换为 Pandas 数组
                        arr = pd_array(arr, copy=False)

            # 如果 dtype 是 ExtensionDtype 类型的对象
            elif isinstance(dtype, ExtensionDtype):
                # 创建 dtype 对应的数组类型对象
                cls = dtype.construct_array_type()
                arr = cls._from_sequence(arr, dtype=dtype, copy=False)

            # 如果 dtype 的数据类型种类在 "mM" 中
            elif dtype.kind in "mM":
                # 调用 maybe_cast_to_datetime 函数尝试转换为日期时间类型
                arr = maybe_cast_to_datetime(arr, dtype)

        return arr

    # 对 content 列表中的每个数组应用 convert 函数进行转换
    arrays = [convert(arr) for arr in content]

    return arrays
```