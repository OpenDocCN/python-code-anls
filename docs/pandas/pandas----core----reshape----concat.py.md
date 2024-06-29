# `D:\src\scipysrc\pandas\pandas\core\reshape\concat.py`

```
"""
Concat routines.
"""

from __future__ import annotations  # 导入未来版本的类型注解支持

from collections import abc  # 导入抽象基类模块
from typing import (  # 导入类型提示相关模块
    TYPE_CHECKING,
    Literal,
    cast,
    overload,
)
import warnings  # 导入警告模块

import numpy as np  # 导入NumPy库

from pandas._libs import lib  # 导入Pandas内部库
from pandas.util._decorators import cache_readonly  # 导入缓存只读装饰器
from pandas.util._exceptions import find_stack_level  # 导入查找堆栈级别异常

from pandas.core.dtypes.common import is_bool  # 导入判断是否为布尔类型的函数
from pandas.core.dtypes.concat import concat_compat  # 导入兼容的连接函数
from pandas.core.dtypes.generic import (  # 导入泛型数据类型相关
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import isna  # 导入判断缺失值的函数

from pandas.core.arrays.categorical import (  # 导入分类数组相关函数
    factorize_from_iterable,
    factorize_from_iterables,
)
import pandas.core.common as com  # 导入Pandas核心通用模块
from pandas.core.indexes.api import (  # 导入索引API
    Index,
    MultiIndex,
    all_indexes_same,
    default_index,
    ensure_index,
    get_objs_combined_axis,
    get_unanimous_names,
)
from pandas.core.internals import concatenate_managers  # 导入内部连接管理器函数

if TYPE_CHECKING:
    from collections.abc import (  # 条件导入集合抽象基类模块
        Callable,
        Hashable,
        Iterable,
        Mapping,
    )

    from pandas._typing import (  # 条件导入Pandas类型提示
        Axis,
        AxisInt,
        HashableT,
    )

    from pandas import (  # 条件导入Pandas核心模块
        DataFrame,
        Series,
    )

# ---------------------------------------------------------------------
# Concatenate DataFrame objects

@overload
def concat(
    objs: Iterable[DataFrame] | Mapping[HashableT, DataFrame],  # 定义objs参数类型为DataFrame的可迭代对象或映射
    *,
    axis: Literal[0, "index"] = ...,  # axis参数可选值为0或字符串"index"
    join: str = ...,  # join参数类型为字符串
    ignore_index: bool = ...,  # ignore_index参数类型为布尔值
    keys: Iterable[Hashable] | None = ...,  # keys参数类型为哈希值的可迭代对象或None
    levels=...,  # levels参数
    names: list[HashableT] | None = ...,  # names参数类型为哈希值的列表或None
    verify_integrity: bool = ...,  # verify_integrity参数类型为布尔值
    sort: bool = ...,  # sort参数类型为布尔值
    copy: bool | lib.NoDefault = ...,  # copy参数类型为布尔值或lib.NoDefault类型
) -> DataFrame: ...  # 返回DataFrame对象

@overload
def concat(
    objs: Iterable[Series] | Mapping[HashableT, Series],  # 定义objs参数类型为Series的可迭代对象或映射
    *,
    axis: Literal[0, "index"] = ...,  # axis参数可选值为0或字符串"index"
    join: str = ...,  # join参数类型为字符串
    ignore_index: bool = ...,  # ignore_index参数类型为布尔值
    keys: Iterable[Hashable] | None = ...,  # keys参数类型为哈希值的可迭代对象或None
    levels=...,  # levels参数
    names: list[HashableT] | None = ...,  # names参数类型为哈希值的列表或None
    verify_integrity: bool = ...,  # verify_integrity参数类型为布尔值
    sort: bool = ...,  # sort参数类型为布尔值
    copy: bool | lib.NoDefault = ...,  # copy参数类型为布尔值或lib.NoDefault类型
) -> Series: ...  # 返回Series对象

@overload
def concat(
    objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],  # 定义objs参数类型为Series或DataFrame的可迭代对象或映射
    *,
    axis: Literal[0, "index"] = ...,  # axis参数可选值为0或字符串"index"
    join: str = ...,  # join参数类型为字符串
    ignore_index: bool = ...,  # ignore_index参数类型为布尔值
    keys: Iterable[Hashable] | None = ...,  # keys参数类型为哈希值的可迭代对象或None
    levels=...,  # levels参数
    names: list[HashableT] | None = ...,  # names参数类型为哈希值的列表或None
    verify_integrity: bool = ...,  # verify_integrity参数类型为布尔值
    sort: bool = ...,  # sort参数类型为布尔值
    copy: bool | lib.NoDefault = ...,  # copy参数类型为布尔值或lib.NoDefault类型
) -> DataFrame | Series: ...  # 返回DataFrame或Series对象

@overload
def concat(
    objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],  # 定义objs参数类型为Series或DataFrame的可迭代对象或映射
    *,
    axis: Literal[1, "columns"],  # axis参数值为1或字符串"columns"
    join: str = ...,  # join参数类型为字符串
    ignore_index: bool = ...,  # ignore_index参数类型为布尔值
    keys: Iterable[Hashable] | None = ...,  # keys参数类型为哈希值的可迭代对象或None
    levels=...,  # levels参数
    names: list[HashableT] | None = ...,  # names参数类型为哈希值的列表或None
    verify_integrity: bool = ...,  # verify_integrity参数类型为布尔值
    sort: bool = ...,  # sort参数类型为布尔值
    copy: bool | lib.NoDefault = ...,  # copy参数类型为布尔值或lib.NoDefault类型
) -> DataFrame: ...  # 返回DataFrame对象
    objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],
    # objs 参数可以接受一个包含 Series 或 DataFrame 的可迭代对象，或者是一个映射，其键为 Hashable 类型，值为 Series 或 DataFrame 类型。

    *,
    # * 表示之后的参数都是关键字参数，不可以通过位置传递。

    axis: Axis = ...,
    # axis 参数用于指定操作的轴向，默认为 ... (根据具体函数决定，通常是 0 或 1)。

    join: str = ...,
    # join 参数用于指定连接的方式，通常是字符串类型的选项，比如 'inner'、'outer' 等。

    ignore_index: bool = ...,
    # ignore_index 参数用于指定是否忽略索引，通常是一个布尔值。

    keys: Iterable[Hashable] | None = ...,
    # keys 参数用于指定用于连接的键，可以是一个可迭代对象，其中元素是可哈希类型，或者为 None。

    levels=...,
    # levels 参数用于多级索引操作时指定级别，具体类型根据函数决定。

    names: list[HashableT] | None = ...,
    # names 参数用于指定多级索引级别的名称，可以是一个列表，其中元素是可哈希类型，或者为 None。

    verify_integrity: bool = ...,
    # verify_integrity 参数用于指定是否验证结果的完整性，通常是一个布尔值。

    sort: bool = ...,
    # sort 参数用于指定是否对结果进行排序，通常是一个布尔值。

    copy: bool | lib.NoDefault = ...,
    # copy 参数用于指定是否复制数据以避免修改原始数据，或者是一个特定的类型（如 lib.NoDefault），具体根据函数的需求。
def concat(
    objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],
    *,
    axis: Axis = 0,  # 指定沿着哪个轴进行连接，默认为 0，即沿着行的方向连接
    join: str = "outer",  # 指定如何处理其他轴上的索引，默认为 'outer'，即保留所有索引
    ignore_index: bool = False,  # 如果为 True，则不使用连接轴上的索引值，默认为 False
    keys: Iterable[Hashable] | None = None,  # 如果传入多级索引，应包含元组。用传入的 keys 构建多级索引
    levels=None,  # 指定构建 MultiIndex 时要使用的特定级别，默认为 None，将从 keys 推断
    names: list[HashableT] | None = None,  # 结果多级索引的级别名称，默认为 None
    verify_integrity: bool = False,  # 检查新连接的轴是否包含重复项，默认为 False
    sort: bool = False,  # 是否对非连接轴排序，默认为 False
    copy: bool | lib.NoDefault = lib.no_default,  # 是否复制对象，默认为 lib.no_default
) -> DataFrame | Series:
    """
    沿着指定轴连接 pandas 对象。

    允许在其他轴上进行可选的集合逻辑操作。

    如果标签在传递的轴编号上相同（或重叠），还可以在连接轴上添加一层层次化索引，这在处理数据时可能会很有用。

    参数
    ----------
    objs : 可迭代对象或映射，包含 Series 或 DataFrame 对象
        如果传入映射，则键将用作 `keys` 参数，除非显式传入 `keys` 参数，此时将选择值（参见下文）。任何 None 对象都将被静默丢弃，除非它们全部为 None，否则会引发 ValueError。
    axis : {0/'index', 1/'columns'}，默认为 0
        进行连接的轴。
    join : {'inner', 'outer'}，默认为 'outer'
        如何处理其他轴（或轴）上的索引。
    ignore_index : bool，默认为 False
        如果为 True，则不使用连接轴上的索引值。结果轴将被标记为 0, ..., n-1。这在连接对象时很有用，其中连接轴没有有意义的索引信息。注意，其他轴上的索引值仍将在连接中受到尊重。
    keys : 序列，默认为 None
        如果传入多级索引，应包含元组。用传入的 keys 作为最外层级别构建层次化索引。
    levels : 序列列表，默认为 None
        用于构建 MultiIndex 的特定级别（唯一值）。否则，它们将从 keys 推断。
    names : 列表，默认为 None
        结果层次化索引中各级别的名称。
    verify_integrity : bool，默认为 False
        检查新连接的轴是否包含重复项。这相对于实际数据连接来说可能非常昂贵。
    sort : bool，默认为 False
        对非连接轴进行排序。唯一的例外是当非连接轴是 DatetimeIndex 且 join='outer' 且轴尚未对齐时。在这种情况下，非连接轴总是按词典顺序排序。
    copy : bool 或 lib.NoDefault，默认为 lib.no_default
        是否复制对象。
    """
    copy : bool, default False
        如果为 False，则避免不必要的数据复制。

        .. note::
            `copy` 关键字将在 pandas 3.0 中改变行为。
            `Copy-on-Write
            <https://pandas.pydata.org/docs/dev/user_guide/copy_on_write.html>`__
            将默认启用，这意味着所有带有 `copy` 关键字的方法将使用延迟复制机制来推迟复制并忽略 `copy` 关键字。`copy` 关键字将在未来的版本中移除。

            您可以通过启用 copy on write 来获得未来的行为和改进 ``pd.options.mode.copy_on_write = True``。

        .. deprecated:: 3.0.0

    Returns
    -------
    object, type of objs
        在沿索引（axis=0）连接所有 ``Series`` 时，返回一个 ``Series``。
        当 ``objs`` 包含至少一个 ``DataFrame`` 时，返回一个 ``DataFrame``。
        在沿列（axis=1）连接时，返回一个 ``DataFrame``。

    See Also
    --------
    DataFrame.join : 使用索引连接 DataFrames。
    DataFrame.merge : 按索引或列合并 DataFrames。

    Notes
    -----
    keys、levels 和 names 参数都是可选的。

    可以在 `这里
    <https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html>`__ 找到此方法与其他组合 pandas 对象工具的演示。

    不推荐使用 for 循环逐行构建 DataFrame。应该构建行列表并一次性使用 concat 创建 DataFrame。

    Examples
    --------
    合并两个 ``Series``。

    >>> s1 = pd.Series(["a", "b"])
    >>> s2 = pd.Series(["c", "d"])
    >>> pd.concat([s1, s2])
    0    a
    1    b
    0    c
    1    d
    dtype: object

    通过将 ``ignore_index`` 选项设置为 ``True``，清除现有索引并在结果中重置它。

    >>> pd.concat([s1, s2], ignore_index=True)
    0    a
    1    b
    2    c
    3    d
    dtype: object

    使用 ``keys`` 选项在数据的最外层级添加层次化索引。

    >>> pd.concat([s1, s2], keys=["s1", "s2"])
    s1  0    a
        1    b
    s2  0    c
        1    d
    dtype: object

    使用 ``names`` 选项为创建的索引键标记名称。

    >>> pd.concat([s1, s2], keys=["s1", "s2"], names=["Series name", "Row ID"])
    Series name  Row ID
    s1           0         a
                 1         b
    s2           0         c
                 1         d
    dtype: object

    合并具有相同列的两个 ``DataFrame`` 对象。

    >>> df1 = pd.DataFrame([["a", 1], ["b", 2]], columns=["letter", "number"])
    >>> df1
      letter  number
    0      a       1
    1      b       2
    >>> df2 = pd.DataFrame([["c", 3], ["d", 4]], columns=["letter", "number"])
    >>> df2
      letter  number
    0      c       3
    1      d       4
    # 组合多个 DataFrame 对象，按行连接它们，返回结果。
    # 如果列名有重叠，非交集部分将填充 NaN 值。
    
    >>> pd.concat([df1, df2])
      letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4
    
    # 组合多个 DataFrame 对象，其中 df3 包含了额外的 "animal" 列。
    # 这些对象按行连接，返回结果。
    >>> df3 = pd.DataFrame(
    ...     [["c", 3, "cat"], ["d", 4, "dog"]], columns=["letter", "number", "animal"]
    ... )
    >>> df3
      letter  number animal
    0      c       3    cat
    1      d       4    dog
    >>> pd.concat([df1, df3], sort=False)
      letter  number animal
    0      a       1    NaN
    1      b       2    NaN
    0      c       3    cat
    1      d       4    dog
    
    # 仅组合多个 DataFrame 对象中列名有重叠的部分，返回结果。
    # 使用 join="inner" 控制关键字参数。
    >>> pd.concat([df1, df3], join="inner")
      letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4
    
    # 水平连接多个 DataFrame 对象，沿着 x 轴（列方向）。
    >>> df4 = pd.DataFrame(
    ...     [["bird", "polly"], ["monkey", "george"]], columns=["animal", "name"]
    ... )
    >>> pd.concat([df1, df4], axis=1)
      letter  number  animal    name
    0      a       1    bird   polly
    1      b       2  monkey  george
    
    # 使用 verify_integrity 选项，防止结果包含重复的索引值。
    >>> df5 = pd.DataFrame([1], index=["a"])
    >>> df5
       0
    a  1
    >>> df6 = pd.DataFrame([2], index=["a"])
    >>> df6
       0
    a  2
    >>> pd.concat([df5, df6], verify_integrity=True)
    Traceback (most recent call last):
        ...
    ValueError: Indexes have overlapping values: ['a']
    
    # 将单行数据追加到 DataFrame 对象的末尾。
    >>> df7 = pd.DataFrame({"a": 1, "b": 2}, index=[0])
    >>> df7
       a  b
    0  1  2
    >>> new_row = pd.Series({"a": 3, "b": 4})
    >>> new_row
    a    3
    b    4
    dtype: int64
    >>> pd.concat([df7, new_row.to_frame().T], ignore_index=True)
       a  b
    0  1  2
    1  3  4
    
    """
    if copy is not lib.no_default:
        warnings.warn(
            "The copy keyword is deprecated and will be removed in a future "
            "version. Copy-on-Write is active in pandas since 3.0 which utilizes "
            "a lazy copy mechanism that defers copies until necessary. Use "
            ".copy() to make an eager copy if necessary.",
            DeprecationWarning,
            stacklevel=find_stack_level(),
        )
    
    # 创建 _Concatenator 对象，用于执行 DataFrame 对象的连接操作。
    op = _Concatenator(
        objs,
        axis=axis,
        ignore_index=ignore_index,
        join=join,
        keys=keys,
        levels=levels,
        names=names,
        verify_integrity=verify_integrity,
        sort=sort,
    )
    
    # 返回连接操作的结果。
    return op.get_result()
    """
    Orchestrates a concatenation operation for BlockManagers
    """

    sort: bool  # 类型注释，表示 sort 属性为布尔类型

    def __init__(
        self,
        objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],
        axis: Axis = 0,
        join: str = "outer",
        keys: Iterable[Hashable] | None = None,
        levels=None,
        names: list[HashableT] | None = None,
        ignore_index: bool = False,
        verify_integrity: bool = False,
        sort: bool = False,
    ) -> None:
        if isinstance(objs, (ABCSeries, ABCDataFrame, str)):
            raise TypeError(
                "first argument must be an iterable of pandas "
                f'objects, you passed an object of type "{type(objs).__name__}"'
            )

        if join == "outer":
            self.intersect = False  # 如果连接方式是 'outer'，设置 intersect 为 False
        elif join == "inner":
            self.intersect = True  # 如果连接方式是 'inner'，设置 intersect 为 True
        else:  # pragma: no cover
            raise ValueError(
                "Only can inner (intersect) or outer (union) join the other axis"
            )

        if not is_bool(sort):
            raise ValueError(
                f"The 'sort' keyword only accepts boolean values; {sort} was passed."
            )
        # Incompatible types in assignment (expression has type "Union[bool, bool_]",
        # variable has type "bool")
        self.sort = sort  # type: ignore[assignment] 设置对象的 sort 属性为传入的 sort 值，忽略类型检查

        self.ignore_index = ignore_index  # 设置对象的 ignore_index 属性
        self.verify_integrity = verify_integrity  # 设置对象的 verify_integrity 属性

        objs, keys, ndims = _clean_keys_and_objs(objs, keys)  # 调用函数处理 objs 和 keys，获取清理后的结果

        # select an object to be our result reference
        sample, objs = _get_sample_object(
            objs, ndims, keys, names, levels, self.intersect
        )  # 调用函数选择一个对象作为结果的参考对象

        # Standardize axis parameter to int
        if sample.ndim == 1:
            from pandas import DataFrame

            axis = DataFrame._get_axis_number(axis)  # 如果 sample 是一维的 Series，则转换 axis 为整数
            self._is_frame = False  # 标记当前对象不是 DataFrame
            self._is_series = True  # 标记当前对象是 Series
        else:
            axis = sample._get_axis_number(axis)  # 否则，转换 axis 为整数
            self._is_frame = True  # 标记当前对象是 DataFrame
            self._is_series = False  # 标记当前对象不是 Series

            # Need to flip BlockManager axis in the DataFrame special case
            axis = sample._get_block_manager_axis(axis)  # 如果是 DataFrame，需要调整 BlockManager 的轴向

        # if we have mixed ndims, then convert to highest ndim
        # creating column numbers as needed
        if len(ndims) > 1:
            objs = self._sanitize_mixed_ndim(objs, sample, ignore_index, axis)
            # 处理混合维度的情况，将 objs 转换为最高维度的类型，并按需创建列号

        self.objs = objs  # 设置对象的 objs 属性为处理后的 objs

        # note: this is the BlockManager axis (since DataFrame is transposed)
        self.bm_axis = axis  # 设置对象的 bm_axis 属性为处理后的 axis
        self.axis = 1 - self.bm_axis if self._is_frame else 0  # 设置对象的 axis 属性为处理后的 axis
        self.keys = keys  # 设置对象的 keys 属性
        self.names = names or getattr(keys, "names", None)  # 设置对象的 names 属性
        self.levels = levels  # 设置对象的 levels 属性

    def _sanitize_mixed_ndim(
        self,
        objs: list[Series | DataFrame],
        sample: Series | DataFrame,
        ignore_index: bool,
        axis: AxisInt,
    ) -> list[Series | DataFrame]:
        # 返回类型声明为列表，其元素可以是 Series 或 DataFrame

        # 创建一个空列表来存储处理后的对象
        new_objs = []

        # 初始化当前列号和最大维度为样本数据的维度
        current_column = 0
        max_ndim = sample.ndim

        # 遍历传入的对象列表
        for obj in objs:
            # 获取当前对象的维度
            ndim = obj.ndim

            # 如果当前对象的维度与样本数据的最大维度相同，则跳过
            if ndim == max_ndim:
                pass

            # 如果当前对象的维度不是样本数据的最大维度减一，则抛出数值错误异常
            elif ndim != max_ndim - 1:
                raise ValueError(
                    "cannot concatenate unaligned mixed dimensional NDFrame objects"
                )

            else:
                # 尝试获取对象的名称属性，如果没有则设置为 None
                name = getattr(obj, "name", None)

                # 如果忽略索引或者名称为 None，则根据轴向设置名称
                if ignore_index or name is None:
                    if axis == 1:
                        # 如果是行合并，则需要所有内容对齐，因此将名称设置为 0
                        name = 0
                    else:
                        # 如果是列合并，则需要确保 Series 具有唯一名称，递增列号以作为名称
                        name = current_column
                        current_column += 1

                    # 根据对象类型重新构造对象，确保不复制数据
                    obj = sample._constructor(obj, copy=False)

                    # 如果对象是 DataFrame 类型，则将列名称设置为范围从 name 到 name + 1
                    if isinstance(obj, ABCDataFrame):
                        obj.columns = range(name, name + 1, 1)
                else:
                    # 否则，根据对象的名称构造一个新的 DataFrame 对象
                    obj = sample._constructor({name: obj}, copy=False)

            # 将处理后的对象添加到新对象列表中
            new_objs.append(obj)

        # 返回处理后的对象列表
        return new_objs
    # 获取结果方法，用于合并或处理数据块
    def get_result(self):
        cons: Callable[..., DataFrame | Series]  # 类型注解，cons 是一个 DataFrame 或 Series 的构造函数
        sample: DataFrame | Series  # sample 可以是 DataFrame 或 Series

        # 仅处理 Series 类型的情况
        if self._is_series:
            sample = cast("Series", self.objs[0])  # 将第一个对象强制转换为 Series 类型

            # 如果按照 bm_axis = 0 方向堆叠块
            if self.bm_axis == 0:
                name = com.consensus_name_attr(self.objs)  # 根据 objs 列表获取共识的名称
                cons = sample._constructor  # 使用 sample 的构造函数来创建新的对象

                arrs = [ser._values for ser in self.objs]  # 提取每个 Series 对象的值数组

                # 在 axis=0 方向上连接数组
                res = concat_compat(arrs, axis=0)

                new_index: Index
                if self.ignore_index:
                    # 如果忽略索引，使用默认的索引
                    new_index = default_index(len(res))
                else:
                    new_index = self.new_axes[0]  # 否则使用指定的新索引

                mgr = type(sample._mgr).from_array(res, index=new_index)  # 从数组 res 创建新的数据管理器

                result = sample._constructor_from_mgr(mgr, axes=mgr.axes)  # 使用数据管理器创建结果对象
                result._name = name  # 设置结果对象的名称
                return result.__finalize__(self, method="concat")  # 返回最终结果对象并进行最终化处理

            # 合并为 DataFrame 中的列
            else:
                data = dict(enumerate(self.objs))  # 创建一个字典，包含对象的枚举编号和对应的对象

                # GH28330: 通过 concat 保留子类化对象
                cons = sample._constructor_expanddim  # 使用 sample 的扩展维度构造函数

                index, columns = self.new_axes  # 获取新索引和列名
                df = cons(data, index=index, copy=False)  # 使用 data 创建一个新的 DataFrame
                df.columns = columns  # 设置 DataFrame 的列名
                return df.__finalize__(self, method="concat")  # 返回最终结果 DataFrame 并进行最终化处理

        # 处理块管理器的合并情况
        else:
            sample = cast("DataFrame", self.objs[0])  # 将第一个对象强制转换为 DataFrame 类型

            mgrs_indexers = []
            for obj in self.objs:
                indexers = {}
                for ax, new_labels in enumerate(self.new_axes):
                    # ::-1 将 BlockManager 的轴转换为 DataFrame 的轴
                    if ax == self.bm_axis:
                        # 在 concat 轴上不进行重新索引
                        continue

                    # 1-ax 将 BlockManager 轴转换为 DataFrame 轴
                    obj_labels = obj.axes[1 - ax]
                    if not new_labels.equals(obj_labels):
                        indexers[ax] = obj_labels.get_indexer(new_labels)

                mgrs_indexers.append((obj._mgr, indexers))

            # 使用 concatenate_managers 函数合并管理器
            new_data = concatenate_managers(
                mgrs_indexers, self.new_axes, concat_axis=self.bm_axis, copy=False
            )

            # 从新数据创建新的对象，并最终化处理
            out = sample._constructor_from_mgr(new_data, axes=new_data.axes)
            return out.__finalize__(self, method="concat")

    @cache_readonly
    # 返回一个索引列表，用于合并操作的轴
    def new_axes(self) -> list[Index]:
        # 如果当前对象是系列且合并轴是1，则维度为2
        if self._is_series and self.bm_axis == 1:
            ndim = 2
        else:
            # 否则取第一个对象的维度作为维度
            ndim = self.objs[0].ndim
        # 返回一个列表，包含各个轴的索引
        return [
            self._get_concat_axis  # 如果索引是合并轴，则返回合并轴的索引
            if i == self.bm_axis  # 判断当前索引是否是合并轴
            else get_objs_combined_axis(  # 否则，调用函数获取对象组合后的轴索引
                self.objs,
                axis=self.objs[0]._get_block_manager_axis(i),  # 传入轴参数和对象管理轴索引
                intersect=self.intersect,  # 传入交集参数
                sort=self.sort,  # 传入排序参数
            )
            for i in range(ndim)  # 对于每个维度范围内的索引，生成相应的轴索引
        ]

    @cache_readonly
    # 获取用于连接轴的索引
    def _get_concat_axis(self) -> Index:
        """
        Return index to be used along concatenation axis.
        """
        # 如果当前对象是系列
        if self._is_series:
            # 如果合并轴是0，则获取每个对象的索引
            if self.bm_axis == 0:
                indexes = [x.index for x in self.objs]
            # 如果忽略索引，则返回默认索引
            elif self.ignore_index:
                idx = default_index(len(self.objs))
                return idx
            # 如果未指定键和名称
            elif self.keys is None:
                names: list[Hashable] = [None] * len(self.objs)
                num = 0
                has_names = False
                # 遍历对象，为无名称的对象分配编号或名称
                for i, x in enumerate(self.objs):
                    if x.ndim != 1:
                        raise TypeError(
                            f"Cannot concatenate type 'Series' with "
                            f"object of type '{type(x).__name__}'"
                        )
                    if x.name is not None:
                        names[i] = x.name
                        has_names = True
                    else:
                        names[i] = num
                        num += 1
                # 如果存在名称，则返回带名称的索引，否则返回默认索引
                if has_names:
                    return Index(names)
                else:
                    return default_index(len(self.objs))
            # 如果有指定的键
            else:
                return ensure_index(self.keys).set_names(self.names)
        else:
            # 否则获取每个对象在指定轴上的索引
            indexes = [x.axes[self.axis] for x in self.objs]

        # 如果忽略索引，则返回默认索引
        if self.ignore_index:
            idx = default_index(sum(len(i) for i in indexes))
            return idx

        # 如果未指定键
        if self.keys is None:
            # 如果存在级别，抛出错误
            if self.levels is not None:
                raise ValueError("levels supported only when keys is not None")
            # 否则进行索引的连接操作
            concat_axis = _concat_indexes(indexes)
        else:
            # 否则进行多级索引的连接操作
            concat_axis = _make_concat_multiindex(
                indexes, self.keys, self.levels, self.names
            )

        # 如果需要验证完整性，检查连接轴是否唯一，否则抛出错误
        if self.verify_integrity:
            if not concat_axis.is_unique:
                overlap = concat_axis[concat_axis.duplicated()].unique()
                raise ValueError(f"Indexes have overlapping values: {overlap}")

        # 返回连接后的轴索引
        return concat_axis
# 清理对象列表和键，确保只保留非空的 Series 和 DataFrame 对象
def _clean_keys_and_objs(
    objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],  # 可迭代对象或映射表，包含 Series 或 DataFrame
    keys,  # 键列表或 None
) -> tuple[list[Series | DataFrame], Index | None, set[int]]:  # 返回值为清理后的对象列表、键的 Index 或 None、对象的 ndim 集合
    """
    Returns
    -------
    clean_objs : list[Series | DataFrame]
        LIst of DataFrame and Series with Nones removed.
    keys : Index | None
        None if keys was None
        Index if objs was a Mapping or keys was not None. Filtered where objs was None.
    ndim : set[int]
        Unique .ndim attribute of obj encountered.
    """
    if isinstance(objs, abc.Mapping):  # 如果 objs 是映射表
        if keys is None:
            keys = objs.keys()  # 如果 keys 为 None，则使用 objs 的键
        objs_list = [objs[k] for k in keys]  # 根据 keys 获取对应的值组成列表
    else:
        objs_list = list(objs)  # 否则将 objs 转换为列表

    if len(objs_list) == 0:
        raise ValueError("No objects to concatenate")  # 如果 objs_list 为空，则抛出 ValueError

    if keys is not None:
        if not isinstance(keys, Index):
            keys = Index(keys)  # 如果 keys 不是 Index 类型，则转换为 Index
        if len(keys) != len(objs_list):
            # GH#43485
            raise ValueError(
                f"The length of the keys ({len(keys)}) must match "
                f"the length of the objects to concatenate ({len(objs_list)})"
            )  # 如果 keys 的长度与 objs_list 的长度不匹配，则抛出 ValueError

    # GH#1649
    key_indices = []  # 存储有效对象索引的列表
    clean_objs = []  # 存储清理后的有效对象列表
    ndims = set()  # 存储对象的 ndim 属性的集合
    for i, obj in enumerate(objs_list):
        if obj is None:
            continue  # 如果 obj 是 None，则跳过
        elif isinstance(obj, (ABCSeries, ABCDataFrame)):  # 如果 obj 是 Series 或 DataFrame 类型
            key_indices.append(i)  # 记录有效对象的索引
            clean_objs.append(obj)  # 将有效对象添加到 clean_objs 中
            ndims.add(obj.ndim)  # 添加对象的 ndim 属性到集合中
        else:
            msg = (
                f"cannot concatenate object of type '{type(obj)}'; "
                "only Series and DataFrame objs are valid"
            )
            raise TypeError(msg)  # 如果 obj 不是 Series 或 DataFrame 类型，则抛出 TypeError

    if keys is not None and len(key_indices) < len(keys):
        keys = keys.take(key_indices)  # 根据有效对象的索引更新 keys

    if len(clean_objs) == 0:
        raise ValueError("All objects passed were None")  # 如果 clean_objs 为空，则抛出 ValueError

    return clean_objs, keys, ndims  # 返回清理后的对象列表、更新后的 keys、对象的 ndim 集合


# 获取样本对象和非空对象列表
def _get_sample_object(
    objs: list[Series | DataFrame],  # Series 或 DataFrame 对象的列表
    ndims: set[int],  # 对象的 ndim 属性集合
    keys,  # 可选的键列表
    names,  # 可选的名称
    levels,  # 可选的级别
    intersect: bool,  # 布尔类型，指示是否求交集
) -> tuple[Series | DataFrame, list[Series | DataFrame]]:  # 返回值为样本对象和非空对象列表
    # get the sample
    # want the highest ndim that we have, and must be non-empty
    # unless all objs are empty
    if len(ndims) > 1:  # 如果 ndims 中有多个不同的 ndim 值
        max_ndim = max(ndims)  # 获取最大的 ndim 值
        for obj in objs:
            if obj.ndim == max_ndim and sum(obj.shape):  # 如果对象的 ndim 与最大值相等且对象不为空
                return obj, objs  # 返回该对象和原始对象列表
    elif keys is None and names is None and levels is None and not intersect:
        # filter out the empties if we have not multi-index possibilities
        # note to keep empty Series as it affect to result columns / name
        if ndims.pop() == 2:
            non_empties = [obj for obj in objs if sum(obj.shape)]
        else:
            non_empties = objs

        if len(non_empties):
            return non_empties[0], non_empties  # 返回第一个非空对象和非空对象列表

    return objs[0], objs  # 如果以上条件都不满足，则返回第一个对象和原始对象列表


# 连接索引对象列表成一个 Index 对象
def _concat_indexes(indexes) -> Index:
    return indexes[0].append(indexes[1:])
# 验证给定的索引列表中的每个级别是否唯一，如果不唯一则抛出值错误异常
def validate_unique_levels(levels: list[Index]) -> None:
    for level in levels:
        if not level.is_unique:
            raise ValueError(f"Level values not unique: {level.tolist()}")

# 创建一个多级索引对象
def _make_concat_multiindex(indexes, keys, levels=None, names=None) -> MultiIndex:
    # 如果 levels 为 None 并且 keys[0] 是元组，或者 levels 不为 None 并且其长度大于 1
    if (levels is None and isinstance(keys[0], tuple)) or (
        levels is not None and len(levels) > 1
    ):
        # 将 keys 中的元素解压缩为列表，并根据需要设置 names
        zipped = list(zip(*keys))
        if names is None:
            names = [None] * len(zipped)

        # 如果 levels 为 None，则从 zipped 中提取因子化的 levels
        if levels is None:
            _, levels = factorize_from_iterables(zipped)
        else:
            # 否则，确保 levels 中每个元素都是索引对象，并验证其唯一性
            levels = [ensure_index(x) for x in levels]
            validate_unique_levels(levels)
    else:
        # 否则，将 keys 包装成列表 zipped，并根据需要设置 names
        zipped = [keys]
        if names is None:
            names = [None]

        # 如果 levels 为 None，则将其设置为 ensure_index(keys) 的唯一值
        if levels is None:
            levels = [ensure_index(keys).unique()]
        else:
            # 否则，确保 levels 中每个元素都是索引对象，并验证其唯一性
            levels = [ensure_index(x) for x in levels]
            validate_unique_levels(levels)

    # 如果 indexes 中的所有索引对象不相同
    if not all_indexes_same(indexes):
        codes_list = []

        # 对于 zipped 中的每个 hlevel 和对应的 level
        for hlevel, level in zip(zipped, levels):
            to_concat = []
            # 如果 hlevel 是索引对象且与 level 相等
            if isinstance(hlevel, Index) and hlevel.equals(level):
                # 计算每个索引对象的长度，并生成相应的重复代码
                lens = [len(idx) for idx in indexes]
                codes_list.append(np.repeat(np.arange(len(hlevel)), lens))
            else:
                # 否则，对于每个 key 和对应的 index
                for key, index in zip(hlevel, indexes):
                    # 找到匹配的代码，包括匹配的 NaN 值作为相等处理
                    mask = (isna(level) & isna(key)) | (level == key)
                    if not mask.any():
                        raise ValueError(f"Key {key} not in level {level}")
                    i = np.nonzero(mask)[0][0]

                    to_concat.append(np.repeat(i, len(index)))
                codes_list.append(np.concatenate(to_concat))

        # 合并索引对象列表
        concat_index = _concat_indexes(indexes)

        # 将这些内容添加到 levels 和 codes_list 的末尾
        if isinstance(concat_index, MultiIndex):
            levels.extend(concat_index.levels)
            codes_list.extend(concat_index.codes)
        else:
            codes, categories = factorize_from_iterable(concat_index)
            levels.append(categories)
            codes_list.append(codes)

        # 如果 names 的长度与 levels 的长度相同，则转换为列表
        if len(names) == len(levels):
            names = list(names)
        else:
            # 否则，确保传递的所有索引具有相同数量的级别
            if not len({idx.nlevels for idx in indexes}) == 1:
                raise AssertionError(
                    "Cannot concat indices that do not have the same number of levels"
                )

            # 还要考虑 names
            names = list(names) + list(get_unanimous_names(*indexes))

        # 返回一个新的 MultiIndex 对象，使用给定的 levels、codes_list 和 names，跳过完整性验证
        return MultiIndex(
            levels=levels, codes=codes_list, names=names, verify_integrity=False
        )

    # 如果 indexes 中只有一个索引对象，则直接返回该索引对象
    new_index = indexes[0]
    n = len(new_index)
    # 计算索引列表的长度
    kpieces = len(indexes)

    # 复制名称和级别列表
    new_names = list(names)
    new_levels = list(levels)

    # 构建新的编码列表
    new_codes = []

    # 针对性能优化做一些处理

    # 遍历压缩后的数据和级别列表
    for hlevel, level in zip(zipped, levels):
        # 确保 hlevel 是一个有效的索引对象
        hlevel_index = ensure_index(hlevel)
        # 获取 level 中 hlevel_index 的索引映射
        mapped = level.get_indexer(hlevel_index)

        # 检查是否有未找到的映射值
        mask = mapped == -1
        if mask.any():
            # 抛出值未在传入级别中找到的错误
            raise ValueError(
                f"Values not found in passed level: {hlevel_index[mask]!s}"
            )

        # 将映射结果重复 n 次添加到新编码列表中
        new_codes.append(np.repeat(mapped, n))

    # 如果 new_index 是一个 MultiIndex 对象
    if isinstance(new_index, MultiIndex):
        # 将新的级别列表扩展到包含 new_index 的级别
        new_levels.extend(new_index.levels)
        # 将 new_index 的编码块 np.tile(lab, kpieces) 添加到新的编码列表中
        new_codes.extend(np.tile(lab, kpieces) for lab in new_index.codes)
    else:
        # 将 new_index 的唯一值作为新的单一级别添加到 new_levels
        new_levels.append(new_index.unique())
        # 获取 new_index 中唯一值的索引映射，并重复 kpieces 次，添加到新编码列表中
        single_codes = new_index.unique().get_indexer(new_index)
        new_codes.append(np.tile(single_codes, kpieces))

    # 如果新名称列表的长度小于新级别列表的长度，则扩展新名称列表
    if len(new_names) < len(new_levels):
        new_names.extend(new_index.names)

    # 返回一个新的 MultiIndex 对象，包含新的级别、编码和名称列表，不验证完整性
    return MultiIndex(
        levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
    )
```