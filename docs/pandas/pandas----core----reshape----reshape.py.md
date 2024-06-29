# `D:\src\scipysrc\pandas\pandas\core\reshape\reshape.py`

```
from __future__ import annotations

import itertools
from typing import (
    TYPE_CHECKING,
    cast,
    overload,
)
import warnings

import numpy as np

from pandas._config.config import get_option

import pandas._libs.reshape as libreshape
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import (
    find_common_type,
    maybe_promote,
)
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_1d_only_ea_dtype,
    is_integer,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import notna

import pandas.core.algorithms as algos
from pandas.core.algorithms import (
    factorize,
    unique,
)
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    default_index,
)
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.sorting import (
    compress_group_index,
    decons_obs_group_ids,
    get_compressed_ids,
    get_group_index,
    get_group_index_sorter,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        Level,
        npt,
    )

    from pandas.core.arrays import ExtensionArray
    from pandas.core.indexes.frozen import FrozenList


class _Unstacker:
    """
    Helper class to unstack data / pivot with multi-level index

    Parameters
    ----------
    index : MultiIndex
        多级索引对象，用于进行数据透视和展开操作
    level : int or str, default last level
        层级参数，指定要“展开”的层级。可以是层级的名称。
    fill_value : scalar, optional
        缺失值填充的默认值，如果子组没有相同的标签集合。默认情况下，缺失值将使用该数据类型的默认填充值填充，例如 float 使用 NaN，datetime 使用 NaT 等。对于整数类型，默认情况下数据将转换为 float，并且缺失值将被设置为 NaN。
    constructor : object
        用于创建展开响应的 Pandas DataFrame 或其子类。如果为 None，则使用 DataFrame。
    sort : bool, default True
        是否对结果进行排序

    Examples
    --------
    >>> index = pd.MultiIndex.from_tuples(
    ...     [("one", "a"), ("one", "b"), ("two", "a"), ("two", "b")]
    ... )
    >>> s = pd.Series(np.arange(1, 5, dtype=np.int64), index=index)
    >>> s
    one  a    1
         b    2
    two  a    3
         b    4
    dtype: int64

    >>> s.unstack(level=-1)
         a  b
    one  1  2
    two  3  4

    >>> s.unstack(level=0)
       one  two
    a    1    3
    b    2    4

    Returns
    -------
    unstacked : DataFrame
        展开后的 DataFrame 对象
    """

    def __init__(
        self, index: MultiIndex, level: Level, constructor, sort: bool = True
    ):
        self.index = index
        self.level = level
        self.constructor = constructor
        self.sort = sort
    ) -> None:
        # 初始化方法，接受构造函数和排序标志
        self.constructor = constructor
        self.sort = sort

        # 移除未使用的索引级别
        self.index = index.remove_unused_levels()

        # 获取当前级别的索引级别编号
        self.level = self.index._get_level_number(level)

        # 当索引中包含NaN时，需要将级别/步长提升1
        self.lift = 1 if -1 in self.index.codes[self.level] else 0

        # 注意：下面的"pop"会就地修改这些属性
        # 创建新的索引级别列表和名称列表
        self.new_index_levels = list(self.index.levels)
        self.new_index_names = list(self.index.names)

        # 弹出当前级别的名称和级别
        self.removed_name = self.new_index_names.pop(self.level)
        self.removed_level = self.new_index_levels.pop(self.level)
        self.removed_level_full = index.levels[self.level]

        # 如果不排序，则处理唯一的代码
        if not self.sort:
            unique_codes = unique(self.index.codes[self.level])
            self.removed_level = self.removed_level.take(unique_codes)
            self.removed_level_full = self.removed_level_full.take(unique_codes)

        # 如果启用了性能警告选项
        if get_option("performance_warnings"):
            # Bug fix GH 20601
            # 如果数据框过大，唯一索引组合的数量将导致在Windows环境下int32溢出
            # 我们希望在发生这种情况之前检查并引发警告
            num_rows = max(index_level.size for index_level in self.new_index_levels)
            num_columns = self.removed_level.size

            # GH20601: 如果单元格数量过高，这将强制溢出
            # GH 26314: 对于许多用户，先前引发的ValueError过于严格
            num_cells = num_rows * num_columns
            if num_cells > np.iinfo(np.int32).max:
                warnings.warn(
                    f"The following operation may generate {num_cells} cells "
                    f"in the resulting pandas object.",
                    PerformanceWarning,
                    stacklevel=find_stack_level(),
                )

        # 生成选择器
        self._make_selectors()

    @cache_readonly
    def _indexer_and_to_sort(
        self,
    ) -> tuple[
        npt.NDArray[np.intp],
        list[np.ndarray],  # 每个都有某些带符号整数dtype
    ]:
        v = self.level

        # 复制索引的代码列表
        codes = list(self.index.codes)
        if not self.sort:
            # 如果不排序，创建新的代码，考虑到标签已经排序
            codes = [factorize(code)[0] for code in codes]
        levs = list(self.index.levels)
        # 要排序的列表，将当前级别放在最后
        to_sort = codes[:v] + codes[v + 1 :] + [codes[v]]
        sizes = tuple(len(x) for x in levs[:v] + levs[v + 1 :] + [levs[v]])

        # 获取压缩的ID以及组索引排序器
        comp_index, obs_ids = get_compressed_ids(to_sort, sizes)
        ngroups = len(obs_ids)

        # 获取组索引排序器
        indexer = get_group_index_sorter(comp_index, ngroups)
        return indexer, to_sort

    @cache_readonly
    def sorted_labels(self) -> list[np.ndarray]:
        # 获取索引器和要排序的内容
        indexer, to_sort = self._indexer_and_to_sort
        # 如果排序，则返回排序后的标签列表
        if self.sort:
            return [line.take(indexer) for line in to_sort]
        # 否则返回未排序的内容
        return to_sort
    # 使用传入的 values 数组和预先计算的 indexer（索引器），通过 algos.take_nd 函数按指定索引取值，生成排序后的数组
    def _make_sorted_values(self, values: np.ndarray) -> np.ndarray:
        indexer, _ = self._indexer_and_to_sort
        sorted_values = algos.take_nd(values, indexer, axis=0)
        return sorted_values

    # 生成选择器数组，并存储到实例属性中
    def _make_selectors(self) -> None:
        new_levels = self.new_index_levels

        # 生成掩码（mask）
        remaining_labels = self.sorted_labels[:-1]
        level_sizes = tuple(len(x) for x in new_levels)

        # 调用 get_compressed_ids 函数获取压缩后的索引和观测 IDs
        comp_index, obs_ids = get_compressed_ids(remaining_labels, level_sizes)
        ngroups = len(obs_ids)

        # 确保 comp_index 是平台整数类型
        comp_index = ensure_platform_int(comp_index)

        # 计算步长（stride），并设置实例的完整形状（full_shape）
        stride = self.index.levshape[self.level] + self.lift
        self.full_shape = ngroups, stride

        # 根据 selector 公式生成掩码（mask）数组
        selector = self.sorted_labels[-1] + stride * comp_index + self.lift
        mask = np.zeros(np.prod(self.full_shape), dtype=bool)
        mask.put(selector, True)

        # 检查掩码中是否有重复值，如果有则抛出 ValueError 异常
        if mask.sum() < len(self.index):
            raise ValueError("Index contains duplicate entries, cannot reshape")

        # 存储压缩索引和掩码到实例属性中
        self.group_index = comp_index
        self.mask = mask

        # 如果需要排序，则设置压缩器（compressor）
        if self.sort:
            self.compressor = comp_index.searchsorted(np.arange(ngroups))
        else:
            self.compressor = np.sort(np.unique(comp_index, return_index=True)[1])

    # 缓存属性装饰器，返回掩码数组是否全为 True 的布尔值
    @cache_readonly
    def mask_all(self) -> bool:
        return bool(self.mask.all())

    # 缓存属性装饰器，返回元组，包含新值数组和列的掩码数组
    @cache_readonly
    def arange_result(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
        # 我们将此结果缓存以便在 ExtensionBlock._unstack 中复用
        dummy_arr = np.arange(len(self.index), dtype=np.intp)
        new_values, mask = self.get_new_values(dummy_arr, fill_value=-1)
        return new_values, mask.any(0)
        # TODO: in all tests we have mask.any(0).all(); can we rely on that?

    # 获取处理后的 DataFrame 结果，用于 obj 对象和值列，如果需要填充则用 fill_value
    def get_result(self, obj, value_columns, fill_value) -> DataFrame:
        values = obj._values

        # 如果 values 是一维数组，则转换为二维数组
        if values.ndim == 1:
            values = values[:, np.newaxis]

        # 如果未提供 value_columns 并且 values 有多列，则抛出 ValueError 异常
        if value_columns is None and values.shape[1] != 1:  # pragma: no cover
            raise ValueError("must pass column labels for multi-column data")

        # 获取新值和不需要的掩码数组
        new_values, _ = self.get_new_values(values, fill_value)
        columns = self.get_new_columns(value_columns)
        index = self.new_index

        # 使用构造器创建新的 DataFrame 结果
        result = self.constructor(
            new_values, index=index, columns=columns, dtype=new_values.dtype, copy=False
        )

        # 检查 values 类型，设置 base 和 new_base
        if isinstance(values, np.ndarray):
            base, new_base = values.base, new_values.base
        elif isinstance(values, NDArrayBackedExtensionArray):
            base, new_base = values._ndarray.base, new_values._ndarray.base
        else:
            base, new_base = 1, 2  # type: ignore[assignment]

        # 如果 base 和 new_base 相同，则添加引用
        if base is new_base:
            # 只有当一个维度大小为 1 时才会执行到这里
            result._mgr.add_references(obj._mgr)

        # 返回结果 DataFrame
        return result
    def get_new_values(self, values, fill_value=None):
        # 如果传入的 values 是一维数组，则将其转换为二维数组，每行一个元素
        if values.ndim == 1:
            values = values[:, np.newaxis]

        # 调用对象内部方法，对传入的 values 进行排序
        sorted_values = self._make_sorted_values(values)

        # 获取完整的形状信息
        length, width = self.full_shape
        stride = values.shape[1]
        result_width = width * stride
        result_shape = (length, result_width)

        # 获取对象内部的 mask 和 mask_all
        mask = self.mask
        mask_all = self.mask_all

        # 如果 mask_all 为 True，并且 values 非空，则直接进行形状重塑
        if mask_all and len(values):
            # TODO: 在什么情况下可以确保 sorted_values 与 values 匹配？当这种情况成立时，我们可以进行切片而不是取值（尤其是对于 EAs）
            new_values = (
                sorted_values.reshape(length, width, stride)
                .swapaxes(1, 2)
                .reshape(result_shape)
            )
            new_mask = np.ones(result_shape, dtype=bool)
            return new_values, new_mask

        # 确定 values 的数据类型
        dtype = values.dtype

        # 如果 mask_all 为 True，则使用 values 的数据类型创建新的数组
        if mask_all:
            dtype = values.dtype
            new_values = np.empty(result_shape, dtype=dtype)
        else:
            # 如果数据类型是 ExtensionDtype，则特别处理
            if isinstance(dtype, ExtensionDtype):
                # GH#41875
                # 假设 fill_value 可以由该数据类型持有，不像非 EAs 的情况那样会提升
                cls = dtype.construct_array_type()
                new_values = cls._empty(result_shape, dtype=dtype)
                new_values[:] = fill_value
            else:
                # 否则，可能需要提升数据类型，并使用 fill_value 填充新数组
                dtype, fill_value = maybe_promote(dtype, fill_value)
                new_values = np.empty(result_shape, dtype=dtype)
                new_values.fill(fill_value)

        # 获取数据类型的名称，并创建相应形状的 mask 数组
        name = dtype.name
        new_mask = np.zeros(result_shape, dtype=bool)

        # 如果需要将 values 的数据类型转换为基本数据类型，则进行相应的转换
        if needs_i8_conversion(values.dtype):
            sorted_values = sorted_values.view("i8")
            new_values = new_values.view("i8")
        else:
            sorted_values = sorted_values.astype(name, copy=False)

        # 调用 libreshape 模块的 unstack 方法，填充 new_values 和 new_mask
        libreshape.unstack(
            sorted_values,
            mask.view("u1"),
            stride,
            length,
            width,
            new_values,
            new_mask.view("u1"),
        )

        # 如果需要重新构建数据类型（例如从 i8 转换为 datetime64），则进行相应的处理
        if needs_i8_conversion(values.dtype):
            # 视图转换为 datetime64，以便包装为 DatetimeArray 并使用 DTA 的视图方法
            new_values = new_values.view("M8[ns]")
            new_values = ensure_wrapped_if_datetimelike(new_values)
            new_values = new_values.view(values.dtype)

        # 返回填充好的 new_values 和 new_mask
        return new_values, new_mask
    # 获取新列的方法，用于处理索引的修改和重命名操作
    def get_new_columns(self, value_columns: Index | None):
        # 如果没有指定值列，则根据条件返回移除指定级别后的重命名结果
        if value_columns is None:
            if self.lift == 0:
                return self.removed_level._rename(name=self.removed_name)

            # 将移除的级别插入到索引的开头，并重命名为指定的名称
            lev = self.removed_level.insert(0, item=self.removed_level._na_value)
            return lev.rename(self.removed_name)

        # 计算步长和宽度
        stride = len(self.removed_level) + self.lift
        width = len(value_columns)
        # 创建一个传播器，用于生成新的代码数组
        propagator = np.repeat(np.arange(width), stride)

        # 新的级别列表，可能是冻结列表或索引列表
        new_levels: FrozenList | list[Index]

        # 如果值列是多重索引类型
        if isinstance(value_columns, MultiIndex):
            # 拼接新的级别列表，将移除的完整级别加入其中
            new_levels = value_columns.levels + (  # type: ignore[has-type]
                self.removed_level_full,
            )
            # 拼接新的名称列表，将移除的名称加入其中
            new_names = value_columns.names + (self.removed_name,)

            # 为每个代码数组生成新的代码，基于传播器
            new_codes = [lab.take(propagator) for lab in value_columns.codes]
        else:
            # 创建新的级别列表，包括值列和移除的完整级别
            new_levels = [
                value_columns,
                self.removed_level_full,
            ]
            # 创建新的名称列表，包括值列的名称和移除的名称
            new_names = [value_columns.name, self.removed_name]
            # 创建新的代码数组，基于传播器
            new_codes = [propagator]

        # 如果存在重复器，则将其添加到代码数组中
        repeater = self._repeater
        new_codes.append(np.tile(repeater, width))

        # 返回新的多重索引对象，包括级别、代码和名称列表
        return MultiIndex(
            levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
        )

    @cache_readonly
    def _repeater(self) -> np.ndarray:
        # 判断完整级别和移除级别的长度是否相等
        if len(self.removed_level_full) != len(self.removed_level):
            # 如果长度不等，通过完整级别索引器映射新的代码到原始级别
            repeater = self.removed_level_full.get_indexer(self.removed_level)
            # 如果有lift值，则在索引器中插入-1作为起始值
            if self.lift:
                repeater = np.insert(repeater, 0, -1)
        else:
            # 如果长度相等，则每个级别项仅使用一次
            stride = len(self.removed_level) + self.lift
            repeater = np.arange(stride) - self.lift

        # 返回生成的重复器数组
        return repeater

    @cache_readonly
    # 定义一个方法 `new_index`，返回值可以是 `MultiIndex` 或 `Index`
    def new_index(self) -> MultiIndex | Index:
        # 如果排序标志为真
        if self.sort:
            # 从排序后的标签中获取除最后一个元素外的所有元素
            labels = self.sorted_labels[:-1]
        else:
            # 否则，获取当前级别的索引编码，创建一个编码列表副本
            v = self.level
            codes = list(self.index.codes)
            labels = codes[:v] + codes[v + 1 :]
        
        # 使用压缩器对标签列表中的每个标签进行压缩操作，返回压缩后的结果编码列表
        result_codes = [lab.take(self.compressor) for lab in labels]

        # 构建新索引
        if len(self.new_index_levels) == 1:
            # 如果新索引级别只有一个
            level, level_codes = self.new_index_levels[0], result_codes[0]
            # 如果任何一个编码为 -1（表示缺失值），则将缺失值插入到该级别的末尾
            if (level_codes == -1).any():
                level = level.insert(len(level), level._na_value)
            # 获取指定编码的级别值，并将其重命名为新索引的名称
            return level.take(level_codes).rename(self.new_index_names[0])

        # 返回一个新的 MultiIndex 对象，使用给定的级别、编码和名称
        return MultiIndex(
            levels=self.new_index_levels,
            codes=result_codes,
            names=self.new_index_names,
            verify_integrity=False,
        )
def _unstack_multiple(
    data: Series | DataFrame, clocs, fill_value=None, sort: bool = True
):
    if len(clocs) == 0:
        return data

    # NOTE: This doesn't deal with hierarchical columns yet
    # 注意：目前还未处理层次化列

    index = data.index
    index = cast(MultiIndex, index)  # caller is responsible for checking
    # 将索引转换为 MultiIndex 类型，调用者需自行检查

    # GH 19966 Make sure if MultiIndexed index has tuple name, they will be
    # recognised as a whole
    # 确保如果 MultiIndexed 索引具有元组名称，则其将作为整体识别
    if clocs in index.names:
        clocs = [clocs]
    clocs = [index._get_level_number(i) for i in clocs]
    # 获取层次位置编号列表

    rlocs = [i for i in range(index.nlevels) if i not in clocs]
    # 获取不在 clocs 中的剩余层次位置列表

    clevels = [index.levels[i] for i in clocs]
    ccodes = [index.codes[i] for i in clocs]
    cnames = [index.names[i] for i in clocs]
    rlevels = [index.levels[i] for i in rlocs]
    rcodes = [index.codes[i] for i in rlocs]
    rnames = [index.names[i] for i in rlocs]
    # 分别获取列层次结构和行层次结构的 levels, codes 和 names

    shape = tuple(len(x) for x in clevels)
    # 计算列层次结构的形状信息

    group_index = get_group_index(ccodes, shape, sort=False, xnull=False)
    # 使用 ccodes 和 shape 获取分组索引

    comp_ids, obs_ids = compress_group_index(group_index, sort=False)
    # 压缩分组索引以获取 comp_ids 和 obs_ids

    recons_codes = decons_obs_group_ids(comp_ids, obs_ids, shape, ccodes, xnull=False)
    # 解压缩观察组 ID 以重建代码

    if not rlocs:
        # Everything is in clocs, so the dummy df has a regular index
        # 如果所有内容都在 clocs 中，则虚拟 DataFrame 具有常规索引
        dummy_index = Index(obs_ids, name="__placeholder__")
    else:
        dummy_index = MultiIndex(
            levels=rlevels + [obs_ids],
            codes=rcodes + [comp_ids],
            names=rnames + ["__placeholder__"],
            verify_integrity=False,
        )
        # 创建多级索引对象，包括行层次和一个占位符 "__placeholder__"

    if isinstance(data, Series):
        dummy = data.copy(deep=False)
        dummy.index = dummy_index
        # 复制数据并设置新的索引

        unstacked = dummy.unstack("__placeholder__", fill_value=fill_value, sort=sort)
        # 对数据进行展开操作，使用 "__placeholder__" 作为展开的列名，指定填充值和是否排序

        new_levels = clevels
        new_names = cnames
        new_codes = recons_codes
        # 设置新的列层次结构信息
    else:
        # 如果数据的列是 MultiIndex 类型，则直接使用数据作为结果
        if isinstance(data.columns, MultiIndex):
            result = data
            # 循环直到 clocs 为空
            while clocs:
                val = clocs.pop(0)
                # 对结果进行指定列的解堆叠操作，填充值为 fill_value，排序方式为 sort
                result = result.unstack(  # type: ignore[assignment]
                    val, fill_value=fill_value, sort=sort
                )
                # 调整 clocs 中的值，确保它们与解堆叠后的列索引对应
                clocs = [v if v < val else v - 1 for v in clocs]

            return result

        # GH#42579 设定 deep=False 以避免合并操作
        dummy_df = data.copy(deep=False)
        dummy_df.index = dummy_index

        # 对 dummy_df 进行 "__placeholder__" 列的解堆叠操作，填充值为 fill_value，排序方式为 sort
        unstacked = dummy_df.unstack(  # type: ignore[assignment]
            "__placeholder__", fill_value=fill_value, sort=sort
        )
        # 如果解堆叠后是 Series 类型，则使用其索引作为新列索引
        if isinstance(unstacked, Series):
            unstcols = unstacked.index
        else:
            unstcols = unstacked.columns
        assert isinstance(unstcols, MultiIndex)  # 用于类型检查，保证 unstcols 是 MultiIndex 类型
        # 构建新的 MultiIndex 层级，包括原始数据列索引的第一个层级和附加的 clevels
        new_levels = [unstcols.levels[0]] + clevels
        # 构建新的 MultiIndex 列名，包括原始数据列名和附加的 cnames
        new_names = [data.columns.name] + cnames

        # 构建新的 MultiIndex 编码，保留原始解堆叠后的第一个编码，再添加 recons_codes 中的编码
        new_codes = [unstcols.codes[0]]
        new_codes.extend(rec.take(unstcols.codes[-1]) for rec in recons_codes)

    # 创建新的 MultiIndex 对象，用新的层级、编码和列名，允许不验证完整性
    new_columns = MultiIndex(
        levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
    )

    # 根据 unstacked 的类型，更新其索引或者列为新的 MultiIndex 对象
    if isinstance(unstacked, Series):
        unstacked.index = new_columns
    else:
        unstacked.columns = new_columns

    return unstacked
@overload
# 函数重载装饰器，用于类型提示，声明 unstack 函数接受不同类型参数并返回不同类型值
def unstack(obj: Series, level, fill_value=..., sort: bool = ...) -> DataFrame: ...


@overload
# 另一种重载，支持同时处理 Series 和 DataFrame
def unstack(
    obj: Series | DataFrame, level, fill_value=..., sort: bool = ...
) -> Series | DataFrame: ...


def unstack(
    obj: Series | DataFrame, level, fill_value=None, sort: bool = True
) -> Series | DataFrame:
    if isinstance(level, (tuple, list)):
        if len(level) != 1:
            # _unstack_multiple only handles MultiIndexes,
            # and isn't needed for a single level
            # 处理多层级索引的情况，如果只有一个层级则不需要 _unstack_multiple
            return _unstack_multiple(obj, level, fill_value=fill_value, sort=sort)
        else:
            level = level[0]

    if not is_integer(level) and not level == "__placeholder__":
        # check if level is valid in case of regular index
        # 检查 level 是否在普通索引中有效
        obj.index._get_level_number(level)

    if isinstance(obj, DataFrame):
        if isinstance(obj.index, MultiIndex):
            # 如果 DataFrame 的索引是 MultiIndex，则调用 _unstack_frame 进行展开操作
            return _unstack_frame(obj, level, fill_value=fill_value, sort=sort)
        else:
            # 否则，对 DataFrame 进行转置并堆叠成 Series
            return obj.T.stack()
    elif not isinstance(obj.index, MultiIndex):
        # GH 36113
        # Give nicer error messages when unstack a Series whose
        # Index is not a MultiIndex.
        # 如果尝试对非 MultiIndex 的 Series 进行展开，抛出 ValueError
        raise ValueError(
            f"index must be a MultiIndex to unstack, {type(obj.index)} was passed"
        )
    else:
        if is_1d_only_ea_dtype(obj.dtype):
            # 如果 Series 的 dtype 是 ExtensionArray 类型，调用 _unstack_extension_series
            return _unstack_extension_series(obj, level, fill_value, sort=sort)
        # 否则，创建 _Unstacker 对象处理展开操作，并返回结果
        unstacker = _Unstacker(
            obj.index, level=level, constructor=obj._constructor_expanddim, sort=sort
        )
        return unstacker.get_result(obj, value_columns=None, fill_value=fill_value)


def _unstack_frame(
    obj: DataFrame, level, fill_value=None, sort: bool = True
) -> DataFrame:
    assert isinstance(obj.index, MultiIndex)  # checked by caller
    # 确保 DataFrame 的索引是 MultiIndex，由调用方检查
    unstacker = _Unstacker(
        obj.index, level=level, constructor=obj._constructor, sort=sort
    )

    if not obj._can_fast_transpose:
        # 如果 DataFrame 不支持快速转置，使用慢速方法进行展开操作
        mgr = obj._mgr.unstack(unstacker, fill_value=fill_value)
        return obj._constructor_from_mgr(mgr, axes=mgr.axes)
    else:
        # 否则，使用 _Unstacker 的快速方法获取结果
        return unstacker.get_result(
            obj, value_columns=obj.columns, fill_value=fill_value
        )


def _unstack_extension_series(
    series: Series, level, fill_value, sort: bool
) -> DataFrame:
    """
    Unstack an ExtensionArray-backed Series.

    The ExtensionDtype is preserved.

    Parameters
    ----------
    series : Series
        A Series with an ExtensionArray for values
    level : Any
        The level name or number.
    fill_value : Any
        The user-level (not physical storage) fill value to use for
        missing values introduced by the reshape. Passed to
        ``series.values.take``.
    sort : bool
        Whether to sort the resulting MuliIndex levels

    Returns
    -------
    DataFrame
        Each column of the DataFrame will have the same dtype as
        the input Series.
    """
    # Defer to the logic in ExtensionBlock._unstack
    # 延迟到 ExtensionBlock._unstack 中的逻辑处理
    # 将 Series 转换为 DataFrame
    df = series.to_frame()
    
    # 对 DataFrame 进行解堆叠操作，根据指定的层级和填充值，可选排序结果
    result = df.unstack(level=level, fill_value=fill_value, sort=sort)

    # 等效于 result.droplevel(level=0, axis=1)，但避免了额外的复制操作
    result.columns = result.columns._drop_level_numbers([0])
    
    # 错误：返回值类型不兼容（得到 "DataFrame | Series"，期望 "DataFrame"）
    return result  # type: ignore[return-value]
# 将 DataFrame 转换为带有多级索引的 Series 或 DataFrame。列将成为结果层次化索引的第二级。
def stack(
    frame: DataFrame, level=-1, dropna: bool = True, sort: bool = True
) -> Series | DataFrame:
    """
    Convert DataFrame to Series with multi-level Index. Columns become the
    second level of the resulting hierarchical index

    Returns
    -------
    stacked : Series or DataFrame
    """

    # 定义一个函数，用于处理索引的因子化，如果索引是唯一的则直接返回，否则进行因子化处理
    def stack_factorize(index):
        if index.is_unique:
            return index, np.arange(len(index))
        codes, categories = factorize_from_iterable(index)
        return categories, codes

    # 获取 DataFrame 的形状信息
    N, K = frame.shape

    # 将负数级别转换为有效的列级别，并检查是否超出范围
    level_num = frame.columns._get_level_number(level)

    # 如果列是 MultiIndex 类型，则调用 _stack_multi_columns 函数进行堆叠
    if isinstance(frame.columns, MultiIndex):
        return _stack_multi_columns(
            frame, level_num=level_num, dropna=dropna, sort=sort
        )
    # 如果索引是 MultiIndex 类型，则构建新的 MultiIndex
    elif isinstance(frame.index, MultiIndex):
        # 复制索引的水平和代码，以适应新的 MultiIndex 结构
        new_levels = list(frame.index.levels)
        new_codes = [lab.repeat(K) for lab in frame.index.codes]

        # 对列进行因子化处理，获取分类和代码
        clev, clab = stack_factorize(frame.columns)
        new_levels.append(clev)
        new_codes.append(np.tile(clab, N).ravel())

        # 构建新的 MultiIndex 对象
        new_names = list(frame.index.names)
        new_names.append(frame.columns.name)
        new_index = MultiIndex(
            levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
        )
    else:
        # 对索引和列分别进行因子化处理，构建新的 MultiIndex
        levels, (ilab, clab) = zip(*map(stack_factorize, (frame.index, frame.columns)))
        codes = ilab.repeat(K), np.tile(clab, N).ravel()
        new_index = MultiIndex(
            levels=levels,
            codes=codes,
            names=[frame.index.name, frame.columns.name],
            verify_integrity=False,
        )

    # 处理 DataFrame 的值，将其扁平化为一维数组
    new_values: ArrayLike
    if not frame.empty and frame._is_homogeneous_type:
        # 对于同质类型的扩展数组，使用特定的拼接方式
        dtypes = list(frame.dtypes._values)
        dtype = dtypes[0]

        if isinstance(dtype, ExtensionDtype):
            arr = dtype.construct_array_type()
            new_values = arr._concat_same_type(
                [col._values for _, col in frame.items()]
            )
            new_values = _reorder_for_extension_array_stack(new_values, N, K)
        else:
            # 对于非扩展数组的同质类型，直接将值扁平化处理
            new_values = frame._values.ravel()

    else:
        # 对于非同质类型的 DataFrame，直接将值扁平化处理
        new_values = frame._values.ravel()

    # 如果 dropna 参数为 True，则去除值和索引中的缺失值
    if dropna:
        mask = notna(new_values)
        new_values = new_values[mask]
        new_index = new_index[mask]

    # 返回构建的 Series 或 DataFrame 对象
    return frame._constructor_sliced(new_values, index=new_index)


# 未完全提供的函数，未提供详细的实现和文档注释
def stack_multiple(frame: DataFrame, level, dropna: bool = True, sort: bool = True):
    # 如果所有传递的级别都匹配列名，那么没有关于如何处理的歧义
    pass
    # 检查是否所有指定的层级名称在DataFrame的列名中均存在
    if all(lev in frame.columns.names for lev in level):
        result = frame
        # 对于每个层级名称，依次将DataFrame进行堆叠操作
        for lev in level:
            # 错误：赋值类型不兼容（表达式类型为"Series | DataFrame"，变量类型为"DataFrame"）
            result = stack(result, lev, dropna=dropna, sort=sort)  # type: ignore[assignment]

    # 否则，如果level中全是整数，说明层级编号可能会随着堆叠操作而变化
    elif all(isinstance(lev, int) for lev in level):
        # 每次堆叠完成后，层级编号会减小，因此在level是整数序列时需要考虑这一点
        result = frame
        # _get_level_number()检查层级编号是否在有效范围内，并将负数转换为正数
        level = [frame.columns._get_level_number(lev) for lev in level]

        # 当存在层级编号时
        while level:
            lev = level.pop(0)
            # 错误：赋值类型不兼容（表达式类型为"Series | DataFrame"，变量类型为"DataFrame"）
            result = stack(result, lev, dropna=dropna, sort=sort)  # type: ignore[assignment]
            # 减小所有大于当前层级编号的层级编号，因为这些层级现在的位置已经向下移动了一个
            level = [v if v <= lev else v - 1 for v in level]

    else:
        # 如果level既不全是层级名称也不全是层级编号，则抛出值错误
        raise ValueError(
            "level should contain all level names or all level "
            "numbers, not a mixture of the two."
        )

    # 返回最终的结果DataFrame
    return result
def _stack_multi_columns(
    frame: DataFrame, level_num: int = -1, dropna: bool = True, sort: bool = True
) -> DataFrame:
    def _convert_level_number(level_num: int, columns: Index):
        """
        Logic for converting the level number to something we can safely pass
        to swaplevel.

        If `level_num` matches a column name return the name from
        position `level_num`, otherwise return `level_num`.
        """
        # 检查指定的 level_num 是否在 columns 的名称中，若在则返回对应名称，否则返回 level_num 本身
        if level_num in columns.names:
            return columns.names[level_num]

        return level_num

    # 复制输入的 DataFrame，确保不进行深度复制
    this = frame.copy(deep=False)
    # 强制类型转换为 MultiIndex
    mi_cols = this.columns  # cast(MultiIndex, this.columns)
    # 断言 mi_cols 是 MultiIndex 类型，调用者有责任确保这一点
    assert isinstance(mi_cols, MultiIndex)

    # 这使得处理变得简单
    if level_num != mi_cols.nlevels - 1:
        # 将选择的 level_num 滚动到末尾以进行处理
        roll_columns = mi_cols
        for i in range(level_num, mi_cols.nlevels - 1):
            # 需要检查整数是否与级别名称冲突
            lev1 = _convert_level_number(i, roll_columns)
            lev2 = _convert_level_number(i + 1, roll_columns)
            # 交换指定的两个级别
            roll_columns = roll_columns.swaplevel(lev1, lev2)
        this.columns = mi_cols = roll_columns

    # 如果列没有按字典顺序排列且排序标志为真
    if not mi_cols._is_lexsorted() and sort:
        # 解决边缘情况，其中 0 是其中一个列名，干扰了基于第一个级别进行排序的尝试
        level_to_sort = _convert_level_number(0, mi_cols)
        # 按指定的 level_to_sort 对索引进行排序，axis=1 表示按列排序
        this = this.sort_index(level=level_to_sort, axis=1)
        mi_cols = this.columns

    # 强制类型转换为 MultiIndex
    mi_cols = cast(MultiIndex, mi_cols)
    # 使用 _stack_multi_column_index 函数生成新列
    new_columns = _stack_multi_column_index(mi_cols)

    # 准备扁平化数据值
    new_data = {}
    # 获取最后一个级别的值和代码
    level_vals = mi_cols.levels[-1]
    level_codes = unique(mi_cols.codes[-1])
    # 如果排序标志为真，则对代码进行排序
    if sort:
        level_codes = np.sort(level_codes)
    # 在 level_vals 的末尾插入一个 None 值
    level_vals_nan = level_vals.insert(len(level_vals), None)

    # 提取 level_codes 对应的 level_vals 值
    level_vals_used = np.take(level_vals_nan, level_codes)
    # 计算 level_codes 的大小
    levsize = len(level_codes)
    # 初始化要删除的列列表
    drop_cols = []
    for key in new_columns:
        try:
            # 获取当前列名在原始数据中的位置索引
            loc = this.columns.get_loc(key)
        except KeyError:
            # 如果列名不存在于原始数据中，将其添加到待删除列列表并继续下一个循环
            drop_cols.append(key)
            continue

        # 可以优化吗？
        # 我们几乎总是返回一个切片
        # 但如果未排序，可能会得到一个布尔索引器
        if not isinstance(loc, slice):
            # 计算索引位置的长度
            slice_len = len(loc)
        else:
            # 计算切片的长度
            slice_len = loc.stop - loc.start

        if slice_len != levsize:
            # 如果长度与预期的不同，则处理为切片
            chunk = this.loc[:, this.columns[loc]]
            # 更新列的标签为特定的等级值
            chunk.columns = level_vals_nan.take(chunk.columns.codes[-1])
            # 重新索引为使用的等级值的列的值
            value_slice = chunk.reindex(columns=level_vals_used).values
        else:
            # 否则，选择数据的子集
            subset = this.iloc[:, loc]
            # 查找子集中的数据类型的共同类型
            dtype = find_common_type(subset.dtypes.tolist())
            if isinstance(dtype, ExtensionDtype):
                # 如果数据类型是扩展类型，则构造数组并连接相同类型的值
                value_slice = dtype.construct_array_type()._concat_same_type(
                    [x._values.astype(dtype, copy=False) for _, x in subset.items()]
                )
                N, K = subset.shape
                idx = np.arange(N * K).reshape(K, N).T.reshape(-1)
                value_slice = value_slice.take(idx)
            else:
                # 否则，直接使用子集的值
                value_slice = subset.values

        if value_slice.ndim > 1:
            # 如果是多维的，展平为一维数组
            value_slice = value_slice.ravel()

        # 将处理后的值存入新数据的对应列中
        new_data[key] = value_slice

    if len(drop_cols) > 0:
        # 如果有需要删除的列，则更新新列列表
        new_columns = new_columns.difference(drop_cols)

    N = len(this)

    if isinstance(this.index, MultiIndex):
        # 如果原始数据的索引是多级索引，则按照当前的处理方法更新新的索引结构
        new_levels = list(this.index.levels)
        new_names = list(this.index.names)
        new_codes = [lab.repeat(levsize) for lab in this.index.codes]
    else:
        # 否则，从当前的索引数据中获取因子化结果
        old_codes, old_levels = factorize_from_iterable(this.index)
        new_levels = [old_levels]
        new_codes = [old_codes.repeat(levsize)]
        new_names = [this.index.name]  # something better?

    # 添加新的等级值和代码到索引结构中
    new_levels.append(level_vals)
    new_codes.append(np.tile(level_codes, N))
    new_names.append(frame.columns.names[level_num])

    # 使用新的索引结构和新的列结构构造结果DataFrame
    new_index = MultiIndex(
        levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False
    )

    # 构造结果DataFrame并返回
    result = frame._constructor(new_data, index=new_index, columns=new_columns)

    if frame.columns.nlevels > 1:
        # 如果原始数据的列有多个层级，则确保结果DataFrame的列只包含期望的列
        desired_columns = frame.columns._drop_level_numbers([level_num]).unique()
        if not result.columns.equals(desired_columns):
            result = result[desired_columns]

    # 更高效的方法来处理这个过程？可以做整个掩码操作，但只会节省一点时间...
    if dropna:
        # 如果需要删除NaN值，则按行删除
        result = result.dropna(axis=0, how="all")

    return result
# 对扩展数组堆叠进行重新排序

def _reorder_for_extension_array_stack(
    arr: ExtensionArray, n_rows: int, n_columns: int
) -> ExtensionArray:
    """
    Re-orders the values when stacking multiple extension-arrays.

    The indirect stacking method used for EAs requires a followup
    take to get the order correct.

    Parameters
    ----------
    arr : ExtensionArray
        扩展数组，需要重新排序的对象
    n_rows, n_columns : int
        原始 DataFrame 中的行数和列数

    Returns
    -------
    taken : ExtensionArray
        经过适当重新排序的原始 `arr`

    Examples
    --------
    >>> arr = np.array(["a", "b", "c", "d", "e", "f"])
    >>> _reorder_for_extension_array_stack(arr, 2, 3)
    array(['a', 'c', 'e', 'b', 'd', 'f'], dtype='<U1')

    >>> _reorder_for_extension_array_stack(arr, 3, 2)
    array(['a', 'd', 'b', 'e', 'c', 'f'], dtype='<U1')
    """
    # 最终采取方法以正确排序顺序
    # idx 是一个类似索引器的数组，形式如下
    # [c0r0, c1r0, c2r0, ...,
    #  c0r1, c1r1, c2r1, ...]
    idx = np.arange(n_rows * n_columns).reshape(n_columns, n_rows).T.reshape(-1)
    return arr.take(idx)


def stack_v3(frame: DataFrame, level: list[int]) -> Series | DataFrame:
    if frame.columns.nunique() != len(frame.columns):
        raise ValueError("Columns with duplicate values are not supported in stack")
    set_levels = set(level)
    stack_cols = frame.columns._drop_level_numbers(
        [k for k in range(frame.columns.nlevels - 1, -1, -1) if k not in set_levels]
    )

    result = stack_reshape(frame, level, set_levels, stack_cols)

    # 构造正确的 MultiIndex，结合 frame 的索引和堆叠的列
    ratio = 0 if frame.empty else len(result) // len(frame)

    index_levels: list | FrozenList
    if isinstance(frame.index, MultiIndex):
        index_levels = frame.index.levels
        index_codes = list(np.tile(frame.index.codes, (1, ratio)))
    else:
        codes, uniques = factorize(frame.index, use_na_sentinel=False)
        index_levels = [uniques]
        index_codes = list(np.tile(codes, (1, ratio)))

    if len(level) > 1:
        # 根据指定的顺序排列列，例如 level=[2, 0, 1]
        sorter = np.argsort(level)
        assert isinstance(stack_cols, MultiIndex)
        ordered_stack_cols = stack_cols._reorder_ilevels(sorter)
    else:
        ordered_stack_cols = stack_cols
    ordered_stack_cols_unique = ordered_stack_cols.unique()
    if isinstance(ordered_stack_cols, MultiIndex):
        column_levels = ordered_stack_cols.levels
        column_codes = ordered_stack_cols.drop_duplicates().codes
    else:
        column_levels = [ordered_stack_cols_unique]
        column_codes = [factorize(ordered_stack_cols_unique, use_na_sentinel=False)[0]]

    # error: Incompatible types in assignment (expression has type "list[ndarray[Any,
    # dtype[Any]]]", variable has type "FrozenList")
    column_codes = [np.repeat(codes, len(frame)) for codes in column_codes]  # type: ignore[assignment]
    # 设置结果的索引为多级索引，合并传入的索引级别和列级别，设置名称并关闭完整性验证
    result.index = MultiIndex(
        levels=index_levels + column_levels,  # 设置多级索引的级别
        codes=index_codes + column_codes,    # 设置多级索引的编码
        names=frame.index.names + list(ordered_stack_cols.names),  # 设置多级索引的名称
        verify_integrity=False,  # 关闭完整性验证
    )

    # 排序结果，采用较快的方式而非调用 sort_index，因为已知所需顺序
    len_df = len(frame)  # 获取数据框的长度
    n_uniques = len(ordered_stack_cols_unique)  # 获取有序堆栈列的唯一值数量
    indexer = np.arange(n_uniques)  # 创建一个索引器数组
    idxs = np.tile(len_df * indexer, len_df) + np.repeat(np.arange(len_df), n_uniques)  # 创建索引数组
    result = result.take(idxs)  # 使用索引数组对结果进行重新排序

    # 如果结果的维度为2且数据框的列级别与给定级别长度相同
    if result.ndim == 2 and frame.columns.nlevels == len(level):
        if len(result.columns) == 0:  # 如果结果的列数为0
            result = Series(index=result.index)  # 将结果转换为系列
        else:
            result = result.iloc[:, 0]  # 否则，仅保留第一列结果数据

    # 如果结果的维度为1
    if result.ndim == 1:
        result.name = None  # 清空结果的名称

    return result  # 返回处理后的结果
# 定义一个函数，用于重新塑造 DataFrame 数据以进行堆叠操作
def stack_reshape(
    frame: DataFrame, level: list[int], set_levels: set[int], stack_cols: Index
) -> Series | DataFrame:
    """Reshape the data of a frame for stack.

    This function takes care of most of the work that stack needs to do. Caller
    will sort the result once the appropriate index is set.

    Parameters
    ----------
    frame: DataFrame
        要进行堆叠操作的 DataFrame。
    level: list of ints.
        需要堆叠的列的级别。
    set_levels: set of ints.
        与 level 相同，但作为一个集合。
    stack_cols: Index.
        当 DataFrame 堆叠时的结果列。

    Returns
    -------
    被堆叠 DataFrame 的数据。
    """
    # 如果需要从列中删除 `level`，则需要按降序排序
    drop_levnums = sorted(level, reverse=True)

    # 收集每个要堆叠的唯一索引的数据
    buf = []
    for idx in stack_cols.unique():
        if len(frame.columns) == 1:
            data = frame.copy(deep=False)
        else:
            if not isinstance(frame.columns, MultiIndex) and not isinstance(idx, tuple):
                # GH#57750 - 如果 frame 是带有元组的索引，下面的 .loc 将失败
                column_indexer = idx
            else:
                # 从 frame 中获取与此 idx 值对应的数据
                if len(level) == 1:
                    idx = (idx,)
                gen = iter(idx)
                column_indexer = tuple(
                    next(gen) if k in set_levels else slice(None)
                    for k in range(frame.columns.nlevels)
                )
            data = frame.loc[:, column_indexer]

        if len(level) < frame.columns.nlevels:
            data.columns = data.columns._drop_level_numbers(drop_levnums)
        elif stack_cols.nlevels == 1:
            if data.ndim == 1:
                data.name = 0
            else:
                data.columns = default_index(len(data.columns))
        buf.append(data)

    if len(buf) > 0 and not frame.empty:
        # 将收集的数据拼接成一个 DataFrame
        result = concat(buf, ignore_index=True)
    else:
        # 输入为空的情况
        if len(level) < frame.columns.nlevels:
            # 拼接后的列顺序可能与删除级别后不同
            new_columns = frame.columns._drop_level_numbers(drop_levnums).unique()
        else:
            new_columns = [0]
        result = DataFrame(columns=new_columns, dtype=frame._values.dtype)

    if len(level) < frame.columns.nlevels:
        # 拼接后的列顺序可能与删除级别后不同
        desired_columns = frame.columns._drop_level_numbers(drop_levnums).unique()
        if not result.columns.equals(desired_columns):
            result = result[desired_columns]

    return result
```