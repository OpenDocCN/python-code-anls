# `D:\src\scipysrc\pandas\pandas\core\groupby\ops.py`

```
# 提供执行分组聚合操作的类。

# 这些类不向用户公开，主要在Cython中提供了分组操作的实现。
# 这些类（BaseGrouper和BinGrouper）包含在SeriesGroupBy和DataFrameGroupBy对象中。
from __future__ import annotations

import collections  # 引入collections模块
import functools  # 引入functools模块
from typing import (
    TYPE_CHECKING,  # 引入类型提示中的TYPE_CHECKING
    Generic,
    final,
)

import numpy as np  # 引入numpy模块

from pandas._libs import (
    NaT,  # 从pandas._libs中引入NaT对象
    lib,
)
import pandas._libs.groupby as libgroupby  # 从pandas._libs中引入libgroupby模块
from pandas._typing import (
    ArrayLike,  # 引入ArrayLike类型
    AxisInt,  # 引入AxisInt类型
    NDFrameT,  # 引入NDFrameT类型
    Shape,  # 引入Shape类型
    npt,  # 引入npt类型
)
from pandas.errors import AbstractMethodError  # 从pandas.errors中引入AbstractMethodError异常
from pandas.util._decorators import cache_readonly  # 从pandas.util._decorators中引入cache_readonly装饰器

from pandas.core.dtypes.cast import (
    maybe_cast_pointwise_result,  # 从pandas.core.dtypes.cast中引入maybe_cast_pointwise_result函数
    maybe_downcast_to_dtype,  # 从pandas.core.dtypes.cast中引入maybe_downcast_to_dtype函数
)
from pandas.core.dtypes.common import (
    ensure_float64,  # 从pandas.core.dtypes.common中引入ensure_float64函数
    ensure_int64,  # 从pandas.core.dtypes.common中引入ensure_int64函数
    ensure_platform_int,  # 从pandas.core.dtypes.common中引入ensure_platform_int函数
    ensure_uint64,  # 从pandas.core.dtypes.common中引入ensure_uint64函数
    is_1d_only_ea_dtype,  # 从pandas.core.dtypes.common中引入is_1d_only_ea_dtype函数
)
from pandas.core.dtypes.missing import (
    isna,  # 从pandas.core.dtypes.missing中引入isna函数
    maybe_fill,  # 从pandas.core.dtypes.missing中引入maybe_fill函数
)

from pandas.core.arrays import Categorical  # 从pandas.core.arrays中引入Categorical数组
from pandas.core.frame import DataFrame  # 从pandas.core.frame中引入DataFrame类
from pandas.core.groupby import grouper  # 从pandas.core.groupby中引入grouper模块
from pandas.core.indexes.api import (
    CategoricalIndex,  # 从pandas.core.indexes.api中引入CategoricalIndex索引
    Index,  # 从pandas.core.indexes.api中引入Index索引
    MultiIndex,  # 从pandas.core.indexes.api中引入MultiIndex索引
    ensure_index,  # 从pandas.core.indexes.api中引入ensure_index函数
)
from pandas.core.series import Series  # 从pandas.core.series中引入Series类
from pandas.core.sorting import (
    compress_group_index,  # 从pandas.core.sorting中引入compress_group_index函数
    decons_obs_group_ids,  # 从pandas.core.sorting中引入decons_obs_group_ids函数
    get_group_index,  # 从pandas.core.sorting中引入get_group_index函数
    get_group_index_sorter,  # 从pandas.core.sorting中引入get_group_index_sorter函数
    get_indexer_dict,  # 从pandas.core.sorting中引入get_indexer_dict函数
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,  # 引入collections.abc中的Callable类
        Generator,  # 引入collections.abc中的Generator类
        Hashable,  # 引入collections.abc中的Hashable类
        Iterator,  # 引入collections.abc中的Iterator类
    )

    from pandas.core.generic import NDFrame  # 从pandas.core.generic中引入NDFrame类


def check_result_array(obj, dtype) -> None:
    # 我们的操作应该是一个聚合/归约操作。如果它返回一个ndarray，这通常意味着传递了无效的操作。
    # 参见test_apply_without_aggregation，test_agg_must_agg
    if isinstance(obj, np.ndarray):  # 如果obj是一个numpy的ndarray对象
        if dtype != object:  # 如果dtype不是object类型
            # 如果它是object类型，函数可能是一个归约/聚合函数，仍然可以返回一个ndarray，例如test_agg_over_numpy_arrays
            raise ValueError("Must produce aggregated value")  # 抛出值错误异常，必须产生聚合值


def extract_result(res):
    """
    提取结果对象，它可能是一个0维的ndarray或者长度为1的0维数组，或者是一个标量
    """
    if hasattr(res, "_values"):  # 如果res具有_values属性
        # 保留EA
        res = res._values  # 将res赋值为其_values属性
        if res.ndim == 1 and len(res) == 1:  # 如果res是一维数组且长度为1
            # 参见test_agg_lambda_with_timezone，test_resampler_grouper.py::test_apply
            res = res[0]  # 将res的第一个元素赋值给res
    return res  # 返回res


class WrappedCythonOp:
    """
    用于调度在_libs.groupby中定义的函数的分发逻辑

    Parameters
    ----------
    kind: str
        操作类型，是聚合还是转换。
    how: str
        操作名称，例如"mean"。
    has_dropped_na: bool
        当dropna=True且分组器包含空值时为True。
    """
    pass  # 空的类，用于描述调度逻辑，参数详细信息见文档字符串说明
    # 不尝试将 Cython 结果强制转换回原始数据类型的函数列表
    # 这些函数的结果不会被转换
    cast_blocklist = frozenset(
        ["any", "all", "rank", "count", "size", "idxmin", "idxmax"]
    )

    # 初始化方法，接受 kind（类型）、how（方法）、has_dropped_na（是否删除了 NA 值）
    def __init__(self, kind: str, how: str, has_dropped_na: bool) -> None:
        self.kind = kind
        self.how = how
        self.has_dropped_na = has_dropped_na

    # Cython 函数的映射表
    _CYTHON_FUNCTIONS: dict[str, dict] = {
        "aggregate": {
            "any": functools.partial(libgroupby.group_any_all, val_test="any"),
            "all": functools.partial(libgroupby.group_any_all, val_test="all"),
            "sum": "group_sum",
            "prod": "group_prod",
            "idxmin": functools.partial(libgroupby.group_idxmin_idxmax, name="idxmin"),
            "idxmax": functools.partial(libgroupby.group_idxmin_idxmax, name="idxmax"),
            "min": "group_min",
            "max": "group_max",
            "mean": "group_mean",
            "median": "group_median_float64",
            "var": "group_var",
            "std": functools.partial(libgroupby.group_var, name="std"),
            "sem": functools.partial(libgroupby.group_var, name="sem"),
            "skew": "group_skew",
            "first": "group_nth",
            "last": "group_last",
            "ohlc": "group_ohlc",
        },
        "transform": {
            "cumprod": "group_cumprod",
            "cumsum": "group_cumsum",
            "cummin": "group_cummin",
            "cummax": "group_cummax",
            "rank": "group_rank",
        },
    }

    # 特定 Cython 函数的参数数量
    _cython_arity = {"ohlc": 4}  # OHLC

    # 类方法：根据 how 参数确定 kind 的方法类别（aggregate 或 transform）
    @classmethod
    def get_kind_from_how(cls, how: str) -> str:
        if how in cls._CYTHON_FUNCTIONS["aggregate"]:
            return "aggregate"
        return "transform"

    # 注意：我们将此方法设为类方法，并传递 kind 和 how 参数，
    #  以便缓存工作在类级别而不是实例级别
    @classmethod
    @functools.cache
    def _get_cython_function(
        cls, kind: str, how: str, dtype: np.dtype, is_numeric: bool
        ):
            # 获取数据类型的字符串表示
            dtype_str = dtype.name
            # 根据分类和处理方式获取对应的 Cython 函数名
            ftype = cls._CYTHON_FUNCTIONS[kind][how]

            # 查看是否存在融合类型版本的函数，仅适用于数值类型
            if callable(ftype):
                f = ftype
            else:
                f = getattr(libgroupby, ftype)
            # 如果是数值类型，则直接返回对应的函数
            if is_numeric:
                return f
            # 如果数据类型是对象类型
            elif dtype == np.dtype(object):
                if how in ["median", "cumprod"]:
                    # 没有融合类型 -> 没有 __signatures__
                    raise NotImplementedError(
                        f"function is not implemented for this dtype: "
                        f"[how->{how},dtype->{dtype_str}]"
                    )
                elif how in ["std", "sem", "idxmin", "idxmax"]:
                    # 我们有一个部分对象，没有 __signatures__
                    return f
                elif how == "skew":
                    # _get_cython_vals 将转换为 float64
                    pass
                elif "object" not in f.__signatures__:
                    # 在此处引发 NotImplementedError 而不是稍后的 TypeError
                    raise NotImplementedError(
                        f"function is not implemented for this dtype: "
                        f"[how->{how},dtype->{dtype_str}]"
                    )
                return f
            else:
                # 如果走到这里，说明存在未实现的情况
                raise NotImplementedError(
                    "This should not be reached. Please report a bug at "
                    "github.com/pandas-dev/pandas/",
                    dtype,
                )

        def _get_cython_vals(self, values: np.ndarray) -> np.ndarray:
            """
            Cast numeric dtypes to float64 for functions that only support that.

            Parameters
            ----------
            values : np.ndarray

            Returns
            -------
            values : np.ndarray
            """
            how = self.how

            # 如果处理方式是 ["median", "std", "sem", "skew"] 中的一种
            if how in ["median", "std", "sem", "skew"]:
                # median 只有一个 float64 的实现
                # 我们应该只有在 is_numeric 的情况下才会到这里，因为非数值类型的情况应该在 _get_cython_function 中引发异常
                values = ensure_float64(values)

            # 如果值的数据类型在 "iu" 中
            elif values.dtype.kind in "iu":
                if how in ["var", "mean"] or (
                    self.kind == "transform" and self.has_dropped_na
                ):
                    # has_dropped_na 检查需要 test_null_group_str_transformer
                    # 结果可能仍然包含 NaN，所以我们必须进行类型转换为 float64
                    values = ensure_float64(values)

                elif how in ["sum", "ohlc", "prod", "cumsum", "cumprod"]:
                    # 在组操作期间避免溢出
                    if values.dtype.kind == "i":
                        values = ensure_int64(values)
                    else:
                        values = ensure_uint64(values)

            return values
    # 根据分组数和值数组确定输出的形状
    def _get_output_shape(self, ngroups: int, values: np.ndarray) -> Shape:
        # 获取操作方式和类型
        how = self.how
        kind = self.kind

        # 根据操作方式获取对应的参数个数
        arity = self._cython_arity.get(how, 1)

        # 初始化输出形状变量
        out_shape: Shape

        # 根据操作方式和类型确定输出形状
        if how == "ohlc":
            out_shape = (ngroups, arity)
        elif arity > 1:
            # 如果参数个数大于1且操作方式不支持，抛出未实现错误
            raise NotImplementedError(
                "arity of more than 1 is not supported for the 'how' argument"
            )
        elif kind == "transform":
            # 如果操作类型为'transform'，输出形状与值数组的形状相同
            out_shape = values.shape
        else:
            # 默认情况下，输出形状为(ngroups, ...)，保留值数组的形状的其余维度
            out_shape = (ngroups,) + values.shape[1:]

        # 返回计算得到的输出形状
        return out_shape

    # 获取输出的数据类型
    def _get_out_dtype(self, dtype: np.dtype) -> np.dtype:
        # 获取操作方式
        how = self.how

        # 根据操作方式确定输出的数据类型
        if how == "rank":
            out_dtype = "float64"
        elif how in ["idxmin", "idxmax"]:
            # 对于'idxmin'和'idxmax'操作，输出的数据类型为整数指针类型
            # Cython 实现只会生成行号；后续会使用这个来从索引中获取值
            out_dtype = "intp"
        else:
            # 对于其他操作方式，根据输入的数据类型的类别确定输出的数据类型
            if dtype.kind in "iufcb":
                out_dtype = f"{dtype.kind}{dtype.itemsize}"
            else:
                out_dtype = "object"

        # 返回计算得到的输出数据类型
        return np.dtype(out_dtype)

    # 获取结果的数据类型
    def _get_result_dtype(self, dtype: np.dtype) -> np.dtype:
        """
        根据输入的数据类型和操作方式，获取结果的期望数据类型。

        Parameters
        ----------
        dtype : np.dtype
            输入数据的类型

        Returns
        -------
        np.dtype
            结果的期望数据类型
        """
        # 获取操作方式
        how = self.how

        # 根据操作方式确定结果的期望数据类型
        if how in ["sum", "cumsum", "sum", "prod", "cumprod"]:
            # 如果操作方式为求和、累计求和、累计乘积等，则根据输入数据类型确定输出数据类型
            if dtype == np.dtype(bool):
                return np.dtype(np.int64)
        elif how in ["mean", "median", "var", "std", "sem"]:
            # 如果操作方式为均值、中位数、方差、标准差、标准误差等，则根据输入数据类型类别确定输出数据类型
            if dtype.kind in "fc":
                return dtype
            elif dtype.kind in "iub":
                return np.dtype(np.float64)

        # 默认情况下，直接返回输入的数据类型作为结果的数据类型
        return dtype

    # 用于Cython操作的维度兼容性检查
    @final
    def _cython_op_ndim_compat(
        self,
        values: np.ndarray,
        *,
        min_count: int,
        ngroups: int,
        comp_ids: np.ndarray,
        mask: npt.NDArray[np.bool_] | None = None,
        result_mask: npt.NDArray[np.bool_] | None = None,
        **kwargs,
    @final
    def _call_cython_op(
        self,
        values: np.ndarray,  # 接受一个 numpy 数组作为参数，假定其维度为 2
        *,
        min_count: int,  # 最小计数值，作为操作的参数
        ngroups: int,  # 分组数，作为操作的参数
        comp_ids: np.ndarray,  # 与组件相关的标识符数组，作为操作的参数
        mask: npt.NDArray[np.bool_] | None,  # 布尔类型的掩码数组或空值，作为操作的参数
        result_mask: npt.NDArray[np.bool_] | None,  # 结果的布尔掩码数组或空值，作为操作的参数
        **kwargs,  # 其他关键字参数
    ):
        if values.ndim == 1:
            # 将一维数组扩展为二维数组，然后调用 Cython 操作，必要时挤压返回结果
            values2d = values[None, :]  # 在行上添加维度，使其成为二维数组
            if mask is not None:
                mask = mask[None, :]  # 如果存在掩码，则在行上添加维度
            if result_mask is not None:
                result_mask = result_mask[None, :]  # 如果存在结果掩码，则在行上添加维度
            res = self._call_cython_op(  # 递归调用本身，处理二维化后的数据
                values2d,
                min_count=min_count,
                ngroups=ngroups,
                comp_ids=comp_ids,
                mask=mask,
                result_mask=result_mask,
                **kwargs,
            )
            if res.shape[0] == 1:
                return res[0]  # 如果结果的第一维长度为1，则返回挤压后的结果数组的第一行

            # 否则假定结果为 OHLC（开盘价、最高价、最低价、收盘价）数据，返回其转置
            return res.T  # 返回结果的转置矩阵

        # 如果输入数组已经是二维的，则直接调用 Cython 操作
        return self._call_cython_op(
            values,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=comp_ids,
            mask=mask,
            result_mask=result_mask,
            **kwargs,
        )

    @final
    def _validate_axis(self, axis: AxisInt, values: ArrayLike) -> None:
        if values.ndim > 2:
            raise NotImplementedError("number of dimensions is currently limited to 2")
        if values.ndim == 2:
            assert axis == 1, axis  # 对于二维数据，要求轴必须为1
        elif not is_1d_only_ea_dtype(values.dtype):
            # 注意：对于一维扩展数组，轴并不总是0，因为我们需要将其视为二维处理
            assert axis == 0  # 对于一维数组，要求轴必须为0

    @final
    def cython_operation(
        self,
        *,
        values: ArrayLike,  # 接受一个类数组作为参数
        axis: AxisInt,  # 轴的整数表示，作为操作的参数
        min_count: int = -1,  # 最小计数值，默认为-1
        comp_ids: np.ndarray,  # 与组件相关的标识符数组，作为操作的参数
        ngroups: int,  # 分组数，作为操作的参数
        **kwargs,  # 其他关键字参数
    ) -> ArrayLike:  # 返回值为类数组
        """
        调用我们的 Cython 函数，并进行适当的预处理和后处理。
        """
        self._validate_axis(axis, values)  # 调用验证轴函数，确保轴设置正确

        if not isinstance(values, np.ndarray):
            # 即 ExtensionArray 类型的情况
            return values._groupby_op(
                how=self.how,  # 操作方式参数
                has_dropped_na=self.has_dropped_na,  # 是否已经删除了 NA 值的标志
                min_count=min_count,  # 最小计数值，作为操作的参数
                ngroups=ngroups,  # 分组数，作为操作的参数
                ids=comp_ids,  # 组件标识符数组，作为操作的参数
                **kwargs,  # 其他关键字参数
            )

        # 对于普通的 numpy 数组，调用多维兼容的 Cython 操作
        return self._cython_op_ndim_compat(
            values,
            min_count=min_count,
            ngroups=ngroups,
            comp_ids=comp_ids,
            mask=None,
            **kwargs,
        )
    def get_indexer_dict(codes_list: list[npt.NDArray[np.intp]], levels: list[npt.NDArray[np.ndarray]]) -> dict:
        """
        Return a dictionary mapping codes to indexers.

        Parameters
        ----------
        codes_list : list of arrays
            List of arrays containing codes for each level.
        levels : list of arrays
            List of arrays containing unique levels for each level.

        Returns
        -------
        dict
            Dictionary mapping codes to indexers.
        """
        if len(levels) == 1:
            return {code: indexer for code, indexer in enumerate(codes_list[0])}
        return {tuple(code): indexer for code, indexer in enumerate(codes_list)}

    """
    This is an internal Grouper class, which actually holds
    the generated groups

    Parameters
    ----------
    axis : Index
        Index object representing the axis to group by
    groupings : list of grouper.Grouping
        List of Grouping instances to handle in this grouper
    sort : bool, default True
        Flag indicating whether the results should be sorted or not
    dropna : bool, default True
        Flag indicating whether to drop NA/null values during grouping
    """
    axis: Index

    def __init__(
        self,
        axis: Index,
        groupings: list[grouper.Grouping],
        sort: bool = True,
        dropna: bool = True,
    ) -> None:
        assert isinstance(axis, Index), axis

        self.axis = axis
        self._groupings = groupings
        self._sort = sort
        self.dropna = dropna

    @property
    def groupings(self) -> list[grouper.Grouping]:
        """
        Property method returning the list of groupings.

        Returns
        -------
        list of grouper.Grouping
            List of Grouping instances used in this grouper.
        """
        return self._groupings

    def __iter__(self) -> Iterator[Hashable]:
        """
        Returns an iterator over the indices of the groups.

        Returns
        -------
        Iterator of Hashable
            Iterator over the group indices.
        """
        return iter(self.indices)

    @property
    def nkeys(self) -> int:
        """
        Property method returning the number of keys (groupings).

        Returns
        -------
        int
            Number of groupings.
        """
        return len(self.groupings)

    def get_iterator(self, data: NDFrameT) -> Iterator[tuple[Hashable, NDFrameT]]:
        """
        Returns a generator yielding tuples of (name, subsetted object)
        for each group.

        Parameters
        ----------
        data : NDFrameT
            Input data to iterate over.

        Yields
        ------
        tuple of (Hashable, NDFrameT)
            Tuple containing the group name and subsetted data object.
        """
        splitter = self._get_splitter(data)
        # TODO: Would be more efficient to skip unobserved for transforms
        keys = self.result_index
        yield from zip(keys, splitter)

    @final
    def _get_splitter(self, data: NDFrame) -> DataSplitter:
        """
        Returns a DataSplitter instance based on the type of input data.

        Parameters
        ----------
        data : NDFrame
            Input data to split.

        Returns
        -------
        DataSplitter
            Instance of DataSplitter appropriate for the input data type.
        """
        if isinstance(data, Series):
            klass: type[DataSplitter] = SeriesSplitter
        else:
            # i.e. DataFrame
            klass = FrameSplitter

        return klass(
            data,
            self.ngroups,
            sorted_ids=self._sorted_ids,
            sort_idx=self.result_ilocs,
        )

    @cache_readonly
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        """
        Property method returning a dictionary of group names to group indices.

        Returns
        -------
        dict of Hashable to npt.NDArray[np.intp]
            Dictionary mapping group names to their corresponding indices.
        """
        if len(self.groupings) == 1 and isinstance(self.result_index, CategoricalIndex):
            # This shows unused categories in indices GH#38642
            return self.groupings[0].indices
        codes_list = [ping.codes for ping in self.groupings]
        return get_indexer_dict(codes_list, self.levels)

    @final
    @cache_readonly
    def result_ilocs(self) -> npt.NDArray[np.intp]:
        """
        Get the original integer locations of result_index in the input.
        """
        # Original indices are where group_index would go via sorting.
        # But when dropna is true, we need to remove null values while accounting for
        # any gaps that then occur because of them.
        # 获取结果索引在输入中的原始整数位置

        ids = self.ids

        if self.has_dropped_na:
            mask = np.where(ids >= 0)
            # Count how many gaps are caused by previous null values for each position
            # 统计每个位置之前的空值导致的间隙数量
            null_gaps = np.cumsum(ids == -1)[mask]
            ids = ids[mask]

        result = get_group_index_sorter(ids, self.ngroups)

        if self.has_dropped_na:
            # Shift by the number of prior null gaps
            # 根据先前空值导致的间隙数量进行偏移
            result += np.take(null_gaps, result)

        return result

    @property
    def codes(self) -> list[npt.NDArray[np.signedinteger]]:
        return [ping.codes for ping in self.groupings]

    @property
    def levels(self) -> list[Index]:
        if len(self.groupings) > 1:
            # mypy doesn't know result_index must be a MultiIndex
            # mypy不知道result_index必须是MultiIndex类型
            return list(self.result_index.levels)  # type: ignore[attr-defined]
        else:
            return [self.result_index]

    @property
    def names(self) -> list[Hashable]:
        return [ping.name for ping in self.groupings]

    @final
    def size(self) -> Series:
        """
        Compute group sizes.
        """
        ids = self.ids
        ngroups = self.ngroups
        out: np.ndarray | list
        if ngroups:
            out = np.bincount(ids[ids != -1], minlength=ngroups)
        else:
            out = []
        return Series(out, index=self.result_index, dtype="int64", copy=False)

    @cache_readonly
    def groups(self) -> dict[Hashable, Index]:
        """dict {group name -> group labels}"""
        if len(self.groupings) == 1:
            return self.groupings[0].groups
        result_index, ids = self.result_index_and_ids
        values = result_index._values
        categories = Categorical(ids, categories=range(len(result_index)))
        result = {
            # mypy is not aware that group has to be an integer
            # mypy不知道group必须是整数类型
            values[group]: self.axis.take(axis_ilocs)  # type: ignore[call-overload]
            for group, axis_ilocs in categories._reverse_indexer().items()
        }
        return result

    @final
    @cache_readonly
    def is_monotonic(self) -> bool:
        # return if my group orderings are monotonic
        # 返回我的分组顺序是否单调
        return Index(self.ids).is_monotonic_increasing

    @final
    @cache_readonly
    def has_dropped_na(self) -> bool:
        """
        Whether grouper has null value(s) that are dropped.
        """
        # 返回分组器是否存在被丢弃的空值
        return bool((self.ids < 0).any())

    @cache_readonly
    def codes_info(self) -> npt.NDArray[np.intp]:
        # return the codes of items in original grouped axis
        # 返回原始分组轴上的项目的编码
        return self.ids

    @final
    @cache_readonly
    # 返回结果索引的长度，即分组结果的数量
    def ngroups(self) -> int:
        return len(self.result_index)

    # 返回结果索引和标识符的第一个元素，通常是结果的索引
    @property
    def result_index(self) -> Index:
        return self.result_index_and_ids[0]

    # 返回结果的标识符数组，通常是分组结果的标识符
    @property
    def ids(self) -> npt.NDArray[np.intp]:
        return self.result_index_and_ids[1]

    # 使用缓存机制，返回观察到的分组器，如果所有分组都已观察到
    # 则返回当前对象自身，否则返回一个新的观察到的分组器对象
    @cache_readonly
    @property
    def observed_grouper(self) -> BaseGrouper:
        if all(ping._observed for ping in self.groupings):
            return self
        return self._observed_grouper

    # 使用缓存机制，返回一个新的观察到的分组器对象
    @cache_readonly
    def _observed_grouper(self) -> BaseGrouper:
        # 从所有分组中获取观察到的分组对象的列表
        groupings = [ping.observed_grouping for ping in self.groupings]
        # 创建一个基础分组器对象，基于给定的参数
        grouper = BaseGrouper(self.axis, groupings, sort=self._sort, dropna=self.dropna)
        return grouper

    # 返回观察到的索引和标识符的元组，通常用于已观察到的分组
    def _ob_index_and_ids(
        self,
        levels: list[Index],
        codes: list[npt.NDArray[np.intp]],
        names: list[Hashable],
    ) -> tuple[MultiIndex, npt.NDArray[np.intp]]:
        # 计算索引的形状
        shape = tuple(len(level) for level in levels)
        # 获取分组索引并压缩，返回观察到的标识符和观察到的分组索引
        group_index = get_group_index(codes, shape, sort=True, xnull=True)
        ob_ids, obs_group_ids = compress_group_index(group_index, sort=self._sort)
        # 确保标识符是平台整数类型
        ob_ids = ensure_platform_int(ob_ids)
        # 解码观察到的分组索引代码
        ob_index_codes = decons_obs_group_ids(
            ob_ids, obs_group_ids, shape, codes, xnull=True
        )
        # 创建多级索引对象
        ob_index = MultiIndex(
            levels=levels,
            codes=ob_index_codes,
            names=names,
            verify_integrity=False,
        )
        # 再次确保标识符是平台整数类型
        ob_ids = ensure_platform_int(ob_ids)
        return ob_index, ob_ids

    # 返回未观察到的索引和标识符的元组，通常用于未观察到的分组
    def _unob_index_and_ids(
        self,
        levels: list[Index],
        codes: list[npt.NDArray[np.intp]],
        names: list[Hashable],
    ) -> tuple[MultiIndex, npt.NDArray[np.intp]]:
        # 计算索引的形状
        shape = tuple(len(level) for level in levels)
        # 获取分组索引，未观察到的情况
        unob_ids = get_group_index(codes, shape, sort=True, xnull=True)
        # 使用 MultiIndex 的静态方法创建一个全组合索引对象
        unob_index = MultiIndex.from_product(levels, names=names)
        # 确保标识符是平台整数类型
        unob_ids = ensure_platform_int(unob_ids)
        return unob_index, unob_ids

    # 返回分组级别的生成器，用于迭代获取分组的每个级别
    @final
    def get_group_levels(self) -> Generator[Index, None, None]:
        # 注意：仅从 _insert_inaxis_grouper 方法调用，该方法仅在 BaseGrouper 中调用，不会在 BinGrouper 中调用
        result_index = self.result_index
        if len(self.groupings) == 1:
            yield result_index
        else:
            for level in range(result_index.nlevels - 1, -1, -1):
                yield result_index.get_level_values(level)

    # ------------------------------------------------------------
    # 聚合函数

    # 执行 Cython 操作的方法，返回操作结果
    @final
    def _cython_operation(
        self,
        kind: str,
        values,
        how: str,
        axis: AxisInt,
        min_count: int = -1,
        **kwargs,
    ) -> ArrayLike:
        """
        返回一个 Cython 操作的返回值。
        """
        # 断言操作类型为 "transform" 或 "aggregate"
        assert kind in ["transform", "aggregate"]

        # 创建一个 WrappedCythonOp 实例，用于执行 Cython 操作
        cy_op = WrappedCythonOp(kind=kind, how=how, has_dropped_na=self.has_dropped_na)

        # 调用 cython_operation 方法执行 Cython 操作，并返回结果
        return cy_op.cython_operation(
            values=values,
            axis=axis,
            min_count=min_count,
            comp_ids=self.ids,
            ngroups=self.ngroups,
            **kwargs,
        )

    @final
    def agg_series(
        self, obj: Series, func: Callable, preserve_dtype: bool = False
    ) -> ArrayLike:
        """
        聚合 Series 的方法。

        Parameters
        ----------
        obj : Series
            要聚合的 Series 对象
        func : callable
            接受一个 Series 并返回标量的函数
        preserve_dtype : bool
            是否已知聚合操作会保持 dtype 不变。

        Returns
        -------
        np.ndarray or ExtensionArray
            返回聚合后的结果，可能是 np.ndarray 或 ExtensionArray
        """

        if not isinstance(obj._values, np.ndarray):
            # 如果 obj 的值不是 np.ndarray 类型，可能会更积极地保留 dtype
            # 因为 maybe_cast_pointwise_result 会在 _from_sequence 中进行尝试/捕获
            # 注意：这里假设 _from_sequence 已严格执行适当的类型转换
            preserve_dtype = True

        # 使用纯 Python 实现的方法对 Series 进行聚合
        result = self._aggregate_series_pure_python(obj, func)

        # 将结果转换为 numpy 数组，避免尝试将对象转换为浮点数
        npvalues = lib.maybe_convert_objects(result, try_float=False)
        
        if preserve_dtype:
            # 如果需要保留 dtype，则调用 maybe_cast_pointwise_result 进行处理
            out = maybe_cast_pointwise_result(npvalues, obj.dtype, numeric_only=True)
        else:
            out = npvalues
        
        return out

    @final
    def _aggregate_series_pure_python(
        self, obj: Series, func: Callable
    ) -> npt.NDArray[np.object_]:
        """
        使用纯 Python 对 Series 进行聚合。

        Parameters
        ----------
        obj : Series
            要聚合的 Series 对象
        func : callable
            接受一个 Series 并返回标量的函数

        Returns
        -------
        np.ndarray
            返回一个对象数组，包含聚合后的结果
        """
        # 创建一个对象数组，用于存储聚合后的结果，长度为 self.ngroups
        result = np.empty(self.ngroups, dtype="O")
        initialized = False

        # 获取用于分割 obj 的 Splitter 对象
        splitter = self._get_splitter(obj)

        # 遍历分割后的每个组
        for i, group in enumerate(splitter):
            # 对每个组应用给定的函数 func 进行聚合
            res = func(group)
            # 提取结果，确保结果的有效性
            res = extract_result(res)

            if not initialized:
                # 仅在第一次迭代时进行结果数组的验证
                check_result_array(res, group.dtype)
                initialized = True

            # 将聚合结果存储在结果数组中的对应位置
            result[i] = res

        return result

    @final
    def apply_groupwise(
        self, f: Callable, data: DataFrame | Series
    # 返回类型为元组，包含一个列表和一个布尔值
    ) -> tuple[list, bool]:
        # 标记是否有变异
        mutated = False
        # 获取数据分组的分隔器
        splitter = self._get_splitter(data)
        # 获取分组键
        group_keys = self.result_index
        # 存储计算结果的列表
        result_values = []

        # 调用 DataSplitter.__iter__ 方法来生成分组键和分隔器的迭代器
        zipped = zip(group_keys, splitter)

        # 遍历分组键和对应的分隔器组
        for key, group in zipped:
            # 设置分组对象的名称属性为当前键值 key
            object.__setattr__(group, "name", key)

            # 获取分组对象的轴
            group_axes = group.axes
            # 对分组对象应用函数 f
            res = f(group)
            # 检查是否有变异
            if not mutated and not _is_indexed_like(res, group_axes):
                mutated = True
            # 将函数应用后的结果添加到结果值列表中
            result_values.append(res)

        # 如果分组键为空且函数 f 是 functools.partial 对象，则进行特定处理
        if len(group_keys) == 0 and getattr(f, "__name__", None) in [
            "skew",
            "sum",
            "prod",
        ]:
            # 如果分组键为空，则调用 f(data.iloc[:0]) 来引发适当的 TypeError
            f(data.iloc[:0])

        # 返回计算结果列表和变异标志
        return result_values, mutated

    # ------------------------------------------------------------
    # 用于对 GroupBy 对象的子集进行排序的方法

    @final
    @cache_readonly
    def _sorted_ids(self) -> npt.NDArray[np.intp]:
        # 从 self.ids 中提取指定结果位置的排序后的 ids
        result = self.ids.take(self.result_ilocs)
        # 如果存在 dropna 属性并且为 True，则过滤掉小于 0 的结果
        if getattr(self, "dropna", True):
            # 对于 BinGrouper，没有 dropna 操作
            result = result[result >= 0]
        # 返回排序后的结果
        return result
class BinGrouper(BaseGrouper):
    """
    This is an internal Grouper class

    Parameters
    ----------
    bins : the split index of binlabels to group the item of axis
    binlabels : the label list
    indexer : np.ndarray[np.intp], optional
        the indexer created by Grouper
        some groupers (TimeGrouper) will sort its axis and its
        group_info is also sorted, so need the indexer to reorder

    Examples
    --------
    bins: [2, 4, 6, 8, 10]
    binlabels: DatetimeIndex(['2005-01-01', '2005-01-03',
        '2005-01-05', '2005-01-07', '2005-01-09'],
        dtype='datetime64[ns]', freq='2D')

    the group_info, which contains the label of each item in grouped
    axis, the index of label in label list, group number, is

    (array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]), array([0, 1, 2, 3, 4]), 5)

    means that, the grouped axis has 10 items, can be grouped into 5
    labels, the first and second items belong to the first label, the
    third and forth items belong to the second label, and so on

    """

    bins: npt.NDArray[np.int64]  # 定义一个成员变量 bins，类型为 np.int64 的 NumPy 数组
    binlabels: Index  # 定义一个成员变量 binlabels，类型为 Index（索引对象）

    def __init__(
        self,
        bins,
        binlabels,
        indexer=None,
    ) -> None:
        self.bins = ensure_int64(bins)  # 将传入的 bins 转换为 int64 类型并赋给 self.bins
        self.binlabels = ensure_index(binlabels)  # 确保 binlabels 是索引对象并赋给 self.binlabels
        self.indexer = indexer  # 将传入的 indexer 赋给 self.indexer

        # 检查 bins 和 binlabels 的长度是否相等，否则会导致后续调用 agg_series 报错
        assert len(self.binlabels) == len(self.bins)

    @cache_readonly
    def groups(self):
        """dict {group name -> group labels}"""
        # 创建一个字典，将 binlabels 和 bins 中对应的值组成键值对，但排除 binlabels 中的 NaT（不是时间戳的缺失值）
        result = {
            key: value
            for key, value in zip(self.binlabels, self.bins)
            if key is not NaT
        }
        return result

    @property
    def nkeys(self) -> int:
        # 返回 1，表示只有一个键（key）
        return 1

    @cache_readonly
    def codes_info(self) -> npt.NDArray[np.intp]:
        # 返回原始分组轴中项的代码（ids）
        ids = self.ids
        if self.indexer is not None:
            # 如果存在 indexer，则按照 (ids, indexer) 的排序进行排序
            sorter = np.lexsort((ids, self.indexer))
            ids = ids[sorter]
        return ids

    def get_iterator(self, data: NDFrame):
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
        slicer = lambda start, edge: data.iloc[start:edge]  # 定义一个切片函数 slicer

        start = 0
        for edge, label in zip(self.bins, self.binlabels):
            if label is not NaT:
                yield label, slicer(start, edge)  # 生成器，返回每个分组的名称和对应的子集对象
            start = edge

        if start < len(data):
            yield self.binlabels[-1], slicer(start, None)  # 返回最后一个分组的名称和对应的子集对象

    @cache_readonly
    # 返回一个字典，其中键是标签，值是对应的索引列表，每个标签对应一个 bin 区间
    def indices(self):
        indices = collections.defaultdict(list)

        i = 0
        for label, bin in zip(self.binlabels, self.bins):
            # 如果 i 小于 bin，则将标签与其对应的索引范围加入到 indices 中
            if i < bin:
                if label is not NaT:
                    indices[label] = list(range(i, bin))
                i = bin
        return indices

    # 返回一个列表，其中包含 self.ids 的单一元素列表
    @cache_readonly
    def codes(self) -> list[npt.NDArray[np.intp]]:
        return [self.ids]

    # 返回一个包含两个元素的元组，第一个元素是经过处理的结果索引，第二个元素是相应的 ids
    @cache_readonly
    def result_index_and_ids(self):
        result_index = self.binlabels
        # 如果 binlabels 的第一个元素是缺失值 NaT，则去除第一个元素
        if len(self.binlabels) != 0 and isna(self.binlabels[0]):
            result_index = result_index[1:]

        ngroups = len(result_index)
        # 计算每个组的重复次数
        rep = np.diff(np.r_[0, self.bins])

        # 确保 rep 的元素为整数类型，适应当前平台
        rep = ensure_platform_int(rep)
        # 根据不同情况生成 ids 数组
        if ngroups == len(self.bins):
            ids = np.repeat(np.arange(ngroups), rep)
        else:
            ids = np.repeat(np.r_[-1, np.arange(ngroups)], rep)
        ids = ensure_platform_int(ids)

        return result_index, ids

    # 返回一个列表，包含 self.binlabels 的单一元素列表
    @property
    def levels(self) -> list[Index]:
        return [self.binlabels]

    # 返回一个列表，包含 self.binlabels.name 的单一元素列表
    @property
    def names(self) -> list[Hashable]:
        return [self.binlabels.name]

    # 返回一个列表，其中包含一个 grouper.Grouping 对象，该对象封装了分组信息
    @property
    def groupings(self) -> list[grouper.Grouping]:
        lev = self.binlabels
        codes = self.ids
        # 获取 labels，即根据 codes 从 lev 中取出的标签
        labels = lev.take(codes)
        # 创建一个 grouper.Grouping 对象 ping，用于封装分组信息
        ping = grouper.Grouping(
            labels, labels, in_axis=False, level=None, uniques=lev._values
        )
        return [ping]

    # 返回当前对象本身，即 BinGrouper 对象
    @property
    def observed_grouper(self) -> BinGrouper:
        return self
# 检查对象是否按照给定的轴进行了索引
def _is_indexed_like(obj, axes) -> bool:
    if isinstance(obj, Series):  # 如果对象是 Pandas Series 类型
        if len(axes) > 1:  # 如果轴的数量大于1
            return False  # 返回 False
        return obj.index.equals(axes[0])  # 返回对象的索引是否与给定轴相同的比较结果
    elif isinstance(obj, DataFrame):  # 如果对象是 Pandas DataFrame 类型
        return obj.index.equals(axes[0])  # 返回对象的索引是否与给定轴相同的比较结果
    
    return False  # 其他情况下返回 False


# ----------------------------------------------------------------------
# Splitting / application

class DataSplitter(Generic[NDFrameT]):
    def __init__(
        self,
        data: NDFrameT,
        ngroups: int,
        *,
        sort_idx: npt.NDArray[np.intp],
        sorted_ids: npt.NDArray[np.intp],
    ) -> None:
        self.data = data  # 初始化数据
        self.ngroups = ngroups  # 初始化分组数目

        self._slabels = sorted_ids  # 设置排序后的标签
        self._sort_idx = sort_idx  # 设置排序的索引

    def __iter__(self) -> Iterator:
        if self.ngroups == 0:  # 如果分组数为0
            # 我们在一个生成器中，而不是引发 StopIteration
            # 我们只是返回来信号结束
            return

        starts, ends = lib.generate_slices(self._slabels, self.ngroups)  # 生成切片的起始和结束位置
        sdata = self._sorted_data  # 获取排序后的数据
        for start, end in zip(starts, ends):  # 遍历起始和结束位置的切片
            yield self._chop(sdata, slice(start, end))  # 生成器产生数据切片后的结果

    @cache_readonly
    def _sorted_data(self) -> NDFrameT:
        return self.data.take(self._sort_idx, axis=0)  # 返回按排序索引排序后的数据

    def _chop(self, sdata, slice_obj: slice) -> NDFrame:
        raise AbstractMethodError(self)  # 抽象方法错误，需要在子类中实现


class SeriesSplitter(DataSplitter):
    def _chop(self, sdata: Series, slice_obj: slice) -> Series:
        # 快速路径，相当于 `sdata.iloc[slice_obj]`
        mgr = sdata._mgr.get_slice(slice_obj)  # 获取管理器对象的切片
        ser = sdata._constructor_from_mgr(mgr, axes=mgr.axes)  # 使用管理器创建新的 Series 对象
        ser._name = sdata.name  # 设置新 Series 对象的名称
        return ser.__finalize__(sdata, method="groupby")  # 返回已完成最后操作的 Series 对象


class FrameSplitter(DataSplitter):
    def _chop(self, sdata: DataFrame, slice_obj: slice) -> DataFrame:
        # 快速路径，相当于：
        # return sdata.iloc[slice_obj]
        mgr = sdata._mgr.get_slice(slice_obj, axis=1)  # 获取管理器对象的切片，指定轴为1
        df = sdata._constructor_from_mgr(mgr, axes=mgr.axes)  # 使用管理器创建新的 DataFrame 对象
        return df.__finalize__(sdata, method="groupby")  # 返回已完成最后操作的 DataFrame 对象
```