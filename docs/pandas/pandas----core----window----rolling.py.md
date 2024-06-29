# `D:\src\scipysrc\pandas\pandas\core\window\rolling.py`

```
"""
Provide a generic structure to support window functions,
similar to how we have a Groupby object.
"""

# 导入必要的模块和函数
from __future__ import annotations

import copy  # 导入深拷贝函数
from datetime import timedelta  # 导入时间间隔类
from functools import partial  # 导入偏函数支持
import inspect  # 导入检查模块
from textwrap import dedent  # 导入缩进文本函数
from typing import (  # 导入类型提示相关模块
    TYPE_CHECKING,
    Any,
    Literal,
)

import numpy as np  # 导入numpy库

from pandas._libs.tslibs import (  # 导入时间序列相关模块
    BaseOffset,
    Timedelta,
    to_offset,
)
import pandas._libs.window.aggregations as window_aggregations  # 导入窗口聚合函数模块
from pandas.compat._optional import import_optional_dependency  # 导入可选依赖导入函数
from pandas.errors import DataError  # 导入数据错误异常类
from pandas.util._decorators import doc  # 导入文档装饰器函数

from pandas.core.dtypes.common import (  # 导入通用数据类型函数
    ensure_float64,
    is_bool,
    is_integer,
    is_numeric_dtype,
    needs_i8_conversion,
)
from pandas.core.dtypes.dtypes import ArrowDtype  # 导入Arrow数据类型
from pandas.core.dtypes.generic import (  # 导入泛型数据类型
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import notna  # 导入非空值判断函数

from pandas.core._numba import executor  # 导入Numba执行器
from pandas.core.algorithms import factorize  # 导入因子化函数
from pandas.core.apply import ResamplerWindowApply  # 导入重采样窗口应用类
from pandas.core.arrays import ExtensionArray  # 导入扩展数组类
from pandas.core.base import SelectionMixin  # 导入选择混合类
import pandas.core.common as com  # 导入通用函数模块
from pandas.core.indexers.objects import (  # 导入索引对象
    BaseIndexer,
    FixedWindowIndexer,
    GroupbyIndexer,
    VariableWindowIndexer,
)
from pandas.core.indexes.api import (  # 导入索引API
    DatetimeIndex,
    Index,
    MultiIndex,
    PeriodIndex,
    TimedeltaIndex,
)
from pandas.core.reshape.concat import concat  # 导入连接函数
from pandas.core.util.numba_ import (  # 导入Numba工具函数
    get_jit_arguments,
    maybe_use_numba,
)
from pandas.core.window.common import (  # 导入窗口通用函数
    flex_binary_moment,
    zsqrt,
)
from pandas.core.window.doc import (  # 导入窗口文档相关模块
    _shared_docs,
    create_section_header,
    kwargs_numeric_only,
    kwargs_scipy,
    numba_notes,
    template_header,
    template_returns,
    template_see_also,
    window_agg_numba_parameters,
    window_apply_parameters,
)
from pandas.core.window.numba_ import (  # 导入Numba窗口函数
    generate_manual_numpy_nan_agg_with_axis,
    generate_numba_apply_func,
    generate_numba_table_func,
)

if TYPE_CHECKING:
    from collections.abc import Callable  # 导入可调用抽象基类
    from collections.abc import (  # 导入集合抽象基类
        Hashable,
        Iterator,
        Sized,
    )

    from pandas._typing import (  # 导入Pandas类型提示
        ArrayLike,
        NDFrameT,
        QuantileInterpolation,
        WindowingRankType,
        npt,
    )

    from pandas import (  # 导入Pandas核心对象
        DataFrame,
        Series,
    )
    from pandas.core.generic import NDFrame  # 导入泛型数据框架类
    from pandas.core.groupby.ops import BaseGrouper  # 导入基础分组器类

from pandas.core.arrays.datetimelike import dtype_to_unit  # 导入日期时间数组类型函数


class BaseWindow(SelectionMixin):
    """Provides utilities for performing windowing operations."""

    _attributes: list[str] = []  # 初始化属性列表
    exclusions: frozenset[Hashable] = frozenset()  # 初始化排除集合
    _on: Index  # 窗口对象基于的索引
    def __init__(
        self,
        obj: NDFrame,
        window=None,
        min_periods: int | None = None,
        center: bool | None = False,
        win_type: str | None = None,
        on: str | Index | None = None,
        closed: str | None = None,
        step: int | None = None,
        method: str = "single",
        *,
        selection=None,
    ) -> None:
        # 初始化方法，用于创建一个新的实例
        self.obj = obj  # 将传入的数据对象存储到实例变量中
        self.on = on  # 存储窗口函数操作的时间轴或列名称
        self.closed = closed  # 存储闭合方式的字符串表示
        self.step = step  # 存储步长参数
        self.window = window  # 存储窗口大小参数
        self.min_periods = min_periods  # 存储最小周期数参数
        self.center = center  # 存储是否在窗口中心操作的布尔值
        self.win_type = win_type  # 存储窗口类型的字符串表示
        self.method = method  # 存储窗口函数的计算方法
        self._win_freq_i8: int | None = None  # 初始化一个私有变量用于存储频率
        # 如果未指定时间轴，使用数据对象的索引作为默认时间轴
        if self.on is None:
            self._on = self.obj.index
        # 如果指定的时间轴是一个索引对象，直接使用它作为私有时间轴
        elif isinstance(self.on, Index):
            self._on = self.on
        # 如果数据对象是一个 DataFrame，并且指定的时间轴是其列之一，创建一个新的索引对象
        elif isinstance(self.obj, ABCDataFrame) and self.on in self.obj.columns:
            self._on = Index(self.obj[self.on])
        # 如果以上条件都不满足，则抛出数值错误，说明指定的时间轴无效
        else:
            raise ValueError(
                f"invalid on specified as {self.on}, "
                "must be a column (of DataFrame), an Index or None"
            )

        self._selection = selection  # 存储选择器参数
        self._validate()  # 调用私有方法进行参数验证
    # 验证方法，确保参数设置正确
    def _validate(self) -> None:
        # 如果 center 参数不为空且不是布尔型，抛出数值错误异常
        if self.center is not None and not is_bool(self.center):
            raise ValueError("center must be a boolean")
        
        # 如果 min_periods 参数不为空
        if self.min_periods is not None:
            # 如果 min_periods 不是整数，抛出数值错误异常
            if not is_integer(self.min_periods):
                raise ValueError("min_periods must be an integer")
            # 如果 min_periods 小于 0，抛出数值错误异常
            if self.min_periods < 0:
                raise ValueError("min_periods must be >= 0")
            # 如果 window 是整数且 min_periods 大于 window，抛出数值错误异常
            if is_integer(self.window) and self.min_periods > self.window:
                raise ValueError(
                    f"min_periods {self.min_periods} must be <= window {self.window}"
                )
        
        # 如果 closed 参数不为空且不在预定义的值列表中，抛出数值错误异常
        if self.closed is not None and self.closed not in [
            "right",
            "both",
            "left",
            "neither",
        ]:
            raise ValueError("closed must be 'right', 'left', 'both' or 'neither'")
        
        # 如果 obj 不是 Series 或 DataFrame 类型，抛出类型错误异常
        if not isinstance(self.obj, (ABCSeries, ABCDataFrame)):
            raise TypeError(f"invalid type: {type(self)}")
        
        # 如果 window 是 BaseIndexer 的子类实例
        if isinstance(self.window, BaseIndexer):
            # 验证传入的 BaseIndexer 子类是否具有正确的 get_window_bounds 签名
            get_window_bounds_signature = inspect.signature(
                self.window.get_window_bounds
            ).parameters.keys()
            expected_signature = inspect.signature(
                BaseIndexer().get_window_bounds
            ).parameters.keys()
            # 如果签名不符合预期，抛出数值错误异常
            if get_window_bounds_signature != expected_signature:
                raise ValueError(
                    f"{type(self.window).__name__} does not implement "
                    f"the correct signature for get_window_bounds"
                )
        
        # 如果 method 不在预定义的值列表中，抛出数值错误异常
        if self.method not in ["table", "single"]:
            raise ValueError("method must be 'table' or 'single")
        
        # 如果 step 参数不为空
        if self.step is not None:
            # 如果 step 不是整数，抛出数值错误异常
            if not is_integer(self.step):
                raise ValueError("step must be an integer")
            # 如果 step 小于 0，抛出数值错误异常
            if self.step < 0:
                raise ValueError("step must be >= 0")

    # 检查窗口边界，确保起始和结束点的一致性
    def _check_window_bounds(
        self, start: np.ndarray, end: np.ndarray, num_vals: int
    ) -> None:
        # 如果起始点和结束点的长度不一致，抛出数值错误异常
        if len(start) != len(end):
            raise ValueError(
                f"start ({len(start)}) and end ({len(end)}) bounds must be the "
                f"same length"
            )
        # 计算预期的起始点和结束点的长度，并检查是否与对象长度一致
        if len(start) != (num_vals + (self.step or 1) - 1) // (self.step or 1):
            raise ValueError(
                f"start and end bounds ({len(start)}) must be the same length "
                f"as the object ({num_vals}) divided by the step ({self.step}) "
                f"if given and rounded up"
            )

    # 根据指定步长切片索引
    def _slice_axis_for_step(self, index: Index, result: Sized | None = None) -> Index:
        """
        根据预设的步长切片索引。
        """
        # 如果 result 为空或者长度与索引长度相同，直接返回索引
        return (
            index
            if result is None or len(result) == len(index)
            else index[:: self.step]
        )
    def _validate_numeric_only(self, name: str, numeric_only: bool) -> None:
        """
        Validate numeric_only argument, raising if invalid for the input.

        Parameters
        ----------
        name : str
            Name of the operator (kernel).
        numeric_only : bool
            Value passed by user.
        """
        # 检查是否选择了单维数据，且要求仅为数值类型，但实际数据类型不是数值类型，则抛出未实现错误
        if (
            self._selected_obj.ndim == 1
            and numeric_only
            and not is_numeric_dtype(self._selected_obj.dtype)
        ):
            raise NotImplementedError(
                f"{type(self).__name__}.{name} does not implement numeric_only"
            )

    def _make_numeric_only(self, obj: NDFrameT) -> NDFrameT:
        """Subset DataFrame to numeric columns.

        Parameters
        ----------
        obj : DataFrame

        Returns
        -------
        obj subset to numeric-only columns.
        """
        # 从数据框中选择数值类型的列，并排除时间跨度列，返回结果
        result = obj.select_dtypes(include=["number"], exclude=["timedelta"])
        return result

    def _create_data(self, obj: NDFrameT, numeric_only: bool = False) -> NDFrameT:
        """
        Split data into blocks & return conformed data.
        """
        # 如果指定了 on 参数，并且不是索引对象，并且数据为二维，排除掉 on 参数所在的列
        if self.on is not None and not isinstance(self.on, Index) and obj.ndim == 2:
            obj = obj.reindex(columns=obj.columns.difference([self.on]))
        # 如果数据是多维的且要求仅为数值类型，则调用 _make_numeric_only 方法进行处理
        if obj.ndim > 1 and numeric_only:
            obj = self._make_numeric_only(obj)
        return obj

    def _gotitem(self, key, ndim, subset=None):
        """
        Sub-classes to define. Return a sliced object.

        Parameters
        ----------
        key : str / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        # 创建一个新的对象以防止别名问题
        if subset is None:
            subset = self.obj

        # 需要创建自身的浅拷贝，并保留相同的分组方式
        kwargs = {attr: getattr(self, attr) for attr in self._attributes}

        # 推断选择的子集
        selection = self._infer_selection(key, subset)
        # 使用推断的选择创建新的实例
        new_win = type(self)(subset, selection=selection, **kwargs)
        return new_win

    def __getattr__(self, attr: str):
        if attr in self._internal_names_set:
            return object.__getattribute__(self, attr)
        if attr in self.obj:
            return self[attr]

        # 如果找不到属性，则抛出 AttributeError 异常
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def _dir_additions(self):
        # 返回对象的 _dir_additions 方法结果
        return self.obj._dir_additions()
    def __repr__(self) -> str:
        """
        Provide a nice str repr of our rolling object.
        """
        # 生成一个字符串表示对象的方法，包含非私有属性的名称和对应的属性值
        attrs_list = (
            f"{attr_name}={getattr(self, attr_name)}"
            for attr_name in self._attributes
            if getattr(self, attr_name, None) is not None and attr_name[0] != "_"
        )
        # 将属性字符串列表连接成一个字符串，用逗号分隔
        attrs = ",".join(attrs_list)
        # 返回对象类型名称和其属性的字符串表示形式
        return f"{type(self).__name__} [{attrs}]"

    def __iter__(self) -> Iterator:
        # 根据选定的对象和轴创建数据对象
        obj = self._selected_obj.set_axis(self._on)
        # 创建数据的计算对象
        obj = self._create_data(obj)
        # 获取窗口索引器
        indexer = self._get_window_indexer()

        # 获取窗口的起始和结束位置
        start, end = indexer.get_window_bounds(
            num_values=len(obj),
            min_periods=self.min_periods,
            center=self.center,
            closed=self.closed,
            step=self.step,
        )
        # 检查窗口边界
        self._check_window_bounds(start, end, len(obj))

        # 遍历窗口的起始和结束位置，生成滚动窗口的结果
        for s, e in zip(start, end):
            result = obj.iloc[slice(s, e)]
            yield result

    def _prep_values(self, values: ArrayLike) -> np.ndarray:
        """Convert input to numpy arrays for Cython routines"""
        # 如果需要将数据类型转换为int64，则抛出未实现的错误
        if needs_i8_conversion(values.dtype):
            raise NotImplementedError(
                f"ops for {type(self).__name__} for this "
                f"dtype {values.dtype} are not implemented"
            )
        # 对于滚动操作，确保数据类型为float64
        # GH #12373 : 滚动函数在float32数据上报错，确保数据被强制转换为float64
        try:
            if isinstance(values, ExtensionArray):
                values = values.to_numpy(np.float64, na_value=np.nan)
            else:
                values = ensure_float64(values)
        except (ValueError, TypeError) as err:
            # 处理无法处理的数据类型错误
            raise TypeError(f"cannot handle this type -> {values.dtype}") from err

        # 将inf转换为nan，以便于C函数处理
        inf = np.isinf(values)
        if inf.any():
            values = np.where(inf, np.nan, values)

        # 返回处理后的数值数组
        return values
    def _insert_on_column(self, result: DataFrame, obj: DataFrame) -> None:
        """
        如果有一个 'on' 列，我们希望将其放回到结果中的相同位置
        """
        from pandas import Series

        if self.on is not None and not self._on.equals(obj.index):
            # 获取 self._on 的列名
            name = self._on.name
            # 创建一个 Series 对象，使用 self._on 的值，索引为 obj 的索引，列名为 name，不复制数据
            extra_col = Series(self._on, index=self.obj.index, name=name, copy=False)
            if name in result.columns:
                # 如果结果中已经存在同名列，考虑覆盖结果？
                # TODO: sure we want to overwrite results?
                result[name] = extra_col
            elif name in result.index.names:
                # 如果列名在结果的索引名称中，则跳过
                pass
            elif name in self._selected_obj.columns:
                # 在与 _selected_obj 中相同位置插入列
                old_cols = self._selected_obj.columns
                new_cols = result.columns
                # 获取旧列名中 name 的位置
                old_loc = old_cols.get_loc(name)
                # 获取新列中与旧列重叠的部分
                overlap = new_cols.intersection(old_cols[:old_loc])
                # 新列位置为重叠部分的长度
                new_loc = len(overlap)
                # 在新位置插入列
                result.insert(new_loc, name, extra_col)
            else:
                # 在末尾插入列
                result[name] = extra_col

    @property
    def _index_array(self) -> npt.NDArray[np.int64] | None:
        """
        返回一个索引数组，根据不同情况选择不同的数据类型
        """
        # 如果 self._on 是 PeriodIndex、DatetimeIndex 或 TimedeltaIndex 类型
        if isinstance(self._on, (PeriodIndex, DatetimeIndex, TimedeltaIndex)):
            return self._on.asi8  # 返回 _on 的整数表示形式
        # 如果 self._on 的数据类型是 ArrowDtype，并且其种类为 'm' 或 'M'
        elif isinstance(self._on.dtype, ArrowDtype) and self._on.dtype.kind in "mM":
            return self._on.to_numpy(dtype=np.int64)  # 返回 _on 的 numpy 数组表示形式，数据类型为 np.int64
        return None  # 其他情况返回 None

    def _resolve_output(self, out: DataFrame, obj: DataFrame) -> DataFrame:
        """
        验证和最终确定输出结果
        """
        # 如果 out 的列数为 0，且 obj 的列数大于 0，则抛出 DataError 异常
        if out.shape[1] == 0 and obj.shape[1] > 0:
            raise DataError("No numeric types to aggregate")
        # 如果 out 的列数为 0，则将 obj 转换为 float64 类型并返回
        if out.shape[1] == 0:
            return obj.astype("float64")

        # 插入列到输出结果中
        self._insert_on_column(out, obj)
        return out

    def _get_window_indexer(self) -> BaseIndexer:
        """
        返回一个索引器类，用于计算窗口的起始和结束边界
        """
        # 如果 self.window 是 BaseIndexer 类型，则直接返回
        if isinstance(self.window, BaseIndexer):
            return self.window
        # 如果存在 self._win_freq_i8，则返回 VariableWindowIndexer 类型的索引器
        if self._win_freq_i8 is not None:
            return VariableWindowIndexer(
                index_array=self._index_array,
                window_size=self._win_freq_i8,
                center=self.center,
            )
        # 否则返回 FixedWindowIndexer 类型的索引器，窗口大小为 self.window
        return FixedWindowIndexer(window_size=self.window)

    def _apply_series(
        self, homogeneous_func: Callable[..., ArrayLike], name: str | None = None
        ):
        """
        应用于 Series 的函数，执行特定操作
        """
    ) -> Series:
        """
        Series version of _apply_columnwise
        """
        # 使用所选对象创建数据
        obj = self._create_data(self._selected_obj)

        if name == "count":
            # GH 12541: 对于计数，特殊处理支持日期类型
            obj = notna(obj).astype(int)
        try:
            # 准备要应用的值，处理异常情况如类型错误或未实现错误
            values = self._prep_values(obj._values)
        except (TypeError, NotImplementedError) as err:
            # 如果没有可聚合的数值类型，抛出数据错误
            raise DataError("No numeric types to aggregate") from err

        # 应用同质函数到值数组
        result = homogeneous_func(values)
        # 为步长切片轴
        index = self._slice_axis_for_step(obj.index, result)
        # 使用结果构造新的 Series 对象并返回
        return obj._constructor(result, index=index, name=obj.name)

    def _apply_columnwise(
        self,
        homogeneous_func: Callable[..., ArrayLike],
        name: str,
        numeric_only: bool = False,
    ) -> DataFrame | Series:
        """
        Apply the given function to the DataFrame broken down into homogeneous
        sub-frames.
        """
        # 验证是否只适用于数值类型
        self._validate_numeric_only(name, numeric_only)
        if self._selected_obj.ndim == 1:
            # 如果所选对象是一维的，调用 _apply_series 处理
            return self._apply_series(homogeneous_func, name)

        # 使用所选对象创建数据，并根据需要限制为数值类型
        obj = self._create_data(self._selected_obj, numeric_only)
        if name == "count":
            # GH 12541: 对于计数，特殊处理支持日期类型
            obj = notna(obj).astype(int)
            # 合并对象管理器以提高性能
            obj._mgr = obj._mgr.consolidate()

        # 初始化结果值和索引列表
        taker = []
        res_values = []
        # 迭代处理每一列数组
        for i, arr in enumerate(obj._iter_column_arrays()):
            # GH#42736 按列而不是块操作
            # 从2.0版本开始，hfunc 将对无关列抛出异常
            try:
                # 准备要应用的值，处理异常情况如类型错误或未实现错误
                arr = self._prep_values(arr)
            except (TypeError, NotImplementedError) as err:
                # 如果无法聚合非数值类型，抛出数据错误
                raise DataError(
                    f"Cannot aggregate non-numeric type: {arr.dtype}"
                ) from err
            # 应用同质函数到数组并存储结果
            res = homogeneous_func(arr)
            res_values.append(res)
            taker.append(i)

        # 为步长切片轴
        index = self._slice_axis_for_step(
            obj.index, res_values[0] if len(res_values) > 0 else None
        )
        # 从数组构造 DataFrame 对象
        df = type(obj)._from_arrays(
            res_values,
            index=index,
            columns=obj.columns.take(taker),
            verify_integrity=False,
        )

        # 解析输出并返回
        return self._resolve_output(df, obj)

    def _apply_tablewise(
        self,
        homogeneous_func: Callable[..., ArrayLike],
        name: str | None = None,
        numeric_only: bool = False,
    ) -> DataFrame:
        """
        Apply the given function to the entire DataFrame.
        """
    ) -> DataFrame | Series:
        """
        Apply the given function to the DataFrame across the entire object
        """
        # 如果所选对象是一维的 Series，则抛出错误，因为方法'table'不适用于 Series 对象。
        if self._selected_obj.ndim == 1:
            raise ValueError("method='table' not applicable for Series objects.")
        # 根据当前选定的对象和 numeric_only 参数创建数据对象
        obj = self._create_data(self._selected_obj, numeric_only)
        # 准备要应用函数的数值数据，转换为 NumPy 数组
        values = self._prep_values(obj.to_numpy())
        # 使用给定的同类函数对值进行处理
        result = homogeneous_func(values)
        # 根据处理结果调整索引
        index = self._slice_axis_for_step(obj.index, result)
        # 根据处理结果调整列，如果结果列数与原始列数相同则保持不变，否则按步长选择列
        columns = (
            obj.columns
            if result.shape[1] == len(obj.columns)
            else obj.columns[:: self.step]
        )
        # 使用处理后的结果构造新的数据对象
        out = obj._constructor(result, index=index, columns=columns)

        # 解析输出结果并返回
        return self._resolve_output(out, obj)

    def _apply_pairwise(
        self,
        target: DataFrame | Series,
        other: DataFrame | Series | None,
        pairwise: bool | None,
        func: Callable[[DataFrame | Series, DataFrame | Series], DataFrame | Series],
        numeric_only: bool,
    ) -> DataFrame | Series:
        """
        Apply the given pairwise function given 2 pandas objects (DataFrame/Series)
        """
        # 根据目标对象和 numeric_only 参数创建数据对象
        target = self._create_data(target, numeric_only)
        # 如果 other 参数未指定，将其设置为目标对象本身
        if other is None:
            other = target
            # 只有默认值未设置
            pairwise = True if pairwise is None else pairwise
        # 如果 other 不是 DataFrame 或 Series 类型，则抛出错误
        elif not isinstance(other, (ABCDataFrame, ABCSeries)):
            raise ValueError("other must be a DataFrame or Series")
        # 如果 other 是二维且 numeric_only 为真，则将其转换为仅包含数值的对象
        elif other.ndim == 2 and numeric_only:
            other = self._make_numeric_only(other)

        # 调用灵活的二元矩阵运算函数，根据 pairwise 参数决定是否进行成对运算
        return flex_binary_moment(target, other, func, pairwise=bool(pairwise))

    def _apply(
        self,
        func: Callable[..., Any],
        name: str,
        numeric_only: bool = False,
        numba_args: tuple[Any, ...] = (),
        **kwargs,
        # 应用函数到当前对象，并根据参数调整行为
    ):
        """
        Rolling statistical measure using supplied function.

        Designed to be used with passed-in Cython array-based functions.

        Parameters
        ----------
        func : callable function to apply
            The function to apply over rolling windows.
        name : str,
            Name of the function being applied.
        numba_args : tuple
            Arguments to be passed when `func` is a numba function.
        **kwargs
            Additional arguments for rolling function and window function.

        Returns
        -------
        y : type of input
            Result of applying `func` over rolling windows.
        """
        # 获取窗口索引器
        window_indexer = self._get_window_indexer()
        # 确定最小数据点数，若未指定则使用窗口索引器的窗口大小
        min_periods = (
            self.min_periods
            if self.min_periods is not None
            else window_indexer.window_size
        )

        def homogeneous_func(values: np.ndarray):
            # 计算函数

            if values.size == 0:
                return values.copy()

            def calc(x):
                # 获取窗口边界
                start, end = window_indexer.get_window_bounds(
                    num_values=len(x),
                    min_periods=min_periods,
                    center=self.center,
                    closed=self.closed,
                    step=self.step,
                )
                # 检查窗口边界
                self._check_window_bounds(start, end, len(x))

                # 执行传入的计算函数
                return func(x, start, end, min_periods, *numba_args)

            with np.errstate(all="ignore"):
                result = calc(values)

            return result

        if self.method == "single":
            # 在单列上应用函数
            return self._apply_columnwise(homogeneous_func, name, numeric_only)
        else:
            # 在整个表上应用函数
            return self._apply_tablewise(homogeneous_func, name, numeric_only)
        ):
            # 获取窗口索引器对象
            window_indexer = self._get_window_indexer()
            # 确定最小周期数，如果未指定则使用窗口索引器的窗口大小
            min_periods = (
                self.min_periods
                if self.min_periods is not None
                else window_indexer.window_size
            )
            # 创建数据对象
            obj = self._create_data(self._selected_obj)
            # 准备数据值并转换为 NumPy 数组
            values = self._prep_values(obj.to_numpy())
            # 如果数据是一维的，则重新形状为二维的
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            # 获取窗口的起始和结束索引
            start, end = window_indexer.get_window_bounds(
                num_values=len(values),
                min_periods=min_periods,
                center=self.center,
                closed=self.closed,
                step=self.step,
            )
            # 检查窗口索引的边界是否有效
            self._check_window_bounds(start, end, len(values))
            # 目前，将所有值映射为浮点数，以匹配 Cython 实现，尽管这是错误的
            # TODO: 可能在将来保留正确的数据类型映射
            # xref #53214
            # 获取执行器中的浮点数数据类型映射
            dtype_mapping = executor.float_dtype_mapping
            # 生成共享的聚合器对象
            aggregator = executor.generate_shared_aggregator(
                func,
                dtype_mapping,
                is_grouped_kernel=False,
                **get_jit_arguments(engine_kwargs),
            )
            # 应用聚合操作并获取结果
            result = aggregator(
                values.T, start=start, end=end, min_periods=min_periods, **func_kwargs
            ).T
            # 对结果进行切片以适应对象的索引
            index = self._slice_axis_for_step(obj.index, result)
            # 如果对象是一维的，则压缩结果并返回构造的对象
            if obj.ndim == 1:
                result = result.squeeze()
                out = obj._constructor(result, index=index, name=obj.name)
                return out
            else:
                # 对结果的列进行切片以适应对象的列索引
                columns = self._slice_axis_for_step(obj.columns, result.T)
                out = obj._constructor(result, index=index, columns=columns)
                # 解析输出对象以适应输入的对象
                return self._resolve_output(out, obj)

        # 定义 aggregate 方法的别名 agg
        def aggregate(self, func, *args, **kwargs):
            # 使用 ResamplerWindowApply 对象进行聚合操作
            result = ResamplerWindowApply(self, func, args=args, kwargs=kwargs).agg()
            # 如果结果为空，则调用 apply 方法处理原始数据
            if result is None:
                return self.apply(func, raw=False, args=args, kwargs=kwargs)
            return result

        # 将 aggregate 方法的别名 agg 暴露为类的方法
        agg = aggregate
    # 定义一个继承自 BaseWindow 的窗口分组类，提供分组窗口操作的功能。
    class BaseWindowGroupby(BaseWindow):
        """
        Provide the groupby windowing facilities.
        """

        # 声明一个私有属性 _grouper，用于存储分组器对象
        _grouper: BaseGrouper
        # 声明一个布尔型属性 _as_index，指示是否以分组键作为索引
        _as_index: bool
        # 声明一个属性 _attributes，包含字符串列表 ["_grouper"]
        _attributes: list[str] = ["_grouper"]

        def __init__(
            self,
            obj: DataFrame | Series,
            *args,
            _grouper: BaseGrouper,
            _as_index: bool = True,
            **kwargs,
        ) -> None:
            # 导入 pandas.core.groupby.ops 中的 BaseGrouper 类
            from pandas.core.groupby.ops import BaseGrouper

            # 如果 _grouper 不是 BaseGrouper 类型的对象，则抛出 ValueError 异常
            if not isinstance(_grouper, BaseGrouper):
                raise ValueError("Must pass a BaseGrouper object.")
            # 将传入的 _grouper 参数赋值给实例的 _grouper 属性
            self._grouper = _grouper
            # 将传入的 _as_index 参数赋值给实例的 _as_index 属性
            self._as_index = _as_index
            # GH 32262: 惯例上，保持分组列在 groupby.<agg_func> 中，但在 groupby.rolling.<agg_func> 中对用户来说是意外的
            # 在 obj 中删除包含在 self._grouper.names 中的列，忽略可能出现的错误
            obj = obj.drop(columns=self._grouper.names, errors="ignore")
            # GH 15354
            # 如果 kwargs 中存在 "step" 参数，则抛出 NotImplementedError 异常
            if kwargs.get("step") is not None:
                raise NotImplementedError("step not implemented for groupby")
            # 调用父类 BaseWindow 的构造函数，初始化对象
            super().__init__(obj, *args, **kwargs)

        def _apply(
            self,
            func: Callable[..., Any],
            name: str,
            numeric_only: bool = False,
            numba_args: tuple[Any, ...] = (),
            **kwargs,
        ) -> DataFrame | Series:
        # 调用父类的_apply方法，传入参数进行计算
        result = super()._apply(
            func,
            name,
            numeric_only,
            numba_args,
            **kwargs,
        )
        # 重建结果的MultiIndex
        # 第一组级别为分组标签
        # 第二组级别为原始DataFrame/Series的索引
        grouped_object_index = self.obj.index
        grouped_index_name = [*grouped_object_index.names]
        groupby_keys = copy.copy(self._grouper.names)
        result_index_names = groupby_keys + grouped_index_name

        # 需要丢弃的列为那些不在原始DataFrame/Series索引中或为None的分组键
        drop_columns = [
            key
            for key in self._grouper.names
            if key not in self.obj.index.names or key is None
        ]

        if len(drop_columns) != len(groupby_keys):
            # 如果有丢弃的列，则从结果中删除这些列（忽略错误）
            result = result.drop(columns=drop_columns, errors="ignore")

        # 获取分组的编码和级别
        codes = self._grouper.codes
        levels = copy.copy(self._grouper.levels)

        # 获取分组的索引并合并成一个数组
        group_indices = self._grouper.indices.values()
        if group_indices:
            indexer = np.concatenate(list(group_indices))
        else:
            indexer = np.array([], dtype=np.intp)
        codes = [c.take(indexer) for c in codes]

        # 如果需要保留原始DataFrame的索引，则将重新排序后的索引附加到分组的编码和级别中
        if grouped_object_index is not None:
            idx = grouped_object_index.take(indexer)
            if not isinstance(idx, MultiIndex):
                idx = MultiIndex.from_arrays([idx])
            codes.extend(list(idx.codes))
            levels.extend(list(idx.levels))

        # 创建新的MultiIndex对象作为结果的索引
        result_index = MultiIndex(
            levels, codes, names=result_index_names, verify_integrity=False
        )

        result.index = result_index
        # 如果不需要作为索引，则将索引重置为一般的整数索引
        if not self._as_index:
            result = result.reset_index(level=list(range(len(groupby_keys))))
        return result

    def _apply_pairwise(
        self,
        target: DataFrame | Series,
        other: DataFrame | Series | None,
        pairwise: bool | None,
        func: Callable[[DataFrame | Series, DataFrame | Series], DataFrame | Series],
        numeric_only: bool,
    ):
        # 在两个对象之间应用函数，返回结果
        ...

    def _create_data(self, obj: NDFrameT, numeric_only: bool = False) -> NDFrameT:
        """
        将数据分割成块并返回符合条件的数据。
        """
        # 确保我们要滚动的对象相对于分组是单调排序的
        # GH 36197
        if not obj.empty:
            groupby_order = np.concatenate(list(self._grouper.indices.values())).astype(
                np.int64
            )
            obj = obj.take(groupby_order)
        return super()._create_data(obj, numeric_only)
    def _gotitem(self, key, ndim, subset=None):
        # 设置索引到实际对象上
        # 这样我们的索引就能够传递到选择的对象中
        # 当我们为 groupby 进行拆分时
        if self.on is not None:
            # GH 43355
            # 如果有指定的 on 列，将对象按照该列设置索引
            subset = self.obj.set_index(self._on)
        # 调用父类的 _gotitem 方法，处理索引和子集
        return super()._gotitem(key, ndim, subset=subset)
class Window(BaseWindow):
    """
    Provide rolling window calculations.

    Parameters
    ----------
    window : int, timedelta, str, offset, or BaseIndexer subclass
        Size of the moving window.

        If an integer, the fixed number of observations used for
        each window.

        If a timedelta, str, or offset, the time period of each window. Each
        window will be a variable sized based on the observations included in
        the time-period. This is only valid for datetimelike indexes.
        To learn more about the offsets & frequency strings, please see
        :ref:`this link<timeseries.offset_aliases>`.

        If a BaseIndexer subclass, the window boundaries
        based on the defined ``get_window_bounds`` method. Additional rolling
        keyword arguments, namely ``min_periods``, ``center``, ``closed`` and
        ``step`` will be passed to ``get_window_bounds``.

    min_periods : int, default None
        Minimum number of observations in window required to have a value;
        otherwise, result is ``np.nan``.

        For a window that is specified by an offset, ``min_periods`` will default to 1.

        For a window that is specified by an integer, ``min_periods`` will default
        to the size of the window.

    center : bool, default False
        If False, set the window labels as the right edge of the window index.

        If True, set the window labels as the center of the window index.

    win_type : str, default None
        If ``None``, all points are evenly weighted.

        If a string, it must be a valid `scipy.signal window function
        <https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows>`__.

        Certain Scipy window types require additional parameters to be passed
        in the aggregation function. The additional parameters must match
        the keywords specified in the Scipy window type method signature.

    on : str, optional
        For a DataFrame, a column label or Index level on which
        to calculate the rolling window, rather than the DataFrame's index.

        Provided integer column is ignored and excluded from result since
        an integer index is not used to calculate the rolling window.

    closed : str, default None
        If ``'right'``, the first point in the window is excluded from calculations.

        If ``'left'``, the last point in the window is excluded from calculations.

        If ``'both'``, no point in the window is excluded from calculations.

        If ``'neither'``, the first and last points in the window are excluded
        from calculations.

        Default ``None`` (``'right'``).

    step : int, default None
        Evaluate the window at every ``step`` result, equivalent to slicing as
        ``[::step]``. ``window`` must be an integer. Using a step argument other
        than None or 1 will produce a result with a different shape than the input.

        .. versionadded:: 1.5.0

    """
    method : str {'single', 'table'}, default 'single'
        # 定义方法参数，可选取值为'single'或'table'，默认为'single'
        .. versionadded:: 1.3.0
        # 指定引入的版本号，此功能从版本1.3.0开始支持
        Execute the rolling operation per single column or row (``'single'``)
        or over the entire object (``'table'``).
        # 根据指定的方法参数执行滚动操作，可以是针对单个列或行（'single'），也可以是整个对象（'table'）
        This argument is only implemented when specifying ``engine='numba'``
        in the method call.
        # 此参数仅在方法调用中指定 ``engine='numba'`` 时实现

    Returns
    -------
    pandas.api.typing.Window or pandas.api.typing.Rolling
        # 返回值为 pandas 中的 Window 或 Rolling 实例对象
        An instance of Window is returned if ``win_type`` is passed. Otherwise,
        an instance of Rolling is returned.
        # 如果传入了 ``win_type`` 参数，则返回一个 Window 实例；否则返回一个 Rolling 实例

    See Also
    --------
    expanding : Provides expanding transformations.
    ewm : Provides exponential weighted functions.
        # 参见相关函数 expanding 和 ewm，提供了扩展转换和指数加权功能

    Notes
    -----
    See :ref:`Windowing Operations <window.generic>` for further usage details
    and examples.
    # 参见窗口操作的详细使用细节和示例

    Examples
    --------
    >>> df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    # 创建一个包含列'B'的 DataFrame，列值为 [0, 1, 2, NaN, 4]

    **window**

    Rolling sum with a window length of 2 observations.
    # 使用长度为2的窗口进行滚动求和操作

    >>> df.rolling(2).sum()
    # 对 DataFrame df 应用窗口大小为2的滚动求和操作

    Rolling sum with a window span of 2 seconds.
    # 使用2秒的时间跨度进行滚动求和操作

    >>> df_time = pd.DataFrame(
    ...     {"B": [0, 1, 2, np.nan, 4]},
    ...     index=[
    ...         pd.Timestamp("20130101 09:00:00"),
    ...         pd.Timestamp("20130101 09:00:02"),
    ...         pd.Timestamp("20130101 09:00:03"),
    ...         pd.Timestamp("20130101 09:00:05"),
    ...         pd.Timestamp("20130101 09:00:06"),
    ...     ],
    ... )
    # 创建一个带有时间索引的 DataFrame df_time，包含列'B'，时间索引分别为指定的时间点

    >>> df_time.rolling("2s").sum()
    # 对 df_time 应用2秒的时间跨度进行滚动求和操作

    Rolling sum with forward looking windows with 2 observations.
    # 使用长度为2的前向窗口进行滚动求和操作

    >>> indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)
    # 创建一个固定前向窗口索引器，窗口大小为2

    >>> df.rolling(window=indexer, min_periods=1).sum()
    # 对 DataFrame df 应用自定义窗口索引器和最小周期数1的滚动求和操作

    **min_periods**

    Rolling sum with a window length of 2 observations, but only needs a minimum of 1
    observation to calculate a value.
    # 使用长度为2的窗口进行滚动求和操作，但是仅需要最少1个观察值来计算结果

    >>> df.rolling(2, min_periods=1).sum()
    # 对 DataFrame df 应用窗口大小为2，最小周期数为1的滚动求和操作

    **center**

    Rolling sum with the result assigned to the center of the window index.
    # 将结果分配给窗口索引的中心进行滚动求和操作

    >>> df.rolling(3, min_periods=1, center=True).sum()
    # 对 DataFrame df 应用窗口大小为3，最小周期数为1，中心为True的滚动求和操作

    >>> df.rolling(3, min_periods=1, center=False).sum()
    # 对 DataFrame df 应用窗口大小为3，最小周期数为1，中心为False的滚动求和操作

    **step**

    Rolling sum with a window length of 2 observations, minimum of 1 observation to
    calculate a value, and a step of 2.
    # 使用长度为2的窗口进行滚动求和操作，最少需要1个观察值来计算结果，步长为2

    >>> df.rolling(2, min_periods=1, step=2).sum()
    # 对 DataFrame df 应用窗口大小为2，最小周期数为1，步长为2的滚动求和操作
    """
    _attributes = [
        "window",
        "min_periods",
        "center",
        "win_type",
        "on",
        "closed",
        "step",
        "method",
    ]

    定义了一个列表 `_attributes`，包含了对象可能具有的属性名称。

    def _validate(self) -> None:
        super()._validate()

        调用父类的 `_validate` 方法，执行基础的验证操作。

        如果 `win_type` 不是字符串类型，抛出数值错误异常。
        
        signal = import_optional_dependency(
            "scipy.signal.windows", extra="Scipy is required to generate window weight."
        )
        self._scipy_weight_generator = getattr(signal, self.win_type, None)
        如果无法从 `scipy.signal.windows` 模块中获取 `win_type` 对应的属性，抛出数值错误异常。

        如果 `window` 是 `BaseIndexer` 的实例，抛出未实现错误异常。
        
        如果 `window` 不是整数或小于0，抛出数值错误异常。

        如果 `method` 不是 `'single'`，抛出未实现错误异常。

    def _center_window(self, result: np.ndarray, offset: int) -> np.ndarray:
        """
        Center the result in the window for weighted rolling aggregations.
        """
        如果 `offset` 大于0，则从结果中间进行截取，以便在加权滚动聚合中心化结果。

    def _apply(
        self,
        func: Callable[[np.ndarray, int, int], np.ndarray],
        name: str,
        numeric_only: bool = False,
        numba_args: tuple[Any, ...] = (),
        **kwargs,
        
    """
    ):
        """
        Rolling with weights statistical measure using supplied function.

        Designed to be used with passed-in Cython array-based functions.

        Parameters
        ----------
        func : callable function to apply
            要应用的可调用函数
        name : str,
            名称字符串
        numeric_only : bool, default False
            是否仅在布尔、整数和浮点数列上操作
        numba_args : tuple
            未使用
        **kwargs
            如果需要，用于 scipy 窗口的额外参数

        Returns
        -------
        y : type of input
            输入类型的返回值
        """
        # "None" not callable  [misc]
        window = self._scipy_weight_generator(  # type: ignore[misc]
            self.window, **kwargs
        )
        offset = (len(window) - 1) // 2 if self.center else 0

        def homogeneous_func(values: np.ndarray):
            # calculation function

            if values.size == 0:
                return values.copy()

            def calc(x):
                additional_nans = np.array([np.nan] * offset)
                x = np.concatenate((x, additional_nans))
                return func(
                    x,
                    window,
                    self.min_periods if self.min_periods is not None else len(window),
                )

            with np.errstate(all="ignore"):
                # Our weighted aggregations return memoryviews
                result = np.asarray(calc(values))

            if self.center:
                result = self._center_window(result, offset)

            return result

        return self._apply_columnwise(homogeneous_func, name, numeric_only)[
            :: self.step
        ]

    @doc(
        _shared_docs["aggregate"],
        see_also=dedent(
            """
        See Also
        --------
        DataFrame.aggregate : Similar DataFrame method.
        Series.aggregate : Similar Series method.
            类似的 Series 方法
        """
        ),
        examples=dedent(
            """
        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> df.rolling(2, win_type="boxcar").agg("mean")
             A    B    C
        0  NaN  NaN  NaN
        1  1.5  4.5  7.5
        2  2.5  5.5  8.5
            """
        ),
        klass="Series/DataFrame",
        axis="",
    )
    def aggregate(self, func, *args, **kwargs):
        result = ResamplerWindowApply(self, func, args=args, kwargs=kwargs).agg()
        if result is None:
            # these must apply directly
            result = func(self)

        return result

    agg = aggregate
    # 定义一个带有文档注释的函数装饰器，用于为 sum 方法添加文档
    @doc(
        template_header,  # 引用预定义的文档模板标题
        create_section_header("Parameters"),  # 创建参数部分的标题
        kwargs_numeric_only,  # 引用预定义的数值参数
        kwargs_scipy,  # 引用预定义的 SciPy 参数
        create_section_header("Returns"),  # 创建返回值部分的标题
        template_returns,  # 引用预定义的返回值模板
        create_section_header("See Also"),  # 创建相关内容部分的标题
        template_see_also,  # 引用预定义的相关内容模板
        create_section_header("Examples"),  # 创建示例部分的标题
        dedent(  # 移除缩进的示例代码，展示如何使用 sum 方法
            """\
        >>> ser = pd.Series([0, 1, 5, 2, 8])
    
        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.
    
        >>> type(ser.rolling(2, win_type='gaussian'))
        <class 'pandas.core.window.rolling.Window'>
    
        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method
        (`sum` in this case):
    
        >>> ser.rolling(2, win_type='gaussian').sum(std=3)
        0         NaN
        1    0.986207
        2    5.917243
        3    6.903450
        4    9.862071
        dtype: float64
        """
        ),
        window_method="rolling",  # 指定窗口方法为 rolling
        aggregation_description="weighted window sum",  # 描述聚合方法为加权窗口求和
        agg_method="sum",  # 指定聚合方法为 sum
    )
    def sum(self, numeric_only: bool = False, **kwargs):
        window_func = window_aggregations.roll_weighted_sum  # 定义窗口函数为 roll_weighted_sum
        # error: Argument 1 to "_apply" of "Window" has incompatible type
        # "Callable[[ndarray, ndarray, int], ndarray]"; expected
        # "Callable[[ndarray, int, int], ndarray]"
        # 调用 self._apply 方法，对窗口函数应用参数进行处理（忽略类型检查）
        return self._apply(
            window_func,  # type: ignore[arg-type]  # 对 window_func 类型进行忽略类型检查
            name="sum",  # 指定函数名称为 sum
            numeric_only=numeric_only,  # 指定是否仅对数值进行操作
            **kwargs,  # 传递额外的关键字参数
        )
    
    # 定义另一个带有文档注释的函数装饰器，用于为 mean 方法添加文档
    @doc(
        template_header,  # 引用预定义的文档模板标题
        create_section_header("Parameters"),  # 创建参数部分的标题
        kwargs_numeric_only,  # 引用预定义的数值参数
        kwargs_scipy,  # 引用预定义的 SciPy 参数
        create_section_header("Returns"),  # 创建返回值部分的标题
        template_returns,  # 引用预定义的返回值模板
        create_section_header("See Also"),  # 创建相关内容部分的标题
        template_see_also,  # 引用预定义的相关内容模板
        create_section_header("Examples"),  # 创建示例部分的标题
        dedent(  # 移除缩进的示例代码，展示如何使用 mean 方法
            """\
        >>> ser = pd.Series([0, 1, 5, 2, 8])
    
        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.
    
        >>> type(ser.rolling(2, win_type='gaussian'))
        <class 'pandas.core.window.rolling.Window'>
    
        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method:
    
        >>> ser.rolling(2, win_type='gaussian').mean(std=3)
        0    NaN
        1    0.5
        2    3.0
        3    3.5
        4    5.0
        dtype: float64
        """
        ),
        window_method="rolling",  # 指定窗口方法为 rolling
        aggregation_description="weighted window mean",  # 描述聚合方法为加权窗口求均值
        agg_method="mean",  # 指定聚合方法为 mean
    )
    def mean(self, numeric_only: bool = False, **kwargs):
        # 定义窗口函数为滚动加权平均
        window_func = window_aggregations.roll_weighted_mean
        # 调用父类方法 `_apply`，应用窗口函数进行均值计算
        # 忽略类型检查的错误，针对参数类型不匹配的问题
        return self._apply(
            window_func,  # type: ignore[arg-type]
            name="mean",
            numeric_only=numeric_only,
            **kwargs,
        )

    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        kwargs_scipy,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type='gaussian'))
        <class 'pandas.core.window.rolling.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method:

        >>> ser.rolling(2, win_type='gaussian').var(std=3)
        0     NaN
        1     0.5
        2     8.0
        3     4.5
        4    18.0
        dtype: float64
        """
        ),
        window_method="rolling",
        aggregation_description="weighted window variance",
        agg_method="var",
    )
    # 定义窗口函数为局部函数，使用部分函数应用设置自由度参数 ddof
    def var(self, ddof: int = 1, numeric_only: bool = False, **kwargs):
        window_func = partial(window_aggregations.roll_weighted_var, ddof=ddof)
        # 移除参数中的 'name' 键，以防在 `_apply` 方法中造成不必要的冲突
        kwargs.pop("name", None)
        # 调用父类方法 `_apply`，应用窗口函数进行方差计算
        return self._apply(window_func, name="var", numeric_only=numeric_only, **kwargs)
    # 使用 @doc 装饰器来注释该方法，指定文档的模板头部信息
    @doc(
        # 使用 template_header 作为模板的标题
        template_header,
        # 创建 "Parameters" 部分的标题
        create_section_header("Parameters"),
        # 指定仅接受数值类型参数的约束条件
        kwargs_numeric_only,
        # 使用 scipy 的参数模板
        kwargs_scipy,
        # 创建 "Returns" 部分的标题
        create_section_header("Returns"),
        # 使用 template_returns 作为返回值描述的模板
        template_returns,
        # 创建 "See Also" 部分的标题
        create_section_header("See Also"),
        # 使用 template_see_also 作为相关链接的模板
        template_see_also,
        # 创建 "Examples" 部分的标题，并缩进示例代码
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([0, 1, 5, 2, 8])

        To get an instance of :class:`~pandas.core.window.rolling.Window` we need
        to pass the parameter `win_type`.

        >>> type(ser.rolling(2, win_type='gaussian'))
        <class 'pandas.core.window.rolling.Window'>

        In order to use the `SciPy` Gaussian window we need to provide the parameters
        `M` and `std`. The parameter `M` corresponds to 2 in our example.
        We pass the second parameter `std` as a parameter of the following method:

        >>> ser.rolling(2, win_type='gaussian').std(std=3)
        0         NaN
        1    0.707107
        2    2.828427
        3    2.121320
        4    4.242641
        dtype: float64
        """
        ),
        # 指定窗口方法为 "rolling"
        window_method="rolling",
        # 描述聚合操作为加权窗口标准差
        aggregation_description="weighted window standard deviation",
        # 聚合方法为 "std"，即标准差
        agg_method="std",
    )
    # 定义方法 std，计算加权窗口标准差，接受自由度 ddof、仅数值类型参数 numeric_only 和额外的关键字参数 kwargs
    def std(self, ddof: int = 1, numeric_only: bool = False, **kwargs):
        # 调用 self.var 方法计算方差，并使用 zsqrt 函数对其求平方根，返回标准差
        return zsqrt(
            self.var(ddof=ddof, name="std", numeric_only=numeric_only, **kwargs)
        )
    # 定义 RollingAndExpandingMixin 类，继承自 BaseWindow
    class RollingAndExpandingMixin(BaseWindow):
        
        # 计算窗口内的元素数量
        def count(self, numeric_only: bool = False):
            # 使用 roll_sum 函数计算滚动窗口的元素和
            window_func = window_aggregations.roll_sum
            return self._apply(window_func, name="count", numeric_only=numeric_only)
        
        # 应用指定函数到窗口中的元素
        def apply(
            self,
            func: Callable[..., Any],
            raw: bool = False,
            engine: Literal["cython", "numba"] | None = None,
            engine_kwargs: dict[str, bool] | None = None,
            args: tuple[Any, ...] | None = None,
            kwargs: dict[str, Any] | None = None,
        ):
            # 如果 args 为 None，则设为空元组
            if args is None:
                args = ()
            # 如果 kwargs 为 None，则设为空字典
            if kwargs is None:
                kwargs = {}

            # 如果 raw 不是布尔值，则抛出 ValueError 异常
            if not is_bool(raw):
                raise ValueError("raw parameter must be `True` or `False`")

            # numba_args 初始化为空元组
            numba_args: tuple[Any, ...] = ()
            # 如果可能使用 numba 引擎
            if maybe_use_numba(engine):
                # 如果 raw 不为 True，则抛出 ValueError 异常
                if raw is False:
                    raise ValueError("raw must be `True` when using the numba engine")
                # 将 args 赋值给 numba_args
                numba_args = args
                # 根据 self.method 的值选择生成 numba 应用函数
                if self.method == "single":
                    apply_func = generate_numba_apply_func(
                        func, **get_jit_arguments(engine_kwargs, kwargs)
                    )
                else:
                    apply_func = generate_numba_table_func(
                        func, **get_jit_arguments(engine_kwargs, kwargs)
                    )
            # 如果 engine 是 'cython' 或 None
            elif engine in ("cython", None):
                # 如果 engine_kwargs 不为 None，则抛出 ValueError 异常
                if engine_kwargs is not None:
                    raise ValueError("cython engine does not accept engine_kwargs")
                # 使用 _generate_cython_apply_func 方法生成 Cython 应用函数
                apply_func = self._generate_cython_apply_func(args, kwargs, raw, func)
            else:
                # 否则抛出 ValueError 异常，要求 engine 必须为 'numba' 或 'cython'
                raise ValueError("engine must be either 'numba' or 'cython'")

            # 调用 _apply 方法，应用生成的函数到窗口中的元素
            return self._apply(
                apply_func,
                name="apply",
                numba_args=numba_args,
            )

        # 生成 Cython 应用函数
        def _generate_cython_apply_func(
            self,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            raw: bool | np.bool_,
            function: Callable[..., Any],
        ) -> Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]:
            # 从 pandas 导入 Series 类
            from pandas import Series

            # 使用 partial 函数创建 window_func，部分应用 roll_apply 函数
            window_func = partial(
                window_aggregations.roll_apply,
                args=args,
                kwargs=kwargs,
                raw=raw,
                function=function,
            )

            # 定义 apply_func 函数，接收 values, begin, end, min_periods, raw 参数
            def apply_func(values, begin, end, min_periods, raw=raw):
                # 如果 raw 不为 True，则将 values 转换为 Series 类对象
                if not raw:
                    # GH 45912
                    values = Series(values, index=self._on, copy=False)
                # 调用 window_func 计算并返回结果
                return window_func(values, begin, end, min_periods)

            # 返回定义的 apply_func 函数
            return apply_func

        # 计算窗口内元素的和
        def sum(
            self,
            numeric_only: bool = False,
            engine: Literal["cython", "numba"] | None = None,
            engine_kwargs: dict[str, bool] | None = None,
    ):
        # 如果引擎可能使用 numba 进行加速
        if maybe_use_numba(engine):
            # 如果方法是 "table"
            if self.method == "table":
                # 生成使用 np.nansum 的手动 numpy NaN 聚合函数
                func = generate_manual_numpy_nan_agg_with_axis(np.nansum)
                # 应用生成的函数到当前对象上
                return self.apply(
                    func,
                    raw=True,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                )
            else:
                # 从 pandas 内部导入滑动最小最大值的 numba 核函数
                from pandas.core._numba.kernels import sliding_sum

                # 使用滑动和函数进行 numba 加速应用
                return self._numba_apply(sliding_sum, engine_kwargs)
        # 如果不适合使用 numba 进行加速，则执行窗口聚合函数
        window_func = window_aggregations.roll_sum
        # 应用窗口函数到当前对象上，并返回结果
        return self._apply(window_func, name="sum", numeric_only=numeric_only)

    def max(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # 如果引擎可能使用 numba 进行加速
        if maybe_use_numba(engine):
            # 如果方法是 "table"
            if self.method == "table":
                # 生成使用 np.nanmax 的手动 numpy NaN 聚合函数
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmax)
                # 应用生成的函数到当前对象上
                return self.apply(
                    func,
                    raw=True,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                )
            else:
                # 从 pandas 内部导入滑动最小最大值的 numba 核函数
                from pandas.core._numba.kernels import sliding_min_max

                # 使用滑动最小最大值函数进行 numba 加速应用，指定为最大值
                return self._numba_apply(sliding_min_max, engine_kwargs, is_max=True)
        # 如果不适合使用 numba 进行加速，则执行窗口最大值聚合函数
        window_func = window_aggregations.roll_max
        # 应用窗口函数到当前对象上，并返回结果
        return self._apply(window_func, name="max", numeric_only=numeric_only)

    def min(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # 如果引擎可能使用 numba 进行加速
        if maybe_use_numba(engine):
            # 如果方法是 "table"
            if self.method == "table":
                # 生成使用 np.nanmin 的手动 numpy NaN 聚合函数
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmin)
                # 应用生成的函数到当前对象上
                return self.apply(
                    func,
                    raw=True,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                )
            else:
                # 从 pandas 内部导入滑动最小最大值的 numba 核函数
                from pandas.core._numba.kernels import sliding_min_max

                # 使用滑动最小最大值函数进行 numba 加速应用，指定为最小值
                return self._numba_apply(sliding_min_max, engine_kwargs, is_max=False)
        # 如果不适合使用 numba 进行加速，则执行窗口最小值聚合函数
        window_func = window_aggregations.roll_min
        # 应用窗口函数到当前对象上，并返回结果
        return self._apply(window_func, name="min", numeric_only=numeric_only)

    def mean(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # 如果引擎可能使用 numba 进行加速
        if maybe_use_numba(engine):
            # 如果方法是 "table"
            if self.method == "table":
                # 生成使用 np.nanmean 的手动 numpy NaN 聚合函数
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmean)
                # 应用生成的函数到当前对象上
                return self.apply(
                    func,
                    raw=True,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                )
            else:
                # 从 pandas 内部导入滑动平均值的 numba 核函数
                from pandas.core._numba.kernels import sliding_mean

                # 使用滑动平均值函数进行 numba 加速应用
                return self._numba_apply(sliding_mean, engine_kwargs)
        # 如果不适合使用 numba 进行加速，则执行窗口平均值聚合函数
        window_func = window_aggregations.roll_mean
        # 应用窗口函数到当前对象上，并返回结果
        return self._apply(window_func, name="mean", numeric_only=numeric_only)
    ):
        # 如果可以使用 numba 引擎，则尝试使用 numba 加速计算
        if maybe_use_numba(engine):
            # 如果聚合方法是 "table"，生成使用 np.nanmean 的手动 numpy 聚合函数
            if self.method == "table":
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmean)
                # 对数据应用生成的函数进行聚合操作，使用原始数据，指定引擎和引擎参数
                return self.apply(
                    func,
                    raw=True,
                    engine=engine,
                    engine_kwargs=engine_kwargs,
                )
            else:
                # 否则，从 pandas.core._numba.kernels 导入 sliding_mean 函数
                from pandas.core._numba.kernels import sliding_mean
                # 使用 sliding_mean 函数进行 numba 加速计算
                return self._numba_apply(sliding_mean, engine_kwargs)
        # 如果不使用 numba 引擎，则使用默认的窗口函数 roll_mean 进行计算
        window_func = window_aggregations.roll_mean
        # 调用 _apply 方法，应用 window_func 进行均值计算，指定操作名称为 "mean"，仅数值列参与计算
        return self._apply(window_func, name="mean", numeric_only=numeric_only)

    def median(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # 如果可以使用 numba 引擎，则尝试使用 numba 加速计算
        if maybe_use_numba(engine):
            # 如果聚合方法是 "table"，生成使用 np.nanmedian 的手动 numpy 聚合函数
            if self.method == "table":
                func = generate_manual_numpy_nan_agg_with_axis(np.nanmedian)
            else:
                # 否则，使用 np.nanmedian 函数
                func = np.nanmedian

            # 对数据应用生成的函数进行聚合操作，使用原始数据，指定引擎和引擎参数
            return self.apply(
                func,
                raw=True,
                engine=engine,
                engine_kwargs=engine_kwargs,
            )
        # 如果不使用 numba 引擎，则使用默认的窗口函数 roll_median_c 进行计算
        window_func = window_aggregations.roll_median_c
        # 调用 _apply 方法，应用 window_func 进行中位数计算，指定操作名称为 "median"，仅数值列参与计算
        return self._apply(window_func, name="median", numeric_only=numeric_only)

    def std(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # 如果可以使用 numba 引擎，则尝试使用 numba 加速计算
        if maybe_use_numba(engine):
            # 如果聚合方法是 "table"，抛出异常，不支持方法为 'table' 的标准差计算
            if self.method == "table":
                raise NotImplementedError("std not supported with method='table'")
            # 否则，从 pandas.core._numba.kernels 导入 sliding_var 函数
            from pandas.core._numba.kernels import sliding_var
            # 对数据应用 sliding_var 函数进行 numba 加速计算，返回标准差的平方根
            return zsqrt(self._numba_apply(sliding_var, engine_kwargs, ddof=ddof))
        # 如果不使用 numba 引擎，则使用默认的窗口函数 roll_var 进行计算
        window_func = window_aggregations.roll_var

        # 定义 zsqrt_func 函数，对标准差计算结果进行平方根处理
        def zsqrt_func(values, begin, end, min_periods):
            return zsqrt(window_func(values, begin, end, min_periods, ddof=ddof))

        # 调用 _apply 方法，应用 zsqrt_func 进行标准差计算，指定操作名称为 "std"，仅数值列参与计算
        return self._apply(
            zsqrt_func,
            name="std",
            numeric_only=numeric_only,
        )

    def var(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # 如果可以使用 numba 引擎，则尝试使用 numba 加速计算
        if maybe_use_numba(engine):
            # 如果聚合方法是 "table"，抛出异常，不支持方法为 'table' 的方差计算
            if self.method == "table":
                raise NotImplementedError("var not supported with method='table'")
            # 否则，从 pandas.core._numba.kernels 导入 sliding_var 函数
            from pandas.core._numba.kernels import sliding_var
            # 对数据应用 sliding_var 函数进行 numba 加速计算，返回方差计算结果
            return self._numba_apply(sliding_var, engine_kwargs, ddof=ddof)
        # 如果不使用 numba 引擎，则使用带有 ddof 参数的 window_aggregations.roll_var 函数进行计算
        window_func = partial(window_aggregations.roll_var, ddof=ddof)
        # 调用 _apply 方法，应用 window_func 进行方差计算，指定操作名称为 "var"，仅数值列参与计算
        return self._apply(
            window_func,
            name="var",
            numeric_only=numeric_only,
        )
    def skew(self, numeric_only: bool = False):
        # 定义窗口函数为 roll_skew，用于计算偏度
        window_func = window_aggregations.roll_skew
        # 调用 _apply 方法，将 roll_skew 应用到当前对象上，计算偏度
        return self._apply(
            window_func,
            name="skew",
            numeric_only=numeric_only,
        )

    def sem(self, ddof: int = 1, numeric_only: bool = False):
        # 检查是否为数值类型，如果不是，抛出异常，错误信息中指出 sem 而不是 std
        self._validate_numeric_only("sem", numeric_only)
        # 计算标准误差，std() 方法返回标准差，计算公式为 std / sqrt(count - ddof)
        return self.std(numeric_only=numeric_only) / (
            self.count(numeric_only=numeric_only) - ddof
        ).pow(0.5)

    def kurt(self, numeric_only: bool = False):
        # 定义窗口函数为 roll_kurt，用于计算峰度
        window_func = window_aggregations.roll_kurt
        # 调用 _apply 方法，将 roll_kurt 应用到当前对象上，计算峰度
        return self._apply(
            window_func,
            name="kurt",
            numeric_only=numeric_only,
        )

    def quantile(
        self,
        q: float,
        interpolation: QuantileInterpolation = "linear",
        numeric_only: bool = False,
    ):
        # 根据分位数 q 的值选择不同的窗口函数
        if q == 1.0:
            window_func = window_aggregations.roll_max
        elif q == 0.0:
            window_func = window_aggregations.roll_min
        else:
            # 创建一个部分应用的函数，用于计算指定分位数的滚动分位数
            window_func = partial(
                window_aggregations.roll_quantile,
                quantile=q,
                interpolation=interpolation,
            )

        # 调用 _apply 方法，将选择的窗口函数应用到当前对象上，计算分位数
        return self._apply(window_func, name="quantile", numeric_only=numeric_only)

    def rank(
        self,
        method: WindowingRankType = "average",
        ascending: bool = True,
        pct: bool = False,
        numeric_only: bool = False,
    ):
        # 创建一个部分应用的函数，用于计算滚动排名
        window_func = partial(
            window_aggregations.roll_rank,
            method=method,
            ascending=ascending,
            percentile=pct,
        )

        # 调用 _apply 方法，将 roll_rank 应用到当前对象上，计算排名
        return self._apply(window_func, name="rank", numeric_only=numeric_only)

    def cov(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        # 如果使用者指定了步长，则抛出未实现错误，因为协方差计算不支持步长参数
        if self.step is not None:
            raise NotImplementedError("step not implemented for cov")
        
        # 确保只有数值类型可以进行协方差计算
        self._validate_numeric_only("cov", numeric_only)

        # 从 pandas 库导入 Series 类
        from pandas import Series

        # 定义计算协方差的函数 cov_func，接受两个参数 x 和 y
        def cov_func(x, y):
            # 将输入的 x 转换为数组，并进行预处理
            x_array = self._prep_values(x)
            # 将输入的 y 转换为数组，并进行预处理
            y_array = self._prep_values(y)
            # 获取窗口索引器
            window_indexer = self._get_window_indexer()
            # 窗口最小期数设定为 self.min_periods 或者窗口索引器的窗口大小
            min_periods = (
                self.min_periods
                if self.min_periods is not None
                else window_indexer.window_size
            )
            # 获取窗口的起始和结束索引
            start, end = window_indexer.get_window_bounds(
                num_values=len(x_array),
                min_periods=min_periods,
                center=self.center,
                closed=self.closed,
                step=self.step,
            )
            # 检查窗口边界的有效性
            self._check_window_bounds(start, end, len(x_array))

            # 在忽略 NaN 的情况下计算 x_array * y_array 的滚动均值
            mean_x_y = window_aggregations.roll_mean(
                x_array * y_array, start, end, min_periods
            )
            # 在忽略 NaN 的情况下计算 x_array 的滚动均值
            mean_x = window_aggregations.roll_mean(x_array, start, end, min_periods)
            # 在忽略 NaN 的情况下计算 y_array 的滚动均值
            mean_y = window_aggregations.roll_mean(y_array, start, end, min_periods)
            # 计算非 NaN 元素的总数，即有效数据点数
            count_x_y = window_aggregations.roll_sum(
                notna(x_array + y_array).astype(np.float64), start, end, 0
            )
            # 计算协方差的结果
            result = (mean_x_y - mean_x * mean_y) * (count_x_y / (count_x_y - ddof))
            # 返回结果作为 Series 对象，保留输入 x 的索引和名称
            return Series(result, index=x.index, name=x.name, copy=False)

        # 应用 pairwise 方法来执行协方差计算
        return self._apply_pairwise(
            self._selected_obj, other, pairwise, cov_func, numeric_only
        )

    # 定义计算相关系数的方法 corr
    def corr(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        ddof: int = 1,
        numeric_only: bool = False,
        ):
        # 如果步长参数不为空，则抛出未实现错误，因为相关性计算不支持步长操作
        if self.step is not None:
            raise NotImplementedError("step not implemented for corr")
        
        # 验证是否只包含数值类型的数据，否则抛出异常
        self._validate_numeric_only("corr", numeric_only)

        # 导入 pandas 的 Series 类
        from pandas import Series

        # 定义计算相关性的函数 corr_func，接受两个参数 x 和 y
        def corr_func(x, y):
            # 准备 x 的数值数组
            x_array = self._prep_values(x)
            # 准备 y 的数值数组
            y_array = self._prep_values(y)
            # 获取窗口索引器
            window_indexer = self._get_window_indexer()
            # 计算最小有效期数，如果未指定则使用窗口索引器的窗口大小
            min_periods = (
                self.min_periods
                if self.min_periods is not None
                else window_indexer.window_size
            )
            # 获取窗口的起始和结束索引
            start, end = window_indexer.get_window_bounds(
                num_values=len(x_array),
                min_periods=min_periods,
                center=self.center,
                closed=self.closed,
                step=self.step,
            )
            # 检查窗口边界是否有效
            self._check_window_bounds(start, end, len(x_array))

            # 在忽略错误的状态下计算平均值乘积
            with np.errstate(all="ignore"):
                mean_x_y = window_aggregations.roll_mean(
                    x_array * y_array, start, end, min_periods
                )
                # 计算 x 的平均值
                mean_x = window_aggregations.roll_mean(x_array, start, end, min_periods)
                # 计算 y 的平均值
                mean_y = window_aggregations.roll_mean(y_array, start, end, min_periods)
                # 计算非缺失值元素的数量
                count_x_y = window_aggregations.roll_sum(
                    notna(x_array + y_array).astype(np.float64), start, end, 0
                )
                # 计算 x 的方差
                x_var = window_aggregations.roll_var(
                    x_array, start, end, min_periods, ddof
                )
                # 计算 y 的方差
                y_var = window_aggregations.roll_var(
                    y_array, start, end, min_periods, ddof
                )
                # 计算相关性的分子部分
                numerator = (mean_x_y - mean_x * mean_y) * (
                    count_x_y / (count_x_y - ddof)
                )
                # 计算相关性的分母部分
                denominator = (x_var * y_var) ** 0.5
                # 计算最终的相关系数
                result = numerator / denominator
            # 返回 Series 对象，包含计算结果，保留原始索引和名称
            return Series(result, index=x.index, name=x.name, copy=False)

        # 调用 _apply_pairwise 方法，应用 corr_func 函数对 self._selected_obj 和 other 执行逐对操作
        return self._apply_pairwise(
            self._selected_obj, other, pairwise, corr_func, numeric_only
        )
class Rolling(RollingAndExpandingMixin):
    # 定义 Rolling 类，继承 RollingAndExpandingMixin 类

    _attributes: list[str] = [
        "window",
        "min_periods",
        "center",
        "win_type",
        "on",
        "closed",
        "step",
        "method",
    ]
    # 定义 Rolling 类的属性列表，包括窗口大小、最小周期数、居中标志、窗口类型、索引、闭合方式、步长和方法

    def _validate(self) -> None:
        # 定义私有方法 _validate，无返回值

        super()._validate()
        # 调用父类 RollingAndExpandingMixin 的 _validate 方法

        # we allow rolling on a datetimelike index
        # 允许在类似日期时间的索引上执行滚动操作
        if (
            self.obj.empty
            or isinstance(self._on, (DatetimeIndex, TimedeltaIndex, PeriodIndex))
            or (isinstance(self._on.dtype, ArrowDtype) and self._on.dtype.kind in "mM")
        ) and isinstance(self.window, (str, BaseOffset, timedelta)):
            # 如果对象为空或者索引是日期时间索引、时间差索引或者周期索引，
            # 或者索引是 ArrowDtype 且类型是 "mM" 中的一种
            # 且窗口是字符串、基本偏移量或者时间差类型

            self._validate_datetimelike_monotonic()
            # 调用 _validate_datetimelike_monotonic 方法验证索引的单调性

            # this will raise ValueError on non-fixed freqs
            # 在非固定频率下会引发 ValueError
            try:
                freq = to_offset(self.window)
                # 尝试将窗口大小转换为偏移量
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"passed window {self.window} is not "
                    "compatible with a datetimelike index"
                ) from err
                # 如果失败，抛出与日期时间索引不兼容的 ValueError 异常

            if isinstance(self._on, PeriodIndex):
                # 如果索引是周期索引
                self._win_freq_i8 = freq.nanos / (  # type: ignore[assignment]
                    self._on.freq.nanos / self._on.freq.n
                )
                # 计算窗口频率
            else:
                try:
                    unit = dtype_to_unit(self._on.dtype)  # type: ignore[arg-type]
                    # 尝试获取索引数据类型对应的时间单位
                except TypeError:
                    # 如果不是日期时间数据类型，例如对于空数据框
                    unit = "ns"
                    # 默认时间单位为纳秒
                self._win_freq_i8 = Timedelta(freq.nanos).as_unit(unit)._value
                # 计算时间差的单位值，并赋给 _win_freq_i8

            # min_periods must be an integer
            # 最小周期数必须是整数
            if self.min_periods is None:
                self.min_periods = 1
                # 如果最小周期数为 None，则设为 1

            if self.step is not None:
                raise NotImplementedError(
                    "step is not supported with frequency windows"
                )
                # 如果步长不为 None，则抛出 NotImplementedError 异常，不支持频率窗口的步长设置

        elif isinstance(self.window, BaseIndexer):
            # 如果窗口是 BaseIndexer 的子类
            # Passed BaseIndexer subclass should handle all other rolling kwargs
            # 传递的 BaseIndexer 子类应该处理所有其他滚动关键字参数
            pass
            # 什么也不做

        elif not is_integer(self.window) or self.window < 0:
            # 如果窗口不是整数或者小于 0
            raise ValueError("window must be an integer 0 or greater")
            # 抛出 ValueError 异常，窗口必须是大于等于 0 的整数

    def _validate_datetimelike_monotonic(self) -> None:
        # 定义私有方法 _validate_datetimelike_monotonic，无返回值

        """
        Validate self._on is monotonic (increasing or decreasing) and has
        no NaT values for frequency windows.
        """
        # 验证 self._on 是单调的（递增或递减）且在频率窗口中没有 NaT 值

        if self._on.hasnans:
            self._raise_monotonic_error("values must not have NaT")
            # 如果索引有 NaT 值，则调用 _raise_monotonic_error 方法抛出异常

        if not (self._on.is_monotonic_increasing or self._on.is_monotonic_decreasing):
            self._raise_monotonic_error("values must be monotonic")
            # 如果索引不是单调递增或递减，则调用 _raise_monotonic_error 方法抛出异常

    def _raise_monotonic_error(self, msg: str):
        # 定义私有方法 _raise_monotonic_error，接收一个字符串参数 msg，无返回值

        on = self.on
        # 将 self.on 赋给变量 on
        if on is None:
            on = "index"
            # 如果 on 是 None，则设为 "index"
        raise ValueError(f"{on} {msg}")
        # 抛出 ValueError 异常，包含错误消息和索引信息
    @doc(
        _shared_docs["aggregate"],  # 使用 _shared_docs 字典中的 "aggregate" 键对应的文档作为注释模板
        see_also=dedent(  # 参见部分，包含了 Series.rolling 和 DataFrame.rolling 的用法
            """
        See Also
        --------
        Series.rolling : Calling object with Series data.
        DataFrame.rolling : Calling object with DataFrame data.
        """
        ),
        examples=dedent(  # 示例部分，展示了如何在 DataFrame 或 Series 上应用 rolling 方法的用例
            """
        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> df.rolling(2).sum()
             A     B     C
        0  NaN   NaN   NaN
        1  3.0   9.0  15.0
        2  5.0  11.0  17.0

        >>> df.rolling(2).agg({"A": "sum", "B": "min"})
             A    B
        0  NaN  NaN
        1  3.0  4.0
        2  5.0  5.0
        """
        ),
        klass="Series/Dataframe",  # 类型标记，指示适用于 Series 或 DataFrame
        axis="",  # 轴向参数，此处为空字符串
    )
    def aggregate(self, func, *args, **kwargs):
        return super().aggregate(func, *args, **kwargs)  # 调用父类的 aggregate 方法并返回结果

    agg = aggregate  # 将 aggregate 方法赋值给变量 agg

    @doc(
        template_header,  # 使用预定义的注释模板中的头部
        create_section_header("Parameters"),  # 参数部分的标题
        kwargs_numeric_only,  # 仅接受数值参数的标记
        create_section_header("Returns"),  # 返回值部分的标题
        template_returns,  # 返回值说明的模板
        create_section_header("See Also"),  # 参见部分的标题
        template_see_also,  # 参见部分的模板
        create_section_header("Examples"),  # 示例部分的标题
        dedent(  # 缩进处理，展示了如何在 Series 上应用 rolling 方法的示例用法
            """
        >>> s = pd.Series([2, 3, np.nan, 10])
        >>> s.rolling(2).count()
        0    NaN
        1    2.0
        2    1.0
        3    1.0
        dtype: float64
        >>> s.rolling(3).count()
        0    NaN
        1    NaN
        2    2.0
        3    2.0
        dtype: float64
        >>> s.rolling(4).count()
        0    NaN
        1    NaN
        2    NaN
        3    3.0
        dtype: float64
        """
        ).replace("\n", "", 1),  # 示例代码，展示了如何在 Series 上应用 rolling 方法进行计数
        window_method="rolling",  # 窗口方法为 rolling
        aggregation_description="count of non NaN observations",  # 聚合描述为非 NaN 观察值的计数
        agg_method="count",  # 使用 count 方法进行聚合
    )
    def count(self, numeric_only: bool = False):
        return super().count(numeric_only)  # 调用父类的 count 方法并返回结果

    @doc(
        template_header,  # 使用预定义的注释模板中的头部
        create_section_header("Parameters"),  # 参数部分的标题
        window_apply_parameters,  # 应用窗口的参数列表
        create_section_header("Returns"),  # 返回值部分的标题
        template_returns,  # 返回值说明的模板
        create_section_header("See Also"),  # 参见部分的标题
        template_see_also,  # 参见部分的模板
        create_section_header("Examples"),  # 示例部分的标题
        dedent(  # 缩进处理，展示了如何在 Series 上应用 rolling 方法的自定义聚合函数的示例用法
            """\
        >>> ser = pd.Series([1, 6, 5, 4])
        >>> ser.rolling(2).apply(lambda s: s.sum() - s.min())
        0    NaN
        1    6.0
        2    6.0
        3    5.0
        dtype: float64
        """
        ),
        window_method="rolling",  # 窗口方法为 rolling
        aggregation_description="custom aggregation function",  # 自定义聚合函数的描述
        agg_method="apply",  # 使用 apply 方法进行聚合
    )
    def apply(
        self,
        func: Callable[..., Any],
        raw: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        return super().apply(func, raw=raw, engine=engine, engine_kwargs=engine_kwargs, args=args, kwargs=kwargs)
    ):
        return super().apply(
            func,
            raw=raw,
            engine=engine,
            engine_kwargs=engine_kwargs,
            args=args,
            kwargs=kwargs,
        )



    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        window_agg_numba_parameters(),
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        numba_notes,
        create_section_header("Examples"),
        dedent(
            """
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> s
        0    1
        1    2
        2    3
        3    4
        4    5
        dtype: int64

        >>> s.rolling(3).sum()
        0     NaN
        1     NaN
        2     6.0
        3     9.0
        4    12.0
        dtype: float64

        >>> s.rolling(3, center=True).sum()
        0     NaN
        1     6.0
        2     9.0
        3    12.0
        4     NaN
        dtype: float64

        For DataFrame, each sum is computed column-wise.

        >>> df = pd.DataFrame({"A": s, "B": s ** 2})
        >>> df
           A   B
        0  1   1
        1  2   4
        2  3   9
        3  4  16
        4  5  25

        >>> df.rolling(3).sum()
              A     B
        0   NaN   NaN
        1   NaN   NaN
        2   6.0  14.0
        3   9.0  29.0
        4  12.0  50.0
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="sum",
        agg_method="sum",
    )



    def sum(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        return super().sum(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )



    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        window_agg_numba_parameters(),
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        numba_notes,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.rolling(2).max()
        0    NaN
        1    2.0
        2    3.0
        3    4.0
        dtype: float64
        """
        ),
        window_method="rolling",
        aggregation_description="maximum",
        agg_method="max",
    )



    def max(
        self,
        numeric_only: bool = False,
        *args,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
        **kwargs,
    ):
        # 调用父类的 max 方法，传递相应的参数
        return super().max(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )
    ):
        return super().max(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )



    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        window_agg_numba_parameters(),
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        numba_notes,
        create_section_header("Examples"),
        dedent(
            """
        Performing a rolling minimum with a window size of 3.

        >>> s = pd.Series([4, 3, 5, 2, 6])
        >>> s.rolling(3).min()
        0    NaN
        1    NaN
        2    3.0
        3    2.0
        4    2.0
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="minimum",
        agg_method="min",
    )



    def min(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        return super().min(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )



    @doc(
        template_header,
        create_section_header("Parameters"),
        kwargs_numeric_only,
        window_agg_numba_parameters(),
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Notes"),
        numba_notes,
        create_section_header("Examples"),
        dedent(
            """
        The below examples will show rolling mean calculations with window sizes of
        two and three, respectively.

        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.rolling(2).mean()
        0    NaN
        1    1.5
        2    2.5
        3    3.5
        dtype: float64

        >>> s.rolling(3).mean()
        0    NaN
        1    NaN
        2    2.0
        3    3.0
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="mean",
        agg_method="mean",
    )



    def mean(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        return super().mean(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )
    # 定义带有文档字符串的装饰器，用于描述函数的参数、返回值、示例等信息
    @doc(
        template_header,  # 插入预定义的模板标题
        create_section_header("Parameters"),  # 创建参数部分的标题
        dedent(
            """
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        """
        ).replace("\n", "", 1),  # 解释 ddof 参数的默认值和用途
        kwargs_numeric_only,  # 描述函数接受的数值型参数
        window_agg_numba_parameters("1.4"),  # 描述与窗口聚合和 Numba 相关的参数
        create_section_header("Returns"),  # 创建返回值部分的标题
        template_returns,  # 插入预定义的返回值模板
        create_section_header("See Also"),  # 创建相关链接部分的标题
        "numpy.std : Equivalent method for NumPy array.\n",  # 提示与 NumPy 中 std 方法的等效性
        template_see_also,  # 插入预定义的相关链接模板
        create_section_header("Notes"),  # 创建注释部分的标题
        dedent(
            """
        The default ``ddof`` of 1 used in :meth:`Series.std` is different
        than the default ``ddof`` of 0 in :func:`numpy.std`.

        A minimum of one period is required for the rolling calculation.\n
        """
        ).replace("\n", "", 1),  # 解释关于默认 ddof 和滚动计算的必要性的注释
        create_section_header("Examples"),  # 创建示例部分的标题
        dedent(
            """
        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])
        >>> s.rolling(3).std()
        0         NaN
        1         NaN
        2    0.577350
        3    1.000000
        4    1.000000
        5    1.154701
        6    0.000000
        dtype: float64
        """
        ).replace("\n", "", 1),  # 展示函数的使用示例
        window_method="rolling",  # 描述使用滚动窗口方法
        aggregation_description="standard deviation",  # 描述聚合方法是标准差
        agg_method="std",  # 指定使用标准差聚合方法
    )
    # 定义 std 方法，用于计算序列的滚动标准差
    def std(
        self,
        ddof: int = 1,  # 参数：自由度修正值，默认为 1
        numeric_only: bool = False,  # 参数：是否仅考虑数值型数据，默认为 False
        engine: Literal["cython", "numba"] | None = None,  # 参数：指定计算引擎，默认为 None
        engine_kwargs: dict[str, bool] | None = None,  # 参数：引擎的额外关键字参数，默认为 None
    ):
        # 调用父类的 std 方法，计算滚动标准差并返回结果
        return super().std(
            ddof=ddof,
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )
    @doc(
        # 使用预定义的模板头部
        template_header,
        # 创建“Parameters”部分的标题
        create_section_header("Parameters"),
        # 定义 ddof 参数及其说明
        dedent(
            """
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        """
        ).replace("\n", "", 1),
        # 引用用于数值参数的关键字参数说明
        kwargs_numeric_only,
        # 使用 numba 引擎的窗口聚合参数
        window_agg_numba_parameters("1.4"),
        # 创建“Returns”部分的标题
        create_section_header("Returns"),
        # 使用预定义的返回值模板
        template_returns,
        # 创建“See Also”部分的标题
        create_section_header("See Also"),
        # 引用 numpy.var 的等效方法
        "numpy.var : Equivalent method for NumPy array.\n",
        # 使用预定义的“See Also”模板
        template_see_also,
        # 创建“Notes”部分的标题
        create_section_header("Notes"),
        # 定义有关 ddof 默认值差异的说明
        dedent(
            """
        The default ``ddof`` of 1 used in :meth:`Series.var` is different
        than the default ``ddof`` of 0 in :func:`numpy.var`.

        A minimum of one period is required for the rolling calculation.\n
        """
        ).replace("\n", "", 1),
        # 创建“Examples”部分的标题
        create_section_header("Examples"),
        # 给出计算示例
        dedent(
            """
        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])
        >>> s.rolling(3).var()
        0         NaN
        1         NaN
        2    0.333333
        3    1.000000
        4    1.000000
        5    1.333333
        6    0.000000
        dtype: float64
        """
        ).replace("\n", "", 1),
        # 定义使用滚动窗口方法的聚合描述
        window_method="rolling",
        # 描述聚合方法为方差
        aggregation_description="variance",
        # 聚合方法为 var
        agg_method="var",
    )
    # 定义 var 方法，接收 ddof、numeric_only、engine 和 engine_kwargs 参数
    def var(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # 调用父类的 var 方法，传递参数并返回结果
        return super().var(
            ddof=ddof,
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    @doc(
        # 使用预定义的模板头部
        template_header,
        # 创建“Parameters”部分的标题
        create_section_header("Parameters"),
        # 使用数值参数的关键字参数说明
        kwargs_numeric_only,
        # 创建“Returns”部分的标题
        create_section_header("Returns"),
        # 使用预定义的返回值模板
        template_returns,
        # 创建“See Also”部分的标题
        create_section_header("See Also"),
        # 引用 scipy.stats.skew 的相关信息
        "scipy.stats.skew : Third moment of a probability density.\n",
        # 使用预定义的“See Also”模板
        template_see_also,
        # 创建“Notes”部分的标题
        create_section_header("Notes"),
        # 定义关于滚动计算需要最少三个周期的说明
        dedent(
            """
        A minimum of three periods is required for the rolling calculation.\n
        """
        ),
        # 创建“Examples”部分的标题
        create_section_header("Examples"),
        # 给出计算示例
        dedent(
            """\
        >>> ser = pd.Series([1, 5, 2, 7, 15, 6])
        >>> ser.rolling(3).skew().round(6)
        0         NaN
        1         NaN
        2    1.293343
        3   -0.585583
        4    0.670284
        5    1.652317
        dtype: float64
        """
        ),
        # 定义使用滚动窗口方法的聚合描述
        window_method="rolling",
        # 描述聚合方法为无偏偏斜度
        aggregation_description="unbiased skewness",
        # 聚合方法为 skew
        agg_method="skew",
    )
    # 定义 skew 方法，接收 numeric_only 参数
    def skew(self, numeric_only: bool = False):
        # 调用父类的 skew 方法，传递参数并返回结果
        return super().skew(numeric_only=numeric_only)
    @doc(
        template_header,  # 使用预定义的文档模板头部
        create_section_header("Parameters"),  # 创建参数部分的标题
        dedent(  # 移除字符串开头的缩进
            """
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        """
        ).replace("\n", "", 1),  # 替换第一个换行符为空格
        kwargs_numeric_only,  # 使用预定义的数字参数关键字
        create_section_header("Returns"),  # 创建返回部分的标题
        template_returns,  # 使用预定义的返回值模板
        create_section_header("See Also"),  # 创建相关内容部分的标题
        template_see_also,  # 使用预定义的相关内容模板
        create_section_header("Notes"),  # 创建注释部分的标题
        "A minimum of one period is required for the calculation.\n\n",  # 添加特定的注释信息
        create_section_header("Examples"),  # 创建示例部分的标题
        dedent(  # 移除字符串开头的缩进
            """
        >>> s = pd.Series([0, 1, 2, 3])
        >>> s.rolling(2, min_periods=1).sem()
        0         NaN
        1    0.707107
        2    0.707107
        3    0.707107
        dtype: float64
        """
        ).replace("\n", "", 1),  # 替换第一个换行符为空格
        window_method="rolling",  # 使用滚动窗口方法
        aggregation_description="standard error of mean",  # 聚合描述为均值的标准误差
        agg_method="sem",  # 使用标准误差方法
    )
    def sem(self, ddof: int = 1, numeric_only: bool = False):
        # Raise here so error message says sem instead of std
        self._validate_numeric_only("sem", numeric_only)  # 调用内部方法验证是否仅限数字
        return self.std(numeric_only=numeric_only) / (
            self.count(numeric_only) - ddof  # 返回标准差除以计数和自由度的平方根
        ).pow(0.5)

    @doc(
        template_header,  # 使用预定义的文档模板头部
        create_section_header("Parameters"),  # 创建参数部分的标题
        kwargs_numeric_only,  # 使用预定义的数字参数关键字
        create_section_header("Returns"),  # 创建返回部分的标题
        template_returns,  # 使用预定义的返回值模板
        create_section_header("See Also"),  # 创建相关内容部分的标题
        "scipy.stats.kurtosis : Reference SciPy method.\n",  # 添加特定的相关内容信息
        template_see_also,  # 使用预定义的相关内容模板
        create_section_header("Notes"),  # 创建注释部分的标题
        "A minimum of four periods is required for the calculation.\n\n",  # 添加特定的注释信息
        create_section_header("Examples"),  # 创建示例部分的标题
        dedent(  # 移除字符串开头的缩进
            """
        The example below will show a rolling calculation with a window size of
        four matching the equivalent function call using `scipy.stats`.

        >>> arr = [1, 2, 3, 4, 999]
        >>> import scipy.stats
        >>> print(f"{{scipy.stats.kurtosis(arr[:-1], bias=False):.6f}}")
        -1.200000
        >>> print(f"{{scipy.stats.kurtosis(arr[1:], bias=False):.6f}}")
        3.999946
        >>> s = pd.Series(arr)
        >>> s.rolling(4).kurt()
        0         NaN
        1         NaN
        2         NaN
        3   -1.200000
        4    3.999946
        dtype: float64
        """
        ).replace("\n", "", 1),  # 替换第一个换行符为空格
        window_method="rolling",  # 使用滚动窗口方法
        aggregation_description="Fisher's definition of kurtosis without bias",  # 聚合描述为无偏的峰度定义
        agg_method="kurt",  # 使用峰度方法
    )
    def kurt(self, numeric_only: bool = False):
        return super().kurt(numeric_only=numeric_only)  # 调用父类的峰度计算方法
    # 定义 quantile 方法，并使用装饰器 @doc 来提供文档说明
    @doc(
        template_header,  # 使用指定的模板头部
        create_section_header("Parameters"),  # 创建参数部分的标题
        dedent(
            """
        q : float
            Quantile to compute. 0 <= quantile <= 1.

            .. deprecated:: 2.1.0
                This was renamed from 'quantile' to 'q' in version 2.1.0.
        interpolation : {{'linear', 'lower', 'higher', 'midpoint', 'nearest'}}
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points `i` and `j`:

                * linear: `i + (j - i) * fraction`, where `fraction` is the
                  fractional part of the index surrounded by `i` and `j`.
                * lower: `i`.
                * higher: `j`.
                * nearest: `i` or `j` whichever is nearest.
                * midpoint: (`i` + `j`) / 2.
        """
        ).replace("\n", "", 1),  # 对参数部分进行格式化和处理
        kwargs_numeric_only,  # 传入关键字参数仅限于数值类型
        create_section_header("Returns"),  # 创建返回值部分的标题
        template_returns,  # 使用标准的返回值模板
        create_section_header("See Also"),  # 创建相关链接部分的标题
        template_see_also,  # 使用标准的相关链接模板
        create_section_header("Examples"),  # 创建示例部分的标题
        dedent(
            """
        >>> s = pd.Series([1, 2, 3, 4])
        >>> s.rolling(2).quantile(.4, interpolation='lower')
        0    NaN
        1    1.0
        2    2.0
        3    3.0
        dtype: float64

        >>> s.rolling(2).quantile(.4, interpolation='midpoint')
        0    NaN
        1    1.5
        2    2.5
        3    3.5
        dtype: float64
        """
        ).replace("\n", "", 1),  # 对示例部分进行格式化和处理
        window_method="rolling",  # 指定窗口方法为滚动窗口
        aggregation_description="quantile",  # 指定聚合描述为分位数
        agg_method="quantile",  # 指定聚合方法为分位数
    )
    # 定义 quantile 方法的函数体
    def quantile(
        self,
        q: float,  # 定义 q 参数为浮点型，表示要计算的分位数
        interpolation: QuantileInterpolation = "linear",  # 定义 interpolation 参数，默认为线性插值
        numeric_only: bool = False,  # 定义 numeric_only 参数，默认为 False，表示计算时包含非数值类型
    ):
        return super().quantile(  # 调用父类的 quantile 方法进行计算
            q=q,
            interpolation=interpolation,
            numeric_only=numeric_only,
        )
    # 使用装饰器 @doc 对方法进行文档化，包括版本新增信息、参数部分标题、参数详细说明、仅数字参数、返回值部分标题、返回值模板、相关参考部分标题、示例部分标题和示例代码
    @doc(
        template_header,
        ".. versionadded:: 1.4.0 \n\n",
        create_section_header("Parameters"),
        dedent(
            """
        method : {{'average', 'min', 'max'}}, default 'average'
            How to rank the group of records that have the same value (i.e. ties):

            * average: average rank of the group
            * min: lowest rank in the group
            * max: highest rank in the group

        ascending : bool, default True
            Whether or not the elements should be ranked in ascending order.
        pct : bool, default False
            Whether or not to display the returned rankings in percentile
            form.
        """
        ).replace("\n", "", 1),
        kwargs_numeric_only,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """
        >>> s = pd.Series([1, 4, 2, 3, 5, 3])
        >>> s.rolling(3).rank()
        0    NaN
        1    NaN
        2    2.0
        3    2.0
        4    3.0
        5    1.5
        dtype: float64

        >>> s.rolling(3).rank(method="max")
        0    NaN
        1    NaN
        2    2.0
        3    2.0
        4    3.0
        5    2.0
        dtype: float64

        >>> s.rolling(3).rank(method="min")
        0    NaN
        1    NaN
        2    2.0
        3    2.0
        4    3.0
        5    1.0
        dtype: float64
        """
        ).replace("\n", "", 1),
        window_method="rolling",
        aggregation_description="rank",
        agg_method="rank",
    )
    # 定义 rank 方法，接受参数 method、ascending、pct、numeric_only，调用父类的 rank 方法进行计算并返回结果
    def rank(
        self,
        method: WindowingRankType = "average",  # 排名方法，默认为 'average'
        ascending: bool = True,  # 是否升序排名，默认为 True
        pct: bool = False,  # 是否以百分比形式显示排名，默认为 False
        numeric_only: bool = False,  # 是否仅对数字进行排名，默认为 False
    ):
        return super().rank(
            method=method,
            ascending=ascending,
            pct=pct,
            numeric_only=numeric_only,
        )
    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """
            other : Series or DataFrame, optional
                如果未提供，则默认为 self，并生成成对的输出。
            pairwise : bool, default None
                如果为 False，则仅使用 self 和 other 之间匹配的列，并输出一个 DataFrame。
                如果为 True，则计算所有成对组合，并在 DataFrame 输入的情况下输出一个 MultiIndexed DataFrame。
                对于缺失元素，只使用完整的成对观测。
            ddof : int, default 1
                Delta 自由度。计算中使用的除数是 ``N - ddof``，其中 ``N`` 表示元素数量。
            """
        ).replace("\n", "", 1),
        kwargs_numeric_only,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
            >>> ser1 = pd.Series([1, 2, 3, 4])
            >>> ser2 = pd.Series([1, 4, 5, 8])
            >>> ser1.rolling(2).cov(ser2)
            0    NaN
            1    1.5
            2    0.5
            3    1.5
            dtype: float64
            """
        ),
        window_method="rolling",
        aggregation_description="sample covariance",
        agg_method="cov",
    )
    # 定义 cov 方法，用于计算协方差
    def cov(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        # 调用父类的 cov 方法，传递参数并返回结果
        return super().cov(
            other=other,
            pairwise=pairwise,
            ddof=ddof,
            numeric_only=numeric_only,
        )

    # 定义 corr 方法，用于计算相关系数
    def corr(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        # 调用父类的 corr 方法，传递参数并返回结果
        return super().corr(
            other=other,
            pairwise=pairwise,
            ddof=ddof,
            numeric_only=numeric_only,
        )
# 设置 Rolling 类的文档字符串为 Window 类的文档字符串
Rolling.__doc__ = Window.__doc__

# 定义 RollingGroupby 类，继承自 BaseWindowGroupby 和 Rolling
class RollingGroupby(BaseWindowGroupby, Rolling):
    """
    提供一个滚动分组的实现。
    """

    # 合并 Rolling 和 BaseWindowGroupby 类的属性列表
    _attributes = Rolling._attributes + BaseWindowGroupby._attributes

    # 返回一个索引器类，用于计算窗口的起始和结束边界
    def _get_window_indexer(self) -> GroupbyIndexer:
        """
        返回一个索引器类，该类将计算窗口的起始和结束边界

        Returns
        -------
        GroupbyIndexer
        """
        rolling_indexer: type[BaseIndexer]
        indexer_kwargs: dict[str, Any] | None = None
        index_array = self._index_array
        
        # 如果窗口类型是 BaseIndexer 的实例
        if isinstance(self.window, BaseIndexer):
            rolling_indexer = type(self.window)
            indexer_kwargs = self.window.__dict__.copy()
            assert isinstance(indexer_kwargs, dict)  # for mypy
            # 后续将使用每个组的索引
            indexer_kwargs.pop("index_array", None)
            window = self.window
        # 如果 _win_freq_i8 不为 None
        elif self._win_freq_i8 is not None:
            rolling_indexer = VariableWindowIndexer
            # 错误：赋值时类型不兼容（表达式类型为 "int"，变量类型为 "BaseIndexer"）
            window = self._win_freq_i8  # type: ignore[assignment]
        else:
            rolling_indexer = FixedWindowIndexer
            window = self.window
        
        # 创建 GroupbyIndexer 对象
        window_indexer = GroupbyIndexer(
            index_array=index_array,
            window_size=window,
            groupby_indices=self._grouper.indices,
            window_indexer=rolling_indexer,
            indexer_kwargs=indexer_kwargs,
        )
        return window_indexer

    # 验证 self._on 中的每个分组是否单调
    def _validate_datetimelike_monotonic(self) -> None:
        """
        验证 self._on 中的每个分组是否单调
        """
        # GH 46061
        if self._on.hasnans:
            self._raise_monotonic_error("values must not have NaT")
        
        # 遍历 self._grouper.indices 中的每个分组
        for group_indices in self._grouper.indices.values():
            group_on = self._on.take(group_indices)
            # 如果分组不是单调递增且不是单调递减
            if not (
                group_on.is_monotonic_increasing or group_on.is_monotonic_decreasing
            ):
                on = "index" if self.on is None else self.on
                raise ValueError(
                    f"Each group within {on} must be monotonic. "
                    f"Sort the values in {on} first."
                )
```