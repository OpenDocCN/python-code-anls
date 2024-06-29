# `D:\src\scipysrc\pandas\pandas\core\groupby\generic.py`

```
"""
Define the SeriesGroupBy and DataFrameGroupBy
classes that hold the groupby interfaces (and some implementations).

These are user facing as the result of the ``df.groupby(...)`` operations,
which here returns a DataFrameGroupBy object.
"""

from __future__ import annotations

from collections import abc  # 导入 collections 模块中的 abc 子模块
from collections.abc import Callable  # 导入 collections.abc 模块中的 Callable 类
from functools import partial  # 导入 functools 模块中的 partial 函数
from textwrap import dedent  # 导入 textwrap 模块中的 dedent 函数
from typing import (
    TYPE_CHECKING,  # 导入 typing 模块中的 TYPE_CHECKING 常量
    Any,
    Literal,
    NamedTuple,
    TypeVar,
    Union,
    cast,
)
import warnings  # 导入 warnings 模块

import numpy as np  # 导入 numpy 库并使用 np 别名

from pandas._libs import Interval  # 从 pandas._libs 模块导入 Interval 类
from pandas._libs.hashtable import duplicated  # 从 pandas._libs.hashtable 模块导入 duplicated 函数
from pandas.errors import SpecificationError  # 从 pandas.errors 模块导入 SpecificationError 异常
from pandas.util._decorators import (  # 从 pandas.util._decorators 模块导入多个装饰器函数
    Appender,
    Substitution,
    doc,
)
from pandas.util._exceptions import find_stack_level  # 从 pandas.util._exceptions 模块导入 find_stack_level 函数

from pandas.core.dtypes.common import (  # 从 pandas.core.dtypes.common 模块导入多个函数
    ensure_int64,
    is_bool,
    is_dict_like,
    is_integer_dtype,
    is_list_like,
    is_numeric_dtype,
    is_scalar,
)
from pandas.core.dtypes.dtypes import (  # 从 pandas.core.dtypes.dtypes 模块导入多个数据类型类
    CategoricalDtype,
    IntervalDtype,
)
from pandas.core.dtypes.inference import is_hashable  # 从 pandas.core.dtypes.inference 模块导入 is_hashable 函数
from pandas.core.dtypes.missing import (  # 从 pandas.core.dtypes.missing 模块导入 isna 和 notna 函数
    isna,
    notna,
)

from pandas.core import algorithms  # 从 pandas.core 模块导入 algorithms 模块
from pandas.core.apply import (  # 从 pandas.core.apply 模块导入多个函数和类
    GroupByApply,
    maybe_mangle_lambdas,
    reconstruct_func,
    validate_func_kwargs,
)
import pandas.core.common as com  # 导入 pandas.core.common 模块并使用 com 别名
from pandas.core.frame import DataFrame  # 从 pandas.core.frame 模块导入 DataFrame 类
from pandas.core.groupby import base  # 从 pandas.core.groupby 模块导入 base 模块
from pandas.core.groupby.groupby import (  # 从 pandas.core.groupby.groupby 模块导入多个函数和类
    GroupBy,
    GroupByPlot,
    _agg_template_frame,
    _agg_template_series,
    _transform_template,
)
from pandas.core.indexes.api import (  # 从 pandas.core.indexes.api 模块导入多个函数和类
    Index,
    MultiIndex,
    all_indexes_same,
    default_index,
)
from pandas.core.series import Series  # 从 pandas.core.series 模块导入 Series 类
from pandas.core.sorting import get_group_index  # 从 pandas.core.sorting 模块导入 get_group_index 函数
from pandas.core.util.numba_ import maybe_use_numba  # 从 pandas.core.util.numba_ 模块导入 maybe_use_numba 函数

from pandas.plotting import boxplot_frame_groupby  # 从 pandas.plotting 模块导入 boxplot_frame_groupby 函数

if TYPE_CHECKING:  # 如果在类型检查模式下
    from collections.abc import (  # 从 collections.abc 模块导入多个类型
        Hashable,
        Sequence,
    )

    from pandas._typing import (  # 从 pandas._typing 模块导入多个类型
        ArrayLike,
        BlockManager,
        CorrelationMethod,
        IndexLabel,
        Manager,
        SingleBlockManager,
        TakeIndexer,
    )

    from pandas import Categorical  # 从 pandas 模块导入 Categorical 类
    from pandas.core.generic import NDFrame  # 从 pandas.core.generic 模块导入 NDFrame 类

# TODO(typing) the return value on this callable should be any *scalar*.
AggScalar = Union[str, Callable[..., Any]]  # 定义 AggScalar 类型别名，可以是字符串或可调用对象
# TODO: validate types on ScalarResult and move to _typing
# Blocked from using by https://github.com/python/mypy/issues/1484
# See note at _mangle_lambda_list
ScalarResult = TypeVar("ScalarResult")  # 定义 ScalarResult 类型变量


class NamedAgg(NamedTuple):
    """
    Helper for column specific aggregation with control over output column names.

    Subclass of typing.NamedTuple.

    Parameters
    ----------
    column : Hashable
        Column label in the DataFrame to apply aggfunc.
    aggfunc : function or str
        Function to apply to the provided column. If string, the name of a built-in
        pandas function.
    """
    See Also
    --------
    DataFrame.groupby : 使用映射器或列系列对 DataFrame 进行分组。

    Examples
    --------
    >>> df = pd.DataFrame({"key": [1, 1, 2], "a": [-1, 0, 1], 1: [10, 11, 12]})
    >>> agg_a = pd.NamedAgg(column="a", aggfunc="min")
    >>> agg_1 = pd.NamedAgg(column=1, aggfunc=lambda x: np.mean(x))
    >>> df.groupby("key").agg(result_a=agg_a, result_1=agg_1)
         result_a  result_1
    key
    1          -1      10.5
    2           1      12.0
    ```

    # 定义命名聚合的参数列名，必须是可哈希对象
    column: Hashable
    # 定义命名聚合的聚合函数，可以是标量聚合函数或自定义聚合函数
    aggfunc: AggScalar
# SeriesGroupBy 类，继承自 GroupBy[Series]
class SeriesGroupBy(GroupBy[Series]):

    # 方法 _wrap_agged_manager，接收一个 Manager 对象 mgr，返回一个 Series 对象 out
    def _wrap_agged_manager(self, mgr: Manager) -> Series:
        # 从 mgr 创建一个新的 Series 对象 out，保持 mgr 的轴信息
        out = self.obj._constructor_from_mgr(mgr, axes=mgr.axes)
        # 设置新 Series 对象的名称为原始 Series 对象的名称
        out._name = self.obj.name
        return out

    # 方法 _get_data_to_aggregate，准备数据进行聚合
    def _get_data_to_aggregate(
        self, *, numeric_only: bool = False, name: str | None = None
    ) -> SingleBlockManager:
        # 获取经过排除后的原始 Series 对象
        ser = self._obj_with_exclusions
        # 获取 Series 的数据管理器
        single = ser._mgr
        # 如果 numeric_only 为 True 但 Series 的 dtype 不是数值类型，则抛出 TypeError
        if numeric_only and not is_numeric_dtype(ser.dtype):
            # GH#41291 匹配 Series 的行为
            kwd_name = "numeric_only"
            raise TypeError(
                f"Cannot use {kwd_name}=True with "
                f"{type(self).__name__}.{name} and non-numeric dtypes."
            )
        # 返回单块数据管理器 single，用于后续聚合操作
        return single

    # _agg_examples_doc 是一个文档字符串，包含了关于 Series 聚合操作的示例
    _agg_examples_doc = dedent(
        """
        Examples
        --------
        >>> s = pd.Series([1, 2, 3, 4])

        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> s.groupby([1, 1, 2, 2]).min()
        1    1
        2    3
        dtype: int64

        >>> s.groupby([1, 1, 2, 2]).agg('min')
        1    1
        2    3
        dtype: int64

        >>> s.groupby([1, 1, 2, 2]).agg(['min', 'max'])
           min  max
        1    1    2
        2    3    4

        The output column names can be controlled by passing
        the desired column names and aggregations as keyword arguments.

        >>> s.groupby([1, 1, 2, 2]).agg(
        ...     minimum='min',
        ...     maximum='max',
        ... )
           minimum  maximum
        1        1        2

        .. versionchanged:: 1.3.0

            The resulting dtype will reflect the return value of the aggregating function.

        >>> s.groupby([1, 1, 2, 2]).agg(lambda x: x.astype(float).min())
        1    1.0
        2    3.0
        dtype: float64
        """
    )

    # 使用 @doc 装饰器，将 _agg_template_series 模板和 _agg_examples_doc 文档注入到 klass="Series" 中
    @doc(_agg_template_series, examples=_agg_examples_doc, klass="Series")
    # 聚合函数，可以接受不同的函数作为参数来对数据进行聚合操作
    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):
        # 判断是否需要重新标记列名
        relabeling = func is None
        columns = None
        if relabeling:
            # 如果需要重新标记列名，则验证和处理函数参数
            columns, func = validate_func_kwargs(kwargs)
            kwargs = {}

        # 如果 func 是字符串
        if isinstance(func, str):
            # 如果可以使用 numba 引擎，并且引擎不为 None
            if maybe_use_numba(engine) and engine is not None:
                # 不是所有聚合函数都支持 numba，只有在用户请求 numba 并且引擎不为 None 时才传递 numba 的 kwargs
                kwargs["engine"] = engine
            # 如果有 engine_kwargs 参数，则传递给 kwargs
            if engine_kwargs is not None:
                kwargs["engine_kwargs"] = engine_kwargs
            # 调用对象的 func 方法，传入 *args 和 **kwargs 参数
            return getattr(self, func)(*args, **kwargs)

        # 如果 func 是可迭代对象（如列表或元组）
        elif isinstance(func, abc.Iterable):
            # 处理可能包含 lambda 表达式的 func
            func = maybe_mangle_lambdas(func)
            # 将 engine 和 engine_kwargs 参数传递给 kwargs
            kwargs["engine"] = engine
            kwargs["engine_kwargs"] = engine_kwargs
            # 使用 _aggregate_multiple_funcs 方法进行多函数聚合操作，返回结果
            ret = self._aggregate_multiple_funcs(func, *args, **kwargs)
            # 如果需要重新标记列名
            if relabeling:
                # 确保 columns 不为 None，用于类型检查（mypy）
                assert columns is not None  # for mypy
                # 将结果的列名修改为 columns
                ret.columns = columns
            # 如果不需要将索引作为结果的一部分
            if not self.as_index:
                # 对结果进行重置索引操作
                ret = ret.reset_index()
            return ret

        # 如果 func 不是字符串也不是可迭代对象
        else:
            # 如果可以使用 numba 引擎
            if maybe_use_numba(engine):
                # 使用 _aggregate_with_numba 方法进行 numba 加速的聚合操作
                return self._aggregate_with_numba(
                    func, *args, engine_kwargs=engine_kwargs, **kwargs
                )

            # 如果没有任何分组
            if self.ngroups == 0:
                # 例如，当没有任何分组时，没有数据可迭代，我们无法进行 dtype 推断，使用现有的 dtype
                obj = self._obj_with_exclusions
                # 创建一个新对象，名称和索引从原对象获取，dtype 使用原对象的 dtype
                return self.obj._constructor(
                    [],
                    name=self.obj.name,
                    index=self._grouper.result_index,
                    dtype=obj.dtype,
                )
            # 使用 _python_agg_general 方法进行一般的 Python 聚合操作
            return self._python_agg_general(func, *args, **kwargs)

    # 将 aggregate 方法的引用命名为 agg
    agg = aggregate

    # 用于一般的 Python 聚合操作，接受函数 func 及其它参数
    def _python_agg_general(self, func, *args, **kwargs):
        # 使用 lambda 表达式创建函数 f，调用 func 对数据进行聚合操作
        f = lambda x: func(x, *args, **kwargs)

        obj = self._obj_with_exclusions
        # 使用 _grouper 对象对数据对象 obj 进行聚合操作，得到结果
        result = self._grouper.agg_series(obj, f)
        # 根据聚合结果创建新对象 res，名称从 obj 获取
        res = obj._constructor(result, name=obj.name)
        # 对聚合结果进行包装处理，并返回
        return self._wrap_aggregated_output(res)
    # 聚合多个函数到一个 DataFrame 上
    def _aggregate_multiple_funcs(self, arg, *args, **kwargs) -> DataFrame:
        # 如果输入是字典，则抛出异常，不支持嵌套重命名
        if isinstance(arg, dict):
            raise SpecificationError("nested renamer is not supported")

        # 如果任何一个元素是 tuple 或者 list，则统一格式化为 (x, x)，否则保持原样
        if any(isinstance(x, (tuple, list)) for x in arg):
            arg = ((x, x) if not isinstance(x, (tuple, list)) else x for x in arg)
        else:
            # 如果是函数列表或函数名列表，则获取对应的可调用名称
            columns = (com.get_callable_name(f) or f for f in arg)
            # 将列名和函数组成元组
            arg = zip(columns, arg)

        # 存储聚合结果的字典，键为输出的标签和位置组成的 OutputKey，值为聚合后的 DataFrame 或 Series
        results: dict[base.OutputKey, DataFrame | Series] = {}

        # 设置 self 的 as_index 属性为 True，在上下文中临时修改
        with com.temp_setattr(self, "as_index", True):
            # 使用索引组合结果，如果 as_index=False，则需要在之后调整索引 (GH#50724)
            for idx, (name, func) in enumerate(arg):
                key = base.OutputKey(label=name, position=idx)
                # 对数据应用聚合函数 func，将结果存储在 results 中
                results[key] = self.aggregate(func, *args, **kwargs)

        # 如果结果中包含任何 DataFrame，则需要进行连接操作
        if any(isinstance(x, DataFrame) for x in results.values()):
            from pandas import concat

            # 将所有 DataFrame 连接起来，沿着列轴(axis=1)，使用 results 的键作为列的标签
            res_df = concat(
                results.values(), axis=1, keys=[key.label for key in results]
            )
            return res_df

        # 将结果重新索引为输出的格式
        indexed_output = {key.position: val for key, val in results.items()}
        # 使用 indexed_output 构造一个扩展维度的对象，索引为 None
        output = self.obj._constructor_expanddim(indexed_output, index=None)
        # 设置输出对象的列名为 results 中每个 key 的 label
        output.columns = Index(key.label for key in results)

        # 返回最终的输出结果
        return output

    # 包装应用后的输出结果
    def _wrap_applied_output(
        self,
        data: Series,
        values: list[Any],
        not_indexed_same: bool = False,
        is_transform: bool = False,
    ) -> DataFrame | Series:
        """
        Wrap the output of SeriesGroupBy.apply into the expected result.

        Parameters
        ----------
        data : Series
            Input data for groupby operation.
        values : List[Any]
            Applied output for each group.
        not_indexed_same : bool, default False
            Whether the applied outputs are not indexed the same as the group axes.

        Returns
        -------
        DataFrame or Series
        """
        if len(values) == 0:
            # 如果应用结果列表为空
            # 处理空列表情况，返回空的DataFrame或Series，根据is_transform选择索引
            if is_transform:
                # 若为transform操作，使用data的索引
                res_index = data.index
            else:
                # 否则使用self._grouper.result_index的索引
                res_index = self._grouper.result_index

            return self.obj._constructor(
                [],
                name=self.obj.name,
                index=res_index,
                dtype=data.dtype,
            )
        assert values is not None

        if isinstance(values[0], dict):
            # 如果应用结果的第一个元素是字典
            # 根据结果的字典创建DataFrame，使用self._grouper.result_index作为索引
            res_df = self.obj._constructor_expanddim(values, index=self._grouper.result_index)
            # 将DataFrame堆叠为Series，并设置Series的名称为self.obj.name
            res_ser = res_df.stack()
            res_ser.name = self.obj.name
            return res_ser
        elif isinstance(values[0], (Series, DataFrame)):
            # 如果应用结果的第一个元素是Series或DataFrame
            # 合并这些对象，根据情况调整索引和是否需要插入轴分组器
            result = self._concat_objects(
                values,
                not_indexed_same=not_indexed_same,
                is_transform=is_transform,
            )
            if isinstance(result, Series):
                # 如果合并结果是Series，设置其名称为self.obj.name
                result.name = self.obj.name
            if not self.as_index and not_indexed_same:
                # 如果不要求作为索引，并且应用结果与组轴不匹配，则插入轴分组器并设置索引为默认索引
                result = self._insert_inaxis_grouper(result)
                result.index = default_index(len(result))
            return result
        else:
            # 处理其他情况，即应用结果为普通数据
            # 使用应用结果创建新的DataFrame，使用self._grouper.result_index作为索引，设置名称为self.obj.name
            result = self.obj._constructor(
                data=values, index=self._grouper.result_index, name=self.obj.name
            )
            if not self.as_index:
                # 如果不要求作为索引，则插入轴分组器并设置索引为默认索引
                result = self._insert_inaxis_grouper(result)
                result.index = default_index(len(result))
            return result

    __examples_series_doc = dedent(
        """
    >>> ser = pd.Series([390.0, 350.0, 30.0, 20.0],
    ...                 index=["Falcon", "Falcon", "Parrot", "Parrot"],
    ...                 name="Max Speed")
    >>> grouped = ser.groupby([1, 1, 2, 2])
    >>> grouped.transform(lambda x: (x - x.mean()) / x.std())
        Falcon    0.707107
        Falcon   -0.707107
        Parrot    0.707107
        Parrot   -0.707107
        Name: Max Speed, dtype: float64

    Broadcast result of the transformation

    >>> grouped.transform(lambda x: x.max() - x.min())
    Falcon    40.0
    Falcon    40.0
    Parrot    10.0
    Parrot    10.0
    Name: Max Speed, dtype: float64

    >>> grouped.transform("mean")
    Falcon    370.0
    Falcon    370.0

"""
    # 以下代码片段可能是从文档或者注释中提取的例子，用来说明函数 `transform` 的使用方法和效果
    Parrot     25.0
    Parrot     25.0
    Name: Max Speed, dtype: float64

    # 自版本 1.3.0 起发生的变化
    .. versionchanged:: 1.3.0

    # 结果的数据类型将反映传入 `func` 的返回值，例如：
    # 使用 `lambda x: x.astype(int).max()` 进行转换后的示例
    >>> grouped.transform(lambda x: x.astype(int).max())
    Falcon    390
    Falcon    390
    Parrot     30
    Parrot     30
    Name: Max Speed, dtype: int64
    """
    )

    @Substitution(klass="Series", example=__examples_series_doc)
    @Appender(_transform_template)
    # 定义 `transform` 方法，用于对 `Series` 进行转换操作
    def transform(self, func, *args, engine=None, engine_kwargs=None, **kwargs):
        # 调用内部 `_transform` 方法进行实际的转换操作
        return self._transform(
            func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs
        )

    # 使用 Cython 进行转换操作的私有方法 `_cython_transform`
    def _cython_transform(self, how: str, numeric_only: bool = False, **kwargs):
        obj = self._obj_with_exclusions

        try:
            # 调用 `_grouper` 的 Cython 操作执行转换操作
            result = self._grouper._cython_operation(
                "transform", obj._values, how, 0, **kwargs
            )
        except NotImplementedError as err:
            # 如果出现 `NotImplementedError`，抛出类型错误，并指明具体原因
            # 例如：`test_groupby_raises_string`
            raise TypeError(f"{how} is not supported for {obj.dtype} dtype") from err

        # 根据结果构造新的对象并返回
        return obj._constructor(result, index=self.obj.index, name=obj.name)

    # 通用的转换操作方法 `_transform_general`
    def _transform_general(
        self, func: Callable, engine, engine_kwargs, *args, **kwargs
    ) -> Series:
        """
        使用可调用函数 `func` 进行转换。
        """
        # 如果可能使用 numba 进行加速，则使用 `_transform_with_numba` 方法
        if maybe_use_numba(engine):
            return self._transform_with_numba(
                func, *args, engine_kwargs=engine_kwargs, **kwargs
            )
        # 确保 `func` 是可调用的
        assert callable(func)
        klass = type(self.obj)

        results = []
        # 遍历 `_grouper` 中的每个分组进行转换操作
        for name, group in self._grouper.get_iterator(
            self._obj_with_exclusions,
        ):
            # 设置分组对象的 `name` 属性，用于特定测试场景
            # 需要在 `test_transform_lambda_with_datetimetz` 中使用
            object.__setattr__(group, "name", name)
            # 调用 `func` 对分组进行转换，并将结果添加到 `results` 列表中
            res = func(group, *args, **kwargs)

            results.append(klass(res, index=group.index))

        # 检查 `results` 是否为空，以避免连接时的值错误
        if results:
            from pandas.core.reshape.concat import concat

            # 使用 `concat` 方法将结果列表连接成一个新的对象 `concatenated`
            concatenated = concat(results, ignore_index=True)
            # 设置结果的索引顺序，并返回结果
            result = self._set_result_index_ordered(concatenated)
        else:
            # 如果 `results` 为空，则创建一个新的 `dtype` 为 `np.float64` 的对象作为结果
            result = self.obj._constructor(dtype=np.float64)

        # 设置结果对象的名称并返回
        result.name = self.obj.name
        return result
    def filter(self, func, dropna: bool = True, *args, **kwargs):
        """
        Filter elements from groups that don't satisfy a criterion.

        Elements from groups are filtered if they do not satisfy the
        boolean criterion specified by func.

        Parameters
        ----------
        func : function
            Criterion to apply to each group. Should return True or False.
        dropna : bool
            Drop groups that do not pass the filter. True by default; if False,
            groups that evaluate False are filled with NaNs.

        Returns
        -------
        Series
            The filtered subset of the original Series.

        Notes
        -----
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "A": ["foo", "bar", "foo", "bar", "foo", "bar"],
        ...         "B": [1, 2, 3, 4, 5, 6],
        ...         "C": [2.0, 5.0, 8.0, 1.0, 2.0, 9.0],
        ...     }
        ... )
        >>> grouped = df.groupby("A")
        >>> df.groupby("A").B.filter(lambda x: x.mean() > 3.0)
        1    2
        3    4
        5    6
        Name: B, dtype: int64
        """
        # 根据传入的函数参数类型，创建对应的函数包装器
        if isinstance(func, str):
            wrapper = lambda x: getattr(x, func)(*args, **kwargs)
        else:
            wrapper = lambda x: func(x, *args, **kwargs)

        # 将 np.nan 解释为 False
        def true_and_notna(x) -> bool:
            # 使用函数包装器对分组元素进行操作
            b = wrapper(x)
            # 检查结果是否不是 NaN 并且为 True
            return notna(b) and b

        try:
            # 生成满足条件的分组索引列表
            indices = [
                self._get_index(name)
                for name, group in self._grouper.get_iterator(self._obj_with_exclusions)
                if true_and_notna(group)
            ]
        except (ValueError, TypeError) as err:
            # 捕获并重新抛出异常，提示过滤器必须返回布尔值结果
            raise TypeError("the filter must return a boolean result") from err

        # 应用过滤器到数据，返回过滤后的结果
        filtered = self._apply_filter(indices, dropna)
        return filtered
    @doc(Series.describe)
    # 使用 Series.describe 方法的文档字符串来注释 describe 方法
    def describe(self, percentiles=None, include=None, exclude=None) -> Series:
        # 调用父类的 describe 方法，传递相同的参数
        return super().describe(
            percentiles=percentiles, include=include, exclude=exclude
        )

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ):
        """
        Count distinct observations in each group.

        Parameters
        ----------
        normalize : bool, default False
            If True, the resulting counts will be normalized to percentages.
        sort : bool, default True
            Sort by frequencies.
        ascending : bool, default False
            Sort in ascending order.
        bins : int or array-like, optional
            Not used.
        dropna : bool, default True
            Don't include NaN in the counts.

        Returns
        -------
        Series
            Count of unique values in each group.

        See Also
        --------
        DataFrame.value_counts : Equivalent function for DataFrame.

        Examples
        --------
        >>> lst = ["a", "a", "b", "b"]
        >>> ser = pd.Series(lst)
        >>> ser
        0    a
        1    a
        2    b
        3    b
        dtype: object
        >>> ser.groupby(ser).value_counts()
        a    2
        b    2
        dtype: int64
        """
        pass

    def take(
        self,
        indices: TakeIndexer,
        **kwargs,
    ):
        """
        Return the elements in the given positional indices.

        Parameters
        ----------
        indices : TakeIndexer
            The indices of the elements to take.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        Series | DataFrame
            Subset of the original object based on indices.

        Notes
        -----
        The behavior of this function can vary depending on the indices provided.

        Examples
        --------
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> indices = [0, 2]
        >>> ser.take(indices)
        0    1
        2    3
        dtype: int64
        """
        pass
    ) -> Series:
        """
        返回每个分组中给定*位置*索引的元素。

        这意味着我们根据对象中元素的实际位置而不是索引属性中的实际值进行索引。
        如果某个分组中不存在请求的索引，则此方法会引发异常。
        若要获取类似的行为但忽略不存在的索引，请参阅:meth:`.SeriesGroupBy.nth`。

        Parameters
        ----------
        indices : array-like
            一个整数数组，指示每个分组中要获取的位置。

        **kwargs
            与:meth:`numpy.take`兼容性。对输出无影响。

        Returns
        -------
        Series
            包含从每个分组中取出的元素的Series。

        See Also
        --------
        Series.take : 沿着轴从Series中获取元素。
        Series.loc : 根据标签选择DataFrame的子集。
        Series.iloc : 根据位置选择DataFrame的子集。
        numpy.take : 沿轴从数组中获取元素。
        SeriesGroupBy.nth : 类似于take，如果索引不存在则不会引发异常。

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [
        ...         ("falcon", "bird", 389.0),
        ...         ("parrot", "bird", 24.0),
        ...         ("lion", "mammal", 80.5),
        ...         ("monkey", "mammal", np.nan),
        ...         ("rabbit", "mammal", 15.0),
        ...     ],
        ...     columns=["name", "class", "max_speed"],
        ...     index=[4, 3, 2, 1, 0],
        ... )
        >>> df
             name   class  max_speed
        4  falcon    bird      389.0
        3  parrot    bird       24.0
        2    lion  mammal       80.5
        1  monkey  mammal        NaN
        0  rabbit  mammal       15.0
        >>> gb = df["name"].groupby([1, 1, 2, 2, 2])

        获取每个分组中行索引为0和1的元素。

        >>> gb.take([0, 1])
        1  4    falcon
           3    parrot
        2  2      lion
           1    monkey
        Name: name, dtype: object

        我们可以使用负整数来获取正索引的反向元素，从对象的末尾开始，就像Python列表一样。

        >>> gb.take([-1, -2])
        1  3    parrot
           4    falcon
        2  0    rabbit
           1    monkey
        Name: name, dtype: object
        """
        result = self._op_via_apply("take", indices=indices, **kwargs)
        return result
    ) -> Series:
        """
        Return unbiased skew within groups.

        Normalized by N-1.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values when computing the result.

        numeric_only : bool, default False
            Include only float, int, boolean columns. Not implemented for Series.

        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        Series
            Unbiased skew within groups.

        See Also
        --------
        Series.skew : Return unbiased skew over requested axis.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [390.0, 350.0, 357.0, np.nan, 22.0, 20.0, 30.0],
        ...     index=[
        ...         "Falcon",
        ...         "Falcon",
        ...         "Falcon",
        ...         "Falcon",
        ...         "Parrot",
        ...         "Parrot",
        ...         "Parrot",
        ...     ],
        ...     name="Max Speed",
        ... )
        >>> ser
        Falcon    390.0
        Falcon    350.0
        Falcon    357.0
        Falcon      NaN
        Parrot     22.0
        Parrot     20.0
        Parrot     30.0
        Name: Max Speed, dtype: float64
        >>> ser.groupby(level=0).skew()
        Falcon    1.525174
        Parrot    1.457863
        Name: Max Speed, dtype: float64
        >>> ser.groupby(level=0).skew(skipna=False)
        Falcon         NaN
        Parrot    1.457863
        Name: Max Speed, dtype: float64
        """

        def alt(obj):
            # This should not be reached since the cython path should raise
            #  TypeError and not NotImplementedError.
            raise TypeError(f"'skew' is not supported for dtype={obj.dtype}")

        return self._cython_agg_general(
            "skew", alt=alt, skipna=skipna, numeric_only=numeric_only, **kwargs
        )


    @property
    @doc(Series.plot.__doc__)
    def plot(self) -> GroupByPlot:
        result = GroupByPlot(self)
        return result


    @doc(Series.nlargest.__doc__)
    def nlargest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> Series:
        f = partial(Series.nlargest, n=n, keep=keep)
        data = self._obj_with_exclusions
        # Don't change behavior if result index happens to be the same, i.e.
        # already ordered and n >= all group sizes.
        result = self._python_apply_general(f, data, not_indexed_same=True)
        return result


    @doc(Series.nsmallest.__doc__)
    def nsmallest(
        self, n: int = 5, keep: Literal["first", "last", "all"] = "first"
    ) -> Series:
        f = partial(Series.nsmallest, n=n, keep=keep)
        data = self._obj_with_exclusions
        # Don't change behavior if result index happens to be the same, i.e.
        # already ordered and n >= all group sizes.
        result = self._python_apply_general(f, data, not_indexed_same=True)
        return result
    ) -> Series:
        # 使用 functools.partial 来创建一个固定参数的函数 f，调用 Series.nsmallest，指定返回的 n 小的值和保留策略 keep
        f = partial(Series.nsmallest, n=n, keep=keep)
        # 获取带排除条件的对象数据
        data = self._obj_with_exclusions
        # 应用通用的 Python 逻辑，传入函数 f 和数据 data，并指定 not_indexed_same=True，保持结果索引的一致性
        result = self._python_apply_general(f, data, not_indexed_same=True)
        # 返回计算结果
        return result

    def idxmin(self, skipna: bool = True) -> Series:
        """
        返回最小值的行标签。

        如果有多个值相等，返回第一个具有该值的行标签。

        参数
        ----------
        skipna : bool, 默认 True
            排除 NA 值。

        返回
        -------
        Index
            最小值的标签。

        异常
        ------
        ValueError
            如果 Series 为空，或 skipna=False 且存在 NA 值。

        参考
        --------
        numpy.argmin : 返回沿给定轴的最小值的索引。
        DataFrame.idxmin : 返回请求轴上最小值的第一个出现的索引。
        Series.idxmax : 返回第一个出现的最大值的索引 *标签*。

        示例
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64

        >>> ser.groupby(["a", "a", "b", "b"]).idxmin()
        a   2023-01-01
        b   2023-02-01
        dtype: datetime64[s]
        """
        # 调用内部方法 _idxmax_idxmin，传入方法名 "idxmin" 和 skipna 参数
        return self._idxmax_idxmin("idxmin", skipna=skipna)
    def idxmax(self, skipna: bool = True) -> Series:
        """
        Return the row label of the maximum value.

        If multiple values equal the maximum, the first row label with that
        value is returned.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA values.

        Returns
        -------
        Index
            Label of the maximum value.

        Raises
        ------
        ValueError
            If the Series is empty or skipna=False and any value is NA.

        See Also
        --------
        numpy.argmax : Return indices of the maximum values
            along the given axis.
        DataFrame.idxmax : Return index of first occurrence of maximum
            over requested axis.
        Series.idxmin : Return index *label* of the first occurrence
            of minimum of values.

        Examples
        --------
        >>> ser = pd.Series(
        ...     [1, 2, 3, 4],
        ...     index=pd.DatetimeIndex(
        ...         ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15"]
        ...     ),
        ... )
        >>> ser
        2023-01-01    1
        2023-01-15    2
        2023-02-01    3
        2023-02-15    4
        dtype: int64

        >>> ser.groupby(["a", "a", "b", "b"]).idxmax()
        a   2023-01-15
        b   2023-02-15
        dtype: datetime64[s]
        """
        # 调用内部方法 _idxmax_idxmin 来执行 idxmax 操作，返回最大值对应的索引
        return self._idxmax_idxmin("idxmax", skipna=skipna)

    @doc(Series.corr.__doc__)
    def corr(
        self,
        other: Series,
        method: CorrelationMethod = "pearson",
        min_periods: int | None = None,
    ) -> Series:
        # 通过_apply方法调用相关性计算，返回两个Series对象之间的相关系数
        result = self._op_via_apply(
            "corr", other=other, method=method, min_periods=min_periods
        )
        return result

    @doc(Series.cov.__doc__)
    def cov(
        self, other: Series, min_periods: int | None = None, ddof: int | None = 1
    ) -> Series:
        # 通过_apply方法调用协方差计算，返回两个Series对象之间的协方差
        result = self._op_via_apply(
            "cov", other=other, min_periods=min_periods, ddof=ddof
        )
        return result

    @property
    def is_monotonic_increasing(self) -> Series:
        """
        Return whether each group's values are monotonically increasing.

        Returns
        -------
        Series

        Examples
        --------
        >>> s = pd.Series([2, 1, 3, 4], index=["Falcon", "Falcon", "Parrot", "Parrot"])
        >>> s.groupby(level=0).is_monotonic_increasing
        Falcon    False
        Parrot     True
        dtype: bool
        """
        # 对每个分组的Series应用is_monotonic_increasing方法，返回布尔Series指示每组是否单调递增
        return self.apply(lambda ser: ser.is_monotonic_increasing)
    # 定义一个方法，用于检查每个分组的数值是否单调递减
    def is_monotonic_decreasing(self) -> Series:
        """
        Return whether each group's values are monotonically decreasing.

        Returns
        -------
        Series

        Examples
        --------
        >>> s = pd.Series([2, 1, 3, 4], index=["Falcon", "Falcon", "Parrot", "Parrot"])
        >>> s.groupby(level=0).is_monotonic_decreasing
        Falcon     True
        Parrot    False
        dtype: bool
        """
        # 应用 lambda 函数，检查每个序列是否单调递减，并返回结果
        return self.apply(lambda ser: ser.is_monotonic_decreasing)

    # 用于生成直方图的方法，支持多种参数配置
    @doc(Series.hist.__doc__)
    def hist(
        self,
        by=None,
        ax=None,
        grid: bool = True,
        xlabelsize: int | None = None,
        xrot: float | None = None,
        ylabelsize: int | None = None,
        yrot: float | None = None,
        figsize: tuple[float, float] | None = None,
        bins: int | Sequence[int] = 10,
        backend: str | None = None,
        legend: bool = False,
        **kwargs,
    ):
        # 调用 _op_via_apply 方法，通过 apply 方法应用 hist 操作
        result = self._op_via_apply(
            "hist",
            by=by,
            ax=ax,
            grid=grid,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            figsize=figsize,
            bins=bins,
            backend=backend,
            legend=legend,
            **kwargs,
        )
        return result

    # 返回每个分组的数据类型的方法
    @property
    @doc(Series.dtype.__doc__)
    def dtype(self) -> Series:
        # 应用 lambda 函数，返回每个序列的数据类型
        return self.apply(lambda ser: ser.dtype)

    # 返回每个分组的唯一值的方法
    def unique(self) -> Series:
        """
        Return unique values for each group.

        It returns unique values for each of the grouped values. Returned in
        order of appearance. Hash table-based unique, therefore does NOT sort.

        Returns
        -------
        Series
            Unique values for each of the grouped values.

        See Also
        --------
        Series.unique : Return unique values of Series object.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     [
        ...         ("Chihuahua", "dog", 6.1),
        ...         ("Beagle", "dog", 15.2),
        ...         ("Chihuahua", "dog", 6.9),
        ...         ("Persian", "cat", 9.2),
        ...         ("Chihuahua", "dog", 7),
        ...         ("Persian", "cat", 8.8),
        ...     ],
        ...     columns=["breed", "animal", "height_in"],
        ... )
        >>> df
               breed     animal   height_in
        0  Chihuahua        dog         6.1
        1     Beagle        dog        15.2
        2  Chihuahua        dog         6.9
        3    Persian        cat         9.2
        4  Chihuahua        dog         7.0
        5    Persian        cat         8.8
        >>> ser = df.groupby("animal")["breed"].unique()
        >>> ser
        animal
        cat              [Persian]
        dog    [Chihuahua, Beagle]
        Name: breed, dtype: object
        """
        # 调用 _op_via_apply 方法，通过 apply 方法应用 unique 操作
        result = self._op_via_apply("unique")
        return result
class DataFrameGroupBy(GroupBy[DataFrame]):
    # `_agg_examples_doc` 是一个文档字符串，包含了一些使用示例，用于说明如何使用聚合功能
    _agg_examples_doc = dedent(
        """
        Examples
        --------
        >>> data = {"A": [1, 1, 2, 2],
        ...         "B": [1, 2, 3, 4],
        ...         "C": [0.362838, 0.227877, 1.267767, -0.562860]}
        >>> df = pd.DataFrame(data)
        >>> df
           A  B         C
        0  1  1  0.362838
        1  1  2  0.227877
        2  2  3  1.267767
        3  2  4 -0.562860
    
        The aggregation is for each column.
    
        >>> df.groupby('A').agg('min')
           B         C
        A
        1  1  0.227877
        2  3 -0.562860
    
        Multiple aggregations
    
        >>> df.groupby('A').agg(['min', 'max'])
            B             C
          min max       min       max
        A
        1   1   2  0.227877  0.362838
        2   3   4 -0.562860  1.267767
    
        Select a column for aggregation
    
        >>> df.groupby('A').B.agg(['min', 'max'])
           min  max
        A
        1    1    2
        2    3    4
    
        User-defined function for aggregation
    
        >>> df.groupby('A').agg(lambda x: sum(x) + 2)
            B           C
        A
        1    5    2.590715
        2    9    2.704907
    
        Different aggregations per column
    
        >>> df.groupby('A').agg({'B': ['min', 'max'], 'C': 'sum'})
            B             C
          min max       sum
        A
        1   1   2  0.590715
        2   3   4  0.704907
    
        To control the output names with different aggregations per column,
        pandas supports "named aggregation"
    
        >>> df.groupby("A").agg(
        ...     b_min=pd.NamedAgg(column="B", aggfunc="min"),
        ...     c_sum=pd.NamedAgg(column="C", aggfunc="sum")
        ... )
           b_min     c_sum
        A
        1      1  0.590715
        2      3  0.704907
    
        - The keywords are the *output* column names
        - The values are tuples whose first element is the column to select
          and the second element is the aggregation to apply to that column.
          Pandas provides the ``pandas.NamedAgg`` namedtuple with the fields
          ``['column', 'aggfunc']`` to make it clearer what the arguments are.
          As usual, the aggregation can be a callable or a string alias.
    
        See :ref:`groupby.aggregate.named` for more.
    
        .. versionchanged:: 1.3.0
    
            The resulting dtype will reflect the return value of the aggregating function.
    
        >>> df.groupby("A")[["B"]].agg(lambda x: x.astype(float).min())
              B
        A
        1   1.0
        2   3.0
        """
    )

    # 使用自定义的文档函数 `doc`，结合 `_agg_examples_doc` 和 `DataFrame` 的说明，生成聚合方法 `agg` 的文档
    agg = aggregate  # 将 `aggregate` 方法赋给 `agg` 属性
    # 定义一个函数 `_python_agg_general`，用于对数据进行通用聚合操作
    def _python_agg_general(self, func, *args, **kwargs):
        # 创建一个 lambda 函数 `f`，该函数将 func 应用于输入的参数
        f = lambda x: func(x, *args, **kwargs)

        # 如果数据没有分组 (self.ngroups == 0)，则应用通用的聚合函数 `_python_apply_general`
        # 返回聚合结果
        if self.ngroups == 0:
            return self._python_apply_general(f, self._selected_obj, is_agg=True)

        # 获取经过排除处理的对象 `_obj_with_exclusions`
        obj = self._obj_with_exclusions

        # 如果对象没有列，则应用通用的聚合函数 `_python_apply_general`，返回聚合结果
        if not len(obj.columns):
            return self._python_apply_general(f, self._selected_obj)

        # 创建一个空的输出字典 `output`，用于存储聚合结果
        output: dict[int, ArrayLike] = {}
        # 遍历对象 `obj` 的每对索引和序列
        for idx, (name, ser) in enumerate(obj.items()):
            # 对序列 `ser` 应用聚合函数 `_grouper.agg_series`，并将结果存储在 `output` 中
            result = self._grouper.agg_series(ser, f)
            output[idx] = result

        # 使用输出字典 `output` 创建一个新的结果对象 `res`
        res = self.obj._constructor(output)
        # 将结果对象的列设置为 `obj` 的列，浅复制而非深复制
        res.columns = obj.columns.copy(deep=False)
        # 将聚合结果进行包装并返回
        return self._wrap_aggregated_output(res)

    # 定义一个函数 `_aggregate_frame`，用于对 DataFrame 进行聚合操作
    def _aggregate_frame(self, func, *args, **kwargs) -> DataFrame:
        # 如果分组器 `_grouper` 的键数量不为 1，则引发断言错误
        if self._grouper.nkeys != 1:
            raise AssertionError("Number of keys must be 1")

        # 获取经过排除处理的对象 `_obj_with_exclusions`
        obj = self._obj_with_exclusions

        # 创建一个空的结果字典 `result`，用于存储聚合结果
        result: dict[Hashable, NDFrame | np.ndarray] = {}
        # 使用分组器 `_grouper` 获取对象 `obj` 的迭代器，并遍历每个分组数据框 `grp_df`
        for name, grp_df in self._grouper.get_iterator(obj):
            # 对分组数据框 `grp_df` 应用指定的聚合函数 `func`，并将结果存储在 `result` 中
            fres = func(grp_df, *args, **kwargs)
            result[name] = fres

        # 获取分组器的结果索引 `result_index`
        result_index = self._grouper.result_index
        # 使用结果字典 `result` 创建一个新的结果对象 `out`
        # 结果对象的索引为 `obj` 的列，列为分组器的结果索引 `result_index`
        out = self.obj._constructor(result, index=obj.columns, columns=result_index)
        # 对结果对象进行转置操作并返回
        out = out.T

        return out

    # 定义一个函数 `_wrap_applied_output`，用于包装应用的输出数据
    def _wrap_applied_output(
        self,
        data: DataFrame,
        values: list,
        not_indexed_same: bool = False,
        is_transform: bool = False,
        ):
            # 如果 values 列表为空
            if len(values) == 0:
                # 如果正在进行转换操作
                if is_transform:
                    # 使用 data 的索引作为结果索引
                    res_index = data.index
                else:
                    # 使用 _grouper 的结果索引作为结果索引
                    res_index = self._grouper.result_index

                # 创建一个新的对象 result，结构与 self.obj 相同，使用 res_index 和 data 的列作为索引和列
                result = self.obj._constructor(index=res_index, columns=data.columns)
                # 将 result 转换为与 data 相同的数据类型
                result = result.astype(data.dtypes)
                return result

            # GH12824
            # 使用 values[0] 在此处会破坏 test_groupby_apply_none_first
            # 找到第一个非 None 值
            first_not_none = next(com.not_none(*values), None)

            if first_not_none is None:
                # GH9684 - 所有值都为 None，返回一个空的 DataFrame
                # GH57775 - 确保保留原始 DataFrame 的列和数据类型
                # 创建一个新的对象 result，结构与 self.obj 相同，只使用 data 的列作为列
                result = self.obj._constructor(columns=data.columns)
                # 将 result 转换为与 data 相同的数据类型
                result = result.astype(data.dtypes)
                return result
            elif isinstance(first_not_none, DataFrame):
                # 返回合并后的 DataFrame 对象，values 是 DataFrame 列表
                return self._concat_objects(
                    values,
                    not_indexed_same=not_indexed_same,
                    is_transform=is_transform,
                )

            # 如果 first_not_none 是 np.ndarray 或 Index 类型
            key_index = self._grouper.result_index if self.as_index else None

            if isinstance(first_not_none, (np.ndarray, Index)):
                # GH#1738: values 是不等长度数组列表
                # 转到外部 else 子句
                # 如果 self._selection 不可哈希
                if not is_hashable(self._selection):
                    # 错误：需要为 "name" 添加类型注释
                    name = tuple(self._selection)  # type: ignore[var-annotated, arg-type]
                else:
                    # 错误：赋值时的不兼容类型
                    # （表达式类型为 "Hashable"，变量类型为 "Tuple[Any, ...]"）
                    name = self._selection  # type: ignore[assignment]
                # 返回切片后的构造函数对象，使用 key_index 和 name
                return self.obj._constructor_sliced(values, index=key_index, name=name)
            elif not isinstance(first_not_none, Series):
                # 如果 values 不是 Series 或类似数组，而是标量
                # self._selection 未传递给 Series，因为结果不应采用原始选择的列名称
                if self.as_index:
                    # 返回切片后的构造函数对象，使用 key_index
                    return self.obj._constructor_sliced(values, index=key_index)
                else:
                    # 创建一个新的对象 result，结构与 values 相同，列为 [self._selection]
                    result = self.obj._constructor(values, columns=[self._selection])
                    # 在 result 中插入轴组合器
                    result = self._insert_inaxis_grouper(result)
                    return result
            else:
                # 如果 values 是 Series
                # 返回封装后的应用输出 Series
                return self._wrap_applied_output_series(
                    values,
                    not_indexed_same,
                    first_not_none,
                    key_index,
                    is_transform,
                )
    def _wrap_applied_output_series(
        self,
        values: list[Series],
        not_indexed_same: bool,
        first_not_none,  # 参数缺少类型注释
        key_index: Index | None,
        is_transform: bool,
    ) -> DataFrame | Series:
        kwargs = first_not_none._construct_axes_dict()  # 使用第一个非空 Series 构建参数字典
        backup = Series(**kwargs)  # 使用参数字典创建备份 Series 对象
        values = [x if (x is not None) else backup for x in values]  # 将空的 Series 替换为备份 Series 对象

        all_indexed_same = all_indexes_same(x.index for x in values)  # 检查所有 Series 的索引是否相同

        if not all_indexed_same:
            # GH 8467
            return self._concat_objects(
                values,
                not_indexed_same=True,  # 返回未索引相同的 Series 连接结果
                is_transform=is_transform,
            )

        # Combine values
        # vstack+constructor is faster than concat and handles MI-columns
        stacked_values = np.vstack([np.asarray(v) for v in values])  # 垂直堆叠所有 Series 的值

        index = key_index  # 使用给定的索引
        columns = first_not_none.index.copy()  # 复制第一个非空 Series 的索引
        if columns.name is None:
            # GH6124 - propagate name of Series when it's consistent
            names = {v.name for v in values}  # 收集所有 Series 的名称
            if len(names) == 1:
                columns.name = next(iter(names))  # 如果所有 Series 的名称一致，则传播此名称

        if stacked_values.dtype == object:
            # We'll have the DataFrame constructor do inference
            stacked_values = stacked_values.tolist()  # 如果值的类型是对象，则转换为列表
        result = self.obj._constructor(stacked_values, index=index, columns=columns)  # 使用堆叠的值构建 DataFrame 对象

        if not self.as_index:
            result = self._insert_inaxis_grouper(result)  # 如果不需要作为索引，插入轴分组器

        return result

    def _cython_transform(
        self,
        how: str,
        numeric_only: bool = False,
        **kwargs,
    ) -> DataFrame:
        # We have multi-block tests
        #  e.g. test_rank_min_int, test_cython_transform_frame
        #  test_transform_numeric_ret
        mgr: BlockManager = self._get_data_to_aggregate(
            numeric_only=numeric_only, name=how  # 获取用于聚合的数据块管理器
        )

        def arr_func(bvalues: ArrayLike) -> ArrayLike:
            return self._grouper._cython_operation(
                "transform", bvalues, how, 1, **kwargs  # 执行 Cython 操作的函数，传递参数
            )

        res_mgr = mgr.apply(arr_func)  # 应用函数到数据块管理器

        res_df = self.obj._constructor_from_mgr(res_mgr, axes=res_mgr.axes)  # 使用管理器构造 DataFrame 对象
        return res_df
    # 使用给定的函数和引擎对数据进行通用转换，支持 Numba 加速
    def _transform_general(self, func, engine, engine_kwargs, *args, **kwargs):
        # 如果可能使用 Numba 引擎，则使用 Numba 加速的转换方法
        if maybe_use_numba(engine):
            return self._transform_with_numba(
                func, *args, engine_kwargs=engine_kwargs, **kwargs
            )
        
        # 从 pandas 库中导入 concat 函数
        from pandas.core.reshape.concat import concat

        # 存储应用过的结果
        applied = []
        # 获取排除了某些对象后的对象
        obj = self._obj_with_exclusions
        # 获取分组迭代器
        gen = self._grouper.get_iterator(obj)
        # 获取快速路径和慢速路径
        fast_path, slow_path = self._define_paths(func, *args, **kwargs)

        # 确定使用快速路径还是慢速路径，通过对第一组进行评估来决定
        try:
            name, group = next(gen)
        except StopIteration:
            pass
        else:
            # 设置组的名称属性
            object.__setattr__(group, "name", name)
            try:
                # 选择路径并进行转换
                path, res = self._choose_path(fast_path, slow_path, group)
            except ValueError as err:
                # 处理特定的 ValueError 异常
                msg = "transform must return a scalar value for each group"
                raise ValueError(msg) from err
            if group.size > 0:
                # 对结果进行包装处理，并添加到应用列表中
                res = _wrap_transform_general_frame(self.obj, group, res)
                applied.append(res)

        # 对剩余的组进行计算和处理
        for name, group in gen:
            if group.size == 0:
                continue
            # 设置组的名称属性
            object.__setattr__(group, "name", name)
            # 使用路径对组进行处理
            res = path(group)

            # 对结果进行包装处理，并添加到应用列表中
            res = _wrap_transform_general_frame(self.obj, group, res)
            applied.append(res)

        # 获取对象的列索引
        concat_index = obj.columns
        # 进行连接操作，将所有应用的结果进行连接
        concatenated = concat(
            applied, axis=0, verify_integrity=False, ignore_index=True
        )
        # 重新按照列索引对连接结果进行排序
        concatenated = concatenated.reindex(concat_index, axis=1)
        # 设置结果的索引顺序并返回
        return self._set_result_index_ordered(concatenated)

    # 示例数据框说明文档，包含了示例代码和其输出
    __examples_dataframe_doc = dedent(
        """
    >>> df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
    ...                           'foo', 'bar'],
    ...                    'B' : ['one', 'one', 'two', 'three',
    ...                           'two', 'two'],
    ...                    'C' : [1, 5, 5, 2, 5, 5],
    ...                    'D' : [2.0, 5., 8., 1., 2., 9.]})
    >>> grouped = df.groupby('A')[['C', 'D']]
    >>> grouped.transform(lambda x: (x - x.mean()) / x.std())
            C         D
    0 -1.154701 -0.577350
    1  0.577350  0.000000
    2  0.577350  1.154701
    3 -1.154701 -1.000000
    4  0.577350 -0.577350
    5  0.577350  1.000000

    Broadcast result of the transformation

    >>> grouped.transform(lambda x: x.max() - x.min())
        C    D
    0  4.0  6.0
    1  3.0  8.0
    2  4.0  6.0
    3  3.0  8.0
    4  4.0  6.0
    5  3.0  8.0

    >>> grouped.transform("mean")
        C    D
    """
    )
    0  3.666667  4.0
    1  4.000000  5.0
    2  3.666667  4.0
    3  4.000000  5.0
    4  3.666667  4.0
    5  4.000000  5.0

    .. versionchanged:: 1.3.0

    The resulting dtype will reflect the return value of the passed ``func``,
    for example:

    >>> grouped.transform(lambda x: x.astype(int).max())
    C  D
    0  5  8
    1  5  9
    2  5  8
    3  5  9
    4  5  8
    5  5  9
    """
    )

# 定义一个装饰器函数 `transform`，用于在 DataFrame 上应用变换函数 `func`
@Substitution(klass="DataFrame", example=__examples_dataframe_doc)
@Appender(_transform_template)
def transform(self, func, *args, engine=None, engine_kwargs=None, **kwargs):
    # 调用内部方法 `_transform` 执行实际的转换操作
    return self._transform(
        func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs
    )

# 定义一个内部方法 `_define_paths`，根据传入的 `func` 参数返回两种路径的函数
def _define_paths(self, func, *args, **kwargs):
    # 如果 `func` 是字符串，则使用快速路径和慢速路径执行字符串对应的方法
    if isinstance(func, str):
        fast_path = lambda group: getattr(group, func)(*args, **kwargs)
        slow_path = lambda group: group.apply(
            lambda x: getattr(x, func)(*args, **kwargs), axis=0
        )
    else:
        # 否则，使用快速路径和慢速路径执行传入的函数 `func`
        fast_path = lambda group: func(group, *args, **kwargs)
        slow_path = lambda group: group.apply(
            lambda x: func(x, *args, **kwargs), axis=0
        )
    return fast_path, slow_path

# 定义一个内部方法 `_choose_path`，根据判断条件选择适当的路径和结果
def _choose_path(self, fast_path: Callable, slow_path: Callable, group: DataFrame):
    # 默认选择慢速路径
    path = slow_path
    res = slow_path(group)

    # 如果只有一个组，则无需评估多条路径
    if self.ngroups == 1:
        return path, res

    # 尝试使用快速路径，如果出现异常则回退到慢速路径
    try:
        res_fast = fast_path(group)
    except AssertionError:
        raise  # pragma: no cover
    except Exception:
        # 对于用户定义的函数，无法预测可能发生的异常
        return path, res

    # 验证快速路径返回的结果类型是否符合预期
    if isinstance(res_fast, DataFrame):
        if not res_fast.columns.equals(group.columns):
            return path, res
    elif isinstance(res_fast, Series):
        if not res_fast.index.equals(group.columns):
            return path, res
    else:
        return path, res

    # 如果快速路径返回结果与慢速路径相同，则选择快速路径
    if res_fast.equals(res):
        path = fast_path

    return path, res
    def filter(self, func, dropna: bool = True, *args, **kwargs) -> DataFrame:
        """
        Filter elements from groups that don't satisfy a criterion.

        Elements from groups are filtered if they do not satisfy the
        boolean criterion specified by func.

        Parameters
        ----------
        func : function
            Criterion to apply to each group. Should return True or False.
        dropna : bool
            Drop groups that do not pass the filter. True by default; if False,
            groups that evaluate False are filled with NaNs.
        *args
            Additional positional arguments to pass to `func`.
        **kwargs
            Additional keyword arguments to pass to `func`.

        Returns
        -------
        DataFrame
            The filtered subset of the original DataFrame.

        Notes
        -----
        Each subframe is endowed the attribute 'name' in case you need to know
        which group you are working on.

        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "A": ["foo", "bar", "foo", "bar", "foo", "bar"],
        ...         "B": [1, 2, 3, 4, 5, 6],
        ...         "C": [2.0, 5.0, 8.0, 1.0, 2.0, 9.0],
        ...     }
        ... )
        >>> grouped = df.groupby("A")
        >>> grouped.filter(lambda x: x["B"].mean() > 3.0)
             A  B    C
        1  bar  2  5.0
        3  bar  4  1.0
        5  bar  6  9.0
        """
        # Initialize an empty list to store indices of groups that pass the filter
        indices = []

        # Get the selected object (DataFrame or Series) to operate on
        obj = self._selected_obj
        
        # Get an iterator over the groups based on the grouper
        gen = self._grouper.get_iterator(obj)

        # Iterate over each group and its corresponding name
        for name, group in gen:
            # Assign 'name' attribute to the group for identification purposes
            object.__setattr__(group, "name", name)

            # Apply the filter function 'func' to the group with additional arguments
            res = func(group, *args, **kwargs)

            # Attempt to squeeze the result to handle different return types
            try:
                res = res.squeeze()
            except AttributeError:  # Allow for scalars and frames to pass
                pass

            # Determine if the result of the filter is a boolean or NaN
            if is_bool(res) or (is_scalar(res) and isna(res)):
                # If the result is not NaN and evaluates to True, add its index to indices
                if notna(res) and res:
                    indices.append(self._get_index(name))
            else:
                # Raise a TypeError if the result is not a scalar boolean
                raise TypeError(
                    f"filter function returned a {type(res).__name__}, "
                    "but expected a scalar bool"
                )

        # Apply the filter using collected indices and return the filtered DataFrame
        return self._apply_filter(indices, dropna)
    def __getitem__(self, key) -> DataFrameGroupBy | SeriesGroupBy:
        # 根据 GitHub issue 23566
        # 检查键是否为元组且长度大于1，如果是，则抛出值错误异常
        if isinstance(key, tuple) and len(key) > 1:
            raise ValueError(
                "Cannot subset columns with a tuple with more than one element. "
                "Use a list instead."
            )
        # 调用父类的 __getitem__ 方法，返回结果
        return super().__getitem__(key)

    def _gotitem(self, key, ndim: int, subset=None):
        """
        子类需定义
        返回一个切片对象

        Parameters
        ----------
        key : string / list of selections
            字符串或选择列表
        ndim : {1, 2}
            结果的请求维度
        subset : object, default None
            要操作的子集
        """
        if ndim == 2:
            if subset is None:
                subset = self.obj
            # 返回一个 DataFrameGroupBy 对象，基于指定的键和属性
            return DataFrameGroupBy(
                subset,
                self.keys,
                level=self.level,
                grouper=self._grouper,
                exclusions=self.exclusions,
                selection=key,
                as_index=self.as_index,
                sort=self.sort,
                group_keys=self.group_keys,
                observed=self.observed,
                dropna=self.dropna,
            )
        elif ndim == 1:
            if subset is None:
                subset = self.obj[key]
            # 返回一个 SeriesGroupBy 对象，基于指定的键和属性
            return SeriesGroupBy(
                subset,
                self.keys,
                level=self.level,
                grouper=self._grouper,
                exclusions=self.exclusions,
                selection=key,
                as_index=self.as_index,
                sort=self.sort,
                group_keys=self.group_keys,
                observed=self.observed,
                dropna=self.dropna,
            )

        # 如果 ndim 不是 1 或 2，则抛出断言错误
        raise AssertionError("invalid ndim for _gotitem")

    def _get_data_to_aggregate(
        self, *, numeric_only: bool = False, name: str | None = None
    ) -> BlockManager:
        # 获取用于聚合的数据
        obj = self._obj_with_exclusions
        mgr = obj._mgr
        if numeric_only:
            # 如果只选择数值类型数据，则获取数值数据的管理器
            mgr = mgr.get_numeric_data()
        # 返回管理器对象
        return mgr

    def _wrap_agged_manager(self, mgr: BlockManager) -> DataFrame:
        # 使用管理器对象构造 DataFrame，并返回
        return self.obj._constructor_from_mgr(mgr, axes=mgr.axes)
    def _apply_to_column_groupbys(self, func) -> DataFrame:
        # 导入 pandas 的 concat 函数
        from pandas.core.reshape.concat import concat

        # 获取当前对象，包括其排除的部分
        obj = self._obj_with_exclusions
        # 获取对象的所有列
        columns = obj.columns
        # 生成以每一列为分组的 SeriesGroupBy 对象生成器
        sgbs = (
            SeriesGroupBy(
                obj.iloc[:, i],
                selection=colname,
                grouper=self._grouper,
                exclusions=self.exclusions,
                observed=self.observed,
            )
            for i, colname in enumerate(obj.columns)
        )
        # 对每个分组应用传入的函数，并收集结果
        results = [func(sgb) for sgb in sgbs]

        # 如果结果为空列表，则创建一个空的 DataFrame，列名为原始列，索引为分组结果索引
        if not len(results):
            # concat 会引发错误
            res_df = DataFrame([], columns=columns, index=self._grouper.result_index)
        else:
            # 否则，将结果通过 concat 连接，列名使用原始列，沿着列方向连接
            res_df = concat(results, keys=columns, axis=1)

        # 如果不要求使用默认索引
        if not self.as_index:
            # 重置结果 DataFrame 的索引为默认索引，并插入在轴上的分组器
            res_df.index = default_index(len(res_df))
            res_df = self._insert_inaxis_grouper(res_df)
        # 返回结果 DataFrame
        return res_df

    def nunique(self, dropna: bool = True) -> DataFrame:
        """
        返回每个位置上唯一元素的计数的 DataFrame。

        参数
        ----------
        dropna : bool, 默认 True
            在计数中不包括 NaN 值。

        返回
        -------
        nunique: DataFrame
            每个位置上唯一元素的计数。

        示例
        --------
        >>> df = pd.DataFrame(
        ...     {
        ...         "id": ["spam", "egg", "egg", "spam", "ham", "ham"],
        ...         "value1": [1, 5, 5, 2, 5, 5],
        ...         "value2": list("abbaxy"),
        ...     }
        ... )
        >>> df
             id  value1 value2
        0  spam       1      a
        1   egg       5      b
        2   egg       5      b
        3  spam       2      a
        4   ham       5      x
        5   ham       5      y

        >>> df.groupby("id").nunique()
              value1  value2
        id
        egg        1       1
        ham        1       2
        spam       2       1

        检查具有相同 id 但值冲突的行：

        >>> df.groupby("id").filter(lambda g: (g.nunique() > 1).any())
             id  value1 value2
        0  spam       1      a
        3  spam       2      a
        4   ham       5      x
        5   ham       5      y
        """
        # 将函数应用于每个列分组，并返回结果
        return self._apply_to_column_groupbys(lambda sgb: sgb.nunique(dropna))

    def idxmax(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
        *args,
        **kwargs
    ):
        #```python
        # 返回具有最大值的索引位置

        # 此处的实现需要补充，请继续完成代码注释
        pass


这里我需要更多关于 `idxmax` 方法的信息，以便能够为你提供准确的注释。
    ) -> DataFrame:
        """
        Return index of first occurrence of maximum in each group.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA values.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series
            Indexes of maxima in each group.

        Raises
        ------
        ValueError
            * If a column is empty or skipna=False and any value is NA.

        See Also
        --------
        Series.idxmax : Return index of the maximum element.

        Notes
        -----
        This method is the DataFrame version of ``ndarray.argmax``.

        Examples
        --------
        Consider a dataset containing food consumption in Argentina.

        >>> df = pd.DataFrame(
        ...     {
        ...         "consumption": [10.51, 103.11, 55.48],
        ...         "co2_emissions": [37.2, 19.66, 1712],
        ...     },
        ...     index=["Pork", "Wheat Products", "Beef"],
        ... )

        >>> df
                        consumption  co2_emissions
        Pork                  10.51         37.20
        Wheat Products       103.11         19.66
        Beef                  55.48       1712.00

        By default, it returns the index for the maximum value in each column.

        >>> df.idxmax()
        consumption     Wheat Products
        co2_emissions             Beef
        dtype: object
        """
        # 调用内部方法 `_idxmax_idxmin` 来计算每个分组中最大值的索引
        return self._idxmax_idxmin("idxmax", numeric_only=numeric_only, skipna=skipna)

    def idxmin(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
        ):
        """
        Return index of first occurrence of minimum in each group.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA values.
        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

        Returns
        -------
        Series
            Indexes of minima in each group.

        Raises
        ------
        ValueError
            * If a column is empty or skipna=False and any value is NA.

        See Also
        --------
        Series.idxmin : Return index of the minimum element.

        Notes
        -----
        This method is the DataFrame version of ``ndarray.argmin``.

        Examples
        --------
        Consider a dataset containing food consumption in Argentina.

        >>> df = pd.DataFrame(
        ...     {
        ...         "consumption": [10.51, 103.11, 55.48],
        ...         "co2_emissions": [37.2, 19.66, 1712],
        ...     },
        ...     index=["Pork", "Wheat Products", "Beef"],
        ... )

        >>> df
                        consumption  co2_emissions
        Pork                  10.51         37.20
        Wheat Products       103.11         19.66
        Beef                  55.48       1712.00

        By default, it returns the index for the minimum value in each column.

        >>> df.idxmin()
        consumption            Pork
        co2_emissions    Wheat Products
        dtype: object
        """
        # 调用内部方法 `_idxmax_idxmin` 来计算每个分组中最小值的索引
        return self._idxmax_idxmin("idxmin", numeric_only=numeric_only, skipna=skipna)
    ) -> DataFrame:
        """
        返回每个分组中最小值的第一次出现的索引。

        参数
        ----------
        skipna : bool, 默认为 True
            排除 NA 值。
        numeric_only : bool, 默认为 False
            只包括 `float`、`int` 或布尔型数据。

            .. versionadded:: 1.5.0

        返回
        -------
        Series
            每个分组中最小值的索引。

        异常
        ------
        ValueError
            * 如果列为空或 skipna=False 且存在 NA 值。

        参见
        --------
        Series.idxmin : 返回最小元素的索引。

        注意
        -----
        该方法是 DataFrame 版本的 ``ndarray.argmin``。

        示例
        --------
        假设数据集包含阿根廷的食品消耗情况。

        >>> df = pd.DataFrame(
        ...     {
        ...         "consumption": [10.51, 103.11, 55.48],
        ...         "co2_emissions": [37.2, 19.66, 1712],
        ...     },
        ...     index=["Pork", "Wheat Products", "Beef"],
        ... )

        >>> df
                        consumption  co2_emissions
        Pork                  10.51         37.20
        Wheat Products       103.11         19.66
        Beef                  55.48       1712.00

        默认情况下，返回每列中最小值的索引。

        >>> df.idxmin()
        consumption                Pork
        co2_emissions    Wheat Products
        dtype: object
        """
        return self._idxmax_idxmin("idxmin", numeric_only=numeric_only, skipna=skipna)

    boxplot = boxplot_frame_groupby

    def value_counts(
        self,
        subset: Sequence[Hashable] | None = None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ):
        """
        计算每个唯一值的频数。

        参数
        ----------
        subset : 序列[Hashable] | None，默认为 None
            计算频数的子集。
        normalize : bool，默认为 False
            是否返回相对频率。
        sort : bool，默认为 True
            是否按频数排序。
        ascending : bool，默认为 False
            是否按升序排序。
        dropna : bool，默认为 True
            是否排除缺失值。

        返回
        -------
        Series
            唯一值的频数。

        示例
        --------
        >>> df = pd.DataFrame({'A': [1, 2, 2, 3, 4], 'B': [2, 3, 3, 4, 5]})
        >>> df.value_counts()
        2    2
        3    2
        4    1
        1    1
        dtype: int64
        """
        pass

    def take(
        self,
        indices: TakeIndexer,
        **kwargs,
    ):
        """
        通过索引获取数据。

        参数
        ----------
        indices : TakeIndexer
            要获取的索引数组。
        **kwargs
            其他关键字参数，传递给底层函数。

        返回
        -------
        DataFrame
            根据索引获取的数据框。

        示例
        --------
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'c', 'd']})
        >>> df.take([1, 3])
               A  B
        1      2  b
        3      4  d
        """
        pass

    def skew(
        self,
        skipna: bool = True,
        numeric_only: bool = False,
        **kwargs,
    ):
        """
        计算数据的偏度。

        参数
        ----------
        skipna : bool，默认为 True
            是否排除 NA 值。
        numeric_only : bool，默认为 False
            是否仅包括数值型数据。
        **kwargs
            其他关键字参数，传递给底层函数。

        返回
        -------
        Series
            每列数据的偏度。

        示例
        --------
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
        >>> df.skew()
        A    0.000000
        B    0.000000
        dtype: float64
        """
        pass
    ) -> DataFrame:
        """
        返回分组内的无偏偏斜度。

        通过N-1进行标准化。

        Parameters
        ----------
        skipna : bool, default True
            计算结果时排除 NA/null 值。

        numeric_only : bool, default False
            只包括 float、int 和 boolean 类型的列。

        **kwargs
            要传递给函数的额外关键字参数。

        Returns
        -------
        DataFrame
            分组内的无偏偏斜度。

        See Also
        --------
        DataFrame.skew : 返回请求轴上的无偏偏斜度。

        Examples
        --------
        >>> arrays = [
        ...     ["falcon", "parrot", "cockatoo", "kiwi", "lion", "monkey", "rabbit"],
        ...     ["bird", "bird", "bird", "bird", "mammal", "mammal", "mammal"],
        ... ]
        >>> index = pd.MultiIndex.from_arrays(arrays, names=("name", "class"))
        >>> df = pd.DataFrame(
        ...     {"max_speed": [389.0, 24.0, 70.0, np.nan, 80.5, 21.5, 15.0]},
        ...     index=index,
        ... )
        >>> df
                        max_speed
        name     class
        falcon   bird        389.0
        parrot   bird         24.0
        cockatoo bird         70.0
        kiwi     bird          NaN
        lion     mammal       80.5
        monkey   mammal       21.5
        rabbit   mammal       15.0
        >>> gb = df.groupby(["class"])
        >>> gb.skew()
                max_speed
        class
        bird     1.628296
        mammal   1.669046
        >>> gb.skew(skipna=False)
                max_speed
        class
        bird          NaN
        mammal   1.669046
        """

        def alt(obj):
            # 不应该到达这里，因为 cython 路径应该会引发 TypeError 而不是 NotImplementedError。
            raise TypeError(f"'skew' is not supported for dtype={obj.dtype}")

        return self._cython_agg_general(
            "skew", alt=alt, skipna=skipna, numeric_only=numeric_only, **kwargs
        )

    @property
    @doc(DataFrame.plot.__doc__)
    def plot(self) -> GroupByPlot:
        """
        返回一个 GroupByPlot 对象，用于绘制分组后数据的图形表示。

        Returns
        -------
        GroupByPlot
            用于绘制分组数据的对象。
        """
        result = GroupByPlot(self)
        return result

    @doc(DataFrame.corr.__doc__)
    def corr(
        self,
        method: str | Callable[[np.ndarray, np.ndarray], float] = "pearson",
        min_periods: int = 1,
        numeric_only: bool = False,
    ) -> DataFrame:
        """
        计算数据帧的相关性。

        Parameters
        ----------
        method : str or callable, default 'pearson'
            相关性计算方法。
        min_periods : int, default 1
            最小观测期数，用于计算结果。
        numeric_only : bool, default False
            只包括 float、int 和 boolean 类型的列。

        Returns
        -------
        DataFrame
            相关性矩阵。

        """
        result = self._op_via_apply(
            "corr", method=method, min_periods=min_periods, numeric_only=numeric_only
        )
        return result

    @doc(DataFrame.cov.__doc__)
    def cov(
        self,
        min_periods: int | None = None,
        ddof: int | None = 1,
        numeric_only: bool = False,
    ) -> DataFrame:
        """
        计算数据帧的协方差。

        Parameters
        ----------
        min_periods : int or None, optional
            最小观测期数，用于计算结果。
        ddof : int or None, default 1
            自由度修正。
        numeric_only : bool, default False
            只包括 float、int 和 boolean 类型的列。

        Returns
        -------
        DataFrame
            协方差矩阵。
        """
        result = self._op_via_apply(
            "cov", min_periods=min_periods, ddof=ddof, numeric_only=numeric_only
        )
        return result
    # 定义一个方法 hist，用于绘制数据的直方图
    def hist(
        self,
        column: IndexLabel | None = None,  # 可选参数，指定绘制直方图的列名或索引标签
        by=None,  # 分组依据，通常用于分组绘制直方图
        grid: bool = True,  # 是否显示网格线
        xlabelsize: int | None = None,  # x 轴标签字体大小，如果为 None 则采用默认大小
        xrot: float | None = None,  # x 轴标签旋转角度，如果为 None 则不旋转
        ylabelsize: int | None = None,  # y 轴标签字体大小，如果为 None 则采用默认大小
        yrot: float | None = None,  # y 轴标签旋转角度，如果为 None 则不旋转
        ax=None,  # matplotlib 的 Axes 对象，用于绘制直方图的坐标轴
        sharex: bool = False,  # 是否共享 x 轴刻度
        sharey: bool = False,  # 是否共享 y 轴刻度
        figsize: tuple[float, float] | None = None,  # 图表大小，如果为 None 则采用默认大小
        layout: tuple[int, int] | None = None,  # 子图布局，指定行列数，如果为 None 则自动调整
        bins: int | Sequence[int] = 10,  # 直方图的箱数或箱边界
        backend: str | None = None,  # 绘图后端，例如 'matplotlib' 或 'plotly'
        legend: bool = False,  # 是否显示图例
        **kwargs,  # 其他绘图参数，传递给具体的绘图函数
    ):
    
    # 定义一个方法 corrwith，用于计算当前数据对象与另一个 DataFrame 或 Series 的相关系数
    def corrwith(
        self,
        other: DataFrame | Series,  # 另一个 DataFrame 或 Series 对象，用于计算相关系数
        drop: bool = False,  # 是否丢弃不匹配的行和列
        method: CorrelationMethod = "pearson",  # 相关系数的计算方法，默认为 Pearson 相关系数
        numeric_only: bool = False,  # 是否仅考虑数值类型的列
    ) -> DataFrame:
        """
        Compute pairwise correlation.

        .. deprecated:: 3.0.0

        Pairwise correlation is computed between rows or columns of
        DataFrame with rows or columns of Series or DataFrame. DataFrames
        are first aligned along both axes before computing the
        correlations.

        Parameters
        ----------
        other : DataFrame, Series
            Object with which to compute correlations.
        drop : bool, default False
            Drop missing indices from result.
        method : {'pearson', 'kendall', 'spearman'} or callable
            Method of correlation:

            * pearson : standard correlation coefficient
            * kendall : Kendall Tau correlation coefficient
            * spearman : Spearman rank correlation
            * callable: callable with input two 1d ndarrays
                and returning a float.

        numeric_only : bool, default False
            Include only `float`, `int` or `boolean` data.

            .. versionadded:: 1.5.0

            .. versionchanged:: 2.0.0
                The default value of ``numeric_only`` is now ``False``.

        Returns
        -------
        Series
            Pairwise correlations.

        See Also
        --------
        DataFrame.corr : Compute pairwise correlation of columns.

        Examples
        --------
        >>> df1 = pd.DataFrame(
        ...     {
        ...         "Day": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        ...         "Data": [6, 6, 8, 5, 4, 2, 7, 3, 9],
        ...     }
        ... )
        >>> df2 = pd.DataFrame(
        ...     {
        ...         "Day": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        ...         "Data": [5, 3, 8, 3, 1, 1, 2, 3, 6],
        ...     }
        ... )

        >>> df1.groupby("Day").corrwith(df2)
                 Data  Day
        Day
        1    0.917663  NaN
        2    0.755929  NaN
        3    0.576557  NaN
        """
        # Issue a warning about the deprecation of the method
        warnings.warn(
            "DataFrameGroupBy.corrwith is deprecated",
            FutureWarning,
            # Determine the stack level dynamically
            stacklevel=find_stack_level(),
        )
        # Perform the correlation operation using apply with 'corrwith' method
        result = self._op_via_apply(
            "corrwith",
            other=other,
            drop=drop,
            method=method,
            numeric_only=numeric_only,
        )
        # Return the resulting Series of pairwise correlations
        return result
def _wrap_transform_general_frame(
    obj: DataFrame, group: DataFrame, res: DataFrame | Series
) -> DataFrame:
    from pandas import concat  # 导入 pandas 中的 concat 函数

    if isinstance(res, Series):
        # 如果 res 是一个 Series 对象
        # 我们需要在另一个维度上广播；这将保留数据类型
        # GH14457
        if res.index.is_(obj.index):  # 检查 res 的索引是否与 obj 的索引相同
            # 创建一个新的 DataFrame，将 res 复制多次并按列连接起来
            res_frame = concat([res] * len(group.columns), axis=1, ignore_index=True)
            res_frame.columns = group.columns  # 设置新 DataFrame 的列名为 group 的列名
            res_frame.index = group.index  # 设置新 DataFrame 的索引为 group 的索引
        else:
            # 使用 obj 的构造器，复制 res 的值多次以填充新的 DataFrame
            res_frame = obj._constructor(
                np.tile(res.values, (len(group.index), 1)),
                columns=group.columns,
                index=group.index,
            )
        assert isinstance(res_frame, DataFrame)  # 断言 res_frame 是 DataFrame 类型
        return res_frame  # 返回生成的 DataFrame 对象
    elif isinstance(res, DataFrame) and not res.index.is_(group.index):
        # 如果 res 是一个 DataFrame，但其索引与 group 的索引不同
        return res._align_frame(group)[0]  # 调整 res 与 group 相匹配的索引，并返回调整后的 DataFrame
    else:
        return res  # 返回原始的 res 对象，可能是 DataFrame 或 Series
```