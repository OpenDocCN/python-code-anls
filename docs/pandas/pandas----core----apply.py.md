# `D:\src\scipysrc\pandas\pandas\core\apply.py`

```
from __future__ import annotations  # 导入用于支持注解的未来特性

import abc  # 导入抽象基类模块
from collections import defaultdict  # 导入 defaultdict 类
from collections.abc import Callable  # 导入 Callable 类型
import functools  # 导入 functools 模块
from functools import partial  # 导入 partial 函数
import inspect  # 导入 inspect 模块
from typing import (  # 导入类型提示相关的模块和类
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)

import numpy as np  # 导入 NumPy 库

from pandas._libs.internals import BlockValuesRefs  # 导入 pandas 内部模块
from pandas._typing import (  # 导入 pandas 类型注解
    AggFuncType,
    AggFuncTypeBase,
    AggFuncTypeDict,
    AggObjType,
    Axis,
    AxisInt,
    NDFrameT,
    npt,
)
from pandas.compat._optional import import_optional_dependency  # 导入可选的依赖项导入函数
from pandas.errors import SpecificationError  # 导入 pandas 错误模块中的 SpecificationError 类
from pandas.util._decorators import cache_readonly  # 导入 pandas 工具模块中的 cache_readonly 装饰器

from pandas.core.dtypes.cast import is_nested_object  # 导入 pandas 核心数据类型转换模块中的 is_nested_object 函数
from pandas.core.dtypes.common import (  # 导入 pandas 核心数据类型常见模块中的函数和类
    is_dict_like,
    is_extension_array_dtype,
    is_list_like,
    is_numeric_dtype,
    is_sequence,
)
from pandas.core.dtypes.dtypes import (  # 导入 pandas 核心数据类型模块中的特定数据类型
    CategoricalDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.generic import (  # 导入 pandas 核心通用数据类型模块中的通用数据类型类
    ABCDataFrame,
    ABCNDFrame,
    ABCSeries,
)

from pandas.core._numba.executor import generate_apply_looper  # 导入 pandas 核心 numba 执行器模块中的 generate_apply_looper 函数
import pandas.core.common as com  # 导入 pandas 核心 common 模块并使用别名 com
from pandas.core.construction import ensure_wrapped_if_datetimelike  # 导入 pandas 核心 construction 模块中的 ensure_wrapped_if_datetimelike 函数
from pandas.core.util.numba_ import (  # 导入 pandas 核心工具模块中的 numba 相关工具函数
    get_jit_arguments,
    prepare_function_arguments,
)

if TYPE_CHECKING:  # 如果是类型检查阶段，导入以下类型
    from collections.abc import (  # 导入 collections.abc 模块中的特定类型
        Generator,
        Hashable,
        Iterable,
        MutableMapping,
        Sequence,
    )

    from pandas import (  # 导入 pandas 库中的特定类
        DataFrame,
        Index,
        Series,
    )
    from pandas.core.groupby import GroupBy  # 导入 pandas 核心 groupby 模块中的 GroupBy 类
    from pandas.core.resample import Resampler  # 导入 pandas 核心 resample 模块中的 Resampler 类
    from pandas.core.window.rolling import BaseWindow  # 导入 pandas 核心 window 模块中的 BaseWindow 类

ResType = dict[int, Any]  # 定义 ResType 类型为键为整数，值为任意类型的字典


def frame_apply(  # 定义函数 frame_apply，用于应用函数到 DataFrame 的行或列上
    obj: DataFrame,  # 参数 obj 是一个 DataFrame 对象
    func: AggFuncType,  # 参数 func 是一个聚合函数类型
    axis: Axis = 0,  # 参数 axis 是轴的索引，默认为 0 表示行
    raw: bool = False,  # 参数 raw 表示是否原始数据，默认为 False
    result_type: str | None = None,  # 参数 result_type 表示结果类型，默认为 None
    by_row: Literal[False, "compat"] = "compat",  # 参数 by_row 表示按行还是兼容处理，默认为 "compat"
    engine: str = "python",  # 参数 engine 表示引擎，默认为 "python"
    engine_kwargs: dict[str, bool] | None = None,  # 参数 engine_kwargs 表示引擎的关键字参数，默认为 None
    args=None,  # 参数 args 用于传递位置参数
    kwargs=None,  # 参数 kwargs 用于传递关键字参数
) -> FrameApply:  # 返回类型为 FrameApply 对象
    """construct and return a row or column based frame apply object"""
    axis = obj._get_axis_number(axis)  # 获取轴的索引号

    klass: type[FrameApply]  # 定义 klass 类型为 FrameApply 类

    if axis == 0:  # 如果轴的索引号为 0
        klass = FrameRowApply  # 则 klass 为 FrameRowApply 类
    elif axis == 1:  # 如果轴的索引号为 1
        klass = FrameColumnApply  # 则 klass 为 FrameColumnApply 类

    _, func, _, _ = reconstruct_func(func, **kwargs)  # 重构函数 func，并传递关键字参数 kwargs
    assert func is not None  # 确保 func 不为空

    return klass(  # 返回 klass 对象的实例化结果
        obj,
        func,
        raw=raw,
        result_type=result_type,
        by_row=by_row,
        engine=engine,
        engine_kwargs=engine_kwargs,
        args=args,
        kwargs=kwargs,
    )


class Apply(metaclass=abc.ABCMeta):  # 定义 Apply 类，使用 abc.ABCMeta 作为元类
    axis: AxisInt  # 属性 axis 表示轴的整数类型

    def __init__(  # 构造函数，初始化 Apply 类对象
        self,
        obj: AggObjType,  # 参数 obj 是聚合对象类型
        func: AggFuncType,  # 参数 func 是聚合函数类型
        raw: bool,  # 参数 raw 表示是否原始数据
        result_type: str | None,  # 参数 result_type 表示结果类型
        *,
        by_row: Literal[False, "compat", "_compat"] = "compat",  # 参数 by_row 表示按行或兼容处理
        engine: str = "python",  # 参数 engine 表示引擎，默认为 "python"
        engine_kwargs: dict[str, bool] | None = None,  # 参数 engine_kwargs 表示引擎关键字参数，默认为 None
        args,  # 位置参数
        kwargs,  # 关键字参数
    ) -> None:
        # 初始化方法，设置对象和原始数据
        self.obj = obj
        self.raw = raw

        # 检查 by_row 参数是否为 False 或者是 ["compat", "_compat"] 中的一个
        assert by_row is False or by_row in ["compat", "_compat"]
        self.by_row = by_row

        # 设置参数和关键字参数
        self.args = args or ()
        self.kwargs = kwargs or {}

        # 设置引擎和引擎关键字参数
        self.engine = engine
        self.engine_kwargs = {} if engine_kwargs is None else engine_kwargs

        # 检查 result_type 参数是否为 None 或者 "reduce", "broadcast", "expand" 中的一个，否则抛出 ValueError
        if result_type not in [None, "reduce", "broadcast", "expand"]:
            raise ValueError(
                "invalid value for result_type, must be one "
                "of {None, 'reduce', 'broadcast', 'expand'}"
            )

        self.result_type = result_type

        # 设置函数对象
        self.func = func

    @abc.abstractmethod
    def apply(self) -> DataFrame | Series:
        # 抽象方法，应用操作并返回 DataFrame 或 Series 对象
        pass

    @abc.abstractmethod
    def agg_or_apply_list_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        # 抽象方法，处理类似列表的操作名称并返回 DataFrame 或 Series 对象
        pass

    @abc.abstractmethod
    def agg_or_apply_dict_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        # 抽象方法，处理类似字典的操作名称并返回 DataFrame 或 Series 对象
        pass

    def agg(self) -> DataFrame | Series | None:
        """
        提供聚合器的实现。

        Returns
        -------
        聚合结果，如果该方法无法执行聚合，则返回 None。
        """
        func = self.func

        # 如果 func 是字符串，则调用 apply_str 方法
        if isinstance(func, str):
            return self.apply_str()

        # 如果 func 是类似字典的对象，则调用 agg_dict_like 方法
        if is_dict_like(func):
            return self.agg_dict_like()
        # 如果 func 是类似列表的对象，则调用 agg_list_like 方法
        elif is_list_like(func):
            # 我们需要一个列表，但不是 'str'
            return self.agg_list_like()

        # 否则返回 None，调用者可以根据需要做出反应
        return None
    def transform(self) -> DataFrame | Series:
        """
        Transform a DataFrame or Series.

        Returns
        -------
        DataFrame or Series
            Result of applying ``func`` along the given axis of the
            Series or DataFrame.

        Raises
        ------
        ValueError
            If the transform function fails or does not transform.
        """
        # 获取对象（DataFrame 或 Series）、转换函数、轴、位置参数和关键字参数
        obj = self.obj
        func = self.func
        axis = self.axis
        args = self.args
        kwargs = self.kwargs

        # 判断是否为 Series
        is_series = obj.ndim == 1

        # 如果 axis 对应列的索引是 1
        if obj._get_axis_number(axis) == 1:
            # 确保不是 Series
            assert not is_series
            # 对 DataFrame 进行转置后应用 transform，再次转置回来
            return obj.T.transform(func, 0, *args, **kwargs).T

        # 如果 func 是类列表对象但不是字典对象
        if is_list_like(func) and not is_dict_like(func):
            # 将 func 转换为等效的字典对象
            func = cast(list[AggFuncTypeBase], func)
            # 如果是 Series，则将每个函数映射到其可调用名称或本身
            if is_series:
                func = {com.get_callable_name(v) or v: v for v in func}
            else:
                # 否则，将每列映射到相同的函数
                func = {col: func for col in obj}

        # 如果 func 是字典对象
        if is_dict_like(func):
            # 将 func 转换为字典对象
            func = cast(AggFuncTypeDict, func)
            return self.transform_dict_like(func)

        # func 可能是字符串或可调用对象
        func = cast(AggFuncTypeBase, func)
        try:
            # 尝试使用字符串或可调用对象进行转换
            result = self.transform_str_or_callable(func)
        except TypeError:
            raise
        except Exception as err:
            raise ValueError("Transform function failed") from err

        # 转换函数可能返回空的 Series 或 DataFrame，如果 dtype 不合适
        if (
            isinstance(result, (ABCSeries, ABCDataFrame))
            and result.empty
            and not obj.empty
        ):
            raise ValueError("Transform function failed")

        # 检查结果是否为合适的 Series 或 DataFrame，并且索引是否相同
        if not isinstance(result, (ABCSeries, ABCDataFrame)) or not result.index.equals(
            obj.index  # type: ignore[arg-type]
        ):
            raise ValueError("Function did not transform")

        # 返回结果
        return result
    def transform_dict_like(self, func) -> DataFrame:
        """
        Compute transform in the case of a dict-like func
        """
        from pandas.core.reshape.concat import concat  # 导入 pandas 中的 concat 函数

        obj = self.obj  # 获取 self 对象的 obj 属性，通常是一个 Series 或者 DataFrame
        args = self.args  # 获取 self 对象的 args 属性，用于传递额外的位置参数
        kwargs = self.kwargs  # 获取 self 对象的 kwargs 属性，用于传递额外的关键字参数

        # transform is currently only for Series/DataFrame
        assert isinstance(obj, ABCNDFrame)  # 断言 obj 是 ABCNDFrame 的实例，即 Series 或 DataFrame

        if len(func) == 0:
            raise ValueError("No transform functions were provided")  # 如果 func 长度为0，则抛出 ValueError 异常

        func = self.normalize_dictlike_arg("transform", obj, func)  # 调用 self 的 normalize_dictlike_arg 方法，用于处理 transform 中的函数参数

        results: dict[Hashable, DataFrame | Series] = {}  # 初始化一个空字典 results，用于存储计算结果，键为 Hashable 类型，值为 DataFrame 或 Series

        for name, how in func.items():
            colg = obj._gotitem(name, ndim=1)  # 调用 obj 的 _gotitem 方法，获取指定 name 对应的列数据，ndim=1 表示操作为一维操作
            results[name] = colg.transform(how, 0, *args, **kwargs)  # 使用 how 对 colg 进行变换操作，将结果存储在 results 中

        return concat(results, axis=1)  # 使用 pandas 的 concat 函数将 results 中的数据按列(axis=1)进行连接，并返回结果 DataFrame

    def transform_str_or_callable(self, func) -> DataFrame | Series:
        """
        Compute transform in the case of a string or callable func
        """
        obj = self.obj  # 获取 self 对象的 obj 属性，通常是一个 Series 或者 DataFrame
        args = self.args  # 获取 self 对象的 args 属性，用于传递额外的位置参数
        kwargs = self.kwargs  # 获取 self 对象的 kwargs 属性，用于传递额外的关键字参数

        if isinstance(func, str):
            return self._apply_str(obj, func, *args, **kwargs)  # 如果 func 是字符串，则调用 self 的 _apply_str 方法进行处理，并返回结果

        # Two possible ways to use a UDF - apply or call directly
        try:
            return obj.apply(func, args=args, **kwargs)  # 尝试使用 obj 的 apply 方法调用 func 函数，并传递 args 和 kwargs
        except Exception:
            return func(obj, *args, **kwargs)  # 如果出现异常，则直接调用 func 函数，并传递 obj、args 和 kwargs

    def agg_list_like(self) -> DataFrame | Series:
        """
        Compute aggregation in the case of a list-like argument.

        Returns
        -------
        Result of aggregation.
        """
        return self.agg_or_apply_list_like(op_name="agg")  # 调用 self 的 agg_or_apply_list_like 方法进行聚合操作，并返回结果

    def compute_list_like(
        self,
        op_name: Literal["agg", "apply"],
        selected_obj: Series | DataFrame,
        kwargs: dict[str, Any],
    ):
        """
        Compute aggregation or apply operation in the case of a list-like argument.

        Parameters
        ----------
        op_name : Literal["agg", "apply"]
            Operation type, either 'agg' for aggregation or 'apply' for apply.
        selected_obj : Series or DataFrame
            Selected object to perform operation on.
        kwargs : dict
            Additional keyword arguments to pass to the operation.

        Returns
        -------
        Result of the specified operation.
        """
        # 省略部分内容，具体实现根据实际情况来补充
        pass
    ) -> tuple[list[Hashable] | Index, list[Any]]:
        """
        Compute agg/apply results for like-like input.

        Parameters
        ----------
        op_name : {"agg", "apply"}
            Operation being performed.
        selected_obj : Series or DataFrame
            Data to perform operation on.
        kwargs : dict
            Keyword arguments to pass to the functions.

        Returns
        -------
        keys : list[Hashable] or Index
            Index labels for result.
        results : list
            Data for result. When aggregating with a Series, this can contain any
            Python objects.
        """
        func = cast(list[AggFuncTypeBase], self.func)
        obj = self.obj

        results = []
        keys = []

        # degenerate case: if selected_obj is a Series (1-dimensional)
        if selected_obj.ndim == 1:
            # Iterate over each function `a` in `func`
            for a in func:
                # Get the column group for the selected_obj
                colg = obj._gotitem(selected_obj.name, ndim=1, subset=selected_obj)
                # Determine arguments to pass to the operation
                args = (
                    [self.axis, *self.args]
                    if include_axis(op_name, colg)
                    else self.args
                )
                # Perform the operation `op_name` on colg with function `a`
                new_res = getattr(colg, op_name)(a, *args, **kwargs)
                results.append(new_res)

                # Get a suitable name for the function `a`
                name = com.get_callable_name(a) or a
                keys.append(name)

        else:
            indices = []
            # Iterate over each column `col` in `selected_obj`
            for index, col in enumerate(selected_obj):
                # Get the column group for the column `col`
                colg = obj._gotitem(col, ndim=1, subset=selected_obj.iloc[:, index])
                # Determine arguments to pass to the operation
                args = (
                    [self.axis, *self.args]
                    if include_axis(op_name, colg)
                    else self.args
                )
                # Perform the operation `op_name` on colg with the list of functions `func`
                new_res = getattr(colg, op_name)(func, *args, **kwargs)
                results.append(new_res)
                indices.append(index)
            # Assign column names from selected_obj based on indices
            keys = selected_obj.columns.take(indices)  # type: ignore[assignment]

        return keys, results

    def wrap_results_list_like(
        self, keys: Iterable[Hashable], results: list[Series | DataFrame]
    ):
        from pandas.core.reshape.concat import concat

        obj = self.obj

        try:
            # Concatenate results into a DataFrame or Series using keys
            return concat(results, keys=keys, axis=1, sort=False)
        except TypeError as err:
            # Handle TypeError caused by attempting to concatenate non-NDFrame objects
            from pandas import Series

            # Convert results to a Series if it's not already a DataFrame or Series
            result = Series(results, index=keys, name=obj.name)
            # Check if result is a nested object, raise ValueError if so
            if is_nested_object(result):
                raise ValueError(
                    "cannot combine transform and aggregation operations"
                ) from err
            return result
    def agg_dict_like(self) -> DataFrame | Series:
        """
        Compute aggregation in the case of a dict-like argument.

        Returns
        -------
        Result of aggregation.
        """
        # 调用agg_or_apply_dict_like方法，执行字典形式的聚合操作
        return self.agg_or_apply_dict_like(op_name="agg")

    def compute_dict_like(
        self,
        op_name: Literal["agg", "apply"],
        selected_obj: Series | DataFrame,
        selection: Hashable | Sequence[Hashable],
        kwargs: dict[str, Any],
    ):
        from pandas import Index
        from pandas.core.reshape.concat import concat

        obj = self.obj  # 获取当前对象的引用

        # 避免在后续的所有和任何操作中进行两次isinstance调用
        is_ndframe = [isinstance(r, ABCNDFrame) for r in result_data]

        if all(is_ndframe):
            # 如果所有结果都是NDFrame对象
            results = [result for result in result_data if not result.empty]
            # 从结果中提取非空的键
            keys_to_use: Iterable[Hashable]
            keys_to_use = [k for k, v in zip(result_index, result_data) if not v.empty]
            # 如果没有非空的键，则使用所有的结果索引作为键，使用所有的结果数据
            if keys_to_use == []:
                keys_to_use = result_index
                results = result_data

            if selected_obj.ndim == 2:
                # 如果selected_obj是二维的，键应当是列，因此保留列名
                ktu = Index(keys_to_use)
                ktu._set_names(selected_obj.columns.names)
                keys_to_use = ktu

            axis: AxisInt = 0 if isinstance(obj, ABCSeries) else 1
            # 沿着指定轴拼接结果
            result = concat(
                results,
                axis=axis,
                keys=keys_to_use,
            )
        elif any(is_ndframe):
            # 如果有混合的NDFrame和标量
            raise ValueError(
                "cannot perform both aggregation "
                "and transformation operations "
                "simultaneously"
            )
        else:
            from pandas import Series

            # 处理标量值列表的情况
            # 如果obj是Series，则使用其名称作为结果的名称
            if obj.ndim == 1:
                obj = cast("Series", obj)
                name = obj.name
            else:
                name = None

            # 创建一个Series对象作为结果
            result = Series(result_data, index=result_index, name=name)

        return result
    def apply_str(self) -> DataFrame | Series:
        """
        Compute apply in case of a string.

        Returns
        -------
        result: Series or DataFrame
            Resulting Series or DataFrame after applying the function.

        """

        # Caller is responsible for checking isinstance(self.f, str)
        func = cast(str, self.func)  # Cast self.func to a string

        obj = self.obj  # Reference to self.obj

        from pandas.core.groupby.generic import (
            DataFrameGroupBy,
            SeriesGroupBy,
        )

        # Support for `frame.transform('method')`
        # Some methods (shift, etc.) require the axis argument, others
        # don't, so inspect and insert if necessary.
        method = getattr(obj, func, None)  # Get method named func from obj
        if callable(method):
            sig = inspect.getfullargspec(method)  # Get full signature of method
            arg_names = (*sig.args, *sig.kwonlyargs)  # Combine args and kwonlyargs
            if self.axis != 0 and (
                "axis" not in arg_names or func in ("corrwith", "skew")
            ):
                raise ValueError(f"Operation {func} does not support axis=1")
            if "axis" in arg_names and not isinstance(
                obj, (SeriesGroupBy, DataFrameGroupBy)
            ):
                self.kwargs["axis"] = self.axis  # Set axis in kwargs if applicable
        return self._apply_str(obj, func, *self.args, **self.kwargs)  # Call _apply_str method with obj, func, args, and kwargs

    def apply_list_or_dict_like(self) -> DataFrame | Series:
        """
        Compute apply in case of a list-like or dict-like.

        Returns
        -------
        result: Series, DataFrame, or None
            Resulting Series, DataFrame, or None based on type of self.func.

        """

        if self.engine == "numba":
            raise NotImplementedError(
                "The 'numba' engine doesn't support list-like/"
                "dict likes of callables yet."
            )

        if self.axis == 1 and isinstance(self.obj, ABCDataFrame):
            return self.obj.T.apply(self.func, 0, args=self.args, **self.kwargs).T
            # Transpose obj if axis=1 and obj is an instance of ABCDataFrame, then apply self.func with specified args and kwargs

        func = self.func  # Reference to self.func
        kwargs = self.kwargs  # Reference to self.kwargs

        if is_dict_like(func):
            result = self.agg_or_apply_dict_like(op_name="apply")  # Apply function for dict-like objects
        else:
            result = self.agg_or_apply_list_like(op_name="apply")  # Apply function for list-like objects

        result = reconstruct_and_relabel_result(result, func, **kwargs)  # Reconstruct and relabel result based on func and kwargs

        return result

    def normalize_dictlike_arg(
        self, how: str, obj: DataFrame | Series, func: AggFuncTypeDict
    ):
        """
        Normalize dict-like arguments.

        Parameters
        ----------
        how : str
            The normalization method.
        obj : DataFrame or Series
            The object to operate on.
        func : AggFuncTypeDict
            The aggregation function.

        Returns
        -------
        None

        """
    ) -> AggFuncTypeDict:
        """
        Handler for dict-like argument.

        Ensures that necessary columns exist if obj is a DataFrame, and
        that a nested renamer is not passed. Also normalizes to all lists
        when values consists of a mix of list and non-lists.
        """
        assert how in ("apply", "agg", "transform")

        # Can't use func.values(); wouldn't work for a Series
        if (
            how == "agg"
            and isinstance(obj, ABCSeries)
            and any(is_list_like(v) for _, v in func.items())
        ) or (any(is_dict_like(v) for _, v in func.items())):
            # GH 15931 - deprecation of renaming keys
            raise SpecificationError("nested renamer is not supported")

        if obj.ndim != 1:
            # Check for missing columns on a frame
            from pandas import Index

            cols = Index(list(func.keys())).difference(obj.columns, sort=True)
            if len(cols) > 0:
                # GH 58474
                raise KeyError(f"Label(s) {list(cols)} do not exist")

        aggregator_types = (list, tuple, dict)

        # if we have a dict of any non-scalars
        # eg. {'A' : ['mean']}, normalize all to
        # be list-likes
        # Cannot use func.values() because arg may be a Series
        if any(isinstance(x, aggregator_types) for _, x in func.items()):
            new_func: AggFuncTypeDict = {}
            for k, v in func.items():
                if not isinstance(v, aggregator_types):
                    new_func[k] = [v]
                else:
                    new_func[k] = v
            func = new_func
        return func


注释：

        """
        Handler for dict-like argument.

        This function processes a dictionary-like argument `func` to ensure consistency and correctness
        when used with various operations (`apply`, `agg`, `transform`). It performs checks and transformations
        based on the type and content of `func`.

        Parameters:
        - how: str
            Specifies the operation type ('apply', 'agg', 'transform').
        
        Returns:
        - AggFuncTypeDict
            The processed and normalized dictionary `func`.
        """

        assert how in ("apply", "agg", "transform")

        # Check if `func` contains nested renamers or unsupported structures
        if (
            how == "agg"
            and isinstance(obj, ABCSeries)
            and any(is_list_like(v) for _, v in func.items())
        ) or (any(is_dict_like(v) for _, v in func.items())):
            # Raise an error if nested renamers are detected
            raise SpecificationError("nested renamer is not supported")

        # Check if `obj` is a DataFrame and ensure all necessary columns exist
        if obj.ndim != 1:
            from pandas import Index

            cols = Index(list(func.keys())).difference(obj.columns, sort=True)
            if len(cols) > 0:
                # Raise a KeyError if any required columns are missing
                raise KeyError(f"Label(s) {list(cols)} do not exist")

        aggregator_types = (list, tuple, dict)

        # Normalize values in `func` to list-likes if they are not already
        if any(isinstance(x, aggregator_types) for _, x in func.items()):
            new_func: AggFuncTypeDict = {}
            for k, v in func.items():
                if not isinstance(v, aggregator_types):
                    new_func[k] = [v]
                else:
                    new_func[k] = v
            func = new_func

        return func



    def _apply_str(self, obj, func: str, *args, **kwargs):
        """
        if arg is a string, then try to operate on it:
        - try to find a function (or attribute) on obj
        - try to find a numpy function
        - raise
        """
        assert isinstance(func, str)

        if hasattr(obj, func):
            # Attempt to get the attribute `func` from `obj`
            f = getattr(obj, func)
            if callable(f):
                # If `f` is callable, apply it with given `args` and `kwargs`
                return f(*args, **kwargs)

            # Handle case where `func` is an attribute but not callable
            assert len(args) == 0
            assert not any(kwarg == "axis" for kwarg in kwargs)
            return f
        elif hasattr(np, func) and hasattr(obj, "__array__"):
            # If `func` is a function in numpy and `obj` supports array operations
            # Exclude cases where `obj` is a Window
            f = getattr(np, func)
            return f(obj, *args, **kwargs)
        else:
            # Raise an AttributeError if `func` is not valid for `obj`
            msg = f"'{func}' is not a valid function for '{type(obj).__name__}' object"
            raise AttributeError(msg)


注释：

        """
        Handle string-based operations on `obj`.

        This method attempts to operate on `obj` using the string `func`:
        - It first tries to find an attribute or method named `func` on `obj`.
        - If `func` is not found as an attribute, it checks if `func` is a numpy function and `obj` supports array operations.
        - Raises an AttributeError if `func` is neither an attribute of `obj` nor a numpy function applicable to `obj`.

        Parameters:
        - obj: object
            The object on which the operation is performed.
        - func: str
            The string indicating the function or attribute to operate on `obj`.
        - *args, **kwargs: additional arguments passed to the function or attribute.

        Returns:
        - The result of applying `func` on `obj`.
        """
    """
    Methods shared by FrameApply and SeriesApply but
    not GroupByApply or ResamplerWindowApply
    """
    # NDFrameApply 类定义，包含 FrameApply 和 SeriesApply 共享的方法，但不包括 GroupByApply 和 ResamplerWindowApply

    obj: DataFrame | Series
    # 类属性 obj 表示当前对象是 DataFrame 或 Series 的实例

    @property
    def index(self) -> Index:
        # 返回对象 obj 的索引
        return self.obj.index

    @property
    def agg_axis(self) -> Index:
        # 返回根据轴进行聚合操作后的索引
        return self.obj._get_agg_axis(self.axis)

    def agg_or_apply_list_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        # 根据操作名称 op_name （'agg' 或 'apply'）对对象进行列表形式的聚合或应用操作
        obj = self.obj
        kwargs = self.kwargs

        if op_name == "apply":
            if isinstance(self, FrameApply):
                by_row = self.by_row
            elif isinstance(self, SeriesApply):
                by_row = "_compat" if self.by_row else False
            else:
                by_row = False
            kwargs = {**kwargs, "by_row": by_row}

        if getattr(obj, "axis", 0) == 1:
            # 如果对象的轴属性不是0，则抛出未实现错误
            raise NotImplementedError("axis other than 0 is not supported")

        keys, results = self.compute_list_like(op_name, obj, kwargs)
        # 计算列表形式的操作的结果键和结果值
        result = self.wrap_results_list_like(keys, results)
        # 封装列表形式操作的结果并返回
        return result

    def agg_or_apply_dict_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        # 根据操作名称 op_name （'agg' 或 'apply'）对对象进行字典形式的聚合或应用操作
        assert op_name in ["agg", "apply"]
        obj = self.obj

        kwargs = {}
        if op_name == "apply":
            by_row = "_compat" if self.by_row else False
            kwargs.update({"by_row": by_row})

        if getattr(obj, "axis", 0) == 1:
            # 如果对象的轴属性不是0，则抛出未实现错误
            raise NotImplementedError("axis other than 0 is not supported")

        selection = None
        result_index, result_data = self.compute_dict_like(
            op_name, obj, selection, kwargs
        )
        # 计算字典形式操作的结果索引和结果数据
        result = self.wrap_results_dict_like(obj, result_index, result_data)
        # 封装字典形式操作的结果并返回
        return result
    # 定义一个函数生成 Numba 应用函数的框架
    def generate_numba_apply_func(
        func, nogil=True, nopython=True, parallel=False
    ) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]:
        pass

    # 抽象方法：使用 Numba 执行应用操作
    @abc.abstractmethod
    def apply_with_numba(self):
        pass

    # 验证数据类型是否符合 Numba 要求
    def validate_values_for_numba(self) -> None:
        # 遍历对象的各列及其数据类型
        for colname, dtype in self.obj.dtypes.items():
            # 如果数据类型不是数值类型，则抛出值错误异常
            if not is_numeric_dtype(dtype):
                raise ValueError(
                    f"Column {colname} must have a numeric dtype. "
                    f"Found '{dtype}' instead"
                )
            # 如果数据类型是扩展数组类型，则抛出值错误异常
            if is_extension_array_dtype(dtype):
                raise ValueError(
                    f"Column {colname} is backed by an extension array, "
                    f"which is not supported by the numba engine."
                )

    # 抽象方法：为轴向包装结果数据
    @abc.abstractmethod
    def wrap_results_for_axis(
        self, results: ResType, res_index: Index
    ) -> DataFrame | Series:
        pass

    # ---------------------------------------------------------------

    # 返回结果列的索引
    @property
    def res_columns(self) -> Index:
        return self.result_columns

    # 返回对象的列索引
    @property
    def columns(self) -> Index:
        return self.obj.columns

    # 读取对象的值（只读缓存）
    @cache_readonly
    def values(self):
        return self.obj.values
    def apply(self) -> DataFrame | Series:
        """compute the results"""

        # dispatch to handle list-like or dict-like
        if is_list_like(self.func):
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support lists of callables yet"
                )
            return self.apply_list_or_dict_like()

        # all empty
        if len(self.columns) == 0 and len(self.index) == 0:
            return self.apply_empty_result()

        # string dispatch
        if isinstance(self.func, str):
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support using "
                    "a string as the callable function"
                )
            return self.apply_str()

        # ufunc
        elif isinstance(self.func, np.ufunc):
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support "
                    "using a numpy ufunc as the callable function"
                )
            with np.errstate(all="ignore"):
                results = self.obj._mgr.apply("apply", func=self.func)
            # _constructor will retain self.index and self.columns
            return self.obj._constructor_from_mgr(results, axes=results.axes)

        # broadcasting
        if self.result_type == "broadcast":
            if self.engine == "numba":
                raise NotImplementedError(
                    "the 'numba' engine doesn't support result_type='broadcast'"
                )
            return self.apply_broadcast(self.obj)

        # one axis empty
        elif not all(self.obj.shape):
            return self.apply_empty_result()

        # raw
        elif self.raw:
            return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)

        # apply standard
        return self.apply_standard()

    def agg(self):
        obj = self.obj
        axis = self.axis

        # TODO: Avoid having to change state
        # Change state to handle aggregation along the specified axis
        self.obj = self.obj if self.axis == 0 else self.obj.T
        self.axis = 0

        result = None
        try:
            # Perform aggregation operation
            result = super().agg()
        finally:
            # Restore original object and axis after aggregation
            self.obj = obj
            self.axis = axis

        # Transpose result if aggregation was along axis 1
        if axis == 1:
            result = result.T if result is not None else result

        # If no result from super().agg(), apply the function directly
        if result is None:
            result = self.obj.apply(self.func, axis, args=self.args, **self.kwargs)

        return result
    def apply_empty_result(self):
        """
        处理空结果情况；至少有一个轴是0

        尝试对空序列应用函数，以确定这是否是一个减少函数
        """
        assert callable(self.func)  # 断言函数属性为可调用对象

        # 如果不要求减少或推断减少，直接返回现有对象的副本
        if self.result_type not in ["reduce", None]:
            return self.obj.copy()

        # 可能需要推断是否减少
        should_reduce = self.result_type == "reduce"

        from pandas import Series

        if not should_reduce:
            try:
                if self.axis == 0:
                    # 在轴为0的情况下尝试应用函数到空序列
                    r = self.func(
                        Series([], dtype=np.float64), *self.args, **self.kwargs
                    )
                else:
                    # 在轴不为0的情况下尝试应用函数到带索引的空序列
                    r = self.func(
                        Series(index=self.columns, dtype=np.float64),
                        *self.args,
                        **self.kwargs,
                    )
            except Exception:
                pass
            else:
                # 根据结果确定是否需要减少
                should_reduce = not isinstance(r, Series)

        if should_reduce:
            if len(self.agg_axis):
                # 如果有聚合轴，对空序列应用函数
                r = self.func(Series([], dtype=np.float64), *self.args, **self.kwargs)
            else:
                # 否则返回 NaN
                r = np.nan

            return self.obj._constructor_sliced(r, index=self.agg_axis)
        else:
            # 否则返回对象的副本
            return self.obj.copy()
    def apply_raw(self, engine="python", engine_kwargs=None):
        """将函数应用到作为numpy数组的值上"""

        def wrap_function(func):
            """
            包装用户提供的函数以解决numpy问题。

            参见 https://github.com/numpy/numpy/issues/8352
            """

            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                if isinstance(result, str):
                    result = np.array(result, dtype=object)
                return result

            return wrapper

        if engine == "numba":
            args, kwargs = prepare_function_arguments(
                self.func,  # type: ignore[arg-type]
                self.args,
                self.kwargs,
            )
            # 错误："_lru_cache_wrapper"的 "__call__"的第1个参数具有不兼容类型
            # "Callable[..., Any] | str | list[Callable[..., Any] | str] |
            # dict[Hashable,Callable[..., Any] | str | list[Callable[..., Any] | str]]";
            # 预期类型为 "Hashable"
            nb_looper = generate_apply_looper(
                self.func,  # type: ignore[arg-type]
                **get_jit_arguments(engine_kwargs, kwargs),
            )
            result = nb_looper(self.values, self.axis, *args)
            # 如果结果是2维的，将其压缩为1维
            result = np.squeeze(result)
        else:
            result = np.apply_along_axis(
                wrap_function(self.func),
                self.axis,
                self.values,
                *self.args,
                **self.kwargs,
            )

        # TODO: 混合类型情况
        if result.ndim == 2:
            return self.obj._constructor(result, index=self.index, columns=self.columns)
        else:
            return self.obj._constructor_sliced(result, index=self.agg_axis)

    def apply_broadcast(self, target: DataFrame) -> DataFrame:
        assert callable(self.func)

        result_values = np.empty_like(target.values)

        # 我们希望比较兼容性的轴
        result_compare = target.shape[0]

        for i, col in enumerate(target.columns):
            res = self.func(target[col], *self.args, **self.kwargs)
            ares = np.asarray(res).ndim

            # 必须是标量或者1维
            if ares > 1:
                raise ValueError("太多维无法广播")
            if ares == 1:
                # 必须匹配返回的维度
                if result_compare != len(res):
                    raise ValueError("无法广播结果")

            result_values[:, i] = res

        # 我们*总是*保留原始的索引/列
        result = self.obj._constructor(
            result_values, index=target.index, columns=target.columns
        )
        return result
    # 应用标准操作，根据引擎类型选择适当的处理方法并返回处理结果和索引
    def apply_standard(self):
        if self.engine == "python":
            # 使用 Python 引擎处理数据，并获取处理结果和索引
            results, res_index = self.apply_series_generator()
        else:
            # 使用 Numba 引擎处理数据，并获取处理结果和索引
            results, res_index = self.apply_series_numba()

        # 封装处理结果并返回
        return self.wrap_results(results, res_index)

    # 使用生成器方式应用函数到数据系列上，并返回结果和索引
    def apply_series_generator(self) -> tuple[ResType, Index]:
        assert callable(self.func)

        # 获取数据生成器和结果索引
        series_gen = self.series_generator
        res_index = self.result_index

        # 存储处理结果的字典
        results = {}

        # 遍历数据生成器，并应用函数到每个数据上
        for i, v in enumerate(series_gen):
            results[i] = self.func(v, *self.args, **self.kwargs)
            # 如果结果是数据系列的视图，则需要进行浅拷贝以避免底层数据被替换
            if isinstance(results[i], ABCSeries):
                results[i] = results[i].copy(deep=False)

        # 返回处理结果和索引
        return results, res_index

    # 使用 Numba 引擎处理数据，并返回处理结果和索引
    def apply_series_numba(self):
        if self.engine_kwargs.get("parallel", False):
            # 如果并行处理选项为真，则抛出未实现错误
            raise NotImplementedError(
                "Parallel apply is not supported when raw=False and engine='numba'"
            )
        # 如果数据对象的索引或列不唯一，则抛出未实现错误
        if not self.obj.index.is_unique or not self.columns.is_unique:
            raise NotImplementedError(
                "The index/columns must be unique when raw=False and engine='numba'"
            )
        # 验证用于 Numba 处理的数值
        self.validate_values_for_numba()
        # 使用 Numba 处理数据并获取结果
        results = self.apply_with_numba()
        return results, self.result_index

    # 封装处理结果为 DataFrame 或 Series 对象，并返回
    def wrap_results(self, results: ResType, res_index: Index) -> DataFrame | Series:
        from pandas import Series

        # 如果结果长度大于零且第一个结果是序列，则尝试推断结果类型并返回封装后的结果
        if len(results) > 0 and 0 in results and is_sequence(results[0]):
            return self.wrap_results_for_axis(results, res_index)

        # 处理标量值的字典结果

        # 空 Series 的默认数据类型是 `object`，但是当结果为空 Series 时，例如 df.mean() 的情况下，
        # 结果应该具有 `float64` 类型。
        constructor_sliced = self.obj._constructor_sliced
        if len(results) == 0 and constructor_sliced is Series:
            result = constructor_sliced(results, dtype=np.float64)
        else:
            result = constructor_sliced(results)
        result.index = res_index

        # 返回封装后的结果
        return result

    # 应用字符串类型函数到数据上，并返回结果为 DataFrame 或 Series 对象
    def apply_str(self) -> DataFrame | Series:
        # 调用者需负责检查 self.func 是否为字符串类型
        # TODO: GH#39993 - 通过 lambda 表达式替换避免特殊情况处理
        if self.func == "size":
            # 特殊情况处理，因为 DataFrame.size 返回单个标量值
            obj = self.obj
            value = obj.shape[self.axis]
            return obj._constructor_sliced(value, index=self.agg_axis)
        # 调用父类的 apply_str 方法处理其余情况
        return super().apply_str()
# 定义一个类 `FrameRowApply`，继承自 `FrameApply`
class FrameRowApply(FrameApply):
    # 设置属性 `axis` 默认为 0
    axis: AxisInt = 0

    # 定义属性 `series_generator`，返回一个生成器，生成每列的 Series 对象
    @property
    def series_generator(self) -> Generator[Series, None, None]:
        return (self.obj._ixs(i, axis=1) for i in range(len(self.columns)))

    # 定义静态方法 `generate_numba_apply_func`，生成一个使用 Numba 加速的函数
    @staticmethod
    @functools.cache
    def generate_numba_apply_func(
        func, nogil=True, nopython=True, parallel=False
    ) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]:
        # 导入可选依赖项 `numba`
        numba = import_optional_dependency("numba")
        from pandas import Series

        # 导入扩展功能 `maybe_cast_str`，用于将字符串对象转换为 NumPy 字符串
        # 注意：这也会加载我们的 numba 扩展
        from pandas.core._numba.extensions import maybe_cast_str

        # 将函数注册为可 JIT 编译的函数
        jitted_udf = numba.extending.register_jitable(func)

        # 定义使用 Numba 编译的函数 `numba_func`
        @numba.jit(nogil=nogil, nopython=nopython, parallel=parallel)
        def numba_func(values, col_names, df_index, *args):
            results = {}
            for j in range(values.shape[1]):
                # 创建 Series 对象
                ser = Series(
                    values[:, j], index=df_index, name=maybe_cast_str(col_names[j])
                )
                # 调用注册的 JIT 函数处理 Series 对象
                results[j] = jitted_udf(ser, *args)
            return results

        return numba_func

    # 定义方法 `apply_with_numba`，应用 Numba 加速处理
    def apply_with_numba(self) -> dict[int, Any]:
        # 将传入的函数转换为 `Callable` 类型
        func = cast(Callable, self.func)
        # 准备函数的参数和关键字参数
        args, kwargs = prepare_function_arguments(func, self.args, self.kwargs)
        # 生成使用 Numba 加速的处理函数
        nb_func = self.generate_numba_apply_func(
            func, **get_jit_arguments(self.engine_kwargs, kwargs)
        )
        # 导入 `set_numba_data` 函数
        from pandas.core._numba.extensions import set_numba_data

        # 处理索引 `index`
        index = self.obj.index
        if index.dtype == "string":
            index = index.astype(object)

        # 处理列名 `columns`
        columns = self.obj.columns
        if columns.dtype == "string":
            columns = columns.astype(object)

        # 将 Numba 字典转换为常规字典
        # 我们在 DataFrame 构造函数中的 isinstance 检查无法通过 Numba 的类型化字典
        with set_numba_data(index) as index, set_numba_data(columns) as columns:
            # 执行 Numba 加速的处理函数并获取结果
            res = dict(nb_func(self.values, columns, index, *args))
        return res

    # 定义属性 `result_index`，返回处理结果的索引
    @property
    def result_index(self) -> Index:
        return self.columns

    # 定义属性 `result_columns`，返回处理结果的列名
    @property
    def result_columns(self) -> Index:
        return self.index

    # 定义方法 `wrap_results_for_axis`，为特定轴包装处理结果
    def wrap_results_for_axis(
        self, results: ResType, res_index: Index
    ) -> DataFrame | Series:
        """定义函数的返回类型为 DataFrame 或 Series"""

        if self.result_type == "reduce":
            # 如果结果类型是 "reduce"，则将结果使用被操作对象的切片构造函数构建
            res = self.obj._constructor_sliced(results)
            # 设置结果的索引为 res_index
            res.index = res_index
            return res

        elif self.result_type is None and all(
            isinstance(x, dict) for x in results.values()
        ):
            # 如果结果类型为 None，并且所有 results 中的值都是字典，则使用被操作对象的切片构造函数构建结果
            res = self.obj._constructor_sliced(results)
            # 设置结果的索引为 res_index
            res.index = res_index
            return res

        try:
            # 尝试使用结果数据实例化被操作对象的构造函数
            result = self.obj._constructor(data=results)
        except ValueError as err:
            if "All arrays must be of the same length" in str(err):
                # 如果出现 "All arrays must be of the same length" 错误，则使用被操作对象的切片构造函数构建结果
                res = self.obj._constructor_sliced(results)
                # 设置结果的索引为 res_index
                res.index = res_index
                return res
            else:
                raise

        if not isinstance(results[0], ABCSeries):
            if len(result.index) == len(self.res_columns):
                # 如果结果中第一个元素不是 ABCSeries 类型，并且结果的索引长度与 res_columns 相等，则将结果的索引设置为 res_columns
                result.index = self.res_columns

        if len(result.columns) == len(res_index):
            # 如果结果的列数与 res_index 的长度相等，则将结果的列名设置为 res_index
            result.columns = res_index

        return result
class FrameColumnApply(FrameApply):
    axis: AxisInt = 1

    def apply_broadcast(self, target: DataFrame) -> DataFrame:
        # 调用父类的 apply_broadcast 方法，传递 target 的转置作为参数，得到结果的转置
        result = super().apply_broadcast(target.T)
        return result.T

    @property
    def series_generator(self) -> Generator[Series, None, None]:
        # 获取 self.values 的引用
        values = self.values
        # 确保如果是日期时间相关数据，则进行包装处理
        values = ensure_wrapped_if_datetimelike(values)
        # 断言 values 至少有一个元素
        assert len(values) > 0

        # 从 self.obj 中获取第一行数据作为初始 Series 对象
        ser = self.obj._ixs(0, axis=0)
        mgr = ser._mgr

        # 检查是否是视图
        is_view = mgr.blocks[0].refs.has_reference()

        if isinstance(ser.dtype, ExtensionDtype):
            # 对于扩展数据类型，处理 obj 中的每一行数据
            obj = self.obj
            for i in range(len(obj)):
                yield obj._ixs(i, axis=0)

        else:
            # 对于普通数据类型，处理 values 中的每个数组和对应的索引名
            for arr, name in zip(values, self.index):
                # 重新设置 ser 对象的 _mgr，确保更新后的值被正确处理
                ser._mgr = mgr
                # 设置 ser 对象的值为当前的数组 arr
                mgr.set_values(arr)
                # 设置 ser 对象的名称为当前的索引名 name
                object.__setattr__(ser, "_name", name)
                if not is_view:
                    # 如果不是视图，重置块的引用，以避免不必要的拷贝写入
                    mgr.blocks[0].refs = BlockValuesRefs(mgr.blocks[0])
                # 返回生成的 ser 对象
                yield ser

    @staticmethod
    @functools.cache
    def generate_numba_apply_func(
        func, nogil=True, nopython=True, parallel=False
    ) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]:
        # 导入 numba 库
        numba = import_optional_dependency("numba")
        from pandas import Series
        from pandas.core._numba.extensions import maybe_cast_str

        # 将 func 注册为可编译的 numba 函数
        jitted_udf = numba.extending.register_jitable(func)

        @numba.jit(nogil=nogil, nopython=nopython, parallel=parallel)
        def numba_func(values, col_names_index, index, *args):
            # 创建一个空字典，用于存储结果
            results = {}
            # 遍历 values 的每一行
            for i in range(values.shape[0]):
                # 创建 Series 对象，将 values[i] 拷贝为其数据，指定列名索引和名称
                ser = Series(
                    values[i].copy(),
                    index=col_names_index,
                    name=maybe_cast_str(index[i]),
                )
                # 调用预编译的 numba 函数处理当前的 Series 对象和额外参数
                results[i] = jitted_udf(ser, *args)

            return results

        return numba_func
    def apply_with_numba(self) -> dict[int, Any]:
        func = cast(Callable, self.func)  # 将self.func强制转换为可调用对象func
        args, kwargs = prepare_function_arguments(func, self.args, self.kwargs)  # 准备函数参数args和kwargs
        nb_func = self.generate_numba_apply_func(
            func, **get_jit_arguments(self.engine_kwargs, kwargs)
        )  # 生成使用Numba优化后的应用函数nb_func

        from pandas.core._numba.extensions import set_numba_data

        # 转换为普通字典，因为在DataFrame构造函数中的isinstance检查不适用于Numba的类型化字典
        with (
            set_numba_data(self.obj.index) as index,  # 设置Numba数据：索引
            set_numba_data(self.columns) as columns,  # 设置Numba数据：列名
        ):
            res = dict(nb_func(self.values, columns, index, *args))  # 调用nb_func计算结果并转换为普通字典

        return res  # 返回结果字典

    @property
    def result_index(self) -> Index:
        return self.index  # 返回对象的索引作为结果索引

    @property
    def result_columns(self) -> Index:
        return self.columns  # 返回对象的列名作为结果列名

    def wrap_results_for_axis(
        self, results: ResType, res_index: Index
    ) -> DataFrame | Series:
        """返回列的结果"""
        result: DataFrame | Series

        # 如果请求扩展结果
        if self.result_type == "expand":
            result = self.infer_to_same_shape(results, res_index)  # 将结果推断为与输入对象相同形状

        # 如果结果不是Series并且不需要推断
        elif not isinstance(results[0], ABCSeries):
            result = self.obj._constructor_sliced(results)  # 使用切片构造函数创建对象
            result.index = res_index  # 设置结果的索引

        # 可能需要推断结果
        else:
            result = self.infer_to_same_shape(results, res_index)  # 将结果推断为与输入对象相同形状

        return result  # 返回结果

    def infer_to_same_shape(self, results: ResType, res_index: Index) -> DataFrame:
        """将结果推断为与输入对象相同形状"""
        result = self.obj._constructor(data=results)  # 使用数据创建对象
        result = result.T  # 对结果进行转置

        # 设置索引
        result.index = res_index

        # 推断数据类型
        result = result.infer_objects()

        return result  # 返回推断后的结果
class SeriesApply(NDFrameApply):
    obj: Series  # 定义一个属性 obj，类型为 Series，表示该对象操作的数据系列
    axis: AxisInt = 0  # 定义一个属性 axis，类型为 AxisInt，默认值为 0，表示应用操作的轴向
    by_row: Literal[False, "compat", "_compat"]  # 定义一个属性 by_row，限定为 False, "compat", "_compat" 中的一种，仅适用于 apply() 方法

    def __init__(
        self,
        obj: Series,
        func: AggFuncType,
        *,
        by_row: Literal[False, "compat", "_compat"] = "compat",  # 初始化方法，接受 Series 对象和聚合函数作为参数，并且可以指定 by_row 属性的默认值为 "compat"
        args,
        kwargs,
    ) -> None:
        super().__init__(
            obj,
            func,
            raw=False,
            result_type=None,
            by_row=by_row,
            args=args,
            kwargs=kwargs,
        )

    def apply(self) -> DataFrame | Series:
        obj = self.obj

        if len(obj) == 0:
            return self.apply_empty_result()  # 如果对象为空，则调用 apply_empty_result 方法返回一个空的 Series 对象

        # 根据 func 的类型分发到不同的处理方法
        if is_list_like(self.func):
            return self.apply_list_or_dict_like()  # 如果 func 是类列表的对象，则调用 apply_list_or_dict_like 方法处理

        if isinstance(self.func, str):
            # 如果 func 是字符串，则尝试分发到对应的处理方法
            return self.apply_str()

        if self.by_row == "_compat":
            return self.apply_compat()  # 如果 by_row 属性为 "_compat"，则调用 apply_compat 方法处理

        # 如果 func 是可调用对象，则调用 apply_standard 方法处理
        return self.apply_standard()

    def agg(self):
        result = super().agg()  # 调用父类的 agg 方法获得处理结果
        if result is None:
            obj = self.obj
            func = self.func
            assert callable(func)  # 确保 func 是可调用对象
            result = func(obj, *self.args, **self.kwargs)  # 对 obj 应用 func 函数，并传入额外的参数和关键字参数
        return result

    def apply_empty_result(self) -> Series:
        obj = self.obj
        return obj._constructor(dtype=obj.dtype, index=obj.index).__finalize__(
            obj, method="apply"
        )  # 返回一个与原对象相同数据类型和索引的 Series 对象，以 "apply" 方法作为最后的处理方法

    def apply_compat(self):
        """compat apply method for funcs in listlikes and dictlikes.

         Used for each callable when giving listlikes and dictlikes of callables to
         apply. Needed for compatibility with Pandas < v2.1.

        .. versionadded:: 2.1.0
        """
        obj = self.obj
        func = self.func

        if callable(func):
            f = com.get_cython_func(func)  # 获取 func 的 Cython 函数表示
            if f and not self.args and not self.kwargs:
                return obj.apply(func, by_row=False)  # 如果存在 Cython 函数且没有额外的参数，则直接调用 obj 的 apply 方法

        try:
            result = obj.apply(func, by_row="compat")  # 尝试使用 "compat" 模式调用 obj 的 apply 方法
        except (ValueError, AttributeError, TypeError):
            result = obj.apply(func, by_row=False)  # 处理异常情况，使用默认的 apply 方式处理
        return result
    # 将 func 强制类型转换为 Callable 类型
    func = cast(Callable, self.func)
    # 获取 self.obj 的引用
    obj = self.obj

    # 如果 func 是 numpy 的通用函数（ufunc）
    if isinstance(func, np.ufunc):
        # 忽略所有的 numpy 错误状态
        with np.errstate(all="ignore"):
            return func(obj, *self.args, **self.kwargs)
    # 如果不是按行应用函数
    elif not self.by_row:
        return func(obj, *self.args, **self.kwargs)

    # 如果有传入 args 或 kwargs
    if self.args or self.kwargs:
        # _map_values 不支持 args 或 kwargs，因此定义一个带有参数的 curried 函数
        def curried(x):
            return func(x, *self.args, **self.kwargs)

    else:
        # 如果没有传入 args 或 kwargs，则直接使用 func 作为 curried 函数
        curried = func

    # 对行进行访问
    # apply 函数没有 `na_action` 关键字参数，为了向后兼容，对于分类数据需要设定 `na_action="ignore"`
    # 当分类数据默认值改变时，应移除 `na_action="ignore"`（参见 GH51645）
    action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
    # 调用 obj 的 _map_values 方法，使用 curried 函数作为映射器，na_action 参数设定为 action
    mapped = obj._map_values(mapper=curried, na_action=action)

    # 如果 mapped 长度大于零且第一个元素是 ABCSeries 类型
    if len(mapped) and isinstance(mapped[0], ABCSeries):
        # 返回一个扩展维度的构造对象，使用 list(mapped) 作为数据，索引使用 obj 的索引
        # 这是为了支持嵌套处理，参见 GH#43986，也涉及 EA 支持的相关讨论 GH#25959
        return obj._constructor_expanddim(list(mapped), index=obj.index)
    else:
        # 否则，返回一个新构造的对象，使用 mapped 作为数据，索引使用 obj 的索引
        # 并确保返回结果使用 apply 方法
        return obj._constructor(mapped, index=obj.index).__finalize__(
            obj, method="apply"
        )
class GroupByApply(Apply):
    obj: GroupBy | Resampler | BaseWindow

    # 初始化方法，接收一个 GroupBy 对象、聚合函数、参数和关键字参数
    def __init__(
        self,
        obj: GroupBy[NDFrameT],
        func: AggFuncType,
        *,
        args,
        kwargs,
    ) -> None:
        # 复制关键字参数 kwargs
        kwargs = kwargs.copy()
        # 确定轴向，默认为 0
        self.axis = obj.obj._get_axis_number(kwargs.get("axis", 0))
        # 调用父类的初始化方法，设置初始参数
        super().__init__(
            obj,
            func,
            raw=False,
            result_type=None,
            args=args,
            kwargs=kwargs,
        )

    # 抽象方法，需要子类实现
    def apply(self):
        raise NotImplementedError

    # 抽象方法，需要子类实现
    def transform(self):
        raise NotImplementedError

    # 处理类似列表的聚合或应用操作，返回 DataFrame 或 Series 对象
    def agg_or_apply_list_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        obj = self.obj
        kwargs = self.kwargs
        # 如果是应用操作，设置 by_row 参数为 False
        if op_name == "apply":
            kwargs = {**kwargs, "by_row": False}

        # 如果对象的轴向不是 0，则抛出未实现的错误
        if getattr(obj, "axis", 0) == 1:
            raise NotImplementedError("axis other than 0 is not supported")

        # 根据对象的维度选择合适的被选对象
        if obj._selected_obj.ndim == 1:
            selected_obj = obj._selected_obj  # 对于 SeriesGroupBy，选择 _selected_obj
        else:
            selected_obj = obj._obj_with_exclusions  # 否则选择 _obj_with_exclusions

        # 只在 GroupBy 对象上设置 as_index=True，而不是在 Window 或 Resample 对象上
        with com.temp_setattr(
            obj, "as_index", True, condition=hasattr(obj, "as_index")
        ):
            # 调用 compute_list_like 方法计算结果
            keys, results = self.compute_list_like(op_name, selected_obj, kwargs)
        # 包装并返回结果
        result = self.wrap_results_list_like(keys, results)
        return result

    # 处理类似字典的聚合或应用操作，返回 DataFrame 或 Series 对象
    def agg_or_apply_dict_like(
        self, op_name: Literal["agg", "apply"]
    ) -> DataFrame | Series:
        from pandas.core.groupby.generic import (
            DataFrameGroupBy,
            SeriesGroupBy,
        )

        assert op_name in ["agg", "apply"]

        obj = self.obj
        kwargs = {}
        # 如果是应用操作，根据 self.by_row 决定设置 by_row 参数
        if op_name == "apply":
            by_row = "_compat" if self.by_row else False
            kwargs.update({"by_row": by_row})

        # 如果对象的轴向不是 0，则抛出未实现的错误
        if getattr(obj, "axis", 0) == 1:
            raise NotImplementedError("axis other than 0 is not supported")

        # 获取被选对象和选择器
        selected_obj = obj._selected_obj
        selection = obj._selection

        # 判断是否为 GroupBy 对象
        is_groupby = isinstance(obj, (DataFrameGroupBy, SeriesGroupBy))

        # Numba Groupby 引擎及其参数的传递
        if is_groupby:
            engine = self.kwargs.get("engine", None)
            engine_kwargs = self.kwargs.get("engine_kwargs", None)
            kwargs.update({"engine": engine, "engine_kwargs": engine_kwargs})

        # 只在 GroupBy 对象上设置 as_index=True
        with com.temp_setattr(
            obj, "as_index", True, condition=hasattr(obj, "as_index")
        ):
            # 调用 compute_dict_like 方法计算结果
            result_index, result_data = self.compute_dict_like(
                op_name, selected_obj, selection, kwargs
            )
        # 包装并返回结果
        result = self.wrap_results_dict_like(selected_obj, result_index, result_data)
        return result


class ResamplerWindowApply(GroupByApply):
    axis: AxisInt = 0
    obj: Resampler | BaseWindow

# 定义一个类成员变量 `obj`，可以是 `Resampler` 类型或者 `BaseWindow` 类型的对象。


    def __init__(
        self,
        obj: Resampler | BaseWindow,
        func: AggFuncType,
        *,
        args,
        kwargs,
    ) -> None:

# 构造函数 `__init__`，初始化对象实例时调用，接受参数 `obj` 作为 `Resampler` 或 `BaseWindow` 对象，`func` 作为聚合函数类型，`args` 和 `kwargs` 作为额外参数和关键字参数。


        super(GroupByApply, self).__init__(
            obj,
            func,
            raw=False,
            result_type=None,
            args=args,
            kwargs=kwargs,
        )

# 调用父类构造函数，传递 `obj`、`func`、`raw=False`、`result_type=None`、`args` 和 `kwargs` 到父类的构造函数中。


    def apply(self):

# 定义类方法 `apply()`，用于子类实现具体的应用逻辑，当前抛出 `NotImplementedError` 表示需要在子类中重写该方法。


        raise NotImplementedError

# 抛出 `NotImplementedError` 异常，提示子类必须重写此方法来实现具体的逻辑。


    def transform(self):

# 定义类方法 `transform()`，同样抛出 `NotImplementedError` 异常，用于子类中实现转换逻辑。


        raise NotImplementedError

# 抛出 `NotImplementedError` 异常，提示子类必须重写此方法来实现具体的逻辑。
# 重新构建函数，根据参数 func 和 kwargs 判断是否需要重新标记列名并对聚合函数进行规范化处理
def reconstruct_func(
    func: AggFuncType | None, **kwargs
) -> tuple[bool, AggFuncType, tuple[str, ...] | None, npt.NDArray[np.intp] | None]:
    """
    This is the internal function to reconstruct func given if there is relabeling
    or not and also normalize the keyword to get new order of columns.

    If named aggregation is applied, `func` will be None, and kwargs contains the
    column and aggregation function information to be parsed;
    If named aggregation is not applied, `func` is either string (e.g. 'min') or
    Callable, or list of them (e.g. ['min', np.max]), or the dictionary of column name
    and str/Callable/list of them (e.g. {'A': 'min'}, or {'A': [np.min, lambda x: x]})

    If relabeling is True, will return relabeling, reconstructed func, column
    names, and the reconstructed order of columns.
    If relabeling is False, the columns and order will be None.

    Parameters
    ----------
    func: agg function (e.g. 'min' or Callable) or list of agg functions
        (e.g. ['min', np.max]) or dictionary (e.g. {'A': ['min', np.max]}).
    **kwargs: dict, kwargs used in is_multi_agg_with_relabel and
        normalize_keyword_aggregation function for relabelling

    Returns
    -------
    relabelling: bool, if there is relabelling or not
    func: normalized and mangled func
    columns: tuple of column names
    order: array of columns indices

    Examples
    --------
    >>> reconstruct_func(None, **{"foo": ("col", "min")})
    (True, defaultdict(<class 'list'>, {'col': ['min']}), ('foo',), array([0]))

    >>> reconstruct_func("min")
    (False, 'min', None, None)
    """
    # 判断是否需要重新标记列名
    relabeling = func is None and is_multi_agg_with_relabel(**kwargs)
    # 初始化列名和列顺序为 None
    columns: tuple[str, ...] | None = None
    order: npt.NDArray[np.intp] | None = None

    # 如果不需要重新标记列名
    if not relabeling:
        # 如果 func 是列表且包含重复的聚合函数名称，抛出错误
        if isinstance(func, list) and len(func) > len(set(func)):
            # GH 28426 will raise error if duplicated function names are used and
            # there is no reassigned name
            raise SpecificationError(
                "Function names must be unique if there is no new column names "
                "assigned"
            )
        # 如果 func 是 None，则抛出类型错误
        if func is None:
            # nicer error message
            raise TypeError("Must provide 'func' or tuples of '(column, aggfunc).")

    # 如果需要重新标记列名
    if relabeling:
        # 调用 normalize_keyword_aggregation 函数对关键字进行规范化处理
        func, columns, order = normalize_keyword_aggregation(  # type: ignore[assignment]
            kwargs
        )
    # 确保 func 不为 None
    assert func is not None

    # 返回结果：是否重新标记列名、规范化后的 func、列名和列顺序
    return relabeling, func, columns, order


# 判断是否存在需要重新标记列名的情况
def is_multi_agg_with_relabel(**kwargs) -> bool:
    """
    # 检查传递给 .agg 方法的 kwargs 是否像多重聚合并包含重新标记。
    
    # Parameters
    # ----------
    # **kwargs : dict
    #     用于传递关键字参数的字典
    
    # Returns
    # -------
    # bool
    #     如果 kwargs 符合多重聚合并包含重新标记则返回 True，否则返回 False
    
    # Examples
    # --------
    # >>> is_multi_agg_with_relabel(a="max")
    # False
    # >>> is_multi_agg_with_relabel(a_max=("a", "max"), a_min=("a", "min"))
    # True
    # >>> is_multi_agg_with_relabel()
    # False
    """
    检查传递给 .agg 方法的关键字参数是否符合多重聚合并包含重新标记的条件。
    返回 True 表示 kwargs 中的每个值是一个长度为 2 的元组，并且 kwargs 的长度大于 0。
    否则返回 False。
    """
    return all(isinstance(v, tuple) and len(v) == 2 for v in kwargs.values()) and (
        len(kwargs) > 0
    )
# 标准化用户提供的“命名聚合”kwargs参数。
# 从新的Mapping[str, NamedAgg]风格的kwargs转换为旧的Dict[str, List[scalar]]格式。

def normalize_keyword_aggregation(
    kwargs: dict,
) -> tuple[
    MutableMapping[Hashable, list[AggFuncTypeBase]],  # 返回类型：转换后的kwargs的字典类型
    tuple[str, ...],                                # 返回类型：用户提供的键的元组
    npt.NDArray[np.intp],                           # 返回类型：列索引的数组
]:
    """
    标准化用户提供的“命名聚合”kwargs参数。
    从新的Mapping[str, NamedAgg]风格的kwargs转换为旧的Dict[str, List[scalar]]格式。

    Parameters
    ----------
    kwargs : dict
        用户提供的聚合规则参数

    Returns
    -------
    aggspec : dict
        转换后的聚合规则字典
    columns : tuple[str, ...]
        用户提供的键的元组
    col_idx_order : List[int]
        列索引的顺序列表
    """
    from pandas.core.indexes.base import Index

    # 标准化聚合函数为Mapping[column, List[func]]，
    # 正常处理，然后修正名称。
    aggspec = defaultdict(list)  # 使用默认字典存储聚合规则
    order = []                    # 存储聚合函数处理顺序
    columns = tuple(kwargs.keys())  # 获取所有键作为元组

    # 遍历kwargs的每个值，其中每个值包含列名和聚合函数
    for column, aggfunc in kwargs.values():
        aggspec[column].append(aggfunc)  # 将聚合函数添加到对应列的列表中
        order.append((column, com.get_callable_name(aggfunc) or aggfunc))  # 记录顺序及函数名

    # 如果在order列表中聚合函数名重复，则进行唯一化处理
    uniquified_order = _make_unique_kwarg_list(order)

    # GH 25719，由于aggspec会改变聚合中分配的列的顺序
    # uniquified_aggspec将存储唯一化后的顺序列表，并将其与order基于索引进行比较
    aggspec_order = [
        (column, com.get_callable_name(aggfunc) or aggfunc)
        for column, aggfuncs in aggspec.items()
        for aggfunc in aggfuncs
    ]
    uniquified_aggspec = _make_unique_kwarg_list(aggspec_order)

    # 通过比较获取列的新索引
    col_idx_order = Index(uniquified_aggspec).get_indexer(uniquified_order)
    return aggspec, columns, col_idx_order


def _make_unique_kwarg_list(
    seq: Sequence[tuple[Any, Any]],
) -> Sequence[tuple[Any, Any]]:
    """
    对order列表中的键值对进行聚合函数名称唯一化处理

    Examples
    --------
    >>> kwarg_list = [("a", "<lambda>"), ("a", "<lambda>"), ("b", "<lambda>")]
    >>> _make_unique_kwarg_list(kwarg_list)
    [('a', '<lambda>_0'), ('a', '<lambda>_1'), ('b', '<lambda>')]
    """
    return [
        (pair[0], f"{pair[1]}_{seq[:i].count(pair)}") if seq.count(pair) > 1 else pair
        for i, pair in enumerate(seq)
    ]


def relabel_result(
    result: DataFrame | Series,
    func: dict[str, list[Callable | str]],
    columns: Iterable[Hashable],
    order: Iterable[int],
) -> dict[Hashable, Series]:
    """
    如果对dataframe进行重命名，则重新排序结果的内部函数，并以字典形式返回重新排序后的结果。

    Parameters
    ----------
    result : DataFrame | Series
        聚合结果
    func : dict[str, list[Callable | str]]
        列名和函数列表的字典
    columns : Iterable[Hashable]
        用于重命名的新列名
    order : Iterable[int]
        重命名的新顺序
    """
    pass  # 函数主体未提供，暂无需注释
    # 导入必要的模块和函数
    from pandas.core.indexes.base import Index

    # 初始化用于重新排序后的结果字典
    reordered_indexes = [
        pair[0] for pair in sorted(zip(columns, order), key=lambda t: t[1])
    ]

    # 初始化一个空的字典，用于存储重新排序后的结果
    reordered_result_in_dict: dict[Hashable, Series] = {}
    
    # 初始化索引计数器
    idx = 0

    # 确定是否需要重新排序的布尔标志
    reorder_mask = not isinstance(result, ABCSeries) and len(result.columns) > 1

    # 遍历每列和对应的聚合函数列表
    for col, fun in funcs.items():
        # 从结果中获取当前列的数据，并删除空值
        s = result[col].dropna()

        # 如果需要重新排序，则根据函数名获取正确的顺序
        if reorder_mask:
            fun = [
                com.get_callable_name(f) if not isinstance(f, str) else f for f in fun
            ]
            # 获取函数在索引中的顺序
            col_idx_order = Index(s.index).get_indexer(fun)
            valid_idx = col_idx_order != -1
            if valid_idx.any():
                # 根据顺序重新排列数据
                s = s.iloc[col_idx_order[valid_idx]]

        # 如果当前列不为空，则将索引设置为重新排序后的索引列表
        if not s.empty:
            s.index = reordered_indexes[idx : idx + len(fun)]
        
        # 将重新排序后的列数据存入结果字典中
        reordered_result_in_dict[col] = s.reindex(columns)
        
        # 更新索引计数器
        idx = idx + len(fun)

    # 返回最终的重新排序结果字典
    return reordered_result_in_dict
def reconstruct_and_relabel_result(result, func, **kwargs) -> DataFrame | Series:
    # 导入 DataFrame 类
    from pandas import DataFrame

    # 调用 reconstruct_func 函数，获取重构后的信息
    relabeling, func, columns, order = reconstruct_func(func, **kwargs)

    # 检查是否需要重新标签
    if relabeling:
        # 这是为了保持列的顺序不变，并且保持新列的顺序不变

        # 如果 reconstruct_func 返回的 relabeling 是 False，那么 columns 和 order 将为 None
        assert columns is not None
        assert order is not None

        # 对结果进行重新标签处理
        result_in_dict = relabel_result(result, func, columns, order)
        # 将结果转换为 DataFrame，以 columns 作为索引
        result = DataFrame(result_in_dict, index=columns)

    return result


# TODO: 不能使用，因为 mypy 不允许我们设置 __name__
#   error: "partial[Any]" has no attribute "__name__"
# 类型是：
#   typing.Sequence[Callable[..., ScalarResult]]
#     -> typing.Sequence[Callable[..., ScalarResult]]:


def _managle_lambda_list(aggfuncs: Sequence[Any]) -> Sequence[Any]:
    """
    可能会修改 lambda 函数列表。

    Parameters
    ----------
    aggfuncs : Sequence
        聚合函数列表

    Returns
    -------
    mangled: list-like
        一个新的 AggSpec 序列，其中 lambda 函数的名称已被修改。

    Notes
    -----
    如果只传递了一个 aggfunc，名称将不会被修改。
    """
    if len(aggfuncs) <= 1:
        # 不会为 .agg([lambda x: .]) 修改名称
        return aggfuncs
    i = 0
    mangled_aggfuncs = []
    for aggfunc in aggfuncs:
        if com.get_callable_name(aggfunc) == "<lambda>":
            # 使用 partial 创建一个新的 lambda 函数，并修改其名称
            aggfunc = partial(aggfunc)
            aggfunc.__name__ = f"<lambda_{i}>"
            i += 1
        mangled_aggfuncs.append(aggfunc)

    return mangled_aggfuncs


def maybe_mangle_lambdas(agg_spec: Any) -> Any:
    """
    创建具有唯一名称的新 lambda 函数。

    Parameters
    ----------
    agg_spec : Any
        传递给 GroupBy.agg 的参数。
        对于非字典型的 agg_spec，原样返回。
        对于字典型的 agg_spec，返回一个带有名称修改的新规范。

    Returns
    -------
    mangled : Any
        与输入相同类型的对象。

    Examples
    --------
    >>> maybe_mangle_lambdas("sum")
    'sum'
    >>> maybe_mangle_lambdas([lambda: 1, lambda: 2])  # doctest: +SKIP
    [<function __main__.<lambda_0>,
     <function pandas...._make_lambda.<locals>.f(*args, **kwargs)>]
    """
    is_dict = is_dict_like(agg_spec)
    if not (is_dict or is_list_like(agg_spec)):
        return agg_spec
    mangled_aggspec = type(agg_spec)()  # dict or OrderedDict

    if is_dict:
        for key, aggfuncs in agg_spec.items():
            if is_list_like(aggfuncs) and not is_dict_like(aggfuncs):
                mangled_aggfuncs = _managle_lambda_list(aggfuncs)
            else:
                mangled_aggfuncs = aggfuncs

            mangled_aggspec[key] = mangled_aggfuncs
    else:
        mangled_aggspec = _managle_lambda_list(agg_spec)

    return mangled_aggspec
# 验证用户提供的命名聚合参数kwargs的类型是否正确。如果aggfunc不是str或callable，则会引发TypeError异常。
def validate_func_kwargs(
    kwargs: dict,
) -> tuple[list[str], list[str | Callable[..., Any]]]:
    """
    Validates types of user-provided "named aggregation" kwargs.
    `TypeError` is raised if aggfunc is not `str` or callable.

    Parameters
    ----------
    kwargs : dict
        用户提供的命名聚合参数字典

    Returns
    -------
    columns : List[str]
        用户提供的键列表
    func : List[Union[str, callable[...,Any]]]
        用户提供的聚合函数列表

    Examples
    --------
    >>> validate_func_kwargs({"one": "min", "two": "max"})
    (['one', 'two'], ['min', 'max'])
    """
    tuple_given_message = "func is expected but received {} in **kwargs."
    # 获取kwargs的键列表作为columns
    columns = list(kwargs)
    func = []
    # 遍历kwargs的值，验证每个值是str或callable类型，然后添加到func列表中
    for col_func in kwargs.values():
        if not (isinstance(col_func, str) or callable(col_func)):
            raise TypeError(tuple_given_message.format(type(col_func).__name__))
        func.append(col_func)
    if not columns:
        no_arg_message = "Must provide 'func' or named aggregation **kwargs."
        # 如果columns为空，则抛出TypeError异常
        raise TypeError(no_arg_message)
    # 返回columns和func列表作为元组
    return columns, func


# 检查操作名称和列组是否适合于给定的操作类型（agg或apply）
def include_axis(op_name: Literal["agg", "apply"], colg: Series | DataFrame) -> bool:
    return isinstance(colg, ABCDataFrame) or (
        isinstance(colg, ABCSeries) and op_name == "agg"
    )
```