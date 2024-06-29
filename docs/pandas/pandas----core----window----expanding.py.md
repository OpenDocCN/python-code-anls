# `D:\src\scipysrc\pandas\pandas\core\window\expanding.py`

```
from __future__ import annotations

from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

from pandas.util._decorators import doc

from pandas.core.indexers.objects import (
    BaseIndexer,
    ExpandingIndexer,
    GroupbyIndexer,
)
from pandas.core.window.doc import (
    _shared_docs,
    create_section_header,
    kwargs_numeric_only,
    numba_notes,
    template_header,
    template_returns,
    template_see_also,
    window_agg_numba_parameters,
    window_apply_parameters,
)
from pandas.core.window.rolling import (
    BaseWindowGroupby,
    RollingAndExpandingMixin,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas._typing import (
        QuantileInterpolation,
        WindowingRankType,
    )

    from pandas import (
        DataFrame,
        Series,
    )
    from pandas.core.generic import NDFrame


class Expanding(RollingAndExpandingMixin):
    """
    Provide expanding window calculations.

    An expanding window yields the value of an aggregation statistic with all the data
    available up to that point in time.

    Parameters
    ----------
    min_periods : int, default 1
        Minimum number of observations in window required to have a value;
        otherwise, result is ``np.nan``.

    method : str {'single', 'table'}, default 'single'
        Execute the rolling operation per single column or row (``'single'``)
        or over the entire object (``'table'``).

        This argument is only implemented when specifying ``engine='numba'``
        in the method call.

        .. versionadded:: 1.3.0

    Returns
    -------
    pandas.api.typing.Expanding
        An instance of Expanding for further expanding window calculations,
        e.g. using the ``sum`` method.

    See Also
    --------
    rolling : Provides rolling window calculations.
    ewm : Provides exponential weighted functions.

    Notes
    -----
    See :ref:`Windowing Operations <window.expanding>` for further usage details
    and examples.

    Examples
    --------
    >>> df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    >>> df
         B
    0  0.0
    1  1.0
    2  2.0
    3  NaN
    4  4.0

    **min_periods**

    Expanding sum with 1 vs 3 observations needed to calculate a value.

    >>> df.expanding(1).sum()
         B
    0  0.0
    1  1.0
    2  3.0
    3  3.0
    4  7.0
    >>> df.expanding(3).sum()
         B
    0  NaN
    1  NaN
    2  3.0
    3  3.0
    4  7.0
    """

    _attributes: list[str] = ["min_periods", "method"]

    def __init__(
        self,
        obj: NDFrame,
        min_periods: int = 1,
        method: str = "single",
        selection=None,
    ) -> None:
        """
        Initialize an Expanding window object.

        Parameters
        ----------
        obj : pandas.core.generic.NDFrame
            The input data structure over which to calculate the expanding window.
        min_periods : int, default 1
            The minimum number of observations in window required to have a valid result.
        method : str, {'single', 'table'}, default 'single'
            Determines how the operation is applied:
            - 'single': Execute the operation per column or row.
            - 'table': Execute the operation over the entire object.
              Only applicable when using 'numba' engine in method calls.
        selection : Any, optional
            Additional selection criteria for the window operation.

        Returns
        -------
        None
        """
        super().__init__(
            obj=obj,
            min_periods=min_periods,
            method=method,
            selection=selection,
        )
    @doc(
        _shared_docs["aggregate"],  # 使用_shared_docs字典中的"aggregate"文档作为文档字符串
        see_also=dedent(  # 嵌套的dedent函数，展开下面的多行字符串，用于提供"See Also"部分
            """
        See Also
        --------
        DataFrame.aggregate : Similar DataFrame method.
        Series.aggregate : Similar Series method.
        """
        ),
        examples=dedent(  # 嵌套的dedent函数，展开下面的多行字符串，提供示例用法
            """
        Examples
        --------
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        >>> df
           A  B  C
        0  1  4  7
        1  2  5  8
        2  3  6  9

        >>> df.ewm(alpha=0.5).mean()
                  A         B         C
        0  1.000000  4.000000  7.000000
        1  1.666667  4.666667  7.666667
        2  2.428571  5.428571  8.428571
        """
        ),
        klass="Series/Dataframe",  # 指定方法适用于Series和DataFrame类
        axis="",  # axis参数为空字符串，未指定特定轴向
    )
    def aggregate(self, func, *args, **kwargs):
        return super().aggregate(func, *args, **kwargs)  # 调用父类的aggregate方法并返回结果

    agg = aggregate  # 创建agg别名指向aggregate方法

    @doc(
        template_header,  # 使用template_header作为文档字符串的模板头部
        create_section_header("Returns"),  # 创建"Returns"小节的头部
        template_returns,  # 使用template_returns作为返回值说明的模板
        create_section_header("See Also"),  # 创建"See Also"小节的头部
        template_see_also,  # 使用template_see_also作为"See Also"部分的模板
        create_section_header("Examples"),  # 创建"Examples"小节的头部
        dedent(  # 嵌套的dedent函数，展开下面的多行字符串，提供示例用法
            """\
        >>> ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        >>> ser.expanding().count()
        a    1.0
        b    2.0
        c    3.0
        d    4.0
        dtype: float64
        """
        ),
        window_method="expanding",  # 指定窗口方法为"expanding"
        aggregation_description="count of non NaN observations",  # 聚合描述为"非NaN观测值的计数"
        agg_method="count",  # 指定聚合方法为"count"
    )
    def count(self, numeric_only: bool = False):
        return super().count(numeric_only=numeric_only)  # 调用父类的count方法并返回结果

    @doc(
        template_header,  # 使用template_header作为文档字符串的模板头部
        create_section_header("Parameters"),  # 创建"Parameters"小节的头部
        window_apply_parameters,  # 使用window_apply_parameters作为窗口应用参数的模板
        create_section_header("Returns"),  # 创建"Returns"小节的头部
        template_returns,  # 使用template_returns作为返回值说明的模板
        create_section_header("See Also"),  # 创建"See Also"小节的头部
        template_see_also,  # 使用template_see_also作为"See Also"部分的模板
        create_section_header("Examples"),  # 创建"Examples"小节的头部
        dedent(  # 嵌套的dedent函数，展开下面的多行字符串，提供示例用法
            """\
        >>> ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        >>> ser.expanding().apply(lambda s: s.max() - 2 * s.min())
        a   -1.0
        b    0.0
        c    1.0
        d    2.0
        dtype: float64
        """
        ),
        window_method="expanding",  # 指定窗口方法为"expanding"
        aggregation_description="custom aggregation function",  # 聚合描述为"自定义聚合函数"
        agg_method="apply",  # 指定聚合方法为"apply"
    )
    def apply(
        self,
        func: Callable[..., Any],  # 接受一个可调用对象作为参数func
        raw: bool = False,  # raw参数，默认为False
        engine: Literal["cython", "numba"] | None = None,  # engine参数，接受特定的字符串或None
        engine_kwargs: dict[str, bool] | None = None,  # engine_kwargs参数，接受字典类型或None
        args: tuple[Any, ...] | None = None,  # args参数，接受任意类型的元组或None
        kwargs: dict[str, Any] | None = None,  # kwargs参数，接受字典类型或None
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
            """\
        >>> ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        >>> ser.expanding().sum()
        a     1.0
        b     3.0
        c     6.0
        d    10.0
        dtype: float64
        """
        ),
        window_method="expanding",
        aggregation_description="sum",
        agg_method="sum",
    )

该装饰器为 `sum` 方法添加文档字符串，详细描述了参数、返回值、相关性、注意事项和示例。包括模板标题、参数部分、数值参数、Numba窗口聚合参数、返回值部分、返回值模板、相关性部分、相关性模板、注意事项部分、Numba注意事项、示例部分和示例。

##
    # 使用装饰器 @doc 注释方法，提供文档字符串和参数说明
    @doc(
        template_header,  # 使用预定义的模板标题
        create_section_header("Parameters"),  # 创建参数部分的标题
        kwargs_numeric_only,  # 导入数字参数的关键字参数说明
        window_agg_numba_parameters(),  # 使用窗口聚合和 Numba 的参数说明
        create_section_header("Returns"),  # 创建返回值部分的标题
        template_returns,  # 使用预定义的返回值模板
        create_section_header("See Also"),  # 创建相关内容部分的标题
        template_see_also,  # 使用预定义的相关内容模板
        create_section_header("Notes"),  # 创建注释部分的标题
        numba_notes,  # 包含有关 Numba 的注释
        create_section_header("Examples"),  # 创建示例部分的标题
        dedent(  # 移除示例字符串的缩进
            """\
        >>> ser = pd.Series([2, 3, 4, 1], index=['a', 'b', 'c', 'd'])
        >>> ser.expanding().min()
        a    2.0
        b    2.0
        c    2.0
        d    1.0
        dtype: float64
        """
        ),
        window_method="expanding",  # 指定窗口方法为 expanding
        aggregation_description="minimum",  # 指定聚合描述为最小值
        agg_method="min",  # 指定聚合方法为最小化
    )
    def min(
        self,
        numeric_only: bool = False,  # 是否仅对数字进行操作，默认为 False
        engine: Literal["cython", "numba"] | None = None,  # 引擎选择，可以是 "cython" 或 "numba" 或 None
        engine_kwargs: dict[str, bool] | None = None,  # 引擎的额外参数字典，键为字符串，值为布尔值或 None
    ):
        # 调用父类的最小化方法，传递相应的参数
        return super().min(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    # 类似上述方法，但是聚合描述为平均值
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
        >>> ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        >>> ser.expanding().mean()
        a    1.0
        b    1.5
        c    2.0
        d    2.5
        dtype: float64
        """
        ),
        window_method="expanding",
        aggregation_description="mean",  # 聚合描述为平均值
        agg_method="mean",  # 聚合方法为均值
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

    # 类似上述方法，但是聚合描述为中位数
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
        >>> ser = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        >>> ser.expanding().median()
        a    1.0
        b    1.5
        c    2.0
        d    2.5
        dtype: float64
        """
        ),
        window_method="expanding",
        aggregation_description="median",  # 聚合描述为中位数
        agg_method="median",  # 聚合方法为中位数
    )
    # 使用 super() 调用父类的 median 方法，传递给父类的参数包括 numeric_only, engine 和 engine_kwargs
    def median(
        self,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # 返回父类的 median 方法的结果
        return super().median(
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    # 使用 @doc 装饰器为 std 方法添加文档
    @doc(
        template_header,
        create_section_header("Parameters"),
        # 定义 ddof 参数的含义和默认值
        dedent(
            """
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.\n
        """
        ).replace("\n", "", 1),
        kwargs_numeric_only,
        # 使用 numba 引擎的窗口聚合参数
        window_agg_numba_parameters("1.4"),
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        # 提示查看 numpy.std 方法的等效用法
        "numpy.std : Equivalent method for NumPy array.\n",
        template_see_also,
        create_section_header("Notes"),
        # 给出 std 方法的特殊说明，包括 ddof 参数的不同之处和滚动计算的最小期数要求
        dedent(
            """
        The default ``ddof`` of 1 used in :meth:`Series.std` is different
        than the default ``ddof`` of 0 in :func:`numpy.std`.

        A minimum of one period is required for the rolling calculation.\n
        """
        ).replace("\n", "", 1),
        create_section_header("Examples"),
        # 给出 std 方法的使用示例
        dedent(
            """
        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])

        >>> s.expanding(3).std()
        0         NaN
        1         NaN
        2    0.577350
        3    0.957427
        4    0.894427
        5    0.836660
        6    0.786796
        dtype: float64
        """
        ).replace("\n", "", 1),
        # 指定窗口方法为 expanding，聚合描述为标准差
        window_method="expanding",
        aggregation_description="standard deviation",
        agg_method="std",
    )
    # 定义 std 方法，接受 ddof, numeric_only, engine 和 engine_kwargs 参数
    def std(
        self,
        ddof: int = 1,
        numeric_only: bool = False,
        engine: Literal["cython", "numba"] | None = None,
        engine_kwargs: dict[str, bool] | None = None,
    ):
        # 调用父类的 std 方法，传递给父类的参数包括 ddof, numeric_only, engine 和 engine_kwargs
        return super().std(
            ddof=ddof,
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )
    # 使用 @doc 装饰器为 var 方法添加文档字符串，指定参数和返回值说明
    @doc(
        template_header,  # 导入的模板头部
        create_section_header("Parameters"),  # 创建参数部分的标题
        dedent(
            """
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.\n
        """
        ).replace("\n", "", 1),  # 参数 ddof 的说明，表示自由度调整
        kwargs_numeric_only,  # 仅限数字参数的说明
        window_agg_numba_parameters("1.4"),  # 使用 numba 引擎的窗口聚合参数
        create_section_header("Returns"),  # 创建返回值部分的标题
        template_returns,  # 导入的返回值模板
        create_section_header("See Also"),  # 创建相关内容部分的标题
        "numpy.var : Equivalent method for NumPy array.\n",  # 参考 NumPy 的等效方法
        template_see_also,  # 导入的参考部分模板
        create_section_header("Notes"),  # 创建注释部分的标题
        dedent(
            """
        The default ``ddof`` of 1 used in :meth:`Series.var` is different
        than the default ``ddof`` of 0 in :func:`numpy.var`.

        A minimum of one period is required for the rolling calculation.\n
        """
        ).replace("\n", "", 1),  # 一些关于方法的额外注释
        create_section_header("Examples"),  # 创建示例部分的标题
        dedent(
            """
        >>> s = pd.Series([5, 5, 6, 7, 5, 5, 5])

        >>> s.expanding(3).var()
        0         NaN
        1         NaN
        2    0.333333
        3    0.916667
        4    0.800000
        5    0.700000
        6    0.619048
        dtype: float64
        """
        ).replace("\n", "", 1),  # 方法的示例用法
        window_method="expanding",  # 扩展窗口方法
        aggregation_description="variance",  # 聚合描述为方差
        agg_method="var",  # 使用方差作为聚合方法
    )
    # 定义 var 方法，传递参数并调用父类的 var 方法进行计算
    def var(
        self,
        ddof: int = 1,  # 自由度调整，默认为 1
        numeric_only: bool = False,  # 是否仅限数字
        engine: Literal["cython", "numba"] | None = None,  # 引擎选择，可以是 cython 或 numba
        engine_kwargs: dict[str, bool] | None = None,  # 引擎参数
    ):
        return super().var(  # 调用父类的 var 方法进行实际计算
            ddof=ddof,
            numeric_only=numeric_only,
            engine=engine,
            engine_kwargs=engine_kwargs,
        )

    # 使用 @doc 装饰器为 sem 方法添加文档字符串，指定参数和返回值说明
    @doc(
        template_header,  # 导入的模板头部
        create_section_header("Parameters"),  # 创建参数部分的标题
        dedent(
            """
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.\n
        """
        ).replace("\n", "", 1),  # 参数 ddof 的说明，表示自由度调整
        kwargs_numeric_only,  # 仅限数字参数的说明
        create_section_header("Returns"),  # 创建返回值部分的标题
        template_returns,  # 导入的返回值模板
        create_section_header("See Also"),  # 创建相关内容部分的标题
        template_see_also,  # 导入的参考部分模板
        create_section_header("Notes"),  # 创建注释部分的标题
        "A minimum of one period is required for the calculation.\n\n",  # 一些关于方法的额外注释
        create_section_header("Examples"),  # 创建示例部分的标题
        dedent(
            """
        >>> s = pd.Series([0, 1, 2, 3])

        >>> s.expanding().sem()
        0         NaN
        1    0.707107
        2    0.707107
        3    0.745356
        dtype: float64
        """
        ).replace("\n", "", 1),  # 方法的示例用法
        window_method="expanding",  # 扩展窗口方法
        aggregation_description="standard error of mean",  # 聚合描述为均值的标准误差
        agg_method="sem",  # 使用 sem 作为聚合方法
    )
    # 定义 sem 方法，传递参数并调用父类的 sem 方法进行计算
    def sem(self, ddof: int = 1, numeric_only: bool = False):
        return super().sem(ddof=ddof, numeric_only=numeric_only)
    # 使用 @doc 装饰器为 skew 方法添加文档注释
    @doc(
        # 使用 template_header 作为模板头部
        template_header,
        # 创建 "Parameters" 部分的节标题
        create_section_header("Parameters"),
        # 使用 kwargs_numeric_only 作为参数描述
        kwargs_numeric_only,
        # 创建 "Returns" 部分的节标题
        create_section_header("Returns"),
        # 使用 template_returns 作为返回值描述模板
        template_returns,
        # 创建 "See Also" 部分的节标题
        create_section_header("See Also"),
        # 引用 scipy.stats.skew 的相关信息
        "scipy.stats.skew : Third moment of a probability density.\n",
        # 使用 template_see_also 作为参考文献的模板
        template_see_also,
        # 创建 "Notes" 部分的节标题
        create_section_header("Notes"),
        # 指出需要至少三个周期进行滚动计算的要求
        "A minimum of three periods is required for the rolling calculation.\n\n",
        # 创建 "Examples" 部分的节标题
        create_section_header("Examples"),
        # 缩进示例代码，展示如何使用 expanding 方法计算 skew
        dedent(
            """\
        >>> ser = pd.Series([-1, 0, 2, -1, 2], index=['a', 'b', 'c', 'd', 'e'])
        >>> ser.expanding().skew()
        a         NaN
        b         NaN
        c    0.935220
        d    1.414214
        e    0.315356
        dtype: float64
        """
        ),
        # 指定使用 expanding 方法进行窗口操作
        window_method="expanding",
        # 指定为无偏 skewness 的聚合描述
        aggregation_description="unbiased skewness",
        # 指定使用 skew 方法进行聚合
        agg_method="skew",
    )
    # 定义 skew 方法，调用父类的 skew 方法，并返回结果
    def skew(self, numeric_only: bool = False):
        return super().skew(numeric_only=numeric_only)
    
    
    # 使用 @doc 装饰器为 kurt 方法添加文档注释
    @doc(
        # 使用 template_header 作为模板头部
        template_header,
        # 创建 "Parameters" 部分的节标题
        create_section_header("Parameters"),
        # 使用 kwargs_numeric_only 作为参数描述
        kwargs_numeric_only,
        # 创建 "Returns" 部分的节标题
        create_section_header("Returns"),
        # 使用 template_returns 作为返回值描述模板
        template_returns,
        # 创建 "See Also" 部分的节标题
        create_section_header("See Also"),
        # 引用 scipy.stats.kurtosis 的相关信息
        "scipy.stats.kurtosis : Reference SciPy method.\n",
        # 使用 template_see_also 作为参考文献的模板
        template_see_also,
        # 创建 "Notes" 部分的节标题
        create_section_header("Notes"),
        # 指出需要至少四个周期进行计算的要求
        "A minimum of four periods is required for the calculation.\n\n",
        # 创建 "Examples" 部分的节标题
        create_section_header("Examples"),
        # 缩进示例代码，展示如何使用 expanding 方法计算 kurt
        dedent(
            """
        The example below will show a rolling calculation with a window size of
        four matching the equivalent function call using `scipy.stats`.
    
        >>> arr = [1, 2, 3, 4, 999]
        >>> import scipy.stats
        >>> print(f"{{scipy.stats.kurtosis(arr[:-1], bias=False):.6f}}")
        -1.200000
        >>> print(f"{{scipy.stats.kurtosis(arr, bias=False):.6f}}")
        4.999874
        >>> s = pd.Series(arr)
        >>> s.expanding(4).kurt()
        0         NaN
        1         NaN
        2         NaN
        3   -1.200000
        4    4.999874
        dtype: float64
        """
        ).replace("\n", "", 1),
        # 指定使用 expanding 方法进行窗口操作
        window_method="expanding",
        # 指定为 Fisher's definition of kurtosis without bias 的聚合描述
        aggregation_description="Fisher's definition of kurtosis without bias",
        # 指定使用 kurt 方法进行聚合
        agg_method="kurt",
    )
    # 定义 kurt 方法，调用父类的 kurt 方法，并返回结果
    def kurt(self, numeric_only: bool = False):
        return super().kurt(numeric_only=numeric_only)
    @doc(
        template_header,  # 使用给定的模板头部信息
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
        ).replace("\n", "", 1),  # 格式化参数部分的文档内容并替换第一个换行符
        kwargs_numeric_only,  # 使用仅限数值的关键字参数
        create_section_header("Returns"),  # 创建返回值部分的标题
        template_returns,  # 使用返回值的模板信息
        create_section_header("See Also"),  # 创建相关内容部分的标题
        template_see_also,  # 使用相关内容的模板信息
        create_section_header("Examples"),  # 创建示例部分的标题
        dedent(
            """\
        >>> ser = pd.Series([1, 2, 3, 4, 5, 6], index=['a', 'b', 'c', 'd', 'e', 'f'])
        >>> ser.expanding(min_periods=4).quantile(.25)
        a     NaN
        b     NaN
        c     NaN
        d    1.75
        e    2.00
        f    2.25
        dtype: float64
        """
        ),  # 格式化示例内容
        window_method="expanding",  # 使用扩展窗口方法
        aggregation_description="quantile",  # 聚合描述为quantile
        agg_method="quantile",  # 使用quantile的聚合方法
    )
    def quantile(
        self,
        q: float,  # 传入的参数q，表示要计算的分位数
        interpolation: QuantileInterpolation = "linear",  # 插值方法，默认为线性
        numeric_only: bool = False,  # 是否仅限数值，默认为False
    ):
        return super().quantile(
            q=q,  # 调用父类的quantile方法，传入参数q
            interpolation=interpolation,  # 传入插值方法
            numeric_only=numeric_only,  # 传入是否仅限数值的参数
        )
    # 使用 @doc 装饰器为 rank 方法添加文档说明和元数据
    @doc(
        # 添加模板头信息
        template_header,
        # 指明版本添加信息
        ".. versionadded:: 1.4.0 \n\n",
        # 创建参数部分的标题
        create_section_header("Parameters"),
        # 添加参数详细描述，包括方法(method)、升序(ascending)、百分位(pct)
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
        # 添加仅限于数字的关键字参数描述
        kwargs_numeric_only,
        # 创建返回值部分的标题
        create_section_header("Returns"),
        # 添加返回值模板
        template_returns,
        # 创建 "See Also" 部分的标题
        create_section_header("See Also"),
        # 添加 "See Also" 模板
        template_see_also,
        # 创建示例部分的标题
        create_section_header("Examples"),
        # 添加示例，展示了不同方法的使用情况
        dedent(
            """
        >>> s = pd.Series([1, 4, 2, 3, 5, 3])
        >>> s.expanding().rank()
        0    1.0
        1    2.0
        2    2.0
        3    3.0
        4    5.0
        5    3.5
        dtype: float64
    
        >>> s.expanding().rank(method="max")
        0    1.0
        1    2.0
        2    2.0
        3    3.0
        4    5.0
        5    4.0
        dtype: float64
    
        >>> s.expanding().rank(method="min")
        0    1.0
        1    2.0
        2    2.0
        3    3.0
        4    5.0
        5    3.0
        dtype: float64
        """
        ).replace("\n", "", 1),
        # 指定窗口方法为 expanding
        window_method="expanding",
        # 指定聚合描述为 rank
        aggregation_description="rank",
        # 指定聚合方法为 rank
        agg_method="rank",
    )
    # 定义 rank 方法，接受 method、ascending、pct 和 numeric_only 参数
    def rank(
        self,
        method: WindowingRankType = "average",
        ascending: bool = True,
        pct: bool = False,
        numeric_only: bool = False,
    ):
        # 调用父类的 rank 方法，传递参数并返回结果
        return super().rank(
            method=method,
            ascending=ascending,
            pct=pct,
            numeric_only=numeric_only,
        )
    @doc(
        template_header,  # 使用预定义的文档模板的头部信息
        create_section_header("Parameters"),  # 创建参数部分的标题
        dedent(
            """
        other : Series or DataFrame, optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndexed DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of elements.
        """
        ).replace("\n", "", 1),  # 格式化参数部分的文本，替换第一个换行符为空字符串
        kwargs_numeric_only,  # 使用预定义的 kwargs_numeric_only 参数
        create_section_header("Returns"),  # 创建返回部分的标题
        template_returns,  # 使用预定义的返回值模板
        create_section_header("See Also"),  # 创建"See Also"部分的标题
        template_see_also,  # 使用预定义的"See Also"内容
        create_section_header("Examples"),  # 创建示例部分的标题
        dedent(
            """\
        >>> ser1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
        >>> ser2 = pd.Series([10, 11, 13, 16], index=['a', 'b', 'c', 'd'])
        >>> ser1.expanding().cov(ser2)
        a         NaN
        b    0.500000
        c    1.500000
        d    3.333333
        dtype: float64
        """
        ),  # 提供一个示例，展示如何使用该方法
        window_method="expanding",  # 设置窗口方法为"expanding"
        aggregation_description="sample covariance",  # 聚合描述为"sample covariance"
        agg_method="cov",  # 使用协方差作为聚合方法
    )
    def cov(
        self,
        other: DataFrame | Series | None = None,  # 参数：其他数据系列或数据帧，可选
        pairwise: bool | None = None,  # 参数：是否计算所有配对组合，布尔型，默认为None
        ddof: int = 1,  # 参数：Delta自由度，整数，默认为1
        numeric_only: bool = False,  # 参数：是否仅考虑数值型数据，布尔型，默认为False
    ):
        return super().cov(
            other=other,  # 调用父类的cov方法，传递其他数据参数
            pairwise=pairwise,  # 传递pairwise参数
            ddof=ddof,  # 传递ddof参数
            numeric_only=numeric_only,  # 传递numeric_only参数
        )
    # 使用 @doc 装饰器为 corr 方法添加文档字符串，指定参数模板头部
    @doc(
        template_header,
        # 创建参数部分的标题
        create_section_header("Parameters"),
        # 去除缩进并格式化参数描述文本，描述 other 参数为 Series 或 DataFrame，可选
        dedent(
            """
            other : Series or DataFrame, optional
                If not supplied then will default to self and produce pairwise
                output.
            pairwise : bool, default None
                If False then only matching columns between self and other will be
                used and the output will be a DataFrame.
                If True then all pairwise combinations will be calculated and the
                output will be a MultiIndexed DataFrame in the case of DataFrame
                inputs. In the case of missing elements, only complete pairwise
                observations will be used.
            """
        ).replace("\n", "", 1),
        # 使用 kwargs_numeric_only 指定数字参数的模板
        kwargs_numeric_only,
        # 创建返回值部分的标题
        create_section_header("Returns"),
        # 指定返回值的模板
        template_returns,
        # 创建 "See Also" 部分的标题
        create_section_header("See Also"),
        # 去除缩进并格式化相关方法的注释，提供 cov 方法和 numpy.corrcoef 的链接
        dedent(
            """
            cov : Similar method to calculate covariance.
            numpy.corrcoef : NumPy Pearson's correlation calculation.
            """
        ).replace("\n", "", 1),
        # 使用 template_see_also 指定相关链接部分的模板
        template_see_also,
        # 创建 "Notes" 部分的标题
        create_section_header("Notes"),
        # 去除缩进并格式化详细的函数说明文本，描述了 Pearson 相关系数的使用和返回结果的一些情况
        dedent(
            """
            This function uses Pearson's definition of correlation
            (https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).

            When `other` is not specified, the output will be self correlation (e.g.
            all 1's), except for :class:`~pandas.DataFrame` inputs with `pairwise`
            set to `True`.

            Function will return ``NaN`` for correlations of equal valued sequences;
            this is the result of a 0/0 division error.

            When `pairwise` is set to `False`, only matching columns between `self` and
            `other` will be used.

            When `pairwise` is set to `True`, the output will be a MultiIndex DataFrame
            with the original index on the first level, and the `other` DataFrame
            columns on the second level.

            In the case of missing elements, only complete pairwise observations
            will be used.\n
            """
        ),
        # 创建 "Examples" 部分的标题
        create_section_header("Examples"),
        # 去除缩进并格式化示例代码，展示了如何使用 corr 方法计算扩展相关性
        dedent(
            """\
            >>> ser1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
            >>> ser2 = pd.Series([10, 11, 13, 16], index=['a', 'b', 'c', 'd'])
            >>> ser1.expanding().corr(ser2)
            a         NaN
            b    1.000000
            c    0.981981
            d    0.975900
            dtype: float64
            """
        ),
        # 指定窗口方法为 "expanding"
        window_method="expanding",
        # 描述聚合方法为 "correlation"
        aggregation_description="correlation",
        # 指定聚合方法为 "corr"
        agg_method="corr",
    )
    # 定义 corr 方法，接受 self, other, pairwise, ddof, numeric_only 参数，并调用其父类的 corr 方法
    def corr(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        ddof: int = 1,
        numeric_only: bool = False,
    ):
        # 调用父类的 corr 方法，传入相应的参数
        return super().corr(
            other=other,
            pairwise=pairwise,
            ddof=ddof,
            numeric_only=numeric_only,
        )
# 定义一个继承自 BaseWindowGroupby 和 Expanding 的类 ExpandingGroupby
class ExpandingGroupby(BaseWindowGroupby, Expanding):
    """
    提供一个扩展的分组操作实现。
    """

    # 合并 BaseWindowGroupby 和 Expanding 类的属性到 _attributes 中
    _attributes = Expanding._attributes + BaseWindowGroupby._attributes

    # 返回一个索引器类，用于计算窗口的起始和结束边界
    def _get_window_indexer(self) -> GroupbyIndexer:
        """
        返回一个索引器类，该类将计算窗口的起始和结束边界

        Returns
        -------
        GroupbyIndexer
        """
        # 创建一个 GroupbyIndexer 对象，传入分组索引和 ExpandingIndexer 类作为窗口索引器
        window_indexer = GroupbyIndexer(
            groupby_indices=self._grouper.indices,
            window_indexer=ExpandingIndexer,
        )
        # 返回创建的窗口索引器对象
        return window_indexer
```