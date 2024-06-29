# `D:\src\scipysrc\pandas\pandas\core\window\ewm.py`

```
# 导入所需模块和类型
from __future__ import annotations

import datetime  # 导入日期时间模块
from functools import partial  # 导入partial函数
from textwrap import dedent  # 导入dedent函数用于移除多行字符串的公共前缀
from typing import TYPE_CHECKING  # 导入类型检查器

import numpy as np  # 导入NumPy库

from pandas._libs.tslibs import Timedelta  # 导入Timedelta类型
import pandas._libs.window.aggregations as window_aggregations  # 导入窗口聚合函数
from pandas.util._decorators import doc  # 导入文档装饰器

from pandas.core.dtypes.common import (  # 导入数据类型判断函数
    is_datetime64_dtype,
    is_numeric_dtype,
)
from pandas.core.dtypes.dtypes import DatetimeTZDtype  # 导入日期时间类型
from pandas.core.dtypes.generic import ABCSeries  # 导入通用的Series类型
from pandas.core.dtypes.missing import isna  # 导入isna函数用于检查缺失值

from pandas.core import common  # 导入通用函数
from pandas.core.arrays.datetimelike import dtype_to_unit  # 导入日期时间单位转换函数
from pandas.core.indexers.objects import (  # 导入索引器类和函数
    BaseIndexer,
    ExponentialMovingWindowIndexer,
    GroupbyIndexer,
)
from pandas.core.util.numba_ import (  # 导入Numba相关函数
    get_jit_arguments,
    maybe_use_numba,
)
from pandas.core.window.common import zsqrt  # 导入常用窗口函数
from pandas.core.window.doc import (  # 导入窗口函数文档相关函数和字符串模板
    _shared_docs,
    create_section_header,
    kwargs_numeric_only,
    numba_notes,
    template_header,
    template_returns,
    template_see_also,
    window_agg_numba_parameters,
)
from pandas.core.window.numba_ import (  # 导入Numba窗口函数相关函数
    generate_numba_ewm_func,
    generate_numba_ewm_table_func,
)
from pandas.core.window.online import (  # 导入在线窗口函数相关类和函数
    EWMMeanState,
    generate_online_numba_ewma_func,
)
from pandas.core.window.rolling import (  # 导入滚动窗口类
    BaseWindow,
    BaseWindowGroupby,
)

if TYPE_CHECKING:
    from pandas._typing import (  # 导入类型定义
        TimedeltaConvertibleTypes,
        npt,
    )

    from pandas import (  # 导入DataFrame和Series类
        DataFrame,
        Series,
    )
    from pandas.core.generic import NDFrame  # 导入通用数据框架类


def get_center_of_mass(  # 定义函数用于计算质心
    comass: float | None,  # comass参数，表示质心
    span: float | None,  # span参数，表示跨度
    halflife: float | None,  # halflife参数，表示半衰期
    alpha: float | None,  # alpha参数，表示衰减因子
) -> float:  # 返回浮点数作为质心
    valid_count = common.count_not_none(comass, span, halflife, alpha)  # 计算非空参数个数
    if valid_count > 1:  # 如果多于一个参数不为空，则抛出值错误
        raise ValueError("comass, span, halflife, and alpha are mutually exclusive")

    # 根据传入的参数计算质心
    if comass is not None:
        if comass < 0:  # 如果质心小于0，则抛出值错误
            raise ValueError("comass must satisfy: comass >= 0")
    elif span is not None:
        if span < 1:  # 如果跨度小于1，则抛出值错误
            raise ValueError("span must satisfy: span >= 1")
        comass = (span - 1) / 2  # 计算跨度为参数span时的质心
    elif halflife is not None:
        if halflife <= 0:  # 如果半衰期小于等于0，则抛出值错误
            raise ValueError("halflife must satisfy: halflife > 0")
        decay = 1 - np.exp(np.log(0.5) / halflife)  # 计算半衰期为参数halflife时的衰减
        comass = 1 / decay - 1  # 计算半衰期为参数halflife时的质心
    elif alpha is not None:
        if alpha <= 0 or alpha > 1:  # 如果衰减因子不在(0, 1]范围内，则抛出值错误
            raise ValueError("alpha must satisfy: 0 < alpha <= 1")
        comass = (1 - alpha) / alpha  # 计算衰减因子为参数alpha时的质心
    else:
        raise ValueError("Must pass one of comass, span, halflife, or alpha")  # 如果没有传入任何参数，则抛出值错误

    return float(comass)  # 返回计算得到的质心作为浮点数


def _calculate_deltas(  # 定义函数用于计算时间差的一半
    times: np.ndarray | NDFrame,  # times参数，表示时间数组或数据框架
    halflife: float | TimedeltaConvertibleTypes | None,  # halflife参数，表示半衰期或时间增量类型
) -> npt.NDArray[np.float64]:  # 返回NumPy数组，数据类型为64位浮点数
    """
    Return the diff of the times divided by the half-life. These values are used in
    """
    the calculation of the ewm mean.

    Parameters
    ----------
    times : np.ndarray, Series
        Times corresponding to the observations. Must be monotonically increasing
        and ``datetime64[ns]`` dtype.
    halflife : float, str, timedelta, optional
        Half-life specifying the decay

    Returns
    -------
    np.ndarray
        Diff of the times divided by the half-life
    """
    # 根据时间数据类型获取单位
    unit = dtype_to_unit(times.dtype)
    # 如果输入的时间是 pandas Series 类型，则获取其内部的 numpy 数组
    if isinstance(times, ABCSeries):
        times = times._values
    # 将时间转换为 int64 类型，并转换为 float64 类型
    _times = np.asarray(times.view(np.int64), dtype=np.float64)
    # 将半衰期转换为指定单位的浮点数值
    _halflife = float(Timedelta(halflife).as_unit(unit)._value)
    # 计算时间差分除以半衰期
    return np.diff(_times) / _halflife
class ExponentialMovingWindow(BaseWindow):
    r"""
    Provide exponentially weighted (EW) calculations.

    Exactly one of ``com``, ``span``, ``halflife``, or ``alpha`` must be
    provided if ``times`` is not provided. If ``times`` is provided,
    ``halflife`` and one of ``com``, ``span`` or ``alpha`` may be provided.

    Parameters
    ----------
    com : float, optional
        Specify decay in terms of center of mass

        :math:`\alpha = 1 / (1 + com)`, for :math:`com \geq 0`.

    span : float, optional
        Specify decay in terms of span

        :math:`\alpha = 2 / (span + 1)`, for :math:`span \geq 1`.

    halflife : float, str, timedelta, optional
        Specify decay in terms of half-life

        :math:`\alpha = 1 - \exp\left(-\ln(2) / halflife\right)`, for
        :math:`halflife > 0`.

        If ``times`` is specified, a timedelta convertible unit over which an
        observation decays to half its value. Only applicable to ``mean()``,
        and halflife value will not apply to the other functions.

    alpha : float, optional
        Specify smoothing factor :math:`\alpha` directly

        :math:`0 < \alpha \leq 1`.

    min_periods : int, default 0
        Minimum number of observations in window required to have a value;
        otherwise, result is ``np.nan``.

    adjust : bool, default True
        Divide by decaying adjustment factor in beginning periods to account
        for imbalance in relative weightings (viewing EWMA as a moving average).

        - When ``adjust=True`` (default), the EW function is calculated using weights
          :math:`w_i = (1 - \alpha)^i`. For example, the EW moving average of the series
          [:math:`x_0, x_1, ..., x_t`] would be:

        .. math::
            y_t = \frac{x_t + (1 - \alpha)x_{t-1} + (1 - \alpha)^2 x_{t-2} + ... + (1 -
            \alpha)^t x_0}{1 + (1 - \alpha) + (1 - \alpha)^2 + ... + (1 - \alpha)^t}

        - When ``adjust=False``, the exponentially weighted function is calculated
          recursively:

        .. math::
            \begin{split}
                y_0 &= x_0\\
                y_t &= (1 - \alpha) y_{t-1} + \alpha x_t,
            \end{split}
    # ignore_na : bool, default False
    #     当计算权重时是否忽略缺失值。
    #
    #     - 当 ``ignore_na=False``（默认）时，权重基于绝对位置。
    #       例如，在计算 [:math:`x_0`, None, :math:`x_2`] 的加权平均时，
    #       如果 ``adjust=True``，则 :math:`x_0` 和 :math:`x_2` 的权重分别为 :math:`(1-\alpha)^2` 和 :math:`1`，
    #       如果 ``adjust=False``，则分别为 :math:`(1-\alpha)^2` 和 :math:`\alpha`。
    #
    #     - 当 ``ignore_na=True`` 时，权重基于相对位置。
    #       例如，在计算 [:math:`x_0`, None, :math:`x_2`] 的加权平均时，
    #       如果 ``adjust=True``，则 :math:`x_0` 和 :math:`x_2` 的权重分别为 :math:`1-\alpha` 和 :math:`1`，
    #       如果 ``adjust=False``，则分别为 :math:`1-\alpha` 和 :math:`\alpha`。
    ignore_na : bool, default False

    # times : np.ndarray, Series, default None
    #
    #     仅适用于 ``mean()`` 方法。
    #
    #     对应于观察时间的数组。必须是单调递增且 ``datetime64[ns]`` 数据类型。
    #
    #     如果是类似于 1-D 数组，则形状与观察值相同的序列。
    times : np.ndarray, Series, default None

    # method : str {'single', 'table'}, default 'single'
    #     .. versionadded:: 1.4.0
    #
    #     执行滚动操作时，是按单列或单行（``'single'``）还是整个对象（``'table'``）进行操作。
    #
    #     此参数仅在方法调用时指定 ``engine='numba'`` 时实现。
    #
    #     仅适用于 ``mean()`` 方法。
    method : str {'single', 'table'}, default 'single'

    # Returns
    # -------
    # pandas.api.typing.ExponentialMovingWindow
    #     用于进一步进行指数加权（EW）计算的 ExponentialMovingWindow 实例，例如使用 ``mean`` 方法。
    Returns
    -------
    pandas.api.typing.ExponentialMovingWindow

    # See Also
    # --------
    # rolling : 提供滚动窗口计算。
    # expanding : 提供扩展转换。
    #
    # Notes
    # -----
    # 查看 :ref:`Windowing Operations <window.exponentially_weighted>` 获取更多用法详情和示例。
    #
    # Examples
    # --------
    # >>> df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
    # >>> df
    #      B
    # 0  0.0
    # 1  1.0
    # 2  2.0
    # 3  NaN
    # 4  4.0
    #
    # >>> df.ewm(com=0.5).mean()
    #           B
    # 0  0.000000
    # 1  0.750000
    # 2  1.615385
    # 3  1.615385
    # 4  3.670213
    #
    # **adjust**
    #
    # >>> df.ewm(com=0.5, adjust=True).mean()
    #           B
    # 0  0.000000
    # 1  0.750000
    # 2  1.615385
    # 3  1.615385
    # 4  3.670213
    #
    # >>> df.ewm(com=0.5, adjust=False).mean()
    #           B
    # 0  0.000000
    # 1  0.666667
    # 2  1.555556
    # 3  1.555556
    # 4  3.650794
    #
    # **ignore_na**
    #
    # >>> df.ewm(com=0.5, ignore_na=True).mean()
    #           B
    # 0  0.000000
    # 1  0.750000
    # 2  1.615385
    # 3  1.615385
    # 4  3.225000
    #
    # >>> df.ewm(com=0.5, ignore_na=False).mean()
    #           B
    # 0  0.000000
    # 1  0.750000
    # 2  1.615385
    # 3  1.615385
    # 以指数加权的方式计算均值，权重由与给定时间序列“times”相关的时间差“halflife”计算得出
    """
    Exponentially weighted mean with weights calculated with a timedelta ``halflife``
    relative to ``times``.
    """
    
    _attributes = [
        "com",           # 指数加权中的压缩因子
        "span",          # 指数加权中的跨度
        "halflife",      # 指数加权中的半衰期，可以是浮点数或时间间隔类型
        "alpha",         # 指数加权中的衰减系数
        "min_periods",   # 最少需要的观测值数量
        "adjust",        # 是否进行偏差校正
        "ignore_na",     # 是否忽略缺失值
        "times",         # 时间序列数据，可以是numpy数组或数据框架
        "method",        # 计算加权平均的方法
    ]
    
    def __init__(
        self,
        obj: NDFrame,                          # 要进行指数加权平均的数据结构
        com: float | None = None,              # 指数加权中的压缩因子，默认为None
        span: float | None = None,             # 指数加权中的跨度，默认为None
        halflife: float | TimedeltaConvertibleTypes | None = None,  # 指数加权中的半衰期，默认为None，可以是浮点数或时间间隔类型
        alpha: float | None = None,            # 指数加权中的衰减系数，默认为None
        min_periods: int | None = 0,           # 执行计算所需的最小观测值数量，默认为0
        adjust: bool = True,                   # 是否进行偏差校正，默认为True
        ignore_na: bool = False,               # 是否忽略缺失值，默认为False
        times: np.ndarray | NDFrame | None = None,  # 时间序列数据，可以是numpy数组或数据框架，默认为None
        method: str = "single",                # 计算加权平均的方法，默认为"single"
        *,
        selection=None,                        # 保留参数，用于未来可能的扩展
    ):
    ) -> None:
        # 初始化函数，继承父类的初始化方法
        super().__init__(
            obj=obj,  # 设置对象数据
            min_periods=1 if min_periods is None else max(int(min_periods), 1),  # 设置最小周期数
            on=None,  # 不应用on参数
            center=False,  # 禁用center参数
            closed=None,  # 不应用closed参数
            method=method,  # 设置方法参数
            selection=selection,  # 设置选择参数
        )
        self.com = com  # 设置指数加权移动平均的com参数
        self.span = span  # 设置指数加权移动平均的span参数
        self.halflife = halflife  # 设置半衰期参数
        self.alpha = alpha  # 设置指数加权平均的alpha参数
        self.adjust = adjust  # 设置是否调整参数
        self.ignore_na = ignore_na  # 设置忽略NA值的参数
        self.times = times  # 设置时间戳参数
        if self.times is not None:
            if not self.adjust:
                raise NotImplementedError("times is not supported with adjust=False.")
            times_dtype = getattr(self.times, "dtype", None)
            if not (
                is_datetime64_dtype(times_dtype)
                or isinstance(times_dtype, DatetimeTZDtype)
            ):
                raise ValueError("times must be datetime64 dtype.")
            if len(self.times) != len(obj):
                raise ValueError("times must be the same length as the object.")
            if not isinstance(self.halflife, (str, datetime.timedelta, np.timedelta64)):
                raise ValueError("halflife must be a timedelta convertible object")
            if isna(self.times).any():
                raise ValueError("Cannot convert NaT values to integer")
            self._deltas = _calculate_deltas(self.times, self.halflife)
            # 当计算COM时，半衰期不再适用
            # 但是如果用户传递其他衰减参数，仍允许计算COM
            if common.count_not_none(self.com, self.span, self.alpha) > 0:
                self._com = get_center_of_mass(self.com, self.span, None, self.alpha)
            else:
                self._com = 1.0
        else:
            if self.halflife is not None and isinstance(
                self.halflife, (str, datetime.timedelta, np.timedelta64)
            ):
                raise ValueError(
                    "halflife can only be a timedelta convertible argument if "
                    "times is not None."
                )
            # 没有指定times时，假设数据点是等间隔的
            self._deltas = np.ones(max(self.obj.shape[0] - 1, 0), dtype=np.float64)
            self._com = get_center_of_mass(
                self.com,
                self.span,
                self.halflife,  # type: ignore[arg-type]
                self.alpha,
            )

    def _check_window_bounds(
        self, start: np.ndarray, end: np.ndarray, num_vals: int
    ) -> None:
        # emw算法是迭代的，每个点都有
        # ExponentialMovingWindowIndexer的"bounds"是整个窗口
        pass
    def _get_window_indexer(self) -> BaseIndexer:
        """
        Return an indexer class that will compute the window start and end bounds
        """
        # 返回一个指数移动窗口索引器的实例，用于计算窗口的起始和结束边界
        return ExponentialMovingWindowIndexer()

    def online(
        self, engine: str = "numba", engine_kwargs=None
    ) -> OnlineExponentialMovingWindow:
        """
        Return an ``OnlineExponentialMovingWindow`` object to calculate
        exponentially moving window aggregations in an online method.

        .. versionadded:: 1.3.0

        Parameters
        ----------
        engine: str, default ``'numba'``
            Execution engine to calculate online aggregations.
            Applies to all supported aggregation methods.

        engine_kwargs : dict, default None
            Applies to all supported aggregation methods.

            * For ``'numba'`` engine, the engine can accept ``nopython``, ``nogil``
              and ``parallel`` dictionary keys. The values must either be ``True`` or
              ``False``. The default ``engine_kwargs`` for the ``'numba'`` engine is
              ``{'nopython': True, 'nogil': False, 'parallel': False}`` and will be
              applied to the function

        Returns
        -------
        OnlineExponentialMovingWindow
        """
        # 返回一个用于执行在线指数移动窗口聚合计算的 ``OnlineExponentialMovingWindow`` 对象
        return OnlineExponentialMovingWindow(
            obj=self.obj,
            com=self.com,
            span=self.span,
            halflife=self.halflife,
            alpha=self.alpha,
            min_periods=self.min_periods,
            adjust=self.adjust,
            ignore_na=self.ignore_na,
            times=self.times,
            engine=engine,
            engine_kwargs=engine_kwargs,
            selection=self._selection,
        )

    @doc(
        _shared_docs["aggregate"],
        see_also=dedent(
            """
        See Also
        --------
        pandas.DataFrame.rolling.aggregate
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

        >>> df.ewm(alpha=0.5).mean()
                  A         B         C
        0  1.000000  4.000000  7.000000
        1  1.666667  4.666667  7.666667
        2  2.428571  5.428571  8.428571
        """
        ),
        klass="Series/Dataframe",
        axis="",
    )
    def aggregate(self, func, *args, **kwargs):
        # 调用父类的聚合方法，对数据进行聚合操作
        return super().aggregate(func, *args, **kwargs)

    agg = aggregate
    @doc(
        template_header,
        create_section_header("Parameters"),  # 创建参数部分的标题
        kwargs_numeric_only,  # 使用数值参数关键字参数模板
        window_agg_numba_parameters(),  # 获取窗口聚合 numba 参数
        create_section_header("Returns"),  # 创建返回部分的标题
        template_returns,  # 使用返回模板
        create_section_header("See Also"),  # 创建相关信息部分的标题
        template_see_also,  # 使用相关信息模板
        create_section_header("Notes"),  # 创建注释部分的标题
        numba_notes,  # 使用 numba 的注释内容
        create_section_header("Examples"),  # 创建示例部分的标题
        dedent(
            """\
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.ewm(alpha=.2).mean()
        0    1.000000
        1    1.555556
        2    2.147541
        3    2.775068
        dtype: float64
        """
        ),  # 缩进示例代码块
        window_method="ewm",  # 设置窗口方法为 ewm
        aggregation_description="(exponential weighted moment) mean",  # 聚合描述为指数加权平均
        agg_method="mean",  # 使用平均方法
    )
    def mean(
        self,
        numeric_only: bool = False,
        engine=None,
        engine_kwargs=None,
    ):
        if maybe_use_numba(engine):  # 如果可能使用 numba 引擎
            if self.method == "single":  # 如果方法为单一
                func = generate_numba_ewm_func  # 使用 numba 生成指数加权移动平均函数
            else:
                func = generate_numba_ewm_table_func  # 使用 numba 生成表格形式的指数加权移动平均函数
            ewm_func = func(
                **get_jit_arguments(engine_kwargs),  # 获取 numba 的 JIT 参数
                com=self._com,  # 设置 com 参数
                adjust=self.adjust,  # 设置 adjust 参数
                ignore_na=self.ignore_na,  # 设置 ignore_na 参数
                deltas=tuple(self._deltas),  # 设置 deltas 参数为 _deltas 元组
                normalize=True,  # 设置 normalize 参数为 True
            )
            return self._apply(ewm_func, name="mean")  # 应用 ewm_func 函数，并命名为 "mean"
        elif engine in ("cython", None):  # 如果引擎为 cython 或者 None
            if engine_kwargs is not None:
                raise ValueError("cython engine does not accept engine_kwargs")  # 抛出错误，cython 引擎不接受 engine_kwargs

            deltas = None if self.times is None else self._deltas  # 如果 times 为 None，则 deltas 为 None，否则为 _deltas
            window_func = partial(
                window_aggregations.ewm,  # 使用窗口聚合 ewm
                com=self._com,  # 设置 com 参数
                adjust=self.adjust,  # 设置 adjust 参数
                ignore_na=self.ignore_na,  # 设置 ignore_na 参数
                deltas=deltas,  # 设置 deltas 参数
                normalize=True,  # 设置 normalize 参数为 True
            )
            return self._apply(window_func, name="mean", numeric_only=numeric_only)  # 应用 window_func 函数，并命名为 "mean"，同时传递 numeric_only 参数
        else:
            raise ValueError("engine must be either 'numba' or 'cython'")  # 抛出错误，引擎必须为 'numba' 或 'cython'

    @doc(
        template_header,
        create_section_header("Parameters"),  # 创建参数部分的标题
        kwargs_numeric_only,  # 使用数值参数关键字参数模板
        window_agg_numba_parameters(),  # 获取窗口聚合 numba 参数
        create_section_header("Returns"),  # 创建返回部分的标题
        template_returns,  # 使用返回模板
        create_section_header("See Also"),  # 创建相关信息部分的标题
        template_see_also,  # 使用相关信息模板
        create_section_header("Notes"),  # 创建注释部分的标题
        numba_notes,  # 使用 numba 的注释内容
        create_section_header("Examples"),  # 创建示例部分的标题
        dedent(
            """\
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.ewm(alpha=.2).sum()
        0    1.000
        1    2.800
        2    5.240
        3    8.192
        dtype: float64
        """
        ),  # 缩进示例代码块
        window_method="ewm",  # 设置窗口方法为 ewm
        aggregation_description="(exponential weighted moment) sum",  # 聚合描述为指数加权和
        agg_method="sum",  # 使用求和方法
    )
    def sum(
        self,
        numeric_only: bool = False,
        engine=None,
        engine_kwargs=None,
    ):
        # 如果未设置 adjust 参数，则抛出 NotImplementedError 异常
        if not self.adjust:
            raise NotImplementedError("sum is not implemented with adjust=False")
        # 如果设置了 times 参数，则抛出 NotImplementedError 异常
        if self.times is not None:
            raise NotImplementedError("sum is not implemented with times")
        # 如果可以使用 numba 引擎，则根据 method 选择生成相应的 numba 函数
        if maybe_use_numba(engine):
            if self.method == "single":
                func = generate_numba_ewm_func
            else:
                func = generate_numba_ewm_table_func
            # 生成 numba 的指数加权移动平均函数，使用传入的引擎参数和其他设置
            ewm_func = func(
                **get_jit_arguments(engine_kwargs),
                com=self._com,
                adjust=self.adjust,
                ignore_na=self.ignore_na,
                deltas=tuple(self._deltas),
                normalize=False,
            )
            # 应用生成的函数到数据上，返回结果
            return self._apply(ewm_func, name="sum")
        # 如果使用的引擎是 "cython" 或者 None
        elif engine in ("cython", None):
            # 如果传入了 engine_kwargs 参数，则抛出 ValueError 异常
            if engine_kwargs is not None:
                raise ValueError("cython engine does not accept engine_kwargs")

            # 根据 times 参数选择是否设置 deltas
            deltas = None if self.times is None else self._deltas
            # 创建局部函数 window_func，使用窗口聚合中的指数加权移动平均函数
            window_func = partial(
                window_aggregations.ewm,
                com=self._com,
                adjust=self.adjust,
                ignore_na=self.ignore_na,
                deltas=deltas,
                normalize=False,
            )
            # 应用 window_func 到数据上，返回结果
            return self._apply(window_func, name="sum", numeric_only=numeric_only)
        else:
            # 如果引擎不是 "numba" 或者 "cython"，则抛出 ValueError 异常
            raise ValueError("engine must be either 'numba' or 'cython'")
        
    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """\
        bias : bool, default False
            Use a standard estimation bias correction.
        """
        ),
        kwargs_numeric_only,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.ewm(alpha=.2).std()
        0         NaN
        1    0.707107
        2    0.995893
        3    1.277320
        dtype: float64
        """
        ),
        window_method="ewm",
        aggregation_description="(exponential weighted moment) standard deviation",
        agg_method="std",
    )
    # 标准差方法，计算指数加权移动平均的标准差
    def std(self, bias: bool = False, numeric_only: bool = False):
        # 如果 numeric_only 为 True，并且数据维度为 1，并且数据类型不是数值型，则抛出 NotImplementedError 异常
        if (
            numeric_only
            and self._selected_obj.ndim == 1
            and not is_numeric_dtype(self._selected_obj.dtype)
        ):
            # 直接抛出异常，确保错误消息中显示 std 而不是 var
            raise NotImplementedError(
                f"{type(self).__name__}.std does not implement numeric_only"
            )
        # 如果设置了 times 参数，则抛出 NotImplementedError 异常
        if self.times is not None:
            raise NotImplementedError("std is not implemented with times")
        # 返回调用 var 方法计算方差后再开平方根的结果，传入的参数包括 bias 和 numeric_only
        return zsqrt(self.var(bias=bias, numeric_only=numeric_only))
    # 使用 @doc 装饰器添加文档注释，描述函数的参数、返回值、示例等信息
    @doc(
        template_header,  # 使用预定义的模板头部
        create_section_header("Parameters"),  # 创建参数部分的标题
        dedent(
            """\
        bias : bool, default False
            Use a standard estimation bias correction.
        """
        ),  # 缩进并描述参数 bias 的含义和默认值
        kwargs_numeric_only,  # 使用关键字参数 numeric_only
        create_section_header("Returns"),  # 创建返回值部分的标题
        template_returns,  # 使用预定义的返回值模板
        create_section_header("See Also"),  # 创建相关内容部分的标题
        template_see_also,  # 使用预定义的相关内容模板
        create_section_header("Examples"),  # 创建示例部分的标题
        dedent(
            """\
        >>> ser = pd.Series([1, 2, 3, 4])
        >>> ser.ewm(alpha=.2).var()
        0         NaN
        1    0.500000
        2    0.991803
        3    1.631547
        dtype: float64
        """
        ),  # 缩进并提供 var 函数的使用示例
        window_method="ewm",  # 指定窗口方法为 ewm
        aggregation_description="(exponential weighted moment) variance",  # 聚合描述为指数加权矩的方差
        agg_method="var",  # 聚合方法为方差
    )
    # 定义 ewm 对象的 var 方法，计算加权移动平均的方差
    def var(self, bias: bool = False, numeric_only: bool = False):
        # 如果存在时间信息，则抛出未实现的错误
        if self.times is not None:
            raise NotImplementedError("var is not implemented with times")
        # 使用 ewmcov 函数作为窗口函数
        window_func = window_aggregations.ewmcov
        # 使用 partial 函数创建 wfunc，配置相关参数
        wfunc = partial(
            window_func,
            com=self._com,
            adjust=self.adjust,
            ignore_na=self.ignore_na,
            bias=bias,
        )

        # 定义 var_func 函数，用于应用窗口函数计算方差
        def var_func(values, begin, end, min_periods):
            return wfunc(values, begin, end, min_periods, values)

        # 调用 _apply 方法，应用 var_func 函数，并指定名称和是否仅数值
        return self._apply(var_func, name="var", numeric_only=numeric_only)

    # 使用 @doc 装饰器添加文档注释，描述函数的参数、返回值、示例等信息
    @doc(
        template_header,  # 使用预定义的模板头部
        create_section_header("Parameters"),  # 创建参数部分的标题
        dedent(
            """\
        other : Series or DataFrame , optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndex DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        bias : bool, default False
            Use a standard estimation bias correction.
        """
        ),  # 缩进并描述参数 other, pairwise, bias 的含义和默认值
        kwargs_numeric_only,  # 使用关键字参数 numeric_only
        create_section_header("Returns"),  # 创建返回值部分的标题
        template_returns,  # 使用预定义的返回值模板
        create_section_header("See Also"),  # 创建相关内容部分的标题
        template_see_also,  # 使用预定义的相关内容模板
        create_section_header("Examples"),  # 创建示例部分的标题
        dedent(
            """\
        >>> ser1 = pd.Series([1, 2, 3, 4])
        >>> ser2 = pd.Series([10, 11, 13, 16])
        >>> ser1.ewm(alpha=.2).cov(ser2)
        0         NaN
        1    0.500000
        2    1.524590
        3    3.408836
        dtype: float64
        """
        ),  # 缩进并提供 cov 函数的使用示例
        window_method="ewm",  # 指定窗口方法为 ewm
        aggregation_description="(exponential weighted moment) sample covariance",  # 聚合描述为指数加权矩的样本协方差
        agg_method="cov",  # 聚合方法为协方差
    )
    @doc(
        template_header,
        create_section_header("Parameters"),
        dedent(
            """\
        other : Series or DataFrame, optional
            If not supplied then will default to self and produce pairwise
            output.
        pairwise : bool, default None
            If False then only matching columns between self and other will be
            used and the output will be a DataFrame.
            If True then all pairwise combinations will be calculated and the
            output will be a MultiIndex DataFrame in the case of DataFrame
            inputs. In the case of missing elements, only complete pairwise
            observations will be used.
        """
        ),
        kwargs_numeric_only,
        create_section_header("Returns"),
        template_returns,
        create_section_header("See Also"),
        template_see_also,
        create_section_header("Examples"),
        dedent(
            """\
        >>> ser1 = pd.Series([1, 2, 3, 4])
        >>> ser2 = pd.Series([10, 11, 13, 16])
        >>> ser1.ewm(alpha=.2).corr(ser2)
        0         NaN
        1    1.000000
        2    0.982821
        3    0.977802
        dtype: float64
        """
        ),
        window_method="ewm",
        aggregation_description="(exponential weighted moment) sample correlation",
        agg_method="corr",
        )
        # 定义一个方法corr，用于计算相关系数
        def corr(
            self,
            other: DataFrame | Series | None = None,
            pairwise: bool | None = None,
            numeric_only: bool = False,
        ):
            # 如果self.times不为空，抛出未实现错误
            if self.times is not None:
                raise NotImplementedError("corr is not implemented with times")

            # 导入pandas的Series类
            from pandas import Series

            # 验证numeric_only参数是否有效
            self._validate_numeric_only("corr", numeric_only)

            # 定义一个内部函数cov_func，用于计算协方差
            def cov_func(x, y):
                # 调用_prep_values方法处理x和y，返回数组
                x_array = self._prep_values(x)
                y_array = self._prep_values(y)
                # 获取窗口索引器
                window_indexer = self._get_window_indexer()
                # 确定最小周期数
                min_periods = (
                    self.min_periods
                    if self.min_periods is not None
                    else window_indexer.window_size
                )
                # 获取窗口边界的起始和结束位置
                start, end = window_indexer.get_window_bounds(
                    num_values=len(x_array),
                    min_periods=min_periods,
                    center=self.center,
                    closed=self.closed,
                    step=self.step,
                )

                # 定义一个内部函数_cov，用于计算指数加权移动协方差
                def _cov(X, Y):
                    return window_aggregations.ewmcov(
                        X,
                        start,
                        end,
                        min_periods,
                        Y,
                        self._com,
                        self.adjust,
                        self.ignore_na,
                        True,
                    )

                # 忽略所有的numpy错误
                with np.errstate(all="ignore"):
                    # 计算x_array和y_array的指数加权移动协方差
                    cov = _cov(x_array, y_array)
                    # 计算x_array和x_array的指数加权移动协方差，即方差
                    x_var = _cov(x_array, x_array)
                    # 计算y_array和y_array的指数加权移动协方差，即方差
                    y_var = _cov(y_array, y_array)
                    # 计算相关系数，即协方差除以x和y的标准差乘积的平方根
                    result = cov / zsqrt(x_var * y_var)
                # 返回一个Series对象，包含计算结果，索引为x的索引，名称为x的名称
                return Series(result, index=x.index, name=x.name, copy=False)

            # 调用_apply_pairwise方法，对self._selected_obj和other应用cov_func函数，
            # 根据pairwise和numeric_only参数选择是否对所有数据进行计算
            return self._apply_pairwise(
                self._selected_obj, other, pairwise, cov_func, numeric_only
            )
class ExponentialMovingWindowGroupby(BaseWindowGroupby, ExponentialMovingWindow):
    """
    Provide an exponential moving window groupby implementation.
    """

    _attributes = ExponentialMovingWindow._attributes + BaseWindowGroupby._attributes

    def __init__(self, obj, *args, _grouper=None, **kwargs) -> None:
        super().__init__(obj, *args, _grouper=_grouper, **kwargs)

        if not obj.empty and self.times is not None:
            # sort the times and recalculate the deltas according to the groups
            groupby_order = np.concatenate(list(self._grouper.indices.values()))
            # Calculate deltas based on sorted times within groups
            self._deltas = _calculate_deltas(
                self.times.take(groupby_order),
                self.halflife,
            )

    def _get_window_indexer(self) -> GroupbyIndexer:
        """
        Return an indexer class that will compute the window start and end bounds

        Returns
        -------
        GroupbyIndexer
            Indexer object for defining window boundaries based on groupby indices
        """
        window_indexer = GroupbyIndexer(
            groupby_indices=self._grouper.indices,
            window_indexer=ExponentialMovingWindowIndexer,
        )
        return window_indexer


class OnlineExponentialMovingWindow(ExponentialMovingWindow):
    def __init__(
        self,
        obj: NDFrame,
        com: float | None = None,
        span: float | None = None,
        halflife: float | TimedeltaConvertibleTypes | None = None,
        alpha: float | None = None,
        min_periods: int | None = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        times: np.ndarray | NDFrame | None = None,
        engine: str = "numba",
        engine_kwargs: dict[str, bool] | None = None,
        *,
        selection=None,
    ) -> None:
        if times is not None:
            # Online operations do not support custom times
            raise NotImplementedError(
                "times is not implemented with online operations."
            )
        super().__init__(
            obj=obj,
            com=com,
            span=span,
            halflife=halflife,
            alpha=alpha,
            min_periods=min_periods,
            adjust=adjust,
            ignore_na=ignore_na,
            times=times,
            selection=selection,
        )
        # Initialize mean state for exponential moving window
        self._mean = EWMMeanState(self._com, self.adjust, self.ignore_na, obj.shape)
        # Check and set the computational engine
        if maybe_use_numba(engine):
            self.engine = engine
            self.engine_kwargs = engine_kwargs
        else:
            # Raise error if 'numba' is not the chosen engine
            raise ValueError("'numba' is the only supported engine")

    def reset(self) -> None:
        """
        Reset the state captured by `update` calls.
        """
        # Reset mean state
        self._mean.reset()

    def aggregate(self, func, *args, **kwargs):
        # Online aggregation not implemented
        raise NotImplementedError("aggregate is not implemented.")

    def std(self, bias: bool = False, *args, **kwargs):
        # Standard deviation calculation not implemented
        raise NotImplementedError("std is not implemented.")

    def corr(
        self,
        other: DataFrame | Series | None = None,
        pairwise: bool | None = None,
        numeric_only: bool = False,
        ...
    # 如果调用了此方法，抛出未实现错误，表明相关的功能尚未实现
    ):
        raise NotImplementedError("corr is not implemented.")

    # 计算协方差矩阵或者系列之间的协方差
    def cov(
        self,
        other: DataFrame | Series | None = None,  # 可选参数，可以是 DataFrame、Series 或者 None
        pairwise: bool | None = None,  # 可选参数，指示是否计算成对的协方差
        bias: bool = False,  # 可选参数，指示是否应用偏差校正
        numeric_only: bool = False,  # 可选参数，指示是否仅计算数值型列的协方差
    ):
        # 如果调用了此方法，抛出未实现错误，表明相关的功能尚未实现
        raise NotImplementedError("cov is not implemented.")

    # 计算方差
    def var(self, bias: bool = False, numeric_only: bool = False):
        # 如果调用了此方法，抛出未实现错误，表明相关的功能尚未实现
        raise NotImplementedError("var is not implemented.")
```