# `D:\src\scipysrc\sympy\sympy\plotting\plot.py`

```
"""Plotting module for SymPy.

A plot is represented by the ``Plot`` class that contains a reference to the
backend and a list of the data series to be plotted. The data series are
instances of classes meant to simplify getting points and meshes from SymPy
expressions. ``plot_backends`` is a dictionary with all the backends.

This module gives only the essential. For all the fancy stuff use directly
the backend. You can get the backend wrapper for every plot from the
``_backend`` attribute. Moreover the data series classes have various useful
methods like ``get_points``, ``get_meshes``, etc, that may
be useful if you wish to use another plotting library.

Especially if you need publication ready graphs and this module is not enough
for you - just get the ``_backend`` attribute and add whatever you want
directly to it. In the case of matplotlib (the common way to graph data in
python) just copy ``_backend.fig`` which is the figure and ``_backend.ax``
which is the axis and work on them as you would on any other matplotlib object.

Simplicity of code takes much greater importance than performance. Do not use it
if you care at all about performance. A new backend instance is initialized
every time you call ``show()`` and the old one is left to the garbage collector.
"""

# Import necessary modules from SymPy and other plotting libraries
from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function, AppliedUndef
from sympy.core.symbol import (Dummy, Symbol, Wild)
from sympy.external import import_module
from sympy.functions import sign
from sympy.plotting.backends.base_backend import Plot
from sympy.plotting.backends.matplotlibbackend import MatplotlibBackend
from sympy.plotting.backends.textbackend import TextBackend
from sympy.plotting.series import (
    LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries,
    ParametricSurfaceSeries, SurfaceOver2DRangeSeries, ContourSeries)
from sympy.plotting.utils import _check_arguments, _plot_sympify
from sympy.tensor.indexed import Indexed

# to maintain back-compatibility, import certain modules (though not used directly)
from sympy.plotting.plotgrid import PlotGrid # noqa: F401
from sympy.plotting.series import BaseSeries # noqa: F401
from sympy.plotting.series import Line2DBaseSeries # noqa: F401
from sympy.plotting.series import Line3DBaseSeries # noqa: F401
from sympy.plotting.series import SurfaceBaseSeries # noqa: F401
from sympy.plotting.series import List2DSeries # noqa: F401
from sympy.plotting.series import GenericDataSeries # noqa: F401
from sympy.plotting.series import centers_of_faces # noqa: F401
from sympy.plotting.series import centers_of_segments # noqa: F401
from sympy.plotting.series import flat # noqa: F401
from sympy.plotting.backends.base_backend import unset_show # noqa: F401
from sympy.plotting.backends.matplotlibbackend import _matplotlib_list # noqa: F401
from sympy.plotting.textplot import textplot # noqa: F401

# Specify additional modules required for doctesting
__doctest_requires__ = {
    # 定义一个元组，包含多个元素，每个元素是一个字符串
    ('plot3d',
     'plot3d_parametric_line',
     'plot3d_parametric_surface',
     'plot_parametric'): ['matplotlib'],
    # XXX: plot 的文档测试可能不应该依赖 matplotlib 库。在文本后端情况下，应该支持 plot(x**2, (x, -5, 5))，但它失败了。
    # 这里引发了一个警告，表示可能不应该依赖 matplotlib 库来运行这些文档测试。

    # 定义一个元组，仅包含一个元素，该元素是字符串 'plot'
    ('plot',): ['matplotlib'],
def _process_summations(sum_bound, *args):
    """Substitute oo (infinity) in the lower/upper bounds of a summation with
    some integer number.

    Parameters
    ==========

    sum_bound : int
        oo will be substituted with this integer number.
    *args : list/tuple
        pre-processed arguments of the form (expr, range, ...)

    Notes
    =====
    Let's consider the following summation: ``Sum(1 / x**2, (x, 1, oo))``.
    The current implementation of lambdify (SymPy 1.12 at the time of
    writing this) will create something of this form:
    ``sum(1 / x**2 for x in range(1, INF))``
    The problem is that ``type(INF)`` is float, while ``range`` requires
    integers: the evaluation fails.
    Instead of modifying ``lambdify`` (which requires a deep knowledge), just
    replace it with some integer number.
    """
    def new_bound(t, bound):
        # Check if t is not a number or is finite, return t as is
        if (not t.is_number) or t.is_finite:
            return t
        # If t is positive, return bound; if negative, return -bound
        if sign(t) >= 0:
            return bound
        return -bound

    # Convert args to list
    args = list(args)
    # Extract the expression from args
    expr = args[0]

    # Define a Wild symbol 'w' with properties to match specific conditions
    w = Wild("w", properties=[
        lambda t: isinstance(t, Sum),  # Match instances of Sum
        lambda t: any((not a[1].is_finite) or (not a[2].is_finite) for i, a in enumerate(t.args) if i > 0)
        # Match cases where either lower or upper bound is not finite
    ])

    # Iterate over all occurrences of 'w' in the expression
    for t in list(expr.find(w)):
        sums_args = list(t.args)
        # Iterate over arguments of the summation
        for i, a in enumerate(sums_args):
            if i > 0:
                # Replace bounds with new_bound function results
                sums_args[i] = (a[0], new_bound(a[1], sum_bound),
                    new_bound(a[2], sum_bound))
        # Create a new Sum object with updated arguments
        s = Sum(*sums_args)
        # Substitute the original 't' with the new Sum object in the expression
        expr = expr.subs(t, s)

    # Update the first argument in 'args' with the modified expression
    args[0] = expr
    return args


def _build_line_series(*args, **kwargs):
    """Loop over the provided arguments and create the necessary line series.
    """
    series = []
    # Extract sum_bound from kwargs, default to 1000 if not provided
    sum_bound = int(kwargs.get("sum_bound", 1000))
    # Iterate over each argument in args
    for arg in args:
        expr, r, label, rendering_kw = arg
        kw = kwargs.copy()
        # If rendering_kw is not None, include it in kw
        if rendering_kw is not None:
            kw["rendering_kw"] = rendering_kw
        # Check if expr is not callable, then process summations in the argument
        if not callable(expr):
            arg = _process_summations(sum_bound, *arg)
        # Create LineOver1DRangeSeries object and append to series list
        series.append(LineOver1DRangeSeries(*arg[:-1], **kw))
    return series


def _create_series(series_type, plot_expr, **kwargs):
    """Extract the rendering_kw dictionary from the provided arguments and
    create an appropriate data series.
    """
    series = []
    # Iterate over each args in plot_expr
    for args in plot_expr:
        kw = kwargs.copy()
        # If the last element of args is not None, include it as rendering_kw in kw
        if args[-1] is not None:
            kw["rendering_kw"] = args[-1]
        # Create series_type object and append to series list
        series.append(series_type(*args[:-1], **kw))
    return series


def _set_labels(series, labels, rendering_kw):
    """Apply the `label` and `rendering_kw` keyword arguments to the series.
    """
    # If labels is not a list or tuple, convert it to a list
    if not isinstance(labels, (list, tuple)):
        labels = [labels]
    # 如果标签列表长度大于0，则进入条件判断
    if len(labels) > 0:
        # 如果标签列表长度为1且数据系列长度大于1，
        # 将同一个标签应用于所有数据系列。这保持了向后兼容性
        if len(labels) == 1 and len(series) > 1:
            labels *= len(series)
        
        # 如果数据系列的数量不等于标签的数量，抛出数值错误异常
        if len(series) != len(labels):
            raise ValueError("The number of labels must be equal to the "
                "number of expressions being plotted.\nReceived "
                f"{len(series)} expressions and {len(labels)} labels")

        # 使用zip函数将每个数据系列与其对应的标签进行关联
        for s, l in zip(series, labels):
            s.label = l

    # 如果有渲染关键字参数
    if rendering_kw:
        # 如果渲染关键字参数是字典类型，则转换为列表
        if isinstance(rendering_kw, dict):
            rendering_kw = [rendering_kw]
        
        # 如果渲染关键字参数列表长度为1，则复制该参数以匹配数据系列数量
        if len(rendering_kw) == 1:
            rendering_kw *= len(series)
        # 如果数据系列的数量不等于渲染关键字参数的数量，抛出数值错误异常
        elif len(series) != len(rendering_kw):
            raise ValueError("The number of rendering dictionaries must be "
                "equal to the number of expressions being plotted.\nReceived "
                f"{len(series)} expressions and {len(labels)} labels")
        
        # 使用zip函数将每个数据系列与其对应的渲染关键字参数进行关联
        for s, r in zip(series, rendering_kw):
            s.rendering_kw = r
def plot_factory(*args, **kwargs):
    # 从 kwargs 中弹出 backend 参数，默认为 "default"
    backend = kwargs.pop("backend", "default")
    
    # 如果 backend 是字符串类型
    if isinstance(backend, str):
        # 如果 backend 是 "default"
        if backend == "default":
            # 尝试导入 matplotlib 模块，指定最低版本为 '1.1.0'，捕获 RuntimeError 异常
            matplotlib = import_module('matplotlib',
                                       min_module_version='1.1.0', catch=(RuntimeError,))
            # 如果导入成功
            if matplotlib:
                # 返回 MatplotlibBackend 类的实例化对象
                return MatplotlibBackend(*args, **kwargs)
            # 如果导入失败，则返回 TextBackend 类的实例化对象
            return TextBackend(*args, **kwargs)
        # 如果 backend 在 plot_backends 字典中存在，则返回对应类的实例化对象
        return plot_backends[backend](*args, **kwargs)
    
    # 如果 backend 是类型，并且是 Plot 类的子类，则返回该类的实例化对象
    elif (type(backend) == type) and issubclass(backend, Plot):
        return backend(*args, **kwargs)
    
    # 如果 backend 类型既不是字符串也不是 Plot 的子类，则抛出 TypeError 异常
    else:
        raise TypeError("backend must be either a string or a subclass of ``Plot``.")


plot_backends = {
    'matplotlib': MatplotlibBackend,
    'text': TextBackend,
}

####New API for plotting module ####

# TODO: Add color arrays for plots.
# TODO: Add more plotting options for 3d plots.
# TODO: Adaptive sampling for 3D plots.

def plot(*args, show=True, **kwargs):
    """Plots a function of a single variable as a curve.

    Parameters
    ==========

    args :
        The first argument is the expression representing the function
        of single variable to be plotted.

        The last argument is a 3-tuple denoting the range of the free
        variable. e.g. ``(x, 0, 5)``

        Typical usage examples are in the following:

        - Plotting a single expression with a single range.
            ``plot(expr, range, **kwargs)``
        - Plotting a single expression with the default range (-10, 10).
            ``plot(expr, **kwargs)``
        - Plotting multiple expressions with a single range.
            ``plot(expr1, expr2, ..., range, **kwargs)``
        - Plotting multiple expressions with multiple ranges.
            ``plot((expr1, range1), (expr2, range2), ..., **kwargs)``

        It is best practice to specify range explicitly because default
        range may change in the future if a more advanced default range
        detection algorithm is implemented.

    show : bool, optional
        The default value is set to ``True``. Set show to ``False`` and
        the function will not display the plot. The returned instance of
        the ``Plot`` class can then be used to save or display the plot
        by calling the ``save()`` and ``show()`` methods respectively.

    line_color : string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

    title : str, optional
        Title of the plot. It is set to the latex representation of
        the expression, if the plot has only one expression.

    label : str, optional
        The label of the expression in the plot. It will be used when
        called with ``legend``. Default is the name of the expression.
        e.g. ``sin(x)``

    xlabel : str or expression, optional
        Label for the x-axis.
        
    # y轴的标签，可以是字符串或表达式，用于显示y轴的标签。
    ylabel : str or expression, optional

    # 设置x轴的缩放类型，可以是'linear'（线性）或'log'（对数）。
    xscale : 'linear' or 'log', optional

    # 设置y轴的缩放类型，可以是'linear'（线性）或'log'（对数）。
    yscale : 'linear' or 'log', optional

    # 设置图形的中心点坐标，可以是一个包含两个浮点数的元组，或者字符串'center'或'auto'。
    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}

    # 设置x轴的限制范围，格式为(min, max)。
    xlim : (float, float), optional
        Denotes the x-axis limits, ``(min, max)```.

    # 设置y轴的限制范围，格式为(min, max)。
    ylim : (float, float), optional
        Denotes the y-axis limits, ``(min, max)```.

    # 用于添加注解的字典列表，每个字典指定了matplotlib.axes.Axes.annotate方法的参数。
    annotations : list, optional
        A list of dictionaries specifying the type of annotation
        required. The keys in the dictionary should be equivalent
        to the arguments of the :external:mod:`matplotlib`'s
        :external:meth:`~matplotlib.axes.Axes.annotate` method.

    # 用于指定要绘制的标记类型的字典列表，每个字典指定了matplotlib.pyplot.plot函数的参数及相关标记参数。
    markers : list, optional
        A list of dictionaries specifying the type the markers required.
        The keys in the dictionary should be equivalent to the arguments
        of the :external:mod:`matplotlib`'s :external:func:`~matplotlib.pyplot.plot()` function
        along with the marker related keyworded arguments.

    # 用于指定要绘制的矩形的尺寸的字典列表，每个字典指定了matplotlib.patches.Rectangle类的参数。
    rectangles : list, optional
        A list of dictionaries specifying the dimensions of the
        rectangles to be plotted. The keys in the dictionary should be
        equivalent to the arguments of the :external:mod:`matplotlib`'s
        :external:class:`~matplotlib.patches.Rectangle` class.

    # 用于指定图形中颜色填充类型的字典，每个字典指定了matplotlib.axes.Axes.fill_between方法的参数。
    fill : dict, optional
        A dictionary specifying the type of color filling required in
        the plot. The keys in the dictionary should be equivalent to the
        arguments of the :external:mod:`matplotlib`'s
        :external:meth:`~matplotlib.axes.Axes.fill_between` method.

    # 默认为True。如果设置为False，并指定n，表示需要均匀抽样。
    # 绘图使用自适应算法进行绘制，递归采样以准确绘制。自适应算法使用中点附近的随机点进一步采样。
    # 因此相同的图可能会略有不同。
    adaptive : bool, optional
        The default value is set to ``True``. Set adaptive to ``False``
        and specify ``n`` if uniform sampling is required.

    # 自适应算法的递归深度。深度值为n时最多采样2^n个点。
    # 如果adaptive标志设置为False，则此参数将被忽略。
    depth : int, optional
        Recursion depth of the adaptive algorithm. A depth of value
        `n` samples a maximum of `2^{n}` points.

    # 当adaptive设置为False时使用。在n个点上均匀采样函数。
    # 如果adaptive标志设置为True，则此参数将被忽略。
    # 此关键字参数取代了nb_of_points，后者应视为废弃。
    n : int, optional
        Used when the ``adaptive`` is set to ``False``. The function
        is uniformly sampled at ``n`` number of points. If the ``adaptive``
        flag is set to ``True``, this will be ignored.
        This keyword argument replaces ``nb_of_points``, which should be
        considered deprecated.
    # 尺寸参数，可选的浮点数元组，表示图形的宽度和高度（单位为英寸）。默认为 None，表示由默认后端设置尺寸。
    
    Examples
    ========
    
    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True
    
       >>> from sympy import symbols
       >>> from sympy.plotting import plot
       >>> x = symbols('x')
    
    Single Plot
    
    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True
    
       >>> plot(x**2, (x, -5, 5))
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-5.0, 5.0)
    
    Multiple plots with single range.
    
    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True
    
       >>> plot(x, x**2, x**3, (x, -5, 5))
       Plot object containing:
       [0]: cartesian line: x for x over (-5.0, 5.0)
       [1]: cartesian line: x**2 for x over (-5.0, 5.0)
       [2]: cartesian line: x**3 for x over (-5.0, 5.0)
    
    Multiple plots with different ranges.
    
    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True
    
       >>> plot((x**2, (x, -6, 6)), (x, (x, -5, 5)))
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-6.0, 6.0)
       [1]: cartesian line: x for x over (-5.0, 5.0)
    
    No adaptive sampling.
    
    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True
    
       >>> plot(x**2, adaptive=False, n=400)
       Plot object containing:
       [0]: cartesian line: x**2 for x over (-10.0, 10.0)
    
    See Also
    ========
    
    Plot, LineOver1DRangeSeries
    
    """
    # 对输入参数进行符号化处理
    args = _plot_sympify(args)
    # 检查参数数量和类型，确保参数符合要求
    plot_expr = _check_arguments(args, 1, 1, **kwargs)
    # 获取参数中的 "params" 键对应的值，默认为 None
    params = kwargs.get("params", None)
    # 用于存储自由变量的集合
    free = set()
    # 遍历每个绘图表达式
    for p in plot_expr:
        # 如果第一个元素不是字符串，则将其加入自由变量集合
        if not isinstance(p[1][0], str):
            free |= {p[1][0]}
        else:
            free |= {Symbol(p[1][0])}
    # 如果存在 params，则从自由变量中移除对应的键
    if params:
        free = free.difference(params.keys())
    # 弹出一个自由变量作为 x，如果集合不为空
    x = free.pop() if free else Symbol("x")
    # 设置默认的 x 轴标签为 x
    kwargs.setdefault('xlabel', x)
    # 设置默认的 y 轴标签为 f(x)
    kwargs.setdefault('ylabel', Function('f')(x))
    
    # 获取标签列表
    labels = kwargs.pop("label", [])
    # 获取渲染参数
    rendering_kw = kwargs.pop("rendering_kw", None)
    # 构建线性系列对象
    series = _build_line_series(*plot_expr, **kwargs)
    # 设置标签
    _set_labels(series, labels, rendering_kw)
    
    # 使用系列对象创建绘图对象
    plots = plot_factory(*series, **kwargs)
    # 如果 show 参数为 True，则显示绘图对象
    if show:
        plots.show()
    # 返回绘图对象
    return plots
# 定义一个函数用于绘制二维参数曲线
def plot_parametric(*args, show=True, **kwargs):
    """
    Plots a 2D parametric curve.

    Parameters
    ==========

    args
        Common specifications are:

        - Plotting a single parametric curve with a range
            ``plot_parametric((expr_x, expr_y), range)``
        - Plotting multiple parametric curves with the same range
            ``plot_parametric((expr_x, expr_y), ..., range)``
        - Plotting multiple parametric curves with different ranges
            ``plot_parametric((expr_x, expr_y, range), ...)``

        ``expr_x`` is the expression representing $x$ component of the
        parametric function.

        ``expr_y`` is the expression representing $y$ component of the
        parametric function.

        ``range`` is a 3-tuple denoting the parameter symbol, start and
        stop. For example, ``(u, 0, 5)``.

        If the range is not specified, then a default range of (-10, 10)
        is used.

        However, if the arguments are specified as
        ``(expr_x, expr_y, range), ...``, you must specify the ranges
        for each expressions manually.

        Default range may change in the future if a more advanced
        algorithm is implemented.

    adaptive : bool, optional
        Specifies whether to use the adaptive sampling or not.

        The default value is set to ``True``. Set adaptive to ``False``
        and specify ``n`` if uniform sampling is required.

    depth :  int, optional
        The recursion depth of the adaptive algorithm. A depth of
        value $n$ samples a maximum of $2^n$ points.

    n : int, optional
        Used when the ``adaptive`` flag is set to ``False``. Specifies the
        number of the points used for the uniform sampling.
        This keyword argument replaces ``nb_of_points``, which should be
        considered deprecated.

    line_color : string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

    label : str, optional
        The label of the expression in the plot. It will be used when
        called with ``legend``. Default is the name of the expression.
        e.g. ``sin(x)``

    xlabel : str, optional
        Label for the x-axis.

    ylabel : str, optional
        Label for the y-axis.

    xscale : 'linear' or 'log', optional
        Sets the scaling of the x-axis.

    yscale : 'linear' or 'log', optional
        Sets the scaling of the y-axis.

    axis_center : (float, float), optional
        Tuple of two floats denoting the coordinates of the center or
        {'center', 'auto'}

    xlim : (float, float), optional
        Denotes the x-axis limits, ``(min, max)```.

    ylim : (float, float), optional
        Denotes the y-axis limits, ``(min, max)```.


    show : bool, optional
        Whether to display the plot immediately after creating it.
        Default is ``True``.

    kwargs : additional keyword arguments
        Additional options passed to the plotting function.
    """
    pass
    # 设置参数 size，用于指定图形的尺寸，单位为英寸，是一个可选的元组 (width, height)
    # 默认为 None，表示尺寸将由默认的后端设置。
    
    Examples
    ========
    
    # 导入必要的函数和符号
    .. plot::
       :context: reset
       :format: doctest
       :include-source: True
    
       >>> from sympy import plot_parametric, symbols, cos, sin
       >>> u = symbols('u')
    
    一个单表达式的参数化绘图示例：
    
    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True
    
       >>> plot_parametric((cos(u), sin(u)), (u, -5, 5))
       包含绘图对象：
       [0]: 参数化笛卡尔线：(cos(u), sin(u))，u范围为(-5.0, 5.0)
    
    多表达式且范围相同的参数化绘图示例：
    
    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True
    
       >>> plot_parametric((cos(u), sin(u)), (u, cos(u)), (u, -10, 10))
       包含绘图对象：
       [0]: 参数化笛卡尔线：(cos(u), sin(u))，u范围为(-10.0, 10.0)
       [1]: 参数化笛卡尔线：(u, cos(u))，u范围为(-10.0, 10.0)
    
    多表达式且每条曲线范围不同的参数化绘图示例：
    
    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True
    
       >>> plot_parametric((cos(u), sin(u), (u, -5, 5)),
       ...     (cos(u), u, (u, -5, 5)))
       包含绘图对象：
       [0]: 参数化笛卡尔线：(cos(u), sin(u))，u范围为(-5.0, 5.0)
       [1]: 参数化笛卡尔线：(cos(u), u)，u范围为(-5.0, 5.0)
    
    Notes
    =====
    
    绘图使用自适应算法进行采样，以精确绘制曲线。该自适应算法使用在两点中点附近的随机点进行进一步采样。
    因此，重复相同的绘图命令可能会因为随机采样的原因而略有不同结果。
    
    如果有多个绘图，则相同的可选参数将同时应用于同一画布中的所有绘图。
    如果要分别设置这些选项，可以索引返回的 "Plot" 对象并设置它。
    
    例如，当你一次指定 "line_color" 时，它会同时应用于两个系列。
    
    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True
    
        >>> from sympy import pi
        >>> expr1 = (u, cos(2*pi*u)/2 + 1/2)
        >>> expr2 = (u, sin(2*pi*u)/2 + 1/2)
        >>> p = plot_parametric(expr1, expr2, (u, 0, 1), line_color='blue')
    
    如果要为特定系列指定线条颜色，应分别索引每个项目并手动应用属性。
    
    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True
    
        >>> p[0].line_color = 'red'
        >>> p.show()
    
    See Also
    ========
    
    Plot, Parametric2DLineSeries
    # 对参数进行符号化处理，确保参数适合绘图
    args = _plot_sympify(args)
    
    # 检查参数，确保至少有两个表达式和一个可选的绘图参数
    plot_expr = _check_arguments(args, 2, 1, **kwargs)
    
    # 从关键字参数中提取标签（label），如果没有则默认为空列表
    labels = kwargs.pop("label", [])
    
    # 从关键字参数中提取渲染参数（rendering_kw），如果没有则默认为None
    rendering_kw = kwargs.pop("rendering_kw", None)
    
    # 使用参数创建 Parametric2DLineSeries 类型的数据系列
    series = _create_series(Parametric2DLineSeries, plot_expr, **kwargs)
    
    # 设置数据系列的标签和渲染参数
    _set_labels(series, labels, rendering_kw)
    
    # 使用数据系列和其他关键字参数创建绘图对象
    plots = plot_factory(*series, **kwargs)
    
    # 如果指定了 show 参数为 True，则展示绘图
    if show:
        plots.show()
    
    # 返回绘图对象
    return plots
def plot3d_parametric_line(*args, show=True, **kwargs):
    """
    Plots a 3D parametric line plot.

    Usage
    =====

    Single plot:

    ``plot3d_parametric_line(expr_x, expr_y, expr_z, range, **kwargs)``

    If the range is not specified, then a default range of (-10, 10) is used.

    Multiple plots.

    ``plot3d_parametric_line((expr_x, expr_y, expr_z, range), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    expr_x : Expression representing the function along x.

    expr_y : Expression representing the function along y.

    expr_z : Expression representing the function along z.

    range : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the parameter variable, e.g., (u, 0, 5).

    Keyword Arguments
    =================

    Arguments for ``Parametric3DLineSeries`` class.

    n : int
        The range is uniformly sampled at ``n`` number of points.
        This keyword argument replaces ``nb_of_points``, which should be
        considered deprecated.

    Aesthetics:

    line_color : string, or float, or function, optional
        Specifies the color for the plot.
        See ``Plot`` to see how to set color for the plots.
        Note that by setting ``line_color``, it would be applied simultaneously
        to all the series.

    label : str
        The label to the plot. It will be used when called with ``legend=True``
        to denote the function with the given label in the plot.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class.

    title : str
        Title of the plot.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin
       >>> from sympy.plotting import plot3d_parametric_line
       >>> u = symbols('u')

    Single plot.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_line(cos(u), sin(u), u, (u, -5, 5))
       Plot object containing:
       [0]: 3D parametric cartesian line: (cos(u), sin(u), u) for u over (-5.0, 5.0)


    Multiple plots.

    """
    # Function definition for plotting 3D parametric line plots
    # Handles both single and multiple plots based on arguments
    # Parses expressions for x, y, z coordinates and their ranges
    # Uses keyword arguments to customize plot aesthetics and attributes
    # Returns a Plot object encapsulating the generated plot(s)
    pass
    .. plot::
       :context: close-figs  # 设置图表上下文为关闭状态，确保不影响其它图表
       :format: doctest  # 指定图表格式为 doctest，通常用于文档测试
       :include-source: True  # 在输出中包含源代码

       >>> plot3d_parametric_line((cos(u), sin(u), u, (u, -5, 5)),
       ...     (sin(u), u**2, u, (u, -5, 5)))
       Plot object containing:
       [0]: 3D parametric cartesian line: (cos(u), sin(u), u) for u over (-5.0, 5.0)
       [1]: 3D parametric cartesian line: (sin(u), u**2, u) for u over (-5.0, 5.0)

    See Also
    ========

    Plot, Parametric3DLineSeries  # 参考链接，指向 Plot 和 Parametric3DLineSeries 相关文档

    """
    args = _plot_sympify(args)  # 使用 _plot_sympify 函数处理输入参数
    plot_expr = _check_arguments(args, 3, 1, **kwargs)  # 检查参数数量并进行必要的参数检查，返回绘图表达式
    kwargs.setdefault("xlabel", "x")  # 设置 x 轴标签，默认为 "x"
    kwargs.setdefault("ylabel", "y")  # 设置 y 轴标签，默认为 "y"
    kwargs.setdefault("zlabel", "z")  # 设置 z 轴标签，默认为 "z"

    labels = kwargs.pop("label", [])  # 从 kwargs 中弹出 "label" 键的值，默认为空列表
    rendering_kw = kwargs.pop("rendering_kw", None)  # 从 kwargs 中弹出 "rendering_kw" 键的值，若不存在则为 None
    series = _create_series(Parametric3DLineSeries, plot_expr, **kwargs)  # 使用 _create_series 函数创建 Parametric3DLineSeries 对象
    _set_labels(series, labels, rendering_kw)  # 设置系列对象的标签和渲染参数

    plots = plot_factory(*series, **kwargs)  # 使用 plot_factory 函数创建绘图对象
    if show:  # 如果 show 参数为 True，则显示绘图对象
        plots.show()
    return plots  # 返回绘图对象
def _plot3d_plot_contour_helper(Series, *args, **kwargs):
    """plot3d and plot_contour are structurally identical. Let's reduce
    code repetition.
    """
    # 如果将此导入放在顶层模块中，可能会导致 SymPy 的可选依赖测试失败。
    # 导入 SymPy 中的 BaseScalar 类
    from sympy.vector import BaseScalar

    # 将参数 args 转换为 sympy 的表达式
    args = _plot_sympify(args)
    # 检查参数数量是否符合要求，并返回绘图表达式
    plot_expr = _check_arguments(args, 1, 2, **kwargs)

    # 初始化空集合以存储自由变量
    free_x = set()
    free_y = set()
    # 定义可能的符号类型
    _types = (Symbol, BaseScalar, Indexed, AppliedUndef)
    # 遍历绘图表达式列表
    for p in plot_expr:
        # 如果第一个自变量是指定的类型之一，添加到自由 x 变量集合中；否则创建一个符号并添加
        free_x |= {p[1][0]} if isinstance(p[1][0], _types) else {Symbol(p[1][0])}
        # 如果第二个自变量是指定的类型之一，添加到自由 y 变量集合中；否则创建一个符号并添加
        free_y |= {p[2][0]} if isinstance(p[2][0], _types) else {Symbol(p[2][0])}
    # 从自由 x 变量集合中弹出一个变量作为 x，如果集合为空则创建一个默认符号 "x"
    x = free_x.pop() if free_x else Symbol("x")
    # 从自由 y 变量集合中弹出一个变量作为 y，如果集合为空则创建一个默认符号 "y"
    y = free_y.pop() if free_y else Symbol("y")
    # 设置关键字参数的默认值：xlabel 为 x，ylabel 为 y，zlabel 为 Function('f')(x, y)
    kwargs.setdefault("xlabel", x)
    kwargs.setdefault("ylabel", y)
    kwargs.setdefault("zlabel", Function('f')(x, y))

    # 如果请求了极坐标离散化并且自动标签已应用，则隐藏 x 和 y 轴上的标签。
    if kwargs.get("is_polar", False):
        # 如果 xlabel 是可调用的，则将其设置为空字符串
        if callable(kwargs["xlabel"]):
            kwargs["xlabel"] = ""
        # 如果 ylabel 是可调用的，则将其设置为空字符串
        if callable(kwargs["ylabel"]):
            kwargs["ylabel"] = ""

    # 弹出 "label" 关键字的值，并将其赋给 labels 变量；如果不存在，则设为空列表
    labels = kwargs.pop("label", [])
    # 弹出 "rendering_kw" 关键字的值，并将其赋给 rendering_kw 变量；如果不存在，则设为 None
    rendering_kw = kwargs.pop("rendering_kw", None)
    # 根据 Series 类型创建绘图数据 series
    series = _create_series(Series, plot_expr, **kwargs)
    # 设置 series 的标签和渲染关键字
    _set_labels(series, labels, rendering_kw)
    # 使用 plot_factory 创建绘图 plots
    plots = plot_factory(*series, **kwargs)
    # 如果关键字参数中没有设置 show=False，则显示 plots
    if kwargs.get("show", True):
        plots.show()
    # 返回绘图 plots
    return plots
    kwargs.setdefault("show", show)
    # 设置默认参数 "show"，如果未提供则使用函数的默认值 show

    return _plot3d_plot_contour_helper(
        SurfaceOver2DRangeSeries, *args, **kwargs)
    # 调用 _plot3d_plot_contour_helper 函数，传入 SurfaceOver2DRangeSeries 类型和其他参数及关键字参数
    # 返回 _plot3d_plot_contour_helper 函数的结果
def plot3d_parametric_surface(*args, show=True, **kwargs):
    """
    Plots a 3D parametric surface plot.

    Explanation
    ===========

    Single plot.

    ``plot3d_parametric_surface(expr_x, expr_y, expr_z, range_u, range_v, **kwargs)``

    If the ranges is not specified, then a default range of (-10, 10) is used.

    Multiple plots.

    ``plot3d_parametric_surface((expr_x, expr_y, expr_z, range_u, range_v), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    expr_x : Expression representing the function along ``x``.

    expr_y : Expression representing the function along ``y``.

    expr_z : Expression representing the function along ``z``.

    range_u : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the u variable, e.g. (u, 0, 5).

    range_v : (:class:`~.Symbol`, float, float)
        A 3-tuple denoting the range of the v variable, e.g. (v, 0, 5).

    Keyword Arguments
    =================

    Arguments for ``ParametricSurfaceSeries`` class:

    n1 : int
        The ``u`` range is sampled uniformly at ``n1`` of points.
        This keyword argument replaces ``nb_of_points_u``, which should be
        considered deprecated.

    n2 : int
        The ``v`` range is sampled uniformly at ``n2`` of points.
        This keyword argument replaces ``nb_of_points_v``, which should be
        considered deprecated.

    Aesthetics:

    surface_color : Function which returns a float
        Specifies the color for the surface of the plot. See
        :class:`~Plot` for more details.

    If there are multiple plots, then the same series arguments are applied for
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.


    Arguments for ``Plot`` class:

    title : str
        Title of the plot.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of the
        overall figure. The default value is set to ``None``, meaning the size will
        be set by the default backend.

    Examples
    ========

    .. plot::
       :context: reset
       :format: doctest
       :include-source: True

       >>> from sympy import symbols, cos, sin
       >>> from sympy.plotting import plot3d_parametric_surface
       >>> u, v = symbols('u v')

    Single plot.

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

       >>> plot3d_parametric_surface(cos(u + v), sin(u - v), u - v,
       ...     (u, -5, 5), (v, -5, 5))
       Plot object containing:
       [0]: parametric cartesian surface: (cos(u + v), sin(u - v), u - v) for u over (-5.0, 5.0) and v over (-5.0, 5.0)


    See Also
    ========

    Plot, ParametricSurfaceSeries

    """

    # Convert input arguments to sympy expressions
    args = _plot_sympify(args)
    # 使用 _check_arguments 函数验证参数并生成绘图表达式
    plot_expr = _check_arguments(args, 3, 2, **kwargs)
    # 设置默认的坐标轴标签，如果未提供则使用 "x"，"y"，"z"
    kwargs.setdefault("xlabel", "x")
    kwargs.setdefault("ylabel", "y")
    kwargs.setdefault("zlabel", "z")

    # 获取参数中的 "label"，并作为列表存储在 labels 中
    labels = kwargs.pop("label", [])
    # 获取参数中的 "rendering_kw"，若未提供则为 None
    rendering_kw = kwargs.pop("rendering_kw", None)
    
    # 使用 _create_series 函数根据参数创建 ParametricSurfaceSeries 对象的系列
    series = _create_series(ParametricSurfaceSeries, plot_expr, **kwargs)
    # 使用 _set_labels 函数为系列设置标签和渲染参数
    _set_labels(series, labels, rendering_kw)
    
    # 使用 plot_factory 函数创建绘图对象 plots，将系列作为参数传入
    plots = plot_factory(*series, **kwargs)
    # 如果 show 参数为 True，则显示绘图对象
    if show:
        plots.show()
    # 返回创建的绘图对象
    return plots
# 绘制函数的等高线图
def plot_contour(*args, show=True, **kwargs):
    """
    Draws contour plot of a function

    Usage
    =====

    Single plot

    ``plot_contour(expr, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plot with the same range.

    ``plot_contour(expr1, expr2, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plots with different ranges.

    ``plot_contour((expr1, range_x, range_y), (expr2, range_x, range_y), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    expr : Expression representing the function along x.

    range_x : (:class:`Symbol`, float, float)
        A 3-tuple denoting the range of the x variable, e.g. (x, 0, 5).

    range_y : (:class:`Symbol`, float, float)
        A 3-tuple denoting the range of the y variable, e.g. (y, 0, 5).

    Keyword Arguments
    =================

    Arguments for ``ContourSeries`` class:

    n1 : int
        The x range is sampled uniformly at ``n1`` of points.
        This keyword argument replaces ``nb_of_points_x``, which should be
        considered deprecated.

    n2 : int
        The y range is sampled uniformly at ``n2`` of points.
        This keyword argument replaces ``nb_of_points_y``, which should be
        considered deprecated.

    Aesthetics:

    surface_color : Function which returns a float
        Specifies the color for the surface of the plot. See
        :class:`sympy.plotting.Plot` for more details.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class:

    title : str
        Title of the plot.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    See Also
    ========

    Plot, ContourSeries

    """
    # 设置默认的显示参数为 show=True
    kwargs.setdefault("show", show)
    # 调用帮助函数 _plot3d_plot_contour_helper，传入 ContourSeries 类型和其他参数
    return _plot3d_plot_contour_helper(ContourSeries, *args, **kwargs)


# 检查参数并转换为表达式和范围的元组形式
def check_arguments(args, expr_len, nb_of_free_symbols):
    """
    Checks the arguments and converts into tuples of the
    form (exprs, ranges).

    Examples
    ========

    ```
    if not args:
        # 如果没有传入参数，则返回空列表
        return []
    if expr_len > 1 and isinstance(args[0], Expr):
        # 如果表达式数量大于1且第一个参数是SymPy表达式对象
        # 表达式长度大于1时，参数是元组
        if len(args) < expr_len:
            # 如果参数长度小于表达式长度，引发数值错误
            raise ValueError("len(args) should not be less than expr_len")
        for i in range(len(args)):
            if isinstance(args[i], Tuple):
                break
        else:
            i = len(args) + 1

        exprs = Tuple(*args[:i])
        # 获取表达式集合中所有自由符号的并集
        free_symbols = list(set().union(*[e.free_symbols for e in exprs]))
        if len(args) == expr_len + nb_of_free_symbols:
            # 如果参数长度等于表达式长度加上自由符号数量
            # 给定了范围
            plots = [exprs + Tuple(*args[expr_len:])]
        else:
            default_range = Tuple(-10, 10)
            ranges = []
            # 对于每个自由符号，设置默认范围为(-10, 10)
            for symbol in free_symbols:
                ranges.append(Tuple(symbol) + default_range)

            for i in range(len(free_symbols) - nb_of_free_symbols):
                ranges.append(Tuple(Dummy()) + default_range)
            # 生成图形的参数组合
            plots = [exprs + Tuple(*ranges)]
        return plots
    # 检查第一个参数是否为表达式（Expr）或者是元组（Tuple），且长度与表达式长度相同且不为3
    if isinstance(args[0], Expr) or (isinstance(args[0], Tuple) and
                                     len(args[0]) == expr_len and
                                     expr_len != 3):
        # 无法处理表达式长度为3的情况。无法区分表达式和范围。
        # 一系列具有相同范围的图形
        for i in range(len(args)):
            # 如果参数是元组且长度与表达式长度不同，则中断循环
            if isinstance(args[i], Tuple) and len(args[i]) != expr_len:
                break
            # 如果参数不是元组，则将其转换为只包含该参数的元组
            if not isinstance(args[i], Tuple):
                args[i] = Tuple(args[i])
        else:
            i = len(args) + 1

        # 提取参数中的表达式列表
        exprs = args[:i]
        # 断言所有的表达式都是 Expr 类型
        assert all(isinstance(e, Expr) for expr in exprs for e in expr)
        # 提取所有表达式中的自由符号，并去重
        free_symbols = list(set().union(*[e.free_symbols for expr in exprs
                                          for e in expr]))

        # 如果自由符号的数量超过预期的自由符号数量，则抛出 ValueError 异常
        if len(free_symbols) > nb_of_free_symbols:
            raise ValueError("The number of free_symbols in the expression "
                             "is greater than %d" % nb_of_free_symbols)
        
        # 如果参数的数量等于 i + nb_of_free_symbols，并且 args[i] 是元组
        if len(args) == i + nb_of_free_symbols and isinstance(args[i], Tuple):
            # 提取范围
            ranges = Tuple(*list(args[
                           i:i + nb_of_free_symbols]))
            # 为每个表达式添加范围，并返回结果列表
            plots = [expr + ranges for expr in exprs]
            return plots
        else:
            # 使用默认范围
            default_range = Tuple(-10, 10)
            ranges = []
            # 对于每个自由符号，生成默认范围的元组
            for symbol in free_symbols:
                ranges.append(Tuple(symbol) + default_range)

            # 如果自由符号的数量少于预期的自由符号数量，使用 Dummy 符号填充剩余范围
            for i in range(nb_of_free_symbols - len(free_symbols)):
                ranges.append(Tuple(Dummy()) + default_range)
            ranges = Tuple(*ranges)
            # 为每个表达式添加范围，并返回结果列表
            plots = [expr + ranges for expr in exprs]
            return plots

    # 如果第一个参数是元组，且长度等于表达式长度加上自由符号的数量
    elif isinstance(args[0], Tuple) and len(args[0]) == expr_len + nb_of_free_symbols:
        # 多个具有不同范围的图形
        for arg in args:
            # 检查每个元组中的前 expr_len 个元素是否是表达式
            for i in range(expr_len):
                if not isinstance(arg[i], Expr):
                    raise ValueError("Expected an expression, given %s" %
                                     str(arg[i]))
            # 检查每个元组中的后 nb_of_free_symbols 个元素是否是长度为3的元组
            for i in range(nb_of_free_symbols):
                if not len(arg[i + expr_len]) == 3:
                    raise ValueError("The ranges should be a tuple of "
                                     "length 3, got %s" % str(arg[i + expr_len]))
        # 如果所有检查都通过，返回参数列表
        return args
```