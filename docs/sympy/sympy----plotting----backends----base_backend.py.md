# `D:\src\scipysrc\sympy\sympy\plotting\backends\base_backend.py`

```
from sympy.plotting.series import BaseSeries, GenericDataSeries
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence

# 定义特定于 matplotlib 的 doctest 依赖，使其测试 Plot.append 和 Plot.extend 方法
__doctest_requires__ = {
    ('Plot.append', 'Plot.extend'): ['matplotlib'],
}

# 全局变量，默认为 True，用于控制是否显示绘图
_show = True

def unset_show():
    """
    禁用 show() 函数，在测试中使用。
    """
    global _show
    _show = False

def _deprecation_msg_m_a_r_f(attr):
    """
    生成关于属性过时警告的消息。

    :param attr: 属性名称
    """
    sympy_deprecation_warning(
        f"The `{attr}` property is deprecated. The `{attr}` keyword "
        "argument should be passed to a plotting function, which generates "
        "the appropriate data series. If needed, index the plot object to "
        "retrieve a specific data series.",
        deprecated_since_version="1.13",
        active_deprecations_target="deprecated-markers-annotations-fill-rectangles",
        stacklevel=4)

def _create_generic_data_series(**kwargs):
    """
    根据关键字参数创建通用数据系列对象列表。

    :param kwargs: 关键字参数，包含 annotations, markers, fill, rectangles
    :return: GenericDataSeries 对象列表
    """
    keywords = ["annotations", "markers", "fill", "rectangles"]
    series = []
    for kw in keywords:
        dictionaries = kwargs.pop(kw, [])
        if dictionaries is None:
            dictionaries = []
        if isinstance(dictionaries, dict):
            dictionaries = [dictionaries]
        for d in dictionaries:
            args = d.pop("args", [])
            series.append(GenericDataSeries(kw, *args, **d))
    return series

class Plot:
    """
    所有后端的基类。每个后端代表一个绘图库，实现了使用 SymPy 绘图函数所需的必要功能。

    对于交互式工作，函数 :func:`plot()` 更适合使用。

    此类允许使用多种后端（如 matplotlib、textplot、旧的 SymPy 的 pyglet 模块、Google
    charts api 等）绘制 SymPy 表达式。

    图形可以包含任意数量的 SymPy 表达式的绘图、点坐标列表等。Plot 类有一个私有属性 _series，
    包含所有要绘制的数据系列（线条或曲面的表达式、点的列表等，都是 BaseSeries 的子类实例）。
    
    图形的定制有两个级别。全局选项涉及整个图形（例如标题、xlabel、scale 等），
    每个数据系列可以有各自的选项（例如名称）和美学效果（例如颜色、点形状、线型等）。

    选项和美学之间的区别在于，美学可以是坐标的函数（或参数化绘图中的参数）。美学的支持值包括：

    - None（后端使用默认值）
    - 一个常数
    - 一个一变量函数（第一个坐标或参数）
    - 一个二变量函数（第一个和第二个坐标或参数）
    """

    def __init__(self):
        """
        构造函数，初始化 Plot 对象。
        """
        self._series = []  # 初始化数据系列为空列表
    - a function of three variables (only in nonparametric 3D plots)
    
    Their implementation depends on the backend so they may not work in some
    backends.
    
    If the plot is parametric and the arity of the aesthetic function permits
    it the aesthetic is calculated over parameters and not over coordinates.
    If the arity does not permit calculation over parameters the calculation is
    done over coordinates.
    
    Only cartesian coordinates are supported for the moment, but you can use
    the parametric plots to plot in polar, spherical and cylindrical
    coordinates.


    The arguments for the constructor Plot must be subclasses of BaseSeries.
    
    Any global option can be specified as a keyword argument.
    
    The global options for a figure are:
    
    - title : str
    - xlabel : str or Symbol
    - ylabel : str or Symbol
    - zlabel : str or Symbol
    - legend : bool
    - xscale : {'linear', 'log'}
    - yscale : {'linear', 'log'}
    - axis : bool
    - axis_center : tuple of two floats or {'center', 'auto'}
    - xlim : tuple of two floats
    - ylim : tuple of two floats
    - aspect_ratio : tuple of two floats or {'auto'}
    - autoscale : bool
    - margin : float in [0, 1]
    - backend : {'default', 'matplotlib', 'text'} or a subclass of BaseBackend
    - size : optional tuple of two floats, (width, height); default: None


    The per data series options and aesthetics are:
    There are none in the base series. See below for options for subclasses.
    
    Some data series support additional aesthetics or options:
    
    :class:`~.LineOver1DRangeSeries`, :class:`~.Parametric2DLineSeries`, and
    :class:`~.Parametric3DLineSeries` support the following:
    
    Aesthetics:
    
    - line_color : string, or float, or function, optional
        Specifies the color for the plot, which depends on the backend being
        used.
        
        For example, if ``MatplotlibBackend`` is being used, then
        Matplotlib string colors are acceptable (``"red"``, ``"r"``,
        ``"cyan"``, ``"c"``, ...).
        Alternatively, we can use a float number, 0 < color < 1, wrapped in a
        string (for example, ``line_color="0.5"``) to specify grayscale colors.
        Alternatively, We can specify a function returning a single
        float value: this will be used to apply a color-loop (for example,
        ``line_color=lambda x: math.cos(x)``).
        
        Note that by setting line_color, it would be applied simultaneously
        to all the series.
    
    Options:
    
    - label : str
    - steps : bool
    - integers_only : bool


    :class:`~.SurfaceOver2DRangeSeries` and :class:`~.ParametricSurfaceSeries`
    support the following:
    
    Aesthetics:
    
    - surface_color : function which returns a float.


    Notes
    =====
    
    How the plotting module works:
    # 当调用绘图函数时，处理提供的表达式，并创建一个包含绘制这些表达式所需信息的
    # :class:`~sympy.plotting.series.BaseSeries` 类的实例列表。最终，这些对象将生成
    # 要绘制的数值数据。
    1. Whenever a plotting function is called, the provided expressions are
       processed and a list of instances of the
       :class:`~sympy.plotting.series.BaseSeries` class is created, containing
       the necessary information to plot the expressions
       (e.g. the expression, ranges, series name, ...). Eventually, these
       objects will generate the numerical data to be plotted.
    
    # 实例化 :class:`~.Plot` 类的一个子类（从现在开始称为后端），该类存储系列的列表
    # 和绘图的主要属性（例如轴标签，标题等）。后端实现了使用某些绘图库生成实际图形的逻辑。
    2. A subclass of :class:`~.Plot` class is instantiated (referred to as
       backend, from now on), which stores the list of series and the main
       attributes of the plot (e.g. axis labels, title, ...).
       The backend implements the logic to generate the actual figure with
       some plotting library.
    
    # 当执行 ``show`` 命令时，逐个处理系列以生成数值数据并将其添加到图形中。后端还会
    # 根据存储在 Plot 实例中的值设置轴标签、标题等。
    3. When the ``show`` command is executed, series are processed one by one
       to generate numerical data and add it to the figure. The backend is also
       going to set the axis labels, title, ..., according to the values stored
       in the Plot instance.
    def __init__(self, *args,
        title=None, xlabel=None, ylabel=None, zlabel=None, aspect_ratio='auto',
        xlim=None, ylim=None, axis_center='auto', axis=True,
        xscale='linear', yscale='linear', legend=False, autoscale=True,
        margin=0, annotations=None, markers=None, rectangles=None,
        fill=None, backend='default', size=None, **kwargs):
        # 初始化函数，接受多个位置参数和关键字参数
        # 设置图形的整体选项
        # 每个选项的可能值在 Plot 类的文档字符串中有描述，没有进行验证检查
        self.title = title  # 设置图的标题
        self.xlabel = xlabel  # 设置 x 轴标签
        self.ylabel = ylabel  # 设置 y 轴标签
        self.zlabel = zlabel  # 设置 z 轴标签
        self.aspect_ratio = aspect_ratio  # 设置图的长宽比
        self.axis_center = axis_center  # 设置轴的中心
        self.axis = axis  # 是否显示轴
        self.xscale = xscale  # 设置 x 轴的比例尺
        self.yscale = yscale  # 设置 y 轴的比例尺
        self.legend = legend  # 是否显示图例
        self.autoscale = autoscale  # 是否自动缩放
        self.margin = margin  # 设置边距
        self._annotations = annotations  # 存储标注数据对象的列表
        self._markers = markers  # 存储标记数据对象的列表
        self._rectangles = rectangles  # 存储矩形数据对象的列表
        self._fill = fill  # 存储填充数据对象的列表

        # 包含要绘制的数据对象。后端应足够智能以遍历此列表。
        self._series = []
        self._series.extend(args)  # 将位置参数添加到数据对象列表中
        self._series.extend(_create_generic_data_series(
            annotations=annotations, markers=markers, rectangles=rectangles,
            fill=fill))  # 将通过函数生成的数据对象添加到列表中

        # 函数用于检查和设置特定名称的属性值
        is_real = \
            lambda lim: all(getattr(i, 'is_real', True) for i in lim)
        is_finite = \
            lambda lim: all(getattr(i, 'is_finite', True) for i in lim)

        def check_and_set(t_name, t):
            if t:
                if not is_real(t):
                    raise ValueError(
                    "All numbers from {}={} must be real".format(t_name, t))
                if not is_finite(t):
                    raise ValueError(
                    "All numbers from {}={} must be finite".format(t_name, t))
                setattr(self, t_name, (float(t[0]), float(t[1])))

        self.xlim = None
        check_and_set("xlim", xlim)  # 检查并设置 x 轴限制
        self.ylim = None
        check_and_set("ylim", ylim)  # 检查并设置 y 轴限制
        self.size = None
        check_and_set("size", size)  # 检查并设置图像大小

    @property
    def _backend(self):
        return self  # 返回当前对象作为后端

    @property
    def backend(self):
        return type(self)  # 返回当前对象的类型作为后端

    def __str__(self):
        series_strs = [('[%d]: ' % i) + str(s)
                       for i, s in enumerate(self._series)]
        return 'Plot object containing:\n' + '\n'.join(series_strs)  # 返回描述对象内容的字符串

    def __getitem__(self, index):
        return self._series[index]  # 获取指定索引的数据对象

    def __setitem__(self, index, *args):
        if len(args) == 1 and isinstance(args[0], BaseSeries):
            self._series[index] = args  # 设置指定索引的数据对象为新值

    def __delitem__(self, index):
        del self._series[index]  # 删除指定索引的数据对象
    def append(self, arg):
        """Adds an element from a plot's series to an existing plot.

        Examples
        ========

        Consider two ``Plot`` objects, ``p1`` and ``p2``. To add the
        second plot's first series object to the first, use the
        ``append`` method, like so:

        .. plot::
           :format: doctest
           :include-source: True

           >>> from sympy import symbols
           >>> from sympy.plotting import plot
           >>> x = symbols('x')
           >>> p1 = plot(x*x, show=False)
           >>> p2 = plot(x, show=False)
           >>> p1.append(p2[0])
           >>> p1
           Plot object containing:
           [0]: cartesian line: x**2 for x over (-10.0, 10.0)
           [1]: cartesian line: x for x over (-10.0, 10.0)
           >>> p1.show()

        See Also
        ========

        extend

        """
        # 检查传入的参数是否是 BaseSeries 的实例
        if isinstance(arg, BaseSeries):
            # 如果是，则将其添加到当前 Plot 对象的 series 列表中
            self._series.append(arg)
        else:
            # 如果不是，则抛出类型错误异常
            raise TypeError('Must specify element of plot to append.')

    def extend(self, arg):
        """Adds all series from another plot.

        Examples
        ========

        Consider two ``Plot`` objects, ``p1`` and ``p2``. To add the
        second plot to the first, use the ``extend`` method, like so:

        .. plot::
           :format: doctest
           :include-source: True

           >>> from sympy import symbols
           >>> from sympy.plotting import plot
           >>> x = symbols('x')
           >>> p1 = plot(x**2, show=False)
           >>> p2 = plot(x, -x, show=False)
           >>> p1.extend(p2)
           >>> p1
           Plot object containing:
           [0]: cartesian line: x**2 for x over (-10.0, 10.0)
           [1]: cartesian line: x for x over (-10.0, 10.0)
           [2]: cartesian line: -x for x over (-10.0, 10.0)
           >>> p1.show()

        """
        # 检查传入的参数是否是 Plot 的实例或者是一个序列
        if isinstance(arg, Plot):
            # 如果是 Plot 的实例，则将其 series 列表扩展到当前 Plot 对象的 series 列表中
            self._series.extend(arg._series)
        elif is_sequence(arg):
            # 如果是序列，则直接将其扩展到当前 Plot 对象的 series 列表中
            self._series.extend(arg)
        else:
            # 如果不是上述两种类型，则抛出类型错误异常
            raise TypeError('Expecting Plot or sequence of BaseSeries')

    def show(self):
        # 未实现的方法，用于显示绘图

    def save(self, path):
        # 未实现的方法，用于保存绘图到指定路径

    def close(self):
        # 未实现的方法，用于关闭绘图

    # deprecations

    @property
    def markers(self):
        """.. deprecated:: 1.13"""
        # 返回已弃用的属性 markers，并打印相应的弃用信息
        _deprecation_msg_m_a_r_f("markers")
        return self._markers

    @markers.setter
    def markers(self, v):
        """.. deprecated:: 1.13"""
        # 设置已弃用的属性 markers，并打印相应的弃用信息，同时根据传入的值创建相应的数据序列并扩展到 series 列表中
        _deprecation_msg_m_a_r_f("markers")
        self._series.extend(_create_generic_data_series(markers=v))
        self._markers = v

    @property
    def annotations(self):
        """.. deprecated:: 1.13"""
        # 返回已弃用的属性 annotations，并打印相应的弃用信息
        _deprecation_msg_m_a_r_f("annotations")
        return self._annotations

    @annotations.setter
    # 定义一个方法 annotations，接受一个参数 v
    def annotations(self, v):
        """.. deprecated:: 1.13"""
        # 发出废弃警告信息，指示 annotations 方法已废弃
        _deprecation_msg_m_a_r_f("annotations")
        # 将生成的通用数据系列添加到 self._series 中，使用参数 v 作为数据
        self._series.extend(_create_generic_data_series(annotations=v))
        # 设置对象的 _annotations 属性为参数 v
        self._annotations = v

    # 定义一个属性方法 rectangles，用于获取 rectangles 属性值
    @property
    def rectangles(self):
        """.. deprecated:: 1.13"""
        # 发出废弃警告信息，指示 rectangles 属性已废弃
        _deprecation_msg_m_a_r_f("rectangles")
        # 返回对象的 _rectangles 属性值
        return self._rectangles

    # 定义 rectangles 属性的 setter 方法，用于设置属性值
    @rectangles.setter
    def rectangles(self, v):
        """.. deprecated:: 1.13"""
        # 发出废弃警告信息，指示 rectangles 属性已废弃
        _deprecation_msg_m_a_r_f("rectangles")
        # 将生成的通用数据系列添加到 self._series 中，使用参数 v 作为数据
        self._series.extend(_create_generic_data_series(rectangles=v))
        # 设置对象的 _rectangles 属性为参数 v
        self._rectangles = v

    # 定义一个属性方法 fill，用于获取 fill 属性值
    @property
    def fill(self):
        """.. deprecated:: 1.13"""
        # 发出废弃警告信息，指示 fill 属性已废弃
        _deprecation_msg_m_a_r_f("fill")
        # 返回对象的 _fill 属性值
        return self._fill

    # 定义 fill 属性的 setter 方法，用于设置属性值
    @fill.setter
    def fill(self, v):
        """.. deprecated:: 1.13"""
        # 发出废弃警告信息，指示 fill 属性已废弃
        _deprecation_msg_m_a_r_f("fill")
        # 将生成的通用数据系列添加到 self._series 中，使用参数 v 作为数据
        self._series.extend(_create_generic_data_series(fill=v))
        # 设置对象的 _fill 属性为参数 v
        self._fill = v
```