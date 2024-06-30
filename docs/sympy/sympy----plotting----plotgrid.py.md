# `D:\src\scipysrc\sympy\sympy\plotting\plotgrid.py`

```
# 从 sympy.external 模块导入 import_module 函数
# 导入 sympy.plotting.backends.base_backend 模块并重命名为 base_backend
from sympy.external import import_module
import sympy.plotting.backends.base_backend as base_backend


# 定义一个字典，指定 doctest 执行时需要的外部依赖
__doctest_requires__ = {
    ("PlotGrid",): ["matplotlib"],
}


class PlotGrid:
    """This class helps to plot subplots from already created SymPy plots
    in a single figure.

    Examples
    ========

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> from sympy import symbols
        >>> from sympy.plotting import plot, plot3d, PlotGrid
        >>> x, y = symbols('x, y')
        >>> p1 = plot(x, x**2, x**3, (x, -5, 5))
        >>> p2 = plot((x**2, (x, -6, 6)), (x, (x, -5, 5)))
        >>> p3 = plot(x**3, (x, -5, 5))
        >>> p4 = plot3d(x*y, (x, -5, 5), (y, -5, 5))

    Plotting vertically in a single line:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> PlotGrid(2, 1, p1, p2)
        PlotGrid object containing:
        Plot[0]:Plot object containing:
        [0]: cartesian line: x for x over (-5.0, 5.0)
        [1]: cartesian line: x**2 for x over (-5.0, 5.0)
        [2]: cartesian line: x**3 for x over (-5.0, 5.0)
        Plot[1]:Plot object containing:
        [0]: cartesian line: x**2 for x over (-6.0, 6.0)
        [1]: cartesian line: x for x over (-5.0, 5.0)

    Plotting horizontally in a single line:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> PlotGrid(1, 3, p2, p3, p4)
        PlotGrid object containing:
        Plot[0]:Plot object containing:
        [0]: cartesian line: x**2 for x over (-6.0, 6.0)
        [1]: cartesian line: x for x over (-5.0, 5.0)
        Plot[1]:Plot object containing:
        [0]: cartesian line: x**3 for x over (-5.0, 5.0)
        Plot[2]:Plot object containing:
        [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)

    Plotting in a grid form:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> PlotGrid(2, 2, p1, p2, p3, p4)
        PlotGrid object containing:
        Plot[0]:Plot object containing:
        [0]: cartesian line: x for x over (-5.0, 5.0)
        [1]: cartesian line: x**2 for x over (-5.0, 5.0)
        [2]: cartesian line: x**3 for x over (-5.0, 5.0)
        Plot[1]:Plot object containing:
        [0]: cartesian line: x**2 for x over (-6.0, 6.0)
        [1]: cartesian line: x for x over (-5.0, 5.0)
        Plot[2]:Plot object containing:
        [0]: cartesian line: x**3 for x over (-5.0, 5.0)
        Plot[3]:Plot object containing:
        [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)

    """
    def __init__(self, nrows, ncolumns, *args, show=True, size=None, **kwargs):
        """
        Parameters
        ==========

        nrows :
            The number of rows that should be in the grid of the
            required subplot.
        ncolumns :
            The number of columns that should be in the grid
            of the required subplot.

        nrows and ncolumns together define the required grid.

        Arguments
        =========

        A list of predefined plot objects entered in a row-wise sequence
        i.e. plot objects which are to be in the top row of the required
        grid are written first, then the second row objects and so on

        Keyword arguments
        =================

        show : Boolean
            The default value is set to ``True``. Set show to ``False`` and
            the function will not display the subplot. The returned instance
            of the ``PlotGrid`` class can then be used to save or display the
            plot by calling the ``save()`` and ``show()`` methods
            respectively.
        size : (float, float), optional
            A tuple in the form (width, height) in inches to specify the size of
            the overall figure. The default value is set to ``None``, meaning
            the size will be set by the default backend.
        """
        # 导入 matplotlib 模块，确保至少版本为 1.1.0，并捕获 RuntimeError 异常
        self.matplotlib = import_module('matplotlib',
            import_kwargs={'fromlist': ['pyplot', 'cm', 'collections']},
            min_module_version='1.1.0', catch=(RuntimeError,))
        # 设置行数和列数
        self.nrows = nrows
        self.ncolumns = ncolumns
        # 初始化一个空列表来存放图表对象
        self._series = []
        # 初始化图形对象为 None
        self._fig = None
        # 存储传入的图表对象到 _series 列表中
        self.args = args
        for arg in args:
            self._series.append(arg._series)
        # 设置图形的大小
        self.size = size
        # 如果 show 为 True 并且 matplotlib 已导入，则显示图形
        if show and self.matplotlib:
            self.show()

    def _create_figure(self):
        # 使用 GridSpec 创建具有指定行数和列数的网格
        gs = self.matplotlib.gridspec.GridSpec(self.nrows, self.ncolumns)
        # 创建映射字典，将每个子图和相应的图表对象关联起来
        mapping = {}
        c = 0
        for i in range(self.nrows):
            for j in range(self.ncolumns):
                if c < len(self.args):
                    mapping[gs[i, j]] = self.args[c]
                c += 1

        # 根据指定的大小创建图形对象
        kw = {} if not self.size else {"figsize": self.size}
        self._fig = self.matplotlib.pyplot.figure(**kw)
        # 遍历映射字典，为每个子图添加相应的坐标系
        for spec, p in mapping.items():
            kw = ({"projection": "3d"} if (len(p._series) > 0 and
                p._series[0].is_3D) else {})
            cur_ax = self._fig.add_subplot(spec, **kw)
            p._plotgrid_fig = self._fig
            p._plotgrid_ax = cur_ax
            p.process_series()

    @property
    def fig(self):
        # 如果图形对象为空，则调用 _create_figure 方法创建图形
        if not self._fig:
            self._create_figure()
        return self._fig

    @property
    def _backend(self):
        # 返回当前实例本身，作为后端对象
        return self

    def close(self):
        # 关闭当前图形
        self.matplotlib.pyplot.close(self.fig)
    # 定义一个方法 `show`，用于显示图形。
    def show(self):
        # 如果 `base_backend._show` 为真，则进行显示操作。
        if base_backend._show:
            # 调整图形以确保布局合适
            self.fig.tight_layout()
            # 使用 `matplotlib.pyplot.show()` 显示图形
            self.matplotlib.pyplot.show()
        else:
            # 否则调用 `close()` 方法关闭图形
            self.close()

    # 定义一个方法 `save`，用于将图形保存到指定路径。
    def save(self, path):
        # 使用 `self.fig.savefig(path)` 将图形保存到给定路径
        self.fig.savefig(path)

    # 重写 `__str__` 方法，返回描述 `PlotGrid` 对象的字符串表示。
    def __str__(self):
        # 使用列表推导式生成每个子图的字符串表示
        plot_strs = [('Plot[%d]:' % i) + str(plot)
                      for i, plot in enumerate(self.args)]
        # 返回包含所有子图描述的 `PlotGrid` 对象的字符串表示
        return 'PlotGrid object containing:\n' + '\n'.join(plot_strs)
```