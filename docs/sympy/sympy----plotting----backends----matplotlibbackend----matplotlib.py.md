# `D:\src\scipysrc\sympy\sympy\plotting\backends\matplotlibbackend\matplotlib.py`

```
# 从 collections.abc 模块中导入 Callable 类
# 从 sympy.core.basic 模块中导入 Basic 类
# 从 sympy.external 模块中导入 import_module 函数
# 从 sympy.plotting.backends.base_backend 模块中导入 base_backend
# 从 sympy.printing.latex 模块中导入 latex 函数
from collections.abc import Callable
from sympy.core.basic import Basic
from sympy.external import import_module
import sympy.plotting.backends.base_backend as base_backend
from sympy.printing.latex import latex


def _str_or_latex(label):
    """
    Convert a label to a LaTeX expression if it's of type Basic, otherwise
    return it as a string.
    """
    if isinstance(label, Basic):
        return latex(label, mode='inline')
    return str(label)


def _matplotlib_list(interval_list):
    """
    Convert a list of bounding rectangular intervals into lists suitable for
    matplotlib's ``fill`` command.
    """
    xlist = []
    ylist = []
    if len(interval_list):
        for intervals in interval_list:
            intervalx = intervals[0]
            intervaly = intervals[1]
            xlist.extend([intervalx.start, intervalx.start,
                          intervalx.end, intervalx.end, None])
            ylist.extend([intervaly.start, intervaly.end,
                          intervaly.end, intervaly.start, None])
    else:
        #XXX Ugly hack. Matplotlib does not accept empty lists for ``fill``
        xlist.extend((None, None, None, None))
        ylist.extend((None, None, None, None))
    return xlist, ylist


# 不需要在每种情况下检查导入 matplotlib 的成功与否；
# 只有在成功导入 matplotlib 后才会使用这个后端
class MatplotlibBackend(base_backend.Plot):
    """
    Implements functionalities to use Matplotlib with SymPy plotting functions.
    """

    def __init__(self, *series, **kwargs):
        """
        Initialize MatplotlibBackend with optional series data and settings.
        """
        super().__init__(*series, **kwargs)
        # 导入 matplotlib 模块，设定需要的导入参数和最低模块版本
        self.matplotlib = import_module('matplotlib',
            import_kwargs={'fromlist': ['pyplot', 'cm', 'collections']},
            min_module_version='1.1.0', catch=(RuntimeError,))
        self.plt = self.matplotlib.pyplot
        self.cm = self.matplotlib.cm
        self.LineCollection = self.matplotlib.collections.LineCollection
        self.aspect = kwargs.get('aspect_ratio', 'auto')
        # 如果 aspect_ratio 不是 'auto'，则计算纵横比例
        if self.aspect != 'auto':
            self.aspect = float(self.aspect[1]) / self.aspect[0]
        # PlotGrid 可以提供其图表和轴以填充来自系列数据的数据。
        self._plotgrid_fig = kwargs.pop("fig", None)
        self._plotgrid_ax = kwargs.pop("ax", None)
    def _create_figure(self):
        """
        创建图形对象并设置坐标轴样式。

        如果已经存在 _plotgrid_fig，则使用它作为图形对象和坐标轴对象；
        否则，创建新的图形对象，并根据系列是否包含 3D 数据决定添加 2D 或 3D 坐标轴。
        """
        def set_spines(ax):
            """
            设置坐标轴样式，将左边和底部的轴设置为零位置，并隐藏右边和顶部的轴。

            Parameters
            ----------
            ax : Axes object
                Matplotlib 的 Axes 对象
            """
            ax.spines['left'].set_position('zero')
            ax.spines['right'].set_color('none')
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        if self._plotgrid_fig is not None:
            self.fig = self._plotgrid_fig
            self.ax = self._plotgrid_ax
            if not any(s.is_3D for s in self._series):
                set_spines(self.ax)
        else:
            self.fig = self.plt.figure(figsize=self.size)
            if any(s.is_3D for s in self._series):
                self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
            else:
                self.ax = self.fig.add_subplot(1, 1, 1)
                set_spines(self.ax)

    @staticmethod
    def get_segments(x, y, z=None):
        """
        将坐标列表转换为线段列表，用于 Matplotlib 的 LineCollection。

        Parameters
        ----------
        x : list
            x 坐标列表

        y : list
            y 坐标列表

        z : list, optional
            z 坐标列表（用于 3D 线段）

        Returns
        -------
        np.ndarray
            包含线段的 numpy 数组
        """
        np = import_module('numpy')
        if z is not None:
            dim = 3
            points = (x, y, z)
        else:
            dim = 2
            points = (x, y)
        points = np.ma.array(points).T.reshape(-1, 1, dim)
        return np.ma.concatenate([points[:-1], points[1:]], axis=1)

    def process_series(self):
        """
        迭代每个 Plot 对象并调用 _process_series() 处理。
        """
        self._create_figure()
        self._process_series(self._series, self.ax)

    def show(self):
        """
        显示图形。

        调用 process_series() 准备图形数据，根据 _show 标志决定是直接显示图形还是关闭。
        """
        self.process_series()
        # TODO: 在修复 https://github.com/ipython/ipython/issues/1255 后，可以取消注释下一行并移除 pyplot.show() 的调用
        # self.fig.show()
        if base_backend._show:
            self.fig.tight_layout()
            self.plt.show()
        else:
            self.close()

    def save(self, path):
        """
        保存图形到指定路径。

        调用 process_series() 准备图形数据，并将图形保存到指定路径。
        
        Parameters
        ----------
        path : str
            保存图形的路径
        """
        self.process_series()
        self.fig.savefig(path)

    def close(self):
        """
        关闭图形。

        使用 plt.close() 方法关闭当前的图形对象。
        """
        self.plt.close(self.fig)
```