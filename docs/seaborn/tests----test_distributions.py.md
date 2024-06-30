# `D:\src\scipysrc\seaborn\tests\test_distributions.py`

```
import itertools
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_rgba

import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from seaborn import distributions as dist
from seaborn.palettes import (
    color_palette,
    light_palette,
)
from seaborn._base import (
    categorical_order,
)
from seaborn._statistics import (
    KDE,
    Histogram,
    _no_scipy,
)
from seaborn.distributions import (
    _DistributionPlotter,
    displot,
    distplot,
    histplot,
    ecdfplot,
    kdeplot,
    rugplot,
)
from seaborn.utils import _version_predates
from seaborn.axisgrid import FacetGrid
from seaborn._testing import (
    assert_plots_equal,
    assert_legends_equal,
    assert_colors_equal,
)


def get_contour_coords(c, filter_empty=False):
    """Provide compatibility for change in contour artist types."""
    if isinstance(c, mpl.collections.LineCollection):
        # See https://github.com/matplotlib/matplotlib/issues/20906
        # 获取线段集合的顶点坐标作为轮廓线的坐标
        return c.get_segments()
    elif isinstance(c, (mpl.collections.PathCollection, mpl.contour.QuadContourSet)):
        # 获取路径集合或者等高线集合中的顶点坐标作为轮廓线的坐标
        # 过滤空轮廓线路径（长度为零），如果启用了过滤空路径选项
        return [
            p.vertices[:np.argmax(p.codes) + 1] for p in c.get_paths()
            if len(p) or not filter_empty
        ]


def get_contour_color(c):
    """Provide compatibility for change in contour artist types."""
    if isinstance(c, mpl.collections.LineCollection):
        # See https://github.com/matplotlib/matplotlib/issues/20906
        # 获取线段集合的颜色作为轮廓线的颜色
        return c.get_color()
    elif isinstance(c, (mpl.collections.PathCollection, mpl.contour.QuadContourSet)):
        # 获取路径集合或者等高线集合的填充颜色或边缘颜色作为轮廓线的颜色
        if c.get_facecolor().size:
            return c.get_facecolor()
        else:
            return c.get_edgecolor()


class TestDistPlot:

    rs = np.random.RandomState(0)
    x = rs.randn(100)

    def test_hist_bins(self):
        # 使用 Freedman-Diaconis 准则计算直方图的边界
        fd_edges = np.histogram_bin_edges(self.x, "fd")
        # 捕获 UserWarning 异常，以确保正确的直方图绘制
        with pytest.warns(UserWarning):
            ax = distplot(self.x)
        # 验证每个直方图条形的位置与边界的近似匹配
        for edge, bar in zip(fd_edges, ax.patches):
            assert pytest.approx(edge) == bar.get_x()

        # 关闭绘图的图像以释放资源
        plt.close(ax.figure)

        # 使用指定的 bin 数量计算直方图的边界
        n = 25
        n_edges = np.histogram_bin_edges(self.x, n)
        # 再次捕获 UserWarning 异常，确保正确的直方图绘制
        with pytest.warns(UserWarning):
            ax = distplot(self.x, bins=n)
        # 验证每个直方图条形的位置与边界的近似匹配
        for edge, bar in zip(n_edges, ax.patches):
            assert pytest.approx(edge) == bar.get_x()
    # 定义测试函数，用于测试 distplot 函数的不同参数组合
    def test_elements(self):

        # 在测试中捕获 UserWarning 警告
        with pytest.warns(UserWarning):

            # 设置 bins 数量为 10
            n = 10
            # 调用 distplot 绘制直方图，不显示密度曲线和数据点
            ax = distplot(self.x, bins=n,
                          hist=True, kde=False, rug=False, fit=None)
            # 断言直方图中的矩形条数等于 bins 数量
            assert len(ax.patches) == 10
            # 断言图中没有绘制任何线条
            assert len(ax.lines) == 0
            # 断言图中没有集合对象
            assert len(ax.collections) == 0

            # 关闭当前图形
            plt.close(ax.figure)
            # 绘制密度曲线，不显示直方图和数据点
            ax = distplot(self.x,
                          hist=False, kde=True, rug=False, fit=None)
            # 断言图中没有矩形条
            assert len(ax.patches) == 0
            # 断言图中绘制了一条线条（密度曲线）
            assert len(ax.lines) == 1
            # 断言图中没有集合对象
            assert len(ax.collections) == 0

            # 关闭当前图形
            plt.close(ax.figure)
            # 绘制数据点，不显示直方图和密度曲线
            ax = distplot(self.x,
                          hist=False, kde=False, rug=True, fit=None)
            # 断言图中没有矩形条
            assert len(ax.patches) == 0
            # 断言图中没有绘制任何线条
            assert len(ax.lines) == 0
            # 断言图中绘制了一个集合对象（数据点）
            assert len(ax.collections) == 1

            # 定义一个类 Norm，模拟一个看起来像 scipy RV 的虚拟对象
            class Norm:
                """Dummy object that looks like a scipy RV"""
                # 定义 fit 方法，返回空元组
                def fit(self, x):
                    return ()

                # 定义 pdf 方法，返回和 x 同样大小的零数组
                def pdf(self, x, *params):
                    return np.zeros_like(x)

            # 关闭当前图形
            plt.close(ax.figure)
            # 调用 distplot 绘制密度曲线，不显示直方图、数据点和数据点
            ax = distplot(
                self.x, hist=False, kde=False, rug=False, fit=Norm())
            # 断言图中没有矩形条
            assert len(ax.patches) == 0
            # 断言图中绘制了一条线条（密度曲线）
            assert len(ax.lines) == 1
            # 断言图中没有集合对象
            assert len(ax.collections) == 0

    # 定义测试函数，测试 distplot 处理包含 NaN 值的数组情况
    def test_distplot_with_nans(self):

        # 创建包含两个子图的图形 f，并获取这两个子图 ax1 和 ax2
        f, (ax1, ax2) = plt.subplots(2)
        # 在 self.x 数组末尾添加一个 NaN 值，创建 x_null 数组
        x_null = np.append(self.x, [np.nan])

        # 在测试中捕获 UserWarning 警告
        with pytest.warns(UserWarning):
            # 在第一个子图 ax1 上绘制 self.x 的密度曲线
            distplot(self.x, ax=ax1)
            # 在第二个子图 ax2 上绘制 x_null 的密度曲线
            distplot(x_null, ax=ax2)

        # 获取第一个子图 ax1 上的第一条线条对象 line1
        line1 = ax1.lines[0]
        # 获取第二个子图 ax2 上的第一条线条对象 line2
        line2 = ax2.lines[0]
        # 断言 line1 和 line2 的数据点数组完全相同
        assert np.array_equal(line1.get_xydata(), line2.get_xydata())

        # 对比第一个子图 ax1 和第二个子图 ax2 上对应位置的矩形条
        for bar1, bar2 in zip(ax1.patches, ax2.patches):
            # 断言两个矩形条的顶点坐标相同
            assert bar1.get_xy() == bar2.get_xy()
            # 断言两个矩形条的高度相同
            assert bar1.get_height() == bar2.get_height()
class SharedAxesLevelTests:

    # 测试颜色设置
    def test_color(self, long_df, **kwargs):
        # 创建一个包含子图的新图形
        ax = plt.figure().subplots()
        # 使用指定数据和参数绘制图形
        self.func(data=long_df, x="y", ax=ax, **kwargs)
        # 断言最后一个颜色与预期颜色相同（不检查透明度）
        assert_colors_equal(self.get_last_color(ax, **kwargs), "C0", check_alpha=False)

        # 创建一个包含子图的新图形
        ax = plt.figure().subplots()
        # 使用指定数据和参数绘制图形
        self.func(data=long_df, x="y", ax=ax, **kwargs)
        # 再次使用相同数据和参数绘制图形
        self.func(data=long_df, x="y", ax=ax, **kwargs)
        # 断言最后一个颜色与预期颜色相同（不检查透明度）
        assert_colors_equal(self.get_last_color(ax, **kwargs), "C1", check_alpha=False)

        # 创建一个包含子图的新图形
        ax = plt.figure().subplots()
        # 使用指定数据、x轴和指定颜色绘制图形
        self.func(data=long_df, x="y", color="C2", ax=ax, **kwargs)
        # 断言最后一个颜色与预期颜色相同（不检查透明度）
        assert_colors_equal(self.get_last_color(ax, **kwargs), "C2", check_alpha=False)


class TestRugPlot(SharedAxesLevelTests):

    # 使用静态方法 rugplot 作为 func 属性
    func = staticmethod(rugplot)

    # 返回最后一个图集合的颜色
    def get_last_color(self, ax, **kwargs):
        return ax.collections[-1].get_color()

    # 断言两个图集合相等
    def assert_rug_equal(self, a, b):
        assert_array_equal(a.get_segments(), b.get_segments())

    # 使用参数化测试，测试长数据
    @pytest.mark.parametrize("variable", ["x", "y"])
    def test_long_data(self, long_df, variable):
        # 获取变量的向量数据
        vector = long_df[variable]
        vectors = [
            variable, vector, np.asarray(vector), vector.to_list(),
        ]

        # 创建新图形和子图
        f, ax = plt.subplots()
        for vector in vectors:
            # 绘制 rugplot
            rugplot(data=long_df, **{variable: vector})

        # 对每对子图集合执行断言，断言它们相等
        for a, b in itertools.product(ax.collections, ax.collections):
            self.assert_rug_equal(a, b)

    # 测试双变量数据
    def test_bivariate_data(self, long_df):
        # 创建一个具有两个子图的新图形
        f, (ax1, ax2) = plt.subplots(ncols=2)

        # 在第一个子图上绘制 rugplot
        rugplot(data=long_df, x="x", y="y", ax=ax1)
        # 在第二个子图上绘制 rugplot
        rugplot(data=long_df, x="x", ax=ax2)
        # 在第二个子图上绘制 rugplot
        rugplot(data=long_df, y="y", ax=ax2)

        # 断言第一个子图的第一个集合与第二个子图的第一个集合相等
        self.assert_rug_equal(ax1.collections[0], ax2.collections[0])
        # 断言第一个子图的第二个集合与第二个子图的第二个集合相等
        self.assert_rug_equal(ax1.collections[1], ax2.collections[1])

    # 测试宽数据和长数据之间的比较
    def test_wide_vs_long_data(self, wide_df):
        # 创建具有两个子图的新图形
        f, (ax1, ax2) = plt.subplots(ncols=2)
        # 在第一个子图上绘制 rugplot
        rugplot(data=wide_df, ax=ax1)
        # 对宽数据的每列绘制 rugplot 并放置在第二个子图中
        for col in wide_df:
            rugplot(data=wide_df, x=col, ax=ax2)

        # 获取宽数据子图的集合段并排序
        wide_segments = np.sort(
            np.array(ax1.collections[0].get_segments())
        )
        # 获取长数据子图的所有集合段并排序
        long_segments = np.sort(
            np.concatenate([c.get_segments() for c in ax2.collections])
        )

        # 断言宽数据和长数据集合段相等
        assert_array_equal(wide_segments, long_segments)

    # 测试扁平向量数据
    def test_flat_vector(self, long_df):
        # 创建新图形和子图
        f, ax = plt.subplots()
        # 绘制 rugplot
        rugplot(data=long_df["x"])
        # 使用指定数据绘制 rugplot
        rugplot(x=long_df["x"])
        # 断言两个子图集合相等
        self.assert_rug_equal(*ax.collections)

    # 测试日期时间数据
    def test_datetime_data(self, long_df):
        # 绘制 rugplot 并获取子图
        ax = rugplot(data=long_df["t"])
        # 获取子图的所有集合段的第一维度
        vals = np.stack(ax.collections[0].get_segments())[:, 0, 0]
        # 断言值与日期时间转换后的长数据相等
        assert_array_equal(vals, mpl.dates.date2num(long_df["t"]))

    # 测试空数据
    def test_empty_data(self):
        # 绘制 rugplot 并获取子图
        ax = rugplot(x=[])
        # 断言子图不包含集合
        assert not ax.collections
    # 测试用例：检查 rugplot 在使用过时功能时是否会触发 UserWarning
    def test_a_deprecation(self, flat_series):
        # 创建一个新的图形和坐标轴
        f, ax = plt.subplots()

        # 使用 pytest 来捕获 UserWarning 异常
        with pytest.warns(UserWarning):
            # 调用 rugplot 函数并生成 rug 图
            rugplot(a=flat_series)
        # 生成另一个 rug 图，不应触发警告
        rugplot(x=flat_series)

        # 使用断言检查两个图的集合是否相同
        self.assert_rug_equal(*ax.collections)

    # 参数化测试：检查 rugplot 在使用过时轴参数时是否会触发 UserWarning
    @pytest.mark.parametrize("variable", ["x", "y"])
    def test_axis_deprecation(self, flat_series, variable):
        # 创建一个新的图形和坐标轴
        f, ax = plt.subplots()

        # 使用 pytest 来捕获 UserWarning 异常
        with pytest.warns(UserWarning):
            # 调用 rugplot 函数并生成 rug 图
            rugplot(flat_series, axis=variable)
        # 使用变量参数调用 rugplot 函数，不应触发警告
        rugplot(**{variable: flat_series})

        # 使用断言检查两个图的集合是否相同
        self.assert_rug_equal(*ax.collections)

    # 测试用例：检查 rugplot 在使用过时垂直参数时是否会触发 UserWarning
    def test_vertical_deprecation(self, flat_series):
        # 创建一个新的图形和坐标轴
        f, ax = plt.subplots()

        # 使用 pytest 来捕获 UserWarning 异常
        with pytest.warns(UserWarning):
            # 调用 rugplot 函数并生成垂直的 rug 图
            rugplot(flat_series, vertical=True)
        # 生成水平的 rug 图，不应触发警告
        rugplot(y=flat_series)

        # 使用断言检查两个图的集合是否相同
        self.assert_rug_equal(*ax.collections)

    # 测试用例：检查 rugplot 在生成 rug 数据时的正确性
    def test_rug_data(self, flat_array):
        # 设置 rug 的高度
        height = .05
        # 调用 rugplot 函数并生成 rug 图
        ax = rugplot(x=flat_array, height=height)
        # 获取 rug 图的线段
        segments = np.stack(ax.collections[0].get_segments())

        # 获取数组大小
        n = flat_array.size
        # 使用断言检查 rug 图的线段是否正确生成
        assert_array_equal(segments[:, 0, 1], np.zeros(n))
        assert_array_equal(segments[:, 1, 1], np.full(n, height))
        assert_array_equal(segments[:, 1, 0], flat_array)

    # 测试用例：检查 rugplot 在使用颜色参数时的正确性
    def test_rug_colors(self, long_df):
        # 调用 rugplot 函数并生成 rug 图
        ax = rugplot(data=long_df, x="x", hue="a")

        # 获取分类顺序和颜色调色板
        order = categorical_order(long_df["a"])
        palette = color_palette()

        # 生成期望的颜色数组
        expected_colors = np.ones((len(long_df), 4))
        for i, val in enumerate(long_df["a"]):
            expected_colors[i, :3] = palette[order.index(val)]

        # 使用断言检查 rug 图的颜色是否正确生成
        assert_array_equal(ax.collections[0].get_color(), expected_colors)

    # 测试用例：检查 rugplot 在不扩展边距时的边距控制功能
    def test_expand_margins(self, flat_array):
        # 创建一个新的图形和坐标轴，并获取初始边距
        f, ax = plt.subplots()
        x1, y1 = ax.margins()

        # 调用 rugplot 函数并生成 rug 图，不扩展边距
        rugplot(x=flat_array, expand_margins=False)
        # 获取调整后的边距
        x2, y2 = ax.margins()

        # 使用断言检查边距是否未改变
        assert x1 == x2
        assert y1 == y2

        # 创建一个新的图形和坐标轴，并获取初始边距
        f, ax = plt.subplots()
        x1, y1 = ax.margins()

        # 设置 rug 的高度
        height = .05
        # 调用 rugplot 函数并生成 rug 图
        rugplot(x=flat_array, height=height)
        # 获取调整后的边距
        x2, y2 = ax.margins()

        # 使用断言检查边距是否按预期调整
        assert x1 == x2
        assert y1 + height * 2 == pytest.approx(y2)

    # 测试用例：检查 rugplot 在生成多个 rug 图时的正确性
    def test_multiple_rugs(self):
        # 生成均匀分布的数据
        values = np.linspace(start=0, stop=1, num=5)
        # 调用 rugplot 函数并生成 rug 图
        ax = rugplot(x=values)
        # 获取初始的 y 轴限制
        ylim = ax.get_ylim()

        # 使用同一坐标轴生成第二个 rug 图，不扩展边距
        rugplot(x=values, ax=ax, expand_margins=False)

        # 使用断言检查 y 轴限制是否未改变
        assert ylim == ax.get_ylim()

    # 测试用例：检查 rugplot 在使用 matplotlib 参数时的正确性
    def test_matplotlib_kwargs(self, flat_series):
        # 设置线宽和透明度参数
        lw = 2
        alpha = .2
        # 调用 rugplot 函数并生成 rug 图
        ax = rugplot(y=flat_series, linewidth=lw, alpha=alpha)
        # 获取 rug 图的集合
        rug = ax.collections[0]

        # 使用断言检查 rug 图的透明度和线宽是否按预期设置
        assert np.all(rug.get_alpha() == alpha)
        assert np.all(rug.get_linewidth() == lw)

    # 测试用例：检查 rugplot 在生成坐标轴标签时的正确性
    def test_axis_labels(self, flat_series):
        # 调用 rugplot 函数并生成 rug 图
        ax = rugplot(x=flat_series)

        # 使用断言检查 x 轴标签是否正确设置
        assert ax.get_xlabel() == flat_series.name
        # 使用断言检查 y 轴标签是否不存在
        assert not ax.get_ylabel()
    # 定义一个测试函数，用于测试对数刻度的绘图功能
    def test_log_scale(self, long_df):
        # 创建一个包含两个子图的图形对象，返回两个轴对象 ax1 和 ax2
        ax1, ax2 = plt.figure().subplots(2)

        # 设置 ax2 的 x 轴为对数刻度
        ax2.set_xscale("log")

        # 在 ax1 上绘制长数据框 long_df 中 z 列的 rug plot（数据分布展示）
        rugplot(data=long_df, x="z", ax=ax1)
        # 在 ax2 上同样绘制长数据框 long_df 中 z 列的 rug plot
        rugplot(data=long_df, x="z", ax=ax2)

        # 从 ax1 的第一个集合（rug plot 数据的集合）中获取线段，并堆叠成数组 rug1
        rug1 = np.stack(ax1.collections[0].get_segments())
        # 从 ax2 的第一个集合中获取线段，并堆叠成数组 rug2
        rug2 = np.stack(ax2.collections[0].get_segments())

        # 断言 rug1 和 rug2 几乎相等，用于测试两个 rug plot 是否相同
        assert_array_almost_equal(rug1, rug2)
class TestKDEPlotUnivariate(SharedAxesLevelTests):
    # 测试类继承自 SharedAxesLevelTests，用于测试 KDE 绘图功能

    func = staticmethod(kdeplot)
    # 设置 func 属性为 kdeplot 静态方法，用于调用 KDE 绘图函数

    def get_last_color(self, ax, fill=True):
        # 定义获取最后一个图形对象颜色的方法，参数 ax 是绘图对象，fill 表示是否填充颜色

        if fill:
            # 如果 fill 参数为 True
            return ax.collections[-1].get_facecolor()
            # 返回最后一个集合对象（通常是填充区域）的面颜色
        else:
            # 如果 fill 参数为 False
            return ax.lines[-1].get_color()
            # 返回最后一条线的颜色

    @pytest.mark.parametrize("fill", [True, False])
    # 使用 pytest 的参数化装饰器，测试 fill 参数为 True 和 False 的情况
    def test_color(self, long_df, fill):
        # 颜色测试方法，测试 KDE 绘图的颜色设置

        super().test_color(long_df, fill=fill)
        # 调用父类的 test_color 方法进行基本测试

        if fill:
            # 如果 fill 参数为 True

            ax = plt.figure().subplots()
            # 创建一个子图对象 ax

            self.func(data=long_df, x="y", facecolor="C3", fill=True, ax=ax)
            # 调用 func 方法绘制 KDE 图，设置面颜色为 "C3"，填充为 True

            assert_colors_equal(self.get_last_color(ax), "C3", check_alpha=False)
            # 使用 assert 检查最后一个图形对象的颜色是否为 "C3"，不检查 alpha 通道

            ax = plt.figure().subplots()
            # 创建另一个子图对象 ax

            self.func(data=long_df, x="y", fc="C4", fill=True, ax=ax)
            # 再次调用 func 方法绘制 KDE 图，设置面颜色为 "C4"，填充为 True

            assert_colors_equal(self.get_last_color(ax), "C4", check_alpha=False)
            # 使用 assert 检查最后一个图形对象的颜色是否为 "C4"，不检查 alpha 通道

    @pytest.mark.parametrize(
        "variable", ["x", "y"],
    )
    # 使用 pytest 参数化装饰器，测试 variable 参数为 "x" 和 "y" 的情况
    def test_long_vectors(self, long_df, variable):
        # 测试长向量情况的方法

        vector = long_df[variable]
        # 获取数据框 long_df 中变量 variable 的向量数据

        vectors = [
            variable, vector, vector.to_numpy(), vector.to_list(),
        ]
        # 构建多种向量形式的列表 vectors

        f, ax = plt.subplots()
        # 创建图形对象 f 和子图对象 ax

        for vector in vectors:
            kdeplot(data=long_df, **{variable: vector})
            # 对每种向量形式调用 kdeplot 绘制 KDE 图

        xdata = [l.get_xdata() for l in ax.lines]
        # 获取所有线条对象的 x 数据

        for a, b in itertools.product(xdata, xdata):
            assert_array_equal(a, b)
            # 使用 assert 检查所有线条对象的 x 数据是否一致

        ydata = [l.get_ydata() for l in ax.lines]
        # 获取所有线条对象的 y 数据

        for a, b in itertools.product(ydata, ydata):
            assert_array_equal(a, b)
            # 使用 assert 检查所有线条对象的 y 数据是否一致

    def test_wide_vs_long_data(self, wide_df):
        # 测试宽数据和长数据对比的方法

        f, (ax1, ax2) = plt.subplots(ncols=2)
        # 创建包含两个子图对象的图形对象 f

        kdeplot(data=wide_df, ax=ax1, common_norm=False, common_grid=False)
        # 在第一个子图 ax1 上绘制 wide_df 的 KDE 图，关闭公共标准化和网格

        for col in wide_df:
            kdeplot(data=wide_df, x=col, ax=ax2)
            # 遍历 wide_df 的每列，在第二个子图 ax2 上分别绘制 KDE 图

        for l1, l2 in zip(ax1.lines[::-1], ax2.lines):
            assert_array_equal(l1.get_xydata(), l2.get_xydata())
            # 使用 assert 检查 ax1 和 ax2 中线条对象的坐标数据是否一致

    def test_flat_vector(self, long_df):
        # 测试扁平向量情况的方法

        f, ax = plt.subplots()
        # 创建图形对象 f 和子图对象 ax

        kdeplot(data=long_df["x"])
        # 在子图 ax 上绘制 long_df 中 "x" 列的 KDE 图

        kdeplot(x=long_df["x"])
        # 在同一个子图 ax 上绘制 long_df 中 "x" 列的 KDE 图

        assert_array_equal(ax.lines[0].get_xydata(), ax.lines[1].get_xydata())
        # 使用 assert 检查第一条和第二条线条对象的坐标数据是否一致

    def test_empty_data(self):
        # 测试空数据情况的方法

        ax = kdeplot(x=[])
        # 在默认子图上绘制一个空数据的 KDE 图

        assert not ax.lines
        # 使用 assert 检查子图上是否没有线条对象

    def test_singular_data(self):
        # 测试单一数据情况的方法

        with pytest.warns(UserWarning):
            ax = kdeplot(x=np.ones(10))
        # 使用 pytest 捕获 UserWarning 警告，绘制一个值为 1 的数组的 KDE 图

        assert not ax.lines
        # 使用 assert 检查子图上是否没有线条对象

        with pytest.warns(UserWarning):
            ax = kdeplot(x=[5])
        # 再次使用 pytest 捕获 UserWarning 警告，绘制一个只包含一个值为 5 的数组的 KDE 图

        assert not ax.lines
        # 使用 assert 检查子图上是否没有线条对象

        with pytest.warns(UserWarning):
            # https://github.com/mwaskom/seaborn/issues/2762
            ax = kdeplot(x=[1929245168.06679] * 18)
        # 一种特殊情况下的 KDE 图绘制，参考 GitHub issue

        assert not ax.lines
        # 使用 assert 检查子图上是否没有线条对象

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # 使用 warnings 模块捕获 UserWarning 警告
            ax = kdeplot(x=[5], warn_singular=False)
            # 绘制一个只包含一个值为 5 的数组的 KDE 图，关闭警告

        assert not ax.lines
        # 使用 assert 检查子图上是否没有线条对象
    # 测试变量赋值功能，使用 long_df 数据集生成一个绘图对象 f 和坐标轴对象 ax
    def test_variable_assignment(self, long_df):

        # 在 ax 上绘制 x 轴密度估计图，并填充区域
        f, ax = plt.subplots()
        kdeplot(data=long_df, x="x", fill=True)
        
        # 在 ax 上绘制 y 轴密度估计图，并填充区域
        kdeplot(data=long_df, y="x", fill=True)

        # 获取 ax.collections 中第一个集合的第一个路径的顶点数据，并赋值给 v0
        v0 = ax.collections[0].get_paths()[0].vertices
        
        # 获取 ax.collections 中第二个集合的第一个路径的顶点数据，并反转列的顺序后赋值给 v1
        v1 = ax.collections[1].get_paths()[0].vertices[:, [1, 0]]

        # 断言 v0 和 v1 的值相等
        assert_array_equal(v0, v1)

    # 测试 vertical 参数的弃用功能，使用 long_df 数据集生成绘图对象 f 和坐标轴对象 ax
    def test_vertical_deprecation(self, long_df):

        # 绘制 y 轴密度估计图，并生成警告信息
        f, ax = plt.subplots()
        kdeplot(data=long_df, y="x")

        with pytest.warns(UserWarning):
            # 绘制 x 轴密度估计图，并设置 vertical=True，触发警告
            kdeplot(data=long_df, x="x", vertical=True)

        # 断言 ax.lines 中第一个线条和第二个线条的数据点相等
        assert_array_equal(ax.lines[0].get_xydata(), ax.lines[1].get_xydata())

    # 测试 bw 参数的弃用功能，使用 long_df 数据集生成绘图对象 f 和坐标轴对象 ax
    def test_bw_deprecation(self, long_df):

        # 绘制 x 轴密度估计图，并设置 bw_method="silverman"
        f, ax = plt.subplots()
        kdeplot(data=long_df, x="x", bw_method="silverman")

        with pytest.warns(UserWarning):
            # 绘制 x 轴密度估计图，并设置 bw="silverman"，触发警告
            kdeplot(data=long_df, x="x", bw="silverman")

        # 断言 ax.lines 中第一个线条和第二个线条的数据点相等
        assert_array_equal(ax.lines[0].get_xydata(), ax.lines[1].get_xydata())

    # 测试 kernel 参数的弃用功能，使用 long_df 数据集生成绘图对象 f 和坐标轴对象 ax
    def test_kernel_deprecation(self, long_df):

        # 绘制 x 轴密度估计图
        f, ax = plt.subplots()
        kdeplot(data=long_df, x="x")

        with pytest.warns(UserWarning):
            # 绘制 x 轴密度估计图，并设置 kernel="epi"，触发警告
            kdeplot(data=long_df, x="x", kernel="epi")

        # 断言 ax.lines 中第一个线条和第二个线条的数据点相等
        assert_array_equal(ax.lines[0].get_xydata(), ax.lines[1].get_xydata())

    # 测试 shade 参数的弃用功能，使用 long_df 数据集生成绘图对象 f 和坐标轴对象 ax
    def test_shade_deprecation(self, long_df):

        # 在绘图对象 f 上创建坐标轴对象 ax，并生成 FutureWarning 警告
        f, ax = plt.subplots()
        with pytest.warns(FutureWarning):
            # 绘制 x 轴密度估计图，并设置 shade=True，触发警告
            kdeplot(data=long_df, x="x", shade=True)
        # 再次绘制 x 轴密度估计图，设置 fill=True
        kdeplot(data=long_df, x="x", fill=True)
        
        # 获取 ax.collections 中的两个填充区域对象
        fill1, fill2 = ax.collections
        
        # 断言两个填充区域对象的顶点数据相等
        assert_array_equal(
            fill1.get_paths()[0].vertices, fill2.get_paths()[0].vertices
        )

    # 测试 hue 参数与颜色处理，使用 long_df 数据集生成绘图对象 ax
    @pytest.mark.parametrize("multiple", ["layer", "stack", "fill"])
    def test_hue_colors(self, long_df, multiple):

        # 绘制 x 轴密度估计图，并根据 hue 列多重染色
        ax = kdeplot(
            data=long_df, x="x", hue="a",
            multiple=multiple,
            fill=True, legend=False
        )

        # 注意：图中 hue 的顺序是反向的
        lines = ax.lines[::-1]
        fills = ax.collections[::-1]

        # 获取默认调色板
        palette = color_palette()

        # 遍历线条和填充区域对象，断言它们的颜色与调色板中的颜色相等
        for line, fill, color in zip(lines, fills, palette):
            assert_colors_equal(line.get_color(), color)
            assert_colors_equal(fill.get_facecolor(), to_rgba(color, .25))

    # 测试 hue 参数与堆叠处理，使用 long_df 数据集生成绘图对象 f 和两个坐标轴对象 ax1, ax2
    def test_hue_stacking(self, long_df):

        # 在 f 对象上创建两个坐标轴对象 ax1, ax2
        f, (ax1, ax2) = plt.subplots(ncols=2)

        # 在 ax1 上绘制 x 轴密度估计图，并根据 hue 列进行图层化，共享网格，不显示图例
        kdeplot(
            data=long_df, x="x", hue="a",
            multiple="layer", common_grid=True,
            legend=False, ax=ax1,
        )
        
        # 在 ax2 上绘制 x 轴密度估计图，并根据 hue 列进行堆叠，不填充区域，不显示图例
        kdeplot(
            data=long_df, x="x", hue="a",
            multiple="stack", fill=False,
            legend=False, ax=ax2,
        )

        # 获取 ax1 和 ax2 中所有线条的 y 数据，分别生成层叠和堆叠密度数据
        layered_densities = np.stack([
            l.get_ydata() for l in ax1.lines
        ])
        stacked_densities = np.stack([
            l.get_ydata() for l in ax2.lines
        ])

        # 断言层叠密度数据的累积和等于堆叠密度数据
        assert_array_equal(layered_densities.cumsum(axis=0), stacked_densities)
    # 定义一个测试方法，用于测试填充曲线的行为，接受一个名为long_df的参数
    def test_hue_filling(self, long_df):

        # 创建一个包含两个子图的绘图对象f，返回两个轴对象ax1和ax2
        f, (ax1, ax2) = plt.subplots(ncols=2)

        # 在第一个子图ax1上绘制长格式数据long_df的核密度估计曲线，按照"a"列分组
        kdeplot(
            data=long_df, x="x", hue="a",
            multiple="layer", common_grid=True,
            legend=False, ax=ax1,
        )
        # 在第二个子图ax2上绘制长格式数据long_df的核密度估计曲线，按照"a"列分组，并填充曲线
        kdeplot(
            data=long_df, x="x", hue="a",
            multiple="fill", fill=False,
            legend=False, ax=ax2,
        )

        # 从ax1和ax2的线对象中提取Y轴数据，并堆叠成数组layered和filled
        layered = np.stack([l.get_ydata() for l in ax1.lines])
        filled = np.stack([l.get_ydata() for l in ax2.lines])

        # 断言layered数组的归一化累积和等于filled数组，用于验证填充曲线的正确性
        assert_array_almost_equal(
            (layered / layered.sum(axis=0)).cumsum(axis=0),
            filled,
        )

    # 使用pytest的参数化装饰器，定义填充参数为"stack"和"fill"的测试方法，接受long_df和multiple参数
    @pytest.mark.parametrize("multiple", ["stack", "fill"])
    def test_fill_default(self, long_df, multiple):

        # 调用kdeplot绘制长格式数据long_df的核密度估计曲线，按照"a"列分组，并返回轴对象ax
        ax = kdeplot(
            data=long_df, x="x", hue="a", multiple=multiple, fill=None
        )

        # 断言轴对象ax的collections属性长度大于0，用于验证是否存在绘图元素
        assert len(ax.collections) > 0

    # 使用pytest的参数化装饰器，定义填充参数为"layer"、"stack"和"fill"的测试方法，接受long_df和multiple参数
    @pytest.mark.parametrize("multiple", ["layer", "stack", "fill"])
    def test_fill_nondefault(self, long_df, multiple):

        # 创建一个包含两个子图的绘图对象f，返回两个轴对象ax1和ax2
        f, (ax1, ax2) = plt.subplots(ncols=2)

        # 定义kdeplot函数的关键字参数字典kws
        kws = dict(data=long_df, x="x", hue="a")
        # 在第一个子图ax1上绘制长格式数据long_df的核密度估计曲线，按照"a"列分组，不填充曲线
        kdeplot(**kws, multiple=multiple, fill=False, ax=ax1)
        # 在第二个子图ax2上绘制长格式数据long_df的核密度估计曲线，按照"a"列分组，并填充曲线
        kdeplot(**kws, multiple=multiple, fill=True, ax=ax2)

        # 断言ax1的collections属性长度为0，用于验证未填充的曲线是否没有绘图元素
        assert len(ax1.collections) == 0
        # 断言ax2的collections属性长度大于0，用于验证填充的曲线是否有绘图元素
        assert len(ax2.collections) > 0

    # 定义一个测试方法，用于测试颜色循环交互的行为，接受一个名为flat_series的参数
    def test_color_cycle_interaction(self, flat_series):

        # 定义一个颜色元组
        color = (.2, 1, .6)

        # 创建一个绘图对象f和轴对象ax
        f, ax = plt.subplots()
        # 绘制flat_series的核密度估计曲线，并返回绘图对象的轴对象
        kdeplot(flat_series)
        # 再次绘制flat_series的核密度估计曲线，并返回绘图对象的轴对象
        kdeplot(flat_series)
        # 断言第一条曲线的颜色为默认的"C0"
        assert_colors_equal(ax.lines[0].get_color(), "C0")
        # 断言第二条曲线的颜色为默认的"C1"
        assert_colors_equal(ax.lines[1].get_color(), "C1")
        # 关闭绘图对象f
        plt.close(f)

        # 创建一个绘图对象f和轴对象ax
        f, ax = plt.subplots()
        # 绘制flat_series的核密度估计曲线，并指定颜色为color
        kdeplot(flat_series, color=color)
        # 再次绘制flat_series的核密度估计曲线，并返回绘图对象的轴对象
        kdeplot(flat_series)
        # 断言第一条曲线的颜色为指定的color
        assert_colors_equal(ax.lines[0].get_color(), color)
        # 断言第二条曲线的颜色为默认的"C0"
        assert_colors_equal(ax.lines[1].get_color(), "C0")
        # 关闭绘图对象f
        plt.close(f)

        # 创建一个绘图对象f和轴对象ax
        f, ax = plt.subplots()
        # 绘制flat_series的填充核密度估计曲线，并返回绘图对象的轴对象
        kdeplot(flat_series, fill=True)
        # 再次绘制flat_series的填充核密度估计曲线，并返回绘图对象的轴对象
        kdeplot(flat_series, fill=True)
        # 断言第一个填充区域的颜色为默认的"C0"的25%透明度
        assert_colors_equal(ax.collections[0].get_facecolor(), to_rgba("C0", .25))
        # 断言第二个填充区域的颜色为默认的"C1"的25%透明度
        assert_colors_equal(ax.collections[1].get_facecolor(), to_rgba("C1", .25))
        # 关闭绘图对象f
        plt.close(f)

    # 使用pytest的参数化装饰器，定义填充参数为True和False的测试方法，接受long_df和fill参数
    @pytest.mark.parametrize("fill", [True, False])
    def test_artist_color(self, long_df, fill):

        # 定义颜色和透明度
        color = (.2, 1, .6)
        alpha = .5

        # 创建一个绘图对象f和轴对象ax
        f, ax = plt.subplots()

        # 绘制long_df["x"]的核密度估计曲线，根据fill参数决定是否填充曲线，并指定颜色为color
        kdeplot(long_df["x"], fill=fill, color=color)
        # 如果填充参数为True，获取最后一个集合对象的填充颜色并扁平化
        if fill:
            artist_color = ax.collections[-1].get_facecolor().squeeze()
        # 如果填充参数为False，获取最后一条线对象的颜色
        else:
            artist_color = ax.lines[-1].get_color()
        # 默认情况下，设置填充的透明度为0.25，否则为1
        default_alpha = .25 if fill else 1
        # 断言填充或线的颜色为指定的color和默认的alpha
        assert_colors_equal(artist_color, to_rgba(color, default_alpha))

        # 再次绘制long_df["x"]的核密度估计曲线，根据fill参数决定是否填充曲线，并指定颜色为color和透明度为alpha
        kdeplot(long_df["x"], fill=fill, color=color, alpha=alpha)
        # 如果填充参数为True，获取最后一个集合对象的填
    # 测试绘制时间尺度的核密度估计图表
    def test_datetime_scale(self, long_df):

        # 创建包含两个子图的图表对象
        f, (ax1, ax2) = plt.subplots(2)
        # 在第一个子图上绘制核密度估计图（填充）
        kdeplot(x=long_df["t"], fill=True, ax=ax1)
        # 在第二个子图上绘制核密度估计图（不填充）
        kdeplot(x=long_df["t"], fill=False, ax=ax2)
        # 断言两个子图的 x 轴范围是否相同
        assert ax1.get_xlim() == ax2.get_xlim()

    # 测试多个参数的检查
    def test_multiple_argument_check(self, long_df):

        # 使用 pytest 检查是否抛出 ValueError 异常，异常信息需要匹配指定的字符串
        with pytest.raises(ValueError, match="`multiple` must be"):
            kdeplot(data=long_df, x="x", hue="a", multiple="bad_input")

    # 测试 cut 参数对核密度估计的影响
    def test_cut(self, rng):

        # 生成服从正态分布的随机数
        x = rng.normal(0, 3, 1000)

        # 创建包含单个子图的图表对象
        f, ax = plt.subplots()
        # 绘制核密度估计图，cut=0，不显示图例
        kdeplot(x=x, cut=0, legend=False)

        # 获取第一条曲线的 x 数据
        xdata_0 = ax.lines[0].get_xdata()
        # 断言第一条曲线的 x 数据范围与原始数据 x 的范围相同
        assert xdata_0.min() == x.min()
        assert xdata_0.max() == x.max()

        # 绘制核密度估计图，cut=2，不显示图例
        kdeplot(x=x, cut=2, legend=False)

        # 获取第二条曲线的 x 数据
        xdata_2 = ax.lines[1].get_xdata()
        # 断言第二条曲线的 x 数据范围比第一条曲线的范围更宽
        assert xdata_2.min() < xdata_0.min()
        assert xdata_2.max() > xdata_0.max()

        # 断言两条曲线的数据点数量相同
        assert len(xdata_0) == len(xdata_2)

    # 测试 clip 参数对核密度估计的影响
    def test_clip(self, rng):

        # 生成服从正态分布的随机数
        x = rng.normal(0, 3, 1000)

        # 定义 clip 参数范围
        clip = -1, 1
        # 绘制核密度估计图，clip 参数限制 x 轴范围
        ax = kdeplot(x=x, clip=clip)

        # 获取第一条曲线的 x 数据
        xdata = ax.lines[0].get_xdata()

        # 断言曲线的 x 数据在 clip 参数范围内
        assert xdata.min() >= clip[0]
        assert xdata.max() <= clip[1]

    # 测试核密度估计是否代表概率密度函数
    def test_line_is_density(self, long_df):

        # 绘制核密度估计图，并获取第一条曲线的 x、y 数据
        ax = kdeplot(data=long_df, x="x", cut=5)
        x, y = ax.lines[0].get_xydata().T
        # 断言曲线的面积积分结果接近于 1
        assert integrate(y, x) == pytest.approx(1)

    # 当 scipy 不可用时，跳过测试
    @pytest.mark.skipif(_no_scipy, reason="Test requires scipy")
    def test_cumulative(self, long_df):

        # 绘制累积核密度估计图，并获取第一条曲线的 y 数据
        ax = kdeplot(data=long_df, x="x", cut=5, cumulative=True)
        y = ax.lines[0].get_ydata()
        # 断言曲线的起始点和终点分别接近于 0 和 1
        assert y[0] == pytest.approx(0)
        assert y[-1] == pytest.approx(1)

    # 当 scipy 可用时，跳过测试
    @pytest.mark.skipif(not _no_scipy, reason="Test requires scipy's absence")
    def test_cumulative_requires_scipy(self, long_df):

        # 使用 pytest 断言当 scipy 可用时会抛出 RuntimeError 异常
        with pytest.raises(RuntimeError):
            kdeplot(data=long_df, x="x", cut=5, cumulative=True)

    # 测试 common_norm 参数对核密度估计的影响
    def test_common_norm(self, long_df):

        # 创建包含两个子图的图表对象
        f, (ax1, ax2) = plt.subplots(ncols=2)

        # 绘制共同归一化的核密度估计图在第一个子图上
        kdeplot(
            data=long_df, x="x", hue="c", common_norm=True, cut=10, ax=ax1
        )
        # 绘制非共同归一化的核密度估计图在第二个子图上
        kdeplot(
            data=long_df, x="x", hue="c", common_norm=False, cut=10, ax=ax2
        )

        # 计算第一个子图中所有曲线的面积总和
        total_area = 0
        for line in ax1.lines:
            xdata, ydata = line.get_xydata().T
            total_area += integrate(ydata, xdata)
        # 断言第一个子图中所有曲线的面积总和接近于 1
        assert total_area == pytest.approx(1)

        # 断言第二个子图中每条曲线的面积积分结果都接近于 1
        for line in ax2.lines:
            xdata, ydata = line.get_xydata().T
            assert integrate(ydata, xdata) == pytest.approx(1)
    # 测试常规网格（不共享轴）的情况
    def test_common_grid(self, long_df):
        # 创建包含两个子图的图像对象
        f, (ax1, ax2) = plt.subplots(ncols=2)

        # 定义分类顺序
        order = "a", "b", "c"

        # 在第一个子图上绘制 KDE 图，不共享网格轴
        kdeplot(
            data=long_df, x="x", hue="a", hue_order=order,
            common_grid=False, cut=0, ax=ax1,
        )
        # 在第二个子图上绘制 KDE 图，共享网格轴
        kdeplot(
            data=long_df, x="x", hue="a", hue_order=order,
            common_grid=True, cut=0, ax=ax2,
        )

        # 对第一个子图的每条线进行断言，确保其 x 范围符合预期
        for line, level in zip(ax1.lines[::-1], order):
            xdata = line.get_xdata()
            assert xdata.min() == long_df.loc[long_df["a"] == level, "x"].min()
            assert xdata.max() == long_df.loc[long_df["a"] == level, "x"].max()

        # 对第二个子图的每条线进行断言，确保其 x 范围符合预期
        for line in ax2.lines:
            xdata = line.get_xdata().T
            assert xdata.min() == long_df["x"].min()
            assert xdata.max() == long_df["x"].max()

    # 测试 KDE 图的带宽方法
    def test_bw_method(self, long_df):
        # 创建包含单个子图的图像对象
        f, ax = plt.subplots()
        # 绘制三个不同带宽的 KDE 图，并关闭图例
        kdeplot(data=long_df, x="x", bw_method=0.2, legend=False)
        kdeplot(data=long_df, x="x", bw_method=1.0, legend=False)
        kdeplot(data=long_df, x="x", bw_method=3.0, legend=False)

        # 获取第一个、第二个、第三个线条对象
        l1, l2, l3 = ax.lines

        # 对带宽不同的三条曲线进行断言，确保其斜率绝对值的均值递增
        assert (
            np.abs(np.diff(l1.get_ydata())).mean()
            > np.abs(np.diff(l2.get_ydata())).mean()
        )

        assert (
            np.abs(np.diff(l2.get_ydata())).mean()
            > np.abs(np.diff(l3.get_ydata())).mean()
        )

    # 测试 KDE 图的带宽调整因子
    def test_bw_adjust(self, long_df):
        # 创建包含单个子图的图像对象
        f, ax = plt.subplots()
        # 绘制三个不同带宽调整因子的 KDE 图，并关闭图例
        kdeplot(data=long_df, x="x", bw_adjust=0.2, legend=False)
        kdeplot(data=long_df, x="x", bw_adjust=1.0, legend=False)
        kdeplot(data=long_df, x="x", bw_adjust=3.0, legend=False)

        # 获取第一个、第二个、第三个线条对象
        l1, l2, l3 = ax.lines

        # 对带宽调整因子不同的三条曲线进行断言，确保其斜率绝对值的均值递增
        assert (
            np.abs(np.diff(l1.get_ydata())).mean()
            > np.abs(np.diff(l2.get_ydata())).mean()
        )

        assert (
            np.abs(np.diff(l2.get_ydata())).mean()
            > np.abs(np.diff(l3.get_ydata())).mean()
        )

    # 测试隐式对数尺度的情况下的 KDE 图
    def test_log_scale_implicit(self, rng):
        # 生成服从对数正态分布的随机数据
        x = rng.lognormal(0, 1, 100)

        # 创建包含两个子图的图像对象
        f, (ax1, ax2) = plt.subplots(ncols=2)

        # 在第一个子图上设置 x 轴为对数尺度，并绘制两次 KDE 图
        ax1.set_xscale("log")
        kdeplot(x=x, ax=ax1)
        kdeplot(x=x, ax=ax1)

        # 获取第一个子图第一条线条的 x 数据
        xdata_log = ax1.lines[0].get_xdata()

        # 对第一条线条的 x 数据进行断言，确保其全部大于 0
        assert (xdata_log > 0).all()
        # 对第一条线条的 x 数据进行断言，确保其二阶差分大于 0
        assert (np.diff(xdata_log, 2) > 0).all()
        # 对第一条线条的 x 数据进行断言，确保其对数变换后的二阶差分接近 0
        assert np.allclose(np.diff(np.log(xdata_log), 2), 0)

        # 创建包含单个子图的图像对象
        f, ax = plt.subplots()
        # 设置 y 轴为对数尺度，并绘制 KDE 图
        ax.set_yscale("log")
        kdeplot(y=x, ax=ax)
        # 断言第一条线条的 x 数据与第一个子图第一条线条的 y 数据相等
        assert_array_equal(ax.lines[0].get_xdata(), ax1.lines[0].get_ydata())
    # 测试对数刻度在不同情况下的显式设置
    def test_log_scale_explicit(self, rng):
        # 生成服从对数正态分布的随机数数组
        x = rng.lognormal(0, 1, 100)

        # 创建包含三个子图的图形对象
        f, (ax1, ax2, ax3) = plt.subplots(ncols=3)

        # 在第一个子图上设置 x 轴为对数刻度，并绘制核密度估计图
        ax1.set_xscale("log")
        kdeplot(x=x, ax=ax1)

        # 在第二个子图上绘制对数刻度的核密度估计图
        kdeplot(x=x, log_scale=True, ax=ax2)

        # 在第三个子图上绘制基底为 10 的对数刻度的核密度估计图
        kdeplot(x=x, log_scale=10, ax=ax3)

        # 检查每个子图的 x 轴刻度是否为对数刻度
        for ax in f.axes:
            assert ax.get_xscale() == "log"

        # 收集每个子图的支持区间（x 轴数据）
        supports = [ax.lines[0].get_xdata() for ax in f.axes]
        # 检查所有子图的支持区间是否一致
        for a, b in itertools.product(supports, supports):
            assert_array_equal(a, b)

        # 收集每个子图的密度估计值（y 轴数据）
        densities = [ax.lines[0].get_ydata() for ax in f.axes]
        # 检查所有子图的密度估计值是否一致
        for a, b in itertools.product(densities, densities):
            assert_array_equal(a, b)

        # 创建仅包含一个子图的图形对象
        f, ax = plt.subplots()
        # 绘制对数刻度的核密度估计图
        kdeplot(y=x, log_scale=True, ax=ax)
        # 检查子图的 y 轴刻度是否为对数刻度
        assert ax.get_yscale() == "log"

    # 测试带有色调的对数刻度设置
    def test_log_scale_with_hue(self, rng):
        # 生成两组服从对数正态分布的随机数数据
        data = rng.lognormal(0, 1, 50), rng.lognormal(0, 2, 100)
        # 绘制带有对数刻度和公共网格的核密度估计图
        ax = kdeplot(data=data, log_scale=True, common_grid=True)
        # 检查两条线的 x 轴数据是否一致
        assert_array_equal(ax.lines[0].get_xdata(), ax.lines[1].get_xdata())

    # 测试带有对数刻度的核密度估计与归一化
    def test_log_scale_normalization(self, rng):
        # 生成服从对数正态分布的随机数数组
        x = rng.lognormal(0, 1, 100)
        # 绘制带有对数刻度和截断参数的核密度估计图
        ax = kdeplot(x=x, log_scale=True, cut=10)
        # 获取绘制的曲线数据点坐标
        xdata, ydata = ax.lines[0].get_xydata().T
        # 计算密度估计曲线下的积分并检查是否近似等于 1
        integral = integrate(ydata, np.log10(xdata))
        assert integral == pytest.approx(1)

    # 测试权重参数对核密度估计的影响
    def test_weights(self):
        # 定义 x 和 weights 数组
        x = [1, 2]
        weights = [2, 1]
        # 绘制带有权重参数的核密度估计图
        ax = kdeplot(x=x, weights=weights, bw_method=.1)
        # 获取绘制的曲线数据点坐标
        xdata, ydata = ax.lines[0].get_xydata().T
        # 根据 xdata 找到最接近 1 和 2 的 ydata 值
        y1 = ydata[np.abs(xdata - 1).argmin()]
        y2 = ydata[np.abs(xdata - 2).argmin()]
        # 检查权重为 2 时，y1 是否近似等于 2 * y2
        assert y1 == pytest.approx(2 * y2)

    # 测试带有权重和归一化参数的核密度估计
    def test_weight_norm(self, rng):
        # 生成服从正态分布的随机数数组
        vals = rng.normal(0, 1, 50)
        # 构造带有权重和归一化参数的随机数数组和权重数组
        x = np.concatenate([vals, vals])
        w = np.repeat([1, 2], 50)
        # 绘制带有权重和归一化参数的核密度估计图
        ax = kdeplot(x=x, weights=w, hue=w, common_norm=True)
        # 获取绘制的曲线数据点坐标
        x1, y1 = ax.lines[0].get_xydata().T
        x2, y2 = ax.lines[1].get_xydata().T
        # 检查权重为 2 时，两条曲线的积分是否近似等于归一化参数 2 倍的权重为 1 时的积分
        assert integrate(y1, x1) == pytest.approx(2 * integrate(y2, x2))

    # 测试核密度估计中的边缘效应
    def test_sticky_edges(self, long_df):
        # 创建包含两个子图的图形对象
        f, (ax1, ax2) = plt.subplots(ncols=2)

        # 绘制带有填充效果的核密度估计图（单变量）
        kdeplot(data=long_df, x="x", fill=True, ax=ax1)
        # 检查第一个子图的填充对象的粘性边缘
        assert ax1.collections[0].sticky_edges.y[:] == [0, np.inf]

        # 绘制带有色调和填充效果的核密度估计图（多变量）
        kdeplot(data=long_df, x="x", hue="a", multiple="fill", fill=True, ax=ax2)
        # 检查第二个子图的填充对象的粘性边缘
        assert ax2.collections[0].sticky_edges.y[:] == [0, 1]

    # 测试核密度估计中的线条样式参数
    def test_line_kws(self, flat_array):
        # 定义线宽和颜色参数
        lw = 3
        color = (.2, .5, .8)
        # 绘制带有自定义线条样式的核密度估计图
        ax = kdeplot(x=flat_array, linewidth=lw, color=color)
        # 获取绘制的曲线对象
        line, = ax.lines
        # 检查曲线的线宽是否与指定的线宽一致
        assert line.get_linewidth() == lw
        # 检查曲线的颜色是否与指定的颜色一致
        assert_colors_equal(line.get_color(), color)

    # 测试输入参数检查功能
    def test_input_checking(self, long_df):
        # 准备错误信息字符串
        err = "The x variable is categorical,"
        # 使用 pytest 检查输入参数中的错误情况
        with pytest.raises(TypeError, match=err):
            kdeplot(data=long_df, x="a")
    # 测试函数：检查 KDE 图的坐标轴标签设置是否正确
    def test_axis_labels(self, long_df):

        # 创建包含两个子图的画布对象
        f, (ax1, ax2) = plt.subplots(ncols=2)

        # 在第一个子图上绘制 KDE 图，以 x 作为变量
        kdeplot(data=long_df, x="x", ax=ax1)
        # 断言第一个子图的 x 轴标签是否为 "x"
        assert ax1.get_xlabel() == "x"
        # 断言第一个子图的 y 轴标签是否为 "Density"
        assert ax1.get_ylabel() == "Density"

        # 在第二个子图上绘制 KDE 图，以 y 作为变量
        kdeplot(data=long_df, y="y", ax=ax2)
        # 断言第二个子图的 x 轴标签是否为 "Density"
        assert ax2.get_xlabel() == "Density"
        # 断言第二个子图的 y 轴标签是否为 "y"

    # 测试函数：检查 KDE 图的图例设置是否正确
    def test_legend(self, long_df):

        # 绘制带图例的 KDE 图，以 x 作为变量，hue 为 "a"
        ax = kdeplot(data=long_df, x="x", hue="a")

        # 断言图例的标题是否为 "a"
        assert ax.legend_.get_title().get_text() == "a"

        # 获取图例中的标签文本对象
        legend_labels = ax.legend_.get_texts()
        # 获取分类顺序
        order = categorical_order(long_df["a"])
        # 遍历标签和顺序，断言标签文本是否与分类顺序相匹配
        for label, level in zip(legend_labels, order):
            assert label.get_text() == level

        # 获取图例中的艺术家对象（线条对象）
        legend_artists = ax.legend_.findobj(mpl.lines.Line2D)
        # 如果 Matplotlib 版本早于 3.5.0b0，调整艺术家对象列表
        if _version_predates(mpl, "3.5.0b0"):
            legend_artists = legend_artists[::2]
        
        # 获取当前调色板
        palette = color_palette()
        # 遍历艺术家对象和调色板，断言艺术家对象的颜色是否与调色板相匹配
        for artist, color in zip(legend_artists, palette):
            assert_colors_equal(artist.get_color(), color)

        # 清除图形对象
        ax.clear()

        # 绘制不带图例的 KDE 图，以 x 作为变量，hue 为 "a"
        kdeplot(data=long_df, x="x", hue="a", legend=False)

        # 断言图例对象是否为空
        assert ax.legend_ is None

    # 测试函数：检查替代参数的处理是否引发了预期的异常
    def test_replaced_kws(self, long_df):
        # 使用 pytest 检查是否引发了 TypeError，并匹配特定的错误消息
        with pytest.raises(TypeError, match=r"`data2` has been removed"):
            kdeplot(data=long_df, x="x", data2="y")
# 定义一个测试类 TestKDEPlotBivariate，用于测试 KDEPlot 双变量核密度估计功能
class TestKDEPlotBivariate:

    # 测试长向量情况下的核密度估计
    def test_long_vectors(self, long_df):
        
        # 绘制长数据框 long_df 的核密度估计图，获取绘图对象 ax1
        ax1 = kdeplot(data=long_df, x="x", y="y")

        # 获取 long_df 数据框中的 x 列
        x = long_df["x"]
        # 构造多种格式的 x 值：原始 Series 对象、转换为 NumPy 数组、转换为 Python 列表
        x_values = [x, x.to_numpy(), x.to_list()]

        # 获取 long_df 数据框中的 y 列
        y = long_df["y"]
        # 构造多种格式的 y 值：原始 Series 对象、转换为 NumPy 数组、转换为 Python 列表
        y_values = [y, y.to_numpy(), y.to_list()]

        # 遍历 x_values 和 y_values 中的值对
        for x, y in zip(x_values, y_values):
            # 创建一个新的图形 f 和轴对象 ax2
            f, ax2 = plt.subplots()
            # 绘制 x 和 y 的核密度估计图到 ax2 上
            kdeplot(x=x, y=y, ax=ax2)

            # 遍历 ax1 和 ax2 的图形集合，并断言它们的偏移相等
            for c1, c2 in zip(ax1.collections, ax2.collections):
                assert_array_equal(c1.get_offsets(), c2.get_offsets())

    # 测试单一数据情况下的核密度估计
    def test_singular_data(self):

        # 测试当 x 是全 1 数组，y 是 0 到 9 的数组时的核密度估计，预期会有 UserWarning 警告
        with pytest.warns(UserWarning):
            ax = dist.kdeplot(x=np.ones(10), y=np.arange(10))
        # 断言 ax 中没有线条绘制
        assert not ax.lines

        # 测试当 x 和 y 都是单一值的情况时的核密度估计，预期会有 UserWarning 警告
        with pytest.warns(UserWarning):
            ax = dist.kdeplot(x=[5], y=[6])
        # 断言 ax 中没有线条绘制
        assert not ax.lines

        # 测试当 x 是重复值数组，y 是 0 到 17 的数组时的核密度估计，预期会有 UserWarning 警告
        with pytest.warns(UserWarning):
            ax = kdeplot(x=[1929245168.06679] * 18, y=np.arange(18))
        # 断言 ax 中没有线条绘制
        assert not ax.lines

        # 使用 warnings.catch_warnings() 来捕获特定类型的警告，此处为 UserWarning
        with warnings.catch_warnings():
            # 设置简单过滤器，将 UserWarning 错误转换为异常
            warnings.simplefilter("error", UserWarning)
            # 测试当 x 和 y 都是单一值的情况时的核密度估计，但关闭单一数据警告
            ax = kdeplot(x=[5], y=[7], warn_singular=False)
        # 断言 ax 中没有线条绘制
        assert not ax.lines

    # 测试填充图形的艺术家
    def test_fill_artists(self, long_df):

        # 遍历 True 和 False 两种填充状态
        for fill in [True, False]:
            # 创建一个新的图形 f 和轴对象 ax
            f, ax = plt.subplots()
            # 绘制 long_df 数据框的 x 和 y 列的核密度估计图，带有 hue 分组和填充状态 fill
            kdeplot(data=long_df, x="x", y="y", hue="c", fill=fill)
            # 遍历轴对象 ax 中的所有图形集合
            for c in ax.collections:
                # 根据 Matplotlib 版本判断集合的类型，并进行断言
                if not _version_predates(mpl, "3.8.0rc1"):
                    assert isinstance(c, mpl.contour.QuadContourSet)
                elif fill or not _version_predates(mpl, "3.5.0b0"):
                    assert isinstance(c, mpl.collections.PathCollection)
                else:
                    assert isinstance(c, mpl.collections.LineCollection)

    # 测试共同标准化参数 common_norm 的核密度估计
    def test_common_norm(self, rng):

        # 创建一个重复值数组 hue，长度为 160
        hue = np.repeat(["a", "a", "a", "b"], 40)
        # 从多变量正态分布中生成 x 和 y 数据，根据 hue 进行偏移调整
        x, y = rng.multivariate_normal([0, 0], [(.2, .5), (.5, 2)], len(hue)).T
        x[hue == "a"] -= 2
        x[hue == "b"] += 2

        # 创建一个包含两个子图的新图形 f 和轴对象 ax1, ax2
        f, (ax1, ax2) = plt.subplots(ncols=2)
        # 在 ax1 上绘制 x 和 y 的核密度估计图，带有 hue 和共同标准化参数 common_norm
        kdeplot(x=x, y=y, hue=hue, common_norm=True, ax=ax1)
        # 在 ax2 上绘制 x 和 y 的核密度估计图，带有 hue 和非共同标准化参数 common_norm
        kdeplot(x=x, y=y, hue=hue, common_norm=False, ax=ax2)

        # 计算 ax1 和 ax2 中每个集合的轮廓坐标数量，并断言非共同标准化参数下的数量大于共同标准化参数下的数量
        n_seg_1 = sum(len(get_contour_coords(c, True)) for c in ax1.collections)
        n_seg_2 = sum(len(get_contour_coords(c, True)) for c in ax2.collections)
        assert n_seg_2 > n_seg_1
    # 测试在对数尺度下绘制核密度估计的效果
    def test_log_scale(self, rng):
        # 生成服从对数正态分布的随机数据
        x = rng.lognormal(0, 1, 100)
        # 生成在 [0, 1] 区间均匀分布的随机数据
        y = rng.uniform(0, 1, 100)

        # 设置对数尺度的等高线级别
        levels = .2, .5, 1

        # 创建包含单个子图的图像对象和轴对象
        f, ax = plt.subplots()
        # 绘制核密度估计的二维 KDE 图，使用对数尺度
        kdeplot(x=x, y=y, log_scale=True, levels=levels, ax=ax)
        # 断言 x 轴使用对数尺度
        assert ax.get_xscale() == "log"
        # 断言 y 轴使用对数尺度
        assert ax.get_yscale() == "log"

        # 创建包含两个子图的图像对象
        f, (ax1, ax2) = plt.subplots(ncols=2)
        # 绘制核密度估计的二维 KDE 图，设置 x 轴为对数尺度，y 轴为线性尺度
        kdeplot(x=x, y=y, log_scale=(10, False), levels=levels, ax=ax1)
        # 断言 ax1 的 x 轴使用对数尺度
        assert ax1.get_xscale() == "log"
        # 断言 ax1 的 y 轴使用线性尺度
        assert ax1.get_yscale() == "linear"

        # 创建一个分布绘图器对象
        p = _DistributionPlotter()
        # 创建 KDE 对象
        kde = KDE()
        # 计算以 10 为底的 x 的对数和 y 的 KDE 密度
        density, (xx, yy) = kde(np.log10(x), y)
        # 将分位数转换为对应的等高线级别
        levels = p._quantile_to_level(density, levels)
        # 在 ax2 中绘制等高线图
        ax2.contour(10 ** xx, yy, density, levels=levels)

        # 比较两个子图的等高线坐标数目是否相等
        for c1, c2 in zip(ax1.collections, ax2.collections):
            assert len(get_contour_coords(c1)) == len(get_contour_coords(c2))
            # 比较每对等高线的坐标数组是否相等
            for arr1, arr2 in zip(get_contour_coords(c1), get_contour_coords(c2)):
                assert_array_equal(arr1, arr2)

    # 测试核密度估计带宽调整的效果
    def test_bandwidth(self, rng):
        n = 100
        # 生成多元正态分布的随机数据
        x, y = rng.multivariate_normal([0, 0], [(.2, .5), (.5, 2)], n).T

        # 创建包含两个子图的图像对象
        f, (ax1, ax2) = plt.subplots(ncols=2)

        # 在 ax1 中绘制未调整带宽的核密度估计图
        kdeplot(x=x, y=y, ax=ax1)
        # 在 ax2 中绘制带有调整带宽的核密度估计图
        kdeplot(x=x, y=y, bw_adjust=2, ax=ax2)

        # 比较两个子图的等高线坐标数组
        for c1, c2 in zip(ax1.collections, ax2.collections):
            seg1, seg2 = get_contour_coords(c1), get_contour_coords(c2)
            if seg1 + seg2:
                # 比较两个子图的等高线在 x 轴方向上的最大值绝对值
                x1 = seg1[0][:, 0]
                x2 = seg2[0][:, 0]
                assert np.abs(x2).max() > np.abs(x1).max()

    # 测试核密度估计中加权数据的效果
    def test_weights(self, rng):
        n = 100
        # 生成多元正态分布的随机数据
        x, y = rng.multivariate_normal([1, 3], [(.2, .5), (.5, 2)], n).T
        # 生成用于色调分组的数组
        hue = np.repeat([0, 1], n // 2)
        # 生成权重数组
        weights = rng.uniform(0, 1, n)

        # 创建包含两个子图的图像对象
        f, (ax1, ax2) = plt.subplots(ncols=2)
        # 在 ax1 中绘制基于色调分组的核密度估计图
        kdeplot(x=x, y=y, hue=hue, ax=ax1)
        # 在 ax2 中绘制基于色调分组和权重的核密度估计图
        kdeplot(x=x, y=y, hue=hue, weights=weights, ax=ax2)

        # 比较两个子图的等高线坐标数组
        for c1, c2 in zip(ax1.collections, ax2.collections):
            if get_contour_coords(c1) and get_contour_coords(c2):
                # 连接两个子图的等高线坐标数组
                seg1 = np.concatenate(get_contour_coords(c1), axis=0)
                seg2 = np.concatenate(get_contour_coords(c2), axis=0)
                # 断言两个子图的等高线坐标数组不相等
                assert not np.array_equal(seg1, seg2)

    # 测试核密度估计在存在色调参数时是否忽略了 cmap 参数
    def test_hue_ignores_cmap(self, long_df):
        # 断言在使用色调参数时发出 UserWarning，指定的 cmap 参数被忽略
        with pytest.warns(UserWarning, match="cmap parameter ignored"):
            ax = kdeplot(data=long_df, x="x", y="y", hue="c", cmap="viridis")

        # 断言第一个等高线集合的颜色与 "C0" 相等
        assert_colors_equal(get_contour_color(ax.collections[0]), "C0")

    # 测试核密度估计中等高线线条的颜色设置
    def test_contour_line_colors(self, long_df):
        # 指定等高线线条的颜色
        color = (.2, .9, .8, 1)
        # 绘制使用指定颜色的核密度估计图
        ax = kdeplot(data=long_df, x="x", y="y", color=color)

        # 检查每个等高线集合的颜色是否与指定颜色相等
        for c in ax.collections:
            assert_colors_equal(get_contour_color(c), color)
    def test_contour_line_cmap(self, long_df):
        # 使用"Blues"调色板生成颜色列表
        color_list = color_palette("Blues", 12)
        # 创建基于颜色列表的颜色映射
        cmap = mpl.colors.ListedColormap(color_list)
        # 绘制核密度估计图并设置颜色映射
        ax = kdeplot(data=long_df, x="x", y="y", cmap=cmap)
        # 遍历图中的所有集合对象
        for c in ax.collections:
            # 获取轮廓线的颜色并检查其是否在颜色列表中
            for color in get_contour_color(c):
                assert to_rgb(color) in color_list

    def test_contour_fill_colors(self, long_df):
        # 设置填充颜色的数量
        n = 6
        # 设置填充颜色的 RGBA 值
        color = (.2, .9, .8, 1)
        # 绘制核密度估计图并设置填充为真和指定的颜色及级别
        ax = kdeplot(
            data=long_df, x="x", y="y", fill=True, color=color, levels=n,
        )
        # 创建颜色映射并获取其颜色数组
        cmap = light_palette(color, reverse=True, as_cmap=True)
        lut = cmap(np.linspace(0, 1, 256))
        # 遍历图中的所有集合对象
        for c in ax.collections:
            # 获取填充颜色并检查其是否在颜色映射中
            for color in c.get_facecolor():
                assert color in lut

    def test_colorbar(self, long_df):
        # 绘制核密度估计图并设置显示颜色条
        ax = kdeplot(data=long_df, x="x", y="y", fill=True, cbar=True)
        # 断言图中的轴数量为 2
        assert len(ax.figure.axes) == 2

    def test_levels_and_thresh(self, long_df):
        # 创建包含两个子图的图形对象
        f, (ax1, ax2) = plt.subplots(ncols=2)

        # 设置级别和阈值
        n = 8
        thresh = .1
        # 设置绘图参数
        plot_kws = dict(data=long_df, x="x", y="y")
        # 绘制两个核密度估计图，分别设置级别和阈值
        kdeplot(**plot_kws, levels=n, thresh=thresh, ax=ax1)
        kdeplot(**plot_kws, levels=np.linspace(thresh, 1, n), ax=ax2)

        # 遍历两个子图的集合对象
        for c1, c2 in zip(ax1.collections, ax2.collections):
            # 断言两个轮廓线集合对象的坐标数相等
            assert len(get_contour_coords(c1)) == len(get_contour_coords(c2))
            # 检查每对轮廓线的坐标数组是否相等
            for arr1, arr2 in zip(get_contour_coords(c1), get_contour_coords(c2)):
                assert_array_equal(arr1, arr2)

        # 使用 pytest 检查是否引发 ValueError 异常
        with pytest.raises(ValueError):
            kdeplot(**plot_kws, levels=[0, 1, 2])

        # 清空两个子图的内容
        ax1.clear()
        ax2.clear()

        # 绘制两个核密度估计图，分别设置级别和阈值为 None 和 0
        kdeplot(**plot_kws, levels=n, thresh=None, ax=ax1)
        kdeplot(**plot_kws, levels=n, thresh=0, ax=ax2)

        # 遍历两个子图的集合对象
        for c1, c2 in zip(ax1.collections, ax2.collections):
            # 断言两个轮廓线集合对象的坐标数相等
            assert len(get_contour_coords(c1)) == len(get_contour_coords(c2))
            # 检查每对轮廓线的坐标数组是否相等
            for arr1, arr2 in zip(get_contour_coords(c1), get_contour_coords(c2)):
                assert_array_equal(arr1, arr2)

        # 遍历两个子图的集合对象
        for c1, c2 in zip(ax1.collections, ax2.collections):
            # 检查两个集合对象的填充颜色数组是否相等
            assert_array_equal(c1.get_facecolors(), c2.get_facecolors())

    def test_quantile_to_level(self, rng):
        # 生成均匀分布的随机数
        x = rng.uniform(0, 1, 100000)
        # 设置分位数
        isoprop = np.linspace(.1, 1, 6)

        # 使用分位数转换函数计算级别
        levels = _DistributionPlotter()._quantile_to_level(x, isoprop)
        # 遍历计算出的级别和分位数
        for h, p in zip(levels, isoprop):
            # 断言计算出的分位数近似等于给定的分位数
            assert (x[x <= h].sum() / x.sum()) == pytest.approx(p, abs=1e-4)

    def test_input_checking(self, long_df):
        # 使用 pytest 检查是否引发 TypeError 异常并包含指定的错误信息
        with pytest.raises(TypeError, match="The x variable is categorical,"):
            kdeplot(data=long_df, x="a", y="y")
class TestHistPlotUnivariate(SharedAxesLevelTests):
    # 继承自 SharedAxesLevelTests 的单变量直方图绘制测试类

    func = staticmethod(histplot)
    # 设置 func 属性为 histplot 的静态方法

    def get_last_color(self, ax, element="bars", fill=True):
        # 定义获取最后一个图形元素颜色的方法，参数包括轴对象 ax，元素类型 element，默认为 bars，是否填充 fill，默认为 True

        if element == "bars":
            if fill:
                return ax.patches[-1].get_facecolor()
            else:
                return ax.patches[-1].get_edgecolor()
        else:
            if fill:
                artist = ax.collections[-1]
                facecolor = artist.get_facecolor()
                edgecolor = artist.get_edgecolor()
                assert_colors_equal(facecolor, edgecolor, check_alpha=False)
                return facecolor
            else:
                return ax.lines[-1].get_color()

    @pytest.mark.parametrize(
        "element,fill",
        itertools.product(["bars", "step", "poly"], [True, False]),
    )
    # 使用 pytest 的参数化装饰器，参数为 element 和 fill，分别为 "bars", "step", "poly" 和 True, False 的组合
    def test_color(self, long_df, element, fill):
        # 颜色测试方法，测试不同情况下的颜色

        super().test_color(long_df, element=element, fill=fill)
        # 调用父类的 test_color 方法进行测试

    @pytest.mark.parametrize(
        "variable", ["x", "y"],
    )
    # 使用 pytest 的参数化装饰器，参数为 variable，取值为 "x" 和 "y"
    def test_long_vectors(self, long_df, variable):
        # 测试长向量的方法，参数包括长数据框 long_df 和变量名称 variable

        vector = long_df[variable]
        vectors = [
            variable, vector, vector.to_numpy(), vector.to_list(),
        ]
        # 创建变量 vector 和 vectors 列表，分别存储变量和不同表示形式的向量

        f, axs = plt.subplots(3)
        # 创建包含 3 个子图的图形对象

        for vector, ax in zip(vectors, axs):
            histplot(data=long_df, ax=ax, **{variable: vector})
            # 对每个子图绘制长数据框的直方图，根据 variable 指定的变量

        bars = [ax.patches for ax in axs]
        # 获取每个子图的条形图列表

        for a_bars, b_bars in itertools.product(bars, bars):
            for a, b in zip(a_bars, b_bars):
                assert_array_equal(a.get_height(), b.get_height())
                assert_array_equal(a.get_xy(), b.get_xy())
                # 断言每对条形图的高度和位置一致

    def test_wide_vs_long_data(self, wide_df):
        # 测试宽数据框与长数据框的方法，参数为宽数据框 wide_df

        f, (ax1, ax2) = plt.subplots(2)
        # 创建包含 2 个子图的图形对象

        histplot(data=wide_df, ax=ax1, common_bins=False)
        # 绘制宽数据框的直方图到第一个子图，禁用公共 bins

        for col in wide_df.columns[::-1]:
            histplot(data=wide_df, x=col, ax=ax2)
            # 对宽数据框的每一列反向绘制直方图到第二个子图

        for a, b in zip(ax1.patches, ax2.patches):
            assert a.get_height() == b.get_height()
            assert a.get_xy() == b.get_xy()
            # 断言每对条形图的高度和位置一致

    def test_flat_vector(self, long_df):
        # 测试扁平向量的方法，参数为长数据框 long_df

        f, (ax1, ax2) = plt.subplots(2)
        # 创建包含 2 个子图的图形对象

        histplot(data=long_df["x"], ax=ax1)
        # 绘制长数据框中变量 "x" 的直方图到第一个子图

        histplot(data=long_df, x="x", ax=ax2)
        # 绘制长数据框的直方图到第二个子图，指定 x 轴变量为 "x"

        for a, b in zip(ax1.patches, ax2.patches):
            assert a.get_height() == b.get_height()
            assert a.get_xy() == b.get_xy()
            # 断言每对条形图的高度和位置一致

    def test_empty_data(self):
        # 测试空数据的方法

        ax = histplot(x=[])
        # 绘制空数据的直方图，返回轴对象 ax

        assert not ax.patches
        # 断言轴对象 ax 中没有条形图

    def test_variable_assignment(self, long_df):
        # 测试变量分配的方法，参数为长数据框 long_df

        f, (ax1, ax2) = plt.subplots(2)
        # 创建包含 2 个子图的图形对象

        histplot(data=long_df, x="x", ax=ax1)
        # 绘制长数据框的直方图到第一个子图，指定 x 轴变量为 "x"

        histplot(data=long_df, y="x", ax=ax2)
        # 绘制长数据框的直方图到第二个子图，指定 y 轴变量为 "x"

        for a, b in zip(ax1.patches, ax2.patches):
            assert a.get_height() == b.get_width()
            # 断言每对条形图的高度和宽度一致

    @pytest.mark.parametrize("element", ["bars", "step", "poly"])
    @pytest.mark.parametrize("multiple", ["layer", "dodge", "stack", "fill"])
    # 使用 pytest 的参数化装饰器，分别为 element 和 multiple，取值为 ["bars", "step", "poly"] 和 ["layer", "dodge", "stack", "fill"]
    # 测试带有色调的填充颜色的直方图绘制
    def test_hue_fill_colors(self, long_df, multiple, element):

        # 使用 histplot 函数绘制直方图，设置 x 轴为 "x" 列，色调为 "a" 列
        ax = histplot(
            data=long_df, x="x", hue="a",
            multiple=multiple, bins=1,
            fill=True, element=element, legend=False,
        )

        # 获取当前颜色板
        palette = color_palette()

        # 根据 multiple 参数和 element 参数设置透明度 a
        if multiple == "layer":
            if element == "bars":
                a = .5
            else:
                a = .25
        else:
            a = .75

        # 遍历 ax.patches 列表，比较每个条形图的颜色与预期颜色是否相等
        for bar, color in zip(ax.patches[::-1], palette):
            assert_colors_equal(bar.get_facecolor(), to_rgba(color, a))

        # 遍历 ax.collections 列表，比较每个多边形的颜色与预期颜色是否相等
        for poly, color in zip(ax.collections[::-1], palette):
            assert_colors_equal(poly.get_facecolor(), to_rgba(color, a))

    # 测试带有色调的堆叠直方图绘制
    def test_hue_stack(self, long_df):

        # 创建包含两个子图的图形对象 f，返回两个 Axes 对象 ax1 和 ax2
        f, (ax1, ax2) = plt.subplots(2)

        # 设置分组数为 n
        n = 10

        # 构建参数字典 kws
        kws = dict(data=long_df, x="x", hue="a", bins=n, element="bars")

        # 在 ax1 上绘制 layer 类型的直方图
        histplot(**kws, multiple="layer", ax=ax1)
        # 在 ax2 上绘制 stack 类型的直方图
        histplot(**kws, multiple="stack", ax=ax2)

        # 将 ax1 中的条形图高度重新排列为 n 行
        layer_heights = np.reshape([b.get_height() for b in ax1.patches], (-1, n))
        # 将 ax2 中的条形图高度重新排列为 n 行
        stack_heights = np.reshape([b.get_height() for b in ax2.patches], (-1, n))
        # 检查堆叠高度是否与层叠高度相等
        assert_array_equal(layer_heights, stack_heights)

        # 将 ax2 中的条形图的坐标重新排列为 n 行
        stack_xys = np.reshape([b.get_xy() for b in ax2.patches], (-1, n, 2))
        # 检查堆叠的 x、y 坐标是否与堆叠高度的累积和相等
        assert_array_equal(
            stack_xys[..., 1] + stack_heights,
            stack_heights.cumsum(axis=0),
        )

    # 测试带有色调的填充直方图绘制
    def test_hue_fill(self, long_df):

        # 创建包含两个子图的图形对象 f，返回两个 Axes 对象 ax1 和 ax2
        f, (ax1, ax2) = plt.subplots(2)

        # 设置分组数为 n
        n = 10

        # 构建参数字典 kws
        kws = dict(data=long_df, x="x", hue="a", bins=n, element="bars")

        # 在 ax1 上绘制 layer 类型的直方图
        histplot(**kws, multiple="layer", ax=ax1)
        # 在 ax2 上绘制 fill 类型的直方图
        histplot(**kws, multiple="fill", ax=ax2)

        # 将 ax1 中的条形图高度重新排列为 n 行
        layer_heights = np.reshape([b.get_height() for b in ax1.patches], (-1, n))
        # 将 ax2 中的条形图高度重新排列为 n 行
        stack_heights = np.reshape([b.get_height() for b in ax2.patches], (-1, n))
        # 检查填充高度是否与层叠高度的归一化值相等
        assert_array_almost_equal(
            layer_heights / layer_heights.sum(axis=0), stack_heights
        )

        # 将 ax2 中的条形图的坐标重新排列为 n 行
        stack_xys = np.reshape([b.get_xy() for b in ax2.patches], (-1, n, 2))
        # 检查填充的 x、y 坐标是否与填充高度的累积和的归一化值相等
        assert_array_almost_equal(
            (stack_xys[..., 1] + stack_heights) / stack_heights.sum(axis=0),
            stack_heights.cumsum(axis=0),
        )

    # 测试带有色调的分组直方图绘制
    def test_hue_dodge(self, long_df):

        # 创建包含两个子图的图形对象 f，返回两个 Axes 对象 ax1 和 ax2
        f, (ax1, ax2) = plt.subplots(2)

        # 设置条宽为 bw
        bw = 2

        # 构建参数字典 kws
        kws = dict(data=long_df, x="x", hue="c", binwidth=bw, element="bars")

        # 在 ax1 上绘制 layer 类型的直方图
        histplot(**kws, multiple="layer", ax=ax1)
        # 在 ax2 上绘制 dodge 类型的直方图
        histplot(**kws, multiple="dodge", ax=ax2)

        # 获取 ax1 中的条形图高度列表
        layer_heights = [b.get_height() for b in ax1.patches]
        # 获取 ax2 中的条形图高度列表
        dodge_heights = [b.get_height() for b in ax2.patches]
        # 检查层叠高度与分组高度是否相等
        assert_array_equal(layer_heights, dodge_heights)

        # 将 ax1 中的条形图 x 坐标重新排列为两行
        layer_xs = np.reshape([b.get_x() for b in ax1.patches], (2, -1))
        # 将 ax2 中的条形图 x 坐标重新排列为两行
        dodge_xs = np.reshape([b.get_x() for b in ax2.patches], (2, -1))
        # 检查分组的 x 坐标是否与层叠的 x 坐标相等
        assert_array_almost_equal(layer_xs[1], dodge_xs[1])
        assert_array_almost_equal(layer_xs[0], dodge_xs[0] - bw / 2)
    def test_hue_as_numpy_dodged(self, long_df):
        # 根据 GitHub 问题链接，用 Seaborn 绘制长格式数据的直方图，设置 dodge 方式绘制
        ax = histplot(
            long_df,
            x="y", hue=long_df["a"].to_numpy(),
            multiple="dodge", bins=1,
        )
        # 断言：检查颜色变量顺序是否反转
        assert ax.patches[1].get_x() < ax.patches[0].get_x()

    def test_multiple_input_check(self, flat_series):
        # 使用 pytest 检查传入参数错误时是否引发 ValueError 异常
        with pytest.raises(ValueError, match="`multiple` must be"):
            histplot(flat_series, multiple="invalid")

    def test_element_input_check(self, flat_series):
        # 使用 pytest 检查传入参数错误时是否引发 ValueError 异常
        with pytest.raises(ValueError, match="`element` must be"):
            histplot(flat_series, element="invalid")

    def test_count_stat(self, flat_series):
        # 绘制直方图并设置统计方式为计数统计
        ax = histplot(flat_series, stat="count")
        # 获取每个条形图的高度列表
        bar_heights = [b.get_height() for b in ax.patches]
        # 断言：检查条形图总高度是否等于数据长度
        assert sum(bar_heights) == len(flat_series)

    def test_density_stat(self, flat_series):
        # 绘制直方图并设置统计方式为密度
        ax = histplot(flat_series, stat="density")
        # 获取每个条形图的高度和宽度列表
        bar_heights = [b.get_height() for b in ax.patches]
        bar_widths = [b.get_width() for b in ax.patches]
        # 断言：检查条形图面积乘积是否大致等于1
        assert np.multiply(bar_heights, bar_widths).sum() == pytest.approx(1)

    def test_density_stat_common_norm(self, long_df):
        # 使用 Seaborn 绘制长格式数据的直方图，设置统计方式为密度，使用共同归一化，绘制元素为条形图
        ax = histplot(
            data=long_df, x="x", hue="a",
            stat="density", common_norm=True, element="bars",
        )
        # 获取每个条形图的高度和宽度列表
        bar_heights = [b.get_height() for b in ax.patches]
        bar_widths = [b.get_width() for b in ax.patches]
        # 断言：检查条形图面积乘积是否大致等于1
        assert np.multiply(bar_heights, bar_widths).sum() == pytest.approx(1)

    def test_density_stat_unique_norm(self, long_df):
        # 使用 Seaborn 绘制长格式数据的直方图，设置统计方式为密度，指定分组柱数、不使用共同归一化、绘制元素为条形图
        n = 10
        ax = histplot(
            data=long_df, x="x", hue="a",
            stat="density", bins=n, common_norm=False, element="bars",
        )
        # 将直方图分成两组条形图
        bar_groups = ax.patches[:n], ax.patches[-n:]

        for bars in bar_groups:
            # 获取每组条形图的高度和宽度列表
            bar_heights = [b.get_height() for b in bars]
            bar_widths = [b.get_width() for b in bars]
            # 计算每组条形图的面积乘积
            bar_areas = np.multiply(bar_heights, bar_widths)
            # 断言：检查每组条形图面积乘积是否大致等于1
            assert bar_areas.sum() == pytest.approx(1)

    @pytest.fixture(params=["probability", "proportion"])
    def height_norm_arg(self, request):
        # 返回测试参数：高度归一化方式
        return request.param

    def test_probability_stat(self, flat_series, height_norm_arg):
        # 绘制直方图并设置统计方式为概率或比例
        ax = histplot(flat_series, stat=height_norm_arg)
        # 获取每个条形图的高度列表
        bar_heights = [b.get_height() for b in ax.patches]
        # 断言：检查条形图总高度是否大致等于1
        assert sum(bar_heights) == pytest.approx(1)

    def test_probability_stat_common_norm(self, long_df, height_norm_arg):
        # 使用 Seaborn 绘制长格式数据的直方图，设置统计方式为概率或比例、使用共同归一化、绘制元素为条形图
        ax = histplot(
            data=long_df, x="x", hue="a",
            stat=height_norm_arg, common_norm=True, element="bars",
        )
        # 获取每个条形图的高度列表
        bar_heights = [b.get_height() for b in ax.patches]
        # 断言：检查条形图总高度是否大致等于1
        assert sum(bar_heights) == pytest.approx(1)
    # 测试函数，用于验证带有不同统计特性的直方图绘制
    def test_probability_stat_unique_norm(self, long_df, height_norm_arg):
        # 设定直方图的条数
        n = 10
        # 绘制直方图，并返回 AxesSubplot 对象
        ax = histplot(
            data=long_df, x="x", hue="a",
            stat=height_norm_arg, bins=n, common_norm=False, element="bars",
        )

        # 按照设定的条数划分成两组条形图
        bar_groups = ax.patches[:n], ax.patches[-n:]

        # 遍历每组条形图
        for bars in bar_groups:
            # 获取每个条形图的高度
            bar_heights = [b.get_height() for b in bars]
            # 断言每组条形图的高度总和约等于 1
            assert sum(bar_heights) == pytest.approx(1)

    # 测试函数，用于验证百分比统计特性的直方图绘制
    def test_percent_stat(self, flat_series):
        # 绘制直方图，并返回 AxesSubplot 对象
        ax = histplot(flat_series, stat="percent")
        # 获取每个条形图的高度
        bar_heights = [b.get_height() for b in ax.patches]
        # 断言所有条形图的高度总和约等于 100
        assert sum(bar_heights) == 100

    # 测试函数，用于验证共享 bins 的直方图绘制
    def test_common_bins(self, long_df):
        # 设定直方图的条数
        n = 10
        # 绘制直方图，并返回 AxesSubplot 对象
        ax = histplot(
            long_df, x="x", hue="a", common_bins=True, bins=n, element="bars",
        )

        # 将直方图分为两组条形图
        bar_groups = ax.patches[:n], ax.patches[-n:]
        # 断言两组条形图的位置和尺寸相等
        assert_array_equal(
            [b.get_xy() for b in bar_groups[0]],
            [b.get_xy() for b in bar_groups[1]]
        )

    # 测试函数，用于验证唯一 bins 的直方图绘制
    def test_unique_bins(self, wide_df):
        # 绘制直方图，并返回 AxesSubplot 对象
        ax = histplot(wide_df, common_bins=False, bins=10, element="bars")

        # 按列分割条形图为多组
        bar_groups = np.split(np.array(ax.patches), len(wide_df.columns))

        # 遍历每一组条形图
        for i, col in enumerate(wide_df.columns[::-1]):
            bars = bar_groups[i]
            # 获取每组条形图的起始位置和结束位置
            start = bars[0].get_x()
            stop = bars[-1].get_x() + bars[-1].get_width()
            # 断言每组条形图的起始位置和结束位置近似等于相应列的最小值和最大值
            assert_array_almost_equal(start, wide_df[col].min())
            assert_array_almost_equal(stop, wide_df[col].max())

    # 测试函数，用于验证带有无穷值的直方图绘制
    def test_range_with_inf(self, rng):
        # 生成随机数据
        x = rng.normal(0, 1, 20)
        # 绘制直方图，并返回 AxesSubplot 对象
        ax = histplot([-np.inf, *x])
        # 获取最左边条形图的位置
        leftmost_edge = min(p.get_x() for p in ax.patches)
        # 断言最左边条形图的位置等于 x 的最小值
        assert leftmost_edge == x.min()

    # 测试函数，用于验证带有缺失权重的直方图绘制
    def test_weights_with_missing(self, null_df):
        # 绘制带有权重的直方图，并返回 AxesSubplot 对象
        ax = histplot(null_df, x="x", weights="s", bins=5)

        # 获取每个条形图的高度
        bar_heights = [bar.get_height() for bar in ax.patches]
        # 计算有效权重的总和
        total_weight = null_df[["x", "s"]].dropna()["s"].sum()
        # 断言所有条形图的高度总和约等于有效权重的总和
        assert sum(bar_heights) == pytest.approx(total_weight)

    # 测试函数，用于验证带有权重归一化的直方图绘制
    def test_weight_norm(self, rng):
        # 生成随机数据和权重
        vals = rng.normal(0, 1, 50)
        x = np.concatenate([vals, vals])
        w = np.repeat([1, 2], 50)
        # 绘制带有权重归一化的直方图，并返回 AxesSubplot 对象
        ax = histplot(
            x=x, weights=w, hue=w, common_norm=True, stat="density", bins=5
        )

        # 根据 hue 顺序反向获取条形图的高度
        y1 = [bar.get_height() for bar in ax.patches[:5]]
        y2 = [bar.get_height() for bar in ax.patches[5:]]

        # 断言第一组条形图的高度总和应等于第二组条形图高度总和的两倍
        assert sum(y1) == 2 * sum(y2)

    # 测试函数，用于验证离散数据的直方图绘制
    def test_discrete(self, long_df):
        # 绘制离散数据的直方图，并返回 AxesSubplot 对象
        ax = histplot(long_df, x="s", discrete=True)

        # 获取数据的最小值和最大值
        data_min = long_df["s"].min()
        data_max = long_df["s"].max()
        # 断言条形图的数量等于数据的取值范围
        assert len(ax.patches) == (data_max - data_min + 1)

        # 遍历每个条形图
        for i, bar in enumerate(ax.patches):
            # 断言每个条形图的宽度等于 1
            assert bar.get_width() == 1
            # 断言每个条形图的起始位置等于数据最小值加上索引调整值
            assert bar.get_x() == (data_min + i - .5)
    # 测试离散分类数据的默认情况下直方图的绘制
    def test_discrete_categorical_default(self, long_df):
        # 绘制数据列"a"的直方图
        ax = histplot(long_df, x="a")
        # 遍历直方图中的每一个条形块
        for i, bar in enumerate(ax.patches):
            # 断言每个条形块的宽度为1
            assert bar.get_width() == 1

    # 测试分类数据在y轴上反转的情况
    def test_categorical_yaxis_inversion(self, long_df):
        # 绘制数据列"a"的直方图，并将y轴反转
        ax = histplot(long_df, y="a")
        # 获取y轴的最大和最小值
        ymax, ymin = ax.get_ylim()
        # 断言最大值大于最小值
        assert ymax > ymin

    # 测试日期时间数据的比例尺
    def test_datetime_scale(self, long_df):
        # 创建包含两个子图的图形对象f，并获取子图对象ax1和ax2
        f, (ax1, ax2) = plt.subplots(2)
        # 在ax1和ax2上分别绘制数据列"t"的填充和非填充的直方图
        histplot(x=long_df["t"], fill=True, ax=ax1)
        histplot(x=long_df["t"], fill=False, ax=ax2)
        # 断言两个子图的x轴限制相同
        assert ax1.get_xlim() == ax2.get_xlim()

    # 使用参数化测试来测试核密度估计（KDE）
    @pytest.mark.parametrize("stat", ["count", "density", "probability"])
    def test_kde(self, flat_series, stat):
        # 绘制平面系列flat_series的直方图，并包括KDE曲线和特定的统计指标
        ax = histplot(
            flat_series, kde=True, stat=stat, kde_kws={"cut": 10}
        )
        # 获取每个条形块的宽度和高度
        bar_widths = [b.get_width() for b in ax.patches]
        bar_heights = [b.get_height() for b in ax.patches]
        # 计算直方图的总面积
        hist_area = np.multiply(bar_widths, bar_heights).sum()

        # 获取KDE曲线的数据点
        density, = ax.lines
        kde_area = integrate(density.get_ydata(), density.get_xdata())

        # 断言KDE曲线的面积近似等于直方图的面积
        assert kde_area == pytest.approx(hist_area)

    # 使用参数化测试和hue参数来测试带有多层或分组的KDE
    @pytest.mark.parametrize("multiple", ["layer", "dodge"])
    @pytest.mark.parametrize("stat", ["count", "density", "probability"])
    def test_kde_with_hue(self, long_df, stat, multiple):
        # 设置每个组中的条形块数量
        n = 10
        # 绘制长数据框long_df的直方图，根据"x"列和"hue"列进行分组，包括KDE曲线和特定的统计指标
        ax = histplot(
            long_df, x="x", hue="c", multiple=multiple,
            kde=True, stat=stat, element="bars",
            kde_kws={"cut": 10}, bins=n,
        )
        # 将条形块分为两组
        bar_groups = ax.patches[:n], ax.patches[-n:]

        for i, bars in enumerate(bar_groups):
            # 获取每组条形块的宽度和高度
            bar_widths = [b.get_width() for b in bars]
            bar_heights = [b.get_height() for b in bars]
            # 计算直方图的总面积
            hist_area = np.multiply(bar_widths, bar_heights).sum()

            # 获取第i个KDE曲线的数据点
            x, y = ax.lines[i].get_xydata().T
            kde_area = integrate(y, x)

            # 根据multiple参数值断言KDE曲线的面积与直方图的面积之间的关系
            if multiple == "layer":
                assert kde_area == pytest.approx(hist_area)
            elif multiple == "dodge":
                assert kde_area == pytest.approx(hist_area * 2)

    # 测试使用默认cut参数的KDE
    def test_kde_default_cut(self, flat_series):
        # 绘制平面系列flat_series的直方图，并获取KDE曲线的支持范围
        ax = histplot(flat_series, kde=True)
        support = ax.lines[0].get_xdata()
        # 断言KDE曲线的支持范围的最小值和最大值与flat_series的最小值和最大值相等
        assert support.min() == flat_series.min()
        assert support.max() == flat_series.max()

    # 测试带有hue参数的KDE的颜色一致性
    def test_kde_hue(self, long_df):
        # 设置每个组中的条形块数量
        n = 10
        # 绘制长数据框long_df的直方图，根据"x"列和"a"列进行分组，包括KDE曲线
        ax = histplot(data=long_df, x="x", hue="a", kde=True, bins=n)

        # 遍历每个条形块和对应的KDE曲线，断言它们的颜色一致
        for bar, line in zip(ax.patches[::n], ax.lines):
            assert_colors_equal(
                bar.get_facecolor(), line.get_color(), check_alpha=False
            )

    # 测试KDE的y轴表示方式
    def test_kde_yaxis(self, flat_series):
        # 创建一个图形对象f和一个子图对象ax
        f, ax = plt.subplots()
        # 绘制平面系列flat_series的x和y轴KDE曲线
        histplot(x=flat_series, kde=True)
        histplot(y=flat_series, kde=True)

        # 获取第一个和第二个KDE曲线的数据点，分别进行断言
        x, y = ax.lines
        assert_array_equal(x.get_xdata(), y.get_ydata())
        assert_array_equal(x.get_ydata(), y.get_xdata())
    # 测试 KDE 绘图的线宽度设置
    def test_kde_line_kws(self, flat_series):
        # 设置线宽度为 5
        lw = 5
        # 绘制直方图和 KDE 曲线，指定 KDE 曲线的线宽度为 lw
        ax = histplot(flat_series, kde=True, line_kws=dict(lw=lw))
        # 断言第一条线的线宽是否为 lw
        assert ax.lines[0].get_linewidth() == lw

    # 测试当数据为单一值时 KDE 绘图的行为
    def test_kde_singular_data(self):
        # 对于所有的警告，捕获并作为错误处理
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # 绘制 KDE 曲线，数据为长度为 10 的全 1 数组
            ax = histplot(x=np.ones(10), kde=True)
        # 断言不应该有任何线条被绘制
        assert not ax.lines

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # 绘制 KDE 曲线，数据为包含一个值为 5 的列表
            ax = histplot(x=[5], kde=True)
        # 断言不应该有任何线条被绘制
        assert not ax.lines

    # 测试直方图元素类型为默认类型的绘制行为
    def test_element_default(self, long_df):
        # 创建包含两个子图的图像
        f, (ax1, ax2) = plt.subplots(2)
        # 在第一个子图中绘制 long_df 数据的直方图
        histplot(long_df, x="x", ax=ax1)
        # 在第二个子图中绘制 long_df 数据的直方图，元素类型为 bars
        histplot(long_df, x="x", ax=ax2, element="bars")
        # 断言两个子图的直方图柱子数量相等
        assert len(ax1.patches) == len(ax2.patches)

        # 创建包含两个子图的图像
        f, (ax1, ax2) = plt.subplots(2)
        # 在第一个子图中绘制 long_df 数据的带颜色的直方图，颜色由 'a' 列决定
        histplot(long_df, x="x", hue="a", ax=ax1)
        # 在第二个子图中绘制 long_df 数据的带颜色的直方图，元素类型为 bars
        histplot(long_df, x="x", hue="a", ax=ax2, element="bars")
        # 断言两个子图的直方图柱子数量相等
        assert len(ax1.patches) == len(ax2.patches)

    # 测试直方图绘制时柱状图元素设置为不填充的行为
    def test_bars_no_fill(self, flat_series):
        # 设置透明度为 0.5
        alpha = .5
        # 绘制 flat_series 数据的直方图，柱状图元素类型为 bars，不填充，透明度为 alpha
        ax = histplot(flat_series, element="bars", fill=False, alpha=alpha)
        # 遍历每个柱子，断言柱子的面颜色为透明，边框颜色的透明度为 alpha
        for bar in ax.patches:
            assert bar.get_facecolor() == (0, 0, 0, 0)
            assert bar.get_edgecolor()[-1] == alpha

    # 测试直方图绘制时步进图元素填充的行为
    def test_step_fill(self, flat_series):
        # 创建包含两个子图的图像
        f, (ax1, ax2) = plt.subplots(2)

        n = 10
        # 在第一个子图中绘制 flat_series 数据的直方图，步进图元素类型为 bars，填充为 True，分成 n 个 bin
        histplot(flat_series, element="bars", fill=True, bins=n, ax=ax1)
        # 在第二个子图中绘制 flat_series 数据的直方图，步进图元素类型为 step，填充为 True，分成 n 个 bin
        histplot(flat_series, element="step", fill=True, bins=n, ax=ax2)

        # 获取第一个子图中每个柱子的高度、宽度和边缘位置
        bar_heights = [b.get_height() for b in ax1.patches]
        bar_widths = [b.get_width() for b in ax1.patches]
        bar_edges = [b.get_x() for b in ax1.patches]

        # 获取第二个子图中填充的对象
        fill = ax2.collections[0]
        # 获取填充对象的路径，反转并提取路径的 x、y 坐标
        x, y = fill.get_paths()[0].vertices[::-1].T

        # 断言数组的相等性，比较第一个子图柱子的边缘位置和第二个子图填充路径的 x、y 坐标
        assert_array_equal(x[1:2 * n:2], bar_edges)
        assert_array_equal(y[1:2 * n:2], bar_heights)

        # 断言最后一个 x、y 坐标分别与最后一个柱子的边缘位置和高度相等
        assert x[n * 2] == bar_edges[-1] + bar_widths[-1]
        assert y[n * 2] == bar_heights[-1]

    # 测试直方图绘制时多边形图元素填充的行为
    def test_poly_fill(self, flat_series):
        # 创建包含两个子图的图像
        f, (ax1, ax2) = plt.subplots(2)

        n = 10
        # 在第一个子图中绘制 flat_series 数据的直方图，多边形图元素类型为 bars，填充为 True，分成 n 个 bin
        histplot(flat_series, element="bars", fill=True, bins=n, ax=ax1)
        # 在第二个子图中绘制 flat_series 数据的直方图，多边形图元素类型为 poly，填充为 True，分成 n 个 bin
        histplot(flat_series, element="poly", fill=True, bins=n, ax=ax2)

        # 获取第一个子图中每个柱子的高度、宽度和边缘位置
        bar_heights = np.array([b.get_height() for b in ax1.patches])
        bar_widths = np.array([b.get_width() for b in ax1.patches])
        bar_edges = np.array([b.get_x() for b in ax1.patches])

        # 获取第二个子图中填充的对象
        fill = ax2.collections[0]
        # 获取填充对象的路径，反转并提取路径的 x、y 坐标
        x, y = fill.get_paths()[0].vertices[::-1].T

        # 断言数组的相等性，比较第一个子图柱子的边缘位置和第二个子图填充路径的 x、y 坐标
        assert_array_equal(x[1:n + 1], bar_edges + bar_widths / 2)
        assert_array_equal(y[1:n + 1], bar_heights)
    # 定义一个测试函数，测试在不填充的情况下绘制多条子图的直方图
    def test_poly_no_fill(self, flat_series):

        # 创建一个包含两个子图的图形对象
        f, (ax1, ax2) = plt.subplots(2)

        # 在第一个子图上绘制不填充的直方图
        n = 10
        histplot(flat_series, element="bars", fill=False, bins=n, ax=ax1)
        
        # 在第二个子图上绘制不填充的多边形直方图
        histplot(flat_series, element="poly", fill=False, bins=n, ax=ax2)

        # 获取第一个子图中每个条形的高度、宽度和边缘位置
        bar_heights = np.array([b.get_height() for b in ax1.patches])
        bar_widths = np.array([b.get_width() for b in ax1.patches])
        bar_edges = np.array([b.get_x() for b in ax1.patches])

        # 获取第二个子图中的线条数据
        x, y = ax2.lines[0].get_xydata().T

        # 断言第二个子图中的多边形顶点横坐标与第一个子图中条形的边缘位置一致
        assert_array_equal(x, bar_edges + bar_widths / 2)
        # 断言第二个子图中的多边形顶点纵坐标与第一个子图中条形的高度一致
        assert_array_equal(y, bar_heights)

```   
    # 定义一个测试函数，测试在不填充的情况下绘制多条子图的步进直方图
    def test_step_no_fill(self, flat_series):

        # 创建一个包含两个子图的图形对象
        f, (ax1, ax2) = plt.subplots(2)

        # 在第一个子图上绘制不填充的直方图
        histplot(flat_series, element="bars", fill=False, ax=ax1)
        
        # 在第二个子图上绘制不填充的步进直方图
        histplot(flat_series, element="step", fill=False, ax=ax2)

        # 获取第一个子图中每个条形的高度、宽度和边缘位置
        bar_heights = [b.get_height() for b in ax1.patches]
        bar_widths = [b.get_width() for b in ax1.patches]
        bar_edges = [b.get_x() for b in ax1.patches]

        # 获取第二个子图中的线条数据
        x, y = ax2.lines[0].get_xydata().T

        # 断言第二个子图中的步进线条横坐标与第一个子图中条形的边缘位置一致
        assert_array_equal(x[:-1], bar_edges)
        # 断言第二个子图中的步进线条纵坐标与第一个子图中条形的高度一致
        assert_array_equal(y[:-1], bar_heights)
        # 断言最后一个点的横坐标等于最后一个条形的边缘位置加上宽度
        assert x[-1] == bar_edges[-1] + bar_widths[-1]
        # 断言最后一个点的纵坐标等于倒数第二个点的纵坐标，即步进直方图的最后一个点垂直于前一个点
        assert y[-1] == y[-2]

```   
    # 定义一个测试函数，测试在填充的情况下绘制步进直方图，并比较 x 和 y 轴的反向顶点数组是否相等
    def test_step_fill_xy(self, flat_series):

        # 创建一个图形对象
        f, ax = plt.subplots()

        # 绘制填充的步进直方图，x 轴和 y 轴分别用 flat_series 数据
        histplot(x=flat_series, element="step", fill=True)
        histplot(y=flat_series, element="step", fill=True)

        # 获取 x 轴和 y 轴的顶点数组
        xverts = ax.collections[0].get_paths()[0].vertices
        yverts = ax.collections[1].get_paths()[0].vertices

        # 断言 x 轴的顶点数组等于 y 轴的顶点数组的反向顶点数组
        assert_array_equal(xverts, yverts[:, ::-1])

```   
    # 定义一个测试函数，测试在不填充的情况下绘制步进直方图，并比较 x 和 y 轴数据的顶点数组是否相等
    def test_step_no_fill_xy(self, flat_series):

        # 创建一个图形对象
        f, ax = plt.subplots()

        # 绘制不填充的步进直方图，x 轴和 y 轴分别用 flat_series 数据
        histplot(x=flat_series, element="step", fill=False)
        histplot(y=flat_series, element="step", fill=False)

        # 获取 x 轴和 y 轴的线条对象
        xline, yline = ax.lines

        # 断言 x 轴线条的横坐标数据等于 y 轴线条的纵坐标数据
        assert_array_equal(xline.get_xdata(), yline.get_ydata())
        # 断言 x 轴线条的纵坐标数据等于 y 轴线条的横坐标数据
        assert_array_equal(xline.get_ydata(), yline.get_xdata())

```   
    # 定义一个测试函数，测试加权直方图的绘制
    def test_weighted_histogram(self):

        # 绘制一个 x 数据为 [0, 1, 2]，权重为 [1, 2, 3] 的离散加权直方图
        ax = histplot(x=[0, 1, 2], weights=[1, 2, 3], discrete=True)

        # 获取直方图每个条形的高度
        bar_heights = [b.get_height() for b in ax.patches]
        
        # 断言直方图的条形高度与权重数据相等
        assert bar_heights == [1, 2, 3]

```   
    # 定义一个测试函数，测试在自动确定 bin 的情况下使用权重的直方图绘制
    def test_weights_with_auto_bins(self, long_df):

        # 测试发出 UserWarning 时绘制长数据框的直方图，并获取图形对象
        with pytest.warns(UserWarning):
            ax = histplot(long_df, x="x", weights="f")
        
        # 断言直方图的条形数量为 10
        assert len(ax.patches) == 10

```   
    # 定义一个测试函数，测试在指定缩小比例时绘制直方图的效果
    def test_shrink(self, long_df):

        # 创建一个包含两个子图的图形对象
        f, (ax1, ax2) = plt.subplots(2)

        # 设定条形宽度和缩小比例
        bw = 2
        shrink = .4

        # 在第一个子图上绘制长数据框的直方图，指定条形宽度
        histplot(long_df, x="x", binwidth=bw, ax=ax1)
        
        # 在第二个子图上绘制长数据框的直方图，指定条形宽度和缩小比例
        histplot(long_df, x="x", binwidth=bw, shrink=shrink, ax=ax2)

        # 遍历并比较两个子图中的每对条形
        for p1, p2 in zip(ax1.patches, ax2.patches):

            # 获取每对条形的宽度
            w1, w2 = p1.get_width(), p2.get_width()
            
            # 断言第二个子图中的条形宽度等于缩小比例乘以第一个子图中的条形宽度
            assert w2 == pytest.approx(shrink * w1)

            # 获取每对条形的横坐标
            x1, x2 = p1.get_x(), p2.get_x()
    def test_log_scale_explicit(self, rng):
        # 生成服从对数正态分布的随机数序列
        x = rng.lognormal(0, 2, 1000)
        # 绘制直方图，设置对数坐标轴，指定数据范围和条形宽度
        ax = histplot(x, log_scale=True, binrange=(-3, 3), binwidth=1)

        # 获取每个条形的宽度
        bar_widths = [b.get_width() for b in ax.patches]
        # 计算每相邻两个条形之间的比例
        steps = np.divide(bar_widths[1:], bar_widths[:-1])
        # 断言比例接近10
        assert np.allclose(steps, 10)

    def test_log_scale_implicit(self, rng):
        # 生成服从对数正态分布的随机数序列
        x = rng.lognormal(0, 2, 1000)

        # 创建一个包含两个子图的图像对象
        f, ax = plt.subplots()
        # 设置 x 轴为对数坐标轴
        ax.set_xscale("log")
        # 绘制直方图，指定数据范围和条形宽度，并将其添加到第一个子图中
        histplot(x, binrange=(-3, 3), binwidth=1, ax=ax)

        # 获取每个条形的宽度
        bar_widths = [b.get_width() for b in ax.patches]
        # 计算每相邻两个条形之间的比例
        steps = np.divide(bar_widths[1:], bar_widths[:-1])
        # 断言比例接近10
        assert np.allclose(steps, 10)

    def test_log_scale_dodge(self, rng):
        # 生成服从对数正态分布的随机数序列
        x = rng.lognormal(0, 2, 100)
        # 创建一个分组条形图，设置对数坐标轴
        hue = np.repeat(["a", "b"], 50)
        ax = histplot(x=x, hue=hue, bins=5, log_scale=True, multiple="dodge")
        # 获取每个条形的左侧和右侧 x 坐标的对数值
        x_min = np.log([b.get_x() for b in ax.patches])
        x_max = np.log([b.get_x() + b.get_width() for b in ax.patches])
        # 断言所有条形的对数宽度相等
        assert np.unique(np.round(x_max - x_min, 10)).size == 1

    def test_log_scale_kde(self, rng):
        # 生成服从对数正态分布的随机数序列
        x = rng.lognormal(0, 1, 1000)
        # 绘制直方图，并添加核密度估计，设置对数坐标轴和分箱数
        ax = histplot(x=x, log_scale=True, kde=True, bins=20)
        # 获取最高条形的高度
        bar_height = max(p.get_height() for p in ax.patches)
        # 获取核密度曲线的最高点高度
        kde_height = max(ax.lines[0].get_ydata())
        # 断言最高条形的高度与核密度曲线的最高点高度接近
        assert bar_height == pytest.approx(kde_height, rel=.1)

    @pytest.mark.parametrize(
        "fill", [True, False],
    )
    def test_auto_linewidth(self, flat_series, fill):
        # 定义一个函数，用于获取第一个条形的线宽
        get_lw = lambda ax: ax.patches[0].get_linewidth()  # noqa: E731

        kws = dict(element="bars", fill=fill)

        # 创建两个子图的图像对象
        f, (ax1, ax2) = plt.subplots(2)
        # 绘制直方图到第一个子图，指定分箱数和其他绘图参数
        histplot(flat_series, **kws, bins=10, ax=ax1)
        # 绘制直方图到第二个子图，指定分箱数和其他绘图参数
        histplot(flat_series, **kws, bins=100, ax=ax2)
        # 断言第一个子图的条形线宽大于第二个子图的条形线宽
        assert get_lw(ax1) > get_lw(ax2)

        # 创建两个指定尺寸的子图的图像对象
        f, ax1 = plt.subplots(figsize=(10, 5))
        f, ax2 = plt.subplots(figsize=(2, 5))
        # 绘制直方图到第一个子图，指定分箱数和其他绘图参数
        histplot(flat_series, **kws, bins=30, ax=ax1)
        # 绘制直方图到第二个子图，指定分箱数和其他绘图参数
        histplot(flat_series, **kws, bins=30, ax=ax2)
        # 断言第一个子图的条形线宽大于第二个子图的条形线宽
        assert get_lw(ax1) > get_lw(ax2)

        # 创建两个相同尺寸的子图的图像对象
        f, ax1 = plt.subplots(figsize=(4, 5))
        f, ax2 = plt.subplots(figsize=(4, 5))
        # 绘制直方图到第一个子图，指定分箱数和其他绘图参数
        histplot(flat_series, **kws, bins=30, ax=ax1)
        # 绘制直方图到第二个子图，指定分箱数和其他绘图参数，并设置对数坐标轴
        histplot(10 ** flat_series, **kws, bins=30, log_scale=True, ax=ax2)
        # 断言第一个子图和第二个子图的条形线宽近似相等
        assert get_lw(ax1) == pytest.approx(get_lw(ax2))

        # 创建两个相同尺寸的子图的图像对象
        f, ax1 = plt.subplots(figsize=(4, 5))
        f, ax2 = plt.subplots(figsize=(4, 5))
        # 绘制直方图到第一个子图，指定分箱数和其他绘图参数，并设置为离散型
        histplot(y=[0, 1, 1], **kws, discrete=True, ax=ax1)
        # 绘制直方图到第二个子图，指定分箱数和其他绘图参数
        histplot(y=["a", "b", "b"], **kws, ax=ax2)
        # 断言第一个子图和第二个子图的条形线宽近似相等
        assert get_lw(ax1) == pytest.approx(get_lw(ax2))

    def test_bar_kwargs(self, flat_series):
        # 定义线宽和边框颜色
        lw = 2
        ec = (1, .2, .9, .5)
        # 绘制直方图，设置条形宽度、边框颜色和线宽，并返回轴对象
        ax = histplot(flat_series, binwidth=1, ec=ec, lw=lw)
        # 断言所有条形的边框颜色相等
        for bar in ax.patches:
            assert_colors_equal(bar.get_edgecolor(), ec)
            # 断言所有条形的线宽相等
            assert bar.get_linewidth() == lw
    # 测试函数：验证在填充曲线元素时的关键字参数设置
    def test_step_fill_kwargs(self, flat_series):
        # 设置线宽度为2
        lw = 2
        # 设置边缘颜色为(1, .2, .9, .5)
        ec = (1, .2, .9, .5)
        # 绘制直方图，使用步进元素，设置边缘颜色和线宽度
        ax = histplot(flat_series, element="step", ec=ec, lw=lw)
        # 获取绘制的多边形对象
        poly = ax.collections[0]
        # 断言多边形边缘颜色与预期相同
        assert_colors_equal(poly.get_edgecolor(), ec)
        # 断言多边形线宽度与预期相同
        assert poly.get_linewidth() == lw
    
    # 测试函数：验证在直线步进元素时的关键字参数设置
    def test_step_line_kwargs(self, flat_series):
        # 设置线宽度为2
        lw = 2
        # 设置线条风格为虚线
        ls = "--"
        # 绘制直方图，使用步进元素且不填充，设置线宽度和线条风格
        ax = histplot(flat_series, element="step", fill=False, lw=lw, ls=ls)
        # 获取绘制的线条对象
        line = ax.lines[0]
        # 断言线条线宽度与预期相同
        assert line.get_linewidth() == lw
        # 断言线条线条风格与预期相同
        assert line.get_linestyle() == ls
    
    # 测试函数：验证在添加标签时的关键字参数设置
    def test_label(self, flat_series):
        # 绘制直方图，设置标签为"a label"
        ax = histplot(flat_series, label="a label")
        # 获取图例句柄和标签
        handles, labels = ax.get_legend_handles_labels()
        # 断言图例句柄数量为1
        assert len(handles) == 1
        # 断言标签内容与预期相同
        assert labels == ["a label"]
    
    # 测试函数：验证默认颜色和清理时的行为
    def test_default_color_scout_cleanup(self, flat_series):
        # 绘制直方图，默认行为
        ax = histplot(flat_series)
        # 断言容器数量为1，即绘制了一个容器
        assert len(ax.containers) == 1
class TestHistPlotBivariate:

    def test_mesh(self, long_df):
        # 创建 Histogram 实例
        hist = Histogram()
        # 使用 Histogram 实例处理数据，获取计数和边缘信息
        counts, (x_edges, y_edges) = hist(long_df["x"], long_df["y"])

        # 绘制直方图并获取网格对象
        ax = histplot(long_df, x="x", y="y")
        # 获取网格的第一个集合对象（mesh）
        mesh = ax.collections[0]
        # 获取网格对象中的数据数组
        mesh_data = mesh.get_array()

        # 断言：验证网格数据与计数数据一致
        assert_array_equal(mesh_data.data.flat, counts.T.flat)
        # 断言：验证网格数据的掩码与计数数据为零的位置一致
        assert_array_equal(mesh_data.mask.flat, counts.T.flat == 0)

        # 创建网格顶点的笛卡尔积迭代器
        edges = itertools.product(y_edges[:-1], x_edges[:-1])
        for i, (y, x) in enumerate(edges):
            # 获取网格对象的路径
            path = mesh.get_paths()[i]
            # 断言：验证路径的起始点与边缘坐标一致
            assert path.vertices[0, 0] == x
            assert path.vertices[0, 1] == y

    def test_mesh_with_hue(self, long_df):
        # 绘制带有色调参数的直方图
        ax = histplot(long_df, x="x", y="y", hue="c")

        # 创建 Histogram 实例并定义分bin参数
        hist = Histogram()
        hist.define_bin_params(long_df["x"], long_df["y"])

        # 根据颜色分组的子数据框迭代
        for i, sub_df in long_df.groupby("c"):
            # 获取当前颜色组的网格集合对象
            mesh = ax.collections[i]
            # 获取网格对象中的数据数组
            mesh_data = mesh.get_array()

            # 使用 Histogram 实例处理子数据，获取计数和边缘信息
            counts, (x_edges, y_edges) = hist(sub_df["x"], sub_df["y"])

            # 断言：验证网格数据与计数数据一致
            assert_array_equal(mesh_data.data.flat, counts.T.flat)
            # 断言：验证网格数据的掩码与计数数据为零的位置一致
            assert_array_equal(mesh_data.mask.flat, counts.T.flat == 0)

            # 创建网格顶点的笛卡尔积迭代器
            edges = itertools.product(y_edges[:-1], x_edges[:-1])
            for i, (y, x) in enumerate(edges):
                # 获取网格对象的路径
                path = mesh.get_paths()[i]
                # 断言：验证路径的起始点与边缘坐标一致
                assert path.vertices[0, 0] == x
                assert path.vertices[0, 1] == y

    def test_mesh_with_hue_unique_bins(self, long_df):
        # 绘制带有唯一分bin参数的直方图
        ax = histplot(long_df, x="x", y="y", hue="c", common_bins=False)

        # 根据颜色分组的子数据框迭代
        for i, sub_df in long_df.groupby("c"):
            # 创建 Histogram 实例
            hist = Histogram()

            # 获取当前颜色组的网格集合对象
            mesh = ax.collections[i]
            # 获取网格对象中的数据数组
            mesh_data = mesh.get_array()

            # 使用 Histogram 实例处理子数据，获取计数和边缘信息
            counts, (x_edges, y_edges) = hist(sub_df["x"], sub_df["y"])

            # 断言：验证网格数据与计数数据一致
            assert_array_equal(mesh_data.data.flat, counts.T.flat)
            # 断言：验证网格数据的掩码与计数数据为零的位置一致
            assert_array_equal(mesh_data.mask.flat, counts.T.flat == 0)

            # 创建网格顶点的笛卡尔积迭代器
            edges = itertools.product(y_edges[:-1], x_edges[:-1])
            for i, (y, x) in enumerate(edges):
                # 获取网格对象的路径
                path = mesh.get_paths()[i]
                # 断言：验证路径的起始点与边缘坐标一致
                assert path.vertices[0, 0] == x
                assert path.vertices[0, 1] == y

    def test_mesh_with_col_unique_bins(self, long_df):
        # 绘制带有唯一分bin参数和列参数的直方图
        g = displot(long_df, x="x", y="y", col="c", common_bins=False)

        # 根据颜色分组的子数据框迭代
        for i, sub_df in long_df.groupby("c"):
            # 创建 Histogram 实例
            hist = Histogram()

            # 获取当前颜色组的网格集合对象
            mesh = g.axes.flat[i].collections[0]
            # 获取网格对象中的数据数组
            mesh_data = mesh.get_array()

            # 使用 Histogram 实例处理子数据，获取计数和边缘信息
            counts, (x_edges, y_edges) = hist(sub_df["x"], sub_df["y"])

            # 断言：验证网格数据与计数数据一致
            assert_array_equal(mesh_data.data.flat, counts.T.flat)
            # 断言：验证网格数据的掩码与计数数据为零的位置一致
            assert_array_equal(mesh_data.mask.flat, counts.T.flat == 0)

            # 创建网格顶点的笛卡尔积迭代器
            edges = itertools.product(y_edges[:-1], x_edges[:-1])
            for i, (y, x) in enumerate(edges):
                # 获取网格对象的路径
                path = mesh.get_paths()[i]
                # 断言：验证路径的起始点与边缘坐标一致
                assert path.vertices[0, 0] == x
                assert path.vertices[0, 1] == y
    # 测试使用对数正态分布生成的数据进行直方图统计
    def test_mesh_log_scale(self, rng):

        # 生成对数正态分布的随机数据 x, y
        x, y = rng.lognormal(0, 1, (2, 1000))
        
        # 创建一个直方图对象
        hist = Histogram()
        
        # 计算直方图统计信息，并返回计数及其边界
        counts, (x_edges, y_edges) = hist(np.log10(x), np.log10(y))
        
        # 在双对数坐标上绘制直方图
        ax = histplot(x=x, y=y, log_scale=True)
        
        # 获取图中的网格数据
        mesh = ax.collections[0]
        
        # 获取网格的数据数组
        mesh_data = mesh.get_array()
        
        # 断言网格数据数组与直方图计数数据数组相等
        assert_array_equal(mesh_data.data.flat, counts.T.flat)
        
        # 生成所有可能的边界组合
        edges = itertools.product(y_edges[:-1], x_edges[:-1])
        
        # 遍历每个边界组合，验证网格路径的顶点坐标是否正确
        for i, (y_i, x_i) in enumerate(edges):
            path = mesh.get_paths()[i]
            assert path.vertices[0, 0] == pytest.approx(10 ** x_i)
            assert path.vertices[0, 1] == pytest.approx(10 ** y_i)

    # 测试网格的阈值功能
    def test_mesh_thresh(self, long_df):

        # 创建一个直方图对象
        hist = Histogram()
        
        # 计算直方图统计信息，并返回计数及其边界
        counts, (x_edges, y_edges) = hist(long_df["x"], long_df["y"])
        
        # 设置阈值
        thresh = 5
        
        # 在图上绘制直方图，并应用阈值
        ax = histplot(long_df, x="x", y="y", thresh=thresh)
        
        # 获取图中的网格数据
        mesh = ax.collections[0]
        
        # 获取网格的数据数组
        mesh_data = mesh.get_array()
        
        # 断言网格数据数组与直方图计数数据数组相等
        assert_array_equal(mesh_data.data.flat, counts.T.flat)
        
        # 断言网格数据的掩码与是否小于等于阈值的条件匹配
        assert_array_equal(mesh_data.mask.flat, (counts <= thresh).T.flat)

    # 测试网格的粘性边界功能
    def test_mesh_sticky_edges(self, long_df):

        # 在图上绘制直方图，并获取网格数据
        ax = histplot(long_df, x="x", y="y", thresh=None)
        mesh = ax.collections[0]
        
        # 断言网格的粘性 x, y 边界是否与数据框中的最小和最大值匹配
        assert mesh.sticky_edges.x == [long_df["x"].min(), long_df["x"].max()]
        assert mesh.sticky_edges.y == [long_df["y"].min(), long_df["y"].max()]

        # 清除图形并重新绘制直方图
        ax.clear()
        ax = histplot(long_df, x="x", y="y")
        mesh = ax.collections[0]
        
        # 断言网格没有粘性边界
        assert not mesh.sticky_edges.x
        assert not mesh.sticky_edges.y

    # 测试网格的共同归一化功能
    def test_mesh_common_norm(self, long_df):

        # 设置统计类型为密度
        stat = "density"
        
        # 在图上绘制直方图，并指定共同归一化及统计类型
        ax = histplot(
            long_df, x="x", y="y", hue="c", common_norm=True, stat=stat,
        )
        
        # 创建直方图对象，并定义数据桶参数
        hist = Histogram(stat="density")
        hist.define_bin_params(long_df["x"], long_df["y"])
        
        # 对数据框按类别分组，并为每个子数据框绘制网格
        for i, sub_df in long_df.groupby("c"):
            mesh = ax.collections[i]
            mesh_data = mesh.get_array()
            
            # 计算子数据框的密度及其边界
            density, (x_edges, y_edges) = hist(sub_df["x"], sub_df["y"])
            
            # 计算比例尺
            scale = len(sub_df) / len(long_df)
            
            # 断言网格数据数组与子数据框密度数据数组乘以比例尺后的结果相等
            assert_array_equal(mesh_data.data.flat, (density * scale).T.flat)

    # 测试网格的唯一归一化功能
    def test_mesh_unique_norm(self, long_df):

        # 设置统计类型为密度
        stat = "density"
        
        # 在图上绘制直方图，并指定不共同归一化及统计类型
        ax = histplot(
            long_df, x="x", y="y", hue="c", common_norm=False, stat=stat,
        )
        
        # 创建直方图对象
        hist = Histogram()
        
        # 定义数据桶参数
        bin_kws = hist.define_bin_params(long_df["x"], long_df["y"])
        
        # 对数据框按类别分组，并为每个子数据框绘制网格
        for i, sub_df in long_df.groupby("c"):
            sub_hist = Histogram(bins=bin_kws["bins"], stat=stat)
            mesh = ax.collections[i]
            mesh_data = mesh.get_array()
            
            # 计算子数据框的密度及其边界
            density, (x_edges, y_edges) = sub_hist(sub_df["x"], sub_df["y"])
            
            # 断言网格数据数组与子数据框密度数据数组相等
            assert_array_equal(mesh_data.data.flat, density.T.flat)

    @pytest.mark.parametrize("stat", ["probability", "proportion", "percent"])
    # 测试网格归一化功能
    def test_mesh_normalization(self, long_df, stat):
        # 创建直方图并指定 x, y 轴数据以及统计参数
        ax = histplot(
            long_df, x="x", y="y", stat=stat,
        )

        # 获取网格数据
        mesh_data = ax.collections[0].get_array()
        # 根据统计参数确定预期的总和，如果没有指定，默认为 1
        expected_sum = {"percent": 100}.get(stat, 1)
        # 断言网格数据的总和是否等于预期总和
        assert mesh_data.data.sum() == expected_sum

    # 测试网格颜色设置
    def test_mesh_colors(self, long_df):
        # 设定颜色为红色
        color = "r"
        # 创建图和坐标轴
        f, ax = plt.subplots()
        # 创建直方图并指定 x, y 轴数据以及颜色
        histplot(
            long_df, x="x", y="y", color=color,
        )
        # 获取网格对象
        mesh = ax.collections[0]
        # 断言当前网格颜色映射是否与红色对应的颜色映射相等
        assert_array_equal(
            mesh.get_cmap().colors,
            _DistributionPlotter()._cmap_from_color(color).colors,
        )

        # 创建新的图和坐标轴
        f, ax = plt.subplots()
        # 创建直方图并指定 x, y 轴数据以及色调
        histplot(
            long_df, x="x", y="y", hue="c",
        )
        # 获取色调调色板
        colors = color_palette()
        # 遍历每个网格对象
        for i, mesh in enumerate(ax.collections):
            # 断言当前网格颜色映射是否与对应颜色的调色板相等
            assert_array_equal(
                mesh.get_cmap().colors,
                _DistributionPlotter()._cmap_from_color(colors[i]).colors,
            )

    # 测试颜色限制设置
    def test_color_limits(self, long_df):
        # 创建包含三个子图的图和坐标轴
        f, (ax1, ax2, ax3) = plt.subplots(3)
        # 定义直方图参数
        kws = dict(data=long_df, x="x", y="y")
        hist = Histogram()
        # 计算 x, y 数据的直方图
        counts, _ = hist(long_df["x"], long_df["y"])

        # 绘制直方图在第一个子图上
        histplot(**kws, ax=ax1)
        # 断言第一个子图的网格对象颜色极限是否与计数的最大值匹配
        assert ax1.collections[0].get_clim() == (0, counts.max())

        # 设置最大值为 10 并在第二个子图上绘制直方图
        vmax = 10
        histplot(**kws, vmax=vmax, ax=ax2)
        # 重新计算 x, y 数据的直方图
        counts, _ = hist(long_df["x"], long_df["y"])
        # 断言第二个子图的网格对象颜色极限是否为 (0, vmax)
        assert ax2.collections[0].get_clim() == (0, vmax)

        # 设置最大百分比为 0.8，阈值为 0.1，计算百分位对应的水平函数
        pmax = .8
        pthresh = .1
        f = _DistributionPlotter()._quantile_to_level

        # 在第三个子图上绘制直方图
        histplot(**kws, pmax=pmax, pthresh=pthresh, ax=ax3)
        # 重新计算 x, y 数据的直方图
        counts, _ = hist(long_df["x"], long_df["y"])
        # 获取网格对象
        mesh = ax3.collections[0]
        # 断言第三个子图的网格对象颜色极限是否为 (0, 百分位对应的水平)
        assert mesh.get_clim() == (0, f(counts, pmax))
        # 断言网格对象数组的掩码是否与计数小于阈值对应的掩码相等
        assert_array_equal(
            mesh.get_array().mask.flat,
            (counts <= f(counts, pthresh)).T.flat,
        )
    # 定义一个测试方法，用于测试直方图绘制函数在不同设置下的行为
    def test_hue_color_limits(self, long_df):
        
        # 创建一个包含四个子图的图形布局，并获取各子图对象
        _, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        
        # 定义绘图关键字参数字典，包括数据、x轴、y轴、色彩（hue）、分箱数等信息
        kws = dict(data=long_df, x="x", y="y", hue="c", bins=4)

        # 创建直方图对象，设置分箱数
        hist = Histogram(bins=kws["bins"])
        
        # 根据数据列定义直方图的分箱参数
        hist.define_bin_params(long_df["x"], long_df["y"])
        
        # 计算整体数据的直方图统计信息
        full_counts, _ = hist(long_df["x"], long_df["y"])

        # 初始化子集统计列表
        sub_counts = []
        
        # 按照色彩（hue）分组，计算每个子集的直方图统计信息
        for _, sub_df in long_df.groupby(kws["hue"]):
            c, _ = hist(sub_df["x"], sub_df["y"])
            sub_counts.append(c)

        # 设置整体数据的颜色分布的最大值
        pmax = .8
        # 设置阈值
        pthresh = .05
        
        # 获取量化分位数转换级别的函数
        f = _DistributionPlotter()._quantile_to_level

        # 在第一个子图上绘制直方图
        histplot(**kws, common_norm=True, ax=ax1)
        # 验证每个网格是否符合预期颜色映射范围
        for i, mesh in enumerate(ax1.collections):
            assert mesh.get_clim() == (0, full_counts.max())

        # 在第二个子图上绘制直方图
        histplot(**kws, common_norm=False, ax=ax2)
        # 验证每个网格是否符合预期颜色映射范围
        for i, mesh in enumerate(ax2.collections):
            assert mesh.get_clim() == (0, sub_counts[i].max())

        # 在第三个子图上绘制直方图，并应用通用的颜色标准化和指定的最大值与阈值
        histplot(**kws, common_norm=True, pmax=pmax, pthresh=pthresh, ax=ax3)
        # 验证每个网格是否符合预期颜色映射范围和掩码
        for i, mesh in enumerate(ax3.collections):
            assert mesh.get_clim() == (0, f(full_counts, pmax))
            assert_array_equal(
                mesh.get_array().mask.flat,
                (sub_counts[i] <= f(full_counts, pthresh)).T.flat,
            )

        # 在第四个子图上绘制直方图，并应用非通用的颜色标准化和指定的最大值与阈值
        histplot(**kws, common_norm=False, pmax=pmax, pthresh=pthresh, ax=ax4)
        # 验证每个网格是否符合预期颜色映射范围和掩码
        for i, mesh in enumerate(ax4.collections):
            assert mesh.get_clim() == (0, f(sub_counts[i], pmax))
            assert_array_equal(
                mesh.get_array().mask.flat,
                (sub_counts[i] <= f(sub_counts[i], pthresh)).T.flat,
            )

    # 定义一个测试方法，用于测试带有颜色条的直方图绘制函数的行为
    def test_colorbar(self, long_df):
        
        # 创建一个包含主轴和颜色条轴的图形布局，并获取各轴对象
        f, ax = plt.subplots()
        # 绘制带颜色条的直方图
        histplot(long_df, x="x", y="y", cbar=True, ax=ax)
        # 验证图形中轴的数量是否符合预期（包括主轴和颜色条轴）
        assert len(ax.figure.axes) == 2

        # 创建一个包含主轴和自定义颜色条轴的图形布局，并获取各轴对象
        f, (ax, cax) = plt.subplots(2)
        # 绘制带颜色条的直方图，并指定颜色条轴
        histplot(long_df, x="x", y="y", cbar=True, cbar_ax=cax, ax=ax)
        # 验证图形中轴的数量是否符合预期（包括主轴和自定义颜色条轴）
        assert len(ax.figure.axes) == 2
# 定义一个测试类 TestECDFPlotUnivariate，继承自 SharedAxesLevelTests
class TestECDFPlotUnivariate(SharedAxesLevelTests):

    # 设置 func 方法为静态方法，使用 ecdfplot 函数
    func = staticmethod(ecdfplot)

    # 定义获取最后一条线的颜色的方法，参数为 ax 对象
    def get_last_color(self, ax):
        return to_rgb(ax.lines[-1].get_color())

    # 参数化测试方法，测试长向量的情况
    @pytest.mark.parametrize("variable", ["x", "y"])
    def test_long_vectors(self, long_df, variable):
        # 获取变量的向量
        vector = long_df[variable]
        # 构建不同类型的向量列表
        vectors = [
            variable, vector, vector.to_numpy(), vector.to_list(),
        ]

        # 创建图形和轴对象
        f, ax = plt.subplots()
        # 遍历不同类型的向量并绘制 ECDF 图
        for vector in vectors:
            ecdfplot(data=long_df, ax=ax, **{variable: vector})

        # 获取所有线条的 x 坐标数据
        xdata = [l.get_xdata() for l in ax.lines]
        # 对比两两线条的 x 坐标数据，确保相等
        for a, b in itertools.product(xdata, xdata):
            assert_array_equal(a, b)

        # 获取所有线条的 y 坐标数据
        ydata = [l.get_ydata() for l in ax.lines]
        # 对比两两线条的 y 坐标数据，确保相等
        for a, b in itertools.product(ydata, ydata):
            assert_array_equal(a, b)

    # 测试带有 hue 参数的情况
    def test_hue(self, long_df):
        # 绘制带有 hue 参数的 ECDF 图
        ax = ecdfplot(long_df, x="x", hue="a")

        # 对比每条线的颜色与色板中的颜色是否相等
        for line, color in zip(ax.lines[::-1], color_palette()):
            assert_colors_equal(line.get_color(), color)

    # 测试传入线条样式参数的情况
    def test_line_kwargs(self, long_df):
        # 定义颜色、线型和线宽
        color = "r"
        ls = "--"
        lw = 3
        # 绘制带有线条样式参数的 ECDF 图
        ax = ecdfplot(long_df, x="x", color=color, ls=ls, lw=lw)

        # 检查每条线的颜色、线型和线宽是否符合预期
        for line in ax.lines:
            assert_colors_equal(line.get_color(), color)
            assert line.get_linestyle() == ls
            assert line.get_linewidth() == lw

    # 参数化测试方法，测试绘图的 drawstyle 参数
    @pytest.mark.parametrize("data_var", ["x", "y"])
    def test_drawstyle(self, flat_series, data_var):
        # 绘制带有 drawstyle 参数的 ECDF 图
        ax = ecdfplot(**{data_var: flat_series})
        # 检查第一条线的绘图风格是否符合预期
        drawstyles = dict(x="steps-post", y="steps-pre")
        assert ax.lines[0].get_drawstyle() == drawstyles[data_var]

    # 参数化测试方法，测试 proportion_limits 参数
    @pytest.mark.parametrize(
        "data_var,stat_var", [["x", "y"], ["y", "x"]],
    )
    def test_proportion_limits(self, flat_series, data_var, stat_var):
        # 绘制带有 proportion_limits 参数的 ECDF 图
        ax = ecdfplot(**{data_var: flat_series})
        # 获取第一条线的数据并检查边界值是否正确
        data = getattr(ax.lines[0], f"get_{stat_var}data")()
        assert data[0] == 0
        assert data[-1] == 1
        sticky_edges = getattr(ax.lines[0].sticky_edges, stat_var)
        assert sticky_edges[:] == [0, 1]

    # 参数化测试方法，测试 complement 参数的情况
    @pytest.mark.parametrize(
        "data_var,stat_var", [["x", "y"], ["y", "x"]],
    )
    def test_proportion_limits_complementary(self, flat_series, data_var, stat_var):
        # 绘制带有 complement 参数的 ECDF 图
        ax = ecdfplot(**{data_var: flat_series}, complementary=True)
        # 获取第一条线的数据并检查边界值是否正确
        data = getattr(ax.lines[0], f"get_{stat_var}data")()
        assert data[0] == 1
        assert data[-1] == 0
        sticky_edges = getattr(ax.lines[0].sticky_edges, stat_var)
        assert sticky_edges[:] == [0, 1]
    # 测试计数统计方法，验证平坦系列的长度
    def test_proportion_count(self, flat_series, data_var, stat_var):
        # 计算平坦系列的长度
        n = len(flat_series)
        # 调用 ecdfplot 函数生成图形，并指定统计方法为 "count"
        ax = ecdfplot(**{data_var: flat_series}, stat="count")
        # 获取图中第一条线的数据点
        data = getattr(ax.lines[0], f"get_{stat_var}data")()
        # 断言第一个数据点为 0
        assert data[0] == 0
        # 断言最后一个数据点为 n
        assert data[-1] == n
        # 获取图中第一条线的 sticky_edges 属性对应统计变量的值
        sticky_edges = getattr(ax.lines[0].sticky_edges, stat_var)
        # 断言 sticky_edges 的值为 [0, n]
        assert sticky_edges[:] == [0, n]

    # 测试权重参数的处理
    def test_weights(self):
        # 调用 ecdfplot 函数绘制图形，指定 x 数据和权重
        ax = ecdfplot(x=[1, 2, 3], weights=[1, 1, 2])
        # 获取图中第一条线的 y 值数据
        y = ax.lines[0].get_ydata()
        # 断言 y 值数据与期望值相等
        assert_array_equal(y, [0, .25, .5, 1])

    # 测试双变量错误处理
    def test_bivariate_error(self, long_df):
        # 使用 pytest 检测是否会抛出 NotImplementedError 异常，匹配错误信息
        with pytest.raises(NotImplementedError, match="Bivariate ECDF plots"):
            # 调用 ecdfplot 函数，传入长格式数据和 x、y 列名
            ecdfplot(data=long_df, x="x", y="y")

    # 测试对数刻度处理
    def test_log_scale(self, long_df):
        # 创建一个包含两个子图的图像对象
        ax1, ax2 = plt.figure().subplots(2)
        # 在第一个子图上绘制非对数刻度的 ECDF 图
        ecdfplot(data=long_df, x="z", ax=ax1)
        # 在第二个子图上绘制对数刻度的 ECDF 图
        ecdfplot(data=long_df, x="z", log_scale=True, ax=ax2)
        # 获取每个图中第一条线的数据点（忽略第一个点，因为它可能是-inf或0）
        line1 = ax1.lines[0].get_xydata()[1:]
        line2 = ax2.lines[0].get_xydata()[1:]
        # 断言两条线的数据点近似相等
        assert_array_almost_equal(line1, line2)
# 定义一个测试类 TestDisPlot，用于测试 displot 函数的不同参数组合与其他图形函数的结果对比
class TestDisPlot:

    # 使用 pytest.mark.parametrize 装饰器，为测试方法 test_versus_single_histplot 参数化设置不同的 kwargs 参数
    @pytest.mark.parametrize(
        "kwargs", [
            dict(),  # 空参数字典
            dict(x="x"),  # 设置 x 参数为 "x"
            dict(x="t"),  # 设置 x 参数为 "t"
            dict(x="a"),  # 设置 x 参数为 "a"
            dict(x="z", log_scale=True),  # 设置 x 参数为 "z"，并启用对数刻度
            dict(x="x", binwidth=4),  # 设置 x 参数为 "x"，并指定 binwidth 为 4
            dict(x="x", weights="f", bins=5),  # 设置 x 参数为 "x"，指定 weights 为 "f"，bins 为 5
            dict(x="x", color="green", linewidth=2, binwidth=4),  # 设置 x 参数为 "x"，并指定颜色为绿色，线宽为2，binwidth 为 4
            dict(x="x", hue="a", fill=False),  # 设置 x 参数为 "x"，hue 参数为 "a"，fill 参数为 False
            dict(x="y", hue="a", fill=False),  # 设置 x 参数为 "y"，hue 参数为 "a"，fill 参数为 False
            dict(x="x", hue="a", multiple="stack"),  # 设置 x 参数为 "x"，hue 参数为 "a"，multiple 参数为 "stack"
            dict(x="x", hue="a", element="step"),  # 设置 x 参数为 "x"，hue 参数为 "a"，element 参数为 "step"
            dict(x="x", hue="a", palette="muted"),  # 设置 x 参数为 "x"，hue 参数为 "a"，palette 参数为 "muted"
            dict(x="x", hue="a", kde=True),  # 设置 x 参数为 "x"，hue 参数为 "a"，启用核密度估计 kde
            dict(x="x", hue="a", stat="density", common_norm=False),  # 设置 x 参数为 "x"，hue 参数为 "a"，stat 参数为 "density"，common_norm 参数为 False
            dict(x="x", y="y"),  # 设置 x 参数为 "x"，y 参数为 "y"
        ],
    )
    # 定义测试方法 test_versus_single_histplot，用于测试 histplot 函数与 displot 函数的结果对比
    def test_versus_single_histplot(self, long_df, kwargs):

        # 调用 histplot 函数生成 ax 对象
        ax = histplot(long_df, **kwargs)
        # 调用 displot 函数生成 g 对象
        g = displot(long_df, **kwargs)
        # 断言 ax 对象与 g 对象的子图 ax 相等
        assert_plots_equal(ax, g.ax)

        # 如果 ax 对象有图例，则断言 ax 的图例与 g 的图例 _legend 相等
        if ax.legend_ is not None:
            assert_legends_equal(ax.legend_, g._legend)

        # 如果有参数传入，则在 long_df 数据框中添加一个列 "_"
        if kwargs:
            long_df["_"] = "_"
            # 调用 displot 函数生成 g2 对象，并指定列 "col" 为 "_"
            g2 = displot(long_df, col="_", **kwargs)
            # 断言 ax 对象与 g2 对象的子图 ax 相等
            assert_plots_equal(ax, g2.ax)

    # 使用 pytest.mark.parametrize 装饰器，为测试方法 test_versus_single_kdeplot 参数化设置不同的 kwargs 参数
    @pytest.mark.parametrize(
        "kwargs", [
            dict(),  # 空参数字典
            dict(x="x"),  # 设置 x 参数为 "x"
            dict(x="t"),  # 设置 x 参数为 "t"
            dict(x="z", log_scale=True),  # 设置 x 参数为 "z"，并启用对数刻度
            dict(x="x", bw_adjust=.5),  # 设置 x 参数为 "x"，并指定 bw_adjust 为 0.5
            dict(x="x", weights="f"),  # 设置 x 参数为 "x"，指定 weights 为 "f"
            dict(x="x", color="green", linewidth=2),  # 设置 x 参数为 "x"，指定颜色为绿色，线宽为 2
            dict(x="x", hue="a", multiple="stack"),  # 设置 x 参数为 "x"，hue 参数为 "a"，multiple 参数为 "stack"
            dict(x="x", hue="a", fill=True),  # 设置 x 参数为 "x"，hue 参数为 "a"，fill 参数为 True
            dict(x="y", hue="a", fill=False),  # 设置 x 参数为 "y"，hue 参数为 "a"，fill 参数为 False
            dict(x="x", hue="a", palette="muted"),  # 设置 x 参数为 "x"，hue 参数为 "a"，palette 参数为 "muted"
            dict(x="x", y="y"),  # 设置 x 参数为 "x"，y 参数为 "y"
        ],
    )
    # 定义测试方法 test_versus_single_kdeplot，用于测试 kdeplot 函数与 displot 函数的结果对比
    def test_versus_single_kdeplot(self, long_df, kwargs):

        # 调用 kdeplot 函数生成 ax 对象
        ax = kdeplot(data=long_df, **kwargs)
        # 调用 displot 函数生成 g 对象，并设置 kind 参数为 "kde"
        g = displot(long_df, kind="kde", **kwargs)
        # 断言 ax 对象与 g 对象的子图 ax 相等
        assert_plots_equal(ax, g.ax)

        # 如果 ax 对象有图例，则断言 ax 的图例与 g 的图例 _legend 相等
        if ax.legend_ is not None:
            assert_legends_equal(ax.legend_, g._legend)

        # 如果有参数传入，则在 long_df 数据框中添加一个列 "_"
        if kwargs:
            long_df["_"] = "_"
            # 调用 displot 函数生成 g2 对象，并设置 kind 参数为 "kde"，列 "col" 为 "_"
            g2 = displot(long_df, kind="kde", col="_", **kwargs)
            # 断言 ax 对象与 g2 对象的子图 ax 相等
            assert_plots_equal(ax, g2.ax)

    # 使用 pytest.mark.parametrize 装饰器，为测试方法参数化设置不同的 kwargs 参数
    @pytest.mark.parametrize(
        "kwargs", [
            dict(),  # 空参数字典
            dict(x="x"),  # 设置 x 参数为 "x"
            dict(x="t"),  # 设置 x 参数为 "t"
            dict(x="z", log_scale=True),  # 设置 x 参数为 "z"，并启用对数刻度
            dict(x="x", weights="f"),  # 设置 x 参数为 "x"，指定 weights 为 "f"
            dict(y="x"),  # 设置 y 参数为 "x"
            dict(x="x", color="green", linewidth=2),  # 设置 x 参数为 "x"，指定颜色为绿色，线宽为 2
            dict(x="x", hue="a", complementary=True),  # 设置 x 参数为 "x"，hue 参数为 "a"，complementary 参数为 True
            dict(x="x", hue="a", stat="count"),  # 设置 x 参数为 "x"，hue 参数为 "a"，stat 参数为 "count"
            dict(x="x", hue="a", palette="muted"),  # 设置 x 参数为 "x"，hue 参数为 "a"，palette 参数为 "muted"
        ],
    )
    # 定义测试方法
    def test_versus_single_ecdfplot(self, long_df, kwargs):
    def test_versus_single_ecdfplot(self, long_df, kwargs):
        # 使用长格式数据和参数绘制 ECDF 图，并获取图形对象 ax
        ax = ecdfplot(data=long_df, **kwargs)
        # 使用长格式数据和参数绘制 displot，kind="ecdf"，获取图形对象 g
        g = displot(long_df, kind="ecdf", **kwargs)
        # 断言 ax 和 g.ax 的图形内容相等
        assert_plots_equal(ax, g.ax)

        # 如果 ax 中有图例，则断言 ax.legend_ 和 g._legend 相等
        if ax.legend_ is not None:
            assert_legends_equal(ax.legend_, g._legend)

        # 如果有传入参数 kwargs，则在长格式数据中添加列 "_"，再次绘制 displot
        if kwargs:
            long_df["_"] = "_"
            g2 = displot(long_df, kind="ecdf", col="_", **kwargs)
            # 断言 ax 和 g2.ax 的图形内容相等
            assert_plots_equal(ax, g2.ax)

    @pytest.mark.parametrize(
        "kwargs", [
            dict(x="x"),
            dict(x="x", y="y"),
            dict(x="x", hue="a"),
        ]
    )
    def test_with_rug(self, long_df, kwargs):
        # 创建一个新的 subplot axes 对象 ax
        ax = plt.figure().subplots()
        # 使用长格式数据和参数绘制 histplot，将图形绘制在 ax 上
        histplot(data=long_df, **kwargs, ax=ax)
        # 使用长格式数据和参数绘制 rugplot，将图形绘制在 ax 上
        rugplot(data=long_df, **kwargs, ax=ax)

        # 使用长格式数据和参数绘制 displot，设置 rug=True
        g = displot(long_df, rug=True, **kwargs)

        # 断言 ax 和 g.ax 的图形内容相等，不包括标签
        assert_plots_equal(ax, g.ax, labels=False)

        # 在长格式数据中添加列 "_"，再次绘制 displot
        long_df["_"] = "_"
        g2 = displot(long_df, col="_", rug=True, **kwargs)

        # 断言 ax 和 g2.ax 的图形内容相等，不包括标签
        assert_plots_equal(ax, g2.ax, labels=False)

    @pytest.mark.parametrize(
        "facet_var", ["col", "row"],
    )
    def test_facets(self, long_df, facet_var):
        # 设置 kwargs，根据 facet_var 的值设置列或行变量
        kwargs = {facet_var: "a"}
        # 使用长格式数据绘制 kdeplot，设置 x="x"，hue="a"，获取图形对象 ax
        ax = kdeplot(data=long_df, x="x", hue="a")
        # 使用长格式数据绘制 displot，设置 x="x"，kind="kde"，根据 kwargs 设置列或行
        g = displot(long_df, x="x", kind="kde", **kwargs)

        # 获取 ax 图例的文本对象
        legend_texts = ax.legend_.get_texts()

        # 对 ax 的每条线进行迭代，比较 facet_ax 中的第一条线的数据
        for i, line in enumerate(ax.lines[::-1]):
            facet_ax = g.axes.flat[i]
            facet_line = facet_ax.lines[0]
            # 断言 ax 和 facet_ax 的第一条线的数据相等
            assert_array_equal(line.get_xydata(), facet_line.get_xydata())

            # 获取 legend_texts[i] 的文本，断言其在 facet_ax 的标题中
            text = legend_texts[i].get_text()
            assert text in facet_ax.get_title()

    @pytest.mark.parametrize("multiple", ["dodge", "stack", "fill"])
    def test_facet_multiple(self, long_df, multiple):
        # 设置 bins 的等分点
        bins = np.linspace(0, 20, 5)
        # 使用长格式数据中 c 列等于 0 的部分数据绘制 histplot，设置 x="x"，hue="a"，hue_order=["a", "b", "c"]，multiple=multiple，bins=bins
        ax = histplot(
            data=long_df[long_df["c"] == 0],
            x="x", hue="a", hue_order=["a", "b", "c"],
            multiple=multiple, bins=bins,
        )

        # 使用长格式数据绘制 displot，设置 x="x"，hue="a"，col="c"，hue_order=["a", "b", "c"]，multiple=multiple，bins=bins
        g = displot(
            data=long_df, x="x", hue="a", col="c", hue_order=["a", "b", "c"],
            multiple=multiple, bins=bins,
        )

        # 断言 ax 和 g.axes_dict[0] 的图形内容相等
        assert_plots_equal(ax, g.axes_dict[0])

    def test_ax_warning(self, long_df):
        # 创建一个新的 subplot axes 对象 ax
        ax = plt.figure().subplots()
        # 使用长格式数据绘制 displot，设置 x="x"，将图形绘制在 ax 上
        with pytest.warns(UserWarning, match="`displot` is a figure-level"):
            displot(long_df, x="x", ax=ax)

    @pytest.mark.parametrize("key", ["col", "row"])
    def test_array_faceting(self, long_df, key):
        # 将长格式数据中的 a 列转换为 NumPy 数组 a
        a = long_df["a"].to_numpy()
        # 获取 a 列的分类顺序值 vals
        vals = categorical_order(a)
        # 使用长格式数据绘制 displot，设置 x="x"，根据 key 设置列或行
        g = displot(long_df, x="x", **{key: a})
        # 断言 g.axes.flat 的长度等于 vals 的长度
        assert len(g.axes.flat) == len(vals)
        # 对 g.axes.flat 和 vals 进行迭代，断言 val 在 ax 的标题中
        for ax, val in zip(g.axes.flat, vals):
            assert val in ax.get_title()

    def test_legend(self, long_df):
        # 使用长格式数据绘制 displot，设置 x="x"，hue="a"，获取图形对象 g
        g = displot(long_df, x="x", hue="a")
        # 断言 g._legend 不为 None
        assert g._legend is not None

    def test_empty(self):
        # 绘制一个空的 displot，传入空列表 x=[]，y=[]
        g = displot(x=[], y=[])
        # 断言 g 是 FacetGrid 类的实例
        assert isinstance(g, FacetGrid)
    # 测试函数，用于测试双变量 ECDF 绘图中的错误处理
    def test_bivariate_ecdf_error(self, long_df):
        
        # 使用 pytest 检测是否会抛出 NotImplementedError 异常
        with pytest.raises(NotImplementedError):
            # 调用 displot 绘制双变量 ECDF 图
            displot(long_df, x="x", y="y", kind="ecdf")

    # 测试函数，用于测试双变量 KDE 绘图中的正常化
    def test_bivariate_kde_norm(self, rng):

        # 生成两个正态分布样本
        x, y = rng.normal(0, 1, (2, 100))
        # 创建一个分组指示变量 z
        z = [0] * 80 + [1] * 20

        # 定义一个函数，用于计算轮廓线数量
        def count_contours(ax):
            # 如果 matplotlib 版本早于 3.8.0rc1，使用不同方法获取轮廓线坐标
            if _version_predates(mpl, "3.8.0rc1"):
                return sum(bool(get_contour_coords(c)) for c in ax.collections)
            else:
                # 否则，使用另一种方法获取轮廓线路径的数量
                return sum(bool(p.vertices.size) for p in ax.collections[0].get_paths())

        # 绘制双变量 KDE 图，并获取返回的图形对象 g
        g = displot(x=x, y=y, col=z, kind="kde", levels=10)
        # 计算第一个子图中的轮廓线数量
        l1 = count_contours(g.axes.flat[0])
        # 计算第二个子图中的轮廓线数量
        l2 = count_contours(g.axes.flat[1])
        # 断言第一个子图中的轮廓线数量大于第二个子图中的轮廓线数量
        assert l1 > l2

        # 再次绘制双变量 KDE 图，关闭共同正态化选项，并获取返回的图形对象 g
        g = displot(x=x, y=y, col=z, kind="kde", levels=10, common_norm=False)
        # 计算第一个子图中的轮廓线数量
        l1 = count_contours(g.axes.flat[0])
        # 计算第二个子图中的轮廓线数量
        l2 = count_contours(g.axes.flat[1])
        # 断言第一个子图中的轮廓线数量等于第二个子图中的轮廓线数量
        assert l1 == l2

    # 测试函数，用于测试双变量直方图绘图中的正常化
    def test_bivariate_hist_norm(self, rng):

        # 生成两个正态分布样本
        x, y = rng.normal(0, 1, (2, 100))
        # 创建一个分组指示变量 z
        z = [0] * 80 + [1] * 20

        # 绘制双变量直方图，并获取返回的图形对象 g
        g = displot(x=x, y=y, col=z, kind="hist")
        # 获取第一个子图中的色块颜色限制
        clim1 = g.axes.flat[0].collections[0].get_clim()
        # 获取第二个子图中的色块颜色限制
        clim2 = g.axes.flat[1].collections[0].get_clim()
        # 断言第一个子图中的色块颜色上限等于第二个子图中的色块颜色上限
        assert clim1 == clim2

        # 再次绘制双变量直方图，关闭共同正态化选项，并获取返回的图形对象 g
        g = displot(x=x, y=y, col=z, kind="hist", common_norm=False)
        # 获取第一个子图中的色块颜色限制
        clim1 = g.axes.flat[0].collections[0].get_clim()
        # 获取第二个子图中的色块颜色限制
        clim2 = g.axes.flat[1].collections[0].get_clim()
        # 断言第一个子图中的色块颜色上限大于第二个子图中的色块颜色上限
        assert clim1[1] > clim2[1]

    # 测试函数，用于测试 FacetGrid 数据处理的正确性
    def test_facetgrid_data(self, long_df):

        # 将长格式数据框转换为字典列表，并绘制 FacetGrid 图形对象 g
        g = displot(
            data=long_df.to_dict(orient="list"),
            x="z",
            hue=long_df["a"].rename("hue_var"),
            col=long_df["c"].to_numpy(),
        )
        # 期望的列集合，包括长格式数据框的所有列名和两个额外的列 "hue_var" 和 "_col_"
        expected_cols = set(long_df.columns.to_list() + ["hue_var", "_col_"])
        # 断言图形对象 g 中的数据列集合等于期望的列集合
        assert set(g.data.columns) == expected_cols
        # 断言图形对象 g 中的 "hue_var" 列与长格式数据框中的 "a" 列数据相等
        assert_array_equal(g.data["hue_var"], long_df["a"])
        # 断言图形对象 g 中的 "_col_" 列与长格式数据框中的 "c" 列数据相等
        assert_array_equal(g.data["_col_"], long_df["c"])
# 定义一个函数用于简单的数值积分，用于测试 KDE（Kernel Density Estimation）代码。
def integrate(y, x):
    # 将输入的 y 和 x 转换为 NumPy 数组
    y = np.asarray(y)
    x = np.asarray(x)
    # 计算 x 的差分，即每个区间的宽度
    dx = np.diff(x)
    # 使用梯形法则计算数值积分，返回结果
    return (dx * y[:-1] + dx * y[1:]).sum() / 2
```