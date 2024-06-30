# `D:\src\scipysrc\seaborn\tests\_marks\test_bar.py`

```
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba, to_rgba_array

import pytest  # 导入 pytest 测试框架
from numpy.testing import assert_array_equal  # 导入 numpy 的数组比较函数

from seaborn._core.plot import Plot  # 导入 seaborn 绘图核心模块的 Plot 类
from seaborn._marks.bar import Bar, Bars  # 导入 seaborn 柱状图相关的 Bar 和 Bars 类


class TestBar:

    def plot_bars(self, variables, mark_kws, layer_kws):
        # 创建 Plot 对象并添加 Bar 标记，根据指定的变量和图层参数进行绘图
        p = Plot(**variables).add(Bar(**mark_kws), **layer_kws).plot()
        ax = p._figure.axes[0]  # 获取绘图对象的第一个坐标轴
        return [bar for barlist in ax.containers for bar in barlist]

    def check_bar(self, bar, x, y, width, height):
        # 检查柱状图的位置和大小是否与预期相符
        assert bar.get_x() == pytest.approx(x)
        assert bar.get_y() == pytest.approx(y)
        assert bar.get_width() == pytest.approx(width)
        assert bar.get_height() == pytest.approx(height)

    def test_categorical_positions_vertical(self):
        # 测试垂直分类变量的柱状图位置
        x = ["a", "b"]
        y = [1, 2]
        w = .8
        bars = self.plot_bars({"x": x, "y": y}, {}, {})
        for i, bar in enumerate(bars):
            self.check_bar(bar, i - w / 2, 0, w, y[i])

    def test_categorical_positions_horizontal(self):
        # 测试水平分类变量的柱状图位置
        x = [1, 2]
        y = ["a", "b"]
        w = .8
        bars = self.plot_bars({"x": x, "y": y}, {}, {})
        for i, bar in enumerate(bars):
            self.check_bar(bar, 0, i - w / 2, x[i], w)

    def test_numeric_positions_vertical(self):
        # 测试垂直数值变量的柱状图位置
        x = [1, 2]
        y = [3, 4]
        w = .8
        bars = self.plot_bars({"x": x, "y": y}, {}, {})
        for i, bar in enumerate(bars):
            self.check_bar(bar, x[i] - w / 2, 0, w, y[i])

    def test_numeric_positions_horizontal(self):
        # 测试水平数值变量的柱状图位置
        x = [1, 2]
        y = [3, 4]
        w = .8
        bars = self.plot_bars({"x": x, "y": y}, {}, {"orient": "h"})
        for i, bar in enumerate(bars):
            self.check_bar(bar, 0, y[i] - w / 2, x[i], w)

    def test_set_properties(self):
        # 测试设置柱状图属性
        x = ["a", "b", "c"]
        y = [1, 3, 2]

        mark = Bar(
            color=".8",
            alpha=.5,
            edgecolor=".3",
            edgealpha=.9,
            edgestyle=(2, 1),
            edgewidth=1.5,
        )

        p = Plot(x, y).add(mark).plot()
        ax = p._figure.axes[0]
        for bar in ax.patches:
            # 检查柱状图的填充颜色和透明度是否与预期相符
            assert bar.get_facecolor() == to_rgba(mark.color, mark.alpha)
            # 检查柱状图的边缘颜色和透明度是否与预期相符
            assert bar.get_edgecolor() == to_rgba(mark.edgecolor, mark.edgealpha)
            # 根据绘图方法中的注释，这里需要做一些调整以获取正确的线宽
            assert bar.get_linewidth() == mark.edgewidth * 2
            # 根据绘图方法中的注释，这里需要计算预期的虚线样式
            expected_dashes = (mark.edgestyle[0] / 2, mark.edgestyle[1] / 2)
            assert bar.get_linestyle() == (0, expected_dashes)
    # 测试用例：测试绘图对象的属性映射

    # 创建包含字符串列表的变量 x 和整数列表的变量 y
    x = ["a", "b"]
    y = [1, 2]
    # 创建一个透明度为 0.2 的柱状图对象
    mark = Bar(alpha=.2)
    # 创建绘图对象 p，使用 x 和 y 进行初始化，并设置颜色属性为 x，边框宽度属性为 y，再添加 mark 对象，并生成图形
    p = Plot(x, y, color=x, edgewidth=y).add(mark).plot()
    # 获取 p 对象的第一个子图的轴对象
    ax = p._figure.axes[0]
    # 获取 p 对象的主题中的颜色循环并转化为列表 colors
    colors = p._theme["axes.prop_cycle"].by_key()["color"]
    # 遍历轴对象中的所有 patches（图形中的基本图元），并进行断言检查
    for i, bar in enumerate(ax.patches):
        # 断言柱状图的填充颜色等于 colors[i] 的 RGBA 值，透明度为 mark.alpha
        assert bar.get_facecolor() == to_rgba(colors[i], mark.alpha)
        # 断言柱状图的边框颜色等于 colors[i] 的 RGBA 值，透明度为 1
        assert bar.get_edgecolor() == to_rgba(colors[i], 1)
    # 断言第一个 patch 的边框宽度小于第二个 patch 的边框宽度
    assert ax.patches[0].get_linewidth() < ax.patches[1].get_linewidth()

    # 测试用例：测试跳过零高度的情况

    # 创建一个柱状图对象 p，使用 ["a", "b", "c"] 和 [1, 0, 2] 进行初始化，并添加柱状图对象 Bar()，然后生成图形
    p = Plot(["a", "b", "c"], [1, 0, 2]).add(Bar()).plot()
    # 获取 p 对象的第一个子图的轴对象
    ax = p._figure.axes[0]
    # 断言轴对象中 patches 的数量为 2
    assert len(ax.patches) == 2

    # 测试用例：测试艺术家关键字参数 clip

    # 创建一个柱状图对象 p，使用 ["a", "b"] 和 [1, 2] 进行初始化，并添加具有 {"clip_on": False} 参数的柱状图对象 Bar()，然后生成图形
    p = Plot(["a", "b"], [1, 2]).add(Bar({"clip_on": False})).plot()
    # 获取 p 对象的第一个子图的轴对象中的第一个 patch
    patch = p._figure.axes[0].patches[0]
    # 断言 patch 的 clipbox 属性为 None，即不进行裁剪
    assert patch.clipbox is None
# 定义一个测试类 TestBars，用于测试 Bars 图形对象的行为
class TestBars:

    # 定义 pytest 的装饰器，为测试方法 x 提供测试数据
    @pytest.fixture
    def x(self):
        # 返回一个包含整数的 Pandas Series，用作测试数据
        return pd.Series([4, 5, 6, 7, 8], name="x")

    # 定义 pytest 的装饰器，为测试方法 y 提供测试数据
    @pytest.fixture
    def y(self):
        # 返回一个包含整数的 Pandas Series，用作测试数据
        return pd.Series([2, 8, 3, 5, 9], name="y")

    # 定义 pytest 的装饰器，为测试方法 color 提供测试数据
    @pytest.fixture
    def color(self):
        # 返回一个包含字符串的 Pandas Series，用作测试数据
        return pd.Series(["a", "b", "c", "a", "c"], name="color")

    # 测试方法，验证垂直方向的柱状图位置
    def test_positions(self, x, y):
        # 创建 Plot 对象，并添加 Bars 图形对象，绘制图形并获取绘图对象
        p = Plot(x, y).add(Bars()).plot()
        # 获取绘图对象的第一个坐标轴
        ax = p._figure.axes[0]
        # 获取第一个集合的路径集合
        paths = ax.collections[0].get_paths()
        # 断言路径数量等于数据 x 的长度
        assert len(paths) == len(x)
        # 遍历路径集合，并逐一验证柱状图的位置
        for i, path in enumerate(paths):
            verts = path.vertices
            # 验证柱状图左边缘 x 坐标
            assert verts[0, 0] == pytest.approx(x[i] - .5)
            # 验证柱状图右边缘 x 坐标
            assert verts[1, 0] == pytest.approx(x[i] + .5)
            # 验证柱状图底部 y 坐标
            assert verts[0, 1] == 0
            # 验证柱状图顶部 y 坐标
            assert verts[3, 1] == y[i]

    # 测试方法，验证水平方向的柱状图位置
    def test_positions_horizontal(self, x, y):
        # 创建 Plot 对象，并添加水平方向的 Bars 图形对象，绘制图形并获取绘图对象
        p = Plot(x=y, y=x).add(Bars(), orient="h").plot()
        # 获取绘图对象的第一个坐标轴
        ax = p._figure.axes[0]
        # 获取第一个集合的路径集合
        paths = ax.collections[0].get_paths()
        # 断言路径数量等于数据 x 的长度
        assert len(paths) == len(x)
        # 遍历路径集合，并逐一验证水平柱状图的位置
        for i, path in enumerate(paths):
            verts = path.vertices
            # 验证柱状图底部 y 坐标
            assert verts[0, 1] == pytest.approx(x[i] - .5)
            # 验证柱状图顶部 y 坐标
            assert verts[3, 1] == pytest.approx(x[i] + .5)
            # 验证柱状图左边缘 x 坐标
            assert verts[0, 0] == 0
            # 验证柱状图右边缘 x 坐标
            assert verts[1, 0] == y[i]

    # 测试方法，验证柱状图的宽度
    def test_width(self, x, y):
        # 创建 Plot 对象，并添加具有指定宽度的 Bars 图形对象，绘制图形并获取绘图对象
        p = Plot(x, y).add(Bars(width=.4)).plot()
        # 获取绘图对象的第一个坐标轴
        ax = p._figure.axes[0]
        # 获取第一个集合的路径集合
        paths = ax.collections[0].get_paths()
        # 遍历路径集合，并逐一验证柱状图的宽度
        for i, path in enumerate(paths):
            verts = path.vertices
            # 验证柱状图左边缘 x 坐标
            assert verts[0, 0] == pytest.approx(x[i] - .2)
            # 验证柱状图右边缘 x 坐标
            assert verts[1, 0] == pytest.approx(x[i] + .2)

    # 测试方法，验证映射颜色和直接设置透明度
    def test_mapped_color_direct_alpha(self, x, y, color):
        # 设置透明度
        alpha = .5
        # 创建 Plot 对象，并添加 Bars 图形对象，绘制图形并获取绘图对象
        p = Plot(x, y, color=color).add(Bars(alpha=alpha)).plot()
        # 获取绘图对象的第一个坐标轴
        ax = p._figure.axes[0]
        # 获取第一个集合的填充颜色
        fcs = ax.collections[0].get_facecolors()
        # 获取主题中的颜色循环，按键 "color" 获取颜色
        C0, C1, C2, *_ = p._theme["axes.prop_cycle"].by_key()["color"]
        # 期望的颜色数组，包含设置的透明度
        expected = to_rgba_array([C0, C1, C2, C0, C2], alpha)
        # 断言填充颜色与期望的颜色数组相等
        assert_array_equal(fcs, expected)

    # 测试方法，验证映射边框宽度
    def test_mapped_edgewidth(self, x, y):
        # 创建 Plot 对象，并添加 Bars 图形对象，绘制图形并获取绘图对象
        p = Plot(x, y, edgewidth=y).add(Bars()).plot()
        # 获取绘图对象的第一个坐标轴
        ax = p._figure.axes[0]
        # 获取第一个集合的线宽度
        lws = ax.collections[0].get_linewidths()
        # 断言线宽度按 y 数据排序后与 y 数据排序后的结果相等
        assert_array_equal(np.argsort(lws), np.argsort(y))

    # 测试方法，验证自动计算边框宽度
    def test_auto_edgewidth(self):
        # 创建两个不同范围的数据集
        x0 = np.arange(10)
        x1 = np.arange(1000)
        # 创建 Plot 对象，并添加 Bars 图形对象，绘制图形并获取绘图对象
        p0 = Plot(x0, x0).add(Bars()).plot()
        p1 = Plot(x1, x1).add(Bars()).plot()
        # 获取第一个图形对象的线宽度
        lw0 = p0._figure.axes[0].collections[0].get_linewidths()
        # 获取第二个图形对象的线宽度
        lw1 = p1._figure.axes[0].collections[0].get_linewidths()
        # 断言第一个图形对象的线宽度大于第二个图形对象的线宽度
        assert (lw0 > lw1).all()
    # 定义一个测试函数，测试未填充的条形图效果
    def test_unfilled(self, x, y):
        # 创建一个绘图对象，并添加未填充的条形图
        p = Plot(x, y).add(Bars(fill=False, edgecolor="C4")).plot()
        # 获取绘图对象的第一个坐标轴
        ax = p._figure.axes[0]
        # 获取第一个集合对象（条形图）的填充颜色
        fcs = ax.collections[0].get_facecolors()
        # 获取第一个集合对象（条形图）的边缘颜色
        ecs = ax.collections[0].get_edgecolors()
        # 获取绘图对象的颜色主题中的颜色列表
        colors = p._theme["axes.prop_cycle"].by_key()["color"]
        # 断言填充颜色数组与指定颜色的 RGBA 数组相等
        assert_array_equal(fcs, to_rgba_array([colors[0]] * len(x), 0))
        # 断言边缘颜色数组与指定颜色的 RGBA 数组相等
        assert_array_equal(ecs, to_rgba_array([colors[4]] * len(x), 1))

    # 定义一个测试函数，测试对数比例尺下的条形图效果
    def test_log_scale(self):
        # 定义 x 和 y 轴数据
        x = y = [1, 10, 100, 1000]
        # 创建一个绘图对象，并添加条形图，指定 x 轴使用对数比例尺
        p = Plot(x, y).add(Bars()).scale(x="log").plot()
        # 获取绘图对象的第一个坐标轴
        ax = p._figure.axes[0]

        # 获取第一个集合对象（条形图）的路径对象列表
        paths = ax.collections[0].get_paths()
        # 遍历路径对象列表的每对相邻路径
        for a, b in zip(paths, paths[1:]):
            # 断言每对相邻路径的第二个顶点 x 坐标近似相等
            assert a.vertices[1, 0] == pytest.approx(b.vertices[0, 0])
```