# `D:\src\scipysrc\seaborn\tests\_marks\test_line.py`

```
import numpy as np
import matplotlib as mpl
from matplotlib.colors import same_color, to_rgba

from numpy.testing import assert_array_equal, assert_array_almost_equal

from seaborn._core.plot import Plot  # 导入 Plot 类
from seaborn._core.moves import Dodge  # 导入 Dodge 类
from seaborn._marks.line import Dash, Line, Path, Lines, Paths, Range  # 导入各种线型类

class TestPath:

    def test_xy_data(self):
        x = [1, 5, 3, np.nan, 2]  # x 坐标数据，包含 NaN 值
        y = [1, 4, 2, 5, 3]  # y 坐标数据
        g = [1, 2, 1, 1, 2]  # 分组数据
        p = Plot(x=x, y=y, group=g).add(Path()).plot()  # 创建 Plot 对象并绘制路径图
        line1, line2 = p._figure.axes[0].get_lines()  # 获取绘图中的两条线

        assert_array_equal(line1.get_xdata(), [1, 3, np.nan])  # 断言第一条线的 x 数据
        assert_array_equal(line1.get_ydata(), [1, 2, np.nan])  # 断言第一条线的 y 数据
        assert_array_equal(line2.get_xdata(), [5, 2])  # 断言第二条线的 x 数据
        assert_array_equal(line2.get_ydata(), [4, 3])  # 断言第二条线的 y 数据

    def test_shared_colors_direct(self):
        x = y = [1, 2, 3]  # 共享 x 和 y 坐标
        color = ".44"  # 颜色值
        m = Path(color=color)  # 创建 Path 对象并指定颜色
        p = Plot(x=x, y=y).add(m).plot()  # 创建 Plot 对象并绘制图形
        line, = p._figure.axes[0].get_lines()  # 获取绘图中的线

        assert same_color(line.get_color(), color)  # 断言线的颜色与指定颜色相同
        assert same_color(line.get_markeredgecolor(), color)  # 断言线的标记边缘颜色与指定颜色相同
        assert same_color(line.get_markerfacecolor(), color)  # 断言线的标记填充颜色与指定颜色相同

    def test_separate_colors_direct(self):
        x = y = [1, 2, 3]  # 共享 x 和 y 坐标
        m = Path(color=".22", edgecolor=".55", fillcolor=".77")  # 创建 Path 对象并指定多个颜色属性
        p = Plot(x=x, y=y).add(m).plot()  # 创建 Plot 对象并绘制图形
        line, = p._figure.axes[0].get_lines()  # 获取绘图中的线

        assert same_color(line.get_color(), m.color)  # 断言线的颜色与 Path 对象的颜色属性相同
        assert same_color(line.get_markeredgecolor(), m.edgecolor)  # 断言线的标记边缘颜色与 Path 对象的边缘颜色属性相同
        assert same_color(line.get_markerfacecolor(), m.fillcolor)  # 断言线的标记填充颜色与 Path 对象的填充颜色属性相同

    def test_shared_colors_mapped(self):
        x = y = [1, 2, 3, 4]  # 共享 x 和 y 坐标
        c = ["a", "a", "b", "b"]  # 颜色映射数据
        m = Path()  # 创建 Path 对象
        p = Plot(x=x, y=y, color=c).add(m).plot()  # 创建 Plot 对象并绘制图形
        ax = p._figure.axes[0]  # 获取绘图的轴
        colors = p._theme["axes.prop_cycle"].by_key()["color"]  # 获取颜色循环中的颜色列表
        for i, line in enumerate(ax.get_lines()):
            assert same_color(line.get_color(), colors[i])  # 断言线的颜色与颜色循环中相应位置的颜色相同
            assert same_color(line.get_markeredgecolor(), colors[i])  # 断言线的标记边缘颜色与颜色循环中相应位置的颜色相同
            assert same_color(line.get_markerfacecolor(), colors[i])  # 断言线的标记填充颜色与颜色循环中相应位置的颜色相同

    def test_separate_colors_mapped(self):
        x = y = [1, 2, 3, 4]  # 共享 x 和 y 坐标
        c = ["a", "a", "b", "b"]  # 颜色映射数据
        d = ["x", "y", "x", "y"]  # 填充颜色映射数据
        m = Path()  # 创建 Path 对象
        p = Plot(x=x, y=y, color=c, fillcolor=d).add(m).plot()  # 创建 Plot 对象并绘制图形
        ax = p._figure.axes[0]  # 获取绘图的轴
        colors = p._theme["axes.prop_cycle"].by_key()["color"]  # 获取颜色循环中的颜色列表
        for i, line in enumerate(ax.get_lines()):
            assert same_color(line.get_color(), colors[i // 2])  # 断言线的颜色与颜色循环中相应位置的颜色相同
            assert same_color(line.get_markeredgecolor(), colors[i // 2])  # 断言线的标记边缘颜色与颜色循环中相应位置的颜色相同
            assert same_color(line.get_markerfacecolor(), colors[i % 2])  # 断言线的标记填充颜色与颜色循环中相应位置的颜色相同
    # 测试颜色带透明度的路径绘制

    # 设置 x 和 y 的值为 [1, 2, 3]
    x = y = [1, 2, 3]
    # 创建一个 Path 对象 m，设置颜色为 (.4, .9, .2, .5)，填充颜色为 (.2, .2, .3, .9)
    m = Path(color=(.4, .9, .2, .5), fillcolor=(.2, .2, .3, .9))
    # 创建 Plot 对象 p，并向其添加 Path 对象 m，然后进行绘图
    p = Plot(x=x, y=y).add(m).plot()
    # 从绘图对象 p 中获取第一个轴的线条对象 line
    line, = p._figure.axes[0].get_lines()
    # 断言线条的颜色与 Path 对象 m 的颜色相同
    assert same_color(line.get_color(), m.color)
    # 断言线条的标记边缘颜色与 Path 对象 m 的颜色相同
    assert same_color(line.get_markeredgecolor(), m.color)
    # 断言线条的标记填充颜色与 Path 对象 m 的填充颜色相同
    assert same_color(line.get_markerfacecolor(), m.fillcolor)

    # 测试仅颜色带透明度的路径绘制

    # 设置 x 和 y 的值为 [1, 2, 3]
    x = y = [1, 2, 3]
    # 创建一个 Path 对象 m，设置颜色为 (.4, .9, .2)，填充颜色为 (.2, .2, .3)，透明度为 .5
    m = Path(color=(.4, .9, .2), fillcolor=(.2, .2, .3), alpha=.5)
    # 创建 Plot 对象 p，并向其添加 Path 对象 m，然后进行绘图
    p = Plot(x=x, y=y).add(m).plot()
    # 从绘图对象 p 中获取第一个轴的线条对象 line
    line, = p._figure.axes[0].get_lines()
    # 断言线条的颜色与带透明度处理后的 Path 对象 m 的颜色相同
    assert same_color(line.get_color(), to_rgba(m.color, m.alpha))
    # 断言线条的标记边缘颜色与带透明度处理后的 Path 对象 m 的颜色相同
    assert same_color(line.get_markeredgecolor(), to_rgba(m.color, m.alpha))
    # 断言线条的标记填充颜色与带透明度处理后的 Path 对象 m 的填充颜色相同
    assert same_color(line.get_markerfacecolor(), to_rgba(m.fillcolor, m.alpha))

    # 测试其它属性直接设置的路径绘制

    # 设置 x 和 y 的值为 [1, 2, 3]
    x = y = [1, 2, 3]
    # 创建一个 Path 对象 m，设置标记为 "s"，线条样式为 "--"，线宽为 3，标记大小为 10，标记边缘宽度为 1
    m = Path(marker="s", linestyle="--", linewidth=3, pointsize=10, edgewidth=1)
    # 创建 Plot 对象 p，并向其添加 Path 对象 m，然后进行绘图
    p = Plot(x=x, y=y).add(m).plot()
    # 从绘图对象 p 中获取第一个轴的线条对象 line
    line, = p._figure.axes[0].get_lines()
    # 断言线条的标记与 Path 对象 m 的标记相同
    assert line.get_marker() == m.marker
    # 断言线条的线条样式与 Path 对象 m 的线条样式相同
    assert line.get_linestyle() == m.linestyle
    # 断言线条的线宽与 Path 对象 m 的线宽相同
    assert line.get_linewidth() == m.linewidth
    # 断言线条的标记大小与 Path 对象 m 的标记大小相同
    assert line.get_markersize() == m.pointsize
    # 断言线条的标记边缘宽度与 Path 对象 m 的标记边缘宽度相同
    assert line.get_markeredgewidth() == m.edgewidth

    # 测试其它属性映射设置的路径绘制

    # 设置 x 和 y 的值为 [1, 2, 3, 4]
    x = y = [1, 2, 3, 4]
    # 设置 g 为 ["a", "a", "b", "b"]
    g = ["a", "a", "b", "b"]
    # 创建一个空的 Path 对象 m
    m = Path()
    # 创建 Plot 对象 p，设置 x、y、marker、linestyle、pointsize 为 g，并向其添加 Path 对象 m，然后进行绘图
    p = Plot(x=x, y=y, marker=g, linestyle=g, pointsize=g).add(m).plot()
    # 从绘图对象 p 中获取前两个轴的线条对象 line1 和 line2
    line1, line2 = p._figure.axes[0].get_lines()
    # 断言 line1 和 line2 的标记不相同
    assert line1.get_marker() != line2.get_marker()
    # 由于 Matplotlib 存储虚线样式时的 bug，此处无法断言线条的线条样式不相同
    # assert line1.get_linestyle() != line2.get_linestyle()
    # 断言 line1 和 line2 的标记大小不相同
    assert line1.get_markersize() != line2.get_markersize()

    # 测试线条端点样式设置

    # 设置 x 和 y 的值为 [1, 2]
    x = y = [1, 2]
    # 设置 rc 字典，包含 "lines.solid_capstyle" 和 "lines.dash_capstyle"
    rc = {"lines.solid_capstyle": "projecting", "lines.dash_capstyle": "round"}
    # 创建 Plot 对象 p，设置 x、y，向其添加空的 Path 对象，并应用 rc 主题设置，然后进行绘图
    p = Plot(x, y).add(Path()).theme(rc).plot()
    # 从绘图对象 p 中获取第一个轴的线条对象 line
    line, = p._figure.axes[0].get_lines()
    # 断言线条的虚线端点样式与 rc 中设置的 "lines.solid_capstyle" 相同
    assert line.get_dash_capstyle() == "projecting"

    # 创建 Plot 对象 p，设置 x、y，向其添加 Path 对象设置线条样式为 "--"，并应用 rc 主题设置，然后进行绘图
    p = Plot(x, y).add(Path(linestyle="--")).theme(rc).plot()
    # 从绘图对象 p 中获取第一个轴的线条对象 line
    line, = p._figure.axes[0].get_lines()
    # 断言线条的虚线端点样式与 rc 中设置的 "lines.dash_capstyle" 相同
    assert line.get_dash_capstyle() == "round"

    # 创建 Plot 对象 p，设置 x、y，向其添加 Path 对象设置实线端点样式为 "butt"，并应用 rc 主题设置，然后进行绘图
    p = Plot(x, y).add(Path({"solid_capstyle": "butt"})).theme(rc).plot()
    # 从绘图对象 p 中获取第一个轴的线条对象 line
    line, = p._figure.axes[0].get_lines()
    # 断言线条的实线端点样式与设置的 "butt" 相同
    assert line.get_solid_capstyle() == "butt"
class TestLine:

    # Most behaviors shared with Path and covered by above tests

    # 测试xy数据的方法
    def test_xy_data(self):
        # 定义x、y、g数组
        x = [1, 5, 3, np.nan, 2]
        y = [1, 4, 2, 5, 3]
        g = [1, 2, 1, 1, 2]
        # 创建Plot对象，并添加Line对象，生成绘图
        p = Plot(x=x, y=y, group=g).add(Line()).plot()
        # 获取绘图中的两条线
        line1, line2 = p._figure.axes[0].get_lines()

        # 断言第一条线的x和y数据是否符合预期
        assert_array_equal(line1.get_xdata(), [1, 3])
        assert_array_equal(line1.get_ydata(), [1, 2])
        # 断言第二条线的x和y数据是否符合预期
        assert_array_equal(line2.get_xdata(), [2, 5])
        assert_array_equal(line2.get_ydata(), [3, 4])


class TestPaths:

    # 测试xy数据的方法
    def test_xy_data(self):
        # 定义x、y、g数组
        x = [1, 5, 3, np.nan, 2]
        y = [1, 4, 2, 5, 3]
        g = [1, 2, 1, 1, 2]
        # 创建Plot对象，并添加Paths对象，生成绘图
        p = Plot(x=x, y=y, group=g).add(Paths()).plot()
        # 获取绘图中的线集合
        lines, = p._figure.axes[0].collections

        # 获取第一个路径对象的顶点并断言其数据是否符合预期
        verts = lines.get_paths()[0].vertices.T
        assert_array_equal(verts[0], [1, 3, np.nan])
        assert_array_equal(verts[1], [1, 2, np.nan])

        # 获取第二个路径对象的顶点并断言其数据是否符合预期
        verts = lines.get_paths()[1].vertices.T
        assert_array_equal(verts[0], [5, 2])
        assert_array_equal(verts[1], [4, 3])

    # 测试设置属性的方法
    def test_set_properties(self):
        # 定义x、y数组
        x = y = [1, 2, 3]
        # 创建Paths对象m，设置颜色、线宽和线型
        m = Paths(color=".737", linewidth=1, linestyle=(3, 1))
        # 创建Plot对象，并添加Paths对象m，生成绘图
        p = Plot(x=x, y=y).add(m).plot()
        # 获取绘图中的线集合
        lines, = p._figure.axes[0].collections

        # 断言获取的线的颜色与m的颜色是否相同
        assert same_color(lines.get_color().squeeze(), m.color)
        # 断言获取的线的线宽与m的线宽是否相同
        assert lines.get_linewidth().item() == m.linewidth
        # 断言获取的线的线型与m的线型是否相同
        assert lines.get_dashes()[0] == (0, list(m.linestyle))

    # 测试映射属性的方法
    def test_mapped_properties(self):
        # 定义x、y、g数组
        x = y = [1, 2, 3, 4]
        g = ["a", "a", "b", "b"]
        # 创建Plot对象，并添加映射颜色、线宽和线型的Paths对象，生成绘图
        p = Plot(x=x, y=y, color=g, linewidth=g, linestyle=g).add(Paths()).plot()
        # 获取绘图中的线集合
        lines, = p._figure.axes[0].collections

        # 断言第一条线和第二条线的颜色不相同
        assert not np.array_equal(lines.get_colors()[0], lines.get_colors()[1])
        # 断言第一条线和第二条线的线宽不相同
        assert lines.get_linewidths()[0] != lines.get_linewidth()[1]
        # 断言第一条线和第二条线的线型不相同
        assert lines.get_linestyle()[0] != lines.get_linestyle()[1]

    # 测试带alpha通道的颜色方法
    def test_color_with_alpha(self):
        # 定义x、y数组
        x = y = [1, 2, 3]
        # 创建带有alpha通道颜色的Paths对象m
        m = Paths(color=(.2, .6, .9, .5))
        # 创建Plot对象，并添加Paths对象m，生成绘图
        p = Plot(x=x, y=y).add(m).plot()
        # 获取绘图中的线集合
        lines, = p._figure.axes[0].collections
        # 断言获取的线的颜色与m的颜色是否相同
        assert same_color(lines.get_colors().squeeze(), m.color)

    # 测试带颜色和alpha通道的方法
    def test_color_and_alpha(self):
        # 定义x、y数组
        x = y = [1, 2, 3]
        # 创建带颜色和alpha通道的Paths对象m
        m = Paths(color=(.2, .6, .9), alpha=.5)
        # 创建Plot对象，并添加Paths对象m，生成绘图
        p = Plot(x=x, y=y).add(m).plot()
        # 获取绘图中的线集合
        lines, = p._figure.axes[0].collections
        # 断言获取的线的颜色与m的颜色是否相同
        assert same_color(lines.get_colors().squeeze(), to_rgba(m.color, m.alpha))
    # 定义一个测试方法，用于测试线段端点样式的设置

    x = y = [1, 2]  # 定义 x 和 y 坐标，均为 [1, 2]
    rc = {"lines.solid_capstyle": "projecting"}  # 定义一个字典 rc，设置线段端点样式为 projecting

    with mpl.rc_context(rc):  # 使用 matplotlib 的 rc_context 上下文管理器，设置图形的绘制样式
        # 创建一个 Plot 对象，添加 Paths 对象，并绘制图形
        p = Plot(x, y).add(Paths()).plot()
        # 获取绘制的图形对象中的第一个轴的第一个集合（collections），即路径集合
        lines = p._figure.axes[0].collections[0]
        # 断言绘制的线段的端点样式是否为 "projecting"
        assert lines.get_capstyle() == "projecting"

        # 创建一个 Plot 对象，添加 Paths 对象并设置线段样式为 "--"，然后绘制图形
        p = Plot(x, y).add(Paths(linestyle="--")).plot()
        # 获取绘制的图形对象中的第一个轴的第一个集合（collections），即路径集合
        lines = p._figure.axes[0].collections[0]
        # 断言绘制的线段的端点样式是否为 "projecting"
        assert lines.get_capstyle() == "projecting"

        # 创建一个 Plot 对象，添加 Paths 对象并设置端点样式为 "butt"，然后绘制图形
        p = Plot(x, y).add(Paths({"capstyle": "butt"})).plot()
        # 获取绘制的图形对象中的第一个轴的第一个集合（collections），即路径集合
        lines = p._figure.axes[0].collections[0]
        # 断言绘制的线段的端点样式是否为 "butt"
        assert lines.get_capstyle() == "butt"
class TestLines:

    def test_xy_data(self):
        # 定义 x, y, g 数据
        x = [1, 5, 3, np.nan, 2]
        y = [1, 4, 2, 5, 3]
        g = [1, 2, 1, 1, 2]
        # 创建 Plot 对象并绘制 Lines
        p = Plot(x=x, y=y, group=g).add(Lines()).plot()
        # 获取绘图中的 Lines 对象
        lines, = p._figure.axes[0].collections

        # 获取第一条路径的顶点坐标并断言
        verts = lines.get_paths()[0].vertices.T
        assert_array_equal(verts[0], [1, 3])
        assert_array_equal(verts[1], [1, 2])

        # 获取第二条路径的顶点坐标并断言
        verts = lines.get_paths()[1].vertices.T
        assert_array_equal(verts[0], [2, 5])
        assert_array_equal(verts[1], [3, 4])

    def test_single_orient_value(self):
        # 定义 x, y 数据
        x = [1, 1, 1]
        y = [1, 2, 3]
        # 创建 Plot 对象并绘制 Lines
        p = Plot(x, y).add(Lines()).plot()
        # 获取绘图中的 Lines 对象
        lines, = p._figure.axes[0].collections
        # 获取路径的顶点坐标并断言
        verts = lines.get_paths()[0].vertices.T
        assert_array_equal(verts[0], x)
        assert_array_equal(verts[1], y)


class TestRange:

    def test_xy_data(self):
        # 定义 x, ymin, ymax 数据
        x = [1, 2]
        ymin = [1, 4]
        ymax = [2, 3]
        # 创建 Plot 对象并绘制 Range
        p = Plot(x=x, ymin=ymin, ymax=ymax).add(Range()).plot()
        # 获取绘图中的 Range 对象
        lines, = p._figure.axes[0].collections

        # 遍历每条路径并断言路径顶点坐标
        for i, path in enumerate(lines.get_paths()):
            verts = path.vertices.T
            assert_array_equal(verts[0], [x[i], x[i]])
            assert_array_equal(verts[1], [ymin[i], ymax[i]])

    def test_auto_range(self):
        # 定义 x, y 数据
        x = [1, 1, 2, 2, 2]
        y = [1, 2, 3, 4, 5]
        # 创建 Plot 对象并绘制 Range
        p = Plot(x=x, y=y).add(Range()).plot()
        # 获取绘图中的 Range 对象
        lines, = p._figure.axes[0].collections
        paths = lines.get_paths()
        # 断言路径顶点坐标
        assert_array_equal(paths[0].vertices, [(1, 1), (1, 2)])
        assert_array_equal(paths[1].vertices, [(2, 3), (2, 5)])

    def test_mapped_color(self):
        # 定义 x, ymin, ymax, group 数据
        x = [1, 2, 1, 2]
        ymin = [1, 4, 3, 2]
        ymax = [2, 3, 1, 4]
        group = ["a", "a", "b", "b"]
        # 创建 Plot 对象并绘制带有颜色映射的 Range
        p = Plot(x=x, ymin=ymin, ymax=ymax, color=group).add(Range()).plot()
        # 获取绘图中的 Range 对象
        lines, = p._figure.axes[0].collections
        # 获取颜色列表
        colors = p._theme["axes.prop_cycle"].by_key()["color"]

        # 遍历每条路径并断言路径顶点坐标及颜色
        for i, path in enumerate(lines.get_paths()):
            verts = path.vertices.T
            assert_array_equal(verts[0], [x[i], x[i]])
            assert_array_equal(verts[1], [ymin[i], ymax[i]])
            assert same_color(lines.get_colors()[i], colors[i // 2])

    def test_direct_properties(self):
        # 定义 x, ymin, ymax 数据
        x = [1, 2]
        ymin = [1, 4]
        ymax = [2, 3]
        # 创建具有直接属性的 Range 对象
        m = Range(color=".654", linewidth=4)
        # 创建 Plot 对象并绘制 Range
        p = Plot(x=x, ymin=ymin, ymax=ymax).add(m).plot()
        # 获取绘图中的 Range 对象
        lines, = p._figure.axes[0].collections

        # 遍历每条路径并断言颜色和线宽
        for i, path in enumerate(lines.get_paths()):
            assert same_color(lines.get_colors()[i], m.color)
            assert lines.get_linewidths()[i] == m.linewidth
    def test_xy_data(self):
        # 定义测试用例数据 x 和 y
        x = [0, 0, 1, 2]
        y = [1, 2, 3, 4]

        # 创建绘图对象 p，绘制散点图并添加虚线风格
        p = Plot(x=x, y=y).add(Dash()).plot()
        # 获取绘图对象中的线集合
        lines, = p._figure.axes[0].collections

        # 遍历每条线的路径
        for i, path in enumerate(lines.get_paths()):
            # 提取路径的顶点信息
            verts = path.vertices.T
            # 断言顶点 x 坐标的准确性
            assert_array_almost_equal(verts[0], [x[i] - .4, x[i] + .4])
            # 断言顶点 y 坐标的准确性
            assert_array_equal(verts[1], [y[i], y[i]])

    def test_xy_data_grouped(self):
        # 定义测试用例数据 x, y 和 color
        x = [0, 0, 1, 2]
        y = [1, 2, 3, 4]
        color = ["a", "b", "a", "b"]

        # 创建绘图对象 p，绘制分组散点图并添加虚线风格
        p = Plot(x=x, y=y, color=color).add(Dash()).plot()
        # 获取绘图对象中的线集合
        lines, = p._figure.axes[0].collections

        # 定义顺序索引
        idx = [0, 2, 1, 3]
        # 遍历每个索引对应的路径
        for i, path in zip(idx, lines.get_paths()):
            # 提取路径的顶点信息
            verts = path.vertices.T
            # 断言顶点 x 坐标的准确性
            assert_array_almost_equal(verts[0], [x[i] - .4, x[i] + .4])
            # 断言顶点 y 坐标的准确性
            assert_array_equal(verts[1], [y[i], y[i]])

    def test_set_properties(self):
        # 定义测试用例数据 x 和 y
        x = [0, 0, 1, 2]
        y = [1, 2, 3, 4]

        # 创建虚线风格对象 m，设置颜色和线宽
        m = Dash(color=".8", linewidth=4)
        # 创建绘图对象 p，绘制散点图并添加虚线风格
        p = Plot(x=x, y=y).add(m).plot()
        # 获取绘图对象中的线集合
        lines, = p._figure.axes[0].collections

        # 遍历每条线的颜色，断言其与风格对象 m 的颜色相同
        for color in lines.get_color():
            assert same_color(color, m.color)
        # 遍历每条线的线宽，断言其与风格对象 m 的线宽相同
        for linewidth in lines.get_linewidth():
            assert linewidth == m.linewidth

    def test_mapped_properties(self):
        # 定义测试用例数据 x, y, color 和 linewidth
        x = [0, 1]
        y = [1, 2]
        color = ["a", "b"]
        linewidth = [1, 2]

        # 创建绘图对象 p，绘制映射属性的散点图并添加虚线风格
        p = Plot(x=x, y=y, color=color, linewidth=linewidth).add(Dash()).plot()
        # 获取绘图对象中的线集合
        lines, = p._figure.axes[0].collections
        # 获取调色板颜色
        palette = p._theme["axes.prop_cycle"].by_key()["color"]

        # 遍历调色板颜色和线集合中的颜色，断言它们相同
        for color, line_color in zip(palette, lines.get_color()):
            assert same_color(color, line_color)

        # 断言第二条线的线宽大于第一条线的线宽
        linewidths = lines.get_linewidths()
        assert linewidths[1] > linewidths[0]

    def test_width(self):
        # 定义测试用例数据 x 和 y
        x = [0, 0, 1, 2]
        y = [1, 2, 3, 4]

        # 创建虚线风格对象，设置线宽为 0.4
        p = Plot(x=x, y=y).add(Dash(width=.4)).plot()
        # 获取绘图对象中的线集合
        lines, = p._figure.axes[0].collections

        # 遍历每条线的路径
        for i, path in enumerate(lines.get_paths()):
            # 提取路径的顶点信息
            verts = path.vertices.T
            # 断言顶点 x 坐标的准确性
            assert_array_almost_equal(verts[0], [x[i] - .2, x[i] + .2])
            # 断言顶点 y 坐标的准确性
            assert_array_equal(verts[1], [y[i], y[i]])

    def test_dodge(self):
        # 定义测试用例数据 x, y 和 group
        x = [0, 1]
        y = [1, 2]
        group = ["a", "b"]

        # 创建绘图对象 p，绘制分组散点图并添加虚线风格和分组效果
        p = Plot(x=x, y=y, group=group).add(Dash(), Dodge()).plot()
        # 获取绘图对象中的线集合
        lines, = p._figure.axes[0].collections

        # 获取路径集合
        paths = lines.get_paths()

        # 断言第一条路径的顶点信息
        v0 = paths[0].vertices.T
        assert_array_almost_equal(v0[0], [-.4, 0])
        assert_array_equal(v0[1], [y[0], y[0]])

        # 断言第二条路径的顶点信息
        v1 = paths[1].vertices.T
        assert_array_almost_equal(v1[0], [1, 1.4])
        assert_array_equal(v1[1], [y[1], y[1]])
```