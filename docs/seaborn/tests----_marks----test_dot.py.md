# `D:\src\scipysrc\seaborn\tests\_marks\test_dot.py`

```
from matplotlib.colors import to_rgba, to_rgba_array  # 导入必要的函数来处理颜色转换
import pytest  # 导入pytest模块，用于单元测试
from numpy.testing import assert_array_equal  # 导入NumPy的数组比较函数

from seaborn.palettes import color_palette  # 从seaborn中导入颜色调色板
from seaborn._core.plot import Plot  # 导入绘图相关的Plot类
from seaborn._marks.dot import Dot, Dots  # 导入点标记相关的类

@pytest.fixture(autouse=True)
def default_palette():
    with color_palette("deep"):  # 使用深色调色板作为默认的配色方案
        yield  # 执行测试之前的设置和清理工作

class DotBase:

    def check_offsets(self, points, x, y):
        # 检查点的偏移量，确保与给定的x和y坐标数组一致
        offsets = points.get_offsets().T
        assert_array_equal(offsets[0], x)
        assert_array_equal(offsets[1], y)

    def check_colors(self, part, points, colors, alpha=None):
        # 检查点的颜色，支持透明度设置
        rgba = to_rgba_array(colors, alpha)
        
        getter = getattr(points, f"get_{part}colors")
        assert_array_equal(getter(), rgba)

class TestDot(DotBase):

    def test_simple(self):
        # 简单的测试用例，绘制包含点标记的图表
        x = [1, 2, 3]
        y = [4, 5, 2]
        p = Plot(x=x, y=y).add(Dot()).plot()  # 绘制包含点标记的图表
        ax = p._figure.axes[0]
        points, = ax.collections
        C0, *_ = p._theme["axes.prop_cycle"].by_key()["color"]
        self.check_offsets(points, x, y)  # 检查点的偏移量是否正确
        self.check_colors("face", points, [C0] * 3, 1)  # 检查点的填充颜色是否正确
        self.check_colors("edge", points, [C0] * 3, 1)  # 检查点的边界颜色是否正确

    def test_filled_unfilled_mix(self):
        # 测试包含填充和未填充点标记的混合图表
        x = [1, 2]
        y = [4, 5]
        marker = ["a", "b"]
        shapes = ["o", "x"]

        mark = Dot(edgecolor="w", stroke=2, edgewidth=1)  # 创建自定义的点标记
        p = Plot(x=x, y=y).add(mark, marker=marker).scale(marker=shapes).plot()  # 绘制带有自定义标记的图表
        ax = p._figure.axes[0]
        points, = ax.collections
        C0, *_ = p._theme["axes.prop_cycle"].by_key()["color"]
        self.check_offsets(points, x, y)  # 检查点的偏移量是否正确
        self.check_colors("face", points, [C0, to_rgba(C0, 0)], None)  # 检查填充颜色是否正确
        self.check_colors("edge", points, ["w", C0], 1)  # 检查边界颜色是否正确

        expected = [mark.edgewidth, mark.stroke]
        assert_array_equal(points.get_linewidths(), expected)  # 检查边界宽度是否正确

    def test_missing_coordinate_data(self):
        # 测试缺失坐标数据时的行为
        x = [1, float("nan"), 3]
        y = [5, 3, 4]

        p = Plot(x=x, y=y).add(Dot()).plot()  # 绘制包含点标记的图表
        ax = p._figure.axes[0]
        points, = ax.collections
        self.check_offsets(points, [1, 3], [5, 4])  # 检查处理缺失数据后的点的偏移量

    @pytest.mark.parametrize("prop", ["color", "fill", "marker", "pointsize"])
    def test_missing_semantic_data(self, prop):
        # 测试缺失语义数据时的行为
        x = [1, 2, 3]
        y = [5, 3, 4]
        z = ["a", float("nan"), "b"]

        p = Plot(x=x, y=y, **{prop: z}).add(Dot()).plot()  # 绘制包含点标记的图表
        ax = p._figure.axes[0]
        points, = ax.collections
        self.check_offsets(points, [1, 3], [5, 4])  # 检查处理缺失数据后的点的偏移量

class TestDots(DotBase):

    def test_simple(self):
        # 简单的测试用例，绘制包含多个点标记的图表
        x = [1, 2, 3]
        y = [4, 5, 2]
        p = Plot(x=x, y=y).add(Dots()).plot()  # 绘制包含多个点标记的图表
        ax = p._figure.axes[0]
        points, = ax.collections
        C0, *_ = p._theme["axes.prop_cycle"].by_key()["color"]
        self.check_offsets(points, x, y)  # 检查点的偏移量是否正确
        self.check_colors("face", points, [C0] * 3, .2)  # 检查填充颜色是否正确，带有指定的透明度
        self.check_colors("edge", points, [C0] * 3, 1)  # 检查边界颜色是否正确
    # 测试设置颜色功能
    def test_set_color(self):
        # 定义 x 和 y 坐标
        x = [1, 2, 3]
        y = [4, 5, 2]
        # 创建一个 Dots 对象 m，并设置颜色为 ".25"
        m = Dots(color=".25")
        # 创建一个 Plot 对象 p，传入 x 和 y 数据，并添加 m 到图中，然后绘制图形
        p = Plot(x=x, y=y).add(m).plot()
        # 获取绘图对象的第一个坐标轴
        ax = p._figure.axes[0]
        # 获取坐标轴上的点集合
        points, = ax.collections
        # 检查偏移量是否正确
        self.check_offsets(points, x, y)
        # 检查点的颜色是否正确设置为 m.color
        self.check_colors("face", points, [m.color] * 3, .2)
        # 检查点的边框颜色是否正确设置为 m.color
        self.check_colors("edge", points, [m.color] * 3, 1)

    # 测试映射颜色功能
    def test_map_color(self):
        # 定义 x、y 和 c（颜色）坐标数据
        x = [1, 2, 3]
        y = [4, 5, 2]
        c = ["a", "b", "a"]
        # 创建一个 Plot 对象 p，传入 x、y 和 c 数据，并添加 Dots 对象到图中，然后绘制图形
        p = Plot(x=x, y=y, color=c).add(Dots()).plot()
        # 获取绘图对象的第一个坐标轴
        ax = p._figure.axes[0]
        # 获取坐标轴上的点集合
        points, = ax.collections
        # 从图形主题中获取颜色循环
        C0, C1, *_ = p._theme["axes.prop_cycle"].by_key()["color"]
        # 检查偏移量是否正确
        self.check_offsets(points, x, y)
        # 检查点的面颜色是否正确映射为 C0、C1、C0
        self.check_colors("face", points, [C0, C1, C0], .2)
        # 检查点的边框颜色是否正确映射为 C0、C1、C0
        self.check_colors("edge", points, [C0, C1, C0], 1)

    # 测试不填充功能
    def test_fill(self):
        # 定义 x、y 和 c（颜色）坐标数据
        x = [1, 2, 3]
        y = [4, 5, 2]
        c = ["a", "b", "a"]
        # 创建一个 Plot 对象 p，传入 x、y 和 c 数据，并添加不填充的 Dots 对象到图中，然后绘制图形
        p = Plot(x=x, y=y, color=c).add(Dots(fill=False)).plot()
        # 获取绘图对象的第一个坐标轴
        ax = p._figure.axes[0]
        # 获取坐标轴上的点集合
        points, = ax.collections
        # 从图形主题中获取颜色循环
        C0, C1, *_ = p._theme["axes.prop_cycle"].by_key()["color"]
        # 检查偏移量是否正确
        self.check_offsets(points, x, y)
        # 检查点的面颜色是否正确映射为 C0、C1、C0（不填充）
        self.check_colors("face", points, [C0, C1, C0], 0)
        # 检查点的边框颜色是否正确映射为 C0、C1、C0
        self.check_colors("edge", points, [C0, C1, C0], 1)

    # 测试点大小功能
    def test_pointsize(self):
        # 定义 x、y 和 s（点大小）坐标数据
        x = [1, 2, 3]
        y = [4, 5, 2]
        s = 3
        # 创建一个 Plot 对象 p，传入 x、y 数据，并添加指定点大小的 Dots 对象到图中，然后绘制图形
        p = Plot(x=x, y=y).add(Dots(pointsize=s)).plot()
        # 获取绘图对象的第一个坐标轴
        ax = p._figure.axes[0]
        # 获取坐标轴上的点集合
        points, = ax.collections
        # 检查偏移量是否正确
        self.check_offsets(points, x, y)
        # 断言点的大小是否正确设置为 s 的平方
        assert_array_equal(points.get_sizes(), [s ** 2] * 3)

    # 测试边框宽度功能
    def test_stroke(self):
        # 定义 x、y 和 s（边框宽度）坐标数据
        x = [1, 2, 3]
        y = [4, 5, 2]
        s = 3
        # 创建一个 Plot 对象 p，传入 x、y 数据，并添加指定边框宽度的 Dots 对象到图中，然后绘制图形
        p = Plot(x=x, y=y).add(Dots(stroke=s)).plot()
        # 获取绘图对象的第一个坐标轴
        ax = p._figure.axes[0]
        # 获取坐标轴上的点集合
        points, = ax.collections
        # 检查偏移量是否正确
        self.check_offsets(points, x, y)
        # 断言点的边框宽度是否正确设置为 s
        assert_array_equal(points.get_linewidths(), [s] * 3)

    # 测试填充和不填充混合功能
    def test_filled_unfilled_mix(self):
        # 定义 x、y、marker 和 shapes 数据
        x = [1, 2]
        y = [4, 5]
        marker = ["a", "b"]
        shapes = ["o", "x"]

        # 创建一个 Dots 对象 mark，并设置边框宽度为 2
        mark = Dots(stroke=2)
        # 创建一个 Plot 对象 p，传入 x 和 y 数据，并添加不同标记和形状的 Dots 对象到图中，然后绘制图形
        p = Plot(x=x, y=y).add(mark, marker=marker).scale(marker=shapes).plot()
        # 获取绘图对象的第一个坐标轴
        ax = p._figure.axes[0]
        # 获取坐标轴上的点集合
        points, = ax.collections
        # 从图形主题中获取颜色循环
        C0, C1, *_ = p._theme["axes.prop_cycle"].by_key()["color"]
        # 检查偏移量是否正确
        self.check_offsets(points, x, y)
        # 检查点的面颜色是否正确设置为 C0、透明度 0.2 和 C0、透明度 0（不填充）
        self.check_colors("face", points, [to_rgba(C0, .2), to_rgba(C0, 0)], None)
        # 检查点的边框颜色是否正确设置为 C0、C0
        self.check_colors("edge", points, [C0, C0], 1)
        # 断言点的边框宽度是否正确设置为 mark.stroke
        assert_array_equal(points.get_linewidths(), [mark.stroke] * 2)
```