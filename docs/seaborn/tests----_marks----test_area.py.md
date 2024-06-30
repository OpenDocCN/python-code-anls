# `D:\src\scipysrc\seaborn\tests\_marks\test_area.py`

```
import matplotlib as mpl  # 导入matplotlib库并使用别名mpl
from matplotlib.colors import to_rgba, to_rgba_array  # 从matplotlib库中导入to_rgba和to_rgba_array函数
from numpy.testing import assert_array_equal  # 从numpy.testing模块中导入assert_array_equal函数
from seaborn._core.plot import Plot  # 从seaborn._core.plot模块中导入Plot类
from seaborn._marks.area import Area, Band  # 从seaborn._marks.area模块中导入Area和Band类


class TestArea:  # 定义测试类TestArea

    def test_single_defaults(self):  # 定义测试方法test_single_defaults
        x, y = [1, 2, 3], [1, 2, 1]  # 设置x和y坐标数组
        p = Plot(x=x, y=y).add(Area()).plot()  # 创建Plot对象并添加Area标记，然后绘制图形
        ax = p._figure.axes[0]  # 获取图形对象的第一个坐标轴
        poly = ax.patches[0]  # 获取坐标轴中的第一个多边形补丁对象
        verts = poly.get_path().vertices.T  # 获取多边形的顶点坐标并转置
        colors = p._theme["axes.prop_cycle"].by_key()["color"]  # 获取图形的颜色循环中的颜色列表

        expected_x = [1, 2, 3, 3, 2, 1, 1]  # 预期的多边形顶点x坐标
        assert_array_equal(verts[0], expected_x)  # 断言多边形顶点x坐标与预期值相等

        expected_y = [0, 0, 0, 1, 2, 1, 0]  # 预期的多边形顶点y坐标
        assert_array_equal(verts[1], expected_y)  # 断言多边形顶点y坐标与预期值相等

        fc = poly.get_facecolor()  # 获取多边形的填充颜色
        assert_array_equal(fc, to_rgba(colors[0], .2))  # 断言填充颜色与预期的颜色和透明度相等

        ec = poly.get_edgecolor()  # 获取多边形的边缘颜色
        assert_array_equal(ec, to_rgba(colors[0], 1))  # 断言边缘颜色与预期的颜色和透明度相等

        lw = poly.get_linewidth()  # 获取多边形的边缘线宽度
        assert_array_equal(lw, mpl.rcParams["patch.linewidth"] * 2)  # 断言边缘线宽度与默认配置的值的两倍相等

    def test_set_properties(self):  # 定义测试方法test_set_properties
        x, y = [1, 2, 3], [1, 2, 1]  # 设置x和y坐标数组
        mark = Area(  # 创建Area对象mark，设置其属性
            color=".33",  # 填充颜色
            alpha=.3,  # 填充透明度
            edgecolor=".88",  # 边缘颜色
            edgealpha=.8,  # 边缘透明度
            edgewidth=2,  # 边缘线宽度
            edgestyle=(0, (2, 1)),  # 边缘线样式
        )
        p = Plot(x=x, y=y).add(mark).plot()  # 创建Plot对象并添加mark标记，然后绘制图形
        ax = p._figure.axes[0]  # 获取图形对象的第一个坐标轴
        poly = ax.patches[0]  # 获取坐标轴中的第一个多边形补丁对象

        fc = poly.get_facecolor()  # 获取多边形的填充颜色
        assert_array_equal(fc, to_rgba(mark.color, mark.alpha))  # 断言填充颜色与mark对象设置的颜色和透明度相等

        ec = poly.get_edgecolor()  # 获取多边形的边缘颜色
        assert_array_equal(ec, to_rgba(mark.edgecolor, mark.edgealpha))  # 断言边缘颜色与mark对象设置的颜色和透明度相等

        lw = poly.get_linewidth()  # 获取多边形的边缘线宽度
        assert_array_equal(lw, mark.edgewidth * 2)  # 断言边缘线宽度与mark对象设置的宽度的两倍相等

        ls = poly.get_linestyle()  # 获取多边形的边缘线样式
        dash_on, dash_off = mark.edgestyle[1]  # 解构边缘线样式元组的第二个元素
        expected = (0, (mark.edgewidth * dash_on / 4, mark.edgewidth * dash_off / 4))  # 计算预期的边缘线样式
        assert ls == expected  # 断言边缘线样式与预期值相等

    def test_mapped_properties(self):  # 定义测试方法test_mapped_properties
        x, y = [1, 2, 3, 2, 3, 4], [1, 2, 1, 1, 3, 2]  # 设置x和y坐标数组
        g = ["a", "a", "a", "b", "b", "b"]  # 分组标识符数组
        cs = [".2", ".8"]  # 颜色标识符数组
        p = Plot(x=x, y=y, color=g, edgewidth=g).scale(color=cs).add(Area()).plot()  # 创建Plot对象并添加Area标记，然后绘制图形
        ax = p._figure.axes[0]  # 获取图形对象的第一个坐标轴

        expected_x = [1, 2, 3, 3, 2, 1, 1], [2, 3, 4, 4, 3, 2, 2]  # 预期的多边形顶点x坐标数组
        expected_y = [0, 0, 0, 1, 2, 1, 0], [0, 0, 0, 2, 3, 1, 0]  # 预期的多边形顶点y坐标数组

        for i, poly in enumerate(ax.patches):  # 遍历坐标轴中的所有多边形补丁对象
            verts = poly.get_path().vertices.T  # 获取多边形的顶点坐标并转置
            assert_array_equal(verts[0], expected_x[i])  # 断言多边形顶点x坐标与预期值相等
            assert_array_equal(verts[1], expected_y[i])  # 断言多边形顶点y坐标与预期值相等

        fcs = [p.get_facecolor() for p in ax.patches]  # 获取所有多边形的填充颜色
        assert_array_equal(fcs, to_rgba_array(cs, .2))  # 断言填充颜色与标识符数组和透明度的预期值相等

        ecs = [p.get_edgecolor() for p in ax.patches]  # 获取所有多边形的边缘颜色
        assert_array_equal(ecs, to_rgba_array(cs, 1))  # 断言边缘颜色与标识符数组和透明度的预期值相等

        lws = [p.get_linewidth() for p in ax.patches]  # 获取所有多边形的边缘线宽度
        assert lws[0] > lws[1]  # 断言第一个多边形的边缘线宽度大于第二个多边形的边缘线宽度
    # 定义一个测试方法 test_unfilled，用于测试某些绘图功能未填充区域的情况

        # 定义两个列表 x 和 y，分别为 x 轴和 y 轴的数据点
        x, y = [1, 2, 3], [1, 2, 1]
        # 定义颜色变量 c，表示区域的边框颜色
        c = ".5"
        # 创建一个 Plot 对象 p，传入 x 和 y 数据，并添加一个不填充的 Area 对象，颜色为 c
        p = Plot(x=x, y=y).add(Area(fill=False, color=c)).plot()
        # 获取 p 对象的第一个坐标轴
        ax = p._figure.axes[0]
        # 获取坐标轴上的第一个图形对象（patch）
        poly = ax.patches[0]
        # 断言获取的图形对象的填充颜色与指定的颜色 c 以及透明度 0 相匹配
        assert poly.get_facecolor() == to_rgba(c, 0)
class TestBand:

    def test_range(self):
        # 定义测试数据
        x, ymin, ymax = [1, 2, 4], [2, 1, 4], [3, 3, 5]
        # 创建 Plot 对象，并添加 Band 对象，生成绘图
        p = Plot(x=x, ymin=ymin, ymax=ymax).add(Band()).plot()
        # 获取绘图对象的第一个坐标轴
        ax = p._figure.axes[0]
        # 获取第一个图形补丁的顶点坐标
        verts = ax.patches[0].get_path().vertices.T

        # 预期的 x 坐标
        expected_x = [1, 2, 4, 4, 2, 1, 1]
        # 断言顶点 x 坐标是否与预期一致
        assert_array_equal(verts[0], expected_x)

        # 预期的 y 坐标
        expected_y = [2, 1, 4, 5, 3, 3, 2]
        # 断言顶点 y 坐标是否与预期一致
        assert_array_equal(verts[1], expected_y)

    def test_auto_range(self):
        # 定义测试数据
        x = [1, 1, 2, 2, 2]
        y = [1, 2, 3, 4, 5]
        # 创建 Plot 对象，并添加 Band 对象，生成绘图
        p = Plot(x=x, y=y).add(Band()).plot()
        # 获取绘图对象的第一个坐标轴
        ax = p._figure.axes[0]
        # 获取第一个图形补丁的顶点坐标
        verts = ax.patches[0].get_path().vertices.T

        # 预期的 x 坐标
        expected_x = [1, 2, 2, 1, 1]
        # 断言顶点 x 坐标是否与预期一致
        assert_array_equal(verts[0], expected_x)

        # 预期的 y 坐标
        expected_y = [1, 3, 5, 2, 1]
        # 断言顶点 y 坐标是否与预期一致
        assert_array_equal(verts[1], expected_y)
```