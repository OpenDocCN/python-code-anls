# `D:\src\scipysrc\pandas\pandas\tests\plotting\frame\test_frame_color.py`

```
"""Test cases for DataFrame.plot"""

import re  # 导入正则表达式模块

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest库

import pandas as pd  # 导入Pandas库
from pandas import DataFrame  # 从Pandas库中导入DataFrame类
import pandas._testing as tm  # 导入Pandas测试工具模块
from pandas.tests.plotting.common import (  # 导入绘图测试的共享模块中的函数和类
    _check_colors,
    _check_plot_works,
    _unpack_cycler,
)
from pandas.util.version import Version  # 导入版本号管理模块

mpl = pytest.importorskip("matplotlib")  # 导入并检查matplotlib库是否可用
plt = pytest.importorskip("matplotlib.pyplot")  # 导入并检查matplotlib.pyplot库是否可用
cm = pytest.importorskip("matplotlib.cm")  # 导入并检查matplotlib.cm库是否可用


def _check_colors_box(bp, box_c, whiskers_c, medians_c, caps_c="k", fliers_c=None):
    # 检查箱线图各部分的颜色
    if fliers_c is None:
        fliers_c = "k"
    _check_colors(bp["boxes"], linecolors=[box_c] * len(bp["boxes"]))
    _check_colors(bp["whiskers"], linecolors=[whiskers_c] * len(bp["whiskers"]))
    _check_colors(bp["medians"], linecolors=[medians_c] * len(bp["medians"]))
    _check_colors(bp["fliers"], linecolors=[fliers_c] * len(bp["fliers"]))
    _check_colors(bp["caps"], linecolors=[caps_c] * len(bp["caps"]))


class TestDataFrameColor:
    @pytest.mark.parametrize("color", list(range(10)))
    def test_mpl2_color_cycle_str(self, color):
        # GH 15516
        color = f"C{color}"  # 构造颜色字符串，例如"C0", "C1", ...
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=["a", "b", "c"]
        )
        _check_plot_works(df.plot, color=color)  # 检查DataFrame绘图函数对指定颜色的工作情况

    def test_color_single_series_list(self):
        # GH 3486
        df = DataFrame({"A": [1, 2, 3]})
        _check_plot_works(df.plot, color=["red"])  # 检查DataFrame绘图函数对单一颜色列表的工作情况

    @pytest.mark.parametrize("color", [(1, 0, 0), (1, 0, 0, 0.5)])
    def test_rgb_tuple_color(self, color):
        # GH 16695
        df = DataFrame({"x": [1, 2], "y": [3, 4]})
        _check_plot_works(df.plot, x="x", y="y", color=color)  # 检查DataFrame绘图函数对RGB元组颜色的工作情况

    def test_color_empty_string(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        with pytest.raises(ValueError, match="Invalid color argument:"):
            df.plot(color="")  # 检查DataFrame绘图函数对空字符串颜色参数的异常处理

    def test_color_and_style_arguments(self):
        df = DataFrame({"x": [1, 2], "y": [3, 4]})
        # 同时传递'color'和'style'参数应该是允许的
        # 只要样式字符串中没有颜色符号即可:
        ax = df.plot(color=["red", "black"], style=["-", "--"])
        # 检查线条样式是否设置正确:
        linestyle = [line.get_linestyle() for line in ax.lines]
        assert linestyle == ["-", "--"]
        # 检查颜色是否设置正确:
        color = [line.get_color() for line in ax.lines]
        assert color == ["red", "black"]
        # 如果样式字符串中有颜色符号，则不应同时传递'color'和'style'参数
        msg = (
            "Cannot pass 'style' string with a color symbol and 'color' keyword "
            "argument. Please use one or the other or pass 'style' without a color "
            "symbol"
        )
        with pytest.raises(ValueError, match=msg):
            df.plot(color=["red", "black"], style=["k-", "r--"])
    @pytest.mark.parametrize(
        "color, expected",
        [
            ("green", ["green"] * 4),  # 参数化测试颜色为"green"时，期望结果是四个"green"
            (["yellow", "red", "green", "blue"], ["yellow", "red", "green", "blue"]),  # 参数化测试多种颜色，期望结果是对应的颜色列表
        ],
    )
    def test_color_and_marker(self, color, expected):
        # GH 21003
        # 创建一个7行4列的随机数据的DataFrame对象
        df = DataFrame(np.random.default_rng(2).random((7, 4)))
        # 在DataFrame对象上绘制图形，使用给定的颜色和样式
        ax = df.plot(color=color, style="d--")
        # 检查图形中线条的颜色是否与期望结果一致
        result = [i.get_color() for i in ax.lines]
        assert result == expected
        # 检查图形中线条的线型是否都为"--"
        assert all(i.get_linestyle() == "--" for i in ax.lines)
        # 检查图形中线条的标记是否都为"d"
        assert all(i.get_marker() == "d" for i in ax.lines)

    def test_bar_colors(self):
        # 获取默认颜色循环
        default_colors = _unpack_cycler(plt.rcParams)

        # 创建一个5行5列的随机正态分布数据的DataFrame对象
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 在DataFrame对象上绘制柱状图
        ax = df.plot.bar()
        # 检查柱子的颜色是否符合默认的颜色循环
        _check_colors(ax.patches[::5], facecolors=default_colors[:5])

    def test_bar_colors_custom(self):
        custom_colors = "rgcby"
        # 创建一个5行5列的随机正态分布数据的DataFrame对象
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 在DataFrame对象上绘制柱状图，使用自定义颜色
        ax = df.plot.bar(color=custom_colors)
        # 检查柱子的颜色是否符合自定义的颜色
        _check_colors(ax.patches[::5], facecolors=custom_colors)

    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_bar_colors_cmap(self, colormap):
        # 创建一个5行5列的随机正态分布数据的DataFrame对象
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        # 在DataFrame对象上绘制柱状图，使用指定的颜色映射
        ax = df.plot.bar(colormap=colormap)
        # 获取颜色映射生成的颜色值列表
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, 5)]
        # 检查柱子的颜色是否符合颜色映射生成的颜色值
        _check_colors(ax.patches[::5], facecolors=rgba_colors)

    def test_bar_colors_single_col(self):
        # 创建一个5行5列的随机正态分布数据的DataFrame对象
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 在DataFrame对象的第一列数据上绘制柱状图，使用指定的颜色
        ax = df.loc[:, [0]].plot.bar(color="DodgerBlue")
        # 检查柱子的颜色是否符合指定的颜色
        _check_colors([ax.patches[0]], facecolors=["DodgerBlue"])

    def test_bar_colors_green(self):
        # 创建一个5行5列的随机正态分布数据的DataFrame对象
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 在DataFrame对象上绘制柱状图，使用绿色
        ax = df.plot(kind="bar", color="green")
        # 检查柱子的颜色是否都为绿色
        _check_colors(ax.patches[::5], facecolors=["green"] * 5)

    def test_bar_user_colors(self):
        # 创建一个包含"A"、"B"两列数据的DataFrame对象，以及一个颜色列
        df = DataFrame(
            {"A": range(4), "B": range(1, 5), "color": ["red", "blue", "blue", "red"]}
        )
        # 在DataFrame对象上绘制柱状图，根据"y"列数据使用指定的颜色
        ax = df.plot.bar(y="A", color=df["color"])
        # 检查柱子的颜色是否符合期望的颜色列表
        result = [p.get_facecolor() for p in ax.patches]
        expected = [
            (1.0, 0.0, 0.0, 1.0),  # 红色
            (0.0, 0.0, 1.0, 1.0),  # 蓝色
            (0.0, 0.0, 1.0, 1.0),  # 蓝色
            (1.0, 0.0, 0.0, 1.0),  # 红色
        ]
        assert result == expected
    def test_if_scatterplot_colorbar_affects_xaxis_visibility(self):
        # 解决问题 #10611，确保颜色条不影响 IPython 内联后端的 x 轴标签和刻度标签。
        
        # 创建一个形状为 (10, 3) 的随机数组
        random_array = np.random.default_rng(2).random((10, 3))
        # 使用随机数组创建 DataFrame，列名分别为 "A label", "B label", "C label"
        df = DataFrame(random_array, columns=["A label", "B label", "C label"])

        # 绘制散点图，只有 x 和 y 轴
        ax1 = df.plot.scatter(x="A label", y="B label")
        # 绘制带颜色条的散点图，颜色条基于 "C label" 列
        ax2 = df.plot.scatter(x="A label", y="B label", c="C label")

        # 获取第一个图的 x 轴次要刻度标签的可见性列表
        vis1 = [vis.get_visible() for vis in ax1.xaxis.get_minorticklabels()]
        # 获取第二个图的 x 轴次要刻度标签的可见性列表
        vis2 = [vis.get_visible() for vis in ax2.xaxis.get_minorticklabels()]
        # 断言两个列表是否相等
        assert vis1 == vis2

        # 获取第一个图的 x 轴主要刻度标签的可见性列表
        vis1 = [vis.get_visible() for vis in ax1.xaxis.get_majorticklabels()]
        # 获取第二个图的 x 轴主要刻度标签的可见性列表
        vis2 = [vis.get_visible() for vis in ax2.xaxis.get_majorticklabels()]
        # 断言两个列表是否相等
        assert vis1 == vis2

        # 断言第一个图和第二个图的 x 轴标签可见性是否相等
        assert (
            ax1.xaxis.get_label().get_visible() == ax2.xaxis.get_label().get_visible()
        )

    def test_if_hexbin_xaxis_label_is_visible(self):
        # 解决问题 #10678，确保颜色条不影响 IPython 内联后端的 x 轴标签和刻度标签。

        # 创建一个形状为 (10, 3) 的随机数组
        random_array = np.random.default_rng(2).random((10, 3))
        # 使用随机数组创建 DataFrame，列名分别为 "A label", "B label", "C label"
        df = DataFrame(random_array, columns=["A label", "B label", "C label"])

        # 绘制 hexbin 图，使用 "A label" 和 "B label" 列，网格大小为 12
        ax = df.plot.hexbin("A label", "B label", gridsize=12)
        
        # 断言 x 轴次要刻度标签全部可见
        assert all(vis.get_visible() for vis in ax.xaxis.get_minorticklabels())
        # 断言 x 轴主要刻度标签全部可见
        assert all(vis.get_visible() for vis in ax.xaxis.get_majorticklabels())
        # 断言 x 轴标签可见
        assert ax.xaxis.get_label().get_visible()

    def test_if_scatterplot_colorbars_are_next_to_parent_axes(self):
        # 创建一个形状为 (10, 3) 的随机数组
        random_array = np.random.default_rng(2).random((10, 3))
        # 使用随机数组创建 DataFrame，列名分别为 "A label", "B label", "C label"
        df = DataFrame(random_array, columns=["A label", "B label", "C label"])

        # 创建一个包含两个子图的图形对象
        fig, axes = plt.subplots(1, 2)
        # 在第一个子图中绘制带颜色条的散点图，颜色条基于 "C label" 列
        df.plot.scatter("A label", "B label", c="C label", ax=axes[0])
        # 在第二个子图中绘制带颜色条的散点图，颜色条基于 "C label" 列
        df.plot.scatter("A label", "B label", c="C label", ax=axes[1])
        # 调整子图的布局使它们紧凑显示
        plt.tight_layout()

        # 获取所有子图的位置的坐标点数组
        points = np.array([ax.get_position().get_points() for ax in fig.axes])
        # 提取所有子图的 x 坐标点
        axes_x_coords = points[:, :, 0]
        # 计算第二个子图和第一个子图之间的父级距离
        parent_distance = axes_x_coords[1, :] - axes_x_coords[0, :]
        # 计算第二个子图和第一个子图之间的颜色条距离
        colorbar_distance = axes_x_coords[3, :] - axes_x_coords[2, :]
        # 断言父级距离和颜色条距离是否接近，允许的绝对误差为 1e-7
        assert np.isclose(parent_distance, colorbar_distance, atol=1e-7).all()

    @pytest.mark.parametrize("cmap", [None, "Greys"])
    def test_scatter_with_c_column_name_with_colors(self, cmap):
        # https://github.com/pandas-dev/pandas/issues/34316

        # 创建一个包含长度和宽度数据的DataFrame
        df = DataFrame(
            [[5.1, 3.5], [4.9, 3.0], [7.0, 3.2], [6.4, 3.2], [5.9, 3.0]],
            columns=["length", "width"],
        )
        # 添加一个指定颜色的列 "species"
        df["species"] = ["r", "r", "g", "g", "b"]
        # 如果提供了cmap参数，则期望产生UserWarning警告
        if cmap is not None:
            with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
                # 绘制散点图，使用指定的x、y轴和颜色映射cmap，以及颜色列"species"
                ax = df.plot.scatter(x=0, y=1, cmap=cmap, c="species")
        else:
            # 否则，绘制散点图，使用指定的x、y轴和颜色列"species"，忽略cmap参数
            ax = df.plot.scatter(x=0, y=1, c="species", cmap=cmap)
        # 确认散点图的颜色条不存在
        assert ax.collections[0].colorbar is None

    def test_scatter_colors(self):
        # 创建一个包含列'a', 'b', 'c'的DataFrame
        df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
        # 确认调用plot.scatter()时同时指定'c'和'color'会抛出TypeError异常
        with pytest.raises(TypeError, match="Specify exactly one of `c` and `color`"):
            df.plot.scatter(x="a", y="b", c="c", color="green")

    def test_scatter_colors_not_raising_warnings(self):
        # GH-53908. 不应该触发UserWarning：没有提供用于颜色映射的数据通过'c'参数，'cmap'参数将被忽略
        # 创建一个包含列'x', 'y'的DataFrame
        df = DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        # 确认在plot.scatter()调用时不会产生警告
        with tm.assert_produces_warning(None):
            df.plot.scatter(x="x", y="y", c="b")

    def test_scatter_colors_default(self):
        # 创建一个包含列'a', 'b', 'c'的DataFrame
        df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
        # 获取默认的颜色配置
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)

        # 绘制散点图，使用列'a'和'b'作为x和y轴，以及颜色列'c'
        ax = df.plot.scatter(x="a", y="b", c="c")
        # 确认绘图使用的颜色与默认颜色匹配
        tm.assert_numpy_array_equal(
            ax.collections[0].get_facecolor()[0],
            np.array(mpl.colors.ColorConverter.to_rgba(default_colors[0])),
        )

    def test_scatter_colors_white(self):
        # 创建一个包含列'a', 'b', 'c'的DataFrame
        df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
        # 绘制散点图，使用列'a'和'b'作为x和y轴，并指定颜色为白色
        ax = df.plot.scatter(x="a", y="b", color="white")
        # 确认散点的颜色为白色
        tm.assert_numpy_array_equal(
            ax.collections[0].get_facecolor()[0],
            np.array([1, 1, 1, 1], dtype=np.float64),
        )

    def test_scatter_colorbar_different_cmap(self):
        # GH 33389
        # 创建一个包含列'x', 'y', 'c', 'x2'的DataFrame
        df = DataFrame({"x": [1, 2, 3], "y": [1, 3, 2], "c": [1, 2, 3]})
        df["x2"] = df["x"] + 1

        # 创建一个新的图形和坐标轴
        _, ax = plt.subplots()
        # 绘制第一个散点图，使用'x', 'y'列作为x和y轴，颜色映射为'cividis'色彩映射
        df.plot("x", "y", c="c", kind="scatter", cmap="cividis", ax=ax)
        # 绘制第二个散点图，使用'x2', 'y'列作为x和y轴，颜色映射为'magma'色彩映射
        df.plot("x2", "y", c="c", kind="scatter", cmap="magma", ax=ax)

        # 确认第一个散点图使用了'cividis'色彩映射
        assert ax.collections[0].cmap.name == "cividis"
        # 确认第二个散点图使用了'magma'色彩映射
        assert ax.collections[1].cmap.name == "magma"

    def test_line_colors(self):
        # 定义自定义颜色序列
        custom_colors = "rgcby"
        # 创建一个包含随机数据的DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        # 绘制包含自定义颜色序列的线图
        ax = df.plot(color=custom_colors)
        # 检查绘图中线条的颜色是否与自定义颜色序列匹配
        _check_colors(ax.get_lines(), linecolors=custom_colors)

        # 关闭所有图形窗口
        plt.close("all")

        # 创建第二个图形窗口，绘制包含自定义颜色序列的线图
        ax2 = df.plot(color=custom_colors)
        lines2 = ax2.get_lines()

        # 检查两个图形中的线条颜色是否一致
        for l1, l2 in zip(ax.get_lines(), lines2):
            assert l1.get_color() == l2.get_color()

    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    # 测试使用指定的颜色映射绘制数据帧的折线图
    def test_line_colors_cmap(self, colormap):
        # 创建一个 5x5 的随机数据帧
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 在指定的颜色映射下绘制数据帧的折线图，并获取绘图对象
        ax = df.plot(colormap=colormap)
        # 根据数据帧的行数在颜色映射中生成对应的 RGBA 颜色列表
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        # 检查绘图对象的折线颜色是否与生成的 RGBA 颜色列表匹配
        _check_colors(ax.get_lines(), linecolors=rgba_colors)

    # 测试当绘制仅包含单列数据帧时的折线颜色
    def test_line_colors_single_col(self):
        # 创建一个 5x5 的随机数据帧
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 当仅包含单列数据帧时，将颜色参数转换为列表，处理类似 df.plot(color='DodgerBlue') 的情况
        ax = df.loc[:, [0]].plot(color="DodgerBlue")
        # 检查绘图对象的折线颜色是否与指定的颜色列表匹配
        _check_colors(ax.lines, linecolors=["DodgerBlue"])

    # 测试绘制数据帧时指定单一颜色的折线
    def test_line_colors_single_color(self):
        # 创建一个 5x5 的随机数据帧
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 在绘制数据帧时指定单一颜色的折线
        ax = df.plot(color="red")
        # 检查绘图对象的折线颜色是否与指定的颜色列表匹配
        _check_colors(ax.get_lines(), linecolors=["red"] * 5)

    # 测试绘制数据帧时使用十六进制颜色值的折线颜色
    def test_line_colors_hex(self):
        # 创建一个 5x5 的随机数据帧
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 自定义颜色列表，包含多个十六进制颜色值
        custom_colors = ["#FF0000", "#0000FF", "#FFFF00", "#000000", "#FFFFFF"]
        # 在绘制数据帧时使用自定义颜色列表作为折线颜色
        ax = df.plot(color=custom_colors)
        # 检查绘图对象的折线颜色是否与自定义颜色列表匹配
        _check_colors(ax.get_lines(), linecolors=custom_colors)

    # 测试绘制数据帧时不修改颜色参数是否有效
    def test_dont_modify_colors(self):
        # 定义颜色列表
        colors = ["r", "g", "b"]
        # 绘制随机数据帧，并使用指定的颜色列表
        DataFrame(np.random.default_rng(2).random((10, 2))).plot(color=colors)
        # 断言颜色列表的长度是否保持不变
        assert len(colors) == 3

    # 测试在子图中绘制数据帧的折线颜色和样式
    def test_line_colors_and_styles_subplots(self):
        # 获取默认颜色循环器的颜色列表
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)

        # 创建一个 5x5 的随机数据帧
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        # 在子图中绘制数据帧，并获取子图对象列表
        axes = df.plot(subplots=True)
        # 遍历子图对象和默认颜色列表，检查每个子图的折线颜色是否与对应的默认颜色匹配
        for ax, c in zip(axes, list(default_colors)):
            _check_colors(ax.get_lines(), linecolors=[c])

    # 测试在子图中使用单一颜色字符串绘制数据帧的折线
    @pytest.mark.parametrize("color", ["k", "green"])
    def test_line_colors_and_styles_subplots_single_color_str(self, color):
        # 创建一个 5x5 的随机数据帧
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 在子图中使用指定的单一颜色字符串绘制数据帧的折线，并获取子图对象列表
        axes = df.plot(subplots=True, color=color)
        # 遍历子图对象，检查每个子图的折线颜色是否与指定的颜色字符串匹配
        for ax in axes:
            _check_colors(ax.get_lines(), linecolors=[color])

    # 测试在子图中使用自定义颜色列表绘制数据帧的折线
    @pytest.mark.parametrize("color", ["rgcby", list("rgcby")])
    def test_line_colors_and_styles_subplots_custom_colors(self, color):
        # 创建一个 5x5 的随机数据帧
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 在子图中使用自定义颜色列表绘制数据帧的折线，并获取子图对象列表
        axes = df.plot(color=color, subplots=True)
        # 遍历子图对象和自定义颜色列表，检查每个子图的折线颜色是否与对应的自定义颜色匹配
        for ax, c in zip(axes, list(color)):
            _check_colors(ax.get_lines(), linecolors=[c])

    # 测试在子图中使用十六进制颜色值的颜色映射绘制数据帧的折线
    def test_line_colors_and_styles_subplots_colormap_hex(self):
        # 创建一个 5x5 的随机数据帧
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 自定义颜色列表，包含多个十六进制颜色值
        custom_colors = ["#FF0000", "#0000FF", "#FFFF00", "#000000", "#FFFFFF"]
        # 在子图中使用自定义颜色列表作为折线颜色
        axes = df.plot(color=custom_colors, subplots=True)
        # 遍历子图对象和自定义颜色列表，检查每个子图的折线颜色是否与对应的自定义颜色匹配
        for ax, c in zip(axes, list(custom_colors)):
            _check_colors(ax.get_lines(), linecolors=[c])

    # 参数化测试：在子图中使用不同的颜色映射绘制数据帧的折线
    @pytest.mark.parametrize("cmap", ["jet", cm.jet])
    def test_line_colors_and_styles_subplots_colormap_subplot(self, cmap):
        # GH 9894
        # 创建一个5x5的随机数据框
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 根据数据框的行数生成一组RGBA颜色值
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        # 使用DataFrame的plot方法创建子图，使用给定的颜色映射
        axes = df.plot(colormap=cmap, subplots=True)
        # 遍历每个子图和对应的颜色，调用_check_colors函数检查线条颜色
        for ax, c in zip(axes, rgba_colors):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_line_colors_and_styles_subplots_single_col(self):
        # GH 9894
        # 创建一个5x5的随机数据框
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 如果只绘制一列数据，将颜色参数转换为列表形式，例如处理类似 df.plot(color='DodgerBlue') 这样的情况
        axes = df.loc[:, [0]].plot(color="DodgerBlue", subplots=True)
        # 调用_check_colors函数检查线条颜色
        _check_colors(axes[0].lines, linecolors=["DodgerBlue"])

    def test_line_colors_and_styles_subplots_single_char(self):
        # GH 9894
        # 创建一个5x5的随机数据框
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 使用单字符样式进行绘制
        axes = df.plot(style="r", subplots=True)
        # 遍历每个子图，调用_check_colors函数检查线条颜色
        for ax in axes:
            _check_colors(ax.get_lines(), linecolors=["r"])

    def test_line_colors_and_styles_subplots_list_styles(self):
        # GH 9894
        # 创建一个5x5的随机数据框
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 使用列表形式的样式进行绘制
        styles = list("rgcby")
        axes = df.plot(style=styles, subplots=True)
        # 遍历每个子图和对应的样式颜色，调用_check_colors函数检查线条颜色
        for ax, c in zip(axes, styles):
            _check_colors(ax.get_lines(), linecolors=[c])

    def test_area_colors(self):
        # 定义自定义颜色列表
        custom_colors = "rgcby"
        # 创建一个5x5的随机数据框
        df = DataFrame(np.random.default_rng(2).random((5, 5)))

        # 使用DataFrame的plot.area方法创建面积图，指定颜色参数为自定义颜色列表
        ax = df.plot.area(color=custom_colors)
        # 调用_check_colors函数检查线条颜色
        _check_colors(ax.get_lines(), linecolors=custom_colors)
        
        # 获取所有子对象中的多边形集合，并检查其颜色
        poly = [
            o
            for o in ax.get_children()
            if isinstance(o, mpl.collections.PolyCollection)
        ]
        _check_colors(poly, facecolors=custom_colors)

        # 获取图例中的处理对象，检查其颜色
        handles, _ = ax.get_legend_handles_labels()
        _check_colors(handles, facecolors=custom_colors)

        # 对每个处理对象断言其透明度为None
        for h in handles:
            assert h.get_alpha() is None

    def test_area_colors_poly(self):
        # 创建一个5x5的随机数据框
        df = DataFrame(np.random.default_rng(2).random((5, 5)))
        # 使用DataFrame的plot.area方法创建面积图，指定颜色映射为"jet"
        ax = df.plot.area(colormap="jet")
        # 生成与数据框行数相等的jet颜色列表
        jet_colors = [mpl.cm.jet(n) for n in np.linspace(0, 1, len(df))]
        # 调用_check_colors函数检查线条颜色
        _check_colors(ax.get_lines(), linecolors=jet_colors)
        
        # 获取所有子对象中的多边形集合，并检查其颜色
        poly = [
            o
            for o in ax.get_children()
            if isinstance(o, mpl.collections.PolyCollection)
        ]
        _check_colors(poly, facecolors=jet_colors)

        # 获取图例中的处理对象，检查其颜色
        handles, _ = ax.get_legend_handles_labels()
        _check_colors(handles, facecolors=jet_colors)
        
        # 对每个处理对象断言其透明度为None
        for h in handles:
            assert h.get_alpha() is None
    def test_area_colors_stacked_false(self):
        # 创建一个 5x5 的随机数据帧
        df = DataFrame(np.random.default_rng(2).random((5, 5)))
        # 使用 Jet 颜色映射生成颜色列表
        jet_colors = [mpl.cm.jet(n) for n in np.linspace(0, 1, len(df))]
        # 在 stacked=False 的情况下绘制面积图，并设置透明度为 0.5
        ax = df.plot.area(colormap=mpl.cm.jet, stacked=False)
        # 检查线条颜色是否与 jet_colors 匹配
        _check_colors(ax.get_lines(), linecolors=jet_colors)
        # 获取所有 PolyCollection 对象，用于检查面积颜色
        poly = [
            o
            for o in ax.get_children()
            if isinstance(o, mpl.collections.PolyCollection)
        ]
        # 将 Jet 颜色列表每个元素的透明度设为 0.5
        jet_with_alpha = [(c[0], c[1], c[2], 0.5) for c in jet_colors]
        # 检查 PolyCollection 的面颜色是否与 jet_with_alpha 匹配
        _check_colors(poly, facecolors=jet_with_alpha)

        # 获取图例的句柄和标签
        handles, _ = ax.get_legend_handles_labels()
        linecolors = jet_with_alpha
        # 检查图例中的线条颜色是否与 linecolors 匹配
        _check_colors(handles[: len(jet_colors)], linecolors=linecolors)
        # 断言所有图例的透明度是否为 0.5
        for h in handles:
            assert h.get_alpha() == 0.5

    def test_hist_colors(self):
        # 解包 Matplotlib 默认的颜色循环器
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)

        # 创建一个 5x5 的标准正态分布数据帧
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 绘制直方图并获取绘制的块对象
        ax = df.plot.hist()
        # 检查直方图块的面颜色是否与 default_colors 的前5个颜色匹配
        _check_colors(ax.patches[::10], facecolors=default_colors[:5])

    def test_hist_colors_single_custom(self):
        # 创建一个 5x5 的标准正态分布数据帧
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 使用自定义颜色列表绘制直方图并获取绘制的块对象
        custom_colors = "rgcby"
        ax = df.plot.hist(color=custom_colors)
        # 检查直方图块的面颜色是否与 custom_colors 匹配
        _check_colors(ax.patches[::10], facecolors=custom_colors)

    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_hist_colors_cmap(self, colormap):
        # 创建一个 5x5 的标准正态分布数据帧
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 使用指定的颜色映射绘制直方图并获取绘制的块对象
        ax = df.plot.hist(colormap=colormap)
        # 生成用于检查的 RGBA 颜色列表
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, 5)]
        # 检查直方图块的面颜色是否与 rgba_colors 匹配
        _check_colors(ax.patches[::10], facecolors=rgba_colors)

    def test_hist_colors_single_col(self):
        # 创建一个 5x5 的标准正态分布数据帧，并选择第一列进行绘制
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.loc[:, [0]].plot.hist(color="DodgerBlue")
        # 检查直方图块的面颜色是否为 "DodgerBlue"
        _check_colors([ax.patches[0]], facecolors=["DodgerBlue"])

    def test_hist_colors_single_color(self):
        # 创建一个 5x5 的标准正态分布数据帧，并以绿色绘制直方图
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        ax = df.plot(kind="hist", color="green")
        # 检查直方图块的面颜色是否为 5 个 "green"
        _check_colors(ax.patches[::10], facecolors=["green"] * 5)

    def test_kde_colors(self):
        # 如果缺少 scipy 库则跳过此测试
        pytest.importorskip("scipy")
        # 创建一个 5x5 的随机数据帧
        df = DataFrame(np.random.default_rng(2).random((5, 5)))
        # 绘制核密度估计图并获取绘制的线对象
        ax = df.plot.kde(color="rgcby")
        # 检查核密度估计线的颜色是否与 "rgcby" 匹配
        _check_colors(ax.get_lines(), linecolors="rgcby")

    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_kde_colors_cmap(self, colormap):
        # 如果缺少 scipy 库则跳过此测试
        pytest.importorskip("scipy")
        # 创建一个 5x5 的标准正态分布数据帧
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 使用指定的颜色映射绘制核密度估计图并获取绘制的线对象
        ax = df.plot.kde(colormap=colormap)
        # 生成用于检查的 RGBA 颜色列表
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
        # 检查核密度估计线的颜色是否与 rgba_colors 匹配
        _check_colors(ax.get_lines(), linecolors=rgba_colors)
    # 定义一个测试函数，用于测试 KDE 图的颜色和样式在子图中的展示

        # 导入 scipy 库，如果导入失败则跳过该测试
        pytest.importorskip("scipy")

        # 从 matplotlib.pyplot 的默认参数中解包默认颜色
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)

        # 创建一个 5x5 的随机数据帧 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        # 使用 KDE 方法绘制数据帧的子图，并返回子图对象
        axes = df.plot(kind="kde", subplots=True)

        # 遍历子图 axes 和默认颜色列表，并检查线条颜色
        for ax, c in zip(axes, list(default_colors)):
            _check_colors(ax.get_lines(), linecolors=[c])

    # 使用参数化测试，测试 KDE 图的颜色和样式在单列字符串中的展示
    @pytest.mark.parametrize("colormap", ["k", "red"])
    def test_kde_colors_and_styles_subplots_single_col_str(self, colormap):
        pytest.importorskip("scipy")

        # 创建一个 5x5 的随机数据帧 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        # 使用 KDE 方法绘制数据帧的单列子图，并返回子图对象
        axes = df.plot(kind="kde", color=colormap, subplots=True)

        # 遍历子图 axes，并检查线条颜色
        for ax in axes:
            _check_colors(ax.get_lines(), linecolors=[colormap])

    # 测试 KDE 图的颜色和样式在自定义颜色列表中的展示
    def test_kde_colors_and_styles_subplots_custom_color(self):
        pytest.importorskip("scipy")

        # 创建一个 5x5 的随机数据帧 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        # 自定义颜色列表
        custom_colors = "rgcby"

        # 使用 KDE 方法绘制数据帧的子图，并返回子图对象
        axes = df.plot(kind="kde", color=custom_colors, subplots=True)

        # 遍历子图 axes 和自定义颜色列表，并检查线条颜色
        for ax, c in zip(axes, list(custom_colors)):
            _check_colors(ax.get_lines(), linecolors=[c])

    # 使用参数化测试，测试 KDE 图的颜色和样式在 colormap 参数中的展示
    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_kde_colors_and_styles_subplots_cmap(self, colormap):
        pytest.importorskip("scipy")

        # 创建一个 5x5 的随机数据帧 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        # 计算每行数据的 RGBA 颜色
        rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]

        # 使用 KDE 方法绘制数据帧的子图，并返回子图对象
        axes = df.plot(kind="kde", colormap=colormap, subplots=True)

        # 遍历子图 axes 和 RGBA 颜色列表，并检查线条颜色
        for ax, c in zip(axes, rgba_colors):
            _check_colors(ax.get_lines(), linecolors=[c])

    # 测试 KDE 图的颜色和样式在单列数据中的展示
    def test_kde_colors_and_styles_subplots_single_col(self):
        pytest.importorskip("scipy")

        # 创建一个 5x5 的随机数据帧 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        # 如果绘制单列帧，则将颜色参数转换为列表
        axes = df.loc[:, [0]].plot(kind="kde", color="DodgerBlue", subplots=True)

        # 检查线条颜色
        _check_colors(axes[0].lines, linecolors=["DodgerBlue"])

    # 测试 KDE 图的颜色和样式在单个字符样式中的展示
    def test_kde_colors_and_styles_subplots_single_char(self):
        pytest.importorskip("scipy")

        # 创建一个 5x5 的随机数据帧 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        # 使用单字符样式绘制 KDE 图，并返回子图对象
        axes = df.plot(kind="kde", style="r", subplots=True)

        # 遍历子图 axes，并检查线条颜色
        for ax in axes:
            _check_colors(ax.get_lines(), linecolors=["r"])

    # 测试 KDE 图的颜色和样式在样式列表中的展示
    def test_kde_colors_and_styles_subplots_list(self):
        pytest.importorskip("scipy")

        # 创建一个 5x5 的随机数据帧 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        # 样式列表
        styles = list("rgcby")

        # 使用样式列表绘制 KDE 图，并返回子图对象
        axes = df.plot(kind="kde", style=styles, subplots=True)

        # 遍历子图 axes 和样式列表，并检查线条颜色
        for ax, c in zip(axes, styles):
            _check_colors(ax.get_lines(), linecolors=[c])
    # 测试箱线图的颜色设置，默认使用全局颜色循环
    def test_boxplot_colors(self):
        # 解包全局颜色循环为默认颜色列表
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)

        # 创建一个 5x5 的随机数据框
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 绘制箱线图并返回字典形式的图形对象
        bp = df.plot.box(return_type="dict")
        # 检查箱线图的颜色设置是否符合预期
        _check_colors_box(
            bp,
            default_colors[0],   # boxes 默认颜色
            default_colors[0],   # whiskers 默认颜色
            default_colors[2],   # medians 默认颜色
            default_colors[0],   # caps 默认颜色
        )

    # 测试使用字典指定箱线图的颜色设置
    def test_boxplot_colors_dict_colors(self):
        # 创建一个 5x5 的随机数据框
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 自定义颜色字典
        dict_colors = {
            "boxes": "#572923",
            "whiskers": "#982042",
            "medians": "#804823",
            "caps": "#123456",
        }
        # 绘制箱线图并返回字典形式的图形对象，使用自定义颜色字典
        bp = df.plot.box(color=dict_colors, sym="r+", return_type="dict")
        # 检查箱线图的颜色设置是否符合预期
        _check_colors_box(
            bp,
            dict_colors["boxes"],
            dict_colors["whiskers"],
            dict_colors["medians"],
            dict_colors["caps"],
            "r",   # fliers 的颜色设置
        )

    # 测试使用部分颜色的字典设置箱线图的颜色
    def test_boxplot_colors_default_color(self):
        # 解包全局颜色循环为默认颜色列表
        default_colors = _unpack_cycler(mpl.pyplot.rcParams)
        # 创建一个 5x5 的随机数据框
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 部分颜色设置的字典
        dict_colors = {"whiskers": "c", "medians": "m"}
        # 绘制箱线图并返回字典形式的图形对象，使用部分颜色设置的字典
        bp = df.plot.box(color=dict_colors, return_type="dict")
        # 检查箱线图的颜色设置是否符合预期
        _check_colors_box(bp, default_colors[0], "c", "m", default_colors[0])

    # 测试使用 colormap 设置箱线图的颜色
    @pytest.mark.parametrize("colormap", ["jet", cm.jet])
    def test_boxplot_colors_cmap(self, colormap):
        # 创建一个 5x5 的随机数据框
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 绘制箱线图并返回字典形式的图形对象，使用 colormap 设置
        bp = df.plot.box(colormap=colormap, return_type="dict")
        # 生成 colormap 对应的颜色列表
        jet_colors = [cm.jet(n) for n in np.linspace(0, 1, 3)]
        # 检查箱线图的颜色设置是否符合预期
        _check_colors_box(
            bp, jet_colors[0], jet_colors[0], jet_colors[2], jet_colors[0]
        )

    # 测试使用单一颜色设置所有元素的箱线图
    def test_boxplot_colors_single(self):
        # 创建一个 5x5 的随机数据框
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 绘制箱线图并返回字典形式的图形对象，所有元素使用单一颜色
        bp = df.plot.box(color="DodgerBlue", return_type="dict")
        # 检查箱线图的颜色设置是否符合预期
        _check_colors_box(bp, "DodgerBlue", "DodgerBlue", "DodgerBlue", "DodgerBlue")

    # 测试使用元组设置所有元素的箱线图颜色及 fliers 的符号
    def test_boxplot_colors_tuple(self):
        # 创建一个 5x5 的随机数据框
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 绘制箱线图并返回字典形式的图形对象，所有元素使用元组颜色，fliers 使用指定符号
        bp = df.plot.box(color=(0, 1, 0), sym="#123456", return_type="dict")
        # 检查箱线图的颜色设置是否符合预期，包括 fliers 的颜色设置
        _check_colors_box(bp, (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), "#123456")

    # 测试颜色字典中包含无效键时是否引发 ValueError
    def test_boxplot_colors_invalid(self):
        # 创建一个 5x5 的随机数据框
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 预期的错误信息
        msg = re.escape(
            "color dict contains invalid key 'xxxx'. The key must be either "
            "['boxes', 'whiskers', 'medians', 'caps']"
        )
        # 检查绘制箱线图时，使用包含无效键的颜色字典是否引发 ValueError
        with pytest.raises(ValueError, match=msg):
            df.plot.box(color={"boxes": "red", "xxxx": "blue"})
    # 测试默认颜色循环设置功能
    def test_default_color_cycle(self):
        # 导入 cycler 库
        import cycler

        # 定义颜色列表
        colors = list("rgbk")
        # 设置当前绘图的轴属性循环为指定颜色循环
        plt.rcParams["axes.prop_cycle"] = cycler.cycler("color", colors)

        # 创建随机数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        # 绘制 DataFrame 的折线图
        ax = df.plot()

        # 获取预期的颜色循环
        expected = _unpack_cycler(plt.rcParams)[:3]
        # 检查绘图中折线的颜色是否符合预期
        _check_colors(ax.get_lines(), linecolors=expected)

    # 测试禁用颜色条功能
    def test_no_color_bar(self):
        # 创建包含随机数据的 DataFrame
        df = DataFrame(
            {
                "A": np.random.default_rng(2).uniform(size=20),
                "B": np.random.default_rng(2).uniform(size=20),
                "C": np.arange(20) + np.random.default_rng(2).uniform(size=20),
            }
        )
        # 绘制 DataFrame 的 hexbin 图，不显示颜色条
        ax = df.plot.hexbin(x="A", y="B", colorbar=None)
        # 断言没有颜色条被创建
        assert ax.collections[0].colorbar is None

    # 测试混合使用 cmap 和 colormap 时抛出异常
    def test_mixing_cmap_and_colormap_raises(self):
        # 创建包含随机数据的 DataFrame
        df = DataFrame(
            {
                "A": np.random.default_rng(2).uniform(size=20),
                "B": np.random.default_rng(2).uniform(size=20),
                "C": np.arange(20) + np.random.default_rng(2).uniform(size=20),
            }
        )
        # 准备异常信息
        msg = "Only specify one of `cmap` and `colormap`"
        # 测试调用 plot.hexbin 时是否引发 TypeError 异常，并且异常信息匹配预期消息
        with pytest.raises(TypeError, match=msg):
            df.plot.hexbin(x="A", y="B", cmap="YlGn", colormap="BuGn")

    # 测试传递自定义条形图颜色
    def test_passed_bar_colors(self):
        # 定义颜色元组列表
        color_tuples = [(0.9, 0, 0, 1), (0, 0.9, 0, 1), (0, 0, 0.9, 1)]
        # 创建 ListedColormap 对象
        colormap = mpl.colors.ListedColormap(color_tuples)
        # 绘制 DataFrame 的条形图，指定使用 colormap
        barplot = DataFrame([[1, 2, 3]]).plot(kind="bar", cmap=colormap)
        # 断言条形图的颜色与预期的颜色元组列表相匹配
        assert color_tuples == [c.get_facecolor() for c in barplot.patches]

    # 测试使用 rcParams 设置条形图颜色
    def test_rcParams_bar_colors(self):
        # 定义颜色元组列表
        color_tuples = [(0.9, 0, 0, 1), (0, 0.9, 0, 1), (0, 0, 0.9, 1)]
        # 使用 mpl.rc_context 上下文环境，设置 axes.prop_cycle 参数为指定的颜色循环
        with mpl.rc_context(rc={"axes.prop_cycle": mpl.cycler("color", color_tuples)}):
            # 绘制 DataFrame 的条形图
            barplot = DataFrame([[1, 2, 3]]).plot(kind="bar")
        # 断言条形图的颜色与预期的颜色元组列表相匹配
        assert color_tuples == [c.get_facecolor() for c in barplot.patches]

    # 测试具有相同列名的列的颜色
    def test_colors_of_columns_with_same_name(self):
        # ISSUE 11136 -> https://github.com/pandas-dev/pandas/issues/11136
        # 创建包含重复列标签的 DataFrame，并测试它们的颜色
        df = DataFrame({"b": [0, 1, 0], "a": [1, 2, 3]})
        df1 = DataFrame({"a": [2, 4, 6]})
        # 沿轴1（列方向）拼接 DataFrame
        df_concat = pd.concat([df, df1], axis=1)
        # 绘制合并后的 DataFrame
        result = df_concat.plot()
        # 获取图例对象
        legend = result.get_legend()
        # 根据 Matplotlib 版本不同，获取不同的图例句柄
        if Version(mpl.__version__) < Version("3.7"):
            handles = legend.legendHandles
        else:
            handles = legend.legend_handles
        # 遍历图例句柄和结果中的线条，断言它们的颜色匹配
        for legend, line in zip(handles, result.lines):
            assert legend.get_color() == line.get_color()

    # 测试无效的 colormap 参数
    def test_invalid_colormap(self):
        # 创建包含随机数据的 DataFrame
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 2)), columns=["A", "B"]
        )
        # 准备异常信息
        msg = "(is not a valid value)|(is not a known colormap)"
        # 测试调用 plot 方法时是否引发 ValueError 或 KeyError 异常，并且异常信息匹配预期消息
        with pytest.raises((ValueError, KeyError), match=msg):
            df.plot(colormap="invalid_colormap")
    def test_dataframe_none_color(self):
        # 定义一个测试函数，用于测试数据框绘图中的颜色为空的情况
        # GH51953 是 GitHub 上的 issue 编号，指明了与此函数相关的问题
        df = DataFrame([[1, 2, 3]])
        # 创建一个包含一行数据的数据框
        ax = df.plot(color=None)
        # 在数据框上绘制图形，颜色参数设置为 None，表示使用默认颜色
        expected = _unpack_cycler(mpl.pyplot.rcParams)[:3]
        # 从 Matplotlib 的全局配置中获取默认的颜色循环并解析前三个颜色
        _check_colors(ax.get_lines(), linecolors=expected)
        # 检查绘图对象中线条的颜色是否符合预期
```