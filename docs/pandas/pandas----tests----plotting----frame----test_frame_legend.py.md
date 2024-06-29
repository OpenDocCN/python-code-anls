# `D:\src\scipysrc\pandas\pandas\tests\plotting\frame\test_frame_legend.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于单元测试

import pandas.util._test_decorators as td  # 导入 pandas 内部的测试装饰器模块

from pandas import (  # 从 pandas 库中导入 DataFrame 和 date_range 函数
    DataFrame,
    date_range,
)
from pandas.tests.plotting.common import (  # 从 pandas 测试模块中导入一些绘图相关的函数
    _check_legend_labels,
    _check_legend_marker,
    _check_text_labels,
)
from pandas.util.version import Version  # 导入 Version 类，用于处理版本信息

mpl = pytest.importorskip("matplotlib")  # 导入并检查 matplotlib 库，如果导入失败则跳过测试


class TestFrameLegend:
    @pytest.mark.xfail(  # 标记测试为预期失败状态，理由是 matplotlib 中的已知 bug
        reason=(
            "Open bug in matplotlib "
            "https://github.com/matplotlib/matplotlib/issues/11357"
        )
    )
    def test_mixed_yerr(self):
        # 创建一个包含两列的 DataFrame 对象
        df = DataFrame([{"x": 1, "a": 1, "b": 1}, {"x": 2, "a": 2, "b": 3}])

        # 在新图形上绘制 'x' vs 'a'，并添加误差线，设置标签为 'orange'
        ax = df.plot("x", "a", c="orange", yerr=0.1, label="orange")
        # 在相同图形上绘制 'x' vs 'b'，使用不同颜色并无误差线，共享 'ax' 轴，设置标签为 'blue'
        df.plot("x", "b", c="blue", yerr=None, ax=ax, label="blue")

        # 获取图例对象
        legend = ax.get_legend()
        # 根据 matplotlib 版本选择不同的处理方式获取图例句柄
        if Version(mpl.__version__) < Version("3.7"):
            result_handles = legend.legendHandles
        else:
            result_handles = legend.legend_handles

        # 断言图例句柄的类型
        assert isinstance(result_handles[0], mpl.collections.LineCollection)
        assert isinstance(result_handles[1], mpl.lines.Line2D)

    def test_legend_false(self):
        # 创建包含两列的 DataFrame 对象
        df = DataFrame({"a": [1, 1], "b": [2, 3]})
        # 创建另一个 DataFrame 对象
        df2 = DataFrame({"d": [2.5, 2.5]})

        # 在新图形上绘制 df 的内容，启用图例，指定颜色映射，第二 Y 轴显示 'b' 列
        ax = df.plot(legend=True, color={"a": "blue", "b": "green"}, secondary_y="b")
        # 在相同图形上绘制 df2 的内容，启用图例，指定颜色映射，共享 'ax' 轴
        df2.plot(legend=True, color={"d": "red"}, ax=ax)
        # 获取图例对象
        legend = ax.get_legend()
        # 根据 matplotlib 版本选择不同的处理方式获取图例句柄
        if Version(mpl.__version__) < Version("3.7"):
            handles = legend.legendHandles
        else:
            handles = legend.legend_handles
        # 断言图例句柄的颜色
        result = [handle.get_color() for handle in handles]
        expected = ["blue", "green", "red"]
        assert result == expected

    @pytest.mark.parametrize("kind", ["line", "bar", "barh", "kde", "area", "hist"])
    def test_df_legend_labels(self, kind):
        pytest.importorskip("scipy")  # 导入并检查 scipy 库，如果导入失败则跳过测试
        # 创建一个随机数据填充的 DataFrame 对象，包含三列
        df = DataFrame(np.random.default_rng(2).random((3, 3)), columns=["a", "b", "c"])
        df2 = DataFrame(  # 创建另一个随机数据填充的 DataFrame 对象，包含三列
            np.random.default_rng(2).random((3, 3)), columns=["d", "e", "f"]
        )
        df3 = DataFrame(  # 创建第三个随机数据填充的 DataFrame 对象，包含三列
            np.random.default_rng(2).random((3, 3)), columns=["g", "h", "i"]
        )
        df4 = DataFrame(  # 创建第四个随机数据填充的 DataFrame 对象，包含三列
            np.random.default_rng(2).random((3, 3)), columns=["j", "k", "l"]
        )

        # 根据参数 'kind' 绘制不同类型的图形，并启用图例
        ax = df.plot(kind=kind, legend=True)
        # 检查图例标签
        _check_legend_labels(ax, labels=df.columns)

        # 在相同图形上绘制 df2 的内容，禁用图例，共享 'ax' 轴
        ax = df2.plot(kind=kind, legend=False, ax=ax)
        # 再次检查图例标签
        _check_legend_labels(ax, labels=df.columns)

        # 在相同图形上绘制 df3 的内容，启用图例，共享 'ax' 轴，并合并图例标签
        ax = df3.plot(kind=kind, legend=True, ax=ax)
        _check_legend_labels(ax, labels=df.columns.union(df3.columns))

        # 在相同图形上绘制 df4 的内容，启用图例，共享 'ax' 轴，并反转图例标签顺序
        ax = df4.plot(kind=kind, legend="reverse", ax=ax)
        expected = list(df.columns.union(df3.columns)) + list(reversed(df4.columns))
        # 检查最终图例标签
        _check_legend_labels(ax, labels=expected)
    def test_df_legend_labels_secondary_y(self):
        # 检查是否有 scipy 库，如果没有则跳过测试
        pytest.importorskip("scipy")
        
        # 创建 DataFrame df，包含3行3列的随机数数据，列名为 ["a", "b", "c"]
        df = DataFrame(np.random.default_rng(2).random((3, 3)), columns=["a", "b", "c"])
        
        # 创建 DataFrame df2，包含3行3列的随机数数据，列名为 ["d", "e", "f"]
        df2 = DataFrame(
            np.random.default_rng(2).random((3, 3)), columns=["d", "e", "f"]
        )
        
        # 创建 DataFrame df3，包含3行3列的随机数数据，列名为 ["g", "h", "i"]
        df3 = DataFrame(
            np.random.default_rng(2).random((3, 3)), columns=["g", "h", "i"]
        )
        
        # 对 df 执行绘图，legend=True，secondary_y="b"
        ax = df.plot(legend=True, secondary_y="b")
        
        # 检查图例标签是否符合预期
        _check_legend_labels(ax, labels=["a", "b (right)", "c"])
        
        # 对 df2 执行绘图，legend=False，将图绘制在已有的 ax 上
        ax = df2.plot(legend=False, ax=ax)
        
        # 再次检查图例标签是否符合预期
        _check_legend_labels(ax, labels=["a", "b (right)", "c"])
        
        # 对 df3 执行条形图绘制，legend=True，secondary_y="h"，将图绘制在已有的 ax 上
        ax = df3.plot(kind="bar", legend=True, secondary_y="h", ax=ax)
        
        # 最后检查图例标签是否符合预期
        _check_legend_labels(ax, labels=["a", "b (right)", "c", "g", "h (right)", "i"])

    def test_df_legend_labels_time_series(self):
        # 检查是否有 scipy 库，如果没有则跳过测试
        pytest.importorskip("scipy")
        
        # 创建日期范围为 "1/1/2014" 开始的3个日期索引
        ind = date_range("1/1/2014", periods=3)
        
        # 创建时间序列 DataFrame df，包含3行3列的标准正态分布随机数数据，列名为 ["a", "b", "c"]，使用 ind 作为索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=["a", "b", "c"],
            index=ind,
        )
        
        # 创建时间序列 DataFrame df2，包含3行3列的标准正态分布随机数数据，列名为 ["d", "e", "f"]，使用 ind 作为索引
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=["d", "e", "f"],
            index=ind,
        )
        
        # 创建时间序列 DataFrame df3，包含3行3列的标准正态分布随机数数据，列名为 ["g", "h", "i"]，使用 ind 作为索引
        df3 = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=["g", "h", "i"],
            index=ind,
        )
        
        # 对 df 执行绘图，legend=True，secondary_y="b"
        ax = df.plot(legend=True, secondary_y="b")
        
        # 检查图例标签是否符合预期
        _check_legend_labels(ax, labels=["a", "b (right)", "c"])
        
        # 对 df2 执行绘图，legend=False，将图绘制在已有的 ax 上
        ax = df2.plot(legend=False, ax=ax)
        
        # 再次检查图例标签是否符合预期
        _check_legend_labels(ax, labels=["a", "b (right)", "c"])
        
        # 对 df3 执行绘图，legend=True，将图绘制在已有的 ax 上
        ax = df3.plot(legend=True, ax=ax)
        
        # 最后检查图例标签是否符合预期
        _check_legend_labels(ax, labels=["a", "b (right)", "c", "g", "h", "i"])

    def test_df_legend_labels_time_series_scatter(self):
        # 检查是否有 scipy 库，如果没有则跳过测试
        pytest.importorskip("scipy")
        
        # 创建日期范围为 "1/1/2014" 开始的3个日期索引
        ind = date_range("1/1/2014", periods=3)
        
        # 创建时间序列 DataFrame df，包含3行3列的标准正态分布随机数数据，列名为 ["a", "b", "c"]，使用 ind 作为索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=["a", "b", "c"],
            index=ind,
        )
        
        # 创建时间序列 DataFrame df2，包含3行3列的标准正态分布随机数数据，列名为 ["d", "e", "f"]，使用 ind 作为索引
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=["d", "e", "f"],
            index=ind,
        )
        
        # 创建时间序列 DataFrame df3，包含3行3列的标准正态分布随机数数据，列名为 ["g", "h", "i"]，使用 ind 作为索引
        df3 = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=["g", "h", "i"],
            index=ind,
        )
        
        # 对 df 执行散点图绘制，x="a"，y="b"，label="data1"
        ax = df.plot.scatter(x="a", y="b", label="data1")
        
        # 检查图例标签是否符合预期
        _check_legend_labels(ax, labels=["data1"])
        
        # 对 df2 执行散点图绘制，x="d"，y="e"，legend=False，label="data2"，将图绘制在已有的 ax 上
        ax = df2.plot.scatter(x="d", y="e", legend=False, label="data2", ax=ax)
        
        # 再次检查图例标签是否符合预期
        _check_legend_labels(ax, labels=["data1"])
        
        # 对 df3 执行散点图绘制，x="g"，y="h"，label="data3"，将图绘制在已有的 ax 上
        ax = df3.plot.scatter(x="g", y="h", label="data3", ax=ax)
        
        # 最后检查图例标签是否符合预期
        _check_legend_labels(ax, labels=["data1", "data3"])
    def test_df_legend_labels_time_series_no_mutate(self):
        # 检查是否能导入 scipy 库，如果不能则跳过这个测试
        pytest.importorskip("scipy")
        # 创建一个日期范围对象作为索引
        ind = date_range("1/1/2014", periods=3)
        # 创建一个 DataFrame，填充随机正态分布数据，指定列名和索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 3)),
            columns=["a", "b", "c"],
            index=ind,
        )
        # 确保标签参数传递正确，并且索引名称不改变
        # 列名不改变
        df5 = df.set_index("a")
        # 在指定列 'b' 上绘制图表，并检查图例标签是否正确
        ax = df5.plot(y="b")
        _check_legend_labels(ax, labels=["b"])
        # 在指定列 'b' 上绘制图表，并指定标签为 'LABEL_b'，检查图例标签是否正确
        ax = df5.plot(y="b", label="LABEL_b")
        _check_legend_labels(ax, labels=["LABEL_b"])
        # 检查 x 轴标签是否为 'a'
        _check_text_labels(ax.xaxis.get_label(), "a")
        # 在指定列 'c' 上绘制图表，并指定标签为 'LABEL_c'，并在同一轴上绘制，检查图例标签是否正确
        ax = df5.plot(y="c", label="LABEL_c", ax=ax)
        _check_legend_labels(ax, labels=["LABEL_b", "LABEL_c"])
        # 断言 DataFrame 的列名列表是否为 ["b", "c"]
        assert df5.columns.tolist() == ["b", "c"]

    def test_missing_marker_multi_plots_on_same_ax(self):
        # GH 18222 issue 的测试
        # 创建包含数据的 DataFrame，列为 ["x", "r", "g", "b"]
        df = DataFrame(data=[[1, 1, 1, 1], [2, 2, 4, 8]], columns=["x", "r", "g", "b"])
        # 创建一个包含 3 个子图的图像，返回图像对象和子图数组
        _, ax = mpl.pyplot.subplots(nrows=1, ncols=3)
        # 左侧图表
        # 在子图 ax[0] 上绘制 x 对 r, g, b 列的数据点图，指定线宽为 0，标记为 "o", "x", "o"，颜色为 r, g, b
        df.plot(x="x", y="r", linewidth=0, marker="o", color="r", ax=ax[0])
        df.plot(x="x", y="g", linewidth=1, marker="x", color="g", ax=ax[0])
        df.plot(x="x", y="b", linewidth=1, marker="o", color="b", ax=ax[0])
        # 检查 ax[0] 上的图例标签是否正确
        _check_legend_labels(ax[0], labels=["r", "g", "b"])
        # 检查 ax[0] 上的图例标记是否正确
        _check_legend_marker(ax[0], expected_markers=["o", "x", "o"])
        
        # 中间图表
        # 在子图 ax[1] 上绘制 x 对 b, r, g 列的数据点图，指定线宽为 1，标记为 "o", "o", "x"，颜色为 b, r, g
        df.plot(x="x", y="b", linewidth=1, marker="o", color="b", ax=ax[1])
        df.plot(x="x", y="r", linewidth=0, marker="o", color="r", ax=ax[1])
        df.plot(x="x", y="g", linewidth=1, marker="x", color="g", ax=ax[1])
        # 检查 ax[1] 上的图例标签是否正确
        _check_legend_labels(ax[1], labels=["b", "r", "g"])
        # 检查 ax[1] 上的图例标记是否正确
        _check_legend_marker(ax[1], expected_markers=["o", "o", "x"])
        
        # 右侧图表
        # 在子图 ax[2] 上绘制 x 对 g, b, r 列的数据点图，指定线宽为 1，标记为 "x", "o", "o"，颜色为 g, b, r
        df.plot(x="x", y="g", linewidth=1, marker="x", color="g", ax=ax[2])
        df.plot(x="x", y="b", linewidth=1, marker="o", color="b", ax=ax[2])
        df.plot(x="x", y="r", linewidth=0, marker="o", color="r", ax=ax[2])
        # 检查 ax[2] 上的图例标签是否正确
        _check_legend_labels(ax[2], labels=["g", "b", "r"])
        # 检查 ax[2] 上的图例标记是否正确
        _check_legend_marker(ax[2], expected_markers=["x", "o", "o"])
    # 测试 DataFrame 的列名是多级索引，使用随机生成的标准正态分布数据
    def test_legend_name(self):
        # 创建一个多级索引 DataFrame，数据为随机生成的标准正态分布
        multi = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            columns=[np.array(["a", "a", "b", "b"]), np.array(["x", "y", "x", "y"])],
        )
        # 设置列名的层级名称为 ["group", "individual"]
        multi.columns.names = ["group", "individual"]

        # 绘制多级索引 DataFrame 的图，并获取绘图对象 ax
        ax = multi.plot()
        # 获取图例对象的标题
        leg_title = ax.legend_.get_title()
        # 检查图例标题的文本标签是否符合预期 "group,individual"
        _check_text_labels(leg_title, "group,individual")

        # 创建一个随机生成的标准正态分布数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 在同一个 ax 上绘制新的 DataFrame 的图，并获取绘图对象 ax
        ax = df.plot(legend=True, ax=ax)
        # 再次获取图例对象的标题
        leg_title = ax.legend_.get_title()
        # 检查图例标题的文本标签是否符合预期 "group,individual"
        _check_text_labels(leg_title, "group,individual")

        # 修改 DataFrame 的列名为 "new"
        df.columns.name = "new"
        # 在同一个 ax 上绘制新的 DataFrame 的图，并获取绘图对象 ax
        ax = df.plot(legend=False, ax=ax)
        # 再次获取图例对象的标题
        leg_title = ax.legend_.get_title()
        # 检查图例标题的文本标签是否符合预期 "group,individual"
        _check_text_labels(leg_title, "group,individual")

        # 在同一个 ax 上绘制新的 DataFrame 的图，并获取绘图对象 ax
        ax = df.plot(legend=True, ax=ax)
        # 再次获取图例对象的标题
        leg_title = ax.legend_.get_title()
        # 检查图例标题的文本标签是否符合预期 "new"
        _check_text_labels(leg_title, "new")

    # 使用参数化测试，测试不同的图形类型的图表生成，同时验证图例的可见性
    @pytest.mark.parametrize(
        "kind",
        [
            "line",
            "bar",
            "barh",
            pytest.param("kde", marks=td.skip_if_no("scipy")),
            "area",
            "hist",
        ],
    )
    def test_no_legend(self, kind):
        # 创建一个随机生成的浮点数（0到1之间）的 DataFrame，列名为 ["a", "b", "c"]
        df = DataFrame(np.random.default_rng(2).random((3, 3)), columns=["a", "b", "c"])
        # 根据指定的图形类型 kind 绘制 DataFrame 的图表，并设置图例不可见
        ax = df.plot(kind=kind, legend=False)
        # 检查图例标签的可见性是否符合预期，应为 False
        _check_legend_labels(ax, visible=False)

    # 测试在同一图中绘制不同列的数据，并验证每个数据列的标记类型
    def test_missing_markers_legend(self):
        # 创建一个随机生成的标准正态分布数据的 DataFrame，列名为 ["A", "B", "C"]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((8, 3)), columns=["A", "B", "C"]
        )
        # 在同一个 ax 上分别绘制三个数据列的数据，设置不同的标记和线条样式
        ax = df.plot(y=["A"], marker="x", linestyle="solid")
        df.plot(y=["B"], marker="o", linestyle="dotted", ax=ax)
        df.plot(y=["C"], marker="<", linestyle="dotted", ax=ax)

        # 检查图例的标签是否包含 ["A", "B", "C"]
        _check_legend_labels(ax, labels=["A", "B", "C"])
        # 检查图例的标记类型是否符合预期 ["x", "o", "<"]
        _check_legend_marker(ax, expected_markers=["x", "o", "<"])

    # 使用样式绘制数据，并验证图例的标签和标记类型
    def test_missing_markers_legend_using_style(self):
        # 创建一个包含四列数据的 DataFrame
        df = DataFrame(
            {
                "A": [1, 2, 3, 4, 5, 6],
                "B": [2, 4, 1, 3, 2, 4],
                "C": [3, 3, 2, 6, 4, 2],
                "X": [1, 2, 3, 4, 5, 6],
            }
        )

        # 创建一个新的图形和绘图对象 ax
        _, ax = mpl.pyplot.subplots()
        # 使用 "." 样式分别绘制 "A", "B", "C" 列的数据，并在同一个 ax 上显示
        for kind in "ABC":
            df.plot("X", kind, label=kind, ax=ax, style=".")

        # 检查图例的标签是否包含 ["A", "B", "C"]
        _check_legend_labels(ax, labels=["A", "B", "C"])
        # 检查图例的标记类型是否为预期的 [".", ".", "."]
        _check_legend_marker(ax, expected_markers=[".", ".", "."])
```