# `D:\src\scipysrc\pandas\pandas\tests\plotting\test_misc.py`

```
"""Test cases for misc plot functions"""

import os

import numpy as np
import pytest

import pandas.util._test_decorators as td

from pandas import (
    DataFrame,  # 导入 DataFrame 类
    Index,  # 导入 Index 类
    Series,  # 导入 Series 类
    Timestamp,  # 导入 Timestamp 类
    date_range,  # 导入 date_range 函数
    interval_range,  # 导入 interval_range 函数
    period_range,  # 导入 period_range 函数
    plotting,  # 导入 plotting 模块
    read_csv,  # 导入 read_csv 函数
)
import pandas._testing as tm
from pandas.tests.plotting.common import (
    _check_colors,  # 导入 _check_colors 函数
    _check_legend_labels,  # 导入 _check_legend_labels 函数
    _check_plot_works,  # 导入 _check_plot_works 函数
    _check_text_labels,  # 导入 _check_text_labels 函数
    _check_ticks_props,  # 导入 _check_ticks_props 函数
)

mpl = pytest.importorskip("matplotlib")  # 导入 matplotlib 模块
plt = pytest.importorskip("matplotlib.pyplot")  # 导入 matplotlib.pyplot 模块
cm = pytest.importorskip("matplotlib.cm")  # 导入 matplotlib.cm 模块

from pandas.plotting._matplotlib.style import get_standard_colors  # 导入 get_standard_colors 函数


@pytest.fixture
def iris(datapath) -> DataFrame:
    """
    The iris dataset as a DataFrame.
    """
    return read_csv(datapath("io", "data", "csv", "iris.csv"))  # 读取 iris 数据集并返回为 DataFrame


@td.skip_if_installed("matplotlib")
def test_import_error_message():
    # GH-19810
    df = DataFrame({"A": [1, 2]})  # 创建 DataFrame

    with pytest.raises(ImportError, match="matplotlib is required for plotting"):
        df.plot()  # 测试绘图时是否会引发 ImportError


def test_get_accessor_args():
    func = plotting._core.PlotAccessor._get_call_args  # 获取绘图参数的函数

    msg = "Called plot accessor for type list, expected Series or DataFrame"
    with pytest.raises(TypeError, match=msg):
        func(backend_name="", data=[], args=[], kwargs={})  # 测试传入列表时是否会引发 TypeError

    msg = "should not be called with positional arguments"
    with pytest.raises(TypeError, match=msg):
        func(backend_name="", data=Series(dtype=object), args=["line", None], kwargs={})  # 测试传入位置参数时是否会引发 TypeError

    x, y, kind, kwargs = func(
        backend_name="",
        data=DataFrame(),
        args=["x"],
        kwargs={"y": "y", "kind": "bar", "grid": False},
    )
    assert x == "x"
    assert y == "y"
    assert kind == "bar"
    assert kwargs == {"grid": False}

    x, y, kind, kwargs = func(
        backend_name="pandas.plotting._matplotlib",
        data=Series(dtype=object),
        args=[],
        kwargs={},
    )
    assert x is None
    assert y is None
    assert kind == "line"
    assert len(kwargs) == 24


@pytest.mark.parametrize("kind", plotting.PlotAccessor._all_kinds)
@pytest.mark.parametrize(
    "data", [DataFrame(np.arange(15).reshape(5, 3)), Series(range(5))]
)
@pytest.mark.parametrize(
    "index",
    [
        Index(range(5)),
        date_range("2020-01-01", periods=5),
        period_range("2020-01-01", periods=5),
    ],
)
def test_savefig(kind, data, index):
    fig, ax = plt.subplots()  # 创建图形和坐标轴
    data.index = index  # 设置数据的索引
    kwargs = {}
    if kind in ["hexbin", "scatter", "pie"]:
        if isinstance(data, Series):
            pytest.skip(f"{kind} not supported with Series")
        kwargs = {"x": 0, "y": 1}
    data.plot(kind=kind, ax=ax, **kwargs)  # 绘制数据图
    fig.savefig(os.devnull)  # 保存图形到空设备


class TestSeriesPlots:
    # 定义测试自相关图函数
    def test_autocorrelation_plot(self):
        # 创建一个包含浮点数的时间序列，从 "2020-01-01" 开始，持续 10 个时间点
        ser = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        # 确保在绘制图时不会产生用户警告
        with tm.assert_produces_warning(None):
            # 检查自相关图函数对于序列的绘图是否正常工作
            _check_plot_works(plotting.autocorrelation_plot, series=ser)
            _check_plot_works(plotting.autocorrelation_plot, series=ser.values)

            # 绘制自相关图并指定标签为 "Test"
            ax = plotting.autocorrelation_plot(ser, label="Test")
        # 检查绘制的图例标签是否为 ["Test"]
        _check_legend_labels(ax, labels=["Test"])

    # 使用参数化标记，测试滞后图函数
    @pytest.mark.parametrize("kwargs", [{}, {"lag": 5}])
    def test_lag_plot(self, kwargs):
        # 创建一个包含浮点数的时间序列，从 "2020-01-01" 开始，持续 10 个时间点
        ser = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        # 检查滞后图函数对于序列的绘图是否正常工作，并根据参数化标记传入的参数进行测试
        _check_plot_works(plotting.lag_plot, series=ser, **kwargs)

    # 定义测试自举法图函数
    def test_bootstrap_plot(self):
        # 创建一个包含浮点数的时间序列，从 "2020-01-01" 开始，持续 10 个时间点
        ser = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        # 检查自举法图函数对于序列的绘图是否正常工作，并指定图形大小为 10
        _check_plot_works(plotting.bootstrap_plot, series=ser, size=10)
class TestDataFramePlots:
    # 使用 pytest 标记参数化测试，参数为 pass_axis，取值为 False 和 True
    @pytest.mark.parametrize("pass_axis", [False, True])
    def test_scatter_matrix_axis(self, pass_axis):
        # 导入 scipy 库，如果导入失败则跳过该测试
        pytest.importorskip("scipy")
        # 赋值 scatter_matrix 变量为 plotting.scatter_matrix 函数
        scatter_matrix = plotting.scatter_matrix

        # 初始化 ax 变量为 None
        ax = None
        # 如果 pass_axis 为 True，则创建一个 3x3 的子图并将其赋给 ax
        if pass_axis:
            _, ax = mpl.pyplot.subplots(3, 3)

        # 创建一个包含随机标准正态分布数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)))

        # 在一个子图上绘制多个散点图的过程中，检查是否产生 UserWarning 警告，不检查堆栈层级
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            # 调用 _check_plot_works 函数，传入 scatter_matrix 函数、DataFrame df、range_padding=0.1、以及 ax 参数
            axes = _check_plot_works(
                scatter_matrix,
                frame=df,
                range_padding=0.1,
                ax=ax,
            )
        
        # 获取 axes[0][0] 子图的 y 轴主要刻度标签
        axes0_labels = axes[0][0].yaxis.get_majorticklabels()
        # 检查标签内容是否符合预期列表 ["-2", "-1", "0"]
        _check_text_labels(axes0_labels, expected)
        # 检查轴的属性设置是否符合预期，包括 xlabelsize=8、xrot=90、ylabelsize=8、yrot=0
        _check_ticks_props(axes, xlabelsize=8, xrot=90, ylabelsize=8, yrot=0)

    # 使用 pytest 标记参数化测试，参数为 pass_axis，取值为 False 和 True
    @pytest.mark.parametrize("pass_axis", [False, True])
    def test_scatter_matrix_axis_smaller(self, pass_axis):
        # 导入 scipy 库，如果导入失败则跳过该测试
        pytest.importorskip("scipy")
        # 赋值 scatter_matrix 变量为 plotting.scatter_matrix 函数
        scatter_matrix = plotting.scatter_matrix

        # 初始化 ax 变量为 None
        ax = None
        # 如果 pass_axis 为 True，则创建一个 3x3 的子图并将其赋给 ax
        if pass_axis:
            _, ax = mpl.pyplot.subplots(3, 3)

        # 创建一个包含随机标准正态分布数据的 DataFrame
        df = DataFrame(np.random.default_rng(11).standard_normal((10, 3)))
        # 修改 df 的第一列数据，使其值减去2后再除以3
        df[0] = (df[0] - 2) / 3

        # 在一个子图上绘制多个散点图的过程中，检查是否产生 UserWarning 警告，不检查堆栈层级
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            # 调用 _check_plot_works 函数，传入 scatter_matrix 函数、DataFrame df、range_padding=0.1、以及 ax 参数
            axes = _check_plot_works(
                scatter_matrix,
                frame=df,
                range_padding=0.1,
                ax=ax,
            )
        
        # 获取 axes[0][0] 子图的 y 轴主要刻度标签
        axes0_labels = axes[0][0].yaxis.get_majorticklabels()
        # 检查标签内容是否符合预期列表 ["-1.25", "-1.0", "-0.75", "-0.5"]
        _check_text_labels(axes0_labels, expected)
        # 检查轴的属性设置是否符合预期，包括 xlabelsize=8、xrot=90、ylabelsize=8、yrot=0
        _check_ticks_props(axes, xlabelsize=8, xrot=90, ylabelsize=8, yrot=0)

    # 使用 pytest 标记慢速测试
    @pytest.mark.slow
    def test_andrews_curves_no_warning(self, iris):
        # 确保调用 plotting.andrews_curves 函数时不会产生 UserWarning 警告
        with tm.assert_produces_warning(None):
            _check_plot_works(plotting.andrews_curves, frame=iris, class_column="Name")

    # 使用 pytest 标记慢速测试，同时使用两个参数化装饰器
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "linecolors",
        [
            ("#556270", "#4ECDC4", "#C7F464"),
            ["dodgerblue", "aquamarine", "seagreen"],
        ],
    )
    @pytest.mark.parametrize(
        "df",
        [
            "iris",
            DataFrame(
                {
                    "A": np.random.default_rng(2).standard_normal(10),
                    "B": np.random.default_rng(2).standard_normal(10),
                    "C": np.random.default_rng(2).standard_normal(10),
                    "Name": ["A"] * 10,
                }
            ),
        ],
    )
    @pytest.mark.slow
    # 定义测试函数，用于验证 andrews_curves 函数的线条颜色设置
    def test_andrews_curves_linecolors(self, request, df, linecolors):
        # 如果 df 是字符串，则从测试请求中获取相应的 fixture 值
        if isinstance(df, str):
            df = request.getfixturevalue(df)
        # 调用 _check_plot_works 函数验证 andrews_curves 绘图是否正常工作，并设置线条颜色
        ax = _check_plot_works(
            plotting.andrews_curves, frame=df, class_column="Name", color=linecolors
        )
        # 调用 _check_colors 函数验证绘图中前 10 条线条的颜色是否符合预期
        _check_colors(
            ax.get_lines()[:10], linecolors=linecolors, mapping=df["Name"][:10]
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "df",
        [
            "iris",
            DataFrame(
                {
                    "A": np.random.default_rng(2).standard_normal(10),
                    "B": np.random.default_rng(2).standard_normal(10),
                    "C": np.random.default_rng(2).standard_normal(10),
                    "Name": ["A"] * 10,
                }
            ),
        ],
    )
    # 定义测试函数，用于验证 andrews_curves 函数的 colormap 设置
    def test_andrews_curves_cmap(self, request, df):
        # 如果 df 是字符串，则从测试请求中获取相应的 fixture 值
        if isinstance(df, str):
            df = request.getfixturevalue(df)
        # 生成一组颜色映射，数量与 df 中的不同类别数相同
        cmaps = [cm.jet(n) for n in np.linspace(0, 1, df["Name"].nunique())]
        # 调用 _check_plot_works 函数验证 andrews_curves 绘图是否正常工作，并设置 colormap
        ax = _check_plot_works(
            plotting.andrews_curves, frame=df, class_column="Name", color=cmaps
        )
        # 调用 _check_colors 函数验证绘图中前 10 条线条的颜色是否符合预期
        _check_colors(ax.get_lines()[:10], linecolors=cmaps, mapping=df["Name"][:10])

    @pytest.mark.slow
    # 定义测试函数，用于验证 andrews_curves 函数的处理能力
    def test_andrews_curves_handle(self):
        # 定义颜色列表
        colors = ["b", "g", "r"]
        # 创建 DataFrame 对象，用于测试
        df = DataFrame({"A": [1, 2, 3], "B": [1, 2, 3], "C": [1, 2, 3], "Name": colors})
        # 调用 andrews_curves 函数绘制图形，并设置线条颜色
        ax = plotting.andrews_curves(df, "Name", color=colors)
        # 获取图例中的句柄和标签，并验证其颜色是否符合预期
        handles, _ = ax.get_legend_handles_labels()
        _check_colors(handles, linecolors=colors)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "color",
        [("#556270", "#4ECDC4", "#C7F464"), ["dodgerblue", "aquamarine", "seagreen"]],
    )
    # 定义测试函数，用于验证 parallel_coordinates 函数的颜色设置
    def test_parallel_coordinates_colors(self, iris, color):
        # 使用 iris 数据集作为测试数据
        df = iris

        # 调用 _check_plot_works 函数验证 parallel_coordinates 绘图是否正常工作，并设置线条颜色
        ax = _check_plot_works(
            plotting.parallel_coordinates, frame=df, class_column="Name", color=color
        )
        # 调用 _check_colors 函数验证绘图中前 10 条线条的颜色是否符合预期
        _check_colors(ax.get_lines()[:10], linecolors=color, mapping=df["Name"][:10])

    @pytest.mark.slow
    # 定义测试函数，用于验证 parallel_coordinates 函数的 colormap 设置
    def test_parallel_coordinates_cmap(self, iris):
        # 使用 iris 数据集作为测试数据
        df = iris

        # 调用 _check_plot_works 函数验证 parallel_coordinates 绘图是否正常工作，并设置 colormap
        ax = _check_plot_works(
            plotting.parallel_coordinates,
            frame=df,
            class_column="Name",
            colormap=cm.jet,
        )
        # 生成一组颜色映射，数量与 df 中的不同类别数相同
        cmaps = [mpl.cm.jet(n) for n in np.linspace(0, 1, df["Name"].nunique())]
        # 调用 _check_colors 函数验证绘图中前 10 条线条的颜色是否符合预期
        _check_colors(ax.get_lines()[:10], linecolors=cmaps, mapping=df["Name"][:10])

    @pytest.mark.slow
    # 定义测试函数，用于验证 parallel_coordinates 函数在不同参数下的线条数量变化
    def test_parallel_coordinates_line_diff(self, iris):
        # 使用 iris 数据集作为测试数据
        df = iris

        # 调用 _check_plot_works 函数验证 parallel_coordinates 绘图是否正常工作，并获取初始线条数量
        ax = _check_plot_works(
            plotting.parallel_coordinates, frame=df, class_column="Name"
        )
        nlines = len(ax.get_lines())
        nxticks = len(ax.xaxis.get_ticklabels())

        # 调用 _check_plot_works 函数验证 parallel_coordinates 绘图是否正常工作，关闭垂直线条显示，并验证线条数量是否符合预期
        ax = _check_plot_works(
            plotting.parallel_coordinates, frame=df, class_column="Name", axvlines=False
        )
        assert len(ax.get_lines()) == (nlines - nxticks)
    # 测试函数，用于验证并处理并行坐标图的生成
    def test_parallel_coordinates_handles(self, iris):
        # 将 iris 数据框赋值给 df
        df = iris
        # 定义颜色列表
        colors = ["b", "g", "r"]
        # 创建新的 DataFrame 包含列 "A", "B", "C", "Name"，并使用 colors 作为 "Name" 列的数据
        df = DataFrame({"A": [1, 2, 3], "B": [1, 2, 3], "C": [1, 2, 3], "Name": colors})
        # 调用 plotting.parallel_coordinates 函数生成并行坐标图，颜色根据 "Name" 列指定
        ax = plotting.parallel_coordinates(df, "Name", color=colors)
        # 获取图例的句柄和标签
        handles, _ = ax.get_legend_handles_labels()
        # 调用 _check_colors 函数验证颜色是否符合预期
        _check_colors(handles, linecolors=colors)

    # 标记，忽略 "Attempting to set:UserWarning" 警告
    @pytest.mark.filterwarnings("ignore:Attempting to set:UserWarning")
    # 测试函数，验证带有排序标签的并行坐标图的生成
    def test_parallel_coordinates_with_sorted_labels(self):
        # 创建包含 "feat" 和 "class" 列的 DataFrame
        df = DataFrame(
            {
                "feat": list(range(30)),
                "class": [2 for _ in range(10)] + [3 for _ in range(10)] + [1 for _ in range(10)],
            }
        )
        # 调用 plotting.parallel_coordinates 函数生成并行坐标图，根据 "class" 列排序标签
        ax = plotting.parallel_coordinates(df, "class", sort_labels=True)
        # 获取所有多边线和标签
        polylines, labels = ax.get_legend_handles_labels()
        # 将颜色和标签组成元组，然后按标签排序
        color_label_tuples = zip([polyline.get_color() for polyline in polylines], labels)
        ordered_color_label_tuples = sorted(color_label_tuples, key=lambda x: x[1])
        # 将相邻的颜色和标签组成元组，验证它们按顺序增加
        prev_next_tuples = zip(list(ordered_color_label_tuples[0:-1]), list(ordered_color_label_tuples[1:]))
        for prev, nxt in prev_next_tuples:
            # 断言前一个元组的标签小于后一个元组的标签，并且颜色也按顺序增加
            assert prev[1] < nxt[1] and prev[0] < nxt[0]

    # 测试函数，验证 radviz 图生成时不产生 UserWarning
    def test_radviz_no_warning(self, iris):
        # 确保在绘制图时不会产生 UserWarning
        with tm.assert_produces_warning(None):
            # 调用 _check_plot_works 函数验证 plotting.radviz 的工作情况，使用 iris 数据框，"Name" 列作为分类列
            _check_plot_works(plotting.radviz, frame=iris, class_column="Name")

    # 参数化测试，验证 radviz 图生成时颜色的不同设置
    @pytest.mark.parametrize(
        "color",
        [("#556270", "#4ECDC4", "#C7F464"), ["dodgerblue", "aquamarine", "seagreen"]],
    )
    def test_radviz_color(self, iris, color):
        # 将 iris 数据框赋值给 df
        df = iris
        # 调用 _check_plot_works 函数验证 plotting.radviz 的工作情况，使用 iris 数据框，"Name" 列作为分类列，颜色参数化设置
        ax = _check_plot_works(plotting.radviz, frame=df, class_column="Name", color=color)
        # 获取前 20 个图形对象中非空标签的图形
        patches = [p for p in ax.patches[:20] if p.get_label() != ""]
        # 调用 _check_colors 函数验证颜色是否符合预期
        _check_colors(patches[:10], facecolors=color, mapping=df["Name"][:10])

    # 测试函数，验证 radviz 图生成时使用 colormap 设置颜色
    def test_radviz_color_cmap(self, iris):
        # 将 iris 数据框赋值给 df
        df = iris
        # 调用 _check_plot_works 函数验证 plotting.radviz 的工作情况，使用 iris 数据框，"Name" 列作为分类列，使用 colormap 设置颜色
        ax = _check_plot_works(plotting.radviz, frame=df, class_column="Name", colormap=cm.jet)
        # 根据不同类别数量生成颜色映射
        cmaps = [mpl.cm.jet(n) for n in np.linspace(0, 1, df["Name"].nunique())]
        # 获取前 20 个图形对象中非空标签的图形
        patches = [p for p in ax.patches[:20] if p.get_label() != ""]
        # 调用 _check_colors 函数验证颜色是否符合预期
        _check_colors(patches, facecolors=cmaps, mapping=df["Name"][:10])

    # 测试函数，验证 radviz 图生成时颜色句柄的处理
    def test_radviz_colors_handles(self):
        # 定义颜色列表
        colors = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]]
        # 创建新的 DataFrame 包含列 "A", "B", "C", "Name"，并使用颜色列表作为 "Name" 列的数据
        df = DataFrame({"A": [1, 2, 3], "B": [2, 1, 3], "C": [3, 2, 1], "Name": ["b", "g", "r"]})
        # 调用 plotting.radviz 函数生成 radviz 图，颜色根据 colors 参数指定
        ax = plotting.radviz(df, "Name", color=colors)
        # 获取图例的句柄和标签
        handles, _ = ax.get_legend_handles_labels()
        # 调用 _check_colors 函数验证颜色是否符合预期
        _check_colors(handles, facecolors=colors)
    def test_subplot_titles(self, iris):
        # 从 iris 数据集中删除 "Name" 列，并取前五行数据
        df = iris.drop("Name", axis=1).head()
        # 使用列名作为子图的标题
        title = list(df.columns)

        # 当 len(title) == len(df) 的情况
        plot = df.plot(subplots=True, title=title)
        # 断言子图的标题与预期的标题相符
        assert [p.get_title() for p in plot] == title

    def test_subplot_titles_too_much(self, iris):
        # 从 iris 数据集中删除 "Name" 列，并取前五行数据
        df = iris.drop("Name", axis=1).head()
        # 使用列名作为子图的标题
        title = list(df.columns)
        # 当 len(title) > len(df) 的情况
        msg = (
            "The length of `title` must equal the number of columns if "
            "using `title` of type `list` and `subplots=True`"
        )
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, title=title + ["kittens > puppies"])

    def test_subplot_titles_too_little(self, iris):
        # 从 iris 数据集中删除 "Name" 列，并取前五行数据
        df = iris.drop("Name", axis=1).head()
        # 使用列名作为子图的标题
        title = list(df.columns)
        # 当 len(title) < len(df) 的情况
        msg = (
            "The length of `title` must equal the number of columns if "
            "using `title` of type `list` and `subplots=True`"
        )
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, title=title[:2])

    def test_subplot_titles_subplots_false(self, iris):
        # 从 iris 数据集中删除 "Name" 列，并取前五行数据
        df = iris.drop("Name", axis=1).head()
        # 使用列名作为子图的标题
        title = list(df.columns)
        # 当 subplots=False 且 title 的类型为 list 的情况
        msg = (
            "Using `title` of type `list` is not supported unless "
            "`subplots=True` is passed"
        )
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=False, title=title)

    def test_subplot_titles_numeric_square_layout(self, iris):
        # 从 iris 数据集中删除 "Name" 列，并取前五行数据
        df = iris.drop("Name", axis=1).head()
        # 使用列名作为子图的标题
        title = list(df.columns)
        # 当 df 有三个数值列，但布局为 (2,2) 的情况
        plot = df.drop("SepalWidth", axis=1).plot(
            subplots=True, layout=(2, 2), title=title[:-1]
        )
        # 获取所有子图的标题
        title_list = [ax.get_title() for sublist in plot for ax in sublist]
        # 断言子图的标题与预期的标题相符
        assert title_list == title[:3] + [""]
    def test_get_standard_colors_default_num_colors(self):
        # 确保使用默认的 color_type 返回指定数量的颜色
        color1 = get_standard_colors(1, color_type="default")
        color2 = get_standard_colors(9, color_type="default")
        color3 = get_standard_colors(20, color_type="default")
        assert len(color1) == 1  # 断言：返回的颜色列表长度为1
        assert len(color2) == 9  # 断言：返回的颜色列表长度为9
        assert len(color3) == 20  # 断言：返回的颜色列表长度为20

    def test_plot_single_color(self):
        # 来自 #20585 的示例。所有的三个柱子应该有相同的颜色
        df = DataFrame(
            {
                "account-start": ["2017-02-03", "2017-03-03", "2017-01-01"],
                "client": ["Alice Anders", "Bob Baker", "Charlie Chaplin"],
                "balance": [-1432.32, 10.43, 30000.00],
                "db-id": [1234, 2424, 251],
                "proxy-id": [525, 1525, 2542],
                "rank": [52, 525, 32],
            }
        )
        ax = df.client.value_counts().plot.bar()  # 绘制柱状图
        colors = [rect.get_facecolor() for rect in ax.get_children()[0:3]]  # 获取柱子的颜色
        assert all(color == colors[0] for color in colors)  # 断言：所有柱子的颜色相同

    def test_get_standard_colors_no_appending(self):
        # GH20726

        # 确保不添加更多的颜色，以便 matplotlib 可以正确循环使用颜色
        color_before = mpl.cm.gnuplot(range(5))  # 获取 gnuplot 调色板的前5个颜色
        color_after = get_standard_colors(1, color=color_before)  # 获取一个标准颜色列表
        assert len(color_after) == len(color_before)  # 断言：返回的颜色列表长度与输入相同

        df = DataFrame(
            np.random.default_rng(2).standard_normal((48, 4)), columns=list("ABCD")
        )

        color_list = mpl.cm.gnuplot(np.linspace(0, 1, 16))  # 生成16种颜色的列表
        p = df.A.plot.bar(figsize=(16, 7), color=color_list)  # 绘制 A 列的柱状图
        assert p.patches[1].get_facecolor() == p.patches[17].get_facecolor()  # 断言：两个柱子的颜色相同

    @pytest.mark.parametrize("kind", ["bar", "line"])
    def test_dictionary_color(self, kind):
        # issue-8193
        # 测试使用字典格式指定图表颜色

        data_files = ["a", "b"]

        expected = [(0.5, 0.24, 0.6), (0.3, 0.7, 0.7)]

        df1 = DataFrame(np.random.default_rng(2).random((2, 2)), columns=data_files)
        dic_color = {"b": (0.3, 0.7, 0.7), "a": (0.5, 0.24, 0.6)}

        ax = df1.plot(kind=kind, color=dic_color)  # 根据字典颜色绘制指定类型的图表
        if kind == "bar":
            colors = [rect.get_facecolor()[0:-1] for rect in ax.get_children()[0:3:2]]
        else:
            colors = [rect.get_color() for rect in ax.get_lines()[0:2]]
        assert all(color == expected[index] for index, color in enumerate(colors))  # 断言：绘制的颜色符合预期

    def test_bar_plot(self):
        # GH38947
        # 测试包含字符串和整数索引的柱状图

        expected = [mpl.text.Text(0, 0, "0"), mpl.text.Text(1, 0, "Total")]

        df = DataFrame(
            {
                "a": [1, 2],
            },
            index=Index([0, "Total"]),
        )
        plot_bar = df.plot.bar()  # 绘制柱状图
        assert all(
            (a.get_text() == b.get_text())
            for a, b in zip(plot_bar.get_xticklabels(), expected)
        )  # 断言：柱状图的 x 轴标签文本符合预期
    # 定义测试方法：test_barh_plot_labels_mixed_integer_string
    def test_barh_plot_labels_mixed_integer_string(self):
        # GH39126
        # 对于同时包含字符串和整数的柱状图进行测试
        # 创建包含两行数据的数据框
        df = DataFrame([{"word": 1, "value": 0}, {"word": "knowledge", "value": 2}])
        # 绘制水平柱状图，并指定 x 轴为 "word"，不显示图例
        plot_barh = df.plot.barh(x="word", legend=None)
        # 预期的 y 轴刻度标签列表
        expected_yticklabels = [
            mpl.text.Text(0, 0, "1"),
            mpl.text.Text(0, 1, "knowledge"),
        ]
        # 断言实际 y 轴刻度标签与预期是否一致
        assert all(
            actual.get_text() == expected.get_text()
            for actual, expected in zip(
                plot_barh.get_yticklabels(), expected_yticklabels
            )
        )

    # 定义测试方法：test_has_externally_shared_axis_x_axis
    def test_has_externally_shared_axis_x_axis(self):
        # GH33819
        # 测试 _has_externally_shared_axis() 是否对 x 轴有效
        func = plotting._matplotlib.tools._has_externally_shared_axis

        # 创建一个新的图形对象
        fig = mpl.pyplot.figure()
        # 在图形上创建 2x4 的子图数组
        plots = fig.subplots(2, 4)

        # 为第一列和第三列创建 *外部* 共享的 x 轴
        plots[0][0] = fig.add_subplot(231, sharex=plots[1][0])
        plots[0][2] = fig.add_subplot(233, sharex=plots[1][2])

        # 对第二列和第三列创建 *内部* 共享的 x 轴
        plots[0][1].twinx()
        plots[0][2].twinx()

        # 第一列仅外部共享
        # 第二列仅内部共享
        # 第三列同时内部和外部共享
        # 第四列都不共享
        assert func(plots[0][0], "x")
        assert not func(plots[0][1], "x")
        assert func(plots[0][2], "x")
        assert not func(plots[0][3], "x")

    # 定义测试方法：test_has_externally_shared_axis_y_axis
    def test_has_externally_shared_axis_y_axis(self):
        # GH33819
        # 测试 _has_externally_shared_axis() 是否对 y 轴有效
        func = plotting._matplotlib.tools._has_externally_shared_axis

        # 创建一个新的图形对象
        fig = mpl.pyplot.figure()
        # 在图形上创建 4x2 的子图数组
        plots = fig.subplots(4, 2)

        # 为第一行和第三行创建 *外部* 共享的 y 轴
        plots[0][0] = fig.add_subplot(321, sharey=plots[0][1])
        plots[2][0] = fig.add_subplot(325, sharey=plots[2][1])

        # 对第二行和第三行创建 *内部* 共享的 y 轴
        plots[1][0].twiny()
        plots[2][0].twiny()

        # 第一行仅外部共享
        # 第二行仅内部共享
        # 第三行同时内部和外部共享
        # 第四行都不共享
        assert func(plots[0][0], "y")
        assert not func(plots[1][0], "y")
        assert func(plots[2][0], "y")
        assert not func(plots[3][0], "y")
    def test_has_externally_shared_axis_invalid_compare_axis(self):
        # GH33819
        # 测试当传递一个无效的 compare_axis 参数时，_has_externally_shared_axis() 是否会引发异常
        func = plotting._matplotlib.tools._has_externally_shared_axis

        fig = mpl.pyplot.figure()
        # 创建一个 4x2 的子图数组
        plots = fig.subplots(4, 2)

        # 创建任意的坐标轴
        plots[0][0] = fig.add_subplot(321, sharey=plots[0][1])

        # 检查当 compare_axis 参数为无效值时，是否会触发预期的异常
        msg = "needs 'x' or 'y' as a second parameter"
        with pytest.raises(ValueError, match=msg):
            func(plots[0][0], "z")

    def test_externally_shared_axes(self):
        # 来自 GH33819 的示例
        # 创建数据
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
            }
        )

        # 创建图表
        fig = mpl.pyplot.figure()
        # 创建一个 2x3 的子图数组
        plots = fig.subplots(2, 3)

        # 创建 *externally* shared axes
        plots[0][0] = fig.add_subplot(231, sharex=plots[1][0])
        # 注意：没有 plots[0][1]，这是只有双轴的情况
        plots[0][2] = fig.add_subplot(233, sharex=plots[1][2])

        # 创建 *internally* shared axes
        # 注意：没有 plots[0][0]，这是仅外部共享的情况
        twin_ax1 = plots[0][1].twinx()
        twin_ax2 = plots[0][2].twinx()

        # 在主要坐标轴上绘制数据
        df["a"].plot(ax=plots[0][0], title="External share only").set_xlabel(
            "this label should never be visible"
        )
        df["a"].plot(ax=plots[1][0])

        df["a"].plot(ax=plots[0][1], title="Internal share (twin) only").set_xlabel(
            "this label should always be visible"
        )
        df["a"].plot(ax=plots[1][1])

        df["a"].plot(ax=plots[0][2], title="Both").set_xlabel(
            "this label should never be visible"
        )
        df["a"].plot(ax=plots[1][2])

        # 在双轴上绘制数据
        df["b"].plot(ax=twin_ax1, color="green")
        df["b"].plot(ax=twin_ax2, color="yellow")

        assert not plots[0][0].xaxis.get_label().get_visible()
        assert plots[0][1].xaxis.get_label().get_visible()
        assert not plots[0][2].xaxis.get_label().get_visible()

    def test_plot_bar_axis_units_timestamp_conversion(self):
        # GH 38736
        # 确保第二个图的字符串 x 轴不会因为第一个图的轴数据而被转换为日期时间
        df = DataFrame(
            [1.0],
            index=[Timestamp("2022-02-22 22:22:22")],
        )
        _check_plot_works(df.plot)
        s = Series({"A": 1.0})
        _check_plot_works(s.plot.bar)
    def test_bar_plt_xaxis_intervalrange(self):
        # GH 38969
        # GH 38969 is a reference to a GitHub issue number.
        # Ensure IntervalIndex x-axis produces a bar plot as expected
        # 确保使用 IntervalIndex 的 x 轴生成预期的柱状图
        expected = [mpl.text.Text(0, 0, "([0, 1],)"), mpl.text.Text(1, 0, "([1, 2],)")]
        # Create a Series with values [1, 2] and an index defined by interval_range(0, 2, closed="both")
        # 使用 interval_range(0, 2, closed="both") 定义的索引创建 Series，其值为 [1, 2]
        s = Series(
            [1, 2],
            index=[interval_range(0, 2, closed="both")],
        )
        # Check that the plotting function works correctly for bar plots
        # 检查柱状图绘制函数的正确性
        _check_plot_works(s.plot.bar)
        # Assert that all x-axis tick labels match the expected labels
        # 断言所有 x 轴刻度标签与预期标签相匹配
        assert all(
            (a.get_text() == b.get_text())
            for a, b in zip(s.plot.bar().get_xticklabels(), expected)
        )
```