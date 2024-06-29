# `D:\src\scipysrc\pandas\pandas\tests\plotting\test_hist_method.py`

```
"""Test cases for .hist method"""

# 导入正则表达式模块
import re

# 导入 numpy 库，并用 np 别名表示
import numpy as np
# 导入 pytest 库，用于测试
import pytest

# 导入 pandas 库中的相关模块和函数
from pandas import (
    DataFrame,      # 数据框架类
    Index,          # 索引类
    Series,         # 系列类
    date_range,     # 日期范围生成函数
    to_datetime,    # 将对象转换为日期时间类型的函数
)

# 导入 pandas 内部的测试工具模块
import pandas._testing as tm
# 导入 pandas 绘图测试中的一些共用函数
from pandas.tests.plotting.common import (
    _check_ax_scales,           # 检查坐标轴比例尺函数
    _check_axes_shape,          # 检查坐标轴形状函数
    _check_colors,              # 检查颜色函数
    _check_legend_labels,       # 检查图例标签函数
    _check_patches_all_filled,  # 检查所有补丁填充函数
    _check_plot_works,          # 检查绘图是否正常工作函数
    _check_text_labels,         # 检查文本标签函数
    _check_ticks_props,         # 检查刻度属性函数
    get_x_axis,                 # 获取 x 轴函数
    get_y_axis,                 # 获取 y 轴函数
)

# 导入 matplotlib 的 pytest 模块
mpl = pytest.importorskip("matplotlib")
# 导入 matplotlib.pyplot 模块并用 plt 别名表示
plt = pytest.importorskip("matplotlib.pyplot")

# 导入 pandas 绘图模块中的 _grouped_hist 函数
from pandas.plotting._matplotlib.hist import _grouped_hist


@pytest.fixture
def ts():
    return Series(
        np.arange(30, dtype=np.float64),
        index=date_range("2020-01-01", periods=30, freq="B"),
        name="ts",
    )


class TestSeriesPlots:
    @pytest.mark.parametrize("kwargs", [{}, {"grid": False}, {"figsize": (8, 10)}])
    def test_hist_legacy_kwargs(self, ts, kwargs):
        _check_plot_works(ts.hist, **kwargs)

    @pytest.mark.parametrize("kwargs", [{}, {"bins": 5}])
    def test_hist_legacy_kwargs_warning(self, ts, kwargs):
        # _check_plot_works adds an ax so catch warning. see GH #13188
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(ts.hist, by=ts.index.month, **kwargs)

    def test_hist_legacy_ax(self, ts):
        fig, ax = mpl.pyplot.subplots(1, 1)
        _check_plot_works(ts.hist, ax=ax, default_axes=True)

    def test_hist_legacy_ax_and_fig(self, ts):
        fig, ax = mpl.pyplot.subplots(1, 1)
        _check_plot_works(ts.hist, ax=ax, figure=fig, default_axes=True)

    def test_hist_legacy_fig(self, ts):
        fig, _ = mpl.pyplot.subplots(1, 1)
        _check_plot_works(ts.hist, figure=fig, default_axes=True)

    def test_hist_legacy_multi_ax(self, ts):
        fig, (ax1, ax2) = mpl.pyplot.subplots(1, 2)
        _check_plot_works(ts.hist, figure=fig, ax=ax1, default_axes=True)
        _check_plot_works(ts.hist, figure=fig, ax=ax2, default_axes=True)

    def test_hist_legacy_by_fig_error(self, ts):
        fig, _ = mpl.pyplot.subplots(1, 1)
        msg = (
            "Cannot pass 'figure' when using the 'by' argument, since a new 'Figure' "
            "instance will be created"
        )
        with pytest.raises(ValueError, match=msg):
            ts.hist(by=ts.index, figure=fig)

    def test_hist_bins_legacy(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        ax = df.hist(bins=2)[0][0]
        assert len(ax.patches) == 2

    def test_hist_layout(self, hist_df):
        df = hist_df
        msg = "The 'layout' keyword is not supported when 'by' is None"
        with pytest.raises(ValueError, match=msg):
            df.height.hist(layout=(1, 1))

        with pytest.raises(ValueError, match=msg):
            df.height.hist(layout=[1, 1])

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "by, layout, axes_num, res_layout",
        [
            ["gender", (2, 1), 2, (2, 1)],  # 参数化测试：按性别分组，指定布局为 (2, 1)，预期轴数为 2，预期布局为 (2, 1)
            ["gender", (3, -1), 2, (3, 1)],  # 参数化测试：按性别分组，指定布局为 (3, -1)，预期轴数为 2，预期布局为 (3, 1)
            ["category", (4, 1), 4, (4, 1)],  # 参数化测试：按类别分组，指定布局为 (4, 1)，预期轴数为 4，预期布局为 (4, 1)
            ["category", (2, -1), 4, (2, 2)],  # 参数化测试：按类别分组，指定布局为 (2, -1)，预期轴数为 4，预期布局为 (2, 2)
            ["category", (3, -1), 4, (3, 2)],  # 参数化测试：按类别分组，指定布局为 (3, -1)，预期轴数为 4，预期布局为 (3, 2)
            ["category", (-1, 4), 4, (1, 4)],  # 参数化测试：按类别分组，指定布局为 (-1, 4)，预期轴数为 4，预期布局为 (1, 4)
            ["classroom", (2, 2), 3, (2, 2)],  # 参数化测试：按教室分组，指定布局为 (2, 2)，预期轴数为 3，预期布局为 (2, 2)
        ],
    )
    def test_hist_layout_with_by(self, hist_df, by, layout, axes_num, res_layout):
        df = hist_df

        # _check_plot_works 在方法调用中添加了 `ax` 参数
        # 因此会收到一个有关清除轴的警告，尽管我们没有明确传递轴，参见 GH #13188
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            axes = _check_plot_works(df.height.hist, by=getattr(df, by), layout=layout)
        _check_axes_shape(axes, axes_num=axes_num, layout=res_layout)  # 检查轴的形状是否符合预期

    def test_hist_layout_with_by_shape(self, hist_df):
        df = hist_df

        axes = df.height.hist(by=df.category, layout=(4, 2), figsize=(12, 7))  # 绘制按类别分组的直方图，指定布局和图表尺寸
        _check_axes_shape(axes, axes_num=4, layout=(4, 2), figsize=(12, 7))  # 检查轴的形状、布局和图表尺寸是否符合预期

    def test_hist_no_overlap(self):
        x = Series(np.random.default_rng(2).standard_normal(2))
        y = Series(np.random.default_rng(2).standard_normal(2))
        plt.subplot(121)  # 创建第一个子图
        x.hist()  # 在第一个子图上绘制 x 的直方图
        plt.subplot(122)  # 创建第二个子图
        y.hist()  # 在第二个子图上绘制 y 的直方图
        fig = plt.gcf()  # 获取当前图形对象
        axes = fig.axes  # 获取图形对象的所有轴
        assert len(axes) == 2  # 断言：轴的数量应为 2

    def test_hist_by_no_extra_plots(self, hist_df):
        df = hist_df
        df.height.hist(by=df.gender)  # 绘制按性别分组的直方图
        assert len(mpl.pyplot.get_fignums()) == 1  # 断言：图形数量应为 1

    def test_plot_fails_when_ax_differs_from_figure(self, ts):
        fig1 = plt.figure(1)  # 创建图形对象 1
        fig2 = plt.figure(2)  # 创建图形对象 2
        ax1 = fig1.add_subplot(111)  # 在图形对象 1 上创建子图
        msg = "passed axis not bound to passed figure"
        with pytest.raises(AssertionError, match=msg):
            ts.hist(ax=ax1, figure=fig2)  # 尝试在不同的图形对象上绘制直方图，预期引发断言错误

    @pytest.mark.parametrize(
        "histtype, expected",
        [
            ("bar", True),         # 参数化测试：histtype 为 "bar"，预期为 True
            ("barstacked", True),  # 参数化测试：histtype 为 "barstacked"，预期为 True
            ("step", False),       # 参数化测试：histtype 为 "step"，预期为 False
            ("stepfilled", True),  # 参数化测试：histtype 为 "stepfilled"，预期为 True
        ],
    )
    def test_histtype_argument(self, histtype, expected):
        # GH23992 验证 histtype 参数的功能
        ser = Series(np.random.default_rng(2).integers(1, 10))
        ax = ser.hist(histtype=histtype)  # 绘制直方图，指定 histtype 参数
        _check_patches_all_filled(ax, filled=expected)  # 检查图中的填充物是否符合预期

    @pytest.mark.parametrize(
        "by, expected_axes_num, expected_layout", [(None, 1, (1, 1)), ("b", 2, (1, 2))]
    )
    def test_hist_with_legend(self, by, expected_axes_num, expected_layout):
        # GH 6279 - Series histogram can have a legend
        # 创建一个包含重复索引的 Series 对象，用随机正态分布填充数据，设置名称为 "a"，索引名为 "b"
        index = 5 * ["1"] + 5 * ["2"]
        s = Series(np.random.default_rng(2).standard_normal(10), index=index, name="a")
        s.index.name = "b"

        # 调用 _check_plot_works 函数，验证 s.hist 方法绘制直方图时默认使用子图和图例
        axes = _check_plot_works(s.hist, default_axes=True, legend=True, by=by)
        # 验证生成的子图数量和期望的数量是否一致
        _check_axes_shape(axes, axes_num=expected_axes_num, layout=expected_layout)
        # 验证图例标签是否正确设置为 "a"
        _check_legend_labels(axes, "a")

    @pytest.mark.parametrize("by", [None, "b"])
    def test_hist_with_legend_raises(self, by):
        # GH 6279 - Series histogram with legend and label raises
        # 创建一个包含重复索引的 Series 对象，用随机正态分布填充数据，设置名称为 "a"，索引名为 "b"
        index = 5 * ["1"] + 5 * ["2"]
        s = Series(np.random.default_rng(2).standard_normal(10), index=index, name="a")
        s.index.name = "b"

        # 使用 pytest 的断言检查 s.hist 方法同时使用图例和标签时是否引发 ValueError 异常
        with pytest.raises(ValueError, match="Cannot use both legend and label"):
            s.hist(legend=True, by=by, label="c")

    def test_hist_kwargs(self, ts):
        _, ax = mpl.pyplot.subplots()
        # 绘制 ts 的直方图，设置箱数为 5，并将结果保存到 ax 中
        ax = ts.plot.hist(bins=5, ax=ax)
        # 断言生成的图形中有 5 个矩形条
        assert len(ax.patches) == 5
        # 验证 y 轴标签是否设置为 "Frequency"
        _check_text_labels(ax.yaxis.get_label(), "Frequency")

    def test_hist_kwargs_horizontal(self, ts):
        _, ax = mpl.pyplot.subplots()
        # 绘制水平方向的 ts 的直方图，设置箱数为 5，并将结果保存到 ax 中
        ax = ts.plot.hist(bins=5, ax=ax)
        ax = ts.plot.hist(orientation="horizontal", ax=ax)
        # 验证 x 轴标签是否设置为 "Frequency"
        _check_text_labels(ax.xaxis.get_label(), "Frequency")

    def test_hist_kwargs_align(self, ts):
        _, ax = mpl.pyplot.subplots()
        # 绘制 ts 的直方图，设置箱数为 5，对齐方式为左对齐，堆叠方式为 True，并将结果保存到 ax 中
        ax = ts.plot.hist(bins=5, ax=ax)
        ax = ts.plot.hist(align="left", stacked=True, ax=ax)

    @pytest.mark.xfail(reason="Api changed in 3.6.0")
    def test_hist_kde(self, ts):
        pytest.importorskip("scipy")
        _, ax = mpl.pyplot.subplots()
        # 绘制 ts 的核密度估计图，设置 y 轴为对数轴，并将结果保存到 ax 中
        ax = ts.plot.hist(logy=True, ax=ax)
        # 验证 y 轴的刻度标签是否为空白
        _check_ax_scales(ax, yaxis="log")
        xlabels = ax.get_xticklabels()
        # 验证 x 轴的刻度标签是否为空白
        _check_text_labels(xlabels, [""] * len(xlabels))
        ylabels = ax.get_yticklabels()
        # 验证 y 轴的刻度标签是否为空白
        _check_text_labels(ylabels, [""] * len(ylabels))

    def test_hist_kde_plot_works(self, ts):
        pytest.importorskip("scipy")
        # 调用 _check_plot_works 函数，验证 ts.plot.kde 方法是否正常工作
        _check_plot_works(ts.plot.kde)

    def test_hist_kde_density_works(self, ts):
        pytest.importorskip("scipy")
        # 调用 _check_plot_works 函数，验证 ts.plot.density 方法是否正常工作
        _check_plot_works(ts.plot.density)

    @pytest.mark.xfail(reason="Api changed in 3.6.0")
    def test_hist_kde_logy(self, ts):
        pytest.importorskip("scipy")
        _, ax = mpl.pyplot.subplots()
        # 绘制 ts 的核密度估计图，设置 y 轴为对数轴，并将结果保存到 ax 中
        ax = ts.plot.kde(logy=True, ax=ax)
        # 验证 y 轴的刻度标签是否为空白
        _check_ax_scales(ax, yaxis="log")
        xlabels = ax.get_xticklabels()
        # 验证 x 轴的刻度标签是否为空白
        _check_text_labels(xlabels, [""] * len(xlabels))
        ylabels = ax.get_yticklabels()
        # 验证 y 轴的刻度标签是否为空白
        _check_text_labels(ylabels, [""] * len(ylabels))
    # 测试直方图和核密度估计的颜色和柱子数
    def test_hist_kde_color_bins(self, ts):
        # 导入 scipy 库，如果失败则跳过这个测试
        pytest.importorskip("scipy")
        # 创建一个新的图表并获取轴对象
        _, ax = mpl.pyplot.subplots()
        # 绘制时间序列 ts 的直方图，y 轴使用对数尺度，10个柱子，颜色为蓝色，将图形绘制在已有的轴上
        ax = ts.plot.hist(logy=True, bins=10, color="b", ax=ax)
        # 检查轴的比例，将 y 轴设置为对数尺度
        _check_ax_scales(ax, yaxis="log")
        # 断言柱状图中柱子的数量为 10
        assert len(ax.patches) == 10
        # 检查柱子的颜色是否符合预期，应为 10 个蓝色柱子
        _check_colors(ax.patches, facecolors=["b"] * 10)

    # 测试核密度估计的颜色
    def test_hist_kde_color(self, ts):
        # 导入 scipy 库，如果失败则跳过这个测试
        pytest.importorskip("scipy")
        # 创建一个新的图表并获取轴对象
        _, ax = mpl.pyplot.subplots()
        # 绘制时间序列 ts 的核密度估计图，y 轴使用对数尺度，颜色为红色，将图形绘制在已有的轴上
        ax = ts.plot.kde(logy=True, color="r", ax=ax)
        # 检查轴的比例，将 y 轴设置为对数尺度
        _check_ax_scales(ax, yaxis="log")
        # 获取图上的所有线条对象
        lines = ax.get_lines()
        # 断言图上只有一条线条
        assert len(lines) == 1
        # 检查线条的颜色是否为红色
        _check_colors(lines, ["r"])
class TestDataFramePlots:
    # 标记测试为慢速测试
    @pytest.mark.slow
    # 测试 DataFrame 的直方图绘制（使用 fixture 'hist_df' 提供的数据）
    def test_hist_df_legacy(self, hist_df):
        # 确保调用 hist 方法时产生 UserWarning 警告，忽略警告栈检查
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(hist_df.hist)

    # 标记测试为慢速测试
    @pytest.mark.slow
    # 测试 DataFrame 的直方图绘制，确保处理布局
    def test_hist_df_legacy_layout(self):
        # 创建一个 DataFrame，填充随机数据和日期时间数据
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        df[2] = to_datetime(
            np.random.default_rng(2).integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        # 确保调用 hist 方法时产生 UserWarning 警告，忽略警告栈检查
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            # 调用 _check_plot_works 函数，不显示网格
            axes = _check_plot_works(df.hist, grid=False)
        # 检查绘制的图形的形状，期望 2x2 的布局
        _check_axes_shape(axes, axes_num=3, layout=(2, 2))
        # 确保 axes[1, 1] 图形不可见
        assert not axes[1, 1].get_visible()

        # 对 DataFrame 的特定列绘制直方图
        _check_plot_works(df[[2]].hist)

    # 标记测试为慢速测试
    @pytest.mark.slow
    # 测试 DataFrame 的直方图绘制，确保处理布局
    def test_hist_df_legacy_layout2(self):
        # 创建一个 DataFrame，填充随机数据
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 1)))
        # 调用 _check_plot_works 函数，绘制直方图
        _check_plot_works(df.hist)

    # 标记测试为慢速测试
    @pytest.mark.slow
    # 测试 DataFrame 的直方图绘制，确保处理布局
    def test_hist_df_legacy_layout3(self):
        # 创建一个 DataFrame，填充随机数据和日期时间数据
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        df[5] = to_datetime(
            np.random.default_rng(2).integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        # 确保调用 hist 方法时产生 UserWarning 警告，忽略警告栈检查
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            # 调用 _check_plot_works 函数，使用 4x2 的布局
            axes = _check_plot_works(df.hist, layout=(4, 2))
        # 检查绘制的图形的形状，期望 6 个子图，4x2 的布局
        _check_axes_shape(axes, axes_num=6, layout=(4, 2))

    # 标记测试为慢速测试，使用参数化测试
    @pytest.mark.slow
    @pytest.mark.parametrize(
        # 参数化测试，包含多个参数字典
        "kwargs", [{"sharex": True, "sharey": True}, {"figsize": (8, 10)}, {"bins": 5}]
    )
    # 测试 DataFrame 的直方图绘制，确保处理不同的参数
    def test_hist_df_legacy_layout_kwargs(self, kwargs):
        # 创建一个 DataFrame，填充随机数据和日期时间数据
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        df[5] = to_datetime(
            np.random.default_rng(2).integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        # 确保调用 hist 方法时产生 UserWarning 警告，忽略警告栈检查
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            # 调用 _check_plot_works 函数，传入不同的 kwargs
            _check_plot_works(df.hist, **kwargs)

    # 标记测试为慢速测试
    @pytest.mark.slow
    # 测试 DataFrame 的直方图绘制，确保处理标签大小和旋转
    def test_hist_df_legacy_layout_labelsize_rot(self, frame_or_series):
        # 确保 xlabelsize 和 xrot 参数被正确处理
        obj = frame_or_series(range(10))
        xf, yf = 20, 18
        xrot, yrot = 30, 40
        # 绘制直方图，设置标签大小和旋转
        axes = obj.hist(xlabelsize=xf, xrot=xrot, ylabelsize=yf, yrot=yrot)
        # 检查绘制的图形的刻度属性，确保设置正确
        _check_ticks_props(axes, xlabelsize=xf, xrot=xrot, ylabelsize=yf, yrot=yrot)
    # 定义测试函数，用于测试带有累积、密度等参数的 Series 直方图绘制
    def test_hist_df_legacy_rectangles(self):
        # 创建一个包含 0 到 9 的 Series 对象
        ser = Series(range(10))
        # 绘制直方图，设置累积为 True，分组数为 4，密度为 True
        ax = ser.hist(cumulative=True, bins=4, density=True)
        # 检查最后一个柱状图的高度是否为 1.0
        rects = [x for x in ax.get_children() if isinstance(x, mpl.patches.Rectangle)]
        # 使用近似测试确认最后一个矩形的高度接近于 1.0
        tm.assert_almost_equal(rects[-1].get_height(), 1.0)

    @pytest.mark.slow
    # 标记为慢速测试
    def test_hist_df_legacy_scale(self):
        # 创建一个包含 0 到 9 的 Series 对象
        ser = Series(range(10))
        # 绘制直方图，设置 y 轴为对数尺度
        ax = ser.hist(log=True)
        # 检查 y 轴的尺度是否为对数尺度
        _check_ax_scales(ax, yaxis="log")

    @pytest.mark.slow
    # 标记为慢速测试
    def test_hist_df_legacy_external_error(self):
        # 创建一个包含 0 到 9 的 Series 对象
        ser = Series(range(10))
        # 使用外部错误捕获装饰器，检查是否从 matplotlib.Axes.hist 中传播属性异常
        with tm.external_error_raised(AttributeError):
            # 调用 hist 方法，并传递一个无效参数 'foo'
            ser.hist(foo="bar")

    # 测试非数值或日期时间类型的 DataFrame 抛出异常
    def test_hist_non_numerical_or_datetime_raises(self):
        # 创建一个包含随机数据的 DataFrame，包括浮点数、整数、日期时间和带有 UTC 标志的日期时间列
        df = DataFrame(
            {
                "a": np.random.default_rng(2).random(10),
                "b": np.random.default_rng(2).integers(0, 10, 10),
                "c": to_datetime(
                    np.random.default_rng(2).integers(
                        1582800000000000000, 1583500000000000000, 10, dtype=np.int64
                    )
                ),
                "d": to_datetime(
                    np.random.default_rng(2).integers(
                        1582800000000000000, 1583500000000000000, 10, dtype=np.int64
                    ),
                    utc=True,
                ),
            }
        )
        # 将 DataFrame 的数据类型转换为对象类型
        df_o = df.astype(object)

        # 设置预期的异常消息
        msg = "hist method requires numerical or datetime columns, nothing to plot."
        # 使用 pytest 的断言检查是否抛出预期的 ValueError 异常，并且异常消息与预期消息匹配
        with pytest.raises(ValueError, match=msg):
            # 调用 DataFrame 的 hist 方法，预期抛出 ValueError 异常
            df_o.hist()

    @pytest.mark.parametrize(
        "layout_test",
        (
            {"layout": None, "expected_size": (2, 2)},  # 默认为 2x2 布局
            {"layout": (2, 2), "expected_size": (2, 2)},
            {"layout": (4, 1), "expected_size": (4, 1)},
            {"layout": (1, 4), "expected_size": (1, 4)},
            {"layout": (3, 3), "expected_size": (3, 3)},
            {"layout": (-1, 4), "expected_size": (1, 4)},
            {"layout": (4, -1), "expected_size": (4, 1)},
            {"layout": (-1, 2), "expected_size": (2, 2)},
            {"layout": (2, -1), "expected_size": (2, 2)},
        ),
    )
    # 参数化测试，用于测试不同的布局设置
    def test_hist_layout(self, layout_test):
        # 创建一个包含标准正态分布随机数据的 DataFrame，包括两列和一个日期时间列
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        # 添加一个日期时间列
        df[2] = to_datetime(
            np.random.default_rng(2).integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        # 绘制 DataFrame 的直方图，并使用 layout_test 参数传递不同的布局设置
        axes = df.hist(layout=layout_test["layout"])
        # 检查绘制的子图数量是否符合预期的布局大小
        expected = layout_test["expected_size"]
        _check_axes_shape(axes, axes_num=3, layout=expected)
    # 测试直方图布局错误的情况
    def test_hist_layout_error(self):
        # 创建一个包含10行2列随机标准正态分布数据的DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        # 添加第3列，该列包含随机生成的日期时间数据
        df[2] = to_datetime(
            np.random.default_rng(2).integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        # 设置错误信息，表明布局大小为1x1时，无法容纳全部4个图表
        msg = "Layout of 1x1 must be larger than required size 3"
        # 使用pytest断言抛出值错误，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            df.hist(layout=(1, 1))

        # 设置错误信息，表明布局格式无效
        msg = re.escape("Layout must be a tuple of (rows, columns)")
        # 使用pytest断言抛出值错误，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            df.hist(layout=(1,))
        # 设置错误信息，表明布局的维度至少要有一个正数
        msg = "At least one dimension of layout must be positive"
        # 使用pytest断言抛出值错误，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            df.hist(layout=(-1, -1))

    # GH 9351
    # 测试紧凑布局功能
    def test_tight_layout(self):
        # 创建一个包含10行2列随机标准正态分布数据的DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        # 添加第3列，该列包含随机生成的日期时间数据
        df[2] = to_datetime(
            np.random.default_rng(2).integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        # 调用_check_plot_works函数，传入df.hist作为参数，并设置default_axes=True
        _check_plot_works(df.hist, default_axes=True)
        # 调用matplotlib.pyplot.tight_layout()方法，使图形布局更紧凑
        mpl.pyplot.tight_layout()

    # 测试直方图子图x轴旋转的情况
    def test_hist_subplot_xrot(self):
        # GH 30288
        # 创建一个DataFrame，包含"length"和"animal"两列数据
        df = DataFrame(
            {
                "length": [1.5, 0.5, 1.2, 0.9, 3],
                "animal": ["pig", "rabbit", "pig", "pig", "rabbit"],
            }
        )
        # 调用_check_plot_works函数，传入df.hist作为参数，并设置default_axes=True以及其他绘图参数
        axes = _check_plot_works(
            df.hist,
            default_axes=True,
            column="length",
            by="animal",
            bins=5,
            xrot=0,
        )
        # 调用_check_ticks_props函数，验证绘图中的x轴标签旋转为0度
        _check_ticks_props(axes, xrot=0)

    @pytest.mark.parametrize(
        "column, expected",
        [
            (None, ["width", "length", "height"]),
            (["length", "width", "height"], ["length", "width", "height"]),
        ],
    )
    # 测试直方图列顺序不变的情况
    def test_hist_column_order_unchanged(self, column, expected):
        # GH29235
        # 创建一个包含"width"、"length"、"height"三列数据的DataFrame，并设置索引
        df = DataFrame(
            {
                "width": [0.7, 0.2, 0.15, 0.2, 1.1],
                "length": [1.5, 0.5, 1.2, 0.9, 3],
                "height": [3, 0.5, 3.4, 2, 1],
            },
            index=["pig", "rabbit", "duck", "chicken", "horse"],
        )
        # 调用_check_plot_works函数，传入df.hist作为参数，并设置default_axes=True以及其他绘图参数
        axes = _check_plot_works(
            df.hist,
            default_axes=True,
            column=column,
            layout=(1, 3),
        )
        # 获取每个子图的标题，验证列的顺序是否不变
        result = [axes[0, i].get_title() for i in range(3)]
        assert result == expected
    @pytest.mark.parametrize(
        "histtype, expected",
        [
            ("bar", True),  # 参数化测试：histtype为"bar"时，期望返回True
            ("barstacked", True),  # 参数化测试：histtype为"barstacked"时，期望返回True
            ("step", False),  # 参数化测试：histtype为"step"时，期望返回False
            ("stepfilled", True),  # 参数化测试：histtype为"stepfilled"时，期望返回True
        ],
    )
    def test_histtype_argument(self, histtype, expected):
        # GH23992 验证histtype参数的功能是否正常
        df = DataFrame(
            np.random.default_rng(2).integers(1, 10, size=(10, 2)), columns=["a", "b"]
        )
        ax = df.hist(histtype=histtype)  # 绘制DataFrame的直方图，指定histtype参数
        _check_patches_all_filled(ax, filled=expected)  # 检查绘图中的填充情况是否符合预期

    @pytest.mark.parametrize("by", [None, "c"])
    @pytest.mark.parametrize("column", [None, "b"])
    def test_hist_with_legend(self, by, column):
        # GH 6279 - DataFrame直方图可以带有图例
        expected_axes_num = 1 if by is None and column is not None else 2  # 预期的子图数量
        expected_layout = (1, expected_axes_num)  # 预期的布局
        expected_labels = column or ["a", "b"]  # 预期的标签，如果column为None则使用默认标签["a", "b"]
        if by is not None:
            expected_labels = [expected_labels] * 2  # 如果指定了by，则标签重复两次

        index = Index(5 * ["1"] + 5 * ["2"], name="c")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            index=index,
            columns=["a", "b"],
        )

        # 当绘图方法自动生成子图时使用default_axes=True
        axes = _check_plot_works(
            df.hist,
            default_axes=True,
            legend=True,
            by=by,
            column=column,
        )

        _check_axes_shape(axes, axes_num=expected_axes_num, layout=expected_layout)  # 检查子图的形状是否符合预期
        if by is None and column is None:
            axes = axes[0]
        for expected_label, ax in zip(expected_labels, axes):
            _check_legend_labels(ax, expected_label)  # 检查图例标签是否符合预期

    @pytest.mark.parametrize("by", [None, "c"])
    @pytest.mark.parametrize("column", [None, "b"])
    def test_hist_with_legend_raises(self, by, column):
        # GH 6279 - DataFrame直方图带有图例和标签时会引发异常
        index = Index(5 * ["1"] + 5 * ["2"], name="c")
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            index=index,
            columns=["a", "b"],
        )

        with pytest.raises(ValueError, match="Cannot use both legend and label"):
            df.hist(legend=True, by=by, column=column, label="d")  # 检查使用图例和标签时是否会引发异常

    def test_hist_df_kwargs(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        _, ax = mpl.pyplot.subplots()
        ax = df.plot.hist(bins=5, ax=ax)
        assert len(ax.patches) == 10  # 检查绘图中的patch数量是否为10

    def test_hist_df_with_nonnumerics(self):
        # GH 9853
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=["A", "B", "C", "D"],
        )
        df["E"] = ["x", "y"] * 5
        _, ax = mpl.pyplot.subplots()
        ax = df.plot.hist(bins=5, ax=ax)
        assert len(ax.patches) == 20  # 检查绘图中的patch数量是否为20
    def test_hist_df_with_nonnumerics_no_bins(self):
        # GH 9853
        # 创建一个包含随机标准正态分布数据的 DataFrame，形状为 (10, 4)，列名为 ["A", "B", "C", "D"]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=["A", "B", "C", "D"],
        )
        # 在 DataFrame 中添加一列 "E"，内容为交替的字符串 "x" 和 "y"
        df["E"] = ["x", "y"] * 5
        # 创建一个新的图表，并返回图表的轴对象 ax，绘制 DataFrame df 的直方图，默认 bins=10
        _, ax = mpl.pyplot.subplots()
        ax = df.plot.hist(ax=ax)  # bins=10
        # 断言图表中的矩形条数量为 40
        assert len(ax.patches) == 40

    def test_hist_secondary_legend(self):
        # GH 9610
        # 创建一个包含随机标准正态分布数据的 DataFrame，形状为 (10, 4)，列名为 ["a", "b", "c", "d"]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)), columns=list("abcd")
        )

        # 创建一个新的图表，并返回图表的轴对象 ax
        _, ax = mpl.pyplot.subplots()
        # 绘制 DataFrame df 中 "a" 列的直方图，显示图例，将轴对象 ax 作为参数传递
        ax = df["a"].plot.hist(legend=True, ax=ax)
        # 绘制 DataFrame df 中 "b" 列的直方图，显示图例，设置为次要 Y 轴显示
        df["b"].plot.hist(ax=ax, legend=True, secondary_y=True)
        # 验证两个图例都绘制在左侧轴上
        # 左侧和右侧轴必须可见
        _check_legend_labels(ax, labels=["a", "b (right)"])
        assert ax.get_yaxis().get_visible()
        assert ax.right_ax.get_yaxis().get_visible()

    def test_hist_secondary_secondary(self):
        # GH 9610
        # 创建一个包含随机标准正态分布数据的 DataFrame，形状为 (10, 4)，列名为 ["a", "b", "c", "d"]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)), columns=list("abcd")
        )
        # 创建一个新的图表，并返回图表的轴对象 ax
        _, ax = mpl.pyplot.subplots()
        # 绘制 DataFrame df 中 "a" 列的直方图，显示图例，设置为次要 Y 轴显示，将轴对象 ax 作为参数传递
        ax = df["a"].plot.hist(legend=True, secondary_y=True, ax=ax)
        # 绘制 DataFrame df 中 "b" 列的直方图，显示图例，设置为次要 Y 轴显示
        df["b"].plot.hist(ax=ax, legend=True, secondary_y=True)
        # 验证两个图例都绘制在左侧轴上
        # 左侧轴必须不可见，右侧轴必须可见
        _check_legend_labels(ax.left_ax, labels=["a (right)", "b (right)"])
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()

    def test_hist_secondary_primary(self):
        # GH 9610
        # 创建一个包含随机标准正态分布数据的 DataFrame，形状为 (10, 4)，列名为 ["a", "b", "c", "d"]
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)), columns=list("abcd")
        )
        # 创建一个新的图表，并返回图表的轴对象 ax
        _, ax = mpl.pyplot.subplots()
        # 绘制 DataFrame df 中 "a" 列的直方图，显示图例，设置为次要 Y 轴显示，将轴对象 ax 作为参数传递
        ax = df["a"].plot.hist(legend=True, secondary_y=True, ax=ax)
        # 绘制 DataFrame df 中 "b" 列的直方图，显示图例，轴对象 ax 作为参数传递
        # 返回右侧轴对象
        df["b"].plot.hist(ax=ax, legend=True)
        # 验证两个图例都绘制在左侧轴上
        # 左侧和右侧轴必须可见
        _check_legend_labels(ax.left_ax, labels=["a (right)", "b"])
        assert ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()
    # 定义一个测试方法，用于测试带有 NaN 和权重的直方图生成
    def test_hist_with_nans_and_weights(self):
        # GH 48884：引用 GitHub 问题编号
        # 创建一个包含 NaN 值的 DataFrame，列名为 'a', 'b', 'c'
        df = DataFrame(
            [[np.nan, 0.2, 0.3], [0.4, np.nan, np.nan], [0.7, 0.8, 0.9]],
            columns=list("abc"),
        )
        # 创建权重数组
        weights = np.array([0.25, 0.3, 0.45])
        # 创建一个没有 NaN 值的 DataFrame，列名同上
        no_nan_df = DataFrame([[0.4, 0.2, 0.3], [0.7, 0.8, 0.9]], columns=list("abc"))
        # 创建对应于没有 NaN 值的权重数组
        no_nan_weights = np.array([[0.3, 0.25, 0.25], [0.45, 0.45, 0.45]])

        # 创建一个新的图表对象和子图 ax0
        _, ax0 = mpl.pyplot.subplots()
        # 使用 df 的数据绘制直方图到 ax0，使用给定的权重
        df.plot.hist(ax=ax0, weights=weights)
        # 从 ax0 的子元素中筛选出所有的矩形对象，存入 rects 列表
        rects = [x for x in ax0.get_children() if isinstance(x, mpl.patches.Rectangle)]
        # 获取每个矩形的高度，存入 heights 列表
        heights = [rect.get_height() for rect in rects]

        # 创建另一个新的图表对象和子图 ax1
        _, ax1 = mpl.pyplot.subplots()
        # 使用 no_nan_df 的数据绘制直方图到 ax1，使用对应的没有 NaN 的权重
        no_nan_df.plot.hist(ax=ax1, weights=no_nan_weights)
        # 从 ax1 的子元素中筛选出所有的矩形对象，存入 no_nan_rects 列表
        no_nan_rects = [
            x for x in ax1.get_children() if isinstance(x, mpl.patches.Rectangle)
        ]
        # 获取每个矩形的高度，存入 no_nan_heights 列表
        no_nan_heights = [rect.get_height() for rect in no_nan_rects]

        # 断言：验证 heights 和 no_nan_heights 中的每个对应元素相等
        assert all(h0 == h1 for h0, h1 in zip(heights, no_nan_heights))

        # 创建一个引发 ValueError 异常的测试情形
        # idxerror_weights 是一个错误形状的权重数组
        idxerror_weights = np.array([[0.3, 0.25], [0.45, 0.45]])

        # 错误消息字符串
        msg = "weights must have the same shape as data, or be a single column"
        # 创建一个新的图表对象和子图 ax2
        _, ax2 = mpl.pyplot.subplots()
        # 使用 no_nan_df 的数据尝试绘制直方图到 ax2，使用错误形状的权重数组
        with pytest.raises(ValueError, match=msg):
            no_nan_df.plot.hist(ax=ax2, weights=idxerror_weights)
class TestDataFrameGroupByPlots:
    # 测试组合直方图（旧版）
    def test_grouped_hist_legacy(self):
        # 使用种子值10初始化随机数生成器
        rs = np.random.default_rng(10)
        # 创建包含10行1列的DataFrame，列名为"A"，数据为标准正态分布随机数
        df = DataFrame(rs.standard_normal((10, 1)), columns=["A"])
        # 创建列"B"，其数据是日期时间数据，范围在812419200000000000到819331200000000000之间
        df["B"] = to_datetime(
            rs.integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        # 创建列"C"，包含10个随机整数，范围在0到4之间
        df["C"] = rs.integers(0, 4, 10)
        # 创建列"D"，包含10个值为"X"的字符串
        df["D"] = ["X"] * 10

        # 调用_grouped_hist函数，对"A"列进行直方图分组，按"C"列进行分组
        axes = _grouped_hist(df.A, by=df.C)
        # 检查返回的axes对象的形状是否符合预期，应为4个子图，布局为2行2列
        _check_axes_shape(axes, axes_num=4, layout=(2, 2))

    # 测试组合直方图（旧版），无列名情况下的axes形状检查
    def test_grouped_hist_legacy_axes_shape_no_col(self):
        # 使用种子值10初始化随机数生成器
        rs = np.random.default_rng(10)
        # 创建包含10行1列的DataFrame，列名为"A"，数据为标准正态分布随机数
        df = DataFrame(rs.standard_normal((10, 1)), columns=["A"])
        # 创建列"B"，其数据是日期时间数据，范围在812419200000000000到819331200000000000之间
        df["B"] = to_datetime(
            rs.integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        # 创建列"C"，包含10个随机整数，范围在0到4之间
        df["C"] = rs.integers(0, 4, 10)
        # 创建列"D"，包含10个值为"X"的字符串
        df["D"] = ["X"] * 10
        # 调用DataFrame的hist方法，按"C"列进行分组绘制直方图
        axes = df.hist(by=df.C)
        # 检查返回的axes对象的形状是否符合预期，应为4个子图，布局为2行2列
        _check_axes_shape(axes, axes_num=4, layout=(2, 2))

    # 测试组合直方图（旧版），单个键的情况下的直方图绘制
    def test_grouped_hist_legacy_single_key(self):
        # 使用种子值2初始化随机数生成器
        rs = np.random.default_rng(2)
        # 创建包含10行1列的DataFrame，列名为"A"，数据为标准正态分布随机数
        df = DataFrame(rs.standard_normal((10, 1)), columns=["A"])
        # 创建列"B"，其数据是日期时间数据，范围在812419200000000000到819331200000000000之间
        df["B"] = to_datetime(
            rs.integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        # 创建列"C"，包含10个随机整数，范围在0到4之间
        df["C"] = rs.integers(0, 4, 10)
        # 创建列"D"，包含10个值为"X"的字符串
        df["D"] = ["X"] * 10
        # 调用DataFrame的hist方法，按"D"列进行单个键的直方图绘制，x轴标签旋转30度
        axes = df.hist(by="D", rot=30)
        # 检查返回的axes对象的形状是否符合预期，应为单个子图，布局为1行1列
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))
        # 检查子图的ticks属性，设置x轴标签旋转角度为30度
        _check_ticks_props(axes, xrot=30)

    # 测试组合直方图（旧版），带有多个参数的kwargs处理
    def test_grouped_hist_legacy_grouped_hist_kwargs(self):
        # 使用种子值2初始化随机数生成器
        rs = np.random.default_rng(2)
        # 创建包含10行1列的DataFrame，列名为"A"，数据为标准正态分布随机数
        df = DataFrame(rs.standard_normal((10, 1)), columns=["A"])
        # 创建列"B"，其数据是日期时间数据，范围在812419200000000000到819331200000000000之间
        df["B"] = to_datetime(
            rs.integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        # 创建列"C"，包含10个随机整数，范围在0到4之间
        df["C"] = rs.integers(0, 4, 10)
        # 调用_grouped_hist函数，对"A"列进行直方图分组，按"C"列进行分组
        # 设置累积分布、分箱数为4、x轴标签大小为20、x轴标签旋转角度为30、y轴标签大小为18、y轴标签旋转角度为40、密度为True
        axes = _grouped_hist(
            df.A,
            by=df.C,
            cumulative=True,
            bins=4,
            xlabelsize=20,
            xrot=30,
            ylabelsize=18,
            yrot=40,
            density=True,
        )
        # 检查每个axes对象中最后一个箱子（索引5）的高度是否为1.0
        for ax in axes.ravel():
            rects = [
                x for x in ax.get_children() if isinstance(x, mpl.patches.Rectangle)
            ]
            height = rects[-1].get_height()
            tm.assert_almost_equal(height, 1.0)
        # 检查子图的ticks属性，设置x轴和y轴标签大小和旋转角度
        _check_ticks_props(axes, xlabelsize=20, xrot=30, ylabelsize=18, yrot=40)
    # 定义测试函数，测试_grouped_hist函数的使用情况
    def test_grouped_hist_legacy_grouped_hist(self):
        # 创建随机数生成器对象
        rs = np.random.default_rng(2)
        # 创建包含随机标准正态分布数据的DataFrame，列名为"A"
        df = DataFrame(rs.standard_normal((10, 1)), columns=["A"])
        # 生成随机日期时间数据并添加到DataFrame的"B"列
        df["B"] = to_datetime(
            rs.integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        # 在DataFrame中添加随机整数数据到"C"列，范围为0到3
        df["C"] = rs.integers(0, 4, 10)
        # 在DataFrame中添加包含字符串"X"的"D"列
        df["D"] = ["X"] * 10
        # 调用_grouped_hist函数，返回绘图坐标轴对象
        axes = _grouped_hist(df.A, by=df.C, log=True)
        # 检查y轴的比例尺是否为对数尺度
        _check_ax_scales(axes, yaxis="log")

    # 定义测试函数，测试_grouped_hist函数在处理外部错误时的行为
    def test_grouped_hist_legacy_external_err(self):
        # 创建随机数生成器对象
        rs = np.random.default_rng(2)
        # 创建包含随机标准正态分布数据的DataFrame，列名为"A"
        df = DataFrame(rs.standard_normal((10, 1)), columns=["A"])
        # 生成随机日期时间数据并添加到DataFrame的"B"列
        df["B"] = to_datetime(
            rs.integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        # 在DataFrame中添加随机整数数据到"C"列，范围为0到3
        df["C"] = rs.integers(0, 4, 10)
        # 在DataFrame中添加包含字符串"X"的"D"列
        df["D"] = ["X"] * 10
        # 使用pytest模块验证，期望从matplotlib.Axes.hist传播属性异常
        with tm.external_error_raised(AttributeError):
            _grouped_hist(df.A, by=df.C, foo="bar")

    # 定义测试函数，测试_grouped_hist函数在处理图形大小错误时的行为
    def test_grouped_hist_legacy_figsize_err(self):
        # 创建随机数生成器对象
        rs = np.random.default_rng(2)
        # 创建包含随机标准正态分布数据的DataFrame，列名为"A"
        df = DataFrame(rs.standard_normal((10, 1)), columns=["A"])
        # 生成随机日期时间数据并添加到DataFrame的"B"列
        df["B"] = to_datetime(
            rs.integers(
                812419200000000000,
                819331200000000000,
                size=10,
                dtype=np.int64,
            )
        )
        # 在DataFrame中添加随机整数数据到"C"列，范围为0到3
        df["C"] = rs.integers(0, 4, 10)
        # 在DataFrame中添加包含字符串"X"的"D"列
        df["D"] = ["X"] * 10
        # 准备错误消息字符串
        msg = "Specify figure size by tuple instead"
        # 使用pytest模块验证，期望抛出指定错误消息的值错误
        with pytest.raises(ValueError, match=msg):
            df.hist(by="C", figsize="default")

    # 定义测试函数，测试_grouped_hist函数在多种情况下的行为
    def test_grouped_hist_legacy2(self):
        # 设定数据集大小
        n = 10
        # 创建身高数据序列，均值166，标准差20
        weight = Series(np.random.default_rng(2).normal(166, 20, size=n))
        # 创建体重数据序列，均值60，标准差10
        height = Series(np.random.default_rng(2).normal(60, 10, size=n))
        # 创建性别数据序列，随机选择0或1
        gender_int = np.random.default_rng(2).choice([0, 1], size=n)
        # 创建包含身高、体重和性别数据的DataFrame
        df_int = DataFrame({"height": height, "weight": weight, "gender": gender_int})
        # 按性别分组DataFrame
        gb = df_int.groupby("gender")
        # 对每个分组应用直方图绘制函数，返回绘图坐标轴对象列表
        axes = gb.hist()
        # 断言返回的坐标轴对象列表长度为2
        assert len(axes) == 2
        # 断言当前绘图窗口数量为2
        assert len(mpl.pyplot.get_fignums()) == 2

    # 使用pytest标记的参数化测试函数，测试不同参数下的布局要求
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "msg, plot_col, by_col, layout",
        [
            [
                "Layout of 1x1 must be larger than required size 2",
                "weight",
                "gender",
                (1, 1),
            ],
            [
                "Layout of 1x3 must be larger than required size 4",
                "height",
                "category",
                (1, 3),
            ],
            [
                "At least one dimension of layout must be positive",
                "height",
                "category",
                (-1, -1),
            ],
        ],
    )
    # 测试函数，用于检查当传入特定参数时，是否会引发 ValueError 异常
    def test_grouped_hist_layout_error(self, hist_df, msg, plot_col, by_col, layout):
        # 将输入的 DataFrame 赋值给 df
        df = hist_df
        # 使用 pytest 检查调用 df.hist() 方法时是否引发预期的 ValueError 异常，并匹配给定的错误消息
        with pytest.raises(ValueError, match=msg):
            df.hist(column=plot_col, by=getattr(df, by_col), layout=layout)

    # 标记为慢速测试的函数，用于检查是否会产生 UserWarning 警告
    def test_grouped_hist_layout_warning(self, hist_df):
        # 将输入的 DataFrame 赋值给 df
        df = hist_df
        # 使用 tm.assert_produces_warning() 检查调用 _check_plot_works() 方法时是否会产生 UserWarning 警告，关闭栈级别检查
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            # 调用 df.hist() 方法，绘制 "height" 列的直方图，按照性别分组，布局为 (2, 1)，返回绘图的 axes 对象
            axes = _check_plot_works(
                df.hist, column="height", by=df.gender, layout=(2, 1)
            )
        # 检查返回的 axes 对象形状是否符合预期的 (2, 1)
        _check_axes_shape(axes, axes_num=2, layout=(2, 1))

    # 标记为慢速测试的函数，使用参数化装饰器进行多组参数化测试
    @pytest.mark.parametrize(
        "layout, check_layout, figsize",
        [[(4, 1), (4, 1), None], [(-1, 1), (4, 1), None], [(4, 2), (4, 2), (12, 8)]],
    )
    def test_grouped_hist_layout_figsize(self, hist_df, layout, check_layout, figsize):
        # 将输入的 DataFrame 赋值给 df
        df = hist_df
        # 调用 df.hist() 方法，绘制 "height" 列的直方图，按照类别分组，使用给定的布局和图像尺寸参数
        axes = df.hist(column="height", by=df.category, layout=layout, figsize=figsize)
        # 检查返回的 axes 对象形状是否符合预期的 axes_num 和 layout
        _check_axes_shape(axes, axes_num=4, layout=check_layout, figsize=figsize)

    # 标记为慢速测试的函数，使用参数化装饰器进行多组参数化测试
    @pytest.mark.parametrize("kwargs", [{}, {"column": "height", "layout": (2, 2)}])
    def test_grouped_hist_layout_by_warning(self, hist_df, kwargs):
        # 将输入的 DataFrame 赋值给 df
        df = hist_df
        # GH 6769: 使用 tm.assert_produces_warning() 检查调用 _check_plot_works() 方法时是否会产生 UserWarning 警告，关闭栈级别检查
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            # 调用 _check_plot_works() 方法，绘制直方图，按照教室分组，使用传入的 kwargs 参数
            axes = _check_plot_works(df.hist, by="classroom", **kwargs)
        # 检查返回的 axes 对象形状是否符合预期的 (2, 2)
        _check_axes_shape(axes, axes_num=3, layout=(2, 2))

    # 标记为慢速测试的函数，使用参数化装饰器进行多组参数化测试
    @pytest.mark.parametrize(
        "kwargs, axes_num, layout",
        [
            [{"by": "gender", "layout": (3, 5)}, 2, (3, 5)],
            [{"column": ["height", "weight", "category"]}, 3, (2, 2)],
        ],
    )
    def test_grouped_hist_layout_axes(self, hist_df, kwargs, axes_num, layout):
        # 将输入的 DataFrame 赋值给 df
        df = hist_df
        # 调用 df.hist() 方法，根据传入的 kwargs 参数绘制直方图
        axes = df.hist(**kwargs)
        # 检查返回的 axes 对象形状是否符合预期的 axes_num 和 layout
        _check_axes_shape(axes, axes_num=axes_num, layout=layout)

    # 测试函数，用于检查在多个 axes 下绘制多列直方图时的行为
    def test_grouped_hist_multiple_axes(self, hist_df):
        # GH 6970, GH 7069: 将输入的 DataFrame 赋值给 df
        df = hist_df

        # 创建一个包含 2 行 3 列 axes 对象的图表
        fig, axes = mpl.pyplot.subplots(2, 3)
        # 调用 df.hist() 方法，绘制 "height", "weight", "category" 列的直方图，放置在第一个 axes 上
        returned = df.hist(column=["height", "weight", "category"], ax=axes[0])
        # 检查返回的 axes 对象形状是否符合预期的 axes_num 和 layout
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        # 使用 tm.assert_numpy_array_equal() 检查返回的数组与 axes[0] 是否相等
        tm.assert_numpy_array_equal(returned, axes[0])
        # 断言返回的第一个 axes 所属的图表对象与创建的 fig 是否相等
        assert returned[0].figure is fig

    # 测试函数，用于检查在多个 axes 下按照类别分组绘制直方图时的行为
    def test_grouped_hist_multiple_axes_no_cols(self, hist_df):
        # GH 6970, GH 7069: 将输入的 DataFrame 赋值给 df
        df = hist_df

        # 创建一个包含 2 行 3 列 axes 对象的图表
        fig, axes = mpl.pyplot.subplots(2, 3)
        # 调用 df.hist() 方法，按照 "classroom" 列分组绘制直方图，放置在第二个 axes 上
        returned = df.hist(by="classroom", ax=axes[1])
        # 检查返回的 axes 对象形状是否符合预期的 axes_num 和 layout
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        # 使用 tm.assert_numpy_array_equal() 检查返回的数组与 axes[1] 是否相等
        tm.assert_numpy_array_equal(returned, axes[1])
        # 断言返回的第一个 axes 所属的图表对象与创建的 fig 是否相等
        assert returned[0].figure is fig
    # 测试函数：测试多轴直方图绘制时的错误处理
    def test_grouped_hist_multiple_axes_error(self, hist_df):
        # GH 6970, GH 7069
        # 将传入的 hist_df 赋给 df
        df = hist_df
        # 创建一个 2x3 的子图对象
        fig, axes = mpl.pyplot.subplots(2, 3)
        # 抛出值错误异常，匹配特定错误消息
        msg = "The number of passed axes must be 1, the same as the output plot"
        with pytest.raises(ValueError, match=msg):
            # 调用 DataFrame 的 hist 方法，绘制 "height" 列的直方图到指定的 axes
            axes = df.hist(column="height", ax=axes)

    # 测试函数：测试共享 x 轴的效果
    def test_axis_share_x(self, hist_df):
        # 将传入的 hist_df 赋给 df
        df = hist_df
        # 按照 "gender" 分组，绘制 "height" 列的直方图，并共享 x 轴
        ax1, ax2 = df.hist(column="height", by=df.gender, sharex=True)

        # 验证 x 轴是否被正确共享
        assert get_x_axis(ax1).joined(ax1, ax2)
        assert get_x_axis(ax2).joined(ax1, ax2)

        # 验证 y 轴未被共享
        assert not get_y_axis(ax1).joined(ax1, ax2)
        assert not get_y_axis(ax2).joined(ax1, ax2)

    # 测试函数：测试共享 y 轴的效果
    def test_axis_share_y(self, hist_df):
        # 将传入的 hist_df 赋给 df
        df = hist_df
        # 按照 "gender" 分组，绘制 "height" 列的直方图，并共享 y 轴
        ax1, ax2 = df.hist(column="height", by=df.gender, sharey=True)

        # 验证 y 轴是否被正确共享
        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)

        # 验证 x 轴未被共享
        assert not get_x_axis(ax1).joined(ax1, ax2)
        assert not get_x_axis(ax2).joined(ax1, ax2)

    # 测试函数：测试同时共享 x 轴和 y 轴的效果
    def test_axis_share_xy(self, hist_df):
        # 将传入的 hist_df 赋给 df
        df = hist_df
        # 按照 "gender" 分组，绘制 "height" 列的直方图，并共享 x 轴和 y 轴
        ax1, ax2 = df.hist(column="height", by=df.gender, sharex=True, sharey=True)

        # 验证 x 轴和 y 轴是否被正确共享
        assert get_x_axis(ax1).joined(ax1, ax2)
        assert get_x_axis(ax2).joined(ax1, ax2)

        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)

    # 参数化测试：测试 histtype 参数的作用
    @pytest.mark.parametrize(
        "histtype, expected",
        [
            ("bar", True),
            ("barstacked", True),
            ("step", False),
            ("stepfilled", True),
        ],
    )
    def test_histtype_argument(self, histtype, expected):
        # GH23992 验证 histtype 参数的功能
        # 创建一个 DataFrame，包含随机整数数据，用于绘制直方图
        df = DataFrame(
            np.random.default_rng(2).integers(1, 10, size=(10, 2)), columns=["a", "b"]
        )
        # 按照 "a" 列分组，根据 histtype 参数绘制直方图
        ax = df.hist(by="a", histtype=histtype)
        # 检查直方图的填充情况是否符合预期
        _check_patches_all_filled(ax, filled=expected)
```