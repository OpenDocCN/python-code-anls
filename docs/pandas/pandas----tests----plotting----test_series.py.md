# `D:\src\scipysrc\pandas\pandas\tests\plotting\test_series.py`

```
"""Test cases for Series.plot"""

# 导入所需的模块和函数
from datetime import datetime
from itertools import chain

import numpy as np
import pytest

# 导入所需的兼容性和测试装饰器
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td

# 导入 pandas 库和所需的类、函数
import pandas as pd
from pandas import (
    DataFrame,
    Series,
    date_range,
    period_range,
    plotting,
)
import pandas._testing as tm

# 导入与绘图相关的测试函数
from pandas.tests.plotting.common import (
    _check_ax_scales,
    _check_axes_shape,
    _check_colors,
    _check_grid_settings,
    _check_has_errorbars,
    _check_legend_labels,
    _check_plot_works,
    _check_text_labels,
    _check_ticks_props,
    _unpack_cycler,
    get_y_axis,
)

# 导入自定义的工作日偏移类
from pandas.tseries.offsets import CustomBusinessDay

# 导入并导入依赖的 matplotlib 包
mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")

# 导入 pandas 绘图模块相关的转换器和样式函数
from pandas.plotting._matplotlib.converter import DatetimeConverter
from pandas.plotting._matplotlib.style import get_standard_colors


# 定义测试使用的时间序列 fixture
@pytest.fixture
def ts():
    return Series(
        np.arange(10, dtype=np.float64),
        index=date_range("2020-01-01", periods=10),
        name="ts",
    )


# 定义测试使用的 Series fixture
@pytest.fixture
def series():
    return Series(
        range(10), dtype=np.float64, name="series", index=[f"i_{i}" for i in range(10)]
    )


# 定义测试 Series 绘图功能的测试类
class TestSeriesPlots:

    # 测试基本绘图功能，参数化测试不同的参数
    @pytest.mark.slow
    @pytest.mark.parametrize("kwargs", [{"label": "foo"}, {"use_index": False}])
    def test_plot(self, ts, kwargs):
        _check_plot_works(ts.plot, **kwargs)

    # 测试绘图中的刻度属性设置
    @pytest.mark.slow
    def test_plot_tick_props(self, ts):
        axes = _check_plot_works(ts.plot, rot=0)
        _check_ticks_props(axes, xrot=0)

    # 测试不同的坐标轴比例（如对数坐标）
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "scale, exp_scale",
        [
            [{"logy": True}, {"yaxis": "log"}],
            [{"logx": True}, {"xaxis": "log"}],
            [{"loglog": True}, {"xaxis": "log", "yaxis": "log"}],
        ],
    )
    def test_plot_scales(self, ts, scale, exp_scale):
        ax = _check_plot_works(ts.plot, style=".", **scale)
        _check_ax_scales(ax, **exp_scale)

    # 测试绘制条形图
    @pytest.mark.slow
    def test_plot_ts_bar(self, ts):
        _check_plot_works(ts[:10].plot.bar)

    # 测试绘制堆叠面积图
    @pytest.mark.slow
    def test_plot_ts_area_stacked(self, ts):
        _check_plot_works(ts.plot.area, stacked=False)

    # 测试绘制带有时间索引的 Series
    def test_plot_iseries(self):
        ser = Series(range(5), period_range("2020-01-01", periods=5))
        _check_plot_works(ser.plot)

    # 测试不同种类的 Series 图表（如线图、条形图、箱线图等）
    @pytest.mark.parametrize(
        "kind",
        [
            "line",
            "bar",
            "barh",
            pytest.param("kde", marks=td.skip_if_no("scipy")),
            "hist",
            "box",
        ],
    )
    def test_plot_series_kinds(self, series, kind):
        _check_plot_works(series[:5].plot, kind=kind)

    # 测试绘制水平条形图
    def test_plot_series_barh(self, series):
        _check_plot_works(series[:10].plot.barh)
    def test_plot_series_bar_ax(self):
        # 调用 _check_plot_works 函数，测试 Series 对象的条形图绘制，设置颜色为黑色
        ax = _check_plot_works(
            Series(np.random.default_rng(2).standard_normal(10)).plot.bar, color="black"
        )
        # 检查条形图的颜色是否符合预期
        _check_colors([ax.patches[0]], facecolors=["black"])

    @pytest.mark.parametrize("kwargs", [{}, {"layout": (-1, 1)}, {"layout": (1, -1)}])
    def test_plot_6951(self, ts, kwargs):
        # GH 6951
        # 调用 _check_plot_works 函数，测试时间序列的绘图功能，通过参数 kwargs 可以设置不同的布局
        ax = _check_plot_works(ts.plot, subplots=True, **kwargs)
        # 检查返回的图形对象的形状是否符合预期，此处期望返回单一的 Axes 对象
        _check_axes_shape(ax, axes_num=1, layout=(1, 1))

    def test_plot_figsize_and_title(self, series):
        # figsize and title
        # 创建一个新的图形和 Axes 对象，并在 Axes 上绘制 Series 对象的图形，设置标题和图形大小
        _, ax = mpl.pyplot.subplots()
        ax = series.plot(title="Test", figsize=(16, 8), ax=ax)
        # 检查图形的标题文本是否正确设置为 "Test"
        _check_text_labels(ax.title, "Test")
        # 检查 Axes 对象的形状、图形大小是否符合预期
        _check_axes_shape(ax, axes_num=1, layout=(1, 1), figsize=(16, 8))

    def test_dont_modify_rcParams(self):
        # GH 8242
        # 测试确保绘图过程中不会修改 rcParams 中的特定属性，以确保全局配置不受影响
        key = "axes.prop_cycle"
        colors = mpl.pyplot.rcParams[key]
        # 创建一个新的图形和 Axes 对象，并在 Axes 上绘制 Series 对象的图形
        _, ax = mpl.pyplot.subplots()
        Series([1, 2, 3]).plot(ax=ax)
        # 断言绘图前后 rcParams 中特定属性的值未发生变化
        assert colors == mpl.pyplot.rcParams[key]

    @pytest.mark.parametrize("kwargs", [{}, {"secondary_y": True}])
    def test_ts_line_lim(self, ts, kwargs):
        # 测试时间序列绘制线图时，限制 x 轴范围的情况
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot(ax=ax, **kwargs)
        # 获取当前 Axes 对象的 x 轴限制范围
        xmin, xmax = ax.get_xlim()
        # 获取图形上的线条对象
        lines = ax.get_lines()
        # 断言线条的起始和结束点在 x 轴的限制范围内
        assert xmin <= lines[0].get_data(orig=False)[0][0]
        assert xmax >= lines[0].get_data(orig=False)[0][-1]

    def test_ts_area_lim(self, ts):
        # 测试时间序列绘制面积图时，限制 x 轴范围的情况
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.area(stacked=False, ax=ax)
        # 获取当前 Axes 对象的 x 轴限制范围
        xmin, xmax = ax.get_xlim()
        # 获取面积图上的第一条线条对象的数据
        line = ax.get_lines()[0].get_data(orig=False)[0]
        # 断言面积图线条的起始和结束点在 x 轴的限制范围内
        assert xmin <= line[0]
        assert xmax >= line[-1]
        # 检查图形的刻度属性，确保 x 轴刻度旋转角度为 0 度
        _check_ticks_props(ax, xrot=0)

    def test_ts_area_lim_xcompat(self, ts):
        # GH 7471
        # 测试时间序列绘制面积图时，限制 x 轴范围，并兼容旧版本的 matplotlib 特性
        _, ax = mpl.pyplot.subplots()
        ax = ts.plot.area(stacked=False, x_compat=True, ax=ax)
        # 获取当前 Axes 对象的 x 轴限制范围
        xmin, xmax = ax.get_xlim()
        # 获取面积图上的第一条线条对象的数据
        line = ax.get_lines()[0].get_data(orig=False)[0]
        # 断言面积图线条的起始和结束点在 x 轴的限制范围内
        assert xmin <= line[0]
        assert xmax >= line[-1]
        # 检查图形的刻度属性，确保 x 轴刻度旋转角度为 30 度
        _check_ticks_props(ax, xrot=30)

    def test_ts_tz_area_lim_xcompat(self, ts):
        # 测试带有时区信息的时间序列绘制面积图时，限制 x 轴范围，并兼容旧版本的 matplotlib 特性
        tz_ts = ts.copy()
        tz_ts.index = tz_ts.tz_localize("GMT").tz_convert("CET")
        _, ax = mpl.pyplot.subplots()
        ax = tz_ts.plot.area(stacked=False, x_compat=True, ax=ax)
        # 获取当前 Axes 对象的 x 轴限制范围
        xmin, xmax = ax.get_xlim()
        # 获取面积图上的第一条线条对象的数据
        line = ax.get_lines()[0].get_data(orig=False)[0]
        # 断言面积图线条的起始和结束点在 x 轴的限制范围内
        assert xmin <= line[0]
        assert xmax >= line[-1]
        # 检查图形的刻度属性，确保 x 轴刻度旋转角度为 0 度
        _check_ticks_props(ax, xrot=0)
    # 测试函数，验证时区处理和绘图功能是否正常
    def test_ts_tz_area_lim_xcompat_secondary_y(self, ts):
        # 复制时间序列以避免修改原始数据
        tz_ts = ts.copy()
        # 将时间序列的索引设为GMT时区，并转换为CET时区
        tz_ts.index = tz_ts.tz_localize("GMT").tz_convert("CET")
        # 创建一个新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 绘制面积图，不堆叠，次要y轴
        ax = tz_ts.plot.area(stacked=False, secondary_y=True, ax=ax)
        # 获取x轴的范围
        xmin, xmax = ax.get_xlim()
        # 获取第一个线条的数据点
        line = ax.get_lines()[0].get_data(orig=False)[0]
        # 断言x轴范围包含第一个和最后一个数据点
        assert xmin <= line[0]
        assert xmax >= line[-1]
        # 检查并调整x轴刻度标签的属性
        _check_ticks_props(ax, xrot=0)

    # 测试函数，验证共享y轴的面积图是否正确绘制
    def test_area_sharey_dont_overwrite(self, ts):
        # GH37942
        # 创建一个包含两个子图的图形对象，共享y轴
        fig, (ax1, ax2) = mpl.pyplot.subplots(1, 2, sharey=True)

        # 绘制时间序列的绝对值面积图到第一个子图
        abs(ts).plot(ax=ax1, kind="area")
        # 绘制时间序列的绝对值面积图到第二个子图
        abs(ts).plot(ax=ax2, kind="area")

        # 断言两个子图的y轴已连接
        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)

    # 测试函数，验证标签功能是否正常
    def test_label(self):
        # 创建一个包含两个数值的序列
        s = Series([1, 2])
        # 创建一个新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 绘制序列s，并添加标签"LABEL"，显示图例
        ax = s.plot(label="LABEL", legend=True, ax=ax)
        # 检查图例标签是否正确设置
        _check_legend_labels(ax, labels=["LABEL"])

    # 测试函数，验证不设置标签时的默认行为
    def test_label_none(self):
        # 创建一个包含两个数值的序列
        s = Series([1, 2])
        # 创建一个新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 绘制序列s，显示图例
        ax = s.plot(legend=True, ax=ax)
        # 检查图例标签是否被设置为空字符串
        _check_legend_labels(ax, labels=[""])

    # 测试函数，验证序列名称作为标签时的行为
    def test_label_ser_name(self):
        # 创建一个包含两个数值的序列，设置名称为"NAME"
        s = Series([1, 2], name="NAME")
        # 创建一个新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 绘制序列s，显示图例
        ax = s.plot(legend=True, ax=ax)
        # 检查图例标签是否被设置为序列的名称"NAME"
        _check_legend_labels(ax, labels=["NAME"])

    # 测试函数，验证设置标签覆盖默认名称时的行为
    def test_label_ser_name_override(self):
        # 创建一个包含两个数值的序列，设置名称为"NAME"
        s = Series([1, 2], name="NAME")
        # 覆盖默认的标签设置
        # 创建一个新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 绘制序列s，并设置标签为"LABEL"，显示图例
        ax = s.plot(legend=True, label="LABEL", ax=ax)
        # 检查图例标签是否被设置为"LABEL"
        _check_legend_labels(ax, labels=["LABEL"])

    # 测试函数，验证设置标签但不绘制图例时的行为
    def test_label_ser_name_override_dont_draw(self):
        # 创建一个包含两个数值的序列，设置名称为"NAME"
        s = Series([1, 2], name="NAME")
        # 添加标签信息，但不绘制图例
        # 创建一个新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 绘制序列s，但不显示图例
        ax = s.plot(legend=False, label="LABEL", ax=ax)
        # 断言图例未被绘制
        assert ax.get_legend() is None  # 尚未被绘制
        ax.legend()  # 手动绘制图例
        # 检查图例标签是否被设置为"LABEL"
        _check_legend_labels(ax, labels=["LABEL"])

    # 测试函数，验证布尔类型数据绘制时的行为
    def test_boolean(self):
        # GH 23719
        # 创建一个包含布尔类型数据的序列
        s = Series([False, False, True])
        # 检查包含布尔类型数据时的绘图行为是否正常
        _check_plot_works(s.plot, include_bool=True)

        # 断言当序列只包含布尔类型数据时，是否抛出预期的类型错误异常
        msg = "no numeric data to plot"
        with pytest.raises(TypeError, match=msg):
            _check_plot_works(s.plot)

    # 使用pytest的参数化装饰器，验证不同索引设置的行为
    @pytest.mark.parametrize("index", [None, date_range("2020-01-01", periods=4)])
    # 测试处理包含 NaN 的 Series 的情况
    def test_line_area_nan_series(self, index):
        # 创建包含 NaN 值的 Series
        values = [1, 2, np.nan, 3]
        d = Series(values, index=index)
        # 调用 _check_plot_works 函数检查绘图功能
        ax = _check_plot_works(d.plot)
        # 获取第一个线条的数据并去除 NaN 值，用于比较
        masked = ax.lines[0].get_ydata()
        # 期望的数据，去除 NaN 后应为 [1, 2, 3]
        exp = np.array([1, 2, 3], dtype=np.float64)
        # 使用断言检查删除 NaN 值后的数组与期望的数组是否相等
        tm.assert_numpy_array_equal(np.delete(masked.data, 2), exp)
        # 检查 masked 对象的 mask 属性是否正确标记了 NaN 值的位置
        tm.assert_numpy_array_equal(masked.mask, np.array([False, False, True, False]))

        # 期望的数据，NaN 被替换为 0 后应为 [1, 2, 0, 3]
        expected = np.array([1, 2, 0, 3], dtype=np.float64)
        # 使用堆叠绘图测试 _check_plot_works 函数
        ax = _check_plot_works(d.plot, stacked=True)
        # 断言堆叠绘图后的第一个线条的数据与期望数据相等
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)
        # 使用区域图绘图测试 _check_plot_works 函数
        ax = _check_plot_works(d.plot.area)
        # 断言区域图绘图后的第一个线条的数据与期望数据相等
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)
        # 使用非堆叠区域图绘图测试 _check_plot_works 函数
        ax = _check_plot_works(d.plot.area, stacked=False)
        # 断言非堆叠区域图绘图后的第一个线条的数据与期望数据相等
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected)

    # 测试 Series 绘制时 use_index=False 的情况
    def test_line_use_index_false(self):
        # 创建具有自定义索引的 Series
        s = Series([1, 2, 3], index=["a", "b", "c"])
        # 设置索引的名称
        s.index.name = "The Index"
        # 创建新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 使用 use_index=False 参数绘制 Series
        ax = s.plot(use_index=False, ax=ax)
        # 检查 x 轴标签是否为空字符串
        label = ax.get_xlabel()
        assert label == ""

    # 测试 Series 绘制时 use_index=False 的情况（使用不同的绘图方法）
    def test_line_use_index_false_diff_var(self):
        # 创建具有自定义索引的 Series
        s = Series([1, 2, 3], index=["a", "b", "c"])
        # 设置索引的名称
        s.index.name = "The Index"
        # 创建新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 使用 use_index=False 参数绘制条形图
        ax2 = s.plot.bar(use_index=False, ax=ax)
        # 检查 x 轴标签是否为空字符串
        label2 = ax2.get_xlabel()
        assert label2 == ""

    # 测试在某些条件下的条形图绘制的 log 轴情况
    @pytest.mark.xfail(
        np_version_gte1p24 and is_platform_linux(),
        reason="Weird rounding problems",
        strict=False,
    )
    @pytest.mark.parametrize("axis, meth", [("yaxis", "bar"), ("xaxis", "barh")])
    def test_bar_log(self, axis, meth):
        # 期望的刻度值，用于测试 log 轴绘图
        expected = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])

        # 创建新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 使用 log=True 和指定的绘图方法绘制 Series 的条形图
        ax = getattr(Series([200, 500]).plot, meth)(log=True, ax=ax)
        # 断言获取的轴对象的刻度位置与期望的刻度值相等
        tm.assert_numpy_array_equal(getattr(ax, axis).get_ticklocs(), expected)

    # 测试在某些条件下的条形图绘制的 log 轴情况，并指定绘图类型为 bar 或 barh
    @pytest.mark.xfail(
        np_version_gte1p24 and is_platform_linux(),
        reason="Weird rounding problems",
        strict=False,
    )
    @pytest.mark.parametrize(
        "axis, kind, res_meth",
        [["yaxis", "bar", "get_ylim"], ["xaxis", "barh", "get_xlim"]],
    )
    def test_bar_log_kind_bar(self, axis, kind, res_meth):
        # GH 9905
        # 期望的刻度值，用于测试 log 轴绘图
        expected = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])

        # 创建新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 使用 log=True 和指定的绘图类型绘制 Series 的条形图
        ax = Series([0.1, 0.01, 0.001]).plot(log=True, kind=kind, ax=ax)
        # 检查获取的轴对象的限制方法的结果是否与预期相等
        ymin = 0.0007943282347242822
        ymax = 0.12589254117941673
        res = getattr(ax, res_meth)()
        tm.assert_almost_equal(res[0], ymin)
        tm.assert_almost_equal(res[1], ymax)
        # 断言获取的轴对象的刻度位置与期望的刻度值相等
        tm.assert_numpy_array_equal(getattr(ax, axis).get_ticklocs(), expected)
    # 定义测试函数，用于测试 bar 图的生成是否正确（忽略索引）
    def test_bar_ignore_index(self):
        # 创建 Series 对象
        df = Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
        # 创建子图对象
        _, ax = mpl.pyplot.subplots()
        # 生成 bar 图并指定不使用索引
        ax = df.plot.bar(use_index=False, ax=ax)
        # 检查 x 轴刻度标签是否正确
        _check_text_labels(ax.get_xticklabels(), ["0", "1", "2", "3"])

    # 定义测试函数，用于测试 bar 图指定颜色是否正确
    def test_bar_user_colors(self):
        # 创建 Series 对象
        s = Series([1, 2, 3, 4])
        # 生成带有指定颜色的 bar 图
        ax = s.plot.bar(color=["red", "blue", "blue", "red"])
        # 获取每个 bar 的颜色
        result = [p.get_facecolor() for p in ax.patches]
        # 期望的颜色值
        expected = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0, 1.0),
        ]
        # 断言实际颜色值与期望颜色值是否一致
        assert result == expected

    # 定义测试函数，用于测试默认旋转角度为 0 的情况
    def test_rotation_default(self):
        # 创建 DataFrame 对象
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 创建子图对象
        _, ax = mpl.pyplot.subplots()
        # 生成绘图，并使用默认旋转角度 0
        axes = df.plot(ax=ax)
        # 检查刻度属性是否正确设置
        _check_ticks_props(axes, xrot=0)

    # 定义测试函数，用于测试旋转角度为 30 的情况
    def test_rotation_30(self):
        # 创建 DataFrame 对象
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 创建子图对象
        _, ax = mpl.pyplot.subplots()
        # 生成绘图，并指定旋转角度为 30
        axes = df.plot(rot=30, ax=ax)
        # 检查刻度属性是否正确设置
        _check_ticks_props(axes, xrot=30)

    # 定义测试函数，用于测试不规则的日期时间索引情况
    def test_irregular_datetime(self):
        # 创建日期范围
        rng = date_range("1/1/2000", "1/15/2000")
        rng = rng[[0, 1, 2, 3, 5, 9, 10, 11, 12]]
        # 创建带有随机数据的 Series 对象
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), rng)
        # 创建子图对象
        _, ax = mpl.pyplot.subplots()
        # 生成绘图
        ax = ser.plot(ax=ax)
        # 将指定的日期时间转换成坐标轴的位置
        xp = DatetimeConverter.convert(datetime(1999, 1, 1), "", ax)
        # 设置 x 轴的显示范围
        ax.set_xlim("1/1/1999", "1/1/2001")
        # 断言转换后的位置与设置的 x 轴范围起始点是否一致
        assert xp == ax.get_xlim()[0]
        # 检查刻度属性是否正确设置
        _check_ticks_props(ax, xrot=30)

    # 定义测试函数，用于测试未排序索引情况下的 x 轴限制范围
    def test_unsorted_index_xlim(self):
        # 创建带有未排序索引的 Series 对象
        ser = Series(
            [0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            index=[1.0, 0.0, 3.0, 2.0, np.nan, 3.0, 2.0],
        )
        # 创建子图对象
        _, ax = mpl.pyplot.subplots()
        # 生成绘图
        ax = ser.plot(ax=ax)
        # 获取 x 轴的当前显示范围
        xmin, xmax = ax.get_xlim()
        # 获取线条对象
        lines = ax.get_lines()
        # 断言 x 轴的显示范围是否包含所有线条的数据范围
        assert xmin <= np.nanmin(lines[0].get_data(orig=False)[0])
        assert xmax >= np.nanmax(lines[0].get_data(orig=False)[0])

    # 定义测试函数，用于测试 Series 对象生成饼图的情况
    def test_pie_series(self):
        # 如果值的总和小于 1.0，将饼图视作比率并绘制半圆。
        # 创建带有随机整数值的 Series 对象
        series = Series(
            np.random.default_rng(2).integers(1, 5),
            index=["a", "b", "c", "d", "e"],
            name="YLABEL",
        )
        # 检查绘图是否正常工作，并返回绘图的 Axes 对象
        ax = _check_plot_works(series.plot.pie)
        # 检查文本标签是否正确设置
        _check_text_labels(ax.texts, series.index)
        # 断言 y 轴标签为空字符串
        assert ax.get_ylabel() == ""

    # 定义测试函数，用于测试 Series 对象生成无标签饼图的情况
    def test_pie_series_no_label(self):
        # 创建带有随机整数值的 Series 对象
        series = Series(
            np.random.default_rng(2).integers(1, 5),
            index=["a", "b", "c", "d", "e"],
            name="YLABEL",
        )
        # 检查绘图是否正常工作，并返回绘图的 Axes 对象，不显示标签
        ax = _check_plot_works(series.plot.pie, labels=None)
        # 检查文本标签是否为空字符串列表
        _check_text_labels(ax.texts, [""] * 5)
    # 测试用例：测试饼图绘制函数处理颜色少于元素数量的情况
    def test_pie_series_less_colors_than_elements(self):
        # 创建一个包含随机整数的 Series 对象，索引为指定的字母，名称为 YLABEL
        series = Series(
            np.random.default_rng(2).integers(1, 5),
            index=["a", "b", "c", "d", "e"],
            name="YLABEL",
        )
        # 定义颜色参数列表
        color_args = ["r", "g", "b"]
        # 调用辅助函数检查饼图绘制，传入颜色参数
        ax = _check_plot_works(series.plot.pie, colors=color_args)

        # 期望的颜色顺序，超出的元素会循环使用前面的颜色
        color_expected = ["r", "g", "b", "r", "g"]
        # 检查图形对象的色块颜色是否与期望一致
        _check_colors(ax.patches, facecolors=color_expected)

    # 测试用例：测试饼图绘制函数处理带标签和颜色参数的情况
    def test_pie_series_labels_and_colors(self):
        # 创建一个包含随机整数的 Series 对象，索引为指定的字母，名称为 YLABEL
        series = Series(
            np.random.default_rng(2).integers(1, 5),
            index=["a", "b", "c", "d", "e"],
            name="YLABEL",
        )
        # 定义标签列表和颜色参数列表
        labels = ["A", "B", "C", "D", "E"]
        color_args = ["r", "g", "b", "c", "m"]
        # 调用辅助函数检查饼图绘制，传入标签和颜色参数
        ax = _check_plot_works(series.plot.pie, labels=labels, colors=color_args)
        # 检查图形对象的文本标签是否与预期标签一致
        _check_text_labels(ax.texts, labels)
        # 检查图形对象的色块颜色是否与期望一致
        _check_colors(ax.patches, facecolors=color_args)

    # 测试用例：测试饼图绘制函数处理自动标签和字体大小参数的情况
    def test_pie_series_autopct_and_fontsize(self):
        # 创建一个包含随机整数的 Series 对象，索引为指定的字母，名称为 YLABEL
        series = Series(
            np.random.default_rng(2).integers(1, 5),
            index=["a", "b", "c", "d", "e"],
            name="YLABEL",
        )
        # 定义颜色参数列表
        color_args = ["r", "g", "b", "c", "m"]
        # 调用辅助函数检查饼图绘制，传入颜色参数、自动标签格式和字体大小
        ax = _check_plot_works(
            series.plot.pie, colors=color_args, autopct="%.2f", fontsize=7
        )
        # 计算每个扇形区域所占比例，并格式化为百分比字符串
        pcts = [f"{s*100:.2f}" for s in series.values / series.sum()]
        # 期望显示的文本标签，包括标签和百分比
        expected_texts = list(chain.from_iterable(zip(series.index, pcts)))
        # 检查图形对象的文本标签是否与期望一致
        _check_text_labels(ax.texts, expected_texts)
        # 检查所有文本标签的字体大小是否为指定的 7
        for t in ax.texts:
            assert t.get_fontsize() == 7

    # 测试用例：测试饼图绘制函数处理包含负值的情况，预期抛出 ValueError
    def test_pie_series_negative_raises(self):
        # 创建一个包含负数的 Series 对象，索引为指定的字母
        series = Series([1, 2, 0, 4, -1], index=["a", "b", "c", "d", "e"])
        # 使用 pytest 断言预期捕获 ValueError 异常，提示不允许负值进行饼图绘制
        with pytest.raises(ValueError, match="pie plot doesn't allow negative values"):
            series.plot.pie()

    # 测试用例：测试饼图绘制函数处理包含 NaN 值的情况
    def test_pie_series_nan(self):
        # 创建一个包含 NaN 值的 Series 对象，索引为指定的字母，名称为 YLABEL
        series = Series([1, 2, np.nan, 4], index=["a", "b", "c", "d"], name="YLABEL")
        # 调用辅助函数检查饼图绘制，传入 NaN 值的 Series
        ax = _check_plot_works(series.plot.pie)
        # 检查图形对象的文本标签是否与预期一致，NaN 值对应空字符串
        _check_text_labels(ax.texts, ["a", "b", "", "d"])

    # 测试用例：测试饼图绘制函数处理包含 NaN 值的 Series 对象
    def test_pie_nan(self):
        # 创建一个包含 NaN 值的 Series 对象
        s = Series([1, np.nan, 1, 1])
        # 创建图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 调用 Series 对象的饼图绘制函数，显示图例，并将轴对象传入
        ax = s.plot.pie(legend=True, ax=ax)
        # 期望的文本标签，NaN 值对应空字符串
        expected = ["0", "", "2", "3"]
        # 获取实际的文本标签内容列表
        result = [x.get_text() for x in ax.texts]
        # 使用断言验证实际结果与期望结果一致
        assert result == expected
    def test_df_series_secondary_legend(self):
        # GH 9779
        # 创建一个 DataFrame 包含 10 行 3 列的标准正态分布随机数，并设置列名为 "abc"
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=list("abc")
        )
        # 创建一个 Series 包含 10 个标准正态分布随机数，并命名为 "x"
        s = Series(np.random.default_rng(2).standard_normal(10), name="x")

        # 创建一个新的图表，并返回图表对象 ax
        _, ax = mpl.pyplot.subplots()
        # 在 ax 上绘制 DataFrame df 的图表
        ax = df.plot(ax=ax)
        # 在同一个 ax 上绘制 Series s 的图表，设置图例显示和次要 Y 轴
        s.plot(legend=True, secondary_y=True, ax=ax)
        # 检查图表上的图例标签，确保包含 ["a", "b", "c", "x (right)"]
        _check_legend_labels(ax, labels=["a", "b", "c", "x (right)"])
        # 断言主 Y 轴和次要 Y 轴是可见的
        assert ax.get_yaxis().get_visible()
        assert ax.right_ax.get_yaxis().get_visible()

    def test_df_series_secondary_legend_both(self):
        # GH 9779
        # 创建一个 DataFrame 包含 10 行 3 列的标准正态分布随机数，并设置列名为 "abc"
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=list("abc")
        )
        # 创建一个 Series 包含 10 个标准正态分布随机数，并命名为 "x"
        s = Series(np.random.default_rng(2).standard_normal(10), name="x")

        # 创建一个新的图表，并返回图表对象 ax
        _, ax = mpl.pyplot.subplots()
        # 在 ax 上绘制 DataFrame df 的图表，设置次要 Y 轴
        ax = df.plot(secondary_y=True, ax=ax)
        # 在同一个 ax 上绘制 Series s 的图表，设置图例显示和次要 Y 轴
        s.plot(legend=True, secondary_y=True, ax=ax)
        # 检查图表左边的次要 Y 轴上的图例标签，确保包含 ["a (right)", "b (right)", "c (right)", "x (right)"]
        _check_legend_labels(ax.left_ax, labels=["a (right)", "b (right)", "c (right)", "x (right)"])
        # 断言图表左边的主 Y 轴不可见，右边的主 Y 轴可见
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()

    def test_df_series_secondary_legend_both_with_axis_2(self):
        # GH 9779
        # 创建一个 DataFrame 包含 10 行 3 列的标准正态分布随机数，并设置列名为 "abc"
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=list("abc")
        )
        # 创建一个 Series 包含 10 个标准正态分布随机数，并命名为 "x"
        s = Series(np.random.default_rng(2).standard_normal(10), name="x")

        # 创建一个新的图表，并返回图表对象 ax
        _, ax = mpl.pyplot.subplots()
        # 在 ax 上绘制 DataFrame df 的图表，设置次要 Y 轴，并标记右边的轴不显示
        ax = df.plot(secondary_y=True, mark_right=False, ax=ax)
        # 在同一个 ax 上绘制 Series s 的图表，设置图例显示和次要 Y 轴
        s.plot(ax=ax, legend=True, secondary_y=True)
        # 检查图表左边的次要 Y 轴上的图例标签，确保包含 ["a", "b", "c", "x (right)"]
        _check_legend_labels(ax.left_ax, ["a", "b", "c", "x (right)"])
        # 断言图表左边的主 Y 轴不可见，右边的主 Y 轴可见
        assert not ax.left_ax.get_yaxis().get_visible()
        assert ax.get_yaxis().get_visible()

    @pytest.mark.parametrize(
        "input_logy, expected_scale", [(True, "log"), ("sym", "symlog")]
    )
    @pytest.mark.parametrize("secondary_kwarg", [{}, {"secondary_y": True}])
    def test_secondary_logy(self, input_logy, expected_scale, secondary_kwarg):
        # GH 25545, GH 24980
        # 创建一个包含 10 个标准正态分布随机数的 Series
        s1 = Series(np.random.default_rng(2).standard_normal(10))
        # 在 s1 上绘制图表，设置 Y 轴为对数或对称对数
        ax1 = s1.plot(logy=input_logy, **secondary_kwarg)
        # 断言获取的 Y 轴比例尺与预期比例尺相同
        assert ax1.get_yscale() == expected_scale
    # 测试当指定了重复的颜色和样式时是否会引发异常
    def test_plot_fails_with_dupe_color_and_style(self):
        # 创建一个包含两个随机标准正态分布值的序列
        x = Series(np.random.default_rng(2).standard_normal(2))
        # 创建一个新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 异常消息内容，指出不能同时使用带有颜色符号的样式字符串和 'color' 关键字参数
        msg = (
            "Cannot pass 'style' string with a color symbol and 'color' keyword "
            "argument. Please use one or the other or pass 'style' without a color "
            "symbol"
        )
        # 使用 pytest 检查是否引发 ValueError 异常，并验证异常消息内容
        with pytest.raises(ValueError, match=msg):
            # 在指定的轴上绘制 x 的数据，使用 'k--' 样式和 'k' 颜色
            x.plot(style="k--", color="k", ax=ax)

    # 使用不同的参数组合对 test_kde_kwargs 函数进行参数化测试
    @pytest.mark.parametrize(
        "bw_method, ind",
        [
            ["scott", 20],
            [None, 20],
            [None, np.int_(20)],
            [0.5, np.linspace(-100, 100, 20)],
        ],
    )
    # 测试 KDE 图的参数传递情况
    def test_kde_kwargs(self, ts, bw_method, ind):
        # 导入 scipy 库，如未成功导入则跳过该测试
        pytest.importorskip("scipy")
        # 调用 _check_plot_works 函数，检查是否能成功绘制 KDE 图
        _check_plot_works(ts.plot.kde, bw_method=bw_method, ind=ind)

    # 测试 Density 图的参数传递情况
    def test_density_kwargs(self, ts):
        # 导入 scipy 库，如未成功导入则跳过该测试
        pytest.importorskip("scipy")
        # 创建一个包含 20 个均匀分布样本点的数组
        sample_points = np.linspace(-100, 100, 20)
        # 调用 _check_plot_works 函数，检查是否能成功绘制 Density 图
        _check_plot_works(ts.plot.density, bw_method=0.5, ind=sample_points)

    # 测试带有指定参数的 KDE 图是否能正确绘制，并验证轴的缩放情况和文本标签
    def test_kde_kwargs_check_axes(self, ts):
        # 导入 scipy 库，如未成功导入则跳过该测试
        pytest.importorskip("scipy")
        # 创建一个新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 创建一个包含 20 个均匀分布样本点的数组
        sample_points = np.linspace(-100, 100, 20)
        # 在指定的轴上绘制 KDE 图，使用对数刻度、0.5 的 bw_method 和指定的样本点数组
        ax = ts.plot.kde(logy=True, bw_method=0.5, ind=sample_points, ax=ax)
        # 检查轴的比例是否符合预期，y 轴应为对数刻度
        _check_ax_scales(ax, yaxis="log")
        # 检查 y 轴标签是否为 "Density"
        _check_text_labels(ax.yaxis.get_label(), "Density")

    # 测试带有缺失值的 KDE 图是否能成功绘制
    def test_kde_missing_vals(self):
        # 导入 scipy 库，如未成功导入则跳过该测试
        pytest.importorskip("scipy")
        # 创建一个包含 50 个均匀分布随机数的序列，并将第一个值设为 NaN
        s = Series(np.random.default_rng(2).uniform(size=50))
        s[0] = np.nan
        # 调用 _check_plot_works 函数，检查是否能成功绘制 KDE 图，并返回轴对象
        axes = _check_plot_works(s.plot.kde)

        # 检查 KDE 图的线条数据是否包含任何缺失值
        assert any(~np.isnan(axes.lines[0].get_xdata()))

    # 根据指定的参数绘制箱线图，并验证轴的缩放情况和文本标签
    @pytest.mark.xfail(reason="Api changed in 3.6.0")
    def test_boxplot_series(self, ts):
        # 创建一个新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 在指定的轴上绘制箱线图，使用对数刻度
        ax = ts.plot.box(logy=True, ax=ax)
        # 检查轴的比例是否符合预期，y 轴应为对数刻度
        _check_ax_scales(ax, yaxis="log")
        # 获取 x 轴刻度的文本标签，并验证其内容与序列的名称相符
        xlabels = ax.get_xticklabels()
        _check_text_labels(xlabels, [ts.name])
        # 获取 y 轴刻度的文本标签，并验证其内容为空字符串
        ylabels = ax.get_yticklabels()
        _check_text_labels(ylabels, [""] * len(ylabels))

    # 使用不同的图类型参数对 test_kind_kwarg 函数进行参数化测试
    @pytest.mark.parametrize(
        "kind",
        plotting.PlotAccessor._common_kinds + plotting.PlotAccessor._series_kinds,
    )
    # 测试 plot 函数的 kind 参数传递情况
    def test_kind_kwarg(self, kind):
        # 导入 scipy 库，如未成功导入则跳过该测试
        pytest.importorskip("scipy")
        # 创建一个包含 0 到 2 的整数序列
        s = Series(range(3))
        # 创建一个新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 在指定的轴上绘制 plot 图，使用不同的 kind 参数
        s.plot(kind=kind, ax=ax)
        # 关闭当前的图形
        mpl.pyplot.close()

    # 使用不同的图类型参数对 test_kind_attr 函数进行参数化测试
    @pytest.mark.parametrize(
        "kind",
        plotting.PlotAccessor._common_kinds + plotting.PlotAccessor._series_kinds,
    )
    # 测试 plot 对象的属性方法调用情况
    def test_kind_attr(self, kind):
        # 导入 scipy 库，如未成功导入则跳过该测试
        pytest.importorskip("scipy")
        # 创建一个包含 0 到 2 的整数序列
        s = Series(range(3))
        # 创建一个新的图形和轴对象
        _, ax = mpl.pyplot.subplots()
        # 调用 plot 对象的 kind 方法，使用不同的 kind 参数
        getattr(s.plot, kind)()
        # 关闭当前的图形
        mpl.pyplot.close()

    # 使用 plotting.PlotAccessor._common_kinds 参数化测试 kind 参数
    @pytest.mark.parametrize("kind", plotting.PlotAccessor._common_kinds)
    # 定义一个测试方法，用于验证当数据类型不合适时，是否会抛出TypeError异常
    def test_invalid_plot_data(self, kind):
        # 创建一个Series对象，包含字符列表["a", "b", "c", "d"]
        s = Series(list("abcd"))
        # 创建一个新的图表和坐标轴对象
        _, ax = mpl.pyplot.subplots()
        # 错误消息字符串
        msg = "no numeric data to plot"
        # 使用pytest检查是否会抛出TypeError异常，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            # 调用Series对象的plot方法，预期会抛出异常
            s.plot(kind=kind, ax=ax)

    # 使用pytest的参数化标记，定义一个测试方法，测试对象数据类型为object时的绘图情况
    @pytest.mark.parametrize("kind", plotting.PlotAccessor._common_kinds)
    def test_valid_object_plot(self, kind):
        # 导入scipy库，如果导入失败则跳过测试
        pytest.importorskip("scipy")
        # 创建一个包含10个对象类型数据的Series对象
        s = Series(range(10), dtype=object)
        # 调用辅助函数检查绘图是否正常工作
        _check_plot_works(s.plot, kind=kind)

    # 使用pytest的参数化标记，定义一个测试方法，验证部分数据类型不合适时是否会抛出TypeError异常
    @pytest.mark.parametrize("kind", plotting.PlotAccessor._common_kinds)
    def test_partially_invalid_plot_data(self, kind):
        # 创建一个包含字符串和数字混合类型数据的Series对象
        s = Series(["a", "b", 1.0, 2])
        # 创建一个新的图表和坐标轴对象
        _, ax = mpl.pyplot.subplots()
        # 错误消息字符串
        msg = "no numeric data to plot"
        # 使用pytest检查是否会抛出TypeError异常，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            # 调用Series对象的plot方法，预期会抛出异常
            s.plot(kind=kind, ax=ax)

    # 定义一个测试方法，验证当指定无效的绘图类型时，是否会抛出ValueError异常
    def test_invalid_kind(self):
        # 创建一个包含整数数据的Series对象
        s = Series([1, 2])
        # 使用pytest检查是否会抛出ValueError异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match="invalid_kind is not a valid plot kind"):
            # 调用Series对象的plot方法，预期会抛出异常
            s.plot(kind="invalid_kind")

    # 定义一个测试方法，验证当日期索引存在重复时，是否能正常绘制图表
    def test_dup_datetime_index_plot(self):
        # 创建两个日期范围对象
        dr1 = date_range("1/1/2009", periods=4)
        dr2 = date_range("1/2/2009", periods=4)
        # 将两个日期范围对象合并成一个索引对象
        index = dr1.append(dr2)
        # 创建一个与索引相同长度的随机数值数组
        values = np.random.default_rng(2).standard_normal(index.size)
        # 创建一个带有重复日期索引的Series对象
        s = Series(values, index=index)
        # 调用辅助函数检查绘图是否正常工作
        _check_plot_works(s.plot)

    # 定义一个测试方法，验证当使用不对称误差条时，是否会抛出ValueError异常
    def test_errorbar_asymmetrical(self):
        # 创建一个包含整数数据的Series对象
        s = Series(np.arange(10), name="x")
        # 创建一个随机的不对称误差数组
        err = np.random.default_rng(2).random((2, 10))
        # 使用plot方法绘制误差条图表，并获取生成的图表对象
        ax = s.plot(yerr=err, xerr=err)
        # 从图表对象中获取特定路径的顶点信息，用于后续比较
        result = np.vstack([i.vertices[:, 1] for i in ax.collections[1].get_paths()])
        # 计算预期结果，与实际结果比较
        expected = (err.T * np.array([-1, 1])) + s.to_numpy().reshape(-1, 1)
        # 使用pytest断言检查结果是否符合预期
        tm.assert_numpy_array_equal(result, expected)

    # 定义一个测试方法，验证当使用不对称误差条但提供了错误的形状时，是否会抛出ValueError异常
    def test_errorbar_asymmetrical_error(self):
        # 创建一个包含整数数据的Series对象
        s = Series(np.arange(10), name="x")
        # 构造错误消息字符串
        msg = (
            "Asymmetrical error bars should be provided "
            f"with the shape \\(2, {len(s)}\\)"
        )
        # 使用pytest检查是否会抛出ValueError异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match=msg):
            # 调用Series对象的plot方法，预期会抛出异常
            s.plot(yerr=np.random.default_rng(2).random((2, 11)))

    # 使用pytest的参数化标记，定义一个测试方法，测试带有误差条的绘图情况
    @pytest.mark.parametrize("kind", ["line", "bar"])
    @pytest.mark.parametrize(
        "yerr",
        [
            Series(np.abs(np.random.default_rng(2).standard_normal(10))),
            np.abs(np.random.default_rng(2).standard_normal(10)),
            list(np.abs(np.random.default_rng(2).standard_normal(10))),
            DataFrame(
                np.abs(np.random.default_rng(2).standard_normal((10, 2))),
                columns=["x", "y"],
            ),
        ],
    )
    def test_errorbar_plot(self, kind, yerr):
        # 创建一个包含整数数据的Series对象
        s = Series(np.arange(10), name="x")
        # 调用辅助函数检查绘图是否正常工作，并验证误差条的存在性
        ax = _check_plot_works(s.plot, yerr=yerr, kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=1)

    # 使用pytest的慢速标记，此处不再有代码
    # 定义测试函数，用于测试在 y 方向误差为 0 的情况下绘制误差条图形
    def test_errorbar_plot_yerr_0(self):
        # 创建一个包含 0 到 9 的 Series 对象
        s = Series(np.arange(10), name="x")
        # 生成一个长度为 10 的随机误差值数组
        s_err = np.abs(np.random.default_rng(2).standard_normal(10))
        # 调用 _check_plot_works 函数绘制图形，并传入 x 方向的误差条数据
        ax = _check_plot_works(s.plot, xerr=s_err)
        # 检查生成的图形是否包含 x 方向误差条为 1、y 方向误差条为 0
        _check_has_errorbars(ax, xerr=1, yerr=0)

    # 标记为慢速测试，参数化测试函数
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "yerr",
        [
            Series(np.abs(np.random.default_rng(2).standard_normal(12))),  # 创建包含 12 个随机误差值的 Series 对象
            DataFrame(  # 创建包含 12 行 2 列的随机误差值的 DataFrame 对象
                np.abs(np.random.default_rng(2).standard_normal((12, 2))),
                columns=["x", "y"],
            ),
        ],
    )
    # 测试时间序列绘图
    def test_errorbar_plot_ts(self, yerr):
        # 生成包含 12 个日期的时间索引
        ix = date_range("1/1/2000", "1/1/2001", freq="ME")
        # 创建一个 Series 对象，其索引为 ix，值为 0 到 11
        ts = Series(np.arange(12), index=ix, name="x")
        # 将 yerr 的索引设置为 ix
        yerr.index = ix

        # 调用 _check_plot_works 函数绘制时间序列图形，并传入 y 方向的误差条数据
        ax = _check_plot_works(ts.plot, yerr=yerr)
        # 检查生成的图形是否包含 x 方向误差条为 0、y 方向误差条为 1
        _check_has_errorbars(ax, xerr=0, yerr=1)

    # 标记为慢速测试，测试无效的 y 方向误差数据形状
    def test_errorbar_plot_invalid_yerr_shape(self):
        # 创建一个包含 0 到 9 的 Series 对象
        s = Series(np.arange(10), name="x")
        # 使用错误的长度和类型的 yerr 数据进行绘图，检查是否会引发 ValueError
        with tm.external_error_raised(ValueError):
            s.plot(yerr=np.arange(11))

    # 标记为慢速测试，测试无效的 y 方向误差数据类型
    def test_errorbar_plot_invalid_yerr(self):
        # 创建一个包含 0 到 9 的 Series 对象
        s = Series(np.arange(10), name="x")
        # 创建一个长度为 10 的字符串列表作为 yerr 数据，检查是否会引发 TypeError
        s_err = ["zzz"] * 10
        with tm.external_error_raised(TypeError):
            s.plot(yerr=s_err)

    # 标记为慢速测试，测试在绘制表格时的情况
    def test_table_true(self, series):
        # 调用 _check_plot_works 函数绘制包含表格的图形
        _check_plot_works(series.plot, table=True)

    # 标记为慢速测试，测试在绘制自身时的情况
    def test_table_self(self, series):
        # 调用 _check_plot_works 函数绘制包含自身的表格图形
        _check_plot_works(series.plot, table=series)

    # 标记为慢速测试，测试 Series 对象的网格设置
    def test_series_grid_settings(self):
        # 导入 scipy 库，用于测试默认是否正确设置了 rcParams['axes.grid']，GH 9792
        pytest.importorskip("scipy")
        # 调用 _check_grid_settings 函数检查网格设置
        _check_grid_settings(
            Series([1, 2, 3]),
            plotting.PlotAccessor._series_kinds + plotting.PlotAccessor._common_kinds,
        )

    # 参数化测试函数，测试标准颜色的获取
    @pytest.mark.parametrize("c", ["r", "red", "green", "#FF0000"])
    def test_standard_colors(self, c):
        # 测试 get_standard_colors 函数，确保给定颜色 c 返回正确结果
        result = get_standard_colors(1, color=c)
        assert result == [c]

        # 测试 get_standard_colors 函数，确保给定颜色列表 [c] 返回正确结果
        result = get_standard_colors(1, color=[c])
        assert result == [c]

        # 测试 get_standard_colors 函数，确保给定颜色 c 返回正确结果的多重复制
        result = get_standard_colors(3, color=c)
        assert result == [c] * 3

        # 测试 get_standard_colors 函数，确保给定颜色列表 [c] 返回正确结果的多重复制
        result = get_standard_colors(3, color=[c])
        assert result == [c] * 3
    def test_standard_colors_all(self):
        # 测试获取标准颜色函数对各种颜色输入的处理

        # 对于每个颜色名称如 'mediumaquamarine'
        for c in mpl.colors.cnames:
            # 测试获取一个颜色的情况，期望结果是列表包含该颜色名
            result = get_standard_colors(num_colors=1, color=c)
            assert result == [c]

            # 测试获取一个颜色的情况（作为列表输入），期望结果同上
            result = get_standard_colors(num_colors=1, color=[c])
            assert result == [c]

            # 测试获取三个相同颜色的情况，期望结果是包含三个相同颜色名的列表
            result = get_standard_colors(num_colors=3, color=c)
            assert result == [c] * 3

            # 测试获取三个相同颜色的情况（作为列表输入），期望结果同上
            result = get_standard_colors(num_colors=3, color=[c])
            assert result == [c] * 3

        # 对于每个单字母颜色如 'k'
        for c in mpl.colors.ColorConverter.colors:
            # 同上四个测试用例，分别验证单字母颜色的不同输入情况
            result = get_standard_colors(num_colors=1, color=c)
            assert result == [c]

            result = get_standard_colors(num_colors=1, color=[c])
            assert result == [c]

            result = get_standard_colors(num_colors=3, color=c)
            assert result == [c] * 3

            result = get_standard_colors(num_colors=3, color=[c])
            assert result == [c] * 3

    def test_series_plot_color_kwargs(self):
        # 测试时间序列绘图的颜色关键字参数处理（GH1890）

        # 创建子图并绘制一系列数据，期望线条颜色为绿色
        _, ax = mpl.pyplot.subplots()
        ax = Series(np.arange(12) + 1).plot(color="green", ax=ax)
        _check_colors(ax.get_lines(), linecolors=["green"])

    def test_time_series_plot_color_kwargs(self):
        # 测试时间序列绘图的颜色关键字参数处理（#1890）

        # 创建子图并绘制一系列时间序列数据，期望线条颜色为绿色
        _, ax = mpl.pyplot.subplots()
        ax = Series(np.arange(12) + 1, index=date_range("1/1/2000", periods=12)).plot(
            color="green", ax=ax
        )
        _check_colors(ax.get_lines(), linecolors=["green"])

    def test_time_series_plot_color_with_empty_kwargs(self):
        # 测试时间序列绘图的颜色处理，使用默认参数情况

        # 获取默认颜色循环器，创建子图并绘制多个时间序列数据，期望使用默认颜色
        def_colors = _unpack_cycler(mpl.rcParams)
        index = date_range("1/1/2000", periods=12)
        s = Series(np.arange(1, 13), index=index)

        ncolors = 3

        _, ax = mpl.pyplot.subplots()
        for i in range(ncolors):
            ax = s.plot(ax=ax)
        _check_colors(ax.get_lines(), linecolors=def_colors[:ncolors])

    def test_xticklabels(self):
        # 测试 x 轴刻度标签的处理（GH11529）

        # 创建带有自定义索引的系列数据，绘制并指定部分 x 轴刻度，验证其标签是否符合预期
        s = Series(np.arange(10), index=[f"P{i:02d}" for i in range(10)])
        _, ax = mpl.pyplot.subplots()
        ax = s.plot(xticks=[0, 3, 5, 9], ax=ax)
        exp = [f"P{i:02d}" for i in [0, 3, 5, 9]]
        _check_text_labels(ax.get_xticklabels(), exp)

    def test_xtick_barPlot(self):
        # 测试条形图 x 轴刻度的处理（GH28172）

        # 创建带有自定义索引的系列数据，绘制为条形图并指定 x 轴刻度，验证其刻度是否符合预期
        s = Series(range(10), index=[f"P{i:02d}" for i in range(10)])
        ax = s.plot.bar(xticks=range(0, 11, 2))
        exp = np.array(list(range(0, 11, 2)))
        tm.assert_numpy_array_equal(exp, ax.get_xticks())

    def test_custom_business_day_freq(self):
        # 测试自定义工作日频率的处理（GH7222）

        # 创建带有自定义工作日频率的时间序列数据，验证绘图函数是否能正常工作
        s = Series(
            range(100, 121),
            index=pd.bdate_range(
                start="2014-05-01",
                end="2014-06-01",
                freq=CustomBusinessDay(holidays=["2014-05-26"]),
            ),
        )

        _check_plot_works(s.plot)
    # 使用 pytest.mark.xfail 标记此测试为预期失败，包含失败原因和相关的 GitHub 提交链接
    @pytest.mark.xfail(
        reason="GH#24426, see also "
        "github.com/pandas-dev/pandas/commit/"
        "ef1bd69fa42bbed5d09dd17f08c44fc8bfc2b685#r61470674"
    )
    # 定义一个测试函数，测试在 inplace 操作时绘图访问器的更新情况
    def test_plot_accessor_updates_on_inplace(self):
        # 创建一个包含 [1, 2, 3, 4] 的 Series 对象
        ser = Series([1, 2, 3, 4])
        # 创建一个新的 Figure 和 Axes 对象
        _, ax = mpl.pyplot.subplots()
        # 在给定的 Axes 上绘制 Series 对象 ser 的内容
        ax = ser.plot(ax=ax)
        # 获取绘图中 x 轴的刻度位置
        before = ax.xaxis.get_ticklocs()

        # 在 inplace 操作后绘制更新后的 Series 对象
        ser.drop([0, 1], inplace=True)
        _, ax = mpl.pyplot.subplots()
        # 获取更新后的 x 轴刻度位置
        after = ax.xaxis.get_ticklocs()
        # 使用测试框架的方法验证两次操作后的刻度位置是否一致
        tm.assert_numpy_array_equal(before, after)

    # 使用 pytest.mark.parametrize 标记参数化测试函数，测试不同的绘图类型对于 Series 的 xlim 设置
    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_plot_xlim_for_series(self, kind):
        # 创建一个包含 [2, 3] 的 Series 对象
        s = Series([2, 3])
        _, ax = mpl.pyplot.subplots()
        # 根据指定的 kind 绘制 Series 对象 s 的内容到给定的 Axes 上
        s.plot(kind=kind, ax=ax)
        # 获取绘图中 x 轴的限制范围
        xlims = ax.get_xlim()

        # 使用断言验证 xlim 的范围是否符合预期：小于 0 和大于 1
        assert xlims[0] < 0
        assert xlims[1] > 1

    # 定义一个测试函数，验证在空 Series 对象上进行绘图的行为
    def test_plot_no_rows(self):
        # 创建一个空的 Series 对象，数据类型为整数
        df = Series(dtype=int)
        # 使用断言验证 Series 是否为空
        assert df.empty
        # 在空的 Series 上进行绘图，获取返回的 Axes 对象
        ax = df.plot()
        # 使用断言验证绘图中是否只有一条线
        assert len(ax.get_lines()) == 1
        # 获取绘图中的第一条线
        line = ax.get_lines()[0]
        # 使用断言验证该线的 x 和 y 数据长度是否为 0
        assert len(line.get_xdata()) == 0
        assert len(line.get_ydata()) == 0

    # 定义一个测试函数，验证在非数值数据的 Series 对象上进行绘图时的行为
    def test_plot_no_numeric_data(self):
        # 创建一个包含非数值数据的 Series 对象
        df = Series(["a", "b", "c"])
        # 使用 pytest.raises 验证在绘制时是否会抛出 TypeError 异常，异常信息包含指定文本
        with pytest.raises(TypeError, match="no numeric data to plot"):
            df.plot()

    # 使用 pytest.mark.parametrize 标记参数化测试函数，验证 Series 的绘图顺序
    @pytest.mark.parametrize(
        "data, index",
        [
            ([1, 2, 3, 4], [3, 2, 1, 0]),
            ([10, 50, 20, 30], [1910, 1920, 1980, 1950]),
        ],
    )
    def test_plot_order(self, data, index):
        # 创建一个根据给定数据和索引创建的 Series 对象
        ser = Series(data=data, index=index)
        # 在 Axes 上绘制 Series 的条形图
        ax = ser.plot(kind="bar")

        # 期望的绘图结果，即按照索引排序后的数据列表
        expected = ser.tolist()
        # 获取绘图中条形对象的 y 轴最大坐标，根据 x 轴最大坐标排序
        result = [
            patch.get_bbox().ymax
            for patch in sorted(ax.patches, key=lambda patch: patch.get_bbox().xmax)
        ]
        # 使用断言验证期望结果与实际结果是否一致
        assert expected == result

    # 定义一个测试函数，验证绘图的样式是否设置正确
    def test_style_single_ok(self):
        # 创建一个包含 [1, 2] 的 Series 对象
        s = Series([1, 2])
        # 在指定样式和颜色下绘制 Series 对象 s 的内容，并获取返回的 Axes 对象
        ax = s.plot(style="s", color="C3")
        # 使用断言验证绘图中第一条线的颜色是否为 "C3"
        assert ax.lines[0].get_color() == "C3"

    # 使用 pytest.mark.parametrize 标记参数化测试函数，测试不同类型的绘图行为
    @pytest.mark.parametrize(
        "index_name, old_label, new_label",
        [(None, "", "new"), ("old", "old", "new"), (None, "", "")],
    )
    @pytest.mark.parametrize("kind", ["line", "area", "bar", "barh", "hist"])
    def test_xlabel_ylabel_series(self, kind, index_name, old_label, new_label):
        # GH 9093
        # 创建一个包含数据 [1, 2, 3, 4] 的 Series 对象
        ser = Series([1, 2, 3, 4])
        # 设置 Series 对象的索引名称为给定的 index_name
        ser.index.name = index_name

        # 根据不同的图表类型设置不同的默认行为，例如 barh 类型的图表
        ax = ser.plot(kind=kind)
        if kind == "barh":
            # 断言横条形图中的 xlabel 为空字符串，ylabel 为旧的标签 old_label
            assert ax.get_xlabel() == ""
            assert ax.get_ylabel() == old_label
        elif kind == "hist":
            # 断言直方图的 xlabel 为空字符串，ylabel 为 "Frequency"
            assert ax.get_xlabel() == ""
            assert ax.get_ylabel() == "Frequency"
        else:
            # 断言其他类型图表的 ylabel 为空字符串，xlabel 为旧的标签 old_label
            assert ax.get_ylabel() == ""
            assert ax.get_xlabel() == old_label

        # 覆盖旧的 xlabel，并使用新的标签 new_label 作为 ylabel
        ax = ser.plot(kind=kind, ylabel=new_label, xlabel=new_label)
        # 断言新的 ylabel 和 xlabel 都为 new_label
        assert ax.get_ylabel() == new_label
        assert ax.get_xlabel() == new_label

    @pytest.mark.parametrize(
        "index",
        [
            pd.timedelta_range(start=0, periods=2, freq="D"),
            [pd.Timedelta(days=1), pd.Timedelta(days=2)],
        ],
    )
    def test_timedelta_index(self, index):
        # GH37454
        # 设置 xlims 为 (3, 1)
        xlims = (3, 1)
        # 创建带有时间增量索引的 Series 对象，并绘制它，设置 x 轴的限制为 xlims
        ax = Series([1, 2], index=index).plot(xlim=xlims)
        # 断言获取到的 x 轴限制与预期的 xlims 相符
        assert ax.get_xlim() == (3, 1)

    def test_series_none_color(self):
        # GH51953
        # 创建包含数据 [1, 2, 3] 的 Series 对象，并绘制它，设置颜色为 None
        series = Series([1, 2, 3])
        ax = series.plot(color=None)
        # 获取预期的默认颜色，与图中线条的颜色进行比较
        expected = _unpack_cycler(mpl.pyplot.rcParams)[:1]
        _check_colors(ax.get_lines(), linecolors=expected)

    @pytest.mark.slow
    def test_plot_no_warning(self, ts):
        # GH 55138
        # TODO(3.0): this can be removed once Period[B] deprecation is enforced
        # 禁止产生警告信息，验证时间序列 ts 的绘制是否正常进行
        with tm.assert_produces_warning(False):
            _ = ts.plot()
```