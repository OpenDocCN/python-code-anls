# `D:\src\scipysrc\pandas\pandas\tests\plotting\frame\test_frame.py`

```
"""Test cases for DataFrame.plot"""

from datetime import (
    date,  # 导入日期类型
    datetime,  # 导入日期时间类型
)
import gc  # 导入垃圾回收模块
import itertools  # 导入迭代工具模块
import re  # 导入正则表达式模块
import string  # 导入字符串处理模块
import weakref  # 导入弱引用模块

import numpy as np  # 导入NumPy库
import pytest  # 导入Pytest测试框架

import pandas.util._test_decorators as td  # 导入Pandas测试装饰器

from pandas.core.dtypes.api import is_list_like  # 导入检查是否为列表型的函数

import pandas as pd  # 导入Pandas库
from pandas import (
    DataFrame,  # 导入DataFrame类
    Index,  # 导入Index类
    MultiIndex,  # 导入MultiIndex类
    PeriodIndex,  # 导入PeriodIndex类
    Series,  # 导入Series类
    bdate_range,  # 导入工作日范围函数
    date_range,  # 导入日期范围函数
    option_context,  # 导入选项上下文管理器
    plotting,  # 导入绘图函数
)
import pandas._testing as tm  # 导入Pandas测试模块
from pandas.tests.plotting.common import (
    _check_ax_scales,  # 导入检查坐标轴比例函数
    _check_axes_shape,  # 导入检查坐标轴形状函数
    _check_box_return_type,  # 导入检查箱线图返回类型函数
    _check_colors,  # 导入检查颜色函数
    _check_data,  # 导入检查数据函数
    _check_grid_settings,  # 导入检查网格设置函数
    _check_has_errorbars,  # 导入检查是否有误差条函数
    _check_legend_labels,  # 导入检查图例标签函数
    _check_plot_works,  # 导入检查绘图是否工作函数
    _check_text_labels,  # 导入检查文本标签函数
    _check_ticks_props,  # 导入检查刻度属性函数
    _check_visible,  # 导入检查可见性函数
    get_y_axis,  # 导入获取Y轴函数
)

from pandas.io.formats.printing import pprint_thing  # 导入打印格式函数

mpl = pytest.importorskip("matplotlib")  # 导入并检查Matplotlib库
plt = pytest.importorskip("matplotlib.pyplot")  # 导入并检查Matplotlib的Pyplot模块


class TestDataFramePlots:
    @pytest.mark.slow
    def test_plot(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),  # 创建一个10行4列的随机标准正态分布DataFrame
            columns=Index(list("ABCD"), dtype=object),  # 指定列索引为'A', 'B', 'C', 'D'
            index=date_range("2000-01-01", periods=10, freq="B"),  # 指定行索引为工作日频率的日期范围
        )
        _check_plot_works(df.plot, grid=False)  # 检查绘图函数是否正常工作，不显示网格

    @pytest.mark.slow
    def test_plot_subplots(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),  # 创建一个10行4列的随机标准正态分布DataFrame
            columns=Index(list("ABCD"), dtype=object),  # 指定列索引为'A', 'B', 'C', 'D'
            index=date_range("2000-01-01", periods=10, freq="B"),  # 指定行索引为工作日频率的日期范围
        )
        # _check_plot_works添加了一个ax参数，因此使用default_axes=True以避免警告
        axes = _check_plot_works(df.plot, default_axes=True, subplots=True)  # 检查绘图函数是否正常工作，创建子图
        _check_axes_shape(axes, axes_num=4, layout=(4, 1))  # 检查子图的形状，应为4行1列

    @pytest.mark.slow
    def test_plot_subplots_negative_layout(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),  # 创建一个10行4列的随机标准正态分布DataFrame
            columns=Index(list("ABCD"), dtype=object),  # 指定列索引为'A', 'B', 'C', 'D'
            index=date_range("2000-01-01", periods=10, freq="B"),  # 指定行索引为工作日频率的日期范围
        )
        axes = _check_plot_works(
            df.plot,
            default_axes=True,
            subplots=True,
            layout=(-1, 2),  # 使用负值布局，指定为2列
        )
        _check_axes_shape(axes, axes_num=4, layout=(2, 2))  # 检查子图的形状，应为2行2列

    @pytest.mark.slow
    def test_plot_subplots_use_index(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),  # 创建一个10行4列的随机标准正态分布DataFrame
            columns=Index(list("ABCD"), dtype=object),  # 指定列索引为'A', 'B', 'C', 'D'
            index=date_range("2000-01-01", periods=10, freq="B"),  # 指定行索引为工作日频率的日期范围
        )
        axes = _check_plot_works(
            df.plot,
            default_axes=True,
            subplots=True,
            use_index=False,  # 不使用索引作为X轴
        )
        _check_ticks_props(axes, xrot=0)  # 检查刻度属性，X轴旋转角度为0度
        _check_axes_shape(axes, axes_num=4, layout=(4, 1))  # 检查子图的形状，应为4行1列

    @pytest.mark.xfail(reason="Api changed in 3.6.0")
    @pytest.mark.slow
    # 定义测试函数，用于测试在传递无效参数时是否引发了 AttributeError 异常
    def test_plot_invalid_arg(self):
        # 创建一个包含两列的 DataFrame 对象
        df = DataFrame({"x": [1, 2], "y": [3, 4]})
        # 定义错误消息，用于匹配异常信息中的字符串
        msg = "'Line2D' object has no property 'blarg'"
        # 使用 pytest 的断言捕获异常，并检查是否引发了 AttributeError，并匹配预期的错误消息
        with pytest.raises(AttributeError, match=msg):
            # 在 DataFrame 上调用 plot.line 方法，并传递一个未知的参数 blarg=True
            df.plot.line(blarg=True)

    # 标记为慢速测试的函数，用于测试绘图的刻度属性
    @pytest.mark.slow
    def test_plot_tick_props(self):
        # 创建一个包含随机数据的 DataFrame 对象，使用默认的随机数生成器
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            # 为 DataFrame 指定索引，使用前十个英文字母
            index=list(string.ascii_letters[:10]),
        )

        # 调用 _check_plot_works 函数，验证绘图函数正常工作并返回轴对象
        ax = _check_plot_works(df.plot, use_index=True)
        # 检查轴对象的刻度属性，设置 x 轴刻度旋转为 0 度
        _check_ticks_props(ax, xrot=0)

    # 标记为慢速测试的函数，使用参数化测试来测试不同的绘图参数
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"yticks": [1, 5, 10]},
            {"xticks": [1, 5, 10]},
            {"ylim": (-100, 100), "xlim": (-100, 100)},
            {"default_axes": True, "subplots": True, "title": "blah"},
        ],
    )
    def test_plot_other_args(self, kwargs):
        # 创建一个包含随机数据的 DataFrame 对象，使用默认的随机数生成器
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            # 为 DataFrame 指定索引，使用前十个英文字母
            index=list(string.ascii_letters[:10]),
        )
        # 调用 _check_plot_works 函数，验证绘图函数正常工作，并传递参数化的 kwargs
        _check_plot_works(df.plot, **kwargs)

    # 标记为慢速测试的函数，用于测试可见性轴的绘图
    @pytest.mark.slow
    def test_plot_visible_ax(self):
        # 创建一个包含随机数据的 DataFrame 对象，使用默认的随机数生成器
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            # 为 DataFrame 指定索引，使用前十个英文字母
            index=list(string.ascii_letters[:10]),
        )
        # 重新创建轴对象，因为 _check_plot_works 函数会执行两次绘图，
        # 一次没有 ax 参数，一次有 ax 参数，新的 sharex 行为不会移除后者轴的可见性
        axes = df.plot(subplots=True, title="blah")
        # 检查轴对象的形状，期望有三个轴对象，布局为 (3, 1)
        _check_axes_shape(axes, axes_num=3, layout=(3, 1))
        # 遍历前两个轴对象
        for ax in axes[:2]:
            _check_visible(ax.xaxis)  # 检查 x 轴是否可见以绘制网格线
            _check_visible(ax.get_xticklabels(), visible=False)  # 检查 x 轴刻度标签是否隐藏
            _check_visible(ax.get_xticklabels(minor=True), visible=False)  # 检查次要 x 轴刻度标签是否隐藏
            _check_visible([ax.xaxis.get_label()], visible=False)  # 检查 x 轴标签是否隐藏
        # 对第三个轴对象执行以下操作
        for ax in [axes[2]]:
            _check_visible(ax.xaxis)  # 检查 x 轴是否可见
            _check_visible(ax.get_xticklabels())  # 检查 x 轴刻度标签是否可见
            _check_visible([ax.xaxis.get_label()])  # 检查 x 轴标签是否可见
            _check_ticks_props(ax, xrot=0)  # 检查轴对象的刻度属性，设置 x 轴刻度旋转为 0 度

    # 标记为慢速测试的函数，用于测试绘图的标题设置
    @pytest.mark.slow
    def test_plot_title(self):
        # 创建一个包含随机数据的 DataFrame 对象，使用默认的随机数生成器
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            # 为 DataFrame 指定索引，使用前十个英文字母
            index=list(string.ascii_letters[:10]),
        )
        # 调用 _check_plot_works 函数，验证绘图函数正常工作，并设置标题为 "blah"
        _check_plot_works(df.plot, title="blah")

    # 标记为慢速测试的函数，用于测试多级索引的绘图
    @pytest.mark.slow
    def test_plot_multiindex(self):
        # 创建一个包含元组索引的 DataFrame 对象，元组包含前十个英文字母和它们的数值索引
        tuples = zip(string.ascii_letters[:10], range(10))
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            # 使用 MultiIndex.from_tuples 创建多级索引
            index=MultiIndex.from_tuples(tuples),
        )
        # 调用 _check_plot_works 函数，验证绘图函数正常工作，并使用索引
        ax = _check_plot_works(df.plot, use_index=True)
        # 检查轴对象的刻度属性，设置 x 轴刻度旋转为 0 度
        _check_ticks_props(ax, xrot=0)
    @pytest.mark.slow
    @pytest.mark.parametrize("layout", [None, (-1, 1)])
    def test_plot_single_column_bar(self, layout):
        # GH 6951
        # Test with single column
        # 创建一个只包含一个列的 DataFrame，列名为 'x'，数据为随机生成的 10 个浮点数
        df = DataFrame({"x": np.random.default_rng(2).random(10)})
        # 调用 _check_plot_works 函数，绘制 df 的条形图，预期子图布局为 layout
        axes = _check_plot_works(df.plot.bar, subplots=True, layout=layout)
        # 检查返回的 axes 数组的形状，预期为 (1, 1)
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    @pytest.mark.slow
    def test_plot_passed_ax(self):
        # 当提供了 ax 参数，并且所需的 axes 数量为 1 时，
        # 应该使用提供的 ax 对象：
        # 创建一个只包含一个列的 DataFrame，列名为 'x'，数据为随机生成的 10 个浮点数
        df = DataFrame({"x": np.random.default_rng(2).random(10)})
        # 创建一个新的图形和轴对象，并返回轴数组
        _, ax = mpl.pyplot.subplots()
        # 使用提供的 ax 对象绘制 df 的条形图，并返回绘制的 axes 数组
        axes = df.plot.bar(subplots=True, ax=ax)
        # 断言返回的 axes 数组长度为 1
        assert len(axes) == 1
        # 检查返回的第一个轴对象是否与提供的 ax 对象相同
        result = ax.axes
        assert result is axes[0]

    @pytest.mark.parametrize(
        "cols, x, y",
        [
            [list("ABCDE"), "A", "B"],
            [["A", "B"], "A", "B"],
            [["C", "A"], "C", "A"],
            [["A", "C"], "A", "C"],
            [["B", "C"], "B", "C"],
            [["A", "D"], "A", "D"],
            [["A", "E"], "A", "E"],
        ],
    )
    def test_nullable_int_plot(self, cols, x, y):
        # GH 32073
        # 创建一个包含多种数据类型的 DataFrame，包括可空整数、日期时间等
        dates = ["2008", "2009", None, "2011", "2012"]
        df = DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [1, 2, 3, 4, 5],
                "C": np.array([7, 5, np.nan, 3, 2], dtype=object),
                "D": pd.to_datetime(dates, format="%Y").view("i8"),
                "E": pd.to_datetime(dates, format="%Y", utc=True).view("i8"),
            }
        )

        # 调用 _check_plot_works 函数，绘制 df[cols] 的图形，指定 x 和 y 轴
        _check_plot_works(df[cols].plot, x=x, y=y)

    @pytest.mark.slow
    @pytest.mark.parametrize("plot", ["line", "bar", "hist", "pie"])
    def test_integer_array_plot_series(self, plot):
        # GH 25587
        # 创建一个包含 UInt32 类型数据的 Series 对象
        arr = pd.array([1, 2, 3, 4], dtype="UInt32")

        s = Series(arr)
        # 调用 getattr 函数，根据 plot 参数选择 s.plot 的相应绘图方法，并进行绘制
        _check_plot_works(getattr(s.plot, plot))

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "plot, kwargs",
        [
            ["line", {}],
            ["bar", {}],
            ["hist", {}],
            ["pie", {"y": "y"}],
            ["scatter", {"x": "x", "y": "y"}],
            ["hexbin", {"x": "x", "y": "y"}],
        ],
    )
    # 测试函数，用于测试绘制整数数组的 DataFrame
    def test_integer_array_plot_df(self, plot, kwargs):
        # GH 25587
        # 创建一个包含整数数组的 Pandas 数组，指定数据类型为 UInt32
        arr = pd.array([1, 2, 3, 4], dtype="UInt32")
        # 创建一个 DataFrame，包含两列 'x' 和 'y'，使用上述整数数组作为数据
        df = DataFrame({"x": arr, "y": arr})
        # 调用 _check_plot_works 函数，检查绘图是否正常工作
        _check_plot_works(getattr(df.plot, plot), **kwargs)

    # 测试函数，用于测试排除非数值列时的行为
    def test_nonnumeric_exclude(self):
        # 创建一个包含非数值数据的 DataFrame
        df = DataFrame({"A": ["x", "y", "z"], "B": [1, 2, 3]})
        # 绘制 DataFrame 的默认图表，并获取绘图的 Axes 对象
        ax = df.plot()
        # 断言绘图中线条的数量为1，确保列 'B' 被绘制
        assert len(ax.get_lines()) == 1  # B was plotted

    # 测试函数，用于测试隐式标签的行为
    def test_implicit_label(self):
        # 创建一个包含随机数的 DataFrame，列名为 'a', 'b', 'c'
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=["a", "b", "c"]
        )
        # 绘制 DataFrame 的图表，指定 'a' 作为 x 轴，'b' 作为 y 轴
        ax = df.plot(x="a", y="b")
        # 调用 _check_text_labels 函数，检查 x 轴标签是否为 'a'
        _check_text_labels(ax.xaxis.get_label(), "a")

    # 测试函数，用于测试不覆盖索引名称的行为
    def test_donot_overwrite_index_name(self):
        # GH 8494
        # 创建一个包含随机数的 DataFrame，2行2列，列名为 'a', 'b'，并指定索引名称为 'NAME'
        df = DataFrame(
            np.random.default_rng(2).standard_normal((2, 2)), columns=["a", "b"]
        )
        # 设置 DataFrame 的索引名称为 'NAME'
        df.index.name = "NAME"
        # 绘制 DataFrame 的 'b' 列，并添加标签为 'LABEL'
        df.plot(y="b", label="LABEL")
        # 断言 DataFrame 的索引名称仍然为 'NAME'
        assert df.index.name == "NAME"

    # 测试函数，用于测试不同 x 和 y 列组合的绘图行为
    def test_plot_xy(self):
        # columns.inferred_type == 'string'
        # 创建一个包含随机数的 DataFrame，4列，列名为 'A', 'B', 'C', 'D'
        # 指定索引为工作日频率，5个日期
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=5, freq="B"),
        )
        # 检查多种 x 和 y 列组合的绘图结果是否一致
        _check_data(df.plot(x=0, y=1), df.set_index("A")["B"].plot())
        _check_data(df.plot(x=0), df.set_index("A").plot())
        _check_data(df.plot(y=0), df.B.plot())
        _check_data(df.plot(x="A", y="B"), df.set_index("A").B.plot())
        _check_data(df.plot(x="A"), df.set_index("A").plot())
        _check_data(df.plot(y="B"), df.B.plot())

    # 测试函数，用于测试整数列的绘图行为
    def test_plot_xy_int_cols(self):
        # 创建一个包含随机数的 DataFrame，4列，列名为 'A', 'B', 'C', 'D'
        # 指定索引为工作日频率，5个日期
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=5, freq="B"),
        )
        # columns.inferred_type == 'integer'
        # 将 DataFrame 的列名改为整数，并检查不同 x 和 y 列组合的绘图结果是否一致
        df.columns = np.arange(1, len(df.columns) + 1)
        _check_data(df.plot(x=1, y=2), df.set_index(1)[2].plot())
        _check_data(df.plot(x=1), df.set_index(1).plot())
        _check_data(df.plot(y=1), df[1].plot())

    # 测试函数，用于测试设置 figsize 和 title 的行为
    def test_plot_xy_figsize_and_title(self):
        # 创建一个包含随机数的 DataFrame，4列，列名为 'A', 'B', 'C', 'D'
        # 指定索引为工作日频率，5个日期
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=5, freq="B"),
        )
        # figsize and title
        # 绘制 DataFrame 的图表，指定 x 列为 1，y 列为 2，设置标题为 'Test'，图表尺寸为 (16, 8)
        ax = df.plot(x=1, y=2, title="Test", figsize=(16, 8))
        # 调用 _check_text_labels 函数，检查图表标题是否为 'Test'
        _check_text_labels(ax.title, "Test")
        # 调用 _check_axes_shape 函数，检查 Axes 对象的形状和尺寸是否符合预期
        _check_axes_shape(ax, axes_num=1, layout=(1, 1), figsize=(16.0, 8.0))

        # columns.inferred_type == 'mixed'
        # TODO add MultiIndex test

    @pytest.mark.parametrize(
        "input_log, expected_log", [(True, "log"), ("sym", "symlog")]
    )
    def test_logscales(self, input_log, expected_log):
        # 创建一个包含整数列 'a' 的 DataFrame，索引为 0 到 99
        df = DataFrame({"a": np.arange(100)}, index=np.arange(100))

        # 使用指定的 y 轴对数尺度绘制 DataFrame 的折线图
        ax = df.plot(logy=input_log)
        # 检查绘图对象的轴比例是否符合预期的 y 轴对数尺度
        _check_ax_scales(ax, yaxis=expected_log)
        # 断言实际的 y 轴尺度与预期值相同
        assert ax.get_yscale() == expected_log

        # 使用指定的 x 轴对数尺度绘制 DataFrame 的折线图
        ax = df.plot(logx=input_log)
        # 检查绘图对象的轴比例是否符合预期的 x 轴对数尺度
        _check_ax_scales(ax, xaxis=expected_log)
        # 断言实际的 x 轴尺度与预期值相同
        assert ax.get_xscale() == expected_log

        # 使用指定的 x 和 y 轴对数尺度绘制 DataFrame 的折线图
        ax = df.plot(loglog=input_log)
        # 检查绘图对象的轴比例是否同时符合预期的 x 和 y 轴对数尺度
        _check_ax_scales(ax, xaxis=expected_log, yaxis=expected_log)
        # 断言实际的 x 和 y 轴尺度均与预期值相同
        assert ax.get_xscale() == expected_log
        assert ax.get_yscale() == expected_log

    @pytest.mark.parametrize("input_param", ["logx", "logy", "loglog"])
    def test_invalid_logscale(self, input_param):
        # GH: 24867
        # 创建一个包含整数列 'a' 的 DataFrame，索引为 0 到 99
        df = DataFrame({"a": np.arange(100)}, index=np.arange(100))

        # 准备错误消息，指出 'input_param' 关键字应为 bool、None 或 'sym'，而不是 'sm'
        msg = f"keyword '{input_param}' should be bool, None, or 'sym', not 'sm'"
        # 使用 pytest 的断言检查是否引发了 ValueError，并检查错误消息是否匹配预期
        with pytest.raises(ValueError, match=msg):
            df.plot(**{input_param: "sm"})

        # 准备警告消息，指出 'PiePlot' 忽略 'input_param' 关键字
        msg = f"PiePlot ignores the '{input_param}' keyword"
        # 使用 tm.assert_produces_warning 检查是否引发了 UserWarning，并检查警告消息是否匹配预期
        with tm.assert_produces_warning(UserWarning, match=msg):
            df.plot.pie(subplots=True, **{input_param: True})

    def test_xcompat(self):
        # 创建一个包含标准正态分布数据的 DataFrame，列名为 'ABCD'，索引为工作日日期范围
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # 使用 x_compat=True 参数绘制 DataFrame 的折线图
        ax = df.plot(x_compat=True)
        # 获取绘图对象的所有线条对象
        lines = ax.get_lines()
        # 断言第一条线条的 x 数据不是 PeriodIndex 类型
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        # 检查绘图对象的刻度属性，设置 x 轴标签的旋转角度为 30 度
        _check_ticks_props(ax, xrot=30)

    def test_xcompat_plot_params(self):
        # 创建一个包含标准正态分布数据的 DataFrame，列名为 'ABCD'，索引为工作日日期范围
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # 将绘图参数中的 'xaxis.compat' 设置为 True
        plotting.plot_params["xaxis.compat"] = True
        # 绘制 DataFrame 的折线图
        ax = df.plot()
        # 获取绘图对象的所有线条对象
        lines = ax.get_lines()
        # 断言第一条线条的 x 数据不是 PeriodIndex 类型
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        # 检查绘图对象的刻度属性，设置 x 轴标签的旋转角度为 30 度
        _check_ticks_props(ax, xrot=30)

    def test_xcompat_plot_params_x_compat(self):
        # 创建一个包含标准正态分布数据的 DataFrame，列名为 'ABCD'，索引为工作日日期范围
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # 将绘图参数中的 'x_compat' 设置为 False
        plotting.plot_params["x_compat"] = False

        # 绘制 DataFrame 的折线图
        ax = df.plot()
        # 获取绘图对象的所有线条对象
        lines = ax.get_lines()
        # 断言第一条线条的 x 数据不是 PeriodIndex 类型
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        # 准备未来警告消息，指出 'PeriodDtype[B] is deprecated'
        msg = r"PeriodDtype\[B\] is deprecated"
        # 使用 tm.assert_produces_warning 检查是否引发了 FutureWarning，并检查警告消息是否匹配预期
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 断言第一条线条的 x 数据是 PeriodIndex 类型
            assert isinstance(PeriodIndex(lines[0].get_xdata()), PeriodIndex)
    def test_xcompat_plot_params_context_manager(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # 如果要一次性绘制多个图表，这个语句非常有用
        with plotting.plot_params.use("x_compat", True):
            # 使用上下文管理器设置参数后，绘制DataFrame的图表
            ax = df.plot()
            # 获取图表的所有线条对象
            lines = ax.get_lines()
            # 断言第一个线条的X轴数据不是PeriodIndex类型
            assert not isinstance(lines[0].get_xdata(), PeriodIndex)
            # 检查并调整X轴刻度属性
            _check_ticks_props(ax, xrot=30)

    def test_xcompat_plot_period(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # 绘制DataFrame的图表
        ax = df.plot()
        # 获取图表的所有线条对象
        lines = ax.get_lines()
        # 断言第一个线条的X轴数据不是PeriodIndex类型
        assert not isinstance(lines[0].get_xdata(), PeriodIndex)
        # 设置警告消息内容
        msg = r"PeriodDtype\[B\] is deprecated "
        # 使用上下文管理器检查是否产生了未来警告，并匹配特定消息
        with tm.assert_produces_warning(FutureWarning, match=msg):
            # 断言第一个线条的X轴数据是PeriodIndex类型
            assert isinstance(PeriodIndex(lines[0].get_xdata()), PeriodIndex)
        # 检查并调整X轴刻度属性
        _check_ticks_props(ax, xrot=0)

    def test_period_compat(self):
        # GH 9012
        # 处理周期数组的转换
        df = DataFrame(
            np.random.default_rng(2).random((21, 2)),
            index=bdate_range(datetime(2000, 1, 1), datetime(2000, 1, 31)),
            columns=["a", "b"],
        )

        # 绘制DataFrame的图表
        df.plot()
        # 添加一条水平线到图表中
        mpl.pyplot.axhline(y=0)

    @pytest.mark.parametrize("index_dtype", [np.int64, np.float64])
    def test_unsorted_index(self, index_dtype):
        # 创建DataFrame，索引为逆序数组，数据类型由参数确定
        df = DataFrame(
            {"y": np.arange(100)},
            index=Index(np.arange(99, -1, -1), dtype=index_dtype),
            dtype=np.int64,
        )
        # 绘制DataFrame的图表
        ax = df.plot()
        # 获取图表中第一个线条对象
        lines = ax.get_lines()[0]
        # 获取线条的数据点
        rs = lines.get_xydata()
        # 将数据点转换为Series对象
        rs = Series(rs[:, 1], rs[:, 0], dtype=np.int64, name="y")
        # 比较Series对象是否与DataFrame的y列相等，忽略索引类型的检查
        tm.assert_series_equal(rs, df.y, check_index_type=False)

    @pytest.mark.parametrize(
        "df",
        [
            # 创建具有指定索引的DataFrame
            DataFrame({"y": [0.0, 1.0, 2.0, 3.0]}, index=[1.0, 0.0, 3.0, 2.0]),
            DataFrame(
                {"y": [0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0]},
                index=[1.0, 0.0, 3.0, 2.0, np.nan, 3.0, 2.0],
            ),
        ],
    )
    def test_unsorted_index_lims(self, df):
        # 绘制DataFrame的图表
        ax = df.plot()
        # 获取X轴的限制范围
        xmin, xmax = ax.get_xlim()
        # 获取图表中的所有线条对象
        lines = ax.get_lines()
        # 断言X轴的最小值小于等于所有线条的最小X轴数据
        assert xmin <= np.nanmin(lines[0].get_data()[0])
        # 断言X轴的最大值大于等于所有线条的最大X轴数据
        assert xmax >= np.nanmax(lines[0].get_data()[0])

    def test_unsorted_index_lims_x_y(self):
        # 创建DataFrame，具有两列' y '和' z '
        df = DataFrame({"y": [0.0, 1.0, 2.0, 3.0], "z": [91.0, 90.0, 93.0, 92.0]})
        # 根据指定的列名x和y绘制DataFrame的图表
        ax = df.plot(x="z", y="y")
        # 获取X轴的限制范围
        xmin, xmax = ax.get_xlim()
        # 获取图表中的所有线条对象
        lines = ax.get_lines()
        # 断言X轴的最小值小于等于所有线条的最小X轴数据
        assert xmin <= np.nanmin(lines[0].get_data()[0])
        # 断言X轴的最大值大于等于所有线条的最大X轴数据
        assert xmax >= np.nanmax(lines[0].get_data()[0])
    # 定义测试函数，用于测试在对数坐标轴下绘制面积图时是否引发值错误异常
    def test_negative_log(self):
        # 创建一个包含随机数据的 DataFrame，并对数据进行取反操作
        df = -DataFrame(
            np.random.default_rng(2).random((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["x", "y", "z", "four"],
        )
        # 定义错误消息字符串
        msg = "Log-y scales are not supported in area plot"
        # 断言绘制对数坐标轴下的面积图时是否抛出指定的值错误异常
        with pytest.raises(ValueError, match=msg):
            df.plot.area(logy=True)
        with pytest.raises(ValueError, match=msg):
            df.plot.area(loglog=True)

    # 定义内部方法，用于比较两组线的堆叠 Y 坐标
    def _compare_stacked_y_cood(self, normal_lines, stacked_lines):
        # 创建一个长度与 normal_lines[0] 的 Y 坐标长度相同的零数组作为基准
        base = np.zeros(len(normal_lines[0].get_data()[1]))
        # 对正常线和堆叠线进行迭代，累加得到基准 Y 坐标
        for nl, sl in zip(normal_lines, stacked_lines):
            base += nl.get_data()[1]  # 获取 Y 坐标
            sy = sl.get_data()[1]
            # 使用断言验证基准 Y 坐标与堆叠 Y 坐标是否相等
            tm.assert_numpy_array_equal(base, sy)

    # 定义测试函数，用于测试线图和区域图在堆叠和非堆叠状态下的绘制
    @pytest.mark.parametrize("kind", ["line", "area"])
    @pytest.mark.parametrize("mult", [1, -1])
    def test_line_area_stacked(self, kind, mult):
        # 创建一个包含随机数据的 DataFrame，并乘以 mult 参数
        df = mult * DataFrame(
            np.random.default_rng(2).random((6, 4)), columns=["w", "x", "y", "z"]
        )

        # 调用 _check_plot_works 函数验证绘图正常工作，分别测试堆叠和非堆叠状态
        ax1 = _check_plot_works(df.plot, kind=kind, stacked=False)
        ax2 = _check_plot_works(df.plot, kind=kind, stacked=True)
        # 调用 _compare_stacked_y_cood 方法比较两个绘图对象的线的堆叠 Y 坐标
        self._compare_stacked_y_cood(ax1.lines, ax2.lines)

    # 定义测试函数，用于测试线图和区域图在堆叠和非堆叠状态下的绘制，针对分开的 DataFrame
    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_line_area_stacked_sep_df(self, kind):
        # 创建一个包含随机数据的 DataFrame，每列包含正负随机值
        sep_df = DataFrame(
            {
                "w": np.random.default_rng(2).random(6),
                "x": np.random.default_rng(2).random(6),
                "y": -np.random.default_rng(2).random(6),
                "z": -np.random.default_rng(2).random(6),
            }
        )
        # 调用 _check_plot_works 函数验证绘图正常工作，分别测试堆叠和非堆叠状态
        ax1 = _check_plot_works(sep_df.plot, kind=kind, stacked=False)
        ax2 = _check_plot_works(sep_df.plot, kind=kind, stacked=True)
        # 比较每个 DataFrame 的前两条线和后两条线的堆叠 Y 坐标
        self._compare_stacked_y_cood(ax1.lines[:2], ax2.lines[:2])
        self._compare_stacked_y_cood(ax1.lines[2:], ax2.lines[2:])

    # 定义测试函数，用于测试混合数据的线图和区域图在堆叠状态下的绘制
    def test_line_area_stacked_mixed(self):
        # 创建一个包含标准正态分布数据的 DataFrame
        mixed_df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["w", "x", "y", "z"],
        )
        # 调用 _check_plot_works 函数验证绘图正常工作，堆叠状态设置为 False
        _check_plot_works(mixed_df.plot, stacked=False)

        # 定义错误消息字符串，用于测试在堆叠状态下混合数据是否引发值错误异常
        msg = (
            "When stacked is True, each column must be either all positive or "
            "all negative. Column 'w' contains both positive and negative "
            "values"
        )
        # 断言绘制堆叠状态下混合数据的线图或区域图是否抛出指定的值错误异常
        with pytest.raises(ValueError, match=msg):
            mixed_df.plot(stacked=True)

    # 参数化装饰器，定义测试函数的参数化选项，用于测试线图和区域图的绘制
    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_line_area_stacked_positive_idx(self, kind):
        # 创建一个 6x4 的随机数据框
        df = DataFrame(
            np.random.default_rng(2).random((6, 4)), columns=["w", "x", "y", "z"]
        )
        # 使用具有严格正值的索引，以防止 matplotlib 警告忽略 xlim
        df2 = df.set_index(df.index + 1)
        # 调用 _check_plot_works 函数，测试绘图功能，使用 logx 和 stacked 参数
        _check_plot_works(df2.plot, kind=kind, logx=True, stacked=True)

    @pytest.mark.parametrize(
        "idx", [range(4), date_range("2023-01-1", freq="D", periods=4)]
    )
    def test_line_area_nan_df(self, idx):
        values1 = [1, 2, np.nan, 3]
        values2 = [3, np.nan, 2, 1]
        # 创建包含 NaN 值的数据框 df，使用给定的索引 idx
        df = DataFrame({"a": values1, "b": values2}, index=idx)

        # 调用 _check_plot_works 函数，获取绘图的 axes 对象
        ax = _check_plot_works(df.plot)
        # 获取第一个线条的数据，去除 NaN 值进行比较
        masked1 = ax.lines[0].get_ydata()
        masked2 = ax.lines[1].get_ydata()
        # 删除特定索引处的数据并与期望值进行比较
        exp = np.array([1, 2, 3], dtype=np.float64)
        tm.assert_numpy_array_equal(np.delete(masked1.data, 2), exp)

        exp = np.array([3, 2, 1], dtype=np.float64)
        tm.assert_numpy_array_equal(np.delete(masked2.data, 1), exp)
        # 检查第一个线条的 mask 数组
        tm.assert_numpy_array_equal(masked1.mask, np.array([False, False, True, False]))
        # 检查第二个线条的 mask 数组
        tm.assert_numpy_array_equal(masked2.mask, np.array([False, True, False, False]))

    @pytest.mark.parametrize(
        "idx", [range(4), date_range("2023-01-1", freq="D", periods=4)]
    )
    def test_line_area_nan_df_stacked(self, idx):
        values1 = [1, 2, np.nan, 3]
        values2 = [3, np.nan, 2, 1]
        # 创建包含 NaN 值的数据框 df，使用给定的索引 idx
        df = DataFrame({"a": values1, "b": values2}, index=idx)

        expected1 = np.array([1, 2, 0, 3], dtype=np.float64)
        expected2 = np.array([3, 0, 2, 1], dtype=np.float64)

        # 调用 _check_plot_works 函数，获取绘图的 axes 对象，设置 stacked=True
        ax = _check_plot_works(df.plot, stacked=True)
        # 检查第一个线条的数据是否符合预期
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected1)
        # 检查第二个线条的数据是否符合预期
        tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected1 + expected2)

    @pytest.mark.parametrize(
        "idx", [range(4), date_range("2023-01-1", freq="D", periods=4)]
    )
    @pytest.mark.parametrize("kwargs", [{}, {"stacked": False}])
    def test_line_area_nan_df_stacked_area(self, idx, kwargs):
        values1 = [1, 2, np.nan, 3]
        values2 = [3, np.nan, 2, 1]
        # 创建包含 NaN 值的数据框 df，使用给定的索引 idx
        df = DataFrame({"a": values1, "b": values2}, index=idx)

        expected1 = np.array([1, 2, 0, 3], dtype=np.float64)
        expected2 = np.array([3, 0, 2, 1], dtype=np.float64)

        # 调用 _check_plot_works 函数，获取绘图的 axes 对象，使用给定的 kwargs
        ax = _check_plot_works(df.plot.area, **kwargs)
        # 检查第一个线条的数据是否符合预期
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected1)
        # 根据 kwargs 检查第二个线条的数据是否符合预期
        if kwargs:
            tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected2)
        else:
            tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected1 + expected2)

        # 调用 _check_plot_works 函数，获取绘图的 axes 对象，设置 stacked=False
        ax = _check_plot_works(df.plot.area, stacked=False)
        # 检查第一个线条的数据是否符合预期
        tm.assert_numpy_array_equal(ax.lines[0].get_ydata(), expected1)
        # 检查第二个线条的数据是否符合预期
        tm.assert_numpy_array_equal(ax.lines[1].get_ydata(), expected2)
    @pytest.mark.parametrize("kwargs", [{}, {"secondary_y": True}])
    def test_line_lim(self, kwargs):
        df = DataFrame(np.random.default_rng(2).random((6, 3)), columns=["x", "y", "z"])
        # 使用给定的参数kwargs绘制DataFrame df 的线性图，并返回绘图对象ax
        ax = df.plot(**kwargs)
        # 获取当前绘图的x轴限制范围
        xmin, xmax = ax.get_xlim()
        # 获取当前绘图的所有线条对象
        lines = ax.get_lines()
        # 断言：确保x轴的最小值小于或等于第一条线条的第一个数据点的x坐标
        assert xmin <= lines[0].get_data()[0][0]
        # 断言：确保x轴的最大值大于或等于第一条线条的最后一个数据点的x坐标

    def test_line_lim_subplots(self):
        df = DataFrame(np.random.default_rng(2).random((6, 3)), columns=["x", "y", "z"])
        # 使用参数secondary_y=True和subplots=True绘制DataFrame df 的子图，并返回所有绘图对象
        axes = df.plot(secondary_y=True, subplots=True)
        # 断言：确保返回的绘图对象数组符合指定的形状和布局
        _check_axes_shape(axes, axes_num=3, layout=(3, 1))
        for ax in axes:
            # 断言：确保每个绘图对象具有属性"left_ax"
            assert hasattr(ax, "left_ax")
            # 断言：确保每个绘图对象不具有属性"right_ax"
            assert not hasattr(ax, "right_ax")
            # 获取当前绘图对象的x轴限制范围
            xmin, xmax = ax.get_xlim()
            # 获取当前绘图对象的所有线条对象
            lines = ax.get_lines()
            # 断言：确保x轴的最小值小于或等于第一条线条的第一个数据点的x坐标
            assert xmin <= lines[0].get_data()[0][0]
            # 断言：确保x轴的最大值大于或等于第一条线条的最后一个数据点的x坐标

    @pytest.mark.xfail(
        strict=False,
        reason="2020-12-01 this has been failing periodically on the "
        "ymin==0 assertion for a week or so.",
    )
    @pytest.mark.parametrize("stacked", [True, False])
    def test_area_lim(self, stacked):
        df = DataFrame(
            np.random.default_rng(2).random((6, 4)), columns=["x", "y", "z", "four"]
        )

        neg_df = -df

        # 调用_check_plot_works函数，检查绘制区域图是否正常工作，并返回绘图对象ax
        ax = _check_plot_works(df.plot.area, stacked=stacked)
        # 获取当前绘图对象的x轴限制范围
        xmin, xmax = ax.get_xlim()
        # 获取当前绘图对象的y轴限制范围
        ymin, ymax = ax.get_ylim()
        # 获取当前绘图对象的所有线条对象
        lines = ax.get_lines()
        # 断言：确保x轴的最小值小于或等于第一条线条的第一个数据点的x坐标
        assert xmin <= lines[0].get_data()[0][0]
        # 断言：确保x轴的最大值大于或等于第一条线条的最后一个数据点的x坐标
        assert xmax >= lines[0].get_data()[0][-1]
        # 断言：确保y轴的最小值为0
        assert ymin == 0

        # 调用_check_plot_works函数，检查绘制负值区域图是否正常工作，并返回绘图对象ax
        ax = _check_plot_works(neg_df.plot.area, stacked=stacked)
        # 获取当前绘图对象的y轴限制范围
        ymin, ymax = ax.get_ylim()
        # 断言：确保y轴的最大值为0
        assert ymax == 0

    def test_area_sharey_dont_overwrite(self):
        # GH37942
        df = DataFrame(np.random.default_rng(2).random((4, 2)), columns=["x", "y"])
        # 创建一个包含两个子图的图形fig和对应的轴对象ax1和ax2，这两个轴对象共享y轴
        fig, (ax1, ax2) = mpl.pyplot.subplots(1, 2, sharey=True)

        # 在ax1上绘制DataFrame df 的区域图
        df.plot(ax=ax1, kind="area")
        # 在ax2上绘制DataFrame df 的区域图
        df.plot(ax=ax2, kind="area")

        # 断言：确保ax1和ax2共享y轴
        assert get_y_axis(ax1).joined(ax1, ax2)
        assert get_y_axis(ax2).joined(ax1, ax2)

    @pytest.mark.parametrize("stacked", [True, False])
    def test_bar_linewidth(self, stacked):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        # 使用参数stacked绘制DataFrame df 的柱状图，并返回绘图对象ax
        ax = df.plot.bar(stacked=stacked, linewidth=2)
        # 遍历所有柱子的矩形对象r，断言每个柱子的线宽为2
        for r in ax.patches:
            assert r.get_linewidth() == 2

    def test_bar_linewidth_subplots(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
        # 创建子图
        axes = df.plot.bar(linewidth=2, subplots=True)
        # 断言：确保返回的绘图对象数组符合指定的形状和布局
        _check_axes_shape(axes, axes_num=5, layout=(5, 1))
        for ax in axes:
            # 遍历每个子图的所有柱子的矩形对象r，断言每个柱子的线宽为2
            for r in ax.patches:
                assert r.get_linewidth() == 2

    @pytest.mark.parametrize(
        "meth, dim", [("bar", "get_width"), ("barh", "get_height")]
    )
    @pytest.mark.parametrize("stacked", [True, False])
    # 定义一个测试方法，用于测试柱状图的宽度是否正确
    def test_bar_barwidth(self, meth, dim, stacked):
        # 创建一个 5x5 的随机数据帧
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        # 设置柱状图的宽度为 0.9
        width = 0.9

        # 调用 DataFrame 的 plot 方法，生成对应类型的图形并返回轴对象 ax
        ax = getattr(df.plot, meth)(stacked=stacked, width=width)
        # 遍历图中的每一个柱形对象 r
        for r in ax.patches:
            # 如果不是堆叠柱状图，断言柱形对象 r 的宽度是否等于总宽度除以列数目
            if not stacked:
                assert getattr(r, dim)() == width / len(df.columns)
            else:
                # 如果是堆叠柱状图，断言柱形对象 r 的宽度是否等于总宽度
                assert getattr(r, dim)() == width

    # 使用 pytest 的参数化标记，定义一个测试方法，测试水平柱状图的宽度是否正确
    @pytest.mark.parametrize(
        "meth, dim", [("bar", "get_width"), ("barh", "get_height")]
    )
    def test_barh_barwidth_subplots(self, meth, dim):
        # 创建一个 5x5 的随机数据帧
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))

        # 设置柱状图的宽度为 0.9
        width = 0.9

        # 调用 DataFrame 的 plot 方法，生成对应类型的图形并返回所有轴对象 axes
        axes = getattr(df.plot, meth)(width=width, subplots=True)
        # 遍历每一个轴对象 ax
        for ax in axes:
            # 遍历每一个轴对象中的柱形对象 r
            for r in ax.patches:
                # 断言柱形对象 r 的宽度是否等于总宽度
                assert getattr(r, dim)() == width

    # 定义一个测试方法，测试柱状图底部偏移是否正确
    def test_bar_bottom_left_bottom(self):
        # 创建一个 5x5 的随机数据帧
        df = DataFrame(np.random.default_rng(2).random((5, 5)))
        # 生成非堆叠柱状图，底部偏移设置为 1
        ax = df.plot.bar(stacked=False, bottom=1)
        # 获取所有柱形对象的底部偏移值并存储在 result 中
        result = [p.get_y() for p in ax.patches]
        # 断言所有柱形对象的底部偏移值是否为 [1, 1, 1, 1, 1] 重复 5 次
        assert result == [1] * 25

        # 生成堆叠柱状图，底部偏移设置为 [-1, -2, -3, -4, -5]
        ax = df.plot.bar(stacked=True, bottom=[-1, -2, -3, -4, -5])
        # 获取前 5 个柱形对象的底部偏移值并存储在 result 中
        result = [p.get_y() for p in ax.patches[:5]]
        # 断言前 5 个柱形对象的底部偏移值是否为 [-1, -2, -3, -4, -5]
        assert result == [-1, -2, -3, -4, -5]

    # 定义一个测试方法，测试水平柱状图左侧偏移是否正确
    def test_bar_bottom_left_left(self):
        # 创建一个 5x5 的随机数据帧
        df = DataFrame(np.random.default_rng(2).random((5, 5)))
        # 生成非堆叠水平柱状图，左侧偏移设置为 [1, 1, 1, 1, 1]
        ax = df.plot.barh(stacked=False, left=np.array([1, 1, 1, 1, 1]))
        # 获取所有柱形对象的左侧偏移值并存储在 result 中
        result = [p.get_x() for p in ax.patches]
        # 断言所有柱形对象的左侧偏移值是否为 [1, 1, 1, 1, 1] 重复 5 次
        assert result == [1] * 25

        # 生成堆叠水平柱状图，左侧偏移设置为 [1, 2, 3, 4, 5]
        ax = df.plot.barh(stacked=True, left=[1, 2, 3, 4, 5])
        # 获取前 5 个柱形对象的左侧偏移值并存储在 result 中
        result = [p.get_x() for p in ax.patches[:5]]
        # 断言前 5 个柱形对象的左侧偏移值是否为 [1, 2, 3, 4, 5]
        assert result == [1, 2, 3, 4, 5]

    # 定义一个测试方法，测试多子图中柱状图底部偏移是否正确
    def test_bar_bottom_left_subplots(self):
        # 创建一个 5x5 的随机数据帧
        df = DataFrame(np.random.default_rng(2).random((5, 5)))
        # 生成多子图中的柱状图，底部偏移设置为 -1
        axes = df.plot.bar(subplots=True, bottom=-1)
        # 遍历每一个子图的轴对象 ax
        for ax in axes:
            # 获取每一个轴对象中柱形对象的底部偏移值并存储在 result 中
            result = [p.get_y() for p in ax.patches]
            # 断言每一个轴对象中所有柱形对象的底部偏移值是否为 [-1] 重复 5 次
            assert result == [-1] * 5

        # 生成多子图中的水平柱状图，左侧偏移设置为 [1, 1, 1, 1, 1]
        axes = df.plot.barh(subplots=True, left=np.array([1, 1, 1, 1, 1]))
        # 遍历每一个子图的轴对象 ax
        for ax in axes:
            # 获取每一个轴对象中柱形对象的左侧偏移值并存储在 result 中
            result = [p.get_x() for p in ax.patches]
            # 断言每一个轴对象中所有柱形对象的左侧偏移值是否为 [1] 重复 5 次
            assert result == [1] * 5

    # 定义一个测试方法，测试柱状图中 NaN 值的处理是否正确
    def test_bar_nan(self):
        # 创建一个带有 NaN 值的数据帧
        df = DataFrame({"A": [10, np.nan, 20], "B": [5, 10, 20], "C": [1, 2, 3]})
        # 生成普通柱状图
        ax = df.plot.bar()
        # 期望的柱形高度列表
        expected = [10, 0, 20, 5, 10, 20, 1, 2, 3]
        # 获取所有柱形对象的高度值并存储在 result 中
        result = [p.get_height() for p in ax.patches]
        # 断言所有柱形对象的高度值是否与期望值列表相等
        assert result == expected

    # 定义一个测试方法，测试堆叠柱状图中 NaN 值的处理是否正确
    def test_bar_nan_stacked(self):
        # 创建一个带有 NaN 值的数据帧
        df = DataFrame({"A": [10, np.nan, 20], "B": [5, 10, 20], "C": [1, 2, 3]})
        # 生成堆叠柱状图
        ax = df.plot.bar(stacked=True
    # 定义测试函数，测试条形图可视化函数的分类数据情况
    def test_bar_categorical(self, idx):
        # GH 13019
        # 创建一个6行5列的随机数数据框，使用指定的索引和列标签
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 5)),
            index=idx(list("ABCDEF")),
            columns=idx(list("abcde")),
        )

        # 绘制条形图，并获取 x 轴刻度位置
        ax = df.plot.bar()
        ticks = ax.xaxis.get_ticklocs()
        # 断言 x 轴刻度是否为预期值
        tm.assert_numpy_array_equal(ticks, np.array([0, 1, 2, 3, 4, 5]))
        # 断言 x 轴的显示范围是否正确
        assert ax.get_xlim() == (-0.5, 5.5)
        # 检查第一个条形的左边缘位置
        assert ax.patches[0].get_x() == -0.25
        # 检查最后一个条形的左边缘位置
        assert ax.patches[-1].get_x() == 5.15

        # 绘制堆叠条形图
        ax = df.plot.bar(stacked=True)
        # 检查 x 轴刻度位置是否保持一致
        tm.assert_numpy_array_equal(ticks, np.array([0, 1, 2, 3, 4, 5]))
        # 断言 x 轴的显示范围是否正确
        assert ax.get_xlim() == (-0.5, 5.5)
        # 检查第一个条形的左边缘位置
        assert ax.patches[0].get_x() == -0.25
        # 检查最后一个条形的左边缘位置
        assert ax.patches[-1].get_x() == 4.75

    # 使用参数化测试进行散点图的测试
    @pytest.mark.parametrize("x, y", [("x", "y"), (1, 2)])
    def test_plot_scatter(self, x, y):
        # 创建一个6行4列的随机数数据框，使用字母表的前6个字母作为索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["x", "y", "z", "four"],
        )

        # 调用辅助函数，检查散点图函数是否正常工作
        _check_plot_works(df.plot.scatter, x=x, y=y)

    # 测试散点图函数对缺少参数的错误处理
    def test_plot_scatter_error(self):
        # 创建一个6行4列的随机数数据框，使用字母表的前6个字母作为索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["x", "y", "z", "four"],
        )
        # 准备异常消息的模板，匹配缺少参数的错误信息
        msg = re.escape("scatter() missing 1 required positional argument: 'y'")
        # 使用 pytest 检查是否抛出预期的 TypeError 异常，并匹配消息模板
        with pytest.raises(TypeError, match=msg):
            df.plot.scatter(x="x")
        msg = re.escape("scatter() missing 1 required positional argument: 'x'")
        with pytest.raises(TypeError, match=msg):
            df.plot.scatter(y="y")

    # 测试散点图在特定数据情况下的形状
    def test_plot_scatter_shape(self):
        # 创建一个6行4列的随机数数据框，使用字母表的前6个字母作为索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["x", "y", "z", "four"],
        )
        # GH 6951
        # 绘制散点图，并返回生成的坐标轴对象列表
        axes = df.plot(x="x", y="y", kind="scatter", subplots=True)
        # 调用辅助函数，检查坐标轴对象列表的形状是否符合预期
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    # 测试在散点图中使用 datetime.time 数据时是否抛出异常
    def test_raise_error_on_datetime_time_data(self):
        # GH 8113, datetime.time 类型在 matplotlib 中不受支持
        # 创建一个包含随机数列的数据框，其中包括一个 datetime.time 类型的列
        df = DataFrame(np.random.default_rng(2).standard_normal(10), columns=["a"])
        # 生成一列时间序列，并将其赋值给数据框的特定列
        df["dtime"] = date_range(start="2014-01-01", freq="h", periods=10).time
        # 准备异常消息的模板，匹配时间类型数据无法处理的错误信息
        msg = "must be a string or a (real )?number, not 'datetime.time'"
        # 使用 pytest 检查是否抛出预期的 TypeError 异常，并匹配消息模板
        with pytest.raises(TypeError, match=msg):
            df.plot(kind="scatter", x="dtime", y="a")

    # 使用参数化测试测试含有日期时间数据的散点图
    @pytest.mark.parametrize("x, y", [("dates", "vals"), (0, 1)])
    def test_scatterplot_datetime_data(self, x, y):
        # GH 30391
        # 生成一个日期范围，包含12个周的日期
        dates = date_range(start=date(2019, 1, 1), periods=12, freq="W")
        # 生成一个随机数列，并以此构建数据框
        vals = np.random.default_rng(2).normal(0, 1, len(dates))
        df = DataFrame({"dates": dates, "vals": vals})

        # 调用辅助函数，检查散点图函数是否正常工作
        _check_plot_works(df.plot.scatter, x=x, y=y)
    @pytest.mark.parametrize(
        "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
    )
    @pytest.mark.parametrize("x, y", [("a", "b"), (0, 1)])
    @pytest.mark.parametrize("b_col", [[2, 3, 4], ["a", "b", "c"]])
    # 定义测试方法，用于测试散点图中数据为对象类型时的情况
    def test_scatterplot_object_data(self, b_col, x, y, infer_string):
        # GH 18755
        # 设置上下文环境，配置将来的字符串推断选项
        with option_context("future.infer_string", infer_string):
            # 创建包含两列数据的数据帧
            df = DataFrame({"a": ["A", "B", "C"], "b": b_col})
            
            # 调用辅助函数检查绘图是否正常工作
            _check_plot_works(df.plot.scatter, x=x, y=y)

    @pytest.mark.parametrize("ordered", [True, False])
    @pytest.mark.parametrize(
        "categories",
        (["setosa", "versicolor", "virginica"], ["versicolor", "virginica", "setosa"]),
    )
    # 定义测试方法，用于测试分类数据的散点图颜色分布
    def test_scatterplot_color_by_categorical(self, ordered, categories):
        # 创建包含两列数值的数据帧
        df = DataFrame(
            [[5.1, 3.5], [4.9, 3.0], [7.0, 3.2], [6.4, 3.2], [5.9, 3.0]],
            columns=["length", "width"],
        )
        # 添加一个名为'species'的分类列
        df["species"] = pd.Categorical(
            ["setosa", "setosa", "virginica", "virginica", "versicolor"],
            ordered=ordered,
            categories=categories,
        )
        # 绘制散点图，按'species'列着色
        ax = df.plot.scatter(x=0, y=1, c="species")
        # 获取颜色条集合
        (colorbar_collection,) = ax.collections
        # 获取颜色条对象
        colorbar = colorbar_collection.colorbar
        
        # 预期的刻度值
        expected_ticks = np.array([0.5, 1.5, 2.5])
        # 获取实际的刻度值
        result_ticks = colorbar.get_ticks()
        # 断言刻度值与预期值相等
        tm.assert_numpy_array_equal(result_ticks, expected_ticks)
        
        # 预期的边界值
        expected_boundaries = np.array([0.0, 1.0, 2.0, 3.0])
        # 获取实际的边界值
        result_boundaries = colorbar._boundaries
        # 断言边界值与预期值相等
        tm.assert_numpy_array_equal(result_boundaries, expected_boundaries)
        
        # 预期的刻度标签
        expected_yticklabels = categories
        # 获取实际的刻度标签
        result_yticklabels = [i.get_text() for i in colorbar.ax.get_ymajorticklabels()]
        # 断言刻度标签与预期值相等
        assert all(i == j for i, j in zip(result_yticklabels, expected_yticklabels))

    @pytest.mark.parametrize("x, y", [("x", "y"), ("y", "x"), ("y", "y")])
    # 定义测试方法，用于测试散点图绘制时包含分类数据的情况
    def test_plot_scatter_with_categorical_data(self, x, y):
        # 在数据帧中创建包含两列数据的字典
        df = DataFrame({"x": [1, 2, 3, 4], "y": pd.Categorical(["a", "b", "a", "c"])})
        
        # 调用辅助函数检查绘图是否正常工作
        _check_plot_works(df.plot.scatter, x=x, y=y)

    @pytest.mark.parametrize("x, y, c", [("x", "y", "z"), (0, 1, 2)])
    # 定义测试方法，用于测试散点图绘制时包含数值型颜色数据的情况
    def test_plot_scatter_with_c(self, x, y, c):
        # 创建包含随机整数的数据帧
        df = DataFrame(
            np.random.default_rng(2).integers(low=0, high=100, size=(6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["x", "y", "z", "four"],
        )
        
        # 绘制散点图，使用指定的列作为颜色
        ax = df.plot.scatter(x=x, y=y, c=c)
        # 断言默认使用的颜色映射名称为'Greys'
        assert ax.collections[0].cmap.name == "Greys"
        
        # 断言颜色条的Y轴标签为'z'
        assert ax.collections[0].colorbar.ax.get_ylabel() == "z"
    def test_plot_scatter_with_c_props(self):
        # 创建一个包含随机整数的 DataFrame，6行4列
        df = DataFrame(
            np.random.default_rng(2).integers(low=0, high=100, size=(6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["x", "y", "z", "four"],
        )
        # 设置 colormap 为 cubehelix，绘制散点图
        cm = "cubehelix"
        ax = df.plot.scatter(x="x", y="y", c="z", colormap=cm)
        # 断言第一个集合的 colormap 名称为 cubehelix
        assert ax.collections[0].cmap.name == cm

        # 验证关闭颜色条是否生效
        ax = df.plot.scatter(x="x", y="y", c="z", colorbar=False)
        assert ax.collections[0].colorbar is None

        # 验证仍然可以绘制单一颜色的散点图
        ax = df.plot.scatter(x=0, y=1, c="red")
        assert ax.collections[0].colorbar is None
        # 检查颜色是否为红色
        _check_colors(ax.collections, facecolors=["r"])

    def test_plot_scatter_with_c_array(self):
        # 确保可以直接传递 np.array 到 matplotlib
        # 参考 https://github.com/pandas-dev/pandas/issues/8852 的 bug 报告

        df = DataFrame({"A": [1, 2], "B": [3, 4]})
        red_rgba = [1.0, 0.0, 0.0, 1.0]
        green_rgba = [0.0, 1.0, 0.0, 1.0]
        rgba_array = np.array([red_rgba, green_rgba])
        ax = df.plot.scatter(x="A", y="B", c=rgba_array)
        # 断言非 colormap 路径中点的面颜色与提供的值相同
        tm.assert_numpy_array_equal(ax.collections[0].get_facecolor(), rgba_array)
        # 下一个绘图中不测试面的颜色，因为它们依赖于 spring colormap，其颜色可能会变化
        float_array = np.array([0.0, 1.0])
        df.plot.scatter(x="A", y="B", c=float_array, cmap="spring")

    def test_plot_scatter_with_s(self):
        # 这是 GH 32904 的参考

        df = DataFrame(
            np.random.default_rng(2).random((10, 3)) * 100, columns=["a", "b", "c"]
        )

        ax = df.plot.scatter(x="a", y="b", s="c")
        # 断言 DataFrame 中 "c" 列的值与绘图集合中点的大小相同
        tm.assert_numpy_array_equal(df["c"].values, right=ax.collections[0].get_sizes())

    def test_plot_scatter_with_norm(self):
        # 在修复 GH 45809 时添加

        df = DataFrame(
            np.random.default_rng(2).random((10, 3)) * 100, columns=["a", "b", "c"]
        )
        # 创建一个对数归一化对象
        norm = mpl.colors.LogNorm()
        ax = df.plot.scatter(x="a", y="b", c="c", norm=norm)
        # 断言集合中第一个对象的归一化对象是预期的对数归一化对象
        assert ax.collections[0].norm is norm
    def test_plot_scatter_without_norm(self):
        # 添加注释以修复 GH 45809 时的问题
        # 创建一个包含随机数据的 DataFrame，共 10 行 3 列，数值范围在 0 到 100 之间
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)) * 100, columns=["a", "b", "c"]
        )
        # 绘制散点图，指定 x 轴为列 'a'，y 轴为列 'b'，颜色由列 'c' 决定
        ax = df.plot.scatter(x="a", y="b", c="c")
        # 获取散点图的归一化对象
        plot_norm = ax.collections[0].norm
        # 计算 DataFrame 列 'c' 的最小值和最大值
        color_min_max = (df.c.min(), df.c.max())
        # 创建默认的归一化对象，以 DataFrame 列 'c' 的最小值和最大值为范围
        default_norm = mpl.colors.Normalize(*color_min_max)
        # 遍历列 'c' 的每个值，检查其归一化后的值是否与默认归一化对象一致
        for value in df.c:
            assert plot_norm(value) == default_norm(value)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},  # 测试默认参数
            {"legend": False},  # 测试禁用图例
            {"default_axes": True, "subplots": True},  # 测试默认轴和子图
            {"stacked": True},  # 测试堆叠条形图
        ],
    )
    def test_plot_bar(self, kwargs):
        # 创建一个包含随机标准正态分布数据的 DataFrame，6 行 4 列
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )

        # 使用 _check_plot_works 函数测试绘制条形图的功能，传入参数 kwargs
        _check_plot_works(df.plot.bar, **kwargs)

    @pytest.mark.slow
    def test_plot_bar_int_col(self):
        # 创建一个包含随机标准正态分布数据的 DataFrame，10 行 15 列
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 15)),
            index=list(string.ascii_letters[:10]),
            columns=range(15),
        )
        # 使用 _check_plot_works 函数测试绘制条形图的功能，使用默认参数
        _check_plot_works(df.plot.bar)

    @pytest.mark.slow
    def test_plot_bar_ticks(self):
        # 创建一个包含两列数据的 DataFrame
        df = DataFrame({"a": [0, 1], "b": [1, 0]})
        # 测试绘制条形图，并检查 x 轴刻度的属性，设置 x 轴标签旋转角度为 90 度
        ax = _check_plot_works(df.plot.bar)
        _check_ticks_props(ax, xrot=90)

        # 绘制条形图，设置 x 轴标签旋转角度为 35 度，字体大小为 10
        ax = df.plot.bar(rot=35, fontsize=10)
        _check_ticks_props(ax, xrot=35, xlabelsize=10, ylabelsize=10)

    @pytest.mark.slow
    def test_plot_barh_ticks(self):
        # 创建一个包含两列数据的 DataFrame
        df = DataFrame({"a": [0, 1], "b": [1, 0]})
        # 测试绘制水平条形图，并检查 y 轴刻度的属性，设置 y 轴标签旋转角度为 0 度
        ax = _check_plot_works(df.plot.barh)
        _check_ticks_props(ax, yrot=0)

        # 绘制水平条形图，设置 y 轴标签旋转角度为 55 度，字体大小为 11
        ax = df.plot.barh(rot=55, fontsize=11)
        _check_ticks_props(ax, yrot=55, ylabelsize=11, xlabelsize=11)

    def test_boxplot(self, hist_df):
        # 获取传入的 DataFrame 对象
        df = hist_df
        # 获取 DataFrame 中的数值列
        numeric_cols = df._get_numeric_data().columns
        # 使用 pprint_thing 函数获取数值列的标签列表
        labels = [pprint_thing(c) for c in numeric_cols]

        # 测试绘制箱线图，并检查 x 轴刻度标签的文本
        ax = _check_plot_works(df.plot.box)
        _check_text_labels(ax.get_xticklabels(), labels)
        # 断言 x 轴刻度的位置是从 1 到数值列长度的整数数组
        tm.assert_numpy_array_equal(
            ax.xaxis.get_ticklocs(), np.arange(1, len(numeric_cols) + 1)
        )
        # 断言绘制的箱线图的线条数是数值列数的 7 倍
        assert len(ax.lines) == 7 * len(numeric_cols)

    def test_boxplot_series(self, hist_df):
        # 获取传入的 DataFrame 对象
        df = hist_df
        # 获取 DataFrame 中的 'height' 列作为 Series 对象
        series = df["height"]
        # 绘制 'height' 列的箱线图，设置 x 轴标签旋转角度为 40 度
        axes = series.plot.box(rot=40)
        _check_ticks_props(axes, xrot=40, yrot=0)

        # 测试绘制 'height' 列的箱线图
        _check_plot_works(series.plot.box)
    # 测试绘制数据框中数值列的箱线图，并验证位置是否正确
    def test_boxplot_series_positions(self, hist_df):
        # 将输入的历史数据框赋给 df 变量
        df = hist_df
        # 定义箱线图中的位置数组
        positions = np.array([1, 6, 7])
        # 在当前数据框上绘制箱线图，并返回绘图对象 ax
        ax = df.plot.box(positions=positions)
        # 获取数据框中的数值列
        numeric_cols = df._get_numeric_data().columns
        # 对数值列的每个列名进行格式化处理，生成标签列表
        labels = [pprint_thing(c) for c in numeric_cols]
        # 检查 x 轴标签是否正确设置
        _check_text_labels(ax.get_xticklabels(), labels)
        # 断言 x 轴的刻度位置与预期位置数组一致
        tm.assert_numpy_array_equal(ax.xaxis.get_ticklocs(), positions)
        # 断言绘图对象中线条的数量是否符合预期
        assert len(ax.lines) == 7 * len(numeric_cols)

    # 测试绘制数据框中数值列的水平箱线图，并验证标签、线条数量是否正确
    def test_boxplot_vertical(self, hist_df):
        # 将输入的历史数据框赋给 df 变量
        df = hist_df
        # 获取数据框中的数值列
        numeric_cols = df._get_numeric_data().columns
        # 对数值列的每个列名进行格式化处理，生成标签列表
        labels = [pprint_thing(c) for c in numeric_cols]

        # 如果是水平箱线图，y 轴标签会被旋转
        ax = df.plot.box(rot=50, fontsize=8, vert=False)
        # 检查绘图对象的 ticks 属性是否符合预期设置
        _check_ticks_props(ax, xrot=0, yrot=50, ylabelsize=8)
        # 检查 y 轴标签是否正确设置
        _check_text_labels(ax.get_yticklabels(), labels)
        # 断言绘图对象中线条的数量是否符合预期
        assert len(ax.lines) == 7 * len(numeric_cols)

    # 测试绘制数据框中数值列的水平子图箱线图，并验证标签、坐标轴缩放是否正确
    @pytest.mark.filterwarnings("ignore:Attempt:UserWarning")
    def test_boxplot_vertical_subplots(self, hist_df):
        # 将输入的历史数据框赋给 df 变量
        df = hist_df
        # 获取数据框中的数值列
        numeric_cols = df._get_numeric_data().columns
        # 对数值列的每个列名进行格式化处理，生成标签列表
        labels = [pprint_thing(c) for c in numeric_cols]
        # 调用辅助函数检查绘图是否正常工作，并返回绘图对象 axes
        axes = _check_plot_works(
            df.plot.box,
            default_axes=True,
            subplots=True,
            vert=False,
            logx=True,
        )
        # 检查绘图对象的形状是否符合预期
        _check_axes_shape(axes, axes_num=3, layout=(1, 3))
        # 检查绘图对象的坐标轴缩放是否正确设置
        _check_ax_scales(axes, xaxis="log")
        # 对每个子图的 y 轴标签进行检查，确保设置正确
        for ax, label in zip(axes, labels):
            _check_text_labels(ax.get_yticklabels(), [label])
            # 断言每个子图中线条的数量是否符合预期
            assert len(ax.lines) == 7

    # 测试绘制数据框中数值列的垂直箱线图，并验证标签、刻度位置是否正确
    def test_boxplot_vertical_positions(self, hist_df):
        # 将输入的历史数据框赋给 df 变量
        df = hist_df
        # 获取数据框中的数值列
        numeric_cols = df._get_numeric_data().columns
        # 对数值列的每个列名进行格式化处理，生成标签列表
        labels = [pprint_thing(c) for c in numeric_cols]
        # 定义箱线图中的位置数组
        positions = np.array([3, 2, 8])
        # 在当前数据框上绘制垂直箱线图，并返回绘图对象 ax
        ax = df.plot.box(positions=positions, vert=False)
        # 检查 y 轴标签是否正确设置
        _check_text_labels(ax.get_yticklabels(), labels)
        # 断言 y 轴的刻度位置与预期位置数组一致
        tm.assert_numpy_array_equal(ax.yaxis.get_ticklocs(), positions)
        # 断言绘图对象中线条的数量是否符合预期
        assert len(ax.lines) == 7 * len(numeric_cols)

    # 测试绘制数据框箱线图时返回类型为无效值时是否引发 ValueError 异常
    def test_boxplot_return_type_invalid(self):
        # 创建一个随机数据框 df，包含四列，六行的标准正态分布数据
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        # 定义期望的错误消息字符串
        msg = "return_type must be {None, 'axes', 'dict', 'both'}"
        # 使用 pytest 的断言检查是否引发 ValueError 异常，并验证错误消息是否匹配
        with pytest.raises(ValueError, match=msg):
            df.plot.box(return_type="not_a_type")

    # 测试绘制数据框箱线图时返回类型为无效类型时的情况
    @pytest.mark.parametrize("return_type", ["dict", "axes", "both"])
    def test_boxplot_return_type_invalid_type(self, return_type):
        # 创建一个随机数据框 df，包含四列，六行的标准正态分布数据
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        # 调用绘图函数，并根据参数 return_type 返回不同类型的结果
        result = df.plot.box(return_type=return_type)
        # 使用辅助函数检查返回类型是否符合预期
        _check_box_return_type(result, return_type)
    # 测试 DataFrame 绘制核密度估计图（KDE）
    def test_kde_df(self):
        # 导入 scipy，如果导入失败则跳过这个测试
        pytest.importorskip("scipy")
        # 创建一个包含随机标准正态分布数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        # 调用 _check_plot_works 函数检查绘图是否正常工作，并返回绘图对象
        ax = _check_plot_works(df.plot, kind="kde")
        # 准备预期的列名的可打印表示形式
        expected = [pprint_thing(c) for c in df.columns]
        # 检查图例标签是否符合预期
        _check_legend_labels(ax, labels=expected)
        # 检查坐标轴刻度属性
        _check_ticks_props(ax, xrot=0)

    # 测试 DataFrame 绘制带有旋转角度的核密度估计图（KDE）
    def test_kde_df_rot(self):
        # 导入 scipy，如果导入失败则跳过这个测试
        pytest.importorskip("scipy")
        # 创建一个包含随机标准正态分布数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        # 调用 DataFrame 的 plot 方法绘制核密度估计图，并设置旋转角度和字体大小
        ax = df.plot(kind="kde", rot=20, fontsize=5)
        # 检查坐标轴刻度属性，包括 x 轴旋转角度和标签字体大小
        _check_ticks_props(ax, xrot=20, xlabelsize=5, ylabelsize=5)

    # 测试 DataFrame 绘制子图形式的核密度估计图（KDE）
    def test_kde_df_subplots(self):
        # 导入 scipy，如果导入失败则跳过这个测试
        pytest.importorskip("scipy")
        # 创建一个包含随机标准正态分布数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        # 调用 _check_plot_works 函数检查绘图是否正常工作，返回绘图对象数组
        axes = _check_plot_works(
            df.plot,
            default_axes=True,
            kind="kde",
            subplots=True,
        )
        # 检查子图的形状是否符合预期，包括子图数量和布局
        _check_axes_shape(axes, axes_num=4, layout=(4, 1))

    # 测试 DataFrame 绘制带有对数 Y 轴的核密度估计图（KDE）
    def test_kde_df_logy(self):
        # 导入 scipy，如果导入失败则跳过这个测试
        pytest.importorskip("scipy")
        # 创建一个包含随机标准正态分布数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        # 调用 DataFrame 的 plot 方法绘制核密度估计图，并设置对数 Y 轴和子图形式显示
        axes = df.plot(kind="kde", logy=True, subplots=True)
        # 检查绘图对象数组的 Y 轴刻度是否为对数形式
        _check_ax_scales(axes, yaxis="log")

    # 测试包含缺失值的 DataFrame 绘制核密度估计图（KDE）
    def test_kde_missing_vals(self):
        # 导入 scipy，如果导入失败则跳过这个测试
        pytest.importorskip("scipy")
        # 创建一个包含随机均匀分布数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).uniform(size=(100, 4)))
        # 在 DataFrame 中的第一行第一列位置插入 NaN 值
        df.loc[0, 0] = np.nan
        # 调用 _check_plot_works 函数检查绘图是否正常工作，并绘制核密度估计图
        _check_plot_works(df.plot, kind="kde")

    # 测试 DataFrame 绘制直方图
    def test_hist_df(self):
        # 创建一个包含随机标准正态分布数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((100, 4)))

        # 调用 _check_plot_works 函数检查绘图是否正常工作，并返回绘图对象
        ax = _check_plot_works(df.plot.hist)
        # 准备预期的列名的可打印表示形式
        expected = [pprint_thing(c) for c in df.columns]
        # 检查图例标签是否符合预期
        _check_legend_labels(ax, labels=expected)

        # 调用 _check_plot_works 函数检查绘图是否正常工作，返回绘图对象数组
        axes = _check_plot_works(
            df.plot.hist,
            default_axes=True,
            subplots=True,
            logy=True,
        )
        # 检查子图的形状是否符合预期，包括子图数量和布局，并检查 Y 轴是否为对数形式
        _check_axes_shape(axes, axes_num=4, layout=(4, 1))
        _check_ax_scales(axes, yaxis="log")

    # 测试 Series 绘制直方图
    def test_hist_df_series(self):
        # 创建一个包含随机数据的 Series
        series = Series(np.random.default_rng(2).random(10))
        # 调用 Series 的 plot 方法绘制直方图，并设置 x 轴旋转角度
        axes = series.plot.hist(rot=40)
        # 检查坐标轴刻度属性，包括 x 轴旋转角度
        _check_ticks_props(axes, xrot=40, yrot=0)

    # 测试 Series 绘制累积密度函数直方图
    def test_hist_df_series_cumulative_density(self):
        # 创建一个包含随机数据的 Series
        series = Series(np.random.default_rng(2).random(10))
        # 调用 Series 的 plot 方法绘制累积密度函数直方图，并设置 bins 和 density 参数
        ax = series.plot.hist(cumulative=True, bins=4, density=True)
        # 检查最后一个柱子（索引为 5）的高度是否为 1.0
        rects = [x for x in ax.get_children() if isinstance(x, mpl.patches.Rectangle)]
        tm.assert_almost_equal(rects[-1].get_height(), 1.0)

    # 测试 Series 绘制累积直方图
    def test_hist_df_series_cumulative(self):
        # 创建一个包含随机数据的 Series
        series = Series(np.random.default_rng(2).random(10))
        # 调用 Series 的 plot 方法绘制累积直方图，并设置 bins 参数
        ax = series.plot.hist(cumulative=True, bins=4)
        # 检查倒数第二个柱子的高度是否为 10.0
        rects = [x for x in ax.get_children() if isinstance(x, mpl.patches.Rectangle)]
        tm.assert_almost_equal(rects[-2].get_height(), 10.0)
    # 测试函数，用于测试 DataFrame 的直方图绘制及其参数设置
    def test_hist_df_orientation(self):
        # 创建一个包含随机数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
        # 使用 DataFrame 的 plot.hist() 方法绘制直方图，设置旋转角度、字体大小和方向
        axes = df.plot.hist(rot=50, fontsize=8, orientation="horizontal")
        # 检查绘图后的坐标轴标签旋转情况
        _check_ticks_props(axes, xrot=0, yrot=50, ylabelsize=8)

    # 参数化测试函数，测试带有权重参数的直方图绘制
    @pytest.mark.parametrize("weight_shape", [(100,), (100, 2)])
    def test_hist_weights(self, weight_shape):
        # GH 33173 GitHub issue 33173 的相关说明
        weights = 0.1 * np.ones(shape=weight_shape)
        # 创建一个包含随机数据的 DataFrame
        df = DataFrame(
            dict(zip(["A", "B"], np.random.default_rng(2).standard_normal((2, 100))))
        )

        # 检查带有权重参数的直方图绘制情况
        ax1 = _check_plot_works(df.plot, kind="hist", weights=weights)
        # 检查不带权重参数的直方图绘制情况
        ax2 = _check_plot_works(df.plot, kind="hist")

        # 获取带权重参数的直方图条形块的高度
        patch_height_with_weights = [patch.get_height() for patch in ax1.patches]

        # 获取不带权重参数的直方图条形块的原始高度，并手动乘以示例权重
        expected_patch_height = [0.1 * patch.get_height() for patch in ax2.patches]

        # 使用测试工具检查两种情况下直方图条形块的高度是否近似相等
        tm.assert_almost_equal(patch_height_with_weights, expected_patch_height)

    # 内部函数，用于检查条形图块的坐标
    def _check_box_coord(
        self,
        patches,
        expected_y=None,
        expected_h=None,
        expected_x=None,
        expected_w=None,
    ):
        # 获取条形图块的 y 坐标、高度、x 坐标和宽度
        result_y = np.array([p.get_y() for p in patches])
        result_height = np.array([p.get_height() for p in patches])
        result_x = np.array([p.get_x() for p in patches])
        result_width = np.array([p.get_width() for p in patches])
        # 数据类型根据上述值确定，不需要检查

        # 如果有预期的 y 坐标，使用测试工具检查结果与预期是否相等
        if expected_y is not None:
            tm.assert_numpy_array_equal(result_y, expected_y, check_dtype=False)
        # 如果有预期的高度，使用测试工具检查结果与预期是否相等
        if expected_h is not None:
            tm.assert_numpy_array_equal(result_height, expected_h, check_dtype=False)
        # 如果有预期的 x 坐标，使用测试工具检查结果与预期是否相等
        if expected_x is not None:
            tm.assert_numpy_array_equal(result_x, expected_x, check_dtype=False)
        # 如果有预期的宽度，使用测试工具检查结果与预期是否相等
        if expected_w is not None:
            tm.assert_numpy_array_equal(result_width, expected_w, check_dtype=False)

    # 参数化测试函数，测试包含不同数据结构的 DataFrame 的直方图绘制
    @pytest.mark.parametrize(
        "data",
        [
            {
                "A": np.repeat(np.array([1, 2, 3, 4, 5]), np.array([10, 9, 8, 7, 6])),
                "B": np.repeat(np.array([1, 2, 3, 4, 5]), np.array([8, 8, 8, 8, 8])),
                "C": np.repeat(np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10])),
            },
            {
                "A": np.repeat(
                    np.array([np.nan, 1, 2, 3, 4, 5]), np.array([3, 10, 9, 8, 7, 6])
                ),
                "B": np.repeat(
                    np.array([1, np.nan, 2, 3, 4, 5]), np.array([8, 3, 8, 8, 8, 8])
                ),
                "C": np.repeat(
                    np.array([1, 2, 3, np.nan, 4, 5]), np.array([6, 7, 8, 3, 9, 10])
                ),
            },
        ],
    )
    # 测试绘制整数列的折线图功能
    def test_plot_int_columns(self):
        # 创建一个具有标准正态分布数据的 DataFrame，并进行累积求和
        df = DataFrame(np.random.default_rng(2).standard_normal((100, 4))).cumsum()
        # 调用 _check_plot_works 函数，验证绘图功能正常，包括图例显示
        _check_plot_works(df.plot, legend=True)

    @pytest.mark.parametrize(
        "markers",
        [
            # 参数化测试用例，测试不同的标记样式
            {0: "^", 1: "+", 2: "o"},
            {0: "^", 1: "+"},
            ["^", "+", "o"],
            ["^", "+"],
        ],
    )
    # 测试根据列设置样式的功能
    def test_style_by_column(self, markers):
        # 清除当前图形，并添加一个子图
        fig = plt.gcf()
        fig.clf()
        fig.add_subplot(111)
        # 创建一个具有标准正态分布数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)))
        # 使用 markers 参数设置图表样式，并获取绘图对象
        ax = df.plot(style=markers)
        # 验证每条线的标记是否与 markers 中指定的相匹配
        for idx, line in enumerate(ax.get_lines()[: len(markers)]):
            assert line.get_marker() == markers[idx]

    # 测试不显示折线图的图例功能
    def test_line_label_none(self):
        # 创建一个 Series 对象
        s = Series([1, 2])
        # 绘制默认样式的折线图，并验证图例为空
        ax = s.plot()
        assert ax.get_legend() is None

        # 绘制带有图例的折线图，并验证图例文本为空字符串
        ax = s.plot(legend=True)
        assert ax.get_legend().get_texts()[0].get_text() == ""

    @pytest.mark.parametrize(
        "props, expected",
        [
            # 参数化测试用例，测试不同的绘图属性设置
            ("boxprops", "boxes"),
            ("whiskerprops", "whiskers"),
            ("capprops", "caps"),
            ("medianprops", "medians"),
        ],
    )
    # 测试通过关键字参数设置绘图箱线图的属性功能
    def test_specified_props_kwd_plot_box(self, props, expected):
        # 创建一个包含随机数的 DataFrame
        df = DataFrame({k: np.random.default_rng(2).random(100) for k in "ABC"})
        # 使用 kwd 参数设置指定的绘图属性，并绘制箱线图
        kwd = {props: {"color": "C1"}}
        result = df.plot.box(return_type="dict", **kwd)

        # 验证箱线的颜色是否为指定的 "C1"
        assert result[expected][0].get_color() == "C1"

    # 测试无序时间序列的绘制功能
    def test_unordered_ts(self):
        # 创建一个时间索引和数值的 DataFrame
        index = [date(2012, 10, 1), date(2012, 9, 1), date(2012, 8, 1)]
        values = [3.0, 2.0, 1.0]
        df = DataFrame(
            np.array(values),
            index=index,
            columns=["test"],
        )
        # 绘制 DataFrame 的折线图，并验证 X 轴和 Y 轴的数据是否与预期相同
        ax = df.plot()
        xticks = ax.lines[0].get_xdata()
        tm.assert_numpy_array_equal(xticks, np.array(index, dtype=object))
        ydata = ax.lines[0].get_ydata()
        tm.assert_numpy_array_equal(ydata, np.array(values))

        # 即使在不对数据进行排序的情况下传递给 matplotlib，刻度依然会排序
        xticks = ax.xaxis.get_ticklabels()
        xlocs = [x.get_position()[0] for x in xticks]
        assert Index(xlocs).is_monotonic_increasing
        xlabels = [x.get_text() for x in xticks]
        assert pd.to_datetime(xlabels, format="%Y-%m-%d").is_monotonic_increasing

    @pytest.mark.parametrize("kind", plotting.PlotAccessor._common_kinds)
    # 测试不同绘图类型的功能
    def test_kind_both_ways(self, kind):
        pytest.importorskip("scipy")
        # 创建一个包含单列 "x" 的 DataFrame
        df = DataFrame({"x": [1, 2, 3]})
        # 绘制指定类型的图表
        df.plot(kind=kind)
        # 使用 getattr 方法调用相同类型的绘图方法
        getattr(df.plot, kind)()

    @pytest.mark.parametrize("kind", ["scatter", "hexbin"])
    # 测试散点图和六边形箱图的绘制功能
    def test_kind_both_ways_x_y(self, kind):
        pytest.importorskip("scipy")
        # 创建一个包含单列 "x" 的 DataFrame
        df = DataFrame({"x": [1, 2, 3]})
        # 绘制指定类型的图表，并传入 X 和 Y 轴的数据
        df.plot("x", "x", kind=kind)
        # 使用 getattr 方法调用相同类型的绘图方法，并传入 X 和 Y 轴的数据
        getattr(df.plot, kind)("x", "x")
    # 定义一个测试方法，用于测试当数据框中的数据无法绘制时是否正确引发异常
    def test_all_invalid_plot_data(self, kind):
        # 创建一个包含单个列"abcd"的数据框
        df = DataFrame(list("abcd"))
        # 设置错误消息
        msg = "no numeric data to plot"
        # 使用 pytest 的 raises 方法检查是否引发了 TypeError 异常，并验证异常消息是否匹配预期
        with pytest.raises(TypeError, match=msg):
            # 调用数据框的 plot 方法，尝试绘制指定类型(kind)的图表
            df.plot(kind=kind)

    # 使用 pytest 的参数化标记，定义一个测试方法，用于测试部分数据无法绘制时是否正确引发异常
    @pytest.mark.parametrize(
        "kind", list(plotting.PlotAccessor._common_kinds) + ["area"]
    )
    def test_partially_invalid_plot_data_numeric(self, kind):
        # 创建一个包含随机数数据的数据框，数据类型为 object
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            dtype=object,
        )
        # 随机选择数据框的部分行，将其设置为字符串"a"，以模拟部分数据无法绘制的情况
        df[np.random.default_rng(2).random(df.shape[0]) > 0.5] = "a"
        # 设置错误消息
        msg = "no numeric data to plot"
        # 使用 pytest 的 raises 方法检查是否引发了 TypeError 异常，并验证异常消息是否匹配预期
        with pytest.raises(TypeError, match=msg):
            # 调用数据框的 plot 方法，尝试绘制指定类型(kind)的图表
            df.plot(kind=kind)

    # 定义一个测试方法，用于测试当使用无效的图表类型时是否正确引发异常
    def test_invalid_kind(self):
        # 创建一个包含随机数数据的数据框
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
        # 设置错误消息
        msg = "invalid_plot_kind is not a valid plot kind"
        # 使用 pytest 的 raises 方法检查是否引发了 ValueError 异常，并验证异常消息是否匹配预期
        with pytest.raises(ValueError, match=msg):
            # 调用数据框的 plot 方法，尝试绘制指定类型(kind)的图表，这里是一个无效的类型
            df.plot(kind="invalid_plot_kind")

    # 使用 pytest 的参数化标记，定义一个测试方法，用于测试无效的 x 和 y 参数时是否正确引发异常
    @pytest.mark.parametrize(
        "x,y,lbl",
        [
            (["B", "C"], "A", "a"),       # x 是列表，y 是单个值
            (["A"], ["B", "C"], ["b", "c"]),  # x 是单个值，y 是列表
        ],
    )
    def test_invalid_xy_args(self, x, y, lbl):
        # 创建一个包含指定列和数据的数据框
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        # 使用 pytest 的 raises 方法检查是否引发了 ValueError 异常，并验证异常消息是否匹配预期
        with pytest.raises(ValueError, match="x must be a label or position"):
            # 调用数据框的 plot 方法，尝试绘制指定的 x 和 y 列，以及标签
            df.plot(x=x, y=y, label=lbl)

    # 定义一个测试方法，用于测试当使用无效的标签时是否正确引发异常
    def test_bad_label(self):
        # 创建一个包含指定列和数据的数据框
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        # 设置错误消息
        msg = "label should be list-like and same length as y"
        # 使用 pytest 的 raises 方法检查是否引发了 ValueError 异常，并验证异常消息是否匹配预期
        with pytest.raises(ValueError, match=msg):
            # 调用数据框的 plot 方法，尝试绘制指定的 x 和 y 列，并使用无效的标签
            df.plot(x="A", y=["B", "C"], label="bad_label")

    # 使用 pytest 的参数化标记，定义一个测试方法，用于测试当 x 或 y 参数是整数时是否正确引发异常
    @pytest.mark.parametrize("x,y", [("A", "B"), (["A"], "B")])
    def test_invalid_xy_args_dup_cols(self, x, y):
        # 创建一个包含指定数据的数据框，列名重复
        df = DataFrame([[1, 3, 5], [2, 4, 6]], columns=list("AAB"))
        # 使用 pytest 的 raises 方法检查是否引发了 ValueError 异常，并验证异常消息是否匹配预期
        with pytest.raises(ValueError, match="x must be a label or position"):
            # 调用数据框的 plot 方法，尝试绘制指定的 x 和 y 列
            df.plot(x=x, y=y)

    # 使用 pytest 的参数化标记，定义一个测试方法，测试当使用列表式的 y 参数时是否绘制成功
    @pytest.mark.parametrize(
        "x,y,lbl,colors",
        [
            ("A", ["B"], ["b"], ["red"]),                # y 是单个列名的列表
            ("A", ["B", "C"], ["b", "c"], ["red", "blue"]),  # y 是多个列名的列表，对应的标签和颜色也是列表
            (0, [1, 2], ["bokeh", "cython"], ["green", "yellow"]),  # x 和 y 都是位置索引，对应的标签和颜色也是列表
        ],
    )
    def test_y_listlike(self, x, y, lbl, colors):
        # 创建一个包含指定列和数据的数据框
        df = DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        # 调用辅助函数 _check_plot_works，验证是否能正确绘制图表，并检查标签和颜色是否匹配预期
        _check_plot_works(df.plot, x="A", y=y, label=lbl)

        # 调用数据框的 plot 方法，尝试绘制指定的 x 和 y 列，并设置对应的标签和颜色
        ax = df.plot(x=x, y=y, label=lbl, color=colors)
        # 验证绘制后图表的线条数量是否和 y 列的数量一致
        assert len(ax.lines) == len(y)
        # 调用辅助函数 _check_colors，验证绘制后的线条颜色是否和预期颜色列表一致
        _check_colors(ax.get_lines(), linecolors=colors)

    # 使用 pytest 的参数化标记，定义一个测试方法，测试当 x 和 y 参数是整数时是否能正确绘制
    @pytest.mark.parametrize("x,y,colnames", [(0, 1, ["A", "B"]), (1, 0, [0, 1])])
    def test_xy_args_integer(self, x, y, colnames):
        # 创建一个包含指定列和数据的数据框
        df = DataFrame({"A": [1, 2], "B": [3, 4]})
        # 重命名数据框的列名，根据参数指定的列名列表
        df.columns = colnames
        # 调用辅助函数 _check_plot_works，验证是否能正确绘制图表
        _check_plot_works(df.plot, x=x, y=y)
    # 定义测试函数，测试DataFrame的hexbin绘图功能的基本用法
    def test_hexbin_basic(self):
        # 创建一个包含随机数据的DataFrame，包括列'A', 'B', 'C'
        df = DataFrame(
            {
                "A": np.random.default_rng(2).uniform(size=20),
                "B": np.random.default_rng(2).uniform(size=20),
                "C": np.arange(20) + np.random.default_rng(2).uniform(size=20),
            }
        )

        # 绘制'A'和'B'列的hexbin图，并返回Axes对象
        ax = df.plot.hexbin(x="A", y="B", gridsize=10)
        # 断言绘图的集合数量为1
        assert len(ax.collections) == 1

    # 定义测试函数，测试DataFrame的hexbin绘图功能在子图中的表现
    def test_hexbin_basic_subplots(self):
        # 创建一个包含随机数据的DataFrame，包括列'A', 'B', 'C'
        df = DataFrame(
            {
                "A": np.random.default_rng(2).uniform(size=20),
                "B": np.random.default_rng(2).uniform(size=20),
                "C": np.arange(20) + np.random.default_rng(2).uniform(size=20),
            }
        )
        # 绘制'A'和'B'列的hexbin图，将图形绘制在子图中
        axes = df.plot.hexbin(x="A", y="B", subplots=True)
        # 断言图形中的Axes数量为2，一个用于绘图，另一个用于颜色条
        assert len(axes[0].figure.axes) == 2
        # 调用_check_axes_shape函数验证返回的Axes形状，期望为单个Axes对象
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    # 使用pytest参数化装饰器，定义测试函数，测试DataFrame的hexbin绘图功能与C列和reduce_C_function参数的结合使用
    @pytest.mark.parametrize("reduce_C", [None, np.std])
    def test_hexbin_with_c(self, reduce_C):
        # 创建一个包含随机数据的DataFrame，包括列'A', 'B', 'C'
        df = DataFrame(
            {
                "A": np.random.default_rng(2).uniform(size=20),
                "B": np.random.default_rng(2).uniform(size=20),
                "C": np.arange(20) + np.random.default_rng(2).uniform(size=20),
            }
        )

        # 绘制'A'和'B'列的hexbin图，并传入'C'列及reduce_C_function参数
        ax = df.plot.hexbin(x="A", y="B", C="C", reduce_C_function=reduce_C)
        # 断言绘图的集合数量为1
        assert len(ax.collections) == 1

    # 使用pytest参数化装饰器，定义测试函数，测试DataFrame的hexbin绘图功能中不同colormap参数的效果
    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({}, "BuGn"),  # 默认colormap为'BuGn'
            ({"colormap": "cubehelix"}, "cubehelix"),
            ({"cmap": "YlGn"}, "YlGn"),
        ],
    )
    def test_hexbin_cmap(self, kwargs, expected):
        # 创建一个包含随机数据的DataFrame，包括列'A', 'B', 'C'
        df = DataFrame(
            {
                "A": np.random.default_rng(2).uniform(size=20),
                "B": np.random.default_rng(2).uniform(size=20),
                "C": np.arange(20) + np.random.default_rng(2).uniform(size=20),
            }
        )
        # 绘制'A'和'B'列的hexbin图，并传入不同的参数kwargs
        ax = df.plot.hexbin(x="A", y="B", **kwargs)
        # 断言图中第一个集合的colormap名称与期望相符
        assert ax.collections[0].cmap.name == expected

    # 定义测试函数，测试DataFrame的pie绘图功能中处理错误输入的情况
    def test_pie_df_err(self):
        # 创建一个包含随机数据的DataFrame，包括5行3列的数据
        df = DataFrame(
            np.random.default_rng(2).random((5, 3)),
            columns=["X", "Y", "Z"],
            index=["a", "b", "c", "d", "e"],
        )
        # 函数调用应该引发值错误，并匹配特定错误消息
        msg = "pie requires either y column or 'subplots=True'"
        with pytest.raises(ValueError, match=msg):
            df.plot.pie()

    # 使用pytest参数化装饰器，定义测试函数，测试DataFrame的pie绘图功能正常情况下的表现
    @pytest.mark.parametrize("y", ["Y", 2])
    def test_pie_df(self, y):
        # 创建一个包含随机数据的DataFrame，包括5行3列的数据
        df = DataFrame(
            np.random.default_rng(2).random((5, 3)),
            columns=["X", "Y", "Z"],
            index=["a", "b", "c", "d", "e"],
        )
        # 调用_check_plot_works函数验证pie绘图正常工作，并传入参数y
        ax = _check_plot_works(df.plot.pie, y=y)
        # 调用_check_text_labels函数验证图中的文本标签
        _check_text_labels(ax.texts, df.index)
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"logy": True},
            {"logx": True, "logy": True},
            {"loglog": True},
        ],
    )
    # 测试函数，用于测试带有不同参数的 errorbar 图形绘制
    def test_errorbar_plot(self, kwargs):
        # 创建一个字典，包含 'x' 和 'y' 键，值分别为从 0 到 11 的数组和从 12 到 1 的数组
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        # 根据字典创建 DataFrame
        df = DataFrame(d)
        # 创建误差字典，包含 'x' 和 'y' 键，值为大小为 0.2 和 0.4 的 12 个元素的数组
        d_err = {"x": np.ones(12) * 0.2, "y": np.ones(12) * 0.4}
        # 根据误差字典创建 DataFrame
        df_err = DataFrame(d_err)

        # 检查线性图的绘制，返回绘图的 Axes 对象
        ax = _check_plot_works(df.plot, yerr=df_err, **kwargs)
        # 检查绘图是否包含误差条
        _check_has_errorbars(ax, xerr=0, yerr=2)
    # 定义一个测试函数，用于测试带有误差条的条形图绘制
    def test_errorbar_plot_bar(self):
        # 创建包含 x 和 y 列的字典数据
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        # 使用字典创建 DataFrame 对象
        df = DataFrame(d)
        # 创建包含 x 和 y 列误差的字典数据
        d_err = {"x": np.ones(12) * 0.2, "y": np.ones(12) * 0.4}
        # 使用字典创建包含误差的 DataFrame 对象
        df_err = DataFrame(d_err)
        # 调用 _check_plot_works 函数，检查带有误差条的条形图绘制，返回绘图对象
        ax = _check_plot_works(
            (df + 1).plot, yerr=df_err, xerr=df_err, kind="bar", log=True
        )
        # 检查绘图对象是否包含误差条
        _check_has_errorbars(ax, xerr=2, yerr=2)

    # 声明一个慢速测试函数，测试带有 yerr 数组的误差条绘制
    @pytest.mark.slow
    def test_errorbar_plot_yerr_array(self):
        # 创建包含 x 和 y 列的字典数据
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        # 使用字典创建 DataFrame 对象
        df = DataFrame(d)
        # 调用 _check_plot_works 函数，检查带有 yerr 数组的误差条绘制，返回绘图对象
        ax = _check_plot_works(df["y"].plot, yerr=np.ones(12) * 0.4)
        # 检查绘图对象是否包含误差条
        _check_has_errorbars(ax, xerr=0, yerr=1)

        # 调用 _check_plot_works 函数，检查带有 yerr 数组的误差条绘制，返回绘图对象
        ax = _check_plot_works(df.plot, yerr=np.ones((2, 12)) * 0.4)
        # 检查绘图对象是否包含误差条
        _check_has_errorbars(ax, xerr=0, yerr=2)

    # 声明一个慢速测试函数，测试带有列名误差条的绘制
    @pytest.mark.slow
    @pytest.mark.parametrize("yerr", ["yerr", "誤差"])
    def test_errorbar_plot_column_name(self, yerr):
        # 创建包含 x 和 y 列的字典数据
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        # 使用字典创建 DataFrame 对象
        df = DataFrame(d)
        # 创建名为 yerr 的列，赋值为误差的数组
        df[yerr] = np.ones(12) * 0.2

        # 调用 _check_plot_works 函数，检查带有列名误差条的绘制，返回绘图对象
        ax = _check_plot_works(df.plot, yerr=yerr)
        # 检查绘图对象是否包含误差条
        _check_has_errorbars(ax, xerr=0, yerr=2)

        # 调用 _check_plot_works 函数，检查带有列名误差条的绘制，指定 x 和 y 列，返回绘图对象
        ax = _check_plot_works(df.plot, y="y", x="x", yerr=yerr)
        # 检查绘图对象是否包含误差条
        _check_has_errorbars(ax, xerr=0, yerr=1)

    # 声明一个慢速测试函数，测试在外部引发 ValueError 的情况
    @pytest.mark.slow
    def test_errorbar_plot_external_valueerror(self):
        # 创建包含 x 和 y 列的字典数据
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        # 使用字典创建 DataFrame 对象
        df = DataFrame(d)
        # 使用外部错误引发上下文，绘制带有随机误差的折线图，并期望引发 ValueError
        with tm.external_error_raised(ValueError):
            df.plot(yerr=np.random.default_rng(2).standard_normal(11))

    # 声明一个慢速测试函数，测试在外部引发 TypeError 的情况
    @pytest.mark.slow
    def test_errorbar_plot_external_typeerror(self):
        # 创建包含 x 和 y 列的字典数据
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        # 使用字典创建 DataFrame 对象
        df = DataFrame(d)
        # 创建包含无效类型数据的 DataFrame 对象，用于 yerr
        df_err = DataFrame({"x": ["zzz"] * 12, "y": ["zzz"] * 12})
        # 使用外部错误引发上下文，绘制带有无效类型误差的折线图，并期望引发 TypeError
        with tm.external_error_raised(TypeError):
            df.plot(yerr=df_err)

    # 声明一个慢速测试函数，测试不同类型 yerr 数据的绘制
    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["line", "bar", "barh"])
    @pytest.mark.parametrize(
        "y_err",
        [
            Series(np.ones(12) * 0.2, name="x"),
            DataFrame({"x": np.ones(12) * 0.2, "y": np.ones(12) * 0.4}),
        ],
    )
    def test_errorbar_plot_different_yerr(self, kind, y_err):
        # 创建包含 x 和 y 列的字典数据
        df = DataFrame({"x": np.arange(12), "y": np.arange(12, 0, -1)})

        # 调用 _check_plot_works 函数，检查不同类型 yerr 数据的绘制，返回绘图对象
        ax = _check_plot_works(df.plot, yerr=y_err, kind=kind)
        # 检查绘图对象是否包含误差条
        _check_has_errorbars(ax, xerr=0, yerr=2)

    # 声明一个慢速测试函数，测试不同类型 yerr 和 xerr 数据的绘制
    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["line", "bar", "barh"])
    @pytest.mark.parametrize(
        "y_err, x_err",
        [
            (
                DataFrame({"x": np.ones(12) * 0.2, "y": np.ones(12) * 0.4}),
                DataFrame({"x": np.ones(12) * 0.2, "y": np.ones(12) * 0.4}),
            ),
            (Series(np.ones(12) * 0.2, name="x"), Series(np.ones(12) * 0.2, name="x")),
            (0.2, 0.2),
        ],
    )
    def test_errorbar_plot_different_yerr(self, kind, y_err, x_err):
        # 创建包含 x 和 y 列的字典数据
        df = DataFrame({"x": np.arange(12), "y": np.arange(12, 0, -1)})

        # 调用 _check_plot_works 函数，检查不同类型 yerr 和 xerr 数据的绘制，返回绘图对象
        ax = _check_plot_works(df.plot, yerr=y_err, xerr=x_err, kind=kind)
        # 检查绘图对象是否包含误差条
        _check_has_errorbars(ax, xerr=0, yerr=2)
    # 定义测试方法，用于测试不同类型的 errorbar 图形的生成，yerr 和 xerr 可以是不同类型的输入
    def test_errorbar_plot_different_yerr_xerr(self, kind, y_err, x_err):
        # 创建包含两列数据的 DataFrame 对象
        df = DataFrame({"x": np.arange(12), "y": np.arange(12, 0, -1)})
        # 调用 _check_plot_works 函数，验证 plot 函数的正确性，生成图形对象 ax
        ax = _check_plot_works(df.plot, yerr=y_err, xerr=x_err, kind=kind)
        # 检查生成的图形对象是否包含误差条，期望 xerr 和 yerr 都为 2
        _check_has_errorbars(ax, xerr=2, yerr=2)

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["line", "bar", "barh"])
    # 定义测试方法，用于测试不同类型的 errorbar 图形在子图中的生成
    def test_errorbar_plot_different_yerr_xerr_subplots(self, kind):
        # 创建包含两列数据的 DataFrame 对象
        df = DataFrame({"x": np.arange(12), "y": np.arange(12, 0, -1)})
        # 创建包含误差数据的 DataFrame 对象
        df_err = DataFrame({"x": np.ones(12) * 0.2, "y": np.ones(12) * 0.4})
        # 调用 _check_plot_works 函数，验证 plot 函数在子图中的正确性，生成图形对象 axes
        axes = _check_plot_works(
            df.plot,
            default_axes=True,
            yerr=df_err,
            xerr=df_err,
            subplots=True,
            kind=kind,
        )
        # 检查生成的子图对象是否包含误差条，期望 xerr 和 yerr 都为 1
        _check_has_errorbars(axes, xerr=1, yerr=1)

    @pytest.mark.xfail(reason="Iterator is consumed", raises=ValueError)
    # 定义测试方法，用于测试使用迭代器作为 yerr 的情况
    def test_errorbar_plot_iterator(self):
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        df = DataFrame(d)
        # 调用 _check_plot_works 函数，验证 plot 函数的正确性，生成图形对象 ax
        # yerr 使用 itertools.repeat 生成的迭代器
        ax = _check_plot_works(df.plot, yerr=itertools.repeat(0.1, len(df)))
        # 检查生成的图形对象是否包含误差条，期望 xerr 为 0，yerr 为 2
        _check_has_errorbars(ax, xerr=0, yerr=2)

    # 定义测试方法，用于测试整数列名的情况
    def test_errorbar_with_integer_column_names(self):
        # 创建包含随机数据的 DataFrame 对象，列名为整数
        df = DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 2))))
        # 创建包含随机误差数据的 DataFrame 对象，列名为整数
        df_err = DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 2))))
        # 调用 _check_plot_works 函数，验证 plot 函数的正确性，生成图形对象 ax
        ax = _check_plot_works(df.plot, yerr=df_err)
        # 检查生成的图形对象是否包含误差条，期望 xerr 为 0，yerr 为 2
        _check_has_errorbars(ax, xerr=0, yerr=2)
        # 再次调用 _check_plot_works 函数，验证 plot 函数的正确性，yerr 为单个数值 0
        ax = _check_plot_works(df.plot, y=0, yerr=1)
        # 检查生成的图形对象是否包含误差条，期望 xerr 和 yerr 都为 1
        _check_has_errorbars(ax, xerr=0, yerr=1)

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["line", "bar"])
    # 定义测试方法，用于测试部分列名和特定 kind 的 errorbar 图形的生成
    def test_errorbar_with_partial_columns_kind(self, kind):
        # 创建包含随机数据的 DataFrame 对象，列数为 3
        df = DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 3))))
        # 创建包含部分列随机误差数据的 DataFrame 对象，列名为 [0, 2]
        df_err = DataFrame(
            np.abs(np.random.default_rng(2).standard_normal((10, 2))), columns=[0, 2]
        )
        # 调用 _check_plot_works 函数，验证 plot 函数的正确性，生成图形对象 ax
        ax = _check_plot_works(df.plot, yerr=df_err, kind=kind)
        # 检查生成的图形对象是否包含误差条，期望 xerr 为 0，yerr 为 2
        _check_has_errorbars(ax, xerr=0, yerr=2)

    @pytest.mark.slow
    # 定义测试方法，用于测试部分列名和时间索引的 errorbar 图形的生成
    def test_errorbar_with_partial_columns_dti(self):
        # 创建包含随机数据的 DataFrame 对象，列数为 3
        df = DataFrame(np.abs(np.random.default_rng(2).standard_normal((10, 3))))
        # 创建包含部分列随机误差数据的 DataFrame 对象，列名为 [0, 2]
        df_err = DataFrame(
            np.abs(np.random.default_rng(2).standard_normal((10, 2))), columns=[0, 2]
        )
        # 创建日期范围的时间索引
        ix = date_range("1/1/2000", periods=10, freq="ME")
        df.set_index(ix, inplace=True)
        df_err.set_index(ix, inplace=True)
        # 调用 _check_plot_works 函数，验证 plot 函数的正确性，生成图形对象 ax
        ax = _check_plot_works(df.plot, yerr=df_err, kind="line")
        # 检查生成的图形对象是否包含误差条，期望 xerr 为 0，yerr 为 2
        _check_has_errorbars(ax, xerr=0, yerr=2)

    @pytest.mark.slow
    @pytest.mark.parametrize("err_box", [lambda x: x, DataFrame])
    # 测试带有部分列箱型图的错误条
    def test_errorbar_with_partial_columns_box(self, err_box):
        # 创建包含 x 和 y 列的字典数据
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        # 从字典数据创建数据帧
        df = DataFrame(d)
        # 使用 err_box 函数生成错误条数据
        err = err_box({"x": np.ones(12) * 0.2, "z": np.ones(12) * 0.4})
        # 检查绘图函数是否正常工作，并获取绘图的坐标轴对象
        ax = _check_plot_works(df.plot, yerr=err)
        # 检查坐标轴对象是否包含误差条
        _check_has_errorbars(ax, xerr=0, yerr=1)

    # 使用参数化测试来测试时间序列绘图中的错误条
    @pytest.mark.parametrize("kind", ["line", "bar", "barh"])
    def test_errorbar_timeseries(self, kind):
        # 创建包含 x 和 y 列的字典数据
        d = {"x": np.arange(12), "y": np.arange(12, 0, -1)}
        # 创建包含 x 和 y 列误差的字典数据
        d_err = {"x": np.ones(12) * 0.2, "y": np.ones(12) * 0.4}

        # 检查时间序列绘图
        ix = date_range("1/1/2000", "1/1/2001", freq="ME")
        # 创建包含索引的数据帧 tdf 和 tdf_err
        tdf = DataFrame(d, index=ix)
        tdf_err = DataFrame(d_err, index=ix)

        # 检查绘图函数是否正常工作，并获取绘图的坐标轴对象
        ax = _check_plot_works(tdf.plot, yerr=tdf_err, kind=kind)
        # 检查坐标轴对象是否包含误差条
        _check_has_errorbars(ax, xerr=0, yerr=2)

        ax = _check_plot_works(tdf.plot, yerr=d_err, kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=2)

        ax = _check_plot_works(tdf.plot, y="y", yerr=tdf_err["x"], kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=1)

        ax = _check_plot_works(tdf.plot, y="y", yerr="x", kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=1)

        ax = _check_plot_works(tdf.plot, yerr=tdf_err, kind=kind)
        _check_has_errorbars(ax, xerr=0, yerr=2)

        # 检查绘图函数是否正常工作，并获取绘图的坐标轴对象
        axes = _check_plot_works(
            tdf.plot,
            default_axes=True,
            kind=kind,
            yerr=tdf_err,
            subplots=True,
        )
        # 检查坐标轴对象是否包含误差条
        _check_has_errorbars(axes, xerr=0, yerr=1)

    # 测试不对称误差条的情况
    def test_errorbar_asymmetrical(self):
        # 生成随机的 3x2x5 的数组作为误差条数据
        err = np.random.default_rng(2).random((3, 2, 5))

        # 每列分别是 [0, 1, 2, 3, 4], [3, 4, 5, 6, 7]...
        # 创建包含 0 到 14 的数据，然后转置为 DataFrame
        df = DataFrame(np.arange(15).reshape(3, 5)).T

        # 在绘图中添加误差条
        ax = df.plot(yerr=err, xerr=err / 2)

        # 获取第一个误差条的 y 值并进行期望值比较
        yerr_0_0 = ax.collections[1].get_paths()[0].vertices[:, 1]
        expected_0_0 = err[0, :, 0] * np.array([-1, 1])
        tm.assert_almost_equal(yerr_0_0, expected_0_0)

        # 检查是否抛出期望的异常消息，验证不对称误差条的形状
        msg = re.escape(
            "Asymmetrical error bars should be provided with the shape (3, 2, 5)"
        )
        with pytest.raises(ValueError, match=msg):
            df.plot(yerr=err.T)

    # 测试表格绘制功能
    def test_table(self):
        # 创建一个包含 10 行 3 列随机数据的 DataFrame，设置索引为字母序列
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        # 检查绘图函数是否正常工作，并添加表格参数
        _check_plot_works(df.plot, table=True)
        _check_plot_works(df.plot, table=df)

        # 检查 GH 35945 的用户警告
        with tm.assert_produces_warning(None):
            ax = df.plot()
            assert len(ax.tables) == 0
            # 添加表格到绘图并验证表格的存在
            plotting.table(ax, df.T)
            assert len(ax.tables) == 1
    def test_errorbar_scatter(self):
        # 创建包含随机数据的 DataFrame，用于散点图测试，包括 "x" 和 "y" 列
        df = DataFrame(
            np.abs(np.random.default_rng(2).standard_normal((5, 2))),
            index=range(5),
            columns=["x", "y"],
        )
        # 创建包含随机误差的 DataFrame，用于散点图误差条测试，包括 "x" 和 "y" 列
        df_err = DataFrame(
            np.abs(np.random.default_rng(2).standard_normal((5, 2))) / 5,
            index=range(5),
            columns=["x", "y"],
        )

        # 调用 _check_plot_works 函数测试散点图绘制，不包含误差条
        ax = _check_plot_works(df.plot.scatter, x="x", y="y")
        # 调用 _check_has_errorbars 函数验证是否添加了误差条，期望 x 方向误差为 0，y 方向误差为 0
        _check_has_errorbars(ax, xerr=0, yerr=0)
        
        # 调用 _check_plot_works 函数测试散点图绘制，包含 x 方向误差条
        ax = _check_plot_works(df.plot.scatter, x="x", y="y", xerr=df_err)
        # 调用 _check_has_errorbars 函数验证是否添加了误差条，期望 x 方向误差为 1，y 方向误差为 0
        _check_has_errorbars(ax, xerr=1, yerr=0)

        # 调用 _check_plot_works 函数测试散点图绘制，包含 y 方向误差条
        ax = _check_plot_works(df.plot.scatter, x="x", y="y", yerr=df_err)
        # 调用 _check_has_errorbars 函数验证是否添加了误差条，期望 x 方向误差为 0，y 方向误差为 1
        _check_has_errorbars(ax, xerr=0, yerr=1)
        
        # 调用 _check_plot_works 函数测试散点图绘制，同时包含 x 和 y 方向误差条
        ax = _check_plot_works(df.plot.scatter, x="x", y="y", xerr=df_err, yerr=df_err)
        # 调用 _check_has_errorbars 函数验证是否添加了误差条，期望 x 方向误差为 1，y 方向误差为 1
        _check_has_errorbars(ax, xerr=1, yerr=1)

    def test_errorbar_scatter_color(self):
        # 定义内部函数 _check_errorbar_color 用于检查误差条颜色
        def _check_errorbar_color(containers, expected, has_err="has_xerr"):
            lines = []
            # 获取包含误差条的容器并存储到 lines 列表中
            errs = next(c.lines for c in ax.containers if getattr(c, has_err, False))
            for el in errs:
                if is_list_like(el):
                    lines.extend(el)
                else:
                    lines.append(el)
            # 从 lines 列表中选出在 ax.collections 中的误差条，并验证其颜色
            err_lines = [x for x in lines if x in ax.collections]
            _check_colors(err_lines, linecolors=np.array([expected] * len(err_lines)))

        # 创建包含随机数据的 DataFrame，用于散点图测试，包括 "a", "b", "c", "d", "e" 列
        df = DataFrame(
            np.abs(np.random.default_rng(2).standard_normal((10, 5))),
            columns=["a", "b", "c", "d", "e"],
        )
        # 绘制散点图，包含 x 方向和 y 方向的误差条，颜色设置为 "red"
        ax = df.plot.scatter(x="a", y="b", xerr="d", yerr="e", c="red")
        # 验证是否添加了误差条，期望 x 方向误差为 1，y 方向误差为 1
        _check_has_errorbars(ax, xerr=1, yerr=1)
        # 验证误差条的颜色为 "red"
        _check_errorbar_color(ax.containers, "red", has_err="has_xerr")
        _check_errorbar_color(ax.containers, "red", has_err="has_yerr")

        # 绘制散点图，包含 y 方向的误差条，颜色设置为 "green"
        ax = df.plot.scatter(x="a", y="b", yerr="e", color="green")
        # 验证是否添加了误差条，期望 x 方向误差为 0，y 方向误差为 1
        _check_has_errorbars(ax, xerr=0, yerr=1)
        # 验证误差条的颜色为 "green"
        _check_errorbar_color(ax.containers, "green", has_err="has_yerr")

    def test_scatter_unknown_colormap(self):
        # GH#48726 测试用例
        # 创建包含列 "a" 和 "b" 的 DataFrame
        df = DataFrame({"a": [1, 2, 3], "b": 4})
        # 使用 pytest 检查是否引发了 ValueError 或 KeyError 异常，并验证异常消息中包含 "'unknown' is not a"
        with pytest.raises((ValueError, KeyError), match="'unknown' is not a"):
            # 绘制散点图，指定 colormap 为 "unknown"，这应该引发异常
            df.plot(x="a", y="b", colormap="unknown", kind="scatter")
    def test_sharex_and_ax(self):
        # 使用 gridspec 生成 4 个子图和相应的轴对象
        gs, axes = _generate_4_axes_via_gridspec()

        # 创建一个包含四列的 DataFrame 对象
        df = DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [1, 2, 3, 4, 5, 6],
                "c": [1, 2, 3, 4, 5, 6],
                "d": [1, 2, 3, 4, 5, 6],
            }
        )

        # 定义内部函数 _check，用于检查每个轴对象的可见性和标签
        def _check(axes):
            for ax in axes:
                assert len(ax.lines) == 1
                # 检查 y 轴标签是否可见
                _check_visible(ax.get_yticklabels(), visible=True)
            for ax in [axes[0], axes[2]]:
                # 对部分轴对象检查 x 轴标签是否不可见（sharex=True 的情况）
                _check_visible(ax.get_xticklabels(), visible=False)
                _check_visible(ax.get_xticklabels(minor=True), visible=False)
            for ax in [axes[1], axes[3]]:
                # 对部分轴对象检查 x 轴标签是否可见（sharex=True 的情况）
                _check_visible(ax.get_xticklabels(), visible=True)
                _check_visible(ax.get_xticklabels(minor=True), visible=True)

        # 在每个轴对象上绘制指定的数据列
        for ax in axes:
            df.plot(x="a", y="b", title="title", ax=ax, sharex=True)
        
        # 调整子图布局
        gs.tight_layout(plt.gcf())
        # 使用定义的 _check 函数检查轴对象的标签和线条
        _check(axes)
        # 关闭所有图形窗口
        plt.close("all")

        # 重新生成 4 个子图和轴对象
        gs, axes = _generate_4_axes_via_gridspec()
        # 使用 df.plot 方法绘制子图，设置 subplots=True 和 sharex=True
        with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
            axes = df.plot(subplots=True, ax=axes, sharex=True)
        # 使用定义的 _check 函数再次检查轴对象的标签和线条
        _check(axes)

    def test_sharex_false_and_ax(self):
        # 使用 gridspec 生成 4 个子图和相应的轴对象
        df = DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [1, 2, 3, 4, 5, 6],
                "c": [1, 2, 3, 4, 5, 6],
                "d": [1, 2, 3, 4, 5, 6],
            }
        )
        gs, axes = _generate_4_axes_via_gridspec()
        
        # 在每个轴对象上使用 df.plot 方法绘制数据，未设置 sharex=True
        # 这意味着所有轴对象的标签应保持不变
        for ax in axes:
            df.plot(x="a", y="b", title="title", ax=ax)

        # 调整子图布局
        gs.tight_layout(plt.gcf())
        # 使用 assert 检查每个轴对象的线条数量为 1
        for ax in axes:
            assert len(ax.lines) == 1
            # 检查 y 轴标签是否可见
            _check_visible(ax.get_yticklabels(), visible=True)
            # 检查 x 轴标签是否可见
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)
    def test_sharey_and_ax(self):
        # 使用 gridspec 生成 4 个子图及其轴对象
        gs, axes = _generate_4_axes_via_gridspec()

        # 创建一个包含四列数据的 DataFrame
        df = DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [1, 2, 3, 4, 5, 6],
                "c": [1, 2, 3, 4, 5, 6],
                "d": [1, 2, 3, 4, 5, 6],
            }
        )

        def _check(axes):
            # 检查每个轴对象，确保每个轴只有一个线条
            for ax in axes:
                assert len(ax.lines) == 1
                # 检查 x 轴刻度标签可见性
                _check_visible(ax.get_xticklabels(), visible=True)
                _check_visible(ax.get_xticklabels(minor=True), visible=True)
            # 对前两个轴检查 y 轴刻度标签可见性
            for ax in [axes[0], axes[1]]:
                _check_visible(ax.get_yticklabels(), visible=True)
            # 对后两个轴检查 y 轴刻度标签是否不可见
            for ax in [axes[2], axes[3]]:
                _check_visible(ax.get_yticklabels(), visible=False)

        # 在每个轴上绘制 DataFrame 的指定列的线图，并共享 y 轴
        for ax in axes:
            df.plot(x="a", y="b", title="title", ax=ax, sharey=True)
        
        # 调整布局使子图紧密排列
        gs.tight_layout(plt.gcf())
        # 对生成的子图进行检查
        _check(axes)
        # 关闭所有图形窗口
        plt.close("all")

        # 再次使用 gridspec 生成 4 个子图及其轴对象
        gs, axes = _generate_4_axes_via_gridspec()
        # 使用 df.plot() 在所有子图上绘制数据，共享 y 轴，并期望产生 UserWarning 警告
        with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
            axes = df.plot(subplots=True, ax=axes, sharey=True)

        # 再次调整布局使子图紧密排列
        gs.tight_layout(plt.gcf())
        # 对生成的子图进行检查
        _check(axes)

    def test_sharey_and_ax_tight(self):
        # 使用 gridspec 生成 4 个子图及其轴对象
        gs, axes = _generate_4_axes_via_gridspec()
        
        # 创建一个包含四列数据的 DataFrame
        df = DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [1, 2, 3, 4, 5, 6],
                "c": [1, 2, 3, 4, 5, 6],
                "d": [1, 2, 3, 4, 5, 6],
            }
        )

        # 在每个轴上绘制 DataFrame 的指定列的线图，未指定共享轴
        for ax in axes:
            df.plot(x="a", y="b", title="title", ax=ax)

        # 调整布局使子图紧密排列
        gs.tight_layout(plt.gcf())

        # 对每个轴对象检查
        for ax in axes:
            # 检查每个轴只有一个线条
            assert len(ax.lines) == 1
            # 检查 y 轴刻度标签可见性
            _check_visible(ax.get_yticklabels(), visible=True)
            # 检查 x 轴刻度标签可见性
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)
    def test_memory_leak(self, kind):
        """Check that every plot type gets properly collected."""
        # 导入必要的库 pytest，如果不存在则跳过测试
        pytest.importorskip("scipy")
        
        args = {}
        # 根据图表类型选择不同的数据框构造方式
        if kind in ["hexbin", "scatter", "pie"]:
            # 构造包含三列数据的数据框
            df = DataFrame(
                {
                    "A": np.random.default_rng(2).uniform(size=20),
                    "B": np.random.default_rng(2).uniform(size=20),
                    "C": np.arange(20) + np.random.default_rng(2).uniform(size=20),
                }
            )
            # 设置参数字典，用于绘制散点图、六边形图或饼图
            args = {"x": "A", "y": "B"}
        elif kind == "area":
            # 构造包含四列数据的数据框，所有值取绝对值
            df = DataFrame(
                np.random.default_rng(2).standard_normal((10, 4)),
                columns=Index(list("ABCD"), dtype=object),
                index=date_range("2000-01-01", periods=10, freq="B"),
            ).abs()
        else:
            # 默认情况下，构造包含四列数据的数据框
            df = DataFrame(
                np.random.default_rng(2).standard_normal((10, 4)),
                columns=Index(list("ABCD"), dtype=object),
                index=date_range("2000-01-01", periods=10, freq="B"),
            )

        # 使用弱引用来检查对象是否能被正确回收
        ref = weakref.ref(df.plot(kind=kind, **args))

        # 关闭所有 matplotlib 图形
        plt.close("all")
        # 强制进行垃圾回收
        gc.collect()
        # 断言引用的对象已经被回收
        assert ref() is None
    def test_df_gridspec_patterns_boxed(self):
        # GH 10819
        # 创建一个时间序列（Series），包含随机生成的数据，索引为从 "1/1/2000" 开始的 10 个日期
        ts = Series(
            np.random.default_rng(2).standard_normal(10),
            index=date_range("1/1/2000", periods=10),
        )

        # 定义一个函数 _get_boxed_grid，用于生成一个 3x3 的网格布局，返回四个子图对象
        def _get_boxed_grid():
            gs = mpl.gridspec.GridSpec(3, 3)
            fig = plt.figure()
            ax1 = fig.add_subplot(gs[:2, :2])  # 第一个子图占据左上角的部分
            ax2 = fig.add_subplot(gs[:2, 2])   # 第二个子图占据右上角的部分
            ax3 = fig.add_subplot(gs[2, :2])   # 第三个子图占据左下角的部分
            ax4 = fig.add_subplot(gs[2, 2])    # 第四个子图占据右下角的部分
            return ax1, ax2, ax3, ax4

        # 调用 _get_boxed_grid 函数获取子图对象
        axes = _get_boxed_grid()

        # 创建一个 DataFrame，包含随机生成的数据，10行4列，行索引使用之前生成的时间序列索引，列索引为 'A', 'B', 'C', 'D'
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            index=ts.index,
            columns=list("ABCD"),
        )

        # 在给定的子图 axes 上绘制 DataFrame df 的子图，并返回绘制后的子图对象
        axes = df.plot(subplots=True, ax=axes)

        # 针对每个子图对象进行断言，确保每个子图只有一条线
        for ax in axes:
            assert len(ax.lines) == 1

            # 检查 y 轴刻度标签是否可见，这些子图的 y 轴刻度标签不是共享的
            _check_visible(ax.get_yticklabels(), visible=True)

            # 检查 x 轴刻度标签是否可见，这些子图的 x 轴刻度标签不是共享的
            _check_visible(ax.get_xticklabels(), visible=True)

            # 检查次要 x 轴刻度标签是否可见，这些子图的 x 轴刻度标签不是共享的
            _check_visible(ax.get_xticklabels(minor=True), visible=True)

        # 关闭所有图形窗口
        plt.close("all")

        # 在同一个网格布局下绘制 DataFrame df 的子图，设置 sharex=True 和 sharey=True，并返回绘制后的子图对象
        axes = _get_boxed_grid()
        
        # 使用断言检查每个子图是否只有一条线
        with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
            axes = df.plot(subplots=True, ax=axes, sharex=True, sharey=True)

        # 针对每个子图对象进行断言，确保每个子图只有一条线
        for ax in axes:
            assert len(ax.lines) == 1

        # 分别对左列和右列的子图对象进行断言，确保左列的 y 轴刻度标签可见，右列的 y 轴刻度标签不可见
        for ax in [axes[0], axes[2]]:
            _check_visible(ax.get_yticklabels(), visible=True)

        for ax in [axes[1], axes[3]]:
            _check_visible(ax.get_yticklabels(), visible=False)

        # 分别对顶行和底行的子图对象进行断言，确保顶行的 x 轴刻度标签不可见，底行的 x 轴刻度标签可见
        for ax in [axes[0], axes[1]]:
            _check_visible(ax.get_xticklabels(), visible=False)
            _check_visible(ax.get_xticklabels(minor=True), visible=False)

        for ax in [axes[2], axes[3]]:
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)

        # 关闭所有图形窗口
        plt.close("all")

    def test_df_grid_settings(self):
        # 确保绘图默认使用 rcParams['axes.grid'] 设置，GH 9792
        _check_grid_settings(
            # 创建一个 DataFrame 包含两列 'a' 和 'b'，每列包含三个整数
            DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]}),
            # 使用 plotting.PlotAccessor._dataframe_kinds
            plotting.PlotAccessor._dataframe_kinds,
            # 传递参数 kws，设置 'x' 为 "a"，'y' 为 "b"
            kws={"x": "a", "y": "b"},
        )

    def test_plain_axes(self):
        # 提供的 ax 本身是 SubplotAxes，但是图形中也包含一个普通的 Axes 对象，GH11556
        # 创建一个图形对象 fig 和一个子图对象 ax
        fig, ax = mpl.pyplot.subplots()
        # 向图形对象 fig 中添加一个坐标系，位置是 [0.2, 0.2, 0.2, 0.2]
        fig.add_axes([0.2, 0.2, 0.2, 0.2])
        # 创建一个包含随机生成数据的时间序列，并在子图对象 ax 上绘制
        Series(np.random.default_rng(2).random(10)).plot(ax=ax)
    def test_plain_axes_df(self):
        # 测试用例：测试在普通 Axes 上绘制 DataFrame 的散点图
        # 使用 numpy 随机生成数据填充 DataFrame
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(8),
                "b": np.random.default_rng(2).standard_normal(8),
            }
        )
        # 创建一个新的 Figure 对象
        fig = mpl.pyplot.figure()
        # 在 Figure 上添加一个 Axes 对象，位置在 (0, 0) 到 (1, 1) 的区域内
        ax = fig.add_axes((0, 0, 1, 1))
        # 绘制 DataFrame 的散点图在指定的 Axes 上，根据 'a' 列着色，使用 'hsv' 颜色映射
        df.plot(kind="scatter", ax=ax, x="a", y="b", c="a", cmap="hsv")

    def test_plain_axes_make_axes_locatable(self):
        # 测试用例：测试在普通 Axes 上使用 make_axes_locatable 创建新的 Axes
        fig, ax = mpl.pyplot.subplots()
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # 使用 make_axes_locatable 方法创建一个 Axes 分割对象
        divider = make_axes_locatable(ax)
        # 在分割后的 Axes 上附加一个新的 Axes 作为 colorbar 的位置，位于右侧，大小为原来的 5%，与原 Axes 的距离为 0.05
        cax = divider.append_axes("right", size="5%", pad=0.05)
        # 在原 Axes 上绘制 Series
        Series(np.random.default_rng(2).random(10)).plot(ax=ax)
        # 在附加的 Axes 上绘制另一个 Series
        Series(np.random.default_rng(2).random(10)).plot(ax=cax)

    def test_plain_axes_make_inset_axes(self):
        # 测试用例：测试在普通 Axes 上使用 inset_axes 创建插入的 Axes
        fig, ax = mpl.pyplot.subplots()
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        # 使用 inset_axes 方法在指定位置 (loc=3) 创建一个插入的 Axes，宽度为 30%，高度为 1.0
        iax = inset_axes(ax, width="30%", height=1.0, loc=3)
        # 在原 Axes 上绘制 Series
        Series(np.random.default_rng(2).random(10)).plot(ax=ax)
        # 在插入的 Axes 上绘制另一个 Series
        Series(np.random.default_rng(2).random(10)).plot(ax=iax)

    @pytest.mark.parametrize("method", ["line", "barh", "bar"])
    def test_secondary_axis_font_size(self, method):
        # 测试用例：测试不同绘图方法下次要坐标轴的字体大小
        # GH: 12565
        # 创建一个 DataFrame，并生成相关数据
        df = (
            DataFrame(
                np.random.default_rng(2).standard_normal((15, 2)), columns=list("AB")
            )
            .assign(C=lambda df: df.B.cumsum())  # 添加新列 C，为 B 列的累积和
            .assign(D=lambda df: df.C * 1.1)  # 添加新列 D，为 C 列的 1.1 倍
        )

        fontsize = 20  # 设置字体大小
        sy = ["C", "D"]  # 设置次要坐标轴对应的列名

        kwargs = {"secondary_y": sy, "fontsize": fontsize, "mark_right": True}
        # 调用 DataFrame 的 plot 对象的指定绘图方法（line/barh/bar），并传入参数 kwargs
        ax = getattr(df.plot, method)(**kwargs)
        # 检查次要坐标轴的刻度属性，验证字体大小
        _check_ticks_props(axes=ax.right_ax, ylabelsize=fontsize)

    def test_x_string_values_ticks(self):
        # 测试用例：测试使用字符串作为 x 轴索引值时的刻度位置
        # GH: 7612, GH: 22334
        # 创建一个 DataFrame，包含销售额、访问量和日期的数据
        df = DataFrame(
            {
                "sales": [3, 2, 3],
                "visits": [20, 42, 28],
                "day": ["Monday", "Tuesday", "Wednesday"],
            }
        )
        # 在 DataFrame 上绘制面积图，以 'day' 列作为 x 轴
        ax = df.plot.area(x="day")
        ax.set_xlim(-1, 3)  # 设置 x 轴的显示范围
        # 获取 x 轴刻度标签的文本内容
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        # 获取 x 轴刻度标签的位置，并以字典形式存储标签和其位置的对应关系
        labels_position = dict(zip(xticklabels, ax.get_xticks()))
        # 断言测试：验证标签是否在正确的位置
        assert labels_position["Monday"] == 0.0
        assert labels_position["Tuesday"] == 1.0
        assert labels_position["Wednesday"] == 2.0
    def test_x_multiindex_values_ticks(self):
        # 测试多索引情况下 x 轴刻度的固定位置
        # GH: 15912
        # 创建一个多重索引，包含年份和季度
        index = MultiIndex.from_product([[2012, 2013], [1, 2]])
        # 创建一个包含随机数据的 DataFrame，使用多重索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((4, 2)),
            columns=["A", "B"],
            index=index,
        )
        # 在 DataFrame 上绘制图形，并获取对应的 Axes 对象
        ax = df.plot()
        # 设置 x 轴的显示范围
        ax.set_xlim(-1, 4)
        # 获取当前 x 轴刻度标签的文本内容
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        # 创建一个字典，将刻度标签与其位置对应起来
        labels_position = dict(zip(xticklabels, ax.get_xticks()))
        # 检查特定刻度标签是否位于正确的位置
        assert labels_position["(2012, 1)"] == 0.0
        assert labels_position["(2012, 2)"] == 1.0
        assert labels_position["(2013, 1)"] == 2.0
        assert labels_position["(2013, 2)"] == 3.0

    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_xlim_plot_line(self, kind):
        # 测试 plot.line 和 plot.area 中 xlim 的设置是否正确
        # GH 27686
        # 创建一个包含两个数据的 DataFrame，使用默认索引
        df = DataFrame([2, 4], index=[1, 2])
        # 在 DataFrame 上根据参数 kind 绘制相应类型的图形，并获取对应的 Axes 对象
        ax = df.plot(kind=kind)
        # 获取当前 x 轴的显示范围
        xlims = ax.get_xlim()
        # 检查 xlim 的设置是否符合预期
        assert xlims[0] < 1
        assert xlims[1] > 2

    def test_xlim_plot_line_correctly_in_mixed_plot_type(self):
        # 测试当 Axes 包含多种不同类型图形时，xlim 的设置是否正确
        # GH 27686
        # 创建一个 Figure 和一个包含多列数据的 DataFrame
        fig, ax = mpl.pyplot.subplots()

        indexes = ["k1", "k2", "k3", "k4"]
        df = DataFrame(
            {
                "s1": [1000, 2000, 1500, 2000],
                "s2": [900, 1400, 2000, 3000],
                "s3": [1500, 1500, 1600, 1200],
                "secondary_y": [1, 3, 4, 3],
            },
            index=indexes,
        )
        # 在 Axes 上绘制柱状图，不堆叠
        df[["s1", "s2", "s3"]].plot.bar(ax=ax, stacked=False)
        # 在 Axes 上绘制包含 secondary_y 列的线图，使用次坐标轴
        df[["secondary_y"]].plot(ax=ax, secondary_y=True)

        # 获取当前 x 轴的显示范围
        xlims = ax.get_xlim()
        # 检查 xlim 的设置是否符合预期
        assert xlims[0] < 0
        assert xlims[1] > 3

        # 确保坐标轴标签也被正确地绘制
        xticklabels = [t.get_text() for t in ax.get_xticklabels()]
        assert xticklabels == indexes

    def test_plot_no_rows(self):
        # GH 27758
        # 创建一个空的 DataFrame，只有列名为 "foo"
        df = DataFrame(columns=["foo"], dtype=int)
        # 检查 DataFrame 是否为空
        assert df.empty
        # 在空 DataFrame 上绘制图形，并获取对应的 Axes 对象
        ax = df.plot()
        # 检查绘制的图形是否只有一条线
        assert len(ax.get_lines()) == 1
        # 获取绘制的线条对象
        line = ax.get_lines()[0]
        # 检查线条对象的 x 数据长度是否为 0
        assert len(line.get_xdata()) == 0
        # 检查线条对象的 y 数据长度是否为 0
        assert len(line.get_ydata()) == 0

    def test_plot_no_numeric_data(self):
        # GH 27759
        # 创建一个包含字符串的 DataFrame
        df = DataFrame(["a", "b", "c"])
        # 使用 pytest 检查在绘图时是否会抛出 TypeError 异常，异常信息应包含指定文本
        with pytest.raises(TypeError, match="no numeric data to plot"):
            df.plot()

    @pytest.mark.parametrize(
        "kind", ("line", "bar", "barh", "hist", "kde", "density", "area", "pie")
    )
    # 定义一个测试方法，用于测试多子图功能
    def test_group_subplot(self, kind):
        # 如果没有安装 scipy 库，则跳过这个测试
        pytest.importorskip("scipy")
        # 创建一个包含多列数据的字典
        d = {
            "a": np.arange(10),
            "b": np.arange(10) + 1,
            "c": np.arange(10) + 1,
            "d": np.arange(10),
            "e": np.arange(10),
        }
        # 从字典创建一个 DataFrame
        df = DataFrame(d)

        # 调用 DataFrame 的 plot 方法创建子图，并传入指定的子图结构和图表类型
        axes = df.plot(subplots=[("b", "e"), ("c", "d")], kind=kind)
        # 断言子图的数量，应为 3（两个分组 + 单独的列 a）
        assert len(axes) == 3

        # 预期的每个子图对应的列标签
        expected_labels = (["b", "e"], ["c", "d"], ["a"])
        # 遍历子图和预期标签，进行进一步检查
        for ax, labels in zip(axes, expected_labels):
            # 如果图表类型不是饼图，则检查图例标签是否匹配预期
            if kind != "pie":
                _check_legend_labels(ax, labels=labels)
            # 如果图表类型是线图，则断言每个子图的线条数量与标签数量相同
            if kind == "line":
                assert len(ax.lines) == len(labels)

    # 定义一个测试方法，用于测试 Series 对象的多子图功能未实现情况
    def test_group_subplot_series_notimplemented(self):
        # 创建一个包含单个元素的 Series 对象
        ser = Series(range(1))
        # 期望的错误消息内容
        msg = "An iterable subplots for a Series"
        # 使用 pytest 检查是否抛出预期的 NotImplementedError 异常，并匹配特定消息
        with pytest.raises(NotImplementedError, match=msg):
            ser.plot(subplots=[("a",)])

    # 定义一个测试方法，用于测试具有多级索引的 DataFrame 对象的多子图功能未实现情况
    def test_group_subplot_multiindex_notimplemented(self):
        # 创建一个具有多级索引的 DataFrame 对象
        df = DataFrame(np.eye(2), columns=MultiIndex.from_tuples([(0, 1), (1, 2)]))
        # 期望的错误消息内容
        msg = "An iterable subplots for a DataFrame with a MultiIndex"
        # 使用 pytest 检查是否抛出预期的 NotImplementedError 异常，并匹配特定消息
        with pytest.raises(NotImplementedError, match=msg):
            df.plot(subplots=[(0, 1)])

    # 定义一个测试方法，用于测试具有非唯一列名的 DataFrame 对象的多子图功能未实现情况
    def test_group_subplot_nonunique_cols_notimplemented(self):
        # 创建一个具有非唯一列名的 DataFrame 对象
        df = DataFrame(np.eye(2), columns=["a", "a"])
        # 期望的错误消息内容
        msg = "An iterable subplots for a DataFrame with non-unique"
        # 使用 pytest 检查是否抛出预期的 NotImplementedError 异常，并匹配特定消息
        with pytest.raises(NotImplementedError, match=msg):
            df.plot(subplots=[("a",)])

    # 使用 pytest 的参数化装饰器，定义一个测试方法，用于测试不良输入时是否引发 ValueError 异常
    @pytest.mark.parametrize(
        "subplots, expected_msg",
        [
            # subplots 应为布尔值或可迭代对象
            (123, "subplots should be a bool or an iterable"),
            # 每个条目应为列表/元组（非可迭代对象的可迭代对象）
            ("a", "each entry should be a list/tuple"),
            # 每个条目应为列表/元组（可迭代对象的非可迭代对象）
            ((1,), "each entry should be a list/tuple"),
            # 每个条目应为列表/元组（可迭代对象的字符串）
            (("a",), "each entry should be a list/tuple"),
        ],
    )
    def test_group_subplot_bad_input(self, subplots, expected_msg):
        # 创建一个包含多列数据的字典
        d = {"a": np.arange(10), "b": np.arange(10)}
        # 从字典创建一个 DataFrame
        df = DataFrame(d)

        # 使用 pytest 检查是否抛出预期的 ValueError 异常，并匹配特定消息
        with pytest.raises(ValueError, match=expected_msg):
            df.plot(subplots=subplots)

    # 定义一个测试方法，用于测试传入无效列名时是否引发 ValueError 异常
    def test_group_subplot_invalid_column_name(self):
        # 创建一个包含多列数据的字典
        d = {"a": np.arange(10), "b": np.arange(10)}
        # 从字典创建一个 DataFrame
        df = DataFrame(d)

        # 使用 pytest 检查是否抛出包含特定消息的 ValueError 异常
        with pytest.raises(ValueError, match=r"Column label\(s\) \['bad_name'\]"):
            df.plot(subplots=[("a", "bad_name")])

    # 定义一个测试方法，用于测试传入重复列名时是否引发 ValueError 异常
    def test_group_subplot_duplicated_column(self):
        # 创建一个包含多列数据的字典
        d = {"a": np.arange(10), "b": np.arange(10), "c": np.arange(10)}
        # 从字典创建一个 DataFrame
        df = DataFrame(d)

        # 使用 pytest 检查是否抛出包含特定消息的 ValueError 异常
        with pytest.raises(ValueError, match="should be in only one subplot"):
            df.plot(subplots=[("a", "b"), ("a", "c")])
    # 使用 pytest.mark.parametrize 装饰器，为测试函数 test_group_subplot_invalid_kind 参数化设置多个参数组合
    @pytest.mark.parametrize("kind", ("box", "scatter", "hexbin"))
    def test_group_subplot_invalid_kind(self, kind):
        # 创建一个字典 d，包含两个键值对，每个值是一个包含 0 到 9 的 NumPy 数组
        d = {"a": np.arange(10), "b": np.arange(10)}
        # 使用字典 d 创建一个 pandas DataFrame 对象 df
        df = DataFrame(d)
        # 使用 pytest.raises 检测是否抛出 ValueError 异常，异常消息需匹配指定的正则表达式
        with pytest.raises(
            ValueError, match="When subplots is an iterable, kind must be one of"
        ):
            # 在 DataFrame 上调用 plot 方法，参数 subplots 是一个元组列表，kind 为传入的参数之一
            df.plot(subplots=[("a", "b")], kind=kind)

    # 使用 pytest.mark.parametrize 装饰器，为测试函数 test_xlabel_ylabel_dataframe_single_plot 参数化设置多个参数组合
    @pytest.mark.parametrize(
        "index_name, old_label, new_label",
        [
            (None, "", "new"),
            ("old", "old", "new"),
            (None, "", ""),
            (None, "", 1),
            (None, "", [1, 2]),
        ],
    )
    # 使用 pytest.mark.parametrize 装饰器，为测试函数 test_xlabel_ylabel_dataframe_single_plot 参数化设置 kind 参数
    @pytest.mark.parametrize("kind", ["line", "area", "bar"])
    def test_xlabel_ylabel_dataframe_single_plot(
        self, kind, index_name, old_label, new_label
    ):
        # 创建一个 2x2 的 DataFrame，包含数字数据，列名为 Type A 和 Type B
        df = DataFrame([[1, 2], [2, 5]], columns=["Type A", "Type B"])
        # 设置 DataFrame 的索引名为 index_name
        df.index.name = index_name

        # 在 DataFrame 上调用 plot 方法，kind 为传入的参数之一
        ax = df.plot(kind=kind)
        # 断言横轴标签是否等于 old_label
        assert ax.get_xlabel() == old_label
        # 断言纵轴标签是否为空字符串
        assert ax.get_ylabel() == ""

        # 在 DataFrame 上调用 plot 方法，kind 为传入的参数之一，设置新的横轴和纵轴标签
        ax = df.plot(kind=kind, ylabel=new_label, xlabel=new_label)
        # 断言纵轴标签是否等于 new_label
        assert ax.get_ylabel() == str(new_label)
        # 断言横轴标签是否等于 new_label
        assert ax.get_xlabel() == str(new_label)

    # 使用 pytest.mark.parametrize 装饰器，为测试函数 test_xlabel_ylabel_dataframe_plane_plot 参数化设置多个参数组合
    @pytest.mark.parametrize(
        "xlabel, ylabel",
        [
            (None, None),
            ("X Label", None),
            (None, "Y Label"),
            ("X Label", "Y Label"),
        ],
    )
    # 使用 pytest.mark.parametrize 装饰器，为测试函数 test_xlabel_ylabel_dataframe_plane_plot 参数化设置 kind 参数
    @pytest.mark.parametrize("kind", ["scatter", "hexbin"])
    def test_xlabel_ylabel_dataframe_plane_plot(self, kind, xlabel, ylabel):
        # 创建一个 2x2 的 DataFrame，包含数字数据，列名为 Type A 和 Type B
        xcol = "Type A"
        ycol = "Type B"
        df = DataFrame([[1, 2], [2, 5]], columns=[xcol, ycol])

        # 在 DataFrame 上调用 plot 方法，kind 为传入的参数之一，设置横轴和纵轴标签
        ax = df.plot(kind=kind, x=xcol, y=ycol, xlabel=xlabel, ylabel=ylabel)
        # 断言横轴标签是否等于 xcol 或者 xlabel（如果 xlabel 为 None 则为 xcol）
        assert ax.get_xlabel() == (xcol if xlabel is None else xlabel)
        # 断言纵轴标签是否等于 ycol 或者 ylabel（如果 ylabel 为 None 则为 ycol）
        assert ax.get_ylabel() == (ycol if ylabel is None else ylabel)

    # 使用 pytest.mark.parametrize 装饰器，为测试函数 test_secondary_y 参数化设置 secondary_y 参数
    @pytest.mark.parametrize("secondary_y", (False, True))
    def test_secondary_y(self, secondary_y):
        # 创建一个包含单个元素 0 的 DataFrame，绘制一个图表，设置次要纵轴，指定标签、纵轴范围和刻度
        ax_df = DataFrame([0]).plot(
            secondary_y=secondary_y, ylabel="Y", ylim=(0, 100), yticks=[99]
        )
        # 遍历图表的所有轴
        for ax in ax_df.figure.axes:
            # 如果当前轴的纵轴可见
            if ax.yaxis.get_visible():
                # 断言纵轴标签是否为 "Y"
                assert ax.get_ylabel() == "Y"
                # 断言纵轴范围是否为 (0, 100)
                assert ax.get_ylim() == (0, 100)
                # 断言纵轴刻度的第一个值是否为 99

                assert ax.get_yticks()[0] == 99

    # 使用 pytest.mark.slow 标记当前测试模块为慢速测试
    def test_plot_no_warning(self):
        # 标识：GH 55138
        # TODO(3.0): 一旦 Period[B] 弃用被执行，可以移除此处代码
        # 创建一个 DataFrame 对象，填充随机生成的标准正态分布数据
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # 禁止产生警告
        with tm.assert_produces_warning(False):
            # 绘制 DataFrame 的图表并忽略返回值
            _ = df.plot()
            # 绘制 DataFrame 转置后的图表并忽略返回值
            _ = df.T.plot()

    @pytest.mark.parametrize("freq", ["h", "7h", "60min", "120min", "3M"])
    def test_plot_period_index_makes_no_right_shift(self, freq):
        # GH#57587
        # 创建一个 PeriodIndex 对象，指定频率 freq 和周期为 4
        idx = pd.period_range("01/01/2000", freq=freq, periods=4)
        # 创建一个 DataFrame 对象，填充指定数据，使用创建的 PeriodIndex 作为索引
        df = DataFrame(
            np.array([0, 1, 0, 1]),
            index=idx,
            columns=["A"],
        )
        # 预期值为 PeriodIndex 对象的值数组
        expected = idx.values

        # 绘制 DataFrame 的图表并获取坐标轴对象
        ax = df.plot()
        # 获取图表中第一条线的 x 轴数据
        result = ax.get_lines()[0].get_xdata()
        # 断言：验证 result 的每个元素是否与 expected 相等
        assert all(str(result[i]) == str(expected[i]) for i in range(4))
# 定义一个函数用于生成一个包含四个子图的网格布局
def _generate_4_axes_via_gridspec():
    # 使用 Matplotlib 的 gridspec 模块创建一个2x2的网格布局对象
    gs = mpl.gridspec.GridSpec(2, 2)
    # 在网格布局中创建左上角的子图对象，并赋给 ax_tl
    ax_tl = plt.subplot(gs[0, 0])
    # 在网格布局中创建左下角的子图对象，并赋给 ax_ll
    ax_ll = plt.subplot(gs[1, 0])
    # 在网格布局中创建右上角的子图对象，并赋给 ax_tr
    ax_tr = plt.subplot(gs[0, 1])
    # 在网格布局中创建右下角的子图对象，并赋给 ax_lr
    ax_lr = plt.subplot(gs[1, 1])

    # 返回网格布局对象和包含四个子图对象的列表
    return gs, [ax_tl, ax_ll, ax_tr, ax_lr]
```