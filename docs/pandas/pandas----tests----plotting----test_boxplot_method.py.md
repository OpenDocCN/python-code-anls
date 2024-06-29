# `D:\src\scipysrc\pandas\pandas\tests\plotting\test_boxplot_method.py`

```
# 导入必要的库和模块
import itertools  # 提供迭代工具函数
import string  # 提供字符串相关的常量和函数

import numpy as np  # 引入NumPy库并命名为np
import pytest  # 引入pytest测试框架

from pandas import (  # 从pandas库中导入多个模块和类
    DataFrame,  # 数据结构：DataFrame
    MultiIndex,  # 数据结构：MultiIndex
    Series,  # 数据结构：Series
    date_range,  # 时间序列相关：date_range
    plotting,  # 绘图相关：plotting
    timedelta_range,  # 时间增量相关：timedelta_range
)
import pandas._testing as tm  # 导入pandas测试模块作为tm别名
from pandas.tests.plotting.common import (  # 从pandas测试模块中导入绘图通用函数
    _check_axes_shape,  # 检查图形轴的形状
    _check_box_return_type,  # 检查盒形图返回类型
    _check_plot_works,  # 检查绘图函数是否正常工作
    _check_ticks_props,  # 检查刻度属性
    _check_visible,  # 检查可见性
)

from pandas.io.formats.printing import pprint_thing  # 从pandas IO模块导入打印函数

mpl = pytest.importorskip("matplotlib")  # 导入matplotlib并确保存在
plt = pytest.importorskip("matplotlib.pyplot")  # 导入matplotlib.pyplot并确保存在

def _check_ax_limits(col, ax):
    # 检查轴限制函数，确保数据列col在轴ax的限制范围内
    y_min, y_max = ax.get_ylim()
    assert y_min <= col.min()  # 断言最小值大于等于数据列的最小值
    assert y_max >= col.max()  # 断言最大值小于等于数据列的最大值

class TestDataFramePlots:
    def test_stacked_boxplot_set_axis(self):
        # 测试堆叠箱线图功能，设置轴
        # 创建包含随机数据的DataFrame对象df
        n = 30
        df = DataFrame(
            {
                "Clinical": np.random.default_rng(2).choice([0, 1, 2, 3], n),
                "Confirmed": np.random.default_rng(2).choice([0, 1, 2, 3], n),
                "Discarded": np.random.default_rng(2).choice([0, 1, 2, 3], n),
            },
            index=np.arange(0, n),
        )
        ax = df.plot(kind="bar", stacked=True)  # 绘制堆叠柱状图并返回轴对象ax
        assert [int(x.get_text()) for x in ax.get_xticklabels()] == df.index.to_list()  # 断言x轴刻度标签的文本与DataFrame的索引匹配
        ax.set_xticks(np.arange(0, n, 10))  # 设置x轴刻度
        plt.draw()  # 更新图形的更改
        assert [int(x.get_text()) for x in ax.get_xticklabels()] == list(
            np.arange(0, n, 10)
        )  # 断言更新后的x轴刻度标签文本正确

    @pytest.mark.slow  # 标记为慢速测试
    @pytest.mark.parametrize(  # 使用参数化装饰器标记测试参数
        "kwargs, warn",
        [
            [{"return_type": "dict"}, None],  # 参数为return_type为dict时，不产生警告
            [{"column": ["one", "two"]}, None],  # 参数为column为['one', 'two']时，不产生警告
            [{"column": ["one", "two"], "by": "indic"}, UserWarning],  # 参数不符合预期，会产生UserWarning警告
            [{"column": ["one"], "by": ["indic", "indic2"]}, None],  # 参数为column为['one']，不产生警告
            [{"by": "indic"}, UserWarning],  # 参数不符合预期，会产生UserWarning警告
            [{"by": ["indic", "indic2"]}, UserWarning],  # 参数不符合预期，会产生UserWarning警告
            [{"notch": 1}, None],  # 参数为notch为1，不产生警告
            [{"by": "indic", "notch": 1}, UserWarning],  # 参数不符合预期，会产生UserWarning警告
        ],
    )
    def test_boxplot_legacy1(self, kwargs, warn):
        # 测试箱线图的兼容性1
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),  # 生成随机正态分布数据的DataFrame对象df
            index=list(string.ascii_letters[:6]),  # 使用字母生成索引
            columns=["one", "two", "three", "four"],  # 指定列名
        )
        df["indic"] = ["foo", "bar"] * 3  # 添加名为indic的列，交替包含'foo'和'bar'
        df["indic2"] = ["foo", "bar", "foo"] * 2  # 添加名为indic2的列，交替包含'foo'和'bar'，长度为6

        # 检查绘图函数是否正常工作，并捕获可能的警告
        with tm.assert_produces_warning(warn, check_stacklevel=False):
            _check_plot_works(df.boxplot, **kwargs)  # 调用_check_plot_works函数测试箱线图绘制功能

    def test_boxplot_legacy1_series(self):
        # 测试箱线图的兼容性1（针对Series）
        ser = Series(np.random.default_rng(2).standard_normal(6))  # 生成随机正态分布数据的Series对象ser
        _check_plot_works(plotting._core.boxplot, data=ser, return_type="dict")  # 调用_check_plot_works函数测试箱线图绘制功能（针对Series）
    def test_boxplot_legacy2(self):
        # 创建一个包含随机数据的 DataFrame，有两列：Col1 和 Col2
        df = DataFrame(
            np.random.default_rng(2).random((10, 2)), columns=["Col1", "Col2"]
        )
        # 在 DataFrame 中添加列 "X"，并赋予值 ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]
        df["X"] = Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        # 在 DataFrame 中添加列 "Y"，并赋予值 ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A"]
        df["Y"] = Series(["A"] * 10)
        # 使用 assert_produces_warning 确保调用 _check_plot_works(df.boxplot, by="X") 时产生 UserWarning 警告
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            _check_plot_works(df.boxplot, by="X")

    def test_boxplot_legacy2_with_ax(self):
        # 创建一个包含随机数据的 DataFrame，有两列：Col1 和 Col2
        df = DataFrame(
            np.random.default_rng(2).random((10, 2)), columns=["Col1", "Col2"]
        )
        # 在 DataFrame 中添加列 "X"，并赋予值 ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]
        df["X"] = Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        # 在 DataFrame 中添加列 "Y"，并赋予值 ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A"]
        df["Y"] = Series(["A"] * 10)
        # 当提供了 ax 参数并且需要一个轴时，应该使用传入的 ax 对象：
        _, ax = mpl.pyplot.subplots()
        # 调用 df.boxplot("Col1", by="X", ax=ax) 并将结果赋给 axes
        axes = df.boxplot("Col1", by="X", ax=ax)
        # 获取 ax 的 axes 对象，并将其赋给 ax_axes
        ax_axes = ax.axes
        # 断言 ax_axes 和 axes 是同一个对象
        assert ax_axes is axes

    def test_boxplot_legacy2_with_ax_return_type(self):
        # 创建一个包含随机数据的 DataFrame，有两列：Col1 和 Col2
        df = DataFrame(
            np.random.default_rng(2).random((10, 2)), columns=["Col1", "Col2"]
        )
        # 在 DataFrame 中添加列 "X"，并赋予值 ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]
        df["X"] = Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        # 在 DataFrame 中添加列 "Y"，并赋予值 ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A"]
        df["Y"] = Series(["A"] * 10)
        # 创建一个新的图形对象 fig 和轴对象 ax
        fig, ax = mpl.pyplot.subplots()
        # 调用 df.groupby("Y").boxplot(ax=ax, return_type="axes") 并将结果赋给 axes
        axes = df.groupby("Y").boxplot(ax=ax, return_type="axes")
        # 获取 ax 的 axes 对象，并将其赋给 ax_axes
        ax_axes = ax.axes
        # 断言 ax_axes 和 axes["A"] 是同一个对象
        assert ax_axes is axes["A"]

    def test_boxplot_legacy2_with_multi_col(self):
        # 创建一个包含随机数据的 DataFrame，有两列：Col1 和 Col2
        df = DataFrame(
            np.random.default_rng(2).random((10, 2)), columns=["Col1", "Col2"]
        )
        # 在 DataFrame 中添加列 "X"，并赋予值 ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]
        df["X"] = Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        # 在 DataFrame 中添加列 "Y"，并赋予值 ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A"]
        df["Y"] = Series(["A"] * 10)
        # 当使用多列和 ax 参数时，应该使用相同的图形
        fig, ax = mpl.pyplot.subplots()
        # 使用 df.boxplot(column=["Col1", "Col2"], by="X", ax=ax, return_type="axes") 进行绘图，并将结果赋给 axes
        msg = "the figure containing the passed axes is being cleared"
        with tm.assert_produces_warning(UserWarning, match=msg):
            axes = df.boxplot(column=["Col1", "Col2"], by="X", ax=ax, return_type="axes")
        # 断言 axes["Col1"].get_figure() 和 fig 是同一个图形对象
        assert axes["Col1"].get_figure() is fig

    def test_boxplot_legacy2_by_none(self):
        # 创建一个包含随机数据的 DataFrame，有两列：Col1 和 Col2
        df = DataFrame(
            np.random.default_rng(2).random((10, 2)), columns=["Col1", "Col2"]
        )
        # 在 DataFrame 中添加列 "X"，并赋予值 ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]
        df["X"] = Series(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        # 在 DataFrame 中添加列 "Y"，并赋予值 ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A"]
        df["Y"] = Series(["A"] * 10)
        # 当 by 参数为 None 时，检查在返回的字典中是否包含所有相关线条
        _, ax = mpl.pyplot.subplots()
        # 调用 df.boxplot(ax=ax, return_type="dict") 并将结果赋给 d
        d = df.boxplot(ax=ax, return_type="dict")
        # 将所有的线条放在一个列表中
        lines = list(itertools.chain.from_iterable(d.values()))
        # 断言 ax.get_lines() 的数量和 lines 的长度相同
        assert len(ax.get_lines()) == len(lines)

    def test_boxplot_return_type_none(self, hist_df):
        # GH 12216; return_type=None & by=None -> axes
        # 调用 hist_df.boxplot() 并将结果赋给 result
        result = hist_df.boxplot()
        # 断言 result 的类型是 mpl.pyplot.Axes 的实例
        assert isinstance(result, mpl.pyplot.Axes)
    def test_boxplot_return_type_legacy(self):
        # 引入的 API 更改在 https://github.com/pandas-dev/pandas/pull/7096
        
        # 创建一个 DataFrame，包含随机生成的标准正态分布数据，6行4列
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        # 设置错误消息
        msg = "return_type must be {'axes', 'dict', 'both'}"
        # 使用 pytest 检查是否引发 ValueError，并匹配指定消息
        with pytest.raises(ValueError, match=msg):
            df.boxplot(return_type="NOT_A_TYPE")

        # 执行 DataFrame 的 boxplot 方法，返回结果
        result = df.boxplot()
        # 调用内部函数 _check_box_return_type，检查返回类型是否为 "axes"
        _check_box_return_type(result, "axes")

    @pytest.mark.parametrize("return_type", ["dict", "axes", "both"])
    def test_boxplot_return_type_legacy_return_type(self, return_type):
        # 引入的 API 更改在 https://github.com/pandas-dev/pandas/pull/7096
        
        # 创建一个 DataFrame，包含随机生成的标准正态分布数据，6行4列
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["one", "two", "three", "four"],
        )
        # 禁止产生警告的上下文
        with tm.assert_produces_warning(False):
            # 执行 DataFrame 的 boxplot 方法，返回结果
            result = df.boxplot(return_type=return_type)
        # 调用内部函数 _check_box_return_type，检查返回类型是否符合参数设定的 return_type
        _check_box_return_type(result, return_type)

    def test_boxplot_axis_limits(self, hist_df):
        # 复制历史数据 DataFrame
        df = hist_df.copy()
        # 为 DataFrame 添加一个 "age" 列，包含随机整数数据
        df["age"] = np.random.default_rng(2).integers(1, 20, df.shape[0])
        # 执行 boxplot 方法，绘制 "height" 和 "weight" 的箱线图，按 "category" 分组
        height_ax, weight_ax = df.boxplot(["height", "weight"], by="category")
        # 检查 "height" 和 "weight" 的轴限制
        _check_ax_limits(df["height"], height_ax)
        _check_ax_limits(df["weight"], weight_ax)
        # 断言 weight_ax 的 y 轴与 height_ax 的 y 轴共享
        assert weight_ax._sharey == height_ax

    def test_boxplot_axis_limits_two_rows(self, hist_df):
        # 复制历史数据 DataFrame
        df = hist_df.copy()
        # 为 DataFrame 添加一个 "age" 列，包含随机整数数据
        df["age"] = np.random.default_rng(2).integers(1, 20, df.shape[0])
        # 执行 boxplot 方法，绘制 "height"、"weight" 和 "age" 的箱线图，按 "category" 分组
        p = df.boxplot(["height", "weight", "age"], by="category")
        # 获取返回的子图对象
        height_ax, weight_ax, age_ax = p[0, 0], p[0, 1], p[1, 0]
        dummy_ax = p[1, 1]

        # 检查 "height"、"weight" 和 "age" 的轴限制
        _check_ax_limits(df["height"], height_ax)
        _check_ax_limits(df["weight"], weight_ax)
        _check_ax_limits(df["age"], age_ax)
        # 断言 weight_ax 和 height_ax 的 y 轴共享
        assert weight_ax._sharey == height_ax
        # 断言 age_ax 的 y 轴未共享
        assert age_ax._sharey == height_ax
        # 断言 dummy_ax 的 y 轴为 None
        assert dummy_ax._sharey is None

    def test_boxplot_empty_column(self):
        # 创建一个 DataFrame，包含随机生成的标准正态分布数据，20行4列
        df = DataFrame(np.random.default_rng(2).standard_normal((20, 4)))
        # 将第一列设置为 NaN
        df.loc[:, 0] = np.nan
        # 调用内部函数 _check_plot_works，检查 boxplot 方法执行正常，返回类型为 "axes"
        _check_plot_works(df.boxplot, return_type="axes")

    def test_figsize(self):
        # 创建一个 DataFrame，包含随机生成的随机数数据，10行5列
        df = DataFrame(
            np.random.default_rng(2).random((10, 5)), columns=["A", "B", "C", "D", "E"]
        )
        # 执行 boxplot 方法，绘制箱线图，设置图形大小为 (12, 8)
        result = df.boxplot(return_type="axes", figsize=(12, 8))
        # 断言结果的图形宽度为 12
        assert result.figure.bbox_inches.width == 12
        # 断言结果的图形高度为 8
        assert result.figure.bbox_inches.height == 8

    def test_fontsize(self):
        # 创建一个 DataFrame，包含列 "a"，数据为 [1, 2, 3, 4, 5, 6]
        df = DataFrame({"a": [1, 2, 3, 4, 5, 6]})
        # 执行 boxplot 方法，绘制 "a" 列的箱线图，设置 x 和 y 轴标签字体大小为 16
        _check_ticks_props(df.boxplot("a", fontsize=16), xlabelsize=16, ylabelsize=16)
    # 定义一个测试方法，用于验证箱线图绘制数值数据的功能
    def test_boxplot_numeric_data(self):
        # GH 22799: GitHub issue编号22799，用于跟踪和管理相关问题
        # 创建一个DataFrame对象，包含多列数据：
        # - "a": 日期范围从"2012-01-01"开始的10个日期
        # - "b": 从正态分布中生成的10个标准正态随机数
        # - "c": 从另一个正态分布中生成的10个随机数，并在每个数上加2
        # - "d": 日期范围从"2012-01-01"开始的10个日期，并转换为字符串类型
        # - "e": 日期范围从"2012-01-01"开始的10个日期，带有UTC时区信息
        # - "f": 时间增量范围为1天的10个时间增量
        df = DataFrame(
            {
                "a": date_range("2012-01-01", periods=10),
                "b": np.random.default_rng(2).standard_normal(10),
                "c": np.random.default_rng(2).standard_normal(10) + 2,
                "d": date_range("2012-01-01", periods=10).astype(str),
                "e": date_range("2012-01-01", periods=10, tz="UTC"),
                "f": timedelta_range("1 days", periods=10),
            }
        )
        # 使用DataFrame的plot方法绘制箱线图，返回绘图对象ax
        ax = df.plot(kind="box")
        # 断言箱线图的x轴刻度标签文本内容为["b", "c"]
        assert [x.get_text() for x in ax.get_xticklabels()] == ["b", "c"]

    @pytest.mark.parametrize(
        "colors_kwd, expected",
        [
            (
                {"boxes": "r", "whiskers": "b", "medians": "g", "caps": "c"},
                {"boxes": "r", "whiskers": "b", "medians": "g", "caps": "c"},
            ),
            ({"boxes": "r"}, {"boxes": "r"}),
            ("r", {"boxes": "r", "whiskers": "r", "medians": "r", "caps": "r"}),
        ],
    )
    # 定义一个参数化测试方法，用于验证颜色关键字在箱线图中的使用
    def test_color_kwd(self, colors_kwd, expected):
        # GH: 26214: GitHub issue编号26214，用于跟踪和管理相关问题
        # 创建一个包含随机数据的DataFrame对象
        df = DataFrame(np.random.default_rng(2).random((10, 2)))
        # 使用DataFrame的boxplot方法绘制箱线图，并设置颜色为colors_kwd，返回结果字典
        result = df.boxplot(color=colors_kwd, return_type="dict")
        # 遍历期望的结果字典expected，断言每个元素的颜色与result中的颜色一致
        for k, v in expected.items():
            assert result[k][0].get_color() == v

    @pytest.mark.parametrize(
        "scheme,expected",
        [
            (
                "dark_background",
                {
                    "boxes": "#8dd3c7",
                    "whiskers": "#8dd3c7",
                    "medians": "#bfbbd9",
                    "caps": "#8dd3c7",
                },
            ),
            (
                "default",
                {
                    "boxes": "#1f77b4",
                    "whiskers": "#1f77b4",
                    "medians": "#2ca02c",
                    "caps": "#1f77b4",
                },
            ),
        ],
    )
    # 定义一个参数化测试方法，用于验证不同主题下箱线图的颜色设置
    def test_colors_in_theme(self, scheme, expected):
        # GH: 40769: GitHub issue编号40769，用于跟踪和管理相关问题
        # 创建一个包含随机数据的DataFrame对象
        df = DataFrame(np.random.default_rng(2).random((10, 2)))
        # 使用plt.style.use方法设置绘图风格为scheme
        plt.style.use(scheme)
        # 使用DataFrame的plot.box方法绘制箱线图，返回结果字典
        result = df.plot.box(return_type="dict")
        # 遍历期望的结果字典expected，断言每个元素的颜色与result中的颜色一致
        for k, v in expected.items():
            assert result[k][0].get_color() == v

    @pytest.mark.parametrize(
        "dict_colors, msg",
        [({"boxes": "r", "invalid_key": "r"}, "invalid key 'invalid_key'")],
    )
    # 定义一个参数化测试方法，用于验证箱线图颜色关键字中的错误情况
    def test_color_kwd_errors(self, dict_colors, msg):
        # GH: 26214: GitHub issue编号26214，用于跟踪和管理相关问题
        # 创建一个包含随机数据的DataFrame对象
        df = DataFrame(np.random.default_rng(2).random((10, 2)))
        # 使用pytest.raises断言捕获预期的ValueError异常，并检查异常消息是否匹配msg
        with pytest.raises(ValueError, match=msg):
            # 调用DataFrame的boxplot方法，使用不正确的颜色字典dict_colors，返回结果字典
            df.boxplot(color=dict_colors, return_type="dict")

    @pytest.mark.parametrize(
        "props, expected",
        [
            ("boxprops", "boxes"),
            ("whiskerprops", "whiskers"),
            ("capprops", "caps"),
            ("medianprops", "medians"),
        ],
    )
    def test_specified_props_kwd(self, props, expected):
        # 创建一个 DataFrame 对象，包含三列数据，每列使用随机生成的浮点数
        df = DataFrame({k: np.random.default_rng(2).random(10) for k in "ABC"})
        # 构建一个关键字参数字典，将 props 参数作为键，值为包含颜色设置的字典
        kwd = {props: {"color": "C1"}}
        # 使用 boxplot 方法绘制箱线图，并传入关键字参数 kwd
        result = df.boxplot(return_type="dict", **kwd)

        # 断言箱线图的颜色属性与预期相符
        assert result[expected][0].get_color() == "C1"

    @pytest.mark.parametrize("vert", [True, False])
    def test_plot_xlabel_ylabel(self, vert):
        # 创建一个 DataFrame 对象，包含三列数据，其中两列使用标准正态分布生成，一列使用字符串
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
                "group": np.random.default_rng(2).choice(["group1", "group2"], 10),
            }
        )
        # 设置 x 轴和 y 轴的标签文本
        xlabel, ylabel = "x", "y"
        # 使用 plot 方法绘制指定类型的图表，并设置参数 vert、xlabel、ylabel
        ax = df.plot(kind="box", vert=vert, xlabel=xlabel, ylabel=ylabel)
        # 断言获取的 x 轴和 y 轴标签文本与预期相符
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel

    @pytest.mark.parametrize("vert", [True, False])
    def test_plot_box(self, vert):
        # GH 54941
        # 使用随机数生成器创建 DataFrame 对象 df1 和 df2，每个对象包含四列数据
        rng = np.random.default_rng(2)
        df1 = DataFrame(rng.integers(0, 100, size=(10, 4)), columns=list("ABCD"))
        df2 = DataFrame(rng.integers(0, 100, size=(10, 4)), columns=list("ABCD"))

        # 设置 x 轴和 y 轴的标签文本
        xlabel, ylabel = "x", "y"
        # 创建包含两个子图的图形窗口，并设置子图共享 y 轴
        _, axs = plt.subplots(ncols=2, figsize=(10, 7), sharey=True)
        # 分别在两个子图上绘制箱线图，设置参数 vert、xlabel、ylabel
        df1.plot.box(ax=axs[0], vert=vert, xlabel=xlabel, ylabel=ylabel)
        df2.plot.box(ax=axs[1], vert=vert, xlabel=xlabel, ylabel=ylabel)
        # 遍历每个子图，断言获取的 x 轴和 y 轴标签文本与预期相符
        for ax in axs:
            assert ax.get_xlabel() == xlabel
            assert ax.get_ylabel() == ylabel

    @pytest.mark.parametrize("vert", [True, False])
    def test_boxplot_xlabel_ylabel(self, vert):
        # 创建一个 DataFrame 对象，包含三列数据，其中两列使用标准正态分布生成，一列使用字符串
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
                "group": np.random.default_rng(2).choice(["group1", "group2"], 10),
            }
        )
        # 设置 x 轴和 y 轴的标签文本
        xlabel, ylabel = "x", "y"
        # 使用 boxplot 方法绘制箱线图，并设置参数 vert、xlabel、ylabel
        ax = df.boxplot(vert=vert, xlabel=xlabel, ylabel=ylabel)
        # 断言获取的 x 轴和 y 轴标签文本与预期相符
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel

    @pytest.mark.parametrize("vert", [True, False])
    def test_boxplot_group_xlabel_ylabel(self, vert):
        # 创建一个 DataFrame 对象，包含三列数据，其中两列使用标准正态分布生成，一列使用字符串
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
                "group": np.random.default_rng(2).choice(["group1", "group2"], 10),
            }
        )
        # 设置 x 轴和 y 轴的标签文本
        xlabel, ylabel = "x", "y"
        # 使用 boxplot 方法按组绘制箱线图，并设置参数 by、vert、xlabel、ylabel
        ax = df.boxplot(by="group", vert=vert, xlabel=xlabel, ylabel=ylabel)
        # 遍历每个子图，断言获取的 x 轴和 y 轴标签文本与预期相符
        for subplot in ax:
            assert subplot.get_xlabel() == xlabel
            assert subplot.get_ylabel() == ylabel
    # 定义一个测试方法，用于测试不带 x 轴和 y 轴标签的箱线图
    def test_boxplot_group_no_xlabel_ylabel(self, vert):
        # 创建一个 DataFrame 包含三列数据：a、b 是从标准正态分布中随机生成的数据，group 列在 "group1" 和 "group2" 中随机选择
        df = DataFrame(
            {
                "a": np.random.default_rng(2).standard_normal(10),
                "b": np.random.default_rng(2).standard_normal(10),
                "group": np.random.default_rng(2).choice(["group1", "group2"], 10),
            }
        )
        # 绘制按照 "group" 列分组的箱线图，vert 参数指定箱线图的方向（垂直或水平）
        ax = df.boxplot(by="group", vert=vert)
        # 遍历箱线图的子图
        for subplot in ax:
            # 如果箱线图是垂直方向的，获取 x 轴标签；否则获取 y 轴标签
            target_label = subplot.get_xlabel() if vert else subplot.get_ylabel()
            # 断言获取的标签内容与预期结果相同
            assert target_label == pprint_thing(["group"])
class TestDataFrameGroupByPlots:
    # 测试使用箱线图绘制数据分组的方法（遗留方法1）
    def test_boxplot_legacy1(self, hist_df):
        # 按照 "gender" 列对数据框进行分组
        grouped = hist_df.groupby(by="gender")
        # 断言产生 UserWarning 警告，不检查堆栈级别
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            # 调用 _check_plot_works 函数，期望返回 axes 对象
            axes = _check_plot_works(grouped.boxplot, return_type="axes")
        # 检查 axes 对象的形状是否符合预期
        _check_axes_shape(list(axes.values), axes_num=2, layout=(1, 2))

    # 测试使用箱线图绘制数据分组的方法，返回类型设置为 axes（遗留方法1）
    def test_boxplot_legacy1_return_type(self, hist_df):
        # 按照 "gender" 列对数据框进行分组
        grouped = hist_df.groupby(by="gender")
        # 调用 _check_plot_works 函数，期望返回单个 axes 对象
        axes = _check_plot_works(grouped.boxplot, subplots=False, return_type="axes")
        # 检查返回的 axes 对象是否符合预期形状
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    # 标记为慢速测试
    @pytest.mark.slow
    # 测试使用箱线图绘制数据分组的方法（遗留方法2）
    def test_boxplot_legacy2(self):
        # 创建一个包含随机数据的 DataFrame
        tuples = zip(string.ascii_letters[:10], range(10))
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=MultiIndex.from_tuples(tuples),
        )
        # 按照第二级别索引对数据框进行分组
        grouped = df.groupby(level=1)
        # 断言产生 UserWarning 警告，不检查堆栈级别
        with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
            # 调用 _check_plot_works 函数，期望返回 axes 对象
            axes = _check_plot_works(grouped.boxplot, return_type="axes")
        # 检查 axes 对象的形状是否符合预期
        _check_axes_shape(list(axes.values), axes_num=10, layout=(4, 3))

    # 标记为慢速测试
    @pytest.mark.slow
    # 测试使用箱线图绘制数据分组的方法，返回类型设置为 axes（遗留方法2）
    def test_boxplot_legacy2_return_type(self):
        # 创建一个包含随机数据的 DataFrame
        tuples = zip(string.ascii_letters[:10], range(10))
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=MultiIndex.from_tuples(tuples),
        )
        # 按照第二级别索引对数据框进行分组
        grouped = df.groupby(level=1)
        # 调用 _check_plot_works 函数，期望返回单个 axes 对象
        axes = _check_plot_works(grouped.boxplot, subplots=False, return_type="axes")
        # 检查返回的 axes 对象是否符合预期形状
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    # 测试分组数据框的绘图方法（检查图形数量）
    def test_grouped_plot_fignums(self):
        n = 10
        # 创建包含随机数据的 Series 对象
        weight = Series(np.random.default_rng(2).normal(166, 20, size=n))
        height = Series(np.random.default_rng(2).normal(60, 10, size=n))
        gender = np.random.default_rng(2).choice(["male", "female"], size=n)
        # 创建一个包含身高、体重和性别的 DataFrame
        df = DataFrame({"height": height, "weight": weight, "gender": gender})
        # 按照 "gender" 列对数据框进行分组
        gb = df.groupby("gender")

        # 调用 plot 方法绘制分组后的图形
        res = gb.plot()
        # 断言绘制的图形数量是否为 2
        assert len(mpl.pyplot.get_fignums()) == 2
        # 断言返回的对象数量是否为 2
        assert len(res) == 2
        # 关闭所有绘图
        plt.close("all")

        # 调用 boxplot 方法绘制分组后的箱线图
        res = gb.boxplot(return_type="axes")
        # 断言绘制的图形数量是否为 1
        assert len(mpl.pyplot.get_fignums()) == 1
        # 断言返回的对象数量是否为 2
        assert len(res) == 2

    # 测试分组数据框的绘图方法（排除列后的图形数量检查）
    def test_grouped_plot_fignums_excluded_col(self):
        n = 10
        # 创建包含随机数据的 Series 对象
        weight = Series(np.random.default_rng(2).normal(166, 20, size=n))
        height = Series(np.random.default_rng(2).normal(60, 10, size=n))
        gender = np.random.default_rng(2).choice(["male", "female"], size=n)
        # 创建一个包含身高、体重和性别的 DataFrame
        df = DataFrame({"height": height, "weight": weight, "gender": gender})
        # 由于性别列被排除，现在可以使用 GH 5610
        # 按照 "gender" 列对数据框进行分组并绘制直方图
        df.groupby("gender").hist()

    # 标记为慢速测试
    @pytest.mark.slow
    # 定义测试方法，验证箱线图返回类型是否正确
    def test_grouped_box_return_type(self, hist_df):
        # 将参数 hist_df 赋值给 df
        df = hist_df

        # 使用旧式方法绘制箱线图，返回类型为 None
        result = df.boxplot(by="gender")
        # 断言返回结果类型为 numpy 数组
        assert isinstance(result, np.ndarray)
        # 调用函数验证返回类型是否符合预期
        _check_box_return_type(
            result, None, expected_keys=["height", "weight", "category"]
        )

    @pytest.mark.slow
    # 定义测试方法，验证分组后箱线图返回类型是否为字典
    def test_grouped_box_return_type_groupby(self, hist_df):
        # 将参数 hist_df 赋值给 df
        df = hist_df
        # 使用 groupby 方法绘制分组箱线图
        result = df.groupby("gender").boxplot(return_type="dict")
        # 调用函数验证返回类型是否符合预期
        _check_box_return_type(result, "dict", expected_keys=["Male", "Female"])

    @pytest.mark.slow
    @pytest.mark.parametrize("return_type", ["dict", "axes", "both"])
    # 定义测试方法，验证根据参数 return_type 的不同返回不同类型的箱线图
    def test_grouped_box_return_type_arg(self, hist_df, return_type):
        # 将参数 hist_df 赋值给 df
        df = hist_df

        # 根据 classroom 列进行分组，绘制箱线图
        returned = df.groupby("classroom").boxplot(return_type=return_type)
        # 调用函数验证返回类型是否符合预期
        _check_box_return_type(returned, return_type, expected_keys=["A", "B", "C"])

        # 根据 classroom 列进行分组，绘制箱线图
        returned = df.boxplot(by="classroom", return_type=return_type)
        # 调用函数验证返回类型是否符合预期
        _check_box_return_type(
            returned, return_type, expected_keys=["height", "weight", "category"]
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("return_type", ["dict", "axes", "both"])
    # 定义测试方法，验证包含重复类别的数据集返回不同类型的箱线图
    def test_grouped_box_return_type_arg_duplcate_cats(self, return_type):
        # 定义列名
        columns2 = "X B C D A".split()
        # 创建包含随机数据的 DataFrame
        df2 = DataFrame(
            np.random.default_rng(2).standard_normal((6, 5)), columns=columns2
        )
        # 定义类别
        categories2 = "A B".split()
        # 添加类别列，每个类别重复三次
        df2["category"] = categories2 * 3

        # 根据 category 列进行分组，绘制箱线图
        returned = df2.groupby("category").boxplot(return_type=return_type)
        # 调用函数验证返回类型是否符合预期
        _check_box_return_type(returned, return_type, expected_keys=categories2)

        # 根据 category 列进行分组，绘制箱线图
        returned = df2.boxplot(by="category", return_type=return_type)
        # 调用函数验证返回类型是否符合预期
        _check_box_return_type(returned, return_type, expected_keys=columns2)

    @pytest.mark.slow
    # 定义测试方法，验证布局为 1x1 时抛出异常
    def test_grouped_box_layout_too_small(self, hist_df):
        # 将参数 hist_df 赋值给 df
        df = hist_df

        # 验证当布局为 1x1 时，抛出指定异常消息的 ValueError 异常
        msg = "Layout of 1x1 must be larger than required size 2"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(column=["weight", "height"], by=df.gender, layout=(1, 1))

    @pytest.mark.slow
    # 定义测试方法，验证当 'by' 参数为 None 时，不支持使用 'layout' 参数
    def test_grouped_box_layout_needs_by(self, hist_df):
        # 将参数 hist_df 赋值给 df
        df = hist_df
        # 验证当 'by' 参数为 None 时，不支持使用 'layout' 参数，抛出指定异常消息的 ValueError 异常
        msg = "The 'layout' keyword is not supported when 'by' is None"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(
                column=["height", "weight", "category"],
                layout=(2, 1),
                return_type="dict",
            )

    @pytest.mark.slow
    # 定义测试方法，验证布局参数中至少有一个正数维度
    def test_grouped_box_layout_positive_layout(self, hist_df):
        # 将参数 hist_df 赋值给 df
        df = hist_df
        # 验证布局参数中至少有一个正数维度，抛出指定异常消息的 ValueError 异常
        msg = "At least one dimension of layout must be positive"
        with pytest.raises(ValueError, match=msg):
            df.boxplot(column=["weight", "height"], by=df.gender, layout=(-1, -1))

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "gb_key, axes_num, rows",
        [["gender", 2, 1], ["category", 4, 2], ["classroom", 3, 2]],
    )
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "col, visible", [["height", False], ["weight", True], ["category", True]]
    )
    def test_grouped_box_layout_visible(self, hist_df, col, visible):
        df = hist_df
        # GH 5897
        # 生成箱线图，按照性别分组，每组显示指定列的数据，返回类型为 axes 对象
        axes = df.boxplot(
            column=["height", "weight", "category"], by="gender", return_type="axes"
        )
        # 检查图形当前的轴的形状是否符合预期，应为 3 个子图在 2x2 布局中
        _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(2, 2))
        # 获取指定列的轴对象
        ax = axes[col]
        # 检查 x 轴刻度标签是否可见，根据 visible 参数决定
        _check_visible(ax.get_xticklabels(), visible=visible)
        # 检查 x 轴标签是否可见，根据 visible 参数决定
        _check_visible([ax.xaxis.get_label()], visible=visible)
    def test_grouped_box_multiple_axes(self, hist_df):
        # GH 6970, GH 7069
        # 从参数中获取数据框
        df = hist_df

        # 检查忽略 sharex / sharey 的警告
        # 此检查应在第一个函数中进行，该函数传递多个轴以绘制 plot、hist 或 boxplot
        # 如果添加了更早按字母顺序排序的其他测试，应更改位置
        with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
            # 创建一个 2x2 的子图，并获取子图对象
            _, axes = mpl.pyplot.subplots(2, 2)
            # 根据 "category" 列对数据进行分组，并绘制 boxplot 到指定的轴对象 axes
            df.groupby("category").boxplot(column="height", return_type="axes", ax=axes)
            # 检查当前图形中轴的形状是否符合预期
            _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=4, layout=(2, 2))

    @pytest.mark.slow
    def test_grouped_box_multiple_axes_on_fig(self, hist_df):
        # GH 6970, GH 7069
        # 从参数中获取数据框
        df = hist_df
        # 创建一个 2x3 的图形和轴对象
        fig, axes = mpl.pyplot.subplots(2, 3)
        with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
            # 根据 "gender" 列对数据进行分组，并绘制 boxplot 到指定的轴对象 axes[0]
            returned = df.boxplot(
                column=["height", "weight", "category"],
                by="gender",
                return_type="axes",
                ax=axes[0],
            )
        # 将返回的轴对象字典值转换为数组
        returned = np.array(list(returned.values))
        # 检查返回的轴对象数组的形状是否符合预期
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        # 断言返回的轴对象与指定的 axes[0] 相同
        tm.assert_numpy_array_equal(returned, axes[0])
        # 断言返回的第一个轴对象所在的图形与指定的 fig 相同

        # 在第二行绘制
        with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
            # 根据 "classroom" 列对数据进行分组，并绘制 boxplot 到指定的轴对象 axes[1]
            returned = df.groupby("classroom").boxplot(
                column=["height", "weight", "category"], return_type="axes", ax=axes[1]
            )
        # 将返回的轴对象字典值转换为数组
        returned = np.array(list(returned.values))
        # 检查返回的轴对象数组的形状是否符合预期
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        # 断言返回的轴对象与指定的 axes[1] 相同
        tm.assert_numpy_array_equal(returned, axes[1])
        # 断言返回的第一个轴对象所在的图形与指定的 fig 相同

    @pytest.mark.slow
    def test_grouped_box_multiple_axes_ax_error(self, hist_df):
        # GH 6970, GH 7069
        # 从参数中获取数据框
        df = hist_df
        # 错误消息
        msg = "The number of passed axes must be 3, the same as the output plot"
        _, axes = mpl.pyplot.subplots(2, 3)
        with pytest.raises(ValueError, match=msg):
            # 传递不同数量的轴对象，与所需的不符
            with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
                # 根据 "classroom" 列对数据进行分组，并绘制 boxplot 到指定的轴对象 axes
                axes = df.groupby("classroom").boxplot(ax=axes)

    def test_fontsize(self):
        # 创建一个数据框
        df = DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [0, 0, 0, 1, 1, 1]})
        # 绘制 "a" 列的 boxplot，按 "b" 列分组，并设置字体大小为 16
        _check_ticks_props(
            df.boxplot("a", by="b", fontsize=16), xlabelsize=16, ylabelsize=16
        )
    @pytest.mark.parametrize(
        "col, expected_xticklabel",
        [  # 定义参数化测试的参数，包括列名和预期的 X 轴刻度标签
            ("v", ["(a, v)", "(b, v)", "(c, v)", "(d, v)", "(e, v)"]),  # 当列为单个 'v' 时的预期 X 轴刻度标签
            (["v"], ["(a, v)", "(b, v)", "(c, v)", "(d, v)", "(e, v)"]),  # 当列为列表 ['v'] 时的预期 X 轴刻度标签
            ("v1", ["(a, v1)", "(b, v1)", "(c, v1)", "(d, v1)", "(e, v1)"]),  # 当列为单个 'v1' 时的预期 X 轴刻度标签
            (["v", "v1"],  # 当列为列表 ['v', 'v1'] 时的预期 X 轴刻度标签，包含不同列组合的标签
                [
                    "(a, v)",
                    "(a, v1)",
                    "(b, v)",
                    "(b, v1)",
                    "(c, v)",
                    "(c, v1)",
                    "(d, v)",
                    "(d, v1)",
                    "(e, v)",
                    "(e, v1)",
                ],
            ),
            (None,  # 当列为 None 时的预期 X 轴刻度标签，与上一个相同
                [
                    "(a, v)",
                    "(a, v1)",
                    "(b, v)",
                    "(b, v1)",
                    "(c, v)",
                    "(c, v1)",
                    "(d, v)",
                    "(d, v1)",
                    "(e, v)",
                    "(e, v1)",
                ],
            ),
        ],
    )
    def test_groupby_boxplot_subplots_false(self, col, expected_xticklabel):
        # GH 16748
        # 创建一个 DataFrame 包含 'cat' 列和随机生成的 'v', 'v1' 列
        df = DataFrame(
            {
                "cat": np.random.default_rng(2).choice(list("abcde"), 100),
                "v": np.random.default_rng(2).random(100),
                "v1": np.random.default_rng(2).random(100),
            }
        )
        # 根据 'cat' 列对 DataFrame 进行分组
        grouped = df.groupby("cat")

        # 调用 _check_plot_works 函数，检查 boxplot 绘图功能，设置 subplots=False
        axes = _check_plot_works(
            grouped.boxplot, subplots=False, column=col, return_type="axes"
        )

        # 获取绘图结果中 X 轴刻度标签的文本内容
        result_xticklabel = [x.get_text() for x in axes.get_xticklabels()]
        # 断言预期的 X 轴刻度标签与实际结果一致
        assert expected_xticklabel == result_xticklabel

    def test_groupby_boxplot_object(self, hist_df):
        # GH 43480
        # 将 hist_df 转换为 object 类型的 DataFrame
        df = hist_df.astype("object")
        # 根据 'gender' 列对 DataFrame 进行分组
        grouped = df.groupby("gender")
        # 设置错误信息文本
        msg = "boxplot method requires numerical columns, nothing to plot"
        # 使用 pytest 检查调用 _check_plot_works 函数时是否会引发 ValueError，并匹配错误信息
        with pytest.raises(ValueError, match=msg):
            _check_plot_works(grouped.boxplot, subplots=False)

    def test_boxplot_multiindex_column(self):
        # GH 16748
        # 定义包含多级索引的 DataFrame
        arrays = [
            ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
            ["one", "two", "one", "two", "one", "two", "one", "two"],
        ]
        tuples = list(zip(*arrays))
        index = MultiIndex.from_tuples(tuples, names=["first", "second"])
        # 创建具有随机标准正态分布值的 DataFrame，使用多级索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((3, 8)),
            index=["A", "B", "C"],
            columns=index,
        )

        col = [("bar", "one"), ("bar", "two")]
        # 调用 _check_plot_works 函数，检查 boxplot 绘图功能，传递列名参数 col
        axes = _check_plot_works(df.boxplot, column=col, return_type="axes")

        # 设置预期的 X 轴刻度标签
        expected_xticklabel = ["(bar, one)", "(bar, two)"]
        # 获取绘图结果中 X 轴刻度标签的文本内容
        result_xticklabel = [x.get_text() for x in axes.get_xticklabels()]
        # 断言预期的 X 轴刻度标签与实际结果一致
        assert expected_xticklabel == result_xticklabel

    @pytest.mark.parametrize("group", ["X", ["X", "Y"]])
    # 定义一个测试函数，用于测试多组别分组的箱线图绘制
    def test_boxplot_multi_groupby_groups(self, group):
        # GH 14701：参考 GitHub 上的 issue 编号，用于跟踪问题或功能请求
        # 创建一个包含随机正态分布数据的 DataFrame，包括两列 Col1 和 Col2
        rows = 20
        df = DataFrame(
            np.random.default_rng(12).normal(size=(rows, 2)), columns=["Col1", "Col2"]
        )
        # 在 DataFrame 中添加两列 X 和 Y，X 列重复值 "A" 和 "B"，Y 列交替值 "C" 和 "D"
        df["X"] = Series(np.repeat(["A", "B"], int(rows / 2)))
        df["Y"] = Series(np.tile(["C", "D"], int(rows / 2)))
        # 按照指定的分组列 group 进行分组
        grouped = df.groupby(group)
        
        # 调用 _check_plot_works 函数，测试 df.boxplot 方法是否正常工作，按照指定的 group 进行分组
        _check_plot_works(df.boxplot, by=group, default_axes=True)
        # 调用 _check_plot_works 函数，测试 df.plot.box 方法是否正常工作，按照指定的 group 进行分组
        _check_plot_works(df.plot.box, by=group, default_axes=True)
        # 调用 _check_plot_works 函数，测试 grouped.boxplot 方法是否正常工作，不指定分组列
        _check_plot_works(grouped.boxplot, default_axes=True)
```