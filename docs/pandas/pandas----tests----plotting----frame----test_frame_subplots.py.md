# `D:\src\scipysrc\pandas\pandas\tests\plotting\frame\test_frame_subplots.py`

```
"""Test cases for DataFrame.plot"""

import string  # 导入字符串模块，用于生成字母序列

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试用例

from pandas.compat import is_platform_linux  # 导入pandas兼容模块，用于平台兼容性检查
from pandas.compat.numpy import np_version_gte1p24  # 导入pandas兼容NumPy模块，用于版本兼容性检查

import pandas as pd  # 导入pandas库，并重命名为pd
from pandas import (  # 导入pandas的子模块和函数
    DataFrame,  # 数据框对象
    Series,  # 系列对象
    date_range,  # 时间范围生成函数
)
import pandas._testing as tm  # 导入pandas的测试辅助模块
from pandas.tests.plotting.common import (  # 导入pandas绘图测试的常用函数
    _check_axes_shape,  # 检查轴形状的函数
    _check_box_return_type,  # 检查返回类型的函数
    _check_legend_labels,  # 检查图例标签的函数
    _check_ticks_props,  # 检查刻度属性的函数
    _check_visible,  # 检查可见性的函数
    _flatten_visible,  # 扁平化可见性的函数
)

from pandas.io.formats.printing import pprint_thing  # 导入pandas格式打印模块的打印函数

mpl = pytest.importorskip("matplotlib")  # 导入并检查matplotlib库是否可用
plt = pytest.importorskip("matplotlib.pyplot")  # 导入并检查matplotlib.pyplot模块是否可用


class TestDataFramePlotsSubplots:
    @pytest.mark.slow  # 标记为慢速测试用例
    @pytest.mark.parametrize("kind", ["bar", "barh", "line", "area"])  # 参数化测试，测试不同的图表类型
    def test_subplots(self, kind):
        df = DataFrame(  # 创建一个数据框对象
            np.random.default_rng(2).random((10, 3)),  # 使用随机数填充的10x3数组作为数据
            index=list(string.ascii_letters[:10]),  # 使用前10个字母作为索引
        )

        axes = df.plot(kind=kind, subplots=True, sharex=True, legend=True)  # 绘制指定类型的子图，共享x轴，显示图例
        _check_axes_shape(axes, axes_num=3, layout=(3, 1))  # 检查子图的形状是否符合预期
        assert axes.shape == (3,)  # 断言子图的形状为(3,)

        for ax, column in zip(axes, df.columns):
            _check_legend_labels(ax, labels=[pprint_thing(column)])  # 检查子图中图例标签的正确性

        for ax in axes[:-2]:
            _check_visible(ax.xaxis)  # 检查x轴是否可见，用于绘制网格
            _check_visible(ax.get_xticklabels(), visible=False)  # 检查x轴刻度标签是否不可见
            if kind != "bar":
                # 修改 https://github.com/pandas-dev/pandas/issues/26714
                _check_visible(ax.get_xticklabels(minor=True), visible=False)  # 检查次要x轴刻度标签是否不可见
            _check_visible(ax.xaxis.get_label(), visible=False)  # 检查x轴标签是否不可见
            _check_visible(ax.get_yticklabels())  # 检查y轴刻度标签是否可见

        _check_visible(axes[-1].xaxis)  # 检查最后一个子图的x轴是否可见
        _check_visible(axes[-1].get_xticklabels())  # 检查最后一个子图的x轴刻度标签是否可见
        _check_visible(axes[-1].get_xticklabels(minor=True))  # 检查最后一个子图的次要x轴刻度标签是否可见
        _check_visible(axes[-1].xaxis.get_label())  # 检查最后一个子图的x轴标签是否可见
        _check_visible(axes[-1].get_yticklabels())  # 检查最后一个子图的y轴刻度标签是否可见

    @pytest.mark.slow  # 标记为慢速测试用例
    @pytest.mark.parametrize("kind", ["bar", "barh", "line", "area"])  # 参数化测试，测试不同的图表类型
    def test_subplots_no_share_x(self, kind):
        df = DataFrame(  # 创建一个数据框对象
            np.random.default_rng(2).random((10, 3)),  # 使用随机数填充的10x3数组作为数据
            index=list(string.ascii_letters[:10]),  # 使用前10个字母作为索引
        )
        axes = df.plot(kind=kind, subplots=True, sharex=False)  # 绘制指定类型的子图，不共享x轴
        for ax in axes:
            _check_visible(ax.xaxis)  # 检查x轴是否可见
            _check_visible(ax.get_xticklabels())  # 检查x轴刻度标签是否可见
            _check_visible(ax.get_xticklabels(minor=True))  # 检查次要x轴刻度标签是否可见
            _check_visible(ax.xaxis.get_label())  # 检查x轴标签是否可见
            _check_visible(ax.get_yticklabels())  # 检查y轴刻度标签是否可见

    @pytest.mark.slow  # 标记为慢速测试用例
    @pytest.mark.parametrize("kind", ["bar", "barh", "line", "area"])  # 参数化测试，测试不同的图表类型
    def test_subplots_no_legend(self, kind):
        df = DataFrame(  # 创建一个数据框对象
            np.random.default_rng(2).random((10, 3)),  # 使用随机数填充的10x3数组作为数据
            index=list(string.ascii_letters[:10]),  # 使用前10个字母作为索引
        )
        axes = df.plot(kind=kind, subplots=True, legend=False)  # 绘制指定类型的子图，不显示图例
        for ax in axes:
            assert ax.get_legend() is None  # 断言子图中不存在图例
    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_subplots_timeseries(self, kind):
        # 使用pytest标记参数化测试，测试种类包括"line"和"area"
        idx = date_range(start="2014-07-01", freq="ME", periods=10)
        # 创建日期范围索引，从2014年7月1日开始，频率为每月末，共10个周期
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=idx)
        # 创建一个包含随机数据的DataFrame，形状为10行3列，索引为idx

        axes = df.plot(kind=kind, subplots=True, sharex=True)
        # 在DataFrame上绘制指定类型(kind)的图形，subplots=True表示每列为一个子图，sharex=True表示共享x轴
        _check_axes_shape(axes, axes_num=3, layout=(3, 1))
        # 调用检查函数确保axes的形状为(3, 1)，即3行1列子图

        for ax in axes[:-2]:
            # 遍历除最后两个子图之外的所有子图
            # GH 7801
            # GH 7801相关注释（假设GH 7801是一个GitHub问题或特性）
            _check_visible(ax.xaxis)  # 检查x轴是否可见以便显示网格
            _check_visible(ax.get_xticklabels(), visible=False)
            # 检查x轴刻度标签是否可见，预期为不可见
            _check_visible(ax.get_xticklabels(minor=True), visible=False)
            # 检查x轴次要刻度标签是否可见，预期为不可见
            _check_visible(ax.xaxis.get_label(), visible=False)
            # 检查x轴标签是否可见，预期为不可见
            _check_visible(ax.get_yticklabels())
            # 检查y轴刻度标签是否可见

        _check_visible(axes[-1].xaxis)
        # 检查最后一个子图的x轴是否可见
        _check_visible(axes[-1].get_xticklabels())
        # 检查最后一个子图的x轴刻度标签是否可见
        _check_visible(axes[-1].get_xticklabels(minor=True))
        # 检查最后一个子图的x轴次要刻度标签是否可见
        _check_visible(axes[-1].xaxis.get_label())
        # 检查最后一个子图的x轴标签是否可见
        _check_visible(axes[-1].get_yticklabels())
        # 检查最后一个子图的y轴刻度标签是否可见
        _check_ticks_props(axes, xrot=0)
        # 检查子图的刻度属性，设置x轴旋转角度为0度

    @pytest.mark.parametrize("kind", ["line", "area"])
    def test_subplots_timeseries_rot(self, kind):
        # 使用pytest标记参数化测试，测试种类包括"line"和"area"
        idx = date_range(start="2014-07-01", freq="ME", periods=10)
        # 创建日期范围索引，从2014年7月1日开始，频率为每月末，共10个周期
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=idx)
        # 创建一个包含随机数据的DataFrame，形状为10行3列，索引为idx
        axes = df.plot(kind=kind, subplots=True, sharex=False, rot=45, fontsize=7)
        # 在DataFrame上绘制指定类型(kind)的图形，subplots=True表示每列为一个子图，sharex=False表示不共享x轴，rot=45表示x轴标签旋转45度，fontsize=7设置标签字体大小

        for ax in axes:
            # 遍历所有子图
            _check_visible(ax.xaxis)
            # 检查每个子图的x轴是否可见
            _check_visible(ax.get_xticklabels())
            # 检查每个子图的x轴刻度标签是否可见
            _check_visible(ax.get_xticklabels(minor=True))
            # 检查每个子图的x轴次要刻度标签是否可见
            _check_visible(ax.xaxis.get_label())
            # 检查每个子图的x轴标签是否可见
            _check_visible(ax.get_yticklabels())
            # 检查每个子图的y轴刻度标签是否可见
            _check_ticks_props(ax, xlabelsize=7, xrot=45, ylabelsize=7)
            # 检查子图的刻度属性，设置x轴标签大小为7，旋转角度为45度，y轴标签大小为7

    @pytest.mark.parametrize(
        "col", ["numeric", "timedelta", "datetime_no_tz", "datetime_all_tz"]
    )
    def test_subplots_timeseries_y_axis(self, col):
        # 使用pytest标记参数化测试，测试列包括"numeric"、"timedelta"、"datetime_no_tz"和"datetime_all_tz"
        # GH16953
        # GH16953相关注释（假设GH16953是一个GitHub问题或特性）
        data = {
            "numeric": np.array([1, 2, 5]),
            "timedelta": [
                pd.Timedelta(-10, unit="s"),
                pd.Timedelta(10, unit="m"),
                pd.Timedelta(10, unit="h"),
            ],
            "datetime_no_tz": [
                pd.to_datetime("2017-08-01 00:00:00"),
                pd.to_datetime("2017-08-01 02:00:00"),
                pd.to_datetime("2017-08-02 00:00:00"),
            ],
            "datetime_all_tz": [
                pd.to_datetime("2017-08-01 00:00:00", utc=True),
                pd.to_datetime("2017-08-01 02:00:00", utc=True),
                pd.to_datetime("2017-08-02 00:00:00", utc=True),
            ],
            "text": ["This", "should", "fail"],
        }
        # 创建包含不同数据类型的数据字典

        testdata = DataFrame(data)
        # 使用数据字典创建DataFrame

        ax = testdata.plot(y=col)
        # 在DataFrame上绘制y轴为指定列(col)的图形
        result = ax.get_lines()[0].get_data()[1]
        # 获取图形的第一条线的y轴数据
        expected = testdata[col].values
        # 获取期望的y轴数据
        assert (result == expected).all()
        # 断言结果与期望一致
    # 定义一个测试方法，用于测试绘制时间序列的图表时的异常处理
    def test_subplots_timeseries_y_text_error(self):
        # 用例名称标识为 GH16953
        data = {
            "numeric": np.array([1, 2, 5]),  # 数字类型的数据
            "text": ["This", "should", "fail"],  # 文本类型的数据
        }
        testdata = DataFrame(data)  # 创建数据框架
        msg = "no numeric data to plot"  # 异常消息字符串
        # 使用 pytest 断言捕获特定异常，并匹配指定消息
        with pytest.raises(TypeError, match=msg):
            testdata.plot(y="text")  # 绘制图表，预期引发 TypeError 异常

    # 标记为预期失败的测试方法，原因是不支持周期、分类和混合时区日期时间类型
    @pytest.mark.xfail(reason="not support for period, categorical, datetime_mixed_tz")
    def test_subplots_timeseries_y_axis_not_supported(self):
        """
        此测试将失败的原因：
            period:
                由于 ``select_dtypes`` 中尚未实现周期类型，
                需要自定义值转换器和刻度格式化器（与 x 轴图表相同处理方式）

            categorical:
                需要自定义值转换器和刻度格式化器（当前不支持 x 轴，与现有情况相同）

            datetime_mixed_tz:
                由于 pandas 处理带有不同时区的 ``Series`` 的 ``datetime`` 对象方式，
                通常将 ``datetime`` 对象转换为带时区感知的形式，有助于解决此问题
        """
        data = {
            "numeric": np.array([1, 2, 5]),  # 数字类型的数据
            "period": [  # 周期类型数据列表
                pd.Period("2017-08-01 00:00:00", freq="h"),
                pd.Period("2017-08-01 02:00", freq="h"),
                pd.Period("2017-08-02 00:00:00", freq="h"),
            ],
            "categorical": pd.Categorical(  # 分类类型数据
                ["c", "b", "a"], categories=["a", "b", "c"], ordered=False
            ),
            "datetime_mixed_tz": [  # 混合时区日期时间类型数据列表
                pd.to_datetime("2017-08-01 00:00:00", utc=True),
                pd.to_datetime("2017-08-01 02:00:00"),
                pd.to_datetime("2017-08-02 00:00:00"),
            ],
        }
        testdata = DataFrame(data)  # 创建数据框架
        # 绘制三种不支持类型的图表，断言检查是否正常绘制
        ax_period = testdata.plot(x="numeric", y="period")
        assert (
            ax_period.get_lines()[0].get_data()[1] == testdata["period"].values
        ).all()
        ax_categorical = testdata.plot(x="numeric", y="categorical")
        assert (
            ax_categorical.get_lines()[0].get_data()[1]
            == testdata["categorical"].values
        ).all()
        ax_datetime_mixed_tz = testdata.plot(x="numeric", y="datetime_mixed_tz")
        assert (
            ax_datetime_mixed_tz.get_lines()[0].get_data()[1]
            == testdata["datetime_mixed_tz"].values
        ).all()

    # 参数化测试方法，测试不同的布局参数是否符合预期的布局结果
    @pytest.mark.parametrize(
        "layout, exp_layout",
        [
            [(2, 2), (2, 2)],  # 布局为 2 行 2 列，预期结果为 (2, 2)
            [(-1, 2), (2, 2)],  # 布局为 (-1, 2)，预期结果为 (2, 2)
            [(2, -1), (2, 2)],  # 布局为 (2, -1)，预期结果为 (2, 2)
            [(1, 4), (1, 4)],  # 布局为 1 行 4 列，预期结果为 (1, 4)
            [(-1, 4), (1, 4)],  # 布局为 (-1, 4)，预期结果为 (1, 4)
            [(4, -1), (4, 1)],  # 布局为 (4, -1)，预期结果为 (4, 1)
        ],
    )
    def test_subplots_layout_multi_column(self, layout, exp_layout):
        # GH 6667
        # 创建一个包含随机数据的 DataFrame，共 10 行 3 列
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )

        # 使用 DataFrame 的 plot 方法绘制子图，按照给定的布局参数 layout
        axes = df.plot(subplots=True, layout=layout)
        # 检查生成的子图的形状是否符合预期
        _check_axes_shape(axes, axes_num=3, layout=exp_layout)
        # 断言子图的形状与期望的布局参数 exp_layout 相等
        assert axes.shape == exp_layout

    def test_subplots_layout_multi_column_error(self):
        # GH 6667
        # 创建一个包含随机数据的 DataFrame，共 10 行 3 列
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        # 定义一个错误消息，用于异常检查
        msg = "Layout of 1x1 must be larger than required size 3"

        # 使用 pytest 检查是否抛出 ValueError 异常，并且异常信息匹配预期的消息
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, layout=(1, 1))

        # 再次定义一个错误消息，用于异常检查
        msg = "At least one dimension of layout must be positive"
        # 使用 pytest 检查是否抛出 ValueError 异常，并且异常信息匹配预期的消息
        with pytest.raises(ValueError, match=msg):
            df.plot(subplots=True, layout=(-1, -1))

    @pytest.mark.parametrize(
        "kwargs, expected_axes_num, expected_layout, expected_shape",
        [
            ({}, 1, (1, 1), (1,)),  # 参数化测试，期望生成单个子图
            ({"layout": (3, 3)}, 1, (3, 3), (3, 3)),  # 参数化测试，期望生成 3x3 布局的子图
        ],
    )
    def test_subplots_layout_single_column(
        self, kwargs, expected_axes_num, expected_layout, expected_shape
    ):
        # GH 6667
        # 创建一个包含随机数据的 DataFrame，共 10 行 1 列
        df = DataFrame(
            np.random.default_rng(2).random((10, 1)),
            index=list(string.ascii_letters[:10]),
        )
        # 使用 DataFrame 的 plot 方法绘制子图，传入额外的参数 kwargs
        axes = df.plot(subplots=True, **kwargs)
        # 检查生成的子图的形状是否符合预期
        _check_axes_shape(
            axes,
            axes_num=expected_axes_num,
            layout=expected_layout,
        )
        # 断言子图的形状与期望的形状参数 expected_shape 相等
        assert axes.shape == expected_shape

    @pytest.mark.slow
    @pytest.mark.parametrize("idx", [range(5), date_range("1/1/2000", periods=5)])
    def test_subplots_warnings(self, idx):
        # GH 9464
        # 使用 tm.assert_produces_warning 确保在测试中不会产生警告
        with tm.assert_produces_warning(None):
            # 创建一个包含随机数据的 DataFrame，共 5 行 4 列，使用传入的 idx 参数作为索引
            df = DataFrame(np.random.default_rng(2).standard_normal((5, 4)), index=idx)
            # 使用 DataFrame 的 plot 方法绘制子图，指定布局为 (3, 2)
            df.plot(subplots=True, layout=(3, 2))

    def test_subplots_multiple_axes(self):
        # GH 5353, 6970, GH 7069
        # 创建一个 2x3 的子图布局，返回图形对象 fig 和子图对象 axes
        fig, axes = mpl.pyplot.subplots(2, 3)
        # 创建一个包含随机数据的 DataFrame，共 10 行 3 列
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )

        # 使用 DataFrame 的 plot 方法绘制子图，将其绘制到指定的 axes[0] 上
        returned = df.plot(subplots=True, ax=axes[0], sharex=False, sharey=False)
        # 检查生成的子图的形状是否符合预期
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        # 断言返回的子图的形状为 (3,)
        assert returned.shape == (3,)
        # 断言返回的第一个子图的图形对象是刚创建的 fig 对象
        assert returned[0].figure is fig

        # 在第二行上绘制子图
        returned = df.plot(subplots=True, ax=axes[1], sharex=False, sharey=False)
        # 再次检查生成的子图的形状是否符合预期
        _check_axes_shape(returned, axes_num=3, layout=(1, 3))
        # 断言返回的子图的形状为 (3,)
        assert returned.shape == (3,)
        # 断言返回的第一个子图的图形对象是刚创建的 fig 对象
        assert returned[0].figure is fig
        # 最后检查整体子图布局的形状是否符合预期
        _check_axes_shape(axes, axes_num=6, layout=(2, 3))
    def test_subplots_multiple_axes_error(self):
        # GH 5353, 6970, GH 7069
        # 创建一个包含随机数据的 DataFrame，使用 ASCII 字母作为索引
        df = DataFrame(
            np.random.default_rng(2).random((10, 3)),
            index=list(string.ascii_letters[:10]),
        )
        # 设置错误消息，要求传入的 axes 数量必须为 3，与输出图形的数量相同
        msg = "The number of passed axes must be 3, the same as the output plot"
        # 创建一个 2x3 的子图，并获取对应的 axes
        _, axes = mpl.pyplot.subplots(2, 3)

        # 使用 pytest 检测是否会抛出 ValueError，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            # 调用 DataFrame 的 plot 方法，传入 subplots=True 和获取的 axes
            df.plot(subplots=True, ax=axes)

    @pytest.mark.parametrize(
        "layout, exp_layout",
        [
            [(2, 1), (2, 2)],
            [(2, -1), (2, 2)],
            [(-1, 2), (2, 2)],
        ],
    )
    def test_subplots_multiple_axes_2_dim(self, layout, exp_layout):
        # GH 5353, 6970, GH 7069
        # 传入 2 维的 axes 和无效的布局参数
        # 无效的布局参数不应影响输入和返回值
        # （警告显示在 TestDataFrameGroupByPlots.test_grouped_box_multiple_axes 中测试）
        # 创建一个 2x2 的子图，并获取对应的 axes
        _, axes = mpl.pyplot.subplots(2, 2)
        # 创建一个包含随机数据的 DataFrame，使用 ASCII 字母作为索引
        df = DataFrame(
            np.random.default_rng(2).random((10, 4)),
            index=list(string.ascii_letters[:10]),
        )
        # 使用 tm.assert_produces_warning 检测是否会产生 UserWarning，并匹配特定的警告消息
        with tm.assert_produces_warning(UserWarning, match="layout keyword is ignored"):
            # 调用 DataFrame 的 plot 方法，传入 subplots=True、axes 和其他参数
            returned = df.plot(
                subplots=True, ax=axes, layout=layout, sharex=False, sharey=False
            )
            # 检查返回的 axes 的形状是否符合预期
            _check_axes_shape(returned, axes_num=4, layout=exp_layout)
            # 断言返回值的形状为 (4,)
            assert returned.shape == (4,)

    def test_subplots_multiple_axes_single_col(self):
        # GH 5353, 6970, GH 7069
        # 单列图
        # 创建一个 1x1 的子图，并获取对应的 axes
        _, axes = mpl.pyplot.subplots(1, 1)
        # 创建一个包含随机数据的 DataFrame，使用 ASCII 字母作为索引
        df = DataFrame(
            np.random.default_rng(2).random((10, 1)),
            index=list(string.ascii_letters[:10]),
        )

        # 调用 DataFrame 的 plot 方法，传入 subplots=True、单个 axes 和其他参数
        axes = df.plot(subplots=True, ax=[axes], sharex=False, sharey=False)
        # 检查返回的 axes 的形状是否符合预期
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))
        # 断言返回值的形状为 (1,)
        assert axes.shape == (1,)
    def test_subplots_ts_share_axes(self):
        # GH 3964
        # 创建一个 3x3 的子图网格，所有子图共享相同的 x 和 y 轴
        _, axes = mpl.pyplot.subplots(3, 3, sharex=True, sharey=True)
        # 调整子图之间的布局，设置左右边距为0.05到0.95，水平和垂直间距为0.3
        mpl.pyplot.subplots_adjust(left=0.05, right=0.95, hspace=0.3, wspace=0.3)
        # 创建一个包含随机数据的 DataFrame，使用日期时间索引
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 9)),
            index=date_range(start="2014-07-01", freq="ME", periods=10),
        )
        # 在每个子图上绘制 DataFrame 的列，设置字体大小为5
        for i, ax in enumerate(axes.ravel()):
            df[i].plot(ax=ax, fontsize=5)

        # 除了底部行以外的行不应该可见
        # 检查每个子图的 x 轴刻度标签是否可见，底部行应可见
        for ax in axes[0:-1].ravel():
            _check_visible(ax.get_xticklabels(), visible=False)

        # 底部行的 x 轴刻度标签应可见
        for ax in axes[-1].ravel():
            _check_visible(ax.get_xticklabels(), visible=True)

        # 第一列的 y 轴刻度标签应可见
        for ax in axes[[0, 1, 2], [0]].ravel():
            _check_visible(ax.get_yticklabels(), visible=True)

        # 其它列的 y 轴刻度标签不应可见
        for ax in axes[[0, 1, 2], [1]].ravel():
            _check_visible(ax.get_yticklabels(), visible=False)
        for ax in axes[[0, 1, 2], [2]].ravel():
            _check_visible(ax.get_yticklabels(), visible=False)

    def test_subplots_sharex_axes_existing_axes(self):
        # GH 9158
        # 创建一个包含数据的字典
        d = {"A": [1.0, 2.0, 3.0, 4.0], "B": [4.0, 3.0, 2.0, 1.0], "C": [5, 1, 3, 4]}
        # 创建一个包含日期时间索引的 DataFrame
        df = DataFrame(d, index=date_range("2014 10 11", "2014 10 14"))

        # 在子图上绘制 DataFrame 的 A 和 B 列，并共享 x 轴
        axes = df[["A", "B"]].plot(subplots=True)
        # 在第一个子图上绘制 DataFrame 的 C 列，并使用辅助 y 轴
        df["C"].plot(ax=axes[0], secondary_y=True)

        # 检查第一个子图的 x 轴刻度标签是否不可见
        _check_visible(axes[0].get_xticklabels(), visible=False)
        # 检查第二个子图的 x 轴刻度标签是否可见
        _check_visible(axes[1].get_xticklabels(), visible=True)
        # 检查所有子图的 y 轴刻度标签是否可见
        for ax in axes.ravel():
            _check_visible(ax.get_yticklabels(), visible=True)

    def test_subplots_dup_columns(self):
        # GH 10962
        # 创建一个包含随机数据的 DataFrame，列名为 'a'
        df = DataFrame(np.random.default_rng(2).random((5, 5)), columns=list("aaaaa"))
        # 在子图上绘制 DataFrame 的每一列，并检查图例标签是否为 'a'，并且每个子图只有一条线
        axes = df.plot(subplots=True)
        for ax in axes:
            _check_legend_labels(ax, labels=["a"])
            assert len(ax.lines) == 1

    def test_subplots_dup_columns_secondary_y(self):
        # GH 10962
        # 创建一个包含随机数据的 DataFrame，列名为 'a'
        df = DataFrame(np.random.default_rng(2).random((5, 5)), columns=list("aaaaa"))
        # 在子图上绘制 DataFrame 的每一列，使用 'a' 列作为辅助 y 轴，检查图例标签和线条数目
        axes = df.plot(subplots=True, secondary_y="a")
        for ax in axes:
            # 当 subplots=True 时，图例标签只包含 'a'
            _check_legend_labels(ax, labels=["a"])
            assert len(ax.lines) == 1

    def test_subplots_dup_columns_secondary_y_no_subplot(self):
        # GH 10962
        # 创建一个包含随机数据的 DataFrame，列名为 'a'
        df = DataFrame(np.random.default_rng(2).random((5, 5)), columns=list("aaaaa"))
        # 在单个子图上绘制 DataFrame，并使用 'a' 列作为辅助 y 轴，检查图例标签和线条数目
        ax = df.plot(secondary_y="a")
        # 检查图例标签是否包含 'a (right)'，并且主轴和辅助轴的线条数目
        _check_legend_labels(ax, labels=["a (right)"] * 5)
        assert len(ax.lines) == 0
        assert len(ax.right_ax.lines) == 5

    @pytest.mark.xfail(
        np_version_gte1p24 and is_platform_linux(),
        reason="Weird rounding problems",
        strict=False,
    )
    def test_bar_log_no_subplots(self):
        # GH3254, GH3298 matplotlib/matplotlib#1882, #1892
        # regressions in 1.2.1
        # 定义预期的对数刻度位置数组
        expected = np.array([0.1, 1.0, 10.0, 100])

        # 创建一个 DataFrame 对象，包含两列数据"A"和"B"
        df = DataFrame({"A": [3] * 5, "B": list(range(1, 6))}, index=range(5))
        # 绘制柱状图，并设置网格和对数刻度
        ax = df.plot.bar(grid=True, log=True)
        # 断言 y 轴的刻度位置是否与预期数组一致
        tm.assert_numpy_array_equal(ax.yaxis.get_ticklocs(), expected)

    @pytest.mark.xfail(
        np_version_gte1p24 and is_platform_linux(),
        reason="Weird rounding problems",
        strict=False,
    )
    def test_bar_log_subplots(self):
        # 定义预期的对数刻度位置数组
        expected = np.array([0.1, 1.0, 10.0, 100.0, 1000.0, 1e4])

        # 创建一个 DataFrame 包含两个 Series，并绘制柱状图的子图，设置对数刻度
        ax = DataFrame([Series([200, 300]), Series([300, 500])]).plot.bar(
            log=True, subplots=True
        )

        # 断言每个子图的 y 轴刻度位置是否与预期数组一致
        tm.assert_numpy_array_equal(ax[0].yaxis.get_ticklocs(), expected)
        tm.assert_numpy_array_equal(ax[1].yaxis.get_ticklocs(), expected)

    def test_boxplot_subplots_return_type_default(self, hist_df):
        # 使用传入的 hist_df 数据创建一个 DataFrame 对象
        df = hist_df

        # 使用默认参数绘制箱线图的子图，返回结果应为 Series 类型
        result = df.plot.box(subplots=True)
        assert isinstance(result, Series)
        # 检查返回的结果类型及其属性
        _check_box_return_type(
            result, None, expected_keys=["height", "weight", "category"]
        )

    @pytest.mark.parametrize("rt", ["dict", "axes", "both"])
    def test_boxplot_subplots_return_type(self, hist_df, rt):
        # 使用传入的 hist_df 数据创建一个 DataFrame 对象
        df = hist_df
        # 使用参数化测试方式测试不同的返回类型，绘制箱线图的子图
        returned = df.plot.box(return_type=rt, subplots=True)
        # 检查返回的结果类型及其属性
        _check_box_return_type(
            returned,
            rt,
            expected_keys=["height", "weight", "category"],
            check_ax_title=False,
        )

    def test_df_subplots_patterns_minorticks(self):
        # GH 10657
        # 创建一个包含随机数据的 DataFrame 对象，指定行索引和列名
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            index=date_range("1/1/2000", periods=10),
            columns=list("AB"),
        )

        # 创建共享 x 轴的两个子图
        _, axes = plt.subplots(2, 1, sharex=True)
        # 在指定的 axes 上绘制 DataFrame 的子图
        axes = df.plot(subplots=True, ax=axes)
        # 对每个子图进行断言，确保每个子图只有一条线可见
        for ax in axes:
            assert len(ax.lines) == 1
            # 检查 y 轴刻度标签是否可见
            _check_visible(ax.get_yticklabels(), visible=True)
        # 第一个子图的 x 轴刻度标签应该隐藏
        _check_visible(axes[0].get_xticklabels(), visible=False)
        _check_visible(axes[0].get_xticklabels(minor=True), visible=False)
        # 第二个子图的 x 轴刻度标签应该可见
        _check_visible(axes[1].get_xticklabels(), visible=True)
        _check_visible(axes[1].get_xticklabels(minor=True), visible=True)
    def test_df_subplots_patterns_minorticks_1st_ax_hidden(self):
        # GH 10657
        # 创建一个包含随机数据的 DataFrame
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            index=date_range("1/1/2000", periods=10),
            columns=list("AB"),
        )
        # 创建一个包含两个子图的图表对象
        _, axes = plt.subplots(2, 1)
        # 断言会产生 UserWarning，其中匹配 "sharex and sharey" 的警告信息
        with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
            # 使用 DataFrame 的 plot 方法绘制子图，并共享 x 轴
            axes = df.plot(subplots=True, ax=axes, sharex=True)
        # 对每个子图进行断言，确保每个子图只有一条线
        for ax in axes:
            assert len(ax.lines) == 1
            # 检查 y 轴刻度标签是否可见
            _check_visible(ax.get_yticklabels(), visible=True)
        # 第一个子图的 x 轴刻度标签必须隐藏
        _check_visible(axes[0].get_xticklabels(), visible=False)
        # 第一个子图的次要 x 轴刻度标签也必须隐藏
        _check_visible(axes[0].get_xticklabels(minor=True), visible=False)
        # 第二个子图的 x 轴刻度标签必须可见
        _check_visible(axes[1].get_xticklabels(), visible=True)
        # 第二个子图的次要 x 轴刻度标签必须可见
        _check_visible(axes[1].get_xticklabels(minor=True), visible=True)

    def test_df_subplots_patterns_minorticks_not_shared(self):
        # GH 10657
        # 创建一个包含随机数据的 DataFrame
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)),
            index=date_range("1/1/2000", periods=10),
            columns=list("AB"),
        )
        # 创建一个包含两个子图的图表对象
        _, axes = plt.subplots(2, 1)
        # 使用 DataFrame 的 plot 方法绘制子图，但不共享轴
        axes = df.plot(subplots=True, ax=axes)
        # 对每个子图进行断言，确保每个子图只有一条线
        for ax in axes:
            assert len(ax.lines) == 1
            # 检查 y 轴和 x 轴刻度标签是否可见
            _check_visible(ax.get_yticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)

    def test_subplots_sharex_false(self):
        # 当 sharex 设置为 False 时进行测试，两个图应该有不同的标签，GH 25160
        # 创建一个包含随机数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).random((10, 2)))
        # 将 DataFrame 的部分数据设置为 NaN
        df.iloc[5:, 1] = np.nan
        df.iloc[:5, 0] = np.nan

        # 创建一个包含两个子图的图表对象
        _, axs = mpl.pyplot.subplots(2, 1)
        # 使用 DataFrame 的 plot 方法在子图上绘制线图，不共享 x 轴
        df.plot.line(ax=axs, subplots=True, sharex=False)

        # 断言第一个子图的预期 x 轴刻度值
        expected_ax1 = np.arange(4.5, 10, 0.5)
        tm.assert_numpy_array_equal(axs[0].get_xticks(), expected_ax1)
        # 断言第二个子图的预期 x 轴刻度值
        expected_ax2 = np.arange(-0.5, 5, 0.5)
        tm.assert_numpy_array_equal(axs[1].get_xticks(), expected_ax2)

    def test_subplots_constrained_layout(self, temp_file):
        # GH 25261
        # 创建一个包含随机数据的 DataFrame
        idx = date_range(start="now", periods=10)
        df = DataFrame(np.random.default_rng(2).random((10, 3)), index=idx)
        kwargs = {}
        # 如果 Figure 类有 get_constrained_layout 方法，则使用约束布局
        if hasattr(mpl.pyplot.Figure, "get_constrained_layout"):
            kwargs["constrained_layout"] = True
        # 创建一个包含两个子图的图表对象
        _, axes = mpl.pyplot.subplots(2, **kwargs)
        # 不会产生任何警告
        with tm.assert_produces_warning(None):
            # 使用 DataFrame 的 plot 方法在第一个子图上绘制
            df.plot(ax=axes[0])
            # 将图表保存到临时文件
            with temp_file.open(mode="wb") as path:
                mpl.pyplot.savefig(path)
    # 使用 pytest 的参数化装饰器，为 test_xlabel_ylabel_dataframe_subplots 方法添加三个不同的测试用例：line、area 和 bar
    @pytest.mark.parametrize("kind", ["line", "area", "bar"])
    def test_xlabel_ylabel_dataframe_subplots(
        self, kind, index_name, old_label, new_label
    ):
        # GH 9093
        # 创建一个 DataFrame 包含两列数据，列名为 "Type A" 和 "Type B"
        df = DataFrame([[1, 2], [2, 5]], columns=["Type A", "Type B"])
        # 设置 DataFrame 的索引名称为 index_name
    
        # 对 DataFrame 进行指定类型（kind）的图形绘制，subplots 参数为 True，返回所有绘图的 Axes 对象列表
        axes = df.plot(kind=kind, subplots=True)
        # 断言所有 Axes 对象的 ylabel 均为空字符串
        assert all(ax.get_ylabel() == "" for ax in axes)
        # 断言所有 Axes 对象的 xlabel 均为 old_label
    
        # 对 DataFrame 进行指定类型（kind）的图形绘制，设置新的 ylabel 和 xlabel，subplots 参数为 True，返回所有绘图的 Axes 对象列表
        axes = df.plot(kind=kind, ylabel=new_label, xlabel=new_label, subplots=True)
        # 断言所有 Axes 对象的 ylabel 均为字符串形式的 new_label
        assert all(ax.get_ylabel() == str(new_label) for ax in axes)
        # 断言所有 Axes 对象的 xlabel 均为字符串形式的 new_label
    
    # 使用 pytest 的参数化装饰器，为 test_bar_align_multiple_columns 方法添加多个不同的测试用例
    @pytest.mark.parametrize(
        "kwargs",
        [
            # stacked center
            {"kind": "bar", "stacked": True},
            {"kind": "bar", "stacked": True, "width": 0.9},
            {"kind": "barh", "stacked": True},
            {"kind": "barh", "stacked": True, "width": 0.9},
            # center
            {"kind": "bar", "stacked": False},
            {"kind": "bar", "stacked": False, "width": 0.9},
            {"kind": "barh", "stacked": False},
            {"kind": "barh", "stacked": False, "width": 0.9},
            # subplots center
            {"kind": "bar", "subplots": True},
            {"kind": "bar", "subplots": True, "width": 0.9},
            {"kind": "barh", "subplots": True},
            {"kind": "barh", "subplots": True, "width": 0.9},
            # align edge
            {"kind": "bar", "stacked": True, "align": "edge"},
            {"kind": "bar", "stacked": True, "width": 0.9, "align": "edge"},
            {"kind": "barh", "stacked": True, "align": "edge"},
            {"kind": "barh", "stacked": True, "width": 0.9, "align": "edge"},
            {"kind": "bar", "stacked": False, "align": "edge"},
            {"kind": "bar", "stacked": False, "width": 0.9, "align": "edge"},
            {"kind": "barh", "stacked": False, "align": "edge"},
            {"kind": "barh", "stacked": False, "width": 0.9, "align": "edge"},
            {"kind": "bar", "subplots": True, "align": "edge"},
            {"kind": "bar", "subplots": True, "width": 0.9, "align": "edge"},
            {"kind": "barh", "subplots": True, "align": "edge"},
            {"kind": "barh", "subplots": True, "width": 0.9, "align": "edge"},
        ],
    )
    def test_bar_align_multiple_columns(self, kwargs):
        # GH2157
        # 创建一个 DataFrame 包含两列数据，列名为 "A" 和 "B"，索引为从 0 到 4
        df = DataFrame({"A": [3] * 5, "B": list(range(5))}, index=range(5))
        # 调用 _check_bar_alignment 方法，传入 DataFrame 和 kwargs 字典作为参数
        self._check_bar_alignment(df, **kwargs)
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kind": "bar", "stacked": False},  # 参数化测试，测试柱状图不堆叠情况
            {"kind": "bar", "stacked": True},   # 参数化测试，测试柱状图堆叠情况
            {"kind": "barh", "stacked": False}, # 参数化测试，测试水平柱状图不堆叠情况
            {"kind": "barh", "stacked": True},  # 参数化测试，测试水平柱状图堆叠情况
            {"kind": "bar", "subplots": True},  # 参数化测试，测试多子图柱状图
            {"kind": "barh", "subplots": True}, # 参数化测试，测试多子图水平柱状图
        ],
    )
    def test_bar_align_single_column(self, kwargs):
        df = DataFrame(np.random.default_rng(2).standard_normal(5))  # 创建一个包含随机数据的DataFrame
        self._check_bar_alignment(df, **kwargs)  # 调用测试辅助方法检查柱状图对齐情况

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kind": "bar", "stacked": False},  # 参数化测试，测试柱状图不堆叠情况
            {"kind": "bar", "stacked": True},   # 参数化测试，测试柱状图堆叠情况
            {"kind": "barh", "stacked": False}, # 参数化测试，测试水平柱状图不堆叠情况
            {"kind": "barh", "stacked": True},  # 参数化测试，测试水平柱状图堆叠情况
            {"kind": "bar", "subplots": True},  # 参数化测试，测试多子图柱状图
            {"kind": "barh", "subplots": True}, # 参数化测试，测试多子图水平柱状图
        ],
    )
    def test_bar_barwidth_position(self, kwargs):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))  # 创建一个包含随机数据的DataFrame
        self._check_bar_alignment(df, width=0.9, position=0.2, **kwargs)  # 调用测试辅助方法检查柱状图对齐情况，并设置柱宽和位置

    @pytest.mark.parametrize("w", [1, 1.0])
    def test_bar_barwidth_position_int(self, w):
        # GH 12979
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))  # 创建一个包含随机数据的DataFrame
        ax = df.plot.bar(stacked=True, width=w)  # 绘制柱状图，设置堆叠和宽度
        ticks = ax.xaxis.get_ticklocs()  # 获取x轴刻度位置
        tm.assert_numpy_array_equal(ticks, np.array([0, 1, 2, 3, 4]))  # 断言刻度位置为数组[0, 1, 2, 3, 4]
        assert ax.get_xlim() == (-0.75, 4.75)  # 断言x轴的显示范围
        # 检查柱子的左边缘位置
        assert ax.patches[0].get_x() == -0.5  # 断言第一个柱子的左边缘位置
        assert ax.patches[-1].get_x() == 3.5   # 断言最后一个柱子的左边缘位置

    @pytest.mark.parametrize(
        "kind, kwargs",
        [
            ["bar", {"stacked": True}],    # 参数化测试，测试堆叠柱状图
            ["barh", {"stacked": False}],  # 参数化测试，测试非堆叠水平柱状图
            ["barh", {"stacked": True}],   # 参数化测试，测试堆叠水平柱状图
            ["bar", {"subplots": True}],   # 参数化测试，测试多子图柱状图
            ["barh", {"subplots": True}],  # 参数化测试，测试多子图水平柱状图
        ],
    )
    def test_bar_barwidth_position_int_width_1(self, kind, kwargs):
        # GH 12979
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))  # 创建一个包含随机数据的DataFrame
        self._check_bar_alignment(df, kind=kind, width=1, **kwargs)  # 调用测试辅助方法检查柱状图对齐情况，并设置柱宽为1

    def _check_bar_alignment(
        self,
        df,
        kind="bar",
        stacked=False,
        subplots=False,
        align="center",
        width=0.5,
        position=0.5,
    ):
        # 根据传入参数绘制 DataFrame 中的数据，返回绘制的 Axes 对象
        axes = df.plot(
            kind=kind,              # 绘图类型，如柱状图、条形图等
            stacked=stacked,        # 是否堆叠（针对柱状图）
            subplots=subplots,      # 是否生成子图
            align=align,            # 对齐方式，"center"或"edge"
            width=width,            # 图形宽度
            position=position,      # 位置参数
            grid=True,              # 是否显示网格
        )

        # 将可能的子图 Axes 对象展平，以便统一处理
        axes = _flatten_visible(axes)

        # 遍历每个 Axes 对象进行后续处理
        for ax in axes:
            if kind == "bar":
                axis = ax.xaxis            # x 轴对象
                ax_min, ax_max = ax.get_xlim()  # 获取 x 轴的最小和最大限制
                # 计算柱状图条的最小和最大边缘位置
                min_edge = min(p.get_x() for p in ax.patches)
                max_edge = max(p.get_x() + p.get_width() for p in ax.patches)
            elif kind == "barh":
                axis = ax.yaxis            # y 轴对象
                ax_min, ax_max = ax.get_ylim()  # 获取 y 轴的最小和最大限制
                # 计算条形图条的最小和最大边缘位置
                min_edge = min(p.get_y() for p in ax.patches)
                max_edge = max(p.get_y() + p.get_height() for p in ax.patches)
            else:
                raise ValueError

            # GH 7498
            # 检查 lim 和条边缘之间的边距
            tm.assert_almost_equal(ax_min, min_edge - 0.25)
            tm.assert_almost_equal(ax_max, max_edge + 0.25)

            # 获取第一个条或柱对象
            p = ax.patches[0]
            if kind == "bar" and (stacked is True or subplots is True):
                edge = p.get_x()  # 获取柱边缘位置
                center = edge + p.get_width() * position  # 计算柱中心位置
            elif kind == "bar" and stacked is False:
                center = p.get_x() + p.get_width() * len(df.columns) * position
                edge = p.get_x()
            elif kind == "barh" and (stacked is True or subplots is True):
                center = p.get_y() + p.get_height() * position
                edge = p.get_y()
            elif kind == "barh" and stacked is False:
                center = p.get_y() + p.get_height() * len(df.columns) * position
                edge = p.get_y()
            else:
                raise ValueError

            # 检查刻度是否位于整数位置
            assert (axis.get_ticklocs() == np.arange(len(df))).all()

            if align == "center":
                # 检查条或柱是否居中
                tm.assert_almost_equal(axis.get_ticklocs()[0], center)
            elif align == "edge":
                # 检查条或柱的边缘是否从刻度开始
                tm.assert_almost_equal(axis.get_ticklocs()[0], edge)
            else:
                raise ValueError

        return axes
```