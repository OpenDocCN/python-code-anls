# `D:\src\scipysrc\seaborn\tests\_core\test_plot.py`

```
# 导入所需的模块和库

import io  # 导入io模块，用于处理字节流
import xml  # 导入xml模块，可能用于处理XML数据
import functools  # 导入functools模块，用于高阶函数的操作
import itertools  # 导入itertools模块，提供迭代器相关的工具函数
import warnings  # 导入warnings模块，用于管理警告信息

import numpy as np  # 导入NumPy库，并使用np别名
import pandas as pd  # 导入Pandas库，并使用pd别名
import matplotlib as mpl  # 导入matplotlib库，并使用mpl别名
import matplotlib.pyplot as plt  # 从matplotlib库中导入pyplot模块，并使用plt别名
from PIL import Image  # 从PIL库中导入Image模块

import pytest  # 导入pytest测试框架
from pandas.testing import assert_frame_equal, assert_series_equal  # 从Pandas测试模块中导入数据帧和序列的相等断言函数
from numpy.testing import assert_array_equal, assert_array_almost_equal  # 从NumPy测试模块中导入数组相等和几乎相等的断言函数

from seaborn._core.plot import Plot, PlotConfig, Default  # 从seaborn库的_core.plot模块中导入Plot类、PlotConfig类和Default类
from seaborn._core.scales import Continuous, Nominal, Temporal  # 从seaborn库的_core.scales模块中导入Continuous类、Nominal类和Temporal类
from seaborn._core.moves import Move, Shift, Dodge  # 从seaborn库的_core.moves模块中导入Move类、Shift类和Dodge类
from seaborn._core.rules import categorical_order  # 从seaborn库的_core.rules模块中导入categorical_order函数
from seaborn._core.exceptions import PlotSpecError  # 从seaborn库的_core.exceptions模块中导入PlotSpecError异常类
from seaborn._marks.base import Mark  # 从seaborn库的_marks.base模块中导入Mark类
from seaborn._stats.base import Stat  # 从seaborn库的_stats.base模块中导入Stat类
from seaborn._marks.dot import Dot  # 从seaborn库的_marks.dot模块中导入Dot类
from seaborn._stats.aggregation import Agg  # 从seaborn库的_stats.aggregation模块中导入Agg类
from seaborn.utils import _version_predates  # 从seaborn库的utils模块中导入_version_predates函数

# 定义assert_vector_equal为assert_series_equal的偏函数
assert_vector_equal = functools.partial(
    assert_series_equal, check_names=False, check_dtype=False,
)


def assert_gridspec_shape(ax, nrows=1, ncols=1):
    # 获取ax对象的gridspec并断言其行数和列数与给定的nrows和ncols参数相等
    gs = ax.get_gridspec()
    assert gs.nrows == nrows
    assert gs.ncols == ncols


class MockMark(Mark):
    # MockMark类继承自Mark类，用于模拟标记对象的行为

    _grouping_props = ["color"]  # 类属性_grouping_props，包含要分组的属性列表

    def __init__(self, *args, **kwargs):
        # 初始化方法，接受任意位置参数和关键字参数

        super().__init__(*args, **kwargs)  # 调用父类的初始化方法
        self.passed_keys = []  # 用于存储传递给_plot方法的键的列表
        self.passed_data = []  # 用于存储传递给_plot方法的数据的列表
        self.passed_axes = []  # 用于存储传递给_plot方法的轴对象的列表
        self.passed_scales = None  # 用于存储传递给_legend_artist方法的比例尺对象
        self.passed_orient = None  # 用于存储传递给_legend_artist方法的方向
        self.n_splits = 0  # 记录_split方法被调用的次数的计数器

    def _plot(self, split_gen, scales, orient):
        # 模拟绘图方法，接受split_gen生成器、scales比例尺对象和orient方向参数

        for keys, data, ax in split_gen():  # 迭代split_gen生成器产生的结果
            self.n_splits += 1  # 每调用一次增加一次计数
            self.passed_keys.append(keys)  # 将keys添加到passed_keys列表
            self.passed_data.append(data)  # 将data添加到passed_data列表
            self.passed_axes.append(ax)  # 将ax添加到passed_axes列表

        self.passed_scales = scales  # 存储传递的比例尺对象
        self.passed_orient = orient  # 存储传递的方向参数

    def _legend_artist(self, variables, value, scales):
        # 模拟图例艺术家方法，接受变量、值和比例尺对象作为参数

        a = mpl.lines.Line2D([], [])  # 创建一个空的Line2D对象a
        a.variables = variables  # 设置a的variables属性为传入的variables
        a.value = value  # 设置a的value属性为传入的value
        return a  # 返回创建的Line2D对象a


class TestInit:
    # TestInit类，用于测试初始化方法和功能

    def test_empty(self):
        # 测试空初始化方法

        p = Plot()  # 创建一个Plot对象p，不传入任何数据
        assert p._data.source_data is None  # 断言source_data属性为None
        assert p._data.source_vars == {}  # 断言source_vars属性为空字典

    def test_data_only(self, long_df):
        # 测试只传入数据的初始化方法

        p = Plot(long_df)  # 创建一个Plot对象p，传入长数据框long_df
        assert p._data.source_data is long_df  # 断言source_data属性为传入的long_df
        assert p._data.source_vars == {}  # 断言source_vars属性为空字典

    def test_df_and_named_variables(self, long_df):
        # 测试传入数据框和命名变量的初始化方法

        variables = {"x": "a", "y": "z"}  # 定义变量字典
        p = Plot(long_df, **variables)  # 创建一个Plot对象p，传入长数据框long_df和变量字典
        for var, col in variables.items():  # 遍历变量字典的键值对
            assert_vector_equal(p._data.frame[var], long_df[col])  # 断言Plot对象中变量对应的数据列与long_df中的对应列相等
        assert p._data.source_data is long_df  # 断言source_data属性为传入的long_df
        assert p._data.source_vars.keys() == variables.keys()  # 断言source_vars属性的键与变量字典的键相等
    # 测试函数，验证包含数据框和混合变量的情况
    def test_df_and_mixed_variables(self, long_df):
        # 定义变量字典，包含字符串变量和数据框列
        variables = {"x": "a", "y": long_df["z"]}
        # 创建 Plot 对象，传入数据框和变量字典作为参数
        p = Plot(long_df, **variables)
        # 遍历变量字典
        for var, col in variables.items():
            # 检查数据框的列与 Plot 对象的数据帧中对应变量的值是否相等
            if isinstance(col, str):
                assert_vector_equal(p._data.frame[var], long_df[col])
            else:
                assert_vector_equal(p._data.frame[var], col)
        # 检查 Plot 对象的源数据是否为 long_df
        assert p._data.source_data is long_df
        # 检查 Plot 对象的源变量名集合是否与变量字典的键集合相同
        assert p._data.source_vars.keys() == variables.keys()

    # 测试函数，验证仅包含向量变量的情况
    def test_vector_variables_only(self, long_df):
        # 定义变量字典，仅包含数据框的列
        variables = {"x": long_df["a"], "y": long_df["z"]}
        # 创建 Plot 对象，传入变量字典作为参数
        p = Plot(**variables)
        # 遍历变量字典
        for var, col in variables.items():
            # 检查数据框的列与 Plot 对象的数据帧中对应变量的值是否相等
            assert_vector_equal(p._data.frame[var], col)
        # 检查 Plot 对象的源数据是否为 None
        assert p._data.source_data is None
        # 检查 Plot 对象的源变量名集合是否与变量字典的键集合相同
        assert p._data.source_vars.keys() == variables.keys()

    # 测试函数，验证仅包含向量变量（但列没有索引）的情况
    def test_vector_variables_no_index(self, long_df):
        # 定义变量字典，包含转换为 NumPy 数组和列表的向量变量
        variables = {"x": long_df["a"].to_numpy(), "y": long_df["z"].to_list()}
        # 创建 Plot 对象，传入变量字典作为参数
        p = Plot(**variables)
        # 遍历变量字典
        for var, col in variables.items():
            # 检查数据框的列与 Plot 对象的数据帧中对应变量的值是否相等
            assert_vector_equal(p._data.frame[var], pd.Series(col))
            # 检查 Plot 对象中变量名对应的数据帧名称是否为 None
            assert p._data.names[var] is None
        # 检查 Plot 对象的源数据是否为 None
        assert p._data.source_data is None
        # 检查 Plot 对象的源变量名集合是否与变量字典的键集合相同
        assert p._data.source_vars.keys() == variables.keys()

    # 测试函数，验证仅包含命名的数据框的情况
    def test_data_only_named(self, long_df):
        # 创建 Plot 对象，传入数据框作为命名参数
        p = Plot(data=long_df)
        # 检查 Plot 对象的源数据是否为 long_df
        assert p._data.source_data is long_df
        # 检查 Plot 对象的源变量名集合是否为空字典
        assert p._data.source_vars == {}

    # 测试函数，验证同时使用位置参数和命名参数指定数据框的情况
    def test_positional_and_named_data(self, long_df):
        # 定义错误信息
        err = "`data` given by both name and position"
        # 使用 pytest 检查同时使用位置参数和命名参数的情况是否抛出 TypeError 异常
        with pytest.raises(TypeError, match=err):
            Plot(long_df, data=long_df)

    # 使用 pytest.mark.parametrize 注册的测试函数，验证同时使用位置参数和命名参数指定 x、y 变量的情况
    @pytest.mark.parametrize("var", ["x", "y"])
    def test_positional_and_named_xy(self, long_df, var):
        # 根据变量 var 构造错误信息
        err = f"`{var}` given by both name and position"
        # 使用 pytest 检查同时使用位置参数和命名参数的情况是否抛出 TypeError 异常
        with pytest.raises(TypeError, match=err):
            Plot(long_df, "a", "b", **{var: "c"})

    # 测试函数，验证仅使用位置参数指定 x、y 变量的情况
    def test_positional_data_x_y(self, long_df):
        # 创建 Plot 对象，仅使用位置参数指定 x、y 变量
        p = Plot(long_df, "a", "b")
        # 检查 Plot 对象的源数据是否为 long_df
        assert p._data.source_data is long_df
        # 检查 Plot 对象的源变量名集合是否为 ["x", "y"]
        assert list(p._data.source_vars) == ["x", "y"]

    # 测试函数，验证仅使用位置参数指定 x、y 变量（但参数为单列）的情况
    def test_positional_x_y(self, long_df):
        # 创建 Plot 对象，仅使用位置参数指定 x、y 变量（均为单列）
        p = Plot(long_df["a"], long_df["b"])
        # 检查 Plot 对象的源数据是否为 None
        assert p._data.source_data is None
        # 检查 Plot 对象的源变量名集合是否为 ["x", "y"]
        assert list(p._data.source_vars) == ["x", "y"]

    # 测试函数，验证仅使用位置参数指定 x 变量的情况
    def test_positional_data_x(self, long_df):
        # 创建 Plot 对象，仅使用位置参数指定 x 变量
        p = Plot(long_df, "a")
        # 检查 Plot 对象的源数据是否为 long_df
        assert p._data.source_data is long_df
        # 检查 Plot 对象的源变量名集合是否为 ["x"]
        assert list(p._data.source_vars) == ["x"]

    # 测试函数，验证仅使用位置参数指定 x 变量（但参数为单列）的情况
    def test_positional_x(self, long_df):
        # 创建 Plot 对象，仅使用位置参数指定 x 变量（单列）
        p = Plot(long_df["a"])
        # 检查 Plot 对象的源数据是否为 None
        assert p._data.source_data is None
        # 检查 Plot 对象的源变量名集合是否为 ["x"]
        assert list(p._data.source_vars) == ["x"]

    # 根据条件跳过测试函数，前提是存在 dataframe interchange 的支持
    @pytest.mark.skipif(
        condition=not hasattr(pd.api, "interchange"),
        reason="Tests behavior assuming support for dataframe interchange"
    )
    def test_positional_interchangeable_dataframe(self, mock_long_df, long_df):
        # 创建 Plot 对象，传入 mock_long_df 作为数据，"x" 作为命名参数
        p = Plot(mock_long_df, x="x")
        # 检查 Plot 对象的源数据与 long_df 是否相等
        assert_frame_equal(p._data.source_data, long_df)
    # 测试函数，用于验证 Plot 类的行为是否符合预期，测试传入过多的位置参数的情况
    def test_positional_too_many(self, long_df):
        # 定义错误信息正则表达式，用于匹配预期的异常错误信息
        err = r"Plot\(\) accepts no more than 3 positional arguments \(data, x, y\)"
        # 使用 pytest 的异常断言，验证调用 Plot 类时传入过多的位置参数是否会抛出 TypeError 异常并匹配预期错误信息
        with pytest.raises(TypeError, match=err):
            # 调用 Plot 类，并传入过多的位置参数 'x', 'y', 'z'
            Plot(long_df, "x", "y", "z")
    
    # 测试函数，用于验证 Plot 类的行为是否符合预期，测试传入未知的关键字参数的情况
    def test_unknown_keywords(self, long_df):
        # 定义错误信息正则表达式，用于匹配预期的异常错误信息
        err = r"Plot\(\) got unexpected keyword argument\(s\): bad"
        # 使用 pytest 的异常断言，验证调用 Plot 类时传入未知的关键字参数是否会抛出 TypeError 异常并匹配预期错误信息
        with pytest.raises(TypeError, match=err):
            # 调用 Plot 类，并传入未知的关键字参数 'bad'
            Plot(long_df, bad="x")
class TestLayerAddition:

    def test_without_data(self, long_df):
        # 创建 Plot 对象，并添加 MockMark，然后绘制图形
        p = Plot(long_df, x="x", y="y").add(MockMark()).plot()
        # 获取图形中的第一个图层
        layer, = p._layers
        # 断言 Plot 对象中的数据帧与图层中的数据帧相等
        assert_frame_equal(p._data.frame, layer["data"].frame, check_dtype=False)

    def test_with_new_variable_by_name(self, long_df):
        # 创建 Plot 对象，并添加 MockMark，指定新的 y 变量名，然后绘制图形
        p = Plot(long_df, x="x").add(MockMark(), y="y").plot()
        # 获取图形中的第一个图层
        layer, = p._layers
        # 断言图层中的数据帧的列名列表为 ["x", "y"]
        assert layer["data"].frame.columns.to_list() == ["x", "y"]
        # 遍历 "xy"，并断言图层中的数据与 long_df 中相应变量相等
        for var in "xy":
            assert_vector_equal(layer["data"].frame[var], long_df[var])

    def test_with_new_variable_by_vector(self, long_df):
        # 创建 Plot 对象，并添加 MockMark，使用 long_df 的 y 向量作为数据，然后绘制图形
        p = Plot(long_df, x="x").add(MockMark(), y=long_df["y"]).plot()
        # 获取图形中的第一个图层
        layer, = p._layers
        # 断言图层中的数据帧的列名列表为 ["x", "y"]
        assert layer["data"].frame.columns.to_list() == ["x", "y"]
        # 遍历 "xy"，并断言图层中的数据与 long_df 中相应变量相等
        for var in "xy":
            assert_vector_equal(layer["data"].frame[var], long_df[var])

    def test_with_late_data_definition(self, long_df):
        # 创建 Plot 对象，并添加 MockMark，延迟定义数据，然后绘制图形
        p = Plot().add(MockMark(), data=long_df, x="x", y="y").plot()
        # 获取图形中的第一个图层
        layer, = p._layers
        # 断言图层中的数据帧的列名列表为 ["x", "y"]
        assert layer["data"].frame.columns.to_list() == ["x", "y"]
        # 遍历 "xy"，并断言图层中的数据与 long_df 中相应变量相等
        for var in "xy":
            assert_vector_equal(layer["data"].frame[var], long_df[var])

    def test_with_new_data_definition(self, long_df):
        # 从 long_df 中随机抽取一部分作为新数据
        long_df_sub = long_df.sample(frac=.5)
        # 创建 Plot 对象，并添加 MockMark，使用 long_df_sub 作为数据，然后绘制图形
        p = Plot(long_df, x="x", y="y").add(MockMark(), data=long_df_sub).plot()
        # 获取图形中的第一个图层
        layer, = p._layers
        # 断言图层中的数据帧的列名列表为 ["x", "y"]
        assert layer["data"].frame.columns.to_list() == ["x", "y"]
        # 遍历 "xy"，并断言图层中的数据与 long_df_sub 对应的部分与 long_df 相等
        for var in "xy":
            assert_vector_equal(
                layer["data"].frame[var], long_df_sub[var].reindex(long_df.index)
            )

    def test_drop_variable(self, long_df):
        # 创建 Plot 对象，并添加 MockMark，移除 y 变量，然后绘制图形
        p = Plot(long_df, x="x", y="y").add(MockMark(), y=None).plot()
        # 获取图形中的第一个图层
        layer, = p._layers
        # 断言图层中的数据帧的列名列表为 ["x"]
        assert layer["data"].frame.columns.to_list() == ["x"]
        # 断言图层中的 x 变量与 long_df 中的 x 变量相等（忽略数据类型）
        assert_vector_equal(layer["data"].frame["x"], long_df["x"], check_dtype=False)

    @pytest.mark.xfail(reason="Need decision on default stat")
    def test_stat_default(self):
        # 定义一个带默认统计方法的 Mark 子类
        class MarkWithDefaultStat(Mark):
            default_stat = Stat

        # 创建 Plot 对象，并添加 MarkWithDefaultStat 实例，然后绘制图形
        p = Plot().add(MarkWithDefaultStat())
        # 获取图形中的第一个图层
        layer, = p._layers
        # 断言图层中的统计方法为默认的 Stat 类
        assert layer["stat"].__class__ is Stat

    def test_stat_nondefault(self):
        # 定义一个带默认统计方法的 Mark 子类
        class MarkWithDefaultStat(Mark):
            default_stat = Stat

        # 定义另一个自定义的统计方法类
        class OtherMockStat(Stat):
            pass

        # 创建 Plot 对象，并添加 MarkWithDefaultStat 实例和 OtherMockStat 实例，然后绘制图形
        p = Plot().add(MarkWithDefaultStat(), OtherMockStat())
        # 获取图形中的第一个图层
        layer, = p._layers
        # 断言图层中的统计方法为 OtherMockStat 类
        assert layer["stat"].__class__ is OtherMockStat

    @pytest.mark.parametrize(
        "arg,expected",
        [("x", "x"), ("y", "y"), ("v", "x"), ("h", "y")],
    )
    # 定义测试方法，验证Plot类在设置orient参数时的行为是否符合预期
    def test_orient(self, arg, expected):

        # 定义模拟的统计类，记录orient参数在调用时的取值
        class MockStatTrackOrient(Stat):
            def __call__(self, data, groupby, orient, scales):
                self.orient_at_call = orient
                return data

        # 定义模拟的移动类，记录orient参数在调用时的取值
        class MockMoveTrackOrient(Move):
            def __call__(self, data, groupby, orient, scales):
                self.orient_at_call = orient
                return data

        # 创建模拟的统计对象和移动对象
        s = MockStatTrackOrient()
        m = MockMoveTrackOrient()

        # 创建Plot对象，添加模拟标记、统计对象和移动对象，并设定orient参数
        Plot(x=[1, 2, 3], y=[1, 2, 3]).add(MockMark(), s, m, orient=arg).plot()

        # 验证统计对象和移动对象在调用时的orient参数取值是否符合预期
        assert s.orient_at_call == expected
        assert m.orient_at_call == expected

    # 定义测试方法，验证Plot类在处理不同数据输入时，正确设置_variables属性
    def test_variable_list(self, long_df):

        # 测试默认情况下，以及指定x和y参数时，_variables属性的取值
        p = Plot(long_df, x="x", y="y")
        assert p._variables == ["x", "y"]

        # 添加模拟标记，再次验证_variables属性的取值
        p = Plot(long_df).add(MockMark(), x="x", y="y")
        assert p._variables == ["x", "y"]

        # 测试在指定y和color参数时，_variables属性的取值
        p = Plot(long_df, y="x", color="a").add(MockMark(), x="y")
        assert p._variables == ["y", "color", "x"]

        # 测试在指定x、y和color参数，且color参数设为None时，_variables属性的取值
        p = Plot(long_df, x="x", y="y", color="a").add(MockMark(), color=None)
        assert p._variables == ["x", "y", "color"]

        # 测试链式调用时，添加多个模拟标记后，_variables属性的取值
        p = (
            Plot(long_df, x="x", y="y")
            .add(MockMark(), color="a")
            .add(MockMark(), alpha="s")
        )
        assert p._variables == ["x", "y", "color", "alpha"]

        # 测试在指定y参数后，使用pair方法添加x参数后，_variables属性的取值
        p = Plot(long_df, y="x").pair(x=["a", "b"])
        assert p._variables == ["y", "x0", "x1"]

    # 定义测试方法，验证Plot类在处理不同类型参数时，正确抛出异常
    def test_type_checks(self):

        # 创建空的Plot对象
        p = Plot()

        # 验证添加非Mark实例时，是否抛出TypeError异常
        with pytest.raises(TypeError, match="mark must be a Mark instance"):
            p.add(MockMark)

        # 定义模拟的统计类和移动类
        class MockStat(Stat):
            pass

        class MockMove(Move):
            pass

        # 验证同时添加多个Stat类型时，是否抛出TypeError异常
        err = "Transforms must have at most one Stat type"
        with pytest.raises(TypeError, match=err):
            p.add(MockMark(), MockStat)

        with pytest.raises(TypeError, match=err):
            p.add(MockMark(), MockMove(), MockStat())

        with pytest.raises(TypeError, match=err):
            p.add(MockMark(), MockMark(), MockStat())
class TestScaling:

    # 测试推断标度类型的方法
    def test_inference(self, long_df):
        # 遍历列名和标度类型的组合
        for col, scale_type in zip("zat", ["Continuous", "Nominal", "Temporal"]):
            # 创建 Plot 对象，并添加 MockMark，绘制图表
            p = Plot(long_df, x=col, y=col).add(MockMark()).plot()
            # 断言标度对象的类名是否与期望的标度类型相符
            for var in "xy":
                assert p._scales[var].__class__.__name__ == scale_type

    # 测试从层数据中推断标度
    def test_inference_from_layer_data(self):
        # 创建 Plot 对象，并添加 MockMark，并指定 x 轴数据
        p = Plot().add(MockMark(), x=["a", "b", "c"]).plot()
        # 断言标度对象在特定数据点处的值
        assert p._scales["x"]("b") == 1

    # 测试推断连接操作中的标度
    def test_inference_joins(self):
        # 创建 Plot 对象，添加不同数据源的 MockMark，并绘制图表
        p = (
            Plot(y=pd.Series([1, 2, 3, 4]))
            .add(MockMark(), x=pd.Series([1, 2]))
            .add(MockMark(), x=pd.Series(["a", "b"], index=[2, 3]))
            .plot()
        )
        # 断言标度对象在特定数据点处的值
        assert p._scales["x"]("a") == 2

    # 测试推断分类转换器的方法
    def test_inferred_categorical_converter(self):
        # 创建 Plot 对象，指定 x 轴数据为字符串列表，并添加 MockMark，并绘制图表
        p = Plot(x=["b", "c", "a"]).add(MockMark()).plot()
        # 获取图表的第一个轴对象
        ax = p._figure.axes[0]
        # 断言 x 轴的单位转换结果
        assert ax.xaxis.convert_units("c") == 1

    # 测试显式指定分类转换器的方法
    def test_explicit_categorical_converter(self):
        # 创建 Plot 对象，指定 y 轴数据为数字列表，并使用 Nominal 标度，并添加 MockMark，并绘制图表
        p = Plot(y=[2, 1, 3]).scale(y=Nominal()).add(MockMark()).plot()
        # 获取图表的第一个轴对象
        ax = p._figure.axes[0]
        # 断言 y 轴的单位转换结果
        assert ax.yaxis.convert_units("3") == 2

    # 标记为预期失败，因为暂未实现 Temporal 自动转换
    @pytest.mark.xfail(reason="Temporal auto-conversion not implemented")
    def test_categorical_as_datetime(self):
        # 创建 Plot 对象，指定 x 轴数据为日期列表，并使用未实现的 Temporal 标度，并添加 MockMark，并绘制图表
        dates = ["1970-01-03", "1970-01-02", "1970-01-04"]
        p = Plot(x=dates).scale(...).add(MockMark()).plot()
        p  # TODO
        ...

    # 测试分面对数标度的方法
    def test_faceted_log_scale(self):
        # 创建 Plot 对象，指定 y 轴数据为数字列表，并分面显示，y 轴使用对数标度，并绘制图表
        p = Plot(y=[1, 10]).facet(col=["a", "b"]).scale(y="log").plot()
        # 遍历图表的每个轴对象
        for ax in p._figure.axes:
            # 获取 y 轴的转换函数对象，并进行转换
            xfm = ax.yaxis.get_transform().transform
            # 断言转换结果与期望结果的数组是否相等
            assert_array_equal(xfm([1, 10, 100]), [0, 1, 2])

    # 测试配对的单一对数标度方法
    def test_paired_single_log_scale(self):
        # 定义两组数据
        x0, x1 = [1, 2, 3], [1, 10, 100]
        # 创建 Plot 对象，进行配对显示，其中 x1 轴使用对数标度，并绘制图表
        p = Plot().pair(x=[x0, x1]).scale(x1="log").plot()
        # 获取线性坐标轴和对数坐标轴的轴对象
        ax_lin, ax_log = p._figure.axes
        # 获取线性坐标轴 x 轴的转换函数对象，并进行转换
        xfm_lin = ax_lin.xaxis.get_transform().transform
        # 断言线性坐标轴 x 轴的转换结果与期望结果的数组是否相等
        assert_array_equal(xfm_lin([1, 10, 100]), [1, 10, 100])
        # 获取对数坐标轴 x 轴的转换函数对象，并进行转换
        xfm_log = ax_log.xaxis.get_transform().transform
        # 断言对数坐标轴 x 轴的转换结果与期望结果的数组是否相等
        assert_array_equal(xfm_log([1, 10, 100]), [0, 1, 2])

    # 测试具有公共回退的配对方法
    def test_paired_with_common_fallback(self):
        # 定义两组数据
        x0, x1 = [1, 2, 3], [1, 10, 100]
        # 创建 Plot 对象，进行配对显示，其中 x 轴使用 pow 标度，x1 轴使用对数标度，并绘制图表
        p = Plot().pair(x=[x0, x1]).scale(x="pow", x1="log").plot()
        # 获取幂次坐标轴和对数坐标轴的轴对象
        ax_pow, ax_log = p._figure.axes
        # 获取幂次坐标轴 x 轴的转换函数对象，并进行转换
        xfm_pow = ax_pow.xaxis.get_transform().transform
        # 断言幂次坐标轴 x 轴的转换结果与期望结果的数组是否相等
        assert_array_equal(xfm_pow([1, 2, 3]), [1, 4, 9])
        # 获取对数坐标轴 x 轴的转换函数对象，并进行转换
        xfm_log = ax_log.xaxis.get_transform().transform
        # 断言对数坐标轴 x 轴的转换结果与期望结果的数组是否相等
        assert_array_equal(xfm_log([1, 10, 100]), [0, 1, 2])

    # 标记为预期失败，因为自定义对数标度需要 log 名称以保持一致性
    @pytest.mark.xfail(reason="Custom log scale needs log name for consistency")
    def test_log_scale_name(self):
        # 创建 Plot 对象，指定 x 轴使用对数标度，并绘制图表
        p = Plot().scale(x="log").plot()
        # 获取图表的第一个轴对象
        ax = p._figure.axes[0]
        # 断言 x 轴的标度类型是否为对数标度
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "linear"
    # 测试用例：验证数据转换为对数后是否可以逆转回原始值
    def test_mark_data_log_transform_is_inverted(self, long_df):
        # 选择列 'z'
        col = "z"
        # 创建 MockMark 对象
        m = MockMark()
        # 使用 Plot 类绘制长格式数据的图表，对 x 轴进行对数缩放，并添加 MockMark 对象后绘制图表
        Plot(long_df, x=col).scale(x="log").add(m).plot()
        # 断言 MockMark 对象接收的数据经过 x 轴对数缩放后能够逆转回原始数据
        assert_vector_equal(m.passed_data[0]["x"], long_df[col])

    # 测试用例：验证在统计处理后进行对数缩放是否正确
    def test_mark_data_log_transfrom_with_stat(self, long_df):
        
        # 定义 Mean 类作为统计处理类
        class Mean(Stat):
            group_by_orient = True

            def __call__(self, data, groupby, orient, scales):
                # 根据 orient 确定其他方向的列名
                other = {"x": "y", "y": "x"}[orient]
                # 对数据进行均值计算
                return groupby.agg(data, {other: "mean"})

        # 选择列 'z' 和分组器 'a'
        col = "z"
        grouper = "a"
        # 创建 MockMark 和 Mean 对象
        m = MockMark()
        s = Mean()

        # 使用 Plot 类绘制长格式数据的图表，对 y 轴进行对数缩放，并添加 MockMark 和 Mean 对象后绘制图表
        Plot(long_df, x=grouper, y=col).scale(y="log").add(m, s).plot()

        # 期望的对数缩放后的数据
        expected = (
            long_df[col]
            .pipe(np.log)
            .groupby(long_df[grouper], sort=False)
            .mean()
            .pipe(np.exp)
            .reset_index(drop=True)
        )
        # 断言 MockMark 对象接收的数据经过 y 轴对数缩放后等于期望的数据
        assert_vector_equal(m.passed_data[0]["y"], expected)

    # 测试用例：验证从分类数据生成标记数据是否正确
    def test_mark_data_from_categorical(self, long_df):
        # 选择列 'a'
        col = "a"
        # 创建 MockMark 对象
        m = MockMark()
        # 使用 Plot 类绘制长格式数据的图表，添加 MockMark 对象后绘制图表
        Plot(long_df, x=col).add(m).plot()

        # 获取分类数据的排序顺序和映射关系
        levels = categorical_order(long_df[col])
        level_map = {x: float(i) for i, x in enumerate(levels)}
        # 断言 MockMark 对象接收的数据映射到数字的结果等于长格式数据列 'a' 映射到数字的结果
        assert_vector_equal(m.passed_data[0]["x"], long_df[col].map(level_map))

    # 测试用例：验证从日期时间数据生成标记数据是否正确
    def test_mark_data_from_datetime(self, long_df):
        # 选择列 't'
        col = "t"
        # 创建 MockMark 对象
        m = MockMark()
        # 使用 Plot 类绘制长格式数据的图表，添加 MockMark 对象后绘制图表
        Plot(long_df, x=col).add(m).plot()

        # 期望的日期时间数据转换为数值
        expected = long_df[col].map(mpl.dates.date2num)
        # 断言 MockMark 对象接收的数据转换为数值后等于期望的结果
        assert_vector_equal(m.passed_data[0]["x"], expected)

    # 测试用例：验证计算变量在图表中的转换是否正确
    def test_computed_var_ticks(self, long_df):
        # 定义 Identity 类作为计算变量的转换类
        class Identity(Stat):
            def __call__(self, df, groupby, orient, scales):
                # 根据 orient 确定其他方向的列名
                other = {"x": "y", "y": "x"}[orient]
                # 将 orient 列复制给 other 列
                return df.assign(**{other: df[orient]})

        # 设定刻度位置
        tick_locs = [1, 2, 5]
        # 创建连续型刻度对象，并设定刻度位置
        scale = Continuous().tick(at=tick_locs)
        # 使用 Plot 类绘制长格式数据的图表，添加 MockMark 和 Identity 对象后设定 y 轴的刻度，并绘制图表
        p = Plot(long_df, "x").add(MockMark(), Identity()).scale(y=scale).plot()
        # 获取图表的第一个坐标轴对象
        ax = p._figure.axes[0]
        # 断言图表坐标轴的 y 轴刻度等于设定的刻度位置
        assert_array_equal(ax.get_yticks(), tick_locs)

    # 测试用例：验证计算变量在对数缩放下的转换是否正确
    def test_computed_var_transform(self, long_df):
        # 定义 Identity 类作为计算变量的转换类
        class Identity(Stat):
            def __call__(self, df, groupby, orient, scales):
                # 根据 orient 确定其他方向的列名
                other = {"x": "y", "y": "x"}[orient]
                # 将 orient 列复制给 other 列
                return df.assign(**{other: df[orient]})

        # 使用 Plot 类绘制长格式数据的图表，添加 MockMark 和 Identity 对象后设定 y 轴为对数缩放，并绘制图表
        p = Plot(long_df, "x").add(MockMark(), Identity()).scale(y="log").plot()
        # 获取图表的第一个坐标轴对象
        ax = p._figure.axes[0]
        # 获取 y 轴的转换函数，并转换指定数据
        xfm = ax.yaxis.get_transform().transform
        # 断言转换后的数据与期望结果相等
        assert_array_equal(xfm([1, 10, 100]), [0, 1, 2])

    # 测试用例：验证在显式指定范围和轴缩放的情况下，绘制图表是否正确
    def test_explicit_range_with_axis_scaling(self):
        # 设定 x、y 最小和最大值
        x = [1, 2, 3]
        ymin = [10, 100, 1000]
        ymax = [20, 200, 2000]
        # 创建 MockMark 对象
        m = MockMark()
        # 使用 Plot 类绘制以 x 为横轴，ymin 和 ymax 为纵轴范围的图表，添加 MockMark 对象后设定 y 轴为对数缩放，并绘制图表
        Plot(x=x, ymin=ymin, ymax=ymax).add(m).scale(y="log").plot()
        # 断言 MockMark 对象接收的数据中的 ymax 列等于设定的 ymax 列数据
        assert_vector_equal(m.passed_data[0]["ymax"], pd.Series(ymax, dtype=float))
    def test_derived_range_with_axis_scaling(self):
        # 定义一个名为 AddOne 的内部类，继承自 Stat 类
        class AddOne(Stat):
            # 重载 __call__ 方法，接受 DataFrame 和可变参数 args
            def __call__(self, df, *args):
                # 返回对 df 的新赋值，增加了名为 ymax 的列，其值为 df["y"] + 1
                return df.assign(ymax=df["y"] + 1)

        # 初始化 x 和 y 列表，分别赋值为 [1, 10, 100]
        x = y = [1, 10, 100]

        # 创建一个 MockMark 对象 m
        m = MockMark()
        # 创建一个 Plot 对象，传入 x 和 y 列表，添加 MockMark 对象 m 和 AddOne 实例，设置 y 轴为对数尺度，然后绘图
        Plot(x, y).add(m, AddOne()).scale(y="log").plot()
        # 断言 MockMark 对象 m 的传递数据中的 ymax 列与预期的 pd.Series([10., 100., 1000.]) 相等
        assert_vector_equal(m.passed_data[0]["ymax"], pd.Series([10., 100., 1000.]))

    def test_facet_categories(self):
        # 创建一个 MockMark 对象 m
        m = MockMark()
        # 创建一个 Plot 对象，传入 x 列为 ["a", "b", "a", "c"]，设置列分面为 ["x", "x", "y", "y"]，添加 MockMark 对象 m，然后绘图
        p = Plot(x=["a", "b", "a", "c"]).facet(col=["x", "x", "y", "y"]).add(m).plot()
        # 获取绘图对象 p 的子图 axes，期望有两个子图
        ax1, ax2 = p._figure.axes
        # 断言 ax1 和 ax2 的 x 轴刻度数量为 3
        assert len(ax1.get_xticks()) == 3
        assert len(ax2.get_xticks()) == 3
        # 断言 MockMark 对象 m 的传递数据中的 x 列分别与预期的 pd.Series([0., 1.], [0, 1]) 和 pd.Series([0., 2.], [2, 3]) 相等
        assert_vector_equal(m.passed_data[0]["x"], pd.Series([0., 1.], [0, 1]))
        assert_vector_equal(m.passed_data[1]["x"], pd.Series([0., 2.], [2, 3]))

    def test_facet_categories_unshared(self):
        # 创建一个 MockMark 对象 m
        m = MockMark()
        # 创建一个 Plot 对象，传入 x 列为 ["a", "b", "a", "c"]，设置列分面为 ["x", "x", "y", "y"]，设置不共享 x 轴，添加 MockMark 对象 m，然后绘图
        p = (
            Plot(x=["a", "b", "a", "c"])
            .facet(col=["x", "x", "y", "y"])
            .share(x=False)
            .add(m)
            .plot()
        )
        # 获取绘图对象 p 的子图 axes，期望有两个子图
        ax1, ax2 = p._figure.axes
        # 断言 ax1 和 ax2 的 x 轴刻度数量为 2
        assert len(ax1.get_xticks()) == 2
        assert len(ax2.get_xticks()) == 2
        # 断言 MockMark 对象 m 的传递数据中的 x 列分别与预期的 pd.Series([0., 1.], [0, 1]) 和 pd.Series([0., 1.], [2, 3]) 相等
        assert_vector_equal(m.passed_data[0]["x"], pd.Series([0., 1.], [0, 1]))
        assert_vector_equal(m.passed_data[1]["x"], pd.Series([0., 1.], [2, 3]))

    def test_facet_categories_single_dim_shared(self):
        # 创建包含元组数据的列表 data
        data = [
            ("a", 1, 1), ("b", 1, 1),
            ("a", 1, 2), ("c", 1, 2),
            ("b", 2, 1), ("d", 2, 1),
            ("e", 2, 2), ("e", 2, 1),
        ]
        # 创建 DataFrame df，列名为 ["x", "row", "col"]，并添加 y 列，所有行赋值为 1
        df = pd.DataFrame(data, columns=["x", "row", "col"]).assign(y=1)
        # 创建一个 MockMark 对象 m
        m = MockMark()
        # 创建一个 Plot 对象，传入 DataFrame df 和 x 列名 "x"，设置行和列分面为 "row" 和 "col"，添加 MockMark 对象 m，设置共享 "row" 的 x 轴，然后绘图
        p = (
            Plot(df, x="x")
            .facet(row="row", col="col")
            .add(m)
            .share(x="row")
            .plot()
        )

        # 获取绘图对象 p 的所有子图 axes
        axs = p._figure.axes
        # 遍历每个子图 ax
        for ax in axs:
            # 断言每个 ax 的 x 轴刻度为 [0, 1, 2]
            assert ax.get_xticks() == [0, 1, 2]

        # 断言 MockMark 对象 m 的传递数据中的 x 列分别与预期的 pd.Series([0., 1.], [0, 1])、pd.Series([0., 2.], [2, 3])、pd.Series([0., 1., 2.], [4, 5, 7])、pd.Series([2.], [6]) 相等
        assert_vector_equal(m.passed_data[0]["x"], pd.Series([0., 1.], [0, 1]))
        assert_vector_equal(m.passed_data[1]["x"], pd.Series([0., 2.], [2, 3]))
        assert_vector_equal(m.passed_data[2]["x"], pd.Series([0., 1., 2.], [4, 5, 7]))
        assert_vector_equal(m.passed_data[3]["x"], pd.Series([2.], [6]))

    def test_pair_categories(self):
        # 创建包含元组数据的列表 data
        data = [("a", "a"), ("b", "c")]
        # 创建 DataFrame df，列名为 ["x1", "x2"]，并添加 y 列，所有行赋值为 1
        df = pd.DataFrame(data, columns=["x1", "x2"]).assign(y=1)
        # 创建一个 MockMark 对象 m
        m = MockMark()
        # 创建一个 Plot 对象，传入 DataFrame df 和 y 列名 "y"，进行 x1 和 x2 的配对绘图，添加 MockMark 对象 m，然后绘图
        p = Plot(df, y="y").pair(x=["x1", "x2"]).add(m).plot()

        # 获取绘图对象 p 的两个子图 axes
        ax1, ax2 = p._figure.axes
        # 断言 ax1 和 ax2 的 x 轴刻度为 [0, 1]
        assert ax1.get_xticks() == [0, 1]
        assert ax2.get_xticks() == [0, 1]
        # 断言 MockMark 对象 m 的传递数据中的 x 列分别与预期的 pd.Series([0., 1.], [0, 1]) 和 pd.Series([0., 1.], [0, 1]) 相等
        assert_vector_equal(m.passed_data[0]["x"], pd.Series([0., 1.], [0, 1]))
        assert_vector_equal(m.passed_data[1]["x"], pd.Series([0., 1.], [0, 1]))
    # 测试函数：test_pair_categories_shared，测试绘图对象的共享特性
    def test_pair_categories_shared(self):
        # 定义测试数据
        data = [("a", "a"), ("b", "c")]
        # 创建包含特定列和新列 'y' 的数据框
        df = pd.DataFrame(data, columns=["x1", "x2"]).assign(y=1)
        # 创建 MockMark 实例
        m = MockMark()
        # 创建 Plot 对象，设置 y 列为因变量，对 x1 和 x2 列创建散点图，并添加 MockMark 实例，启用 x 轴共享
        p = Plot(df, y="y").pair(x=["x1", "x2"]).add(m).share(x=True).plot()

        # 遍历绘图对象的所有子图
        for ax in p._figure.axes:
            # 断言 x 轴刻度应为 [0, 1, 2]
            assert ax.get_xticks() == [0, 1, 2]
        # 打印 MockMark 实例的传递数据
        print(m.passed_data)
        # 断言 MockMark 实例传递的数据中第一个 "x" 键对应的 Series 对象是否为预期值
        assert_vector_equal(m.passed_data[0]["x"], pd.Series([0., 1.], [0, 1]))
        # 断言 MockMark 实例传递的数据中第二个 "x" 键对应的 Series 对象是否为预期值
        assert_vector_equal(m.passed_data[1]["x"], pd.Series([0., 2.], [0, 1]))

    # 测试函数：test_identity_mapping_linewidth，测试线条宽度的身份映射
    def test_identity_mapping_linewidth(self):

        # 创建 MockMark 实例
        m = MockMark()
        # 定义 x 和 y 数据
        x = y = [1, 2, 3, 4, 5]
        # 定义线条宽度 Series 对象
        lw = pd.Series([.5, .1, .1, .9, 3])
        # 创建 Plot 对象，指定 x 和 y 数据，线条宽度为 lw，禁用线条宽度缩放
        Plot(x=x, y=y, linewidth=lw).scale(linewidth=None).add(m).plot()
        # 断言 MockMark 实例中传递的线条宽度映射是否与原始线条宽度数据相等
        assert_vector_equal(m.passed_scales["linewidth"](lw), lw)

    # 测试函数：test_pair_single_coordinate_stat_orient，测试散点图单坐标统计方向
    def test_pair_single_coordinate_stat_orient(self, long_df):

        # 定义 MockStat 类，继承自 Stat 类
        class MockStat(Stat):
            # 重载 __call__ 方法，处理数据、分组、方向和比例尺
            def __call__(self, data, groupby, orient, scales):
                # 记录传入的方向参数
                self.orient = orient
                return data

        # 创建 MockStat 实例
        s = MockStat()
        # 创建 Plot 对象，使用长格式数据框 long_df，对 x 和 y 列创建散点图，并添加 MockMark 实例和 MockStat 实例
        Plot(long_df).pair(x=["x", "y"]).add(MockMark(), s).plot()
        # 断言 MockStat 实例中记录的方向参数是否为 "x"
        assert s.orient == "x"

    # 测试函数：test_inferred_nominal_passed_to_stat，测试推断的名义变量传递给统计对象
    def test_inferred_nominal_passed_to_stat(self):

        # 定义 MockStat 类，继承自 Stat 类
        class MockStat(Stat):
            # 重载 __call__ 方法，处理数据、分组、方向和比例尺
            def __call__(self, data, groupby, orient, scales):
                # 记录传入的比例尺参数
                self.scales = scales
                return data

        # 创建 MockStat 实例
        s = MockStat()
        # 定义 y 数据
        y = ["a", "a", "b", "c"]
        # 创建 Plot 对象，使用 y 数据创建绘图，并添加 MockMark 实例和 MockStat 实例
        Plot(y=y).add(MockMark(), s).plot()
        # 断言 MockStat 实例中记录的比例尺参数中 "y" 键对应的类型是否为 "Nominal"
        assert s.scales["y"].__class__.__name__ == "Nominal"

    # 测试函数：test_identity_mapping_color_strings，测试颜色字符串的身份映射
    @pytest.mark.xfail(
        reason="Correct output representation for color with identity scale undefined"
    )
    def test_identity_mapping_color_strings(self):

        # 创建 MockMark 实例
        m = MockMark()
        # 定义 x 和 y 数据
        x = y = [1, 2, 3]
        # 定义颜色字符串列表
        c = ["C0", "C2", "C1"]
        # 创建 Plot 对象，指定 x 和 y 数据，颜色为 c，禁用颜色缩放
        Plot(x=x, y=y, color=c).scale(color=None).add(m).plot()
        # 期望的颜色值数组
        expected = mpl.colors.to_rgba_array(c)[:, :3]
        # 断言 MockMark 实例中传递的颜色映射是否与期望的颜色值数组相等
        assert_array_equal(m.passed_scales["color"](c), expected)

    # 测试函数：test_identity_mapping_color_tuples，测试颜色元组的身份映射
    def test_identity_mapping_color_tuples(self):

        # 创建 MockMark 实例
        m = MockMark()
        # 定义 x 和 y 数据
        x = y = [1, 2, 3]
        # 定义颜色元组列表
        c = [(1, 0, 0), (0, 1, 0), (1, 0, 0)]
        # 创建 Plot 对象，指定 x 和 y 数据，颜色为 c，禁用颜色缩放
        Plot(x=x, y=y, color=c).scale(color=None).add(m).plot()
        # 期望的颜色值数组
        expected = mpl.colors.to_rgba_array(c)[:, :3]
        # 断言 MockMark 实例中传递的颜色映射是否与期望的颜色值数组相等
        assert_array_equal(m.passed_scales["color"](c), expected)

    # 测试函数：test_undefined_variable_raises，测试未定义变量引发异常
    @pytest.mark.xfail(
        reason="Need decision on what to do with scale defined for unused variable"
    )
    def test_undefined_variable_raises(self):

        # 创建 Plot 对象，指定 x 和 color 数据，设置 y 变量为连续比例尺
        p = Plot(x=[1, 2, 3], color=["a", "b", "c"]).scale(y=Continuous())
        # 定义预期的异常信息
        err = r"No data found for variable\(s\) with explicit scale: {'y'}"
        # 断言调用 plot() 方法时抛出 RuntimeError 异常，并且异常信息与预期信息匹配
        with pytest.raises(RuntimeError, match=err):
            p.plot()
    # 定义测试函数，用于测试在修改 x 轴参数时的预期行为
    def test_nominal_x_axis_tweaks(self):

        # 创建 Plot 对象，指定 x 和 y 轴数据
        p = Plot(x=["a", "b", "c"], y=[1, 2, 3])
        # 绘制图形并获取第一个轴对象
        ax1 = p.plot()._figure.axes[0]
        # 断言当前轴的 x 轴限制是否为 (-0.5, 2.5)
        assert ax1.get_xlim() == (-.5, 2.5)
        # 断言 x 轴网格线是否全部不可见
        assert not any(x.get_visible() for x in ax1.xaxis.get_gridlines())

        # 设定新的 x 轴限制
        lim = (-1, 2.1)
        # 对 Plot 对象进行 x 轴限制操作，并绘制图形，获取新的轴对象
        ax2 = p.limit(x=lim).plot()._figure.axes[0]
        # 断言新轴对象的 x 轴限制是否与设定的 lim 相同
        assert ax2.get_xlim() == lim

    # 定义测试函数，用于测试在修改 y 轴参数时的预期行为
    def test_nominal_y_axis_tweaks(self):

        # 创建 Plot 对象，指定 x 和 y 轴数据
        p = Plot(x=[1, 2, 3], y=["a", "b", "c"])
        # 绘制图形并获取第一个轴对象
        ax1 = p.plot()._figure.axes[0]
        # 断言当前轴的 y 轴限制是否为 (2.5, -0.5)
        assert ax1.get_ylim() == (2.5, -.5)
        # 断言 y 轴网格线是否全部不可见
        assert not any(y.get_visible() for y in ax1.yaxis.get_gridlines())

        # 设定新的 y 轴限制
        lim = (-1, 2.1)
        # 对 Plot 对象进行 y 轴限制操作，并绘制图形，获取新的轴对象
        ax2 = p.limit(y=lim).plot()._figure.axes[0]
        # 断言新轴对象的 y 轴限制是否与设定的 lim 相同
        assert ax2.get_ylim() == lim
# 定义一个测试类 TestPlotting，用于测试绘图功能
class TestPlotting:

    # 测试 Matplotlib 对象的创建
    def test_matplotlib_object_creation(self):
        # 创建 Plot 实例并绘图，返回结果给 p
        p = Plot().plot()
        # 断言 p._figure 是 mpl.figure.Figure 的实例
        assert isinstance(p._figure, mpl.figure.Figure)
        # 对于 p._subplots 中的每个子图，断言其 "ax" 键对应的值是 mpl.axes.Axes 的实例
        for sub in p._subplots:
            assert isinstance(sub["ax"], mpl.axes.Axes)

    # 测试添加空 MockMark 的情况
    def test_empty(self):
        # 创建 MockMark 实例 m
        m = MockMark()
        # 创建 Plot 实例并将 m 添加进去后绘图
        Plot().add(m).plot()
        # 断言 m.n_splits 等于 0
        assert m.n_splits == 0
        # 断言 m.passed_data 为假值（通常是空列表或 None）
        assert not m.passed_data

    # 测试无方向变化的情况
    def test_no_orient_variance(self):
        # 定义示例数据 x, y
        x, y = [0, 0], [1, 2]
        # 创建 MockMark 实例 m
        m = MockMark()
        # 创建 Plot 实例并传入 x, y 数据，并将 m 添加进去后绘图
        Plot(x, y).add(m).plot()
        # 断言 m.passed_data[0]["x"] 等于 x
        assert_array_equal(m.passed_data[0]["x"], x)
        # 断言 m.passed_data[0]["y"] 等于 y
        assert_array_equal(m.passed_data[0]["y"], y)

    # 测试单一分割和单层的情况
    def test_single_split_single_layer(self, long_df):
        # 创建 MockMark 实例 m
        m = MockMark()
        # 创建 Plot 实例并传入 long_df 数据，指定 x="f", y="z"，并将 m 添加进去后绘图
        p = Plot(long_df, x="f", y="z").add(m).plot()
        # 断言 m.n_splits 等于 1
        assert m.n_splits == 1
        # 断言 m.passed_keys[0] 等于空字典
        assert m.passed_keys[0] == {}
        # 断言 m.passed_axes 等于 p._subplots 中每个子图的 "ax" 键对应的值
        assert m.passed_axes == [sub["ax"] for sub in p._subplots]
        # 对于 p._data.frame 中的每列 col，断言 m.passed_data[0][col] 等于 p._data.frame[col]
        for col in p._data.frame:
            assert_series_equal(m.passed_data[0][col], p._data.frame[col])

    # 测试单一分割和多层的情况
    def test_single_split_multi_layer(self, long_df):
        # 定义多个绘图属性字典 vs
        vs = [{"color": "a", "linewidth": "z"}, {"color": "b", "pattern": "c"}]

        # 定义一个不分组的 MockMark 子类 NoGroupingMark
        class NoGroupingMark(MockMark):
            _grouping_props = []

        # 创建两个 NoGroupingMark 实例 ms
        ms = [NoGroupingMark(), NoGroupingMark()]
        # 创建 Plot 实例并传入 long_df 数据，依次将 ms[0] 和 ms[1] 添加进去，使用相应的 vs 属性字典后绘图
        Plot(long_df).add(ms[0], **vs[0]).add(ms[1], **vs[1]).plot()

        # 对每个 m, v 对，断言 m.passed_data[0][var] 等于 long_df[col]
        for m, v in zip(ms, vs):
            for var, col in v.items():
                assert_vector_equal(m.passed_data[0][var], long_df[col])

    # 检查单变量分割的情况
    def check_splits_single_var(
        self, data, mark, data_vars, split_var, split_col, split_keys
    ):
        # 断言 mark.n_splits 等于 split_keys 的长度
        assert mark.n_splits == len(split_keys)
        # 断言 mark.passed_keys 等于 [{split_var: key} for key in split_keys]
        assert mark.passed_keys == [{split_var: key} for key in split_keys]

        # 对于每个 split_keys 中的索引 i 和键 key，分别检查数据
        for i, key in enumerate(split_keys):
            # 从 data 中选取 split_col 列值为 key 的数据
            split_data = data[data[split_col] == key]
            # 对每个 data_vars 中的变量 var 和列 col，断言 mark.passed_data[i][var] 等于 split_data[col]
            for var, col in data_vars.items():
                assert_array_equal(mark.passed_data[i][var], split_data[col])

    # 检查多变量分割的情况
    def check_splits_multi_vars(
        self, data, mark, data_vars, split_vars, split_cols, split_keys
    ):
        # 断言 mark.n_splits 等于 split_keys 各维度长度的乘积
        assert mark.n_splits == np.prod([len(ks) for ks in split_keys])

        # 生成预期的分割键组合 expected_keys
        expected_keys = [
            dict(zip(split_vars, level_keys))
            for level_keys in itertools.product(*split_keys)
        ]
        # 断言 mark.passed_keys 等于 expected_keys
        assert mark.passed_keys == expected_keys

        # 对于每个 split_keys 中的组合 keys，分别检查数据
        for i, keys in enumerate(itertools.product(*split_keys)):
            # 初始化使用行 use_rows
            use_rows = pd.Series(True, data.index)
            # 对于每个 split_vars 中的变量 var, col 和键 key，更新 use_rows
            for var, col, key in zip(split_vars, split_cols, keys):
                use_rows &= data[col] == key
            # 从 data 中选取符合条件的数据 split_data
            split_data = data[use_rows]
            # 对每个 data_vars 中的变量 var 和列 col，断言 mark.passed_data[i][var] 等于 split_data[col]
            for var, col in data_vars.items():
                assert_array_equal(mark.passed_data[i][var], split_data[col])

    # 使用 pytest.mark.parametrize 注入的参数化测试，测试 split_var 为 "color" 和 "group" 两种情况
    @pytest.mark.parametrize(
        "split_var", [
            "color",  # 明确声明在 Mark 类上的属性
            "group",  # 隐式用于所有 Mark 类的属性
        ])
    # 测试函数，用于单个变量分组情况的测试
    def test_one_grouping_variable(self, long_df, split_var):
        
        # 定义分割列名
        split_col = "a"
        # 定义数据变量字典，包括固定的 "x", "y" 和传入的分割变量及其对应的列名
        data_vars = {"x": "f", "y": "z", split_var: split_col}
        
        # 创建 MockMark 实例
        m = MockMark()
        # 创建 Plot 实例，并根据给定数据变量添加 MockMark，然后绘制图形
        p = Plot(long_df, **data_vars).add(m).plot()
        
        # 获取长数据框中分割列的分类顺序
        split_keys = categorical_order(long_df[split_col])
        # 获取绘图对象的子图列表
        sub, *_ = p._subplots
        # 断言 MockMark 实例的传递轴是否与分割键列表中每个键对应的子图轴一致
        assert m.passed_axes == [sub["ax"] for _ in split_keys]
        # 调用检查单变量分割的方法，验证结果
        self.check_splits_single_var(
            long_df, m, data_vars, split_var, split_col, split_keys
        )
    
    # 测试函数，用于两个变量分组情况的测试
    def test_two_grouping_variables(self, long_df):
        
        # 定义多个分割变量和对应的列名
        split_vars = ["color", "group"]
        split_cols = ["a", "b"]
        # 定义数据变量字典，包括固定的 "y" 和从分割变量和列名列表中生成的键值对
        data_vars = {"y": "z", **{var: col for var, col in zip(split_vars, split_cols)}}
        
        # 创建 MockMark 实例
        m = MockMark()
        # 创建 Plot 实例，并根据给定数据变量添加 MockMark，然后绘制图形
        p = Plot(long_df, **data_vars).add(m).plot()
        
        # 获取长数据框中每个分割列的分类顺序列表
        split_keys = [categorical_order(long_df[col]) for col in split_cols]
        # 获取绘图对象的子图列表
        sub, *_ = p._subplots
        # 断言 MockMark 实例的传递轴是否与 itertools.product(*split_keys) 的每个子组合对应的子图轴一致
        assert m.passed_axes == [
            sub["ax"] for _ in itertools.product(*split_keys)
        ]
        # 调用检查多变量分割的方法，验证结果
        self.check_splits_multi_vars(
            long_df, m, data_vars, split_vars, split_cols, split_keys
        )
    
    # 测试函数，用于指定宽度情况的测试
    def test_specified_width(self, long_df):
        
        # 创建 MockMark 实例
        m = MockMark()
        # 创建 Plot 实例，指定 x 和 y 列，并添加 MockMark，并指定宽度列 "z"，然后绘制图形
        Plot(long_df, x="x", y="y").add(m, width="z").plot()
        # 断言传递给 MockMark 实例的数据的 "width" 列是否与长数据框中的 "z" 列近似相等
        assert_array_almost_equal(m.passed_data[0]["width"], long_df["z"])
    
    # 测试函数，用于没有子组情况的图面板测试
    def test_facets_no_subgroups(self, long_df):
        
        # 定义分面变量和列名
        split_var = "col"
        split_col = "b"
        # 定义数据变量字典，包括固定的 "x", "y" 和分面变量及其对应的列名
        data_vars = {"x": "f", "y": "z"}
        
        # 创建 MockMark 实例
        m = MockMark()
        # 创建 Plot 实例，根据给定数据变量和分面变量参数分面，并添加 MockMark，然后绘制图形
        p = Plot(long_df, **data_vars).facet(**{split_var: split_col}).add(m).plot()
        
        # 获取长数据框中分面列的分类顺序
        split_keys = categorical_order(long_df[split_col])
        # 断言 MockMark 实例的传递轴是否与图面板对象的轴列表一致
        assert m.passed_axes == list(p._figure.axes)
        # 调用检查单变量分面的方法，验证结果
        self.check_splits_single_var(
            long_df, m, data_vars, split_var, split_col, split_keys
        )
    
    # 测试函数，用于一个子组情况的图面板测试
    def test_facets_one_subgroup(self, long_df):
        
        # 定义分面变量及其列名，以及组变量及其列名
        facet_var, facet_col = fx = "col", "a"
        group_var, group_col = gx = "group", "b"
        split_vars, split_cols = zip(*[fx, gx])
        # 定义数据变量字典，包括固定的 "x", "y" 和组变量及其列名
        data_vars = {"x": "f", "y": "z", group_var: group_col}
        
        # 创建 MockMark 实例
        m = MockMark()
        # 创建 Plot 实例，根据给定数据变量和分面变量参数分面，并添加 MockMark，然后绘制图形
        p = (
            Plot(long_df, **data_vars)
            .facet(**{facet_var: facet_col})
            .add(m)
            .plot()
        )
        
        # 获取长数据框中分面列和组列的分类顺序列表
        split_keys = [categorical_order(long_df[col]) for col in [facet_col, group_col]]
        # 断言 MockMark 实例的传递轴是否与图面板对象的轴列表一致，且每个组列分类顺序的子轴都对应
        assert m.passed_axes == [
            ax
            for ax in list(p._figure.axes)
            for _ in categorical_order(long_df[group_col])
        ]
        # 调用检查多变量分面的方法，验证结果
        self.check_splits_multi_vars(
            long_df, m, data_vars, split_vars, split_cols, split_keys
        )
    # 测试特定图层的禁用功能
    def test_layer_specific_facet_disabling(self, long_df):
        # 定义轴变量映射关系
        axis_vars = {"x": "y", "y": "z"}
        # 定义行变量
        row_var = "a"

        # 创建 MockMark 对象
        m = MockMark()
        # 创建 Plot 对象并设置轴变量映射关系和行变量，并添加 MockMark 对象，生成图表
        p = Plot(long_df, **axis_vars).facet(row=row_var).add(m, row=None).plot()

        # 获取行变量列的分类顺序
        col_levels = categorical_order(long_df[row_var])
        # 断言生成的图表对象的轴数量等于行变量列的分类数目
        assert len(p._figure.axes) == len(col_levels)

        # 遍历 MockMark 对象传递的数据
        for data in m.passed_data:
            # 遍历轴变量映射关系
            for var, col in axis_vars.items():
                # 断言 data 中的变量数据与 long_df 中对应列数据相等
                assert_vector_equal(data[var], long_df[col])

    # 测试成对变量
    def test_paired_variables(self, long_df):
        # 定义 x 和 y 的变量组合
        x = ["x", "y"]
        y = ["f", "z"]

        # 创建 MockMark 对象
        m = MockMark()
        # 创建 Plot 对象并设置数据源，并将变量 x 和 y 成对，添加 MockMark 对象，生成图表
        Plot(long_df).pair(x, y).add(m).plot()

        # 生成 x 和 y 的变量组合迭代器
        var_product = itertools.product(x, y)

        # 遍历 MockMark 对象传递的数据和变量组合
        for data, (x_i, y_i) in zip(m.passed_data, var_product):
            # 断言 data 中 x 变量数据与 long_df 中对应列数据相等，转换为 float 类型
            assert_vector_equal(data["x"], long_df[x_i].astype(float))
            # 断言 data 中 y 变量数据与 long_df 中对应列数据相等，转换为 float 类型
            assert_vector_equal(data["y"], long_df[y_i].astype(float))

    # 测试单维度的成对变量
    def test_paired_one_dimension(self, long_df):
        # 定义 x 的变量组合
        x = ["y", "z"]

        # 创建 MockMark 对象
        m = MockMark()
        # 创建 Plot 对象并设置数据源，并将变量 x 成对，添加 MockMark 对象，生成图表
        Plot(long_df).pair(x).add(m).plot()

        # 遍历 MockMark 对象传递的数据和 x 变量组合
        for data, x_i in zip(m.passed_data, x):
            # 断言 data 中 x 变量数据与 long_df 中对应列数据相等，转换为 float 类型
            assert_vector_equal(data["x"], long_df[x_i].astype(float))

    # 测试带有子集的成对变量
    def test_paired_variables_one_subset(self, long_df):
        # 定义 x 和 y 的变量组合以及分组变量
        x = ["x", "y"]
        y = ["f", "z"]
        group = "a"

        # 简化 long_df 中 "x" 列的向量比较
        long_df["x"] = long_df["x"].astype(float)

        # 创建 MockMark 对象
        m = MockMark()
        # 创建 Plot 对象并设置数据源和分组变量，并将变量 x 和 y 成对，添加 MockMark 对象，生成图表
        Plot(long_df, group=group).pair(x, y).add(m).plot()

        # 获取分组变量的分类顺序
        groups = categorical_order(long_df[group])
        # 生成 x、y 和 groups 的变量组合迭代器
        var_product = itertools.product(x, y, groups)

        # 遍历 MockMark 对象传递的数据和变量组合
        for data, (x_i, y_i, g_i) in zip(m.passed_data, var_product):
            # 获取符合当前组合的行索引
            rows = long_df[group] == g_i
            # 断言 data 中 x 变量数据与 long_df 中对应列数据相等，转换为 float 类型
            assert_vector_equal(data["x"], long_df.loc[rows, x_i])
            # 断言 data 中 y 变量数据与 long_df 中对应列数据相等，转换为 float 类型
            assert_vector_equal(data["y"], long_df.loc[rows, y_i])

    # 测试成对变量和分面效果
    def test_paired_and_faceted(self, long_df):
        # 定义 x 的变量组合和分面变量
        x = ["y", "z"]
        y = "f"
        row = "c"

        # 创建 MockMark 对象
        m = MockMark()
        # 创建 Plot 对象并设置数据源和 y 变量，并将 x 变量成对，添加 MockMark 对象，生成图表并分面
        Plot(long_df, y=y).facet(row=row).pair(x).add(m).plot()

        # 获取分面变量的分类顺序
        facets = categorical_order(long_df[row])
        # 生成 x 和 facets 的变量组合迭代器
        var_product = itertools.product(x, facets)

        # 遍历 MockMark 对象传递的数据和变量组合
        for data, (x_i, f_i) in zip(m.passed_data, var_product):
            # 获取符合当前组合的行索引
            rows = long_df[row] == f_i
            # 断言 data 中 x 变量数据与 long_df 中对应列数据相等
            assert_vector_equal(data["x"], long_df.loc[rows, x_i])
            # 断言 data 中 y 变量数据与预定义的 y 值相等
            assert_vector_equal(data["y"], long_df.loc[rows, y])

    # 测试默认主题设置
    def test_theme_default(self):
        # 创建 Plot 对象并生成图表
        p = Plot().plot()
        # 断言图表第一个轴的背景颜色与预定义颜色值相同
        assert mpl.colors.same_color(p._figure.axes[0].get_facecolor(), "#EAEAF2")

    # 测试自定义主题参数设置
    def test_theme_params(self):
        # 定义自定义的颜色值
        color = ".888"
        # 创建 Plot 对象并设置自定义主题参数，生成图表
        p = Plot().theme({"axes.facecolor": color}).plot()
        # 断言图表第一个轴的背景颜色与自定义颜色值相同
        assert mpl.colors.same_color(p._figure.axes[0].get_facecolor(), color)

    # 测试主题设置错误的情况
    def test_theme_error(self):
        # 创建 Plot 对象
        p = Plot()
        # 使用 pytest 检查调用 theme() 方法时参数个数不符合预期是否会抛出 TypeError 异常
        with pytest.raises(TypeError, match=r"theme\(\) takes 2 positional"):
            p.theme("arg1", "arg2")
    # 测试主题验证功能
    def test_theme_validation(self):
        
        p = Plot()
        # 使用 pytest 检查是否引发 ValueError 异常，并检查异常消息中是否包含特定字符串
        with pytest.raises(ValueError, match="Key axes.linewidth:"):
            p.theme({"axes.linewidth": "thick"})

        # 使用 pytest 检查是否引发 KeyError 异常，并检查异常消息中是否包含特定字符串
        with pytest.raises(KeyError, match="not.a.key is not a valid rc"):
            p.theme({"not.a.key": True})

    # 测试统计功能
    def test_stat(self, long_df):

        # 备份原始数据框
        orig_df = long_df.copy(deep=True)

        # 创建 MockMark 实例
        m = MockMark()
        # 创建 Plot 实例，使用长数据框作为数据源，指定 x 和 y 列，添加 MockMark 实例和 Agg 统计操作，然后绘图
        Plot(long_df, x="a", y="z").add(m, Agg()).plot()

        # 预期结果是对长数据框按 "a" 列分组后的 "z" 列的平均值，并重置索引
        expected = long_df.groupby("a", sort=False)["z"].mean().reset_index(drop=True)
        # 断言 MockMark 实例接收到的数据与预期数据相等
        assert_vector_equal(m.passed_data[0]["y"], expected)

        # 断言长数据框未被修改
        assert_frame_equal(long_df, orig_df)   # Test data was not mutated

    # 测试移动功能
    def test_move(self, long_df):

        # 备份原始数据框
        orig_df = long_df.copy(deep=True)

        # 创建 MockMark 实例
        m = MockMark()
        # 创建 Plot 实例，使用长数据框作为数据源，指定 x 和 y 列，添加 MockMark 实例和 Shift 移动操作，然后绘图
        Plot(long_df, x="z", y="z").add(m, Shift(x=1)).plot()
        # 断言 MockMark 实例接收到的 "x" 列数据与长数据框 "z" 列数据加 1 后相等
        assert_vector_equal(m.passed_data[0]["x"], long_df["z"] + 1)
        # 断言 MockMark 实例接收到的 "y" 列数据与长数据框 "z" 列数据相等
        assert_vector_equal(m.passed_data[0]["y"], long_df["z"])

        # 断言长数据框未被修改
        assert_frame_equal(long_df, orig_df)   # Test data was not mutated

    # 测试统计和移动功能结合
    def test_stat_and_move(self, long_df):

        # 创建 MockMark 实例
        m = MockMark()
        # 创建 Plot 实例，使用长数据框作为数据源，指定 x 和 y 列，添加 MockMark 实例、Agg 统计操作和 Shift 移动操作，然后绘图
        Plot(long_df, x="a", y="z").add(m, Agg(), Shift(y=1)).plot()

        # 预期结果是对长数据框按 "a" 列分组后的 "z" 列的平均值，并重置索引，然后加 1
        expected = long_df.groupby("a", sort=False)["z"].mean().reset_index(drop=True)
        # 断言 MockMark 实例接收到的 "y" 列数据与预期数据加 1 后相等
        assert_vector_equal(m.passed_data[0]["y"], expected + 1)

    # 测试统计和对数尺度功能
    def test_stat_log_scale(self, long_df):

        # 备份原始数据框
        orig_df = long_df.copy(deep=True)

        # 创建 MockMark 实例
        m = MockMark()
        # 创建 Plot 实例，使用长数据框作为数据源，指定 x 和 y 列，添加 MockMark 实例和 Agg 统计操作，然后应用对数尺度缩放到 y 轴，最后绘图
        Plot(long_df, x="a", y="z").add(m, Agg()).scale(y="log").plot()

        # 计算长数据框 "a" 列和对数化后的 "z" 列
        x = long_df["a"]
        y = np.log10(long_df["z"])
        # 预期结果是对长数据框按 "a" 列分组后的对数化后的 "z" 列的平均值，并重置索引
        expected = y.groupby(x, sort=False).mean().reset_index(drop=True)
        # 断言 MockMark 实例接收到的 "y" 列数据与预期数据 10 的指数次方后相等
        assert_vector_equal(m.passed_data[0]["y"], 10 ** expected)

        # 断言长数据框未被修改
        assert_frame_equal(long_df, orig_df)   # Test data was not mutated

    # 测试移动和对数尺度功能
    def test_move_log_scale(self, long_df):

        # 创建 MockMark 实例
        m = MockMark()
        # 创建 Plot 实例，使用长数据框作为数据源，指定 x 和 y 列，应用对数尺度缩放到 x 轴，然后添加 MockMark 实例和 Shift 移动操作，最后绘图
        Plot(
            long_df, x="z", y="z"
        ).scale(x="log").add(m, Shift(x=-1)).plot()
        # 断言 MockMark 实例接收到的 "x" 列数据与长数据框 "z" 列数据除以 10 后相等
        assert_vector_equal(m.passed_data[0]["x"], long_df["z"] / 10)

    # 测试多次移动功能
    def test_multi_move(self, long_df):

        # 创建 MockMark 实例
        m = MockMark()
        # 创建移动操作堆栈
        move_stack = [Shift(1), Shift(2)]
        # 创建 Plot 实例，使用长数据框作为数据源，指定 x 和 y 列，然后添加 MockMark 实例和移动操作堆栈中的所有移动操作，最后绘图
        Plot(long_df, x="x", y="y").add(m, *move_stack).plot()
        # 断言 MockMark 实例接收到的 "x" 列数据与长数据框 "x" 列数据加 3 后相等
        assert_vector_equal(m.passed_data[0]["x"], long_df["x"] + 3)

    # 测试多次移动功能和配对
    def test_multi_move_with_pairing(self, long_df):
        # 创建 MockMark 实例
        m = MockMark()
        # 创建移动操作堆栈
        move_stack = [Shift(1), Shift(2)]
        # 创建 Plot 实例，使用长数据框作为数据源，指定 x 列和配对的 y 列集合，然后添加 MockMark 实例和移动操作堆栈中的所有移动操作，最后绘图
        Plot(long_df, x="x").pair(y=["y", "z"]).add(m, *move_stack).plot()
        # 遍历 MockMark 实例接收到的每个数据框
        for frame in m.passed_data:
            # 断言每个数据框的 "x" 列数据与长数据框 "x" 列数据加 3 后相等
            assert_vector_equal(frame["x"], long_df["x"] + 3)
    # 测试移动数据范围的功能
    def test_move_with_range(self, long_df):
        # 创建一个示例数据
        x = [0, 0, 1, 1, 2, 2]
        group = [0, 1, 0, 1, 0, 1]
        ymin = np.arange(6)
        ymax = np.arange(6) * 2

        # 创建一个 MockMark 对象
        m = MockMark()
        # 使用 Plot 类创建一个图，并添加数据和 Dodge 装饰器进行绘制
        Plot(x=x, group=group, ymin=ymin, ymax=ymax).add(m, Dodge()).plot()

        # 定义用于计算偏移的标志
        signs = [-1, +1]
        # 遍历传递给 MockMark 对象的数据中的每个组
        for i, df in m.passed_data[0].groupby("group"):
            # 断言每个组的 x 值与预期的偏移计算结果相等
            assert_array_equal(df["x"], np.arange(3) + signs[i] * 0.2)

    # 测试方法克隆的功能
    def test_methods_clone(self, long_df):
        # 创建一个 Plot 对象 p1，并在其上添加 MockMark 对象，并进行分面处理
        p1 = Plot(long_df, "x", "y")
        p2 = p1.add(MockMark()).facet("a")

        # 断言 p1 和 p2 是不同的对象
        assert p1 is not p2
        # 断言 p1 的图层为空
        assert not p1._layers
        # 断言 p1 的分面规格为空
        assert not p1._facet_spec

    # 测试默认不使用 pyplot 的功能
    def test_default_is_no_pyplot(self):
        # 创建一个 Plot 对象 p，并进行绘图操作
        p = Plot().plot()

        # 断言当前没有任何 pyplot 图形存在
        assert not plt.get_fignums()
        # 断言 p 的 _figure 属性是 mpl.figure.Figure 类的实例
        assert isinstance(p._figure, mpl.figure.Figure)

    # 测试使用 pyplot 的功能
    def test_with_pyplot(self):
        # 创建一个 Plot 对象 p，并使用 pyplot 进行绘图
        p = Plot().plot(pyplot=True)

        # 断言当前有一个 pyplot 图形存在
        assert len(plt.get_fignums()) == 1
        fig = plt.gcf()
        # 断言 p 的 _figure 属性与当前 pyplot 图形是同一个对象
        assert p._figure is fig

    # 测试显示功能
    def test_show(self):
        # 创建一个 Plot 对象 p
        p = Plot()

        # 在捕获警告时尝试显示图形，不阻塞当前进程
        with warnings.catch_warnings(record=True) as msg:
            out = p.show(block=False)
        # 断言 show 方法返回 None
        assert out is None
        # 断言 p 没有 _figure 属性
        assert not hasattr(p, "_figure")

        # 断言当前有一个 pyplot 图形存在
        assert len(plt.get_fignums()) == 1
        fig = plt.gcf()

        # 检查是否使用了 GUI 后端，如果未使用则检查是否收到了警告信息
        gui_backend = (
            fig.canvas.manager.show != mpl.backend_bases.FigureManagerBase.show
        )
        if not gui_backend:
            assert msg

    # 测试保存功能
    def test_save(self):
        # 创建一个字节流缓冲区
        buf = io.BytesIO()

        # 创建一个 Plot 对象 p，并将图形保存到缓冲区
        p = Plot().save(buf)
        # 断言 save 方法返回的对象是 Plot 类的实例
        assert isinstance(p, Plot)
        # 打开保存在字节流缓冲区中的图像，并断言其格式为 PNG
        img = Image.open(buf)
        assert img.format == "PNG"

        # 创建一个字符串流缓冲区
        buf = io.StringIO()
        # 将图形保存为 SVG 格式，并检查保存结果的标签是否为 SVG
        Plot().save(buf, format="svg")
        tag = xml.etree.ElementTree.fromstring(buf.getvalue()).tag
        assert tag == "{http://www.w3.org/2000/svg}svg"

    # 测试布局尺寸功能
    def test_layout_size(self):
        # 定义一个尺寸大小
        size = (4, 2)
        # 创建一个 Plot 对象 p，并设置布局尺寸后进行绘图
        p = Plot().layout(size=size).plot()
        # 断言 p 的图形尺寸与设定的尺寸相等
        assert tuple(p._figure.get_size_inches()) == size

    # 根据 matplotlib 版本选择性跳过测试布局范围功能
    @pytest.mark.skipif(
        _version_predates(mpl, "3.6"),
        reason="mpl<3.6 does not have get_layout_engine",
    )
    def test_layout_extent(self):
        # 创建一个 Plot 对象 p，并设置布局范围后进行绘图
        p = Plot().layout(extent=(.1, .2, .6, 1)).plot()
        # 断言 p 的布局引擎返回的矩形范围与预期值相等
        assert p._figure.get_layout_engine().get()["rect"] == [.1, .2, .5, .8]

    # 根据 matplotlib 版本选择性跳过测试约束布局范围功能
    @pytest.mark.skipif(
        _version_predates(mpl, "3.6"),
        reason="mpl<3.6 does not have get_layout_engine",
    )
    def test_constrained_layout_extent(self):
        # 创建一个 Plot 对象 p，并设置约束布局引擎和布局范围后进行绘图
        p = Plot().layout(engine="constrained", extent=(.1, .2, .6, 1)).plot()
        # 断言 p 的布局引擎返回的矩形范围与预期值相等
        assert p._figure.get_layout_engine().get()["rect"] == [.1, .2, .5, .8]

    # 测试基础布局范围功能
    def test_base_layout_extent(self):
        # 创建一个 Plot 对象 p，并设置基础布局引擎和布局范围后进行绘图
        p = Plot().layout(engine=None, extent=(.1, .2, .6, 1)).plot()
        # 断言 p 的 subplotpars 属性与预期值相等
        assert p._figure.subplotpars.left == 0.1
        assert p._figure.subplotpars.right == 0.6
        assert p._figure.subplotpars.bottom == 0.2
        assert p._figure.subplotpars.top == 1
    # 定义测试方法，用于测试在给定轴上绘图的功能
    def test_on_axes(self):

        # 创建一个空的 Figure 对象，并生成一个子图对象
        ax = mpl.figure.Figure().subplots()

        # 创建一个 MockMark 对象
        m = MockMark()

        # 创建一个 Plot 对象，将数据 [1] 和 [2] 绘制在 ax 上，并添加 MockMark 对象 m
        p = Plot([1], [2]).on(ax).add(m).plot()

        # 断言 MockMark 对象的 passed_axes 属性包含生成的子图对象 ax
        assert m.passed_axes == [ax]

        # 断言 Plot 对象的 _figure 属性引用了相同的 Figure 对象
        assert p._figure is ax.figure

    # 使用 pytest 的参数化装饰器，测试在给定 Figure 对象上绘图的功能
    @pytest.mark.parametrize("facet", [True, False])
    def test_on_figure(self, facet):

        # 创建一个空的 Figure 对象
        f = mpl.figure.Figure()

        # 创建一个 MockMark 对象
        m = MockMark()

        # 创建一个 Plot 对象，将数据 [1, 2] 和 [3, 4] 绘制在 f 上，并添加 MockMark 对象 m
        p = Plot([1, 2], [3, 4]).on(f).add(m)

        # 如果 facet 参数为 True，则进行分面处理
        if facet:
            p = p.facet(["a", "b"])

        # 绘制图形并获取返回的 Plot 对象
        p = p.plot()

        # 断言 MockMark 对象的 passed_axes 属性包含 Figure 对象 f 的所有子图对象
        assert m.passed_axes == f.axes

        # 断言 Plot 对象的 _figure 属性引用了相同的 Figure 对象 f
        assert p._figure is f

    # 使用 pytest 的参数化装饰器，测试在给定子图对象上绘图的功能
    @pytest.mark.parametrize("facet", [True, False])
    def test_on_subfigure(self, facet):

        # 创建两个子图对象，分别属于同一个 Figure 对象
        sf1, sf2 = mpl.figure.Figure().subfigures(2)

        # 在 sf1 上创建一个子图对象
        sf1.subplots()

        # 创建一个 MockMark 对象
        m = MockMark()

        # 创建一个 Plot 对象，将数据 [1, 2] 和 [3, 4] 绘制在 sf2 上，并添加 MockMark 对象 m
        p = Plot([1, 2], [3, 4]).on(sf2).add(m)

        # 如果 facet 参数为 True，则进行分面处理
        if facet:
            p = p.facet(["a", "b"])

        # 绘制图形并获取返回的 Plot 对象
        p = p.plot()

        # 断言 MockMark 对象的 passed_axes 属性包含 sf2.figure 中除了第一个子图对象之外的所有子图对象
        assert m.passed_axes == sf2.figure.axes[1:]

        # 断言 Plot 对象的 _figure 属性引用了相同的 Figure 对象 sf2.figure
        assert p._figure is sf2.figure

    # 测试 Plot 对象的类型检查功能
    def test_on_type_check(self):

        # 创建一个 Plot 对象
        p = Plot()

        # 使用 pytest 的上下文管理器检查调用 on 方法时的类型错误
        with pytest.raises(TypeError, match="The `Plot.on`.+<class 'list'>"):
            p.on([])

    # 测试在子图对象上绘图时的错误处理功能
    def test_on_axes_with_subplots_error(self):

        # 创建一个空的 Figure 对象，并生成一个子图对象
        ax = mpl.figure.Figure().subplots()

        # 创建一个 Plot 对象，并应用 facet 方法在子图对象上创建分面
        p1 = Plot().facet(["a", "b"]).on(ax)

        # 使用 pytest 的上下文管理器检查创建多个子图对象时的运行时错误
        with pytest.raises(RuntimeError, match="Cannot create multiple subplots"):
            p1.plot()

        # 创建一个 Plot 对象，并应用 pair 方法在子图对象上创建配对图
        p2 = Plot().pair([["a", "b"], ["x", "y"]]).on(ax)

        # 使用 pytest 的上下文管理器检查创建多个子图对象时的运行时错误
        with pytest.raises(RuntimeError, match="Cannot create multiple subplots"):
            p2.plot()

    # 使用 pytest 的条件装饰器，根据 matplotlib 版本跳过测试
    @pytest.mark.skipif(
        _version_predates(mpl, "3.6"),
        reason="Requires newer matplotlib layout engine API"
    )
    # 测试使用默认布局算法的功能
    def test_on_layout_algo_default(self):

        # 定义一个 MockEngine 类，模拟 constrained layout engine
        class MockEngine(mpl.layout_engine.ConstrainedLayoutEngine):
            ...

        # 创建一个 Figure 对象，指定布局引擎为 MockEngine 类的实例
        f = mpl.figure.Figure(layout=MockEngine())

        # 创建一个 Plot 对象，并将其绘制在 Figure 对象 f 上
        p = Plot().on(f).plot()

        # 获取 Plot 对象的布局引擎并断言其类名为 "MockEngine"
        layout_engine = p._figure.get_layout_engine()
        assert layout_engine.__class__.__name__ == "MockEngine"

    # 使用 pytest 的条件装饰器，根据 matplotlib 版本跳过测试
    @pytest.mark.skipif(
        _version_predates(mpl, "3.6"),
        reason="Requires newer matplotlib layout engine API"
    )
    # 测试使用指定布局算法的功能
    def test_on_layout_algo_spec(self):

        # 创建一个 Figure 对象，指定布局为 "constrained"
        f = mpl.figure.Figure(layout="constrained")

        # 创建一个 Plot 对象，并设置其布局引擎为 "tight"，然后绘制图形
        p = Plot().on(f).layout(engine="tight").plot()

        # 获取 Plot 对象的布局引擎并断言其类名为 "TightLayoutEngine"
        layout_engine = p._figure.get_layout_engine()
        assert layout_engine.__class__.__name__ == "TightLayoutEngine"

    # 测试从构造函数中设置轴标签的功能
    def test_axis_labels_from_constructor(self, long_df):

        # 创建一个 Plot 对象，并绘制其结果，获取绘制后的轴对象
        ax, = Plot(long_df, x="a", y="b").plot()._figure.axes

        # 断言 x 轴的标签为 "a"
        assert ax.get_xlabel() == "a"

        # 断言 y 轴的标签为 "b"
        assert ax.get_ylabel() == "b"

        # 创建一个 Plot 对象，将 long_df 中的两列数据作为 x 和 y，绘制其结果，获取绘制后的轴对象
        ax, = Plot(x=long_df["a"], y=long_df["b"].to_numpy()).plot()._figure.axes

        # 断言 x 轴的标签为 "a"
        assert ax.get_xlabel() == "a"

        # 断言 y 轴的标签为空字符串
        assert ax.get_ylabel() == ""
    # 测试函数：验证从图层中获取的坐标轴标签是否正确
    def test_axis_labels_from_layer(self, long_df):
        # 创建一个模拟的标记对象
        m = MockMark()

        # 添加数据并绘制图表，获取绘图后的轴对象
        ax, = Plot(long_df).add(m, x="a", y="b").plot()._figure.axes
        # 断言 x 轴标签是否为 "a"
        assert ax.get_xlabel() == "a"
        # 断言 y 轴标签是否为 "b"
        assert ax.get_ylabel() == "b"

        # 创建另一个 Plot 对象，添加数据并绘制图表，获取绘图后的轴对象
        p = Plot().add(m, x=long_df["a"], y=long_df["b"].to_list())
        ax, = p.plot()._figure.axes
        # 断言 x 轴标签是否为 "a"
        assert ax.get_xlabel() == "a"
        # 断言 y 轴标签是否为空字符串
        assert ax.get_ylabel() == ""

    # 测试函数：验证轴标签是否正确设置为第一个名称
    def test_axis_labels_are_first_name(self, long_df):
        # 创建一个模拟的标记对象
        m = MockMark()
        # 创建 Plot 对象，并设置 x 轴和 y 轴的数据，同时添加两个标记对象
        p = (
            Plot(long_df, x=long_df["z"].to_list(), y="b")
            .add(m, x="a")
            .add(m, x="x", y="y")
        )
        # 绘制图表后获取轴对象
        ax, = p.plot()._figure.axes
        # 断言 x 轴标签是否为 "a"
        assert ax.get_xlabel() == "a"
        # 断言 y 轴标签是否为 "b"
        assert ax.get_ylabel() == "b"

    # 测试函数：验证限制范围的设置是否正确
    def test_limits(self, long_df):
        # 设置限制范围
        limit = (-2, 24)
        # 创建 Plot 对象，设置 x 轴和 y 轴的数据，并限制 x 轴范围后绘制图表
        p = Plot(long_df, x="x", y="y").limit(x=limit).plot()
        # 获取绘图后的轴对象
        ax = p._figure.axes[0]
        # 断言 x 轴的限制范围是否与设定值相等
        assert ax.get_xlim() == limit

        # 设置日期类型的限制范围
        limit = (np.datetime64("2005-01-01"), np.datetime64("2008-01-01"))
        # 创建 Plot 对象，设置 x 轴和 y 轴的数据，并限制 x 轴范围为日期类型后绘制图表
        p = Plot(long_df, x="d", y="y").limit(x=limit).plot()
        # 获取绘图后的轴对象
        ax = p._figure.axes[0]
        # 将日期限制范围转换为 Matplotlib 可接受的格式后，断言 x 轴的限制范围是否与设定值相等
        assert ax.get_xlim() == tuple(mpl.dates.date2num(limit))

        # 设置类别类型的限制范围
        limit = ("b", "c")
        # 创建 Plot 对象，设置 x 轴和 y 轴的数据，并限制 x 轴范围为类别类型后绘制图表
        p = Plot(x=["a", "b", "c", "d"], y=[1, 2, 3, 4]).limit(x=limit).plot()
        # 获取绘图后的轴对象
        ax = p._figure.axes[0]
        # 断言 x 轴的限制范围是否与设定值相等
        assert ax.get_xlim() == (0.5, 2.5)

    # 测试函数：验证轴标签的设置是否正确
    def test_labels_axis(self, long_df):
        # 设置 y 轴的标签文本
        label = "Y axis"
        # 创建 Plot 对象，并设置 x 轴和 y 轴的数据，并设置 y 轴的标签文本后绘制图表
        p = Plot(long_df, x="x", y="y").label(y=label).plot()
        # 获取绘图后的轴对象
        ax = p._figure.axes[0]
        # 断言 y 轴的标签文本是否与设定值相等
        assert ax.get_ylabel() == label

        # 设置 y 轴的标签文本为大写处理函数
        label = str.capitalize
        # 创建 Plot 对象，并设置 x 轴和 y 轴的数据，并设置 y 轴的标签文本为大写处理函数后绘制图表
        p = Plot(long_df, x="x", y="y").label(y=label).plot()
        # 获取绘图后的轴对象
        ax = p._figure.axes[0]
        # 断言 y 轴的标签文本是否为 "Y"
        assert ax.get_ylabel() == "Y"

    # 测试函数：验证图例标签的设置是否正确
    def test_labels_legend(self, long_df):
        # 创建一个模拟的标记对象
        m = MockMark()

        # 设置图例标签文本
        label = "A"
        # 创建 Plot 对象，并设置 x 轴和 y 轴的数据，设置颜色后添加标记对象，并设置颜色的图例标签文本后绘制图表
        p = Plot(long_df, x="x", y="y", color="a").add(m).label(color=label).plot()
        # 断言图表的第一个图例标题文本是否与设定值相等
        assert p._figure.legends[0].get_title().get_text() == label

        # 设置图例标签文本为大写处理函数
        func = str.capitalize
        # 创建 Plot 对象，并设置 x 轴和 y 轴的数据，设置颜色后添加标记对象，并设置颜色的图例标签文本为大写处理函数后绘制图表
        p = Plot(long_df, x="x", y="y", color="a").add(m).label(color=func).plot()
        # 断言图表的第一个图例标题文本是否与设定值相等
        assert p._figure.legends[0].get_title().get_text() == label

    # 测试函数：验证图表的分面标签是否正确设置
    def test_labels_facets(self):
        # 设置测试数据
        data = {"a": ["b", "c"], "x": ["y", "z"]}
        # 创建 Plot 对象，并设置数据后添加分面，设置列标签为大写处理函数，行标签为数学符号 "$x$" 后绘制图表
        p = Plot(data).facet("a", "x").label(col=str.capitalize, row="$x$").plot()
        # 获取绘图后的轴对象数组，并重塑为二维数组
        axs = np.reshape(p._figure.axes, (2, 2))
        # 遍历二维数组的每个轴对象，并验证标题文本是否符合预期
        for (i, j), ax in np.ndenumerate(axs):
            expected = f"A {data['a'][j]} | $x$ {data['x'][i]}"
            assert ax.get_title() == expected

    # 测试函数：验证单个图表的标题标签是否正确设置
    def test_title_single(self):
        # 设置标题文本
        label = "A"
        # 创建 Plot 对象，并设置标题文本后绘制图表
        p = Plot().label(title=label).plot()
        # 断言图表的第一个轴对象的标题文本是否与设定值相等
        assert p._figure.axes[0].get_title() == label
    # 定义测试方法：验证标题分面功能
    def test_title_facet_function(self):
        
        # 准备标题列表
        titles = ["a", "b"]
        # 创建 Plot 对象并应用标题分面设置，设定标题格式为首字母大写
        p = Plot().facet(titles).label(title=str.capitalize).plot()
        # 遍历图形对象的所有子图
        for i, ax in enumerate(p._figure.axes):
            # 断言每个子图的标题是否等于对应标题的大写形式
            assert ax.get_title() == titles[i].upper()
        
        # 准备列和行的列表
        cols, rows = ["a", "b"], ["x", "y"]
        # 再次创建 Plot 对象并应用列和行分面设置，设定标题格式为首字母大写
        p = Plot().facet(cols, rows).label(title=str.capitalize).plot()
        # 遍历图形对象的所有子图
        for i, ax in enumerate(p._figure.axes):
            # 生成预期的标题，格式为 "列名 | 行名" 的大写形式
            expected = " | ".join([cols[i % 2].upper(), rows[i // 2].upper()])
            # 断言每个子图的标题是否等于预期的标题
            assert ax.get_title() == expected
class TestExceptions:
    
    # 测试设置异常情况下的比例设置
    def test_scale_setup(self):
        # 初始化变量 x, y, color 为列表 ["a", "b"]
        x = y = color = ["a", "b"]
        # 定义错误的调色板名称
        bad_palette = "not_a_palette"
        # 创建 Plot 对象 p，设置颜色为 color，并尝试使用错误的调色板 bad_palette
        p = Plot(x, y, color=color).add(MockMark()).scale(color=bad_palette)
        
        # 设置错误消息
        msg = "Scale setup failed for the `color` variable."
        # 断言绘图操作会引发 PlotSpecError 异常，并匹配特定消息 msg
        with pytest.raises(PlotSpecError, match=msg) as err:
            p.plot()
        # 断言异常的原因是 ValueError 类型
        assert isinstance(err.value.__cause__, ValueError)
        # 断言异常消息中包含 bad_palette 字符串
        assert bad_palette in str(err.value.__cause__)

    # 测试坐标缩放异常情况
    def test_coordinate_scaling(self):
        # 初始化变量 x 和 y
        x = ["a", "b"]
        y = [1, 2]
        # 创建 Plot 对象 p，设置 x 轴缩放为 Temporal()
        p = Plot(x, y).add(MockMark()).scale(x=Temporal())
        
        # 设置错误消息
        msg = "Scaling operation failed for the `x` variable."
        # 断言绘图操作会引发 PlotSpecError 异常，并匹配特定消息 msg
        with pytest.raises(PlotSpecError, match=msg) as err:
            p.plot()
        # 断言异常对象具有 __cause__ 属性
        assert hasattr(err.value, "__cause__")

    # 测试语义缩放异常情况
    def test_semantic_scaling(self):
        
        # 定义一个会抛出异常的子类 ErrorRaising，继承自 Continuous
        class ErrorRaising(Continuous):
            # 重写 _setup 方法，在其中抛出 ValueError 异常
            def _setup(self, data, prop, axis=None):
                def f(x):
                    raise ValueError("This is a test")
                new = super()._setup(data, prop, axis)
                new._pipeline = [f]
                return new
        
        # 初始化变量 x, y, color 为列表 [1, 2]
        x = y = color = [1, 2]
        # 创建 Plot 对象 p，设置颜色为 color，并尝试使用 ErrorRaising() 进行颜色缩放
        p = Plot(x, y, color=color).add(Dot()).scale(color=ErrorRaising())
        # 设置错误消息
        msg = "Scaling operation failed for the `color` variable."
        # 断言绘图操作会引发 PlotSpecError 异常，并匹配特定消息 msg
        with pytest.raises(PlotSpecError, match=msg) as err:
            p.plot()
        # 断言异常的原因是 ValueError 类型
        assert isinstance(err.value.__cause__, ValueError)
        # 断言异常消息中完全匹配 "This is a test" 字符串
        assert str(err.value.__cause__) == "This is a test"


class TestFacetInterface:
    
    # Pytest fixture，用于参数化测试维度参数 dim
    @pytest.fixture(scope="class", params=["row", "col"])
    def dim(self, request):
        return request.param
    
    # Pytest fixture，用于参数化测试排序参数 reorder
    @pytest.fixture(scope="class", params=["reverse", "subset", "expand"])
    def reorder(self, request):
        return {
            "reverse": lambda x: x[::-1],
            "subset": lambda x: x[:-1],
            "expand": lambda x: x + ["z"],
        }[request.param]

    # 检查一维分面结果的方法
    def check_facet_results_1d(self, p, df, dim, key, order=None):
        # 绘制图形并获取绘图对象 p
        p = p.plot()
        
        # 根据 order 参数对 df[key] 进行分类排序
        order = categorical_order(df[key], order)
        # 断言绘图对象的轴数与 order 的长度相等
        assert len(p._figure.axes) == len(order)

        # 获取另一个维度的名称
        other_dim = {"row": "col", "col": "row"}[dim]

        # 遍历每个子图 subplot 和其对应的分类 level
        for subplot, level in zip(p._subplots, order):
            # 断言 subplot 的 dim 维度等于对应的 level
            assert subplot[dim] == level
            # 断言 subplot 的 other_dim 维度为 None
            assert subplot[other_dim] is None
            # 断言 subplot 的轴标题与 level 名称匹配
            assert subplot["ax"].get_title() == f"{level}"
            # 调用 assert_gridspec_shape 方法，检查 subplot 的轴网格形状
            assert_gridspec_shape(subplot["ax"], **{f"n{dim}s": len(order)})

    # 测试一维分面图
    def test_1d(self, long_df, dim):
        key = "a"
        # 创建 Plot 对象 p，根据 dim 维度参数进行分面
        p = Plot(long_df).facet(**{dim: key})
        # 调用 check_facet_results_1d 方法，检查分面结果
        self.check_facet_results_1d(p, long_df, dim, key)

    # 测试将一维向量作为分面参数
    def test_1d_as_vector(self, long_df, dim):
        key = "a"
        # 创建 Plot 对象 p，将 long_df[key] 作为分面参数
        p = Plot(long_df).facet(**{dim: long_df[key]})
        # 调用 check_facet_results_1d 方法，检查分面结果
        self.check_facet_results_1d(p, long_df, dim, key)
    # 定义一个测试方法，用于测试一维数据的绘图，并按指定顺序排列
    def test_1d_with_order(self, long_df, dim, reorder):

        # 设置要操作的数据列
        key = "a"
        # 根据数据列的分类顺序重新排序
        order = reorder(categorical_order(long_df[key]))
        # 使用长格式数据创建绘图对象，并进行分面展示，按指定维度和顺序
        p = Plot(long_df).facet(**{dim: key, "order": order})
        # 检查分面绘图的结果是否正确
        self.check_facet_results_1d(p, long_df, dim, key, order)

    # 检查二维分面绘图的结果
    def check_facet_results_2d(self, p, df, variables, order=None):

        # 将绘图对象转换为图形对象
        p = p.plot()

        # 如果未指定顺序，则根据数据的分类顺序创建顺序字典
        if order is None:
            order = {dim: categorical_order(df[key]) for dim, key in variables.items()}

        # 计算行列的所有组合
        levels = itertools.product(*[order[dim] for dim in ["row", "col"]])
        # 断言
    # 测试在绘图中共享坐标轴功能
    def test_axis_sharing(self, long_df):
        # 定义变量字典，用于指定行和列的名称
        variables = {"row": "a", "col": "c"}
        
        # 创建 Plot 对象并使用指定的变量进行分面绘图
        p = Plot(long_df).facet(**variables)
        
        # 绘制图形并获取根节点及其余的轴对象
        p1 = p.plot()
        root, *other = p1._figure.axes
        
        # 遍历 xy 轴，检查是否共享轴
        for axis in "xy":
            shareset = getattr(root, f"get_shared_{axis}_axes")()
            assert all(shareset.joined(root, ax) for ax in other)
        
        # 禁止 x 和 y 轴的共享，重新绘制图形并获取根节点及其余的轴对象
        p2 = p.share(x=False, y=False).plot()
        root, *other = p2._figure.axes
        
        # 遍历 xy 轴，检查是否没有共享轴
        for axis in "xy":
            shareset = getattr(root, f"get_shared_{axis}_axes")()
            assert not any(shareset.joined(root, ax) for ax in other)
        
        # 指定 x 轴为 "col"，y 轴为 "row"，重新绘制图形
        p3 = p.share(x="col", y="row").plot()
        
        # 计算行和列的分类顺序长度，并重新整形为轴矩阵
        shape = (
            len(categorical_order(long_df[variables["row"]])),
            len(categorical_order(long_df[variables["col"]])),
        )
        axes_matrix = np.reshape(p3._figure.axes, shape)
        
        # 遍历轴矩阵，检查共享的轴和非共享的轴
        for (shared, unshared), vectors in zip(
            ["yx", "xy"], [axes_matrix, axes_matrix.T]
        ):
            for root, *other in vectors:
                shareset = {
                    axis: getattr(root, f"get_shared_{axis}_axes")() for axis in "xy"
                }
                assert all(shareset[shared].joined(root, ax) for ax in other)
                assert not any(shareset[unshared].joined(root, ax) for ax in other)

    # 测试未共享的间距功能
    def test_unshared_spacing(self):
        # 定义测试数据
        x = [1, 2, 10, 20]
        y = [1, 2, 3, 4]
        col = [1, 1, 2, 2]
        
        # 创建 MockMark 对象
        m = MockMark()
        
        # 创建 Plot 对象并设置列分面，添加 MockMark 对象并禁止 x 轴共享
        Plot(x, y).facet(col).add(m).share(x=False).plot()
        
        # 检查宽度数组是否与预期相等
        assert_array_almost_equal(m.passed_data[0]["width"], [0.8, 0.8])
        assert_array_equal(m.passed_data[1]["width"], [8, 8])

    # 测试列包装功能
    def test_col_wrapping(self):
        # 定义列名称列表和包装数量
        cols = list("abcd")
        wrap = 3
        
        # 创建 Plot 对象并设置列分面和包装数，绘制图形
        p = Plot().facet(col=cols, wrap=wrap).plot()
        
        # 检查图形的轴数是否符合预期
        assert len(p._figure.axes) == 4
        # 检查网格规范的形状是否符合预期
        assert_gridspec_shape(p._figure.axes[0], len(cols) // wrap + 1, wrap)

        # TODO 测试轴标签和标题

    # 测试行包装功能
    def test_row_wrapping(self):
        # 定义行名称列表和包装数量
        rows = list("abcd")
        wrap = 3
        
        # 创建 Plot 对象并设置行分面和包装数，绘制图形
        p = Plot().facet(row=rows, wrap=wrap).plot()
        
        # 检查网格规范的形状是否符合预期
        assert_gridspec_shape(p._figure.axes[0], wrap, len(rows) // wrap + 1)
        # 检查图形的轴数是否符合预期
        assert len(p._figure.axes) == 4

        # TODO 测试轴标签和标题
class TestPairInterface:

    # 检查图形中的坐标对应关系
    def check_pair_grid(self, p, x, y):

        # 生成 x 和 y 的笛卡尔积
        xys = itertools.product(y, x)

        # 遍历笛卡尔积以及子图列表中的子图对象
        for (y_i, x_j), subplot in zip(xys, p._subplots):

            # 获取当前子图的坐标轴对象
            ax = subplot["ax"]

            # 断言 x 轴标签是否为空字符串（如果 x_j 为 None），否则为 x_j
            assert ax.get_xlabel() == "" if x_j is None else x_j

            # 断言 y 轴标签是否为空字符串（如果 y_i 为 None），否则为 y_i
            assert ax.get_ylabel() == "" if y_i is None else y_i

            # 断言子图的网格形状是否符合预期
            assert_gridspec_shape(subplot["ax"], len(y), len(x))

    # 使用 pytest 参数化装饰器定义测试用例，测试所有的数值类型
    @pytest.mark.parametrize("vector_type", [list, pd.Index])
    def test_all_numeric(self, long_df, vector_type):

        # 定义 x 和 y 变量
        x, y = ["x", "y", "z"], ["s", "f"]

        # 创建 Plot 对象，生成图形并进行测试
        p = Plot(long_df).pair(vector_type(x), vector_type(y)).plot()
        self.check_pair_grid(p, x, y)

    # 测试当传入单个变量键值时是否会引发异常
    def test_single_variable_key_raises(self, long_df):

        # 创建 Plot 对象
        p = Plot(long_df)

        # 定义错误消息
        err = "You must pass a sequence of variable keys to `y`"

        # 断言调用 pair 方法时传入单个变量键值是否会抛出 TypeError 异常
        with pytest.raises(TypeError, match=err):
            p.pair(x=["x", "y"], y="z")

    # 使用 pytest 参数化装饰器定义测试用例，测试单个维度
    @pytest.mark.parametrize("dim", ["x", "y"])
    def test_single_dimension(self, long_df, dim):

        # 定义变量字典
        variables = {"x": None, "y": None}

        # 根据传入的维度参数设置对应的变量列表
        variables[dim] = ["x", "y", "z"]

        # 创建 Plot 对象，生成图形并进行测试
        p = Plot(long_df).pair(**variables).plot()

        # 调整变量字典，将 None 值转换为包含单个元素列表
        variables = {k: [v] if v is None else v for k, v in variables.items()}

        # 检查图形的坐标对应关系
        self.check_pair_grid(p, **variables)

    # 测试在不交叉的情况下生成图形
    def test_non_cross(self, long_df):

        # 定义 x 和 y 变量
        x = ["x", "y"]
        y = ["f", "z"]

        # 创建 Plot 对象，生成图形并进行测试
        p = Plot(long_df).pair(x, y, cross=False).plot()

        # 遍历子图列表进行断言
        for i, subplot in enumerate(p._subplots):
            ax = subplot["ax"]
            assert ax.get_xlabel() == x[i]
            assert ax.get_ylabel() == y[i]
            assert_gridspec_shape(ax, 1, len(x))

        # 检查根坐标轴和其他坐标轴之间是否共享任何轴
        root, *other = p._figure.axes
        for axis in "xy":
            shareset = getattr(root, f"get_shared_{axis}_axes")()
            assert not any(shareset.joined(root, ax) for ax in other)

    # 测试使用向量列表生成图形
    def test_list_of_vectors(self, long_df):

        # 定义 x 变量列表
        x_vars = ["x", "z"]

        # 创建 Plot 对象，生成图形并进行测试
        p = Plot(long_df, y="y").pair(x=[long_df[x] for x in x_vars]).plot()

        # 断言图形的坐标轴数量与 x 变量列表长度相等
        assert len(p._figure.axes) == len(x_vars)

        # 遍历坐标轴对象进行断言
        for ax, x_i in zip(p._figure.axes, x_vars):
            assert ax.get_xlabel() == x_i

    # 测试在没有任何变量的情况下生成图形
    def test_with_no_variables(self, long_df):

        # 创建 Plot 对象，生成图形并进行测试
        p = Plot(long_df).pair().plot()

        # 断言图形的坐标轴数量为 1
        assert len(p._figure.axes) == 1

    # 测试使用分面显示图形
    def test_with_facets(self, long_df):

        # 定义 x、y 和列变量
        x = "x"
        y = ["y", "z"]
        col = "a"

        # 创建 Plot 对象，生成图形并进行测试
        p = Plot(long_df, x=x).facet(col).pair(y=y).plot()

        # 获取分面变量的分类顺序
        facet_levels = categorical_order(long_df[col])

        # 生成 y 和列变量的笛卡尔积
        dims = itertools.product(y, facet_levels)

        # 遍历笛卡尔积以及子图列表中的子图对象进行断言
        for (y_i, col_i), subplot in zip(dims, p._subplots):

            # 获取当前子图的坐标轴对象
            ax = subplot["ax"]

            # 断言 x 轴标签
            assert ax.get_xlabel() == x

            # 断言 y 轴标签
            assert ax.get_ylabel() == y_i

            # 断言子图的标题
            assert ax.get_title() == f"{col_i}"

            # 断言子图的网格形状是否符合预期
            assert_gridspec_shape(ax, len(y), len(facet_levels))

    # 使用 pytest 参数化装饰器定义测试用例，测试不同变量的排列组合
    @pytest.mark.parametrize("variables", [("rows", "y"), ("columns", "x")])
    # 测试函数，验证在面板重叠时是否会引发错误
    def test_error_on_facet_overlap(self, long_df, variables):

        # 解包变量元组
        facet_dim, pair_axis = variables
        # 创建 Plot 对象，并进行面板化和配对操作
        p = Plot(long_df).facet(**{facet_dim[:3]: "a"}).pair(**{pair_axis: ["x", "y"]})
        # 期望的错误信息
        expected = f"Cannot facet the {facet_dim} while pairing on `{pair_axis}`."
        # 断言引发 RuntimeError，并匹配期望的错误信息
        with pytest.raises(RuntimeError, match=expected):
            p.plot()

    # 使用参数化测试装饰器，验证在包装重叠时是否会引发错误
    @pytest.mark.parametrize("variables", [("columns", "y"), ("rows", "x")])
    def test_error_on_wrap_overlap(self, long_df, variables):

        # 解包变量元组
        facet_dim, pair_axis = variables
        # 创建 Plot 对象，并进行包装、配对操作
        p = (
            Plot(long_df)
            .facet(wrap=2, **{facet_dim[:3]: "a"})
            .pair(**{pair_axis: ["x", "y"]})
        )
        # 期望的错误信息
        expected = f"Cannot wrap the {facet_dim} while pairing on `{pair_axis}``."
        # 断言引发 RuntimeError，并匹配期望的错误信息
        with pytest.raises(RuntimeError, match=expected):
            p.plot()

    # 测试函数，验证轴共享功能
    def test_axis_sharing(self, long_df):

        # 创建 Plot 对象，并进行配对操作
        p = Plot(long_df).pair(x=["a", "b"], y=["y", "z"])
        # 设置形状
        shape = 2, 2
        # 调用 plot 方法，获取第一个图表对象
        p1 = p.plot()
        # 将 axes 属性的矩阵形状化为 shape
        axes_matrix = np.reshape(p1._figure.axes, shape)

        # 遍历 axes_matrix，测试行级共享
        for root, *other in axes_matrix:
            # 获取共享的 x 轴
            x_shareset = getattr(root, "get_shared_x_axes")()
            # 断言所有其他轴都不共享 root 轴的 x 轴
            assert not any(x_shareset.joined(root, ax) for ax in other)
            # 获取共享的 y 轴
            y_shareset = getattr(root, "get_shared_y_axes")()
            # 断言所有其他轴都共享 root 轴的 y 轴
            assert all(y_shareset.joined(root, ax) for ax in other)

        # 遍历 axes_matrix.T，测试列级共享
        for root, *other in axes_matrix.T:
            # 获取共享的 x 轴
            x_shareset = getattr(root, "get_shared_x_axes")()
            # 断言所有其他轴都共享 root 轴的 x 轴
            assert all(x_shareset.joined(root, ax) for ax in other)
            # 获取共享的 y 轴
            y_shareset = getattr(root, "get_shared_y_axes")()
            # 断言所有其他轴都不共享 root 轴的 y 轴
            assert not any(y_shareset.joined(root, ax) for ax in other)

        # 调用 share 方法禁用 x 和 y 轴的共享，获取第二个图表对象
        p2 = p.share(x=False, y=False).plot()
        # 解包 p2._figure.axes 的根轴和其他轴
        root, *other = p2._figure.axes
        # 遍历 x 和 y 轴
        for axis in "xy":
            # 获取共享的轴
            shareset = getattr(root, f"get_shared_{axis}_axes")()
            # 断言所有其他轴都不共享 root 轴的该轴
            assert not any(shareset.joined(root, ax) for ax in other)

    # 测试函数，验证带有面板的轴共享功能
    def test_axis_sharing_with_facets(self, long_df):

        # 创建 Plot 对象，并进行配对和面板化操作
        p = Plot(long_df, y="y").pair(x=["a", "b"]).facet(row="c").plot()
        # 设置形状
        shape = 2, 2
        # 将 axes 属性的矩阵形状化为 shape
        axes_matrix = np.reshape(p._figure.axes, shape)

        # 遍历 axes_matrix，测试行级共享
        for root, *other in axes_matrix:
            # 获取共享的 x 轴
            x_shareset = getattr(root, "get_shared_x_axes")()
            # 断言所有其他轴都不共享 root 轴的 x 轴
            assert not any(x_shareset.joined(root, ax) for ax in other)
            # 获取共享的 y 轴
            y_shareset = getattr(root, "get_shared_y_axes")()
            # 断言所有其他轴都共享 root 轴的 y 轴
            assert all(y_shareset.joined(root, ax) for ax in other)

        # 遍历 axes_matrix.T，测试列级共享
        for root, *other in axes_matrix.T:
            # 获取共享的 x 轴
            x_shareset = getattr(root, "get_shared_x_axes")()
            # 断言所有其他轴都共享 root 轴的 x 轴
            assert all(x_shareset.joined(root, ax) for ax in other)
            # 获取共享的 y 轴
            y_shareset = getattr(root, "get_shared_y_axes")()
            # 断言所有其他轴都共享 root 轴的 y 轴
            assert all(y_shareset.joined(root, ax) for ax in other)
    # 测试 x 变量包装功能
    def test_x_wrapping(self, long_df):

        # 定义要绘制的 x 变量列表
        x_vars = ["f", "x", "y", "z"]
        # 每行包含的图形数量
        wrap = 3
        # 创建 Plot 对象，并使用 pair 方法设置 x 变量和包装数，然后绘图
        p = Plot(long_df, y="y").pair(x=x_vars, wrap=wrap).plot()

        # 断言第一个图形的布局，检查是否按预期的行列布局
        assert_gridspec_shape(p._figure.axes[0], len(x_vars) // wrap + 1, wrap)
        # 断言图形的数量与 x 变量列表的长度一致
        assert len(p._figure.axes) == len(x_vars)
        # 遍历每个轴对象和对应的 x 变量，检查 x 轴标签是否正确设置
        for ax, var in zip(p._figure.axes, x_vars):
            label = ax.xaxis.get_label()
            assert label.get_visible()
            assert label.get_text() == var

    # 测试 y 变量包装功能
    def test_y_wrapping(self, long_df):

        # 定义要绘制的 y 变量列表
        y_vars = ["f", "x", "y", "z"]
        # 每行包含的图形数量
        wrap = 3
        # 创建 Plot 对象，并使用 pair 方法设置 y 变量和包装数，然后绘图
        p = Plot(long_df, x="x").pair(y=y_vars, wrap=wrap).plot()

        # 计算期望的行数和列数
        n_row, n_col = wrap, len(y_vars) // wrap + 1
        # 断言第一个图形的布局，检查是否按预期的行列布局
        assert_gridspec_shape(p._figure.axes[0], n_row, n_col)
        # 断言图形的数量与 y 变量列表的长度一致
        assert len(p._figure.axes) == len(y_vars)
        
        # 创建一个空的标签数组
        label_array = np.empty(n_row * n_col, object)
        # 将 y 变量列表填充到标签数组中
        label_array[:len(y_vars)] = y_vars
        # 重新整形标签数组以匹配图形的行列顺序
        label_array = label_array.reshape((n_row, n_col), order="F")
        # 将标签数组扁平化，去除空值，形成一个一维列表
        label_array = [y for y in label_array.flat if y is not None]
        
        # 遍历每个轴对象和对应的 y 变量，检查 y 轴标签是否正确设置
        for i, ax in enumerate(p._figure.axes):
            label = ax.yaxis.get_label()
            assert label.get_visible()
            assert label.get_text() == label_array[i]

    # 测试非交叉包装功能
    def test_non_cross_wrapping(self, long_df):

        # 定义要绘制的 x 和 y 变量列表
        x_vars = ["a", "b", "c", "t"]
        y_vars = ["f", "x", "y", "z"]
        # 每行包含的图形数量
        wrap = 3

        # 创建 Plot 对象，并使用 pair 方法设置 x 和 y 变量以及包装数，然后绘图
        p = (
            Plot(long_df, x="x")
            .pair(x=x_vars, y=y_vars, wrap=wrap, cross=False)
            .plot()
        )

        # 断言第一个图形的布局，检查是否按预期的行列布局
        assert_gridspec_shape(p._figure.axes[0], len(x_vars) // wrap + 1, wrap)
        # 断言图形的数量与 x 变量列表的长度一致
        assert len(p._figure.axes) == len(x_vars)

    # 测试交叉时变量长度不匹配的错误
    def test_cross_mismatched_lengths(self, long_df):

        # 创建 Plot 对象
        p = Plot(long_df)
        # 使用 pytest 的断言检查是否抛出预期的 ValueError 异常
        with pytest.raises(ValueError, match="Lengths of the `x` and `y`"):
            # 调用 pair 方法时传递不匹配长度的 x 和 y 变量列表，且禁用交叉
            p.pair(x=["a", "b"], y=["x", "y", "z"], cross=False)

    # 测试方向推断功能
    def test_orient_inference(self, long_df):

        # 用于捕获推断的方向列表
        orient_list = []

        # 定义一个 Mock 类来捕获方向推断的操作
        class CaptureOrientMove(Move):
            def __call__(self, data, groupby, orient, scales):
                orient_list.append(orient)
                return data

        # 创建 Plot 对象，并设置 x 变量，然后使用 pair 方法设置 y 变量，添加 MockMark 和捕获方向操作，最后绘图
        (
            Plot(long_df, x="x")
            .pair(y=["b", "z"])
            .add(MockMark(), CaptureOrientMove())
            .plot()
        )

        # 断言捕获到的方向列表是否符合预期
        assert orient_list == ["y", "x"]

    # 测试计算坐标推断方向功能
    def test_computed_coordinate_orient_inference(self, long_df):

        # 定义一个 Mock 类来模拟计算统计操作
        class MockComputeStat(Stat):
            def __call__(self, df, groupby, orient, scales):
                # 根据推断的方向设置 MockMark 的属性
                other = {"x": "y", "y": "x"}[orient]
                return df.assign(**{other: df[orient] * 2})

        # 创建一个 MockMark 对象
        m = MockMark()
        # 创建 Plot 对象，并设置 y 变量，然后添加 MockMark 和 MockComputeStat，最后绘图
        Plot(long_df, y="y").add(m, MockComputeStat()).plot()
        # 使用 pytest 断言检查 MockMark 的 passed_orient 属性是否符合预期
        assert m.passed_orient == "y"

    # 测试当同时分面 col 和 row 时传递 order 的错误
    def test_two_variables_single_order_error(self, long_df):

        # 创建 Plot 对象
        p = Plot(long_df)
        # 使用 pytest 断言检查是否抛出预期的 RuntimeError 异常
        err = "When faceting on both col= and row=, passing `order`"
        with pytest.raises(RuntimeError, match=err):
            # 调用 facet 方法时同时传递 col 和 row 参数，并传递 order 参数
            p.facet(col="a", row="b", order=["a", "b", "c"])
    # 定义测试方法，用于测试绘图库的限制功能
    def test_limits(self, long_df):
        # 设定 x 和 x1 轴的限制范围
        lims = (-3, 10), (-2, 24)
        # 创建 Plot 对象 p，传入长格式数据 long_df，设置 y 轴为 "y"，绘制 x 和 z 的关系图，
        # 并设置 x 轴的限制为 lims[0]，x1 轴的限制为 lims[1]，然后绘制图形
        p = Plot(long_df, y="y").pair(x=["x", "z"]).limit(x=lims[0], x1=lims[1]).plot()
        # 遍历 p 对象的所有子图 axes，与 lims 中的限制一一比较，确保设置的限制生效
        for ax, lim in zip(p._figure.axes, lims):
            assert ax.get_xlim() == lim

    # 定义测试方法，用于测试绘图库的标签功能
    def test_labels(self, long_df):
        # 设定 x 轴的标签为 "zed"
        label = "zed"
        # 创建 Plot 对象 p，传入长格式数据 long_df，设置 y 轴为 "y"，
        # 绘制 x 和 z 的关系图，并设置 x 轴标签格式化为大写，x1 轴标签为 label
        p = (
            Plot(long_df, y="y")
            .pair(x=["x", "z"])
            .label(x=str.capitalize, x1=label)
        )
        # 绘制图形并获取生成的子图 axes
        ax0, ax1 = p.plot()._figure.axes
        # 断言第一个子图的 x 轴标签为 "X"
        assert ax0.get_xlabel() == "X"
        # 断言第二个子图的 x 轴标签为预设的 label
        assert ax1.get_xlabel() == label
# 定义一个名为 TestLabelVisibility 的测试类，用于测试标签可见性相关功能
class TestLabelVisibility:

    # 测试单个子图的方法，接受 long_df 作为参数
    def test_single_subplot(self, long_df):
        # 设置 x 和 y 轴的标签
        x, y = "a", "z"
        # 创建 Plot 对象 p，使用 long_df 数据，指定 x 和 y 轴，并绘制图形
        p = Plot(long_df, x=x, y=y).plot()
        # 获取第一个子图 subplot
        subplot, *_ = p._subplots
        # 获取 subplot 的轴对象 ax
        ax = subplot["ax"]
        # 断言 x 轴的标签是否可见
        assert ax.xaxis.get_label().get_visible()
        # 断言 y 轴的标签是否可见
        assert ax.yaxis.get_label().get_visible()
        # 断言 x 轴上所有刻度标签是否可见
        assert all(t.get_visible() for t in ax.get_xticklabels())
        # 断言 y 轴上所有刻度标签是否可见
        assert all(t.get_visible() for t in ax.get_yticklabels())

    # 使用参数化测试，测试带有 facet_kws 和 pair_kws 参数的方法
    @pytest.mark.parametrize(
        "facet_kws,pair_kws", [({"col": "b"}, {}), ({}, {"x": ["x", "y", "f"]})]
    )
    def test_1d_column(self, long_df, facet_kws, pair_kws):
        # 根据 pair_kws 设置 x 的值
        x = None if "x" in pair_kws else "a"
        # 设置 y 的值为 "z"
        y = "z"
        # 创建 Plot 对象 p，使用 long_df 数据，指定 x 和 y 轴，并绘制图形
        p = Plot(long_df, x=x, y=y).plot()
        # 获取第一个子图 first 和其他子图 other
        first, *other = p._subplots

        # 获取第一个子图的轴对象 ax
        ax = first["ax"]
        # 断言 x 轴的标签是否可见
        assert ax.xaxis.get_label().get_visible()
        # 断言 y 轴的标签是否可见
        assert ax.yaxis.get_label().get_visible()
        # 断言 x 轴上所有刻度标签是否可见
        assert all(t.get_visible() for t in ax.get_xticklabels())
        # 断言 y 轴上所有刻度标签是否可见
        assert all(t.get_visible() for t in ax.get_yticklabels())

        # 遍历其他子图
        for s in other:
            # 获取当前子图的轴对象 ax
            ax = s["ax"]
            # 断言 x 轴的标签是否可见
            assert ax.xaxis.get_label().get_visible()
            # 断言 y 轴的标签是否不可见
            assert not ax.yaxis.get_label().get_visible()
            # 断言 x 轴上所有刻度标签是否可见
            assert all(t.get_visible() for t in ax.get_xticklabels())
            # 断言 y 轴上所有刻度标签是否不可见
            assert not any(t.get_visible() for t in ax.get_yticklabels())

    # 使用参数化测试，测试带有 facet_kws 和 pair_kws 参数的方法
    @pytest.mark.parametrize(
        "facet_kws,pair_kws", [({"row": "b"}, {}), ({}, {"y": ["x", "y", "f"]})]
    )
    def test_1d_row(self, long_df, facet_kws, pair_kws):
        # 设置 x 的值为 "z"
        x = "z"
        # 根据 pair_kws 设置 y 的值
        y = None if "y" in pair_kws else "z"
        # 创建 Plot 对象 p，使用 long_df 数据，指定 x 和 y 轴，并绘制图形
        p = Plot(long_df, x=x, y=y).plot()
        # 获取第一个子图 first 和其他子图 other
        first, *other = p._subplots

        # 获取第一个子图的轴对象 ax
        ax = first["ax"]
        # 断言 x 轴的标签是否可见
        assert ax.xaxis.get_label().get_visible()
        # 断言 x 轴上所有刻度标签是否可见
        assert all(t.get_visible() for t in ax.get_xticklabels())
        # 断言 y 轴的标签是否可见
        assert ax.yaxis.get_label().get_visible()
        # 断言 y 轴上所有刻度标签是否可见
        assert all(t.get_visible() for t in ax.get_yticklabels())

        # 遍历其他子图
        for s in other:
            # 获取当前子图的轴对象 ax
            ax = s["ax"]
            # 断言 x 轴的标签是否不可见
            assert not ax.xaxis.get_label().get_visible()
            # 断言 y 轴的标签是否可见
            assert ax.yaxis.get_label().get_visible()
            # 断言 x 轴上所有刻度标签是否不可见
            assert not any(t.get_visible() for t in ax.get_xticklabels())
            # 断言 y 轴上所有刻度标签是否可见
            assert all(t.get_visible() for t in ax.get_yticklabels())
    def test_1d_column_wrapped(self):
        # 创建一个 Plot 对象，并指定列分面为 ["a", "b", "c", "d"]，每行最多展示3列，然后生成图表
        p = Plot().facet(col=["a", "b", "c", "d"], wrap=3).plot()
        # 获取生成的子图列表
        subplots = list(p._subplots)

        # 对于第一个子图和最后一个子图，验证其 y 轴标签可见，并且所有 y 轴刻度标签可见
        for s in [subplots[0], subplots[-1]]:
            ax = s["ax"]
            assert ax.yaxis.get_label().get_visible()
            assert all(t.get_visible() for t in ax.get_yticklabels())

        # 对于除了第一个和最后一个之外的子图，验证其 x 轴标签可见，并根据 matplotlib 版本检查 x 轴刻度参数
        for s in subplots[1:]:
            ax = s["ax"]
            assert ax.xaxis.get_label().get_visible()
            # 如果 matplotlib 版本不早于 3.7，则验证 x 轴刻度参数的 "labelleft" 属性
            if not _version_predates(mpl, "3.7"):
                assert ax.xaxis.get_tick_params()["labelleft"]
            else:
                assert len(ax.get_xticklabels()) > 0
            assert all(t.get_visible() for t in ax.get_xticklabels())

        # 对于除了第一个和最后一个之外的子图，验证其 y 轴标签不可见，并且没有任何 y 轴刻度标签可见
        for s in subplots[1:-1]:
            ax = s["ax"]
            assert not ax.yaxis.get_label().get_visible()
            assert not any(t.get_visible() for t in ax.get_yticklabels())

        # 验证第一个子图的 x 轴标签不可见，并且没有任何 x 轴刻度标签可见
        ax = subplots[0]["ax"]
        assert not ax.xaxis.get_label().get_visible()
        assert not any(t.get_visible() for t in ax.get_xticklabels())

    def test_1d_row_wrapped(self):
        # 创建一个 Plot 对象，并指定行分面为 ["a", "b", "c", "d"]，每行最多展示3行，然后生成图表
        p = Plot().facet(row=["a", "b", "c", "d"], wrap=3).plot()
        # 获取生成的子图列表
        subplots = list(p._subplots)

        # 对于除了最后一个之外的子图，验证其 y 轴标签可见，并且所有 y 轴刻度标签可见
        for s in subplots[:-1]:
            ax = s["ax"]
            assert ax.yaxis.get_label().get_visible()
            assert all(t.get_visible() for t in ax.get_yticklabels())

        # 对于倒数第二个和最后一个子图，验证其 x 轴标签可见，并根据 matplotlib 版本检查 x 轴刻度参数
        for s in subplots[-2:]:
            ax = s["ax"]
            assert ax.xaxis.get_label().get_visible()
            # 如果 matplotlib 版本不早于 3.7，则验证 x 轴刻度参数的 "labelleft" 属性
            if not _version_predates(mpl, "3.7"):
                assert ax.xaxis.get_tick_params()["labelleft"]
            else:
                assert len(ax.get_xticklabels()) > 0
            assert all(t.get_visible() for t in ax.get_xticklabels())

        # 对于除了倒数第二个和最后一个之外的子图，验证其 x 轴标签不可见，并且没有任何 x 轴刻度标签可见
        for s in subplots[:-2]:
            ax = s["ax"]
            assert not ax.xaxis.get_label().get_visible()
            assert not any(t.get_visible() for t in ax.get_xticklabels())

        # 验证最后一个子图的 y 轴标签不可见，并且没有任何 y 轴刻度标签可见
        ax = subplots[-1]["ax"]
        assert not ax.yaxis.get_label().get_visible()
        assert not any(t.get_visible() for t in ax.get_yticklabels())

    def test_1d_column_wrapped_non_cross(self, long_df):
        # 创建一个 Plot 对象，并指定 x 列为 ["a", "b", "c"]，y 列为 ["x", "y", "z"]，每行最多展示2列，且不交叉，然后生成图表
        p = (
            Plot(long_df)
            .pair(x=["a", "b", "c"], y=["x", "y", "z"], wrap=2, cross=False)
            .plot()
        )
        # 遍历生成的子图列表
        for s in p._subplots:
            ax = s["ax"]
            assert ax.xaxis.get_label().get_visible()
            assert all(t.get_visible() for t in ax.get_xticklabels())
            assert ax.yaxis.get_label().get_visible()
            assert all(t.get_visible() for t in ax.get_yticklabels())
    # 定义一个名为 test_2d 的测试方法，用于测试某个绘图对象的二维图形
    def test_2d(self):

        # 创建 Plot 对象，并设置列（col）和行（row）的分面维度，生成图形并返回
        p = Plot().facet(col=["a", "b"], row=["x", "y"]).plot()
        # 获取图形中的子图列表
        subplots = list(p._subplots)

        # 对于前两个子图执行以下断言：
        for s in subplots[:2]:
            ax = s["ax"]
            # 断言 x 轴标签不可见
            assert not ax.xaxis.get_label().get_visible()
            # 断言 x 轴刻度标签全部可见
            assert not any(t.get_visible() for t in ax.get_xticklabels())

        # 对于后两个子图执行以下断言：
        for s in subplots[2:]:
            ax = s["ax"]
            # 断言 x 轴标签可见
            assert ax.xaxis.get_label().get_visible()
            # 断言 x 轴刻度标签全部可见
            assert all(t.get_visible() for t in ax.get_xticklabels())

        # 对于第一个和第三个子图执行以下断言：
        for s in [subplots[0], subplots[2]]:
            ax = s["ax"]
            # 断言 y 轴标签可见
            assert ax.yaxis.get_label().get_visible()
            # 断言 y 轴刻度标签全部可见
            assert all(t.get_visible() for t in ax.get_yticklabels())

        # 对于第二个和第四个子图执行以下断言：
        for s in [subplots[1], subplots[3]]:
            ax = s["ax"]
            # 断言 y 轴标签不可见
            assert not ax.yaxis.get_label().get_visible()
            # 断言 y 轴刻度标签全部可见
            assert not any(t.get_visible() for t in ax.get_yticklabels())

    # 定义一个名为 test_2d_unshared 的测试方法，用于测试某个绘图对象的非共享二维图形
    def test_2d_unshared(self):

        # 创建 Plot 对象，并设置列（col）和行（row）的分面维度，并取消 x 和 y 轴的共享，生成图形并返回
        p = (
            Plot()
            .facet(col=["a", "b"], row=["x", "y"])
            .share(x=False, y=False)
            .plot()
        )
        # 获取图形中的子图列表
        subplots = list(p._subplots)

        # 对于前两个子图执行以下断言：
        for s in subplots[:2]:
            ax = s["ax"]
            # 断言 x 轴标签不可见
            assert not ax.xaxis.get_label().get_visible()
            # 断言 x 轴刻度标签全部可见
            assert all(t.get_visible() for t in ax.get_xticklabels())

        # 对于后两个子图执行以下断言：
        for s in subplots[2:]:
            ax = s["ax"]
            # 断言 x 轴标签可见
            assert ax.xaxis.get_label().get_visible()
            # 断言 x 轴刻度标签全部可见
            assert all(t.get_visible() for t in ax.get_xticklabels())

        # 对于第一个和第三个子图执行以下断言：
        for s in [subplots[0], subplots[2]]:
            ax = s["ax"]
            # 断言 y 轴标签可见
            assert ax.yaxis.get_label().get_visible()
            # 断言 y 轴刻度标签全部可见
            assert all(t.get_visible() for t in ax.get_yticklabels())

        # 对于第二个和第四个子图执行以下断言：
        for s in [subplots[1], subplots[3]]:
            ax = s["ax"]
            # 断言 y 轴标签不可见
            assert not ax.yaxis.get_label().get_visible()
            # 断言 y 轴刻度标签全部可见
            assert all(t.get_visible() for t in ax.get_yticklabels())
# 定义一个名为 TestLegend 的测试类
class TestLegend:

    # 使用 pytest 的 fixture 装饰器定义 xy 方法，返回一个包含 x 和 y 列表的字典
    @pytest.fixture
    def xy(self):
        return dict(x=[1, 2, 3, 4], y=[1, 2, 3, 4])

    # 测试单层单变量情况
    def test_single_layer_single_variable(self, xy):

        # 创建一个名为 s 的 pandas Series 对象，内容为 ["a", "b", "a", "c"]，名称为 "s"
        s = pd.Series(["a", "b", "a", "c"], name="s")
        # 创建 Plot 对象，并添加 MockMark，根据 xy 字典设置参数，根据 s 的值设置颜色，生成绘图并获取返回结果
        p = Plot(**xy).add(MockMark(), color=s).plot()
        # 从绘图结果中获取图例内容
        e, = p._legend_contents

        # 根据 s 的值获取分类顺序的标签
        labels = categorical_order(s)

        # 断言图例内容的第一个元素为 (s.name, s.name)
        assert e[0] == (s.name, s.name)
        # 断言图例内容的最后一个元素为 labels
        assert e[-1] == labels

        # 获取图例中的艺术家对象
        artists = e[1]
        # 断言艺术家对象的数量与标签数量相同
        assert len(artists) == len(labels)
        # 遍历艺术家对象和标签，断言艺术家对象为 mpl.artist.Artist 类型，并且其值和变量为预期值
        for a, label in zip(artists, labels):
            assert isinstance(a, mpl.artist.Artist)
            assert a.value == label
            assert a.variables == ["color"]

    # 测试单层共同变量情况
    def test_single_layer_common_variable(self, xy):

        # 创建一个名为 s 的 pandas Series 对象，内容为 ["a", "b", "a", "c"]，名称为 "s"
        s = pd.Series(["a", "b", "a", "c"], name="s")
        # 根据 s 创建一个包含颜色和标记的字典
        sem = dict(color=s, marker=s)
        # 创建 Plot 对象，并添加 MockMark，根据 xy 字典设置参数，根据 sem 字典的值设置颜色和标记，生成绘图并获取返回结果
        p = Plot(**xy).add(MockMark(), **sem).plot()
        # 从绘图结果中获取图例内容
        e, = p._legend_contents

        # 根据 s 的值获取分类顺序的标签
        labels = categorical_order(s)

        # 断言图例内容的第一个元素为 (s.name, s.name)
        assert e[0] == (s.name, s.name)
        # 断言图例内容的最后一个元素为 labels
        assert e[-1] == labels

        # 获取图例中的艺术家对象
        artists = e[1]
        # 断言艺术家对象的数量与标签数量相同
        assert len(artists) == len(labels)
        # 遍历艺术家对象和标签，断言艺术家对象为 mpl.artist.Artist 类型，并且其值和变量为预期值
        for a, label in zip(artists, labels):
            assert isinstance(a, mpl.artist.Artist)
            assert a.value == label
            assert a.variables == list(sem)

    # 测试单层共同未命名变量情况
    def test_single_layer_common_unnamed_variable(self, xy):

        # 创建一个 numpy 数组 s，内容为 ["a", "b", "a", "c"]
        s = np.array(["a", "b", "a", "c"])
        # 根据 s 创建一个包含颜色和标记的字典
        sem = dict(color=s, marker=s)
        # 创建 Plot 对象，并添加 MockMark，根据 xy 字典设置参数，根据 sem 字典的值设置颜色和标记，生成绘图并获取返回结果
        p = Plot(**xy).add(MockMark(), **sem).plot()

        # 从绘图结果中获取图例内容
        e, = p._legend_contents

        # 根据 s 的唯一值获取排序后的标签列表
        labels = list(np.unique(s))  # 假定为已排序顺序

        # 断言图例内容的第一个元素为 ("", id(s))
        assert e[0] == ("", id(s))
        # 断言图例内容的最后一个元素为 labels
        assert e[-1] == labels

        # 获取图例中的艺术家对象
        artists = e[1]
        # 断言艺术家对象的数量与标签数量相同
        assert len(artists) == len(labels)
        # 遍历艺术家对象和标签，断言艺术家对象为 mpl.artist.Artist 类型，并且其值和变量为预期值
        for a, label in zip(artists, labels):
            assert isinstance(a, mpl.artist.Artist)
            assert a.value == label
            assert a.variables == list(sem)

    # 测试单层多变量情况
    def test_single_layer_multi_variable(self, xy):

        # 创建一个名为 s1 的 pandas Series 对象，内容为 ["a", "b", "a", "c"]，名称为 "s1"
        s1 = pd.Series(["a", "b", "a", "c"], name="s1")
        # 创建一个名为 s2 的 pandas Series 对象，内容为 ["m", "m", "p", "m"]，名称为 "s2"
        s2 = pd.Series(["m", "m", "p", "m"], name="s2")
        # 根据 s1 和 s2 创建一个包含颜色和标记的字典
        sem = dict(color=s1, marker=s2)
        # 创建 Plot 对象，并添加 MockMark，根据 xy 字典设置参数，根据 sem 字典的值设置颜色和标记，生成绘图并获取返回结果
        p = Plot(**xy).add(MockMark(), **sem).plot()
        # 从绘图结果中获取两个图例内容 e1 和 e2
        e1, e2 = p._legend_contents

        # 创建一个字典，用于存储变量名到列名的映射关系
        variables = {v.name: k for k, v in sem.items()}

        # 分别对 e1 和 e2 进行断言
        for e, s in zip([e1, e2], [s1, s2]):
            # 断言图例内容的第一个元素为 (s.name, s.name)
            assert e[0] == (s.name, s.name)

            # 根据 s 的值获取分类顺序的标签
            labels = categorical_order(s)
            # 断言图例内容的最后一个元素为 labels
            assert e[-1] == labels

            # 获取图例中的艺术家对象
            artists = e[1]
            # 断言艺术家对象的数量与标签数量相同
            assert len(artists) == len(labels)
            # 遍历艺术家对象和标签，断言艺术家对象为 mpl.artist.Artist 类型，并且其值和变量为预期值
            for a, label in zip(artists, labels):
                assert isinstance(a, mpl.artist.Artist)
                assert a.value == label
                assert a.variables == [variables[s.name]]
    # 定义一个测试方法，用于测试单变量多层图表
    def test_multi_layer_single_variable(self, xy):
        # 创建一个包含字符串的 Pandas Series 对象
        s = pd.Series(["a", "b", "a", "c"], name="s")
        # 创建一个 Plot 对象，使用给定的参数 xy 和 Series s 作为颜色参数，添加两个 MockMark 对象，然后绘制图表
        p = Plot(**xy, color=s).add(MockMark()).add(MockMark()).plot()
        # 获取图例内容的引用
        e1, e2 = p._legend_contents

        # 获取分类顺序的标签
        labels = categorical_order(s)

        # 对于每个图例内容 e，执行断言
        for e in [e1, e2]:
            # 断言第一个条目是 (s.name, s.name)
            assert e[0] == (s.name, s.name)

            # 重新获取分类顺序的标签
            labels = categorical_order(s)
            # 断言最后一个条目与标签相同
            assert e[-1] == labels

            # 获取艺术家列表
            artists = e[1]
            # 断言艺术家列表的长度与标签列表的长度相同
            assert len(artists) == len(labels)
            # 遍历艺术家和标签，执行多个断言
            for a, label in zip(artists, labels):
                assert isinstance(a, mpl.artist.Artist)
                assert a.value == label
                assert a.variables == ["color"]

    # 定义一个测试方法，用于测试多变量多层图表
    def test_multi_layer_multi_variable(self, xy):
        # 创建两个包含字符串的 Pandas Series 对象
        s1 = pd.Series(["a", "b", "a", "c"], name="s1")
        s2 = pd.Series(["m", "m", "p", "m"], name="s2")
        # 创建一个包含颜色和标记的字典
        sem = dict(color=s1), dict(marker=s2)
        # 创建变量映射字典
        variables = {"s1": "color", "s2": "marker"}
        # 创建一个 Plot 对象，添加两个 MockMark 对象并传入 sem 字典的值作为参数，然后绘制图表
        p = Plot(**xy).add(MockMark(), **sem[0]).add(MockMark(), **sem[1]).plot()
        # 获取图例内容的引用
        e1, e2 = p._legend_contents

        # 对于每个图例内容 e 和对应的 Series s，执行断言
        for e, s in zip([e1, e2], [s1, s2]):
            # 断言第一个条目是 (s.name, s.name)
            assert e[0] == (s.name, s.name)

            # 获取分类顺序的标签
            labels = categorical_order(s)
            # 断言最后一个条目与标签相同
            assert e[-1] == labels

            # 获取艺术家列表
            artists = e[1]
            # 断言艺术家列表的长度与标签列表的长度相同
            assert len(artists) == len(labels)
            # 遍历艺术家和标签，执行多个断言
            for a, label in zip(artists, labels):
                assert isinstance(a, mpl.artist.Artist)
                assert a.value == label
                assert a.variables == [variables[s.name]]

    # 定义一个测试方法，用于测试包含不同类型艺术家的多层图表
    def test_multi_layer_different_artists(self, xy):
        # 定义 MockMark1 类，继承自 MockMark，用于返回 Line2D 艺术家
        class MockMark1(MockMark):
            def _legend_artist(self, variables, value, scales):
                return mpl.lines.Line2D([], [])

        # 定义 MockMark2 类，继承自 MockMark，用于返回 Patch 艺术家
        class MockMark2(MockMark):
            def _legend_artist(self, variables, value, scales):
                return mpl.patches.Patch()

        # 创建一个包含字符串的 Pandas Series 对象
        s = pd.Series(["a", "b", "a", "c"], name="s")
        # 创建一个 Plot 对象，使用给定的参数 xy 和 Series s 作为颜色参数，添加 MockMark1 和 MockMark2 对象，然后绘制图表
        p = Plot(**xy, color=s).add(MockMark1()).add(MockMark2()).plot()

        # 获取图表的图例对象
        legend, = p._figure.legends

        # 获取分类顺序的名称列表
        names = categorical_order(s)
        # 获取图例中的文本标签列表
        labels = [t.get_text() for t in legend.get_texts()]
        # 断言图例文本标签与分类顺序名称列表相同
        assert labels == names

        # 如果版本不早于 Matplotlib 3.5，进一步断言图例内容的艺术家类型和名称列表长度相同
        if not _version_predates(mpl, "3.5"):
            contents = legend.get_children()[0]
            assert len(contents.findobj(mpl.lines.Line2D)) == len(names)
            assert len(contents.findobj(mpl.patches.Patch)) == len(names)

    # 定义一个测试方法，用于测试三层图表
    def test_three_layers(self, xy):
        # 定义 MockMarkLine 类，继承自 MockMark，用于返回 Line2D 艺术家
        class MockMarkLine(MockMark):
            def _legend_artist(self, variables, value, scales):
                return mpl.lines.Line2D([], [])

        # 创建一个包含字符串的 Pandas Series 对象
        s = pd.Series(["a", "b", "a", "c"], name="s")
        # 创建一个 Plot 对象，使用给定的参数 xy 和 Series s 作为颜色参数
        p = Plot(**xy, color=s)
        # 添加三个 MockMarkLine 对象，并绘制图表
        for _ in range(3):
            p = p.add(MockMarkLine())
        p = p.plot()
        # 获取图表的图例文本对象
        texts = p._figure.legends[0].get_texts()
        # 断言图例文本对象的数量与 Series s 唯一值的数量相同
        assert len(texts) == len(s.unique())
    # 测试函数，验证在设置缩放时是否忽略了身份
    def test_identity_scale_ignored(self, xy):
        # 创建包含字符串的 Pandas Series 对象
        s = pd.Series(["r", "g", "b", "g"])
        # 创建 Plot 对象，添加 MockMark，并根据 Series 对象设置颜色，然后取消颜色缩放，绘制图形
        p = Plot(**xy).add(MockMark(), color=s).scale(color=None).plot()
        # 断言图形的图例内容为空
        assert not p._legend_contents

    # 测试函数，验证在 add 方法中是否可以抑制图例的显示
    def test_suppression_in_add_method(self, xy):
        # 创建包含字符串的 Pandas Series 对象
        s = pd.Series(["a", "b", "a", "c"], name="s")
        # 创建 Plot 对象，添加 MockMark，并根据 Series 对象设置颜色，并且设置 legend=False，绘制图形
        p = Plot(**xy).add(MockMark(), color=s, legend=False).plot()
        # 断言图形的图例内容为空
        assert not p._legend_contents

    # 测试函数，验证匿名标题是否可以被正确设置为空字符串
    def test_anonymous_title(self, xy):
        # 创建 Plot 对象，设置颜色并添加 MockMark，绘制图形
        p = Plot(**xy, color=["a", "b", "c", "d"]).add(MockMark()).plot()
        # 获取图形的图例对象
        legend, = p._figure.legends
        # 断言图例的标题文本是否为空字符串
        assert legend.get_title().get_text() == ""

    # 测试函数，验证自定义 Mark 类中是否可以完全禁用图例
    def test_legendless_mark(self, xy):
        # 定义一个不包含图例的 MockMark 子类
        class NoLegendMark(MockMark):
            def _legend_artist(self, variables, value, scales):
                return None
        # 创建 Plot 对象，设置颜色并添加 NoLegendMark 类的实例，绘制图形
        p = Plot(**xy, color=["a", "b", "c", "d"]).add(NoLegendMark()).plot()
        # 断言图形没有图例
        assert not p._figure.legends

    # 测试函数，验证图例中的文本是否大于指定的偏移量
    def test_legend_has_no_offset(self, xy):
        # 根据 xy 字典中的 x 值创建一个颜色数组
        color = np.add(xy["x"], 1e8)
        # 创建 Plot 对象，设置颜色并添加 MockMark，绘制图形
        p = Plot(**xy, color=color).add(MockMark()).plot()
        # 获取图形的第一个图例对象
        legend = p._figure.legends[0]
        # 断言图例中的文本是否存在
        assert legend.texts
        # 验证图例文本中的每个值是否大于 1e7
        for text in legend.texts:
            assert float(text.get_text()) > 1e7

    # 测试函数，验证多层叠加的图例是否正确显示
    def test_layer_legend(self, xy):
        # 创建 Plot 对象，依次添加 MockMark，并为每个图层设置不同的标签，绘制图形
        p = Plot(**xy).add(MockMark(), label="a").add(MockMark(), label="b").plot()
        # 获取图形的第一个图例对象
        legend = p._figure.legends[0]
        # 断言图例中的文本是否存在
        assert legend.texts
        # 验证图例文本与预期的标签列表是否一致
        for text, expected in zip(legend.texts, "ab"):
            assert text.get_text() == expected

    # 测试函数，验证带有比例尺图例的多层叠加图是否正确显示
    def test_layer_legend_with_scale_legend(self, xy):
        # 创建包含字符串的 Pandas Series 对象
        s = pd.Series(["a", "b", "a", "c"], name="s")
        # 创建 Plot 对象，设置颜色并添加 MockMark，为整体设置标签，绘制图形
        p = Plot(**xy, color=s).add(MockMark(), label="x").plot()
        # 获取图形的第一个图例对象
        legend = p._figure.legends[0]
        # 获取图例中的文本列表
        texts = [t.get_text() for t in legend.findobj(mpl.text.Text)]
        # 断言图例文本中包含"x"，并且包含 Series 中的每个唯一值
        assert "x" in texts
        for val in s.unique():
            assert val in texts

    # 测试函数，验证带有标题的多层叠加图例是否正确显示
    def test_layer_legend_title(self, xy):
        # 创建 Plot 对象，添加 MockMark，并设置图例的标题为"layer"，绘制图形
        p = Plot(**xy).add(MockMark(), label="x").label(legend="layer").plot()
        # 断言图形的第一个图例对象的标题文本是否为"layer"
        assert p._figure.legends[0].get_title().get_text() == "layer"
class TestDefaultObject:
    
    def test_default_repr(self):
        # 检查默认对象的字符串表示是否为 "<default>"
        assert repr(Default()) == "<default>"


class TestThemeConfig:
    
    @pytest.fixture(autouse=True)
    def reset_config(self):
        # 重置配置的 pytest fixture，确保每个测试方法运行前后状态一致
        yield
        # 在测试方法运行完毕后执行重置 Plot 配置主题
        Plot.config.theme.reset()

    def test_default(self):
        # 测试默认主题设置下绘图的轴面颜色
        p = Plot().plot()
        ax = p._figure.axes[0]
        expected = Plot.config.theme["axes.facecolor"]
        assert mpl.colors.same_color(ax.get_facecolor(), expected)

    def test_setitem(self):
        # 测试通过设置项方式修改主题的轴面颜色
        color = "#CCC"
        Plot.config.theme["axes.facecolor"] = color
        p = Plot().plot()
        ax = p._figure.axes[0]
        assert mpl.colors.same_color(ax.get_facecolor(), color)

    def test_update(self):
        # 测试通过更新方法修改主题的轴面颜色
        color = "#DDD"
        Plot.config.theme.update({"axes.facecolor": color})
        p = Plot().plot()
        ax = p._figure.axes[0]
        assert mpl.colors.same_color(ax.get_facecolor(), color)

    def test_reset(self):
        # 测试重置主题设置后轴面颜色是否恢复到原始值
        orig = Plot.config.theme["axes.facecolor"]
        Plot.config.theme.update({"axes.facecolor": "#EEE"})
        Plot.config.theme.reset()
        p = Plot().plot()
        ax = p._figure.axes[0]
        assert mpl.colors.same_color(ax.get_facecolor(), orig)

    def test_copy(self):
        # 测试主题配置的复制功能
        key, val = "axes.facecolor", ".95"
        orig = Plot.config.theme[key]
        theme = Plot.config.theme.copy()
        theme.update({key: val})
        assert Plot.config.theme[key] == orig

    def test_html_repr(self):
        # 测试主题配置对象的 HTML 表示
        res = Plot.config.theme._repr_html_()
        for tag in ["div", "table", "tr", "td"]:
            assert res.count(f"<{tag}") == res.count(f"</{tag}")

        for key in Plot.config.theme:
            assert f"<td>{key}:</td>" in res


class TestDisplayConfig:
    
    @pytest.fixture(autouse=True)
    def reset_config(self):
        # 重置显示配置的 pytest fixture，确保每个测试方法运行前后状态一致
        yield
        # 在测试方法运行完毕后执行更新显示配置为默认值
        Plot.config.display.update(PlotConfig().display)

    def test_png_format(self):
        # 测试 PNG 格式的显示配置
        Plot.config.display["format"] = "png"

        assert Plot()._repr_svg_() is None
        assert Plot().plot()._repr_svg_() is None

        def assert_valid_png(p):
            # 辅助函数：验证 PNG 图片格式
            data, metadata = p._repr_png_()
            img = Image.open(io.BytesIO(data))
            assert img.format == "PNG"
            assert sorted(metadata) == ["height", "width"]

        assert_valid_png(Plot())
        assert_valid_png(Plot().plot())

    def test_svg_format(self):
        # 测试 SVG 格式的显示配置
        Plot.config.display["format"] = "svg"

        assert Plot()._repr_png_() is None
        assert Plot().plot()._repr_png_() is None

        def assert_valid_svg(p):
            # 辅助函数：验证 SVG 图片格式
            res = p._repr_svg_()
            root = xml.etree.ElementTree.fromstring(res)
            assert root.tag == "{http://www.w3.org/2000/svg}svg"

        assert_valid_svg(Plot())
        assert_valid_svg(Plot().plot())
    # 测试 PNG 图片缩放功能
    def test_png_scaling(self):

        # 设置绘图配置的缩放比例为 1.0
        Plot.config.display["scaling"] = 1.
        # 调用绘图对象的方法获取 PNG 图片数据和元数据
        res1, meta1 = Plot()._repr_png_()

        # 设置绘图配置的缩放比例为 0.5
        Plot.config.display["scaling"] = .5
        # 再次调用绘图对象的方法获取 PNG 图片数据
        res2, meta2 = Plot()._repr_png_()

        # 断言两次获取的 PNG 图片元数据宽度应该相差一半
        assert meta1["width"] / 2 == meta2["width"]
        # 断言两次获取的 PNG 图片元数据高度应该相差一半
        assert meta1["height"] / 2 == meta2["height"]

        # 使用 PIL 库打开第一张 PNG 图片数据
        img1 = Image.open(io.BytesIO(res1))
        # 使用 PIL 库打开第二张 PNG 图片数据
        img2 = Image.open(io.BytesIO(res2))
        # 断言两张 PNG 图片的尺寸应该相同
        assert img1.size == img2.size

    # 测试 SVG 图片缩放功能
    def test_svg_scaling(self):

        # 设置绘图配置的输出格式为 SVG
        Plot.config.display["format"] = "svg"

        # 设置绘图配置的缩放比例为 1.0
        Plot.config.display["scaling"] = 1.
        # 调用绘图对象的方法获取 SVG 图片数据
        res1 = Plot()._repr_svg_()

        # 设置绘图配置的缩放比例为 0.5
        Plot.config.display["scaling"] = .5
        # 再次调用绘图对象的方法获取 SVG 图片数据
        res2 = Plot()._repr_svg_()

        # 使用 xml.etree.ElementTree 解析第一张 SVG 图片数据
        root1 = xml.etree.ElementTree.fromstring(res1)
        # 使用 xml.etree.ElementTree 解析第二张 SVG 图片数据
        root2 = xml.etree.ElementTree.fromstring(res2)

        # 定义函数从 SVG 元素根节点中获取指定维度的值
        def getdim(root, dim):
            return float(root.attrib[dim][:-2])

        # 断言两张 SVG 图片的宽度应该相差一半
        assert getdim(root1, "width") / 2 == getdim(root2, "width")
        # 断言两张 SVG 图片的高度应该相差一半
        assert getdim(root1, "height") / 2 == getdim(root2, "height")

    # 测试 PNG 图片高分辨率模式功能
    def test_png_hidpi(self):

        # 调用绘图对象的方法获取 PNG 图片数据和元数据
        res1, meta1 = Plot()._repr_png_()

        # 设置绘图配置的高分辨率显示模式为关闭
        Plot.config.display["hidpi"] = False
        # 再次调用绘图对象的方法获取 PNG 图片数据和元数据
        res2, meta2 = Plot()._repr_png_()

        # 断言两次获取的 PNG 图片元数据宽度应该相等
        assert meta1["width"] == meta2["width"]
        # 断言两次获取的 PNG 图片元数据高度应该相等
        assert meta1["height"] == meta2["height"]

        # 使用 PIL 库打开第一张 PNG 图片数据
        img1 = Image.open(io.BytesIO(res1))
        # 使用 PIL 库打开第二张 PNG 图片数据
        img2 = Image.open(io.BytesIO(res2))
        # 断言两张 PNG 图片的宽度应该相差一半
        assert img1.size[0] // 2 == img2.size[0]
        # 断言两张 PNG 图片的高度应该相差一半
        assert img1.size[1] // 2 == img2.size[1]
```