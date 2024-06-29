# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_category.py`

```py
"""Catch all for categorical functions"""
# 引入警告模块，用于可能出现的警告处理
import warnings

# 引入 pytest 测试框架
import pytest
# 引入 numpy 库，用于数值计算
import numpy as np

# 引入 matplotlib 库及其子模块
import matplotlib as mpl
# 从 matplotlib.axes 模块中引入 Axes 类
from matplotlib.axes import Axes
# 引入 matplotlib.pyplot 模块，并重命名为 plt
import matplotlib.pyplot as plt
# 引入 matplotlib.category 模块
import matplotlib.category as cat
# 从 matplotlib.testing.decorators 模块中引入 check_figures_equal 装饰器
from matplotlib.testing.decorators import check_figures_equal


class TestUnitData:
    # 定义测试用例列表
    test_cases = [('single', (["hello world"], [0])),
                  ('unicode', (["Здравствуйте мир"], [0])),
                  ('mixed', (['A', "np.nan", 'B', "3.14", "мир"],
                             [0, 1, 2, 3, 4]))]
    # 解构赋值得到测试用例的标识和数据
    ids, data = zip(*test_cases)

    # 参数化测试方法，参数为 data，ids 为测试用例标识
    @pytest.mark.parametrize("data, locs", data, ids=ids)
    def test_unit(self, data, locs):
        # 创建 cat.UnitData 对象
        unit = cat.UnitData(data)
        # 断言 _mapping 的键与 data 相同
        assert list(unit._mapping.keys()) == data
        # 断言 _mapping 的值与 locs 相同
        assert list(unit._mapping.values()) == locs

    # 测试更新方法
    def test_update(self):
        # 初始化数据和位置
        data = ['a', 'd']
        locs = [0, 1]

        # 更新数据和位置
        data_update = ['b', 'd', 'e']
        unique_data = ['a', 'd', 'b', 'e']
        updated_locs = [0, 1, 2, 3]

        # 创建 cat.UnitData 对象
        unit = cat.UnitData(data)
        # 断言 _mapping 的键与 data 相同
        assert list(unit._mapping.keys()) == data
        # 断言 _mapping 的值与 locs 相同
        assert list(unit._mapping.values()) == locs

        # 更新 unit 对象
        unit.update(data_update)
        # 断言 _mapping 的键为更新后的唯一数据
        assert list(unit._mapping.keys()) == unique_data
        # 断言 _mapping 的值为更新后的位置
        assert list(unit._mapping.values()) == updated_locs

    # 失败的测试用例列表
    failing_test_cases = [("number", 3.14), ("nan", np.nan),
                          ("list", [3.14, 12]), ("mixed type", ["A", 2])]

    # 解构赋值得到失败测试用例的标识和数据
    fids, fdata = zip(*test_cases)

    # 参数化测试方法，参数为 fdata，ids 为 fids
    @pytest.mark.parametrize("fdata", fdata, ids=fids)
    def test_non_string_fails(self, fdata):
        # 使用 pytest.raises 检测 TypeError 异常
        with pytest.raises(TypeError):
            cat.UnitData(fdata)

    # 参数化测试方法，参数为 fdata，ids 为 fids
    @pytest.mark.parametrize("fdata", fdata, ids=fids)
    def test_non_string_update_fails(self, fdata):
        # 创建 cat.UnitData 对象
        unitdata = cat.UnitData()
        # 使用 pytest.raises 检测 TypeError 异常
        with pytest.raises(TypeError):
            unitdata.update(fdata)


class FakeAxis:
    # 初始化 FakeAxis 类，接受 units 参数
    def __init__(self, units):
        self.units = units


class TestStrCategoryConverter:
    """
    Based on the pandas conversion and factorization tests:

    ref: /pandas/tseries/tests/test_converter.py
         /pandas/tests/test_algos.py:TestFactorize
    """
    # 定义测试用例列表
    test_cases = [("unicode", ["Здравствуйте мир"]),
                  ("ascii", ["hello world"]),
                  ("single", ['a', 'b', 'c']),
                  ("integer string", ["1", "2"]),
                  ("single + values>10", ["A", "B", "C", "D", "E", "F", "G",
                                          "H", "I", "J", "K", "L", "M", "N",
                                          "O", "P", "Q", "R", "S", "T", "U",
                                          "V", "W", "X", "Y", "Z"])]

    # 解构赋值得到测试用例的标识和数据
    ids, values = zip(*test_cases)

    # 失败的测试用例列表
    failing_test_cases = [("mixed", [3.14, 'A', np.inf]),
                          ("string integer", ['42', 42])]

    # 解构赋值得到失败测试用例的标识和数据
    fids, fvalues = zip(*failing_test_cases)

    # 自动使用的 pytest fixture
    @pytest.fixture(autouse=True)
    # 初始化 mock_axis 方法，设置类中的 cc 和 unit 属性，创建一个 FakeAxis 实例
    def mock_axis(self, request):
        self.cc = cat.StrCategoryConverter()  # 初始化 StrCategoryConverter 实例给 self.cc
        # self.unit 应该被一个真实的模拟单位替换
        self.unit = cat.UnitData()  # 初始化 UnitData 实例给 self.unit
        self.ax = FakeAxis(self.unit)  # 使用 self.unit 创建 FakeAxis 实例给 self.ax

    # 使用 pytest.mark.parametrize 注解的测试方法，测试 cc.convert 方法
    @pytest.mark.parametrize("vals", values, ids=ids)
    def test_convert(self, vals):
        np.testing.assert_allclose(self.cc.convert(vals, self.ax.units,
                                                   self.ax),
                                   range(len(vals)))  # 断言 cc.convert 方法的结果与 range(len(vals)) 相近

    # 使用 pytest.mark.parametrize 注解的测试方法，测试 cc.convert 方法处理单个字符串的情况
    @pytest.mark.parametrize("value", ["hi", "мир"], ids=["ascii", "unicode"])
    def test_convert_one_string(self, value):
        assert self.cc.convert(value, self.unit, self.ax) == 0  # 断言 cc.convert 方法处理单个字符串时的结果为 0

    # 使用 pytest.mark.parametrize 注解的测试方法，测试 cc.convert 方法处理失败的情况
    @pytest.mark.parametrize("fvals", fvalues, ids=fids)
    def test_convert_fail(self, fvals):
        with pytest.raises(TypeError):  # 断言 cc.convert 方法处理失败情况时会抛出 TypeError 异常
            self.cc.convert(fvals, self.unit, self.ax)

    # 测试 cc.axisinfo 方法
    def test_axisinfo(self):
        axis = self.cc.axisinfo(self.unit, self.ax)  # 调用 cc.axisinfo 方法返回的结果赋给 axis
        assert isinstance(axis.majloc, cat.StrCategoryLocator)  # 断言 axis.majloc 是 StrCategoryLocator 类的实例
        assert isinstance(axis.majfmt, cat.StrCategoryFormatter)  # 断言 axis.majfmt 是 StrCategoryFormatter 类的实例

    # 测试 cc.default_units 方法
    def test_default_units(self):
        assert isinstance(self.cc.default_units(["a"], self.ax), cat.UnitData)  # 断言 cc.default_units 方法返回的结果是 UnitData 类的实例
# 定义包含可用于绘图的函数列表
PLOT_LIST = [Axes.scatter, Axes.plot, Axes.bar]
# 定义与每个绘图函数对应的标识符列表
PLOT_IDS = ["scatter", "plot", "bar"]

# 定义一个测试类 TestStrCategoryLocator
class TestStrCategoryLocator:
    # 定义测试方法 test_StrCategoryLocator
    def test_StrCategoryLocator(self):
        # 创建一个包含整数的列表作为位置信息
        locs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # 创建一个 UnitData 对象，用于测试目的
        unit = cat.UnitData([str(j) for j in locs])
        # 使用 UnitData 对象的映射数据创建 StrCategoryLocator 对象
        ticks = cat.StrCategoryLocator(unit._mapping)
        # 断言获取到的刻度值与预期的位置信息列表 locs 相等
        np.testing.assert_array_equal(ticks.tick_values(None, None), locs)

    # 使用参数化测试装饰器，循环测试 PLOT_LIST 中的每个绘图函数
    @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
    # 定义测试方法 test_StrCategoryLocatorPlot，接受一个绘图函数作为参数
    def test_StrCategoryLocatorPlot(self, plotter):
        # 创建一个图形对象的子图
        ax = plt.figure().subplots()
        # 使用给定的绘图函数 plotter 绘制图形
        plotter(ax, [1, 2, 3], ["a", "b", "c"])
        # 断言 y 轴主刻度定位器的值与预期的范围内的整数相等
        np.testing.assert_array_equal(ax.yaxis.major.locator(), range(3))


# 定义一个测试类 TestStrCategoryFormatter
class TestStrCategoryFormatter:
    # 定义测试案例列表，包括 ASCII 和 Unicode 字符串
    test_cases = [("ascii", ["hello", "world", "hi"]),
                  ("unicode", ["Здравствуйте", "привет"])]
    # 解压测试案例为标识符列表和数据列表
    ids, cases = zip(*test_cases)

    # 使用参数化测试装饰器，循环测试 cases 列表中的每个数据作为 ydata
    @pytest.mark.parametrize("ydata", cases, ids=ids)
    # 定义测试方法 test_StrCategoryFormatter，接受一个数据列表作为参数
    def test_StrCategoryFormatter(self, ydata):
        # 创建一个 UnitData 对象，用于测试目的
        unit = cat.UnitData(ydata)
        # 使用 UnitData 对象的映射数据创建 StrCategoryFormatter 对象
        labels = cat.StrCategoryFormatter(unit._mapping)
        # 遍历每个数据和对应索引，断言标签函数给出的标签与数据相等
        for i, d in enumerate(ydata):
            assert labels(i, i) == d
            assert labels(i, None) == d

    # 使用参数化测试装饰器，循环测试 cases 列表中的每个数据作为 ydata
    @pytest.mark.parametrize("ydata", cases, ids=ids)
    # 使用参数化测试装饰器，循环测试 PLOT_LIST 中的每个绘图函数
    @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
    # 定义测试方法 test_StrCategoryFormatterPlot，接受一个数据列表和绘图函数作为参数
    def test_StrCategoryFormatterPlot(self, ydata, plotter):
        # 创建一个图形对象的子图
        ax = plt.figure().subplots()
        # 使用给定的绘图函数 plotter 绘制图形
        plotter(ax, range(len(ydata)), ydata)
        # 遍历每个数据和对应索引，断言 y 轴主刻度格式化器给出的标签与数据相等
        for i, d in enumerate(ydata):
            assert ax.yaxis.major.formatter(i) == d
        # 断言超出索引范围的标签为空字符串
        assert ax.yaxis.major.formatter(i+1) == ""


# 定义一个用于测试轴的辅助函数 axis_test
def axis_test(axis, labels):
    # 获取主刻度的位置信息列表
    ticks = list(range(len(labels)))
    # 断言轴对象的主刻度位置信息与预期的 ticks 列表相等
    np.testing.assert_array_equal(axis.get_majorticklocs(), ticks)
    # 获取图形标签列表，使用 StrCategoryFormatter 的 _text 方法解码字节为 UTF-8
    graph_labels = [axis.major.formatter(i, i) for i in ticks]
    # 断言图形标签与标签列表中的每个标签相等
    assert graph_labels == [cat.StrCategoryFormatter._text(l) for l in labels]
    # 断言轴对象的单位映射键与标签列表中的每个标签相等
    assert list(axis.units._mapping.keys()) == [l for l in labels]
    # 断言轴对象的单位映射值与 ticks 列表中的每个位置相等
    assert list(axis.units._mapping.values()) == ticks


# 定义一个测试类 TestPlotBytes
class TestPlotBytes:
    # 定义字节数据的测试案例，包括字符串列表、字节列表和字节 ndarray
    bytes_cases = [('string list', ['a', 'b', 'c']),
                   ('bytes list', [b'a', b'b', b'c']),
                   ('bytes ndarray', np.array([b'a', b'b', b'c']))]

    # 解压测试案例为标识符列表和数据列表
    bytes_ids, bytes_data = zip(*bytes_cases)

    # 使用参数化测试装饰器，循环测试 PLOT_LIST 中的每个绘图函数
    @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
    # 使用参数化测试装饰器，循环测试 bytes_data 列表中的每个数据作为 bdata
    @pytest.mark.parametrize("bdata", bytes_data, ids=bytes_ids)
    # 定义测试方法 test_plot_bytes，接受一个绘图函数和字节数据作为参数
    def test_plot_bytes(self, plotter, bdata):
        # 创建一个图形对象的子图
        ax = plt.figure().subplots()
        # 创建一个计数数组
        counts = np.array([4, 6, 5])
        # 使用给定的绘图函数 plotter 绘制图形
        plotter(ax, bdata, counts)
        # 测试辅助函数 axis_test 来测试 x 轴
        axis_test(ax.xaxis, bdata)


# 定义一个测试类 TestPlotNumlike
class TestPlotNumlike:
    # 定义类似数字数据的测试案例，包括字符串列表、字符串 ndarray、字节列表和字节 ndarray
    numlike_cases = [('string list', ['1', '11', '3']),
                     ('string ndarray', np.array(['1', '11', '3'])),
                     ('bytes list', [b'1', b'11', b'3']),
                     ('bytes ndarray', np.array([b'1', b'11', b'3']))]
    # 解压测试案例为标识符列表和数据列表
    numlike_ids, numlike_data = zip(*numlike_cases)

    # 使用参数化测试装饰器，循环测试 PLOT_LIST 中的每个绘图函数
    @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
    # 使用 pytest 的参数化测试，参数包括 numlike_data 和 numlike_ids
    @pytest.mark.parametrize("ndata", numlike_data, ids=numlike_ids)
    # 定义测试函数 test_plot_numlike，接受参数 plotter 和 ndata
    def test_plot_numlike(self, plotter, ndata):
        # 创建一个新的图形并获取其子图对象
        ax = plt.figure().subplots()
        # 定义一个包含数值的 NumPy 数组 counts
        counts = np.array([4, 6, 5])
        # 调用 plotter 函数，绘制图形，传入 ax、ndata 和 counts 作为参数
        plotter(ax, ndata, counts)
        # 调用 axis_test 函数，对 x 轴进行单元测试，传入 ax.xaxis 和 ndata 作为参数
        axis_test(ax.xaxis, ndata)
class TestPlotTypes:
    # 使用参数化标记，针对 PLOT_LIST 中的每个绘图函数进行测试
    @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
    def test_plot_unicode(self, plotter):
        # 创建一个图形对象，并获取其子图
        ax = plt.figure().subplots()
        # 定义包含非英语字符的单词列表
        words = ['Здравствуйте', 'привет']
        # 使用指定的绘图函数在图形上绘制单词列表
        plotter(ax, words, [0, 1])
        # 对 x 轴进行测试，验证单词是否正确显示
        axis_test(ax.xaxis, words)

    # 定义测试数据的夹具
    @pytest.fixture
    def test_data(self):
        self.x = ["hello", "happy", "world"]
        self.xy = [2, 6, 3]
        self.y = ["Python", "is", "fun"]
        self.yx = [3, 4, 5]

    # 使用 test_data 夹具，并针对 PLOT_LIST 中的每个绘图函数进行参数化测试
    @pytest.mark.usefixtures("test_data")
    @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
    def test_plot_xaxis(self, test_data, plotter):
        # 创建一个图形对象，并获取其子图
        ax = plt.figure().subplots()
        # 使用指定的绘图函数在图形上绘制 self.x 和 self.xy 数据
        plotter(ax, self.x, self.xy)
        # 对 x 轴进行测试，验证数据是否正确显示
        axis_test(ax.xaxis, self.x)

    # 使用 test_data 夹具，并针对 PLOT_LIST 中的每个绘图函数进行参数化测试
    @pytest.mark.usefixtures("test_data")
    @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
    def test_plot_yaxis(self, test_data, plotter):
        # 创建一个图形对象，并获取其子图
        ax = plt.figure().subplots()
        # 使用指定的绘图函数在图形上绘制 self.yx 和 self.y 数据
        plotter(ax, self.yx, self.y)
        # 对 y 轴进行测试，验证数据是否正确显示
        axis_test(ax.yaxis, self.y)

    # 使用 test_data 夹具，并针对 PLOT_LIST 中的每个绘图函数进行参数化测试
    @pytest.mark.usefixtures("test_data")
    @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
    def test_plot_xyaxis(self, test_data, plotter):
        # 创建一个图形对象，并获取其子图
        ax = plt.figure().subplots()
        # 使用指定的绘图函数在图形上绘制 self.x 和 self.y 数据
        plotter(ax, self.x, self.y)
        # 对 x 和 y 轴进行测试，验证数据是否正确显示
        axis_test(ax.xaxis, self.x)
        axis_test(ax.yaxis, self.y)

    # 针对 PLOT_LIST 中的每个绘图函数进行参数化测试
    @pytest.mark.parametrize("plotter", PLOT_LIST, ids=PLOT_IDS)
    def test_update_plot(self, plotter):
        # 创建一个图形对象，并获取其子图
        ax = plt.figure().subplots()
        # 使用指定的绘图函数多次更新图形，每次使用不同的数据
        plotter(ax, ['a', 'b'], ['e', 'g'])
        plotter(ax, ['a', 'b', 'd'], ['f', 'a', 'b'])
        plotter(ax, ['b', 'c', 'd'], ['g', 'e', 'd'])
        # 对 x 和 y 轴进行测试，验证更新后的数据是否正确显示
        axis_test(ax.xaxis, ['a', 'b', 'd', 'c'])
        axis_test(ax.yaxis, ['e', 'g', 'f', 'a', 'b', 'd'])

    # 定义失败的测试用例列表
    failing_test_cases = [("mixed", ['A', 3.14]),
                          ("number integer", ['1', 1]),
                          ("string integer", ['42', 42]),
                          ("missing", ['12', np.nan])]

    # 将失败测试用例的标识和数据分离为两个列表
    fids, fvalues = zip(*failing_test_cases)

    # 定义绘图函数列表，包含散点图、柱状图和绘图，其中绘图标记为预期失败
    plotters = [Axes.scatter, Axes.bar,
                pytest.param(Axes.plot, marks=pytest.mark.xfail)]

    # 针对 plotters 中的每个绘图函数以及每个失败测试用例进行参数化测试
    @pytest.mark.parametrize("plotter", plotters)
    @pytest.mark.parametrize("xdata", fvalues, ids=fids)
    def test_mixed_type_exception(self, plotter, xdata):
        # 创建一个图形对象，并获取其子图
        ax = plt.figure().subplots()
        # 使用指定的绘图函数尝试在图形上绘制混合类型的数据，预期引发 TypeError 异常
        with pytest.raises(TypeError):
            plotter(ax, xdata, [1, 2])

    # 针对 plotters 中的每个绘图函数以及每个失败测试用例进行参数化测试
    @pytest.mark.parametrize("plotter", plotters)
    @pytest.mark.parametrize("xdata", fvalues, ids=fids)
    def test_mixed_type_update_exception(self, plotter, xdata):
        # 创建一个图形对象，并获取其子图
        ax = plt.figure().subplots()
        # 使用指定的绘图函数多次尝试在图形上绘制混合类型的数据，预期引发 TypeError 异常
        with pytest.raises(TypeError):
            plotter(ax, [0, 3], [1, 3])
            plotter(ax, xdata, [1, 2])


# 应用默认样式上下文，并使用 check_figures_equal 夹具进行测试，生成 PNG 扩展的图像
@mpl.style.context('default')
@check_figures_equal(extensions=["png"])
def test_overriding_units_in_plot(fig_test, fig_ref):
    from datetime import datetime

    # 定义多个日期时间对象
    t0 = datetime(2018, 3, 1)
    t1 = datetime(2018, 3, 2)
    t2 = datetime(2018, 3, 3)
    t3 = datetime(2018, 3, 4)
    # 创建测试图和参考图的子图
    ax_test = fig_test.subplots()
    ax_ref = fig_ref.subplots()
    # 遍历测试图和参考图的子图以及对应的参数字典
    for ax, kwargs in zip([ax_test, ax_ref],
                          ({}, dict(xunits=None, yunits=None))):
        # 在当前子图上绘制线条，使用传入的参数字典
        # 第一次调用绘图函数
        ax.plot([t0, t1], ["V1", "V2"], **kwargs)
        # 获取 x 轴和 y 轴的单位
        x_units = ax.xaxis.units
        y_units = ax.yaxis.units
        # 再次调用绘图函数，确保不会触发异常
        ax.plot([t2, t3], ["V1", "V2"], **kwargs)
        # 断言：检查 x 轴和 y 轴的单位属性是否保持不变
        assert x_units is ax.xaxis.units
        assert y_units is ax.yaxis.units
# 定义一个测试函数，用于确保没有在空数据上发出过时警告。参见 issue #22640。
def test_no_deprecation_on_empty_data():
    # 创建一个包含图表和轴对象的 subplot
    f, ax = plt.subplots()
    # 更新 x 轴的单位为列表 ["a", "b"]，此处主要是个例子，并不实际影响图表内容
    ax.xaxis.update_units(["a", "b"])
    # 绘制一个空的线图，即不包含任何数据点的图表
    ax.plot([], [])


# 定义一个测试函数，用于测试绘制直方图的功能
def test_hist():
    # 创建一个包含图表和轴对象的 subplot
    fig, ax = plt.subplots()
    # 使用指定的数据绘制直方图，数据为 ['a', 'b', 'a', 'c', 'ff']
    n, bins, patches = ax.hist(['a', 'b', 'a', 'c', 'ff'])
    # 断言直方图的数据形状应为 (10,)，即包含 10 个条形的高度数据
    assert n.shape == (10,)
    # 使用 np.testing.assert_allclose 来验证直方图的数据是否接近预期值 [2., 0., 0., 1., 0., 0., 1., 0., 0., 1.]
    np.testing.assert_allclose(n, [2., 0., 0., 1., 0., 0., 1., 0., 0., 1.])


# 定义一个测试函数，用于测试设置坐标轴范围的功能
def test_set_lim():
    # 创建一个包含图表和轴对象的 subplot
    f, ax = plt.subplots()
    # 绘制一个简单的线图，包含坐标点 ["a", "b", "c", "d"] 和对应的数值 [1, 2, 3, 4]
    ax.plot(["a", "b", "c", "d"], [1, 2, 3, 4])
    # 使用 warnings.catch_warnings() 来捕获警告信息
    with warnings.catch_warnings():
        # 设置 x 轴的限制范围为 ("b", "c")
        ax.set_xlim("b", "c")
```