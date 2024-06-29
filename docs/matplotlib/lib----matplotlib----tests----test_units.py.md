# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_units.py`

```
# 导入必要的模块和库
from datetime import datetime, timezone, timedelta  # 导入日期时间相关的类和函数
import platform  # 导入平台信息模块
from unittest.mock import MagicMock  # 导入用于模拟对象的 MagicMock 类

import matplotlib.pyplot as plt  # 导入 matplotlib 的绘图模块
from matplotlib.testing.decorators import check_figures_equal, image_comparison  # 导入用于测试的装饰器和图像对比函数
import matplotlib.units as munits  # 导入 matplotlib 的单位处理模块
from matplotlib.category import UnitData  # 导入 matplotlib 的单位数据处理模块
import numpy as np  # 导入数值计算库 numpy
import pytest  # 导入 pytest 测试框架


# 基础类，封装了带单位的 numpy 数组
class Quantity:
    def __init__(self, data, units):
        self.magnitude = data  # 数组数据
        self.units = units  # 数组单位

    # 将数据转换到新的单位
    def to(self, new_units):
        factors = {('hours', 'seconds'): 3600, ('minutes', 'hours'): 1 / 60,
                   ('minutes', 'seconds'): 60, ('feet', 'miles'): 1 / 5280.,
                   ('feet', 'inches'): 12, ('miles', 'inches'): 12 * 5280}
        if self.units != new_units:
            mult = factors[self.units, new_units]
            return Quantity(mult * self.magnitude, new_units)
        else:
            return Quantity(self.magnitude, self.units)

    # 返回对象的浅拷贝
    def __copy__(self):
        return Quantity(self.magnitude, self.units)

    # 获取对象的属性
    def __getattr__(self, attr):
        return getattr(self.magnitude, attr)

    # 获取对象的子项
    def __getitem__(self, item):
        if np.iterable(self.magnitude):
            return Quantity(self.magnitude[item], self.units)
        else:
            return Quantity(self.magnitude, self.units)

    # 将对象转换为 numpy 数组
    def __array__(self):
        return np.asarray(self.magnitude)


@pytest.fixture
def quantity_converter():
    # 创建转换接口的实例，并进行方法调用的模拟
    qc = munits.ConversionInterface()

    def convert(value, unit, axis):
        if hasattr(value, 'units'):
            return value.to(unit).magnitude
        elif np.iterable(value):
            try:
                return [v.to(unit).magnitude for v in value]
            except AttributeError:
                return [Quantity(v, axis.get_units()).to(unit).magnitude
                        for v in value]
        else:
            return Quantity(value, axis.get_units()).to(unit).magnitude

    def default_units(value, axis):
        if hasattr(value, 'units'):
            return value.units
        elif np.iterable(value):
            for v in value:
                if hasattr(v, 'units'):
                    return v.units
            return None

    # 对转换接口的方法进行模拟
    qc.convert = MagicMock(side_effect=convert)
    qc.axisinfo = MagicMock(side_effect=lambda u, a:
                            munits.AxisInfo(label=u, default_limits=(0, 100)))
    qc.default_units = MagicMock(side_effect=default_units)
    return qc


# 测试转换机制是否正常工作，适用于作为 numpy 数组外观的类（如 pint）
@image_comparison(['plot_pint.png'], style='mpl20',
                  tol=0 if platform.machine() == 'x86_64' else 0.03)
def test_numpy_facade(quantity_converter):
    # 使用先前的默认设置以匹配现有的基准图像
    plt.rcParams['axes.formatter.limits'] = -7, 7

    # 注册类
    # 将 Quantity 类型注册到 munits.registry 中，并设置其对应的单位转换器为 quantity_converter
    munits.registry[Quantity] = quantity_converter

    # 创建两个 Quantity 对象 y 和 x，分别表示里程和时间
    y = Quantity(np.linspace(0, 30), 'miles')
    x = Quantity(np.linspace(0, 5), 'hours')

    # 创建图表和轴对象
    fig, ax = plt.subplots()
    # 调整图表布局，使左侧留出空间以容纳标签
    fig.subplots_adjust(left=0.15)
    # 绘制 y vs x，线条颜色为 'tab:blue'
    ax.plot(x, y, 'tab:blue')
    # 添加一条水平线，高度为 26400 英尺，颜色为 'tab:red'
    ax.axhline(Quantity(26400, 'feet'), color='tab:red')
    # 添加一条垂直线，位置为 120 分钟，颜色为 'tab:green'
    ax.axvline(Quantity(120, 'minutes'))
    # 设置 y 轴的单位为英寸
    ax.yaxis.set_units('inches')
    # 设置 x 轴的单位为秒
    ax.xaxis.set_units('seconds')

    # 断言确保 quantity_converter 的方法 convert、axisinfo 和 default_units 都被调用过
    assert quantity_converter.convert.called
    assert quantity_converter.axisinfo.called
    assert quantity_converter.default_units.called
# Tests gh-8908
# 定义一个测试函数，用于比较生成的图像是否符合预期
@image_comparison(['plot_masked_units.png'], remove_text=True, style='mpl20',
                  tol=0 if platform.machine() == 'x86_64' else 0.02)
def test_plot_masked_units():
    # 创建一维数组数据，范围从-5到5
    data = np.linspace(-5, 5)
    # 创建一个带掩码的 masked array，掩盖条件为数据在(-2, 2)之间
    data_masked = np.ma.array(data, mask=(data > -2) & (data < 2))
    # 创建一个带单位的 Quantity 对象，单位为米
    data_masked_units = Quantity(data_masked, 'meters')

    # 创建图形和子图
    fig, ax = plt.subplots()
    # 在子图上绘制带单位的数据
    ax.plot(data_masked_units)


# 定义一个测试函数，检查在设置空集合限制时是否能正常工作
def test_empty_set_limits_with_units(quantity_converter):
    # 注册 Quantity 类到单位转换器
    munits.registry[Quantity] = quantity_converter

    # 创建图形和子图
    fig, ax = plt.subplots()
    # 设置 X 轴的限制，单位为米
    ax.set_xlim(Quantity(-1, 'meters'), Quantity(6, 'meters'))
    # 设置 Y 轴的限制，单位为小时
    ax.set_ylim(Quantity(-1, 'hours'), Quantity(16, 'hours'))


# 定义一个测试函数，用于比较生成的图像是否符合预期
@image_comparison(['jpl_bar_units.png'],
                  savefig_kwarg={'dpi': 120}, style='mpl20')
def test_jpl_bar_units():
    # 导入 JPL 单位测试模块，并注册单位
    import matplotlib.testing.jpl_units as units
    units.register()

    # 创建一个持续时间对象，单位为地球时间秒
    day = units.Duration("ET", 24.0 * 60.0 * 60.0)
    # 创建 X 轴数据，单位为公里
    x = [0 * units.km, 1 * units.km, 2 * units.km]
    # 创建条形宽度数据，单位为地球时间天
    w = [1 * day, 2 * day, 3 * day]
    # 创建一个 JPL 时间对象，指定日期为 2009 年 4 月 26 日
    b = units.Epoch("ET", dt=datetime(2009, 4, 26))
    
    # 创建图形和子图
    fig, ax = plt.subplots()
    # 在子图上绘制条形图，底部起点为 b
    ax.bar(x, w, bottom=b)
    # 设置 Y 轴的限制，范围从 b - 1 天到 b + w[-1] + 1.001 天
    ax.set_ylim([b - 1 * day, b + w[-1] + (1.001) * day])


# 定义一个测试函数，用于比较生成的图像是否符合预期
@image_comparison(['jpl_barh_units.png'],
                  savefig_kwarg={'dpi': 120}, style='mpl20')
def test_jpl_barh_units():
    # 导入 JPL 单位测试模块，并注册单位
    import matplotlib.testing.jpl_units as units
    units.register()

    # 创建一个持续时间对象，单位为地球时间秒
    day = units.Duration("ET", 24.0 * 60.0 * 60.0)
    # 创建 Y 轴数据，单位为公里
    x = [0 * units.km, 1 * units.km, 2 * units.km]
    # 创建条形宽度数据，单位为地球时间天
    w = [1 * day, 2 * day, 3 * day]
    # 创建一个 JPL 时间对象，指定日期为 2009 年 4 月 26 日
    b = units.Epoch("ET", dt=datetime(2009, 4, 26))

    # 创建图形和子图
    fig, ax = plt.subplots()
    # 在子图上绘制水平条形图，左侧起点为 b
    ax.barh(x, w, left=b)
    # 设置 X 轴的限制，范围从 b - 1 天到 b + w[-1] + 1.001 天
    ax.set_xlim([b - 1 * day, b + w[-1] + (1.001) * day])


# 定义一个测试函数，用于检查 JPL 时间单位的一致性
def test_jpl_datetime_units_consistent():
    # 导入 JPL 单位测试模块，并注册单位
    import matplotlib.testing.jpl_units as units
    units.register()

    # 创建一个日期时间对象
    dt = datetime(2009, 4, 26)
    # 创建一个 JPL 时间对象，指定日期为 2009 年 4 月 26 日
    jpl = units.Epoch("ET", dt=dt)
    # 获取日期时间对象的转换器并转换
    dt_conv = munits.registry.get_converter(dt).convert(dt, None, None)
    # 获取 JPL 时间对象的转换器并转换
    jpl_conv = munits.registry.get_converter(jpl).convert(jpl, None, None)
    # 断言日期时间对象和 JPL 时间对象的转换结果是否相等
    assert dt_conv == jpl_conv


# 定义一个测试函数，检查在绘制空数组时是否能正常工作
def test_empty_arrays():
    # 检查使用日期时间的空数组绘制散点图是否正常工作
    plt.scatter(np.array([], dtype='datetime64[ns]'), np.array([]))


# 定义一个测试函数，检查在绘制散点图时遇到被掩盖元素时是否能正常工作
def test_scatter_element0_masked():
    # 创建日期时间数组
    times = np.arange('2005-02', '2005-03', dtype='datetime64[D]')
    # 创建 Y 轴数据，其中第一个元素被设置为 NaN
    y = np.arange(len(times), dtype=float)
    y[0] = np.nan
    # 创建图形和子图
    fig, ax = plt.subplots()
    # 在子图上绘制散点图
    ax.scatter(times, y)
    # 刷新图形
    fig.canvas.draw()


# 定义一个测试函数，检查在绘制误差条时使用混合单位是否能正常工作
def test_errorbar_mixed_units():
    # 创建 X 轴数据，范围从 0 到 9
    x = np.arange(10)
    # 创建 Y 轴数据，每个元素为不同的日期时间对象
    y = [datetime(2020, 5, i * 2 + 1) for i in x]
    # 创建图形和子图
    fig, ax = plt.subplots()
    # 在子图上绘制带有时间间隔的误差条
    ax.errorbar(x, y, timedelta(days=0.5))
    # 刷新图形
    fig.canvas.draw()


# 定义一个测试函数，用于检查子类化后的日期时间对象是否能正常绘制
@check_figures_equal(extensions=["png"])
def test_subclass(fig_test, fig_ref):
    # 定义一个继承自 datetime 的子类
    class subdate(datetime):
        pass

    # 在测试图和参考图上创建子图并绘制
    fig_test.subplots().plot(subdate(2000, 1, 1), 0, "o")
    fig_ref.subplots().plot(datetime(2000, 1, 1), 0, "o")


# 定义一个测试函数，检查在共享轴上使用 Quantity 对象是否能正常工作
def test_shared_axis_quantity(quantity_converter):
    # 将 Quantity 类注册到 munits.registry 中，使用 quantity_converter 进行转换
    munits.registry[Quantity] = quantity_converter

    # 创建一个 Quantity 对象 x，表示时间单位为 "hours"
    x = Quantity(np.linspace(0, 1, 10), "hours")

    # 创建两个 Quantity 对象 y1 和 y2，分别表示长度单位为 "feet"
    y1 = Quantity(np.linspace(1, 2, 10), "feet")
    y2 = Quantity(np.linspace(3, 4, 10), "feet")

    # 创建包含两个子图的图形对象 fig，共享 x 轴和 y 轴
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all', sharey='all')

    # 在第一个子图 ax1 上绘制 x 和 y1 的关系图
    ax1.plot(x, y1)

    # 在第二个子图 ax2 上绘制 x 和 y2 的关系图
    ax2.plot(x, y2)

    # 断言两个子图的 x 轴单位都为 "hours"
    assert ax1.xaxis.get_units() == ax2.xaxis.get_units() == "hours"

    # 断言两个子图的 y 轴单位都为 "feet"
    assert ax1.yaxis.get_units() == ax2.yaxis.get_units() == "feet"

    # 将第一个子图 ax1 的 x 轴单位设置为 "seconds"
    ax1.xaxis.set_units("seconds")

    # 将第二个子图 ax2 的 y 轴单位设置为 "inches"
    ax2.yaxis.set_units("inches")

    # 断言设置后，两个子图的 x 轴单位都为 "seconds"
    assert ax1.xaxis.get_units() == ax2.xaxis.get_units() == "seconds"

    # 断言设置后，两个子图的 y 轴单位都为 "inches"
    assert ax1.yaxis.get_units() == ax2.yaxis.get_units() == "inches"
```python`
def test_shared_axis_datetime():
    # datetime uses dates.DateConverter
    # 创建一个列表，包含 2020 年每个月的 UTC 时间
    y1 = [datetime(2020, i, 1, tzinfo=timezone.utc) for i in range(1, 13)]
    # 创建一个列表，包含 2021 年每个月的 UTC 时间
    y2 = [datetime(2021, i, 1, tzinfo=timezone.utc) for i in range(1, 13)]
    # 创建一个 1x2 的子图，两个子图共享 y 轴
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # 在第一个子图上绘制 y1 数据
    ax1.plot(y1)
    # 在第二个子图上绘制 y2 数据
    ax2.plot(y2)
    # 设置第一个子图 y 轴的单位为 UTC 时间（时区为 +5 小时）
    ax1.yaxis.set_units(timezone(timedelta(hours=5)))
    # 验证第二个子图的 y 轴单位是否为 UTC 时间（时区为 +5 小时）
    assert ax2.yaxis.units == timezone(timedelta(hours=5))


def test_shared_axis_categorical():
    # str uses category.StrCategoryConverter
    # 创建一个字典 d1，键为 'a' 和 'b'，值分别为 1 和 2
    d1 = {"a": 1, "b": 2}
    # 创建一个字典 d2，键为 'a' 和 'b'，值分别为 3 和 4
    d2 = {"a": 3, "b": 4}
    # 创建一个 1x2 的子图，两个子图共享 x 轴和 y 轴
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    # 在第一个子图上绘制 d1 的键和值
    ax1.plot(d1.keys(), d1.values())
    # 在第二个子图上绘制 d2 的键和值
    ax2.plot(d2.keys(), d2.values())
    # 设置第一个子图 x 轴的单位为类别数据 ['c', 'd']
    ax1.xaxis.set_units(UnitData(["c", "d"]))
    # 验证第二个子图 x 轴的单位中是否包含 'c'
    assert "c" in ax2.xaxis.get_units()._mapping.keys()


def test_empty_default_limits(quantity_converter):
    # 将量度单位注册到量度转换器中
    munits.registry[Quantity] = quantity_converter
    # 创建一个图形和一个坐标轴对象
    fig, ax1 = plt.subplots()
    # 设置 x 轴单位为 10 英里
    ax1.xaxis.update_units(Quantity([10], "miles"))
    # 绘制图形但不渲染
    fig.draw_without_rendering()
    # 验证 x 轴的显示范围是否为 (0, 100)
    assert ax1.get_xlim() == (0, 100)
    # 设置 y 轴单位为 10 英里
    ax1.yaxis.update_units(Quantity([10], "miles"))
    # 绘制图形但不渲染
    fig.draw_without_rendering()
    # 验证 y 轴的显示范围是否为 (0, 100)
    assert ax1.get_ylim() == (0, 100)

    # 创建一个图形和一个坐标轴对象
    fig, ax = plt.subplots()
    # 添加水平线 y=30
    ax.axhline(30)
    # 绘制 x 轴数据为英里，y 轴数据为英尺的线图
    ax.plot(Quantity(np.arange(0, 3), "miles"),
            Quantity(np.arange(0, 6, 2), "feet"))
    # 绘制图形但不渲染
    fig.draw_without_rendering()
    # 验证 x 轴的显示范围是否为 (0, 2)
    assert ax.get_xlim() == (0, 2)
    # 验证 y 轴的显示范围是否为 (0, 30)
    assert ax.get_ylim() == (0, 30)

    # 创建一个图形和一个坐标轴对象
    fig, ax = plt.subplots()
    # 添加垂直线 x=30
    ax.axvline(30)
    # 绘制 x 轴数据为英里，y 轴数据为英尺的线图
    ax.plot(Quantity(np.arange(0, 3), "miles"),
            Quantity(np.arange(0, 6, 2), "feet"))
    # 绘制图形但不渲染
    fig.draw_without_rendering()
    # 验证 x 轴的显示范围是否为 (0, 30)
    assert ax.get_xlim() == (0, 30)
    # 验证 y 轴的显示范围是否为 (0, 4)
    assert ax.get_ylim() == (0, 4)

    # 创建一个图形和一个坐标轴对象
    fig, ax = plt.subplots()
    # 设置 x 轴单位为 10 英里
    ax.xaxis.update_units(Quantity([10], "miles"))
    # 添加水平线 y=30
    ax.axhline(30)
    # 绘制图形但不渲染
    fig.draw_without_rendering()
    # 验证 x 轴的显示范围是否为 (0, 100)
    assert ax.get_xlim() == (0, 100)
    # 验证 y 轴的显示范围是否为 (28.5, 31.5)
    assert ax.get_ylim() == (28.5, 31.5)

    # 创建一个图形和一个坐标轴对象
    fig, ax = plt.subplots()
    # 设置 y 轴单位为 10 英里
    ax.yaxis.update_units(Quantity([10], "miles"))
    # 添加垂直线 x=30
    ax.axvline(30)
    # 绘制图形但不渲染
    fig.draw_without_rendering()
    # 验证 y 轴的显示范围是否为 (0, 100)
    assert ax.get_ylim() == (0, 100)
    # 验证 x 轴的显示范围是否为 (28.5, 31.5)
    assert ax.get_xlim() == (28.5, 31.5)


# test array-like objects...
class Kernel:
    def __init__(self, array):
        # 将输入数组转换为 NumPy 数组
        self._array = np.asanyarray(array)

    def __array__(self, dtype=None, copy=None):
        # 如果指定了数据类型且与当前数组数据类型不同，检查是否需要复制数组
        if dtype is not None and dtype != self._array.dtype:
            if copy is not None and not copy:
                raise ValueError(
                    f"Converting array from {self._array.dtype} to "
                    f"{dtype} requires a copy"
                )

        # 将数组转换为指定数据类型，返回原数组或其副本
        arr = np.asarray(self._array, dtype=dtype)
        return (arr if not copy else np.copy(arr))

    @property
    def shape(self):
        # 返回数组的形状
        return self._array.shape


def test_plot_kernel():
    # 进行简单的测试，验证 Kernel 对象的绘图是否成功
    kernel = Kernel([1, 2, 3, 4, 5])
    # 绘制 Kernel 对象
    plt.plot(kernel)
```