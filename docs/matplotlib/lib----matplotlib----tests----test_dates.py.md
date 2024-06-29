# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_dates.py`

```py
import datetime  # 导入 datetime 模块，用于处理日期和时间

import dateutil.tz  # 导入 dateutil.tz 模块，支持时区操作
import dateutil.rrule  # 导入 dateutil.rrule 模块，用于处理复杂的重复规则
import functools  # 导入 functools 模块，提供高阶函数支持
import numpy as np  # 导入 numpy 库，并使用 np 别名
import pytest  # 导入 pytest 测试框架

from matplotlib import rc_context, style  # 从 matplotlib 中导入 rc_context 和 style
import matplotlib.dates as mdates  # 导入 matplotlib 中的日期处理模块
import matplotlib.pyplot as plt  # 导入 matplotlib 的绘图模块
from matplotlib.testing.decorators import image_comparison  # 导入 matplotlib 测试相关模块
import matplotlib.ticker as mticker  # 导入 matplotlib 的刻度设置模块


def test_date_numpyx():
    # 测试 numpy 日期处理的正确性...
    base = datetime.datetime(2017, 1, 1)  # 创建一个基础日期时间对象
    time = [base + datetime.timedelta(days=x) for x in range(0, 3)]  # 创建一个日期时间列表
    timenp = np.array(time, dtype='datetime64[ns]')  # 使用 numpy 创建日期时间数组
    data = np.array([0., 2., 1.])  # 创建一个数据数组
    fig = plt.figure(figsize=(10, 2))  # 创建一个大小为 10x2 的图形对象
    ax = fig.add_subplot(1, 1, 1)  # 添加一个子图到图形对象上
    h, = ax.plot(time, data)  # 绘制时间与数据的关系图，并获取句柄
    hnp, = ax.plot(timenp, data)  # 绘制 numpy 时间与数据的关系图，并获取句柄
    np.testing.assert_equal(h.get_xdata(orig=False), hnp.get_xdata(orig=False))  # 断言两个图的 X 数据相等
    fig = plt.figure(figsize=(10, 2))  # 创建另一个大小为 10x2 的图形对象
    ax = fig.add_subplot(1, 1, 1)  # 添加一个子图到新图形对象上
    h, = ax.plot(data, time)  # 绘制数据与时间的关系图，并获取句柄
    hnp, = ax.plot(data, timenp)  # 绘制数据与 numpy 时间的关系图，并获取句柄
    np.testing.assert_equal(h.get_ydata(orig=False), hnp.get_ydata(orig=False))  # 断言两个图的 Y 数据相等


@pytest.mark.parametrize('t0', [datetime.datetime(2017, 1, 1, 0, 1, 1),

                                [datetime.datetime(2017, 1, 1, 0, 1, 1),
                                 datetime.datetime(2017, 1, 1, 1, 1, 1)],

                                [[datetime.datetime(2017, 1, 1, 0, 1, 1),
                                  datetime.datetime(2017, 1, 1, 1, 1, 1)],
                                 [datetime.datetime(2017, 1, 1, 2, 1, 1),
                                  datetime.datetime(2017, 1, 1, 3, 1, 1)]]])
@pytest.mark.parametrize('dtype', ['datetime64[s]',
                                   'datetime64[us]',
                                   'datetime64[ms]',
                                   'datetime64[ns]'])
def test_date_date2num_numpy(t0, dtype):
    time = mdates.date2num(t0)  # 将时间转换为 matplotlib 数字格式
    tnp = np.array(t0, dtype=dtype)  # 使用指定的 dtype 创建 numpy 数组
    nptime = mdates.date2num(tnp)  # 将 numpy 时间转换为 matplotlib 数字格式
    np.testing.assert_equal(time, nptime)  # 断言转换后的时间相等


@pytest.mark.parametrize('dtype', ['datetime64[s]',
                                   'datetime64[us]',
                                   'datetime64[ms]',
                                   'datetime64[ns]'])
def test_date2num_NaT(dtype):
    t0 = datetime.datetime(2017, 1, 1, 0, 1, 1)  # 创建一个基础日期时间对象
    tmpl = [mdates.date2num(t0), np.nan]  # 创建一个模板列表
    tnp = np.array([t0, 'NaT'], dtype=dtype)  # 使用指定的 dtype 创建包含 NaT 的 numpy 数组
    nptime = mdates.date2num(tnp)  # 将 numpy 时间转换为 matplotlib 数字格式
    np.testing.assert_array_equal(tmpl, nptime)  # 断言转换后的时间数组相等


@pytest.mark.parametrize('units', ['s', 'ms', 'us', 'ns'])
def test_date2num_NaT_scalar(units):
    tmpl = mdates.date2num(np.datetime64('NaT', units))  # 将 NaT 转换为 matplotlib 数字格式
    assert np.isnan(tmpl)  # 断言转换后的结果为 NaN


def test_date2num_masked():
    # Without tzinfo
    base = datetime.datetime(2022, 12, 15)  # 创建一个基础日期时间对象
    dates = np.ma.array([base + datetime.timedelta(days=(2 * i))
                         for i in range(7)], mask=[0, 1, 1, 0, 0, 0, 1])  # 创建一个掩码数组
    npdates = mdates.date2num(dates)  # 将掩码数组转换为 matplotlib 数字格式
    # 断言验证 npdates 的掩码是否与期望的一致
    np.testing.assert_array_equal(np.ma.getmask(npdates),
                                  (False, True, True, False, False, False,
                                   True))

    # 使用时区信息创建基准日期时间对象
    base = datetime.datetime(2022, 12, 15, tzinfo=mdates.UTC)
    # 创建带有掩码的日期时间数组
    dates = np.ma.array([base + datetime.timedelta(days=(2 * i))
                         for i in range(7)], mask=[0, 1, 1, 0, 0, 0, 1])
    # 将日期时间数组转换为数值表示的日期时间
    npdates = mdates.date2num(dates)
    # 再次断言验证 npdates 的掩码是否与期望的一致
    np.testing.assert_array_equal(np.ma.getmask(npdates),
                                  (False, True, True, False, False, False,
                                   True))
def test_date_empty():
    # 创建一个包含图和坐标轴的 subplot 对象
    fig, ax = plt.subplots()
    # 设置 x 轴为日期格式
    ax.xaxis_date()
    # 绘制图形但不进行渲染
    fig.draw_without_rendering()
    # 断言 x 轴的限制范围接近于给定的日期数值
    np.testing.assert_allclose(ax.get_xlim(),
                               [mdates.date2num(np.datetime64('1970-01-01')),
                                mdates.date2num(np.datetime64('1970-01-02'))])

    # 重置日期起点测试示例
    mdates._reset_epoch_test_example()
    # 设置新的日期起点为 '0000-12-31'
    mdates.set_epoch('0000-12-31')
    # 创建另一个包含图和坐标轴的 subplot 对象
    fig, ax = plt.subplots()
    # 设置 x 轴为日期格式
    ax.xaxis_date()
    # 绘制图形但不进行渲染
    fig.draw_without_rendering()
    # 断言 x 轴的限制范围接近于给定的日期数值
    np.testing.assert_allclose(ax.get_xlim(),
                               [mdates.date2num(np.datetime64('1970-01-01')),
                                mdates.date2num(np.datetime64('1970-01-02'))])
    # 重置日期起点测试示例
    mdates._reset_epoch_test_example()


def test_date_not_empty():
    # 创建一个图对象
    fig = plt.figure()
    # 在图中添加一个子图
    ax = fig.add_subplot()

    # 在子图中绘制一条线
    ax.plot([50, 70], [1, 2])
    # 设置 x 轴为日期格式
    ax.xaxis.axis_date()
    # 断言 x 轴的限制范围接近于给定的数值
    np.testing.assert_allclose(ax.get_xlim(), [50, 70])


def test_axhline():
    # 创建一个包含图和坐标轴的 subplot 对象
    fig, ax = plt.subplots()
    # 在坐标轴上添加水平线
    ax.axhline(1.5)
    # 在图上绘制一些日期数据
    ax.plot([np.datetime64('2016-01-01'), np.datetime64('2016-01-02')], [1, 2])
    # 断言 x 轴的限制范围接近于给定的日期数值
    np.testing.assert_allclose(ax.get_xlim(),
                               [mdates.date2num(np.datetime64('2016-01-01')),
                                mdates.date2num(np.datetime64('2016-01-02'))])

    # 重置日期起点测试示例
    mdates._reset_epoch_test_example()
    # 设置新的日期起点为 '0000-12-31'
    mdates.set_epoch('0000-12-31')
    # 创建另一个包含图和坐标轴的 subplot 对象
    fig, ax = plt.subplots()
    # 在坐标轴上添加水平线
    ax.axhline(1.5)
    # 在图上绘制一些日期数据
    ax.plot([np.datetime64('2016-01-01'), np.datetime64('2016-01-02')], [1, 2])
    # 断言 x 轴的限制范围接近于给定的日期数值
    np.testing.assert_allclose(ax.get_xlim(),
                               [mdates.date2num(np.datetime64('2016-01-01')),
                                mdates.date2num(np.datetime64('2016-01-02'))])
    # 重置日期起点测试示例
    mdates._reset_epoch_test_example()


@image_comparison(['date_axhspan.png'])
def test_date_axhspan():
    # 使用日期输入测试 axhspan
    t0 = datetime.datetime(2009, 1, 20)
    tf = datetime.datetime(2009, 1, 21)
    # 创建一个包含图和坐标轴的 subplot 对象
    fig, ax = plt.subplots()
    # 在坐标轴上添加一个水平带
    ax.axhspan(t0, tf, facecolor="blue", alpha=0.25)
    # 设置 y 轴的限制范围
    ax.set_ylim(t0 - datetime.timedelta(days=5),
                tf + datetime.timedelta(days=5))
    # 调整图的子图布局
    fig.subplots_adjust(left=0.25)


@image_comparison(['date_axvspan.png'])
def test_date_axvspan():
    # 使用日期输入测试 axvspan
    t0 = datetime.datetime(2000, 1, 20)
    tf = datetime.datetime(2010, 1, 21)
    # 创建一个包含图和坐标轴的 subplot 对象
    fig, ax = plt.subplots()
    # 在坐标轴上添加一个垂直带
    ax.axvspan(t0, tf, facecolor="blue", alpha=0.25)
    # 设置 x 轴的限制范围
    ax.set_xlim(t0 - datetime.timedelta(days=720),
                tf + datetime.timedelta(days=720))
    # 自动调整 x 轴的日期显示格式
    fig.autofmt_xdate()


@image_comparison(['date_axhline.png'])
def test_date_axhline():
    # 使用日期输入测试 axhline
    t0 = datetime.datetime(2009, 1, 20)
    # 创建一个包含图和坐标轴的 subplot 对象
    fig, ax = plt.subplots()
    # 在坐标轴上添加一条水平线
    ax.axhline(t0, color="blue", lw=3)
    # 设置图表的纵坐标范围，以起始时间 t0 减去 5 天，终止时间 tf 加上 5 天来确定
    ax.set_ylim(t0 - datetime.timedelta(days=5),
                tf + datetime.timedelta(days=5))
    # 调整图表的子图布局，左侧边距设置为 0.25
    fig.subplots_adjust(left=0.25)
@image`
# 使用 `image_comparison` 装饰器来测试日期轴垂直线图的生成
@image_comparison(['date_axvline.png'])
def test_date_axvline():
    # 使用日期时间创建起始时间 t0 和结束时间 tf
    t0 = datetime.datetime(2000, 1, 20)
    tf = datetime.datetime(2000, 1, 21)
    # 创建图形对象 fig 和坐标轴对象 ax
    fig, ax = plt.subplots()
    # 在坐标轴上添加垂直线，起始时间为 t0，线条颜色为红色，线宽为 3
    ax.axvline(t0, color="red", lw=3)
    # 设置 x 轴的显示范围，左边界为 t0 减去 5 天，右边界为 tf 加上 5 天
    ax.set_xlim(t0 - datetime.timedelta(days=5),
                tf + datetime.timedelta(days=5))
    # 自动调整 x 轴的日期显示格式
    fig.autofmt_xdate()


def test_too_many_date_ticks(caplog):
    # 测试 SF 2715172 问题，详情见链接
    # 设置日历时间相同的 t0 和 tf，触发 transforms.nonsingular 中的扩展器调用，
    # 导致 DayLocator 中的刻度过多。预期会产生一个 WARNING 级别的日志。
    caplog.set_level("WARNING")
    t0 = datetime.datetime(2000, 1, 20)
    tf = datetime.datetime(2000, 1, 20)
    # 创建图形对象 fig 和坐标轴对象 ax
    fig, ax = plt.subplots()
    # 使用 pytest 来检测 UserWarning 是否被触发
    with pytest.warns(UserWarning) as rec:
        # 设置 x 轴的显示范围为 (t0, tf)，自动调整
        ax.set_xlim((t0, tf), auto=True)
        assert len(rec) == 1
        assert ('Attempting to set identical low and high xlims'
                in str(rec[0].message))
    # 绘制空图以创建坐标轴
    ax.plot([], [])
    # 设置 x 轴主刻度定位器为 DayLocator
    ax.xaxis.set_major_locator(mdates.DayLocator())
    # 获取 x 轴主刻度定位器返回的值
    v = ax.xaxis.get_major_locator()()
    assert len(v) > 1000
    # 由于主定位器在放置次要刻度线和标签位置时也会被调用，因此警告会多次发出
    assert caplog.records and all(
        record.name == "matplotlib.ticker" and record.levelname == "WARNING"
        for record in caplog.records)
    assert len(caplog.records) > 0


def _new_epoch_decorator(thefunc):
    # 创建一个新的装饰器，用于将函数 func 的时间戳重置为 '2000-01-01'
    @functools.wraps(thefunc)
    def wrapper():
        # 调用 mdates 模块的 _reset_epoch_test_example 函数
        mdates._reset_epoch_test_example()
        # 设置全局时间戳为 '2000-01-01'
        mdates.set_epoch('2000-01-01')
        # 执行原始函数 thefunc
        thefunc()
        # 再次调用 _reset_epoch_test_example 重置时间戳
        mdates._reset_epoch_test_example()
    return wrapper


@image_comparison(['RRuleLocator_bounds.png'])
def test_RRuleLocator():
    # 导入单位测试相关的模块并注册
    import matplotlib.testing.jpl_units as units
    units.register()
    # 创建图形对象 fig
    fig = plt.figure()
    # 创建坐标轴对象 ax
    ax = plt.subplot()
    ax.set_autoscale_on(True)
    # 在坐标轴上绘制数据点，起始时间 t0 和结束时间 tf
    ax.plot([t0, tf], [0.0, 1.0], marker='o')
    # 创建年度规则定位器对象 rrule，每隔 500 年触发一次
    rrule = mdates.rrulewrapper(dateutil.rrule.YEARLY, interval=500)
    # 使用 rrule 创建规则定位器 locator
    locator = mdates.RRuleLocator(rrule)
    # 设置坐标轴 x 轴的主刻度定位器为 locator
    ax.xaxis.set_major_locator(locator)
    # 设置坐标轴 x 轴的主刻度格式化器为 AutoDateFormatter
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    # 自动调整视图范围
    ax.autoscale_view()
    # 自动调整 x 轴的日期显示格式
    fig.autofmt_xdate()


def test_RRuleLocator_dayrange():
    # 创建日定位器 loc
    loc = mdates.DayLocator()
    # 创建起始时间 x1 和结束时间 y1
    x1 = datetime.datetime(year=1, month=1, day=1, tzinfo=mdates.UTC)
    y1 = datetime.datetime(year=1, month=1, day=16, tzinfo=mdates.UTC)
    # 获取日定位器 loc 在 x1 和 y1 之间的刻度值
    loc.tick_values(x1, y1)
    # 成功时，不应抛出溢出错误


def test_RRuleLocator_close_minmax():
    # 测试封闭的最小值和最大值情况
    # 创建一个每5秒的规则对象，用于时间轴的刻度间隔
    rrule = mdates.rrulewrapper(dateutil.rrule.SECONDLY, interval=5)
    # 使用上面定义的规则对象创建时间刻度定位器
    loc = mdates.RRuleLocator(rrule)
    # 创建两个日期时间对象，d1和d2，分别代表2020年1月1日和同一日的第一微秒
    d1 = datetime.datetime(year=2020, month=1, day=1)
    d2 = datetime.datetime(year=2020, month=1, day=1, microsecond=1)
    # 预期的字符串形式日期时间列表，用于断言检查
    expected = ['2020-01-01 00:00:00+00:00',
                '2020-01-01 00:00:00.000001+00:00']
    # 使用日期时间定位器的tick_values方法获取在d1和d2之间的时间刻度，并将其转换为字符串列表，然后进行断言检查
    assert list(map(str, mdates.num2date(loc.tick_values(d1, d2)))) == expected
@image_comparison(['DateFormatter_fractionalSeconds.png'])
# 定义一个装饰器函数，用于测试生成图片对比
def test_DateFormatter():
    import matplotlib.testing.jpl_units as units
    # 导入JPL单位测试模块

    units.register()
    # 注册单位

    # 确保DateFormatter可以在分数秒的间隔上显示刻度标记
    t0 = datetime.datetime(2001, 1, 1, 0, 0, 0)
    # 设置起始时间为2001年1月1日0时0分0秒
    tf = datetime.datetime(2001, 1, 1, 0, 0, 1)
    # 设置结束时间为2001年1月1日0时0分1秒

    fig = plt.figure()
    # 创建一个新的图形
    ax = plt.subplot()
    # 创建一个子图

    ax.set_autoscale_on(True)
    # 开启自动缩放模式
    ax.plot([t0, tf], [0.0, 1.0], marker='o')
    # 绘制 t0 和 tf 之间的线段，使用圆圈标记

    ax.autoscale_view()
    # 自动缩放视图
    fig.autofmt_xdate()
    # 自动格式化X轴上的日期


def test_locator_set_formatter():
    """
    Test if setting the locator only will update the AutoDateFormatter to use
    the new locator.
    """
    plt.rcParams["date.autoformatter.minute"] = "%d %H:%M"
    # 设置分钟的自动格式化方式为 "%d %H:%M"
    t = [datetime.datetime(2018, 9, 30, 8, 0),
         datetime.datetime(2018, 9, 30, 8, 59),
         datetime.datetime(2018, 9, 30, 10, 30)]
    # 创建一个日期时间列表 t
    x = [2, 3, 1]
    # 创建一个数值列表 x

    fig, ax = plt.subplots()
    # 创建一个包含子图的图形
    ax.plot(t, x)
    # 绘制 t 对应的数据 x
    ax.xaxis.set_major_locator(mdates.MinuteLocator((0, 30)))
    # 设置X轴主刻度的定位器为分钟定位器，间隔为0和30分钟
    fig.canvas.draw()
    # 绘制图形的画布
    ticklabels = [tl.get_text() for tl in ax.get_xticklabels()]
    # 获取X轴刻度标签的文本内容
    expected = ['30 08:00', '30 08:30', '30 09:00',
                '30 09:30', '30 10:00', '30 10:30']
    # 期望的刻度标签内容列表
    assert ticklabels == expected
    # 断言刻度标签内容与期望相同

    ax.xaxis.set_major_locator(mticker.NullLocator())
    # 设置X轴主刻度的定位器为空定位器
    ax.xaxis.set_minor_locator(mdates.MinuteLocator((5, 55)))
    # 设置X轴次刻度的定位器为分钟定位器，间隔为5和55分钟
    decoy_loc = mdates.MinuteLocator((12, 27))
    # 创建一个假定的分钟定位器
    ax.xaxis.set_minor_formatter(mdates.AutoDateFormatter(decoy_loc))
    # 设置X轴次刻度的格式化器为自动日期格式化器，使用假定的分钟定位器

    ax.xaxis.set_minor_locator(mdates.MinuteLocator((15, 45)))
    # 设置X轴次刻度的定位器为分钟定位器，间隔为15和45分钟
    fig.canvas.draw()
    # 绘制图形的画布
    ticklabels = [tl.get_text() for tl in ax.get_xticklabels(which="minor")]
    # 获取次刻度标签的文本内容
    expected = ['30 08:15', '30 08:45', '30 09:15', '30 09:45', '30 10:15']
    # 期望的次刻度标签内容列表
    assert ticklabels == expected
    # 断言次刻度标签内容与期望相同


def test_date_formatter_callable():

    class _Locator:
        def _get_unit(self): return -11
        # 定义一个内部类_Locator，并提供一个_get_unit方法，返回-11

    def callable_formatting_function(dates, _):
        # 定义一个可调用的日期格式化函数，接受日期列表和占位符参数
        return [dt.strftime('%d-%m//%Y') for dt in dates]
        # 返回日期列表中各个日期的自定义格式化字符串列表

    formatter = mdates.AutoDateFormatter(_Locator())
    # 创建一个自动日期格式化器，使用_Locator类的实例作为定位器
    formatter.scaled[-10] = callable_formatting_function
    # 将可调用的格式化函数关联到缩放级别为-10的格式化器
    assert formatter([datetime.datetime(2014, 12, 25)]) == ['25-12//2014']
    # 断言使用格式化器格式化日期后的结果与预期相同


@pytest.mark.parametrize('delta, expected', [
    (datetime.timedelta(weeks=52 * 200),
     [r'$\mathdefault{%d}$' % year for year in range(1990, 2171, 20)]),
    (datetime.timedelta(days=30),
     [r'$\mathdefault{1990{-}01{-}%02d}$' % day for day in range(1, 32, 3)]),
    (datetime.timedelta(hours=20),
     [r'$\mathdefault{01{-}01\;%02d}$' % hour for hour in range(0, 21, 2)]),
    (datetime.timedelta(minutes=10),
     [r'$\mathdefault{01\;00{:}%02d}$' % minu for minu in range(0, 11)]),
])
# 参数化测试，测试不同的时间间隔（delta）和预期结果（expected）
def test_date_formatter_usetex(delta, expected):
    style.use("default")
    # 使用默认样式

    d1 = datetime.datetime(1990, 1, 1)
    # 创建一个日期时间对象d1
    # 计算出结束日期 d2，是起始日期 d1 加上时间增量 delta
    d2 = d1 + delta
    
    # 创建一个自动日期定位器 locator，不生成间隔倍数
    locator = mdates.AutoDateLocator(interval_multiples=False)
    # 创建一个虚拟坐标轴以供定位器使用
    locator.create_dummy_axis()
    # 设置定位器的视图间隔，从日期 d1 到日期 d2 的数值表示
    locator.axis.set_view_interval(mdates.date2num(d1), mdates.date2num(d2))
    
    # 使用定位器创建自动日期格式化器 formatter，使用 LaTeX 渲染选项
    formatter = mdates.AutoDateFormatter(locator, usetex=True)
    # 断言日期格式化器应用于定位器中每个位置后的结果与期望的列表 expected 相同
    assert [formatter(loc) for loc in locator()] == expected
def test_drange():
    """
    This test should check if drange works as expected, and if all the
    rounding errors are fixed
    """
    # 定义开始时间为2011年1月1日，时区为UTC
    start = datetime.datetime(2011, 1, 1, tzinfo=mdates.UTC)
    # 定义结束时间为2011年1月2日，时区为UTC
    end = datetime.datetime(2011, 1, 2, tzinfo=mdates.UTC)
    # 定义时间间隔为1小时
    delta = datetime.timedelta(hours=1)
    
    # 预期调用mdates.drange(start, end, delta)返回24个时间点，因为drange返回半开区间[start, end)
    assert len(mdates.drange(start, end, delta)) == 24

    # 如果结束时间微调为稍早一些，预期返回仍然是24个时间点
    end = end - datetime.timedelta(microseconds=1)
    assert len(mdates.drange(start, end, delta)) == 24

    # 如果结束时间稍晚一点，预期返回的时间点数量增加到25个
    end = end + datetime.timedelta(microseconds=2)
    assert len(mdates.drange(start, end, delta)) == 25

    # 重置结束时间
    end = datetime.datetime(2011, 1, 2, tzinfo=mdates.UTC)

    # 使用复杂的浮点数测试drange函数：
    # 4小时等于1/6天，这是一个“危险”的浮点数
    delta = datetime.timedelta(hours=4)
    daterange = mdates.drange(start, end, delta)
    assert len(daterange) == 6
    assert mdates.num2date(daterange[-1]) == (end - delta)


@_new_epoch_decorator
def test_auto_date_locator():
    def _create_auto_date_locator(date1, date2):
        # 创建一个自动日期定位器，禁止间隔倍数
        locator = mdates.AutoDateLocator(interval_multiples=False)
        locator.create_dummy_axis()
        locator.axis.set_view_interval(*mdates.date2num([date1, date2]))
        return locator

    d1 = datetime.datetime(1990, 1, 1)
    # 遍历结果集，对每个时间增量和预期结果进行测试
    for t_delta, expected in results:
        d2 = d1 + t_delta
        locator = _create_auto_date_locator(d1, d2)
        # 断言自动定位器返回的日期字符串列表与预期结果相符
        assert list(map(str, mdates.num2date(locator()))) == expected

    # 测试最大刻度
    locator = mdates.AutoDateLocator(interval_multiples=False)
    assert locator.maxticks == {0: 11, 1: 12, 3: 11, 4: 12, 5: 11, 6: 11, 7: 8}

    # 测试指定月份最大刻度
    locator = mdates.AutoDateLocator(maxticks={dateutil.rrule.MONTHLY: 5})
    assert locator.maxticks == {0: 11, 1: 5, 3: 11, 4: 12, 5: 11, 6: 11, 7: 8}

    # 测试最大刻度为5
    locator = mdates.AutoDateLocator(maxticks=5)
    assert locator.maxticks == {0: 5, 1: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5}


@_new_epoch_decorator
def test_auto_date_locator_intmult():
    def _create_auto_date_locator(date1, date2):
        # 创建一个自动日期定位器，允许间隔倍数
        locator = mdates.AutoDateLocator(interval_multiples=True)
        locator.create_dummy_axis()
        locator.axis.set_view_interval(*mdates.date2num([date1, date2]))
        return locator

    d1 = datetime.datetime(1997, 1, 1)
    # 遍历结果集，对每个时间增量和预期结果进行测试
    for t_delta, expected in results:
        d2 = d1 + t_delta
        locator = _create_auto_date_locator(d1, d2)
        # 断言自动定位器返回的日期字符串列表与预期结果相符
        assert list(map(str, mdates.num2date(locator()))) == expected


def test_concise_formatter_subsecond():
    locator = mdates.AutoDateLocator(interval_multiples=True)
    formatter = mdates.ConciseDateFormatter(locator)
    year_1996 = 9861.0
    # 使用 formatter 对象格式化给定时间戳列表的字符串表示
    strings = formatter.format_ticks([
        year_1996,  # 起始时间戳对应的格式化字符串
        year_1996 + 500 / mdates.MUSECONDS_PER_DAY,  # 起始时间戳增加500微秒后的格式化字符串
        year_1996 + 900 / mdates.MUSECONDS_PER_DAY  # 起始时间戳增加900微秒后的格式化字符串
    ])
    # 断言生成的格式化字符串列表与预期的列表相等
    assert strings == ['00:00', '00.0005', '00.0009']
def test_concise_formatter():
    # 定义测试函数，用于测试日期格式化器的功能

    def _create_auto_date_locator(date1, date2):
        # 创建自动日期定位器和简洁日期格式化器的绘图环境
        fig, ax = plt.subplots()

        # 创建自动日期定位器，并启用间隔倍数
        locator = mdates.AutoDateLocator(interval_multiples=True)
        # 使用自动日期定位器创建简洁日期格式化器
        formatter = mdates.ConciseDateFormatter(locator)
        # 设置Y轴的主要定位器和格式化器
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylim(date1, date2)
        # 绘制图形
        fig.canvas.draw()
        # 获取Y轴刻度标签的文本内容
        sts = [st.get_text() for st in ax.get_yticklabels()]
        return sts

    # 设置起始日期
    d1 = datetime.datetime(1997, 1, 1)
    # 设置测试结果集
    results = ([datetime.timedelta(weeks=52 * 200),
                [str(t) for t in range(1980, 2201, 20)]
                ],
               [datetime.timedelta(weeks=52),
                ['1997', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                 'Sep', 'Oct', 'Nov', 'Dec']
                ],
               [datetime.timedelta(days=141),
                ['Jan', '15', 'Feb', '15', 'Mar', '15', 'Apr', '15',
                 'May', '15']
                ],
               [datetime.timedelta(days=40),
                ['Jan', '05', '09', '13', '17', '21', '25', '29', 'Feb',
                 '05', '09']
                ],
               [datetime.timedelta(hours=40),
                ['Jan-01', '04:00', '08:00', '12:00', '16:00', '20:00',
                 'Jan-02', '04:00', '08:00', '12:00', '16:00']
                ],
               [datetime.timedelta(minutes=20),
                ['00:00', '00:05', '00:10', '00:15', '00:20']
                ],
               [datetime.timedelta(seconds=40),
                ['00:00', '05', '10', '15', '20', '25', '30', '35', '40']
                ],
               [datetime.timedelta(seconds=2),
                ['59.5', '00:00', '00.5', '01.0', '01.5', '02.0', '02.5']
                ],
               )
    
    # 对于每个测试用例，执行测试
    for t_delta, expected in results:
        d2 = d1 + t_delta
        # 调用内部函数，获取日期格式化后的字符串列表
        strings = _create_auto_date_locator(d1, d2)
        # 断言获取的日期字符串列表与预期结果相等
        assert strings == expected


@pytest.mark.parametrize('t_delta, expected', [
    # 参数化测试用例，验证简洁日期格式化器的显示偏移功能
    (datetime.timedelta(seconds=0.01), '1997-Jan-01 00:00'),
    (datetime.timedelta(minutes=1), '1997-Jan-01 00:01'),
    (datetime.timedelta(hours=1), '1997-Jan-01'),
    (datetime.timedelta(days=1), '1997-Jan-02'),
    (datetime.timedelta(weeks=1), '1997-Jan'),
    (datetime.timedelta(weeks=26), ''),  # 预期空字符串，暂不测试
    (datetime.timedelta(weeks=520), '')   # 预期空字符串，暂不测试
])
def test_concise_formatter_show_offset(t_delta, expected):
    # 测试简洁日期格式化器显示偏移功能

    d1 = datetime.datetime(1997, 1, 1)
    d2 = d1 + t_delta

    # 创建绘图环境
    fig, ax = plt.subplots()
    # 创建自动日期定位器
    locator = mdates.AutoDateLocator()
    # 使用自动日期定位器创建简洁日期格式化器
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # 绘制直线图
    ax.plot([d1, d2], [0, 0])
    fig.canvas.draw()
    # 断言获取的偏移量与预期结果相等
    assert formatter.get_offset() == expected


def test_concise_converter_stays():
    # This test demonstrates problems introduced by gh-23417 (reverted in gh-25278)
    # In particular, downstream libraries like Pandas had their designated converters
    # 该测试演示了由gh-23417引入的问题（在gh-25278中已撤消）
    # 特别是，像Pandas这样的下游库有其指定的转换器
    pass
    # 创建一个包含两个日期时间对象的列表，作为 x 轴的数据点
    x = [datetime.datetime(2000, 1, 1), datetime.datetime(2020, 2, 20)]
    
    # 创建一个包含两个数值的列表，作为对应的 y 轴数据点
    y = [0, 1]
    
    # 创建一个新的图形对象和轴对象，用于绘制图形
    fig, ax = plt.subplots()
    
    # 在轴对象上绘制 x 和 y 数据点的折线图
    ax.plot(x, y)
    
    # 设置 x 轴的日期转换器为 ConciseDateConverter 类的实例，并将其赋值给变量 conv
    ax.xaxis.converter = conv = mdates.ConciseDateConverter()
    
    # 断言检查 x 轴的单位是否为 None
    assert ax.xaxis.units is None
    
    # 设置 x 轴的限制范围为 x 列表中的最小和最大日期时间对象
    ax.set_xlim(*x)
    
    # 断言检查设置后 x 轴的日期转换器是否与之前设置的 conv 相同
    assert ax.xaxis.converter == conv
def test_offset_changes():
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()

    # 定义起始日期和结束日期
    d1 = datetime.datetime(1997, 1, 1)
    d2 = d1 + datetime.timedelta(weeks=520)

    # 创建一个自动日期定位器
    locator = mdates.AutoDateLocator()
    # 创建一个简洁日期格式化器
    formatter = mdates.ConciseDateFormatter(locator)
    # 设置坐标轴主刻度定位器和格式化器
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # 在坐标轴上绘制一条直线
    ax.plot([d1, d2], [0, 0])
    # 在不进行渲染的情况下绘制图形
    fig.draw_without_rendering()
    # 断言日期格式化器的偏移值为空字符串
    assert formatter.get_offset() == ''

    # 设置 x 轴的限制为指定时间范围
    ax.set_xlim(d1, d1 + datetime.timedelta(weeks=3))
    # 再次绘制图形
    fig.draw_without_rendering()
    # 断言日期格式化器的偏移值为 '1997-Jan'
    assert formatter.get_offset() == '1997-Jan'

    # 设置 x 轴的限制为另一个时间范围
    ax.set_xlim(d1 + datetime.timedelta(weeks=7), d1 + datetime.timedelta(weeks=30))
    # 再次绘制图形
    fig.draw_without_rendering()
    # 断言日期格式化器的偏移值为 '1997'
    assert formatter.get_offset() == '1997'

    # 恢复 x 轴的限制为最初的时间范围
    ax.set_xlim(d1, d1 + datetime.timedelta(weeks=520))
    # 再次绘制图形
    fig.draw_without_rendering()
    # 断言日期格式化器的偏移值为空字符串
    assert formatter.get_offset() == ''


@pytest.mark.parametrize('t_delta, expected', [
    # 测试用例参数化：测试不同时间增量对应的期望输出
    (datetime.timedelta(weeks=52 * 200),
     ['$\\mathdefault{%d}$' % (t, ) for t in range(1980, 2201, 20)]),
    (datetime.timedelta(days=40),
     ['Jan', '$\\mathdefault{05}$', '$\\mathdefault{09}$',
      '$\\mathdefault{13}$', '$\\mathdefault{17}$', '$\\mathdefault{21}$',
      '$\\mathdefault{25}$', '$\\mathdefault{29}$', 'Feb',
      '$\\mathdefault{05}$', '$\\mathdefault{09}$']),
    (datetime.timedelta(hours=40),
     ['Jan$\\mathdefault{{-}01}$', '$\\mathdefault{04{:}00}$',
      '$\\mathdefault{08{:}00}$', '$\\mathdefault{12{:}00}$',
      '$\\mathdefault{16{:}00}$', '$\\mathdefault{20{:}00}$',
      'Jan$\\mathdefault{{-}02}$', '$\\mathdefault{04{:}00}$',
      '$\\mathdefault{08{:}00}$', '$\\mathdefault{12{:}00}$',
      '$\\mathdefault{16{:}00}$']),
    (datetime.timedelta(seconds=2),
     ['$\\mathdefault{59.5}$', '$\\mathdefault{00{:}00}$',
      '$\\mathdefault{00.5}$', '$\\mathdefault{01.0}$',
      '$\\mathdefault{01.5}$', '$\\mathdefault{02.0}$',
      '$\\mathdefault{02.5}$']),
])
def test_concise_formatter_usetex(t_delta, expected):
    # 创建起始日期和结束日期
    d1 = datetime.datetime(1997, 1, 1)
    d2 = d1 + t_delta

    # 创建一个自动日期定位器，使用间隔倍数
    locator = mdates.AutoDateLocator(interval_multiples=True)
    # 创建一个虚拟坐标轴
    locator.create_dummy_axis()
    # 设置定位器的视图间隔为指定日期范围
    locator.axis.set_view_interval(mdates.date2num(d1), mdates.date2num(d2))

    # 创建一个使用TeX排版的简洁日期格式化器
    formatter = mdates.ConciseDateFormatter(locator, usetex=True)
    # 断言格式化指定定位器的刻度输出与期望相符
    assert formatter.format_ticks(locator()) == expected


def test_concise_formatter_formats():
    # 定义多种日期格式
    formats = ['%Y', '%m/%Y', 'day: %d',
               '%H hr %M min', '%H hr %M min', '%S.%f sec']

    def _create_auto_date_locator(date1, date2):
        # 创建一个新的图形和坐标轴对象
        fig, ax = plt.subplots()

        # 创建一个自动日期定位器，使用间隔倍数
        locator = mdates.AutoDateLocator(interval_multiples=True)
        # 创建一个使用自定义格式列表的简洁日期格式化器
        formatter = mdates.ConciseDateFormatter(locator, formats=formats)
        # 设置坐标轴主刻度定位器和格式化器
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylim(date1, date2)
        # 绘制图形并获取刻度标签的文本
        fig.canvas.draw()
        sts = [st.get_text() for st in ax.get_yticklabels()]
        return sts

    # 定义起始日期
    d1 = datetime.datetime(1997, 1, 1)
    results = (
        [datetime.timedelta(weeks=52 * 200), [str(t) for t in range(1980,
         2201, 20)]],  # 定义一个时间增量为5200年的结果组合，包含1980到2200年每20年的字符串表示列表
        [datetime.timedelta(weeks=52), [
            '1997', '02/1997', '03/1997', '04/1997', '05/1997', '06/1997',
            '07/1997', '08/1997', '09/1997', '10/1997', '11/1997', '12/1997',
            ]],  # 定义一个时间增量为52周的结果组合，包含1997年每个月份的字符串表示列表
        [datetime.timedelta(days=141), [
            '01/1997', 'day: 15', '02/1997', 'day: 15', '03/1997', 'day: 15',
            '04/1997', 'day: 15', '05/1997', 'day: 15',
            ]],  # 定义一个时间增量为141天的结果组合，包含1997年每月15日的字符串表示列表
        [datetime.timedelta(days=40), [
            '01/1997', 'day: 05', 'day: 09', 'day: 13', 'day: 17', 'day: 21',
            'day: 25', 'day: 29', '02/1997', 'day: 05', 'day: 09',
            ]],  # 定义一个时间增量为40天的结果组合，包含1997年每月5日和9日的字符串表示列表
        [datetime.timedelta(hours=40), [
            'day: 01', '04 hr 00 min', '08 hr 00 min', '12 hr 00 min',
            '16 hr 00 min', '20 hr 00 min', 'day: 02', '04 hr 00 min',
            '08 hr 00 min', '12 hr 00 min', '16 hr 00 min',
            ]],  # 定义一个时间增量为40小时的结果组合，包含每天的几个固定时间点的字符串表示列表
        [datetime.timedelta(minutes=20), ['00 hr 00 min', '00 hr 05 min',
         '00 hr 10 min', '00 hr 15 min', '00 hr 20 min']],  # 定义一个时间增量为20分钟的结果组合，包含每5分钟的字符串表示列表
        [datetime.timedelta(seconds=40), [
            '00 hr 00 min', '05.000000 sec', '10.000000 sec',
            '15.000000 sec', '20.000000 sec', '25.000000 sec',
            '30.000000 sec', '35.000000 sec', '40.000000 sec',
            ]],  # 定义一个时间增量为40秒的结果组合，包含每5秒的字符串表示列表
        [datetime.timedelta(seconds=2), [
            '59.500000 sec', '00 hr 00 min', '00.500000 sec', '01.000000 sec',
            '01.500000 sec', '02.000000 sec', '02.500000 sec',
            ]],  # 定义一个时间增量为2秒的结果组合，包含每0.5秒的字符串表示列表
        )
    for t_delta, expected in results:
        d2 = d1 + t_delta  # 计算 d1 增加 t_delta 后的日期时间
        strings = _create_auto_date_locator(d1, d2)  # 使用 _create_auto_date_locator 函数生成 d1 到 d2 之间的日期时间字符串列表
        assert strings == expected  # 断言生成的字符串列表与预期结果列表相同
def test_concise_formatter_zformats():
    zero_formats = ['', "'%y", '%B', '%m-%d', '%S', '%S.%f']

    def _create_auto_date_locator(date1, date2):
        # 创建一个新的图形和轴对象
        fig, ax = plt.subplots()

        # 创建一个自动日期定位器
        locator = mdates.AutoDateLocator(interval_multiples=True)
        # 使用ConciseDateFormatter创建一个日期格式化器，指定零格式列表
        formatter = mdates.ConciseDateFormatter(locator, zero_formats=zero_formats)
        
        # 设置主要刻度的定位器和格式化器
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
        
        # 设置y轴的数值范围
        ax.set_ylim(date1, date2)
        
        # 绘制图形
        fig.canvas.draw()
        
        # 获取y轴刻度标签的文本内容
        sts = [st.get_text() for st in ax.get_yticklabels()]
        return sts

    # 定义起始日期d1
    d1 = datetime.datetime(1997, 1, 1)
    
    # 定义测试结果集，每个结果包括时间增量和预期结果
    results = ([datetime.timedelta(weeks=52 * 200),
                [str(t) for t in range(1980, 2201, 20)]
                ],
               [datetime.timedelta(weeks=52),
                ["'97", 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                ],
               [datetime.timedelta(days=141),
                ['January', '15', 'February', '15', 'March',
                    '15', 'April', '15', 'May', '15']
                ],
               [datetime.timedelta(days=40),
                ['January', '05', '09', '13', '17', '21',
                    '25', '29', 'February', '05', '09']
                ],
               [datetime.timedelta(hours=40),
                ['01-01', '04:00', '08:00', '12:00', '16:00', '20:00',
                    '01-02', '04:00', '08:00', '12:00', '16:00']
                ],
               [datetime.timedelta(minutes=20),
                ['00', '00:05', '00:10', '00:15', '00:20']
                ],
               [datetime.timedelta(seconds=40),
                ['00', '05', '10', '15', '20', '25', '30', '35', '40']
                ],
               [datetime.timedelta(seconds=2),
                ['59.5', '00.0', '00.5', '01.0', '01.5', '02.0', '02.5']
                ],
               )
    
    # 对每组时间增量和预期结果进行迭代测试
    for t_delta, expected in results:
        # 计算结束日期d2
        d2 = d1 + t_delta
        # 调用_create_auto_date_locator函数，获取返回的刻度标签文本列表
        strings = _create_auto_date_locator(d1, d2)
        # 断言刻度标签文本列表与预期结果相同
        assert strings == expected


def test_concise_formatter_tz():
    def _create_auto_date_locator(date1, date2, tz):
        # 创建一个新的图形和轴对象
        fig, ax = plt.subplots()

        # 创建一个自动日期定位器
        locator = mdates.AutoDateLocator(interval_multiples=True)
        # 使用ConciseDateFormatter创建一个日期格式化器，指定时区
        formatter = mdates.ConciseDateFormatter(locator, tz=tz)
        
        # 设置主要刻度的定位器和格式化器
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
        
        # 设置y轴的数值范围
        ax.set_ylim(date1, date2)
        
        # 绘制图形
        fig.canvas.draw()
        
        # 获取y轴刻度标签的文本内容和y轴偏移文本的内容
        sts = [st.get_text() for st in ax.get_yticklabels()]
        offset_text = ax.yaxis.get_offset_text().get_text()
        return sts, offset_text

    # 定义起始日期d1，并且指定时区为UTC
    d1 = datetime.datetime(1997, 1, 1).replace(tzinfo=datetime.timezone.utc)
    # 定义一个包含多个测试案例的结果列表，每个案例包含预期的时间增量、预期的时间字符串列表和预期的偏移量字符串
    results = ([datetime.timedelta(hours=40),
                ['03:00', '07:00', '11:00', '15:00', '19:00', '23:00',
                 '03:00', '07:00', '11:00', '15:00', '19:00'],
                "1997-Jan-02"
                ],
               [datetime.timedelta(minutes=20),
                ['03:00', '03:05', '03:10', '03:15', '03:20'],
                "1997-Jan-01"
                ],
               [datetime.timedelta(seconds=40),
                ['03:00', '05', '10', '15', '20', '25', '30', '35', '40'],
                "1997-Jan-01 03:00"
                ],
               [datetime.timedelta(seconds=2),
                ['59.5', '03:00', '00.5', '01.0', '01.5', '02.0', '02.5'],
                "1997-Jan-01 03:00"
                ],
               )

    # 定义一个新的时区对象，偏移量为+3小时
    new_tz = datetime.timezone(datetime.timedelta(hours=3))
    
    # 遍历每个测试案例
    for t_delta, expected_strings, expected_offset in results:
        # 计算起始时间 d1 加上当前测试案例的时间增量 t_delta 得到结束时间 d2
        d2 = d1 + t_delta
        
        # 调用函数 _create_auto_date_locator 计算起始时间 d1 和结束时间 d2 之间的时间字符串列表和偏移量
        strings, offset = _create_auto_date_locator(d1, d2, new_tz)
        
        # 断言计算得到的时间字符串列表与预期的一致
        assert strings == expected_strings
        
        # 断言计算得到的偏移量与预期的一致
        assert offset == expected_offset
def test_auto_date_locator_intmult_tz():
    def _create_auto_date_locator(date1, date2, tz):
        # 创建一个自动日期定位器对象，允许使用时间间隔的倍数，并指定时区
        locator = mdates.AutoDateLocator(interval_multiples=True, tz=tz)
        # 创建一个虚拟坐标轴，用于日期定位器的初始化
        locator.create_dummy_axis()
        # 设置坐标轴的视图间隔，根据日期对象转换为数字后的范围
        locator.axis.set_view_interval(*mdates.date2num([date1, date2]))
        return locator

    # 获取时区对象，此处为加拿大太平洋时间
    tz = dateutil.tz.gettz('Canada/Pacific')
    # 创建一个带有时区信息的日期时间对象
    d1 = datetime.datetime(1997, 1, 1, tzinfo=tz)
    # 对于结果中的每个时间增量和期望结果执行以下操作
    for t_delta, expected in results:
        # 在非经典模式下使用运行时上下文
        with rc_context({'_internal.classic_mode': False}):
            # 计算第二个日期时间对象
            d2 = d1 + t_delta
            # 使用内部函数创建自动日期定位器
            locator = _create_auto_date_locator(d1, d2, tz)
            # 将日期转换为字符串列表，并应用指定的时区
            st = list(map(str, mdates.num2date(locator(), tz=tz)))
            # 断言转换后的字符串列表与期望结果相等
            assert st == expected


@image_comparison(['date_inverted_limit.png'])
def test_date_inverted_limit():
    # 使用日期输入测试水平线绘制
    t0 = datetime.datetime(2009, 1, 20)
    tf = datetime.datetime(2009, 1, 31)
    # 创建图形和轴对象
    fig, ax = plt.subplots()
    # 在轴上绘制水平线，指定颜色和线宽
    ax.axhline(t0, color="blue", lw=3)
    # 设置轴的垂直方向的限制，包括额外的时间间隔
    ax.set_ylim(t0 - datetime.timedelta(days=5),
                tf + datetime.timedelta(days=5))
    # 反转垂直轴方向
    ax.invert_yaxis()
    # 调整图形子图的左边距


def _test_date2num_dst(date_range, tz_convert):
    # 时区设置

    BRUSSELS = dateutil.tz.gettz('Europe/Brussels')
    UTC = mdates.UTC

    # 创建一个以UTC时区为基础的时区感知日期时间对象列表
    # 时间间隔为0b0.0000011天，以避免浮点数舍入问题
    dtstart = datetime.datetime(2014, 3, 30, 0, 0, tzinfo=UTC)
    interval = datetime.timedelta(minutes=33, seconds=45)
    interval_days = interval.seconds / 86400
    N = 8

    # 使用日期范围函数生成日期时间对象列表
    dt_utc = date_range(start=dtstart, freq=interval, periods=N)
    # 将UTC时间转换为布鲁塞尔时区时间
    dt_bxl = tz_convert(dt_utc, BRUSSELS)
    # 计算预期的序列浮点数
    t0 = 735322.0 + mdates.date2num(np.datetime64('0000-12-31'))
    expected_ordinalf = [t0 + (i * interval_days) for i in range(N)]
    # 使用日期转换函数将日期时间对象列表转换为序列浮点数
    actual_ordinalf = list(mdates.date2num(dt_bxl))

    # 断言实际的序列浮点数与预期的序列浮点数相等
    assert actual_ordinalf == expected_ordinalf


def test_date2num_dst():
    # 测试Github问题＃3896，针对日期转换在夏令时转换期间使用时区感知的pandas日期范围对象
    # 定义一个继承自 datetime.datetime 的自定义类 dt_tzaware
    class dt_tzaware(datetime.datetime):
        """
        This bug specifically occurs because of the normalization behavior of
        pandas Timestamp objects, so in order to replicate it, we need a
        datetime-like object that applies timezone normalization after
        subtraction.
        """
    
        # 重载减法运算符 -
        def __sub__(self, other):
            # 调用父类的减法运算符 -
            r = super().__sub__(other)
            # 获取结果对象的时区信息
            tzinfo = getattr(r, 'tzinfo', None)
    
            # 如果结果对象有时区信息
            if tzinfo is not None:
                # 获取时区信息的 normalize 方法
                localizer = getattr(tzinfo, 'normalize', None)
                # 如果存在 normalize 方法，对结果对象进行时区规范化
                if localizer is not None:
                    r = tzinfo.normalize(r)
    
            # 如果结果对象是 datetime.datetime 类型，调用 mk_tzaware 方法将其转换为带时区的对象
            if isinstance(r, datetime.datetime):
                r = self.mk_tzaware(r)
    
            return r
    
        # 重载加法运算符 +
        def __add__(self, other):
            # 调用父类的加法运算符 +
            return self.mk_tzaware(super().__add__(other))
    
        # 将 datetime 对象转换为指定时区的方法
        def astimezone(self, tzinfo):
            # 调用父类的 astimezone 方法进行时区转换
            dt = super().astimezone(tzinfo)
            # 调用 mk_tzaware 方法将转换后的 datetime 对象转换为带时区的对象
            return self.mk_tzaware(dt)
    
        # 类方法：将普通的 datetime 对象转换为带时区的 dt_tzaware 对象
        @classmethod
        def mk_tzaware(cls, datetime_obj):
            kwargs = {}
            attrs = ('year',
                     'month',
                     'day',
                     'hour',
                     'minute',
                     'second',
                     'microsecond',
                     'tzinfo')
    
            # 遍历 datetime 对象的属性，构建参数字典
            for attr in attrs:
                val = getattr(datetime_obj, attr, None)
                if val is not None:
                    kwargs[attr] = val
    
            # 使用参数字典创建带时区的 dt_tzaware 对象并返回
            return cls(**kwargs)
    
    
    # 定义一个类似于 pandas.date_range 的 date_range 函数
    def date_range(start, freq, periods):
        # 使用 dt_tzaware 类的类方法 mk_tzaware 将起始时间转换为带时区的对象
        dtstart = dt_tzaware.mk_tzaware(start)
    
        # 返回一个列表，包含起始时间后面的多个时间点，每个时间点间隔为 freq ，共计 periods 个时间点
        return [dtstart + (i * freq) for i in range(periods)]
    
    
    # 定义一个将时间列表转换到新时区的 tz_convert 函数
    def tz_convert(dt_list, tzinfo):
        # 使用列表推导式，将时间列表中的每个时间对象转换为新时区的时间对象
        return [d.astimezone(tzinfo) for d in dt_list]
    
    
    # 调用 _test_date2num_dst 函数，未提供相关代码
    _test_date2num_dst(date_range, tz_convert)
def test_date2num_dst_pandas(pd):
    # 测试github问题＃3896，在date2num周围处理带时区的pandas date_range对象的夏令时转换

    def tz_convert(*args):
        return pd.DatetimeIndex.tz_convert(*args).astype(object)

    # 调用内部函数_test_date2num_dst，传入pd.date_range函数和tz_convert函数
    _test_date2num_dst(pd.date_range, tz_convert)


def _test_rrulewrapper(attach_tz, get_tz):
    # 获取澳大利亚悉尼时区
    SYD = get_tz('Australia/Sydney')

    # 使用attach_tz函数将datetime.datetime对象附加时区
    dtstart = attach_tz(datetime.datetime(2017, 4, 1, 0), SYD)
    dtend = attach_tz(datetime.datetime(2017, 4, 4, 0), SYD)

    # 创建日期规则对象rule，以每日频率生成日期，从dtstart开始
    rule = mdates.rrulewrapper(freq=dateutil.rrule.DAILY, dtstart=dtstart)

    # 获取规则对象在dtstart和dtend之间的所有日期
    act = rule.between(dtstart, dtend)
    # 期望的结果列表exp，包含两个带有UTC时区信息的datetime.datetime对象
    exp = [datetime.datetime(2017, 4, 1, 13, tzinfo=dateutil.tz.tzutc()),
           datetime.datetime(2017, 4, 2, 14, tzinfo=dateutil.tz.tzutc())]

    # 断言act等于exp
    assert act == exp


def test_rrulewrapper():
    def attach_tz(dt, zi):
        return dt.replace(tzinfo=zi)

    # 调用内部函数_test_rrulewrapper，传入attach_tz函数和dateutil.tz.gettz函数
    _test_rrulewrapper(attach_tz, dateutil.tz.gettz)

    # 获取澳大利亚悉尼时区
    SYD = dateutil.tz.gettz('Australia/Sydney')
    dtstart = datetime.datetime(2017, 4, 1, 0)
    dtend = datetime.datetime(2017, 4, 4, 0)

    # 创建日期规则对象rule，以每日频率生成日期，从dtstart开始，带有悉尼时区信息，直到dtend结束
    rule = mdates.rrulewrapper(freq=dateutil.rrule.DAILY, dtstart=dtstart,
                               tzinfo=SYD, until=dtend)

    # 断言rule在dtstart之后的下一个日期等于datetime.datetime(2017, 4, 2, 0, 0, tzinfo=SYD)
    assert rule.after(dtstart) == datetime.datetime(2017, 4, 2, 0, 0,
                                                    tzinfo=SYD)
    # 断言rule在dtend之前的上一个日期等于datetime.datetime(2017, 4, 3, 0, 0, tzinfo=SYD)
    assert rule.before(dtend) == datetime.datetime(2017, 4, 3, 0, 0,
                                                   tzinfo=SYD)

    # 测试__getattr__的部分功能
    # 断言rule的_base_tzinfo属性等于SYD
    assert rule._base_tzinfo == SYD
    # 断言rule的_interval属性等于1


@pytest.mark.pytz
def test_rrulewrapper_pytz():
    # 确保rrules支持pytz时区
    pytz = pytest.importorskip("pytz")

    def attach_tz(dt, zi):
        return zi.localize(dt)

    # 调用内部函数_test_rrulewrapper，传入attach_tz函数和pytz.timezone函数
    _test_rrulewrapper(attach_tz, pytz.timezone)


@pytest.mark.pytz
def test_yearlocator_pytz():
    pytz = pytest.importorskip("pytz")

    # 获取美国纽约时区
    tz = pytz.timezone('America/New_York')
    # 生成一个包含2000个日期的列表x，每个日期增加一天
    x = [tz.localize(datetime.datetime(2010, 1, 1))
         + datetime.timedelta(i) for i in range(2000)]

    # 创建一个自动日期定位器locator，使用tz时区
    locator = mdates.AutoDateLocator(interval_multiples=True, tz=tz)
    locator.create_dummy_axis()

    # 设置locator的视图间隔，从x的第一个日期前推一天到最后一个日期后推一天
    locator.axis.set_view_interval(mdates.date2num(x[0])-1.0,
                                   mdates.date2num(x[-1])+1.0)

    # 创建一个numpy数组t，包含一组日期数字
    t = np.array([733408.208333, 733773.208333, 734138.208333,
                  734503.208333, 734869.208333, 735234.208333, 735599.208333])
    # 将t从旧纪元转换为新纪元...
    t = t + mdates.date2num(np.datetime64('0000-12-31'))

    # 使用np.testing.assert_allclose断言t与locator()的所有值近似相等
    np.testing.assert_allclose(t, locator())

    # 期望的结果列表expected，包含了一组日期的字符串表示，带有纽约时区信息
    expected = ['2009-01-01 00:00:00-05:00',
                '2010-01-01 00:00:00-05:00', '2011-01-01 00:00:00-05:00',
                '2012-01-01 00:00:00-05:00', '2013-01-01 00:00:00-05:00',
                '2014-01-01 00:00:00-05:00', '2015-01-01 00:00:00-05:00']
    # 将locator()返回的日期转换为字符串列表st
    st = list(map(str, mdates.num2date(locator(), tz=tz)))
    # 断言st等于expected
    assert st == expected
    # 使用断言来验证 locator 对象的 tick_values 方法返回的结果是否与预期的数组接近
    assert np.allclose(locator.tick_values(x[0], x[1]), np.array(
        [14610.20833333, 14610.33333333, 14610.45833333, 14610.58333333,
         14610.70833333, 14610.83333333, 14610.95833333, 14611.08333333,
         14611.20833333]))
    
    # 使用断言来验证 locator 对象通过 get_locator 方法返回的对象的 tick_values 方法返回的结果是否与预期的数组接近
    assert np.allclose(locator.get_locator(x[1], x[0]).tick_values(x[0], x[1]),
                       np.array(
        [14610.20833333, 14610.33333333, 14610.45833333, 14610.58333333,
         14610.70833333, 14610.83333333, 14610.95833333, 14611.08333333,
         14611.20833333]))
# 定义测试函数 test_YearLocator
def test_YearLocator():
    # 定义内部函数 _create_year_locator，用于创建 YearLocator 实例
    def _create_year_locator(date1, date2, **kwargs):
        # 使用给定的参数创建 YearLocator 对象
        locator = mdates.YearLocator(**kwargs)
        # 创建虚拟的坐标轴
        locator.create_dummy_axis()
        # 设置坐标轴的视图间隔为 date1 到 date2 之间的日期范围
        locator.axis.set_view_interval(mdates.date2num(date1),
                                       mdates.date2num(date2))
        return locator

    # 设定起始日期为 1990 年 1 月 1 日
    d1 = datetime.datetime(1990, 1, 1)
    # 定义测试结果集，包含不同的时间间隔、参数和期望的日期列表
    results = ([datetime.timedelta(weeks=52 * 200),
                {'base': 20, 'month': 1, 'day': 1},
                ['1980-01-01 00:00:00+00:00', '2000-01-01 00:00:00+00:00',
                 '2020-01-01 00:00:00+00:00', '2040-01-01 00:00:00+00:00',
                 '2060-01-01 00:00:00+00:00', '2080-01-01 00:00:00+00:00',
                 '2100-01-01 00:00:00+00:00', '2120-01-01 00:00:00+00:00',
                 '2140-01-01 00:00:00+00:00', '2160-01-01 00:00:00+00:00',
                 '2180-01-01 00:00:00+00:00', '2200-01-01 00:00:00+00:00']
                ],
               [datetime.timedelta(weeks=52 * 200),
                {'base': 20, 'month': 5, 'day': 16},
                ['1980-05-16 00:00:00+00:00', '2000-05-16 00:00:00+00:00',
                 '2020-05-16 00:00:00+00:00', '2040-05-16 00:00:00+00:00',
                 '2060-05-16 00:00:00+00:00', '2080-05-16 00:00:00+00:00',
                 '2100-05-16 00:00:00+00:00', '2120-05-16 00:00:00+00:00',
                 '2140-05-16 00:00:00+00:00', '2160-05-16 00:00:00+00:00',
                 '2180-05-16 00:00:00+00:00', '2200-05-16 00:00:00+00:00']
                ],
               [datetime.timedelta(weeks=52 * 5),
                {'base': 20, 'month': 9, 'day': 25},
                ['1980-09-25 00:00:00+00:00', '2000-09-25 00:00:00+00:00']
                ],
               )

    # 遍历测试结果集
    for delta, arguments, expected in results:
        # 计算结束日期
        d2 = d1 + delta
        # 使用内部函数创建 YearLocator 对象
        locator = _create_year_locator(d1, d2, **arguments)
        # 断言 YearLocator 返回的日期列表与期望列表一致
        assert list(map(str, mdates.num2date(locator()))) == expected


# 定义测试函数 test_DayLocator
def test_DayLocator():
    # 使用 pytest 检查 interval 为负数时是否会抛出 ValueError 异常
    with pytest.raises(ValueError):
        mdates.DayLocator(interval=-1)
    with pytest.raises(ValueError):
        mdates.DayLocator(interval=-1.5)
    # 使用 pytest 检查 interval 为非整数时是否会抛出 ValueError 异常
    with pytest.raises(ValueError):
        mdates.DayLocator(interval=0)
    with pytest.raises(ValueError):
        mdates.DayLocator(interval=1.3)
    # 使用 interval 为 1.0 创建 DayLocator 对象，测试是否能成功创建
    mdates.DayLocator(interval=1.0)


# 定义测试函数 test_tz_utc
def test_tz_utc():
    # 创建带有 UTC 时区信息的 datetime 对象
    dt = datetime.datetime(1970, 1, 1, tzinfo=mdates.UTC)
    # 断言对象的时区名为 'UTC'
    assert dt.tzname() == 'UTC'


# 使用 pytest.mark.parametrize 注入参数化测试
@pytest.mark.parametrize("x, tdelta",
                         # 定义参数化测试的参数 x 和期望的 timedelta 值 tdelta
                         [(1, datetime.timedelta(days=1)),
                          ([1, 1.5], [datetime.timedelta(days=1),
                                      datetime.timedelta(days=1.5)])])
def test_num2timedelta(x, tdelta):
    # 使用 mdates.num2timedelta 转换参数 x，检查结果是否与期望值 tdelta 相等
    dt = mdates.num2timedelta(x)
    assert dt == tdelta


# 定义测试函数 test_datetime64_in_list
def test_datetime64_in_list():
    # 创建包含 np.datetime64 对象的列表 dt
    dt = [np.datetime64('2000-01-01'), np.datetime64('2001-01-01')]
    # 使用 mdates.date2num 将列表中的日期转换为数值表示 dn
    dn = mdates.date2num(dt)
    # 将固定值从旧时代转换为新时代的 epoch
    t = (np.array([730120.,  730486.]) +
         mdates.date2num(np.datetime64('0000-12-31')))
    # 使用 NumPy 的测试工具进行断言，验证变量 dn 和 t 是否相等
    np.testing.assert_equal(dn, t)
def test_change_epoch():
    # 创建一个日期对象，表示2000年1月1日
    date = np.datetime64('2000-01-01')

    # 调用私有方法以重置时间起点示例
    mdates._reset_epoch_test_example()
    # 获取当前时间起点
    mdates.get_epoch()  # Set default.

    # 使用 pytest 来断言会抛出 RuntimeError 异常
    with pytest.raises(RuntimeError):
        # 下面的设置操作应该失败，因为时间起点已被设置过
        # 一旦时间起点被使用过，则不能再次设置
        mdates.set_epoch('0000-01-01')

    # 重置时间起点示例
    mdates._reset_epoch_test_example()
    # 设置时间起点为 '1970-01-01'
    mdates.set_epoch('1970-01-01')
    # 计算 date 相对于 '1970-01-01' 的天数差并转换为整数
    dt = (date - np.datetime64('1970-01-01')).astype('datetime64[D]')
    dt = dt.astype('int')
    # 使用 numpy 的断言方法，验证日期转换后的数值
    np.testing.assert_equal(mdates.date2num(date), float(dt))

    # 重置时间起点示例
    mdates._reset_epoch_test_example()
    # 设置时间起点为 '0000-12-31'
    mdates.set_epoch('0000-12-31')
    # 使用 numpy 的断言方法，验证日期转换后的数值
    np.testing.assert_equal(mdates.date2num(date), 730120.0)

    # 重置时间起点示例
    mdates._reset_epoch_test_example()
    # 设置时间起点为 '1970-01-01T01:00:00'
    mdates.set_epoch('1970-01-01T01:00:00')
    # 使用 numpy 的断言方法，验证日期转换后的数值，应考虑小时部分
    np.testing.assert_allclose(mdates.date2num(date), dt - 1./24.)
    
    # 重置时间起点示例
    mdates._reset_epoch_test_example()
    # 设置时间起点为 '1970-01-01T00:00:00'
    np.testing.assert_allclose(
        # 使用 numpy 的断言方法，验证特定日期的转换数值
        mdates.date2num(np.datetime64('1970-01-01T12:00:00')),
        0.5)


def test_warn_notintervals():
    # 创建一个日期数组，从 '2001-01-10' 到 '2001-03-04' 的日期范围
    dates = np.arange('2001-01-10', '2001-03-04', dtype='datetime64[D]')
    # 使用 AutoDateLocator 创建一个日期定位器，设置 interval_multiples 为 False
    locator = mdates.AutoDateLocator(interval_multiples=False)
    # 修改 locator 的 intervald 属性的值，用于测试目的
    locator.intervald[3] = [2]
    # 创建一个虚拟坐标轴
    locator.create_dummy_axis()
    # 设置坐标轴的视图区间为日期范围内的数值范围
    locator.axis.set_view_interval(mdates.date2num(dates[0]),
                                   mdates.date2num(dates[-1]))
    # 使用 pytest 来检查是否发出了 UserWarning，匹配给定的警告信息字符串
    with pytest.warns(UserWarning, match="AutoDateLocator was unable"):
        # 调用 locator() 方法，触发警告
        locs = locator()


def test_change_converter():
    # 设置 matplotlib 的日期转换器为 'concise'
    plt.rcParams['date.converter'] = 'concise'
    # 创建一个日期数组，从 '2020-01-01' 到 '2020-05-01' 的日期范围
    dates = np.arange('2020-01-01', '2020-05-01', dtype='datetime64[D]')
    # 创建一个图形对象和轴对象
    fig, ax = plt.subplots()

    # 绘制日期和对应的序号
    ax.plot(dates, np.arange(len(dates)))
    fig.canvas.draw()
    # 断言第一个 x 轴刻度标签的文本为 'Jan'
    assert ax.get_xticklabels()[0].get_text() == 'Jan'
    # 断言第二个 x 轴刻度标签的文本为 '15'
    assert ax.get_xticklabels()[1].get_text() == '15'

    # 将 matplotlib 的日期转换器设置为 'auto'
    plt.rcParams['date.converter'] = 'auto'
    # 再次创建图形对象和轴对象
    fig, ax = plt.subplots()

    # 绘制日期和对应的序号
    ax.plot(dates, np.arange(len(dates)))
    fig.canvas.draw()
    # 断言第一个 x 轴刻度标签的文本为 'Jan 01 2020'
    assert ax.get_xticklabels()[0].get_text() == 'Jan 01 2020'
    # 断言第二个 x 轴刻度标签的文本为 'Jan 15 2020'
    assert ax.get_xticklabels()[1].get_text() == 'Jan 15 2020'
    # 使用 pytest 来断言设置无效值会引发 ValueError 异常
    with pytest.raises(ValueError):
        plt.rcParams['date.converter'] = 'boo'


def test_change_interval_multiples():
    # 设置 matplotlib 的日期 interval_multiples 为 False
    plt.rcParams['date.interval_multiples'] = False
    # 创建一个日期数组，从 '2020-01-10' 到 '2020-05-01' 的日期范围
    dates = np.arange('2020-01-10', '2020-05-01', dtype='datetime64[D]')
    # 创建一个图形对象和轴对象
    fig, ax = plt.subplots()

    # 绘制日期和对应的序号
    ax.plot(dates, np.arange(len(dates)))
    fig.canvas.draw()
    # 断言第一个 x 轴刻度标签的文本为 'Jan 10 2020'
    assert ax.get_xticklabels()[0].get_text() == 'Jan 10 2020'
    # 断言第二个 x 轴刻度标签的文本为 'Jan 24 2020'
    assert ax.get_xticklabels()[1].get_text() == 'Jan 24 2020'

    # 将 matplotlib 的日期 interval_multiples 设置为 True
    plt.rcParams['date.interval_multiples'] = 'True'
    # 再次创建图形对象和轴对象
    fig, ax = plt.subplots()

    # 绘制日期和对应的序号
    ax.plot(dates, np.arange(len(dates)))
    fig.canvas.draw()
    # 断言第一个 x 轴刻度标签的文本为 'Jan 15 2020'
    assert ax.get_xticklabels()[0].get_text() == 'Jan 15 2020'
    # 断言第二个 x 轴刻度标签的文本为 'Feb 01 2020'


def test_DateLocator():
    # 此测试函数的注释未提供，可能是为了演示或其他目的未包含在内。
    pass
    # 创建一个日期定位器对象
    locator = mdates.DateLocator()

    # 测试非奇异性
    assert locator.nonsingular(0, np.inf) == (0, 1)
    assert locator.nonsingular(0, 1) == (0, 1)
    assert locator.nonsingular(1, 0) == (0, 1)
    assert locator.nonsingular(0, 0) == (-2, 2)

    # 创建一个虚拟的坐标轴
    locator.create_dummy_axis()

    # 检查默认值
    assert locator.datalim_to_dt() == (
        datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc),
        datetime.datetime(1970, 1, 2, 0, 0, tzinfo=datetime.timezone.utc))

    # 检查时区是否为 UTC
    assert locator.tz == mdates.UTC

    # 设置一个时区字符串为 'Iceland'
    tz_str = 'Iceland'
    iceland_tz = dateutil.tz.gettz(tz_str)

    # 检查时区不是 'Iceland'
    assert locator.tz != iceland_tz

    # 将定位器的时区设置为 'Iceland'
    locator.set_tzinfo('Iceland')

    # 现在检查时区是否为 'Iceland'
    assert locator.tz == iceland_tz

    # 创建一个新的虚拟坐标轴
    locator.create_dummy_axis()

    # 设置数据间隔为指定日期范围内的数值表示
    locator.axis.set_data_interval(*mdates.date2num(["2022-01-10", "2022-01-08"]))

    # 检查转换后的数据边界到日期时间
    assert locator.datalim_to_dt() == (
        datetime.datetime(2022, 1, 8, 0, 0, tzinfo=iceland_tz),
        datetime.datetime(2022, 1, 10, 0, 0, tzinfo=iceland_tz))

    # 设置 rcParam 参数 'timezone' 为 'Iceland'
    plt.rcParams['timezone'] = tz_str

    # 创建一个类似的新日期定位器对象
    locator = mdates.DateLocator()

    # 再次检查时区是否为 'Iceland'
    assert locator.tz == iceland_tz

    # 测试无效的时区值
    with pytest.raises(ValueError, match="Aiceland is not a valid timezone"):
        mdates.DateLocator(tz="Aiceland")
    with pytest.raises(TypeError, match="tz must be string or tzinfo subclass."):
        mdates.DateLocator(tz=1)
def test_datestr2num():
    # 检查 datestr2num 函数将日期字符串转换为序列号的正确性
    assert mdates.datestr2num('2022-01-10') == 19002.0
    # 创建一个 datetime.date 对象，作为默认日期参数
    dt = datetime.date(year=2022, month=1, day=10)
    # 检查 datestr2num 函数在给定默认日期参数的情况下的正确性
    assert mdates.datestr2num('2022-01', default=dt) == 19002.0
    # 检查 datestr2num 函数批量处理日期字符串列表的正确性，并返回对应的序列号数组
    assert np.all(mdates.datestr2num(
        ['2022-01', '2022-02'], default=dt
        ) == np.array([19002., 19033.]))
    # 检查 datestr2num 函数对空列表的处理，确保返回数组大小为 0
    assert mdates.datestr2num([]).size == 0
    # 检查 datestr2num 函数在空列表且给定默认日期参数的情况下的正确处理
    assert mdates.datestr2num([], datetime.date(year=2022,
                                                month=1, day=10)).size == 0


@pytest.mark.parametrize('kwarg',
                         ('formats', 'zero_formats', 'offset_formats'))
def test_concise_formatter_exceptions(kwarg):
    # 创建一个 AutoDateLocator 对象作为 ConciseDateFormatter 的定位器
    locator = mdates.AutoDateLocator()
    # 准备一个关键字参数字典，其中关键字由参数 'kwarg' 指定
    kwargs = {kwarg: ['', '%Y']}
    # 检查 ConciseDateFormatter 在给定非法参数时是否会引发 ValueError 异常，并匹配特定错误信息
    match = f"{kwarg} argument must be a list"
    with pytest.raises(ValueError, match=match):
        mdates.ConciseDateFormatter(locator, **kwargs)


def test_concise_formatter_call():
    # 创建一个 AutoDateLocator 对象作为 ConciseDateFormatter 的定位器
    locator = mdates.AutoDateLocator()
    # 创建 ConciseDateFormatter 对象
    formatter = mdates.ConciseDateFormatter(locator)
    # 检查 ConciseDateFormatter 对序列号进行格式化的正确性
    assert formatter(19002.0) == '2022'
    # 检查 format_data_short 方法对序列号进行详细日期时间格式化的正确性
    assert formatter.format_data_short(19002.0) == '2022-01-10 00:00:00'


def test_datetime_masked():
    # 确保所有数据都被屏蔽后会回退到视图限制中设置的日期时间
    x = np.array([datetime.datetime(2017, 1, n) for n in range(1, 6)])
    y = np.array([1, 2, 3, 4, 5])
    m = np.ma.masked_greater(y, 0)

    # 创建一个包含图形和坐标轴对象的图形窗口
    fig, ax = plt.subplots()
    # 在坐标轴上绘制屏蔽数据的折线图
    ax.plot(x, m)
    # 检查坐标轴的 x 轴限制是否被正确设置为 (0, 1)
    assert ax.get_xlim() == (0, 1)


@pytest.mark.parametrize('val', (-1000000, 10000000))
def test_num2date_error(val):
    # 检查 num2date 函数在处理异常值时是否引发 ValueError 异常，并匹配特定错误信息
    with pytest.raises(ValueError, match=f"Date ordinal {val} converts"):
        mdates.num2date(val)


def test_num2date_roundoff():
    # 检查 num2date 函数在处理浮点数时的四舍五入行为的正确性
    assert mdates.num2date(100000.0000578702) == datetime.datetime(
        2243, 10, 17, 0, 0, 4, 999980, tzinfo=datetime.timezone.utc)
    # 再略大一点的测试，步长为 20 微秒
    assert mdates.num2date(100000.0000578703) == datetime.datetime(
        2243, 10, 17, 0, 0, 5, tzinfo=datetime.timezone.utc)


def test_DateFormatter_settz():
    # 创建一个包含时区信息的日期时间对象
    time = mdates.date2num(datetime.datetime(2011, 1, 1, 0, 0,
                                             tzinfo=mdates.UTC))
    # 创建一个 DateFormatter 对象，指定日期时间格式
    formatter = mdates.DateFormatter('%Y-%b-%d %H:%M')
    # 检查 DateFormatter 对默认 UTC 时区的日期时间进行格式化的正确性
    assert formatter(time) == '2011-Jan-01 00:00'

    # 设置时区为 'Pacific/Kiritimati'
    formatter.set_tzinfo('Pacific/Kiritimati')
    # 检查 DateFormatter 对设置了特定时区的日期时间进行格式化的正确性
    assert formatter(time) == '2011-Jan-01 14:00'
```