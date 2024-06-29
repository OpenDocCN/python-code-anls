# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_polar.py`

```
# 导入 NumPy 库，并使用别名 np
import numpy as np
# 从 NumPy 测试模块中导入 assert_allclose 函数
from numpy.testing import assert_allclose
# 导入 pytest 库
import pytest

# 导入 Matplotlib 库，并使用别名 mpl
import matplotlib as mpl
# 从 Matplotlib 中导入 pyplot 模块，并使用别名 plt
from matplotlib import pyplot as plt
# 从 Matplotlib 测试装饰器中导入 image_comparison 和 check_figures_equal 函数
from matplotlib.testing.decorators import image_comparison, check_figures_equal

# 定义装饰器函数 image_comparison，用于比较图像结果
@image_comparison(['polar_axes'], style='default', tol=0.012)
def test_polar_annotations():
    # 可以在不同位置和坐标系中指定 xypoint 和 xytext，并选择是否连接线和标记点。
    # 注释也适用于极坐标轴。在下面的例子中，xy 点在本地坐标中（xycoords 默认为 'data'）。
    # 对于极坐标轴，这是在（theta, radius）空间中的坐标。本例中的文本放置在图形的分数坐标系中。
    # 水平和垂直对齐等文本关键字参数也将被尊重。

    # 设置一些数据
    r = np.arange(0.0, 1.0, 0.001)
    theta = 2.0 * 2.0 * np.pi * r

    # 创建图形对象
    fig = plt.figure()
    # 添加极坐标子图
    ax = fig.add_subplot(polar=True)
    # 绘制极坐标线
    line, = ax.plot(theta, r, color='#ee8d18', lw=3)
    line, = ax.plot((0, 0), (0, 1), color="#0000ff", lw=1)

    # 指定索引
    ind = 800
    thisr, thistheta = r[ind], theta[ind]
    # 在指定位置绘制标记点
    ax.plot([thistheta], [thisr], 'o')
    # 添加注释
    ax.annotate('a polar annotation',
                xy=(thistheta, thisr),  # theta, radius
                xytext=(0.05, 0.05),    # fraction, fraction
                textcoords='figure fraction',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='left',
                verticalalignment='baseline',
                )

    # 设置 x 轴刻度参数
    ax.tick_params(axis='x', tick1On=True, tick2On=True, direction='out')

# 定义装饰器函数 image_comparison，用于比较图像结果
@image_comparison(['polar_coords'], style='default', remove_text=True, tol=0.014)
def test_polar_coord_annotations():
    # 您还可以在笛卡尔坐标轴上使用极坐标表示法。这里本地坐标系 ('data') 是笛卡尔的，
    # 因此如果要使用（theta, radius），则需要将 xycoords 和 textcoords 指定为 'polar'。
    
    # 创建椭圆对象
    el = mpl.patches.Ellipse((0, 0), 10, 20, facecolor='r', alpha=0.5)

    # 创建图形对象
    fig = plt.figure()
    # 添加子图，并指定其纵横比
    ax = fig.add_subplot(aspect='equal')

    # 添加椭圆对象到子图中
    ax.add_artist(el)
    el.set_clip_box(ax.bbox)

    # 添加注释
    ax.annotate('the top',
                xy=(np.pi/2., 10.),      # theta, radius
                xytext=(np.pi/3, 20.),   # theta, radius
                xycoords='polar',
                textcoords='polar',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='left',
                verticalalignment='baseline',
                clip_on=True,  # clip to the axes bounding box
                )

    # 设置 x 和 y 轴限制
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)

# 定义装饰器函数 image_comparison，用于比较图像结果
@image_comparison(['polar_alignment.png'])
def test_polar_alignment():
    # 测试更改极坐标图的垂直/水平对齐方式。
    
    # 创建角度和网格值数组
    angles = np.arange(0, 360, 90)
    grid_values = [0, 0.2, 0.4, 0.6, 0.8, 1]

    # 创建图形对象
    fig = plt.figure()
    # 定义绘图区域
    rect = [0.1, 0.1, 0.8, 0.8]
    # 在图形 fig 上添加一个极坐标子图，位置和大小由 rect 指定，用于水平方向
    horizontal = fig.add_axes(rect, polar=True, label='horizontal')
    
    # 设置水平方向极坐标子图的角度刻度线
    horizontal.set_thetagrids(angles)
    
    # 在图形 fig 上添加另一个极坐标子图，位置和大小由 rect 指定，用于垂直方向
    vertical = fig.add_axes(rect, polar=True, label='vertical')
    
    # 设置垂直方向极坐标子图的背景不可见（透明）
    vertical.patch.set_visible(False)
    
    # 遍历前两个子图（水平和垂直方向），为每个子图设置径向网格线
    for i in range(2):
        fig.axes[i].set_rgrids(
            grid_values, angle=angles[i],
            horizontalalignment='left', verticalalignment='top')
# 定义一个测试函数，测试创建极坐标图形时的特定行为
def test_polar_twice():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 绘制第一组极坐标图，角度 [1, 2]，半径 [.1, .2]
    plt.polar([1, 2], [.1, .2])
    # 绘制第二组极坐标图，角度 [3, 4]，半径 [.3, .4]
    plt.polar([3, 4], [.3, .4])
    # 断言图形对象 fig 中只创建了一个极坐标轴对象
    assert len(fig.axes) == 1, 'More than one polar Axes created.'


# 使用装饰器检查两个图形对象是否相等的测试函数
@check_figures_equal()
def test_polar_wrap(fig_test, fig_ref):
    # 在测试图形对象上添加一个极坐标子图
    ax = fig_test.add_subplot(projection="polar")
    # 绘制极坐标图，角度转换为弧度，角度 [179, -179]，半径 [0.2, 0.1]
    ax.plot(np.deg2rad([179, -179]), [0.2, 0.1])
    # 绘制另一个极坐标图，角度 [2, -2]，半径 [0.2, 0.1]
    ax.plot(np.deg2rad([2, -2]), [0.2, 0.1])
    
    # 在参考图形对象上添加一个极坐标子图
    ax = fig_ref.add_subplot(projection="polar")
    # 绘制极坐标图，角度 [179, 181]，半径 [0.2, 0.1]
    ax.plot(np.deg2rad([179, 181]), [0.2, 0.1])
    ax.plot(np.deg2rad([2, 358]), [0.2, 0.1])  # 绘制另一个极坐标图


# 使用装饰器检查两个图形对象是否相等的测试函数
@check_figures_equal()
def test_polar_units_1(fig_test, fig_ref):
    # 导入单位测试模块
    import matplotlib.testing.jpl_units as units
    units.register()
    # 角度和半径数据
    xs = [30.0, 45.0, 60.0, 90.0]
    ys = [1.0, 2.0, 3.0, 4.0]

    # 在测试图形对象上创建图形
    plt.figure(fig_test.number)
    # 绘制极坐标图，角度单位为度，由单位模块提供支持
    plt.polar([x * units.deg for x in xs], ys)

    # 在参考图形对象上添加一个极坐标子图
    ax = fig_ref.add_subplot(projection="polar")
    # 绘制极坐标图，角度转换为弧度
    ax.plot(np.deg2rad(xs), ys)
    # 设置 x 轴标签为 "deg"
    ax.set(xlabel="deg")


# 使用装饰器检查两个图形对象是否相等的测试函数
@check_figures_equal()
def test_polar_units_2(fig_test, fig_ref):
    # 导入单位测试模块
    import matplotlib.testing.jpl_units as units
    units.register()
    # 角度和半径数据
    xs = [30.0, 45.0, 60.0, 90.0]
    xs_deg = [x * units.deg for x in xs]
    ys = [1.0, 2.0, 3.0, 4.0]
    ys_km = [y * units.km for y in ys]

    # 在测试图形对象上创建图形
    plt.figure(fig_test.number)
    # 绘制极坐标图，角度和半径单位为弧度和公里
    plt.polar(xs_deg, ys_km, thetaunits="rad", runits="km")
    # 断言 x 轴的主格式化器是否是单位模块提供的 UnitDblFormatter 类型的实例
    assert isinstance(plt.gca().xaxis.get_major_formatter(),
                      units.UnitDblFormatter)

    # 在参考图形对象上添加一个极坐标子图
    ax = fig_ref.add_subplot(projection="polar")
    # 绘制极坐标图，角度转换为弧度
    ax.plot(np.deg2rad(xs), ys)
    # 设置 x 轴主格式化器为函数格式化器
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter("{:.12}".format))
    # 设置坐标轴标签
    ax.set(xlabel="rad", ylabel="km")


# 使用图像比较测试检查图像的外观是否匹配的测试函数
@image_comparison(['polar_rmin'], style='default')
def test_polar_rmin():
    # 创建半径和角度数据
    r = np.arange(0, 3.0, 0.01)
    theta = 2*np.pi*r

    # 创建图形对象
    fig = plt.figure()
    # 添加极坐标子图，设置最小半径为 0.5，最大半径为 2.0
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.plot(theta, r)
    ax.set_rmax(2.0)
    ax.set_rmin(0.5)


# 使用图像比较测试检查图像的外观是否匹配的测试函数
@image_comparison(['polar_negative_rmin'], style='default')
def test_polar_negative_rmin():
    # 创建半径和角度数据
    r = np.arange(-3.0, 0.0, 0.01)
    theta = 2*np.pi*r

    # 创建图形对象
    fig = plt.figure()
    # 添加极坐标子图，设置最小半径为 -3.0，最大半径为 0.0
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.plot(theta, r)
    ax.set_rmax(0.0)
    ax.set_rmin(-3.0)


# 使用图像比较测试检查图像的外观是否匹配的测试函数
@image_comparison(['polar_rorigin'], style='default')
def test_polar_rorigin():
    # 创建半径和角度数据
    r = np.arange(0, 3.0, 0.01)
    theta = 2*np.pi*r

    # 创建图形对象
    fig = plt.figure()
    # 添加极坐标子图，设置最小半径为 0.5，最大半径为 2.0，设置原点半径为 0.0
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.plot(theta, r)
    ax.set_rmax(2.0)
    ax.set_rmin(0.5)
    ax.set_rorigin(0.0)


# 使用图像比较测试检查图像的外观是否匹配的测试函数
@image_comparison(['polar_invertedylim.png'], style='default')
def test_polar_invertedylim():
    # 创建图形对象
    fig = plt.figure()
    # 添加极坐标子图，设置半径轴反转为 True
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_ylim(2, 0)


# 使用图像比较测试检查图像的外观是否匹配的测试函数
@image_comparison(['polar_invertedylim_rorigin.png'], style='default')
def test_polar_invertedylim_rorigin():
    # 创建图形对象
    fig = plt.figure()
    # 添加极坐标子图，设置半径轴反转为 True
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.yaxis.set_inverted(True)
    # 设置半径轴限制为反转的 (2, 0)，不调用 set_rlim，以检查结果
    # 在绘制之前确保视图限制(view limits)正确地更新。
    # 绘制一条线段，起点 (0, 0) 终点 (0, 2)，颜色为 "none" 即透明色。
    ax.plot([0, 0], [0, 2], c="none")
    
    # 设置图形对象的边距为 0，确保图形绘制时边界紧贴坐标轴。
    ax.margins(0)
    
    # 设置极坐标系的原点到半径方向的偏移为 3。
    ax.set_rorigin(3)
@image_comparison(['polar_theta_position'], style='default')
# 定义测试函数，用于比较生成的图像与参考图像的差异，图像比较类型为 'default'
def test_polar_theta_position():
    # 创建半径数据数组，从0到3，步长为0.01
    r = np.arange(0, 3.0, 0.01)
    # 计算对应的角度数据数组，每个半径值乘以2π
    theta = 2*np.pi*r

    # 创建新的图形对象
    fig = plt.figure()
    # 在图形上添加极坐标轴，位置和大小参数分别为 [左, 下, 宽, 高]
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    # 绘制极坐标图，角度为 theta，半径为 r
    ax.plot(theta, r)
    # 设置极坐标的零角度位置为 "NW"（北西），偏移角度为30度
    ax.set_theta_zero_location("NW", 30)
    # 设置极坐标的角度方向为顺时针
    ax.set_theta_direction('clockwise')


@image_comparison(['polar_rlabel_position'], style='default')
# 定义测试函数，用于比较生成的图像与参考图像的差异，图像比较类型为 'default'
def test_polar_rlabel_position():
    # 创建新的图形对象
    fig = plt.figure()
    # 在图形上添加极坐标子图
    ax = fig.add_subplot(projection='polar')
    # 设置极坐标半径标签的位置为315度
    ax.set_rlabel_position(315)
    # 设置极坐标的刻度参数，使其自动旋转
    ax.tick_params(rotation='auto')


@image_comparison(['polar_theta_wedge'], style='default')
# 定义测试函数，用于比较生成的图像与参考图像的差异，图像比较类型为 'default'
def test_polar_theta_limits():
    # 创建半径数据数组，从0到3，步长为0.01
    r = np.arange(0, 3.0, 0.01)
    # 计算对应的角度数据数组，每个半径值乘以2π
    theta = 2*np.pi*r

    # 设置不同的起始角度和结束角度数组
    theta_mins = np.arange(15.0, 361.0, 90.0)
    theta_maxs = np.arange(50.0, 361.0, 90.0)
    DIRECTIONS = ('out', 'in', 'inout')

    # 创建包含多个子图的图形对象，行数由 theta_mins 的长度决定，列数由 theta_maxs 的长度决定
    fig, axs = plt.subplots(len(theta_mins), len(theta_maxs),
                            subplot_kw={'polar': True},
                            figsize=(8, 6))

    # 遍历每一个子图
    for i, start in enumerate(theta_mins):
        for j, end in enumerate(theta_maxs):
            ax = axs[i, j]
            ax.plot(theta, r)
            # 根据起始角度和结束角度设置极坐标的角度范围
            if start < end:
                ax.set_thetamin(start)
                ax.set_thetamax(end)
            else:
                # 如果起始角度大于等于结束角度，则绘制顺时针方向的图形
                ax.set_thetamin(end)
                ax.set_thetamax(start)
                ax.set_theta_direction('clockwise')
            # 设置极坐标刻度的参数，包括方向和旋转
            ax.tick_params(tick1On=True, tick2On=True,
                           direction=DIRECTIONS[i % len(DIRECTIONS)],
                           rotation='auto')
            # 设置极坐标的纵轴刻度参数，包括标签显示和旋转
            ax.yaxis.set_tick_params(label2On=True, rotation='auto')
            # 设置极坐标的横轴主定位器参数，兼容旧版本
            ax.xaxis.get_major_locator().base.set_params(
                steps=[1, 2, 2.5, 5, 10])


@check_figures_equal(extensions=["png"])
# 定义图像比较测试函数，比较测试图形和参考图形的相等性，图像文件扩展名为 'png'
def test_polar_rlim(fig_test, fig_ref):
    # 在测试图形上添加极坐标子图
    ax = fig_test.subplots(subplot_kw={'polar': True})
    # 设置极坐标半径范围的上下限
    ax.set_rlim(top=10)
    ax.set_rlim(bottom=.5)

    # 在参考图形上添加极坐标子图
    ax = fig_ref.subplots(subplot_kw={'polar': True})
    ax.set_rmax(10.)
    ax.set_rmin(.5)


@check_figures_equal(extensions=["png"])
# 定义图像比较测试函数，比较测试图形和参考图形的相等性，图像文件扩展名为 'png'
def test_polar_rlim_bottom(fig_test, fig_ref):
    # 在测试图形上添加极坐标子图
    ax = fig_test.subplots(subplot_kw={'polar': True})
    # 设置极坐标半径范围的下限和上限
    ax.set_rlim(bottom=[.5, 10])

    # 在参考图形上添加极坐标子图
    ax = fig_ref.subplots(subplot_kw={'polar': True})
    ax.set_rmax(10.)
    ax.set_rmin(.5)


# 定义极坐标的半径范围测试函数
def test_polar_rlim_zero():
    # 创建一个新的图形对象，并添加极坐标子图
    ax = plt.figure().add_subplot(projection='polar')
    # 绘制极坐标图形
    ax.plot(np.arange(10), np.arange(10) + .01)
    # 断言极坐标的半径范围下限为0
    assert ax.get_ylim()[0] == 0


# 定义无数据情况下的极坐标测试函数
def test_polar_no_data():
    # 在当前图形上添加极坐标子图
    plt.subplot(projection="polar")
    ax = plt.gca()
    # 断言极坐标的半径范围下限为0，上限为1
    assert ax.get_rmin() == 0 and ax.get_rmax() == 1
    plt.close("all")
    # 使用 plt.polar() 调用，触发无数据时的自动缩放行为
    plt.polar()
    ax = plt.gca()
    # 再次断言极坐标的半径范围下限为0，上限为1
    assert ax.get_rmin() == 0 and ax.get_rmax() == 1


# 定义默认对数极坐标范围测试函数
def test_polar_default_log_lims():
    # 在当前图形上添加对数极坐标子图
    plt.subplot(projection='polar')
    ax = plt.gca()
    # 设置极坐标的半径刻度为对数刻度
    ax.set_rscale('log')
    # 断言确保 ax 对象的最小半径大于 0
    assert ax.get_rmin() > 0
def test_polar_not_datalim_adjustable():
    # 创建一个极坐标子图，并获取对应的坐标轴对象
    ax = plt.figure().add_subplot(projection="polar")
    # 使用 pytest 来检查设置不支持 "datalim" 调整模式时是否引发 ValueError 异常
    with pytest.raises(ValueError):
        ax.set_adjustable("datalim")


def test_polar_gridlines():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个极坐标子图
    ax = fig.add_subplot(polar=True)
    # 设置网格线的透明度为 0.2，使主要网格线变得更轻
    ax.grid(alpha=0.2)
    # 隐藏 y 轴刻度标签，在版本 2.1.0 中无效
    plt.setp(ax.yaxis.get_ticklabels(), visible=False)
    # 绘制图形对象的画布
    fig.canvas.draw()
    # 断言 x 轴主刻度线的网格线透明度为 0.2
    assert ax.xaxis.majorTicks[0].gridline.get_alpha() == .2
    # 断言 y 轴主刻度线的网格线透明度为 0.2
    assert ax.yaxis.majorTicks[0].gridline.get_alpha() == .2


def test_get_tightbbox_polar():
    # 创建一个包含极坐标投影的图形和对应的坐标轴对象
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # 绘制图形对象的画布
    fig.canvas.draw()
    # 获取坐标轴的紧凑边界框
    bb = ax.get_tightbbox(fig.canvas.get_renderer())
    # 使用 assert_allclose 来断言紧凑边界框的四个极值
    assert_allclose(
        bb.extents, [107.7778,  29.2778, 539.7847, 450.7222], rtol=1e-03)


@check_figures_equal(extensions=["png"])
def test_polar_interpolation_steps_constant_r(fig_test, fig_ref):
    # 检查额外的半圈是否会造成任何变化，这里禁用了抗锯齿
    p1 = (fig_test.add_subplot(121, projection="polar")
          .bar([0], [1], 3*np.pi, edgecolor="none", antialiased=False))
    p2 = (fig_test.add_subplot(122, projection="polar")
          .bar([0], [1], -3*np.pi, edgecolor="none", antialiased=False))
    p3 = (fig_ref.add_subplot(121, projection="polar")
          .bar([0], [1], 2*np.pi, edgecolor="none", antialiased=False))
    p4 = (fig_ref.add_subplot(122, projection="polar")
          .bar([0], [1], -2*np.pi, edgecolor="none", antialiased=False))


@check_figures_equal(extensions=["png"])
def test_polar_interpolation_steps_variable_r(fig_test, fig_ref):
    # 添加极坐标子图，并绘制一条连接指定点的曲线
    l, = fig_test.add_subplot(projection="polar").plot([0, np.pi/2], [1, 2])
    # 设置路径插值步数为 100
    l.get_path()._interpolation_steps = 100
    # 添加另一个极坐标子图，并绘制另一条曲线
    fig_ref.add_subplot(projection="polar").plot(
        np.linspace(0, np.pi/2, 101), np.linspace(1, 2, 101))


def test_thetalim_valid_invalid():
    # 创建一个极坐标子图
    ax = plt.subplot(projection='polar')
    # 设置角度范围为 0 到 2π，不会引发异常
    ax.set_thetalim(0, 2 * np.pi)  # doesn't raise.
    # 设置角度范围为 800 到 440，不会引发异常
    ax.set_thetalim(thetamin=800, thetamax=440)  # doesn't raise.
    # 使用 pytest 检查设置角度范围为 0 到 3π 时是否引发 ValueError 异常
    with pytest.raises(ValueError,
                       match='angle range must be less than a full circle'):
        ax.set_thetalim(0, 3 * np.pi)
    # 使用 pytest 检查设置角度范围为 800 到 400 时是否引发 ValueError 异常
    with pytest.raises(ValueError,
                       match='angle range must be less than a full circle'):
        ax.set_thetalim(thetamin=800, thetamax=400)


def test_thetalim_args():
    # 创建一个极坐标子图
    ax = plt.subplot(projection='polar')
    # 设置角度范围为 0 到 1
    ax.set_thetalim(0, 1)
    # 使用 assert 来断言角度范围的最小值和最大值是否为弧度制的 0 和 1
    assert tuple(np.radians((ax.get_thetamin(), ax.get_thetamax()))) == (0, 1)
    # 设置角度范围为 2 到 3
    ax.set_thetalim((2, 3))
    # 使用 assert 来断言角度范围的最小值和最大值是否为弧度制的 2 和 3
    assert tuple(np.radians((ax.get_thetamin(), ax.get_thetamax()))) == (2, 3)


def test_default_thetalocator():
    # 理想情况下我们会检查 AAAABBC，但是当前最小的轴仅在 150° 处放置一个刻度，
    # 因为 MaxNLocator 没有一种方法来接受 15° 而拒绝 150°。
    # 使用 `plt.subplot_mosaic` 创建一个布局为 "AAAABB." 的多子图的极坐标图形(fig)，同时返回子图对象(axs)
    fig, axs = plt.subplot_mosaic(
        "AAAABB.", subplot_kw={"projection": "polar"})
    
    # 遍历所有子图(axs.values())，设置极坐标角度范围为 0 到 π
    for ax in axs.values():
        ax.set_thetalim(0, np.pi)
    
    # 再次遍历所有子图(axs.values())，获取 x 轴主要刻度位置并转换为角度值列表
    ticklocs = np.degrees(ax.xaxis.get_majorticklocs()).tolist()
    
    # 断言检查：确保角度值约为 90 度的刻度存在于列表中
    assert pytest.approx(90) in ticklocs
    
    # 断言检查：确保角度值约为 100 度的刻度不存在于列表中
    assert pytest.approx(100) not in ticklocs
# 定义一个测试函数，用于测试 axvspan 方法的功能
def test_axvspan():
    # 创建一个极坐标子图
    ax = plt.subplot(projection="polar")
    # 在极坐标图上绘制一个角度范围的填充区域
    span = ax.axvspan(0, np.pi/4)
    # 断言填充区域的路径插值步数大于1
    assert span.get_path()._interpolation_steps > 1


# 使用装饰器 check_figures_equal，定义一个测试函数，用于测试在删除共享极坐标轴时的行为
@check_figures_equal(extensions=["png"])
def test_remove_shared_polar(fig_ref, fig_test):
    # 删除共享的极坐标轴之前，会导致崩溃。测试删除它们，保留网格的左下角坐标轴，以避免共享轴的刻度标签可见性问题（正在修复中）。
    axs = fig_ref.subplots(
        2, 2, sharex=True, subplot_kw={"projection": "polar"})
    # 遍历并删除指定索引位置的子图
    for i in [0, 1, 3]:
        axs.flat[i].remove()
    axs = fig_test.subplots(
        2, 2, sharey=True, subplot_kw={"projection": "polar"})
    # 遍历并删除指定索引位置的子图
    for i in [0, 1, 3]:
        axs.flat[i].remove()


# 定义一个测试函数，测试共享极坐标轴保留刻度标签的行为
def test_shared_polar_keeps_ticklabels():
    # 创建一个包含共享极坐标轴的子图网格
    fig, axs = plt.subplots(
        2, 2, subplot_kw={"projection": "polar"}, sharex=True, sharey=True)
    fig.canvas.draw()
    # 断言特定子图的 x 轴主刻度标签可见
    assert axs[0, 1].xaxis.majorTicks[0].get_visible()
    # 断言特定子图的 y 轴主刻度标签可见
    assert axs[0, 1].yaxis.majorTicks[0].get_visible()
    # 使用 subplot_mosaic 方法创建子图网格
    fig, axs = plt.subplot_mosaic(
        "ab\ncd", subplot_kw={"projection": "polar"}, sharex=True, sharey=True)
    fig.canvas.draw()
    # 断言特定子图的 x 轴主刻度标签可见
    assert axs["b"].xaxis.majorTicks[0].get_visible()
    # 断言特定子图的 y 轴主刻度标签可见
    assert axs["b"].yaxis.majorTicks[0].get_visible()


# 定义一个测试函数，测试 axvline 和 axvspan 方法对极坐标轴不修改限制范围的行为
def test_axvline_axvspan_do_not_modify_rlims():
    # 创建一个极坐标子图
    ax = plt.subplot(projection="polar")
    # 在极坐标图上绘制一个角度范围的填充区域
    ax.axvspan(0, 1)
    # 在极坐标图上绘制一个垂直线
    ax.axvline(.5)
    # 绘制一条简单的线图
    ax.plot([.1, .2])
    # 断言当前极坐标轴的 y 轴限制范围
    assert ax.get_ylim() == (0, .2)


# 定义一个测试函数，测试光标精度的行为
def test_cursor_precision():
    # 创建一个极坐标子图
    ax = plt.subplot(projection="polar")
    # 断言不同半径对应不同角度精度的坐标格式化字符串
    assert ax.format_coord(0, 0.005) == "θ=0.0π (0°), r=0.005"
    assert ax.format_coord(0, .1) == "θ=0.00π (0°), r=0.100"
    assert ax.format_coord(0, 1) == "θ=0.000π (0.0°), r=1.000"
    assert ax.format_coord(1, 0.005) == "θ=0.3π (57°), r=0.005"
    assert ax.format_coord(1, .1) == "θ=0.32π (57°), r=0.100"
    assert ax.format_coord(1, 1) == "θ=0.318π (57.3°), r=1.000"
    assert ax.format_coord(2, 0.005) == "θ=0.6π (115°), r=0.005"
    assert ax.format_coord(2, .1) == "θ=0.64π (115°), r=0.100"
    assert ax.format_coord(2, 1) == "θ=0.637π (114.6°), r=1.000"


# 定义一个测试函数，测试自定义数据格式化行为的行为
def test_custom_fmt_data():
    # 创建一个极坐标子图
    ax = plt.subplot(projection="polar")
    # 定义一个百万级别的数据格式化函数
    def millions(x):
        return '$%1.1fM' % (x*1e-6)

    # 测试仅 x 轴的格式化器
    ax.fmt_xdata = None
    ax.fmt_ydata = millions
    assert ax.format_coord(12, 2e7) == "θ=3.8197186342π (687.54935416°), r=$20.0M"
    assert ax.format_coord(1234, 2e6) == "θ=392.794399551π (70702.9919191°), r=$2.0M"
    assert ax.format_coord(3, 100) == "θ=0.95493π (171.887°), r=$0.0M"

    # 测试仅 y 轴的格式化器
    ax.fmt_xdata = millions
    ax.fmt_ydata = None
    assert ax.format_coord(2e5, 1) == "θ=$0.2M, r=1.000"
    assert ax.format_coord(1, .1) == "θ=$0.0M, r=0.100"
    assert ax.format_coord(1e6, 0.005) == "θ=$1.0M, r=0.005"

    # 测试同时存在 x 和 y 轴的格式化器
    ax.fmt_xdata = millions
    ax.fmt_ydata = millions
    # 对 ax 对象调用 format_coord 方法，验证其输出是否符合预期
    assert ax.format_coord(2e6, 2e4*3e5) == "θ=$2.0M, r=$6000.0M"
    # 再次对 ax 对象调用 format_coord 方法，验证其输出是否符合预期
    assert ax.format_coord(1e18, 12891328123) == "θ=$1000000000000.0M, r=$12891.3M"
    # 第三次对 ax 对象调用 format_coord 方法，验证其输出是否符合预期
    assert ax.format_coord(63**7, 1081968*1024) == "θ=$3938980.6M, r=$1107.9M"
@image_comparison(['polar_log.png'], style='default')
# 声明一个装饰器函数，用于比较生成的图像与参考图像的差异
def test_polar_log():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上添加一个极坐标子图
    ax = fig.add_subplot(polar=True)

    # 设置极坐标的径向刻度为对数尺度
    ax.set_rscale('log')
    # 设置极坐标的径向坐标范围
    ax.set_rlim(1, 1000)

    # 生成一百个点，绘制极坐标图形，横坐标为0到2π，纵坐标为0到100的对数值
    n = 100
    ax.plot(np.linspace(0, 2 * np.pi, n), np.logspace(0, 2, n))


def test_polar_neg_theta_lims():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上添加一个极坐标子图
    ax = fig.add_subplot(projection='polar')
    
    # 设置极坐标的角度范围为从 -π 到 π
    ax.set_thetalim(-np.pi, np.pi)
    
    # 获取极坐标轴上 x 轴的刻度标签文本内容
    labels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
    # 断言获取到的标签文本应与预期的极坐标角度标签一致
    assert labels == ['-180°', '-135°', '-90°', '-45°', '0°', '45°', '90°', '135°']
```