# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\mplot3d\tests\test_axes3d.py`

```py
import functools  # 导入 functools 模块，用于函数式编程的工具
import itertools  # 导入 itertools 模块，用于创建迭代器的函数
import platform   # 导入 platform 模块，用于访问平台相关的属性和功能

import pytest  # 导入 pytest 模块，用于编写和执行测试用例

from mpl_toolkits.mplot3d import Axes3D, axes3d, proj3d, art3d  # 导入 3D 绘图相关的类和函数
from mpl_toolkits.mplot3d.axes3d import _Quaternion as Quaternion  # 导入特定的 3D 绘图工具类
import matplotlib as mpl  # 导入 matplotlib 库的核心模块
from matplotlib.backend_bases import (MouseButton, MouseEvent,
                                      NavigationToolbar2)  # 导入 matplotlib 的基础类
from matplotlib import cm  # 导入 colormap 相关的功能
from matplotlib import colors as mcolors, patches as mpatch  # 导入颜色和图形补丁相关的功能
from matplotlib.testing.decorators import image_comparison, check_figures_equal  # 导入测试装饰器
from matplotlib.testing.widgets import mock_event  # 导入模拟事件的函数
from matplotlib.collections import LineCollection, PolyCollection  # 导入线集合和多边形集合类
from matplotlib.patches import Circle, PathPatch  # 导入圆和路径补丁类
from matplotlib.path import Path  # 导入路径类
from matplotlib.text import Text  # 导入文本类

import matplotlib.pyplot as plt  # 导入 matplotlib 的绘图接口
import numpy as np  # 导入 NumPy 库，用于数值计算


mpl3d_image_comparison = functools.partial(
    image_comparison, remove_text=True, style='default')  # 创建带有默认设置的 3D 图像比较函数


def plot_cuboid(ax, scale):
    # 在 3D 坐标轴上绘制一个长方体，其尺寸由 scale 参数决定 (x, y, z)
    r = [0, 1]
    pts = itertools.combinations(np.array(list(itertools.product(r, r, r))), 2)
    for start, end in pts:
        if np.sum(np.abs(start - end)) == r[1] - r[0]:
            ax.plot3D(*zip(start*np.array(scale), end*np.array(scale)))


@check_figures_equal(extensions=["png"])
def test_invisible_axes(fig_test, fig_ref):
    ax = fig_test.subplots(subplot_kw=dict(projection='3d'))
    ax.set_visible(False)  # 设置 3D 坐标轴不可见


@mpl3d_image_comparison(['grid_off.png'], style='mpl20')
def test_grid_off():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.grid(False)  # 关闭 3D 坐标轴的网格线


@mpl3d_image_comparison(['invisible_ticks_axis.png'], style='mpl20')
def test_invisible_ticks_axis():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xticks([])  # 设置 x 轴刻度为空列表，即不显示 x 轴刻度
    ax.set_yticks([])  # 设置 y 轴刻度为空列表，即不显示 y 轴刻度
    ax.set_zticks([])  # 设置 z 轴刻度为空列表，即不显示 z 轴刻度
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_visible(False)  # 设置坐标轴线不可见


@mpl3d_image_comparison(['axis_positions.png'], remove_text=False, style='mpl20')
def test_axis_positions():
    positions = ['upper', 'lower', 'both', 'none']
    fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'})
    for ax, pos in zip(axs.flatten(), positions):
        for axis in ax.xaxis, ax.yaxis, ax.zaxis:
            axis.set_label_position(pos)  # 设置坐标轴标签位置
            axis.set_ticks_position(pos)  # 设置坐标轴刻度位置
        title = f'{pos}'
        ax.set(xlabel='x', ylabel='y', zlabel='z', title=title)  # 设置坐标轴标签和标题


@mpl3d_image_comparison(['aspects.png'], remove_text=False, style='mpl20')
def test_aspects():
    aspects = ('auto', 'equal', 'equalxy', 'equalyz', 'equalxz', 'equal')
    _, axs = plt.subplots(2, 3, subplot_kw={'projection': '3d'})

    for ax in axs.flatten()[0:-1]:
        plot_cuboid(ax, scale=[1, 1, 5])  # 绘制长方体，覆盖 github #25443
    # 绘制一个立方体，以覆盖 github #25443
    plot_cuboid(axs[1][2], scale=[1, 1, 1])

    for i, ax in enumerate(axs.flatten()):
        ax.set_title(aspects[i])  # 设置子图标题
        ax.set_box_aspect((3, 4, 5))  # 设置坐标轴盒子的长宽比
        ax.set_aspect(aspects[i], adjustable='datalim')  # 设置坐标轴的等比例缩放方式
    # 在 axs 列表中的第二行第三列位置设置标题为 'equal (cube)'
    axs[1][2].set_title('equal (cube)')
# 使用装饰器生成包含单个图像比较的测试函数
@mpl3d_image_comparison(['aspects_adjust_box.png'],
                        remove_text=False, style='mpl20')
def test_aspects_adjust_box():
    # 定义不同的比例模式
    aspects = ('auto', 'equal', 'equalxy', 'equalyz', 'equalxz')
    # 创建包含多个子图的图形对象，每个子图具有 3D 投影
    fig, axs = plt.subplots(1, len(aspects), subplot_kw={'projection': '3d'},
                            figsize=(11, 3))

    # 遍历子图列表，为每个子图绘制立方体，并设置标题和比例
    for i, ax in enumerate(axs):
        plot_cuboid(ax, scale=[4, 3, 5])
        ax.set_title(aspects[i])
        ax.set_aspect(aspects[i], adjustable='box')


def test_axes3d_repr():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个 3D 投影的子图
    ax = fig.add_subplot(projection='3d')
    # 设置子图的标签、标题以及三个坐标轴的标签
    ax.set_label('label')
    ax.set_title('title')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # 断言子图的字符串表示是否符合预期格式
    assert repr(ax) == (
        "<Axes3D: label='label', "
        "title={'center': 'title'}, xlabel='x', ylabel='y', zlabel='z'>")


@mpl3d_image_comparison(['axes3d_primary_views.png'], style='mpl20',
                        tol=0.05 if platform.machine() == "arm64" else 0)
def test_axes3d_primary_views():
    # 定义多个视图参数组合 (elev, azim, roll)
    views = [(90, -90, 0),  # XY
             (0, -90, 0),   # XZ
             (0, 0, 0),     # YZ
             (-90, 90, 0),  # -XY
             (0, 90, 0),    # -XZ
             (0, 180, 0)]   # -YZ
    # 创建包含多个子图的图形对象，每个子图具有 3D 投影
    fig, axs = plt.subplots(2, 3, subplot_kw={'projection': '3d'})
    # 遍历子图列表，为每个子图设置坐标轴标签、正交投影类型和视角
    for i, ax in enumerate(axs.flat):
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_proj_type('ortho')
        ax.view_init(elev=views[i][0], azim=views[i][1], roll=views[i][2])
    # 调整子图布局使其紧凑显示
    plt.tight_layout()


@mpl3d_image_comparison(['bar3d.png'], style='mpl20')
def test_bar3d():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个 3D 投影的子图
    ax = fig.add_subplot(projection='3d')
    # 使用循环为子图添加不同颜色的 3D 条形图
    for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
        xs = np.arange(20)
        ys = np.arange(20)
        cs = [c] * len(xs)
        cs[0] = 'c'
        ax.bar(xs, ys, zs=z, zdir='y', align='edge', color=cs, alpha=0.8)


def test_bar3d_colors():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个 3D 投影的子图
    ax = fig.add_subplot(projection='3d')
    # 使用循环为子图添加不同命名颜色的 3D 条形图
    for c in ['red', 'green', 'blue', 'yellow']:
        xs = np.arange(len(c))
        ys = np.zeros_like(xs)
        zs = np.zeros_like(ys)
        # 色彩名称与 xs/ys/zs 的长度相同，不会被分割成单个字母
        ax.bar3d(xs, ys, zs, 1, 1, 1, color=c)


@mpl3d_image_comparison(['bar3d_shaded.png'], style='mpl20')
def test_bar3d_shaded():
    # 创建网格坐标
    x = np.arange(4)
    y = np.arange(5)
    x2d, y2d = np.meshgrid(x, y)
    x2d, y2d = x2d.ravel(), y2d.ravel()
    # 计算每个柱状体的高度，避免深度为零的盒子
    z = x2d + y2d + 1  
    # 定义多个视图参数组合 (elev, azim, roll)
    views = [(30, -60, 0), (30, 30, 30), (-30, 30, -90), (300, -30, 0)]
    # 创建图形对象，每个子图具有不同的 3D 投影
    fig = plt.figure(figsize=plt.figaspect(1 / len(views)))
    axs = fig.subplots(
        1, len(views),
        subplot_kw=dict(projection='3d')
    )
    for ax, (elev, azim, roll) in zip(axs, views):
        # 遍历 axs 和 views 列表，其中 ax 是当前的坐标轴对象，(elev, azim, roll) 是 views 中的元组解包
        ax.bar3d(x2d, y2d, x2d * 0, 1, 1, z, shade=True)
        # 在当前的坐标轴上绘制一个三维柱状图，使用给定的参数：x2d, y2d 为柱状图的位置坐标，x2d * 0 为柱状图的底部高度，1, 1 为柱状图的宽度和深度，z 为柱状图的高度，shade=True 表示使用阴影效果
        ax.view_init(elev=elev, azim=azim, roll=roll)
        # 设置当前坐标轴的视角，elev 为仰角，azim 为方位角，roll 为滚动角度
    fig.canvas.draw()
    # 绘制整个图形的画布
@mpl3d_image_comparison(['bar3d_notshaded.png'], style='mpl20')
def test_bar3d_notshaded():
    # 创建一个新的 Matplotlib 图形对象
    fig = plt.figure()
    # 添加一个带有 3D 投影的子图
    ax = fig.add_subplot(projection='3d')
    # 创建一个长度为 4 的一维数组 x 和长度为 5 的一维数组 y
    x = np.arange(4)
    y = np.arange(5)
    # 使用 meshgrid 函数将 x 和 y 转换为二维数组
    x2d, y2d = np.meshgrid(x, y)
    # 将二维数组展平为一维数组
    x2d, y2d = x2d.ravel(), y2d.ravel()
    # 计算柱状图的高度 z，这里使用了 x 和 y 的加和作为高度
    z = x2d + y2d
    # 在 3D 坐标系中绘制柱状图，shade 参数设置为 False，不进行阴影渲染
    ax.bar3d(x2d, y2d, x2d * 0, 1, 1, z, shade=False)
    # 绘制完成后刷新图形
    fig.canvas.draw()


def test_bar3d_lightsource():
    # 创建一个新的 Matplotlib 图形对象
    fig = plt.figure()
    # 添加一个带有 3D 投影的子图
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # 创建一个光源对象，设置其方位角为 0 度，仰角为 90 度
    ls = mcolors.LightSource(azdeg=0, altdeg=90)

    length, width = 3, 4
    area = length * width

    # 使用 meshgrid 函数创建 x 和 y 的网格
    x, y = np.meshgrid(np.arange(length), np.arange(width))
    # 将网格展平为一维数组
    x = x.ravel()
    y = y.ravel()
    # 计算柱状图的高度 dz
    dz = x + y

    # 生成一组颜色，每个颜色对应一个柱体，使用 coolwarm 颜色映射
    color = [cm.coolwarm(i/area) for i in range(area)]

    # 在 3D 坐标系中绘制带有阴影和光源效果的柱状图
    collection = ax.bar3d(x=x, y=y, z=0,
                          dx=1, dy=1, dz=dz,
                          color=color, shade=True, lightsource=ls)

    # 进行颜色测试，确保自定义的 90° 光源相对于默认光源产生不同的顶部颜色效果
    # 并且这些颜色与 colormap 的颜色非常接近（在浮点数舍入误差内）
    np.testing.assert_array_max_ulp(color, collection._facecolor3d[1::6], 4)


@mpl3d_image_comparison(
    ['contour3d.png'], style='mpl20',
    tol=0.002 if platform.machine() in ('aarch64', 'ppc64le', 's390x') else 0)
def test_contour3d():
    # 设置 Matplotlib 的默认参数，自动调整 3D 坐标轴的边距（在重新生成图像时需要移除）
    plt.rcParams['axes3d.automargin'] = True
    # 创建一个新的 Matplotlib 图形对象
    fig = plt.figure()
    # 添加一个带有 3D 投影的子图
    ax = fig.add_subplot(projection='3d')
    # 使用 axes3d 模块的 get_test_data 函数获取测试数据 X, Y, Z
    X, Y, Z = axes3d.get_test_data(0.05)
    # 在 3D 坐标系中绘制等高线图，沿 z 轴偏移 -100，使用 coolwarm 颜色映射
    ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    # 在 3D 坐标系中绘制等高线图，沿 x 轴偏移 -40，使用 coolwarm 颜色映射
    ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    # 在 3D 坐标系中绘制等高线图，沿 y 轴偏移 40，使用 coolwarm 颜色映射
    ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
    # 设置 3D 坐标轴的范围
    ax.axis(xmin=-40, xmax=40, ymin=-40, ymax=40, zmin=-100, zmax=100)


@mpl3d_image_comparison(['contour3d_extend3d.png'], style='mpl20')
def test_contour3d_extend3d():
    # 设置 Matplotlib 的默认参数，自动调整 3D 坐标轴的边距（在重新生成图像时需要移除）
    plt.rcParams['axes3d.automargin'] = True
    # 创建一个新的 Matplotlib 图形对象
    fig = plt.figure()
    # 添加一个带有 3D 投影的子图
    ax = fig.add_subplot(projection='3d')
    # 使用 axes3d 模块的 get_test_data 函数获取测试数据 X, Y, Z
    X, Y, Z = axes3d.get_test_data(0.05)
    # 在 3D 坐标系中绘制填充的等高线图，沿 z 轴偏移 -100，使用 coolwarm 颜色映射
    ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm, extend3d=True)
    # 设置 3D 坐标轴的范围
    ax.set_xlim(-30, 30)
    ax.set_ylim(-20, 40)
    ax.set_zlim(-80, 80)


@mpl3d_image_comparison(['contourf3d.png'], style='mpl20')
def test_contourf3d():
    # 设置 Matplotlib 的默认参数，自动调整 3D 坐标轴的边距（在重新生成图像时需要移除）
    plt.rcParams['axes3d.automargin'] = True
    # 创建一个新的 Matplotlib 图形对象
    fig = plt.figure()
    # 添加一个带有 3D 投影的子图
    ax = fig.add_subplot(projection='3d')
    # 使用 axes3d 模块的 get_test_data 函数获取测试数据 X, Y, Z
    X, Y, Z = axes3d.get_test_data(0.05)
    # 在 3D 坐标系中绘制填充的等高线图，沿 z 轴偏移 -100，使用 coolwarm 颜色映射
    ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    # 在 3D 坐标系中绘制填充的等高线图，沿 x 轴偏移 -40，使用 coolwarm 颜色映射
    ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    # 在 3D 坐标系中绘制填充的等高线图，沿 y 轴偏移 40，使用 coolwarm 颜色映射
    ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
    # 设置 3D 坐标轴的范围
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-100, 100)


@mpl3d_image_comparison(['contourf3d_fill.png'], style='mpl20')
def test_contourf3d_fill():
    # 设置3D图形的自动边距，这在重新生成图像时需要移除
    plt.rcParams['axes3d.automargin'] = True  # Remove when image is regenerated
    
    # 创建一个新的图形对象
    fig = plt.figure()
    
    # 添加一个3D子图到当前图形中
    ax = fig.add_subplot(projection='3d')
    
    # 创建一个 X-Y 平面的网格，范围是从-2到2，步长为0.25
    X, Y = np.meshgrid(np.arange(-2, 2, 0.25), np.arange(-2, 2, 0.25))
    
    # 创建一个与 X 形状相同的数组，并将所有值都设为0
    Z = X.clip(0, 0)
    
    # 在 z=0 平面上创建一些孔洞，这可以导致渲染错误，除非 Poly3DCollection
    # 具有路径代码信息（问题＃4784）
    Z[::5, ::5] = 0.1
    
    # 在3D坐标轴上绘制填充等高线图
    ax.contourf(X, Y, Z, offset=0, levels=[-0.1, 0], cmap=cm.coolwarm)
    
    # 设置 x 轴的显示范围
    ax.set_xlim(-2, 2)
    
    # 设置 y 轴的显示范围
    ax.set_ylim(-2, 2)
    
    # 设置 z 轴的显示范围
    ax.set_zlim(-1, 1)
@pytest.mark.parametrize('extend, levels', [['both', [2, 4, 6]],
                                            ['min', [2, 4, 6, 8]],
                                            ['max', [0, 2, 4, 6]]])
@check_figures_equal(extensions=["png"])
def test_contourf3d_extend(fig_test, fig_ref, extend, levels):
    X, Y = np.meshgrid(np.arange(-2, 2, 0.25), np.arange(-2, 2, 0.25))
    # Z is in the range [0, 8]
    Z = X**2 + Y**2

    # Manually set the over/under colors to be the end of the colormap
    cmap = mpl.colormaps['viridis'].copy()
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(255))
    # Set vmin/max to be the min/max values plotted on the reference image
    kwargs = {'vmin': 1, 'vmax': 7, 'cmap': cmap}

    # Add a 3D subplot to the reference figure and plot contour filled plot
    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.contourf(X, Y, Z, levels=[0, 2, 4, 6, 8], **kwargs)

    # Add a 3D subplot to the test figure and plot contour filled plot with parameters
    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.contourf(X, Y, Z, levels, extend=extend, **kwargs)

    # Set limits for all axes in both reference and test subplots
    for ax in [ax_ref, ax_test]:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-10, 10)


@mpl3d_image_comparison(['tricontour.png'], tol=0.02, style='mpl20')
def test_tricontour():
    plt.rcParams['axes3d.automargin'] = True  # Remove when image is regenerated
    fig = plt.figure()

    np.random.seed(19680801)
    x = np.random.rand(1000) - 0.5
    y = np.random.rand(1000) - 0.5
    z = -(x**2 + y**2)

    # Add a 3D subplot for the first part of the tricontour plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.tricontour(x, y, z)

    # Add a 3D subplot for the filled part of the tricontour plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.tricontourf(x, y, z)


def test_contour3d_1d_input():
    # Check that 1D sequences of different length for {x, y} doesn't error
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    nx, ny = 30, 20
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-10, 10, ny)
    z = np.random.randint(0, 2, [ny, nx])
    ax.contour(x, y, z, [0.5])


@mpl3d_image_comparison(['lines3d.png'], style='mpl20')
def test_lines3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z ** 2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    ax.plot(x, y, z)


@check_figures_equal(extensions=["png"])
def test_plot_scalar(fig_test, fig_ref):
    # Add a 3D subplot to the test figure and plot a scalar point
    ax1 = fig_test.add_subplot(projection='3d')
    ax1.plot([1], [1], "o")

    # Add a 3D subplot to the reference figure and plot a scalar point
    ax2 = fig_ref.add_subplot(projection='3d')
    ax2.plot(1, 1, "o")


def test_invalid_line_data():
    # Test cases with invalid Line3D inputs raising expected errors
    with pytest.raises(RuntimeError, match='x must be'):
        art3d.Line3D(0, [], [])
    with pytest.raises(RuntimeError, match='y must be'):
        art3d.Line3D([], 0, [])
    with pytest.raises(RuntimeError, match='z must be'):
        art3d.Line3D([], [], 0)

    line = art3d.Line3D([], [], [])
    with pytest.raises(RuntimeError, match='x must be'):
        line.set_data_3d(0, [], [])
    with pytest.raises(RuntimeError, match='y must be'):
        line.set_data_3d([], 0, [])
    # 使用 pytest 模块中的 `raises` 方法来验证代码块中是否引发了 RuntimeError 异常，并检查异常消息是否匹配特定模式 'z must be'
    with pytest.raises(RuntimeError, match='z must be'):
        # 调用 `set_data_3d` 方法，传入空列表作为参数，并设置第三个参数为 0
        line.set_data_3d([], [], 0)
@mpl3d_image_comparison(['mixedsubplot.png'], style='mpl20')
def test_mixedsubplots():
    # 定义函数 f(t)，返回 np.cos(2*np.pi*t) * np.exp(-t) 的值
    def f(t):
        return np.cos(2*np.pi*t) * np.exp(-t)

    # 生成 t1 和 t2 的值，分别为从 0 到 5，步长为 0.1 和 0.02
    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)

    # 设置绘图参数，调整 axes3d.automargin 为 True（重新生成图像时需要移除）
    plt.rcParams['axes3d.automargin'] = True
    # 创建一个 2x1 的图形，并指定为 fig 对象
    fig = plt.figure(figsize=plt.figaspect(2.))
    # 在 fig 上添加第一个子图 ax，并绘制 t1 和 t2 对应的函数曲线
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t1, f(t1), 'bo', t2, f(t2), 'k--', markerfacecolor='green')
    ax.grid(True)

    # 在 fig 上添加第二个子图 ax，并设置为 3D 投影
    ax = fig.add_subplot(2, 1, 2, projection='3d')
    # 创建网格 X 和 Y，范围为 -5 到 5，步长为 0.25
    X, Y = np.meshgrid(np.arange(-5, 5, 0.25), np.arange(-5, 5, 0.25))
    # 计算 R 和 Z 分别为 np.hypot(X, Y) 和 np.sin(R)
    R = np.hypot(X, Y)
    Z = np.sin(R)

    # 绘制 3D 曲面图，设置 rcount 和 ccount 为 40，linewidth 为 0，抗锯齿关闭
    ax.plot_surface(X, Y, Z, rcount=40, ccount=40,
                    linewidth=0, antialiased=False)

    # 设置 Z 轴的显示范围为 -1 到 1
    ax.set_zlim3d(-1, 1)


@check_figures_equal(extensions=['png'])
def test_tight_layout_text(fig_test, fig_ref):
    # 对 tight_layout 忽略 text() 函数的效果进行测试
    ax1 = fig_test.add_subplot(projection='3d')
    ax1.text(.5, .5, .5, s='some string')
    fig_test.tight_layout()

    # 创建 fig_ref 上的子图 ax2，设置为 3D 投影，并对 fig_ref 进行 tight_layout
    ax2 = fig_ref.add_subplot(projection='3d')
    fig_ref.tight_layout()
    ax2.text(.5, .5, .5, s='some string')


@mpl3d_image_comparison(['scatter3d.png'], style='mpl20')
def test_scatter3d():
    # 设置绘图参数，调整 axes3d.automargin 为 True（重新生成图像时需要移除）
    plt.rcParams['axes3d.automargin'] = True
    # 创建一个新的图形对象 fig
    fig = plt.figure()
    # 在 fig 上添加一个 3D 投影的子图 ax
    ax = fig.add_subplot(projection='3d')
    # 绘制第一个散点图，设置颜色为红色，标记为圆圈
    ax.scatter(np.arange(10), np.arange(10), np.arange(10),
               c='r', marker='o')
    # 创建新的数组 x, y, z，分别从 10 到 20
    x = y = z = np.arange(10, 20)
    # 绘制第二个散点图，设置颜色为蓝色，标记为三角形
    ax.scatter(x, y, z, c='b', marker='^')
    # 修改 z[-1] 的值为 0，检查 scatter() 函数是否复制数据
    z[-1] = 0
    # 确保空的散点图不会导致错误
    ax.scatter([], [], [], c='r', marker='X')


@mpl3d_image_comparison(['scatter3d_color.png'], style='mpl20')
def test_scatter3d_color():
    # 设置绘图参数，调整 axes3d.automargin 为 True（重新生成图像时需要移除）
    plt.rcParams['axes3d.automargin'] = True
    # 创建一个新的图形对象 fig
    fig = plt.figure()
    # 在 fig 上添加一个 3D 投影的子图 ax
    ax = fig.add_subplot(projection='3d')

    # 检查 'none' 颜色的效果，两者应该叠加以产生与仅设置 `color` 相同的效果
    ax.scatter(np.arange(10), np.arange(10), np.arange(10),
               facecolor='r', edgecolor='none', marker='o')
    ax.scatter(np.arange(10), np.arange(10), np.arange(10),
               facecolor='none', edgecolor='r', marker='o')

    # 绘制第三个散点图，设置颜色为蓝色，标记为正方形
    ax.scatter(np.arange(10, 20), np.arange(10, 20), np.arange(10, 20),
               color='b', marker='s')


@mpl3d_image_comparison(['scatter3d_linewidth.png'], style='mpl20')
def test_scatter3d_linewidth():
    # 创建一个新的图形对象 fig
    fig = plt.figure()
    # 在 fig 上添加一个 3D 投影的子图 ax
    ax = fig.add_subplot(projection='3d')

    # 检查数组样式的 linewidth 是否能够设置
    ax.scatter(np.arange(10), np.arange(10), np.arange(10),
               marker='o', linewidth=np.arange(10))


@check_figures_equal(extensions=['png'])
def test_scatter3d_linewidth_modification(fig_ref, fig_test):
    # 修改 Path3DCollection 的 linewidths 为数组样式后进行测试
    # （该测试未提供完整的代码，在这里不需要额外的注释）
    # 在测试图中添加一个3D子图，设置投影为3D
    ax_test = fig_test.add_subplot(projection='3d')

    # 在测试图中绘制一个散点图，使用10个数据点，设置标记为圆形
    c = ax_test.scatter(np.arange(10), np.arange(10), np.arange(10),
                        marker='o')

    # 设置散点的边界线宽度，使用从0到9的整数数组
    c.set_linewidths(np.arange(10))

    # 在参考图中添加一个3D子图，设置投影为3D
    ax_ref = fig_ref.add_subplot(projection='3d')

    # 在参考图中绘制一个散点图，使用10个数据点，设置标记为圆形，同时设置边界线宽度
    ax_ref.scatter(np.arange(10), np.arange(10), np.arange(10), marker='o',
                   linewidths=np.arange(10))
@check_figures_equal(extensions=['png'])
def test_scatter3d_modification(fig_ref, fig_test):
    """
    在修改后正确处理 Path3DCollection 属性。
    """
    # 在 fig_test 中添加一个 3D 子图
    ax_test = fig_test.add_subplot(projection='3d')
    # 创建一个散点图，并设置属性
    c = ax_test.scatter(np.arange(10), np.arange(10), np.arange(10),
                        marker='o')
    c.set_facecolor('C1')  # 设置散点的填充色为 'C1'
    c.set_edgecolor('C2')  # 设置散点的边缘色为 'C2'
    c.set_alpha([0.3, 0.7] * 5)  # 设置散点的透明度数组
    assert c.get_depthshade()  # 断言深度阴影默认开启
    c.set_depthshade(False)  # 关闭散点的深度阴影
    assert not c.get_depthshade()  # 再次断言深度阴影已关闭
    c.set_sizes(np.full(10, 75))  # 设置散点的大小为 75
    c.set_linewidths(3)  # 设置散点的边框宽度为 3

    # 在 fig_ref 中添加一个 3D 子图，并绘制与上述散点图相同属性的参考图
    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.scatter(np.arange(10), np.arange(10), np.arange(10), marker='o',
                   facecolor='C1', edgecolor='C2', alpha=[0.3, 0.7] * 5,
                   depthshade=False, s=75, linewidths=3)


@pytest.mark.parametrize('depthshade', [True, False])
@check_figures_equal(extensions=['png'])
def test_scatter3d_sorting(fig_ref, fig_test, depthshade):
    """
    测试散点属性在排序时的正确性。
    """
    # 创建一个网格数据
    y, x = np.mgrid[:10, :10]
    z = np.arange(x.size).reshape(x.shape)

    # 设置不同散点的大小、填充色、边缘色和边框宽度
    sizes = np.full(z.shape, 25)
    sizes[0::2, 0::2] = 100
    sizes[1::2, 1::2] = 100

    facecolors = np.full(z.shape, 'C0')
    facecolors[:5, :5] = 'C1'
    facecolors[6:, :4] = 'C2'
    facecolors[6:, 6:] = 'C3'

    edgecolors = np.full(z.shape, 'C4')
    edgecolors[1:5, 1:5] = 'C5'
    edgecolors[5:9, 1:5] = 'C6'
    edgecolors[5:9, 5:9] = 'C7'

    linewidths = np.full(z.shape, 2)
    linewidths[0::2, 0::2] = 5
    linewidths[1::2, 1::2] = 5

    # 将数据展平以便处理
    x, y, z, sizes, facecolors, edgecolors, linewidths = [
        a.flatten()
        for a in [x, y, z, sizes, facecolors, edgecolors, linewidths]
    ]

    # 在 fig_ref 中添加一个 3D 子图，并绘制排序后的散点图
    ax_ref = fig_ref.add_subplot(projection='3d')
    sets = (np.unique(a) for a in [sizes, facecolors, edgecolors, linewidths])
    for s, fc, ec, lw in itertools.product(*sets):
        subset = (
            (sizes != s) |
            (facecolors != fc) |
            (edgecolors != ec) |
            (linewidths != lw)
        )
        subset = np.ma.masked_array(z, subset, dtype=float)

        # 当关闭深度阴影时，颜色作为单项列表传递；
        # 下面的重塑操作是为了禁用单路径优化，
        # 因为全散点的优化应用于多种颜色。
        fc = np.repeat(fc, sum(~subset.mask))

        ax_ref.scatter(x, y, subset, s=s, fc=fc, ec=ec, lw=lw, alpha=1,
                       depthshade=depthshade)

    # 在 fig_test 中添加一个 3D 子图，并绘制测试时的散点图
    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.scatter(x, y, z, s=sizes, fc=facecolors, ec=edgecolors,
                    lw=linewidths, alpha=1, depthshade=depthshade)


@pytest.mark.parametrize('azim', [-50, 130])  # 黄色优先，蓝色优先
@check_figures_equal(extensions=['png'])
def test_marker_draw_order_data_reversed(fig_test, fig_ref, azim):
    """
    测试数据反转时标记的绘制顺序。
    """
    Test that the draw order does not depend on the data point order.

    For the given viewing angle at azim=-50, the yellow marker should be in
    front. For azim=130, the blue marker should be in front.
    """
    # 定义数据点的坐标
    x = [-1, 1]
    y = [1, -1]
    z = [0, 0]
    # 定义每个数据点对应的颜色
    color = ['b', 'y']
    # 在测试图中添加一个带有3D投影的子图
    ax = fig_test.add_subplot(projection='3d')
    # 在3D图中绘制散点图，指定数据点的坐标、大小和颜色
    ax.scatter(x, y, z, s=3500, c=color)
    # 设置视角，即观察角度的方位
    ax.view_init(elev=0, azim=azim, roll=0)
    # 在参考图中添加一个带有3D投影的子图
    ax = fig_ref.add_subplot(projection='3d')
    # 在3D图中绘制散点图，但反转数据点的顺序，以测试绘制顺序是否影响视觉效果
    ax.scatter(x[::-1], y[::-1], z[::-1], s=3500, c=color[::-1])
    # 设置视角，与测试图中相同的观察角度的方位
    ax.view_init(elev=0, azim=azim, roll=0)
@check_figures_equal(extensions=['png'])
def test_marker_draw_order_view_rotated(fig_test, fig_ref):
    """
    Test that the draw order changes with the direction.

    If we rotate *azim* by 180 degrees and exchange the colors, the plot
    plot should look the same again.
    """
    azim = 130  # 设置方位角为130度
    x = [-1, 1]  # 定义 x 坐标
    y = [1, -1]  # 定义 y 坐标
    z = [0, 0]  # 定义 z 坐标
    color = ['b', 'y']  # 定义颜色数组
    ax = fig_test.add_subplot(projection='3d')  # 在测试图上添加三维子图
    ax.set_axis_off()  # 关闭坐标轴显示
    ax.scatter(x, y, z, s=3500, c=color)  # 绘制散点图
    ax.view_init(elev=0, azim=azim, roll=0)  # 设置视角为0度仰角，130度方位角，0度滚动角
    ax = fig_ref.add_subplot(projection='3d')  # 在参考图上添加三维子图
    ax.set_axis_off()  # 关闭坐标轴显示
    ax.scatter(x, y, z, s=3500, c=color[::-1])  # 绘制散点图，颜色颠倒
    ax.view_init(elev=0, azim=azim - 180, roll=0)  # 设置视角为0度仰角，130度方位角减去180度，0度滚动角


@mpl3d_image_comparison(['plot_3d_from_2d.png'], tol=0.019, style='mpl20')
def test_plot_3d_from_2d():
    fig = plt.figure()  # 创建一个新的图形对象
    ax = fig.add_subplot(projection='3d')  # 在图形中添加三维子图
    xs = np.arange(0, 5)  # 创建一个从0到4的数组
    ys = np.arange(5, 10)  # 创建一个从5到9的数组
    ax.plot(xs, ys, zs=0, zdir='x')  # 沿着 x 方向绘制曲线


@mpl3d_image_comparison(['fill_between_quad.png'], style='mpl20')
def test_fill_between_quad():
    fig = plt.figure()  # 创建一个新的图形对象
    ax = fig.add_subplot(projection='3d')  # 在图形中添加三维子图

    theta = np.linspace(0, 2*np.pi, 50)  # 创建一个从0到2*pi的均匀分布数组

    x1 = np.cos(theta)  # 计算第一个曲面的 x 坐标
    y1 = np.sin(theta)  # 计算第一个曲面的 y 坐标
    z1 = 0.1 * np.sin(6 * theta)  # 计算第一个曲面的 z 坐标

    x2 = 0.6 * np.cos(theta)  # 计算第二个曲面的 x 坐标
    y2 = 0.6 * np.sin(theta)  # 计算第二个曲面的 y 坐标
    z2 = 2  # 第二个曲面的 z 坐标

    where = (theta < np.pi/2) | (theta > 3*np.pi/2)  # 定义一个布尔数组，选择曲面填充的区域

    # 因为 x1 == x2, y1 == y2, 或 z1 == z2 的情况均不成立，fill_between 方法将映射到 'quad' 模式
    ax.fill_between(x1, y1, z1, x2, y2, z2,
                    where=where, mode='auto', alpha=0.5, edgecolor='k')


@mpl3d_image_comparison(['fill_between_polygon.png'], style='mpl20')
def test_fill_between_polygon():
    fig = plt.figure()  # 创建一个新的图形对象
    ax = fig.add_subplot(projection='3d')  # 在图形中添加三维子图

    theta = np.linspace(0, 2*np.pi, 50)  # 创建一个从0到2*pi的均匀分布数组

    x1 = x2 = theta  # 设置 x1 和 x2 为相同的数组
    y1 = y2 = 0  # 设置 y1 和 y2 为0
    z1 = np.cos(theta)  # 计算第一个曲面的 z 坐标
    z2 = z1 + 1  # 计算第二个曲面的 z 坐标

    where = (theta < np.pi/2) | (theta > 3*np.pi/2)  # 定义一个布尔数组，选择曲面填充的区域

    # 因为 x1 == x2 和 y1 == y2 的情况均成立，fill_between 方法将映射到 'polygon' 模式
    ax.fill_between(x1, y1, z1, x2, y2, z2,
                    where=where, mode='auto', edgecolor='k')


@mpl3d_image_comparison(['surface3d.png'], style='mpl20')
def test_surface3d():
    # 在重新生成此测试图像时，请删除此行。
    plt.rcParams['pcolormesh.snap'] = False

    fig = plt.figure()  # 创建一个新的图形对象
    ax = fig.add_subplot(projection='3d')  # 在图形中添加三维子图
    X = np.arange(-5, 5, 0.25)  # 创建一个从-5到4.75的均匀分布数组
    Y = np.arange(-5, 5, 0.25)  # 创建一个从-5到4.75的均匀分布数组
    X, Y = np.meshgrid(X, Y)  # 根据 X 和 Y 生成网格坐标
    R = np.hypot(X, Y)  # 计算 X 和 Y 的欧几里得距离
    Z = np.sin(R)  # 计算 Z 值，使用欧几里得距离的正弦值
    surf = ax.plot_surface(X, Y, Z, rcount=40, ccount=40, cmap=cm.coolwarm,
                           lw=0, antialiased=False)  # 绘制三维曲面图
    plt.rcParams['axes3d.automargin'] = True  # 在重新生成图像时，请删除此行
    ax.set_zlim(-1.01, 1.01)  # 设置 Z 轴的限制
    fig.colorbar(surf, shrink=0.5, aspect=5)  # 在图形上添加颜色条
# 使用装饰器设置图像比较，用于测试 3D 表面图的标签偏移和刻度位置
@image_comparison(['surface3d_label_offset_tick_position.png'], style='mpl20')
def test_surface3d_label_offset_tick_position():
    # 设置绘图参数以自动调整3D坐标轴的边距，在重新生成图像时需移除此行
    plt.rcParams['axes3d.automargin'] = True
    # 创建一个带有3D投影的子图
    ax = plt.figure().add_subplot(projection="3d")

    # 创建 x, y 网格，范围是 [0, 6*pi] 和 [0, 4*pi]，步长为 0.25
    x, y = np.mgrid[0:6 * np.pi:0.25, 0:4 * np.pi:0.25]
    # 计算 z 值，使用绝对值的余弦和
    z = np.sqrt(np.abs(np.cos(x) + np.cos(y)))

    # 绘制3D表面图，将 x, y, z 值乘以不同的倍数以改变比例，使用秋季色彩映射，步长为2
    ax.plot_surface(x * 1e5, y * 1e6, z * 1e8, cmap='autumn', cstride=2, rstride=2)
    # 设置 x, y, z 轴标签
    ax.set_xlabel("X label")
    ax.set_ylabel("Y label")
    ax.set_zlabel("Z label")

    # 绘图完成后更新图像的绘图区域
    ax.figure.canvas.draw()


# 使用装饰器设置图像比较，用于测试阴影的3D表面图
@mpl3d_image_comparison(['surface3d_shaded.png'], style='mpl20')
def test_surface3d_shaded():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个带有3D投影的子图
    ax = fig.add_subplot(projection='3d')

    # 创建 X, Y 值范围为 [-5, 5]，步长为 0.25 的网格
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    # 计算 R 和 Z 值，分别为半径和正弦函数
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)

    # 绘制3D表面图，使用步长为5的行和列步幅，颜色为浅绿色，线宽为1，关闭抗锯齿效果
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5, color=[0.25, 1, 0.25], lw=1, antialiased=False)
    # 在重新生成图像时需移除此行
    plt.rcParams['axes3d.automargin'] = True
    # 设置 Z 轴的限制范围
    ax.set_zlim(-1.01, 1.01)


# 使用装饰器设置图像比较，用于测试带屏蔽数据的3D表面图
@mpl3d_image_comparison(['surface3d_masked.png'], style='mpl20')
def test_surface3d_masked():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个带有3D投影的子图
    ax = fig.add_subplot(projection='3d')

    # 创建 x 和 y 值
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    y = [1, 2, 3, 4, 5, 6, 7, 8]

    # 创建 x, y 的网格
    x, y = np.meshgrid(x, y)
    # 创建带有屏蔽值的矩阵
    matrix = np.array([
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 1, 2, 3, 4, 4, 4, 3, 2, 1, 1],
        [-1, -1., 4, 5, 6, 8, 6, 5, 4, 3, -1.],
        [-1, -1., 7, 8, 11, 12, 11, 8, 7, -1., -1.],
        [-1, -1., 8, 9, 10, 16, 10, 9, 10, 7, -1.],
        [-1, -1., -1., 12, 16, 20, 16, 12, 11, -1., -1.],
        [-1, -1., -1., -1., 22, 24, 22, 20, 18, -1., -1.],
        [-1, -1., -1., -1., -1., 28, 26, 25, -1., -1., -1.],
    ])
    # 创建带有掩码的 z 值数组
    z = np.ma.masked_less(matrix, 0)
    # 根据 z 值的范围创建归一化对象
    norm = mcolors.Normalize(vmax=z.max(), vmin=z.min())
    # 根据归一化对象和色彩映射创建颜色数组
    colors = mpl.cm.ScalarMappable(norm=norm, cmap="plasma").to_rgba(z)
    # 绘制带有颜色数组的3D表面图
    ax.plot_surface(x, y, z, facecolors=colors)
    # 设置视角的初始位置
    ax.view_init(30, -80, 0)


# 使用装饰器设置图像比较，用于测试散点图掩码和3D表面图
@check_figures_equal(extensions=["png"])
def test_plot_scatter_masks(fig_test, fig_ref):
    # 创建一个线性空间为 [0, 10] 的 x 值，线性空间为 [0, 10] 的 y 值，以及其对应的正弦余弦值
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    z = np.sin(x) * np.cos(y)
    # 创建一个掩码，其中 z 值大于 0
    mask = z > 0

    # 创建带有掩码的 z 值数组
    z_masked = np.ma.array(z, mask=mask)
    # 在 fig_test 上创建一个带有3D投影的子图
    ax_test = fig_test.add_subplot(projection='3d')
    # 绘制散点图，使用掩码后的 z 值
    ax_test.scatter(x, y, z_masked)
    # 绘制线条图，使用掩码后的 x, y, z 值
    ax_test.plot(x, y, z_masked)

    # 将掩码中的值设置为 NaN
    x[mask] = y[mask] = z[mask] = np.nan
    # 在 fig_ref 上创建一个带有3D投影的子图
    ax_ref = fig_ref.add_subplot(projection='3d')
    # 绘制散点图，使用掩码后的 x, y, z 值
    ax_ref.scatter(x, y, z)
    # 绘制线条图，使用掩码后的 x, y, z 值
    ax_ref.plot(x, y, z)


# 使用装饰器设置图像比较，用于测试不带颜色参数的3D表面图
@check_figures_equal(extensions=["png"])
def test_plot_surface_None_arg(fig_test, fig_ref):
    # 创建一个网格，范围为 [0, 5] 的 x, y 值，以及其对应的 z 值
    x, y = np.meshgrid(np.arange(5), np.arange(5))
    z = x + y
    # 在 fig_test 上创建一个带有3D投影的子图，绘制表面图，不使用面颜色参数
    ax_test = fig_test.add_subplot(projection='3
@mpl3d_image_comparison(['surface3d_masked_strides.png'], style='mpl20')
def test_surface3d_masked_strides():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上添加一个三维子图
    ax = fig.add_subplot(projection='3d')

    # 生成二维网格数据
    x, y = np.mgrid[-6:6.1:1, -6:6.1:1]
    # 根据条件创建一个掩码数组
    z = np.ma.masked_less(x * y, 2)

    # 绘制三维表面图
    ax.plot_surface(x, y, z, rstride=4, cstride=4)
    # 设置视角
    ax.view_init(60, -45, 0)


@mpl3d_image_comparison(['text3d.png'], remove_text=False, style='mpl20')
def test_text3d():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上添加一个三维子图
    ax = fig.add_subplot(projection='3d')

    # 定义多个方向和坐标值
    zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
    xs = (2, 6, 4, 9, 7, 2)
    ys = (6, 4, 8, 7, 2, 2)
    zs = (4, 2, 5, 6, 1, 7)

    # 遍历并在图中添加多个文本标签
    for zdir, x, y, z in zip(zdirs, xs, ys, zs):
        label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir)
        ax.text(x, y, z, label, zdir)

    # 在指定位置添加红色文本标签
    ax.text(1, 1, 1, "red", color='red')
    # 在二维位置添加文本标签
    ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes)
    # 更新图形参数以自动调整边距（用于重新生成图像时删除）
    plt.rcParams['axes3d.automargin'] = True
    # 设置三维坐标轴的限制和标签
    ax.set_xlim3d(0, 10)
    ax.set_ylim3d(0, 10)
    ax.set_zlim3d(0, 10)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')


@check_figures_equal(extensions=['png'])
def test_text3d_modification(fig_ref, fig_test):
    # 修改文本位置后应与直接设置相同
    zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
    xs = (2, 6, 4, 9, 7, 2)
    ys = (6, 4, 8, 7, 2, 2)
    zs = (4, 2, 5, 6, 1, 7)

    # 在测试图中添加子图并设置坐标轴范围
    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.set_xlim3d(0, 10)
    ax_test.set_ylim3d(0, 10)
    ax_test.set_zlim3d(0, 10)
    # 遍历并在测试图中添加多个文本标签
    for zdir, x, y, z in zip(zdirs, xs, ys, zs):
        t = ax_test.text(0, 0, 0, f'({x}, {y}, {z}), dir={zdir}')
        t.set_position_3d((x, y, z), zdir=zdir)

    # 在参考图中添加子图并设置坐标轴范围
    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.set_xlim3d(0, 10)
    ax_ref.set_ylim3d(0, 10)
    ax_ref.set_zlim3d(0, 10)
    # 遍历并在参考图中添加多个文本标签
    for zdir, x, y, z in zip(zdirs, xs, ys, zs):
        ax_ref.text(x, y, z, f'({x}, {y}, {z}), dir={zdir}', zdir=zdir)


@mpl3d_image_comparison(['trisurf3d.png'], tol=0.061, style='mpl20')
def test_trisurf3d():
    # 定义角度和半径数量
    n_angles = 36
    n_radii = 8
    # 根据角度和半径数量生成角度和半径数组
    radii = np.linspace(0.125, 1.0, n_radii)
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi/n_angles

    # 根据生成的角度、半径和公式计算出 x、y、z 坐标
    x = np.append(0, (radii*np.cos(angles)).flatten())
    y = np.append(0, (radii*np.sin(angles)).flatten())
    z = np.sin(-x*y)

    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上添加一个三维子图
    ax = fig.add_subplot(projection='3d')
    # 绘制三角形表面图
    ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.2)


@mpl3d_image_comparison(['trisurf3d_shaded.png'], tol=0.03, style='mpl20')
def test_trisurf3d_shaded():
    # 定义角度和半径数量
    n_angles = 36
    n_radii = 8
    # 根据角度和半径数量生成角度和半径数组
    radii = np.linspace(0.125, 1.0, n_radii)
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += np.pi/n_angles
    # 创建一个数组 x，包括一个额外的零和一系列角度乘以半径的余弦值，并展开成一维数组
    x = np.append(0, (radii*np.cos(angles)).flatten())
    
    # 创建一个数组 y，包括一个额外的零和一系列角度乘以半径的正弦值，并展开成一维数组
    y = np.append(0, (radii*np.sin(angles)).flatten())
    
    # 创建一个数组 z，包括 x 和 y 元素的乘积的负正弦值
    z = np.sin(-x*y)
    
    # 创建一个新的三维图形对象
    fig = plt.figure()
    
    # 添加一个三维子图到图形中，使用三维投影
    ax = fig.add_subplot(projection='3d')
    
    # 在三维坐标系上绘制三角面图，使用 x, y, z 数组作为数据点，设置颜色为 [1, 0.5, 0]，线宽为 0.2
    ax.plot_trisurf(x, y, z, color=[1, 0.5, 0], linewidth=0.2)
# 使用装饰器 mpl3d_image_comparison，比较生成的图像和预期图像是否相同，文件名为 'wireframe3d.png'，风格设置为 'mpl20'
@mpl3d_image_comparison(['wireframe3d.png'], style='mpl20')
# 定义测试函数 test_wireframe3d
def test_wireframe3d():
    # 创建一个新的三维图形对象
    fig = plt.figure()
    # 在图形上添加一个三维子图
    ax = fig.add_subplot(projection='3d')
    # 从模块 axes3d 获取测试数据 X, Y, Z
    X, Y, Z = axes3d.get_test_data(0.05)
    # 绘制三维线框图，设置行数和列数为 13
    ax.plot_wireframe(X, Y, Z, rcount=13, ccount=13)


@mpl3d_image_comparison(['wireframe3dzerocstride.png'], style='mpl20')
def test_wireframe3dzerocstride():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    # 绘制三维线框图，设置行数为 13，列数为 0
    ax.plot_wireframe(X, Y, Z, rcount=13, ccount=0)


@mpl3d_image_comparison(['wireframe3dzerorstride.png'], style='mpl20')
def test_wireframe3dzerorstride():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    # 绘制三维线框图，设置行步长为 0，列步长为 10
    ax.plot_wireframe(X, Y, Z, rstride=0, cstride=10)


# 定义测试函数 test_wireframe3dzerostrideraises
def test_wireframe3dzerostrideraises():
    # 创建一个新的三维图形对象
    fig = plt.figure()
    # 在图形上添加一个三维子图
    ax = fig.add_subplot(projection='3d')
    # 从模块 axes3d 获取测试数据 X, Y, Z
    X, Y, Z = axes3d.get_test_data(0.05)
    # 使用 pytest 来断言绘制三维线框图时设置行步长和列步长为 0 会引发 ValueError 异常
    with pytest.raises(ValueError):
        ax.plot_wireframe(X, Y, Z, rstride=0, cstride=0)


# 定义测试函数 test_mixedsamplesraises
def test_mixedsamplesraises():
    # 创建一个新的三维图形对象
    fig = plt.figure()
    # 在图形上添加一个三维子图
    ax = fig.add_subplot(projection='3d')
    # 从模块 axes3d 获取测试数据 X, Y, Z
    X, Y, Z = axes3d.get_test_data(0.05)
    # 使用 pytest 来断言绘制三维线框图时设置行步长为 10，列数为 50 会引发 ValueError 异常
    with pytest.raises(ValueError):
        ax.plot_wireframe(X, Y, Z, rstride=10, ccount=50)
    # 使用 pytest 来断言绘制三维表面图时设置列步长为 50，行数为 10 会引发 ValueError 异常
    with pytest.raises(ValueError):
        ax.plot_surface(X, Y, Z, cstride=50, rcount=10)


# 使用装饰器 mpl3d_image_comparison，比较生成的图像和预期图像是否相同，文件名为 'quiver3d.png'，风格设置为 'mpl20'，容差为 0.003
@mpl3d_image_comparison(['quiver3d.png'], style='mpl20', tol=0.003)
def test_quiver3d():
    # 设置绘图参数，自动调整三维坐标轴的边距，重新生成图像时需要移除此设置
    plt.rcParams['axes3d.automargin'] = True  # Remove when image is regenerated
    # 创建一个新的三维图形对象
    fig = plt.figure()
    # 在图形上添加一个三维子图
    ax = fig.add_subplot(projection='3d')
    # 定义箭头的位置：尖端、中部、尾端
    pivots = ['tip', 'middle', 'tail']
    # 定义箭头的颜色
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    # 遍历箭头位置和颜色，绘制箭头和对应的散点
    for i, (pivot, color) in enumerate(zip(pivots, colors)):
        # 创建网格，定义箭头的起点坐标
        x, y, z = np.meshgrid([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5])
        # 定义箭头的方向向量
        u = -x
        v = -y
        w = -z
        # 每个箭头集合在 z 方向上偏移 2*i
        z += 2 * i
        # 绘制三维箭头图
        ax.quiver(x, y, z, u, v, w, length=1, pivot=pivot, color=color)
        # 绘制散点图
        ax.scatter(x, y, z, color=color)
    # 设置 x 轴的显示范围
    ax.set_xlim(-3, 3)
    # 设置 y 轴的显示范围
    ax.set_ylim(-3, 3)
    # 设置 z 轴的显示范围
    ax.set_zlim(-1, 5)


# 使用装饰器 check_figures_equal，比较生成的测试图形和参考图形是否相同，文件扩展名为 "png"
@check_figures_equal(extensions=["png"])
def test_quiver3d_empty(fig_test, fig_ref):
    # 在参考图形上添加一个三维子图
    fig_ref.add_subplot(projection='3d')
    # 定义空的 x, y, z, u, v, w 数组
    x = y = z = u = v = w = []
    # 在测试图形上添加一个三维子图
    ax = fig_test.add_subplot(projection='3d')
    # 绘制空的三维箭头图
    ax.quiver(x, y, z, u, v, w, length=0.1, pivot='tip', normalize=True)


@mpl3d_image_comparison(['quiver3d_masked.png'], style='mpl20')
def test_quiver3d_masked():
    # 创建一个新的三维图形对象
    fig = plt.figure()
    # 在图形上添加一个三维子图
    ax = fig.add_subplot(projection='3d')

    # 使用 mgrid 创建网格，因为 masked_where 不太喜欢广播...
    x, y, z = np.mgrid[-1:0.8:10j, -1:0.8:10j, -1:0.6:3j]

    # 定义箭头的方向向量
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (2/3)**0.5 * np.cos(np.pi * x) * np.cos(np.pi * y) * np.sin(np.pi * z)

    # 绘制三维箭头图
    ax.quiver(x, y, z, u, v, w)
    # 使用 NumPy 中的掩码数组功能，将 u 中满足条件的元素设为掩码值
    u = np.ma.masked_where((-0.4 < x) & (x < 0.1), u, copy=False)
    # 使用 NumPy 中的掩码数组功能，将 v 中满足条件的元素设为掩码值
    v = np.ma.masked_where((0.1 < y) & (y < 0.7), v, copy=False)
    
    # 在 3D 坐标系上绘制箭头图，箭头的起点为 (x, y, z)，方向由 (u, v, w) 给出，
    # 箭头长度为 0.1，箭头末端指向，向量标准化
    ax.quiver(x, y, z, u, v, w, length=0.1, pivot='tip', normalize=True)
@mpl3d_image_comparison(['quiver3d_colorcoded.png'], style='mpl20')
def test_quiver3d_colorcoded():
    # 创建一个新的 Matplotlib 图形对象
    fig = plt.figure()
    # 添加一个带有 3D 投影的子图
    ax = fig.add_subplot(projection='3d')

    # 初始化一些数据
    x = y = dx = dz = np.zeros(10)
    z = dy = np.arange(10.)

    # 根据 dy 的值生成颜色映射
    color = plt.cm.Reds(dy/dy.max())
    # 绘制 3D 矢量图
    ax.quiver(x, y, z, dx, dy, dz, colors=color)
    # 设置 y 轴的范围
    ax.set_ylim(0, 10)


def test_patch_modification():
    # 创建一个新的 Matplotlib 图形对象
    fig = plt.figure()
    # 添加一个带有 3D 投影的子图
    ax = fig.add_subplot(projection="3d")
    # 创建一个圆形 Patch 对象，并添加到子图中
    circle = Circle((0, 0))
    ax.add_patch(circle)
    # 将 2D 图形转换为 3D 图形
    art3d.patch_2d_to_3d(circle)
    # 设置圆形的填充颜色为红色
    circle.set_facecolor((1.0, 0.0, 0.0, 1))

    # 断言圆形的填充颜色与预期相同
    assert mcolors.same_color(circle.get_facecolor(), (1, 0, 0, 1))
    # 刷新图形画布
    fig.canvas.draw()
    # 再次断言圆形的填充颜色与预期相同
    assert mcolors.same_color(circle.get_facecolor(), (1, 0, 0, 1))


@check_figures_equal(extensions=['png'])
def test_patch_collection_modification(fig_test, fig_ref):
    # 测试在创建后修改 Patch3DCollection 属性是否有效
    patch1 = Circle((0, 0), 0.05)
    patch2 = Circle((0.1, 0.1), 0.03)
    # 创建一个 Patch3DCollection 对象，包含两个圆形 Patch
    facecolors = np.array([[0., 0.5, 0., 1.], [0.5, 0., 0., 0.5]])
    c = art3d.Patch3DCollection([patch1, patch2], linewidths=3)

    # 在测试图中添加 3D 集合对象
    ax_test = fig_test.add_subplot(projection='3d')
    ax_test.add_collection3d(c)
    # 设置边缘颜色为 'C2'
    c.set_edgecolor('C2')
    # 设置填充颜色为 facecolors
    c.set_facecolor(facecolors)
    # 设置透明度为 0.7
    c.set_alpha(0.7)
    # 断言是否启用了深度阴影
    assert c.get_depthshade()
    # 禁用深度阴影
    c.set_depthshade(False)
    # 断言是否禁用了深度阴影
    assert not c.get_depthshade()

    # 重新定义 patch1 和 patch2，并创建一个新的 Patch3DCollection 对象
    patch1 = Circle((0, 0), 0.05)
    patch2 = Circle((0.1, 0.1), 0.03)
    facecolors = np.array([[0., 0.5, 0., 1.], [0.5, 0., 0., 0.5]])
    c = art3d.Patch3DCollection([patch1, patch2], linewidths=3,
                                edgecolor='C2', facecolor=facecolors,
                                alpha=0.7, depthshade=False)

    # 在参考图中添加 3D 集合对象
    ax_ref = fig_ref.add_subplot(projection='3d')
    ax_ref.add_collection3d(c)


def test_poly3dcollection_verts_validation():
    poly = [[0, 0, 1], [0, 1, 1], [0, 1, 0], [0, 0, 0]]
    # 使用 pytest 断言，验证是否抛出 ValueError 异常
    with pytest.raises(ValueError, match=r'list of \(N, 3\) array-like'):
        art3d.Poly3DCollection(poly)  # 应当使用 Poly3DCollection([poly])

    poly = np.array(poly, dtype=float)
    # 使用 pytest 断言，验证是否抛出 ValueError 异常
    with pytest.raises(ValueError, match=r'list of \(N, 3\) array-like'):
        art3d.Poly3DCollection(poly)  # 应当使用 Poly3DCollection([poly])


@mpl3d_image_comparison(['poly3dcollection_closed.png'], style='mpl20')
def test_poly3dcollection_closed():
    # 创建一个新的 Matplotlib 图形对象
    fig = plt.figure()
    # 添加一个带有 3D 投影的子图
    ax = fig.add_subplot(projection='3d')

    # 定义两个多边形的顶点数组
    poly1 = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]], float)
    poly2 = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], float)
    # 创建两个 Poly3DCollection 对象，一个是闭合的，一个是不闭合的
    c1 = art3d.Poly3DCollection([poly1], linewidths=3, edgecolor='k',
                                facecolor=(0.5, 0.5, 1, 0.5), closed=True)
    c2 = art3d.Poly3DCollection([poly2], linewidths=3, edgecolor='k',
                                facecolor=(1, 0.5, 0.5, 0.5), closed=False)
    # 将两个多边形集合添加到子图中
    ax.add_collection3d(c1, autolim=False)
    ax.add_collection3d(c2, autolim=False)


def test_poly_collection_2d_to_3d_empty():
    # 创建一个空的 PolyCollection 对象
    poly = PolyCollection([])
    # 将二维多边形集合转换为三维
    art3d.poly_collection_2d_to_3d(poly)
    # 断言poly对象是Poly3DCollection类型的实例
    assert isinstance(poly, art3d.Poly3DCollection)
    # 断言poly对象的路径列表为空列表
    assert poly.get_paths() == []

    # 创建一个带有3D投影的子图
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # 将poly对象添加到3D轴上
    ax.add_artist(poly)
    # 执行三维投影并返回最小z值
    minz = poly.do_3d_projection()
    # 断言最小z值是NaN
    assert np.isnan(minz)

    # 确保绘图正常工作，绘制图形到画布
    fig.canvas.draw()
@mpl3d_image_comparison(['poly3dcollection_alpha.png'], style='mpl20')
# 定义用于测试 Poly3DCollection alpha 参数的函数
def test_poly3dcollection_alpha():
    # 创建一个新的 3D 图形对象
    fig = plt.figure()
    # 添加一个 3D 子图
    ax = fig.add_subplot(projection='3d')

    # 定义两个多边形的顶点
    poly1 = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]], float)
    poly2 = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], float)

    # 创建第一个 Poly3DCollection 对象，设置线宽、边缘颜色、填充颜色和闭合属性
    c1 = art3d.Poly3DCollection([poly1], linewidths=3, edgecolor='k',
                                facecolor=(0.5, 0.5, 1), closed=True)
    c1.set_alpha(0.5)  # 设置透明度为 0.5

    # 创建第二个 Poly3DCollection 对象，设置线宽和闭合属性
    c2 = art3d.Poly3DCollection([poly2], linewidths=3, closed=False)
    c2.set_facecolor((1, 0.5, 0.5))  # 设置填充颜色
    c2.set_edgecolor('k')  # 设置边缘颜色
    c2.set_alpha(0.5)  # 设置透明度为 0.5

    # 将两个 Poly3DCollection 对象添加到 3D 子图中
    ax.add_collection3d(c1, autolim=False)
    ax.add_collection3d(c2, autolim=False)


@mpl3d_image_comparison(['add_collection3d_zs_array.png'], style='mpl20')
# 定义用于测试 add_collection3d 函数与 zs 为数组参数的函数
def test_add_collection3d_zs_array():
    # 生成 theta 和 z 的数值序列
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    # 创建点的数组，以及连接这些点形成线段的数组
    points = np.column_stack([x, y, z]).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 创建新的 3D 图形对象
    fig = plt.figure()
    # 添加一个 3D 子图
    ax = fig.add_subplot(projection='3d')

    # 创建 LineCollection 对象，使用前两个维度的数据，设置颜色映射和归一化对象
    norm = plt.Normalize(0, 2*np.pi)
    lc = LineCollection(segments[:, :, :2], cmap='twilight', norm=norm)
    lc.set_array(np.mod(theta, 2*np.pi))  # 设置颜色映射数组
    # 将 LineCollection 对象添加到 3D 子图中，并沿着 z 轴的数组位置设置
    line = ax.add_collection3d(lc, zs=segments[:, :, 2])

    assert line is not None

    plt.rcParams['axes3d.automargin'] = True  # 图像重新生成时需移除此行
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 6)
    ax.set_zlim(-2, 2)


@mpl3d_image_comparison(['add_collection3d_zs_scalar.png'], style='mpl20')
# 定义用于测试 add_collection3d 函数与 zs 为标量参数的函数
def test_add_collection3d_zs_scalar():
    # 生成 theta 的数值序列，以及 z 和 r 的数值
    theta = np.linspace(0, 2 * np.pi, 100)
    z = 1
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)

    # 创建点的数组，以及连接这些点形成线段的数组
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 创建新的 3D 图形对象
    fig = plt.figure()
    # 添加一个 3D 子图
    ax = fig.add_subplot(projection='3d')

    # 创建 LineCollection 对象，设置颜色映射和归一化对象
    norm = plt.Normalize(0, 2*np.pi)
    lc = LineCollection(segments, cmap='twilight', norm=norm)
    lc.set_array(theta)  # 设置颜色映射数组
    # 将 LineCollection 对象添加到 3D 子图中，并沿着 z 轴的标量位置设置
    line = ax.add_collection3d(lc, zs=z)

    assert line is not None

    plt.rcParams['axes3d.automargin'] = True  # 图像重新生成时需移除此行
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 6)
    ax.set_zlim(0, 2)


# 定义用于测试 Line3DCollection 自动缩放功能的函数
def test_line3dCollection_autoscaling():
    # 创建新的 3D 图形对象
    fig = plt.figure()
    # 添加一个 3D 子图
    ax = fig.add_subplot(projection='3d')

    # 定义三条 3D 线段的端点坐标
    lines = [[(0, 0, 0), (1, 4, 2)],
             [(1, 1, 3), (2, 0, 2)],
             [(1, 0, 4), (1, 4, 5)]]

    # 创建 Line3DCollection 对象
    lc = art3d.Line3DCollection(lines)
    # 将 Line3DCollection 对象添加到 3D 子图中
    ax.add_collection3d(lc)
    
    # 断言自动缩放后的坐标范围符合预期
    assert np.allclose(ax.get_xlim3d(), (-0.041666666666666664, 2.0416666666666665))
    assert np.allclose(ax.get_ylim3d(), (-0.08333333333333333, 4.083333333333333))
    assert np.allclose(ax.get_zlim3d(), (-0.10416666666666666, 5.104166666666667))
def test_poly3dCollection_autoscaling():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个带有3D投影的子图
    ax = fig.add_subplot(projection='3d')
    # 定义一个三维多边形顶点的数组
    poly = np.array([[0, 0, 0], [1, 1, 3], [1, 0, 4]])
    # 创建一个三维多边形集合对象，并添加到子图中
    col = art3d.Poly3DCollection([poly])
    ax.add_collection3d(col)
    # 断言子图的x轴、y轴、z轴的数据范围近似于指定的值
    assert np.allclose(ax.get_xlim3d(), (-0.020833333333333332, 1.0208333333333333))
    assert np.allclose(ax.get_ylim3d(), (-0.020833333333333332, 1.0208333333333333))
    assert np.allclose(ax.get_zlim3d(), (-0.0833333333333333, 4.083333333333333))


@mpl3d_image_comparison(['axes3d_labelpad.png'],
                        remove_text=False, style='mpl20')
def test_axes3d_labelpad():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个带有3D投影的坐标轴对象
    ax = fig.add_axes(Axes3D(fig))
    # 断言x轴标签的间距与rcParams中定义的一致
    assert ax.xaxis.labelpad == mpl.rcParams['axes.labelpad']
    # 设置x轴标签的文本和间距
    ax.set_xlabel('X LABEL', labelpad=10)
    assert ax.xaxis.labelpad == 10
    # 设置y轴和z轴的标签文本
    ax.set_ylabel('Y LABEL')
    ax.set_zlabel('Z LABEL', labelpad=20)
    assert ax.zaxis.labelpad == 20
    assert ax.get_zlabel() == 'Z LABEL'
    # 或者手动设置y轴和z轴的标签间距
    ax.yaxis.labelpad = 20
    ax.zaxis.labelpad = -40

    # 利用for循环调整y轴主刻度的标签间距
    for i, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.set_pad(tick.get_pad() + 5 - i * 5)


@mpl3d_image_comparison(['axes3d_cla.png'], remove_text=False, style='mpl20')
def test_axes3d_cla():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个带有3D投影的子图
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # 关闭三维子图的坐标轴显示
    ax.set_axis_off()
    # 清除当前三维子图，确保显示的是3D（而非2D）效果
    ax.cla()


@mpl3d_image_comparison(['axes3d_rotated.png'],
                        remove_text=False, style='mpl20')
def test_axes3d_rotated():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个带有3D投影的子图
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    # 设置子图的视角，使其旋转90度并倾斜45度，从上向下看应为正方形
    ax.view_init(90, 45, 0)


def test_plotsurface_1d_raises():
    # 创建一维随机数据
    x = np.linspace(0.5, 10, num=100)
    y = np.linspace(0.5, 10, num=100)
    X, Y = np.meshgrid(x, y)
    z = np.random.randn(100)

    # 创建一个新的图形对象，包含两个子图
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # 断言在绘制三维表面时引发值错误异常
    with pytest.raises(ValueError):
        ax.plot_surface(X, Y, z)


def _test_proj_make_M():
    # 定义视角点、参考点、视线方向等参数
    E = np.array([1000, -1000, 2000])
    R = np.array([100, 100, 100])
    V = np.array([0, 0, 1])
    roll = 0
    # 计算并返回视角变换矩阵M
    u, v, w = proj3d._view_axes(E, R, V, roll)
    viewM = proj3d._view_transformation_uvw(u, v, w, E)
    perspM = proj3d._persp_transformation(100, -100, 1)
    M = np.dot(perspM, viewM)
    return M


def test_proj_transform():
    # 获取视角变换矩阵M及其逆矩阵
    M = _test_proj_make_M()
    invM = np.linalg.inv(M)

    # 创建三维坐标数据
    xs = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0]) * 300.0
    ys = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0]) * 300.0
    zs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) * 300.0

    # 对坐标数据进行投影变换和逆变换，断言变换后的坐标近似等于原始坐标
    txs, tys, tzs = proj3d.proj_transform(xs, ys, zs, M)
    ixs, iys, izs = proj3d.inv_transform(txs, tys, tzs, invM)
    np.testing.assert_almost_equal(ixs, xs)
    # 使用 NumPy 测试模块中的函数 `assert_almost_equal` 检查 `iys` 是否与 `ys` 几乎相等
    np.testing.assert_almost_equal(iys, ys)
    # 使用 NumPy 测试模块中的函数 `assert_almost_equal` 检查 `izs` 是否与 `zs` 几乎相等
    np.testing.assert_almost_equal(izs, zs)
def _test_proj_draw_axes(M, s=1, *args, **kwargs):
    # 定义三个坐标轴的起点和终点
    xs = [0, s, 0, 0]
    ys = [0, 0, s, 0]
    zs = [0, 0, 0, s]
    # 将三维坐标点投影到二维平面上
    txs, tys, tzs = proj3d.proj_transform(xs, ys, zs, M)
    # 分别获取投影后的坐标点
    o, ax, ay, az = zip(txs, tys)
    # 定义三条线段
    lines = [(o, ax), (o, ay), (o, az)]

    # 创建带有投影坐标轴的新图表
    fig, ax = plt.subplots(*args, **kwargs)
    # 创建线段集合对象并添加到坐标轴上
    linec = LineCollection(lines)
    ax.add_collection(linec)
    # 在图表上标记投影点 'o', 'x', 'y', 'z'
    for x, y, t in zip(txs, tys, ['o', 'x', 'y', 'z']):
        ax.text(x, y, t)

    return fig, ax


@mpl3d_image_comparison(['proj3d_axes_cube.png'], style='mpl20')
def test_proj_axes_cube():
    # 获取变换矩阵 M
    M = _test_proj_make_M()

    # 定义立方体顶点的编号序列
    ts = '0 1 2 3 0 4 5 6 7 4'.split()
    # 定义立方体各顶点的 x, y, z 坐标并乘以缩放因子
    xs = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0]) * 300.0
    ys = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0]) * 300.0
    zs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) * 300.0

    # 将立方体顶点投影到二维平面上
    txs, tys, tzs = proj3d.proj_transform(xs, ys, zs, M)

    # 创建包含坐标轴的图表
    fig, ax = _test_proj_draw_axes(M, s=400)

    # 在图表上绘制散点图，并根据 tzs 值着色
    ax.scatter(txs, tys, c=tzs)
    # 在图表上绘制连线并设置为红色
    ax.plot(txs, tys, c='r')
    # 在图表上标记立方体顶点的编号
    for x, y, t in zip(txs, tys, ts):
        ax.text(x, y, t)

    # 设置图表参数，这里的设置是为了重新生成图像时可以删除
    plt.rcParams['axes3d.automargin'] = True  # 重新生成图像时删除
    # 设置坐标轴的 x 和 y 范围
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)


@mpl3d_image_comparison(['proj3d_axes_cube_ortho.png'], style='mpl20')
def test_proj_axes_cube_ortho():
    # 定义观察视角参数
    E = np.array([200, 100, 100])
    R = np.array([0, 0, 0])
    V = np.array([0, 0, 1])
    roll = 0
    # 根据观察视角参数生成视角变换矩阵和正交变换矩阵，然后组合得到总变换矩阵 M
    u, v, w = proj3d._view_axes(E, R, V, roll)
    viewM = proj3d._view_transformation_uvw(u, v, w, E)
    orthoM = proj3d._ortho_transformation(-1, 1)
    M = np.dot(orthoM, viewM)

    # 定义立方体顶点的编号序列
    ts = '0 1 2 3 0 4 5 6 7 4'.split()
    # 定义立方体各顶点的 x, y, z 坐标并乘以缩放因子
    xs = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 0]) * 100
    ys = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0]) * 100
    zs = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) * 100

    # 将立方体顶点投影到二维平面上
    txs, tys, tzs = proj3d.proj_transform(xs, ys, zs, M)

    # 创建包含坐标轴的图表
    fig, ax = _test_proj_draw_axes(M, s=150)

    # 在图表上绘制散点图，并根据 tzs 值修改散点大小
    ax.scatter(txs, tys, s=300 - tzs)
    # 在图表上绘制连线并设置为红色
    ax.plot(txs, tys, c='r')
    # 在图表上标记立方体顶点的编号
    for x, y, t in zip(txs, tys, ts):
        ax.text(x, y, t)

    # 设置图表参数，这里的设置是为了重新生成图像时可以删除
    plt.rcParams['axes3d.automargin'] = True  # 重新生成图像时删除
    # 设置坐标轴的 x 和 y 范围
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)


def test_world():
    # 定义世界坐标系的范围
    xmin, xmax = 100, 120
    ymin, ymax = -100, 100
    zmin, zmax = 0.1, 0.2
    # 计算世界变换矩阵 M
    M = proj3d.world_transformation(xmin, xmax, ymin, ymax, zmin, zmax)
    # 使用断言检查 M 的值是否在预期范围内
    np.testing.assert_allclose(M,
                               [[5e-2, 0, 0, -5],
                                [0, 5e-3, 0, 5e-1],
                                [0, 0, 1e1, -1],
                                [0, 0, 0, 1]])


def test_autoscale():
    # 创建带有 3D 投影的图表对象
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # 断言检查 z 轴是否为线性缩放
    assert ax.get_zscale() == 'linear'
    # 设置视图边距为 0，并调整 x, y, z 方向的边距
    ax._view_margin = 0
    ax.margins(x=0, y=.1, z=.2)
    # 在图表上绘制线段
    ax.plot([0, 1], [0, 1], [0, 1])
    # 断言检查图表的 w 范围是否为预期值
    assert ax.get_w_lims() == (0, 1, -.1, 1.1, -.2, 1.2)
    # 关闭自动缩放，并将 z 轴的自动缩放打开
    ax.autoscale(False)
    ax.set_autoscalez_on(True)
    # 在图表上再次绘制线段
    ax.plot([0, 2], [0, 2], [0, 2])
    # 断言检查图表的 w 范围是否为预期值
    assert ax.get_w_lims() == (0, 1, -.1, 1.1, -.4, 2.4)
    # 对 x 轴进行自动缩放
    # 在三维坐标轴上绘制一条从 (0,0,0) 到 (2,2,2) 的直线
    ax.plot([0, 2], [0, 2], [0, 2])
    # 使用断言检查当前坐标轴对象 ax 的三维坐标轴范围是否为 (xmin, xmax, ymin, ymax, zmin, zmax) = (0, 2, -0.1, 1.1, -0.4, 2.4)，如果不符合则会引发 AssertionError
    assert ax.get_w_lims() == (0, 2, -0.1, 1.1, -0.4, 2.4)
@pytest.mark.parametrize('axis', ('x', 'y', 'z'))
@pytest.mark.parametrize('auto', (True, False, None))
def test_unautoscale(axis, auto):
    # 创建一个新的 Matplotlib 图形对象
    fig = plt.figure()
    # 在图形对象上添加一个 3D 子图
    ax = fig.add_subplot(projection='3d')

    # 创建一些示例数据
    x = np.arange(100)
    y = np.linspace(-0.1, 0.1, 100)
    # 在 3D 子图上绘制散点图
    ax.scatter(x, y)

    # 根据 axis 参数动态获取对应的函数引用
    get_autoscale_on = getattr(ax, f'get_autoscale{axis}_on')
    set_lim = getattr(ax, f'set_{axis}lim')
    get_lim = getattr(ax, f'get_{axis}lim')

    # 根据 auto 参数设定是否自动缩放
    post_auto = get_autoscale_on() if auto is None else auto

    # 设置坐标轴限制，并校验是否设置成功
    set_lim((-0.5, 0.5), auto=auto)
    assert post_auto == get_autoscale_on()
    # 重新绘制图形
    fig.canvas.draw()
    # 使用 NumPy 测试断言确保坐标轴限制被正确设置
    np.testing.assert_array_equal(get_lim(), (-0.5, 0.5))


def test_axes3d_focal_length_checks():
    # 创建一个新的 Matplotlib 图形对象
    fig = plt.figure()
    # 在图形对象上添加一个 3D 子图
    ax = fig.add_subplot(projection='3d')
    # 测试透视投影类型下，焦距为 0 是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        ax.set_proj_type('persp', focal_length=0)
    # 测试正交投影类型下，焦距为 1 是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        ax.set_proj_type('ortho', focal_length=1)


@mpl3d_image_comparison(['axes3d_focal_length.png'],
                        remove_text=False, style='mpl20')
def test_axes3d_focal_length():
    # 创建一个新的 Matplotlib 图形对象和 3D 子图
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    # 在两个子图上分别设置透视投影类型和无限的焦距
    axs[0].set_proj_type('persp', focal_length=np.inf)
    axs[1].set_proj_type('persp', focal_length=0.15)


@mpl3d_image_comparison(['axes3d_ortho.png'], remove_text=False, style='mpl20')
def test_axes3d_ortho():
    # 创建一个新的 Matplotlib 图形对象
    fig = plt.figure()
    # 在图形对象上添加一个 3D 子图，并设置为正交投影类型
    ax = fig.add_subplot(projection='3d')
    ax.set_proj_type('ortho')


@mpl3d_image_comparison(['axes3d_isometric.png'], style='mpl20')
def test_axes3d_isometric():
    # 导入必要的模块和函数
    from itertools import combinations, product
    # 创建一个新的 Matplotlib 图形对象和 3D 子图，设置为正交投影类型和指定的盒子纵横比
    fig, ax = plt.subplots(subplot_kw=dict(
        projection='3d',
        proj_type='ortho',
        box_aspect=(4, 4, 4)
    ))
    r = (-1, 1)  # 生成三维空间内坐标的范围
    # 遍历坐标的所有组合，绘制符合条件的直线段
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if abs(s - e).sum() == r[1] - r[0]:
            ax.plot3D(*zip(s, e), c='k')
    # 设置视角初始化参数
    ax.view_init(elev=np.degrees(np.arctan(1. / np.sqrt(2))), azim=-45, roll=0)
    # 打开 3D 子图的网格显示
    ax.grid(True)


@pytest.mark.parametrize('value', [np.inf, np.nan])
@pytest.mark.parametrize(('setter', 'side'), [
    ('set_xlim3d', 'left'),
    ('set_xlim3d', 'right'),
    ('set_ylim3d', 'bottom'),
    ('set_ylim3d', 'top'),
    ('set_zlim3d', 'bottom'),
    ('set_zlim3d', 'top'),
])
def test_invalid_axes_limits(setter, side, value):
    # 创建一个新的 Matplotlib 图形对象
    fig = plt.figure()
    # 在图形对象上添加一个 3D 子图
    obj = fig.add_subplot(projection='3d')
    limit = {side: value}
    # 测试设置不合法的坐标轴限制时是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        getattr(obj, setter)(**limit)


class TestVoxels:
    @mpl3d_image_comparison(['voxels-simple.png'], style='mpl20')
    def test_simple(self):
        # 创建一个新的 Matplotlib 图形对象和 3D 子图
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # 创建三维数据的索引
        x, y, z = np.indices((5, 4, 3))
        # 创建体素数据并在 3D 子图上绘制
        voxels = (x == y) | (y == z)
        ax.voxels(voxels)

    @mpl3d_image_comparison(['voxels-edge-style.png'], style='mpl20')
    def test_edge_style(self):
        # 创建一个包含3D投影的子图
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # 创建一个5x5x4的索引数组
        x, y, z = np.indices((5, 5, 4))
        # 根据条件生成布尔数组，表示是否在球体内部
        voxels = ((x - 2)**2 + (y - 2)**2 + (z - 1.5)**2) < 2.2**2
        # 绘制体素，并设置线宽和边缘颜色
        v = ax.voxels(voxels, linewidths=3, edgecolor='C1')

        # 改变一个体素的边缘颜色
        v[max(v.keys())].set_edgecolor('C2')

    @mpl3d_image_comparison(['voxels-named-colors.png'], style='mpl20')
    def test_named_colors(self):
        """Test with colors set to a 3D object array of strings."""
        # 创建一个包含3D投影的子图
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # 创建一个10x10x10的索引数组
        x, y, z = np.indices((10, 10, 10))
        # 根据条件生成布尔数组，表示是否为对角线或者反对角线上的体素
        voxels = (x == y) | (y == z)
        voxels = voxels & ~(x * y * z < 1)
        # 创建一个包含颜色信息的对象数组
        colors = np.full((10, 10, 10), 'C0', dtype=np.object_)
        # 根据条件设置部分体素的颜色
        colors[(x < 5) & (y < 5)] = '0.25'
        colors[(x + z) < 10] = 'cyan'
        # 绘制体素，并设置面颜色
        ax.voxels(voxels, facecolors=colors)

    @mpl3d_image_comparison(['voxels-rgb-data.png'], style='mpl20')
    def test_rgb_data(self):
        """Test with colors set to a 4d float array of rgb data."""
        # 创建一个包含3D投影的子图
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # 创建一个10x10x10的索引数组
        x, y, z = np.indices((10, 10, 10))
        # 根据条件生成布尔数组，表示是否为对角线或者反对角线上的体素
        voxels = (x == y) | (y == z)
        # 创建一个包含RGB颜色信息的数组
        colors = np.zeros((10, 10, 10, 3))
        colors[..., 0] = x / 9  # 设置红色通道
        colors[..., 1] = y / 9  # 设置绿色通道
        colors[..., 2] = z / 9  # 设置蓝色通道
        # 绘制体素，并设置面颜色
        ax.voxels(voxels, facecolors=colors)

    @mpl3d_image_comparison(['voxels-alpha.png'], style='mpl20')
    def test_alpha(self):
        # 创建一个包含3D投影的子图
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # 创建一个10x10x10的索引数组
        x, y, z = np.indices((10, 10, 10))
        # 根据条件生成布尔数组，表示是否为对角线或者相邻体素
        v1 = x == y
        v2 = np.abs(x - y) < 2
        voxels = v1 | v2
        # 创建一个包含RGBA颜色信息的数组
        colors = np.zeros((10, 10, 10, 4))
        colors[v2] = [1, 0, 0, 0.5]  # 设置部分体素的颜色和透明度
        colors[v1] = [0, 1, 0, 0.5]
        # 绘制体素，并设置面颜色
        v = ax.voxels(voxels, facecolors=colors)

        # 确保返回的对象类型为字典
        assert type(v) is dict
        # 遍历字典，确保每个体素坐标都存在于布尔数组中，并且对应的多边形对象是Poly3DCollection类型
        for coord, poly in v.items():
            assert voxels[coord], "faces returned for absent voxel"
            assert isinstance(poly, art3d.Poly3DCollection)

    @mpl3d_image_comparison(['voxels-xyz.png'],
                            tol=0.01, remove_text=False, style='mpl20')
    def test_xyz(self):
        # 创建一个包含 3D 投影的图形和轴对象
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        def midpoints(x):
            sl = ()
            for i in range(x.ndim):
                # 计算每个维度上的中点坐标
                x = (x[sl + np.index_exp[:-1]] +
                     x[sl + np.index_exp[1:]]) / 2.0
                sl += np.index_exp[:]
            return x

        # 准备一些坐标，并为每个坐标附加 RGB 值
        r, g, b = np.indices((17, 17, 17)) / 16.0
        rc = midpoints(r)
        gc = midpoints(g)
        bc = midpoints(b)

        # 定义一个围绕 [0.5, 0.5, 0.5] 的球体
        sphere = (rc - 0.5)**2 + (gc - 0.5)**2 + (bc - 0.5)**2 < 0.5**2

        # 组合颜色分量
        colors = np.zeros(sphere.shape + (3,))
        colors[..., 0] = rc
        colors[..., 1] = gc
        colors[..., 2] = bc

        # 绘制所有内容
        ax.voxels(r, g, b, sphere,
                  facecolors=colors,
                  edgecolors=np.clip(2*colors - 0.5, 0, 1),  # 增亮边缘颜色
                  linewidth=0.5)

    def test_calling_conventions(self):
        # 创建 3D 投影的图形和轴对象
        x, y, z = np.indices((3, 4, 5))
        filled = np.ones((2, 3, 4))
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # 所有有效的调用约定
        for kw in (dict(), dict(edgecolor='k')):
            ax.voxels(filled, **kw)  # 使用填充数据绘制体素
            ax.voxels(filled=filled, **kw)  # 使用关键字参数填充数据绘制体素
            ax.voxels(x, y, z, filled, **kw)  # 使用位置参数和填充数据绘制体素
            ax.voxels(x, y, z, filled=filled, **kw)  # 使用位置参数和关键字参数填充数据绘制体素

        # 重复的参数
        with pytest.raises(TypeError, match='voxels'):
            ax.voxels(x, y, z, filled, filled=filled)

        # 缺少参数
        with pytest.raises(TypeError, match='voxels'):
            ax.voxels(x, y)

        # x, y, z 是只能作为位置参数 - 这将它们作为 Poly3DCollection 的属性传递
        with pytest.raises(AttributeError):
            ax.voxels(filled=filled, x=x, y=y, z=z)
def test_line3d_set_get_data_3d():
    # 定义三维数据点的坐标
    x, y, z = [0, 1], [2, 3], [4, 5]
    x2, y2, z2 = [6, 7], [8, 9], [10, 11]
    # 创建新的三维图形对象
    fig = plt.figure()
    # 添加一个三维子图
    ax = fig.add_subplot(projection='3d')
    # 在三维坐标系中绘制线条
    lines = ax.plot(x, y, z)
    # 获取绘制的线条对象
    line = lines[0]
    # 断言获取的线条数据与设置的数据一致
    np.testing.assert_array_equal((x, y, z), line.get_data_3d())
    # 设置线条新的数据点
    line.set_data_3d(x2, y2, z2)
    # 断言获取的线条数据与设置的新数据一致
    np.testing.assert_array_equal((x2, y2, z2), line.get_data_3d())
    # 设置线条的 x 轴数据
    line.set_xdata(x)
    # 设置线条的 y 轴数据
    line.set_ydata(y)
    # 设置线条的 z 轴属性，指定 zs 为 z，沿 z 轴方向
    line.set_3d_properties(zs=z, zdir='z')
    # 断言获取的线条数据与设置的数据一致
    np.testing.assert_array_equal((x, y, z), line.get_data_3d())
    # 设置线条的 z 轴属性，指定 zs 为 0，沿 z 轴方向
    line.set_3d_properties(zs=0, zdir='z')
    # 断言获取的线条数据与设置的数据一致，此时 z 轴数据为全零
    np.testing.assert_array_equal((x, y, np.zeros_like(z)), line.get_data_3d())


@check_figures_equal(extensions=["png"])
def test_inverted(fig_test, fig_ref):
    # 在测试图中添加三维子图并绘制线条，然后反转 y 轴
    ax = fig_test.add_subplot(projection="3d")
    ax.plot([1, 1, 10, 10], [1, 10, 10, 10], [1, 1, 1, 10])
    ax.invert_yaxis()
    # 在参考图中添加三维子图并反转 y 轴，然后绘制线条
    ax = fig_ref.add_subplot(projection="3d")
    ax.invert_yaxis()
    ax.plot([1, 1, 10, 10], [1, 10, 10, 10], [1, 1, 1, 10])


def test_inverted_cla():
    # GitHub PR #5450. 设置自动缩放应重置轴为非反转状态。
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # 1. 测试一个新的轴默认情况下不反转
    assert not ax.xaxis_inverted()
    assert not ax.yaxis_inverted()
    assert not ax.zaxis_inverted()
    # 设置轴的限制为反转状态
    ax.set_xlim(1, 0)
    ax.set_ylim(1, 0)
    ax.set_zlim(1, 0)
    assert ax.xaxis_inverted()
    assert ax.yaxis_inverted()
    assert ax.zaxis_inverted()
    # 清除当前轴的内容并断言轴恢复到非反转状态
    ax.cla()
    assert not ax.xaxis_inverted()
    assert not ax.yaxis_inverted()
    assert not ax.zaxis_inverted()


def test_ax3d_tickcolour():
    # 创建新的三维图形对象
    fig = plt.figure()
    # 创建一个三维坐标系
    ax = Axes3D(fig)

    # 设置 x、y、z 轴的刻度颜色为红色
    ax.tick_params(axis='x', colors='red')
    ax.tick_params(axis='y', colors='red')
    ax.tick_params(axis='z', colors='red')
    # 绘制画布以更新图形
    fig.canvas.draw()

    # 遍历 x、y、z 轴的主要刻度线，并断言它们的颜色是否为红色
    for tick in ax.xaxis.get_major_ticks():
        assert tick.tick1line._color == 'red'
    for tick in ax.yaxis.get_major_ticks():
        assert tick.tick1line._color == 'red'
    for tick in ax.zaxis.get_major_ticks():
        assert tick.tick1line._color == 'red'


@check_figures_equal(extensions=["png"])
def test_ticklabel_format(fig_test, fig_ref):
    # 创建 4x5 的子图数组，每个子图都是三维投影
    axs = fig_test.subplots(4, 5, subplot_kw={"projection": "3d"})
    for ax in axs.flat:
        # 设置 x 轴限制在 1e7 到 1e7 + 10 之间
        ax.set_xlim(1e7, 1e7 + 10)
    # 遍历每一行的子图，设置不同的刻度标签格式
    for row, name in zip(axs, ["x", "y", "z", "both"]):
        row[0].ticklabel_format(
            axis=name, style="plain")
        row[1].ticklabel_format(
            axis=name, scilimits=(-2, 2))
        row[2].ticklabel_format(
            axis=name, useOffset=not mpl.rcParams["axes.formatter.useoffset"])
        row[3].ticklabel_format(
            axis=name, useLocale=not mpl.rcParams["axes.formatter.use_locale"])
        row[4].ticklabel_format(
            axis=name,
            useMathText=not mpl.rcParams["axes.formatter.use_mathtext"])
    # 定义一个函数，用于获取给定轴对象上指定名称的主要格式化程序
    def get_formatters(ax, names):
        return [getattr(ax, name).get_major_formatter() for name in names]

    # 创建一个4行5列的子图数组，并指定每个子图的投影为3D
    axs = fig_ref.subplots(4, 5, subplot_kw={"projection": "3d"})
    # 对每个子图对象进行迭代，设置其 x 轴范围为 1e7 到 1e7 + 10
    for ax in axs.flat:
        ax.set_xlim(1e7, 1e7 + 10)
    # 对每一行及其对应的轴名称列表进行迭代
    for row, names in zip(
            axs, [["xaxis"], ["yaxis"], ["zaxis"], ["xaxis", "yaxis", "zaxis"]]
    ):
        # 对每个轴的第一个元素应用相应名称的格式化程序，设置为非科学计数法
        for fmt in get_formatters(row[0], names):
            fmt.set_scientific(False)
        # 对每个轴的第二个元素应用相应名称的格式化程序，设置幂限制为 (-2, 2)
        for fmt in get_formatters(row[1], names):
            fmt.set_powerlimits((-2, 2))
        # 对每个轴的第三个元素应用相应名称的格式化程序，根据全局配置决定是否使用偏移量
        for fmt in get_formatters(row[2], names):
            fmt.set_useOffset(not mpl.rcParams["axes.formatter.useoffset"])
        # 对每个轴的第四个元素应用相应名称的格式化程序，根据全局配置决定是否使用本地化设置
        for fmt in get_formatters(row[3], names):
            fmt.set_useLocale(not mpl.rcParams["axes.formatter.use_locale"])
        # 对每个轴的第五个元素应用相应名称的格式化程序，根据全局配置决定是否使用数学文本渲染
        for fmt in get_formatters(row[4], names):
            fmt.set_useMathText(
                not mpl.rcParams["axes.formatter.use_mathtext"])
# 使用装饰器检查生成的图像是否相等，扩展名为 "png"
@check_figures_equal(extensions=["png"])
def test_quiver3D_smoke(fig_test, fig_ref):
    # 设置箭头的旋转点为 "middle"
    pivot = "middle"
    
    # 创建网格
    x, y, z = np.meshgrid(
        np.arange(-0.8, 1, 0.2),
        np.arange(-0.8, 1, 0.2),
        np.arange(-0.8, 1, 0.8)
    )
    # 初始化箭头的方向为单位向量
    u = v = w = np.ones_like(x)

    # 对参考图像和测试图像进行迭代
    for fig, length in zip((fig_ref, fig_test), (1, 1.0)):
        # 添加一个三维子图到当前图像中
        ax = fig.add_subplot(projection="3d")
        # 绘制三维箭头
        ax.quiver(x, y, z, u, v, w, length=length, pivot=pivot)


# 使用装饰器比较生成的图像是否相同，预期输出为 "minor_ticks.png"，样式为 "mpl20"
@image_comparison(["minor_ticks.png"], style="mpl20")
def test_minor_ticks():
    # 创建一个包含三维投影的子图
    ax = plt.figure().add_subplot(projection="3d")
    # 设置 X 轴次要刻度的位置
    ax.set_xticks([0.25], minor=True)
    # 设置 X 轴次要刻度的标签
    ax.set_xticklabels(["quarter"], minor=True)
    # 设置 Y 轴次要刻度的位置
    ax.set_yticks([0.33], minor=True)
    # 设置 Y 轴次要刻度的标签
    ax.set_yticklabels(["third"], minor=True)
    # 设置 Z 轴次要刻度的位置
    ax.set_zticks([0.50], minor=True)
    # 设置 Z 轴次要刻度的标签
    ax.set_zticklabels(["half"], minor=True)


# 使用装饰器比较生成的三维图像是否符合预期，预期输出为 "errorbar3d_errorevery.png"，样式为 "mpl20"
@mpl3d_image_comparison(['errorbar3d_errorevery.png'], style='mpl20', tol=0.003)
def test_errorbar3d_errorevery():
    """Tests errorevery functionality for 3D errorbars."""
    # 创建一个新的图像对象
    fig = plt.figure()
    # 添加一个包含三维投影的子图
    ax = fig.add_subplot(projection='3d')

    # 生成测试数据
    t = np.arange(0, 2*np.pi+.1, 0.01)
    x, y, z = np.sin(t), np.cos(3*t), np.sin(5*t)

    # 设置错误条的显示频率
    estep = 15
    i = np.arange(t.size)
    zuplims = (i % estep == 0) & (i // estep % 3 == 0)
    zlolims = (i % estep == 0) & (i // estep % 3 == 2)

    # 绘制带有错误条的三维线图
    ax.errorbar(x, y, z, 0.2, zuplims=zuplims, zlolims=zlolims,
                errorevery=estep)


# 使用装饰器比较生成的三维图像是否符合预期，预期输出为 "errorbar3d.png"，样式为 "mpl20"
@mpl3d_image_comparison(['errorbar3d.png'], style='mpl20',
                        tol=0.02 if platform.machine() == 'arm64' else 0)
def test_errorbar3d():
    """Tests limits, color styling, and legend for 3D errorbars."""
    # 创建一个新的图像对象
    fig = plt.figure()
    # 添加一个包含三维投影的子图
    ax = fig.add_subplot(projection='3d')

    # 设置数据和误差条参数
    d = [1, 2, 3, 4, 5]
    e = [.5, .5, .5, .5, .5]
    ax.errorbar(x=d, y=d, z=d, xerr=e, yerr=e, zerr=e, capsize=3,
                zuplims=[False, True, False, True, True],
                zlolims=[True, False, False, True, False],
                yuplims=True,
                ecolor='purple', label='Error lines')
    # 添加图例到图像中
    ax.legend()


# 使用装饰器比较生成的图像是否相同，预期输出为 "stem3d.png"，样式为 "mpl20"
@image_comparison(['stem3d.png'], style='mpl20', tol=0.008)
def test_stem3d():
    # 设置全局参数，以便生成的图像符合预期
    plt.rcParams['axes3d.automargin'] = True  # Remove when image is regenerated
    # 创建包含多个子图的图像对象
    fig, axs = plt.subplots(2, 3, figsize=(8, 6),
                            constrained_layout=True,
                            subplot_kw={'projection': '3d'})

    # 生成数据
    theta = np.linspace(0, 2*np.pi)
    x = np.cos(theta - np.pi/2)
    y = np.sin(theta - np.pi/2)
    z = theta

    # 对每个子图进行迭代
    for ax, zdir in zip(axs[0], ['x', 'y', 'z']):
        # 绘制三维柱状图
        ax.stem(x, y, z, orientation=zdir)
        # 设置子图标题
        ax.set_title(f'orientation={zdir}')

    # 生成更多的数据
    x = np.linspace(-np.pi/2, np.pi/2, 20)
    y = np.ones_like(x)
    z = np.cos(x)
    # 对于每个子图 axs[1] 和对应的轴方向 ['x', 'y', 'z']，进行迭代处理
    for ax, zdir in zip(axs[1], ['x', 'y', 'z']):
        # 在当前轴上绘制 stem 图形，显示数据 x, y, z
        markerline, stemlines, baseline = ax.stem(
            x, y, z,
            linefmt='C4-.', markerfmt='C1D', basefmt='C2',
            orientation=zdir)
        # 设置子图标题，描述当前绘图的方向
        ax.set_title(f'orientation={zdir}')
        # 设置 markerline 的样式，使其无填充色且边缘宽度为 2
        markerline.set(markerfacecolor='none', markeredgewidth=2)
        # 设置 baseline 的线宽为 3
        baseline.set_linewidth(3)
@image_comparison(["equal_box_aspect.png"], style="mpl20")
def test_equal_box_aspect():
    from itertools import product, combinations  # 导入 itertools 模块中的 product 和 combinations 函数

    fig = plt.figure()  # 创建一个新的图形对象
    ax = fig.add_subplot(projection="3d")  # 在图形对象上添加一个3D子图

    # Make data
    u = np.linspace(0, 2 * np.pi, 100)  # 在0到2π之间生成100个均匀间隔的数值
    v = np.linspace(0, np.pi, 100)  # 在0到π之间生成100个均匀间隔的数值
    x = np.outer(np.cos(u), np.sin(v))  # 计算x坐标的网格数据
    y = np.outer(np.sin(u), np.sin(v))  # 计算y坐标的网格数据
    z = np.outer(np.ones_like(u), np.cos(v))  # 计算z坐标的网格数据

    # Plot the surface
    ax.plot_surface(x, y, z)  # 绘制三维曲面图

    # draw cube
    r = [-1, 1]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):  # 遍历顶点对的组合
        if np.sum(np.abs(s - e)) == r[1] - r[0]:  # 如果顶点对之间的距离等于边长
            ax.plot3D(*zip(s, e), color="b")  # 绘制立方体的一条边

    # Make axes limits
    xyzlim = np.column_stack(  # 生成包含三个轴限制的数组
        [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
    )
    XYZlim = [min(xyzlim[0]), max(xyzlim[1])]  # 计算三个轴限制的最小和最大值
    ax.set_xlim3d(XYZlim)  # 设置x轴的数据限制
    ax.set_ylim3d(XYZlim)  # 设置y轴的数据限制
    ax.set_zlim3d(XYZlim)  # 设置z轴的数据限制
    ax.axis('off')  # 关闭坐标轴的显示
    ax.set_box_aspect((1, 1, 1))  # 设置坐标轴的盒子方面比例为1:1:1

    with pytest.raises(ValueError, match="Argument zoom ="):  # 检查是否引发 ValueError 异常
        ax.set_box_aspect((1, 1, 1), zoom=-1)  # 尝试使用无效的缩放参数

def test_colorbar_pos():
    num_plots = 2
    fig, axs = plt.subplots(1, num_plots, figsize=(4, 5),  # 创建具有两个子图的图形对象
                            constrained_layout=True,  # 启用约束布局
                            subplot_kw={'projection': '3d'})  # 设置子图的投影类型为3D
    for ax in axs:
        p_tri = ax.plot_trisurf(np.random.randn(5), np.random.randn(5),  # 绘制三角网格的曲面
                                np.random.randn(5))

    cbar = plt.colorbar(p_tri, ax=axs, orientation='horizontal')  # 在水平方向上添加颜色条

    fig.canvas.draw()  # 绘制图形对象的画布
    # check that actually on the bottom
    assert cbar.ax.get_position().extents[1] < 0.2  # 检查颜色条是否位于底部的位置

def test_inverted_zaxis():
    fig = plt.figure()  # 创建一个新的图形对象
    ax = fig.add_subplot(projection='3d')  # 在图形对象上添加一个3D子图
    ax.set_zlim(0, 1)  # 设置z轴的数据限制为0到1
    assert not ax.zaxis_inverted()  # 断言z轴未被反转
    assert ax.get_zlim() == (0, 1)  # 断言z轴的数据限制
    assert ax.get_zbound() == (0, 1)  # 断言z轴的边界限制

    # Change bound
    ax.set_zbound((0, 2))  # 修改z轴的边界限制
    assert not ax.zaxis_inverted()  # 断言z轴未被反转
    assert ax.get_zlim() == (0, 2)  # 断言z轴的数据限制
    assert ax.get_zbound() == (0, 2)  # 断言z轴的边界限制

    # Change invert
    ax.invert_zaxis()  # 反转z轴
    assert ax.zaxis_inverted()  # 断言z轴被反转
    assert ax.get_zlim() == (2, 0)  # 断言z轴的数据限制
    assert ax.get_zbound() == (0, 2)  # 断言z轴的边界限制

    # Set upper bound
    ax.set_zbound(upper=1)  # 设置z轴的上边界
    assert ax.zaxis_inverted()  # 断言z轴被反转
    assert ax.get_zlim() == (1, 0)  # 断言z轴的数据限制
    assert ax.get_zbound() == (0, 1)  # 断言z轴的边界限制

    # Set lower bound
    ax.set_zbound(lower=2)  # 设置z轴的下边界
    assert ax.zaxis_inverted()  # 断言z轴被反转
    assert ax.get_zlim() == (2, 1)  # 断言z轴的数据限制
    assert ax.get_zbound() == (1, 2)  # 断言z轴的边界限制

def test_set_zlim():
    fig = plt.figure()  # 创建一个新的图形对象
    ax = fig.add_subplot(projection='3d')  # 在图形对象上添加一个3D子图
    assert np.allclose(ax.get_zlim(), (-1/48, 49/48))  # 断言z轴的数据限制接近预期值
    ax.set_zlim(zmax=2)  # 设置z轴的最大数据限制为2
    assert np.allclose(ax.get_zlim(), (-1/48, 2))  # 断言z轴的数据限制接近预期值
    ax.set_zlim(zmin=1)  # 设置z轴的最小数据限制为1
    assert ax.get_zlim() == (1, 2)  # 断言z轴的数据限制为1到2之间

    with pytest.raises(
            TypeError, match="Cannot pass both 'lower' and 'min'"):
        ax.set_zlim(bottom=0, zmin=1)  # 尝试同时传递'lower'和'min'参数，预期引发TypeError异常
    with pytest.raises(
            TypeError, match="Cannot pass both 'upper' and 'max'"):
        ax.set_zlim(top=0, zmax=1)  # 尝试同时传递'upper'和'max'参数，预期引发TypeError异常
@check_figures_equal(extensions=["png"])
def test_shared_view(fig_test, fig_ref):
    # 设置视角的俯仰、方位和滚动角度
    elev, azim, roll = 5, 20, 30
    # 在测试图中添加第一个子图，使用三维投影
    ax1 = fig_test.add_subplot(131, projection="3d")
    # 在测试图中添加第二个子图，使用三维投影，并共享第一个子图的视图
    ax2 = fig_test.add_subplot(132, projection="3d", shareview=ax1)
    # 在测试图中添加第三个子图，使用三维投影
    ax3 = fig_test.add_subplot(133, projection="3d")
    # 将第三个子图共享第一个子图的视图
    ax3.shareview(ax1)
    # 设置第二个子图的视角为预定义的俯仰、方位和滚动角度，并共享视图
    ax2.view_init(elev=elev, azim=azim, roll=roll, share=True)

    # 在参考图中的指定子图编号中添加三维投影，并设置相同的视角
    for subplot_num in (131, 132, 133):
        ax = fig_ref.add_subplot(subplot_num, projection="3d")
        ax.view_init(elev=elev, azim=azim, roll=roll)


def test_shared_axes_retick():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形中添加第一个子图，使用三维投影
    ax1 = fig.add_subplot(211, projection="3d")
    # 在图形中添加第二个子图，使用三维投影，并共享第一个子图的 Z 轴
    ax2 = fig.add_subplot(212, projection="3d", sharez=ax1)
    # 在第一个子图中绘制线条
    ax1.plot([0, 1], [0, 1], [0, 2])
    # 设置第一个子图的 Z 轴刻度
    ax1.set_zticks([-0.5, 0, 2, 2.5])
    # 检查设置共享轴刻度是否同步
    assert ax1.get_zlim() == (-0.5, 2.5)
    assert ax2.get_zlim() == (-0.5, 2.5)


def test_quaternion():
    # 创建标量为 1 的四元数对象，向量为 [0, 0, 0]
    q1 = Quaternion(1, [0, 0, 0])
    # 断言四元数的标量部分为 1，向量部分为 [0, 0, 0]
    assert q1.scalar == 1
    assert (q1.vector == [0, 0, 0]).all
    # 断言取负操作后四元数的标量部分为 -1，向量部分为 [0, 0, 0]
    assert (-q1).scalar == -1
    assert ((-q1).vector == [0, 0, 0]).all
    # 创建一个标量为 0，向量为 [1, 0, 0] 的四元数对象
    qi = Quaternion(0, [1, 0, 0])
    # 断言四元数的标量部分为 0，向量部分为 [1, 0, 0]
    assert qi.scalar == 0
    assert (qi.vector == [1, 0, 0]).all
    # 创建一个标量为 0，向量为 [0, 1, 0] 的四元数对象
    qj = Quaternion(0, [0, 1, 0])
    # 断言四元数的标量部分为 0，向量部分为 [0, 1, 0]
    assert qj.scalar == 0
    assert (qj.vector == [0, 1, 0]).all
    # 创建一个标量为 0，向量为 [0, 0, 1] 的四元数对象
    qk = Quaternion(0, [0, 0, 1])
    # 断言四元数的标量部分为 0，向量部分为 [0, 0, 1]
    assert qk.scalar == 0
    assert (qk.vector == [0, 0, 1]).all
    # 断言 i^2 = j^2 = k^2 = -1
    assert qi*qi == -q1
    assert qj*qj == -q1
    assert qk*qk == -q1
    # 断言单位四元数乘以任意向量得到原向量
    assert q1*qi == qi
    assert q1*qj == qj
    assert q1*qk == qk
    # 断言乘法规则：i*j=k, j*k=i, k*i=j
    assert qi*qj == qk
    assert qj*qk == qi
    assert qk*qi == qj
    assert qj*qi == -qk
    assert qk*qj == -qi
    assert qi*qk == -qj
    # 断言乘法运算 __mul__
    assert (Quaternion(2, [3, 4, 5]) * Quaternion(6, [7, 8, 9])
            == Quaternion(-86, [28, 48, 44]))
    # 断言共轭运算 conjugate()
    for q in [q1, qi, qj, qk]:
        assert q.conjugate().scalar == q.scalar
        assert (q.conjugate().vector == -q.vector).all
        assert q.conjugate().conjugate() == q
        assert ((q*q.conjugate()).vector == 0).all
    # 断言范数 norm
    q0 = Quaternion(0, [0, 0, 0])
    assert q0.norm == 0
    assert q1.norm == 1
    assert qi.norm == 1
    assert qj.norm == 1
    assert qk.norm == 1
    for q in [q0, q1, qi, qj, qk]:
        assert q.norm == (q*q.conjugate()).scalar
    # 断言归一化 normalize()
    for q in [
        Quaternion(2, [0, 0, 0]),
        Quaternion(0, [3, 0, 0]),
        Quaternion(0, [0, 4, 0]),
        Quaternion(0, [0, 0, 5]),
        Quaternion(6, [7, 8, 9])
    ]:
        assert q.normalize().norm == 1
    # 断言倒数 reciprocal()
    for q in [q1, qi, qj, qk]:
        assert q*q.reciprocal() == q1
        assert q.reciprocal()*q == q1
    # 断言旋转函数 rotate()
    assert (qi.rotate([1, 2, 3]) == np.array([1, -2, -3])).all
    # rotate_from_to():
    # 略
    # 使用三个不同的旋转向量 r1, r2 和对应的四元数 q 进行循环验证
    for r1, r2, q in [
        ([1, 0, 0], [0, 1, 0], Quaternion(np.sqrt(1/2), [0, 0, np.sqrt(1/2)])),
        ([1, 0, 0], [0, 0, 1], Quaternion(np.sqrt(1/2), [0, -np.sqrt(1/2), 0])),
        ([1, 0, 0], [1, 0, 0], Quaternion(1, [0, 0, 0]))
    ]:
        # 断言：使用 Quaternion.rotate_from_to() 方法计算 r1 到 r2 的旋转得到的四元数应与预期的 q 相等
        assert Quaternion.rotate_from_to(r1, r2) == q

    # 特殊情况：使用 r1 和 -r1 进行旋转计算
    for r1 in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]:
        r1 = np.array(r1)
        # 使用 pytest.warns() 检测 UserWarning，计算 r1 到 -r1 的旋转得到的四元数 q
        with pytest.warns(UserWarning):
            q = Quaternion.rotate_from_to(r1, -r1)
        # 断言：得到的四元数 q 的范数为 1
        assert np.isclose(q.norm, 1)
        # 断言：得到的四元数 q 的向量部分与 r1 正交
        assert np.dot(q.vector, r1) == 0

    # 使用不同的欧拉角 elev, azim, roll 和尺度 mag 进行循环验证
    for elev, azim, roll in [(0, 0, 0),
                             (90, 0, 0), (0, 90, 0), (0, 0, 90),
                             (0, 30, 30), (30, 0, 30), (30, 30, 0),
                             (47, 11, -24)]:
        for mag in [1, 2]:
            # 根据给定的欧拉角创建四元数 q
            q = Quaternion.from_cardan_angles(
                np.deg2rad(elev), np.deg2rad(azim), np.deg2rad(roll))
            # 断言：得到的四元数 q 的范数为 1
            assert np.isclose(q.norm, 1)
            # 将四元数 q 缩放到指定的尺度 mag
            q = Quaternion(mag * q.scalar, mag * q.vector)
            # 使用 as_cardan_angles() 方法将四元数 q 转换回欧拉角 e, a, r
            e, a, r = np.rad2deg(Quaternion.as_cardan_angles(q))
            # 断言：得到的欧拉角 e, a, r 与原始的 elev, azim, roll 相近
            assert np.isclose(e, elev)
            assert np.isclose(a, azim)
            assert np.isclose(r, roll)
def test_rotate():
    """Test rotating using the left mouse button."""
    # 循环测试不同的旋转角度和平移量组合
    for roll, dx, dy, new_elev, new_azim, new_roll in [
            [0, 0.5, 0, 0, -90, 0],
            [30, 0.5, 0, 30, -90, 0],
            [0, 0, 0.5, -90, 0, 0],
            [30, 0, 0.5, -60, -90, 90],
            [0, 0.5, 0.5, -45, -90, 45],
            [30, 0.5, 0.5, -15, -90, 45]]:
        # 创建一个新的 3D 图形对象
        fig = plt.figure()
        # 添加一个 3D 坐标轴
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # 设置视角的初始旋转角度
        ax.view_init(0, 0, roll)
        # 绘制图形的画布内容
        ax.figure.canvas.draw()

        # 模拟鼠标拖动来改变方向
        ax._button_press(
            mock_event(ax, button=MouseButton.LEFT, xdata=0, ydata=0))
        ax._on_move(
            mock_event(ax, button=MouseButton.LEFT,
                           xdata=dx*ax._pseudo_w, ydata=dy*ax._pseudo_h))
        ax.figure.canvas.draw()

        # 断言新的视角参数是否接近预期值
        assert np.isclose(ax.elev, new_elev)
        assert np.isclose(ax.azim, new_azim)
        assert np.isclose(ax.roll, new_roll)


def test_pan():
    """Test mouse panning using the middle mouse button."""

    def convert_lim(dmin, dmax):
        """Convert min/max limits to center and range."""
        # 计算坐标轴的中心点和范围
        center = (dmin + dmax) / 2
        range_ = dmax - dmin
        return center, range_

    # 创建一个新的 3D 图形对象，并添加散点
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(0, 0, 0)
    ax.figure.canvas.draw()

    # 获取初始的坐标轴限制，并转换为中心点和范围
    x_center0, x_range0 = convert_lim(*ax.get_xlim3d())
    y_center0, y_range0 = convert_lim(*ax.get_ylim3d())
    z_center0, z_range0 = convert_lim(*ax.get_zlim3d())

    # 模拟鼠标斜向移动来进行平移
    ax._button_press(
        mock_event(ax, button=MouseButton.MIDDLE, xdata=0, ydata=0))
    ax._on_move(
        mock_event(ax, button=MouseButton.MIDDLE, xdata=1, ydata=1))

    # 获取平移后的坐标轴中心点和范围
    x_center, x_range = convert_lim(*ax.get_xlim3d())
    y_center, y_range = convert_lim(*ax.get_ylim3d())
    z_center, z_range = convert_lim(*ax.get_zlim3d())

    # 断言范围未发生变化
    assert x_range == pytest.approx(x_range0)
    assert y_range == pytest.approx(y_range0)
    assert z_range == pytest.approx(z_range0)

    # 断言中心点已经发生变化
    assert x_center != pytest.approx(x_center0)
    assert y_center != pytest.approx(y_center0)
    assert z_center != pytest.approx(z_center0)
@pytest.mark.parametrize("tool,button,key,expected",
                         [("zoom", MouseButton.LEFT, None,  # zoom in
                          ((0.00, 0.06), (0.01, 0.07), (0.02, 0.08))),
                          ("zoom", MouseButton.LEFT, 'x',  # zoom in
                          ((-0.01, 0.10), (-0.03, 0.08), (-0.06, 0.06))),
                          ("zoom", MouseButton.LEFT, 'y',  # zoom in
                          ((-0.07, 0.05), (-0.04, 0.08), (0.00, 0.12))),
                          ("zoom", MouseButton.RIGHT, None,  # zoom out
                          ((-0.09, 0.15), (-0.08, 0.17), (-0.07, 0.18))),
                          ("pan", MouseButton.LEFT, None,  # pan
                          ((-0.70, -0.58), (-1.04, -0.91), (-1.27, -1.15))),
                          ("pan", MouseButton.LEFT, 'x',  # pan
                          ((-0.97, -0.84), (-0.58, -0.46), (-0.06, 0.06))),
                          ("pan", MouseButton.LEFT, 'y',  # pan
                          ((0.20, 0.32), (-0.51, -0.39), (-1.27, -1.15)))])
def test_toolbar_zoom_pan(tool, button, key, expected):
    # NOTE: The expected zoom values are rough ballparks of moving in the view
    #       to make sure we are getting the right direction of motion.
    #       The specific values can and should change if the zoom movement
    #       scaling factor gets updated.
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形对象上添加一个3D子图
    ax = fig.add_subplot(projection='3d')
    # 在3D子图上绘制一个散点图
    ax.scatter(0, 0, 0)
    # 绘制图形对象的画布
    fig.canvas.draw()
    # 获取当前的三维坐标轴范围
    xlim0, ylim0, zlim0 = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()

    # 鼠标从 (0, 0) 到 (1, 1)
    d0 = (0, 0)
    d1 = (1, 1)
    # 转换成屏幕坐标 ("s")。事件仅以像素精度定义，因此将像素值四舍五入，
    # 然后与对应的 xdata/ydata 进行比较，它们接近但不相等于 d0/d1。
    s0 = ax.transData.transform(d0).astype(int)
    s1 = ax.transData.transform(d1).astype(int)

    # 设置鼠标事件
    start_event = MouseEvent(
        "button_press_event", fig.canvas, *s0, button, key=key)
    stop_event = MouseEvent(
        "button_release_event", fig.canvas, *s1, button, key=key)

    # 创建一个导航工具栏对象
    tb = NavigationToolbar2(fig.canvas)
    # 根据工具类型执行缩放或平移操作
    if tool == "zoom":
        tb.zoom()
        tb.press_zoom(start_event)
        tb.drag_zoom(stop_event)
        tb.release_zoom(stop_event)
    else:
        tb.pan()
        tb.press_pan(start_event)
        tb.drag_pan(stop_event)
        tb.release_pan(stop_event)

    # 断言实际的三维坐标轴范围与预期范围非常接近，但可能不完全相等
    xlim, ylim, zlim = expected
    assert ax.get_xlim3d() == pytest.approx(xlim, abs=0.01)
    assert ax.get_ylim3d() == pytest.approx(ylim, abs=0.01)
    assert ax.get_zlim3d() == pytest.approx(zlim, abs=0.01)

    # 确保后退、前进和主页按钮正常工作
    tb.back()
    assert ax.get_xlim3d() == pytest.approx(xlim0)
    assert ax.get_ylim3d() == pytest.approx(ylim0)
    assert ax.get_zlim3d() == pytest.approx(zlim0)

    tb.forward()
    # 断言当前 3D 坐标轴的 X 轴范围与期望值在指定精度内相等
    assert ax.get_xlim3d() == pytest.approx(xlim, abs=0.01)
    # 断言当前 3D 坐标轴的 Y 轴范围与期望值在指定精度内相等
    assert ax.get_ylim3d() == pytest.approx(ylim, abs=0.01)
    # 断言当前 3D 坐标轴的 Z 轴范围与期望值在指定精度内相等
    assert ax.get_zlim3d() == pytest.approx(zlim, abs=0.01)

    # 将 3D 坐标轴复位到初始状态
    tb.home()
    # 断言复位后 3D 坐标轴的 X 轴范围与初始值相等
    assert ax.get_xlim3d() == pytest.approx(xlim0)
    # 断言复位后 3D 坐标轴的 Y 轴范围与初始值相等
    assert ax.get_ylim3d() == pytest.approx(ylim0)
    # 断言复位后 3D 坐标轴的 Z 轴范围与初始值相等
    assert ax.get_zlim3d() == pytest.approx(zlim0)
@mpl.style.context('default')
@check_figures_equal(extensions=["png"])
def test_scalarmap_update(fig_test, fig_ref):
    # 生成 x, y, z 坐标网格
    x, y, z = np.array(list(itertools.product(*[np.arange(0, 5, 1),
                                                np.arange(0, 5, 1),
                                                np.arange(0, 5, 1)]))).T
    # 根据 x 和 y 计算颜色值
    c = x + y

    # 在测试图中添加一个 3D 子图
    ax_test = fig_test.add_subplot(111, projection='3d')
    # 在测试图中绘制散点图
    sc_test = ax_test.scatter(x, y, z, c=c, s=40, cmap='viridis')
    # 强制刷新绘图
    fig_test.canvas.draw()
    # 标记散点图为“过时”
    sc_test.changed()

    # 在参考图中添加一个 3D 子图
    ax_ref = fig_ref.add_subplot(111, projection='3d')
    # 在参考图中绘制散点图
    sc_ref = ax_ref.scatter(x, y, z, c=c, s=40, cmap='viridis')


def test_subfigure_simple():
    # 测试子图能否正常工作的简单测试
    fig = plt.figure()
    # 创建一个包含 1 行 2 列子图的图形对象
    sf = fig.subfigures(1, 2)
    # 在第一个子图中添加一个 3D 子图
    ax = sf[0].add_subplot(1, 1, 1, projection='3d')
    # 在第二个子图中添加一个带有标签的 3D 子图
    ax = sf[1].add_subplot(1, 1, 1, projection='3d', label='other')


# 在重新生成测试图像时更新样式
@image_comparison(baseline_images=['computed_zorder'], remove_text=True,
                  extensions=['png'], style=('mpl20'))
def test_computed_zorder():
    # 当图像重新生成时移除这行，它设置了 3D 图的自动边距
    plt.rcParams['axes3d.automargin'] = True
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加四个子图，其中前两个是 3D 子图
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    # 禁用第二个子图的计算 Z 顺序
    ax2.computed_zorder = False

    # 创建一个水平平面
    corners = ((0, 0, 0), (0, 5, 0), (5, 5, 0), (5, 0, 0))
    for ax in (ax1, ax2):
        # 添加一个三维多边形集合（平面）
        tri = art3d.Poly3DCollection([corners],
                                     facecolors='white',
                                     edgecolors='black',
                                     zorder=1)
        ax.add_collection3d(tri)

        # 绘制一个向量
        ax.plot((2, 2), (2, 2), (0, 4), c='red', zorder=2)

        # 绘制一些点
        ax.scatter((3, 3), (1, 3), (1, 3), c='red', zorder=10)

        # 设置坐标轴的限制
        ax.set_xlim((0, 5.0))
        ax.set_ylim((0, 5.0))
        ax.set_zlim((0, 2.5))

    # 添加第三个子图和第四个子图，均为 3D 子图
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')
    # 禁用第四个子图的计算 Z 顺序
    ax4.computed_zorder = False

    # 定义坐标网格和角度
    dim = 10
    X, Y = np.meshgrid((-dim, dim), (-dim, dim))
    Z = np.zeros((2, 2))

    angle = 0.5
    X2, Y2 = np.meshgrid((-dim, dim), (0, dim))
    Z2 = Y2 * angle
    X3, Y3 = np.meshgrid((-dim, dim), (-dim, 0))
    Z3 = Y3 * angle

    # 创建圆形的坐标点
    r = 7
    M = 1000
    th = np.linspace(0, 2 * np.pi, M)
    x, y, z = r * np.cos(th),  r * np.sin(th), angle * r * np.sin(th)
    for ax in (ax3, ax4):
        # 绘制第一个三维曲面，使用蓝色，半透明度为0.5，无线条，放置于最底层
        ax.plot_surface(X2, Y3, Z3,
                        color='blue',
                        alpha=0.5,
                        linewidth=0,
                        zorder=-1)
        
        # 根据条件绘制一组线条，线宽为5，虚线样式，绿色，放置于第0层
        ax.plot(x[y < 0], y[y < 0], z[y < 0],
                lw=5,
                linestyle='--',
                color='green',
                zorder=0)

        # 绘制第二个三维曲面，使用红色，半透明度为0.5，无线条，放置于第1层
        ax.plot_surface(X, Y, Z,
                        color='red',
                        alpha=0.5,
                        linewidth=0,
                        zorder=1)

        # 绘制一个圆环，半径为r，平面角度th，放置于第2层
        ax.plot(r * np.sin(th), r * np.cos(th), np.zeros(M),
                lw=5,
                linestyle='--',
                color='black',
                zorder=2)

        # 绘制第三个三维曲面，使用蓝色，半透明度为0.5，无线条，放置于第3层
        ax.plot_surface(X2, Y2, Z2,
                        color='blue',
                        alpha=0.5,
                        linewidth=0,
                        zorder=3)

        # 根据条件绘制另一组线条，线宽为5，虚线样式，绿色，放置于第4层
        ax.plot(x[y > 0], y[y > 0], z[y > 0], lw=5,
                linestyle='--',
                color='green',
                zorder=4)
        
        # 设置视角的仰角为20度，方位角为-20度，不进行旋转，隐藏坐标轴
        ax.view_init(elev=20, azim=-20, roll=0)
        ax.axis('off')
def test_format_coord():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上添加一个3D子图
    ax = fig.add_subplot(projection='3d')
    # 创建一个包含10个元素的数组
    x = np.arange(10)
    # 在3D子图上绘制x和sin(x)的图形
    ax.plot(x, np.sin(x))
    # 设置xv和yv的值
    xv = 0.1
    yv = 0.1
    # 绘制图形并更新画布
    fig.canvas.draw()
    # 断言根据给定的xv和yv坐标，格式化后的坐标字符串
    assert ax.format_coord(xv, yv) == 'x=10.5227, y pane=1.0417, z=0.1444'

    # 修改视角参数
    ax.view_init(roll=30, vertical_axis="y")
    # 再次绘制图形并更新画布
    fig.canvas.draw()
    # 断言修改后的格式化坐标字符串
    assert ax.format_coord(xv, yv) == 'x pane=9.1875, y=0.9761, z=0.1291'

    # 重置视角参数
    ax.view_init()
    # 再次绘制图形并更新画布
    fig.canvas.draw()
    # 断言重置后的格式化坐标字符串
    assert ax.format_coord(xv, yv) == 'x=10.5227, y pane=1.0417, z=0.1444'

    # 检查正交投影
    ax.set_proj_type('ortho')
    # 再次绘制图形并更新画布
    fig.canvas.draw()
    # 断言正交投影后的格式化坐标字符串
    assert ax.format_coord(xv, yv) == 'x=10.8869, y pane=1.0417, z=0.1528'

    # 检查非默认透视投影
    ax.set_proj_type('persp', focal_length=0.1)
    # 再次绘制图形并更新画布
    fig.canvas.draw()
    # 断言非默认透视投影后的格式化坐标字符串
    assert ax.format_coord(xv, yv) == 'x=9.0620, y pane=1.0417, z=0.1110'


def test_get_axis_position():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上添加一个3D子图
    ax = fig.add_subplot(projection='3d')
    # 创建一个包含10个元素的数组
    x = np.arange(10)
    # 在3D子图上绘制x和sin(x)的图形
    ax.plot(x, np.sin(x))
    # 绘制图形并更新画布
    fig.canvas.draw()
    # 断言获取轴位置的方法返回正确的元组
    assert ax.get_axis_position() == (False, True, False)


def test_margins():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上添加一个3D子图
    ax = fig.add_subplot(projection='3d')
    # 设置边距为0.2
    ax.margins(0.2)
    # 断言获取边距的方法返回正确的元组
    assert ax.margins() == (0.2, 0.2, 0.2)
    # 修改边距参数为0.1, 0.2, 0.3
    ax.margins(0.1, 0.2, 0.3)
    # 断言获取边距的方法返回正确的元组
    assert ax.margins() == (0.1, 0.2, 0.3)
    # 修改x轴边距为0
    ax.margins(x=0)
    # 断言获取边距的方法返回正确的元组
    assert ax.margins() == (0, 0.2, 0.3)
    # 修改y轴边距为0.1
    ax.margins(y=0.1)
    # 断言获取边距的方法返回正确的元组
    assert ax.margins() == (0, 0.1, 0.3)
    # 修改z轴边距为0
    ax.margins(z=0)
    # 断言获取边距的方法返回正确的元组
    assert ax.margins() == (0, 0.1, 0)


def test_margin_getters():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上添加一个3D子图
    ax = fig.add_subplot(projection='3d')
    # 设置边距为0.1, 0.2, 0.3
    ax.margins(0.1, 0.2, 0.3)
    # 断言获取各轴边距的方法返回正确的值
    assert ax.get_xmargin() == 0.1
    assert ax.get_ymargin() == 0.2
    assert ax.get_zmargin() == 0.3


@pytest.mark.parametrize('err, args, kwargs, match', (
        # 测试边距参数值错误的情况
        (ValueError, (-1,), {}, r'margin must be greater than -0\.5'),
        (ValueError, (1, -1, 1), {}, r'margin must be greater than -0\.5'),
        (ValueError, (1, 1, -1), {}, r'margin must be greater than -0\.5'),
        (ValueError, tuple(), {'x': -1}, r'margin must be greater than -0\.5'),
        (ValueError, tuple(), {'y': -1}, r'margin must be greater than -0\.5'),
        (ValueError, tuple(), {'z': -1}, r'margin must be greater than -0\.5'),
        # 测试参数类型错误的情况
        (TypeError, (1, ), {'x': 1}, 'Cannot pass both positional and keyword'),
        (TypeError, (1, ), {'x': 1, 'y': 1, 'z': 1}, 'Cannot pass both positional and keyword'),
        (TypeError, (1, ), {'x': 1, 'y': 1}, 'Cannot pass both positional and keyword'),
        (TypeError, (1, 1), {}, 'Must pass a single positional argument for'),
))
def test_margins_errors(err, args, kwargs, match):
    # 使用pytest的断言检查异常情况
    with pytest.raises(err, match=match):
        # 创建一个新的图形对象
        fig = plt.figure()
        # 在图形上添加一个3D子图
        ax = fig.add_subplot(projection='3d')
        # 调用margins方法并传入参数
        ax.margins(*args, **kwargs)
    # 在 fig_ref 图中添加一个具有 3D 投影的子图
    ax = fig_ref.add_subplot(projection="3d")
    
    # 创建一个文本对象 txt，文本内容包含数学表达式 'Foo bar $\int$'，位置在 (0.5, 0.5)
    txt = Text(0.5, 0.5, r'Foo bar $\int$')
    
    # 将文本 txt 转换为 3D 文本，放置在 z=1 的位置
    art3d.text_2d_to_3d(txt, z=1)
    
    # 将文本对象 txt 添加到子图 ax 中
    ax.add_artist(txt)
    
    # 断言文本对象 txt 的 3D 位置为 (0.5, 0.5, 1)
    assert txt.get_position_3d() == (0.5, 0.5, 1)
    
    # 在 fig_test 图中添加一个具有 3D 投影的子图
    ax = fig_test.add_subplot(projection="3d")
    
    # 创建一个 Text3D 对象 t3d，文本内容为 'Foo bar $\int$'，位置在 (0.5, 0.5, 1)
    t3d = art3d.Text3D(0.5, 0.5, 1, r'Foo bar $\int$')
    
    # 将 Text3D 对象 t3d 添加到子图 ax 中
    ax.add_artist(t3d)
    
    # 断言 Text3D 对象 t3d 的 3D 位置为 (0.5, 0.5, 1)
    assert t3d.get_position_3d() == (0.5, 0.5, 1)
def test_draw_single_lines_from_Nx1():
    # Smoke test for GH#23459
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个带有三维投影的子图
    ax = fig.add_subplot(projection='3d')
    # 绘制三维图中的单条线段
    ax.plot([[0], [1]], [[0], [1]], [[0], [1]])


@check_figures_equal(extensions=["png"])
def test_pathpatch_3d(fig_test, fig_ref):
    # 在参考图中添加一个带有三维投影的子图
    ax = fig_ref.add_subplot(projection="3d")
    # 创建一个单位矩形的路径对象
    path = Path.unit_rectangle()
    # 创建一个路径补丁对象
    patch = PathPatch(path)
    # 将二维路径补丁对象转换为三维路径补丁对象，并添加到图中
    art3d.pathpatch_2d_to_3d(patch, z=(0, 0.5, 0.7, 1, 0), zdir='y')
    ax.add_artist(patch)

    # 在测试图中添加一个带有三维投影的子图
    ax = fig_test.add_subplot(projection="3d")
    # 创建一个三维路径补丁对象，并添加到图中
    pp3d = art3d.PathPatch3D(path, zs=(0, 0.5, 0.7, 1, 0), zdir='y')
    ax.add_artist(pp3d)


@image_comparison(baseline_images=['scatter_spiral.png'],
                  remove_text=True,
                  style='mpl20')
def test_scatter_spiral():
    # 设置绘图参数，移除自动边距设置
    plt.rcParams['axes3d.automargin'] = True  # 在重新生成图像时移除
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个带有三维投影的子图
    ax = fig.add_subplot(projection='3d')
    # 生成角度范围
    th = np.linspace(0, 2 * np.pi * 6, 256)
    # 绘制散点图
    sc = ax.scatter(np.sin(th), np.cos(th), th, s=(1 + th * 5), c=th ** 2)

    # 强制至少进行一次绘制操作
    fig.canvas.draw()


def test_Poly3DCollection_get_path():
    # Smoke test to see that get_path does not raise
    # See GH#27361
    # 创建一个包含三维投影的图形对象和一个子图对象
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # 创建一个圆形路径对象
    p = Circle((0, 0), 1.0)
    # 将二维路径对象转换为三维路径对象
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p)
    # 获取路径对象
    p.get_path()


def test_Poly3DCollection_get_facecolor():
    # Smoke test to see that get_facecolor does not raise
    # See GH#4067
    # 生成二维网格数据
    y, x = np.ogrid[1:10:100j, 1:10:100j]
    # 计算表面数据
    z2 = np.cos(x) ** 3 - np.sin(y) ** 2
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个带有三维投影的子图
    ax = fig.add_subplot(111, projection='3d')
    # 绘制三维表面图
    r = ax.plot_surface(x, y, z2, cmap='hot')
    # 获取表面图的颜色
    r.get_facecolor()


def test_Poly3DCollection_get_edgecolor():
    # Smoke test to see that get_edgecolor does not raise
    # See GH#4067
    # 生成二维网格数据
    y, x = np.ogrid[1:10:100j, 1:10:100j]
    # 计算表面数据
    z2 = np.cos(x) ** 3 - np.sin(y) ** 2
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个带有三维投影的子图
    ax = fig.add_subplot(111, projection='3d')
    # 绘制三维表面图
    r = ax.plot_surface(x, y, z2, cmap='hot')
    # 获取表面图的边缘颜色
    r.get_edgecolor()


@pytest.mark.parametrize(
    "vertical_axis, proj_expected, axis_lines_expected, tickdirs_expected",
    # 参数化测试函数，以验证不同情况下的预期行为
    [
        (
            "z",  # 字符串 "z"
            [  # 包含四个子列表的列表，每个子列表代表一个行向量
                [0.0, 1.142857, 0.0, -0.571429],  # 第一个行向量
                [0.0, 0.0, 0.857143, -0.428571],  # 第二个行向量
                [0.0, 0.0, 0.0, -10.0],  # 第三个行向量
                [-1.142857, 0.0, 0.0, 10.571429],  # 第四个行向量
            ],
            [  # 包含三个元组的列表，每个元组包含两个子列表，每个子列表包含两个浮点数
                ([0.05617978, 0.06329114], [-0.04213483, -0.04746835]),  # 第一个元组
                ([-0.06329114, 0.06329114], [-0.04746835, -0.04746835]),  # 第二个元组
                ([-0.06329114, -0.06329114], [-0.04746835, 0.04746835]),  # 第三个元组
            ],
            [1, 0, 0],  # 包含三个整数的列表
        ),
        (
            "y",  # 字符串 "y"
            [  # 包含四个子列表的列表，每个子列表代表一个行向量
                [1.142857, 0.0, 0.0, -0.571429],  # 第一个行向量
                [0.0, 0.857143, 0.0, -0.428571],  # 第二个行向量
                [0.0, 0.0, 0.0, -10.0],  # 第三个行向量
                [0.0, 0.0, -1.142857, 10.571429],  # 第四个行向量
            ],
            [  # 包含三个元组的列表，每个元组包含两个子列表，每个子列表包含两个浮点数
                ([-0.06329114, 0.06329114], [0.04746835, 0.04746835]),  # 第一个元组
                ([0.06329114, 0.06329114], [-0.04746835, 0.04746835]),  # 第二个元组
                ([-0.05617978, -0.06329114], [0.04213483, 0.04746835]),  # 第三个元组
            ],
            [2, 2, 0],  # 包含三个整数的列表
        ),
        (
            "x",  # 字符串 "x"
            [  # 包含四个子列表的列表，每个子列表代表一个行向量
                [0.0, 0.0, 1.142857, -0.571429],  # 第一个行向量
                [0.857143, 0.0, 0.0, -0.428571],  # 第二个行向量
                [0.0, 0.0, 0.0, -10.0],  # 第三个行向量
                [0.0, -1.142857, 0.0, 10.571429],  # 第四个行向量
            ],
            [  # 包含三个元组的列表，每个元组包含两个子列表，每个子列表包含两个浮点数
                ([-0.06329114, -0.06329114], [0.04746835, -0.04746835]),  # 第一个元组
                ([0.06329114, 0.05617978], [0.04746835, 0.04213483]),  # 第二个元组
                ([0.06329114, -0.06329114], [0.04746835, 0.04746835]),  # 第三个元组
            ],
            [1, 2, 1],  # 包含三个整数的列表
        ),
    ]
def test_view_init_vertical_axis(
    vertical_axis, proj_expected, axis_lines_expected, tickdirs_expected
):
    """
    Test the actual projection, axis lines and ticks matches expected values.

    Parameters
    ----------
    vertical_axis : str
        Axis to align vertically.
    proj_expected : ndarray
        Expected values from ax.get_proj().
    axis_lines_expected : tuple of arrays
        Edgepoints of the axis line. Expected values retrieved according
        to ``ax.get_[xyz]axis().line.get_data()``.
    tickdirs_expected : list of int
        indexes indicating which axis to create a tick line along.
    """
    rtol = 2e-06

    # Create a 3D subplot
    ax = plt.subplot(1, 1, 1, projection="3d")
    
    # Set the view initialization parameters
    ax.view_init(elev=0, azim=0, roll=0, vertical_axis=vertical_axis)
    
    # Force a redraw of the figure canvas
    ax.figure.canvas.draw()

    # Assert the projection matrix matches expected values
    proj_actual = ax.get_proj()
    np.testing.assert_allclose(proj_expected, proj_actual, rtol=rtol)

    # Iterate over each axis (x, y, z)
    for i, axis in enumerate([ax.get_xaxis(), ax.get_yaxis(), ax.get_zaxis()]):
        # Assert the axis lines match expected edgepoints
        axis_line_expected = axis_lines_expected[i]
        axis_line_actual = axis.line.get_data()
        np.testing.assert_allclose(axis_line_expected, axis_line_actual,
                                   rtol=rtol)

        # Assert the tick directions match expected values
        tickdir_expected = tickdirs_expected[i]
        tickdir_actual = axis._get_tickdir('default')
        np.testing.assert_array_equal(tickdir_expected, tickdir_actual)


@pytest.mark.parametrize("vertical_axis", ["x", "y", "z"])
def test_on_move_vertical_axis(vertical_axis: str) -> None:
    """
    Test vertical axis is respected when rotating the plot interactively.
    """
    # Create a 3D subplot
    ax = plt.subplot(1, 1, 1, projection="3d")
    
    # Set the view initialization parameters
    ax.view_init(elev=0, azim=0, roll=0, vertical_axis=vertical_axis)
    
    # Force a redraw of the figure canvas
    ax.figure.canvas.draw()

    # Record the projection matrix before interacting with the plot
    proj_before = ax.get_proj()

    # Simulate a button press event
    event_click = mock_event(ax, button=MouseButton.LEFT, xdata=0, ydata=1)
    ax._button_press(event_click)

    # Simulate a mouse move event
    event_move = mock_event(ax, button=MouseButton.LEFT, xdata=0.5, ydata=0.8)
    ax._on_move(event_move)

    # Assert that the vertical axis index matches the expected vertical axis
    assert ax._axis_names.index(vertical_axis) == ax._vertical_axis

    # Assert that the plot projection has changed after interaction
    proj_after = ax.get_proj()
    np.testing.assert_raises(
        AssertionError, np.testing.assert_allclose, proj_before, proj_after
    )


@pytest.mark.parametrize(
    "vertical_axis, aspect_expected",
    [
        ("x", [1.190476, 0.892857, 1.190476]),
        ("y", [0.892857, 1.190476, 1.190476]),
        ("z", [1.190476, 1.190476, 0.892857]),
    ],
)
def test_set_box_aspect_vertical_axis(vertical_axis, aspect_expected):
    # Create a 3D subplot
    ax = plt.subplot(1, 1, 1, projection="3d")
    
    # Set the view initialization parameters
    ax.view_init(elev=0, azim=0, roll=0, vertical_axis=vertical_axis)
    
    # Force a redraw of the figure canvas
    ax.figure.canvas.draw()

    # Set the box aspect ratio to None
    ax.set_box_aspect(None)

    # Assert that the box aspect ratio matches expected values
    np.testing.assert_allclose(aspect_expected, ax._box_aspect, rtol=1e-6)
# 对比图像，测试函数是否与基准图像匹配
@image_comparison(baseline_images=['arc_pathpatch.png'],
                  remove_text=True,
                  style='mpl20')
def test_arc_pathpatch():
    # 创建具有三维投影的子图
    ax = plt.subplot(1, 1, 1, projection="3d")
    # 创建一个椭圆路径补丁对象
    a = mpatch.Arc((0.5, 0.5), width=0.5, height=0.9,
                   angle=20, theta1=10, theta2=130)
    # 将路径补丁对象添加到子图中
    ax.add_patch(a)
    # 将2D路径补丁对象转换为3D，并添加到轴上
    art3d.pathpatch_2d_to_3d(a, z=0, zdir='z')


# 对比图像，测试函数是否与基准图像匹配
@image_comparison(baseline_images=['panecolor_rcparams.png'],
                  remove_text=True,
                  style='mpl20')
def test_panecolor_rcparams():
    # 在特定的rc上下文中设置3D轴的面板颜色
    with plt.rc_context({'axes3d.xaxis.panecolor': 'r',
                         'axes3d.yaxis.panecolor': 'g',
                         'axes3d.zaxis.panecolor': 'b'}):
        # 创建一个指定大小的图形对象
        fig = plt.figure(figsize=(1, 1))
        # 添加一个具有3D投影的子图
        fig.add_subplot(projection='3d')


# 检查图形是否相等的测试函数，用于比较不同的图像文件
@check_figures_equal(extensions=["png"])
def test_mutating_input_arrays_y_and_z(fig_test, fig_ref):
    """
    Test to see if the `z` axis does not get mutated
    after a call to `Axes3D.plot`

    test cases came from GH#8990
    """
    # 在测试图上添加一个具有3D投影的子图
    ax1 = fig_test.add_subplot(111, projection='3d')
    # 设置初始的x、y、z坐标值
    x = [1, 2, 3]
    y = [0.0, 0.0, 0.0]
    z = [0.0, 0.0, 0.0]
    # 绘制x、y、z坐标的散点图
    ax1.plot(x, y, z, 'o-')

    # 改变y和z的值以获得非平凡的线
    y[:] = [1, 2, 3]
    z[:] = [1, 2, 3]

    # 在参考图上添加一个具有3D投影的子图
    ax2 = fig_ref.add_subplot(111, projection='3d')
    # 设置初始的x、y、z坐标值
    x = [1, 2, 3]
    y = [0.0, 0.0, 0.0]
    z = [0.0, 0.0, 0.0]
    # 绘制x、y、z坐标的散点图
    ax2.plot(x, y, z, 'o-')


def test_scatter_masked_color():
    """
    Test color parameter usage with non-finite coordinate arrays.

    GH#26236
    """
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个具有3D投影的子图
    ax = fig.add_subplot(projection='3d')
    # 使用包含非有限坐标数组的x、y、z和颜色来绘制散点图
    path3d = ax.scatter(x, y, z, color=colors)

    # 断言偏移量的长度与面颜色的长度相等
    assert len(path3d.get_offsets()) ==\
           len(super(type(path3d), path3d).get_facecolors())


# 对比图像，测试函数是否与基准图像匹配
@mpl3d_image_comparison(['surface3d_zsort_inf.png'], style='mpl20')
def test_surface3d_zsort_inf():
    # 设置3D绘图的自动边距参数以True
    plt.rcParams['axes3d.automargin'] = True  # 在重新生成图像后移除此行
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个具有3D投影的子图
    ax = fig.add_subplot(projection='3d')

    # 创建一个网格的x、y坐标
    x, y = np.mgrid[-2:2:0.1, -2:2:0.1]
    # 创建z坐标，一半设置为无穷大
    z = np.sin(x)**2 + np.cos(y)**2
    z[x.shape[0] // 2:, x.shape[1] // 2:] = np.inf

    # 绘制3D表面图，并使用jet颜色映射
    ax.plot_surface(x, y, z, cmap='jet')
    # 设置视角
    ax.view_init(elev=45, azim=145)


def test_Poly3DCollection_init_value_error():
    # smoke test，用于确保输入检查有效
    # GH#26420
    # 使用pytest来检测是否引发了ValueError异常，并匹配特定的错误消息
    with pytest.raises(ValueError,
                       match='You must provide facecolors, edgecolors, '
                        'or both for shade to work.'):
        # 创建一个三维多边形集合对象，传入shade=True参数
        poly = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0]], float)
        c = art3d.Poly3DCollection([poly], shade=True)


def test_ndarray_color_kwargs_value_error():
    # smoke test
    # 创建一个新的图形窗口
    fig = plt.figure()
    # 在图形窗口中添加一个3D子图
    ax = fig.add_subplot(111, projection='3d')
    # 在3D子图中绘制一个散点图，使用颜色参数，确保可以接受ndarray类型的颜色
    ax.scatter(1, 0, 0, color=np.array([0, 0, 0, 1]))
    # 刷新图形窗口的画布，以便更新显示
    fig.canvas.draw()
```