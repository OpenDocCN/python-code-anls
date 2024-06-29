# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_lines.py`

```py
"""
Tests specific to the lines module.
"""

import itertools  # 导入 itertools 模块，用于创建迭代器的函数
import platform  # 导入 platform 模块，用于访问平台相关的系统信息
import timeit  # 导入 timeit 模块，用于测量小段 Python 代码的执行时间
from types import SimpleNamespace  # 导入 SimpleNamespace 类，用于创建命名空间对象

from cycler import cycler  # 导入 cycler 模块，用于定义循环器对象
import numpy as np  # 导入 NumPy 库，并使用 np 别名
from numpy.testing import assert_array_equal  # 导入 NumPy 测试模块中的数组相等断言函数
import pytest  # 导入 pytest 测试框架

import matplotlib  # 导入 matplotlib 库
import matplotlib as mpl  # 使用 mpl 别名导入 matplotlib
from matplotlib import _path  # 导入 matplotlib 中的 _path 模块
import matplotlib.lines as mlines  # 导入 matplotlib 中的 lines 模块
from matplotlib.markers import MarkerStyle  # 导入 matplotlib 中的 MarkerStyle 类
from matplotlib.path import Path  # 导入 matplotlib 中的 Path 类
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，并使用 plt 别名
import matplotlib.transforms as mtransforms  # 导入 matplotlib 中的 transforms 模块
from matplotlib.testing.decorators import image_comparison, check_figures_equal  # 导入测试装饰器


def test_segment_hits():
    """Test a problematic case."""
    cx, cy = 553, 902  # 定义 cx 和 cy 变量
    x, y = np.array([553., 553.]), np.array([95., 947.])  # 定义 x 和 y 变量
    radius = 6.94  # 定义 radius 变量
    assert_array_equal(mlines.segment_hits(cx, cy, x, y, radius), [0])  # 断言 segment_hits 函数的返回值为 [0]


# Runtimes on a loaded system are inherently flaky. Not so much that a rerun
# won't help, hopefully.
@pytest.mark.flaky(reruns=3)  # 标记该测试函数为 flaky，允许重试最多 3 次
def test_invisible_Line_rendering():
    """
    GitHub issue #1256 identified a bug in Line.draw method

    Despite visibility attribute set to False, the draw method was not
    returning early enough and some pre-rendering code was executed
    though not necessary.

    Consequence was an excessive draw time for invisible Line instances
    holding a large number of points (Npts> 10**6)
    """
    # 创建大量的 x 和 y 数据:
    N = 10**7
    x = np.linspace(0, 1, N)  # 生成从 0 到 1 的 N 个等间隔的数值
    y = np.random.normal(size=N)  # 生成 N 个符合正态分布的随机数

    # 创建一个绘图图形:
    fig = plt.figure()
    ax = plt.subplot()

    # 创建一个“大型”Line2D实例:
    l = mlines.Line2D(x, y)
    l.set_visible(False)  # 将实例的可见性设置为 False
    # 但不将其添加到 Axis 实例 `ax` 中

    # [这里交互式的平移和缩放响应非常迅速]
    # 计时绘制画布:
    t_no_line = min(timeit.repeat(fig.canvas.draw, number=1, repeat=3))
    # (大约 25 毫秒)

    # 添加大型不可见的 Line:
    ax.add_line(l)

    # [现在交互式的平移和缩放非常缓慢]
    # 计时绘制画布:
    t_invisible_line = min(timeit.repeat(fig.canvas.draw, number=1, repeat=3))
    # 对于 N = 10**7 点，大约为 290 毫秒

    slowdown_factor = t_invisible_line / t_no_line
    slowdown_threshold = 2  # 尝试避免误报失败
    assert slowdown_factor < slowdown_threshold


def test_set_line_coll_dash():
    fig, ax = plt.subplots()  # 创建一个包含单个子图的图形和坐标轴对象
    np.random.seed(0)
    # 测试为线集合设置线型。
    # 这不应该产生错误。
    ax.contour(np.random.randn(20, 30), linestyles=[(0, (3, 3))])


def test_invalid_line_data():
    with pytest.raises(RuntimeError, match='xdata must be'):
        mlines.Line2D(0, [])  # 测试创建 Line2D 实例时的异常情况
    with pytest.raises(RuntimeError, match='ydata must be'):
        mlines.Line2D([], 1)  # 测试创建 Line2D 实例时的异常情况

    line = mlines.Line2D([], [])
    with pytest.raises(RuntimeError, match='x must be'):
        line.set_xdata(0)  # 测试设置 Line2D 实例 x 数据时的异常情况
    with pytest.raises(RuntimeError, match='y must be'):
        line.set_ydata(0)  # 测试设置 Line2D 实例 y 数据时的异常情况
@image_comparison(['line_dashes'], remove_text=True, tol=0.003)
# 定义测试函数，用于比较绘制的图像是否与参考图像相同，测试线型的不同表示方式
def test_line_dashes():
    # 设置容差，用于浮点运算重排序后的误差允许范围
    # 当重新生成图像时应移除这个设置
    fig, ax = plt.subplots()

    # 绘制带有自定义虚线样式的线条，线宽为5
    ax.plot(range(10), linestyle=(0, (3, 3)), lw=5)


def test_line_colors():
    fig, ax = plt.subplots()
    # 绘制无颜色的线条
    ax.plot(range(10), color='none')
    # 绘制红色的线条
    ax.plot(range(10), color='r')
    # 绘制灰度为0.3的线条
    ax.plot(range(10), color='.3')
    # 绘制RGBA颜色为红色的线条
    ax.plot(range(10), color=(1, 0, 0, 1))
    # 绘制RGB颜色为红色的线条
    ax.plot(range(10), color=(1, 0, 0))
    fig.canvas.draw()


def test_valid_colors():
    line = mlines.Line2D([], [])
    # 测试设置非法颜色时是否抛出值错误异常
    with pytest.raises(ValueError):
        line.set_color("foobar")


def test_linestyle_variants():
    fig, ax = plt.subplots()
    # 遍历不同的线型变体
    for ls in ["-", "solid", "--", "dashed",
               "-.", "dashdot", ":", "dotted",
               (0, None), (0, ()), (0, []),  # gh-22930
               ]:
        # 绘制指定线型的线条
        ax.plot(range(10), linestyle=ls)
    fig.canvas.draw()


def test_valid_linestyles():
    line = mlines.Line2D([], [])
    # 测试设置非法线型时是否抛出值错误异常
    with pytest.raises(ValueError):
        line.set_linestyle('aardvark')


@image_comparison(['drawstyle_variants.png'], remove_text=True,
                  tol=0.03 if platform.machine() == 'arm64' else 0)
# 比较不同绘制样式的图像，确保处理良好，即使是非常长的线条
def test_drawstyle_variants():
    fig, axs = plt.subplots(6)
    dss = ["default", "steps-mid", "steps-pre", "steps-post", "steps", None]
    # 检查不同绘制样式的处理是否正确，即使是非常长的线条也能正确显示
    for ax, ds in zip(axs.flat, dss):
        ax.plot(range(2000), drawstyle=ds)
        ax.set(xlim=(0, 2), ylim=(0, 2))


@check_figures_equal(extensions=('png',))
# 测试不使用子切片与变换的情况下是否能正确绘制图像
def test_no_subslice_with_transform(fig_ref, fig_test):
    ax = fig_ref.add_subplot()
    x = np.arange(2000)
    ax.plot(x + 2000, x)

    ax = fig_test.add_subplot()
    t = mtransforms.Affine2D().translate(2000.0, 0.0)
    ax.plot(x, x, transform=t+ax.transData)


def test_valid_drawstyles():
    line = mlines.Line2D([], [])
    # 测试设置非法绘制样式时是否抛出值错误异常
    with pytest.raises(ValueError):
        line.set_drawstyle('foobar')


def test_set_drawstyle():
    x = np.linspace(0, 2*np.pi, 10)
    y = np.sin(x)

    fig, ax = plt.subplots()
    line, = ax.plot(x, y)
    # 设置绘制样式为steps-pre
    line.set_drawstyle("steps-pre")
    # 断言线条路径顶点数量是否符合预期
    assert len(line.get_path().vertices) == 2*len(x)-1

    # 恢复默认绘制样式
    line.set_drawstyle("default")
    # 断言线条路径顶点数量是否符合预期
    assert len(line.get_path().vertices) == len(x)


@image_comparison(
    ['line_collection_dashes'], remove_text=True, style='mpl20',
    tol=0 if platform.machine() == 'x86_64' else 0.65)
# 比较设置线集合的虚线样式的图像
def test_set_line_coll_dash_image():
    fig, ax = plt.subplots()
    np.random.seed(0)
    # 绘制随机数据的等高线图，并设置线型为(0, (3, 3))
    ax.contour(np.random.randn(20, 30), linestyles=[(0, (3, 3))])


@image_comparison(['marker_fill_styles.png'], remove_text=True)
# 测试标记填充样式的图像比较
def test_marker_fill_styles():
    # 使用 itertools.cycle 创建一个无限循环的迭代器，包含多种颜色表示方式和一个 NumPy 数组
    colors = itertools.cycle([[0, 0, 1], 'g', '#ff0000', 'c', 'm', 'y',
                              np.array([0, 0, 0])])
    # 指定备选的另一种颜色
    altcolor = 'lightgreen'

    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()

    # 下面的注释解释了为什么使用了一个硬编码的标记列表
    # 这些标记对应于 MarkerStyle.filled_markers 的早期版本；
    # 虽然该属性的值已更改，但我们保留了旧值以避免重新生成基准图像。
    # 当重新生成图像时，应替换为 mlines.Line2D.filled_markers。
    for j, marker in enumerate("ov^<>8sp*hHDdPX"):
        for i, fs in enumerate(mlines.Line2D.fillStyles):
            # 获取下一个颜色
            color = next(colors)
            # 在坐标轴上绘制点
            ax.plot(j * 10 + x, y + i + .5 * (j % 2),
                    marker=marker,  # 设置标记样式
                    markersize=20,  # 设置标记大小
                    markerfacecoloralt=altcolor,  # 设置备选标记颜色
                    fillstyle=fs,  # 设置填充样式
                    label=fs,  # 设置标签
                    linewidth=5,  # 设置线宽
                    color=color,  # 设置线条颜色
                    markeredgecolor=color,  # 设置标记边缘颜色
                    markeredgewidth=2)  # 设置标记边缘宽度

    # 设置 y 轴的范围
    ax.set_ylim([0, 7.5])
    # 设置 x 轴的范围
    ax.set_xlim([-5, 155])
def test_markerfacecolor_fillstyle():
    """Test that markerfacecolor does not override fillstyle='none'."""
    # 使用 plt.plot 绘制一条曲线，指定 marker 为 'o'，fillstyle 为 'none'，markerfacecolor 为 'red'
    l, = plt.plot([1, 3, 2], marker=MarkerStyle('o', fillstyle='none'),
                  markerfacecolor='red')
    # 断言曲线的 fillstyle 是否为 'none'
    assert l.get_fillstyle() == 'none'
    # 断言曲线的 markerfacecolor 是否为 'none'
    assert l.get_markerfacecolor() == 'none'


@image_comparison(['scaled_lines'], style='default')
def test_lw_scaling():
    # 生成一个包含多条曲线的图像，每条曲线具有不同的 linestyle 和 lw（线宽）
    th = np.linspace(0, 32)
    fig, ax = plt.subplots()
    lins_styles = ['dashed', 'dotted', 'dashdot']
    cy = cycler(matplotlib.rcParams['axes.prop_cycle'])
    for j, (ls, sty) in enumerate(zip(lins_styles, cy)):
        for lw in np.linspace(.5, 10, 10):
            # 在坐标系上绘制曲线，设置 linestyle、lw 和 sty（样式属性）
            ax.plot(th, j*np.ones(50) + .1 * lw, linestyle=ls, lw=lw, **sty)


def test_is_sorted_and_has_non_nan():
    # 断言给定的数组是否已排序且不含 NaN 值
    assert _path.is_sorted_and_has_non_nan(np.array([1, 2, 3]))
    assert _path.is_sorted_and_has_non_nan(np.array([1, np.nan, 3]))
    assert not _path.is_sorted_and_has_non_nan([3, 5] + [np.nan] * 100 + [0, 2])
    # 创建一条包含 NaN 的曲线
    plt.plot([np.nan] * n, range(n))


@check_figures_equal()
def test_step_markers(fig_test, fig_ref):
    # 在测试图和参考图上绘制包含步进标记的曲线
    fig_test.subplots().step([0, 1], "-o")
    fig_ref.subplots().plot([0, 0, 1], [0, 1, 1], "-o", markevery=[0, 2])


@pytest.mark.parametrize("parent", ["figure", "axes"])
@check_figures_equal(extensions=('png',))
def test_markevery(fig_test, fig_ref, parent):
    np.random.seed(42)
    x = np.linspace(0, 1, 14)
    y = np.random.rand(len(x))

    cases_test = [None, 4, (2, 5), [1, 5, 11],
                  [0, -1], slice(5, 10, 2),
                  np.arange(len(x))[y > 0.5],
                  0.3, (0.3, 0.4)]
    cases_ref = ["11111111111111", "10001000100010", "00100001000010",
                 "01000100000100", "10000000000001", "00000101010000",
                 "01110001110110", "11011011011110", "01010011011101"]

    if parent == "figure":
        # 不支持浮点数形式的 markevery（相对于轴大小）
        cases_test = cases_test[:-2]
        cases_ref = cases_ref[:-2]

        def add_test(x, y, *, markevery):
            # 在测试图中添加一条带有指定 markevery 的曲线
            fig_test.add_artist(
                mlines.Line2D(x, y, marker="o", markevery=markevery))

        def add_ref(x, y, *, markevery):
            # 在参考图中添加一条带有指定 markevery 的曲线
            fig_ref.add_artist(
                mlines.Line2D(x, y, marker="o", markevery=markevery))

    elif parent == "axes":
        axs_test = iter(fig_test.subplots(3, 3).flat)
        axs_ref = iter(fig_ref.subplots(3, 3).flat)

        def add_test(x, y, *, markevery):
            # 在测试图的各个子图中添加带有指定 markevery 的曲线
            next(axs_test).plot(x, y, "-gD", markevery=markevery)

        def add_ref(x, y, *, markevery):
            # 在参考图的各个子图中添加带有指定 markevery 的曲线
            next(axs_ref).plot(x, y, "-gD", markevery=markevery)

    for case in cases_test:
        add_test(x, y, markevery=case)

    for case in cases_ref:
        me = np.array(list(case)).astype(int).astype(bool)
        add_ref(x, y, markevery=me)
# 定义一个测试函数，用于测试在不支持的相对大小情况下的 markevery 参数设置
def test_markevery_figure_line_unsupported_relsize():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 向图形对象添加一个折线对象，设置其标记为圆形，并指定每隔一半显示一个标记点
    fig.add_artist(mlines.Line2D([0, 1], [0, 1], marker="o", markevery=.5))
    # 使用 pytest 断言检查是否引发 ValueError 异常
    with pytest.raises(ValueError):
        # 绘制图形对象的画布内容
        fig.canvas.draw()


# 定义一个测试函数，测试将 MarkerStyle 对象作为标记参数设置时的行为
def test_marker_as_markerstyle():
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制一个折线，并设置其标记为 Diamond (D) 的 MarkerStyle 对象
    line, = ax.plot([2, 4, 3], marker=MarkerStyle("D"))
    # 绘制图形对象的画布内容
    fig.canvas.draw()
    # 使用断言验证折线对象的当前标记是否为 "D"
    assert line.get_marker() == "D"

    # 继续进行附加的功能测试：
    # 修改折线的标记为 "s" 并重新绘制
    line.set_marker("s")
    fig.canvas.draw()
    # 将标记设置为 Triangle (o) 的 MarkerStyle 对象，并重新绘制
    line.set_marker(MarkerStyle("o"))
    fig.canvas.draw()
    
    # 测试 Path 对象的往返路径
    # 创建一个三角形的 Path 对象
    triangle1 = Path._create_closed([[-1, -1], [1, -1], [0, 2]])
    # 绘制一个折线，使用 Triangle (o) 的 MarkerStyle 对象，并设置标记大小为 22
    line2, = ax.plot([1, 3, 2], marker=MarkerStyle(triangle1), ms=22)
    # 绘制一个折线，直接使用三角形的 Path 对象作为标记，并设置标记大小为 22
    line3, = ax.plot([0, 2, 1], marker=triangle1, ms=22)

    # 使用断言验证折线对象的当前标记顶点是否与三角形的 Path 对象的顶点一致
    assert_array_equal(line2.get_marker().vertices, triangle1.vertices)
    assert_array_equal(line3.get_marker().vertices, triangle1.vertices)


# 使用 image_comparison 装饰器定义一个测试函数，验证带有条纹线条的绘图行为
@image_comparison(['striped_line.png'], remove_text=True, style='mpl20')
def test_striped_lines():
    # 使用随机数生成器创建一个默认种子为 19680801 的随机数对象
    rng = np.random.default_rng(19680801)
    # 创建一个新的图形对象和坐标轴对象
    _, ax = plt.subplots()
    # 绘制一条线条，线条颜色为橙色，间隔颜色为蓝色，线型为虚线，线宽为 5，无标签
    ax.plot(rng.uniform(size=12), color='orange', gapcolor='blue',
            linestyle='--', lw=5, label=' ')
    # 绘制一条线条，线条颜色为红色，间隔颜色为黑色，使用自定义虚线风格，线宽为 5，无标签，透明度为 0.5
    ax.plot(rng.uniform(size=12), color='red', gapcolor='black',
            linestyle=(0, (2, 5, 4, 2)), lw=5, label=' ', alpha=0.5)
    # 添加图例，设置图例句柄长度为 5
    ax.legend(handlelength=5)


# 使用 check_figures_equal 装饰器定义一个测试函数，测试在不同的线型设置下的图形绘制
@check_figures_equal()
def test_odd_dashes(fig_test, fig_ref):
    # 向 fig_test 图形对象添加一个子图，并绘制一条线，设置其虚线为 [1, 2, 3]
    fig_test.add_subplot().plot([1, 2], dashes=[1, 2, 3])
    # 向 fig_ref 图形对象添加一个子图，并绘制一条线，设置其虚线为 [1, 2, 3, 1, 2, 3]
    fig_ref.add_subplot().plot([1, 2], dashes=[1, 2, 3, 1, 2, 3])


# 定义一个测试函数，测试在图形中选择元素的行为
def test_picking():
    # 创建一个新的图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 创建一个 SimpleNamespace 对象，模拟鼠标事件，位于图形框的中心位置
    mouse_event = SimpleNamespace(x=fig.bbox.width // 2,
                                  y=fig.bbox.height // 2 + 15)

    # 创建一条线，并设置其可选（picker=True）
    l0, = ax.plot([0, 1], [0, 1], picker=True)
    # 检查是否找到了选定的元素
    found, indices = l0.contains(mouse_event)
    assert not found  # 预期不应找到

    # 创建另一条线，并设置其可选（picker=True），同时指定更大的选取半径（pickradius=20）
    l1, = ax.plot([0, 1], [0, 1], picker=True, pickradius=20)
    # 再次检查是否找到了选定的元素
    found, indices = l1.contains(mouse_event)
    assert found  # 预期应该找到
    assert_array_equal(indices['ind'], [0])  # 验证选中的索引为 [0]

    # 对第三条线进行同样的测试，首先不应找到选定的元素
    l2, = ax.plot([0, 1], [0, 1], picker=True)
    found, indices = l2.contains(mouse_event)
    assert not found  # 预期不应找到
    # 修改线条的选取半径为 20
    l2.set_pickradius(20)
    # 再次检查是否找到了选定的元素
    found, indices = l2.contains(mouse_event)
    assert found  # 预期应该找到
    assert_array_equal(indices['ind'], [0])  # 验证选中的索引为 [0]


# 使用 check_figures_equal 装饰器定义一个测试函数，测试图形输入复制的行为
@check_figures_equal()
def test_input_copy(fig_test, fig_ref):
    # 创建一个从 0 到 6 步长为 2 的数组
    t = np.arange(0, 6, 2)
    # 向 fig_test 图形对象添加一个子图，并绘制一条线，标记为点和线
    l, = fig_test.add_subplot().plot(t, t, ".-")
    # 修改数组 t 的值，触发缓存失效
    t[:] = range(3)
    # 将线条的绘制风格设置为 "steps"，触发缓存失效
    l.set_drawstyle("steps")
    # 向 fig_ref 图形对象添加一个子图，并绘制一条线，标记为点和线，绘制风格为 "steps"
    fig_ref.add_subplot().plot([0, 2, 4], [0, 2, 4], ".-", drawstyle="steps")


# 使用 check_figures_equal 装饰器定义一个测试函数，测试设置 markevery 的 prop_cycle 行为
@check_figures_equal(extensions=["png"])
def test_markevery_prop_cycle(fig_test, fig_ref):
    """Test that we can set markevery prop_cycle."""
    # 定义不同的测试用例，包括 None、整数、元组、列表、切片对象和浮点数等
    cases = [None, 8, (30, 8), [16, 24, 30], [0, -1],
             slice(100, 200, 3), 0.1, 0.3, 1.5,
             (0.0, 0.1), (0.45, 0.1)]

    # 从 matplotlib 的 colormaps 中选择 'jet' 色彩映射
    cmap = mpl.colormaps['jet']
    # 生成与 cases 中测试用例数量相等的颜色列表，用于不同的数据点
    colors = cmap(np.linspace(0.2, 0.8, len(cases)))

    # 在 x 轴上生成从 -1 到 1 的均匀分布的数据点
    x = np.linspace(-1, 1)
    # 计算 y 值作为 x 的平方乘以 5
    y = 5 * x**2

    # 将一个子图添加到 fig_ref 图形中
    axs = fig_ref.add_subplot()

    # 遍历 cases 列表中的每个测试用例，绘制对应的数据点图形
    for i, markevery in enumerate(cases):
        axs.plot(y - i, 'o-', markevery=markevery, color=colors[i])

    # 设置 matplotlib 默认属性循环，分别指定 markevery 和 color 参数为 cases 和 colors
    matplotlib.rcParams['axes.prop_cycle'] = cycler(markevery=cases,
                                                    color=colors)

    # 将一个子图添加到 fig_test 图形中
    ax = fig_test.add_subplot()

    # 遍历 cases 列表中的每个测试用例，绘制对应的数据点图形（此处未使用 markevery 参数）
    for i, _ in enumerate(cases):
        ax.plot(y - i, 'o-')
def test_axline_setters():
    # 创建一个新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上创建一条线，指定起点和斜率
    line1 = ax.axline((.1, .1), slope=0.6)
    # 在坐标轴上创建另一条线，指定起点和终点
    line2 = ax.axline((.1, .1), (.8, .4))
    
    # 测试设置 xy1、xy2 和斜率的方法
    # 这些操作不应该产生错误
    line1.set_xy1(.2, .3)
    line1.set_slope(2.4)
    line2.set_xy1(.3, .2)
    line2.set_xy2(.6, .8)
    
    # 测试获取 xy1、xy2 和斜率的方法
    # 应该返回修改后的值
    assert line1.get_xy1() == (.2, .3)
    assert line1.get_slope() == 2.4
    assert line2.get_xy1() == (.3, .2)
    assert line2.get_xy2() == (.6, .8)
    
    # 测试同时设置 xy2 和斜率的情况
    # 这些测试应该引发 ValueError 异常
    with pytest.raises(ValueError,
                       match="Cannot set an 'xy2' value while 'slope' is set"):
        line1.set_xy2(.2, .3)

    with pytest.raises(ValueError,
                       match="Cannot set a 'slope' value while 'xy2' is set"):
        line2.set_slope(3)
```