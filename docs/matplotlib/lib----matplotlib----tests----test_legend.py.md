# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_legend.py`

```py
import collections
import itertools
import platform
import time
from unittest import mock
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pytest

from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
import matplotlib.legend as mlegend
from matplotlib import _api, rc_context
from matplotlib.font_manager import FontProperties


def test_legend_ordereddict():
    # smoketest that ordereddict inputs work...
    
    X = np.random.randn(10)  # 生成一个包含10个随机数的numpy数组X
    Y = np.random.randn(10)  # 生成一个包含10个随机数的numpy数组Y
    labels = ['a'] * 5 + ['b'] * 5  # 创建一个包含10个元素的标签列表，前5个为'a'，后5个为'b'
    colors = ['r'] * 5 + ['g'] * 5  # 创建一个包含10个元素的颜色列表，前5个为'red'，后5个为'green'

    fig, ax = plt.subplots()  # 创建一个新的图形和轴对象
    for x, y, label, color in zip(X, Y, labels, colors):
        ax.scatter(x, y, label=label, c=color)  # 在轴上绘制散点图，每个散点有指定的标签和颜色

    handles, labels = ax.get_legend_handles_labels()  # 获取图例中的句柄和标签
    legend = collections.OrderedDict(zip(labels, handles))  # 创建有序字典，将标签和句柄配对
    ax.legend(legend.values(), legend.keys(),  # 添加图例到轴上，传入句柄和标签
              loc='center left', bbox_to_anchor=(1, .5))  # 设置图例的位置为左中，并指定边界框的位置


@image_comparison(['legend_auto1'], remove_text=True)
def test_legend_auto1():
    """Test automatic legend placement"""
    fig, ax = plt.subplots()  # 创建一个新的图形和轴对象
    x = np.arange(100)  # 创建一个包含0到99的numpy数组x
    ax.plot(x, 50 - x, 'o', label='y=1')  # 在轴上绘制线图，并指定标签
    ax.plot(x, x - 50, 'o', label='y=-1')  # 在轴上绘制线图，并指定标签
    ax.legend(loc='best')  # 自动放置图例


@image_comparison(['legend_auto2'], remove_text=True)
def test_legend_auto2():
    """Test automatic legend placement"""
    fig, ax = plt.subplots()  # 创建一个新的图形和轴对象
    x = np.arange(100)  # 创建一个包含0到99的numpy数组x
    b1 = ax.bar(x, x, align='edge', color='m')  # 在轴上绘制柱状图，指定对齐方式和颜色
    b2 = ax.bar(x, x[::-1], align='edge', color='g')  # 在轴上绘制柱状图，指定对齐方式和颜色
    ax.legend([b1[0], b2[0]], ['up', 'down'], loc='best')  # 自动放置图例，并传入句柄和标签


@image_comparison(['legend_auto3'])
def test_legend_auto3():
    """Test automatic legend placement"""
    fig, ax = plt.subplots()  # 创建一个新的图形和轴对象
    x = [0.9, 0.1, 0.1, 0.9, 0.9, 0.5]  # 创建一个包含6个元素的列表x
    y = [0.95, 0.95, 0.05, 0.05, 0.5, 0.5]  # 创建一个包含6个元素的列表y
    ax.plot(x, y, 'o-', label='line')  # 在轴上绘制线图，并指定标签
    ax.set_xlim(0.0, 1.0)  # 设置x轴的显示范围
    ax.set_ylim(0.0, 1.0)  # 设置y轴的显示范围
    ax.legend(loc='best')  # 自动放置图例


def test_legend_auto4():
    """
    Check that the legend location with automatic placement is the same,
    whatever the histogram type is. Related to issue #9580.
    """
    # NB: barstacked is pointless with a single dataset.
    fig, axs = plt.subplots(ncols=3, figsize=(6.4, 2.4))  # 创建包含3个子图的图形对象，并指定尺寸
    leg_bboxes = []  # 创建一个空列表用于存储图例的边界框信息
    for ax, ht in zip(axs.flat, ('bar', 'step', 'stepfilled')):
        ax.set_title(ht)  # 设置子图标题为当前直方图类型
        # A high bar on the left but an even higher one on the right.
        ax.hist([0] + 5*[9], bins=range(10), label="Legend", histtype=ht)  # 在子图上绘制直方图，并指定直方图类型和标签
        leg = ax.legend(loc="best")  # 自动放置图例
        fig.canvas.draw()  # 绘制图形对象
        leg_bboxes.append(
            leg.get_window_extent().transformed(ax.transAxes.inverted()))  # 获取图例的边界框信息并存储在列表中

    # The histogram type "bar" is assumed to be the correct reference.
    # 检查第二个腿框的边界框是否与第一个腿框的边界框相近
    assert_allclose(leg_bboxes[1].bounds, leg_bboxes[0].bounds)
    # 检查第三个腿框的边界框是否与第一个腿框的边界框相近
    assert_allclose(leg_bboxes[2].bounds, leg_bboxes[0].bounds)
def test_legend_auto5():
    """
    Check that the automatic placement handle a rather complex
    case with non rectangular patch. Related to issue #9580.
    """
    # 创建一个包含两个子图的图形对象，每个子图尺寸为 9.6x4.8 英寸
    fig, axs = plt.subplots(ncols=2, figsize=(9.6, 4.8))

    # 存储图例框的列表
    leg_bboxes = []

    # 遍历每个子图及其对应的位置
    for ax, loc in zip(axs.flat, ("center", "best")):
        # 在子图上添加三个图形补丁对象：椭圆 Ellipse、多边形 Polygon 和楔形 Wedge
        # 这些图形用于测试图例的正确放置
        for _patch in [
                mpatches.Ellipse(
                    xy=(0.5, 0.9), width=0.8, height=0.2, fc="C1"),
                mpatches.Polygon(np.array([
                    [0, 1], [0, 0], [1, 0], [1, 1], [0.9, 1.0], [0.9, 0.1],
                    [0.1, 0.1], [0.1, 1.0], [0.1, 1.0]]), fc="C1"),
                mpatches.Wedge((0.5, 0.5), 0.5, 0, 360, width=0.05, fc="C0")
                ]:
            ax.add_patch(_patch)

        # 在子图上添加一条线段，并为其添加标签 "A segment"
        ax.plot([0.1, 0.9], [0.9, 0.9], label="A segment")  # sthg to label

        # 添加图例，并指定位置为 loc
        leg = ax.legend(loc=loc)

        # 刷新图形以确保图例被正确绘制
        fig.canvas.draw()

        # 将图例的窗口范围添加到列表中，用于后续断言比较
        leg_bboxes.append(
            leg.get_window_extent().transformed(ax.transAxes.inverted()))

    # 断言：比较第二个图例框的边界与第一个的边界是否接近
    assert_allclose(leg_bboxes[1].bounds, leg_bboxes[0].bounds)


@image_comparison(['legend_various_labels'], remove_text=True)
def test_various_labels():
    # tests all sorts of label types
    # 创建一个图形对象
    fig = plt.figure()

    # 添加一个子图
    ax = fig.add_subplot(121)

    # 在子图上绘制一系列数据点，并为每个数据点添加不同的标签
    ax.plot(np.arange(4), 'o', label=1)
    ax.plot(np.linspace(4, 4.1), 'o', label='Développés')
    ax.plot(np.arange(4, 1, -1), 'o', label='__nolegend__')

    # 添加图例，设置点的数量为1，位置为 'best'
    ax.legend(numpoints=1, loc='best')


def test_legend_label_with_leading_underscore():
    """
    Test that artists with labels starting with an underscore are not added to
    the legend, and that a warning is issued if one tries to add them
    explicitly.
    """
    # 创建一个图形对象和子图
    fig, ax = plt.subplots()

    # 在子图上绘制一条线段，并为其设置标签 '_foo'
    line, = ax.plot([0, 1], label='_foo')

    # 使用 pytest 的 warns 方法检查是否有 MatplotlibDeprecationWarning 警告信息
    with pytest.warns(_api.MatplotlibDeprecationWarning, match="with an underscore"):
        # 尝试添加带有下划线开头标签的图例句柄
        legend = ax.legend(handles=[line])

    # 断言：确保图例句柄列表长度为0，即未成功添加带有下划线开头标签的图例句柄
    assert len(legend.legend_handles) == 0


@image_comparison(['legend_labels_first.png'], remove_text=True,
                  tol=0.013 if platform.machine() == 'arm64' else 0)
def test_labels_first():
    # test labels to left of markers
    # 创建一个图形对象和子图
    fig, ax = plt.subplots()

    # 在子图上绘制三条线段，并为每条线段设置不同的标签
    ax.plot(np.arange(10), '-o', label=1)
    ax.plot(np.ones(10)*5, ':x', label="x")
    ax.plot(np.arange(20, 10, -1), 'd', label="diamond")

    # 添加图例，设置 markerfirst 参数为 False，使得标签位于标记之后
    ax.legend(loc='best', markerfirst=False)


@image_comparison(['legend_multiple_keys.png'], remove_text=True,
                  tol=0.013 if platform.machine() == 'arm64' else 0)
def test_multiple_keys():
    # test legend entries with multiple keys
    # 创建一个图形对象和子图
    fig, ax = plt.subplots()

    # 绘制三条线段，并分别获取它们的句柄
    p1, = ax.plot([1, 2, 3], '-o')
    p2, = ax.plot([2, 3, 4], '-x')
    p3, = ax.plot([3, 4, 5], '-d')
    # 在图表中添加图例，分别显示三种条目，每种条目包括一个元组和对应的标签
    ax.legend([(p1, p2), (p2, p1), p3], ['two keys', 'pad=0', 'one key'],
              # 设置每个图例条目中要显示的点数
              numpoints=1,
              # 定义处理程序映射，将每个元组映射到相应的处理程序对象
              handler_map={(p1, p2): HandlerTuple(ndivide=None),
                           (p2, p1): HandlerTuple(ndivide=None, pad=0)})
# 对比测试两张图像，验证 alpha 通道设置是否正确，移除图中的文本
@image_comparison(['rgba_alpha.png'], remove_text=True,
                  tol=0 if platform.machine() == 'x86_64' else 0.03)
def test_alpha_rgba():
    # 创建图像和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上绘制折线图，线宽为5
    ax.plot(range(10), lw=5)
    # 添加图例，包含一个长标签，并设置位置为中心
    leg = plt.legend(['Longlabel that will go away'], loc='center')
    # 设置图例背景颜色的 alpha 通道为 0.5
    leg.legendPatch.set_facecolor([1, 0, 0, 0.5])


# 对比测试两张图像，验证 rcparam 设置下 alpha 通道是否正确，移除图中的文本
@image_comparison(['rcparam_alpha.png'], remove_text=True,
                  tol=0 if platform.machine() == 'x86_64' else 0.03)
def test_alpha_rcparam():
    # 创建图像和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上绘制折线图，线宽为5
    ax.plot(range(10), lw=5)
    # 在 rc 参数上下文中设置图例框的 alpha 为 0.75
    with mpl.rc_context(rc={'legend.framealpha': .75}):
        # 添加图例，包含一个长标签，并设置位置为中心
        leg = plt.legend(['Longlabel that will go away'], loc='center')
        # 由于 rcparam 设置了 alpha，以下设置的 alpha 将被覆盖
        leg.legendPatch.set_facecolor([1, 0, 0, 0.5])


# 对比测试两张图像，验证各种绘图元素的细节展示是否正确，移除图中的文本
@image_comparison(['fancy'], remove_text=True, tol=0.05)
def test_fancy():
    # 使用 subplot 触发某些未在其他地方测试的 offsetbox 功能
    plt.subplot(121)
    # 绘制线图
    plt.plot([5] * 10, 'o--', label='XX')
    # 绘制散点图
    plt.scatter(np.arange(10), np.arange(10, 0, -1), label='XX\nXX')
    # 绘制带误差线的图
    plt.errorbar(np.arange(10), np.arange(10), xerr=0.5,
                 yerr=0.5, label='XX')
    # 添加图例，设置位置为左中，使用两列显示，带阴影和标题
    plt.legend(loc="center left", bbox_to_anchor=[1.0, 0.5],
               ncols=2, shadow=True, title="My legend", numpoints=1)


# 对比测试两张图像，验证图例框的 alpha 设置是否正确，移除图中的文本
@image_comparison(['framealpha'], remove_text=True,
                  tol=0 if platform.machine() == 'x86_64' else 0.024)
def test_framealpha():
    # 创建数据
    x = np.linspace(1, 100, 100)
    y = x
    # 绘制线图
    plt.plot(x, y, label='mylabel', lw=10)
    # 添加图例，设置图例框的 alpha 为 0.5
    plt.legend(framealpha=0.5)


# 对比测试两张图像，验证 rc 参数设置是否正确，移除图中的文本
@image_comparison(['scatter_rc3', 'scatter_rc1'], remove_text=True)
def test_rc():
    # 使用 subplot 触发某些未在其他地方测试的 offsetbox 功能
    plt.figure()
    ax = plt.subplot(121)
    # 绘制散点图
    ax.scatter(np.arange(10), np.arange(10, 0, -1), label='three')
    # 添加图例，设置位置为左中，带标题
    ax.legend(loc="center left", bbox_to_anchor=[1.0, 0.5],
              title="My legend")

    # 修改 rc 参数，设置图例散点数为1
    mpl.rcParams['legend.scatterpoints'] = 1
    plt.figure()
    ax = plt.subplot(121)
    # 绘制散点图
    ax.scatter(np.arange(10), np.arange(10, 0, -1), label='one')
    # 添加图例，设置位置为左中，带标题
    ax.legend(loc="center left", bbox_to_anchor=[1.0, 0.5],
              title="My legend")


# 对比测试两张图像，验证图例展开模式是否正常工作，移除图中的文本
@image_comparison(['legend_expand'], remove_text=True)
def test_legend_expand():
    """Test expand mode"""
    # 测试展开模式
    legend_modes = [None, "expand"]
    # 创建多个子图
    fig, axs = plt.subplots(len(legend_modes), 1)
    x = np.arange(100)
    # 对于给定的每个绘图轴 ax 和对应的图例模式 mode，执行以下操作：
    for ax, mode in zip(axs, legend_modes):
        # 在当前轴上绘制 y=1 的数据点，使用圆圈标记 'o'，并指定标签为 'y=1'
        ax.plot(x, 50 - x, 'o', label='y=1')
        # 在左上角添加图例 l1，位置为 'upper left'，使用指定的图例模式 mode
        l1 = ax.legend(loc='upper left', mode=mode)
        # 将 l1 添加到轴上，以保证多个图例的正确显示
        ax.add_artist(l1)
        # 在当前轴上绘制 y=-1 的数据点，使用圆圈标记 'o'，并指定标签为 'y=-1'
        ax.plot(x, x - 50, 'o', label='y=-1')
        # 在右侧添加图例 l2，位置为 'right'，使用指定的图例模式 mode
        l2 = ax.legend(loc='right', mode=mode)
        # 将 l2 添加到轴上，以保证多个图例的正确显示
        ax.add_artist(l2)
        # 在左下角添加图例，位置为 'lower left'，使用指定的图例模式 mode，并设置列数为 2
        ax.legend(loc='lower left', mode=mode, ncols=2)
@image_comparison(['hatching'], remove_text=True, style='default')
# 使用 image_comparison 装饰器比较图像，参数包括要比较的图像列表、是否移除文本以及样式设置为默认

def test_hatching():
    # 当重新生成此图像时，移除图例文本
    # 当重新生成此测试图像时，移除此行
    plt.rcParams['text.kerning_factor'] = 6
    # 设置文本间距因子为6，影响文本渲染的间距和布局

    fig, ax = plt.subplots()
    # 创建图和轴对象

    # Patches
    patch = plt.Rectangle((0, 0), 0.3, 0.3, hatch='xx',
                          label='Patch\ndefault color\nfilled')
    # 创建矩形补丁对象，指定位置、宽度、高度和填充图案'hatch'，设置标签

    ax.add_patch(patch)
    # 将补丁对象添加到轴上

    patch = plt.Rectangle((0.33, 0), 0.3, 0.3, hatch='||', edgecolor='C1',
                          label='Patch\nexplicit color\nfilled')
    # 创建矩形补丁对象，指定位置、宽度、高度、边缘颜色和填充图案'hatch'，设置标签

    ax.add_patch(patch)
    # 将补丁对象添加到轴上

    patch = plt.Rectangle((0, 0.4), 0.3, 0.3, hatch='xx', fill=False,
                          label='Patch\ndefault color\nunfilled')
    # 创建矩形补丁对象，指定位置、宽度、高度、无填充和填充图案'hatch'，设置标签

    ax.add_patch(patch)
    # 将补丁对象添加到轴上

    patch = plt.Rectangle((0.33, 0.4), 0.3, 0.3, hatch='||', fill=False,
                          edgecolor='C1',
                          label='Patch\nexplicit color\nunfilled')
    # 创建矩形补丁对象，指定位置、宽度、高度、无填充、边缘颜色和填充图案'hatch'，设置标签

    ax.add_patch(patch)
    # 将补丁对象添加到轴上

    # Paths
    ax.fill_between([0, .15, .3], [.8, .8, .8], [.9, 1.0, .9],
                    hatch='+', label='Path\ndefault color')
    # 在路径间填充区域，指定 x 范围、y 范围、填充图案'hatch'和标签

    ax.fill_between([.33, .48, .63], [.8, .8, .8], [.9, 1.0, .9],
                    hatch='+', edgecolor='C2', label='Path\nexplicit color')
    # 在路径间填充区域，指定 x 范围、y 范围、填充图案'hatch'、边缘颜色和标签

    ax.set_xlim(-0.01, 1.1)
    # 设置 x 轴显示范围

    ax.set_ylim(-0.01, 1.1)
    # 设置 y 轴显示范围

    ax.legend(handlelength=4, handleheight=4)
    # 创建图例，设置图例条目的长度和高度
    # 测试无参数情况下的图例
    def test_legend_no_args(self):
        # 绘制一条线并设置其标签
        lines = plt.plot(range(10), label='hello world')
        # 使用 mock.patch 创建虚拟的 Legend 对象
        with mock.patch('matplotlib.legend.Legend') as Legend:
            # 调用 plt.legend() 方法
            plt.legend()
        # 验证 Legend.assert_called_with 是否被调用，并检查参数
        Legend.assert_called_with(plt.gca(), lines, ['hello world'])

    # 测试指定位置参数 handles 和 labels 的图例
    def test_legend_positional_handles_labels(self):
        # 绘制一条线
        lines = plt.plot(range(10))
        # 使用 mock.patch 创建虚拟的 Legend 对象
        with mock.patch('matplotlib.legend.Legend') as Legend:
            # 调用 plt.legend(lines, ['hello world']) 方法
            plt.legend(lines, ['hello world'])
        # 验证 Legend.assert_called_with 是否被调用，并检查参数
        Legend.assert_called_with(plt.gca(), lines, ['hello world'])

    # 测试仅指定位置参数 handles 的图例
    def test_legend_positional_handles_only(self):
        # 绘制一条线
        lines = plt.plot(range(10))
        # 使用 pytest.raises 检测是否抛出 TypeError 异常，并匹配指定的错误信息
        with pytest.raises(TypeError, match='but found an Artist'):
            # 调用 plt.legend(lines) 方法，通常这种情况下会出现错误
            plt.legend(lines)

    # 测试仅指定位置参数 labels 的图例
    def test_legend_positional_labels_only(self):
        # 绘制一条线并设置其标签
        lines = plt.plot(range(10), label='hello world')
        # 使用 mock.patch 创建虚拟的 Legend 对象
        with mock.patch('matplotlib.legend.Legend') as Legend:
            # 调用 plt.legend(['foobar']) 方法
            plt.legend(['foobar'])
        # 验证 Legend.assert_called_with 是否被调用，并检查参数
        Legend.assert_called_with(plt.gca(), lines, ['foobar'])

    # 测试指定三个参数的图例
    def test_legend_three_args(self):
        # 绘制一条线并设置其标签
        lines = plt.plot(range(10), label='hello world')
        # 使用 mock.patch 创建虚拟的 Legend 对象
        with mock.patch('matplotlib.legend.Legend') as Legend:
            # 调用 plt.legend(lines, ['foobar'], loc='right') 方法
            plt.legend(lines, ['foobar'], loc='right')
        # 验证 Legend.assert_called_with 是否被调用，并检查参数
        Legend.assert_called_with(plt.gca(), lines, ['foobar'], loc='right')

    # 测试指定 handler_map 参数的图例
    def test_legend_handler_map(self):
        # 绘制一条线并设置其标签
        lines = plt.plot(range(10), label='hello world')
        # 使用 mock.patch 创建虚拟的 _get_legend_handles_labels 函数
        with mock.patch('matplotlib.legend._get_legend_handles_labels') as handles_labels:
            # 设置 handles_labels 的返回值
            handles_labels.return_value = lines, ['hello world']
            # 调用 plt.legend(handler_map={'1': 2}) 方法
            plt.legend(handler_map={'1': 2})
        # 验证 handles_labels.assert_called_with 是否被调用，并检查参数
        handles_labels.assert_called_with([plt.gca()], {'1': 2})

    # 测试指定 kwargs 参数 handles 的图例
    def test_legend_kwargs_handles_only(self):
        # 创建一个包含多个子图的图形和轴对象
        fig, ax = plt.subplots()
        # 在第一个子图上绘制三条线，并设置各自的标签
        x = np.linspace(0, 1, 11)
        ln1, = ax.plot(x, x, label='x')
        ln2, = ax.plot(x, 2*x, label='2x')
        ln3, = ax.plot(x, 3*x, label='3x')
        # 使用 mock.patch 创建虚拟的 Legend 对象
        with mock.patch('matplotlib.legend.Legend') as Legend:
            # 调用 ax.legend(handles=[ln3, ln2]) 方法
            ax.legend(handles=[ln3, ln2])  # 逆序并不包含 ln1
        # 验证 Legend.assert_called_with 是否被调用，并检查参数
        Legend.assert_called_with(ax, [ln3, ln2], ['3x', '2x'])

    # 测试指定 kwargs 参数 labels 的图例
    def test_legend_kwargs_labels_only(self):
        # 创建一个包含多个子图的图形和轴对象
        fig, ax = plt.subplots()
        # 在第一个子图上绘制两条线
        x = np.linspace(0, 1, 11)
        ln1, = ax.plot(x, x)
        ln2, = ax.plot(x, 2*x)
        # 使用 mock.patch 创建虚拟的 Legend 对象
        with mock.patch('matplotlib.legend.Legend') as Legend:
            # 调用 ax.legend(labels=['x', '2x']) 方法
            ax.legend(labels=['x', '2x'])
        # 验证 Legend.assert_called_with 是否被调用，并检查参数
        Legend.assert_called_with(ax, [ln1, ln2], ['x', '2x'])
    def test_legend_kwargs_handles_labels(self):
        fig, ax = plt.subplots()  # 创建一个新的图形和一个子图对象
        th = np.linspace(0, 2*np.pi, 1024)  # 生成一个包含 1024 个点的角度数组
        lns, = ax.plot(th, np.sin(th), label='sin')  # 在子图上绘制正弦曲线，并设定标签为 'sin'
        lnc, = ax.plot(th, np.cos(th), label='cos')  # 在子图上绘制余弦曲线，并设定标签为 'cos'
        with mock.patch('matplotlib.legend.Legend') as Legend:
            # 使用 mock.patch 模拟 legend.Legend 类，用于测试
            # labels 参数覆盖了 lnc 和 lns 的标签 ('sin', 'cos')，修改为 ('a', 'b')
            ax.legend(labels=('a', 'b'), handles=(lnc, lns))
        Legend.assert_called_with(ax, (lnc, lns), ('a', 'b'))  # 断言 Legend 被调用，参数为 ax, (lnc, lns), ('a', 'b')

    def test_warn_mixed_args_and_kwargs(self):
        fig, ax = plt.subplots()  # 创建一个新的图形和一个子图对象
        th = np.linspace(0, 2*np.pi, 1024)  # 生成一个包含 1024 个点的角度数组
        lns, = ax.plot(th, np.sin(th), label='sin')  # 在子图上绘制正弦曲线，并设定标签为 'sin'
        lnc, = ax.plot(th, np.cos(th), label='cos')  # 在子图上绘制余弦曲线，并设定标签为 'cos'
        with pytest.warns(DeprecationWarning) as record:
            # 使用 pytest.warns 捕获 DeprecationWarning 警告信息
            ax.legend((lnc, lns), labels=('a', 'b'))  # 使用位置参数和关键字参数来设置图例
        assert len(record) == 1  # 断言捕获的警告记录长度为 1
        assert str(record[0].message).startswith(
            "You have mixed positional and keyword arguments, some input may "
            "be discarded.")  # 断言警告消息的开头部分符合预期的警告内容

    def test_parasite(self):
        from mpl_toolkits.axes_grid1 import host_subplot  # 导入 host_subplot 函数

        host = host_subplot(111)  # 创建一个主轴 subplot
        par = host.twinx()  # 创建一个与主轴共享 x 轴的次级轴 subplot

        p1, = host.plot([0, 1, 2], [0, 1, 2], label="Density")  # 在主轴上绘制密度曲线，并设定标签为 'Density'
        p2, = par.plot([0, 1, 2], [0, 3, 2], label="Temperature")  # 在次级轴上绘制温度曲线，并设定标签为 'Temperature'

        with mock.patch('matplotlib.legend.Legend') as Legend:
            plt.legend()  # 在当前图形中添加默认的图例
        Legend.assert_called_with(host, [p1, p2], ['Density', 'Temperature'])
        # 断言 Legend 被调用，参数为 host（主轴 subplot）、[p1, p2]（密度和温度曲线）、['Density', 'Temperature']（对应标签）
class TestLegendFigureFunction:
    # Tests the legend function for figure

    # 测试图例函数，用于图形

    def test_legend_handle_label(self):
        # Creates a new figure and axis
        fig, ax = plt.subplots()
        # Plots a line on the axis
        lines = ax.plot(range(10))
        # Mocks the Legend class
        with mock.patch('matplotlib.legend.Legend') as Legend:
            # Adds a legend to the figure with specified lines and labels
            fig.legend(lines, ['hello world'])
        # Asserts that Legend was called with specific arguments
        Legend.assert_called_with(fig, lines, ['hello world'],
                                  bbox_transform=fig.transFigure)

    def test_legend_no_args(self):
        # Creates a new figure and axis
        fig, ax = plt.subplots()
        # Plots a line on the axis with a label
        lines = ax.plot(range(10), label='hello world')
        # Mocks the Legend class
        with mock.patch('matplotlib.legend.Legend') as Legend:
            # Adds a legend to the figure without arguments
            fig.legend()
        # Asserts that Legend was called with specific arguments
        Legend.assert_called_with(fig, lines, ['hello world'],
                                  bbox_transform=fig.transFigure)

    def test_legend_label_arg(self):
        # Creates a new figure and axis
        fig, ax = plt.subplots()
        # Plots a line on the axis
        lines = ax.plot(range(10))
        # Mocks the Legend class
        with mock.patch('matplotlib.legend.Legend') as Legend:
            # Adds a legend to the figure with specified labels
            fig.legend(['foobar'])
        # Asserts that Legend was called with specific arguments
        Legend.assert_called_with(fig, lines, ['foobar'],
                                  bbox_transform=fig.transFigure)

    def test_legend_label_three_args(self):
        # Creates a new figure and axis
        fig, ax = plt.subplots()
        # Plots a line on the axis
        lines = ax.plot(range(10))
        # Checks that TypeError is raised with specific message
        with pytest.raises(TypeError, match="0-2"):
            # Adds a legend to the figure with specified labels and position
            fig.legend(lines, ['foobar'], 'right')
        # Checks that TypeError is raised with specific message
        with pytest.raises(TypeError, match="0-2"):
            # Adds a legend to the figure with specified labels, position, and additional argument
            fig.legend(lines, ['foobar'], 'right', loc='left')

    def test_legend_kw_args(self):
        # Creates a new figure with two subplots
        fig, axs = plt.subplots(1, 2)
        # Plots a line on the first subplot
        lines = axs[0].plot(range(10))
        # Plots a line on the second subplot
        lines2 = axs[1].plot(np.arange(10) * 2.)
        # Mocks the Legend class
        with mock.patch('matplotlib.legend.Legend') as Legend:
            # Adds a legend to the figure with specified location, labels, and handles
            fig.legend(loc='right', labels=('a', 'b'), handles=(lines, lines2))
        # Asserts that Legend was called with specific arguments
        Legend.assert_called_with(
            fig, (lines, lines2), ('a', 'b'), loc='right',
            bbox_transform=fig.transFigure)

    def test_warn_args_kwargs(self):
        # Creates a new figure with two subplots
        fig, axs = plt.subplots(1, 2)
        # Plots a line on the first subplot
        lines = axs[0].plot(range(10))
        # Plots a line on the second subplot
        lines2 = axs[1].plot(np.arange(10) * 2.)
        # Checks for DeprecationWarning
        with pytest.warns(DeprecationWarning) as record:
            # Adds a legend to the figure with mixed positional and keyword arguments
            fig.legend((lines, lines2), labels=('a', 'b'))
        # Asserts that a warning message was issued
        assert len(record) == 1
        assert str(record[0].message).startswith(
            "You have mixed positional and keyword arguments, some input may "
            "be discarded.")


def test_figure_legend_outside():
    # Generates a list of legend positions
    todos = ['upper ' + pos for pos in ['left', 'center', 'right']]
    todos += ['lower ' + pos for pos in ['left', 'center', 'right']]
    todos += ['left ' + pos for pos in ['lower', 'center', 'upper']]
    todos += ['right ' + pos for pos in ['lower', 'center', 'upper']]

    # Defines coordinates for various legend positions
    upperext = [20.347556,  27.722556, 790.583, 545.499]
    lowerext = [20.347556,  71.056556, 790.583, 588.833]
    leftext = [151.681556, 27.722556, 790.583, 588.833]
    rightext = [20.347556,  27.722556, 659.249, 588.833]
    # 定义一个包含重复值的列表，用于定义坐标轴边界的扩展
    axbb = [upperext, upperext, upperext,
            lowerext, lowerext, lowerext,
            leftext, leftext, leftext,
            rightext, rightext, rightext]

    # 定义包含坐标轴和图例边界框的列表，每个元素是一个四元素列表，表示坐标轴或图例的边界坐标
    legbb = [[10., 555., 133., 590.],     # upper left
             [338.5, 555., 461.5, 590.],  # upper center
             [667, 555., 790.,  590.],    # upper right
             [10., 10., 133.,  45.],      # lower left
             [338.5, 10., 461.5,  45.],   # lower center
             [667., 10., 790.,  45.],     # lower right
             [10., 10., 133., 45.],       # left lower
             [10., 282.5, 133., 317.5],   # left center
             [10., 555., 133., 590.],     # left upper
             [667, 10., 790., 45.],       # right lower
             [667., 282.5, 790., 317.5],  # right center
             [667., 555., 790., 590.]]    # right upper

    # 遍历给定的todos列表，其中包含要执行的操作
    for nn, todo in enumerate(todos):
        # 打印当前要执行的操作
        print(todo)
        # 创建一个包含子图的图形对象fig，并使用约束布局和指定的DPI
        fig, axs = plt.subplots(constrained_layout=True, dpi=100)
        # 在子图axs上绘制简单的折线图，添加标签为'Boo1'
        axs.plot(range(10), label='Boo1')
        # 在图形上添加位于指定位置'todo'外部的图例
        leg = fig.legend(loc='outside ' + todo)
        # 在不进行渲染的情况下绘制图形
        fig.draw_without_rendering()

        # 断言检查当前坐标轴的窗口范围是否与预期的axbb[nn]边界匹配
        assert_allclose(axs.get_window_extent().extents,
                        axbb[nn])
        # 断言检查当前图例的窗口范围是否与预期的legbb[nn]边界匹配
        assert_allclose(leg.get_window_extent().extents,
                        legbb[nn])
@image_comparison(['legend_stackplot.png'],
                  tol=0.031 if platform.machine() == 'arm64' else 0)
# 定义一个测试函数，用于测试使用 stackplot 生成的 PolyCollection 是否正确显示图例
def test_legend_stackplot():
    """Test legend for PolyCollection using stackplot."""
    # 创建一个新的图形和轴
    fig, ax = plt.subplots()
    # 创建数据
    x = np.linspace(0, 10, 10)
    y1 = 1.0 * x
    y2 = 2.0 * x + 1
    y3 = 3.0 * x + 2
    # 使用 stackplot 绘制堆叠区域图，并设置标签
    ax.stackplot(x, y1, y2, y3, labels=['y1', 'y2', 'y3'])
    # 设置 x 轴和 y 轴的显示范围
    ax.set_xlim((0, 10))
    ax.set_ylim((0, 70))
    # 添加图例，位置设为最佳适配
    ax.legend(loc='best')


def test_cross_figure_patch_legend():
    # 创建两个图形和对应的轴
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    # 在第一个图形上创建柱状图
    brs = ax.bar(range(3), range(3))
    # 在第二个图形上创建图例，指定柱状图对象和标签 'foo'
    fig2.legend(brs, 'foo')


def test_nanscatter():
    # 创建图形和轴
    fig, ax = plt.subplots()

    # 绘制包含 NaN 值的散点图，设置颜色、标记、大小等属性
    h = ax.scatter([np.nan], [np.nan], marker="o",
                   facecolor="r", edgecolor="r", s=3)

    # 添加散点图的图例
    ax.legend([h], ["scatter"])

    # 创建新的图形和轴
    fig, ax = plt.subplots()
    # 循环绘制三种颜色的散点图
    for color in ['red', 'green', 'blue']:
        n = 750
        x, y = np.random.rand(2, n)
        scale = 200.0 * np.random.rand(n)
        # 绘制散点图，设置颜色、标签、透明度、边缘颜色等属性
        ax.scatter(x, y, c=color, s=scale, label=color,
                   alpha=0.3, edgecolors='none')

    # 添加图例，默认位置
    ax.legend()
    # 显示网格
    ax.grid(True)


def test_legend_repeatcheckok():
    # 创建图形和轴
    fig, ax = plt.subplots()
    # 在轴上绘制两个点，分别设置不同的颜色和标记，并添加标签 'test'
    ax.scatter(0.0, 1.0, color='k', marker='o', label='test')
    ax.scatter(0.5, 0.0, color='r', marker='v', label='test')
    # 添加图例
    ax.legend()
    # 获取图例的句柄和标签
    hand, lab = mlegend._get_legend_handles_labels([ax])
    # 断言图例标签的长度为 2
    assert len(lab) == 2

    # 创建新的图形和轴
    fig, ax = plt.subplots()
    # 在轴上绘制两个点，设置相同的颜色和标记，并添加标签 'test'
    ax.scatter(0.0, 1.0, color='k', marker='o', label='test')
    ax.scatter(0.5, 0.0, color='k', marker='v', label='test')
    # 添加图例
    ax.legend()
    # 获取图例的句柄和标签
    hand, lab = mlegend._get_legend_handles_labels([ax])
    # 断言图例标签的长度为 2
    assert len(lab) == 2


@image_comparison(['not_covering_scatter.png'])
def test_not_covering_scatter():
    # 定义颜色列表
    colors = ['b', 'g', 'r']

    # 循环绘制三个散点图，使用不同的颜色
    for n in range(3):
        plt.scatter([n], [n], color=colors[n])

    # 添加图例，使用 'foo' 作为标签，位置设为最佳适配
    plt.legend(['foo', 'foo', 'foo'], loc='best')
    # 设置 x 轴和 y 轴的显示范围
    plt.gca().set_xlim(-0.5, 2.2)
    plt.gca().set_ylim(-0.5, 2.2)


@image_comparison(['not_covering_scatter_transform.png'])
def test_not_covering_scatter_transform():
    # 创建平移变换对象，将散点图位置偏移至左上角，默认自动位置
    offset = mtransforms.Affine2D().translate(-20, 20)
    x = np.linspace(0, 30, 1000)
    # 绘制直线图
    plt.plot(x, x)

    # 在特定变换下绘制散点图，偏移应用于当前坐标系
    plt.scatter([20], [10], transform=offset + plt.gca().transData)

    # 添加图例，标签为 'foo' 和 'bar'，位置设为最佳适配
    plt.legend(['foo', 'bar'], loc='best')


def test_linecollection_scaled_dashes():
    # 定义多组线段的列表
    lines1 = [[(0, .5), (.5, 1)], [(.3, .6), (.2, .2)]]
    lines2 = [[[0.7, .2], [.8, .4]], [[.5, .7], [.6, .1]]]
    lines3 = [[[0.6, .2], [.8, .4]], [[.5, .7], [.1, .1]]]
    # 创建 LineCollection 对象，设置不同的线型和线宽
    lc1 = mcollections.LineCollection(lines1, linestyles="--", lw=3)
    lc2 = mcollections.LineCollection(lines2, linestyles="-.")
    lc3 = mcollections.LineCollection(lines3, linestyles=":", lw=.5)

    # 创建图形和轴
    fig, ax = plt.subplots()
    # 将 LineCollection 对象添加到轴上
    ax.add_collection(lc1)
    ax.add_collection(lc2)
    ax.add_collection(lc3)

    # 添加图例，标签分别为 "line1", "line2", 'line 3'，位置设为最佳适配
    leg = ax.legend([lc1, lc2, lc3], ["line1", "line2", 'line 3'])
    # 从图例对象 `leg` 中获取前三个图例句柄，并分别赋值给 h1, h2, h3
    h1, h2, h3 = leg.legend_handles

    # 遍历 lc1, lc2, lc3 和 h1, h2, h3，对每对对象执行断言检查
    for oh, lh in zip((lc1, lc2, lc3), (h1, h2, h3)):
        # 检查每个 oh（lc1, lc2, lc3）的第一个线条样式是否与对应 lh（h1, h2, h3）的虚线模式相匹配
        assert oh.get_linestyles()[0] == lh._dash_pattern
def test_handler_numpoints():
    """Test legend handler with numpoints <= 1."""
    # 创建一个包含单个图表和坐标轴的图形对象
    fig, ax = plt.subplots()
    # 绘制一条简单的曲线，设置标签为 'test'
    ax.plot(range(5), label='test')
    # 添加图例，设置 numpoints 参数为 0.5，这个值不合法，会被截断为整数 0


def test_text_nohandler_warning():
    """Test that Text artists with labels raise a warning"""
    # 创建一个包含单个图表和坐标轴的图形对象
    fig, ax = plt.subplots()
    # 绘制一条包含单个数据点的曲线，设置标签为 "mock data"
    ax.plot([0], label="mock data")
    # 在图表上添加一个文本对象，设置其位置和标签
    ax.text(x=0, y=0, s="text", label="label")
    # 测试在调用 legend() 时是否会产生 UserWarning 警告
    with pytest.warns(UserWarning) as record:
        ax.legend()
    # 确保产生了且仅产生了一个警告记录
    assert len(record) == 1

    # 下面的代码不应该产生警告
    f, ax = plt.subplots()
    # 绘制一个随机颜色的网格图
    ax.pcolormesh(np.random.uniform(0, 1, (10, 10)))
    # 使用 warnings 模块捕获并过滤所有警告，将警告设置为错误级别
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # 调用 get_legend_handles_labels() 方法，不应该产生警告


def test_empty_bar_chart_with_legend():
    """Test legend when bar chart is empty with a label."""
    # 相关于问题 #13003。调用 plt.legend() 不应该引发 IndexError。
    # 绘制一个空的柱状图，并设置一个标签 'test'
    plt.bar([], [], label='test')
    # 添加图例
    plt.legend()


@image_comparison(['shadow_argument_types.png'], remove_text=True, style='mpl20',
                  tol=0.028 if platform.machine() == 'arm64' else 0)
def test_shadow_argument_types():
    # 测试不同阴影参数的效果
    fig, ax = plt.subplots()
    # 绘制一条简单的曲线，设置标签为 'test'
    ax.plot([1, 2, 3], label='test')

    # 测试不同的阴影配置
    # 以及不同的颜色指定方式
    legs = (ax.legend(loc='upper left', shadow=True),    # True
            ax.legend(loc='upper right', shadow=False),  # False
            ax.legend(loc='center left',                 # string
                      shadow={'color': 'red', 'alpha': 0.1}),
            ax.legend(loc='center right',                # tuple
                      shadow={'color': (0.1, 0.2, 0.5), 'oy': -5}),
            ax.legend(loc='lower left',                   # tab
                      shadow={'color': 'tab:cyan', 'ox': 10})
            )
    for l in legs:
        ax.add_artist(l)
    # 添加默认位置的图例
    ax.legend(loc='lower right')  # default


def test_shadow_invalid_argument():
    # 测试传递给 legend 阴影参数的无效参数是否会引发 ValueError
    fig, ax = plt.subplots()
    # 绘制一条简单的曲线，设置标签为 'test'
    ax.plot([1, 2, 3], label='test')
    # 使用 pytest.raises 检查是否会引发 ValueError，并匹配错误消息中的 "dict or bool"
    with pytest.raises(ValueError, match="dict or bool"):
        ax.legend(loc="upper left", shadow="aardvark")  # Bad argument


def test_shadow_framealpha():
    # 测试当阴影为 True 且未显式传递 framealpha 时是否会激活 framealpha
    fig, ax = plt.subplots()
    # 绘制一条包含 100 个数据点的曲线，设置标签为 "test"
    ax.plot(range(100), label="test")
    # 添加带有阴影和白色背景色的图例
    leg = ax.legend(shadow=True, facecolor='w')
    # 断言图例的框架透明度为 1
    assert leg.get_frame().get_alpha() == 1


def test_legend_title_empty():
    # 测试如果不设置图例标题，它将返回一个空字符串，并且标题不可见
    fig, ax = plt.subplots()
    # 绘制一条包含 10 个数据点的曲线，设置标签为 "mock data"
    ax.plot(range(10), label="mock data")
    # 添加图例
    leg = ax.legend()
    # 断言图例标题的文本内容为空字符串
    assert leg.get_title().get_text() == ""
    # 断言图例标题不可见
    assert not leg.get_title().get_visible()
# 测试确保图例在不同 dpi 下返回预期的范围...
def test_legend_proper_window_extent():
    # 创建 DPI 为 100 的图表对象
    fig, ax = plt.subplots(dpi=100)
    # 绘制简单的折线图，并添加图例 'Aardvark'
    ax.plot(range(10), label='Aardvark')
    # 获取图例对象
    leg = ax.legend()
    # 获取图例在画布上的窗口范围，并获取其左上角 x 坐标
    x01 = leg.get_window_extent(fig.canvas.get_renderer()).x0

    # 创建 DPI 为 200 的图表对象
    fig, ax = plt.subplots(dpi=200)
    # 绘制简单的折线图，并添加图例 'Aardvark'
    ax.plot(range(10), label='Aardvark')
    # 获取图例对象
    leg = ax.legend()
    # 获取图例在画布上的窗口范围，并获取其左上角 x 坐标
    x02 = leg.get_window_extent(fig.canvas.get_renderer()).x0
    # 断言两个 x 坐标的值近似相等
    assert pytest.approx(x01*2, 0.1) == x02


def test_window_extent_cached_renderer():
    # 创建 DPI 为 100 的图表对象
    fig, ax = plt.subplots(dpi=100)
    # 绘制简单的折线图，并添加图例 'Aardvark'
    ax.plot(range(10), label='Aardvark')
    # 获取图例对象
    leg = ax.legend()
    # 在绘制前，获取图例在画布上的窗口范围，以确保使用缓存的渲染器
    leg2 = fig.legend()
    # 绘制图表
    fig.canvas.draw()
    # 检查 get_window_extent 是否使用了缓存的渲染器
    leg.get_window_extent()
    leg2.get_window_extent()


def test_legend_title_fontprop_fontsize():
    # 测试 title_fontsize 关键字参数
    plt.plot(range(10), label="mock data")
    # 断言带有无效 title_fontsize 和 title_fontproperties 的图例抛出 ValueError
    with pytest.raises(ValueError):
        plt.legend(title='Aardvark', title_fontsize=22,
                   title_fontproperties={'family': 'serif', 'size': 22})

    # 创建带有自定义 title_fontproperties 的图例对象
    leg = plt.legend(title='Aardvark', title_fontproperties=FontProperties(
                                       family='serif', size=22))
    # 断言图例标题的字体大小为 22
    assert leg.get_title().get_size() == 22

    # 创建包含多个子图的图表对象
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flat
    # 在第一个子图上绘制折线图，并添加带有自定义 title_fontsize 的图例
    axes[0].plot(range(10), label="mock data")
    leg0 = axes[0].legend(title='Aardvark', title_fontsize=22)
    # 断言第一个子图的图例标题字体大小为 22
    assert leg0.get_title().get_fontsize() == 22

    # 在第二个子图上绘制折线图，并添加带有自定义 title_fontproperties 的图例
    axes[1].plot(range(10), label="mock data")
    leg1 = axes[1].legend(title='Aardvark',
                          title_fontproperties={'family': 'serif', 'size': 22})
    # 断言第二个子图的图例标题字体大小为 22
    assert leg1.get_title().get_fontsize() == 22

    # 在第三个子图上绘制折线图，并通过全局设置移除 title_fontsize 的自定义
    axes[2].plot(range(10), label="mock data")
    mpl.rcParams['legend.title_fontsize'] = None
    leg2 = axes[2].legend(title='Aardvark',
                          title_fontproperties={'family': 'serif'})
    # 断言第三个子图的图例标题字体大小等于全局默认字体大小
    assert leg2.get_title().get_fontsize() == mpl.rcParams['font.size']

    # 在第四个子图上绘制折线图，并添加只有部分全局设置的图例
    axes[3].plot(range(10), label="mock data")
    leg3 = axes[3].legend(title='Aardvark')
    # 断言第四个子图的图例标题字体大小等于全局默认字体大小
    assert leg3.get_title().get_fontsize() == mpl.rcParams['font.size']

    # 在第五个子图上绘制折线图，并添加带有全局设置 title_fontsize 的图例
    axes[4].plot(range(10), label="mock data")
    mpl.rcParams['legend.title_fontsize'] = 20
    leg4 = axes[4].legend(title='Aardvark',
                          title_fontproperties={'family': 'serif'})
    # 断言第五个子图的图例标题字体大小为 20
    assert leg4.get_title().get_fontsize() == 20

    # 在第六个子图上绘制折线图，并添加只有部分全局设置的图例
    axes[5].plot(range(10), label="mock data")
    leg5 = axes[5].legend(title='Aardvark')
    # 断言第六个子图的图例标题字体大小为 20（与全局设置相同）
    assert leg5.get_title().get_fontsize() == 20


@pytest.mark.parametrize('alignment', ('center', 'left', 'right'))
def test_legend_alignment(alignment):
    # 创建简单的图表对象
    fig, ax = plt.subplots()
    # 绘制简单的折线图，并添加带有指定对齐方式的图例 'Aardvark'
    leg = ax.legend(title="Aardvark", alignment=alignment)
    # 断言图例的第一个子对象的对齐方式与预期相符
    assert leg.get_children()[0].align == alignment
    # 断言图例的对齐方式与预期相符
    assert leg.get_alignment() == alignment


@pytest.mark.parametrize('loc', ('center', 'best',))
def test_ax_legend_set_loc(loc):
    # 创建简单的图表对象
    fig, ax = plt.subplots()
    # 在图形 ax 上绘制数据，使用默认的横坐标范围 [0, 10)，并添加标签 'test'
    ax.plot(range(10), label='test')
    
    # 在图形上添加图例，并将返回的图例对象赋给变量 leg
    leg = ax.legend()
    
    # 设置图例的位置为 loc 所指定的位置
    leg.set_loc(loc)
    
    # 断言确认图例对象的位置代码与预期的 mlegend.Legend.codes[loc] 相符
    assert leg._get_loc() == mlegend.Legend.codes[loc]
@pytest.mark.parametrize('loc', ('outside right', 'right',))
def test_fig_legend_set_loc(loc):
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 在轴上绘制一条曲线，并设置标签为'test'
    ax.plot(range(10), label='test')
    # 创建图例对象并将其添加到图形中
    leg = fig.legend()
    # 设置图例的位置
    leg.set_loc(loc)

    # 根据输入的位置字符串确定对应的位置代码
    loc = loc.split()[1] if loc.startswith("outside") else loc
    # 断言图例当前的位置代码与预期的位置代码相符
    assert leg._get_loc() == mlegend.Legend.codes[loc]


@pytest.mark.parametrize('alignment', ('center', 'left', 'right'))
def test_legend_set_alignment(alignment):
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 在轴上绘制一条曲线，并设置标签为'test'
    ax.plot(range(10), label='test')
    # 创建轴上的图例对象
    leg = ax.legend()
    # 设置图例的对齐方式
    leg.set_alignment(alignment)
    # 断言第一个图例项的对齐方式与设置的对齐方式相符
    assert leg.get_children()[0].align == alignment
    # 断言图例的当前对齐方式与设置的对齐方式相符
    assert leg.get_alignment() == alignment


@pytest.mark.parametrize('color', ('red', 'none', (.5, .5, .5)))
def test_legend_labelcolor_single(color):
    # 测试单一颜色的标签颜色设置
    fig, ax = plt.subplots()
    # 在轴上绘制三条曲线，并设置各自的标签
    ax.plot(np.arange(10), np.arange(10)*1, label='#1')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3')

    # 创建轴上的图例对象，并设置标签颜色
    leg = ax.legend(labelcolor=color)
    # 验证所有图例文本的颜色与设置的颜色相同
    for text in leg.get_texts():
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_labelcolor_list():
    # 测试使用颜色列表的标签颜色设置
    fig, ax = plt.subplots()
    # 在轴上绘制三条曲线，并设置各自的标签
    ax.plot(np.arange(10), np.arange(10)*1, label='#1')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3')

    # 创建轴上的图例对象，并设置标签颜色为列表形式
    leg = ax.legend(labelcolor=['r', 'g', 'b'])
    # 验证每个图例文本的颜色与对应颜色列表中的颜色相同
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_labelcolor_linecolor():
    # 测试标签颜色设置为'linecolor'的情况
    fig, ax = plt.subplots()
    # 在轴上绘制三条曲线，并设置各自的标签和颜色
    ax.plot(np.arange(10), np.arange(10)*1, label='#1', color='r')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2', color='g')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3', color='b')

    # 创建轴上的图例对象，并设置标签颜色为线条颜色
    leg = ax.legend(labelcolor='linecolor')
    # 验证每个图例文本的颜色与对应线条颜色相同
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_pathcollection_labelcolor_linecolor():
    # 测试PathCollection类型图例文本颜色设置为'linecolor'的情况
    fig, ax = plt.subplots()
    # 在轴上绘制三组散点图，并设置各自的标签和颜色
    ax.scatter(np.arange(10), np.arange(10)*1, label='#1', c='r')
    ax.scatter(np.arange(10), np.arange(10)*2, label='#2', c='g')
    ax.scatter(np.arange(10), np.arange(10)*3, label='#3', c='b')

    # 创建轴上的图例对象，并设置标签颜色为线条颜色
    leg = ax.legend(labelcolor='linecolor')
    # 验证每个图例文本的颜色与对应线条颜色相同
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_pathcollection_labelcolor_linecolor_iterable():
    # 测试PathCollection类型图例文本颜色设置为'linecolor'，并使用可迭代颜色的情况
    fig, ax = plt.subplots()
    # 随机选择颜色并在轴上绘制一组散点图，设置标签
    colors = np.random.default_rng().choice(['r', 'g', 'b'], 10)
    ax.scatter(np.arange(10), np.arange(10)*1, label='#1', c=colors)

    # 创建轴上的图例对象，并设置标签颜色为线条颜色
    leg = ax.legend(labelcolor='linecolor')
    # 获取 legend 对象中的文本对象，并将其解构赋值给变量 text
    text, = leg.get_texts()
    # 断言文本对象的颜色与 'black' 相同，确保颜色正确
    assert mpl.colors.same_color(text.get_color(), 'black')
def test_legend_pathcollection_labelcolor_linecolor_cmap():
    # 测试在使用 colormap 的情况下，labelcolor='linecolor' 对 PathCollection 的影响
    fig, ax = plt.subplots()
    # 创建散点图，并使用 colormap 来着色
    ax.scatter(np.arange(10), np.arange(10), c=np.arange(10), label='#1')

    # 添加图例，并设置图例文本颜色为 'linecolor'
    leg = ax.legend(labelcolor='linecolor')
    # 获取图例文本对象
    text, = leg.get_texts()
    # 断言图例文本的颜色与预期的 'black' 相同
    assert mpl.colors.same_color(text.get_color(), 'black')


def test_legend_labelcolor_markeredgecolor():
    # 测试在 labelcolor='markeredgecolor' 的情况下，对折线图的影响
    fig, ax = plt.subplots()
    # 绘制三条折线图，并指定各自的 markeredgecolor
    ax.plot(np.arange(10), np.arange(10)*1, label='#1', markeredgecolor='r')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2', markeredgecolor='g')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3', markeredgecolor='b')

    # 添加图例，并设置图例文本颜色为 'markeredgecolor'
    leg = ax.legend(labelcolor='markeredgecolor')
    # 遍历图例文本对象，断言其颜色与相应的 markeredgecolor 相同
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_pathcollection_labelcolor_markeredgecolor():
    # 测试在 labelcolor='markeredgecolor' 的情况下，对 PathCollection 的影响
    fig, ax = plt.subplots()
    # 创建三个散点图，并指定各自的 edgecolor
    ax.scatter(np.arange(10), np.arange(10)*1, label='#1', edgecolor='r')
    ax.scatter(np.arange(10), np.arange(10)*2, label='#2', edgecolor='g')
    ax.scatter(np.arange(10), np.arange(10)*3, label='#3', edgecolor='b')

    # 添加图例，并设置图例文本颜色为 'markeredgecolor'
    leg = ax.legend(labelcolor='markeredgecolor')
    # 遍历图例文本对象，断言其颜色与相应的 markeredgecolor 相同
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_pathcollection_labelcolor_markeredgecolor_iterable():
    # 测试在 labelcolor='markeredgecolor' 的情况下，对 PathCollection 使用 iterable 的 edgecolor 的影响
    fig, ax = plt.subplots()
    # 生成随机颜色数组作为散点图的 edgecolor
    colors = np.random.default_rng().choice(['r', 'g', 'b'], 10)
    ax.scatter(np.arange(10), np.arange(10)*1, label='#1', edgecolor=colors)

    # 添加图例，并设置图例文本颜色为 'markeredgecolor'
    leg = ax.legend(labelcolor='markeredgecolor')
    # 遍历图例文本对象，断言其颜色为 'k'（黑色）
    for text, color in zip(leg.get_texts(), ['k']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_pathcollection_labelcolor_markeredgecolor_cmap():
    # 测试在 labelcolor='markeredgecolor' 的情况下，对 PathCollection 使用 colormap 的影响
    fig, ax = plt.subplots()
    # 使用 colormap 来着色散点图的 edgecolor
    edgecolors = mpl.cm.viridis(np.random.rand(10))
    ax.scatter(
        np.arange(10),
        np.arange(10),
        label='#1',
        c=np.arange(10),
        edgecolor=edgecolors,
        cmap="Reds"
    )

    # 添加图例，并设置图例文本颜色为 'markeredgecolor'
    leg = ax.legend(labelcolor='markeredgecolor')
    # 遍历图例文本对象，断言其颜色为 'k'（黑色）
    for text, color in zip(leg.get_texts(), ['k']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_labelcolor_markerfacecolor():
    # 测试在 labelcolor='markerfacecolor' 的情况下，对折线图的影响
    fig, ax = plt.subplots()
    # 绘制三条折线图，并指定各自的 markerfacecolor
    ax.plot(np.arange(10), np.arange(10)*1, label='#1', markerfacecolor='r')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2', markerfacecolor='g')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3', markerfacecolor='b')
    # 在图形 ax 上创建图例，并设置标签颜色为与标记面颜色相同的颜色
    leg = ax.legend(labelcolor='markerfacecolor')
    
    # 遍历图例中的文本和预定义的颜色列表，确保每个文本的颜色与对应的颜色一致
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)
def test_legend_pathcollection_labelcolor_markerfacecolor():
    # 测试对 PathCollection 上 labelcolor='markerfacecolor' 的标签颜色
    # 创建一个图和轴对象
    fig, ax = plt.subplots()
    # 绘制三个散点图，每个散点图有不同的标签和面部颜色
    ax.scatter(np.arange(10), np.arange(10)*1, label='#1', facecolor='r')
    ax.scatter(np.arange(10), np.arange(10)*2, label='#2', facecolor='g')
    ax.scatter(np.arange(10), np.arange(10)*3, label='#3', facecolor='b')

    # 添加图例，使得标签颜色与散点面部颜色相同
    leg = ax.legend(labelcolor='markerfacecolor')
    # 遍历图例中的文本和颜色，断言它们的颜色与预期的一致
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_pathcollection_labelcolor_markerfacecolor_iterable():
    # 测试带有可迭代颜色的 PathCollection 上 labelcolor='markerfacecolor' 的标签颜色
    # 创建一个图和轴对象
    fig, ax = plt.subplots()
    # 生成随机颜色列表，并绘制散点图
    colors = np.random.default_rng().choice(['r', 'g', 'b'], 10)
    ax.scatter(np.arange(10), np.arange(10)*1, label='#1', facecolor=colors)

    # 添加图例，使得标签颜色与散点面部颜色相同
    leg = ax.legend(labelcolor='markerfacecolor')
    # 遍历图例中的文本和颜色，断言它们的颜色与预期的一致
    for text, color in zip(leg.get_texts(), ['k']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_pathcollection_labelcolor_markfacecolor_cmap():
    # 测试带有 colormap 的 PathCollection 上 labelcolor='markerfacecolor' 的标签颜色
    # 创建一个图和轴对象
    fig, ax = plt.subplots()
    # 使用 viridis colormap 生成颜色，并绘制散点图
    facecolors = mpl.cm.viridis(np.random.rand(10))
    ax.scatter(
        np.arange(10),
        np.arange(10),
        label='#1',
        c=np.arange(10),
        facecolor=facecolors
    )

    # 添加图例，使得标签颜色与散点面部颜色相同
    leg = ax.legend(labelcolor='markerfacecolor')
    # 遍历图例中的文本和颜色，断言它们的颜色与预期的一致
    for text, color in zip(leg.get_texts(), ['k']):
        assert mpl.colors.same_color(text.get_color(), color)


@pytest.mark.parametrize('color', ('red', 'none', (.5, .5, .5)))
def test_legend_labelcolor_rcparam_single(color):
    # 测试单一颜色的 rcParams legend.labelcolor 设置
    # 创建一个图和轴对象
    fig, ax = plt.subplots()
    # 绘制三条线，每条线有不同的标签
    ax.plot(np.arange(10), np.arange(10)*1, label='#1')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3')

    # 设置 rcParams 中 legend.labelcolor 的颜色
    mpl.rcParams['legend.labelcolor'] = color
    # 添加图例
    leg = ax.legend()
    # 遍历图例中的文本，断言它们的颜色与预期的一致
    for text in leg.get_texts():
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_labelcolor_rcparam_linecolor():
    # 测试 rcParams 中 legend.labelcolor 设置为 linecolor 的情况
    # 创建一个图和轴对象
    fig, ax = plt.subplots()
    # 绘制三条线，每条线有不同的颜色和标签
    ax.plot(np.arange(10), np.arange(10)*1, label='#1', color='r')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2', color='g')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3', color='b')

    # 设置 rcParams 中 legend.labelcolor 的颜色为 linecolor
    mpl.rcParams['legend.labelcolor'] = 'linecolor'
    # 添加图例
    leg = ax.legend()
    # 遍历图例中的文本和颜色，断言它们的颜色与预期的一致
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


def test_legend_labelcolor_rcparam_markeredgecolor():
    # 测试 labelcolor='markeredgecolor' 的标签颜色设置
    # 创建一个图和轴对象
    fig, ax = plt.subplots()
    # 绘制三条线，每条线有不同的标签和标记边缘颜色
    ax.plot(np.arange(10), np.arange(10)*1, label='#1', markeredgecolor='r')
    # 在图形 ax 上绘制折线图，横坐标为从0到9的整数序列，纵坐标为对应的整数序列乘以2，设置图例标签为'#2'，标记边缘颜色为绿色
    ax.plot(np.arange(10), np.arange(10)*2, label='#2', markeredgecolor='g')
    
    # 在图形 ax 上绘制折线图，横坐标为从0到9的整数序列，纵坐标为对应的整数序列乘以3，设置图例标签为'#3'，标记边缘颜色为蓝色
    ax.plot(np.arange(10), np.arange(10)*3, label='#3', markeredgecolor='b')
    
    # 设置全局参数，将图例标签的文本颜色设为'markeredgecolor'，实际效果为设置为字符串 'markeredgecolor'
    mpl.rcParams['legend.labelcolor'] = 'markeredgecolor'
    
    # 在图形 ax 上创建图例对象
    leg = ax.legend()
    
    # 遍历图例中的文本对象和指定的颜色列表 ['r', 'g', 'b']
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        # 断言每个文本对象的颜色与列表中对应的颜色相同，用于验证颜色设置是否正确
        assert mpl.colors.same_color(text.get_color(), color)
# 测试 legend.labelcolor 参数在 labelcolor='markeredgecolor' 时的效果
def test_legend_labelcolor_rcparam_markeredgecolor_short():
    # 创建一个图形和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制三条折线图，并指定每条线的标签和 markeredgecolor 颜色
    ax.plot(np.arange(10), np.arange(10)*1, label='#1', markeredgecolor='r')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2', markeredgecolor='g')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3', markeredgecolor='b')

    # 设置全局参数 legend.labelcolor 为 'mec'，即使用 markeredgecolor 作为标签颜色
    mpl.rcParams['legend.labelcolor'] = 'mec'
    # 创建图例对象
    leg = ax.legend()
    # 遍历图例中的文本和预期颜色，确保它们相同
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


# 测试 legend.labelcolor 参数在 labelcolor='markerfacecolor' 时的效果
def test_legend_labelcolor_rcparam_markerfacecolor():
    # 创建一个图形和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制三条折线图，并指定每条线的标签和 markerfacecolor 颜色
    ax.plot(np.arange(10), np.arange(10)*1, label='#1', markerfacecolor='r')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2', markerfacecolor='g')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3', markerfacecolor='b')

    # 设置全局参数 legend.labelcolor 为 'markerfacecolor'，即使用 markerfacecolor 作为标签颜色
    mpl.rcParams['legend.labelcolor'] = 'markerfacecolor'
    # 创建图例对象
    leg = ax.legend()
    # 遍历图例中的文本和预期颜色，确保它们相同
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


# 测试 legend.labelcolor 参数在 labelcolor='mfc' 时的效果
def test_legend_labelcolor_rcparam_markerfacecolor_short():
    # 创建一个图形和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制三条折线图，并指定每条线的标签和 markerfacecolor 颜色
    ax.plot(np.arange(10), np.arange(10)*1, label='#1', markerfacecolor='r')
    ax.plot(np.arange(10), np.arange(10)*2, label='#2', markerfacecolor='g')
    ax.plot(np.arange(10), np.arange(10)*3, label='#3', markerfacecolor='b')

    # 设置全局参数 legend.labelcolor 为 'mfc'，即使用 markerfacecolor 作为标签颜色
    mpl.rcParams['legend.labelcolor'] = 'mfc'
    # 创建图例对象
    leg = ax.legend()
    # 遍历图例中的文本和预期颜色，确保它们相同
    for text, color in zip(leg.get_texts(), ['r', 'g', 'b']):
        assert mpl.colors.same_color(text.get_color(), color)


# 测试 legend 对象的可拖动设置功能
@pytest.mark.filterwarnings("ignore:No artists with labels found to put in legend")
def test_get_set_draggable():
    # 创建一个默认的图例对象
    legend = plt.legend()
    # 断言图例对象不可拖动
    assert not legend.get_draggable()
    # 设置图例对象为可拖动
    legend.set_draggable(True)
    # 再次断言图例对象现在是可拖动的
    assert legend.get_draggable()
    # 将图例对象设置为不可拖动
    legend.set_draggable(False)
    # 断言图例对象现在不可拖动
    assert not legend.get_draggable()


# 测试 legend 对象的拖动属性设置
@pytest.mark.parametrize('draggable', (True, False))
def test_legend_draggable(draggable):
    # 创建一个图形和坐标轴对象
    fig, ax = plt.subplots()
    # 绘制一条折线，并添加图例，设置拖动属性为可变参数
    ax.plot(range(10), label='shabnams')
    leg = ax.legend(draggable=draggable)
    # 断言图例对象的拖动属性是否符合预期
    assert leg.get_draggable() is draggable


# 测试图例对象中各个句柄的透明度设置
def test_alpha_handles():
    # 绘制直方图，并设置透明度和标签
    x, n, hh = plt.hist([1, 2, 3], alpha=0.25, label='data', color='red')
    # 创建图例对象
    legend = plt.legend()
    # 遍历图例中的句柄，并设置它们的透明度为 1.0
    for lh in legend.legend_handles:
        lh.set_alpha(1.0)
    # 断言最后一个句柄的填充颜色与直方图的颜色一致
    assert lh.get_facecolor()[:-1] == hh[1].get_facecolor()[:-1]
    # 断言最后一个句柄的边框颜色与直方图的边框颜色一致
    assert lh.get_edgecolor()[:-1] == hh[1].get_edgecolor()[:-1]


# 测试使用 LaTeX 字体设置时的警告信息
@needs_usetex
def test_usetex_no_warn(caplog):
    # 设置字体相关的 rcParams 参数
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Computer Modern'
    mpl.rcParams['text.usetex'] = True

    # 创建一个图形和坐标轴对象，并绘制一条线，并添加带标题的图例
    fig, ax = plt.subplots()
    ax.plot(0, 0, label='input')
    ax.legend(title="My legend")

    # 绘制图形并捕获日志
    fig.canvas.draw()
    # 断言日志中不包含特定的警告信息
    assert "Font family ['serif'] not found." not in caplog.text
def test_warn_big_data_best_loc(monkeypatch):
    # 强制 _find_best_position 方法认为花费了很长时间
    counter = itertools.count(0, step=1.5)
    monkeypatch.setattr(time, 'perf_counter', lambda: next(counter))

    # 创建图形和轴对象
    fig, ax = plt.subplots()
    fig.canvas.draw()  # 以便稍后调用 draw_artist

    # 在所有可能的图例位置上放置一条线
    x = [0.9, 0.1, 0.1, 0.9, 0.9, 0.5]
    y = [0.95, 0.95, 0.05, 0.05, 0.5, 0.5]
    ax.plot(x, y, 'o-', label='line')

    # 设置图例位置为 "best"，在这个上下文中，可能会出现警告信息
    with rc_context({'legend.loc': 'best'}):
        legend = ax.legend()

    # 使用 pytest 的 warns 方法检查是否有 UserWarning 被触发
    with pytest.warns(UserWarning,
                      match='Creating legend with loc="best" can be slow with large '
                      'amounts of data.') as records:
        fig.draw_artist(legend)  # 不必绘制线条，因为这很慢
    # _find_best_position 方法被调用两次，因此警告消息会重复出现
    assert len(records) == 2


def test_no_warn_big_data_when_loc_specified(monkeypatch):
    # 强制 _find_best_position 方法认为花费了很长时间
    counter = itertools.count(0, step=1.5)
    monkeypatch.setattr(time, 'perf_counter', lambda: next(counter))

    # 创建图形和轴对象
    fig, ax = plt.subplots()
    fig.canvas.draw()

    # 在所有可能的图例位置上放置一条线
    x = [0.9, 0.1, 0.1, 0.9, 0.9, 0.5]
    y = [0.95, 0.95, 0.05, 0.05, 0.5, 0.5]
    ax.plot(x, y, 'o-', label='line')

    # 明确指定图例位置为 "best"，不应触发警告
    legend = ax.legend('best')
    fig.draw_artist(legend)  # 检查是否没有发出警告


@pytest.mark.parametrize('label_array', [['low', 'high'],
                                         ('low', 'high'),
                                         np.array(['low', 'high'])])
def test_plot_multiple_input_multiple_label(label_array):
    # 测试 ax.plot() 多维输入和多个标签
    x = [1, 2, 3]
    y = [[1, 2],
         [2, 5],
         [4, 9]]

    # 创建图形和轴对象
    fig, ax = plt.subplots()
    ax.plot(x, y, label=label_array)
    leg = ax.legend()
    legend_texts = [entry.get_text() for entry in leg.get_texts()]
    assert legend_texts == ['low', 'high']


@pytest.mark.parametrize('label', ['one', 1, int])
def test_plot_multiple_input_single_label(label):
    # 测试 ax.plot() 多维输入和单个标签
    x = [1, 2, 3]
    y = [[1, 2],
         [2, 5],
         [4, 9]]

    # 创建图形和轴对象
    fig, ax = plt.subplots()
    ax.plot(x, y, label=label)
    leg = ax.legend()
    legend_texts = [entry.get_text() for entry in leg.get_texts()]
    assert legend_texts == [str(label)] * 2


@pytest.mark.parametrize('label_array', [['low', 'high'],
                                         ('low', 'high'),
                                         np.array(['low', 'high'])])
def test_plot_single_input_multiple_label(label_array):
    # 测试 ax.plot() 一维数组输入和可迭代标签
    x = [1, 2, 3]
    y = [2, 5, 6]

    # 创建图形和轴对象
    fig, ax = plt.subplots()
    # 使用 pytest 模块捕获 matplotlib 中的 MatplotlibDeprecationWarning 警告，并匹配特定的警告信息字符串
    with pytest.warns(mpl.MatplotlibDeprecationWarning,
                      match='Passing label as a length 2 sequence'):
        # 在图形 ax 上绘制曲线，使用 label_array 作为标签数组
        ax.plot(x, y, label=label_array)
    
    # 添加图例到图形 ax，并将结果保存在 leg 变量中
    leg = ax.legend()
    
    # 断言图例中文本的数量为 1
    assert len(leg.get_texts()) == 1
    
    # 断言图例中第一个文本的内容与 label_array 转换为字符串后的内容相等
    assert leg.get_texts()[0].get_text() == str(label_array)
def test_plot_single_input_list_label():
    # 创建一个包含空图形和轴的图形对象
    fig, ax = plt.subplots()
    # 绘制一条线，输入是一个包含两个单点列表的列表，标签为单个字符串列表
    line, = ax.plot([[0], [1]], label=['A'])
    # 断言线条对象的标签是否为'A'
    assert line.get_label() == 'A'


def test_plot_multiple_label_incorrect_length_exception():
    # 检查是否在给定多个标签但行数不匹配时引发异常
    with pytest.raises(ValueError):
        x = [1, 2, 3]
        y = [[1, 2],
             [2, 5],
             [4, 9]]
        label = ['high', 'low', 'medium']
        # 创建一个包含图形和轴的图形对象
        fig, ax = plt.subplots()
        # 绘制多条线，使用给定的标签
        ax.plot(x, y, label=label)


def test_legend_face_edgecolor():
    # 测试 PolyCollection 图例处理程序的 'face' 边缘颜色
    fig, ax = plt.subplots()
    # 填充一个区域，并设置面部颜色和边缘颜色
    ax.fill_between([0, 1, 2], [1, 2, 3], [2, 3, 4],
                    facecolor='r', edgecolor='face', label='Fill')
    # 添加图例
    ax.legend()


def test_legend_text_axes():
    # 创建一个包含图形和轴的图形对象
    fig, ax = plt.subplots()
    # 绘制一条线，并为其设置标签
    ax.plot([1, 2], [3, 4], label='line')
    # 获取图例对象
    leg = ax.legend()
    # 断言图例的轴与当前轴相同
    assert leg.axes is ax
    # 断言图例的第一个文本对象的轴与当前轴相同
    assert leg.get_texts()[0].axes is ax


def test_handlerline2d():
    # 测试一致性标记，用于整体 Line2D 图例处理程序
    fig, ax = plt.subplots()
    # 散点图
    ax.scatter([0, 1], [0, 1], marker="v")
    # 创建包含一个标记的 Line2D 对象列表
    handles = [mlines.Line2D([0], [0], marker="v")]
    # 添加图例，并验证标记的一致性
    leg = ax.legend(handles, ["Aardvark"], numpoints=1)
    assert handles[0].get_marker() == leg.legend_handles[0].get_marker()


def test_subfigure_legend():
    # 测试图例能否添加到子图
    subfig = plt.figure().subfigures()
    # 获取子图的轴对象
    ax = subfig.subplots()
    # 绘制一条线，并为其设置标签
    ax.plot([0, 1], [0, 1], label="line")
    # 添加图例
    leg = subfig.legend()
    # 断言图例所属的图形对象与当前子图相同
    assert leg.figure is subfig


def test_setting_alpha_keeps_polycollection_color():
    # 测试设置 alpha 保持 PolyCollection 颜色
    # 填充一个区域，并设置其颜色和标签
    pc = plt.fill_between([0, 1], [2, 3], color='#123456', label='label')
    # 获取图例并设置其 alpha
    patch = plt.legend().get_patches()[0]
    patch.set_alpha(0.5)
    # 断言图例的面部颜色与填充区域的面部颜色相同
    assert patch.get_facecolor()[:3] == tuple(pc.get_facecolor()[0][:3])
    # 断言图例的边缘颜色与填充区域的边缘颜色相同
    assert patch.get_edgecolor()[:3] == tuple(pc.get_edgecolor()[0][:3])


def test_legend_markers_from_line2d():
    # 测试从 Line2D 复制标记到图例线条
    _markers = ['.', '*', 'v']
    fig, ax = plt.subplots()
    # 创建包含不可见线和指定标记的 Line2D 对象列表
    lines = [mlines.Line2D([0], [0], ls='None', marker=mark)
             for mark in _markers]
    labels = ["foo", "bar", "xyzzy"]
    # 添加图例，并验证标记的一致性
    legend = ax.legend(lines, labels)

    # 获取图例中线条的标记和标签
    new_markers = [line.get_marker() for line in legend.get_lines()]
    new_labels = [text.get_text() for text in legend.get_texts()]

    # 断言原始标记、图例中线条的标记和标签相同
    assert markers == new_markers == _markers
    assert labels == new_labels


@check_figures_equal()
def test_ncol_ncols(fig_test, fig_ref):
    # 测试 ncol 和 ncols 的工作性能
    strings = ["a", "b", "c", "d", "e", "f"]
    ncols = 3
    # 添加图例，并使用指定的列数
    fig_test.legend(strings, ncol=ncols)
    fig_ref.legend(strings, ncols=ncols)


def test_loc_invalid_tuple_exception():
    # 检查当图例的 loc 参数不是包含两个数字的元组时是否引发异常
    # 创建一个包含图形和轴的图形对象
    fig, ax = plt.subplots()
    # 使用 pytest 模块来测试异常情况，预期会抛出 ValueError 异常，并检查匹配的错误消息
    with pytest.raises(ValueError, match=('loc must be string, coordinate '
                       'tuple, or an integer 0-10, not \\(1.1,\\)')):
        # 在图形对象 ax 上添加图例，指定位置参数为 (1.1,)，标签为 ["mock data"]
        ax.legend(loc=(1.1, ), labels=["mock data"])

    # 使用 pytest 模块来测试异常情况，预期会抛出 ValueError 异常，并检查匹配的错误消息
    with pytest.raises(ValueError, match=('loc must be string, coordinate '
                       'tuple, or an integer 0-10, not \\(0.481, 0.4227, 0.4523\\)')):
        # 在图形对象 ax 上添加图例，指定位置参数为 (0.481, 0.4227, 0.4523)，标签为 ["mock data"]
        ax.legend(loc=(0.481, 0.4227, 0.4523), labels=["mock data"])

    # 使用 pytest 模块来测试异常情况，预期会抛出 ValueError 异常，并检查匹配的错误消息
    with pytest.raises(ValueError, match=('loc must be string, coordinate '
                       'tuple, or an integer 0-10, not \\(0.481, \'go blue\'\\)')):
        # 在图形对象 ax 上添加图例，指定位置参数为 (0.481, "go blue")，标签为 ["mock data"]
        ax.legend(loc=(0.481, "go blue"), labels=["mock data"])
# 测试有效的位置参数为元组
def test_loc_valid_tuple():
    # 创建一个包含图形和轴的对象
    fig, ax = plt.subplots()
    # 添加图例到轴上，位置为 (0.481, 0.442)，标签为 ["mock data"]
    ax.legend(loc=(0.481, 0.442), labels=["mock data"])
    # 添加图例到轴上，位置为 (1, 2)，标签为 ["mock data"]
    ax.legend(loc=(1, 2), labels=["mock data"])


# 测试有效的位置参数为列表（不推荐，但可以）
def test_loc_valid_list():
    # 创建一个包含图形和轴的对象
    fig, ax = plt.subplots()
    # 添加图例到轴上，位置为 [0.481, 0.442]，标签为 ["mock data"]
    ax.legend(loc=[0.481, 0.442], labels=["mock data"])
    # 添加图例到轴上，位置为 [1, 2]，标签为 ["mock data"]
    ax.legend(loc=[1, 2], labels=["mock data"])


# 测试无效的位置参数为列表，应引发异常
def test_loc_invalid_list_exception():
    # 创建一个包含图形和轴的对象
    fig, ax = plt.subplots()
    # 使用 pytest 检查是否引发 ValueError 异常，匹配特定错误消息
    with pytest.raises(ValueError, match=('loc must be string, coordinate '
                       'tuple, or an integer 0-10, not \\[1.1, 2.2, 3.3\\]')):
        # 尝试添加图例到轴上，位置为 [1.1, 2.2, 3.3]，标签为 ["mock data"]
        ax.legend(loc=[1.1, 2.2, 3.3], labels=["mock data"])


# 测试无效的位置参数类型，应引发异常
def test_loc_invalid_type():
    # 创建一个包含图形和轴的对象
    fig, ax = plt.subplots()
    # 使用 pytest 检查是否引发 ValueError 异常，匹配特定错误消息
    with pytest.raises(ValueError, match=("loc must be string, coordinate "
                       "tuple, or an integer 0-10, not {'not': True}")):
        # 尝试添加图例到轴上，位置为 {'not': True}，标签为 ["mock data"]
        ax.legend(loc={'not': True}, labels=["mock data"])


# 测试位置参数为数值类型的有效性检查
def test_loc_validation_numeric_value():
    # 创建一个包含图形和轴的对象
    fig, ax = plt.subplots()
    # 添加图例到轴上，位置为 0，标签为 ["mock data"]
    ax.legend(loc=0, labels=["mock data"])
    # 添加图例到轴上，位置为 1，标签为 ["mock data"]
    ax.legend(loc=1, labels=["mock data"])
    # 添加图例到轴上，位置为 5，标签为 ["mock data"]
    ax.legend(loc=5, labels=["mock data"])
    # 添加图例到轴上，位置为 10，标签为 ["mock data"]
    ax.legend(loc=10, labels=["mock data"])
    # 使用 pytest 检查是否引发 ValueError 异常，匹配特定错误消息
    with pytest.raises(ValueError, match=('loc must be string, coordinate '
                       'tuple, or an integer 0-10, not 11')):
        # 尝试添加图例到轴上，位置为 11，标签为 ["mock data"]
        ax.legend(loc=11, labels=["mock data"])
    # 使用 pytest 检查是否引发 ValueError 异常，匹配特定错误消息
    with pytest.raises(ValueError, match=('loc must be string, coordinate '
                       'tuple, or an integer 0-10, not -1')):
        # 尝试添加图例到轴上，位置为 -1，标签为 ["mock data"]
        ax.legend(loc=-1, labels=["mock data"])


# 测试位置参数为字符串类型的有效性检查
def test_loc_validation_string_value():
    # 创建一个包含图形和轴的对象
    fig, ax = plt.subplots()
    labels = ["mock data"]
    # 添加图例到轴上，位置为 'best'，标签为 ["mock data"]
    ax.legend(loc='best', labels=labels)
    # 添加图例到轴上，位置为 'upper right'，标签为 ["mock data"]
    ax.legend(loc='upper right', labels=labels)
    # 添加图例到轴上，位置为 'best'，标签为 ["mock data"]
    ax.legend(loc='best', labels=labels)
    # 添加图例到轴上，位置为 'upper right'，标签为 ["mock data"]
    ax.legend(loc='upper right', labels=labels)
    # 添加图例到轴上，位置为 'upper left'，标签为 ["mock data"]
    ax.legend(loc='upper left', labels=labels)
    # 添加图例到轴上，位置为 'lower left'，标签为 ["mock data"]
    ax.legend(loc='lower left', labels=labels)
    # 添加图例到轴上，位置为 'lower right'，标签为 ["mock data"]
    ax.legend(loc='lower right', labels=labels)
    # 添加图例到轴上，位置为 'right'，标签为 ["mock data"]
    ax.legend(loc='right', labels=labels)
    # 添加图例到轴上，位置为 'center left'，标签为 ["mock data"]
    ax.legend(loc='center left', labels=labels)
    # 添加图例到轴上，位置为 'center right'，标签为 ["mock data"]
    ax.legend(loc='center right', labels=labels)
    # 添加图例到轴上，位置为 'lower center'，标签为 ["mock data"]
    ax.legend(loc='lower center', labels=labels)
    # 添加图例到轴上，位置为 'upper center'，标签为 ["mock data"]
    ax.legend(loc='upper center', labels=labels)
    # 使用 pytest 检查是否引发 ValueError 异常，匹配特定错误消息
    with pytest.raises(ValueError, match="'wrong' is not a valid value for"):
        # 尝试添加图例到轴上，位置为 'wrong'，标签为 ["mock data"]
        ax.legend(loc='wrong', labels=labels)


# 测试图例句柄和标签不匹配时的警告
def test_legend_handle_label_mismatch():
    # 创建一个包含图形和轴的对象，并绘制两条曲线
    pl1, = plt.plot(range(10))
    pl2, = plt.plot(range(10))
    # 使用 pytest 检查是否产生 UserWarning 警告，匹配特定警告消息
    with pytest.warns(UserWarning, match="number of handles and labels"):
        # 创建图例，指定句柄和标签
        legend = plt.legend(handles=[pl1, pl2], labels=["pl1", "pl2", "pl3"])
        # 断言图例句柄数量为 2
        assert len(legend.legend_handles) == 2
        # 断言图例文本数量为 2
        assert len(legend.get_texts()) == 2


# 测试图例句柄和标签不匹配时的警告，使用迭代器而不是 len()
def test_legend_handle_label_mismatch_no_len():
    # 创建一个包含图形和轴的对象，并绘制两条曲线
    pl1, = plt.plot(range(10))
    pl2, = plt.plot(range(10))
    # 创建图例，句柄和标签使用迭代器
    legend = plt.legend(handles=iter([pl1, pl2]),
                        labels=iter(["pl1", "pl2", "pl3"]))
    # 断言图例句柄数量为 2
    assert len(legend.legend_handles) == 2
    # 断言图例文本数量为 2
    assert len(legend.get_texts()) == 2


# 测试无标签时是否产生警告
def test_legend_nolabels_warning():
    # 绘制一条曲线
    plt.plot([1, 2, 3])
    # 使用 pytest 的上下文管理器来检测是否抛出 UserWarning 异常，并且异常消息需要匹配 "No artists with labels found"
    with pytest.raises(UserWarning, match="No artists with labels found"):
        # 调用 matplotlib.pyplot 的 legend() 函数来生成图例
        plt.legend()
@pytest.mark.filterwarnings("ignore:No artists with labels found to put in legend")
# 用于标记测试，忽略警告消息，确保在没有带标签的艺术品时不显示图例

def test_legend_nolabels_draw():
    # 绘制简单的折线图
    plt.plot([1, 2, 3])
    # 添加图例
    plt.legend()
    # 断言当前轴存在图例对象
    assert plt.gca().get_legend() is not None


def test_legend_loc_polycollection():
    # 测试对于多边形集合 'best' 位置的图例放置是否正确
    x = [3, 4, 5]
    y1 = [1, 1, 1]
    y2 = [5, 5, 5]
    leg_bboxes = []
    # 创建包含两个子图的图形对象
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    for ax, loc in zip(axs.flat, ('best', 'lower left')):
        # 填充多边形区域并添加标签
        ax.fill_between(x, y1, y2, color='gray', alpha=0.5, label='Shaded Area')
        # 设置子图的 X 和 Y 轴范围
        ax.set_xlim(0, 6)
        ax.set_ylim(-1, 5)
        # 添加图例并获取图例对象
        leg = ax.legend(loc=loc)
        # 绘制图形
        fig.canvas.draw()
        # 将图例框的边界信息添加到列表中
        leg_bboxes.append(
            leg.get_window_extent().transformed(ax.transAxes.inverted()))
    # 断言两个子图的图例框边界相同
    assert_allclose(leg_bboxes[1].bounds, leg_bboxes[0].bounds)


def test_legend_text():
    # 测试当图中有文本时 'best' 位置的图例放置是否正确
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    leg_bboxes = []
    for ax, loc in zip(axs.flat, ('best', 'lower left')):
        x = [1, 2]
        y = [2, 1]
        # 绘制折线图并添加标签
        ax.plot(x, y, label='plot name')
        # 在图中添加文本
        ax.text(1.5, 2, 'some text blahblah', verticalalignment='top')
        # 添加图例并获取图例对象
        leg = ax.legend(loc=loc)
        # 绘制图形
        fig.canvas.draw()
        # 将图例框的边界信息添加到列表中
        leg_bboxes.append(
            leg.get_window_extent().transformed(ax.transAxes.inverted()))
    # 断言两个子图的图例框边界相同
    assert_allclose(leg_bboxes[1].bounds, leg_bboxes[0].bounds)


def test_boxplot_legend_labels():
    # 测试当传递 `label` 参数时是否生成图例条目
    np.random.seed(19680801)
    data = np.random.random((10, 4))
    fig, axs = plt.subplots(nrows=1, ncols=4)
    legend_labels = ['box A', 'box B', 'box C', 'box D']

    # 测试图例标签和传递给图例的补丁
    bp1 = axs[0].boxplot(data, patch_artist=True, label=legend_labels)
    # 断言盒图的箱子的标签与预期标签相同
    assert [v.get_label() for v in bp1['boxes']] == legend_labels
    handles, labels = axs[0].get_legend_handles_labels()
    # 断言图例的标签与预期标签相同
    assert labels == legend_labels
    # 断言所有图例的处理对象都是 PathPatch 类型
    assert all(isinstance(h, mpl.patches.PathPatch) for h in handles)

    # 测试不带 `box` 的图例
    bp2 = axs[1].boxplot(data, label=legend_labels, showbox=False)
    # 没有箱子时，图例条目应从中位数传递
    assert [v.get_label() for v in bp2['medians']] == legend_labels
    handles, labels = axs[1].get_legend_handles_labels()
    # 断言图例的标签与预期标签相同
    assert labels == legend_labels
    # 断言所有图例的处理对象都是 Line2D 类型
    assert all(isinstance(h, mpl.lines.Line2D) for h in handles)

    # 测试标签数量与箱子数量不同时的图例
    with pytest.raises(ValueError, match='values must have same the length'):
        bp3 = axs[2].boxplot(data, label=legend_labels[:-1])

    # 测试当传递字符串标签时，只有第一个箱子获得标签
    bp4 = axs[3].boxplot(data, label='box A')
    # 断言第一个箱子的中位数标签为 'box A'
    assert bp4['medians'][0].get_label() == 'box A'
    # 断言语句：验证所有的中位数对象标签是否以下划线开头
    assert all(x.get_label().startswith("_") for x in bp4['medians'][1:])
```