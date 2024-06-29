# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_subplots.py`

```py
import itertools  # 导入 itertools 模块，用于生成迭代器的工具函数
import platform  # 导入 platform 模块，用于访问平台相关的系统信息

import numpy as np  # 导入 NumPy 库，用于支持大量的数学函数和数据结构
import pytest  # 导入 pytest 库，用于编写简单而有效的单元测试

from matplotlib.axes import Axes, SubplotBase  # 从 matplotlib.axes 模块导入 Axes 和 SubplotBase 类
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块并重命名为 plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison  # 从 matplotlib.testing.decorators 模块导入函数装饰器


def check_shared(axs, x_shared, y_shared):
    """
    x_shared and y_shared are n x n boolean matrices; entry (i, j) indicates
    whether the x (or y) axes of subplots i and j should be shared.
    """
    # 使用 itertools.product 生成 axs, x_shared 和 y_shared 的组合迭代器
    for (i1, ax1), (i2, ax2), (i3, (name, shared)) in itertools.product(
            enumerate(axs),
            enumerate(axs),
            enumerate(zip("xy", [x_shared, y_shared]))):
        if i2 <= i1:
            continue
        # 断言 axs[0]._shared_axes[name].joined(ax1, ax2) 是否等于 shared[i1, i2]
        assert axs[0]._shared_axes[name].joined(ax1, ax2) == shared[i1, i2], \
            "axes %i and %i incorrectly %ssharing %s axis" % (
                i1, i2, "not " if shared[i1, i2] else "", name)


def check_ticklabel_visible(axs, x_visible, y_visible):
    """Check that the x and y ticklabel visibility is as specified."""
    # 遍历 axs，以及 x_visible 和 y_visible 的元素
    for i, (ax, vx, vy) in enumerate(zip(axs, x_visible, y_visible)):
        # 检查 x 轴的刻度标签及其可见性
        for l in ax.get_xticklabels() + [ax.xaxis.offsetText]:
            assert l.get_visible() == vx, \
                    f"Visibility of x axis #{i} is incorrectly {vx}"
        # 检查 y 轴的刻度标签及其可见性
        for l in ax.get_yticklabels() + [ax.yaxis.offsetText]:
            assert l.get_visible() == vy, \
                    f"Visibility of y axis #{i} is incorrectly {vy}"
        # 如果 vx 为 False，则检查 x 轴的标签是否为空
        if not vx:
            assert ax.get_xlabel() == ""
        # 如果 vy 为 False，则检查 y 轴的标签是否为空
        if not vy:
            assert ax.get_ylabel() == ""


def check_tick1_visible(axs, x_visible, y_visible):
    """
    Check that the x and y tick visibility is as specified.

    Note: This only checks the tick1line, i.e. bottom / left ticks.
    """
    # 检查 x 和 y 轴的 tick1line（底部/左侧刻度线）可见性
    for ax, visible, in zip(axs, x_visible):
        for tick in ax.xaxis.get_major_ticks():
            assert tick.tick1line.get_visible() == visible
    for ax, y_visible, in zip(axs, y_visible):
        for tick in ax.yaxis.get_major_ticks():
            assert tick.tick1line.get_visible() == visible


def test_shared():
    rdim = (4, 4, 2)  # 定义元组 rdim，包含 4、4、2 三个值
    share = {
            'all': np.ones(rdim[:2], dtype=bool),  # 创建全为 True 的布尔矩阵
            'none': np.zeros(rdim[:2], dtype=bool),  # 创建全为 False 的布尔矩阵
            'row': np.array([  # 创建指定的布尔矩阵
                [False, True, False, False],
                [True, False, False, False],
                [False, False, False, True],
                [False, False, True, False]]),
            'col': np.array([  # 创建指定的布尔矩阵
                [False, False, True, False],
                [False, False, False, True],
                [True, False, False, False],
                [False, True, False, False]]),
            }
    visible = {
            'x': {
                'all': [False, False, True, True],  # x轴所有标签是否可见的设置：左上、右上、左下、右下
                'col': [False, False, True, True],  # x轴列标签是否可见的设置：左上、右上、左下、右下
                'row': [True] * 4,  # x轴行标签是否可见的设置：全部显示
                'none': [True] * 4,  # x轴无标签时是否可见的设置：全部显示
                False: [True] * 4,  # x轴False时是否可见的设置：全部显示
                True: [False, False, True, True],  # x轴True时是否可见的设置：左下、右下
                },
            'y': {
                'all': [True, False, True, False],  # y轴所有标签是否可见的设置：左上、右上、左下、右下
                'col': [True] * 4,  # y轴列标签是否可见的设置：全部显示
                'row': [True, False, True, False],  # y轴行标签是否可见的设置：左上、左下、右上、右下
                'none': [True] * 4,  # y轴无标签时是否可见的设置：全部显示
                False: [True] * 4,  # y轴False时是否可见的设置：全部显示
                True: [True, False, True, False],  # y轴True时是否可见的设置：左上、左下、右上、右下
                },
            }
    share[False] = share['none']  # 共享设置中False的映射到'none'
    share[True] = share['all']  # 共享设置中True的映射到'all'

    # test default
    f, ((a1, a2), (a3, a4)) = plt.subplots(2, 2)  # 创建一个2x2的子图
    axs = [a1, a2, a3, a4]  # 将子图保存到列表中
    check_shared(axs, share['none'], share['none'])  # 检查共享设置为'none'的子图
    plt.close(f)  # 关闭图形对象，释放资源

    # test all option combinations
    ops = [False, True, 'all', 'none', 'row', 'col', 0, 1]  # 测试的所有选项组合
    for xo in ops:
        for yo in ops:
            f, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, sharex=xo, sharey=yo)  # 创建共享x轴和y轴的2x2子图
            axs = [a1, a2, a3, a4]  # 将子图保存到列表中
            check_shared(axs, share[xo], share[yo])  # 检查共享设置为xo和yo的子图
            check_ticklabel_visible(axs, visible['x'][xo], visible['y'][yo])  # 检查刻度标签的可见性设置
            plt.close(f)  # 关闭图形对象，释放资源
@pytest.mark.parametrize('remove_ticks', [True, False])
# 使用 pytest 的参数化装饰器，为 test_label_outer 函数提供两种不同的 remove_ticks 参数值
def test_label_outer(remove_ticks):
    # 创建一个包含 2x2 子图的图像 f 和对应的轴 axs，共享 x 和 y 轴
    f, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    # 遍历每个子图的轴对象
    for ax in axs.flat:
        # 设置每个轴对象的 x 标签为 "foo"，y 标签为 "bar"
        ax.set(xlabel="foo", ylabel="bar")
        # 调用 label_outer 方法，可能传入 remove_inner_ticks 参数来控制是否移除内部刻度线
        ax.label_outer(remove_inner_ticks=remove_ticks)
    # 检查轴对象中标签是否可见，期望的结果分别是 [False, False, True, True] 和 [True, False, True, False]
    check_ticklabel_visible(
        axs.flat, [False, False, True, True], [True, False, True, False])
    # 根据 remove_ticks 参数检查轴对象中 tick1 是否可见
    if remove_ticks:
        check_tick1_visible(
            axs.flat, [False, False, True, True], [True, False, True, False])
    else:
        check_tick1_visible(
            axs.flat, [True, True, True, True], [True, True, True, True])


def test_label_outer_span():
    # 创建一个新的图像对象
    fig = plt.figure()
    # 使用 gridspec 在图像上添加 3x3 的网格布局
    gs = fig.add_gridspec(3, 3)
    # 将子图添加到不同的网格位置，形成自定义布局
    a1 = fig.add_subplot(gs[0, 0:2])  # 占据第一行第一列到第一行第二列
    a2 = fig.add_subplot(gs[1:3, 0])   # 占据第二行到第三行第一列
    a3 = fig.add_subplot(gs[1, 2])     # 占据第二行第三列
    a4 = fig.add_subplot(gs[2, 1])     # 占据第三行第二列
    # 对图像中所有轴对象调用 label_outer 方法
    for ax in fig.axes:
        ax.label_outer()
    # 检查图像中所有轴对象的 ticklabel 是否可见，期望结果为 [False, True, False, True] 和 [True, True, False, False]
    check_ticklabel_visible(
        fig.axes, [False, True, False, True], [True, True, False, False])


def test_label_outer_non_gridspec():
    # 在图像上创建一个轴对象
    ax = plt.axes((0, 0, 1, 1))
    # 调用 label_outer 方法，这里不会产生效果
    ax.label_outer()  # Does nothing.
    # 检查轴对象的 ticklabel 是否可见，期望结果为 [True] 和 [True]
    check_ticklabel_visible([ax], [True], [True])


def test_shared_and_moved():
    # 创建包含两个子图的图像对象，共享 y 轴
    f, (a1, a2) = plt.subplots(1, 2, sharey=True)
    # 检查第二个子图的 ticklabel 是否可见，期望结果为 [True] 和 [False]
    check_ticklabel_visible([a2], [True], [False])
    # 调用 yaxis 的 tick_left 方法
    a2.yaxis.tick_left()
    # 再次检查第二个子图的 ticklabel 是否可见，期望结果为 [True] 和 [False]
    check_ticklabel_visible([a2], [True], [False])

    # 创建包含两行一列的子图布局，共享 x 轴
    f, (a1, a2) = plt.subplots(2, 1, sharex=True)
    # 检查第一个子图的 ticklabel 是否可见，期望结果为 [False] 和 [True]
    check_ticklabel_visible([a1], [False], [True])
    # 调用 xaxis 的 tick_bottom 方法
    a2.xaxis.tick_bottom()
    # 再次检查第一个子图的 ticklabel 是否可见，期望结果为 [False] 和 [True]
    check_ticklabel_visible([a1], [False], [True])


def test_exceptions():
    # 使用 pytest 的 raises 方法检查是否抛出 ValueError 异常
    with pytest.raises(ValueError):
        plt.subplots(2, 2, sharex='blah')
    with pytest.raises(ValueError):
        plt.subplots(2, 2, sharey='blah')


@image_comparison(['subplots_offset_text'],
                  tol=0.028 if platform.machine() == 'arm64' else 0)
# 使用 image_comparison 装饰器，比较生成的图像与参考图像是否相符
def test_subplots_offsettext():
    # 创建一组数据点
    x = np.arange(0, 1e10, 1e9)
    y = np.arange(0, 100, 10)+1e4
    # 创建一个包含 2x2 子图的图像对象，共享第一列的 x 轴和全部行的 y 轴
    fig, axs = plt.subplots(2, 2, sharex='col', sharey='all')
    # 在每个子图中绘制数据点
    axs[0, 0].plot(x, x)
    axs[1, 0].plot(x, x)
    axs[0, 1].plot(y, x)
    axs[1, 1].plot(y, x)


@pytest.mark.parametrize("top", [True, False])
@pytest.mark.parametrize("bottom", [True, False])
@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("right", [True, False])
# 使用 pytest 的参数化装饰器，为 test_subplots_hide_ticklabels 函数提供多组参数
def test_subplots_hide_ticklabels(top, bottom, left, right):
    # 理想情况下，测试隐藏刻度标签是否成功
    # 当前设置 rcParams 未能移动偏移文本，因此暂时保留 test_subplots_offsettext
    pass
    # 使用 plt.rc_context() 创建一个上下文管理器，用于临时设置 matplotlib 的刻度标签显示方式
    with plt.rc_context({"xtick.labeltop": top, "xtick.labelbottom": bottom,
                         "ytick.labelleft": left, "ytick.labelright": right}):
        # 创建一个包含3行3列子图的 figure，并共享 x 和 y 轴刻度
        axs = plt.figure().subplots(3, 3, sharex=True, sharey=True)
    # 使用 np.ndenumerate() 遍历 axs 中的子图及其索引
    for (i, j), ax in np.ndenumerate(axs):
        # 获取当前子图的 x 轴顶部刻度标签显示状态
        xtop = ax.xaxis._major_tick_kw["label2On"]
        # 获取当前子图的 x 轴底部刻度标签显示状态
        xbottom = ax.xaxis._major_tick_kw["label1On"]
        # 获取当前子图的 y 轴左侧刻度标签显示状态
        yleft = ax.yaxis._major_tick_kw["label1On"]
        # 获取当前子图的 y 轴右侧刻度标签显示状态
        yright = ax.yaxis._major_tick_kw["label2On"]
        # 使用 assert 断言验证各个刻度标签的显示状态是否符合预期
        assert xtop == (top and i == 0)
        assert xbottom == (bottom and i == 2)
        assert yleft == (left and j == 0)
        assert yright == (right and j == 2)
# 使用 pytest 模块的 mark.parametrize 装饰器为测试函数 test_subplots_hide_axislabels 提供参数化测试
@pytest.mark.parametrize("xlabel_position", ["bottom", "top"])
@pytest.mark.parametrize("ylabel_position", ["left", "right"])
def test_subplots_hide_axislabels(xlabel_position, ylabel_position):
    # 创建一个包含 3x3 子图的图形对象，子图共享 x 和 y 轴
    axs = plt.figure().subplots(3, 3, sharex=True, sharey=True)
    # 使用 np.ndenumerate 函数遍历 axs 中的子图
    for (i, j), ax in np.ndenumerate(axs):
        # 设置子图的 x 轴标签为 "foo"，y 轴标签为 "bar"
        ax.set(xlabel="foo", ylabel="bar")
        # 设置 x 轴标签的位置为指定位置
        ax.xaxis.set_label_position(xlabel_position)
        # 设置 y 轴标签的位置为指定位置
        ax.yaxis.set_label_position(ylabel_position)
        # 调用 label_outer 方法，根据标签是否覆盖其他部分决定是否隐藏子图的标签
        ax.label_outer()
        # 使用断言验证子图的 x 轴标签是否符合预期条件
        assert bool(ax.get_xlabel()) == (
            xlabel_position == "bottom" and i == 2
            or xlabel_position == "top" and i == 0)
        # 使用断言验证子图的 y 轴标签是否符合预期条件
        assert bool(ax.get_ylabel()) == (
            ylabel_position == "left" and j == 0
            or ylabel_position == "right" and j == 2)


# 定义测试函数 test_get_gridspec，测试 matplotlib 子图对象的 get_gridspec 方法
def test_get_gridspec():
    # 创建一个图形对象和一个子图对象
    fig, ax = plt.subplots()
    # 使用断言验证子图对象的子图规格对象与其自身的 gridspec 相等
    assert ax.get_subplotspec().get_gridspec() == ax.get_gridspec()


# 定义测试函数 test_dont_mutate_kwargs，测试 matplotlib 的子图创建时是否修改传入的参数字典
def test_dont_mutate_kwargs():
    # 定义 subplot_kw 和 gridspec_kw 两个字典作为子图参数和子图网格规格参数
    subplot_kw = {'sharex': 'all'}
    gridspec_kw = {'width_ratios': [1, 2]}
    # 创建一个包含 1 行 2 列子图的图形对象和子图对象
    fig, ax = plt.subplots(1, 2, subplot_kw=subplot_kw,
                           gridspec_kw=gridspec_kw)
    # 使用断言验证子图参数和网格规格参数未被修改
    assert subplot_kw == {'sharex': 'all'}
    assert gridspec_kw == {'width_ratios': [1, 2]}


# 使用 pytest 模块的 mark.parametrize 装饰器为测试函数 test_width_and_height_ratios 提供参数化测试
@pytest.mark.parametrize("width_ratios", [None, [1, 3, 2]])
@pytest.mark.parametrize("height_ratios", [None, [1, 2]])
@check_figures_equal(extensions=['png'])
def test_width_and_height_ratios(fig_test, fig_ref,
                                 height_ratios, width_ratios):
    # 在 fig_test 和 fig_ref 图形对象上创建一个 2x3 子图，指定高度和宽度比例
    fig_test.subplots(2, 3, height_ratios=height_ratios,
                      width_ratios=width_ratios)
    # 在 fig_ref 图形对象上创建一个 2x3 子图，指定高度和宽度比例作为网格规格参数
    fig_ref.subplots(2, 3, gridspec_kw={
                     'height_ratios': height_ratios,
                     'width_ratios': width_ratios})


# 使用 pytest 模块的 mark.parametrize 装饰器为测试函数 test_width_and_height_ratios_mosaic 提供参数化测试
@pytest.mark.parametrize("width_ratios", [None, [1, 3, 2]])
@pytest.mark.parametrize("height_ratios", [None, [1, 2]])
@check_figures_equal(extensions=['png'])
def test_width_and_height_ratios_mosaic(fig_test, fig_ref,
                                        height_ratios, width_ratios):
    # 定义一个子图网格布局的规格说明
    mosaic_spec = [['A', 'B', 'B'], ['A', 'C', 'D']]
    # 在 fig_test 图形对象上使用 mosaic_spec 和指定的高度和宽度比例创建子图
    fig_test.subplot_mosaic(mosaic_spec, height_ratios=height_ratios,
                            width_ratios=width_ratios)
    # 在 fig_ref 图形对象上使用 mosaic_spec 和指定的高度和宽度比例作为网格规格参数创建子图
    fig_ref.subplot_mosaic(mosaic_spec, gridspec_kw={
                           'height_ratios': height_ratios,
                           'width_ratios': width_ratios})


# 使用 pytest 模块的 mark.parametrize 装饰器为测试函数 test_ratio_overlapping_kws 提供参数化测试
@pytest.mark.parametrize('method,args', [
    ('subplots', (2, 3)),
    ('subplot_mosaic', ('abc;def', ))
    ]
)
def test_ratio_overlapping_kws(method, args):
    # 使用 pytest 的断言验证在创建子图时传入高度比例参数会引发 ValueError 异常
    with pytest.raises(ValueError, match='height_ratios'):
        getattr(plt, method)(*args, height_ratios=[1, 2],
                             gridspec_kw={'height_ratios': [1, 2]})
    # 使用 pytest 的断言验证在创建子图时传入宽度比例参数会引发 ValueError 异常
    with pytest.raises(ValueError, match='width_ratios'):
        getattr(plt, method)(*args, width_ratios=[1, 2, 3],
                             gridspec_kw={'width_ratios': [1, 2, 3]})


# 定义测试函数 test_old_subplot_compat，测试 matplotlib 的 figure 对象的 subplot 方法的兼容性
def test_old_subplot_compat():
    # 创建一个图形对象
    fig = plt.figure()
    # 使用 assert 语句检查 fig.add_subplot() 返回的对象是否是 SubplotBase 类的实例
    assert isinstance(fig.add_subplot(), SubplotBase)
    
    # 使用 assert 语句检查 fig.add_axes(rect=[0, 0, 1, 1]) 返回的对象是否不是 SubplotBase 类的实例
    assert not isinstance(fig.add_axes(rect=[0, 0, 1, 1]), SubplotBase)
    
    # 使用 pytest.raises 捕获 TypeError 异常，确保以下代码块会抛出 TypeError 异常
    with pytest.raises(TypeError):
        # 创建 Axes 对象时，传递一个无效的参数 rect=[0, 0, 1, 1]，这应该会引发 TypeError 异常
        Axes(fig, [0, 0, 1, 1], rect=[0, 0, 1, 1])
```