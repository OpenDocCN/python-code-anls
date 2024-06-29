# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_constrainedlayout.py`

```py
# 导入垃圾回收模块
import gc
# 导入平台信息模块
import platform

# 导入 numpy 库，并简称为 np
import numpy as np
# 导入 pytest 测试框架
import pytest

# 导入 matplotlib 库，并简称为 mpl
import matplotlib as mpl
# 从 matplotlib.testing.decorators 模块中导入 image_comparison 装饰器
from matplotlib.testing.decorators import image_comparison
# 导入 matplotlib.pyplot 库，并简称为 plt
import matplotlib.pyplot as plt
# 导入 matplotlib.transforms 模块，并简称为 mtransforms
import matplotlib.transforms as mtransforms
# 导入 matplotlib 的 gridspec 和 ticker 模块
from matplotlib import gridspec, ticker


def example_plot(ax, fontsize=12, nodec=False):
    # 在坐标轴 ax 上绘制一条简单的线图
    ax.plot([1, 2])
    # 设置刻度定位器参数为 3
    ax.locator_params(nbins=3)
    if not nodec:
        # 如果 nodec 参数为 False，则设置 x 和 y 轴的标签和标题，字体大小为 fontsize
        ax.set_xlabel('x-label', fontsize=fontsize)
        ax.set_ylabel('y-label', fontsize=fontsize)
        ax.set_title('Title', fontsize=fontsize)
    else:
        # 如果 nodec 参数为 True，则清空 x 和 y 轴的刻度标签
        ax.set_xticklabels([])
        ax.set_yticklabels([])


def example_pcolor(ax, fontsize=12):
    # 定义网格步长 dx 和 dy
    dx, dy = 0.6, 0.6
    # 创建二维网格坐标 y 和 x
    y, x = np.mgrid[slice(-3, 3 + dy, dy),
                    slice(-3, 3 + dx, dx)]
    # 定义一个二维函数 z，并计算其值
    z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
    # 在坐标轴 ax 上绘制伪彩色图，并返回对象 pcm
    pcm = ax.pcolormesh(x, y, z[:-1, :-1], cmap='RdBu_r', vmin=-1., vmax=1.,
                        rasterized=True)
    # 设置 x 和 y 轴的标签和标题，字体大小为 fontsize
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)
    return pcm


@image_comparison(['constrained_layout1.png'])
def test_constrained_layout1():
    """Test constrained_layout for a single subplot"""
    # 创建一个带有约束布局的图形对象
    fig = plt.figure(layout="constrained")
    # 在图形对象上添加一个子图
    ax = fig.add_subplot()
    # 调用 example_plot 函数在子图上进行绘制，设置字体大小为 24
    example_plot(ax, fontsize=24)


@image_comparison(['constrained_layout2.png'])
def test_constrained_layout2():
    """Test constrained_layout for 2x2 subplots"""
    # 创建一个带有约束布局的图形对象，并生成一个 2x2 的子图数组
    fig, axs = plt.subplots(2, 2, layout="constrained")
    # 遍历所有子图，并调用 example_plot 函数在每个子图上进行绘制，设置字体大小为 24
    for ax in axs.flat:
        example_plot(ax, fontsize=24)


@image_comparison(['constrained_layout3.png'])
def test_constrained_layout3():
    """Test constrained_layout for colorbars with subplots"""
    # 创建一个带有约束布局的图形对象，并生成一个 2x2 的子图数组
    fig, axs = plt.subplots(2, 2, layout="constrained")
    # 遍历所有子图，并调用 example_pcolor 函数在每个子图上进行绘制，设置字体大小为 24
    for nn, ax in enumerate(axs.flat):
        pcm = example_pcolor(ax, fontsize=24)
        # 根据子图序号 nn 设置 colorbar 的间距 pad
        if nn == 3:
            pad = 0.08
        else:
            pad = 0.02  # 默认值
        # 在图形对象上添加 colorbar，与当前子图 ax 相关联，设置间距 pad
        fig.colorbar(pcm, ax=ax, pad=pad)


@image_comparison(['constrained_layout4.png'])
def test_constrained_layout4():
    """Test constrained_layout for a single colorbar with subplots"""
    # 创建一个带有约束布局的图形对象，并生成一个 2x2 的子图数组
    fig, axs = plt.subplots(2, 2, layout="constrained")
    # 遍历所有子图，并调用 example_pcolor 函数在每个子图上进行绘制，设置字体大小为 24
    for ax in axs.flat:
        pcm = example_pcolor(ax, fontsize=24)
    # 在图形对象上添加 colorbar，与所有子图 axs 相关联，设置间距 pad 和缩放 shrink
    fig.colorbar(pcm, ax=axs, pad=0.01, shrink=0.6)


@image_comparison(['constrained_layout5.png'], tol=0.002)
def test_constrained_layout5():
    """
    Test constrained_layout for a single colorbar with subplots,
    colorbar bottom
    """
    # 创建一个带有约束布局的图形对象，并生成一个 2x2 的子图数组
    fig, axs = plt.subplots(2, 2, layout="constrained")
    # 遍历所有子图，并调用 example_pcolor 函数在每个子图上进行绘制，设置字体大小为 24
    for ax in axs.flat:
        pcm = example_pcolor(ax, fontsize=24)
    # 在图形对象上添加 colorbar，与所有子图 axs 相关联，设置间距 pad 和缩放 shrink，
    # 并指定 colorbar 位置为底部
    fig.colorbar(pcm, ax=axs,
                 use_gridspec=False, pad=0.01, shrink=0.6,
                 location='bottom')


@image_comparison(['constrained_layout6.png'], tol=0.002)
def test_constrained_layout6():
    """Test constrained_layout for nested gridspecs"""
    # 当重新生成此测试图像时，请移除此行。
    # 设置 matplotlib 参数，禁用 pcolormesh 的快照功能
    plt.rcParams['pcolormesh.snap'] = False
    
    # 创建一个新的 Figure 对象
    fig = plt.figure(layout="constrained")
    
    # 在 Figure 上添加一个 1x2 的网格布局
    gs = fig.add_gridspec(1, 2, figure=fig)
    
    # 在第一个子图中添加一个 2x2 的子网格布局
    gsl = gs[0].subgridspec(2, 2)
    
    # 在第二个子图中添加一个 1x2 的子网格布局
    gsr = gs[1].subgridspec(1, 2)
    
    # 初始化一个空列表来存储左侧子图的 Axes 对象
    axsl = []
    
    # 遍历左侧子网格布局中的每一个子图
    for gs in gsl:
        # 在当前子网格中添加一个 Axes 对象
        ax = fig.add_subplot(gs)
        # 将创建的 Axes 对象添加到 axsl 列表中
        axsl += [ax]
        # 调用 example_plot 函数，在 Axes 上绘制示例图形，设置字体大小为 12
        example_plot(ax, fontsize=12)
    
    # 设置左侧子图中最后一个 Axes 对象的 x 轴标签
    ax.set_xlabel('x-label\nMultiLine')
    
    # 初始化一个空列表来存储右侧子图的 Axes 对象
    axsr = []
    
    # 遍历右侧子网格布局中的每一个子图
    for gs in gsr:
        # 在当前子网格中添加一个 Axes 对象
        ax = fig.add_subplot(gs)
        # 将创建的 Axes 对象添加到 axsr 列表中
        axsr += [ax]
        # 调用 example_pcolor 函数，在 Axes 上绘制示例的伪彩色图，设置字体大小为 12
        pcm = example_pcolor(ax, fontsize=12)
    
    # 在 Figure 上添加一个颜色条，关联到右侧子图中的 Axes 对象，设置位置在底部
    fig.colorbar(pcm, ax=axsr,
                 pad=0.01, shrink=0.99, location='bottom',
                 ticks=ticker.MaxNLocator(nbins=5))
def test_identical_subgridspec():
    # 创建一个带有约束布局的新图形对象
    fig = plt.figure(constrained_layout=True)

    # 在图形对象中添加一个 2x1 的总网格规范
    GS = fig.add_gridspec(2, 1)

    # 在第一个网格中添加一个 1x3 的子网格规范
    GSA = GS[0].subgridspec(1, 3)
    # 在第二个网格中添加一个 1x3 的子网格规范
    GSB = GS[1].subgridspec(1, 3)

    axa = []
    axb = []
    # 在每个子网格中添加子图，并将其存储在对应的列表中
    for i in range(3):
        axa += [fig.add_subplot(GSA[i])]
        axb += [fig.add_subplot(GSB[i])]

    # 绘制图形，但不进行渲染
    fig.draw_without_rendering()
    # 断言：检查第一行的第一个子图底部的位置是否在第二行的第一个子图顶部的位置之上
    assert axa[0].get_position().y0 > axb[0].get_position().y1


def test_constrained_layout7():
    """测试在 GridSpec 中未设置 fig 时是否会产生警告"""
    # 使用 pytest 的 warns 函数来检查是否有特定的警告消息
    with pytest.warns(
        UserWarning, match=('There are no gridspecs with layoutgrids. '
                            'Possibly did not call parent GridSpec with '
                            'the "figure" keyword')):
        # 创建一个带有约束布局的新图形对象
        fig = plt.figure(layout="constrained")
        # 创建一个 1x2 的总网格规范
        gs = gridspec.GridSpec(1, 2)
        # 在第一个网格规范上创建一个包含 2x2 的子网格规范
        gsl = gridspec.GridSpecFromSubplotSpec(2, 2, gs[0])
        # 在第二个网格规范上创建一个包含 1x2 的子网格规范
        gsr = gridspec.GridSpecFromSubplotSpec(1, 2, gs[1])
        # 在每个子网格规范中添加子图
        for gs in gsl:
            fig.add_subplot(gs)
        # 需要触发绘图以产生警告
        fig.draw_without_rendering()


@image_comparison(['constrained_layout8.png'])
def test_constrained_layout8():
    """测试非完全填充的网格规范"""
    # 创建一个大小为 (10, 5) 的新图形对象，并设置约束布局
    fig = plt.figure(figsize=(10, 5), layout="constrained")
    # 创建一个 3x5 的总网格规范，并将其关联到当前图形对象
    gs = gridspec.GridSpec(3, 5, figure=fig)
    axs = []
    for j in [0, 1]:
        if j == 0:
            ilist = [1]
        else:
            ilist = [0, 4]
        for i in ilist:
            # 在指定位置添加子图，并将其存储在列表中
            ax = fig.add_subplot(gs[j, i])
            axs += [ax]
            # 调用示例函数 example_pcolor 绘制子图内容
            example_pcolor(ax, fontsize=9)
            if i > 0:
                ax.set_ylabel('')
            if j < 1:
                ax.set_xlabel('')
            ax.set_title('')
    # 在最后一行添加一个子图，并将其存储在列表中
    ax = fig.add_subplot(gs[2, :])
    axs += [ax]
    # 调用示例函数 example_pcolor 绘制子图内容
    pcm = example_pcolor(ax, fontsize=9)

    # 添加颜色条，将其关联到所有子图，并设置其属性
    fig.colorbar(pcm, ax=axs, pad=0.01, shrink=0.6)


@image_comparison(['constrained_layout9.png'])
def test_constrained_layout9():
    """测试处理 suptitle 以及 sharex 和 sharey"""
    # 创建一个包含 2x2 子图的新图形对象，并设置约束布局以及 sharex 和 sharey 的参数
    fig, axs = plt.subplots(2, 2, layout="constrained",
                            sharex=False, sharey=False)
    for ax in axs.flat:
        # 调用示例函数 example_pcolor 绘制子图内容，并设置轴标签为空字符串
        pcm = example_pcolor(ax, fontsize=24)
        ax.set_xlabel('')
        ax.set_ylabel('')
    # 设置所有子图的纵横比为 2
    ax.set_aspect(2.)
    # 添加颜色条，将其关联到所有子图，并设置其属性
    fig.colorbar(pcm, ax=axs, pad=0.01, shrink=0.6)
    # 设置图形的总标题
    fig.suptitle('Test Suptitle', fontsize=28)


@image_comparison(['constrained_layout10.png'])
def test_constrained_layout10():
    """测试处理位于轴外的图例"""
    # 创建一个包含 2x2 子图的新图形对象，并设置约束布局
    fig, axs = plt.subplots(2, 2, layout="constrained")
    for ax in axs.flat:
        # 在每个子图上绘制简单的折线图，并添加标签
        ax.plot(np.arange(12), label='This is a label')
    # 将图例放置在左上角，但是超出轴的边界
    ax.legend(loc='center left', bbox_to_anchor=(0.8, 0.5))


@image_comparison(['constrained_layout11.png'])
def test_constrained_layout11():
    """测试处理多层嵌套的网格规范"""
    # 创建一个大小为 (13, 3) 的新图形对象，并设置约束布局
    fig = plt.figure(layout="constrained", figsize=(13, 3))
    # 创建一个1行2列的网格布局对象gs0，并将其绑定到给定的图形fig上
    gs0 = gridspec.GridSpec(1, 2, figure=fig)
    
    # 根据子图规范1行2列创建一个网格布局对象gsl，并将其绑定到gs0的第一个位置上
    gsl = gridspec.GridSpecFromSubplotSpec(1, 2, gs0[0])
    
    # 根据子图规范2行2列创建一个网格布局对象gsl0，并将其绑定到gsl的第二个位置上
    gsl0 = gridspec.GridSpecFromSubplotSpec(2, 2, gsl[1])
    
    # 在图形fig上添加一个子图，位置由gs0的第二个位置决定，并将该子图对象赋给变量ax
    ax = fig.add_subplot(gs0[1])
    
    # 调用example_plot函数，在ax子图上创建一个示例图，使用字体大小为9
    example_plot(ax, fontsize=9)
    
    # 初始化一个空列表axs，用于存储后续创建的子图对象
    axs = []
    
    # 遍历gsl0中的每个网格规范gs，依次在图形fig上添加一个子图，将子图对象添加到axs列表中
    for gs in gsl0:
        ax = fig.add_subplot(gs)
        axs += [ax]
        # 调用example_pcolor函数，在当前子图ax上创建一个示例伪彩图，使用字体大小为9，并将返回的对象赋给pcm
        pcm = example_pcolor(ax, fontsize=9)
    
    # 在图形fig上添加一个颜色条，关联到axs列表中的所有子图，设置收缩因子为0.6，长宽比为70
    fig.colorbar(pcm, ax=axs, shrink=0.6, aspect=70.)
    
    # 在图形fig上添加一个子图，位置由gsl的第一个位置决定，并将该子图对象赋给变量ax
    ax = fig.add_subplot(gsl[0])
    
    # 调用example_plot函数，在ax子图上创建一个示例图，使用字体大小为9
    example_plot(ax, fontsize=9)
@image_comparison(['constrained_layout11rat.png'])
def test_constrained_layout11rat():
    """Test for multiple nested gridspecs with width_ratios"""

    # 创建一个新的图形对象，使用“constrained”布局，尺寸为(10, 3)
    fig = plt.figure(layout="constrained", figsize=(10, 3))

    # 创建一个1行2列的 GridSpec 对象 gs0，指定其所属图形为 fig，并设置列的宽度比例为[6, 1]
    gs0 = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[6, 1])

    # 在 gs0[0] 上创建一个包含1行2列的 GridSpec 对象 gsl
    gsl = gridspec.GridSpecFromSubplotSpec(1, 2, gs0[0])

    # 在 gsl[1] 上创建一个包含2行2列的 GridSpec 对象 gsl0，并指定行的高度比例为[2, 1]
    gsl0 = gridspec.GridSpecFromSubplotSpec(2, 2, gsl[1], height_ratios=[2, 1])

    # 在 fig 上添加一个子图，位置为 gs0[1]
    ax = fig.add_subplot(gs0[1])

    # 在 ax 上绘制示例图，字体大小为9
    example_plot(ax, fontsize=9)

    # 创建一个空列表 axs 用于存放子图对象
    axs = []

    # 遍历 gsl0 中的每一个子图对象 gs
    for gs in gsl0:
        # 在 fig 上添加一个子图，位置为 gs
        ax = fig.add_subplot(gs)
        # 将新添加的子图对象 ax 加入 axs 列表
        axs += [ax]
        # 在 ax 上绘制例子的色彩图，字体大小为9，并返回其色彩图对象
        pcm = example_pcolor(ax, fontsize=9)

    # 在图形 fig 上添加一个颜色条，绑定到 axs 列表中的子图对象，收缩因子为0.6，长宽比为70
    fig.colorbar(pcm, ax=axs, shrink=0.6, aspect=70.)

    # 在 fig 上添加一个子图，位置为 gsl[0]
    ax = fig.add_subplot(gsl[0])

    # 在 ax 上绘制示例图，字体大小为9
    example_plot(ax, fontsize=9)


@image_comparison(['constrained_layout12.png'])
def test_constrained_layout12():
    """Test that very unbalanced labeling still works."""

    # 创建一个新的图形对象，使用“constrained”布局，尺寸为(6, 8)
    fig = plt.figure(layout="constrained", figsize=(6, 8))

    # 创建一个6行2列的 GridSpec 对象 gs0，指定其所属图形为 fig
    gs0 = gridspec.GridSpec(6, 2, figure=fig)

    # 在 gs0[:3, 1] 的位置添加一个子图 ax1
    ax1 = fig.add_subplot(gs0[:3, 1])

    # 在 gs0[3:, 1] 的位置添加一个子图 ax2
    ax2 = fig.add_subplot(gs0[3:, 1])

    # 在 ax1 上绘制示例图，字体大小为18
    example_plot(ax1, fontsize=18)

    # 在 ax2 上绘制示例图，字体大小为18
    example_plot(ax2, fontsize=18)

    # 在 gs0[0:2, 0] 的位置添加一个子图 ax，并禁用坐标轴标签
    ax = fig.add_subplot(gs0[0:2, 0])
    example_plot(ax, nodec=True)

    # 在 gs0[2:4, 0] 的位置添加一个子图 ax，并禁用坐标轴标签
    ax = fig.add_subplot(gs0[2:4, 0])
    example_plot(ax, nodec=True)

    # 在 gs0[4:, 0] 的位置添加一个子图 ax，并禁用坐标轴标签
    ax = fig.add_subplot(gs0[4:, 0])
    example_plot(ax, nodec=True)

    # 设置 ax 的 x 轴标签为 'x-label'
    ax.set_xlabel('x-label')


@image_comparison(['constrained_layout13.png'], tol=2.e-2)
def test_constrained_layout13():
    """Test that padding works."""

    # 创建一个新的图形对象 fig 和 2x2的子图对象数组 axs，使用“constrained”布局
    fig, axs = plt.subplots(2, 2, layout="constrained")

    # 遍历 axs 中的每一个子图对象 ax
    for ax in axs.flat:
        # 在 ax 上绘制例子的色彩图，字体大小为12，并返回其色彩图对象
        pcm = example_pcolor(ax, fontsize=12)
        # 在 ax 上添加颜色条，收缩因子为0.6，长宽比为20，内边距为0.02
        fig.colorbar(pcm, ax=ax, shrink=0.6, aspect=20., pad=0.02)

    # 使用 pytest 引发一个 TypeError 异常，测试捕获布局引擎的设置异常
    with pytest.raises(TypeError):
        fig.get_layout_engine().set(wpad=1, hpad=2)

    # 设置图形的布局引擎，调整水平和垂直间距为24/72
    fig.get_layout_engine().set(w_pad=24./72., h_pad=24./72.)


@image_comparison(['constrained_layout14.png'])
def test_constrained_layout14():
    """Test that padding works."""

    # 创建一个新的图形对象 fig 和 2x2的子图对象数组 axs，使用“constrained”布局
    fig, axs = plt.subplots(2, 2, layout="constrained")

    # 遍历 axs 中的每一个子图对象 ax
    for ax in axs.flat:
        # 在 ax 上绘制例子的色彩图，字体大小为12，并返回其色彩图对象
        pcm = example_pcolor(ax, fontsize=12)
        # 在图形 fig 上设置布局引擎的参数，包括水平和垂直间距为3/72，行和列的间距为0.2
        fig.get_layout_engine().set(
            w_pad=3./72., h_pad=3./72.,
            hspace=0.2, wspace=0.2)


@image_comparison(['constrained_layout15.png'])
def test_constrained_layout15():
    """Test that rcparams work."""

    # 设置 matplotlib 的 rcParams，启用 constrained_layout
    mpl.rcParams['figure.constrained_layout.use'] = True

    # 创建一个2x2的子图对象数组 axs
    fig, axs = plt.subplots(2, 2)

    # 遍历 axs 中的每一个子图对象 ax
    for ax in axs.flat:
        # 在 ax 上绘制示例图，字体大小为12
        example_plot(ax, fontsize=12)


@image_comparison(['constrained_layout16.png'])
def test_constrained_layout16():
    """Test ax.set_position."""

    # 创建一个新的图形对象 fig 和一个子图对象 ax，使用“constrained”布局
    fig, ax = plt.subplots(layout="constrained")

    # 在 ax 上绘制示例图，字体大小为12
    example_plot(ax, fontsize=12)

    # 在图形 fig 上添加一个坐标轴，位置为[0.2, 0.2, 0.4, 0.4]
    ax2 = fig.add_axes([0.2, 0.2, 0.4, 0.4])


@image_comparison(['constrained_layout17.png'])
def test_constrained_layout17():
    """Test uneven gridspecs"""

    # 创建一个新的图形对象 fig，使用“constrained”布局
    fig = plt.figure(layout="constrained")

    # 创建一个3行3列的 GridSpec 对象 gs，指定其所属图形为 fig
    gs = gridspec.GridSpec(3, 3, figure=fig)
    # 创建一个子图ax1，位于网格布局gs的第一行第一列
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 创建一个子图ax2，位于网格布局gs的第一行从第二列开始到最后一列
    ax2 = fig.add_subplot(gs[0, 1:])
    
    # 创建一个子图ax3，位于网格布局gs的第二行开始到倒数第二列
    ax3 = fig.add_subplot(gs[1:, 0:2])
    
    # 创建一个子图ax4，位于网格布局gs的第二行开始，最后一列
    ax4 = fig.add_subplot(gs[1:, -1])
    
    # 在每个子图上调用example_plot函数，用于绘制示例图形
    example_plot(ax1)
    example_plot(ax2)
    example_plot(ax3)
    example_plot(ax4)
def test_constrained_layout18():
    """Test twinx"""
    # 创建包含受限布局的子图
    fig, ax = plt.subplots(layout="constrained")
    # 创建第二个共享x轴的轴
    ax2 = ax.twinx()
    # 在第一个轴上绘制示例图
    example_plot(ax)
    # 在第二个轴上绘制示例图，使用24号字体
    example_plot(ax2, fontsize=24)
    # 在不渲染的情况下绘制整个图形
    fig.draw_without_rendering()
    # 断言两个轴的位置范围(extents)是否完全相同
    assert all(ax.get_position().extents == ax2.get_position().extents)


def test_constrained_layout19():
    """Test twiny"""
    # 创建包含受限布局的子图
    fig, ax = plt.subplots(layout="constrained")
    # 创建第二个共享y轴的轴
    ax2 = ax.twiny()
    # 在第一个轴上绘制示例图
    example_plot(ax)
    # 在第二个轴上绘制示例图，使用24号字体
    example_plot(ax2, fontsize=24)
    # 清空第二个轴的标题
    ax2.set_title('')
    # 清空第一个轴的标题
    ax.set_title('')
    # 在不渲染的情况下绘制整个图形
    fig.draw_without_rendering()
    # 断言两个轴的位置范围(extents)是否完全相同
    assert all(ax.get_position().extents == ax2.get_position().extents)


def test_constrained_layout20():
    """Smoke test cl does not mess up added Axes"""
    # 生成一组均匀分布的数据点
    gx = np.linspace(-5, 5, 4)
    # 计算网格数据
    img = np.hypot(gx, gx[:, None])

    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个轴到图形上，位置为[0, 0, 1, 1]
    ax = fig.add_axes([0, 0, 1, 1])
    # 绘制网格数据的伪彩色图
    mesh = ax.pcolormesh(gx, gx, img[:-1, :-1])
    # 添加色条到图形上
    fig.colorbar(mesh)


def test_constrained_layout21():
    """#11035: repeated calls to suptitle should not alter the layout"""
    # 创建包含受限布局的子图
    fig, ax = plt.subplots(layout="constrained")

    # 添加第一个全局标题
    fig.suptitle("Suptitle0")
    # 在不渲染的情况下绘制整个图形
    fig.draw_without_rendering()
    # 复制第一个轴的位置范围(extents)
    extents0 = np.copy(ax.get_position().extents)

    # 添加第二个全局标题
    fig.suptitle("Suptitle1")
    # 在不渲染的情况下绘制整个图形
    fig.draw_without_rendering()
    # 复制第一个轴的位置范围(extents)
    extents1 = np.copy(ax.get_position().extents)

    # 断言两次添加标题后轴的位置范围(extents)是否完全相同
    np.testing.assert_allclose(extents0, extents1)


def test_constrained_layout22():
    """#11035: suptitle should not be include in CL if manually positioned"""
    # 创建包含受限布局的子图
    fig, ax = plt.subplots(layout="constrained")

    # 在不渲染的情况下绘制整个图形
    fig.draw_without_rendering()
    # 复制第一个轴的位置范围(extents)
    extents0 = np.copy(ax.get_position().extents)

    # 手动设置第一个全局标题的位置
    fig.suptitle("Suptitle", y=0.5)
    # 在不渲染的情况下绘制整个图形
    fig.draw_without_rendering()
    # 复制第一个轴的位置范围(extents)
    extents1 = np.copy(ax.get_position().extents)

    # 断言手动设置标题后轴的位置范围(extents)是否与未设置标题时完全相同
    np.testing.assert_allclose(extents0, extents1)


def test_constrained_layout23():
    """
    Comment in #11035: suptitle used to cause an exception when
    reusing a figure w/ CL with ``clear=True``.
    """
    # 循环两次
    for i in range(2):
        # 创建包含受限布局的子图，清空之前的内容
        fig = plt.figure(layout="constrained", clear=True, num="123")
        # 添加1行2列的网格规范
        gs = fig.add_gridspec(1, 2)
        # 在第一个网格规范中添加2行2列的子网格规范
        sub = gs[0].subgridspec(2, 2)
        # 添加全局标题
        fig.suptitle(f"Suptitle{i}")


@image_comparison(['test_colorbar_location.png'],
                  remove_text=True, style='mpl20')
def test_colorbar_location():
    """
    Test that colorbar handling is as expected for various complicated
    cases...
    """
    # 当重新生成此测试图像时，请删除此行。
    plt.rcParams['pcolormesh.snap'] = False

    # 创建包含受限布局的子图，4行5列
    fig, axs = plt.subplots(4, 5, layout="constrained")
    # 遍历每个轴
    for ax in axs.flat:
        # 在每个轴上绘制伪彩色图
        pcm = example_pcolor(ax)
        # 清空x轴标签
        ax.set_xlabel('')
        # 清空y轴标签
        ax.set_ylabel('')
    # 在第二列的所有轴上添加色条
    fig.colorbar(pcm, ax=axs[:, 1], shrink=0.4)
    # 在最后一行的前两列轴上添加底部位置的色条
    fig.colorbar(pcm, ax=axs[-1, :2], shrink=0.5, location='bottom')
    # 在第一行的第三列至最后一列轴上添加底部位置的色条，设置边距为0.05
    fig.colorbar(pcm, ax=axs[0, 2:], shrink=0.5, location='bottom', pad=0.05)
    # 在倒数第二行的第四列至最后一列轴上添加顶部位置的色条
    fig.colorbar(pcm, ax=axs[-2, 3:], shrink=0.5, location='top')
    # 在第一行的第一列轴上添加左侧位置的色条
    fig.colorbar(pcm, ax=axs[0, 0], shrink=0.5, location='left')
    # 在第二行到第三行、第三列的子图(axs[1:3, 2])上创建一个颜色条(colorbar)
    # pcm 是一个绘图对象，用于表示颜色映射
    # shrink=0.5 表示颜色条长度为默认长度的一半
    # location='right' 表示将颜色条放置在图像的右侧
    fig.colorbar(pcm, ax=axs[1:3, 2], shrink=0.5, location='right')
def test_hidden_axes():
    # 测试当将一个 Axes 设置为不可见时，constrained_layout 仍然有效。
    # 注意，虽然不可见的 Axes 仍然占据布局中的空间
    fig, axs = plt.subplots(2, 2, layout="constrained")
    # 将第一行第二列的 Axes 设置为不可见
    axs[0, 1].set_visible(False)
    # 在不渲染的情况下绘制图形
    fig.draw_without_rendering()
    # 复制第一行第一列 Axes 的位置信息
    extents1 = np.copy(axs[0, 0].get_position().extents)

    np.testing.assert_allclose(
        extents1, [0.045552, 0.543288, 0.47819, 0.982638], rtol=1e-5)


def test_colorbar_align():
    # 遍历不同的 colorbar 位置进行测试
    for location in ['right', 'left', 'top', 'bottom']:
        fig, axs = plt.subplots(2, 2, layout="constrained")
        cbs = []
        for nn, ax in enumerate(axs.flat):
            ax.tick_params(direction='in')
            # 创建示例的伪彩图，并添加 colorbar 到当前 Axes
            pc = example_pcolor(ax)
            cb = fig.colorbar(pc, ax=ax, location=location, shrink=0.6,
                              pad=0.04)
            cbs += [cb]
            cb.ax.tick_params(direction='in')
            # 如果不是第二个 Axes，则隐藏 colorbar 的坐标轴刻度
            if nn != 1:
                cb.ax.xaxis.set_ticks([])
                cb.ax.yaxis.set_ticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
        # 设置图形的布局参数
        fig.get_layout_engine().set(w_pad=4 / 72, h_pad=4 / 72,
                                    hspace=0.1, wspace=0.1)

        # 在不渲染的情况下绘制图形
        fig.draw_without_rendering()
        # 根据 colorbar 的位置方向进行位置对齐的断言
        if location in ['left', 'right']:
            np.testing.assert_allclose(cbs[0].ax.get_position().x0,
                                       cbs[2].ax.get_position().x0)
            np.testing.assert_allclose(cbs[1].ax.get_position().x0,
                                       cbs[3].ax.get_position().x0)
        else:
            np.testing.assert_allclose(cbs[0].ax.get_position().y0,
                                       cbs[1].ax.get_position().y0)
            np.testing.assert_allclose(cbs[2].ax.get_position().y0,
                                       cbs[3].ax.get_position().y0)


@image_comparison(['test_colorbars_no_overlapV.png'], style='mpl20')
def test_colorbars_no_overlapV():
    # 创建一个固定大小的图形，并设置 constrained layout
    fig = plt.figure(figsize=(2, 4), layout="constrained")
    # 创建两个共享坐标轴的子图
    axs = fig.subplots(2, 1, sharex=True, sharey=True)
    for ax in axs:
        # 隐藏 y 轴主刻度标签
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.tick_params(axis='both', direction='in')
        # 绘制简单的图像并添加垂直方向的 colorbar
        im = ax.imshow([[1, 2], [3, 4]])
        fig.colorbar(im, ax=ax, orientation="vertical")
    # 设置图形的总标题
    fig.suptitle("foo")


@image_comparison(['test_colorbars_no_overlapH.png'], style='mpl20')
def test_colorbars_no_overlapH():
    # 创建一个固定大小的图形，并设置 constrained layout
    fig = plt.figure(figsize=(4, 2), layout="constrained")
    fig.suptitle("foo")
    # 创建两个共享坐标轴的子图
    axs = fig.subplots(1, 2, sharex=True, sharey=True)
    for ax in axs:
        # 隐藏 y 轴主刻度标签
        ax.yaxis.set_major_formatter(ticker.NullFormatter())
        ax.tick_params(axis='both', direction='in')
        # 绘制简单的图像并添加水平方向的 colorbar
        im = ax.imshow([[1, 2], [3, 4]])
        fig.colorbar(im, ax=ax, orientation="horizontal")


def test_manually_set_position():
    # 创建一个包含两个子图的图形，并设置 constrained layout
    fig, axs = plt.subplots(1, 2, layout="constrained")
    # 手动设置第一个子图的位置
    axs[0].set_position([0.2, 0.2, 0.3, 0.3])
    # 在图形对象上执行绘制操作，但不执行渲染
    fig.draw_without_rendering()
    
    # 获取子图 axs[0] 的位置信息
    pp = axs[0].get_position()
    
    # 使用 NumPy 测试断言确保 pp 与期望值 [[0.2, 0.2], [0.5, 0.5]] 接近
    np.testing.assert_allclose(pp, [[0.2, 0.2], [0.5, 0.5]])
    
    # 创建一个包含两个子图的图形对象 fig，布局使用 "constrained"
    fig, axs = plt.subplots(1, 2, layout="constrained")
    
    # 设置 axs[0] 的位置为 [0.2, 0.2, 0.3, 0.3]
    axs[0].set_position([0.2, 0.2, 0.3, 0.3])
    
    # 在 axs[0] 上创建一个 20x20 的随机颜色网格图，并将其保存为 pc 对象
    pc = axs[0].pcolormesh(np.random.rand(20, 20))
    
    # 在图形对象 fig 上为 axs[0] 添加颜色条
    fig.colorbar(pc, ax=axs[0])
    
    # 再次执行图形对象的绘制操作，但不进行渲染
    fig.draw_without_rendering()
    
    # 获取更新后的 axs[0] 的位置信息
    pp = axs[0].get_position()
    
    # 使用 NumPy 测试断言确保 pp 与期望值 [[0.2, 0.2], [0.44, 0.5]] 接近
    np.testing.assert_allclose(pp, [[0.2, 0.2], [0.44, 0.5]])
@image_comparison(['test_bboxtight.png'],
                  remove_text=True, style='mpl20',
                  savefig_kwarg={'bbox_inches': 'tight'})
# 定义一个装饰器函数，用于比较图像，参数包括预期输出图像的文件名、是否移除文本、样式等
def test_bboxtight():
    # 创建一个包含受约束布局的子图
    fig, ax = plt.subplots(layout="constrained")
    # 设置子图的纵横比为1：1
    ax.set_aspect(1.)


@image_comparison(['test_bbox.png'],
                  remove_text=True, style='mpl20',
                  savefig_kwarg={'bbox_inches':
                                 mtransforms.Bbox([[0.5, 0], [2.5, 2]])})
# 定义一个装饰器函数，用于比较图像，参数包括预期输出图像的文件名、是否移除文本、样式以及保存图像时的边界框设置
def test_bbox():
    # 创建一个包含受约束布局的子图
    fig, ax = plt.subplots(layout="constrained")
    # 设置子图的纵横比为1：1


def test_align_labels():
    """
    Tests for a bug in which constrained layout and align_ylabels on
    three unevenly sized subplots, one of whose y tick labels include
    negative numbers, drives the non-negative subplots' y labels off
    the edge of the plot
    """
    # 创建一个包含三个子图的图像，并使用约束布局，设置子图的高度比例为1、1、0.7
    fig, (ax3, ax1, ax2) = plt.subplots(3, 1, layout="constrained",
                                        figsize=(6.4, 8),
                                        gridspec_kw={"height_ratios": (1, 1,
                                                                       0.7)})

    # 设置第一个子图的y轴范围为0到1，并设置y轴标签为"Label"
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Label")

    # 设置第二个子图的y轴范围为-1.5到1.5，并设置y轴标签为"Label"
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_ylabel("Label")

    # 设置第三个子图的y轴范围为0到1，并设置y轴标签为"Label"
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Label")

    # 对子图进行y轴标签对齐
    fig.align_ylabels(axs=(ax3, ax1, ax2))

    # 在绘制图像前获取对齐后各子图y轴标签的边界框
    fig.draw_without_rendering()
    after_align = [ax1.yaxis.label.get_window_extent(),
                   ax2.yaxis.label.get_window_extent(),
                   ax3.yaxis.label.get_window_extent()]
    # 确保标签大致对齐
    np.testing.assert_allclose([after_align[0].x0, after_align[2].x0],
                               after_align[1].x0, rtol=0, atol=1e-05)
    # 确保标签不会超出边缘
    assert after_align[0].x0 >= 1


def test_suplabels():
    # 创建一个包含受约束布局的子图
    fig, ax = plt.subplots(layout="constrained")
    # 在绘制图像前获取当前子图的紧凑边界框
    fig.draw_without_rendering()
    pos0 = ax.get_tightbbox(fig.canvas.get_renderer())
    # 在图像上方添加全局x标签"Boo"
    fig.supxlabel('Boo')
    # 在图像右侧添加全局y标签"Booy"
    fig.supylabel('Booy')
    # 再次绘制图像并获取更新后的子图紧凑边界框
    fig.draw_without_rendering()
    pos = ax.get_tightbbox(fig.canvas.get_renderer())
    # 确保新添加的标签位置在旧边界框的上方和右侧
    assert pos.y0 > pos0.y0 + 10.0
    assert pos.x0 > pos0.x0 + 10.0

    # 创建一个包含受约束布局的子图
    fig, ax = plt.subplots(layout="constrained")
    # 在绘制图像前获取当前子图的紧凑边界框
    fig.draw_without_rendering()
    pos0 = ax.get_tightbbox(fig.canvas.get_renderer())
    # 检查指定x（y）不会破坏布局
    fig.supxlabel('Boo', x=0.5)
    fig.supylabel('Boo', y=0.5)
    # 再次绘制图像并获取更新后的子图紧凑边界框
    fig.draw_without_rendering()
    pos = ax.get_tightbbox(fig.canvas.get_renderer())
    # 确保新添加的标签位置在旧边界框的上方和右侧
    assert pos.y0 > pos0.y0 + 10.0
    assert pos.x0 > pos0.x0 + 10.0


def test_gridspec_addressing():
    # 创建一个新的图像对象
    fig = plt.figure()
    # 向图像添加一个3x3的网格布局
    gs = fig.add_gridspec(3, 3)
    # 在网格中添加一个跨越第1列和第2列、第1行至最后一行的子图
    sp = fig.add_subplot(gs[0:, 1:])
    # 在不渲染的情况下绘制图像
    fig.draw_without_rendering()


def test_discouraged_api():
    # 创建一个包含受约束布局的子图
    fig, ax = plt.subplots(constrained_layout=True)
    # 在不渲染的情况下绘制图像
    fig.draw_without_rendering()
    # 使用 pytest 模块捕获 PendingDeprecationWarning 警告，并匹配特定的警告消息字符串
    with pytest.warns(PendingDeprecationWarning,
                      match="will be deprecated"):
        # 创建一个新的图形对象和轴对象
        fig, ax = plt.subplots()
        # 设置图形对象使用约束布局
        fig.set_constrained_layout(True)
        # 在不渲染的情况下绘制图形
        fig.draw_without_rendering()
    
    # 使用 pytest 模块捕获 PendingDeprecationWarning 警告，并匹配特定的警告消息字符串
    with pytest.warns(PendingDeprecationWarning,
                      match="will be deprecated"):
        # 创建一个新的图形对象和轴对象
        fig, ax = plt.subplots()
        # 设置图形对象使用指定的约束布局参数
        fig.set_constrained_layout({'w_pad': 0.02, 'h_pad': 0.02})
        # 在不渲染的情况下绘制图形
        fig.draw_without_rendering()
def test_kwargs():
    # 创建一个包含子图的图形对象，设置子图布局参数为{'h_pad': 0.02}
    fig, ax = plt.subplots(constrained_layout={'h_pad': 0.02})
    # 在不渲染的情况下绘制图形
    fig.draw_without_rendering()


def test_rect():
    # 创建一个包含子图的图形对象，设置布局参数为'constrained'
    fig, ax = plt.subplots(layout='constrained')
    # 获取图形的布局引擎并设置其矩形区域为[0, 0, 0.5, 0.5]
    fig.get_layout_engine().set(rect=[0, 0, 0.5, 0.5])
    # 在不渲染的情况下绘制图形
    fig.draw_without_rendering()
    # 获取子图的位置信息
    ppos = ax.get_position()
    # 断言子图右上角的 x 坐标小于 0.5
    assert ppos.x1 < 0.5
    # 断言子图右上角的 y 坐标小于 0.5

    fig, ax = plt.subplots(layout='constrained')
    # 获取图形的布局引擎并设置其矩形区域为[0.2, 0.2, 0.3, 0.3]
    fig.get_layout_engine().set(rect=[0.2, 0.2, 0.3, 0.3])
    # 在不渲染的情况下绘制图形
    fig.draw_without_rendering()
    # 获取子图的位置信息
    ppos = ax.get_position()
    # 断言子图右上角的 x 坐标小于 0.5
    assert ppos.x1 < 0.5
    # 断言子图右上角的 y 坐标小于 0.5
    # 断言子图左下角的 x 坐标大于 0.2
    assert ppos.x0 > 0.2
    # 断言子图左下角的 y 坐标大于 0.2


def test_compressed1():
    # 创建一个包含3行2列子图的图形对象，设置布局参数为'compressed'，共享 x 和 y 轴
    fig, axs = plt.subplots(3, 2, layout='compressed', sharex=True, sharey=True)
    # 在每个子图上绘制一个随机的20x20的图像
    for ax in axs.flat:
        pc = ax.imshow(np.random.randn(20, 20))

    # 在图形上添加颜色条
    fig.colorbar(pc, ax=axs)
    # 在不渲染的情况下绘制图形
    fig.draw_without_rendering()

    # 获取第一个子图的位置信息
    pos = axs[0, 0].get_position()
    # 断言第一个子图左下角的 x 坐标接近于 0.2344，容忍度为 1e-3
    np.testing.assert_allclose(pos.x0, 0.2344, atol=1e-3)
    # 获取第二个子图的位置信息
    pos = axs[0, 1].get_position()
    # 断言第二个子图右上角的 x 坐标接近于 0.7024，容忍度为 1e-3
    np.testing.assert_allclose(pos.x1, 0.7024, atol=1e-3)

    # 创建一个包含2行3列子图的图形对象，设置布局参数为'compressed'，共享 x 和 y 轴，图形尺寸为(5, 4)
    fig, axs = plt.subplots(2, 3, layout='compressed', sharex=True, sharey=True, figsize=(5, 4))
    # 在每个子图上绘制一个随机的20x20的图像
    for ax in axs.flat:
        pc = ax.imshow(np.random.randn(20, 20))

    # 在图形上添加颜色条
    fig.colorbar(pc, ax=axs)
    # 在不渲染的情况下绘制图形
    fig.draw_without_rendering()

    # 获取第一个子图的位置信息
    pos = axs[0, 0].get_position()
    # 断言第一个子图左下角的 x 坐标接近于 0.06195，容忍度为 1e-3
    np.testing.assert_allclose(pos.x0, 0.06195, atol=1e-3)
    # 断言第一个子图右上角的 y 坐标接近于 0.8537，容忍度为 1e-3
    np.testing.assert_allclose(pos.y1, 0.8537, atol=1e-3)
    # 获取第六个子图的位置信息
    pos = axs[1, 2].get_position()
    # 断言第六个子图右上角的 x 坐标接近于 0.8618，容忍度为 1e-3
    np.testing.assert_allclose(pos.x1, 0.8618, atol=1e-3)
    # 断言第六个子图左下角的 y 坐标接近于 0.1934，容忍度为 1e-3
    np.testing.assert_allclose(pos.y0, 0.1934, atol=1e-3)


@pytest.mark.parametrize('arg, state', [
    (True, True),
    (False, False),
    ({}, True),
    ({'rect': None}, True)
])
def test_set_constrained_layout(arg, state):
    # 创建一个包含子图的图形对象，设置约束布局参数为 arg
    fig, ax = plt.subplots(constrained_layout=arg)
    # 断言图形的约束布局状态是否与 state 相符
    assert fig.get_constrained_layout() is state


def test_constrained_toggle():
    # 创建一个包含子图的图形对象
    fig, ax = plt.subplots()
    # 在警告即将过时的情况下进行测试
    with pytest.warns(PendingDeprecationWarning):
        # 设置约束布局为 True，并断言约束布局状态为 True
        fig.set_constrained_layout(True)
        assert fig.get_constrained_layout()
        # 设置约束布局为 False，并断言约束布局状态为 False
        fig.set_constrained_layout(False)
        assert not fig.get_constrained_layout()
        # 再次设置约束布局为 True，并断言约束布局状态为 True
        fig.set_constrained_layout(True)
        assert fig.get_constrained_layout()


def test_layout_leak():
    # 确保在使用 LayoutGrid 时没有循环引用
    # GH #25853
    # 创建一个包含约束布局的图形对象，尺寸为 (10, 10)
    fig = plt.figure(constrained_layout=True, figsize=(10, 10))
    # 添加一个子图
    fig.add_subplot()
    # 在不渲染的情况下绘制图形
    fig.draw_without_rendering()
    # 关闭所有图形窗口
    plt.close("all")
    # 删除图形对象
    del fig
    # 执行垃圾回收
    gc.collect()
    # 断言 gc 对象中不存在 LayoutGrid 的实例
    assert not any(isinstance(obj, mpl._layoutgrid.LayoutGrid) for obj in gc.get_objects())
```