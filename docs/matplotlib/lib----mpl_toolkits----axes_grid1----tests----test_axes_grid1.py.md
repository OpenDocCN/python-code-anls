# `D:\src\scipysrc\matplotlib\lib\mpl_toolkits\axes_grid1\tests\test_axes_grid1.py`

```
# 导入所需模块和函数
from itertools import product  # 导入 itertools 模块中的 product 函数
import io  # 导入 io 模块
import platform  # 导入 platform 模块

import matplotlib as mpl  # 导入 matplotlib 库并将其重命名为 mpl
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块并将其重命名为 plt
import matplotlib.ticker as mticker  # 导入 matplotlib.ticker 模块并将其重命名为 mticker
from matplotlib import cbook  # 从 matplotlib 库中导入 cbook 模块
from matplotlib.backend_bases import MouseEvent  # 从 matplotlib.backend_bases 模块中导入 MouseEvent 类
from matplotlib.colors import LogNorm  # 从 matplotlib.colors 模块中导入 LogNorm 类
from matplotlib.patches import Circle, Ellipse  # 从 matplotlib.patches 模块中导入 Circle 和 Ellipse 类
from matplotlib.transforms import Bbox, TransformedBbox  # 从 matplotlib.transforms 模块中导入 Bbox 和 TransformedBbox 类
from matplotlib.testing.decorators import (  # 从 matplotlib.testing.decorators 模块中导入装饰器
    check_figures_equal, image_comparison, remove_ticks_and_titles)

from mpl_toolkits.axes_grid1 import (  # 从 mpl_toolkits.axes_grid1 中导入以下模块和类
    axes_size as Size,
    host_subplot, make_axes_locatable,
    Grid, AxesGrid, ImageGrid)
from mpl_toolkits.axes_grid1.anchored_artists import (  # 从 mpl_toolkits.axes_grid1.anchored_artists 中导入以下类
    AnchoredAuxTransformBox, AnchoredDrawingArea, AnchoredEllipse,
    AnchoredDirectionArrows, AnchoredSizeBar)
from mpl_toolkits.axes_grid1.axes_divider import (  # 从 mpl_toolkits.axes_grid1.axes_divider 中导入以下函数和类
    Divider, HBoxDivider, make_axes_area_auto_adjustable, SubplotDivider,
    VBoxDivider)
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes  # 从 mpl_toolkits.axes_grid1.axes_rgb 中导入 RGBAxes 类
from mpl_toolkits.axes_grid1.inset_locator import (  # 从 mpl_toolkits.axes_grid1.inset_locator 中导入以下函数和类
    zoomed_inset_axes, mark_inset, inset_axes, BboxConnectorPatch,
    InsetPosition)
import mpl_toolkits.axes_grid1.mpl_axes  # 导入 mpl_toolkits.axes_grid1.mpl_axes 模块
import pytest  # 导入 pytest 模块

import numpy as np  # 导入 numpy 库并将其重命名为 np
from numpy.testing import assert_array_equal, assert_array_almost_equal  # 从 numpy.testing 模块中导入两个函数


# 定义测试函数 test_divider_append_axes()
def test_divider_append_axes():
    # 创建一个新的图形窗口和一个子图
    fig, ax = plt.subplots()
    # 使用 make_axes_locatable 函数创建一个分隔器对象
    divider = make_axes_locatable(ax)
    # 在分隔器上附加并返回多个辅助轴
    axs = {
        "main": ax,
        "top": divider.append_axes("top", 1.2, pad=0.1, sharex=ax),
        "bottom": divider.append_axes("bottom", 1.2, pad=0.1, sharex=ax),
        "left": divider.append_axes("left", 1.2, pad=0.1, sharey=ax),
        "right": divider.append_axes("right", 1.2, pad=0.1, sharey=ax),
    }
    # 绘制图形以确保布局已经计算
    fig.canvas.draw()
    # 获取各辅助轴的窗口范围
    bboxes = {k: axs[k].get_window_extent() for k in axs}
    # 获取图形的 DPI 设置
    dpi = fig.dpi
    # 断言各辅助轴的高度与预期值相近
    assert bboxes["top"].height == pytest.approx(1.2 * dpi)
    assert bboxes["bottom"].height == pytest.approx(1.2 * dpi)
    assert bboxes["left"].width == pytest.approx(1.2 * dpi)
    assert bboxes["right"].width == pytest.approx(1.2 * dpi)
    # 断言各辅助轴与主轴之间的垂直间距相近
    assert bboxes["top"].y0 - bboxes["main"].y1 == pytest.approx(0.1 * dpi)
    assert bboxes["main"].y0 - bboxes["bottom"].y1 == pytest.approx(0.1 * dpi)
    # 断言各辅助轴与主轴之间的水平间距相近
    assert bboxes["main"].x0 - bboxes["left"].x1 == pytest.approx(0.1 * dpi)
    assert bboxes["right"].x0 - bboxes["main"].x1 == pytest.approx(0.1 * dpi)
    # 断言左辅助轴与主轴与右辅助轴的纵坐标相同
    assert bboxes["left"].y0 == bboxes["main"].y0 == bboxes["right"].y0
    assert bboxes["left"].y1 == bboxes["main"].y1 == bboxes["right"].y1
    # 断言顶部辅助轴与主轴与底部辅助轴的横坐标相同
    assert bboxes["top"].x0 == bboxes["main"].x0 == bboxes["bottom"].x0
    assert bboxes["top"].x1 == bboxes["main"].x1 == bboxes["bottom"].x1


# 使用装饰器 image_comparison 进行测试，检查生成的图像是否正确
@image_comparison(['twin_axes_empty_and_removed'], extensions=["png"], tol=1,
                  style=('classic', '_classic_test_patch'))
# 定义测试函数 test_twin_axes_empty_and_removed()
def test_twin_axes_empty_and_removed():
    # 更新 matplotlib 的默认参数，调整字体大小以避免重叠
    mpl.rcParams.update(
        {"font.size": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
    generators = ["twinx", "twiny", "twin"]
    modifiers = ["", "host invisible", "twin removed", "twin invisible",
                 "twin removed\nhost invisible"]
    # 创建一个未修改的主子图作为参考
    h = host_subplot(len(modifiers)+1, len(generators), 2)
    h.text(0.5, 0.5, "host_subplot",
           horizontalalignment="center", verticalalignment="center")
    # 创建带有不同修改（twin*，可见性）的主子图
    for i, (mod, gen) in enumerate(product(modifiers, generators),
                                   len(generators) + 1):
        # 根据修改和生成器类型创建主子图对象
        h = host_subplot(len(modifiers)+1, len(generators), i)
        # 根据生成器类型创建对应的子图对象
        t = getattr(h, gen)()
        # 如果修改包含 "twin invisible"，则设置子图全部轴不可见
        if "twin invisible" in mod:
            t.axis[:].set_visible(False)
        # 如果修改包含 "twin removed"，则移除子图
        if "twin removed" in mod:
            t.remove()
        # 如果修改包含 "host invisible"，则设置主子图全部轴不可见
        if "host invisible" in mod:
            h.axis[:].set_visible(False)
        # 在主子图中心添加文本，显示生成器类型和对应修改（如果有）
        h.text(0.5, 0.5, gen + ("\n" + mod if mod else ""),
               horizontalalignment="center", verticalalignment="center")
    # 调整子图之间的水平和垂直间距
    plt.subplots_adjust(wspace=0.5, hspace=1)
def test_twin_axes_both_with_units():
    # 创建一个主轴对象
    host = host_subplot(111)
    # 在使用过程中捕获关于Matplotlib过时功能的警告
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        # 在主轴上绘制日期数据，x轴非日期，y轴日期
        host.plot_date([0, 1, 2], [0, 1, 2], xdate=False, ydate=True)
    # 创建一个与主轴共享x轴的次轴对象
    twin = host.twinx()
    # 在次轴上绘制一些数据
    twin.plot(["a", "b", "c"])
    # 断言主轴第一个刻度标签的文本为"00:00:00"
    assert host.get_yticklabels()[0].get_text() == "00:00:00"
    # 断言次轴第一个刻度标签的文本为"a"


def test_axesgrid_colorbar_log_smoketest():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 创建一个包含单个子图的网格对象
    grid = AxesGrid(fig, 111,  # modified to be only subplot
                    nrows_ncols=(1, 1),
                    ngrids=1,
                    label_mode="L",
                    cbar_location="top",
                    cbar_mode="single",
                    )
    # 创建一个随机数据矩阵
    Z = 10000 * np.random.rand(10, 10)
    # 在网格的第一个子图上显示图像，使用对数标准化
    im = grid[0].imshow(Z, interpolation="nearest", norm=LogNorm())
    # 在颜色条轴上添加颜色条
    grid.cbar_axes[0].colorbar(im)


def test_inset_colorbar_tight_layout_smoketest():
    # 创建一个包含单个子图的图形对象
    fig, ax = plt.subplots(1, 1)
    # 在主图上绘制散点图
    pts = ax.scatter([0, 1], [0, 1], c=[1, 5])

    # 创建一个嵌入的轴对象，用于放置颜色条
    cax = inset_axes(ax, width="3%", height="70%")
    # 在嵌入的轴上添加颜色条
    plt.colorbar(pts, cax=cax)

    # 捕获包含Axes的警告信息
    with pytest.warns(UserWarning, match="This figure includes Axes"):
        # 调整图形布局以适应子图
        plt.tight_layout()


@image_comparison(['inset_locator.png'], style='default', remove_text=True)
def test_inset_locator():
    # 创建一个具有指定尺寸的子图对象
    fig, ax = plt.subplots(figsize=[5, 4])

    # 准备演示图像数据
    Z = cbook.get_sample_data("axes_grid/bivariate_normal.npy")
    extent = (-3, 4, -4, 3)
    Z2 = np.zeros((150, 150))
    ny, nx = Z.shape
    Z2[30:30+ny, 30:30+nx] = Z

    # 在主图上显示图像
    ax.imshow(Z2, extent=extent, interpolation="nearest",
              origin="lower")

    # 创建一个放大的嵌入轴对象
    axins = zoomed_inset_axes(ax, zoom=6, loc='upper right')
    # 在放大的嵌入轴上显示图像
    axins.imshow(Z2, extent=extent, interpolation="nearest",
                 origin="lower")
    # 设置放大的嵌入轴的主刻度定位器参数
    axins.yaxis.get_major_locator().set_params(nbins=7)
    axins.xaxis.get_major_locator().set_params(nbins=7)
    # 设置放大区域的范围
    x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    # 不显示放大轴的刻度
    plt.xticks(visible=False)
    plt.yticks(visible=False)

    # 在主图上绘制放大区域的边界框及其与放大轴区域之间的连接线
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # 在图上添加一个带有尺寸标尺的锚定对象
    asb = AnchoredSizeBar(ax.transData,
                          0.5,
                          '0.5',
                          loc='lower center',
                          pad=0.1, borderpad=0.5, sep=5,
                          frameon=False)
    ax.add_artist(asb)


@image_comparison(['inset_axes.png'], style='default', remove_text=True)
def test_inset_axes():
    # 创建一个具有指定尺寸的子图对象
    fig, ax = plt.subplots(figsize=[5, 4])

    # 准备演示图像数据
    Z = cbook.get_sample_data("axes_grid/bivariate_normal.npy")
    extent = (-3, 4, -4, 3)
    Z2 = np.zeros((150, 150))
    ny, nx = Z.shape
    Z2[30:30+ny, 30:30+nx] = Z
    # 在主图 `ax` 上显示图像 `Z2`，使用指定的范围和插值方式，原点在底部
    ax.imshow(Z2, extent=extent, interpolation="nearest", origin="lower")

    # 创建插图的坐标轴 `axins`，使用 `bbox_transform` 参数将其放置在主图 `ax` 上的指定位置
    axins = inset_axes(ax, width=1., height=1., bbox_to_anchor=(1, 1),
                       bbox_transform=ax.transAxes)

    # 在插图 `axins` 上显示图像 `Z2`，使用指定的范围和插值方式，原点在底部
    axins.imshow(Z2, extent=extent, interpolation="nearest", origin="lower")
    
    # 设置插图 `axins` 的 y 轴和 x 轴主要定位器的数量为 7
    axins.yaxis.get_major_locator().set_params(nbins=7)
    axins.xaxis.get_major_locator().set_params(nbins=7)
    
    # 设置插图 `axins` 的 x 和 y 轴的显示范围
    x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    # 在主图 `ax` 上隐藏 x 和 y 轴的刻度
    plt.xticks(visible=False)
    plt.yticks(visible=False)

    # 在主图 `ax` 上绘制插图 `axins` 的边界框和边框连接线，位置由 loc1 和 loc2 指定
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # 在主图 `ax` 上添加一个固定大小的比例尺条 `asb`
    asb = AnchoredSizeBar(ax.transData,
                          0.5,
                          '0.5',
                          loc='lower center',
                          pad=0.1, borderpad=0.5, sep=5,
                          frameon=False)
    ax.add_artist(asb)
def test_inset_axes_complete():
    # 设置图像的 DPI
    dpi = 100
    # 设置图像的尺寸
    figsize = (6, 5)
    # 创建一个新的图像和轴对象
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # 调整子图的位置，left, bottom, right, top 分别为 0.1, 0.1, 0.9, 0.9
    fig.subplots_adjust(.1, .1, .9, .9)

    # 创建一个插入轴对象，宽度为 2，高度为 2，边框填充为 0
    ins = inset_axes(ax, width=2., height=2., borderpad=0)
    fig.canvas.draw()
    # 断言插入轴的位置与预期的位置几乎相等
    assert_array_almost_equal(
        ins.get_position().extents,
        [(0.9*figsize[0]-2.)/figsize[0], (0.9*figsize[1]-2.)/figsize[1],
         0.9, 0.9])

    # 创建一个插入轴对象，宽度为 "40%"，高度为 "30%"，边框填充为 0
    ins = inset_axes(ax, width="40%", height="30%", borderpad=0)
    fig.canvas.draw()
    # 断言插入轴的位置与预期的位置几乎相等
    assert_array_almost_equal(
        ins.get_position().extents, [.9-.8*.4, .9-.8*.3, 0.9, 0.9])

    # 创建一个插入轴对象，宽度为 1，高度为 1.2，锚点为 (200, 100)，位置为 3，边框填充为 0
    ins = inset_axes(ax, width=1., height=1.2, bbox_to_anchor=(200, 100),
                     loc=3, borderpad=0)
    fig.canvas.draw()
    # 断言插入轴的位置与预期的位置几乎相等
    assert_array_almost_equal(
        ins.get_position().extents,
        [200/dpi/figsize[0], 100/dpi/figsize[1],
         (200/dpi+1)/figsize[0], (100/dpi+1.2)/figsize[1]])

    # 创建两个插入轴对象，宽度和高度分别为 "35%" 和 "60%"，位置为 3，边框填充为 1
    ins1 = inset_axes(ax, width="35%", height="60%", loc=3, borderpad=1)
    ins2 = inset_axes(ax, width="100%", height="100%",
                      bbox_to_anchor=(0, 0, .35, .60),
                      bbox_transform=ax.transAxes, loc=3, borderpad=1)
    fig.canvas.draw()
    # 断言两个插入轴的位置范围完全相同
    assert_array_equal(ins1.get_position().extents,
                       ins2.get_position().extents)

    # 使用 pytest 检查是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        ins = inset_axes(ax, width="40%", height="30%",
                         bbox_to_anchor=(0.4, 0.5))

    # 使用 pytest 检查是否会引发 UserWarning 警告
    with pytest.warns(UserWarning):
        ins = inset_axes(ax, width="40%", height="30%",
                         bbox_transform=ax.transAxes)


def test_inset_axes_tight():
    # 检测到 inset_axes 在使用 bbox_inches="tight" 时会出现问题
    fig, ax = plt.subplots()
    # 创建一个插入轴对象，宽度为 1.3，高度为 0.9
    inset_axes(ax, width=1.3, height=0.9)

    # 创建一个字节流对象
    f = io.BytesIO()
    # 将图像保存到字节流中，使用 bbox_inches="tight" 参数
    fig.savefig(f, bbox_inches="tight")


@image_comparison(['fill_facecolor.png'], remove_text=True, style='mpl20')
def test_fill_facecolor():
    fig, ax = plt.subplots(1, 5)
    fig.set_size_inches(5, 5)
    for i in range(1, 4):
        ax[i].yaxis.set_visible(False)
    ax[4].yaxis.tick_right()
    bbox = Bbox.from_extents(0, 0.4, 1, 0.6)

    # 使用 'fc' 字段设置填充为蓝色
    bbox1 = TransformedBbox(bbox, ax[0].transData)
    bbox2 = TransformedBbox(bbox, ax[1].transData)
    # 创建 BboxConnectorPatch 对象，设置边框颜色为 "r"，填充颜色为 "b"
    p = BboxConnectorPatch(
        bbox1, bbox2, loc1a=1, loc2a=2, loc1b=4, loc2b=3,
        ec="r", fc="b")
    p.set_clip_on(False)
    ax[0].add_patch(p)
    # 创建一个放大区域的插图轴对象，设置 x 和 y 轴的范围
    axins = zoomed_inset_axes(ax[0], 1, loc='upper right')
    axins.set_xlim(0, 0.2)
    axins.set_ylim(0, 0.2)
    plt.gca().axes.xaxis.set_ticks([])
    plt.gca().axes.yaxis.set_ticks([])
    # 在主图和插图之间标记插图区域，填充颜色为 "b"，边框颜色为 "0.5"
    mark_inset(ax[0], axins, loc1=2, loc2=4, fc="b", ec="0.5")

    # 使用 'facecolor' 字段设置填充为黄色
    bbox3 = TransformedBbox(bbox, ax[1].transData)
    bbox4 = TransformedBbox(bbox, ax[2].transData)
    # 创建 BboxConnectorPatch 对象，设置边框颜色为 "r"，填充颜色为 "yellow"
    # 创建一个 BboxConnectorPatch 对象，连接两个给定的bbox（bbox3和bbox4），设置连接点的位置loc1a、loc2a、loc1b、loc2b，边框颜色为红色，填充颜色为黄色
    p = BboxConnectorPatch(
        bbox3, bbox4, loc1a=1, loc2a=2, loc1b=4, loc2b=3,
        ec="r", facecolor="y")
    # 设置图形对象的裁剪功能为关闭
    p.set_clip_on(False)
    # 将创建的图形对象p添加到第二个子图(ax[1])中
    ax[1].add_patch(p)
    
    # 在第二个子图(ax[1])上创建一个放大镜视图，放大比例为1，位置在右上角
    axins = zoomed_inset_axes(ax[1], 1, loc='upper right')
    # 设置放大镜视图的X轴范围
    axins.set_xlim(0, 0.2)
    # 设置放大镜视图的Y轴范围
    axins.set_ylim(0, 0.2)
    # 设置放大镜视图的X轴刻度为空
    plt.gca().axes.xaxis.set_ticks([])
    # 设置放大镜视图的Y轴刻度为空
    plt.gca().axes.yaxis.set_ticks([])
    # 标记放大镜视图，连接原始图(ax[1])和放大镜视图(axins)，标记区域的填充颜色为黄色，边框颜色为0.5灰色
    mark_inset(ax[1], axins, loc1=2, loc2=4, facecolor="y", ec="0.5")

    # 创建两个经过变换的bbox对象，分别应用于第三个(ax[2])和第四个(ax[3])子图的数据坐标系
    bbox5 = TransformedBbox(bbox, ax[2].transData)
    bbox6 = TransformedBbox(bbox, ax[3].transData)
    # 创建一个BboxConnectorPatch对象，连接bbox5和bbox6，设置连接点的位置loc1a、loc2a、loc1b、loc2b，边框颜色为红色，填充颜色为绿色
    p = BboxConnectorPatch(
        bbox5, bbox6, loc1a=1, loc2a=2, loc1b=4, loc2b=3,
        ec="r", color="g")
    # 设置图形对象的裁剪功能为关闭
    p.set_clip_on(False)
    # 将创建的图形对象p添加到第三个子图(ax[2])中
    ax[2].add_patch(p)
    # 在第三个子图(ax[2])上创建一个放大镜视图，放大比例为1，位置在右上角
    axins = zoomed_inset_axes(ax[2], 1, loc='upper right')
    # 设置放大镜视图的X轴范围
    axins.set_xlim(0, 0.2)
    # 设置放大镜视图的Y轴范围
    axins.set_ylim(0, 0.2)
    # 设置放大镜视图的X轴刻度为空
    plt.gca().axes.xaxis.set_ticks([])
    # 设置放大镜视图的Y轴刻度为空
    plt.gca().axes.yaxis.set_ticks([])
    # 标记放大镜视图，连接原始图(ax[2])和放大镜视图(axins)，标记区域的填充颜色为绿色，边框颜色为0.5灰色
    mark_inset(ax[2], axins, loc1=2, loc2=4, color="g", ec="0.5")

    # 创建两个经过变换的bbox对象，分别应用于第四个(ax[3])和第五个(ax[4])子图的数据坐标系
    bbox7 = TransformedBbox(bbox, ax[3].transData)
    bbox8 = TransformedBbox(bbox, ax[4].transData)
    # 创建一个BboxConnectorPatch对象，连接bbox7和bbox8，设置连接点的位置loc1a、loc2a、loc1b、loc2b，边框颜色为红色，填充颜色为绿色，填充效果为False（不填充）
    p = BboxConnectorPatch(
        bbox7, bbox8, loc1a=1, loc2a=2, loc1b=4, loc2b=3,
        ec="r", fc="g", fill=False)
    # 设置图形对象的裁剪功能为关闭
    p.set_clip_on(False)
    # 将创建的图形对象p添加到第四个子图(ax[3])中
    ax[3].add_patch(p)
    # 在第四个子图(ax[3])上创建一个放大镜视图，放大比例为1，位置在右上角
    axins = zoomed_inset_axes(ax[3], 1, loc='upper right')
    # 设置放大镜视图的X轴范围
    axins.set_xlim(0, 0.2)
    # 设置放大镜视图的Y轴范围
    axins.set_ylim(0, 0.2)
    # 设置放大镜视图的X轴刻度为空
    axins.xaxis.set_ticks([])
    # 设置放大镜视图的Y轴刻度为空
    axins.yaxis.set_ticks([])
    # 标记放大镜视图，连接原始图(ax[3])和放大镜视图(axins)，标记区域的填充效果为False（不填充），边框颜色为0.5灰色，填充颜色为绿色
    mark_inset(ax[3], axins, loc1=2, loc2=4, fc="g", ec="0.5", fill=False)
# Update style when regenerating the test image
# 用于比较图片，检查样式是否需要更新
@image_comparison(['zoomed_axes.png', 'inverted_zoomed_axes.png'],
                  style=('classic', '_classic_test_patch'),
                  tol=0.02 if platform.machine() == 'arm64' else 0)
def test_zooming_with_inverted_axes():
    # 创建一个新的图形和坐标轴
    fig, ax = plt.subplots()
    # 在坐标轴上绘制一条线
    ax.plot([1, 2, 3], [1, 2, 3])
    # 设置坐标轴范围
    ax.axis([1, 3, 1, 3])
    # 创建一个缩放插图的坐标轴，位于原坐标轴的右下角，放大倍数为2.5倍
    inset_ax = zoomed_inset_axes(ax, zoom=2.5, loc='lower right')
    # 设置插图坐标轴的范围
    inset_ax.axis([1.1, 1.4, 1.1, 1.4])

    # 创建另一个新的图形和坐标轴
    fig, ax = plt.subplots()
    # 在坐标轴上绘制一条线
    ax.plot([1, 2, 3], [1, 2, 3])
    # 设置坐标轴范围，这次是反向的
    ax.axis([3, 1, 3, 1])
    # 创建一个缩放插图的坐标轴，位于原坐标轴的右下角，放大倍数为2.5倍
    inset_ax = zoomed_inset_axes(ax, zoom=2.5, loc='lower right')
    # 设置插图坐标轴的范围
    inset_ax.axis([1.4, 1.1, 1.4, 1.1])


# Update style when regenerating the test image
# 用于比较图片，检查样式是否需要更新
@image_comparison(['anchored_direction_arrows.png'],
                  tol=0 if platform.machine() == 'x86_64' else 0.01,
                  style=('classic', '_classic_test_patch'))
def test_anchored_direction_arrows():
    # 创建一个新的图形和坐标轴
    fig, ax = plt.subplots()
    # 在坐标轴上显示一个空的10x10矩阵
    ax.imshow(np.zeros((10, 10)), interpolation='nearest')

    # 创建一个锚定方向箭头对象，放置在坐标轴的转换坐标系中，显示X和Y方向
    simple_arrow = AnchoredDirectionArrows(ax.transAxes, 'X', 'Y')
    # 将箭头对象添加到坐标轴中
    ax.add_artist(simple_arrow)


# Update style when regenerating the test image
# 用于比较图片，检查样式是否需要更新
@image_comparison(['anchored_direction_arrows_many_args.png'],
                  style=('classic', '_classic_test_patch'))
def test_anchored_direction_arrows_many_args():
    # 创建一个新的图形和坐标轴
    fig, ax = plt.subplots()
    # 在坐标轴上显示一个全为1的10x10矩阵
    ax.imshow(np.ones((10, 10)))

    # 创建一个具有多个参数的锚定方向箭头对象
    direction_arrows = AnchoredDirectionArrows(
            ax.transAxes, 'A', 'B', loc='upper right', color='red',
            aspect_ratio=-0.5, pad=0.6, borderpad=2, frameon=True, alpha=0.7,
            sep_x=-0.06, sep_y=-0.08, back_length=0.1, head_width=9,
            head_length=10, tail_width=5)
    # 将箭头对象添加到坐标轴中
    ax.add_artist(direction_arrows)


# 测试坐标轴定位器的位置
def test_axes_locatable_position():
    # 创建一个新的图形和坐标轴
    fig, ax = plt.subplots()
    # 创建一个坐标轴定位器
    divider = make_axes_locatable(ax)
    # 在上下文中设置特定的rc参数
    with mpl.rc_context({"figure.subplot.wspace": 0.02}):
        # 添加一个右侧的轴，大小为整个图形高度的5%
        cax = divider.append_axes('right', size='5%')
    # 绘制图形
    fig.canvas.draw()
    # 断言右侧轴的宽度是否接近预期值
    assert np.isclose(cax.get_position(original=False).width,
                      0.03621495327102808)


@image_comparison(['image_grid_each_left_label_mode_all.png'], style='mpl20',
                  savefig_kwarg={'bbox_inches': 'tight'})
def test_image_grid_each_left_label_mode_all():
    # 创建一个包含100个元素的数组，reshape成10x10的矩阵
    imdata = np.arange(100).reshape((10, 10))

    # 创建一个新的图形
    fig = plt.figure(1, (3, 3))
    # 创建一个图像网格，包含3行2列，每个子图都有自己的颜色条、标签模式为'all'
    grid = ImageGrid(fig, (1, 1, 1), nrows_ncols=(3, 2), axes_pad=(0.5, 0.3),
                     cbar_mode="each", cbar_location="left", cbar_size="15%",
                     label_mode="all")
    # 断言网格的分割器类型为SubplotDivider
    assert isinstance(grid.get_divider(), SubplotDivider)
    # 断言网格的坐标轴间隔与预期值相符
    assert grid.get_axes_pad() == (0.5, 0.3)
    # 断言网格的纵横比默认为True
    assert grid.get_aspect()  # True by default for ImageGrid
    # 对于每个子图和其对应的颜色条轴，绘制图像并添加颜色条
    for ax, cax in zip(grid, grid.cbar_axes):
        im = ax.imshow(imdata, interpolation='none')
        cax.colorbar(im)
# 使用装饰器 image_comparison，比较测试函数的输出图像与预期的参考图像，保存图像的配置参数设置为 bbox_inches='tight'
@image_comparison(['image_grid_single_bottom_label_mode_1.png'], style='mpl20',
                  savefig_kwarg={'bbox_inches': 'tight'})
# 定义测试函数 test_image_grid_single_bottom
def test_image_grid_single_bottom():
    # 创建一个 10x10 的数组作为图像数据
    imdata = np.arange(100).reshape((10, 10))

    # 创建一个大小为 (2.5, 1.5) 的 Figure 对象
    fig = plt.figure(1, (2.5, 1.5))
    
    # 创建一个包含 3 个子图的 ImageGrid 对象，子图布局为 1 行 3 列
    grid = ImageGrid(fig, (0, 0, 1, 1), nrows_ncols=(1, 3),
                     axes_pad=(0.2, 0.15), cbar_mode="single",
                     cbar_location="bottom", cbar_size="10%", label_mode="1")
    
    # 断言 grid.get_divider() 返回的类型是 Divider 类型
    assert type(grid.get_divider()) is Divider
    
    # 遍历 3 个子图，分别显示图像数据 imdata，设置插值方式为 'none'
    for i in range(3):
        im = grid[i].imshow(imdata, interpolation='none')
    
    # 在颜色条轴上添加颜色条，使用最后一个子图的图像数据 im 作为颜色条参考
    grid.cbar_axes[0].colorbar(im)


# 定义测试函数 test_image_grid_label_mode_invalid
def test_image_grid_label_mode_invalid():
    # 创建一个新的 Figure 对象
    fig = plt.figure()
    
    # 使用 pytest.raises 断言捕获 ValueError 异常，并匹配异常信息 "'foo' is not a valid value for mode"
    with pytest.raises(ValueError, match="'foo' is not a valid value for mode"):
        # 创建一个 ImageGrid 对象，设置子图布局为 (2, 1)，label_mode 参数设置为 "foo"，应该触发异常
        ImageGrid(fig, (0, 0, 1, 1), (2, 1), label_mode="foo")


# 使用装饰器 image_comparison，比较测试函数的输出图像与预期的参考图像，移除图像中的文本，并设置保存图像的配置参数 bbox_inches='tight'
@image_comparison(['image_grid.png'],
                  remove_text=True, style='mpl20',
                  savefig_kwarg={'bbox_inches': 'tight'})
# 定义测试函数 test_image_grid
def test_image_grid():
    # 创建一个 10x10 的数组作为图像数据
    im = np.arange(100).reshape((10, 10))

    # 创建一个大小为 (4, 4) 的 Figure 对象
    fig = plt.figure(1, (4, 4))
    
    # 创建一个包含 4 个子图的 ImageGrid 对象，子图布局为 2 行 2 列，子图之间的间距为 0.1
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.1)
    
    # 断言 grid.get_axes_pad() 返回的子图间距为 (0.1, 0.1)
    assert grid.get_axes_pad() == (0.1, 0.1)
    
    # 遍历 4 个子图，分别显示图像数据 im，设置插值方式为 'nearest'
    for i in range(4):
        grid[i].imshow(im, interpolation='nearest')


# 定义测试函数 test_gettightbbox
def test_gettightbbox():
    # 创建一个大小为 (8, 6) 的 Figure 对象和一个 Axes 对象 ax
    fig, ax = plt.subplots(figsize=(8, 6))

    # 在 ax 上绘制一条直线
    l, = ax.plot([1, 2, 3], [0, 1, 0])

    # 创建一个倍增的插图轴对象 ax_zoom，放大倍数为 4 倍
    ax_zoom = zoomed_inset_axes(ax, 4)
    
    # 在 ax_zoom 上绘制一条直线
    ax_zoom.plot([1, 2, 3], [0, 1, 0])

    # 标记 ax 和 ax_zoom 之间的区域，设置边框颜色为 '0.3'
    mark_inset(ax, ax_zoom, loc1=1, loc2=3, fc="none", ec='0.3')

    # 移除图像的刻度和标题
    remove_ticks_and_titles(fig)
    
    # 获取 Figure 对象的紧凑边界框，使用 fig.canvas.get_renderer() 获取渲染器
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    
    # 使用 np.testing.assert_array_almost_equal 进行边界框坐标的精确比较
    np.testing.assert_array_almost_equal(bbox.extents,
                                         [-17.7, -13.9, 7.2, 5.4])


# 使用 pytest.mark.parametrize 进行参数化测试，测试不同的点击事件
@pytest.mark.parametrize("click_on", ["big", "small"])
# 使用 pytest.mark.parametrize 进行参数化测试，测试不同的 big_on_axes 和 small_on_axes 组合
@pytest.mark.parametrize("big_on_axes,small_on_axes", [
    ("gca", "gca"),
    ("host", "host"),
    ("host", "parasite"),
    ("parasite", "host"),
    ("parasite", "parasite")
])
# 定义测试函数 test_picking_callbacks_overlap
def test_picking_callbacks_overlap(big_on_axes, small_on_axes, click_on):
    """Test pick events on normal, host or parasite axes."""
    # 创建两个矩形对象 big 和 small，并为它们设置 picker 属性为 5
    big = plt.Rectangle((0.25, 0.25), 0.5, 0.5, picker=5)
    small = plt.Rectangle((0.4, 0.4), 0.2, 0.2, facecolor="r", picker=5)
    
    # 事件接收器函数 on_pick，将事件添加到 received_events 列表中
    received_events = []
    def on_pick(event):
        received_events.append(event)
    
    # 连接 'pick_event' 事件到当前图形的 canvas 上的 on_pick 函数
    plt.gcf().canvas.mpl_connect('pick_event', on_pick)
    # 将大矩形和小矩形所在的轴保存在元组中，以备后续使用
    rectangles_on_axes = (big_on_axes, small_on_axes)
    
    # 设置不同类型的轴，用于绘图
    axes = {"gca": None, "host": None, "parasite": None}
    
    # 如果大矩形在指定轴上，获取当前的轴对象
    if "gca" in rectangles_on_axes:
        axes["gca"] = plt.gca()
    
    # 如果大矩形或者小矩形在指定轴上，创建主轴对象并设置寄生轴
    if "host" in rectangles_on_axes or "parasite" in rectangles_on_axes:
        axes["host"] = host_subplot(111)
        axes["parasite"] = axes["host"].twin()
    
    # 在对应的轴上添加大矩形和小矩形
    axes[big_on_axes].add_patch(big)
    axes[small_on_axes].add_patch(small)
    
    # 模拟使用鼠标点击事件
    if click_on == "big":
        click_axes = axes[big_on_axes]
        axes_coords = (0.3, 0.3)
    else:
        click_axes = axes[small_on_axes]
        axes_coords = (0.5, 0.5)
    
    # 实际情况下，鼠标事件只会在主轴上发生，不会在寄生轴上发生
    if click_axes is axes["parasite"]:
        click_axes = axes["host"]
    
    # 将轴上的坐标转换为画布上的坐标，并创建一个模拟的鼠标事件
    (x, y) = click_axes.transAxes.transform(axes_coords)
    m = MouseEvent("button_press_event", click_axes.figure.canvas, x, y,
                   button=1)
    
    # 模拟鼠标点击操作
    click_axes.pick(m)
    
    # 检查接收到的事件数量是否符合预期
    expected_n_events = 2 if click_on == "small" else 1
    assert len(received_events) == expected_n_events
    
    # 获取事件中的矩形对象并进行断言检查
    event_rects = [event.artist for event in received_events]
    assert big in event_rects
    
    # 如果点击的是小矩形，则继续断言小矩形是否在事件列表中
    if click_on == "small":
        assert small in event_rects
# 使用 image_comparison 装饰器比较生成的图像与 anchord_artists.png 是否一致，移除文本，使用 mpl20 风格
@image_comparison(['anchored_artists.png'], remove_text=True, style='mpl20')
def test_anchored_artists():
    # 创建一个大小为 (3, 3) 的图像和坐标轴
    fig, ax = plt.subplots(figsize=(3, 3))
    # 创建一个 AnchoredDrawingArea 对象，位于右上角，不带边框
    ada = AnchoredDrawingArea(40, 20, 0, 0, loc='upper right', pad=0., frameon=False)
    # 创建一个圆形对象 p1，位于 (10, 10)，半径为 10
    p1 = Circle((10, 10), 10)
    ada.drawing_area.add_artist(p1)
    # 创建一个圆形对象 p2，位于 (30, 10)，半径为 5，填充颜色为红色
    p2 = Circle((30, 10), 5, fc="r")
    ada.drawing_area.add_artist(p2)
    ax.add_artist(ada)

    # 创建一个 AnchoredAuxTransformBox 对象，位于左上角
    box = AnchoredAuxTransformBox(ax.transData, loc='upper left')
    # 创建一个椭圆对象 el，位于原点，宽度为 0.1，高度为 0.4，角度为 30 度，颜色为青色
    el = Ellipse((0, 0), width=0.1, height=0.4, angle=30, color='cyan')
    box.drawing_area.add_artist(el)
    ax.add_artist(box)

    # 创建一个 AnchoredEllipse 对象，位于左下角，宽度为 0.1，高度为 0.25，角度为 -60 度
    ae = AnchoredEllipse(ax.transData, width=0.1, height=0.25, angle=-60, loc='lower left', pad=0.5, borderpad=0.4, frameon=True)
    ax.add_artist(ae)

    # 创建一个 AnchoredSizeBar 对象，位于右下角，长度为 0.2，标签为 "0.2 units"，填充颜色为绿色
    asb = AnchoredSizeBar(ax.transData, 0.2, r"0.2 units", loc='lower right', pad=0.3, borderpad=0.4, sep=4, fill_bar=True, frameon=False, label_top=True, prop={'size': 20}, size_vertical=0.05, color='green')
    ax.add_artist(asb)


# 测试 HBoxDivider 类
def test_hbox_divider():
    # 创建两个 4x5 和 5x4 的数组
    arr1 = np.arange(20).reshape((4, 5))
    arr2 = np.arange(20).reshape((5, 4))

    # 创建一个包含两个子图的图像
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(arr1)
    ax2.imshow(arr2)

    pad = 0.5  # inches.
    # 创建一个水平分割线
    divider = HBoxDivider(fig, 111, horizontal=[Size.AxesX(ax1), Size.Fixed(pad), Size.AxesX(ax2)], vertical=[Size.AxesY(ax1), Size.Scaled(1), Size.AxesY(ax2)])
    ax1.set_axes_locator(divider.new_locator(0))
    ax2.set_axes_locator(divider.new_locator(2))

    fig.canvas.draw()
    p1 = ax1.get_position()
    p2 = ax2.get_position()
    assert p1.height == p2.height
    assert p2.width / p1.width == pytest.approx((4 / 5) ** 2)


# 测试 VBoxDivider 类
def test_vbox_divider():
    # 创建两个 4x5 和 5x4 的数组
    arr1 = np.arange(20).reshape((4, 5))
    arr2 = np.arange(20).reshape((5, 4))

    # 创建一个包含两个子图的图像
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(arr1)
    ax2.imshow(arr2)

    pad = 0.5  # inches.
    # 创建一个垂直分割线
    divider = VBoxDivider(fig, 111, horizontal=[Size.AxesX(ax1), Size.Scaled(1), Size.AxesX(ax2)], vertical=[Size.AxesY(ax1), Size.Fixed(pad), Size.AxesY(ax2)])
    ax1.set_axes_locator(divider.new_locator(0))
    ax2.set_axes_locator(divider.new_locator(2))

    fig.canvas.draw()
    p1 = ax1.get_position()
    p2 = ax2.get_position()
    assert p1.width == p2.width
    assert p1.height / p2.height == pytest.approx((4 / 5) ** 2)


# 测试 AxesGrid 类
def test_axes_class_tuple():
    fig = plt.figure()
    axes_class = (mpl_toolkits.axes_grid1.mpl_axes.Axes, {})
    gr = AxesGrid(fig, 111, nrows_ncols=(1, 1), axes_class=axes_class)


# 测试 Grid axes_all, axes_row 和 axes_column 之间的关系
def test_grid_axes_lists():
    """Test Grid axes_all, axes_row and axes_column relationship."""
    # 创建一个新的空白图形对象
    fig = plt.figure()
    # 在图形对象上创建一个网格，1行1列，图形在第一个位置
    grid = Grid(fig, 111, (2, 3), direction="row")
    # 断言网格对象与其所有子图对象数组相等
    assert_array_equal(grid, grid.axes_all)
    # 断言网格对象的行子图数组与列子图数组的转置相等
    assert_array_equal(grid.axes_row, np.transpose(grid.axes_column))
    # 断言网格对象与按行展平的子图数组相等，附带描述信息"row"
    assert_array_equal(grid, np.ravel(grid.axes_row), "row")
    # 断言网格对象的几何形状为(2, 3)
    assert grid.get_geometry() == (2, 3)
    # 使用方向为"column"重新创建网格对象
    grid = Grid(fig, 111, (2, 3), direction="column")
    # 断言网格对象与按列展平的子图数组相等，附带描述信息"column"
    assert_array_equal(grid, np.ravel(grid.axes_column), "column")
# 使用 pytest 的 parametrize 装饰器来定义多个测试用例，方便测试不同的输入参数
@pytest.mark.parametrize('direction', ('row', 'column'))
def test_grid_axes_position(direction):
    """Test positioning of the axes in Grid."""
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上创建一个 Grid 对象，指定布局和方向
    grid = Grid(fig, 111, (2, 2), direction=direction)
    # 获取每个轴的定位器对象列表
    loc = [ax.get_axes_locator() for ax in np.ravel(grid.axes_row)]
    # 检查在 x 方向上的定位器参数
    assert loc[1].args[0] > loc[0].args[0]
    assert loc[0].args[0] == loc[2].args[0]
    assert loc[3].args[0] == loc[1].args[0]
    # 检查在 y 方向上的定位器参数
    assert loc[2].args[1] < loc[0].args[1]
    assert loc[0].args[1] == loc[1].args[1]
    assert loc[3].args[1] == loc[2].args[1]


# 使用 pytest 的 parametrize 装饰器来定义多个测试用例，测试 Grid 对象的错误情况
@pytest.mark.parametrize('rect, ngrids, error, message', (
    ((1, 1), None, TypeError, "Incorrect rect format"),
    (111, -1, ValueError, "ngrids must be positive"),
    (111, 7, ValueError, "ngrids must be positive"),
))
def test_grid_errors(rect, ngrids, error, message):
    # 创建一个新的图形对象
    fig = plt.figure()
    # 使用 pytest.raises 来检测 Grid 对象在不同错误情况下是否会引发异常
    with pytest.raises(error, match=message):
        Grid(fig, rect, (2, 3), ngrids=ngrids)


# 使用 pytest 的 parametrize 装饰器来定义多个测试用例，测试 Divider 对象的错误情况
@pytest.mark.parametrize('anchor, error, message', (
    (None, TypeError, "anchor must be str"),
    ("CC", ValueError, "'CC' is not a valid value for anchor"),
    ((1, 1, 1), TypeError, "anchor must be str"),
))
def test_divider_errors(anchor, error, message):
    # 创建一个新的图形对象
    fig = plt.figure()
    # 使用 pytest.raises 来检测 Divider 对象在不同错误情况下是否会引发异常
    with pytest.raises(error, match=message):
        Divider(fig, [0, 0, 1, 1], [Size.Fixed(1)], [Size.Fixed(1)],
                anchor=anchor)


# 使用 image_comparison 装饰器来测试 mark_inset 函数绘制的子图在不同样式下的一致性
@check_figures_equal(extensions=["png"])
def test_mark_inset_unstales_viewlim(fig_test, fig_ref):
    # 创建两个图形对象，每个包含两个子图
    inset, full = fig_test.subplots(1, 2)
    full.plot([0, 5], [0, 5])
    inset.set(xlim=(1, 2), ylim=(1, 2))
    # 调用 mark_inset 函数在 full 子图中标记 inset 子图的位置
    mark_inset(full, inset, 1, 4)

    inset, full = fig_ref.subplots(1, 2)
    full.plot([0, 5], [0, 5])
    inset.set(xlim=(1, 2), ylim=(1, 2))
    mark_inset(full, inset, 1, 4)
    # 手动更新 fig_ref 中 full 子图的视图限制
    fig_ref.canvas.draw()


# 测试 make_axes_area_auto_adjustable 函数是否正确调整子图的边界框
def test_auto_adjustable():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上添加一个坐标轴对象
    ax = fig.add_axes([0, 0, 1, 1])
    pad = 0.1
    # 调用 make_axes_area_auto_adjustable 函数自动调整坐标轴区域的边界框
    make_axes_area_auto_adjustable(ax, pad=pad)
    fig.canvas.draw()
    # 获取调整后的紧凑边界框对象
    tbb = ax.get_tightbbox()
    # 检查紧凑边界框的四个边界是否符合预期值
    assert tbb.x0 == pytest.approx(pad * fig.dpi)
    assert tbb.x1 == pytest.approx(fig.bbox.width - pad * fig.dpi)
    assert tbb.y0 == pytest.approx(pad * fig.dpi)
    assert tbb.y1 == pytest.approx(fig.bbox.height - pad * fig.dpi)


# 使用 image_comparison 装饰器来测试 RGBAxes 对象绘制 RGB 图像的一致性
@image_comparison(['rgb_axes.png'], remove_text=True,
                  style=('classic', '_classic_test_patch'))
def test_rgb_axes():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上创建一个 RGBAxes 对象，并绘制随机的 RGB 图像
    ax = RGBAxes(fig, (0.1, 0.1, 0.8, 0.8), pad=0.1)
    rng = np.random.default_rng(19680801)
    r = rng.random((5, 5))
    g = rng.random((5, 5))
    b = rng.random((5, 5))
    ax.imshow_rgb(r, g, b, interpolation='none')


# 使用 image_comparison 装饰器来测试 mark_inset 函数中 inset 子图位置的一致性
@image_comparison(['insetposition.png'], remove_text=True,
                  style=('classic', '_classic_test_patch'))
def test_insetposition():
    # 创建一个新的图形和子图对象，尺寸为2x2
    fig, ax = plt.subplots(figsize=(2, 2))
    # 在当前图形中创建一个新的坐标轴，覆盖整个图形
    ax_ins = plt.axes([0, 0, 1, 1])
    # 引发 MatplotlibDeprecationWarning 警告时使用 pytest 捕获警告
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        # 创建插入位置对象 ip，相对于 ax 在指定的位置和尺寸
        ip = InsetPosition(ax, [0.2, 0.25, 0.5, 0.4])
    # 设置 ax_ins 的坐标轴定位器为 ip
    ax_ins.set_axes_locator(ip)


@image_comparison(['imagegrid_cbar_mode.png'],
                  remove_text=True, style='mpl20', tol=0.3)
def test_imagegrid_cbar_mode_edge():
    # 创建一个新的图形对象，尺寸为18x9
    fig = plt.figure(figsize=(18, 9))

    # 定义子图的位置，方向和颜色栏位置
    positions = (241, 242, 243, 244, 245, 246, 247, 248)
    directions = ['row']*4 + ['column']*4
    cbar_locations = ['left', 'right', 'top', 'bottom']*2

    # 遍历每个子图的位置，方向和颜色栏位置
    for position, direction, location in zip(
            positions, directions, cbar_locations):
        # 在 fig 上创建一个 ImageGrid 对象
        grid = ImageGrid(fig, position,
                         nrows_ncols=(2, 2),
                         direction=direction,
                         cbar_location=location,
                         cbar_size='20%',
                         cbar_mode='edge')
        # 获取每个子图对象
        ax1, ax2, ax3, ax4 = grid

        # 在每个子图上显示不同的图像
        ax1.imshow(arr, cmap='nipy_spectral')
        ax2.imshow(arr.T, cmap='hot')
        ax3.imshow(np.hypot(arr, arr.T), cmap='jet')
        ax4.imshow(np.arctan2(arr, arr.T), cmap='hsv')

        # 清空每个子图的颜色栏轴，然后重新创建新的颜色栏
        for ax in grid:
            ax.cax.cla()
            cb = ax.cax.colorbar(ax.images[0])


def test_imagegrid():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上创建一个 1x1 的 ImageGrid 对象
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 1))
    # 获取第一个子图对象
    ax = grid[0]
    # 在子图上显示图像，并使用对数标准化
    im = ax.imshow([[1, 2]], norm=mpl.colors.LogNorm())
    # 创建并显示颜色栏
    cb = ax.cax.colorbar(im)
    # 断言颜色栏的 locator 是 LogLocator 类的实例
    assert isinstance(cb.locator, mticker.LogLocator)


def test_removal():
    # 导入必要的库
    import matplotlib.pyplot as plt
    import mpl_toolkits.axisartist as AA
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上创建一个主子图
    ax = host_subplot(111, axes_class=AA.Axes, figure=fig)
    # 在子图上填充一条曲线
    col = ax.fill_between(range(5), 0, range(5))
    # 更新图形画布
    fig.canvas.draw()
    # 移除之前填充的曲线
    col.remove()
    # 再次更新图形画布
    fig.canvas.draw()


@image_comparison(['anchored_locator_base_call.png'], style="mpl20")
def test_anchored_locator_base_call():
    # 创建一个新的图形对象，尺寸为3x3
    fig = plt.figure(figsize=(3, 3))
    # 在图形上创建两个子图对象
    fig1, fig2 = fig.subfigures(nrows=2, ncols=1)

    # 在第一个子图上创建一个新的坐标轴对象
    ax = fig1.subplots()
    # 设置坐标轴的属性
    ax.set(aspect=1, xlim=(-15, 15), ylim=(-20, 5))
    ax.set(xticks=[], yticks=[])

    # 获取示例数据 Z
    Z = cbook.get_sample_data("axes_grid/bivariate_normal.npy")
    extent = (-3, 4, -4, 3)

    # 在 ax 上创建一个放大的插图坐标轴对象
    axins = zoomed_inset_axes(ax, zoom=2, loc="upper left")
    axins.set(xticks=[], yticks=[])

    # 在插图坐标轴上显示图像 Z
    axins.imshow(Z, extent=extent, origin="lower")


def test_grid_with_axes_class_not_overriding_axis():
    # 在一个新的图形对象上创建一个网格，使用普通的 Axes 类
    Grid(plt.figure(), 111, (2, 2), axes_class=mpl.axes.Axes)
    # 在一个新的图形对象上创建一个 RGBAxes，使用普通的 Axes 类
    RGBAxes(plt.figure(), 111, axes_class=mpl.axes.Axes)
```