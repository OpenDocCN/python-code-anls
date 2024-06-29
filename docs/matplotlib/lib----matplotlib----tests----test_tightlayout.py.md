# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_tightlayout.py`

```
# 导入警告模块，用于管理警告信息
import warnings

# 导入 NumPy 库，并导入 assert_array_equal 函数用于数组比较
import numpy as np
from numpy.testing import assert_array_equal

# 导入 Pytest 测试框架
import pytest

# 导入 Matplotlib 库及其相关模块
import matplotlib as mpl
# 导入 Matplotlib 的图像比较装饰器
from matplotlib.testing.decorators import image_comparison
# 导入 Matplotlib 的绘图接口
import matplotlib.pyplot as plt
# 导入 Matplotlib 的偏移框和绘图区域
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
# 导入 Matplotlib 的图形对象 Rectangle
from matplotlib.patches import Rectangle


# 定义一个示例绘图函数，绘制简单的折线图
def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)  # 设置坐标轴刻度参数
    ax.set_xlabel('x-label', fontsize=fontsize)  # 设置 x 轴标签
    ax.set_ylabel('y-label', fontsize=fontsize)  # 设置 y 轴标签
    ax.set_title('Title', fontsize=fontsize)  # 设置图表标题


# 图像比较测试函数，测试单个子图的 tight_layout
@image_comparison(['tight_layout1'], tol=1.9)
def test_tight_layout1():
    """Test tight_layout for a single subplot."""
    fig, ax = plt.subplots()
    example_plot(ax, fontsize=24)  # 调用示例绘图函数
    plt.tight_layout()  # 自动调整子图布局


# 图像比较测试函数，测试多个子图的 tight_layout
@image_comparison(['tight_layout2'])
def test_tight_layout2():
    """Test tight_layout for multiple subplots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    example_plot(ax1)  # 调用示例绘图函数，绘制子图1
    example_plot(ax2)  # 调用示例绘图函数，绘制子图2
    example_plot(ax3)  # 调用示例绘图函数，绘制子图3
    example_plot(ax4)  # 调用示例绘图函数，绘制子图4
    plt.tight_layout()  # 自动调整子图布局


# 图像比较测试函数，测试多个子图的 tight_layout
@image_comparison(['tight_layout3'])
def test_tight_layout3():
    """Test tight_layout for multiple subplots."""
    ax1 = plt.subplot(221)  # 创建第1个子图
    ax2 = plt.subplot(223)  # 创建第2个子图
    ax3 = plt.subplot(122)  # 创建第3个子图
    example_plot(ax1)  # 调用示例绘图函数，绘制子图1
    example_plot(ax2)  # 调用示例绘图函数，绘制子图2
    example_plot(ax3)  # 调用示例绘图函数，绘制子图3
    plt.tight_layout()  # 自动调整子图布局


# 图像比较测试函数，测试 subplot2grid 的 tight_layout
@image_comparison(['tight_layout4'], freetype_version=('2.5.5', '2.6.1'),
                  tol=0.015)
def test_tight_layout4():
    """Test tight_layout for subplot2grid."""
    ax1 = plt.subplot2grid((3, 3), (0, 0))  # 创建第1个子图
    ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)  # 创建第2个子图
    ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)  # 创建第3个子图
    ax4 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)  # 创建第4个子图
    example_plot(ax1)  # 调用示例绘图函数，绘制子图1
    example_plot(ax2)  # 调用示例绘图函数，绘制子图2
    example_plot(ax3)  # 调用示例绘图函数，绘制子图3
    example_plot(ax4)  # 调用示例绘图函数，绘制子图4
    plt.tight_layout()  # 自动调整子图布局


# 图像比较测试函数，测试图像的 tight_layout
@image_comparison(['tight_layout5'])
def test_tight_layout5():
    """Test tight_layout for image."""
    ax = plt.subplot()  # 创建子图
    arr = np.arange(100).reshape((10, 10))  # 创建一个 10x10 的数组
    ax.imshow(arr, interpolation="none")  # 在子图上显示图像
    plt.tight_layout()  # 自动调整子图布局


# 图像比较测试函数，测试 gridspec 的 tight_layout
@image_comparison(['tight_layout6'])
def test_tight_layout6():
    """Test tight_layout for gridspec."""

    # This raises warnings since tight layout cannot
    # do this fully automatically. But the test is
    # correct since the layout is manually edited
    # 这里会引发警告，因为 tight layout 无法完全自动化完成。
    # 但是测试是正确的，因为布局是手动编辑的。
    # 忽略警告信息，以便在绘制过程中不显示用户警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        
        # 创建一个新的图形对象
        fig = plt.figure()

        # 创建一个2行1列的网格布局对象
        gs1 = mpl.gridspec.GridSpec(2, 1)
        
        # 在图形上添加子图ax1和ax2，分别位于gs1的第0行和第1行
        ax1 = fig.add_subplot(gs1[0])
        ax2 = fig.add_subplot(gs1[1])

        # 在ax1和ax2上绘制示例图
        example_plot(ax1)
        example_plot(ax2)

        # 调整gs1的布局，使其填充整个fig的指定区域
        gs1.tight_layout(fig, rect=[0, 0, 0.5, 1])

        # 创建一个3行1列的网格布局对象gs2
        gs2 = mpl.gridspec.GridSpec(3, 1)

        # 遍历gs2的每一个子图布局ss，在fig上添加对应的子图ax，并在每个ax上绘制示例图
        for ss in gs2:
            ax = fig.add_subplot(ss)
            example_plot(ax)
            # 设置每个子图的标题为空字符串
            ax.set_title("")
            # 设置每个子图的x轴标签为空字符串
            ax.set_xlabel("")

        # 在整个fig的右半部分（0.5到1之间）调整gs2的布局，设置水平间距为0.45
        gs2.tight_layout(fig, rect=[0.5, 0, 1, 1], h_pad=0.45)

        # 计算gs1和gs2布局的顶部和底部位置，取最小顶部和最大底部作为整体布局的顶部和底部位置
        top = min(gs1.top, gs2.top)
        bottom = max(gs1.bottom, gs2.bottom)

        # 调整gs1的布局，使其填充整个fig的指定区域，保持相对位置不变
        gs1.tight_layout(fig, rect=[None, 0 + (bottom-gs1.bottom),
                                    0.5, 1 - (gs1.top-top)])
        
        # 调整gs2的布局，使其填充整个fig的指定区域，保持相对位置不变，设置水平间距为0.45
        gs2.tight_layout(fig, rect=[0.5, 0 + (bottom-gs2.bottom),
                                    None, 1 - (gs2.top-top)],
                         h_pad=0.45)
@image_comparison(['tight_layout7'], tol=1.9)
def test_tight_layout7():
    """Test for tight_layout with left and right titles."""
    # 设置字体大小为 24
    fontsize = 24
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 绘制简单的线图
    ax.plot([1, 2])
    # 设置定位器参数，将轴分成 3 个区间
    ax.locator_params(nbins=3)
    # 设置 x 轴标签和 y 轴标签的字体大小和内容
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    # 设置左侧标题，位置在左边，字体大小为 24
    ax.set_title('Left Title', loc='left', fontsize=fontsize)
    # 设置右侧标题，位置在右边，字体大小为 24
    ax.set_title('Right Title', loc='right', fontsize=fontsize)
    # 自动调整子图布局
    plt.tight_layout()


@image_comparison(['tight_layout8'], tol=0.005)
def test_tight_layout8():
    """Test automatic use of tight_layout."""
    # 创建一个新的图形对象
    fig = plt.figure()
    # 设置图形对象的布局引擎为 'tight'，设置内边距为 0.1
    fig.set_layout_engine(layout='tight', pad=0.1)
    # 添加一个子图到图形对象中
    ax = fig.add_subplot()
    # 在子图上绘制一个示例图形，设置字体大小为 24
    example_plot(ax, fontsize=24)
    # 绘制图形但不进行渲染
    fig.draw_without_rendering()


@image_comparison(['tight_layout9'])
def test_tight_layout9():
    """Test tight_layout for non-visible subplots."""
    # 创建一个 2x2 的子图网格
    f, axarr = plt.subplots(2, 2)
    # 设置子图 [1][1] 不可见
    axarr[1][1].set_visible(False)
    # 自动调整子图布局
    plt.tight_layout()


def test_outward_ticks():
    """Test automatic use of tight_layout."""
    # 创建一个新的图形对象
    fig = plt.figure()
    # 添加一个 2x2 的子图网格，并在左上角添加子图
    ax = fig.add_subplot(221)
    # 设置 x 轴和 y 轴刻度的参数，方向为 'out'，长度为 16，宽度为 3
    ax.xaxis.set_tick_params(tickdir='out', length=16, width=3)
    ax.yaxis.set_tick_params(tickdir='out', length=16, width=3)
    # 设置次要刻度的参数，方向为 'out'，长度为 32，宽度为 3，显示次要刻度，并在位置为 0 处添加次要刻度
    ax.xaxis.set_tick_params(
        tickdir='out', length=32, width=3, tick1On=True, which='minor')
    ax.yaxis.set_tick_params(
        tickdir='out', length=32, width=3, tick1On=True, which='minor')
    ax.xaxis.set_ticks([0], minor=True)
    ax.yaxis.set_ticks([0], minor=True)
    # 添加一个 2x2 的子图网格，并在右上角添加子图
    ax = fig.add_subplot(222)
    # 设置 x 轴和 y 轴刻度的参数，方向为 'in'，长度为 32，宽度为 3
    ax.xaxis.set_tick_params(tickdir='in', length=32, width=3)
    ax.yaxis.set_tick_params(tickdir='in', length=32, width=3)
    # 添加一个 2x2 的子图网格，并在左下角添加子图
    ax = fig.add_subplot(223)
    # 设置 x 轴和 y 轴刻度的参数，方向为 'inout'，长度为 32，宽度为 3
    ax.xaxis.set_tick_params(tickdir='inout', length=32, width=3)
    ax.yaxis.set_tick_params(tickdir='inout', length=32, width=3)
    # 添加一个 2x2 的子图网格，并在右下角添加子图
    ax = fig.add_subplot(224)
    # 设置 x 轴和 y 轴刻度的参数，方向为 'out'，长度为 32，宽度为 3
    ax.xaxis.set_tick_params(tickdir='out', length=32, width=3)
    ax.yaxis.set_tick_params(tickdir='out', length=32, width=3)
    # 自动调整子图布局
    plt.tight_layout()
    # 以下数值经过视觉检查，确认它们对应了正确的紧凑布局，考虑了刻度的影响
    expected = [
        [[0.091, 0.607], [0.433, 0.933]],
        [[0.579, 0.607], [0.922, 0.933]],
        [[0.091, 0.140], [0.433, 0.466]],
        [[0.579, 0.140], [0.922, 0.466]],
    ]
    for nn, ax in enumerate(fig.axes):
        # 断言子图位置的数组与预期的位置数组近似相等
        assert_array_equal(np.round(ax.get_position().get_points(), 3),
                           expected[nn])


def add_offsetboxes(ax, size=10, margin=.1, color='black'):
    """
    Surround ax with OffsetBoxes
    """
    # 定义边框的大小和颜色
    m, mp = margin, 1+margin
    # 定义边框的锚点位置
    anchor_points = [(-m, -m), (-m, .5), (-m, mp),
                     (mp, .5), (.5, mp), (mp, mp),
                     (.5, -m), (mp, -m), (.5, -m)]
    # 遍历给定的锚点列表，每个锚点都是一个二维坐标元组
    for point in anchor_points:
        # 创建一个指定大小的绘图区域对象
        da = DrawingArea(size, size)
        # 创建一个背景矩形对象，用指定的颜色填充
        background = Rectangle((0, 0), width=size,
                               height=size,
                               facecolor=color,
                               edgecolor='None',
                               linewidth=0,
                               antialiased=False)
        # 将背景矩形添加到绘图区域中
        da.add_artist(background)

        # 创建一个偏移框对象，用于在绘图区域中放置 anchored box
        anchored_box = AnchoredOffsetbox(
            loc='center',
            child=da,
            pad=0.,
            frameon=False,
            bbox_to_anchor=point,  # 指定偏移框的位置
            bbox_transform=ax.transAxes,  # 指定坐标系变换
            borderpad=0.
        )
        # 将 anchored box 添加到当前的坐标轴对象中
        ax.add_artist(anchored_box)
    
    # 返回最后一个添加的 anchored box 对象
    return anchored_box
@image_comparison(['tight_layout_offsetboxes1', 'tight_layout_offsetboxes2'])
# 定义一个测试函数，比较两个图像的布局是否相同
def test_tight_layout_offsetboxes():
    # 创建包含4个子图的布局
    rows = cols = 2  # 设定行数和列数均为2
    colors = ['red', 'blue', 'green', 'yellow']  # 设置颜色列表
    x = y = [0, 1]  # 设置用于绘制的数据

    def _subplots():
        # 创建子图并返回每个子图的轴对象
        _, axs = plt.subplots(rows, cols)
        axs = axs.flat
        for ax, color in zip(axs, colors):
            ax.plot(x, y, color=color)  # 在每个轴上绘制对角线
            add_offsetboxes(ax, 20, color=color)  # 在每个轴上添加偏移框
        return axs

    # 第一次调用_subplots()，并使用tight_layout进行布局优化
    axs = _subplots()
    plt.tight_layout()

    # 第二次调用_subplots()，使右侧轴的偏移框不可见，并再次使用tight_layout进行布局优化
    axs = _subplots()
    for ax in (axs[cols-1::rows]):
        for child in ax.get_children():
            if isinstance(child, AnchoredOffsetbox):
                child.set_visible(False)
    plt.tight_layout()


def test_empty_layout():
    """Test that tight layout doesn't cause an error when there are no Axes."""
    fig = plt.gcf()  # 获取当前图形对象
    fig.tight_layout()  # 对图形进行布局优化


@pytest.mark.parametrize("label", ["xlabel", "ylabel"])
def test_verybig_decorators(label):
    """Test that no warning emitted when xlabel/ylabel too big."""
    fig, ax = plt.subplots(figsize=(3, 2))  # 创建指定大小的图形和轴对象
    ax.set(**{label: 'a' * 100})  # 设置x或y标签文本为100个'a'


def test_big_decorators_horizontal():
    """Test that doesn't warn when xlabel too big."""
    fig, axs = plt.subplots(1, 2, figsize=(3, 2))  # 创建包含两个子图的图形对象
    axs[0].set_xlabel('a' * 30)  # 设置第一个子图的x标签为30个'a'
    axs[1].set_xlabel('b' * 30)  # 设置第二个子图的x标签为30个'b'


def test_big_decorators_vertical():
    """Test that doesn't warn when ylabel too big."""
    fig, axs = plt.subplots(2, 1, figsize=(3, 2))  # 创建包含两个子图的图形对象
    axs[0].set_ylabel('a' * 20)  # 设置第一个子图的y标签为20个'a'
    axs[1].set_ylabel('b' * 20)  # 设置第二个子图的y标签为20个'b'


def test_badsubplotgrid():
    # 测试不匹配子图网格时是否会发出警告而不是错误
    plt.subplot2grid((4, 5), (0, 0))  # 创建子图网格
    # 这是错误的输入：
    plt.subplot2grid((5, 5), (0, 3), colspan=3, rowspan=5)  # 创建另一个子图网格
    with pytest.warns(UserWarning):
        plt.tight_layout()  # 尝试对图形进行布局优化


def test_collapsed():
    # 测试如果装饰物占据的空间超过了可用宽度，调用tight_layout将不会应用
    fig, ax = plt.subplots(tight_layout=True)  # 创建图形和轴对象，并开启紧凑布局选项
    ax.set_xlim([0, 1])  # 设置x轴的范围
    ax.set_ylim([0, 1])  # 设置y轴的范围

    ax.annotate('BIG LONG STRING', xy=(1.25, 2), xytext=(10.5, 1.75),
                annotation_clip=False)  # 在轴上添加注释

    p1 = ax.get_position()  # 获取轴的位置信息
    with pytest.warns(UserWarning):
        plt.tight_layout()  # 尝试对图形进行布局优化
        p2 = ax.get_position()  # 再次获取轴的位置信息
        assert p1.width == p2.width  # 断言原始和优化后的轴宽度相同
    # 测试传递rect参数不会导致崩溃
    # 使用 pytest 框架进行测试时，检测是否会产生 UserWarning 警告
    with pytest.warns(UserWarning):
        # 调整图表布局，使其在指定的矩形区域内更紧凑
        plt.tight_layout(rect=[0, 0, 0.8, 0.8])
def test_suptitle():
    # 创建一个包含单个子图的新图形和轴对象，自动调整布局以确保子图之间没有重叠
    fig, ax = plt.subplots(tight_layout=True)
    # 在图形顶部添加超标题 "foo"
    st = fig.suptitle("foo")
    # 在子图上设置标题 "bar"
    t = ax.set_title("bar")
    # 绘制图形的画布
    fig.canvas.draw()
    # 断言超标题的顶端位置高于标题的底端位置
    assert st.get_window_extent().y0 > t.get_window_extent().y1


@pytest.mark.backend("pdf")
def test_non_agg_renderer(monkeypatch, recwarn):
    # 保存未修补的 RendererBase 类的 __init__ 方法的引用
    unpatched_init = mpl.backend_bases.RendererBase.__init__

    def __init__(self, *args, **kwargs):
        # 检查确保我们只实例化 PDF 渲染器来执行 PDF 紧凑布局
        assert isinstance(self, mpl.backends.backend_pdf.RendererPdf)
        unpatched_init(self, *args, **kwargs)

    # 使用 monkeypatch 修改 RendererBase 类的 __init__ 方法
    monkeypatch.setattr(mpl.backend_bases.RendererBase, "__init__", __init__)
    # 创建一个包含单个子图的新图形和轴对象
    fig, ax = plt.subplots()
    # 自动调整布局以确保子图之间没有重叠
    fig.tight_layout()


def test_manual_colorbar():
    # 这应该会警告，但不会引发异常
    # 创建一个包含两个子图的新图形对象
    fig, axes = plt.subplots(1, 2)
    # 在第二个子图上绘制散点图，并设置颜色映射
    pts = axes[1].scatter([0, 1], [0, 1], c=[1, 5])
    # 获取第二个子图的位置矩形
    ax_rect = axes[1].get_position()
    # 在图形上添加新的轴对象，用于显示颜色条
    cax = fig.add_axes(
        [ax_rect.x1 + 0.005, ax_rect.y0, 0.015, ax_rect.height]
    )
    # 在颜色条轴上添加颜色条
    fig.colorbar(pts, cax=cax)
    # 使用 pytest 的 warn 断言捕获用户警告，验证警告信息中包含"This figure includes Axes"
    with pytest.warns(UserWarning, match="This figure includes Axes"):
        # 自动调整布局以确保子图之间没有重叠
        fig.tight_layout()


def test_clipped_to_axes():
    # 确保 _fully_clipped_to_axes() 在所有投影类型的默认条件下返回 True
    # 创建一个新的图形对象，大小为 6x2
    arr = np.arange(100).reshape((10, 10))
    fig = plt.figure(figsize=(6, 2))
    # 添加三个子图，分别使用不同的投影类型
    ax1 = fig.add_subplot(131, projection='rectilinear')
    ax2 = fig.add_subplot(132, projection='mollweide')
    ax3 = fig.add_subplot(133, projection='polar')
    for ax in (ax1, ax2, ax3):
        # 关闭子图的网格
        ax.grid(False)
        # 在每个子图上绘制折线图和伪彩色图，并验证是否完全裁剪到子图内部
        h, = ax.plot(arr[:, 0])
        m = ax.pcolor(arr)
        assert h._fully_clipped_to_axes()
        assert m._fully_clipped_to_axes()
        # 修改裁剪路径以非默认条件，验证不再完全裁剪到子图内部
        rect = Rectangle((0, 0), 0.5, 0.5, transform=ax.transAxes)
        h.set_clip_path(rect)
        m.set_clip_path(rect.get_path(), rect.get_transform())
        assert not h._fully_clipped_to_axes()
        assert not m._fully_clipped_to_axes()


def test_tight_pads():
    # 测试紧凑布局的 'pad' 参数，这个功能即将被弃用
    fig, ax = plt.subplots()
    with pytest.warns(PendingDeprecationWarning,
                      match='will be deprecated'):
        # 设置图形的紧凑布局，指定 'pad' 参数为 0.15
        fig.set_tight_layout({'pad': 0.15})
    # 在不进行渲染的情况下绘制图形
    fig.draw_without_rendering()


def test_tight_kwargs():
    # 测试紧凑布局的关键字参数
    # 创建一个包含单个子图的新图形和轴对象，指定紧凑布局的 'pad' 参数为 0.15
    fig, ax = plt.subplots(tight_layout={'pad': 0.15})
    # 在不进行渲染的情况下绘制图形
    fig.draw_without_rendering()


def test_tight_toggle():
    # 测试紧凑布局的开关功能
    fig, ax = plt.subplots()
    with pytest.warns(PendingDeprecationWarning):
        # 启用紧凑布局模式
        fig.set_tight_layout(True)
        # 断言图形当前处于紧凑布局模式
        assert fig.get_tight_layout()
        # 关闭紧凑布局模式
        fig.set_tight_layout(False)
        # 断言图形当前不处于紧凑布局模式
        assert not fig.get_tight_layout()
        # 再次启用紧凑布局模式
        fig.set_tight_layout(True)
        # 断言图形当前处于紧凑布局模式
        assert fig.get_tight_layout()
```