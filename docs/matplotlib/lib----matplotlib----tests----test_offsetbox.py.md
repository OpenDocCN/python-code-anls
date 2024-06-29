# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_offsetbox.py`

```py
# 导入必要的模块和函数
from collections import namedtuple
import io

import numpy as np
from numpy.testing import assert_allclose
import pytest

# 导入 matplotlib 相关模块
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.backend_bases import MouseButton, MouseEvent

from matplotlib.offsetbox import (
    AnchoredOffsetbox, AnnotationBbox, AnchoredText, DrawingArea, HPacker,
    OffsetBox, OffsetImage, PaddedBox, TextArea, VPacker, _get_packed_offsets)


@image_comparison(['offsetbox_clipping'], remove_text=True)
def test_offsetbox_clipping():
    """
    - 创建一个绘图对象
    - 在坐标轴中心放置一个具有子绘图区域（DrawingArea）的 AnchoredOffsetbox
    - 给绘图区域（DrawingArea）设置灰色背景
    - 在绘图区域的边界上放置一条黑色线条
    - 确保黑色线条被剪裁到绘图区域的边缘
    """
    fig, ax = plt.subplots()
    size = 100
    da = DrawingArea(size, size, clip=True)  # 创建一个大小为size的绘图区域，并启用剪裁
    assert da.clip_children  # 断言绘图区域启用了子元素剪裁
    bg = mpatches.Rectangle((0, 0), size, size,
                            facecolor='#CCCCCC',
                            edgecolor='None',
                            linewidth=0)  # 创建一个灰色背景矩形
    line = mlines.Line2D([-size*.5, size*1.5], [size/2, size/2],
                         color='black',
                         linewidth=10)  # 创建一条黑色线条
    anchored_box = AnchoredOffsetbox(
        loc='center',  # 设置锚定位置为中心
        child=da,
        pad=0.,
        frameon=False,
        bbox_to_anchor=(.5, .5),  # 设置框的锚定点
        bbox_transform=ax.transAxes,  # 使用坐标变换来指定框的位置
        borderpad=0.)

    da.add_artist(bg)  # 将灰色背景添加到绘图区域
    da.add_artist(line)  # 将黑色线条添加到绘图区域
    ax.add_artist(anchored_box)  # 将锚定的偏移框添加到坐标轴
    ax.set_xlim((0, 1))  # 设置坐标轴的x轴范围
    ax.set_ylim((0, 1))  # 设置坐标轴的y轴范围


def test_offsetbox_clip_children():
    """
    - 创建一个绘图对象
    - 在坐标轴中心放置一个具有子绘图区域（DrawingArea）的 AnchoredOffsetbox
    - 给绘图区域（DrawingArea）设置灰色背景
    - 在绘图区域的边界上放置一条黑色线条
    - 确保黑色线条被剪裁到绘图区域的边缘
    - 验证绘图对象不是陈旧的
    - 将绘图区域的子元素剪裁属性设置为True
    - 断言绘图对象已经陈旧
    """
    fig, ax = plt.subplots()
    size = 100
    da = DrawingArea(size, size, clip=True)  # 创建一个大小为size的绘图区域，并启用剪裁
    bg = mpatches.Rectangle((0, 0), size, size,
                            facecolor='#CCCCCC',
                            edgecolor='None',
                            linewidth=0)  # 创建一个灰色背景矩形
    line = mlines.Line2D([-size*.5, size*1.5], [size/2, size/2],
                         color='black',
                         linewidth=10)  # 创建一条黑色线条
    anchored_box = AnchoredOffsetbox(
        loc='center',  # 设置锚定位置为中心
        child=da,
        pad=0.,
        frameon=False,
        bbox_to_anchor=(.5, .5),  # 设置框的锚定点
        bbox_transform=ax.transAxes,  # 使用坐标变换来指定框的位置
        borderpad=0.)

    da.add_artist(bg)  # 将灰色背景添加到绘图区域
    da.add_artist(line)  # 将黑色线条添加到绘图区域
    ax.add_artist(anchored_box)  # 将锚定的偏移框添加到坐标轴

    fig.canvas.draw()  # 绘制图形
    assert not fig.stale  # 断言图形对象不是陈旧的
    da.clip_children = True  # 设置绘图区域的子元素剪裁属性为True
    assert fig.stale  # 断言图形对象已经陈旧


def test_offsetbox_loc_codes():
    # 检查不同有效的字符串位置代码是否都适用于 AnchoredOffsetbox
    pass  # 此处为待实现的测试功能，暂时未提供具体实现
    # 定义一个包含位置名称和对应代码的字典
    codes = {'upper right': 1,
             'upper left': 2,
             'lower left': 3,
             'lower right': 4,
             'right': 5,
             'center left': 6,
             'center right': 7,
             'lower center': 8,
             'upper center': 9,
             'center': 10,
             }
    
    # 创建一个新的图形和轴对象，用于绘制图表
    fig, ax = plt.subplots()
    
    # 创建一个大小为100x100的绘图区域
    da = DrawingArea(100, 100)
    
    # 遍历位置和代码的字典中的每一个位置
    for code in codes:
        # 根据当前位置创建一个带有偏移框的对象
        anchored_box = AnchoredOffsetbox(loc=code, child=da)
        # 将带有偏移框的对象添加到图形轴上
        ax.add_artist(anchored_box)
    
    # 在图形上绘制出所有的子元素
    fig.canvas.draw()
def test_expand_with_tight_layout():
    # 创建一个包含单个图和轴对象的 Figure 对象
    fig, ax = plt.subplots()

    d1 = [1, 2]
    d2 = [2, 1]
    # 在同一个轴上绘制两个数据系列，并添加标签
    ax.plot(d1, label='series 1')
    ax.plot(d2, label='series 2')
    # 添加一个图例，设置为两列展开模式
    ax.legend(ncols=2, mode='expand')

    # 调整图像布局，以解决以前发生的崩溃问题
    fig.tight_layout()


@pytest.mark.parametrize('widths',
                         ([150], [150, 150, 150], [0.1], [0.1, 0.1]))
@pytest.mark.parametrize('total', (250, 100, 0, -1, None))
@pytest.mark.parametrize('sep', (250, 1, 0, -1))
@pytest.mark.parametrize('mode', ("expand", "fixed", "equal"))
def test_get_packed_offsets(widths, total, sep, mode):
    # 检查由于连续的类似问题票证（至少 #10476 和 #10784）而引发的一组（相当武断的）参数
    # 这些问题与调用高层函数（例如 `Axes.legend`）时在本函数内触发的边缘情况有关
    # 这些只是一些额外的烟雾测试。输出未经测试。
    _get_packed_offsets(widths, total, sep, mode=mode)


_Params = namedtuple('_Params', 'wd_list, total, sep, expected')


@pytest.mark.parametrize('widths, total, sep, expected', [
    _Params(  # total=None
        [3, 1, 2], total=None, sep=1, expected=(8, [0, 4, 6])),
    _Params(  # total larger than required
        [3, 1, 2], total=10, sep=1, expected=(10, [0, 4, 6])),
    _Params(  # total smaller than required
        [3, 1, 2], total=5, sep=1, expected=(5, [0, 4, 6])),
])
def test_get_packed_offsets_fixed(widths, total, sep, expected):
    # 调用 `_get_packed_offsets` 函数，使用固定模式进行测试，并验证结果
    result = _get_packed_offsets(widths, total, sep, mode='fixed')
    assert result[0] == expected[0]
    assert_allclose(result[1], expected[1])


@pytest.mark.parametrize('widths, total, sep, expected', [
    _Params(  # total=None (implicit 1)
        [.1, .1, .1], total=None, sep=None, expected=(1, [0, .45, .9])),
    _Params(  # total larger than sum of widths
        [3, 1, 2], total=10, sep=1, expected=(10, [0, 5, 8])),
    _Params(  # total smaller sum of widths: overlapping boxes
        [3, 1, 2], total=5, sep=1, expected=(5, [0, 2.5, 3])),
])
def test_get_packed_offsets_expand(widths, total, sep, expected):
    # 调用 `_get_packed_offsets` 函数，使用展开模式进行测试，并验证结果
    result = _get_packed_offsets(widths, total, sep, mode='expand')
    assert result[0] == expected[0]
    assert_allclose(result[1], expected[1])


@pytest.mark.parametrize('widths, total, sep, expected', [
    _Params(  # total larger than required
        [3, 2, 1], total=6, sep=None, expected=(6, [0, 2, 4])),
    _Params(  # total smaller sum of widths: overlapping boxes
        [3, 2, 1, .5], total=2, sep=None, expected=(2, [0, 0.5, 1, 1.5])),
    _Params(  # total larger than required
        [.5, 1, .2], total=None, sep=1, expected=(6, [0, 2, 4])),
    # total=None, sep=None 情况在下面单独测试
])
def test_get_packed_offsets_equal(widths, total, sep, expected):
    # 调用 `_get_packed_offsets` 函数，使用等宽模式进行测试，并验证结果
    result = _get_packed_offsets(widths, total, sep, mode='equal')
    assert result[0] == expected[0]
    # 使用 NumPy 的 assert_allclose 函数检查 result 列表中索引为 1 的值是否与 expected 列表中索引为 1 的值在允许误差范围内相等
    assert_allclose(result[1], expected[1])
# 抛出 ValueError 异常，因为 total 和 sep 参数不能同时为 None
def test_get_packed_offsets_equal_total_none_sep_none():
    with pytest.raises(ValueError):
        _get_packed_offsets([1, 1, 1], total=None, sep=None, mode='equal')


# 使用 pytest 参数化装饰器，为 test_picking 函数指定多个参数组合进行测试
@pytest.mark.parametrize('child_type', ['draw', 'image', 'text'])
@pytest.mark.parametrize('boxcoords',
                         ['axes fraction', 'axes pixels', 'axes points',
                          'data'])
def test_picking(child_type, boxcoords):
    # 根据 child_type 类型创建相应的 picking_child 对象
    # 'draw' 类型的处理
    if child_type == 'draw':
        picking_child = DrawingArea(5, 5)
        picking_child.add_artist(mpatches.Rectangle((0, 0), 5, 5, linewidth=0))
    # 'image' 类型的处理
    elif child_type == 'image':
        im = np.ones((5, 5))
        im[2, 2] = 0
        picking_child = OffsetImage(im)
    # 'text' 类型的处理
    elif child_type == 'text':
        picking_child = TextArea('\N{Black Square}', textprops={'fontsize': 5})
    else:
        # 如果出现未知的 child_type 类型，断言失败
        assert False, f'Unknown picking child type {child_type}'

    # 创建新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 创建 AnnotationBbox 对象，并设置其属性
    ab = AnnotationBbox(picking_child, (0.5, 0.5), boxcoords=boxcoords)
    ab.set_picker(True)
    # 将 AnnotationBbox 对象添加到坐标轴中
    ax.add_artist(ab)

    # 创建一个空列表 calls 来存储事件回调
    calls = []
    # 连接图形的 pick_event 事件，并将回调函数添加到 calls 列表中
    fig.canvas.mpl_connect('pick_event', lambda event: calls.append(event))

    # 根据 boxcoords 参数不同，计算事件发生点的坐标位置
    if boxcoords == 'axes points':
        x, y = ax.transAxes.transform_point((0, 0))
        x += 0.5 * fig.dpi / 72
        y += 0.5 * fig.dpi / 72
    elif boxcoords == 'axes pixels':
        x, y = ax.transAxes.transform_point((0, 0))
        x += 0.5
        y += 0.5
    else:
        x, y = ax.transAxes.transform_point((0.5, 0.5))
    
    # 绘制图形
    fig.canvas.draw()
    # 清空 calls 列表
    calls.clear()
    # 模拟鼠标事件，模拟左键按下事件
    MouseEvent(
        "button_press_event", fig.canvas, x, y, MouseButton.LEFT)._process()
    # 断言 calls 列表长度为 1，并且第一个事件的 artist 属性为 ab
    assert len(calls) == 1 and calls[0].artist == ab

    # 当坐标轴的限制改变足够隐藏 *xy* 点时，注释 *不应* 被事件在其原始中心点处选中
    ax.set_xlim(-1, 0)
    ax.set_ylim(-1, 0)
    fig.canvas.draw()
    calls.clear()
    MouseEvent(
        "button_press_event", fig.canvas, x, y, MouseButton.LEFT)._process()
    # 断言 calls 列表长度为 0
    assert len(calls) == 0


# 使用 image_comparison 装饰器，测试 AnchoredText 的水平对齐方式，并生成比较图像 'anchoredtext_align.png'
@image_comparison(['anchoredtext_align.png'], remove_text=True, style='mpl20')
def test_anchoredtext_horizontal_alignment():
    # 创建新的图形和坐标轴对象
    fig, ax = plt.subplots()

    # 创建三个 AnchoredText 对象，测试不同的水平对齐方式
    text0 = AnchoredText("test\ntest long text", loc="center left",
                         pad=0.2, prop={"ha": "left"})
    ax.add_artist(text0)
    text1 = AnchoredText("test\ntest long text", loc="center",
                         pad=0.2, prop={"ha": "center"})
    ax.add_artist(text1)
    text2 = AnchoredText("test\ntest long text", loc="center right",
                         pad=0.2, prop={"ha": "right"})
    ax.add_artist(text2)


# 使用 pytest 参数化装饰器，测试 AnnotationBbox 对象在不同 extent_kind 参数下的边界情况
@pytest.mark.parametrize("extent_kind", ["window_extent", "tightbbox"])
def test_annotationbbox_extents(extent_kind):
    # 更新默认的 pyplot 参数设置
    plt.rcParams.update(plt.rcParamsDefault)
    # 创建新的图形和坐标轴对象，指定图形大小和 DPI
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)

    # 设置坐标轴的范围
    ax.axis([0, 1, 0, 1])
    # 创建一个标注对象并添加到图形中，标注文本为"Annotation"，箭头指向右上方
    an1 = ax.annotate("Annotation", xy=(.9, .9), xytext=(1.1, 1.1),
                      arrowprops=dict(arrowstyle="->"), clip_on=False,
                      va="baseline", ha="left")

    # 创建一个绘图区域对象，大小为20x20，添加一个圆形到该区域
    da = DrawingArea(20, 20, 0, 0, clip=True)
    p = mpatches.Circle((-10, 30), 32)
    da.add_artist(p)

    # 创建一个包含绘图区域的注释框对象，并添加到图形中
    ab3 = AnnotationBbox(da, [.5, .5], xybox=(-0.2, 0.5), xycoords='data',
                         boxcoords="axes fraction", box_alignment=(0., .5),
                         arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab3)

    # 创建一个偏移图像对象，填充为随机的10x10数组，并将其添加到图形中
    im = OffsetImage(np.random.rand(10, 10), zoom=3)
    im.image.axes = ax
    ab6 = AnnotationBbox(im, (0.5, -.3), xybox=(0, 75),
                         xycoords='axes fraction',
                         boxcoords="offset points", pad=0.3,
                         arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab6)

    # 测试标注对象an1的边界框
    bb1 = getattr(an1, f"get_{extent_kind}")()

    # 预期的标注对象an1的边界框坐标
    target1 = [332.9, 242.8, 467.0, 298.9]
    assert_allclose(bb1.extents, target1, atol=2)

    # 测试注释框对象ab3的边界框
    bb3 = getattr(ab3, f"get_{extent_kind}")()

    # 预期的注释框对象ab3的边界框坐标
    target3 = [-17.6, 129.0, 200.7, 167.9]
    assert_allclose(bb3.extents, target3, atol=2)

    # 测试注释框对象ab6的边界框
    bb6 = getattr(ab6, f"get_{extent_kind}")()

    # 预期的注释框对象ab6的边界框坐标
    target6 = [180.0, -32.0, 230.0, 92.9]
    assert_allclose(bb6.extents, target6, atol=2)

    # 测试使用bbox_inches='tight'保存图形，确保输出的图像形状正确
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    shape = plt.imread(buf).shape
    targetshape = (350, 504, 4)
    assert_allclose(shape, targetshape, atol=2)

    # 简单的tight_layout测试，确保没有错误
    fig.canvas.draw()
    fig.tight_layout()
    fig.canvas.draw()
def test_zorder():
    # 测试 OffsetBox 类的 zorder 属性是否正确设置为 42
    assert OffsetBox(zorder=42).zorder == 42


def test_arrowprops_copied():
    # 创建一个 DrawingArea 对象 da，大小为 20x20，位于 (0, 0)，启用剪切
    da = DrawingArea(20, 20, 0, 0, clip=True)
    # 定义箭头属性字典 arrowprops
    arrowprops = {"arrowstyle": "->", "relpos": (.3, .7)}
    # 创建 AnnotationBbox 对象 ab，将 da 放置在 [.5, .5] 处，偏移为 (-0.2, 0.5)，xycoords='data'，boxcoords='axes fraction'，箭头属性为 arrowprops
    ab = AnnotationBbox(da, [.5, .5], xybox=(-0.2, 0.5), xycoords='data',
                        boxcoords="axes fraction", box_alignment=(0., .5),
                        arrowprops=arrowprops)
    # 断言 ab.arrowprops 不是 arrowprops 对象本身
    assert ab.arrowprops is not ab
    # 断言 arrowprops["relpos"] 与原始定义一致
    assert arrowprops["relpos"] == (.3, .7)


@pytest.mark.parametrize("align", ["baseline", "bottom", "top",
                                   "left", "right", "center"])
def test_packers(align):
    # 设置 DPI 以匹配点数，便于以下数学计算
    fig = plt.figure(dpi=72)
    # 获取渲染器对象
    renderer = fig.canvas.get_renderer()

    # 定义两个 DrawingArea 对象 r1 和 r2，各自大小为 x1, y1 和 x2, y2
    x1, y1 = 10, 30
    x2, y2 = 20, 60
    r1 = DrawingArea(x1, y1)
    r2 = DrawingArea(x2, y2)

    # 创建 HPacker 对象 hpacker，包含 r1 和 r2，对齐方式为 align
    hpacker = HPacker(children=[r1, r2], align=align)
    # 在渲染器上绘制 hpacker
    hpacker.draw(renderer)
    # 获取 hpacker 的边界框 bbox
    bbox = hpacker.get_bbox(renderer)
    # 获取偏移量 px, py
    px, py = hpacker.get_offset(bbox, renderer)
    # 断言边界框的大小和位置正确
    assert_allclose(bbox.bounds, (0, 0, x1 + x2, max(y1, y2)))
    
    # 根据对齐方式调整内部元素的位置
    if align in ("baseline", "left", "bottom"):
        y_height = 0
    elif align in ("right", "top"):
        y_height = y2 - y1
    elif align == "center":
        y_height = (y2 - y1) / 2
    # 断言每个子元素的偏移量正确
    assert_allclose([child.get_offset() for child in hpacker.get_children()],
                    [(px, py + y_height), (px + x1, py)])

    # 创建 VPacker 对象 vpacker，包含 r1 和 r2，对齐方式为 align
    vpacker = VPacker(children=[r1, r2], align=align)
    # 在渲染器上绘制 vpacker
    vpacker.draw(renderer)
    # 获取 vpacker 的边界框 bbox
    bbox = vpacker.get_bbox(renderer)
    # 获取偏移量 px, py
    px, py = vpacker.get_offset(bbox, renderer)
    # 断言边界框的大小和位置正确
    assert_allclose(bbox.bounds, (0, -max(y1, y2), max(x1, x2), y1 + y2))
    
    # 根据对齐方式调整内部元素的位置
    if align in ("baseline", "left", "bottom"):
        x_height = 0
    elif align in ("right", "top"):
        x_height = x2 - x1
    elif align == "center":
        x_height = (x2 - x1) / 2
    # 断言每个子元素的偏移量正确
    assert_allclose([child.get_offset() for child in vpacker.get_children()],
                    [(px + x_height, py), (px, py - y2)])


def test_paddedbox_default_values():
    # 创建一个带有 AnchoredText "foo" 的 PaddedBox 对象 pb，在左上角添加到 ax
    fig, ax = plt.subplots()
    at = AnchoredText("foo", 'upper left')
    pb = PaddedBox(at, patch_attrs={'facecolor': 'r'}, draw_frame=True)
    ax.add_artist(pb)
    # 绘制图形但不进行渲染
    fig.draw_without_rendering()


def test_annotationbbox_properties():
    # 创建一个 AnnotationBbox 对象 ab，包含大小为 20x20 的 DrawingArea，位于 (0.5, 0.5)，xycoords='data'
    ab = AnnotationBbox(DrawingArea(20, 20, 0, 0, clip=True), (0.5, 0.5),
                        xycoords='data')
    # 断言 ab 的 xyann 属性等于 (0.5, 0.5)，如果 xybox 未指定的话
    assert ab.xyann == (0.5, 0.5)
    # 断言 ab 的 anncoords 属性为 'data'，如果 boxcoords 未指定的话
    assert ab.anncoords == 'data'

    # 创建另一个 AnnotationBbox 对象 ab，包含大小为 20x20 的 DrawingArea，位于 (0.5, 0.5)，xybox=(-0.2, 0.4)，xycoords='data'，boxcoords='axes fraction'
    ab = AnnotationBbox(DrawingArea(20, 20, 0, 0, clip=True), (0.5, 0.5),
                        xybox=(-0.2, 0.4), xycoords='data',
                        boxcoords='axes fraction')
    # 断言：检查 ab 对象的 xyann 属性是否等于 (-0.2, 0.4)，这里是关于 xybox 的说明（如果有的话）
    assert ab.xyann == (-0.2, 0.4)
    
    # 断言：检查 ab 对象的 anncoords 属性是否等于 'axes fraction'，这里是关于 boxcoords 的说明（如果有的话）
    assert ab.anncoords == 'axes fraction'
# 定义一个测试函数，用于验证 TextArea 类的属性和方法
def test_textarea_properties():
    # 创建一个包含文本 'Foo' 的 TextArea 对象
    ta = TextArea('Foo')
    # 断言获取文本内容是否为 'Foo'
    assert ta.get_text() == 'Foo'
    # 断言获取是否为多行基线偏移值，应为 False
    assert not ta.get_multilinebaseline()

    # 设置 TextArea 的文本内容为 'Bar'
    ta.set_text('Bar')
    # 设置多行基线偏移值为 True
    ta.set_multilinebaseline(True)
    # 断言获取文本内容是否为 'Bar'
    assert ta.get_text() == 'Bar'
    # 断言获取多行基线偏移值是否为 True
    assert ta.get_multilinebaseline()


# 使用装饰器检查两个图形是否相等的测试函数
@check_figures_equal()
def test_textarea_set_text(fig_test, fig_ref):
    # 在参考图形中添加子图并添加文本 'Foo' 到左上角
    ax_ref = fig_ref.add_subplot()
    text0 = AnchoredText("Foo", "upper left")
    ax_ref.add_artist(text0)

    # 在测试图形中添加子图并添加文本 'Bar' 到左上角
    ax_test = fig_test.add_subplot()
    text1 = AnchoredText("Bar", "upper left")
    ax_test.add_artist(text1)
    # 将文本 'Bar' 修改为 'Foo'
    text1.txt.set_text("Foo")


# 使用图像比较功能来测试 PaddedBox 类
@image_comparison(['paddedbox.png'], remove_text=True, style='mpl20')
def test_paddedbox():
    # 创建一个图形和坐标系
    fig, ax = plt.subplots()

    # 创建一个文本内容为 'foo' 的 TextArea 对象，并设置填充、颜色等属性，添加到图中
    ta = TextArea("foo")
    pb = PaddedBox(ta, pad=5, patch_attrs={'facecolor': 'r'}, draw_frame=True)
    ab = AnchoredOffsetbox('upper left', child=pb)
    ax.add_artist(ab)

    # 创建一个文本内容为 'bar' 的 TextArea 对象，并设置不同的填充、颜色属性，添加到图中
    ta = TextArea("bar")
    pb = PaddedBox(ta, pad=10, patch_attrs={'facecolor': 'b'})
    ab = AnchoredOffsetbox('upper right', child=pb)
    ax.add_artist(ab)

    # 创建一个文本内容为 'foobar' 的 TextArea 对象，并设置不同的填充、边框属性，添加到图中
    ta = TextArea("foobar")
    pb = PaddedBox(ta, pad=15, draw_frame=True)
    ab = AnchoredOffsetbox('lower right', child=pb)
    ax.add_artist(ab)


# 测试移除可拖动对象的函数
def test_remove_draggable():
    # 创建一个图形和坐标系
    fig, ax = plt.subplots()
    # 在图中添加一个注释并设置其可拖动
    an = ax.annotate("foo", (.5, .5))
    an.draggable(True)
    # 移除注释对象
    an.remove()
    # 模拟鼠标事件，处理按钮释放事件
    MouseEvent("button_release_event", fig.canvas, 1, 1)._process()


# 测试在子图中使用可拖动对象的函数
def test_draggable_in_subfigure():
    # 创建一个图形对象
    fig = plt.figure()
    # 在子图中添加注释 'foo' 并设置其可拖动
    ann = fig.subfigures().add_axes([0, 0, 1, 1]).annotate("foo", (0, 0))
    ann.draggable(True)
    # 绘制图形，直到第一次绘制完成，使文本对象可选
    fig.canvas.draw()
    # 模拟鼠标事件，处理按钮按下事件
    MouseEvent("button_press_event", fig.canvas, 1, 1)._process()
    # 断言检查是否成功使注释对象变为可拖动状态
    assert ann._draggable.got_artist
```