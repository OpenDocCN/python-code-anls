# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_text.py`

```
# 导入需要的模块和函数
from datetime import datetime
import io
import warnings

import numpy as np
from numpy.testing import assert_almost_equal
from packaging.version import parse as parse_version
import pyparsing
import pytest

import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont  # 导入字体相关的模块和函数
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
from matplotlib.text import Text, Annotation, OffsetFrom

# 解析pyparsing版本号
pyparsing_version = parse_version(pyparsing.__version__)

# 定义测试函数，并使用image_comparison装饰器指定比较的图片
@image_comparison(['font_styles'])
def test_font_styles():

    # 定义内部函数，用于查找匹配的Matplotlib字体
    def find_matplotlib_font(**kw):
        prop = FontProperties(**kw)
        path = findfont(prop, directory=mpl.get_data_path())
        return FontProperties(fname=path)

    # 忽略特定警告信息
    warnings.filterwarnings(
        'ignore',
        r"findfont: Font family \[u?'Foo'\] not found. Falling back to .",
        UserWarning,
        module='matplotlib.font_manager')

    # 创建一个图形对象和轴对象
    fig, ax = plt.subplots()

    # 查找并设置普通字体属性
    normal_font = find_matplotlib_font(
        family="sans-serif",
        style="normal",
        variant="normal",
        size=14)
    # 在图中添加注释对象，并设置字体属性为普通字体
    a = ax.annotate(
        "Normal Font",
        (0.1, 0.1),
        xycoords='axes fraction',
        fontproperties=normal_font)
    # 断言注释对象的字体名称为'DejaVu Sans'
    assert a.get_fontname() == 'DejaVu Sans'
    # 断言注释对象的字体风格为'normal'
    assert a.get_fontstyle() == 'normal'
    # 断言注释对象的字体变体为'normal'
    assert a.get_fontvariant() == 'normal'
    # 断言注释对象的字体粗细为'normal'
    assert a.get_weight() == 'normal'
    # 断言注释对象的字体拉伸为'normal'
    assert a.get_stretch() == 'normal'

    # 查找并设置粗体字体属性
    bold_font = find_matplotlib_font(
        family="Foo",
        style="normal",
        variant="normal",
        weight="bold",
        stretch=500,
        size=14)
    # 在图中添加粗体字体的注释对象
    ax.annotate(
        "Bold Font",
        (0.1, 0.2),
        xycoords='axes fraction',
        fontproperties=bold_font)

    # 查找并设置粗斜体字体属性
    bold_italic_font = find_matplotlib_font(
        family="sans serif",
        style="italic",
        variant="normal",
        weight=750,
        stretch=500,
        size=14)
    # 在图中添加粗斜体字体的注释对象
    ax.annotate(
        "Bold Italic Font",
        (0.1, 0.3),
        xycoords='axes fraction',
        fontproperties=bold_italic_font)

    # 查找并设置轻字体属性
    light_font = find_matplotlib_font(
        family="sans-serif",
        style="normal",
        variant="normal",
        weight=200,
        stretch=500,
        size=14)
    # 在图中添加轻字体的注释对象
    ax.annotate(
        "Light Font",
        (0.1, 0.4),
        xycoords='axes fraction',
        fontproperties=light_font)

    # 查找并设置紧凑字体属性
    condensed_font = find_matplotlib_font(
        family="sans-serif",
        style="normal",
        variant="normal",
        weight=500,
        stretch=100,
        size=14)
    # 在图形 ax 上添加注释 "Condensed Font"，位于相对坐标 (0.1, 0.5) 处
    ax.annotate(
        "Condensed Font",
        (0.1, 0.5),
        xycoords='axes fraction',
        fontproperties=condensed_font)
    
    # 设置图形 ax 的 x 轴刻度为空列表，即不显示 x 轴刻度
    ax.set_xticks([])
    
    # 设置图形 ax 的 y 轴刻度为空列表，即不显示 y 轴刻度
    ax.set_yticks([])
@image_comparison(['multiline'])
def test_multiline():
    # 创建新的图形窗口
    plt.figure()
    # 添加子图到当前图形中，1 行 1 列的布局，当前是第 1 个子图
    ax = plt.subplot(1, 1, 1)
    # 设置子图标题为 "multiline\ntext alignment"
    ax.set_title("multiline\ntext alignment")

    # 在指定位置 (0.2, 0.5) 添加文本 "TpTpTp\n$M$\nTpTpTp"，字体大小为 20，水平居中，垂直向上对齐
    plt.text(0.2, 0.5, "TpTpTp\n$M$\nTpTpTp", size=20, ha="center", va="top")

    # 在指定位置 (0.5, 0.5) 添加文本 "TpTpTp\n$M^{M^{M^{M}}}$\nTpTpTp"，字体大小为 20，
    # 水平居中，垂直向上对齐
    plt.text(0.5, 0.5, "TpTpTp\n$M^{M^{M^{M}}}$\nTpTpTp", size=20, ha="center", va="top")

    # 在指定位置 (0.8, 0.5) 添加文本 "TpTpTp\n$M_{q_{q_{q}}}$\nTpTpTp"，字体大小为 20，
    # 水平居中，垂直向上对齐
    plt.text(0.8, 0.5, "TpTpTp\n$M_{q_{q_{q}}}$\nTpTpTp", size=20, ha="center", va="top")

    # 设置 x 轴的范围为 0 到 1
    plt.xlim(0, 1)
    # 设置 y 轴的范围为 0 到 0.8
    plt.ylim(0, 0.8)

    # 设置 x 轴的刻度为空列表，即不显示刻度
    ax.set_xticks([])
    # 设置 y 轴的刻度为空列表，即不显示刻度
    ax.set_yticks([])


@image_comparison(['multiline2'], style='mpl20')
def test_multiline2():
    # 设置文本的字距因子为 6（kerning_factor）
    plt.rcParams['text.kerning_factor'] = 6

    # 创建一个新的图形和一个子图
    fig, ax = plt.subplots()

    # 设置子图的 x 轴范围为 [0, 1.4]
    ax.set_xlim([0, 1.4])
    # 设置子图的 y 轴范围为 [0, 2]
    ax.set_ylim([0, 2])
    # 在 y = 0.5 的位置绘制一条水平线，颜色为 'C2'，线宽为 0.3
    ax.axhline(0.5, color='C2', linewidth=0.3)

    # 定义一组文本列表 sts
    sts = ['Line', '2 Lineg\n 2 Lg', '$\\sum_i x $', 'hi $\\sum_i x $\ntest',
           'test\n $\\sum_i x $', '$\\sum_i x $\n $\\sum_i x $']

    # 获取渲染器对象
    renderer = fig.canvas.get_renderer()

    # 定义一个函数 draw_box，用于在文本周围绘制一个矩形框
    def draw_box(ax, tt):
        r = mpatches.Rectangle((0, 0), 1, 1, clip_on=False,
                               transform=ax.transAxes)
        r.set_bounds(
            tt.get_window_extent(renderer)
            .transformed(ax.transAxes.inverted())
            .bounds)
        ax.add_patch(r)

    # 水平文本对齐方式设置为 'left'
    horal = 'left'

    # 遍历 sts 列表的索引和元素
    for nn, st in enumerate(sts):
        # 在子图上指定位置添加文本 st，位置计算为 (0.2 * nn + 0.1, 0.5)，水平对齐方式为 horal，
        # 垂直对齐方式为 'bottom'
        tt = ax.text(0.2 * nn + 0.1, 0.5, st, horizontalalignment=horal,
                     verticalalignment='bottom')
        # 绘制文本周围的矩形框
        draw_box(ax, tt)

    # 在位置 (1.2, 0.5) 添加文本 'Bottom align'，颜色为 'C2'
    ax.text(1.2, 0.5, 'Bottom align', color='C2')

    # 绘制一条 y = 1.3 的水平线，颜色为 'C2'，线宽为 0.3
    ax.axhline(1.3, color='C2', linewidth=0.3)

    # 重复前面的文本绘制和框绘制过程，垂直对齐方式为 'top'
    for nn, st in enumerate(sts):
        tt = ax.text(0.2 * nn + 0.1, 1.3, st, horizontalalignment=horal,
                     verticalalignment='top')
        draw_box(ax, tt)

    # 在位置 (1.2, 1.3) 添加文本 'Top align'，颜色为 'C2'
    ax.text(1.2, 1.3, 'Top align', color='C2')

    # 绘制一条 y = 1.8 的水平线，颜色为 'C2'，线宽为 0.3
    ax.axhline(1.8, color='C2', linewidth=0.3)

    # 重复前面的文本绘制和框绘制过程，垂直对齐方式为 'baseline'
    for nn, st in enumerate(sts):
        tt = ax.text(0.2 * nn + 0.1, 1.8, st, horizontalalignment=horal,
                     verticalalignment='baseline')
        draw_box(ax, tt)

    # 在位置 (1.2, 1.8) 添加文本 'Baseline align'，颜色为 'C2'
    ax.text(1.2, 1.8, 'Baseline align', color='C2')

    # 绘制一条 y = 0.1 的水平线，颜色为 'C2'，线宽为 0.3，文本垂直对齐方式为 'bottom'，且旋转角度为 20 度
    ax.axhline(0.1, color='C2', linewidth=0.3)
    for nn, st in enumerate(sts):
        tt = ax.text(0.2 * nn + 0.1, 0.1, st, horizontalalignment=horal,
                     verticalalignment='bottom', rotation=20)
        draw_box(ax, tt)
    # 在位置 (1.2, 0.1) 添加文本 'Bot align, rot20'，颜色为 'C2'
    ax.text(1.2, 0.1, 'Bot align, rot20', color='C2')


@image_comparison(['antialiased.png'], style='mpl20')
def test_antialiasing():
    # 设置全局参数，文本抗锯齿效果为 False，即关闭抗锯齿
    mpl.rcParams['text.antialiased'] = False  # Passed arguments should override.

    # 创建一个尺寸为 (5.25, 0.75) 的新图形
    fig = plt.figure(figsize=(5.25, 0.75))

    # 在图形的指定位置添加文本 "antialiased"，水平和垂直对齐方式均为 'center'，开启抗锯齿
    fig.text(0.3, 0.75, "antialiased", horizontalalignment='center',
             verticalalignment='center', antialiased=True)

    # 在图形的指定位置添加文本 "$\sqrt{x}$"，水平和垂直对齐方式均为 'center'，开启抗锯齿
    fig.text(0.3, 0.25, r"$\sqrt{x}$", horizontalalignment='center',
             verticalalignment='center', antialiased=True)

    # 再次设置文本抗锯齿效果为 True，即开启抗锯齿
    # 在图形对象上添加文本，设置文本位置和内容，并且关闭抗锯齿效果
    fig.text(0.7, 0.75, "not antialiased", horizontalalignment='center',
             verticalalignment='center', antialiased=False)
    
    # 在图形对象上添加文本，设置文本位置和内容（使用数学符号），并且关闭抗锯齿效果
    fig.text(0.7, 0.25, r"$\sqrt{x}$", horizontalalignment='center',
             verticalalignment='center', antialiased=False)
    
    # 修改Matplotlib的全局参数，设置文本的抗锯齿效果为关闭，但不影响已有的文本。
    mpl.rcParams['text.antialiased'] = False
def test_afm_kerning():
    # 查找指定字体（Helvetica）的 AFM 文件路径
    fn = mpl.font_manager.findfont("Helvetica", fontext="afm")
    # 使用二进制模式打开 AFM 文件
    with open(fn, 'rb') as fh:
        # 创建 AFM 对象
        afm = mpl._afm.AFM(fh)
    # 断言检查字符串 'VAVAVAVAVAVA' 的宽度和高度是否符合预期
    assert afm.string_width_height('VAVAVAVAVAVA') == (7174.0, 718)


@image_comparison(['text_contains.png'])
def test_contains():
    # 创建新的图形对象
    fig = plt.figure()
    # 创建新的坐标轴对象
    ax = plt.axes()

    # 创建模拟的鼠标事件对象
    mevent = MouseEvent('button_press_event', fig.canvas, 0.5, 0.5, 1, None)

    # 在指定范围内生成坐标网格
    xs = np.linspace(0.25, 0.75, 30)
    ys = np.linspace(0.25, 0.75, 30)
    xs, ys = np.meshgrid(xs, ys)

    # 在坐标轴上添加文本对象，指定位置、文本内容和属性
    txt = plt.text(
        0.5, 0.4, 'hello world', ha='center', fontsize=30, rotation=30)
    # 如果需要绘制文本的边界框，取消注释以下行
    # txt.set_bbox(dict(edgecolor='black', facecolor='none'))

    # 绘制文本。这是必要的，因为 contains 方法只能在存在渲染器时工作。
    fig.canvas.draw()

    # 遍历坐标网格中的每个点
    for x, y in zip(xs.flat, ys.flat):
        # 将坐标转换为 AxesTransform 后，设置鼠标事件的位置
        mevent.x, mevent.y = plt.gca().transAxes.transform([x, y])
        # 检查文本对象是否包含鼠标事件，获取结果和其他信息
        contains, _ = txt.contains(mevent)
        # 根据包含情况设置点的颜色
        color = 'yellow' if contains else 'red'

        # 捕获视图限制（viewLim），绘制点，并重置视图限制
        vl = ax.viewLim.frozen()
        ax.plot(x, y, 'o', color=color)
        ax.viewLim.set(vl)


def test_annotation_contains():
    # 检查 Annotation.contains 方法是否分别查看文本和箭头的边界框，而不是联合边界框。
    fig, ax = plt.subplots()
    # 创建带注释的对象，指定文本和箭头的位置、文本内容和箭头属性
    ann = ax.annotate(
        "hello", xy=(.4, .4), xytext=(.6, .6), arrowprops={"arrowstyle": "->"})
    fig.canvas.draw()  # 同样需要为了与 test_contains 相同的原因。
    # 创建模拟的鼠标事件对象，设置其位置
    event = MouseEvent(
        "button_press_event", fig.canvas, *ax.transData.transform((.5, .6)))
    # 断言检查 Annotation 对象是否包含鼠标事件
    assert ann.contains(event) == (False, {})


@pytest.mark.parametrize('err, xycoords, match', (
    (TypeError, print, "xycoords callable must return a BboxBase or Transform, not a"),
    (TypeError, [0, 0], r"'xycoords' must be an instance of str, tuple"),
    (ValueError, "foo", "'foo' is not a valid coordinate"),
    (ValueError, "foo bar", "'foo bar' is not a valid coordinate"),
    (ValueError, "offset foo", "xycoords cannot be an offset coordinate"),
    (ValueError, "axes foo", "'foo' is not a recognized unit"),
))
def test_annotate_errors(err, xycoords, match):
    # 创建新的图形和坐标轴对象
    fig, ax = plt.subplots()
    # 使用 pytest 检查是否能够捕获特定错误类型和匹配的错误消息
    with pytest.raises(err, match=match):
        # 添加带注释的文本到坐标轴，指定位置、文本内容和其他属性
        ax.annotate('xy', (0, 0), xytext=(0.5, 0.5), xycoords=xycoords)
        fig.canvas.draw()


@image_comparison(['titles'])
def test_titles():
    # 创建新的图形对象
    plt.figure()
    # 创建子图对象
    ax = plt.subplot(1, 1, 1)
    # 设置左侧和右侧的标题
    ax.set_title("left title", loc="left")
    ax.set_title("right title", loc="right")
    # 设置不显示刻度线
    ax.set_xticks([])
    ax.set_yticks([])


@image_comparison(['text_alignment'], style='mpl20')
def test_alignment():
    # 创建新的图形对象
    plt.figure()
    # 创建子图对象
    ax = plt.subplot(1, 1, 1)

    x = 0.1
    # 对于每个旋转角度（0度和30度）
    for rotation in (0, 30):
        # 对于每种对齐方式（顶部，底部，基线和中心）
        for alignment in ('top', 'bottom', 'baseline', 'center'):
            # 在坐标 x 处绘制文本，显示对齐方式和 " Tj"，设置垂直对齐方式和旋转角度
            ax.text(
                x, 0.5, alignment + " Tj", va=alignment, rotation=rotation,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            # 在坐标 x 处绘制数学公式，设置垂直对齐方式和旋转角度
            ax.text(
                x, 1.0, r'$\sum_{i=0}^{j}$', va=alignment, rotation=rotation)
            # 增加 x 值，用于下一个文本的水平位置
            x += 0.1

    # 在图上绘制横线段，水平位置从 0 到 1，垂直位置为 0.5
    ax.plot([0, 1], [0.5, 0.5])
    # 在图上绘制另一条横线段，水平位置从 0 到 1，垂直位置为 1.0
    ax.plot([0, 1], [1.0, 1.0])

    # 设置 x 轴范围从 0 到 1
    ax.set_xlim(0, 1)
    # 设置 y 轴范围从 0 到 1.5
    ax.set_ylim(0, 1.5)
    # 清空 x 轴刻度
    ax.set_xticks([])
    # 清空 y 轴刻度
    ax.set_yticks([])
@image_comparison(['axes_titles.png'])
# 使用 Matplotlib 的 image_comparison 装饰器，用于比较图像结果
def test_axes_titles():
    # 创建新的图像
    plt.figure()
    # 在图像中创建子图
    ax = plt.subplot(1, 1, 1)
    # 设置子图的标题为 'center'，位置在中心，字体大小为20，字重为700
    ax.set_title('center', loc='center', fontsize=20, fontweight=700)
    # 设置子图的标题为 'left'，位置在左侧，字体大小为12，字重为400
    ax.set_title('left', loc='left', fontsize=12, fontweight=400)
    # 设置子图的标题为 'right'，位置在右侧，字体大小为12，字重为400
    ax.set_title('right', loc='right', fontsize=12, fontweight=400)


def test_set_position():
    # 创建新的图像和坐标系对象
    fig, ax = plt.subplots()

    # 测试 set_position 方法
    # 创建注释对象并设置初始位置
    ann = ax.annotate(
        'test', (0, 0), xytext=(0, 0), textcoords='figure pixels')
    # 绘制图像
    fig.canvas.draw()

    # 获取初始位置的边界框
    init_pos = ann.get_window_extent(fig.canvas.renderer)
    # 设置位移值
    shift_val = 15
    # 设置注释的新位置
    ann.set_position((shift_val, shift_val))
    # 重新绘制图像
    fig.canvas.draw()
    # 获取设置后的位置边界框
    post_pos = ann.get_window_extent(fig.canvas.renderer)

    # 断言检查位置变化是否正确
    for a, b in zip(init_pos.min, post_pos.min):
        assert a + shift_val == b

    # 再次测试 xyann 属性
    ann = ax.annotate(
        'test', (0, 0), xytext=(0, 0), textcoords='figure pixels')
    fig.canvas.draw()

    init_pos = ann.get_window_extent(fig.canvas.renderer)
    shift_val = 15
    ann.xyann = (shift_val, shift_val)
    fig.canvas.draw()
    post_pos = ann.get_window_extent(fig.canvas.renderer)

    for a, b in zip(init_pos.min, post_pos.min):
        assert a + shift_val == b


def test_char_index_at():
    # 创建新的图像对象
    fig = plt.figure()
    # 在图像中添加文本对象
    text = fig.text(0.1, 0.9, "")

    # 设置文本内容为 'i' 并获取其边界框
    text.set_text("i")
    bbox = text.get_window_extent()
    size_i = bbox.x1 - bbox.x0

    # 设置文本内容为 'm' 并获取其边界框
    text.set_text("m")
    bbox = text.get_window_extent()
    size_m = bbox.x1 - bbox.x0

    # 设置文本内容为 'iiiimmmm' 并获取其边界框和原点位置
    text.set_text("iiiimmmm")
    bbox = text.get_window_extent()
    origin = bbox.x0

    # 使用断言检查字符索引位置的计算是否正确
    assert text._char_index_at(origin - size_i) == 0  # 第一个字符左侧
    assert text._char_index_at(origin) == 0
    assert text._char_index_at(origin + 0.499*size_i) == 0
    assert text._char_index_at(origin + 0.501*size_i) == 1
    assert text._char_index_at(origin + size_i*3) == 3
    assert text._char_index_at(origin + size_i*4 + size_m*3) == 7
    assert text._char_index_at(origin + size_i*4 + size_m*4) == 8
    assert text._char_index_at(origin + size_i*4 + size_m*10) == 8


@pytest.mark.parametrize('text', ['', 'O'], ids=['empty', 'non-empty'])
# 使用 pytest.mark.parametrize 对 test_non_default_dpi 函数进行参数化，测试不同的文本内容
def test_non_default_dpi(text):
    # 创建新的图像和坐标系对象
    fig, ax = plt.subplots()

    # 在坐标系中添加文本对象
    t1 = ax.text(0.5, 0.5, text, ha='left', va='bottom')
    # 绘制图像
    fig.canvas.draw()
    # 获取图像的 DPI
    dpi = fig.dpi

    # 获取文本对象的边界框
    bbox1 = t1.get_window_extent()
    # 使用不同的 DPI 获取文本对象的边界框
    bbox2 = t1.get_window_extent(dpi=dpi * 10)
    # 使用 np.testing.assert_allclose 断言检查两者边界框坐标点的近似程度
    np.testing.assert_allclose(bbox2.get_points(), bbox1.get_points() * 10,
                               rtol=5e-2)
    # 断言检查调用 Text.get_window_extent 后图像 DPI 没有永久性改变
    assert fig.dpi == dpi


def test_get_rotation_string():
    # 使用断言检查 Text(rotation='horizontal') 返回的旋转角度是否为 0
    assert Text(rotation='horizontal').get_rotation() == 0.
    # 使用断言检查 Text(rotation='vertical') 返回的旋转角度是否为 90
    assert Text(rotation='vertical').get_rotation() == 90.


def test_get_rotation_float():
    # 对于浮点数旋转角度，使用断言检查其返回的旋转角度是否与设定值相等
    for i in [15., 16.70, 77.4]:
        assert Text(rotation=i).get_rotation() == i


def test_get_rotation_int():
    # 对于整数旋转角度，使用断言检查其返回的旋转角度是否与设定值相等
    for i in [67, 16, 41]:
        assert Text(rotation=i).get_rotation() == float(i)
# 测试函数，验证在设置了错误的旋转模式时是否会引发 ValueError 异常
def test_get_rotation_raises():
    # 使用 pytest.raises 检查是否引发 ValueError 异常
    with pytest.raises(ValueError):
        Text(rotation='hozirontal')


# 测试函数，验证当旋转角度为 None 时，get_rotation 方法返回值是否为 0.0
def test_get_rotation_none():
    # 断言 Text 类实例化对象的 get_rotation 方法返回值为 0.0
    assert Text(rotation=None).get_rotation() == 0.0


# 测试函数，验证当旋转角度超过 360 度时，get_rotation 方法是否取模为正确的值
def test_get_rotation_mod360():
    # 使用循环和 assert_almost_equal 函数验证多个旋转角度的 get_rotation 方法返回值
    for i, j in zip([360., 377., 720+177.2], [0., 17., 177.2]):
        assert_almost_equal(Text(rotation=i).get_rotation(), j)


# 参数化测试函数，验证在不同的水平对齐 (ha) 和垂直对齐 (va) 模式下，旋转角度为 0 时的行为
@pytest.mark.parametrize("ha", ["center", "right", "left"])
@pytest.mark.parametrize("va", ["center", "top", "bottom", "baseline", "center_baseline"])
def test_null_rotation_with_rotation_mode(ha, va):
    # 创建绘图对象
    fig, ax = plt.subplots()
    # 定义文本关键字参数
    kw = dict(rotation=0, va=va, ha=ha)
    # 使用不同旋转模式绘制两个相同位置的文本对象
    t0 = ax.text(.5, .5, 'test', rotation_mode='anchor', **kw)
    t1 = ax.text(.5, .5, 'test', rotation_mode='default', **kw)
    # 绘制图形
    fig.canvas.draw()
    # 断言两个文本对象的窗口范围是否相同
    assert_almost_equal(t0.get_window_extent(fig.canvas.renderer).get_points(),
                        t1.get_window_extent(fig.canvas.renderer).get_points())


# 图像比较测试函数，验证文本框是否被裁剪
@image_comparison(['text_bboxclip'])
def test_bbox_clipping():
    # 在图中绘制带背景颜色和裁剪属性的文本
    plt.text(0.9, 0.2, 'Is bbox clipped?', backgroundcolor='r', clip_on=True)
    t = plt.text(0.9, 0.5, 'Is fancy bbox clipped?', clip_on=True)
    # 设置文本框样式
    t.set_bbox({"boxstyle": "round, pad=0.1"})


# 图像比较测试函数，验证在负坐标轴坐标系中添加标注的行为
@image_comparison(['annotation_negative_ax_coords.png'])
def test_annotation_negative_ax_coords():
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在不同坐标系中添加标注，验证其行为
    ax.annotate('+ pts',
                xytext=[30, 20], textcoords='axes points',
                xy=[30, 20], xycoords='axes points', fontsize=32)
    ax.annotate('- pts',
                xytext=[30, -20], textcoords='axes points',
                xy=[30, -20], xycoords='axes points', fontsize=32,
                va='top')
    ax.annotate('+ frac',
                xytext=[0.75, 0.05], textcoords='axes fraction',
                xy=[0.75, 0.05], xycoords='axes fraction', fontsize=32)
    ax.annotate('- frac',
                xytext=[0.75, -0.05], textcoords='axes fraction',
                xy=[0.75, -0.05], xycoords='axes fraction', fontsize=32,
                va='top')
    ax.annotate('+ pixels',
                xytext=[160, 25], textcoords='axes pixels',
                xy=[160, 25], xycoords='axes pixels', fontsize=32)
    ax.annotate('- pixels',
                xytext=[160, -25], textcoords='axes pixels',
                xy=[160, -25], xycoords='axes pixels', fontsize=32,
                va='top')


# 图像比较测试函数，验证在负图形坐标系中添加标注的行为
@image_comparison(['annotation_negative_fig_coords.png'])
def test_annotation_negative_fig_coords():
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在不同坐标系中添加标注，验证其行为
    ax.annotate('+ pts',
                xytext=[10, 120], textcoords='figure points',
                xy=[10, 120], xycoords='figure points', fontsize=32)
    ax.annotate('- pts',
                xytext=[-10, 180], textcoords='figure points',
                xy=[-10, 180], xycoords='figure points', fontsize=32,
                va='top')
    # 在图形上添加注释文本 '+ frac'，使用相对图形的坐标和文本坐标
    ax.annotate('+ frac',
                xytext=[0.05, 0.55], textcoords='figure fraction',
                xy=[0.05, 0.55], xycoords='figure fraction', fontsize=32)
    
    # 在图形上添加注释文本 '- frac'，使用相对图形的坐标和文本坐标，垂直对齐方式为顶部
    ax.annotate('- frac',
                xytext=[-0.05, 0.5], textcoords='figure fraction',
                xy=[-0.05, 0.5], xycoords='figure fraction', fontsize=32,
                va='top')
    
    # 在图形上添加注释文本 '+ pixels'，使用像素坐标和文本坐标
    ax.annotate('+ pixels',
                xytext=[50, 50], textcoords='figure pixels',
                xy=[50, 50], xycoords='figure pixels', fontsize=32)
    
    # 在图形上添加注释文本 '- pixels'，使用像素坐标和文本坐标，垂直对齐方式为顶部
    ax.annotate('- pixels',
                xytext=[-50, 100], textcoords='figure pixels',
                xy=[-50, 100], xycoords='figure pixels', fontsize=32,
                va='top')
# 定义一个测试函数，用于验证文本对象的更新机制
def test_text_stale():
    # 创建一个包含两个子图的图形对象
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # 绘制图形
    plt.draw_all()
    # 断言：ax1 和 ax2 的状态不是过时的
    assert not ax1.stale
    assert not ax2.stale
    assert not fig.stale

    # 在 ax1 上添加文本对象 'aardvark'
    txt1 = ax1.text(.5, .5, 'aardvark')
    # 断言：ax1 和文本对象 txt1 的状态为过时的
    assert ax1.stale
    assert txt1.stale
    assert fig.stale

    # 在 ax2 上添加注释 'aardvark'
    ann1 = ax2.annotate('aardvark', xy=[.5, .5])
    # 断言：ax2 和注释对象 ann1 的状态为过时的
    assert ax2.stale
    assert ann1.stale
    assert fig.stale

    # 再次绘制图形
    plt.draw_all()
    # 断言：ax1 和 ax2 的状态不是过时的
    assert not ax1.stale
    assert not ax2.stale
    assert not fig.stale


# 使用图像比较功能测试文本裁剪效果
@image_comparison(['agg_text_clip.png'])
def test_agg_text_clip():
    # 设定随机数种子
    np.random.seed(1)
    # 创建包含两个子图的图形对象
    fig, (ax1, ax2) = plt.subplots(2)
    # 对于随机生成的 10 个坐标点，在 ax1 上添加带裁剪的文本 "foo"，在 ax2 上添加普通文本 "foo"
    for x, y in np.random.rand(10, 2):
        ax1.text(x, y, "foo", clip_on=True)
        ax2.text(x, y, "foo")


# 测试文本大小绑定功能
def test_text_size_binding():
    # 设定全局字体大小为 10
    mpl.rcParams['font.size'] = 10
    # 创建字体属性对象，设定大小为 'large'
    fp = mpl.font_manager.FontProperties(size='large')
    # 获取字体属性对象的点大小
    sz1 = fp.get_size_in_points()
    # 设定全局字体大小为 100
    mpl.rcParams['font.size'] = 100

    # 断言：前后两次获取的字体点大小应该相等
    assert sz1 == fp.get_size_in_points()


# 使用图像比较功能测试字体缩放效果
@image_comparison(['font_scaling.pdf'])
def test_font_scaling():
    # 设定 PDF 输出的字体类型
    mpl.rcParams['pdf.fonttype'] = 42
    # 创建图形对象和坐标轴对象，设定尺寸为 (6.4, 12.4)
    fig, ax = plt.subplots(figsize=(6.4, 12.4))
    # 设置坐标轴的主要刻度定位器为空
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    # 设定 y 轴范围
    ax.set_ylim(-10, 600)

    # 对于字体大小从 4 到 42（步长为 2）的范围内，循环添加文本对象到图形中
    for i, fs in enumerate(range(4, 43, 2)):
        ax.text(0.1, i*30, f"{fs} pt font size", fontsize=fs)


# 使用参数化测试验证两行文本的行间距功能
@pytest.mark.parametrize('spacing1, spacing2', [(0.4, 2), (2, 0.4), (2, 2)])
def test_two_2line_texts(spacing1, spacing2):
    # 定义文本字符串，包含两行
    text_string = 'line1\nline2'
    # 创建图形对象
    fig = plt.figure()
    # 获取渲染器对象
    renderer = fig.canvas.get_renderer()

    # 添加两个文本对象到图形中，分别设定不同的行间距
    text1 = fig.text(0.25, 0.5, text_string, linespacing=spacing1)
    text2 = fig.text(0.25, 0.5, text_string, linespacing=spacing2)
    # 绘制图形
    fig.canvas.draw()

    # 获取文本对象的窗口范围
    box1 = text1.get_window_extent(renderer=renderer)
    box2 = text2.get_window_extent(renderer=renderer)

    # 断言：行间距只影响文本对象的高度
    assert box1.width == box2.width
    if spacing1 == spacing2:
        assert box1.height == box2.height
    else:
        assert box1.height != box2.height


# 测试文本行间距验证功能
def test_validate_linespacing():
    # 断言：当行间距参数为非法类型时，应该抛出 TypeError 异常
    with pytest.raises(TypeError):
        plt.text(.25, .5, "foo", linespacing="abc")


# 测试非有限位置的文本对象
def test_nonfinite_pos():
    # 创建图形对象和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上添加具有非有限位置的文本对象 'nan' 和 'inf'
    ax.text(0, np.nan, 'nan')
    ax.text(np.inf, 0, 'inf')
    # 绘制图形
    fig.canvas.draw()


# 测试文本提示因子在不同后端的一致性
def test_hinting_factor_backends():
    # 设定文本提示因子
    plt.rcParams['text.hinting_factor'] = 1
    # 创建图形对象
    fig = plt.figure()
    # 在图形上添加文本对象 'some text'
    t = fig.text(0.5, 0.5, 'some text')

    # 保存图形为 SVG 格式
    fig.savefig(io.BytesIO(), format='svg')
    # 获取文本对象的窗口范围间隔
    expected = t.get_window_extent().intervalx

    # 保存图形为 PNG 格式
    fig.savefig(io.BytesIO(), format='png')
    # 断言：后端应该在指定的误差范围内一致地应用文本提示因子
    np.testing.assert_allclose(t.get_window_extent().intervalx, expected,
                               rtol=0.1)


# 测试 usetex 功能是否被复制
@needs_usetex
def test_usetex_is_copied():
    # 间接测试更新属性时是否复制了 usetex 状态
    fig = plt.figure()
    plt.rcParams["text.usetex"] = False
    # 创建一个子图 ax1，放置在 1x2 网格的第一个位置 (1行2列中的第1列)
    ax1 = fig.add_subplot(121)
    
    # 设置 matplotlib 使用 LaTeX 渲染文本
    plt.rcParams["text.usetex"] = True
    
    # 创建另一个子图 ax2，放置在 1x2 网格的第二个位置 (1行2列中的第2列)
    ax2 = fig.add_subplot(122)
    
    # 在绘图画布上绘制当前图形
    fig.canvas.draw()
    
    # 遍历包含 (ax1, False) 和 (ax2, True) 的元组列表
    for ax, usetex in [(ax1, False), (ax2, True)]:
        # 遍历当前子图的 x 轴主刻度标签
        for t in ax.xaxis.majorTicks:
            # 断言当前刻度标签的 LaTeX 渲染状态与 usetex 变量相符
            assert t.label1.get_usetex() == usetex
@needs_usetex
def test_single_artist_usetex():
    # 使用 usetex 标记的单个艺术家，不会通过 mathtext 解析器处理
    # 对于 Agg 后端，mathtext 解析器当前无法解析 \frac12，需要用 \frac{1}{2} 替代
    fig = plt.figure()
    # 在图形中添加文本，使用 LaTeX 渲染，显示 \frac12
    fig.text(.5, .5, r"$\frac12$", usetex=True)
    # 绘制图形到画布上
    fig.canvas.draw()


@pytest.mark.parametrize("fmt", ["png", "pdf", "svg"])
def test_single_artist_usenotex(fmt):
    # 检查即使 rcParam 开启了 usetex，也可以将单个艺术家标记为不使用 usetex
    # "2_2_2" 无法传递给 TeX，所以此处禁用 usetex
    plt.rcParams["text.usetex"] = True
    fig = plt.figure()
    # 在图形中添加文本，不使用 LaTeX 渲染，显示 "2_2_2"
    fig.text(.5, .5, "2_2_2", usetex=False)
    # 将图形保存为指定格式
    fig.savefig(io.BytesIO(), format=fmt)


@image_comparison(['text_as_path_opacity.svg'])
def test_text_as_path_opacity():
    plt.figure()
    # 关闭坐标轴显示
    plt.gca().set_axis_off()
    # 添加透明度为 0.5 的文本 'c'
    plt.text(0.25, 0.25, 'c', color=(0, 0, 0, 0.5))
    # 添加透明度为 0.5 的文本 'a'
    plt.text(0.25, 0.5, 'a', alpha=0.5)
    # 添加颜色透明度为 0.5，颜色完全不透明的文本 'x'
    plt.text(0.25, 0.75, 'x', alpha=0.5, color=(0, 0, 0, 1))


@image_comparison(['text_as_text_opacity.svg'])
def test_text_as_text_opacity():
    mpl.rcParams['svg.fonttype'] = 'none'
    plt.figure()
    plt.gca().set_axis_off()
    # 添加透明度为 0.5 的文本 '50% using `color`'，颜色透明度为 0.5
    plt.text(0.25, 0.25, '50% using `color`', color=(0, 0, 0, 0.5))
    # 添加透明度为 0.5 的文本 '50% using `alpha`'
    plt.text(0.25, 0.5, '50% using `alpha`', alpha=0.5)
    # 添加颜色透明度为 0.5，颜色完全不透明的文本 '50% using `alpha` and 100% `color`'
    plt.text(0.25, 0.75, '50% using `alpha` and 100% `color`', alpha=0.5,
             color=(0, 0, 0, 1))


def test_text_repr():
    # 确保文本的 repr 不会对类别出错
    plt.plot(['A', 'B'], [1, 2])
    repr(plt.text(['A'], 0.5, 'Boo'))


def test_annotation_update():
    fig, ax = plt.subplots(1, 1)
    # 添加注释到图形中
    an = ax.annotate('annotation', xy=(0.5, 0.5))
    # 获取注释对象在画布上的窗口区域
    extent1 = an.get_window_extent(fig.canvas.get_renderer())
    # 调整图形布局
    fig.tight_layout()
    # 再次获取调整后注释对象在画布上的窗口区域
    extent2 = an.get_window_extent(fig.canvas.get_renderer())

    # 断言两次获取的窗口区域不完全相等，允许相对误差为 1e-6
    assert not np.allclose(extent1.get_points(), extent2.get_points(),
                           rtol=1e-6)


@check_figures_equal(extensions=["png"])
def test_annotation_units(fig_test, fig_ref):
    ax = fig_test.add_subplot()
    ax.plot(datetime.now(), 1, "o")  # 隐式设置坐标轴范围
    # 添加带单位的注释文本 'x' 到图形中
    ax.annotate("x", (datetime.now(), 0.5), xycoords=("data", "axes fraction"),
                xytext=(0, 0), textcoords="offset points")
    ax = fig_ref.add_subplot()
    ax.plot(datetime.now(), 1, "o")
    # 添加带单位的注释文本 'x' 到图形中
    ax.annotate("x", (datetime.now(), 0.5), xycoords=("data", "axes fraction"))


@image_comparison(['large_subscript_title.png'], style='mpl20')
def test_large_subscript_title():
    # 当这个测试图像重新生成时删除此行。
    # 设置文本的字距因子为 6，标题在 Y 轴上的位置设为 None
    plt.rcParams['text.kerning_factor'] = 6
    plt.rcParams['axes.titley'] = None

    fig, axs = plt.subplots(1, 2, figsize=(9, 2.5), constrained_layout=True)
    ax = axs[0]
    # 设置子图标题为带有下标的公式
    ax.set_title(r'$\sum_{i} x_i$')
    # 在第一个子图(ax)上设置标题为'New way'，并将标题位置设置为左对齐
    ax.set_title('New way', loc='left')
    # 设置第一个子图(ax)的X轴刻度标签为空列表，即不显示X轴刻度标签
    ax.set_xticklabels([])
    
    # 切换到第二个子图(ax)，设置其标题为数学公式'$\sum_{i} x_i$'，并将标题位置稍微调高到1.01
    ax = axs[1]
    ax.set_title(r'$\sum_{i} x_i$', y=1.01)
    # 继续在第二个子图(ax)上设置标题为'Old Way'，并将标题位置设置为左对齐
    ax.set_title('Old Way', loc='left')
    # 设置第二个子图(ax)的X轴刻度标签为空列表，即不显示X轴刻度标签
    ax.set_xticklabels([])
@pytest.mark.parametrize(
    "x, rotation, halign",
    [(0.7, 0, 'left'),  # 参数组合：x=0.7, rotation=0, halign='left'
     (0.5, 95, 'left'),  # 参数组合：x=0.5, rotation=95, halign='left'
     (0.3, 0, 'right'),  # 参数组合：x=0.3, rotation=0, halign='right'
     (0.3, 185, 'left')])  # 参数组合：x=0.3, rotation=185, halign='left'
def test_wrap(x, rotation, halign):
    # 创建一个大小为18x18的新图形对象
    fig = plt.figure(figsize=(18, 18))
    # 创建一个3行3列的网格布局，并将其关联到上面创建的图形对象
    gs = GridSpec(nrows=3, ncols=3, figure=fig)
    # 在网格布局的第(1,1)个位置添加子图
    subfig = fig.add_subfigure(gs[1, 1])
    # 设置一个长文本，确保需要多次换行
    s = 'This is a very long text that should be wrapped multiple times.'
    # 在子图中添加文本，设置其位置(x, 0.7)，并指定是否自动换行、旋转角度及水平对齐方式
    text = subfig.text(x, 0.7, s, wrap=True, rotation=rotation, ha=halign)
    # 绘制整个图形
    fig.canvas.draw()
    # 断言文本对象的换行后文本内容是否符合预期
    assert text._get_wrapped_text() == ('This is a very long\n'
                                        'text that should be\n'
                                        'wrapped multiple\n'
                                        'times.')


def test_mathwrap():
    # 创建一个大小为6x4的新图形对象
    fig = plt.figure(figsize=(6, 4))
    # 设置一个包含数学文本的长字符串
    s = r'This is a very $\overline{\mathrm{long}}$ line of Mathtext.'
    # 在图形中添加文本，位置为(0, 0.5)，设置文本大小为40，并指定是否自动换行
    text = fig.text(0, 0.5, s, size=40, wrap=True)
    # 绘制整个图形
    fig.canvas.draw()
    # 断言文本对象的换行后文本内容是否符合预期
    assert text._get_wrapped_text() == ('This is a very $\\overline{\\mathrm{long}}$\n'
                                        'line of Mathtext.')


def test_get_window_extent_wrapped():
    # 创建一个大小为3x3的新图形对象
    fig1 = plt.figure(figsize=(3, 3))
    # 设置一个过长的超标题，并指定需要自动换行
    fig1.suptitle("suptitle that is clearly too long in this case", wrap=True)
    # 获取超标题的绘图区域
    window_extent_test = fig1._suptitle.get_window_extent()

    # 创建另一个大小为3x3的新图形对象
    fig2 = plt.figure(figsize=(3, 3))
    # 设置一个显式换行的超标题
    fig2.suptitle("suptitle that is clearly\ntoo long in this case")
    # 获取超标题的绘图区域
    window_extent_ref = fig2._suptitle.get_window_extent()

    # 断言两个超标题的垂直范围是否相同
    assert window_extent_test.y0 == window_extent_ref.y0
    assert window_extent_test.y1 == window_extent_ref.y1


def test_long_word_wrap():
    # 创建一个大小为6x4的新图形对象
    fig = plt.figure(figsize=(6, 4))
    # 添加一个过长的单词，但设置为不自动换行
    text = fig.text(9.5, 8, 'Alonglineoftexttowrap', wrap=True)
    # 绘制整个图形
    fig.canvas.draw()
    # 断言文本对象的换行后文本内容是否符合预期
    assert text._get_wrapped_text() == 'Alonglineoftexttowrap'


def test_wrap_no_wrap():
    # 创建一个大小为6x4的新图形对象
    fig = plt.figure(figsize=(6, 4))
    # 添加一个不需要换行的文本
    text = fig.text(0, 0, 'non wrapped text', wrap=True)
    # 绘制整个图形
    fig.canvas.draw()
    # 断言文本对象的换行后文本内容是否符合预期
    assert text._get_wrapped_text() == 'non wrapped text'


@check_figures_equal(extensions=["png"])
def test_buffer_size(fig_test, fig_ref):
    # 在旧版的Agg渲染器上，大的非ASCII单字符字符串（如"€"）可能会被剪裁，因为渲染缓冲区大小可能由较小字符"a"的物理大小决定。
    ax = fig_test.add_subplot()
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["€", "a"])
    ax.yaxis.majorTicks[1].label1.set_color("w")

    ax = fig_ref.add_subplot()
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["€", ""])


def test_fontproperties_kwarg_precedence():
    """测试kwargs参数优先于字体属性默认值。"""
    plt.figure()
    # 创建 x 轴标签文本对象，设置标签内容为 "value"，字体属性为 Times New Roman，字体大小为 40.0
    text1 = plt.xlabel("value", fontproperties='Times New Roman', size=40.0)
    
    # 创建 y 轴标签文本对象，设置标签内容为 "counts"，字体大小为 40.0，字体属性为 Times New Roman
    text2 = plt.ylabel("counts", size=40.0, fontproperties='Times New Roman')
    
    # 断言验证 text1 文本对象的字体大小是否为 40.0
    assert text1.get_size() == 40.0
    
    # 断言验证 text2 文本对象的字体大小是否为 40.0
    assert text2.get_size() == 40.0
# 测试函数：测试文本旋转功能
def test_transform_rotates_text():
    # 获取当前图形的坐标轴
    ax = plt.gca()
    # 创建一个旋转30度的仿射变换对象
    transform = mtransforms.Affine2D().rotate_deg(30)
    # 在坐标轴上创建一个文本对象，并应用上述旋转变换
    text = ax.text(0, 0, 'test', transform=transform,
                   transform_rotates_text=True)
    # 获取文本对象的旋转角度
    result = text.get_rotation()
    # 断言文本对象的旋转角度近似等于30度
    assert_almost_equal(result, 30)


# 测试函数：测试更新操作不改变输入字典
def test_update_mutate_input():
    # 输入字典
    inp = dict(fontproperties=FontProperties(weight="bold"),
               bbox=None)
    # 创建输入字典的副本
    cache = dict(inp)
    # 创建文本对象
    t = Text()
    # 更新文本对象的属性
    t.update(inp)
    # 断言输入字典的'fontproperties'属性未被修改
    assert inp['fontproperties'] == cache['fontproperties']
    # 断言输入字典的'bbox'属性未被修改
    assert inp['bbox'] == cache['bbox']


# 测试函数：测试不支持的旋转值
@pytest.mark.parametrize('rotation', ['invalid string', [90]])
def test_invalid_rotation_values(rotation):
    # 断言使用不支持的旋转值时抛出 ValueError 异常
    with pytest.raises(
            ValueError,
            match=("rotation must be 'vertical', 'horizontal' or a number")):
        Text(0, 0, 'foo', rotation=rotation)


# 测试函数：测试不支持的颜色值
def test_invalid_color():
    # 断言使用不支持的颜色值时抛出 ValueError 异常
    with pytest.raises(ValueError):
        plt.figtext(.5, .5, "foo", c="foobar")


# 图像对比测试：测试 PDF 输出中的字符间距调整
@image_comparison(['text_pdf_kerning.pdf'], style='mpl20')
def test_pdf_kerning():
    # 创建新图形
    plt.figure()
    # 添加文本到图形，并指定字体大小
    plt.figtext(0.1, 0.5, "ATATATATATATATATATA", size=30)


# 测试函数：测试不支持的脚本语言
def test_unsupported_script(recwarn):
    # 创建新图形
    fig = plt.figure()
    # 在图形上添加文本，包含孟加拉数字0
    t = fig.text(.5, .5, "\N{BENGALI DIGIT ZERO}")
    # 绘制图形
    fig.canvas.draw()
    # 断言所有警告消息为用户警告类型
    assert all(isinstance(warn.message, UserWarning) for warn in recwarn)
    # 断言警告消息内容符合预期
    assert (
        [warn.message.args for warn in recwarn] ==
        [(r"Glyph 2534 (\N{BENGALI DIGIT ZERO}) missing from font(s) "
            + f"{t.get_fontname()}.",),
         (r"Matplotlib currently does not support Bengali natively.",)])


# 标记为预期失败：关于此 xfail 的更多信息请参见 gh-26152
@pytest.mark.xfail(pyparsing_version.release == (3, 1, 0),
                   reason="Error messages are incorrect with pyparsing 3.1.0")
def test_parse_math():
    # 创建新图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上添加数学表达式文本，使用错误的语法
    ax.text(0, 0, r"$ \wrong{math} $", parse_math=False)
    # 绘制图形
    fig.canvas.draw()

    # 再次添加数学表达式文本，使用正确的语法
    ax.text(0, 0, r"$ \wrong{math} $", parse_math=True)
    # 断言使用错误的数学表达式时抛出 ValueError 异常
    with pytest.raises(ValueError, match='Unknown symbol'):
        fig.canvas.draw()


# 标记为预期失败：关于此 xfail 的更多信息请参见 gh-26152
@pytest.mark.xfail(pyparsing_version.release == (3, 1, 0),
                   reason="Error messages are incorrect with pyparsing 3.1.0")
def test_parse_math_rcparams():
    # 默认情况下 parse_math 为 True
    fig, ax = plt.subplots()
    # 在坐标轴上添加数学表达式文本，使用错误的语法
    ax.text(0, 0, r"$ \wrong{math} $")
    # 断言使用错误的数学表达式时抛出 ValueError 异常
    with pytest.raises(ValueError, match='Unknown symbol'):
        fig.canvas.draw()

    # 将 rcParams 设置为 False
    with mpl.rc_context({'text.parse_math': False}):
        fig, ax = plt.subplots()
        # 在坐标轴上添加数学表达式文本，使用错误的语法
        ax.text(0, 0, r"$ \wrong{math} $")
        # 绘制图形
        fig.canvas.draw()


# 图像对比测试：测试 PDF 输出中使用字形类型 42 的文本字符间距调整
@image_comparison(['text_pdf_font42_kerning.pdf'], style='mpl20')
def test_pdf_font42_kerning():
    # 设置全局参数，指定 PDF 输出使用字形类型 42
    plt.rcParams['pdf.fonttype'] = 42
    # 创建新图形
    plt.figure()
    # 添加文本到图形，并指定字体大小
    plt.figtext(0.1, 0.5, "ATAVATAVATAVATAVATA", size=30)


# 图像对比测试：测试 PDF 输出中超出基本多文种平面（BMP）字符的处理
@image_comparison(['text_pdf_chars_beyond_bmp.pdf'], style='mpl20')
def test_pdf_chars_beyond_bmp():
    # 设置全局参数，指定 PDF 输出使用字形类型 42
    plt.rcParams['pdf.fonttype'] = 42
    # 设置 matplotlib 的全局参数，指定数学文本使用的字体集为 'stixsans'
    plt.rcParams['mathtext.fontset'] = 'stixsans'
    # 创建一个新的图形
    plt.figure()
    # 在图形上添加一段文本，使用数学模式显示文本 "Mass $m$ ⤈"，设置字体大小为 30
    plt.figtext(0.1, 0.5, "Mass $m$ \U00010308", size=30)
@needs_usetex
# 使用 usetex 装饰器标记这个函数，表示它需要使用 TeX 渲染引擎来处理文本

def test_metrics_cache():
    # 清空 mpl.text._get_text_metrics_with_cache_impl 的缓存
    mpl.text._get_text_metrics_with_cache_impl.cache_clear()

    # 创建一个新的图形对象
    fig = plt.figure()

    # 在图形上添加文本，不使用 TeX 渲染
    fig.text(.3, .5, "foo\nbar")

    # 在图形上添加文本，使用 TeX 渲染
    fig.text(.3, .5, "foo\nbar", usetex=True)

    # 在图形上添加文本，使用 TeX 渲染
    fig.text(.5, .5, "foo\nbar", usetex=True)

    # 绘制整个图形
    fig.canvas.draw()

    # 获取图形的渲染器
    renderer = fig._get_renderer()

    # ys 是一个字典，用于存储每个字符串在 y 轴上的绘制位置
    ys = {}

    # 定义一个函数 call，用于在渲染时记录每个字符串的位置
    def call(*args, **kwargs):
        renderer, x, y, s, *_ = args
        ys.setdefault(s, set()).add(y)

    # 将自定义的绘制函数 call 赋值给 renderer.draw_tex
    renderer.draw_tex = call

    # 再次绘制图形
    fig.canvas.draw()

    # 断言，确保 "foo" 和 "bar" 这两个 TeX 字符串在 y 轴上的位置相同
    assert [*ys] == ["foo", "bar"]

    # 断言，确保 "foo" 和 "bar" 这两个 TeX 字符串在 y 轴上只有一个绘制位置
    assert len(ys["foo"]) == len(ys["bar"]) == 1

    # 获取缓存信息
    info = mpl.text._get_text_metrics_with_cache_impl.cache_info()

    # 断言，确保在布局计算时，每个字符串都会缓存未命中（miss），在绘制时都会命中（hit），
    # 但是 "foo\nbar" 由于被绘制两次，所以命中两次
    assert info.hits > info.misses


def test_annotate_offset_fontsize():
    # 测试 offset_fontsize 参数的作用，并确保使用准确的值
    fig, ax = plt.subplots()

    # 定义文本的坐标系
    text_coords = ['offset points', 'offset fontsize']

    # 定义文本的坐标位置
    xy_text = [(10, 10), (1, 1)]

    # 在坐标 (0.5, 0.5) 处添加注释文本，并分别使用不同的 textcoords 和 xytext 参数
    anns = [ax.annotate('test', xy=(0.5, 0.5),
                        xytext=xy_text[i],
                        fontsize='10',
                        xycoords='data',
                        textcoords=text_coords[i]) for i in range(2)]

    # 获取每个注释对象的窗口边界
    points_coords, fontsize_coords = [ann.get_window_extent() for ann in anns]

    # 绘制图形
    fig.canvas.draw()

    # 断言，确保两个文本注释对象的窗口边界相同
    assert str(points_coords) == str(fontsize_coords)


def test_get_set_antialiased():
    # 测试 Text 对象的抗锯齿属性

    # 创建一个 Text 对象，并断言其初始抗锯齿状态与全局设置相同
    txt = Text(.5, .5, "foo\nbar")
    assert txt._antialiased == mpl.rcParams['text.antialiased']
    assert txt.get_antialiased() == mpl.rcParams['text.antialiased']

    # 设置 Text 对象的抗锯齿属性为 True，并进行断言确认
    txt.set_antialiased(True)
    assert txt._antialiased is True
    assert txt.get_antialiased() == txt._antialiased

    # 设置 Text 对象的抗锯齿属性为 False，并进行断言确认
    txt.set_antialiased(False)
    assert txt._antialiased is False
    assert txt.get_antialiased() == txt._antialiased


def test_annotation_antialiased():
    # 测试 Annotation 对象的抗锯齿属性

    # 创建一个抗锯齿为 True 的 Annotation 对象，并进行断言确认
    annot = Annotation("foo\nbar", (.5, .5), antialiased=True)
    assert annot._antialiased is True
    assert annot.get_antialiased() == annot._antialiased

    # 创建一个抗锯齿为 False 的 Annotation 对象，并进行断言确认
    annot2 = Annotation("foo\nbar", (.5, .5), antialiased=False)
    assert annot2._antialiased is False
    assert annot2.get_antialiased() == annot2._antialiased

    # 创建一个抗锯齿为 False 的 Annotation 对象，并将其抗锯齿属性设置为 True，进行断言确认
    annot3 = Annotation("foo\nbar", (.5, .5), antialiased=False)
    annot3.set_antialiased(True)
    assert annot3.get_antialiased() is True
    assert annot3._antialiased is True

    # 创建一个默认抗锯齿属性的 Annotation 对象，并进行断言确认
    annot4 = Annotation("foo\nbar", (.5, .5))
    assert annot4._antialiased == mpl.rcParams['text.antialiased']
@check_figures_equal(extensions=["png"])
def test_annotate_and_offsetfrom_copy_input(fig_test, fig_ref):
    # Both approaches place the text (10, 0) pixels away from the center of the line.

    # 在测试图表上添加子图
    ax = fig_test.add_subplot()
    # 绘制一条直线
    l, = ax.plot([0, 2], [0, 2])
    # 设置偏移的坐标，相对于直线中心的位置
    of_xy = np.array([.5, .5])
    # 在图上注释文本“foo”，使用 OffsetFrom 类型的坐标系
    ax.annotate("foo", textcoords=OffsetFrom(l, of_xy), xytext=(10, 0),
                xy=(0, 0))  # xy is unused.
    
    # 修改 of_xy 数组的值
    of_xy[:] = 1
    
    # 在参考图表上添加子图
    ax = fig_ref.add_subplot()
    # 绘制同样的直线
    l, = ax.plot([0, 2], [0, 2])
    # 设置注释点的坐标
    an_xy = np.array([.5, .5])
    # 在图上注释文本“foo”，使用 "offset points" 类型的坐标系
    ax.annotate("foo", xy=an_xy, xycoords=l, xytext=(10, 0), textcoords="offset points")
    
    # 修改 an_xy 数组的值
    an_xy[:] = 2


@check_figures_equal()
def test_text_antialiased_off_default_vs_manual(fig_test, fig_ref):
    # 在测试图表上添加文本，关闭文本的抗锯齿效果
    fig_test.text(0.5, 0.5, '6 inches x 2 inches', antialiased=False)

    # 设置全局配置，关闭所有文本的抗锯齿效果
    mpl.rcParams['text.antialiased'] = False
    # 在参考图表上添加文本
    fig_ref.text(0.5, 0.5, '6 inches x 2 inches')


@check_figures_equal()
def test_text_antialiased_on_default_vs_manual(fig_test, fig_ref):
    # 在测试图表上添加文本，启用文本的抗锯齿效果
    fig_test.text(0.5, 0.5, '6 inches x 2 inches', antialiased=True)

    # 设置全局配置，启用所有文本的抗锯齿效果
    mpl.rcParams['text.antialiased'] = True
    # 在参考图表上添加文本
    fig_ref.text(0.5, 0.5, '6 inches x 2 inches')


def test_text_annotation_get_window_extent():
    # 创建 DPI 为 100 的图形对象
    figure = Figure(dpi=100)
    # 创建指定分辨率和尺寸的渲染器
    renderer = RendererAgg(200, 200, 100)

    # 只包含文本注释
    annotation = Annotation('test', xy=(0, 0), xycoords='figure pixels')
    annotation.set_figure(figure)

    # 创建文本对象
    text = Text(text='test', x=0, y=0)
    text.set_figure(figure)

    # 获取注释框的窗口范围
    bbox = annotation.get_window_extent(renderer=renderer)

    # 获取文本框的窗口范围
    text_bbox = text.get_window_extent(renderer=renderer)
    # 断言注释框和文本框的宽度相等
    assert bbox.width == text_bbox.width
    # 断言注释框和文本框的高度相等
    assert bbox.height == text_bbox.height

    # 获取文本的宽度、高度和下降
    _, _, d = renderer.get_text_width_height_descent(
        'text', annotation._fontproperties, ismath=False)
    _, _, lp_d = renderer.get_text_width_height_descent(
        'lp', annotation._fontproperties, ismath=False)
    below_line = max(d, lp_d)

    # 这些数字特定于当前 Text 实现
    points = bbox.get_points()
    # 断言注释框左上角 x 坐标为 0.0
    assert points[0, 0] == 0.0
    # 断言注释框右下角 x 坐标等于文本框的宽度
    assert points[1, 0] == text_bbox.width
    # 断言注释框左上角 y 坐标为 -below_line
    assert points[0, 1] == -below_line
    # 断言注释框右下角 y 坐标等于文本框的高度减去下降值
    assert points[1, 1] == text_bbox.height - below_line


def test_text_with_arrow_annotation_get_window_extent():
    headwidth = 21
    # 创建 DPI 为 100 的图形对象和子图
    fig, ax = plt.subplots(dpi=100)
    # 在子图上添加文本
    txt = ax.text(s='test', x=0, y=0)
    # 在子图上添加带箭头的注释
    ann = ax.annotate(
        'test',
        xy=(0.0, 50.0),
        xytext=(50.0, 50.0), xycoords='figure pixels',
        arrowprops={
            'facecolor': 'black', 'width': 2,
            'headwidth': headwidth, 'shrink': 0.0})

    plt.draw()
    # 获取图形对象的渲染器
    renderer = fig.canvas.renderer
    # 获取文本的窗口范围
    text_bbox = txt.get_window_extent(renderer=renderer)
    # 获取带箭头注释的窗口范围
    bbox = ann.get_window_extent(renderer=renderer)
    # 获取箭头的窗口范围
    arrow_bbox = ann.arrow_patch.get_window_extent(renderer)
    # 获取带箭头注释文本的窗口范围
    # 获取注释对象（annotation）的边界框（bounding box）信息
    ann_txt_bbox = Text.get_window_extent(ann)

    # 确保注释框的宽度比文本框宽度多 50 像素
    assert bbox.width == text_bbox.width + 50.0

    # 确保注释文本的边界框与作为Text对象相同字符串的边界框尺寸相同
    assert ann_txt_bbox.height == text_bbox.height
    assert ann_txt_bbox.width == text_bbox.width

    # 计算箭头 + 文本预期的整体边界框
    expected_bbox = mtransforms.Bbox.union([ann_txt_bbox, arrow_bbox])

    # 确保实际的边界框高度与预期边界框高度几乎相等
    assert_almost_equal(bbox.height, expected_bbox.height)
def test_arrow_annotation_get_window_extent():
    # 设置 DPI（每英寸点数）
    dpi = 100
    # 计算每个点的像素数
    dots_per_point = dpi / 72
    # 创建图形对象，并设置图形宽度和高度为2.0英寸
    figure = Figure(dpi=dpi)
    figure.set_figwidth(2.0)
    figure.set_figheight(2.0)
    # 创建渲染器对象，指定宽度和高度为200像素，分辨率为100
    renderer = RendererAgg(200, 200, 100)

    # 创建带箭头的文本注释，箭头尺寸单位为点
    annotation = Annotation(
        '', xy=(0.0, 50.0), xytext=(50.0, 50.0), xycoords='figure pixels',
        arrowprops={
            'facecolor': 'black', 'width': 8, 'headwidth': 10, 'shrink': 0.0})
    annotation.set_figure(figure)
    annotation.draw(renderer)

    # 获取注释框的坐标范围
    bbox = annotation.get_window_extent()
    # 获取注释框的四个顶点坐标
    points = bbox.get_points()

    # 断言注释框的宽度为50.0像素
    assert bbox.width == 50.0
    # 断言注释框的高度接近于10.0点乘以每点像素数
    assert_almost_equal(bbox.height, 10.0 * dots_per_point)
    # 断言注释框左上角顶点的 x 坐标为0.0
    assert points[0, 0] == 0.0
    # 断言注释框左上角顶点的 y 坐标为50.0减去5乘以每点像素数
    assert points[0, 1] == 50.0 - 5 * dots_per_point


def test_empty_annotation_get_window_extent():
    # 创建图形对象，并设置图形宽度和高度为2.0英寸
    figure = Figure(dpi=100)
    figure.set_figwidth(2.0)
    figure.set_figheight(2.0)
    # 创建渲染器对象，指定宽度和高度为200像素，分辨率为100
    renderer = RendererAgg(200, 200, 100)

    # 创建不带箭头的文本注释
    annotation = Annotation(
        '', xy=(0.0, 50.0), xytext=(0.0, 50.0), xycoords='figure pixels')
    annotation.set_figure(figure)
    annotation.draw(renderer)

    # 获取注释框的坐标范围
    bbox = annotation.get_window_extent()
    # 获取注释框的四个顶点坐标
    points = bbox.get_points()

    # 断言注释框左上角顶点的 x 和 y 坐标都为0.0
    assert points[0, 0] == 0.0
    assert points[1, 0] == 0.0
    # 断言注释框右下角顶点的 y 坐标为50.0
    assert points[1, 1] == 50.0
    # 断言注释框左上角顶点的 y 坐标为50.0
    assert points[0, 1] == 50.0


@image_comparison(baseline_images=['basictext_wrap'],
                  extensions=['png'])
def test_basic_wrap():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 设置坐标轴范围
    plt.axis([0, 10, 0, 10])
    # 定义一个长字符串，希望在图形内部自动换行显示
    t = "This is a really long string that I'd rather have wrapped so that" \
        " it doesn't go outside of the figure, but if it's long enough it" \
        " will go off the top or bottom!"
    # 在指定位置绘制文本，左对齐，可以自动换行
    plt.text(4, 1, t, ha='left', rotation=15, wrap=True)
    plt.text(6, 5, t, ha='left', rotation=15, wrap=True)
    plt.text(5, 5, t, ha='right', rotation=-15, wrap=True)
    plt.text(5, 10, t, fontsize=18, style='oblique', ha='center',
             va='top', wrap=True)
    plt.text(3, 4, t, family='serif', style='italic', ha='right', wrap=True)
    plt.text(-1, 0, t, ha='left', rotation=-15, wrap=True)


@image_comparison(baseline_images=['fonttext_wrap'],
                  extensions=['png'])
def test_font_wrap():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 设置坐标轴范围
    plt.axis([0, 10, 0, 10])
    # 定义一个长字符串，希望在图形内部自动换行显示
    t = "This is a really long string that I'd rather have wrapped so that" \
        " it doesn't go outside of the figure, but if it's long enough it" \
        " will go off the top or bottom!"
    # 在指定位置绘制文本，可以自动换行，并设置不同字体属性
    plt.text(4, -1, t, fontsize=18, family='serif', ha='left', rotation=15,
             wrap=True)
    plt.text(6, 5, t, family='sans serif', ha='left', rotation=15, wrap=True)
    plt.text(5, 10, t, weight='heavy', ha='center', va='top', wrap=True)
    plt.text(3, 4, t, family='monospace', ha='right', wrap=True)
    plt.text(-1, 0, t, fontsize=14, style='italic', ha='left', rotation=-15,
             wrap=True)
```