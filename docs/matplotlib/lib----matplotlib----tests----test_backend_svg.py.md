# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_backend_svg.py`

```
# 导入 datetime 模块，用于处理日期和时间
import datetime
# 从 io 模块中导入 BytesIO 类，用于操作二进制数据的内存缓冲区
from io import BytesIO
# 从 pathlib 模块中导入 Path 类，用于处理文件路径
from pathlib import Path
# 导入 xml.etree.ElementTree 模块，用于操作和处理 XML 数据
import xml.etree.ElementTree
# 导入 xml.parsers.expat 模块，用于解析 XML 数据
import xml.parsers.expat

# 导入 pytest 模块，用于编写和运行测试
import pytest

# 导入 numpy 模块，并使用别名 np
import numpy as np

# 导入 matplotlib 库，并从中导入 Figure、Text 等类和模块
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.text import Text
import matplotlib.pyplot as plt
# 从 matplotlib.testing.decorators 模块中导入装饰器函数 check_figures_equal、image_comparison
from matplotlib.testing.decorators import check_figures_equal, image_comparison
# 导入 matplotlib.testing._markers 模块，用于标记测试需要使用 TeX
from matplotlib.testing._markers import needs_usetex
# 导入 matplotlib.font_manager 模块，用于管理和操作字体
from matplotlib import font_manager as fm
# 从 matplotlib.offsetbox 模块中导入 OffsetImage、AnnotationBbox 等类和模块
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)


# 定义测试函数 test_visibility
def test_visibility():
    # 创建一个新的图形对象和轴对象
    fig, ax = plt.subplots()

    # 生成一组等间隔的数据点
    x = np.linspace(0, 4 * np.pi, 50)
    # 计算正弦函数值
    y = np.sin(x)
    # 创建与 y 相同维度的误差条数据
    yerr = np.ones_like(y)

    # 绘制带有误差条的数据点，并将返回的 Artist 对象保存在 a, b, c 中
    a, b, c = ax.errorbar(x, y, yerr=yerr, fmt='ko')
    # 循环遍历 b 中的每个 Artist 对象，并设置其可见性为 False
    for artist in b:
        artist.set_visible(False)

    # 使用 BytesIO 创建一个二进制数据流对象 fd
    with BytesIO() as fd:
        # 将图形保存为 SVG 格式到二进制数据流中
        fig.savefig(fd, format='svg')
        # 获取保存后的二进制数据
        buf = fd.getvalue()

    # 创建一个 XML 解析器对象 parser
    parser = xml.parsers.expat.ParserCreate()
    # 解析 SVG 数据 buf，如果数据无效将引发 ExpatError
    parser.Parse(buf)  # this will raise ExpatError if the svg is invalid


# 使用 image_comparison 装饰器，比较生成的图像是否与参考图像一致
@image_comparison(['fill_black_with_alpha.svg'], remove_text=True)
def test_fill_black_with_alpha():
    # 创建一个新的图形对象和轴对象
    fig, ax = plt.subplots()
    # 绘制散点图，指定颜色为黑色、透明度为 0.1，点的大小为 10000
    ax.scatter(x=[0, 0.1, 1], y=[0, 0, 0], c='k', alpha=0.1, s=10000)


# 使用 image_comparison 装饰器，比较生成的图像是否与参考图像一致
@image_comparison(['noscale'], remove_text=True)
def test_noscale():
    # 创建一个网格状的数据
    X, Y = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1))
    # 计算网格点的 sin(Y^2) 值
    Z = np.sin(Y ** 2)

    # 创建一个新的图形对象和轴对象
    fig, ax = plt.subplots()
    # 绘制灰度图像，禁用插值
    ax.imshow(Z, cmap='gray', interpolation='none')


# 定义测试函数 test_text_urls
def test_text_urls():
    # 创建一个新的图形对象
    fig = plt.figure()

    # 设置测试 URL
    test_url = "http://test_text_urls.matplotlib.org"
    # 设置图形的总标题，并指定超链接地址为 test_url
    fig.suptitle("test_text_urls", url=test_url)

    # 使用 BytesIO 创建一个二进制数据流对象 fd
    with BytesIO() as fd:
        # 将图形保存为 SVG 格式到二进制数据流中
        fig.savefig(fd, format='svg')
        # 获取保存后的二进制数据，并解码为字符串
        buf = fd.getvalue().decode()

    # 构建预期的 SVG 标签字符串
    expected = f'<a xlink:href="{test_url}">'
    # 断言预期的标签是否存在于 SVG 数据 buf 中
    assert expected in buf


# 使用 image_comparison 装饰器，比较生成的图像是否与参考图像一致
@image_comparison(['bold_font_output.svg'])
def test_bold_font_output():
    # 创建一个新的图形对象和轴对象
    fig, ax = plt.subplots()
    # 绘制简单的折线图
    ax.plot(np.arange(10), np.arange(10))
    # 设置 x 轴标签的文本和字体加粗
    ax.set_xlabel('nonbold-xlabel')
    # 设置 y 轴标签的文本和字体加粗
    ax.set_ylabel('bold-ylabel', fontweight='bold')
    # 设置图形标题文本和字体加粗
    ax.set_title('bold-title', fontweight='bold')


# 使用 image_comparison 装饰器，比较生成的图像是否与参考图像一致
@image_comparison(['bold_font_output_with_none_fonttype.svg'])
def test_bold_font_output_with_none_fonttype():
    # 设置 SVG 渲染时不嵌入字体文件
    plt.rcParams['svg.fonttype'] = 'none'
    # 创建一个新的图形对象和轴对象
    fig, ax = plt.subplots()
    # 绘制简单的折线图
    ax.plot(np.arange(10), np.arange(10))
    # 设置 x 轴标签的文本和字体加粗
    ax.set_xlabel('nonbold-xlabel')
    # 设置 y 轴标签的文本和字体加粗
    ax.set_ylabel('bold-ylabel', fontweight='bold')
    # 设置图形标题文本和字体加粗
    ax.set_title('bold-title', fontweight='bold')


# 使用 check_figures_equal 装饰器，检查两个图形对象是否相等
@check_figures_equal(tol=20)
def test_rasterized(fig_test, fig_ref):
    # 生成时间序列数据
    t = np.arange(0, 100) * (2.3)
    x = np.cos(t)
    y = np.sin(t)

    # 创建参考图形的轴对象
    ax_ref = fig_ref.subplots()
    # 绘制参考图形的曲线
    ax_ref.plot(x, y, "-", c="r", lw=10)
    ax_ref.plot(x+1, y, "-", c="b", lw=10)

    # 创建测试图形的轴对象
    ax_test = fig_test.subplots()
    # 绘制测试图形的曲线，并开启光栅化选项
    ax_test.plot(x, y, "-", c="r", lw=10, rasterized=True)
    ax_test.plot(x+1, y, "-", c="b", lw=10, rasterized=True)


# 使用 check_figures_equal 装饰器，检查两个图形对象是否相等
@check_figures_equal()
def test_rasterized_ordering(fig_test, fig_ref):
    # 生成时间序列数据
    t = np.arange(0, 100) * (2.3)
    x = np.cos(t)
    y = np.sin(t)

    # 创建参考图形的轴对象
    ax_ref = fig_ref.subplots()
    # 设置参考图表的 X 轴范围为 0 到 3
    ax_ref.set_xlim(0, 3)
    # 设置参考图表的 Y 轴范围为 -1.1 到 1.1
    ax_ref.set_ylim(-1.1, 1.1)
    # 在参考图表上绘制红色线条，线宽为 10，使用光栅化处理
    ax_ref.plot(x, y, "-", c="r", lw=10, rasterized=True)
    # 在参考图表上绘制蓝色线条，线宽为 10，不使用光栅化处理
    ax_ref.plot(x+1, y, "-", c="b", lw=10, rasterized=False)
    # 在参考图表上绘制绿色线条，线宽为 10，使用光栅化处理
    ax_ref.plot(x+2, y, "-", c="g", lw=10, rasterized=True)
    # 在参考图表上绘制品红色线条，线宽为 10，使用光栅化处理
    ax_ref.plot(x+3, y, "-", c="m", lw=10, rasterized=True)

    # 创建测试图表的子图 ax_test
    ax_test = fig_test.subplots()
    # 设置测试图表的 X 轴范围为 0 到 3
    ax_test.set_xlim(0, 3)
    # 设置测试图表的 Y 轴范围为 -1.1 到 1.1
    ax_test.set_ylim(-1.1, 1.1)
    # 在测试图表上绘制红色线条，线宽为 10，使用光栅化处理，层次顺序为 1.1
    ax_test.plot(x, y, "-", c="r", lw=10, rasterized=True, zorder=1.1)
    # 在测试图表上绘制绿色线条，线宽为 10，使用光栅化处理，层次顺序为 1.3
    ax_test.plot(x+2, y, "-", c="g", lw=10, rasterized=True, zorder=1.3)
    # 在测试图表上绘制品红色线条，线宽为 10，使用光栅化处理，层次顺序为 1.4
    ax_test.plot(x+3, y, "-", c="m", lw=10, rasterized=True, zorder=1.4)
    # 在测试图表上绘制蓝色线条，线宽为 10，不使用光栅化处理，层次顺序为 1.2
    ax_test.plot(x+1, y, "-", c="b", lw=10, rasterized=False, zorder=1.2)
# 使用装饰器定义一个测试函数，用于比较两个图形的相等性，接受参数tol（容差值）和extensions（文件扩展名列表）
@check_figures_equal(tol=5, extensions=['svg', 'pdf'])
def test_prevent_rasterization(fig_test, fig_ref):
    # 设置标记的位置
    loc = [0.05, 0.05]

    # 在参考图中创建子图
    ax_ref = fig_ref.subplots()

    # 在参考图的子图中绘制一个带有标记的点
    ax_ref.plot([loc[0]], [loc[1]], marker="x", c="black", zorder=2)

    # 创建一个文本区域对象
    b = mpl.offsetbox.TextArea("X")
    # 创建一个带注释的文本框，并添加到参考图的子图中
    abox = mpl.offsetbox.AnnotationBbox(b, loc, zorder=2.1)
    ax_ref.add_artist(abox)

    # 在测试图中创建子图
    ax_test = fig_test.subplots()
    # 在测试图的子图中绘制一个带有标记的点，并设置为光栅化
    ax_test.plot([loc[0]], [loc[1]], marker="x", c="black", zorder=2,
                 rasterized=True)

    # 创建一个文本区域对象
    b = mpl.offsetbox.TextArea("X")
    # 创建一个带注释的文本框，并添加到测试图的子图中
    abox = mpl.offsetbox.AnnotationBbox(b, loc, zorder=2.1)
    ax_test.add_artist(abox)


# 定义一个测试函数，用于计算图形中特定标签的出现次数
def test_count_bitmaps():
    def count_tag(fig, tag):
        # 使用 BytesIO 创建一个内存文件对象
        with BytesIO() as fd:
            # 将图形保存为 SVG 格式到内存文件对象中
            fig.savefig(fd, format='svg')
            # 将文件对象中的内容解码为字符串
            buf = fd.getvalue().decode()
        # 返回特定标签在解码后的字符串中出现的次数
        return buf.count(f"<{tag}")

    # 创建第一个图形对象
    fig1 = plt.figure()
    # 在第一个图形对象中添加一个子图
    ax1 = fig1.add_subplot(1, 1, 1)
    # 关闭子图的坐标轴
    ax1.set_axis_off()
    # 在子图中绘制5条不光栅化的直线
    for n in range(5):
        ax1.plot([0, 20], [0, n], "b-", rasterized=False)
    # 断言图形中没有包含“image”标签的对象
    assert count_tag(fig1, "image") == 0
    # 断言图形中包含6个“path”标签的对象（轴和线条）
    assert count_tag(fig1, "path") == 6

    # 创建第二个图形对象
    fig2 = plt.figure()
    # 在第二个图形对象中添加一个子图
    ax2 = fig2.add_subplot(1, 1, 1)
    # 关闭子图的坐标轴
    ax2.set_axis_off()
    # 在子图中绘制5条光栅化的直线
    for n in range(5):
        ax2.plot([0, 20], [0, n], "b-", rasterized=True)
    # 断言图形中包含1个“image”标签的对象
    assert count_tag(fig2, "image") == 1
    # 断言图形中包含1个“path”标签的对象（轴）
    assert count_tag(fig2, "path") == 1

    # 创建第三个图形对象
    fig3 = plt.figure()
    # 在第三个图形对象中添加一个子图
    ax3 = fig3.add_subplot(1, 1, 1)
    # 关闭子图的坐标轴
    ax3.set_axis_off()
    # 在子图中同时绘制光栅化和不光栅化的直线
    for n in range(5):
        ax3.plot([0, 20], [n, 0], "b-", rasterized=False)
        ax3.plot([0, 20], [0, n], "b-", rasterized=True)
    # 断言图形中包含5个“image”标签的对象
    assert count_tag(fig3, "image") == 5
    # 断言图形中包含6个“path”标签的对象（轴和线条）
    assert count_tag(fig3, "path") == 6

    # 创建第四个图形对象
    fig4 = plt.figure()
    # 在第四个图形对象中添加一个子图
    ax4 = fig4.add_subplot(1, 1, 1)
    # 关闭子图的坐标轴
    ax4.set_axis_off()
    # 将整个子图设为光栅化
    ax4.set_rasterized(True)
    # 在子图中同时绘制光栅化和不光栅化的直线
    for n in range(5):
        ax4.plot([0, 20], [n, 0], "b-", rasterized=False)
        ax4.plot([0, 20], [0, n], "b-", rasterized=True)
    # 断言图形中包含1个“image”标签的对象
    assert count_tag(fig4, "image") == 1
    # 断言图形中包含1个“path”标签的对象（轴）
    assert count_tag(fig4, "path") == 1

    # 创建第五个图形对象
    fig5 = plt.figure()
    # 将图形对象的 suppressComposite 属性设置为 True
    fig5.suppressComposite = True
    # 在第五个图形对象中添加一个子图
    ax5 = fig5.add_subplot(1, 1, 1)
    # 关闭子图的坐标轴
    ax5.set_axis_off()
    # 在子图中绘制5条光栅化的直线
    for n in range(5):
        ax5.plot([0, 20], [0, n], "b-", rasterized=True)
    # 断言图形中包含5个“image”标签的对象
    assert count_tag(fig5, "image") == 5
    # 断言图形中包含1个“path”标签的对象（轴）
    assert count_tag(fig5, "path") == 1  # axis patch


# 使用默认样式上下文管理器和 LaTeX 渲染文本，测试绘制 Unicode 文本
@mpl.style.context('default')
@needs_usetex
def test_unicode_won():
    # 创建一个图形对象
    fig = Figure()
    # 在图形对象中添加一个文本对象，包含 LaTeX 公式 "\textwon"
    fig.text(.5, .5, r'\textwon', usetex=True)

    # 使用 BytesIO 创建一个内存文件对象
    with BytesIO() as fd:
        # 将图形保存为 SVG 格式到内存文件对象中
        fig.savefig(fd, format='svg')
        # 获取内存文件对象的内容
        buf = fd.getvalue()

    # 使用 ElementTree 解析 SVG 内容
    tree = xml.etree.ElementTree.fromstring(buf)
    # 设置 SVG 命名空间
    ns = 'http://www.w3.org/2000/svg'
    # 定义预期的路径 ID
    won_id = 'SFSS3583-8e'
    # 断言 SVG 树中存在具有特定路径 ID 的路径元素
    assert len(tree.findall(f'.//{{{ns}}}path[@d][@id="{won_id}"]')) == 1
    # 使用断言检查格式化后的字符串是否存在于XML树中指定命名空间的"use"元素的属性值中
    assert f'#{won_id}' in tree.find(f'.//{{{ns}}}use').attrib.values()
def test_svgnone_with_data_coordinates():
    # 更新全局绘图参数，禁用SVG字体嵌入，设置字体压缩
    plt.rcParams.update({'svg.fonttype': 'none', 'font.stretch': 'condensed'})
    # 预期的文本内容，用于验证输出SVG中是否包含此文本
    expected = 'Unlikely to appear by chance'

    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 在轴上添加文本，指定日期和位置
    ax.text(np.datetime64('2019-06-30'), 1, expected)
    # 设置X轴的限制范围为2019年全年
    ax.set_xlim(np.datetime64('2019-01-01'), np.datetime64('2019-12-31'))
    # 设置Y轴的限制范围为0到2
    ax.set_ylim(0, 2)

    # 使用字节流创建一个内存文件对象
    with BytesIO() as fd:
        # 将图形保存为SVG格式到内存文件
        fig.savefig(fd, format='svg')
        # 将文件指针移到文件开头
        fd.seek(0)
        # 读取整个文件内容并解码为字符串
        buf = fd.read().decode()

    # 断言预期的文本和"condensed"字样都在SVG内容中
    assert expected in buf and "condensed" in buf


def test_gid():
    """Test that object gid appears in output svg."""
    # 导入需要的模块和类
    from matplotlib.offsetbox import OffsetBox
    from matplotlib.axis import Tick

    # 创建一个新的图形对象
    fig = plt.figure()

    # 添加三个子图，分别为普通子图、极坐标子图和3D子图
    ax1 = fig.add_subplot(131)
    ax1.imshow([[1., 2.], [2., 3.]], aspect="auto")
    ax1.scatter([1, 2, 3], [1, 2, 3], label="myscatter")
    ax1.plot([2, 3, 1], label="myplot")
    ax1.legend()
    ax1a = ax1.twinx()
    ax1a.bar([1, 2, 3], [1, 2, 3])

    ax2 = fig.add_subplot(132, projection="polar")
    ax2.plot([0, 1.5, 3], [1, 2, 3])

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.plot([1, 2], [1, 2], [1, 2])

    # 在绘制图形之前进行画布绘制
    fig.canvas.draw()

    # 创建一个空的字典，用于存储对象的gid和对象本身的对应关系
    gdic = {}
    # 遍历图形中的所有对象，包括自身
    for idx, obj in enumerate(fig.findobj(include_self=True)):
        # 检查对象是否可见
        if obj.get_visible():
            # 生成唯一的gid
            gid = f"test123{obj.__class__.__name__}_{idx}"
            # 将gid和对象存储到字典中
            gdic[gid] = obj
            # 设置对象的gid属性
            obj.set_gid(gid)

    # 使用字节流创建一个内存文件对象
    with BytesIO() as fd:
        # 将图形保存为SVG格式到内存文件
        fig.savefig(fd, format='svg')
        # 从内存文件中读取全部内容并解码为字符串
        buf = fd.getvalue().decode()

    # 定义一个函数，用于判断是否应包含指定的gid和对象
    def include(gid, obj):
        # 排除一些不应出现在SVG中的对象
        if isinstance(obj, OffsetBox):
            return False
        if isinstance(obj, Text):
            if obj.get_text() == "":
                return False
            elif obj.axes is None:
                return False
        if isinstance(obj, plt.Line2D):
            xdata, ydata = obj.get_data()
            if len(xdata) == len(ydata) == 1:
                return False
            elif not hasattr(obj, "axes") or obj.axes is None:
                return False
        if isinstance(obj, Tick):
            loc = obj.get_loc()
            if loc == 0:
                return False
            vi = obj.get_view_interval()
            if loc < min(vi) or loc > max(vi):
                return False
        return True

    # 遍历存储gid和对象的字典
    for gid, obj in gdic.items():
        # 如果应包含此gid和对象，则断言gid存在于SVG内容中
        if include(gid, obj):
            assert gid in buf


def test_savefig_tight():
    # 检查禁用绘图渲染器时，关闭/打开组的正确性
    plt.savefig(BytesIO(), format="svgz", bbox_inches="tight")


def test_url():
    # 测试对象的URL在输出的SVG中是否出现

    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()

    # 添加散点图并设置每个点的URL
    s = ax.scatter([1, 2, 3], [4, 5, 6])
    s.set_urls(['https://example.com/foo', 'https://example.com/bar', None])

    # 添加线图并设置其URL
    p, = plt.plot([1, 3], [6, 5])
    p.set_url('https://example.com/baz')

    # 创建一个字节流对象
    b = BytesIO()
    # 将图形保存为SVG格式到字节流中
    fig.savefig(b, format='svg')
    # 读取字节流中的全部内容
    b = b.getvalue()
    # 对列表中每个字节字符串进行断言检查，确保其与 b 中的某个元素拼接后的结果存在于 b 中
    for v in [b'foo', b'bar', b'baz']:
        assert b'https://example.com/' + v in b
# 设置环境变量'SOURCE_DATE_EPOCH'为'19680801'，用于模拟固定的时间戳
def test_url_tick(monkeypatch):
    monkeypatch.setenv('SOURCE_DATE_EPOCH', '19680801')

    # 创建一个包含图形和轴的 subplot 对象 fig1 和 ax
    fig1, ax = plt.subplots()
    # 在轴上绘制散点图
    ax.scatter([1, 2, 3], [4, 5, 6])
    # 遍历 y 轴的主要刻度，设置每个刻度的 URL
    for i, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.set_url(f'https://example.com/{i}')

    # 创建另一个 subplot 对象 fig2 和 ax
    fig2, ax = plt.subplots()
    # 在轴上绘制散点图
    ax.scatter([1, 2, 3], [4, 5, 6])
    # 遍历 y 轴的主要刻度，设置每个刻度标签的 URL
    for i, tick in enumerate(ax.yaxis.get_major_ticks()):
        tick.label1.set_url(f'https://example.com/{i}')
        tick.label2.set_url(f'https://example.com/{i}')

    # 创建一个字节流对象 b1，将 fig1 保存为 SVG 格式
    b1 = BytesIO()
    fig1.savefig(b1, format='svg')
    b1 = b1.getvalue()

    # 创建一个字节流对象 b2，将 fig2 保存为 SVG 格式
    b2 = BytesIO()
    fig2.savefig(b2, format='svg')
    b2 = b2.getvalue()

    # 断言：确保每个刻度的 URL 被编码为 ASCII 并存在于 b1 中
    for i in range(len(ax.yaxis.get_major_ticks())):
        assert f'https://example.com/{i}'.encode('ascii') in b1
    # 断言：确保 b1 和 b2 的内容相等
    assert b1 == b2


# 测试保存 SVG 文件时的默认元数据设置
def test_svg_default_metadata(monkeypatch):
    # 设置环境变量'SOURCE_DATE_EPOCH'为'19680801'
    monkeypatch.setenv('SOURCE_DATE_EPOCH', '19680801')

    # 创建一个包含图形和轴的 subplot 对象 fig 和 ax
    fig, ax = plt.subplots()
    # 使用 BytesIO 创建一个文件对象 fd
    with BytesIO() as fd:
        # 将 fig 保存为 SVG 格式到 fd 中
        fig.savefig(fd, format='svg')
        # 获取 fd 的内容并解码为字符串 buf
        buf = fd.getvalue().decode()

    # 断言：检查 buf 中是否包含正确的 Creator 元数据
    assert mpl.__version__ in buf
    # 断言：检查 buf 中是否包含正确的 Date 元数据
    assert '1970-08-16' in buf
    # 断言：检查 buf 中是否包含正确的 Format 元数据
    assert 'image/svg+xml' in buf
    # 断言：检查 buf 中是否包含正确的 Type 元数据
    assert 'StillImage' in buf

    # 使用 BytesIO 创建一个文件对象 fd
    with BytesIO() as fd:
        # 将 fig 保存为 SVG 格式到 fd 中，清除默认元数据
        fig.savefig(fd, format='svg', metadata={'Date': None, 'Creator': None,
                                                'Format': None, 'Type': None})
        # 获取 fd 的内容并解码为字符串 buf
        buf = fd.getvalue().decode()

    # 断言：检查 buf 中是否不包含 Creator 元数据
    assert mpl.__version__ not in buf
    # 断言：检查 buf 中是否不包含 Date 元数据
    assert '1970-08-16' not in buf
    # 断言：检查 buf 中是否不包含 Format 元数据
    assert 'image/svg+xml' not in buf
    # 断言：检查 buf 中是否不包含 Type 元数据
    assert 'StillImage' not in buf


# 测试清除默认 SVG 文件元数据
def test_svg_clear_default_metadata(monkeypatch):
    # 设置环境变量'SOURCE_DATE_EPOCH'为'19680801'
    monkeypatch.setenv('SOURCE_DATE_EPOCH', '19680801')

    # 定义包含默认元数据的字典
    metadata_contains = {'creator': mpl.__version__, 'date': '1970-08-16',
                         'format': 'image/svg+xml', 'type': 'StillImage'}

    # 创建一个包含图形和轴的 subplot 对象 fig 和 ax
    fig, ax = plt.subplots()
    # 遍历元数据包含的每个名称
    for name in metadata_contains:
        # 创建一个字节流对象
        with BytesIO() as fd:
            # 将图形保存为SVG格式，并将指定元数据添加到图形中
            fig.savefig(fd, format='svg', metadata={name.title(): None})
            # 从字节流中获取数据并解码为字符串
            buf = fd.getvalue().decode()

        # 解析SVG数据并生成ElementTree的根元素
        root = xml.etree.ElementTree.fromstring(buf)
        # 在SVG的metadata部分查找RDF命名空间下的Work元素
        work, = root.findall(f'./{SVGNS}metadata/{RDFNS}RDF/{CCNS}Work')

        # 遍历元数据名称列表，查找对应的数据项
        for key in metadata_contains:
            # 查找工作元素中特定的元数据项
            data = work.findall(f'./{DCNS}{key}')
            if key == name:
                # 断言当前处理的元数据项不存在，因为我们已经清除它
                assert not data
                # 继续处理下一个元数据项
                continue

            # 断言其他所有元数据项应该存在
            # 因为元素仍在工作中，因此我们可以查找并转换为XML字符串
            data, = data
            xmlstr = xml.etree.ElementTree.tostring(data, encoding="unicode")
            # 断言特定的元数据值应该在XML字符串中
            assert metadata_contains[key] in xmlstr
def test_svg_clear_all_metadata():
    # 确保将所有默认元数据设置为 `None` 后，输出中删除元数据标签。

    # 创建图像和轴对象
    fig, ax = plt.subplots()
    
    # 使用 BytesIO 创建临时字节流对象
    with BytesIO() as fd:
        # 将图像保存为 SVG 格式，并设置元数据为指定的值
        fig.savefig(fd, format='svg', metadata={'Date': None, 'Creator': None,
                                                'Format': None, 'Type': None})
        # 获取字节流的值并解码为字符串
        buf = fd.getvalue().decode()

    # SVG 的命名空间
    SVGNS = '{http://www.w3.org/2000/svg}'

    # 解析 SVG 文件内容为 XML 元素树的根节点
    root = xml.etree.ElementTree.fromstring(buf)
    
    # 断言没有找到指定命名空间下的 metadata 标签
    assert not root.findall(f'./{SVGNS}metadata')


def test_svg_metadata():
    # 单值的元数据项
    single_value = ['Coverage', 'Identifier', 'Language', 'Relation', 'Source',
                    'Title', 'Type']
    # 多值的元数据项
    multi_value = ['Contributor', 'Creator', 'Keywords', 'Publisher', 'Rights']
    
    # 构建完整的元数据字典
    metadata = {
        'Date': [datetime.date(1968, 8, 1),
                 datetime.datetime(1968, 8, 2, 1, 2, 3)],
        'Description': 'description\ntext',
        **{k: f'{k} foo' for k in single_value},
        **{k: [f'{k} bar', f'{k} baz'] for k in multi_value},
    }

    # 创建图像对象
    fig = plt.figure()
    with BytesIO() as fd:
        # 将图像保存为 SVG 格式，并使用完整的元数据字典
        fig.savefig(fd, format='svg', metadata=metadata)
        buf = fd.getvalue().decode()

    # 定义 SVG、RDF、CC 和 DC 的命名空间
    SVGNS = '{http://www.w3.org/2000/svg}'
    RDFNS = '{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
    CCNS = '{http://creativecommons.org/ns#}'
    DCNS = '{http://purl.org/dc/elements/1.1/}'

    # 解析 SVG 文件内容为 XML 元素树的根节点
    root = xml.etree.ElementTree.fromstring(buf)
    
    # 查找并获取 RDF 元素
    rdf, = root.findall(f'./{SVGNS}metadata/{RDFNS}RDF')

    # 检查单值元素
    titles = [node.text for node in root.findall(f'./{SVGNS}title')]
    assert titles == [metadata['Title']]
    types = [node.attrib[f'{RDFNS}resource']
             for node in rdf.findall(f'./{CCNS}Work/{DCNS}type')]
    assert types == [metadata['Type']]
    for k in ['Description', *single_value]:
        if k == 'Type':
            continue
        values = [node.text
                  for node in rdf.findall(f'./{CCNS}Work/{DCNS}{k.lower()}')]
        assert values == [metadata[k]]

    # 检查多值元素
    for k in multi_value:
        if k == 'Keywords':
            continue
        values = [
            node.text
            for node in rdf.findall(
                f'./{CCNS}Work/{DCNS}{k.lower()}/{CCNS}Agent/{DCNS}title')]
        assert values == metadata[k]

    # 检查特殊元素
    dates = [node.text for node in rdf.findall(f'./{CCNS}Work/{DCNS}date')]
    assert dates == ['1968-08-01/1968-08-02T01:02:03']

    values = [node.text for node in
              rdf.findall(f'./{CCNS}Work/{DCNS}subject/{RDFNS}Bag/{RDFNS}li')]
    assert values == metadata['Keywords']


@image_comparison(["multi_font_aspath.svg"], tol=1.8)
def test_multi_font_type3():
    # 创建特定字体属性
    fp = fm.FontProperties(family=["WenQuanYi Zen Hei"])
    # 检查特定字体文件是否存在
    if Path(fm.findfont(fp)).name != "wqy-zenhei.ttc":
        pytest.skip("Font may be missing")

    # 设置全局字体属性
    plt.rc('font', family=['DejaVu Sans', 'WenQuanYi Zen Hei'], size=27)
    # 设置 matplotlib 使用 SVG 输出，并将字体类型设为路径格式
    plt.rc('svg', fonttype='path')
    
    # 创建一个新的图形对象
    fig = plt.figure()
    
    # 在图形上添加文本，指定位置 (0.15, 0.475)，文本内容为 "There are 几个汉字 in between!"
    fig.text(0.15, 0.475, "There are 几个汉字 in between!")
@image_comparison(["multi_font_astext.svg"])
# 声明一个装饰器，用于测试图像的比较，比较结果将保存在文件 multi_font_astext.svg 中
def test_multi_font_type42():
    # 创建一个字体属性对象，使用 WenQuanYi Zen Hei 字体
    fp = fm.FontProperties(family=["WenQuanYi Zen Hei"])
    # 检查找到的字体文件是否为 wqy-zenhei.ttc，否则跳过测试
    if Path(fm.findfont(fp)).name != "wqy-zenhei.ttc":
        pytest.skip("Font may be missing")

    # 创建一个新的图像对象
    fig = plt.figure()
    # 设置 SVG 输出的字体类型为 'none'
    plt.rc('svg', fonttype='none')

    # 设置默认字体家族为 'DejaVu Sans', 'WenQuanYi Zen Hei'，字体大小为 27
    plt.rc('font', family=['DejaVu Sans', 'WenQuanYi Zen Hei'], size=27)
    # 在图像中添加文本，包含中文字符 "几个汉字"
    fig.text(0.15, 0.475, "There are 几个汉字 in between!")


@pytest.mark.parametrize('metadata,error,message', [
    ({'Date': 1}, TypeError, "Invalid type for Date metadata. Expected str"),
    ({'Date': [1]}, TypeError,
     "Invalid type for Date metadata. Expected iterable"),
    ({'Keywords': 1}, TypeError,
     "Invalid type for Keywords metadata. Expected str"),
    ({'Keywords': [1]}, TypeError,
     "Invalid type for Keywords metadata. Expected iterable"),
    ({'Creator': 1}, TypeError,
     "Invalid type for Creator metadata. Expected str"),
    ({'Creator': [1]}, TypeError,
     "Invalid type for Creator metadata. Expected iterable"),
    ({'Title': 1}, TypeError,
     "Invalid type for Title metadata. Expected str"),
    ({'Format': 1}, TypeError,
     "Invalid type for Format metadata. Expected str"),
    ({'Foo': 'Bar'}, ValueError, "Unknown metadata key"),
    ])
# 测试不正确的 SVG 元数据处理
def test_svg_incorrect_metadata(metadata, error, message):
    # 使用 pytest 检查保存 SVG 文件时是否会抛出预期的异常和匹配的错误消息
    with pytest.raises(error, match=message), BytesIO() as fd:
        # 创建一个新的图像对象
        fig = plt.figure()
        # 将图像保存为 SVG 格式，并传入指定的元数据
        fig.savefig(fd, format='svg', metadata=metadata)


def test_svg_escape():
    # 创建一个新的图像对象
    fig = plt.figure()
    # 在图像中添加文本，包含需要转义的字符 "<'\"&>"
    fig.text(0.5, 0.5, "<\'\"&>", gid="<\'\"&>")
    # 使用 BytesIO 创建一个内存缓冲区
    with BytesIO() as fd:
        # 将图像保存为 SVG 格式
        fig.savefig(fd, format='svg')
        # 从内存缓冲区中获取保存的 SVG 数据，并解码为字符串格式
        buf = fd.getvalue().decode()
        # 断言转义后的字符 "&lt;&apos;&quot;&amp;&gt;" 是否存在于 SVG 数据中
        assert '&lt;&apos;&quot;&amp;&gt;"' in buf


@pytest.mark.parametrize("font_str", [
    "'DejaVu Sans', 'WenQuanYi Zen Hei', 'Arial', sans-serif",
    "'DejaVu Serif', 'WenQuanYi Zen Hei', 'Times New Roman', serif",
    "'Arial', 'WenQuanYi Zen Hei', cursive",
    "'Impact', 'WenQuanYi Zen Hei', fantasy",
    "'DejaVu Sans Mono', 'WenQuanYi Zen Hei', 'Courier New', monospace",
    # These do not work because the logic to get the font metrics will not find
    # WenQuanYi as the fallback logic stops with the first fallback font:
    # "'DejaVu Sans Mono', 'Courier New', 'WenQuanYi Zen Hei', monospace",
    # "'DejaVu Sans', 'Arial', 'WenQuanYi Zen Hei', sans-serif",
    # "'DejaVu Serif', 'Times New Roman', 'WenQuanYi Zen Hei',  serif",
])
@pytest.mark.parametrize("include_generic", [True, False])
# 测试 SVG 字体字符串的处理
def test_svg_font_string(font_str, include_generic):
    # 创建一个字体属性对象，使用 WenQuanYi Zen Hei 字体
    fp = fm.FontProperties(family=["WenQuanYi Zen Hei"])
    # 检查找到的字体文件是否为 wqy-zenhei.ttc，否则跳过测试
    if Path(fm.findfont(fp)).name != "wqy-zenhei.ttc":
        pytest.skip("Font may be missing")

    # 解析字体字符串，将其拆分为显示字体、其他字体和通用字体
    explicit, *rest, generic = map(
        lambda x: x.strip("'"), font_str.split(", ")
    )
    # 通过通用字体的长度设置字体大小
    size = len(generic)
    # 如果需要包含通用字体，则将其加入其他字体列表中
    if include_generic:
        rest = rest + [generic]
    # 根据设置修改默认字体配置
    plt.rcParams[f"font.{generic}"] = rest
    plt.rcParams["font.size"] = size
    plt.rcParams["svg.fonttype"] = "none"
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()

    # 根据指定的字体通用名称确定通用选项列表
    if generic == "sans-serif":
        generic_options = ["sans", "sans-serif", "sans serif"]
    else:
        generic_options = [generic]

    # 遍历通用选项列表，测试回退是否有效
    for generic_name in generic_options:
        # 在图中添加文本，测试指定字体族和通用字体名称
        ax.text(0.5, 0.5, "There are 几个汉字 in between!",
                family=[explicit, generic_name], ha="center")
        
        # 在图中添加文本，测试去重是否有效
        ax.text(0.5, 0.1, "There are 几个汉字 in between!",
                family=[explicit, *rest, generic_name], ha="center")
    
    # 关闭轴的坐标显示
    ax.axis("off")

    # 使用字节流保存图形为 SVG 格式
    with BytesIO() as fd:
        fig.savefig(fd, format="svg")
        buf = fd.getvalue()

    # 解析 SVG 格式的图像数据，生成 XML 元素树
    tree = xml.etree.ElementTree.fromstring(buf)

    # 定义 XML 命名空间
    ns = "http://www.w3.org/2000/svg"

    # 初始化文本元素计数
    text_count = 0

    # 遍历 XML 树中所有的文本元素
    for text_element in tree.findall(f".//{{{ns}}}text"):
        text_count += 1
        
        # 从文本元素的样式中提取字体信息
        font_info = dict(
            map(lambda x: x.strip(), _.strip().split(":"))
            for _ in dict(text_element.items())["style"].split(";")
        )["font"]

        # 断言提取的字体信息与预期的字体字符串一致
        assert font_info == f"{size}px {font_str}"

    # 断言文本元素的总数与图中实际添加的文本数相等
    assert text_count == len(ax.texts)
def test_annotationbbox_gid():
    # 测试对象 gid 是否出现在输出的 SVG 文件的 AnnotationBbox 中。
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形对象中添加一个子图
    ax = fig.add_subplot()
    # 创建一个全为1的32x32的numpy数组作为图像数据
    arr_img = np.ones((32, 32))
    # 设定图像在子图中的位置
    xy = (0.3, 0.55)

    # 创建一个偏移图像对象，缩放比例为0.1
    imagebox = OffsetImage(arr_img, zoom=0.1)
    # 将图像对象与当前子图关联
    imagebox.image.axes = ax

    # 创建 AnnotationBbox 对象，将图像盒子放置在指定的数据坐标 (xy) 处，
    # 盒子的坐标使用偏移点 (120., -80.)，数据坐标和盒子坐标都使用默认设置
    ab = AnnotationBbox(imagebox, xy,
                        xybox=(120., -80.),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.5,
                        arrowprops=dict(
                            arrowstyle="->",
                            connectionstyle="angle,angleA=0,angleB=90,rad=3")
                        )
    # 设置 AnnotationBbox 的 gid 属性为 "a test for issue 20044"
    ab.set_gid("a test for issue 20044")
    # 将 AnnotationBbox 对象添加到子图中
    ax.add_artist(ab)

    # 创建一个字节流对象
    with BytesIO() as fd:
        # 将图形保存为 SVG 格式到字节流中
        fig.savefig(fd, format='svg')
        # 从字节流中获取数据并转换为 UTF-8 格式的字符串
        buf = fd.getvalue().decode('utf-8')

    # 预期 SVG 输出中应包含的字符串
    expected = '<g id="a test for issue 20044">'
    # 断言预期字符串是否在 SVG 输出中
    assert expected in buf
```