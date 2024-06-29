# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_backend_ps.py`

```
# 导入必要的库
from collections import Counter
from pathlib import Path
import io
import re
import tempfile

import numpy as np
import pytest

# 导入 matplotlib 相关模块和函数
from matplotlib import cbook, path, patheffects, font_manager as fm
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.testing._markers import needs_ghostscript, needs_usetex
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib as mpl
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# 此测试在 AppVeyor 上经常遇到 TeX 缓存锁定问题。
@pytest.mark.flaky(reruns=3)
# 参数化测试，测试不同的纸张大小和方向
@pytest.mark.parametrize('papersize', ['letter', 'figure'])
@pytest.mark.parametrize('orientation', ['portrait', 'landscape'])
# 参数化测试，测试不同的格式、是否使用对数、以及不同的 rcParams 设置
@pytest.mark.parametrize('format, use_log, rcParams', [
    ('ps', False, {}),
    ('ps', False, {'ps.usedistiller': 'ghostscript'}),
    ('ps', False, {'ps.usedistiller': 'xpdf'}),
    ('ps', False, {'text.usetex': True}),
    ('eps', False, {}),
    ('eps', True, {'ps.useafm': True}),
    ('eps', False, {'text.usetex': True}),
], ids=[
    'ps',
    'ps with distiller=ghostscript',
    'ps with distiller=xpdf',
    'ps with usetex',
    'eps',
    'eps afm',
    'eps with usetex'
])
# 定义测试函数，保存图形为 StringIO 对象
def test_savefig_to_stringio(format, use_log, rcParams, orientation, papersize):
    # 更新 matplotlib 的 rcParams 设置
    mpl.rcParams.update(rcParams)
    
    # 如果使用 ghostscript 作为 distiller，则检查是否可用 gs 可执行文件
    if mpl.rcParams["ps.usedistiller"] == "ghostscript":
        try:
            mpl._get_executable_info("gs")
        except mpl.ExecutableNotFoundError as exc:
            pytest.skip(str(exc))
    # 如果使用 xpdf 作为 distiller，则检查是否可用 gs 和 pdftops 可执行文件
    elif mpl.rcParams["ps.usedistiller"] == "xpdf":
        try:
            mpl._get_executable_info("gs")  # 实际上是检查 ps2pdf 是否可用
            mpl._get_executable_info("pdftops")
        except mpl.ExecutableNotFoundError as exc:
            pytest.skip(str(exc))
    
    # 创建一个新的图形和轴对象
    fig, ax = plt.subplots()
    # 使用 io 模块创建一个 String 字节流对象 s_buf 和一个 Bytes 字节流对象 b_buf
    with io.StringIO() as s_buf, io.BytesIO() as b_buf:
        # 如果 use_log 为真，则设置图表的 y 轴为对数坐标系
        if use_log:
            ax.set_yscale('log')

        # 绘制简单的线条图，示例数据为 (1,1) 和 (2,2)
        ax.plot([1, 2], [1, 2])
        
        # 设置图表的标题，包括特定条件下的附加内容
        title = "Déjà vu"
        if not mpl.rcParams["text.usetex"]:
            title += " \N{MINUS SIGN}\N{EURO SIGN}"
        ax.set_title(title)

        # 初始化一个空的允许异常列表
        allowable_exceptions = []
        
        # 根据配置决定是否添加特定异常类到允许异常列表中
        if mpl.rcParams["text.usetex"]:
            allowable_exceptions.append(RuntimeError)
        if mpl.rcParams["ps.useafm"]:
            allowable_exceptions.append(mpl.MatplotlibDeprecationWarning)
        
        # 尝试保存图表到内存中的两个缓冲区，捕获特定的允许异常
        try:
            fig.savefig(s_buf, format=format, orientation=orientation,
                        papertype=papersize)
            fig.savefig(b_buf, format=format, orientation=orientation,
                        papertype=papersize)
        except tuple(allowable_exceptions) as exc:
            pytest.skip(str(exc))

        # 断言字符串和字节流缓冲区仍然处于打开状态
        assert not s_buf.closed
        assert not b_buf.closed
        
        # 获取并编码 s_buf 和 b_buf 的值作为 ASCII 和字节串
        s_val = s_buf.getvalue().encode('ascii')
        b_val = b_buf.getvalue()

        # 根据输出格式为 'ps' 时的特定条件进行断言，检查生成的内容是否符合预期
        if format == 'ps':
            # 默认 figsize = (8, 6) 英寸 = (576, 432) 点 = (203.2, 152.4) 毫米。
            # 横向方向会交换尺寸。
            if mpl.rcParams["ps.usedistiller"] == "xpdf":
                # 某些版本特别显示 letter/203x152，但不是所有版本都是这样的，
                # 所以我们只能使用这种更简单的测试。
                if papersize == 'figure':
                    assert b'letter' not in s_val.lower()
                else:
                    assert b'letter' in s_val.lower()
            elif mpl.rcParams["ps.usedistiller"] or mpl.rcParams["text.usetex"]:
                # 根据方向和纸张类型设置期望的宽度值，并进行相应的断言
                width = b'432.0' if orientation == 'landscape' else b'576.0'
                wanted = (b'-dDEVICEWIDTHPOINTS=' + width if papersize == 'figure'
                          else b'-sPAPERSIZE')
                assert wanted in s_val
            else:
                # 根据纸张类型进行简单的断言，检查是否包含特定的文档尺寸标识符
                if papersize == 'figure':
                    assert b'%%DocumentPaperSizes' not in s_val
                else:
                    assert b'%%DocumentPaperSizes' in s_val

        # 去除 CreationDate: 字段，因为 ghostscript 和 cairo 不遵循 SOURCE_DATE_EPOCH
        s_val = re.sub(b"(?<=\n%%CreationDate: ).*", b"", s_val)
        b_val = re.sub(b"(?<=\n%%CreationDate: ).*", b"", b_val)

        # 断言处理后的 s_val 和 b_val 在换行符上的替换结果是相等的
        assert s_val == b_val.replace(b'\r\n', b'\n')
def test_patheffects():
    # 设置路径效果，用于在图形中添加描边效果
    mpl.rcParams['path.effects'] = [
        patheffects.withStroke(linewidth=4, foreground='w')]
    # 创建图形和轴
    fig, ax = plt.subplots()
    # 绘制简单的线图
    ax.plot([1, 2, 3])
    # 使用字节流保存图形为 PS 格式
    with io.BytesIO() as ps:
        fig.savefig(ps, format='ps')


@needs_usetex
@needs_ghostscript
def test_tilde_in_tempfilename(tmp_path):
    # 在临时目录路径中使用 ~ 符号（例如在 Windows 系统中的 TMP 或 TEMP 目录下，
    # 当用户名很长时，Windows 会使用短名称），这会在早期版本的 Matplotlib 中破坏 LaTeX 的正常运行
    base_tempdir = tmp_path / "short-1"
    base_tempdir.mkdir()
    # 更改用于新临时目录的路径，内部用于 PS 后端写入文件
    with cbook._setattr_cm(tempfile, tempdir=str(base_tempdir)):
        # 使用 LaTeX 渲染文本
        mpl.rcParams['text.usetex'] = True
        # 绘制简单的线图
        plt.plot([1, 2, 3, 4])
        # 添加 X 轴标签，使用 LaTeX 格式
        plt.xlabel(r'\textbf{time} (s)')
        # 使用 PS 后端保存图形文件
        plt.savefig(base_tempdir / 'tex_demo.eps', format="ps")


@image_comparison(["empty.eps"])
def test_transparency():
    # 创建图形和轴
    fig, ax = plt.subplots()
    # 关闭轴线
    ax.set_axis_off()
    # 绘制透明度为 0 的红色线条
    ax.plot([0, 1], color="r", alpha=0)
    # 在图中添加文本 "foo"，颜色为红色，透明度为 0
    ax.text(.5, .5, "foo", color="r", alpha=0)


@needs_usetex
@image_comparison(["empty.eps"])
def test_transparency_tex():
    # 设置使用 LaTeX 渲染文本
    mpl.rcParams['text.usetex'] = True
    # 创建图形和轴
    fig, ax = plt.subplots()
    # 关闭轴线
    ax.set_axis_off()
    # 绘制透明度为 0 的红色线条
    ax.plot([0, 1], color="r", alpha=0)
    # 在图中添加文本 "foo"，颜色为红色，透明度为 0
    ax.text(.5, .5, "foo", color="r", alpha=0)


def test_bbox():
    # 创建图形和轴
    fig, ax = plt.subplots()
    # 使用字节流保存图形为 EPS 格式
    with io.BytesIO() as buf:
        fig.savefig(buf, format='eps')
        buf = buf.getvalue()

    # 从 EPS 文件的内容中查找 BoundingBox 和 HiResBoundingBox
    bb = re.search(b'^%%BoundingBox: (.+) (.+) (.+) (.+)$', buf, re.MULTILINE)
    assert bb
    hibb = re.search(b'^%%HiResBoundingBox: (.+) (.+) (.+) (.+)$', buf,
                     re.MULTILINE)
    assert hibb

    for i in range(1, 5):
        # BoundingBox 必须使用整数，并且应该是 HiResBoundingBox 的上下限
        assert b'.' not in bb.group(i)
        assert int(bb.group(i)) == pytest.approx(float(hibb.group(i)), 1)


@needs_usetex
def test_failing_latex():
    """测试 LaTeX 子进程调用失败情况"""
    # 设置使用 LaTeX 渲染文本
    mpl.rcParams['text.usetex'] = True
    # 尝试绘制带有双下标的 LaTeX 文本，这会导致失败
    plt.xlabel("$22_2_2$")
    # 断言捕获到 RuntimeError 异常
    with pytest.raises(RuntimeError):
        plt.savefig(io.BytesIO(), format="ps")


@needs_usetex
def test_partial_usetex(caplog):
    # 设置日志记录级别为警告
    caplog.set_level("WARNING")
    # 在图形中添加文本 "foo" 和 "bar"，使用 LaTeX 渲染
    plt.figtext(.1, .1, "foo", usetex=True)
    plt.figtext(.2, .2, "bar", usetex=True)
    # 使用 PS 后端保存图形文件
    plt.savefig(io.BytesIO(), format="ps")
    # 断言日志中只有一条记录
    record, = caplog.records  # asserts there's a single record.
    assert "as if usetex=False" in record.getMessage()


@needs_usetex
def test_usetex_preamble(caplog):
    # 更新 Matplotlib 参数，设置使用 LaTeX 渲染文本，并指定一些 LaTeX preamble 包
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{color,graphicx,textcomp}",
    })
    # 在当前图形中添加文本 "foo"，位置位于图形中心
    plt.figtext(.5, .5, "foo")
    # 将当前图形保存为 PostScript 格式，保存到一个字节流中
    plt.savefig(io.BytesIO(), format="ps")
# 使用 @image_comparison 装饰器，比较测试当前函数生成的图像和已有的 EPS 图像是否相同
@image_comparison(["useafm.eps"])
def test_useafm():
    # 设置 Matplotlib 参数，启用字体管理
    mpl.rcParams["ps.useafm"] = True
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 关闭坐标轴显示
    ax.set_axis_off()
    # 在坐标轴上添加水平线
    ax.axhline(.5)
    # 在指定位置添加文本
    ax.text(.5, .5, "qk")


# 使用 @image_comparison 装饰器，比较测试当前函数生成的图像和已有的 EPS 图像是否相同
@image_comparison(["type3.eps"])
def test_type3_font():
    # 在图形中心添加文本
    plt.figtext(.5, .5, "I/J")


# 使用 @image_comparison 装饰器，比较测试当前函数生成的图像和已有的 EPS 图像是否相同
@image_comparison(["coloredhatcheszerolw.eps"])
def test_colored_hatch_zero_linewidth():
    # 获取当前坐标轴对象
    ax = plt.gca()
    # 在坐标轴上添加椭圆补丁，设置填充为无，边缘为红色，线宽为0
    ax.add_patch(Ellipse((0, 0), 1, 1, hatch='/', facecolor='none',
                         edgecolor='r', linewidth=0))
    # 在坐标轴上添加椭圆补丁，设置填充为无，边缘为绿色，线宽为0.2
    ax.add_patch(Ellipse((0.5, 0.5), 0.5, 0.5, hatch='+', facecolor='none',
                         edgecolor='g', linewidth=0.2))
    # 在坐标轴上添加椭圆补丁，设置填充为无，边缘为蓝色，线宽为0
    ax.add_patch(Ellipse((1, 1), 0.3, 0.8, hatch='\\', facecolor='none',
                         edgecolor='b', linewidth=0))
    # 关闭坐标轴显示
    ax.set_axis_off()


# 使用 @check_figures_equal 装饰器，确保测试当前函数生成的图像和参考图像相同
@check_figures_equal(extensions=["eps"])
def test_text_clip(fig_test, fig_ref):
    # 添加子图到测试图形对象
    ax = fig_test.add_subplot()
    # 在图形的指定位置添加文本，设置剪切开启
    ax.text(0, 0, "hello", transform=fig_test.transFigure, clip_on=True)
    # 添加子图到参考图形对象
    fig_ref.add_subplot()


# 使用 @needs_ghostscript 装饰器，确保测试当前函数需要 Ghostscript 环境
@needs_ghostscript
def test_d_glyph(tmp_path):
    # 创建图形对象
    fig = plt.figure()
    # 在图形中心添加文本
    fig.text(.5, .5, "def")
    # 保存图形为 EPS 格式
    out = tmp_path / "test.eps"
    fig.savefig(out)
    # 比较转换输出，应该不会引发异常
    mpl.testing.compare.convert(out, cache=False)


# 使用 @image_comparison 装饰器，比较测试当前函数生成的图像和已有的 EPS 图像是否相同，使用 mpl20 样式
@image_comparison(["type42_without_prep.eps"], style='mpl20')
def test_type42_font_without_prep():
    # 设置 Matplotlib 参数，字体类型为 Type 42，数学文本字体集为 STIX
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["mathtext.fontset"] = "stix"
    # 在图形中心添加文本，包含数学公式
    plt.figtext(0.5, 0.5, "Mass $m$")


# 使用 @pytest.mark.parametrize 标记，测试当前函数的多个参数组合
@pytest.mark.parametrize('fonttype', ["3", "42"])
def test_fonttype(fonttype):
    # 设置 Matplotlib 参数，字体类型根据参数值设置
    mpl.rcParams["ps.fonttype"] = fonttype
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴指定位置添加文本
    ax.text(0.25, 0.5, "Forty-two is the answer to everything!")
    # 创建字节流缓冲区
    buf = io.BytesIO()
    # 将图形保存为 EPS 格式到字节流中
    fig.savefig(buf, format="ps")
    # 准备搜索字节流中的字体类型定义
    test = b'/FontType ' + bytes(f"{fonttype}", encoding='utf-8') + b' def'
    # 使用正则表达式搜索字节流中的文本
    assert re.search(test, buf.getvalue(), re.MULTILINE)


# 测试当前函数，验证虚线在 PS 输出中不会中断
def test_linedash():
    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    # 在坐标轴上绘制虚线
    ax.plot([0, 1], linestyle="--")
    # 创建字节流缓冲区
    buf = io.BytesIO()
    # 将图形保存为 PS 格式到字节流中
    fig.savefig(buf, format="ps")
    # 断言字节流大小大于0
    assert buf.tell() > 0


# 测试当前函数，检验空行问题（gh#23954）
def test_empty_line():
    # 创建 Figure 对象
    figure = Figure()
    # 在图形中心添加文本，包含多个空行
    figure.text(0.5, 0.5, "\nfoo\n\n")
    # 创建字节流缓冲区
    buf = io.BytesIO()
    # 将图形保存为 EPS 格式到字节流中
    figure.savefig(buf, format='eps')
    # 再次将图形保存为 PS 格式到字节流中
    figure.savefig(buf, format='ps')


# 测试当前函数，验证不会重复定义问题
def test_no_duplicate_definition():
    # 创建 Figure 对象
    fig = Figure()
    # 创建具有极坐标投影的 4x4 子图数组
    axs = fig.subplots(4, 4, subplot_kw=dict(projection="polar"))
    # 遍历每个子图对象
    for ax in axs.flat:
        # 设置子图的坐标轴不显示刻度
        ax.set(xticks=[], yticks=[])
        # 在子图中绘制简单线条
        ax.plot([1, 2])
    # 设置整体图形的标题
    fig.suptitle("hello, world")
    # 创建字符串 IO 缓冲区
    buf = io.StringIO()
    # 将图形保存为 EPS 格式到字符串 IO 缓冲区中
    fig.savefig(buf, format='eps')
    # 重置字符串 IO 缓冲区的位置为开头
    buf.seek(0)
    # 读取每行以 '/' 开头的行，并返回第一个单词（以空格分隔）
    wds = [ln.partition(' ')[0] for
           ln in buf.readlines()
           if ln.startswith('/')]
    # 使用 Counter 对象统计 wds 列表中每个元素的出现次数，然后取最大值
    # 断言：确保最大出现次数为 1，即所有元素都只出现一次，否则抛出异常
    assert max(Counter(wds).values()) == 1
@image_comparison(["multi_font_type3.eps"], tol=0.51)
def test_multi_font_type3():
    # 设置字体属性为文泉驿正黑
    fp = fm.FontProperties(family=["WenQuanYi Zen Hei"])
    # 检查是否能找到指定的字体文件，否则跳过测试
    if Path(fm.findfont(fp)).name != "wqy-zenhei.ttc":
        pytest.skip("Font may be missing")

    # 设置字体为DejaVu Sans和文泉驿正黑，大小为27
    plt.rc('font', family=['DejaVu Sans', 'WenQuanYi Zen Hei'], size=27)
    # 设置ps渲染器的字体类型为Type 3
    plt.rc('ps', fonttype=3)

    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上添加文本，包括一些汉字
    fig.text(0.15, 0.475, "There are 几个汉字 in between!")


@image_comparison(["multi_font_type42.eps"], tol=1.6)
def test_multi_font_type42():
    # 设置字体属性为文泉驿正黑
    fp = fm.FontProperties(family=["WenQuanYi Zen Hei"])
    # 检查是否能找到指定的字体文件，否则跳过测试
    if Path(fm.findfont(fp)).name != "wqy-zenhei.ttc":
        pytest.skip("Font may be missing")

    # 设置字体为DejaVu Sans和文泉驿正黑，大小为27
    plt.rc('font', family=['DejaVu Sans', 'WenQuanYi Zen Hei'], size=27)
    # 设置ps渲染器的字体类型为Type 42
    plt.rc('ps', fonttype=42)

    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上添加文本，包括一些汉字
    fig.text(0.15, 0.475, "There are 几个汉字 in between!")


@image_comparison(["scatter.eps"])
def test_path_collection():
    # 创建一个随机数生成器
    rng = np.random.default_rng(19680801)
    # 生成随机的x坐标和y坐标
    xvals = rng.uniform(0, 1, 10)
    yvals = rng.uniform(0, 1, 10)
    # 生成随机的点大小
    sizes = rng.uniform(30, 100, 10)
    # 创建一个包含子图的新图形对象
    fig, ax = plt.subplots()
    # 在子图上绘制散点图
    ax.scatter(xvals, yvals, sizes, edgecolor=[0.9, 0.2, 0.1], marker='<')
    # 设置子图不显示坐标轴
    ax.set_axis_off()
    # 创建一组基本几何路径
    paths = [path.Path.unit_regular_polygon(i) for i in range(3, 7)]
    # 生成随机的偏移量
    offsets = rng.uniform(0, 200, 20).reshape(10, 2)
    # 指定几何路径的大小
    sizes = [0.02, 0.04]
    # 创建路径集合对象，添加到子图
    pc = mcollections.PathCollection(paths, sizes, zorder=-1,
                                     facecolors='yellow', offsets=offsets)
    ax.add_collection(pc)
    # 设置子图的x轴范围
    ax.set_xlim(0, 1)


@image_comparison(["colorbar_shift.eps"], savefig_kwarg={"bbox_inches": "tight"},
                  style="mpl20")
def test_colorbar_shift(tmp_path):
    # 创建一个颜色映射对象，指定颜色列表和边界规范
    cmap = mcolors.ListedColormap(["r", "g", "b"])
    norm = mcolors.BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)
    # 绘制散点图并加上颜色条
    plt.scatter([0, 1], [1, 1], c=[0, 1], cmap=cmap, norm=norm)
    plt.colorbar()


def test_auto_papersize_deprecation():
    # 创建一个新的图形对象
    fig = plt.figure()
    # 测试在保存时发出Matplotlib过时警告，指定图形格式为eps和自动纸张类型
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        fig.savefig(io.BytesIO(), format='eps', papertype='auto')

    # 测试在设置PS输出时发出Matplotlib过时警告，设置ps.papersize为自动
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        mpl.rcParams['ps.papersize'] = 'auto'
```