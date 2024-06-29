# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_backend_pgf.py`

```
# 导入 datetime 模块，用于处理日期和时间
import datetime
# 导入 BytesIO 类，用于处理二进制数据的内存缓冲
from io import BytesIO
# 导入 os 模块，提供了与操作系统交互的功能
import os
# 导入 shutil 模块，用于文件和目录的高级操作
import shutil

# 导入第三方库 numpy，并重命名为 np
import numpy as np
# 从 packaging.version 模块中导入 parse 函数，并重命名为 parse_version
from packaging.version import parse as parse_version
# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 matplotlib 库，并重命名为 mpl
import matplotlib as mpl
# 从 matplotlib 中导入 pyplot 模块，并重命名为 plt
import matplotlib.pyplot as plt
# 从 matplotlib.testing 模块中导入一些函数和类
from matplotlib.testing import _has_tex_package, _check_for_pgf
# 从 matplotlib.testing.exceptions 模块中导入 ImageComparisonFailure 异常类
from matplotlib.testing.exceptions import ImageComparisonFailure
# 从 matplotlib.testing.compare 模块中导入 compare_images 函数
from matplotlib.testing.compare import compare_images
# 从 matplotlib.backends.backend_pgf 模块中导入 PdfPages 类
from matplotlib.backends.backend_pgf import PdfPages
# 从 matplotlib.testing.decorators 模块中导入一些装饰器和函数
from matplotlib.testing.decorators import (
    _image_directories, check_figures_equal, image_comparison)
# 从 matplotlib.testing._markers 模块中导入一些标记函数
from matplotlib.testing._markers import (
    needs_ghostscript, needs_pgf_lualatex, needs_pgf_pdflatex,
    needs_pgf_xelatex)

# 使用 _image_directories 函数获取 baseline_dir 和 result_dir
baseline_dir, result_dir = _image_directories(lambda: 'dummy func')


# 定义函数 compare_figure，用于比较生成的图像和预期图像
def compare_figure(fname, savefig_kwargs={}, tol=0):
    # 设置保存实际图像的路径和文件名
    actual = os.path.join(result_dir, fname)
    # 保存当前图形到 actual 文件
    plt.savefig(actual, **savefig_kwargs)

    # 设置保存预期图像的路径和文件名
    expected = os.path.join(result_dir, "expected_%s" % fname)
    # 复制 baseline_dir 中对应的文件到 expected 文件
    shutil.copyfile(os.path.join(baseline_dir, fname), expected)
    # 比较实际图像和预期图像，如果有误差则抛出 ImageComparisonFailure 异常
    err = compare_images(expected, actual, tol=tol)
    if err:
        raise ImageComparisonFailure(err)


# 使用装饰器标记需要 xelatex 和 Ghostscript 的测试函数，且标记测试后端为 pgf
@needs_pgf_xelatex
@needs_ghostscript
@pytest.mark.backend('pgf')
def test_tex_special_chars(tmp_path):
    # 创建一个新的图形对象
    fig = plt.figure()
    # 在图形上添加文本，包含特殊字符
    fig.text(.5, .5, "%_^ $a_b^c$")
    # 创建一个字节流缓冲区
    buf = BytesIO()
    # 将图形保存为 PNG 格式到 buf 缓冲区，使用 pgf 后端
    fig.savefig(buf, format="png", backend="pgf")
    buf.seek(0)
    # 从 buf 中读取图像数据
    t = plt.imread(buf)
    # 断言图像数据中不全是白色（即不全是 1）
    assert not (t == 1).all()  # The leading "%" didn't eat up everything.


# 定义函数 create_figure，用于创建一个包含多种元素的图形
def create_figure():
    # 创建一个新的图形
    plt.figure()
    # 生成 x 数据
    x = np.linspace(0, 1, 15)

    # 绘制线图
    plt.plot(x, x ** 2, "b-")

    # 绘制带有标记的线图
    plt.plot(x, 1 - x**2, "g>")

    # 填充区域并设置填充样式
    plt.fill_between([0., .4], [.4, 0.], hatch='//', facecolor="lightgray",
                     edgecolor="red")
    plt.fill([3, 3, .8, .8, 3], [2, -2, -2, 0, 2], "b")

    # 添加文本和排版设置
    plt.plot([0.9], [0.5], "ro", markersize=3)
    plt.text(0.9, 0.5, 'unicode (ü, °, \N{Section Sign}) and math ($\\mu_i = x_i^2$)',
             ha='right', fontsize=20)
    plt.ylabel('sans-serif, blue, $\\frac{\\sqrt{x}}{y^2}$..',
               family='sans-serif', color='blue')
    plt.text(1, 1, 'should be clipped as default clip_box is Axes bbox',
             fontsize=20, clip_on=True)

    # 设置 x 和 y 轴的范围
    plt.xlim(0, 1)
    plt.ylim(0, 1)


# 使用 xelatex 后端生成 PDF 的图形比较测试函数
@needs_pgf_xelatex
@pytest.mark.backend('pgf')
@image_comparison(['pgf_xelatex.pdf'], style='default')
def test_xelatex():
    # 更新 rcParams 以使用 xelatex 后端生成图形
    rc_xelatex = {'font.family': 'serif',
                  'pgf.rcfonts': False}
    mpl.rcParams.update(rc_xelatex)
    # 创建图形并进行比较
    create_figure()


# 尝试获取 Ghostscript 的版本信息，如果版本小于 9.50 则将 _old_gs_version 设置为 True
try:
    _old_gs_version = \
        mpl._get_executable_info('gs').version < parse_version('9.50')
except mpl.ExecutableNotFoundError:
    # 如果找不到 Ghostscript 则将 _old_gs_version 设置为 True
    _old_gs_version = True


# 使用 pdflatex 后端生成 PDF 的图形比较测试函数
@needs_pgf_pdflatex
@pytest.mark.skipif(not _has_tex_package('type1ec'), reason='needs type1ec.sty')
# 标记此测试用例在缺少 'ucs' 包时跳过执行
@pytest.mark.skipif(not _has_tex_package('ucs'), reason='needs ucs.sty')
# 标记此测试用例使用 'pgf' 后端
@pytest.mark.backend('pgf')
# 对图像进行比较，预期结果是 'pgf_pdflatex.pdf'，使用 'default' 风格，允许的误差为 11.71（如果是旧的 GhostScript 版本则为 0）
@image_comparison(['pgf_pdflatex.pdf'], style='default',
                  tol=11.71 if _old_gs_version else 0)
# 定义测试函数 test_pdflatex
def test_pdflatex():
    # 配置参数字典 rc_pdflatex，用于设置使用 pdflatex 的绘图配置
    rc_pdflatex = {'font.family': 'serif',
                   'pgf.rcfonts': False,
                   'pgf.texsystem': 'pdflatex',
                   'pgf.preamble': ('\\usepackage[utf8x]{inputenc}'  # 使用 UTF-8 编码
                                    '\\usepackage[T1]{fontenc}')  # 使用 T1 字体编码
                   }
    # 更新全局的 matplotlib 配置参数
    mpl.rcParams.update(rc_pdflatex)
    # 创建图形
    create_figure()


# 定义测试函数 test_rcupdate，用于测试每个图形更新的 rc 参数
@needs_pgf_xelatex
@needs_pgf_pdflatex
@mpl.style.context('default')
# 标记使用 'pgf' 后端
@pytest.mark.backend('pgf')
def test_rcupdate():
    # 不同的 rc 参数设置列表
    rc_sets = [{'font.family': 'sans-serif',
                'font.size': 30,
                'figure.subplot.left': .2,
                'lines.markersize': 10,
                'pgf.rcfonts': False,
                'pgf.texsystem': 'xelatex'},  # 使用 xelatex
               {'font.family': 'monospace',
                'font.size': 10,
                'figure.subplot.left': .1,
                'lines.markersize': 20,
                'pgf.rcfonts': False,
                'pgf.texsystem': 'pdflatex',
                'pgf.preamble': ('\\usepackage[utf8x]{inputenc}'  # 使用 UTF-8 编码
                                 '\\usepackage[T1]{fontenc}'  # 使用 T1 字体编码
                                 '\\usepackage{sfmath}')}]  # 使用 sfmath 包
    # 如果是旧版本的 GhostScript，tol 设为 [0, 13.2]，否则设为 [0, 0]
    tol = [0, 13.2] if _old_gs_version else [0, 0]
    # 遍历 rc_sets 中的每个设置
    for i, rc_set in enumerate(rc_sets):
        # 使用当前 rc 参数上下文
        with mpl.rc_context(rc_set):
            # 检查是否需要跳过测试，如果需要缺少相应的 LaTeX 包
            for substring, pkg in [('sfmath', 'sfmath'), ('utf8x', 'ucs')]:
                if (substring in mpl.rcParams['pgf.preamble']
                        and not _has_tex_package(pkg)):
                    pytest.skip(f'needs {pkg}.sty')
            # 创建图形
            create_figure()
            # 比较生成的图形与预期的 'pgf_rcupdate{i + 1}.pdf' 文件
            compare_figure(f'pgf_rcupdate{i + 1}.pdf', tol=tol[i])


# 定义测试函数 test_pathclip，用于测试后端路径裁剪功能
@needs_pgf_xelatex
@mpl.style.context('default')
# 标记使用 'pgf' 后端
@pytest.mark.backend('pgf')
def test_pathclip():
    # 设定随机种子
    np.random.seed(19680801)
    # 更新 matplotlib 配置参数，设置 'serif' 字体和关闭 pgf.rcfonts
    mpl.rcParams.update({'font.family': 'serif', 'pgf.rcfonts': False})
    # 创建包含两个子图的图形对象
    fig, axs = plt.subplots(1, 2)

    # 在第一个子图上绘制一条从 (0, 0) 到 (1e100, 1e100) 的直线
    axs[0].plot([0., 1e100], [0., 1e100])
    # 设置第一个子图的 x 轴和 y 轴的范围
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)

    # 在第二个子图上绘制散点图和直方图
    axs[1].scatter([0, 1], [1, 1])
    axs[1].hist(np.random.normal(size=1000), bins=20, range=[-10, 10])
    # 设置第二个子图的 x 轴为对数刻度
    axs[1].set_xscale('log')

    # 将生成的图形保存为 PDF 格式，不进行图像比较
    fig.savefig(BytesIO(), format="pdf")


# 定义测试函数 test_mixedmode，用于测试混合模式渲染
@needs_pgf_xelatex
# 标记使用 'pgf' 后端
@pytest.mark.backend('pgf')
# 对比生成的图像 'pgf_mixedmode.pdf'，使用 'default' 风格
@image_comparison(['pgf_mixedmode.pdf'], style='default')
def test_mixedmode():
    # 更新 matplotlib 配置参数，设置 'serif' 字体和关闭 pgf.rcfonts
    mpl.rcParams.update({'font.family': 'serif', 'pgf.rcfonts': False})
    # 创建网格 Y, X，范围是 [-1, 1]，40个点
    Y, X = np.ogrid[-1:1:40j, -1:1:40j]
    # 使用 pcolor 绘制 X^2 + Y^2 的颜色图，并设置光栅化处理为 True
    plt.pcolor(X**2 + Y**2).set_rasterized(True)


# 定义测试函数 test_bbox_inches，用于测试 bbox_inches 裁剪
@needs_pgf_xelatex
@mpl.style.context('default')
# 标记使用 'pgf' 后端
@pytest.mark.backend('pgf')
def test_bbox_inches():
    # 更新 matplotlib 配置参数，设置 'serif' 字体和关闭 pgf.rcfonts
    mpl.rcParams.update({'font.family': 'serif', 'pgf.rcfonts': False})
    # 创建包含两个子图的图形对象，并返回这两个子图的引用给变量fig和(ax1, ax2)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # 在第一个子图ax1上绘制一条线，横坐标为[0, 1, 2, 3, 4]
    ax1.plot(range(5))
    
    # 在第二个子图ax2上绘制一条线，横坐标为[0, 1, 2, 3, 4]
    ax2.plot(range(5))
    
    # 调整子图的布局，使它们之间没有重叠部分
    plt.tight_layout()
    
    # 获取第一个子图ax1的绘图区域（bbox），并将其按照图形的dpi缩放因子转换为指定的单位
    bbox = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    
    # 比较当前图形与另一个图形的结果，保存为PDF格式的文件，并指定保存参数为bbox_inches的值为上一步计算得到的bbox
    compare_figure('pgf_bbox_inches.pdf', savefig_kwargs={'bbox_inches': bbox}, tol=0)
# 使用默认样式上下文装饰测试函数，使得测试使用默认样式环境运行
@mpl.style.context('default')
# 使用 pytest.mark.backend 装饰器标记测试使用的后端为 'pgf'
@pytest.mark.backend('pgf')
# 使用 pytest.mark.parametrize 参数化测试，system 参数为不同的 LaTeX 引擎
@pytest.mark.parametrize('system', [
    # 标记为需要 'lualatex' 引擎的测试用例
    pytest.param('lualatex', marks=[needs_pgf_lualatex]),
    # 标记为需要 'pdflatex' 引擎的测试用例
    pytest.param('pdflatex', marks=[needs_pgf_pdflatex]),
    # 标记为需要 'xelatex' 引擎的测试用例
    pytest.param('xelatex', marks=[needs_pgf_xelatex]),
])
# 定义测试函数 test_pdf_pages，接受 system 参数
def test_pdf_pages(system):
    # 配置 pdflatex 相关的 matplotlib 参数字典
    rc_pdflatex = {
        'font.family': 'serif',
        'pgf.rcfonts': False,
        'pgf.texsystem': system,
    }
    # 更新 matplotlib 的全局参数
    mpl.rcParams.update(rc_pdflatex)

    # 创建第一个子图和轴对象
    fig1, ax1 = plt.subplots()
    # 在第一个轴上绘制简单的折线图
    ax1.plot(range(5))
    # 调整第一个图的布局
    fig1.tight_layout()

    # 创建第二个子图和轴对象，设置特定的图形尺寸
    fig2, ax2 = plt.subplots(figsize=(3, 2))
    # 在第二个轴上绘制简单的折线图
    ax2.plot(range(5))
    # 调整第二个图的布局
    fig2.tight_layout()

    # 设置输出路径为 result_dir 目录下的以 system 名称命名的 PDF 文件
    path = os.path.join(result_dir, f'pdfpages_{system}.pdf')

    # 定义 PDF 的元数据字典
    md = {
        'Author': 'me',
        'Title': 'Multipage PDF with pgf',
        'Subject': 'Test page',
        'Keywords': 'test,pdf,multipage',
        'ModDate': datetime.datetime(
            1968, 8, 1, tzinfo=datetime.timezone(datetime.timedelta(0))),
        'Trapped': 'Unknown'
    }

    # 使用 PdfPages 上下文管理器打开 PDF 文件，指定元数据
    with PdfPages(path, metadata=md) as pdf:
        # 将第一个图保存到 PDF
        pdf.savefig(fig1)
        # 将第二个图保存到 PDF
        pdf.savefig(fig2)
        # 再次将第一个图保存到 PDF
        pdf.savefig(fig1)

        # 断言 PDF 页面数为 3 页
        assert pdf.get_pagecount() == 3


# 使用默认样式上下文装饰测试函数，测试 PDF 页面的元数据检查
@mpl.style.context('default')
# 使用 pytest.mark.backend 装饰器标记测试使用的后端为 'pgf'
@pytest.mark.backend('pgf')
# 使用 pytest.mark.parametrize 参数化测试，system 参数为不同的 LaTeX 引擎
@pytest.mark.parametrize('system', [
    # 标记为需要 'lualatex' 引擎的测试用例
    pytest.param('lualatex', marks=[needs_pgf_lualatex]),
    # 标记为需要 'pdflatex' 引擎的测试用例
    pytest.param('pdflatex', marks=[needs_pgf_pdflatex]),
    # 标记为需要 'xelatex' 引擎的测试用例
    pytest.param('xelatex', marks=[needs_pgf_xelatex]),
])
# 定义测试函数 test_pdf_pages_metadata_check，接受 monkeypatch 和 system 参数
def test_pdf_pages_metadata_check(monkeypatch, system):
    # 导入 pikepdf，如果导入失败则跳过测试
    pikepdf = pytest.importorskip('pikepdf')
    # 设置环境变量 SOURCE_DATE_EPOCH 为 '0'
    monkeypatch.setenv('SOURCE_DATE_EPOCH', '0')

    # 更新 matplotlib 的全局参数，指定 pgf.texsystem 为 system
    mpl.rcParams.update({'pgf.texsystem': system})

    # 创建子图和轴对象
    fig, ax = plt.subplots()
    # 在轴上绘制简单的折线图
    ax.plot(range(5))

    # 定义 PDF 的元数据字典
    md = {
        'Author': 'me',
        'Title': 'Multipage PDF with pgf',
        'Subject': 'Test page',
        'Keywords': 'test,pdf,multipage',
        'ModDate': datetime.datetime(
            1968, 8, 1, tzinfo=datetime.timezone(datetime.timedelta(0))),
        'Trapped': 'True'
    }
    # 设置输出路径为 result_dir 目录下的以 system 名称命名的 PDF 文件
    path = os.path.join(result_dir, f'pdfpages_meta_check_{system}.pdf')

    # 使用 PdfPages 上下文管理器打开 PDF 文件，指定元数据
    with PdfPages(path, metadata=md) as pdf:
        # 将图保存到 PDF
        pdf.savefig(fig)

    # 使用 pikepdf 打开 PDF 文件，获取文档信息
    with pikepdf.Pdf.open(path) as pdf:
        info = {k: str(v) for k, v in pdf.docinfo.items()}

    # 如果 '/PTEX.FullBanner' 存在于信息中，则删除该条目
    if '/PTEX.FullBanner' in info:
        del info['/PTEX.FullBanner']
    # 如果 '/PTEX.Fullbanner' 存在于信息中，则删除该条目
    if '/PTEX.Fullbanner' in info:
        del info['/PTEX.Fullbanner']

    # 获取生产者信息，并断言其为特定值，或者在 system 为 'lualatex' 且生产者信息包含 'LuaTeX' 时断言成功
    producer = info.pop('/Producer')
    assert producer == f'Matplotlib pgf backend v{mpl.__version__}' or (
            system == 'lualatex' and 'LuaTeX' in producer)
    # 使用断言检查变量 info 的内容是否等于以下字典，如果不等则抛出异常
    assert info == {
        '/Author': 'me',  # 文档作者信息
        '/CreationDate': 'D:19700101000000Z',  # 文档创建日期
        '/Creator': f'Matplotlib v{mpl.__version__}, https://matplotlib.org',  # 创建文档的软件及其版本信息
        '/Keywords': 'test,pdf,multipage',  # 文档的关键词
        '/ModDate': 'D:19680801000000Z',  # 文档修改日期
        '/Subject': 'Test page',  # 文档主题
        '/Title': 'Multipage PDF with pgf',  # 文档标题
        '/Trapped': '/True',  # 文档的陷阱状态
    }
@needs_pgf_xelatex
# 使用装饰器标记，表明这个测试函数需要 PGF 或 XeLaTeX 支持
def test_multipage_keep_empty(tmp_path):
    # 测试空的 PDF 文件

    # 当 keep_empty 未设置时，会留下一个空的 PDF 文件
    fn = tmp_path / "a.pdf"
    with pytest.warns(mpl.MatplotlibDeprecationWarning), PdfPages(fn) as pdf:
        pass
    assert fn.exists()

    # 当 keep_empty=True 时，会留下一个空的 PDF 文件
    fn = tmp_path / "b.pdf"
    with pytest.warns(mpl.MatplotlibDeprecationWarning), \
            PdfPages(fn, keep_empty=True) as pdf:
        pass
    assert fn.exists()

    # 当 keep_empty=False 时，空的 PDF 文件会在之后被删除
    fn = tmp_path / "c.pdf"
    with PdfPages(fn, keep_empty=False) as pdf:
        pass
    assert not fn.exists()

    # 测试包含内容的 PDF 文件，它们不应被删除

    # 当 keep_empty 未设置时，会留下一个非空的 PDF 文件
    fn = tmp_path / "d.pdf"
    with PdfPages(fn) as pdf:
        pdf.savefig(plt.figure())
    assert fn.exists()

    # 当 keep_empty=True 时，会留下一个非空的 PDF 文件
    fn = tmp_path / "e.pdf"
    with pytest.warns(mpl.MatplotlibDeprecationWarning), \
            PdfPages(fn, keep_empty=True) as pdf:
        pdf.savefig(plt.figure())
    assert fn.exists()

    # 当 keep_empty=False 时，会留下一个非空的 PDF 文件
    fn = tmp_path / "f.pdf"
    with PdfPages(fn, keep_empty=False) as pdf:
        pdf.savefig(plt.figure())
    assert fn.exists()


@needs_pgf_xelatex
# 使用装饰器标记，表明这个测试函数需要 PGF 或 XeLaTeX 支持
def test_tex_restart_after_error():
    # 创建一个图形对象
    fig = plt.figure()
    # 设置图形的总标题，包含一个错误的 LaTeX 命令
    fig.suptitle(r"\oops")
    # 断言捕获到 ValueError 异常
    with pytest.raises(ValueError):
        # 将图形保存为 PGF 格式到字节流
        fig.savefig(BytesIO(), format="pgf")

    # 重新创建一个图形对象，从头开始
    fig = plt.figure()
    # 设置图形的总标题，这次不含错误的 LaTeX 命令
    fig.suptitle(r"this is ok")
    # 将图形保存为 PGF 格式到字节流
    fig.savefig(BytesIO(), format="pgf")


@needs_pgf_xelatex
# 使用装饰器标记，表明这个测试函数需要 PGF 或 XeLaTeX 支持
def test_bbox_inches_tight():
    # 创建一个图形和轴对象
    fig, ax = plt.subplots()
    # 在轴上显示一个简单的图像
    ax.imshow([[0, 1], [2, 3]])
    # 将图形保存为 PDF 格式到字节流，使用 PGF 后端，紧凑边界框
    fig.savefig(BytesIO(), format="pdf", backend="pgf", bbox_inches="tight")


@needs_pgf_xelatex
@needs_ghostscript
# 使用装饰器标记，表明这个测试函数需要 PGF 和 Ghostscript 支持
def test_png_transparency():  # 实际上，也是测试 PNG 是否正常工作。
    # 创建一个字节流对象
    buf = BytesIO()
    # 创建一个图形对象，并将其保存为 PNG 格式到字节流，使用 PGF 后端，透明背景
    plt.figure().savefig(buf, format="png", backend="pgf", transparent=True)
    # 将字节流指针移到开头
    buf.seek(0)
    # 从字节流中读取 PNG 图像数据
    t = plt.imread(buf)
    # 断言图像的所有像素的 alpha 通道都为 0，即完全透明
    assert (t[..., 3] == 0).all()  # 完全透明。


@needs_pgf_xelatex
# 使用装饰器标记，表明这个测试函数需要 PGF 或 XeLaTeX 支持
def test_unknown_font(caplog):
    # 设置日志捕获级别为 WARNING
    with caplog.at_level("WARNING"):
        # 设置 Matplotlib 的字体家族为一个不存在的值
        mpl.rcParams["font.family"] = "this-font-does-not-exist"
        # 在图上添加一个文本
        plt.figtext(.5, .5, "hello, world")
        # 将图形保存为 PGF 格式到字节流
        plt.savefig(BytesIO(), format="pgf")
    # 断言日志中包含特定的警告信息
    assert "Ignoring unknown font: this-font-does-not-exist" in [
        r.getMessage() for r in caplog.records]


@check_figures_equal(extensions=["pdf"])
@pytest.mark.parametrize("texsystem", ("pdflatex", "xelatex", "lualatex"))
@pytest.mark.backend("pgf")
# 使用装饰器标记，比较图形是否相等，并设置测试参数化，支持的 LaTeX 系统
def test_minus_signs_with_tex(fig_test, fig_ref, texsystem):
    # 检查当前的 LaTeX 系统是否支持 PGF
    if not _check_for_pgf(texsystem):
        pytest.skip(texsystem + ' + pgf is required')
    # 设置 Matplotlib 使用的 PGF LaTeX 系统
    mpl.rcParams["pgf.texsystem"] = texsystem
    # 在测试图形对象上添加包含减号的文本
    fig_test.text(.5, .5, "$-1$")
    # 在参考图形对象上添加包含减号符号的文本
    fig_ref.text(.5, .5, "$\N{MINUS SIGN}1$")
# 使用 pytest 标记指定后端为 "pgf"，表示这是一个针对 PGF 图形库的测试用例
@pytest.mark.backend("pgf")
def test_sketch_params():
    # 创建一个大小为 3x3 英寸的图形和轴对象
    fig, ax = plt.subplots(figsize=(3, 3))
    # 设置 x 轴和 y 轴的刻度为空列表，即不显示刻度
    ax.set_xticks([])
    ax.set_yticks([])
    # 关闭轴的边框
    ax.set_frame_on(False)
    # 绘制一条简单的直线，并获取其句柄
    handle, = ax.plot([0, 1])
    # 设置直线的草图参数，包括缩放比例、长度和随机性种子
    handle.set_sketch_params(scale=5, length=30, randomness=42)

    # 创建一个 BytesIO 对象 fd 作为临时存储图形文件的字节流
    with BytesIO() as fd:
        # 将图形保存为 pgf 格式到字节流 fd 中
        fig.savefig(fd, format='pgf')
        # 将字节流 fd 的内容解码为字符串 buf
        buf = fd.getvalue().decode()

    # 预期的 pgf 基线字符串，用于检查保存的图形内容是否包含在其中
    baseline = r"""\pgfpathmoveto{\pgfqpoint{0.375000in}{0.300000in}}%
\pgfpathlineto{\pgfqpoint{2.700000in}{2.700000in}}%
\usepgfmodule{decorations}%
\usepgflibrary{decorations.pathmorphing}%
\pgfkeys{/pgf/decoration/.cd, """ \
    r"""segment length = 0.150000in, amplitude = 0.100000in}%
\pgfmathsetseed{42}%
\pgfdecoratecurrentpath{random steps}%
\pgfusepath{stroke}%"""
    # 断言保存的 pgf 内容中包含预期的基线内容
    assert baseline in buf


# 检查文档字体大小是否一致设置的测试用例
# 使用 @needs_pgf_xelatex 标记确保需要 xelatex 引擎支持
@pytest.mark.skipif(
    not _has_tex_package('unicode-math'), reason='needs unicode-math.sty'
)
# 使用 pytest 标记指定后端为 "pgf"
@pytest.mark.backend('pgf')
# 使用 image_comparison 比较图像，排除文本后进行比较
@image_comparison(['pgf_document_font_size.pdf'], style='default', remove_text=True)
def test_document_font_size():
    # 更新 matplotlib 的配置，使用 xelatex 引擎、禁用 rcfonts 和加载 unicode-math 包
    mpl.rcParams.update({
        'pgf.texsystem': 'xelatex',
        'pgf.rcfonts': False,
        'pgf.preamble': r'\usepackage{unicode-math}',
    })
    # 创建一个图形对象
    plt.figure()
    # 绘制一个空图例项，包含一个非常长的数学标签和一些文本
    plt.plot([],
             label=r'$this is a very very very long math label a \times b + 10^{-3}$ '
                   r'and some text'
             )
    # 绘制一个空图例项，显示文档字体大小信息
    plt.plot([],
             label=r'\normalsize the document font size is \the\fontdimen6\font'
             )
    # 添加图例
    plt.legend()
```