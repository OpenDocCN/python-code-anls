# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_backend_pdf.py`

```
# 导入需要的模块和库
import datetime  # 用于处理日期和时间
import decimal  # 提供高精度的十进制运算支持
import io  # 提供了对字节流的支持
import os  # 提供了对操作系统功能的访问
from pathlib import Path  # 提供了处理文件路径的类和方法

import numpy as np  # 用于科学计算的库
import pytest  # 用于编写和运行测试的库

import matplotlib as mpl  # 绘图库的核心模块
from matplotlib import (  # 从 matplotlib 中导入多个子模块和函数
    pyplot as plt, rcParams, font_manager as fm
)
from matplotlib.cbook import _get_data_path  # 获取 matplotlib 的数据路径
from matplotlib.ft2font import FT2Font  # 提供 FreeType 字体功能
from matplotlib.font_manager import findfont, FontProperties  # 字体管理相关功能
from matplotlib.backends._backend_pdf_ps import get_glyphs_subset  # 获取 PDF/PS 字形子集
from matplotlib.backends.backend_pdf import PdfPages  # 处理 PDF 文件的类
from matplotlib.patches import Rectangle  # 提供绘制矩形的功能
from matplotlib.testing.decorators import check_figures_equal, image_comparison  # 测试相关的装饰器
from matplotlib.testing._markers import needs_usetex  # 测试相关的标记，用于 LaTeX 渲染

@image_comparison(['pdf_use14corefonts.pdf'])
def test_use14corefonts():
    # 设置 PDF 后端选项，使用 14 种核心字体
    rcParams['pdf.use14corefonts'] = True
    rcParams['font.family'] = 'sans-serif'  # 字体族设置为无衬线字体
    rcParams['font.size'] = 8  # 设置字体大小为 8
    rcParams['font.sans-serif'] = ['Helvetica']  # 指定使用的无衬线字体为 Helvetica
    rcParams['pdf.compression'] = 0  # PDF 不压缩

    # 定义包含多行文本的字符串
    text = '''A three-line text positioned just above a blue line
and containing some French characters and the euro symbol:
"Merci pépé pour les 10 €"'''

    # 创建图形和坐标轴对象
    fig, ax = plt.subplots()
    ax.set_title('Test PDF backend with option use14corefonts=True')  # 设置标题
    ax.text(0.5, 0.5, text, horizontalalignment='center',  # 在指定位置绘制文本
            verticalalignment='bottom',
            fontsize=14)
    ax.axhline(0.5, linewidth=0.5)  # 绘制水平线


@pytest.mark.parametrize('fontname, fontfile', [
    ('DejaVu Sans', 'DejaVuSans.ttf'),  # 参数化测试用例，使用 DejaVu Sans 字体
    ('WenQuanYi Zen Hei', 'wqy-zenhei.ttc'),  # 参数化测试用例，使用 WenQuanYi Zen Hei 字体
])
@pytest.mark.parametrize('fonttype', [3, 42])  # 参数化测试用例，测试不同的字体类型
def test_embed_fonts(fontname, fontfile, fonttype):
    # 如果指定字体的文件名与当前系统中找到的字体文件名不一致，则跳过该测试
    if Path(findfont(FontProperties(family=[fontname]))).name != fontfile:
        pytest.skip(f'Font {fontname!r} may be missing')

    rcParams['pdf.fonttype'] = fonttype  # 设置 PDF 输出的字体类型
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])  # 绘制简单的曲线图
    ax.set_title('Axes Title', font=fontname)  # 设置坐标轴标题的字体为指定的字体
    fig.savefig(io.BytesIO(), format='pdf')  # 将图形保存为 PDF 格式


def test_multipage_pagecount():
    # 使用 PdfPages 创建一个内存中的 PDF 对象
    with PdfPages(io.BytesIO()) as pdf:
        assert pdf.get_pagecount() == 0  # 检查 PDF 中页面数量为 0
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])  # 绘制简单的曲线图
        fig.savefig(pdf, format="pdf")  # 将图形保存到 PDF 中
        assert pdf.get_pagecount() == 1  # 再次检查 PDF 中页面数量为 1
        pdf.savefig()  # 将当前图形保存到 PDF 中
        assert pdf.get_pagecount() == 2  # 再次检查 PDF 中页面数量为 2


def test_multipage_properfinalize():
    pdfio = io.BytesIO()
    with PdfPages(pdfio) as pdf:
        for i in range(10):
            fig, ax = plt.subplots()
            ax.set_title('This is a long title')  # 设置长标题
            fig.savefig(pdf, format="pdf")  # 将图形保存到 PDF 中
    s = pdfio.getvalue()
    assert s.count(b'startxref') == 1  # 检查 PDF 字节流中 'startxref' 出现的次数为 1
    assert len(s) < 40000  # 检查 PDF 字节流的长度小于 40000


def test_multipage_keep_empty(tmp_path):
    # 测试空 PDF 文件保留的行为

    # 未设置 keep_empty 的情况下，将会保留空的 PDF 文件
    fn = tmp_path / "a.pdf"
    with pytest.warns(mpl.MatplotlibDeprecationWarning), PdfPages(fn) as pdf:
        pass
    assert fn.exists()  # 检查文件存在性

    # 设置 keep_empty=True 的情况下，将会保留空的 PDF 文件
    fn = tmp_path / "b.pdf"
    # 使用 pytest.warns 检测并忽略 MatplotlibDeprecationWarning 警告，同时创建 PdfPages 对象保存到文件 fn
    # 这个 PDF 文件会保留空白页，因为设置了 keep_empty=True
    with pytest.warns(mpl.MatplotlibDeprecationWarning), \
            PdfPages(fn, keep_empty=True) as pdf:
        pass
    # 确保文件 fn 存在
    assert fn.exists()

    # 创建一个临时文件 fn，并创建 PdfPages 对象保存到文件 fn
    # 这个 PDF 文件会在创建后被删除，因为设置了 keep_empty=False
    fn = tmp_path / "c.pdf"
    with PdfPages(fn, keep_empty=False) as pdf:
        pass
    # 确保文件 fn 不存在
    assert not fn.exists()

    # 创建一个临时文件 fn，并创建 PdfPages 对象保存到文件 fn
    # 在文件中添加了内容后，文件 fn 应该会被保留，因为 keep_empty 未设置
    fn = tmp_path / "d.pdf"
    with PdfPages(fn) as pdf:
        pdf.savefig(plt.figure())
    # 确保文件 fn 存在
    assert fn.exists()

    # 创建一个临时文件 fn，并创建 PdfPages 对象保存到文件 fn
    # 在文件中添加了内容后，文件 fn 应该会被保留，因为设置了 keep_empty=True
    fn = tmp_path / "e.pdf"
    with pytest.warns(mpl.MatplotlibDeprecationWarning), \
            PdfPages(fn, keep_empty=True) as pdf:
        pdf.savefig(plt.figure())
    # 确保文件 fn 存在
    assert fn.exists()

    # 创建一个临时文件 fn，并创建 PdfPages 对象保存到文件 fn
    # 在文件中添加了内容后，文件 fn 应该会被保留，因为设置了 keep_empty=False
    fn = tmp_path / "f.pdf"
    with PdfPages(fn, keep_empty=False) as pdf:
        pdf.savefig(plt.figure())
    # 确保文件 fn 存在
    assert fn.exists()
# 测试组合图像的功能
def test_composite_image():
    # 测试保存图形时是否可以将多个图像合并成单个复合图像
    X, Y = np.meshgrid(np.arange(-5, 5, 1), np.arange(-5, 5, 1)
    # 创建一个网格
    Z = np.sin(Y ** 2)
    # 计算正弦值
    fig, ax = plt.subplots()
    # 创建图形和坐标轴
    ax.set_xlim(0, 3)
    # 设置坐标轴的 x 范围
    ax.imshow(Z, extent=[0, 1, 0, 1])
    # 在坐标轴上显示图像
    ax.imshow(Z[::-1], extent=[2, 3, 0, 1])
    # 在坐标轴上显示另一个图像
    plt.rcParams['image.composite_image'] = True
    # 设置图像参数
    with PdfPages(io.BytesIO()) as pdf:
        # 创建 PDF 文件对象
        fig.savefig(pdf, format="pdf")
        # 将图形保存到 PDF 文件中
        assert len(pdf._file._images) == 1
        # 断言 PDF 文件中的图像数量为 1
    plt.rcParams['image.composite_image'] = False
    # 重置图像参数
    with PdfPages(io.BytesIO()) as pdf:
        # 创建 PDF 文件对象
        fig.savefig(pdf, format="pdf")
        # 将图形保存到 PDF 文件中
        assert len(pdf._file._images) == 2
        # 断言 PDF 文件中的图像数量为 2


# 测试索引图像的功能
def test_indexed_image():
    # 低颜色计数的图像应该压缩为调色板索引格式
    pikepdf = pytest.importorskip('pikepdf')

    data = np.zeros((256, 1, 3), dtype=np.uint8)
    # 创建一个 256x1x3 的零数组
    data[:, 0, 0] = np.arange(256)
    # 为索引图像设置最大唯一颜色

    rcParams['pdf.compression'] = True
    # 设置 PDF 压缩参数
    fig = plt.figure()
    # 创建图形对象
    fig.figimage(data, resize=True)
    # 在图形上显示图像
    buf = io.BytesIO()
    # 创建字节流对象
    fig.savefig(buf, format='pdf', dpi='figure')
    # 将图形保存到 PDF 文件中

    with pikepdf.Pdf.open(buf) as pdf:
        # 打开 PDF 文件
        page, = pdf.pages
        image, = page.images.values()
        pdf_image = pikepdf.PdfImage(image)
        # 获取 PDF 图像对象
        assert pdf_image.indexed
        # 断言 PDF 图像为索引图像
        pil_image = pdf_image.as_pil_image()
        rgb = np.asarray(pil_image.convert('RGB'))
        # 将 PDF 图像转换为 RGB 数组

    np.testing.assert_array_equal(data, rgb)
    # 断言数据与 RGB 数组相等


# 测试保存图形元数据的功能
def test_savefig_metadata(monkeypatch):
    pikepdf = pytest.importorskip('pikepdf')
    monkeypatch.setenv('SOURCE_DATE_EPOCH', '0')

    fig, ax = plt.subplots()
    # 创建图形和坐标轴
    ax.plot(range(5))

    md = {
        'Author': 'me',
        'Title': 'Multipage PDF',
        'Subject': 'Test page',
        'Keywords': 'test,pdf,multipage',
        'ModDate': datetime.datetime(
            1968, 8, 1, tzinfo=datetime.timezone(datetime.timedelta(0))),
        'Trapped': 'True'
    }
    # 创建元数据字典
    buf = io.BytesIO()
    # 创建字节流对象
    fig.savefig(buf, metadata=md, format='pdf')
    # 将图形保存到 PDF 文件中

    with pikepdf.Pdf.open(buf) as pdf:
        # 打开 PDF 文件
        info = {k: str(v) for k, v in pdf.docinfo.items()}
        # 获取 PDF 文件信息

    assert info == {
        '/Author': 'me',
        '/CreationDate': 'D:19700101000000Z',
        '/Creator': f'Matplotlib v{mpl.__version__}, https://matplotlib.org',
        '/Keywords': 'test,pdf,multipage',
        '/ModDate': 'D:19680801000000Z',
        '/Producer': f'Matplotlib pdf backend v{mpl.__version__}',
        '/Subject': 'Test page',
        '/Title': 'Multipage PDF',
        '/Trapped': '/True',
    }
    # 断言 PDF 文件信息与预期相符


# 测试无效元数据的功能
def test_invalid_metadata():
    fig, ax = plt.subplots()

    with pytest.warns(UserWarning,
                      match="Unknown infodict keyword: 'foobar'."):
        # 捕获警告信息
        fig.savefig(io.BytesIO(), format='pdf', metadata={'foobar': 'invalid'})
        # 将图形保存到 PDF 文件中，包含无效元数据
    # 使用 pytest 模块捕获 UserWarning 异常，确保警告信息包含指定字符串
    with pytest.warns(UserWarning,
                      match='not an instance of datetime.datetime.'):
        # 将图表保存为 PDF 格式，输出到一个字节流中，并设置元数据 ModDate
        fig.savefig(io.BytesIO(), format='pdf',
                    metadata={'ModDate': '1968-08-01'})

    # 使用 pytest 模块捕获 UserWarning 异常，确保警告信息包含指定字符串
    with pytest.warns(UserWarning,
                      match='not one of {"True", "False", "Unknown"}'):
        # 将图表保存为 PDF 格式，输出到一个字节流中，并设置元数据 Trapped
        fig.savefig(io.BytesIO(), format='pdf', metadata={'Trapped': 'foo'})

    # 使用 pytest 模块捕获 UserWarning 异常，确保警告信息包含指定字符串
    with pytest.warns(UserWarning, match='not an instance of str.'):
        # 将图表保存为 PDF 格式，输出到一个字节流中，并设置元数据 Title
        fig.savefig(io.BytesIO(), format='pdf', metadata={'Title': 1234})
def test_multipage_metadata(monkeypatch):
    # 导入 pikepdf 库，如果不存在则跳过测试
    pikepdf = pytest.importorskip('pikepdf')
    # 设置环境变量 SOURCE_DATE_EPOCH 为 '0'
    monkeypatch.setenv('SOURCE_DATE_EPOCH', '0')

    # 创建一个图形和轴对象
    fig, ax = plt.subplots()
    # 在轴上绘制一个简单的图形
    ax.plot(range(5))

    # 定义 PDF 元数据字典
    md = {
        'Author': 'me',  # 作者
        'Title': 'Multipage PDF',  # 标题
        'Subject': 'Test page',  # 主题
        'Keywords': 'test,pdf,multipage',  # 关键词
        'ModDate': datetime.datetime(1968, 8, 1, tzinfo=datetime.timezone(datetime.timedelta(0))),  # 修改日期
        'Trapped': 'True'  # 是否受限
    }
    # 创建一个字节流对象
    buf = io.BytesIO()
    # 使用 PdfPages 创建 PDF 文件，并添加图形页，同时传入元数据
    with PdfPages(buf, metadata=md) as pdf:
        pdf.savefig(fig)  # 将图形保存到 PDF 中
        pdf.savefig(fig)  # 再次保存相同的图形到 PDF 中

    # 使用 pikepdf 打开创建的 PDF 文件
    with pikepdf.Pdf.open(buf) as pdf:
        # 提取 PDF 文档信息，并转换为字符串形式的字典
        info = {k: str(v) for k, v in pdf.docinfo.items()}

    # 断言 PDF 文档信息与预期相符
    assert info == {
        '/Author': 'me',  # 作者
        '/CreationDate': 'D:19700101000000Z',  # 创建日期
        '/Creator': f'Matplotlib v{mpl.__version__}, https://matplotlib.org',  # 创建工具及其版本信息
        '/Keywords': 'test,pdf,multipage',  # 关键词
        '/ModDate': 'D:19680801000000Z',  # 修改日期
        '/Producer': f'Matplotlib pdf backend v{mpl.__version__}',  # PDF 生成工具及其版本信息
        '/Subject': 'Test page',  # 主题
        '/Title': 'Multipage PDF',  # 标题
        '/Trapped': '/True',  # 是否受限
    }


def test_text_urls():
    # 导入 pikepdf 库，如果不存在则跳过测试
    pikepdf = pytest.importorskip('pikepdf')

    # 定义用于测试的 URL
    test_url = 'https://test_text_urls.matplotlib.org/'

    # 创建一个指定大小的图形对象
    fig = plt.figure(figsize=(2, 1))
    # 在图形中添加文本，并指定文本链接的 URL
    fig.text(0.1, 0.1, 'test plain 123', url=f'{test_url}plain')
    fig.text(0.1, 0.4, 'test mathtext $123$', url=f'{test_url}mathtext')

    # 创建一个字节流对象
    with io.BytesIO() as fd:
        # 将图形保存为 PDF 格式
        fig.savefig(fd, format='pdf')

        # 使用 pikepdf 打开创建的 PDF 文件
        with pikepdf.Pdf.open(fd) as pdf:
            # 获取 PDF 页面的注释对象
            annots = pdf.pages[0].Annots

            # 在上下文管理器中迭代 PDF 注释对象，确保能够正确访问
            for y, fragment in [('0.1', 'plain'), ('0.4', 'mathtext')]:
                # 查找具有指定 URL 的注释对象
                annot = next(
                    (a for a in annots if a.A.URI == f'{test_url}{fragment}'),
                    None)
                # 断言找到了符合条件的注释对象
                assert annot is not None
                # 断言 QuadPoints 属性不存在
                assert getattr(annot, 'QuadPoints', None) is None
                # 断言注释对象的位置信息，单位为点（每英寸 72 点）
                assert annot.Rect[1] == decimal.Decimal(y) * 72


def test_text_rotated_urls():
    # 导入 pikepdf 库，如果不存在则跳过测试
    pikepdf = pytest.importorskip('pikepdf')

    # 定义用于测试的 URL
    test_url = 'https://test_text_urls.matplotlib.org/'

    # 创建一个指定大小的图形对象
    fig = plt.figure(figsize=(1, 1))
    # 在图形中添加旋转后的文本，并指定文本链接的 URL
    fig.text(0.1, 0.1, 'N', rotation=45, url=f'{test_url}')
    # 使用 BytesIO 创建一个内存中的字节流对象，用于保存图形的 PDF 数据
    with io.BytesIO() as fd:
        # 将图形保存为 PDF 格式，并写入到字节流对象中
        fig.savefig(fd, format='pdf')

        # 使用 pikepdf 打开保存在字节流中的 PDF 文件
        with pikepdf.Pdf.open(fd) as pdf:
            # 获取 PDF 第一页的注释对象列表
            annots = pdf.pages[0].Annots

            # 在上下文管理器内部迭代 PDF 注释对象
            # 查找具有特定 URI 的注释对象，如果找到则返回该注释对象，否则返回 None
            annot = next(
                (a for a in annots if a.A.URI == f'{test_url}'),
                None)
            # 断言确保找到了目标注释对象
            assert annot is not None
            # 断言确保目标注释对象具有 QuadPoints 属性且不为 None
            assert getattr(annot, 'QuadPoints', None) is not None
            # 断言确保注释对象的左上角坐标位置（单位为点，每英寸 72 点）
            assert annot.Rect[0] == \
               annot.QuadPoints[6] - decimal.Decimal('0.00001')
@needs_usetex
def test_text_urls_tex():
    # 导入必要的模块，并确保 pikepdf 可用
    pikepdf = pytest.importorskip('pikepdf')

    # 定义测试用的 URL
    test_url = 'https://test_text_urls.matplotlib.org/'

    # 创建一个 2x1 大小的图像
    fig = plt.figure(figsize=(2, 1))
    # 在图像中添加一个使用 TeX 渲染的文本，包含 URL 链接
    fig.text(0.1, 0.7, 'test tex $123$', usetex=True, url=f'{test_url}tex')

    # 将图像保存为 PDF 格式到内存中的字节流
    with io.BytesIO() as fd:
        fig.savefig(fd, format='pdf')

        # 使用 pikepdf 打开保存的 PDF
        with pikepdf.Pdf.open(fd) as pdf:
            # 获取 PDF 页面的注释
            annots = pdf.pages[0].Annots

            # 在 PDF 注释中查找特定 URL 的注释对象
            # 注意：必须在上下文管理器内进行注释的迭代，否则可能会由于 PDF 结构问题而失败
            annot = next(
                (a for a in annots if a.A.URI == f'{test_url}tex'),
                None)
            assert annot is not None
            # 断言注释对象的位置在 PDF 页面上的坐标（单位是点，72 点为一英寸）
            assert annot.Rect[1] == decimal.Decimal('0.7') * 72


def test_pdfpages_fspath():
    # 使用 PdfPages 将空图保存为 PDF 文件，验证其正常运行
    with PdfPages(Path(os.devnull)) as pdf:
        pdf.savefig(plt.figure())


@image_comparison(['hatching_legend.pdf'])
def test_hatching_legend():
    """Test for correct hatching on patches in legend"""
    # 创建一个图像，验证图例中填充的正确交叉线模式
    fig = plt.figure(figsize=(1, 2))

    # 创建两个矩形，设置填充颜色和交叉线模式
    a = Rectangle([0, 0], 0, 0, facecolor="green", hatch="XXXX")
    b = Rectangle([0, 0], 0, 0, facecolor="blue", hatch="XXXX")

    # 在图像中添加图例，包含两种颜色和交叉线模式的矩形
    fig.legend([a, b, a, b], ["", "", "", ""])


@image_comparison(['grayscale_alpha.pdf'])
def test_grayscale_alpha():
    """Masking images with NaN did not work for grayscale images"""
    # 创建一个灰度图像，并在图像中的 NaN 区域进行遮罩处理的测试
    x, y = np.ogrid[-2:2:.1, -2:2:.1]
    dd = np.exp(-(x**2 + y**2))
    dd[dd < .1] = np.nan
    fig, ax = plt.subplots()
    ax.imshow(dd, interpolation='none', cmap='gray_r')
    ax.set_xticks([])
    ax.set_yticks([])


@mpl.style.context('default')
@check_figures_equal(extensions=["pdf", "eps"])
def test_pdf_eps_savefig_when_color_is_none(fig_test, fig_ref):
    # 测试在颜色为 "none" 时保存 PDF 和 EPS 格式图像的一致性
    ax_test = fig_test.add_subplot()
    ax_test.set_axis_off()
    ax_test.plot(np.sin(np.linspace(-5, 5, 100)), "v", c="none")
    ax_ref = fig_ref.add_subplot()
    ax_ref.set_axis_off()


@needs_usetex
def test_failing_latex():
    """Test failing latex subprocess call"""
    # 测试 LaTeX 渲染失败的情况，期望抛出 RuntimeError 异常
    plt.xlabel("$22_2_2$", usetex=True)  # This fails with "Double subscript"
    with pytest.raises(RuntimeError):
        plt.savefig(io.BytesIO(), format="pdf")


def test_empty_rasterized():
    # 检查空图像以光栅化方式保存为 PDF 文件的情况
    fig, ax = plt.subplots()
    ax.plot([], [], rasterized=True)
    fig.savefig(io.BytesIO(), format="pdf")


@image_comparison(['kerning.pdf'])
def test_kerning():
    # 测试文本字距调整对 PDF 渲染的影响
    fig = plt.figure()
    s = "AVAVAVAVAVAVAVAV€AAVV"
    fig.text(0, .25, s, size=5)
    fig.text(0, .75, s, size=20)


def test_glyphs_subset():
    fpath = str(_get_data_path("fonts/ttf/DejaVuSerif.ttf"))
    chars = "these should be subsetted! 1234567890"

    # 创建非子集 FT2Font
    nosubfont = FT2Font(fpath)
    nosubfont.set_text(chars)

    # 创建子集 FT2Font
    subfont = FT2Font(get_glyphs_subset(fpath, chars))
    subfont.set_text(chars)

    nosubcmap = nosubfont.get_charmap()
    # 获取子字体的字符映射表
    subcmap = subfont.get_charmap()

    # 断言：所有唯一字符必须在子字体的字符映射表中存在
    assert {*chars} == {chr(key) for key in subcmap}

    # 断言：子字体的字符映射表条目数应少于未经子集化处理的字体的字符映射表条目数
    assert len(subcmap) < len(nosubcmap)

    # 断言：由于两个对象分配了相同的字符集，子字体的字形数应与未经子集化处理的字体的字形数相等
    assert subfont.get_num_glyphs() == nosubfont.get_num_glyphs()
# 使用 @image_comparison 装饰器比较生成的图像与参考图像 "multi_font_type3.pdf"，
# 容差(tol)为 4.6
@image_comparison(["multi_font_type3.pdf"], tol=4.6)
# 定义测试函数 test_multi_font_type3
def test_multi_font_type3():
    # 使用 WenQuanYi Zen Hei 字体创建 FontProperties 对象
    fp = fm.FontProperties(family=["WenQuanYi Zen Hei"])
    # 检查系统中是否安装了 "wqy-zenhei.ttc" 字体文件，若无则跳过测试
    if Path(fm.findfont(fp)).name != "wqy-zenhei.ttc":
        pytest.skip("Font may be missing")

    # 设定图表字体为 'DejaVu Sans' 和 'WenQuanYi Zen Hei'，大小为 27
    plt.rc('font', family=['DejaVu Sans', 'WenQuanYi Zen Hei'], size=27)
    # 设定 PDF 输出的字体类型为 3
    plt.rc('pdf', fonttype=3)

    # 创建图表对象
    fig = plt.figure()
    # 在图表上添加文本，包括几个汉字
    fig.text(0.15, 0.475, "There are 几个汉字 in between!")


# 使用 @image_comparison 装饰器比较生成的图像与参考图像 "multi_font_type42.pdf"，
# 容差(tol)为 2.2
@image_comparison(["multi_font_type42.pdf"], tol=2.2)
# 定义测试函数 test_multi_font_type42
def test_multi_font_type42():
    # 使用 WenQuanYi Zen Hei 字体创建 FontProperties 对象
    fp = fm.FontProperties(family=["WenQuanYi Zen Hei"])
    # 检查系统中是否安装了 "wqy-zenhei.ttc" 字体文件，若无则跳过测试
    if Path(fm.findfont(fp)).name != "wqy-zenhei.ttc":
        pytest.skip("Font may be missing")

    # 设定图表字体为 'DejaVu Sans' 和 'WenQuanYi Zen Hei'，大小为 27
    plt.rc('font', family=['DejaVu Sans', 'WenQuanYi Zen Hei'], size=27)
    # 设定 PDF 输出的字体类型为 42
    plt.rc('pdf', fonttype=42)

    # 创建图表对象
    fig = plt.figure()
    # 在图表上添加文本，包括几个汉字
    fig.text(0.15, 0.475, "There are 几个汉字 in between!")


# 使用 pytest.mark.parametrize 装饰器传递参数进行多组测试
@pytest.mark.parametrize('family_name, file_name',
                         [("Noto Sans", "NotoSans-Regular.otf"),
                          ("FreeMono", "FreeMono.otf")])
# 定义测试函数 test_otf_font_smoke，参数包括字体名称和文件名称
def test_otf_font_smoke(family_name, file_name):
    # 使用指定字体名称创建 FontProperties 对象
    fp = fm.FontProperties(family=[family_name])
    # 检查系统中是否安装了对应的字体文件，若无则跳过测试
    if Path(fm.findfont(fp)).name != file_name:
        pytest.skip(f"Font {family_name} may be missing")

    # 设定图表字体为指定字体，大小为 27
    plt.rc('font', family=[family_name], size=27)

    # 创建图表对象
    fig = plt.figure()
    # 在图表上添加文本，包括 Cyrillic 字符
    fig.text(0.15, 0.475, "Привет мир!")
    # 将图表保存为 PDF 格式到内存中
    fig.savefig(io.BytesIO(), format="pdf")
```