# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_font_manager.py`

```
# 导入必要的模块和库
from io import BytesIO, StringIO  # 导入字节流和字符串流模块
import gc  # 导入垃圾回收模块
import multiprocessing  # 导入多进程模块
import os  # 导入操作系统模块
from pathlib import Path  # 导入路径处理模块
from PIL import Image  # 导入图像处理模块
import shutil  # 导入文件操作模块
import sys  # 导入系统模块
import warnings  # 导入警告模块

import numpy as np  # 导入数值计算库
import pytest  # 导入测试框架

# 导入 Matplotlib 相关模块
from matplotlib.font_manager import (
    findfont, findSystemFonts, FontEntry, FontProperties, fontManager,
    json_dump, json_load, get_font, is_opentype_cff_font,
    MSUserFontDirectories, _get_fontconfig_fonts, ttfFontProperty)
from matplotlib import cbook, ft2font, pyplot as plt, rc_context, figure as mfigure
from matplotlib.testing import subprocess_run_helper

# 检查是否安装了 fc-list 命令行工具
has_fclist = shutil.which('fc-list') is not None


# 测试字体优先级的函数
def test_font_priority():
    # 设置上下文环境，指定字体为 'cmmi10' 和 'Bitstream Vera Sans'
    with rc_context(rc={
            'font.sans-serif':
            ['cmmi10', 'Bitstream Vera Sans']}):
        # 查找并返回 'sans-serif' 字体的文件路径
        fontfile = findfont(FontProperties(family=["sans-serif"]))
    # 断言找到的字体文件名为 'cmmi10.ttf'
    assert Path(fontfile).name == 'cmmi10.ttf'

    # 测试 get_charmap 方法，该方法不再在内部使用
    font = get_font(fontfile)
    cmap = font.get_charmap()
    # 断言字符映射长度为 131
    assert len(cmap) == 131
    # 断言特定字符的映射值为 30
    assert cmap[8729] == 30


# 测试字体权重计算的函数
def test_score_weight():
    # 断言 "regular" 到 "regular" 的权重为 0
    assert 0 == fontManager.score_weight("regular", "regular")
    # 断言 "bold" 到 "bold" 的权重为 0
    assert 0 == fontManager.score_weight("bold", "bold")
    # 断言 "normal" 到 "bold" 之间的权重在 0 到 "normal" 到 "bold" 之间的权重之间
    assert (0 < fontManager.score_weight(400, 400) <
            fontManager.score_weight("normal", "bold"))
    # 断言 "normal" 到 "regular" 之间的权重在 0 到 "normal" 到 "bold" 之间的权重之间
    assert (0 < fontManager.score_weight("normal", "regular") <
            fontManager.score_weight("normal", "bold"))
    # 断言 "normal" 到 "regular" 的权重等于 400 到 400 的权重
    assert (fontManager.score_weight("normal", "regular") ==
            fontManager.score_weight(400, 400))


# 测试字体管理对象的 JSON 序列化和反序列化功能
def test_json_serialization(tmp_path):
    # 无法在 Windows 上多次打开 NamedTemporaryFile，因此使用临时目录代替
    json_dump(fontManager, tmp_path / "fontlist.json")
    copy = json_load(tmp_path / "fontlist.json")
    # 忽略警告，验证对不同字体属性的查找
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'findfont: Font family.*not found')
        for prop in ({'family': 'STIXGeneral'},
                     {'family': 'Bitstream Vera Sans', 'weight': 700},
                     {'family': 'no such font family'}):
            fp = FontProperties(**prop)
            # 断言两个对象对特定字体属性的查找结果相同
            assert (fontManager.findfont(fp, rebuild_if_missing=False) ==
                    copy.findfont(fp, rebuild_if_missing=False))


# 测试 OTF 字体的识别功能
def test_otf():
    # 指定一个已知的 OTF 字体文件路径
    fname = '/usr/share/fonts/opentype/freefont/FreeMono.otf'
    # 如果文件存在，则断言该文件是 OTF/CFF 格式的字体文件
    if Path(fname).exists():
        assert is_opentype_cff_font(fname)
    # 遍历已知的字体列表，检查是否有 OTF 格式的字体文件，并验证其格式是否正确
    for f in fontManager.ttflist:
        if 'otf' in f.fname:
            with open(f.fname, 'rb') as fd:
                res = fd.read(4) == b'OTTO'
            assert res == is_opentype_cff_font(f.fname)


# 使用 pytest 装饰器标记，根据平台和是否安装 fontconfig 决定是否跳过测试
@pytest.mark.skipif(sys.platform == "win32" or not has_fclist,
                    reason='no fontconfig installed')
def test_get_fontconfig_fonts():
    # 断言获取的 fontconfig 字体列表长度大于 1
    assert len(_get_fontconfig_fonts()) > 1


# 使用 pytest 的参数化装饰器，测试不同的 hinting_factor 值对字体渲染的影响
@pytest.mark.parametrize('factor', [2, 4, 6, 8])
def test_hinting_factor(factor):
    # 查找并返回 "sans-serif" 字体的文件路径
    font = findfont(FontProperties(family=["sans-serif"]))

    # 使用不同的 hinting_factor 测试同一个字体文件
    font1 = get_font(font, hinting_factor=1)
    # 清空并设置字体对象 `font1` 的大小为 12 像素高，100 像素宽
    font1.clear()
    font1.set_size(12, 100)
    # 在字体对象 `font1` 上设置文本为 'abc'
    font1.set_text('abc')
    # 获取设置文本后的宽度和高度作为预期输出
    expected = font1.get_width_height()
    
    # 根据指定的 hinting factor 获取经过hint处理后的字体对象 `hinted_font`
    hinted_font = get_font(font, hinting_factor=factor)
    # 清空并设置经过 hint 处理后的字体对象 `hinted_font` 的大小为 12 像素高，100 像素宽
    hinted_font.clear()
    hinted_font.set_size(12, 100)
    # 在经过 hint 处理后的字体对象 `hinted_font` 上设置文本为 'abc'
    hinted_font.set_text('abc')
    # 断言经过 hint 处理后的字体对象 `hinted_font` 的宽度和高度与预期值 `expected` 在相对误差 0.1 的范围内完全匹配
    np.testing.assert_allclose(hinted_font.get_width_height(), expected,
                               rtol=0.1)
# 测试特定的 UTF-16 编码的 TrueType 字体文件是否可以正常加载
def test_utf16m_sfnt():
    try:
        # 在 fontManager 的 ttflist 中查找下一个符合条件的字体条目
        entry = next(entry for entry in fontManager.ttflist
                     if Path(entry.fname).name == "seguisbi.ttf")
    except StopIteration:
        # 如果找不到指定的字体文件，跳过测试并显示相应信息
        pytest.skip("Couldn't find seguisbi.ttf font to test against.")
    else:
        # 检查是否成功从字体的 sfnt 表中读取了 "semibold"，并设置其权重
        assert entry.weight == 600


# 测试查找 TrueType 字体集合文件 (.ttc) 的功能
def test_find_ttc():
    # 使用指定的字体家族创建 FontProperties 对象
    fp = FontProperties(family=["WenQuanYi Zen Hei"])
    # 查找并验证指定字体是否存在，若不存在则跳过测试
    if Path(findfont(fp)).name != "wqy-zenhei.ttc":
        pytest.skip("Font wqy-zenhei.ttc may be missing")
    # 创建一个图形对象和轴对象
    fig, ax = plt.subplots()
    # 在图形上添加文本，使用指定字体属性渲染中文字符
    ax.text(.5, .5, "\N{KANGXI RADICAL DRAGON}", fontproperties=fp)
    # 尝试以多种格式保存图形，但并不真正保存到文件中
    for fmt in ["raw", "svg", "pdf", "ps"]:
        fig.savefig(BytesIO(), format=fmt)


# 测试查找 Noto Sans CJK SC 字体的功能
def test_find_noto():
    # 使用指定的字体家族创建 FontProperties 对象
    fp = FontProperties(family=["Noto Sans CJK SC", "Noto Sans CJK JP"])
    # 查找并获取当前字体的文件名
    name = Path(findfont(fp)).name
    # 验证找到的字体文件是否符合预期，若不符合则跳过测试
    if name not in ("NotoSansCJKsc-Regular.otf", "NotoSansCJK-Regular.ttc"):
        pytest.skip(f"Noto Sans CJK SC font may be missing (found {name})")

    # 创建一个图形对象和轴对象
    fig, ax = plt.subplots()
    # 在图形上添加文本，测试用例中使用中文字符
    ax.text(0.5, 0.5, 'Hello, 你好', fontproperties=fp)
    # 尝试以多种格式保存图形，但并不真正保存到文件中
    for fmt in ["raw", "svg", "pdf", "ps"]:
        fig.savefig(BytesIO(), format=fmt)


# 测试处理无效字体文件的功能
def test_find_invalid(tmp_path):
    # 测试函数是否会正确抛出 FileNotFoundError 异常
    with pytest.raises(FileNotFoundError):
        get_font(tmp_path / 'non-existent-font-name.ttf')

    with pytest.raises(FileNotFoundError):
        get_font(str(tmp_path / 'non-existent-font-name.ttf'))

    with pytest.raises(FileNotFoundError):
        get_font(bytes(tmp_path / 'non-existent-font-name.ttf'))

    # 引入特定的 FT2Font 类，并测试特定异常类型的抛出
    from matplotlib.ft2font import FT2Font
    with pytest.raises(TypeError, match='font file or a binary-mode file'):
        FT2Font(StringIO())  # type: ignore[arg-type]


# 仅在满足特定条件时运行测试，限定于 Linux 系统且 fontconfig 已安装
@pytest.mark.skipif(sys.platform != 'linux' or not has_fclist,
                    reason='only Linux with fontconfig installed')
def test_user_fonts_linux(tmpdir, monkeypatch):
    font_test_file = 'mpltest.ttf'

    # 预设条件：确保测试字体在系统中不存在
    fonts = findSystemFonts()
    if any(font_test_file in font for font in fonts):
        pytest.skip(f'{font_test_file} already exists in system fonts')

    # 准备一个临时的用户字体目录
    user_fonts_dir = tmpdir.join('fonts')
    user_fonts_dir.ensure(dir=True)
    shutil.copyfile(Path(__file__).parent / font_test_file,
                    user_fonts_dir.join(font_test_file))

    with monkeypatch.context() as m:
        m.setenv('XDG_DATA_HOME', str(tmpdir))
        _get_fontconfig_fonts.cache_clear()
        # 现在，字体应该可以在系统中找到
        fonts = findSystemFonts()
        assert any(font_test_file in font for font in fonts)

    # 确保清除临时目录的缓存
    _get_fontconfig_fonts.cache_clear()
    """Smoke test that addfont() accepts pathlib.Path."""
    # 定义一个测试文件的文件名
    font_test_file = 'mpltest.ttf'
    # 获取当前脚本文件的父目录，并拼接上测试文件名，形成完整的文件路径
    path = Path(__file__).parent / font_test_file
    try:
        # 调用字体管理器的 addfont 方法，传入文件路径作为参数
        fontManager.addfont(path)
        # 查找字体管理器中已添加的字体列表，找到刚添加的字体
        added, = [font for font in fontManager.ttflist
                  if font.fname.endswith(font_test_file)]
        # 从字体管理器中移除刚添加的字体
        fontManager.ttflist.remove(added)
    finally:
        # 查找字体管理器中所有以测试文件名结尾的字体
        to_remove = [font for font in fontManager.ttflist
                     if font.fname.endswith(font_test_file)]
        # 逐个从字体管理器中移除这些字体
        for font in to_remove:
            fontManager.ttflist.remove(font)
@pytest.mark.skipif(sys.platform != 'win32', reason='Windows only')
# 标记为条件跳过测试，仅在 Windows 平台下运行
def test_user_fonts_win32():
    if not (os.environ.get('APPVEYOR') or os.environ.get('TF_BUILD')):
        pytest.xfail("This test should only run on CI (appveyor or azure) "
                     "as the developer's font directory should remain "
                     "unchanged.")
    pytest.xfail("We need to update the registry for this test to work")
    font_test_file = 'mpltest.ttf'

    # Precondition: the test font should not be available
    fonts = findSystemFonts()
    if any(font_test_file in font for font in fonts):
        pytest.skip(f'{font_test_file} already exists in system fonts')

    user_fonts_dir = MSUserFontDirectories[0]

    # Make sure that the user font directory exists (this is probably not the
    # case on Windows versions < 1809)
    os.makedirs(user_fonts_dir)

    # Copy the test font to the user font directory
    shutil.copy(Path(__file__).parent / font_test_file, user_fonts_dir)

    # Now, the font should be available
    fonts = findSystemFonts()
    assert any(font_test_file in font for font in fonts)


def _model_handler(_):
    fig, ax = plt.subplots()
    fig.savefig(BytesIO(), format="pdf")
    plt.close()


@pytest.mark.skipif(not hasattr(os, "register_at_fork"),
                    reason="Cannot register at_fork handlers")
# 标记为条件跳过测试，如果操作系统不支持在 fork 时注册处理程序
def test_fork():
    _model_handler(0)  # Make sure the font cache is filled.
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(processes=2) as pool:
        pool.map(_model_handler, range(2))


def test_missing_family(caplog):
    plt.rcParams["font.sans-serif"] = ["this-font-does-not-exist"]
    with caplog.at_level("WARNING"):
        findfont("sans")
    assert [rec.getMessage() for rec in caplog.records] == [
        "findfont: Font family ['sans'] not found. "
        "Falling back to DejaVu Sans.",
        "findfont: Generic family 'sans' not found because none of the "
        "following families were found: this-font-does-not-exist",
    ]


def _test_threading():
    import threading
    from matplotlib.ft2font import LOAD_NO_HINTING
    import matplotlib.font_manager as fm

    def loud_excepthook(args):
        raise RuntimeError("error in thread!")

    threading.excepthook = loud_excepthook

    N = 10
    b = threading.Barrier(N)

    def bad_idea(n):
        b.wait(timeout=5)
        for j in range(100):
            font = fm.get_font(fm.findfont("DejaVu Sans"))
            font.set_text(str(n), 0.0, flags=LOAD_NO_HINTING)

    threads = [
        threading.Thread(target=bad_idea, name=f"bad_thread_{j}", args=(j,))
        for j in range(N)
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join(timeout=9)
        if t.is_alive():
            raise RuntimeError("thread failed to join")


def test_fontcache_thread_safe():
    pytest.importorskip('threading')
    # 导入 threading 库，如果不可导入则跳过测试

    subprocess_run_helper(_test_threading, timeout=10)


def test_fontentry_dataclass():
    # 创建一个 FontEntry 对象，指定字体名称为 'font-name'
    fontent = FontEntry(name='font-name')

    # 调用 FontEntry 对象的 _repr_png_() 方法，获取表示字体的 PNG 数据
    png = fontent._repr_png_()

    # 使用 BytesIO 将 PNG 数据封装成字节流，并通过 Image.open() 打开为图像对象
    img = Image.open(BytesIO(png))

    # 断言图像对象的宽度大于 0
    assert img.width > 0

    # 断言图像对象的高度大于 0
    assert img.height > 0

    # 调用 FontEntry 对象的 _repr_html_() 方法，获取表示字体的 HTML 数据
    html = fontent._repr_html_()

    # 断言 HTML 数据以 "<img src=\"data:image/png;base64" 开头
    assert html.startswith("<img src=\"data:image/png;base64")
def test_fontentry_dataclass_invalid_path():
    # 使用 pytest 来断言 FileNotFoundError 是否被触发
    with pytest.raises(FileNotFoundError):
        # 创建 FontEntry 实例，指定一个无效路径，并命名为 'font-name'
        fontent = FontEntry(fname='/random', name='font-name')
        # 调用 _repr_html_ 方法
        fontent._repr_html_()


@pytest.mark.skipif(sys.platform == 'win32', reason='Linux or OS only')
def test_get_font_names():
    # 获取 matplotlib 的字体数据路径列表
    paths_mpl = [cbook._get_data_path('fonts', subdir) for subdir in ['ttf']]
    # 在指定路径中查找字体文件，使用 'ttf' 作为字体文件扩展名
    fonts_mpl = findSystemFonts(paths_mpl, fontext='ttf')
    # 查找系统中安装的所有字体文件，使用 'ttf' 作为字体文件扩展名
    fonts_system = findSystemFonts(fontext='ttf')
    ttf_fonts = []
    # 遍历 matplotlib 和系统字体文件路径中的所有字体文件
    for path in fonts_mpl + fonts_system:
        try:
            # 使用 FT2Font 加载字体文件
            font = ft2font.FT2Font(path)
            # 创建字体属性对象 ttfFontProperty
            prop = ttfFontProperty(font)
            # 将字体名称添加到 ttf_fonts 列表中
            ttf_fonts.append(prop.name)
        except Exception:
            pass
    # 去除重复的字体名称并排序，得到可用的字体列表
    available_fonts = sorted(list(set(ttf_fonts)))
    # 获取 matplotlib 当前已知的字体名称列表并排序
    mpl_font_names = sorted(fontManager.get_font_names())
    # 断言两个集合相等，即可用字体与 matplotlib 字体管理器中的字体一致
    assert set(available_fonts) == set(mpl_font_names)
    # 断言可用字体的数量与 matplotlib 字体管理器中的字体数量相等
    assert len(available_fonts) == len(mpl_font_names)
    # 断言两个排序后的列表相等，验证字体名称的顺序也相同
    assert available_fonts == mpl_font_names


def test_donot_cache_tracebacks():

    class SomeObject:
        pass

    def inner():
        # 创建 SomeObject 实例
        x = SomeObject()
        # 创建一个新的 Figure 对象
        fig = mfigure.Figure()
        # 创建一个子图
        ax = fig.subplots()
        # 向图形中添加文本 'aardvark'，使用不存在的字体 'doesnotexist'
        fig.text(.5, .5, 'aardvark', family='doesnotexist')
        # 使用 BytesIO 作为输出流
        with BytesIO() as out:
            # 捕获所有警告信息
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # 将图形保存到输出流中，格式为 'raw'
                fig.savefig(out, format='raw')

    # 调用 inner 函数
    inner()

    # 遍历当前所有的 Python 对象
    for obj in gc.get_objects():
        # 如果对象是 SomeObject 的实例
        if isinstance(obj, SomeObject):
            # 如果 inner 函数中创建的对象仍然存在，则测试失败
            pytest.fail("object from inner stack still alive")
```