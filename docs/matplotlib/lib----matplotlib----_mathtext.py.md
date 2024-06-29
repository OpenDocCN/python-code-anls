# `D:\src\scipysrc\matplotlib\lib\matplotlib\_mathtext.py`

```
"""
Implementation details for :mod:`.mathtext`.
"""

# 引入未来版本的注解特性
from __future__ import annotations

# 引入必要的库和模块
import abc
import copy
import enum
import functools
import logging
import os
import re
import types
import unicodedata
import string
import typing as T
from typing import NamedTuple

# 引入第三方库
import numpy as np
from pyparsing import (
    Empty, Forward, Literal, NotAny, oneOf, OneOrMore, Optional,
    ParseBaseException, ParseException, ParseExpression, ParseFatalException,
    ParserElement, ParseResults, QuotedString, Regex, StringEnd, ZeroOrMore,
    pyparsing_common, Group)

# 引入 matplotlib 库及相关模块
import matplotlib as mpl
from . import cbook
from ._mathtext_data import (
    latex_to_bakoma, stix_glyph_fixes, stix_virtual_fonts, tex2uni)
from .font_manager import FontProperties, findfont, get_font
from .ft2font import FT2Font, FT2Image, KERNING_DEFAULT

# 引入版本控制相关库
from packaging.version import parse as parse_version
from pyparsing import __version__ as pyparsing_version

# 根据不同的 pyparsing 版本选择导入 nestedExpr 或 nested_expr
if parse_version(pyparsing_version).major < 3:
    from pyparsing import nestedExpr as nested_expr
else:
    from pyparsing import nested_expr

# 如果是类型检查，则导入 Iterable 和 Glyph
if T.TYPE_CHECKING:
    from collections.abc import Iterable
    from .ft2font import Glyph

# 启用 pyparsing 的 Packrat parsing 优化
ParserElement.enablePackrat()

# 设置日志记录器
_log = logging.getLogger("matplotlib.mathtext")

##############################################################################
# FONTS


def get_unicode_index(symbol: str) -> int:  # Publicly exported.
    r"""
    Return the integer index (from the Unicode table) of *symbol*.

    Parameters
    ----------
    symbol : str
        A single (Unicode) character, a TeX command (e.g. r'\pi') or a Type1
        symbol name (e.g. 'phi').
    """
    try:  # This will succeed if symbol is a single Unicode char
        return ord(symbol)
    except TypeError:
        pass
    try:  # Is symbol a TeX symbol (i.e. \alpha)
        return tex2uni[symbol.strip("\\")]
    except KeyError as err:
        raise ValueError(
            f"{symbol!r} is not a valid Unicode character or TeX/Type1 symbol"
            ) from err


class VectorParse(NamedTuple):
    """
    The namedtuple type returned by ``MathTextParser("path").parse(...)``.

    Attributes
    ----------
    width, height, depth : float
        The global metrics.
    glyphs : list
        The glyphs including their positions.
    rect : list
        The list of rectangles.
    """
    width: float
    height: float
    depth: float
    glyphs: list[tuple[FT2Font, float, int, float, float]]
    rects: list[tuple[float, float, float, float]]

VectorParse.__module__ = "matplotlib.mathtext"


class RasterParse(NamedTuple):
    """
    The namedtuple type returned by ``MathTextParser("agg").parse(...)``.

    Attributes
    ----------
    ox, oy : float
        The offsets are always zero.
    width, height, depth : float
        The global metrics.
    image : FT2Image
        A raster image.
    """
    ox: float
    oy: float
    width: float
    height: float
    depth: float
    image: FT2Image
# 将`RasterParse`类的`__module__`属性设置为"matplotlib.mathtext"
RasterParse.__module__ = "matplotlib.mathtext"

# 定义一个名为`Output`的类
class Output:
    """
    `ship`操作框的结果：包含定位的字形和矩形的列表。

    该类不暴露给最终用户，而是由`.MathTextParser.parse`转换为`VectorParse`或`RasterParse`。
    """

    # 初始化方法，接受一个名为`box`的`Box`对象作为参数
    def __init__(self, box: Box):
        self.box = box
        self.glyphs: list[tuple[float, float, FontInfo]] = []  # (ox, oy, info)：存储字形的位置信息和字体信息的列表
        self.rects: list[tuple[float, float, float, float]] = []  # (x1, y1, x2, y2)：存储矩形的位置信息的列表

    # 将输出转换为`VectorParse`对象的方法
    def to_vector(self) -> VectorParse:
        # 对宽度、高度和深度进行上取整操作
        w, h, d = map(
            np.ceil, [self.box.width, self.box.height, self.box.depth])
        # 提取每个字形的信息并调整其位置信息，存储在`gs`列表中
        gs = [(info.font, info.fontsize, info.num, ox, h - oy + info.offset)
              for ox, oy, info in self.glyphs]
        # 提取每个矩形的位置信息，存储在`rs`列表中
        rs = [(x1, h - y2, x2 - x1, y2 - y1)
              for x1, y1, x2, y2 in self.rects]
        # 返回一个`VectorParse`对象
        return VectorParse(w, h + d, d, gs, rs)

    # 将输出转换为`RasterParse`对象的方法，支持反锯齿化选项
    def to_raster(self, *, antialiased: bool) -> RasterParse:
        # 计算字形和矩形的边界范围
        xmin = min([*[ox + info.metrics.xmin for ox, oy, info in self.glyphs],
                    *[x1 for x1, y1, x2, y2 in self.rects], 0]) - 1
        ymin = min([*[oy - info.metrics.ymax for ox, oy, info in self.glyphs],
                    *[y1 for x1, y1, x2, y2 in self.rects], 0]) - 1
        xmax = max([*[ox + info.metrics.xmax for ox, oy, info in self.glyphs],
                    *[x2 for x1, y1, x2, y2 in self.rects], 0]) + 1
        ymax = max([*[oy - info.metrics.ymin for ox, oy, info in self.glyphs],
                    *[y2 for x1, y1, x2, y2 in self.rects], 0]) + 1
        # 计算图像的宽度、高度和深度
        w = xmax - xmin
        h = ymax - ymin - self.box.depth
        d = ymax - ymin - self.box.height
        # 创建一个FT2Image对象，用于生成栅格化图像
        image = FT2Image(np.ceil(w), np.ceil(h + max(d, 0)))

        # 移动字形和矩形以填充图像，并进行反锯齿化处理
        shifted = ship(self.box, (-xmin, -ymin))

        # 将每个字形渲染到图像上
        for ox, oy, info in shifted.glyphs:
            info.font.draw_glyph_to_bitmap(
                image, ox, oy - info.metrics.iceberg, info.glyph,
                antialiased=antialiased)
        
        # 将每个矩形渲染到图像上
        for x1, y1, x2, y2 in shifted.rects:
            height = max(int(y2 - y1) - 1, 0)
            if height == 0:
                center = (y2 + y1) / 2
                y = int(center - (height + 1) / 2)
            else:
                y = int(y1)
            image.draw_rect_filled(int(x1), y, np.ceil(x2), y + height)
        
        # 返回一个`RasterParse`对象
        return RasterParse(0, 0, w, h + d, d, image)


# 定义一个名为`FontMetrics`的命名元组
class FontMetrics(NamedTuple):
    """
    字体的度量信息。

    Attributes
    ----------
    advance : float
        字形的前进距离（以点为单位）。
    """
    # advance: float
    # 描述：字形的水平前进距离，即从当前字形到下一个字形的水平距离
    
    # height: float
    # 描述：字形的高度，以点（points）为单位
    
    # width: float
    # 描述：字形的宽度，以点（points）为单位
    
    # xmin: float
    # 描述：字形的墨迹矩形（ink rectangle）的最小 X 坐标值
    
    # xmax: float
    # 描述：字形的墨迹矩形（ink rectangle）的最大 X 坐标值
    
    # ymin: float
    # 描述：字形的墨迹矩形（ink rectangle）的最小 Y 坐标值
    
    # ymax: float
    # 描述：字形的墨迹矩形（ink rectangle）的最大 Y 坐标值
    
    # iceberg: float
    # 描述：从基线到字形顶部的距离，这对应于 TeX 中“height”的定义
    
    # slanted: bool
    # 描述：指示字形是否应被视为“倾斜”的布尔值，目前用于字距调整（kerning）子/上标。
class FontInfo(NamedTuple):
    # 定义一个命名元组 FontInfo，包含字体对象、字体大小、PostScript 名称、字体度量、编号和字形偏移量
    font: FT2Font
    fontsize: float
    postscript_name: str
    metrics: FontMetrics
    num: int
    glyph: Glyph
    offset: float


class Fonts(abc.ABC):
    """
    An abstract base class for a system of fonts to use for mathtext.

    The class must be able to take symbol keys and font file names and
    return the character metrics.  It also delegates to a backend class
    to do the actual drawing.
    """

    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: int):
        """
        Parameters
        ----------
        default_font_prop : `~.font_manager.FontProperties`
            The default non-math font, or the base font for Unicode (generic)
            font rendering.
        load_glyph_flags : int
            Flags passed to the glyph loader (e.g. ``FT_Load_Glyph`` and
            ``FT_Load_Char`` for FreeType-based fonts).
        """
        # 初始化 Fonts 类的实例，设置默认字体属性和加载字形的标志
        self.default_font_prop = default_font_prop
        self.load_glyph_flags = load_glyph_flags

    def get_kern(self, font1: str, fontclass1: str, sym1: str, fontsize1: float,
                 font2: str, fontclass2: str, sym2: str, fontsize2: float,
                 dpi: float) -> float:
        """
        Get the kerning distance for font between *sym1* and *sym2*.

        See `~.Fonts.get_metrics` for a detailed description of the parameters.
        """
        # 返回字体之间的紧缩距离，默认情况下返回 0.0
        return 0.

    def _get_font(self, font: str) -> FT2Font:
        # 抽象方法，子类需要实现获取特定字体的功能
        raise NotImplementedError

    def _get_info(self, font: str, font_class: str, sym: str, fontsize: float,
                  dpi: float) -> FontInfo:
        # 抽象方法，子类需要实现获取特定字体、字体类、符号和字体大小的详细信息
        raise NotImplementedError

    def get_metrics(self, font: str, font_class: str, sym: str, fontsize: float,
                    dpi: float) -> FontMetrics:
        r"""
        Parameters
        ----------
        font : str
            One of the TeX font names: "tt", "it", "rm", "cal", "sf", "bf",
            "default", "regular", "bb", "frak", "scr".  "default" and "regular"
            are synonyms and use the non-math font.
        font_class : str
            One of the TeX font names (as for *font*), but **not** "bb",
            "frak", or "scr".  This is used to combine two font classes.  The
            only supported combination currently is ``get_metrics("frak", "bf",
            ...)``.
        sym : str
            A symbol in raw TeX form, e.g., "1", "x", or "\sigma".
        fontsize : float
            Font size in points.
        dpi : float
            Rendering dots-per-inch.

        Returns
        -------
        FontMetrics
        """
        # 获取指定字体、字体类、符号和字体大小的字体度量信息
        info = self._get_info(font, font_class, sym, fontsize, dpi)
        return info.metrics
    def render_glyph(self, output: Output, ox: float, oy: float, font: str,
                     font_class: str, sym: str, fontsize: float, dpi: float) -> None:
        """
        在位置 (*ox*, *oy*) 绘制指定的字形，使用剩余的参数（详见 `get_metrics` 进行详细描述）。
        """
        # 获取字形的信息，包括字体、字形类别、符号、字号和 DPI
        info = self._get_info(font, font_class, sym, fontsize, dpi)
        # 将绘制的字形信息添加到输出对象的字形列表中
        output.glyphs.append((ox, oy, info))

    def render_rect_filled(self, output: Output,
                           x1: float, y1: float, x2: float, y2: float) -> None:
        """
        绘制从 (*x1*, *y1*) 到 (*x2*, *y2*) 的填充矩形。
        """
        # 将填充矩形的坐标范围添加到输出对象的矩形列表中
        output.rects.append((x1, y1, x2, y2))

    def get_xheight(self, font: str, fontsize: float, dpi: float) -> float:
        """
        获取给定 *font* 和 *fontsize* 的 x-height。
        """
        # 抛出未实现错误，需要在子类中实现具体的功能
        raise NotImplementedError()

    def get_underline_thickness(self, font: str, fontsize: float, dpi: float) -> float:
        """
        获取与给定字体匹配的下划线的厚度。用作绘制分数线或根号线等的基本单位。
        """
        # 抛出未实现错误，需要在子类中实现具体的功能
        raise NotImplementedError()

    def get_sized_alternatives_for_symbol(self, fontname: str,
                                          sym: str) -> list[tuple[str, str]]:
        """
        如果您的字体提供同一符号的多个大小，请进行重写。应返回一个列表，其中包含与 *sym* 匹配的各种大小的符号。
        表达式渲染器将从该列表中选择适合特定情况的最合适大小。
        """
        # 默认返回具有给定符号的单一大小的字体名称和符号的元组列表
        return [(fontname, sym)]
    # TruetypeFonts 类，继承自 Fonts 类，并使用 ABCMeta 作为元类
    """
    A generic base class for all font setups that use Truetype fonts
    (through FT2Font).
    """

    # 初始化方法，接收 default_font_prop 和 load_glyph_flags 两个参数
    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: int):
        # 调用父类 Fonts 的初始化方法
        super().__init__(default_font_prop, load_glyph_flags)
        
        # 使用 functools.cache 对 self._get_info 方法进行缓存，类型标注中忽略 method-assign 警告
        self._get_info = functools.cache(self._get_info)  # type: ignore[method-assign]
        
        # 字典，用于存储字体对象
        self._fonts = {}
        
        # 字典，用于存储字体映射关系
        self.fontmap: dict[str | int, str] = {}

        # 查找默认字体文件名，并获取默认字体对象
        filename = findfont(self.default_font_prop)
        default_font = get_font(filename)
        
        # 将默认字体对象存储在 self._fonts 中
        self._fonts['default'] = default_font
        self._fonts['regular'] = default_font

    # 根据字体名称或索引获取字体对象
    def _get_font(self, font: str | int) -> FT2Font:
        if font in self.fontmap:
            basename = self.fontmap[font]
        else:
            # 如果字体不在 fontmap 中，则假设 font 是 str 类型
            # 注意：int 参数只会由子类传递，子类将 int 键放入 self.fontmap 中
            basename = T.cast(str, font)
        
        # 获取缓存的字体对象
        cached_font = self._fonts.get(basename)
        
        # 如果缓存中没有，并且 basename 对应的文件存在，则加载并缓存字体对象
        if cached_font is None and os.path.exists(basename):
            cached_font = get_font(basename)
            self._fonts[basename] = cached_font
            self._fonts[cached_font.postscript_name] = cached_font
            self._fonts[cached_font.postscript_name.lower()] = cached_font
        
        # 强制类型转换为 FT2Font 类型，FIXME: 不确定这个转换是否保证安全
        return T.cast(FT2Font, cached_font)

    # 根据字体、字形、字体大小和 DPI 获取偏移量
    def _get_offset(self, font: FT2Font, glyph: Glyph, fontsize: float,
                    dpi: float) -> float:
        # 如果字体的 PostScript 名称为 'Cmex10'，计算并返回特定的偏移量
        if font.postscript_name == 'Cmex10':
            return (glyph.height / 64 / 2) + (fontsize/3 * dpi/72)
        # 默认情况下返回 0
        return 0.

    # 获取字形信息，抛出未实现错误
    def _get_glyph(self, fontname: str, font_class: str,
                   sym: str) -> tuple[FT2Font, int, bool]:
        raise NotImplementedError

    # _get_info 方法的返回值在每个实例中被缓存
    # 获取字体信息，包括字形、编号和倾斜度
    def _get_info(self, fontname: str, font_class: str, sym: str, fontsize: float,
                  dpi: float) -> FontInfo:
        # 调用内部方法获取字形、编号和倾斜度
        font, num, slanted = self._get_glyph(fontname, font_class, sym)
        # 设置字体大小和 DPI
        font.set_size(fontsize, dpi)
        # 加载特定字形的字符
        glyph = font.load_char(num, flags=self.load_glyph_flags)

        # 将字形的边界框值转换为浮点数并缩放
        xmin, ymin, xmax, ymax = [val/64.0 for val in glyph.bbox]
        # 获取字形的偏移量
        offset = self._get_offset(font, glyph, fontsize, dpi)
        # 创建字体度量对象，包括字符的水平进度、高度、宽度及边界框
        metrics = FontMetrics(
            advance = glyph.linearHoriAdvance/65536.0,
            height  = glyph.height/64.0,
            width   = glyph.width/64.0,
            xmin    = xmin,
            xmax    = xmax,
            ymin    = ymin+offset,
            ymax    = ymax+offset,
            # iceberg 是 TeX 的 "height" 的等效值
            iceberg = glyph.horiBearingY/64.0 + offset,
            slanted = slanted
            )

        # 返回包含字体信息的对象
        return FontInfo(
            font            = font,
            fontsize        = fontsize,
            postscript_name = font.postscript_name,
            metrics         = metrics,
            num             = num,
            glyph           = glyph,
            offset          = offset
            )

    # 获取字体的 x-height 值
    def get_xheight(self, fontname: str, fontsize: float, dpi: float) -> float:
        # 获取指定字体对象
        font = self._get_font(fontname)
        # 设置字体大小和 DPI
        font.set_size(fontsize, dpi)
        # 获取字体中的 'pclt' 表
        pclt = font.get_sfnt_table('pclt')
        if pclt is None:
            # 如果 'pclt' 表不存在，采用简易的 x-height 计算方式
            metrics = self.get_metrics(
                fontname, mpl.rcParams['mathtext.default'], 'x', fontsize, dpi)
            return metrics.iceberg
        # 计算 x-height 的值
        xHeight = (pclt['xHeight'] / 64.0) * (fontsize / 12.0) * (dpi / 100.0)
        return xHeight

    # 获取字体的下划线粗细
    def get_underline_thickness(self, font: str, fontsize: float, dpi: float) -> float:
        # 此函数曾经从字体度量中获取下划线粗细，但信息不稳定，因此硬编码处理
        return ((0.75 / 12.0) * fontsize * dpi) / 72.0

    # 获取字距
    def get_kern(self, font1: str, fontclass1: str, sym1: str, fontsize1: float,
                 font2: str, fontclass2: str, sym2: str, fontsize2: float,
                 dpi: float) -> float:
        # 如果字体和字号匹配，则获取两字符之间的字距
        if font1 == font2 and fontsize1 == fontsize2:
            # 获取第一个字符的信息
            info1 = self._get_info(font1, fontclass1, sym1, fontsize1, dpi)
            # 获取第二个字符的信息
            info2 = self._get_info(font2, fontclass2, sym2, fontsize2, dpi)
            # 使用第一个字符的字体对象获取两字符间的字距，默认为 KERNING_DEFAULT
            font = info1.font
            return font.get_kerning(info1.num, info2.num, KERNING_DEFAULT) / 64
        # 否则调用超类的方法获取字距
        return super().get_kern(font1, fontclass1, sym1, fontsize1,
                                font2, fontclass2, sym2, fontsize2, dpi)
class BakomaFonts(TruetypeFonts):
    """
    Use the Bakoma TrueType fonts for rendering.

    Symbols are strewn about a number of font files, each of which has
    its own proprietary 8-bit encoding.
    """

    # 字体映射表，将简称映射到具体的字体文件名
    _fontmap = {
        'cal': 'cmsy10',
        'rm':  'cmr10',
        'tt':  'cmtt10',
        'it':  'cmmi10',
        'bf':  'cmb10',
        'sf':  'cmss10',
        'ex':  'cmex10',
    }

    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: int):
        self._stix_fallback = StixFonts(default_font_prop, load_glyph_flags)

        # 调用父类构造函数初始化
        super().__init__(default_font_prop, load_glyph_flags)

        # 遍历字体映射表，查找并设置每个字体文件的完整路径
        for key, val in self._fontmap.items():
            fullpath = findfont(val)
            self.fontmap[key] = fullpath
            self.fontmap[val] = fullpath

    # 倾斜符号集合，包含特定需要倾斜处理的符号
    _slanted_symbols = set(r"\int \oint".split())

    def _get_glyph(self, fontname: str, font_class: str,
                   sym: str) -> tuple[FT2Font, int, bool]:
        font = None
        # 检查字体映射中是否包含当前字体名，以及当前符号是否在 latex_to_bakoma 映射中
        if fontname in self.fontmap and sym in latex_to_bakoma:
            basename, num = latex_to_bakoma[sym]
            # 判断是否需要倾斜处理当前符号
            slanted = (basename == "cmmi10") or sym in self._slanted_symbols
            # 获取指定基本名称的字体对象
            font = self._get_font(basename)
        elif len(sym) == 1:
            # 对于单个字符的处理，根据字体名称判断是否需要倾斜
            slanted = (fontname == "it")
            # 获取指定字体名称的字体对象
            font = self._get_font(fontname)
            if font is not None:
                num = ord(sym)
        # 如果成功获取字体对象并且符号存在于字体中，则返回字体对象、符号编码、倾斜状态
        if font is not None and font.get_char_index(num) != 0:
            return font, num, slanted
        else:
            # 否则，使用备用字体对象进行处理
            return self._stix_fallback._get_glyph(fontname, font_class, sym)

    # Bakoma 字体包含许多预设大小的替代符号。AutoSizedChar 类将使用这些替代符号，
    # 并选择最接近所需大小的符号。
    # 定义一个私有静态变量，用于存储不同符号的大小变体及其对应的操作指令
    _size_alternatives = {
        '(':           [('rm', '('), ('ex', '\xa1'), ('ex', '\xb3'),
                        ('ex', '\xb5'), ('ex', '\xc3')],
        ')':           [('rm', ')'), ('ex', '\xa2'), ('ex', '\xb4'),
                        ('ex', '\xb6'), ('ex', '\x21')],
        '{':           [('cal', '{'), ('ex', '\xa9'), ('ex', '\x6e'),
                        ('ex', '\xbd'), ('ex', '\x28')],
        '}':           [('cal', '}'), ('ex', '\xaa'), ('ex', '\x6f'),
                        ('ex', '\xbe'), ('ex', '\x29')],
        # 对于 '[' ，第四种大小在 BaKoMa 字体中神秘地缺失，因此在 '[' 和 ']' 都省略了第四种大小
        '[':           [('rm', '['), ('ex', '\xa3'), ('ex', '\x68'),
                        ('ex', '\x22')],
        ']':           [('rm', ']'), ('ex', '\xa4'), ('ex', '\x69'),
                        ('ex', '\x23')],
        r'\lfloor':    [('ex', '\xa5'), ('ex', '\x6a'),
                        ('ex', '\xb9'), ('ex', '\x24')],
        r'\rfloor':    [('ex', '\xa6'), ('ex', '\x6b'),
                        ('ex', '\xba'), ('ex', '\x25')],
        r'\lceil':     [('ex', '\xa7'), ('ex', '\x6c'),
                        ('ex', '\xbb'), ('ex', '\x26')],
        r'\rceil':     [('ex', '\xa8'), ('ex', '\x6d'),
                        ('ex', '\xbc'), ('ex', '\x27')],
        r'\langle':    [('ex', '\xad'), ('ex', '\x44'),
                        ('ex', '\xbf'), ('ex', '\x2a')],
        r'\rangle':    [('ex', '\xae'), ('ex', '\x45'),
                        ('ex', '\xc0'), ('ex', '\x2b')],
        r'\__sqrt__':  [('ex', '\x70'), ('ex', '\x71'),
                        ('ex', '\x72'), ('ex', '\x73')],
        r'\backslash': [('ex', '\xb2'), ('ex', '\x2f'),
                        ('ex', '\xc2'), ('ex', '\x2d')],
        r'/':          [('rm', '/'), ('ex', '\xb1'), ('ex', '\x2e'),
                        ('ex', '\xcb'), ('ex', '\x2c')],
        r'\widehat':   [('rm', '\x5e'), ('ex', '\x62'), ('ex', '\x63'),
                        ('ex', '\x64')],
        r'\widetilde': [('rm', '\x7e'), ('ex', '\x65'), ('ex', '\x66'),
                        ('ex', '\x67')],
        r'<':          [('cal', 'h'), ('ex', 'D')],
        r'>':          [('cal', 'i'), ('ex', 'E')]
        }

    # 遍历一组别名和目标符号的元组列表，将别名对应的大小变体设置为与目标符号相同
    for alias, target in [(r'\leftparen', '('),
                          (r'\rightparen', ')'),
                          (r'\leftbrace', '{'),
                          (r'\rightbrace', '}'),
                          (r'\leftbracket', '['),
                          (r'\rightbracket', ']'),
                          (r'\{', '{'),
                          (r'\}', '}'),
                          (r'\[', '['),
                          (r'\]', ']')]:
        _size_alternatives[alias] = _size_alternatives[target]

    # 定义一个方法，返回指定符号在特定字体下的大小变体列表
    def get_sized_alternatives_for_symbol(self, fontname: str,
                                          sym: str) -> list[tuple[str, str]]:
        return self._size_alternatives.get(sym, [(fontname, sym)])
class UnicodeFonts(TruetypeFonts):
    """
    An abstract base class for handling Unicode fonts.

    While some reasonably complete Unicode fonts (such as DejaVu) may
    work in some situations, the only Unicode font I'm aware of with a
    complete set of math symbols is STIX.

    This class will "fallback" on the Bakoma fonts when a required
    symbol cannot be found in the font.
    """

    # Some glyphs are not present in the `cmr10` font, and must be brought in
    # from `cmsy10`. Map the Unicode indices of those glyphs to the indices at
    # which they are found in `cmsy10`.
    _cmr10_substitutions = {
        0x00D7: 0x00A3,  # Multiplication sign.
        0x2212: 0x00A1,  # Minus sign.
    }

    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: int):
        # This must come first so the backend's owner is set correctly
        fallback_rc = mpl.rcParams['mathtext.fallback']
        # Determine the font class based on the 'mathtext.fallback' setting
        font_cls: type[TruetypeFonts] | None = {
            'stix': StixFonts,
            'stixsans': StixSansFonts,
            'cm': BakomaFonts
        }.get(fallback_rc)
        # Create a fallback font object using the determined font class
        self._fallback_font = (font_cls(default_font_prop, load_glyph_flags)
                               if font_cls else None)

        super().__init__(default_font_prop, load_glyph_flags)
        # Populate font mappings for various styles ('cal', 'rm', 'tt', etc.)
        for texfont in "cal rm tt it bf sf bfit".split():
            prop = mpl.rcParams['mathtext.' + texfont]
            font = findfont(prop)
            self.fontmap[texfont] = font
        # Include 'cmex10' font in font mappings
        prop = FontProperties('cmex10')
        font = findfont(prop)
        self.fontmap['ex'] = font

        # Include STIX sized alternatives for glyphs if fallback is STIX
        if isinstance(self._fallback_font, StixFonts):
            stixsizedaltfonts = {
                 0: 'STIXGeneral',
                 1: 'STIXSizeOneSym',
                 2: 'STIXSizeTwoSym',
                 3: 'STIXSizeThreeSym',
                 4: 'STIXSizeFourSym',
                 5: 'STIXSizeFiveSym'}

            # Populate font mappings with STIX sized alternatives
            for size, name in stixsizedaltfonts.items():
                fullpath = findfont(name)
                self.fontmap[size] = fullpath
                self.fontmap[name] = fullpath

    _slanted_symbols = set(r"\int \oint".split())

    def _map_virtual_font(self, fontname: str, font_class: str,
                          uniindex: int) -> tuple[str, int]:
        # Directly return the input fontname and uniindex as a tuple
        return fontname, uniindex
    # 定义一个方法 `_get_glyph`，接受三个参数：字体名称 `fontname`，字体类别 `font_class`，符号 `sym`，返回一个元组
    def _get_glyph(self, fontname: str, font_class: str,
                   sym: str) -> tuple[FT2Font, int, bool]:
        # 尝试获取符号 `sym` 的 Unicode 码位
        try:
            uniindex = get_unicode_index(sym)
            found_symbol = True  # 标记找到符号
        except ValueError:
            # 如果找不到对应的 Unicode 映射，则使用问号的 Unicode 码位
            uniindex = ord('?')
            found_symbol = False
            # 记录警告日志，表示没有找到符号 `sym` 的 TeX 到 Unicode 映射
            _log.warning("No TeX to Unicode mapping for %a.", sym)

        # 调用 `_map_virtual_font` 方法，映射虚拟字体
        fontname, uniindex = self._map_virtual_font(
            fontname, font_class, uniindex)

        # 初始化新的字体名称为输入的字体名称
        new_fontname = fontname

        # 只有在 'it' 模式下，类别为 'L' 的字符才会斜体显示，希腊大写字母应该是正体的
        if found_symbol:
            if fontname == 'it' and uniindex < 0x10000:
                char = chr(uniindex)
                # 检查字符的 Unicode 类别是否为字母（L 开头），或者名称是否以 "GREEK CAPITAL" 开头
                if (unicodedata.category(char)[0] != "L"
                        or unicodedata.name(char).startswith("GREEK CAPITAL")):
                    new_fontname = 'rm'  # 如果不是字母或者是希腊大写字母，则使用正体字体 'rm'

            # 检查符号是否应该斜体显示，如果是 `_slanted_symbols` 中的符号则斜体显示
            slanted = (new_fontname == 'it') or sym in self._slanted_symbols
            found_symbol = False  # 重置找到符号的标记
            # 获取指定字体名称的字体对象
            font = self._get_font(new_fontname)
            if font is not None:
                # 如果符号 `uniindex` 在 `_cmr10_substitutions` 中，且当前字体是 "cmr10"，则替换字体为 "cmsy10"
                if (uniindex in self._cmr10_substitutions
                        and font.family_name == "cmr10"):
                    font = get_font(
                        cbook._get_data_path("fonts/ttf/cmsy10.ttf"))
                    # 使用 `_cmr10_substitutions` 中对应的替换符号的 Unicode 码位
                    uniindex = self._cmr10_substitutions[uniindex]
                # 获取符号 `uniindex` 的字形索引
                glyphindex = font.get_char_index(uniindex)
                # 如果找到了对应的字形索引，则标记找到符号
                if glyphindex != 0:
                    found_symbol = True

        # 如果未找到符号
        if not found_symbol:
            # 如果有备用字体 `_fallback_font`
            if self._fallback_font:
                # 在 `'it'` 或 `'regular'` 模式下，并且 `_fallback_font` 是 `StixFonts` 类型时，使用正体字体 `'rm'`
                if (fontname in ('it', 'regular')
                        and isinstance(self._fallback_font, StixFonts)):
                    fontname = 'rm'

                # 从 `_fallback_font` 获取指定字体名称、字体类别、符号 `sym` 的字形信息
                g = self._fallback_font._get_glyph(fontname, font_class, sym)
                # 获取字形的家族名称
                family = g[0].family_name
                # 如果家族名称在 `BakomaFonts._fontmap` 的值列表中，则更改家族名称为 "Computer Modern"
                if family in list(BakomaFonts._fontmap.values()):
                    family = "Computer Modern"
                # 记录信息日志，表示使用备用字体替换符号 `sym` 的字形信息
                _log.info("Substituting symbol %s from %s", sym, family)
                return g  # 返回备用字体的字形信息

            else:
                # 如果没有备用字体 `_fallback_font`
                if (fontname in ('it', 'regular')
                        and isinstance(self, StixFonts)):
                    return self._get_glyph('rm', font_class, sym)  # 使用正体字体 `'rm'`
                # 记录警告日志，表示字体 `new_fontname` 没有符号 `sym` 的字形信息，使用虚拟符号替代
                _log.warning("Font %r does not have a glyph for %a [U+%x], "
                             "substituting with a dummy symbol.",
                             new_fontname, sym, uniindex)
                # 获取正体字体 `'rm'` 的字形信息
                font = self._get_font('rm')
                uniindex = 0xA4  # 货币符号，作为缺省符号
                slanted = False  # 不使用斜体显示

        # 返回最终的字体对象 `font`、Unicode 码位 `uniindex`、斜体显示标记 `slanted`
        return font, uniindex, slanted
    # 定义一个方法，用于获取给定字体名称和符号的大小调整后的备选项列表
    def get_sized_alternatives_for_symbol(self, fontname: str,
                                          sym: str) -> list[tuple[str, str]]:
        # 如果存在备用字体对象（self._fallback_font），则调用其对应方法获取备选项
        if self._fallback_font:
            return self._fallback_font.get_sized_alternatives_for_symbol(
                fontname, sym)
        # 如果没有备用字体对象，则返回一个包含当前字体名称和符号的元组的列表作为默认备选项
        return [(fontname, sym)]
class DejaVuFonts(UnicodeFonts, metaclass=abc.ABCMeta):
    _fontmap: dict[str | int, str] = {}

    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: int):
        # This must come first so the backend's owner is set correctly
        # 检查当前实例是否为 DejaVuSerifFonts 类的实例
        if isinstance(self, DejaVuSerifFonts):
            # 如果是，则设置回退字体为 StixFonts 类的实例
            self._fallback_font = StixFonts(default_font_prop, load_glyph_flags)
        else:
            # 否则设置回退字体为 StixSansFonts 类的实例
            self._fallback_font = StixSansFonts(default_font_prop, load_glyph_flags)
        # 初始化 BakomaFonts 实例
        self.bakoma = BakomaFonts(default_font_prop, load_glyph_flags)
        # 调用 TruetypeFonts 的初始化方法
        TruetypeFonts.__init__(self, default_font_prop, load_glyph_flags)
        # 添加 Stix 字体大小相关替代字体到字体映射中
        self._fontmap.update({
            1: 'STIXSizeOneSym',
            2: 'STIXSizeTwoSym',
            3: 'STIXSizeThreeSym',
            4: 'STIXSizeFourSym',
            5: 'STIXSizeFiveSym',
        })
        # 遍历字体映射，为每个映射的字体查找完整路径并更新到字体映射中
        for key, name in self._fontmap.items():
            fullpath = findfont(name)
            self.fontmap[key] = fullpath
            self.fontmap[name] = fullpath

    def _get_glyph(self, fontname: str, font_class: str,
                   sym: str) -> tuple[FT2Font, int, bool]:
        # Override prime symbol to use Bakoma.
        # 如果符号为 '\prime'，则使用 Bakoma 字体实例的 _get_glyph 方法获取字形信息
        if sym == r'\prime':
            return self.bakoma._get_glyph(fontname, font_class, sym)
        else:
            # 否则，检查显示字体中是否存在该符号的字形
            uniindex = get_unicode_index(sym)
            font = self._get_font('ex')
            if font is not None:
                glyphindex = font.get_char_index(uniindex)
                if glyphindex != 0:
                    # 如果存在该字形，则调用父类 UnicodeFonts 的 _get_glyph 方法获取字形信息
                    return super()._get_glyph('ex', font_class, sym)
            # 如果不存在，则返回常规字形
            return super()._get_glyph(fontname, font_class, sym)


class DejaVuSerifFonts(DejaVuFonts):
    """
    A font handling class for the DejaVu Serif fonts

    If a glyph is not found it will fallback to Stix Serif
    """
    _fontmap = {
        'rm': 'DejaVu Serif',
        'it': 'DejaVu Serif:italic',
        'bf': 'DejaVu Serif:weight=bold',
        'bfit': 'DejaVu Serif:italic:bold',
        'sf': 'DejaVu Sans',
        'tt': 'DejaVu Sans Mono',
        'ex': 'DejaVu Serif Display',
        0:    'DejaVu Serif',
    }


class DejaVuSansFonts(DejaVuFonts):
    """
    A font handling class for the DejaVu Sans fonts

    If a glyph is not found it will fallback to Stix Sans
    """
    _fontmap = {
        'rm': 'DejaVu Sans',
        'it': 'DejaVu Sans:italic',
        'bf': 'DejaVu Sans:weight=bold',
        'bfit': 'DejaVu Sans:italic:bold',
        'sf': 'DejaVu Sans',
        'tt': 'DejaVu Sans Mono',
        'ex': 'DejaVu Sans Display',
        0:    'DejaVu Sans',
    }


class StixFonts(UnicodeFonts):
    """
    A font handling class for the STIX fonts.

    In addition to what UnicodeFonts provides, this class:
    # 字体映射表，将字体风格简称或编号映射到对应的字体文件名
    _fontmap: dict[str | int, str] = {
        'rm': 'STIXGeneral',                    # 普通字体
        'it': 'STIXGeneral:italic',             # 斜体字体
        'bf': 'STIXGeneral:weight=bold',        # 粗体字体
        'bfit': 'STIXGeneral:italic:bold',      # 斜体粗体字体
        'nonunirm': 'STIXNonUnicode',           # 非Unicode普通字体
        'nonuniit': 'STIXNonUnicode:italic',    # 非Unicode斜体字体
        'nonunibf': 'STIXNonUnicode:weight=bold',  # 非Unicode粗体字体
        0: 'STIXGeneral',                       # 尺寸为0的符号字体
        1: 'STIXSizeOneSym',                    # 尺寸为1的符号字体
        2: 'STIXSizeTwoSym',                    # 尺寸为2的符号字体
        3: 'STIXSizeThreeSym',                  # 尺寸为3的符号字体
        4: 'STIXSizeFourSym',                   # 尺寸为4的符号字体
        5: 'STIXSizeFiveSym',                   # 尺寸为5的符号字体
    }
    _fallback_font = None  # 默认的备用字体设为None
    _sans = False  # 是否使用无衬线字体，默认为False
    
    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: int):
        # 调用TruetypeFonts类的初始化方法
        TruetypeFonts.__init__(self, default_font_prop, load_glyph_flags)
        # 遍历字体映射表，查找每种字体对应的完整路径，并将结果存入字体映射表中
        for key, name in self._fontmap.items():
            fullpath = findfont(name)
            self.fontmap[key] = fullpath
            self.fontmap[name] = fullpath
    # 处理嵌入在其他字体中的“虚拟字体”
    def _map_virtual_font(self, fontname: str, font_class: str,
                          uniindex: int) -> tuple[str, int]:
        # 检查是否有针对特定字体的映射
        font_mapping = stix_virtual_fonts.get(fontname)
        # 如果是无衬线字体且没有映射，并且字体名称不是 'regular' 或 'default'
        if (self._sans and font_mapping is None
                and fontname not in ('regular', 'default')):
            # 使用无衬线字体的虚拟字体映射
            font_mapping = stix_virtual_fonts['sf']
            doing_sans_conversion = True
        else:
            doing_sans_conversion = False

        # 根据映射类型确定使用哪种映射方式
        if isinstance(font_mapping, dict):
            try:
                mapping = font_mapping[font_class]
            except KeyError:
                mapping = font_mapping['rm']
        elif isinstance(font_mapping, list):
            mapping = font_mapping
        else:
            mapping = None

        # 如果存在映射，则通过二分查找确定源字形的位置
        if mapping is not None:
            # 二分查找源字形
            lo = 0
            hi = len(mapping)
            while lo < hi:
                mid = (lo+hi)//2
                range = mapping[mid]
                if uniindex < range[0]:
                    hi = mid
                elif uniindex <= range[1]:
                    break
                else:
                    lo = mid + 1

            # 如果找到了对应的范围，则调整 uniindex 和字体名称
            if range[0] <= uniindex <= range[1]:
                uniindex = uniindex - range[0] + range[3]
                fontname = range[2]
            elif not doing_sans_conversion:
                # 否则生成一个虚拟字符
                uniindex = 0x1
                fontname = mpl.rcParams['mathtext.default']

        # 修正一些不正确的字形
        if fontname in ('rm', 'it'):
            uniindex = stix_glyph_fixes.get(uniindex, uniindex)

        # 处理私有使用区域的字形
        if fontname in ('it', 'rm', 'bf', 'bfit') and 0xe000 <= uniindex <= 0xf8ff:
            fontname = 'nonuni' + fontname

        return fontname, uniindex

    @functools.cache
    def get_sized_alternatives_for_symbol(  # type: ignore[override]
            self,
            fontname: str,
            sym: str) -> list[tuple[str, str]] | list[tuple[int, str]]:
        # 修正符号中的一些转义字符
        fixes = {
            '\\{': '{', '\\}': '}', '\\[': '[', '\\]': ']',
            '<': '\N{MATHEMATICAL LEFT ANGLE BRACKET}',
            '>': '\N{MATHEMATICAL RIGHT ANGLE BRACKET}',
        }
        sym = fixes.get(sym, sym)
        try:
            # 获取符号的 Unicode 码点
            uniindex = get_unicode_index(sym)
        except ValueError:
            return [(fontname, sym)]
        
        # 查找符号在不同字体大小下的替代形式
        alternatives = [(i, chr(uniindex)) for i in range(6)
                        if self._get_font(i).get_char_index(uniindex) != 0]
        
        # STIX 中的根号符号在最大尺寸下具有不正确的度量标准，导致与主体断开连接
        if sym == r'\__sqrt__':
            alternatives = alternatives[:-1]
        
        return alternatives
class StixSansFonts(StixFonts):
    """
    A font handling class for the STIX fonts (that uses sans-serif
    characters by default).
    """
    _sans = True



##############################################################################
# TeX-LIKE BOX MODEL

# The following is based directly on the document 'woven' from the
# TeX82 source code.  This information is also available in printed
# form:
#
#    Knuth, Donald E.. 1986.  Computers and Typesetting, Volume B:
#    TeX: The Program.  Addison-Wesley Professional.
#
# The most relevant "chapters" are:
#    Data structures for boxes and their friends
#    Shipping pages out (ship())
#    Packaging (hpack() and vpack())
#    Data structures for math mode
#    Subroutines for math mode
#    Typesetting math formulas
#
# Many of the docstrings below refer to a numbered "node" in that
# book, e.g., node123
#
# Note that (as TeX) y increases downward, unlike many other parts of
# matplotlib.

# How much text shrinks when going to the next-smallest level.
SHRINK_FACTOR   = 0.7
# The number of different sizes of chars to use, beyond which they will not
# get any smaller
NUM_SIZE_LEVELS = 6


class FontConstantsBase:
    """
    A set of constants that controls how certain things, such as sub-
    and superscripts are laid out.  These are all metrics that can't
    be reliably retrieved from the font metrics in the font itself.
    """
    # Percentage of x-height of additional horiz. space after sub/superscripts
    script_space: T.ClassVar[float] = 0.05

    # Percentage of x-height that sub/superscripts drop below the baseline
    subdrop: T.ClassVar[float] = 0.4

    # Percentage of x-height that superscripts are raised from the baseline
    sup1: T.ClassVar[float] = 0.7

    # Percentage of x-height that subscripts drop below the baseline
    sub1: T.ClassVar[float] = 0.3

    # Percentage of x-height that subscripts drop below the baseline when a
    # superscript is present
    sub2: T.ClassVar[float] = 0.5

    # Percentage of x-height that sub/superscripts are offset relative to the
    # nucleus edge for non-slanted nuclei
    delta: T.ClassVar[float] = 0.025

    # Additional percentage of last character height above 2/3 of the
    # x-height that superscripts are offset relative to the subscript
    # for slanted nuclei
    delta_slanted: T.ClassVar[float] = 0.2

    # Percentage of x-height that superscripts and subscripts are offset for
    # integrals
    delta_integral: T.ClassVar[float] = 0.1


class ComputerModernFontConstants(FontConstantsBase):
    script_space = 0.075
    subdrop = 0.2
    sup1 = 0.45
    sub1 = 0.2
    sub2 = 0.3
    delta = 0.075
    delta_slanted = 0.3
    delta_integral = 0.3


class STIXFontConstants(FontConstantsBase):
    script_space = 0.1
    sup1 = 0.8
    sub2 = 0.6
    delta = 0.05
    delta_slanted = 0.3
    delta_integral = 0.3


class STIXSansFontConstants(FontConstantsBase):
    script_space = 0.05
    sup1 = 0.8
    delta_slanted = 0.6



# End of code block
    # 定义一个变量 delta_integral，赋值为 0.3
    delta_integral = 0.3
class DejaVuSerifFontConstants(FontConstantsBase):
    pass


class DejaVuSansFontConstants(FontConstantsBase):
    pass


# Maps font family names to the FontConstantBase subclass to use
_font_constant_mapping = {
    'DejaVu Sans': DejaVuSansFontConstants,
    'DejaVu Sans Mono': DejaVuSansFontConstants,
    'DejaVu Serif': DejaVuSerifFontConstants,
    'cmb10': ComputerModernFontConstants,
    'cmex10': ComputerModernFontConstants,
    'cmmi10': ComputerModernFontConstants,
    'cmr10': ComputerModernFontConstants,
    'cmss10': ComputerModernFontConstants,
    'cmsy10': ComputerModernFontConstants,
    'cmtt10': ComputerModernFontConstants,
    'STIXGeneral': STIXFontConstants,
    'STIXNonUnicode': STIXFontConstants,
    'STIXSizeFiveSym': STIXFontConstants,
    'STIXSizeFourSym': STIXFontConstants,
    'STIXSizeThreeSym': STIXFontConstants,
    'STIXSizeTwoSym': STIXFontConstants,
    'STIXSizeOneSym': STIXFontConstants,
    # Map the fonts we used to ship, just for good measure
    'Bitstream Vera Sans': DejaVuSansFontConstants,
    'Bitstream Vera': DejaVuSansFontConstants,
    }


def _get_font_constant_set(state: ParserState) -> type[FontConstantsBase]:
    # Get the font constants corresponding to the font family name from the state
    constants = _font_constant_mapping.get(
        state.fontset._get_font(state.font).family_name, FontConstantsBase)
    # Check if the font is STIX and the fontset is StixSansFonts, use STIXSansFontConstants
    if (constants is STIXFontConstants and
            isinstance(state.fontset, StixSansFonts)):
        return STIXSansFontConstants
    # Return the determined constants for the font family name
    return constants


class Node:
    """A node in the TeX box model."""

    def __init__(self) -> None:
        # Initialize node size to zero
        self.size = 0

    def __repr__(self) -> str:
        # Return a string representation of the class name
        return type(self).__name__

    def get_kerning(self, next: Node | None) -> float:
        # Return zero kerning distance
        return 0.0

    def shrink(self) -> None:
        """
        Shrinks one level smaller.  There are only three levels of
        sizes, after which things will no longer get smaller.
        """
        # Increase the size attribute by one level to simulate shrinking
        self.size += 1

    def render(self, output: Output, x: float, y: float) -> None:
        """Render this node."""
        # Placeholder for rendering function, not implemented here


class Box(Node):
    """A node with a physical location."""

    def __init__(self, width: float, height: float, depth: float) -> None:
        # Initialize a box with width, height, and depth attributes
        super().__init__()
        self.width  = width
        self.height = height
        self.depth  = depth

    def shrink(self) -> None:
        # Override shrink method to shrink width, height, and depth of the box
        super().shrink()
        if self.size < NUM_SIZE_LEVELS:
            self.width  *= SHRINK_FACTOR
            self.height *= SHRINK_FACTOR
            self.depth  *= SHRINK_FACTOR

    def render(self, output: Output,  # type: ignore[override]
               x1: float, y1: float, x2: float, y2: float) -> None:
        # Placeholder for rendering function, not implemented here
        pass


class Vbox(Box):
    """A box with only height (zero width)."""

    def __init__(self, height: float, depth: float):
        # Initialize a vertical box with height and depth attributes
        super().__init__(0., height, depth)
    """A box with only width (zero height and depth)."""
    
    # 定义一个只有宽度的盒子，高度和深度都为零

    def __init__(self, width: float):
        # 调用父类的构造方法初始化盒子对象，设置宽度为给定的浮点数值，高度和深度都设为零
        super().__init__(width, 0., 0.)
# 表示一个字符节点，继承自 Node 类
class Char(Node):
    """
    A single character.

    Unlike TeX, the font information and metrics are stored with each `Char`
    to make it easier to lookup the font metrics when needed.  Note that TeX
    boxes have a width, height, and depth, unlike Type1 and TrueType which use
    a full bounding box and an advance in the x-direction.  The metrics must
    be converted to the TeX model, and the advance (if different from width)
    must be converted into a `Kern` node when the `Char` is added to its parent
    `Hlist`.
    """

    def __init__(self, c: str, state: ParserState):
        super().__init__()
        # 存储字符本身
        self.c = c
        # 获取字体设置相关的状态信息
        self.fontset = state.fontset
        self.font = state.font
        self.font_class = state.font_class
        self.fontsize = state.fontsize
        self.dpi = state.dpi
        # 实际宽度、高度和深度将在打包阶段设置，根据实际字体大小
        self._update_metrics()

    def __repr__(self) -> str:
        return '`%s`' % self.c

    def _update_metrics(self) -> None:
        # 获取字符的字体度量信息
        metrics = self._metrics = self.fontset.get_metrics(
            self.font, self.font_class, self.c, self.fontsize, self.dpi)
        # 如果字符是空格，则使用 advance 作为宽度
        if self.c == ' ':
            self.width = metrics.advance
        else:
            self.width = metrics.width
        self.height = metrics.iceberg
        # 深度为负的高度和冰山深度差值
        self.depth = -(metrics.iceberg - metrics.height)

    def is_slanted(self) -> bool:
        # 判断字符是否倾斜
        return self._metrics.slanted

    def get_kerning(self, next: Node | None) -> float:
        """
        Return the amount of kerning between this and the given character.

        This method is called when characters are strung together into `Hlist`
        to create `Kern` nodes.
        """
        # 计算当前字符与下一个字符之间的紧排距离
        advance = self._metrics.advance - self.width
        kern = 0.
        if isinstance(next, Char):
            # 获取当前字符与下一个字符之间的紧排距离
            kern = self.fontset.get_kern(
                self.font, self.font_class, self.c, self.fontsize,
                next.font, next.font_class, next.c, next.fontsize,
                self.dpi)
        return advance + kern

    def render(self, output: Output, x: float, y: float) -> None:
        # 渲染当前字符
        self.fontset.render_glyph(
            output, x, y,
            self.font, self.font_class, self.c, self.fontsize, self.dpi)

    def shrink(self) -> None:
        # 缩小字符的尺寸
        super().shrink()
        if self.size < NUM_SIZE_LEVELS:
            self.fontsize *= SHRINK_FACTOR
            self.width    *= SHRINK_FACTOR
            self.height   *= SHRINK_FACTOR
            self.depth    *= SHRINK_FACTOR


class Accent(Char):
    """
    The font metrics need to be dealt with differently for accents,
    since they are already offset correctly from the baseline in
    TrueType fonts.
    """
    # 更新字体指标信息并设置对象内部的相关属性
    def _update_metrics(self) -> None:
        # 获取字体指标数据并存储在 _metrics 属性中，使用给定的字体、字体类别、字符、字体大小和 DPI 参数
        metrics = self._metrics = self.fontset.get_metrics(
            self.font, self.font_class, self.c, self.fontsize, self.dpi)
        # 计算字体宽度，使用 xmax 和 xmin 指标
        self.width = metrics.xmax - metrics.xmin
        # 计算字体高度，使用 ymax 和 ymin 指标
        self.height = metrics.ymax - metrics.ymin
        # 深度属性设为 0
        self.depth = 0
    
    # 调用父类的 shrink 方法，然后更新字体指标信息
    def shrink(self) -> None:
        super().shrink()
        # 更新字体指标信息
        self._update_metrics()
    
    # 渲染字形到指定的输出位置
    def render(self, output: Output, x: float, y: float) -> None:
        # 在输出上渲染字形，调整 x 和 y 的起始位置以匹配字体指标的 xmin 和 ymin
        self.fontset.render_glyph(
            output, x - self._metrics.xmin, y + self._metrics.ymin,
            self.font, self.font_class, self.c, self.fontsize, self.dpi)
class List(Box):
    """A list of nodes (either horizontal or vertical)."""

    def __init__(self, elements: T.Sequence[Node]):
        # 调用父类构造函数初始化坐标为 (0., 0., 0.)
        super().__init__(0., 0., 0.)
        # 设置偏移量为 0，任意选取的偏移量
        self.shift_amount = 0.   # An arbitrary offset
        # 存储此列表的子节点
        self.children = [*elements]  # The child nodes of this list
        # 下面的参数在 vpack 和 hpack 函数中设置
        self.glue_set     = 0.   # The glue setting of this list
        self.glue_sign    = 0    # 0: normal, -1: shrinking, 1: stretching
        self.glue_order   = 0    # The order of infinity (0 - 3) for the glue

    def __repr__(self) -> str:
        # 返回列表的字符串表示，包括宽度、高度、深度、偏移量和子节点信息
        return '{}<w={:.02f} h={:.02f} d={:.02f} s={:.02f}>[{}]'.format(
            super().__repr__(),
            self.width, self.height,
            self.depth, self.shift_amount,
            ', '.join([repr(x) for x in self.children]))

    def _set_glue(self, x: float, sign: int, totals: list[float],
                  error_type: str) -> None:
        # 设置列表的粘连参数
        self.glue_order = o = next(
            # 查找此列表成员中使用的最高粘连阶数
            (i for i in range(len(totals))[::-1] if totals[i] != 0), 0)
        self.glue_sign = sign
        if totals[o] != 0.:
            self.glue_set = x / totals[o]
        else:
            self.glue_sign = 0
            self.glue_ratio = 0.
        if o == 0:
            if len(self.children):
                # 如果子节点不为空，记录警告日志
                _log.warning("%s %s: %r",
                             error_type, type(self).__name__, self)

    def shrink(self) -> None:
        # 收缩列表及其子节点
        for child in self.children:
            child.shrink()
        super().shrink()
        # 如果大小小于 NUM_SIZE_LEVELS，缩小偏移量和粘连设置
        if self.size < NUM_SIZE_LEVELS:
            self.shift_amount *= SHRINK_FACTOR
            self.glue_set     *= SHRINK_FACTOR


class Hlist(List):
    """A horizontal list of boxes."""

    def __init__(self, elements: T.Sequence[Node], w: float = 0.0,
                 m: T.Literal['additional', 'exactly'] = 'additional',
                 do_kern: bool = True):
        # 调用父类构造函数初始化水平列表
        super().__init__(elements)
        if do_kern:
            # 如果需要，进行字符之间的紧排
            self.kern()
        # 对列表进行水平打包
        self.hpack(w=w, m=m)

    def kern(self) -> None:
        """
        Insert `Kern` nodes between `Char` nodes to set kerning.

        The `Char` nodes themselves determine the amount of kerning they need
        (in `~Char.get_kerning`), and this function just creates the correct
        linked list.
        """
        new_children = []
        num_children = len(self.children)
        if num_children:
            for i in range(num_children):
                elem = self.children[i]
                if i < num_children - 1:
                    next = self.children[i + 1]
                else:
                    next = None

                new_children.append(elem)
                # 获取当前字符与下一个字符之间的紧排距离
                kerning_distance = elem.get_kerning(next)
                if kerning_distance != 0.:
                    # 如果紧排距离不为零，插入 Kern 节点
                    kern = Kern(kerning_distance)
                    new_children.append(kern)
            self.children = new_children
    def hpack(self, w: float = 0.0,
              m: T.Literal['additional', 'exactly'] = 'additional') -> None:
        r"""
        Compute the dimensions of the resulting boxes, and adjust the glue if
        one of those dimensions is pre-specified.  The computed sizes normally
        enclose all of the material inside the new box; but some items may
        stick out if negative glue is used, if the box is overfull, or if a
        ``\vbox`` includes other boxes that have been shifted left.

        Parameters
        ----------
        w : float, default: 0
            A width.
        m : {'exactly', 'additional'}, default: 'additional'
            Whether to produce a box whose width is 'exactly' *w*; or a box
            with the natural width of the contents, plus *w* ('additional').

        Notes
        -----
        The defaults produce a box with the natural width of the contents.
        """
        # I don't know why these get reset in TeX.  Shift_amount is pretty
        # much useless if we do.
        # self.shift_amount = 0.
        h = 0.  # 初始化高度为 0
        d = 0.  # 初始化深度为 0
        x = 0.  # 初始化宽度为 0
        total_stretch = [0.] * 4  # 初始化总拉伸量列表
        total_shrink = [0.] * 4  # 初始化总收缩量列表
        for p in self.children:
            if isinstance(p, Char):
                x += p.width  # 累加字符的宽度到 x
                h = max(h, p.height)  # 更新最大高度
                d = max(d, p.depth)  # 更新最大深度
            elif isinstance(p, Box):
                x += p.width  # 累加盒子的宽度到 x
                if not np.isinf(p.height) and not np.isinf(p.depth):
                    s = getattr(p, 'shift_amount', 0.)
                    h = max(h, p.height - s)  # 更新最大高度，考虑偏移量
                    d = max(d, p.depth + s)  # 更新最大深度，考虑偏移量
            elif isinstance(p, Glue):
                glue_spec = p.glue_spec
                x += glue_spec.width  # 累加粘连的宽度到 x
                total_stretch[glue_spec.stretch_order] += glue_spec.stretch  # 累加粘连的拉伸量
                total_shrink[glue_spec.shrink_order] += glue_spec.shrink  # 累加粘连的收缩量
            elif isinstance(p, Kern):
                x += p.width  # 累加紧排的宽度到 x
        self.height = h  # 设置盒子的高度
        self.depth = d  # 设置盒子的深度

        if m == 'additional':
            w += x  # 如果 m 是 'additional'，则将宽度 w 增加到 x 上
        self.width = w  # 设置盒子的宽度为 w
        x = w - x  # 计算剩余宽度

        if x == 0.:
            self.glue_sign = 0  # 如果剩余宽度为 0，则设置粘连符号为 0
            self.glue_order = 0  # 设置粘连顺序为 0
            self.glue_ratio = 0.  # 设置粘连比例为 0
            return
        if x > 0.:
            self._set_glue(x, 1, total_stretch, "Overful")  # 如果剩余宽度大于 0，则设置粘连为过满状态
        else:
            self._set_glue(x, -1, total_shrink, "Underful")  # 如果剩余宽度小于 0，则设置粘连为欠满状态
class Vlist(List):
    """A vertical list of boxes."""

    def __init__(self, elements: T.Sequence[Node], h: float = 0.0,
                 m: T.Literal['additional', 'exactly'] = 'additional'):
        # 调用父类的初始化方法，传入节点序列作为元素
        super().__init__(elements)
        # 调用 vpack 方法对垂直列表进行封装，设置高度 h 和模式 m
        self.vpack(h=h, m=m)

    def vpack(self, h: float = 0.0,
              m: T.Literal['additional', 'exactly'] = 'additional',
              l: float = np.inf) -> None:
        """
        Compute the dimensions of the resulting boxes, and to adjust the glue
        if one of those dimensions is pre-specified.

        Parameters
        ----------
        h : float, default: 0
            A height.
        m : {'exactly', 'additional'}, default: 'additional'
            Whether to produce a box whose height is 'exactly' *h*; or a box
            with the natural height of the contents, plus *h* ('additional').
        l : float, default: np.inf
            The maximum height.

        Notes
        -----
        The defaults produce a box with the natural height of the contents.
        """
        # 初始化一些变量
        # 初始化宽度为 0
        w = 0.
        # 初始化深度为 0
        d = 0.
        # 初始化 x 为 0
        x = 0.
        # 初始化总伸长为 [0., 0., 0., 0.]
        total_stretch = [0.] * 4
        # 初始化总收缩为 [0., 0., 0., 0.]
        total_shrink = [0.] * 4
        # 遍历垂直列表的子元素
        for p in self.children:
            # 如果 p 是一个 Box 类型的对象
            if isinstance(p, Box):
                # 更新 x 的值
                x += d + p.height
                # 更新 d 的值为 p 的深度
                d = p.depth
                # 如果 p 的宽度不是无穷大
                if not np.isinf(p.width):
                    # 获取 p 的 shift_amount 属性，如果不存在，默认为 0
                    s = getattr(p, 'shift_amount', 0.)
                    # 更新 w 的值为 p 的宽度加上 shift_amount 后的最大值
                    w = max(w, p.width + s)
            # 如果 p 是一个 Glue 类型的对象
            elif isinstance(p, Glue):
                # 更新 x 的值
                x += d
                # 重置 d 为 0
                d = 0.
                # 获取 p 的 glue_spec 属性
                glue_spec = p.glue_spec
                # 更新 x 的值为 glue_spec 的宽度
                x += glue_spec.width
                # 根据 glue_spec 的伸长顺序更新总伸长列表
                total_stretch[glue_spec.stretch_order] += glue_spec.stretch
                # 根据 glue_spec 的收缩顺序更新总收缩列表
                total_shrink[glue_spec.shrink_order] += glue_spec.shrink
            # 如果 p 是一个 Kern 类型的对象
            elif isinstance(p, Kern):
                # 更新 x 的值
                x += d + p.width
                # 重置 d 为 0
                d = 0.
            # 如果 p 是一个 Char 类型的对象，抛出 RuntimeError 异常
            elif isinstance(p, Char):
                raise RuntimeError(
                    "Internal mathtext error: Char node found in Vlist")

        # 设置垂直列表的宽度为 w
        self.width = w
        # 如果 d 大于 l，则更新 x 的值和垂直列表的深度为 l
        if d > l:
            x += d - l
            self.depth = l
        else:
            self.depth = d

        # 如果 m 为 'additional'，则将 h 加上 x
        if m == 'additional':
            h += x
        # 设置垂直列表的高度为 h
        self.height = h
        # 计算并设置 x 的值
        x = h - x

        # 如果 x 等于 0，则设置以下属性为 0 并返回
        if x == 0:
            self.glue_sign = 0
            self.glue_order = 0
            self.glue_ratio = 0.
            return

        # 如果 x 大于 0，则调用 _set_glue 方法设置伸长属性，否则设置收缩属性
        if x > 0.:
            self._set_glue(x, 1, total_stretch, "Overful")
        else:
            self._set_glue(x, -1, total_shrink, "Underful")


class Rule(Box):
    """
    A solid black rectangle.

    It has *width*, *depth*, and *height* fields just as in an `Hlist`.
    However, if any of these dimensions is inf, the actual value will be
    determined by running the rule up to the boundary of the innermost
    enclosing box.  This is called a "running dimension".  The width is never
    """
    # 实现一个表示实心黑色矩形的类 Rule，继承自 Box 类
    """
    在一个 `Hlist` 中渲染，其高度和深度不会影响 `Vlist`。
    """

    def __init__(self, width: float, height: float, depth: float, state: ParserState):
        # 调用父类的初始化方法，传递宽度、高度和深度参数
        super().__init__(width, height, depth)
        # 将状态对象中的字体集合赋值给当前对象的字体集合属性
        self.fontset = state.fontset

    def render(self, output: Output,  # type: ignore[override]
               x: float, y: float, w: float, h: float) -> None:
        # 使用字体集合对象中的方法，在指定区域内渲染填充矩形
        self.fontset.render_rect_filled(output, x, y, x + w, y + h)


这段代码是一个类的定义和方法定义的示例，以下是每行代码的注释说明：

1. `"""`：多行字符串的开头，用于描述类中渲染方法的行为。
2. `def __init__(self, width: float, height: float, depth: float, state: ParserState):`：类的初始化方法，接受宽度、高度、深度和解析器状态作为参数。
3. `super().__init__(width, height, depth)`：调用父类（可能是一个名为 `super()` 的父类）的初始化方法，传递宽度、高度和深度参数。
4. `self.fontset = state.fontset`：将解析器状态对象中的字体集合赋值给当前对象的 `fontset` 属性。
5. `def render(self, output: Output,  # type: ignore[override]`：定义一个渲染方法，接受输出、位置及尺寸参数，类型标注中忽略类型检查和覆盖标记。
6. `self.fontset.render_rect_filled(output, x, y, x + w, y + h)`：使用当前对象的字体集合对象调用 `render_rect_filled` 方法，在指定的输出对象中渲染一个填充的矩形区域。

这些注释详细解释了每行代码的作用和含义，符合要求的注释风格和格式。
class Hrule(Rule):
    """Convenience class to create a horizontal rule."""

    def __init__(self, state: ParserState, thickness: float | None = None):
        # 如果未提供厚度参数，则使用当前解析状态中的下划线厚度
        if thickness is None:
            thickness = state.get_current_underline_thickness()
        # 水平线的高度和深度都是厚度的一半
        height = depth = thickness * 0.5
        # 调用父类 Rule 的构造函数来初始化水平线的属性
        super().__init__(np.inf, height, depth, state)


class Vrule(Rule):
    """Convenience class to create a vertical rule."""

    def __init__(self, state: ParserState):
        # 获取当前解析状态中的下划线厚度作为垂直线的厚度
        thickness = state.get_current_underline_thickness()
        # 调用父类 Rule 的构造函数来初始化垂直线的属性
        super().__init__(thickness, np.inf, np.inf, state)


class _GlueSpec(NamedTuple):
    width: float
    stretch: float
    stretch_order: int
    shrink: float
    shrink_order: int


_GlueSpec._named = {  # type: ignore[attr-defined]
    # 定义不同的 Glue 类型及其对应的 GlueSpec
    'fil':         _GlueSpec(0., 1., 1, 0., 0),
    'fill':        _GlueSpec(0., 1., 2, 0., 0),
    'filll':       _GlueSpec(0., 1., 3, 0., 0),
    'neg_fil':     _GlueSpec(0., 0., 0, 1., 1),
    'neg_fill':    _GlueSpec(0., 0., 0, 1., 2),
    'neg_filll':   _GlueSpec(0., 0., 0, 1., 3),
    'empty':       _GlueSpec(0., 0., 0, 0., 0),
    'ss':          _GlueSpec(0., 1., 1, -1., 1),
}


class Glue(Node):
    """
    Most of the information in this object is stored in the underlying
    ``_GlueSpec`` class, which is shared between multiple glue objects.
    (This is a memory optimization which probably doesn't matter anymore, but
    it's easier to stick to what TeX does.)
    """

    def __init__(self,
                 glue_type: _GlueSpec | T.Literal["fil", "fill", "filll",
                                                  "neg_fil", "neg_fill", "neg_filll",
                                                  "empty", "ss"]):
        # 初始化 Glue 对象，根据传入的 glue_type 确定 GlueSpec
        super().__init__()
        if isinstance(glue_type, str):
            # 如果 glue_type 是字符串，则从 _GlueSpec._named 中获取对应的 GlueSpec
            glue_spec = _GlueSpec._named[glue_type]  # type: ignore[attr-defined]
        elif isinstance(glue_type, _GlueSpec):
            # 如果 glue_type 是 _GlueSpec 类型，则直接使用该 GlueSpec
            glue_spec = glue_type
        else:
            # 如果既不是字符串也不是 _GlueSpec 类型，则抛出数值错误
            raise ValueError("glue_type must be a glue spec name or instance")
        self.glue_spec = glue_spec

    def shrink(self) -> None:
        # 缩小 Glue 对象的大小，如果大小小于 NUM_SIZE_LEVELS
        super().shrink()
        if self.size < NUM_SIZE_LEVELS:
            g = self.glue_spec
            # 根据 SHRINK_FACTOR 缩小 GlueSpec 中的宽度
            self.glue_spec = g._replace(width=g.width * SHRINK_FACTOR)


class HCentered(Hlist):
    """
    A convenience class to create an `Hlist` whose contents are
    centered within its enclosing box.
    """

    def __init__(self, elements: list[Node]):
        # 在水平居中的 Hlist 中加入 Glue 对象作为间距
        super().__init__([Glue('ss'), *elements, Glue('ss')], do_kern=False)


class VCentered(Vlist):
    """
    A convenience class to create a `Vlist` whose contents are
    centered within its enclosing box.
    """

    def __init__(self, elements: list[Node]):
        # 在垂直居中的 Vlist 中加入 Glue 对象作为间距
        super().__init__([Glue('ss'), *elements, Glue('ss')])


class Kern(Node):
    """
    A `Kern` node has a width field to specify a (normally
    negative) amount of spacing. This spacing correction appears in
    """
    height = 0
    depth = 0
    """

    # 初始化一个高度和深度为0的空间节点
    height = 0
    depth = 0

    def __init__(self, width: float):
        # 调用父类的构造方法初始化
        super().__init__()
        # 设置节点的宽度
        self.width = width

    def __repr__(self) -> str:
        # 返回节点的字符串表示，格式化宽度保留两位小数
        return "k%.02f" % self.width

    def shrink(self) -> None:
        # 调用父类的收缩方法
        super().shrink()
        # 如果节点尺寸小于预定义的尺寸级别数目，缩小节点的宽度
        if self.size < NUM_SIZE_LEVELS:
            self.width *= SHRINK_FACTOR
class AutoHeightChar(Hlist):
    """
    A character as close to the given height and depth as possible.

    When using a font with multiple height versions of some characters (such as
    the BaKoMa fonts), the correct glyph will be selected, otherwise this will
    always just return a scaled version of the glyph.
    """

    def __init__(self, c: str, height: float, depth: float, state: ParserState,
                 always: bool = False, factor: float | None = None):
        # 获取给定字符在当前字体下的不同高度版本
        alternatives = state.fontset.get_sized_alternatives_for_symbol(
            state.font, c)

        # 获取当前字体的 x-height（小写字母x的高度）
        xHeight = state.fontset.get_xheight(
            state.font, state.fontsize, state.dpi)

        # 复制当前解析状态
        state = state.copy()
        target_total = height + depth

        # 遍历所有备选字体和符号
        for fontname, sym in alternatives:
            state.font = fontname
            char = Char(sym, state)

            # 如果字符的总高度（高度+深度）足够接近目标高度，则选择该字符
            if char.height + char.depth >= target_total - 0.2 * xHeight:
                break

        # 计算字符在垂直方向上的偏移量
        shift = 0.0
        if state.font != 0 or len(alternatives) == 1:
            if factor is None:
                factor = target_total / (char.height + char.depth)
            state.fontsize *= factor
            char = Char(sym, state)

            shift = (depth - char.depth)

        # 调用父类构造函数，将字符列表传入，并设置偏移量
        super().__init__([char])
        self.shift_amount = shift


class AutoWidthChar(Hlist):
    """
    A character as close to the given width as possible.

    When using a font with multiple width versions of some characters (such as
    the BaKoMa fonts), the correct glyph will be selected, otherwise this will
    always just return a scaled version of the glyph.
    """

    def __init__(self, c: str, width: float, state: ParserState, always: bool = False,
                 char_class: type[Char] = Char):
        # 获取给定字符在当前字体下的不同宽度版本
        alternatives = state.fontset.get_sized_alternatives_for_symbol(
            state.font, c)

        # 复制当前解析状态
        state = state.copy()

        # 遍历所有备选字体和符号
        for fontname, sym in alternatives:
            state.font = fontname
            char = char_class(sym, state)

            # 如果字符的宽度足够接近目标宽度，则选择该字符
            if char.width >= width:
                break

        # 计算缩放因子，使字符宽度接近目标宽度
        factor = width / char.width
        state.fontsize *= factor
        char = char_class(sym, state)

        # 调用父类构造函数，将字符列表传入，并设置字符的宽度属性
        super().__init__([char])
        self.width = char.width


def ship(box: Box, xy: tuple[float, float] = (0, 0)) -> Output:
    """
    Ship out *box* at offset *xy*, converting it to an `Output`.

    Since boxes can be inside of boxes inside of boxes, the main work of `ship`
    is done by two mutually recursive routines, `hlist_out` and `vlist_out`,
    which traverse the `Hlist` nodes and `Vlist` nodes inside of horizontal
    and vertical boxes.  The global variables used in TeX to store state as it
    processes have become local variables here.
    """
    ox, oy = xy
    cur_v = 0.
    cur_h = 0.
    off_h = ox
    off_v = oy + box.height
    output = Output(box)

    def clamp(value: float) -> float:
        # 如果值小于 -1e9，则返回 -1e9；如果值大于 1e9，则返回 1e9；否则返回原值
        return -1e9 if value < -1e9 else +1e9 if value > +1e9 else value

    def hlist_out(box: Hlist) -> None:
        nonlocal cur_v, cur_h, off_h, off_v

        cur_g = 0  # 初始化当前粘连值为 0
        cur_glue = 0.  # 初始化当前粘连值为 0.0
        glue_order = box.glue_order  # 获取盒子的粘连顺序
        glue_sign = box.glue_sign  # 获取盒子的粘连符号
        base_line = cur_v  # 记录当前基线位置
        left_edge = cur_h  # 记录当前左边缘位置

        for p in box.children:
            if isinstance(p, Char):
                # 如果是字符对象，则在指定位置渲染字符并更新当前横坐标
                p.render(output, cur_h + off_h, cur_v + off_v)
                cur_h += p.width  # 更新当前横坐标位置
            elif isinstance(p, Kern):
                cur_h += p.width  # 如果是紧排对象，则只更新当前横坐标位置
            elif isinstance(p, List):
                # 如果是列表对象
                if len(p.children) == 0:
                    cur_h += p.width  # 如果列表为空，则只更新当前横坐标位置
                else:
                    edge = cur_h  # 记录当前边缘位置
                    cur_v = base_line + p.shift_amount  # 更新当前纵坐标位置
                    if isinstance(p, Hlist):
                        hlist_out(p)  # 递归处理水平列表对象
                    elif isinstance(p, Vlist):
                        vlist_out(p)  # 调用处理垂直列表对象的函数
                    else:
                        assert False, "unreachable code"  # 如果出现意外的列表类型，则抛出断言错误
                    cur_h = edge + p.width  # 恢复当前横坐标位置
                    cur_v = base_line  # 恢复当前纵坐标位置
            elif isinstance(p, Box):
                # 如果是盒子对象
                rule_height = p.height  # 获取盒子的高度
                rule_depth = p.depth  # 获取盒子的深度
                rule_width = p.width  # 获取盒子的宽度
                if np.isinf(rule_height):
                    rule_height = box.height  # 如果高度为无穷大，则使用盒子的高度
                if np.isinf(rule_depth):
                    rule_depth = box.depth  # 如果深度为无穷大，则使用盒子的深度
                if rule_height > 0 and rule_width > 0:
                    cur_v = base_line + rule_depth  # 更新当前纵坐标位置
                    p.render(output,
                             cur_h + off_h, cur_v + off_v,
                             rule_width, rule_height)  # 渲染盒子并更新当前坐标位置
                    cur_v = base_line  # 恢复当前纵坐标位置
                cur_h += rule_width  # 更新当前横坐标位置
            elif isinstance(p, Glue):
                # 如果是粘连对象
                glue_spec = p.glue_spec  # 获取粘连规格
                rule_width = glue_spec.width - cur_g  # 计算规则宽度
                if glue_sign != 0:  # 如果粘连符号不为零（正常情况）
                    if glue_sign == 1:  # 如果是拉伸状态
                        if glue_spec.stretch_order == glue_order:
                            cur_glue += glue_spec.stretch  # 更新当前粘连值
                            cur_g = round(clamp(box.glue_set * cur_glue))  # 计算并更新当前粘连量
                    elif glue_spec.shrink_order == glue_order:
                        cur_glue += glue_spec.shrink  # 更新当前粘连值
                        cur_g = round(clamp(box.glue_set * cur_glue))  # 计算并更新当前粘连量
                rule_width += cur_g  # 加上当前粘连量
                cur_h += rule_width  # 更新当前横坐标位置
    # 定义一个函数 vlist_out，接收一个 Vlist 对象作为参数，不返回任何结果
    def vlist_out(box: Vlist) -> None:
        # 使用 nonlocal 关键字声明外部作用域中的变量
        nonlocal cur_v, cur_h, off_h, off_v

        # 初始化当前行中的 glue 相关变量
        cur_g = 0
        cur_glue = 0.
        glue_order = box.glue_order
        glue_sign = box.glue_sign

        # 记录当前水平和垂直偏移
        left_edge = cur_h
        cur_v -= box.height
        top_edge = cur_v

        # 遍历 Vlist 中的每一个子元素
        for p in box.children:
            # 如果子元素是 Kern 对象，则调整当前垂直位置
            if isinstance(p, Kern):
                cur_v += p.width
            # 如果子元素是 List 对象
            elif isinstance(p, List):
                # 如果 List 中没有子元素，则根据高度和深度调整当前垂直位置
                if len(p.children) == 0:
                    cur_v += p.height + p.depth
                else:
                    # 否则，根据子元素的类型调整当前水平位置，并递归处理子元素
                    cur_v += p.height
                    cur_h = left_edge + p.shift_amount
                    save_v = cur_v
                    p.width = box.width
                    if isinstance(p, Hlist):
                        hlist_out(p)
                    elif isinstance(p, Vlist):
                        vlist_out(p)
                    else:
                        assert False, "unreachable code"
                    cur_v = save_v + p.depth
                    cur_h = left_edge
            # 如果子元素是 Box 对象
            elif isinstance(p, Box):
                # 获取 Box 的高度、深度和宽度，并根据需要进行调整
                rule_height = p.height
                rule_depth = p.depth
                rule_width = p.width
                if np.isinf(rule_width):
                    rule_width = box.width
                rule_height += rule_depth
                # 如果高度和深度均大于 0，则根据当前位置渲染 Box 对象到输出
                if rule_height > 0 and rule_depth > 0:
                    cur_v += rule_height
                    p.render(output,
                             cur_h + off_h, cur_v + off_v,
                             rule_width, rule_height)
            # 如果子元素是 Glue 对象
            elif isinstance(p, Glue):
                # 获取 Glue 对象的规范，并根据其属性调整当前 glue 变量和垂直位置
                glue_spec = p.glue_spec
                rule_height = glue_spec.width - cur_g
                if glue_sign != 0:  # normal
                    if glue_sign == 1:  # stretching
                        if glue_spec.stretch_order == glue_order:
                            cur_glue += glue_spec.stretch
                            cur_g = round(clamp(box.glue_set * cur_glue))
                    elif glue_spec.shrink_order == glue_order:  # shrinking
                        cur_glue += glue_spec.shrink
                        cur_g = round(clamp(box.glue_set * cur_glue))
                rule_height += cur_g
                cur_v += rule_height
            # 如果子元素是 Char 对象，则抛出运行时错误
            elif isinstance(p, Char):
                raise RuntimeError(
                    "Internal mathtext error: Char node found in vlist")

    # 确保 box 是一个 Hlist 对象，然后调用 hlist_out 函数处理该对象
    assert isinstance(box, Hlist)
    hlist_out(box)
    # 返回 output 结果
    return output
##############################################################################
# PARSER

# 定义一个函数，用于生成解析器错误
def Error(msg: str) -> ParserElement:
    """Helper class to raise parser errors."""
    # 内部函数，用于抛出解析致命异常，接受字符串、位置和解析结果作为参数
    def raise_error(s: str, loc: int, toks: ParseResults) -> T.Any:
        raise ParseFatalException(s, loc, msg)

    return Empty().setParseAction(raise_error)


class ParserState:
    """
    Parser state.

    States are pushed and popped from a stack as necessary, and the "current"
    state is always at the top of the stack.

    Upon entering and leaving a group { } or math/non-math, the stack is pushed
    and popped accordingly.
    """

    def __init__(self, fontset: Fonts, font: str, font_class: str, fontsize: float,
                 dpi: float):
        # 初始化解析器状态，接受字体集合、字体名称、字体类别、字体大小和 DPI 作为参数
        self.fontset = fontset
        self._font = font
        self.font_class = font_class
        self.fontsize = fontsize
        self.dpi = dpi

    def copy(self) -> ParserState:
        # 复制当前解析器状态对象
        return copy.copy(self)

    @property
    def font(self) -> str:
        # 返回当前字体名称
        return self._font

    @font.setter
    def font(self, name: str) -> None:
        # 设置字体名称，并根据名称更新字体类别
        if name in ('rm', 'it', 'bf', 'bfit'):
            self.font_class = name
        self._font = name

    def get_current_underline_thickness(self) -> float:
        """Return the underline thickness for this state."""
        # 返回当前状态下划线的厚度
        return self.fontset.get_underline_thickness(
            self.font, self.fontsize, self.dpi)


def cmd(expr: str, args: ParserElement) -> ParserElement:
    r"""
    Helper to define TeX commands.

    ``cmd("\cmd", args)`` is equivalent to
    ``"\cmd" - (args | Error("Expected \cmd{arg}{...}"))`` where the names in
    the error message are taken from element names in *args*.  If *expr*
    already includes arguments (e.g. "\cmd{arg}{...}"), then they are stripped
    when constructing the parse element, but kept (and *expr* is used as is) in
    the error message.
    """
    # 内部函数，用于获取解析器元素的名称生成器
    def names(elt: ParserElement) -> T.Generator[str, None, None]:
        if isinstance(elt, ParseExpression):
            for expr in elt.exprs:
                yield from names(expr)
        elif elt.resultsName:
            yield elt.resultsName

    # 根据表达式拆分命令名称
    csname = expr.split("{", 1)[0]
    # 构建错误消息中的命令名称及其参数列表
    err = (csname + "".join("{%s}" % name for name in names(args))
           if expr == csname else expr)
    # 返回解析器元素，该元素在不匹配时会触发错误
    return csname - (args | Error(f"Expected {err}"))


class Parser:
    """
    A pyparsing-based parser for strings containing math expressions.

    Raw text may also appear outside of pairs of ``$``.

    The grammar is based directly on that in TeX, though it cuts a few corners.
    """

    class _MathStyle(enum.Enum):
        # 定义数学表达式的不同样式
        DISPLAYSTYLE = 0
        TEXTSTYLE = 1
        SCRIPTSTYLE = 2
        SCRIPTSCRIPTSTYLE = 3
    # 定义一个包含各种二元运算符的集合
    _binary_operators = set(
      '+ * - \N{MINUS SIGN}'
      r'''
      \pm             \sqcap                   \rhd
      \mp             \sqcup                   \unlhd
      \times          \vee                     \unrhd
      \div            \wedge                   \oplus
      \ast            \setminus                \ominus
      \star           \wr                      \otimes
      \circ           \diamond                 \oslash
      \bullet         \bigtriangleup           \odot
      \cdot           \bigtriangledown         \bigcirc
      \cap            \triangleleft            \dagger
      \cup            \triangleright           \ddagger
      \uplus          \lhd                     \amalg
      \dotplus        \dotminus                \Cap
      \Cup            \barwedge                \boxdot
      \boxminus       \boxplus                 \boxtimes
      \curlyvee       \curlywedge              \divideontimes
      \doublebarwedge \leftthreetimes          \rightthreetimes
      \slash          \veebar                  \barvee
      \cupdot         \intercal                \amalg
      \circledcirc    \circleddash             \circledast
      \boxbar         \obar                    \merge
      \minuscolon     \dotsminusdots
      '''.split())
    # 创建一个包含关系符号的集合，这些符号被视为集合的元素
    _relation_symbols = set(r'''
      = < > :                  # 基本比较符号
      \leq          \geq          \equiv       \models  # 不等关系、等价、模型关系
      \prec         \succ         \sim         \perp    # 偏序、严格序、相似、垂直
      \preceq       \succeq       \simeq       \mid     # 小于等于、大于等于、相似于、竖线
      \ll           \gg           \asymp       \parallel  # 双小于、双大于、渐近等于、平行
      \subset       \supset       \approx      \bowtie  # 子集、超集、约等于、蝴蝶
      \subseteq     \supseteq     \cong        \Join    # 子集等于、超集等于、同构、连接
      \sqsubset     \sqsupset     \neq         \smile   # 方子集、方超集、不等于、笑脸
      \sqsubseteq   \sqsupseteq   \doteq       \frown   # 方子集等于、方超集等于、点等于、皱眉
      \in           \ni           \propto      \vdash   # 属于、包含、比例、竖线符号
      \dashv        \dots         \doteqdot    \leqq    # 竖线上倒T、省略号、点等点、小于等于等于
      \geqq         \lneqq        \gneqq       \lessgtr # 大于等于等于、小于不等于、大于不等于、小于大于
      \leqslant     \geqslant     \eqgtr       \eqless  # 小于等倾斜、大于等倾斜、等大于、等小于
      \eqslantless  \eqslantgtr   \lesseqgtr   \backsim # 等小于倾斜大于、等斜大于、小于等大于、反相似
      \backsimeq    \lesssim      \gtrsim      \precsim # 反相等、小于相似、大于相似、前相似
      \precnsim     \gnsim        \lnsim       \succsim # 前不相似、大不相似、小不相似、后相似
      \succnsim     \nsim         \lesseqqgtr  \gtreqqless  # 后不相似、不相似、小等大、大等小
      \gtreqless    \subseteqq    \supseteqq   \subsetneqq  # 大等小、子集等于、超集等于、真子集等于
      \supsetneqq   \lessapprox   \approxeq    \gtrapprox   # 真超集等于、小约等于、约等于、大约等于
      \precapprox   \succapprox   \precnapprox \succnapprox # 前约等于、后约等于、前不约等于、后不约等于
      \npreccurlyeq \nsucccurlyeq \nsqsubseteq \nsqsupseteq  # 不前曲小等、不后曲大等、不方子集等、不方超集等
      \sqsubsetneq  \sqsupsetneq  \nlesssim    \ngtrsim     # 方子集不等于、方超集不等于、不小相似、不大相似
      \nlessgtr     \ngtrless     \lnapprox    \gnapprox    # 不小大、不大小、不小约等于、不大约等于
      \napprox      \approxeq     \approxident \lll         # 不约等于、约等于、约等、三小于
      \ggg          \nparallel    \Vdash       \Vvdash      # 三大于、不平行、双竖线、双双竖线
      \nVdash       \nvdash       \vDash       \nvDash      # 不双竖线、不竖线、竖线、不竖线
      \nVDash       \oequal       \simneqq     \triangle    # 不双竖线、等圆圈、不等不等、三角形
      \triangleq         \triangleeq         \triangleleft  # 三角形、三角等、三角左
      \triangleright     \ntriangleleft      \ntriangleright # 三角右、不三角左、不三角右
      \trianglelefteq    \ntrianglelefteq    \trianglerighteq # 三角等左、不三角等左、三角等右
      \ntrianglerighteq  \blacktriangleleft  \blacktriangleright # 不三角等右、黑三角左、黑三角右
      \equalparallel     \measuredrightangle \varlrtriangle  # 等平行、测量右角、变角左右
      \Doteq        \Bumpeq       \Subset      \Supset       # 点等、帽子等、子集、超集
      \backepsilon  \because      \therefore   \bot          # 反角、因为、所以、底
      \top          \bumpeq       \circeq      \coloneq      # 顶、帽子、圈等、冒等
      \curlyeqprec  \curlyeqsucc  \eqcirc      \eqcolon      # 单曲线前、单曲线后、等圆、等冒
      \eqsim        \fallingdotseq \gtrdot     \gtrless      # 等相似、下落点等、大点、大小
      \ltimes       \rtimes       \lessdot     \ne           # 小时间、大时间、小点、不等于
      \ncong        \nequiv       \ngeq        \ngtr         # 不同、不等同、不大等、不大小
      \nleq         \nless        \nmid        \notin        # 不小等、不小、不中、不包含
      \nprec        \nsubset      \nsubseteq   \nsucc        # 不前、不子集、不子集等、不后
      \nsupset      \nsupseteq    \pitchfork   \preccurlyeq  # 不超集、不超集等、叉、前曲
      \risingdotseq \subsetneq    \succcurlyeq \supsetneq    # 上升点等、子集不等、后曲、超集不等
      \varpropto    \vartriangleleft \scurel    # 方符、方角左、爪
      \vartriangleright \rightangle \equal     # 方角右、直角、等
      \backcong     \eqdef        \wedgeq       \questeq      # 反同、等同、角等、问等
      \between      \veeeq        \disin        \varisins     # 之间、三角等、不在、变在
      \isins        \isindot      \varisinobar  \isinobar     # 在、点、变点、在点
      \isinvb       \isinE        \nisd         \varnis       # 在点、在等、不等、变
      \nis          \varniobar    \niobar       \bagmember    # 不变、变、袋在
      \ratio        \Equiv        \stareq       \measeq       # 比例、等价、点等、衡等
      \arceq        \rightassert  \rightModels  \smallin      # 叉等、就业、模型、小
      \smallowns    \notsmallowns \nsimeq'''.split())
    # 定义集合，包含数学箭头符号的 LaTeX 表示
    _arrow_symbols = set(r"""
     \leftarrow \longleftarrow \uparrow \Leftarrow \Longleftarrow
     \Uparrow \rightarrow \longrightarrow \downarrow \Rightarrow
     \Longrightarrow \Downarrow \leftrightarrow \updownarrow
     \longleftrightarrow \updownarrow \Leftrightarrow
     \Longleftrightarrow \Updownarrow \mapsto \longmapsto \nearrow
     \hookleftarrow \hookrightarrow \searrow \leftharpoonup
     \rightharpoonup \swarrow \leftharpoondown \rightharpoondown
     \nwarrow \rightleftharpoons \leadsto \dashrightarrow
     \dashleftarrow \leftleftarrows \leftrightarrows \Lleftarrow
     \Rrightarrow \twoheadleftarrow \leftarrowtail \looparrowleft
     \leftrightharpoons \curvearrowleft \circlearrowleft \Lsh
     \upuparrows \upharpoonleft \downharpoonleft \multimap
     \leftrightsquigarrow \rightrightarrows \rightleftarrows
     \rightrightarrows \rightleftarrows \twoheadrightarrow
     \rightarrowtail \looparrowright \rightleftharpoons
     \curvearrowright \circlearrowright \Rsh \downdownarrows
     \upharpoonright \downharpoonright \rightsquigarrow \nleftarrow
     \nrightarrow \nLeftarrow \nRightarrow \nleftrightarrow
     \nLeftrightarrow \to \Swarrow \Searrow \Nwarrow \Nearrow
     \leftsquigarrow \overleftarrow \overleftrightarrow \cwopencirclearrow
     \downzigzagarrow \cupleftarrow \rightzigzagarrow \twoheaddownarrow
     \updownarrowbar \twoheaduparrow \rightarrowbar \updownarrows
     \barleftarrow \mapsfrom \mapsdown \mapsup \Ldsh \Rdsh
     """.split())
    
    # 定义集合，包含空格分隔的数学操作符号和关系符号的 LaTeX 表示
    _spaced_symbols = _binary_operators | _relation_symbols | _arrow_symbols
    
    # 定义集合，包含逗号、分号、句点和 LaTeX 表示的标点符号
    _punctuation_symbols = set(r', ; . ! \ldotp \cdotp'.split())
    
    # 定义集合，包含上下限操作符的 LaTeX 表示
    _overunder_symbols = set(r'''
       \sum \prod \coprod \bigcap \bigcup \bigsqcup \bigvee
       \bigwedge \bigodot \bigotimes \bigoplus \biguplus
       '''.split())
    
    # 定义集合，包含上下限函数的名称
    _overunder_functions = set("lim liminf limsup sup max min".split())
    
    # 定义集合，包含积分和多重积分符号的 LaTeX 表示
    _dropsub_symbols = set(r'\int \oint \iint \oiint \iiint \oiiint \iiiint'.split())
    
    # 定义集合，包含字体名称
    _fontnames = set("rm cal it tt sf bf bfit "
                     "default bb frak scr regular".split())
    
    # 定义集合，包含数学函数的名称
    _function_names = set("""
      arccos csc ker min arcsin deg lg Pr arctan det lim sec arg dim
      liminf sin cos exp limsup sinh cosh gcd ln sup cot hom log tan
      coth inf max tanh""".split())
    
    # 定义集合，包含具有歧义的分隔符的 LaTeX 表示
    _ambi_delims = set(r"""
      | \| / \backslash \uparrow \downarrow \updownarrow \Uparrow
      \Downarrow \Updownarrow . \vert \Vert""".split())
    
    # 定义集合，包含左定界符的 LaTeX 表示
    _left_delims = set(r"""
      ( [ \{ < \lfloor \langle \lceil \lbrace \leftbrace \lbrack \leftparen \lgroup
      """.split())
    
    # 定义集合，包含右定界符的 LaTeX 表示
    _right_delims = set(r"""
      ) ] \} > \rfloor \rangle \rceil \rbrace \rightbrace \rbrack \rightparen \rgroup
      """.split())
    
    # 定义集合，包含所有定界符的 LaTeX 表示
    _delims = _left_delims | _right_delims | _ambi_delims
    
    # 定义集合，包含希腊小写字母的 Unicode 名称的最后一个单词的小写形式
    _small_greek = set([unicodedata.name(chr(i)).split()[-1].lower() for i in
                       range(ord('\N{GREEK SMALL LETTER ALPHA}'),
                             ord('\N{GREEK SMALL LETTER OMEGA}') + 1)])
    # 创建一个包含所有拉丁字母的集合，用于后续字符解析
    _latin_alphabets = set(string.ascii_letters)
    
    # 解析输入的数学表达式字符串 *s*，使用给定的字体对象 *fonts_object* 进行输出，
    # 在给定的 *fontsize* 和 *dpi* 下进行解析
    # 返回由 `Node` 实例组成的解析树
    def parse(self, s: str, fonts_object: Fonts, fontsize: float, dpi: float) -> Hlist:
        # 初始化状态栈，压入初始解析状态
        self._state_stack = [ParserState(fonts_object, 'default', 'rm', fontsize, dpi)]
        # 初始化空的 em 宽度缓存字典
        self._em_width_cache: dict[tuple[str, float, float], float] = {}
        try:
            # 尝试解析输入的表达式字符串 *s*
            result = self._expression.parseString(s)
        except ParseBaseException as err:
            # 如果解析失败，抛出带有详细解释的 ValueError 异常
            raise ValueError("\n" + ParseException.explain(err, 0)) from None
        finally:
            # 清空状态栈
            self._state_stack = []
        # 重置为非上标或下标状态
        self._in_subscript_or_superscript = False
        # 清空 em 宽度缓存
        self._em_width_cache = {}
        # 重置 pyparsing 的元素缓存
        ParserElement.resetCache()
        # 返回解析结果的第一个元素，预期是 Hlist 类型
        return T.cast(Hlist, result[0])  # Known return type from main.
    
    # 获取当前解析器的状态
    def get_state(self) -> ParserState:
        return self._state_stack[-1]
    
    # 弹出状态栈顶部的状态对象
    def pop_state(self) -> None:
        self._state_stack.pop()
    
    # 将当前状态复制并推入状态栈
    def push_state(self) -> None:
        self._state_stack.append(self.get_state().copy())
    
    # 将解析结果转换为单个 Hlist 对象的列表
    def main(self, toks: ParseResults) -> list[Hlist]:
        return [Hlist(toks.asList())]
    
    # 解析数学表达式中的字符串，确保完全解析，并返回 ParseResults 对象
    def math_string(self, toks: ParseResults) -> ParseResults:
        return self._math_expression.parseString(toks[0][1:-1], parseAll=True)
    
    # 处理数学模式下的解析结果，返回一个 Hlist 对象的列表，并弹出当前状态
    def math(self, toks: ParseResults) -> T.Any:
        hlist = Hlist(toks.asList())
        self.pop_state()
        return [hlist]
    
    # 处理非数学模式下的解析结果，将转义符号处理为实际字符，并返回一个 Hlist 对象的列表
    # 同时推入新的状态，将字体设置为数学文本默认字体
    def non_math(self, toks: ParseResults) -> T.Any:
        s = toks[0].replace(r'\$', '$')
        symbols = [Char(c, self.get_state()) for c in s]
        hlist = Hlist(symbols)
        self.push_state()
        self.get_state().font = mpl.rcParams['mathtext.default']
        return [hlist]
    
    # 静态方法，将输入的解析结果转换为浮点数字面量
    float_literal = staticmethod(pyparsing_common.convertToFloat)
    
    # 处理文本模式下的解析结果，将字符串转换为由 Char 对象组成的 Hlist 对象，并推出当前状态
    def text(self, toks: ParseResults) -> T.Any:
        self.push_state()
        state = self.get_state()
        state.font = 'rm'
        hlist = Hlist([Char(c, state) for c in toks[1]])
        self.pop_state()
        return [hlist]
    def _make_space(self, percentage: float) -> Kern:
        # 定义一个私有方法 _make_space，用于生成一个 Kern 对象，表示给定百分比的空白空间

        # 获取当前状态
        state = self.get_state()
        # 构建缓存键，包括字体、字号和 DPI
        key = (state.font, state.fontsize, state.dpi)
        # 从缓存中获取 em 的宽度
        width = self._em_width_cache.get(key)
        if width is None:
            # 如果缓存中没有，从字体集中获取 'it' 样式下 'm' 字符的度量信息
            metrics = state.fontset.get_metrics(
                'it', mpl.rcParams['mathtext.default'], 'm',
                state.fontsize, state.dpi)
            # 提取 advance 属性作为 em 宽度
            width = metrics.advance
            # 将 em 宽度存入缓存
            self._em_width_cache[key] = width
        # 返回 Kern 对象，表示计算得到的空白空间
        return Kern(width * percentage)

    _space_widths = {
        r'\,':         0.16667,   # 紧缩间距，相当于 3/18 em = 3 mu
        r'\thinspace': 0.16667,   # 细间距，相当于 3/18 em = 3 mu
        r'\/':         0.16667,   # 斜线间距，相当于 3/18 em = 3 mu
        r'\>':         0.22222,   # 大于间距，相当于 4/18 em = 4 mu
        r'\:':         0.22222,   # 冒号间距，相当于 4/18 em = 4 mu
        r'\;':         0.27778,   # 分号间距，相当于 5/18 em = 5 mu
        r'\ ':         0.33333,   # 标准间距，相当于 6/18 em = 6 mu
        r'~':          0.33333,   # 不换行空格，相当于 6/18 em = 6 mu
        r'\enspace':   0.5,       # 空格，相当于 9/18 em = 9 mu
        r'\quad':      1,         # quad 空格，相当于 1 em = 18 mu
        r'\qquad':     2,         # qquad 空格，相当于 2 em = 36 mu
        r'\!':         -0.16667,  # 负间距，相当于 -3/18 em = -3 mu
    }

    def space(self, toks: ParseResults) -> T.Any:
        # 定义 space 方法，接收 ParseResults 类型参数 toks，返回一个 Kern 对象列表
        num = self._space_widths[toks["space"]]
        # 获取指定空格命令对应的宽度百分比
        box = self._make_space(num)
        # 返回 Kern 对象的列表
        return [box]

    def customspace(self, toks: ParseResults) -> T.Any:
        # 定义 customspace 方法，接收 ParseResults 类型参数 toks，返回一个 Kern 对象列表
        return [self._make_space(toks["space"])]
    def symbol(self, s: str, loc: int,
               toks: ParseResults | dict[str, str]) -> T.Any:
        # 获取符号字符
        c = toks["sym"]
        # 如果符号是减号 "-", 替换为 Unicode 中的减号符号 "−"
        if c == "-":
            c = "\N{MINUS SIGN}"
        try:
            # 尝试创建符号对象 Char，并传入当前状态
            char = Char(c, self.get_state())
        except ValueError as err:
            # 如果出现值错误，抛出致命的解析异常，显示未知符号信息
            raise ParseFatalException(s, loc,
                                      "Unknown symbol: %s" % c) from err

        if c in self._spaced_symbols:
            # 在字符串中找到位于当前位置之前的前一个字符，以处理像 $=-2$, ${ -2}$, $ -2$, 或 $   -2$ 的情况
            prev_char = next((c for c in s[:loc][::-1] if c != ' '), '')
            # 对于二元操作符位于字符串开头或者在上标或下标中的情况，不添加空格
            if (self._in_subscript_or_superscript or (
                    c in self._binary_operators and (
                    len(s[:loc].split()) == 0 or prev_char in {
                        '{', *self._left_delims, *self._relation_symbols}))):
                return [char]
            else:
                # 对于其他情况，在符号前后添加空格，创建水平列表对象
                return [Hlist([self._make_space(0.2),
                               char,
                               self._make_space(0.2)],
                              do_kern=True)]
        elif c in self._punctuation_symbols:
            # 在字符串中找到位于当前位置之前和之后的非空格字符
            prev_char = next((c for c in s[:loc][::-1] if c != ' '), '')
            next_char = next((c for c in s[loc + 1:] if c != ' '), '')

            # 对于逗号 ",", 如果位于大括号 {} 内，则不添加空格
            if c == ',':
                if prev_char == '{' and next_char == '}':
                    return [char]

            # 对于小数点 ".", 如果前后字符都是数字，则不添加空格
            if c == '.' and prev_char.isdigit() and next_char.isdigit():
                return [char]
            else:
                # 对于其他情况，在符号后添加空格，创建水平列表对象
                return [Hlist([char, self._make_space(0.2)], do_kern=True)]
        # 返回符号对象的列表
        return [char]

    def unknown_symbol(self, s: str, loc: int, toks: ParseResults) -> T.Any:
        # 抛出致命的解析异常，显示未知符号信息
        raise ParseFatalException(s, loc, f"Unknown symbol: {toks['name']}")
    # 定义一个映射，将 LaTeX 中的重音名称映射到对应的组合字符命令
    _accent_map = {
        r'hat':            r'\circumflexaccent',
        r'breve':          r'\combiningbreve',
        r'bar':            r'\combiningoverline',
        r'grave':          r'\combininggraveaccent',
        r'acute':          r'\combiningacuteaccent',
        r'tilde':          r'\combiningtilde',
        r'dot':            r'\combiningdotabove',
        r'ddot':           r'\combiningdiaeresis',
        r'dddot':          r'\combiningthreedotsabove',
        r'ddddot':         r'\combiningfourdotsabove',
        r'vec':            r'\combiningrightarrowabove',
        r'"':              r'\combiningdiaeresis',
        r"`":              r'\combininggraveaccent',
        r"'":              r'\combiningacuteaccent',
        r'~':              r'\combiningtilde',
        r'.':              r'\combiningdotabove',
        r'^':              r'\circumflexaccent',
        r'overrightarrow': r'\rightarrow',
        r'overleftarrow':  r'\leftarrow',
        r'mathring':       r'\circ',
    }

    # 定义一组宽重音的集合，用于判断是否需要自动调整宽度
    _wide_accents = set(r"widehat widetilde widebar".split())

    # 处理重音命令的方法，返回一个节点对象
    def accent(self, toks: ParseResults) -> T.Any:
        # 获取当前状态
        state = self.get_state()
        # 获取当前下划线的厚度
        thickness = state.get_current_underline_thickness()
        # 获取重音类型和符号
        accent = toks["accent"]
        sym = toks["sym"]
        accent_box: Node
        
        # 根据重音类型选择不同的处理方式
        if accent in self._wide_accents:
            # 如果是宽重音，创建一个自动宽度字符对象
            accent_box = AutoWidthChar(
                '\\' + accent, sym.width, state, char_class=Accent)
        else:
            # 否则，创建一个普通的重音对象
            accent_box = Accent(self._accent_map[accent], state)
        
        # 如果重音是 'mathring'，进一步调整其尺寸
        if accent == 'mathring':
            accent_box.shrink()
            accent_box.shrink()
        
        # 创建一个水平居中的节点，将符号和重音放置其中
        centered = HCentered([Hbox(sym.width / 4.0), accent_box])
        # 将该节点水平打包成指定宽度
        centered.hpack(sym.width, 'exactly')
        
        # 返回一个竖直列表，包含居中的节点、上下两个厚度为 thickness 的盒子和符号的水平列表
        return Vlist([
                centered,
                Vbox(0., thickness * 2.0),
                Hlist([sym])
                ])

    # 处理函数名称的方法，返回一个水平列表对象
    def function(self, s: str, loc: int, toks: ParseResults) -> T.Any:
        # 调用 operatorname 方法获取水平列表对象
        hlist = self.operatorname(s, loc, toks)
        # 设置该水平列表对象的函数名称属性
        hlist.function_name = toks["name"]
        # 返回设置了函数名称属性的水平列表对象
        return hlist
    def operatorname(self, s: str, loc: int, toks: ParseResults) -> T.Any:
        # 将当前状态压栈，以便稍后恢复
        self.push_state()
        # 获取当前状态
        state = self.get_state()
        # 设置字体为罗马体
        state.font = 'rm'
        # 初始化空列表，用于存储节点
        hlist_list: list[Node] = []
        # 修改 Chars 的字体，但保留 Kerns 不变
        name = toks["name"]
        for c in name:
            if isinstance(c, Char):
                # 设置字符的字体为罗马体
                c.font = 'rm'
                # 更新字符的度量信息
                c._update_metrics()
                hlist_list.append(c)
            elif isinstance(c, str):
                # 将字符串转换为 Char 节点并添加到列表中
                hlist_list.append(Char(c, state))
            else:
                # 添加其他类型的节点到列表中
                hlist_list.append(c)
        # 计算下一个字符的位置
        next_char_loc = loc + len(name) + 1
        # 如果 name 是 ParseResults 对象，则增加相应的长度
        if isinstance(name, ParseResults):
            next_char_loc += len('operatorname{}')
        # 获取下一个非空格字符
        next_char = next((c for c in s[next_char_loc:] if c != ' '), '')
        # 定义分隔符集合
        delimiters = self._delims | {'^', '_'}
        # 如果下一个字符不是分隔符，并且 name 不在 _overunder_functions 中
        if (next_char not in delimiters and
                name not in self._overunder_functions):
            # 添加细空格，除非后面紧跟着括号等字符
            hlist_list += [self._make_space(self._space_widths[r'\,'])]
        # 恢复之前的状态
        self.pop_state()
        # 如果后面紧跟着上标或下标，设置标志为真
        # 此标志告诉 subsuper 在此运算符后添加空格
        if next_char in {'^', '_'}:
            self._in_subscript_or_superscript = True
        else:
            self._in_subscript_or_superscript = False

        # 返回由 hlist_list 组成的 Hlist 对象
        return Hlist(hlist_list)

    def start_group(self, toks: ParseResults) -> T.Any:
        # 将当前状态压栈
        self.push_state()
        # 处理 LaTeX 风格的字体标记
        if toks.get("font"):
            self.get_state().font = toks.get("font")
        # 返回空列表
        return []

    def group(self, toks: ParseResults) -> T.Any:
        # 创建包含给定组的 Hlist 对象，并以列表形式返回
        grp = Hlist(toks.get("group", []))
        return [grp]

    def required_group(self, toks: ParseResults) -> T.Any:
        # 返回包含给定组的 Hlist 对象
        return Hlist(toks.get("group", []))

    optional_group = required_group

    def end_group(self) -> T.Any:
        # 弹出当前状态
        self.pop_state()
        # 返回空列表
        return []

    def unclosed_group(self, s: str, loc: int, toks: ParseResults) -> T.Any:
        # 抛出解析严重错误异常，指示缺少 '}'
        raise ParseFatalException(s, len(s), "Expected '}'")

    def font(self, toks: ParseResults) -> T.Any:
        # 设置当前状态的字体
        self.get_state().font = toks["font"]
        # 返回空列表
        return []

    def is_overunder(self, nucleus: Node) -> bool:
        # 如果核心是 Char 类型且其字符在 _overunder_symbols 中，则返回 True
        if isinstance(nucleus, Char):
            return nucleus.c in self._overunder_symbols
        # 如果核心是 Hlist 类型且具有属性 'function_name'，且 'function_name' 在 _overunder_functions 中，则返回 True
        elif isinstance(nucleus, Hlist) and hasattr(nucleus, 'function_name'):
            return nucleus.function_name in self._overunder_functions
        # 否则返回 False
        return False

    def is_dropsub(self, nucleus: Node) -> bool:
        # 如果核心是 Char 类型且其字符在 _dropsub_symbols 中，则返回 True
        if isinstance(nucleus, Char):
            return nucleus.c in self._dropsub_symbols
        # 否则返回 False
        return False

    def is_slanted(self, nucleus: Node) -> bool:
        # 如果核心是 Char 类型且其字体是斜体，则返回 True
        if isinstance(nucleus, Char):
            return nucleus.is_slanted()
        # 否则返回 False
        return False
    def _genfrac(self, ldelim: str, rdelim: str, rule: float | None, style: _MathStyle,
                 num: Hlist, den: Hlist) -> T.Any:
        # 获取当前解析器的状态
        state = self.get_state()
        # 获取当前下划线的粗细
        thickness = state.get_current_underline_thickness()

        # 根据当前样式多次调整分子和分母的大小
        for _ in range(style.value):
            num.shrink()
            den.shrink()
        
        # 创建水平居中的分子和分母盒子
        cnum = HCentered([num])
        cden = HCentered([den])
        # 计算分子和分母盒子的最大宽度
        width = max(num.width, den.width)
        # 将分子和分母盒子调整为相同的宽度
        cnum.hpack(width, 'exactly')
        cden.hpack(width, 'exactly')
        
        # 创建垂直列表，包含分子、空白、水平线、空白、分母
        vlist = Vlist([cnum,                      # 分子
                       Vbox(0, thickness * 2.0),  # 空白
                       Hrule(state, rule),        # 水平线
                       Vbox(0, thickness * 2.0),  # 空白
                       cden                       # 分母
                       ])

        # 调整垂直列表的位置，使得分数线位于等号的中间
        metrics = state.fontset.get_metrics(
            state.font, mpl.rcParams['mathtext.default'],
            '=', state.fontsize, state.dpi)
        shift = (cden.height -
                 ((metrics.ymax + metrics.ymin) / 2 -
                  thickness * 3.0))
        vlist.shift_amount = shift

        # 将结果装入水平盒子并添加额外的空白
        result = [Hlist([vlist, Hbox(thickness * 2.)])]
        
        # 如果存在左右定界符，则自动调整定界符的大小
        if ldelim or rdelim:
            if ldelim == '':
                ldelim = '.'
            if rdelim == '':
                rdelim = '.'
            return self._auto_sized_delimiter(ldelim,
                                              T.cast(list[T.Union[Box, Char, str]],
                                                     result),
                                              rdelim)
        
        # 返回最终的结果
        return result

    def style_literal(self, toks: ParseResults) -> T.Any:
        # 将解析结果转换为数学样式对象并返回
        return self._MathStyle(int(toks["style_literal"]))

    def genfrac(self, toks: ParseResults) -> T.Any:
        # 调用_genfrac方法生成分数，并返回结果
        return self._genfrac(
            toks.get("ldelim", ""), toks.get("rdelim", ""),
            toks["rulesize"], toks.get("style", self._MathStyle.TEXTSTYLE),
            toks["num"], toks["den"])

    def frac(self, toks: ParseResults) -> T.Any:
        # 使用当前下划线粗细创建文本样式的分数并返回结果
        return self._genfrac(
            "", "", self.get_state().get_current_underline_thickness(),
            self._MathStyle.TEXTSTYLE, toks["num"], toks["den"])

    def dfrac(self, toks: ParseResults) -> T.Any:
        # 使用当前下划线粗细创建显示样式的分数并返回结果
        return self._genfrac(
            "", "", self.get_state().get_current_underline_thickness(),
            self._MathStyle.DISPLAYSTYLE, toks["num"], toks["den"])

    def binom(self, toks: ParseResults) -> T.Any:
        # 创建二项式，并使用默认的左右定界符
        return self._genfrac(
            "(", ")", 0,
            self._MathStyle.TEXTSTYLE, toks["num"], toks["den"])
    # 定义一个方法 `_genset`，接受参数 s（字符串）、loc（位置）、toks（解析结果）并返回泛型类型 T 的任意值
    def _genset(self, s: str, loc: int, toks: ParseResults) -> T.Any:
        # 从解析结果中获取注释对象
        annotation = toks["annotation"]
        # 从解析结果中获取正文对象
        body = toks["body"]
        # 获取当前下划线的厚度
        thickness = self.get_state().get_current_underline_thickness()

        # 缩小注释对象
        annotation.shrink()
        # 创建水平居中的注释对象
        centered_annotation = HCentered([annotation])
        # 创建水平居中的正文对象
        centered_body = HCentered([body])
        # 计算宽度为注释对象和正文对象宽度的最大值
        width = max(centered_annotation.width, centered_body.width)
        # 将注释对象和正文对象包装成指定宽度的盒子
        centered_annotation.hpack(width, 'exactly')
        centered_body.hpack(width, 'exactly')

        # 计算垂直间隔为当前下划线厚度的三倍
        vgap = thickness * 3
        # 判断符号是否为 \underset
        if s[loc + 1] == "u":  # \underset
            # 创建垂直列表，顺序为正文对象、垂直间隔的盒子、注释对象
            vlist = Vlist([
                centered_body,               # 正文对象
                Vbox(0, vgap),               # 空白间隔
                centered_annotation          # 注释对象
            ])
            # 调整使正文对象与注释对象在垂直方向上对齐
            vlist.shift_amount = centered_body.depth + centered_annotation.height + vgap
        else:  # \overset
            # 创建垂直列表，顺序为注释对象、垂直间隔的盒子、正文对象
            vlist = Vlist([
                centered_annotation,         # 注释对象
                Vbox(0, vgap),               # 空白间隔
                centered_body                # 正文对象
            ])

        # 为了在符号之间添加水平间隔：将垂直列表包装在水平列表中，并扩展为 Hbox(0, horizontal_gap)
        # 返回处理后的垂直列表对象
        return vlist

    # 将 _genset 方法赋值给 overset 和 underset，使它们成为 _genset 方法的别名
    overset = underset = _genset
    # 定义一个方法用于计算平方根符号的呈现
    def sqrt(self, toks: ParseResults) -> T.Any:
        # 从解析结果中获取根部对象和值对象
        root = toks.get("root")
        body = toks["value"]
        # 获取当前状态和下划线的厚度
        state = self.get_state()
        thickness = state.get_current_underline_thickness()

        # 计算主体的高度，加上一点额外的空间，避免显得太拥挤
        height = body.height - body.shift_amount + thickness * 5.0
        depth = body.depth + body.shift_amount
        # 创建一个自动调整高度的字符对象，用于根号的绘制
        check = AutoHeightChar(r'\__sqrt__', height, depth, state, always=True)
        height = check.height - check.shift_amount
        depth = check.depth + check.shift_amount

        # 在主体的左右各添加一点额外空间
        padded_body = Hlist([Hbox(2 * thickness), body, Hbox(2 * thickness)])
        rightside = Vlist([Hrule(state), Glue('fill'), padded_body])
        # 伸展在水平线和主体之间的粘合剂
        rightside.vpack(height + (state.fontsize * state.dpi) / (100.0 * 12.0),
                        'exactly', depth)

        # 添加根号并将其向上移动，使其位于水平线上方
        # 数值 0.6 是一个硬编码的调整值 ;)
        if not root:
            root = Box(check.width * 0.5, 0., 0.)
        else:
            root = Hlist(root)
            root.shrink()
            root.shrink()

        root_vlist = Vlist([Hlist([root])])
        root_vlist.shift_amount = -height * 0.6

        # 构建水平列表，包含根号、负面空隙（将根号放置在水平线之上）、根号本身、主体
        hlist = Hlist([root_vlist,               # 根号
                       Kern(-check.width * 0.5),
                       check,                    # 根号符号
                       rightside])               # 主体
        return [hlist]

    # 定义一个方法用于绘制上划线
    def overline(self, toks: ParseResults) -> T.Any:
        # 从解析结果中获取主体对象
        body = toks["body"]

        state = self.get_state()
        thickness = state.get_current_underline_thickness()

        # 计算主体的高度和深度
        height = body.height - body.shift_amount + thickness * 3.0
        depth = body.depth + body.shift_amount

        # 将上划线放置在主体的上方
        rightside = Vlist([Hrule(state), Glue('fill'), Hlist([body])])

        # 伸展在水平线和主体之间的粘合剂
        rightside.vpack(height + (state.fontsize * state.dpi) / (100.0 * 12.0),
                        'exactly', depth)

        # 构建水平列表，包含主体和上划线
        hlist = Hlist([rightside])
        return [hlist]
    # 定义一个方法 _auto_sized_delimiter，接受前缀字符串 front，中部列表 middle，后缀字符串 back，并返回任意类型
    def _auto_sized_delimiter(self, front: str,
                              middle: list[Box | Char | str],
                              back: str) -> T.Any:
        # 获取当前对象的状态
        state = self.get_state()
        
        # 如果 middle 列表不为空
        if len(middle):
            # 计算中部元素的最大高度和深度
            height = max([x.height for x in middle if not isinstance(x, str)])
            depth = max([x.depth for x in middle if not isinstance(x, str)])
            factor = None
            
            # 遍历 middle 列表的元素
            for idx, el in enumerate(middle):
                # 如果当前元素是 \middle
                if el == r'\middle':
                    # 获取下一个元素，应该是 p.delims 中的一个
                    c = T.cast(str, middle[idx + 1])
                    # 如果 c 不是 '.'，则创建一个 AutoHeightChar 对象
                    if c != '.':
                        middle[idx + 1] = AutoHeightChar(
                                c, height, depth, state, factor=factor)
                    else:
                        # 否则移除当前元素 c
                        middle.remove(c)
                    # 移除当前元素 \middle
                    del middle[idx]
            
            # 剩下的 middle 元素应该只有 \middle 和作为字符串的分隔符，它们已经被移除
            middle_part = T.cast(list[T.Union[Box, Char]], middle)
        else:
            # 如果 middle 列表为空，则设置高度和深度为 0，因子为 1.0，中部为空列表
            height = 0
            depth = 0
            factor = 1.0
            middle_part = []
        
        # 创建一个空列表 parts，用来存放结果
        parts: list[Node] = []
        
        # 如果 front 不是 '.'，则将其作为 AutoHeightChar 添加到 parts 中
        if front != '.':
            parts.append(
                AutoHeightChar(front, height, depth, state, factor=factor))
        
        # 将 middle_part 中的元素添加到 parts 中
        parts.extend(middle_part)
        
        # 如果 back 不是 '.'，则将其作为 AutoHeightChar 添加到 parts 中
        if back != '.':
            parts.append(
                AutoHeightChar(back, height, depth, state, factor=factor))
        
        # 创建一个 Hlist 对象，将 parts 作为参数
        hlist = Hlist(parts)
        
        # 返回 hlist 对象作为结果
        return hlist

    # 定义一个方法 auto_delim，接受一个 ParseResults 对象 toks，并返回任意类型
    def auto_delim(self, toks: ParseResults) -> T.Any:
        # 调用 _auto_sized_delimiter 方法，传入左边界 toks["left"]、中部 toks["mid"]（如果存在），右边界 toks["right"]
        return self._auto_sized_delimiter(
            toks["left"],
            # 如果 toks 中包含 "mid" 键，将其作为列表返回；在需要 pyparsing 3 时，可以移除此条件
            toks["mid"].asList() if "mid" in toks else [],
            toks["right"])

    # 定义一个方法 boldsymbol，接受一个 ParseResults 对象 toks，并返回任意类型
    def boldsymbol(self, toks: ParseResults) -> T.Any:
        # 压入当前状态
        self.push_state()
        
        # 获取当前对象的状态
        state = self.get_state()
        
        # 创建一个空列表 hlist，用来存放结果
        hlist: list[Node] = []
        
        # 获取名为 "value" 的 toks 对象
        name = toks["value"]
        
        # 遍历名为 "value" 的 toks 对象中的每个元素 c
        for c in name:
            # 如果 c 是 Hlist 类型的实例
            if isinstance(c, Hlist):
                # 获取 c 的 children 列表中的第二个元素
                k = c.children[1]
                # 如果 k 是 Char 类型的实例
                if isinstance(k, Char):
                    # 将 k 的字体设置为 "bf"（粗体）
                    k.font = "bf"
                    # 更新 k 的度量信息
                    k._update_metrics()
                # 将 c 添加到 hlist 中
                hlist.append(c)
            
            # 如果 c 是 Char 类型的实例
            elif isinstance(c, Char):
                # 将 c 的字体设置为 "bf"（粗体）
                c.font = "bf"
                # 如果 c 是拉丁字母或小希腊字母的一部分
                if (c.c in self._latin_alphabets or
                   c.c[1:] in self._small_greek):
                    # 将 c 的字体设置为 "bfit"（粗斜体）
                    c.font = "bfit"
                    # 更新 c 的度量信息
                    c._update_metrics()
                # 更新 c 的度量信息
                c._update_metrics()
                # 将 c 添加到 hlist 中
                hlist.append(c)
            
            # 否则将 c 添加到 hlist 中
            else:
                hlist.append(c)
        
        # 弹出之前压入的状态
        self.pop_state()
        
        # 返回一个 Hlist 对象，将 hlist 作为参数
        return Hlist(hlist)
    # 定义一个方法用于处理子堆栈，接收解析结果作为参数，并返回任意类型的结果
    def substack(self, toks: ParseResults) -> T.Any:
        # 从解析结果中获取 "parts" 键对应的值
        parts = toks["parts"]
        # 获取当前状态
        state = self.get_state()
        # 获取当前下划线的厚度
        thickness = state.get_current_underline_thickness()

        # 使用列表推导式创建一个由 Hlist 对象组成的列表
        hlist = [Hlist(k) for k in parts[0]]
        # 计算 hlist 中最宽的元素的宽度
        max_width = max(map(lambda c: c.width, hlist))

        # 创建一个空列表用于存储垂直堆叠的对象
        vlist = []
        # 遍历 hlist 中的每个子对象
        for sub in hlist:
            # 创建一个水平居中的盒子，包含当前子对象
            cp = HCentered([sub])
            # 将该盒子调整为指定的最大宽度
            cp.hpack(max_width, 'exactly')
            # 将调整后的盒子添加到 vlist 中
            vlist.append(cp)

        # 使用列表推导式创建一个由交替的 vlist 和指定高度的 Vbox 对象组成的列表，形成堆栈
        stack = [val
                 for pair in zip(vlist, [Vbox(0, thickness * 2)] * len(vlist))
                 for val in pair]
        # 删除堆栈中最后一个元素
        del stack[-1]
        # 将 stack 转换为一个垂直列表对象
        vlt = Vlist(stack)
        # 将 vlt 放入水平列表中，并将结果作为最终结果返回
        result = [Hlist([vlt])]
        return result
```