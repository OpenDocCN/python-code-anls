# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_pdf.py`

```py
"""
PDF Matplotlib 后端。

作者：Jouni K Seppänen <jks@iki.fi> 及其他人。

该模块负责生成 PDF 格式的图形输出。
"""

# 导入所需的模块和库
import codecs  # 提供编解码器注册和流接口
from datetime import timezone, datetime  # 处理日期时间相关功能
from enum import Enum  # 支持枚举类型
from functools import total_ordering  # 提供了类装饰器，简化了类的比较运算
from io import BytesIO  # 提供了用于处理二进制数据的流
import itertools  # 提供了用于创建迭代器的函数
import logging  # 提供了灵活的日志记录功能
import math  # 数学函数库
import os  # 提供了与操作系统交互的功能
import string  # 提供了一些常用的字符串操作
import struct  # 提供了 Python 值和 C 结构体的转换
import sys  # 提供了对 Python 解释器的访问
import time  # 提供了时间处理的函数
import types  # 提供了操作类型和类型对象的功能
import warnings  # 提供了警告控制的功能
import zlib  # 提供了压缩功能

# 导入第三方库
import numpy as np  # 处理数组和矩阵的数值运算
from PIL import Image  # Python Imaging Library，用于图像处理

import matplotlib as mpl  # Matplotlib 库的核心模块
from matplotlib import (
    _api, _text_helpers, _type1font, cbook, dviread)  # Matplotlib 内部模块
from matplotlib._pylab_helpers import Gcf  # Matplotlib 的 PyLab 辅助类
from matplotlib.backend_bases import (  # Matplotlib 后端基础类
    _Backend, FigureCanvasBase, FigureManagerBase, GraphicsContextBase,
    RendererBase)
from matplotlib.backends.backend_mixed import MixedModeRenderer  # 混合模式渲染器
from matplotlib.figure import Figure  # 图形图像的基础类
from matplotlib.font_manager import (  # 字体管理器，用于获取字体信息
    get_font, fontManager as _fontManager)
from matplotlib._afm import AFM  # Adobe 字体度量
from matplotlib.ft2font import (  # FreeType 2 字体渲染
    FIXED_WIDTH, ITALIC, LOAD_NO_SCALE, LOAD_NO_HINTING, KERNING_UNFITTED, FT2Font)
from matplotlib.transforms import Affine2D, BboxBase  # 坐标变换和包围框基类
from matplotlib.path import Path  # 定义和操作路径对象
from matplotlib.dates import UTC  # 处理日期和时间的类和函数
from matplotlib import _path  # 路径处理模块
from . import _backend_pdf_ps  # 导入本地模块 _backend_pdf_ps

_log = logging.getLogger(__name__)  # 获取本模块的日志记录器

# 概述
#
# 对 PDF 语法的低级了解主要在 pdfRepr 函数及其它几个类中体现，
# 包括 Reference、Name、Operator 和 Stream。PdfFile 类了解 PDF 文档的总体结构，
# 它提供了用于在文件中写入任意字符串的 "write" 方法，以及在写入对象之前通过 pdfRepr 函数传递对象的 "output" 方法。
# RendererPdf 类调用 output 方法，该类包含各种 draw_foo 方法用于绘制不同的图形元素。
# 每个 draw_foo 方法在输出命令之前调用 self.check_gc，该方法检查是否需要修改 PDF 图形状态，并输出必要的命令。
# GraphicsContextPdf 表示 PDF 图形状态，其 "delta" 方法返回修改状态的命令。

# 在配置文件中添加 "pdf.use14corefonts: True" 可以仅使用 14 种 PDF 核心字体。
# 这些字体无需嵌入，每个 PDF 查看应用程序都必须包含它们。这样可以生成非常轻量级的 PDF 文件，
# 可以直接在使用 pdfTeX 生成的 LaTeX 或 ConTeXt 文档中使用，而无需进行任何转换。

# 这些字体包括：Helvetica、Helvetica-Bold、Helvetica-Oblique、
# Helvetica-BoldOblique、Courier、Courier-Bold、Courier-Oblique、
# Courier-BoldOblique、Times-Roman、Times-Bold、Times-Italic、
# Times-BoldItalic、Symbol、ZapfDingbats。

# 一些注意点：
#
# 1. 裁剪路径只能通过从状态堆栈中弹出来扩展。因此，在缩小裁剪路径之前必须将状态推送到堆栈中。
#    GraphicsContextPdf 类负责处理这一点。
#
# 2. 有时需要引用某些东西（如字体、颜色等），这些引用通常使用名称或者对象引用。
#    这些细节在代码中通过一些特定的类和函数来处理。
# image, or extended graphics state, which contains the alpha value)
# in the page stream by a name that needs to be defined outside the
# stream.  PdfFile provides the methods fontName, imageObject, and
# alphaState for this purpose.  The implementations of these methods
# should perhaps be generalized.

# TODOs:
#
# * encoding of fonts, including mathtext fonts and Unicode support
# * TTF support has lots of small TODOs, e.g., how do you know if a font
#   is serif/sans-serif, or symbolic/non-symbolic?
# * draw_quad_mesh


def _fill(strings, linelen=75):
    """
    Make one string from sequence of strings, with whitespace in between.

    The whitespace is chosen to form lines of at most *linelen* characters,
    if possible.
    """
    currpos = 0  # Initialize the current position in the output string
    lasti = 0  # Initialize the index for the last processed string segment
    result = []  # Initialize an empty list to store the resulting lines of text
    for i, s in enumerate(strings):
        length = len(s)  # Calculate the length of the current string segment
        if currpos + length < linelen:
            currpos += length + 1  # Update current position with added segment length and whitespace
        else:
            result.append(b' '.join(strings[lasti:i]))  # Join strings into a line and add to result
            lasti = i  # Update last processed index
            currpos = length  # Reset current position to current segment length
    result.append(b' '.join(strings[lasti:]))  # Join remaining strings into a final line
    return b'\n'.join(result)  # Join all lines with newline characters and return as bytes


def _create_pdf_info_dict(backend, metadata):
    """
    Create a PDF infoDict based on user-supplied metadata.

    A default ``Creator``, ``Producer``, and ``CreationDate`` are added, though
    the user metadata may override it. The date may be the current time, or a
    time set by the ``SOURCE_DATE_EPOCH`` environment variable.

    Metadata is verified to have the correct keys and their expected types. Any
    unknown keys/types will raise a warning.

    Parameters
    ----------
    backend : str
        The name of the backend to use in the Producer value.

    metadata : dict[str, Union[str, datetime, Name]]
        A dictionary of metadata supplied by the user with information
        following the PDF specification, also defined in
        `~.backend_pdf.PdfPages` below.

        If any value is *None*, then the key will be removed. This can be used
        to remove any pre-defined values.

    Returns
    -------
    dict[str, Union[str, datetime, Name]]
        A validated dictionary of metadata.
    """

    # get source date from SOURCE_DATE_EPOCH, if set
    # See https://reproducible-builds.org/specs/source-date-epoch/
    source_date_epoch = os.getenv("SOURCE_DATE_EPOCH")
    if source_date_epoch:
        source_date = datetime.fromtimestamp(int(source_date_epoch), timezone.utc)
        source_date = source_date.replace(tzinfo=UTC)
    else:
        source_date = datetime.today()

    info = {
        'Creator': f'Matplotlib v{mpl.__version__}, https://matplotlib.org',
        'Producer': f'Matplotlib {backend} backend v{mpl.__version__}',
        'CreationDate': source_date,
        **metadata
    }
    info = {k: v for (k, v) in info.items() if v is not None}  # Filter out None values from info dictionary

    def is_string_like(x):
        return isinstance(x, str)
    is_string_like.text_for_warning = "an instance of str"
    def is_date(x):
        # 检查参数 x 是否是 datetime 类型的实例
        return isinstance(x, datetime)
    # 为函数 is_date 添加一个属性，用于警告文本
    is_date.text_for_warning = "an instance of datetime.datetime"

    def check_trapped(x):
        # 检查参数 x 是否是 Name 类型的实例，并且其名称在 (b'True', b'False', b'Unknown') 中
        if isinstance(x, Name):
            return x.name in (b'True', b'False', b'Unknown')
        else:
            # 否则检查 x 是否在 ('True', 'False', 'Unknown') 中
            return x in ('True', 'False', 'Unknown')
    # 为函数 check_trapped 添加一个属性，用于警告文本
    check_trapped.text_for_warning = 'one of {"True", "False", "Unknown"}'

    # 定义关键字及其对应的检查函数的映射关系
    keywords = {
        'Title': is_string_like,
        'Author': is_string_like,
        'Subject': is_string_like,
        'Keywords': is_string_like,
        'Creator': is_string_like,
        'Producer': is_string_like,
        'CreationDate': is_date,
        'ModDate': is_date,
        'Trapped': check_trapped,
    }
    # 遍历传入的信息字典 info
    for k in info:
        # 如果 info 中的键 k 不在 keywords 中
        if k not in keywords:
            # 发出外部警告，指出未知的 infodict 关键字
            _api.warn_external(f'Unknown infodict keyword: {k!r}. '
                               f'Must be one of {set(keywords)!r}.')
        # 否则，如果 keywords 中定义的检查函数不通过 info[k] 的值
        elif not keywords[k](info[k]):
            # 发出外部警告，指出 infodict 关键字 k 的值不合法
            _api.warn_external(f'Bad value for infodict keyword {k}. '
                               f'Got {info[k]!r} which is not '
                               f'{keywords[k].text_for_warning}.')
    # 如果 info 中有 'Trapped' 键
    if 'Trapped' in info:
        # 将 info['Trapped'] 转换为 Name 对象
        info['Trapped'] = Name(info['Trapped'])

    # 返回处理后的 info 字典
    return info
# 将给定的 datetime 对象转换为符合 PDF 规范的字符串表示形式
def _datetime_to_pdf(d):
    r = d.strftime('D:%Y%m%d%H%M%S')  # 格式化日期时间为字符串
    z = d.utcoffset()  # 获取 UTC 偏移量
    if z is not None:
        z = z.seconds  # 如果存在偏移量，获取偏移秒数
    else:
        if time.daylight:
            z = time.altzone  # 如果没有偏移量，根据是否为夏令时获取时区偏移
        else:
            z = time.timezone
    if z == 0:
        r += 'Z'  # 如果偏移为零，添加 'Z' 表示零时区
    elif z < 0:
        r += "+%02d'%02d'" % ((-z) // 3600, (-z) % 3600)  # 处理负偏移情况
    else:
        r += "-%02d'%02d'" % (z // 3600, z % 3600)  # 处理正偏移情况
    return r  # 返回符合 PDF 规范的日期时间字符串


# 计算经过指定角度旋转后的矩形的顶点坐标
def _calculate_quad_point_coordinates(x, y, width, height, angle=0):
    angle = math.radians(-angle)  # 将角度转换为弧度，并取其负数
    sin_angle = math.sin(angle)  # 计算正弦值
    cos_angle = math.cos(angle)  # 计算余弦值
    a = x + height * sin_angle  # 计算第一个顶点的坐标
    b = y + height * cos_angle  # 计算第二个顶点的坐标
    c = x + width * cos_angle + height * sin_angle  # 计算第三个顶点的坐标
    d = y - width * sin_angle + height * cos_angle  # 计算第四个顶点的坐标
    e = x + width * cos_angle  # 计算水平宽度的端点坐标
    f = y - width * sin_angle  # 计算垂直高度的端点坐标
    return ((x, y), (e, f), (c, d), (a, b))  # 返回旋转后矩形的四个顶点坐标


# 获取旋转后矩形的顶点坐标和覆盖旋转矩形的矩形坐标
def _get_coordinates_of_block(x, y, width, height, angle=0):
    vertices = _calculate_quad_point_coordinates(x, y, width,
                                                 height, angle)  # 获取旋转后矩形的顶点坐标
    pad = 0.00001 if angle % 90 else 0  # 如果角度为 90 的倍数，则添加微小的偏移量
    min_x = min(v[0] for v in vertices) - pad  # 计算最小 X 坐标
    min_y = min(v[1] for v in vertices) - pad  # 计算最小 Y 坐标
    max_x = max(v[0] for v in vertices) + pad  # 计算最大 X 坐标
    max_y = max(v[1] for v in vertices) + pad  # 计算最大 Y 坐标
    return (tuple(itertools.chain.from_iterable(vertices)),  # 返回顶点坐标的扁平化列表和覆盖矩形的坐标
            (min_x, min_y, max_x, max_y))


# 创建一个用于嵌入 URL 的链接注释对象
def _get_link_annotation(gc, x, y, width, height, angle=0):
    quadpoints, rect = _get_coordinates_of_block(x, y, width, height, angle)  # 获取旋转矩形的顶点坐标和覆盖矩形的坐标
    link_annotation = {
        'Type': Name('Annot'),  # 注释类型
        'Subtype': Name('Link'),  # 注释子类型
        'Rect': rect,  # 注释所占用的矩形范围
        'Border': [0, 0, 0],  # 注释边框设置为无
        'A': {
            'S': Name('URI'),  # 操作类型为 URI
            'URI': gc.get_url(),  # 获取注释的 URL
        },
    }
    if angle % 90:
        link_annotation['QuadPoints'] = quadpoints  # 如果角度不是 90 的倍数，则添加 QuadPoints 属性
    return link_annotation  # 返回链接注释对象


# PDF 字符串可以包含除了不平衡的括号和反斜杠外的任何八位数据，需要对特定字符进行转义
_str_escapes = str.maketrans({
    '\\': '\\\\',  # 反斜杠转义为双反斜杠
    '(': '\\(',  # 左括号转义为反斜杠左括号
    ')': '\\)',  # 右括号转义为反斜杠右括号
    '\n': '\\n',  # 换行符转义为反斜杠 n
    '\r': '\\r'   # 回车符转义为反斜杠 r
})


# 定义一个函数，用于生成 PDF 的表示对象
def pdfRepr(obj):
    """Map Python objects to PDF syntax."""

    # 检查对象是否有 pdfRepr 方法，如果有则调用该方法返回其 PDF 表示形式
    if hasattr(obj, 'pdfRepr'):
        return obj.pdfRepr()

    # 处理浮点数。PDF 不支持科学计数法（如1.0e-10），需要使用 %f 来表示，精度为小数点后十位。
    # 如果数值不是有限的，则抛出异常。
    elif isinstance(obj, (float, np.floating)):
        if not np.isfinite(obj):
            raise ValueError("Can only output finite numbers in PDF")
        r = b"%.10f" % obj
        return r.rstrip(b'0').rstrip(b'.')

    # 处理布尔值。需先判断是否为布尔类型，因为 isinstance(True, int) 结果也为真。
    elif isinstance(obj, bool):
        return [b'false', b'true'][obj]

    # 处理整数，直接输出为字节表示的整数。
    elif isinstance(obj, (int, np.integer)):
        return b"%d" % obj

    # 处理非 ASCII Unicode 字符串。如果是 ASCII 字符串，则转换为 ASCII 编码；否则使用 UTF-16BE 编码，并在开头加上字节顺序标记。
    elif isinstance(obj, str):
        return pdfRepr(obj.encode('ascii') if obj.isascii()
                       else codecs.BOM_UTF16_BE + obj.encode('UTF-16BE'))

    # 处理字节串。将字节串用括号括起来，同时处理其中的反斜杠和括号进行转义。
    elif isinstance(obj, bytes):
        return (
            b'(' +
            obj.decode('latin-1').translate(_str_escapes).encode('latin-1')
            + b')')

    # 处理字典。字典的键必须是 PDF 名称对象（Name），如果键是字符串，则将其转换为 PDF 名称对象；值则递归调用 pdfRepr 函数处理。
    elif isinstance(obj, dict):
        return _fill([
            b"<<",
            *[Name(k).pdfRepr() + b" " + pdfRepr(v) for k, v in obj.items()],
            b">>",
        ])

    # 处理列表或元组。
    elif isinstance(obj, (list, tuple)):
        return _fill([b"[", *[pdfRepr(val) for val in obj], b"]"])

    # 处理空值（null 关键字）。
    elif obj is None:
        return b'null'

    # 处理日期对象。
    elif isinstance(obj, datetime):
        return pdfRepr(_datetime_to_pdf(obj))

    # 处理边界框对象。
    elif isinstance(obj, BboxBase):
        return _fill([pdfRepr(val) for val in obj.bounds])

    # 未知类型的对象，抛出类型错误异常。
    else:
        raise TypeError(f"Don't know a PDF representation for {type(obj)} "
                        "objects")
def _font_supports_glyph(fonttype, glyph):
    """
    Returns True if the font is able to provide codepoint *glyph* in a PDF.

    For a Type 3 font, this method returns True only for single-byte
    characters. For Type 42 fonts this method return True if the character is
    from the Basic Multilingual Plane.
    """
    # 如果字体类型为 Type 3，则返回 True，仅当字符编码小于等于 255 时
    if fonttype == 3:
        return glyph <= 255
    # 如果字体类型为 Type 42，则返回 True，仅当字符编码小于等于 65535 时
    if fonttype == 42:
        return glyph <= 65535
    # 抛出未实现错误，表明不支持其他类型的字体
    raise NotImplementedError()


class Reference:
    """
    PDF reference object.

    Use PdfFile.reserveObject() to create References.
    """

    def __init__(self, id):
        # 初始化方法，设置引用对象的 ID
        self.id = id

    def __repr__(self):
        # 返回对象的字符串表示形式，包含引用对象的 ID
        return "<Reference %d>" % self.id

    def pdfRepr(self):
        # 返回 PDF 格式的表示，包含引用对象的 ID
        return b"%d 0 R" % self.id

    def write(self, contents, file):
        # 将引用对象写入文件
        write = file.write
        write(b"%d 0 obj\n" % self.id)  # 写入对象头部
        write(pdfRepr(contents))  # 写入对象内容
        write(b"\nendobj\n")  # 写入对象尾部


@total_ordering
class Name:
    """PDF name object."""
    __slots__ = ('name',)
    _hexify = {c: '#%02x' % c
               for c in {*range(256)} - {*range(ord('!'), ord('~') + 1)}}

    def __init__(self, name):
        # 初始化方法，根据输入的名称创建 Name 对象
        if isinstance(name, Name):
            self.name = name.name
        else:
            if isinstance(name, bytes):
                name = name.decode('ascii')
            # 对输入的名称进行 ASCII 编码处理并转换为字节格式
            self.name = name.translate(self._hexify).encode('ascii')

    def __repr__(self):
        # 返回对象的字符串表示形式，包含名称对象的名称
        return "<Name %s>" % self.name

    def __str__(self):
        # 返回对象的字符串表示形式，包含名称对象的名称
        return '/' + self.name.decode('ascii')

    def __eq__(self, other):
        # 比较方法，判断两个名称对象是否相等
        return isinstance(other, Name) and self.name == other.name

    def __lt__(self, other):
        # 比较方法，判断当前名称对象是否小于另一个名称对象
        return isinstance(other, Name) and self.name < other.name

    def __hash__(self):
        # 返回对象的哈希值，用于集合等数据结构中
        return hash(self.name)

    def pdfRepr(self):
        # 返回 PDF 格式的名称表示
        return b'/' + self.name


class Verbatim:
    """Store verbatim PDF command content for later inclusion in the stream."""
    def __init__(self, x):
        # 初始化方法，存储待包含到流中的 PDF 命令内容
        self._x = x

    def pdfRepr(self):
        # 返回 PDF 格式的命令内容
        return self._x


class Op(Enum):
    """PDF operators (not an exhaustive list)."""

    close_fill_stroke = b'b'
    fill_stroke = b'B'
    fill = b'f'
    closepath = b'h'
    close_stroke = b's'
    stroke = b'S'
    endpath = b'n'
    begin_text = b'BT'
    end_text = b'ET'
    curveto = b'c'
    rectangle = b're'
    lineto = b'l'
    moveto = b'm'
    concat_matrix = b'cm'
    use_xobject = b'Do'
    setgray_stroke = b'G'
    setgray_nonstroke = b'g'
    setrgb_stroke = b'RG'
    setrgb_nonstroke = b'rg'
    setcolorspace_stroke = b'CS'
    setcolorspace_nonstroke = b'cs'
    setcolor_stroke = b'SCN'
    setcolor_nonstroke = b'scn'
    setdash = b'd'
    setlinejoin = b'j'
    setlinecap = b'J'
    setgstate = b'gs'
    gsave = b'q'
    grestore = b'Q'
    textpos = b'Td'
    selectfont = b'Tf'
    textmatrix = b'Tm'
    show = b'Tj'
    showkern = b'TJ'
    setlinewidth = b'w'
    clip = b'W'
    shading = b'sh'

    def pdfRepr(self):
        # 返回 PDF 格式的操作符表示
        return self.value

    @classmethod
    # 定义一个类方法，用于生成 PDF 操作符来绘制路径

    Parameters
    ----------
    fill : bool
        是否使用填充颜色填充路径
    stroke : bool
        是否使用线条颜色描边路径

    Returns
    -------
    function
        返回对应的 PDF 操作符函数

    # 根据参数决定返回的 PDF 操作符函数
    if stroke:
        if fill:
            # 如果同时填充和描边路径，则返回填充并描边的操作符函数
            return cls.fill_stroke
        else:
            # 如果只描边路径，则返回描边的操作符函数
            return cls.stroke
    else:
        if fill:
            # 如果只填充路径，则返回填充的操作符函数
            return cls.fill
        else:
            # 如果既不填充也不描边，则返回结束路径的操作符函数
            return cls.endpath
class Stream:
    """
    PDF stream object.

    This has no pdfRepr method. Instead, call begin(), then output the
    contents of the stream by calling write(), and finally call end().
    """
    __slots__ = ('id', 'len', 'pdfFile', 'file', 'compressobj', 'extra', 'pos')

    def __init__(self, id, len, file, extra=None, png=None):
        """
        Parameters
        ----------
        id : int
            Object id of the stream.
        len : Reference or None
            An unused Reference object for the length of the stream;
            None means to use a memory buffer so the length can be inlined.
        file : PdfFile
            The underlying object to write the stream to.
        extra : dict from Name to anything, or None
            Extra key-value pairs to include in the stream header.
        png : dict or None
            If the data is already png encoded, the decode parameters.
        """
        self.id = id            # object id，对象的标识号
        self.len = len          # id of length object，长度对象的标识号
        self.pdfFile = file     # PdfFile 对象，用于写入流的底层对象
        self.file = file.fh     # file to which the stream is written，流写入的文件对象
        self.compressobj = None # compression object，压缩对象
        if extra is None:
            self.extra = dict()
        else:
            self.extra = extra.copy()  # 深拷贝额外的头部信息
        if png is not None:
            self.extra.update({'Filter':      Name('FlateDecode'),
                               'DecodeParms': png})

        self.pdfFile.recordXref(self.id)  # 记录对象的交叉引用
        if mpl.rcParams['pdf.compression'] and not png:
            self.compressobj = zlib.compressobj(
                mpl.rcParams['pdf.compression'])  # 如果启用压缩，创建压缩对象
        if self.len is None:
            self.file = BytesIO()  # 如果长度为空，使用内存缓冲区
        else:
            self._writeHeader()  # 否则，写入头部信息并记录当前位置
            self.pos = self.file.tell()

    def _writeHeader(self):
        write = self.file.write
        write(b"%d 0 obj\n" % self.id)  # 写入对象编号
        dict = self.extra
        dict['Length'] = self.len
        if mpl.rcParams['pdf.compression']:
            dict['Filter'] = Name('FlateDecode')  # 如果启用压缩，设置过滤器为 FlateDecode

        write(pdfRepr(dict))  # 写入头部信息的 PDF 表示形式
        write(b"\nstream\n")  # 写入流开始标记

    def end(self):
        """Finalize stream."""
        self._flush()  # 刷新缓冲区
        if self.len is None:
            contents = self.file.getvalue()  # 获取缓冲区内容
            self.len = len(contents)  # 更新长度
            self.file = self.pdfFile.fh  # 恢复到原始文件对象
            self._writeHeader()  # 重新写入头部信息
            self.file.write(contents)  # 写入内容
            self.file.write(b"\nendstream\nendobj\n")  # 写入流结束标记和对象结束标记
        else:
            length = self.file.tell() - self.pos  # 计算长度
            self.file.write(b"\nendstream\nendobj\n")  # 写入流结束标记和对象结束标记
            self.pdfFile.writeObject(self.len, length)  # 写入对象的长度信息

    def write(self, data):
        """Write some data on the stream."""
        if self.compressobj is None:
            self.file.write(data)  # 如果没有压缩对象，直接写入数据
        else:
            compressed = self.compressobj.compress(data)  # 否则，压缩数据并写入
            self.file.write(compressed)  # 写入压缩后的数据
    def _flush(self):
        """Flush the compression object."""
        
        # 检查压缩对象是否存在
        if self.compressobj is not None:
            # 刷新压缩对象，获取压缩后的数据
            compressed = self.compressobj.flush()
            # 将压缩后的数据写入文件
            self.file.write(compressed)
            # 清空压缩对象，以便下次使用
            self.compressobj = None
# 定义一个函数，用于获取 PDF 字符过程（字形数据和路径信息）
def _get_pdf_charprocs(font_path, glyph_ids):
    # 使用指定路径获取字体对象
    font = get_font(font_path, hinting_factor=1)
    # 计算单位转换因子，将字形数据转换为 PS 单位（1/1000 的像素单位）
    conv = 1000 / font.units_per_EM  # Conversion to PS units (1/1000's).
    # 初始化一个空字典，用于存储字符过程信息
    procs = {}
    # 遍历每个字形 ID
    for glyph_id in glyph_ids:
        # 加载指定字形 ID 的字形数据，不进行缩放处理
        g = font.load_glyph(glyph_id, LOAD_NO_SCALE)
        # 计算字形数据的一些关键信息：水平进度、边界框等，乘以单位转换因子并四舍五入
        d1 = (np.array([g.horiAdvance, 0, *g.bbox]) * conv + .5).astype(int)
        # 获取字体路径信息和控制点
        v, c = font.get_path()
        # 将路径信息转换为 TrueType 的内部单位（1/64 的像素单位）
        v = (v * 64).astype(int)  # Back to TrueType's internal units (1/64's).
        # 处理旧版 ttconv 代码兼容性：控制点在两个四边形之间的情况
        quads, = np.nonzero(c == 3)
        quads_on = quads[1::2]
        quads_mid_on = np.array(
            sorted({*quads_on} & {*(quads - 1)} & {*(quads + 1)}), int)
        implicit = quads_mid_on[
            (v[quads_mid_on]
             == ((v[quads_mid_on - 1] + v[quads_mid_on + 1]) / 2).astype(int))
            .all(axis=1)]
        # 处理特定字体和字形 ID 的后退兼容性问题
        if (font.postscript_name, glyph_id) in [
                ("DejaVuSerif-Italic", 77),  # j
                ("DejaVuSerif-Italic", 135),  # \AA
        ]:
            v[:, 0] -= 1  # Hard-coded backcompat (FreeType shifts glyph by 1).
        # 将路径信息再次乘以单位转换因子并四舍五入处理
        v = (v * conv + .5).astype(int)  # As above re: truncation vs rounding.
        # 修复隐式控制点问题，再次进行四舍五入处理
        v[implicit] = (
            (v[implicit - 1] + v[implicit + 1]) / 2).astype(int)
        # 将字形名称及其字符过程信息存入字典
        procs[font.get_glyph_name(glyph_id)] = (
            " ".join(map(str, d1)).encode("ascii") + b" d1\n"
            + _path.convert_to_string(
                Path(v, c), None, None, False, None, -1,
                # no code for quad Beziers triggers auto-conversion to cubics.
                [b"m", b"l", b"", b"c", b"h"], True)
            + b"f")
    # 返回所有字符过程信息的字典
    return procs

# 定义一个 PDF 文件对象的类
class PdfFile:
    """PDF file object."""
    # 结束当前页面流，准备创建新页面
    def newPage(self, width, height):
        self.endStream()

        # 设置页面的宽度和高度
        self.width, self.height = width, height
        # 创建内容对象和注释对象的占位符
        contentObject = self.reserveObject('page contents')
        annotsObject = self.reserveObject('annotations')
        # 定义页面的属性和特性
        thePage = {'Type': Name('Page'),
                   'Parent': self.pagesObject,
                   'Resources': self.resourceObject,
                   'MediaBox': [0, 0, 72 * width, 72 * height],
                   'Contents': contentObject,
                   'Annots': annotsObject,
                   }
        # 创建页面对象并写入 PDF
        pageObject = self.reserveObject('page')
        self.writeObject(pageObject, thePage)
        # 将页面对象添加到页面列表中
        self.pageList.append(pageObject)
        # 添加页面的注释对象及其关联的注释列表
        self._annotations.append((annotsObject, self.pageAnnotations))

        # 开始页面内容流
        self.beginStream(contentObject.id,
                         self.reserveObject('length of content stream'))
        # 初始化 PDF 图形状态以匹配默认的 Matplotlib 图形上下文（颜色空间和线段连接样式）
        self.output(Name('DeviceRGB'), Op.setcolorspace_stroke)
        self.output(Name('DeviceRGB'), Op.setcolorspace_nonstroke)
        self.output(GraphicsContextPdf.joinstyles['round'], Op.setlinejoin)

        # 清除下一页的注释列表
        self.pageAnnotations = []

    # 创建新的文本注释
    def newTextnote(self, text, positionRect=[-100, -100, 0, 0]):
        # 创建一个文本类型的注释
        theNote = {'Type': Name('Annot'),
                   'Subtype': Name('Text'),
                   'Contents': text,
                   'Rect': positionRect,
                   }
        # 将该注释添加到页面的注释列表中
        self.pageAnnotations.append(theNote)

    # 获取子集化后的 PostScript 名称
    def _get_subsetted_psname(self, ps_name, charmap):
        # 将整数编码为基数为 26 的字符串
        def toStr(n, base):
            if n < base:
                return string.ascii_uppercase[n]
            else:
                return (
                    toStr(n // base, base) + string.ascii_uppercase[n % base]
                )

        # 计算字符映射的哈希值，并使用基数为 26 的编码转换为字符串
        hashed = hash(frozenset(charmap.keys())) % ((sys.maxsize + 1) * 2)
        prefix = toStr(hashed, 26)

        # 返回前缀加上原始的 PostScript 名称，用加号分隔
        return prefix[:6] + "+" + ps_name
    def finalize(self):
        """
        Write out the various deferred objects and the pdf end matter.
        """

        # 结束当前流
        self.endStream()
        # 写入注解对象
        self._write_annotations()
        # 写入字体对象
        self.writeFonts()
        # 写入扩展图形状态
        self.writeExtGSTates()
        # 写入软遮罩组
        self._write_soft_mask_groups()
        # 写入阴影填充对象
        self.writeHatches()
        # 写入高级渐变三角形对象
        self.writeGouraudTriangles()
        
        # 收集和处理 XObjects
        xobjects = {
            name: ob for image, name, ob in self._images.values()}
        for tup in self.markers.values():
            xobjects[tup[0]] = tup[1]
        for name, value in self.multi_byte_charprocs.items():
            xobjects[name] = value
        for name, path, trans, ob, join, cap, padding, filled, stroked \
                in self.paths:
            xobjects[name] = ob
        
        # 写入 XObjects 对象
        self.writeObject(self.XObjectObject, xobjects)
        # 写入图像对象
        self.writeImages()
        # 写入标记对象
        self.writeMarkers()
        # 写入路径集合模板对象
        self.writePathCollectionTemplates()
        # 写入页对象
        self.writeObject(self.pagesObject,
                         {'Type': Name('Pages'),
                          'Kids': self.pageList,
                          'Count': len(self.pageList)})
        # 写入信息字典
        self.writeInfoDict()

        # 最终化文件
        self.writeXref()
        self.writeTrailer()

    def close(self):
        """
        Flush all buffers and free all resources.
        """

        # 结束当前流
        self.endStream()
        # 如果传入的文件对象存在，刷新其缓冲区
        if self.passed_in_file_object:
            self.fh.flush()
        else:
            # 否则，如果有原始文件对象，则将缓冲区内容写入
            if self.original_file_like is not None:
                self.original_file_like.write(self.fh.getvalue())
            # 关闭文件句柄
            self.fh.close()

    def write(self, data):
        # 如果当前流为空，则直接写入数据到文件句柄
        if self.currentstream is None:
            self.fh.write(data)
        else:
            # 否则写入数据到当前流
            self.currentstream.write(data)

    def output(self, *data):
        # 将数据转换为 PDF 表示，并写入文件
        self.write(_fill([pdfRepr(x) for x in data]))
        # 写入换行符
        self.write(b'\n')

    def beginStream(self, id, len, extra=None, png=None):
        # 断言当前流为空
        assert self.currentstream is None
        # 开始新的流对象
        self.currentstream = Stream(id, len, self, extra, png)

    def endStream(self):
        # 如果当前流对象不为空，结束当前流
        if self.currentstream is not None:
            self.currentstream.end()
            self.currentstream = None

    def outputStream(self, ref, data, *, extra=None):
        # 开始输出流
        self.beginStream(ref.id, None, extra)
        # 写入数据到当前流
        self.currentstream.write(data)
        # 结束当前流
        self.endStream()

    def _write_annotations(self):
        # 遍历注解对象和对应的注解，写入到文件中
        for annotsObject, annotations in self._annotations:
            self.writeObject(annotsObject, annotations)
    def fontName(self, fontprop):
        """
        Select a font based on fontprop and return a name suitable for
        Op.selectfont. If fontprop is a string, it will be interpreted
        as the filename of the font.
        """

        # 如果 fontprop 是字符串，则将其作为字体文件名
        if isinstance(fontprop, str):
            filenames = [fontprop]
        # 如果使用14个核心字体（core fonts），根据属性查找字体文件名列表
        elif mpl.rcParams['pdf.use14corefonts']:
            filenames = _fontManager._find_fonts_by_props(
                fontprop, fontext='afm', directory=RendererPdf._afm_font_dir
            )
        # 否则，根据属性查找字体文件名列表
        else:
            filenames = _fontManager._find_fonts_by_props(fontprop)
        first_Fx = None
        # 遍历文件名列表中的每个文件名
        for fname in filenames:
            # 获取字体名对应的值
            Fx = self.fontNames.get(fname)
            # 如果 first_Fx 为空，则将其设为当前的 Fx
            if not first_Fx:
                first_Fx = Fx
            # 如果 Fx 为 None，则分配一个新的字体序号，并记录字体名和字体序号的关联
            if Fx is None:
                Fx = next(self._internal_font_seq)
                self.fontNames[fname] = Fx
                _log.debug('Assigning font %s = %r', Fx, fname)
                # 如果 first_Fx 为空，则将其设为当前的 Fx
                if not first_Fx:
                    first_Fx = Fx

        # find_fontsprop 的第一个值总是遵循 findfont 的值，因此在技术上没有行为变更
        return first_Fx

    def dviFontName(self, dvifont):
        """
        Given a dvi font object, return a name suitable for Op.selectfont.
        This registers the font information in ``self.dviFontInfo`` if not yet
        registered.
        """

        # 检查是否已经注册了 dvifont 对应的字体信息，如果是，则返回其对应的 pdfname
        dvi_info = self.dviFontInfo.get(dvifont.texname)
        if dvi_info is not None:
            return dvi_info.pdfname

        # 从 pdftex.map 文件中查找 TeX 字体文件映射信息
        tex_font_map = dviread.PsfontsMap(dviread.find_tex_file('pdftex.map'))
        # 获取 dvifont.texname 对应的 psfont 对象
        psfont = tex_font_map[dvifont.texname]
        # 如果 psfont 的 filename 为 None，则抛出异常，表示找不到可用的字体文件
        if psfont.filename is None:
            raise ValueError(
                "No usable font file found for {} (TeX: {}); "
                "the font may lack a Type-1 version"
                .format(psfont.psname, dvifont.texname))

        # 分配一个新的内部字体序号，并记录字体信息到 self.dviFontInfo 中
        pdfname = next(self._internal_font_seq)
        _log.debug('Assigning font %s = %s (dvi)', pdfname, dvifont.texname)
        self.dviFontInfo[dvifont.texname] = types.SimpleNamespace(
            dvifont=dvifont,
            pdfname=pdfname,
            fontfile=psfont.filename,
            basefont=psfont.psname,
            encodingfile=psfont.encoding,
            effects=psfont.effects)
        return pdfname
    # 定义一个方法，用于将字体信息写入 PDF 文档
    def writeFonts(self):
        # 初始化一个空字典来存储字体信息
        fonts = {}
        # 遍历已排序的 self.dviFontInfo 字典，包含了字体信息
        for dviname, info in sorted(self.dviFontInfo.items()):
            # 获取字体的 PDF 名称
            Fx = info.pdfname
            # 记录调试信息，指示正在嵌入 Type-1 字体
            _log.debug('Embedding Type-1 font %s from dvi.', dviname)
            # 调用 _embedTeXFont 方法将字体嵌入文档中，并将结果存储在字体字典中
            fonts[Fx] = self._embedTeXFont(info)
        # 遍历已排序的 self.fontNames 列表，包含了字体文件名
        for filename in sorted(self.fontNames):
            # 获取字体文件名对应的 Fx 值
            Fx = self.fontNames[filename]
            # 记录调试信息，指示正在嵌入字体文件
            _log.debug('Embedding font %s.', filename)
            # 如果文件名以 '.afm' 结尾，执行以下操作
            if filename.endswith('.afm'):
                # 记录调试信息，指示正在写入 AFM 字体
                _log.debug('Writing AFM font.')
                # 调用 _write_afm_font 方法将 AFM 字体写入文档，并将结果存储在字体字典中
                fonts[Fx] = self._write_afm_font(filename)
            else:
                # 如果不是 AFM 字体文件，记录调试信息，指示正在写入 TrueType 字体
                _log.debug('Writing TrueType font.')
                # 获取字符跟踪器中使用的特定字体文件的字符集合
                chars = self._character_tracker.used.get(filename)
                # 如果存在字符集合，调用 embedTTF 方法将 TrueType 字体嵌入文档，并将结果存储在字体字典中
                if chars:
                    fonts[Fx] = self.embedTTF(filename, chars)
        # 将字体字典写入 self.fontObject 对象
        self.writeObject(self.fontObject, fonts)

    # 定义一个方法，用于写入 AFM 字体到 PDF 文档中
    def _write_afm_font(self, filename):
        # 使用二进制方式打开 AFM 字体文件
        with open(filename, 'rb') as fh:
            # 创建 AFM 对象
            font = AFM(fh)
        # 获取字体名称
        fontname = font.get_fontname()
        # 创建字体字典，包含字体类型、子类型、基础字体名称和编码方式
        fontdict = {'Type': Name('Font'),
                    'Subtype': Name('Type1'),
                    'BaseFont': Name(fontname),
                    'Encoding': Name('WinAnsiEncoding')}
        # 为字体字典对象保留一个 PDF 对象编号
        fontdictObject = self.reserveObject('font dictionary')
        # 将字体字典写入 PDF 文档中
        self.writeObject(fontdictObject, fontdict)
        # 返回字体字典对象的编号
        return fontdictObject
    # 嵌入 TeX 字体到 PDF 中，接受字体信息对象作为参数
    def _embedTeXFont(self, fontinfo):
        # 记录调试信息，包括嵌入的 TeX 字体名称和字体信息字典
        _log.debug('Embedding TeX font %s - fontinfo=%s',
                   fontinfo.dvifont.texname, fontinfo.__dict__)

        # 生成字体宽度对象
        widthsObject = self.reserveObject('font widths')
        # 写入字体宽度数据到 PDF 中
        self.writeObject(widthsObject, fontinfo.dvifont.widths)

        # 生成字体字典对象
        fontdictObject = self.reserveObject('font dictionary')
        # 构建字体字典，包括字体类型、子类型、字符范围、宽度对象等信息
        fontdict = {
            'Type':      Name('Font'),
            'Subtype':   Name('Type1'),
            'FirstChar': 0,
            'LastChar':  len(fontinfo.dvifont.widths) - 1,
            'Widths':    widthsObject,
            }

        # 如果需要编码
        if fontinfo.encodingfile is not None:
            # 设置编码信息，包括编码类型和差异列表
            fontdict['Encoding'] = {
                'Type': Name('Encoding'),
                'Differences': [
                    0, *map(Name, dviread._parse_enc(fontinfo.encodingfile))],
            }

        # 如果未指定字体文件，输出警告信息并停止
        if fontinfo.fontfile is None:
            _log.warning(
                "Because of TeX configuration (pdftex.map, see updmap option "
                "pdftexDownloadBase14) the font %s is not embedded. This is "
                "deprecated as of PDF 1.5 and it may cause the consumer "
                "application to show something that was not intended.",
                fontinfo.basefont)
            # 设置基本字体名称到字典中，并写入字典对象到 PDF 中，然后返回字典对象
            fontdict['BaseFont'] = Name(fontinfo.basefont)
            self.writeObject(fontdictObject, fontdict)
            return fontdictObject

        # 存在要嵌入的字体文件，读取并应用任何效果
        t1font = _type1font.Type1Font(fontinfo.fontfile)
        if fontinfo.effects:
            t1font = t1font.transform(fontinfo.effects)
        # 设置字体的基本名称到字典中
        fontdict['BaseFont'] = Name(t1font.prop['FontName'])

        # 字体描述符可能在不同编码的 Type-1 字体之间共享，只有在不存在描述符时才创建新的
        effects = (fontinfo.effects.get('slant', 0.0),
                   fontinfo.effects.get('extend', 1.0))
        fontdesc = self.type1Descriptors.get((fontinfo.fontfile, effects))
        if fontdesc is None:
            # 创建 Type-1 字体描述符，并将其存储在缓存中
            fontdesc = self.createType1Descriptor(t1font, fontinfo.fontfile)
            self.type1Descriptors[(fontinfo.fontfile, effects)] = fontdesc
        # 设置字体描述符到字典中
        fontdict['FontDescriptor'] = fontdesc

        # 写入字典对象到 PDF 中
        self.writeObject(fontdictObject, fontdict)
        return fontdictObject
    def createType1Descriptor(self, t1font, fontfile):
        # 创建并写入Type-1字体的字体描述符和字体文件

        # 分配一个对象用于字体描述符和字体文件
        fontdescObject = self.reserveObject('font descriptor')
        fontfileObject = self.reserveObject('font file')

        # 获取字体的斜体角度和固定间距属性
        italic_angle = t1font.prop['ItalicAngle']
        fixed_pitch = t1font.prop['isFixedPitch']

        # 初始化标志位
        flags = 0
        # 如果字体具有固定宽度，则设置固定宽度标志位
        if fixed_pitch:
            flags |= 1 << 0
        # TODO: serif（暂未处理）
        if 0:
            flags |= 1 << 1
        # TODO: symbolic（大多数TeX字体都是符号字体）
        if 1:
            flags |= 1 << 2
        # 非符号字体
        else:
            flags |= 1 << 5
        # 如果字体具有斜体角度，则设置斜体标志位
        if italic_angle:
            flags |= 1 << 6
        # TODO: all caps（暂未处理）
        if 0:
            flags |= 1 << 16
        # TODO: small caps（暂未处理）
        if 0:
            flags |= 1 << 17
        # TODO: force bold（暂未处理）
        if 0:
            flags |= 1 << 18

        # 获取字体文件对象
        ft2font = get_font(fontfile)

        # 创建字体描述符的字典
        descriptor = {
            'Type':        Name('FontDescriptor'),
            'FontName':    Name(t1font.prop['FontName']),
            'Flags':       flags,
            'FontBBox':    ft2font.bbox,
            'ItalicAngle': italic_angle,
            'Ascent':      ft2font.ascender,
            'Descent':     ft2font.descender,
            'CapHeight':   1000,  # TODO: 确定这个值
            'XHeight':     500,   # TODO: 确定这个值
            'FontFile':    fontfileObject,
            'FontFamily':  t1font.prop['FamilyName'],
            'StemV':       50,    # TODO: 确定这个值
            # （参考修订版3874；但并非所有TeX发行版都有AFM文件！）
            # 'FontWeight': 400 = Regular, 700 = Bold 的数字表示字体的粗细
            }

        # 写入字体描述符对象
        self.writeObject(fontdescObject, descriptor)

        # 输出字体文件流
        self.outputStream(fontfileObject, b"".join(t1font.parts[:2]),
                          extra={'Length1': len(t1font.parts[0]),
                                 'Length2': len(t1font.parts[1]),
                                 'Length3': 0})

        # 返回字体描述符对象
        return fontdescObject

    def _get_xobject_glyph_name(self, filename, glyph_name):
        # 获取XObject的字形名称
        Fx = self.fontName(filename)
        return "-".join([
            Fx.name.decode(),
            os.path.splitext(os.path.basename(filename))[0],
            glyph_name])

    _identityToUnicodeCMap = b"""/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo
<< /Registry (Adobe)
   /Ordering (UCS)
   /Supplement 0
>> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <ffff>
endcodespacerange
%d beginbfrange
%s
endbfrange
endcmap
CMapName currentdict /CMap defineresource pop
end
end"""

这段代码片段是 PostScript 语言的一部分，用于定义一个名为 `/Adobe-Identity-UCS` 的字符映射表 (CMap)，以及一些相关的字形编码和映射。


    def alphaState(self, alpha):
        """Return name of an ExtGState that sets alpha to the given value."""

        state = self.alphaStates.get(alpha, None)
        if state is not None:
            return state[0]

        name = next(self._alpha_state_seq)
        self.alphaStates[alpha] = \
            (name, {'Type': Name('ExtGState'),
                    'CA': alpha[0], 'ca': alpha[1]})
        return name

这是一个类方法 `alphaState`，用于获取设置 alpha 值的 ExtGState 的名称。如果已经存在该 alpha 值对应的 ExtGState，则直接返回其名称；否则，创建一个新的 ExtGState，并将其名称和属性存储在 `alphaStates` 字典中。


    def _soft_mask_state(self, smask):
        """
        Return an ExtGState that sets the soft mask to the given shading.

        Parameters
        ----------
        smask : Reference
            Reference to a shading in DeviceGray color space, whose luminosity
            is to be used as the alpha channel.

        Returns
        -------
        Name
        """

        state = self._soft_mask_states.get(smask, None)
        if state is not None:
            return state[0]

        name = next(self._soft_mask_seq)
        groupOb = self.reserveObject('transparency group for soft mask')
        self._soft_mask_states[smask] = (
            name,
            {
                'Type': Name('ExtGState'),
                'AIS': False,
                'SMask': {
                    'Type': Name('Mask'),
                    'S': Name('Luminosity'),
                    'BC': [1],
                    'G': groupOb
                }
            }
        )
        self._soft_mask_groups.append((
            groupOb,
            {
                'Type': Name('XObject'),
                'Subtype': Name('Form'),
                'FormType': 1,
                'Group': {
                    'S': Name('Transparency'),
                    'CS': Name('DeviceGray')
                },
                'Matrix': [1, 0, 0, 1, 0, 0],
                'Resources': {'Shading': {'S': smask}},
                'BBox': [0, 0, 1, 1]
            },
            [Name('S'), Op.shading]
        ))
        return name

这是一个类方法 `_soft_mask_state`，用于获取设置软蒙版 (soft mask) 的 ExtGState。如果已经存在该软蒙版对应的 ExtGState，则直接返回其名称；否则，创建一个新的 ExtGState，并将其名称和属性存储在 `_soft_mask_states` 字典中。同时，将新创建的软蒙版组信息存储在 `_soft_mask_groups` 列表中。


    def writeExtGSTates(self):
        self.writeObject(
            self._extGStateObject,
            dict([
                *self.alphaStates.values(),
                *self._soft_mask_states.values()
            ])
        )

这是一个类方法 `writeExtGSTates`，用于将所有已定义的 ExtGState 写入输出。它使用 `writeObject` 方法将 `_extGStateObject` 中存储的所有 alpha 值和软蒙版的 ExtGState 写入到输出中。


    def _write_soft_mask_groups(self):
        for ob, attributes, content in self._soft_mask_groups:
            self.beginStream(ob.id, None, attributes)
            self.output(*content)
            self.endStream()

这是一个类方法 `_write_soft_mask_groups`，用于将所有存储的软蒙版组信息 `_soft_mask_groups` 写入输出。它遍历 `_soft_mask_groups` 列表，并使用 `beginStream`、`output` 和 `endStream` 方法将每个软蒙版组的内容写入输出。
    def hatchPattern(self, hatch_style):
        # 如果输入的填充样式不为 None，则将其边缘、填充和填充图案元组化，因为 numpy 数组不可哈希化
        if hatch_style is not None:
            edge, face, hatch = hatch_style
            if edge is not None:
                edge = tuple(edge)
            if face is not None:
                face = tuple(face)
            hatch_style = (edge, face, hatch)

        # 获取已存在的填充图案，如果存在则直接返回
        pattern = self.hatchPatterns.get(hatch_style, None)
        if pattern is not None:
            return pattern

        # 如果填充图案不存在，则分配一个新的名称
        name = next(self._hatch_pattern_seq)
        # 将新的填充图案与其名称关联存储在 hatchPatterns 字典中
        self.hatchPatterns[hatch_style] = name
        return name

    def writeHatches(self):
        # 初始化 hatchDict 作为存储填充图案的字典
        hatchDict = dict()
        # 定义填充图案的边长
        sidelen = 72.0
        # 遍历 self.hatchPatterns 中的每个填充样式和对应的名称
        for hatch_style, name in self.hatchPatterns.items():
            # 为当前填充图案分配一个新的对象
            ob = self.reserveObject('hatch pattern')
            # 将填充图案名称与对象关联存储在 hatchDict 字典中
            hatchDict[name] = ob
            # 定义填充图案的资源，包括处理 PDF、文本和图像的过程集
            res = {'Procsets':
                   [Name(x) for x in "PDF Text ImageB ImageC ImageI".split()]}
            # 开始填充图案的流
            self.beginStream(
                ob.id, None,
                {'Type': Name('Pattern'),
                 'PatternType': 1, 'PaintType': 1, 'TilingType': 1,
                 'BBox': [0, 0, sidelen, sidelen],
                 'XStep': sidelen, 'YStep': sidelen,
                 'Resources': res,
                 # 将原点位置更改以匹配 Agg 在左上角的位置
                 'Matrix': [1, 0, 0, 1, 0, self.height * 72]})

            # 获取填充图案的描边颜色、填充颜色和填充图案
            stroke_rgb, fill_rgb, hatch = hatch_style
            # 输出描边颜色
            self.output(stroke_rgb[0], stroke_rgb[1], stroke_rgb[2],
                        Op.setrgb_stroke)
            # 如果填充颜色不为 None，则输出填充颜色
            if fill_rgb is not None:
                self.output(fill_rgb[0], fill_rgb[1], fill_rgb[2],
                            Op.setrgb_nonstroke,
                            0, 0, sidelen, sidelen, Op.rectangle,
                            Op.fill)

            # 输出 matplotlib 配置中定义的填充图案线宽度
            self.output(mpl.rcParams['hatch.linewidth'], Op.setlinewidth)

            # 输出填充图案路径操作，将 hatch 应用于 Path.hatch，通过 Affine2D 对其进行缩放，不进行简化
            self.output(*self.pathOperations(
                Path.hatch(hatch),
                Affine2D().scale(sidelen),
                simplify=False))
            # 输出填充和描边操作
            self.output(Op.fill_stroke)

            # 结束填充图案的流
            self.endStream()
        
        # 将填充图案对象及其名称字典写入文档中
        self.writeObject(self.hatchObject, hatchDict)

    def addGouraudTriangles(self, points, colors):
        """
        Add a Gouraud triangle shading.

        Parameters
        ----------
        points : np.ndarray
            Triangle vertices, shape (n, 3, 2)
            where n = number of triangles, 3 = vertices, 2 = x, y.
        colors : np.ndarray
            Vertex colors, shape (n, 3, 1) or (n, 3, 4)
            as with points, but last dimension is either (gray,)
            or (r, g, b, alpha).

        Returns
        -------
        Name, Reference
        """
        # 分配一个新的 Gouraud 三角形名称
        name = Name('GT%d' % len(self.gouraudTriangles))
        # 为新的 Gouraud 三角形分配一个对象
        ob = self.reserveObject(f'Gouraud triangle {name}')
        # 将新的 Gouraud 三角形名称、对象、顶点和颜色元组添加到 gouraudTriangles 列表中
        self.gouraudTriangles.append((name, ob, points, colors))
        return name, ob
    # 将高尔德三角形写入 PDF 文档
    def writeGouraudTriangles(self):
        # 创建空字典用于存储高尔德三角形的信息
        gouraudDict = dict()
        # 遍历每个高尔德三角形的名称、对象、顶点、颜色信息
        for name, ob, points, colors in self.gouraudTriangles:
            # 将每个三角形的名称关联到对象
            gouraudDict[name] = ob
            # 获取顶点数组的形状信息
            shape = points.shape
            # 将顶点数组展平为一维数组
            flat_points = points.reshape((shape[0] * shape[1], 2))
            # 获取颜色数组的通道数
            colordim = colors.shape[2]
            # 断言颜色通道数为 1 或 4
            assert colordim in (1, 4)
            # 将颜色数组展平为一维数组
            flat_colors = colors.reshape((shape[0] * shape[1], colordim))
            # 如果颜色通道数为 4，则去除 alpha 通道
            if colordim == 4:
                colordim = 3
            # 计算顶点的最小和最大值，并扩展范围
            points_min = np.min(flat_points, axis=0) - (1 << 8)
            points_max = np.max(flat_points, axis=0) + (1 << 8)
            # 计算缩放因子，用于将顶点映射到 PDF 页面范围
            factor = 0xffffffff / (points_max - points_min)

            # 开始写入 PDF 流对象
            self.beginStream(
                ob.id, None,
                {'ShadingType': 4,
                 'BitsPerCoordinate': 32,
                 'BitsPerComponent': 8,
                 'BitsPerFlag': 8,
                 'ColorSpace': Name(
                     'DeviceRGB' if colordim == 3 else 'DeviceGray'
                 ),
                 'AntiAlias': False,
                 'Decode': ([points_min[0], points_max[0],
                             points_min[1], points_max[1]]
                            + [0, 1] * colordim),
                 })

            # 创建 PDF 流数据数组
            streamarr = np.empty(
                (shape[0] * shape[1],),
                dtype=[('flags', 'u1'),
                       ('points', '>u4', (2,)),
                       ('colors', 'u1', (colordim,))])
            # 初始化流数据数组的标志位为 0
            streamarr['flags'] = 0
            # 将顶点数据映射并存入流数据数组
            streamarr['points'] = (flat_points - points_min) * factor
            # 将颜色数据映射并存入流数据数组，将颜色值缩放到 0-255 范围
            streamarr['colors'] = flat_colors[:, :colordim] * 255.0

            # 将流数据数组写入 PDF 文档
            self.write(streamarr.tobytes())
            # 结束 PDF 流对象的写入
            self.endStream()

        # 将创建的高尔德对象及其字典写入 PDF 文档
        self.writeObject(self.gouraudObject, gouraudDict)

    # 返回给定图像的 XObject 名称，用于表示该图像
    def imageObject(self, image):
        """Return name of an image XObject representing the given image."""

        # 尝试从缓存中获取图像的条目信息
        entry = self._images.get(id(image), None)
        # 如果已经存在，则返回对应的名称
        if entry is not None:
            return entry[1]

        # 否则生成一个新的图像名称
        name = next(self._image_seq)
        # 为图像名称预留 PDF 对象
        ob = self.reserveObject(f'image {name}')
        # 将图像及其名称关联存入缓存
        self._images[id(image)] = (image, name, ob)
        # 返回新生成的图像名称
        return name

    # 解压图像数组 *im*，返回 ``(data, alpha)``，形状为 ``(height, width, 3)`` (RGB) 或 ``(height, width, 1)`` (灰度或 alpha)
    def _unpack(self, im):
        """
        Unpack image array *im* into ``(data, alpha)``, which have shape
        ``(height, width, 3)`` (RGB) or ``(height, width, 1)`` (grayscale or
        alpha), except that alpha is None if the image is fully opaque.
        """
        # 颠倒图像数组的顺序
        im = im[::-1]
        # 如果图像数组的维度为 2，表示为灰度图像或完全不透明的 RGB 图像
        if im.ndim == 2:
            return im, None
        else:
            # 提取 RGB 通道的图像数据，并以 C 顺序存储
            rgb = im[:, :, :3]
            rgb = np.array(rgb, order='C')
            # 如果图像数组的第三个维度为 4，表示包含 alpha 通道
            if im.shape[2] == 4:
                # 提取 alpha 通道的数据
                alpha = im[:, :, 3][..., None]
                # 如果 alpha 通道全为 255，则表示完全不透明，将 alpha 设置为 None
                if np.all(alpha == 255):
                    alpha = None
                else:
                    # 否则将 alpha 数据以 C 顺序存储
                    alpha = np.array(alpha, order='C')
            else:
                # 否则图像没有 alpha 通道，将 alpha 设置为 None
                alpha = None
            # 返回提取的 RGB 数据及 alpha 数据
            return rgb, alpha
    # 定义一个私有方法 _writePng，用于将图片 img 以 PNG 格式写入 PDF 文件，使用带有 Flate 压缩的预测器。
    def _writePng(self, img):
        """
        Write the image *img* into the pdf file using png
        predictors with Flate compression.
        """
        # 创建一个字节流缓冲区
        buffer = BytesIO()
        # 将图片 img 以 PNG 格式保存到字节流缓冲区中
        img.save(buffer, format="png")
        # 将字节流缓冲区的读取指针移动到偏移量为 8 的位置
        buffer.seek(8)
        # 初始化 PNG 数据、位深度和调色板
        png_data = b''
        bit_depth = palette = None
        # 循环处理字节流中的数据段
        while True:
            # 从缓冲区中读取 8 字节数据，并按照格式解析为长度和类型
            length, type = struct.unpack(b'!L4s', buffer.read(8))
            # 如果类型为以下之一：IHDR、PLTE、IDAT
            if type in [b'IHDR', b'PLTE', b'IDAT']:
                # 从缓冲区中读取指定长度的数据段
                data = buffer.read(length)
                # 如果读取的数据长度不等于预期长度，则抛出运行时异常
                if len(data) != length:
                    raise RuntimeError("truncated data")
                # 根据类型分别处理数据
                if type == b'IHDR':
                    bit_depth = int(data[8])
                elif type == b'PLTE':
                    palette = data
                elif type == b'IDAT':
                    png_data += data
            # 如果类型为 IEND，则退出循环
            elif type == b'IEND':
                break
            # 否则，移动缓冲区读取指针到下一个数据段的开头，并跳过 CRC
            else:
                buffer.seek(length, 1)
            # 移动缓冲区读取指针到当前位置的下一个数据段的开头，并跳过 CRC
            buffer.seek(4, 1)
        # 返回处理后的 PNG 数据、位深度和调色板
        return png_data, bit_depth, palette
    # 将图像数据写入 PDF 文档中的对象 *id*，支持灰度 ``(height, width, 1)`` 或 RGB ``(height, width, 3)`` 格式的图像，
    # 可选的软遮罩 *smask* 是一个 ``(height, width, 1)`` 的数组，可以为 None。
    def _writeImg(self, data, id, smask=None):
        # 获取图像的高度、宽度和颜色通道数
        height, width, color_channels = data.shape
        # 构建图像对象的属性字典
        obj = {'Type': Name('XObject'),
               'Subtype': Name('Image'),
               'Width': width,
               'Height': height,
               'ColorSpace': Name({1: 'DeviceGray', 3: 'DeviceRGB'}[color_channels]),  # 根据颜色通道数选择颜色空间
               'BitsPerComponent': 8}  # 每个颜色通道的比特数
    
        # 如果有软遮罩，则添加软遮罩信息到对象字典中
        if smask:
            obj['SMask'] = smask
    
        # 如果开启 PDF 压缩选项
        if mpl.rcParams['pdf.compression']:
            # 如果图像是单通道的，压缩时去掉单通道维度
            if data.shape[-1] == 1:
                data = data.squeeze(axis=-1)
            # 配置 PNG 压缩选项
            png = {'Predictor': 10, 'Colors': color_channels, 'Columns': width}
            # 使用 PIL 库从图像数据创建图像对象
            img = Image.fromarray(data)
            # 获取图像颜色列表
            img_colors = img.getcolors(maxcolors=256)
    
            # 如果图像是 RGB 且颜色数量不超过 256，转换为索引颜色以减小文件大小
            if color_channels == 3 and img_colors is not None:
                num_colors = len(img_colors)
                palette = np.array([comp for _, color in img_colors for comp in color], dtype=np.uint8)
                palette24 = ((palette[0::3].astype(np.uint32) << 16) |
                             (palette[1::3].astype(np.uint32) << 8) |
                             palette[2::3])
                rgb24 = ((data[:, :, 0].astype(np.uint32) << 16) |
                         (data[:, :, 1].astype(np.uint32) << 8) |
                         data[:, :, 2])
                indices = np.argsort(palette24).astype(np.uint8)
                rgb8 = indices[np.searchsorted(palette24, rgb24, sorter=indices)]
                img = Image.fromarray(rgb8, mode='P')
                img.putpalette(palette)
                # 写入 PNG 数据
                png_data, bit_depth, palette = self._writePng(img)
                if bit_depth is None or palette is None:
                    raise RuntimeError("invalid PNG header")
                palette = palette[:num_colors * 3]  # 去除多余的调色板填充，适用于 Pillow>=9
                obj['ColorSpace'] = [Name('Indexed'), Name('DeviceRGB'),
                                     num_colors - 1, palette]
                obj['BitsPerComponent'] = bit_depth
                png['Colors'] = 1
                png['BitsPerComponent'] = bit_depth
            else:
                # 直接写入 PNG 数据
                png_data, _, _ = self._writePng(img)
        else:
            png = None
        
        # 开始写入图像流到 PDF 文档
        self.beginStream(
            id,
            self.reserveObject('length of image stream'),
            obj,
            png=png
            )
        
        # 如果有 PNG 数据，则将其写入当前流
        if png:
            self.currentstream.write(png_data)
        else:
            # 否则直接将图像数据转换为字节并写入当前流
            self.currentstream.write(data.tobytes())
        
        # 结束图像流的写入
        self.endStream()
    # 遍历 self._images 中的每个元素，获取图像数据、名称和对象信息
    def writeImages(self):
        for img, name, ob in self._images.values():
            # 解包图像数据，包括数据和可能的alpha通道数据
            data, adata = self._unpack(img)
            # 如果存在alpha通道数据
            if adata is not None:
                # 为smask对象保留一个对象ID
                smaskObject = self.reserveObject("smask")
                # 将alpha通道数据写入smaskObject
                self._writeImg(adata, smaskObject.id)
            else:
                smaskObject = None
            # 将图像数据写入对象ob，同时写入smaskObject（如果存在）
            self._writeImg(data, ob.id, smaskObject)

    # 创建一个标记X对象，代表给定路径
    def markerObject(self, path, trans, fill, stroke, lw, joinstyle,
                     capstyle):
        """Return name of a marker XObject representing the given path."""
        # self.markers被markerObject、writeMarkers、close方法使用：
        # 映射从（路径操作，是否填充，是否描边）到[name, 对象引用, 边界框, 线宽]的列表
        # 这使得不同的draw_markers调用可以共享XObject，只要图形上下文足够相似：
        # 颜色等可以变化，但填充和描边的选择不能变化。
        # 我们需要一个边界框来包围所有的XObject路径，
        # 但由于线宽可能不同，我们在self.markers中存储所有出现的最大线宽。
        # close()方法与self.markers有些紧密耦合，因为它期望self.markers中每个值的前两个组件是名称和对象引用。
        # 根据给定的路径、变换和简化选项生成路径操作序列
        pathops = self.pathOperations(path, trans, simplify=False)
        key = (tuple(pathops), bool(fill), bool(stroke), joinstyle, capstyle)
        result = self.markers.get(key)
        if result is None:
            # 创建一个新的名称和对象，并为其保留一个对象ID
            name = Name('M%d' % len(self.markers))
            ob = self.reserveObject('marker %d' % len(self.markers))
            # 获取路径在给定变换下的边界框
            bbox = path.get_extents(trans)
            # 将新创建的标记信息存储在self.markers中
            self.markers[key] = [name, ob, bbox, lw]
        else:
            # 如果标记已经存在，更新最大线宽
            if result[-1] < lw:
                result[-1] = lw
            name = result[0]
        # 返回标记对象的名称
        return name
    def writeMarkers(self):
        # 遍历 self.markers 字典中的每一个条目
        for ((pathops, fill, stroke, joinstyle, capstyle),
             (name, ob, bbox, lw)) in self.markers.items():
            # 计算扩展后的 bbox，以确保线条不会超出其边界
            bbox = bbox.padded(lw * 5)
            # 开始一个新的图形流对象，设置其属性
            self.beginStream(
                ob.id, None,
                {'Type': Name('XObject'), 'Subtype': Name('Form'),
                 'BBox': list(bbox.extents)})
            # 输出当前图形对象的线段连接样式
            self.output(GraphicsContextPdf.joinstyles[joinstyle],
                        Op.setlinejoin)
            # 输出当前图形对象的线条端点样式
            self.output(GraphicsContextPdf.capstyles[capstyle], Op.setlinecap)
            # 输出当前路径操作序列
            self.output(*pathops)
            # 输出路径填充和描边操作
            self.output(Op.paint_path(fill, stroke))
            # 结束当前图形流对象
            self.endStream()

    def pathCollectionObject(self, gc, path, trans, padding, filled, stroked):
        # 为路径集合对象生成一个唯一的名称
        name = Name('P%d' % len(self.paths))
        # 为路径集合对象预留一个 PDF 对象
        ob = self.reserveObject('path %d' % len(self.paths))
        # 将路径信息添加到 self.paths 列表中
        self.paths.append(
            (name, path, trans, ob, gc.get_joinstyle(), gc.get_capstyle(),
             padding, filled, stroked))
        return name

    def writePathCollectionTemplates(self):
        # 遍历 self.paths 列表中的每一个路径集合对象
        for (name, path, trans, ob, joinstyle, capstyle, padding, filled,
             stroked) in self.paths:
            # 根据路径、变换和填充扩展获取路径的边界框
            bbox = path.get_extents(trans)
            # 如果边界框的尺寸存在非有限值，则将其设置为默认值 [0, 0, 0, 0]
            if not np.all(np.isfinite(bbox.extents)):
                extents = [0, 0, 0, 0]
            else:
                bbox = bbox.padded(padding)
                extents = list(bbox.extents)
            # 开始一个新的图形流对象，设置其属性
            self.beginStream(
                ob.id, None,
                {'Type': Name('XObject'), 'Subtype': Name('Form'),
                 'BBox': extents})
            # 输出当前图形对象的线段连接样式
            self.output(GraphicsContextPdf.joinstyles[joinstyle],
                        Op.setlinejoin)
            # 输出当前图形对象的线条端点样式
            self.output(GraphicsContextPdf.capstyles[capstyle], Op.setlinecap)
            # 获取路径操作序列
            pathops = self.pathOperations(path, trans, simplify=False)
            # 输出路径操作序列
            self.output(*pathops)
            # 输出路径填充和描边操作
            self.output(Op.paint_path(filled, stroked))
            # 结束当前图形流对象
            self.endStream()

    @staticmethod
    def pathOperations(path, transform, clip=None, simplify=None, sketch=None):
        # 将路径转换为字符串表示，包括变换、裁剪等信息
        return [Verbatim(_path.convert_to_string(
            path, transform, clip, simplify, sketch,
            6,
            [Op.moveto.value, Op.lineto.value, b'', Op.curveto.value,
             Op.closepath.value],
            True))]
    def writePath(self, path, transform, clip=False, sketch=None):
        # 如果 clip 被设置为 True，则定义剪辑区域为整个页面大小的矩形
        if clip:
            clip = (0.0, 0.0, self.width * 72, self.height * 72)
            # 获取路径对象的简化属性
            simplify = path.should_simplify
        else:
            # 如果 clip 未设置，则剪辑区域为 None
            clip = None
            # 简化属性设置为 False
            simplify = False
        # 调用 pathOperations 方法生成路径操作的命令
        cmds = self.pathOperations(path, transform, clip, simplify=simplify,
                                   sketch=sketch)
        # 输出生成的命令
        self.output(*cmds)

    def reserveObject(self, name=''):
        """
        Reserve an ID for an indirect object.

        The name is used for debugging in case we forget to print out
        the object with writeObject.
        """
        # 从 _object_seq 中获取下一个可用的 ID
        id = next(self._object_seq)
        # 在 xrefTable 中添加一个新条目，暂时用 None 和 0 初始化，记录名字以便调试
        self.xrefTable.append([None, 0, name])
        # 返回一个新的 Reference 对象，使用刚刚生成的 ID
        return Reference(id)

    def recordXref(self, id):
        # 记录对象在文件中的偏移量
        self.xrefTable[id][0] = self.fh.tell() - self.tell_base

    def writeObject(self, object, contents):
        # 记录对象的交叉引用信息
        self.recordXref(object.id)
        # 使用对象的写方法将内容写入文件
        object.write(contents, self)

    def writeXref(self):
        """Write out the xref table."""
        # 记录起始交叉引用表的位置
        self.startxref = self.fh.tell() - self.tell_base
        # 写入交叉引用表的头部信息
        self.write(b"xref\n0 %d\n" % len(self.xrefTable))
        # 遍历 xrefTable 中的每个条目，生成对应的文本信息并写入文件
        for i, (offset, generation, name) in enumerate(self.xrefTable):
            if offset is None:
                # 如果偏移量为 None，则抛出断言错误
                raise AssertionError(
                    'No offset for object %d (%s)' % (i, name))
            else:
                # 根据对象名字选择不同的键
                key = b"f" if name == 'the zero object' else b"n"
                # 格式化输出偏移量、生成数、键，并将结果写入文件
                text = b"%010d %05d %b \n" % (offset, generation, key)
                self.write(text)

    def writeInfoDict(self):
        """Write out the info dictionary, checking it for good form"""

        # 为信息字典对象预留一个 ID
        self.infoObject = self.reserveObject('info')
        # 使用 writeObject 方法将信息字典写入文件
        self.writeObject(self.infoObject, self.infoDict)

    def writeTrailer(self):
        """Write out the PDF trailer."""

        # 写入 PDF 尾部的起始标识
        self.write(b"trailer\n")
        # 写入 PDF 尾部的主要信息，包括文件大小、根对象、信息字典对象
        self.write(pdfRepr(
            {'Size': len(self.xrefTable),
             'Root': self.rootObject,
             'Info': self.infoObject}))
        # 可以添加 'ID'
        # 写入 PDF 尾部的交叉引用起始位置和结束标识
        self.write(b"\nstartxref\n%d\n%%%%EOF\n" % self.startxref)
class RendererPdf(_backend_pdf_ps.RendererPDFPSBase):
    # PDF 渲染器，继承自 _backend_pdf_ps.RendererPDFPSBase

    _afm_font_dir = cbook._get_data_path("fonts/pdfcorefonts")
    # AFM 字体目录，使用 cbook._get_data_path 获取路径

    _use_afm_rc_name = "pdf.use14corefonts"
    # 使用 AFM 字体的配置名称，值为 "pdf.use14corefonts"

    def __init__(self, file, image_dpi, height, width):
        # 初始化方法，接受文件对象 file，图像 DPI image_dpi，高度 height，宽度 width
        super().__init__(width, height)
        # 调用父类的初始化方法，设置宽度和高度
        self.file = file
        # 设置文件属性
        self.gc = self.new_gc()
        # 初始化 gc 属性，调用 self.new_gc() 方法
        self.image_dpi = image_dpi
        # 设置图像 DPI 属性

    def finalize(self):
        # 结束方法，输出最终内容到文件
        self.file.output(*self.gc.finalize())
        # 调用文件对象的 output 方法，传入 gc 的 finalize 方法的结果作为参数

    def check_gc(self, gc, fillcolor=None):
        # 检查 gc 方法，传入 gc 对象和填充颜色 fillcolor
        orig_fill = getattr(gc, '_fillcolor', (0., 0., 0.))
        # 获取 gc 对象的 _fillcolor 属性，若无则为默认值 (0., 0., 0.)
        gc._fillcolor = fillcolor
        # 设置 gc 对象的 _fillcolor 属性为传入的 fillcolor

        orig_alphas = getattr(gc, '_effective_alphas', (1.0, 1.0))
        # 获取 gc 对象的 _effective_alphas 属性，若无则为默认值 (1.0, 1.0)

        if gc.get_rgb() is None:
            # 如果 gc 对象的 RGB 颜色为 None
            # 这里的颜色应该无关紧要，因为线宽应该为 0，除非受到 rcParams 中全局设置的影响，因此为了安全起见，设置 alpha 为 0
            gc.set_foreground((0, 0, 0, 0), isRGBA=True)

        if gc._forced_alpha:
            # 如果 gc 对象有强制 alpha 值
            gc._effective_alphas = (gc._alpha, gc._alpha)
            # 设置 gc 的有效 alpha 值为 gc._alpha 的元组
        elif fillcolor is None or len(fillcolor) < 4:
            # 否则如果 fillcolor 为 None 或者长度小于 4
            gc._effective_alphas = (gc._rgb[3], 1.0)
            # 设置 gc 的有效 alpha 值为 gc._rgb[3] 和 1.0 的元组
        else:
            gc._effective_alphas = (gc._rgb[3], fillcolor[3])
            # 否则设置 gc 的有效 alpha 值为 gc._rgb[3] 和 fillcolor[3] 的元组

        delta = self.gc.delta(gc)
        # 计算当前 gc 对象与 self.gc 的差异
        if delta:
            self.file.output(*delta)
            # 如果有差异，则将 delta 写入文件

        # 恢复 gc，避免不必要的副作用
        gc._fillcolor = orig_fill
        gc._effective_alphas = orig_alphas
        # 恢复 gc 的 _fillcolor 和 _effective_alphas 属性为原始值

    def get_image_magnification(self):
        # 获取图像放大倍率的方法
        return self.image_dpi/72.0
        # 返回图像 DPI 除以 72.0 的结果作为放大倍率

    def draw_image(self, gc, x, y, im, transform=None):
        # 绘制图像方法，传入 gc 对象，坐标 x 和 y，图像 im，变换 transform
        h, w = im.shape[:2]
        # 获取图像的高度 h 和宽度 w
        if w == 0 or h == 0:
            return
            # 如果宽度或高度为 0，则返回

        if transform is None:
            # 如果没有变换矩阵
            gc.set_alpha(1.0)
            # 设置 gc 的 alpha 值为 1.0

        self.check_gc(gc)
        # 调用 check_gc 方法检查 gc

        w = 72.0 * w / self.image_dpi
        h = 72.0 * h / self.image_dpi
        # 计算图像的宽度和高度

        imob = self.file.imageObject(im)
        # 创建图像对象 imob

        if transform is None:
            self.file.output(Op.gsave,
                             w, 0, 0, h, x, y, Op.concat_matrix,
                             imob, Op.use_xobject, Op.grestore)
            # 如果没有变换矩阵，则输出 gsave、concat_matrix、use_xobject、grestore 操作
        else:
            tr1, tr2, tr3, tr4, tr5, tr6 = transform.frozen().to_values()
            # 否则获取变换矩阵的值

            self.file.output(Op.gsave,
                             1, 0, 0, 1, x, y, Op.concat_matrix,
                             tr1, tr2, tr3, tr4, tr5, tr6, Op.concat_matrix,
                             imob, Op.use_xobject, Op.grestore)
            # 输出 gsave、concat_matrix、concat_matrix、use_xobject、grestore 操作

    def draw_path(self, gc, path, transform, rgbFace=None):
        # 绘制路径方法，传入 gc 对象，路径 path，变换 transform，填充颜色 rgbFace
        self.check_gc(gc, rgbFace)
        # 调用 check_gc 方法检查 gc 和填充颜色 rgbFace

        self.file.writePath(
            path, transform,
            rgbFace is None and gc.get_hatch_path() is None,
            gc.get_sketch_params())
        # 将路径信息写入文件

        self.file.output(self.gc.paint())
        # 输出 gc 的 paint 操作到文件
    def draw_markers(self, gc, marker_path, marker_trans, path, trans,
                     rgbFace=None):
        # 绘制标记点，继承了文档字符串的功能

        # 判断是否需要使用 RendererBase 中的 draw_markers 方法
        len_marker_path = len(marker_path)
        uses = len(path)
        if len_marker_path * uses < len_marker_path + uses + 5:
            # 如果条件成立，则调用 RendererBase 的 draw_markers 方法并返回
            RendererBase.draw_markers(self, gc, marker_path, marker_trans,
                                      path, trans, rgbFace)
            return

        # 检查并设置绘图上下文的填充和描边属性
        self.check_gc(gc, rgbFace)
        fill = gc.fill(rgbFace)
        stroke = gc.stroke()

        # 获取输出流对象
        output = self.file.output
        # 创建标记对象
        marker = self.file.markerObject(
            marker_path, marker_trans, fill, stroke, self.gc._linewidth,
            gc.get_joinstyle(), gc.get_capstyle())

        # 开始绘图操作
        output(Op.gsave)
        lastx, lasty = 0, 0
        for vertices, code in path.iter_segments(
                trans,
                clip=(0, 0, self.file.width*72, self.file.height*72),
                simplify=False):
            if len(vertices):
                x, y = vertices[-2:]
                # 检查顶点是否在有效绘制范围内，若不在则跳过
                if not (0 <= x <= self.file.width * 72
                        and 0 <= y <= self.file.height * 72):
                    continue
                dx, dy = x - lastx, y - lasty
                # 应用变换矩阵并使用标记对象绘制
                output(1, 0, 0, 1, dx, dy, Op.concat_matrix,
                       marker, Op.use_xobject)
                lastx, lasty = x, y
        # 完成绘图操作
        output(Op.grestore)

    def draw_gouraud_triangles(self, gc, points, colors, trans):
        # 绘制高拉德三角形

        # 断言点和颜色数量相同
        assert len(points) == len(colors)
        if len(points) == 0:
            return
        # 断言点数据维度和形状正确
        assert points.ndim == 3
        assert points.shape[1] == 3
        assert points.shape[2] == 2
        # 断言颜色数据维度和形状正确
        assert colors.ndim == 3
        assert colors.shape[1] == 3
        assert colors.shape[2] in (1, 4)

        # 重新调整点的形状以便进行变换
        shape = points.shape
        points = points.reshape((shape[0] * shape[1], 2))
        tpoints = trans.transform(points)
        tpoints = tpoints.reshape(shape)
        # 向文件对象添加高拉德三角形
        name, _ = self.file.addGouraudTriangles(tpoints, colors)
        output = self.file.output

        if colors.shape[2] == 1:
            # 灰度色彩
            gc.set_alpha(1.0)
            self.check_gc(gc)
            output(name, Op.shading)
            return

        alpha = colors[0, 0, 3]
        if np.allclose(alpha, colors[:, :, 3]):
            # 单一 alpha 值
            gc.set_alpha(alpha)
            self.check_gc(gc)
            output(name, Op.shading)
        else:
            # 不同的 alpha 值：使用软掩码
            alpha = colors[:, :, 3][:, :, None]
            _, smask_ob = self.file.addGouraudTriangles(tpoints, alpha)
            gstate = self.file._soft_mask_state(smask_ob)
            output(Op.gsave, gstate, Op.setgstate,
                   name, Op.shading,
                   Op.grestore)
    # 设置文本位置和角度，用于输出文本位置信息到 PDF 文件
    def _setup_textpos(self, x, y, angle, oldx=0, oldy=0, oldangle=0):
        # 如果角度和旧角度均为零，则直接输出相对位置信息
        if angle == oldangle == 0:
            self.file.output(x - oldx, y - oldy, Op.textpos)
        else:
            # 转换角度为弧度
            angle = math.radians(angle)
            # 输出旋转和平移矩阵到 PDF 文件
            self.file.output(math.cos(angle), math.sin(angle),
                             -math.sin(angle), math.cos(angle),
                             x, y, Op.textmatrix)
            # 输出文本位置信息到 PDF 文件
            self.file.output(0, 0, Op.textpos)
    
    # 绘制数学文本，包括解析和绘制数学表达式的功能
    def draw_mathtext(self, gc, x, y, s, prop, angle):
        # TODO: 修正定位和编码
        # 解析数学文本，获取其宽度、高度、下降、字形和矩形列表
        width, height, descent, glyphs, rects = \
            self._text2path.mathtext_parser.parse(s, 72, prop)
    
        # 如果存在超链接，则添加到 PDF 注释
        if gc.get_url() is not None:
            self.file._annotations[-1][1].append(_get_link_annotation(
                gc, x, y, width, height, angle))
    
        # 获取 PDF 字体类型
        fonttype = mpl.rcParams['pdf.fonttype']
    
        # 设置全局变换矩阵，用于整个数学表达式
        a = math.radians(angle)
        self.file.output(Op.gsave)
        self.file.output(math.cos(a), math.sin(a),
                         -math.sin(a), math.cos(a),
                         x, y, Op.concat_matrix)
    
        # 检查图形上下文和其 RGB 颜色
        self.check_gc(gc, gc._rgb)
        prev_font = None, None
        oldx, oldy = 0, 0
        unsupported_chars = []
    
        # 开始输出文本
        self.file.output(Op.begin_text)
        for font, fontsize, num, ox, oy in glyphs:
            # 跟踪字符和字形
            self.file._character_tracker.track_glyph(font, num)
            fontname = font.fname
            if not _font_supports_glyph(fonttype, num):
                # 不支持的字符必须单独处理
                unsupported_chars.append((font, fontsize, ox, oy, num))
            else:
                # 设置文本位置信息
                self._setup_textpos(ox, oy, 0, oldx, oldy)
                oldx, oldy = ox, oy
                if (fontname, fontsize) != prev_font:
                    # 选择字体
                    self.file.output(self.file.fontName(fontname), fontsize,
                                     Op.selectfont)
                    prev_font = fontname, fontsize
                # 显示字符
                self.file.output(self.encode_string(chr(num), fonttype),
                                 Op.show)
        # 结束输出文本
        self.file.output(Op.end_text)
    
        # 输出不支持的字符
        for font, fontsize, ox, oy, num in unsupported_chars:
            self._draw_xobject_glyph(
                font, fontsize, font.get_char_index(num), ox, oy)
    
        # 绘制数学布局中的水平线
        for ox, oy, width, height in rects:
            self.file.output(Op.gsave, ox, oy, width, height,
                             Op.rectangle, Op.fill, Op.grestore)
    
        # 弹出全局变换
        self.file.output(Op.grestore)
    
    # 编码字符串根据不同的 PDF 字体类型
    def encode_string(self, s, fonttype):
        if fonttype in (1, 3):
            return s.encode('cp1252', 'replace')  # 使用 cp1252 编码
        return s.encode('utf-16be', 'replace')  # 使用 utf-16be 编码
    def _draw_xobject_glyph(self, font, fontsize, glyph_idx, x, y):
        """Draw a multibyte character from a Type 3 font as an XObject."""
        # 获取字形的名称
        glyph_name = font.get_glyph_name(glyph_idx)
        # 获取 XObject 的名称
        name = self.file._get_xobject_glyph_name(font.fname, glyph_name)
        # 输出操作序列到 PDF 文件中，绘制 XObject 字形
        self.file.output(
            Op.gsave,                                # 图形状态保存
            0.001 * fontsize, 0, 0, 0.001 * fontsize, # 变换矩阵缩放字形大小
            x, y,                                    # 字形位置坐标
            Op.concat_matrix,                        # 连接矩阵
            Name(name),                              # XObject 字形的名称
            Op.use_xobject,                          # 使用 XObject
            Op.grestore,                             # 图形状态恢复
        )

    def new_gc(self):
        """Return a new GraphicsContextPdf object."""
        # 返回一个新的 GraphicsContextPdf 对象
        return GraphicsContextPdf(self.file)
class GraphicsContextPdf(GraphicsContextBase):
    # GraphicsContextPdf 类，继承自 GraphicsContextBase 类

    def __init__(self, file):
        # 构造函数，初始化 GraphicsContextPdf 对象
        super().__init__()
        # 调用父类的构造函数初始化
        self._fillcolor = (0.0, 0.0, 0.0)
        # 设置填充颜色为黑色
        self._effective_alphas = (1.0, 1.0)
        # 设置有效透明度为完全不透明
        self.file = file
        # 初始化文件属性
        self.parent = None
        # 初始化父对象为 None

    def __repr__(self):
        # 返回对象的详细信息的字符串表示，排除文件和父对象信息
        d = dict(self.__dict__)
        del d['file']
        del d['parent']
        return repr(d)

    def stroke(self):
        """
        返回路径是否需要描边的判断条件。

        在 PDF 中，当线宽大于 0 且透明度大于 0 时，且颜色不是完全透明时才需要描边。
        """
        return (self._linewidth > 0 and self._alpha > 0 and
                (len(self._rgb) <= 3 or self._rgb[3] != 0.0))

    def fill(self, *args):
        """
        返回路径是否需要填充的判断条件。

        可选参数可用于指定替代的填充颜色。需要填充的条件包括使用了填充图案或者颜色不是完全透明。
        """
        if len(args):
            _fillcolor = args[0]
        else:
            _fillcolor = self._fillcolor
        return (self._hatch or
                (_fillcolor is not None and
                 (len(_fillcolor) <= 3 or _fillcolor[3] != 0.0)))

    def paint(self):
        """
        返回适当的 PDF 操作符以使路径被描边、填充或同时进行。
        """
        return Op.paint_path(self.fill(), self.stroke())

    capstyles = {'butt': 0, 'round': 1, 'projecting': 2}
    joinstyles = {'miter': 0, 'round': 1, 'bevel': 2}

    def capstyle_cmd(self, style):
        # 返回设置线端风格的 PDF 操作符列表
        return [self.capstyles[style], Op.setlinecap]

    def joinstyle_cmd(self, style):
        # 返回设置线连接风格的 PDF 操作符列表
        return [self.joinstyles[style], Op.setlinejoin]

    def linewidth_cmd(self, width):
        # 返回设置线宽的 PDF 操作符列表
        return [width, Op.setlinewidth]

    def dash_cmd(self, dashes):
        # 返回设置线段样式的 PDF 操作符列表
        offset, dash = dashes
        if dash is None:
            dash = []
            offset = 0
        return [list(dash), offset, Op.setdash]

    def alpha_cmd(self, alpha, forced, effective_alphas):
        # 返回设置透明度的 PDF 操作符列表
        name = self.file.alphaState(effective_alphas)
        return [name, Op.setgstate]

    def hatch_cmd(self, hatch, hatch_color):
        # 返回设置填充图案的 PDF 操作符列表
        if not hatch:
            if self._fillcolor is not None:
                return self.fillcolor_cmd(self._fillcolor)
            else:
                return [Name('DeviceRGB'), Op.setcolorspace_nonstroke]
        else:
            hatch_style = (hatch_color, self._fillcolor, hatch)
            name = self.file.hatchPattern(hatch_style)
            return [Name('Pattern'), Op.setcolorspace_nonstroke,
                    name, Op.setcolor_nonstroke]
    commands = (
        # 定义一系列的命令元组，每个元组包含属性名称和对应的命令方法
        # 必须首先处理，因为可能会执行弹出操作
        (('_cliprect', '_clippath'), clip_cmd),
        # 处理 alpha 命令
        (('_alpha', '_forced_alpha', '_effective_alphas'), alpha_cmd),
        # 处理 capstyle 命令
        (('_capstyle',), capstyle_cmd),
        # 处理 fillcolor 命令
        (('_fillcolor',), fillcolor_cmd),
        # 处理 joinstyle 命令
        (('_joinstyle',), joinstyle_cmd),
        # 处理 linewidth 命令
        (('_linewidth',), linewidth_cmd),
        # 处理 dashes 命令
        (('_dashes',), dash_cmd),
        # 处理 rgb 命令
        (('_rgb',), rgb_cmd),
        # 必须在 fillcolor 和 rgb 之后处理
        (('_hatch', '_hatch_color'), hatch_cmd),
        )
    def delta(self, other):
        """
        Copy properties of other into self and return PDF commands
        needed to transform *self* into *other*.
        """
        cmds = []  # 初始化一个空列表，用于存储生成的 PDF 命令
        fill_performed = False  # 填充操作是否已执行的标志

        # 遍历 self 的命令列表
        for params, cmd in self.commands:
            different = False  # 不同标志，用于检测属性是否不同

            # 检查每个参数是否在 self 和 other 之间不同
            for p in params:
                ours = getattr(self, p)  # 获取 self 对象的属性值
                theirs = getattr(other, p)  # 获取 other 对象的属性值
                try:
                    if ours is None or theirs is None:
                        different = ours is not theirs
                    else:
                        different = bool(ours != theirs)
                except ValueError:
                    ours = np.asarray(ours)  # 转换为 NumPy 数组
                    theirs = np.asarray(theirs)  # 转换为 NumPy 数组
                    different = (ours.shape != theirs.shape or
                                 np.any(ours != theirs))
                if different:
                    break

            # 如果更新了填充颜色，则需要更新刻度填充
            if params == ('_hatch', '_hatch_color') and fill_performed:
                different = True

            # 如果属性不同，则生成相应的 PDF 命令并更新 self 的属性
            if different:
                if params == ('_fillcolor',):
                    fill_performed = True  # 标记填充操作已执行
                theirs = [getattr(other, p) for p in params]  # 获取 other 对象的属性值列表
                cmds.extend(cmd(self, *theirs))  # 执行命令生成 PDF 命令并添加到 cmds 列表
                for p in params:
                    setattr(self, p, getattr(other, p))  # 更新 self 对象的属性值

        return cmds  # 返回生成的 PDF 命令列表

    def copy_properties(self, other):
        """
        Copy properties of other into self.
        """
        super().copy_properties(other)  # 调用父类方法复制属性
        fillcolor = getattr(other, '_fillcolor', self._fillcolor)  # 获取填充颜色属性值
        effective_alphas = getattr(other, '_effective_alphas',
                                   self._effective_alphas)  # 获取有效透明度属性值
        self._fillcolor = fillcolor  # 更新 self 对象的填充颜色属性值
        self._effective_alphas = effective_alphas  # 更新 self 对象的有效透明度属性值

    def finalize(self):
        """
        Make sure every pushed graphics state is popped.
        """
        cmds = []  # 初始化一个空列表，用于存储生成的 PDF 命令
        while self.parent is not None:
            cmds.extend(self.pop())  # 依次将弹出的图形状态命令添加到 cmds 列表
        return cmds  # 返回生成的 PDF 命令列表
# 定义一个多页 PDF 文件的类 PdfPages
class PdfPages:
    """
    A multi-page PDF file.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> # Initialize:
    >>> with PdfPages('foo.pdf') as pdf:
    ...     # As many times as you like, create a figure fig and save it:
    ...     fig = plt.figure()
    ...     pdf.savefig(fig)
    ...     # When no figure is specified the current figure is saved
    ...     pdf.savefig()

    Notes
    -----
    In reality `PdfPages` is a thin wrapper around `PdfFile`, in order to avoid
    confusion when using `~.pyplot.savefig` and forgetting the format argument.
    """

    _UNSET = object()

    # 初始化方法，创建一个新的 PdfPages 对象
    def __init__(self, filename, keep_empty=_UNSET, metadata=None):
        """
        Create a new PdfPages object.

        Parameters
        ----------
        filename : str or path-like or file-like
            Plots using `PdfPages.savefig` will be written to a file at this location.
            The file is opened when a figure is saved for the first time (overwriting
            any older file with the same name).

        keep_empty : bool, optional
            If set to False, then empty pdf files will be deleted automatically
            when closed.

        metadata : dict, optional
            Information dictionary object (see PDF reference section 10.2.1
            'Document Information Dictionary'), e.g.:
            ``{'Creator': 'My software', 'Author': 'Me', 'Title': 'Awesome'}``.

            The standard keys are 'Title', 'Author', 'Subject', 'Keywords',
            'Creator', 'Producer', 'CreationDate', 'ModDate', and
            'Trapped'. Values have been predefined for 'Creator', 'Producer'
            and 'CreationDate'. They can be removed by setting them to `None`.
        """
        # 存储文件名
        self._filename = filename
        # 存储元数据信息
        self._metadata = metadata
        # 文件对象初始化为空
        self._file = None
        # 如果 keep_empty 被设置且不为 _UNSET，则发出警告
        if keep_empty and keep_empty is not self._UNSET:
            _api.warn_deprecated("3.8", message=(
                "Keeping empty pdf files is deprecated since %(since)s and support "
                "will be removed %(removal)s."))
        # 存储 keep_empty 属性
        self._keep_empty = keep_empty

    # keep_empty 属性，用于兼容性警告
    keep_empty = _api.deprecate_privatize_attribute("3.8")

    # 上下文管理器方法，返回实例本身
    def __enter__(self):
        return self

    # 上下文管理器方法，关闭 PdfPages 对象
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # 确保文件已经打开，若未打开则初始化一个 PdfFile 对象
    def _ensure_file(self):
        if self._file is None:
            self._file = PdfFile(self._filename, metadata=self._metadata)  # init.
        return self._file
    def close(self):
        """
        Finalize this object, making the underlying file a complete
        PDF file.
        """
        # 如果文件对象存在，则执行 finalize() 方法完成文件，并关闭文件
        if self._file is not None:
            self._file.finalize()
            self._file.close()
            self._file = None
        # 如果 _keep_empty 为 True 或未设置，显示弃用警告信息，并创建并关闭空的 PdfFile 对象以触发文件生成
        elif self._keep_empty:  # True *or* UNSET.
            _api.warn_deprecated("3.8", message=(
                "Keeping empty pdf files is deprecated since %(since)s and support "
                "will be removed %(removal)s."))
            PdfFile(self._filename, metadata=self._metadata).close()  # touch the file.

    def infodict(self):
        """
        Return a modifiable information dictionary object
        (see PDF reference section 10.2.1 'Document Information
        Dictionary').
        """
        # 确保文件存在并返回信息字典对象以便修改
        return self._ensure_file().infoDict

    def savefig(self, figure=None, **kwargs):
        """
        Save a `.Figure` to this file as a new page.

        Any other keyword arguments are passed to `~.Figure.savefig`.

        Parameters
        ----------
        figure : `.Figure` or int, default: the active figure
            The figure, or index of the figure, that is saved to the file.
        """
        # 如果 figure 不是 Figure 对象，则获取活动 figure 或指定的 figure
        if not isinstance(figure, Figure):
            if figure is None:
                manager = Gcf.get_active()
            else:
                manager = Gcf.get_fig_manager(figure)
            # 如果 manager 为 None，则抛出 ValueError 异常
            if manager is None:
                raise ValueError(f"No figure {figure}")
            # 将 figure 设置为 manager 的 canvas 的 figure
            figure = manager.canvas.figure
        # 强制使用 pdf 后端保存 figure 到当前 PdfPages 文件中
        figure.savefig(self, format="pdf", backend="pdf", **kwargs)

    def get_pagecount(self):
        """Return the current number of pages in the multipage pdf file."""
        # 确保文件存在并返回当前页面数目
        return len(self._ensure_file().pageList)

    def attach_note(self, text, positionRect=[-100, -100, 0, 0]):
        """
        Add a new text note to the page to be saved next. The optional
        positionRect specifies the position of the new note on the
        page. It is outside the page per default to make sure it is
        invisible on printouts.
        """
        # 确保文件存在并在当前页面添加新的文本注释，指定的 positionRect 参数控制注释在页面上的位置
        self._ensure_file().newTextnote(text, positionRect)
class FigureCanvasPdf(FigureCanvasBase):
    # 继承自FigureCanvasBase的文档字符串

    fixed_dpi = 72
    filetypes = {'pdf': 'Portable Document Format'}

    def get_default_filetype(self):
        # 返回默认文件类型为'pdf'
        return 'pdf'

    def print_pdf(self, filename, *,
                  bbox_inches_restore=None, metadata=None):
        # 打印为PDF格式的方法

        dpi = self.figure.dpi
        self.figure.dpi = 72  # PDF中每英寸有72个点
        width, height = self.figure.get_size_inches()
        if isinstance(filename, PdfPages):
            file = filename._ensure_file()
        else:
            file = PdfFile(filename, metadata=metadata)
        try:
            file.newPage(width, height)  # 创建新页面
            renderer = MixedModeRenderer(
                self.figure, width, height, dpi,
                RendererPdf(file, dpi, height, width),
                bbox_inches_restore=bbox_inches_restore)
            self.figure.draw(renderer)  # 绘制图形到renderer
            renderer.finalize()  # 完成渲染
            if not isinstance(filename, PdfPages):
                file.finalize()  # 完成文件写入
        finally:
            if isinstance(filename, PdfPages):  # 如果是PdfPages，结束当前页面
                file.endStream()
            else:            # 如果不是PdfPages，则关闭文件
                file.close()

    def draw(self):
        self.figure.draw_without_rendering()  # 调用figure对象的无渲染绘制方法
        return super().draw()  # 调用父类的draw方法


FigureManagerPdf = FigureManagerBase  # 设置FigureManagerPdf为FigureManagerBase的别名


@_Backend.export
class _BackendPdf(_Backend):
    FigureCanvas = FigureCanvasPdf  # 将FigureCanvasPdf设置为_BackendPdf类的FigureCanvas
```