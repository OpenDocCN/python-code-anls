# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\_backend_pdf_ps.py`

```py
"""
Common functionality between the PDF and PS backends.
"""

# 导入所需模块和库
from io import BytesIO
import functools

from fontTools import subset  # 导入字体子集化工具

import matplotlib as mpl
from .. import font_manager, ft2font  # 导入本地的字体管理器和ft2font模块
from .._afm import AFM  # 导入AFM模块
from ..backend_bases import RendererBase  # 导入渲染器基类


@functools.lru_cache(50)
# 使用functools的lru_cache装饰器，缓存函数调用结果，最多缓存50个不同的调用结果
def _cached_get_afm_from_fname(fname):
    with open(fname, "rb") as fh:
        return AFM(fh)  # 从文件对象创建AFM对象


def get_glyphs_subset(fontfile, characters):
    """
    Subset a TTF font

    Reads the named fontfile and restricts the font to the characters.
    Returns a serialization of the subset font as file-like object.

    Parameters
    ----------
    fontfile : str
        Path to the font file
    characters : str
        Continuous set of characters to include in subset
    """

    options = subset.Options(glyph_names=True, recommended_glyphs=True)

    # Prevent subsetting extra tables.
    options.drop_tables += [
        'FFTM',  # FontForge Timestamp.
        'PfEd',  # FontForge personal table.
        'BDF',  # X11 BDF header.
        'meta',  # Metadata stores design/supported languages (meaningless for subsets).
    ]

    # if fontfile is a ttc, specify font number
    if fontfile.endswith(".ttc"):
        options.font_number = 0

    with subset.load_font(fontfile, options) as font:
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(text=characters)  # 填充子集器以包含指定字符
        subsetter.subset(font)  # 创建指定字符的字体子集
        fh = BytesIO()  # 创建一个BytesIO对象
        font.save(fh, reorderTables=False)  # 将字体子集保存到BytesIO对象中，不重新排序表
        return fh  # 返回保存字体子集的BytesIO对象


class CharacterTracker:
    """
    Helper for font subsetting by the pdf and ps backends.

    Maintains a mapping of font paths to the set of character codepoints that
    are being used from that font.
    """

    def __init__(self):
        self.used = {}  # 初始化字典，用于记录每个字体路径和所使用的字符集

    def track(self, font, s):
        """Record that string *s* is being typeset using font *font*."""
        char_to_font = font._get_fontmap(s)  # 获取字符串s中每个字符使用的字体映射
        for _c, _f in char_to_font.items():
            self.used.setdefault(_f.fname, set()).add(ord(_c))  # 记录每个字符对应的字体路径和字符编码

    def track_glyph(self, font, glyph):
        """Record that codepoint *glyph* is being typeset using font *font*."""
        self.used.setdefault(font.fname, set()).add(glyph)  # 记录使用特定字体和字符编码的图元


class RendererPDFPSBase(RendererBase):
    # The following attributes must be defined by the subclasses:
    # - _afm_font_dir
    # - _use_afm_rc_name

    def __init__(self, width, height):
        super().__init__()  # 调用父类的初始化方法
        self.width = width  # 设置渲染器宽度
        self.height = height  # 设置渲染器高度

    def flipy(self):
        # docstring inherited
        return False  # 返回False，PDF和PS中y轴从底向上增长

    def option_scale_image(self):
        # docstring inherited
        return True  # 返回True，PDF和PS支持任意图像缩放

    def option_image_nocomposite(self):
        # docstring inherited
        # Decide whether to composite image based on rcParam value.
        return not mpl.rcParams["image.composite_image"]  # 根据rcParam值决定是否进行图像合成
    # 继承的文档字符串中描述的方法，返回画布的宽度和高度，单位为点（1 英寸 = 72 点）
    def get_canvas_width_height(self):
        # 返回画布宽度和高度各自乘以 72.0 后的值
        return self.width * 72.0, self.height * 72.0

    # 继承的文档字符串中描述的方法，返回文本的宽度、高度和下降值
    def get_text_width_height_descent(self, s, prop, ismath):
        # 如果 ismath 为 "TeX"，调用父类的方法获取文本的宽度、高度和下降值
        if ismath == "TeX":
            return super().get_text_width_height_descent(s, prop, ismath)
        # 如果 ismath 为真，则使用 mathtext_parser 解析字符串 s，并返回宽度、高度和深度
        elif ismath:
            parse = self._text2path.mathtext_parser.parse(s, 72, prop)
            return parse.width, parse.height, parse.depth
        # 如果使用 AFM 字体渲染，根据属性 prop 获取 AFM 字体文件的边界框和下降值，按比例缩放后返回
        elif mpl.rcParams[self._use_afm_rc_name]:
            font = self._get_font_afm(prop)
            l, b, w, h, d = font.get_str_bbox_and_descent(s)
            scale = prop.get_size_in_points() / 1000
            w *= scale
            h *= scale
            d *= scale
            return w, h, d
        # 否则，使用 TTF 字体渲染，设置文本内容后获取宽度和高度，再获取下降值，按比例缩放后返回
        else:
            font = self._get_font_ttf(prop)
            font.set_text(s, 0.0, flags=ft2font.LOAD_NO_HINTING)
            w, h = font.get_width_height()
            d = font.get_descent()
            scale = 1 / 64
            w *= scale
            h *= scale
            d *= scale
            return w, h, d

    # 根据属性 prop 获取 AFM 字体文件名，并从指定目录 self._afm_font_dir 中查找该字体文件
    def _get_font_afm(self, prop):
        fname = font_manager.findfont(
            prop, fontext="afm", directory=self._afm_font_dir)
        return _cached_get_afm_from_fname(fname)

    # 根据属性 prop 获取 TTF 字体文件名，并从字体管理器中查找匹配的字体文件名列表 fnames，再获取相应的字体对象
    def _get_font_ttf(self, prop):
        fnames = font_manager.fontManager._find_fonts_by_props(prop)
        font = font_manager.get_font(fnames)
        # 清除字体缓存，设置字体大小为 prop 中指定的大小（以点为单位），72 为 DPI
        font.clear()
        font.set_size(prop.get_size_in_points(), 72)
        return font
```