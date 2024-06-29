# `D:\src\scipysrc\matplotlib\lib\matplotlib\textpath.py`

```py
from collections import OrderedDict
import logging
import urllib.parse

import numpy as np

from matplotlib import _text_helpers, dviread
from matplotlib.font_manager import (
    FontProperties, get_font, fontManager as _fontManager
)
from matplotlib.ft2font import LOAD_NO_HINTING, LOAD_TARGET_LIGHT
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D

_log = logging.getLogger(__name__)


class TextToPath:
    """A class that converts strings to paths."""

    FONT_SCALE = 100.  # 字体缩放比例
    DPI = 72  # DPI 设置

    def __init__(self):
        self.mathtext_parser = MathTextParser('path')  # 创建数学文本解析器对象
        self._texmanager = None  # TexManager 对象初始化为 None

    def _get_font(self, prop):
        """
        Find the `FT2Font` matching font properties *prop*, with its size set.
        """
        filenames = _fontManager._find_fonts_by_props(prop)  # 根据字体属性查找字体文件名
        font = get_font(filenames)  # 获取对应字体文件的 FT2Font 对象
        font.set_size(self.FONT_SCALE, self.DPI)  # 设置字体大小和 DPI
        return font

    def _get_hinting_flag(self):
        return LOAD_NO_HINTING  # 返回字体加载时的提示标志

    def _get_char_id(self, font, ccode):
        """
        Return a unique id for the given font and character-code set.
        """
        return urllib.parse.quote(f"{font.postscript_name}-{ccode:x}")  # 返回字符对应的唯一标识符

    def get_text_width_height_descent(self, s, prop, ismath):
        fontsize = prop.get_size_in_points()  # 获取字体大小（以点为单位）

        if ismath == "TeX":
            return TexManager().get_text_width_height_descent(s, fontsize)  # 如果是 TeX 格式，使用 TexManager 计算文本宽度、高度和下降

        scale = fontsize / self.FONT_SCALE  # 计算缩放比例

        if ismath:
            prop = prop.copy()
            prop.set_size(self.FONT_SCALE)  # 设置字体大小为 FONT_SCALE
            width, height, descent, *_ = \
                self.mathtext_parser.parse(s, 72, prop)  # 解析数学文本，返回宽度、高度、下降和其他信息
            return width * scale, height * scale, descent * scale

        font = self._get_font(prop)  # 获取字体对象
        font.set_text(s, 0.0, flags=LOAD_NO_HINTING)  # 设置字体文本和加载标志
        w, h = font.get_width_height()  # 获取文本宽度和高度
        w /= 64.0  # 转换为标准像素
        h /= 64.0
        d = font.get_descent()  # 获取文本下降
        d /= 64.0
        return w * scale, h * scale, d * scale  # 返回缩放后的宽度、高度和下降
    # 定义一个方法，将文本 *s* 转换为 matplotlib.path.Path 的顶点和代码元组
    def get_text_path(self, prop, s, ismath=False):
        """
        Convert text *s* to path (a tuple of vertices and codes for
        matplotlib.path.Path).

        Parameters
        ----------
        prop : `~matplotlib.font_manager.FontProperties`
            The font properties for the text.
        s : str
            The text to be converted.
        ismath : {False, True, "TeX"}
            If True, use mathtext parser.  If "TeX", use tex for rendering.

        Returns
        -------
        verts : list
            A list of arrays containing the (x, y) coordinates of the vertices.
        codes : list
            A list of path codes.

        Examples
        --------
        Create a list of vertices and codes from a text, and create a `.Path`
        from those::

            from matplotlib.path import Path
            from matplotlib.text import TextToPath
            from matplotlib.font_manager import FontProperties

            fp = FontProperties(family="Comic Neue", style="italic")
            verts, codes = TextToPath().get_text_path(fp, "ABC")
            path = Path(verts, codes, closed=False)

        Also see `TextPath` for a more direct way to create a path from a text.
        """
        # 根据不同的 ismath 参数值选择合适的方法获取字形信息、映射和矩形列表
        if ismath == "TeX":
            glyph_info, glyph_map, rects = self.get_glyphs_tex(prop, s)
        elif not ismath:
            font = self._get_font(prop)
            glyph_info, glyph_map, rects = self.get_glyphs_with_font(font, s)
        else:
            glyph_info, glyph_map, rects = self.get_glyphs_mathtext(prop, s)

        # 初始化空的顶点和代码列表
        verts, codes = [], []

        # 遍历字形信息列表，获取顶点和代码，根据字形信息中的位置和缩放进行调整
        for glyph_id, xposition, yposition, scale in glyph_info:
            verts1, codes1 = glyph_map[glyph_id]
            verts.extend(verts1 * scale + [xposition, yposition])
            codes.extend(codes1)

        # 将矩形列表的顶点和代码添加到总的顶点和代码列表中
        for verts1, codes1 in rects:
            verts.extend(verts1)
            codes.extend(codes1)

        # 确保空字符串或只包含空格和换行符的文本也能返回有效或空的路径
        if not verts:
            verts = np.empty((0, 2))

        # 返回顶点和代码列表作为结果
        return verts, codes
    def get_glyphs_with_font(self, font, s, glyph_map=None,
                             return_new_glyphs_only=False):
        """
        Convert string *s* to vertices and codes using the provided ttf font.
        """

        # 如果 glyph_map 为 None，则创建一个有序字典
        if glyph_map is None:
            glyph_map = OrderedDict()

        # 根据 return_new_glyphs_only 参数确定是否创建新的 glyph_map_new 字典
        if return_new_glyphs_only:
            glyph_map_new = OrderedDict()
        else:
            glyph_map_new = glyph_map

        # 初始化空列表用于存储 x 坐标和字形 ID
        xpositions = []
        glyph_ids = []

        # 遍历 _text_helpers.layout(s, font) 返回的对象列表
        for item in _text_helpers.layout(s, font):
            # 获取字符的 ID
            char_id = self._get_char_id(item.ft_object, ord(item.char))
            glyph_ids.append(char_id)
            xpositions.append(item.x)
            # 如果字符 ID 不在 glyph_map 中，则将其加入 glyph_map_new 字典
            if char_id not in glyph_map:
                glyph_map_new[char_id] = item.ft_object.get_path()

        # 初始化 ypositions 列表，填充为 0
        ypositions = [0] * len(xpositions)
        # 初始化 sizes 列表，填充为 1.0
        sizes = [1.] * len(xpositions)

        # 初始化 rects 列表
        rects = []

        # 返回结果元组，包含字形 ID、坐标信息、字形数据字典和矩形列表
        return (list(zip(glyph_ids, xpositions, ypositions, sizes)),
                glyph_map_new, rects)

    def get_glyphs_mathtext(self, prop, s, glyph_map=None,
                            return_new_glyphs_only=False):
        """
        Parse mathtext string *s* and convert it to a (vertices, codes) pair.
        """

        # 复制 prop 对象，设置其大小为 FONT_SCALE
        prop = prop.copy()
        prop.set_size(self.FONT_SCALE)

        # 使用 mathtext_parser 解析 s 字符串，获取宽度、高度、下降、字形和矩形列表
        width, height, descent, glyphs, rects = self.mathtext_parser.parse(
            s, self.DPI, prop)

        # 如果 glyph_map 为 None，则创建一个有序字典
        if not glyph_map:
            glyph_map = OrderedDict()

        # 根据 return_new_glyphs_only 参数确定是否创建新的 glyph_map_new 字典
        if return_new_glyphs_only:
            glyph_map_new = OrderedDict()
        else:
            glyph_map_new = glyph_map

        # 初始化空列表用于存储 x 坐标、y 坐标、字形 ID 和尺寸
        xpositions = []
        ypositions = []
        glyph_ids = []
        sizes = []

        # 遍历 glyphs 列表中的字形数据
        for font, fontsize, ccode, ox, oy in glyphs:
            # 获取字符的 ID
            char_id = self._get_char_id(font, ccode)
            # 如果字符 ID 不在 glyph_map 中，则加载字符路径并加入 glyph_map_new 字典
            if char_id not in glyph_map:
                font.clear()
                font.set_size(self.FONT_SCALE, self.DPI)
                font.load_char(ccode, flags=LOAD_NO_HINTING)
                glyph_map_new[char_id] = font.get_path()

            # 添加 x 和 y 坐标、字形 ID 和尺寸到相应列表中
            xpositions.append(ox)
            ypositions.append(oy)
            glyph_ids.append(char_id)
            size = fontsize / self.FONT_SCALE
            sizes.append(size)

        # 初始化空列表用于存储矩形信息
        myrects = []
        # 遍历 rects 列表中的矩形数据，构建顶点和代码对
        for ox, oy, w, h in rects:
            vert1 = [(ox, oy), (ox, oy + h), (ox + w, oy + h),
                     (ox + w, oy), (ox, oy), (0, 0)]
            code1 = [Path.MOVETO,
                     Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO,
                     Path.CLOSEPOLY]
            myrects.append((vert1, code1))

        # 返回结果元组，包含字形 ID、坐标信息、字形数据字典和矩形列表
        return (list(zip(glyph_ids, xpositions, ypositions, sizes)),
                glyph_map_new, myrects)
    def get_glyphs_tex(self, prop, s, glyph_map=None,
                       return_new_glyphs_only=False):
        """Convert the string *s* to vertices and codes using usetex mode."""
        # 使用 usetex 模式将字符串 *s* 转换为顶点和代码，大部分参考自 PDF 后端。

        # 通过 TexManager 创建 DVI 文件，使用 FONT_SCALE 进行缩放
        dvifile = TexManager().make_dvi(s, self.FONT_SCALE)
        
        # 使用 dviread 库打开 DVI 文件，并以指定 DPI 解析
        with dviread.Dvi(dvifile, self.DPI) as dvi:
            page, = dvi  # 从 DVI 文件中获取页面信息

        # 如果 glyph_map 未提供，则使用 OrderedDict 创建空的字典
        if glyph_map is None:
            glyph_map = OrderedDict()

        # 如果 return_new_glyphs_only 为 True，则创建新的有序字典；否则使用提供的 glyph_map
        if return_new_glyphs_only:
            glyph_map_new = OrderedDict()
        else:
            glyph_map_new = glyph_map

        # 初始化存储 glyph_ids、xpositions、ypositions 和 sizes 的列表
        glyph_ids, xpositions, ypositions, sizes = [], [], [], []

        # 遍历页面中的文本信息，获取字体信息并设置字符ID
        for text in page.text:
            font = get_font(text.font_path)
            char_id = self._get_char_id(font, text.glyph)
            
            # 如果 char_id 不在 glyph_map 中，则加载字形数据并存储到 glyph_map_new 中
            if char_id not in glyph_map:
                font.clear()
                font.set_size(self.FONT_SCALE, self.DPI)
                glyph_name_or_index = text.glyph_name_or_index
                
                # 根据字形名称或索引加载字形数据到字体对象中
                if isinstance(glyph_name_or_index, str):
                    index = font.get_name_index(glyph_name_or_index)
                    font.load_glyph(index, flags=LOAD_TARGET_LIGHT)
                elif isinstance(glyph_name_or_index, int):
                    self._select_native_charmap(font)
                    font.load_char(
                        glyph_name_or_index, flags=LOAD_TARGET_LIGHT)
                else:  # 不应该出现的情况
                    raise TypeError(f"Glyph spec of unexpected type: "
                                    f"{glyph_name_or_index!r}")
                
                # 将字形路径存储到 glyph_map_new 中
                glyph_map_new[char_id] = font.get_path()

            # 将 char_id、text.x、text.y 和 text.font_size 的相关信息存储到相应列表中
            glyph_ids.append(char_id)
            xpositions.append(text.x)
            ypositions.append(text.y)
            sizes.append(text.font_size / self.FONT_SCALE)

        # 初始化空列表 myrects，用于存储页面中的矩形区域信息
        myrects = []

        # 遍历页面中的 boxes，获取其顶点和代码，添加到 myrects 列表中
        for ox, oy, h, w in page.boxes:
            vert1 = [(ox, oy), (ox + w, oy), (ox + w, oy + h),
                     (ox, oy + h), (ox, oy), (0, 0)]
            code1 = [Path.MOVETO,
                     Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO,
                     Path.CLOSEPOLY]
            myrects.append((vert1, code1))

        # 返回包含 glyph_ids、glyph_map_new 和 myrects 的元组作为结果
        return (list(zip(glyph_ids, xpositions, ypositions, sizes)),
                glyph_map_new, myrects)

    @staticmethod
    def _select_native_charmap(font):
        # 选择本地字符映射表（通常是 Adobe 字符映射表）。
        for charmap_code in [
                1094992451,  # ADOBE_CUSTOM.
                1094995778,  # ADOBE_STANDARD.
        ]:
            try:
                font.select_charmap(charmap_code)
            except (ValueError, RuntimeError):
                pass
            else:
                break
        else:
            _log.warning("No supported encoding in font (%s).", font.fname)
# 创建一个 TextToPath 的实例，用于处理文本转路径的操作
text_to_path = TextToPath()

# 定义一个 TextPath 类，继承自 Path 类，用于将文本转换为路径表示
class TextPath(Path):
    """
    Create a path from the text.
    """

    def __init__(self, xy, s, size=None, prop=None,
                 _interpolation_steps=1, usetex=False):
        r"""
        Create a path from the text. Note that it simply is a path,
        not an artist. You need to use the `.PathPatch` (or other artists)
        to draw this path onto the canvas.

        Parameters
        ----------
        xy : tuple or array of two float values
            Position of the text. For no offset, use ``xy=(0, 0)``.

        s : str
            The text to convert to a path.

        size : float, optional
            Font size in points. Defaults to the size specified via the font
            properties *prop*.

        prop : `~matplotlib.font_manager.FontProperties`, optional
            Font property. If not provided, will use a default
            `.FontProperties` with parameters from the
            :ref:`rcParams<customizing-with-dynamic-rc-settings>`.

        _interpolation_steps : int, optional
            (Currently ignored)

        usetex : bool, default: False
            Whether to use tex rendering.

        Examples
        --------
        The following creates a path from the string "ABC" with Helvetica
        font face; and another path from the latex fraction 1/2::

            from matplotlib.text import TextPath
            from matplotlib.font_manager import FontProperties

            fp = FontProperties(family="Helvetica", style="italic")
            path1 = TextPath((12, 12), "ABC", size=12, prop=fp)
            path2 = TextPath((0, 0), r"$\frac{1}{2}$", size=12, usetex=True)

        Also see :doc:`/gallery/text_labels_and_annotations/demo_text_path`.
        """
        # 循环引用
        from matplotlib.text import Text

        # 从参数中获取字体属性对象，如果 prop 未提供则使用默认的 FontProperties
        prop = FontProperties._from_any(prop)
        # 如果未指定字体大小，则使用属性中定义的大小
        if size is None:
            size = prop.get_size_in_points()

        # 设置文本的位置坐标
        self._xy = xy
        # 设置文本的大小
        self.set_size(size)

        # 缓存的顶点数组，初始为 None
        self._cached_vertices = None
        # 对文本进行预处理，处理数学表达式
        s, ismath = Text(usetex=usetex)._preprocess_math(s)
        # 调用 text_to_path 的方法获取文本路径，并用这些路径数据初始化 Path 对象
        super().__init__(
            *text_to_path.get_text_path(prop, s, ismath=ismath),
            _interpolation_steps=_interpolation_steps,
            readonly=True)
        # 是否应该简化路径的标志，默认为 False
        self._should_simplify = False

    def set_size(self, size):
        """Set the text size."""
        # 设置文本的大小
        self._size = size
        self._invalid = True

    def get_size(self):
        """Get the text size."""
        # 获取文本的大小
        return self._size

    @property
    def vertices(self):
        """
        Return the cached path after updating it if necessary.
        """
        # 重新验证路径，并返回缓存的顶点数组
        self._revalidate_path()
        return self._cached_vertices

    @property
    def codes(self):
        """
        Return the codes
        """
        # 返回路径的代码
        return self._codes
    def _revalidate_path(self):
        """
        Update the path if necessary.

        The path for the text is initially create with the font size of
        `.FONT_SCALE`, and this path is rescaled to other size when necessary.
        """
        # 检查是否需要重新验证路径
        if self._invalid or self._cached_vertices is None:
            # 创建仿射变换对象，缩放当前大小至标准字体大小的比例，并平移至指定位置
            tr = (Affine2D()
                  .scale(self._size / text_to_path.FONT_SCALE)
                  .translate(*self._xy))
            # 使用仿射变换对象对顶点坐标进行转换，更新缓存的顶点数据
            self._cached_vertices = tr.transform(self._vertices)
            # 设置缓存顶点数据为只读状态，防止意外修改
            self._cached_vertices.flags.writeable = False
            # 标记路径验证已完成
            self._invalid = False
```