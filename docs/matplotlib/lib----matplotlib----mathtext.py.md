# `D:\src\scipysrc\matplotlib\lib\matplotlib\mathtext.py`

```py
r"""
A module for parsing a subset of the TeX math syntax and rendering it to a
Matplotlib backend.

For a tutorial of its usage, see :ref:`mathtext`.  This
document is primarily concerned with implementation details.

The module uses pyparsing_ to parse the TeX expression.

.. _pyparsing: https://pypi.org/project/pyparsing/

The Bakoma distribution of the TeX Computer Modern fonts, and STIX
fonts are supported.  There is experimental support for using
arbitrary fonts, but results may vary without proper tweaking and
metrics for those fonts.
"""

import functools
import logging

import matplotlib as mpl
from matplotlib import _api, _mathtext
from matplotlib.ft2font import LOAD_NO_HINTING
from matplotlib.font_manager import FontProperties
from ._mathtext import (  # noqa: reexported API
    RasterParse, VectorParse, get_unicode_index)

_log = logging.getLogger(__name__)


get_unicode_index.__module__ = __name__

##############################################################################
# MAIN


class MathTextParser:
    _parser = None
    _font_type_mapping = {
        'cm':          _mathtext.BakomaFonts,      # 映射 'cm' 到 Bakoma 字体集
        'dejavuserif': _mathtext.DejaVuSerifFonts, # 映射 'dejavuserif' 到 DejaVu Serif 字体集
        'dejavusans':  _mathtext.DejaVuSansFonts,  # 映射 'dejavusans' 到 DejaVu Sans 字体集
        'stix':        _mathtext.StixFonts,        # 映射 'stix' 到 STIX 字体集
        'stixsans':    _mathtext.StixSansFonts,    # 映射 'stixsans' 到 STIX Sans 字体集
        'custom':      _mathtext.UnicodeFonts,     # 映射 'custom' 到 Unicode 字体集
    }

    def __init__(self, output):
        """
        Create a MathTextParser for the given backend *output*.

        Parameters
        ----------
        output : {"path", "agg"}
            Whether to return a `VectorParse` ("path") or a
            `RasterParse` ("agg", or its synonym "macosx").
        """
        self._output_type = _api.check_getitem(
            {"path": "vector", "agg": "raster", "macosx": "raster"},
            output=output.lower())
    # 解析给定的数学表达式 *s*，在给定的 *dpi* 下进行解析
    # 如果提供了 *prop*，它是一个 `.FontProperties` 对象，指定数学表达式中使用的默认字体
    # 结果被缓存，因此多次使用相同的表达式调用 `parse` 应该很快

    # 复制一份 prop 以便缓存（因为 prop 是可变的），如果 prop 不为 None 的话
    prop = prop.copy() if prop is not None else None

    # 获取文本抗锯齿设置，如果未提供则从全局设置中获取
    antialiased = mpl._val_or_rc(antialiased, 'text.antialiased')

    # 导入后端 agg 的相关模块
    from matplotlib.backends import backend_agg

    # 根据 self._output_type 的值选择加载字形的标志
    load_glyph_flags = {
        "vector": LOAD_NO_HINTING,
        "raster": backend_agg.get_hinting_flag(),
    }[self._output_type]

    # 调用 _parse_cached 方法进行解析，并返回解析结果
    return self._parse_cached(s, dpi, prop, antialiased, load_glyph_flags)


@functools.lru_cache(50)
def _parse_cached(self, s, dpi, prop, antialiased, load_glyph_flags):
    # 如果 prop 为 None，则创建一个默认的 FontProperties 对象
    if prop is None:
        prop = FontProperties()

    # 根据 prop 的数学字体系列获取字体集合类
    fontset_class = _api.check_getitem(
        self._font_type_mapping, fontset=prop.get_math_fontfamily())

    # 根据字体集合类和加载字形标志创建字体集合对象
    fontset = fontset_class(prop, load_glyph_flags)

    # 获取字体大小（以点为单位）
    fontsize = prop.get_size_in_points()

    # 如果 self._parser 为 None，则全局缓存解析器
    if self._parser is None:
        self.__class__._parser = _mathtext.Parser()

    # 使用解析器解析数学表达式 s，返回一个盒子对象
    box = self._parser.parse(s, fontset, fontsize, dpi)

    # 将解析的盒子对象发送到输出处理模块
    output = _mathtext.ship(box)

    # 根据 self._output_type 返回相应类型的输出
    if self._output_type == "vector":
        return output.to_vector()
    elif self._output_type == "raster":
        return output.to_raster(antialiased=antialiased)
def math_to_image(s, filename_or_obj, prop=None, dpi=None, format=None,
                  *, color=None):
    """
    Given a math expression, renders it in a closely-clipped bounding
    box to an image file.

    Parameters
    ----------
    s : str
        A math expression.  The math portion must be enclosed in dollar signs.
    filename_or_obj : str or path-like or file-like
        Where to write the image data.
    prop : `.FontProperties`, optional
        The size and style of the text.
    dpi : float, optional
        The output dpi.  If not set, the dpi is determined as for
        `.Figure.savefig`.
    format : str, optional
        The output format, e.g., 'svg', 'pdf', 'ps' or 'png'.  If not set, the
        format is determined as for `.Figure.savefig`.
    color : str, optional
        Foreground color, defaults to :rc:`text.color`.
    """
    from matplotlib import figure  # 导入 matplotlib 的 figure 模块

    parser = MathTextParser('path')  # 创建 MathTextParser 对象，'path' 为路径参数

    # 解析数学表达式，获取宽度、高度、深度等信息，dpi 默认为 72，使用指定的字体属性
    width, height, depth, _, _ = parser.parse(s, dpi=72, prop=prop)

    # 创建一个 Figure 对象，设置尺寸为数学表达式所需的宽度和高度
    fig = figure.Figure(figsize=(width / 72.0, height / 72.0))

    # 在 Figure 对象上添加文本，位置为 (0, depth/height)，使用指定的字体属性和颜色
    fig.text(0, depth/height, s, fontproperties=prop, color=color)

    # 将 Figure 对象保存为图像文件，可以指定 dpi 和格式
    fig.savefig(filename_or_obj, dpi=dpi, format=format)

    # 返回深度值作为结果
    return depth
```