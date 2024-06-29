# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_svg.py`

```
import base64
import codecs
import datetime
import gzip
import hashlib
from io import BytesIO
import itertools
import logging
import os
import re
import uuid

import numpy as np
from PIL import Image

import matplotlib as mpl
from matplotlib import cbook, font_manager as fm
from matplotlib.backend_bases import (
     _Backend, FigureCanvasBase, FigureManagerBase, RendererBase)
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.colors import rgb2hex
from matplotlib.dates import UTC
from matplotlib.path import Path
from matplotlib import _path
from matplotlib.transforms import Affine2D, Affine2DBase


_log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# SimpleXMLWriter class
#
# Based on an original by Fredrik Lundh, but modified here to:
#   1. Support modern Python idioms
#   2. Remove encoding support (it's handled by the file writer instead)
#   3. Support proper indentation
#   4. Minify things a little bit

# --------------------------------------------------------------------
# The SimpleXMLWriter module is
#
# Copyright (c) 2001-2004 by Fredrik Lundh
#
# By obtaining, using, and/or copying this software and/or its
# associated documentation, you agree that you have read, understood,
# and will comply with the following terms and conditions:
#
# Permission to use, copy, modify, and distribute this software and
# its associated documentation for any purpose and without fee is
# hereby granted, provided that the above copyright notice appears in
# all copies, and that both that copyright notice and this permission
# notice appear in supporting documentation, and that the name of
# Secret Labs AB or the author not be used in advertising or publicity
# pertaining to distribution of the software without specific, written
# prior permission.
#
# SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD
# TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANT-
# ABILITY AND FITNESS.  IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR
# BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THIS SOFTWARE.
# --------------------------------------------------------------------

# 函数用于转义 CDATA 中的特殊字符
def _escape_cdata(s):
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    return s

# 正则表达式对象，用于转义 XML 注释中的特殊字符
_escape_xml_comment = re.compile(r'-(?=-)')

# 函数用于转义 XML 注释中的特殊字符
def _escape_comment(s):
    s = _escape_cdata(s)
    return _escape_xml_comment.sub('- ', s)

# 函数用于转义 XML 属性值中的特殊字符
def _escape_attrib(s):
    s = s.replace("&", "&amp;")
    s = s.replace("'", "&apos;")
    s = s.replace('"', "&quot;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    return s

# 未完成的函数定义，用于转义 XML 属性值中的特殊字符
def _quote_escape_attrib(s):
    # 如果字符串 s 中不包含双引号，则返回双引号包裹的转义后的 s；否则，如果不包含单引号，则返回单引号包裹的转义后的 s；否则，返回双引号包裹的转义后的 s。
    return ('"' + _escape_cdata(s) + '"' if '"' not in s else
            "'" + _escape_cdata(s) + "'" if "'" not in s else
            '"' + _escape_attrib(s) + '"')
def _short_float_fmt(x):
    """
    Create a short string representation of a float, which is %f
    formatting with trailing zeros and the decimal point removed.
    """
    # Format float 'x' to a string without trailing zeros and the decimal point if unnecessary
    return f'{x:f}'.rstrip('0').rstrip('.')


class XMLWriter:
    """
    Parameters
    ----------
    file : writable text file-like object
    """

    def __init__(self, file):
        # Assign file.write method to self.__write for writing to the provided file object
        self.__write = file.write
        # If file object has 'flush' attribute, assign its flush method to self.flush for flushing output
        if hasattr(file, "flush"):
            self.flush = file.flush
        # Initialize __open to 0 indicating no start tag is open
        self.__open = 0  # true if start tag is open
        # Initialize __tags as empty list to hold XML tags
        self.__tags = []
        # Initialize __data as empty list to accumulate character data for XML elements
        self.__data = []
        # Initialize __indentation with 64 spaces for XML indentation
        self.__indentation = " " * 64

    def __flush(self, indent=True):
        # Flush internal buffers: write open tag closing and accumulated character data
        if self.__open:
            if indent:
                self.__write(">\n")
            else:
                self.__write(">")
            self.__open = 0
        if self.__data:
            data = ''.join(self.__data)
            self.__write(_escape_cdata(data))
            self.__data = []

    def start(self, tag, attrib={}, **extra):
        """
        Open a new element.  Attributes can be given as keyword
        arguments, or as a string/string dictionary. The method returns
        an opaque identifier that can be passed to the :meth:`close`
        method, to close all open elements up to and including this one.

        Parameters
        ----------
        tag
            Element tag.
        attrib
            Attribute dictionary.  Alternatively, attributes can be given as
            keyword arguments.

        Returns
        -------
        An element identifier.
        """
        # Flush any existing data before starting a new XML element
        self.__flush()
        # Escape tag name for XML
        tag = _escape_cdata(tag)
        # Clear accumulated data for the current element
        self.__data = []
        # Append the escaped tag name to __tags list
        self.__tags.append(tag)
        # Write indentation and start tag with attributes
        self.__write(self.__indentation[:len(self.__tags) - 1])
        self.__write(f"<{tag}")
        for k, v in {**attrib, **extra}.items():
            if v:
                # Escape attribute key and value for XML
                k = _escape_cdata(k)
                v = _quote_escape_attrib(v)
                self.__write(f' {k}={v}')
        self.__open = 1  # Set open tag flag to true
        return len(self.__tags) - 1

    def comment(self, comment):
        """
        Add a comment to the output stream.

        Parameters
        ----------
        comment : str
            Comment text.
        """
        # Flush any accumulated data before writing the comment
        self.__flush()
        # Write indentation and formatted comment
        self.__write(self.__indentation[:len(self.__tags)])
        self.__write(f"<!-- {_escape_comment(comment)} -->\n")

    def data(self, text):
        """
        Add character data to the output stream.

        Parameters
        ----------
        text : str
            Character data.
        """
        # Append character data to __data list for later writing
        self.__data.append(text)
    def end(self, tag=None, indent=True):
        """
        Close the current XML element.

        Parameters
        ----------
        tag
            Optional. Tag name of the element to close. Must match the last opened tag.
        indent : bool, default: True
            Flag indicating whether to indent the closing tag.

        Raises
        ------
        AssertionError
            If `tag` is specified but doesn't match the last opened tag.
            If there are no open tags to close.

        Notes
        -----
        If `tag` is not provided, closes the most recently opened element.
        """
        if tag:
            # Check if there are open tags and assert that the specified tag matches the last opened tag
            assert self.__tags, f"unbalanced end({tag})"
            assert _escape_cdata(tag) == self.__tags[-1], \
                f"expected end({self.__tags[-1]}), got {tag}"
        else:
            # Assert that there are open tags to close
            assert self.__tags, "unbalanced end()"
        
        # Pop the last opened tag
        tag = self.__tags.pop()
        
        # If there is pending data, flush it
        if self.__data:
            self.__flush(indent)
        elif self.__open:
            # If the element was self-closed, write the self-closing tag
            self.__open = 0
            self.__write("/>\n")
            return
        
        # If indent flag is True, write indentation based on the number of open tags
        if indent:
            self.__write(self.__indentation[:len(self.__tags)])
        
        # Write the closing tag
        self.__write(f"</{tag}>\n")

    def close(self, id):
        """
        Close elements up to (and including) the element identified by the given identifier.

        Parameters
        ----------
        id
            Identifier of the element to close, obtained from the :meth:`start` method.

        Notes
        -----
        This method calls :meth:`end` iteratively to close elements until it reaches
        the element identified by `id`.
        """
        while len(self.__tags) > id:
            self.end()

    def element(self, tag, text=None, attrib={}, **extra):
        """
        Add a complete XML element.

        Parameters
        ----------
        tag
            Tag name of the element.
        text
            Optional. Text content of the element.
        attrib : dict, optional
            Optional attributes for the element.
        **extra
            Additional keyword arguments treated as attributes.

        Notes
        -----
        This method sequentially calls :meth:`start`, :meth:`data`, and :meth:`end`
        to add a complete XML element to the output.
        """
        self.start(tag, attrib, **extra)
        if text:
            self.data(text)
        self.end(indent=False)

    def flush(self):
        """
        Placeholder method to flush the output stream.

        Notes
        -----
        This method is intended to be replaced by the constructor or subclasses
        to perform actual flushing of the output stream.
        """
        pass  # replaced by the constructor
# 创建一个私有函数 `_generate_transform`，用于生成转换操作的字符串表示
def _generate_transform(transform_list):
    # 初始化一个空列表，用于存储转换操作的各个部分
    parts = []
    # 遍历转换列表中的每对类型和数值
    for type, value in transform_list:
        # 检查是否为缩放操作且值为 (1,) 或 (1, 1)，或者是否为平移操作且值为 (0, 0)，或者是否为旋转操作且值为 (0,)
        if (type == 'scale' and (value == (1,) or value == (1, 1))
                or type == 'translate' and value == (0, 0)
                or type == 'rotate' and value == (0,)):
            # 如果满足上述条件，则跳过该操作
            continue
        # 如果是矩阵操作且值是 Affine2DBase 类的实例，则将其转换为数值形式
        if type == 'matrix' and isinstance(value, Affine2DBase):
            value = value.to_values()
        # 将操作类型和对应的值格式化成字符串，并添加到 parts 列表中
        parts.append('{}({})'.format(
            type, ' '.join(_short_float_fmt(x) for x in value)))
    # 返回所有转换操作的字符串表示，用空格连接
    return ' '.join(parts)


# 创建一个私有函数 `_generate_css`，用于生成 CSS 样式字符串
def _generate_css(attrib):
    # 将字典 attrib 中的键值对转换为样式字符串，每个键值对用分号分隔
    return "; ".join(f"{k}: {v}" for k, v in attrib.items())


# 创建一个全局变量 `_capstyle_d`，用于将一些线段末端风格描述转换为统一的风格字符串
_capstyle_d = {'projecting': 'square', 'butt': 'butt', 'round': 'round'}


# 创建一个私有函数 `_check_is_str`，用于检查参数 info 是否为字符串类型，若不是则抛出 TypeError
def _check_is_str(info, key):
    if not isinstance(info, str):
        raise TypeError(f'Invalid type for {key} metadata. Expected str, not '
                        f'{type(info)}.')


# 创建一个私有函数 `_check_is_iterable_of_str`，用于检查参数 infos 是否为字符串迭代类型，若不是则抛出 TypeError
def _check_is_iterable_of_str(infos, key):
    if np.iterable(infos):
        for info in infos:
            if not isinstance(info, str):
                raise TypeError(f'Invalid type for {key} metadata. Expected '
                                f'iterable of str, not {type(info)}.')
    else:
        raise TypeError(f'Invalid type for {key} metadata. Expected str or '
                        f'iterable of str, not {type(infos)}.')


# 创建一个名为 RendererSVG 的类，继承自 RendererBase 类
class RendererSVG(RendererBase):
    # 定义初始化方法，接受多个参数来设置 SVG 渲染器的各种属性和初始状态
    def __init__(self, width, height, svgwriter, basename=None, image_dpi=72,
                 *, metadata=None):
        # 调用父类 RendererBase 的初始化方法
        super().__init__()
        # 设置 SVG 渲染器的宽度和高度
        self.width = width
        self.height = height
        # 设置 SVG 渲染器的写入器（即 XMLWriter 对象）
        self.writer = XMLWriter(svgwriter)
        # 设置 SVG 渲染器的图像 DPI（用于栅格化操作）
        self.image_dpi = image_dpi  # 实际栅格化操作使用的 DPI
        
        # 如果未提供 basename，则尝试从 svgwriter 中获取文件名，并确保其为字符串类型
        if basename is None:
            basename = getattr(svgwriter, "name", "")
            if not isinstance(basename, str):
                basename = ""
        # 设置 SVG 渲染器的基本名称
        self.basename = basename
        
        # 初始化一些内部字典和计数器等属性，用于管理 SVG 渲染的状态和对象
        self._groupd = {}  # 组对象字典
        self._image_counter = itertools.count()  # 图像计数器
        self._clipd = {}  # 裁剪路径字典
        self._markers = {}  # 标记对象字典
        self._path_collection_id = 0  # 路径集合 ID 计数器
        self._hatchd = {}  # 纹理填充字典
        self._has_gouraud = False  # 是否使用 Gouraud 着色
        self._n_gradients = 0  # 渐变数量计数器
        
        # 初始化完成后，调用内部方法来进一步初始化 SVG 渲染器
        self._init()
        
        # 调用父类的初始化方法，执行一些额外的初始化工作
        super().__init__()
        
        # 创建一个字典 `_glyph_map`，用于映射字符到其渲染数据的关系
        self._glyph_map = dict()
        # 将高度和宽度转换为短浮点数格式的字符串，用于 SVG 的视图框和尺寸设定
        str_height = _short_float_fmt(height)
        str_width = _short_float_fmt(width)
        # 在 SVG 写入器中写入文档头部声明
        svgwriter.write(svgProlog)
        # 开始写入 SVG 标签，并设置其宽度、高度、视图框、XML 命名空间等属性
        self._start_id = self.writer.start(
            'svg',
            width=f'{str_width}pt',
            height=f'{str_height}pt',
            viewBox=f'0 0 {str_width} {str_height}',
            xmlns="http://www.w3.org/2000/svg",
            version="1.1",
            attrib={'xmlns:xlink': "http://www.w3.org/1999/xlink"})
        # 写入元数据信息到 SVG 文件中
        self._write_metadata(metadata)
        # 写入默认样式信息到 SVG 文件中
        self._write_default_style()
    
    # 定义一个 finalize 方法，用于在 SVG 渲染结束时进行一些收尾工作
    def finalize(self):
        # 写入所有裁剪路径的信息到 SVG 文件中
        self._write_clips()
        # 写入所有纹理填充的信息到 SVG 文件中
        self._write_hatches()
        # 关闭 SVG 文件的根标签，并完成文件写入操作
        self.writer.close(self._start_id)
        # 清空并刷新写入器的缓冲区
        self.writer.flush()
    # 写入默认样式定义
    def _write_default_style(self):
        # 获取写入器对象
        writer = self.writer
        # 生成默认 CSS 样式
        default_style = _generate_css({
            'stroke-linejoin': 'round',
            'stroke-linecap': 'butt'})
        # 开始写入 <defs> 标签
        writer.start('defs')
        # 在 <defs> 中添加 <style> 标签，定义全局样式
        writer.element('style', type='text/css', text='*{%s}' % default_style)
        # 结束 <defs> 标签
        writer.end('defs')

    # 创建唯一标识符
    def _make_id(self, type, content):
        # 获取哈希盐值，用于生成唯一标识符
        salt = mpl.rcParams['svg.hashsalt']
        if salt is None:
            salt = str(uuid.uuid4())
        # 创建 SHA-256 哈希对象
        m = hashlib.sha256()
        m.update(salt.encode('utf8'))
        m.update(str(content).encode('utf8'))
        # 返回格式化的唯一标识符
        return f'{type}{m.hexdigest()[:10]}'

    # 创建翻转变换
    def _make_flip_transform(self, transform):
        # 创建翻转变换，将对象沿 Y 轴翻转
        return transform + Affine2D().scale(1, -1).translate(0, self.height)

    # 获取图案填充的唯一标识符
    def _get_hatch(self, gc, rgbFace):
        """
        Create a new hatch pattern
        """
        # 如果存在填充颜色，转换为元组形式
        if rgbFace is not None:
            rgbFace = tuple(rgbFace)
        # 获取图案的边缘颜色
        edge = gc.get_hatch_color()
        if edge is not None:
            edge = tuple(edge)
        # 根据填充样式、边缘颜色和图案类型创建字典键
        dictkey = (gc.get_hatch(), rgbFace, edge)
        # 查找已存在的图案标识符，如果不存在则创建新的
        oid = self._hatchd.get(dictkey)
        if oid is None:
            oid = self._make_id('h', dictkey)
            # 将新创建的图案添加到图案字典中
            self._hatchd[dictkey] = ((gc.get_hatch_path(), rgbFace, edge), oid)
        else:
            _, oid = oid
        # 返回图案的唯一标识符
        return oid

    # 写入图案定义
    def _write_hatches(self):
        # 如果没有图案数据，直接返回
        if not len(self._hatchd):
            return
        # 定义图案大小
        HATCH_SIZE = 72
        # 获取写入器对象
        writer = self.writer
        # 开始写入 <defs> 标签
        writer.start('defs')
        # 遍历图案字典中的每个图案数据
        for (path, face, stroke), oid in self._hatchd.values():
            # 开始写入 <pattern> 标签，定义图案属性
            writer.start(
                'pattern',
                id=oid,
                patternUnits="userSpaceOnUse",
                x="0", y="0", width=str(HATCH_SIZE),
                height=str(HATCH_SIZE))
            # 将路径数据转换为指定坐标系下的路径数据
            path_data = self._convert_path(
                path,
                Affine2D()
                .scale(HATCH_SIZE).scale(1.0, -1.0).translate(0, HATCH_SIZE),
                simplify=False)
            # 确定图案的填充颜色
            if face is None:
                fill = 'none'
            else:
                fill = rgb2hex(face)
            # 在图案中添加填充矩形
            writer.element(
                'rect',
                x="0", y="0", width=str(HATCH_SIZE+1),
                height=str(HATCH_SIZE+1),
                fill=fill)
            # 定义图案样式
            hatch_style = {
                    'fill': rgb2hex(stroke),
                    'stroke': rgb2hex(stroke),
                    'stroke-width': str(mpl.rcParams['hatch.linewidth']),
                    'stroke-linecap': 'butt',
                    'stroke-linejoin': 'miter'
                    }
            # 如果边缘颜色透明度小于 1，添加透明度属性
            if stroke[3] < 1:
                hatch_style['stroke-opacity'] = str(stroke[3])
            # 在图案中添加路径
            writer.element(
                'path',
                d=path_data,
                style=_generate_css(hatch_style)
                )
            # 结束 <pattern> 标签
            writer.end('pattern')
        # 结束 <defs> 标签
        writer.end('defs')
    # 从 GraphicsContext 和 rgbFace 生成样式字典
    def _get_style_dict(self, gc, rgbFace):
        attrib = {}  # 初始化一个空的属性字典

        forced_alpha = gc.get_forced_alpha()  # 获取强制 alpha 值

        # 如果有 hatch pattern，则设置 fill 属性为对应的 SVG 样式
        if gc.get_hatch() is not None:
            attrib['fill'] = f"url(#{self._get_hatch(gc, rgbFace)})"
            # 如果 rgbFace 不为 None 且是 RGBA 格式且 alpha 不为 1.0，且没有强制 alpha，则设置 fill-opacity 属性
            if (rgbFace is not None and len(rgbFace) == 4 and rgbFace[3] != 1.0
                    and not forced_alpha):
                attrib['fill-opacity'] = _short_float_fmt(rgbFace[3])
        else:
            # 如果没有 hatch pattern
            if rgbFace is None:
                attrib['fill'] = 'none'  # 如果 rgbFace 为 None，则设置 fill 属性为 'none'
            else:
                # 如果 rgbFace 不为 None，则设置 fill 属性为 rgbFace 的十六进制表示
                if tuple(rgbFace[:3]) != (0, 0, 0):
                    attrib['fill'] = rgb2hex(rgbFace)
                # 如果 rgbFace 是 RGBA 格式且 alpha 不为 1.0，且没有强制 alpha，则设置 fill-opacity 属性
                if (len(rgbFace) == 4 and rgbFace[3] != 1.0
                        and not forced_alpha):
                    attrib['fill-opacity'] = _short_float_fmt(rgbFace[3])

        # 如果有强制 alpha 并且 GraphicsContext 的 alpha 值不为 1.0，则设置 opacity 属性
        if forced_alpha and gc.get_alpha() != 1.0:
            attrib['opacity'] = _short_float_fmt(gc.get_alpha())

        # 获取虚线样式信息
        offset, seq = gc.get_dashes()
        if seq is not None:
            # 设置 stroke-dasharray 属性为虚线的长度序列的 CSV 表示
            attrib['stroke-dasharray'] = ','.join(
                _short_float_fmt(val) for val in seq)
            # 设置 stroke-dashoffset 属性为虚线的偏移量
            attrib['stroke-dashoffset'] = _short_float_fmt(float(offset))

        linewidth = gc.get_linewidth()  # 获取线宽
        if linewidth:
            rgb = gc.get_rgb()  # 获取 RGB 颜色值
            attrib['stroke'] = rgb2hex(rgb)  # 设置 stroke 属性为线条的颜色的十六进制表示
            # 如果没有强制 alpha 并且颜色的 alpha 值不为 1.0，则设置 stroke-opacity 属性
            if not forced_alpha and rgb[3] != 1.0:
                attrib['stroke-opacity'] = _short_float_fmt(rgb[3])
            # 如果线宽不为 1.0，则设置 stroke-width 属性
            if linewidth != 1.0:
                attrib['stroke-width'] = _short_float_fmt(linewidth)
            # 如果线条的连接风格不是 'round'，则设置 stroke-linejoin 属性
            if gc.get_joinstyle() != 'round':
                attrib['stroke-linejoin'] = gc.get_joinstyle()
            # 如果线条的端点风格不是 'butt'，则设置 stroke-linecap 属性
            if gc.get_capstyle() != 'butt':
                attrib['stroke-linecap'] = _capstyle_d[gc.get_capstyle()]

        return attrib  # 返回生成的样式属性字典

    # 从 GraphicsContext 和 rgbFace 生成样式字符串
    def _get_style(self, gc, rgbFace):
        return _generate_css(self._get_style_dict(gc, rgbFace))

    # 从 GraphicsContext 获取剪辑属性
    def _get_clip_attrs(self, gc):
        cliprect = gc.get_clip_rectangle()  # 获取剪辑矩形
        clippath, clippath_trans = gc.get_clip_path()  # 获取剪辑路径和剪辑路径的变换

        if clippath is not None:
            # 如果有剪辑路径，则生成唯一的 ID，并设置 clip-path 属性
            clippath_trans = self._make_flip_transform(clippath_trans)
            dictkey = (id(clippath), str(clippath_trans))
        elif cliprect is not None:
            # 如果没有剪辑路径但有剪辑矩形，则生成唯一的 ID，并设置 clip-path 属性
            x, y, w, h = cliprect.bounds
            y = self.height-(y+h)
            dictkey = (x, y, w, h)
        else:
            return {}  # 如果既没有剪辑路径也没有剪辑矩形，则返回空字典

        clip = self._clipd.get(dictkey)  # 查看是否已经存在该剪辑区域的缓存
        if clip is None:
            oid = self._make_id('p', dictkey)  # 如果不存在，则生成新的唯一 ID
            if clippath is not None:
                self._clipd[dictkey] = ((clippath, clippath_trans), oid)
            else:
                self._clipd[dictkey] = (dictkey, oid)
        else:
            clip, oid = clip

        return {'clip-path': f'url(#{oid})'}  # 返回剪辑区域的属性字典，使用生成的唯一 ID
    def _write_clips(self):
        # 如果没有剪辑数据，则直接返回
        if not len(self._clipd):
            return
        # 获取写入器对象的引用
        writer = self.writer
        # 开始写入 "defs" 标签
        writer.start('defs')
        # 遍历剪辑字典中的值
        for clip, oid in self._clipd.values():
            # 开始写入 "clipPath" 标签，设置其 id 属性为 oid
            writer.start('clipPath', id=oid)
            # 根据剪辑数据的长度选择处理路径或者矩形
            if len(clip) == 2:
                # 如果长度为2，则clip包含路径和变换信息
                clippath, clippath_trans = clip
                # 将路径数据转换为SVG路径字符串
                path_data = self._convert_path(
                    clippath, clippath_trans, simplify=False)
                # 写入路径元素
                writer.element('path', d=path_data)
            else:
                # 否则，clip包含矩形的位置和大小信息
                x, y, w, h = clip
                # 写入矩形元素
                writer.element(
                    'rect',
                    x=_short_float_fmt(x),
                    y=_short_float_fmt(y),
                    width=_short_float_fmt(w),
                    height=_short_float_fmt(h))
            # 结束 "clipPath" 标签
            writer.end('clipPath')
        # 结束 "defs" 标签
        writer.end('defs')

    def open_group(self, s, gid=None):
        # 继承文档字符串描述
        if gid:
            # 如果提供了gid，则使用指定的gid作为组的id
            self.writer.start('g', id=gid)
        else:
            # 否则，生成一个唯一的组id并开始写入 "g" 标签
            self._groupd[s] = self._groupd.get(s, 0) + 1
            self.writer.start('g', id=f"{s}_{self._groupd[s]:d}")

    def close_group(self, s):
        # 继承文档字符串描述
        # 结束当前组标签
        self.writer.end('g')

    def option_image_nocomposite(self):
        # 继承文档字符串描述
        # 返回是否不使用合成图像的选项
        return not mpl.rcParams['image.composite_image']

    def _convert_path(self, path, transform=None, clip=None, simplify=None,
                      sketch=None):
        # 如果需要剪辑，则设置剪辑范围为整个绘图区域
        if clip:
            clip = (0.0, 0.0, self.width, self.height)
        else:
            clip = None
        # 调用内部方法将路径对象转换为SVG路径字符串
        return _path.convert_to_string(
            path, transform, clip, simplify, sketch, 6,
            [b'M', b'L', b'Q', b'C', b'z'], False).decode('ascii')

    def draw_path(self, gc, path, transform, rgbFace=None):
        # 继承文档字符串描述
        # 创建变换矩阵并处理是否需要剪辑和简化路径
        trans_and_flip = self._make_flip_transform(transform)
        clip = (rgbFace is None and gc.get_hatch_path() is None)
        simplify = path.should_simplify and clip
        # 转换路径对象为SVG路径字符串
        path_data = self._convert_path(
            path, trans_and_flip, clip=clip, simplify=simplify,
            sketch=gc.get_sketch_params())

        # 如果有超链接，则开始写入 "a" 标签，并设置超链接属性
        if gc.get_url() is not None:
            self.writer.start('a', {'xlink:href': gc.get_url()})
        # 写入路径元素，包括路径数据、剪辑属性和样式属性
        self.writer.element('path', d=path_data, **self._get_clip_attrs(gc),
                            style=self._get_style(gc, rgbFace))
        # 如果有超链接，则结束 "a" 标签
        if gc.get_url() is not None:
            self.writer.end('a')
    # 绘制标记点的方法，使用指定的绘图上下文和路径信息进行绘制
    def draw_markers(
            self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        # 如果路径中没有顶点，直接返回，不进行绘制
        if not len(path.vertices):
            return

        # 获取当前对象的写入器
        writer = self.writer
        # 将标记路径转换为路径数据，应用标记变换和镜像变换
        path_data = self._convert_path(
            marker_path,
            marker_trans + Affine2D().scale(1.0, -1.0),
            simplify=False)
        # 获取样式字典，用于绘制
        style = self._get_style_dict(gc, rgbFace)
        # 生成路径数据和样式的元组作为字典的键
        dictkey = (path_data, _generate_css(style))
        # 根据键从标记字典中获取标记对象的 ID
        oid = self._markers.get(dictkey)
        # 生成只包含描边相关属性的样式字典
        style = _generate_css({k: v for k, v in style.items()
                              if k.startswith('stroke')})

        # 如果未找到对应的标记对象 ID
        if oid is None:
            # 创建一个新的唯一 ID 作为标记对象的 ID
            oid = self._make_id('m', dictkey)
            # 开始定义 XML 的 <defs> 元素
            writer.start('defs')
            # 在 <defs> 元素中添加 <path> 元素，用于定义标记的形状和样式
            writer.element('path', id=oid, d=path_data, style=style)
            # 结束 <defs> 元素的定义
            writer.end('defs')
            # 将新创建的标记对象 ID 存入标记字典中，以备后续使用
            self._markers[dictkey] = oid

        # 开始定义 XML 的 <g> 元素，用于组合多个标记对象
        writer.start('g', **self._get_clip_attrs(gc))
        # 创建变换矩阵，用于处理坐标系的翻转
        trans_and_flip = self._make_flip_transform(trans)
        # 定义属性字典，用于描述使用标记对象的 <use> 元素
        attrib = {'xlink:href': f'#{oid}'}
        # 定义剪切区域，用于限制绘制的范围
        clip = (0, 0, self.width*72, self.height*72)
        # 遍历路径的段，获取每个段的顶点和代码
        for vertices, code in path.iter_segments(
                trans_and_flip, clip=clip, simplify=False):
            # 如果顶点列表非空
            if len(vertices):
                # 获取最后两个顶点作为当前位置的坐标
                x, y = vertices[-2:]
                # 将坐标格式化为短浮点数字符串，并添加到属性字典中
                attrib['x'] = _short_float_fmt(x)
                attrib['y'] = _short_float_fmt(y)
                # 获取当前的样式并添加到属性字典中
                attrib['style'] = self._get_style(gc, rgbFace)
                # 在 <g> 元素中添加 <use> 元素，使用之前定义的属性字典
                writer.element('use', attrib=attrib)
        # 结束定义 XML 的 <g> 元素
        writer.end('g')
    def draw_path_collection(self, gc, master_transform, paths, all_transforms,
                             offsets, offset_trans, facecolors, edgecolors,
                             linewidths, linestyles, antialiaseds, urls,
                             offset_position):
        # 判断是否值得进行优化的粗略计算:
        # 在线绘制路径的成本是
        #    (len_path + 5) * uses_per_path
        # 定义+使用的成本是
        #    (len_path + 3) + 9 * uses_per_path
        # 计算路径的长度
        len_path = len(paths[0].vertices) if len(paths) > 0 else 0
        # 计算每条路径的使用次数
        uses_per_path = self._iter_collection_uses_per_path(
            paths, all_transforms, offsets, facecolors, edgecolors)
        # 判断是否应该进行优化
        should_do_optimization = \
            len_path + 9 * uses_per_path + 3 < (len_path + 5) * uses_per_path
        # 如果不需要优化，则调用父类方法绘制路径集合并返回
        if not should_do_optimization:
            return super().draw_path_collection(
                gc, master_transform, paths, all_transforms,
                offsets, offset_trans, facecolors, edgecolors,
                linewidths, linestyles, antialiaseds, urls,
                offset_position)

        # 获取写入器
        writer = self.writer
        # 存储路径代码
        path_codes = []
        # 开始定义路径集合
        writer.start('defs')
        # 遍历原始路径和变换
        for i, (path, transform) in enumerate(self._iter_collection_raw_paths(
                master_transform, paths, all_transforms)):
            # 转换坐标系，确保Y轴朝下
            transform = Affine2D(transform.get_matrix()).scale(1.0, -1.0)
            # 转换路径数据为字符串格式
            d = self._convert_path(path, transform, simplify=False)
            # 生成路径对象ID
            oid = 'C{:x}_{:x}_{}'.format(
                self._path_collection_id, i, self._make_id('', d))
            # 添加路径元素到写入器中
            writer.element('path', id=oid, d=d)
            # 将路径对象ID添加到路径代码列表中
            path_codes.append(oid)
        # 结束定义路径集合
        writer.end('defs')

        # 遍历路径集合并绘制每个路径
        for xo, yo, path_id, gc0, rgbFace in self._iter_collection(
                gc, path_codes, offsets, offset_trans,
                facecolors, edgecolors, linewidths, linestyles,
                antialiaseds, urls, offset_position):
            # 获取路径关联的URL
            url = gc0.get_url()
            # 如果URL不为空，开始创建超链接元素
            if url is not None:
                writer.start('a', attrib={'xlink:href': url})
            # 获取裁剪属性
            clip_attrs = self._get_clip_attrs(gc0)
            # 如果存在裁剪属性，创建包含裁剪属性的组
            if clip_attrs:
                writer.start('g', **clip_attrs)
            # 定义使用路径的属性
            attrib = {
                'xlink:href': f'#{path_id}',
                'x': _short_float_fmt(xo),
                'y': _short_float_fmt(self.height - yo),
                'style': self._get_style(gc0, rgbFace)
            }
            # 创建路径使用元素
            writer.element('use', attrib=attrib)
            # 如果存在裁剪属性，结束裁剪组
            if clip_attrs:
                writer.end('g')
            # 如果URL不为空，结束超链接元素
            if url is not None:
                writer.end('a')

        # 增加路径集合ID以便下一次绘制
        self._path_collection_id += 1
    # 绘制高尔德三角形到指定图形上下文
    def draw_gouraud_triangles(self, gc, triangles_array, colors_array,
                               transform):
        # 获取写入器对象
        writer = self.writer
        # 开始一个新的SVG组元素，并传递裁剪属性
        writer.start('g', **self._get_clip_attrs(gc))
        # 冻结变换对象，确保不可变状态
        transform = transform.frozen()
        # 创建变换及翻转操作的组合
        trans_and_flip = self._make_flip_transform(transform)

        # 如果尚未使用高尔德绘制，则进行初始化
        if not self._has_gouraud:
            self._has_gouraud = True
            # 开始一个新的SVG滤镜元素，用于颜色添加
            writer.start(
                'filter',
                id='colorAdd')
            # 添加一个feComposite元素，用于颜色叠加操作
            writer.element(
                'feComposite',
                attrib={'in': 'SourceGraphic'},
                in2='BackgroundImage',
                operator='arithmetic',
                k2="1", k3="1")
            writer.end('filter')
            # 添加一个feColorMatrix滤镜元素，用于修正透明度
            writer.start(
                'filter',
                id='colorMat')
            writer.element(
                'feColorMatrix',
                attrib={'type': 'matrix'},
                values='1 0 0 0 0 \n0 1 0 0 0 \n0 0 1 0 0 \n1 1 1 1 0 \n0 0 0 0 1 ')
            writer.end('filter')

        # 遍历三角形数组和颜色数组，绘制高尔德三角形
        for points, colors in zip(triangles_array, colors_array):
            self._draw_gouraud_triangle(trans_and_flip.transform(points), colors)
        # 结束SVG组元素
        writer.end('g')

    # 返回图像缩放选项
    def option_scale_image(self):
        # 继承的文档字符串，表明该方法返回True
        return True

    # 获取图像放大倍数
    def get_image_magnification(self):
        return self.image_dpi / 72.0
    def draw_image(self, gc, x, y, im, transform=None):
        # 绘制图像到 SVG 中，支持应用变换

        # 获取图像的高度和宽度
        h, w = im.shape[:2]

        # 如果图像宽度或高度为0，则直接返回，不进行绘制
        if w == 0 or h == 0:
            return

        # 获取裁剪属性
        clip_attrs = self._get_clip_attrs(gc)

        # 如果存在裁剪属性，则创建一个 <g> 元素，并添加裁剪属性
        if clip_attrs:
            # 无法直接将裁剪路径应用于图像，因为图像有变换，裁剪路径也会被应用
            self.writer.start('g', **clip_attrs)

        # 获取图像的 URL
        url = gc.get_url()

        # 如果 URL 不为空，则创建一个 <a> 元素，并添加 xlink:href 属性
        if url is not None:
            self.writer.start('a', attrib={'xlink:href': url})

        # 创建属性字典
        attrib = {}

        # 获取或生成图像的全局唯一标识符 (gid)
        oid = gc.get_gid()

        # 如果配置允许内联 SVG 图像
        if mpl.rcParams['svg.image_inline']:
            # 将图像转换为 PNG 格式，并编码为 base64 字符串
            buf = BytesIO()
            Image.fromarray(im).save(buf, format="png")
            oid = oid or self._make_id('image', buf.getvalue())
            attrib['xlink:href'] = (
                "data:image/png;base64,\n" +
                base64.b64encode(buf.getvalue()).decode('ascii'))
        else:
            # 如果没有设置 basename，则无法将图像数据保存到文件系统
            if self.basename is None:
                raise ValueError("Cannot save image data to filesystem when "
                                 "writing SVG to an in-memory buffer")
            # 生成保存图像的文件名
            filename = f'{self.basename}.image{next(self._image_counter)}.png'
            _log.info('Writing image file for inclusion: %s', filename)
            # 将图像保存为文件
            Image.fromarray(im).save(filename)
            oid = oid or 'Im_' + self._make_id('image', filename)
            attrib['xlink:href'] = filename

        # 设置图像的 id 属性
        attrib['id'] = oid

        # 如果没有指定变换 (transform)
        if transform is None:
            # 根据图像 DPI 调整宽度和高度
            w = 72.0 * w / self.image_dpi
            h = 72.0 * h / self.image_dpi

            # 创建 <image> 元素，添加属性和变换
            self.writer.element(
                'image',
                transform=_generate_transform([
                    ('scale', (1, -1)), ('translate', (0, -h))]),
                x=_short_float_fmt(x),
                y=_short_float_fmt(-(self.height - y - h)),
                width=_short_float_fmt(w), height=_short_float_fmt(h),
                attrib=attrib)
        else:
            # 获取图形上下文中的透明度
            alpha = gc.get_alpha()

            # 如果透明度不为1.0，则设置 opacity 属性
            if alpha != 1.0:
                attrib['opacity'] = _short_float_fmt(alpha)

            # 计算图像的变换矩阵，包括缩放、用户定义的变换和坐标转换
            flipped = (
                Affine2D().scale(1.0 / w, 1.0 / h) +
                transform +
                Affine2D()
                .translate(x, y)
                .scale(1.0, -1.0)
                .translate(0.0, self.height))

            # 设置 transform 属性和 style 属性，用于更高质量的图像渲染
            attrib['transform'] = _generate_transform(
                [('matrix', flipped.frozen())])
            attrib['style'] = (
                'image-rendering:crisp-edges;'
                'image-rendering:pixelated')

            # 创建 <image> 元素，添加属性
            self.writer.element(
                'image',
                width=_short_float_fmt(w), height=_short_float_fmt(h),
                attrib=attrib)

        # 如果 URL 不为空，则结束 <a> 元素
        if url is not None:
            self.writer.end('a')

        # 如果存在裁剪属性，则结束 <g> 元素
        if clip_attrs:
            self.writer.end('g')
    def _update_glyph_map_defs(self, glyph_map_new):
        """
        Emit definitions for not-yet-defined glyphs, and record them as having
        been defined.
        """
        # 获取当前对象的写入器
        writer = self.writer
        # 如果传入的字形映射不为空
        if glyph_map_new:
            # 开始写入 <defs> 标签
            writer.start('defs')
            # 遍历新字形映射中的每个字符 ID 及其顶点和路径代码
            for char_id, (vertices, codes) in glyph_map_new.items():
                # 调整字符 ID，将 "%20" 替换为 "_"
                char_id = self._adjust_char_id(char_id)
                # 将顶点和代码转换为路径数据，将 x64 缩放回到 FreeType 的内部单位
                path_data = self._convert_path(
                    Path(vertices * 64, codes), simplify=False)
                # 写入 <path> 元素，包括 id、d 属性和变换信息
                writer.element(
                    'path', id=char_id, d=path_data,
                    transform=_generate_transform([('scale', (1 / 64,))]))
            # 结束 <defs> 标签的写入
            writer.end('defs')
            # 更新内部字形映射表
            self._glyph_map.update(glyph_map_new)

    def _adjust_char_id(self, char_id):
        # 将字符 ID 中的 "%20" 替换为 "_"
        return char_id.replace("%20", "_")
    def _draw_text_as_path(self, gc, x, y, s, prop, angle, ismath, mtext=None):
        # 获取写入器对象，用于生成 SVG
        writer = self.writer

        # 将字符串 s 作为注释写入 SVG
        writer.comment(s)

        # 获取字符映射表
        glyph_map = self._glyph_map

        # 获取文本到路径的转换器对象
        text2path = self._text2path

        # 获取绘制文本的颜色并转换为十六进制格式
        color = rgb2hex(gc.get_rgb())

        # 获取文本的字体大小（以点为单位）
        fontsize = prop.get_size_in_points()

        # 设置样式字典，根据文本颜色设置填充颜色
        style = {}
        if color != '#000000':
            style['fill'] = color

        # 获取文本的透明度
        alpha = gc.get_alpha() if gc.get_forced_alpha() else gc.get_rgb()[3]
        if alpha != 1:
            style['opacity'] = _short_float_fmt(alpha)

        # 计算字体缩放比例
        font_scale = fontsize / text2path.FONT_SCALE

        # 设置 SVG 元素的属性
        attrib = {
            'style': _generate_css(style),  # 生成 CSS 样式字符串
            'transform': _generate_transform([  # 生成变换列表
                ('translate', (x, y)),  # 平移变换
                ('rotate', (-angle,)),  # 旋转变换（角度为负）
                ('scale', (font_scale, -font_scale))]),  # 缩放变换（Y 轴反向）
        }

        # 开始一个 SVG 组（group）元素，应用上述属性
        writer.start('g', attrib=attrib)

        # 如果不是数学文本
        if not ismath:
            # 获取文本的字体对象
            font = text2path._get_font(prop)

            # 使用指定字体获取文本的字形信息
            _glyphs = text2path.get_glyphs_with_font(
                font, s, glyph_map=glyph_map, return_new_glyphs_only=True)
            glyph_info, glyph_map_new, rects = _glyphs

            # 更新字符映射表
            self._update_glyph_map_defs(glyph_map_new)

            # 遍历每个字形信息
            for glyph_id, xposition, yposition, scale in glyph_info:
                attrib = {'xlink:href': f'#{glyph_id}'}

                # 如果 xposition 不为零，则添加 x 属性
                if xposition != 0.0:
                    attrib['x'] = _short_float_fmt(xposition)

                # 如果 yposition 不为零，则添加 y 属性
                if yposition != 0.0:
                    attrib['y'] = _short_float_fmt(yposition)

                # 添加一个 'use' 元素到 SVG 中
                writer.element('use', attrib=attrib)

        # 如果是数学文本
        else:
            # 根据 ismath 类型选择相应的方法获取字形信息
            if ismath == "TeX":
                _glyphs = text2path.get_glyphs_tex(
                    prop, s, glyph_map=glyph_map, return_new_glyphs_only=True)
            else:
                _glyphs = text2path.get_glyphs_mathtext(
                    prop, s, glyph_map=glyph_map, return_new_glyphs_only=True)
            glyph_info, glyph_map_new, rects = _glyphs

            # 更新字符映射表
            self._update_glyph_map_defs(glyph_map_new)

            # 遍历每个字符的信息
            for char_id, xposition, yposition, scale in glyph_info:
                char_id = self._adjust_char_id(char_id)

                # 添加一个 'use' 元素到 SVG 中，应用变换属性
                writer.element(
                    'use',
                    transform=_generate_transform([
                        ('translate', (xposition, yposition)),  # 平移变换
                        ('scale', (scale,)),  # 缩放变换
                        ]),
                    attrib={'xlink:href': f'#{char_id}'})  # 引用字符 ID

            # 遍历每个矩形的顶点和代码，创建 SVG 路径元素
            for verts, codes in rects:
                path = Path(verts, codes)
                path_data = self._convert_path(path, simplify=False)
                writer.element('path', d=path_data)  # 添加路径元素到 SVG 中

        # 结束 SVG 组元素
        writer.end('g')
    # 绘制文本方法，用于在画布上绘制文本内容
    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # 继承的文档字符串

        # 获取文本的剪裁属性
        clip_attrs = self._get_clip_attrs(gc)
        # 如果存在剪裁属性，则创建包含这些属性的 SVG 分组元素
        if clip_attrs:
            self.writer.start('g', **clip_attrs)

        # 如果文本具有 URL，则将其包装在超链接元素 'a' 中
        if gc.get_url() is not None:
            self.writer.start('a', {'xlink:href': gc.get_url()})

        # 根据配置决定是将文本绘制为路径还是普通文本
        if mpl.rcParams['svg.fonttype'] == 'path':
            self._draw_text_as_path(gc, x, y, s, prop, angle, ismath, mtext)
        else:
            self._draw_text_as_text(gc, x, y, s, prop, angle, ismath, mtext)

        # 如果文本具有 URL，则关闭 'a' 元素
        if gc.get_url() is not None:
            self.writer.end('a')

        # 如果存在剪裁属性，则关闭包含这些属性的 SVG 分组元素
        if clip_attrs:
            self.writer.end('g')

    # 翻转 Y 轴方法
    def flipy(self):
        # 继承的文档字符串
        return True

    # 获取画布宽度和高度方法
    def get_canvas_width_height(self):
        # 继承的文档字符串
        return self.width, self.height

    # 获取文本宽度、高度和下降值方法
    def get_text_width_height_descent(self, s, prop, ismath):
        # 继承的文档字符串
        return self._text2path.get_text_width_height_descent(s, prop, ismath)
class FigureCanvasSVG(FigureCanvasBase):
    # SVG 文件的输出类型定义，包括普通 SVG 和压缩 SVGZ
    filetypes = {'svg': 'Scalable Vector Graphics',
                 'svgz': 'Scalable Vector Graphics'}

    # 固定的 DPI 值为 72
    fixed_dpi = 72

    def print_svg(self, filename, *, bbox_inches_restore=None, metadata=None):
        """
        Parameters
        ----------
        filename : str or path-like or file-like
            输出目标；如果是字符串，则会打开一个文件进行写入。

        metadata : dict[str, Any], optional
            SVG 文件中的元数据，定义为字符串、日期时间或字符串列表的键值对，例如
            {'Creator': 'My software', 'Contributor': ['Me', 'My Friend'], 'Title': 'Awesome'}。
            支持的标准键和它们的值类型包括：
            - 字符串类型：'Coverage', 'Description', 'Format', 'Identifier', 'Language', 'Relation',
              'Source', 'Title', 'Type'。
            - 字符串或字符串列表类型：'Contributor', 'Creator', 'Keywords', 'Publisher', 'Rights'。
            - 字符串、日期、日期时间或它们的元组类型：'Date'。如果不是字符串，则会格式化为 ISO 8601 格式。

            预定义了 'Creator', 'Date', 'Format' 和 'Type' 的值。可以通过将它们设置为 `None` 来移除它们。

            这些信息编码为 Dublin Core Metadata。

            Dublin Core Metadata 的详细信息参见 https://www.dublincore.org/specifications/dublin-core/

        """
        # 使用 cbook.open_file_cm 打开文件，以写入模式，使用 UTF-8 编码
        with cbook.open_file_cm(filename, "w", encoding="utf-8") as fh:
            # 如果文件不需要 Unicode 编码，则使用 utf-8 编码写入
            if not cbook.file_requires_unicode(fh):
                fh = codecs.getwriter('utf-8')(fh)
            # 设置图形的 DPI 为 72
            dpi = self.figure.dpi
            self.figure.dpi = 72
            # 获取图形的尺寸（单位为英寸），转换为像素
            width, height = self.figure.get_size_inches()
            w, h = width * 72, height * 72
            # 创建 MixedModeRenderer 对象，用于渲染 SVG
            renderer = MixedModeRenderer(
                self.figure, width, height, dpi,
                RendererSVG(w, h, fh, image_dpi=dpi, metadata=metadata),
                bbox_inches_restore=bbox_inches_restore)
            # 绘制图形
            self.figure.draw(renderer)
            # 完成渲染
            renderer.finalize()

    def print_svgz(self, filename, **kwargs):
        # 使用 gzip 压缩的方式写入 SVGZ 文件
        with cbook.open_file_cm(filename, "wb") as fh, \
                gzip.GzipFile(mode='w', fileobj=fh) as gzipwriter:
            # 调用 print_svg 方法，将输出写入到 gzipwriter 中
            return self.print_svg(gzipwriter, **kwargs)

    def get_default_filetype(self):
        # 返回默认的文件类型为 'svg'
        return 'svg'

    def draw(self):
        # 在不进行渲染的情况下绘制图形
        self.figure.draw_without_rendering()
        return super().draw()


FigureManagerSVG = FigureManagerBase


svgProlog = """\
<?xml version="1.0" encoding="utf-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
"""

# 将 _BackendSVG 导出为 _Backend 类的一部分
@_Backend.export
class _BackendSVG(_Backend):
    # 后端版本信息为当前 Matplotlib 版本号
    backend_version = mpl.__version__
    # 使用 FigureCanvasSVG 作为 SVG 后端的画布对象
    FigureCanvas = FigureCanvasSVG
```