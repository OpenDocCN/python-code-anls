# `D:\src\scipysrc\matplotlib\lib\matplotlib\dviread.py`

```py
"""
A module for reading dvi files output by TeX. Several limitations make
this not (currently) useful as a general-purpose dvi preprocessor, but
it is currently used by the pdf backend for processing usetex text.

Interface::

  with Dvi(filename, 72) as dvi:
      # iterate over pages:
      for page in dvi:
          w, h, d = page.width, page.height, page.descent
          for x, y, font, glyph, width in page.text:
              fontname = font.texname
              pointsize = font.size
              ...
          for x, y, height, width in page.boxes:
              ...
"""

from collections import namedtuple  # 导入命名元组模块
import enum  # 导入枚举类型模块
from functools import lru_cache, partial, wraps  # 导入缓存、部分函数和装饰器模块
import logging  # 导入日志模块
import os  # 导入操作系统功能模块
from pathlib import Path  # 导入路径操作模块
import re  # 导入正则表达式模块
import struct  # 导入结构化数据处理模块
import subprocess  # 导入子进程管理模块
import sys  # 导入系统相关功能模块

import numpy as np  # 导入数值计算库numpy

from matplotlib import _api, cbook  # 导入matplotlib的部分子模块

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象

# Many dvi related files are looked for by external processes, require
# additional parsing, and are used many times per rendering, which is why they
# are cached using lru_cache().
# 许多与 DVI 相关的文件由外部进程查找，需要额外的解析，并且在每次渲染时被多次使用，因此使用 lru_cache() 进行缓存。

# Dvi is a bytecode format documented in
# https://ctan.org/pkg/dvitype
# https://texdoc.org/serve/dvitype.pdf/0
#
# The file consists of a preamble, some number of pages, a postamble,
# and a finale. Different opcodes are allowed in different contexts,
# so the Dvi object has a parser state:
#
#   pre:       expecting the preamble
#   outer:     between pages (followed by a page or the postamble,
#              also e.g. font definitions are allowed)
#   page:      processing a page
#   post_post: state after the postamble (our current implementation
#              just stops reading)
#   finale:    the finale (unimplemented in our current implementation)
#
# DVI 是一种字节码格式，详细记录在以下文档中：
# https://ctan.org/pkg/dvitype
# https://texdoc.org/serve/dvitype.pdf/0
#
# DVI 文件包括前言、若干页、尾声和结尾部分。不同的操作码在不同的上下文中允许使用，
# 因此 Dvi 对象具有解析状态:
#
#   pre:       期望前言
#   outer:     在页面之间（后面是页面或尾声，也允许例如字体定义）
#   page:      处理页面
#   post_post: 尾声后的状态（我们当前的实现只是停止读取）
#   finale:    结尾部分（在当前实现中未实现）

_dvistate = enum.Enum('DviState', 'pre outer inpage post_post finale')  # 定义 Dvi 解析器的状态枚举类型

# The marks on a page consist of text and boxes. A page also has dimensions.
Page = namedtuple('Page', 'text boxes height width descent')  # 定义命名元组 Page，表示页面包含文本、方框、高度、宽度和下降值
Box = namedtuple('Box', 'x y height width')  # 定义命名元组 Box，表示页面中的方框包含 x 坐标、y 坐标、高度和宽度


# Also a namedtuple, for backcompat.
class Text(namedtuple('Text', 'x y font glyph width')):
    """
    A glyph in the dvi file.

    The *x* and *y* attributes directly position the glyph.  The *font*,
    *glyph*, and *width* attributes are kept public for back-compatibility,
    but users wanting to draw the glyph themselves are encouraged to instead
    load the font specified by `font_path` at `font_size`, warp it with the
    effects specified by `font_effects`, and load the glyph specified by
    `glyph_name_or_index`.
    """

    def _get_pdftexmap_entry(self):
        return PsfontsMap(find_tex_file("pdftex.map"))[self.font.texname]

    @property
    def font_path(self):
        """
        The `~pathlib.Path` to the font for this glyph.

        Retrieve the filename of the font associated with this glyph from
        pdftex.map. If no suitable filename is found, raise a ValueError
        indicating the issue.
        """
        psfont = self._get_pdftexmap_entry()
        if psfont.filename is None:
            raise ValueError("No usable font file found for {} ({}); "
                             "the font may lack a Type-1 version"
                             .format(psfont.psname.decode("ascii"),
                                     psfont.texname.decode("ascii")))
        return Path(psfont.filename)

    @property
    def font_size(self):
        """
        The font size.

        Return the font size of the current glyph, accessed through its font
        attribute.
        """
        return self.font.size

    @property
    def font_effects(self):
        """
        The "font effects" dict for this glyph.

        Retrieve and return the font effects (such as SlantFont and ExtendFont)
        specific to this glyph from pdftex.map.
        """
        return self._get_pdftexmap_entry().effects

    @property
    def glyph_name_or_index(self):
        """
        Either the glyph name or the native charmap glyph index.

        Determine whether to use the Adobe glyph names from pdftex.map for
        converting DVI indices to glyph names, or directly use the font's native
        charmap if no encoding is specified in pdftex.map.
        """
        entry = self._get_pdftexmap_entry()
        return (_parse_enc(entry.encoding)[self.glyph]
                if entry.encoding is not None else self.glyph)
# Opcode argument parsing
#
# Each of the following functions takes a Dvi object and delta, which is the
# difference between the opcode and the minimum opcode with the same meaning.
# Dvi opcodes often encode the number of argument bytes in this delta.

_arg_mapping = dict(
    # raw: Return delta as is.
    raw=lambda dvi, delta: delta,
    # u1: Read 1 byte as an unsigned number.
    u1=lambda dvi, delta: dvi._arg(1, signed=False),
    # u4: Read 4 bytes as an unsigned number.
    u4=lambda dvi, delta: dvi._arg(4, signed=False),
    # s4: Read 4 bytes as a signed number.
    s4=lambda dvi, delta: dvi._arg(4, signed=True),
    # slen: Read delta bytes as a signed number, or None if delta is None.
    slen=lambda dvi, delta: dvi._arg(delta, signed=True) if delta else None,
    # slen1: Read (delta + 1) bytes as a signed number.
    slen1=lambda dvi, delta: dvi._arg(delta + 1, signed=True),
    # ulen1: Read (delta + 1) bytes as an unsigned number.
    ulen1=lambda dvi, delta: dvi._arg(delta + 1, signed=False),
    # olen1: Read (delta + 1) bytes as an unsigned number if less than 4 bytes,
    # as a signed number if 4 bytes.
    olen1=lambda dvi, delta: dvi._arg(delta + 1, signed=(delta == 3)),
)

def _dispatch(table, min, max=None, state=None, args=('raw',)):
    """
    Decorator for dispatch by opcode. Sets the values in *table*
    from *min* to *max* to this method, adds a check that the Dvi state
    matches *state* if not None, reads arguments from the file according
    to *args*.

    Parameters
    ----------
    table : dict[int, callable]
        The dispatch table to be filled in.

    min, max : int
        Range of opcodes that calls the registered function; *max* defaults to
        *min*.

    state : _dvistate, optional
        State of the Dvi object in which these opcodes are allowed.

    args : list[str], default: ['raw']
        Sequence of argument specifications:

        - 'raw': opcode minus minimum
        - 'u1': read one unsigned byte
        - 'u4': read four bytes, treat as an unsigned number
        - 's4': read four bytes, treat as a signed number
        - 'slen': read (opcode - minimum) bytes, treat as signed
        - 'slen1': read (opcode - minimum + 1) bytes, treat as signed
        - 'ulen1': read (opcode - minimum + 1) bytes, treat as unsigned
        - 'olen1': read (opcode - minimum + 1) bytes, treat as unsigned
          if under four bytes, signed if four bytes
    """
    def decorate(method):
        get_args = [_arg_mapping[x] for x in args]

        @wraps(method)
        def wrapper(self, byte):
            # Check if state precondition is satisfied
            if state is not None and self.state != state:
                raise ValueError("state precondition failed")
            # Call the decorated method with arguments determined by get_args
            return method(self, *[f(self, byte-min) for f in get_args])
        
        # Register wrapper function in the dispatch table
        if max is None:
            table[min] = wrapper
        else:
            for i in range(min, max+1):
                # Ensure that the table entry is initially None
                assert table[i] is None
                table[i] = wrapper
        
        return wrapper
    return decorate



# 返回函数装饰器对象


这行代码将函数 `decorate` 作为结果返回。
class Dvi:
    """
    A reader for a dvi ("device-independent") file, as produced by TeX.

    The current implementation can only iterate through pages in order,
    and does not even attempt to verify the postamble.

    This class can be used as a context manager to close the underlying
    file upon exit. Pages can be read via iteration. Here is an overly
    simple way to extract text without trying to detect whitespace::

        >>> with matplotlib.dviread.Dvi('input.dvi', 72) as dvi:
        ...     for page in dvi:
        ...         print(''.join(chr(t.glyph) for t in page.text))
    """

    # dispatch table
    # 创建一个长度为 256 的列表作为调度表格，初始值均为 None
    _dtable = [None] * 256
    # 使用偏函数 partial 将 _dispatch 函数与 _dtable 绑定
    _dispatch = partial(_dispatch, _dtable)

    def __init__(self, filename, dpi):
        """
        Read the data from the file named *filename* and convert
        TeX's internal units to units of *dpi* per inch.
        *dpi* only sets the units and does not limit the resolution.
        Use None to return TeX's internal units.
        """
        # 记录调试信息，输出 Dvi 文件名
        _log.debug('Dvi: %s', filename)
        # 打开给定文件名的文件，以二进制模式
        self.file = open(filename, 'rb')
        # 设置 DPI（每英寸点数），用于单位转换
        self.dpi = dpi
        # 字典，用于存储字体信息
        self.fonts = {}
        # 设定初始状态为 _dvistate.pre
        self.state = _dvistate.pre
        # 初始化为 None，表示没有丢失的字体
        self._missing_font = None

    def __enter__(self):
        """Context manager enter method, does nothing."""
        return self

    def __exit__(self, etype, evalue, etrace):
        """
        Context manager exit method, closes the underlying file if it is open.
        """
        # 调用 close 方法关闭文件
        self.close()

    def __iter__(self):
        """
        Iterate through the pages of the file.

        Yields
        ------
        Page
            Details of all the text and box objects on the page.
            The Page tuple contains lists of Text and Box tuples and
            the page dimensions, and the Text and Box tuples contain
            coordinates transformed into a standard Cartesian
            coordinate system at the dpi value given when initializing.
            The coordinates are floating point numbers, but otherwise
            precision is not lost and coordinate values are not clipped to
            integers.
        """
        # 当读取到新页面时，通过 _output 方法生成 Page 对象并返回
        while self._read():
            yield self._output()

    def close(self):
        """Close the underlying file if it is open."""
        # 如果文件没有关闭，则关闭文件
        if not self.file.closed:
            self.file.close()
    def _output(self):
        """
        Output the text and boxes belonging to the most recent page.
        page = dvi._output()
        """
        # Initialize variables to track minimum and maximum coordinates
        minx, miny, maxx, maxy = np.inf, np.inf, -np.inf, -np.inf
        maxy_pure = -np.inf
        
        # Iterate over each element in self.text and self.boxes
        for elt in self.text + self.boxes:
            if isinstance(elt, Box):
                # If element is a Box object
                x, y, h, w = elt
                e = 0  # zero depth
            else:  # If element is a glyph
                x, y, font, g, w = elt
                # Calculate height and depth of the glyph using font information
                h, e = font._height_depth_of(g)
            
            # Update minimum and maximum coordinates
            minx = min(minx, x)
            miny = min(miny, y - h)
            maxx = max(maxx, x + w)
            maxy = max(maxy, y + e)
            maxy_pure = max(maxy_pure, y)
        
        # Adjust maxy_pure if _baseline_v is defined
        if self._baseline_v is not None:
            maxy_pure = self._baseline_v  # This should normally be the case.
            self._baseline_v = None

        # Handle case where there are no text or boxes
        if not self.text and not self.boxes:
            return Page(text=[], boxes=[], width=0, height=0, descent=0)

        # Handle case where dpi is not defined
        if self.dpi is None:
            # Output raw DVI coordinates
            return Page(text=self.text, boxes=self.boxes,
                        width=maxx-minx, height=maxy_pure-miny,
                        descent=maxy-maxy_pure)

        # Convert from TeX's "scaled points" to dpi units
        d = self.dpi / (72.27 * 2**16)
        descent = (maxy - maxy_pure) * d

        # Convert coordinates for text elements to dpi units
        text = [Text((x-minx)*d, (maxy-y)*d - descent, f, g, w*d)
                for (x, y, f, g, w) in self.text]
        
        # Convert coordinates for box elements to dpi units
        boxes = [Box((x-minx)*d, (maxy-y)*d - descent, h*d, w*d)
                 for (x, y, h, w) in self.boxes]

        # Return a Page object with converted coordinates and dimensions
        return Page(text=text, boxes=boxes, width=(maxx-minx)*d,
                    height=(maxy_pure-miny)*d, descent=descent)
    def _read(self):
        """
        Read one page from the file. Return True if successful,
        False if there were no more pages.
        """
        # Pages appear to start with the sequence
        #   bop (begin of page)
        #   xxx comment
        #   <push, ..., pop>  # if using chemformula
        #   down
        #   push
        #     down
        #     <push, push, xxx, right, xxx, pop, pop>  # if using xcolor
        #     down
        #     push
        #       down (possibly multiple)
        #       push  <=  here, v is the baseline position.
        #         etc.
        # (dviasm is useful to explore this structure.)
        # Thus, we use the vertical position at the first time the stack depth
        # reaches 3, while at least three "downs" have been executed (excluding
        # those popped out (corresponding to the chemformula preamble)), as the
        # baseline (the "down" count is necessary to handle xcolor).
        
        down_stack = [0]  # 初始化一个堆栈，用于跟踪“down”指令的数量
        self._baseline_v = None  # 初始化基线位置
        
        while True:
            byte = self.file.read(1)[0]  # 读取文件的下一个字节
            self._dtable[byte](self, byte)  # 根据字节调用相应的处理函数
            if self._missing_font:
                raise self._missing_font  # 如果缺少字体，则抛出异常
            
            name = self._dtable[byte].__name__  # 获取处理函数的名称
            if name == "_push":
                down_stack.append(down_stack[-1])  # 处理"push"指令，堆栈深度加一
            elif name == "_pop":
                down_stack.pop()  # 处理"pop"指令，堆栈深度减一
            elif name == "_down":
                down_stack[-1] += 1  # 处理"down"指令，当前堆栈顶部的值加一
            
            # 在堆栈深度为3，并且至少执行了四个"down"指令时，记录垂直位置作为基线
            if (self._baseline_v is None
                    and len(getattr(self, "stack", [])) == 3
                    and down_stack[-1] >= 4):
                self._baseline_v = self.v
            
            if byte == 140:                         # 页面结束的标志
                return True
            if self.state is _dvistate.post_post:   # 文件结束的标志
                self.close()
                return False

    def _arg(self, nbytes, signed=False):
        """
        Read and return a big-endian integer *nbytes* long.
        Signedness is determined by the *signed* keyword.
        """
        return int.from_bytes(self.file.read(nbytes), "big", signed=signed)

    @_dispatch(min=0, max=127, state=_dvistate.inpage)
    def _set_char_immediate(self, char):
        self._put_char_real(char)
        if isinstance(self.fonts[self.f], FileNotFoundError):
            return
        self.h += self.fonts[self.f]._width_of(char)

    @_dispatch(min=128, max=131, state=_dvistate.inpage, args=('olen1',))
    def _set_char(self, char):
        self._put_char_real(char)
        if isinstance(self.fonts[self.f], FileNotFoundError):
            return
        self.h += self.fonts[self.f]._width_of(char)

    @_dispatch(132, state=_dvistate.inpage, args=('s4', 's4'))
    def _set_rule(self, a, b):
        self._put_rule_real(a, b)
        self.h += b

    @_dispatch(min=133, max=136, state=_dvistate.inpage, args=('olen1',))
    def _put_char(self, char):
        self._put_char_real(char)
    def _put_char_real(self, char):
        font = self.fonts[self.f]  # 获取当前字体对象
        if isinstance(font, FileNotFoundError):  # 检查字体对象是否为 FileNotFoundError 类型
            self._missing_font = font  # 如果是 FileNotFoundError 类型，则将其赋值给 _missing_font
        elif font._vf is None:  # 检查字体对象的 _vf 属性是否为 None
            # 如果 _vf 属性为 None，则创建新的 Text 对象并添加到 self.text 列表中
            self.text.append(Text(self.h, self.v, font, char, font._width_of(char)))
        else:
            # 如果 _vf 属性不为 None，则按照字体对象的 _vf[char].text 属性的信息创建新的 Text 对象，并添加到 self.text 列表中
            scale = font._scale
            for x, y, f, g, w in font._vf[char].text:
                newf = DviFont(scale=_mul2012(scale, f._scale),
                               tfm=f._tfm, texname=f.texname, vf=f._vf)
                self.text.append(Text(self.h + _mul2012(x, scale),
                                      self.v + _mul2012(y, scale),
                                      newf, g, newf._width_of(g)))
            # 根据字体对象的 _vf[char].boxes 属性的信息创建新的 Box 对象，并添加到 self.boxes 列表中
            self.boxes.extend([Box(self.h + _mul2012(x, scale),
                                   self.v + _mul2012(y, scale),
                                   _mul2012(a, scale), _mul2012(b, scale))
                               for x, y, a, b in font._vf[char].boxes])

    @_dispatch(137, state=_dvistate.inpage, args=('s4', 's4'))
    def _put_rule(self, a, b):
        self._put_rule_real(a, b)  # 调用 _put_rule_real 方法处理传入的参数 a 和 b

    def _put_rule_real(self, a, b):
        if a > 0 and b > 0:
            # 如果参数 a 和 b 均大于 0，则创建新的 Box 对象并添加到 self.boxes 列表中
            self.boxes.append(Box(self.h, self.v, a, b))

    @_dispatch(138)
    def _nop(self, _):
        pass  # 空操作，什么也不做

    @_dispatch(139, state=_dvistate.outer, args=('s4',)*11)
    def _bop(self, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, p):
        # 初始化页面状态及相关属性，创建空列表 self.text 和 self.boxes
        self.state = _dvistate.inpage
        self.h, self.v, self.w, self.x, self.y, self.z = 0, 0, 0, 0, 0, 0
        self.stack = []
        self.text = []          # Text 对象的列表
        self.boxes = []         # Box 对象的列表

    @_dispatch(140, state=_dvistate.inpage)
    def _eop(self, _):
        # 结束页面状态，删除页面状态相关的属性
        self.state = _dvistate.outer
        del self.h, self.v, self.w, self.x, self.y, self.z, self.stack

    @_dispatch(141, state=_dvistate.inpage)
    def _push(self, _):
        # 将当前页面状态的坐标等信息压入栈中
        self.stack.append((self.h, self.v, self.w, self.x, self.y, self.z))

    @_dispatch(142, state=_dvistate.inpage)
    def _pop(self, _):
        # 从栈中弹出并恢复页面状态的坐标等信息
        self.h, self.v, self.w, self.x, self.y, self.z = self.stack.pop()

    @_dispatch(min=143, max=146, state=_dvistate.inpage, args=('slen1',))
    def _right(self, b):
        # 将当前横坐标 self.h 向右移动 b 个单位
        self.h += b

    @_dispatch(min=147, max=151, state=_dvistate.inpage, args=('slen',))
    def _right_w(self, new_w):
        if new_w is not None:
            self.w = new_w  # 如果有指定新的宽度 new_w，则更新 self.w
        self.h += self.w  # 将当前横坐标 self.h 向右移动 self.w 个单位

    @_dispatch(min=152, max=156, state=_dvistate.inpage, args=('slen',))
    def _right_x(self, new_x):
        if new_x is not None:
            self.x = new_x  # 如果有指定新的 x 值 new_x，则更新 self.x
        self.h += self.x  # 将当前横坐标 self.h 向右移动 self.x 个单位

    @_dispatch(min=157, max=160, state=_dvistate.inpage, args=('slen1',))
    def _down(self, a):
        # 将当前纵坐标 self.v 向下移动 a 个单位
        self.v += a

    @_dispatch(min=161, max=165, state=_dvistate.inpage, args=('slen',))
    def _down_y(self, new_y):
        if new_y is not None:
            self.y = new_y  # 如果有指定新的 y 值 new_y，则更新 self.y
        self.v += self.y  # 将当前纵坐标 self.v 向下移动 self.y 个单位
    # 定义一个装饰器函数，用于处理 DVI 文件中的特定命令范围，处理 _down_z 方法
    @_dispatch(min=166, max=170, state=_dvistate.inpage, args=('slen',))
    def _down_z(self, new_z):
        # 如果传入了新的 z 值，则更新当前对象的 z 属性
        if new_z is not None:
            self.z = new_z
        # 更新对象的 v 属性，增加 z 的值
        self.v += self.z
    
    # 定义一个装饰器函数，处理 _fnt_num_immediate 方法
    @_dispatch(min=171, max=234, state=_dvistate.inpage)
    def _fnt_num_immediate(self, k):
        # 设置对象的 f 属性为给定的 k 值
        self.f = k
    
    # 定义一个装饰器函数，处理 _fnt_num 方法
    @_dispatch(min=235, max=238, state=_dvistate.inpage, args=('olen1',))
    def _fnt_num(self, new_f):
        # 设置对象的 f 属性为给定的 new_f 值
        self.f = new_f
    
    # 定义一个装饰器函数，处理 _xxx 方法
    @_dispatch(min=239, max=242, args=('ulen1',))
    def _xxx(self, datalen):
        # 从文件中读取特定长度的数据到 special 变量
        special = self.file.read(datalen)
        # 调试输出遇到的特殊数据，转换非打印字符为十六进制形式
        _log.debug(
            'Dvi._xxx: encountered special: %s',
            ''.join([chr(ch) if 32 <= ch < 127 else '<%02x>' % ch
                     for ch in special]))
    
    # 定义一个装饰器函数，处理 _fnt_def 方法
    @_dispatch(min=243, max=246, args=('olen1', 'u4', 'u4', 'u4', 'u1', 'u1'))
    def _fnt_def(self, k, c, s, d, a, l):
        # 调用 _fnt_def_real 方法，传入相应的参数
        self._fnt_def_real(k, c, s, d, a, l)
    
    # 定义一个实际处理 _fnt_def 方法的函数
    def _fnt_def_real(self, k, c, s, d, a, l):
        # 从文件中读取长度为 a+l 的数据到 n 变量
        n = self.file.read(a + l)
        # 将最后 l 个字节解码为 ASCII 字符串，得到字体名称 fontname
        fontname = n[-l:].decode('ascii')
        try:
            # 根据字体名称获取对应的 tfm 文件信息
            tfm = _tfmfile(fontname)
        except FileNotFoundError as exc:
            # 如果文件未找到，将异常存储到字体字典中，并返回
            self.fonts[k] = exc
            return
        # 检查 tfm 校验和，如果不为零且与传入的 c 值不相等，则引发 ValueError 异常
        if c != 0 and tfm.checksum != 0 and c != tfm.checksum:
            raise ValueError('tfm checksum mismatch: %s' % n)
        try:
            # 根据字体名称获取对应的 vf 文件信息
            vf = _vffile(fontname)
        except FileNotFoundError:
            vf = None
        # 将字体信息存储到字体字典中，包括比例因子、tfm 文件信息、字体名称、vf 文件信息
        self.fonts[k] = DviFont(scale=s, tfm=tfm, texname=n, vf=vf)
    
    # 定义一个装饰器函数，处理 _pre 方法
    @_dispatch(247, state=_dvistate.pre, args=('u1', 'u4', 'u4', 'u4', 'u1'))
    def _pre(self, i, num, den, mag, k):
        # 从文件中读取 k 长度的数据，这是 DVI 文件中的注释
        self.file.read(k)  # comment in the dvi file
        # 如果 i 不等于 2，则引发 ValueError 异常，表示未知的 DVI 格式
        if i != 2:
            raise ValueError("Unknown dvi format %d" % i)
        # 如果 num 和 den 不等于标准值，引发 ValueError 异常，表示非标准单位
        if num != 25400000 or den != 7227 * 2**16:
            raise ValueError("Nonstandard units in dvi file")
        # 如果 mag 不等于 1000，则引发 ValueError 异常，表示非标准放大倍数
        if mag != 1000:
            raise ValueError("Nonstandard magnification in dvi file")
        # 将对象状态设置为 outer，表示处理完前导数据
        self.state = _dvistate.outer
    
    # 定义一个装饰器函数，处理 _post 方法
    @_dispatch(248, state=_dvistate.outer)
    def _post(self, _):
        # 将对象状态设置为 post_post，表示处理完后续数据
        self.state = _dvistate.post_post
        # TODO: 实际上读取尾注和结尾的内容？
        # 目前 post_post 只触发关闭文件操作
    
    # 定义一个装饰器函数，处理 _dispatch 中指定的命令
    @_dispatch(249)
    def _placeholder(self):
        pass  # Placeholder function, currently does nothing
    # 定义一个未实现的方法 `_post_post`，用于在子类中实现具体逻辑
    def _post_post(self, _):
        # 抛出未实现错误，提示该方法需要在子类中被实现
        raise NotImplementedError

    # 使用装饰器 `_dispatch` 标记 `_malformed` 方法，设置参数范围为 250 到 255
    @_dispatch(min=250, max=255)
    def _malformed(self, offset):
        # 抛出值错误，指示在偏移量 250 + offset 处发现未知命令
        raise ValueError(f"unknown command: byte {250 + offset}")
class DviFont:
    """
    Encapsulation of a font that a DVI file can refer to.

    This class holds a font's texname and size, supports comparison,
    and knows the widths of glyphs in the same units as the AFM file.
    There are also internal attributes (for use by dviread.py) that
    are *not* used for comparison.

    The size is in Adobe points (converted from TeX points).

    Parameters
    ----------
    scale : float
        Factor by which the font is scaled from its natural size.
    tfm : Tfm
        TeX font metrics for this font
    texname : bytes
       Name of the font as used internally by TeX and friends, as an ASCII
       bytestring.  This is usually very different from any external font
       names; `PsfontsMap` can be used to find the external name of the font.
    vf : Vf
       A TeX "virtual font" file, or None if this font is not virtual.

    Attributes
    ----------
    texname : bytes
        Internal name of the font as used by TeX.
    size : float
        Size of the font in Adobe points, converted from TeX points.
    widths : list
        List of glyph widths in units relative to the point size.
    """

    __slots__ = ('texname', 'size', 'widths', '_scale', '_vf', '_tfm')

    def __init__(self, scale, tfm, texname, vf):
        # 检查 texname 是否为 bytes 类型
        _api.check_isinstance(bytes, texname=texname)
        # 设置私有属性 _scale 和 _tfm
        self._scale = scale
        self._tfm = tfm
        # 设置公共属性 texname 和 _vf
        self.texname = texname
        self._vf = vf
        # 计算并设置字体大小 size，转换自 TeX 点到 Adobe 点
        self.size = scale * (72.0 / (72.27 * 2**16))
        try:
            # 尝试获取 tfm.width 中的最大值，如果为空则设为 0
            nchars = max(tfm.width) + 1
        except ValueError:
            nchars = 0
        # 生成 widths 列表，包含每个字符的宽度，单位是相对于点大小的 1/1000
        self.widths = [(1000*tfm.width.get(char, 0)) >> 20
                       for char in range(nchars)]

    def __eq__(self, other):
        # 检查两个对象是否相等
        return (type(self) is type(other)
                and self.texname == other.texname and self.size == other.size)

    def __ne__(self, other):
        # 检查两个对象是否不相等
        return not self.__eq__(other)

    def __repr__(self):
        # 返回对象的字符串表示形式，用于调试和显示
        return f"<{type(self).__name__}: {self.texname}>"

    def _width_of(self, char):
        """Width of char in dvi units."""
        # 获取字符 char 的宽度，使用私有属性 _tfm 和 _scale 进行计算
        width = self._tfm.width.get(char, None)
        if width is not None:
            return _mul2012(width, self._scale)
        # 如果找不到字符宽度，则记录日志并返回 0
        _log.debug('No width for char %d in font %s.', char, self.texname)
        return 0
    # 计算字符在 DVI 单位中的高度和深度
    def _height_depth_of(self, char):
        """Height and depth of char in dvi units."""
        # 初始化空列表用于存放结果
        result = []
        # 遍历高度和深度的度量值和名称对
        for metric, name in ((self._tfm.height, "height"),
                             (self._tfm.depth, "depth")):
            # 获取字符在度量值中的数值，如果不存在则记录日志并添加0
            value = metric.get(char, None)
            if value is None:
                _log.debug('No %s for char %d in font %s',
                           name, char, self.texname)
                result.append(0)
            else:
                # 将数值乘以缩放比例，并添加到结果列表中
                result.append(_mul2012(value, self._scale))
        
        # 对于符号字体（如 cmsyXX），其中字符0（"minus"）有一个非零下降值，
        # 但我们实际上关心的是栅格化深度，以便对齐由 dvipng 生成的图像。
        if re.match(br'^cmsy\d+$', self.texname) and char == 0:
            # 如果符合特定条件，将最后一个结果设置为0，以确保对齐
            result[-1] = 0
        
        # 返回结果列表，包含字符的高度和深度
        return result
# 定义一个名为 Vf 的类，继承自 Dvi 类
class Vf(Dvi):
    r"""
    A virtual font (\*.vf file) containing subroutines for dvi files.

    Parameters
    ----------
    filename : str or path-like
        虚拟字体文件的文件名或路径

    Notes
    -----
    The virtual font format is a derivative of dvi:
    http://mirrors.ctan.org/info/knuth/virtual-fonts
    This class reuses some of the machinery of `Dvi`
    but replaces the `_read` loop and dispatch mechanism.
    虚拟字体格式是 dvi 的一种衍生格式，详细信息参见上述链接。
    该类重用了 `Dvi` 的某些机制，但替换了 `_read` 循环和调度机制。

    Examples
    --------
    ::

        vf = Vf(filename)
        glyph = vf[code]
        glyph.text, glyph.boxes, glyph.width
    """

    # 初始化方法，接受一个文件名参数 filename
    def __init__(self, filename):
        # 调用父类 Dvi 的初始化方法
        super().__init__(filename, 0)
        try:
            # 初始化实例变量
            self._first_font = None
            self._chars = {}
            # 调用本类的 _read 方法读取虚拟字体文件内容
            self._read()
        finally:
            # 在 finally 块中调用 close 方法确保资源释放
            self.close()

    # 实现索引运算符 [] 的方法，以 code 作为键返回 _chars 字典中对应的值
    def __getitem__(self, code):
        return self._chars[code]
    def _read(self):
        """
        Read one page from the file. Return True if successful,
        False if there were no more pages.
        """
        packet_char, packet_ends = None, None
        packet_len, packet_width = None, None
        while True:
            byte = self.file.read(1)[0]
            # 如果处于页面内部，执行 DVI 指令
            if self.state is _dvistate.inpage:
                # 计算当前字节位置
                byte_at = self.file.tell()-1
                # 如果到达包结尾，完成包处理并准备退出包状态
                if byte_at == packet_ends:
                    self._finalize_packet(packet_char, packet_width)
                    packet_len, packet_char, packet_width = None, None, None
                    # 继续执行非包状态的代码
                elif byte_at > packet_ends:
                    raise ValueError("Packet length mismatch in vf file")
                else:
                    # 处理非结尾字节的合适操作码
                    if byte in (139, 140) or byte >= 243:
                        raise ValueError(
                            "Inappropriate opcode %d in vf file" % byte)
                    Dvi._dtable[byte](self, byte)
                    continue

            # 处于非包状态
            if byte < 242:          # 短包 (长度由字节给出)
                packet_len = byte
                packet_char, packet_width = self._arg(1), self._arg(3)
                packet_ends = self._init_packet(byte)
                self.state = _dvistate.inpage
            elif byte == 242:       # 长包
                packet_len, packet_char, packet_width = \
                            [self._arg(x) for x in (4, 4, 4)]
                self._init_packet(packet_len)
            elif 243 <= byte <= 246:
                k = self._arg(byte - 242, byte == 246)
                c, s, d, a, l = [self._arg(x) for x in (4, 4, 4, 1, 1)]
                self._fnt_def_real(k, c, s, d, a, l)
                if self._first_font is None:
                    self._first_font = k
            elif byte == 247:       # 前导部分
                i, k = self._arg(1), self._arg(1)
                x = self.file.read(k)
                cs, ds = self._arg(4), self._arg(4)
                self._pre(i, x, cs, ds)
            elif byte == 248:       # 后文 (一系列 248)
                break
            else:
                raise ValueError("Unknown vf opcode %d" % byte)

    def _init_packet(self, pl):
        # 如果不处于外部状态，抛出异常
        if self.state != _dvistate.outer:
            raise ValueError("Misplaced packet in vf file")
        # 初始化各个变量
        self.h, self.v, self.w, self.x, self.y, self.z = 0, 0, 0, 0, 0, 0
        self.stack, self.text, self.boxes = [], [], []
        self.f = self._first_font
        self._missing_font = None
        # 返回下一个数据包的位置
        return self.file.tell() + pl
    # 如果没有缺失字体信息，则将当前字符（packet_char）的页面信息存入 _chars 字典中
    # 这包括文本内容、框信息、宽度（packet_width），高度（None）、下降高度（None）
    def _finalize_packet(self, packet_char, packet_width):
        if not self._missing_font:  # 如果没有缺失字体信息
            self._chars[packet_char] = Page(
                text=self.text, boxes=self.boxes, width=packet_width,
                height=None, descent=None)
        self.state = _dvistate.outer  # 设置状态为 _dvistate.outer，表示处理完成

    # 处理 VF 文件中的预设（pre）命令，确保状态处于预设状态
    def _pre(self, i, x, cs, ds):
        if self.state is not _dvistate.pre:  # 如果当前状态不是预设状态，则抛出错误
            raise ValueError("pre command in middle of vf file")
        if i != 202:  # 如果 i 不等于 202，表示不识别的 VF 文件格式
            raise ValueError("Unknown vf format %d" % i)
        if len(x):  # 如果 x 非空，记录 VF 文件的注释信息
            _log.debug('vf file comment: %s', x)
        self.state = _dvistate.outer  # 设置状态为 _dvistate.outer，表示处理完成
        # cs = checksum, ds = design size
def _mul2012(num1, num2):
    """Multiply two numbers in 20.12 fixed point format."""
    # 以 20.12 固定点格式相乘，将结果向右移动 20 位，得到整数部分
    return (num1*num2) >> 20


class Tfm:
    """
    A TeX Font Metric file.

    This implementation covers only the bare minimum needed by the Dvi class.

    Parameters
    ----------
    filename : str or path-like

    Attributes
    ----------
    checksum : int
       Used for verifying against the dvi file.
    design_size : int
       Design size of the font (unknown units)
    width, height, depth : dict
       Dimensions of each character, need to be scaled by the factor
       specified in the dvi file. These are dicts because indexing may
       not start from 0.
    """
    __slots__ = ('checksum', 'design_size', 'width', 'height', 'depth')

    def __init__(self, filename):
        _log.debug('opening tfm file %s', filename)
        # 使用二进制读取文件
        with open(filename, 'rb') as file:
            # 读取文件头部信息
            header1 = file.read(24)
            lh, bc, ec, nw, nh, nd = struct.unpack('!6H', header1[2:14])
            _log.debug('lh=%d, bc=%d, ec=%d, nw=%d, nh=%d, nd=%d',
                       lh, bc, ec, nw, nh, nd)
            # 读取文件头部后续信息
            header2 = file.read(4*lh)
            # 解析头部信息中的校验和和设计尺寸
            self.checksum, self.design_size = struct.unpack('!2I', header2[:8])
            # 读取字符信息，宽度、高度、深度信息
            char_info = file.read(4*(ec-bc+1))
            widths = struct.unpack(f'!{nw}i', file.read(4*nw))
            heights = struct.unpack(f'!{nh}i', file.read(4*nh))
            depths = struct.unpack(f'!{nd}i', file.read(4*nd))
        # 初始化字符宽度、高度、深度字典
        self.width, self.height, self.depth = {}, {}, {}
        # 根据索引和字符信息填充宽度、高度、深度字典
        for idx, char in enumerate(range(bc, ec+1)):
            byte0 = char_info[4*idx]
            byte1 = char_info[4*idx+1]
            self.width[char] = widths[byte0]
            self.height[char] = heights[byte1 >> 4]
            self.depth[char] = depths[byte1 & 0xf]


PsFont = namedtuple('PsFont', 'texname psname effects encoding filename')


class PsfontsMap:
    """
    A psfonts.map formatted file, mapping TeX fonts to PS fonts.

    Parameters
    ----------
    filename : str or path-like

    Notes
    -----
    For historical reasons, TeX knows many Type-1 fonts by different
    names than the outside world. (For one thing, the names have to
    fit in eight characters.) Also, TeX's native fonts are not Type-1
    but Metafont, which is nontrivial to convert to PostScript except
    as a bitmap. While high-quality conversions to Type-1 format exist
    and are shipped with modern TeX distributions, we need to know
    which Type-1 fonts are the counterparts of which native fonts. For
    these reasons a mapping is needed from internal font names to font
    file names.

    A texmf tree typically includes mapping files called e.g.
    :file:`psfonts.map`, :file:`pdftex.map`, or :file:`dvipdfm.map`.
    The file :file:`psfonts.map` is used by :program:`dvips`,
    """
    # PS 字体映射类，用于将 TeX 字体映射到 PS 字体

    def __init__(self, filename):
        _log.debug('opening psfonts.map file %s', filename)
        # 打开 psfonts.map 文件并读取
        with open(filename, 'r') as file:
            pass
            # 这里可以继续处理 psfonts.map 文件的读取逻辑
    """
        :file:`pdftex.map` by :program:`pdfTeX`, and :file:`dvipdfm.map`
        by :program:`dvipdfm`. :file:`psfonts.map` might avoid embedding
        the 35 PostScript fonts (i.e., have no filename for them, as in
        the Times-Bold example above), while the pdf-related files perhaps
        only avoid the "Base 14" pdf fonts. But the user may have
        configured these files differently.
    
        Examples
        --------
        >>> map = PsfontsMap(find_tex_file('pdftex.map'))
        >>> entry = map[b'ptmbo8r']
        >>> entry.texname
        b'ptmbo8r'
        >>> entry.psname
        b'Times-Bold'
        >>> entry.encoding
        '/usr/local/texlive/2008/texmf-dist/fonts/enc/dvips/base/8r.enc'
        >>> entry.effects
        {'slant': 0.16700000000000001}
        >>> entry.filename
        """
    
        # 定义一个包含指定属性的 PsfontsMap 类，以缓存文件名与 PsfontsMap 对象的映射
        __slots__ = ('_filename', '_unparsed', '_parsed')
    
        # 使用 LRU 缓存装饰器，创建 PsfontsMap 对象的新实例
        @lru_cache
        def __new__(cls, filename):
            # 创建一个新的 PsfontsMap 对象
            self = object.__new__(cls)
            # 解码文件名以适应操作系统
            self._filename = os.fsdecode(filename)
            # 打开指定的文件名
            with open(filename, 'rb') as file:
                # 初始化未解析的行字典
                self._unparsed = {}
                # 遍历文件的每一行
                for line in file:
                    # 以空格分割行，获取第一个单词作为 tfmname
                    tfmname = line.split(b' ', 1)[0]
                    # 将行按 tfmname 分类存储在未解析字典中
                    self._unparsed.setdefault(tfmname, []).append(line)
            # 初始化已解析字典
            self._parsed = {}
            # 返回创建的 PsfontsMap 对象
            return self
    
        # 获取指定 texname 对应的条目
        def __getitem__(self, texname):
            # 确保 texname 是字节类型
            assert isinstance(texname, bytes)
            # 如果 texname 在未解析字典中
            if texname in self._unparsed:
                # 遍历未解析字典中 texname 对应的所有行
                for line in self._unparsed.pop(texname):
                    # 如果解析并缓存了该行，则返回解析结果
                    if self._parse_and_cache_line(line):
                        break
            try:
                # 返回解析后的结果
                return self._parsed[texname]
            except KeyError:
                # 如果 texname 不存在于已解析字典中，则抛出 LookupError 异常
                raise LookupError(
                    f"An associated PostScript font (required by Matplotlib) "
                    f"could not be found for TeX font {texname.decode('ascii')!r} "
                    f"in {self._filename!r}; this problem can often be solved by "
                    f"installing a suitable PostScript font package in your TeX "
                    f"package manager") from None
def _parse_enc(path):
    r"""
    Parse a \*.enc file referenced from a psfonts.map style file.

    The format supported by this function is a tiny subset of PostScript.

    Parameters
    ----------
    path : `os.PathLike`
        文件路径，用于读取 \*.enc 文件。

    Returns
    -------
    list
        列表中的第 n 个条目是第 n 个字形的 PostScript 字形名。
    """
    # 读取文件内容并去除注释行
    no_comments = re.sub("%.*", "", Path(path).read_text(encoding="ascii"))
    # 从无注释内容中提取包含在方括号中的部分
    array = re.search(r"(?s)\[(.*)\]", no_comments).group(1)
    # 将提取出的内容按空白分割成行，并且去除空行
    lines = [line for line in array.split() if line]
    # 如果所有行都以斜杠开头，则移除斜杠并作为结果返回
    if all(line.startswith("/") for line in lines):
        return [line[1:] for line in lines]
    else:
        # 如果格式不符合预期，则引发 ValueError 异常
        raise ValueError(f"Failed to parse {path} as Postscript encoding")


class _LuatexKpsewhich:
    @lru_cache  # A singleton.
    def __new__(cls):
        self = object.__new__(cls)
        # 创建 luatex 进程用于执行 kpsewhich.lua 脚本
        self._proc = self._new_proc()
        return self

    def _new_proc(self):
        # 启动 luatex 进程，并执行 kpsewhich.lua 脚本
        return subprocess.Popen(
            ["luatex", "--luaonly",
             str(cbook._get_data_path("kpsewhich.lua"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def search(self, filename):
        # 检查当前 luatex 进程是否终止，如果是则重新启动
        if self._proc.poll() is not None:  # Dead, restart it.
            self._proc = self._new_proc()
        # 向 luatex 进程的标准输入发送文件名并刷新
        self._proc.stdin.write(os.fsencode(filename) + b"\n")
        self._proc.stdin.flush()
        # 读取 luatex 进程的标准输出，并解码结果
        out = self._proc.stdout.readline().rstrip()
        # 如果输出为 "nil"，则返回 None；否则解码输出结果并返回
        return None if out == b"nil" else os.fsdecode(out)


@lru_cache
def find_tex_file(filename):
    """
    Find a file in the texmf tree using kpathsea_.

    The kpathsea library, provided by most existing TeX distributions, both
    on Unix-like systems and on Windows (MikTeX), is invoked via a long-lived
    luatex process if luatex is installed, or via kpsewhich otherwise.

    .. _kpathsea: https://www.tug.org/kpathsea/

    Parameters
    ----------
    filename : str or path-like
        要在 texmf 树中查找的文件名。

    Raises
    ------
    FileNotFoundError
        如果找不到文件。
    """

    # 我们期望这些始终是 ascii 编码，但出于谨慎考虑使用 utf-8
    if isinstance(filename, bytes):
        # 如果输入文件名为字节字符串，则将其解码为 utf-8 格式
        filename = filename.decode('utf-8', errors='replace')

    try:
        # 尝试创建 _LuatexKpsewhich 的实例
        lk = _LuatexKpsewhich()
    except FileNotFoundError:
        # 如果找不到 luatex，退回直接调用 kpsewhich 的方式
        lk = None  # Fallback to directly calling kpsewhich, as below.

    if lk:
        # 使用 _LuatexKpsewhich 实例搜索指定的文件名
        path = lk.search(filename)
    else:
        if sys.platform == 'win32':
            # 在 Windows 平台上，kpathsea 可以使用 utf-8 编码处理命令行参数和输出。
            # 设置 `command_line_encoding` 环境变量以强制其始终使用 utf-8 编码。参见 Matplotlib 问题 #11848。
            kwargs = {'env': {**os.environ, 'command_line_encoding': 'utf-8'},
                      'encoding': 'utf-8'}
        else:  # 在 POSIX 上，通过等效的 os.fsdecode() 处理。
            kwargs = {'encoding': sys.getfilesystemencoding(),
                      'errors': 'surrogateescape'}

        try:
            # 调用 `kpsewhich` 命令查找 `filename` 文件，并通过 `_log` 记录。
            # 使用给定的 kwargs 进行子进程调用。
            path = (cbook._check_and_log_subprocess(['kpsewhich', filename],
                                                    _log, **kwargs)
                    .rstrip('\n'))
        except (FileNotFoundError, RuntimeError):
            # 如果发生文件未找到或运行时错误，将 path 设置为 None。
            path = None

    if path:
        # 如果找到了文件路径，则返回该路径。
        return path
    else:
        # 如果未找到文件路径，则抛出 FileNotFoundError 异常。
        raise FileNotFoundError(
            f"Matplotlib 的 TeX 实现搜索了名为 {filename!r} 的文件，但在您的 texmf 树中找不到它")
# 导入必要的模块和函数装饰器
if __name__ == '__main__':
    # 导入参数解析模块
    from argparse import ArgumentParser
    # 导入 itertools 模块，用于分组操作
    import itertools

    # 创建参数解析器对象
    parser = ArgumentParser()
    # 添加位置参数 'filename'，表示要处理的文件名
    parser.add_argument("filename")
    # 添加可选参数 'dpi'，表示像素密度，默认为 None
    args = parser.parse_args()

    # 使用 'with' 上下文管理，打开 Dvi 文件，并自动关闭
    with Dvi(args.filename, args.dpi) as dvi:
        # 使用 'PsfontsMap' 类查找 'pdftex.map' 文件，创建字体映射对象
        fontmap = PsfontsMap(find_tex_file('pdftex.map'))
        
        # 遍历 Dvi 对象中的每一页
        for page in dvi:
            # 打印页面信息，包括宽度、高度和下降值
            print(f"=== new page === "
                  f"(w: {page.width}, h: {page.height}, d: {page.descent})")
            
            # 使用 'itertools.groupby' 将页面文本按字体分组
            for font, group in itertools.groupby(
                    page.text, lambda text: text.font):
                # 打印字体信息，包括名称和缩放比例
                print(f"font: {font.texname.decode('latin-1')!r}\t"
                      f"scale: {font._scale / 2 ** 20}")
                print("x", "y", "glyph", "chr", "w", "(glyphs)", sep="\t")
                
                # 遍历文本组中的每个文本对象
                for text in group:
                    # 打印文本对象的位置、字形、字符（可打印字符或 '.'）、宽度
                    print(text.x, text.y, text.glyph,
                          chr(text.glyph) if chr(text.glyph).isprintable()
                          else ".",
                          text.width, sep="\t")
            
            # 如果页面有盒子对象，则打印盒子对象的位置和尺寸
            if page.boxes:
                print("x", "y", "h", "w", "", "(boxes)", sep="\t")
                for box in page.boxes:
                    print(box.x, box.y, box.height, box.width, sep="\t")
```