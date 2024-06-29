# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_ps.py`

```py
# 导入必要的模块和库
import codecs  # 用于编码解码字符串
import datetime  # 处理日期和时间的模块
from enum import Enum  # 枚举类型的支持
import functools  # 函数工具，提供装饰器和其他高阶函数
from io import StringIO  # 用于在内存中操作文本数据的 I/O 工具
import itertools  # 提供操作迭代器的函数
import logging  # 记录日志消息
import math  # 数学函数
import os  # 提供与操作系统交互的功能
import pathlib  # 操作路径的工具
import shutil  # 文件操作工具
from tempfile import TemporaryDirectory  # 创建临时目录的工具
import time  # 时间相关的功能

import numpy as np  # 数值计算库

import matplotlib as mpl  # 绘图库 matplotlib 的主命名空间
from matplotlib import _api, cbook, _path, _text_helpers  # matplotlib 内部工具
from matplotlib._afm import AFM  # Adobe 字体度量
from matplotlib.backend_bases import (  # matplotlib 后端基类
    _Backend, FigureCanvasBase, FigureManagerBase, RendererBase)
from matplotlib.cbook import is_writable_file_like, file_requires_unicode  # matplotlib 内部工具函数
from matplotlib.font_manager import get_font  # 获取字体管理器
from matplotlib.ft2font import LOAD_NO_SCALE, FT2Font  # FreeType 字体渲染器
from matplotlib._ttconv import convert_ttf_to_ps  # 转换 TrueType 字体为 PostScript
from matplotlib._mathtext_data import uni2type1  # 数学文本数据映射
from matplotlib.path import Path  # 定义路径
from matplotlib.texmanager import TexManager  # LaTeX 文本管理器
from matplotlib.transforms import Affine2D  # 二维仿射变换
from matplotlib.backends.backend_mixed import MixedModeRenderer  # 混合模式渲染器
from . import _backend_pdf_ps  # 导入本地的 PDF/PS 后端

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器
debugPS = False  # 是否启用 PostScript 调试模式，默认关闭

@api.caching_module_getattr
class __getattr__:  # 定义一个获取属性的类
    psDefs = _api.deprecated("3.8", obj_type="")(property(lambda self: _psDefs))  # 获取过时属性 psDefs

# 支持的页面尺寸定义
papersize = {'letter': (8.5, 11),
             'legal': (8.5, 14),
             'ledger': (11, 17),
             'a0': (33.11, 46.81),
             'a1': (23.39, 33.11),
             'a2': (16.54, 23.39),
             'a3': (11.69, 16.54),
             'a4': (8.27, 11.69),
             'a5': (5.83, 8.27),
             'a6': (4.13, 5.83),
             'a7': (2.91, 4.13),
             'a8': (2.05, 2.91),
             'a9': (1.46, 2.05),
             'a10': (1.02, 1.46),
             'b0': (40.55, 57.32),
             'b1': (28.66, 40.55),
             'b2': (20.27, 28.66),
             'b3': (14.33, 20.27),
             'b4': (10.11, 14.33),
             'b5': (7.16, 10.11),
             'b6': (5.04, 7.16),
             'b7': (3.58, 5.04),
             'b8': (2.51, 3.58),
             'b9': (1.76, 2.51),
             'b10': (1.26, 1.76)}

def _get_papertype(w, h):
    """
    根据给定的宽度和高度获取纸张类型。

    Args:
        w (float): 宽度
        h (float): 高度

    Returns:
        str: 纸张类型字符串
    """
    for key, (pw, ph) in sorted(papersize.items(), reverse=True):
        if key.startswith('l'):
            continue
        if w < pw and h < ph:
            return key
    return 'a0'

def _nums_to_str(*args, sep=" "):
    """
    将一组数字转换为字符串形式，用指定分隔符连接。

    Args:
        args (float): 可变参数列表
        sep (str): 分隔符，默认为空格

    Returns:
        str: 连接后的字符串
    """
    return sep.join(f"{arg:1.3f}".rstrip("0").rstrip(".") for arg in args)

def _move_path_to_path_or_stream(src, dst):
    """
    将文件 src 的内容移动到路径或文件流 dst。

    如果 dst 是路径，则不复制 src 的元数据。

    Args:
        src (str): 源文件路径
        dst (str or file-like): 目标路径或文件流
    """
    if is_writable_file_like(dst):
        fh = (open(src, encoding='latin-1')
              if file_requires_unicode(dst)
              else open(src, 'rb'))
        with fh:
            shutil.copyfileobj(fh, dst)
    else:
        shutil.move(src, dst, copy_function=shutil.copyfile)

def _font_to_ps_type3(font_path, chars):
    """
    将指定字体路径的 TrueType 字体转换为 Type 3 PostScript 字体。

    Args:
        font_path (str): 字体文件路径
        chars (str): 字符集

    Returns:
        None
    """
    Subset *chars* from the font at *font_path* into a Type 3 font.

    Parameters
    ----------
    font_path : path-like
        Path to the font to be subsetted.
    chars : str
        The characters to include in the subsetted font.

    Returns
    -------
    str
        The string representation of a Type 3 font, which can be included
        verbatim into a PostScript file.
    """
    # 使用给定路径获取字体对象，hinting_factor 参数设置为 1
    font = get_font(font_path, hinting_factor=1)
    # 获取每个字符在字体中的字形索引，存储在列表 glyph_ids 中
    glyph_ids = [font.get_char_index(c) for c in chars]

    # 定义 Type 3 字体的前导部分，以字符串形式返回
    preamble = """\
# 定义一个以%!PS-Adobe-3.0 Resource-Font开头的字符串，用于创建Type 3字体
def _font_to_ps_type42(font_path, chars, fh):
    """
    Subset *chars* from the font at *font_path* into a Type 42 font at *fh*.

    Parameters
    ----------
    font_path : path-like
        Path to the font to be subsetted.
    chars : str
        The characters to include in the subsetted font.
    fh : file-like
        Where to write the font.
    """
    # 根据给定的字体路径打开字体文件，将指定字符集合输出到Type 42字体文件中
    subset_str = ''.join(chr(c) for c in chars)
    # 记录调试信息，显示正在创建的字符集
    _log.debug("SUBSET %s characters: %s", font_path, subset_str)
    try:
        # 尝试获取字体的子集数据
        fontdata = _backend_pdf_ps.get_glyphs_subset(font_path, subset_str)
        # 记录调试信息，显示子集化字体的转换前后大小
        _log.debug("SUBSET %s %d -> %d", font_path, os.stat(font_path).st_size,
                   fontdata.getbuffer().nbytes)

        # 使用子集化后的字体数据创建 FreeType2 字体对象
        font = FT2Font(fontdata)
        
        # 获取字符集 chars 中每个字符的 glyph_id
        glyph_ids = [font.get_char_index(c) for c in chars]
        
        # 使用临时目录创建临时文件 tmp.ttf
        with TemporaryDirectory() as tmpdir:
            tmpfile = os.path.join(tmpdir, "tmp.ttf")

            # 将字体数据写入临时文件
            with open(tmpfile, 'wb') as tmp:
                tmp.write(fontdata.getvalue())

            # 调用 convert_ttf_to_ps 将 TrueType 字体转换为 PostScript
            # TODO: 允许 convert_ttf_to_ps 输入文件对象 (BytesIO)
            convert_ttf_to_ps(os.fsencode(tmpfile), fh, 42, glyph_ids)
    
    except RuntimeError:
        # 捕获运行时错误，并记录警告日志，指出后端不支持选定的字体
        _log.warning(
            "The PostScript backend does not currently "
            "support the selected font.")
        # 重新抛出 RuntimeError 异常
        raise


这段代码的注释按照要求对每行代码进行了解释，包括尝试获取字体子集数据、记录调试信息、创建 FreeType2 字体对象、写入临时文件、调用字体转换函数等。
    # 定义一个装饰器函数，用于条件下在 PS 文件中添加方法名的注释
    def _log_if_debug_on(meth):
        """
        Wrap `RendererPS` method *meth* to emit a PS comment with the method name,
        if the global flag `debugPS` is set.
        """
        @functools.wraps(meth)
        def wrapper(self, *args, **kwargs):
            # 如果 debugPS 全局标志被设置，向 PS 文件中写入方法名的注释
            if debugPS:
                self._pswriter.write(f"% {meth.__name__}\n")
            # 调用原始方法并返回结果
            return meth(self, *args, **kwargs)

        return wrapper


class RendererPS(_backend_pdf_ps.RendererPDFPSBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles.
    """

    _afm_font_dir = cbook._get_data_path("fonts/afm")
    _use_afm_rc_name = "ps.useafm"

    def __init__(self, width, height, pswriter, imagedpi=72):
        # Although postscript itself is dpi independent, we need to inform the
        # image code about a requested dpi to generate high resolution images
        # and them scale them before embedding them.
        super().__init__(width, height)
        self._pswriter = pswriter
        # 如果 mpl.rcParams['text.usetex'] 为真，初始化文本计数器和 psfrag 列表
        if mpl.rcParams['text.usetex']:
            self.textcnt = 0
            self.psfrag = []
        self.imagedpi = imagedpi

        # 当前渲染器状态的初始化（None 表示未初始化）
        self.color = None
        self.linewidth = None
        self.linejoin = None
        self.linecap = None
        self.linedash = None
        self.fontname = None
        self.fontsize = None
        self._hatches = {}
        self.image_magnification = imagedpi / 72
        self._clip_paths = {}
        self._path_collection_id = 0

        # 字符追踪器的初始化
        self._character_tracker = _backend_pdf_ps.CharacterTracker()
        # 生成警告的缓存函数，以便警告只会输出一次
        self._logwarn_once = functools.cache(_log.warning)

    def _is_transparent(self, rgb_or_rgba):
        # 检查颜色是否透明
        if rgb_or_rgba is None:
            return True  # 与 rgbFace 语义一致，视为透明
        elif len(rgb_or_rgba) == 4:
            if rgb_or_rgba[3] == 0:
                return True
            if rgb_or_rgba[3] != 1:
                # PostScript 后端不支持透明度，部分透明的图形会被渲染为不透明
                self._logwarn_once(
                    "The PostScript backend does not support transparency; "
                    "partially transparent artists will be rendered opaque.")
            return False
        else:  # len() == 3.
            return False

    def set_color(self, r, g, b, store=True):
        # 设置颜色
        if (r, g, b) != self.color:
            self._pswriter.write(f"{_nums_to_str(r)} setgray\n"
                                 if r == g == b else
                                 f"{_nums_to_str(r, g, b)} setrgbcolor\n")
            if store:
                self.color = (r, g, b)

    def set_linewidth(self, linewidth, store=True):
        # 设置线宽
        linewidth = float(linewidth)
        if linewidth != self.linewidth:
            self._pswriter.write(f"{_nums_to_str(linewidth)} setlinewidth\n")
            if store:
                self.linewidth = linewidth

    @staticmethod
    def _linejoin_cmd(linejoin):
        # 支持直接传递整数值是为了向后兼容。
        # 根据给定的 linejoin 值选择对应的线连接方式，转换为整数值。
        linejoin = {'miter': 0, 'round': 1, 'bevel': 2, 0: 0, 1: 1, 2: 2}[
            linejoin]
        return f"{linejoin:d} setlinejoin\n"

    def set_linejoin(self, linejoin, store=True):
        # 如果传入的 linejoin 值与当前设置不同，设置新的 linejoin。
        if linejoin != self.linejoin:
            # 写入设定线连接方式的命令到 PostScript 写入器。
            self._pswriter.write(self._linejoin_cmd(linejoin))
            if store:
                # 如果 store 为真，则更新当前对象的 linejoin 属性。
                self.linejoin = linejoin

    @staticmethod
    def _linecap_cmd(linecap):
        # 支持直接传递整数值是为了向后兼容。
        # 根据给定的 linecap 值选择对应的线端方式，转换为整数值。
        linecap = {'butt': 0, 'round': 1, 'projecting': 2, 0: 0, 1: 1, 2: 2}[
            linecap]
        return f"{linecap:d} setlinecap\n"

    def set_linecap(self, linecap, store=True):
        # 如果传入的 linecap 值与当前设置不同，设置新的 linecap。
        if linecap != self.linecap:
            # 写入设定线端方式的命令到 PostScript 写入器。
            self._pswriter.write(self._linecap_cmd(linecap))
            if store:
                # 如果 store 为真，则更新当前对象的 linecap 属性。
                self.linecap = linecap

    def set_linedash(self, offset, seq, store=True):
        # 如果当前设置的线型不为 None，则检查是否与新设置的相同，若相同则直接返回。
        if self.linedash is not None:
            oldo, oldseq = self.linedash
            if np.array_equal(seq, oldseq) and oldo == offset:
                return

        # 根据给定的偏移量和线型序列设置线型。
        self._pswriter.write(f"[{_nums_to_str(*seq)}] {_nums_to_str(offset)} setdash\n"
                             if seq is not None and len(seq) else
                             "[] 0 setdash\n")
        if store:
            # 如果 store 为真，则更新当前对象的 linedash 属性。
            self.linedash = (offset, seq)

    def set_font(self, fontname, fontsize, store=True):
        # 如果传入的字体名和字号与当前设置不同，则设置新的字体和字号。
        if (fontname, fontsize) != (self.fontname, self.fontsize):
            # 写入设定字体和字号的命令到 PostScript 写入器。
            self._pswriter.write(f"/{fontname} {fontsize:1.3f} selectfont\n")
            if store:
                # 如果 store 为真，则更新当前对象的 fontname 和 fontsize 属性。
                self.fontname = fontname
                self.fontsize = fontsize

    def create_hatch(self, hatch):
        # 定义默认的方形边长和图案名称，如果已存在对应图案，则直接返回。
        sidelen = 72
        if hatch in self._hatches:
            return self._hatches[hatch]
        name = 'H%d' % len(self._hatches)
        linewidth = mpl.rcParams['hatch.linewidth']
        pageheight = self.height * 72
        # 写入定义图案的 PostScript 命令到 PostScript 写入器。
        self._pswriter.write(f"""\
  << /PatternType 1
     /PaintType 2
     /TilingType 2
     /BBox[0 0 {sidelen:d} {sidelen:d}]
     /XStep {sidelen:d}
     /YStep {sidelen:d}

     /PaintProc {{
        pop
        {linewidth:g} setlinewidth
    def get_image_magnification(self):
        """
        获取传递给 draw_image 的图像放大倍数因子。
        允许后端将图像显示在与其他艺术家不同的分辨率上。
        """
        return self.image_magnification

    def _convert_path(self, path, transform, clip=False, simplify=None):
        """
        将给定路径转换为字符串表示形式，并应用变换。
        可选择进行剪裁和简化。
        """
        if clip:
            clip = (0.0, 0.0, self.width * 72.0, self.height * 72.0)
        else:
            clip = None
        return _path.convert_to_string(
            path, transform, clip, simplify, None,
            6, [b"m", b"l", b"", b"c", b"cl"], True).decode("ascii")

    def _get_clip_cmd(self, gc):
        """
        根据图形上下文获取剪切路径的命令。
        如果存在矩形剪切区域或自定义路径剪切区域，则返回相应的命令字符串。
        """
        clip = []
        rect = gc.get_clip_rectangle()
        if rect is not None:
            clip.append(f"{_nums_to_str(*rect.p0, *rect.size)} rectclip\n")
        path, trf = gc.get_clip_path()
        if path is not None:
            key = (path, id(trf))
            custom_clip_cmd = self._clip_paths.get(key)
            if custom_clip_cmd is None:
                custom_clip_cmd = "c%d" % len(self._clip_paths)
                self._pswriter.write(f"""\
/{custom_clip_cmd} {{
{self._convert_path(path, trf, simplify=False)}
clip
newpath
}} bind def
""")
                self._clip_paths[key] = custom_clip_cmd
            clip.append(f"{custom_clip_cmd}\n")
        return "".join(clip)

    @_log_if_debug_on
    def draw_image(self, gc, x, y, im, transform=None):
        """
        绘制图像到给定位置，可选地应用变换。
        """
        h, w = im.shape[:2]
        imagecmd = "false 3 colorimage"
        data = im[::-1, :, :3]  # 垂直翻转的 RGB 值。
        hexdata = data.tobytes().hex("\n", -64)  # 每行最多显示 128 个字符。

        if transform is None:
            matrix = "1 0 0 1 0 0"
            xscale = w / self.image_magnification
            yscale = h / self.image_magnification
        else:
            matrix = " ".join(map(str, transform.frozen().to_values()))
            xscale = 1.0
            yscale = 1.0

        self._pswriter.write(f"""\
gsave
{self._get_clip_cmd(gc)}
{x:g} {y:g} translate
[{matrix}] concat
{xscale:g} {yscale:g} scale
/DataString {w:d} string def
{w:d} {h:d} 8 [ {w:d} 0 0 -{h:d} 0 {h:d} ]
{{
currentfile DataString readhexstring pop
}} bind {imagecmd}
{hexdata}
grestore
""")

    @_log_if_debug_on
    def draw_path(self, gc, path, transform, rgbFace=None):
        """
        绘制路径到画布上，应用变换和可选的填充颜色。
        """
        clip = rgbFace is None and gc.get_hatch_path() is None
        simplify = path.should_simplify and clip
        ps = self._convert_path(path, transform, clip=clip, simplify=simplify)
        self._draw_ps(ps, gc, rgbFace)
    def draw_markers(
            self, gc, marker_path, marker_trans, path, trans, rgbFace=None):
        # 绘制标记点的方法

        ps_color = (
            None
            if self._is_transparent(rgbFace)  # 如果填充色是透明的，则为 None
            else f'{_nums_to_str(rgbFace[0])} setgray'  # 如果填充色是灰度，则设置灰度值
            if rgbFace[0] == rgbFace[1] == rgbFace[2]  # 如果填充色的RGB值相同
            else f'{_nums_to_str(*rgbFace[:3])} setrgbcolor')  # 否则设置RGB颜色值

        # 构建通用的标记点命令:

        # 不希望平移是全局的
        ps_cmd = ['/o {', 'gsave', 'newpath', 'translate']

        lw = gc.get_linewidth()  # 获取线宽
        alpha = (gc.get_alpha()  # 获取透明度
                 if gc.get_forced_alpha() or len(gc.get_rgb()) == 3
                 else gc.get_rgb()[3])  # 如果指定了强制透明度或RGB长度为3，则使用透明度值
        stroke = lw > 0 and alpha > 0  # 判断是否需要描边
        if stroke:
            ps_cmd.append('%.1f setlinewidth' % lw)  # 设置线宽
            ps_cmd.append(self._linejoin_cmd(gc.get_joinstyle()))  # 设置连接样式
            ps_cmd.append(self._linecap_cmd(gc.get_capstyle()))  # 设置端点样式

        ps_cmd.append(self._convert_path(marker_path, marker_trans,
                                         simplify=False))  # 转换标记路径

        if rgbFace:
            if stroke:
                ps_cmd.append('gsave')
            if ps_color:
                ps_cmd.extend([ps_color, 'fill'])  # 填充颜色
            if stroke:
                ps_cmd.append('grestore')

        if stroke:
            ps_cmd.append('stroke')  # 描边

        ps_cmd.extend(['grestore', '} bind def'])  # 恢复绘图状态并定义命令

        for vertices, code in path.iter_segments(
                trans,
                clip=(0, 0, self.width*72, self.height*72),
                simplify=False):
            if len(vertices):
                x, y = vertices[-2:]
                ps_cmd.append(f"{x:g} {y:g} o")  # 在路径上绘制标记点的坐标

        ps = '\n'.join(ps_cmd)  # 将所有命令连接成 PS 格式字符串
        self._draw_ps(ps, gc, rgbFace, fill=False, stroke=False)  # 使用绘图上下文绘制 PS 字符串内容

    @_log_if_debug_on
        def draw_path_collection(self, gc, master_transform, paths, all_transforms,
                                 offsets, offset_trans, facecolors, edgecolors,
                                 linewidths, linestyles, antialiaseds, urls,
                                 offset_position):
            # 是否值得进行优化？粗略计算：
            # 在行内发出路径的成本是
            #     (len_path + 2) * uses_per_path
            # 定义和使用的成本是
            #     (len_path + 3) + 3 * uses_per_path
            len_path = len(paths[0].vertices) if len(paths) > 0 else 0
            # 计算每个路径的使用次数
            uses_per_path = self._iter_collection_uses_per_path(
                paths, all_transforms, offsets, facecolors, edgecolors)
            # 是否应该进行优化的决策条件
            should_do_optimization = \
                len_path + 3 * uses_per_path + 3 < (len_path + 2) * uses_per_path
            if not should_do_optimization:
                # 如果不应进行优化，则调用基类的绘制路径集合方法并返回结果
                return RendererBase.draw_path_collection(
                    self, gc, master_transform, paths, all_transforms,
                    offsets, offset_trans, facecolors, edgecolors,
                    linewidths, linestyles, antialiaseds, urls,
                    offset_position)

            path_codes = []
            # 遍历处理每个原始路径及其变换
            for i, (path, transform) in enumerate(self._iter_collection_raw_paths(
                    master_transform, paths, all_transforms)):
                # 创建路径的唯一标识名称
                name = 'p%d_%d' % (self._path_collection_id, i)
                # 将路径转换为字节表示
                path_bytes = self._convert_path(path, transform, simplify=False)
                # 将路径代码添加到列表中
                self._pswriter.write(f"""
    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        # docstring inherited
        # 检查图形上下文的RGB颜色是否完全透明，如果是则特殊处理并返回
        if self._is_transparent(gc.get_rgb()):
            return  # Special handling for fully transparent.

        # 如果对象没有"psfrag"属性，则警告用户关于usetex=True的设置限制，并绘制文本
        if not hasattr(self, "psfrag"):
            self._logwarn_once(
                "The PS backend determines usetex status solely based on "
                "rcParams['text.usetex'] and does not support having "
                "usetex=True only for some elements; this element will thus "
                "be rendered as if usetex=False.")
            self.draw_text(gc, x, y, s, prop, angle, False, mtext)
            return

        # 获取文本的宽度、高度和基线
        w, h, bl = self.get_text_width_height_descent(s, prop, ismath="TeX")
        # 获取字体大小
        fontsize = prop.get_size_in_points()
        # 构造唯一的文本标识符
        thetext = 'psmarker%d' % self.textcnt
        # 获取图形上下文的RGB颜色并转换成字符串形式
        color = _nums_to_str(*gc.get_rgb()[:3], sep=',')
        # 根据字体系列设置字体命令
        fontcmd = {'sans-serif': r'{\sffamily %s}',
                   'monospace': r'{\ttfamily %s}'}.get(
                       mpl.rcParams['font.family'][0], r'{\rmfamily %s}')
        # 根据颜色和文本内容生成TeX格式的文本
        s = fontcmd % s
        tex = r'\color[rgb]{%s} %s' % (color, s)

        # 计算文本在旋转角度后的位置
        rangle = np.radians(angle + 90)
        pos = _nums_to_str(x - bl * np.cos(rangle), y - bl * np.sin(rangle))
        # 将psfrag标签和TeX格式的文本添加到psfrag列表中
        self.psfrag.append(
            r'\psfrag{%s}[bl][bl][1][%f]{\fontsize{%f}{%f}%s}' % (
                thetext, angle, fontsize, fontsize*1.25, tex))

        # 将文本的PS绘制命令写入PS输出流
        self._pswriter.write(f"""\
gsave
{pos} moveto
({thetext})
show
grestore
""")
        self.textcnt += 1
    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # 绘制文本到指定位置，支持不同的文本处理方式和字体类型

        if self._is_transparent(gc.get_rgb()):
            return  # 如果颜色是全透明的，特殊处理并直接返回

        if ismath == 'TeX':
            return self.draw_tex(gc, x, y, s, prop, angle)  # 如果是TeX格式的数学文本，调用相应的绘制函数

        if ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)  # 如果是数学模式文本，调用数学文本绘制函数

        stream = []  # 用于存储绘制文本的信息，每个元素是(ps_name, x, char_name)的元组列表

        if mpl.rcParams['ps.useafm']:
            font = self._get_font_afm(prop)  # 获取Adobe字体度量文件（AFM）格式的字体对象
            ps_name = (font.postscript_name.encode("ascii", "replace")
                        .decode("ascii"))  # 获取字体的PostScript名称，并处理非ASCII字符
            scale = 0.001 * prop.get_size_in_points()  # 计算字体的缩放比例
            thisx = 0
            last_name = None  # 上一个字符的字体名称，用于计算字符之间的间距
            for c in s:
                name = uni2type1.get(ord(c), f"uni{ord(c):04X}")  # 获取Unicode字符对应的字体名称或者默认名称
                try:
                    width = font.get_width_from_char_name(name)  # 获取字符的宽度
                except KeyError:
                    name = 'question'  # 如果字符不存在，使用默认的'question'字符
                    width = font.get_width_char('?')  # 获取'?'字符的宽度
                kern = font.get_kern_dist_from_name(last_name, name)  # 获取字符之间的字距
                last_name = name
                thisx += kern * scale  # 根据字距调整字符的x位置
                stream.append((ps_name, thisx, name))  # 将字符的绘制信息添加到stream列表中
                thisx += width * scale  # 根据字符宽度调整下一个字符的位置

        else:
            font = self._get_font_ttf(prop)  # 获取TrueType字体对象
            self._character_tracker.track(font, s)  # 跟踪使用的字符
            for item in _text_helpers.layout(s, font):
                ps_name = (item.ft_object.postscript_name
                           .encode("ascii", "replace").decode("ascii"))  # 获取TrueType字体的PostScript名称，并处理非ASCII字符
                glyph_name = item.ft_object.get_glyph_name(item.glyph_idx)  # 获取字符的字形名称
                stream.append((ps_name, item.x, glyph_name))  # 将字符的绘制信息添加到stream列表中
        self.set_color(*gc.get_rgb())  # 设置绘制文本的颜色

        for ps_name, group in itertools. \
                groupby(stream, lambda entry: entry[0]):
            self.set_font(ps_name, prop.get_size_in_points(), False)  # 设置当前字体
            thetext = "\n".join(f"{x:g} 0 m /{name:s} glyphshow"
                                for _, x, name in group)  # 构建每个字形的PostScript指令
            self._pswriter.write(f"""\  # 将文本内容转换为PostScript指令并写入到PostScript文件中
    # 保存当前图形状态
    gsave
    # 获取剪切命令并写入PostScript文件
    {self._get_clip_cmd(gc)}
    # 将原点移动到指定的坐标位置(x, y)
    {x:g} {y:g} translate
    # 围绕原点按照指定角度旋转坐标系
    {angle:g} rotate
    # 在PostScript文件中写入文本内容
    {thetext}
    # 恢复之前保存的图形状态
    grestore


    # 如果调试开启，则记录函数调用日志
    @_log_if_debug_on
    # 使用Matplotlib的mathtext解析器绘制数学文本
    def draw_mathtext(self, gc, x, y, s, prop, angle):
        """Draw the math text using matplotlib.mathtext."""
        # 解析数学文本并返回其宽度、高度、下降值、字形和矩形
        width, height, descent, glyphs, rects = \
            self._text2path.mathtext_parser.parse(s, 72, prop)
        # 设置绘图颜色为当前图形状态的RGB颜色
        self.set_color(*gc.get_rgb())
        # 在PostScript文件中写入平移和旋转操作的命令
        self._pswriter.write(
            f"gsave\n"
            f"{x:g} {y:g} translate\n"
            f"{angle:g} rotate\n")
        lastfont = None
        for font, fontsize, num, ox, oy in glyphs:
            # 跟踪使用的字形
            self._character_tracker.track_glyph(font, num)
            if (font.postscript_name, fontsize) != lastfont:
                # 在PostScript文件中写入选择字体的命令
                lastfont = font.postscript_name, fontsize
                self._pswriter.write(
                    f"/{font.postscript_name} {fontsize} selectfont\n")
            glyph_name = (
                font.get_name_char(chr(num)) if isinstance(font, AFM) else
                font.get_glyph_name(font.get_char_index(num)))
            # 在PostScript文件中写入绘制字形的命令
            self._pswriter.write(
                f"{ox:g} {oy:g} moveto\n"
                f"/{glyph_name} glyphshow\n")
        for ox, oy, w, h in rects:
            # 在PostScript文件中写入绘制矩形的命令
            self._pswriter.write(f"{ox} {oy} {w} {h} rectfill\n")
        # 恢复之前保存的图形状态
        self._pswriter.write("grestore\n")


    # 如果调试开启，则记录函数调用日志
    @_log_if_debug_on
    # 绘制Gouraud三角形
    def draw_gouraud_triangles(self, gc, points, colors, trans):
        assert len(points) == len(colors)
        if len(points) == 0:
            return
        assert points.ndim == 3
        assert points.shape[1] == 3
        assert points.shape[2] == 2
        assert colors.ndim == 3
        assert colors.shape[1] == 3
        assert colors.shape[2] == 4

        shape = points.shape
        # 将三维点坐标展平为二维数组
        flat_points = points.reshape((shape[0] * shape[1], 2))
        # 使用变换函数对坐标进行转换
        flat_points = trans.transform(flat_points)
        # 将颜色数组展平为二维数组
        flat_colors = colors.reshape((shape[0] * shape[1], 4))
        # 计算点的最小和最大值，并添加安全边界
        points_min = np.min(flat_points, axis=0) - (1 << 12)
        points_max = np.max(flat_points, axis=0) + (1 << 12)
        # 计算缩放因子
        factor = np.ceil((2 ** 32 - 1) / (points_max - points_min))

        xmin, ymin = points_min
        xmax, ymax = points_max

        # 创建包含数据的结构化数组
        data = np.empty(
            shape[0] * shape[1],
            dtype=[('flags', 'u1'), ('points', '2>u4'), ('colors', '3u1')])
        data['flags'] = 0
        # 将点坐标映射到指定范围内，并乘以缩放因子
        data['points'] = (flat_points - points_min) * factor
        # 将颜色值乘以255，并存储到数据结构中
        data['colors'] = flat_colors[:, :3] * 255.0
        # 将数据转换为十六进制字符串，并按每行64个字符进行换行
        hexdata = data.tobytes().hex("\n", -64)  # Linewrap to 128 chars.

        # 在PostScript文件中写入渐变三角形填充的命令
        self._pswriter.write(f"""\
gsave
<< /ShadingType 4
   /ColorSpace [/DeviceRGB]
   /BitsPerCoordinate 32
   /BitsPerComponent 8
   /BitsPerFlag 8
   /AntiAlias true
   /Decode [ {xmin:g} {xmax:g} {ymin:g} {ymax:g} 0 1 0 1 0 1 ]
   /DataSource <
{hexdata}
>
>>
shfill
grestore
""")
    def _draw_ps(self, ps, gc, rgbFace, *, fill=True, stroke=True):
        """
        Emit the PostScript snippet *ps* with all the attributes from *gc*
        applied.  *ps* must consist of PostScript commands to construct a path.

        The *fill* and/or *stroke* kwargs can be set to False if the *ps*
        string already includes filling and/or stroking, in which case
        `_draw_ps` is just supplying properties and clipping.
        """
        # 定义一个局部函数 write，用于向 PostScript 文件中写入内容
        write = self._pswriter.write
        # 检查是否可能需要描边
        mightstroke = (gc.get_linewidth() > 0
                       and not self._is_transparent(gc.get_rgb()))
        # 如果不可能描边，则强制设置描边为 False
        if not mightstroke:
            stroke = False
        # 如果填充颜色是透明的，则强制设置填充为 False
        if self._is_transparent(rgbFace):
            fill = False
        # 获取图案填充样式
        hatch = gc.get_hatch()

        # 如果可能需要描边，则设置当前图形上下文的线宽、连接方式、端点样式、虚线样式
        if mightstroke:
            self.set_linewidth(gc.get_linewidth())
            self.set_linejoin(gc.get_joinstyle())
            self.set_linecap(gc.get_capstyle())
            self.set_linedash(*gc.get_dashes())
        # 如果可能需要描边或者存在图案填充，则设置当前颜色为图形上下文的 RGB 颜色值
        if mightstroke or hatch:
            self.set_color(*gc.get_rgb()[:3])
        # 保存当前图形状态
        write('gsave\n')

        # 写入剪切命令到 PostScript 文件中
        write(self._get_clip_cmd(gc))

        # 写入路径构造命令 *ps* 到 PostScript 文件中，并去除首尾空白字符
        write(ps.strip())
        write("\n")

        # 如果需要填充，则根据情况保存图形状态并填充 RGB 颜色值
        if fill:
            # 如果需要描边或者存在图案填充，则保存当前图形状态
            if stroke or hatch:
                write("gsave\n")
            # 设置当前颜色为填充颜色，不存储当前颜色到堆栈
            self.set_color(*rgbFace[:3], store=False)
            write("fill\n")
            # 如果需要描边或者存在图案填充，则恢复之前保存的图形状态
            if stroke or hatch:
                write("grestore\n")

        # 如果存在图案填充，则创建图案并填充
        if hatch:
            hatch_name = self.create_hatch(hatch)
            write("gsave\n")
            # 写入图案填充的颜色值和图案名称
            write(_nums_to_str(*gc.get_hatch_color()[:3]))
            write(f" {hatch_name} setpattern fill grestore\n")

        # 如果需要描边，则进行描边操作
        if stroke:
            write("stroke\n")

        # 恢复之前保存的图形状态
        write("grestore\n")
class _Orientation(Enum):
    portrait, landscape = range(2)

    def swap_if_landscape(self, shape):
        return shape[::-1] if self.name == "landscape" else shape



class FigureCanvasPS(FigureCanvasBase):
    fixed_dpi = 72
    filetypes = {'ps': 'Postscript',
                 'eps': 'Encapsulated Postscript'}

    def get_default_filetype(self):
        return 'ps'

    def _print_ps(
            self, fmt, outfile, *,
            metadata=None, papertype=None, orientation='portrait',
            bbox_inches_restore=None, **kwargs):

        dpi = self.figure.dpi
        self.figure.dpi = 72  # Override the dpi kwarg

        dsc_comments = {}
        
        # 如果输出文件路径是字符串或路径对象，获取文件名并将其作为标题加入到 DSC 注释中
        if isinstance(outfile, (str, os.PathLike)):
            filename = pathlib.Path(outfile).name
            dsc_comments["Title"] = \
                filename.encode("ascii", "replace").decode("ascii")
        
        # 获取 Creator 信息，如果未提供则使用默认信息
        dsc_comments["Creator"] = (metadata or {}).get(
            "Creator",
            f"Matplotlib v{mpl.__version__}, https://matplotlib.org/")
        
        # 根据 SOURCE_DATE_EPOCH 环境变量设置 CreationDate，或使用当前时间
        source_date_epoch = os.getenv("SOURCE_DATE_EPOCH")
        dsc_comments["CreationDate"] = (
            datetime.datetime.fromtimestamp(
                int(source_date_epoch),
                datetime.timezone.utc).strftime("%a %b %d %H:%M:%S %Y")
            if source_date_epoch
            else time.ctime())
        
        # 将 DSC 注释格式化为符合规范的字符串形式
        dsc_comments = "\n".join(
            f"%%{k}: {v}" for k, v in dsc_comments.items())

        # 如果未指定 papertype，则使用默认的 ps.papersize 参数
        if papertype is None:
            papertype = mpl.rcParams['ps.papersize']
        papertype = papertype.lower()
        
        # 检查 papertype 是否在允许的列表中
        _api.check_in_list(['figure', 'auto', *papersize], papertype=papertype)

        # 根据指定的 orientation 字符串获取对应的 Orientation 枚举值
        orientation = _api.check_getitem(
            _Orientation, orientation=orientation.lower())

        # 根据 text.usetex 配置选择打印函数，使用 Tex 渲染或普通打印方式
        printer = (self._print_figure_tex
                   if mpl.rcParams['text.usetex'] else
                   self._print_figure)
        
        # 调用打印函数，打印图形到输出文件，传递各种参数
        printer(fmt, outfile, dpi=dpi, dsc_comments=dsc_comments,
                orientation=orientation, papertype=papertype,
                bbox_inches_restore=bbox_inches_restore, **kwargs)
        # 按照指定的格式打印图形到TeX文件
        def _print_figure_tex(
                self, fmt, outfile, *,
                dpi, dsc_comments, orientation, papertype,
                bbox_inches_restore=None):
            """
            如果 :rc:`text.usetex` 为 True，则创建临时的tex/eps文件对，
            以便TeX通过PSFrags包管理文本布局。这些文件会被处理以生成最终的ps或eps文件。

            其余行为与 `._print_figure` 相同。
            """
            # 检查是否为EPS格式
            is_eps = fmt == 'eps'

            # 获取图形的尺寸（英寸单位）
            width, height = self.figure.get_size_inches()
            xo = 0
            yo = 0

            # 计算图形的边界框（以英寸为单位）
            llx = xo
            lly = yo
            urx = llx + self.figure.bbox.width
            ury = lly + self.figure.bbox.height
            bbox = (llx, lly, urx, ury)

            # 初始化一个字符串IO对象来存储PS内容
            self._pswriter = StringIO()

            # 创建一个PS渲染器
            ps_renderer = RendererPS(width, height, self._pswriter, imagedpi=dpi)

            # 创建混合模式渲染器
            renderer = MixedModeRenderer(self.figure,
                                         width, height, dpi, ps_renderer,
                                         bbox_inches_restore=bbox_inches_restore)

            # 将图形绘制到渲染器上
            self.figure.draw(renderer)

            # 将渲染后的结果写入临时文件，完成后移动到输出文件
            with TemporaryDirectory() as tmpdir:
                tmppath = pathlib.Path(tmpdir, "tmp.ps")
                tmppath.write_text(
                    f"""\
%!PS-Adobe-3.0 EPSF-3.0
%%LanguageLevel: 3
{dsc_comments}
{_get_bbox_header(bbox)}
%%EndComments
%%BeginProlog
/mpldict {len(_psDefs)} dict def
mpldict begin
{"".join(_psDefs)}
end
%%EndProlog
mpldict begin
{_nums_to_str(xo, yo)} translate
0 0 {_nums_to_str(width*72, height*72)} rectclip
{self._pswriter.getvalue()}
end
showpage
""",
                encoding="latin-1")

            if orientation is _Orientation.landscape:  # 现在准备旋转
                width, height = height, width
                bbox = (lly, llx, ury, urx)

            # 如果是 EPS 或者 papertype 是 'figure'，将纸张大小设置为图形大小
            # 生成的 PS 文件会有正确的边界框，因此不需要调用 'pstoeps'
            if is_eps or papertype == 'figure':
                paper_width, paper_height = orientation.swap_if_landscape(
                    self.figure.get_size_inches())
            else:
                # 如果 papertype 是 'auto'，确定纸张类型
                if papertype == 'auto':
                    papertype = _get_papertype(width, height)
                # 根据 papertype 设置纸张宽度和高度
                paper_width, paper_height = papersize[papertype]

            # 转换 PSFrag 标记到文本，生成包含 PSFrag 标记的 LaTeX 文档
            # LaTeX/dvips 生成包含实际文本的 postscript 文件
            psfrag_rotated = _convert_psfrags(
                tmppath, ps_renderer.psfrag, paper_width, paper_height,
                orientation.name)

            # 根据配置选择 distiller，或者使用 LaTeX 时进行处理
            if (mpl.rcParams['ps.usedistiller'] == 'ghostscript'
                    or mpl.rcParams['text.usetex']):
                _try_distill(gs_distill,
                             tmppath, is_eps, ptype=papertype, bbox=bbox,
                             rotated=psfrag_rotated)
            elif mpl.rcParams['ps.usedistiller'] == 'xpdf':
                _try_distill(xpdf_distill,
                             tmppath, is_eps, ptype=papertype, bbox=bbox,
                             rotated=psfrag_rotated)

            # 移动生成的路径到输出文件或流中
            _move_path_to_path_or_stream(tmppath, outfile)

    # 部分函数定义，用于生成 PS 或 EPS 输出
    print_ps = functools.partialmethod(_print_ps, "ps")
    print_eps = functools.partialmethod(_print_ps, "eps")

    def draw(self):
        # 绘制图形，不进行渲染
        self.figure.draw_without_rendering()
        return super().draw()


def _convert_psfrags(tmppath, psfrags, paper_width, paper_height, orientation):
    """
    当使用带有 postscript 的 LaTeX 后端时，我们将 PSFrag 标记写入临时的 postscript 文件，
    每个标记标记一个位置，供 LaTeX 渲染文本。convert_psfrags 生成一个包含命令的 LaTeX 文档，
    用于将这些标记转换为文本。LaTeX/dvips 生成包含实际文本的 postscript 文件。
    """
    with mpl.rc_context({
            "text.latex.preamble":
            mpl.rcParams["text.latex.preamble"] +
            mpl.texmanager._usepackage_if_not_loaded("color") +
            mpl.texmanager._usepackage_if_not_loaded("graphicx") +
            mpl.texmanager._usepackage_if_not_loaded("psfrag") +
            r"\geometry{papersize={%(width)sin,%(height)sin},margin=0in}"
            % {"width": paper_width, "height": paper_height}
    }):
        dvifile = TexManager().make_dvi(
            "\n"
            r"\begin{figure}""\n"
            r"  \centering\leavevmode""\n"
            r"  %(psfrags)s""\n"
            r"  \includegraphics*[angle=%(angle)s]{%(epsfile)s}""\n"
            r"\end{figure}"
            % {
                "psfrags": "\n".join(psfrags),  # 将 psfrags 列表中的内容用换行连接起来
                "angle": 90 if orientation == 'landscape' else 0,  # 如果 orientation 是 'landscape' 则角度为 90 度，否则为 0 度
                "epsfile": tmppath.resolve().as_posix(),  # 获取 tmppath 的绝对路径，并转换为字符串形式
            },
            fontsize=10)  # 使用默认字体大小 10

    with TemporaryDirectory() as tmpdir:  # 创建一个临时目录 tmpdir
        psfile = os.path.join(tmpdir, "tmp.ps")  # 在 tmpdir 中创建一个临时 ps 文件路径
        cbook._check_and_log_subprocess(
            ['dvips', '-q', '-R0', '-o', psfile, dvifile], _log)  # 调用 dvips 命令将 dvifile 转换为 ps 文件，并输出到 psfile，同时记录日志到 _log
        shutil.move(psfile, tmppath)  # 将生成的 ps 文件移动到 tmppath

    # 检查 dvips 是否创建了横向纸张的 ps 文件。不知何故，以上 latex+dvips 会使得某些特定尺寸的图形（如 8.3 英寸、5.8 英寸，即 a5 纸）以横向模式生成 ps 文件，并且最终输出的边界框会混乱。我们检查生成的 ps 文件是否为横向，并返回这一信息。返回值将在 pstoeps 步骤中用于恢复正确的边界框。2010-06-05 JJL
    with open(tmppath) as fh:  # 打开 tmppath 文件作为 fh
        psfrag_rotated = "Landscape" in fh.read(1000)  # 从文件中读取前 1000 个字符，检查是否包含 "Landscape" 字符串
    return psfrag_rotated  # 返回 psfrag_rotated 变量，表示是否生成了横向 ps 文件
def _try_distill(func, tmppath, *args, **kwargs):
    try:
        # 尝试调用给定的函数 func，传入 tmppath 和其他参数
        func(str(tmppath), *args, **kwargs)
    except mpl.ExecutableNotFoundError as exc:
        # 如果捕获到 ExecutableNotFoundError 异常，则记录警告信息并跳过蒸馏步骤
        _log.warning("%s.  Distillation step skipped.", exc)


def gs_distill(tmpfile, eps=False, ptype='letter', bbox=None, rotated=False):
    """
    Use ghostscript's pswrite or epswrite device to distill a file.
    This yields smaller files without illegal encapsulated postscript
    operators. The output is low-level, converting text to outlines.
    """

    if eps:
        # 如果 eps 为 True，则设置输出选项为 "-dEPSCrop"
        paper_option = ["-dEPSCrop"]
    elif ptype == "figure":
        # 如果 ptype 为 "figure"，根据 bbox 设置输出选项
        # bbox 的左下角为 (0, 0)，右上角对应纸张尺寸
        paper_option = [f"-dDEVICEWIDTHPOINTS={bbox[2]}",
                        f"-dDEVICEHEIGHTPOINTS={bbox[3]}"]
    else:
        # 否则根据指定的纸张类型设置输出选项
        paper_option = [f"-sPAPERSIZE={ptype}"]

    # 生成临时的 .ps 文件名
    psfile = tmpfile + '.ps'
    # 获取 ps.distiller.res 的 dpi 设置
    dpi = mpl.rcParams['ps.distiller.res']

    # 调用 Ghostscript 进行文件转换，生成 ps2write 格式的输出
    cbook._check_and_log_subprocess(
        [mpl._get_executable_info("gs").executable,
         "-dBATCH", "-dNOPAUSE", "-r%d" % dpi, "-sDEVICE=ps2write",
         *paper_option, f"-sOutputFile={psfile}", tmpfile],
        _log)

    # 删除原始临时文件
    os.remove(tmpfile)
    # 将生成的 .ps 文件移动到原始临时文件的位置
    shutil.move(psfile, tmpfile)

    # 如果是 eps 格式，尝试恢复原始的 bounding box
    if eps:
        # 对于某些 Ghostscript 的版本，上述步骤可能导致 ps 文件的 bbox 不正确，暂不调整 bbox
        pstoeps(tmpfile, bbox, rotated=rotated)


def xpdf_distill(tmpfile, eps=False, ptype='letter', bbox=None, rotated=False):
    """
    Use ghostscript's ps2pdf and xpdf's/poppler's pdftops to distill a file.
    This yields smaller files without illegal encapsulated postscript
    operators. This distiller is preferred, generating high-level postscript
    output that treats text as text.
    """
    mpl._get_executable_info("gs")  # 有效地检查是否存在 ps2pdf 可执行文件
    mpl._get_executable_info("pdftops")

    if eps:
        # 如果 eps 为 True，则设置输出选项为 "-dEPSCrop"
        paper_option = ["-dEPSCrop"]
    elif ptype == "figure":
        # 如果 ptype 为 "figure"，根据 bbox 设置输出选项
        # bbox 的左下角为 (0, 0)，右上角对应纸张尺寸
        paper_option = [f"-dDEVICEWIDTHPOINTS#{bbox[2]}",
                        f"-dDEVICEHEIGHTPOINTS#{bbox[3]}"]
    else:
        # 否则根据指定的纸张类型设置输出选项
        paper_option = [f"-sPAPERSIZE#{ptype}"]
    # 使用临时目录作为工作空间
    with TemporaryDirectory() as tmpdir:
        # 创建临时PDF文件路径
        tmppdf = pathlib.Path(tmpdir, "tmp.pdf")
        # 创建临时PS文件路径
        tmpps = pathlib.Path(tmpdir, "tmp.ps")
        
        # 调用 cbook 模块的 _check_and_log_subprocess 函数执行命令行操作
        # 使用 ps2pdf 将 PS 文件转换为 PDF，同时传递多个选项作为命令行参数
        # 注意选项使用 `-foo#bar` 格式而非 `-foo=bar`，以便兼容 Windows 平台
        cbook._check_and_log_subprocess(
            ["ps2pdf",
             "-dAutoFilterColorImages#false",
             "-dAutoFilterGrayImages#false",
             "-sAutoRotatePages#None",
             "-sGrayImageFilter#FlateEncode",
             "-sColorImageFilter#FlateEncode",
             *paper_option,  # 包含其他可能的纸张选项
             tmpfile, tmppdf], _log)
        
        # 使用 cbook 模块的 _check_and_log_subprocess 函数执行命令行操作
        # 使用 pdftops 将 PDF 文件转换为 PS 文件
        cbook._check_and_log_subprocess(
            ["pdftops", "-paper", "match", "-level3", tmppdf, tmpps], _log)
        
        # 将生成的 PS 文件移动/覆盖到原始文件的位置
        shutil.move(tmpps, tmpfile)
    
    # 如果 eps 为 True，则调用 pstoeps 函数将 PDF 文件转换为 EPS 格式
    if eps:
        pstoeps(tmpfile)
# 将此函数标记为废弃，适用于 Python 3.9 版本以前
@_api.deprecated("3.9")
def get_bbox_header(lbrt, rotated=False):
    """
    Return a postscript header string for the given bbox lbrt=(l, b, r, t).
    Optionally, return rotate command.
    """
    # 调用内部函数 _get_bbox_header 获取包围框的 PostScript 头部字符串，
    # 并根据 rotated 参数决定是否添加旋转命令
    return _get_bbox_header(lbrt), (_get_rotate_command(lbrt) if rotated else "")


def _get_bbox_header(lbrt):
    """Return a PostScript header string for bounding box *lbrt*=(l, b, r, t)."""
    l, b, r, t = lbrt
    # 构建包围框的 PostScript 格式字符串，包括标准精度和高精度两种格式
    return (f"%%BoundingBox: {int(l)} {int(b)} {math.ceil(r)} {math.ceil(t)}\n"
            f"%%HiResBoundingBox: {l:.6f} {b:.6f} {r:.6f} {t:.6f}")


def _get_rotate_command(lbrt):
    """Return a PostScript 90° rotation command for bounding box *lbrt*=(l, b, r, t)."""
    l, b, r, t = lbrt
    # 返回旋转命令的 PostScript 格式字符串
    return f"{l+r:.2f} {0:.2f} translate\n90 rotate"


def pstoeps(tmpfile, bbox=None, rotated=False):
    """
    Convert the postscript to encapsulated postscript.  The bbox of
    the eps file will be replaced with the given *bbox* argument. If
    None, original bbox will be used.
    """
    # 将临时 PostScript 文件转换为 EPS 文件
    epsfile = tmpfile + '.eps'
    with open(epsfile, 'wb') as epsh, open(tmpfile, 'rb') as tmph:
        write = epsh.write
        # 修改头部信息：
        for line in tmph:
            if line.startswith(b'%!PS'):
                # 替换文件头部信息为 EPS 格式
                write(b"%!PS-Adobe-3.0 EPSF-3.0\n")
                if bbox:
                    # 如果有提供新的 bbox 参数，则更新 EPS 文件的包围框信息
                    write(_get_bbox_header(bbox).encode('ascii') + b'\n')
            elif line.startswith(b'%%EndComments'):
                # 复制原始文件中的尾部信息，插入新的 Prolog 部分和页面定义
                write(line)
                write(b'%%BeginProlog\n'
                      b'save\n'
                      b'countdictstack\n'
                      b'mark\n'
                      b'newpath\n'
                      b'/showpage {} def\n'
                      b'/setpagedevice {pop} def\n'
                      b'%%EndProlog\n'
                      b'%%Page 1 1\n')
                if rotated:  # 需要旋转输出的 EPS 文件
                    write(_get_rotate_command(bbox).encode('ascii') + b'\n')
                break
            elif bbox and line.startswith((b'%%Bound', b'%%HiResBound',
                                           b'%%DocumentMedia', b'%%Pages')):
                # 忽略包围框相关的原始文件头部信息
                pass
            else:
                # 将未被修改的行直接写入 EPS 文件
                write(line)
        # 写入剩余的文件内容，并修改尾部信息
        # 在第二个循环中进行，以确保不修改嵌入 EPS 文件的头部信息
        for line in tmph:
            if line.startswith(b'%%EOF'):
                # 写入 EPS 文件的尾部信息
                write(b'cleartomark\n'
                      b'countdictstack\n'
                      b'exch sub { end } repeat\n'
                      b'restore\n'
                      b'showpage\n'
                      b'%%EOF\n')
            elif line.startswith(b'%%PageBoundingBox'):
                # 忽略页面包围框信息
                pass
            else:
                # 将未修改的行写入 EPS 文件
                write(line)

    # 删除临时文件并将生成的 EPS 文件移回原始文件位置
    os.remove(tmpfile)
    shutil.move(epsfile, tmpfile)


# 将 FigureManagerPS 设定为 FigureManagerBase 的别名
FigureManagerPS = FigureManagerBase
# PostScript 字典 mpldict。此字典实现了大部分 matplotlib 原语和一些缩写。

# 引用：
# https://www.adobe.com/content/dam/acom/en/devnet/actionscript/articles/PLRM.pdf
# http://preserve.mactech.com/articles/mactech/Vol.09/09.04/PostscriptTutorial
# http://www.math.ubc.ca/people/faculty/cass/graphics/text/www/

# 定义一个名为 _psDefs 的列表，包含多个 PostScript 定义。
_psDefs = [
    # "/_d { bind def } bind def"
    # 定义名为 _d 的函数，用于绑定定义其他函数。注意不能简写为 /d，因为当嵌入 Type3 字体时，
    # 可能会定义一个名为 "d" 的字形，使用 "/d{...} d" 可以在局部覆盖定义。

    "/_d { bind def } bind def",

    # "/m { moveto } _d"
    # 定义名为 m 的函数，用于在 PostScript 中进行移动到操作（moveto）。

    "/m { moveto } _d",

    # "/l { lineto } _d"
    # 定义名为 l 的函数，用于在 PostScript 中进行画线到操作（lineto）。

    "/l { lineto } _d",

    # "/r { rlineto } _d"
    # 定义名为 r 的函数，用于在 PostScript 中进行相对画线到操作（rlineto）。

    "/r { rlineto } _d",

    # "/c { curveto } _d"
    # 定义名为 c 的函数，用于在 PostScript 中进行曲线到操作（curveto）。

    "/c { curveto } _d",

    # "/cl { closepath } _d"
    # 定义名为 cl 的函数，用于在 PostScript 中进行封闭路径操作（closepath）。

    "/cl { closepath } _d",

    # "/ce { closepath eofill } _d"
    # 定义名为 ce 的函数，用于在 PostScript 中进行填充封闭路径操作（closepath eofill）。

    "/ce { closepath eofill } _d",

    # "/sc { setcachedevice } _d"
    # 定义名为 sc 的函数，用于在 PostScript 中设置缓存设备操作（setcachedevice）。

    "/sc { setcachedevice } _d",
]
```