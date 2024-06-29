# `D:\src\scipysrc\matplotlib\lib\matplotlib\texmanager.py`

```
r"""
Support for embedded TeX expressions in Matplotlib.

Requirements:

* LaTeX.
* \*Agg backends: dvipng>=1.6.
* PS backend: PSfrag, dvips, and Ghostscript>=9.0.
* PDF and SVG backends: if LuaTeX is present, it will be used to speed up some
  post-processing steps, but note that it is not used to parse the TeX string
  itself (only LaTeX is supported).

To enable TeX rendering of all text in your Matplotlib figure, set
:rc:`text.usetex` to True.

TeX and dvipng/dvips processing results are cached
in ~/.matplotlib/tex.cache for reuse between sessions.

`TexManager.get_rgba` can also be used to directly obtain raster output as RGBA
NumPy arrays.
"""

import functools
import hashlib
import logging
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook, dviread

_log = logging.getLogger(__name__)


def _usepackage_if_not_loaded(package, *, option=None):
    """
    Output LaTeX code that loads a package (possibly with an option) if it
    hasn't been loaded yet.

    LaTeX cannot load twice a package with different options, so this helper
    can be used to protect against users loading arbitrary packages/options in
    their custom preamble.
    """
    # 如果尚未加载指定的 LaTeX 包，输出加载该包的 LaTeX 代码
    option = f"[{option}]" if option is not None else ""
    return (
        r"\makeatletter"
        r"\@ifpackageloaded{%(package)s}{}{\usepackage%(option)s{%(package)s}}"
        r"\makeatother"
    ) % {"package": package, "option": option}


class TexManager:
    """
    Convert strings to dvi files using TeX, caching the results to a directory.

    The cache directory is called ``tex.cache`` and is located in the directory
    returned by `.get_cachedir`.

    Repeated calls to this constructor always return the same instance.
    """

    # 类变量，用于向后兼容，已被标记为私有属性（自版本3.8起）
    texcache = _api.deprecate_privatize_attribute("3.8")
    # 缓存 TexManager 实例的路径，位于 matplotlib 缓存目录下的 tex.cache 中
    _texcache = os.path.join(mpl.get_cachedir(), 'tex.cache')
    # 存储灰度数组的缓存，用于后续的处理
    _grey_arrayd = {}

    # 支持的字体系列
    _font_families = ('serif', 'sans-serif', 'cursive', 'monospace')
    _font_preambles = {
        'new century schoolbook': r'\renewcommand{\rmdefault}{pnc}',
        'bookman': r'\renewcommand{\rmdefault}{pbk}',
        'times': r'\usepackage{mathptmx}',
        'palatino': r'\usepackage{mathpazo}',
        'zapf chancery': r'\usepackage{chancery}',
        'cursive': r'\usepackage{chancery}',
        'charter': r'\usepackage{charter}',
        'serif': '',
        'sans-serif': '',
        'helvetica': r'\usepackage{helvet}',
        'avant garde': r'\usepackage{avant}',
        'courier': r'\usepackage{courier}',
        # Loading the type1ec package ensures that cm-super is installed, which
        # is necessary for Unicode computer modern.  (It also allows the use of
        # computer modern at arbitrary sizes, but that's just a side effect.)
        'monospace': r'\usepackage{type1ec}',
        'computer modern roman': r'\usepackage{type1ec}',
        'computer modern sans serif': r'\usepackage{type1ec}',
        'computer modern typewriter': r'\usepackage{type1ec}',
    }


    _font_types = {
        'new century schoolbook': 'serif',
        'bookman': 'serif',
        'times': 'serif',
        'palatino': 'serif',
        'zapf chancery': 'cursive',
        'charter': 'serif',
        'helvetica': 'sans-serif',
        'avant garde': 'sans-serif',
        'courier': 'monospace',
        'computer modern roman': 'serif',
        'computer modern sans serif': 'sans-serif',
        'computer modern typewriter': 'monospace',
    }


    @functools.lru_cache  # 始终返回相同的实例。
    def __new__(cls):
        Path(cls._texcache).mkdir(parents=True, exist_ok=True)
        return object.__new__(cls)


    @classmethod
    def _get_font_family_and_reduced(cls):
        """Return the font family name and whether the font is reduced."""
        ff = mpl.rcParams['font.family']  # 获取当前 matplotlib 字体设置
        ff_val = ff[0].lower() if len(ff) == 1 else None  # 将字体名称转换为小写
        if len(ff) == 1 and ff_val in cls._font_families:  # 如果字体名称在已知字体列表中
            return ff_val, False  # 返回字体名称和不减小字体标志
        elif len(ff) == 1 and ff_val in cls._font_preambles:  # 如果字体名称在字体前导信息中
            return cls._font_types[ff_val], True  # 返回对应的字体类型和减小字体标志
        else:
            _log.info('font.family must be one of (%s) when text.usetex is '
                      'True. serif will be used by default.',
                      ', '.join(cls._font_families))  # 记录警告日志，指出默认将使用衬线字体
            return 'serif', False  # 默认返回衬线字体和不减小字体标志
    @classmethod
    def _get_font_preamble_and_command(cls):
        # 获取请求的字体族和是否为缩减字体
        requested_family, is_reduced_font = cls._get_font_family_and_reduced()

        # 初始化存储预言（preamble）的字典
        preambles = {}

        # 遍历所有注册的字体族
        for font_family in cls._font_families:
            # 如果是缩减字体并且字体族匹配请求的字体族
            if is_reduced_font and font_family == requested_family:
                # 使用默认的字体预言（preamble）
                preambles[font_family] = cls._font_preambles[
                    mpl.rcParams['font.family'][0].lower()]
            else:
                # 获取当前字体族的所有字体配置
                rcfonts = mpl.rcParams[f"font.{font_family}"]
                # 遍历并转换为小写，查找匹配的字体预言（preamble）
                for i, font in enumerate(map(str.lower, rcfonts)):
                    if font in cls._font_preambles:
                        # 找到匹配的字体预言（preamble），记录日志并跳出循环
                        preambles[font_family] = cls._font_preambles[font]
                        _log.debug(
                            'family: %s, package: %s, font: %s, skipped: %s',
                            font_family, cls._font_preambles[font], rcfonts[i],
                            ', '.join(rcfonts[:i]),
                        )
                        break
                else:
                    # 如果没有找到匹配的字体预言（preamble），记录警告日志并使用默认的
                    _log.info('No LaTeX-compatible font found for the %s font'
                              'family in rcParams. Using default.',
                              font_family)
                    preambles[font_family] = cls._font_preambles[font_family]

        # 需要在 LaTeX 文件的 preamble 中包含以下包和命令：
        cmd = {preambles[family]
               for family in ['serif', 'sans-serif', 'monospace']}
        # 如果请求的字体族是 cursive，则添加相应的命令
        if requested_family == 'cursive':
            cmd.add(preambles['cursive'])
        # 添加额外的命令以支持文档中的字体类型
        cmd.add(r'\usepackage{type1cm}')
        # 按字母顺序排序并连接所有命令和包，形成完整的 preamble
        preamble = '\n'.join(sorted(cmd))
        # 根据请求的字体族确定字体命令
        fontcmd = (r'\sffamily' if requested_family == 'sans-serif' else
                   r'\ttfamily' if requested_family == 'monospace' else
                   r'\rmfamily')
        return preamble, fontcmd

    @classmethod
    def get_basefile(cls, tex, fontsize, dpi=None):
        """
        Return a filename based on a hash of the string, fontsize, and dpi.
        """
        # 生成文件名的哈希值，基于文本内容、字体大小和 dpi 设置
        src = cls._get_tex_source(tex, fontsize) + str(dpi)
        filehash = hashlib.md5(src.encode('utf-8')).hexdigest()
        filepath = Path(cls._texcache)

        num_letters, num_levels = 2, 2
        # 根据哈希值创建路径结构
        for i in range(0, num_letters*num_levels, num_letters):
            filepath = filepath / Path(filehash[i:i+2])

        # 创建目录，如果不存在则创建
        filepath.mkdir(parents=True, exist_ok=True)
        # 返回最终的文件路径
        return os.path.join(filepath, filehash)

    @classmethod
    def get_font_preamble(cls):
        """
        Return a string containing font configuration for the tex preamble.
        """
        # 获取字体配置的 preamble 字符串
        font_preamble, command = cls._get_font_preamble_and_command()
        return font_preamble

    @classmethod
    def get_custom_preamble(cls):
        """Return a string containing user additions to the tex preamble."""
        # 返回用户自定义的 preamble 配置
        return mpl.rcParams['text.latex.preamble']
    @classmethod
    def _get_tex_source(cls, tex, fontsize):
        """Return the complete TeX source for processing a TeX string."""
        # 获取完整的 TeX 源代码，用于处理给定的 TeX 字符串
        font_preamble, fontcmd = cls._get_font_preamble_and_command()
        # 计算基线间距
        baselineskip = 1.25 * fontsize
        # 拼接并返回 TeX 源代码的列表形式
        return "\n".join([
            r"\documentclass{article}",
            r"% Pass-through \mathdefault, which is used in non-usetex mode",
            r"% to use the default text font but was historically suppressed",
            r"% in usetex mode.",
            r"\newcommand{\mathdefault}[1]{#1}",
            font_preamble,
            r"\usepackage[utf8]{inputenc}",
            r"\DeclareUnicodeCharacter{2212}{\ensuremath{-}}",
            r"% geometry is loaded before the custom preamble as ",
            r"% convert_psfrags relies on a custom preamble to change the ",
            r"% geometry.",
            r"\usepackage[papersize=72in, margin=1in]{geometry}",
            cls.get_custom_preamble(),
            r"% Use `underscore` package to take care of underscores in text.",
            r"% The [strings] option allows to use underscores in file names.",
            _usepackage_if_not_loaded("underscore", option="strings"),
            r"% Custom packages (e.g. newtxtext) may already have loaded ",
            r"% textcomp with different options.",
            _usepackage_if_not_loaded("textcomp"),
            r"\pagestyle{empty}",
            r"\begin{document}",
            r"% The empty hbox ensures that a page is printed even for empty",
            r"% inputs, except when using psfrag which gets confused by it.",
            r"% matplotlibbaselinemarker is used by dviread to detect the",
            r"% last line's baseline.",
            rf"\fontsize{{{fontsize}}}{{{baselineskip}}}%",
            r"\ifdefined\psfrag\else\hbox{}\fi%",
            rf"{{{fontcmd} {tex}}}%",
            r"\end{document}",
        ])

    @classmethod
    def make_tex(cls, tex, fontsize):
        """
        Generate a tex file to render the tex string at a specific font size.

        Return the file name.
        """
        # 生成一个用于渲染指定字体大小的 TeX 文件
        texfile = cls.get_basefile(tex, fontsize) + ".tex"
        # 将 TeX 源代码写入到文件中，使用 UTF-8 编码
        Path(texfile).write_text(cls._get_tex_source(tex, fontsize),
                                 encoding='utf-8')
        # 返回生成的 TeX 文件名
        return texfile
    @classmethod
    def _run_checked_subprocess(cls, command, tex, *, cwd=None):
        # 调试信息：记录并格式化子进程的命令
        _log.debug(cbook._pformat_subprocess(command))
        try:
            # 执行命令，并捕获输出结果
            report = subprocess.check_output(
                command, cwd=cwd if cwd is not None else cls._texcache,
                stderr=subprocess.STDOUT)
        except FileNotFoundError as exc:
            # 如果命令未找到，抛出运行时错误
            raise RuntimeError(
                f'Failed to process string with tex because {command[0]} '
                'could not be found') from exc
        except subprocess.CalledProcessError as exc:
            # 如果命令执行失败，抛出运行时错误，并提供详细信息
            raise RuntimeError(
                '{prog} was not able to process the following string:\n'
                '{tex!r}\n\n'
                'Here is the full command invocation and its output:\n\n'
                '{format_command}\n\n'
                '{exc}\n\n'.format(
                    prog=command[0],
                    format_command=cbook._pformat_subprocess(command),
                    tex=tex.encode('unicode_escape'),
                    exc=exc.output.decode('utf-8', 'backslashreplace'))
                ) from None
        # 调试信息：记录子进程的输出报告
        _log.debug(report)
        # 返回子进程的输出报告
        return report

    @classmethod
    def make_dvi(cls, tex, fontsize):
        """
        Generate a dvi file containing latex's layout of tex string.

        Return the file name.
        """
        # 获取基本文件名
        basefile = cls.get_basefile(tex, fontsize)
        # 构建 dvi 文件名
        dvifile = '%s.dvi' % basefile
        # 如果 dvi 文件不存在，则生成它
        if not os.path.exists(dvifile):
            # 创建 tex 文件的路径对象
            texfile = Path(cls.make_tex(tex, fontsize))
            # 设置当前工作目录为 dvi 文件的父目录
            cwd = Path(dvifile).parent
            # 在临时目录中生成 dvi 文件，以避免竞态条件
            with TemporaryDirectory(dir=cwd) as tmpdir:
                tmppath = Path(tmpdir)
                # 运行检查过的子进程来生成 dvi 文件
                cls._run_checked_subprocess(
                    ["latex", "-interaction=nonstopmode", "--halt-on-error",
                     f"--output-directory={tmppath.name}",
                     f"{texfile.name}"], tex, cwd=cwd)
                # 原子性地替换临时生成的 dvi 文件
                (tmppath / Path(dvifile).name).replace(dvifile)
        # 返回生成的 dvi 文件名
        return dvifile

    @classmethod
    @classmethod
    def make_png(cls, tex, fontsize, dpi):
        """
        生成包含 LaTeX 渲染的 tex 字符串的 PNG 文件。

        返回文件名。
        """
        basefile = cls.get_basefile(tex, fontsize, dpi)
        pngfile = '%s.png' % basefile
        # 查看 get_rgba 函数关于背景的讨论
        if not os.path.exists(pngfile):
            dvifile = cls.make_dvi(tex, fontsize)
            cmd = ["dvipng", "-bg", "Transparent", "-D", str(dpi),
                   "-T", "tight", "-o", pngfile, dvifile]
            # 当进行测试时，为了可重现性禁用 FreeType 渲染；
            # 但 dvipng 1.16 存在一个 bug（在 f3ff241 中修复）会导致 --freetype0 模式失效，
            # 因此我们保持 FreeType 启用；图像会略微有所偏差。
            if (getattr(mpl, "_called_from_pytest", False) and
                    mpl._get_executable_info("dvipng").raw_version != "1.16"):
                cmd.insert(1, "--freetype0")
            cls._run_checked_subprocess(cmd, tex)
        return pngfile

    @classmethod
    def get_grey(cls, tex, fontsize=None, dpi=None):
        """返回 alpha 通道。"""
        if not fontsize:
            fontsize = mpl.rcParams['font.size']
        if not dpi:
            dpi = mpl.rcParams['savefig.dpi']
        key = cls._get_tex_source(tex, fontsize), dpi
        alpha = cls._grey_arrayd.get(key)
        if alpha is None:
            pngfile = cls.make_png(tex, fontsize, dpi)
            rgba = mpl.image.imread(os.path.join(cls._texcache, pngfile))
            cls._grey_arrayd[key] = alpha = rgba[:, :, -1]
        return alpha

    @classmethod
    def get_rgba(cls, tex, fontsize=None, dpi=None, rgb=(0, 0, 0)):
        """
        返回 LaTeX 渲染的 tex 字符串作为 RGBA 数组。

        Examples
        --------
        >>> texmanager = TexManager()
        >>> s = r"\TeX\ is $\displaystyle\sum_n\frac{-e^{i\pi}}{2^n}$!"
        >>> Z = texmanager.get_rgba(s, fontsize=12, dpi=80, rgb=(1, 0, 0))
        """
        alpha = cls.get_grey(tex, fontsize, dpi)
        rgba = np.empty((*alpha.shape, 4))
        rgba[..., :3] = mpl.colors.to_rgb(rgb)
        rgba[..., -1] = alpha
        return rgba

    @classmethod
    def get_text_width_height_descent(cls, tex, fontsize, renderer=None):
        """返回文本的宽度、高度和下降量。"""
        if tex.strip() == '':
            return 0, 0, 0
        dvifile = cls.make_dvi(tex, fontsize)
        dpi_fraction = renderer.points_to_pixels(1.) if renderer else 1
        with dviread.Dvi(dvifile, 72 * dpi_fraction) as dvi:
            page, = dvi
        # 需要返回总高度（包括下降量）。
        return page.width, page.height + page.descent, page.descent
```