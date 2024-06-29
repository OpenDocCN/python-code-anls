# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\backend_pgf.py`

```py
import codecs  # 导入 codecs 模块，用于处理各种编解码操作
import datetime  # 导入 datetime 模块，用于处理日期和时间
import functools  # 导入 functools 模块，提供高阶函数和操作工具
from io import BytesIO  # 从 io 模块中导入 BytesIO 类，用于操作内存中的二进制数据
import logging  # 导入 logging 模块，用于记录日志
import math  # 导入 math 模块，提供数学运算函数
import os  # 导入 os 模块，提供与操作系统交互的功能
import pathlib  # 导入 pathlib 模块，用于操作文件路径
import shutil  # 导入 shutil 模块，提供高级文件操作功能
import subprocess  # 导入 subprocess 模块，用于运行子进程
from tempfile import TemporaryDirectory  # 从 tempfile 模块中导入 TemporaryDirectory 类，用于创建临时目录
import weakref  # 导入 weakref 模块，支持弱引用对象的垃圾回收

from PIL import Image  # 从 PIL 库中导入 Image 模块，用于处理图像

import matplotlib as mpl  # 导入 matplotlib 库，并将其命名为 mpl
from matplotlib import _api, cbook, font_manager as fm  # 从 matplotlib 中导入 _api, cbook, fm 等模块
from matplotlib.backend_bases import (  # 从 matplotlib.backend_bases 模块中导入 _Backend, FigureCanvasBase, FigureManagerBase, RendererBase 类
    _Backend, FigureCanvasBase, FigureManagerBase, RendererBase
)
from matplotlib.backends.backend_mixed import MixedModeRenderer  # 从 matplotlib.backends.backend_mixed 模块中导入 MixedModeRenderer 类
from matplotlib.backends.backend_pdf import (  # 从 matplotlib.backends.backend_pdf 模块中导入 _create_pdf_info_dict, _datetime_to_pdf 函数
    _create_pdf_info_dict, _datetime_to_pdf)
from matplotlib.path import Path  # 从 matplotlib.path 模块中导入 Path 类
from matplotlib.figure import Figure  # 从 matplotlib.figure 模块中导入 Figure 类
from matplotlib.font_manager import FontProperties  # 从 matplotlib.font_manager 模块中导入 FontProperties 类
from matplotlib._pylab_helpers import Gcf  # 从 matplotlib._pylab_helpers 模块中导入 Gcf 类

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象

_DOCUMENTCLASS = r"\documentclass{article}"  # 定义 LaTeX 文档类为 article 类型


# Note: When formatting floating point values, it is important to use the
# %f/{:f} format rather than %s/{} to avoid triggering scientific notation,
# which is not recognized by TeX.

def _get_preamble():
    """Prepare a LaTeX preamble based on the rcParams configuration."""
    font_size_pt = FontProperties(
        size=mpl.rcParams["font.size"]
    ).get_size_in_points()  # 获取当前 matplotlib 配置中的字体大小并转换为点（pt）单位
    return "\n".join([
        # Remove Matplotlib's custom command \mathdefault.  (Not using
        # \mathnormal instead since this looks odd with Computer Modern.)
        r"\def\mathdefault#1{#1}",  # 定义 LaTeX 命令，取消 Matplotlib 的自定义 \mathdefault 命令
        # Use displaystyle for all math.
        r"\everymath=\expandafter{\the\everymath\displaystyle}",  # 设置所有数学公式使用 displaystyle 显示风格
        # Set up font sizes to match font.size setting.
        # If present, use the KOMA package scrextend to adjust the standard
        # LaTeX font commands (\tiny, ..., \normalsize, ..., \Huge) accordingly.
        # Otherwise, only set \normalsize, manually.
        r"\IfFileExists{scrextend.sty}{",
        r"  \usepackage[fontsize=%fpt]{scrextend}" % font_size_pt,
        r"}{",
        r"  \renewcommand{\normalsize}{\fontsize{%f}{%f}\selectfont}"
        % (font_size_pt, 1.2 * font_size_pt),
        r"  \normalsize",
        r"}",
        # Allow pgf.preamble to override the above definitions.
        mpl.rcParams["pgf.preamble"],  # 将 matplotlib 的 pgf.preamble 加入到 LaTeX 的 preamble 中
        *([
            r"\ifdefined\pdftexversion\else  % non-pdftex case.",
            r"  \usepackage{fontspec}",
        ] + [
            r"  \%s{%s}[Path=\detokenize{%s/}]"
            % (command, path.name, path.parent.as_posix())
            for command, path in zip(
                ["setmainfont", "setsansfont", "setmonofont"],
                [pathlib.Path(fm.findfont(family))
                 for family in ["serif", "sans\\-serif", "monospace"]]
            )
        ] + [r"\fi"] if mpl.rcParams["pgf.rcfonts"] else []),
        # Documented as "must come last".
        mpl.texmanager._usepackage_if_not_loaded("underscore", option="strings"),  # 使用 mpl.texmanager 中的方法加载 underscore 包
    ])


# It's better to use only one unit for all coordinates, since the
# arithmetic in latex seems to produce inaccurate conversions.
latex_pt_to_in = 1. / 72.27  # 定义 LaTeX 点（pt）到英寸（in）的转换比例
# LaTeX中的长度单位转换：1 LaTeX point (pt) 等于多少英寸 (in)
latex_in_to_pt = 1. / latex_pt_to_in

# Matplotlib中的长度单位转换：1 point (pt) 等于多少英寸 (in)
mpl_pt_to_in = 1. / 72.

# Matplotlib中的长度单位转换：1 英寸 (in) 等于多少 point (pt)
mpl_in_to_pt = 1. / mpl_pt_to_in


def _tex_escape(text):
    r"""
    对在 LaTeX 文档中使用的文本进行必要和/或有用的替换。
    """
    return text.replace("\N{MINUS SIGN}", r"\ensuremath{-}")


def _writeln(fh, line):
    # 在写入行末尾添加 '%' 可以防止 TeX 插入多余的空格
    # (https://tex.stackexchange.com/questions/7453)
    fh.write(line)
    fh.write("%\n")


def _escape_and_apply_props(s, prop):
    """
    生成一个 TeX 字符串，用于以 *prop* 字体属性渲染字符串 *s*，同时对 *s* 应用任何必要的转义。
    """
    commands = []

    families = {"serif": r"\rmfamily", "sans": r"\sffamily",
                "sans-serif": r"\sffamily", "monospace": r"\ttfamily"}
    family = prop.get_family()[0]
    if family in families:
        commands.append(families[family])
    elif not mpl.rcParams["pgf.rcfonts"]:
        commands.append(r"\fontfamily{\familydefault}")
    elif any(font.name == family for font in fm.fontManager.ttflist):
        commands.append(
            r"\ifdefined\pdftexversion\else\setmainfont{%s}\rmfamily\fi" % family)
    else:
        _log.warning("Ignoring unknown font: %s", family)

    size = prop.get_size_in_points()
    commands.append(r"\fontsize{%f}{%f}" % (size, size * 1.2))

    styles = {"normal": r"", "italic": r"\itshape", "oblique": r"\slshape"}
    commands.append(styles[prop.get_style()])

    boldstyles = ["semibold", "demibold", "demi", "bold", "heavy",
                  "extra bold", "black"]
    if prop.get_weight() in boldstyles:
        commands.append(r"\bfseries")

    commands.append(r"\selectfont")
    return (
        "{"
        + "".join(commands)
        + r"\catcode`\^=\active\def^{\ifmmode\sp\else\^{}\fi}"
        # 正常来说，将 % 的 catcode 设置为 12 ("normal character") 应该足够了；
        # 这在 TeXLive 2021 上有效，但在 2018 年版本上不行，所以我们也将其设置为活动字符。
        + r"\catcode`\%=\active\def%{\%}"
        + _tex_escape(s)
        + "}"
    )


def _metadata_to_str(key, value):
    """将元数据键/值转换为 hyperref 可接受的格式。"""
    if isinstance(value, datetime.datetime):
        value = _datetime_to_pdf(value)
    elif key == 'Trapped':
        value = value.name.decode('ascii')
    else:
        value = str(value)
    return f'{key}={{{value}}}'


def make_pdf_to_png_converter():
    """返回一个将 pdf 文件转换为 png 文件的函数。"""
    try:
        mpl._get_executable_info("pdftocairo")
    except mpl.ExecutableNotFoundError:
        pass
    else:
        return lambda pdffile, pngfile, dpi: subprocess.check_output(
            ["pdftocairo", "-singlefile", "-transp", "-png", "-r", "%d" % dpi,
             pdffile, os.path.splitext(pngfile)[0]],
            stderr=subprocess.STDOUT)
    try:
        gs_info = mpl._get_executable_info("gs")
    # 如果捕获到 mpl.ExecutableNotFoundError 异常，则忽略，不做任何处理
    except mpl.ExecutableNotFoundError:
        pass
    # 如果没有捕获到异常，则执行以下代码块
    else:
        # 返回一个 lambda 函数，接受 pdffile、pngfile、dpi 三个参数，用于执行转换操作
        return lambda pdffile, pngfile, dpi: subprocess.check_output(
            # 调用 Ghostscript 执行命令，将 PDF 文件转换为 PNG 文件
            [gs_info.executable,
             '-dQUIET', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-dNOPROMPT',
             '-dUseCIEColor', '-dTextAlphaBits=4',
             '-dGraphicsAlphaBits=4', '-dDOINTERPOLATE',
             '-sDEVICE=pngalpha', '-sOutputFile=%s' % pngfile,
             '-r%d' % dpi, pdffile],
            stderr=subprocess.STDOUT)
    # 如果没有执行 return 返回结果（通常因为没有找到合适的 PDF 到 PNG 渲染器），抛出 RuntimeError 异常
    raise RuntimeError("No suitable pdf to png renderer found.")
class LatexError(Exception):
    # 自定义异常类，用于处理 LaTeX 相关错误
    def __init__(self, message, latex_output=""):
        super().__init__(message)
        self.latex_output = latex_output

    def __str__(self):
        # 返回异常消息字符串，若存在 LaTeX 输出，则添加到消息末尾
        s, = self.args
        if self.latex_output:
            s += "\n" + self.latex_output
        return s


class LatexManager:
    """
    LaTeX 管理器，打开 LaTeX 应用实例来确定文本元素的度量。
    可以通过设置 `.rcParams` 中的字体或自定义导言区来修改 LaTeX 环境。
    """

    @staticmethod
    def _build_latex_header():
        # 构建 LaTeX 文件的头部内容
        latex_header = [
            _DOCUMENTCLASS,  # LaTeX 文档类
            # 在注释中包含 TeX 程序名称以进行缓存失效检查
            rf"% !TeX program = {mpl.rcParams['pgf.texsystem']}",
            # 测试 \includegraphics 是否支持 interpolate 选项
            r"\usepackage{graphicx}",
            _get_preamble(),  # 获取 LaTeX 文件的导言区
            r"\begin{document}",  # LaTeX 文件正文开始
            r"\typeout{pgf_backend_query_start}",  # 输出查询起始信息
        ]
        return "\n".join(latex_header)

    @classmethod
    def _get_cached_or_new(cls):
        """
        如果头部和 TeX 系统未更改，则返回先前的 LatexManager 实例；否则返回一个新实例。
        """
        return cls._get_cached_or_new_impl(cls._build_latex_header())

    @classmethod
    @functools.lru_cache(1)
    def _get_cached_or_new_impl(cls, header):  # _get_cached_or_new 的辅助方法
        return cls()

    def _stdin_writeln(self, s):
        # 如果 self.latex 为空，则设置 LaTeX 进程
        if self.latex is None:
            self._setup_latex_process()
        # 将字符串 s 写入 LaTeX 进程的标准输入
        self.latex.stdin.write(s)
        self.latex.stdin.write("\n")  # 写入换行符
        self.latex.stdin.flush()  # 刷新标准输入缓冲区

    def _expect(self, s):
        # 将字符串 s 转换为字符列表
        s = list(s)
        chars = []
        while True:
            # 从 LaTeX 进程的标准输出中读取一个字符
            c = self.latex.stdout.read(1)
            chars.append(c)  # 将字符添加到列表中
            # 检查是否匹配字符串 s 的结尾
            if chars[-len(s):] == s:
                break
            # 如果没有读取到字符，则终止 LaTeX 进程并引发 LatexError 异常
            if not c:
                self.latex.kill()
                self.latex = None
                raise LatexError("LaTeX process halted", "".join(chars))
        return "".join(chars)  # 返回匹配的字符组成的字符串

    def _expect_prompt(self):
        # 等待匹配标准输出中的换行符 '*'
        return self._expect("\n*")
    # 初始化函数，用于设置 LaTeX 运行的临时目录，并注册其在销毁时进行清理
    def __init__(self):
        self._tmpdir = TemporaryDirectory()  # 创建临时目录对象
        self.tmpdir = self._tmpdir.name  # 获取临时目录的路径
        self._finalize_tmpdir = weakref.finalize(self, self._tmpdir.cleanup)  # 注册在销毁时清理临时目录的回调函数

        # 测试 LaTeX 设置，确保子进程的干净启动
        self._setup_latex_process(expect_reply=False)  # 调用方法设置 LaTeX 进程
        stdout, stderr = self.latex.communicate("\n\\makeatletter\\@@end\n")  # 向 LaTeX 进程发送测试输入
        if self.latex.returncode != 0:  # 检查 LaTeX 进程返回码
            raise LatexError(
                f"LaTeX errored (probably missing font or error in preamble) "
                f"while processing the following input:\n"
                f"{self._build_latex_header()}",
                stdout)
        self.latex = None  # 将 self.latex 设置为 None，在首次使用时重新设置

        # 每个实例的缓存
        self._get_box_metrics = functools.lru_cache(self._get_box_metrics)

    # 设置 LaTeX 进程的方法，打开真实工作的 LaTeX 进程，并注册在销毁时清理的回调
    def _setup_latex_process(self, *, expect_reply=True):
        try:
            self.latex = subprocess.Popen(
                [mpl.rcParams["pgf.texsystem"], "-halt-on-error"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                encoding="utf-8", cwd=self.tmpdir)  # 启动 LaTeX 子进程
        except FileNotFoundError as err:
            raise RuntimeError(
                f"{mpl.rcParams['pgf.texsystem']!r} not found; install it or change "
                f"rcParams['pgf.texsystem'] to an available TeX implementation"
            ) from err
        except OSError as err:
            raise RuntimeError(
                f"Error starting {mpl.rcParams['pgf.texsystem']!r}") from err

        def finalize_latex(latex):
            latex.kill()  # 终止 LaTeX 进程
            try:
                latex.communicate()  # 等待进程终止
            except RuntimeError:
                latex.wait()

        self._finalize_latex = weakref.finalize(
            self, finalize_latex, self.latex)  # 注册在销毁时清理 LaTeX 进程的回调函数

        # 向 LaTeX 进程写入带有 'pgf_backend_query_start' 标记的头信息
        self._stdin_writeln(self._build_latex_header())
        if expect_reply:  # 如果期望接收响应，则等待 'pgf_backend_query_start' 标记的出现
            self._expect("*pgf_backend_query_start")
            self._expect_prompt()

    # 获取文本在当前 LaTeX 环境中排版时的宽度、总高度和下降高度（以 TeX 点为单位）
    def get_width_height_descent(self, text, prop):
        """
        获取文本在当前 LaTeX 环境中排版时的宽度、总高度和下降高度（以 TeX 点为单位）。
        """
        return self._get_box_metrics(_escape_and_apply_props(text, prop))  # 调用方法计算文本的度量值
    def _get_box_metrics(self, tex):
        """
        Get the width, total height and descent (in TeX points) for a TeX
        command's output in the current LaTeX environment.
        """
        # 将文本框发送到TeX并请求度量类型输出。

        self._stdin_writeln(
            # \sbox 在其参数内部不处理字符类别分配，
            # 因此在外部重复分配 "^" 和 "%" 的字符类别。
            r"{\catcode`\^=\active\catcode`\%%=\active\sbox0{%s}"
            r"\typeout{\the\wd0,\the\ht0,\the\dp0}}"
            % tex)

        try:
            # 期待获得TeX回应
            answer = self._expect_prompt()
        except LatexError as err:
            # 在此及以下，使用'{}'代替{!r}，以避免加倍所有反斜杠。
            raise ValueError("Error measuring {}\nLaTeX Output:\n{}"
                             .format(tex, err.latex_output)) from err

        try:
            # 从回应字符串解析度量。最后一行是提示符，倒数第二行是来自 \typeout 的空白行。
            width, height, offset = answer.splitlines()[-3].split(",")
        except Exception as err:
            raise ValueError("Error measuring {}\nLaTeX Output:\n{}"
                             .format(tex, answer)) from err

        # 将字符串转换为浮点数，去除TeX单位(pt)。
        w, h, o = float(width[:-2]), float(height[:-2]), float(offset[:-2])

        # LaTeX返回的高度从基线到顶部；
        # Matplotlib期望的高度从底部到顶部。
        return w, h + o, o
# 使用 functools.lru_cache 装饰器，缓存函数调用结果，最多缓存一个
@functools.lru_cache(1)
# 定义获取插入图片命令的函数
def _get_image_inclusion_command():
    # 获取或创建 LatexManager 实例
    man = LatexManager._get_cached_or_new()
    # 向 LatexManager 写入命令，插入指定路径下的图片
    man._stdin_writeln(
        r"\includegraphics[interpolate=true]{%s}"
        # 在 Windows 上不要处理反斜杠
        % cbook._get_data_path("images/matplotlib.png").as_posix())
    try:
        # 等待 LatexManager 的提示符
        man._expect_prompt()
        # 成功后返回插入图片的命令
        return r"\includegraphics"
    except LatexError:
        # 如果出现 LatexError，清除缓存并返回另一个插入图片的命令
        LatexManager._get_cached_or_new_impl.cache_clear()
        return r"\pgfimage"


# 定义 RendererPgf 类，继承自 RendererBase
class RendererPgf(RendererBase):

    # 构造方法，初始化 PGF 渲染器
    def __init__(self, figure, fh):
        """
        Create a new PGF renderer that translates any drawing instruction
        into text commands to be interpreted in a latex pgfpicture environment.

        Attributes
        ----------
        figure : `~matplotlib.figure.Figure`
            Matplotlib figure to initialize height, width and dpi from.
        fh : file-like
            File handle for the output of the drawing commands.
        """

        super().__init__()
        self.dpi = figure.dpi  # 设置 DPI 属性
        self.fh = fh  # 设置输出文件句柄属性
        self.figure = figure  # 设置图形属性
        self.image_counter = 0  # 图片计数器初始化为 0

    # 实现绘制标记方法，继承自基类 RendererBase
    def draw_markers(self, gc, marker_path, marker_trans, path, trans,
                     rgbFace=None):
        # docstring 继承

        _writeln(self.fh, r"\begin{pgfscope}")  # 写入 PGF 环境的开始命令

        # 将显示单位转换为英寸
        f = 1. / self.dpi

        # 设置样式和剪辑
        self._print_pgf_clip(gc)  # 打印 PGF 剪辑样式
        self._print_pgf_path_styles(gc, rgbFace)  # 打印 PGF 路径样式

        # 构建标记定义
        bl, tr = marker_path.get_extents(marker_trans).get_points()
        coords = bl[0] * f, bl[1] * f, tr[0] * f, tr[1] * f
        _writeln(self.fh,
                 r"\pgfsys@defobject{currentmarker}"
                 r"{\pgfqpoint{%fin}{%fin}}{\pgfqpoint{%fin}{%fin}}{" % coords)  # 定义当前标记对象
        self._print_pgf_path(None, marker_path, marker_trans)  # 打印 PGF 路径
        self._pgf_path_draw(stroke=gc.get_linewidth() != 0.0,
                            fill=rgbFace is not None)  # 绘制 PGF 路径
        _writeln(self.fh, r"}")

        maxcoord = 16383 / 72.27 * self.dpi  # LaTeX 中的最大尺寸限制
        clip = (-maxcoord, -maxcoord, maxcoord, maxcoord)

        # 对每个顶点绘制标记
        for point, code in path.iter_segments(trans, simplify=False,
                                              clip=clip):
            x, y = point[0] * f, point[1] * f
            _writeln(self.fh, r"\begin{pgfscope}")  # 开始新的 PGF 环境
            _writeln(self.fh, r"\pgfsys@transformshift{%fin}{%fin}" % (x, y))  # 转换并平移
            _writeln(self.fh, r"\pgfsys@useobject{currentmarker}{}")  # 使用当前标记对象
            _writeln(self.fh, r"\end{pgfscope}")  # 结束当前 PGF 环境

        _writeln(self.fh, r"\end{pgfscope}")  # 结束最外层 PGF 环境
    def draw_path(self, gc, path, transform, rgbFace=None):
        # 绘制路径函数，使用给定的图形上下文(gc)、路径(path)和变换(transform)，可选填充颜色(rgbFace)
        
        # 将命令写入文件，开始一个pgfscope环境
        _writeln(self.fh, r"\begin{pgfscope}")
        
        # 绘制路径
        self._print_pgf_clip(gc)  # 打印PGF剪裁路径的命令
        self._print_pgf_path_styles(gc, rgbFace)  # 打印PGF路径样式的命令
        self._print_pgf_path(gc, path, transform, rgbFace)  # 打印PGF路径绘制命令
        self._pgf_path_draw(stroke=gc.get_linewidth() != 0.0, fill=rgbFace is not None)  # 执行PGF路径绘制命令
        
        # 结束pgfscope环境
        _writeln(self.fh, r"\end{pgfscope}")

        # 如果有填充图案，则在顶部绘制
        if gc.get_hatch():
            _writeln(self.fh, r"\begin{pgfscope}")
            self._print_pgf_path_styles(gc, rgbFace)  # 打印PGF路径样式的命令

            # 结合剪裁和路径以进行剪裁
            self._print_pgf_clip(gc)  # 打印PGF剪裁路径的命令
            self._print_pgf_path(gc, path, transform, rgbFace)  # 打印PGF路径绘制命令
            _writeln(self.fh, r"\pgfusepath{clip}")  # 执行PGF使用剪裁路径命令

            # 构建图案定义
            _writeln(self.fh,
                     r"\pgfsys@defobject{currentpattern}"
                     r"{\pgfqpoint{0in}{0in}}{\pgfqpoint{1in}{1in}}{")
            _writeln(self.fh, r"\begin{pgfscope}")
            _writeln(self.fh,
                     r"\pgfpathrectangle"
                     r"{\pgfqpoint{0in}{0in}}{\pgfqpoint{1in}{1in}}")
            _writeln(self.fh, r"\pgfusepath{clip}")
            scale = mpl.transforms.Affine2D().scale(self.dpi)
            self._print_pgf_path(None, gc.get_hatch_path(), scale)  # 打印PGF路径绘制命令
            self._pgf_path_draw(stroke=True)  # 执行PGF路径绘制命令
            _writeln(self.fh, r"\end{pgfscope}")
            _writeln(self.fh, r"}")

            # 重复图案，填充路径的边界矩形
            f = 1. / self.dpi
            (xmin, ymin), (xmax, ymax) = \
                path.get_extents(transform).get_points()
            xmin, xmax = f * xmin, f * xmax
            ymin, ymax = f * ymin, f * ymax
            repx, repy = math.ceil(xmax - xmin), math.ceil(ymax - ymin)
            _writeln(self.fh,
                     r"\pgfsys@transformshift{%fin}{%fin}" % (xmin, ymin))
            for iy in range(repy):
                for ix in range(repx):
                    _writeln(self.fh, r"\pgfsys@useobject{currentpattern}{}")  # 执行PGF使用图案对象命令
                    _writeln(self.fh, r"\pgfsys@transformshift{1in}{0in}")  # 执行PGF图案变换命令
                _writeln(self.fh, r"\pgfsys@transformshift{-%din}{0in}" % repx)  # 执行PGF图案变换命令
                _writeln(self.fh, r"\pgfsys@transformshift{0in}{1in}")  # 执行PGF图案变换命令

            _writeln(self.fh, r"\end{pgfscope}")  # 结束pgfscope环境
    def _print_pgf_clip(self, gc):
        f = 1. / self.dpi
        # 获取当前图形上下文的裁剪框
        bbox = gc.get_clip_rectangle()
        if bbox:
            # 获取裁剪框的两个顶点坐标和宽高
            p1, p2 = bbox.get_points()
            w, h = p2 - p1
            # 计算裁剪框在 PGF 中的坐标和尺寸，并写入文件
            coords = p1[0] * f, p1[1] * f, w * f, h * f
            _writeln(self.fh,
                     r"\pgfpathrectangle"
                     r"{\pgfqpoint{%fin}{%fin}}{\pgfqpoint{%fin}{%fin}}"
                     % coords)
            _writeln(self.fh, r"\pgfusepath{clip}")

        # 检查是否存在裁剪路径
        clippath, clippath_trans = gc.get_clip_path()
        if clippath is not None:
            # 如果存在裁剪路径，则打印裁剪路径到 PGF 文件
            self._print_pgf_path(gc, clippath, clippath_trans)
            _writeln(self.fh, r"\pgfusepath{clip}")

    def _print_pgf_path_styles(self, gc, rgbFace):
        # 线帽样式映射表
        capstyles = {"butt": r"\pgfsetbuttcap",
                     "round": r"\pgfsetroundcap",
                     "projecting": r"\pgfsetrectcap"}
        # 写入当前线帽样式到 PGF 文件
        _writeln(self.fh, capstyles[gc.get_capstyle()])

        # 连接样式映射表
        joinstyles = {"miter": r"\pgfsetmiterjoin",
                      "round": r"\pgfsetroundjoin",
                      "bevel": r"\pgfsetbeveljoin"}
        # 写入当前连接样式到 PGF 文件
        _writeln(self.fh, joinstyles[gc.get_joinstyle()])

        # 是否有填充色
        has_fill = rgbFace is not None

        # 如果需要强制透明度，则设置填充和描边的透明度
        if gc.get_forced_alpha():
            fillopacity = strokeopacity = gc.get_alpha()
        else:
            strokeopacity = gc.get_rgb()[3]
            fillopacity = rgbFace[3] if has_fill and len(rgbFace) > 3 else 1.0

        # 如果有填充色，则定义并设置当前填充色到 PGF 文件
        if has_fill:
            _writeln(self.fh,
                     r"\definecolor{currentfill}{rgb}{%f,%f,%f}"
                     % tuple(rgbFace[:3]))
            _writeln(self.fh, r"\pgfsetfillcolor{currentfill}")
        # 如果有填充色且透明度不为1.0，则设置填充透明度到 PGF 文件
        if has_fill and fillopacity != 1.0:
            _writeln(self.fh, r"\pgfsetfillopacity{%f}" % fillopacity)

        # 线宽度和颜色
        lw = gc.get_linewidth() * mpl_pt_to_in * latex_in_to_pt
        stroke_rgba = gc.get_rgb()
        # 设置当前线宽到 PGF 文件
        _writeln(self.fh, r"\pgfsetlinewidth{%fpt}" % lw)
        # 定义并设置当前描边色到 PGF 文件
        _writeln(self.fh,
                 r"\definecolor{currentstroke}{rgb}{%f,%f,%f}"
                 % stroke_rgba[:3])
        _writeln(self.fh, r"\pgfsetstrokecolor{currentstroke}")
        # 如果描边透明度不为1.0，则设置描边透明度到 PGF 文件
        if strokeopacity != 1.0:
            _writeln(self.fh, r"\pgfsetstrokeopacity{%f}" % strokeopacity)

        # 线型样式
        dash_offset, dash_list = gc.get_dashes()
        if dash_list is None:
            # 如果没有虚线样式，则设置空的虚线到 PGF 文件
            _writeln(self.fh, r"\pgfsetdash{}{0pt}")
        else:
            # 如果有虚线样式，则设置虚线样式到 PGF 文件
            _writeln(self.fh,
                     r"\pgfsetdash{%s}{%fpt}"
                     % ("".join(r"{%fpt}" % dash for dash in dash_list),
                        dash_offset))

    def _pgf_path_draw(self, stroke=True, fill=False):
        actions = []
        # 如果需要描边，则添加描边动作
        if stroke:
            actions.append("stroke")
        # 如果需要填充，则添加填充动作
        if fill:
            actions.append("fill")
        # 将绘制路径动作写入 PGF 文件
        _writeln(self.fh, r"\pgfusepath{%s}" % ",".join(actions))
    def option_scale_image(self):
        # docstring inherited
        # 返回 True，表示支持图像缩放选项
        return True

    def option_image_nocomposite(self):
        # docstring inherited
        # 检查是否禁用了图像合成，如果是则返回 True
        return not mpl.rcParams['image.composite_image']

    def draw_image(self, gc, x, y, im, transform=None):
        # docstring inherited
        # 获取图像的高度和宽度
        h, w = im.shape[:2]
        # 如果图像的宽度或高度为0，则直接返回，不进行绘制
        if w == 0 or h == 0:
            return

        # 检查文件句柄的存在性
        if not os.path.exists(getattr(self.fh, "name", "")):
            # 如果文件句柄不存在，则抛出异常
            raise ValueError(
                "streamed pgf-code does not support raster graphics, consider "
                "using the pgf-to-pdf option")

        # 将图像保存为 PNG 文件
        path = pathlib.Path(self.fh.name)
        fname_img = "%s-img%d.png" % (path.stem, self.image_counter)
        Image.fromarray(im[::-1]).save(path.parent / fname_img)
        self.image_counter += 1

        # 在 pgf 图片中引用这个图像
        _writeln(self.fh, r"\begin{pgfscope}")
        self._print_pgf_clip(gc)
        f = 1. / self.dpi  # 从显示坐标到英寸的转换因子
        if transform is None:
            # 如果没有提供变换参数，则进行平移变换
            _writeln(self.fh,
                     r"\pgfsys@transformshift{%fin}{%fin}" % (x * f, y * f))
            w, h = w * f, h * f
        else:
            # 如果提供了变换参数，则进行仿射变换
            tr1, tr2, tr3, tr4, tr5, tr6 = transform.frozen().to_values()
            _writeln(self.fh,
                     r"\pgfsys@transformcm{%f}{%f}{%f}{%f}{%fin}{%fin}" %
                     (tr1 * f, tr2 * f, tr3 * f, tr4 * f,
                      (tr5 + x) * f, (tr6 + y) * f))
            w = h = 1  # 缩放已经包含在变换中
        interp = str(transform is None).lower()  # PDF 阅读器中的插值设置
        _writeln(self.fh,
                 r"\pgftext[left,bottom]"
                 r"{%s[interpolate=%s,width=%fin,height=%fin]{%s}}" %
                 (_get_image_inclusion_command(),
                  interp, w, h, fname_img))
        _writeln(self.fh, r"\end{pgfscope}")

    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        # docstring inherited
        # 使用 LaTeX 渲染文本
        self.draw_text(gc, x, y, s, prop, angle, ismath="TeX", mtext=mtext)
    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # docstring inherited
        # 准备字符串以供TeX使用，对其进行转义并应用属性
        s = _escape_and_apply_props(s, prop)

        # 将文本绘制命令写入文件流
        _writeln(self.fh, r"\begin{pgfscope}")
        # 根据图形上下文设置绘制区域裁剪
        self._print_pgf_clip(gc)

        # 获取绘制文本的透明度并设置到绘图库中
        alpha = gc.get_alpha()
        if alpha != 1.0:
            _writeln(self.fh, r"\pgfsetfillopacity{%f}" % alpha)
            _writeln(self.fh, r"\pgfsetstrokeopacity{%f}" % alpha)
        
        # 获取绘制文本的RGB颜色并定义为TeX中的颜色
        rgb = tuple(gc.get_rgb())[:3]
        _writeln(self.fh, r"\definecolor{textcolor}{rgb}{%f,%f,%f}" % rgb)
        _writeln(self.fh, r"\pgfsetstrokecolor{textcolor}")
        _writeln(self.fh, r"\pgfsetfillcolor{textcolor}")
        s = r"\color{textcolor}" + s

        # 获取图形的DPI（每英寸像素数）
        dpi = self.figure.dpi
        text_args = []

        # 如果有mtext并且可以支持文本锚定，则获取原始坐标和对齐信息
        if mtext and (
                (angle == 0 or
                 mtext.get_rotation_mode() == "anchor") and
                mtext.get_verticalalignment() != "center_baseline"):
            pos = mtext.get_unitless_position()
            x, y = mtext.get_transform().transform(pos)
            halign = {"left": "left", "right": "right", "center": ""}
            valign = {"top": "top", "bottom": "bottom",
                      "baseline": "base", "center": ""}
            text_args.extend([
                f"x={x/dpi:f}in",
                f"y={y/dpi:f}in",
                halign[mtext.get_horizontalalignment()],
                valign[mtext.get_verticalalignment()],
            ])
        else:
            # 如果不支持文本锚定，则使用Matplotlib提供的文本布局
            text_args.append(f"x={x/dpi:f}in, y={y/dpi:f}in, left, base")

        # 如果文本有旋转角度，则添加旋转角度信息
        if angle != 0:
            text_args.append("rotate=%f" % angle)

        # 将文本绘制命令写入文件流
        _writeln(self.fh, r"\pgftext[%s]{%s}" % (",".join(text_args), s))
        _writeln(self.fh, r"\end{pgfscope}")

    def get_text_width_height_descent(self, s, prop, ismath):
        # docstring inherited
        # 获取文本的宽度、高度和下降量，并将单位转换为显示单位
        w, h, d = (LatexManager._get_cached_or_new()
                   .get_width_height_descent(s, prop))
        # TODO: 应该使用latex_pt_to_in而不是mpl_pt_to_in，但在文本周围留出更多空间看起来更好，
        # 而且LaTeX报告的边界框非常窄
        f = mpl_pt_to_in * self.dpi
        return w * f, h * f, d * f

    def flipy(self):
        # docstring inherited
        # 始终返回False，不翻转Y轴
        return False

    def get_canvas_width_height(self):
        # docstring inherited
        # 返回画布的宽度和高度，单位为像素
        return (self.figure.get_figwidth() * self.dpi,
                self.figure.get_figheight() * self.dpi)

    def points_to_pixels(self, points):
        # docstring inherited
        # 将点数转换为像素数，使用Matplotlib的点数转换比例和当前DPI
        return points * mpl_pt_to_in * self.dpi
class FigureCanvasPgf(FigureCanvasBase):
    # 定义文件类型字典，用于后续输出文件格式的选择
    filetypes = {"pgf": "LaTeX PGF picture",
                 "pdf": "LaTeX compiled PGF picture",
                 "png": "Portable Network Graphics", }

    # 返回默认的文件类型为 'pdf'
    def get_default_filetype(self):
        return 'pdf'

    # 将 PGF 图形输出到文件句柄 fh 中
    def _print_pgf_to_fh(self, fh, *, bbox_inches_restore=None):

        # PGF 图形的头部文本，提供了包含图形的 LaTeX 文档输入方法和必需的包引用
        header_text = """%% Creator: Matplotlib, PGF backend
%%
%% To include the figure in your LaTeX document, write
%%   \\input{<filename>.pgf}
%%
%% Make sure the required packages are loaded in your preamble
%%   \\usepackage{pgf}
%%
%% Also ensure that all the required font packages are loaded; for instance,
%% the lmodern package is sometimes necessary when using math font.
%%   \\usepackage{lmodern}
%%
%% Figures using additional raster images can only be included by \\input if
%% they are in the same directory as the main LaTeX file. For loading figures
%% from other directories you can use the `import` package
%%   \\usepackage{import}
%%
"""

        # 追加后端使用的导言部分作为调试的注释
        header_info_preamble = ["%% Matplotlib used the following preamble"]
        for line in _get_preamble().splitlines():
            header_info_preamble.append("%%   " + line)
        header_info_preamble.append("%%")
        header_info_preamble = "\n".join(header_info_preamble)

        # 获取图形的宽度和高度（单位为英寸），以及 DPI（每英寸点数）
        w, h = self.figure.get_figwidth(), self.figure.get_figheight()
        dpi = self.figure.dpi

        # 创建 pgfpicture 环境并写入 PGF 代码
        fh.write(header_text)
        fh.write(header_info_preamble)
        fh.write("\n")
        _writeln(fh, r"\begingroup")
        _writeln(fh, r"\makeatletter")
        _writeln(fh, r"\begin{pgfpicture}")
        _writeln(fh,
                 r"\pgfpathrectangle{\pgfpointorigin}{\pgfqpoint{%fin}{%fin}}"
                 % (w, h))
        _writeln(fh, r"\pgfusepath{use as bounding box, clip}")

        # 创建混合模式渲染器，将图形渲染为 PGF 代码
        renderer = MixedModeRenderer(self.figure, w, h, dpi,
                                     RendererPgf(self.figure, fh),
                                     bbox_inches_restore=bbox_inches_restore)
        self.figure.draw(renderer)

        # 结束 pgfpicture 环境
        _writeln(fh, r"\end{pgfpicture}")
        _writeln(fh, r"\makeatother")
        _writeln(fh, r"\endgroup")

    # 将图形输出为 PGF，保存到文件名或文件句柄 fname_or_fh 中
    def print_pgf(self, fname_or_fh, **kwargs):
        """
        Output pgf macros for drawing the figure so it can be included and
        rendered in latex documents.
        """
        # 使用 cbook 打开文件名或文件句柄 fname_or_fh，写入 UTF-8 编码的 PGF 宏
        with cbook.open_file_cm(fname_or_fh, "w", encoding="utf-8") as file:
            # 如果文件不需要 Unicode，使用 UTF-8 编码写入
            if not cbook.file_requires_unicode(file):
                file = codecs.getwriter("utf-8")(file)
            # 调用 _print_pgf_to_fh 方法将 PGF 输出到文件
            self._print_pgf_to_fh(file, **kwargs)
    def print_pdf(self, fname_or_fh, *, metadata=None, **kwargs):
        """Use LaTeX to compile a pgf generated figure to pdf."""
        # 获取图形的尺寸（宽度和高度）
        w, h = self.figure.get_size_inches()

        # 创建用于 PDF 的元数据字典
        info_dict = _create_pdf_info_dict('pgf', metadata or {})
        # 将元数据字典转换为字符串形式，用逗号分隔
        pdfinfo = ','.join(
            _metadata_to_str(k, v) for k, v in info_dict.items())

        # 使用临时目录进行操作
        with TemporaryDirectory() as tmpdir:
            tmppath = pathlib.Path(tmpdir)
            # 调用 print_pgf 方法生成 pgf 格式的文件
            self.print_pgf(tmppath / "figure.pgf", **kwargs)
            # 将 LaTeX 代码写入 figure.tex 文件
            (tmppath / "figure.tex").write_text(
                "\n".join([
                    _DOCUMENTCLASS,
                    r"\usepackage[pdfinfo={%s}]{hyperref}" % pdfinfo,
                    r"\usepackage[papersize={%fin,%fin}, margin=0in]{geometry}"
                    % (w, h),
                    r"\usepackage{pgf}",
                    _get_preamble(),
                    r"\begin{document}",
                    r"\centering",
                    r"\input{figure.pgf}",
                    r"\end{document}",
                ]), encoding="utf-8")
            # 获取 LaTeX 命令（如 pdflatex）并执行编译
            texcommand = mpl.rcParams["pgf.texsystem"]
            cbook._check_and_log_subprocess(
                [texcommand, "-interaction=nonstopmode", "-halt-on-error",
                 "figure.tex"], _log, cwd=tmpdir)
            # 将生成的 figure.pdf 文件复制到目标文件名或文件句柄
            with (tmppath / "figure.pdf").open("rb") as orig, \
                 cbook.open_file_cm(fname_or_fh, "wb") as dest:
                shutil.copyfileobj(orig, dest)  # 复制文件内容到目标文件

    def print_png(self, fname_or_fh, **kwargs):
        """Use LaTeX to compile a pgf figure to pdf and convert it to png."""
        # 创建 PDF 到 PNG 转换器
        converter = make_pdf_to_png_converter()
        # 使用临时目录进行操作
        with TemporaryDirectory() as tmpdir:
            tmppath = pathlib.Path(tmpdir)
            pdf_path = tmppath / "figure.pdf"
            png_path = tmppath / "figure.png"
            # 调用 print_pdf 方法生成 figure.pdf 文件
            self.print_pdf(pdf_path, **kwargs)
            # 使用转换器将 PDF 转换为 PNG
            converter(pdf_path, png_path, dpi=self.figure.dpi)
            # 将生成的 figure.png 文件复制到目标文件名或文件句柄
            with png_path.open("rb") as orig, \
                 cbook.open_file_cm(fname_or_fh, "wb") as dest:
                shutil.copyfileobj(orig, dest)  # 复制文件内容到目标文件

    def get_renderer(self):
        # 返回基于 pgf 的渲染器对象
        return RendererPgf(self.figure, None)

    def draw(self):
        # 绘制图形，但不进行渲染
        self.figure.draw_without_rendering()
        # 调用父类的 draw 方法
        return super().draw()
FigureManagerPgf = FigureManagerBase

# 将 FigureManagerPgf 设置为 FigureManagerBase 的别名


@_Backend.export
class _BackendPgf(_Backend):

# 定义一个类 _BackendPgf 继承自 _Backend，并将其标记为导出类


FigureCanvas = FigureCanvasPgf

# 将 FigureCanvas 设置为 FigureCanvasPgf 的别名


class PdfPages:
    """
    A multi-page PDF file using the pgf backend

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
    """

    _UNSET = object()

    def __init__(self, filename, *, keep_empty=_UNSET, metadata=None):
        """
        Create a new PdfPages object.

        Parameters
        ----------
        filename : str or path-like
            Plots using `PdfPages.savefig` will be written to a file at this
            location. Any older file with the same name is overwritten.

        keep_empty : bool, default: True
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

            Note that some versions of LaTeX engines may ignore the 'Producer'
            key and set it to themselves.
        """
        self._output_name = filename
        self._n_figures = 0
        if keep_empty and keep_empty is not self._UNSET:
            _api.warn_deprecated("3.8", message=(
                "Keeping empty pdf files is deprecated since %(since)s and support "
                "will be removed %(removal)s."))
        self._keep_empty = keep_empty
        self._metadata = (metadata or {}).copy()
        self._info_dict = _create_pdf_info_dict('pgf', self._metadata)
        self._file = BytesIO()

# PdfPages 类的构造函数，初始化一个新的 PdfPages 对象，设置文件名、元数据、文件对象等属性


    keep_empty = _api.deprecate_privatize_attribute("3.8")

# 将 keep_empty 属性设置为私有，并标记为在 3.8 版本中已弃用的属性


    def _write_header(self, width_inches, height_inches):
        pdfinfo = ','.join(
            _metadata_to_str(k, v) for k, v in self._info_dict.items())
        latex_header = "\n".join([
            _DOCUMENTCLASS,
            r"\usepackage[pdfinfo={%s}]{hyperref}" % pdfinfo,
            r"\usepackage[papersize={%fin,%fin}, margin=0in]{geometry}"
            % (width_inches, height_inches),
            r"\usepackage{pgf}",
            _get_preamble(),
            r"\setlength{\parindent}{0pt}",
            r"\begin{document}%",
        ])
        self._file.write(latex_header.encode('utf-8'))

# 内部方法 _write_header 用于写入 PDF 头部信息，包括文档类、超链接设置、页面大小、文档前导等


    def __enter__(self):
        return self

# 实现上下文管理器方法 __enter__，返回 PdfPages 对象本身
    # 定义了对象的退出方法，处理对象的清理工作，调用 close 方法关闭对象
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # 关闭对象，完成 LaTeX 运行和将最终的 PDF 文件移动到指定的文件名
    def close(self):
        """
        Finalize this object, running LaTeX in a temporary directory
        and moving the final pdf file to *filename*.
        """
        # 写入 LaTeX 结束命令到文件
        self._file.write(rb'\end{document}\n')
        # 如果存在图形，运行 LaTeX 以生成 PDF
        if self._n_figures > 0:
            self._run_latex()
        # 如果保留空文件，警告过时并创建一个空的 PDF 文件
        elif self._keep_empty:
            _api.warn_deprecated("3.8", message=(
                "Keeping empty pdf files is deprecated since %(since)s and support "
                "will be removed %(removal)s."))
            open(self._output_name, 'wb').close()
        # 关闭文件对象
        self._file.close()

    # 运行 LaTeX 命令，将生成的 PDF 文件移动到指定路径
    def _run_latex(self):
        # 获取 LaTeX 命令
        texcommand = mpl.rcParams["pgf.texsystem"]
        # 在临时目录中创建 tex 源文件并写入内容
        with TemporaryDirectory() as tmpdir:
            tex_source = pathlib.Path(tmpdir, "pdf_pages.tex")
            tex_source.write_bytes(self._file.getvalue())
            # 执行 LaTeX 命令生成 PDF 文件，并将其移动到指定输出路径
            cbook._check_and_log_subprocess(
                [texcommand, "-interaction=nonstopmode", "-halt-on-error",
                 tex_source],
                _log, cwd=tmpdir)
            shutil.move(tex_source.with_suffix(".pdf"), self._output_name)

    # 将 Figure 对象保存为 PDF 文件的一页
    def savefig(self, figure=None, **kwargs):
        """
        Save a `.Figure` to this file as a new page.

        Any other keyword arguments are passed to `~.Figure.savefig`.

        Parameters
        ----------
        figure : `.Figure` or int, default: the active figure
            The figure, or index of the figure, that is saved to the file.
        """
        # 如果 figure 不是 Figure 对象，则获取当前活动的 Figure
        if not isinstance(figure, Figure):
            if figure is None:
                manager = Gcf.get_active()
            else:
                manager = Gcf.get_fig_manager(figure)
            if manager is None:
                raise ValueError(f"No figure {figure}")
            figure = manager.canvas.figure

        # 获取 Figure 的尺寸
        width, height = figure.get_size_inches()
        # 如果是第一次保存图形，则写入文件头部信息
        if self._n_figures == 0:
            self._write_header(width, height)
        else:
            # 写入页面尺寸信息到文件，根据不同的 LaTeX 引擎选择不同的命令
            self._file.write(
                rb'\newpage'
                rb'\ifdefined\pdfpagewidth\pdfpagewidth\else\pagewidth\fi=%fin'
                rb'\ifdefined\pdfpageheight\pdfpageheight\else\pageheight\fi=%fin'
                b'%%\n' % (width, height)
            )
        # 调用 Figure 对象的 savefig 方法保存为 pgf 格式的页面到文件
        figure.savefig(self._file, format="pgf", backend="pgf", **kwargs)
        # 更新保存的图形数量
        self._n_figures += 1

    # 返回当前多页 PDF 文件中的页面数量
    def get_pagecount(self):
        """Return the current number of pages in the multipage pdf file."""
        return self._n_figures
```