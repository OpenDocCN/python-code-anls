# `D:\src\scipysrc\sympy\sympy\printing\preview.py`

```
import os  # 导入标准库 os，用于操作系统相关功能
from os.path import join  # 从 os.path 中导入 join 函数，用于路径拼接
import shutil  # 导入 shutil 库，用于高级文件操作
import tempfile  # 导入 tempfile 库，用于创建临时文件和目录

try:
    from subprocess import STDOUT, CalledProcessError, check_output  # 尝试导入子进程相关模块
except ImportError:
    pass  # 如果 ImportError，则忽略

from sympy.utilities.decorator import doctest_depends_on  # 导入 sympy 库中的装饰器相关模块
from sympy.utilities.misc import debug  # 导入 sympy 库中的调试相关模块
from .latex import latex  # 导入当前目录下的 latex 模块

__doctest_requires__ = {('preview',): ['pyglet']}  # 定义一个文档测试依赖的字典，指定了 'preview' 需要 'pyglet'

def _check_output_no_window(*args, **kwargs):
    # 避免在 Windows 下显示 cmd.exe 窗口
    if os.name == 'nt':
        creation_flag = 0x08000000  # CREATE_NO_WINDOW
    else:
        creation_flag = 0  # 默认值
    return check_output(*args, creationflags=creation_flag, **kwargs)  # 调用 check_output 函数，传递参数和标志位

def system_default_viewer(fname, fmt):
    """ 使用系统默认的查看器打开文件。

    实际上，Python 无法确定系统查看器何时完成。因此，我们确保传递的文件不会在此期间被删除，
    此函数也不会尝试阻塞。

    Args:
        fname (str): 要打开的文件名
        fmt (str): 文件格式

    Raises:
        ImportError: 如果无法导入相关模块
    """
    # 将文件复制到一个不会被删除的新临时文件中
    with tempfile.NamedTemporaryFile(prefix='sympy-preview-',
                                     suffix=os.path.splitext(fname)[1],
                                     delete=False) as temp_f:
        with open(fname, 'rb') as f:
            shutil.copyfileobj(f, temp_f)

    import platform
    if platform.system() == 'Darwin':
        import subprocess
        subprocess.call(('open', temp_f.name))  # macOS 下使用 'open' 命令打开文件
    elif platform.system() == 'Windows':
        os.startfile(temp_f.name)  # Windows 下使用 os.startfile 打开文件
    else:
        import subprocess
        subprocess.call(('xdg-open', temp_f.name))  # 其他系统下使用 'xdg-open' 命令打开文件

def pyglet_viewer(fname, fmt):
    """ 使用 pyglet 库预览图片文件。

    Args:
        fname (str): 要预览的文件名
        fmt (str): 文件格式

    Raises:
        ImportError: 如果无法导入 pyglet 库
        ValueError: 如果无法解码指定格式的文件
    """
    try:
        from pyglet import window, image, gl  # 导入 pyglet 相关模块
        from pyglet.window import key  # 导入 pyglet 窗口模块的键盘键码
        from pyglet.image.codecs import ImageDecodeException  # 导入 pyglet 图像解码异常
    except ImportError:
        raise ImportError("pyglet is required for preview.\n visit https://pyglet.org/")  # 如果导入失败，抛出 ImportError

    try:
        img = image.load(fname)  # 尝试加载指定文件的图像数据
    except ImageDecodeException:
        raise ValueError("pyglet preview does not work for '{}' files.".format(fmt))  # 如果无法解码，抛出 ValueError

    offset = 25  # 设置偏移量

    config = gl.Config(double_buffer=False)  # 创建 OpenGL 配置对象
    win = window.Window(
        width=img.width + 2*offset,  # 设置窗口宽度
        height=img.height + 2*offset,  # 设置窗口高度
        caption="SymPy",  # 窗口标题
        resizable=False,  # 窗口不可调整大小
        config=config  # 使用上面创建的配置
    )

    win.set_vsync(False)  # 禁用垂直同步

    def on_close():
        """ 窗口关闭时的回调函数 """
        win.has_exit = True

    win.on_close = on_close  # 设置窗口关闭时的回调函数

    def on_key_press(symbol, modifiers):
        """ 键盘按键按下时的回调函数 """
        if symbol in [key.Q, key.ESCAPE]:
            on_close()  # 如果按下 Q 键或 ESC 键，关闭窗口

    win.on_key_press = on_key_press  # 设置键盘按键按下时的回调函数

    def on_expose():
        """ 窗口暴露（显示内容）时的回调函数 """
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)  # 设置清除颜色为白色
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)  # 清除颜色缓冲区

        img.blit(
            (win.width - img.width) / 2,  # 计算图像绘制位置的 x 坐标
            (win.height - img.height) / 2  # 计算图像绘制位置的 y 坐标
        )

    win.on_expose = on_expose  # 设置窗口暴露时的回调函数

    while not win.has_exit:
        win.dispatch_events()  # 处理所有事件
        win.flip()  # 更新窗口显示
    # 捕获键盘中断异常，通常用户按下 Ctrl+C 时会触发该异常
    except KeyboardInterrupt:
        # 如果捕获到键盘中断异常，则不做任何处理，直接跳过
        pass

    # 关闭窗口对象
    win.close()
# 定义一个函数，生成渲染给定表达式的 LaTeX 文档字符串
def _get_latex_main(expr, *, preamble=None, packages=(), extra_preamble=None,
                    euler=True, fontsize=None, **latex_settings):
    """
    Generate string of a LaTeX document rendering ``expr``.
    """

    # 如果未指定 LaTeX 前文，则默认使用以下包并设置字体大小
    if preamble is None:
        actual_packages = packages + ("amsmath", "amsfonts")
        # 根据参数决定是否包含 Euler 字体支持
        if euler:
            actual_packages += ("euler",)
        # 构建 LaTeX 的包含命令字符串
        package_includes = "\n" + "\n".join(["\\usepackage{%s}" % p
                                             for p in actual_packages])
        # 如果有额外的 LaTeX 前文，加入到包含命令字符串中
        if extra_preamble:
            package_includes += extra_preamble

        # 如果未指定字体大小，默认为 12pt；如果是整数，转换为字符串格式
        if not fontsize:
            fontsize = "12pt"
        elif isinstance(fontsize, int):
            fontsize = "{}pt".format(fontsize)
        # 构建完整的 LaTeX 前文
        preamble = r"""\documentclass[varwidth,%s]{standalone}
%s

\begin{document}
""" % (fontsize, package_includes)
    else:
        # 如果已经指定了自定义的 LaTeX 前文，不允许同时设置 packages 或 extra_preamble
        if packages or extra_preamble:
            raise ValueError("The \"packages\" or \"extra_preamble\" keywords"
                             "must not be set if a "
                             "custom LaTeX preamble was specified")

    # 如果 expr 是字符串，则直接使用它作为 LaTeX 字符串
    if isinstance(expr, str):
        latex_string = expr
    else:
        # 否则，将表达式转换为 LaTeX 字符串
        latex_string = ('$\\displaystyle ' +
                        latex(expr, mode='plain', **latex_settings) +
                        '$')

    # 返回完整的 LaTeX 文档字符串，包括前文、表达式内容和结束命令
    return preamble + '\n' + latex_string + '\n\n' + r"\end{document}"


``````python
@doctest_depends_on(exe=('latex', 'dvipng'), modules=('pyglet',),
            disable_viewers=('evince', 'gimp', 'superior-dvi-viewer'))
def preview(expr, output='png', viewer=None, euler=True, packages=(),
            filename=None, outputbuffer=None, preamble=None, dvioptions=None,
            outputTexFile=None, extra_preamble=None, fontsize=None,
            **latex_settings):
    r"""
    View expression or LaTeX markup in PNG, DVI, PostScript or PDF form.

    If the expr argument is an expression, it will be exported to LaTeX and
    then compiled using the available TeX distribution.  The first argument,
    'expr', may also be a LaTeX string.  The function will then run the
    appropriate viewer for the given output format or use the user defined
    one. By default png output is generated.

    By default pretty Euler fonts are used for typesetting (they were used to
    typeset the well known "Concrete Mathematics" book). For that to work, you
    need the 'eulervm.sty' LaTeX style (in Debian/Ubuntu, install the
    texlive-fonts-extra package). If you prefer default AMS fonts or your
    system lacks 'eulervm' LaTeX package then unset the 'euler' keyword
    argument.

    To use viewer auto-detection, lets say for 'png' output, issue

    >>> from sympy import symbols, preview, Symbol
    >>> x, y = symbols("x,y")

    >>> preview(x + y, output='png')

    This will choose 'pyglet' by default. To select a different one, do

    >>> preview(x + y, output='png', viewer='gimp')

    The 'png' format is considered special. For all other formats the rules
    # 如果输出格式不是 'dvi'，可以通过 'dvioptions' 参数设置 'dvi'+输出转换工具的命令行选项，这些选项必须以字符串列表的形式提供（参见 subprocess.Popen）
    If the value of 'output' is different from 'dvi' then command line
    options can be set ('dvioptions' argument) for the execution of the
    'dvi'+output conversion tool. These options have to be in the form of a
    list of strings (see ``subprocess.Popen``).
    
    # 附加的关键字参数将传递给 :func:`~sympy.printing.latex.latex` 调用，例如 ``symbol_names`` 标志
    Additional keyword args will be passed to the :func:`~sympy.printing.latex.latex` call,
    e.g., the ``symbol_names`` flag.
    
    # 通过将所需的文件名传递给 'outputTexFile' 关键字参数，可以将生成的 TeX 文件写入文件。要将 TeX 代码写入名为 "sample.tex" 的文件，并运行默认的 png 查看器显示生成的位图，请执行以下操作
    For post-processing the generated TeX File can be written to a file by
    passing the desired filename to the 'outputTexFile' keyword
    argument. To write the TeX code to a file named
    ``"sample.tex"`` and run the default png viewer to display the resulting
    bitmap, do
    >>> preview(x + y, outputTexFile="sample.tex")


    """
    # 如果没有指定查看器，并且输出格式是 "png"，尝试导入 pyglet 库
    if viewer is None and output == "png":
        try:
            import pyglet  # noqa: F401
        except ImportError:
            pass
        else:
            viewer = pyglet_viewer

    # 查找已知应用程序
    if viewer is None:
        # 根据外观美观程度排序，优先级由高到低
        candidates = {
            "dvi": ["evince", "okular", "kdvi", "xdvi"],
            "ps": ["evince", "okular", "gsview", "gv"],
            "pdf": ["evince", "okular", "kpdf", "acroread", "xpdf", "gv"],
        }

        # 遍历指定输出格式的候选查看器列表
        for candidate in candidates.get(output, []):
            # 查找候选查看器的可执行文件路径
            path = shutil.which(candidate)
            if path is not None:
                viewer = path
                break

    # 否则，使用系统默认的文件关联查看器
    if viewer is None:
        viewer = system_default_viewer

    # 如果查看器是 "file"，则需要指定文件名
    if viewer == "file":
        if filename is None:
            raise ValueError("filename has to be specified if viewer=\"file\"")
    # 如果查看器是 "BytesIO"，则输出缓冲区必须是一个兼容 BytesIO 的对象
    elif viewer == "BytesIO":
        if outputbuffer is None:
            raise ValueError("outputbuffer has to be a BytesIO "
                             "compatible object if viewer=\"BytesIO\"")
    # 否则，查看器应当是一个可调用对象或者是系统中已知的可执行文件
    elif not callable(viewer) and not shutil.which(viewer):
        raise OSError("Unrecognized viewer: %s" % viewer)

    # 获取 Latex 主表达式的主体内容，并生成 Latex 代码
    latex_main = _get_latex_main(expr, preamble=preamble, packages=packages,
                                 euler=euler, extra_preamble=extra_preamble,
                                 fontsize=fontsize, **latex_settings)

    # 输出调试信息，显示生成的 Latex 代码内容
    debug("Latex code:")
    debug(latex_main)
```