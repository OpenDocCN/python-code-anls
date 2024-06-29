# `D:\src\scipysrc\matplotlib\galleries\users_explain\text\pgf.py`

```py
r"""
.. redirect-from:: /tutorials/text/pgf

.. _pgf:

************************************************************
Text rendering with XeLaTeX/LuaLaTeX via the ``pgf`` backend
************************************************************

Using the ``pgf`` backend, Matplotlib can export figures as pgf drawing
commands that can be processed with pdflatex, xelatex or lualatex. XeLaTeX and
LuaLaTeX have full Unicode support and can use any font that is installed in
the operating system, making use of advanced typographic features of OpenType,
AAT and Graphite. Pgf pictures created by ``plt.savefig('figure.pgf')``
can be embedded as raw commands in LaTeX documents. Figures can also be
directly compiled and saved to PDF with ``plt.savefig('figure.pdf')`` by
switching the backend ::

    matplotlib.use('pgf')

or by explicitly requesting the use of the ``pgf`` backend ::

    plt.savefig('figure.pdf', backend='pgf')

or by registering it for handling pdf output ::

    from matplotlib.backends.backend_pgf import FigureCanvasPgf
    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

The last method allows you to keep using regular interactive backends and to
save xelatex, lualatex or pdflatex compiled PDF files from the graphical user
interface.  Note that, in that case, the interactive display will still use the
standard interactive backends (e.g., QtAgg), and in particular use latex to
compile relevant text snippets.

Matplotlib's pgf support requires a recent LaTeX_ installation that includes
the TikZ/PGF packages (such as TeXLive_), preferably with XeLaTeX or LuaLaTeX
installed. If either pdftocairo or ghostscript is present on your system,
figures can optionally be saved to PNG images as well. The executables
for all applications must be located on your :envvar:`PATH`.

`.rcParams` that control the behavior of the pgf backend:

=================  =====================================================
Parameter          Documentation
=================  =====================================================
pgf.preamble       Lines to be included in the LaTeX preamble
pgf.rcfonts        Setup fonts from rc params using the fontspec package
pgf.texsystem      Either "xelatex" (default), "lualatex" or "pdflatex"
=================  =====================================================

.. note::

   TeX defines a set of special characters, such as::

     # $ % & ~ _ ^ \ { }

   Generally, these characters must be escaped correctly. For convenience,
   some characters (_, ^, %) are automatically escaped outside of math
   environments. Other characters are not escaped as they are commonly needed
   in actual TeX expressions. However, one can configure TeX to treat them as
   "normal" characters (known as "catcode 12" to TeX) via a custom preamble,
   such as::

     plt.rcParams["pgf.preamble"] = (
         r"\AtBeginDocument{\catcode`\&=12\catcode`\#=12}")

.. _pgf-rcfonts:


Multi-Page PDF Files
====================


"""


Explanation:

# 多行注释，介绍使用Matplotlib的pgf后端导出图形的功能和特性，以及相关的LaTeX集成方法和注意事项
# 使用 pgf 后端创建支持多页 PDF 文件，需要导入相关的类和模块
from matplotlib.backends.backend_pgf import PdfPages
import matplotlib.pyplot as plt

# 使用 PdfPages 类创建一个名为 'multipage.pdf' 的 PDF 文件，设置作者为 'Me'，并将其赋给 pdf 变量
with PdfPages('multipage.pdf', metadata={'author': 'Me'}) as pdf:
    # 创建第一个图形 fig1 和对应的坐标轴 ax1
    fig1, ax1 = plt.subplots()
    # 在 ax1 上绘制一条简单的折线图
    ax1.plot([1, 5, 3])
    # 将当前图形 fig1 保存到 pdf 文件中
    pdf.savefig(fig1)

    # 创建第二个图形 fig2 和对应的坐标轴 ax2
    fig2, ax2 = plt.subplots()
    # 在 ax2 上绘制一条简单的折线图
    ax2.plot([1, 5, 3])
    # 将当前图形 fig2 保存到 pdf 文件中
    pdf.savefig(fig2)
# 确保在 LaTeX 文档中能够实现你尝试做的事情，
# 确保你的 LaTeX 语法有效，并在必要时使用原始字符串以避免意外的转义序列。

# :rc:`pgf.preamble` 提供了很大的灵活性，但也有很多可能引起问题的地方。
# 当遇到问题时，尝试简化或禁用自定义的前言部分。

# 配置 ``unicode-math`` 环境可能有些棘手。
# 例如，TeXLive 分发提供了一组数学字体，这些字体通常不会被系统广泛安装。
# XeLaTeX 无法通过字体名称找到这些字体，因此你可能需要指定 ``\setmathfont{xits-math.otf}``
# 而不是 ``\setmathfont{XITS Math}``，或者确保这些字体对你的操作系统可用。
# 参见 `tex.stackexchange.com question`__ 获取更多详细信息。
#
# __ https://tex.stackexchange.com/q/43642/

# 如果 Matplotlib 使用的字体配置与你的 LaTeX 文档不一致，
# 那么导入图形时文本元素的对齐可能会出错。
# 如果不确定 Matplotlib 用于布局的 ``.pgf`` 文件的头部字体设置，请检查它。

# 向量图像以及 ``.pgf`` 文件如果图形中有很多对象的话可能会变得很臃肿。
# 这在图像处理或者大型散点图中可能会出现。
# 在极端情况下，这可能会导致 TeX 内存耗尽：“TeX capacity exceeded, sorry”。
# 可以配置 LaTeX 以增加生成 ``.pdf`` 图像时可用的内存量，参见 `tex.stackexchange.com <https://tex.stackexchange.com/q/7953/>`_。
# 另一种方法是使用 ``rasterized=True`` 关键字或者 ``.set_rasterized(True)`` 方法“栅格化”引起问题的图形部分，
# 如 :doc:`this example </gallery/misc/rasterization_demo>` 所示。

# 各种数学字体只有在加载相应的字体包时才会被编译和渲染。
# 特别是在希腊字母上使用 ``\mathbf{}`` 时，默认的计算机现代字体可能不包含它们，
# 因此字母不会被渲染出来。在这种情况下，应该加载 ``lmodern`` 包。

# 如果仍然需要帮助，请参阅 :ref:`reporting-problems`

# 参考链接：
# LaTeX: http://www.tug.org
# TeXLive: http://www.tug.org/texlive/
```