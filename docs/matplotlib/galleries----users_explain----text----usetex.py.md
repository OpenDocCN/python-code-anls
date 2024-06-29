# `D:\src\scipysrc\matplotlib\galleries\users_explain\text\usetex.py`

```
"""
.. redirect-from:: /tutorials/text/usetex

.. _usetex:

*************************
Text rendering with LaTeX
*************************

Matplotlib can use LaTeX to render text.  This is activated by setting
``text.usetex : True`` in your rcParams, or by setting the ``usetex`` property
to True on individual `.Text` objects.  Text handling through LaTeX is slower
than Matplotlib's very capable :ref:`mathtext <mathtext>`, but
is more flexible, since different LaTeX packages (font packages, math packages,
etc.) can be used. The results can be striking, especially when you take care
to use the same fonts in your figures as in the main document.

Matplotlib's LaTeX support requires a working LaTeX_ installation.  For
the \*Agg backends, dvipng_ is additionally required; for the PS backend,
PSfrag_, dvips_ and Ghostscript_ are additionally required.  For the PDF
and SVG backends, if LuaTeX is present, it will be used to speed up some
post-processing steps, but note that it is not used to parse the TeX string
itself (only LaTeX is supported).  The executables for these external
dependencies must all be located on your :envvar:`PATH`.

Only a small number of font families (defined by the PSNFSS_ scheme) are
supported.  They are listed here, with the corresponding LaTeX font selection
commands and LaTeX packages, which are automatically used.

=========================== =================================================
generic family              fonts
=========================== =================================================
serif (``\rmfamily``)       Computer Modern Roman, Palatino (``mathpazo``),
                            Times (``mathptmx``),  Bookman (``bookman``),
                            New Century Schoolbook (``newcent``),
                            Charter (``charter``)

sans-serif (``\sffamily``)  Computer Modern Serif, Helvetica (``helvet``),
                            Avant Garde (``avant``)

cursive (``\rmfamily``)     Zapf Chancery (``chancery``)

monospace (``\ttfamily``)   Computer Modern Typewriter, Courier (``courier``)
=========================== =================================================

The default font family (which does not require loading any LaTeX package) is
Computer Modern.  All other families are Adobe fonts.  Times and Palatino each
have their own accompanying math fonts, while the other Adobe serif fonts make
use of the Computer Modern math fonts.

To enable LaTeX and select a font, use e.g.::

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })

or equivalently, set your :ref:`matplotlibrc <customizing>` to::

    text.usetex : true
    font.family : Helvetica

It is also possible to instead set ``font.family`` to one of the generic family
names and then configure the corresponding generic family; e.g.::

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })
"""
# 这段文本是Matplotlib文档中的一部分，包含了关于使用TeX排版和PostScript选项的说明。
# 具体的示例和注意事项列出了在使用Matplotlib时可能会遇到的一些问题和解决方案。

(this was the required approach until Matplotlib 3.5).

Here is the standard example,
:doc:`/gallery/text_labels_and_annotations/tex_demo`:

.. figure:: /gallery/text_labels_and_annotations/images/sphx_glr_tex_demo_001.png
   :target: /gallery/text_labels_and_annotations/tex_demo.html
   :align: center

# 上面展示了一个标准示例的图像，该示例展示了TeX排版的效果，是Matplotlib文档中的一个图片示例。

Note that display math mode (``$$ e=mc^2 $$``) is not supported, but adding the
command ``\displaystyle``, as in the above demo, will produce the same results.

# 注意，Matplotlib不支持显示数学模式（``$$ e=mc^2 $$``），但在上述示例中添加 ``\displaystyle`` 命令可以达到相同的效果。

Non-ASCII characters (e.g. the degree sign in the y-label above) are supported
to the extent that they are supported by inputenc_.

# 支持非ASCII字符（例如上面y轴标签中的度符号），支持的程度取决于inputenc_支持的程度。

.. note::
   For consistency with the non-usetex case, Matplotlib special-cases newlines,
   so that single-newlines yield linebreaks (rather than being interpreted as
   whitespace in standard LaTeX).

   Matplotlib uses the underscore_ package so that underscores (``_``) are
   printed "as-is" in text mode (rather than causing an error as in standard
   LaTeX).  Underscores still introduce subscripts in math mode.

# 注意：
# - 为了与不使用TeX的情况保持一致，Matplotlib特殊处理换行符，使得单个换行符产生换行（而不是在标准LaTeX中被解释为空格）。
# - Matplotlib使用underscore_包，使得在文本模式下下划线（``_``）被原样打印（而不像标准LaTeX中会导致错误）。在数学模式下下划线仍然引入下标。

.. note::
   Certain characters require special escaping in TeX, such as::

     # $ % & ~ ^ \ { } \( \) \[ \]

   Therefore, these characters will behave differently depending on
   :rc:`text.usetex`.  As noted above, underscores (``_``) do not require
   escaping outside of math mode.

# 注意：
# - 在TeX中，某些字符需要特殊的转义，例如：# $ % & ~ ^ \ { } \( \) \[ \]
# - 因此，这些字符的行为取决于 :rc:`text.usetex` 的设置。如上所述，在数学模式之外，下划线（``_``）不需要转义。

PostScript options
==================

In order to produce encapsulated PostScript (EPS) files that can be embedded
in a new LaTeX document, the default behavior of Matplotlib is to distill the
output, which removes some PostScript operators used by LaTeX that are illegal
in an EPS file. This step produces results which may be unacceptable to some
users, because the text is coarsely rasterized and converted to bitmaps, which
are not scalable like standard PostScript, and the text is not searchable. One
workaround is to set :rc:`ps.distiller.res` to a higher value (perhaps 6000)
in your rc settings, which will produce larger files but may look better and
scale reasonably. A better workaround, which requires Poppler_ or Xpdf_, can
be activated by changing :rc:`ps.usedistiller` to ``xpdf``. This alternative
produces PostScript without rasterizing text, so it scales properly, can be
edited in Adobe Illustrator, and searched text in pdf documents.

# PostScript选项
# ==================

# 为了生成可以嵌入新的LaTeX文档中的封装PostScript（EPS）文件，Matplotlib的默认行为是对输出进行精炼处理，
# 这会移除一些由LaTeX使用但在EPS文件中非法的PostScript运算符。这一步骤产生的结果可能对某些用户不可接受，
# 因为文本会粗略转换成位图，而不像标准PostScript那样可伸缩，且文本不可搜索。一个解决方法是在rc设置中将
# :rc:`ps.distiller.res` 设置为较高的值（也许是6000），这将产生更大的文件但可能看起来更好，也可以合理地
# 缩放。一个更好的解决方法是，需要Poppler_或Xpdf_，可以通过将 :rc:`ps.usedistiller` 设置为 ``xpdf`` 来
# 激活。这种方法产生的PostScript文件不会对文本进行位图化处理，因此可以正常缩放，可以在Adobe Illustrator中
# 编辑，并可以在PDF文档中搜索文本。

.. _usetex-hangups:

Possible hangups
================

* On Windows, the :envvar:`PATH` environment variable may need to be modified
  to include the directories containing the latex, dvipng and ghostscript
  executables. See :ref:`environment-variables` and
  :ref:`setting-windows-environment-variables` for details.

# 可能的问题
# ===============

# * 在Windows上，可能需要修改 :envvar:`PATH` 环境变量，以包含包含latex、dvipng和ghostscript可执行文件的目录。
#   详细信息请参见 :ref:`environment-variables` 和 :ref:`setting-windows-environment-variables`。

* Using MiKTeX with Computer Modern fonts, if you get odd \*Agg and PNG
  results, go to MiKTeX/Options and update your format files

# 如果使用MiKTeX与Computer Modern字体，并且遇到奇怪的 \*Agg 和 PNG 结果，请访问MiKTeX/选项并更新您的格式文件。

* On Ubuntu and Gentoo, the base texlive install does not ship with
  the type1cm package. You may need to install some of the extra
  packages to get all the goodies that come bundled with other LaTeX
  distributions.

# 在Ubuntu和Gentoo上，基本的texlive安装不包含type1cm包。您可能需要安装一些额外的包才能获取捆绑在其他LaTeX发行版中的所有好处。
# 一些进展已经取得，使得 Matplotlib 直接使用 dvi 文件进行文本布局。
# 这使得 LaTeX 可以在 pdf 和 svg 后端以及 *Agg 和 PS 后端中用于文本布局。
# 未来，一个 LaTeX 安装可能是唯一的外部依赖。

# 遇到问题时可以尝试以下步骤：

# 尝试删除你的 .matplotlib/tex.cache 目录。如果不知道如何找到 .matplotlib 目录，
# 可参考 locating-matplotlib-config-dir。

# 确保 LaTeX、dvipng 和 ghostscript 都正常工作并在你的 PATH 环境变量中。

# 确保你尝试的操作在 LaTeX 文档中是可行的，你的 LaTeX 语法是有效的，并且
# 必要时使用原始字符串以避免意外的转义序列。

# text.latex.preamble 选项不受官方支持。这个选项提供了很大的灵活性，但也
# 可能导致各种问题。在向邮件列表报告问题之前，请禁用此选项。

# 如果仍然需要帮助，请参考 reporting-problems。
```