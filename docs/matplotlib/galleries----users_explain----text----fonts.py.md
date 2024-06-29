# `D:\src\scipysrc\matplotlib\galleries\users_explain\text\fonts.py`

```
r"""
.. redirect-from:: /users/fonts
.. redirect-from:: /users/explain/fonts

.. _fonts:

Fonts in Matplotlib
===================

Matplotlib needs fonts to work with its text engine, some of which are shipped
alongside the installation.  The default font is `DejaVu Sans
<https://dejavu-fonts.github.io>`_ which covers most European writing systems.
However, users can configure the default fonts, and provide their own custom
fonts.  See :ref:`Customizing text properties <text_props>` for
details and :ref:`font-nonlatin` in particular for glyphs not supported by
DejaVu Sans.

Matplotlib also provides an option to offload text rendering to a TeX engine
(``usetex=True``), see :ref:`Text rendering with LaTeX
<usetex>`.

Fonts in PDF and PostScript
---------------------------

Fonts have a long (and sometimes incompatible) history in computing, leading to
different platforms supporting different types of fonts.  In practice,
Matplotlib supports three font specifications (in addition to pdf 'core fonts',
which are explained later in the guide):

.. list-table:: Type of Fonts
   :header-rows: 1

   * - Type 1 (PDF)
     - Type 3 (PDF/PS)
     - TrueType (PDF)
   * - One of the oldest types, introduced by Adobe
     - Similar to Type 1 in terms of introduction
     - Newer than previous types, used commonly today, introduced by Apple
   * - Restricted subset of PostScript, charstrings are in bytecode
     - Full PostScript language, allows embedding arbitrary code
       (in theory, even render fractals when rasterizing!)
     - Include a virtual machine that can execute code!
   * - These fonts support font hinting
     - Do not support font hinting
     - Hinting supported (virtual machine processes the "hints")
   * - Non-subsetted through Matplotlib
     - Subsetted via external module ttconv
     - Subsetted via external module
       `fontTools <https://github.com/fonttools/fonttools>`__

.. note::

   Adobe disabled__ support for authoring with Type 1 fonts in January 2023.

   __ https://helpx.adobe.com/fonts/kb/postscript-type-1-fonts-end-of-support.html

Other font specifications which Matplotlib supports:

- Type 42 fonts (PS):

  - PostScript wrapper around TrueType fonts
  - 42 is the `Answer to Life, the Universe, and Everything!
    <https://en.wikipedia.org/wiki/Answer_to_Life,_the_Universe,_and_Everything>`_
  - Matplotlib uses the external library
    `fontTools <https://github.com/fonttools/fonttools>`__ to subset these types of
    fonts

- OpenType fonts:

  - OpenType is a new standard for digital type fonts, developed jointly by
    Adobe and Microsoft
  - Generally contain a much larger character set!
  - Limited support with Matplotlib

Font subsetting
^^^^^^^^^^^^^^^

The PDF and PostScript formats support embedding fonts in files, allowing the
display program to correctly render the text, independent of what fonts are
installed on the viewer's computer and without the need to pre-rasterize the text.

"""
`
# 这段文档解释了在 Matplotlib 中如何处理字体，以确保图像输出的文本不失真
This ensures that if the output is zoomed or resized the text does not become
pixelated.  However, embedding full fonts in the file can lead to large output
files, particularly with fonts with many glyphs such as those that support CJK
(Chinese/Japanese/Korean).

# 解决方法是仅嵌入文档中实际使用的字体字形，称为字体子集，避免文件过大
The solution to this problem is to subset the fonts used in the document and
only embed the glyphs actually used.  This gets both vector text and small
files sizes.  Computing the subset of the font required and writing the new
(reduced) font are both complex problem and thus Matplotlib relies on
`fontTools <https://fonttools.readthedocs.io/en/latest/>`__ and a vendored fork
of ttconv.

# 当前支持 Type 3、Type 42 和 TrueType 字体子集，Type 1 字体不支持
Currently Type 3, Type 42, and TrueType fonts are subsetted.  Type 1 fonts are not.

# 介绍了 PDF 和 PostScript 文档中的核心字体
Core Fonts
^^^^^^^^^^

# 除了嵌入字体之外，PostScript 和 PDF 规范中有 14 个核心字体，符合规范的查看器必须确保这些字体可用
In addition to the ability to embed fonts, as part of the `PostScript
<https://en.wikipedia.org/wiki/PostScript_fonts#Core_Font_Set>`_ and `PDF
specification
<https://docs.oracle.com/cd/E96927_01/TSG/FAQ/What%20are%20the%2014%20base%20fonts%20distributed%20with%20Acroba.html>`_
there are 14 Core Fonts that compliant viewers must ensure are available.  If
you restrict your document to only these fonts you do not have to embed any
font information in the document but still get vector text.

# 如果只使用这些字体，可以生成非常轻量级的文档
This is especially helpful to generate *really lightweight* documents::

    # 激活 PDF 后端的核心字体支持
    plt.rcParams["pdf.use14corefonts"] = True
    # 激活 PostScript 后端的字体支持
    plt.rcParams["ps.useafm"] = True

    chars = "AFM ftw!"
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, chars)

    # 保存为 PDF 格式
    fig.savefig("AFM_PDF.pdf", format="pdf")
    # 保存为 PostScript 格式
    fig.savefig("AFM_PS.ps", format="ps")

# 介绍 SVG 格式中的字体处理方式
Fonts in SVG
------------

# 文本输出到 SVG 的方式可以通过 :rc:`svg.fonttype` 进行控制：
Text can output to SVG in two ways controlled by :rc:`svg.fonttype`:

# 作为路径（``'path'``）在 SVG 中输出
- as a path (``'path'``) in the SVG
# 作为 SVG 元素中的字符串，带有字体样式（``'none'``）
- as string in the SVG with font styling on the element (``'none'``)

# 当以 ``'path'`` 格式保存时，Matplotlib 会计算使用的字形路径，并写入输出。这样 SVG 在所有计算机上看起来都一样，不受安装字体的影响，缺点是文本不可编辑
When saving via ``'path'`` Matplotlib will compute the path of the glyphs used
as vector paths and write those to the output.  The advantage of doing so is
that the SVG will look the same on all computers independent of what fonts are
installed.  However the text will not be editable after the fact.

# 与此不同，使用 ``'none'`` 格式保存将生成更小的文件，文本直接显示在标记中，显示效果可能因 SVG 查看器和可用字体而异
In contrast, saving with ``'none'`` will result in smaller files and the
text will appear directly in the markup.  However, the appearance may vary
based on the SVG viewer and what fonts are available.

# 介绍 Agg 渲染器的字体处理方式，依赖 FreeType
Fonts in Agg
------------

# 要输出图像文件到 Agg 格式，Matplotlib 依赖于 `FreeType <https://www.freetype.org/>`_
To output text to raster formats via Agg, Matplotlib relies on `FreeType
<https://www.freetype.org/>`_.  Because the exact rendering of the glyphs
changes between FreeType versions we pin to a specific version for our image
comparison tests.

# 介绍 Matplotlib 如何选择字体的过程，包括创建 `.FontProperties` 对象
How Matplotlib selects fonts
----------------------------

# 在 Matplotlib 中，使用字体的过程包括三个步骤：
Internally, using a font in Matplotlib is a three step process:

# 1. 创建一个 `.FontProperties` 对象（显式或隐式创建）
1. a `.FontProperties` object is created (explicitly or implicitly)
# 2. 根据 `.FontProperties` 对象，使用 `.FontManager` 上的方法选择最接近的“最佳”字体，
#    Matplotlib 将会使用这些字体（SVG 模式除外的所有情况）来渲染文本。
# 3. Python 中的字体对象代理被后端代码用来渲染文本，具体细节依赖于后端通过 `.font_manager.get_font` 实现。

# 选择“最佳”字体的算法是 CSS1 规范修改版的一部分，这个算法主要参考了 Web 浏览器使用的算法。
# 算法考虑了字体系列名称（例如 "Arial", "Noto Sans CJK", "Hack" 等）、大小、样式和粗细。
# 除了直接映射到字体的系列名称外，还有五个“通用字体系列名称”（serif、monospace、fantasy、cursive 和 sans-serif），
# 它们会被内部映射到一组字体中的任意一个。

# 目前公共 API 中用于步骤 2 的方法是 `.FontManager.findfont`（全局 `.FontManager` 实例上的方法别名为 `.font_manager.findfont`），
# 它只能找到一个单一字体，并返回文件系统中字体的绝对路径。

# 字体回退
# ---------
# 由于没有覆盖整个 Unicode 空间的字体，因此用户可能需要混合使用无法从单一字体满足的字形集。
# 尽管在 Figure 中使用多个字体已经可能，分别应用于不同的 `.Text` 实例，但以前不可能在同一个 `.Text` 实例中使用多个字体（就像 Web 浏览器那样）。
# 自 Matplotlib 3.6 版本起，Agg、SVG、PDF 和 PS 后端会在单个 `.Text` 实例中通过多个字体“回退”：

# .. plot::
#    :include-source:
#    :caption: 使用 2 种字体渲染的字符串 "There are 几个汉字 in between!"

#    fig, ax = plt.subplots()
#    ax.text(
#        .5, .5, "There are 几个汉字 in between!",
#        family=['DejaVu Sans', 'Noto Sans CJK JP', 'Noto Sans TC'],
#        ha='center'
#    )

# 内部实现是通过将 `.FontProperties` 对象上的“字体系列”设置为字体系列列表来完成的。
# 一个（当前是私有的）API 提取所有找到的字体路径列表，然后构建一个单一的 `.ft2font.FT2Font` 对象，该对象了解所有字体。
# 每个字符串的字形都使用列表中第一个包含该字形的字体进行渲染。

# 这项工作的主要部分是由 Aitik Gupta 在 Google 2021 年夏季代码大会（Google Summer of Code）的支持下完成。
```