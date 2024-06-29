# `D:\src\scipysrc\matplotlib\galleries\users_explain\text\text_props.py`

```py
# 这部分代码似乎是文档或注释，采用 reStructuredText 格式，用于描述 Matplotlib 中文本的属性和布局控制方法。
# 包含了一些基本的文本属性，如 alpha（透明度）、backgroundcolor（背景颜色）、bbox（文本框）、clip_box（裁剪框）、clip_on（是否裁剪）、clip_path（裁剪路径）、color（颜色）、family（字体族）、fontproperties（字体属性）、horizontalalignment（水平对齐方式）、label（标签）、linespacing（行间距）、multialignment（多行对齐方式）、name or fontname（字体名称）、picker（拾取器）、position（位置）、rotation（旋转角度）、size or fontsize（字体大小）、style or fontstyle（字体风格）、text（文本内容）、transform（变换方式）、variant（字体变体）、verticalalignment（垂直对齐方式）、visible（可见性）、weight or fontweight（字体粗细）、x（x坐标）、y（y坐标）、zorder（堆叠顺序）。
# 文档还提到可以通过 horizontalalignment、verticalalignment 和 multialignment 参数控制文本的布局方式。
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块

import matplotlib.patches as patches  # 导入 matplotlib 的 patches 模块

# 在 axes 坐标系中创建一个矩形
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

fig = plt.figure()  # 创建一个新的图形对象
ax = fig.add_axes([0, 0, 1, 1])  # 在图形上添加一个 axes 对象，全图显示

# 创建一个矩形 patch 对象，设置其属性如下：
# - 位置和大小使用 axes 坐标系
# - 不填充
# - 不剪切
p = patches.Rectangle(
    (left, bottom), width, height,
    fill=False, transform=ax.transAxes, clip_on=False
    )

ax.add_patch(p)  # 将矩形添加到 axes 对象中显示出来

# 在 axes 坐标系中添加文本 'left top'，设置其水平对齐方式为左对齐，垂直对齐方式为顶部对齐
ax.text(left, bottom, 'left top',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)

# 在 axes 坐标系中添加文本 'left bottom'，设置其水平对齐方式为左对齐，垂直对齐方式为底部对齐
ax.text(left, bottom, 'left bottom',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes)

# 在 axes 坐标系中添加文本 'right bottom'，设置其水平对齐方式为右对齐，垂直对齐方式为底部对齐
ax.text(right, top, 'right bottom',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes)

# 在 axes 坐标系中添加文本 'right top'，设置其水平对齐方式为右对齐，垂直对齐方式为顶部对齐
ax.text(right, top, 'right top',
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax.transAxes)

# 在 axes 坐标系中添加文本 'center top'，设置其水平对齐方式为居中，垂直对齐方式为顶部对齐
ax.text(right, bottom, 'center top',
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.transAxes)

# 在 axes 坐标系中添加文本 'right center'，设置其水平对齐方式为右对齐，垂直对齐方式为居中
# 同时设置文本旋转为垂直方向
ax.text(left, 0.5*(bottom+top), 'right center',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

# 在 axes 坐标系中添加文本 'left center'，设置其水平对齐方式为左对齐，垂直对齐方式为居中
# 同时设置文本旋转为垂直方向
ax.text(left, 0.5*(bottom+top), 'left center',
        horizontalalignment='left',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

# 在 axes 坐标系中添加文本 'middle'，设置其水平对齐方式和垂直对齐方式均为居中
# 同时设置文本字体大小为20，颜色为红色
ax.text(0.5*(left+right), 0.5*(bottom+top), 'middle',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=20, color='red',
        transform=ax.transAxes)

# 在 axes 坐标系中添加文本 'centered'，设置其水平对齐方式和垂直对齐方式均为居中
# 同时设置文本旋转为垂直方向
ax.text(right, 0.5*(bottom+top), 'centered',
        horizontalalignment='center',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

# 在 axes 坐标系中添加文本 'rotated\nwith newlines'，设置其水平对齐方式和垂直对齐方式均为居中
# 同时设置文本旋转为45度，并且文本内容包含换行符
ax.text(left, top, 'rotated\nwith newlines',
        horizontalalignment='center',
        verticalalignment='center',
        rotation=45,
        transform=ax.transAxes)

ax.set_axis_off()  # 将 axes 对象的坐标轴设为不可见状态
plt.show()  # 显示绘制的图形
# for mathematical expressions, use the rcParams beginning with ``mathtext``
# (see :ref:`mathtext <mathtext-fonts>`).
#
# +---------------------+----------------------------------------------------+
# | rcParam             | usage                                              |
# +=====================+====================================================+
# | ``'font.family'``   | List of font families (installed on user's machine)|
# |                     | and/or ``{'cursive', 'fantasy', 'monospace',       |
# |                     | 'sans', 'sans serif', 'sans-serif', 'serif'}``.    |
# |                     |                                                    |
# +---------------------+----------------------------------------------------+
# |  ``'font.style'``   | The default style, ex ``'normal'``,                |
# |                     | ``'italic'``.                                      |
# |                     |                                                    |
# +---------------------+----------------------------------------------------+
# | ``'font.variant'``  | Default variant, ex ``'normal'``, ``'small-caps'`` |
# |                     | (untested)                                         |
# +---------------------+----------------------------------------------------+
# | ``'font.stretch'``  | Default stretch, ex ``'normal'``, ``'condensed'``  |
# |                     | (incomplete)                                       |
# |                     |                                                    |
# +---------------------+----------------------------------------------------+
# |  ``'font.weight'``  | Default weight.  Either string or integer          |
# |                     |                                                    |
# |                     |                                                    |
# +---------------------+----------------------------------------------------+
# |   ``'font.size'``   | Default font size in points.  Relative font sizes  |
# |                     | (``'large'``, ``'x-small'``) are computed against  |
# |                     | this size.                                         |
# +---------------------+----------------------------------------------------+
#
# Matplotlib can use font families installed on the user's computer, i.e.
# Helvetica, Times, etc. Font families can also be specified with
# generic-family aliases like (``{'cursive', 'fantasy', 'monospace',
# 'sans', 'sans serif', 'sans-serif', 'serif'}``).
#
# .. note::
#    To access the full list of available fonts: ::
#
#       matplotlib.font_manager.get_font_names()
#
# The mapping between the generic family aliases and actual font families
# (mentioned at :ref:`default rcParams <customizing>`)
# is controlled by the following rcParams:
#
#
# +------------------------------------------+--------------------------------+
# | CSS-based generic-family alias           | rcParam with mappings          |
# +==========================================+================================+
# +==========================================+================================+
# | ``'serif'``                              | ``'font.serif'``               |
# +------------------------------------------+--------------------------------+
# | ``'monospace'``                          | ``'font.monospace'``           |
# +------------------------------------------+--------------------------------+
# | ``'fantasy'``                            | ``'font.fantasy'``             |
# +------------------------------------------+--------------------------------+
# | ``'cursive'``                            | ``'font.cursive'``             |
# +------------------------------------------+--------------------------------+
# | ``{'sans', 'sans serif', 'sans-serif'}`` | ``'font.sans-serif'``          |
#
#
# 如果在 ``'font.family'`` 中出现了任何通用字体名称，我们将替换该条目
# 为相应 rcParam 映射中的所有条目。
# 例如: ::
#
#    matplotlib.rcParams['font.family'] = ['Family1', 'serif', 'Family2']
#    matplotlib.rcParams['font.serif'] = ['SerifFamily1', 'SerifFamily2']
#
#    # 这实际上转换为:
#    matplotlib.rcParams['font.family'] = ['Family1', 'SerifFamily1', 'SerifFamily2', 'Family2']
#
#
# .. _font-nonlatin:
#
# 非拉丁字形的文本
# ==========================
#
# 从 v2.0 版本开始，DejaVu 是默认字体，支持许多西方字母表的字形，但不支持
# 诸如中文、韩文或日文等其他文字系统。
#
# 若要设置默认字体以支持所需的代码点，请将字体名称前置到 ``'font.family'`` 中（推荐），或添加到
# 所需别名列表中。::
#
#    # 第一种方法
#    matplotlib.rcParams['font.family'] = ['Source Han Sans TW', 'sans-serif']
#
#    # 第二种方法
#    matplotlib.rcParams['font.family'] = ['sans-serif']
#    matplotlib.rcParams['sans-serif'] = ['Source Han Sans TW', ...]
#
# 通用字体别名列表包含与 Matplotlib 一起提供的字体（因此它们有 100% 的存在可能性），或者是
# 大多数系统中存在概率非常高的字体。
#
# 设置自定义字体系列的良好实践是在字体系列列表末尾附加一个通用字体系列作为最后的备选方案。
#
# 也可以在 :file:`.matplotlibrc` 文件中设置它::
#
#    font.family: Source Han Sans TW, Arial, sans-serif
#
# 若要基于每个艺术家设置所使用的字体，请使用文档中记录的 *name*、*fontname* 或
# *fontproperties* 关键字参数，详见 :ref:`text_props`。
#
#
# 在 Linux 中，`fc-list <https://linux.die.net/man/1/fc-list>`__ 可以是一个
# 有用的工具，用于发现字体名称；例如 ::
#
#    $ fc-list :lang=zh family
#    Noto to Sans Mono CJK TC,Noto Sans Mono CJK TC Bold
#    Noto Sans CJK TC,Noto Sans CJK TC Medium
#    Noto Sans CJK TC,Noto Sans CJK TC DemiLight
# 列出所有支持中文的字体。
# 这段代码用于注释或说明一组字体名称，这些字体都支持中文。
# 每行以字体名称开始，后面跟着字体的粗细或风格信息。
# 注释的目的是提供对这些字体的描述，特别是它们适合用于显示中文。
#
# 示例中的每行注释对应一个字体及其风格信息。
#
#    Noto Sans CJK KR,Noto Sans CJK KR Black
#    Noto Sans CJK TC,Noto Sans CJK TC Black
#    Noto Sans Mono CJK TC,Noto Sans Mono CJK TC Regular
#    Noto Sans CJK SC,Noto Sans CJK SC Light
```