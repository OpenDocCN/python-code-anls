# `D:\src\scipysrc\matplotlib\lib\matplotlib\sphinxext\plot_directive.py`

```
"""
A directive for including a Matplotlib plot in a Sphinx document
================================================================

This is a Sphinx extension providing a reStructuredText directive
``.. plot::`` for including a plot in a Sphinx document.

In HTML output, ``.. plot::`` will include a .png file with a link
to a high-res .png and .pdf.  In LaTeX output, it will include a .pdf.

The plot content may be defined in one of three ways:

1. **A path to a source file** as the argument to the directive::

     .. plot:: path/to/plot.py

   When a path to a source file is given, the content of the
   directive may optionally contain a caption for the plot::

     .. plot:: path/to/plot.py

        The plot caption.

   Additionally, one may specify the name of a function to call (with
   no arguments) immediately after importing the module::

     .. plot:: path/to/plot.py plot_function1

2. Included as **inline content** to the directive::

     .. plot::

        import matplotlib.pyplot as plt
        plt.plot([1, 2, 3], [4, 5, 6])
        plt.title("A plotting exammple")

3. Using **doctest** syntax::

     .. plot::

        A plotting example:
        >>> import matplotlib.pyplot as plt
        >>> plt.plot([1, 2, 3], [4, 5, 6])

Options
-------

The ``.. plot::`` directive supports the following options:

``:format:`` : {'python', 'doctest'}
    The format of the input.  If unset, the format is auto-detected.

``:include-source:`` : bool
    Whether to display the source code. The default can be changed using
    the ``plot_include_source`` variable in :file:`conf.py` (which itself
    defaults to False).

``:show-source-link:`` : bool
    Whether to show a link to the source in HTML. The default can be
    changed using the ``plot_html_show_source_link`` variable in
    :file:`conf.py` (which itself defaults to True).

``:context:`` : bool or str
    If provided, the code will be run in the context of all previous plot
    directives for which the ``:context:`` option was specified.  This only
    applies to inline code plot directives, not those run from files. If
    the ``:context: reset`` option is specified, the context is reset
    for this and future plots, and previous figures are closed prior to
    running the code. ``:context: close-figs`` keeps the context but closes
    previous figures before running the code.

``:nofigs:`` : bool
    If specified, the code block will be run, but no figures will be
    inserted.  This is usually useful with the ``:context:`` option.

``:caption:`` : str
    If specified, the option's argument will be used as a caption for the
    figure. This overwrites the caption given in the content, when the plot
    is generated from a file.

Additionally, this directive supports all the options of the `image directive
<https://docutils.sourceforge.io/docs/ref/rst/directives.html#image>`_,
except for ``:target:`` (since plot will add its own target).  These include


# 标识文档的开始和摘要，指定Sphinx文档中包含Matplotlib图的指令

This is a Sphinx extension providing a reStructuredText directive
``.. plot::`` for including a plot in a Sphinx document.

# 在HTML输出中，``.. plot::`` 将包含一个.png文件，并带有到高分辨率.png和.pdf的链接。在LaTeX输出中，它将包含一个.pdf。

# 可以通过以下三种方式定义绘图内容：

# 1. 作为指令参数的源文件路径::

#   .. plot:: path/to/plot.py

#    当给定源文件路径时，指令的内容可以选择包含图的标题::

#      .. plot:: path/to/plot.py

#         图的标题。

#    此外，还可以指定导入模块后立即调用的函数名称（不带参数）::

#      .. plot:: path/to/plot.py plot_function1

# 2. 包含为指令的**内联内容**::

#      .. plot::

#         import matplotlib.pyplot as plt
#         plt.plot([1, 2, 3], [4, 5, 6])
#         plt.title("A plotting exammple")

# 3. 使用**doctest**语法::

#      .. plot::

#         绘图示例：
#         >>> import matplotlib.pyplot as plt
#         >>> plt.plot([1, 2, 3], [4, 5, 6])

# 选项
# -------

# ``.. plot::`` 指令支持以下选项：

# ``:format:`` : {'python', 'doctest'}
#     输入的格式。如果未设置，将自动检测格式。

# ``:include-source:`` : bool
#     是否显示源代码。默认值可以通过 :file:`conf.py` 中的 ``plot_include_source`` 变量更改（默认为False）。

# ``:show-source-link:`` : bool
#     是否在HTML中显示源链接。默认值可以通过 :file:`conf.py` 中的 ``plot_html_show_source_link`` 变量更改（默认为True）。

# ``:context:`` : bool or str
#     如果提供，则代码将在所有先前为其指定了 ``:context:`` 选项的绘图指令的上下文中运行。这仅适用于内联代码绘图指令，不适用于从文件运行的指令。如果指定了 ``:context: reset`` 选项，则上下文将在此及以后的绘图中重置，并且在运行代码之前将关闭先前的图形。 ``:context: close-figs`` 保持上下文但在运行代码之前关闭先前的图形。

# ``:nofigs:`` : bool
#     如果指定，则将运行代码块，但不会插入任何图形。这通常与 ``:context:`` 选项一起使用。

# ``:caption:`` : str
#     如果指定，则选项的参数将用作图的标题。当从文件生成图时，此选项覆盖内容中给定的标题。

# 此外，此指令支持 `image directive <https://docutils.sourceforge.io/docs/ref/rst/directives.html#image>`_ 的所有选项，除了 ``:target:`` （因为绘图将添加自己的目标）。其中包括
# plot_include_source选项控制是否包含源代码，默认为False。
plot_include_source
    # 默认为False的include-source选项的默认值。
    Default value for the include-source option (default: False).

# plot_html_show_source_link选项控制是否在HTML中显示到源文件的链接，默认为True。
plot_html_show_source_link
    # 控制是否在HTML中显示指向源文件的链接，默认为True。
    Whether to show a link to the source in HTML (default: True).

# plot_pre_code选项用于在每个绘图之前执行的代码，默认为None。
plot_pre_code
    # 每个绘图之前要执行的代码。如果为None（默认值），则默认为以下字符串：
    # 
    # import numpy as np
    # from matplotlib import pyplot as plt
    # 
    # 要在每个绘图之前执行的代码。
    Code that should be executed before each plot. If None (the default),
    it will default to a string containing::

        import numpy as np
        from matplotlib import pyplot as plt

# plot_basedir选项指定了plot::文件名相对于的基本目录，默认为None或空字符串。
plot_basedir
    # plot::指定文件名的基本目录。如果为None或为空（默认），则文件名相对于包含指令的文件所在的目录。
    Base directory, to which ``plot::`` file names are relative to.
    If None or empty (the default), file names are relative to the
    directory where the file containing the directive is.

# plot_formats选项指定要生成的文件格式，默认为['png', 'hires.png', 'pdf']。
plot_formats
    # 要生成的文件格式列表（默认为['png', 'hires.png', 'pdf']）。
    # 列表可以包含元组或字符串，如[(suffix, dpi), suffix, ...]。
    # 如果未提供DPI，默认会选择合理的默认值。
    # 从命令行通过sphinx_build传递列表时，应该使用suffix:dpi，suffix:dpi的格式。
    File formats to generate (default: ['png', 'hires.png', 'pdf']).
    List of tuples or strings::

        [(suffix, dpi), suffix, ...]

    that determine the file format and the DPI. For entries whose
    DPI was omitted, sensible defaults are chosen. When passing from
    the command line through sphinx_build the list should be passed as
    suffix:dpi,suffix:dpi, ...

# plot_html_show_formats选项控制是否在HTML中显示到文件的链接，默认为True。
plot_html_show_formats
    # 控制是否在HTML中显示到文件的链接，默认为True。
    Whether to show links to the files in HTML (default: True).

# plot_rcparams选项是一个字典，包含应用于每个绘图之前的非标准rcParams，默认为空字典{}。
plot_rcparams
    # 包含在每个绘图之前应用的非标准rcParams的字典（默认为空字典{}）。
    A dictionary containing any non-standard rcParams that should
    be applied before each plot (default: {}).

# plot_apply_rcparams选项控制是否在没有使用context选项的情况下应用rcParams，默认情况下为True。
plot_apply_rcparams
    # 默认情况下，当没有使用context选项时，会应用rcParams。如果设置了此配置选项，
    # 则会覆盖此行为，并在每个绘图之前应用rcParams。
    By default, rcParams are applied when ``:context:`` option is not used
    in a plot directive.  If set, this configuration option overrides this
    behavior and applies rcParams before each plot.

# plot_working_directory选项控制默认的工作目录，默认情况下会更改到示例所在目录。
plot_working_directory
    # 默认情况下，工作目录将更改为示例所在的目录，以便代码可以访问其数据文件（如果有）。
    # 同时，它的路径将添加到sys.path中，以便可以导入其旁边的任何辅助模块。
    # 可以使用此配置选项指定一个中心目录（也添加到sys.path中），用于存放所有代码的数据文件和辅助模块。

# plot_template选项用于提供一个定制的重构文本模板。
plot_template
    # 提供一个自定义的模板，用于准备重构文本。

# plot_srcset选项允许响应式图像分辨率的srcset图像选项列表。
plot_srcset
    # 允许响应式图像分辨率的srcset图像选项列表。列表包含多个字符串，每个字符串是乘法因子后跟"x"。
    # 例如["2.0x", "1.5x"]。"2.0x"将使用默认"png"分辨率乘以2创建一个png图像。
    # 如果指定了plot_srcset，plot指令会使用sphinxext_figmpl_directive_api（而不是通常的figure指令）
    # 在生成的中间rst文件中。
    # plot_srcset选项与singlehtml构建不兼容，会引发错误。
import contextlib
import doctest
from io import StringIO
import itertools
import os
from os.path import relpath
from pathlib import Path
import re
import shutil
import sys
import textwrap
import traceback

from docutils.parsers.rst import directives, Directive
from docutils.parsers.rst.directives.images import Image
import jinja2  # Sphinx dependency.

from sphinx.errors import ExtensionError

import matplotlib
from matplotlib.backend_bases import FigureManagerBase
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers, cbook

matplotlib.use("agg")

__version__ = 2

# -----------------------------------------------------------------------------
# Registration hook
# -----------------------------------------------------------------------------


def _option_boolean(arg):
    if not arg or not arg.strip():
        # 如果没有给定参数，假定作为一个标志使用
        return True
    elif arg.strip().lower() in ('no', '0', 'false'):
        return False
    elif arg.strip().lower() in ('yes', '1', 'true'):
        return True
    else:
        raise ValueError(f'{arg!r} unknown boolean')


def _option_context(arg):
    if arg in [None, 'reset', 'close-figs']:
        return arg
    # 参数应为 None、'reset' 或 'close-figs'
    raise ValueError("Argument should be None or 'reset' or 'close-figs'")


def _option_format(arg):
    return directives.choice(arg, ('python', 'doctest'))


def mark_plot_labels(app, document):
    """
    To make plots referenceable, we need to move the reference from the
    "htmlonly" (or "latexonly") node to the actual figure node itself.
    """
    # 为使图表可引用，我们需要将引用从“htmlonly”（或“latexonly”）节点移动到实际的图表节点
    for name, explicit in document.nametypes.items():
        if not explicit:
            continue
        labelid = document.nameids[name]
        if labelid is None:
            continue
        node = document.ids[labelid]
        if node.tagname in ('html_only', 'latex_only'):
            for n in node:
                if n.tagname == 'figure':
                    sectname = name
                    for c in n:
                        if c.tagname == 'caption':
                            sectname = c.astext()
                            break

                    node['ids'].remove(labelid)
                    node['names'].remove(name)
                    n['ids'].append(labelid)
                    n['names'].append(name)
                    document.settings.env.labels[name] = \
                        document.settings.env.docname, labelid, sectname
                    break


class PlotDirective(Directive):
    """
    The ``.. plot::`` directive, as documented in the module's docstring.
    """
    # ``.. plot::`` 指令，如模块文档字符串中所述。

    has_content = True
    # 定义指令要求的必需参数数目
    required_arguments = 0
    # 定义指令可以接受的可选参数数目
    optional_arguments = 2
    # 定义是否允许最后一个参数后有空格
    final_argument_whitespace = False
    # 定义指令可以接受的选项及其处理函数的映射关系
    option_spec = {
        'alt': directives.unchanged,  # 'alt' 选项，值保持不变
        'height': directives.length_or_unitless,  # 'height' 选项，可以是长度或者无单位的数值
        'width': directives.length_or_percentage_or_unitless,  # 'width' 选项，可以是长度、百分比或者无单位的数值
        'scale': directives.nonnegative_int,  # 'scale' 选项，必须是非负整数
        'align': Image.align,  # 'align' 选项，与图像对齐相关的处理
        'class': directives.class_option,  # 'class' 选项，处理指定的类选项
        'include-source': _option_boolean,  # 'include-source' 选项，布尔类型的选项处理函数
        'show-source-link': _option_boolean,  # 'show-source-link' 选项，布尔类型的选项处理函数
        'format': _option_format,  # 'format' 选项，处理指定的格式选项
        'context': _option_context,  # 'context' 选项，处理指定的上下文选项
        'nofigs': directives.flag,  # 'nofigs' 选项，标记选项，无参数
        'caption': directives.unchanged,  # 'caption' 选项，值保持不变
        }

    def run(self):
        """Run the plot directive."""
        try:
            # 调用 run 函数处理指令的执行，传递参数、内容、选项等信息
            return run(self.arguments, self.content, self.options,
                       self.state_machine, self.state, self.lineno)
        except Exception as e:
            # 捕获异常并抛出自定义的错误，包含异常信息
            raise self.error(str(e))
# 复制 CSS 文件到指定目录
def _copy_css_file(app, exc):
    # 检查异常是否为空并且生成的文档格式为 HTML
    if exc is None and app.builder.format == 'html':
        # 获取要复制的 CSS 文件源路径
        src = cbook._get_data_path('plot_directive/plot_directive.css')
        # 设置目标路径为输出目录下的 _static 文件夹
        dst = app.outdir / Path('_static')
        # 如果目标文件夹不存在则创建
        dst.mkdir(exist_ok=True)
        # 使用 shutil 的 copyfile 方法复制文件，避免复制源文件的权限
        shutil.copyfile(src, dst / Path('plot_directive.css'))


# 设置 Sphinx 应用程序的配置
def setup(app):
    # 将 app 参数赋值给 setup 对象的属性
    setup.app = app
    # 将 app.config 参数赋值给 setup 对象的属性
    setup.config = app.config
    # 将 app.confdir 参数赋值给 setup 对象的属性
    setup.confdir = app.confdir
    # 添加自定义指令 'plot'，使用 PlotDirective 类处理
    app.add_directive('plot', PlotDirective)
    # 添加配置值 plot_pre_code，默认为 None
    app.add_config_value('plot_pre_code', None, True)
    # 添加配置值 plot_include_source，默认为 False
    app.add_config_value('plot_include_source', False, True)
    # 添加配置值 plot_html_show_source_link，默认为 True
    app.add_config_value('plot_html_show_source_link', True, True)
    # 添加配置值 plot_formats，默认为 ['png', 'hires.png', 'pdf']
    app.add_config_value('plot_formats', ['png', 'hires.png', 'pdf'], True)
    # 添加配置值 plot_basedir，默认为 None
    app.add_config_value('plot_basedir', None, True)
    # 添加配置值 plot_html_show_formats，默认为 True
    app.add_config_value('plot_html_show_formats', True, True)
    # 添加配置值 plot_rcparams，默认为空字典
    app.add_config_value('plot_rcparams', {}, True)
    # 添加配置值 plot_apply_rcparams，默认为 False
    app.add_config_value('plot_apply_rcparams', False, True)
    # 添加配置值 plot_working_directory，默认为 None
    app.add_config_value('plot_working_directory', None, True)
    # 添加配置值 plot_template，默认为 None
    app.add_config_value('plot_template', None, True)
    # 添加配置值 plot_srcset，默认为空列表
    app.add_config_value('plot_srcset', [], True)
    # 在 doctree-read 事件上连接 mark_plot_labels 函数
    app.connect('doctree-read', mark_plot_labels)
    # 添加 CSS 文件 'plot_directive.css' 到应用程序
    app.add_css_file('plot_directive.css')
    # 在 build-finished 事件上连接 _copy_css_file 函数
    app.connect('build-finished', _copy_css_file)
    # 设置 metadata 字典，包含并行读写安全标记和 Matplotlib 版本信息
    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True,
                'version': matplotlib.__version__}
    # 返回 metadata 字典作为结果
    return metadata


# -----------------------------------------------------------------------------
# 处理 doctest
# -----------------------------------------------------------------------------


# 检查文本中是否包含 doctest
def contains_doctest(text):
    try:
        # 尝试编译文本，检查是否为有效的 Python 代码
        compile(text, '<string>', 'exec')
        return False
    except SyntaxError:
        pass
    # 使用正则表达式检查文本中是否包含 '>>>'，表示可能存在 doctest
    r = re.compile(r'^\s*>>>', re.M)
    m = r.search(text)
    return bool(m)


# 在 plt.show() 处分割代码
def _split_code_at_show(text, function_name):
    """根据 plt.show() 分割代码。"""

    # 检查文本中是否包含 doctest
    is_doctest = contains_doctest(text)
    if function_name is None:
        parts = []
        part = []
        # 遍历文本的每一行
        for line in text.split("\n"):
            # 如果不是 doctest 并且以 'plt.show(' 开头，或者是 doctest 且是单行 '>>> plt.show()'，则将该行添加到 part 中
            if ((not is_doctest and line.startswith('plt.show(')) or
                   (is_doctest and line.strip() == '>>> plt.show()')):
                part.append(line)
                # 将 part 中的内容作为一个部分，添加到 parts 中，并重置 part 为空列表
                parts.append("\n".join(part))
                part = []
            else:
                part.append(line)
        # 如果 part 中有剩余内容，则作为最后一个部分添加到 parts 中
        if "\n".join(part).strip():
            parts.append("\n".join(part))
    else:
        # 如果指定了函数名，则将整个文本作为唯一部分
        parts = [text]
    # 返回是否为 doctest 和分割后的部分列表
    return is_doctest, parts


# -----------------------------------------------------------------------------
# 模板
# -----------------------------------------------------------------------------


# 源代码模板
_SOURCECODE = """
{{ source_code }}
# HTML 渲染分支的条件语句，根据 src_name 或者 html_show_formats 的值以及 multi_image 的情况来决定是否显示下载链接和格式化图像
{% if src_name or (html_show_formats and not multi_image) %}
(
{%- if src_name -%}
:download:`Source code <{{ build_dir }}/{{ src_name }}>`
{%- endif -%}
{%- if html_show_formats and not multi_image -%}
  {%- for img in images -%}
    {%- for fmt in img.formats -%}
      {%- if src_name or not loop.first -%}, {% endif -%}
      :download:`{{ fmt }} <{{ build_dir }}/{{ img.basename }}.{{ fmt }}>`
    {%- endfor -%}
  {%- endfor -%}
{%- endif -%}
)
{% endif %}
"""

# _SOURCECODE 作为模板的一部分，用于非 HTML 渲染时显示多格式图像的模板
TEMPLATE_SRCSET = _SOURCECODE + """
   {% for img in images %}
   .. figure-mpl:: {{ build_dir }}/{{ img.basename }}.{{ default_fmt }}
      {% for option in options -%}
      {{ option }}
      {% endfor %}
      {%- if caption -%}
      {{ caption }}  {# appropriate leading whitespace added beforehand #}
      {% endif -%}
      {%- if srcset -%}
        :srcset: {{ build_dir }}/{{ img.basename }}.{{ default_fmt }}
        {%- for sr in srcset -%}
            , {{ build_dir }}/{{ img.basename }}.{{ sr }}.{{ default_fmt }} {{sr}}
        {%- endfor -%}
      {% endif %}

   {% if html_show_formats and multi_image %}
   (
    {%- for fmt in img.formats -%}
    {%- if not loop.first -%}, {% endif -%}
    :download:`{{ fmt }} <{{ build_dir }}/{{ img.basename }}.{{ fmt }}>`
    {%- endfor -%}
   )
   {% endif %}


   {% endfor %}

.. only:: not html

   {% for img in images %}
   .. figure-mpl:: {{ build_dir }}/{{ img.basename }}.*
      {% for option in options -%}
      {{ option }}
      {% endfor %}

      {{ caption }}  {# appropriate leading whitespace added beforehand #}
   {% endfor %}

"""

# _SOURCECODE 作为模板的一部分，用于 HTML 和非 HTML 渲染时显示单格式图像的模板
TEMPLATE = _SOURCECODE + """

   {% for img in images %}
   .. figure:: {{ build_dir }}/{{ img.basename }}.{{ default_fmt }}
      {% for option in options -%}
      {{ option }}
      {% endfor %}

      {% if html_show_formats and multi_image -%}
        (
        {%- for fmt in img.formats -%}
        {%- if not loop.first -%}, {% endif -%}
        :download:`{{ fmt }} <{{ build_dir }}/{{ img.basename }}.{{ fmt }}>`
        {%- endfor -%}
        )
      {%- endif -%}

      {{ caption }}  {# appropriate leading whitespace added beforehand #}
   {% endfor %}

.. only:: not html

   {% for img in images %}
   .. figure:: {{ build_dir }}/{{ img.basename }}.*
      {% for option in options -%}
      {{ option }}
      {% endfor %}

      {{ caption }}  {# appropriate leading whitespace added beforehand #}
   {% endfor %}

"""

# 异常时的 HTML 渲染模板
exception_template = """
.. only:: html

   [`source code <%(linkdir)s/%(basename)s.py>`__]

Exception occurred rendering plot.

"""

# 所有指定 :context: 选项的指令的绘图上下文
plot_context = dict()


class ImageFile:
    def __init__(self, basename, dirname):
        self.basename = basename
        self.dirname = dirname
        self.formats = []

    def filename(self, format):
        return os.path.join(self.dirname, f"{self.basename}.{format}")
    # 定义一个方法 `filenames`，该方法属于类的实例方法（self 参数表明它可以访问类的实例属性和方法）
    # 返回一个列表，列表中的每个元素是通过调用 self.filename(fmt) 方法得到的结果，fmt 是 self.formats 中的每个元素
    def filenames(self):
        return [self.filename(fmt) for fmt in self.formats]
def out_of_date(original, derived, includes=None):
    """
    Return whether *derived* is out-of-date relative to *original* or any of
    the RST files included in it using the RST include directive (*includes*).
    *derived* and *original* are full paths, and *includes* is optionally a
    list of full paths which may have been included in the *original*.
    """
    # 检查派生文件是否存在，如果不存在则认为过时
    if not os.path.exists(derived):
        return True

    # 如果 includes 未指定，设为空列表
    if includes is None:
        includes = []

    # 将要检查的文件列表包括 original 和 includes 中的所有文件
    files_to_check = [original, *includes]

    def out_of_date_one(original, derived_mtime):
        # 检查 original 文件是否存在且其修改时间晚于 derived 文件的修改时间
        return (os.path.exists(original) and
                derived_mtime < os.stat(original).st_mtime)

    # 获取 derived 文件的修改时间
    derived_mtime = os.stat(derived).st_mtime
    # 如果任何一个文件过时，则返回 True
    return any(out_of_date_one(f, derived_mtime) for f in files_to_check)


class PlotError(RuntimeError):
    pass


def _run_code(code, code_path, ns=None, function_name=None):
    """
    Import a Python module from a path, and run the function given by
    name, if function_name is not None.
    """
    
    # 将工作目录更改为示例所在的目录，以便访问其数据文件（如果有）。将其路径添加到 sys.path 中，以便导入其旁边的任何辅助模块。
    pwd = os.getcwd()
    if setup.config.plot_working_directory is not None:
        try:
            os.chdir(setup.config.plot_working_directory)
        except OSError as err:
            raise OSError(f'{err}\n`plot_working_directory` option in '
                          f'Sphinx configuration file must be a valid '
                          f'directory path') from err
        except TypeError as err:
            raise TypeError(f'{err}\n`plot_working_directory` option in '
                            f'Sphinx configuration file must be a string or '
                            f'None') from err
    elif code_path is not None:
        dirname = os.path.abspath(os.path.dirname(code_path))
        os.chdir(dirname)

    # 在上下文中设置 sys 的属性，设置 argv 为 code_path，path 为当前工作目录和 sys.path 的组合
    with cbook._setattr_cm(
            sys, argv=[code_path], path=[os.getcwd(), *sys.path]), \
            contextlib.redirect_stdout(StringIO()):
        try:
            # 如果命名空间为 None，则设为一个空字典
            if ns is None:
                ns = {}
            # 如果命名空间为空
            if not ns:
                # 如果 plot_pre_code 为 None，则执行默认的导入语句
                if setup.config.plot_pre_code is None:
                    exec('import numpy as np\n'
                         'from matplotlib import pyplot as plt\n', ns)
                # 否则执行配置中的预设代码
                else:
                    exec(str(setup.config.plot_pre_code), ns)
            # 如果代码包含 "__main__"，则将 '__name__' 设置为 '__main__'
            if "__main__" in code:
                ns['__name__'] = '__main__'

            # 屏蔽非交互式 show() 方法，以避免触发警告
            with cbook._setattr_cm(FigureManagerBase, show=lambda self: None):
                # 执行给定的代码字符串，将结果存储在命名空间中
                exec(code, ns)
                # 如果指定了函数名，则执行该函数
                if function_name is not None:
                    exec(function_name + "()", ns)

        except (Exception, SystemExit) as err:
            # 捕获所有异常和系统退出，并抛出 PlotError 异常
            raise PlotError(traceback.format_exc()) from err
        finally:
            # 最终将工作目录还原为之前的目录
            os.chdir(pwd)
    # 返回命名空间
    return ns
def render_figures(code, code_path, output_dir, output_base, context,
                   function_name, config, context_reset=False,
                   close_figs=False,
                   code_includes=None):
    """
    Run a pyplot script and save the images in *output_dir*.

    Save the images under *output_dir* with file names derived from
    *output_base*
    """

    # 如果指定了函数名，将输出基础文件名修改为包含函数名的格式
    if function_name is not None:
        output_base = f'{output_base}_{function_name}'

    # 获取绘图格式列表，根据配置文件中的设置
    formats = get_plot_formats(config)

    # 尝试确定所有图像是否已经存在
    is_doctest, code_pieces = _split_code_at_show(code, function_name)

    # 创建输出图像文件对象
    img = ImageFile(output_base, output_dir)

    # 遍历各种图像格式及其 DPI 设置
    for format, dpi in formats:
        # 如果上下文非空或者代码文件已过期，则标记所有文件未完全存在
        if context or out_of_date(code_path, img.filename(format),
                                  includes=code_includes):
            all_exists = False
            break
        img.formats.append(format)
    else:
        # 如果所有文件都已经存在，则直接返回代码和图像文件对象列表
        all_exists = True
        return [(code, [img])]

    # 如果不是所有文件都已存在，则需要进一步处理
    results = []
    # 遍历代码片段列表，同时获取索引和每个代码片段
    for i, code_piece in enumerate(code_pieces):
        images = []
        
        # 使用 itertools.count() 生成无限序列，用于生成唯一的图像文件名
        for j in itertools.count():
            # 根据代码片段数量决定图像文件名的格式
            if len(code_pieces) > 1:
                img = ImageFile('%s_%02d_%02d' % (output_base, i, j),
                                output_dir)
            else:
                img = ImageFile('%s_%02d' % (output_base, j), output_dir)
            
            # 遍历图像格式和 DPI 组合
            for fmt, dpi in formats:
                # 检查上下文或者文件是否过时，若符合条件则标记文件不存在并中断当前格式遍历
                if context or out_of_date(code_path, img.filename(fmt),
                                          includes=code_includes):
                    all_exists = False
                    break
                # 将格式添加到图像对象的格式列表中
                img.formats.append(fmt)

            # 假设一种格式存在，那么所有格式都应该存在
            if not all_exists:
                all_exists = (j > 0)
                break
            # 将生成的图像对象添加到 images 列表中
            images.append(img)
        
        # 如果存在任何一个格式文件不存在，则中断当前代码片段的处理
        if not all_exists:
            break
        # 将当前代码片段及其生成的所有图像对象添加到结果列表中
        results.append((code_piece, images))
    else:
        # 如果所有代码片段都成功生成图像文件，则标记所有文件都存在
        all_exists = True

    # 如果所有图像文件都存在，则返回结果列表
    if all_exists:
        return results

    # 如果有任何图像文件不存在，则重新构建结果列表
    results = []
    ns = plot_context if context else {}

    # 如果需要重置上下文，则清空配置并重置 plot_context
    if context_reset:
        clear_state(config.plot_rcparams)
        plot_context.clear()

    # 根据上下文设置是否需要关闭图形窗口
    close_figs = not context or close_figs

    # 再次遍历代码片段列表
    for i, code_piece in enumerate(code_pieces):

        # 如果不需要上下文或者需要应用 plot_rcparams，则清空状态
        if not context or config.plot_apply_rcparams:
            clear_state(config.plot_rcparams, close_figs)
        # 如果需要关闭图形窗口，则关闭所有图形
        elif close_figs:
            plt.close('all')

        # 运行代码片段，获取生成的图形
        _run_code(doctest.script_from_examples(code_piece) if is_doctest
                  else code_piece,
                  code_path, ns, function_name)

        images = []
        fig_managers = _pylab_helpers.Gcf.get_all_fig_managers()
        
        # 遍历所有图形管理器获取图形对象
        for j, figman in enumerate(fig_managers):
            # 根据图形和代码片段数量决定图像文件名的格式
            if len(fig_managers) == 1 and len(code_pieces) == 1:
                img = ImageFile(output_base, output_dir)
            elif len(code_pieces) == 1:
                img = ImageFile("%s_%02d" % (output_base, j), output_dir)
            else:
                img = ImageFile("%s_%02d_%02d" % (output_base, i, j),
                                output_dir)
            # 将生成的图像对象添加到 images 列表中
            images.append(img)

            # 遍历图像格式和 DPI 组合
            for fmt, dpi in formats:
                try:
                    # 尝试保存图形到文件
                    figman.canvas.figure.savefig(img.filename(fmt), dpi=dpi)
                    
                    # 如果是默认格式且需要生成 srcset，则生成多倍图像
                    if fmt == formats[0][0] and config.plot_srcset:
                        srcset = _parse_srcset(config.plot_srcset)
                        for mult, suffix in srcset.items():
                            fm = f'{suffix}.{fmt}'
                            img.formats.append(fm)
                            figman.canvas.figure.savefig(img.filename(fm),
                                                         dpi=int(dpi * mult))
                except Exception as err:
                    # 捕获异常并抛出 PlotError
                    raise PlotError(traceback.format_exc()) from err
                # 将格式添加到图像对象的格式列表中
                img.formats.append(fmt)

        # 将当前代码片段及其生成的所有图像对象添加到结果列表中
        results.append((code_piece, images))
    # 如果上下文不存在或者配置要求应用运行时参数，则执行以下操作
    if not context or config.plot_apply_rcparams:
        # 根据配置清除绘图运行时参数的状态
        clear_state(config.plot_rcparams, close=not context)
    
    # 返回计算得到的结果
    return results
    # 获取文档对象
    document = state_machine.document
    # 获取配置对象
    config = document.settings.env.config
    # 检查是否在选项中设置了 'nofigs'，返回布尔值
    nofigs = 'nofigs' in options

    # 如果配置中设置了 plot_srcset 并且当前构建器为 'singlehtml'，则抛出 ExtensionError 异常
    if config.plot_srcset and setup.app.builder.name == 'singlehtml':
        raise ExtensionError(
            'plot_srcset option not compatible with single HTML writer')

    # 获取可用的绘图格式列表
    formats = get_plot_formats(config)
    # 获取默认的绘图格式
    default_fmt = formats[0][0]

    # 设置选项中的 'include-source'，默认为配置中的 plot_include_source
    options.setdefault('include-source', config.plot_include_source)
    # 设置选项中的 'show-source-link'，默认为配置中的 plot_html_show_source_link
    options.setdefault('show-source-link', config.plot_html_show_source_link)

    # 如果选项中包含 'class'
    if 'class' in options:
        # 将 'class' 解析为字符串列表，并添加 'plot-directive' 到列表开头
        options['class'] = ['plot-directive'] + options['class']
    else:
        # 如果选项中没有 'class'，则设置 'class' 为包含 'plot-directive' 的列表
        options.setdefault('class', ['plot-directive'])
    # 是否保留上下文信息，根据选项中是否包含 'context' 来决定
    keep_context = 'context' in options
    # 如果保留上下文信息，将其保存到 context_opt 中，否则为 None
    context_opt = None if not keep_context else options['context']

    # 获取当前文档的源文件路径
    rst_file = document.attributes['source']
    # 获取当前文档的源文件所在目录路径
    rst_dir = os.path.dirname(rst_file)

    # 如果 arguments 非空
    if len(arguments):
        # 如果配置中没有设置 plot_basedir
        if not config.plot_basedir:
            # 根据构建器的源目录和指令的首个参数创建源文件名
            source_file_name = os.path.join(setup.app.builder.srcdir,
                                            directives.uri(arguments[0]))
        else:
            # 否则，根据配置中的 plot_basedir 和指令的首个参数创建源文件名
            source_file_name = os.path.join(setup.confdir, config.plot_basedir,
                                            directives.uri(arguments[0]))
        # 如果有内容，则将其作为标题
        caption = '\n'.join(content)

        # 确保标题的使用不会引起歧义
        if "caption" in options:
            if caption:
                # 如果同时在内容和选项中指定了标题，则抛出 ValueError 异常
                raise ValueError(
                    'Caption specified in both content and options.'
                    ' Please remove ambiguity.'
                )
            # 使用选项中的 caption
            caption = options["caption"]

        # 如果参数的长度为 2，将第二个参数作为函数名，否则为 None
        if len(arguments) == 2:
            function_name = arguments[1]
        else:
            function_name = None

        # 读取源文件的内容
        code = Path(source_file_name).read_text(encoding='utf-8')
        # 获取源文件的基本名称
        output_base = os.path.basename(source_file_name)
    else:
        # 否则，源文件名为当前 rst 文件
        source_file_name = rst_file
        # 从内容中生成代码文本，并进行缩进处理
        code = textwrap.dedent("\n".join(map(str, content)))
        # 计数器用于生成唯一的文件名
        counter = document.attributes.get('_plot_counter', 0) + 1
        document.attributes['_plot_counter'] = counter
        # 获取源文件的基本名称和扩展名
        base, ext = os.path.splitext(os.path.basename(source_file_name))
        # 生成输出文件的基本名称，包括计数器
        output_base = '%s-%d.py' % (base, counter)
        function_name = None
        # 获取选项中的 caption，如果不存在则为空字符串
        caption = options.get('caption', '')

    # 获取输出文件的基本名称和扩展名
    base, source_ext = os.path.splitext(output_base)
    # 如果扩展名为 '.py', '.rst', '.txt' 中的一种，则保持输出基本名称不变
    if source_ext in ('.py', '.rst', '.txt'):
        output_base = base
    else:
        source_ext = ''

    # 确保 LaTeX 的 includegraphics 不会因为文件名中包含 '.' 而出错
    output_base = output_base.replace('.', '-')

    # 判断是否在 doctest 格式中使用？
    # 检查代码中是否包含文档测试
    is_doctest = contains_doctest(code)

    # 检查是否需要特定格式，若为 'python' 则禁用文档测试，否则启用
    if 'format' in options:
        if options['format'] == 'python':
            is_doctest = False
        else:
            is_doctest = True

    # 确定输出目录的名称片段
    # 计算源文件相对路径，相对于 setup.confdir
    source_rel_name = relpath(source_file_name, setup.confdir)
    # 获取源文件相对目录，并去除开头的目录分隔符
    source_rel_dir = os.path.dirname(source_rel_name).lstrip(os.path.sep)

    # build_dir: 用于存放输出文件（临时）
    # 构建输出目录的完整路径，位于 setup.app.doctreedir 的子目录 'plot_directive' 中
    build_dir = os.path.join(os.path.dirname(setup.app.doctreedir),
                             'plot_directive',
                             source_rel_dir)

    # 规范化路径，解决路径中的 '..'，同时处理路径分隔符的变化
    build_dir = os.path.normpath(build_dir)
    # 创建目录，如果不存在则创建，exist_ok=True 表示如果已存在也不报错
    os.makedirs(build_dir, exist_ok=True)

    # 如何从 RST 文件中链接到文件
    try:
        # 计算 build_dir 相对于 rst_dir 的相对路径，将路径分隔符替换为 '/'
        build_dir_link = relpath(build_dir, rst_dir).replace(os.path.sep, '/')
    except ValueError:
        # 在 Windows 上，如果路径和起始点不在同一驱动器上，relpath 可能会引发 ValueError
        build_dir_link = build_dir

    # 获取包含在 RST 文件中的包含文件列表，以便在包含的文件中的绘图发生变化时更新输出
    try:
        # 从文档的 include_log 属性中获取包含的文件列表
        source_file_includes = [os.path.join(os.getcwd(), t[0])
                                for t in state.document.include_log]
    except AttributeError:
        # 在 docutils < 0.17 版本中，document.include_log 属性不存在，需要检查 state_machine
        possible_sources = {os.path.join(setup.confdir, t[0])
                            for t in state_machine.input_lines.items}
        # 筛选出实际存在的文件
        source_file_includes = [f for f in possible_sources
                                if os.path.isfile(f)]
    
    # 从包含的文件列表中移除当前源文件本身
    try:
        source_file_includes.remove(source_file_name)
    except ValueError:
        # 如果源文件名不在列表中则忽略
        pass

    # 如果需要显示源代码链接，则保存脚本文件
    if options['show-source-link']:
        # 将代码转换为文档测试脚本（如果当前文件是 RST 文件且包含文档测试），否则直接使用原始代码
        Path(build_dir, output_base + source_ext).write_text(
            doctest.script_from_examples(code)
            if source_file_name == rst_file and is_doctest
            else code,
            encoding='utf-8')

    # 生成图形
    # 尝试渲染图表并生成结果
    try:
        results = render_figures(code=code,
                                 code_path=source_file_name,
                                 output_dir=build_dir,
                                 output_base=output_base,
                                 context=keep_context,
                                 function_name=function_name,
                                 config=config,
                                 context_reset=context_opt == 'reset',
                                 close_figs=context_opt == 'close-figs',
                                 code_includes=source_file_includes)
        # 初始化错误列表
        errors = []
    # 捕获图表渲染过程中的异常
    except PlotError as err:
        # 获取报告器对象
        reporter = state.memo.reporter
        # 创建系统消息对象，报告异常信息和位置
        sm = reporter.system_message(
            2, "Exception occurred in plotting {}\n from {}:\n{}".format(
                output_base, source_file_name, err),
            line=lineno)
        # 将结果设置为仅包含错误的列表
        results = [(code, [])]
        # 将系统消息添加到错误列表中
        errors = [sm]

    # 根据配置和标题生成合适的标题文本
    if caption and config.plot_srcset:
        caption = f':caption: {caption}'
    elif caption:
        # 如果存在标题但不使用源集，则格式化标题以适当缩进
        caption = '\n' + '\n'.join('      ' + line.strip()
                                   for line in caption.split('\n'))

    # 生成输出的reStructuredText
    total_lines = []
    for j, (code_piece, images) in enumerate(results):
        # 如果选项中包含源码，则准备源码块
        if options['include-source']:
            if is_doctest:
                lines = ['', *code_piece.splitlines()]
            else:
                lines = ['.. code-block:: python', '',
                         *textwrap.indent(code_piece, '    ').splitlines()]
            source_code = "\n".join(lines)
        else:
            source_code = ""

        # 如果禁用图像，则清空图像列表
        if nofigs:
            images = []

        # 构建选项列表
        opts = [
            f':{key}: {val}' for key, val in options.items()
            if key in ('alt', 'height', 'width', 'scale', 'align', 'class')]

        # 第一次循环时（j == 0）根据选项显示源链接
        if j == 0 and options['show-source-link']:
            src_name = output_base + source_ext
        else:
            src_name = None

        # 根据配置生成模板
        if config.plot_srcset:
            srcset = [*_parse_srcset(config.plot_srcset).values()]
            template = TEMPLATE_SRCSET
        else:
            srcset = None
            template = TEMPLATE

        # 使用模板渲染结果
        result = jinja2.Template(config.plot_template or template).render(
            default_fmt=default_fmt,
            build_dir=build_dir_link,
            src_name=src_name,
            multi_image=len(images) > 1,
            options=opts,
            srcset=srcset,
            images=images,
            source_code=source_code,
            html_show_formats=config.plot_html_show_formats and len(images),
            caption=caption)
        # 将渲染后的结果添加到总行列表中
        total_lines.extend(result.split("\n"))
        total_lines.extend("\n")

    # 如果生成了输出行，则将其插入到状态机中
    if total_lines:
        state_machine.insert_input(total_lines, source=source_file_name)

    # 返回错误列表
    return errors
```