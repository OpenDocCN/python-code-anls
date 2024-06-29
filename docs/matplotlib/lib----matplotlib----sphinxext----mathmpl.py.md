# `D:\src\scipysrc\matplotlib\lib\matplotlib\sphinxext\mathmpl.py`

```py
r"""
A role and directive to display mathtext in Sphinx
==================================================

The ``mathmpl`` Sphinx extension creates a mathtext image in Matplotlib and
shows it in html output. Thus, it is a true and faithful representation of what
you will see if you pass a given LaTeX string to Matplotlib (see
:ref:`mathtext`).

.. warning::
    In most cases, you will likely want to use one of `Sphinx's builtin Math
    extensions
    <https://www.sphinx-doc.org/en/master/usage/extensions/math.html>`__
    instead of this one. The builtin Sphinx math directive uses MathJax to
    render mathematical expressions, and addresses accessibility concerns that
    ``mathmpl`` doesn't address.

Mathtext may be included in two ways:

1. Inline, using the role::

     This text uses inline math: :mathmpl:`\alpha > \beta`.

   which produces:

     This text uses inline math: :mathmpl:`\alpha > \beta`.

2. Standalone, using the directive::

     Here is some standalone math:

     .. mathmpl::

         \alpha > \beta

   which produces:

     Here is some standalone math:

     .. mathmpl::

         \alpha > \beta

Options
-------

The ``mathmpl`` role and directive both support the following options:

fontset : str, default: 'cm'
    The font set to use when displaying math. See :rc:`mathtext.fontset`.

fontsize : float
    The font size, in points. Defaults to the value from the extension
    configuration option defined below.

Configuration options
---------------------

The mathtext extension has the following configuration options:

mathmpl_fontsize : float, default: 10.0
    Default font size, in points.

mathmpl_srcset : list of str, default: []
    Additional image sizes to generate when embedding in HTML, to support
    `responsive resolution images
    <https://developer.mozilla.org/en-US/docs/Learn/HTML/Multimedia_and_embedding/Responsive_images>`__.
    The list should contain additional x-descriptors (``'1.5x'``, ``'2x'``,
    etc.) to generate (1x is the default and always included.)

"""

import hashlib                  # 导入 hashlib 模块，用于生成哈希值
from pathlib import Path        # 从 pathlib 模块中导入 Path 类

from docutils import nodes      # 导入 docutils 中的 nodes 模块
from docutils.parsers.rst import Directive, directives  # 导入 RST 解析相关的模块
import sphinx                   # 导入 sphinx 模块
from sphinx.errors import ConfigError, ExtensionError  # 导入 Sphinx 的错误类型

import matplotlib as mpl        # 导入 matplotlib 库，并使用别名 mpl
from matplotlib import _api, mathtext  # 从 matplotlib 中导入 _api 和 mathtext 模块
from matplotlib.rcsetup import validate_float_or_None  # 从 matplotlib 的 rcsetup 模块中导入验证函数


# Define LaTeX math node:
class latex_math(nodes.General, nodes.Element):
    pass


def fontset_choice(arg):
    return directives.choice(arg, mathtext.MathTextParser._font_type_mapping)

def math_role(role, rawtext, text, lineno, inliner,
              options={}, content=[]):
    i = rawtext.find('`')
    latex = rawtext[i+1:-1]
    node = latex_math(rawtext)
    node['latex'] = latex
    node['fontset'] = options.get('fontset', 'cm')
    node['fontsize'] = options.get('fontsize',
                                   setup.app.config.mathmpl_fontsize)
    return [node], []
# 设置 math_role 的选项字典，包括字体集和字体大小
math_role.options = {'fontset': fontset_choice,
                     'fontsize': validate_float_or_None}


class MathDirective(Directive):
    """
    The ``.. mathmpl::`` directive, as documented in the module's docstring.
    """
    # 指令需要内容
    has_content = True
    # 没有必选参数
    required_arguments = 0
    # 没有可选参数
    optional_arguments = 0
    # 最终参数不允许有空白
    final_argument_whitespace = False
    # 指令支持的选项，包括字体集和字体大小
    option_spec = {'fontset': fontset_choice,
                   'fontsize': validate_float_or_None}

    def run(self):
        # 将内容连接成一个 LaTeX 字符串
        latex = ''.join(self.content)
        # 调用 latex_math 函数处理 LaTeX 字符串，生成节点
        node = latex_math(self.block_text)
        # 将 LaTeX 表达式和选项存储在节点中
        node['latex'] = latex
        node['fontset'] = self.options.get('fontset', 'cm')
        # 获取字体大小选项，若未设置则使用默认值
        node['fontsize'] = self.options.get('fontsize',
                                            setup.app.config.mathmpl_fontsize)
        # 返回包含节点的列表
        return [node]


# This uses mathtext to render the expression
def latex2png(latex, filename, fontset='cm', fontsize=10, dpi=100):
    # 设置 matplotlib 的渲染上下文，包括数学文本字体集和字体大小
    with mpl.rc_context({'mathtext.fontset': fontset, 'font.size': fontsize}):
        try:
            # 调用 mathtext.math_to_image 将 LaTeX 转换为 PNG 图像
            depth = mathtext.math_to_image(
                f"${latex}$", filename, dpi=dpi, format="png")
        except Exception:
            # 如果渲染失败，记录警告信息
            _api.warn_external(f"Could not render math expression {latex}")
            # 深度设为 0
            depth = 0
    # 返回渲染的深度
    return depth


# LaTeX to HTML translation stuff:
def latex2html(node, source):
    # 判断节点是否是行内节点
    inline = isinstance(node.parent, nodes.TextElement)
    # 获取节点中的 LaTeX 表达式、字体集和字体大小
    latex = node['latex']
    fontset = node['fontset']
    fontsize = node['fontsize']
    # 使用 hashlib 生成唯一名称，基于 LaTeX 表达式、字体集和字体大小的哈希值
    name = 'math-{}'.format(
        hashlib.md5(f'{latex}{fontset}{fontsize}'.encode()).hexdigest()[-10:])
    
    # 设置存储路径
    destdir = Path(setup.app.builder.outdir, '_images', 'mathmpl')
    # 创建目录，如果不存在
    destdir.mkdir(parents=True, exist_ok=True)
    
    # 设置目标文件名和路径
    dest = destdir / f'{name}.png'
    # 调用 latex2png 函数将 LaTeX 转换为 PNG 图像，并返回渲染深度
    depth = latex2png(latex, dest, fontset, fontsize=fontsize)

    # 设置 srcset，用于响应式图像
    srcset = []
    for size in setup.app.config.mathmpl_srcset:
        filename = f'{name}-{size.replace(".", "_")}.png'
        # 生成不同分辨率的 PNG 图像，每种分辨率对应一个 srcset 条目
        latex2png(latex, destdir / filename, fontset, fontsize=fontsize,
                  dpi=100 * float(size[:-1]))
        srcset.append(
            f'{setup.app.builder.imgpath}/mathmpl/{filename} {size}')
    if srcset:
        srcset = (f'srcset="{setup.app.builder.imgpath}/mathmpl/{name}.png, ' +
                  ', '.join(srcset) + '" ')

    # 根据节点类型设置 HTML 标签的 class 和 style
    if inline:
        cls = ''
    else:
        cls = 'class="center" '
    if inline and depth != 0:
        style = 'style="position: relative; bottom: -%dpx"' % (depth + 1)
    else:
        style = ''

    # 返回 HTML 图像标签
    return (f'<img src="{setup.app.builder.imgpath}/mathmpl/{name}.png"'
            f' {srcset}{cls}{style}/>')


def _config_inited(app, config):
    # 检查是否存在高分辨率图片设置
    # 使用 enumerate() 遍历列表 app.config.mathmpl_srcset，并获取索引 i 和对应的 size 值
    for i, size in enumerate(app.config.mathmpl_srcset):
        # 检查 size 的最后一个字符是否为 'x'
        if size[-1] == 'x':  # "2x" = "2.0"
            # 尝试将 size 去除最后一个字符并转换为浮点数，验证其有效性
            try:
                float(size[:-1])
            except ValueError:
                # 如果转换出错，抛出 ConfigError 异常，提示无效的 mathmpl_srcset 参数值格式
                raise ConfigError(
                    f'Invalid value for mathmpl_srcset parameter: {size!r}. '
                    'Must be a list of strings with the multiplicative '
                    'factor followed by an "x".  e.g. ["2.0x", "1.5x"]')
        else:
            # 如果 size 最后一个字符不是 'x'，抛出 ConfigError 异常，提示无效的 mathmpl_srcset 参数值格式
            raise ConfigError(
                f'Invalid value for mathmpl_srcset parameter: {size!r}. '
                'Must be a list of strings with the multiplicative '
                'factor followed by an "x".  e.g. ["2.0x", "1.5x"]')
# 将传入的 app 参数设置为全局变量 setup.app
def setup(app):
    setup.app = app
    # 向 Sphinx 应用添加数学公式相关的配置参数，并设置默认值
    app.add_config_value('mathmpl_fontsize', 10.0, True)
    app.add_config_value('mathmpl_srcset', [], True)
    
    try:
        # 尝试连接到 'config-inited' 事件，如果 Sphinx 版本 >= 1.8
        app.connect('config-inited', _config_inited)  # Sphinx 1.8+
    except ExtensionError:
        # 如果发生 ExtensionError，回退到使用 'env-updated' 事件连接
        app.connect('env-updated', lambda app, env: _config_inited(app, None))

    # 定义 HTML-Translator 中处理 LaTeX 数学公式的访问/离开方法
    def visit_latex_math_html(self, node):
        # 获取文档的源文件路径
        source = self.document.attributes['source']
        # 将 LaTeX 转换为 HTML 并添加到文档主体中
        self.body.append(latex2html(node, source))

    def depart_latex_math_html(self, node):
        # 离开 HTML-Translator 处理 LaTeX 数学公式时不做任何操作
        pass

    # 定义 LaTeX-Translator 中处理 LaTeX 数学公式的访问/离开方法
    def visit_latex_math_latex(self, node):
        # 检查当前节点是否为行内公式
        inline = isinstance(node.parent, nodes.TextElement)
        if inline:
            # 如果是行内公式，将 LaTeX 表达式添加到文档主体中
            self.body.append('$%s$' % node['latex'])
        else:
            # 如果是块级公式，添加 LaTeX 代码块到文档主体中
            self.body.extend(['\\begin{equation}',
                              node['latex'],
                              '\\end{equation}'])

    def depart_latex_math_latex(self, node):
        # 离开 LaTeX-Translator 处理 LaTeX 数学公式时不做任何操作
        pass

    # 向 Sphinx 应用添加节点类型 latex_math，并关联 HTML 和 LaTeX 处理方法
    app.add_node(latex_math,
                 html=(visit_latex_math_html, depart_latex_math_html),
                 latex=(visit_latex_math_latex, depart_latex_math_latex))
    
    # 向 Sphinx 应用添加 mathmpl 角色，并关联 math_role 函数
    app.add_role('mathmpl', math_role)
    # 向 Sphinx 应用添加 mathmpl 指令，并关联 MathDirective 类
    app.add_directive('mathmpl', MathDirective)
    
    # 根据 Sphinx 版本判断，向应用添加 math 角色和指令
    if sphinx.version_info < (1, 8):
        app.add_role('math', math_role)
        app.add_directive('math', MathDirective)

    # 定义并返回模块的元数据，表示此模块支持并行读取和写入
    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata
```