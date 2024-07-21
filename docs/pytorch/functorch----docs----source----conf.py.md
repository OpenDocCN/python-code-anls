# `.\pytorch\functorch\docs\source\conf.py`

```py
# 当前文件的执行目录被设置为其所在的目录。

# 并非所有可能的配置值都在此自动生成的文件中。
# 注释掉的配置值展示了默认设置。

# 如果扩展（或用于自动文档的模块）位于另一个目录中，
# 在此处将这些目录添加到 sys.path。如果目录是相对于文档根目录的，
# 使用 os.path.abspath 来使其绝对化，就像这里展示的一样。
import os

# 导入 functorch 模块
import functorch

# import sys

# sphinx-autobuild 的源代码目录，相对于此文件所在的位置
# sys.path.insert(0, os.path.abspath('../..'))

# 从环境变量中获取 RELEASE 值，默认为 False
RELEASE = os.environ.get("RELEASE", False)

# 导入 pytorch_sphinx_theme 主题模块
import pytorch_sphinx_theme

# -- General configuration ------------------------------------------------

# sphinx 的要求版本从 docs/requirements.txt 文件中设置

# 将要使用的 Sphinx 扩展模块名称作为字符串添加到这里。
# 它们可以是随 Sphinx 一起提供的扩展（命名为 'sphinx.ext.*'）或您自定义的扩展。

extensions = [
    "sphinx.ext.autodoc",         # 自动生成文档
    "sphinx.ext.autosummary",     # 自动生成摘要
    "sphinx.ext.doctest",         # 测试文档中的代码示例是否正确
    "sphinx.ext.intersphinx",     # 支持链接到外部文档的跳转
    "sphinx.ext.todo",            # 添加 TODO 支持
    "sphinx.ext.coverage",        # 收集文档覆盖率信息
    "sphinx.ext.napoleon",        # 支持 Google 和 Numpy 风格的文档字符串
    "sphinx.ext.viewcode",        # 添加源码链接
    # 'sphinxcontrib.katex',
    "sphinx.ext.autosectionlabel",# 自动生成文档的标签
    "sphinx_copybutton",          # 添加复制按钮
    "myst_nb",                    # 支持 Jupyter 笔记本
]

# sys.path.insert(0, os.path.abspath('./notebooks'))

# 生成自动生成摘要文件的模板
# autosummary_generate = True
numpydoc_show_class_members = False

# autosectionlabel 在出现重复部分名称时会抛出警告。
# 以下配置告诉 autosectionlabel 不要对不同文档中的重复部分名称抛出警告。
autosectionlabel_prefix_document = True

# 告知 myst 不要执行 ipynb 教程。
nb_execution_mode = "off"

# katex 选项
#
#

katex_prerender = True

napoleon_use_ivar = True

# 生成自动生成摘要文件的模板
autosummary_generate = True

# 将包含模板的路径添加到这里，相对于此目录
templates_path = ["_templates"]

# 源文件名的后缀（们）。
# 您可以指定多个后缀，作为字符串列表：
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# 主 ToC 树文档。
master_doc = "index"

# 有关项目的一般信息。
project = "functorch"
copyright = "PyTorch Contributors"
author = "PyTorch Contributors"
functorch_version = str(functorch.__version__)

# 正在文档化的项目的版本信息，用作替换 |version| 和 |release| 的位置，
# 也用于文档中的其他各处。
#
# 短的 X.Y 版本。
# TODO: 在 v1.0 时改为 [:2]
version = "nightly (" + functorch_version + ")"
# 完整的版本，包括 alpha/beta/rc 标签。
# TODO: 验证这是否按预期工作
release = "nightly"

# 在这里设置自定义的 html_title。
# 如果未设置，默认为 " ".join(project, release, "documentation")
# TODO: I don't know if this flag works, please check before using it
# 如果 RELEASE 为真，则抛出 RuntimeError 异常
if RELEASE:
    raise RuntimeError("NYI")
    # 移除版本号中的哈希（以 'a' 开头）（若存在）
    # version_end = functorch_version.find('a')
    # if version_end == -1:
    #     html_title = " ".join((project, functorch_version, "documentation"))
    #     version = functorch_version
    # else:
    #     html_title = " ".join((project, functorch_version[:version_end], "documentation"))
    #     version = functorch_version[:version_end]
    # release = version

# Sphinx 自动生成内容的语言设置，详细支持语言列表请参考文档
language = "en"

# 忽略源文件中的指定文件和目录的模式列表（相对于源目录）
# 这些模式同时也影响到 html_static_path 和 html_extra_path
exclude_patterns = ["notebooks/colab**", "notebooks/_src/**"]

# Pygments（语法高亮）所使用的样式
pygments_style = "sphinx"

# 若为 True，则 `todo` 和 `todoList` 生成输出；否则生成空值
todo_include_todos = True

# 禁用文档字符串继承
autodoc_inherit_docstrings = False

# 禁用显示类型注解，这可能会非常冗长
autodoc_typehints = "none"

# 启用在文档字符串的第一行覆盖函数签名
autodoc_docstring_signature = True

# -- katex javascript in header
#
#    def setup(app):
#    app.add_javascript("https://cdn.jsdelivr.net/npm/katex@0.10.0-beta/dist/katex.min.js")


# -- HTML 输出选项 ----------------------------------------------
#
# 用于 HTML 和 HTML 帮助页面的主题设置，查看文档以获取内置主题列表
html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# 主题选项是特定于主题的，用于进一步定制主题的外观和感觉
# 详细的选项列表，请查阅相应主题的文档
html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
    "pytorch_project": "functorch",
    "navigation_with_keys": True,
    "analytics_id": "UA-117752657-2",
}

# 添加包含自定义静态文件（如样式表）的路径，相对于此目录
# 这些文件会在内置静态文件之后复制，因此 "default.css" 将覆盖内置的 "default.css"
html_static_path = ["_static"]

# 自定义的 CSS 文件列表
html_css_files = [
    "css/custom.css",
]


# Sphinx 自动调用的函数，使得 `conf.py` 成为一个 "扩展"
def setup(app):
    # 注意：在 Sphinx 1.8+ 中，`html_css_files` 是官方的配置值，
    # 可以移到此函数之外（同时可以删除 setup(app) 函数）。
    html_css_files = [
        "https://cdn.jsdelivr.net/npm/katex@0.10.0-beta/dist/katex.min.css"
    ]
    # 获取适当的函数引用用于向 Sphinx 应用程序添加 CSS 文件。在 Sphinx 1.8 中更名为 `add_css_file`，
    # 在 1.7 及更早版本中使用 `add_stylesheet`（在 1.8 版本中已弃用）。
    add_css = getattr(app, "add_css_file", app.add_stylesheet)
    # 遍历每个 CSS 文件，调用适当的函数将其添加到 Sphinx 应用程序中。
    for css_file in html_css_files:
        add_css(css_file)
# -- Options for HTMLHelp output ------------------------------------------

# HTML 帮助文档输出的文件基础名。
htmlhelp_basename = "PyTorchdoc"


# -- Options for LaTeX output ---------------------------------------------

# LaTeX 输出的选项设置。
latex_elements = {
    # 纸张大小 ('letterpaper' 或 'a4paper')。
    #
    # 'papersize': 'letterpaper',
    # 字体大小 ('10pt', '11pt' 或 '12pt')。
    #
    # 'pointsize': '10pt',
    # LaTeX 导言中的附加内容。
    #
    # 'preamble': '',
    # LaTeX 图片的浮动对齐方式。
    #
    # 'figure_align': 'htbp',
}

# 将文档树分组到 LaTeX 文件中的选项。元组列表
# (源文件名, 目标文件名, 标题,
#  作者, 文档类 [howto, manual, 或自定义类])。
latex_documents = [
    (
        master_doc,
        "pytorch.tex",
        "PyTorch Documentation",
        "Torch Contributors",
        "manual",
    ),
]


# -- Options for manual page output ---------------------------------------

# 每个手册页输出的条目。元组列表
# (源文件名, 名称, 描述, 作者, 手册节号)。
man_pages = [(master_doc, "functorch", "functorch Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# 将文档树分组到 Texinfo 文件中的选项。元组列表
# (源文件名, 目标文件名, 标题, 作者,
#  目录菜单条目, 描述, 类别)
texinfo_documents = [
    (
        master_doc,
        "functorch",
        "functorch Documentation",
        author,
        "functorch",
        "One line description of project.",
        "Miscellaneous",
    ),
]


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

import sphinx.ext.doctest

# -- A patch that prevents Sphinx from cross-referencing ivar tags -------
# 防止 Sphinx 跨引用 ivar 标签的补丁。
# 详细见 http://stackoverflow.com/a/41184353/3343043

from docutils import nodes
from sphinx import addnodes
from sphinx.util.docfields import TypedField

# Without this, doctest adds any example with a `>>>` as a test
# 如果没有这个设置，doctest 会将任何带有 `>>>` 的示例视为测试。
doctest_test_doctest_blocks = ""
# 设置 doctest 的默认标志为显示省略号。
doctest_default_flags = sphinx.ext.doctest.doctest.ELLIPSIS
# 在全局范围内设置 doctest 的全局设置。
doctest_global_setup = """
import torch
try:
    import torchvision
except ImportError:
    torchvision = None
"""


def patched_make_field(self, types, domain, items, **kw):
    # `kw` catches `env=None` needed for newer sphinx while maintaining
    #  backwards compatibility when passed along further down!
    # `kw` 捕获了新版 Sphinx 中需要的 `env=None`，同时在传递时保持向后兼容性。

    # (List, unicode, Tuple) -> nodes.field
    # (列表, Unicode, 元组) -> nodes.field

    pass
    # 定义处理单个字段项的函数，接受字段参数和内容作为参数
    def handle_item(fieldarg, content):
        # 创建一个段落节点
        par = nodes.paragraph()
        # 将字段参数作为强调的文本添加到段落中
        par += addnodes.literal_strong("", fieldarg)  # Patch: this line added
        
        # 下面的代码被注释掉，因为引入了一个修补行用于处理不一致的问题
        # par.extend(self.make_xrefs(self.rolename, domain, fieldarg,
        #                           addnodes.literal_strong))
        
        # 检查字段参数是否在类型字典中
        if fieldarg in types:
            # 在段落中添加左括号
            par += nodes.Text(" (")
            # 使用 .pop() 方法从类型字典中取出字段类型，以避免在文档树中插入相同的节点两次
            fieldtype = types.pop(fieldarg)
            
            # 如果字段类型只有一个节点并且是 Text 类型，则将其转换为字符串
            if len(fieldtype) == 1 and isinstance(fieldtype[0], nodes.Text):
                typename = "".join(n.astext() for n in fieldtype)
                # 将特定的类型字符串替换为带有 python: 前缀的形式
                typename = typename.replace("int", "python:int")
                typename = typename.replace("long", "python:long")
                typename = typename.replace("float", "python:float")
                typename = typename.replace("bool", "python:bool")
                typename = typename.replace("type", "python:type")
                # 使用 make_xrefs 方法创建交叉引用，将类型名称转换为强调形式
                par.extend(
                    self.make_xrefs(
                        self.typerolename,
                        domain,
                        typename,
                        addnodes.literal_emphasis,
                        **kw,
                    )
                )
            else:
                # 将字段类型节点直接添加到段落中
                par += fieldtype
            
            # 在段落中添加右括号
            par += nodes.Text(")")
        
        # 在段落末尾添加分隔符 " -- "
        par += nodes.Text(" -- ")
        # 将内容添加到段落中
        par += content
        # 返回组装好的段落节点
        return par

    # 创建字段名节点，使用 self.label 作为标签内容
    fieldname = nodes.field_name("", self.label)
    
    # 如果只有一个字段项且可以折叠
    if len(items) == 1 and self.can_collapse:
        # 获取第一个字段项的字段参数和内容
        fieldarg, content = items[0]
        # 调用 handle_item 函数处理字段项，获取处理后的节点
        bodynode = handle_item(fieldarg, content)
    else:
        # 如果有多个字段项或不能折叠，则创建一个列表类型的节点作为 bodynode
        bodynode = self.list_type()
        # 遍历所有字段项，每个都调用 handle_item 处理后添加到列表节点中
        for fieldarg, content in items:
            bodynode += nodes.list_item("", handle_item(fieldarg, content))
    
    # 创建字段内容节点
    fieldbody = nodes.field_body("", bodynode)
    # 返回包含字段名节点和字段内容节点的字段节点
    return nodes.field("", fieldname, fieldbody)
# 将 TypedField 类的 make_field 方法替换为 patched_make_field 方法
TypedField.make_field = patched_make_field

# 设置复制按钮的提示文本，指示在命令行中输入的提示符或正则表达式
copybutton_prompt_text = r">>> |\.\.\. "

# 声明复制按钮的提示文本是否为正则表达式，这里设置为 True
copybutton_prompt_is_regexp = True
```