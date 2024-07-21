# `.\pytorch\docs\cpp\source\conf.py`

```
# PyTorch文档构建配置文件，由sphinx-quickstart在2016年12月23日13:31:47创建。

# 这个文件通过execfile()执行，并将当前目录设置为其所在的目录。

# 注意，并非所有可能的配置值都包含在这个自动生成的文件中。

# 所有的配置值都有一个默认值；被注释掉的值展示了默认设置。

# 如果扩展（或要使用autodoc自动文档生成的模块）在其他目录中，
# 在这里添加这些目录到sys.path。如果目录是相对于文档根目录的，
# 使用os.path.abspath将其转换为绝对路径，如下所示。
import os

# NB: C++ API文档生成使用doxygen / breathe / exhale目前仅在nightlies版本上启用
# （而不是在主干或PR上），因为CI中的OOM错误。详情请见https://github.com/pytorch/pytorch/issues/79992。

# 将当前目录（即'.'）的绝对路径插入sys.path中。
# sys.path.insert(0, os.path.abspath('.'))

# 导入textwrap模块。
import textwrap

# -- General configuration ------------------------------------------------

# 如果你的文档需要一个最小的Sphinx版本，这里声明它。
needs_sphinx = "3.1.2"

# 根据环境变量RUN_DOXYGEN的值（如果是"true"），决定是否运行doxygen。
run_doxygen = os.environ.get("RUN_DOXYGEN", "false") == "true"

# 将Sphinx扩展模块的名称作为字符串添加到这里。
# 可以是随Sphinx一起提供的扩展（命名为'sphinx.ext.*'）或者自定义的扩展。
extensions = [
    "sphinx.ext.intersphinx",
] + (["breathe", "exhale"] if run_doxygen else [])

# 用于intersphinx的映射字典，指定pytorch文档的URL和其他相关信息。
intersphinx_mapping = {"pytorch": ("https://pytorch.org/docs/main", None)}

# 设置与breathe / exhale通信的绝对路径，
# 以期望/应该由breathe / exhale修剪的项目。
# 此文件为{repo_root}/docs/cpp/source/conf.py
this_file_dir = os.path.abspath(os.path.dirname(__file__))
doxygen_xml_dir = os.path.join(
    os.path.dirname(this_file_dir),  # {repo_root}/docs/cpp
    "build",  # {repo_root}/docs/cpp/build
    "xml",  # {repo_root}/docs/cpp/build/xml
)
repo_root = os.path.dirname(  # {repo_root}
    os.path.dirname(  # {repo_root}/docs
        os.path.dirname(  # {repo_root}/docs/cpp
            this_file_dir  # {repo_root}/docs/cpp/source
        )
    )
)

# 设置breathe项目的路径，这里指定为doxygen生成的XML文件所在的目录。
breathe_projects = {"PyTorch": doxygen_xml_dir}
# 指定默认的breathe项目。
breathe_default_project = "PyTorch"

# 设置exhale扩展的参数。
exhale_args = {
    ############################################################################
    # 下面这些参数是必需的。                                                   #
    ############################################################################
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "Library API",
    "doxygenStripFromPath": repo_root,
    ############################################################################
    # 建议使用的可选参数。                                                    #
    ############################################################################
    "createTreeView": True,
    "exhaleExecutesDoxygen": True,
    "exhaleUseDoxyfile": True,
    "verboseBuild": True,
}
    ############################################################################
    # HTML Theme specific configurations.                                      #
    ############################################################################
    # 以下是针对 HTML 主题的特定配置。
    
    # Fix broken Sphinx RTD Theme 'Edit on GitHub' links
    # 修复 Sphinx RTD 主题中 'Edit on GitHub' 链接损坏的问题
    # 在 FAQ 页面搜索 'Edit on GitHub'：
    #     http://exhale.readthedocs.io/en/latest/faq.html
    "pageLevelConfigMeta": ":github_url: https://github.com/pytorch/pytorch",
    
    ############################################################################
    # Individual page layout example configuration.                            #
    ############################################################################
    # 个别页面布局示例配置。
    
    # Example of adding contents directives on custom kinds with custom title
    # 添加内容指令到具有自定义标题的自定义种类示例
    "contentsTitle": "Page Contents",
    "kindsWithContentsDirectives": ["class", "file", "namespace", "struct"],
    
    # Exclude PIMPL files from class hierarchy tree and namespace pages.
    # 从类层次结构树和命名空间页面中排除 PIMPL 文件。
    "listingExclude": [r".*Impl$"],
    
    ############################################################################
    # Main library page layout example configuration.                          #
    ############################################################################
    # 主库页面布局示例配置。
    
    "afterTitleDescription": textwrap.dedent(
        """
        Welcome to the developer reference for the PyTorch C++ API.
    """
    ),
# } 不明确的单行注释，可能是代码片段中的遗留注释

# 设置Sphinx文档的主要域为C++语言
primary_domain = "cpp"

# 设置Pygments代码高亮语言为C++
highlight_language = "cpp"

# 添加包含模板的路径，相对于当前目录
# templates_path = ['_templates']

# 指定源文件的后缀名
source_suffix = ".rst"

# 指定主目录文档文件
master_doc = "index"

# 关于项目的一般信息
project = "PyTorch"
copyright = "2022, PyTorch Contributors"
author = "PyTorch Contributors"

# 项目版本信息，用于替换|version|和|release|，同时在其他地方使用
# 短版本号 X.Y
# TODO: 在v1.0版本时更改为[:2]
version = "main"
# 完整版本号，包括alpha/beta/rc标签
# TODO: 验证此处是否按预期工作
release = "main"

# 用于Sphinx生成的内容的语言。参考文档以获取支持的语言列表。
# 也用于通过gettext目录进行内容翻译。
# 通常情况下，你可以通过命令行为这些情况设置"language"。
language = None

# 忽略查找源文件时匹配的文件和目录模式列表，相对于源目录
# 这些模式也会影响到html_static_path和html_extra_path
exclude_patterns = []

# Pygments（语法高亮）样式的名称
pygments_style = "sphinx"

# 如果为True，则`todo`和`todoList`会生成输出，否则不生成任何内容。
todo_include_todos = True

# -- HTML输出选项 ----------------------------------------------

# 用于HTML和HTML帮助页面的主题。查看文档以获取内置主题列表。
html_theme = "pytorch_sphinx_theme"

# 主题选项是特定主题的，用于进一步自定义主题的外观和感觉。
# 有关每个主题可用选项的列表，请参阅文档。
html_theme_options = {
    "canonical_url": "https://pytorch.org/docs/stable/",
    "pytorch_project": "docs",
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
}

# NOTE: 共享Python文档资源
# 指定HTML页面的Logo路径
html_logo = os.path.join(
    repo_root, "docs", "source", "_static", "img", "pytorch-logo-dark-unstable.png"
)

# 添加包含自定义静态文件（如样式表）的路径，
# 相对于当前目录。这些文件会在内置静态文件之后复制，
# 因此名为"default.css"的文件将覆盖内置的"default.css"。
# NOTE: 共享Python文档资源
html_static_path = [os.path.join(repo_root, "docs", "cpp", "source", "_static")]


# Sphinx自动调用此函数，使得这个`conf.py`成为一个"extension"。
def setup(app):
    # NOTE: 在Sphinx 1.8+中，`html_css_files`是官方的配置值，
    # 可以移出这个函数（和setup(app)函数）之外。
    pass
    # 定义一个包含单个字符串元素（文件名）的列表，包含了要添加的 CSS 文件名
    html_css_files = ["cpp_theme.css"]

    # 根据 Sphinx 应用的版本选择合适的方法来添加 CSS 文件：
    # - 在 Sphinx 1.8 及以上版本中，使用 `add_css_file`
    # - 在 Sphinx 1.7 及更早版本中，使用 `add_stylesheet`（在 1.8 版本中已弃用）
    add_css = getattr(app, "add_css_file", app.add_stylesheet)

    # 遍历 html_css_files 列表中的每个文件名，逐个添加到 Sphinx 应用中
    for css_file in html_css_files:
        add_css(css_file)
# -- Options for HTMLHelp output ------------------------------------------

# 用于生成 HTML 帮助文档的输出文件基础名称
htmlhelp_basename = 'PyTorchdoc'


# -- Options for LaTeX output ---------------------------------------------

# LaTeX 输出的配置项
latex_elements = {
    # 纸张大小 ('letterpaper' 或 'a4paper')
    #
    # 'papersize': 'letterpaper',
    # 字体大小 ('10pt', '11pt' 或 '12pt')
    #
    # 'pointsize': '10pt',
    # LaTeX 导言中的额外内容
    #
    # 'preamble': '',
    # LaTeX 图片（浮动体）的对齐方式
    #
    # 'figure_align': 'htbp',
}

# 将文档树分组成 LaTeX 文件。元组列表
# (源文件, 目标文件名, 标题,
#  作者, 文档类别 [howto, manual, 或自定义类别]).
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

# 每个手册页的配置选项。元组列表
# (源文件, 名称, 描述, 作者, 手册页节号).
man_pages = [(master_doc, "PyTorch", "PyTorch Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# 将文档树分组成 Texinfo 文件的配置选项。元组列表
# (源文件, 目标文件名, 标题, 作者,
#  目录菜单项, 描述, 类别)
texinfo_documents = [
    (
        master_doc,
        "PyTorch",
        "PyTorch Documentation",
        author,
        "PyTorch",
        "One line description of project.",
        "Miscellaneous",
    ),
]
```