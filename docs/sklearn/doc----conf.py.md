# `D:\src\scipysrc\scikit-learn\doc\conf.py`

```
# 导入所需的标准库模块
import os  # 提供与操作系统交互的功能
import re  # 提供正则表达式操作的支持
import sys  # 提供对解释器运行时环境的访问
import warnings  # 控制警告的显示
from datetime import datetime  # 从datetime模块中导入datetime类
from pathlib import Path  # 提供处理文件路径的类和函数

# 导入第三方库中的特定模块和函数
from sklearn.externals._packaging.version import parse  # 导入版本解析函数parse
from sklearn.utils._testing import turn_warnings_into_errors  # 导入转换警告为错误的函数

# 如果需要，将额外的扩展模块所在目录添加到sys.path中
sys.path.insert(0, os.path.abspath("."))  # 添加当前目录的绝对路径到sys.path
sys.path.insert(0, os.path.abspath("sphinxext"))  # 添加sphinxext目录的绝对路径到sys.path

# 导入用于模板处理的jinja2库
import jinja2  # 模板引擎，用于生成文本输出
import sphinx_gallery  # 用于生成示例画廊的工具模块
from github_link import make_linkcode_resolve  # 导入函数make_linkcode_resolve用于解析GitHub链接
from sphinx_gallery.notebook import add_code_cell, add_markdown_cell  # 导入用于在笔记本中添加代码和Markdown单元的函数
from sphinx_gallery.sorting import ExampleTitleSortKey  # 导入示例标题排序的关键函数

try:
    # 尝试配置plotly以将其输出集成到由sphinx-gallery生成的HTML页面中
    import plotly.io as pio  # 导入plotly的输入输出模块

    pio.renderers.default = "sphinx_gallery"  # 设置plotly默认渲染器为"sphinx_gallery"
except ImportError:
    # 如果导入plotly失败，则允许在不运行需要plotly的示例时渲染文档
    pass

# -- 一般配置 ---------------------------------------------------------------

# 在此处添加任何要使用的Sphinx扩展模块的名称字符串
extensions = [
    "sphinx.ext.autodoc",  # 自动文档生成扩展
    "sphinx.ext.autosummary",  # 自动生成摘要扩展
    "numpydoc",  # NumPy风格的文档字符串解析扩展
    "sphinx.ext.linkcode",  # 提供通过代码中的链接跳转到源代码的功能
    "sphinx.ext.doctest",  # 执行文档中的doctest示例的扩展
    "sphinx.ext.intersphinx",  # 支持链接到其他Sphinx文档的扩展
    "sphinx.ext.imgconverter",  # 图像转换器扩展
    "sphinx_gallery.gen_gallery",  # 生成示例画廊的扩展
    "sphinx-prompt",  # 控制复制代码片段时的提示文本的扩展
    "sphinx_copybutton",  # 提供复制按钮以便用户复制代码片段的扩展
    "sphinxext.opengraph",  # Open Graph协议扩展，用于社交媒体分享
    "matplotlib.sphinxext.plot_directive",  # 提供在文档中插入matplotlib图形的指令扩展
    "sphinxcontrib.sass",  # 提供Sass支持的扩展
    "sphinx_remove_toctrees",  # 移除指定目录的目录树扩展
    "sphinx_design",  # 提供设计相关功能的扩展
    # 查看sphinxext目录中的扩展
    "allow_nan_estimators",  # 允许NaN评估器的扩展
    "autoshortsummary",  # 自动生成简短摘要的扩展
    "doi_role",  # 支持DOI链接的角色扩展
    "dropdown_anchors",  # 提供下拉锚点功能的扩展
    "move_gallery_links",  # 移动示例画廊链接的扩展
    "override_pst_pagetoc",  # 覆盖PST页面目录的扩展
    "sphinx_issues",  # 支持在文档中链接到问题跟踪系统的扩展
]

# 指定复制代码片段时用于识别提示符的文本
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True  # 指示复制按钮的提示文本是否为正则表达式
copybutton_exclude = "style"  # 排除复制的样式文件

try:
    import jupyterlite_sphinx  # 尝试导入jupyterlite_sphinx模块

    extensions.append("jupyterlite_sphinx")  # 将jupyterlite_sphinx扩展添加到extensions列表中
    with_jupyterlite = True  # 设置标志以指示已加载jupyterlite_sphinx扩展
except ImportError:
    # 在某些情况下，不要求安装jupyterlite_sphinx模块，例如在doc-min-dependencies构建中
    warnings.warn(
        "jupyterlite_sphinx is not installed, you need to install it "
        "if you want JupyterLite links to appear in each example"
    )
    with_jupyterlite = False  # 设置标志以指示未加载jupyterlite_sphinx扩展
# 是否生成包含 `plot::` 指令的示例，条件是代码中包含 `import matplotlib` 或 `from matplotlib import`
numpydoc_use_plots = True

# `::plot` 指令的选项设置:
# https://matplotlib.org/stable/api/sphinxext_plot_directive_api.html
plot_formats = ["png"]  # 图形输出格式为 PNG
plot_include_source = True  # 包含源代码
plot_html_show_formats = False  # 不在 HTML 中显示可用的输出格式
plot_html_show_source_link = False  # 不在 HTML 中显示源链接

# 不需要类成员的表格，因为 `sphinxext/override_pst_pagetoc.py` 会在次级侧边栏中显示它们
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False

# 希望在页内显示类成员的 toc，而不是每个条目单独一页
numpydoc_class_members_toctree = False

# 对于数学公式，默认使用 mathjax，如果设置了 NO_MATHJAX 环境变量，则使用 svg
if os.environ.get("NO_MATHJAX"):
    extensions.append("sphinx.ext.imgmath")  # 添加 imgmath 扩展
    imgmath_image_format = "svg"  # 设置图像格式为 SVG
    mathjax_path = ""  # 空字符串，因为不使用 mathjax
else:
    extensions.append("sphinx.ext.mathjax")  # 添加 mathjax 扩展
    mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"  # MathJax CDN 路径

# 添加包含模板的路径，相对于当前目录
templates_path = ["templates"]

# 即使没有引用，也要生成 autosummary
autosummary_generate = True

# 源文件的后缀名
source_suffix = ".rst"

# 源文件的编码
source_encoding = "utf-8"

# 主 ToC 文档
root_doc = "index"

# 关于项目的一般信息
project = "scikit-learn"
copyright = f"2007 - {datetime.now().year}, scikit-learn developers (BSD License)"

# 项目版本信息，替代 |version| 和 |release|，也在其他地方使用
import sklearn

parsed_version = parse(sklearn.__version__)
version = ".".join(parsed_version.base_version.split(".")[:2])  # 获取短版本号 X.Y
if parsed_version.is_postrelease:
    release = parsed_version.base_version  # 如果是后发布版本，则使用完整版本号
else:
    release = sklearn.__version__  # 否则使用当前版本号

# 用于 Sphinx 自动生成内容的语言设置
# language = None

# 替换 |today| 的选项：设置 today 为非假值时使用，否则使用 today_fmt 作为 strftime 调用的格式
# today = ''
# today_fmt = '%B %d, %Y'

# 忽略查找源文件时匹配的模式列表，相对于源目录
exclude_patterns = [
    "_build",  # 构建目录
    "templates",  # 模板目录
    "includes",  # 包含目录
    "**/sg_execution_times.rst",  # 特定文件的匹配模式
]

# reST 默认角色（用于标记 `text`），在所有文档中使用
default_role = "literal"

# 如果为真，则在所有交叉引用文本（如 .. function::）后附加 '()'
add_function_parentheses = False

# 如果为真，则将当前模块名添加到所有描述单元标题的前面（例如 .. function::）
# add_module_names = True

# 如果设置为True，则会显示sectionauthor和moduleauthor指令在输出中的效果。
# 默认情况下它们被忽略。
# show_authors = False

# 用于模块索引排序时忽略的前缀列表。
# modindex_common_prefix = []


# -- HTML输出选项 -----------------------------------------------------------

# 用于HTML和HTML帮助页面的主题。目前内置的主题有'default'和'sphinxdoc'。
html_theme = "pydata_sphinx_theme"

# 主题选项是特定主题的定制化参数，用于进一步调整主题的外观和感觉。
# 关于每个主题可用选项的列表，请参见文档。
html_theme_options = {
    # -- General configuration ------------------------------------------------
    "sidebar_includehidden": True,  # 是否在侧边栏中包含隐藏的内容
    "use_edit_page_button": True,    # 是否显示编辑页面按钮
    "external_links": [],            # 外部链接列表
    "icon_links_label": "Icon Links",# 图标链接的标签
    "icon_links": [                  # 图标链接列表
        {
            "name": "GitHub",
            "url": "https://github.com/scikit-learn/scikit-learn",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    "analytics": {                   # 分析选项
        "plausible_analytics_domain": "scikit-learn.org",
        "plausible_analytics_url": "https://views.scientific-python.org/js/script.js",
    },
    # 如果在article_footer_items中包含"prev-next"，则将show_prev_next设置为True会重复显示前后链接。
    # 参见 https://github.com/pydata/pydata-sphinx-theme/blob/b731dc230bc26a3d1d1bb039c56c977a9b3d25d8/src/pydata_sphinx_theme/theme/pydata_sphinx_theme/layout.html#L118-L129
    "show_prev_next": False,         # 是否显示上一页和下一页链接
    "search_bar_text": "Search the docs ...",  # 搜索栏的提示文本
    "navigation_with_keys": False,   # 是否启用导航键操作
    "collapse_navigation": False,    # 是否折叠导航栏
    "navigation_depth": 2,           # 导航深度
    "show_nav_level": 1,             # 显示导航级别
    "show_toc_level": 1,             # 显示目录级别
    "navbar_align": "left",          # 导航栏对齐方式
    "header_links_before_dropdown": 5,  # 下拉菜单之前的头部链接数量
    "header_dropdown_text": "More",  # 下拉菜单的文本
    # 切换器需要一个包含文档版本列表的JSON文件，由`build_tools/circle/list_versions.py`脚本生成，并放置在`js/`静态目录下；
    # 然后会复制到构建文档的`_static`目录下。
    "switcher": {
        "json_url": "https://scikit-learn.org/dev/_static/versions.json",
        "version_match": release,   # 版本匹配
    },
    # 如果文档构建流水线失败，可以将check_switcher设置为False。
    # 参见 https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html#configure-switcher-json-url
    "check_switcher": True,          # 是否检查切换器
    "pygments_light_style": "tango", # 亮色主题样式
    "pygments_dark_style": "monokai",# 暗色主题样式
    "logo": {                        # Logo配置
        "alt_text": "scikit-learn homepage",
        "image_relative": "logos/scikit-learn-logo-small.png",
        "image_light": "logos/scikit-learn-logo-small.png",
        "image_dark": "logos/scikit-learn-logo-small.png",
    },
    "surface_warnings": True,        # 是否显示表面警告
}
    # -- Template placement in theme layouts ----------------------------------
    
    # 定义导航栏的开始部分，包含导航栏标志（logo）
    "navbar_start": ["navbar-logo"],
    
    # 定义导航栏的中间部分，包含导航栏导航项目
    # 注意：navbar_center的对齐方式由navbar_align控制
    "navbar_center": ["navbar-nav"],
    
    # 定义导航栏的结束部分，包含主题切换器、导航栏图标链接、版本切换器
    "navbar_end": ["theme-switcher", "navbar-icon-links", "version-switcher"],
    
    # navbar_persistent指定的项目在移动设备上也会保持在导航栏的右侧
    "navbar_persistent": ["search-button"],
    
    # 文章标题部分开始处，包含面包屑导航
    "article_header_start": ["breadcrumbs"],
    
    # 文章标题部分结束处，不包含任何项目
    "article_header_end": [],
    
    # 文章底部项目，包含上一篇和下一篇文章链接
    "article_footer_items": ["prev-next"],
    
    # 内容底部项目，不包含任何项目
    "content_footer_items": [],
    
    # 使用html_sidebars将页面模式映射到侧边栏模板列表
    "primary_sidebar_end": [],
    
    # 页脚开始部分，包含版权声明
    "footer_start": ["copyright"],
    
    # 页脚中间部分，不包含任何项目
    "footer_center": [],
    
    # 页脚结束部分，不包含任何项目
    "footer_end": [],
    
    # 当指定为字典时，键应该遵循 glob 风格的模式，如 https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-exclude_patterns
    # 特别地，"**" 指定所有页面的默认设置
    # 使用 :html_theme.sidebar_secondary.remove: 可以对整个文件进行侧边栏移除
    "secondary_sidebar_items": {"**": ["page-toc", "sourcelink"]},
    
    # 是否显示版本警告横幅
    "show_version_warning_banner": True,
    
    # 公告部分，目前未指定任何内容
    "announcement": None,
}

# 在这里添加包含自定义主题的路径，相对于当前目录
# html_theme_path = ["themes"]

# Sphinx 文档集的名称。如果为 None，则默认为 "<project> v<release> documentation"。
# html_title = None

# 导航栏的较短标题，默认与 html_title 相同。
html_short_title = "scikit-learn"

# 用作文档 favicon 的图像文件名（位于静态路径内）。应为 16x16 或 32x32 像素的 Windows 图标文件 (.ico)。
html_favicon = "logos/favicon.ico"

# 添加包含自定义静态文件（如样式表）的路径，相对于当前目录。这些文件将在内置静态文件之后复制，例如 "default.css" 将覆盖内置的 "default.css"。
html_static_path = ["images", "css", "js"]

# 如果非空，则在每个页面底部插入一个 'Last updated on:' 时间戳，使用给定的 strftime 格式。
# html_last_updated_fmt = '%b %d, %Y'

# 自定义侧边栏模板，将文档名称映射到模板名称。
# 用于移除没有目录的页面上的左侧边栏的解决方法
# 更好的解决方案应该遵循以下合并：
# https://github.com/pydata/pydata-sphinx-theme/pull/1682
html_sidebars = {
    "install": [],
    "getting_started": [],
    "glossary": [],
    "faq": [],
    "support": [],
    "related_projects": [],
    "roadmap": [],
    "governance": [],
    "about": [],
}

# 应该渲染到页面的其他模板，将页面名称映射到模板名称。
html_additional_pages = {"index": "index.html"}

# 额外复制的文件
# html_extra_path = []

# 额外的 JS 文件
html_js_files = [
    "scripts/dropdown.js",
    "scripts/version-switcher.js",
]

# 使用 sphinxcontrib-sass 将 scss 文件编译为 css 文件
sass_src_dir, sass_out_dir = "scss", "css/styles"
sass_targets = {
    f"{file.stem}.scss": f"{file.stem}.css"
    for file in Path(sass_src_dir).glob("*.scss")
}

# 额外的 CSS 文件，应该是 `sass_targets` 值的子集
html_css_files = ["styles/colors.css", "styles/custom.css"]


def add_js_css_files(app, pagename, templatename, context, doctree):
    """Load additional JS and CSS files only for certain pages.

    Note that `html_js_files` and `html_css_files` are included in all pages and
    should be used for the ones that are used by multiple pages. All page-specific
    JS and CSS files should be added here instead.
    """
    if pagename == "api/index":
        # 外部: jQuery 和 DataTables
        app.add_js_file("https://code.jquery.com/jquery-3.7.0.js")
        app.add_js_file("https://cdn.datatables.net/2.0.0/js/dataTables.min.js")
        app.add_css_file(
            "https://cdn.datatables.net/2.0.0/css/dataTables.dataTables.min.css"
        )
        # 内部: API 搜索初始化和样式
        app.add_js_file("scripts/api-search.js")
        app.add_css_file("styles/api-search.css")
    # 如果页面名为 "index"，添加 index 页面的 CSS 文件
    elif pagename == "index":
        app.add_css_file("styles/index.css")
    # 如果页面名为 "install"，添加 install 页面的 CSS 文件
    elif pagename == "install":
        app.add_css_file("styles/install.css")
    # 如果页面名以 "modules/generated/" 开头，添加 API 页面的 CSS 文件
    elif pagename.startswith("modules/generated/"):
        app.add_css_file("styles/api.css")
# 如果为 False，则不生成模块索引。
html_domain_indices = False

# 如果为 False，则不生成索引。
html_use_index = False

# 如果为 True，则将索引拆分为每个字母的单独页面。
# html_split_index = False

# 如果为 True，则在页面中添加到 reST 源代码的链接。
# html_show_sourcelink = True

# 如果为 True，则会生成一个 OpenSearch 描述文件，并且所有页面都会包含一个指向它的 <link> 标签。
# 此选项的值必须是生成的 HTML 的基本 URL。
# html_use_opensearch = ''

# 如果非空，则这是 HTML 文件的文件名后缀（例如 ".xhtml"）。
# html_file_suffix = ''

# HTML 帮助生成器的输出文件基础名称。
htmlhelp_basename = "scikit-learndoc"

# 如果为 True，则将 reST 源代码包含在 HTML 构建中作为 _sources/name。
html_copy_source = True

# 将变量添加到模板中。
html_context = {}

# 查找最新版本的发布重点，并将其放入 HTML 上下文以供 index.html 使用。
release_highlights_dir = Path("..") / "examples" / "release_highlights"
# 找到具有最新版本号的发布重点
latest_highlights = sorted(release_highlights_dir.glob("plot_release_highlights_*.py"))[
    -1
]
latest_highlights = latest_highlights.with_suffix("").name
html_context["release_highlights"] = (
    f"auto_examples/release_highlights/{latest_highlights}"
)

# 从名称中获取版本号，假设发布重点的命名形式为 plot_release_highlights_0_22_0
highlight_version = ".".join(latest_highlights.split("_")[-3:-1])
html_context["release_highlights_version"] = highlight_version

# 重定向字典，将旧链接映射到新链接
redirects = {
    "documentation": "index",
    "contents": "index",
    "preface": "index",
    "modules/classes": "api/index",
    "auto_examples/feature_selection/plot_permutation_test_for_classification": (
        "auto_examples/model_selection/plot_permutation_tests_for_classification"
    ),
    "modules/model_persistence": "model_persistence",
    "auto_examples/linear_model/plot_bayesian_ridge": (
        "auto_examples/linear_model/plot_ard"
    ),
    "auto_examples/model_selection/grid_search_text_feature_extraction.py": (
        "auto_examples/model_selection/plot_grid_search_text_feature_extraction.py"
    ),
    "auto_examples/miscellaneous/plot_changed_only_pprint_parameter": (
        "auto_examples/miscellaneous/plot_estimator_representation"
    ),
    "auto_examples/decomposition/plot_beta_divergence": (
        "auto_examples/applications/plot_topics_extraction_with_nmf_lda"
    ),
    "auto_examples/svm/plot_svm_nonlinear": "auto_examples/svm/plot_svm_kernels",
    "auto_examples/ensemble/plot_adaboost_hastie_10_2": (
        "auto_examples/ensemble/plot_adaboost_multiclass"
    ),
    "auto_examples/decomposition/plot_pca_3d": (
        "auto_examples/decomposition/plot_pca_iris"
    ),
}
    "auto_examples/exercises/plot_cv_digits.py": (
        "auto_examples/model_selection/plot_nested_cross_validation_iris.py"
    ),
    "tutorial/machine_learning_map/index.html": "machine_learning_map/index.html",
}

# 将重定向字典赋值给 HTML 上下文变量
html_context["redirects"] = redirects

# 遍历重定向字典中的每个旧链接，将其添加到 HTML 附加页面字典中，并指向 redirects.html
for old_link in redirects:
    html_additional_pages[old_link] = "redirects.html"

# 设置是否为开发版本的标志，参见 https://github.com/scikit-learn/scikit-learn/pull/22550
html_context["is_devrelease"] = parsed_version.is_devrelease


# -- Options for LaTeX output ------------------------------------------------

# LaTeX 输出选项
latex_elements = {
    # 纸张尺寸 ('letterpaper' 或 'a4paper').
    # 'papersize': 'letterpaper',
    # 字体大小 ('10pt', '11pt' 或 '12pt').
    # 'pointsize': '10pt',
    # LaTeX 导言部分的额外内容.
    "preamble": r"""
        \usepackage{amsmath}\usepackage{amsfonts}\usepackage{bm}
        \usepackage{morefloats}\usepackage{enumitem} \setlistdepth{10}
        \let\oldhref\href
        \renewcommand{\href}[2]{\oldhref{#1}{\hbox{#2}}}
        """
}

# 将文档树分组成 LaTeX 文件的元组列表
latex_documents = [
    (
        "contents",
        "user_guide.tex",
        "scikit-learn user guide",
        "scikit-learn developers",
        "manual",
    ),
]

# 放置在标题页顶部的图像文件的名称（相对于当前目录）
latex_logo = "logos/scikit-learn-logo.png"

# 附加到所有手册的附录文档
# latex_appendices = []

# 如果为 False，则不生成模块索引
latex_domain_indices = False

trim_doctests_flags = True

# intersphinx 配置
intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "skops": ("https://skops.readthedocs.io/en/stable/", None),
}

# 解析版本号并设置 Binder 分支
v = parse(release)
if v.release is None:
    raise ValueError(
        "Ill-formed version: {!r}. Version should follow PEP440".format(version)
    )

if v.is_devrelease:
    binder_branch = "main"
else:
    major, minor = v.release[:2]
    binder_branch = "{}.{}.X".format(major, minor)


class SubSectionTitleOrder:
    """按子标题排序示例图库.

    假设每个子节都有一个 README.txt 存在，并使用装饰的横线 '---' 作为子节名称.
    """

    def __init__(self, src_dir):
        self.src_dir = src_dir
        self.regex = re.compile(r"^([\w ]+)\n-", re.MULTILINE)

    def __repr__(self):
        return "<%s>" % (self.__class__.__name__,)
    # 定义一个特殊的方法，使对象可以像函数一样被调用，传入目录路径参数
    def __call__(self, directory):
        # 将源目录和给定目录名连接并规范化路径
        src_path = os.path.normpath(os.path.join(self.src_dir, directory))

        # 如果目录的基本名称为 "release_highlights"，强制返回字符串 "0"
        if os.path.basename(src_path) == "release_highlights":
            return "0"

        # 组合目录路径和 "README.txt" 文件名，形成 README 文件的完整路径
        readme = os.path.join(src_path, "README.txt")

        try:
            # 尝试以只读方式打开 README 文件
            with open(readme, "r") as f:
                # 读取文件内容
                content = f.read()
        except FileNotFoundError:
            # 如果文件不存在，则返回当前目录名
            return directory

        # 使用预定义的正则表达式搜索 README 文件内容
        title_match = self.regex.search(content)
        # 如果找到匹配项，则返回第一个捕获组的内容
        if title_match is not None:
            return title_match.group(1)
        # 否则返回当前目录名
        return directory
class SKExampleTitleSortKey(ExampleTitleSortKey):
    """Sorts release highlights based on version number."""
    
    def __call__(self, filename):
        # 调用父类方法以获取标题
        title = super().__call__(filename)
        # 定义标题前缀
        prefix = "plot_release_highlights_"

        # 如果文件名不以指定前缀开头，则使用标题进行排序
        if not str(filename).startswith(prefix):
            return title

        # 提取文件名中的主要和次要版本号，并将其转换为浮点数
        major_minor = filename[len(prefix):].split("_")[:2]
        version_float = float(".".join(major_minor))

        # 返回负数以确保最新版本的高亮显示首先出现
        return -version_float


def notebook_modification_function(notebook_content, notebook_filename):
    # 将笔记本内容转换为字符串
    notebook_content_str = str(notebook_content)
    # 定义警告模板字符串
    warning_template = "\n".join(
        [
            "<div class='alert alert-{message_class}'>",
            "",
            "# JupyterLite warning",
            "",
            "{message}",
            "</div>",
        ]
    )

    # 设置警告类别和消息内容
    message_class = "warning"
    message = (
        "Running the scikit-learn examples in JupyterLite is experimental and you may"
        " encounter some unexpected behavior.\n\nThe main difference is that imports"
        " will take a lot longer than usual, for example the first `import sklearn` can"
        " take roughly 10-20s.\n\nIf you notice problems, feel free to open an"
        " [issue](https://github.com/scikit-learn/scikit-learn/issues/new/choose)"
        " about it."
    )

    # 使用警告模板格式化消息内容
    markdown = warning_template.format(message_class=message_class, message=message)

    # 创建一个空的虚拟笔记本内容字典
    dummy_notebook_content = {"cells": []}
    # 将 Markdown 单元格添加到虚拟笔记本内容中
    add_markdown_cell(dummy_notebook_content, markdown)

    # 初始化代码行列表
    code_lines = []

    # 检查笔记本内容字符串中是否包含特定模块名，如果是，则添加相应的安装命令
    if "seaborn" in notebook_content_str:
        code_lines.append("%pip install seaborn")
    if "plotly.express" in notebook_content_str:
        code_lines.append("%pip install plotly")
    if "skimage" in notebook_content_str:
        code_lines.append("%pip install scikit-image")
    if "polars" in notebook_content_str:
        code_lines.append("%pip install polars")
    if "fetch_" in notebook_content_str:
        # 对于特定模式的字符串，扩展代码行列表以包含多条命令
        code_lines.extend(
            [
                "%pip install pyodide-http",
                "import pyodide_http",
                "pyodide_http.patch_all()",
            ]
        )
    # 总是导入 matplotlib 和 pandas，以避免 Pyodide 在函数内部导入的限制
    code_lines.extend(["import matplotlib", "import pandas"])

    # 如果存在代码行，则在代码行列表的开头添加注释行
    if code_lines:
        code_lines = ["# JupyterLite-specific code"] + code_lines
        # 将代码行列表合并成一个代码字符串
        code = "\n".join(code_lines)
        # 将代码单元格添加到虚拟笔记本内容中
        add_code_cell(dummy_notebook_content, code)

    # 将虚拟笔记本内容的单元格与原始笔记本内容的单元格合并
    notebook_content["cells"] = (
        dummy_notebook_content["cells"] + notebook_content["cells"]
    )


# 获取 scikit-learn 默认全局配置
default_global_config = sklearn.get_config()


def reset_sklearn_config(gallery_conf, fname):
    """将 scikit-learn 配置重置为默认值。"""
    # 使用默认全局配置重置 scikit-learn 配置
    sklearn.set_config(**default_global_config)


# 定义 scikit-learn 示例的目录
sg_examples_dir = "../examples"
# 定义 Sphinx Gallery 示例目录
sg_gallery_dir = "auto_examples"
# 定义 Sphinx Gallery 配置
sphinx_gallery_conf = {
    # 模块文档的名称，这里是sklearn
    "doc_module": "sklearn",
    # 后向引用的目录路径，连接在"modules/generated"下面
    "backreferences_dir": os.path.join("modules", "generated"),
    # 是否显示内存信息，这里设置为False
    "show_memory": False,
    # 参考链接的URL字典，这里只有"sklearn"一个键，值为None
    "reference_url": {"sklearn": None},
    # 示例代码目录的列表，包含sg_examples_dir的路径
    "examples_dirs": [sg_examples_dir],
    # 画廊目录的列表，包含sg_gallery_dir的路径
    "gallery_dirs": [sg_gallery_dir],
    # 子节标题的排序方式，使用sg_examples_dir的SubSectionTitleOrder函数
    "subsection_order": SubSectionTitleOrder(sg_examples_dir),
    # 在子节内部的排序键，使用SKExampleTitleSortKey
    "within_subsection_order": SKExampleTitleSortKey,
    # Binder配置字典，指定了org、repo、binderhub_url、branch、dependencies和use_jupyter_lab等项
    "binder": {
        "org": "scikit-learn",
        "repo": "scikit-learn",
        "binderhub_url": "https://mybinder.org",
        "branch": binder_branch,
        "dependencies": "./binder/requirements.txt",
        "use_jupyter_lab": True,
    },
    # 避免生成过多的交叉链接
    "inspect_global_variables": False,
    # 移除配置中的注释
    "remove_config_comments": True,
    # 是否显示画廊示例，这里设置为True
    "plot_gallery": "True",
    # 推荐系统的配置，启用且配置了n_examples和min_df等参数
    "recommender": {"enable": True, "n_examples": 4, "min_df": 12},
    # 重置的模块列表，包括"matplotlib"、"seaborn"和reset_sklearn_config函数
    "reset_modules": ("matplotlib", "seaborn", reset_sklearn_config),
}
if with_jupyterlite:
    # 如果配置中包含 with_jupyterlite 标志，则设置 sphinx_gallery_conf 中的 "jupyterlite" 字段，
    # 用于指定笔记本修改函数。
    sphinx_gallery_conf["jupyterlite"] = {
        "notebook_modification_function": notebook_modification_function
    }

# Secondary sidebar configuration for pages generated by sphinx-gallery

# 对于由 sphinx-gallery 生成的页面的次级侧边栏配置

# 对于 gallery 的索引页面和每个嵌套部分，我们通过指定空列表（无组件）来隐藏次级侧边栏，
# 因为这些页面没有有意义的页面目录，而且由于它们是生成的，所以 "sourcelink" 也没有用处。

# 对于每个示例页面，我们保留由 "**" 键指定的默认配置 ["page-toc", "sourcelink"]。
# 这些页面需要 "page-toc"。"sourcelink" 也是必要的，否则当 "page-toc" 为空时，次级侧边栏将退化，
# 而脚本 `sphinxext/move_gallery_links.py` 将失败（它假设次级侧边栏的存在）。
# 脚本将在最后移除 "sourcelink"。

html_theme_options["secondary_sidebar_items"][f"{sg_gallery_dir}/index"] = []
for sub_sg_dir in (Path(".") / sg_examples_dir).iterdir():
    if sub_sg_dir.is_dir():
        html_theme_options["secondary_sidebar_items"][
            f"{sg_gallery_dir}/{sub_sg_dir.name}/index"
        ] = []


# The following dictionary contains the information used to create the
# thumbnails for the front page of the scikit-learn home page.
# key: first image in set
# values: (number of plot in set, height of thumbnail)
# 下面的字典包含用于创建 scikit-learn 主页前页面缩略图的信息。
# 键：集合中的第一张图片
# 值：（集合中的图数，缩略图高度）
carousel_thumbs = {"sphx_glr_plot_classifier_comparison_001.png": 600}


# enable experimental module so that experimental estimators can be
# discovered properly by sphinx
# 启用实验模块，以便 sphinx 能够正确发现实验性评估器
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.experimental import enable_halving_search_cv  # noqa


def make_carousel_thumbs(app, exception):
    """produces the final resized carousel images"""
    """生成最终调整大小的轮播图片"""
    if exception is not None:
        return
    print("Preparing carousel images")

    image_dir = os.path.join(app.builder.outdir, "_images")
    for glr_plot, max_width in carousel_thumbs.items():
        image = os.path.join(image_dir, glr_plot)
        if os.path.exists(image):
            c_thumb = os.path.join(image_dir, glr_plot[:-4] + "_carousel.png")
            sphinx_gallery.gen_rst.scale_image(image, c_thumb, max_width, 190)


def filter_search_index(app, exception):
    if exception is not None:
        return

    # searchindex only exist when generating html
    # 仅在生成 html 时才存在 searchindex
    if app.builder.name != "html":
        return

    print("Removing methods from search index")

    searchindex_path = os.path.join(app.builder.outdir, "searchindex.js")
    with open(searchindex_path, "r") as f:
        searchindex_text = f.read()

    searchindex_text = re.sub(r"{__init__.+?}", "{}", searchindex_text)
    searchindex_text = re.sub(r"{__call__.+?}", "{}", searchindex_text)

    with open(searchindex_path, "w") as f:
        f.write(searchindex_text)


# Config for sphinx_issues

# we use the issues path for PRs since the issues URL will forward
# 我们在 PRs 中使用 issues 路径，因为 issues URL 将会转发
# GitHub 上 scikit-learn 项目的路径
issues_github_path = "scikit-learn/scikit-learn"


def disable_plot_gallery_for_linkcheck(app):
    # 如果当前使用 linkcheck 构建器，则禁用示例图库
    if app.builder.name == "linkcheck":
        sphinx_gallery_conf["plot_gallery"] = "False"


def setup(app):
    # 当使用 linkcheck 时，通过设置较小的优先级来避免运行示例
    # （默认优先级为 500，sphinx-gallery 也使用 builder-inited 事件）
    app.connect("builder-inited", disable_plot_gallery_for_linkcheck, priority=50)

    # 在为单个页面创建 HTML 之前触发
    app.connect("html-page-context", add_js_css_files)

    # 用于在代码示例中隐藏/显示提示符
    app.connect("build-finished", make_carousel_thumbs)
    app.connect("build-finished", filter_search_index)


# 以下用于 sphinx.ext.linkcode，提供到 GitHub 的链接
linkcode_resolve = make_linkcode_resolve(
    "sklearn",
    (
        "https://github.com/scikit-learn/"
        "scikit-learn/blob/{revision}/"
        "{package}/{path}#L{lineno}"
    ),
)

# 忽略特定的用户警告类别和消息
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=(
        "Matplotlib is currently using agg, which is a"
        " non-GUI backend, so cannot show the figure."
    ),
)
# 如果环境变量 SKLEARN_WARNINGS_AS_ERRORS 不为 "0"，则将警告转换为错误
if os.environ.get("SKLEARN_WARNINGS_AS_ERRORS", "0") != "0":
    turn_warnings_into_errors()

# 将函数与类名大小写无关的文件名映射关系
autosummary_filename_map = {
    "sklearn.cluster.dbscan": "dbscan-function",
    "sklearn.covariance.oas": "oas-function",
    "sklearn.decomposition.fastica": "fastica-function",
}


# 配置 sphinxext.opengraph

ogp_site_url = "https://scikit-learn/stable/"
ogp_image = "https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png"
ogp_use_first_image = True
ogp_site_name = "scikit-learn"

# 配置 linkcheck，检查文档中的坏链接

# 忽略 'whats_new' 中的所有链接，以避免多次请求 GitHub 和击中速率限制
linkcheck_exclude_documents = [r"whats_new/.*"]

# 设置默认超时时间，使一些站点的链接能更快失败
linkcheck_timeout = 10

# 允许从 doi.org 进行重定向
linkcheck_allowed_redirects = {r"https://doi.org/.+": r".*"}

# 忽略一些已知错误的链接
linkcheck_ignore = [
    # 忽略指向本地 html 文件的链接，如在 image 指令的 :target: 字段中
    r"^..?/",
    # 忽略指向特定 pdf 页面的链接，因为 linkcheck 无法处理它们（'utf-8' 解码错误）
    r"http://www.utstat.toronto.edu/~rsalakhu/sta4273/notes/Lecture2.pdf#page=.*",
    (
        "https://www.fordfoundation.org/media/2976/roads-and-bridges"
        "-the-unseen-labor-behind-our-digital-infrastructure.pdf#page=.*"
    ),
    # 忽略错误标记的链接
    (
        "https://www.researchgate.net/publication/"
        "233096619_A_Dendrite_Method_for_Cluster_Analysis"
    ),
    # 列表中包含多个链接地址，用逗号分隔
    (
        "https://www.researchgate.net/publication/221114584_Random_Fourier"
        "_Approximations_for_Skewed_Multiplicative_Histogram_Kernels"
    ),
    (
        "https://www.researchgate.net/publication/4974606_"
        "Hedonic_housing_prices_and_the_demand_for_clean_air"
    ),
    (
        "https://www.researchgate.net/profile/Anh-Huy-Phan/publication/220241471_Fast_"
        "Local_Algorithms_for_Large_Scale_Nonnegative_Matrix_and_Tensor_Factorizations"
    ),
    # 单独的 DOI 链接
    "https://doi.org/10.13140/RG.2.2.35280.02565",
    (
        "https://www.microsoft.com/en-us/research/uploads/prod/2006/01/"
        "Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf"
    ),
    "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-99-87.pdf",
    "https://microsoft.com/",
    "https://www.jstor.org/stable/2984099",
    "https://stat.uw.edu/sites/default/files/files/reports/2000/tr371.pdf",
    # 下面是一些失效的链接，作为参考案例
    # 失效的链接：bestofmedia.com
    "http://www.bestofmedia.com",
    # 失效的链接：data-publica.com
    "http://www.data-publica.com/",
    # 失效的链接：livelovely.com
    "https://livelovely.com",
    # 失效的链接：mars.com
    "https://www.mars.com/global",
    # 失效的链接：yhat.com
    "https://www.yhat.com",
    # 忽略某些动态创建的锚点链接，详见 GitHub 问题说明
    r"https://github.com/conda-forge/miniforge#miniforge",
    r"https://github.com/joblib/threadpoolctl/"
    "#setting-the-maximum-size-of-thread-pools",
    r"https://stackoverflow.com/questions/5836335/"
    "consistently-create-same-random-numpy-array/5837352#comment6712034_5837352",
# Config for sphinx-remove-toctrees
# 定义用于 sphinx-remove-toctrees 的配置

remove_from_toctrees = ["metadata_routing.rst"]
# 需要从目录树中移除的文件列表

# Use a browser-like user agent to avoid some "403 Client Error: Forbidden for
# url" errors. This is taken from the variable navigator.userAgent inside a
# browser console.
# 使用类似浏览器的用户代理以避免某些“403 Client Error: Forbidden for url”错误。
# 此内容源自浏览器控制台中的 navigator.userAgent 变量。

user_agent = (
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:100.0) Gecko/20100101 Firefox/100.0"
)
# 设置用户代理字符串，模拟浏览器行为，防止某些错误

# Use Github token from environment variable to avoid Github rate limits when
# checking Github links
# 使用环境变量中的 Github 令牌以避免在检查 Github 链接时遇到速率限制问题
github_token = os.getenv("GITHUB_TOKEN")

if github_token is None:
    linkcheck_request_headers = {}
else:
    linkcheck_request_headers = {
        "https://github.com/": {"Authorization": f"token {github_token}"},
    }
# 如果未设置 GitHub 令牌，则不添加请求头；否则，使用设置的令牌进行认证

# -- Convert .rst.template files to .rst ---------------------------------------

from api_reference import API_REFERENCE, DEPRECATED_API_REFERENCE

from sklearn._min_dependencies import dependent_packages
# 导入依赖包信息

# If development build, link to local page in the top navbar; otherwise link to the
# development version; see https://github.com/scikit-learn/scikit-learn/pull/22550
# 如果是开发版本，则在顶部导航栏链接到本地页面；否则链接到开发版本；参见链接中的详细说明
if parsed_version.is_devrelease:
    development_link = "developers/index"
else:
    development_link = "https://scikit-learn.org/dev/developers/index.html"
# 根据版本信息设置开发链接地址

# Define the templates and target files for conversion
# Each entry is in the format (template name, file name, kwargs for rendering)
# 定义转换的模板和目标文件
# 每个条目格式为（模板名称，文件名称，用于渲染的关键字参数）

rst_templates = [
    ("index", "index", {"development_link": development_link}),
    (
        "min_dependency_table",
        "min_dependency_table",
        {"dependent_packages": dependent_packages},
    ),
    (
        "min_dependency_substitutions",
        "min_dependency_substitutions",
        {"dependent_packages": dependent_packages},
    ),
    (
        "api/index",
        "api/index",
        {
            "API_REFERENCE": sorted(API_REFERENCE.items(), key=lambda x: x[0]),
            "DEPRECATED_API_REFERENCE": sorted(
                DEPRECATED_API_REFERENCE.items(), key=lambda x: x[0], reverse=True
            ),
        },
    ),
]
# 定义需要转换的模板和目标文件列表

# Convert each module API reference page
# 逐个转换每个模块的 API 参考页面
for module in API_REFERENCE:
    rst_templates.append(
        (
            "api/module",
            f"api/{module}",
            {"module": module, "module_info": API_REFERENCE[module]},
        )
    )
# 添加每个模块的 API 参考页面到转换列表中

# Convert the deprecated API reference page (if there exists any)
# 转换已弃用的 API 参考页面（如果存在的话）
if DEPRECATED_API_REFERENCE:
    rst_templates.append(
        (
            "api/deprecated",
            "api/deprecated",
            {
                "DEPRECATED_API_REFERENCE": sorted(
                    DEPRECATED_API_REFERENCE.items(), key=lambda x: x[0], reverse=True
                )
            },
        )
    )
# 添加已弃用的 API 参考页面到转换列表中

for rst_template_name, rst_target_name, kwargs in rst_templates:
    # Read the corresponding template file into jinja2
    # 读取相应的模板文件到 jinja2
    with (Path(".") / f"{rst_template_name}.rst.template").open(
        "r", encoding="utf-8"
    ) as f:
        t = jinja2.Template(f.read())

    # Render the template and write to the target
    # 渲染模板并写入目标文件
    # 使用指定的文件名 `rst_target_name` 生成一个新的文件路径，并以写模式打开该文件
    with (Path(".") / f"{rst_target_name}.rst").open("w", encoding="utf-8") as f:
        # 将使用模板 `t` 渲染得到的内容写入打开的文件中，使用关键字参数 `kwargs`
        f.write(t.render(**kwargs))
```