# `D:\src\scipysrc\matplotlib\doc\conf.py`

```
# 导入必要的库和模块
from datetime import datetime, timezone  # 导入日期时间处理模块
import logging  # 导入日志记录模块
import os  # 导入操作系统相关模块
from pathlib import Path  # 导入路径操作相关模块
import shutil  # 导入文件操作相关模块
import subprocess  # 导入子进程管理模块
import sys  # 导入系统相关模块
import time  # 导入时间相关模块
from urllib.parse import urlsplit, urlunsplit  # 导入URL解析相关模块
import warnings  # 导入警告处理模块

from packaging.version import parse as parse_version  # 导入版本解析模块
import sphinx  # 导入Sphinx文档生成工具
import yaml  # 导入YAML配置文件解析模块

import matplotlib  # 导入Matplotlib绘图库


# 输出正在构建 Matplotlib 文档的版本信息
print(f"Building Documentation for Matplotlib: {matplotlib.__version__}")

# Release mode enables optimizations and other related options.
# 检查是否为发布版本构建（启用优化和其他相关选项）
is_release_build = tags.has('release')  # noqa

# 检查是否在 CircleCI 环境下运行
CIRCLECI = 'CIRCLECI' in os.environ
# 检查是否将此构建部署到 matplotlib.org/devdocs
# 这里是 .circleci/deploy-docs.sh 中相同逻辑的副本
DEVDOCS = (
    CIRCLECI and
    (os.environ.get("CIRCLE_PROJECT_USERNAME") == "matplotlib") and
    (os.environ.get("CIRCLE_BRANCH") == "main") and
    (not os.environ.get("CIRCLE_PULL_REQUEST", "").startswith(
        "https://github.com/matplotlib/matplotlib/pull")))


def _parse_skip_subdirs_file():
    """
    从 .mpl_skip_subdirs.yaml 中读取不需要构建的子目录列表，
    用于 `make html-skip-subdirs` 命令。子目录路径相对于顶层目录。
    注意不能跳过 'users' 目录，因为它包含目录内容，但可以跳过其子目录。
    使用这个功能可以加快部分构建速度。
    """
    default_skip_subdirs = [
        'users/prev_whats_new/*', 'users/explain/*', 'api/*', 'gallery/*',
        'tutorials/*', 'plot_types/*', 'devel/*']
    try:
        with open(".mpl_skip_subdirs.yaml", 'r') as fin:
            print('Reading subdirectories to skip from',
                  '.mpl_skip_subdirs.yaml')
            out = yaml.full_load(fin)
        return out['skip_subdirs']
    except FileNotFoundError:
        # 创建默认的 .mpl_skip_subdirs.yaml 文件：
        with open(".mpl_skip_subdirs.yaml", 'w') as fout:
            yamldict = {'skip_subdirs': default_skip_subdirs,
                        'comment': 'For use with make html-skip-subdirs'}
            yaml.dump(yamldict, fout)
        print('Skipping subdirectories, but .mpl_skip_subdirs.yaml',
              'not found so creating a default one. Edit this file',
              'to customize which directories are included in build.')

        return default_skip_subdirs


skip_subdirs = []
# 如果命令行参数中包含 'skip_sub_dirs=1'，则调用 _parse_skip_subdirs_file() 函数
if 'skip_sub_dirs=1' in sys.argv:
    skip_subdirs = _parse_skip_subdirs_file()

# 使用 SOURCE_DATE_EPOCH 解析年份，如果未设置则使用当前时间。
# 获取环境变量 'SOURCE_DATE_EPOCH' 的值，如果不存在则使用当前时间戳，将其转换为 datetime 对象后取出年份
sourceyear = datetime.fromtimestamp(
    int(os.environ.get('SOURCE_DATE_EPOCH', time.time())), timezone.utc).year

# 如果您的扩展位于另一个目录中，请在此处添加。如果目录相对于文档根目录，请使用 os.path.abspath 将其转换为绝对路径
sys.path.append(os.path.abspath('.'))
sys.path.append('.')

# 通用配置
# ---------------------

# 如果未显式捕获警告，警告应导致文档构建失败。这对于在画廊中消除已弃用的用法特别有用。
warnings.filterwarnings('error', append=True)

# 在此处添加任何 Sphinx 扩展模块的名称作为字符串。它们可以是随 Sphinx 一起提供的扩展（命名为 'sphinx.ext.*'）或自定义的扩展。
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx',
    'sphinx.ext.ifconfig',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'numpydoc',  # 需要在 autodoc 之后加载
    'sphinx_gallery.gen_gallery',
    'matplotlib.sphinxext.mathmpl',
    'matplotlib.sphinxext.plot_directive',
    'matplotlib.sphinxext.roles',
    'matplotlib.sphinxext.figmpl_directive',
    'sphinxcontrib.inkscapeconverter',
    'sphinxext.github',
    'sphinxext.math_symbol_table',
    'sphinxext.missing_references',
    'sphinxext.mock_gui_toolkits',
    'sphinxext.skip_deprecated',
    'sphinxext.redirect_from',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx_tags',
]

# 需要排除的文档模式列表
exclude_patterns = [
    'api/prev_api_changes/api_changes_*/*',
    '**/*inc.rst',
    'users/explain/index.rst'  # 页面没有内容，但是 sphinx gallery 需要它
]

# 添加要跳过的子目录到排除模式列表中
exclude_patterns += skip_subdirs

# 检查依赖项的函数
def _check_dependencies():
    # 创建扩展名和相应模块名的映射字典，其中扩展名从 extensions 列表中获取
    names = {
        **{ext: ext.split(".")[0] for ext in extensions},
        # 显式列出不是扩展的依赖项，或者它们的 PyPI 包名称与（顶层）模块名称不匹配。
        "colorspacious": 'colorspacious',
        "mpl_sphinx_theme": 'mpl_sphinx_theme',
        "sphinxcontrib.inkscapeconverter": 'sphinxcontrib-svg2pdfconverter',
    }
    missing = []
    # 检查每个依赖项是否可以导入，如果不能则添加到 missing 列表中
    for name in names:
        try:
            __import__(name)
        except ImportError:
            missing.append(names[name])
    # 如果存在缺失的依赖项，则抛出 ImportError 异常
    if missing:
        raise ImportError(
            "The following dependencies are missing to build the "
            f"documentation: {', '.join(missing)}")

    # 调试 sphinx-pydata-theme 和 mpl-theme-version
    if 'mpl_sphinx_theme' not in missing:
        import pydata_sphinx_theme
        import mpl_sphinx_theme
        # 打印当前使用的 pydata sphinx theme 和 mpl sphinx theme 的版本
        print(f"pydata sphinx theme: {pydata_sphinx_theme.__version__}")
        print(f"mpl sphinx theme: {mpl_sphinx_theme.__version__}")
    # 检查是否存在名为 'dot' 的可执行文件，用于构建文档中的图形
    if shutil.which('dot') is None:
        # 如果找不到 'dot' 可执行文件，则抛出 OSError 异常并显示错误信息
        raise OSError(
            "No binary named dot - graphviz must be installed to build the "
            "documentation")
    
    # 检查是否存在名为 'latex' 的可执行文件，用于构建文档中的 LaTeX 内容
    if shutil.which('latex') is None:
        # 如果找不到 'latex' 可执行文件，则抛出 OSError 异常并显示错误信息
        raise OSError(
            "No binary named latex - a LaTeX distribution must be installed to build "
            "the documentation")
# 检查依赖项是否满足
_check_dependencies()

# 导入 sphinx_gallery 前先检查依赖项是否已经满足

# 导入 sphinx_gallery 模块
import sphinx_gallery

# 如果 sphinx_gallery 版本大于等于 0.16.0，则设置特定的变量值
if parse_version(sphinx_gallery.__version__) >= parse_version('0.16.0'):
    gallery_order_sectionorder = 'sphinxext.gallery_order.sectionorder'
    gallery_order_subsectionorder = 'sphinxext.gallery_order.subsectionorder'
    clear_basic_units = 'sphinxext.util.clear_basic_units'
    matplotlib_reduced_latex_scraper = 'sphinxext.util.matplotlib_reduced_latex_scraper'
else:
    # 如果版本小于 0.16.0，从 sphinxext 目录的 gallery_order 模块导入特定的类和函数
    # 以支持自定义排序部分和子部分的 gallery
    from sphinxext.gallery_order import (
        sectionorder as gallery_order_sectionorder,
        subsectionorder as gallery_order_subsectionorder)
    from sphinxext.util import clear_basic_units, matplotlib_reduced_latex_scraper

# 导入 gen_rst 函数用于后续的签名 monkey patch
from sphinx_gallery import gen_rst

# 忽略掉 plt.show() 调用在非 GUI 后端环境下可能出现的警告信息
warnings.filterwarnings('ignore', category=UserWarning,
                        message=r'(\n|.)*is non-interactive, and thus cannot be shown')

# 自动生成摘要
autosummary_generate = True
# 关闭自动文档中类型提示
autodoc_typehints = "none"

# 忽略掉来自于 importlib 模块中关于已弃用模块的警告信息，这些模块在移除后会自动消失
warnings.filterwarnings('ignore', category=DeprecationWarning,
                        module='importlib',  # used by sphinx.autodoc.importer
                        message=r'(\n|.)*module was deprecated.*')

# 在自动文档中显示函数签名的文档字符串
autodoc_docstring_signature = True
# 自动文档默认选项，所有成员都显示，包括未记录的成员
autodoc_default_options = {'members': None, 'undoc-members': None}

# 忽略掉来自 sphinx.util.inspect 模块中关于已弃用类级属性的警告信息
warnings.filterwarnings('ignore', category=DeprecationWarning,
                        module='sphinx.util.inspect')

# 启用详细提示模式
nitpicky = True
# 设置为 True 可以更新允许失败的标记
missing_references_write_json = False
# 设置为 True 可以忽略未使用的忽略警告
missing_references_warn_unused_ignores = False

# 设置 intersphinx 映射，指定外部文档链接
intersphinx_mapping = {
    'Pillow': ('https://pillow.readthedocs.io/en/stable/', None),
    'cycler': ('https://matplotlib.org/cycler/', None),
    'dateutil': ('https://dateutil.readthedocs.io/en/stable/', None),
    'ipykernel': ('https://ipykernel.readthedocs.io/en/latest/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'pytest': ('https://pytest.org/en/stable/', None),
    'python': ('https://docs.python.org/3/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'tornado': ('https://www.tornadoweb.org/en/stable/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
    'meson-python': ('https://meson-python.readthedocs.io/en/stable/', None),
    'pip': ('https://pip.pypa.io/en/stable/', None),
}
# 定义一个包含几个字符串的列表，每个字符串表示一个目录名
gallery_dirs = [f'{ed}' for ed in
                ['gallery', 'tutorials', 'plot_types', 'users/explain']
                if f'{ed}/*' not in skip_subdirs]

# 创建一个空列表用于存储修改后的目录名
example_dirs = []

# 遍历 gallery_dirs 列表中的每个目录名
for gd in gallery_dirs:
    # 替换目录名中的特定子字符串，生成新的目录名
    gd = gd.replace('gallery', 'examples').replace('users/explain', 'users_explain')
    # 将修改后的目录名添加到 example_dirs 列表中
    example_dirs += [f'../galleries/{gd}']

# 配置 Sphinx-Gallery 的设置
sphinx_gallery_conf = {
    'backreferences_dir': Path('api', '_as_gen'),
    # 对于本地和 CI 构建，跳过压缩图片的步骤
    'compress_images': ('thumbnails', 'images') if is_release_build else (),
    'doc_module': ('matplotlib', 'mpl_toolkits'),
    'examples_dirs': example_dirs,
    'filename_pattern': '^((?!sgskip).)*$',
    'gallery_dirs': gallery_dirs,
    'image_scrapers': (matplotlib_reduced_latex_scraper, ),
    'image_srcset': ["2x"],
    'junit': '../test-results/sphinx-gallery/junit.xml' if CIRCLECI else '',
    'matplotlib_animations': True,
    'min_reported_time': 1,
    'plot_gallery': 'True',  # sphinx-gallery/913
    'reference_url': {'matplotlib': None},
    'remove_config_comments': True,
    'reset_modules': ('matplotlib', clear_basic_units),
    'subsection_order': gallery_order_sectionorder,
    'thumbnail_size': (320, 224),
    'within_subsection_order': gallery_order_subsectionorder,
    'capture_repr': (),
    'copyfile_regex': r'.*\.rst',
}

# 如果命令行参数包含 'plot_gallery=0'
if 'plot_gallery=0' in sys.argv:
    # 不创建图库图片，抑制其他文档部分链接到这些图片时触发的警告

    # 定义一个过滤器函数来过滤掉特定的警告消息
    def gallery_image_warning_filter(record):
        msg = record.msg
        # 检查消息是否以特定的模式开头，如果是则返回 False
        for pattern in (sphinx_gallery_conf['gallery_dirs'] +
                        ['_static/constrained_layout']):
            if msg.startswith(f'image file not readable: {pattern}'):
                return False

        # 如果消息是特定的一条警告，则返回 False
        if msg == 'Could not obtain image size. :scale: option is ignored.':
            return False

        # 其他情况返回 True，保留该警告消息
        return True

    # 获取名为 'sphinx' 的日志记录器对象
    logger = logging.getLogger('sphinx')
    # 添加过滤器到日志记录器中
    logger.addFilter(gallery_image_warning_filter)

# Sphinx 标签配置
tags_create_tags = True
tags_page_title = "All tags"
tags_create_badges = True

# 定义标签的颜色映射
tags_badge_colors = {
    "animation": "primary",
    "component:*": "secondary",
    "event-handling": "success",
    "interactivity:*": "dark",
    "plot-type:*": "danger",
    "*": "light"  # 默认值
}

# 设置数学绘图的字体大小
mathmpl_fontsize = 11.0

# 设置数学绘图的图像源集
mathmpl_srcset = ['2x']

# 通过修改 gallery 头文件来包含搜索关键字
gen_rst.EXAMPLE_HEADER = """
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "{0}"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. meta::
        :keywords: codex

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_{1}>`
        to download the full example code.{2}

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_{1}:

"""

# 在此处添加包含模板的路径，相对于当前目录
# 模板路径列表，包含一个路径名'_templates'
templates_path = ['_templates']

# 源文件名后缀
source_suffix = '.rst'

# 源文件编码格式，默认为UTF-8
source_encoding = "utf-8"

# 顶层的toctree文档，Sphinx 4.0中更名为root_doc
root_doc = master_doc = 'index'

# 通用替换内容

try:
    # 从git中获取最新的SHA标识作为版本号
    SHA = subprocess.check_output(
        ['git', 'describe', '--dirty']).decode('utf-8').strip()
except (subprocess.CalledProcessError, FileNotFoundError):
    # 处理git未安装的情况，使用setuptools_scm的版本号作为替代
    SHA = matplotlib.__version__

# HTML上下文变量，包含文档版本信息
html_context = {
    "doc_version": SHA,
}

# 项目名称
project = 'Matplotlib'

# 版权信息，包含年份和开发团队信息
copyright = (
    '2002–2012 John Hunter, Darren Dale, Eric Firing, Michael Droettboom '
    'and the Matplotlib development team; '
    f'2012–{sourceyear} The Matplotlib development team'
)

# 默认的|version|和|release|替换内容，在文档的多处使用
version = matplotlib.__version__  # X.Y版本号
release = version  # 包含alpha/beta/rc标签的完整版本号

# |today|替换内容的日期格式
today_fmt = '%B %d, %Y'

# 不包含在构建中的文档列表
unused_docs = []

# 如果为True，在交叉引用文本中将添加括号
# add_function_parentheses = True

# 如果为True，在所有描述单元标题中将包含当前模块名称
# add_module_names = True

# 如果为True，输出中将显示sectionauthor和moduleauthor指令，默认忽略
# show_authors = False

# 用于Pygments（语法高亮）的样式名称
pygments_style = 'sphinx'

# 默认的角色
default_role = 'obj'

# 图表指令的配置
# ----------------

# 根据构建目标确定构建的plot_formats
formats = {'html': ('png', 100), 'latex': ('pdf', 100)}
plot_formats = [formats[target] for target in ['html', 'latex']
                if target in sys.argv] or list(formats.values())
# 用于<img>标签的srcset参数，用于生成2倍大小的图像
plot_srcset = ['2x']

# GitHub扩展配置

github_project_url = "https://github.com/matplotlib/matplotlib/"


# HTML输出选项
# -------------

def add_html_cache_busting(app, pagename, templatename, context, doctree):
    """
    添加CSS和JavaScript资源的缓存破坏查询。

    如果路径不是绝对路径（即来自'_static'），将Matplotlib版本作为查询添加到HTML中的链接引用。
    """
    # 导入需要的模块和类
    from sphinx.builders.html import Stylesheet, JavaScript

    # 从上下文中获取传入的 css_tag 和 js_tag 函数
    css_tag = context['css_tag']
    js_tag = context['js_tag']

    # 定义用于带有缓存破坏功能的 css_tag 函数
    def css_tag_with_cache_busting(css):
        # 检查 css 是否是 Stylesheet 类的实例且具有文件名
        if isinstance(css, Stylesheet) and css.filename is not None:
            # 解析文件名的 URL
            url = urlsplit(css.filename)
            # 如果 URL 中没有指定主机名和查询参数
            if not url.netloc and not url.query:
                # 在 URL 中添加 SHA 参数
                url = url._replace(query=SHA)
                # 创建新的 Stylesheet 对象，带有更新后的 URL 和原有属性
                css = Stylesheet(urlunsplit(url), priority=css.priority,
                                 **css.attributes)
        # 调用原始的 css_tag 函数并返回结果
        return css_tag(css)

    # 定义用于带有缓存破坏功能的 js_tag 函数
    def js_tag_with_cache_busting(js):
        # 检查 js 是否是 JavaScript 类的实例且具有文件名
        if isinstance(js, JavaScript) and js.filename is not None:
            # 解析文件名的 URL
            url = urlsplit(js.filename)
            # 如果 URL 中没有指定主机名和查询参数
            if not url.netloc and not url.query:
                # 在 URL 中添加 SHA 参数
                url = url._replace(query=SHA)
                # 创建新的 JavaScript 对象，带有更新后的 URL 和原有属性
                js = JavaScript(urlunsplit(url), priority=js.priority,
                                **js.attributes)
        # 调用原始的 js_tag 函数并返回结果
        return js_tag(js)

    # 更新上下文中的 css_tag 和 js_tag 函数为带有缓存破坏功能的新函数
    context['css_tag'] = css_tag_with_cache_busting
    context['js_tag'] = js_tag_with_cache_busting
# HTML和HTML帮助页面使用的样式表。文件必须存在于Sphinx的static/路径中，或者自定义路径html_static_path中的一个。
html_css_files = [
    "mpl.css",
]

# Sphinx文档使用的主题名称。
html_theme = "mpl_sphinx_theme"

# 这组Sphinx文档的名称。如果为None，则默认为"<project> v<release> documentation"。
# html_title = None

# 放置在侧边栏顶部的图像文件名（位于静态路径内）。
html_theme_options = {
    "navbar_links": "internal",
    # 在本地和CI构建中跳过collapse_navigation以提高性能。
    "collapse_navigation": not is_release_build,
    "show_prev_next": False,
    "switcher": {
        # 向switcher.json的URL添加唯一查询。服务器会忽略它，但浏览器会将其用作缓存的一部分键，
        # 因此当进行新的meso版本发布时，switcher将在stable和devdocs上“迅速”更新。
        "json_url": (
            "https://output.circle-artifacts.com/output/job/"
            f"{os.environ['CIRCLE_WORKFLOW_JOB_ID']}/artifacts/"
            f"{os.environ['CIRCLE_NODE_INDEX']}"
            "/doc/build/html/_static/switcher.json" if CIRCLECI and not DEVDOCS else
            f"https://matplotlib.org/devdocs/_static/switcher.json?{SHA}"
        ),
        "version_match": (
            # 要显示的起始版本。这必须在switcher.json中。
            # 我们要么转到“stable”，要么转到“devdocs”。
            'stable' if matplotlib.__version_info__.releaselevel == 'final'
            else 'devdocs')
    },
    "navbar_end": ["theme-switcher", "version-switcher", "mpl_icon_links"],
    "navbar_persistent": ["search-button"],
    "footer_start": ["copyright", "sphinx-version", "doc_version"],
    # 我们覆盖了来自pydata-sphinx-theme的公告模板，这里特殊的值表示使用未发布的横幅。
    # 如果我们需要一个实际的公告，那么就像平常一样在这里放置文本。
    "announcement": "unreleased" if not is_release_build else "",
    "show_version_warning_banner": True,
}

# 是否包含分析数据。如果是发布构建，则包含在html_theme_options中。
include_analytics = is_release_build
if include_analytics:
    html_theme_options["analytics"] = {
        "plausible_analytics_domain": "matplotlib.org",
        "plausible_analytics_url": "https://views.scientific-python.org/js/script.js"
    }

# 添加任何包含自定义静态文件（如样式表）的路径，相对于当前目录。
# 它们在内置静态文件之后复制，因此文件名为"default.css"将覆盖内置"default.css"。
html_static_path = ['_static']

# 如果非空，则是生成的HTML文件的文件名后缀。默认为".html"。
html_file_suffix = '.html'

# 这使得此站点上所有页面的规范链接为html_baseurl。
html_baseurl = 'https://matplotlib.org/stable/'
# 如果不为空字符串，则在每个页面底部插入一个“Last updated on:”时间戳，使用指定的strftime格式。
html_last_updated_fmt = '%b %d, %Y'

# 索引页面的内容模板。
html_index = 'index.html'

# 自定义侧边栏模板，将文档名称映射到模板名称。
# html_sidebars = {}

# 自定义侧边栏模板，将页面名称映射到模板。
html_sidebars = {
    "index": [
        # 'sidebar_announcement.html',
        "cheatsheet_sidebar.html",
        "donate_sidebar.html",
    ],
    # 'release_notes'页面没有侧边栏，因为该页面只是一个子页面链接的集合。侧边栏会重复所有子页面的标题，
    # 实际上会重复页面的所有内容。
    "users/release_notes": ["empty_sidebar.html"],
    # '**': ['localtoc.html', 'pagesource.html']
}

# 不包含文档源文件的链接。
html_show_sourcelink = False

# 仅复制相关代码，不包括 '>>>' 提示符。
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

# 如果为True，在HTML文档中添加索引。
html_use_index = False

# 如果为True，除了一般索引外，还生成特定于域的索引。
# 例如Python域，这是全局模块索引。
html_domain_index = False

# 如果为True，reST源文件将作为 _sources/<name> 包含在HTML构建中。
# html_copy_source = True

# 如果为True，将输出一个OpenSearch描述文件，并且所有页面都将包含一个指向它的 <link> 标签。
html_use_opensearch = 'https://matplotlib.org/stable'

# HTML帮助生成器的输出文件基础名称。
htmlhelp_basename = 'Matplotlibdoc'

# 使用排版引号字符。
smartquotes = False

# 网站图标的路径。
html_favicon = '_static/favicon.ico'

# LaTeX输出选项
# ------------------------

# 纸张大小（'letter' 或 'a4'）。
latex_paper_size = 'letter'

# 将文档树分组为LaTeX文件。
# 元组列表：
#   (源起始文件，目标名称，标题，作者，
#    文档类 [howto/manual])
latex_documents = [
    (root_doc, 'Matplotlib.tex', 'Matplotlib',
     'John Hunter\\and Darren Dale\\and Eric Firing\\and Michael Droettboom'
     '\\and and the matplotlib development team', 'manual'),
]

# 放置在标题页顶部的图像文件的名称（相对于此目录）。
latex_logo = None

# 使用Unicode感知的LaTeX引擎。
latex_engine = 'xelatex'  # 或 'lualatex'

latex_elements = {}

# 保持babel在xelatex中的使用（Sphinx默认为polyglossia）。
# 如果删除或更改此键，则必须清理latex构建目录。
latex_elements['babel'] = r'\usepackage{babel}'

# 字体配置
# 修复fontspec将“转换为PDF中的右卷曲引号的问题。
# 参考 https://github.com/sphinx-doc/sphinx/pull/6888/
latex_elements['fontenc'] = r'''
\usepackage{fontspec}
\defaultfontfeatures[\rmfamily,\sffamily,\ttfamily]{}
'''

# Sphinx 2.0默认采用GNU FreeFont，但不包含所有
# the Unicode codepoints needed for the section about Mathtext
# "Writing mathematical expressions"

# 设置 LaTeX 元素中的字体包，用于支持数学表达式
latex_elements['fontpkg'] = r"""
\IfFontExistsTF{XITS}{
 \setmainfont{XITS}  # 如果 XITS 字体存在，则设置为主要字体
}{
 \setmainfont{XITS}[  # 否则使用 XITS 字体的不同样式
  Extension      = .otf,
  UprightFont    = *-Regular,
  ItalicFont     = *-Italic,
  BoldFont       = *-Bold,
  BoldItalicFont = *-BoldItalic,
]}
\IfFontExistsTF{FreeSans}{
 \setsansfont{FreeSans}  # 如果 FreeSans 字体存在，则设置为无衬线字体
}{
 \setsansfont{FreeSans}[  # 否则使用 FreeSans 字体的不同样式
  Extension      = .otf,
  UprightFont    = *,
  ItalicFont     = *Oblique,
  BoldFont       = *Bold,
  BoldItalicFont = *BoldOblique,
]}
\IfFontExistsTF{FreeMono}{
 \setmonofont{FreeMono}  # 如果 FreeMono 字体存在，则设置为等宽字体
}{
 \setmonofont{FreeMono}[  # 否则使用 FreeMono 字体的不同样式
  Extension      = .otf,
  UprightFont    = *,
  ItalicFont     = *Oblique,
  BoldFont       = *Bold,
  BoldItalicFont = *BoldOblique,
]}
% needed for \mathbb (blackboard alphabet) to actually work
\usepackage{unicode-math}  # 使用 unicode-math 宏包以实际支持 \mathbb（黑板粗体字母）
\IfFontExistsTF{XITS Math}{
 \setmathfont{XITS Math}  # 如果 XITS Math 数学字体存在，则设置为数学字体
}{
 \setmathfont{XITSMath-Regular}[  # 否则使用 XITSMath-Regular 数学字体的扩展
  Extension      = .otf,
]}
"""

# Fix fancyhdr complaining about \headheight being too small
# 修复 fancyhdr 对 \headheight 过小的抱怨问题
latex_elements['passoptionstopackages'] = r"""
    \PassOptionsToPackage{headheight=14pt}{geometry}  # 传递选项到 geometry 宏包，设置 headheight 为 14pt
"""

# Additional stuff for the LaTeX preamble.
# LaTeX 前言的其他内容

# 显示目录中的部分和章节
latex_elements['preamble'] = r"""
   % Show Parts and Chapters in Table of Contents
   \setcounter{tocdepth}{0}
   % One line per author on title page
   \DeclareRobustCommand{\and}%
     {\end{tabular}\kern-\tabcolsep\\\begin{tabular}[t]{c}}%
   \usepackage{etoolbox}
   \AtBeginEnvironment{sphinxthebibliography}{\appendix\part{Appendices}}
   \usepackage{expdlist}
   \let\latexdescription=\description
   \def\description{\latexdescription{}{} \breaklabel}
   % But expdlist old LaTeX package requires fixes:
   % 1) remove extra space
   \makeatletter
   \patchcmd\@item{{\@breaklabel} }{{\@breaklabel}}{}{}
   \makeatother
   % 2) fix bug in expdlist's way of breaking the line after long item label
   \makeatletter
   \def\breaklabel{%
       \def\@breaklabel{%
           \leavevmode\par
           % now a hack because Sphinx inserts \leavevmode after term node
           \def\leavevmode{\def\leavevmode{\unhbox\voidb@x}}%
      }%
   }
   \makeatother
"""

# Sphinx 1.5 provides this to avoid "too deeply nested" LaTeX error
# and usage of "enumitem" LaTeX package is unneeded.
# Value can be increased but do not set it to something such as 2048
# which needlessly would trigger creation of thousands of TeX macros
# Sphinx 1.5 提供了此功能以避免 "too deeply nested" 的 LaTeX 错误，
# 不需要使用 "enumitem" LaTeX 宏包
# 可以增加这个值，但不要设置得太大，如 2048，否则会不必要地触发数千个 TeX 宏的创建
latex_elements['maxlistdepth'] = '10'

# 设置文档使用的字体大小为 11 磅
latex_elements['pointsize'] = '11pt'

# Better looking general index in PDF
# PDF 中更美观的一般索引显示
latex_elements['printindex'] = r'\footnotesize\raggedright\printindex'

# Documents to append as an appendix to all manuals.
# 作为附录附加到所有手册的文档
latex_appendices = []

# If false, no module index is generated.
# 如果为 false，则不生成模块索引
latex_use_modindex = True

# 设置 LaTeX 顶层章节的分级为 'part'
latex_toplevel_sectioning = 'part'

# Show both class-level docstring and __init__ docstring in class
# documentation
# 在类文档中显示类级别的文档字符串和 __init__ 方法的文档字符串
autoclass_content = 'both'

texinfo_documents = [
    # 定义一个包含多个字段的元组，描述了一个文档或资源的信息
    (root_doc, 'matplotlib', 'Matplotlib Documentation',
     'John Hunter@*Darren Dale@*Eric Firing@*Michael Droettboom@*'
     'The matplotlib development team',
     'Matplotlib', "Python plotting package", 'Programming',
     1),
# numpydoc config

# 设置是否显示类成员，默认为 False
numpydoc_show_class_members = False

# 定义继承图的属性，设置图的大小和样式
inheritance_graph_attrs = dict(size='1000.0', splines='polyline')

# 定义继承图中节点的属性，设置节点的高度、边距和线条宽度
inheritance_node_attrs = dict(height=0.02, margin=0.055, penwidth=1, width=0.01)

# 定义继承图中边的属性，设置线条宽度
inheritance_edge_attrs = dict(penwidth=1)

# 设置使用的 Graphviz 可执行文件路径
graphviz_dot = shutil.which('dot')

# 设置 Graphviz 输出格式为 SVG
graphviz_output_format = 'svg'

# -----------------------------------------------------------------------------
# Source code links
# -----------------------------------------------------------------------------
# 是否链接到 GitHub
link_github = True

# 如果链接到 GitHub，则导入 inspect 模块和链接代码解析函数
if link_github:
    import inspect
    
    extensions.append('sphinx.ext.linkcode')

    def linkcode_resolve(domain, info):
        """
        确定 Python 对象对应的 URL
        """
        if domain != 'py':
            return None
        
        modname = info['module']
        fullname = info['fullname']
        
        submod = sys.modules.get(modname)
        if submod is None:
            return None
        
        obj = submod
        for part in fullname.split('.'):
            try:
                obj = getattr(obj, part)
            except AttributeError:
                return None
        
        if inspect.isfunction(obj):
            obj = inspect.unwrap(obj)
        try:
            fn = inspect.getsourcefile(obj)
        except TypeError:
            fn = None
        if not fn or fn.endswith('__init__.py'):
            try:
                fn = inspect.getsourcefile(sys.modules[obj.__module__])
            except (TypeError, AttributeError, KeyError):
                fn = None
        if not fn:
            return None
        
        try:
            source, lineno = inspect.getsourcelines(obj)
        except (OSError, TypeError):
            lineno = None
        
        # 构造行号范围字符串，用于指向具体代码行
        linespec = (f"#L{lineno:d}-L{lineno + len(source) - 1:d}"
                    if lineno else "")
        
        # 设置起始目录为 matplotlib 的上一级目录
        startdir = Path(matplotlib.__file__).parent.parent
        try:
            fn = os.path.relpath(fn, start=startdir).replace(os.path.sep, '/')
        except ValueError:
            return None
        
        # 如果文件路径不以 'matplotlib/' 或 'mpl_toolkits/' 开头，则返回 None
        if not fn.startswith(('matplotlib/', 'mpl_toolkits/')):
            return None
        
        # 解析 matplotlib 版本号
        version = parse_version(matplotlib.__version__)
        tag = 'main' if version.is_devrelease else f'v{version.public}'
        
        # 返回 GitHub 上具体代码位置的 URL
        return ("https://github.com/matplotlib/matplotlib/blob"
                f"/{tag}/lib/{fn}{linespec}")
else:
    # 如果不链接到 GitHub，则添加视图代码扩展
    extensions.append('sphinx.ext.viewcode')

# -----------------------------------------------------------------------------
# Sphinx setup
# -----------------------------------------------------------------------------
def setup(app):
    # 根据版本号是否包含 'post', 'dev', 'alpha', 'beta' 等字样来确定构建类型
    if any(st in version for st in ('post', 'dev', 'alpha', 'beta')):
        bld_type = 'dev'
    else:
        bld_type = 'rel'
    
    # 添加配置值 'skip_sub_dirs' 到应用
    app.add_config_value('skip_sub_dirs', 0, '')
    # 添加配置值到Sphinx应用程序，配置键为'releaselevel'，对应的值为'bld_type'，作用域为'env'
    app.add_config_value('releaselevel', bld_type, 'env')
    
    # 如果Sphinx版本信息的前两位小于(7, 1)，则连接到'html-page-context'事件，
    # 调用add_html_cache_busting函数，并设置优先级为1000
    if sphinx.version_info[:2] < (7, 1):
        app.connect('html-page-context', add_html_cache_busting, priority=1000)
```