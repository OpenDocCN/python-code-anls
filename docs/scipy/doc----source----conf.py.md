# `D:\src\scipysrc\scipy\doc\source\conf.py`

```
# 导入必要的标准库和第三方模块
import math
import os
from os.path import relpath, dirname  # 导入部分 os.path 模块内容
import re  # 导入正则表达式模块
import sys  # 导入系统相关功能模块
import warnings  # 导入警告处理模块
from datetime import date  # 从 datetime 模块导入 date 类
from docutils import nodes  # 导入文档工具模块中的 nodes 类
from docutils.parsers.rst import Directive  # 从 reStructuredText 解析模块导入 Directive 类

# 导入 intersphinx_registry 模块中的 get_intersphinx_mapping 函数
from intersphinx_registry import get_intersphinx_mapping
# 导入 matplotlib 及其 pyplot 子模块
import matplotlib
import matplotlib.pyplot as plt
# 从 numpydoc.docscrape_sphinx 模块导入 SphinxDocString 类
from numpydoc.docscrape_sphinx import SphinxDocString
# 从 sphinx.util 模块中导入 inspect 函数
from sphinx.util import inspect

# 导入 scipy 及其子模块
import scipy
# 从 scipy._lib._util 模块导入 _rng_html_rewrite 函数
from scipy._lib._util import _rng_html_rewrite
# 从 scipy._lib.uarray 模块导入 ua 对象
import scipy._lib.uarray as ua
# 从 scipy.stats._distn_infrastructure 模块导入 rv_generic 类
from scipy.stats._distn_infrastructure import rv_generic
# 从 scipy.stats._multivariate 模块导入 multi_rv_generic 类

# 对 inspect.isdescriptor 进行临时性修改
old_isdesc = inspect.isdescriptor
inspect.isdescriptor = (lambda obj: old_isdesc(obj)
                        and not isinstance(obj, ua._Function))

# 设置环境变量，用于构建 scipy.fft 文档
os.environ['_SCIPY_BUILDING_DOC'] = 'True'

# -----------------------------------------------------------------------------
# 一般配置
# -----------------------------------------------------------------------------

# 在这里添加任何 Sphinx 扩展模块的名称，作为字符串。它们可以是与 Sphinx 一起提供的扩展（命名为 'sphinx.ext.*'）
# 或自定义的扩展。
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 导入 numpydoc.docscrape 模块，用于文档解析
import numpydoc.docscrape as np_docscrape  # noqa: E402

# 设置 Sphinx 的扩展模块列表
extensions = [
    'sphinx.ext.autodoc',  # 自动文档生成
    'sphinx.ext.autosummary',  # 自动生成摘要
    'sphinx.ext.coverage',  # 测试覆盖率生成
    'sphinx.ext.mathjax',  # 数学公式支持
    'sphinx.ext.intersphinx',  # 其他文档的交叉引用支持
    'numpydoc',  # NumPy 风格的文档支持
    'sphinx_design',  # Sphinx 设计扩展
    'scipyoptdoc',  # SciPy 优化文档
    'doi_role',  # DOI 角色支持
    'matplotlib.sphinxext.plot_directive',  # Matplotlib 图表指令支持
    'myst_nb',  # MyST 格式的 Jupyter 笔记本支持
    'jupyterlite_sphinx',  # JupyterLite Sphinx 支持
]


# 对于可能存在的 matplotlibrc 配置进行设置，以免影响到其他部分
matplotlib.use('agg')
plt.ioff()

# 添加包含模板的路径，相对于当前目录
templates_path = ['_templates']

# 源文件名的后缀
source_suffix = '.rst'

# 主 toctree 文档
master_doc = 'index'

# 一般替换。
project = 'SciPy'
# 版权声明
copyright = '2008-%s, The SciPy community' % date.today().year

# 默认情况下的 |version| 和 |release| 替换，也用于构建文档中的各种其他位置。
version = re.sub(r'\.dev.*$', r'.dev', scipy.__version__)
release = version

# 如果环境变量中有 'CIRCLE_JOB'，并且 'CIRCLE_BRANCH' 不是 'main' 分支，则使用 CircleCI 的分支作为版本号
if os.environ.get('CIRCLE_JOB', False) and \
        os.environ.get('CIRCLE_BRANCH', '') != 'main':
    version = os.environ['CIRCLE_BRANCH']
    release = version

# 打印项目名称及版本号
print(f"{project} (VERSION {version})")

# 替换 |today| 的两种选项之一：设置 today 为某个非假值，则使用它：
# today = ''
# 否则，使用 today_fmt 作为 strftime 调用的格式。
today_fmt = '%B %d, %Y'

# 不包括在构建中的文档列表。
# unused_docs = []

# reST 默认角色（用于此标记：`text`），用于所有文档。
default_role = "autolink"
# List of directories, relative to source directories, that shouldn't be searched
# for source files.
exclude_dirs = []

# Glob-style patterns to exclude files, '**.ipynb' matches any IPython Notebook file.
exclude_patterns = [
    "**.ipynb",
]

# If true, parentheses will be appended to cross-reference text for functions.
add_function_parentheses = False

# If true, prepend the current module name to all description unit titles.
# This feature is commented out by default.
#add_module_names = True

# If true, display sectionauthor and moduleauthor directives in the output.
# These directives are ignored by default.
# show_authors = False

# The name of the Pygments syntax highlighting style to use.
# pygments_style = 'sphinx'

# Ensure all internal links are checked strictly.
nitpicky = True

# List of (type, name) tuples to ignore in nitpicky mode.
# These entries suppress warnings about missing references for specific classes or methods.
nitpick_ignore = [
    ("py:class", "a shallow copy of D"),
    ("py:class", "a set-like object providing a view on D's keys"),
    ("py:class", "a set-like object providing a view on D's items"),
    ("py:class", "an object providing a view on D's values"),
    ("py:class", "None.  Remove all items from D."),
    ("py:class", "(k, v), remove and return some (key, value) pair as a"),
    ("py:class", "None.  Update D from dict/iterable E and F."),
    ("py:class", "v, remove specified key and return the corresponding value."),
]

# Reset warnings to default settings.
warnings.resetwarnings()

# Filter warnings to raise errors for all warnings.
warnings.filterwarnings('error')

# Allow and display warnings from the 'sphinx' module.
warnings.filterwarnings('default', module='sphinx')

# Ignore specific warnings by message regex for various modules.
warnings.filterwarnings('ignore', message=".*OpenSSL\.rand is deprecated", category=DeprecationWarning)
warnings.filterwarnings('ignore', message=".*distutils Version", category=DeprecationWarning)

# Ignore a specific warning related to matplotlib and pyparsing.
warnings.filterwarnings('ignore', message="Exception creating Regex for oneOf.*", category=SyntaxWarning)

# Allow warnings with specified messages for examples and suppress them after first occurrence.
# TODO: These should ideally be fixed in the code.
for key in (
        'invalid escape sequence',  # numpydoc 0.8 has some bad escape chars
        'The integral is probably divergent',  # stats.mielke example
        'underflow encountered in square',  # signal.filtfilt underflow
        'underflow encountered in multiply',  # scipy.spatial.HalfspaceIntersection
        'underflow encountered in nextafter',  # tuterial/interpolate.rst
        'underflow encountered in exp',  # stats.skewnorm, stats.norminvgauss, stats.gaussian_kde
        ):
    warnings.filterwarnings('once', message='.*' + key)
# docutils warnings when using notebooks (see gh-17322)
# 在使用笔记本时的docutils警告（参见gh-17322），这些警告有望在不久的将来被移除

for key in (
    r"The frontend.OptionParser class will be replaced",
    r"The frontend.Option class will be removed",
    ):
    # 忽略特定的DeprecationWarning警告消息，这些警告与OptionParser和Option类相关
    warnings.filterwarnings('ignore', message=key, category=DeprecationWarning)

warnings.filterwarnings(
    'ignore',
    message=r'.*is obsoleted by Node.findall()',
    category=PendingDeprecationWarning,
)
# 忽略特定的PendingDeprecationWarning警告消息，与Node.findall()相关

warnings.filterwarnings(
    'ignore',
    message=r'There is no current event loop',
    category=DeprecationWarning,
)
# 忽略特定的DeprecationWarning警告消息，指示当前没有事件循环

# TODO: remove after gh-19228 resolved:
# 待解决gh-19228后删除此处内容，目前忽略特定的DeprecationWarning警告消息，与路径相关的已过时功能有关
warnings.filterwarnings(
    'ignore',
    message=r'.*path is deprecated.*',
    category=DeprecationWarning,
)

# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

# 指定HTML主题
html_theme = 'pydata_sphinx_theme'

# 指定HTML页面的logo和favicon
html_logo = '_static/logo.svg'
html_favicon = '_static/favicon.ico'

# 指定HTML页面的侧边栏内容
html_sidebars = {
    "index": "search-button-field",
    "**": ["search-button-field", "sidebar-nav-bs"]
}

# 配置HTML主题选项
html_theme_options = {
    "github_url": "https://github.com/scipy/scipy",
    "twitter_url": "https://twitter.com/SciPy_team",
    "header_links_before_dropdown": 6,
    "icon_links": [],
    "logo": {
        "text": "SciPy",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
    "switcher": {
        "json_url": "https://scipy.github.io/devdocs/_static/version_switcher.json",
        "version_match": version,
    },
    "show_version_warning_banner": True,
    "secondary_sidebar_items": ["page-toc"],
    # The service https://plausible.io is used to gather simple
    # and privacy-friendly analytics for the site. The dashboard can be accessed
    # at https://analytics.scientific-python.org/docs.scipy.org
    # The Scientific-Python community is hosting and managing the account.
    # 配置用于网站的简单和隐私友好的分析服务
    "analytics": {
        "plausible_analytics_domain": "docs.scipy.org",
        "plausible_analytics_url": "https://views.scientific-python.org/js/script.js",
    },
}

# 如果版本号中包含'dev'，则调整HTML主题选项中的内容
if 'dev' in version:
    html_theme_options["switcher"]["version_match"] = "development"
    html_theme_options["show_version_warning_banner"] = False

if 'versionwarning' in tags:  # noqa: F821
    # 特定于docs.scipy.org部署的内容
    # 参见https://github.com/scipy/docs.scipy.org/blob/main/_static/versionwarning.js_t
    src = ('var script = document.createElement("script");\n'
           'script.type = "text/javascript";\n'
           'script.src = "/doc/_static/versionwarning.js";\n'
           'document.head.appendChild(script);')
    html_context = {
        'VERSIONCHECK_JS': src
    }
    html_js_files = ['versioncheck.js']

# 指定HTML页面的标题
html_title = f"{project} v{version} Manual"
# 指定HTML页面的静态资源路径
html_static_path = ['_static']
# 指定HTML页面的最后更新时间格式
html_last_updated_fmt = '%b %d, %Y'

# 指定HTML页面的CSS文件
html_css_files = [
    "scipy.css",
    "try_examples.css",
]
# 定义一个空字典，用于存储额外的 HTML 页面
html_additional_pages = {}

# 设置为使用模块索引
html_use_modindex = True

# 禁用域索引
html_domain_indices = False

# 禁止复制源文件到输出目录
html_copy_source = False

# 设置生成的 HTML 文件后缀为 .html
html_file_suffix = '.html'

# 设置 MathJax 的路径
mathjax_path = "scipy-mathjax/MathJax.js?config=scipy-mathjax"

# -----------------------------------------------------------------------------
# Intersphinx 配置
# -----------------------------------------------------------------------------
# 使用 get_intersphinx_mapping 函数获取外部包的 intersphinx 映射
intersphinx_mapping = get_intersphinx_mapping(
    packages={"python", "numpy", "neps", "matplotlib", "asv", "statsmodels", "mpmath"}
)

# -----------------------------------------------------------------------------
# Numpy 扩展
# -----------------------------------------------------------------------------

# 如果想从 XML 文件进行虚拟导入来支持所有自动文档
phantom_import_file = 'dump.xml'

# 为示例部分生成图表
numpydoc_use_plots = True

# 定义额外的公共方法列表
np_docscrape.ClassDoc.extra_public_methods = [
    '__call__', '__mul__', '__getitem__', '__len__',
]

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

# 自动生成摘要页
autosummary_generate = True

# 映射函数和类名相同但大小写不同的情况，避免生成重复的文档页
autosummary_filename_map = {
    "scipy.odr.odr": "odr-function",
    "scipy.signal.czt": "czt-function",
    "scipy.signal.ShortTimeFFT.t": "scipy.signal.ShortTimeFFT.t.lower",
}


# -----------------------------------------------------------------------------
# Autodoc
# -----------------------------------------------------------------------------

# 默认的自动文档选项配置
autodoc_default_options = {
    'inherited-members': None,
}

# 关闭类型提示
autodoc_typehints = 'none'


# -----------------------------------------------------------------------------
# Coverage checker
# -----------------------------------------------------------------------------

# 忽略的模块列表
coverage_ignore_modules = r"""
    """.split()

# 忽略的函数列表
coverage_ignore_functions = r"""
    test($|_) (some|all)true bitwise_not cumproduct pkgload
    generic\.
    """.split()

# 忽略的类列表
coverage_ignore_classes = r"""
    """.split()

# C 语言覆盖路径列表
coverage_c_path = []

# C 语言覆盖正则表达式字典
coverage_c_regexes = {}

# 忽略的 C 语言条目列表
coverage_ignore_c_items = {}


#------------------------------------------------------------------------------
# Matplotlib plot_directive options
#------------------------------------------------------------------------------

# 图表指令的前置代码
plot_pre_code = """
import warnings
# 针对多个警告关键字进行过滤设置，忽略特定警告消息
for key in (
        'interp2d` is deprecated',  # Deprecation of scipy.interpolate.interp2d
        'scipy.misc',  # scipy.misc deprecated in v1.10.0; use scipy.datasets
        '`kurtosistest` p-value may be',  # intentionally "bad" example in docstring
        ):
    warnings.filterwarnings(action='ignore', message='.*' + key + '.*')

# 导入 NumPy 库，并设置随机种子为 123
import numpy as np
np.random.seed(123)

# 设置绘图相关的变量
plot_include_source = True  # 包含源代码
plot_formats = [('png', 96)]  # 绘图格式为 PNG，分辨率为 96 dpi
plot_html_show_formats = False  # HTML 不显示绘图格式选项
plot_html_show_source_link = False  # HTML 不显示源代码链接

# 计算黄金比例 phi
phi = (math.sqrt(5) + 1)/2

# 设置字体大小为 13 像素，转换为英寸单位
font_size = 13*72/96.0  # 13 px

# 配置绘图参数
plot_rcparams = {
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': font_size,
    'figure.figsize': (3*phi, 3),  # 图形尺寸
    'figure.subplot.bottom': 0.2,
    'figure.subplot.left': 0.2,
    'figure.subplot.right': 0.9,
    'figure.subplot.top': 0.85,
    'figure.subplot.wspace': 0.4,
    'text.usetex': False,  # 不使用 LaTeX 渲染文本
}

# -----------------------------------------------------------------------------
# Notebook tutorials with MyST-NB
# -----------------------------------------------------------------------------

# 设置笔记本执行模式为自动
nb_execution_mode = "auto"

# 忽略由 jupyterlite-sphinx 生成的交互式示例笔记本
nb_execution_excludepatterns = ["_contents/*.ipynb"]

# 禁止在添加脚注时创建过渡语法
myst_footnote_transition = False

# 启用 MyST 扩展功能
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "substitution",
]

# 设置渲染 Markdown 的格式为 MyST
nb_render_markdown_format = "myst"
render_markdown_format = "myst"

# 修复 Jupyter 笔记本中 MathJax 对象的渲染
myst_update_mathjax = False

#------------------------------------------------------------------------------
# Interactive examples with jupyterlite-sphinx
#------------------------------------------------------------------------------

# 全局启用尝试示例功能
global_enable_try_examples = True

# 设置全局按钮文本和警告文本
try_examples_global_button_text = "Try it in your browser!"
try_examples_global_warning_text = (
    "SciPy's interactive examples with Jupyterlite are experimental and may"
    " not always work as expected. Execution of cells containing imports may"
    " result in large downloads (up to 60MB of content for the first import"
    " from SciPy). Load times when importing from SciPy may take roughly 10-20"
    " seconds. If you notice any problems, feel free to open an"
    " [issue](https://github.com/scipy/scipy/issues/new/choose)."
)

# -----------------------------------------------------------------------------
# Source code links
# -----------------------------------------------------------------------------

# 导入 inspect 模块，用于后续的源代码链接功能
import inspect  # noqa: E402

# 尝试导入三个可能的源代码链接模块，成功导入后将其加入扩展列表
for name in ['sphinx.ext.linkcode', 'linkcode', 'numpydoc.linkcode']:
    try:
        __import__(name)
        extensions.append(name)
        break
    # 如果发生 ImportError 异常，则忽略，什么都不做
    except ImportError:
        pass
else:
    print("NOTE: linkcode extension not found -- no links to source generated")



# 如果未找到 linkcode 扩展，则打印提示信息，表示未生成源代码的链接
else:
    print("NOTE: linkcode extension not found -- no links to source generated")



def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
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
        except Exception:
            return None

    # Use the original function object if it is wrapped.
    while hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__
    # SciPy's distributions are instances of *_gen. Point to this
    # class since it contains the implementation of all the methods.
    if isinstance(obj, (rv_generic, multi_rv_generic)):
        obj = obj.__class__
    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    startdir = os.path.abspath(os.path.join(dirname(scipy.__file__), '..'))
    fn = relpath(fn, start=startdir).replace(os.path.sep, '/')

    if fn.startswith('scipy/'):
        m = re.match(r'^.*dev0\+([a-f0-9]+)$', scipy.__version__)
        base_url = "https://github.com/scipy/scipy/blob"
        if m:
            return f"{base_url}/{m.group(1)}/{fn}{linespec}"
        elif 'dev' in scipy.__version__:
            return f"{base_url}/main/{fn}{linespec}"
        else:
            return f"{base_url}/v{scipy.__version__}/{fn}{linespec}"
    else:
        return None



# 定义一个函数 linkcode_resolve，用于确定 Python 对象对应的 URL
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
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
        except Exception:
            return None

    # Use the original function object if it is wrapped.
    while hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__
    # SciPy's distributions are instances of *_gen. Point to this
    # class since it contains the implementation of all the methods.
    if isinstance(obj, (rv_generic, multi_rv_generic)):
        obj = obj.__class__
    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    startdir = os.path.abspath(os.path.join(dirname(scipy.__file__), '..'))
    fn = relpath(fn, start=startdir).replace(os.path.sep, '/')

    if fn.startswith('scipy/'):
        m = re.match(r'^.*dev0\+([a-f0-9]+)$', scipy.__version__)
        base_url = "https://github.com/scipy/scipy/blob"
        if m:
            return f"{base_url}/{m.group(1)}/{fn}{linespec}"
        elif 'dev' in scipy.__version__:
            return f"{base_url}/main/{fn}{linespec}"
        else:
            return f"{base_url}/v{scipy.__version__}/{fn}{linespec}"
    else:
        return None



# 设置 SphinxDocString 的 _str_examples 属性，以便覆盖 numpydoc 渲染包含 rng 的示例逻辑
SphinxDocString._str_examples = _rng_html_rewrite(
    SphinxDocString._str_examples
)



class LegacyDirective(Directive):
    """
    Adapted from docutils/parsers/rst/directives/admonitions.py

    Uses a default text if the directive does not have contents. If it does,
    the default text is concatenated to the contents.

    """
    has_content = True
    node_class = nodes.admonition
    optional_arguments = 1



# 定义 LegacyDirective 类，它继承自 Directive 类
class LegacyDirective(Directive):
    """
    Adapted from docutils/parsers/rst/directives/admonitions.py

    Uses a default text if the directive does not have contents. If it does,
    the default text is concatenated to the contents.

    """
    # 指示该指令是否包含内容
    has_content = True
    # 指定生成的节点类为 admonition 类型
    node_class = nodes.admonition
    # 指定可选参数的数量为 1
    optional_arguments = 1
    def run(self):
        try:
            obj = self.arguments[0]  # 尝试获取参数列表中的第一个参数作为对象名
        except IndexError:
            # 如果参数列表为空，使用默认文本
            obj = "submodule"
        text = (f"This {obj} is considered legacy and will no longer receive "
                "updates. This could also mean it will be removed in future "
                "SciPy versions.")

        try:
            self.content[0] = text + " " + self.content[0]
        except IndexError:
            # 如果内容列表为空，使用默认文本，并获取源和行号
            source, lineno = self.state_machine.get_source_and_line(
                self.lineno
            )
            self.content.append(
                text,
                source=source,
                offset=lineno
            )
        text = '\n'.join(self.content)
        # 创建警告（admonition）节点，将由 `nested_parse` 填充内容
        admonition_node = self.node_class(rawsource=text)
        # 设置自定义标题
        title_text = "Legacy"
        textnodes, _ = self.state.inline_text(title_text, self.lineno)
        title = nodes.title(title_text, '', *textnodes)
        # 添加标题到警告节点
        admonition_node += title
        # 为 CSS 样式设置自定义类
        admonition_node['classes'] = ['admonition-legacy']
        # 解析指令内容并填充到警告节点中
        self.state.nested_parse(self.content, self.content_offset,
                                admonition_node)
        # 返回填充好内容的警告节点作为列表
        return [admonition_node]
# 定义一个函数 setup，用于设置应用程序
def setup(app):
    # 向应用程序添加一个指令，指令名称为 "legacy"，指令实现为 LegacyDirective 类
    app.add_directive("legacy", LegacyDirective)
```