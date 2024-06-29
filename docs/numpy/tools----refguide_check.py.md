# `.\numpy\tools\refguide_check.py`

```
"""
refguide_check.py [OPTIONS] [-- ARGS]

- Check for a NumPy submodule whether the objects in its __all__ dict
  correspond to the objects included in the reference guide.
- Check docstring examples
- Check example blocks in RST files

Example of usage::

    $ python tools/refguide_check.py

Note that this is a helper script to be able to check if things are missing;
the output of this script does need to be checked manually.  In some cases
objects are left out of the refguide for a good reason (it's an alias of
another function, or deprecated, or ...)

Another use of this helper script is to check validity of code samples
in docstrings::

    $ python tools/refguide_check.py --doctests ma

or in RST-based documentations::

    $ python tools/refguide_check.py --rst doc/source

"""

import copy  # 导入copy模块，用于复制对象
import doctest  # 导入doctest模块，用于执行文档字符串中的测试
import inspect  # 导入inspect模块，用于检查对象的属性和方法
import io  # 导入io模块，用于处理流
import os  # 导入os模块，用于与操作系统交互
import re  # 导入re模块，用于正则表达式操作
import shutil  # 导入shutil模块，用于文件和目录操作
import sys  # 导入sys模块，用于系统相关的参数和功能
import tempfile  # 导入tempfile模块，用于创建临时文件和目录
import warnings  # 导入warnings模块，用于警告控制

import docutils.core  # 导入docutils.core模块，用于处理reStructuredText文档
from argparse import ArgumentParser  # 导入ArgumentParser类，用于命令行参数解析
from contextlib import contextmanager, redirect_stderr  # 导入上下文管理器和错误重定向函数
from doctest import NORMALIZE_WHITESPACE, ELLIPSIS, IGNORE_EXCEPTION_DETAIL  # 导入doctest中的常量

from docutils.parsers.rst import directives  # 导入reStructuredText解析器中的指令处理器

import sphinx  # 导入sphinx模块，用于文档生成工具
import numpy as np  # 导入numpy模块，并使用np作为别名

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'doc', 'sphinxext'))
# 将sphinxext目录添加到系统路径，以便导入自定义Sphinx扩展

from numpydoc.docscrape_sphinx import get_doc_object  # 导入获取文档对象的函数

SKIPBLOCK = doctest.register_optionflag('SKIPBLOCK')
# 注册一个名为SKIPBLOCK的自定义选项标志，用于doctest模块的扩展功能

# 启用特定的Sphinx指令
from sphinx.directives.other import SeeAlso, Only  # 导入Sphinx其他指令
directives.register_directive('seealso', SeeAlso)  # 注册seealso指令
directives.register_directive('only', Only)  # 注册only指令

BASE_MODULE = "numpy"  # 定义基础模块名称为numpy

PUBLIC_SUBMODULES = [
    "f2py",
    "linalg",
    "lib",
    "lib.format",
    "lib.mixins",
    "lib.recfunctions",
    "lib.scimath",
    "lib.stride_tricks",
    "lib.npyio",
    "lib.introspect",
    "lib.array_utils",
    "fft",
    "char",
    "rec",
    "ma",
    "ma.extras",
    "ma.mrecords",
    "polynomial",
    "polynomial.chebyshev",
    "polynomial.hermite",
    "polynomial.hermite_e",
    "polynomial.laguerre",
    "polynomial.legendre",
    "polynomial.polynomial",
    "matrixlib",
    "random",
    "strings",
    "testing",
]
# 定义公共子模块列表，包含NumPy模块中的各个子模块名称

# 这些模块的文档包含在父模块中
OTHER_MODULE_DOCS = {
    'fftpack.convolve': 'fftpack',
    'io.wavfile': 'io',
    'io.arff': 'io',
}
# 指定这些模块的文档内容包含在父模块中，以字典形式存储

# 这些名称已知会导致doctest失败，故意保留这种状态
# 例如，有时可以接受伪代码等情况
#
# 可选地，通过将字典值设置为方法名称的集合来跳过方法的子集
DOCTEST_SKIPDICT = {
    # NumPy文档字符串中从SciPy导入内容的情况:
    'numpy.lib.vectorize': None,
    'numpy.random.standard_gamma': None,
    'numpy.random.gamma': None,
    'numpy.random.vonmises': None,
    'numpy.random.power': None,
    'numpy.random.zipf': None,
    # NumPy文档字符串中从其他第三方库导入内容的情况:
    'numpy._core.from_dlpack': None,
}
# 定义一个字典，包含已知会导致doctest失败的函数或方法名称
    # 禁用对特定类的导入，因为在 doctest 中与远程或本地文件 IO 会出现问题：
    'numpy.lib.npyio.DataSource': None,
    'numpy.lib.Repository': None,
# 跳过非 numpy RST 文件和历史发布说明
# 任何单目录完全匹配将跳过该目录及其所有子目录。
# 任何完全匹配（如'doc/release'）将扫描子目录但跳过匹配目录中的文件。
# 任何文件名将跳过该文件。
RST_SKIPLIST = [
    'scipy-sphinx-theme',
    'sphinxext',
    'neps',
    'changelog',
    'doc/release',
    'doc/source/release',
    'doc/release/upcoming_changes',
    'c-info.ufunc-tutorial.rst',
    'c-info.python-as-glue.rst',
    'f2py.getting-started.rst',
    'f2py-examples.rst',
    'arrays.nditer.cython.rst',
    'how-to-verify-bug.rst',
    # 参见 PR 17222，这些应该修复
    'basics.dispatch.rst',
    'basics.subclassing.rst',
    'basics.interoperability.rst',
    'misc.rst',
    'TESTS.rst'
]

# 这些名称不需要在所有 REFERENCE GUIDE 中必须出现，尽管在 autosummary:: 列表中
REFGUIDE_ALL_SKIPLIST = [
    r'scipy\.sparse\.linalg',
    r'scipy\.spatial\.distance',
    r'scipy\.linalg\.blas\.[sdczi].*',
    r'scipy\.linalg\.lapack\.[sdczi].*',
]

# 这些名称不需要在 autosummary:: 列表中出现，尽管在 ALL 中
REFGUIDE_AUTOSUMMARY_SKIPLIST = [
    # 注意: NumPy 是否应该在 autosummary 列表和 __all__ 之间有更好的匹配？暂时，TR 没有被说服这是一个优先事项 - 专注于执行/修正文档字符串
    r'numpy\.*',
]
# 在 scipy.signal 命名空间中废弃的窗口函数
for name in ('barthann', 'bartlett', 'blackmanharris', 'blackman', 'bohman',
             'boxcar', 'chebwin', 'cosine', 'exponential', 'flattop',
             'gaussian', 'general_gaussian', 'hamming', 'hann', 'hanning',
             'kaiser', 'nuttall', 'parzen', 'slepian', 'triang', 'tukey'):
    REFGUIDE_AUTOSUMMARY_SKIPLIST.append(r'scipy\.signal\.' + name)

# 是否拥有 matplotlib，默认为 False
HAVE_MATPLOTLIB = False
    """
    The `names_dict` is updated by reference and accessible in calling method

    Parameters
    ----------
    module : ModuleType
        The module, whose docstrings is to be searched
    names_dict : dict
        Dictionary which contains module name as key and a set of found
        function names and directives as value

    Returns
    -------
    None
    """

    # 定义用于匹配文档字符串中函数和指令名称的正则表达式模式列表
    patterns = [
        r"^\s\s\s([a-z_0-9A-Z]+)(\s+-+.*)?$",  # 匹配函数名或指令名格式
        r"^\.\. (?:data|function)::\s*([a-z_0-9A-Z]+)\s*$"  # 匹配数据或函数指令格式
    ]

    # 如果模块名为'scipy.constants'，添加特定于该模块的正则表达式模式
    if module.__name__ == 'scipy.constants':
        patterns += ["^``([a-z_0-9A-Z]+)``"]

    # 编译所有正则表达式模式
    patterns = [re.compile(pattern) for pattern in patterns]

    # 获取模块的名称
    module_name = module.__name__

    # 遍历模块的文档字符串的每一行
    for line in module.__doc__.splitlines():
        # 检查是否匹配模块名称的注释行，并更新模块名称
        res = re.search(r"^\s*\.\. (?:currentmodule|module):: ([a-z0-9A-Z_.]+)\s*$", line)
        if res:
            module_name = res.group(1)
            continue

        # 遍历所有正则表达式模式，匹配当前行
        for pattern in patterns:
            res = re.match(pattern, line)
            if res is not None:
                # 提取匹配的函数或指令名称
                name = res.group(1)
                # 构建模块名和函数/指令名的完整条目
                entry = '.'.join([module_name, name])
                # 将函数或指令名称添加到names_dict中对应模块名的集合中
                names_dict.setdefault(module_name, set()).add(name)
                break
# 返回一个经过处理后的 __all__ 字典的副本，移除了无关的条目
def get_all_dict(module):
    if hasattr(module, "__all__"):
        # 如果模块有 __all__ 属性，则深拷贝它的值
        all_dict = copy.deepcopy(module.__all__)
    else:
        # 否则，深拷贝模块的所有属性名，并移除以 "_" 开头的属性名
        all_dict = copy.deepcopy(dir(module))
        all_dict = [name for name in all_dict
                    if not name.startswith("_")]

    # 移除特定的字符串 'absolute_import', 'division', 'print_function'
    for name in ['absolute_import', 'division', 'print_function']:
        try:
            all_dict.remove(name)
        except ValueError:
            pass

    # 如果剩余的 all_dict 为空列表，则表示可能是一个纯文档模块，将 '__doc__' 加入其中
    if not all_dict:
        all_dict.append('__doc__')

    # 进一步过滤掉模块属性，只保留可调用的函数或类名
    all_dict = [name for name in all_dict
                if not inspect.ismodule(getattr(module, name, None))]

    deprecated = []
    not_deprecated = []

    # 对每个属性进行分类，将已弃用的和未弃用的分别加入不同的列表中
    for name in all_dict:
        f = getattr(module, name, None)
        if callable(f) and is_deprecated(f):
            deprecated.append(name)
        else:
            not_deprecated.append(name)

    # 计算其他未归类的模块属性
    others = set(dir(module)).difference(set(deprecated)).difference(set(not_deprecated))

    return not_deprecated, deprecated, others


# 比较 all_dict 中的属性和其他属性，返回三个集合：仅存在于 all_dict 中的，仅存在于 names 中的，以及在 names 中但在 others 中缺失的
def compare(all_dict, others, names, module_name):
    only_all = set()

    # 遍历 all_dict 中的属性，将不在 names 中的加入 only_all 集合
    for name in all_dict:
        if name not in names:
            for pat in REFGUIDE_AUTOSUMMARY_SKIPLIST:
                if re.match(pat, module_name + '.' + name):
                    break
            else:
                only_all.add(name)

    only_ref = set()
    missing = set()

    # 遍历 names 中的属性，将不在 all_dict 中的加入 only_ref 集合，将不在 others 中的加入 missing 集合
    for name in names:
        if name not in all_dict:
            for pat in REFGUIDE_ALL_SKIPLIST:
                if re.match(pat, module_name + '.' + name):
                    if name not in others:
                        missing.add(name)
                    break
            else:
                only_ref.add(name)

    return only_all, only_ref, missing


# 检查模块 f 是否已弃用
def is_deprecated(f):
    # 省略了返回值说明，这是一个检查模块 f 是否已弃用的函数
    # 使用 `warnings.catch_warnings` 捕获警告信息，记录在变量 `w` 中
    with warnings.catch_warnings(record=True) as w:
        # 设置警告过滤器，使得 DeprecationWarning 被视为错误
        warnings.simplefilter("error")
        try:
            # 调用函数 `f`，传入一个不正确的关键字参数字典
            f(**{"not a kwarg": None})
        except DeprecationWarning:
            # 如果捕获到 DeprecationWarning，表示函数使用了不推荐使用的特性，返回 True
            return True
        except Exception:
            # 捕获所有其他异常，不做处理
            pass
        # 如果没有捕获到 DeprecationWarning，则返回 False
        return False
# 检查 `all_dict` 是否与 `names` 在 `module_name` 中一致，确保没有废弃或多余的对象。
# 返回一个列表，每个元素为 (name, success_flag, output)，表示检查结果。

def check_items(all_dict, names, deprecated, others, module_name, dots=True):
    """
    Check that `all_dict` is consistent with the `names` in `module_name`
    For instance, that there are no deprecated or extra objects.

    Parameters
    ----------
    all_dict : list
        待检查的对象列表

    names : set
        参考指南中的对象名称集合

    deprecated : list
        废弃的对象列表

    others : list
        其他对象列表

    module_name : ModuleType
        模块名称

    dots : bool
        是否打印每次检查的点符号

    Returns
    -------
    list
        返回 [(name, success_flag, output)...] 的列表
    """

    # 计算 `all_dict` 和 `names` 的长度
    num_all = len(all_dict)
    num_ref = len(names)

    # 初始化输出字符串
    output = ""

    # 添加非废弃对象数量到输出
    output += "Non-deprecated objects in __all__: %i\n" % num_all
    # 添加参考指南中对象数量到输出
    output += "Objects in refguide: %i\n\n" % num_ref

    # 比较 `all_dict`、`others` 和 `names`，找出只在其中一个中存在的对象
    only_all, only_ref, missing = compare(all_dict, others, names, module_name)
    
    # 找出在参考指南中但已被废弃的对象
    dep_in_ref = only_ref.intersection(deprecated)
    # 从只在参考指南中的对象中去除已废弃的对象
    only_ref = only_ref.difference(deprecated)

    # 如果存在已废弃的对象，将它们添加到输出中
    if len(dep_in_ref) > 0:
        output += "Deprecated objects in refguide::\n\n"
        for name in sorted(deprecated):
            output += "    " + name + "\n"

    # 如果没有任何不一致，返回成功标志和输出
    if len(only_all) == len(only_ref) == len(missing) == 0:
        if dots:
            output_dot('.')
        return [(None, True, output)]
    else:
        # 如果存在在 `all_dict` 中但不在参考指南中的对象，将它们添加到输出中
        if len(only_all) > 0:
            output += "ERROR: objects in %s.__all__ but not in refguide::\n\n" % module_name
            for name in sorted(only_all):
                output += "    " + name + "\n"

            output += "\nThis issue can be fixed by adding these objects to\n"
            output += "the function listing in __init__.py for this module\n"

        # 如果存在在参考指南中但不在 `all_dict` 中的对象，将它们添加到输出中
        if len(only_ref) > 0:
            output += "ERROR: objects in refguide but not in %s.__all__::\n\n" % module_name
            for name in sorted(only_ref):
                output += "    " + name + "\n"

            output += "\nThis issue should likely be fixed by removing these objects\n"
            output += "from the function listing in __init__.py for this module\n"
            output += "or adding them to __all__.\n"

        # 如果存在缺失的对象，将它们添加到输出中
        if len(missing) > 0:
            output += "ERROR: missing objects::\n\n"
            for name in sorted(missing):
                output += "    " + name + "\n"

        # 如果需要打印点符号，打印 'F' 表示失败
        if dots:
            output_dot('F')
        return [(None, False, output)]


def validate_rst_syntax(text, name, dots=True):
    """
    Validates the doc string in a snippet of documentation
    `text` from file `name`

    Parameters
    ----------
    text : str
        待验证的文档字符串内容

    name : str
        文档所属文件名

    dots : bool
        是否打印每次检查的点符号

    Returns
    -------
    (bool, str)
        返回元组，第一个元素表示验证结果，第二个元素是相关输出信息
    """

    # 如果文档字符串为空，返回验证失败和相关错误信息
    if text is None:
        if dots:
            output_dot('E')
        return False, "ERROR: %s: no documentation" % (name,)
    # 定义一个包含可忽略项的集合，这些项在处理文档时可以忽略
    ok_unknown_items = set([
        'mod', 'doc', 'currentmodule', 'autosummary', 'data', 'attr',
        'obj', 'versionadded', 'versionchanged', 'module', 'class',
        'ref', 'func', 'toctree', 'moduleauthor', 'term', 'c:member',
        'sectionauthor', 'codeauthor', 'eq', 'doi', 'DOI', 'arXiv', 'arxiv'
    ])

    # 创建一个字符串流用于捕获错误信息
    error_stream = io.StringIO()

    # 定义一个函数 resolve，用于返回给定名称的 URL，通常返回一个假的 URL
    def resolve(name, is_label=False):
        return ("http://foo", name)

    # 定义一个特定的令牌字符串
    token = '<RST-VALIDATE-SYNTAX-CHECK>'

    # 使用 docutils 库的 publish_doctree 函数处理文本，返回文档树
    docutils.core.publish_doctree(
        text, token,
        settings_overrides = dict(halt_level=5,
                                  traceback=True,
                                  default_reference_context='title-reference',
                                  default_role='emphasis',
                                  link_base='',
                                  resolve_name=resolve,
                                  stylesheet_path='',
                                  raw_enabled=0,
                                  file_insertion_enabled=0,
                                  warning_stream=error_stream))

    # 从错误流中获取错误信息字符串
    error_msg = error_stream.getvalue()

    # 根据特定令牌分割错误信息，得到错误列表
    errors = error_msg.split(token)

    # 初始化成功标志为 True，输出字符串为空
    success = True
    output = ""

    # 遍历错误列表，处理每个错误信息
    for error in errors:
        lines = error.splitlines()
        if not lines:
            continue

        # 使用正则表达式匹配并检查是否是未知的解释文本角色或指令类型错误
        m = re.match(r'.*Unknown (?:interpreted text role|directive type) "(.*)".*$', lines[0])
        if m:
            # 如果匹配到未知项并且在可忽略项集合中，则跳过此错误
            if m.group(1) in ok_unknown_items:
                continue

        # 使用正则表达式匹配是否是数学指令错误，忽略“label”选项错误
        m = re.match(r'.*Error in "math" directive:.*unknown option: "label"', " ".join(lines), re.S)
        if m:
            continue

        # 将错误信息格式化为输出字符串，并设置成功标志为 False
        output += name + lines[0] + "::\n    " + "\n    ".join(lines[1:]).rstrip() + "\n"
        success = False

    # 如果有任何错误，则在输出字符串中添加分隔线和原始文本内容
    if not success:
        output += "    " + "-"*72 + "\n"
        for lineno, line in enumerate(text.splitlines()):
            output += "    %-4d    %s\n" % (lineno+1, line)
        output += "    " + "-"*72 + "\n\n"

    # 如果 dots 变量存在，则输出对应的点号（成功）或字母 'F'（失败）
    if dots:
        output_dot('.' if success else 'F')

    # 返回处理结果的成功标志和生成的输出字符串
    return success, output
# 输出一个消息到指定流，默认为标准错误流
def output_dot(msg='.', stream=sys.stderr):
    stream.write(msg)  # 将消息写入指定流
    stream.flush()  # 立即刷新流，确保消息被输出


# 检查模块的 reStructuredText 格式化情况
def check_rest(module, names, dots=True):
    """
    Check reStructuredText formatting of docstrings

    Parameters
    ----------
    module : ModuleType
        要检查的模块对象

    names : set
        包含要检查的名称的集合

    Returns
    -------
    result : list
        包含元组 (module_name, success_flag, output) 的列表
    """

    try:
        skip_types = (dict, str, unicode, float, int)  # 尝试定义要跳过的对象类型
    except NameError:
        # Python 3 中，unicode 类型不再存在
        skip_types = (dict, str, float, int)  # 定义要跳过的对象类型

    results = []  # 初始化结果列表

    # 如果模块名称不在预定义的 OTHER_MODULE_DOCS 中，则进行检查
    if module.__name__[6:] not in OTHER_MODULE_DOCS:
        # 调用 validate_rst_syntax 函数，验证模块的文档字符串语法
        results += [(module.__name__,) +
                    validate_rst_syntax(inspect.getdoc(module),
                                        module.__name__, dots=dots)]

    for name in names:
        full_name = module.__name__ + '.' + name
        obj = getattr(module, name, None)

        if obj is None:
            # 如果对象不存在，则记录结果
            results.append((full_name, False, "%s has no docstring" % (full_name,)))
            continue
        elif isinstance(obj, skip_types):
            # 如果对象的类型在跳过类型中，则跳过此次循环
            continue

        if inspect.ismodule(obj):
            text = inspect.getdoc(obj)  # 获取模块的文档字符串
        else:
            try:
                text = str(get_doc_object(obj))  # 获取对象的文档字符串并转换为字符串类型
            except Exception:
                import traceback
                # 如果获取文档字符串时出现异常，则记录错误信息
                results.append((full_name, False,
                                "Error in docstring format!\n" +
                                traceback.format_exc()))
                continue

        # 检查文档字符串中是否包含不可打印字符
        m = re.search("([\x00-\x09\x0b-\x1f])", text)
        if m:
            # 如果文档字符串包含不可打印字符，则记录警告消息
            msg = ("Docstring contains a non-printable character %r! "
                   "Maybe forgot r\"\"\"?" % (m.group(1),))
            results.append((full_name, False, msg))
            continue

        try:
            src_file = short_path(inspect.getsourcefile(obj))  # 获取对象源文件的简短路径
        except TypeError:
            src_file = None

        if src_file:
            file_full_name = src_file + ':' + full_name  # 构建带文件信息的完整名称
        else:
            file_full_name = full_name

        # 调用 validate_rst_syntax 函数，验证文本的 reStructuredText 语法
        results.append((full_name,) + validate_rst_syntax(text, file_full_name, dots=dots))

    return results  # 返回检查结果列表


### Doctest helpers ####

# 在运行例子时使用的命名空间
DEFAULT_NAMESPACE = {'np': np}

# 在检查中使用的命名空间
CHECK_NAMESPACE = {
      'np': np,
      'numpy': np,
      'assert_allclose': np.testing.assert_allclose,
      'assert_equal': np.testing.assert_equal,
      # 识别 numpy 的表现形式
      'array': np.array,
      'matrix': np.matrix,
      'int64': np.int64,
      'uint64': np.uint64,
      'int8': np.int8,
      'int32': np.int32,
      'float32': np.float32,
      'float64': np.float64,
      'dtype': np.dtype,
      'nan': np.nan,
      'inf': np.inf,
      'StringIO': io.StringIO,
}


class DTRunner(doctest.DocTestRunner):
    """
    The doctest runner
    """
    DIVIDER = "\n"  # 定义用于分隔输出的分隔符
    # 初始化函数，用于设置测试项的名称、检查器、详细模式和选项标志
    def __init__(self, item_name, checker=None, verbose=None, optionflags=0):
        # 设置实例的测试项名称
        self._item_name = item_name
        # 调用父类 DocTestRunner 的初始化方法，设置检查器、详细模式和选项标志
        doctest.DocTestRunner.__init__(self, checker=checker, verbose=verbose,
                                       optionflags=optionflags)

    # 报告测试项名称的输出，可选择在输出前添加换行符
    def _report_item_name(self, out, new_line=False):
        # 如果存在测试项名称并需要添加新行，则在输出中添加换行符
        if self._item_name is not None:
            if new_line:
                out("\n")
            # 将测试项名称设为 None，确保只输出一次
            self._item_name = None

    # 报告测试开始的方法，设置检查器的源码，并调用父类的 report_start 方法
    def report_start(self, out, test, example):
        # 设置检查器的源码为示例的源码
        self._checker._source = example.source
        # 调用父类 DocTestRunner 的 report_start 方法进行报告
        return doctest.DocTestRunner.report_start(self, out, test, example)

    # 报告测试成功的方法，根据详细模式决定是否输出测试项名称，并调用父类的 report_success 方法
    def report_success(self, out, test, example, got):
        # 如果设置了详细模式，则输出测试项名称（新行形式）
        if self._verbose:
            self._report_item_name(out, new_line=True)
        # 调用父类 DocTestRunner 的 report_success 方法进行报告
        return doctest.DocTestRunner.report_success(self, out, test, example, got)

    # 报告意外异常的方法，输出测试项名称，并调用父类的 report_unexpected_exception 方法
    def report_unexpected_exception(self, out, test, example, exc_info):
        # 输出测试项名称
        self._report_item_name(out)
        # 调用父类 DocTestRunner 的 report_unexpected_exception 方法进行报告
        return doctest.DocTestRunner.report_unexpected_exception(
            self, out, test, example, exc_info)

    # 报告测试失败的方法，输出测试项名称，并调用父类的 report_failure 方法
    def report_failure(self, out, test, example, got):
        # 输出测试项名称
        self._report_item_name(out)
        # 调用父类 DocTestRunner 的 report_failure 方法进行报告
        return doctest.DocTestRunner.report_failure(self, out, test,
                                                    example, got)
class Checker(doctest.OutputChecker):
    """
    自定义的输出检查器，继承自 doctest.OutputChecker 类。
    """

    # 正则表达式模式，用于匹配对象地址字符串
    obj_pattern = re.compile('at 0x[0-9a-fA-F]+>')

    # 创建一个默认的 doctest.OutputChecker 实例
    vanilla = doctest.OutputChecker()

    # 随机标记集合，用于匹配可能含有随机内容的注释或字符串
    rndm_markers = {'# random', '# Random', '#random', '#Random', "# may vary",
                    "# uninitialized", "#uninitialized", "# uninit"}

    # 停用词集合，包含一些常见的用于绘图和数据显示的方法或属性名
    stopwords = {'plt.', '.hist', '.show', '.ylim', '.subplot(',
                 'set_title', 'imshow', 'plt.show', '.axis(', '.plot(',
                 '.bar(', '.title', '.ylabel', '.xlabel', 'set_ylim', 'set_xlim',
                 '# reformatted', '.set_xlabel(', '.set_ylabel(', '.set_zlabel(',
                 '.set(xlim=', '.set(ylim=', '.set(xlabel=', '.set(ylabel='}

    def __init__(self, parse_namedtuples=True, ns=None, atol=1e-8, rtol=1e-2):
        """
        初始化方法，用于设置对象的各种属性。

        参数:
        - parse_namedtuples: 是否解析命名元组，默认为 True
        - ns: 命名空间，用于检查，如果为 None 则使用 CHECK_NAMESPACE
        - atol: 绝对误差容限，默认为 1e-8
        - rtol: 相对误差容限，默认为 1e-2
        """
        self.parse_namedtuples = parse_namedtuples
        self.atol, self.rtol = atol, rtol
        if ns is None:
            self.ns = CHECK_NAMESPACE
        else:
            self.ns = ns
    # 如果期望值和实际值相等，则返回True，表示通过检查
    def check_output(self, want, got, optionflags):
        if want == got:
            return True

        # 在源文本中跳过停用词
        if any(word in self._source for word in self.stopwords):
            return True

        # 跳过随机标记
        if any(word in want for word in self.rndm_markers):
            return True

        # 跳过函数/对象地址
        if self.obj_pattern.search(got):
            return True

        # 忽略注释（例如 signal.freqresp）
        if want.lstrip().startswith("#"):
            return True

        # 尝试使用标准的 doctest
        try:
            if self.vanilla.check_output(want, got, optionflags):
                return True
        except Exception:
            pass

        # 尝试将字符串转换为对象
        try:
            a_want = eval(want, dict(self.ns))
            a_got = eval(got, dict(self.ns))
        except Exception:
            # 可能是打印 numpy 数组的情况
            s_want = want.strip()
            s_got = got.strip()
            cond = (s_want.startswith("[") and s_want.endswith("]") and
                    s_got.startswith("[") and s_got.endswith("]"))
            if cond:
                # 重新插入逗号并重试比较
                s_want = ", ".join(s_want[1:-1].split())
                s_got = ", ".join(s_got[1:-1].split())
                return self.check_output(s_want, s_got, optionflags)

            if not self.parse_namedtuples:
                return False
            # 假设 "want" 是一个元组，"got" 是类似 MoodResult(statistic=10, pvalue=0.1) 的情况。
            # 将后者转换为元组 (10, 0.1)，然后进行比较。
            try:
                num = len(a_want)
                regex = (r'[\w\d_]+\(' +
                         ', '.join([r'[\w\d_]+=(.+)']*num) +
                         r'\)')
                grp = re.findall(regex, got.replace('\n', ' '))
                if len(grp) > 1:  # 目前只支持一个匹配
                    return False
                # 再次折叠成一个元组
                got_again = '(' + ', '.join(grp[0]) + ')'
                return self.check_output(want, got_again, optionflags)
            except Exception:
                return False

        # 如果上述尝试失败，则尝试使用 numpy 进行比较
        try:
            return self._do_check(a_want, a_got)
        except Exception:
            # 异构元组，例如 (1, np.array([1., 2.]))
            try:
                return all(self._do_check(w, g) for w, g in zip(a_want, a_got))
            except (TypeError, ValueError):
                return False
    # 定义一个方法 `_do_check`，用于比较两个值 `want` 和 `got` 是否相等或者在数值上接近
    def _do_check(self, want, got):
        # 尝试执行以下操作，确保正确处理所有类似于 numpy 对象、字符串和异构元组
        try:
            # 如果 want 等于 got，返回 True
            if want == got:
                return True
        # 捕获所有异常，不做任何处理，继续执行
        except Exception:
            pass
        # 使用 NumPy 的 allclose 方法比较 want 和 got 是否在给定的绝对误差和相对误差范围内数值接近
        return np.allclose(want, got, atol=self.atol, rtol=self.rtol)
# 运行修改后的 doctest 测试集合 `tests`

def _run_doctests(tests, full_name, verbose, doctest_warnings):
    """
    Run modified doctests for the set of `tests`.

    Parameters
    ----------
    tests : list
        包含测试用例的列表

    full_name : str
        完整名称字符串

    verbose : bool
        是否输出详细信息

    doctest_warnings : bool
        是否输出 doctest 的警告信息

    Returns
    -------
    tuple(bool, list)
        返回一个元组，包含成功标志和输出信息列表
    """
    flags = NORMALIZE_WHITESPACE | ELLIPSIS
    # 创建 DTRunner 实例，设置检查器和选项标志
    runner = DTRunner(full_name, checker=Checker(), optionflags=flags,
                      verbose=verbose)

    output = io.StringIO(newline='')
    success = True

    # 将 stderr 重定向到 stdout 或 output
    tmp_stderr = sys.stdout if doctest_warnings else output

    @contextmanager
    def temp_cwd():
        cwd = os.getcwd()
        tmpdir = tempfile.mkdtemp()
        try:
            os.chdir(tmpdir)
            yield tmpdir
        finally:
            os.chdir(cwd)
            shutil.rmtree(tmpdir)

    # 运行测试，并尝试恢复全局状态
    cwd = os.getcwd()
    with np.errstate(), np.printoptions(), temp_cwd() as tmpdir, \
            redirect_stderr(tmp_stderr):
        # 尝试确保随机种子不可重现
        np.random.seed(None)

        ns = {}
        for t in tests:
            # 更新测试的全局命名空间，以避免变量在测试块之间丢失
            t.globs.update(ns)
            # 将文件名转换为相对于当前工作目录的短路径
            t.filename = short_path(t.filename, cwd)
            # 处理测试选项
            if any([SKIPBLOCK in ex.options for ex in t.examples]):
                continue
            # 运行测试，并将结果写入输出
            fails, successes = runner.run(t, out=output.write, clear_globs=False)
            if fails > 0:
                success = False
            ns = t.globs

    # 将输出指针移到开头并读取输出内容
    output.seek(0)
    return success, output.read()


def check_doctests(module, verbose, ns=None,
                   dots=True, doctest_warnings=False):
    """
    Check code in docstrings of the module's public symbols.

    Parameters
    ----------
    module : ModuleType
        要检查的模块对象

    verbose : bool
        是否输出详细信息

    ns : dict
        模块的命名空间

    dots : bool

    doctest_warnings : bool

    Returns
    -------
    results : list
        返回结果列表 [(item_name, success_flag, output), ...]
    """
    if ns is None:
        ns = dict(DEFAULT_NAMESPACE)

    # 遍历非过时的模块项
    results = []
    # 遍历给定模块的所有函数名
    for name in get_all_dict(module)[0]:
        # 构建完整的函数名，包括模块名前缀
        full_name = module.__name__ + '.' + name

        # 检查是否需要跳过该函数的测试，根据全名在跳过字典中查找
        if full_name in DOCTEST_SKIPDICT:
            # 如果在跳过字典中找到该函数名，则获取需要跳过的测试方法列表
            skip_methods = DOCTEST_SKIPDICT[full_name]
            # 如果跳过方法列表为 None，则跳过当前函数的测试
            if skip_methods is None:
                continue
        else:
            # 如果未在跳过字典中找到该函数名，则设置跳过方法列表为 None
            skip_methods = None

        try:
            # 尝试获取当前函数的对象
            obj = getattr(module, name)
        except AttributeError:
            # 如果获取对象失败，记录缺失的项并继续下一个函数的处理
            import traceback
            results.append((full_name, False,
                            "Missing item!\n" +
                            traceback.format_exc()))
            continue

        # 创建一个 doctest 查找器对象
        finder = doctest.DocTestFinder()
        try:
            # 使用查找器找到当前函数的 doctest 测试集合
            tests = finder.find(obj, name, globs=dict(ns))
        except Exception:
            # 如果获取 doctest 失败，记录失败信息并继续下一个函数的处理
            import traceback
            results.append((full_name, False,
                            "Failed to get doctests!\n" +
                            traceback.format_exc()))
            continue

        # 如果需要跳过特定方法的测试，则过滤掉跳过列表中的测试方法
        if skip_methods is not None:
            tests = [i for i in tests if
                     i.name.partition(".")[2] not in skip_methods]

        # 运行当前函数的 doctest 测试，并获取测试结果和输出信息
        success, output = _run_doctests(tests, full_name, verbose,
                                        doctest_warnings)

        # 如果需要打印简化的测试结果（成功或失败）
        if dots:
            output_dot('.' if success else 'F')

        # 将当前函数的测试结果记录到结果列表中
        results.append((full_name, success, output))

        # 如果有 matplotlib 模块，关闭所有打开的图形窗口
        if HAVE_MATPLOTLIB:
            import matplotlib.pyplot as plt
            plt.close('all')

    # 返回所有函数测试的结果列表
    return results
# 检查指定文本文件中的文档测试代码。
def check_doctests_testfile(fname, verbose, ns=None,
                   dots=True, doctest_warnings=False):
    """
    Check code in a text file.

    Mimic `check_doctests` above, differing mostly in test discovery.
    (which is borrowed from stdlib's doctest.testfile here,
     https://github.com/python-git/python/blob/master/Lib/doctest.py)

    Parameters
    ----------
    fname : str
        File name
    verbose : bool
        是否输出详细信息
    ns : dict
        Name space，命名空间
    dots : bool
        是否显示点
    doctest_warnings : bool
        是否显示文档测试警告

    Returns
    -------
    list
        List of [(item_name, success_flag, output), ...]

    Notes
    -----

    refguide can be signalled to skip testing code by adding
    ``#doctest: +SKIP`` to the end of the line. If the output varies or is
    random, add ``# may vary`` or ``# random`` to the comment. for example

    >>> plt.plot(...)  # doctest: +SKIP
    >>> random.randint(0,10)
    5 # random

    We also try to weed out pseudocode:
    * We maintain a list of exceptions which signal pseudocode,
    * We split the text file into "blocks" of code separated by empty lines
      and/or intervening text.
    * If a block contains a marker, the whole block is then assumed to be
      pseudocode. It is then not being doctested.

    The rationale is that typically, the text looks like this:

    blah
    <BLANKLINE>
    >>> from numpy import some_module   # pseudocode!
    >>> func = some_module.some_function
    >>> func(42)                  # still pseudocode
    146
    <BLANKLINE>
    blah
    <BLANKLINE>
    >>> 2 + 3        # real code, doctest it
    5

    """
    # 如果命名空间为None，则使用默认的CHECK_NAMESPACE命名空间
    if ns is None:
        ns = CHECK_NAMESPACE
    # 存储测试结果的列表
    results = []

    # 获取文件名的基本名称和完整路径
    _, short_name = os.path.split(fname)
    # 如果文件名在DOCTEST_SKIPDICT字典中，直接返回空结果列表
    if short_name in DOCTEST_SKIPDICT:
        return results

    # 打开文件并读取文件内容到文本变量中
    full_name = fname
    with open(fname, encoding='utf-8') as f:
        text = f.read()

    # 用于标识伪代码块的集合
    PSEUDOCODE = set(['some_function', 'some_module', 'import example',
                      'ctypes.CDLL',     # likely need compiling, skip it
                      'integrate.nquad(func,'  # ctypes integrate tutotial
    ])

    # 将文本分割为多个代码块，并尝试检测和排除伪代码块
    parser = doctest.DocTestParser()
    # 存储有效部分的列表
    good_parts = []
    # 基础行号初始化为0
    base_line_no = 0
    # 按双换行符 '\n\n' 分割文本，处理每个部分
    for part in text.split('\n\n'):
        try:
            # 尝试从当前部分获取 doctest 测试
            tests = parser.get_doctest(part, ns, fname, fname, base_line_no)
        except ValueError as e:
            # 如果捕获到 ValueError 异常
            if e.args[0].startswith('line '):
                # 修正错误消息中的行号，因为 `parser.get_doctest` 不会在错误消息中增加 base_line_no
                parts = e.args[0].split()
                parts[1] = str(int(parts[1]) + base_line_no)
                e.args = (' '.join(parts),) + e.args[1:]
            # 重新抛出异常
            raise

        # 检查是否有伪代码关键字，如果有则跳过该部分
        if any(word in ex.source for word in PSEUDOCODE
                                 for ex in tests.examples):
            # 跳过该部分
            pass
        else:
            # 将看起来像是好的代码部分添加到 good_parts 中以供后续 doctest
            good_parts.append((part, base_line_no))

        # 更新 base_line_no，增加当前部分中换行符 '\n' 的数量再加 2
        base_line_no += part.count('\n') + 2

    # 重新组装好的部分并对其进行 doctest 测试
    tests = []
    for good_text, line_no in good_parts:
        tests.append(parser.get_doctest(good_text, ns, fname, fname, line_no))
    # 运行所有 doctest 测试并获取结果
    success, output = _run_doctests(tests, full_name, verbose,
                                    doctest_warnings)

    # 如果 dots 为真，则输出相应的点或失败的标志
    if dots:
        output_dot('.' if success else 'F')

    # 将当前模块的测试结果添加到 results 列表中
    results.append((full_name, success, output))

    # 如果有 matplotlib 库，关闭所有打开的图形窗口
    if HAVE_MATPLOTLIB:
        import matplotlib.pyplot as plt
        plt.close('all')

    # 返回最终的测试结果列表
    return results
# 定义生成器函数，用于遍历 `base_path` 及其子目录，跳过在 RST_SKIPLIST 中指定的文件或目录，并生成每个具有指定后缀的文件路径

def iter_included_files(base_path, verbose=0, suffixes=('.rst',)):
    """
    Generator function to walk `base_path` and its subdirectories, skipping
    files or directories in RST_SKIPLIST, and yield each file with a suffix in
    `suffixes`

    Parameters
    ----------
    base_path : str
        Base path of the directory to be processed
    verbose : int
        Verbosity level (default is 0)
    suffixes : tuple
        Tuple of suffixes to filter files (default is ('.rst',))

    Yields
    ------
    path : str
        Path of the directory and its subdirectories containing files with specified suffixes
    """
    
    # 如果 `base_path` 存在且是一个文件，则直接生成该文件路径
    if os.path.exists(base_path) and os.path.isfile(base_path):
        yield base_path
    
    # 遍历 `base_path` 及其子目录
    for dir_name, subdirs, files in os.walk(base_path, topdown=True):
        # 如果当前目录在 RST_SKIPLIST 中，则跳过其中的文件
        if dir_name in RST_SKIPLIST:
            if verbose > 0:
                sys.stderr.write('skipping files in %s' % dir_name)
            files = []  # 清空文件列表以跳过文件
        
        # 如果当前目录中存在在 RST_SKIPLIST 中的子目录，则移除以跳过这些子目录
        for p in RST_SKIPLIST:
            if p in subdirs:
                if verbose > 0:
                    sys.stderr.write('skipping %s and subdirs' % p)
                subdirs.remove(p)
        
        # 遍历当前目录中的文件
        for f in files:
            # 如果文件具有指定后缀且不在 RST_SKIPLIST 中，则生成文件的完整路径
            if (os.path.splitext(f)[1] in suffixes and
                    f not in RST_SKIPLIST):
                yield os.path.join(dir_name, f)


def check_documentation(base_path, results, args, dots):
    """
    Check examples in any *.rst located inside `base_path`.
    Add the output to `results`.

    See Also
    --------
    check_doctests_testfile
    """
    
    # 遍历 `base_path` 中包含的所有 `.rst` 文件，检查其中的示例文档
    for filename in iter_included_files(base_path, args.verbose):
        if dots:
            sys.stderr.write(filename + ' ')
            sys.stderr.flush()

        # 检查给定文件中的 doctest 测试，并将结果添加到 `results` 列表中
        tut_results = check_doctests_testfile(
            filename,
            (args.verbose >= 2), dots=dots,
            doctest_warnings=args.doctest_warnings)

        # 创建一个空的“模块”用于报告结果时需要
        def scratch():
            pass
        scratch.__name__ = filename
        results.append((scratch, tut_results))
        if dots:
            sys.stderr.write('\n')
            sys.stderr.flush()


def init_matplotlib():
    """
    Check feasibility of matplotlib initialization.
    """
    
    # 尝试导入 matplotlib 并设置使用 Agg 后端，标记是否成功导入 matplotlib
    global HAVE_MATPLOTLIB

    try:
        import matplotlib
        matplotlib.use('Agg')
        HAVE_MATPLOTLIB = True
    except ImportError:
        HAVE_MATPLOTLIB = False


def main(argv):
    """
    Validates the docstrings of all the pre decided set of
    modules for errors and docstring standards.
    """
    
    # 解析命令行参数，验证预定义模块的文档字符串的正确性和标准
    parser = ArgumentParser(usage=__doc__.lstrip())
    parser.add_argument("module_names", metavar="SUBMODULES", default=[],
                        nargs='*', help="Submodules to check (default: all public)")
    parser.add_argument("--doctests", action="store_true",
                        help="Run also doctests on ")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--doctest-warnings", action="store_true",
                        help="Enforce warning checking for doctests")
    # 添加一个命令行参数 --rst，它可以是一个可选的位置参数，如果没有提供默认为None，常量为'doc'，用来指定运行 *rst 文件中的示例，这些文件在指定的目录中递归查找，默认为'doc'
    parser.add_argument("--rst", nargs='?', const='doc', default=None,
                        help=("Run also examples from *rst files "
                              "discovered walking the directory(s) specified, "
                              "defaults to 'doc'"))
    
    # 解析命令行参数
    args = parser.parse_args(argv)

    # 初始化空列表，用于存放模块对象
    modules = []
    # 初始化空字典，用于存放模块名称映射
    names_dict = {}

    # 如果未提供模块名称参数，则使用预定义的公共子模块列表和基础模块
    if not args.module_names:
        args.module_names = list(PUBLIC_SUBMODULES) + [BASE_MODULE]

    # 设置环境变量，用于启用 SciPy 的 PIL 图像查看器
    os.environ['SCIPY_PIL_IMAGE_VIEWER'] = 'true'

    # 复制模块名称列表，以便动态修改
    module_names = list(args.module_names)
    # 循环处理模块名称列表中的每一个名称
    for name in module_names:
        # 如果模块名称在 OTHER_MODULE_DOCS 中有映射，则使用映射后的名称替换原名称，并确保新名称未在列表中
        if name in OTHER_MODULE_DOCS:
            name = OTHER_MODULE_DOCS[name]
            if name not in module_names:
                module_names.append(name)

    # 初始化布尔变量 dots 和 success，以及空列表 results 和 errormsgs
    dots = True
    success = True
    results = []
    errormsgs = []

    # 如果指定运行 doctests 或者 rst 文件，则初始化 matplotlib
    if args.doctests or args.rst:
        init_matplotlib()

    # 遍历模块名称列表中的每一个子模块名称
    for submodule_name in module_names:
        # 设置模块名称的前缀为 BASE_MODULE + '.'
        prefix = BASE_MODULE + '.'
        # 如果子模块名称不以前缀开头且不等于 BASE_MODULE，则加上前缀
        if not (
            submodule_name.startswith(prefix) or
            submodule_name == BASE_MODULE
        ):
            module_name = prefix + submodule_name
        else:
            module_name = submodule_name
        
        # 动态导入模块
        __import__(module_name)
        module = sys.modules[module_name]

        # 如果子模块名称不在 OTHER_MODULE_DOCS 中，调用 find_names 函数找到模块中的名称，并更新到 names_dict 中
        if submodule_name not in OTHER_MODULE_DOCS:
            find_names(module, names_dict)

        # 如果子模块名称在命令行参数中指定的模块名称中，则将模块对象添加到 modules 列表中
        if submodule_name in args.module_names:
            modules.append(module)

    # 如果指定运行 doctests 或者不运行 rst 文件，则打印运行检查的模块数量信息
    if args.doctests or not args.rst:
        print("Running checks for %d modules:" % (len(modules),))
        # 遍历 modules 列表中的每一个模块对象
        for module in modules:
            # 如果 dots 为 True，则将模块名称打印到标准错误
            if dots:
                sys.stderr.write(module.__name__ + ' ')
                sys.stderr.flush()

            # 获取模块中的所有符号字典、过时的名称集合和其他名称集合
            all_dict, deprecated, others = get_all_dict(module)
            # 获取模块名称对应的名称集合
            names = names_dict.get(module.__name__, set())

            # 初始化模块检查结果列表
            mod_results = []
            # 检查模块中的所有项，并将结果添加到 mod_results 中
            mod_results += check_items(all_dict, names, deprecated, others,
                                       module.__name__)
            # 检查模块中未包含的名称，并将结果添加到 mod_results 中
            mod_results += check_rest(module, set(names).difference(deprecated),
                                      dots=dots)
            # 如果指定运行 doctests，则运行模块中的 doctests，并将结果添加到 mod_results 中
            if args.doctests:
                mod_results += check_doctests(module, (args.verbose >= 2), dots=dots,
                                              doctest_warnings=args.doctest_warnings)

            # 断言 mod_results 中的每个元素都是元组类型
            for v in mod_results:
                assert isinstance(v, tuple), v

            # 将模块及其检查结果作为元组添加到 results 列表中
            results.append((module, mod_results))

            # 如果 dots 为 True，则换行打印标准错误
            if dots:
                sys.stderr.write('\n')
                sys.stderr.flush()
    # 如果传入了 rst 参数，则进行以下操作
    if args.rst:
        # 获取当前脚本文件所在目录的绝对路径，并拼接上一级目录作为基础目录
        base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
        # 构建 rst 文件相对于基础目录的相对路径
        rst_path = os.path.relpath(os.path.join(base_dir, args.rst))
        # 如果构建的 rst 文件路径存在，则输出检查信息
        if os.path.exists(rst_path):
            print('\nChecking files in %s:' % rst_path)
            # 调用函数检查文档，并传递相应参数
            check_documentation(rst_path, results, args, dots)
        else:
            # 如果构建的 rst 文件路径不存在，则输出错误信息并记录错误消息
            sys.stderr.write(f'\ninvalid --rst argument "{args.rst}"')
            errormsgs.append('invalid directory argument to --rst')
        # 如果 dots 参数为真，则输出换行到标准错误输出
        if dots:
            sys.stderr.write("\n")
            sys.stderr.flush()

    # 报告检查结果
    for module, mod_results in results:
        # 检查当前模块的所有结果是否成功
        success = all(x[1] for x in mod_results)
        # 如果有任何一个检查不成功，则记录错误消息
        if not success:
            errormsgs.append(f'failed checking {module.__name__}')

        # 如果所有检查成功，并且 verbose 等于 0，则继续下一个模块
        if success and args.verbose == 0:
            continue

        # 输出模块名称
        print("")
        print("=" * len(module.__name__))
        print(module.__name__)
        print("=" * len(module.__name__))
        print("")

        # 遍历当前模块的检查结果
        for name, success, output in mod_results:
            # 如果名称为 None，且检查不成功或 verbose 大于等于 1，则输出相关信息
            if name is None:
                if not success or args.verbose >= 1:
                    print(output.strip())
                    print("")
            # 如果名称不为 None，且检查不成功或 verbose 大于等于 2，则输出相关信息
            elif not success or (args.verbose >= 2 and output.strip()):
                print(name)
                print("-" * len(name))
                print("")
                print(output.strip())
                print("")

    # 如果错误消息列表长度为 0，则输出所有检查通过的消息，并退出程序返回状态 0
    if len(errormsgs) == 0:
        print("\nOK: all checks passed!")
        sys.exit(0)
    # 否则输出错误消息列表，并退出程序返回状态 1
    else:
        print('\nERROR: ', '\n        '.join(errormsgs))
        sys.exit(1)
# 如果当前模块是直接被执行的主程序
if __name__ == '__main__':
    # 调用 main 函数，并传递命令行参数列表作为参数
    main(argv=sys.argv[1:])
```