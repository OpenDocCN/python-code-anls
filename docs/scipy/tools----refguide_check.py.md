# `D:\src\scipysrc\scipy\tools\refguide_check.py`

```
#!/usr/bin/env python3
"""
refguide_check.py [OPTIONS] [-- ARGS]

Check for a Scipy submodule whether the objects in its __all__ dict
correspond to the objects included in the reference guide.

Example of usage::

    $ python3 refguide_check.py optimize

Note that this is a helper script to be able to check if things are missing;
the output of this script does need to be checked manually.  In some cases
objects are left out of the refguide for a good reason (it's an alias of
another function, or deprecated, or ...)

"""

# 导入必要的库和模块
import copy               # 导入 copy 模块，用于深拷贝对象
import inspect            # 导入 inspect 模块，用于检查对象的属性和方法
import io                 # 导入 io 模块，用于处理文件流
import os                 # 导入 os 模块，提供与操作系统交互的功能
import re                 # 导入 re 模块，用于正则表达式的操作
import sys                # 导入 sys 模块，提供对 Python 解释器的访问
import warnings           # 导入 warnings 模块，用于警告管理
from argparse import ArgumentParser  # 导入 ArgumentParser 类，用于解析命令行参数

import docutils.core      # 导入 docutils.core 模块，用于处理 reStructuredText 文档
from docutils.parsers.rst import directives  # 导入 directives 子模块，提供解析和处理 reStructuredText 指令

from numpydoc.docscrape_sphinx import get_doc_object  # 导入 numpydoc_sphinx 模块的 get_doc_object 函数
from numpydoc.docscrape import NumpyDocString         # 导入 numpydoc 模块的 NumpyDocString 类
from scipy.stats._distr_params import distcont, distdiscrete  # 从 scipy.stats._distr_params 模块导入 distcont 和 distdiscrete
from scipy import stats  # 导入 scipy.stats 模块的所有内容

# Enable specific Sphinx directives
from sphinx.directives.other import SeeAlso, Only  # 从 sphinx.directives.other 模块导入 SeeAlso 和 Only 指令
directives.register_directive('seealso', SeeAlso)   # 注册 reStructuredText 指令 'seealso'
directives.register_directive('only', Only)         # 注册 reStructuredText 指令 'only'

BASE_MODULE = "scipy"  # 定义基础模块名称为 "scipy"

PUBLIC_SUBMODULES = [   # 定义公共子模块列表
    'cluster',
    'cluster.hierarchy',
    'cluster.vq',
    'constants',
    'datasets',
    'fft',
    'fftpack',
    'fftpack.convolve',
    'integrate',
    'interpolate',
    'io',
    'io.arff',
    'io.matlab',
    'io.wavfile',
    'linalg',
    'linalg.blas',
    'linalg.lapack',
    'linalg.interpolative',
    'misc',
    'ndimage',
    'odr',
    'optimize',
    'signal',
    'signal.windows',
    'sparse',
    'sparse.csgraph',
    'sparse.linalg',
    'spatial',
    'spatial.distance',
    'spatial.transform',
    'special',
    'stats',
    'stats.mstats',
    'stats.contingency',
    'stats.qmc',
    'stats.sampling'
]

# Docs for these modules are included in the parent module
OTHER_MODULE_DOCS = {   # 定义其他模块文档的映射关系
    'fftpack.convolve': 'fftpack',
    'io.wavfile': 'io',
    'io.arff': 'io',
}

# these names are not required to be present in ALL despite being in
# autosummary:: listing
REFGUIDE_ALL_SKIPLIST = [   # 定义不需要出现在 __all__ 中的名称列表
    r'scipy\.sparse\.csgraph',
    r'scipy\.sparse\.linalg',
    r'scipy\.linalg\.blas\.[sdczi].*',
    r'scipy\.linalg\.lapack\.[sdczi].*',
]

# these names are not required to be in an autosummary:: listing
# despite being in ALL
REFGUIDE_AUTOSUMMARY_SKIPLIST = [   # 定义不需要出现在 autosummary 中的名称列表
    r'scipy\.special\..*_roots',     # 旧的别名不需要在 autosummary 中
    r'scipy\.special\.jn',           # jn 是 jv 的别名
    r'scipy\.ndimage\.sum',          # sum 是 sum_labels 的别名
    r'scipy\.linalg\.solve_lyapunov',# 废弃的名称
    r'scipy\.stats\.contingency\.chi2_contingency',
    r'scipy\.stats\.contingency\.expected_freq',
    r'scipy\.stats\.contingency\.margins',
    r'scipy\.stats\.reciprocal',     # reciprocal 是 lognormal 的别名
    r'scipy\.stats\.trapz',          # trapz 是 trapezoid 的别名
]
# deprecated windows in scipy.signal namespace
# 将一组字符串命名添加到 REFGUIDE_AUTOSUMMARY_SKIPLIST 列表中，用于后续引用和过滤
for name in ('barthann', 'bartlett', 'blackmanharris', 'blackman', 'bohman',
             'boxcar', 'chebwin', 'cosine', 'exponential', 'flattop',
             'gaussian', 'general_gaussian', 'hamming', 'hann', 'hanning',
             'kaiser', 'nuttall', 'parzen', 'triang', 'tukey'):
    REFGUIDE_AUTOSUMMARY_SKIPLIST.append(r'scipy\.signal\.' + name)


def short_path(path, cwd=None):
    """
    Return relative or absolute path name, whichever is shortest.
    """
    # 如果 path 不是字符串，则直接返回它，无需处理
    if not isinstance(path, str):
        return path
    # 如果未提供当前工作目录 cwd，则使用当前工作目录
    if cwd is None:
        cwd = os.getcwd()
    # 获取 path 的绝对路径和相对于当前工作目录 cwd 的相对路径
    abspath = os.path.abspath(path)
    relpath = os.path.relpath(path, cwd)
    # 返回路径名最短的那个，可以是绝对路径或相对路径
    if len(abspath) <= len(relpath):
        return abspath
    else:
        return relpath


def find_names(module, names_dict):
    # Refguide entries:
    #
    # - 3 spaces followed by function name, and maybe some spaces, some
    #   dashes, and an explanation; only function names listed in
    #   refguide are formatted like this (mostly, there may be some false
    #   positives)
    #
    # - special directives, such as data and function
    #
    # - (scipy.constants only): quoted list
    #
    # 定义用于匹配文档中函数名和特殊指令的正则表达式模式列表
    patterns = [
        r"^\s\s\s([a-z_0-9A-Z]+)(\s+-+.*)?$",
        r"^\.\. (?:data|function)::\s*([a-z_0-9A-Z]+)\s*$"
    ]

    # 如果 module 是 scipy.constants 模块，增加另一个用于匹配 quoted list 的正则表达式模式
    if module.__name__ == 'scipy.constants':
        patterns += ["^``([a-z_0-9A-Z]+)``"]

    # 编译所有正则表达式模式
    patterns = [re.compile(pattern) for pattern in patterns]
    # 获取模块的名称
    module_name = module.__name__

    # 遍历模块的文档字符串的每一行
    for line in module.__doc__.splitlines():
        # 匹配当前模块的指令，更新 module_name
        res = re.search(
            r"^\s*\.\. (?:currentmodule|module):: ([a-z0-9A-Z_.]+)\s*$",
            line
        )
        if res:
            module_name = res.group(1)
            continue

        # 遍历所有定义的正则表达式模式，匹配文档中的内容
        for pattern in patterns:
            res = re.match(pattern, line)
            if res is not None:
                name = res.group(1)
                # 将匹配到的函数名或指令名添加到 names_dict 中对应模块名的集合中
                names_dict.setdefault(module_name, set()).add(name)
                break


def get_all_dict(module):
    """Return a copy of the __all__ dict with irrelevant items removed."""
    # 如果模块具有 "__all__" 属性，则深拷贝该属性的值作为 all_dict
    if hasattr(module, "__all__"):
        all_dict = copy.deepcopy(module.__all__)
    else:
        # 否则，深拷贝模块所有属性的名称作为 all_dict，但过滤掉以 "_" 开头的属性名
        all_dict = copy.deepcopy(dir(module))
        all_dict = [name for name in all_dict
                    if not name.startswith("_")]
    # 移除特定的名称，这些名称通常不是模块的公共接口
    for name in ['absolute_import', 'division', 'print_function']:
        try:
            all_dict.remove(name)
        except ValueError:
            pass

    # 过滤掉 all_dict 中的模块对象，只保留可调用的函数名或其他对象
    all_dict = [name for name in all_dict
                if not inspect.ismodule(getattr(module, name, None))]

    deprecated = []
    not_deprecated = []
    # 遍历 all_dict 中的每个名称，判断是否被标记为 deprecated
    for name in all_dict:
        f = getattr(module, name, None)
        if callable(f) and is_deprecated(f):
            deprecated.append(name)
        else:
            not_deprecated.append(name)
    # 创建一个集合，包含 module 对象中存在的所有属性名
    others = (set(dir(module))
              # 从集合中移除 deprecated 集合中包含的属性名
              .difference(set(deprecated))
              # 从集合中再移除 not_deprecated 集合中包含的属性名
              .difference(set(not_deprecated)))

    # 返回三个集合：未弃用属性的集合、已弃用属性的集合、以及其他未分类的属性集合
    return not_deprecated, deprecated, others
# 比较函数，用于返回在 __all__、refguide 中的唯一对象集合以及完全缺失的对象集合
def compare(all_dict, others, names, module_name):
    # 初始化一个空集合，用于存放只在 __all__ 中存在的对象名
    only_all = set()
    # 遍历 all_dict 中的每个对象名
    for name in all_dict:
        # 如果对象名不在 names 中
        if name not in names:
            # 遍历 REFGUIDE_AUTOSUMMARY_SKIPLIST 中的模式
            for pat in REFGUIDE_AUTOSUMMARY_SKIPLIST:
                # 如果模式与 module_name + '.' + name 匹配
                if re.match(pat, module_name + '.' + name):
                    break
            else:
                # 如果没有匹配到任何模式，则将对象名添加到 only_all 集合中
                only_all.add(name)

    # 初始化两个空集合，用于存放只在 refguide 中存在的对象名和完全缺失的对象名
    only_ref = set()
    missing = set()
    # 遍历 names 中的每个对象名
    for name in names:
        # 如果对象名不在 all_dict 中
        if name not in all_dict:
            # 遍历 REFGUIDE_ALL_SKIPLIST 中的模式
            for pat in REFGUIDE_ALL_SKIPLIST:
                # 如果模式与 module_name + '.' + name 匹配
                if re.match(pat, module_name + '.' + name):
                    # 如果 name 不在 others 中，则将 name 添加到 missing 集合中
                    if name not in others:
                        missing.add(name)
                    break
            else:
                # 如果没有匹配到任何模式，则将对象名添加到 only_ref 集合中
                only_ref.add(name)

    # 返回三个集合，分别是只在 __all__ 中存在的对象集合、只在 refguide 中存在的对象集合和完全缺失的对象集合
    return only_all, only_ref, missing


# 判断函数是否已被弃用
def is_deprecated(f):
    # 使用 warnings 模块捕获警告信息
    with warnings.catch_warnings(record=True):
        # 设置只捕获 DeprecationWarning 类型的警告
        warnings.simplefilter("error")
        try:
            # 调用函数 f，传递一个非关键字参数 "not a kwarg"
            f(**{"not a kwarg":None})
        except DeprecationWarning:
            # 如果捕获到 DeprecationWarning 警告，则返回 True
            return True
        except Exception:
            pass
        # 如果没有捕获到 DeprecationWarning 警告，则返回 False
        return False


# 检查项目中的对象
def check_items(all_dict, names, deprecated, others, module_name, dots=True):
    # 计算 all_dict 和 names 的长度
    num_all = len(all_dict)
    num_ref = len(names)

    # 初始化一个空字符串，用于存放输出信息
    output = ""

    # 将非弃用的对象数目添加到输出信息中
    output += "Non-deprecated objects in __all__: %i\n" % num_all
    # 将 refguide 中的对象数目添加到输出信息中
    output += "Objects in refguide: %i\n\n" % num_ref

    # 调用 compare 函数，获取只在 __all__、refguide 中存在的对象集合和完全缺失的对象集合
    only_all, only_ref, missing = compare(all_dict, others, names, module_name)
    # 计算同时存在于 refguide 和被弃用对象集合中的对象集合
    dep_in_ref = only_ref.intersection(deprecated)
    # 从只在 refguide 中存在的对象集合中移除已被弃用的对象集合
    only_ref = only_ref.difference(deprecated)

    # 如果存在于 refguide 和被弃用对象集合中的对象数目大于 0
    if len(dep_in_ref) > 0:
        # 将 "Deprecated objects in refguide::" 添加到输出信息中
        output += "Deprecated objects in refguide::\n\n"
        # 遍历已被弃用对象集合中的每个对象名，并将其添加到输出信息中
        for name in sorted(deprecated):
            output += "    " + name + "\n"

    # 如果只在 __all__、refguide 和完全缺失的对象集合中的对象数目均为 0
    if len(only_all) == len(only_ref) == len(missing) == 0:
        # 如果 dots 为 True，则调用 output_dot 函数
        if dots:
            output_dot('.')
        # 返回一个包含 None、True 和输出信息的元组的列表
        return [(None, True, output)]
    else:
        # 如果只在 __all__ 中存在的对象集合中的对象数目大于 0
        if len(only_all) > 0:
            # 将错误信息添加到输出信息中，说明只在 __all__ 中存在但不在 refguide 中的对象
            output += (
                f"ERROR: objects in {module_name}.__all__ but not in refguide::\n\n"
            )
            # 遍历只在 __all__ 中存在的对象集合中的每个对象名，并将其添加到输出信息中
            for name in sorted(only_all):
                output += "    " + name + "\n"

            # 添加解决此问题的建议到输出信息中
            output += "\nThis issue can be fixed by adding these objects to\n"
            output += "the function listing in __init__.py for this module\n"

        # 如果只在 refguide 中存在的对象集合中的对象数目大于 0
        if len(only_ref) > 0:
            # 将错误信息添加到输出信息中，说明只在 refguide 中存在但不在 __all__ 中的对象
            output += (
                f"ERROR: objects in refguide but not in {module_name}.__all__::\n\n"
            )
            # 遍历只在 refguide 中存在的对象集合中的每个对象名，并将其添加到输出信息中
            for name in sorted(only_ref):
                output += "    " + name + "\n"

            # 添加解决此问题的建议到输出信息中
            output += "\nThis issue should likely be fixed by removing these objects\n"
            output += "from the function listing in __init__.py for this module\n"
            output += "or adding them to __all__.\n"

        # 如果完全缺失的对象集合中的对象数目大于 0
        if len(missing) > 0:
            # 将错误信息添加到输出信息中，说明缺失的对象
            output += "ERROR: missing objects::\n\n"
            # 遍历完全缺失的对象集合中的每个对象名，并将其添加到输出信息中
            for name in sorted(missing):
                output += "    " + name + "\n"

        # 如果 dots 为 True，则调用 output_dot 函数
        if dots:
            output_dot('F')
        # 返回一个包含 None、False 和输出信息的元组的列表
        return [(None, False, output)]
def validate_rst_syntax(text, name, dots=True):
    # 如果文本为空，则根据 dots 参数输出错误信息并返回 False 和错误消息
    if text is None:
        if dots:
            output_dot('E')
        return False, f"ERROR: {name}: no documentation"

    # 允许的未知项集合
    ok_unknown_items = set([
        'mod', 'currentmodule', 'autosummary', 'data', 'legacy',
        'obj', 'versionadded', 'versionchanged', 'module', 'class', 'meth',
        'ref', 'func', 'toctree', 'moduleauthor', 'deprecated',
        'sectionauthor', 'codeauthor', 'eq', 'doi', 'DOI', 'arXiv', 'arxiv',
        'versionremoved',
    ])

    # 创建一个用于捕获错误信息的 StringIO 对象
    error_stream = io.StringIO()

    # 解析函数，用于返回占位符和名称的链接
    def resolve(name, is_label=False):
        return ("http://foo", name)

    # 设置验证过程的标记
    token = '<RST-VALIDATE-SYNTAX-CHECK>'

    # 使用 docutils 库执行文档树的发布和验证
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

    # 获取错误信息并按标记分割
    error_msg = error_stream.getvalue()
    errors = error_msg.split(token)
    success = True
    output = ""

    # 遍历每个错误信息段落
    for error in errors:
        lines = error.splitlines()
        if not lines:
            continue

        # 检查是否为未知项错误，如果是则跳过
        m = re.match(
            r'.*Unknown (?:interpreted text role|directive type) "(.*)".*$',
            lines[0]
        )
        if m:
            if m.group(1) in ok_unknown_items:
                continue

        # 检查是否为数学指令中的未知选项错误，如果是则跳过
        m = re.match(
            r'.*Error in "math" directive:.*unknown option: "label"', " ".join(lines),
            re.S
        )
        if m:
            continue

        # 按格式输出错误信息
        output += (
            name + lines[0] + "::\n    " + "\n    ".join(lines[1:]).rstrip() + "\n"
        )
        success = False

    # 如果存在错误，则打印源代码并添加分隔线
    if not success:
        output += "    " + "-"*72 + "\n"
        for lineno, line in enumerate(text.splitlines()):
            output += "    %-4d    %s\n" % (lineno+1, line)
        output += "    " + "-"*72 + "\n\n"

    # 如果 dots 参数为 True，则输出状态信息
    if dots:
        output_dot('.' if success else 'F')
    # 返回检验结果和输出信息
    return success, output


def output_dot(msg='.', stream=sys.stderr):
    # 输出指定的消息到流并刷新流
    stream.write(msg)
    stream.flush()


def check_rest(module, names, dots=True):
    """
    Check reStructuredText formatting of docstrings

    Returns: [(name, success_flag, output), ...]
    """
    # 跳过检查的数据类型集合
    skip_types = (dict, str, float, int)

    # 结果列表
    results = []

    # 如果模块名不在预定义的其他模块文档列表中，则进行格式验证
    if module.__name__[6:] not in OTHER_MODULE_DOCS:
        # 执行验证函数，并将结果添加到结果列表中
        results += [(module.__name__,) +
                    validate_rst_syntax(inspect.getdoc(module),
                                        module.__name__, dots=dots)]
    # 遍历传入的名称列表
    for name in names:
        # 构建完整的对象名称，包括模块名和对象名
        full_name = module.__name__ + '.' + name
        # 通过名称获取对象
        obj = getattr(module, name, None)

        # 如果获取的对象为空，记录错误信息并继续下一个循环
        if obj is None:
            results.append((full_name, False, f"{full_name} has no docstring"))
            continue
        # 如果对象属于应跳过的类型，则直接跳过
        elif isinstance(obj, skip_types):
            continue

        # 如果对象是模块，获取其文档字符串
        if inspect.ismodule(obj):
            text = inspect.getdoc(obj)
        else:
            # 否则，尝试获取对象的文档信息
            try:
                text = str(get_doc_object(obj))
            except Exception:
                import traceback
                # 记录文档格式化异常的详细信息，并继续下一个循环
                results.append((full_name, False,
                                "Error in docstring format!\n" +
                                traceback.format_exc()))
                continue

        # 检查文档字符串是否包含不可打印字符
        m = re.search(".*?([\x00-\x09\x0b-\x1f]).*", text)
        if m:
            # 如果存在不可打印字符，记录错误信息
            msg = ("Docstring contains a non-printable character "
                   f"{m.group(1)!r} in the line\n\n{m.group(0)!r}\n\n"
                   "Maybe forgot r\"\"\"?")
            results.append((full_name, False, msg))
            continue

        # 尝试获取对象的源文件路径，并将其与完整对象名称结合
        try:
            src_file = short_path(inspect.getsourcefile(obj))
        except TypeError:
            src_file = None

        # 根据是否成功获取源文件路径构建完整的文件名或对象名称
        if src_file:
            file_full_name = src_file + ':' + full_name
        else:
            file_full_name = full_name

        # 验证文档字符串的 reStructuredText 语法，并将验证结果添加到结果列表
        results.append(
            (full_name,) + validate_rst_syntax(text, file_full_name, dots=dots)
        )

    # 返回处理结果列表
    return results
# 检查分布形状参数的名称与分布方法关键字之间的冲突
def check_dist_keyword_names():
    # 从 distcont 和 distdiscrete 中提取分布名称，放入集合 distnames
    distnames = set(distdata[0] for distdata in distcont + distdiscrete)
    # 初始化结果列表 mod_results
    mod_results = []
    # 遍历 distnames 中的每个分布名称
    for distname in distnames:
        # 从 stats 模块中获取名称为 distname 的属性对象，通常是一个分布对象
        dist = getattr(stats, distname)

        # 获取 dist 对象中的所有方法成员
        method_members = inspect.getmembers(dist, predicate=inspect.ismethod)
        # 提取所有方法名称，不包括以 '_' 开头的方法
        method_names = [method[0] for method in method_members
                        if not method[0].startswith('_')]
        # 遍历每个方法名称
        for methodname in method_names:
            # 获取 dist 对象中名称为 methodname 的方法对象
            method = getattr(dist, methodname)
            try:
                # 尝试从方法的文档字符串中获取 'Parameters' 部分的内容
                params = NumpyDocString(method.__doc__)['Parameters']
            except TypeError:
                # 如果出现类型错误，则说明方法参数文档不完整，生成相关的错误信息并添加到 mod_results 中
                result = (f'stats.{distname}.{methodname}', False,
                          "Method parameters are not documented properly.")
                mod_results.append(result)
                continue

            # 如果分布对象没有形状参数，则跳过冲突检查
            if not dist.shapes:
                continue
            # 将形状参数列表拆分为集合 shape_names
            shape_names = dist.shapes.split(', ')

            # 提取方法的参数名称集合 param_names1
            param_names1 = set(param.name for param in params)
            # 获取方法的签名中的参数名称集合 param_names2
            param_names2 = set(inspect.signature(method).parameters)
            # 将两个参数集合合并为 param_names
            param_names = param_names1.union(param_names2)

            # 检查参数名称集合与形状参数名称集合的交集
            intersection = param_names.intersection(shape_names)

            # 如果存在交集，则表示有冲突，生成相应的错误信息并添加到 mod_results 中
            if intersection:
                message = ("Distribution/method keyword collision: "
                           f"{intersection} ")
                result = (f'stats.{distname}.{methodname}', False, message)
            else:
                # 否则表示没有冲突，生成空消息并添加到 mod_results 中
                result = (f'stats.{distname}.{methodname}', True, '')
            mod_results.append(result)

    # 返回所有结果的列表 mod_results
    return mod_results
    # 遍历给定的模块名称列表
    for submodule_name in module_names:
        # 设置模块名称的前缀
        prefix = BASE_MODULE + '.'
        # 检查子模块名称是否以指定前缀开头，若不是则添加前缀
        if not submodule_name.startswith(prefix):
            module_name = prefix + submodule_name
        else:
            module_name = submodule_name

        # 动态导入模块
        __import__(module_name)
        # 获取导入后的模块对象
        module = sys.modules[module_name]

        # 如果子模块名称不在其他模块文档字典中，则查找模块中的名称
        if submodule_name not in OTHER_MODULE_DOCS:
            find_names(module, names_dict)

        # 如果子模块名称在命令行参数中指定的模块名称列表中，则将模块对象添加到列表中
        if submodule_name in args.module_names:
            modules.append(module)

    # 初始化变量，用于记录检查过程中的状态和结果
    dots = True
    success = True
    results = []

    # 打印即将运行检查的模块数量信息
    print("Running checks for %d modules:" % (len(modules),))

    # 遍历待检查的模块列表
    for module in modules:
        # 如果 dots 变量为 True，则在输出模块名称前打印空格（除第一个模块外）
        if dots:
            if module is not modules[0]:
                sys.stderr.write(' ')
            sys.stderr.write(module.__name__ + ' ')
            sys.stderr.flush()

        # 获取模块中的全部函数、类和其他对象的字典，同时获取已弃用的和其他信息
        all_dict, deprecated, others = get_all_dict(module)
        # 获取当前模块名称对应的名称集合
        names = names_dict.get(module.__name__, set())

        # 存储当前模块检查的结果列表
        mod_results = []
        # 检查模块中的全部对象，记录检查结果
        mod_results += check_items(all_dict, names, deprecated, others, module.__name__)
        # 检查模块中剩余的未被记录的对象，记录检查结果
        mod_results += check_rest(module, set(names).difference(deprecated),
                                  dots=dots)
        # 对于特定模块 scipy.stats，执行额外的检查并记录结果
        if module.__name__ == 'scipy.stats':
            mod_results += check_dist_keyword_names()

        # 确保 mod_results 中每个元素都是元组类型
        for v in mod_results:
            assert isinstance(v, tuple), v

        # 将当前模块的检查结果添加到总结果列表中
        results.append((module, mod_results))

    # 如果 dots 变量为 True，则在模块名称输出完成后换行
    if dots:
        sys.stderr.write("\n")
        sys.stderr.flush()

    # 报告检查结果
    all_success = True

    # 遍历总结果列表，检查每个模块的检查结果
    for module, mod_results in results:
        # 检查当前模块的检查结果是否全部成功
        success = all(x[1] for x in mod_results)
        # 更新总体成功标志
        all_success = all_success and success

        # 如果当前模块检查成功且命令行参数中设置为不输出详细信息，则继续下一个模块
        if success and args.verbose == 0:
            continue

        # 打印当前模块名称及分隔线
        print("")
        print("=" * len(module.__name__))
        print(module.__name__)
        print("=" * len(module.__name__))
        print("")

        # 遍历当前模块的检查结果，打印每个对象的名称及相关输出信息
        for name, success, output in mod_results:
            if name is None:
                # 如果名称为 None，则打印输出信息（除非检查成功且命令行参数设置为不输出）
                if not success or args.verbose >= 1:
                    print(output.strip())
                    print("")
            elif not success or (args.verbose >= 2 and output.strip()):
                # 打印对象名称及相关输出信息
                print(name)
                print("-"*len(name))
                print("")
                print(output.strip())
                print("")

    # 如果所有模块检查成功，则打印通过的消息并退出程序
    if all_success:
        print("\nOK: refguide checks passed!")
        sys.exit(0)
    else:
        # 否则打印错误消息并退出程序
        print("\nERROR: refguide have errors")
        sys.exit(1)
# 如果当前脚本作为主程序被执行（而不是被导入到其他模块中执行），则执行以下代码块
if __name__ == '__main__':
    # 调用主函数 main，并将命令行参数传递给它
    main(argv=sys.argv[1:])
```