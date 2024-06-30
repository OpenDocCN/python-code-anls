# `D:\src\scipysrc\sympy\bin\coverage_doctest.py`

```
#!/usr/bin/env python

"""
Program to test that all methods/functions have at least one example
doctest.  Also checks if docstrings are imported into Sphinx. For this to
work, the Sphinx docs need to be built first.  Use "cd doc; make html" to
build the Sphinx docs.

Usage:

./bin/coverage_doctest.py sympy/core

or

./bin/coverage_doctest.py sympy/core/basic.py

If no arguments are given, all files in sympy/ are checked.
"""

# 导入未来的 print 函数兼容性
from __future__ import print_function

# 导入标准库模块
import os
import sys
import inspect
from argparse import ArgumentParser, RawDescriptionHelpFormatter

# 尝试导入 HTMLParser，Python 3 中为 html.parser
try:
    from HTMLParser import HTMLParser
except ImportError:
    from html.parser import HTMLParser

# 从 sympy.utilities.misc 中导入 filldedent 函数
from sympy.utilities.misc import filldedent

# 加载颜色模板，与 sympy/testing/runtests.py 中的颜色模板重复
color_templates = (
    ("Black", "0;30"),
    ("Red", "0;31"),
    ("Green", "0;32"),
    ("Brown", "0;33"),
    ("Blue", "0;34"),
    ("Purple", "0;35"),
    ("Cyan", "0;36"),
    ("LightGray", "0;37"),
    ("DarkGray", "1;30"),
    ("LightRed", "1;31"),
    ("LightGreen", "1;32"),
    ("Yellow", "1;33"),
    ("LightBlue", "1;34"),
    ("LightPurple", "1;35"),
    ("LightCyan", "1;36"),
    ("White", "1;37"),
)

# 初始化颜色字典
colors = {}
for name, value in color_templates:
    colors[name] = value

# 设置控制台输出格式
c_normal = '\033[0m'
c_color = '\033[%sm'

# 定义打印标题函数，支持颜色输出
def print_header(name, underline=None, color=None):
    print()
    if color:
        print("%s%s%s" % (c_color % colors[color], name, c_normal))
    else:
        print(name)
    if underline and not color:
        print(underline * len(name))

# 定义打印覆盖率详情的函数
def print_coverage(module_path, c, c_missing_doc, c_missing_doctest, c_indirect_doctest, c_sph, f, f_missing_doc, f_missing_doctest,
                   f_indirect_doctest, f_sph, score, total_doctests, total_members,
                   sphinx_score, total_sphinx, verbose=False, no_color=False,
                   sphinx=True):
    """ Prints details (depending on verbose) of a module """

    # 定义不同部分的颜色
    doctest_color = "Brown"
    sphinx_color = "DarkGray"
    less_100_color = "Red"
    less_50_color = "LightRed"
    equal_100_color = "Green"
    big_header_color = "LightPurple"
    small_header_color = "Purple"

    # 根据参数和设置输出覆盖率详情
    if no_color:
        score_string = "Doctests: %s%% (%s of %s)" % (score, total_doctests,
            total_members)
    elif score < 100:
        if score < 50:
            score_string = "%sDoctests:%s %s%s%% (%s of %s)%s" % \
                (c_color % colors[doctest_color], c_normal, c_color % colors[less_50_color], score, total_doctests, total_members, c_normal)
        else:
            score_string = "%sDoctests:%s %s%s%% (%s of %s)%s" % \
                (c_color % colors[doctest_color], c_normal, c_color % colors[less_100_color], score, total_doctests, total_members, c_normal)
    else:
        # 构建包含 Doctest 分数信息的字符串，使用格式化字符串进行格式化
        score_string = "%sDoctests:%s %s%s%% (%s of %s)%s" % \
            (c_color % colors[doctest_color], c_normal, c_color % colors[equal_100_color], score, total_doctests, total_members, c_normal)

    if sphinx:
        # 如果启用 Sphinx 检查
        if no_color:
            # 如果禁用颜色输出，创建包含 Sphinx 分数信息的字符串
            sphinx_score_string = "Sphinx: %s%% (%s of %s)" % (sphinx_score,
                total_members - total_sphinx, total_members)
        elif sphinx_score < 100:
            # 根据 Sphinx 分数设置不同的输出格式
            if sphinx_score < 50:
                # 当 Sphinx 分数小于 50% 时的字符串格式化
                sphinx_score_string = "%sSphinx:%s %s%s%% (%s of %s)%s" % \
                    (c_color % colors[sphinx_color], c_normal, c_color %
                     colors[less_50_color], sphinx_score, total_members - total_sphinx,
                     total_members, c_normal)
            else:
                # 当 Sphinx 分数在 50% 到 100% 之间时的字符串格式化
                sphinx_score_string = "%sSphinx:%s %s%s%% (%s of %s)%s" % \
                    (c_color % colors[sphinx_color], c_normal, c_color %
                     colors[less_100_color], sphinx_score, total_members -
                     total_sphinx, total_members, c_normal)
        else:
            # 当 Sphinx 分数为 100% 时的字符串格式化
            sphinx_score_string = "%sSphinx:%s %s%s%% (%s of %s)%s" % \
                (c_color % colors[sphinx_color], c_normal, c_color %
                 colors[equal_100_color], sphinx_score, total_members -
                 total_sphinx, total_members, c_normal)
    if verbose:
        # 如果启用详细输出，打印模块路径和分隔线
        print('\n' + '-'*70)
        print(module_path)
        print('-'*70)
    else:
        # 如果不启用详细输出
        if sphinx:
            # 如果启用 Sphinx 检查，打印包含 Doctest 和 Sphinx 分数信息的字符串
            print("%s: %s %s" % (module_path, score_string, sphinx_score_string))
        else:
            # 如果未启用 Sphinx 检查，仅打印包含 Doctest 分数信息的字符串
            print("%s: %s" % (module_path, score_string))
    # 如果 verbose 标志为 True，则输出类相关信息的标题，并根据条件打印相应的类信息或警告
    if verbose:
        # 打印类相关信息的大标题，如果不禁用颜色并且设置了大标题颜色
        print_header('CLASSES', '*', not no_color and big_header_color)
        # 如果没有类信息，则打印相应的提示信息
        if not c:
            print_header('No classes found!')
        else:
            # 如果存在缺少文档字符串的类，则打印相关提示和具体信息
            if c_missing_doc:
                print_header('Missing docstrings', '-', not no_color and small_header_color)
                for md in c_missing_doc:
                    print('  * ' + md)
            # 如果存在缺少 doctest 的类，则打印相关提示和具体信息
            if c_missing_doctest:
                print_header('Missing doctests', '-', not no_color and small_header_color)
                for md in c_missing_doctest:
                    print('  * ' + md)
            # 如果存在间接的 doctest 警告，打印相应提示和警告信息，并说明如何在文档字符串中取消此警告
            if c_indirect_doctest:
                print_header('Indirect doctests', '-', not no_color and small_header_color)
                for md in c_indirect_doctest:
                    print('  * ' + md)
                print('\n    Use \"# indirect doctest\" in the docstring to suppress this warning')
            # 如果存在未导入到 Sphinx 文档的类，打印相应提示和信息
            if c_sph:
                print_header('Not imported into Sphinx', '-', not no_color and small_header_color)
                for md in c_sph:
                    print('  * ' + md)

        # 打印函数相关信息的大标题
        print_header('FUNCTIONS', '*', not no_color and big_header_color)
        # 如果没有函数信息，则打印相应的提示信息
        if not f:
            print_header('No functions found!')
        else:
            # 如果存在缺少文档字符串的函数，则打印相关提示和具体信息
            if f_missing_doc:
                print_header('Missing docstrings', '-', not no_color and small_header_color)
                for md in f_missing_doc:
                    print('  * ' + md)
            # 如果存在缺少 doctest 的函数，则打印相关提示和具体信息
            if f_missing_doctest:
                print_header('Missing doctests', '-', not no_color and small_header_color)
                for md in f_missing_doctest:
                    print('  * ' + md)
            # 如果存在间接的 doctest 警告，打印相应提示和警告信息，并说明如何在文档字符串中取消此警告
            if f_indirect_doctest:
                print_header('Indirect doctests', '-', not no_color and small_header_color)
                for md in f_indirect_doctest:
                    print('  * ' + md)
                print('\n    Use \"# indirect doctest\" in the docstring to suppress this warning')
            # 如果存在未导入到 Sphinx 文档的函数，打印相应提示和信息
            if f_sph:
                print_header('Not imported into Sphinx', '-', not no_color and small_header_color)
                for md in f_sph:
                    print('  * ' + md)

    # 如果 verbose 标志为 True，则打印分割线和评分信息，如果开启了 Sphinx，则还打印 Sphinx 相关的评分信息
    if verbose:
        print('\n' + '-'*70)
        # 打印基本的评分信息
        print(score_string)
        # 如果开启了 Sphinx，打印 Sphinx 相关的评分信息
        if sphinx:
            print(sphinx_score_string)
        # 打印分割线
        print('-'*70)
# 判断成员是否包含间接文档，返回布尔值
def _is_indirect(member, doc):
    d = member in doc  # 检查成员是否在文档中出现
    e = 'indirect doctest' in doc  # 检查文档中是否包含'indirect doctest'
    if not d and not e:
        return True  # 如果都不包含，则返回True
    else:
        return False  # 如果任一条件成立，则返回False

# 获取函数对象的参数列表及默认值，处理可变参数和关键字参数
def _get_arg_list(name, fobj):
    trunc = 20  # 有时参数长度可能非常大

    argspec = inspect.getfullargspec(fobj)  # 获取函数对象的参数规范信息

    arg_list = []

    if argspec.args:
        for arg in argspec.args:
            arg_list.append(str(arg))  # 将参数名转为字符串并加入列表中

    arg_list.reverse()

    # 添加默认值
    if argspec.defaults:
        for i in range(len(argspec.defaults)):
            arg_list[i] = str(arg_list[i]) + '=' + str(argspec.defaults[-i])  # 添加参数默认值

    arg_list.reverse()

    # 添加可变参数
    if argspec.varargs:
        arg_list.append(argspec.varargs)

    # 添加关键字参数
    if argspec.varkw:
        arg_list.append(argspec.varkw)

    # 截断过长的参数值
    arg_list = [x[:trunc] for x in arg_list]

    # 构建参数字符串（用括号括起来）
    str_param = "%s(%s)" % (name, ', '.join(arg_list))

    return str_param  # 返回参数字符串

# 根据文件/目录路径和基础路径获取模块名
def get_mod_name(path, base):
    rel_path = os.path.relpath(path, base)  # 获取相对路径

    # 去除文件扩展名
    rel_path, ign = os.path.splitext(rel_path)

    # 将路径分隔符替换为.以构成模块路径
    file_module = ""
    h, t = os.path.split(rel_path)
    while h or t:
        if t:
            file_module = t + '.' + file_module
        h, t = os.path.split(h)

    return file_module[:-1]  # 返回模块名

# 继承HTMLParser类，用于在HTML中查找特定标签（div），并记录其ID属性到列表中
class FindInSphinx(HTMLParser):
    is_imported = []

    def handle_starttag(self, tag, attr):
        a = dict(attr)
        if tag == "div" and a.get('class', None) == "viewcode-block":
            self.is_imported.append(a['id'])

# 在Sphinx生成的HTML文档中查找特定模块的导入情况
def find_sphinx(name, mod_path, found={}):
    if mod_path in found:  # 如果缓存中存在结果
        return name in found[mod_path]  # 直接返回模块是否在导入列表中

    doc_path = mod_path.split('.')  # 将模块路径拆分为路径列表
    doc_path[-1] += '.html'  # 在最后一部分路径中添加.html后缀
    sphinx_path = os.path.join(sympy_top, 'doc', '_build', 'html', '_modules', *doc_path)  # 构建Sphinx文档的路径
    if not os.path.exists(sphinx_path):  # 如果路径不存在则返回False
        return False
    with open(sphinx_path) as f:
        html_txt = f.read()  # 读取Sphinx生成的HTML文档内容
    p = FindInSphinx()
    p.feed(html_txt)  # 解析HTML内容，查找特定标签并记录其ID属性
    found[mod_path] = p.is_imported  # 将模块路径及其导入列表存入缓存
    return name in p.is_imported  # 返回模块是否在导入列表中的结果

# 处理函数以获取有关文档的信息
def process_function(name, c_name, b_obj, mod_path, f_skip, f_missing_doc, f_missing_doctest, f_indirect_doctest,
                     f_has_doctest, skip_list, sph, sphinx=True):
    """
    处理函数以获取有关文档的信息。
    假定调用此子例程的函数已经验证它是有效的模块函数。
    """
    if name in skip_list:  # 如果函数名在跳过列表中，则直接返回False
        return False, False

    # 在最后添加，因为inspect.getsourcelines速度较慢
    # 初始化标志位，用于指示是否需要添加缺失的文档、doctest等
    add_missing_doc = False
    add_missing_doctest = False
    add_indirect_doctest = False
    
    # 标志位，指示是否处于 Sphinx 文档生成环境中
    in_sphinx = True
    
    # 标志位，指示是否包含函数的 doctest
    f_doctest = False
    
    # 标志位，指示是否为函数
    function = False

    # 如果对象是一个类，则获取类中的属性对象和完整名称
    if inspect.isclass(b_obj):
        obj = getattr(b_obj, name)
        obj_name = c_name + '.' + name
    else:
        obj = b_obj
        obj_name = name

    # 获取对象的参数列表的完整名称
    full_name = _get_arg_list(name, obj)

    # 如果函数名以 '_' 开头，将其完整名称添加到跳过列表中
    if name.startswith('_'):
        f_skip.append(full_name)
    else:
        # 获取对象的文档字符串
        doc = obj.__doc__
        if isinstance(doc, str):
            # 如果文档字符串为空，则需要添加缺失文档标志
            if not doc:
                add_missing_doc = True
            # 如果文档字符串中不包含 '>>>'，则需要添加缺失 doctest 标志
            elif not '>>>' in doc:
                add_missing_doctest = True
            # 如果文档字符串中存在间接的 doctest，则需要添加间接 doctest 标志
            elif _is_indirect(name, doc):
                add_indirect_doctest = True
            else:
                # 否则，标记存在函数级别的 doctest
                f_doctest = True
        elif doc is None:
            # 如果文档字符串为 None，则这是在文档字符串中定义的函数
            f_doctest = True
        else:
            # 抛出异常，说明文档类型不符合预期
            raise TypeError('Current doc type for ', print(obj), ' is ', type(doc), '. Docstring must be a string, property, or none')

        # 标记为函数
        function = True

        # 如果处于 Sphinx 环境中，则查找对象的 Sphinx 文档
        if sphinx:
            in_sphinx = find_sphinx(obj_name, mod_path)

    # 如果需要添加缺失文档、doctest，或者不在 Sphinx 环境中
    if add_missing_doc or add_missing_doctest or add_indirect_doctest or not in_sphinx:
        try:
            # 获取对象定义的源代码行号
            line_no = inspect.getsourcelines(obj)[1]
        except IOError:
            # 当源代码不存在时，返回 False
            # 表示函数实际上不存在
            return False, False

        # 构造包含行号的完整名称
        full_name = "LINE %d: %s" % (line_no, full_name)
        
        # 根据需要添加的类型，将完整名称添加到相应的列表中
        if add_missing_doc:
            f_missing_doc.append(full_name)
        elif add_missing_doctest:
            f_missing_doctest.append(full_name)
        elif add_indirect_doctest:
            f_indirect_doctest.append(full_name)
        
        # 如果不在 Sphinx 环境中，将完整名称添加到 sph 列表中
        if not in_sphinx:
            sph.append(full_name)

    # 返回是否存在函数级别的 doctest 和是否为函数的标志位
    return f_doctest, function
# 给定模块路径，构建所有类和函数的索引，并检查每个类和函数的文档和文档测试覆盖率
def coverage(module_path, verbose=False, no_color=False, sphinx=True):
    """
    Given a module path, builds an index of all classes and functions
    contained. It then goes through each of the classes/functions to get
    the docstring and doctest coverage of the module.
    """

    # 导入包并查找其成员
    m = None
    try:
        __import__(module_path)
        m = sys.modules[module_path]
    except Exception as a:
        # 最可能的原因是缺少 __init__
        print("%s could not be loaded due to %s." % (module_path, repr(a)))
        return 0, 0, 0

    # 存储被跳过的类和函数
    c_skipped = []
    c_missing_doc = []
    c_missing_doctest = []
    c_has_doctest = []
    c_indirect_doctest = []
    classes = 0
    c_doctests = 0
    c_sph = []

    f_skipped = []
    f_missing_doc = []
    f_missing_doctest = []
    f_has_doctest = []
    f_indirect_doctest = []
    functions = 0
    f_doctests = 0
    f_sph = []

    # 被跳过的成员列表
    skip_members = ['__abstractmethods__']

    # 获取模块的成员列表
    m_members = dir(m)
    # 遍历给定的模块成员列表
    for member in m_members:

        # 首先检查是否跳过该函数，因为如果与 getattr 结合使用会引发错误
        if member in skip_members:
            continue

        # 获取成员对象，并确定其所属的模块
        obj = getattr(m, member)
        obj_mod = inspect.getmodule(obj)

        # 如果函数不属于当前模块，则跳过处理
        if not obj_mod or not obj_mod.__name__ == module_path:
            continue

        # 如果是函数或方法
        if inspect.isfunction(obj) or inspect.ismethod(obj):

            # 处理函数，获取相关信息并更新统计
            f_dt, f = process_function(member, '', obj, module_path,
                f_skipped, f_missing_doc, f_missing_doctest, f_indirect_doctest, f_has_doctest, skip_members,
                f_sph, sphinx=sphinx)
            if f:
                functions += 1
            if f_dt:
                f_doctests += 1

        # 如果是类，则处理其方法
        elif inspect.isclass(obj):

            # 处理类本身，获取相关信息并更新统计
            c_dt, c, source = process_class(member, obj, c_skipped, c_missing_doc,
                c_missing_doctest, c_indirect_doctest, c_has_doctest, module_path, c_sph, sphinx=sphinx)
            if not c:
                continue
            else:
                classes += 1
            if c_dt:
                c_doctests += 1

            # 遍历类的成员方法
            for f_name in obj.__dict__:

                # 如果成员方法被跳过或以下划线开头，则继续下一个循环
                if f_name in skip_members or f_name.startswith('_'):
                    continue

                # 检查类中是否存在对应的方法定义
                if not ("def " + f_name) in ' '.join(source):
                    continue

                # 获取类成员方法对象，并确定其所属模块
                f_obj = getattr(obj, f_name)
                obj_mod = inspect.getmodule(f_obj)

                # 如果函数不属于当前模块，则跳过处理
                if not obj_mod or not obj_mod.__name__ == module_path:
                    continue

                # 如果是函数或方法
                if inspect.isfunction(f_obj) or inspect.ismethod(f_obj):

                    # 处理函数，获取相关信息并更新统计
                    f_dt, f = process_function(f_name, member, obj,
                        module_path, f_skipped, f_missing_doc, f_missing_doctest, f_indirect_doctest, f_has_doctest,
                        skip_members, f_sph, sphinx=sphinx)
                    if f:
                        functions += 1
                    if f_dt:
                        f_doctests += 1

    # 计算文档测试覆盖率的百分比
    total_doctests = c_doctests + f_doctests
    total_members = classes + functions
    if total_members:
        score = 100 * float(total_doctests) / (total_members)
    else:
        score = 100
    score = int(score)

    # 如果使用了 Sphinx 文档生成工具
    if sphinx:
        total_sphinx = len(c_sph) + len(f_sph)
        if total_members:
            # 计算 Sphinx 文档覆盖率的百分比
            sphinx_score = 100 - 100 * float(total_sphinx) / total_members
        else:
            sphinx_score = 100
        sphinx_score = int(sphinx_score)
    # 否则情况下，初始化总共的 sphinx 数量和 sphinx 得分为 0
    else:
        total_sphinx = 0
        sphinx_score = 0

    # 按行号对函数/类进行排序
    c_missing_doc = sorted(c_missing_doc, key=lambda x: int(x.split()[1][:-1]))
    c_missing_doctest = sorted(c_missing_doctest, key=lambda x: int(x.split()[1][:-1]))
    c_indirect_doctest = sorted(c_indirect_doctest, key=lambda x: int(x.split()[1][:-1]))

    f_missing_doc = sorted(f_missing_doc, key=lambda x: int(x.split()[1][:-1]))
    f_missing_doctest = sorted(f_missing_doctest, key=lambda x: int(x.split()[1][:-1]))
    f_indirect_doctest = sorted(f_indirect_doctest, key=lambda x: int(x.split()[1][:-1]))

    # 打印覆盖率信息，包括模块路径、类信息、缺失文档的类、缺失 doctest 的类、间接 doctest 的类、
    # 类 sphinx 信息、函数信息、缺失文档的函数、缺失 doctest 的函数、间接 doctest 的函数、
    # 函数 sphinx 信息、得分、总 doctest 数、总成员数、
    # sphinx 得分、总 sphinx 数，可以选择是否详细显示、是否使用颜色、是否包含 sphinx 检查
    print_coverage(module_path, classes, c_missing_doc, c_missing_doctest, c_indirect_doctest, c_sph, functions, f_missing_doc,
                   f_missing_doctest, f_indirect_doctest, f_sph, score, total_doctests, total_members,
                   sphinx_score, total_sphinx, verbose=verbose,
                   no_color=no_color, sphinx=sphinx)

    # 返回总 doctest 数、总 sphinx 数、总成员数
    return total_doctests, total_sphinx, total_members
def go(sympy_top, file, verbose=False, no_color=False, exact=True, sphinx=True):
    # 定义一个空列表，用于存放需要跳过的文件路径字符串
    skip_paths = []

    # 如果传入的文件路径是一个目录
    if os.path.isdir(file):
        # 初始化统计变量
        doctests, total_sphinx, num_functions = 0, 0, 0
        # 遍历目录下的所有文件和子目录
        for F in os.listdir(file):
            # 递归调用 go 函数处理子目录或文件，累加返回的结果
            _doctests, _total_sphinx,  _num_functions = go(sympy_top, '%s/%s' % (file, F),
                verbose=verbose, no_color=no_color, exact=False, sphinx=sphinx)
            doctests += _doctests
            total_sphinx += _total_sphinx
            num_functions += _num_functions
        return doctests, total_sphinx, num_functions
    
    # 如果文件不是以 .py 或 .pyx 结尾，或者是 '__init__.py'，或者不精确匹配且包含特定标识的文件，或者在 skip_paths 中指定的文件
    if (not (file.endswith((".py", ".pyx"))) or
        file.endswith('__init__.py') or
        not exact and ('test_' in file or 'bench_' in file or
        any(name in file for name in skip_paths))):
        # 返回统计值为 0
        return 0, 0, 0
    
    # 如果文件不存在
    if not os.path.exists(file):
        # 打印错误信息并退出程序
        print("File(%s does not exist." % file)
        sys.exit(1)

    # 构造模块名的相对路径
    return coverage(get_mod_name(file, sympy_top), verbose=verbose,
        no_color=no_color, sphinx=sphinx)

if __name__ == "__main__":
    # 获取当前文件的绝对路径并设置为 bintest_dir
    bintest_dir = os.path.abspath(os.path.dirname(__file__))   # bin/cover...
    # 获取 bintest_dir 的父目录路径并设置为 sympy_top
    sympy_top = os.path.split(bintest_dir)[0]      # ../
    # 将 sympy_top 和 'sympy' 拼接为 sympy_dir
    sympy_dir = os.path.join(sympy_top, 'sympy')  # ../sympy/
    # 如果 sympy_dir 是一个目录，则将 sympy_top 添加到 sys.path 的开头
    if os.path.isdir(sympy_dir):
        sys.path.insert(0, sympy_top)

    # 设置用法说明字符串
    usage = "usage: ./bin/doctest_coverage.py PATHS"

    # 创建参数解析器对象
    parser = ArgumentParser(
        description=__doc__,
        usage=usage,
        formatter_class=RawDescriptionHelpFormatter,
    )

    # 添加命令行参数 path，默认为 [sympy_top]/sympy
    parser.add_argument("path", nargs='*', default=[os.path.join(sympy_top, 'sympy')])
    # 添加命令行参数 -v 或 --verbose，用于打印详细信息，默认为 False
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
            default=False)
    # 添加命令行参数 --no-colors，用于禁用颜色，默认为 False
    parser.add_argument("--no-colors", action="store_true", dest="no_color",
            help="use no colors", default=False)
    # 添加命令行参数 --no-sphinx，用于禁用 Sphinx 报告，默认为 True
    parser.add_argument("--no-sphinx", action="store_false", dest="sphinx",
            help="don't report Sphinx coverage", default=True)

    # 解析命令行参数
    args = parser.parse_args()

    # 如果启用了 Sphinx 报告但没有构建文档，则输出错误信息并退出程序
    if args.sphinx and not os.path.exists(os.path.join(sympy_top, 'doc', '_build', 'html')):
        print(filldedent("""
            Cannot check Sphinx coverage without a documentation build.
            To build the docs, run "cd doc; make html".  To skip
            checking Sphinx coverage, pass --no-sphinx.
            """))
        sys.exit(1)

    # 设置全局覆盖标志为 True
    full_coverage = True
    # 对于传入的每一个路径，规范化路径格式
    for file in args.path:
        file = os.path.normpath(file)
        # 打印当前处理的文件路径
        print('DOCTEST COVERAGE for %s' % (file))
        # 打印分隔线
        print('='*70)
        print()
        # 调用函数 go() 处理文档测试和 Sphinx 文档覆盖率
        doctests, total_sphinx, num_functions = go(sympy_top, file, verbose=args.verbose,
            no_color=args.no_color, sphinx=args.sphinx)
        
        # 根据函数数量判断分数，若函数数量为 0，则得分为 100
        if num_functions == 0:
            score = 100
            sphinx_score = 100
        else:
            # 计算文档测试覆盖率得分
            score = 100 * float(doctests) / num_functions
            score = int(score)
            # 如果文档测试数小于函数数，则不是全覆盖
            if doctests < num_functions:
                full_coverage = False

            # 如果开启了 Sphinx 选项
            if args.sphinx:
                # 计算 Sphinx 文档覆盖率得分
                sphinx_score = 100 - 100 * float(total_sphinx) / num_functions
                sphinx_score = int(sphinx_score)
                # 如果 Sphinx 文档不全覆盖
                if total_sphinx > 0:
                    full_coverage = False
        
        # 打印分隔线
        print()
        print('='*70)

        # 根据选项打印不同格式的文档测试分数
        if args.no_color:
            print("TOTAL DOCTEST SCORE for %s: %s%% (%s of %s)" % \
                (get_mod_name(file, sympy_top), score, doctests, num_functions))
        elif score < 100:
            print("TOTAL DOCTEST SCORE for %s: %s%s%% (%s of %s)%s" % \
                (get_mod_name(file, sympy_top), c_color % (colors["Red"]),
                score, doctests, num_functions, c_normal))
        else:
            print("TOTAL DOCTEST SCORE for %s: %s%s%% (%s of %s)%s" % \
                (get_mod_name(file, sympy_top), c_color % (colors["Green"]),
                score, doctests, num_functions, c_normal))

        # 如果开启了 Sphinx 选项，打印 Sphinx 文档覆盖率分数
        if args.sphinx:
            if args.no_color:
                print("TOTAL SPHINX SCORE for %s: %s%% (%s of %s)" % \
                    (get_mod_name(file, sympy_top), sphinx_score,
                     num_functions - total_sphinx, num_functions))
            elif sphinx_score < 100:
                print("TOTAL SPHINX SCORE for %s: %s%s%% (%s of %s)%s" % \
                    (get_mod_name(file, sympy_top), c_color % (colors["Red"]),
                    sphinx_score, num_functions - total_sphinx, num_functions, c_normal))
            else:
                print("TOTAL SPHINX SCORE for %s: %s%s%% (%s of %s)%s" % \
                    (get_mod_name(file, sympy_top), c_color % (colors["Green"]),
                    sphinx_score, num_functions - total_sphinx, num_functions, c_normal))

        # 打印空行
        print()
        # 根据全覆盖情况退出程序
        sys.exit(not full_coverage)
```