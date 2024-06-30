# `D:\src\scipysrc\sympy\sympy\testing\tests\test_code_quality.py`

```
# coding=utf-8
# 从os模块导入walk, sep, pardir
from os import walk, sep, pardir
# 从os.path模块导入split, join, abspath, exists, isfile
from os.path import split, join, abspath, exists, isfile
# 从glob模块导入glob
from glob import glob
# 导入re模块，用于处理正则表达式
import re
# 导入random模块，用于生成随机数
import random
# 导入ast模块，用于抽象语法树分析
import ast

# 从sympy.testing.pytest模块导入raises
from sympy.testing.pytest import raises
# 从sympy.testing.quality_unicode模块导入_test_this_file_encoding
from sympy.testing.quality_unicode import _test_this_file_encoding

# System path separator (usually slash or backslash) to be
# used with excluded files, e.g.
#     exclude = set([
#                    "%(sep)smpmath%(sep)s" % sepd,
#                   ])
# 定义变量sepd，包含系统路径分隔符的字典，用于文件排除
sepd = {"sep": sep}

# path and sympy_path
# 设置SYMPY_PATH为当前文件的父目录的父目录的绝对路径，即sympy/
SYMPY_PATH = abspath(join(split(__file__)[0], pardir, pardir))
# 断言确保SYMPY_PATH路径存在
assert exists(SYMPY_PATH)

# 设置TOP_PATH为SYMPY_PATH的父目录的绝对路径
TOP_PATH = abspath(join(SYMPY_PATH, pardir))
# 设置BIN_PATH为TOP_PATH下的bin目录的路径
BIN_PATH = join(TOP_PATH, "bin")
# 设置EXAMPLES_PATH为TOP_PATH下的examples目录的路径
EXAMPLES_PATH = join(TOP_PATH, "examples")

# Error messages
# 定义一系列错误消息字符串，用于代码质量检测
message_space = "File contains trailing whitespace: %s, line %s."
message_implicit = "File contains an implicit import: %s, line %s."
message_tabs = "File contains tabs instead of spaces: %s, line %s."
message_carriage = "File contains carriage returns at end of line: %s, line %s"
message_str_raise = "File contains string exception: %s, line %s"
message_gen_raise = "File contains generic exception: %s, line %s"
message_old_raise = "File contains old-style raise statement: %s, line %s, \"%s\""
message_eof = "File does not end with a newline: %s, line %s"
message_multi_eof = "File ends with more than 1 newline: %s, line %s"
message_test_suite_def = "Function should start with 'test_' or '_': %s, line %s"
message_duplicate_test = "This is a duplicate test function: %s, line %s"
message_self_assignments = "File contains assignments to self/cls: %s, line %s."
message_func_is = "File contains '.func is': %s, line %s."
message_bare_expr = "File contains bare expression: %s, line %s."

# 定义正则表达式对象，用于检测特定的代码风格问题
implicit_test_re = re.compile(r'^\s*(>>> )?(\.\.\. )?from .* import .*\*')
str_raise_re = re.compile(
    r'^\s*(>>> )?(\.\.\. )?raise(\s+(\'|\")|\s*(\(\s*)+(\'|\"))')
gen_raise_re = re.compile(
    r'^\s*(>>> )?(\.\.\. )?raise(\s+Exception|\s*(\(\s*)+Exception)')
old_raise_re = re.compile(r'^\s*(>>> )?(\.\.\. )?raise((\s*\(\s*)|\s+)\w+\s*,')
test_suite_def_re = re.compile(r'^def\s+(?!(_|test))[^(]*\(\s*\)\s*:$')
test_ok_def_re = re.compile(r'^def\s+test_.*:$')
test_file_re = re.compile(r'.*[/\\]test_.*\.py$')
func_is_re = re.compile(r'\.\s*func\s+is')


def tab_in_leading(s):
    """Returns True if there are tabs in the leading whitespace of a line,
    including the whitespace of docstring code samples."""
    # 检查字符串s开头是否包含制表符，包括文档字符串代码示例中的空白部分
    n = len(s) - len(s.lstrip())
    if not s[n:n + 3] in ['...', '>>>']:
        check = s[:n]
    else:
        smore = s[n + 3:]
        check = s[:n] + smore[:len(smore) - len(smore.lstrip())]
    return not (check.expandtabs() == check)


def find_self_assignments(s):
    """Returns a list of "bad" assignments: if there are instances
    of assigning to the first argument of the class method (except
    for staticmethod's).
    """
    t = [n for n in ast.parse(s).body if isinstance(n, ast.ClassDef)]

    bad = []
    # 遍历参数t中的每个元素c，其中c是一个ast节点对象
    for c in t:
        # 遍历节点c的body属性中的每个元素n，body包含节点c的子节点列表
        for n in c.body:
            # 如果n不是ast.FunctionDef类型，则跳过当前循环继续下一个节点
            if not isinstance(n, ast.FunctionDef):
                continue
            # 如果n节点的decorator_list中有任何一个元素是ast.Name类型且其id为'staticmethod'，则跳过当前循环
            if any(d.id == 'staticmethod'
                   for d in n.decorator_list if isinstance(d, ast.Name)):
                continue
            # 如果n节点的名称为'__new__'，则跳过当前循环
            if n.name == '__new__':
                continue
            # 如果n节点的参数args.args为空，则跳过当前循环
            if not n.args.args:
                continue
            # 获取n节点参数列表args.args中的第一个参数，并保存其名称
            first_arg = n.args.args[0].arg

            # 遍历节点n的所有子节点m
            for m in ast.walk(n):
                # 如果子节点m是ast.Assign类型
                if isinstance(m, ast.Assign):
                    # 遍历赋值表达式的目标列表m.targets
                    for a in m.targets:
                        # 如果目标a是ast.Name类型且其id等于first_arg，则将当前节点m添加到列表bad中
                        if isinstance(a, ast.Name) and a.id == first_arg:
                            bad.append(m)
                        # 如果目标a是ast.Tuple类型且其元素列表a.elts中有任何一个元素是ast.Name类型且id等于first_arg，则将当前节点m添加到列表bad中
                        elif (isinstance(a, ast.Tuple) and
                              any(q.id == first_arg for q in a.elts
                                  if isinstance(q, ast.Name))):
                            bad.append(m)

    # 返回收集到的节点列表bad
    return bad
# 检查目录树中指定文件的函数，跳过在排除集合中出现的文件名
def check_directory_tree(base_path, file_check, exclusions=set(), pattern="*.py"):
    """
    Checks all files in the directory tree (with base_path as starting point)
    with the file_check function provided, skipping files that contain
    any of the strings in the set provided by exclusions.
    """
    if not base_path:
        return
    # 遍历目录树，获取每个目录的路径(root)、子目录(dirs)、和文件列表(files)
    for root, dirs, files in walk(base_path):
        # 使用指定的文件名模式在每个目录下获取文件列表，并调用file_check函数检查每个文件
        check_files(glob(join(root, pattern)), file_check, exclusions)


# 检查文件列表中的文件的函数，跳过在排除集合中出现的文件名
def check_files(files, file_check, exclusions=set(), pattern=None):
    """
    Checks all files with the file_check function provided, skipping files
    that contain any of the strings in the set provided by exclusions.
    """
    if not files:
        return
    # 遍历文件列表
    for fname in files:
        # 如果文件不存在或不是常规文件，则跳过
        if not exists(fname) or not isfile(fname):
            continue
        # 如果文件名中包含任何排除集合中的字符串，则跳过
        if any(ex in fname for ex in exclusions):
            continue
        # 如果未提供模式或文件名匹配模式，则调用file_check函数检查文件
        if pattern is None or re.match(pattern, fname):
            file_check(fname)


class _Visit(ast.NodeVisitor):
    """return the line number corresponding to the
    line on which a bare expression appears if it is a binary op
    or a comparison that is not in a with block.

    EXAMPLES
    ========

    >>> import ast
    >>> class _Visit(ast.NodeVisitor):
    ...     def visit_Expr(self, node):
    ...         if isinstance(node.value, (ast.BinOp, ast.Compare)):
    ...             print(node.lineno)
    ...     def visit_With(self, node):
    ...         pass  # no checking there
    ...
    >>> code='''x = 1    # line 1
    ... for i in range(3):
    ...     x == 2       # <-- 3
    ... if x == 2:
    ...     x == 3       # <-- 5
    ...     x + 1        # <-- 6
    ...     x = 1
    ...     if x == 1:
    ...         print(1)
    ... while x != 1:
    ...     x == 1       # <-- 11
    ... with raises(TypeError):
    ...     c == 1
    ...     raise TypeError
    ... assert x == 1
    ... '''
    >>> _Visit().visit(ast.parse(code))
    3
    5
    6
    11
    """
    def visit_Expr(self, node):
        # 如果节点值为二进制运算或比较操作，则打印节点所在的行号
        if isinstance(node.value, (ast.BinOp, ast.Compare)):
            assert None, message_bare_expr % ('', node.lineno)
    def visit_With(self, node):
        # 不执行任何检查
        pass


BareExpr = _Visit()


def line_with_bare_expr(code):
    """return None or else 0-based line number of code on which
    a bare expression appeared.
    """
    # 解析给定的代码，创建抽象语法树
    tree = ast.parse(code)
    try:
        # 调用BareExpr对象访问抽象语法树
        BareExpr.visit(tree)
    except AssertionError as msg:
        assert msg.args
        msg = msg.args[0]
        assert msg.startswith(message_bare_expr.split(':', 1)[0])
        # 返回出现裸表达式的行号（基于0的索引）
        return int(msg.rsplit(' ', 1)[1].rstrip('.'))  # the line number


def test_files():
    """
    """
    This test tests all files in SymPy and checks that:
      o no lines contains a trailing whitespace
      o no lines end with \r\n
      o no line uses tabs instead of spaces
      o that the file ends with a single newline
      o there are no general or string exceptions
      o there are no old style raise statements
      o name of arg-less test suite functions start with _ or test_
      o no duplicate function names that start with test_
      o no assignments to self variable in class methods
      o no lines contain ".func is" except in the test suite
      o there is no do-nothing expression like `a == b` or `x + 1`
    """

    # 定义名为 test 的函数，接收文件名作为参数
    def test(fname):
        # 以 UTF-8 编码打开文件，并调用 test_this_file 函数进行测试
        with open(fname, encoding="utf8") as test_file:
            test_this_file(fname, test_file)
        # 再次以 UTF-8 编码打开文件，并调用 _test_this_file_encoding 函数进行测试
        with open(fname, encoding='utf8') as test_file:
            _test_this_file_encoding(fname, test_file)
    # 定义一个测试函数，用于检查指定文件中的测试情况
    def test_this_file(fname, test_file):
        idx = None  # 初始化行索引为 None
        code = test_file.read()  # 读取测试文件的所有内容
        test_file.seek(0)  # 将文件读取指针定位到文件头部，准备重新读取文件

        # 根据操作系统路径分隔符判断 Python 文件名
        py = fname if sep not in fname else fname.rsplit(sep, 1)[-1]

        # 如果文件名以 'test_' 开头，则查找单独表达式的行号
        if py.startswith('test_'):
            idx = line_with_bare_expr(code)

        # 如果找到了单独表达式的行号，则断言测试失败，给出相应的错误信息
        if idx is not None:
            assert False, message_bare_expr % (fname, idx + 1)

        line = None  # 初始化当前行为 None，用于标记文件为空的情况
        tests = 0  # 初始化测试计数器
        test_set = set()  # 使用集合记录测试函数名，避免重复
        for idx, line in enumerate(test_file):  # 遍历测试文件的每一行，同时记录行号和内容

            # 如果文件名符合测试文件的正则表达式，则进行以下检查
            if test_file_re.match(fname):

                # 如果当前行匹配测试套件定义的正则表达式，则断言测试失败
                if test_suite_def_re.match(line):
                    assert False, message_test_suite_def % (fname, idx + 1)

                # 如果当前行匹配测试函数定义成功能的正则表达式，则进行以下检查
                if test_ok_def_re.match(line):
                    tests += 1  # 增加测试计数
                    test_set.add(line[3:].split('(')[0].strip())  # 将测试函数名添加到集合中

                    # 如果集合中的元素数量不等于测试计数，表示有重复的测试函数名，则断言测试失败
                    if len(test_set) != tests:
                        assert False, message_duplicate_test % (fname, idx + 1)

            # 如果当前行以空格或制表符结尾，则断言测试失败，给出相应的错误信息
            if line.endswith((" \n", "\t\n")):
                assert False, message_space % (fname, idx + 1)

            # 如果当前行以回车换行符结尾，则断言测试失败，给出相应的错误信息
            if line.endswith("\r\n"):
                assert False, message_carriage % (fname, idx + 1)

            # 如果当前行开头有制表符，则断言测试失败，给出相应的错误信息
            if tab_in_leading(line):
                assert False, message_tabs % (fname, idx + 1)

            # 如果当前行匹配字符串引发异常的正则表达式，则断言测试失败，给出相应的错误信息
            if str_raise_re.search(line):
                assert False, message_str_raise % (fname, idx + 1)

            # 如果当前行匹配生成器引发异常的正则表达式，则断言测试失败，给出相应的错误信息
            if gen_raise_re.search(line):
                assert False, message_gen_raise % (fname, idx + 1)

            # 如果当前行匹配隐式测试的正则表达式，并且文件名不在排除列表中，则断言测试失败，给出相应的错误信息
            if (implicit_test_re.search(line) and
                    not list(filter(lambda ex: ex in fname, import_exclude))):
                assert False, message_implicit % (fname, idx + 1)

            # 如果当前行匹配函数是否正则表达式，并且文件名不匹配测试文件的正则表达式，则断言测试失败，给出相应的错误信息
            if func_is_re.search(line) and not test_file_re.search(fname):
                assert False, message_func_is % (fname, idx + 1)

            # 如果当前行匹配旧式异常引发的正则表达式，则断言测试失败，给出相应的错误信息
            result = old_raise_re.search(line)
            if result is not None:
                assert False, message_old_raise % (
                    fname, idx + 1, result.group(2))

        # 如果文件不为空，则进行额外的文件结尾检查
        if line is not None:
            # 如果最后一行是空行且行号大于零，则断言测试失败，给出相应的错误信息
            if line == '\n' and idx > 0:
                assert False, message_multi_eof % (fname, idx + 1)
            # 如果最后一行不是以换行符结尾，则断言测试失败，给出相应的错误信息
            elif not line.endswith('\n'):
                assert False, message_eof % (fname, idx + 1)


    # 在顶层测试的文件列表
    top_level_files = [join(TOP_PATH, file) for file in [
        "isympy.py",
        "build.py",
        "setup.py",
    ]]
    # 从所有测试中排除的文件列表
    # 创建一个集合，用于存放需要排除的文件名，这些文件名是通过格式化字符串生成的，每个元素是一个字符串
    exclude = {
        "%(sep)ssympy%(sep)sparsing%(sep)sautolev%(sep)s_antlr%(sep)sautolevparser.py" % sepd,
        "%(sep)ssympy%(sep)sparsing%(sep)sautolev%(sep) s_antlr%(sep)sautolevlexer.py" % sepd,
        "%(sep)ssympy%(sep)sparsing%(sep)sautolev%(sep)s_antlr%(sep)sautolevlistener.py" % sepd,
        "%(sep)ssympy%(sep)sparsing%(sep)slatex%(sep)s_antlr%(sep)slatexparser.py" % sepd,
        "%(sep)ssympy%(sep)sparsing%(sep)slatex%(sep)s_antlr%(sep)slatexlexer.py" % sepd,
    }
    # 需要从隐式导入测试中排除的文件集合
    
    # 创建一个集合，用于存放需要排除的导入文件，这些文件名是通过格式化字符串生成的，每个元素是一个字符串
    import_exclude = {
        # 允许在顶层 __init__.py 中使用 glob 导入：
        "%(sep)ssympy%(sep)s__init__.py" % sepd,
        # 下面的 __init__.py 应该修复：
        # XXX: 实际上，它们使用了有用的导入模式（DRY）
        "%(sep)svector%(sep)s__init__.py" % sepd,
        "%(sep)smechanics%(sep)s__init__.py" % sepd,
        "%(sep)squantum%(sep)s__init__.py" % sepd,
        "%(sep)spolys%(sep)s__init__.py" % sepd,
        "%(sep)spolys%(sep)sdomains%(sep)s__init__.py" % sepd,
        # 交互式 SymPy 执行 ``from sympy import *``：
        "%(sep)sinteractive%(sep)ssession.py" % sepd,
        # isympy.py 执行 ``from sympy import *``：
        "%(sep)sisympy.py" % sepd,
        # 下面两个是导入时间测试：
        "%(sep)sbin%(sep)ssympy_time.py" % sepd,
        "%(sep)sbin%(sep)ssympy_time_cache.py" % sepd,
        # 来自 Python 标准库的文件：
        "%(sep)sparsing%(sep)ssympy_tokenize.py" % sepd,
        # 应该修复的文件：
        "%(sep)splotting%(sep)spygletplot%(sep)s" % sepd,
        # 文档字符串中的误报
        "%(sep)sbin%(sep)stest_external_imports.py" % sepd,
        "%(sep)sbin%(sep)stest_submodule_imports.py" % sepd,
        # 这些是可以在某个时候移除的已弃用存根：
        "%(sep)sutilities%(sep)sruntests.py" % sepd,
        "%(sep)sutilities%(sep)spytest.py" % sepd,
        "%(sep)sutilities%(sep)srandtest.py" % sepd,
        "%(sep)sutilities%(sep)stmpfiles.py" % sepd,
        "%(sep)sutilities%(sep)squality_unicode.py" % sepd,
    }
    # 需要从隐式导入中排除的文件集合
    
    # 使用指定的函数检查顶层文件中的内容，并进行测试
    check_files(top_level_files, test)
    # 使用指定的函数检查指定路径下的目录结构，并进行测试，排除了指定的文件扩展名
    check_directory_tree(BIN_PATH, test, {"~", ".pyc", ".sh", ".mjs"}, "*")
    # 使用指定的函数检查指定路径下的目录结构，并进行测试，排除了指定的文件集合
    check_directory_tree(SYMPY_PATH, test, exclude)
    # 使用指定的函数检查指定路径下的目录结构，并进行测试，排除了指定的文件集合
    check_directory_tree(EXAMPLES_PATH, test, exclude)
def _with_space(c):
    # 返回字符串 c，带有随机数量的前导空格
    return random.randint(0, 10)*' ' + c


def test_raise_statement_regular_expression():
    candidates_ok = [
        "some text # raise Exception, 'text'",
        "raise ValueError('text') # raise Exception, 'text'",
        "raise ValueError('text')",
        "raise ValueError",
        "raise ValueError('text')",
        "raise ValueError('text') #,",
        # 在文档字符串中讨论异常
        ''''"""This function will raise ValueError, except when it doesn't"""''',
        "raise (ValueError('text')",
    ]
    str_candidates_fail = [
        "raise 'exception'",
        "raise 'Exception'",
        'raise "exception"',
        'raise "Exception"',
        "raise 'ValueError'",
    ]
    gen_candidates_fail = [
        "raise Exception('text') # raise Exception, 'text'",
        "raise Exception('text')",
        "raise Exception",
        "raise Exception('text')",
        "raise Exception('text') #,",
        "raise Exception, 'text'",
        "raise Exception, 'text' # raise Exception('text')",
        "raise Exception, 'text' # raise Exception, 'text'",
        ">>> raise Exception, 'text'",
        ">>> raise Exception, 'text' # raise Exception('text')",
        ">>> raise Exception, 'text' # raise Exception, 'text'",
    ]
    old_candidates_fail = [
        "raise Exception, 'text'",
        "raise Exception, 'text' # raise Exception('text')",
        "raise Exception, 'text' # raise Exception, 'text'",
        ">>> raise Exception, 'text'",
        ">>> raise Exception, 'text' # raise Exception('text')",
        ">>> raise Exception, 'text' # raise Exception, 'text'",
        "raise ValueError, 'text'",
        "raise ValueError, 'text' # raise Exception('text')",
        "raise ValueError, 'text' # raise Exception, 'text'",
        ">>> raise ValueError, 'text'",
        ">>> raise ValueError, 'text' # raise Exception('text')",
        ">>> raise ValueError, 'text' # raise Exception, 'text'",
        "raise(ValueError,",
        "raise (ValueError,",
        "raise( ValueError,",
        "raise ( ValueError,",
        "raise(ValueError ,",
        "raise (ValueError ,",
        "raise( ValueError ,",
        "raise ( ValueError ,",
    ]

    # 对于 candidates_ok 中的每个语句，确保三种正则表达式都不匹配
    for c in candidates_ok:
        assert str_raise_re.search(_with_space(c)) is None, c
        assert gen_raise_re.search(_with_space(c)) is None, c
        assert old_raise_re.search(_with_space(c)) is None, c
    
    # 对于 str_candidates_fail 中的每个语句，确保 str_raise_re 可以匹配
    for c in str_candidates_fail:
        assert str_raise_re.search(_with_space(c)) is not None, c
    
    # 对于 gen_candidates_fail 中的每个语句，确保 gen_raise_re 可以匹配
    for c in gen_candidates_fail:
        assert gen_raise_re.search(_with_space(c)) is not None, c
    
    # 对于 old_candidates_fail 中的每个语句，确保 old_raise_re 可以匹配
    for c in old_candidates_fail:
        assert old_raise_re.search(_with_space(c)) is not None, c


def test_implicit_imports_regular_expression():
    # 以下是用于测试的候选代码列表，这些代码应该符合特定的模式以通过测试。
    candidates_ok = [
        "from sympy import something",  # 符合格式 "from sympy import something"
        ">>> from sympy import something",  # 符合格式 ">>> from sympy import something"
        "from sympy.somewhere import something",  # 符合格式 "from sympy.somewhere import something"
        ">>> from sympy.somewhere import something",  # 符合格式 ">>> from sympy.somewhere import something"
        "import sympy",  # 符合格式 "import sympy"
        ">>> import sympy",  # 符合格式 ">>> import sympy"
        "import sympy.something.something",  # 符合格式 "import sympy.something.something"
        "... import sympy",  # 符合格式 "... import sympy"
        "... import sympy.something.something",  # 符合格式 "... import sympy.something.something"
        "... from sympy import something",  # 符合格式 "... from sympy import something"
        "... from sympy.somewhere import something",  # 符合格式 "... from sympy.somewhere import something"
        ">> from sympy import *",  # 符合格式 ">> from sympy import *"，用于允许 'fake' docstrings
        "# from sympy import *",  # 符合格式 "# from sympy import *"
        "some text # from sympy import *",  # 符合格式 "some text # from sympy import *"
    ]
    
    # 以下是用于测试的候选代码列表，这些代码应该不符合特定的模式以通过测试。
    candidates_fail = [
        "from sympy import *",  # 不符合格式 "from sympy import *"
        ">>> from sympy import *",  # 不符合格式 ">>> from sympy import *"
        "from sympy.somewhere import *",  # 不符合格式 "from sympy.somewhere import *"
        ">>> from sympy.somewhere import *",  # 不符合格式 ">>> from sympy.somewhere import *"
        "... from sympy import *",  # 不符合格式 "... from sympy import *"
        "... from sympy.somewhere import *",  # 不符合格式 "... from sympy.somewhere import *"
    ]
    
    # 对于每个通过测试的候选代码，确保其不能匹配隐式导入的正则表达式，否则断言失败。
    for c in candidates_ok:
        assert implicit_test_re.search(_with_space(c)) is None, c
    
    # 对于每个不通过测试的候选代码，确保其匹配隐式导入的正则表达式，否则断言失败。
    for c in candidates_fail:
        assert implicit_test_re.search(_with_space(c)) is not None, c
# 定义函数 test_test_suite_defs，用于测试测试套件定义的函数
def test_test_suite_defs():
    # 定义通过的候选列表，包含正确定义的函数示例
    candidates_ok = [
        "    def foo():\n",
        "def foo(arg):\n",
        "def _foo():\n",
        "def test_foo():\n",
    ]
    # 定义失败的候选列表，包含不正确定义的函数示例
    candidates_fail = [
        "def foo():\n",
        "def foo() :\n",
        "def foo( ):\n",
        "def  foo():\n",
    ]
    # 对于每个通过的候选，验证其不匹配测试套件定义的正则表达式
    for c in candidates_ok:
        assert test_suite_def_re.search(c) is None, c
    # 对于每个失败的候选，验证其匹配测试套件定义的正则表达式
    for c in candidates_fail:
        assert test_suite_def_re.search(c) is not None, c


# 定义函数 test_test_duplicate_defs，用于测试重复定义的函数
def test_test_duplicate_defs():
    # 定义通过的候选列表，包含没有重复定义的函数示例
    candidates_ok = [
        "def foo():\ndef foo():\n",
        "def test():\ndef test_():\n",
        "def test_():\ndef test__():\n",
    ]
    # 定义失败的候选列表，包含有重复定义的函数示例
    candidates_fail = [
        "def test_():\ndef test_ ():\n",
        "def test_1():\ndef  test_1():\n",
    ]
    # 定义成功的结果
    ok = (None, 'check')
    
    # 定义函数 check，用于检查文件中的重复定义
    def check(file):
        tests = 0
        test_set = set()
        for idx, line in enumerate(file.splitlines()):
            if test_ok_def_re.match(line):
                tests += 1
                test_set.add(line[3:].split('(')[0].strip())
                if len(test_set) != tests:
                    return False, message_duplicate_test % ('check', idx + 1)
        return None, 'check'
    
    # 对于每个通过的候选，验证其结果为成功
    for c in candidates_ok:
        assert check(c) == ok
    # 对于每个失败的候选，验证其结果不为成功
    for c in candidates_fail:
        assert check(c) != ok


# 定义函数 test_find_self_assignments，用于测试自我赋值的检测函数
def test_find_self_assignments():
    # 定义通过的候选列表，包含没有自我赋值的类方法示例
    candidates_ok = [
        "class A(object):\n    def foo(self, arg): arg = self\n",
        "class A(object):\n    def foo(self, arg): self.prop = arg\n",
        "class A(object):\n    def foo(self, arg): obj, obj2 = arg, self\n",
        "class A(object):\n    @classmethod\n    def bar(cls, arg): arg = cls\n",
        "class A(object):\n    def foo(var, arg): arg = var\n",
    ]
    # 定义失败的候选列表，包含有自我赋值的类方法示例
    candidates_fail = [
        "class A(object):\n    def foo(self, arg): self = arg\n",
        "class A(object):\n    def foo(self, arg): obj, self = arg, arg\n",
        "class A(object):\n    def foo(self, arg):\n        if arg: self = arg",
        "class A(object):\n    @classmethod\n    def foo(cls, arg): cls = arg\n",
        "class A(object):\n    def foo(var, arg): var = arg\n",
    ]
    
    # 对于每个通过的候选，验证其结果为空列表（即没有自我赋值）
    for c in candidates_ok:
        assert find_self_assignments(c) == []
    # 对于每个失败的候选，验证其结果不为空列表（即存在自我赋值）
    for c in candidates_fail:
        assert find_self_assignments(c) != []


# 定义函数 test_test_unicode_encoding，用于测试测试文件的编码
def test_test_unicode_encoding():
    unicode_whitelist = ['foo']
    unicode_strict_whitelist = ['bar']

    # 第一个测试文件名和内容，预期抛出 AssertionError
    fname = 'abc'
    test_file = ['α']
    raises(AssertionError, lambda: _test_this_file_encoding(
        fname, test_file, unicode_whitelist, unicode_strict_whitelist))

    # 第二个测试文件名和内容，预期不抛出异常
    fname = 'abc'
    test_file = ['abc']
    _test_this_file_encoding(
        fname, test_file, unicode_whitelist, unicode_strict_whitelist)

    # 第三个测试文件名和内容，预期抛出 AssertionError
    fname = 'foo'
    test_file = ['abc']
    raises(AssertionError, lambda: _test_this_file_encoding(
        fname, test_file, unicode_whitelist, unicode_strict_whitelist))

    # 第四个测试文件名和内容，未提供预期结果部分的代码，无法继续
    # 调用名为 _test_this_file_encoding 的函数，并传入以下参数：
    # - fname: 变量或值，可能是文件名或路径
    # - test_file: 变量或值，可能是测试文件相关的参数
    # - unicode_whitelist: 变量或值，可能是与Unicode相关的白名单
    # - unicode_strict_whitelist: 变量或值，可能是与Unicode严格相关的白名单
    _test_this_file_encoding(
        fname, test_file, unicode_whitelist, unicode_strict_whitelist)
```