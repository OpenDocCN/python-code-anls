# `D:\src\scipysrc\scipy\scipy\_lib\tests\test_warnings.py`

```
"""
Tests which scan for certain occurrences in the code, they may not find
all of these occurrences but should catch almost all. This file was adapted
from NumPy.
"""

# 导入必要的库
import os
from pathlib import Path
import ast
import tokenize

# 导入 SciPy 库
import scipy

# 导入 Pytest 测试框架
import pytest


# 定义一个 AST 节点访问器 ParseCall
class ParseCall(ast.NodeVisitor):
    def __init__(self):
        self.ls = []  # 初始化一个空列表

    # 处理访问 Attribute 节点的方法
    def visit_Attribute(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        self.ls.append(node.attr)  # 将属性名添加到列表中

    # 处理访问 Name 节点的方法
    def visit_Name(self, node):
        self.ls.append(node.id)  # 将名称添加到列表中


# 定义一个 AST 节点访问器 FindFuncs
class FindFuncs(ast.NodeVisitor):
    def __init__(self, filename):
        super().__init__()
        self.__filename = filename  # 初始化文件名
        self.bad_filters = []  # 初始化坏的过滤器列表
        self.bad_stacklevels = []  # 初始化坏的堆栈级别列表

    # 处理访问 Call 节点的方法
    def visit_Call(self, node):
        p = ParseCall()  # 创建 ParseCall 实例
        p.visit(node.func)  # 访问函数节点
        ast.NodeVisitor.generic_visit(self, node)

        # 检查函数调用的最后一个部分是否为 'simplefilter' 或 'filterwarnings'
        if p.ls[-1] == 'simplefilter' or p.ls[-1] == 'filterwarnings':
            # 获取过滤器调用的第一个参数
            match node.args[0]:
                case ast.Constant() as c:
                    argtext = c.value
                case ast.JoinedStr() as js:
                    # 如果是 f-string，则只取常量部分
                    argtext = "".join(
                        x.value for x in js.values if isinstance(x, ast.Constant)
                    )
                case _:
                    raise ValueError("unknown ast node type")

            # 检查过滤器是否设置为 'ignore'
            if argtext == "ignore":
                self.bad_filters.append(
                    f"{self.__filename}:{node.lineno}")  # 将错误信息添加到坏的过滤器列表中

        # 检查函数调用是否为 'warnings.warn'，且如果存在的话，检查其堆栈级别
        if p.ls[-1] == 'warn' and (
                len(p.ls) == 1 or p.ls[-2] == 'warnings'):

            if self.__filename == "_lib/tests/test_warnings.py":
                # 如果是特定文件则跳过检查
                return

            # 检查是否存在 stacklevel 参数
            if len(node.args) == 3:
                return
            args = {kw.arg for kw in node.keywords}
            if "stacklevel" not in args:
                self.bad_stacklevels.append(
                    f"{self.__filename}:{node.lineno}")  # 将错误信息添加到坏的堆栈级别列表中


# 定义一个 Pytest 的 session 级别的 fixture
@pytest.fixture(scope="session")
def warning_calls():
    base = Path(scipy.__file__).parent  # 获取 SciPy 库所在路径的父目录路径

    bad_filters = []  # 初始化坏的过滤器列表
    bad_stacklevels = []  # 初始化坏的堆栈级别列表

    # 遍历基础路径下所有的 Python 文件
    for path in base.rglob("*.py"):
        # 使用 tokenize 打开文件，自动检测编码
        with tokenize.open(str(path)) as file:
            tree = ast.parse(file.read(), filename=str(path))  # 解析文件内容为 AST
            finder = FindFuncs(path.relative_to(base))  # 创建 FindFuncs 实例
            finder.visit(tree)  # 访问 AST 树
            bad_filters.extend(finder.bad_filters)  # 扩展坏的过滤器列表
            bad_stacklevels.extend(finder.bad_stacklevels)  # 扩展坏的堆栈级别列表
    # 返回两个变量 bad_filters 和 bad_stacklevels
    return bad_filters, bad_stacklevels
@pytest.mark.fail_slow(40)
@pytest.mark.slow
def test_warning_calls_filters(warning_calls):
    bad_filters, bad_stacklevels = warning_calls

    # 不要在代码库中添加过滤器，因为这些过滤器不是线程安全的。我们的目标是仅在测试中使用 np.testing.suppress_warnings 来进行过滤。
    # 然而，在某些情况下，可能需要过滤警告，因为我们无法（轻易地）修复其根本原因，而且我们不希望用户在正确使用 SciPy 时看到某些警告。
    # 因此，在这里列出例外情况。仅在确有必要时添加新条目。

    # 允许的过滤器列表
    allowed_filters = (
        os.path.join('datasets', '_fetchers.py'),
        os.path.join('datasets', '__init__.py'),
        os.path.join('optimize', '_optimize.py'),
        os.path.join('optimize', '_constraints.py'),
        os.path.join('optimize', '_nnls.py'),
        os.path.join('signal', '_ltisys.py'),
        os.path.join('sparse', '__init__.py'),  # np.matrix pending-deprecation
        os.path.join('stats', '_discrete_distns.py'),  # gh-14901
        os.path.join('stats', '_continuous_distns.py'),
        os.path.join('stats', '_binned_statistic.py'),  # gh-19345
        os.path.join('stats', '_stats_py.py'),  # gh-20743
        os.path.join('stats', 'tests', 'test_axis_nan_policy.py'),  # gh-20694
        os.path.join('_lib', '_util.py'),  # gh-19341
        os.path.join('sparse', 'linalg', '_dsolve', 'linsolve.py'),  # gh-17924
        "conftest.py",
    )

    # 过滤掉不在允许列表中的不良过滤器条目
    bad_filters = [item for item in bad_filters if item.split(':')[0] not in allowed_filters]

    # 如果存在不良过滤器，抛出断言错误
    if bad_filters:
        raise AssertionError(
            "warning ignore filter should not be used, instead, use\n"
            "numpy.testing.suppress_warnings (in tests only);\n"
            "found in:\n    {}".format(
                "\n    ".join(bad_filters)))
```