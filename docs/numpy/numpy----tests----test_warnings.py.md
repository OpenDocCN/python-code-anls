# `.\numpy\numpy\tests\test_warnings.py`

```py
"""
Tests which scan for certain occurrences in the code, they may not find
all of these occurrences but should catch almost all.
"""
# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 Path 类，用于处理文件路径
from pathlib import Path

# 导入 ast 模块，用于解析 Python 抽象语法树
import ast

# 导入 tokenize 模块，用于解析 Python 源代码的词法分析
import tokenize

# 导入 numpy 库，用于获取 numpy 模块的安装路径
import numpy


# 定义 ParseCall 类，继承自 ast.NodeVisitor，用于解析函数调用时的节点
class ParseCall(ast.NodeVisitor):
    def __init__(self):
        self.ls = []  # 初始化空列表，用于存储函数调用时的属性和名称

    def visit_Attribute(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        self.ls.append(node.attr)  # 记录属性名称

    def visit_Name(self, node):
        self.ls.append(node.id)  # 记录名称


# 定义 FindFuncs 类，继承自 ast.NodeVisitor，用于查找特定函数调用
class FindFuncs(ast.NodeVisitor):
    def __init__(self, filename):
        super().__init__()
        self.__filename = filename  # 存储当前文件名

    def visit_Call(self, node):
        p = ParseCall()
        p.visit(node.func)  # 解析函数调用
        ast.NodeVisitor.generic_visit(self, node)

        # 检查特定函数调用情况
        if p.ls[-1] == 'simplefilter' or p.ls[-1] == 'filterwarnings':
            if node.args[0].value == "ignore":
                raise AssertionError(
                    "warnings should have an appropriate stacklevel; found in "
                    "{} on line {}".format(self.__filename, node.lineno))

        if p.ls[-1] == 'warn' and (
                len(p.ls) == 1 or p.ls[-2] == 'warnings'):

            if "testing/tests/test_warnings.py" == self.__filename:
                # 在特定测试文件中，忽略检查
                return

            # 检查是否存在 stacklevel 参数
            if len(node.args) == 3:
                return
            args = {kw.arg for kw in node.keywords}
            if "stacklevel" in args:
                return
            raise AssertionError(
                "warnings should have an appropriate stacklevel; found in "
                "{} on line {}".format(self.__filename, node.lineno))


@pytest.mark.slow
def test_warning_calls():
    # combined "ignore" and stacklevel error
    base = Path(numpy.__file__).parent  # 获取 numpy 模块的安装路径

    # 遍历指定目录下的 Python 文件
    for path in base.rglob("*.py"):
        if base / "testing" in path.parents:
            continue
        if path == base / "__init__.py":
            continue
        if path == base / "random" / "__init__.py":
            continue
        if path == base / "conftest.py":
            continue
        # 使用 tokenize.open 打开文件，自动检测编码
        with tokenize.open(str(path)) as file:
            tree = ast.parse(file.read())  # 解析文件内容为抽象语法树
            FindFuncs(path).visit(tree)  # 查找特定函数调用
```