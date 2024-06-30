# `D:\src\scipysrc\scipy\tools\check_test_name.py`

```
"""
MIT License

Copyright (c) 2020 Marco Gorelli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Check that test names start with `test`, and that test classes start with
`Test`.
"""

from __future__ import annotations  # 允许类型注解支持循环引用

import ast  # 引入抽象语法树模块
import os  # 引入操作系统相关功能模块
from pathlib import Path  # 引入处理文件和目录路径的模块
import sys  # 引入系统相关功能模块
from collections.abc import Iterator, Sequence  # 引入抽象基类模块
import itertools  # 引入迭代器模块

PRAGMA = "# skip name check"  # 定义一个跳过名称检查的标志字符串


def _find_names(node: ast.Module) -> Iterator[str]:
    # 遍历抽象语法树节点，返回所有的名称
    for _node in ast.walk(node):
        if isinstance(_node, ast.Name):
            yield _node.id
        elif isinstance(_node, ast.Attribute):
            yield _node.attr


def _is_fixture(node: ast.expr) -> bool:
    # 判断节点是否为 pytest 的 fixture
    if isinstance(node, ast.Call):
        node = node.func
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "fixture"
        and isinstance(node.value, ast.Name)
        and node.value.id == "pytest"
    )


def is_misnamed_test_func(
    node: ast.expr | ast.stmt, names: Sequence[str], line: str
) -> bool:
    # 判断是否为命名不规范的测试函数
    return (
        isinstance(node, ast.FunctionDef)
        and not node.name.startswith("test")
        and names.count(node.name) == 0
        and not any(
            _is_fixture(decorator) for decorator in node.decorator_list
            )
        and PRAGMA not in line
        and node.name
        not in ("teardown_method", "setup_method",
                "teardown_class", "setup_class",
                "setup_module", "teardown_module")
    )


def is_misnamed_test_class(
    node: ast.expr | ast.stmt, names: Sequence[str], line: str
) -> bool:
    # 判断是否为命名不规范的测试类
    return (
        isinstance(node, ast.ClassDef)
        and not node.name.startswith("Test")
        and names.count(node.name) == 0
        and PRAGMA not in line
        # Some of the KDTreeTest use a decorator to setup tests so these are
        # actually fine
        and "KDTreeTest" not in [
            decorator.id for decorator in node.decorator_list
            ]
    )


def main(content: str, file: str) -> int:
    # 将内容按行分割，形成字符串列表
    lines = content.splitlines()
    # 解析代码内容，生成抽象语法树（AST）
    tree = ast.parse(content)
    # 查找并列出所有的标识符名称
    names = list(_find_names(tree))
    # 初始化返回值为0，用于记录发现的问题数量
    ret = 0
    # 遍历抽象语法树中的每一个顶层节点
    for node in tree.body:
        # 检查当前节点是否是命名不规范的测试函数
        if is_misnamed_test_func(node, names, lines[node.lineno - 1]):
            # 打印命名不规范的测试函数的相关信息
            print(
                f"{file}:{node.lineno}:{node.col_offset} "
                f"found test function '{node.name}' which does not start with"
                " 'test'"
            )
            # 设置返回值为1，表示发现了问题
            ret = 1
        # 检查当前节点是否是命名不规范的测试类
        elif is_misnamed_test_class(node, names, lines[node.lineno - 1]):
            # 打印命名不规范的测试类的相关信息
            print(
                f"{file}:{node.lineno}:{node.col_offset} "
                f"found test class '{node.name}' which does not start with"
                " 'Test'"
            )
            # 设置返回值为1，表示发现了问题
            ret = 1
        # 如果当前节点是类定义，并且类名不在标识符名称列表中，并且没有包含特定的预处理语句
        if (
            isinstance(node, ast.ClassDef)
            and names.count(node.name) == 0
            and PRAGMA not in lines[node.lineno - 1]
        ):
            # 遍历当前类定义节点的每一个子节点
            for _node in node.body:
                # 检查当前子节点是否是命名不规范的测试函数
                if is_misnamed_test_func(_node, names,
                                         lines[_node.lineno - 1]):
                    # 检查当前函数是否在测试类的任何位置使用，以避免误报
                    should_continue = False
                    # 遍历指定路径下的所有测试文件及一个特定的文件
                    for _file in itertools.chain(
                        Path("scipy").rglob("**/tests/**/test*.py"),
                        ["scipy/_lib/_testutils.py"],
                    ):
                        # 打开并读取测试文件内容
                        with open(os.path.join(_file)) as fd:
                            _content = fd.read()
                        # 如果当前函数被测试类中的某处使用，则继续下一个节点的检查
                        if f"self.{_node.name}" in _content:
                            should_continue = True
                            break
                    # 如果需要继续，则跳过当前节点的检查
                    if should_continue:
                        continue
    
                    # 打印命名不规范的测试函数的相关信息
                    print(
                        f"{file}:{_node.lineno}:{_node.col_offset} "
                        f"found test function '{_node.name}' which does not "
                        "start with 'test'"
                    )
                    # 设置返回值为1，表示发现了问题
                    ret = 1
    # 返回总的问题数量
    return ret
# 如果这个脚本是作为主程序被执行（而不是作为模块被导入），则执行以下代码块
if __name__ == "__main__":
    # 初始化返回值为0
    ret = 0
    # 使用Path对象找到所有以"scipy"开头的文件，且文件路径中包含"tests"目录及其子目录，文件名以"test"开头且以".py"结尾的文件
    path = Path("scipy").rglob("**/tests/**/test*.py")

    # 遍历符合条件的每一个文件
    for file in path:
        # 获取文件的基本名称（不包含路径部分）
        filename = os.path.basename(file)
        # 使用utf-8编码打开文件，并将文件内容读取为字符串
        with open(file, encoding="utf-8") as fd:
            content = fd.read()
        # 调用名为main的函数，传入文件内容和文件对象，将返回值按位与到ret中
        ret |= main(content, file)

    # 以ret的值作为退出状态码退出当前程序
    sys.exit(ret)
```