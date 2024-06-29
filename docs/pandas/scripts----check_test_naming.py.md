# `D:\src\scipysrc\pandas\scripts\check_test_naming.py`

```
"""
Check that test names start with `test`, and that test classes start with `Test`.

This is meant to be run as a pre-commit hook - to run it manually, you can do:

    pre-commit run check-test-naming --all-files

NOTE: if this finds a false positive, you can add the comment `# not a test` to the
class or function definition. Though hopefully that shouldn't be necessary.
"""
# 导入未来的注释语法支持
from __future__ import annotations

# 导入需要的模块
import argparse
import ast
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

# 如果是类型检查模式，则导入特定的集合抽象类
if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
        Sequence,
    )

# 用于标识不是测试的类或函数的 pragma
PRAGMA = "# not a test"


# 从抽象语法树中查找名称的生成器函数
def _find_names(node: ast.Module) -> Iterator[str]:
    for _node in ast.walk(node):
        if isinstance(_node, ast.Name):
            yield _node.id
        elif isinstance(_node, ast.Attribute):
            yield _node.attr


# 检查节点是否为 fixture 装饰器
def _is_fixture(node: ast.expr) -> bool:
    if isinstance(node, ast.Call):
        node = node.func
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "fixture"
        and isinstance(node.value, ast.Name)
        and node.value.id == "pytest"
    )


# 检查节点是否为 register_extension_dtype 函数调用
def _is_register_dtype(node):
    return isinstance(node, ast.Name) and node.id == "register_extension_dtype"


# 判断函数是否命名不规范
def is_misnamed_test_func(
    node: ast.expr | ast.stmt, names: Sequence[str], line: str
) -> bool:
    return (
        isinstance(node, ast.FunctionDef)  # 是函数定义节点
        and not node.name.startswith("test")  # 函数名不以 'test' 开头
        and names.count(node.name) == 0  # 函数名不在名称列表中
        and not any(_is_fixture(decorator) for decorator in node.decorator_list)  # 不是 fixture 装饰器
        and PRAGMA not in line  # 不包含不是测试的 pragma 注释
        and node.name
        not in ("teardown_method", "setup_method", "teardown_class", "setup_class")  # 不是特定的测试方法
    )


# 判断类是否命名不规范
def is_misnamed_test_class(
    node: ast.expr | ast.stmt, names: Sequence[str], line: str
) -> bool:
    return (
        isinstance(node, ast.ClassDef)  # 是类定义节点
        and not node.name.startswith("Test")  # 类名不以 'Test' 开头
        and names.count(node.name) == 0  # 类名不在名称列表中
        and not any(_is_register_dtype(decorator) for decorator in node.decorator_list)  # 不是 register_extension_dtype 装饰器
        and PRAGMA not in line  # 不包含不是测试的 pragma 注释
    )


# 主函数，处理文件内容和文件名
def main(content: str, file: str) -> int:
    lines = content.splitlines()  # 将内容拆分为行
    tree = ast.parse(content)  # 解析内容为抽象语法树
    names = list(_find_names(tree))  # 查找抽象语法树中的所有名称
    ret = 0  # 返回值初始化为 0
    # 遍历抽象语法树中的每个节点
    for node in tree.body:
        # 检查节点是否是被错误命名的测试函数，并根据情况输出错误信息
        if is_misnamed_test_func(node, names, lines[node.lineno - 1]):
            print(
                f"{file}:{node.lineno}:{node.col_offset} "
                "found test function which does not start with 'test'"
            )
            ret = 1
        # 检查节点是否是被错误命名的测试类，并根据情况输出错误信息
        elif is_misnamed_test_class(node, names, lines[node.lineno - 1]):
            print(
                f"{file}:{node.lineno}:{node.col_offset} "
                "found test class which does not start with 'Test'"
            )
            ret = 1
        # 如果节点是类定义节点，并且满足以下条件：
        # - 类名不在已知类名列表中
        # - 类没有注册为特定数据类型的装饰器
        # - 在节点所在行中没有出现特定的PRAGMA
        elif (
            isinstance(node, ast.ClassDef)
            and names.count(node.name) == 0
            and not any(
                _is_register_dtype(decorator) for decorator in node.decorator_list
            )
            and PRAGMA not in lines[node.lineno - 1]
        ):
            # 遍历类定义节点的主体
            for _node in node.body:
                # 检查节点是否是被错误命名的测试函数，并根据情况输出错误信息
                if is_misnamed_test_func(_node, names, lines[_node.lineno - 1]):
                    # 检查该函数是否被其父类的其他方法使用
                    assert isinstance(_node, ast.FunctionDef)  # help mypy
                    should_continue = False
                    # 遍历pandas/tests目录及其子目录下的所有.py文件
                    for _file in (Path("pandas") / "tests").rglob("*.py"):
                        with open(os.path.join(_file), encoding="utf-8") as fd:
                            _content = fd.read()
                        # 如果在文件内容中发现了self.<function_name>的调用，则跳过此函数的检查
                        if f"self.{_node.name}" in _content:
                            should_continue = True
                            break
                    if should_continue:
                        continue

                    # 输出错误信息，指出发现了不以'test'开头的测试函数
                    print(
                        f"{file}:{_node.lineno}:{_node.col_offset} "
                        "found test function which does not start with 'test'"
                    )
                    ret = 1
    # 返回检查结果
    return ret
if __name__ == "__main__":
    # 如果作为主程序执行，则开始程序的入口

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    parser.add_argument("paths", nargs="*")
    # 添加一个位置参数 "paths"，可以接受零个或多个文件路径

    args = parser.parse_args()
    # 解析命令行参数，并返回一个命名空间对象 args，包含传递给脚本的参数值

    ret = 0
    # 初始化一个变量 ret，用于存储最终的返回状态，初始为 0

    for file in args.paths:
        # 遍历参数中的每个文件路径

        filename = os.path.basename(file)
        # 获取文件的基本名称（不包含路径部分）

        if not (filename.startswith("test") and filename.endswith(".py")):
            # 如果文件名不以 "test" 开头或不以 ".py" 结尾，则跳过当前循环，继续下一个文件
            continue
        
        with open(file, encoding="utf-8") as fd:
            # 打开文件，指定编码为 UTF-8，使用 with 语句确保文件操作后自动关闭文件
            content = fd.read()
            # 读取文件内容到变量 content

        ret |= main(content, file)
        # 调用 main 函数处理文件内容，并将返回值与 ret 进行按位或运算，更新 ret 的状态

    sys.exit(ret)
    # 以 ret 的值作为状态码退出程序，sys.exit() 会终止程序并返回状态码
```