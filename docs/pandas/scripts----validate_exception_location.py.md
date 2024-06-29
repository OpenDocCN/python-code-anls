# `D:\src\scipysrc\pandas\scripts\validate_exception_location.py`

```
"""
Validate that the exceptions and warnings are in appropriate places.

Checks for classes that inherit a python exception and warning and
flags them, unless they are exempted from checking. Exempt meaning
the exception/warning is defined in testing.rst. Testing.rst contains
a list of pandas defined exceptions and warnings. This list is kept
current by other pre-commit hook, pandas_errors_documented.py.
This hook maintains that errors.__init__.py and testing.rst are in-sync.
Therefore, the exception or warning should be defined or imported in
errors.__init__.py. Ideally, the exception or warning is defined unless
there's special reason to import it.

Prints the exception/warning that do not follow this convention.

Usage::

As a pre-commit hook:
    pre-commit run validate-errors-locations --all-files
"""
# 导入必要的模块和库
from __future__ import annotations

import argparse  # 导入命令行参数解析模块
import ast  # 导入抽象语法树模块，用于解析Python代码
import pathlib  # 导入处理文件路径的模块
import sys  # 导入系统相关的功能模块
from typing import TYPE_CHECKING  # 导入类型提示相关的模块

if TYPE_CHECKING:
    from collections.abc import Sequence  # 如果是类型检查阶段，导入Sequence类型

# 定义API路径
API_PATH = pathlib.Path("doc/source/reference/testing.rst").resolve()
# 定义错误信息模板
ERROR_MESSAGE = (
    "The following exception(s) and/or warning(s): {errors} exist(s) outside of "
    "pandas/errors/__init__.py. Please either define them in "
    "pandas/errors/__init__.py. Or, if not possible then import them in "
    "pandas/errors/__init__.py.\n"
)


def get_warnings_and_exceptions_from_api_path() -> set[str]:
    """
    从API路径中读取并返回文档中定义的异常和警告集合。

    Returns:
        set[str]: 包含在API路径中文档定义的异常和警告的集合
    """
    with open(API_PATH, encoding="utf-8") as f:
        # 从文档中提取所有包含"errors"的行，获取异常和警告的名称并返回集合
        doc_errors = {
            line.split(".")[1].strip() for line in f.readlines() if "errors" in line
        }
        return doc_errors


class Visitor(ast.NodeVisitor):
    """
    AST节点访问器，用于访问Python代码中的类定义节点，并检查是否继承异常或警告类。

    Attributes:
        path (str): 文件路径
        exception_set (set[str]): 预定义的异常和警告集合
        found_exceptions (set): 发现的异常类集合
    """

    def __init__(self, path: str, exception_set: set[str]) -> None:
        """
        初始化方法。

        Args:
            path (str): 文件路径
            exception_set (set[str]): 预定义的异常和警告集合
        """
        self.path = path
        self.exception_set = exception_set
        self.found_exceptions = set()

    def visit_ClassDef(self, node) -> None:
        """
        访问类定义节点，并检查其是否继承异常或警告类。

        Args:
            node (ast.ClassDef): 类定义节点
        """
        def is_an_exception_subclass(base_id: str) -> bool:
            """
            检查基类标识是否为异常或警告类。

            Args:
                base_id (str): 基类标识字符串

            Returns:
                bool: 如果是异常或警告类返回True，否则返回False
            """
            return base_id == "Exception" or base_id.endswith(("Warning", "Error"))

        exception_classes = []

        # 遍历类的基类，检查是否为异常或警告类
        for base in node.bases:
            base_id = getattr(base, "id", None)
            if base_id and is_an_exception_subclass(base_id):
                exception_classes.append(base_id)

        # 如果类继承了异常或警告类，则将类名添加到发现的异常类集合中
        if exception_classes:
            self.found_exceptions.add(node.name)


def validate_exception_and_warning_placement(
    file_path: str, file_content: str, errors: set[str]
) -> None:
    """
    验证异常和警告的放置位置是否正确，并输出不符合约定的异常和警告信息。

    Args:
        file_path (str): 文件路径
        file_content (str): 文件内容
        errors (set[str]): 预定义的异常和警告集合
    """
    # 解析文件内容为抽象语法树
    tree = ast.parse(file_content)
    # 创建AST节点访问器对象
    visitor = Visitor(file_path, errors)
    # 访问抽象语法树
    visitor.visit(tree)

    # 计算不在预定义异常和警告集合中的异常类
    misplaced_exceptions = visitor.found_exceptions.difference(errors)

    # 如果存在不在预定义集合中的异常类，则打印错误信息
    # 标志存在于pandas/errors/__init__.py之外的pandas定义的异常或警告
    # 如果存在被放错位置的异常列表，则执行以下代码块
    if misplaced_exceptions:
        # 格式化错误消息，将异常列表转换为逗号分隔的字符串，并插入到错误消息模板中
        msg = ERROR_MESSAGE.format(errors=", ".join(misplaced_exceptions))
        # 将错误消息写入标准输出
        sys.stdout.write(msg)
        # 终止程序运行，退出状态码为1，表示发生了错误
        sys.exit(1)
# 定义主函数，用于处理命令行参数并执行主要逻辑
def main(argv: Sequence[str] | None = None) -> None:
    # 创建命令行参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加一个位置参数 "paths"，允许接受多个路径作为输入
    parser.add_argument("paths", nargs="*")
    # 解析命令行参数
    args = parser.parse_args(argv)

    # 从 API 路径获取警告和异常信息的集合
    error_set = get_warnings_and_exceptions_from_api_path()

    # 遍历每个传入的路径
    for path in args.paths:
        # 使用 UTF-8 编码打开文件，并使用上下文管理器确保文件操作安全
        with open(path, encoding="utf-8") as fd:
            # 读取文件内容到变量 content
            content = fd.read()
        # 验证文件中异常和警告的位置是否合法，传入文件路径、内容和异常集合作为参数
        validate_exception_and_warning_placement(path, content, error_set)


if __name__ == "__main__":
    # 如果脚本作为主程序运行，则调用主函数 main()
    main()
```