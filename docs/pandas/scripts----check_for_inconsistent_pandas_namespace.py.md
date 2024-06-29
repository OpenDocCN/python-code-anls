# `D:\src\scipysrc\pandas\scripts\check_for_inconsistent_pandas_namespace.py`

```
"""
Check that test suite file doesn't use the pandas namespace inconsistently.

We check for cases of ``Series`` and ``pd.Series`` appearing in the same file
(likewise for other pandas objects).

This is meant to be run as a pre-commit hook - to run it manually, you can do:

    pre-commit run inconsistent-namespace-usage --all-files

To automatically fixup a given file, you can pass `--replace`, e.g.

    python scripts/check_for_inconsistent_pandas_namespace.py test_me.py --replace

though note that you may need to manually fixup some imports and that you will also
need the additional dependency `tokenize-rt` (which is left out from the pre-commit
hook so that it uses the same virtualenv as the other local ones).

The general structure is similar to that of some plugins from
https://github.com/asottile/pyupgrade .
"""

import argparse  # 导入用于解析命令行参数的模块
import ast  # 导入用于抽象语法树操作的模块
from collections.abc import (  # 导入抽象基类中的 MutableMapping 和 Sequence
    MutableMapping,
    Sequence,
)
import sys  # 导入系统相关的模块
from typing import NamedTuple  # 导入用于类型注解的 NamedTuple


ERROR_MESSAGE = (  # 定义错误信息的格式字符串
    "{path}:{lineno}:{col_offset}: "
    "Found both '{prefix}.{name}' and '{name}' in {path}"
)


class OffsetWithNamespace(NamedTuple):  # 定义命名元组 OffsetWithNamespace，用于存储行号、偏移量和命名空间
    lineno: int
    col_offset: int
    namespace: str


class Visitor(ast.NodeVisitor):  # 定义继承自 ast.NodeVisitor 的 Visitor 类
    def __init__(self) -> None:
        self.pandas_namespace: MutableMapping[OffsetWithNamespace, str] = {}  # 初始化 pandas 命名空间映射
        self.imported_from_pandas: set[str] = set()  # 初始化从 pandas 导入的名称集合

    def visit_Attribute(self, node: ast.Attribute) -> None:  # 处理 Attribute 节点的访问
        if isinstance(node.value, ast.Name) and node.value.id in {"pandas", "pd"}:
            # 如果节点值是 ast.Name 类型且其 id 是 "pandas" 或 "pd"
            offset_with_namespace = OffsetWithNamespace(
                node.lineno, node.col_offset, node.value.id
            )
            self.pandas_namespace[offset_with_namespace] = node.attr  # 将属性名称与命名空间存入映射
        self.generic_visit(node)  # 继续处理其他节点

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # 处理 ImportFrom 节点的访问
        if node.module is not None and "pandas" in node.module:
            # 如果模块名不为空且包含 "pandas"
            self.imported_from_pandas.update(name.name for name in node.names)
            # 更新从 pandas 导入的名称集合
        self.generic_visit(node)  # 继续处理其他节点


def replace_inconsistent_pandas_namespace(visitor: Visitor, content: str) -> str:
    from tokenize_rt import (  # 从 tokenize-rt 模块导入多个函数
        reversed_enumerate,
        src_to_tokens,
        tokens_to_src,
    )

    tokens = src_to_tokens(content)  # 将源代码转换为 token 列表
    for n, i in reversed_enumerate(tokens):  # 反向遍历 token 列表
        offset_with_namespace = OffsetWithNamespace(i.offset[0], i.offset[1], i.src)
        if (
            offset_with_namespace in visitor.pandas_namespace
            and visitor.pandas_namespace[offset_with_namespace]
            in visitor.imported_from_pandas
        ):
            # 如果找到不一致的 pandas 命名空间使用
            # 替换 `pd`
            tokens[n] = i._replace(src="")
            # 替换 `.`
            tokens[n + 1] = tokens[n + 1]._replace(src="")

    new_src: str = tokens_to_src(tokens)  # 将 token 列表转换回源代码
    return new_src  # 返回修复后的源代码


def check_for_inconsistent_pandas_namespace(
    content: str, path: str, *, replace: bool
) -> str | None:
    tree = ast.parse(content)  # 解析源代码为抽象语法树

    visitor = Visitor()  # 创建 Visitor 实例
    visitor.visit(tree)  # 遍历抽象语法树

    if replace:
        return replace_inconsistent_pandas_namespace(visitor, content)
    else:
        return None  # 如果不进行替换，返回 None
    # 计算从 pandas 导入的模块与 visitor.pandas_namespace 中值的交集
    inconsistencies = visitor.imported_from_pandas.intersection(
        visitor.pandas_namespace.values()
    )

    # 如果没有不一致的命名空间使用，表示无需替换，返回空值
    if not inconsistencies:
        # No inconsistent namespace usage, nothing to replace.
        return None

    # 如果需要替换不一致的命名空间
    if not replace:
        # 从不一致的命名空间集合中弹出一个不一致项
        inconsistency = inconsistencies.pop()
        # 查找第一个匹配不一致值的键，并获取其行号、列偏移和前缀
        lineno, col_offset, prefix = next(
            key for key, val in visitor.pandas_namespace.items() if val == inconsistency
        )
        # 格式化错误信息，包括行号、列偏移、前缀、不一致值和路径
        msg = ERROR_MESSAGE.format(
            lineno=lineno,
            col_offset=col_offset,
            prefix=prefix,
            name=inconsistency,
            path=path,
        )
        # 将错误信息输出到标准输出
        sys.stdout.write(msg)
        # 退出程序，状态码为1
        sys.exit(1)

    # 替换不一致的 pandas 命名空间并返回结果
    return replace_inconsistent_pandas_namespace(visitor, content)
# 定义程序的主函数，接受一个可选的命令行参数序列 argv
def main(argv: Sequence[str] | None = None) -> None:
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加一个接受任意数量位置参数的参数 paths
    parser.add_argument("paths", nargs="*")
    # 添加一个布尔类型的选项参数 --replace，如果出现则设置为 True
    parser.add_argument("--replace", action="store_true")
    # 解析命令行参数
    args = parser.parse_args(argv)

    # 遍历命令行参数中的每个路径
    for path in args.paths:
        # 使用 utf-8 编码打开文件路径对应的文件，并将其内容读取为字符串
        with open(path, encoding="utf-8") as fd:
            content = fd.read()
        
        # 调用函数检查 content 中是否存在不一致的 pandas 命名空间，并可能进行替换
        new_content = check_for_inconsistent_pandas_namespace(
            content, path, replace=args.replace
        )
        
        # 如果不需要替换或者 new_content 为 None，则继续下一个路径的处理
        if not args.replace or new_content is None:
            continue
        
        # 使用 utf-8 编码打开路径对应的文件，并将 new_content 写入文件中
        with open(path, "w", encoding="utf-8") as fd:
            fd.write(new_content)

# 如果当前脚本作为主程序运行，则调用 main 函数
if __name__ == "__main__":
    main()
```