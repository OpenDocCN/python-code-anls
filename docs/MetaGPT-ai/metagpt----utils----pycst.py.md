# `MetaGPT\metagpt\utils\pycst.py`

```py

# 导入必要的模块和类型
from __future__ import annotations
from typing import Union
import libcst as cst
from libcst._nodes.module import Module

# 定义一个类型别名，表示可能包含文档字符串的节点
DocstringNode = Union[cst.Module, cst.ClassDef, cst.FunctionDef]

# 从节点的主体中提取文档字符串
def get_docstring_statement(body: DocstringNode) -> cst.SimpleStatementLine:
    # 如果节点是模块，则获取其主体
    if isinstance(body, cst.Module):
        body = body.body
    else:
        body = body.body.body

    # 如果主体为空，则返回 None
    if not body:
        return

    # 获取主体的第一个语句
    statement = body[0]
    if not isinstance(statement, cst.SimpleStatementLine):
        return

    # 逐级检查语句的结构，找到文档字符串
    expr = statement
    while isinstance(expr, (cst.BaseSuite, cst.SimpleStatementLine)):
        if len(expr.body) == 0:
            return None
        expr = expr.body[0]

    # 检查表达式的值是否为简单字符串或连接字符串
    if not isinstance(expr, cst.Expr):
        return None
    val = expr.value
    if not isinstance(val, (cst.SimpleString, cst.ConcatenatedString)):
        return None

    # 检查评估后的值是否为字节类型
    evaluated_value = val.evaluated_value
    if isinstance(evaluated_value, bytes):
        return None

    return statement

# 检查节点是否具有特定装饰器
def has_decorator(node: DocstringNode, name: str) -> bool:
    return hasattr(node, "decorators") and any(
        (hasattr(i.decorator, "value") and i.decorator.value == name)
        or (hasattr(i.decorator, "func") and hasattr(i.decorator.func, "value") and i.decorator.func.value == name)
        for i in node.decorators
    )

# 用于收集 CST 中文档字符串的访问者类
class DocstringCollector(cst.CSTVisitor):
    # 初始化方法
    def __init__(self):
        self.stack: list[str] = []
        self.docstrings: dict[tuple[str, ...], cst.SimpleStatementLine] = {}

    # 访问模块节点
    def visit_Module(self, node: cst.Module) -> bool | None:
        self.stack.append("")

    # 离开模块节点
    def leave_Module(self, node: cst.Module) -> None:
        return self._leave(node)

    # 访问类定义节点
    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self.stack.append(node.name.value)

    # 离开类定义节点
    def leave_ClassDef(self, node: cst.ClassDef) -> None:
        return self._leave(node)

    # 访问函数定义节点
    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self.stack.append(node.name.value)

    # 离开函数定义节点
    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        return self._leave(node)

    # 离开节点的通用方法
    def _leave(self, node: DocstringNode) -> None:
        key = tuple(self.stack)
        self.stack.pop()
        if has_decorator(node, "overload"):
            return
        statement = get_docstring_statement(node)
        if statement:
            self.docstrings[key] = statement

# 用于替换 CST 中文档字符串的转换器类
class DocstringTransformer(cst.CSTTransformer):
    # 初始化方法
    def __init__(
        self,
        docstrings: dict[tuple[str, ...], cst.SimpleStatementLine],
    ):
        self.stack: list[str] = []
        self.docstrings = docstrings

    # 访问模块节点
    def visit_Module(self, node: cst.Module) -> bool | None:
        self.stack.append("")

    # 离开模块节点
    def leave_Module(self, original_node: Module, updated_node: Module) -> Module:
        return self._leave(original_node, updated_node)

    # 访问类定义节点
    def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
        self.stack.append(node.name.value)

    # 离开类定义节点
    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
        return self._leave(original_node, updated_node)

    # 访问函数定义节点
    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self.stack.append(node.name.value)

    # 离开函数定义节点
    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:
        return self._leave(original_node, updated_node)

    # 离开节点的通用方法
    def _leave(self, original_node: DocstringNode, updated_node: DocstringNode) -> DocstringNode:
        key = tuple(self.stack)
        self.stack.pop()
        if has_decorator(updated_node, "overload"):
            return updated_node
        statement = self.docstrings.get(key)
        if not statement:
            return updated_node
        original_statement = get_docstring_statement(original_node)
        if isinstance(updated_node, cst.Module):
            body = updated_node.body
            if original_statement:
                return updated_node.with_changes(body=(statement, *body[1:]))
            else:
                updated_node = updated_node.with_changes(body=(statement, cst.EmptyLine(), *body))
                return updated_node
        body = updated_node.body.body[1:] if original_statement else updated_node.body.body
        return updated_node.with_changes(body=updated_node.body.with_changes(body=(statement, *body)))

# 合并文档字符串的函数
def merge_docstring(code: str, documented_code: str) -> str:
    # 解析原始代码和文档代码的 CST
    code_tree = cst.parse_module(code)
    documented_code_tree = cst.parse_module(documented_code)

    # 收集文档字符串
    visitor = DocstringCollector()
    documented_code_tree.visit(visitor)

    # 替换原始代码中的文档字符串
    transformer = DocstringTransformer(visitor.docstrings)
    modified_tree = code_tree.visit(transformer)

    return modified_tree.code

```