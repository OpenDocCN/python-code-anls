# `markdown\scripts\griffe_extensions.py`

```

# 导入必要的模块和类
from __future__ import annotations
import ast
from typing import TYPE_CHECKING
import textwrap
from griffe import Docstring, Extension
from griffe.docstrings.dataclasses import DocstringSectionAdmonition, DocstringSectionText

# 如果是类型检查，导入额外的模块
if TYPE_CHECKING:
    from griffe import Class, Function, ObjectNode

# 检查对象是否被标记为deprecated
def _deprecated(obj: Class | Function) -> str | None:
    for decorator in obj.decorators:
        if decorator.callable_path == "markdown.util.deprecated":
            return ast.literal_eval(str(decorator.value.arguments[0]))
    return None

# 创建一个用于处理deprecated的扩展
class DeprecatedExtension(Extension):
    """Griffe extension for `@markdown.util.deprecated` decorator support."""

    # 在类的实例上添加已弃用类的部分
    def on_class_instance(self, node: ast.AST | ObjectNode, cls: Class) -> None:  # noqa: ARG002
        """Add section to docstrings of deprecated classes."""
        if message := _deprecated(cls):
            self._insert_message(cls, message)
            cls.labels.add("deprecated")

    # 在函数的实例上添加已弃用函数的部分
    def on_function_instance(self, node: ast.AST | ObjectNode, func: Function) -> None:  # noqa: ARG002
        """Add section to docstrings of deprecated functions."""
        if message := _deprecated(func):
            self._insert_message(func, message)
            func.labels.add("deprecated")

# 创建一个用于处理优先级表的扩展
class PriorityTableExtension(Extension):
    """ Griffe extension to insert a table of processor priority in specified functions. """

    # 初始化函数，接受一个路径列表作为参数
    def __init__(self, paths: list[str] | None = None) -> None:
        super().__init__()
        self.paths = paths

    # 将对象名称包装在引用链接中
    def linked_obj(self, value: str, path: str) -> str:
        """ Wrap object name in reference link. """
        return f'[`{value}`][{path}.{value}]'

    # 在函数的实例上添加优先级表
    def on_function_instance(self, node: ast.AST | ObjectNode, func: Function) -> None:  # noqa: ARG002
        """Add table to specified function docstrings."""
        if self.paths and func.path not in self.paths:
            return  # skip objects that were not selected

        # 创建表头
        data = [
            'Class Instance | Name | Priority',
            '-------------- | ---- | :------:'
        ]

        # 从函数的源代码中提取表格内容
        for obj in node.body:
            # 提取传递给`util.Registry.register`的参数
            if isinstance(obj, ast.Expr) and isinstance(obj.value, ast.Call) and obj.value.func.attr == 'register':
                _args = obj.value.args
                cls = self.linked_obj(_args[0].func.id, func.path.rsplit('.', 1)[0])
                name = _args[1].value
                priority = str(_args[2].value)
                if func.name == ('build_inlinepatterns'):
                    # 包括Pattern：传递给类的第一个参数
                    if isinstance(_args[0].args[0], ast.Constant):
                        # Pattern是一个字符串
                        value = f'`"{_args[0].args[0].value}"`'
                    else:
                        # Pattern是一个变量
                        value = self.linked_obj(_args[0].args[0].id, func.path.rsplit('.', 1)[0])
                    cls = f'{cls}({value})'
                data.append(f'{cls} | `{name}` | `{priority}`')

        table = '\n'.join(data)
        body = (
            f"Return a [`{func.returns.canonical_name}`][{func.returns.canonical_path}] instance which contains "
            "the following collection of classes with their assigned names and priorities.\n\n"
            f"{table}"
        )

        # 添加到文档字符串中
        if not func.docstring:
            func.docstring = Docstring("", parent=func)
        sections = func.docstring.parsed
        sections.append(DocstringSectionText(body, title="Priority Table"))

```