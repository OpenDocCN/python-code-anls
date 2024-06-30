# `D:\src\scipysrc\sympy\sympy\codegen\__init__.py`

```
"""
The ``sympy.codegen`` module contains classes and functions for building
abstract syntax trees of algorithms. These trees may then be printed by the
code-printers in ``sympy.printing``.

There are several submodules available:
- ``sympy.codegen.ast``: AST nodes useful across multiple languages.
- ``sympy.codegen.cnodes``: AST nodes useful for the C family of languages.
- ``sympy.codegen.fnodes``: AST nodes useful for Fortran.
- ``sympy.codegen.cfunctions``: functions specific to C (C99 math functions)
- ``sympy.codegen.ffunctions``: functions specific to Fortran (e.g. ``kind``).
"""

# 从当前目录导入以下符号
from .ast import (
    Assignment, aug_assign, CodeBlock, For, Attribute, Variable, Declaration,
    While, Scope, Print, FunctionPrototype, FunctionDefinition, FunctionCall
)

# 将以下符号加入模块的公共接口列表
__all__ = [
    'Assignment', 'aug_assign', 'CodeBlock', 'For', 'Attribute', 'Variable',
    'Declaration', 'While', 'Scope', 'Print', 'FunctionPrototype',
    'FunctionDefinition', 'FunctionCall',
]
```