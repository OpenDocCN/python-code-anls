# `D:\src\scipysrc\sympy\sympy\strategies\__init__.py`

```
""" Rewrite Rules

DISCLAIMER: This module is experimental. The interface is subject to change.

A rule is a function that transforms one expression into another

    Rule :: Expr -> Expr

A strategy is a function that says how a rule should be applied to a syntax
tree. In general strategies take rules and produce a new rule

    Strategy :: [Rules], Other-stuff -> Rule

This allows developers to separate a mathematical transformation from the
algorithmic details of applying that transformation. The goal is to separate
the work of mathematical programming from algorithmic programming.

Submodules

strategies.rl         - some fundamental rules
strategies.core       - generic non-SymPy specific strategies
strategies.traverse   - strategies that traverse a SymPy tree
strategies.tools      - some conglomerate strategies that do depend on SymPy
"""

# Importing necessary modules and functions from local packages
from . import rl                     # 导入本地的 rl 模块
from . import traverse               # 导入本地的 traverse 模块
from .rl import rm_id, unpack, flatten, sort, glom, distribute, rebuild  # 从 rl 模块导入指定函数
from .util import new                # 导入本地的 util 模块中的 new 函数
from .core import (                  # 从 core 模块导入多个函数
    condition, debug, chain, null_safe, do_one, exhaust, minimize, tryit)
from .tools import canon, typed      # 从 tools 模块导入 canon 和 typed 函数
from . import branch                 # 导入本地的 branch 模块

# 导出的模块和函数列表
__all__ = [
    'rl',                            # 导出 rl 模块
    'traverse',                      # 导出 traverse 模块
    'rm_id', 'unpack', 'flatten', 'sort', 'glom', 'distribute', 'rebuild',  # 导出 rl 模块中的函数
    'new',                           # 导出 new 函数
    'condition', 'debug', 'chain', 'null_safe', 'do_one', 'exhaust',         # 导出 core 模块中的函数
    'minimize', 'tryit',
    'canon', 'typed',                # 导出 tools 模块中的函数
    'branch',                        # 导出 branch 模块
]
```