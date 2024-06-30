# `D:\src\scipysrc\sympy\sympy\unify\__init__.py`

```
""" Unification in SymPy

See sympy.unify.core docstring for algorithmic details

See http://matthewrocklin.com/blog/work/2012/11/01/Unification/ for discussion
"""

# 导入 usympy 模块中的 unify 和 rebuild 函数
from .usympy import unify, rebuild
# 导入 rewrite 模块中的 rewriterule 函数
from .rewrite import rewriterule

# 定义公开的接口列表，包括 unify 和 rebuild 函数
__all__ = [
    'unify', 'rebuild',

    # 添加 rewriterule 函数到公开接口列表
    'rewriterule',
]
```