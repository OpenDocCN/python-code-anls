# `.\pytorch\torch\fx\experimental\unification\__init__.py`

```py
# 禁止 mypy 报告错误码 attr-defined 的错误
# 从当前包的 core 模块导入 unify 和 reify 函数，禁止 Flake8 报错 F403
from .core import unify, reify  # noqa: F403
# 从当前包的 more 模块导入 unifiable 函数，禁止 Flake8 报错 F403
from .more import unifiable  # noqa: F403
# 从当前包的 variable 模块导入 var, isvar, vars, variables, Var 类，禁止 Flake8 报错 F403
from .variable import var, isvar, vars, variables, Var  # noqa: F403
```