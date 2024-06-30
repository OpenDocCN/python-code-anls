# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion_constr\__init__.py`

```
"""This module contains the equality constrained SQP solver."""

# 导入_minimize_trustregion_constr模块中的_minimize_trustregion_constr函数
from .minimize_trustregion_constr import _minimize_trustregion_constr

# 定义该模块中可以被外部访问的接口，仅包括_minimize_trustregion_constr函数
__all__ = ['_minimize_trustregion_constr']
```