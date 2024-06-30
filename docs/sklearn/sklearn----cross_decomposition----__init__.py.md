# `D:\src\scipysrc\scikit-learn\sklearn\cross_decomposition\__init__.py`

```
`
"""
Algorithms for cross decomposition.
交叉分解算法模块说明。
"""

# 从._pls模块中导入CCA, PLSSVD, PLSCanonical, PLSRegression这几个类或函数
from ._pls import CCA, PLSSVD, PLSCanonical, PLSRegression

# 定义导出的模块成员列表，只包括以下四个成员
__all__ = ["PLSCanonical", "PLSRegression", "PLSSVD", "CCA"]
```