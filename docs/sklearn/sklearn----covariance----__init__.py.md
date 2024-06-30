# `D:\src\scipysrc\scikit-learn\sklearn\covariance\__init__.py`

```
"""
Methods and algorithms to robustly estimate covariance.

They estimate the covariance of features at given sets of points, as well as the
precision matrix defined as the inverse of the covariance. Covariance estimation is
closely related to the theory of Gaussian graphical models.
"""

# 从自定义模块导入异常值检测相关的EllipticEnvelope类
from ._elliptic_envelope import EllipticEnvelope
# 从自定义模块导入经验协方差估计相关的类和函数
from ._empirical_covariance import (
    EmpiricalCovariance,
    empirical_covariance,
    log_likelihood,
)
# 从自定义模块导入图形Lasso相关的类和函数
from ._graph_lasso import GraphicalLasso, GraphicalLassoCV, graphical_lasso
# 从自定义模块导入鲁棒协方差估计相关的类和函数
from ._robust_covariance import MinCovDet, fast_mcd
# 从自定义模块导入收缩协方差估计相关的类和函数
from ._shrunk_covariance import (
    OAS,
    LedoitWolf,
    ShrunkCovariance,
    ledoit_wolf,
    ledoit_wolf_shrinkage,
    oas,
    shrunk_covariance,
)

# 导出的全部变量名列表，用于模块级别的导入
__all__ = [
    "EllipticEnvelope",
    "EmpiricalCovariance",
    "GraphicalLasso",
    "GraphicalLassoCV",
    "LedoitWolf",
    "MinCovDet",
    "OAS",
    "ShrunkCovariance",
    "empirical_covariance",
    "fast_mcd",
    "graphical_lasso",
    "ledoit_wolf",
    "ledoit_wolf_shrinkage",
    "log_likelihood",
    "oas",
    "shrunk_covariance",
]
```