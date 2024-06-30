# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_glm\__init__.py`

```
# 导入所需模块和类，从当前包（当前目录）下的 glm 模块中分别导入以下类：
# GammaRegressor：用于伽马分布的回归器
# PoissonRegressor：用于泊松分布的回归器
# TweedieRegressor：用于 Tweedie 分布的回归器
# _GeneralizedLinearRegressor：广义线性回归器的基类或私有实现
from .glm import (
    GammaRegressor,
    PoissonRegressor,
    TweedieRegressor,
    _GeneralizedLinearRegressor,
)

# 定义 __all__ 列表，用于指定当前模块中公开的符号（变量、类、函数等）
__all__ = [
    "_GeneralizedLinearRegressor",  # 将 _GeneralizedLinearRegressor 加入 __all__ 中
    "PoissonRegressor",             # 将 PoissonRegressor 加入 __all__ 中
    "GammaRegressor",               # 将 GammaRegressor 加入 __all__ 中
    "TweedieRegressor",             # 将 TweedieRegressor 加入 __all__ 中
]
```