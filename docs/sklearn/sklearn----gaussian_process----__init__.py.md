# `D:\src\scipysrc\scikit-learn\sklearn\gaussian_process\__init__.py`

```
"""Gaussian process based regression and classification."""

# 从当前包中导入 kernels 模块，用于高斯过程的核函数
from . import kernels

# 从当前包中导入 GaussianProcessClassifier 类，用于高斯过程分类器
from ._gpc import GaussianProcessClassifier

# 从当前包中导入 GaussianProcessRegressor 类，用于高斯过程回归器
from ._gpr import GaussianProcessRegressor

# 定义了当前模块对外公开的接口，包括 GaussianProcessRegressor、GaussianProcessClassifier 和 kernels
__all__ = ["GaussianProcessRegressor", "GaussianProcessClassifier", "kernels"]
```