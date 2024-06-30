# `D:\src\scipysrc\scikit-learn\sklearn\mixture\__init__.py`

```
"""Mixture modeling algorithms."""

# 导入贝叶斯高斯混合模型类
from ._bayesian_mixture import BayesianGaussianMixture
# 导入高斯混合模型类
from ._gaussian_mixture import GaussianMixture

# 定义公开接口，只暴露以下两个类
__all__ = ["GaussianMixture", "BayesianGaussianMixture"]
```