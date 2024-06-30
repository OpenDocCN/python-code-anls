# `D:\src\scipysrc\scikit-learn\sklearn\_loss\__init__.py`

```
"""
The :mod:`sklearn._loss` module includes loss function classes suitable for
fitting classification and regression tasks.
"""

# 从当前目录中导入特定的损失函数类
from .loss import (
    AbsoluteError,               # 绝对误差损失函数
    HalfBinomialLoss,            # 半二项损失函数
    HalfGammaLoss,               # 半伽马损失函数
    HalfMultinomialLoss,         # 半多项式损失函数
    HalfPoissonLoss,             # 半泊松损失函数
    HalfSquaredError,            # 半平方误差损失函数
    HalfTweedieLoss,             # 半 Tweedie 损失函数
    HalfTweedieLossIdentity,     # 半 Tweedie 损失函数（恒等链接）
    HuberLoss,                   # Huber 损失函数
    PinballLoss,                 # 钉子损失函数
)

# 导出的符号列表，包含可以从模块中直接访问的损失函数类名称
__all__ = [
    "HalfSquaredError",          # 半平方误差损失函数
    "AbsoluteError",             # 绝对误差损失函数
    "PinballLoss",               # 钉子损失函数
    "HuberLoss",                 # Huber 损失函数
    "HalfPoissonLoss",           # 半泊松损失函数
    "HalfGammaLoss",             # 半伽马损失函数
    "HalfTweedieLoss",           # 半 Tweedie 损失函数
    "HalfTweedieLossIdentity",   # 半 Tweedie 损失函数（恒等链接）
    "HalfBinomialLoss",          # 半二项损失函数
    "HalfMultinomialLoss",       # 半多项式损失函数
]
```