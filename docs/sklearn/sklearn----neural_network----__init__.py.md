# `D:\src\scipysrc\scikit-learn\sklearn\neural_network\__init__.py`

```
"""Models based on neural networks."""

# SPDX-License-Identifier: BSD-3-Clause
# 引入多层感知机分类器和回归器以及伯努利限制玻尔兹曼机模块
from ._multilayer_perceptron import MLPClassifier, MLPRegressor
# 引入伯努利限制玻尔兹曼机模块
from ._rbm import BernoulliRBM

# 将模块内需要公开的类列在 __all__ 列表中
__all__ = ["BernoulliRBM", "MLPClassifier", "MLPRegressor"]
```