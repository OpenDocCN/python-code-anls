# `numpy-ml\numpy_ml\linear_models\__init__.py`

```
# 一个包含各种线性模型的模块

# 导入 Ridge 回归模型
from .ridge import RidgeRegression
# 导入广义线性模型
from .glm import GeneralizedLinearModel
# 导入逻辑回归模型
from .logistic import LogisticRegression
# 导入贝叶斯回归模型（已知方差）
from .bayesian_regression import (
    BayesianLinearRegressionKnownVariance,
    BayesianLinearRegressionUnknownVariance,
)
# 导入高斯朴素贝叶斯分类器
from .naive_bayes import GaussianNBClassifier
# 导入线性回归模型
from .linear_regression import LinearRegression
```