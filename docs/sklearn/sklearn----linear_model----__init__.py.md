# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\__init__.py`

```
# 导入线性模型相关的各种类和函数

# 详细文档请参见 http://scikit-learn.sourceforge.net/modules/sgd.html 和
# http://scikit-learn.sourceforge.net/modules/linear_model.html

# 从 _base 模块导入 LinearRegression 类
from ._base import LinearRegression

# 从 _bayes 模块导入 ARDRegression 和 BayesianRidge 类
from ._bayes import ARDRegression, BayesianRidge

# 从 _coordinate_descent 模块导入一系列类和函数
from ._coordinate_descent import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    MultiTaskElasticNet,
    MultiTaskElasticNetCV,
    MultiTaskLasso,
    MultiTaskLassoCV,
    enet_path,
    lasso_path,
)

# 从 _glm 模块导入 GammaRegressor、PoissonRegressor 和 TweedieRegressor 类
from ._glm import GammaRegressor, PoissonRegressor, TweedieRegressor

# 从 _huber 模块导入 HuberRegressor 类
from ._huber import HuberRegressor

# 从 _least_angle 模块导入一系列类和函数
from ._least_angle import (
    Lars,
    LarsCV,
    LassoLars,
    LassoLarsCV,
    LassoLarsIC,
    lars_path,
    lars_path_gram,
)

# 从 _logistic 模块导入 LogisticRegression 和 LogisticRegressionCV 类
from ._logistic import LogisticRegression, LogisticRegressionCV

# 从 _omp 模块导入一系列类和函数
from ._omp import (
    OrthogonalMatchingPursuit,
    OrthogonalMatchingPursuitCV,
    orthogonal_mp,
    orthogonal_mp_gram,
)

# 从 _passive_aggressive 模块导入 PassiveAggressiveClassifier 和 PassiveAggressiveRegressor 类
from ._passive_aggressive import PassiveAggressiveClassifier, PassiveAggressiveRegressor

# 从 _perceptron 模块导入 Perceptron 类
from ._perceptron import Perceptron

# 从 _quantile 模块导入 QuantileRegressor 类
from ._quantile import QuantileRegressor

# 从 _ransac 模块导入 RANSACRegressor 类
from ._ransac import RANSACRegressor

# 从 _ridge 模块导入一系列类和函数
from ._ridge import Ridge, RidgeClassifier, RidgeClassifierCV, RidgeCV, ridge_regression

# 从 _sgd_fast 模块导入 Hinge、Huber、Log、ModifiedHuber 和 SquaredLoss 类
from ._sgd_fast import Hinge, Huber, Log, ModifiedHuber, SquaredLoss

# 从 _stochastic_gradient 模块导入 SGDClassifier、SGDOneClassSVM 和 SGDRegressor 类
from ._stochastic_gradient import SGDClassifier, SGDOneClassSVM, SGDRegressor

# 从 _theil_sen 模块导入 TheilSenRegressor 类
from ._theil_sen import TheilSenRegressor

# __all__ 列表包含了该模块中所有公开的类和函数的名称，用于控制导入时的可见性
__all__ = [
    "ARDRegression",
    "BayesianRidge",
    "ElasticNet",
    "ElasticNetCV",
    "Hinge",
    "Huber",
    "HuberRegressor",
    "Lars",
    "LarsCV",
    "Lasso",
    "LassoCV",
    "LassoLars",
    "LassoLarsCV",
    "LassoLarsIC",
    "LinearRegression",
    "Log",
    "LogisticRegression",
    "LogisticRegressionCV",
    "ModifiedHuber",
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "PassiveAggressiveClassifier",
    "PassiveAggressiveRegressor",
    "Perceptron",
    "QuantileRegressor",
    "Ridge",
    "RidgeCV",
    "RidgeClassifier",
    "RidgeClassifierCV",
    "SGDClassifier",
    "SGDRegressor",
    "SGDOneClassSVM",
    "SquaredLoss",
    "TheilSenRegressor",
    "enet_path",
    "lars_path",
    "lars_path_gram",
    "lasso_path",
    "orthogonal_mp",
    "orthogonal_mp_gram",
    "ridge_regression",
    "RANSACRegressor",
    "PoissonRegressor",
    "GammaRegressor",
    "TweedieRegressor",
]
```