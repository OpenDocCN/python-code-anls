# `D:\src\scipysrc\scikit-learn\sklearn\model_selection\__init__.py`

```
# 导入必要的类型模块，用于类型提示
import typing

# 从内部模块导入分类器阈值调整相关的类
from ._classification_threshold import (
    FixedThresholdClassifier,
    TunedThresholdClassifierCV,
)

# 导入绘图相关的类
from ._plot import LearningCurveDisplay, ValidationCurveDisplay

# 导入搜索相关的类和函数
from ._search import GridSearchCV, ParameterGrid, ParameterSampler, RandomizedSearchCV

# 导入数据集划分相关的类
from ._split import (
    BaseCrossValidator,
    BaseShuffleSplit,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,
    PredefinedSplit,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    check_cv,
    train_test_split,
)

# 导入验证相关的函数
from ._validation import (
    cross_val_predict,
    cross_val_score,
    cross_validate,
    learning_curve,
    permutation_test_score,
    validation_curve,
)

# 如果在类型检查器（如mypy）中，避免对实验性评估器（例如HalvingGridSearchCV和HalvingRandomSearchCV）的错误
# TODO: 一旦评估器不再是实验性的，应移除此检查。
if typing.TYPE_CHECKING:
    from ._search_successive_halving import (  # noqa
        HalvingGridSearchCV,
        HalvingRandomSearchCV,
    )

# __all__列表定义了在from module import *时可导入的符号列表
__all__ = [
    "BaseCrossValidator",
    "BaseShuffleSplit",
    "GridSearchCV",
    "TimeSeriesSplit",
    "KFold",
    "GroupKFold",
    "GroupShuffleSplit",
    "LeaveOneGroupOut",
    "LeaveOneOut",
    "LeavePGroupsOut",
    "LeavePOut",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "ParameterGrid",
    "ParameterSampler",
    "PredefinedSplit",
    "RandomizedSearchCV",
    "ShuffleSplit",
    "StratifiedKFold",
    "StratifiedGroupKFold",
    "StratifiedShuffleSplit",
    "FixedThresholdClassifier",
    "TunedThresholdClassifierCV",
    "check_cv",
    "cross_val_predict",
    "cross_val_score",
    "cross_validate",
    "learning_curve",
    "LearningCurveDisplay",
    "permutation_test_score",
    "train_test_split",
    "validation_curve",
    "ValidationCurveDisplay",
]

# 如果试图访问不在__all__列表中的属性时，引发AttributeError异常
def __getattr__(name):
    # 如果尝试访问实验性评估器，则引发ImportError异常
    if name in {"HalvingGridSearchCV", "HalvingRandomSearchCV"}:
        raise ImportError(
            f"{name} is experimental and the API might change without any "
            "deprecation cycle. To use it, you need to explicitly import "
            "enable_halving_search_cv:\n"
            "from sklearn.experimental import enable_halving_search_cv"
        )
    raise AttributeError(f"module {__name__} has no attribute {name}")
```