# `D:\src\scipysrc\scikit-learn\sklearn\inspection\__init__.py`

```
"""Tools for model inspection."""

# 导入局部依赖分析函数
from ._partial_dependence import partial_dependence
# 导入排列重要性分析函数
from ._permutation_importance import permutation_importance
# 导入决策边界显示类
from ._plot.decision_boundary import DecisionBoundaryDisplay
# 导入局部依赖显示类
from ._plot.partial_dependence import PartialDependenceDisplay

# 将以下标识符添加到模块的公共接口中
__all__ = [
    "partial_dependence",        # 局部依赖分析函数
    "permutation_importance",     # 排列重要性分析函数
    "PartialDependenceDisplay",  # 展示局部依赖的显示类
    "DecisionBoundaryDisplay",   # 决策边界显示的显示类
]
```