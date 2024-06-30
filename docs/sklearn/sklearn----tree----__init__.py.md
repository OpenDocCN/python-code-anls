# `D:\src\scipysrc\scikit-learn\sklearn\tree\__init__.py`

```
"""Decision tree based models for classification and regression."""

# 从 _classes 模块导入以下类，用于分类和回归的决策树模型
from ._classes import (
    BaseDecisionTree,              # 基础决策树模型的基类
    DecisionTreeClassifier,        # 决策树分类器模型
    DecisionTreeRegressor,         # 决策树回归器模型
    ExtraTreeClassifier,           # 额外树分类器模型
    ExtraTreeRegressor,            # 额外树回归器模型
)

# 从 _export 模块导入用于决策树的导出工具函数
from ._export import export_graphviz,  # 导出决策树模型到 Graphviz 格式
                      export_text,     # 导出决策树模型的文本表示
                      plot_tree        # 绘制决策树图形化展示工具

# __all__ 是一个列表，声明了当前模块中哪些对象可以通过 from module import * 的方式导入
__all__ = [
    "BaseDecisionTree",             # 基础决策树模型
    "DecisionTreeClassifier",       # 决策树分类器模型
    "DecisionTreeRegressor",        # 决策树回归器模型
    "ExtraTreeClassifier",          # 额外树分类器模型
    "ExtraTreeRegressor",           # 额外树回归器模型
    "export_graphviz",              # 导出决策树模型到 Graphviz 格式的函数
    "plot_tree",                    # 绘制决策树图形化展示的函数
    "export_text",                  # 导出决策树模型的文本表示函数
]
```