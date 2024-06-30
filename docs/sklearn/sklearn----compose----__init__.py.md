# `D:\src\scipysrc\scikit-learn\sklearn\compose\__init__.py`

```
"""
Meta-estimators for building composite models with transformers.

In addition to its current contents, this module will eventually be home to
refurbished versions of :class:`~sklearn.pipeline.Pipeline` and
:class:`~sklearn.pipeline.FeatureUnion`.
"""

# 导入 ColumnTransformer 相关模块，用于构建复合模型
from ._column_transformer import (
    ColumnTransformer,  # 导入 ColumnTransformer 类
    make_column_selector,  # 导入 make_column_selector 函数
    make_column_transformer,  # 导入 make_column_transformer 函数
)
# 导入 TransformedTargetRegressor 类，用于目标变量的转换
from ._target import TransformedTargetRegressor

# 模块中公开的对象列表
__all__ = [
    "ColumnTransformer",  # 将 ColumnTransformer 加入公开对象列表
    "make_column_transformer",  # 将 make_column_transformer 加入公开对象列表
    "TransformedTargetRegressor",  # 将 TransformedTargetRegressor 加入公开对象列表
    "make_column_selector",  # 将 make_column_selector 加入公开对象列表
]
```