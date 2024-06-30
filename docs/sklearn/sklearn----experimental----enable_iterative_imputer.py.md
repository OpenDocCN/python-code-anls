# `D:\src\scipysrc\scikit-learn\sklearn\experimental\enable_iterative_imputer.py`

```
"""
Enables IterativeImputer

The API and results of this estimator might change without any deprecation
cycle.

Importing this file dynamically sets :class:`~sklearn.impute.IterativeImputer`
as an attribute of the impute module::

    >>> # explicitly require this experimental feature
    >>> from sklearn.experimental import enable_iterative_imputer  # noqa
    >>> # now you can import normally from impute
    >>> from sklearn.impute import IterativeImputer
"""

# 从相对路径导入 impute 模块
from .. import impute
# 从 impute 模块导入 IterativeImputer 类
from ..impute._iterative import IterativeImputer

# 使用 setattr 避免在 monkeypatching 时出现 mypy 错误
# 将 IterativeImputer 类设置为 impute 模块的属性
setattr(impute, "IterativeImputer", IterativeImputer)
# 向 impute 模块的 __all__ 属性中添加 IterativeImputer，确保在导入时被包含
impute.__all__ += ["IterativeImputer"]
```