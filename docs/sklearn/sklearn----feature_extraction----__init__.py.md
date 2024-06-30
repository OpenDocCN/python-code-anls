# `D:\src\scipysrc\scikit-learn\sklearn\feature_extraction\__init__.py`

```
"""Feature extraction from raw data."""

# 导入模块 `text`，使用相对路径导入当前包下的子模块
from . import text

# 导入 `DictVectorizer` 类，使用相对路径导入当前包下的子模块
from ._dict_vectorizer import DictVectorizer

# 导入 `FeatureHasher` 类，使用相对路径导入当前包下的子模块
from ._hash import FeatureHasher

# 导入 `grid_to_graph` 和 `img_to_graph` 函数，使用相对路径导入当前包下的子模块 `image`
from .image import grid_to_graph, img_to_graph

# 定义一个列表，包含了当前模块中所有公开的符号
__all__ = [
    "DictVectorizer",  # 将 `DictVectorizer` 添加到公开接口
    "image",           # 将 `image` 模块添加到公开接口
    "img_to_graph",    # 将 `img_to_graph` 函数添加到公开接口
    "grid_to_graph",   # 将 `grid_to_graph` 函数添加到公开接口
    "text",            # 将 `text` 模块添加到公开接口
    "FeatureHasher",   # 将 `FeatureHasher` 添加到公开接口
]
```