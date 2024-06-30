# `D:\src\scipysrc\scikit-learn\sklearn\model_selection\tests\common.py`

```
"""
Common utilities for testing model selection.
"""

import numpy as np  # 导入 NumPy 库

from sklearn.model_selection import KFold  # 导入 KFold 模型选择工具


class OneTimeSplitter:
    """A wrapper to make KFold single entry cv iterator"""

    def __init__(self, n_splits=4, n_samples=99):
        self.n_splits = n_splits  # 初始化折数
        self.n_samples = n_samples  # 初始化样本数
        self.indices = iter(KFold(n_splits=n_splits).split(np.ones(n_samples)))  # 创建 KFold 对象的迭代器

    def split(self, X=None, y=None, groups=None):
        """Split can be called only once"""
        for index in self.indices:  # 遍历迭代器中的索引
            yield index  # 返回索引

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits  # 返回折数
```