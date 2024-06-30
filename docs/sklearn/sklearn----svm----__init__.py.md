# `D:\src\scipysrc\scikit-learn\sklearn\svm\__init__.py`

```
"""Support vector machine algorithms."""

# 导入必要的模块和函数，这些模块和函数用于支持向量机算法的实现
# 详细文档请参考 http://scikit-learn.sourceforge.net/modules/svm.html

# 以下是该模块的作者信息和许可证信息
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# 从内部模块中导入所需的函数和类
from ._bounds import l1_min_c
from ._classes import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM

# __all__ 列表定义了该模块中哪些对象可以被 * 导入
__all__ = [
    "LinearSVC",    # 线性支持向量分类器类
    "LinearSVR",    # 线性支持向量回归类
    "NuSVC",        # 支持向量分类器类（支持类别加权）
    "NuSVR",        # 支持向量回归类（支持样本加权）
    "OneClassSVM",  # 单类别支持向量机类
    "SVC",          # 支持向量分类器类
    "SVR",          # 支持向量回归类
    "l1_min_c",     # 计算 L1 正则化下的最小 C 值
]
```