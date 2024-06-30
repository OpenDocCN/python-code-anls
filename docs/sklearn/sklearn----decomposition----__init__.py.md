# `D:\src\scipysrc\scikit-learn\sklearn\decomposition\__init__.py`

```
# 矩阵分解算法模块的声明与介绍文档
"""Matrix decomposition algorithms.

These include PCA, NMF, ICA, and more. Most of the algorithms of this module can be
regarded as dimensionality reduction techniques.
"""

# 导入随机化奇异值分解函数
from ..utils.extmath import randomized_svd
# 导入字典学习相关模块和函数
from ._dict_learning import (
    DictionaryLearning,
    MiniBatchDictionaryLearning,
    SparseCoder,
    dict_learning,
    dict_learning_online,
    sparse_encode,
)
# 导入因子分析模块
from ._factor_analysis import FactorAnalysis
# 导入快速独立成分分析模块和函数
from ._fastica import FastICA, fastica
# 导入增量主成分分析模块
from ._incremental_pca import IncrementalPCA
# 导入核主成分分析模块
from ._kernel_pca import KernelPCA
# 导入潜在狄利克雷分配模块
from ._lda import LatentDirichletAllocation
# 导入非负矩阵分解模块和函数
from ._nmf import (
    NMF,
    MiniBatchNMF,
    non_negative_factorization,
)
# 导入主成分分析模块
from ._pca import PCA
# 导入稀疏主成分分析模块
from ._sparse_pca import MiniBatchSparsePCA, SparsePCA
# 导入截断奇异值分解模块
from ._truncated_svd import TruncatedSVD

# 导出所有模块和函数的名称列表
__all__ = [
    "DictionaryLearning",
    "FastICA",
    "IncrementalPCA",
    "KernelPCA",
    "MiniBatchDictionaryLearning",
    "MiniBatchNMF",
    "MiniBatchSparsePCA",
    "NMF",
    "PCA",
    "SparseCoder",
    "SparsePCA",
    "dict_learning",
    "dict_learning_online",
    "fastica",
    "non_negative_factorization",
    "randomized_svd",
    "sparse_encode",
    "FactorAnalysis",
    "TruncatedSVD",
    "LatentDirichletAllocation",
]
```