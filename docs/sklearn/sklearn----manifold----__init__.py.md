# `D:\src\scipysrc\scikit-learn\sklearn\manifold\__init__.py`

```
"""Data embedding techniques."""

# 导入 Isomap 类
from ._isomap import Isomap
# 导入 LocallyLinearEmbedding 类和 locally_linear_embedding 函数
from ._locally_linear import LocallyLinearEmbedding, locally_linear_embedding
# 导入 MDS 类和 smacof 函数
from ._mds import MDS, smacof
# 导入 SpectralEmbedding 类和 spectral_embedding 函数
from ._spectral_embedding import SpectralEmbedding, spectral_embedding
# 导入 TSNE 类和 trustworthiness 函数
from ._t_sne import TSNE, trustworthiness

# 指定模块中公开的对象列表
__all__ = [
    "locally_linear_embedding",
    "LocallyLinearEmbedding",
    "Isomap",
    "MDS",
    "smacof",
    "SpectralEmbedding",
    "spectral_embedding",
    "TSNE",
    "trustworthiness",
]
```