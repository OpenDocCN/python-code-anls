# `D:\src\scipysrc\pandas\pandas\core\sparse\api.py`

```
# 导入稀疏数据类型 SparseDtype 和稀疏数组 SparseArray
from pandas.core.dtypes.dtypes import SparseDtype
from pandas.core.arrays.sparse import SparseArray

# 将 SparseArray 和 SparseDtype 添加到模块的公开接口中
__all__ = ["SparseArray", "SparseDtype"]
```