# `.\pytorch\torch\ao\pruning\_experimental\data_sparsifier\__init__.py`

```py
# 导入当前目录中的base_data_sparsifier模块中的BaseDataSparsifier类
from .base_data_sparsifier import BaseDataSparsifier
# 导入当前目录中的data_norm_sparsifier模块中的DataNormSparsifier类
from .data_norm_sparsifier import DataNormSparsifier

# 定义__all__列表，用于声明当前模块中可以被导出的符号（变量、类、函数等）
__all__ = [
    "BaseDataSparsifier",  # 将BaseDataSparsifier类添加到__all__列表中
    "DataNormSparsifier",  # 将DataNormSparsifier类添加到__all__列表中
]
```