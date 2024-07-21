# `.\pytorch\torch\utils\data\datapipes\dataframe\__init__.py`

```py
# 从 torch.utils.data.datapipes.dataframe.dataframes 中导入 CaptureDataFrame 和 DFIterDataPipe 类
# 以及从 torch.utils.data.datapipes.dataframe.datapipes 中导入 DataFramesAsTuplesPipe 类
from torch.utils.data.datapipes.dataframe.dataframes import (
    CaptureDataFrame,
    DFIterDataPipe,
)
from torch.utils.data.datapipes.dataframe.datapipes import DataFramesAsTuplesPipe

# 定义一个包含需要公开的类名的列表
__all__ = ["CaptureDataFrame", "DFIterDataPipe", "DataFramesAsTuplesPipe"]

# 断言语句，确保 __all__ 列表中的类名按字母顺序排列
assert __all__ == sorted(__all__)
```