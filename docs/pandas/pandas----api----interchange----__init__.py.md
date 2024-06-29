# `D:\src\scipysrc\pandas\pandas\api\interchange\__init__.py`

```
"""
Public API for DataFrame interchange protocol.
"""

# 从 pandas 库中导入 DataFrame 类
from pandas.core.interchange.dataframe_protocol import DataFrame
# 从 pandas 库中导入 from_dataframe 函数
from pandas.core.interchange.from_dataframe import from_dataframe

# 定义模块中对外暴露的全部接口
__all__ = ["from_dataframe", "DataFrame"]
```