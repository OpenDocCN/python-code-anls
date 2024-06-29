# `D:\src\scipysrc\pandas\pandas\tests\extension\json\__init__.py`

```
# 从 pandas 库的测试模块中导入以下三个对象：JSONArray, JSONDtype, make_data
from pandas.tests.extension.json.array import (
    JSONArray,
    JSONDtype,
    make_data,
)

# 设置当前模块可以导出的公共对象列表，只包括 JSONArray, JSONDtype 和 make_data
__all__ = ["JSONArray", "JSONDtype", "make_data"]
```