# `D:\src\scipysrc\pandas\pandas\tests\extension\date\__init__.py`

```
# 从pandas测试模块中导入日期数组（DateArray）和日期数据类型（DateDtype）
from pandas.tests.extension.date.array import (
    DateArray,
    DateDtype,
)

# 将DateArray和DateDtype添加到__all__列表中，以便它们可以通过`from module import *`的形式被导入
__all__ = ["DateArray", "DateDtype"]
```