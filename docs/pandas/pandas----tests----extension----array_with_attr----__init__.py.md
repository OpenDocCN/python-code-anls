# `D:\src\scipysrc\pandas\pandas\tests\extension\array_with_attr\__init__.py`

```
# 导入 pandas 库中的扩展测试模块中的特定类和类型
from pandas.tests.extension.array_with_attr.array import (
    FloatAttrArray,  # 导入 FloatAttrArray 类，用于处理带有属性的数组
    FloatAttrDtype,  # 导入 FloatAttrDtype 类，用于处理带有属性的浮点数类型
)

# 指定当前模块中可以公开访问的类和类型
__all__ = ["FloatAttrArray", "FloatAttrDtype"]
```