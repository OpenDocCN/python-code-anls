# `D:\src\scipysrc\pandas\pandas\tests\extension\decimal\__init__.py`

```
# 从 pandas 的测试模块中导入 decimal 数组相关的组件
from pandas.tests.extension.decimal.array import (
    DecimalArray,   # 导入 DecimalArray 类，用于处理 decimal 类型的数组
    DecimalDtype,   # 导入 DecimalDtype 类，用于定义 decimal 类型的数据类型
    make_data,      # 导入 make_data 函数，用于生成测试数据
    to_decimal,     # 导入 to_decimal 函数，用于将其他类型转换为 decimal 类型
)

# 将以下组件列入此模块的公共接口
__all__ = ["DecimalArray", "DecimalDtype", "to_decimal", "make_data"]
```