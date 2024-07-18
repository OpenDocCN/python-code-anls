# `.\graphrag\graphrag\index\utils\is_null.py`

```py
# 定义一个 Python 脚本，声明版权信息及许可证
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入 math 模块，用于数学运算
import math
# 导入 Any 类型，用于指示可以是任意类型的参数和返回值
from typing import Any

# 定义一个函数 is_null，用于检查值是否为 null 或 NaN
def is_null(value: Any) -> bool:
    """Check if value is null or is nan."""
    
    # 定义内部函数 is_none，检查值是否为 None
    def is_none() -> bool:
        return value is None
    
    # 定义内部函数 is_nan，检查值是否为浮点数且为 NaN
    def is_nan() -> bool:
        return isinstance(value, float) and math.isnan(value)
    
    # 返回值为 is_none() 或 is_nan() 的逻辑或结果
    return is_none() or is_nan()
```