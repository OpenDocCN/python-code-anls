# `D:\src\scipysrc\pandas\pandas\tests\base\common.py`

```
# 从 typing 模块导入 Any 类型
# 从 pandas 模块导入 Index 类
from typing import Any
from pandas import Index

# 定义函数 allow_na_ops，用于确定是否跳过包含 NaN 的测试用例
def allow_na_ops(obj: Any) -> bool:
    # 检查 obj 是否为 Index 类型且推断类型为布尔型
    is_bool_index = isinstance(obj, Index) and obj.inferred_type == "boolean"
    # 返回结果：如果不是布尔型 Index 且 obj 能够容纳 NaN，则返回 True
    return not is_bool_index and obj._can_hold_na
```