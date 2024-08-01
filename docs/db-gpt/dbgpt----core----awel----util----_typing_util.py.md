# `.\DB-GPT-src\dbgpt\core\awel\util\_typing_util.py`

```py
# 导入需要的类型提示模块 Any
from typing import Any

# 定义一个函数 _parse_bool，用于将输入值解析为布尔类型
def _parse_bool(v: Any) -> bool:
    """Parse a value to bool."""
    # 如果输入值 v 是 None，返回 False
    if v is None:
        return False
    # 将输入值 v 转换为小写字符串后，如果在指定的假值列表中，则返回 False
    if str(v).lower() in ["false", "0", "", "no", "off"]:
        return False
    # 否则，将输入值 v 转换为布尔类型并返回
    return bool(v)
```