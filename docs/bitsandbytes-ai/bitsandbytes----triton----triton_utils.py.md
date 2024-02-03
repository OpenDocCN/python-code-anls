# `bitsandbytes\bitsandbytes\triton\triton_utils.py`

```py
# 导入 importlib 模块，用于动态导入模块
import importlib

# 检查是否 triton 模块可用，返回布尔值
def is_triton_available():
    # 使用 importlib.util.find_spec 函数查找 triton 模块的规范对象，如果找到则返回 True，否则返回 False
    return importlib.util.find_spec("triton") is not None
```