# `.\pytorch\torch\testing\_internal\test_module\no_future_div.py`

```py
# 忽略 mypy 对错误的检查
# 导入 torch 库，并忽略 F401 错误（未使用的导入警告）
import torch  # noqa: F401


# 定义一个函数 div_int_nofuture，用于整数除法
def div_int_nofuture():
    # 返回整数除法结果 1 除以 2
    return 1 / 2


# 定义一个函数 div_float_nofuture，用于浮点数除法
def div_float_nofuture():
    # 返回浮点数除法结果 3.14 除以 0.125
    return 3.14 / 0.125
```