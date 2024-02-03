# `bitsandbytes\tests\helpers.py`

```py
# 导入必要的模块
from itertools import product
import random
from typing import Any

import torch

# 设置随机数生成器
test_dims_rng = random.Random(42)

# 生成一个包含 n 个随机整数的列表
def get_test_dims(min: int, max: int, *, n: int) -> list[int]:
    return [test_dims_rng.randint(min, max) for _ in range(n)]

# 根据值的类型格式化字符串
def format_with_label(label: str, value: Any) -> str:
    if isinstance(value, bool):
        formatted = "T" if value else "F"
    elif isinstance(value, (list, tuple)) and all(isinstance(v, bool) for v in value):
        formatted = "".join("T" if b else "F" for b in value)
    else:
        formatted = str(value)
    return f"{label}={formatted}"

# 返回一个函数，用于格式化给定标签的值
def id_formatter(label: str):
    """
    Return a function that formats the value given to it with the given label.
    """
    return lambda value: format_with_label(label, value)

# 定义不同 torch 数据类型的名称映射
DTYPE_NAMES = {
    torch.bfloat16: "bf16",
    torch.bool: "bool",
    torch.float16: "fp16",
    torch.float32: "fp32",
    torch.float64: "fp64",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.int8: "int8",
}

# 返回给定 torch 数据类型的描述字符串
def describe_dtype(dtype: torch.dtype) -> str:
    return DTYPE_NAMES.get(dtype) or str(dtype).rpartition(".")[2]

# 定义常量值
TRUE_FALSE = (True, False)
BOOLEAN_TRIPLES = list(
    product(TRUE_FALSE, repeat=3)
)  # all combinations of (bool, bool, bool)
BOOLEAN_TUPLES = list(product(TRUE_FALSE, repeat=2))  # all combinations of (bool, bool)
```