# `.\pytorch\torch\cuda\amp\common.py`

```py
# 设置 mypy 参数以允许未类型化的定义
# 导入 find_spec 函数用于检查模块是否存在
from importlib.util import find_spec

# 导入 torch 模块
import torch

# 定义导出的符号列表，这里仅包含 "amp_definitely_not_available"
__all__ = ["amp_definitely_not_available"]

# 定义函数 amp_definitely_not_available，用于检查 AMP 是否可用
def amp_definitely_not_available():
    # 返回逻辑非表达式，检查 CUDA 是否可用或者 torch_xla 模块是否存在
    return not (torch.cuda.is_available() or find_spec("torch_xla"))
```