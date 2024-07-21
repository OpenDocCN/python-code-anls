# `.\pytorch\test\fx\named_tup.py`

```py
# 导入 NamedTuple 类型定义
from typing import NamedTuple
# 导入 PyTorch 库
import torch

# 定义一个名为 MyNamedTup 的自定义命名元组
class MyNamedTup(NamedTuple):
    # 定义命名元组的字段 i，类型为 torch.Tensor
    i: torch.Tensor
    # 定义命名元组的字段 f，类型为 torch.Tensor
    f: torch.Tensor
```