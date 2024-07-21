# `.\pytorch\test\jit\test_aten_pow.py`

```py
# Owner(s): ["oncall: jit"]
# 导入 PyTorch 库
import torch
# 导入测试相关的工具类 TestCase
from torch.testing._internal.common_utils import TestCase

# 定义 TestAtenPow 类，继承自 TestCase，用于测试 torch.pow 函数的功能
class TestAtenPow(TestCase):
```