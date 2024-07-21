# `.\pytorch\test\quantization\jit\test_fusion_passes.py`

```
# 导入 torch 库
import torch
# 导入用于测试的 FileCheck 工具
from torch.testing import FileCheck
# 导入量化测试相关的基类 QuantizationTestCase
from torch.testing._internal.common_quantization import QuantizationTestCase

# 定义测试类 TestFusionPasses，继承自 QuantizationTestCase
class TestFusionPasses(QuantizationTestCase):
```