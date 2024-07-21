# `.\pytorch\test\ao\sparsity\test_qlinear_packed_params.py`

```
#!/usr/bin/env python3
# Owner(s): ["oncall: mobile"]

# 导入临时文件模块
import tempfile

# 导入PyTorch相关模块
import torch
from torch.ao.nn.sparse.quantized.dynamic.linear import Linear
from torch.testing._internal.common_quantization import skipIfNoFBGEMM, skipIfNoQNNPACK
from torch.testing._internal.common_quantized import (
    override_cpu_allocator_for_qnnpack,
    override_quantized_engine,
    qengine_is_qnnpack,
)
from torch.testing._internal.common_utils import TestCase


# 定义测试类 TestQlinearPackedParams，继承自 TestCase 类
class TestQlinearPackedParams(TestCase):
    
    # 装饰器，如果没有安装 FBGEMM 库，则跳过此测试
    @skipIfNoFBGEMM
    def test_qlinear_packed_params_fbgemm(self):
        # 设置随机种子为0
        torch.manual_seed(0)
        # 使用 FBGEMM 引擎运行下面的测试
        with override_quantized_engine("fbgemm"):
            # 调用 qlinear_packed_params_test 方法进行测试，禁止非零零点
            self.qlinear_packed_params_test(allow_non_zero_zero_points=False)

    # 装饰器，如果没有安装 QNNPACK 库，则跳过此测试
    @skipIfNoQNNPACK
    def test_qlinear_packed_params_qnnpack(self):
        # 设置随机种子为0
        torch.manual_seed(0)
        # 使用 QNNPACK 引擎运行下面的测试
        with override_quantized_engine("qnnpack"):
            # 使用 QNNPACK 的 CPU 分配器来运行测试，检查是否 QNNPACK 引擎
            with override_cpu_allocator_for_qnnpack(qengine_is_qnnpack()):
                # 调用 qlinear_packed_params_test 方法进行测试，允许非零零点
                self.qlinear_packed_params_test(allow_non_zero_zero_points=True)
```