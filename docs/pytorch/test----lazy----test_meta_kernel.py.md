# `.\pytorch\test\lazy\test_meta_kernel.py`

```
# Owner(s): ["oncall: jit"]

# 导入 PyTorch 库
import torch
# 导入 PyTorch 内部的延迟加载模块
import torch._lazy
# 导入 PyTorch 内部的时间序列后端模块
import torch._lazy.ts_backend
# 从 torch 模块中导入 float16 和 float32 数据类型
from torch import float16, float32
# 导入测试用例基类
from torch.testing._internal.common_utils import TestCase

# 初始化延迟加载模块的时间序列后端
torch._lazy.ts_backend.init()

# 测试 MetaKernel 类
class TestMetaKernel(TestCase):
    
    # 测试 addmm 方法在无效数据类型时的行为
    def test_addmm_invalid_dtype(self):
        """Tests that the addmm meta kernel returns the correct output type"""
        # 创建一个所有元素为 1 的 2x2 的 float16 类型张量，并转换为 lazy 模式
        input = torch.ones(2, 2, dtype=torch.float16).to("lazy")
        self.assertTrue(input.dtype == torch.float16)

        # 创建一个无偏置的线性层，输入输出都是 float32 类型，也转换为 lazy 模式
        fc_nobias = torch.nn.Linear(2, 2, bias=False, dtype=float32).to("lazy")

        # 断言在此操作中会引发异常
        with self.assertRaises(Exception):
            out_nobias = fc_nobias(input)

    # 测试 addmm 方法的基本功能
    def test_addmm(self):
        """Tests that the addmm meta kernel returns the correct output type"""
        # 创建一个所有元素为 1 的 2x2 的 float16 类型张量，并转换为 lazy 模式
        input = torch.ones(2, 2, dtype=torch.float16).to("lazy")
        self.assertEqual(input.dtype, torch.float16)

        # 创建一个无偏置的线性层，输入输出都是 float16 类型，也转换为 lazy 模式
        fc_nobias = torch.nn.Linear(2, 2, bias=False, dtype=float16).to("lazy")
        # 对输入进行线性变换
        out_nobias = fc_nobias(input)
        self.assertEqual(out_nobias.dtype, torch.float16)

        # 创建一个有偏置的线性层，输入输出都是 float16 类型，也转换为 lazy 模式
        fc_bias = torch.nn.Linear(2, 2, bias=True, dtype=float16).to("lazy")
        # 对输入进行线性变换
        out_bias = fc_bias(input)
        self.assertEqual(out_bias.dtype, torch.float16)

    # 测试在无效设备上进行加法操作时的行为
    def test_add_invalid_device(self):
        # 断言在此操作中会引发 RuntimeError，错误信息包含 "not a lazy tensor"
        with self.assertRaisesRegex(RuntimeError, ".*not a lazy tensor.*"):
            _ = torch.tensor([1], device="cpu") + torch.tensor([1], device="lazy")
```