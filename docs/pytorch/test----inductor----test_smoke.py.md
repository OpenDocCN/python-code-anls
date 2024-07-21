# `.\pytorch\test\inductor\test_smoke.py`

```
# Owner(s): ["module: inductor"]
# 导入日志模块和单元测试模块
import logging
import unittest

# 导入PyTorch相关模块
import torch
import torch._logging

# 导入自定义的测试用例和测试工具函数
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CUDA, HAS_GPU

# 定义一个简单的多层感知机模型类
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义两个全连接层
        self.l1 = torch.nn.Linear(1, 6)
        self.l2 = torch.nn.Linear(6, 1)

    def forward(self, x=None):
        # 前向传播函数，先通过第一层全连接层，然后使用ReLU激活函数
        x = torch.relu(self.l1(x))
        # 再通过第二层全连接层，再次使用ReLU激活函数
        x = torch.relu(self.l2(x))
        return x

# 定义一个简单的函数，用于计算输入的平方
def _test_f(x):
    return x * x

# 定义一个测试类，继承自TestCase
class SmokeTest(TestCase):
    # 装饰器，用于标记该测试函数，若没有GPU则跳过
    @unittest.skipIf(not HAS_GPU, "Triton is not available")
    def test_mlp(self):
        # 设置多个日志级别
        torch._logging.set_logs(
            dynamo=logging.DEBUG, inductor=logging.DEBUG, aot=logging.DEBUG
        )

        # 编译MLP模型，并将其部署到指定的GPU类型
        mlp = torch.compile(MLP().to(GPU_TYPE))
        # 多次执行MLP模型的前向传播
        for _ in range(3):
            mlp(torch.randn(1, device=GPU_TYPE))

        # 恢复日志设置为默认
        torch._logging.set_logs()

    # 装饰器，用于标记该测试函数，若没有GPU则跳过
    @unittest.skipIf(not HAS_GPU, "Triton is not available")
    def test_compile_decorator(self):
        # 使用torch.compile装饰器定义一个函数
        @torch.compile
        def foo(x):
            return torch.sin(x) + x.min()

        # 使用torch.compile装饰器定义另一个函数，并指定模式为"reduce-overhead"
        @torch.compile(mode="reduce-overhead")
        def bar(x):
            return x * x

        # 多次执行foo和bar函数
        for _ in range(3):
            foo(torch.full((3, 4), 0.7, device=GPU_TYPE))
            bar(torch.rand((2, 2), device=GPU_TYPE))

    # 测试编译器选项不合法时是否引发异常
    def test_compile_invalid_options(self):
        with self.assertRaises(RuntimeError):
            opt_f = torch.compile(_test_f, mode="ha")

# 如果该脚本作为主程序运行
if __name__ == "__main__":
    # 导入并执行测试用例
    from torch._inductor.test_case import run_tests

    # 如果是Linux系统且有GPU可用
    if IS_LINUX and HAS_GPU:
        # 如果没有CUDA或者第一个GPU设备的主要版本号小于等于5
        if (not HAS_CUDA) or torch.cuda.get_device_properties(0).major <= 5:
            run_tests()
```