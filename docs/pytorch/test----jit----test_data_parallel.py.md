# `.\pytorch\test\jit\test_data_parallel.py`

```
# Owner(s): ["oncall: jit"]

# 导入必要的库
import os  # 导入操作系统模块
import sys  # 导入系统模块
import unittest  # 导入单元测试模块

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.parallel as dp  # 导入PyTorch的数据并行模块

# 让 test/ 目录下的辅助文件可以被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, RUN_CUDA_MULTI_GPU  # 导入测试工具函数和标记

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义测试类，继承自 JitTestCase
class TestDataParallel(JitTestCase):
    
    # 定义一个简单的模块 Mpy
    class Mpy(torch.nn.Module):
        def __init__(self):
            super(TestDataParallel.Mpy, self).__init__()
            # 定义一个包含线性层、批归一化、ReLU和线性层的序列模块
            self.m = nn.Sequential(
                nn.Linear(2, 2), nn.BatchNorm1d(2), nn.ReLU(), nn.Linear(2, 2)
            )

        @torch.jit.ignore
        def forward(self, input):
            return self.m(input)

    # 定义另一个简单的模块 Mpy1
    class Mpy1(torch.nn.Module):
        def __init__(self, block):
            super(TestDataParallel.Mpy1, self).__init__()
            self.m = block  # 使用传入的块作为模块的成员变量

        @torch.jit.ignore
        def forward(self, input):
            return self.m.forward(input)

    # 定义一个复合模块 Mpy2
    class Mpy2(torch.nn.Module):
        def __init__(self, block1, block2):
            super(TestDataParallel.Mpy2, self).__init__()
            self.m1 = block1  # 使用传入的第一个块作为模块的成员变量
            self.m2 = block2  # 使用传入的第二个块作为模块的成员变量

        @torch.jit.ignore
        def forward(self, input):
            x = self.m1.forward(input)  # 对输入应用第一个块
            return self.m2(x)  # 对第一个块的输出应用第二个块

    # 定义一个脚本模块 Msm
    class Msm(torch.jit.ScriptModule):
        __constants__ = ["m"]

        def __init__(self):
            super(TestDataParallel.Msm, self).__init__()
            # 定义一个包含线性层、批归一化、ReLU和线性层的序列模块
            self.m = nn.Sequential(
                nn.Linear(2, 2), nn.BatchNorm1d(2), nn.ReLU(), nn.Linear(2, 2)
            )

        @torch.jit.script_method
        def forward(self, input):
            return self.m(input)

    # 定义一个脚本模块 Msm1
    class Msm1(torch.jit.ScriptModule):
        def __init__(self, block):
            super(TestDataParallel.Msm1, self).__init__()
            self.block = block  # 使用传入的块作为模块的成员变量

        @torch.jit.script_method
        def forward(self, input):
            x = self.block(input)  # 对输入应用传入的块
            return x

    # 检查模块副本的设备分配情况
    def check_replicas(self, module, replicas, input_shape=(2, 2)):
        input = torch.randn(input_shape).cuda()  # 生成一个随机输入张量，并移到 GPU 上
        expected_output = module(input).data  # 计算预期输出
        for i, replica in enumerate(replicas):
            for p in replica.parameters():
                self.assertEqual(p.get_device(), i)  # 断言每个参数在正确的设备上
            for b in replica.buffers():
                self.assertEqual(b.get_device(), i)  # 断言每个缓冲区在正确的设备上
            replica_input = input.cuda(i)  # 将输入移到指定的 GPU 上
            self.assertEqual(replica(replica_input).data, expected_output)  # 断言副本的输出与预期输出相等

    # 如果不支持多 GPU，则跳过测试
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_python_submodule_script(self):
        module = self.Mpy1(self.Msm()).cuda()  # 创建一个 Mpy1 实例，使用 Msm 实例作为块，并移到 GPU 上
        replicas = dp.replicate(module, {0, 1})  # 复制模块到多个 GPU 上
        self.check_replicas(module, replicas)  # 检查副本的设备分配情况
    # 如果不支持多 GPU，则跳过此测试
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_shared_module(self):
        # 创建主模块实例
        s = self.Msm()
        # 使用主模块创建第一个子模块实例
        p1 = self.Mpy1(s)
        # 使用第一个子模块和主模块创建第二个子模块实例，并将其移到 GPU 上
        module = self.Mpy2(p1, s).cuda()
        # 复制模块到多个 GPU 上
        replicas = dp.replicate(module, {0, 1})
        # 检查复制的模块是否符合预期
        self.check_replicas(module, replicas)
    
    # 如果不支持多 GPU，则跳过此测试
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_traced_module(self):
        # 对 Mpy1 的跟踪模块进行 JIT 编译，并移到 GPU 上
        module = torch.jit.trace(self.Mpy1(self.Mpy()), torch.ones(2, 2)).cuda()
        # 复制模块到多个 GPU 上
        replicas = dp.replicate(module, {0, 1})
        # 检查复制的模块是否符合预期
        self.check_replicas(module, replicas)
    
    # 如果不支持多 GPU，则跳过此测试
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_tensor_sharing(self):
        # 创建主模块实例，并移到 GPU 上
        module = self.Msm1(self.Msm()).cuda()
        # 复制模块到多个 GPU 上
        replica = dp.replicate(module, {0, 1})
    
        # 定义函数，用于检查两个张量是否共享相同的内存位置和设备
        def assert_share_data(t1, t2):
            # 只检查它们是否指向同一设备上的相同内存
            if t1.device != t2.device:
                return False
            if t1.storage().data_ptr() != t2.storage().data_ptr():
                return False
            return True
    
        # 检查每个 GPU 上的第一个副本与主模块的参数是否共享数据
        for p1, p2 in zip(module.parameters(), replica[0].parameters()):
            self.assertTrue(assert_share_data(p1, p2))
    
        # 检查每个 GPU 上的第一个副本与主模块的缓冲区是否共享数据
        for p1, p2 in zip(module.buffers(), replica[0].buffers()):
            self.assertTrue(assert_share_data(p1, p2))
    
        # 检查每个 GPU 上的第二个副本与主模块的参数是否没有共享数据
        for p1, p2 in zip(module.parameters(), replica[1].parameters()):
            self.assertFalse(assert_share_data(p1, p2))
    
        # 检查每个 GPU 上的第二个副本与主模块的缓冲区是否没有共享数据
        for p1, p2 in zip(module.buffers(), replica[1].buffers()):
            self.assertFalse(assert_share_data(p1, p2))
    
    # 如果不支持多 GPU，则跳过此测试
    @unittest.skipIf(not RUN_CUDA_MULTI_GPU, "multi-GPU not supported")
    def test_tensor_sharing_with_forward(self):
        # 创建主模块实例，并移到 GPU 上
        module = self.Msm1(self.Msm()).cuda()
        # 复制模块到多个 GPU 上
        replica = dp.replicate(module, {0, 1})
        # 创建一个张量并移到 GPU 上，启用梯度计算
        x = torch.ones(2, 2, requires_grad=True).cuda()
        # 执行第一次前向传播
        first_forward = module(x)
        # 对第一次前向传播的结果执行反向传播
        first_forward.sum().backward()
        with torch.no_grad():
            # 针对每个参数，使用 .data 来避免版本计数增加
            # 以下前向传播创建的图可能会有问题，但我们不会通过它们进行反向传播，所以没问题
            for p in module.parameters():
                p.data -= 1.0 * p.grad
        # 执行第二次前向传播
        second_forward = module(x)
    
        # 检查同一 GPU 上的副本是否浅复制了原始参数和缓冲区
        r0_forward = replica[0](x)
        self.assertEqual(second_forward, r0_forward)
    
        # 检查不同 GPU 上的副本是否深复制了原始参数和缓冲区
        x1 = torch.ones(2, 2, requires_grad=True).cuda(device=1)
        r1_forward = replica[1](x1)
        self.assertEqual(first_forward, r1_forward)
```