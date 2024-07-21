# `.\pytorch\test\distributed\_composable\test_checkpoint.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import unittest
from collections import deque, OrderedDict
from contextlib import ContextDecorator, contextmanager, nullcontext
from copy import deepcopy
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributed._composable import checkpoint
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.checkpoint import CheckpointError


# 自定义类 MemoryDelta，继承自 ContextDecorator 上下文装饰器
class MemoryDelta(ContextDecorator):
    def __init__(self, device: torch.device):
        self.device: torch.device = device
        self.active_memory_enter: int = 0
        self.active_memory_exit: int = 0

    # 进入上下文时调用的方法
    def __enter__(self):
        # 如果设备类型是 CUDA，则获取当前活跃内存字节数，否则为 0
        self.active_memory_enter = (
            torch.cuda.memory_stats()["active_bytes.all.current"]
            if self.device.type == "cuda"
            else 0
        )
        return self

    # 退出上下文时调用的方法
    def __exit__(self, *exc):
        # 如果设备类型是 CUDA，则获取当前活跃内存字节数，否则为 0
        self.active_memory_exit = (
            torch.cuda.memory_stats()["active_bytes.all.current"]
            if self.device.type == "cuda"
            else 0
        )

    # 计算活跃内存变化量的方法
    def delta(self) -> int:
        return self.active_memory_exit - self.active_memory_enter


# 定义一个简单的神经网络模型 ToyModel
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(100, 100)
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )

    # 前向传播方法
    def forward(self, x):
        return self.seq(self.l1(x))


# 定义一个包含随机参数的神经网络模型 RandomModel
class RandomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.randn(100, 100))

    # 前向传播方法
    def forward(self, x):
        y = torch.matmul(self.p, torch.randn(100, 100, device=self.p.device))
        return torch.matmul(x, y)


# 定义一个包含多个输出的神经网络模型 MultiOutputModel
class MultiOutputModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn((100, 100), device=device))
        self.w2 = nn.Parameter(torch.randn((100, 100), device=device))

    # 前向传播方法，返回两个张量作为输出
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x @ self.w1
        z = nn.functional.relu(z)
        z = z @ self.w2
        return z.sin(), z.cos()


# 定义一个包含多个输入的神经网络模型 MultiInputModel
class MultiInputModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.w = nn.Parameter(torch.randn((100, 100), device=device))

    # 前向传播方法，接受一个包含两个张量的元组作为输入，返回一个张量作为输出
    def forward(self, xs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        assert len(xs) == 2, f"Expects 2 args but got {len(xs)}"
        x, y = xs
        z = x + y
        z = z @ self.w
        return nn.functional.relu(z)


# 定义一个测试类 TestCheckpoint，继承自 TestCase
class TestCheckpoint(TestCase):
    # 计算给定张量输出的计算图中函数的数量
    def _get_graph_size(self, out: torch.Tensor) -> int:
        # 使用双端队列来追踪计算图中的函数
        q = deque([out.grad_fn])
        num_functions = 0
        while len(q):
            fn = q.pop()
            num_functions += 1
            # 遍历当前函数的下一级函数
            for next_fn, _ in fn.next_functions:
                if next_fn:
                    q.append(next_fn)

        return num_functions

    # 仅测试张量的情况下
    def _test_tensor_only(
        self,
        net: nn.Module,
        x: torch.Tensor,
    ) -> None:
        # 克隆输入张量，分别为两个测试准备输入
        x1 = x.clone()
        x2 = x.clone()
        # 设置张量需要梯度信息
        x1.requires_grad = True
        x2.requires_grad = True

        # 克隆网络，为两种不同的测试准备网络
        net1 = net
        net2 = deepcopy(net)

        # 无检查点的情况下进行计算和内存变化的检测
        with MemoryDelta(x.device) as mem1:
            # 计算 net1 的输出并计算其梯度
            loss1 = net1(x1).sum()
        # 获取计算图大小
        graph_size1 = self._get_graph_size(loss1)
        loss1.backward()

        # 使用检查点的情况下进行计算和内存变化的检测
        checkpoint(net2.seq)
        with MemoryDelta(x.device) as mem2:
            # 计算 net2 的输出并计算其梯度
            loss2 = net2(x2).sum()
        loss2.backward()

        # 如果张量在 CUDA 上，确保使用检查点时内存变化小于不使用检查点时的内存变化
        if x.is_cuda:
            self.assertTrue(mem2.delta() < mem1.delta())

        # 检查两个网络的参数梯度是否相等
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            self.assertEqual(p1.grad, p2.grad)

    # 在 CPU 上测试张量的情况
    def test_tensor_only_cpu(self):
        x = torch.randn(20, 100)
        net = ToyModel()
        self._test_tensor_only(net, x)

    # 在 GPU 上测试张量的情况
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_tensor_only_gpu(self):
        x = torch.randn(20, 100, device="cuda:0")
        net = ToyModel().to("cuda:0")
        self._test_tensor_only(net, x)

    # 测试随机张量在 CPU 上的情况
    def test_random_cpu(self):
        # 创建随机张量并设置需要梯度信息
        x1 = torch.randn(20, 100, requires_grad=True)
        x2 = x1.clone()

        # 创建随机模型并进行深拷贝
        net1 = RandomModel()
        net2 = deepcopy(net1)

        # 备份 CPU 的随机数生成状态，计算 net1 的输出并计算其梯度
        cpu_rng_state = torch.get_rng_state()
        net1(x1).sum().backward()
        # 恢复 CPU 的随机数生成状态，使用检查点计算 net2 的输出并计算其梯度
        torch.set_rng_state(cpu_rng_state)
        checkpoint(net2)(x2).sum().backward()

        # 检查两个网络的参数梯度是否相等
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            self.assertEqual(p1.grad, p2.grad)

    # 测试多个参数的情况
    def test_multi_args(self):
        """
        测试具有多个输出参数的模块和因此具有多个反向函数输入参数的检查点。
        """
        # 创建 CPU 设备
        device = torch.device("cpu")
        # 创建包含多输出模型和多输入模型的顺序网络
        net1 = nn.Sequential(
            MultiOutputModel(device),
            MultiInputModel(device),
            MultiOutputModel(device),
            MultiInputModel(device),
        )
        # 深拷贝网络 net1 为 net2
        net2 = deepcopy(net1)
        # 在 net2 的特定位置设置检查点
        checkpoint(net2[0])
        checkpoint(net2[2])
        # 创建随机张量并设置需要梯度信息
        x1 = torch.randn(20, 100, requires_grad=True)
        x2 = x1.clone()
        # 计算 net1 的输出并计算其梯度
        net1(x1).sum().backward()
        # 计算 net2 的输出并计算其梯度
        net2(x2).sum().backward()
        # 检查两个网络的参数梯度是否相等
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            self.assertEqual(p1.grad, p2.grad)
    # 定义一个测试函数，用于测试在 forward 方法中出现错误时是否清除状态
    def test_clears_state_on_error_in_forward(self):
        # 定义一个自定义的 PyTorch 模型类 MyModel
        class MyModel(torch.nn.Module):
            # 初始化方法，接受一个布尔值参数 raise_in_recomp
            def __init__(self, raise_in_recomp):
                super().__init__()
                # 初始化前向计算次数为 0
                self.fwd_count = 0
                # 设置是否在重计算中抛出异常的标志
                self.raise_in_recomp = raise_in_recomp
                # 定义一个线性层
                self.a = torch.nn.Linear(2, 2)

            # 前向计算方法，接受输入 x
            def forward(self, x):
                # 如果 raise_in_recomp 为 True 并且是第一次前向计算，则抛出异常
                if self.raise_in_recomp and self.fwd_count == 1:
                    raise RuntimeError("foo")
                else:
                    # 如果不是重计算或者不是第一次前向计算，抛出异常
                    if not self.raise_in_recomp:
                        # 在第一次前向计算时抛出异常
                        raise RuntimeError("foo")
                    # 前向计算次数加一
                    self.fwd_count += 1
                    # 返回线性层的输出
                    return self.a(x)

        # 创建一个 MyModel 实例，设置 raise_in_recomp 为 True
        m = MyModel(raise_in_recomp=True)
        # 将模型 m 放入有序字典中，并使用 Sequential 封装
        m_seq = torch.nn.Sequential(OrderedDict({"m": m}))
        # 对模型进行检查点操作
        checkpoint(m_seq.m)
        # 创建一个输入张量
        inp = torch.randn(1, 2)
        # 对模型进行前向计算并求和
        out = m_seq(inp).sum()
        # 断言前向计算中应该抛出 RuntimeError 异常
        # 这里使用 assertRaisesRegex 确保异常消息中包含 "foo"
        with self.assertRaisesRegex(RuntimeError, "foo"):
            out.backward()

        # 检查 _ac_generator 是否被清除
        self.assertEqual(None, checkpoint.state(m)._ac_generator)

        # 创建一个新的 MyModel 实例，设置 raise_in_recomp 为 False
        m = MyModel(raise_in_recomp=False)
        # 对模型进行检查点操作
        checkpoint(m)
        # 创建一个输入张量
        inp = torch.randn(1, 2)
        # 断言第一次前向计算中应该抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            m(inp)

        # 再次检查 _ac_generator 是否被清除
        self.assertEqual(None, checkpoint.state(m)._ac_generator)
# 如果当前脚本作为主程序运行（而非被导入为模块），则执行 run_tests 函数
if __name__ == "__main__":
    run_tests()
```