# `.\pytorch\test\distributed\optim\test_zero_redundancy_optimizer.py`

```py
# Owner(s): ["oncall: distributed"]

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy  # 导入深拷贝模块
import os  # 导入操作系统模块
import sys  # 导入系统模块
import unittest  # 导入单元测试模块
from contextlib import nullcontext  # 导入上下文管理模块中的 nullcontext
from typing import Any, cast, List  # 导入类型提示相关的模块

import numpy as np  # 导入 NumPy 库

import torch  # 导入 PyTorch 库
import torch.distributed as dist  # 导入 PyTorch 分布式训练模块

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)  # 如果分布式训练不可用，输出跳过测试信息
    sys.exit(0)  # 退出程序

from torch.distributed.algorithms.ddp_comm_hooks.ddp_zero_hook import (
    hook_with_zero_step,  # 导入 DDP 零步骤钩子函数
    hook_with_zero_step_interleaved,  # 导入交替的 DDP 零步骤钩子函数
)
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook  # 导入默认的全reduce钩子函数
from torch.distributed.algorithms.join import Join, Joinable, JoinHook  # 导入分布式 join 相关类
from torch.distributed.optim import ZeroRedundancyOptimizer  # 导入零冗余优化器
from torch.distributed.optim.zero_redundancy_optimizer import _broadcast_object  # 导入广播对象函数
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入分布式数据并行类别名 DDP
from torch.optim import AdamW, SGD  # 导入 AdamW 和 SGD 优化器
from torch.testing._internal import common_distributed  # 导入内部的通用分布式测试模块
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入实例化参数化测试函数
    IS_WINDOWS,  # 导入是否为 Windows 系统的标志
    parametrize,  # 导入参数化装饰器
    run_tests,  # 导入运行测试函数
    TEST_WITH_ASAN,  # 导入是否使用 ASAN 的标志
    TEST_WITH_DEV_DBG_ASAN,  # 导入是否使用开发版 ASAN 的标志
)

try:
    import torchvision  # 尝试导入 torchvision 库
    HAS_TORCHVISION = True  # 设置标志，表明成功导入了 torchvision
except ImportError:
    HAS_TORCHVISION = False  # 设置标志，表明未能导入 torchvision


# 在运行 CUDA 且不是 Windows 系统时，使用 NCCL 后端；否则使用 GLOO 后端
def _get_backend_for_tests():
    return (
        dist.Backend.NCCL
        if not IS_WINDOWS and torch.cuda.is_available()
        else dist.Backend.GLOO
    )


BACKEND = _get_backend_for_tests()  # 获取测试使用的后端


@unittest.skipIf(TEST_WITH_ASAN or TEST_WITH_DEV_DBG_ASAN, "CUDA + ASAN does not work.")
class TestZeroRedundancyOptimizer(common_distributed.MultiProcessTestCase):
    def setUp(self):
        super().setUp()  # 调用父类的 setUp 方法
        os.environ["WORLD_SIZE"] = str(self.world_size)  # 设置环境变量 WORLD_SIZE
        self._spawn_processes()  # 启动多进程

    @property
    def device(self):
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )  # 返回设备类型，如果有 CUDA 则返回 CUDA 设备，否则返回 CPU 设备

    @property
    def world_size(self):
        return 1  # 返回世界大小，默认为 1

    def tearDown(self):
        try:
            torch.distributed.destroy_process_group()  # 销毁进程组
        except AssertionError:
            pass
        try:
            os.remove(self.file_name)  # 尝试删除文件名对应的文件
        except OSError:
            pass

    def dist_init(self, rank, world_size=-1, backend=BACKEND):
        if world_size < 1:
            world_size = self.world_size  # 如果未指定世界大小，使用默认值
        store = dist.FileStore(self.file_name, world_size)  # 创建文件存储对象
        return dist.init_process_group(
            backend=backend,
            store=store,
            rank=rank,
            world_size=world_size,
        )  # 初始化进程组


# TODO: skip_but_pass_in_sandcastle_if does not work here.
# 使用unittest模块的装饰器，根据条件跳过测试，条件包括是否启用了ASAN或DEV_DBG_ASAN
@unittest.skipIf(TEST_WITH_ASAN or TEST_WITH_DEV_DBG_ASAN, "CUDA + ASAN does not work.")
# 继承自TestZeroRedundancyOptimizer类的TestZeroRedundancyOptimizerSingleRank类
class TestZeroRedundancyOptimizerSingleRank(TestZeroRedundancyOptimizer):
    
    # 测试状态字典函数的功能
    def test_state_dict(self):
        """Check that ZeroRedundancyOptimizer exposes the expected state dict
        interface, irrespective of the sharding."""
        
        # 初始化分布式环境，使用当前rank
        self.dist_init(self.rank)
        
        # 定义学习率LR1和LR2，动量MOMENTUM
        LR1 = 0.1
        LR2 = 0.01
        MOMENTUM = 0.9
        
        # 设置接收方rank为0，因为world size为1，所以只有一个rank
        RECIPIENT_RANK = 0  # rank 0 is the only rank since the world size is 1
        
        # 创建张量x，设置在指定设备上，并标记需要梯度计算
        x = torch.tensor([1.0], device=self.device, requires_grad=True)
        
        # 创建ZeroRedundancyOptimizer对象o，使用SGD优化器，设置LR1和动量MOMENTUM
        o = ZeroRedundancyOptimizer(
            [x],
            optimizer_class=SGD,
            lr=LR1,
            momentum=MOMENTUM,
        )
        
        # 对张量x进行反向传播
        x.backward()
        
        # 执行一步优化
        o.step()
        
        # 断言张量x的值为0.9，与预期一致
        self.assertEqual(x, torch.tensor([0.9], device=self.device))
        
        # 断言动量缓存的状态与预期一致
        self.assertEqual(
            o.optim.state[x]["momentum_buffer"],
            torch.tensor([1.0], device=self.device),
        )

        # 清零梯度
        o.zero_grad()
        
        # 合并状态字典到接收方rank
        o.consolidate_state_dict(to=RECIPIENT_RANK)
        
        # 获取当前状态字典
        state_dict = o.state_dict()

        # 检查状态字典是否包含符合PyTorch规范的键
        self.assertIn("param_groups", state_dict.keys())
        self.assertIn("state", state_dict.keys())

        # 检查状态字典的参数组中是否包含预期的键和值
        self.assertEqual(state_dict["param_groups"][0]["lr"], 0.1)
        self.assertEqual(state_dict["param_groups"][0]["momentum"], 0.9)
        self.assertFalse(state_dict["param_groups"][0]["nesterov"])
        self.assertEqual(state_dict["param_groups"][0]["weight_decay"], 0.0)
        self.assertEqual(state_dict["param_groups"][0]["dampening"], 0.0)

        # 检查状态字典的状态和`param_groups`属性是否同步
        for k in state_dict["param_groups"][0]:
            if k != "params":
                self.assertEqual(
                    state_dict["param_groups"][0][k],
                    o.param_groups[0][k],
                )

        # 使用LR2重新加载状态字典到对象o
        o = ZeroRedundancyOptimizer([x], optimizer_class=SGD, lr=LR2)
        o.load_state_dict(state_dict)
        
        # 断言动量缓存的状态与预期一致
        self.assertEqual(
            o.optim.state[x]["momentum_buffer"],
            torch.tensor([1.0], device=self.device),
        )

        # 重新加载后，检查`param_groups`属性中的学习率使用LR1而不是LR2
        self.assertEqual(o.param_groups[0]["lr"], LR1)
        
        # 对张量x进行反向传播
        x.backward()
        
        # 执行一步优化
        o.step()
        
        # 断言张量x的值为0.71，与预期一致
        self.assertEqual(x, torch.tensor([0.71], device=self.device))
        
        # 断言动量缓存的状态与预期一致
        self.assertEqual(
            o.optim.state[x]["momentum_buffer"],
            torch.tensor([1.9], device=self.device),
        )

        # 检查`param_groups`中暴露的`params`在正确的设备上
        self.assertEqual(o.param_groups[0]["params"][0].device, x.device)
    def test_lr_scheduler(self):
        """Check that a normal PyTorch ``lr_scheduler`` is usable with
        ZeroRedundancyOptimizer."""
        # 初始化分布式环境，设置当前进程的排名
        self.dist_init(self.rank)
        NUM_ITERS = 5
        LR = 0.01
        # 创建张量 x 和 x2 在指定设备上，允许梯度计算
        x = torch.tensor([1.0], device=self.device, requires_grad=True)
        x2 = torch.tensor([1.0], device=self.device, requires_grad=True)
        # 使用 ZeroRedundancyOptimizer 包装 x 的优化器，使用 SGD 优化算法和学习率 LR
        o = ZeroRedundancyOptimizer([x], optimizer_class=SGD, lr=LR)
        # 创建普通的 SGD 优化器 o2，优化张量 x2，使用相同的学习率 LR
        o2 = torch.optim.SGD([x2], lr=LR)
        # 使用 StepLR 调度器 s，管理 ZeroRedundancyOptimizer o，设置步长为 1
        s = torch.optim.lr_scheduler.StepLR(o, 1)
        # 使用 StepLR 调度器 s2，管理普通 SGD 优化器 o2，设置步长为 1
        s2 = torch.optim.lr_scheduler.StepLR(o2, 1)
        # 迭代 NUM_ITERS 次
        for _ in range(NUM_ITERS):
            # 计算张量 x 的梯度
            x.backward()
            # 清零优化器 o 的梯度
            o.zero_grad()
            # 在优化器 o 上执行优化步骤
            o.step()
            # 在调度器 s 上进行步长调度
            s.step()
            # 计算张量 x2 的梯度
            x2.backward()
            # 清零普通 SGD 优化器 o2 的梯度
            o2.zero_grad()
            # 在普通 SGD 优化器 o2 上执行优化步骤
            o2.step()
            # 在调度器 s2 上进行步长调度
            s2.step()
            # 断言张量 x 与张量 x2 相等
            self.assertEqual(x, x2)

    def test_step_with_kwargs(self):
        """Check that the ``step(**kwargs)`` interface is properly exposed."""
        # 初始化分布式环境，设置当前进程的排名
        self.dist_init(self.rank)
        LR = 0.1

        class SGDWithStepKWArg(torch.optim.SGD):
            def step(self, closure=None, kwarg=None):
                # 调用父类的 step 方法
                super().step()
                # 向 kwarg 参数添加值 5
                kwarg.append(5)

        # 创建张量 x 在指定设备上，允许梯度计算
        kwarg: List[Any] = []
        x = torch.tensor([1.0], device=self.device, requires_grad=True)
        # 使用 ZeroRedundancyOptimizer 包装 x 的优化器，使用自定义的 SGDWithStepKWArg 优化算法和学习率 LR
        o = ZeroRedundancyOptimizer(
            [x],
            optimizer_class=SGDWithStepKWArg,
            lr=LR,
        )
        # 计算张量 x 的梯度
        x.backward()
        # 在 ZeroRedundancyOptimizer o 上执行优化步骤，传递额外的 kwarg 参数
        o.step(0, kwarg=kwarg)
        # 断言 kwarg 的值为 [5]
        self.assertEqual(kwarg, [5])
        # 断言张量 x 的值为 [0.9]（经过一次步骤后的预期值）
        self.assertEqual(x, torch.tensor([0.9], device=self.device))

    def test_step_with_extra_inner_key(self):
        """Check that ZeroRedundancyOptimizer wrapping an optimizer that adds
        extra keys to ``param_groups`` exposes those keys through ZeRO's own
        ``param_groups``."""
        # 初始化分布式环境，设置当前进程的排名
        self.dist_init(self.rank)
        LR = 0.1

        class SGDWithNewKey(torch.optim.SGD):
            # 添加一个新的 key 到 param_groups 的虚拟优化器
            def step(self, closure=None):
                super().step()
                # 设置 param_groups 的第一个组的新 key 的值为 0.1
                self.param_groups[0]["new_key"] = 0.1

        # 创建张量 x 在指定设备上，允许梯度计算
        x = torch.tensor([1.0], device=self.device, requires_grad=True)
        # 使用 ZeroRedundancyOptimizer 包装 x 的优化器，使用自定义的 SGDWithNewKey 优化算法和学习率 LR
        o = ZeroRedundancyOptimizer([x], optimizer_class=SGDWithNewKey, lr=LR)
        # 计算张量 x 的梯度
        x.backward()
        # 在 ZeroRedundancyOptimizer o 上执行优化步骤
        o.step()
        # 断言 ZeRO 的 param_groups 的第一个组的 new_key 的值为 0.1
        self.assertEqual(o.param_groups[0]["new_key"], 0.1)
        # 断言张量 x 的值为 [0.9]（经过一次步骤后的预期值）
        self.assertEqual(x, torch.tensor([0.9], device=self.device))
    # 定义测试方法，验证没有闭包的情况下 `step()` 方法的预期处理方式
    def test_step_without_closure(self):
        """Check that the ``step()`` method (without closure) is handled as
        expected."""
        # 初始化分布式环境，设定当前进程的 rank
        self.dist_init(self.rank)
        # 设置学习率 LR
        LR = 0.1

        # 定义继承自 torch.optim.SGD 的 SGDWithoutClosure 类，重写了 `step()` 方法
        class SGDWithoutClosure(torch.optim.SGD):
            def step(self):
                return super().step()

        # 创建一个张量 x，设备为 self.device，并要求计算梯度
        x = torch.tensor([1.0], device=self.device, requires_grad=True)
        # 初始化一个 ZeroRedundancyOptimizer 对象 o，用于优化参数 x
        o = ZeroRedundancyOptimizer(
            [x],
            optimizer_class=SGDWithoutClosure,  # 使用自定义的优化器类
            lr=LR,
        )
        # 对张量 x 进行反向传播
        x.backward()
        # 执行优化器 o 的 `step()` 方法
        o.step()
        # 断言张量 x 的值为预期值 [0.9]
        self.assertEqual(x, torch.tensor([0.9], device=self.device))

    # 测试 `zero_grad` 方法是否被正确处理
    def test_zero_grad(self):
        """Check that the ``zero_grad`` method is properly handled."""
        # 初始化分布式环境，设定当前进程的 rank
        self.dist_init(self.rank)
        # 设置学习率 LR
        LR = 0.01
        # 创建一个随机张量 x
        x = torch.rand(1)
        # 创建一个线性层 m，输入维度为 1，输出维度为 1
        m = torch.nn.Linear(1, 1)
        # 初始化一个 ZeroRedundancyOptimizer 对象 o，用于优化 m 的参数
        o = ZeroRedundancyOptimizer(m.parameters(), optimizer_class=SGD, lr=LR)
        # 对张量 x 进行线性层 m 的前向传播
        y = m(x)
        # 对 y 进行反向传播，传入梯度 x
        y.backward(x)
        # 断言 m 的权重参数的梯度不全为零
        self.assertNotEqual(m.weight.grad, torch.zeros_like(m.weight))
        # 断言 m 的偏置参数的梯度不全为零
        self.assertNotEqual(m.bias.grad, torch.zeros_like(m.bias))
        # 执行优化器 o 的 `zero_grad()` 方法，清空所有参数的梯度
        o.zero_grad()
        # 断言 m 的权重参数的梯度为 None
        self.assertIsNone(m.weight.grad)
        # 断言 m 的偏置参数的梯度为 None
        self.assertIsNone(m.bias.grad)
    def test_constructor(self):
        """Check the robustness of the ZeroRedundancyOptimizer constructor by
        passing different values for the ``params`` argument."""
        # 初始化分布
        self.dist_init(self.rank)
        # 设置学习率
        LR = 0.01
        # 创建一个包含三个线性层的神经网络模型
        m = torch.nn.Sequential(
            torch.nn.Linear(5, 10),
            torch.nn.Linear(10, 10),
            torch.nn.Linear(10, 10),
        )
        # 测试不同的构造器输入形式：(输入, 期望错误类型)
        ctor_inputs = [
            ([], ValueError),  # 空参数列表
            (torch.randn(1), TypeError),  # 非可迭代对象：`torch.Tensor`
            (1.2, TypeError),  # 非可迭代对象：`float`
            (
                [
                    {"params": [l.weight for l in m]},
                    {"params": [l.bias for l in m]},
                ],
                None,
            ),  # 字典组成的可迭代对象
            (
                list(m.parameters()) + [42],
                TypeError,
            ),  # 包含无效类型的可迭代对象
            (m.parameters(), None),  # `params` 是生成器
            (list(m.parameters()), None),  # `params` 是列表
        ]
        # 遍历构造器输入进行测试
        for ctor_input, error in ctor_inputs:
            # 如果有错误，验证是否抛出指定类型的异常；否则使用空环境
            context = self.assertRaises(error) if error else nullcontext()
            with context:
                # 使用构造器输入创建 ZeroRedundancyOptimizer 对象
                ZeroRedundancyOptimizer(
                    ctor_input,
                    optimizer_class=SGD,
                    lr=LR,
                )

        # 更彻底地测试使用多个参数组进行构造
        WD = 0.01
        BETAS = (0.9, 0.999)
        EPS = 1e-8
        params = [
            {"params": [l.weight for l in m], "weight_decay": 0.0},
            {"params": [l.bias for l in m], "weight_decay": WD},
        ]
        # 使用多个参数组构造 ZeroRedundancyOptimizer 对象
        o = ZeroRedundancyOptimizer(
            params,
            optimizer_class=AdamW,
            lr=LR,
            betas=BETAS,
            eps=EPS,
        )
        # 断言期望的参数组数量为 2
        assert (
            len(o.param_groups) == 2
        ), f"Expected 2 ZeRO param groups, but got {len(o.param_groups)}"
        # 断言期望的本地优化器参数组数量为 2
        assert len(o.optim.param_groups) == 2, (
            "Expected 2 local optimizer param groups, but got "
            f"{len(o.optim.param_groups)}"
        )
    def test_same_dense_param_type(self):
        """
        检查 ZeroRedundancyOptimizer 是否在输入参数包含稀疏张量或不同的密集类型时引发异常。

        注意：一旦添加对稀疏参数和不同参数类型的支持，应该删除此测试。
        """
        # 初始化分布式环境，使用当前的排名（rank）
        self.dist_init(self.rank)
        # 设置学习率 LR
        LR = 0.01
        # 定义输入数据列表，包含不同类型的张量作为测试案例
        inputs = [
            [torch.sparse_coo_tensor(size=(2, 3))],  # 包含一个稀疏张量的列表
            [torch.FloatTensor(1), torch.DoubleTensor(1)],  # 包含不同类型的密集张量的列表
            [
                torch.FloatTensor(1),
                torch.FloatTensor(1),
                torch.sparse_coo_tensor(size=(2, 3)),  # 同时包含稀疏张量的列表
            ],
        ]
        # 对每个输入进行测试
        for input in inputs:
            # 断言期望引发 ValueError 异常
            with self.assertRaises(ValueError):
                # 创建 ZeroRedundancyOptimizer 实例，并传入当前的输入作为参数
                ZeroRedundancyOptimizer(input, optimizer_class=SGD, lr=LR)
class TestZeroRedundancyOptimizerDistributed(TestZeroRedundancyOptimizer):
    @property
    def device(self):
        # 返回当前设备：如果CUDA可用，则返回以self.rank为设备编号的torch.device对象，否则返回CPU设备
        return (
            torch.device(self.rank)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    @property
    def world_size(self):
        # 返回当前的分布式训练的全局大小，取CUDA设备数量的最大值和2之间的最小值，范围为2到4
        return min(4, max(2, torch.cuda.device_count()))

    @property
    def context(self):
        # 返回一个上下文管理器：如果CUDA不可用，则返回一个nullcontext对象，否则返回以self.rank为设备编号的CUDA设备对象
        return (
            nullcontext()
            if not torch.cuda.is_available()
            else torch.cuda.device(self.rank)
        )

    def _check_same_model_params(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        message: str = "",
    ) -> None:
        # 检查两个模型的参数是否一致
        for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
            torch.testing.assert_close(
                p_a,
                p_b,
                atol=1e-3,
                rtol=1e-5,
                msg=f"Model parameters differ:\n{p_a} {p_b}\n" + message,
            )
        # 检查两个模型的缓冲区是否一致
        for b_a, b_b in zip(model_a.buffers(), model_b.buffers()):
            torch.testing.assert_close(
                b_a,
                b_b,
                msg=f"Model buffers differ:\n{b_a} {b_b}\n" + message,
            )

    @common_distributed.skip_if_no_gpu
    @common_distributed.skip_if_rocm
    def test_step(self):
        """Check that ZeroRedundancyOptimizer properly exposes the ``step()``
        interface."""
        self.dist_init(self.rank, world_size=self.world_size)  # 初始化分布式设置

        LR = 0.01

        with self.context:
            x = torch.tensor([float(self.rank + 1)], device=self.device)  # 创建张量x，放在当前设备上
            m = torch.nn.Linear(1, 1)  # 创建线性模型m
            m.weight.data = torch.tensor([[1.0]])  # 设置模型权重
            m.bias.data = torch.tensor([2.0])  # 设置模型偏置
            m = m.to(self.device)  # 将模型移动到当前设备
            m_zero = copy.deepcopy(m).to(self.device)  # 深拷贝m并移动到当前设备，命名为m_zero

            o = SGD(m.parameters(), lr=LR)  # 创建SGD优化器o，用于普通模型m的参数优化
            o_zero = ZeroRedundancyOptimizer(  # 创建零冗余优化器o_zero，用于零冗余模型m_zero的参数优化
                m_zero.parameters(),
                optimizer_class=SGD,
                lr=LR,
            )

            y = m(x)  # 使用模型m计算输出y
            y.backward(x)  # 计算y关于x的梯度
            y_zero = m_zero(x)  # 使用零冗余模型m_zero计算输出y_zero
            y_zero.backward(x)  # 计算y_zero关于x的梯度

            for p in m.parameters():
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)  # 在分布式环境中对模型m的梯度进行全局求和
                p.grad.data /= self.world_size  # 平均化梯度
            o.step()  # 执行模型m的优化步骤

            for p in m_zero.parameters():
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)  # 在分布式环境中对零冗余模型m_zero的梯度进行全局求和
                p.grad.data /= self.world_size  # 平均化梯度
            o_zero.step()  # 执行零冗余模型m_zero的优化步骤

            self.assertEqual(m.weight, m_zero.weight)  # 断言模型m和模型m_zero的权重是否相等
            self.assertEqual(m.bias, m_zero.bias)  # 断言模型m和模型m_zero的偏置是否相等

    @common_distributed.skip_if_no_gpu
    @common_distributed.skip_if_rocm
    def test_step_with_closure(self):
        """Check that ZeroRedundancyOptimizer properly exposes the
        ``step(closure)`` interface."""
        # 初始化分布式环境
        self.dist_init(self.rank, world_size=self.world_size)

        # 进入上下文管理器
        with self.context:
            # 对于每个 bucket_view 的设置（False 或 True）
            for bucket_view in [False, True]:
                # 设置变量 x_val 为当前进程的 rank 加 1
                x_val = self.rank + 1
                # 设置权重为 1.0
                weight = 1.0
                # 设置偏置为 2.0
                bias = 2.0
                # 设置误差为 1.0
                error = 1.0
                # 构建目标张量，包含 x_val * weight + bias + error
                target = torch.tensor(
                    [x_val * weight + bias + error],
                    device=self.device,
                )
                # 使用 L1 损失函数
                loss_fn = torch.nn.L1Loss()

                # 构建张量 x，包含当前进程的 x_val
                x = torch.tensor([float(x_val)], device=self.device)
                # 构建线性模型 m，输入维度 1，输出维度 1
                m = torch.nn.Linear(1, 1)
                # 设置 m 的权重为 [[weight]]
                m.weight.data = torch.tensor([[weight]])
                # 设置 m 的偏置为 [bias]
                m.bias.data = torch.tensor([bias])
                # 将模型 m 移动到指定设备上
                m.to(self.device)

                # 初始化 ZeroRedundancyOptimizer
                o = ZeroRedundancyOptimizer(
                    m.parameters(),
                    optimizer_class=SGD,
                    parameters_as_bucket_view=bucket_view,
                    lr=0.1,
                )

                # 计算模型输出 y
                y = m(x)
                # 计算损失函数梯度
                y.backward(x)
                # 对模型参数进行分布式求和归约，并除以进程数
                for p in m.parameters():
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    p.grad.data /= self.world_size

                # 定义闭包函数 closure
                def closure():
                    # 梯度清零
                    o.zero_grad()
                    # 计算模型输出
                    output = m(x)
                    # 计算损失函数
                    loss = loss_fn(output, target)
                    # 反向传播计算梯度
                    loss.backward()
                    return loss

                # 调用优化器的 step 方法执行一步优化，并返回损失值
                loss = o.step(closure=closure)

                # 断言损失值为预期的 error
                self.assertEqual(loss, torch.tensor(error))
                # 断言模型权重更新正确
                self.assertEqual(m.weight, torch.tensor([[1.1]]))
                # 断言模型偏置更新正确
                self.assertEqual(m.bias, torch.tensor([2.1]))

    @common_distributed.skip_if_no_gpu
    def test_lr_scheduler(self):
        """Check that a normal PyTorch ``lr_scheduler`` is usable with
        ZeroRedundancyOptimizer."""
        # 初始化分布式环境
        self.dist_init(self.rank)
        # 构建张量 x 在指定设备上，需要梯度
        x = torch.tensor([1.0], device=self.device, requires_grad=True)
        # 构建另一个张量 x2 在指定设备上，需要梯度
        x2 = torch.tensor([1.0], device=self.device, requires_grad=True)
        # 初始化 ZeroRedundancyOptimizer 对象 o，优化张量 x
        o = ZeroRedundancyOptimizer([x], optimizer_class=SGD, lr=0.01)
        # 初始化标准 SGD 优化器对象 o2，优化张量 x2
        o2 = torch.optim.SGD([x2], lr=0.01)
        # 初始化学习率调度器 s，使用 o 的步长调度器
        s = torch.optim.lr_scheduler.StepLR(o, 1)
        # 初始化学习率调度器 s2，使用 o2 的步长调度器
        s2 = torch.optim.lr_scheduler.StepLR(o2, 1)
        # 进行 5 次迭代
        for _ in range(5):
            # 张量 x 反向传播
            x.backward()
            # o 清零梯度
            o.zero_grad()
            # o 执行一步优化
            o.step()
            # o 的学习率调度器执行一步调度
            s.step()
            # 张量 x2 反向传播
            x2.backward()
            # o2 清零梯度
            o2.zero_grad()
            # o2 执行一步优化
            o2.step()
            # o2 的学习率调度器执行一步调度
            s2.step()
            # 断言张量 x 与 x2 相等
            self.assertEqual(x, x2)
    # 定义一个测试方法，用于检查 ZeroRedundancyOptimizer 在构造时的分片参数设置。
    def test_sharding(self):
        """
        Check ZeroRedundancyOptimizer's parameter sharding at construction
        time.

        NOTE: The correctness of this test depends on the ZeRO implementation
        using the sorted-greedy partitioning algorithm. For details, see
        ``ZeroRedundancyOptimizer._partition_parameters()`` in
        zero_redundancy_optimizer.py.
        """
        # 在分布式环境中初始化当前进程的分布式配置
        self.dist_init(self.rank)
        # 设置学习率 LR
        LR = 0.01
        # 定义参数尺寸列表
        sizes = [9, 7, 5, 3]
        # 初始化参数列表
        params = []
        # 将每个尺寸重复 self.world_size 次，然后生成对应的随机数张量并加入 params 列表
        for size in sizes * self.world_size:
            params.append(torch.rand(size, 1))
        # 使用 ZeroRedundancyOptimizer 初始化优化器 o，传入参数列表 params 和 SGD 优化器类以及学习率 LR
        o = ZeroRedundancyOptimizer(params, optimizer_class=SGD, lr=LR)
        # 断言：验证优化器 o 的参数组中所有参数的元素数量之和是否等于 sizes 列表中所有元素的和
        self.assertEqual(
            sum(x.numel() for x in o.optim.param_groups[0]["params"]),
            sum(sizes),
        )
    def test_add_param_group(self):
        """Check that ZeroRedundancyOptimizer properly handles adding a new
        parameter group a posteriori and that all ranks get a shard of the
        contained parameters.

        NOTE: The correctness of this test depends on the ZeRO implementation
        using the sorted-greedy partitioning algorithm. For details, see
        ``ZeroRedundancyOptimizer._partition_parameters()`` in
        zero_redundancy_optimizer.py.
        """
        # 初始化分布式环境，使用当前进程的 rank
        self.dist_init(self.rank)
        LR = 0.01

        # Test with all parameters trainable to begin with
        def all_trainable():
            params = []
            sizes = [9, 7, 5, 3]
            sizes_world = sizes * self.world_size
            for size in sizes_world[:-1]:
                # 创建具有随机值的张量，并添加到参数列表中
                params.append(torch.rand(size, 1))

            # 确保所有参数都是可训练的，以便它们被考虑在基于大小的参数分区中
            for p in params:
                p.requires_grad = True

            # 创建 ZeroRedundancyOptimizer 对象，使用 SGD 优化器和指定的学习率
            o = ZeroRedundancyOptimizer(params, optimizer_class=SGD, lr=LR)
            # 断言初始参数组的数量为 1
            self.assertEqual(len(o.param_groups), 1)
            # 添加新的参数组到优化器中
            o.add_param_group({"params": [torch.rand(3, 1)]})
            # 验证新的参数组被正确地添加到了分区中，使得所有分区具有相同的元素
            self.assertEqual(len(o.param_groups), 2)
            # 验证所有参数组中参数的总数量与预期的总大小相等
            self.assertEqual(
                sum(x.numel() for g in o.optim.param_groups for x in g["params"]),
                sum(sizes),
            )
            # 断言优化器当前的参数组数量为 2
            self.assertEqual(len(o.optim.param_groups), 2)

        # Test a pathological config with a first big non-trainable param
        def some_trainable():
            params = []
            # 创建具有随机值的张量，并添加到参数列表中
            for size in [100, 3, 5, 2, 6, 4]:
                params.append(torch.rand(size, 1))

            # 确保除第一个参数外，所有参数都是可训练的，以便它们被考虑在基于大小的参数分区中
            for p in params[1:]:
                p.requires_grad = True

            # 创建 ZeroRedundancyOptimizer 对象，使用 SGD 优化器和指定的学习率
            o = ZeroRedundancyOptimizer(params, optimizer_class=SGD, lr=LR)
            # 断言初始参数组的数量为 1
            self.assertEqual(len(o.param_groups), 1)
            # 添加新的参数组到优化器中
            o.add_param_group({"params": [torch.rand(3, 1)]})
            # 断言新的参数组被正确地添加到了分区中
            self.assertEqual(len(o.param_groups), 2)
            # 断言优化器当前的参数组数量为 2
            self.assertEqual(len(o.optim.param_groups), 2)

        # 调用函数以进行测试
        all_trainable()
        some_trainable()

    @common_distributed.skip_if_no_gpu
    def test_multiple_param_groups(self):
        """
        Check parity between constructing ZeRO with multiple parameter groups
        upfront versus adding parameter groups to ZeRO after construction
        versus a non-sharded optimizer.
        """
        # 初始化分布式设置，使用给定的排名（rank）
        self.dist_init(self.rank)
        # 定义批量大小和迭代次数
        BATCH_SIZE, NUM_ITERS = 8, 3
        # 定义输入、隐藏和输出维度
        INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 5, 10, 5
        # 定义权重衰减和学习率
        WD, LR = 0.01, 0.01

        # 创建模型1，2，3，均为深度复制的模型1
        model1 = torch.nn.Sequential(
            torch.nn.Linear(INPUT_DIM, HIDDEN_DIM),
            torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )
        model2 = copy.deepcopy(model1)
        model3 = copy.deepcopy(model1)

        # 将模型1，2，3移至指定设备（GPU或CPU）
        model1 = model1.to(self.device)
        model2 = model2.to(self.device)
        model3 = model3.to(self.device)

        # 生成一组输入数据，每个输入数据为随机张量，移至指定设备
        inputs = [
            torch.randn(BATCH_SIZE, INPUT_DIM).to(self.device) for _ in range(NUM_ITERS)
        ]

        # 使用两个参数组一起构建`optim1`
        optim1 = ZeroRedundancyOptimizer(
            [
                {"params": [l.weight for l in model1], "weight_decay": 0.0},
                {"params": [l.bias for l in model1], "weight_decay": WD},
            ],
            optimizer_class=AdamW,
            lr=LR,
        )

        # 构建`optim2`，先添加第二个参数组
        optim2 = ZeroRedundancyOptimizer(
            [l.weight for l in model2],
            optimizer_class=AdamW,
            lr=LR,
            weight_decay=0.0,
        )
        optim2.add_param_group({"params": [l.bias for l in model2], "weight_decay": WD})

        # 构建`optim3`作为非分片优化器
        optim3 = AdamW(
            [
                {"params": [l.weight for l in model3], "weight_decay": 0.0},
                {"params": [l.bias for l in model3], "weight_decay": WD},
            ],
            lr=LR,
        )

        # 在几次迭代中检查各优化器的效果对比
        for input in inputs:
            for model, optim in (
                (model1, optim1),
                (model2, optim2),
                (model3, optim3),
            ):
                optim.zero_grad()
                out = model(input)
                loss = out.sum()
                loss.backward()
                optim.step()

            # 检查各模型的层参数在优化后的一致性
            for layer1, layer2, layer3 in zip(model1, model2, model3):
                torch.testing.assert_close(layer1.weight, layer2.weight)
                torch.testing.assert_close(layer1.weight, layer3.weight)
                torch.testing.assert_close(layer1.bias, layer2.bias)
                torch.testing.assert_close(layer1.bias, layer3.bias)

    @common_distributed.skip_if_no_gpu
    @common_distributed.skip_if_rocm
    def test_collect_shards(self):
        """检查状态合并机制和由ZeroRedundancyOptimizer暴露的状态字典。"""
        # 初始化分布式环境，设置当前进程的等级
        self.dist_init(self.rank)
        LR = 1e-3  # 学习率设定为0.001
        MOMENTUM = 0.99  # 动量设定为0.99
        BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 3, 20, 10, 5  # 定义批量大小和输入、隐藏、输出维度
        REFERENCE_RANK = 0  # 参考等级设定为0
        target = torch.rand((BATCH_SIZE, OUTPUT_DIM), device=self.device)  # 随机生成目标张量
        inputs = torch.rand((BATCH_SIZE, INPUT_DIM), device=self.device)  # 随机生成输入张量
        model = torch.nn.Sequential(
            torch.nn.Linear(INPUT_DIM, HIDDEN_DIM),  # 输入到隐藏层的线性变换
            torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM),  # 隐藏层到输出层的线性变换
        ).to(self.device)  # 将模型移动到指定设备上
        loss_fn = torch.nn.L1Loss()  # 定义L1损失函数
        loss_fn.to(self.device)  # 将损失函数移动到指定设备上
        optimizer = ZeroRedundancyOptimizer(
            model.parameters(),  # 将模型参数传递给优化器
            optimizer_class=SGD,  # 优化器选择SGD类
            lr=LR,  # 学习率设定为LR
            momentum=MOMENTUM,  # 确保存在需要分片的状态
        )

        def closure():
            optimizer.zero_grad()  # 清空梯度
            output = model(inputs)  # 模型前向传播
            loss = loss_fn(output, target)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            return loss

        # 运行一个虚拟步骤，以便优化器状态字典存在
        _ = optimizer.step(closure=closure)

        # 获取参考等级上的优化器状态
        optimizer.consolidate_state_dict(to=REFERENCE_RANK)
        if self.rank == REFERENCE_RANK:
            # 检查状态是否具有正确的大小
            optimizer_state_dict = optimizer.state_dict()
            self.assertEqual(
                len(optimizer_state_dict["state"]),  # 断言状态字典中的状态数量
                len(list(model.parameters())),  # 断言模型参数列表的长度
            )
        else:
            optimizer_state_dict = {}  # 在非参考等级上将优化器状态字典置为空字典

        # 在所有等级上加载优化器状态，不会有任何异常
        optimizer_state_dict = _broadcast_object(
            optimizer_state_dict,  # 待广播的对象为优化器状态字典
            src_rank=REFERENCE_RANK,  # 源等级为参考等级
            group=dist.group.WORLD,  # 分布式组为WORLD
            device=self.device,  # 设备为指定设备
        )
        optimizer.load_state_dict(optimizer_state_dict)  # 加载优化器状态字典

    @common_distributed.skip_if_no_gpu
    @parametrize(
        "optimizer_class_str",
        ["Adam", "AdamW", "SGD"],
        # 使用字符串以满足内部测试名称解析器的要求
    )
    @parametrize(
        "maximize",
        [False, True],
    )
    def test_local_optimizer_parity(
        self,
        optimizer_class_str: str,
        maximize: bool,
    @common_distributed.requires_nccl()
    @common_distributed.skip_if_no_gpu
    def test_zero_join_gpu(self):
        """检查ZeRO连接钩子是否允许在GPU上处理不均匀的输入。"""
        self._test_zero_join(self.device)

    @common_distributed.requires_gloo()
    def test_zero_join_cpu(self):
        """检查ZeRO连接钩子是否允许在CPU上处理不均匀的输入。"""
        self._test_zero_join(torch.device("cpu"))

    @common_distributed.skip_if_lt_x_gpu(4)
    @parametrize(
        "parameters_as_bucket_view",
        [False, True],
    )
    def test_zero_model_parallel(
        self,
        parameters_as_bucket_view: bool,
    ):
        """Check that ZeRO works with model parallelism where the model's
        layers are assigned to different devices."""
        # 如果当前进程的 rank 大于等于 2，则直接返回，不执行测试
        if self.rank >= 2:
            return
        # 初始化分布式环境，设置当前进程的 rank 和总的 world_size 为 2
        self.dist_init(self.rank, world_size=2)
        # 调用 _test_zero_model_parallel 方法，执行 ZeRO 模型并行测试
        self._test_zero_model_parallel(parameters_as_bucket_view)

    def _test_ddp_zero_overlap(
        self,
        device,
        hook_constructor,
        gradient_as_bucket_view,
        static_graph,
        **kwargs,
    # NOTE: The test is skipped if using Windows since functional optimizers
    # are not currently supported.
    @common_distributed.skip_if_win32()
    @common_distributed.requires_nccl()
    @common_distributed.skip_if_no_gpu
    @common_distributed.skip_if_rocm
    @parametrize(
        "use_gpu",
        [True],
        # Add `False` once the Gloo sync issue causing hangs is fixed
        # See: https://github.com/pytorch/pytorch/issues/62300
    )
    @parametrize(
        "use_interleaved_hook",
        [False, True],
    )
    @parametrize(
        "gradient_as_bucket_view",
        [False, True],
    )
    @parametrize(
        "static_graph",
        [False, True],
    )
    @parametrize(
        "shard_buckets",
        [False, True],
    )
    def test_ddp_zero_overlap(
        self,
        use_gpu: bool,
        use_interleaved_hook: bool,
        gradient_as_bucket_view: bool,
        static_graph: bool,
        shard_buckets: bool,
    ):
        """
        Check that overlapping DDP with ZeRO using the given method determined
        by ``hook_constructor`` and ``shard_buckets`` and using the given ZeRO
        and DDP arguments achieves parity with DDP using a local optimizer.
        """
        # 根据 use_gpu 决定设备是 GPU 还是 CPU
        device = torch.device(self.rank) if use_gpu else torch.device("cpu")
        # 获取用于测试的后端
        backend = _get_backend_for_tests()
        # 初始化分布式环境，设置当前进程的 rank 和总的 world_size
        self.dist_init(self.rank, self.world_size, backend)
        # 根据 use_interleaved_hook 选择使用不同的 hook 构造函数
        hook_constructor = (
            hook_with_zero_step
            if not use_interleaved_hook
            else hook_with_zero_step_interleaved
        )

        # 调用 _test_ddp_zero_overlap 方法执行 DDP 和 ZeRO 重叠测试
        self._test_ddp_zero_overlap(
            device,
            hook_constructor,
            gradient_as_bucket_view,
            static_graph,
            shard_buckets=shard_buckets,
        )
# 实例化参数化测试类 TestZeroRedundancyOptimizerSingleRank，以准备执行相关的测试
instantiate_parametrized_tests(TestZeroRedundancyOptimizerSingleRank)

# 实例化参数化测试类 TestZeroRedundancyOptimizerDistributed，以准备执行相关的测试
instantiate_parametrized_tests(TestZeroRedundancyOptimizerDistributed)

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    # 警告：在这里不应该使用 unittest 模块运行测试，否则测试可能不会被正确注册
    run_tests()
```