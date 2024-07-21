# `.\pytorch\test\distributed\_tensor\debug\test_comm_mode.py`

```
# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist

import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor

from torch.distributed._tensor.debug.comm_mode import CommDebugMode
from torch.distributed._tensor.placement_types import Shard
from torch.testing._internal.common_distributed import requires_nccl
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule
from torch.testing._internal.distributed.fake_pg import FakeStore

c10d_functional = torch.ops.c10d_functional
c10d_ops = torch.ops.c10d


class TestCommMode(TestCase):
    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    def setUp(self):
        super().setUp()
        # 设置测试所需的分布式环境
        self.world_size = 2
        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=1, world_size=self.world_size, store=store
        )
        # 确定设备类型是 CUDA 还是 CPU
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        # 获取默认的进程组
        self.world_pg = dist.distributed_c10d._get_default_group()

    def checksAssert(self, comm_mode, key, expected_value, expected_total_value):
        # 检查通信模式的断言
        comm_counts = comm_mode.get_comm_counts()
        self.assertEqual(comm_mode.get_total_counts(), expected_total_value)
        self.assertEqual(comm_counts[key], expected_value)

        return

    def test_comm_mode(self):
        world_pg = self.world_pg

        class WrapperModel(nn.Module):
            def __init__(self, device):
                super().__init__()
                # 初始化包装模型
                self.model = MLPModule(device=device)

            def forward(self, x):
                # 在模型前向传播过程中使用分布式函数进行通信
                x = funcol.all_gather_tensor(x, 0, world_pg)
                x = funcol.reduce_scatter_tensor(x, "sum", 0, world_pg)
                out = self.model(x)
                return funcol.all_reduce(out, "sum", world_pg)

        model = WrapperModel(self.device_type)

        # 启动通信调试模式
        comm_mode = CommDebugMode()
        with comm_mode:
            model(torch.randn(20, 10, device=self.device_type))

        # 检查通信计数
        comm_counts = comm_mode.get_comm_counts()
        self.assertEqual(comm_mode.get_total_counts(), 3)
        self.assertEqual(comm_counts[c10d_functional.all_reduce], 1)
        self.assertEqual(comm_counts[c10d_functional.all_gather_into_tensor], 1)
        self.assertEqual(comm_counts[c10d_functional.reduce_scatter_tensor], 1)
    def test_comm_mode_coalesced(self):
        # 获取测试中的世界进程组对象
        world_pg = self.world_pg

        class WrapperModelCoalesced(nn.Module):
            def __init__(self, device):
                super().__init__()
                # 初始化包装模型，使用 MLP 模块
                self.model = MLPModule(device=device)

            def forward(self, x):
                # 对输入张量进行全局收集操作，使用进程组 world_pg，返回结果张量
                x = funcol.all_gather_tensor(x, 0, world_pg)
                # 对输入张量进行归约散射操作，求和，使用进程组 world_pg，返回结果张量
                x = funcol.reduce_scatter_tensor(x, "sum", 0, world_pg)
                # 将处理后的张量输入到模型中进行前向传播
                out = self.model(x)
                # 对输出张量进行全局归约操作，使用进程组 world_pg，返回归约后的结果张量列表
                return funcol.all_reduce_coalesced([out], "sum", world_pg)

        # 创建 WrapperModelCoalesced 类的实例，传入设备类型参数
        model = WrapperModelCoalesced(self.device_type)

        # 创建通信调试模式对象
        comm_mode = CommDebugMode()
        # 进入通信调试模式
        with comm_mode:
            # 使用模型进行前向传播，传入随机张量作为输入
            model(torch.randn(20, 10, device=self.device_type))

        # 获取通信操作计数
        comm_counts = comm_mode.get_comm_counts()
        # 断言总通信次数为 3
        self.assertEqual(comm_mode.get_total_counts(), 3)
        # 断言 all_reduce_coalesced 操作计数为 1
        self.assertEqual(comm_counts[c10d_functional.all_reduce_coalesced], 1)
        # 断言 all_gather_into_tensor 操作计数为 1
        self.assertEqual(comm_counts[c10d_functional.all_gather_into_tensor], 1)
        # 断言 reduce_scatter_tensor 操作计数为 1
        self.assertEqual(comm_counts[c10d_functional.reduce_scatter_tensor], 1)

    def test_comm_mode_with_dtensor(self):
        # 获取测试中的世界进程组对象
        world_pg = self.world_pg
        # 创建设备网格对象，传入设备类型和世界大小列表
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        def f(x, y):
            # 执行矩阵乘法操作，返回结果张量
            return torch.mm(x, y)

        # 创建通信调试模式对象
        comm_mode = CommDebugMode()
        # 创建两个随机张量 x 和 y，均设置 requires_grad 为 True
        x = torch.randn(4, 8, requires_grad=True)
        y = torch.randn(4, 32, requires_grad=True)
        # 将本地张量 x 转换为分布式张量 x_dtensor，传入设备网格、分片列表
        x_dtensor = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
        # 将本地张量 y 转换为分布式张量 y_dtensor，传入设备网格、分片列表
        y_dtensor = DTensor.from_local(y, mesh, [Shard(0)], run_check=False)

        # 进入通信调试模式
        with comm_mode:
            # 执行函数 f，传入分布式张量 x_dtensor 和 y_dtensor 作为参数
            f(x_dtensor, y_dtensor)

        # 获取通信操作计数
        comm_counts = comm_mode.get_comm_counts()
        # 断言总通信次数为 1
        self.assertEqual(comm_mode.get_total_counts(), 1)
        # 断言 all_reduce 操作计数为 0
        self.assertEqual(comm_counts[c10d_functional.all_reduce], 0)
        # 断言 all_gather_into_tensor 操作计数为 1
        self.assertEqual(comm_counts[c10d_functional.all_gather_into_tensor], 1)
        # 断言 reduce_scatter_tensor 操作计数为 0
        self.assertEqual(comm_counts[c10d_functional.reduce_scatter_tensor], 0)
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```