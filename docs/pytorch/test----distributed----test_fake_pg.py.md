# `.\pytorch\test\distributed\test_fake_pg.py`

```py
# Owner(s): ["oncall: distributed"]

import sys  # 导入 sys 模块，用于访问系统相关功能
import unittest  # 导入 unittest 模块，用于编写和运行单元测试

import torch  # 导入 PyTorch 库
import torch.distributed as dist  # 导入 PyTorch 分布式模块
import torch.distributed._functional_collectives as funcol  # 导入分布式函数集合
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch.distributed._tensor import DeviceMesh, init_device_mesh, Shard  # 导入分布式相关的张量和初始化函数
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 导入 FullyShardedDataParallel 模块
from torch.distributed.tensor.parallel import (
    ColwiseParallel,  # 导入张量的列并行模块
    parallelize_module,  # 导入模块并行化函数
    RowwiseParallel,  # 导入张量的行并行模块
)
from torch.fx.experimental.proxy_tensor import make_fx  # 导入代理张量的实验性功能
from torch.testing import FileCheck  # 导入文件检查工具
from torch.testing._internal.common_utils import run_tests, TestCase  # 导入运行测试和测试用例相关的工具
from torch.testing._internal.distributed._tensor.common_dtensor import MLPModule  # 导入分布式张量测试相关模块
from torch.testing._internal.distributed.fake_pg import FakeStore  # 导入虚拟进程组存储模块

if not dist.is_available():  # 如果分布式不可用
    print("Distributed not available, skipping tests", file=sys.stderr)  # 输出提示信息到标准错误流
    sys.exit(0)  # 退出程序

HAS_CUDA = torch.cuda.is_available()  # 检查是否有 CUDA 可用

class TestFakePG(TestCase):
    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()  # 销毁进程组

    def test_all_reduce(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)  # 初始化虚拟进程组

        output = torch.ones(3, 3) * dist.get_rank()
        dist.all_reduce(output)  # 对所有进程执行全局归约操作
        self.assertEqual(tuple(output.shape), (3, 3))  # 断言输出张量的形状为 (3, 3)

    def test_allgather(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)  # 初始化虚拟进程组

        input_tensor = torch.ones(3, 3) * dist.get_rank()
        output_tensors = [torch.empty_like(input_tensor) for _ in range(2)]
        dist.all_gather(output_tensors, input_tensor)  # 所有进程间进行全局聚集操作
        for _, out_tensor in enumerate(output_tensors):
            self.assertEqual(tuple(out_tensor.shape), (3, 3))  # 断言输出张量的形状为 (3, 3)

    def test_reduce_scatter(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=1, world_size=2, store=store)  # 初始化虚拟进程组

        to_reduce_scatter = [torch.ones(3, 3) * rank for rank in range(2)]
        output_tensor = torch.empty(3, 3)

        dist.reduce_scatter(output_tensor, to_reduce_scatter)  # 对所有进程执行归约分散操作
        self.assertEqual(tuple(output_tensor.shape), (3, 3))  # 断言输出张量的形状为 (3, 3)

    @unittest.skipIf(not HAS_CUDA, "No CUDA")
    def test_construct_fsdp(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)  # 初始化虚拟进程组
        FSDP(nn.Linear(2, 3, device="cuda"))  # 在 CUDA 设备上构建 FullyShardedDataParallel 对象

    @unittest.skipIf(not HAS_CUDA, "No CUDA")
    def test_fsdp_fake_e2e(self):
        # 创建一个 HashStore 对象作为存储
        store = dist.HashStore()
        # 使用 fake 后端初始化进程组，rank=0，总进程数为2，使用上面创建的 store
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        
        # 创建一个神经网络模块，包含两个线性层和一个 ReLU 激活函数，使用 CUDA 设备
        my_module = nn.Sequential(
            nn.Linear(2, 3, device="cuda"),
            nn.ReLU(),
            nn.Linear(3, 2, device="cuda"),
        )
        
        # 使用 FSDP 包装神经网络模块，保持原始参数
        sharded_module = FSDP(my_module, use_orig_params=True)
        
        # 使用 Adam 优化器优化 FSDP 模块的参数
        optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        
        # 创建输入张量
        input = torch.randn(2, 2)
        
        # 将输入张量传递给 FSDP 模块进行前向传播
        x = sharded_module(input)
        
        # 计算输出张量的总和作为损失
        loss = x.sum()
        
        # 反向传播损失
        loss.backward()
        
        # 使用优化器更新模型参数
        optim.step()

    @unittest.skipIf(not HAS_CUDA, "No CUDA")
    def test_fake_pg_tracing(self):
        # 创建一个 HashStore 对象作为存储
        store = dist.HashStore()
        # 使用 fake 后端初始化进程组，rank=0，总进程数为2，使用上面创建的 store
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        
        # 获取默认的分布式组
        default_pg = dist.distributed_c10d._get_default_group()

        def allgather_fn(tensor):
            # 使用 funcol.all_gather_tensor 函数在默认组中收集张量
            return funcol.all_gather_tensor(tensor, 0, default_pg)

        # 使用 make_fx 包装 allgather_fn 函数，并在 CUDA 设备上应用
        gm = make_fx(allgather_fn)(torch.randn(2, 2, device="cuda"))
        
        # 检查生成的图中是否包含 "all_gather" 和 "wait_tensor"，并运行检查
        FileCheck().check("all_gather").check("wait_tensor").run(str(gm.graph))

    def test_broadcast(self):
        # 创建一个 FakeStore 对象作为存储
        store = FakeStore()
        # 使用 fake 后端初始化进程组，rank=0，总进程数为2，使用上面创建的 store
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        
        # 创建一个全为1的张量作为输出
        output = torch.ones(3, 3)
        
        # 使用 rank=0 进程广播输出张量
        dist.broadcast(output, src=0)
        
        # 断言输出张量的形状为 (3, 3)
        self.assertEqual(tuple(output.shape), (3, 3))
        
        # 创建一个全为1的张量作为输出
        output = torch.ones(3, 3)
        
        # 使用 rank=1 进程广播输出张量
        dist.broadcast(output, src=1)
        
        # 断言输出张量的形状为 (3, 3)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_scatter(self):
        # 创建一个 FakeStore 对象作为存储
        store = FakeStore()
        # 使用 fake 后端初始化进程组，rank=0，总进程数为2，使用上面创建的 store
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        
        # 创建一个全为1的张量作为输出
        output = torch.ones(3, 3)
        
        # 创建要分散的张量列表
        to_scatter = [torch.ones(3, 3) * rank for rank in range(2)]
        
        # 使用默认进程组分散输出张量
        dist.scatter(output, to_scatter)
        
        # 断言输出张量的形状为 (3, 3)
        self.assertEqual(tuple(output.shape), (3, 3))
        
        # 创建一个全为1的张量作为输出
        output = torch.ones(3, 3)
        
        # 使用 rank=1 进程分散输出张量
        dist.scatter(output, None, src=1)
        
        # 断言输出张量的形状为 (3, 3)
        self.assertEqual(tuple(output.shape), (3, 3))

    def test_alltoall(self):
        # 创建一个 FakeStore 对象作为存储
        store = FakeStore()
        # 使用 fake 后端初始化进程组，rank=0，总进程数为2，使用上面创建的 store
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        
        # 创建包含两个全为1的张量的列表作为输出和输入
        output_list = [torch.ones(3, 3) for _ in range(2)]
        input_list = [torch.ones(3, 3) for _ in range(2)]
        
        # 使用默认进程组进行全对全通信
        dist.all_to_all(output_list, input_list)
        
        # 断言输出张量列表的长度为2，并且每个张量的形状为 (3, 3)
        self.assertEqual(len(output_list), 2)
        for output in output_list:
            self.assertEqual(tuple(output.shape), (3, 3))

    def test_alltoall_base(self):
        # 创建一个 FakeStore 对象作为存储
        store = FakeStore()
        # 使用 fake 后端初始化进程组，rank=0，总进程数为2，使用上面创建的 store
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)
        
        # 创建全为1的输出和输入张量
        out_tensor = torch.ones(3, 3)
        in_tensor = torch.ones(3, 3)
        
        # 定义输出和输入张量的分割策略为 [1, 1]
        output_split = [1, 1]
        input_split = [1, 1]
        
        # 使用默认进程组进行全对全通信
        dist.all_to_all_single(out_tensor, in_tensor, output_split, input_split)
        
        # 断言输出张量的形状为 (3, 3)
        self.assertEqual(tuple(out_tensor.shape), (3, 3))
    # 定义发送数据的测试函数
    def test_send(self):
        # 创建一个虚拟存储
        store = FakeStore()
        # 初始化进程组，使用虚拟后端，设置当前进程的排名和总进程数，以及存储
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        # 创建一个全为1的3x3张量
        tensor = torch.ones(3, 3)
        # 发送张量到排名为1的进程
        dist.send(tensor, 1)
        # 断言发送后张量的形状仍为(3, 3)
        self.assertEqual(tuple(tensor.shape), (3, 3))

    # 定义接收数据的测试函数
    def test_recv(self):
        # 创建一个虚拟存储
        store = FakeStore()
        # 初始化进程组，使用虚拟后端，设置当前进程的排名和总进程数，以及存储
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        # 创建一个全为1的3x3张量
        output = torch.ones(3, 3)
        # 接收来自排名为1的进程的张量数据
        dist.recv(output, 1)
        # 断言接收后张量的形状仍为(3, 3)
        self.assertEqual(tuple(output.shape), (3, 3))

    # 定义端到端的TP+FSDP与fake后端的集成测试函数
    @unittest.skipIf(not HAS_CUDA, "No CUDA or TP+FSDP")
    def test_fsdp_tp_fake_e2e(self):
        # 设置世界大小为4，每个TP组大小为2
        world_size = 4
        tp_size = 2

        # 创建一个哈希存储对象
        store = dist.HashStore()
        # 初始化进程组，使用虚拟后端，设置当前进程的排名和总进程数，以及存储
        dist.init_process_group(
            backend="fake", rank=0, world_size=world_size, store=store
        )

        # 创建设备网格，指定使用cuda设备，构建设备索引网格
        device_mesh = DeviceMesh("cuda", torch.arange(0, world_size).view(-1, tp_size))
        device_mesh = init_device_mesh(
            "cuda", (world_size // tp_size, tp_size), mesh_dim_names=["dp", "tp"]
        )

        # 定义序列并行化计划和成对并行化计划
        sequence_parallelize_plan = {
            "net1": ColwiseParallel(input_layouts=Shard(0)),
            "net2": RowwiseParallel(output_layouts=Shard(0)),
        }
        pairwise_parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }

        # 循环执行序列并行化和成对并行化计划
        for parallel_plan in [sequence_parallelize_plan, pairwise_parallelize_plan]:
            # 并行化模块，使用MLPModule作为基础模块，将其部署到tp设备上，应用并行化计划
            my_module = parallelize_module(
                MLPModule(device="cuda"),
                device_mesh["tp"],
                parallel_plan,
            )

            # 使用FSDP进行模块分片，使用原始参数，设备网格应用到dp维度
            sharded_module = FSDP(
                my_module, use_orig_params=True, device_mesh=device_mesh["dp"]
            )
            # 使用Adam优化器优化分片模块的参数
            optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)

            # 执行10次训练迭代
            for i in range(10):
                # 获取当前进程的dp排名
                dp_rank = dist.get_rank()
                # 设置随机种子，确保每次迭代使用不同的种子
                torch.manual_seed(i + dp_rank)
                # 在当前进程的GPU上生成随机输入张量
                input = torch.randn(20, 10).cuda(dist.get_rank())
                # 将输入数据输入到分片模块中，获取输出
                x = sharded_module(input)
                # 计算输出的总损失
                loss = x.sum()
                # 反向传播计算梯度
                loss.backward()
                # 使用优化器更新模型参数
                optim.step()
# 如果当前脚本作为主程序执行（而不是被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```