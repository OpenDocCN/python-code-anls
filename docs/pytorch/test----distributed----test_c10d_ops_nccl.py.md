# `.\pytorch\test\distributed\test_c10d_ops_nccl.py`

```
# Owner(s): ["oncall: distributed"]
# This test file contains positive tests for c10d with NCCL backend.
# During the test, it is expected that ProcessGroup will not be aborted, destroyed or incur fatal error.
# Please be mindful of this when adding tests here.
# If you need to add tests for group creation, abort or destroy, please add tests in test_c10d_nccl.py.

# There are two ways to launch tests in this file:
# 1. Run this file directly with `python test_c10d_ops_nccl.py`
# 2. Use multi-process launcher, e.g. `torchrun --standalone --nproc-per-node 2 test_c10d_ops_nccl.py`

# 导入需要的模块和库
import math  # 导入数学库
import os  # 导入操作系统库
import sys  # 导入系统库
import tempfile  # 导入临时文件库

import torch  # 导入PyTorch
import torch.distributed as c10d  # 导入PyTorch分布式通信库

# 如果 c10d 或者 NCCL 不可用，跳过测试
if not c10d.is_available() or not c10d.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed as dist  # 导入分布式通信库
from torch.testing._internal.common_cuda import TEST_MULTIGPU  # 导入多GPU测试标志
from torch.testing._internal.common_distributed import (
    init_multigpu_helper,  # 导入多GPU初始化辅助函数
    MultiProcContinousTest,  # 导入多进程连续测试类
    requires_nccl,  # 导入NCCL要求装饰器
)
from torch.testing._internal.common_utils import (
    skip_but_pass_in_sandcastle_if,  # 导入在沙堡中跳过但通过的函数
    skipIfRocm,  # 导入在ROCm平台下跳过的函数
    TEST_WITH_DEV_DBG_ASAN,  # 导入带ASAN调试标志的测试函数
)

# 如果使用ASAN调试，跳过测试
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues", file=sys.stderr
    )
    sys.exit(0)

class ProcessGroupNCCLOpTest(MultiProcContinousTest):
    @classmethod
    def backend_str(cls) -> str:
        return "nccl"

    @classmethod
    def opts(cls, high_priority_stream=False):
        opts = c10d.ProcessGroupNCCL.Options()
        opts.is_high_priority_stream = high_priority_stream
        return opts

    @property
    def rank_to_GPU(self):
        # 返回排名到GPU映射
        return init_multigpu_helper(self.world_size, "nccl")

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_empty_tensors(self):
        pg = self.pg
        local_device_idx = self.rank_to_GPU[self.rank][0]

        xs = [torch.FloatTensor([]).cuda(local_device_idx)]
        pg.broadcast(xs).wait()
        self.assertEqual(0, xs[0].numel())

        pg.allreduce(xs).wait()
        self.assertEqual(0, xs[0].numel())

        pg.reduce(xs).wait()
        self.assertEqual(0, xs[0].numel())

        ys = [
            [
                torch.FloatTensor([]).cuda(local_device_idx)
                for _ in range(self.world_size)
            ]
        ]
        pg.allgather(ys, xs).wait()
        for y in ys[0]:
            self.assertEqual(0, y.numel())

        ys = [torch.FloatTensor([]).cuda(local_device_idx)]
        xs = [
            [
                torch.FloatTensor([]).cuda(local_device_idx)
                for _ in range(self.world_size)
            ]
        ]
        pg.reduce_scatter(ys, xs).wait()
        self.assertEqual(0, ys[0].numel())

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    # 定义测试方法，测试广播操作
    def test_broadcast_ops(self):
        # 获取当前测试类的属性 pg
        pg = self.pg

        # 定义广播函数，接收输入列表 xs、根节点的排名 rootRank、根节点的张量 rootTensor
        def broadcast(xs, rootRank, rootTensor):
            # 创建广播选项对象 opts
            opts = c10d.BroadcastOptions()
            # 设置广播选项的根节点排名
            opts.rootRank = rootRank
            # 设置广播选项的根节点张量
            opts.rootTensor = rootTensor
            # 调用进程组 pg 的广播方法，返回工作句柄 work
            work = pg.broadcast(xs, opts)
            # 等待广播完成
            work.wait()
            # 返回输入列表 xs
            return xs

        # 每个排名依次作为根节点
        for i in range(self.world_size):
            # 创建包含当前排名的张量 x，并将其放置在对应 GPU 上
            x = torch.tensor([self.rank]).cuda(self.rank_to_GPU[self.rank][0])
            # 调用广播函数，广播张量 x，根节点为 i，根节点张量为 0
            output = broadcast([x], i, 0)
            # 断言广播后的输出与预期的张量 [i] 相等
            self.assertEqual(torch.tensor([i]), output[0])

            # 创建预期的张量，维度为 (i+1, i+1)，并填充值为 i+1
            expected_tensor = torch.empty([i + 1, i + 1]).fill_(i + 1)
            # 创建多个张量 xs，每个张量维度为 (i+1, i+1)，填充值为 -1，并放置在对应 GPU 上
            xs = [
                torch.empty([i + 1, i + 1]).fill_(-1).cuda(device=device_idx)
                for device_idx in self.rank_to_GPU[self.rank]
            ]

            # 在多个输入张量上进行测试（同一排名上的多个 GPU）
            for j in range(len(xs)):
                # 如果当前排名为 i，则将预期张量放置在对应 GPU 上的 xs[j]
                if self.rank == i:
                    xs[j] = expected_tensor.cuda(device=self.rank_to_GPU[self.rank][j])

                # 调用广播函数，广播张量列表 xs，根节点为 i，根节点张量为 j
                broadcast(xs, i, j)

                # 对每个张量进行断言，确保其与预期张量相等
                for tensor in xs:
                    self.assertEqual(tensor, expected_tensor)

    # 标记需要使用 NCCL 库的测试方法，测试稀疏全局归约操作
    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_sparse_allreduce_ops(self):
        # 获取当前测试类的属性 pg
        pg = self.pg

        # 创建稀疏张量 indices 和 values，并生成稀疏 COO 张量 sparse_tensor
        indices = torch.tensor([[0, 1]])
        values = torch.tensor([[1, 2, 0], [4, 0, 6]])
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size=(2, 3)).to(
            self.rank
        )

        # 尝试执行稀疏全局归约操作，捕获可能的运行时异常
        try:
            # 将稀疏张量 sparse_tensor 放入列表 tensor_list
            tensor_list = [sparse_tensor]
            # 调用进程组 pg 的全局归约方法，返回工作句柄 work
            work = pg.allreduce(tensor_list)
            # 等待全局归约完成
            work.wait()

            # 断言列表 tensor_list 中的稀疏张量与预期的密集张量 a 相等
            a = torch.tensor([[2, 4, 0], [8, 0, 12]]).to(self.rank)
            self.assertEqual(tensor_list[0], a)
        except RuntimeError as e:
            # 如果异常信息指示 NCCL 不支持稀疏张量的全局归约，则忽略异常
            if "NCCL does not support all_reduce with sparse tensors" in str(e):
                pass
            else:
                # 如果是其他异常，重新抛出该异常
                raise
    # 定义一个测试函数，用于测试多GPU环境下的Allreduce操作
    def test_allreduce_ops(self):
        # 获取当前设备上的GPU数量
        device_count = torch.cuda.device_count()
        # 获取当前进程组
        pg = self.pg
        # 获取当前进程在本地GPU上的设备ID
        local_device_id = self.rank_to_GPU[self.rank][0]

        # 定义一个执行Allreduce操作的内部函数
        def allreduce(tensors, op):
            # 创建Allreduce选项对象
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            # 执行Allreduce操作，返回一个用于等待操作完成的future对象
            work = pg.allreduce(tensors, opts)
            # 等待操作完成
            work.wait()

        # Sum（求和操作）
        # 创建包含当前进程排名加一的张量列表，并放置在当前设备上
        tensors = [torch.tensor([self.rank + 1]).cuda(local_device_id)]
        # 执行求和操作
        allreduce(tensors, c10d.ReduceOp.SUM)
        # 验证求和结果是否正确
        ndev = self.world_size
        self.assertEqual(
            torch.tensor([ndev * (ndev + 1) // 2]),
            tensors[0],
        )

        # Avg（平均值操作，仅在NCCL版本高于等于2.10时可用）
        if torch.cuda.nccl.version() >= (2, 10, 0):
            # 创建包含当前进程排名加一的浮点张量列表，并放置在当前设备上
            tensors = [torch.tensor([self.rank + 1.0]).cuda(local_device_id)]
            # 执行平均值操作
            allreduce(tensors, c10d.ReduceOp.AVG)
            # 验证平均值操作结果是否正确
            ndev = self.world_size
            self.assertEqual(
                torch.tensor([ndev * (ndev + 1.0) / (2.0 * ndev)]),
                tensors[0],
            )

        # Premul Sum（预乘和操作，仅在NCCL版本高于等于2.11.1时可用）
        if torch.cuda.nccl.version() >= (2, 11, 1):
            # 遍历不同数据类型和乘法因子的组合
            for dtype in torch.half, torch.float, torch.double:
                for factor in (
                    3.0,
                    torch.tensor([5.0], device=local_device_id, dtype=dtype),
                ):
                    # 创建包含当前进程排名加一的张量列表，并放置在当前设备上，并转换为指定数据类型
                    tensors = [
                        torch.tensor([self.rank + 1])
                        .cuda(local_device_id)
                        .to(dtype=dtype)
                    ]
                    # 执行预乘和操作
                    allreduce(tensors, c10d._make_nccl_premul_sum(factor))
                    # 验证预乘和操作结果是否正确
                    self.assertEqual(
                        factor
                        * torch.tensor(
                            [self.world_size * (self.world_size + 1) / 2],
                            dtype=dtype,
                            device=local_device_id,
                        ),
                        tensors[0],
                    )

        # Product（乘积操作）
        # 创建包含当前进程排名加一的张量列表，并放置在当前设备上
        tensors = [torch.tensor([self.rank + 1]).cuda(local_device_id)]
        # 执行乘积操作
        allreduce(tensors, c10d.ReduceOp.PRODUCT)
        # 验证乘积操作结果是否正确
        self.assertEqual(torch.tensor([math.factorial(self.world_size)]), tensors[0])

        # Min（最小值操作）
        # 创建包含当前进程排名加一的张量列表，并放置在当前设备上
        tensors = [torch.tensor([self.rank + 1]).cuda(local_device_id)]
        # 执行最小值操作
        allreduce(tensors, c10d.ReduceOp.MIN)
        # 验证最小值操作结果是否正确
        self.assertEqual(torch.tensor([1]), tensors[0])

        # Max（最大值操作）
        # 创建包含当前进程排名加一的张量列表，并放置在当前设备上
        tensors = [torch.tensor([self.rank + 1]).cuda(local_device_id)]
        # 执行最大值操作
        allreduce(tensors, c10d.ReduceOp.MAX)
        # 验证最大值操作结果是否正确
        self.assertEqual(torch.tensor([self.world_size]), tensors[0])

        # 遍历位操作和相应的错误信息
        for op, err in zip(
            (c10d.ReduceOp.BAND, c10d.ReduceOp.BOR, c10d.ReduceOp.BXOR),
            ("ReduceOp.BAND", "ReduceOp.BOR", "ReduceOp.BXOR"),
        ):
            # 断言使用NCCL时不能使用位操作，抛出值错误并包含错误信息
            with self.assertRaisesRegex(ValueError, "Cannot use " + err + " with NCCL"):
                allreduce(tensors, op)

    # 添加装饰器，要求当前测试依赖NCCL
    @requires_nccl()
    # 在沙盒环境中跳过此测试，除非测试多GPU功能可用
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @skipIfRocm()
    def test_alltoall_ops_with_cudafree_race(self):
        # 获取当前测试组的进程组对象
        pg = self.pg
        # 创建 AllToAllOptions 对象
        opts = c10d.AllToAllOptions()
        # 获取本地设备的 CUDA 设备名称
        local_device = f"cuda:{self.rank_to_GPU[self.rank][0]}"
        # 设置当前 CUDA 设备
        torch.cuda.set_device(local_device)
        # 在指定 CUDA 设备上生成随机张量 input 和 output
        input = torch.rand(1000, 1000, device=local_device)
        output = torch.rand(1000, 1000, device=local_device)
        # 创建用于竞争的张量列表 race_tensors
        race_tensors = []
        
        # 创建用于竞争的张量，与 alltoall 集合进行竞争
        for _ in range(10):
            tmp = []
            for i in range(5):
                tmp.append(torch.rand(10 ** (3 + i), device=local_device))
            race_tensors.append(tmp)

        # 进行迭代，pop 出 race_tensors 中的张量，执行 alltoall_base 操作
        for i in range(10):
            race_tensors.pop()
            work = pg.alltoall_base(output, input, [], [], opts)
            # 执行 torch.cuda.empty_cache() 触发 cudaFree
            torch.cuda.empty_cache()
            # 等待操作完成
            work.wait()
        
        # 同步 CUDA 设备
        torch.cuda.synchronize(device=local_device)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_allreduce_in_cudagraph(self):
        # 获取当前测试组的进程组对象
        pg = self.pg
        # 获取本地设备索引
        local_device_idx = self.rank_to_GPU[self.rank][0]
        # 设置当前 CUDA 设备
        with torch.cuda.device(local_device_idx):
            # 创建一个包含单个浮点数的张量列表 xs
            xs = [torch.FloatTensor([1]).cuda(local_device_idx)]

            # 执行一次 allreduce 操作并等待其完成
            pg.allreduce(xs).wait()
            # 断言操作结果为 2
            self.assertEqual(xs[0].item(), 2)

            # 创建一个 CUDA 图对象 graph
            graph = torch.cuda.CUDAGraph()
            # 在 CUDA 图上下文中执行 allreduce 操作并等待其完成
            with torch.cuda.graph(graph):
                pg.allreduce(xs).wait()
            # 断言操作结果为 2
            self.assertEqual(xs[0].item(), 2)

            # 回放 CUDA 图两次
            graph.replay()
            graph.replay()
            # 断言操作结果为 8
            self.assertEqual(xs[0].item(), 8)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @skipIfRocm()
    def test_nccl_watchdog_cudagraph(self):
        # 测试看门狗不会因为不允许的事件查询而导致图形崩溃
        # 获取当前测试组的进程组对象
        pg = self.pg
        # 获取当前进程的 GPU 索引
        rank = self.rank_to_GPU[self.rank][0]
        # 设置当前 CUDA 设备
        with torch.cuda.device(rank):
            for i in range(10):
                # 创建包含单个浮点数的张量列表 xs 和 ys
                xs = [torch.FloatTensor([1]).cuda(rank)]
                ys = [torch.FloatTensor([4]).cuda(rank)]
                # 执行多次 allreduce 操作并等待其完成
                for _ in range(30):
                    pg.allreduce(xs[0]).wait()

                # 创建 CUDA 图对象 graph
                graph = torch.cuda.CUDAGraph()
                # 在 CUDA 图上下文中执行一系列操作并等待其完成
                with torch.cuda.graph(graph):
                    xs[0] += 0.0
                    pg.allreduce(xs[0]).wait()
                    pg.allreduce(xs[0]).wait()
                    pg.allreduce(xs[0]).wait()
                    xs[0] += 0.0

                # 多次回放 CUDA 图
                for _ in range(100):
                    graph.replay()
    def test_reduce_ops(self):
        # 获取self.pg的引用
        pg = self.pg
        # 获取本地设备ID
        local_device_id = self.rank_to_GPU[self.rank][0]

        # 定义reduce函数，用于执行reduce操作
        def reduce(xs, rootRank, rootTensor, op=None):
            # 创建ReduceOptions对象
            opts = c10d.ReduceOptions()
            # 设置根节点的rank
            opts.rootRank = rootRank
            # 设置根节点的张量
            opts.rootTensor = rootTensor
            # 如果指定了操作符，则设置reduce操作符
            if op:
                opts.reduceOp = op
            # 执行reduce操作，返回一个Work对象
            work = pg.reduce(xs, opts)
            # 等待reduce操作完成
            work.wait()

        # 遍历每一个根张量
        for rt in range(self.world_size):
            # 创建包含当前rank加1的Tensor列表，并移到指定的本地GPU
            tensors = [torch.tensor([self.rank + 1]).cuda(local_device_id)]

            # 执行reduce操作
            reduce(tensors, rt, 0)

            # 如果当前rank等于根节点的rank
            if self.rank == rt:
                # 断言当前Tensor等于预期值
                self.assertEqual(
                    torch.tensor([self.world_size * (self.world_size + 1) // 2]),
                    tensors[0],
                )
            else:
                # 断言当前Tensor等于预期值
                self.assertEqual(
                    torch.tensor([self.rank + 1]),
                    tensors[0],
                )

            # 对于每一个操作符op和其对应的错误信息err
            for op, err in zip(
                (c10d.ReduceOp.BAND, c10d.ReduceOp.BOR, c10d.ReduceOp.BXOR),
                ("ReduceOp.BAND", "ReduceOp.BOR", "ReduceOp.BXOR"),
            ):
                # 使用断言检查是否抛出预期的异常
                with self.assertRaisesRegex(
                    ValueError, "Cannot use " + err + " with NCCL"
                ):
                    reduce(tensors, self.rank, rt, op)

            # 如果当前CUDA NCCL的版本大于等于(2, 11, 1)
            # 对于每一个因子factor
            if torch.cuda.nccl.version() >= (2, 11, 1):
                for factor in (3.0, torch.tensor([5.0], device=local_device_id)):
                    # 如果因子是Tensor类型，则获取其CPU上的值
                    if isinstance(factor, torch.Tensor):
                        factor_ref = factor.cpu().item()
                    else:
                        factor_ref = factor
                    # 创建包含当前rank加1.0的浮点数Tensor列表，并移到指定的本地GPU
                    float_tensors = [
                        torch.tensor(
                            [self.rank + 1.0], device=f"cuda:{local_device_id}"
                        )
                    ]
                    # 创建包含当前(rank+1.0)*factor_ref的浮点数Tensor列表，并移到指定的本地GPU
                    float_tensors_ref = [
                        torch.tensor(
                            [(self.rank + 1.0) * factor_ref],
                            device=f"cuda:{local_device_id}",
                        )
                    ]

                    # 执行reduce操作，预先乘以因子
                    reduce(float_tensors_ref, rt, 0)
                    reduce(float_tensors, rt, 0, c10d._make_nccl_premul_sum(factor))
                    # 如果当前rank等于根节点的rank
                    # 断言两个浮点数Tensor相等
                    if self.rank == rt:
                        self.assertEqual(float_tensors_ref[0], float_tensors[0])
    def test_allgather_ops(self):
        # 获取当前进程组
        pg = self.pg
        # 获取当前进程对应的本地 GPU 设备 ID 列表
        local_device_ids = self.rank_to_GPU[self.rank]

        def allgather(output_ts, input_ts):
            # 调用进程组的 allgather 方法进行数据收集
            work = pg.allgather(output_ts, input_ts)
            # 等待所有异步操作完成
            return work.wait()

        # 创建本地 GPU 设备上的张量列表
        tensors = [torch.empty(2, 2).fill_(2).cuda(device=i) for i in local_device_ids]
        output_tensors = []
        expected_output = []

        # 准备每个 GPU 上的输出和预期输出张量列表
        output_per_gpu = (
            [torch.empty(2, 2).fill_(-1)] * len(local_device_ids) * self.world_size
        )
        expected_per_gpu = (
            [torch.empty(2, 2).fill_(2)] * len(local_device_ids) * self.world_size
        )

        # 将每个 GPU 上的输出和预期输出张量列表添加到对应的列表中
        for gpu in local_device_ids:
            output_tensors.append([t.cuda(device=gpu) for t in output_per_gpu])
            expected_output.append([t.cuda(device=gpu) for t in expected_per_gpu])

        # 执行 allgather 操作
        result = allgather(output_tensors, tensors)

        # 验证结果
        self.assertEqual(output_tensors, expected_output)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_allgather_base_ops(self):
        # 获取当前进程组
        pg = self.pg
        # 获取当前进程对应的本地 GPU 设备 ID
        local_device_id = self.rank_to_GPU[self.rank][0]

        def allgather_base(output_t, input_t):
            # 调用进程组的 _allgather_base 方法进行基础 allgather 操作
            work = pg._allgather_base(output_t, input_t)
            # 等待异步操作完成
            work.wait()

        # 创建当前进程在本地 GPU 上的输入张量
        tensor = torch.tensor([self.rank]).cuda(local_device_id)
        # 创建当前进程在本地 GPU 上的输出张量
        output_t = torch.empty((self.world_size), dtype=tensor.dtype).cuda(
            local_device_id
        )

        # 执行基础 allgather 操作
        allgather_base(output_t, tensor)

        # 验证结果
        self.assertEqual(torch.arange(self.world_size), output_t)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    # 定义测试函数 test_allgather_base_basics，测试分布式环境下的 allgather_base 函数
    def test_allgather_base_basics(self):
        # 获取当前测试中的进程组
        pg = self.pg
        # 获取当前进程在 GPU 中的本地设备 ID
        local_device_id = self.rank_to_GPU[self.rank][0]

        # 定义 allgather_base 函数，调用进程组的 _allgather_base 方法并等待完成
        def allgather_base(output_t, input_t):
            work = pg._allgather_base(output_t, input_t)
            work.wait()

        # 预期会抛出 ValueError 异常，因为输出张量的大小必须等于世界大小乘以输入张量的大小
        with self.assertRaisesRegex(
            ValueError,
            "output tensor size must be equal to world_size times input tensor size",
        ):
            # 创建一个包含当前进程编号的张量，并将其放在指定的本地 GPU 上
            tensor = torch.tensor([self.rank]).cuda(local_device_id)
            # 创建一个空张量作为输出，其大小比世界大小多一
            output_t = torch.empty((self.world_size + 1), dtype=tensor.dtype).cuda(
                local_device_id
            )
            # 调用 allgather_base 函数，由于输出张量大小不正确，会触发异常
            allgather_base(output_t, tensor)

        # 预期会抛出 TypeError 异常，因为输出张量必须与输入张量具有相同的数据类型
        with self.assertRaisesRegex(
            TypeError, "output tensor must have the same type as input tensor"
        ):
            # 创建一个浮点类型的张量，并将其放在指定的本地 GPU 上
            tensor = torch.tensor([self.rank], dtype=torch.float).cuda(local_device_id)
            # 创建一个长整型的空张量作为输出
            output_t = torch.empty((self.world_size + 1), dtype=torch.long).cuda(
                local_device_id
            )
            # 调用 allgather_base 函数，由于输出张量类型不匹配，会触发异常
            allgather_base(output_t, tensor)

    # 装饰器，要求当前测试函数依赖于 NCCL 库
    @requires_nccl()
    # 装饰器，在不支持多 GPU 或者在沙盒环境中时跳过测试
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    # 定义测试函数 test_gather_ops，测试分布式环境下的 gather 函数
    def test_gather_ops(self):
        # 获取当前测试中的进程组
        pg = self.pg
        # 获取当前进程在多个 GPU 上的本地设备 ID 列表
        local_device_ids = self.rank_to_GPU[self.rank]
        # 计算当前进程可用的 GPU 数量
        num_gpus = len(local_device_ids)

        # 定义 gather 函数，用于执行 Gather 操作，并等待完成
        def gather(output_t, input_t, rootRank):
            # 创建 GatherOptions 对象
            opts = c10d.GatherOptions()
            opts.rootRank = rootRank
            # 如果当前进程是根节点进程，则调用进程组的 gather 方法
            if rootRank == self.rank:
                work = pg.gather(output_t, input_t, opts)
            else:
                # 如果当前进程不是根节点进程，则传递空列表给 gather 方法
                work = pg.gather([], input_t, opts)
            # 等待 gather 操作完成
            work.wait()

        # 初始化输入张量列表
        tensors = []
        for device_id in local_device_ids:
            # 对于每个本地设备 ID，创建包含当前进程编号的张量，并将其放在对应的 GPU 上
            tensors.append(torch.tensor([self.rank]).cuda(device_id))

        # 初始化输出张量列表
        output_ts = []
        for idx in range(num_gpus):
            gpu_idx = local_device_ids[idx]
            output_ts.append([])
            for rank in range(self.world_size):
                # 对于每个 GPU 和每个进程，创建一个初始值为 -1 的张量
                output_ts[idx].append(torch.tensor([-1]).cuda(gpu_idx))

        # 期望的结果是一个二维列表，包含从 0 到世界大小的每个进程编号的张量
        expected = [[torch.tensor([rank]) for rank in range(self.world_size)]]
        for rank in range(self.world_size):
            # 对每个进程调用 gather 函数
            gather(output_ts, tensors, rank)
            # 如果当前进程是根节点进程，则验证输出张量与预期结果是否相等
            if rank == self.rank:
                self.assertEqual(expected, output_ts)
    # 测试多GPU情况下gather函数的性能
    def test_gather_stress(self):
        # 获取当前进程组
        pg = self.pg
        # 获取本地设备ID列表
        local_device_ids = self.rank_to_GPU[self.rank]
        # 计算本地GPU数量
        num_gpus = len(local_device_ids)

        # 定义gather函数，用于收集输出张量到指定根节点的处理器上
        def gather(output_t, input_t, rootRank):
            # 创建GatherOptions对象
            opts = c10d.GatherOptions()
            opts.rootRank = rootRank
            # 如果当前进程是根节点，则调用pg.gather函数收集数据
            if rootRank == self.rank:
                work = pg.gather(output_t, input_t, opts)
            # 如果当前进程不是根节点，则仅分配空间
            else:
                work = pg.gather([], input_t, opts)
            # 等待gather操作完成
            work.wait()

        # 定义压力测试数据长度
        stress_length = 1000

        # 初始化输入张量列表
        tensors = []
        for i in range(stress_length):
            tensors.append([])
            for device_id in local_device_ids:
                tensors[i].append(torch.tensor([self.rank]).cuda(device_id))

        # 初始化输出张量列表
        output_ts = []
        for i in range(stress_length):
            output_ts.append([[] for _ in range(num_gpus)])
            for idx, ls in enumerate(output_ts[i]):
                gpu_idx = local_device_ids[idx]
                for _ in range(self.world_size):
                    ls.append(torch.tensor([-1]).cuda(gpu_idx))

        # 预期输出结果
        expected = [[torch.tensor([rank]) for rank in range(self.world_size)]]

        # 对每个根节点进行gather操作，并进行验证
        for i in range(stress_length):
            for rank in range(self.world_size):
                gather(output_ts[i], tensors[i], rank)
                # 验证gather的输出是否与预期结果相符
                if rank == self.rank:
                    self.assertEqual(output_ts[i], expected)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    # 测试gather函数的各种边界条件和异常情况
    def test_gather_checks(self):
        # 获取当前进程组
        pg = self.pg
        # 获取本地设备ID
        device_id = self.rank_to_GPU[self.rank][0]

        # 初始化输入张量
        tensor = torch.tensor([self.rank]).cuda(device_id)

        # 初始化输出张量列表
        output_ts = []
        for rank in range(self.world_size):
            output_ts.append(torch.tensor([-1]).cuda(device_id))

        # 检查根节点为负数时是否抛出异常
        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = -1
            pg.gather([output_ts], [tensor], opts)

        # 检查根节点类型错误时是否抛出异常
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
            pg.gather([output_ts], [tensor], 0)

        # 检查根节点超出范围时是否抛出异常
        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.world_size
            pg.gather([output_ts], [tensor], opts)

        # 检查没有张量参数时是否抛出异常
        with self.assertRaisesRegex(
            # 从分发器抛出的错误消息
            RuntimeError,
            "There were no tensor arguments to this function",
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather([output_ts], [], opts)
    def test_scatter_ops(self):
        # 获取当前测试实例的进程组
        pg = self.pg
        # 获取当前进程的本地设备 ID 列表
        local_device_ids = self.rank_to_GPU[self.rank]
        # 计算本地设备的数量
        num_gpus = len(local_device_ids)

        def scatter(output_t, input_t, rootRank):
            # 创建 ScatterOptions 对象
            opts = c10d.ScatterOptions()
            # 设置根进程的排名
            opts.rootRank = rootRank
            # 如果是根进程，则执行 scatter 操作
            if rootRank == self.rank:
                work = pg.scatter(output_t, input_t, opts)
            # 如果不是根进程，则执行空的 scatter 操作
            else:
                work = pg.scatter(output_t, [], opts)
            # 等待 scatter 操作完成
            work.wait()

        # 初始化输出张量列表
        tensors = []
        for device_id in local_device_ids:
            # 在每个 GPU 上创建一个包含 [-1] 的张量，并移动到 CUDA 设备
            tensors.append(torch.tensor([-1]).cuda(device_id))

        # 初始化输入张量列表
        scatter_list = []
        for idx in range(num_gpus):
            gpu_idx = local_device_ids[idx]
            scatter_list.append([])
            for rank in range(self.world_size):
                # 在每个 GPU 上创建包含当前进程排名的张量，并移动到 CUDA 设备
                scatter_list[idx].append(torch.tensor([rank]).cuda(gpu_idx))

        # 测试每个进程排名的 scatter 操作
        expected = [torch.tensor([self.rank])]
        for rank in range(self.world_size):
            # 对 tensors 执行 scatter 操作
            scatter(tensors, scatter_list, rank)
            # 验证结果是否符合预期
            self.assertEqual(expected, tensors)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_scatter_stress(self):
        # 获取当前测试实例的进程组
        pg = self.pg
        # 获取当前进程的本地设备 ID 列表
        local_device_ids = self.rank_to_GPU[self.rank]
        # 计算本地设备的数量
        num_gpus = len(local_device_ids)

        def scatter(output_t, input_t, rootRank):
            # 创建 ScatterOptions 对象
            opts = c10d.ScatterOptions()
            # 设置根进程的排名
            opts.rootRank = rootRank
            # 如果是根进程，则执行 scatter 操作
            if rootRank == self.rank:
                work = pg.scatter(output_t, input_t, opts)
            # 如果不是根进程，则执行空的 scatter 操作
            else:
                work = pg.scatter(output_t, [], opts)
            # 等待 scatter 操作完成
            work.wait()

        stress_length = 1000

        # 初始化输出张量列表
        tensors = []
        for i in range(stress_length):
            tensors.append([])
            for device_id in local_device_ids:
                # 在每个 GPU 上创建一个包含 [-1] 的张量，并移动到 CUDA 设备
                tensors[i].append(torch.tensor([-1]).cuda(device_id))

        # 初始化输入张量列表
        scatter_list = []
        for i in range(stress_length):
            scatter_list.append([[] for _ in range(num_gpus)])
            for idx, ls in enumerate(scatter_list[i]):
                gpu_idx = local_device_ids[idx]
                for rank in range(self.world_size):
                    # 在每个 GPU 上创建包含当前进程排名的张量，并移动到 CUDA 设备
                    ls.append(torch.tensor([rank]).cuda(gpu_idx))

        # 测试每个进程排名的 scatter 操作
        expected = [torch.tensor([self.rank])]
        for i in range(stress_length):
            for rank in range(self.world_size):
                # 对 tensors 执行 scatter 操作
                scatter(tensors[i], scatter_list[i], rank)
                # 验证结果是否符合预期
                self.assertEqual(tensors[i], expected)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_scatter_checks(self):
        # 获取当前测试用例中的进程组
        pg = self.pg
        # 获取当前进程在GPU字典中对应的本地设备ID列表
        local_device_ids = self.rank_to_GPU[self.rank]
        # 获取本地设备ID的数量
        num_gpus = len(local_device_ids)

        # 初始化输出张量列表
        tensors = []
        # 为每个设备ID创建一个包含值为-1的张量，并放置在对应设备上
        for device_id in local_device_ids:
            tensors.append(torch.tensor([-1]).cuda(device_id))

        # 初始化输入的scatter列表
        scatter_list = []
        # 对于每个GPU，创建一个空列表
        for idx in range(num_gpus):
            gpu_idx = local_device_ids[idx]
            scatter_list.append([])
            # 对于每个rank，创建一个包含当前rank值的张量，并放置在对应的GPU上
            for rank in range(self.world_size):
                scatter_list[idx].append(torch.tensor([rank]).cuda(gpu_idx))

        # 预期抛出值错误异常，验证根rank参数的有效性
        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = -1
            pg.scatter(tensors, scatter_list, opts)

        # 预期抛出类型错误异常，验证scatter函数的参数类型不兼容
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
            pg.scatter(tensors, scatter_list, 0)

        # 预期抛出值错误异常，验证根rank参数的有效性
        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.world_size
            pg.scatter(tensors, scatter_list, opts)

        # 预期抛出运行时错误异常，验证scatter函数调用时参数为空
        with self.assertRaisesRegex(
            # 从分发器抛出的错误消息
            RuntimeError,
            "There were no tensor arguments to this function",
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([], scatter_list, opts)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_reduce_scatter_base_basics(self):
        # 获取当前测试用例中的进程组
        pg = self.pg
        # 获取当前进程在GPU字典中对应的本地设备ID
        local_device_id = self.rank_to_GPU[self.rank][0]

        def reduce_scatter_base(output_t, input_t):
            # 执行基础的reduce scatter操作
            work = pg._reduce_scatter_base(output_t, input_t)
            work.wait()

        # 预期抛出值错误异常，验证输入张量的大小必须与输出大小乘以world size相同
        with self.assertRaisesRegex(
            ValueError,
            "input tensor must be the same size as output size times world size",
        ):
            # 创建输入张量和输出张量，但输出张量的大小不正确
            input_t = torch.tensor([self.rank]).cuda(local_device_id)
            output_t = torch.empty((self.world_size + 1), dtype=input_t.dtype).cuda(
                local_device_id
            )
            # 由于输出张量的大小不正确，操作将失败
            reduce_scatter_base(output_t, input_t)

        # 预期抛出类型错误异常，验证输入张量必须与输出张量的类型相同
        with self.assertRaisesRegex(
            TypeError, "input tensor must be the same type as the output tensor."
        ):
            # 创建一个浮点类型的张量和一个整型的输出张量
            tensor = torch.tensor([self.rank], dtype=torch.float).cuda(local_device_id)
            output_t = torch.empty((self.world_size + 1), dtype=torch.long).cuda(
                local_device_id
            )
            # 由于张量的类型不同，操作将失败
            reduce_scatter_base(output_t, tensor)
    # 如果不测试多 GPU，跳过该测试，但在沙堡环境中继续执行（NCCL 测试需要至少 2 个 GPU）
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_reduce_scatter_base_ops(self):
        pg = self.pg
        local_device_id = self.rank_to_GPU[self.rank][0]
    
        # 定义 reduce_scatter_base 函数，用于执行 reduce_scatter 操作
        def reduce_scatter_base(output_t, input_t):
            work = pg._reduce_scatter_base(output_t, input_t)
            work.wait()
    
        # reduce_scatter_base 对 GPU 数量不敏感。
        # 每个进程贡献一个张量，与 GPU 数量无关
        output_t = torch.empty([1]).cuda(local_device_id)
        tensor = torch.arange(self.world_size, dtype=output_t.dtype).cuda(
            local_device_id
        )
    
        # 执行 reduce_scatter_base 操作
        reduce_scatter_base(output_t, tensor)
    
        # 验证操作的正确性
        self.assertEqual(output_t[0], self.rank * self.world_size)
    
    # 要求 NCCL 支持
    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_barrier(self):
        pg = self.pg
        local_device_ids = self.rank_to_GPU[self.rank]
    
        # 定义 allreduce 函数，执行全局归约操作
        def allreduce(tensors):
            opts = c10d.AllreduceOptions()
            work = pg.allreduce(tensors, opts)
            return work
    
        # 为每个 GPU 组合张量，以便执行 collective 操作
        tensors_list = [[] for _ in range(len(local_device_ids))]
        for i in range(1, len(local_device_ids) + 1):
            for j in range(i):
                tensors_list[i - 1].append(
                    torch.tensor([j + 1]).cuda(local_device_ids[j])
                )
    
        works = []
        for tensors in tensors_list:
            work = allreduce(tensors)
            works.append(work)
    
        # 执行 barrier 操作以确保之前的所有操作完成
        pg.barrier().wait()
    
        # 验证操作的正确性
        for i in range(1, len(local_device_ids) + 1):
            for j in range(i):
                self.assertEqual(
                    torch.tensor([(j + 1) * self.world_size]), tensors_list[i - 1][j]
                )
    
    # 要求 NCCL 支持
    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_send_recv(self):
        pg = self.pg
        device = self.rank_to_GPU[self.rank][0]
    
        # 设置随机种子，生成随机张量
        torch.manual_seed(0)
        send_tensor = torch.rand(10, 10, device=device)
    
        # 如果是 rank 0 进程，发送张量到 rank 1 进程
        if self.rank == 0:
            dist.send(send_tensor, 1)
    
        # 如果是 rank 1 进程，接收来自 rank 0 进程的张量，并进行验证
        if self.rank == 1:
            recv_tensor = torch.rand(10, 10, device=device)
            dist.recv(recv_tensor, 0)
            self.assertEqual(send_tensor, recv_tensor)
    def test_send_recv_complex(self):
        # 获取当前测试的进程组对象
        pg = self.pg
        # 获取当前进程对应的设备
        device = self.rank_to_GPU[self.rank][0]

        # 生成相同的随机复数张量
        torch.manual_seed(0)
        send_tensor = torch.rand(10, 10, dtype=torch.cfloat, device=device)
        # 如果当前进程是rank 0，发送张量
        if self.rank == 0:
            dist.send(send_tensor, 1)
        # 如果当前进程是rank 1，接收张量并验证
        if self.rank == 1:
            recv_tensor = torch.rand(10, 10, dtype=torch.cfloat, device=device)
            dist.recv(recv_tensor, 0)
            self.assertEqual(send_tensor, recv_tensor)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_send_recv_object_list(self):
        # 获取当前进程对应的设备
        device = self.rank_to_GPU[self.rank][0]

        # 根据rank设置值，构建对象列表
        val = 99 if self.rank == 0 else None
        object_list = [val] * self.world_size
        # 如果当前进程是rank 0，发送对象列表
        if self.rank == 0:
            dist.send_object_list(object_list, 1, device=device)
        # 如果当前进程是rank 1，接收对象列表并验证
        if self.rank == 1:
            dist.recv_object_list(object_list, 0, device=device)
            self.assertEqual(object_list[0], 99)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_tensor_register_hook(self):
        # 设置环境变量以启用张量注册分配器钩子
        os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] = "1"

        # 获取当前测试的进程组对象
        pg = self.pg
        # 获取当前进程对应的本地设备 ID
        local_device_id = self.rank_to_GPU[self.rank][0]

        # 定义一个函数来进行全局收集
        def allgather_base(output_t, input_t):
            # 调用进程组的全局收集函数，并等待完成
            work = pg._allgather_base(output_t, input_t)
            work.wait()

        # allgather_base函数不依赖于GPU数量。
        # 每个rank贡献一个张量，不考虑GPU数量
        tensor = torch.tensor([self.rank]).cuda(local_device_id)
        output_t = torch.empty((self.world_size), dtype=tensor.dtype).cuda(
            local_device_id
        )

        allgather_base(output_t, tensor)

        # 验证结果
        self.assertEqual(torch.arange(self.world_size), output_t)

        # 删除设置的环境变量
        del os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"]
if __name__ == "__main__":
    # 获取环境变量 RANK 的值并转换为整数，如果不存在则默认为 -1
    rank = int(os.getenv("RANK", -1))
    # 获取环境变量 WORLD_SIZE 的值并转换为整数，默认为 2
    world_size = int(os.getenv("WORLD_SIZE", 2))

    if rank != -1:
        # 如果 rank 不等于 -1，说明程序是由 torchrun 或其他多进程启动器启动的。
        # 直接运行测试的指定排名的进程组NCCLOp测试。
        ProcessGroupNCCLOpTest.run_rank(rank, world_size)
    else:
        # 如果 rank 等于 -1，说明程序是单进程启动的。
        # 需要创建子进程来运行测试，并为 `init_process_group` 函数提供一个会合文件。
        # 创建一个临时文件对象作为会合文件，并且不在关闭后删除。
        rdvz_file = tempfile.NamedTemporaryFile(delete=False).name
        # 使用 torch.multiprocessing.spawn 函数创建多个进程来运行指定函数。
        torch.multiprocessing.spawn(
            ProcessGroupNCCLOpTest.run_rank,
            nprocs=world_size,  # 指定创建的进程数
            args=(world_size, rdvz_file),  # 参数传递给 ProcessGroupNCCLOpTest.run_rank 函数
        )
```