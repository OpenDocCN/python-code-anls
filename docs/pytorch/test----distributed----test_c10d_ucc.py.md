# `.\pytorch\test\distributed\test_c10d_ucc.py`

```py
# 导入必要的库和模块
import copy  # 导入复制操作相关的模块
import logging  # 导入日志记录相关的模块
import math  # 导入数学计算相关的模块
import operator  # 导入操作符相关的模块
import os  # 导入操作系统相关的模块
import random  # 导入随机数生成相关的模块
import sys  # 导入系统相关的模块
import tempfile  # 导入临时文件处理相关的模块
from functools import reduce  # 导入函数工具相关的模块

import torch  # 导入PyTorch库
import torch.distributed as c10d  # 导入分布式通信相关的PyTorch库

# 如果分布式通信或者UCC不可用，则输出消息并退出程序
if not c10d.is_available() or not c10d.is_ucc_available():
    print("c10d UCC not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import test_c10d_common  # 导入自定义的分布式通信测试相关模块
from test_c10d_common import (  # 导入自定义的分布式通信测试相关的函数和类
    gpus_for_rank,
    ModuleForDdpCommHook,
    SparseGradientModule,
    Task,
)

import torch.distributed as dist  # 导入PyTorch分布式通信模块
import torch.nn.functional as F  # 导入PyTorch中的函数操作模块
import torch.testing._internal.common_utils as common  # 导入PyTorch内部的通用工具模块
from torch import nn  # 导入PyTorch中的神经网络模块
from torch.nn.parallel import DistributedDataParallel  # 导入分布式数据并行模块
from torch.testing._internal.common_distributed import (  # 导入PyTorch分布式通信的常用功能模块
    MultiProcessTestCase,
    requires_ucc,
    skip_if_lt_x_gpu,
    verify_ddp_error_logged,
)
from torch.testing._internal.common_utils import (  # 导入PyTorch内部的通用工具模块
    retry_on_connect_failures,
    run_tests,
    skip_but_pass_in_sandcastle,
    TestCase,
)


def simple_reduce_tests(rank, world_size):
    # 定义一系列简单的reduce操作的测试用例
    tests = [
        (
            c10d.ReduceOp.SUM,  # 使用SUM操作符进行求和
            torch.tensor([rank + 1.0]),  # 创建包含当前rank+1的张量
            torch.tensor([float(world_size * (world_size + 1) / 2)]),  # 期望的求和结果
        ),
        (
            c10d.ReduceOp.PRODUCT,  # 使用PRODUCT操作符进行乘积
            torch.tensor([rank + 1.0]),  # 创建包含当前rank+1的张量
            torch.tensor([float(math.factorial(world_size))]),  # 期望的乘积结果
        ),
        (
            c10d.ReduceOp.MIN,  # 使用MIN操作符进行最小值比较
            torch.tensor([rank + 1.0]),  # 创建包含当前rank+1的张量
            torch.tensor([1.0]),  # 期望的最小值结果
        ),
        (
            c10d.ReduceOp.MAX,  # 使用MAX操作符进行最大值比较
            torch.tensor([rank + 1.0]),  # 创建包含当前rank+1的张量
            torch.tensor([world_size]),  # 期望的最大值结果
        ),
    ]

    # 生成BAND操作的测试用例
    # 每次迭代中设置的位数变化，以检查输出是否相应变化
    for i in range(4):
        vin = rank | (1 << i)  # 构造输入vin，将当前rank和1左移i位进行OR运算
        vout = 1 << i  # 预期的输出结果，将1左移i位
        tests.append(
            (
                c10d.ReduceOp.BAND,  # 使用BAND操作符进行按位与运算
                torch.tensor([vin], dtype=torch.int32),  # 创建包含vin的整型张量
                torch.tensor([vout], dtype=torch.int32),  # 期望的按位与结果
            ),
        )

    # 生成BOR操作的测试用例
    # 每次迭代中，通过每个rank贡献多个进行预先OR运算的值来模拟更大的世界大小
    for i in range(1, 5):
        vin = reduce(operator.or_, [rank * i + j for j in range(i)])  # 构造输入vin，使用reduce和列表推导式进行OR运算
        vout = reduce(operator.or_, range(world_size * i))  # 预期的输出结果，使用reduce和range进行OR运算
        tests.append(
            (
                c10d.ReduceOp.BOR,  # 使用BOR操作符进行按位或运算
                torch.tensor([vin], dtype=torch.int32),  # 创建包含vin的整型张量
                torch.tensor([vout], dtype=torch.int32),  # 期望的按位或结果
            ),
        )

    # 生成XOR操作的测试用例
    # 每次迭代中，通过每个rank贡献多个预先XOR运算的值来模拟更大的世界大小
    # 循环执行4次，每次计算一个不同的vin和vout并添加到tests列表中
    for i in range(1, 5):
        # 计算vin：使用reduce函数对列表[rank * i + j for j in range(i)]进行按位异或运算
        vin = reduce(operator.xor, [rank * i + j for j in range(i)])
        
        # 计算vout：使用reduce函数对范围为range(world_size * i)的整数进行按位异或运算
        vout = reduce(operator.xor, range(world_size * i))
        
        # 将元组 (c10d.ReduceOp.BXOR, torch.tensor([vin], dtype=torch.int32), torch.tensor([vout], dtype=torch.int32)) 添加到tests列表中
        tests.append(
            (
                c10d.ReduceOp.BXOR,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )
    
    # 返回填充了四个不同vin和vout的tests列表
    return tests
class RendezvousEnvTest(TestCase):
    # 使用 UCC 协议运行测试
    @requires_ucc()
    # 在连接失败时进行重试
    @retry_on_connect_failures
    # 测试日志初始化功能
    def test_logging_init(self):
        # 设置环境变量 WORLD_SIZE 为 "1"
        os.environ["WORLD_SIZE"] = "1"
        # 设置环境变量 MASTER_ADDR 为 "127.0.0.1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        # 设置环境变量 MASTER_PORT 为找到的空闲端口号
        os.environ["MASTER_PORT"] = str(common.find_free_port())
        # 设置环境变量 RANK 为 "0"
        os.environ["RANK"] = "0"

        # 记录当前日志处理器
        previous_handlers = logging.root.handlers

        # 使用 UCC 后端和环境初始化方法初始化进程组
        c10d.init_process_group(backend="ucc", init_method="env://")

        # 获取当前的日志处理器
        current_handlers = logging.root.handlers
        # 断言：当前日志处理器与之前的日志处理器数量相同
        self.assertEqual(len(previous_handlers), len(current_handlers))
        # 逐个比较当前和之前的日志处理器是否相同
        for current, previous in zip(current_handlers, previous_handlers):
            self.assertEqual(current, previous)

        # 销毁进程组
        c10d.destroy_process_group()


class TimeoutTest(test_c10d_common.AbstractTimeoutTest, TestCase):
    # 使用 UCC 协议运行测试
    @requires_ucc()
    # 在连接失败时进行重试
    @retry_on_connect_failures
    # 测试默认存储超时设置（UCC）
    def test_default_store_timeout_ucc(self):
        # 调用基类方法进行测试
        self._test_default_store_timeout("ucc")


class ProcessGroupUCCTest(MultiProcessTestCase):
    # 创建 UCC 进程组
    def _create_process_group_ucc(self):
        # 使用文件存储和指定的进程数量创建文件存储器
        store = c10d.FileStore(self.file_name, self.world_size)
        # 返回 UCC 进程组对象
        return c10d.ProcessGroupUCC(store, self.rank, self.world_size)

    # 设置测试环境
    def setUp(self):
        super().setUp()
        # 生成多进程环境
        self._spawn_processes()

    # 清理测试环境
    def tearDown(self):
        super().tearDown()
        try:
            # 尝试移除文件
            os.remove(self.file_name)
        except OSError:
            pass

    # 使用 UCC 协议运行测试
    @requires_ucc()
    # 测试空张量操作
    def test_empty_tensors(self):
        # 创建 UCC 进程组
        pg = self._create_process_group_ucc()

        # 创建空 FloatTensor 列表
        xs = [torch.FloatTensor([])]
        # 广播张量并获取未来对象
        fut = pg.broadcast(xs).get_future()
        fut.wait()
        # 获取广播后的输出
        output = fut.value()
        # 断言：输出张量的元素数量为 0
        self.assertEqual(0, output[0].numel())
        # 断言：输出张量与原始输入张量相等（不精确比较数据类型）
        self.assertEqual(xs[0], output[0], exact_dtype=False)

    # TODO: 添加错误检查测试

    # 测试广播基础功能
    def _test_broadcast_basics(self, fn):
        # 创建 UCC 进程组
        pg = self._create_process_group_ucc()

        # 定义广播函数
        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            fut = pg.broadcast(xs, opts).get_future()
            fut.wait()
            return fut.value()

        # 每个进程都作为根节点运行一次
        for i in range(self.world_size):
            # 使用输入张量创建 x
            x = fn(torch.tensor([self.rank]))
            # 广播张量并获取输出
            output = broadcast([x], i, 0)
            # 断言：输出张量与预期值相等（不精确比较数据类型）
            self.assertEqual(torch.tensor([i]), output[0], exact_dtype=False)

            # TODO: UCC 目前不支持多张量输入

        # 测试重载的便捷函数
        x = torch.tensor([self.rank + 1.0])
        fut = pg.broadcast(x, root=0).get_future()
        fut.wait()
        result = fut.value()
        # 断言：结果与预期值相等
        self.assertEqual(torch.tensor([1.0]), result[0])

    # 使用 UCC 协议运行测试
    @requires_ucc()
    # 测试广播基础功能
    def test_broadcast_basics(self):
        # 测试广播基础功能，使用克隆函数
        self._test_broadcast_basics(lambda t: t.clone())

    # TODO: test_broadcast_basics_cuda 在本地超时
    # 定义测试所有reduce基本功能的方法，接受一个函数作为参数
    def _test_allreduce_basics(self, fn):
        # 创建 UCC 进程组
        pg = self._create_process_group_ucc()

        # 单输入测试
        tests = simple_reduce_tests(self.rank, self.world_size)
        for op, input, expected in tests:
            # 创建 AllreduceOptions 对象，设置操作类型
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            # 调用传入的函数处理输入数据
            tensor = fn(input)
            # 执行 allreduce 操作，并获取 Future 对象
            fut = pg.allreduce([tensor], opts).get_future()
            fut.wait()
            # 获取操作结果
            result = fut.value()
            # 断言期望结果与实际结果相等
            self.assertEqual(expected, result[0], exact_dtype=False)

        # TODO: UCC 目前不支持多张量输入

        # 测试重载的便捷函数（默认使用 sum 操作）
        x = fn(torch.tensor([self.rank + 1.0]))
        fut = pg.allreduce(x).get_future()
        fut.wait()
        result = fut.value()
        # 断言期望结果与实际结果相等
        self.assertEqual(
            torch.tensor([float(self.world_size * (self.world_size + 1) / 2)]),
            result[0],
        )

    @requires_ucc()
    # 测试 allreduce 基本功能
    def test_allreduce_basics(self):
        self._test_allreduce_basics(lambda t: t.clone())

    # TODO: test_allreduce_basics_cuda 在本地超时

    # 定义测试所有gather基本功能的方法，接受一个函数作为参数
    def _test_allgather_basics(self, fn):
        # 创建 UCC 进程组
        pg = self._create_process_group_ucc()

        # TODO: 使用每个排名的 N 个输入张量运行测试；目前，UCC 仅支持单个张量输入，所以 N=1
        for n in [1]:
            input = [fn(torch.tensor([n * self.rank + i])) for i in range(n)]
            output = [
                [fn(torch.tensor([-1])) for _ in range(n * self.world_size)]
                for _ in range(n)
            ]
            expected_output = [
                [fn(torch.tensor([i])) for i in range(n * self.world_size)]
                for _ in range(n)
            ]
            fut = pg.allgather(output, input).get_future()
            fut.wait()
            # 获取操作结果
            result = fut.value()
            if n == 1:
                result = [result]
            # 断言期望结果与实际结果相等
            self.assertEqual(expected_output, result)

    # 测试 allgather 基本功能
    def test_allgather_basics(self):
        self._test_allgather_basics(lambda t: t.clone())

    # 定义测试reduce基本功能的方法，接受一个函数作为参数
    def _test_reduce_basics(self, fn):
        # 创建 UCC 进程组
        pg = self._create_process_group_ucc()
        # 使用简单的reduce测试生成器
        for op, input, output in simple_reduce_tests(self.rank, self.world_size):
            for root in range(self.world_size):
                # 创建 ReduceOptions 对象，设置操作类型和根节点排名
                opts = c10d.ReduceOptions()
                opts.reduceOp = op
                opts.rootRank = root
                # 调用传入的函数处理输入数据
                tmp = fn(input)
                # 执行 reduce 操作，并获取 Future 对象
                fut = pg.reduce([tmp], opts).get_future()
                fut.wait()
                # 获取操作结果
                result = fut.value()
                # 如果是根节点，断言期望结果与实际结果相等
                if root == self.rank:
                    self.assertEqual(output, result[0], exact_dtype=False)

    @requires_ucc()
    # 测试 reduce 基本功能
    def test_reduce_basics(self):
        self._test_reduce_basics(lambda t: t.clone())

    # TODO: test_reduce_basics_cuda 在本地超时

    @requires_ucc()
    # 定义一个测试方法，用于测试全对全发送和接收功能
    def test_send_recv_all_to_all(self):
        # 创建 UCC 进程组
        pg = self._create_process_group_ucc()

        # 为输入和输出预分配张量
        inputs = [torch.tensor([self.rank]) for _ in range(self.world_size)]
        outputs = [torch.tensor([-1]) for _ in range(self.world_size)]

        # 发送操作
        send_work = []
        for i in range(self.world_size):
            # 跳过自己的发送
            if i == self.rank:
                continue
            send_work.append(pg.send([inputs[i]], i, 0))

        # 接收操作
        recv_work = []
        for i in range(self.world_size):
            # 跳过自己的接收
            if i == self.rank:
                continue
            recv_work.append(pg.recv([outputs[i]], i, 0))

        # 等待发送操作完成
        for work in send_work:
            work.wait()
            self.assertTrue(work.is_completed())

        # 等待接收操作完成
        for work in recv_work:
            work.wait()
            self.assertTrue(work.is_completed())

        # 测试除了自己之外的每个输出是否包含相应的进程排名
        for i in range(self.world_size):
            if i == self.rank:
                continue
            self.assertEqual(torch.tensor([i]), outputs[i])

    # TODO: test_barrier_implies_wait fails with numerical mismatch, will investigate later
    # 在沙盘中跳过但在实际环境中运行，原因是数值不匹配，稍后进行调查
    @skip_but_pass_in_sandcastle("fails with numerical mismatch, skip for now")
    @requires_ucc()
    def test_barrier_implies_wait(self):
        # 创建 UCC 进程组
        pg = self._create_process_group_ucc()

        # 启动 allreduce 操作
        size = (100, 100)
        num = 16
        tensors = [torch.full(size, float(i)) for i in range(num)]
        for tensor in tensors:
            # 注意: 泄露返回的工作句柄
            pg.allreduce(tensor)

        # Barrier 应确保所有先前的工作已完成
        pg.barrier().get_future().wait()

        for i, tensor in enumerate(tensors):
            self.assertEqual(torch.full(size, float(i * self.world_size)), tensor)
# 定义一个测试类，继承自CommonDistributedDataParallelTest和MultiProcessTestCase
class DistributedDataParallelTest(
    test_c10d_common.CommonDistributedDataParallelTest, MultiProcessTestCase
):
    # 设置测试环境
    def setUp(self):
        super().setUp()
        # 生成子进程来执行测试
        self._spawn_processes()

    # 获取进程组对象
    def _get_process_group(self):
        # 获取存储对象
        store = self._get_store()
        # 初始化进程组，使用 "ucc" 后端，传入存储对象、当前进程的排名和总进程数
        c10d.init_process_group(
            "ucc", store=store, rank=self.rank, world_size=self.world_size
        )
        # 返回默认的分布式进程组对象
        return c10d.distributed_c10d._get_default_group()

    # 测试使用 UCC 后端的分布式训练
    def _test_ucc_backend(
        self, devices, device_ids, multi_device=False, gradient_as_bucket_view=False
    ):
        # 获取当前进程组
        process_group = self._get_process_group()
        # 调用基类方法来测试 DDP（分布式数据并行）功能，传入进程组、设备列表、设备 ID 列表、多设备标志、梯度作为桶视图标志
        self._test_ddp_with_process_group(
            process_group, devices, device_ids, multi_device, gradient_as_bucket_view
        )

    # 使用 UCC 后端测试 CPU 模式
    @requires_ucc()
    def test_ucc_backend_cpu_module(self):
        self._test_ucc_backend([torch.device("cpu")], None)

    # 使用 UCC 后端测试 CPU 模式，并设置梯度作为桶视图
    @requires_ucc()
    def test_ucc_backend_cpu_module_grad_is_view(self):
        self._test_ucc_backend(
            [torch.device("cpu")], None, gradient_as_bucket_view=True
        )

    # 使用 UCC 后端测试单个 GPU 模式，设备 ID 为整数列表
    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_ucc_backend_1gpu_module_device_ids_integer_list(self):
        # 获取当前进程的 GPU 设备 ID 列表中的第一个设备 ID
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        # 将整数设备 ID 转换为 torch 设备对象列表
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        # 使用 UCC 后端测试 DDP
        self._test_ucc_backend(devices, int_devices)

    # 使用 UCC 后端测试单个 GPU 模式，设备 ID 为 torch 设备对象列表
    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_ucc_backend_1gpu_module_device_ids_torch_device_list(self):
        # 获取当前进程的 GPU 设备 ID 列表中的第一个设备 ID
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        # 将整数设备 ID 转换为 torch 设备对象列表
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        # 使用 UCC 后端测试 DDP
        self._test_ucc_backend(devices, devices)

    # TODO: test_ucc_backend_2gpu_module and test_ucc_backend_4gpu_module
    # 需要 broadcast_coalesced，目前 UCC 不支持该功能

    # 使用 UCC 后端测试双 GPU 模式，设备 ID 为整数列表
    @skip_but_pass_in_sandcastle(
        "requires broadcast coalesced, which is not supported by ucc currently"
    )
    @requires_ucc()
    @skip_if_lt_x_gpu(4)
    def test_ucc_backend_2gpu_module(self):
        # 获取当前进程的 GPU 设备 ID 列表中的前两个设备 ID
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        # 将整数设备 ID 转换为 torch 设备对象列表
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        # 使用 UCC 后端测试 DDP，启用多设备模式
        self._test_ucc_backend(devices, None, multi_device=True)

    # 使用 UCC 后端测试四 GPU 模式，设备 ID 为整数列表
    @skip_but_pass_in_sandcastle(
        "requires broadcast coalesced, which is not supported by ucc currently"
    )
    @requires_ucc()
    @skip_if_lt_x_gpu(8)
    def test_ucc_backend_4gpu_module(self):
        # 获取当前进程的 GPU 设备 ID 列表中的前四个设备 ID
        int_devices = gpus_for_rank(self.world_size)[self.rank][:4]
        # 将整数设备 ID 转换为 torch 设备对象列表
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        # 使用 UCC 后端测试 DDP，启用多设备模式
        self._test_ucc_backend(devices, None, multi_device=True)

    # 测试全局和局部未使用参数梯度
    def _test_global_local_unused_params_grad(
        self, gradient_as_bucket_view=False, static_graph=False
    ):
        """
        By simulating a multi-task training, this test is to make sure:
        1) DDP does not touch the grad of globally unused parameters.
        2) DDP does update the grad of locally unused parameters.
        """

        # 定义一个继承自 nn.Module 的模块，用于测试全局和局部未使用参数的梯度更新
        class GlobalLocalUnusedParamModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.t0 = Task()
                self.t1 = Task()
                self.task_unused = Task()

            def task_parameters(self):
                # 返回模块中的任务参数，包括 t0 和 t1 的参数
                return (self.t0.p, self.t1.p, self.task_unused.p)

            def forward(self, x, rank):
                # 根据 rank 执行不同的任务 t0 或 t1，并返回结果
                return self.t0(x) if rank == 0 else self.t1(x)

        def run_and_verify_grad(model):
            # 运行前向传播
            output = model(8, self.rank)

            # 此时所有参数的梯度应该为 None
            t0_p, t1_p, task_unused_p = model.module.task_parameters()
            self.assertIsNone(t0_p.grad)
            self.assertIsNone(t1_p.grad)
            self.assertIsNone(task_unused_p.grad)

            # 运行反向传播
            output.mean().backward()

            # 现在局部未使用的参数应该在所有 rank 上更新了梯度
            # 全局未使用的参数仍应该是 None
            self.assertIsNotNone(t0_p.grad)
            self.assertIsNotNone(t1_p.grad)
            self.assertIsNone(task_unused_p.grad)

        # 获取进程组
        process_group = self._get_process_group()

        # 在 CPU 上进行测试
        cpu_model = DistributedDataParallel(
            GlobalLocalUnusedParamModule().cpu(),
            process_group=process_group,
            find_unused_parameters=True,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )
        run_and_verify_grad(cpu_model)

        # 在 GPU 上进行测试
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            GlobalLocalUnusedParamModule().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            find_unused_parameters=True,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )
        run_and_verify_grad(gpu_model)

    # TODO: times out
    @skip_but_pass_in_sandcastle("times out")
    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad(self):
        # 调用 _test_global_local_unused_params_grad 方法进行全局和局部未使用参数梯度测试
        self._test_global_local_unused_params_grad()

    # TODO: times out
    @skip_but_pass_in_sandcastle("times out")
    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad_with_grad_is_view(self):
        # 调用 _test_global_local_unused_params_grad 方法进行全局和局部未使用参数梯度测试，
        # 并设置 gradient_as_bucket_view 为 True
        self._test_global_local_unused_params_grad(gradient_as_bucket_view=True)

    # TODO: times out
    @skip_but_pass_in_sandcastle("times out")
    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    # 调用内部方法 `_test_global_local_unused_params_grad`，并设置静态图模式为真
    def test_global_local_unused_params_grad_with_static_graph(self):
        self._test_global_local_unused_params_grad(static_graph=True)

    # 跳过此测试，并在沙盒环境中通过，原因是超时
    @skip_but_pass_in_sandcastle("times out")
    # 要求使用 UCC（通信库）进行测试
    @requires_ucc()
    # 如果 GPU 数量少于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_find_unused_parameters_when_unused_parameters_empty(self):
        """
        An empty unused_parameters array does not imply find_unused_parameters =
        false. This test makes sure that DDP allreduces unused parameters
        accordingly where the forward pass in some process uses all parameters.
        This unit test creates a module that uses all parameters in rank = 0, and
        has unused parameters in other ranks.
        """

        # 定义一个测试用的模块，包含两个任务对象
        class FindUnusedParamModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.t0 = Task()
                self.t1 = Task()

            # 返回两个任务对象的参数
            def task_parameters(self):
                return (self.t0.p, self.t1.p)

            # 模块的前向传播方法，根据 rank 决定使用哪个任务对象进行计算
            def forward(self, x, rank):
                return self.t1(self.t0(x)) if rank == 0 else self.t1(x)

        # 验证模型的梯度更新情况
        def run_and_verify_grad(model):
            # 运行前向传播
            output = model(8, self.rank)

            # 此时所有参数的梯度应为 None
            [self.assertIsNone(t_p.grad) for t_p in model.module.task_parameters()]

            # 运行反向传播
            output.mean().backward()

            # 现在未使用的参数应在所有进程中更新梯度
            [self.assertIsNotNone(t_p.grad) for t_p in model.module.task_parameters()]

        # 获取当前进程组
        process_group = self._get_process_group()

        # 在 CPU 上进行测试
        cpu_model = DistributedDataParallel(
            FindUnusedParamModule().cpu(),
            process_group=process_group,
            find_unused_parameters=True,
        )
        run_and_verify_grad(cpu_model)

        # 在 GPU 上进行测试
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            FindUnusedParamModule().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            find_unused_parameters=True,
        )
        run_and_verify_grad(gpu_model)

    # 要求使用 UCC（通信库）进行测试
    @requires_ucc()
    def test_ignored_output(self):
        """
        Test that the output of a model can be ignored and that there is no
        implicit requirement that `backward` gets called.
        """
        # 获取进程组对象
        process_group = self._get_process_group()

        # 定义一个忽略输出的模型类
        class IgnoredOutput(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义第一个全连接层，输入维度为2，输出维度为10，无偏置
                self.fc1 = nn.Linear(2, 10, bias=False)
                # 定义第二个全连接层，输入维度为10，输出维度为4，无偏置
                self.fc2 = nn.Linear(10, 4, bias=False)
                # 定义ReLU激活函数
                self.relu = nn.ReLU()

            def forward(self, x):
                # 前向传播函数
                x = self.relu(self.fc1(x))  # 第一层全连接 + ReLU
                x = self.relu(self.fc2(x))  # 第二层全连接 + ReLU
                return F.softmax(x, dim=1)  # 对最后一层输出进行softmax归一化

        # 创建分布式数据并行模型，使用忽略输出的模型，将其转换为浮点数
        model = DistributedDataParallel(
            IgnoredOutput().float(),
            process_group=process_group,
        )

        # 定义批处理大小
        batch_size = 4
        # 定义交叉熵损失函数
        criterion = nn.CrossEntropyLoss()
        # 创建随机输入张量，形状为[batch_size, 2]
        input = torch.rand([batch_size, 2], dtype=torch.float)
        # 创建随机目标张量，形状为[batch_size]
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])

        # 运行几个迭代，其中忽略输出结果
        for _ in range(4):
            output = model(input)  # 模型前向传播
            del output  # 删除输出张量，忽略输出结果

        # 运行几个迭代，其中使用输出结果计算损失并进行反向传播
        for _ in range(4):
            output = model(input)  # 模型前向传播
            loss = criterion(output, target)  # 计算损失
            loss.backward()  # 反向传播计算梯度

    @requires_ucc()
    def test_ignored_output_with_unused_parameters(self):
        """
        Test that the output of a model can be ignored and that there is no
        implicit requirement that `backward` gets called, if not all model
        parameters participated in computing the model output.
        """
        # 获取进程组对象
        process_group = self._get_process_group()

        # 定义一个带有未使用参数的忽略输出模型类
        class IgnoredOutputWithUnusedParameters(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义第一个全连接层，输入维度为2，输出维度为10，无偏置
                self.fc1 = nn.Linear(2, 10, bias=False)
                # 定义第二个全连接层，输入维度为10，输出维度为4，无偏置
                self.fc2 = nn.Linear(10, 4, bias=False)
                # 定义第三个全连接层，输入维度为4，输出维度为4，无偏置
                self.fc3 = nn.Linear(4, 4, bias=False)
                # 定义ReLU激活函数
                self.relu = nn.ReLU()

            def forward(self, x):
                # 前向传播函数
                x = self.relu(self.fc1(x))  # 第一层全连接 + ReLU
                x = self.relu(self.fc2(x))  # 第二层全连接 + ReLU
                return F.softmax(x, dim=1)  # 对最后一层输出进行softmax归一化

        # 创建分布式数据并行模型，使用带有未使用参数的忽略输出模型，将其转换为浮点数
        model = DistributedDataParallel(
            IgnoredOutputWithUnusedParameters().float(),
            process_group=process_group,
            find_unused_parameters=True,
        )

        # 定义批处理大小
        batch_size = 4
        # 定义交叉熵损失函数
        criterion = nn.CrossEntropyLoss()
        # 创建随机输入张量，形状为[batch_size, 2]
        input = torch.rand([batch_size, 2], dtype=torch.float)
        # 创建随机目标张量，形状为[batch_size]
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])

        # 运行几个迭代，其中忽略输出结果
        for _ in range(4):
            output = model(input)  # 模型前向传播
            del output  # 删除输出张量，忽略输出结果

        # 运行几个迭代，其中使用输出结果计算损失并进行反向传播
        for _ in range(4):
            output = model(input)  # 模型前向传播
            loss = criterion(output, target)  # 计算损失
            loss.backward()  # 反向传播计算梯度
    def _run_and_verify_sparse_gradients(self, vanilla_model, ddp_model):
        # 定义倍数，用于确定批次大小
        mult = 2
        # 计算批次大小，考虑多进程数
        batch_size = mult * self.world_size
        # 定义交叉熵损失函数
        criterion = nn.CrossEntropyLoss()
        # 创建随机整数张量作为输入数据
        input = torch.randint(0, 10, [batch_size, 2])
        # 创建随机整数张量作为目标标签
        target = torch.randint(0, 10, [batch_size])

        # 使用整个批次数据对单进程版本的模型进行前向传播、损失计算和反向传播
        criterion(vanilla_model(input), target).backward()

        # 使用部分批次数据对多进程版本的模型进行前向传播、损失计算和反向传播
        partial_input = input.split(mult)[self.rank]
        partial_target = target.split(mult)[self.rank]
        criterion(ddp_model(partial_input), partial_target).backward()

        # 检查梯度是否稀疏且相同
        vanilla_parameter = next(vanilla_model.parameters())
        ddp_parameter = next(ddp_model.parameters())
        self.assertEqual(
            vanilla_parameter.grad.coalesce(), ddp_parameter.grad.coalesce()
        )

    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def _test_sparse_gradients(self, gradient_as_bucket_view=False):
        # 获取进程组
        process_group = self._get_process_group()

        # 设置随机种子以确保各进程中初始化的权重和输入数据相同
        torch.manual_seed(1337)

        # 创建稀疏梯度模型和分布式数据并行模型
        vanilla_model = SparseGradientModule()
        ddp_model = DistributedDataParallel(
            copy.deepcopy(vanilla_model),
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # 运行并验证稀疏梯度计算
        self._run_and_verify_sparse_gradients(vanilla_model, ddp_model)

    # TODO: backward pass: input tensor has to be dense
    @skip_but_pass_in_sandcastle("backward pass: input tensor has to be dense")
    @requires_ucc()
    def test_sparse_gradients(self):
        # 执行稀疏梯度测试
        self._test_sparse_gradients()

    # TODO: backward pass: input tensor has to be dense
    @skip_but_pass_in_sandcastle("backward pass: input tensor has to be dense")
    @requires_ucc()
    def test_sparse_gradients_grad_is_view(self):
        # 执行视图梯度的稀疏梯度测试
        self._test_sparse_gradients(gradient_as_bucket_view=True)

    @requires_ucc()
    def test_ddp_comm_hook_future_passing_cpu(self):
        """
        This unit test verifies whether the Future object is passed properly.
        The callback function creates a Future object and sets a value to it.
        """
        # 获取进程组
        process_group = self._get_process_group()

        # 在 CPU 上测试分布式数据并行模型
        cpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().cpu(), process_group=process_group
        )

        # 注册 DDP 通信钩子
        cpu_model.register_comm_hook(None, self._simple_hook)

        # 检查梯度是否与回调函数返回的值相等
        # 如果没有使用通信钩子，结果应为 0.25 * torch.ones(2, 2)
        self._run_and_verify_hook(cpu_model, 8, 2 * torch.ones(2, 2))

    def _gpu_model_with_ddp_comm_hook(
        self, process_group, hook=None, gradient_as_bucket_view=False, state=None
    ):
        # 省略部分代码...
    ):
        # 根据当前进程的排名和总GPU数量，选择设备ID
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        # 创建带有分布式数据并行的GPU模型
        gpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().to(device_id),  # 将模型移动到指定设备
            device_ids=[device_id],  # 指定使用的设备ID列表
            process_group=process_group,  # 指定进程组
            gradient_as_bucket_view=gradient_as_bucket_view,  # 梯度作为桶视图
        )

        # 如果提供了hook，注册DDP通信钩子
        if hook is not None:
            gpu_model.register_comm_hook(state, hook)

        # 返回配置好的GPU模型
        return gpu_model

    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_future_passing_gpu_ucc(self):
        """
        This unit test verifies whether the Future object is passed properly using ucc backend.
        The hook callback function creates a Future object and sets a value to it.
        """
        process_group = self._get_process_group()

        # 获取带有_simple_hook注册的GPU模型
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, self._simple_hook)

        # 检查梯度是否等于_simple_hook的回调函数返回的值
        # 如果没有comm_hook，结果应为0.25乘以torch.ones(2, 2)
        self._run_and_verify_hook(gpu_model, 8, 2 * torch.ones(2, 2))

    @requires_ucc()
    def test_ddp_invalid_comm_hook_init(self):
        """
        This unit test makes sure that register_comm_hook properly checks the format
        of hook defined by user. The Python hook must be callable. This test also
        checks whether bucket annotation checked properly if defined.
        """
        process_group = self._get_process_group()

        # 创建带有分布式数据并行的模型，但未指定通信钩子
        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        # 检查是否正确抛出TypeError异常，如果hook不可调用
        with self.assertRaisesRegex(TypeError, "Communication hook must be callable."):
            model.register_comm_hook(state=None, hook=1)

        # 检查是否正确抛出ValueError异常，如果bucket注释未正确定义为dist.GradBucket
        with self.assertRaisesRegex(
            ValueError, "bucket annotation should be dist.GradBucket."
        ):

            def comm_hook(
                state: object, bucket: int
            ) -> torch.futures.Future[torch.Tensor]:
                return torch.futures.Future()

            # 尝试注册comm_hook，但bucket注释未正确定义
            model.register_comm_hook(state=None, hook=comm_hook)

    @requires_ucc()
    def test_ddp_invalid_comm_hook_return_type(self):
        """
        This test checks whether return annotation checked properly if defined. It also
        checks whether an internal error is thrown if return type is incorrect and user
        hasn't specified any return type annotation.
        """
        # 获取进程组
        process_group = self._get_process_group()

        # 创建 DistributedDataParallel 模型
        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        # 期望的错误信息
        expected_err = (
            "Communication hook: return annotation should be torch.futures.Future"
        )
        # 断言捕获异常并验证错误信息
        with self.assertRaisesRegex(
            ValueError,
            expected_err,
        ):
            # 定义通信钩子函数 comm_hook，声明参数类型和返回类型
            def comm_hook(state: object, bucket: dist.GradBucket) -> int:
                return torch.futures.Future()

            # 注册通信钩子函数到模型
            model.register_comm_hook(state=None, hook=comm_hook)

        # 验证 DDP 错误日志中是否包含期望的错误信息
        verify_ddp_error_logged(model, expected_err)

        # 断言捕获异常并验证错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            "callback must return a torch.futures.Future object, but got",
        ):
            # 定义通信钩子函数 comm_hook，声明参数类型但未指定返回类型
            def comm_hook(state: object, bucket: dist.GradBucket):
                return 1

            # 注册通信钩子函数到模型
            model.register_comm_hook(state=None, hook=comm_hook)

            # 运行前向传播
            output = model(8, self.rank)

            # 运行反向传播
            output.mean().backward()

    @requires_ucc()
    def test_ddp_comm_hook_register_just_once(self):
        """
        DDP communication hook can only be registered once. This test validates whether
        the error is thrown properly when register_comm_hook is called more than once.
        """
        # 获取进程组
        process_group = self._get_process_group()

        # 创建 DistributedDataParallel 模型
        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        # 定义一个空的钩子函数 dummy_hook
        def dummy_hook(state, bucket):
            fut = torch.futures.Future()
            fut.set_result([bucket.buffer()])
            return fut

        # 注册 dummy_hook 作为通信钩子函数到模型
        model.register_comm_hook(None, dummy_hook)

        # 断言捕获异常并验证错误信息
        with self.assertRaisesRegex(
            RuntimeError,
            "register_comm_hook or register_builtin_comm_hook can only be called once.",
        ):
            # 尝试再次注册相同的通信钩子函数 dummy_hook，应触发异常
            model.register_comm_hook(None, dummy_hook)

    # TODO: backward pass: input tensor must be dense
    @skip_but_pass_in_sandcastle("backward pass: input tensor has to be dense")
    @requires_ucc()
    def test_ddp_comm_hook_sparse_gradients(self):
        """
        Runs "test_sparse_gradients" unit test with DDP communication hook. We define a
        simple hook that does allreduce and works with ucc backend for this test.
        """
        # 获取当前进程组，用于分布式训练
        process_group = self._get_process_group()

        # 确保在所有进程中初始化的权重和输入是相同的
        torch.manual_seed(1337)

        # 创建一个原始的稀疏梯度模型
        vanilla_model = SparseGradientModule()

        # 使用分布式数据并行包装原始模型，传入进程组
        ddp_model = DistributedDataParallel(
            copy.deepcopy(vanilla_model),
            process_group=process_group,
        )

        # 定义一个基于ucc后端的allreduce通信钩子函数
        def allreduce_hook_ucc(
            state: object, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:
            # 定义一个函数，用于异步操作完成后将结果除以2 * world_size
            def div_by_world_size(fut):
                return fut.wait()[0] / self.world_size

            # 使用进程组进行梯度的allreduce操作，并获取其Future对象
            fut = process_group.allreduce([bucket.buffer()]).get_future()
            # 将结果通过指定的函数进行处理
            return fut.then(div_by_world_size)

        # 将定义的通信钩子函数注册到DDP模型中
        ddp_model.register_comm_hook(None, allreduce_hook_ucc)

        # 运行并验证稀疏梯度的操作
        self._run_and_verify_sparse_gradients(vanilla_model, ddp_model)
# 定义一个测试类，继承自AbstractCommTest和MultiProcessTestCase类
class CommTest(test_c10d_common.AbstractCommTest, MultiProcessTestCase):

    # 属性装饰器，返回设备类型为"cpu"
    @property
    def device(self):
        return "cpu"

    # 初始化方法，在每个测试方法执行前调用
    def setUp(self):
        super().setUp()  # 调用父类的setUp方法
        self._spawn_processes()  # 调用自定义方法_spawn_processes()

    # 清理方法，在每个测试方法执行后调用
    def tearDown(self):
        super().tearDown()  # 调用父类的tearDown方法
        try:
            os.remove(self.file_name)  # 尝试删除self.file_name指定的文件
        except OSError:
            pass  # 如果出现OSError异常，直接跳过

    # 装饰器函数，要求当前测试依赖UCC
    # 要求至少有2个GPU，否则跳过测试
    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_default_pg_ucc(self):
        self._test_sequence_num_set_default_pg(backend="ucc")

    # 装饰器函数，要求当前测试依赖UCC
    # 要求至少有2个GPU，否则跳过测试
    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_ucc_new_group(self):
        self._test_sequence_num_set_new_group(backend="ucc")

    # 装饰器函数，要求当前测试依赖UCC
    # 要求至少有2个GPU，否则跳过测试
    def test_sequence_num_incremented_ucc_default(self):
        self._test_sequence_num_incremented_default_group("ucc")

    # 装饰器函数，要求当前测试依赖UCC
    # 要求至少有4个GPU，否则跳过测试
    def test_sequence_num_incremented_ucc_subgroup(self):
        # 如果world_size小于4，跳过测试但在Sandcastle中标记为通过
        if self.world_size < 4:
            return skip_but_pass_in_sandcastle("Test requires world_size of at least 4")
        self._test_sequence_num_incremented_subgroup("ucc")

    # 装饰器函数，标记在M60上会失败
    # 要求当前测试依赖UCC
    def test_ucc_barrier_device_ids(self):
        # 使用c10d.FileStore创建存储，world_size为self.world_size
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用UCC后端，指定rank和world_size，使用store作为存储
        c10d.init_process_group(
            backend="ucc", rank=self.rank, world_size=self.world_size, store=store
        )
        # 使用断言验证RuntimeError中是否包含"device_ids not supported"信息
        with self.assertRaisesRegex(RuntimeError, "device_ids not supported"):
            c10d.barrier(device_ids=[self.rank])

    # 装饰器函数，标记在M60上会失败
    # 要求至少有2个GPU，要求当前测试依赖UCC
    def test_ucc_warn_not_in_group(self):
        self._test_warn_not_in_group(backend="ucc")

    # 装饰器函数，要求至少有2个GPU，要求当前测试依赖UCC
    def test_ucc_rank_membership(self):
        self._test_rank_membership(backend="ucc")

    # 装饰器函数，要求至少有2个GPU，要求当前测试依赖UCC
    def test_tensor_dtype_mismatch(self):
        self._test_tensor_dtype_mismatch(backend="ucc")

    # 装饰器函数，要求至少有2个GPU，要求当前测试依赖UCC
    def test_tensor_dtype_complex(self):
        self._test_tensor_dtype_complex(backend="ucc")


# 测试类，继承自CompilerTest类
class CompilerTest(test_c10d_common.CompilerTest):

    # 属性装饰器，返回world_size为2
    @property
    def world_size(self):
        return 2

    # 私有方法，获取默认进程组
    def _get_default_group(self):
        # 使用c10d.FileStore创建存储，world_size为self.world_size
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用UCC后端，指定rank和world_size，使用store作为存储
        dist.init_process_group(
            backend="ucc",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        # 返回默认组的引用
        return dist.distributed_c10d._get_default_group()

    # 装饰器函数，要求至少有2个GPU
    def test_allreduce_work_wait_gpu(self):
        self._test_allreduce_work_wait(
            torch.ones(2, 2, device=self.rank) * self.rank,
        )

    # 装饰器函数，要求至少有2个GPU
    def test_allgather_work_wait_gpu(self):
        self._test_allgather_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    # 装饰器函数，要求至少有2个GPU
    # 测试广播操作在 GPU 上的工作和等待
    def test_broadcast_work_wait_gpu(self):
        self._test_broadcast_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)
    
    # 如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_nested_comm_tensor_wrapping_gpu(self):
        # 测试嵌套通信和张量包装在 GPU 上的操作
        self._test_nested_comm_tensor_wrapping(
            torch.ones(2, 2, device=self.rank) * self.rank
        )
    
    # 测试连续通信的工作和等待在 GPU 上的操作
    @skip_if_lt_x_gpu(2)
    def test_consecutive_comm_work_wait_gpu(self):
        self._test_consecutive_comm_work_wait(
            torch.ones(2, 2, device=self.rank) * self.rank
        )
    
    # 测试所有规约操作在 CPU 上的工作和等待
    def test_allreduce_work_wait_cpu(self):
        self._test_allreduce_work_wait(
            torch.ones(2, 2) * self.rank,
        )
    
    # 测试所有聚合操作在 CPU 上的工作和等待
    def test_allgather_work_wait_cpu(self):
        self._test_allgather_work_wait(torch.ones(2, 2) * self.rank)
    
    # 测试广播操作在 CPU 上的工作和等待
    def test_broadcast_work_wait_cpu(self):
        self._test_broadcast_work_wait(torch.ones(2, 2) * self.rank)
    
    # 测试嵌套通信和张量包装在 CPU 上的操作
    def test_nested_comm_tensor_wrapping_cpu(self):
        self._test_nested_comm_tensor_wrapping(torch.ones(2, 2) * self.rank)
    
    # 测试连续通信的工作和等待在 CPU 上的操作
    def test_consecutive_comm_work_wait_cpu(self):
        self._test_consecutive_comm_work_wait(torch.ones(2, 2) * self.rank)
class UccProcessGroupWithDispatchedCollectivesTests(
    test_c10d_common.ProcessGroupWithDispatchedCollectivesTests
):
    @skip_but_pass_in_sandcastle("Fails on M60")
    @requires_ucc()
    @skip_if_lt_x_gpu(1)
    def test_collectives(self):
        # 使用 UCC 后端进行集体通信测试
        # 包括 reduce、broadcast、all_reduce、all_gather、reduce_scatter、barrier、all_to_all、scatter
        self._test_collectives(backend="ucc")

    @skip_but_pass_in_sandcastle("Fails on M60")
    @requires_ucc()
    @skip_if_lt_x_gpu(1)
    def test_allgather_base(self):
        # 创建文件存储，用于分布式进程组初始化
        store = dist.FileStore(self.file_name, self.world_size)
        # 使用 UCC 后端初始化进程组
        dist.init_process_group(
            "ucc",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 指定使用 CUDA 设备
        device = "cuda"
        # 创建一个在 CUDA 设备上的全为1的张量
        tensor = torch.ones(10, 10, device=torch.device(device))
        # 创建一个在 CUDA 设备上的全为0的输出张量
        output_tensor = torch.zeros(10, 10, device=torch.device(device))
        # 执行 all_gather 操作，将各进程的张量收集到输出张量中
        dist.all_gather_into_tensor(output_tensor, tensor)
        # 断言输出张量与输入张量相等
        self.assertEqual(output_tensor, tensor)


if __name__ == "__main__":
    # 断言在主进程中 CUDA 上下文未初始化
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    # 运行测试
    run_tests()
```