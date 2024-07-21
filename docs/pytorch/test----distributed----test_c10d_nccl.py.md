# `.\pytorch\test\distributed\test_c10d_nccl.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入所需的模块和库
import copy  # 导入深拷贝模块
import json  # 导入 JSON 模块
import os  # 导入操作系统模块
import pickle  # 导入 pickle 序列化模块
import random  # 导入随机数模块
import re  # 导入正则表达式模块
import signal  # 导入信号处理模块
import sys  # 导入系统模块
import tempfile  # 导入临时文件模块
import threading  # 导入线程模块
import time  # 导入时间模块
import warnings  # 导入警告模块
from contextlib import contextmanager  # 导入上下文管理器模块
from datetime import datetime, timedelta  # 导入日期时间模块
from enum import auto, Enum  # 导入枚举模块
from itertools import chain, product  # 导入迭代器模块
from unittest import mock, SkipTest  # 导入单元测试模块

import torch  # 导入 PyTorch 深度学习框架
import torch.distributed as c10d  # 导入分布式支持模块 c10d

# 如果 c10d 或者 NCCL 不可用，输出信息并退出程序
if not c10d.is_available() or not c10d.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from typing import Dict, List  # 导入类型提示模块

import test_c10d_common  # 导入自定义测试模块 test_c10d_common
from test_c10d_common import (  # 导入特定类和函数
    ConvNet,
    DoubleGpuNet,
    gpus_for_rank,
    ModuleForDdpCommHook,
)

import torch.distributed as dist  # 导入分布式支持模块 dist
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default  # 导入默认通信钩子模块 default
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD  # 导入powerSGD通信钩子模块
import torch.nn.functional as F  # 导入神经网络函数模块 F
import torch.testing._internal.common_utils as common  # 导入测试常用工具模块 common
from torch import nn  # 导入神经网络模块 nn
from torch._C._distributed_c10d import OpType  # 导入分布式操作类型模块 OpType
from torch.nn.parallel import DistributedDataParallel  # 导入分布式数据并行模块 DistributedDataParallel
from torch.testing._internal.common_cuda import TEST_MULTIGPU  # 导入多 GPU 测试模块 TEST_MULTIGPU
from torch.testing._internal.common_distributed import (  # 导入分布式测试模块
    get_timeout,
    init_multigpu_helper,
    MultiProcessTestCase,
    requires_gloo,
    requires_nccl,
    requires_nccl_version,
    skip_if_lt_x_gpu,
    skip_if_rocm,
    TEST_SKIPS,
    with_dist_debug_levels,
    with_nccl_blocking_wait,
)
from torch.testing._internal.common_utils import (  # 导入测试常用工具模块
    instantiate_parametrized_tests,
    parametrize,
    retry_on_connect_failures,
    run_tests,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
    TEST_CUDA,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_WITH_ROCM,
    TestCase,
)

# 如果启用了 ASAN 调试，输出信息并退出程序
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues", file=sys.stderr
    )
    sys.exit(0)

# 检查是否支持 bfloat16，仅 CUDA 11+ 支持
BFLOAT16_AVAILABLE = torch.cuda.is_available() and (
    (torch.version.cuda is not None and int(torch.version.cuda.split(".")[0]) >= 11)
    or torch.version.hip is not None
)


class RendezvousEnvTest(TestCase):
    # 测试类：RendezvousEnvTest，继承自 TestCase

    @retry_on_connect_failures
    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "No GPUs available, skipping test")
    # 装饰器：重试连接失败，需要 NCCL 支持，如果没有 GPU 可用则跳过测试

class TimeoutTest(test_c10d_common.AbstractTimeoutTest, TestCase):
    # 测试类：TimeoutTest，继承自 AbstractTimeoutTest 和 TestCase

    @requires_nccl()
    @retry_on_connect_failures
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "No GPUs available, skipping test")
    # 装饰器：需要 NCCL 支持，重试连接失败，如果没有 GPU 可用则跳过测试

    def test_default_store_timeout_nccl(self):
        # 测试方法：test_default_store_timeout_nccl

        self._test_default_store_timeout("nccl")
        # 调用内部方法 _test_default_store_timeout，传入参数 "nccl"


class ProcessGroupNCCLNoGPUTest(TestCase):
    # 测试类：ProcessGroupNCCLNoGPUTest，继承自 TestCase

    MAIN_PROCESS_RANK = 0

    def setUp(self):
        # 设置测试环境

        self.rank = self.MAIN_PROCESS_RANK
        self.world_size = 1
        self.file = tempfile.NamedTemporaryFile(delete=False)

    def tearDown(self):
        # 清理测试环境

        pass

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(TEST_CUDA, "GPUs are available, skipping test")
    # 装饰器：需要 NCCL 支持，如果有 GPU 可用则跳过测试
    # 定义测试方法，用于测试在没有 GPU 的情况下初始化
    def test_init_no_gpus(self):
        # 使用 c10d.FileStore 初始化一个文件存储对象，传入文件名和世界大小参数
        store = c10d.FileStore(self.file.name, self.world_size)
        # 使用断言检查是否抛出 ValueError 异常，并且异常信息包含特定字符串
        with self.assertRaisesRegex(
            ValueError, "ProcessGroupNCCL is only supported with GPUs, no GPUs found!"
        ):
            # 尝试创建 c10d.ProcessGroupNCCL 对象，传入存储对象、进程编号和世界大小参数
            c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
    # 定义一个测试类，继承自 MultiProcessTestCase
    class ProcessGroupNCCLGroupTest(MultiProcessTestCase):

        # 创建一个 NCCL 进程组的方法，使用指定的存储、选项和设备 ID
        def _create_process_group_nccl(self, store, opts, device_id=None):
            # 使用 c10d.init_process_group 初始化 NCCL 进程组
            c10d.init_process_group(
                "nccl",
                world_size=self.world_size,
                rank=self.rank,
                store=store,
                pg_options=opts,
                device_id=device_id,
            )
            # 获取默认的 NCCL 进程组
            pg = c10d.distributed_c10d._get_default_group()
            return pg

        # 创建一个 ProcessGroupNCCL 的选项对象，可以指定是否使用高优先级流
        def opts(self, high_priority_stream=False):
            opts = c10d.ProcessGroupNCCL.Options()
            opts.is_high_priority_stream = high_priority_stream
            return opts

        # 设置测试环境，在调用父类的 setUp 方法前跳过返回码检查
        def setUp(self):
            super().setUp()
            # 由于某些 CUDA 版本中子进程不会正常退出，需要跳过返回码检查
            self.skip_return_code_checks = [
                self.test_nan_assert_float16.__wrapped__,
                self.test_nan_assert_float32.__wrapped__,
                self.test_nan_assert_float64.__wrapped__,
            ]

            # 设置环境变量 TORCH_NCCL_ASYNC_ERROR_HANDLING 为 "1"，覆盖 TORCH_NCCL_BLOCKING_WAIT 的设置
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
            # 调用 _spawn_processes 方法来启动测试中需要的进程
            self._spawn_processes()

        # 清理测试环境，在调用父类的 tearDown 方法后尝试删除文件
        def tearDown(self):
            super().tearDown()
            try:
                os.remove(self.file_name)
            except OSError:
                pass

        # 返回世界大小，这里固定为 2
        @property
        def world_size(self):
            return 2

        # 返回一个映射关系，将 rank 映射到 GPU 设备上，使用 nccl 后端
        @property
        def rank_to_GPU(self):
            return init_multigpu_helper(self.world_size, "nccl")

        # 测试函数，需要 NCCL 后端的支持，跳过条件不符合的情况
        @requires_nccl()
        @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 1 GPU")
        @skip_if_lt_x_gpu(1)
        def test_nccl_dist_backend_error(self):
            # 创建一个文件存储，用于进程组通信
            store = c10d.FileStore(self.file_name, self.world_size)
            # 创建 NCCL 进程组并传入相应的选项
            self._create_process_group_nccl(store, self.opts())

            # 在当前进程组中广播一个 CUDA 张量，期望抛出 dist.DistBackendError 异常
            with self.assertRaises(dist.DistBackendError) as cm:
                dist.broadcast(torch.tensor([1, 2, 3]).cuda(), 0)
            # 断言捕获的异常类型为 RuntimeError
            self.assertTrue(isinstance(cm.exception, dist.DistError))

            # 再次断言捕获的异常类型为 RuntimeError
            self.assertIsInstance(cm.exception, RuntimeError)

        # 测试函数，需要 NCCL 后端的支持，跳过条件不符合的情况
        @requires_nccl()
        @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_abort_pg(self):
        # 在这个测试中禁用 ASYNC_ERROR_HANDLING，以确保可以编程方式中止进程组。
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

        # 创建一个文件存储，用于进程组通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建并初始化一个 NCCL 进程组
        self._create_process_group_nccl(store, self.opts())
        # 获取当前进程的 GPU 设备
        device = self.rank_to_GPU[self.rank][0]

        # 在指定设备上创建一个大小为 10x10 的随机张量
        t = torch.rand(10, 10, device=device)
        # 执行首次全局归约操作，用于初始化状态
        dist.all_reduce(t)

        def abortpg():
            # 获取默认的分布式组，并关闭与指定设备相关的后端
            c10d.distributed_c10d._get_default_group()._get_backend(
                torch.device(device)
            )._shutdown()

        # 初始化 DDP 以确保在调用 "destroy_process_group" 时不会调用 ProcessGroupNCCL 析构函数，
        # 因为 DDP 会持有进程组的引用。运行一次 DDP 的单次迭代以初始化状态。
        model = DistributedDataParallel(
            torch.nn.Linear(10, 10).to(device), device_ids=[device]
        )
        model(t).sum().backward()

        # 现在模拟集体操作卡住的情况，调用中止操作可以使我们解除阻塞
        if self.rank == 0:
            dist.all_reduce(t)

            # 在我们被卡住之前安排一个线程来中止进程组
            thread = threading.Thread(target=abortpg)
            thread.start()

            # 如果没有中止，由于 d2h，我们会在这里被卡住。
            t_cpu = t.cpu()

            thread.join()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_close_pg(self):
        # 在这个测试中禁用 ASYNC_ERROR_HANDLING，以确保可以编程方式中止进程组。
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

        # 创建一个文件存储，用于进程组通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建并初始化一个 NCCL 进程组
        pg = self._create_process_group_nccl(store, self.opts())
        # 获取当前进程的 GPU 设备
        device = self.rank_to_GPU[self.rank][0]

        # 在指定设备上创建一个大小为 10x10 的随机张量
        t = torch.rand(10, 10, device=device)
        # 执行首次全局归约操作，用于初始化状态
        pg.allreduce(t)

        # 销毁进程组并验证进程组不再有效
        dist.destroy_process_group()
        with self.assertRaises(dist.DistBackendError):
            # 尝试在已销毁的进程组上执行 allreduce 操作，应该引发异常
            pg.allreduce([t])

        # 释放进程组对象
        del pg

    CUDA_12_AND_ABOVE = torch.cuda.is_available() and (
        torch.version.cuda is not None and int(torch.version.cuda.split(".")[0]) >= 12
    )

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(
        not (TEST_MULTIGPU and CUDA_12_AND_ABOVE),
        "NCCL test requires 2+ GPUs and Device side assert could cause unexpected errors in lower versions of CUDA",
    )
    @parametrize("type", [torch.float16, torch.float32, torch.float64])
    @skip_if_rocm
    # 定义一个测试方法，用于测试处理 NaN 值的断言
    def test_nan_assert(self, type):
        # 设置环境变量，启用 Torch NCCL 的 NaN 检查
        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        # 创建一个基于文件的存储，用于进程组的通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建 NCCL 进程组
        pg = self._create_process_group_nccl(store, self.opts())
        # 确定当前进程的 GPU 设备
        device = self.rank_to_GPU[self.rank][0]
        # 创建一个指定类型和设备的张量，填充为当前进程的 rank
        size = (10, 10)
        nan_tensor = torch.full(size, self.rank, dtype=type, device=device)
        # 在张量中随机选择一个元素，将其设置为 NaN
        i = random.randint(0, nan_tensor.size(0) - 1)
        j = random.randint(0, nan_tensor.size(1) - 1)
        nan_tensor[i, j] = float("nan")
        # 使用断言确保在进行全局归约时会抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            pg.allreduce(nan_tensor)
        # 销毁进程组
        dist.destroy_process_group()
        # 恢复环境变量的设置，禁用 Torch NCCL 的 NaN 检查
        os.environ["TORCH_NCCL_NAN_CHECK"] = "0"

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    # 定义一个测试方法，用于测试在终止进程组之前销毁对象的情况
    def test_destruct_before_terminate_pg(self):
        # 为了确保可以编程地中止进程组，禁用此测试的异步错误处理
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        # 创建一个基于文件的存储，用于进程组的通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建 NCCL 进程组
        pg = self._create_process_group_nccl(store, self.opts())
        # 确定当前进程的 GPU 设备
        device = self.rank_to_GPU[self.rank][0]

        t = torch.rand(10, 10, device=device)
        # 执行一次全局归约，以初始化状态
        pg.allreduce(t)
        # 强制在终止通信之前销毁进程组对象
        del pg

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    # 定义一个测试方法，用于测试在销毁进程组中中止操作的情况
    def test_abort_in_destroy_pg(self):
        # 为了确保可以编程地中止进程组，禁用此测试的异步错误处理
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

        # 创建一个基于文件的存储，用于进程组的通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建 NCCL 进程组
        pg = self._create_process_group_nccl(store, self.opts())
        # 确定当前进程的 GPU 设备
        device = self.rank_to_GPU[self.rank][0]

        t = torch.rand(10, 10, device=device)
        # 执行一次全局归约，以初始化状态
        pg.allreduce(t)

        # 销毁进程组并验证由于通信已关闭，进程组不再处于工作状态
        dist.destroy_process_group()
        with self.assertRaises(dist.DistBackendError):
            pg.allreduce([t])

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(
        torch.cuda.device_count() < 2, "NCCL test requires 2+ GPUs"
    )
    # 定义一个测试方法，用于在测试多 GPU 环境时跳过测试但在沙堡中通过
    def test_multigpu(self):
        pass
    def test_close_multi_pg_unordered(self):
        # 创建一个文件存储对象，用于进程组通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建一个新的 NCCL 进程组
        pg = self._create_process_group_nccl(store, self.opts())
        # 获取当前进程对应的 GPU 设备
        device = self.rank_to_GPU[self.rank][0]
        # 在指定设备上生成一个 10x10 的随机张量
        t = torch.rand(10, 10, device=device)
        # 执行第一次全局归约操作，用于初始化默认进程组的通信器
        pg.allreduce(t).wait()
        # 创建两个包含进程组 [0, 1] 的新进程组
        new_pg1 = c10d.new_group([0, 1])
        new_pg2 = c10d.new_group([0, 1])
        # 如果当前进程的排名是 0 或 1
        if self.rank == 0 or self.rank == 1:
            # 在当前设备上生成两个 10x10 的随机张量
            t1 = torch.rand(10, 10, device=device)
            t2 = torch.rand(10, 10, device=device)
            # 对 new_pg1 和 new_pg2 分别执行全局归约操作
            new_pg1.allreduce(t1).wait()
            new_pg2.allreduce(t2).wait()
        # 如果当前进程的排名是 0
        if self.rank == 0:
            # 销毁 new_pg2 进程组
            dist.destroy_process_group(new_pg2)
            # 强制删除 new_pg2 对象
            del new_pg2
            # 销毁 new_pg1 进程组
            dist.destroy_process_group(new_pg1)
            # 强制删除 new_pg1 对象
            del new_pg1
        # 如果当前进程的排名是 1
        if self.rank == 1:
            # 销毁 new_pg1 进程组
            c10d.destroy_process_group(new_pg1)
            # 强制删除 new_pg1 对象
            del new_pg1
            # 销毁 new_pg2 进程组
            dist.destroy_process_group(new_pg2)
            # 强制删除 new_pg2 对象
            del new_pg2
        # 销毁默认的进程组
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(
        torch.cuda.device_count() < 2, "NCCL test requires 2+ GPUs"
    )
    def test_abort_in_destroy_multi_pgs(self):
        # 创建一个文件存储对象，用于进程组通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建一个新的 NCCL 进程组
        pg = self._create_process_group_nccl(store, self.opts())
        # 获取当前进程对应的 GPU 设备
        device = self.rank_to_GPU[self.rank][0]
        # 在指定设备上生成一个 10x10 的随机张量
        t = torch.rand(10, 10, device=device)
        # 执行第一次全局归约操作，用于初始化默认进程组的通信器
        pg.allreduce(t).wait()
        # 创建两个包含进程组 [0, 1] 的新进程组
        new_pg1 = c10d.new_group([0, 1])
        new_pg2 = c10d.new_group([0, 1])
        # 在当前设备上生成两个 10x10 的随机张量
        t1 = torch.rand(10, 10, device=device)
        t2 = torch.rand(10, 10, device=device)
        # 对 new_pg1 和 new_pg2 分别执行全局归约操作
        new_pg1.allreduce(t1).wait()
        new_pg2.allreduce(t2).wait()
        # 获取默认进程组的后端，并验证其 comm_split_count 为 2
        backend = pg._get_backend(torch.device(device))
        self.assertEqual(backend.comm_split_count(), 2)
        # 关闭所有 NCCL 进程组的通信
        dist.destroy_process_group()
    def test_abort_in_destroy_mixed_empty_pgs(self):
        # 使用指定的文件名和进程数创建文件存储对象
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用指定的存储对象和选项创建新的进程组对象
        pg = self._create_process_group_nccl(store, self.opts())
        # 根据当前进程的排名获取对应的 GPU 设备
        device = self.rank_to_GPU[self.rank][0]
        # 在指定的 GPU 设备上创建一个 10x10 的随机张量
        t = torch.rand(10, 10, device=device)
        # 执行第一次全局归约操作，初始化默认进程组的通信器
        pg.allreduce(t).wait()
        # 创建一个不含通信器初始化的进程组 PG1，因为没有对其调用集体操作
        new_pg1 = c10d.new_group([0, 1])
        new_pg2 = c10d.new_group([0, 1])
        t2 = torch.rand(10, 10, device=device)

        # 对新进程组 new_pg2 执行全局归约操作
        new_pg2.allreduce(t2).wait()
        # 获取默认进程组的后端，应验证其通信分割计数为1
        backend = pg._get_backend(torch.device(device))
        self.assertEqual(backend.comm_split_count(), 1)
        # 关闭所有的 NCCL 进程组
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(
        torch.cuda.device_count() < 2, "NCCL test requires 2+ GPUs"
    )
    def test_file_store_check(self):
        # 设置环境变量 TORCH_NCCL_ASYNC_ERROR_HANDLING 为 "0"
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        # 设置环境变量 TORCH_NCCL_ENABLE_MONITORING 为 "0"
        os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
        # 设置环境变量 TORCH_NCCL_DUMP_ON_TIMEOUT 为 "1"
        os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
        # 设置环境变量 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC 为 "0"

        # 创建一个文件存储对象，使用指定的文件名和进程数
        # self.file_name 使用 "delete=False" 创建
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化分布式进程组，使用 NCCL 后端，指定当前进程的排名和总进程数，使用上述创建的存储对象
        dist.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )
        # 获取默认的分布式进程组对象
        pg = dist.distributed_c10d._get_default_group()
        # 验证当前进程在进程组中的排名
        self.assertEqual(pg.rank(), self.rank)
        # 验证进程组中的进程总数
        self.assertEqual(pg.size(), self.world_size)
        # 等待足够的时间以执行多次 check() 操作
        time.sleep(2)
        # 关闭所有的 NCCL 进程组
        dist.destroy_process_group()

    def _check_nccl_timeout(self, expected_timeout):
        # 获取默认的分布式进程组对象
        pg = dist.distributed_c10d._get_default_group()
        # 获取指定 CUDA 设备的后端选项
        options = pg._get_backend(torch.device(f"cuda:{self.rank}")).options
        # 验证选项中的超时值是否与期望值相等
        self.assertEqual(options._timeout, expected_timeout)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "No GPUs available, skipping test")
    # 定义一个测试方法，用于初始化 NCCL 过程组并测试超时相关功能
    def test_init_process_group_nccl_timeout(self):
        # 在 init_process_group 中，nccl 被特殊处理，其选项类别与其他过程组不同。需要测试 nccl 的特定边界情况。

        # 使用 c10d.FileStore 创建一个存储对象，用于分布式进程组通信
        store = c10d.FileStore(self.file_name, self.world_size)
        
        # 基础选项字典，指定 backend 为 "nccl"，存储为上面创建的 store，指定当前进程的 rank 和总的 world_size
        base_opts = dict(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )

        # 测试来自 init_process_group 默认参数的超时值
        dist.init_process_group(**base_opts)
        # 检查 NCCL 的超时时间是否为默认值
        self._check_nccl_timeout(torch.distributed.constants.default_pg_nccl_timeout)
        dist.destroy_process_group()

        # 测试通过 `timeout` 关键字参数设置的超时值是否生效
        new_timeout = timedelta(seconds=123)
        dist.init_process_group(**base_opts, timeout=new_timeout)
        # 检查 NCCL 的超时时间是否为新设置的值
        self._check_nccl_timeout(new_timeout)
        dist.destroy_process_group()

        # 测试通过 `pg_options` 关键字参数设置的超时值被忽略并发出警告，以 `timeout` 关键字参数设置的值为准
        opts = dist.ProcessGroupNCCL.Options()
        opts._timeout = timedelta(seconds=123)
        with warnings.catch_warnings(record=True) as w:
            dist.init_process_group(**base_opts, pg_options=opts)
            # TODO(whc) i verified that we are indeed emitting this warning, and i can't figure out why i can't catch it.
            # self.assertEqual(len(w), 1)
            # self.assertTrue("pg_options._timeout was specified" in str(w[-1].message))
        # 检查 NCCL 的超时时间是否恢复为默认值，并且发出了警告
        self._check_nccl_timeout(torch.distributed.constants.default_pg_nccl_timeout)
        dist.destroy_process_group()

        # 测试通过 `pg_options` 关键字参数设置的超时值被忽略并发出警告，以 `timeout` 关键字参数设置的值为准
        opts = dist.ProcessGroupNCCL.Options()
        opts._timeout = timedelta(seconds=123)
        dist.init_process_group(
            **base_opts, pg_options=opts, timeout=timedelta(seconds=1240)
        )
        # 检查 NCCL 的超时时间是否设置为新指定的值
        self._check_nccl_timeout(timedelta(seconds=1240))
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("backend", [None, "nccl"])
    # 定义一个测试方法，用于设置 NCCL 过程组的超时时间
    def test_set_nccl_pg_timeout(self, backend):
        # 创建一个基于文件的存储对象，指定文件名和世界大小
        store = c10d.FileStore(self.file_name, self.world_size)
        # 构建参数字典，包括后端类型、存储对象、进程组的排名和世界大小，以及超时时间为 123 秒
        opts = dict(
            backend=backend,
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=123),
        )
        # 初始化进程组，根据参数字典 opts
        dist.init_process_group(**opts)
        # 获取默认的分布式进程组
        pg = dist.distributed_c10d._get_default_group()
        # 对 CUDA 设备上的随机张量进行全局归约操作
        pg.allreduce(torch.rand(10).cuda(self.rank))
        # 检查 NCCL 超时是否设置为 123 秒
        self._check_nccl_timeout(timedelta(seconds=123))
        # 获取指定 CUDA 设备的后端，设置默认超时时间为 23 秒
        pg._get_backend(torch.device(f"cuda:{self.rank}"))._set_default_timeout(
            timedelta(seconds=23)
        )
        # 再次检查 NCCL 超时是否设置为 23 秒
        self._check_nccl_timeout(timedelta(seconds=23))
        # 对 CUDA 设备上的随机张量进行全局归约操作
        pg.allreduce(torch.rand(10).cuda(self.rank))
        # 设置进程组的超时时间为 252 秒
        c10d.distributed_c10d._set_pg_timeout(timedelta(seconds=252), pg)
        # 最后检查 NCCL 超时是否设置为 252 秒
        self._check_nccl_timeout(timedelta(seconds=252))

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    # 如果 CUDA NCCL 版本为 NCCLX，则跳过测试
    @skip_but_pass_in_sandcastle_if(
        torch.cuda.nccl.version()[-1] == "x", "NCCL test not for NCCLX"
    )
    # 测试通信分组优化
    def test_comm_split_optimization(self):
        # 创建基于文件的存储对象，指定文件名和世界大小
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建 NCCL 进程组，使用指定的存储对象和选项
        pg = self._create_process_group_nccl(store, self.opts())

        # 测试在每个设备的后端上进行懒惰分组行为
        for device in self.rank_to_GPU[self.rank]:
            # 获取指定设备的后端对象
            backend = pg._get_backend(torch.device(device))

            # 只有在原始进程组懒惰创建通信器时才会进行分组，因此首先验证在创建新组并在原始 pg 上运行操作时未进行分组
            ng = c10d.new_group()
            # 创建一个张量并在指定设备上广播
            tensor = torch.tensor([self.rank]).cuda(device)
            pg.broadcast(tensor, 0)
            # 断言分组计数为 0
            self.assertEqual(backend.comm_split_count(), 0)

            # 新组将在首次使用时强制对原始组进行分组
            ng.broadcast(tensor, 0)
            # 断言分组计数为 1
            self.assertEqual(backend.comm_split_count(), 1)
    def test_comm_split_subgroup(self):
        # 测试在通过特定设备 ID 初始化进程组时，使用 `ncclCommSplit` 创建世界的较小子组。
        store = c10d.FileStore(self.file_name, self.world_size)
        # 根据当前进程的排名选择对应的 CUDA 设备
        device = torch.device(f"cuda:{self.rank}")
        # 创建 NCCL 进程组，使用给定的存储、选项和设备 ID
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        # 获取当前进程组的后端（backend）
        backend = pg._get_backend(torch.device(device))

        # 创建一个在 CUDA 设备上全为当前进程排名的张量
        tensor = torch.full((1,), self.rank).cuda(device)
        original_tensor = tensor.clone()
        # 创建一个新的进程组包含单个进程 0
        ng = c10d.new_group([0])

        # 由于设备 ID 被传递给 init_process_group，comm split 会立即发生。
        self.assertEqual(backend.comm_split_count(), 1)
        if self.rank == 0:
            # 如果当前进程是 0，则广播张量到进程组 ng
            dist.broadcast(tensor, 0, group=ng)

        # 在集体操作之后不会再发生额外的 comm split。
        self.assertEqual(backend.comm_split_count(), 1)
        self.assertEqual(tensor, original_tensor)

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_non_blocking_init(self):
        # 测试使用非阻塞模式创建进程组，但不是立即执行
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
        os.environ["TORCH_NCCL_NONBLOCKING_TIMEOUT"] = "100"
        # 使用文件存储和世界大小创建存储对象
        store = c10d.FileStore(self.file_name, self.world_size)
        # 根据当前进程的排名选择对应的 GPU 设备
        device = self.rank_to_GPU[self.rank][0]
        # 创建 NCCL 进程组，使用给定的存储和选项
        pg = self._create_process_group_nccl(store, self.opts())
        # 获取当前进程组的后端（backend）
        backend = pg._get_backend(torch.device(device))
        # 初始时没有 comm split 发生
        self.assertEqual(backend.comm_split_count(), 0)
        # 创建一个在指定设备上的随机张量
        reduce_tensor = torch.rand(10, 10, device=device)
        # 执行一个全reduce操作，这将触发进程组的 comm 初始化
        pg.allreduce(reduce_tensor).wait()
        # 创建一个新的进程组
        new_pg = c10d.new_group()
        # 即使在 pg 的集体调用之后，新 pg 的 comm 直到自己的集体调用之前也不会被初始化
        self.assertEqual(backend.comm_split_count(), 0)
        # 在新进程组上广播张量到进程 0
        broadcast_tensor = torch.tensor([self.rank]).cuda(device)
        new_pg.broadcast(broadcast_tensor, 0).wait()
        # 确保 comm split 发生了一次
        self.assertEqual(backend.comm_split_count(), 1)
    def test_non_blocking_with_eager_init(self):
        # Test creating a process group eagerly with nonblocking mode when
        # we've passed a specific device_id to init_process_group.
        
        # 设置环境变量以启用 Torch NCCL 的非阻塞通信模式
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
        # 设置环境变量以定义非阻塞通信的超时时间
        os.environ["TORCH_NCCL_NONBLOCKING_TIMEOUT"] = "100"
        
        # 创建一个文件存储对象，用于多进程间的通信
        store = c10d.FileStore(self.file_name, self.world_size)
        
        # 指定设备为当前进程的 CUDA 设备
        device = torch.device(f"cuda:{self.rank}")
        
        # 绑定设备以触发急切初始化模式
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        
        # 获取当前进程组的后端类型
        backend = pg._get_backend(torch.device(device))
        
        # 断言当前进程组的通信分裂数量为 0
        self.assertEqual(backend.comm_split_count(), 0)
        
        # 创建一个在指定设备上的随机张量，用于后续的全reduce操作
        reduce_tensor = torch.rand(10, 10, device=device)
        
        # 执行全reduce操作，但只有在初始化成功后才会将操作提交到 CUDA 流
        pg.allreduce(reduce_tensor).wait()
        
        # 创建一个新的进程组
        new_pg = c10d.new_group()
        
        # 即使在当前进程组的集体调用后，新进程组的通信也不会在其自身的集体调用之前初始化
        self.assertEqual(backend.comm_split_count(), 0)
        
        # 创建一个在指定设备上的张量，用于广播操作
        broadcast_tensor = torch.tensor([self.rank]).cuda(device)
        
        # 执行广播操作
        new_pg.broadcast(broadcast_tensor, 0).wait()
        
        # 断言当前进程组的通信分裂数量为 1
        self.assertEqual(backend.comm_split_count(), 1)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_get_uid(self):
        # 创建一个文件存储对象，用于多进程间的通信
        store = c10d.FileStore(self.file_name, self.world_size)
        
        # 指定设备为当前进程的 CUDA 设备
        device = torch.device(f"cuda:{self.rank}")
        
        # 创建一个新的 NCCL 进程组
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        
        # 导入私有函数 _get_process_group_uid 来获取进程组的唯一标识符
        from torch.distributed.distributed_c10d import _get_process_group_uid
        
        # 断言获取到的进程组 UID 为 0
        self.assertEqual(_get_process_group_uid(pg), 0)
        
        # 创建一个包含多个进程的新进程组
        pg_2 = c10d.new_group([0, 1])
        
        # 断言获取到的进程组 UID 为 1
        self.assertEqual(_get_process_group_uid(pg_2), 1)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_set_process_group_desc(self):
        # 创建一个文件存储对象，用于多进程间的通信
        store = c10d.FileStore(self.file_name, self.world_size)
        
        # 指定设备为当前进程的 CUDA 设备
        device = torch.device(f"cuda:{self.rank}")
        
        # 创建默认名称的 NCCL 进程组
        pg_default = self._create_process_group_nccl(
            store, self.opts(), device_id=device
        )
        
        # 断言默认进程组的描述为 "default_pg"
        self.assertEqual(pg_default.group_desc, "default_pg")
        
        # 创建一个自定义描述的新进程组
        pg_1 = c10d.new_group([0, 1], group_desc="test_purpose")
        
        # 断言新进程组的描述为 "test_purpose"
        self.assertEqual(pg_1.group_desc, "test_purpose")
        
        # 创建一个未命名的新进程组
        pg_2 = c10d.new_group([0, 1])
        
        # 断言新进程组的描述为 "undefined"
        self.assertEqual(pg_2.group_desc, "undefined")
class DistributedDataParallelTest(
    test_c10d_common.CommonDistributedDataParallelTest, MultiProcessTestCase
):
    def setUp(self):
        super().setUp()
        # 设置环境变量以便测试 TORCH_NCCL_BLOCKING_WAIT 功能，覆盖 TORCH_NCCL_ASYNC_ERROR_HANDLING 的设置
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # 启动多进程来进行测试
        self._spawn_processes()

    def _get_process_group(self):
        # 获取存储对象
        store = self._get_store()
        # 初始化 NCCL 进程组
        c10d.init_process_group(
            "nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        return c10d.distributed_c10d._get_default_group()

    def _test_nccl_backend(
        self, devices, device_ids, multi_device=False, gradient_as_bucket_view=False
    ):
        # 获取进程组对象
        process_group = self._get_process_group()
        # 使用给定的进程组对象进行 DDP 测试
        self._test_ddp_with_process_group(
            process_group, devices, device_ids, multi_device, gradient_as_bucket_view
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_propagate_error_reason(self):
        # 设置环境变量以便测试 TORCH_NCCL_BLOCKING_WAIT 功能，而不是 ASYNC_ERROR_HANDLING，
        # 否则进程会被终止，无法检查错误
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        # 创建文件存储对象
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建 NCCL 进程组对象并设置超时时间
        pg = c10d.ProcessGroupNCCL(
            store, self.rank, self.world_size, timeout=timedelta(seconds=15)
        )
        # 创建 GLOO 进程组对象
        pg_gloo = c10d.ProcessGroupGloo(store, self.rank, self.world_size)
        # 执行进程组的 barrier 操作并等待一段时间
        pg.barrier().wait(timedelta(seconds=5))
        # 模拟 rank 0 进程的阻塞状态
        if self.rank == 0:
            pg_gloo.barrier().wait()
        # 在 GPU 上创建输入张量
        inp = torch.ones(1).cuda(self.rank)

        if self.rank != 0:
            # 由于 rank 0 没有调用 allreduce，因此超时
            with self.assertRaises(dist.DistBackendError):
                pg.allreduce([inp]).wait(timedelta(seconds=5))

            # 现在如果非零 rank 尝试使用通信器，则应记录原始失败原因
            try:
                pg.allreduce([torch.ones(2).cuda(self.rank)]).wait()
            except dist.DistBackendError as e:
                self.assertTrue("aborted" in str(e))
            else:
                self.fail("Expected error to be raised!")

            # 解除 rank 0 的阻塞状态
            pg_gloo.barrier().wait()

        # TODO: We can also test that if rank 0 attempts to use the communicator,
        # then we should error out with the info that it was aborted due to
        # timeout on another rank. Although this would only be the case after
        # the watchdog has run on the rank, and there is no reliable way
        # to confirm it has run.

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    # 测试在不允许多设备 ID 的情况下使用 NCCL 后端
    def test_nccl_backend_multi_device_ids_not_allowed(self):
        # 生成 GPU 设备索引列表
        int_devices = list(range(torch.cuda.device_count()))
        # 创建对应的 Torch 设备列表
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        # 断言引发 ValueError，其错误信息包含 "device_ids can only be None or contain a single element."
        with self.assertRaisesRegex(
            ValueError, "device_ids can only be None or contain a single element."
        ):
            # 调用测试函数，传入设备列表和设备索引列表
            self._test_nccl_backend(devices, int_devices)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    # 测试在单设备模块中，使用设备 ID 为 None 的情况
    def test_nccl_backend_single_device_module_device_ids_None(self):
        self._test_nccl_backend(None, None)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    # 测试在单设备模块中，使用空设备 ID 列表的情况
    def test_nccl_backend_single_device_module_empty_device_ids(self):
        # 这里测试了接受空列表作为 `device_ids` 的向后兼容性，
        # 尽管我们现在不再推荐此用法，而是推荐使用默认值 `None`，
        # 这与多设备模块和 CPU 模块的默认行为一致。
        self._test_nccl_backend(None, [])

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    # 测试在多设备模块中，使用设备 ID 为 None 的情况
    def test_nccl_backend_multi_device_module_device_ids_None(self):
        # 获取当前进程需要使用的 GPU 设备索引列表的前两个
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        # 创建对应的 Torch 设备列表
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        # 调用测试函数，传入设备列表和设备 ID 为 None，指定为多设备模式
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    # 测试在单 GPU 模块中，使用整数列表作为设备 ID 的情况
    def test_nccl_backend_1gpu_module_device_ids_integer_list(self):
        # 获取当前进程需要使用的 GPU 设备索引列表的第一个
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        # 创建对应的 Torch 设备列表
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        # 调用测试函数，传入设备列表和设备索引列表
        self._test_nccl_backend(devices, int_devices)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    # 测试在单 GPU 模块中，使用 Torch 设备列表作为设备 ID 的情况
    def test_nccl_backend_1gpu_module_device_ids_torch_device_list(self):
        # 获取当前进程需要使用的 GPU 设备索引列表的第一个
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        # 创建对应的 Torch 设备列表
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        # 调用测试函数，传入设备列表和 Torch 设备列表
        self._test_nccl_backend(devices, devices)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    # 测试在双 GPU 模块中使用 NCCL 后端
    def test_nccl_backend_2gpu_module(self):
        # 获取当前进程需要使用的前两个 GPU 设备索引列表
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        # 创建对应的 Torch 设备列表
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        # 调用测试函数，传入设备列表和设备 ID 为 None，指定为多设备模式
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(8)
    # 测试在四 GPU 模块中使用 NCCL 后端
    def test_nccl_backend_4gpu_module(self):
        # 获取当前进程需要使用的前四个 GPU 设备索引列表
        int_devices = gpus_for_rank(self.world_size)[self.rank][:4]
        # 创建对应的 Torch 设备列表
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        # 调用测试函数，传入设备列表和设备 ID 为 None，指定为多设备模式
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    # 定义一个测试函数，用于测试多设备模块配置
    def test_ddp_multi_device_module_config(self):
        # 获取当前进程可用的 GPU 列表
        gpus = gpus_for_rank(self.world_size)[self.rank]

        # 断言至少有两个 GPU 可以使用
        self.assertTrue(len(gpus) >= 2, "expecting at least 2 gpus per process")

        # 获取进程组
        process_group = self._get_process_group()

        # 选取前两个 GPU
        gpus = gpus[:2]

        # 创建一个双 GPU 网络模型
        model = DoubleGpuNet(gpus)

        # 测试使用 DistributedDataParallel 初始化模型时的错误处理
        with self.assertRaisesRegex(
            ValueError,
            "DistributedDataParallel device_ids and output_device arguments only work with "
            "single-device/multiple-device GPU modules or CPU modules",
        ):
            # 尝试使用 DistributedDataParallel 初始化模型，设定输出设备为第二个 GPU
            ddp_model = DistributedDataParallel(
                model, output_device=gpus[1], process_group=process_group
            )

        # 测试使用 DistributedDataParallel 初始化模型时的错误处理
        with self.assertRaisesRegex(
            ValueError, "device_ids can only be None or contain a single element."
        ):
            # 尝试使用 DistributedDataParallel 初始化模型，设定多个设备 ID
            ddp_model = DistributedDataParallel(
                model, device_ids=gpus, process_group=process_group
            )

        # 测试使用 DistributedDataParallel 初始化模型时的错误处理
        with self.assertRaisesRegex(
            ValueError, "input module must be on the same type of devices"
        ):
            # 将模型的第一个全连接层移到 CPU 上
            model.fc1 = model.fc1.cpu()
            # 尝试使用 DistributedDataParallel 初始化模型
            ddp_model = DistributedDataParallel(model, process_group=process_group)

        # 将模型移到 CPU 上
        model = model.cpu()

        # 测试使用 DistributedDataParallel 初始化模型时的错误处理
        with self.assertRaisesRegex(
            ValueError, "device_ids can only be None or contain a single element."
        ):
            # 尝试使用 DistributedDataParallel 初始化模型，设定多个设备 ID
            ddp_model = DistributedDataParallel(
                model, device_ids=gpus, process_group=process_group
            )

    # 定义一个测试函数，用于测试混合精度训练
    def _test_fp16(self, gradient_as_bucket_view=False):
        # 获取进程组
        process_group = self._get_process_group()

        # 获取当前进程可用的 GPU 列表
        gpus = gpus_for_rank(self.world_size)[self.rank]

        # 创建一个在第一个 GPU 上运行的半精度线性模型
        model = nn.Linear(1, 1, bias=False).cuda(gpus[0]).half()
        nn.init.constant_(model.weight, 1)

        # 使用 DistributedDataParallel 初始化模型，设定设备 ID 为第一个 GPU
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[gpus[0]],
            process_group=process_group,
            bucket_cap_mb=0.001,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # 创建输入张量，将其放在第一个 GPU 上，类型为半精度
        input = torch.tensor([[2**15]]).cuda(gpus[0]).half()

        # 将模型设为训练模式
        ddp_model.train()

        # 对输入进行模型前向计算
        output = ddp_model(input)

        # 计算损失并反向传播
        loss = output.sum()
        loss.backward()

        # 断言模型的梯度中不含有无穷大值
        self.assertFalse(any(torch.isinf(p.grad).any() for p in ddp_model.parameters()))
    # 调用内部函数 _test_arbitrary_forward_return_value，设置 gradient_as_bucket_view 参数为 True 进行测试
    def test_arbitrary_forward_return_value_grad_is_view(self):
        self._test_arbitrary_forward_return_value(gradient_as_bucket_view=True)

    # 使用 requires_nccl 装饰器，要求至少有两个 GPU，跳过测试如果 GPU 数量不足
    def test_ddp_with_lazy_parameters(self):
        # 获取进程组对象
        process_group = self._get_process_group()
        # 使用断言确保在使用未初始化参数的模块时会引发 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError, "Modules with uninitialized parameters"
        ):
            # 使用 DistributedDataParallel 尝试包装一个 LazyLinear 模块，传入进程组对象
            DistributedDataParallel(
                torch.nn.LazyLinear(10), process_group=process_group
            )

    # TODO: Combine the following tests once https://github.com/pytorch/pytorch/issues/55967
    # is resolved.
    # 使用 requires_nccl 装饰器，要求至少有两个 GPU，同时设置分布式调试级别为 "DETAIL"
    def test_find_unused_parameters_kwarg_debug_detail(self):
        # 调用内部函数 _test_find_unused_parameters_kwarg 进行测试
        self._test_find_unused_parameters_kwarg()

    # 使用 requires_nccl 装饰器，要求至少有两个 GPU，同时设置分布式调试级别为 "INFO"
    def test_find_unused_parameters_kwarg_debug_info(self):
        # 调用内部函数 _test_find_unused_parameters_kwarg 进行测试
        self._test_find_unused_parameters_kwarg()

    # 使用 requires_nccl 装饰器，要求至少有两个 GPU，同时设置分布式调试级别为 "OFF"
    def test_find_unused_parameters_kwarg_debug_off(self):
        # 调用内部函数 _test_find_unused_parameters_kwarg 进行测试
        self._test_find_unused_parameters_kwarg()

    # 使用 requires_nccl 装饰器，要求至少有两个 GPU，同时设置分布式调试级别为 "DETAIL"
    def test_find_unused_parameters_kwarg_grad_is_view_debug_detail(self):
        # 调用内部函数 _test_find_unused_parameters_kwarg，设置 gradient_as_bucket_view 参数为 True 进行测试
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    # 使用 requires_nccl 装饰器，要求至少有两个 GPU，同时设置分布式调试级别为 "INFO"
    def test_find_unused_parameters_kwarg_grad_is_view_debug_info(self):
        # 调用内部函数 _test_find_unused_parameters_kwarg，设置 gradient_as_bucket_view 参数为 True 进行测试
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    # 使用 requires_nccl 装饰器，要求至少有两个 GPU，同时设置分布式调试级别为 "OFF"
    def test_find_unused_parameters_kwarg_grad_is_view_debug_off(self):
        # 调用内部函数 _test_find_unused_parameters_kwarg，设置 gradient_as_bucket_view 参数为 True 进行测试
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)
    def _test_multiple_outputs_multiple_backward(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        # 获取当前进程组
        process_group = self._get_process_group()

        # 定义具有多个输出的模型类
        class MultipleOutputModule(nn.Module):
            def __init__(self):
                super().__init__()

                # 定义内部函数来返回一个网络模块序列
                def define_module():
                    return nn.Sequential(
                        nn.Linear(2, 10, bias=False),
                        nn.ReLU(),
                        nn.Linear(10, 4, bias=False),
                        nn.ReLU(),
                    )

                # 创建两个相同结构的网络模块
                self.module0 = define_module()
                self.module1 = define_module()

            def forward(self, x):
                # 返回两个输出的 softmax 结果
                return (
                    F.softmax(self.module0(x), dim=1),
                    F.softmax(self.module1(x), dim=1),
                )

        # 获取当前进程的设备 ID
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        # 创建分布式数据并行模型，将其转移到指定设备
        model = DistributedDataParallel(
            MultipleOutputModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # 设置批量大小
        batch_size = 4
        # 定义交叉熵损失函数
        criterion = nn.CrossEntropyLoss()
        # 创建随机输入数据
        input = torch.rand([batch_size, 2], dtype=torch.float)
        # 创建随机目标标签数据，将其转移到指定设备
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        # 计算模型输出和第一个输出的损失，并进行反向传播
        output1, output2 = model(input)
        loss1 = criterion(output1, target)
        loss1.backward()
        # 计算第二个输出的损失，并进行反向传播
        loss2 = criterion(output2, target)
        loss2.backward()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_multiple_outputs_multiple_backward(self):
        # 调用 _test_multiple_outputs_multiple_backward 方法进行测试
        self._test_multiple_outputs_multiple_backward()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_multiple_outputs_multiple_backward_grad_is_view(self):
        # 调用 _test_multiple_outputs_multiple_backward 方法进行测试，设置梯度视图参数为 True
        self._test_multiple_outputs_multiple_backward(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_no_grad(self):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        # 获取当前进程组的信息
        process_group = self._get_process_group()

        class NoGradModule(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义两个全连接层，输入维度为2，输出维度分别为10和4，无偏置
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                # 前向传播函数，先通过fc1进行线性变换，再ReLU激活，最后通过fc2进行线性变换和ReLU激活
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        # 获取当前进程对应的GPU设备ID
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        # 创建分布式数据并行模型，使用NoGradModule实例，转换为float型并移到指定设备上
        model = DistributedDataParallel(
            NoGradModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )

        batch_size = 4
        input = torch.rand([batch_size, 2], dtype=torch.float)

        def check_no_grads():
            # 检查模型中的参数是否需要梯度，并且梯度是否为None
            for p in model.parameters():
                self.assertTrue(p.requires_grad)
                self.assertIsNone(p.grad)

        # 初始化后，所有参数的梯度都未设置
        check_no_grads()

        # 使用torch.no_grad()运行forward函数
        with torch.no_grad():
            output = model(input)
            self.assertTrue(isinstance(output, torch.Tensor))

        # 检查运行forward后，所有参数的梯度仍未设置
        check_no_grads()
    # 定义一个测试函数来验证累积梯度的模块
    def _test_accumulate_gradients_module(self, gradient_as_bucket_view=False):
        # 这不是推荐的累积梯度实现方式，但我们希望确保 DDP 不会干扰底层模块。
        
        # 获取当前进程的 GPU 设备列表
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        # 将设备列表转换为 torch 设备对象列表
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        # 获取处理组对象
        process_group = self._get_process_group()
        # 设置全局批量大小
        global_batch_size = self.world_size

        # 准备单设备模块，返回原始模型、分布式数据并行模型、输入数据和目标数据
        model, ddp_model, input, target = self._prepare_single_device_module(
            process_group, devices, devices, global_batch_size, gradient_as_bucket_view
        )

        # 定义单步模型训练函数
        def step_model(model, input, target):
            model.train()
            # 模型前向传播
            output = model(input)
            # 计算损失
            loss = F.mse_loss(output, target.to(output.device))
            # 反向传播计算梯度
            loss.backward()

        # 使用 torch.no_grad() 确保累积梯度可以在没有梯度更新的情况下工作
        with torch.no_grad():
            ddp_model.train()
            # 执行模型前向传播
            ddp_model.module(input)

        # 检查两个模型参数在4次迭代中的状态
        # 使用4次迭代是因为我们在减少和不减少之间交替，希望确保两种方式都可以切换。
        for iteration in range(4):
            # 执行单步模型训练
            step_model(model, input, target)

            if iteration % 2 == 0:
                # 跳过调用 prepare_for_backward 的情况下进行梯度同步
                step_model(
                    ddp_model.module,
                    input[self.rank : (self.rank + 1)],
                    target[self.rank : (self.rank + 1)],
                )
                # 检查模型参数的梯度是否不相等
                for i, j in zip(model.parameters(), ddp_model.parameters()):
                    self.assertNotEqual(i.grad, j.grad)
            else:
                # 执行模型训练，包括梯度同步
                step_model(
                    ddp_model,
                    input[self.rank : (self.rank + 1)],
                    target[self.rank : (self.rank + 1)],
                )
                # 检查模型参数的梯度是否相等，设置相对和绝对的误差容差
                for i, j in zip(model.parameters(), ddp_model.parameters()):
                    self.assertEqual(i.grad, j.grad, rtol=1.3e-06, atol=5e-5)

            # 打乱输入数据，以确保 DDP 输入的数据不同
            torch.manual_seed(1337 + iteration)
            input = input[torch.randperm(global_batch_size)]

    # 装饰器，要求使用 NCCL
    @requires_nccl()
    # 装饰器，如果 GPU 数量少于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 测试累积梯度的模块
    def test_accumulate_gradients_module(self):
        # 调用累积梯度模块测试函数
        self._test_accumulate_gradients_module()

    # 装饰器，要求使用 NCCL
    @requires_nccl()
    # 装饰器，如果 GPU 数量少于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 测试带有梯度视图的累积梯度的模块
    def test_accumulate_gradients_module_with_grad_is_view(self):
        # 调用带有梯度视图的累积梯度模块测试函数
        self._test_accumulate_gradients_module(gradient_as_bucket_view=True)

    # 装饰器，要求使用 NCCL
    @requires_nccl()
    # 装饰器，如果 GPU 数量少于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_failure_recovery(self):
        # 获取处理组对象
        process_group = self._get_process_group()

        # 需要为恢复的 FileStore 创建单独的文件，因为第一个 FileStore 在析构时会被删除。
        recovery_filename = self.file_name + "_recovery"

        if self.rank == 0:
            # 由恢复的 FileStore 删除文件
            open(recovery_filename, "w").close()

        # 在此处不需要运行屏障，因为 DDP 会同步

        # 定义测试用的模型类
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        # 获取当前设备 ID
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        # 创建模型实例并移动到指定设备
        model = TestModel().float().to(device_id)
        # 使用 DistributedDataParallel 封装模型
        ddp = DistributedDataParallel(
            model,
            device_ids=[device_id],
            process_group=process_group,
        )

        # 定义批处理大小和损失函数
        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        # 创建随机输入和目标张量
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        # 进行多轮训练和反向传播
        for _ in range(6):
            output = ddp(input)
            loss = criterion(output, target)
            loss.backward()

        # 释放 DDP 实例
        del ddp
        # 销毁进程组
        c10d.destroy_process_group(process_group)

        # 创建 FileStore 实例用于存储恢复文件
        store = c10d.FileStore(recovery_filename, self.world_size)
        # 初始化新的进程组
        c10d.init_process_group(
            "nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        # 获取默认进程组
        process_group = c10d.distributed_c10d._get_default_group()
        # 使用新的进程组再次封装模型
        ddp = DistributedDataParallel(
            model,
            device_ids=[device_id],
            process_group=process_group,
        )

        # 重新定义输入和目标张量
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        # 再次进行多轮训练和反向传播
        for _ in range(6):
            output = ddp(input)
            loss = criterion(output, target)
            loss.backward()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_pass_default_pg(self):
        # 初始化进程组
        dist.init_process_group(
            "nccl",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )

        # 获取默认进程组并销毁
        default_pg = c10d.distributed_c10d._get_default_group()
        dist.destroy_process_group(default_pg)
        # 检查初始化状态为 False
        self.assertFalse(dist.is_initialized())

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_grad_layout_1devicemodule_1replicaperprocess(self):
        # 设定第一个设备为 CUDA 设备，并告知 DDP 仅使用一个设备
        dev0 = torch.device("cuda:" + str(gpus_for_rank(self.world_size)[self.rank][0]))
        # 告知 DDP 仅使用一个设备
        replica_devices = [dev0]
        # 告知 _test_grad_layout 在当前进程的第一个分配设备上构建 ConvNet 的所有层
        layer_devs = dev0
        # 设置本地批量大小为 8
        local_batch_size = 8
        # 调用 _test_grad_layout 方法，传入设备列表、层设备列表和本地批量大小
        self._test_grad_layout(replica_devices, layer_devs, local_batch_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @skip_if_rocm
    def test_grad_layout_2devicemodule(self):
        # 获取当前进程的两个 GPU 设备编号
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        dev0 = torch.device("cuda:" + str(int_devices[0]))
        dev1 = torch.device("cuda:" + str(int_devices[1]))
        # 对于多设备模块，默认情况下 DDP 不会进行复制
        replica_devices = None
        # 告知 _test_grad_layout 在当前进程的两个设备上构建 ConvNet，每个设备有两层
        layer_devs = [dev0] * 2 + [dev1] * 2
        # 设置本地批量大小为 8
        local_batch_size = 8
        # 调用 _test_grad_layout 方法，传入设备列表、层设备列表和本地批量大小
        self._test_grad_layout(replica_devices, layer_devs, local_batch_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_param_layout_mismatch_error(self):
        # 获取当前进程组
        process_group = self._get_process_group()

        # 获取当前进程的第一个 GPU 设备编号
        dev0 = torch.device("cuda:" + str(gpus_for_rank(self.world_size)[self.rank][0]))
        # 告知 ConvNet 所有层将在该设备上构建
        layer_devs = dev0
        # 如果当前进程是第 0 号进程，则使用 torch.contiguous_format；否则使用 torch.channels_last
        layer_formats = (
            [torch.contiguous_format] * 4
            if self.rank == 0
            else [torch.channels_last] * 4
        )
        # 所有层的数据类型均为 torch.float
        layer_dtypes = [torch.float] * 4

        # 创建 ConvNet 模型
        m = ConvNet(layer_devs, layer_formats, layer_dtypes)
        # 如果当前进程是第 0 号进程
        if self.rank == 0:
            # 使用 DDP 在多个设备上复制模型
            m_ddp = DistributedDataParallel(
                m, device_ids=[dev0], process_group=process_group
            )
        else:
            # 在异常情况下，验证错误消息是否包含特定文本
            with self.assertRaisesRegex(
                RuntimeError,
                ".* appears not to match strides of the same param in process 0",
            ):
                # 使用 DDP 在多个设备上复制模型，预期会出现上述异常
                m_ddp = DistributedDataParallel(
                    m, device_ids=[dev0], process_group=process_group
                )

    def _gpu_model_with_ddp_comm_hook(
        self,
        process_group,
        hook=None,
        gradient_as_bucket_view=False,
        state=None,
        static_graph=False,
    ):
        # 获取当前进程的第一个 GPU 设备编号
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        # 创建带有 DDP 通信钩子的 GPU 模型
        gpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )

        # 如果存在通信钩子，则注册之
        if hook is not None:
            gpu_model.register_comm_hook(state, hook)

        return gpu_model

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_future_passing_gpu_nccl(self):
        """
        This unit test verifies whether the Future object is passed properly using nccl backend.
        The hook callback function creates a Future object and sets a value to it.
        """
        # 获取当前进程组
        process_group = self._get_process_group()

        # 使用 self._simple_hook 注册后的 GPU 模型
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, self._simple_hook)

        # 运行并验证钩子后，检查梯度是否与 simple_hook 的 then 回调返回值相等
        # 如果没有通信钩子，结果将为 0.25 * torch.ones(2, 2)
        self._run_and_verify_hook(gpu_model, 8, 2 * torch.ones(2, 2))


    def _test_ddp_comm_hook_allreduce_hook_nccl(
        self, gradient_as_bucket_view=False, static_graph=False
    ):
        """
        This unit test verifies whether a DDP communication hook that just calls
        allreduce gives the same result with the case of no hook registered.
        Without the then callback, the future_value in reducer is no longer
        a PyObject, and this unit test verifies future_value is properly checked.
        """
        # 获取当前进程组
        process_group = self._get_process_group()

        def allreduce_hook(
            state: object, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:
            # 准备要进行 allreduce 的张量
            tensors = [bucket.buffer() / self.world_size]
            # 执行 allreduce 操作并获取其 Future
            return (
                process_group.allreduce(tensors)
                .get_future()
                .then(lambda fut: fut.value()[0])
            )

        # 获取使用 allreduce_hook 注册后的 GPU 模型
        gpu_model = self._gpu_model_with_ddp_comm_hook(
            process_group, allreduce_hook, gradient_as_bucket_view, static_graph
        )

        # 运行并验证钩子后，检查梯度是否与没有钩子的 DDP 返回的结果相等
        self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))
    # 定义一个用于测试默认 DDP 通信钩子的方法，针对 NCCL 后端
    def _test_default_ddp_comm_hooks_nccl(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether default Python DDP communication hooks ALLREDUCE, FP16_COMPRESS
        and BF16_COMPRESS, can give the same result with the case of no hook registered.
        """
        # 获取当前进程组
        process_group = self._get_process_group()

        # 对于这些默认的 DDP 通信钩子，唯一的状态是进程组
        state = process_group
        hook_options = [default.allreduce_hook, default.fp16_compress_hook]
        if (
            not TEST_WITH_ROCM
            and BFLOAT16_AVAILABLE
            and c10d.is_nccl_available()
            and torch.cuda.nccl.version() >= (2, 10)
        ):
            # 如果满足条件，添加 BF16_COMPRESS 钩子选项
            hook_options.append(default.bf16_compress_hook)
        
        # 遍历所有钩子选项
        for hook in hook_options:
            # 使用指定的钩子来获取带有 DDP 通信钩子的 GPU 模型
            # 第一个参数 'process_group' 用于初始化测试环境，
            # 所以不能用 'state' 替换，尽管它们具有相同的值。
            gpu_model = self._gpu_model_with_ddp_comm_hook(
                process_group, hook, gradient_as_bucket_view, state
            )

            # 检查梯度是否与没有钩子注册时的 DDP 返回的结果相等
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    # 定义一个用于测试 FP16_WRAPPER 包装器的方法
    def _test_fp16_compress_wrapper(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether wrapping the ALLREDUCE and POWER_SGD hooks with
        the FP16_WRAPPER can give the same result as when there is no hook registered.
        """
        # 获取当前进程组
        process_group = self._get_process_group()
        
        # 创建一个 PowerSGDState 对象，用于 POWER_SGD 钩子
        powerSGD_state = powerSGD.PowerSGDState(process_group=process_group)

        # 钩子参数列表，每个元组包含钩子函数和相应的状态
        hook_args = [
            (powerSGD.powerSGD_hook, powerSGD_state),
            (default.allreduce_hook, process_group),
        ]

        # 遍历所有钩子参数
        for hook, state in hook_args:
            # 使用 FP16_WRAPPER 包装器来包装 ALLREDUCE 和 POWER_SGD 钩子
            gpu_model = self._gpu_model_with_ddp_comm_hook(
                process_group,
                default.fp16_compress_wrapper(hook),
                gradient_as_bucket_view,
                state,
            )

            # 检查梯度是否与没有钩子注册时的 DDP 返回的结果相等
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))
    def _test_bf16_compress_wrapper(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether wrapping the ALLREDUCE and POWER_SGD hooks with
        the BF16_WRAPPER can give the same result as when there is no hook registered.
        """
        # 获取当前进程组
        process_group = self._get_process_group()
        # 创建 POWER_SGD 状态对象
        powerSGD_state = powerSGD.PowerSGDState(process_group=process_group)

        # 定义钩子参数列表
        hook_args = [
            (powerSGD.powerSGD_hook, powerSGD_state),
            (default.allreduce_hook, process_group),
        ]

        # 遍历钩子参数列表
        for hook, state in hook_args:
            # 使用包装后的 hook 构建 GPU 模型
            gpu_model = self._gpu_model_with_ddp_comm_hook(
                process_group,
                default.bf16_compress_wrapper(hook),
                gradient_as_bucket_view,
                state,
            )

            # 检查梯度是否与没有 hook 的情况下返回的结果相同
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_powerSGD_ddp_comm_hook_nccl(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether Python DDP communication hook POWER_SGD
        can give the same result with the case of no hook registered.
        """
        # 获取当前进程组
        process_group = self._get_process_group()

        # 测试带有注册钩子的 GPU 模型
        # 使用不同的算法配置测试钩子
        for use_error_feedback, warm_start, batch_tensors_with_same_shape in product(
            [True, False],
            [True, False],
            [True, False],
        ):
            # 创建 POWER_SGD 状态对象
            state = powerSGD.PowerSGDState(
                process_group=process_group,
                matrix_approximation_rank=1,
                use_error_feedback=use_error_feedback,
                warm_start=warm_start,
                batch_tensors_with_same_shape=batch_tensors_with_same_shape,
            )
            # 遍历钩子列表
            for hook in [powerSGD.powerSGD_hook, powerSGD.batched_powerSGD_hook]:
                # 使用钩子构建 GPU 模型
                gpu_model = self._gpu_model_with_ddp_comm_hook(
                    process_group, hook, gradient_as_bucket_view, state
                )

                # 检查梯度是否与没有 hook 的情况下返回的结果相同
                self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))
    def _test_builtin_ddp_comm_hooks_nccl(self, gradient_as_bucket_view=False):
        """
        This method is a unit test for verifying the behavior of built-in C++ DDP communication hooks,
        specifically ALLREDUCE and FP16_COMPRESS, with and without hooks registered.
        
        Parameters:
        - gradient_as_bucket_view: Boolean flag indicating whether gradients should be treated as bucket views.
        """
        # 获取当前进程组的分组对象
        process_group = self._get_process_group()

        # 遍历所有内置通信钩子类型
        for comm_hook_type in [
            dist.BuiltinCommHookType.ALLREDUCE,
            dist.BuiltinCommHookType.FP16_COMPRESS,
        ]:
            # 使用指定的内置 DDP 通信钩子类型和参数获取 GPU 模型
            gpu_model = self._gpu_model_with_builtin_ddp_comm_hook(
                process_group, comm_hook_type, gradient_as_bucket_view
            )

            # 运行并验证钩子效果，比较结果是否与无钩子情况一致
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))
    def test_bf16_compress_wrapper_is_view(self):
        self._test_bf16_compress_wrapper(gradient_as_bucket_view=True)


        # 测试函数：测试 bf16_compress_wrapper 是否作为视图
        # 调用 _test_bf16_compress_wrapper 方法，传入参数 gradient_as_bucket_view=True
        self._test_bf16_compress_wrapper(gradient_as_bucket_view=True)



    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_builtin_ddp_comm_hooks_nccl_grad_is_view(self):
        self._test_builtin_ddp_comm_hooks_nccl(gradient_as_bucket_view=True)


        # 测试函数：测试内置的 DDP 通信钩子 nccl 是否作为视图
        # 装饰器要求使用 NCCL，跳过少于 2 个 GPU 的情况
        # 调用 _test_builtin_ddp_comm_hooks_nccl 方法，传入参数 gradient_as_bucket_view=True
        self._test_builtin_ddp_comm_hooks_nccl(gradient_as_bucket_view=True)



    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_powerSGD_ddp_comm_hook_nccl_grad_is_view(self):
        self._test_powerSGD_ddp_comm_hook_nccl(gradient_as_bucket_view=True)


        # 测试函数：测试 PowerSGD DDP 通信钩子 nccl 是否作为视图
        # 装饰器要求使用 NCCL，跳过少于 2 个 GPU 的情况
        # 调用 _test_powerSGD_ddp_comm_hook_nccl 方法，传入参数 gradient_as_bucket_view=True
        self._test_powerSGD_ddp_comm_hook_nccl(gradient_as_bucket_view=True)



    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_with_then_hook_nccl(self):
        """
        This unit test verifies whether a DDP communication hook that calls allreduce and then
        multiplies the result by ten and divides by two gives the expected result.
        """
        process_group = self._get_process_group()

        def allreduce_with_then_hook(
            state: object, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:
            tensors = [bucket.buffer() / self.world_size]
            fut = process_group.allreduce(tensors).get_future()

            def mult(fut):
                # Multiply the result by 10.
                return 10 * fut.value()[0]

            def div(fut):
                # Divide the result by 2.
                return 0.5 * fut.value()

            return fut.then(mult).then(div)

        # Get GPU model with allreduce_with_then_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(
            process_group, allreduce_with_then_hook
        )

        # check whether the grads are equal to what allreduce returns multiplied by 5.
        # without the comm_hook, result would be still 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(gpu_model, 8, 1.25 * torch.ones(2, 2))


        # 测试函数：测试 DDP 通信钩子 allreduce_with_then_hook_nccl
        """
        此单元测试验证一个 DDP 通信钩子，调用 allreduce，然后将结果乘以十再除以二是否给出了预期的结果。
        """
        # 获取进程组
        process_group = self._get_process_group()

        def allreduce_with_then_hook(
            state: object, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:
            # 准备张量列表，包含对应缓冲区除以进程组大小后的结果
            tensors = [bucket.buffer() / self.world_size]
            # 执行 allreduce 并获取 Future 对象
            fut = process_group.allreduce(tensors).get_future()

            def mult(fut):
                # 将结果乘以 10
                return 10 * fut.value()[0]

            def div(fut):
                # 将结果除以 2
                return 0.5 * fut.value()

            # 返回经过乘法和除法处理的 Future 对象
            return fut.then(mult).then(div)

        # 获取带有注册 allreduce_with_then_hook 的 GPU 模型
        gpu_model = self._gpu_model_with_ddp_comm_hook(
            process_group, allreduce_with_then_hook
        )

        # 检查梯度是否等于 allreduce 返回的结果乘以 5
        # 如果没有通信钩子，结果仍为 0.25 * torch.ones(2, 2)
        self._run_and_verify_hook(gpu_model, 8, 1.25 * torch.ones(2, 2))



    class AcceptsParam(torch.nn.Module):
        def __init__(self, p, factor):
            super().__init__()
            self.a = p
            self.f = factor

        def forward(self, input):
            return input + self.a * self.f


    class AcceptsParam(torch.nn.Module):
        # 接受参数的 PyTorch 模块
        def __init__(self, p, factor):
            # 初始化方法
            super().__init__()
            # 设置属性 a 和 f
            self.a = p
            self.f = factor

        def forward(self, input):
            # 前向传播方法，返回输入加上属性 a 乘以属性 f 的结果
            return input + self.a * self.f



    @requires_nccl()
    @skip_if_lt_x_gpu(2)


    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    # 装饰器要求使用 NCCL，跳过少于 2 个 GPU 的情况
    # 定义测试分布式数据并行权重共享的方法，使用了torch.nn.Parameter定义了一个需要梯度的参数
    def test_ddp_weight_sharing(self):
        # 获取当前进程组
        process_group = self._get_process_group()

        # 定义大小为2048*2048的张量
        size = 2048 * 2048
        # 获取当前进程的rank
        dev = self.rank
        # 获取总的进程数
        world = self.world_size

        # 创建一个随机张量参数
        p = torch.nn.Parameter(torch.randn(size, requires_grad=True))

        # 针对try_set_to_none和use_bucket_view的组合进行迭代
        for try_set_to_none, use_bucket_view in product((False, True), (False, True)):
            # 构建包含两个AcceptsParam层的序列模型，并将其移到GPU上
            m = torch.nn.Sequential(
                self.AcceptsParam(p, dev + 1), self.AcceptsParam(p, dev + 1)
            ).cuda(dev)

            # 使用分布式数据并行将模型m分布到多个GPU上，并指定相关参数
            m = torch.nn.parallel.DistributedDataParallel(
                m,
                bucket_cap_mb=1,
                gradient_as_bucket_view=use_bucket_view,
                device_ids=[dev],
                process_group=process_group,
            )

            # 进行3次迭代
            for i in range(3):
                # 清除梯度
                m.zero_grad(set_to_none=try_set_to_none)
                # 执行前向传播和反向传播
                m(1).sum().backward()

                # 根据特定的rank计算期望的梯度值
                analytic = torch.full_like(
                    p, 2.0 * (world * (world + 1.0) / 2.0) / world, device=dev
                )
                # 检查每个参数的梯度是否符合预期值
                for name, p in m.named_parameters():
                    self.assertEqual(
                        p.grad,
                        analytic,
                        "mismatch at "
                        + name
                        + ".grad for "
                        + f"set_to_none = {try_set_to_none}, use_bucket_view = {use_bucket_view}",
                    )

    # 使用NCCL时的装饰器，确保至少有两个GPU可用
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_packed_sequence(self):
        """
        Tests that DDP with ``device_ids`` specified can run a forward and
        backward pass with ``PackedSequence`` s with parity compared to a local
        version of the model.
        """
        # 创建一个文件存储对象，用于分布式进程组通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化分布式进程组，使用 NCCL 后端
        process_group = dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 定义一些示例序列
        seqs = ["sequence_sequence", "seq", "sequence"]
        # 创建一个包含所有字符（包括填充字符 '<pad>'）的词汇表
        vocab = ["<pad>"] + sorted({ch for seq in seqs for ch in seq})
        # 将每个序列中的字符转换为词汇表索引
        vectorized_seqs = [[vocab.index(tok) for tok in seq] for seq in seqs]
        # 设置随机种子以确保嵌入和 LSTM 的结果在不同 GPU 上也是确定性的
        torch.manual_seed(0)
        # 创建一个嵌入层，保持在 CPU 上
        embed = nn.Embedding(len(vocab), 4)
        # 创建一个 LSTM 模型，在指定的 GPU 设备上运行
        lstm = nn.LSTM(input_size=4, hidden_size=2, batch_first=True).to(self.rank)
        # 使用分布式数据并行化封装 LSTM 模型
        lstm_ddp = DistributedDataParallel(
            copy.deepcopy(lstm),
            device_ids=[self.rank],
            process_group=process_group,
        )
        # 检查本地 LSTM 模型和分布式封装后的 LSTM 模型的参数是否一致
        for p1, p2 in zip(lstm.parameters(), lstm_ddp.module.parameters()):
            self.assertEqual(p1, p2)
        # 计算每个序列的长度，并按照长度降序排序
        seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))
        seq_tensor = torch.Tensor(
            torch.zeros((len(vectorized_seqs), seq_lengths.max()))
        ).long()
        # 根据排序后的索引重新排列序列张量
        seq_tensor = seq_tensor[permutation_idx]
        # 对嵌入后的序列张量创建压缩填充序列
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_seq_tensor,
            seq_lengths,
            batch_first=True,
        )
        # 对嵌入后的序列张量创建压缩填充序列（为 DDP 复制）
        packed_input_ddp = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_seq_tensor.detach().clone(),
            seq_lengths,
            batch_first=True,
        )
        # 显式将输入移动到指定 GPU 上进行本地模型计算
        packed_output, (ht, ct) = lstm(packed_input.to(self.rank))
        # 让 DDP 内部将输入移动到 GPU 上进行计算
        packed_output_ddp, (ht_ddp, ct_ddp) = lstm_ddp(packed_input_ddp)
        # 检查本地模型和 DDP 模型的输出是否一致
        self.assertEqual(packed_output.data, packed_output_ddp.data)
        self.assertEqual(ht, ht_ddp)
        self.assertEqual(ct, ct_ddp)
        # 计算压缩输出的梯度
        packed_output.data.sum().backward()
        packed_output_ddp.data.sum().backward()
        # 检查本地 LSTM 模型和 DDP 模型的梯度是否一致
        for p1, p2 in zip(lstm.parameters(), lstm_ddp.parameters()):
            self.assertEqual(p1.grad, p2.grad)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_channels_last_contig(self):
        # 获取当前进程组
        process_group = self._get_process_group()
        # 根据当前进程的rank选择CUDA设备
        device = torch.device(f"cuda:{self.rank}")
        # 创建一个CUDA tensor，形状为(2, 16, 768, 1152)，数据类型为float32，使用通道最后的内存布局
        tensor = torch.ones((2, 16, 768, 1152), dtype=torch.float32, device=device).to(
            memory_format=torch.channels_last
        )
        # 在进程组中广播tensor
        process_group.broadcast([tensor]).wait()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_complex_params(self):
        # 定义一个FFT模型
        class FFTModel(nn.Module):
            def __init__(self, hin, win, n_features):
                super().__init__()
                self.hin = hin
                self.win = win
                # 定义一个复数类型的参数张量，形状为(n_features, n_features, hin, win // 2 + 1)
                self.weight = nn.Parameter(
                    torch.ones(
                        (n_features, n_features, hin, win // 2 + 1), dtype=torch.cfloat
                    )
                )

            def forward(self, x):
                # 对输入张量进行二维实值快速傅里叶变换
                xc = torch.fft.rfft2(
                    x, s=(self.hin, self.win), dim=(-2, -1), norm="ortho"
                )
                # 使用einsum计算xc和self.weight的乘积
                xcw = torch.einsum("nchw,cohw->nohw", xc, self.weight)
                # 对xcw进行二维逆傅里叶变换
                x = torch.fft.irfft2(xcw, dim=(-2, -1), norm="ortho")
                return x

        # 获取当前进程组
        process_group = self._get_process_group()
        # 获取当前进程对应的CUDA设备ID
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        # 设置输入张量的形状：(N, C, H, W) = (1, 16, 64, 64)
        N, C, H, W = 1, 16, 64, 64
        # 创建分布式数据并行模型，使用FFTModel作为模型，在指定的CUDA设备上进行计算
        ddp_model = DistributedDataParallel(
            FFTModel(hin=H, win=W, n_features=C).to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )
        # 定义优化器
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

        # 创建输入张量，全为1，形状为(N, C, H, W)
        inp = torch.ones((N, C, H, W), dtype=torch.float32)

        # 训练步骤
        # 前向传播
        out = ddp_model(inp)
        # 计算损失
        loss = torch.sum(out)
        # 反向传播
        loss.backward()
        # 优化器执行一步参数更新
        optimizer.step()

        # 同步CUDA设备，确保所有设备上的操作都已完成
        torch.cuda.synchronize(device=device_id)
class WorkHookTest(MultiProcessTestCase):
    # 定义一个测试类 WorkHookTest，继承自 MultiProcessTestCase
    @property
    def world_size(self):
        # 返回并发测试的进程数，这里固定为 2
        return 2

    def setUp(self):
        # 设置测试的前置条件，调用父类的 setUp 方法
        super().setUp()
        # 设置环境变量 TORCH_NCCL_ENABLE_TIMING 为 "1"，启用 CUDA 事件的定时功能
        os.environ["TORCH_NCCL_ENABLE_TIMING"] = "1"
        # 启动测试中的多个进程
        self._spawn_processes()

    def tearDown(self):
        # 清理测试后的状态，调用父类的 tearDown 方法
        super().tearDown()
        # 删除环境变量 TORCH_NCCL_ENABLE_TIMING
        del os.environ["TORCH_NCCL_ENABLE_TIMING"]
        try:
            # 尝试删除测试中创建的文件
            os.remove(self.file_name)
        except OSError:
            # 如果文件不存在则捕获异常
            pass

    def _get_store(self):
        # 返回一个分布式文件存储对象，使用 self.file_name 和 self.world_size 参数
        return dist.FileStore(self.file_name, self.world_size)

    def _get_process_group(self):
        # 获取并初始化进程组，使用 "nccl" 后端，连接到指定的存储和进程组信息
        store = self._get_store()
        c10d.init_process_group(
            "nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        return c10d.distributed_c10d._get_default_group()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_on_completion_hook_broadcast(self):
        # 根据装饰器要求，仅在满足特定条件时运行该测试方法
        # 获取进程组
        pg = self._get_process_group()
        # 记录钩子函数被触发的次数和每次触发的持续时间
        num_hook_fired = 0
        durations: List[float] = []

        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            # 定义一个钩子函数，接收 work_info 参数
            nonlocal num_hook_fired, durations
            num_hook_fired += 1
            # 记录每次钩子函数调用的持续时间（秒）
            durations.append(work_info.active_duration.total_seconds())

        # 注册钩子函数到进程组中
        pg._register_on_completion_hook(hook)
        # 创建一个张量并在 CUDA 设备上执行广播操作，等待操作完成
        tensor = torch.ones([2, 3]).cuda(self.rank) * self.rank
        pg.broadcast([tensor]).wait()
        pg.broadcast([tensor]).wait()

        # 注意：调用 destroy_process_group 方法是必要的，以等待所有挂起的工作完成
        c10d.destroy_process_group(pg)

        # 断言钩子函数被调用的次数为 2
        self.assertEqual(num_hook_fired, 2)
        # 断言记录的持续时间列表长度为 2
        self.assertEqual(len(durations), 2)
        # 断言每个持续时间大于 0
        for duration in durations:
            self.assertTrue(duration > 0)

        # 断言广播后的张量为在当前进程中填充零值的张量
        self.assertEqual(tensor, torch.zeros([2, 3]).cuda(self.rank))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    # 定义一个测试方法，用于测试完成钩子函数的混合操作
    def test_on_completion_hook_mixed_ops(self):
        # 获取当前进程组
        pg = self._get_process_group()
        # 初始化钩子函数被调用次数和持续时间列表
        num_hook_fired = 0
        durations: List[float] = []

        # 定义钩子函数，用于处理工作信息，记录钩子函数被调用次数和每次调用的持续时间
        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            nonlocal num_hook_fired, durations
            num_hook_fired += 1
            durations.append(work_info.active_duration.total_seconds())

        # 将钩子函数注册到进程组的完成钩子中
        pg._register_on_completion_hook(hook)

        # 创建一个在 GPU 上的全1张量
        tensor = torch.ones([2, 3]).cuda(self.rank)
        # 创建一个与tensor相同形状的张量列表
        tensor_list = [torch.empty_like(tensor) for _ in range(self.world_size)]

        # 执行异步操作：全局归约操作
        pg.allreduce(tensor)
        # 执行异步操作：全局聚集操作
        pg.allgather(tensor_list, tensor)
        # 再次执行异步操作：全局归约操作
        pg.allreduce(tensor)

        # 注意：调用 destroy_process_group 方法等待所有挂起的工作完成
        c10d.destroy_process_group(pg)

        # 断言钩子函数被调用3次
        self.assertEqual(num_hook_fired, 3)
        # 断言持续时间列表包含3个元素
        self.assertEqual(len(durations), 3)
        # 断言每个持续时间大于0
        for duration in durations:
            self.assertTrue(duration > 0)

        # 断言张量经过全局归约后的值符合预期
        self.assertEqual(
            tensor,
            torch.ones([2, 3]).cuda(self.rank) * self.world_size * self.world_size,
        )

        # 断言张量列表中的每个张量经过全局聚集后的值符合预期
        self.assertEqual(
            tensor_list,
            [
                torch.ones([2, 3]).cuda(self.rank) * self.world_size
                for _ in range(self.world_size)
            ],
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    # 定义一个测试函数，用于测试分布式数据并行处理完成钩子函数
    def test_on_completion_hook_with_ddp(self):
        # 获取进程组对象
        pg = self._get_process_group()
        # 记录每种操作类型触发钩子的次数的字典
        num_hook_fired: Dict[int, int] = {}
        # 记录每种操作类型执行时间的字典
        durations: Dict[OpType, List[float]] = {}

        # 定义钩子函数，处理分布式训练过程中的操作信息
        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            nonlocal num_hook_fired, durations
            op_type = work_info.op_type
            # 如果操作类型不在记录中，则初始化记录
            if op_type not in num_hook_fired:
                num_hook_fired[op_type] = 0
                durations[op_type] = []
            # 记录操作类型钩子被触发的次数
            num_hook_fired[op_type] += 1
            # 记录操作类型执行时间（秒）
            durations[op_type].append(work_info.active_duration.total_seconds())

        # 注册钩子函数到进程组对象
        pg._register_on_completion_hook(hook)

        # 定义网络层数
        nlayers = 10
        # 创建一个具有多层线性层的神经网络
        net = nn.Sequential(
            *[nn.Linear(1000, 1000, bias=False) for _ in range(nlayers)]
        ).to(self.rank)

        # 使用分布式数据并行处理封装神经网络
        ddp = DistributedDataParallel(
            net,
            device_ids=[self.rank],
            process_group=pg,
            bucket_cap_mb=1,
        )

        # 等待所有挂起的工作完成
        pg._wait_for_pending_works()

        # 断言：期望DDP通过广播同步模型参数
        self.assertTrue(num_hook_fired[OpType.BROADCAST] > 0)
        
        # 如果存在ALLREDUCE操作类型，获取其钩子触发次数；否则置为0
        ctor_allreduce = (
            num_hook_fired[OpType.ALLREDUCE]
            if OpType.ALLREDUCE in num_hook_fired
            else 0
        )

        # 创建一个张量并将其移到指定GPU设备上
        x = torch.zeros(2, 1000).cuda(self.rank)
        # 对张量执行DDP处理，并对其求和后进行反向传播
        ddp(x).sum().backward()

        # 销毁进程组对象
        c10d.destroy_process_group(pg)

        # 断言：确保ALLREDUCE操作类型的钩子被触发
        self.assertTrue(OpType.ALLREDUCE in num_hook_fired)
        # 断言：确保至少有一个ALLREDUCE操作类型的钩子被触发
        self.assertTrue(num_hook_fired[OpType.ALLREDUCE] - ctor_allreduce > 0)
        # 断言：确保所有操作的执行时间都大于0秒
        self.assertTrue(all(duration > 0 for duration in chain(*(durations.values()))))
    # 定义一个测试方法，用于测试在完成钩子（hook）时的全部聚合对象操作
    def test_on_completion_hook_all_gather_object(self):
        # 设置当前 CUDA 设备为指定的排名（rank）
        torch.cuda.set_device(self.rank)

        # 获取当前进程组（process group）
        pg = self._get_process_group()
        
        # 用于记录钩子（hook）被触发的次数，按操作类型分类
        num_hook_fired: Dict[int, int] = {}
        
        # 用于记录每种操作类型的持续时间列表
        durations: Dict[OpType, List[float]] = {}

        # 定义一个钩子函数，用于在任务完成时调用
        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            nonlocal num_hook_fired, durations
            # 获取操作类型
            op_type = work_info.op_type
            # 如果该操作类型尚未记录，初始化计数器和持续时间列表
            if op_type not in num_hook_fired:
                num_hook_fired[op_type] = 0
                durations[op_type] = []
            # 增加该操作类型的钩子触发次数
            num_hook_fired[op_type] += 1
            # 记录该操作的活动持续时间（以秒为单位）
            durations[op_type].append(work_info.active_duration.total_seconds())

        # 注册钩子函数到进程组
        pg._register_on_completion_hook(hook)

        # 创建一个包含当前进程信息的对象
        obj = {"rank": self.rank, "world_size": self.world_size}
        # 创建一个与进程数量相等的对象列表，用于存放收集的对象信息
        obj_list = [None for _ in range(self.world_size)]

        # 在进程组中执行全部聚合对象操作
        c10d.all_gather_object(obj_list, obj, group=pg)

        # 遍历收集到的对象列表，进行断言验证
        for r, o in enumerate(obj_list):
            self.assertTrue(isinstance(o, dict))
            self.assertTrue(set(o.keys()), {"rank", "world_size"})
            self.assertEqual(o["rank"], r)
            self.assertEqual(o["world_size"], self.world_size)

        # 销毁进程组，结束任务
        c10d.destroy_process_group(pg)

        # 断言确保 ALLGATHER 操作类型至少触发一次
        self.assertTrue(OpType.ALLGATHER in num_hook_fired)
        # 断言只有一种操作类型被触发
        self.assertEqual(len(num_hook_fired), 1)
        # 断言 ALLGATHER 操作类型确切地触发了两次
        self.assertEqual(num_hook_fired[OpType.ALLGATHER], 2)
        # 断言所有的 ALLGATHER 操作持续时间都大于零秒
        self.assertTrue(all(duration > 0 for duration in durations[OpType.ALLGATHER]))

    # 标记为需要 NCCL 支持的测试方法，并且要求至少有两个 GPU
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_on_completion_hook_seq(self):
        # 获取当前进程组
        pg = self._get_process_group()
        
        # 记录钩子（hook）被触发的次数
        num_hook_fired = 0
        # 记录最后一个工作的序列号
        seq: int = -1
        # 记录工作的数量
        work: int = 0

        # 定义一个钩子函数，用于在任务完成时调用
        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            nonlocal num_hook_fired, seq
            # 增加钩子被触发的次数
            num_hook_fired += 1
            # 记录当前工作的序列号
            seq = work_info.seq

        # 注册钩子函数到进程组
        pg._register_on_completion_hook(hook)

        # 创建一个在 GPU 上的张量，每个 GPU 广播一个张量
        tensor = torch.ones([2, 3]).cuda(self.rank) * self.rank
        # 执行多次广播操作
        work_count = 3
        for i in range(work_count):
            work += 1
            # 在进程组中广播张量，并等待操作完成
            pg.broadcast([tensor]).wait()

        # 注意：销毁进程组是必要的，以等待所有挂起的工作完成
        c10d.destroy_process_group(pg)

        # 断言确保钩子被触发的次数与工作数量相等
        self.assertEqual(num_hook_fired, work_count)
        # 断言所有工作的序列号都被记录下来
        self.assertEqual(work, seq)
class NcclErrorHandlingTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        # Need to skip return code checking for these tests since the child
        # processes don't exit cleanly.
        # 设置需要跳过返回码检查的测试方法列表，因为子进程无法正常退出。
        self.skip_return_code_checks = [
            self.test_nccl_errors_blocking_abort.__wrapped__,
            self.test_nccl_errors_blocking_sigkill.__wrapped__,
            self.test_nccl_errors_blocking_sigterm.__wrapped__,
            self.test_nccl_errors_blocking_nonzero_exit.__wrapped__,
        ]
        # 设置环境变量，强制启用 TORCH_NCCL_ASYNC_ERROR_HANDLING，以便测试正常执行。
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()  # 调用方法启动多个进程

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def op_timeout_sec(self):
        return 3  # 返回操作超时的时间（秒）

    @property
    def world_size(self):
        return 3  # 返回进程组的大小为 3

    @property
    def blocking_wait_error_msg(self):
        return "timeout"  # 返回阻塞等待超时的错误信息

    def _run_all_reduce(self, pg):
        pg.allreduce(torch.rand(10).cuda(self.rank))  # 执行全局归约操作

    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm
    @skip_but_pass_in_sandcastle("Test does not pass when run locally")
    def test_nccl_errors_nonblocking(self):
        # Note: we unset and restore TORCH_NCCL_ASYNC_ERROR_HANDLING for this test
        # since test_c10d_common runs with async error handling by default, but this
        # tests behavior when it is not enabled.
        # 为了测试异步错误处理未启用时的行为，我们在此测试中取消并恢复 TORCH_NCCL_ASYNC_ERROR_HANDLING 设置。
        prev_nccl_async_error_handling = os.environ.get(
            "TORCH_NCCL_ASYNC_ERROR_HANDLING", None
        )
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        process_group.allreduce(torch.rand(10).cuda(self.rank))  # 执行全局归约操作
        if self.rank == 0:
            # This allreduce does not block Python thread as allreduce enqueues
            # the cuda operation, and then wait only blocks the current cuda
            # stream.
            # 当前 allreduce 操作不会阻塞 Python 线程，因为 allreduce 将 cuda 操作入队，
            # 然后 wait 方法仅阻塞当前的 cuda 流。
            work = process_group.allreduce(torch.rand(10).cuda(self.rank))
            work.wait()

            # Now the work scheduled next should hang forever since the previous
            # allreduce will never complete.
            # 现在计划执行的下一个工作应该会永远挂起，因为前一个 allreduce 永远不会完成。
            t = threading.Thread(target=self._run_all_reduce, args=(process_group,))
            t.daemon = True
            t.start()
            t.join(int(get_timeout(self.id()) / 5))  # 等待线程执行，超时时间为总超时时间的五分之一
            self.assertTrue(t.is_alive())  # 断言线程仍然存活

        if prev_nccl_async_error_handling is not None:
            os.environ[
                "TORCH_NCCL_ASYNC_ERROR_HANDLING"
            ] = prev_nccl_async_error_handling
    # 测试函数，用于检查在使用 NCCL 进行通信时的错误处理和清理功能
    def _test_nccl_errors_blocking(self, func):
        # 使用文件存储创建分布式存储对象，指定文件名和世界大小
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建 NCCL 进程组对象，使用文件存储，指定当前进程的排名和总进程数，并设置超时时间为10秒
        process_group = c10d.ProcessGroupNCCL(
            store,
            self.rank,
            self.world_size,
            timeout=timedelta(seconds=10),
        )
        # 在所有 GPU 设备上进行全局归约操作，传输随机生成的张量数据
        process_group.allreduce(torch.rand(10).cuda(self.rank))
        
        # 如果当前进程的排名为0
        if self.rank == 0:
            # 再次进行全局归约操作，并传入当前进程的 GPU 设备数据
            work = process_group.allreduce(torch.rand(10).cuda(self.rank))
            # 使用断言检查是否抛出了 DistBackendError 异常，异常消息可能因运行环境不同而不同
            with self.assertRaisesRegex(dist.DistBackendError, ""):
                # 忽略错误消息的检查，以确保测试在不同环境下均可通过
                work.wait(timeout=timedelta(seconds=self.op_timeout_sec))
            
            # 执行一些 GPU 操作，以确保 CUDA 没有被卡住
            # 注意到如果 NCCL 通信器未能正确中止就抛出了 RuntimeError，CUDA 有可能会被卡住
            a = torch.rand(10).cuda(self.rank)
        
        # 如果当前进程的排名为1
        elif self.rank == 1:
            # 清理一些结构（例如：在关闭前的 FileStore 文件）
            # 删除 process_group 变量以释放资源，并调用传入的 func 函数执行清理操作
            del process_group
            func()

    # 测试 NCCL 错误处理和清理功能，在正常退出时的情况下
    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm
    def test_nccl_errors_blocking_clean_exit(self):
        self._test_nccl_errors_blocking(lambda: sys.exit(0))

    # 测试 NCCL 错误处理和清理功能，在非正常退出时的情况下（返回非零退出码）
    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm
    def test_nccl_errors_blocking_nonzero_exit(self):
        self._test_nccl_errors_blocking(lambda: sys.exit(1))

    # 测试 NCCL 错误处理和清理功能，在通过 os.abort() 终止进程时的情况下
    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm
    @skip_but_pass_in_sandcastle(
        "Frequently times out see https://github.com/pytorch/pytorch/issues/58920"
    )
    def test_nccl_errors_blocking_abort(self):
        self._test_nccl_errors_blocking(lambda: os.abort())

    # 测试 NCCL 错误处理和清理功能，在通过 os.kill(os.getpid(), signal.SIGKILL) 发送 SIGKILL 信号终止进程时的情况下
    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm
    def test_nccl_errors_blocking_sigkill(self):
        self._test_nccl_errors_blocking(lambda: os.kill(os.getpid(), signal.SIGKILL))

    # 测试 NCCL 错误处理和清理功能，在通过 os.kill(os.getpid(), signal.SIGTERM) 发送 SIGTERM 信号终止进程时的情况下
    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm
    def test_nccl_errors_blocking_sigterm(self):
        self._test_nccl_errors_blocking(lambda: os.kill(os.getpid(), signal.SIGTERM))

    # 测试 NCCL 错误处理和清理功能的其他情况
    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm
    def test_nccl_errors_blocking_other_cases(self):
        self._test_nccl_errors_blocking(lambda: None)
    # 定义一个测试方法，用于测试带有屏障的 NCCL 阻塞等待功能
    def test_nccl_blocking_wait_with_barrier(self):
        # 创建一个基于文件的存储对象，用于进程组的通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建一个 NCCL 进程组对象，指定当前进程的排名和总进程数，并设置超时时间为10秒
        process_group = c10d.ProcessGroupNCCL(
            store,
            self.rank,
            self.world_size,
            timeout=timedelta(seconds=10),
        )
        # 在进程组上调用屏障方法，等待所有进程达到屏障
        process_group.barrier().wait()
        # 如果当前进程的排名为0
        if self.rank == 0:
            # 使用断言来验证在特定条件下是否会抛出 DistBackendError 异常
            with self.assertRaisesRegex(dist.DistBackendError, ""):
                # 调用屏障方法，设置超时时间为操作超时秒数
                process_group.barrier().wait(
                    timeout=timedelta(seconds=self.op_timeout_sec)
                )

    # 定义一个方法，在不合法的 NCCL 阻塞等待环境中运行，验证是否会抛出 RuntimeError 异常
    def _run_invalid_nccl_blocking_wait_env(self, val):
        # 设置环境变量 TORCH_NCCL_BLOCKING_WAIT 为指定的值
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = val
        # 创建一个基于文件的存储对象，用于进程组的通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用断言来验证在初始化时是否会抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

    # 标记需要 NCCL 支持的测试方法，要求至少有3个 GPU，用于测试不合法的 NCCL 阻塞等待环境
    @requires_nccl()
    @skip_if_lt_x_gpu(3)
    def test_invalid_nccl_blocking_wait_env(self):
        # 分别运行几种不合法的 NCCL 阻塞等待环境
        self._run_invalid_nccl_blocking_wait_env("abc")
        self._run_invalid_nccl_blocking_wait_env("-1")
        self._run_invalid_nccl_blocking_wait_env("2147483647")
        self._run_invalid_nccl_blocking_wait_env("4294967295")

    # 标记需要 NCCL 支持、Gloo 支持的测试方法，要求至少有3个 GPU，用于测试 NCCL 超时功能
    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_gloo()
    @skip_if_lt_x_gpu(3)
    def test_nccl_timeout(self):
        # 创建一个基于文件的存储对象，用于进程组的通信
        store = c10d.FileStore(self.file_name, self.world_size)

        # 初始化 NCCL 进程组对象，指定当前进程的排名和总进程数，并设置超时时间为10秒
        process_group = c10d.ProcessGroupNCCL(
            store, self.rank, self.world_size, timeout=timedelta(seconds=10)
        )

        # 创建一个 Gloo 进程组对象，用于协调各个进程之间的通信
        pg_gloo = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        # 创建一个时间间隔对象，表示失败的集体操作超时时间为100毫秒
        failed_collective_timeout = timedelta(milliseconds=100)

        # 在 NCCL 进程组上执行 allreduce 操作，等待所有进程完成
        process_group.allreduce(torch.rand(10).cuda(self.rank)).wait(
            timeout=timedelta(seconds=5)
        )

        # 如果当前进程的排名为0
        if self.rank == 0:
            # 使用断言来验证在特定条件下是否会抛出 DistBackendError 异常，检查阻塞等待时的错误消息
            with self.assertRaisesRegex(
                dist.DistBackendError, self.blocking_wait_error_msg
            ):
                # 在 NCCL 进程组上执行 allreduce 操作，设置超时时间为失败的集体超时时间
                process_group.allreduce(torch.rand(10).cuda(self.rank)).wait(
                    timeout=failed_collective_timeout
                )
            # 执行一个屏障操作，告知其他排名继续执行
            pg_gloo.barrier().wait()
        else:
            # 等待排名为0的进程失败
            try:
                pg_gloo.barrier().wait()
            except Exception as e:
                # 抛出异常，说明等待排名为0的进程超时
                raise ValueError(
                    f"Rank {self.rank} barrier timed out waiting for rank 0 with error: {str(e)}"
                ) from e
class CommTest(test_c10d_common.AbstractCommTest, MultiProcessTestCase):
    @property
    def device(self):
        # 返回当前进程的 CUDA 设备名称
        return f"cuda:{self.rank}"

    def setUp(self):
        super().setUp()
        # 设置环境变量 TORCH_NCCL_ASYNC_ERROR_HANDLING 为 "1"，用于测试目的
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # 调用父类的 setUp 方法来初始化测试环境
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            # 尝试删除临时文件，如果文件不存在则忽略异常
            os.remove(self.file_name)
        except OSError:
            pass

    def _test_broadcast_coalesced(self, process_group, device, root_rank):
        half = torch.float16

        # 如果设备是 CPU，则使用 float32 类型，因为 CPU 不支持 float16
        if device == torch.device("cpu"):
            half = torch.float32

        # 创建一个包含多个块的张量列表 target
        target = torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float64, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)

        # 如果当前进程是广播的根节点，则复制 target 张量到 tensors 中
        if self.rank == root_rank:
            tensors = [tensor.clone() for tensor in target]
        else:
            # 如果当前进程不是广播的根节点，则创建与 target 张量相同形状的零张量列表
            tensors = [torch.zeros_like(tensor) for tensor in target]

        # 如果当前进程不是广播的根节点，则验证 tensors 和 target 不相等
        if self.rank != root_rank:
            self.assertNotEqual(tensors, target)

        # 使用 c10d._broadcast_coalesced 方法广播 tensors 张量列表
        c10d._broadcast_coalesced(
            process_group, tensors, buffer_size=256, src=root_rank
        )

        # 如果当前进程不是广播的根节点，则验证 tensors 和 target 相等
        if self.rank != root_rank:
            self.assertEqual(tensors, target)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_broadcast_coalesced_nccl(self):
        # 使用 c10d.FileStore 创建存储，用于初始化进程组
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化 NCCL 进程组
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        # 获取默认的分布式进程组
        process_group = c10d.distributed_c10d._get_default_group()
        # 设置当前进程的设备为 CUDA 设备
        device = torch.device("cuda:%d" % self.rank)
        ranks = [0, 1]
        # 针对每个 root_rank 进行广播测试
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    # 定义一个测试方法，用于测试使用 NCCL 进行全局归约操作（coalesced），不需要任何参数
    def test_all_reduce_coalesced_nccl(self):
        # 创建一个文件存储对象，用于分布式进程组通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化分布式进程组，使用 NCCL 后端，传入文件存储对象、进程的rank和总进程数
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        # 获取默认的进程组
        process_group = c10d.distributed_c10d._get_default_group()
        # 设置当前设备为 CUDA 设备的指定 GPU
        device = torch.device("cuda:%d" % self.rank)
        # 创建一组张量列表，每个张量都是在指定设备上创建的，类型为 float
        tensors = [
            torch.full((60 + i,), self.rank + 1 + i, device=device, dtype=torch.float)
            for i in range(5)
        ]
        # 执行全局归约操作（coalesced），将结果存入传入的进程组
        torch.distributed.all_reduce_coalesced(tensors, group=process_group)
        # 验证每个张量是否等于其对应的期望值
        for i, t in enumerate(tensors):
            self.assertEqual(
                t,
                torch.full_like(
                    t, self.world_size * (i + (self.world_size + 1.0) / 2.0)
                ),
            )

    # 装饰器声明，要求当前测试需要使用 NCCL
    @requires_nccl()
    # 装饰器声明，如果 GPU 小于 2，则跳过当前测试
    @skip_if_lt_x_gpu(2)
    # 定义一个测试方法，用于测试使用 NCCL 进行 float8 类型的全局归约操作时的异常处理
    def test_all_reduce_coalesced_nccl_float8_errors(self):
        # 创建一个文件存储对象，用于分布式进程组通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化分布式进程组，使用 NCCL 后端，传入文件存储对象、进程的rank和总进程数
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        # 获取默认的进程组
        process_group = c10d.distributed_c10d._get_default_group()
        # 设置当前设备为 CUDA 设备的指定 GPU
        device = torch.device("cuda:%d" % self.rank)
        # 创建一组张量列表，每个张量都是在指定设备上创建的，类型为 float8，这里会触发异常
        tensors = [
            torch.full(
                (60 + i,), self.rank + 1 + i, device=device, dtype=torch.float
            ).to(torch.float8_e4m3fn)
            for i in range(5)
        ]
        # 使用断言来验证是否捕获到特定的运行时异常
        with self.assertRaisesRegex(
            RuntimeError,
            "Float8 dtypes are not currenlty supported for NCCL reductions",
        ):
            # 执行全局归约操作（coalesced），将结果存入传入的进程组
            torch.distributed.all_reduce_coalesced(tensors, group=process_group)

    # 装饰器声明，要求当前测试需要使用 NCCL
    @requires_nccl()
    # 装饰器声明，如果 GPU 小于 2，则跳过当前测试
    @skip_if_lt_x_gpu(2)
    # 定义一个测试方法，用于测试使用 NCCL 进行全局归约操作（coalesced）的管理器功能
    def test_all_reduce_coalesced_manager_nccl(self):
        # 创建一个文件存储对象，用于分布式进程组通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化分布式进程组，使用 NCCL 后端，传入文件存储对象、进程的rank和总进程数
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        # 获取默认的进程组
        process_group = c10d.distributed_c10d._get_default_group()
        # 设置当前设备为 CUDA 设备的指定 GPU
        device = torch.device("cuda:%d" % self.rank)
        # 创建一组张量列表，每个张量都是在指定设备上创建的，类型为 float
        tensors = [
            torch.full((60 + i,), self.rank + 1 + i, device=device, dtype=torch.float)
            for i in range(5)
        ]
        # 使用 torch.distributed._coalescing_manager 管理器来执行全局归约操作
        with torch.distributed._coalescing_manager(
            group=process_group, device=device, async_ops=True
        ) as cm:
            # 对每个张量执行全局归约操作
            for tensor in tensors:
                torch.distributed.all_reduce(tensor)
        # 验证管理器中工作的数量是否正确
        self.assertEqual(len(cm.works), 1)
        # 等待所有异步操作完成
        cm.wait()
        # 验证每个张量是否等于其对应的期望值
        for i, t in enumerate(tensors):
            self.assertEqual(
                t,
                torch.full_like(
                    t, self.world_size * (i + (self.world_size + 1.0) / 2.0)
                ),
            )
    def test_intra_node_comm_all_reduce(self):
        # 导入需要的模块和函数
        from torch._C._distributed_c10d import _get_intra_node_comm_usage_counter
        from torch.testing._internal.common_cuda import SM80OrLater

        # 遍历每个节点进行通信测试
        for peer in range(self.world_size):
            if peer == self.rank:
                continue
            # 检查当前设备是否可以与peer节点直接通信，否则跳过测试
            if not torch._C._cuda_canDeviceAccessPeer(self.rank, peer):
                raise SkipTest("Test requires p2p access")

        # 检查CUDA架构是否支持SM80或更高版本，否则跳过测试
        if not SM80OrLater:
            raise SkipTest("Test requires sm>=80")

        # 使用FileStore创建分布式存储，使用环境变量启用内节点通信测试
        store = c10d.FileStore(self.file_name, self.world_size)
        os.environ["ENABLE_INTRA_NODE_COMM"] = "1"
        os.environ["TEST_INTRA_NODE_COMM"] = "1"
        # 设置当前设备为指定rank
        torch.cuda.set_device(self.rank)
        # 初始化进程组，使用nccl后端
        c10d.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )
        expect = self.world_size * (self.world_size - 1) // 2

        # 内节点通信目前仅支持SUM和BF16操作
        # 验证在下面两种配置中未使用IntraNodeComm
        t = torch.full((4 * 1024 // 2,), self.rank).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())
        self.assertEqual(_get_intra_node_comm_usage_counter(), 0)

        t = torch.full((4 * 1024 // 2,), self.rank, dtype=torch.bfloat16).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.AVG)
        self.assertEqual(_get_intra_node_comm_usage_counter(), 0)

        # 验证IntraNodeComm在使用了最多10MB后被计数
        t = torch.full((4 * 1024 // 2,), self.rank, dtype=torch.bfloat16).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())
        self.assertEqual(_get_intra_node_comm_usage_counter(), 1)

        t = torch.full((512 * 1024 // 2,), self.rank, dtype=torch.bfloat16).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())
        self.assertEqual(_get_intra_node_comm_usage_counter(), 2)

        t = torch.full((10 * 1024**2 // 2,), self.rank, dtype=torch.bfloat16).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())
        self.assertEqual(_get_intra_node_comm_usage_counter(), 3)

        # 验证超过10MB后不再使用IntraNodeComm
        t = torch.full(
            (10 * 1024**2 // 2 + 1,), self.rank, dtype=torch.bfloat16
        ).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())
        self.assertEqual(_get_intra_node_comm_usage_counter(), 3)

        # 销毁进程组
        c10d.destroy_process_group()
    # 装饰器，要求在运行测试函数前检查是否满足NCCL的条件
    @requires_nccl()
    # 测试：检查序列号在NCCL子组中增加
    def test_sequence_num_incremented_nccl_subgroup(self):
        # 如果集群规模小于4，则跳过测试
        if self.world_size < 4:
            return skip_but_pass_in_sandcastle("Test requires world_size of at least 4")
        # 调用具体的测试函数，使用NCCL作为后端
        self._test_sequence_num_incremented_subgroup("nccl")

    # 装饰器，要求在运行测试函数前检查是否满足NCCL的条件
    @requires_nccl()
    # 装饰器，如果GPU数量小于2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 测试：设置NCCL新组中的序列号
    def test_sequence_num_set_nccl_new_group(self):
        # 设置当前CUDA设备为当前进程的排名
        torch.cuda.set_device(self.rank)
        # 调用具体的测试函数，使用NCCL作为后端
        self._test_sequence_num_set_new_group(backend="nccl")

    # 测试函数：验证NCCL选项的传递
    def _test_pass_nccl_options(self, pg_opts):
        # 创建文件存储对象，用于进程组的初始化
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化NCCL进程组，传递进程组选项
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            pg_options=pg_opts,
        )

        # 使用新的进程组选项创建进程组
        pg = c10d.new_group([0, 1], pg_options=pg_opts)
        # 测试进程组的功能是否符合预期
        t = torch.tensor([self.rank + 1] * 10).cuda(self.rank)
        pg.allreduce(t).wait()
        expected_tensor = torch.tensor([3] * 10).cuda(self.rank)
        self.assertEqual(expected_tensor, t)

    # 装饰器，要求在运行测试函数前检查是否满足NCCL的条件
    @requires_nccl()
    # 装饰器，如果GPU数量小于2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 测试：使用高优先级流传递NCCL选项
    def test_pass_nccl_options_high_priority_stream(self):
        # 创建NCCL进程组选项对象
        pg_opts = c10d.ProcessGroupNCCL.Options()
        pg_opts.is_high_priority_stream = True
        # 调用测试NCCL选项传递的函数
        self._test_pass_nccl_options(pg_opts)

    # 装饰器，要求在运行测试函数前检查是否满足NCCL的条件
    @requires_nccl()
    # 要求NCCL版本至少为2.17，否则跳过测试
    @requires_nccl_version(
        (2, 17), "Need NCCL 2.17+ for configuring NCCL communicators"
    )
    # 装饰器，如果GPU数量小于2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 测试：传递NCCL选项进行配置
    def test_pass_nccl_options_config(self):
        # 创建NCCL进程组选项对象
        pg_opts = c10d.ProcessGroupNCCL.Options()
        # 配置NCCL选项的最大CTAs、最小CTAs、CGA集群大小和网络名称
        pg_opts.config.max_ctas = 4
        pg_opts.config.min_ctas = 2
        pg_opts.config.cga_cluster_size = 2
        pg_opts.config.net_name = "Socket"
        # 创建临时文件来存储NCCL调试信息
        nccl_debug_file = tempfile.NamedTemporaryFile()
        # 设置环境变量NCCL_DEBUG和NCCL_DEBUG_FILE以启用调试
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_DEBUG_FILE"] = nccl_debug_file.name

        # 调用测试NCCL选项传递的函数
        self._test_pass_nccl_options(pg_opts)

        # 验证是否正确配置了通信
        nccl_debug_file_content = nccl_debug_file.read()
        max_ctas = re.search(rb"Max CTAs.*(\d+)|$", nccl_debug_file_content).group(1)
        min_ctas = re.search(rb"Min CTAs.*(\d+)|$", nccl_debug_file_content).group(1)
        cga_cluster_size = re.search(
            rb"CGA cluster.*(\d+)|$", nccl_debug_file_content
        ).group(1)
        net_name = re.search(
            rb"Using network.([a-zA-z]+)|$", nccl_debug_file_content
        ).group(1)
        # 断言各配置是否与预期一致
        self.assertEqual(pg_opts.config.max_ctas, int(max_ctas))
        self.assertEqual(pg_opts.config.min_ctas, int(min_ctas))
        self.assertEqual(pg_opts.config.cga_cluster_size, int(cga_cluster_size))
        self.assertEqual(pg_opts.config.net_name, net_name.decode())

    # 装饰器，要求在运行测试函数前检查是否满足NCCL的条件
    @requires_nccl()
    # 如果GPU数量小于4，则跳过测试
    @skip_if_lt_x_gpu(4)
    # 定义一个测试函数，用于测试使用 NCCL 后端的进程同步和通信
    def test_nccl_barrier(self):
        # 创建一个基于文件的存储对象，用于多进程间通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用 NCCL 后端，设置当前进程的排名和总进程数，并指定存储对象
        c10d.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )

        # 创建一个大小为 10 的张量，每个元素值为当前进程排名加一，放置在指定 GPU 上
        t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
        # 对所有进程执行张量的全局归约操作
        c10d.all_reduce(t)
        # 创建预期结果张量，每个元素值为 3，放置在指定 GPU 上
        expected_tensor = torch.tensor([3] * 10).cuda(2 * self.rank)
        # 断言当前张量与预期结果张量相等
        self.assertEqual(expected_tensor, t)

        # 使用新的进程组进行测试
        pg = c10d.new_group([0, 1])
        # 重新创建张量 t，并执行新进程组的全局归约操作，等待操作完成
        t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
        pg.allreduce(t).wait()
        # 断言当前张量与预期结果张量相等
        self.assertEqual(expected_tensor, t)

        # 使用只包含进程 0 的新进程组进行测试（仅进程 0 执行以下操作）
        pg = c10d.new_group([0])
        if self.rank == 0:
            # 重新创建张量 t，并执行进程组的全局归约操作，等待操作完成
            t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            expected_tensor = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            pg.allreduce(t).wait()
            # 断言当前张量与预期结果张量相等
            self.assertEqual(expected_tensor, t)

        # 使用只包含进程 1 的新进程组进行测试（仅进程 1 执行以下操作）
        pg = c10d.new_group([1])
        if self.rank == 1:
            # 重新创建张量 t，并执行进程组的全局归约操作，等待操作完成
            t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            expected_tensor = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            pg.allreduce(t).wait()
            # 断言当前张量与预期结果张量相等
            self.assertEqual(expected_tensor, t)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    # 测试使用 device_ids 参数执行 barrier 操作
    def test_nccl_barrier_device_ids(self):
        # 创建一个基于文件的存储对象，用于多进程间通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用 NCCL 后端，设置当前进程的排名和总进程数，并指定存储对象
        c10d.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )

        # 执行 barrier 操作，指定当前进程的 GPU 设备 ID
        c10d.barrier(device_ids=[self.rank])

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    # 测试使用单个 GPU 设备 ID 作为 barrier 函数的参数（应引发 TypeError 异常）
    def test_nccl_barrier_device_ids_function_argument(self):
        # 创建一个基于文件的存储对象，用于多进程间通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用 NCCL 后端，设置当前进程的排名和总进程数，并指定存储对象
        c10d.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )

        # 使用断言检测 barrier 函数调用时提供单个 GPU 设备 ID 的异常情况
        with self.assertRaisesRegex(TypeError, "Invalid function argument"):
            c10d.barrier(device_ids=self.rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["DETAIL"])
    # 测试在调试级别为 DETAIL 时警告未在进程组中的情况
    def test_nccl_warn_not_in_group_debug_detail(self):
        # 调用 _test_warn_not_in_group 方法，使用 NCCL 后端，调试级别为 DETAIL
        self._test_warn_not_in_group(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["INFO"])
    # 测试在调试级别为 INFO 时警告未在进程组中的情况
    def test_nccl_warn_not_in_group_debug_info(self):
        # 调用 _test_warn_not_in_group 方法，使用 NCCL 后端，调试级别为 INFO
        self._test_warn_not_in_group(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["OFF"])
    # 测试在调试级别为 OFF 时警告未在进程组中的情况
    def test_nccl_warn_not_in_group_debug_off(self):
        # 调用 _test_warn_not_in_group 方法，使用 NCCL 后端，调试级别为 OFF
        self._test_warn_not_in_group(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    # 测试检查进程在 NCCL 进程组中的成员身份
    def test_nncl_rank_membership(self):
        # 调用 _test_rank_membership 方法，使用 NCCL 后端
        self._test_rank_membership(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    # 测试检查张量数据类型不匹配的情况
    def test_tensor_dtype_mismatch(self):
        # 调用 _test_tensor_dtype_mismatch 方法，使用 NCCL 后端
        self._test_tensor_dtype_mismatch(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    # 其他未完整显示的测试函数定义...
    # 定义一个测试方法，用于测试复杂张量的数据类型
    def test_tensor_dtype_complex(self):
        # 调用内部方法 _test_tensor_dtype_complex，使用后端为 "nccl"
        self._test_tensor_dtype_complex(backend="nccl")
# 定义一个继承自test_c10d_common.CompilerTest的测试类CompilerTest
class CompilerTest(test_c10d_common.CompilerTest):

    # 定义一个属性world_size，返回值为2，表示进程组的大小为2
    @property
    def world_size(self):
        return 2

    # 定义一个方法_get_default_group，用于获取默认的分布式通信组
    def _get_default_group(self):
        # 创建一个文件存储对象，用于进程之间的通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用NCCL后端，指定当前进程的rank和总的进程数，以及文件存储对象
        dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        # 返回默认的C10d分布式通信组
        return dist.distributed_c10d._get_default_group()

    # 定义一个测试方法test_allreduce_work_wait_gpu，测试全局归约操作在GPU上的工作情况
    @skip_if_lt_x_gpu(2)
    def test_allreduce_work_wait_gpu(self):
        # 调用测试方法_test_allreduce_work_wait，传入一个在当前rank设备上全为当前rank值的2x2张量
        self._test_allreduce_work_wait(
            torch.ones(2, 2, device=self.rank) * self.rank,
        )

    # 定义一个测试方法test_allgather_work_wait_gpu，测试全局聚集操作在GPU上的工作情况
    @skip_if_lt_x_gpu(2)
    def test_allgather_work_wait_gpu(self):
        # 调用测试方法_test_allgather_work_wait，传入一个在当前rank设备上全为当前rank值的2x2张量
        self._test_allgather_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    # 定义一个测试方法test_allgather_into_tensor_work_wait_gpu，测试全局聚集到张量操作在GPU上的工作情况
    @skip_if_lt_x_gpu(2)
    def test_allgather_into_tensor_work_wait_gpu(self):
        # 调用测试方法_test_allgather_into_tensor_work_wait，传入一个在当前rank设备上全为当前rank值的2x2张量
        self._test_allgather_into_tensor_work_wait(
            torch.ones(2, 2, device=self.rank) * self.rank
        )

    # 定义一个测试方法test_reduce_scatter_work_wait_gpu，测试分散-归约操作在GPU上的工作情况
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_work_wait_gpu(self):
        # 调用测试方法_test_reduce_scatter_work_wait，传入一个在当前rank设备上全为当前rank值的2x2张量
        self._test_reduce_scatter_work_wait(
            torch.ones(2, 2, device=self.rank) * self.rank
        )

    # 定义一个测试方法test_reduce_scatter_tensor_work_wait_gpu，测试张量分散-归约操作在GPU上的工作情况
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor_work_wait_gpu(self):
        # 调用测试方法_test_reduce_scatter_tensor_work_wait，传入一个在当前rank设备上全为当前rank值的4x4张量
        self._test_reduce_scatter_tensor_work_wait(
            torch.ones(4, 4, device=self.rank) * self.rank
        )

    # 定义一个测试方法test_broadcast_work_wait_gpu，测试广播操作在GPU上的工作情况
    @skip_if_lt_x_gpu(2)
    def test_broadcast_work_wait_gpu(self):
        # 调用测试方法_test_broadcast_work_wait，传入一个在当前rank设备上全为当前rank值的2x2张量
        self._test_broadcast_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    # 定义一个测试方法test_scatter_work_wait_gpu，测试分散操作在GPU上的工作情况
    @skip_if_lt_x_gpu(2)
    def test_scatter_work_wait_gpu(self):
        # 调用测试方法_test_scatter_work_wait，传入一个在当前rank设备上全为当前rank值的2x2张量
        self._test_scatter_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    # 定义一个测试方法test_alltoall_work_wait_gpu，测试全互换操作在GPU上的工作情况
    @skip_if_lt_x_gpu(2)
    def test_alltoall_work_wait_gpu(self):
        # 调用测试方法_test_alltoall_work_wait，传入一个在当前rank设备上全为当前rank值的2x2张量
        self._test_alltoall_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    # 定义一个测试方法test_nested_comm_tensor_wrapping，测试嵌套通信张量包装操作
    @skip_if_lt_x_gpu(2)
    def test_nested_comm_tensor_wrapping(self):
        # 调用测试方法_test_nested_comm_tensor_wrapping，传入一个在当前rank设备上全为当前rank值的2x2张量
        self._test_nested_comm_tensor_wrapping(
            torch.ones(2, 2, device=self.rank) * self.rank
        )

    # 定义一个测试方法test_consecutive_comm_work_wait_gpu，测试连续通信操作在GPU上的工作情况
    @skip_if_lt_x_gpu(2)
    def test_consecutive_comm_work_wait_gpu(self):
        # 调用测试方法_test_consecutive_comm_work_wait，传入一个在当前rank设备上全为当前rank值的2x2张量
        self._test_consecutive_comm_work_wait(
            torch.ones(2, 2, device=self.rank) * self.rank
        )

    # 定义一个需要NCCL支持的测试方法test_reduce_scatter_base_k，测试基于k的张量分散-归约操作
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_base_k(self):
        # 创建一个文件存储对象，用于进程之间的通信
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用NCCL后端，指定当前进程的rank和总的进程数，以及文件存储对象
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 创建一个全为0的2x1张量，数据类型为int64，放置在当前rank设备上
        output_tensor = torch.zeros(2, dtype=torch.int64).to(self.rank)
        # 创建一个从0到2*world_size-1的整数张量，数据类型为int64，放置在当前rank设备上
        input_tensors = torch.arange(self.world_size * 2, dtype=torch.int64).to(
            self.rank
        )
        # 将输入张量重塑为形状为(world_size, 2)的张量
        input_tensors = torch.reshape(input_tensors, (self.world_size, 2))
        # 执行张量分散-归约操作，将结果存储到output_tensor中
        dist.reduce_scatter_tensor(output_tensor, input_tensors)
        # 断言output_tensor的值与input_tensors中对应rank行的值乘以world_size相等
        self.assertEqual(output_tensor, input_tensors[self.rank] * self.world_size)
    # 测试 reduce_scatter_tensor_coalesced 方法
    def test_reduce_scatter_tensor_coalesced(self):
        # 创建一个文件存储对象
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用 NCCL 后端
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 创建一个全零的输出张量
        output_tensors = torch.zeros(2, 2).to(self.rank)
        # 创建多个全一的输入张量列表
        input_tensors = [torch.ones(2, 2).to(self.rank) for _ in range(self.world_size)]
        # 使用 coalescing_manager 上下文管理器
        with dist._coalescing_manager():
            # 对每个进程进行 reduce_scatter_tensor 操作
            for i in range(self.world_size):
                dist.reduce_scatter_tensor(output_tensors[i], input_tensors[i])
        # 断言输出张量等于输入张量乘以进程数
        self.assertEqual(output_tensors, input_tensors[self.rank] * self.world_size)

    # 测试 reduce_scatter_base_k_float8_errors 方法
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_base_k_float8_errors(self):
        # 创建一个文件存储对象
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用 NCCL 后端
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 创建一个全零的输出张量，使用不支持的 float8_e4m3fn 数据类型
        output_tensor = (
            torch.zeros(2, dtype=torch.float32).to(torch.float8_e4m3fn).to(self.rank)
        )
        # 创建一个输入张量，使用不支持的 float8_e4m3fn 数据类型
        input_tensors = (
            torch.arange(self.world_size * 2, dtype=torch.float32)
            .to(torch.float8_e4m3fn)
            .to(self.rank)
        )
        # 重塑输入张量的形状
        input_tensors = torch.reshape(input_tensors, (self.world_size, 2))
        # 断言抛出异常，说明不支持 float8 数据类型
        with self.assertRaisesRegex(
            RuntimeError,
            "Float8 dtypes are not currenlty supported for NCCL reductions",
        ):
            dist.reduce_scatter_tensor(output_tensor, input_tensors)

    # 测试 reduce_scatter_tensor_coalesced_float8_errors 方法
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor_coalesced_float8_errors(self):
        # 创建一个文件存储对象
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用 NCCL 后端
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 创建一个全零的输出张量，使用不支持的 float8_e5m2 数据类型
        output_tensors = torch.zeros(2, 2).to(torch.float8_e5m2).to(self.rank)
        # 创建多个全一的输入张量列表，使用不支持的 float8_e5m2 数据类型
        input_tensors = [
            torch.ones(2, 2).to(torch.float8_e5m2).to(self.rank)
            for _ in range(self.world_size)
        ]

        # 断言抛出异常，说明不支持 float8 数据类型
        with self.assertRaisesRegex(
            RuntimeError,
            "Float8 dtypes are not currenlty supported for NCCL reductions",
        ):
            # 使用 coalescing_manager 上下文管理器
            with dist._coalescing_manager():
                # 对每个进程进行 reduce_scatter_tensor 操作
                for i in range(self.world_size):
                    dist.reduce_scatter_tensor(output_tensors[i], input_tensors[i])
            # 断言输出张量等于输入张量
            self.assertEqual(output_tensors, input_tensors[self.rank])
# 定义枚举类，列出设置设备方法的选项
class SetDeviceMethod(Enum):
    TORCH_CUDA_SET = auto()  # 通过调用 torch.cuda.set_device 进行设备设置
    COLLECTIVE_ARGUMENT = auto()  # 用于广播对象列表的参数


# 继承自 test_c10d_common.ProcessGroupWithDispatchedCollectivesTests 的测试类
class NcclProcessGroupWithDispatchedCollectivesTests(
    test_c10d_common.ProcessGroupWithDispatchedCollectivesTests
):
    @requires_nccl()  # 标记需要 NCCL 支持
    @skip_if_lt_x_gpu(1)  # 如果 GPU 数量小于 1 则跳过测试
    def test_collectives(self):
        self._test_collectives(backend="nccl")

    @requires_nccl()  # 标记需要 NCCL 支持
    @skip_if_lt_x_gpu(1)  # 如果 GPU 数量小于 1 则跳过测试
    def test_allreduce_coalesced(self):
        self._test_allreduce_coalesced(backend="nccl")

    @requires_nccl()  # 标记需要 NCCL 支持
    @skip_if_lt_x_gpu(1)  # 如果 GPU 数量小于 1 则跳过测试
    def test_all_to_all_single(self):
        self._test_all_to_all_single(backend="nccl")

    @requires_nccl()  # 标记需要 NCCL 支持
    @skip_if_lt_x_gpu(1)  # 如果 GPU 数量小于 1 则跳过测试
    def test_allgather_base(self):
        # 创建文件存储对象，指定文件名和节点数
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用 NCCL 后端
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        device = "cuda"
        tensor = torch.ones(10, 10, device=torch.device(device))  # 创建一个大小为 10x10 的张量，放置在指定设备上
        output_tensor = torch.zeros(10, 10, device=torch.device(device))  # 创建一个相同大小的零张量，放置在指定设备上
        dist.all_gather_into_tensor(output_tensor, tensor)  # 执行全局收集操作
        self.assertEqual(output_tensor, tensor)  # 断言输出张量与输入张量相等

    @requires_nccl()  # 标记需要 NCCL 支持
    @skip_if_lt_x_gpu(1)  # 如果 GPU 数量小于 1 则跳过测试
    @parametrize("float8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_allgather_float8(self, float8_dtype):
        # 创建文件存储对象，指定文件名和节点数
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用 NCCL 后端
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        device = "cuda"
        # 创建一个大小为 10x16 的张量，放置在指定设备上，并转换为指定的 float8 数据类型
        tensor = torch.ones(10, 16, device=torch.device(device)).to(float8_dtype)
        # 创建一个相同大小的零张量，放置在指定设备上，并转换为指定的 float8 数据类型
        output_tensor = torch.zeros(10, 16, device=torch.device(device)).to(
            float8_dtype
        )
        dist.all_gather_into_tensor(output_tensor, tensor)  # 执行全局收集操作
        # 将输出张量视图转换为 float32 类型后，断言与输入张量视图相等
        self.assertEqual(output_tensor.view(torch.float32), tensor.view(torch.float32))


# 实例化参数化测试
instantiate_parametrized_tests(NcclProcessGroupWithDispatchedCollectivesTests)


# 继承自 test_c10d_common.AbstractLargeCommTest 和 MultiProcessTestCase 的大型通信测试类
class LargeCommTest(test_c10d_common.AbstractLargeCommTest, MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        # 设置环境变量 TORCH_NCCL_ASYNC_ERROR_HANDLING 为 1
        # 这将覆盖 TORCH_NCCL_BLOCKING_WAIT，因此使用 TORCH_NCCL_BLOCKING_WAIT 的测试将按预期进行测试
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()  # 调用函数以启动多进程

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)  # 尝试删除文件
        except OSError:
            pass

    @property
    def device(self):
        return self.rank  # 返回进程的排名作为设备标识符

    @requires_nccl()  # 标记需要 NCCL 支持
    @skip_if_lt_x_gpu(4)  # 如果 GPU 数量小于 4 则跳过测试
    def test_new_group_local_sync(self):
        self._test_new_group_local_sync(backend="nccl")

    @requires_nccl()  # 标记需要 NCCL 支持
    @skip_if_lt_x_gpu(4)  # 如果 GPU 数量小于 4 则跳过测试
    def test_new_group_local_sync_sanity_check(self):
        self._test_new_group_local_sync_sanity_check(backend="nccl")
    # 使用 @requires_nccl 装饰器，要求当前测试依赖于 NCCL 后端
    # 使用 @skip_if_lt_x_gpu 装饰器，如果 GPU 数量少于 4，则跳过该测试
    def test_new_group_local_sync_duplicated_pg(self):
        # 调用 _test_new_group_local_sync_duplicate_pg 方法，指定使用 NCCL 后端进行测试
        self._test_new_group_local_sync_duplicate_pg(backend="nccl")
    
    # 定义一个初始化方法 _init_two_pg2_subgroups，设置默认的世界大小为 4
    def _init_two_pg2_subgroups(self, world_size: int = 4):
        # 如果世界大小不为 4，则抛出 NotImplementedError 异常
        if world_size != 4:
            raise NotImplementedError(
                f"need world size of 4 to get 2 subgroup PGs, but got world size of {world_size}"
            )
        # 创建一个基于文件的存储对象，用于存储分布式进程组信息
        store = c10d.FileStore(self.file_name, world_size)
        # 初始化分布式进程组，指定后端为 NCCL，存储为之前创建的文件存储对象，指定当前进程的 rank 和总进程数
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=world_size
        )
        # 每个进程创建相同的两个子组，包括当前进程未使用的子组
        a_group = c10d.new_group([0, 1])
        b_group = c10d.new_group([2, 3])
        return a_group if self.rank < 2 else b_group
    
    # 使用 @requires_nccl 装饰器，要求当前测试依赖于 NCCL 后端
    # 使用 @skip_if_lt_x_gpu 装饰器，如果 GPU 数量少于 4，则跳过该测试
    def test_gather_subgroup(self):
        world_size = 4
        # 如果当前进程的 rank 大于等于世界大小，则直接返回，只测试确切为 4 个 GPU 的情况
        if self.rank >= world_size:
            return
    
        # 初始化两个包含两个子组的分布式进程组
        subgroup = self._init_two_pg2_subgroups(world_size)
        # 根据当前进程的 rank，指定使用的 CUDA 设备
        device = torch.device("cuda:%d" % self.rank)
        # 创建一个在当前设备上全为当前进程 rank 的 tensor 输入
        input = torch.ones((10,), device=device) * self.rank
        # 如果当前进程的 rank 为 0 或 2
        if self.rank == 0 or self.rank == 2:
            # 创建一个与输入 tensor 形状相同的空 tensor 列表，用于存放 gather 操作的结果
            gather_list = [torch.empty_like(input) for _ in range(subgroup.size())]
            # 执行 gather 操作，将输入 tensor 收集到 gather_list 中，目标进程为当前进程的 rank，使用指定的分布式进程组
            torch.distributed.gather(
                input,
                gather_list=gather_list,
                dst=self.rank,
                group=subgroup,
                async_op=False,
            )
            # 对于每个收集到的 tensor，验证其是否符合预期
            for src in range(len(gather_list)):
                expected = (torch.ones_like(input) * self.rank) + src
                self.assertEqual(gather_list[src], expected)
        else:
            # 对于 rank 不为 0 或 2 的进程，执行 gather 操作，目标进程为当前进程的 rank - 1，使用指定的分布式进程组
            torch.distributed.gather(
                input,
                gather_list=None,
                dst=self.rank - 1,
                group=subgroup,
                async_op=False,
            )
    def test_gather_object_subgroup(self):
        # 设置世界大小为4
        world_size = 4
        # 如果当前进程排名大于等于世界大小，直接返回，测试只为4个GPU编写
        if self.rank >= world_size:
            return

        # 初始化包含两个进程组的子组
        subgroup = self._init_two_pg2_subgroups(world_size)

        # 解决差异 #1
        # 必须设置设备，否则 gather_object 函数从 'current_device = _get_pg_default_device(group)' 中获取错误的设备
        torch.cuda.set_device(self.rank)

        # 创建输入字典
        input = {"rank": self.rank}
        # 如果当前进程排名为0或2
        if self.rank == 0 or self.rank == 2:
            # 解决差异 #2
            # 另一个奇怪的地方- 为什么要让我在列表中指定一些空对象？
            # 空列表应该是有效的，但这会引发错误
            gather_list = [{}, {}]
            # 使用分布式机制收集对象到 gather_list 中，目标进程为 self.rank，使用 subgroup 分组
            torch.distributed.gather_object(
                input, object_gather_list=gather_list, dst=self.rank, group=subgroup
            )
            # 验证 gather_list 中每个来源的数据是否正确
            for src in range(len(gather_list)):
                self.assertEqual(gather_list[src]["rank"], self.rank + src)
        else:
            # 使用分布式机制收集对象到 None（空）的 object_gather_list 中，目标进程为 self.rank - 1，使用 subgroup 分组
            torch.distributed.gather_object(
                input, object_gather_list=None, dst=self.rank - 1, group=subgroup
            )

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_reduce_subgroup(self):
        # 设置世界大小为4
        world_size = 4
        # 如果当前进程排名大于等于世界大小，直接返回
        if self.rank >= world_size:
            return
        # 初始化包含两个进程组的子组
        subgroup = self._init_two_pg2_subgroups(world_size)
        # 设置设备为当前进程的 CUDA 设备
        device = torch.device("cuda:%d" % self.rank)
        # 创建一个由当前进程排名构成的张量 x，使用当前设备
        x = torch.ones((10,), device=device) * self.rank
        # 如果当前进程排名为0或2
        if self.rank == 0 or self.rank == 2:
            # 创建期望结果张量，其值为 x 的每个元素加上 1，使用当前设备
            expected = x + torch.ones((10,), device=device) * (self.rank + 1)
            # 使用 c10d.reduce 函数在组 subgroup 中将张量 x 归约（reduce）到目标进程 self.rank，同步操作
            c10d.reduce(x, dst=self.rank, group=subgroup, async_op=False)
            # 验证 x 是否等于期望结果张量 expected
            self.assertEqual(x, expected)
        else:
            # 使用 c10d.reduce 函数在组 subgroup 中将张量 x 归约（reduce）到目标进程 self.rank - 1，同步操作
            c10d.reduce(x, dst=self.rank - 1, group=subgroup, async_op=False)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @parametrize("async_op", [True, False])
    def test_send_recv_subgroup(self, async_op):
        # 设置世界大小为4
        world_size = 4
        # 如果当前进程排名大于等于世界大小，直接返回
        if self.rank >= world_size:
            return
        # 初始化包含两个进程组的子组
        subgroup = self._init_two_pg2_subgroups(world_size)
        # 设置设备为当前进程的 CUDA 设备
        device = torch.device("cuda:%d" % self.rank)
        # 如果当前进程排名为0或2
        if self.rank == 0 or self.rank == 2:
            # 创建一个空张量 x，使用当前设备
            x = torch.empty((10,), device=device)
            # 如果 async_op 为 True，使用异步接收函数 c10d.irecv 接收从进程 self.rank + 1 发送的数据，使用 subgroup 分组
            if async_op:
                c10d.irecv(x, src=self.rank + 1, group=subgroup).wait()
            else:
                # 否则，使用同步接收函数 c10d.recv 接收从进程 self.rank + 1 发送的数据，使用 subgroup 分组
                c10d.recv(x, src=self.rank + 1, group=subgroup)
            # 创建期望结果张量，其值为每个元素都为 self.rank + 1，使用当前设备
            expected = torch.ones((10,), device=device) * (self.rank + 1)
            # 验证接收的张量 x 是否等于期望结果张量 expected
            self.assertEqual(x, expected)
        else:
            # 创建张量 x，其值为每个元素都为 self.rank，使用当前设备
            x = torch.ones((10,), device=device) * self.rank
            # 如果 async_op 为 True，使用异步发送函数 c10d.isend 发送数据 x 到进程 self.rank - 1，使用 subgroup 分组
            if async_op:
                c10d.isend(x, dst=self.rank - 1, group=subgroup).wait()
            else:
                # 否则，使用同步发送函数 c10d.send 发送数据 x 到进程 self.rank - 1，使用 subgroup 分组
                c10d.send(x, dst=self.rank - 1, group=subgroup)
    # 定义一个测试方法，用于测试广播子组功能
    def test_broadcast_subgroup(self):
        # 定义全局变量，表示总的进程数量为4
        world_size = 4
        # 如果当前进程的排名超过或等于总进程数量，直接返回，不执行后续代码
        if self.rank >= world_size:
            return
        # 初始化一个包含两个进程组的子组
        subgroup = self._init_two_pg2_subgroups(world_size)
        # 根据当前进程的排名，确定使用的 CUDA 设备
        device = torch.device("cuda:%d" % self.rank)
        # 如果当前进程的排名是0或2
        if self.rank == 0 or self.rank == 2:
            # 创建一个在指定 CUDA 设备上的空张量
            x = torch.empty((10,), device=device)
            # 对张量 x 执行广播操作，从排名为 self.rank+1 的进程接收数据，使用指定的子组
            c10d.broadcast(x, src=self.rank + 1, group=subgroup)
            # 创建一个期望的张量，其值为 (self.rank + 1) 的标量值重复 10 次
            expected = torch.ones((10,), device=device) * (self.rank + 1)
            # 断言张量 x 的值与期望值相等
            self.assertEqual(x, expected)
        else:
            # 创建一个在指定 CUDA 设备上，所有元素为 self.rank 的张量
            x = torch.ones((10,), device=device) * self.rank
            # 对张量 x 执行广播操作，从当前排名的进程发送数据，使用指定的子组
            c10d.broadcast(x, src=self.rank, group=subgroup)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @parametrize(
        "set_device",
        [SetDeviceMethod.TORCH_CUDA_SET, SetDeviceMethod.COLLECTIVE_ARGUMENT],
    )
    # 定义一个测试方法，用于测试发送接收对象列表到子组的功能
    def test_send_recv_object_list_subgroup(self, set_device: SetDeviceMethod):
        # 定义全局变量，表示总的进程数量为4
        world_size = 4
        # 如果当前进程的排名超过或等于总进程数量，直接返回，不执行后续代码
        if self.rank >= world_size:
            return
        # 初始化一个包含两个进程组的子组
        subgroup = self._init_two_pg2_subgroups(world_size)
        # 根据测试方法的设备设置方式，确定使用的设备类型和设备对象
        if set_device == SetDeviceMethod.TORCH_CUDA_SET:
            # 设置当前进程的 CUDA 设备
            torch.cuda.set_device(self.rank)
            device = None
        else:
            # 使用指定 CUDA 设备编号创建设备对象
            device = torch.device("cuda:%d" % self.rank)
        # 如果当前进程的排名是0或2
        if self.rank == 0 or self.rank == 2:
            # 创建一个空的对象列表
            x = [{}]
            # 从排名为 self.rank+1 的进程接收对象列表 x 的数据，使用指定的子组和设备
            c10d.recv_object_list(x, src=self.rank + 1, group=subgroup, device=device)
            # 创建一个期望的对象列表，包含一个字典，其中键为 "rank"，值为 self.rank + 1
            expected = [{"rank": self.rank + 1}]
            # 断言接收到的对象列表 x 与期望的对象列表 expected 相等
            self.assertEqual(x, expected)
        else:
            # 创建一个对象列表，包含一个字典，键为 "rank"，值为当前进程的排名 self.rank
            x = [{"rank": self.rank}]
            # 将对象列表 x 发送到排名为 self.rank-1 的进程，使用指定的子组和设备
            c10d.send_object_list(x, dst=self.rank - 1, group=subgroup, device=device)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @parametrize(
        "set_device",
        [SetDeviceMethod.TORCH_CUDA_SET, SetDeviceMethod.COLLECTIVE_ARGUMENT],
    )
    # 定义一个测试方法，用于测试广播对象列表到子组的功能
    def test_broadcast_object_list_subgroup(self, set_device: SetDeviceMethod):
        # 定义全局变量，表示总的进程数量为4
        world_size = 4
        # 如果当前进程的排名超过或等于总进程数量，直接返回，不执行后续代码
        if self.rank >= world_size:
            return
        # 初始化一个包含两个进程组的子组
        subgroup = self._init_two_pg2_subgroups(world_size)
        # 根据测试方法的设备设置方式，确定使用的设备类型和设备对象
        if set_device == SetDeviceMethod.TORCH_CUDA_SET:
            # 设置当前进程的 CUDA 设备
            torch.cuda.set_device(self.rank)
            device = None
        else:
            # 使用指定 CUDA 设备编号创建设备对象
            device = torch.device("cuda:%d" % self.rank)
        # 如果当前进程的排名是0或2
        if self.rank == 0 or self.rank == 2:
            # 创建一个空的对象列表
            x = [{}]
            # 从排名为 self.rank+1 的进程广播对象列表 x 的数据，使用指定的子组和设备
            c10d.broadcast_object_list(
                x, src=self.rank + 1, group=subgroup, device=device
            )
            # 创建一个期望的对象列表，包含一个字典，其中键为 "rank"，值为 self.rank + 1
            expected = [{"rank": self.rank + 1}]
            # 断言接收到的对象列表 x 与期望的对象列表 expected 相等
            self.assertEqual(x, expected)
        else:
            # 创建一个对象列表，包含一个字典，键为 "rank"，值为当前进程的排名 self.rank
            x = [{"rank": self.rank}]
            # 从当前进程广播对象列表 x 的数据，使用指定的子组和设备
            c10d.broadcast_object_list(x, src=self.rank, group=subgroup, device=device)
    # 定义一个测试方法，用于测试分布式环境下的scatter操作
    def test_scatter_subgroup(self):
        # 定义集群的大小为4
        world_size = 4
        # 如果当前进程的rank超过了集群的大小，则直接返回
        if self.rank >= world_size:
            return
        # 初始化一个包含两个子组的分组
        subgroup = self._init_two_pg2_subgroups(world_size)
        # 设置当前设备为CUDA设备中的特定设备
        device = torch.device("cuda:%d" % self.rank)
        # 创建一个在指定设备上的空Tensor
        x = torch.empty((10,), device=device)
        # 创建一个期望的Tensor，包含当前rank值的Tensor
        expected = torch.ones((10,), device=device) * self.rank
        # 如果当前rank为0或2，则执行scatter操作，从rank+1的进程接收数据
        if self.rank == 0 or self.rank == 2:
            c10d.scatter(x, scatter_list=None, src=self.rank + 1, group=subgroup)
        else:
            # 否则，准备一个scatter_list，包含两个Tensor
            scatter_list = [
                torch.ones((10,), device=device) * (self.rank - 1),
                torch.ones((10,), device=device) * self.rank,
            ]
            # 执行scatter操作，将数据发送到当前rank进程
            c10d.scatter(x, scatter_list=scatter_list, src=self.rank, group=subgroup)
        # 断言操作结果与期望值相等
        self.assertEqual(x, expected)

    # 要求使用NCCL后端进行测试
    @requires_nccl()
    # 要求至少有4个GPU才能运行该测试
    @skip_if_lt_x_gpu(4)
    # 定义一个测试方法，用于测试分布式环境下的对象列表scatter操作
    def test_scatter_object_list_subgroup(self):
        # 定义集群的大小为4
        world_size = 4
        # 如果当前进程的rank超过了集群的大小，则直接返回
        if self.rank >= world_size:
            return
        # 初始化一个包含两个子组的分组
        subgroup = self._init_two_pg2_subgroups(world_size)
        # 设置当前CUDA设备的设备ID为当前rank值
        torch.cuda.set_device(self.rank)
        # 初始化一个用于接收scatter结果的对象列表
        scatter_object_output_list = [None]
        # 期望的结果是一个包含当前rank信息的字典
        expected = [{"rank": self.rank}]
        # 如果当前rank为0或2，则执行scatter操作，从rank+1的进程接收数据
        if self.rank == 0 or self.rank == 2:
            c10d.scatter_object_list(
                scatter_object_output_list=scatter_object_output_list,
                scatter_object_input_list=None,
                src=self.rank + 1,
                group=subgroup,
            )
        else:
            # 否则，准备一个scatter操作的输入对象列表
            scatter_object_input_list = [
                {"rank": self.rank - 1},
                {"rank": self.rank},
            ]
            # 执行scatter操作，将对象列表发送到当前rank进程
            c10d.scatter_object_list(
                scatter_object_output_list=scatter_object_output_list,
                scatter_object_input_list=scatter_object_input_list,
                src=self.rank,
                group=subgroup,
            )
        # 断言操作结果与期望值相等
        self.assertEqual(scatter_object_output_list, expected)
# 实例化一个参数化测试类 LargeCommTest
instantiate_parametrized_tests(LargeCommTest)

# 定义 SparseCollective 类，继承自 MultiProcessTestCase
class SparseCollective(MultiProcessTestCase):

    # 定义属性 world_size，返回值为 1
    @property
    def world_size(self):
        return 1

    # 设置测试环境，在执行每个测试方法前调用
    def setUp(self):
        super().setUp()
        # 设置环境变量 TORCH_NCCL_ASYNC_ERROR_HANDLING 为 "1"，
        # 覆盖 TORCH_NCCL_BLOCKING_WAIT 设置，确保测试行为符合预期
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # 调用 _spawn_processes 方法，启动多进程测试
        self._spawn_processes()

    # 清理测试环境，在执行每个测试方法后调用
    def tearDown(self):
        super().tearDown()
        try:
            # 尝试删除 self.file_name 指定的文件
            os.remove(self.file_name)
        except OSError:
            pass

    # 定义内部类 ToyModel，继承自 nn.Module
    class ToyModel(nn.Module):
        def __init__(self, rank, vocab_size, embedding_dim):
            super().__init__()
            # 创建一个稀疏 Embedding 层，并将其部署到指定的 rank
            self.embedding = nn.Embedding(vocab_size, embedding_dim, sparse=True).to(
                rank
            )
            # 创建一个线性层，并将其部署到指定的 rank
            self.linear = nn.Linear(embedding_dim, 1).to(rank)

        def forward(self, inputs):
            # 前向传播函数，计算嵌入后的均值
            embedded = self.embedding(inputs)
            # embedded 的形状: (batch_size, sequence_length, embedding_dim)
            flattened = torch.mean(embedded, dim=1)
            # flattened 的形状: (batch_size, embedding_dim)
            output = self.linear(flattened)
            # output 的形状: (batch_size, 1)
            return output

    # 标记为需要 NCCL 支持的分布式测试，并至少要求 1 个 GPU
    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_ddp_set_sparse_metadata(self):
        # 创建一个基于文件的存储，用于进程之间的通信
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用 NCCL 后端
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

        vocab_size = 5

        # 创建 ToyModel 实例，使用分布式数据并行
        model = SparseCollective.ToyModel(
            self.rank, vocab_size=vocab_size, embedding_dim=10
        )
        ddp_model = DistributedDataParallel(model)
        # 创建输入张量，部署到指定的 rank
        inputs = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]]).to(self.rank)
        # 设置 DDP 模型的稀疏元数据
        indices = torch.Tensor(list(range(vocab_size)))
        ddp_model._set_sparse_metadata({"embedding.weight": indices})
        # 执行前向传播
        try:
            output = ddp_model(inputs)
            loss = output.sum()

            # 执行反向传播
            loss.backward()
            # 断言稀疏权重的梯度与预期的稀疏索引一致
            self.assertTrue(ddp_model.module.embedding.weight.grad.indices, indices)
        except RuntimeError as e:
            # 如果抛出的 RuntimeError 包含 "NCCL does not support all_reduce with sparse tensors"，
            # 则忽略这个错误
            if "NCCL does not support all_reduce with sparse tensors" in str(e):
                pass
            else:
                # 如果是其他错误，则重新抛出异常
                raise
    def setUp(self):
        super().setUp()
        # 设置环境变量，禁用 TORCH_NCCL_ENABLE_TIMING，以便在测试中跳过时间记录
        os.environ[
            "TORCH_NCCL_ENABLE_TIMING"
        ] = "0"  # see 'timing_enabled' parametrized tests
        # 设置环境变量，指定 TORCH_NCCL_TRACE_BUFFER_SIZE 为 1000
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1000"
        # 设置环境变量，启用 TORCH_NCCL_DUMP_ON_TIMEOUT 来在超时时转储信息
        os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
        # 创建临时目录对象
        self.tempdir = tempfile.TemporaryDirectory()
        # 设置环境变量，指定 TORCH_NCCL_DEBUG_INFO_TEMP_FILE 和 TORCH_NCCL_DEBUG_INFO_PIPE_FILE
        # 使用临时目录的基本路径
        os.environ["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] = self._trace_basename()
        os.environ["TORCH_NCCL_DEBUG_INFO_PIPE_FILE"] = self._trace_basename()
        # 启动多进程
        self._spawn_processes()

    @classmethod
    def _run(
        cls, parent_conn, rank: int, test_name: str, file_name: str, parent_pipe
    ) -> None:
        # 设置类属性 parent 为 parent_conn
        cls.parent = parent_conn
        # 调用父类的 _run 方法，传递参数 rank, test_name, file_name, parent_pipe
        super()._run(rank, test_name, file_name, parent_pipe)

    @property
    def local_device(self):
        # 返回当前进程对应的 CUDA 设备
        return torch.device("cuda", self.rank_to_GPU[self.rank][0])

    def _join_processes(self, fn):
        # 我们需要对 sys.exit() 进行打补丁，因为 skip_if 会使用 sys.exit()
        # 这样当前进程的退出码不会被捕获到。
        with mock.patch("sys.exit") as exit_mock:
            # 执行传入的函数 fn
            fn()
        # 调用父类的 _join_processes 方法，传入函数 fn
        super()._join_processes(fn)

    def _spawn_processes(self) -> None:
        # 获取 spawn 上下文的进程对象
        proc = torch.multiprocessing.get_context("spawn").Process
        # 初始化 children_pipes 和 parent_pipes
        self.children_pipes = []
        parent_pipes = []
        # 创建 self.world_size 个进程间管道，并添加到相应列表中
        for i in range(self.world_size):
            parent_conn, child_conn = torch.multiprocessing.Pipe()
            self.children_pipes.append(child_conn)
            parent_pipes.append(parent_conn)
        # 创建 parent_pipes 的迭代器
        piter = iter(parent_pipes)

        def wrap(*positional, args, **kwargs):
            # 将下一个 parent_conn 添加到参数中，创建进程
            args = (next(piter), *args)
            return proc(*positional, args=args, **kwargs)

        # 启动进程
        self._start_processes(wrap)

    def _create_process_group_nccl(self):
        # 创建 dist.FileStore 对象
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化 NCCL 进程组
        c10d.init_process_group(
            "nccl", world_size=self.world_size, rank=self.rank, store=store
        )
        # 获取默认的进程组
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    def tearDown(self):
        super().tearDown()
        # 尝试删除 self.file_name 指定的文件（如果存在）
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        # 返回当前进程组的大小
        return 2

    @property
    def rank_to_GPU(self):
        # 返回将 rank 映射到 GPU 的字典
        return init_multigpu_helper(self.world_size, "nccl")

    def _trace_basename(self):
        # 返回跟踪文件的基本路径，用于设置环境变量
        return os.path.join(self.tempdir.name, "trace_")

    def _trace_name(self, rank):
        # 返回跟踪文件的完整路径名，根据给定的 rank
        return self._trace_basename() + str(rank)

    def started_or_scheduled(self, timing_enabled):
        # 根据 timing_enabled 的值返回 "started" 或 "scheduled"
        return "started" if timing_enabled else "scheduled"
# 定义 NCCLTraceTest 类，继承自 NCCLTraceTestBase 类
class NCCLTraceTest(NCCLTraceTestBase):
    # 应用 requires_nccl 装饰器，确保需要 NCCL 支持
    @requires_nccl()
    # 如果不满足 TEST_MULTIGPU 条件，则在沙堡环境中跳过此测试，但允许通过
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    # 参数化测试用例，使 timing_enabled 参数分别取 True 和 False 两个值
    @parametrize("timing_enabled", [True, False])
    # 参数化测试用例，使 include_collectives 参数分别取 True 和 False 两个值
    @parametrize("include_collectives", [True, False])
    # 定义一个测试方法，接受两个布尔参数：timing_enabled 表示是否启用时间跟踪，include_collectives 表示是否包含集合操作
    def test_short(self, timing_enabled, include_collectives):
        # 如果当前进程的排名等于主进程的排名，直接返回，不执行后续代码
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        # 创建一个基于 NCCL 的进程组
        pg = self._create_process_group_nccl()
        # 如果启用了时间跟踪，则启用集合操作的时间跟踪
        if timing_enabled:
            pg._enable_collectives_timing()
        # 获取本地设备信息
        device = self.local_device
        # 创建一个形状为 (3, 4) 的张量 a，张量的值为当前进程的排名，张量存储在指定的设备上
        a = torch.full((3, 4), float(self.rank), device=device)
        # 执行两次全局归约操作，将结果保存在 f 中
        for i in range(2):
            f = pg.allreduce(a)
        # 等待操作完成
        f.wait()
        # 同步 CUDA 设备，确保所有操作完成
        torch.cuda.synchronize(device=device)

        # 等待一秒钟，以确保 duration_ms 字段可以最大程度地填充，因为只能在 "dump()" API 之外才能发生
        time.sleep(1)
        # 根据 include_collectives 参数加载 NCCL 跟踪数据
        if include_collectives:
            t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
        else:
            t = pickle.loads(
                torch._C._distributed_c10d._dump_nccl_trace(
                    includeCollectives=False, includeStackTraces=None, onlyActive=None
                )
            )
        # 获取 NCCL 跟踪数据的版本号，并断言其为 "2.2"
        ver = t["version"]
        self.assertEqual(ver, "2.2")
        # 获取进程组的配置信息
        pg_config = t["pg_config"]
        self.assertEqual(len(pg_config), 1)
        # 获取默认进程组的信息
        default_pg_info = pg_config["0"]
        self.assertIn("name", default_pg_info)
        self.assertIn("desc", default_pg_info)
        self.assertIn("ranks", default_pg_info)
        # 解析全局排名信息并断言其长度与世界大小相等
        global_ranks = pg_config["0"]["ranks"]
        self.assertEqual(len(json.loads(global_ranks)), self.world_size)
        # 如果包含集合操作，则继续验证相关条目
        if include_collectives:
            # 断言条目数为 2
            self.assertEqual(len(t["entries"]), 2)
            t = t["entries"]
            self.assertEqual(len(t), 2)
            # 获取最后一个条目，并验证其相关字段
            last = t[-1]
            self.assertEqual(last["process_group"], ("0", "default_pg"))
            self.assertEqual(last["state"], "completed")
            s = last["time_discovered_started_ns"]
            f = last["time_discovered_completed_ns"]
            self.assertEqual(last["record_id"], 1)
            self.assertIsNotNone(f)
            # 如果启用了时间跟踪，验证开始时间早于结束时间
            if timing_enabled:
                self.assertIsNotNone(s)
                self.assertTrue(s <= f)
            # 验证栈帧中包含 "test_c10d_nccl.py"
            self.assertIn("test_c10d_nccl.py", str(last["frames"]))
            self.assertEqual(last["input_sizes"], ((3, 4),))
            self.assertEqual(last["input_dtypes"], ["Float"])
            self.assertEqual(last["output_sizes"], ((3, 4),))
            self.assertEqual(last["output_dtypes"], ["Float"])
            self.assertEqual(last["collective_seq_id"], 2)
            self.assertEqual(last["timeout_ms"], 600000)
            # 验证事件创建时间在当前时间的前后一分钟之间
            now = datetime.now()
            event_created_time = datetime.fromtimestamp(
                last["time_created_ns"] / 1000000000
            )
            before_test = now - timedelta(minutes=1)
            self.assertTrue(before_test < event_created_time < now)
            # 如果启用了时间跟踪，验证持续时间在合理范围内
            if timing_enabled:
                # 非常宽松的边界，测得在 devgpu 上为 0.036 毫秒
                self.assertTrue(0 < last["duration_ms"] < 100)
            else:
                # 如果未启用时间跟踪，确保没有 duration_ms 字段
                self.assertTrue("duration_ms" not in last)
        else:
            # 如果不包含集合操作，确保条目中没有 "entries" 字段
            self.assertTrue("entries" not in t)
    # 要求NCCL库可用，装饰器标记
    @requires_nccl()
    # 如果不是测试多GPU，则跳过但在沙堡环境中通过，提示NCCL测试需要至少2个GPU
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    # 定义测试函数：管道转储测试
    def test_dump_pipe(self):
        # 定义带超时的打开文件函数
        def open_file_with_timeout(file_path, mode, timeout=1.0):
            start_time = time.time()
            while time.time() - start_time < timeout:
                if os.path.exists(file_path):
                    return open(file_path, mode)
                time.sleep(0.1)
            raise FileNotFoundError

        # 如果当前进程是主进程
        if self.rank == self.MAIN_PROCESS_RANK:
            # 对于每个子进程管道，确保接收到"next"消息
            for c in self.children_pipes:
                self.assertEqual(c.recv(), "next")

            # 创建转储文件名及管道文件名
            dump_file = self._trace_name(rank=0)
            pipe_file = dump_file + ".pipe"
            # 使用带超时的打开文件函数，以写入方式打开管道文件
            with open_file_with_timeout(pipe_file, "w") as f:
                f.write("1\n")
            # 使用带超时的打开文件函数，以二进制读取方式打开转储文件
            with open_file_with_timeout(dump_file, "rb", timeout=10.0) as f:
                # 断言转储文件中包含字符串"all_reduce"
                self.assertTrue("all_reduce" in str(pickle.load(f)))

            # 对于每个子进程管道，发送"next"消息
            for c in self.children_pipes:
                c.send("next")
            return

        # 如果不是主进程，则创建NCCL进程组
        pg = self._create_process_group_nccl()
        device = self.local_device
        # 在本地设备上创建全为当前进程rank值的张量
        a = torch.full((3, 4), float(self.rank), device=device)
        # 执行两次全局归约操作
        for i in range(2):
            f = pg.allreduce(a)
        # 等待所有操作完成
        f.wait()
        # 同步CUDA设备
        torch.cuda.synchronize(device=device)
        # 向父进程发送"next"消息
        self.parent.send("next")
        # 接收来自父进程的消息
        self.parent.recv()
    # 测试长时间运行的情况
    def test_long(self):
        # 设置环境变量 TORCH_NCCL_TRACE_BUFFER_SIZE 为 10
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "10"
        # 如果当前进程是主进程，则直接返回
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        # 创建 NCCL 进程组
        pg = self._create_process_group_nccl()
        # 获取本地设备
        device = self.local_device
        # 创建一个大小为 (3, 4) 的张量，填充为当前进程的 rank 值，放在指定的设备上
        a = torch.full((3, 4), float(self.rank), device=device)
        # 执行两次循环
        for i in range(2):
            # 测试其他的原语确保它们的字符串是有效的
            xs = [torch.ones(3, 4, device=device)]
            # 广播张量 xs 到所有进程，并等待完成
            pg.broadcast(xs).wait()
            # 对所有进程的张量 xs 执行全局归约操作，并等待完成
            pg.allreduce(xs).wait()
            # 对所有进程的张量 xs 执行归约操作，并等待完成
            pg.reduce(xs).wait()
            # 创建一个空的张量列表 ys，用于存储从所有进程收集的数据
            ys = [[torch.empty(3, 4, device=device) for _ in range(self.world_size)]]
            # 执行全局聚集操作，将 xs 数据收集到 ys 中，并等待完成
            pg.allgather(ys, xs).wait()
            # 对 xs 执行归约分散操作，将结果存储在 ys 中，并等待完成
            pg.reduce_scatter(xs, ys).wait()
            # 对张量 a 执行全局归约操作，并等待完成
            f = pg.allreduce(a)
        # 等待 f 的操作完成
        f.wait()
        # 同步 CUDA 设备
        torch.cuda.synchronize(device=device)
        # 加载并反序列化 NCCL 的跟踪结果
        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
        t = t["entries"]
        # 断言跟踪条目数量为 10
        self.assertEqual(len(t), 10)
        # 获取第一个和最后一个条目
        first = t[0]
        last = t[-1]
        # 断言最后一个条目的名称为 "nccl:all_reduce"
        self.assertEqual(last["profiling_name"], "nccl:all_reduce")
        # 断言最后一个条目的状态为 "completed"
        self.assertEqual(last["state"], "completed")
        # 断言最后一个条目的堆栈跟踪中包含 "test_c10d_nccl.py"
        self.assertIn("test_c10d_nccl.py", str(last["frames"]))
        # 断言最后一个条目的输入大小为 (3, 4)
        self.assertEqual(last["input_sizes"], ((3, 4),))
        # 断言最后一个条目的输入数据类型为 ["Float"]
        self.assertEqual(last["input_dtypes"], ["Float"])
        # 断言最后一个条目的输出大小为 (3, 4)
        self.assertEqual(last["output_sizes"], ((3, 4),))
        # 断言最后一个条目的输出数据类型为 ["Float"]
        self.assertEqual(last["output_dtypes"], ["Float"])
        # 断言最后一个条目的超时时间为 600000 毫秒
        self.assertEqual(last["timeout_ms"], 600000)
        # 断言最后一个条目的集体序列 ID 与第一个条目的差为 9
        self.assertEqual(last["collective_seq_id"] - first["collective_seq_id"], 9)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    # 测试在所有工作都已完成时进行跟踪
    def test_trace_while_all_works_retired(self):
        # 设置环境变量 TORCH_NCCL_TRACE_BUFFER_SIZE 为 10
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "10"
        # 如果当前进程是主进程，则直接返回
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        # 创建 NCCL 进程组
        pg = self._create_process_group_nccl()
        # 获取本地设备
        device = self.local_device
        # 发送多于缓冲区大小的工作，以覆盖先前的条目
        for i in range(12):
            a = [torch.ones(3, 4, device=device)]
            # 广播张量 a 到所有进程，并等待完成
            pg.broadcast(a).wait()
        # 同步 CUDA 设备
        torch.cuda.synchronize(device=device)

        # 等待所有工作完成
        pg._wait_for_pending_works()
        # 加载并反序列化 NCCL 的跟踪结果
        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
        t = t["entries"]
        # 断言跟踪条目数量为 10
        self.assertEqual(len(t), 10)
        # 获取最后一个条目
        last = t[-1]
        # 断言最后一个条目的 retired 属性为 True
        self.assertEqual(last["retired"], True)
        # 断言最后一个条目的状态为 "completed"
        self.assertEqual(last["state"], "completed")

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("timing_enabled", [True, False])
    @parametrize("only_active", [True, False])
    # 定义测试函数，用于跟踪活跃状态下的操作
    def test_trace_while_active(self, timing_enabled, only_active):
        # 如果当前进程是主进程
        if self.rank == self.MAIN_PROCESS_RANK:
            # 等待所有子进程发送 "next" 消息
            for c in self.children_pipes:
                self.assertEqual(c.recv(), "next")
            # 向所有子进程发送 "next" 消息
            for c in self.children_pipes:
                c.send("next")
            return

        # 创建基于 NCCL 的进程组
        pg = self._create_process_group_nccl()
        # 如果启用了计时
        if timing_enabled:
            # 开启集体操作的时间统计
            pg._enable_collectives_timing()
        # 获取本地设备
        device = self.local_device
        # 设置当前 CUDA 设备
        with torch.cuda.device(device):
            # 创建一个填充值为当前进程排名的张量 a
            a = torch.full((3, 4), float(self.rank), device=device)

            # 执行全局归约操作并等待完成
            pg.allreduce(a).wait()
            # 创建 CUDA 事件 e 并记录
            e = torch.cuda.Event()
            e.record()
            # 如果当前进程排名不为 0
            if self.rank != 0:
                # 再次执行全局归约操作并等待完成
                pg.allreduce(a).wait()
            # 同步 CUDA 事件 e
            e.synchronize()
            # 反序列化 NCCL 跟踪信息
            t = pickle.loads(
                torch._C._distributed_c10d._dump_nccl_trace(onlyActive=only_active)
            )
            # 获取跟踪条目
            t = t["entries"]
            # 如果仅跟踪活跃操作
            if only_active:
                # 如果当前进程是排名为 0 的主进程
                if self.rank == 0:
                    # 断言跟踪条目长度为 0
                    self.assertEqual(len(t), 0)
                else:
                    # 断言跟踪条目长度为 1
                    self.assertEqual(len(t), 1)
            # 如果不仅跟踪活跃操作
            if not only_active:
                # 如果当前进程是排名为 0 的主进程
                if self.rank == 0:
                    # 断言最后一个跟踪条目的名称为 "nccl:all_reduce"
                    self.assertEqual(t[-1]["profiling_name"], "nccl:all_reduce")
                    # 断言最后一个跟踪条目的集合序列 ID 为 1
                    self.assertEqual(t[-1]["collective_seq_id"], 1)
                    # 断言最后一个跟踪条目的状态为 "completed"
                    self.assertEqual(t[-1]["state"], "completed")
                else:
                    # 断言最后一个跟踪条目的名称为 "nccl:all_reduce"
                    self.assertEqual(t[-1]["profiling_name"], "nccl:all_reduce")
                    # 断言最后一个跟踪条目的集合序列 ID 为 2
                    self.assertEqual(t[-1]["collective_seq_id"], 2)
                    # 断言最后一个跟踪条目的状态为根据计时启用情况决定的状态
                    self.assertEqual(
                        t[-1]["state"], self.started_or_scheduled(timing_enabled)
                    )

            # 向父进程发送 "next" 消息
            self.parent.send("next")
            # 断言从父进程接收到的消息为 "next"
            self.assertEqual("next", self.parent.recv())
            # 如果当前进程是排名为 0 的主进程
            if self.rank == 0:
                # 执行全局归约操作并等待完成
                pg.allreduce(a).wait()
            # 同步 CUDA 设备
            torch.cuda.synchronize(device=device)
    # 在测试函数中，验证当前进程是否为主进程
    def test_trace_while_stuck(self, timing_enabled):
        if self.rank == self.MAIN_PROCESS_RANK:
            # 如果是主进程，依次接收每个子进程管道发送的"next"消息
            for c in self.children_pipes:
                self.assertEqual(c.recv(), "next")
            # 向每个子进程管道发送"next"消息
            for c in self.children_pipes:
                c.send("next")
            # 函数结束
            return

        # 创建 NCCL 进程组对象
        pg = self._create_process_group_nccl()
        # 如果启用了时间跟踪，开启集合通信的时间统计
        if timing_enabled:
            pg._enable_collectives_timing()

        # 设置当前设备为本地 CUDA 设备
        device = self.local_device
        with torch.cuda.device(device):
            # 创建一个形状为 (3, 4) 的张量，填充值为当前进程的排名，存储在指定 CUDA 设备上
            a = torch.full((3, 4), float(self.rank), device=device)

            # 执行全局归约操作并等待完成
            pg.allreduce(a).wait()
            # 创建一个 CUDA 事件对象
            e = torch.cuda.Event()
            # 记录当前事件
            e.record()

            # 定义一个收集跟踪信息的函数
            def gather_trace():
                # 同步 CUDA 事件
                e.synchronize()
                # 让另一个线程有时间填充 CUDA 缓冲区
                time.sleep(5)
                # 从序列化的 NCCL 跟踪信息中加载数据
                t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
                # 获取跟踪信息的条目列表
                t = t["entries"]
                # 断言最后一个跟踪条目的性能名称为 "nccl:all_reduce"
                self.assertEqual(t[-1]["profiling_name"], "nccl:all_reduce")
                if self.rank == 0:
                    # 如果当前进程的排名为 0，断言最后一个条目的集合序列 ID 为 1
                    self.assertEqual(t[-1]["collective_seq_id"], 1)
                    # 断言最后一个条目的状态为 "completed"
                    self.assertEqual(t[-1]["state"], "completed")
                else:
                    # 如果当前进程的排名不为 0，断言最后一个条目的集合序列 ID 为 2
                    self.assertEqual(t[-1]["collective_seq_id"], 2)
                    # 断言最后一个条目的状态与启动或计划状态匹配（根据是否启用时间跟踪）
                    self.assertEqual(
                        t[-1]["state"], self.started_or_scheduled(timing_enabled)
                    )
                    # 断言最后一个条目的发现完成时间为 None
                    self.assertIsNone(t[-1]["time_discovered_completed_ns"])
                # 最终向父进程发送"next"消息，使缺少的排名为 0 的进程继续，解除非零排名的阻塞状态
                self.parent.send("next")

            # 如果当前进程的排名不为 0
            if self.rank != 0:
                # 执行全局归约操作并等待完成
                pg.allreduce(a).wait()
                # 创建一个线程，目标函数为 gather_trace
                th = threading.Thread(target=gather_trace)
                # 启动线程执行
                th.start()
                # 填充 CUDA 缓冲区，大约 1024 个事件后将会阻塞
                for i in range(2000):
                    a = a + a
                # 等待线程执行完成
                th.join()
            else:
                # 如果当前进程的排名为 0，直接执行 gather_trace 函数
                gather_trace()

            # 断言从父进程接收到的消息为"next"
            self.assertEqual("next", self.parent.recv())
            # 如果当前进程的排名为 0，再次执行全局归约操作并等待完成
            if self.rank == 0:
                pg.allreduce(a).wait()
            # 同步当前 CUDA 设备
            torch.cuda.synchronize(device=device)
    # 测试单个发送和接收操作的方法
    def test_individual_send_recv(self, op_sizes, timing_enabled):
        """
        'WorkEnqueue' was skipped for isendirecv, leading to segfault on dump_entries when update_state tried to use
        a destructed Work obj's cuda events
        """

        # 如果当前进程是主进程，则直接返回，不执行后续代码
        if self.rank == self.MAIN_PROCESS_RANK:
            return

        # 创建一个 NCCL 进程组对象
        pg = self._create_process_group_nccl()

        # 如果启用了时间统计，则开启集合操作的时间统计
        if timing_enabled:
            pg._enable_collectives_timing()

        # 定义重复次数
        num_repeats = 10
        # 获取操作大小列表的长度，即操作次数
        ops_per_repeat = len(op_sizes)

        # 执行多次重复操作
        for i in range(num_repeats):
            # 遍历操作大小列表
            for input_sizes in op_sizes:
                # 创建一个在本地设备上的全零张量
                tensor = torch.zeros(input_sizes).to(self.local_device)

                # 根据当前进程的不同情况进行数据接收或发送
                if self.rank == 0:
                    # 接收数据
                    dist.recv(tensor, 1)
                elif self.rank == 1:
                    # 发送数据，并对本地数据乘以2
                    tensor *= 2
                    dist.send(tensor, 0)

        # 同步本地设备的 CUDA 线程
        torch.cuda.synchronize(device=self.local_device)

        # 如果启用了时间统计
        if timing_enabled:
            # 等待守护线程处理工作队列
            time.sleep(1)

        # 从序列化对象中加载 NCCL 跟踪数据
        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())

        # 断言跟踪数据的长度符合预期
        self.assertEqual(len(t["entries"]), num_repeats * (ops_per_repeat))

        # 初始化预期的序列和操作 ID
        expected_seq = 1
        expected_op_id = 1

        # 遍历所有跟踪条目
        for seq in range(num_repeats * ops_per_repeat):
            # 获取当前操作大小
            input_sizes = op_sizes[seq % ops_per_repeat]

            # 根据当前进程的角色确定性能分析的名称
            profiling_name = "nccl:recv 0<-1" if self.rank == 0 else "nccl:send 1->0"

            # 断言跟踪条目的各个属性符合预期值
            self.assertEqual(t["entries"][seq]["profiling_name"], profiling_name)
            self.assertEqual(t["entries"][seq]["p2p_seq_id"], expected_seq)
            expected_seq += 1
            self.assertEqual(t["entries"][seq]["op_id"], expected_op_id)
            expected_op_id += 1
            self.assertEqual(t["entries"][seq]["input_sizes"], [input_sizes])
            self.assertEqual(t["entries"][seq]["output_sizes"], [input_sizes])
            self.assertEqual(t["entries"][seq]["state"], "completed")

            # 如果启用了时间统计，则断言持续时间在合理范围内
            if timing_enabled:
                duration = t["entries"][seq]["duration_ms"]
                self.assertTrue(0.001 < duration < 10000, duration)
            else:
                # 如果未启用时间统计，则确保跟踪条目中不包含持续时间字段
                self.assertTrue("duration_ms" not in t["entries"][seq])

    # TODO(whc) support and test coalesced collectives that use the c++ start/end group thingy instead of python
    # coalescing manager

    # TODO(whc) test out other ops (And combinations of ops, if that's valid?)
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @parametrize("timing_enabled", [True, False])
    # 定义测试方法，用于测试协同管理器在集合操作中的表现
    def test_coalescing_manager_collective(self, timing_enabled):
        """
        The coalescing manager api works by accumulating operations in python via a contextmanager, and then making
        one call into c++ to an <op>_coalesced API.  It has limited support for ops and has been added recently to
        avoid overheads of making individual py-cpp calls.  This complicates flight recording..

        For now, flight recording of coalescing_manager collectives is less detailed than cpp coalesced collectives.
        """
        # 如果当前进程是主进程，则直接返回，不执行后续操作
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        # 创建基于 NCCL 的进程组对象
        pg = self._create_process_group_nccl()
        # 如果启用了时间统计，则开启集合操作的时间统计
        if timing_enabled:
            pg._enable_collectives_timing()

        # 创建一个全零的输出张量，将其移动到当前进程所在的设备上
        output_tensors = torch.zeros(2, 2).to(self.rank)
        # 创建多个全一的输入张量列表，每个张量移动到当前进程所在的设备上
        input_tensors = [torch.ones(2, 2).to(self.rank) for _ in range(self.world_size)]

        # TODO(whc) make this work with bigger world or something
        # 断言当前进程的总数是2，如果不是则会触发断言错误
        self.assertEqual(self.world_size, 2, self.world_size)

        # 使用分布式通信的协同管理器进行上下文管理
        with dist._coalescing_manager():
            # 对每个进程执行分布式归约-分散操作，将结果存储在output_tensors中
            for i in range(self.world_size):
                dist.reduce_scatter_tensor(output_tensors[i], input_tensors[i])
        # 断言输出张量等于当前进程对应的输入张量乘以世界大小
        self.assertEqual(output_tensors, input_tensors[self.rank] * self.world_size)

        # 在当前设备上同步所有流中的操作
        torch.cuda.synchronize(device=self.rank)

        # 如果启用了时间统计
        if timing_enabled:
            # 等待看门狗线程处理工作队列
            time.sleep(1)

        # 反序列化 NCCL 跟踪信息，返回一个跟踪对象t
        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())

        # 断言跟踪条目的数量为1，表示执行了reduce_scatter_tensor_coalesced和endCoalescing两个操作
        self.assertEqual(
            len(t["entries"]), 1
        )  # one for the reduce_scatter_tensor_coalesced, one for the endCoalescing
        # 断言跟踪条目的性能名称为"nccl:reduce_scatter_tensor_coalesced"
        self.assertEqual(
            t["entries"][0]["profiling_name"], "nccl:reduce_scatter_tensor_coalesced"
        )
        # 断言跟踪条目的集合序列 ID 为1
        self.assertEqual(t["entries"][0]["collective_seq_id"], 1)
        # 断言跟踪条目的输入大小为[[2, 2], [2, 2]]
        self.assertEqual(t["entries"][0]["input_sizes"], [[2, 2], [2, 2]])
        # 断言跟踪条目的输出大小为[[2], [2]]
        self.assertEqual(
            t["entries"][0]["output_sizes"],
            [
                [
                    2,
                ],
                [
                    2,
                ],
            ],
        )
        # 断言跟踪条目的状态为"completed"
        self.assertEqual(t["entries"][0]["state"], "completed")
        # 如果启用了时间统计，断言跟踪条目的持续时间在0.001到10000之间
        if timing_enabled:
            duration = t["entries"][0]["duration_ms"]
            self.assertTrue(0.001 < duration < 10000, duration)
        else:
            # 如果未启用时间统计，则确保跟踪条目中没有"duration_ms"字段
            self.assertTrue("duration_ms" not in t["entries"][0])
def check_if_test_is_skipped(fn):
    # 定义装饰器函数，用于检查测试是否应跳过
    def wrapper(self, *args, **kwargs):
        # 遍历所有测试跳过值
        for skip in TEST_SKIPS.values():
            # 如果第一个进程的退出码与某个跳过值相同，则调用父类的检查返回码方法
            if self.processes[0].exitcode == skip.exit_code:
                return MultiProcessTestCase._check_return_codes(self, *args, **kwargs)
        # 否则，继续执行原始函数
        return fn(self, *args, **kwargs)

    return wrapper


class NCCLTraceTestDumpOnTimeoutBase(NCCLTraceTestBase):
    timeout_sec = 1

    def _create_process_group_nccl(self):
        # 创建文件存储对象，用于进程组
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化 NCCL 进程组
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=NCCLTraceTestDumpOnTimeoutBase.timeout_sec),
        )
        # 获取默认的进程组
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    @check_if_test_is_skipped
    def _check_return_codes(self, elapsed_time):
        # 基础测试基础设施假设进程以匹配的返回码退出，
        # 但在这个测试中，我们希望 rank0 中止，rank1 干净退出
        self.assertEqual(self.processes[0].exitcode, -6)
        self.assertEqual(self.processes[1].exitcode, 0)

    def _wait_process(self, rank, timeout):
        try:
            # 等待进程退出，并返回其退出码
            self.processes[rank].join(timeout)
            return self.processes[rank].exitcode
        except TimeoutError:
            return None


class NCCLTraceTestDumpOnTimeout(NCCLTraceTestDumpOnTimeoutBase):
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @parametrize("timing_enabled", [True, False])
        # 定义测试函数，用于测试超时情况下的数据转储，传入是否启用计时的标志
        def test_timeout_dumps(self, timing_enabled):
            # 设置环境变量以设定 Torch NCCL 协调检查毫秒数为 1000
            os.environ["TORCH_NCCL_COORD_CHECK_MILSEC"] = "1000"
            # 设置环境变量以设定 Torch NCCL 心跳超时秒数为 1
            os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1"

            # 如果当前进程的 rank 为主进程的 rank
            if self.rank == self.MAIN_PROCESS_RANK:
                # 等待 rank0 进程崩溃，超时时间为 90 秒，预期返回 -6
                self.assertEqual(self._wait_process(0, timeout=90), -6)
                # 打开 rank0 的跟踪文件，并以二进制模式读取
                with open(self._trace_name(rank=0), "rb") as f:
                    # 使用 pickle 加载数据
                    t = pickle.load(f)
                    # 从加载的数据中获取 "entries" 键对应的值
                    t = t["entries"]
                    # 断言获取的条目数为 2
                    self.assertEqual(len(t), 2)
                    # 断言第一个条目的 collective_seq_id 为 1
                    self.assertEqual(t[0]["collective_seq_id"], 1)
                    # 断言第一个条目的 state 为 "completed"
                    self.assertEqual(t[0]["state"], "completed")
                    # 断言第二个条目的 collective_seq_id 为 2
                    self.assertEqual(t[1]["collective_seq_id"], 2)
                    # 断言第二个条目的 state 为根据 timing_enabled 返回的结果
                    self.assertEqual(
                        t[1]["state"], self.started_or_scheduled(timing_enabled)
                    )

                # 断言当前目录下不存在 rank 为 1 的跟踪文件
                self.assertFalse(os.path.exists(self._trace_name(rank=1)))

                # 返回函数
                return

            # 创建 NCCL 进程组
            pg = self._create_process_group_nccl()
            # 如果启用计时
            if timing_enabled:
                # 强制在设置中禁用计时，因为没有 'disable' 函数
                pg._enable_collectives_timing()

            # 获取本地设备
            device = self.local_device
            # 在指定 CUDA 设备上进行操作
            with torch.cuda.device(device):
                # 创建一个形状为 (3, 4) 的张量，其值为当前进程的 rank
                a = torch.full((3, 4), float(self.rank), device=device)

                # 执行所有进程组的全局归约操作并等待完成
                pg.allreduce(a).wait()
                # 如果当前进程的 rank 为 0
                if self.rank == 0:
                    # 再次执行全局归约操作并等待完成
                    pg.allreduce(a).wait()

                # 在指定的 CUDA 设备上同步
                # rank 0 在同步之前将崩溃，但 rank 1 将迅速且干净地退出
                torch.cuda.synchronize(device=device)
# 实例化具有参数的测试用例，使用 ProcessGroupNCCLGroupTest 进行实例化
instantiate_parametrized_tests(ProcessGroupNCCLGroupTest)

# 实例化具有参数的测试用例，使用 NCCLTraceTestDumpOnTimeout 进行实例化
instantiate_parametrized_tests(NCCLTraceTestDumpOnTimeout)

# 实例化具有参数的测试用例，使用 NCCLTraceTest 进行实例化
instantiate_parametrized_tests(NCCLTraceTest)

# 定义 NCCLTraceTestTimeoutDumpOnStuckRanks 类，继承自 NCCLTraceTestDumpOnTimeoutBase 类
class NCCLTraceTestTimeoutDumpOnStuckRanks(NCCLTraceTestDumpOnTimeoutBase):

    # 装饰器：检查是否跳过测试
    @check_if_test_is_skipped
    def _check_return_codes(self, elapsed_time):
        # 基础测试基础设施假设进程以匹配的返回码退出，
        # 但在这个测试中，我们希望 rank0 中止，rank1 干净退出
        self.assertEqual(self.processes[0].exitcode, -6)
        self.assertEqual(self.processes[1].exitcode, -6)

    # 装饰器：要求存在 NCCL 库
    @requires_nccl()
    # 装饰器：如果 GPU 数量小于 2 则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_timeout_dumps_on_stuck_ranks(self):
        # 设置环境变量，使得 rank0 在检测到超时后更快崩溃
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1"
        # 恢复此环境变量为其先前的默认值，以防其他测试更改了它
        os.environ["TORCH_NCCL_COORD_CHECK_MILSEC"] = "1000"

        if self.rank == self.MAIN_PROCESS_RANK:
            # 等待 rank0 和 rank1 在超时前崩溃，
            # 我们依赖于 rank1 睡眠足够长时间以转储调试信息。
            self.assertEqual(self._wait_process(0, timeout=90), -6)
            self.assertEqual(self._wait_process(1, timeout=90), -6)
            # 确保文件存在以及是否在 rank0 和 rank1 上
            self.assertTrue(os.path.exists(self._trace_name(rank=1)))
            self.assertTrue(os.path.exists(self._trace_name(rank=0)))
            with open(self._trace_name(rank=0), "rb") as f:
                t = pickle.load(f)
                t = t["entries"]
                self.assertEqual(len(t), 2)
            with open(self._trace_name(rank=1), "rb") as f:
                t = pickle.load(f)
                t = t["entries"]
                self.assertEqual(len(t), 1)
                self.assertEqual(t[0]["collective_seq_id"], 1)
                self.assertEqual(t[0]["state"], "completed")
            return

        # 创建 NCCL 进程组
        pg = self._create_process_group_nccl()
        device = self.local_device
        with torch.cuda.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            # 执行全局归约操作
            pg.allreduce(a).wait()
            if self.rank == 0:
                pg.allreduce(a).wait()

            # rank 0 会被卡住，超时然后向所有 rank 信号超时
            torch.cuda.synchronize(device=device)

            if self.rank == 1:
                # 强制 rank 1 空闲，以便在全局信号后最终超时，并转储调试信息。
                time.sleep(600)


class NcclErrorDumpTest(NCCLTraceTestBase):
    def _wait_process(self, rank, timeout):
        try:
            self.processes[rank].join(timeout)
            return self.processes[rank].exitcode
        except TimeoutError:
            return None

    # 装饰器：检查是否跳过测试
    @check_if_test_is_skipped
   `
    # 检查返回代码的函数，比较进程的退出码，确保符合预期
    def _check_return_codes(self, elapsed_time):
        # 断言进程 0 的退出码应为 -6，基准测试框架假设进程退出码匹配
        self.assertEqual(self.processes[0].exitcode, -6)
        # 断言进程 1 的退出码应为 1
        self.assertEqual(self.processes[1].exitcode, 1)

    # 装饰器，要求使用 NCCL 功能
    @requires_nccl()
    # 装饰器，要求 NCCL 版本至少为 2.4.0，并提供错误检查功能
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    # 装饰器，跳过 GPU 数量小于 2 的测试
    @skip_if_lt_x_gpu(2)
    # 装饰器，跳过 ROCm 环境的测试
    @skip_if_rocm
    def test_nccl_errors_dump(self):
        # 设置环境变量以启用 NCCL 异步错误处理
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # 设置 NCCL 跟踪缓冲区大小为 1000
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1000"
        # 设置超时时，NCCL 在超时后转储
        os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
        # 设置心跳超时时间为 5 秒
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "5"

        # 如果当前进程是主进程
        if self.rank == self.MAIN_PROCESS_RANK:
            # 等待 rank0 和 rank1 崩溃，然后查找转储文件
            self.assertEqual(self._wait_process(0, timeout=90), -6)
            self.assertEqual(self._wait_process(1, timeout=90), 1)
            # 验证 rank0 的跟踪文件是否存在
            self.assertTrue(os.path.exists(self._trace_name(rank=0)))
            return

        # 创建文件存储对象
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建 NCCL 进程组
        process_group = c10d.ProcessGroupNCCL(
            store,
            self.rank,
            self.world_size,
            timeout=timedelta(seconds=10),
        )
        # 执行 allreduce 操作，将随机数据进行求和，使用当前 rank 的 GPU
        process_group.allreduce(torch.rand(10).cuda(self.rank))
        # 如果当前进程是 rank 0
        if self.rank == 0:
            # 执行 allreduce 操作，期望在执行中引发错误
            work = process_group.allreduce(torch.rand(10).cuda(self.rank))
            # 期望在下面的代码块中引发 DistBackendError 错误
            with self.assertRaisesRegex(dist.DistBackendError, ""):
                # 等待 NCCL 操作完成，阻塞当前流
                work.wait()
                # 在 GPU 上执行一些操作
                a = torch.rand(10).cuda(self.rank)
        # 如果当前进程是 rank 1
        elif self.rank == 1:
            # 清理结构（如 FileStore 的文件等），在退出前
            del process_group
            # 进程 1 退出，退出码为 1
            sys.exit(1)
# 如果脚本作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 断言，检查当前是否没有初始化 CUDA 上下文，否则抛出异常信息
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    # 调用运行测试的函数
    run_tests()
```