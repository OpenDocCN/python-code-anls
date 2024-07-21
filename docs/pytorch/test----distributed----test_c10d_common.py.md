# `.\pytorch\test\distributed\test_c10d_common.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import copy                 # 导入copy模块，用于对象的深拷贝操作
import os                   # 导入os模块，提供了访问操作系统功能的接口
import pickle               # 导入pickle模块，用于序列化和反序列化Python对象
import sys                  # 导入sys模块，提供了访问与Python解释器相关的变量和函数
import tempfile             # 导入tempfile模块，用于创建临时文件和目录
import threading            # 导入threading模块，提供了线程相关的操作函数
import time                 # 导入time模块，提供了处理时间的各种函数
from contextlib import nullcontext  # 导入nullcontext函数，用于创建一个什么都不做的上下文管理器
from dataclasses import dataclass  # 导入dataclass装饰器，用于定义数据类
from datetime import timedelta  # 导入timedelta类，用于表示时间间隔
from itertools import product  # 导入product函数，用于迭代器的笛卡尔积操作
from sys import platform     # 导入platform变量，获取当前运行Python的平台信息
from typing import Callable, Dict, Optional  # 导入类型提示，用于静态类型检查

import torch                # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式包

if not dist.is_available():  # 如果分布式包不可用
    print("distributed package not available, skipping tests", file=sys.stderr)  # 输出错误信息到标准错误流
    sys.exit(0)             # 退出程序

import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD  # 导入powerSGD通信钩子模块
import torch.distributed.distributed_c10d as c10d  # 导入分布式c10d模块
import torch.nn.functional as F  # 导入PyTorch的函数式接口模块
import torch.testing._internal.common_utils as common  # 导入测试常用工具模块
from torch import nn         # 从torch中导入nn模块
from torch.distributed._spmd.comm_tensor import _wait_comm, CommTensor  # 导入分布式通信相关模块
from torch.fx.experimental.proxy_tensor import make_fx  # 导入代理张量的实验性FX模块
from torch.nn.parallel import DistributedDataParallel  # 导入分布式数据并行模块
from torch.testing._internal.common_distributed import (  # 导入内部分布式测试常用工具
    MultiProcessTestCase, skip_if_lt_x_gpu
)
from torch.testing._internal.common_utils import (  # 导入测试常用工具
    instantiate_parametrized_tests, load_tests, parametrize,
    retry_on_connect_failures, run_tests, TEST_WITH_DEV_DBG_ASAN, TestCase
)
from torch.utils.checkpoint import checkpoint  # 导入检查点模块，用于模型中间状态的存储和恢复

if TEST_WITH_DEV_DBG_ASAN:  # 如果处于开发/调试ASAN模式
    print("Multiprocessing spawn is not compatible with dev/dbg asan", file=sys.stderr)  # 输出警告信息到标准错误流
    sys.exit(0)             # 退出程序

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests     # 将load_tests函数赋值给自身，用于在sandcastle上进行测试分片时过滤测试用例

if platform == "darwin":    # 如果运行平台是macOS
    LOOPBACK = "lo0"        # 设置环回接口为lo0
else:
    LOOPBACK = "lo"          # 否则设置环回接口为lo

torch.backends.cuda.matmul.allow_tf32 = False  # 禁用CUDA矩阵乘法的TF32支持


def gpus_for_rank(world_size):
    """Multigpu tests are designed to simulate the multi nodes with multi
    GPUs on each node. Nccl backend requires equal #GPUs in each process.
    On a single node, all visible GPUs are evenly
    divided to subsets, each process only uses a subset.
    """
    visible_devices = list(range(torch.cuda.device_count()))  # 获取所有可见的CUDA设备列表
    gpus_per_process = torch.cuda.device_count() // world_size  # 计算每个进程需要使用的GPU数目
    gpus_for_rank = []
    for rank in range(world_size):
        # 将可见GPU按照进程数平均划分成子集，每个进程只使用其中的一个子集
        gpus_for_rank.append(
            visible_devices[rank * gpus_per_process : (rank + 1) * gpus_per_process]
        )
    return gpus_for_rank  # 返回每个进程应该使用的GPU列表


class AbstractTimeoutTest:
    # 定义一个测试函数，用于测试存储超时情况，接受后端类型、初始化方法和用于通信的列表参数
    def _test_store_timeout(self, backend, init_method, c2p):
        try:
            # 初始化进程组，使用指定的后端、初始化方法、单个进程的全局大小、当前进程的排名和超时时间为1秒
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=1,
                rank=0,
                timeout=timedelta(seconds=1),
            )
            # 获取默认存储
            default_store = c10d._get_default_store()
            # 记录开始时间
            tik = time.time()
            # 使用断言来捕获期望的 RuntimeError 异常，并且异常信息包含"Timeout"
            with self.assertRaisesRegex(RuntimeError, "Timeout"):
                default_store.get("nonexistent key")
            # 记录结束时间
            tok = time.time()
            # 销毁进程组
            dist.destroy_process_group()
            # 将超时时长添加到通信列表 c2p 中
            c2p.append(float(tok - tik))
        except RuntimeError as e:
            # 捕获 "Address already in use" 错误并将其报告给主线程
            c2p.append(e)

    # 初始化方法生成器，根据不同平台生成不同的初始化方法
    def _init_methods(self):
        # 创建临时文件
        f = tempfile.NamedTemporaryFile(delete=False)
        if sys.platform == "win32":
            # Windows 平台下返回文件路径的 URL 形式，并关闭临时文件
            yield "file:///{}".format(f.name.replace("\\", "/"))
            f.close()
        else:
            # 非 Windows 平台返回文件路径的 URL 形式，并关闭临时文件
            yield f"file://{f.name}"
            f.close()
            # 返回本地主机上可用的随机端口的 TCP URL
            yield "tcp://127.0.0.1:%d" % common.find_free_port()

    # 测试默认存储超时的方法，接受后端类型参数
    def _test_default_store_timeout(self, backend):
        # 遍历初始化方法生成器返回的所有初始化方法
        for init_method in self._init_methods():
            # 创建用于通信的空列表 c2p
            c2p = []
            # 创建线程，目标为 _test_store_timeout 方法，传入后端类型、初始化方法和通信列表 c2p
            t = threading.Thread(
                target=self._test_store_timeout, args=(backend, init_method, c2p)
            )
            # 将线程设置为守护线程并启动
            t.daemon = True
            t.start()
            # 等待线程执行完成或超时（5秒）
            t.join(5)

            # 断言通信列表 c2p 的长度为1
            self.assertEqual(1, len(c2p))
            # 如果 c2p 中的第一个元素是 float 类型
            if isinstance(c2p[0], float):
                # 等待时间应该在1秒内，使用3秒来排除误报
                self.assertGreater(3, c2p[0])
            # 如果 c2p 中的第一个元素是 RuntimeError 类型
            elif isinstance(c2p[0], RuntimeError):
                # 将 RuntimeError 抛出，由 @retry_on_connect_failures 处理错误
                raise c2p[0]
            else:
                # 抛出意外的异常类型错误
                raise RuntimeError(f"Unexpected type {type(c2p[0])}")
class TimeoutTest(TestCase):
    # 定义一个测试类 TimeoutTest，继承自 TestCase

    @retry_on_connect_failures
    # 应用装饰器 retry_on_connect_failures

    def test_store_based_barrier(self):
        # 定义测试方法 test_store_based_barrier

        f = tempfile.NamedTemporaryFile(delete=False)
        # 创建一个临时文件对象 f

        port = common.find_free_port()
        # 调用 common 模块的 find_free_port 函数，获取一个空闲端口号 port

        def thread_work(timeout, init_type, world_size, rank, error_list):
            # 定义内部函数 thread_work，接受超时时间 timeout、初始化类型 init_type、世界大小 world_size、进程排名 rank 和错误列表 error_list 作为参数

            if init_type == "file":
                # 如果初始化类型为 "file"
                barrier_store = dist.FileStore(f.name)
                # 创建一个文件存储对象 barrier_store，使用临时文件 f.name 作为文件名

            elif init_type == "tcp":
                # 如果初始化类型为 "tcp"
                barrier_store = dist.TCPStore(
                    "localhost",
                    port,
                    world_size,
                    is_master=rank == 0,
                    wait_for_workers=False,
                )
                # 创建一个 TCP 存储对象 barrier_store，使用本地主机 "localhost" 和端口号 port

            elif init_type == "hash":
                # 如果初始化类型为 "hash"
                barrier_store = dist.HashStore()
                # 创建一个哈希存储对象 barrier_store

            try:
                # 尝试执行以下代码块
                if rank != world_size - 1:
                    # 如果当前进程排名不是 world_size - 1

                    c10d._store_based_barrier(
                        rank=rank,
                        store=barrier_store,
                        group_name="_",
                        rendezvous_count=world_size,
                        timeout=timeout,
                        logging_interval=timeout / 2,
                    )
                    # 调用 c10d 模块的 _store_based_barrier 函数，使用当前进程排名、存储对象 barrier_store、分组名称 "_"、会合计数 world_size、超时时间 timeout 和日志间隔时间 timeout / 2

            except torch.distributed.DistStoreError as e:
                # 捕获 torch.distributed.DistStoreError 异常，并将异常对象赋值给 e
                self.assertTrue(isinstance(e, torch.distributed.DistError))
                # 断言 e 是 torch.distributed.DistError 类型的对象
                error_list.append(e)
                # 将异常 e 添加到错误列表 error_list 中

        world_size = 4
        # 定义世界大小为 4
        error_list = []
        # 创建一个空的错误列表 error_list
        threads = []
        # 创建一个空的线程列表 threads

        for init_type in ["file", "tcp", "hash"]:
            # 遍历初始化类型列表 ["file", "tcp", "hash"]

            for rank in range(world_size):
                # 遍历世界大小范围内的排名 rank

                t = threading.Thread(
                    target=thread_work,
                    args=(
                        timedelta(seconds=3),
                        init_type,
                        world_size,
                        rank,
                        error_list,
                    ),
                )
                # 创建一个线程对象 t，目标函数是 thread_work，参数包括超时时间 3 秒、初始化类型、世界大小、排名和错误列表

                threads.append(t)
                # 将线程对象 t 添加到线程列表 threads 中
                t.start()
                # 启动线程 t

            for i, thread in enumerate(threads):
                # 遍历线程列表 threads 的索引和线程

                thread.join()
                # 等待线程完成

            # 断言错误列表中的异常数等于世界大小减 1
            self.assertEqual(len(error_list), world_size - 1)
            
            # 遍历错误列表中的每个异常
            for error in error_list:
                # 断言错误消息中包含指定的字符串
                self.assertTrue(
                    "Timed out initializing process group in store based barrier"
                    in error.args[0]
                )
            error_list = []
            # 清空错误列表
            threads = []
            # 清空线程列表
    `
    # 初始化函数，接受 GPUs 列表并设置模型的各层
    def __init__(self, gpus):
        # 调用父类的初始化方法
        super().__init__()
        # 定义第一个全连接层，输入维度为2，输出维度为10，无偏置，移到指定的第一个 GPU 上
        self.fc1 = nn.Linear(2, 10, bias=False).to(gpus[0])
        # 定义第二个全连接层，输入维度为10，输出维度为50，无偏置，移到指定的第二个 GPU 上
        self.fc2 = nn.Linear(10, 50, bias=False).to(gpus[1])
        # 定义第三个全连接层，输入维度为50，输出维度为4，无偏置，移到指定的第二个 GPU 上
        self.fc3 = nn.Linear(50, 4, bias=False).to(gpus[1])
        # 定义 ReLU 激活函数
        self.relu = nn.ReLU()
        # 定义一个不需要梯度的参数，初始值为 [2, 2]，移到指定的第一个 GPU 上
        self.no_grad_param = nn.Parameter(
            torch.tensor([2, 2]).long(), requires_grad=False
        ).to(gpus[0])
    
    # 前向传播函数，接受输入 x 并返回模型的输出
    def forward(self, x):
        # 获取每个全连接层的设备信息
        dev0 = self.fc1.weight.device
        dev1 = self.fc2.weight.device
        # 将输入 x 移到第一个全连接层的设备上，通过第一个全连接层并应用 ReLU 激活函数
        x = self.relu(self.fc1(x.to(dev0)))
        # 将结果 x 移到第二个全连接层的设备上，通过第二个全连接层并应用 ReLU 激活函数
        x = self.relu(self.fc2(x.to(dev1)))
        # 通过第三个全连接层得到最终输出 x
        x = self.fc3(x)
        # 对输出进行 softmax 操作，并将结果移回第一个全连接层的设备上
        return F.softmax(x, dim=1).to(dev0)
class QuadraGpuNet(nn.Module):
    # 定义一个四GPU网络模型
    def __init__(self, gpus):
        super().__init__()
        # 第一层全连接层，输入维度2，输出维度10，无偏置，放置在gpus[0]设备上
        self.fc1 = nn.Linear(2, 10, bias=False).to(gpus[0])
        # 第二层全连接层，输入维度10，输出维度50，无偏置，放置在gpus[1]设备上
        self.fc2 = nn.Linear(10, 50, bias=False).to(gpus[1])
        # 第三层全连接层，输入维度50，输出维度4，无偏置，放置在gpus[2]设备上
        self.fc3 = nn.Linear(50, 4, bias=False).to(gpus[2])
        # 第四层全连接层，输入维度4，输出维度4，无偏置，放置在gpus[3]设备上
        self.fc4 = nn.Linear(4, 4, bias=False).to(gpus[3])
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 不可训练的参数，设为常量[2, 2]，放置在gpus[0]设备上
        self.no_grad_param = nn.Parameter(
            torch.tensor([2, 2]).long(), requires_grad=False
        ).to(gpus[0])

    def forward(self, x):
        # 获取各层权重所在的设备
        dev0 = self.fc1.weight.device
        dev1 = self.fc2.weight.device
        dev2 = self.fc3.weight.device
        dev3 = self.fc4.weight.device
        # 将输入数据x移动到fc1权重所在的设备，并经过ReLU激活
        x = self.relu(self.fc1(x.to(dev0)))
        # 将x移动到fc2权重所在的设备，并经过ReLU激活
        x = self.relu(self.fc2(x.to(dev1)))
        # 将x移动到fc3权重所在的设备，并经过ReLU激活
        x = self.relu(self.fc3(x.to(dev2)))
        # 将x移动到fc4权重所在的设备，最后的线性变换
        x = self.fc4(x.to(dev3))
        # 对输出进行softmax，并将结果移动到fc1权重所在的设备上
        return F.softmax(x, dim=1).to(dev0)


class ConvNet(nn.Module):
    # 定义卷积神经网络模型
    def __init__(self, gpus, layouts, dtypes):
        super().__init__()
        self.dtypes = dtypes
        # 如果gpus是列表则使用，否则复制gpus四次
        if isinstance(gpus, list):
            self.layer_gpus = gpus
        else:
            gpus = [gpus] * 4
        # 定义四个卷积层，每层的设备、内存格式和数据类型由layouts和dtypes指定
        self.conv0 = torch.nn.Conv2d(8, 16, (2, 2)).to(
            device=gpus[0], memory_format=layouts[0], dtype=dtypes[0]
        )
        self.conv1 = torch.nn.Conv2d(16, 32, (2, 2)).to(
            device=gpus[1], memory_format=layouts[1], dtype=dtypes[1]
        )
        self.conv2 = torch.nn.Conv2d(32, 16, (2, 2)).to(
            device=gpus[2], memory_format=layouts[2], dtype=dtypes[2]
        )
        self.conv3 = torch.nn.Conv2d(16, 8, (2, 2)).to(
            device=gpus[3], memory_format=layouts[3], dtype=dtypes[3]
        )

    def forward(self, x):
        # 将输入数据x转换为指定的第一个数据类型
        x = x.to(self.dtypes[0])
        # 选择使用layer_gpus列表或者默认设备x.device作为每层卷积的设备，并指定数据类型
        gpus = self.layer_gpus if hasattr(self, "layer_gpus") else [x.device] * 4
        x = self.conv0(x).to(device=gpus[1], dtype=self.dtypes[1])
        x = self.conv1(x).to(device=gpus[2], dtype=self.dtypes[2])
        x = self.conv2(x).to(device=gpus[3], dtype=self.dtypes[3])
        return self.conv3(x)


class Task(nn.Module):
    # 定义一个任务模块，包含一个参数p
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones(2, 2))

    def forward(self, x):
        # 返回参数p与输入x的和
        return self.p + x


class ModuleForDdpCommHook(nn.Module):
    # 用于分布式数据并行通信钩子的模块
    def __init__(self):
        super().__init__()
        self.t0 = Task()

    def forward(self, x, rank):
        # 调用Task模块处理输入x和rank
        return self.t0(x + rank)


class SparseGradientModule(nn.Module):
    # 稀疏梯度模块，包含一个稀疏的EmbeddingBag层
    def __init__(self):
        super().__init__()
        self.embedding = nn.EmbeddingBag(10, 10, sparse=True)

    def forward(self, x):
        # 对输入x进行softmax后返回
        return F.softmax(self.embedding(x), dim=1)


class CommonDistributedDataParallelTest:
    # 通用的分布式数据并行测试类，用于测试分布式训练
    # 执行在每个测试方法后的清理工作，确保删除测试文件以避免测试干扰
    def tearDown(self):
        # DistributedDataParallel 测试似乎没有调用 FileStore 析构函数
        # TODO: 调查这个测试，已知它存在问题
        # 使用这个 hack 来移除该测试生成的文件
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    # 返回当前测试环境的节点数（假设为固定值 2）
    def world_size(self):
        return 2

    def _prepare_single_device_module(
        self,
        process_group,
        devices,
        device_ids,
        global_batch_size,
        gradient_as_bucket_view=False,
    ):
        # 创建一个单设备的神经网络模型
        model = Net()
        # 确定使用的设备，如果没有指定设备则默认使用 cuda 设备
        device = devices[0] if devices else torch.device("cuda:%d" % self.rank)
        # 使用 DistributedDataParallel 将模型复制到指定设备，并分布式训练
        ddp_model = DistributedDataParallel(
            copy.deepcopy(model).to(device),
            device_ids=device_ids,
            process_group=process_group,
            bucket_cap_mb=0.001,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # 将原始模型也移到指定设备上
        model.to(device)

        # 创建输入数据和目标数据，移动到指定设备
        input = torch.randn(global_batch_size, 2).to(device)
        target = torch.randn(global_batch_size, 4).to(device)

        return model, ddp_model, input, target

    def _prepare_multi_device_module(
        self,
        process_group,
        devices,
        device_ids,
        global_batch_size,
        gradient_as_bucket_view=False,
    ):
        # 确保设备列表长度为 2 或 4，针对不同的设备数量选择不同的网络模型
        self.assertTrue(
            len(devices) == 2 or len(devices) == 4,
            f"unexpected devices for ddp tests {devices}",
        )
        if len(devices) == 2:
            model = DoubleGpuNet(devices)
        elif len(devices) == 4:
            model = QuadraGpuNet(devices)

        # 使用 DistributedDataParallel 将模型复制到多个设备，并分布式训练
        ddp_model = DistributedDataParallel(
            copy.deepcopy(model),
            device_ids=device_ids,
            process_group=process_group,
            bucket_cap_mb=0.001,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # 创建输入数据和目标数据，移动到第一个设备
        input = torch.randn(global_batch_size, 2).cuda(devices[0])
        target = torch.randn(global_batch_size, 4)

        return model, ddp_model, input, target

    def _get_store(self):
        # 返回一个分布式文件存储对象，使用测试类中指定的文件名和节点数
        return dist.FileStore(self.file_name, self.world_size)

    def _get_process_group(self):
        # 子类需要实现此方法，以提供正确的进程组对象
        raise NotImplementedError("To be implemented by child class")

    def _train_model(
        self, model, input_var, target, loss, run_checkpoint=False, use_reentrant=True
    ):
        # 设置模型为训练模式
        model.train()
        # 如果需要运行检查点，则使用检查点功能来计算输出
        if run_checkpoint:
            output = checkpoint(model, input_var, use_reentrant=use_reentrant)
        else:
            output = model(input_var)
        # 计算损失并进行反向传播
        l = loss(output, target)
        l.backward()

    def _test_ddp_checkpointing(
        self,
        input_model,
        process_group,
        use_bucket_view,
        find_unused_parameters=False,
        static_graph=False,
        run_checkpoint=False,
        use_reentrant=True,
        allow_none_grads=False,
        # 为了复现相同的训练结果，设置 CUDA 设备
        torch.cuda.set_device(self.rank)
        # 设置随机种子
        torch.manual_seed(31415)
        # 深度复制输入模型并将其移动到 CUDA 设备上
        model = copy.deepcopy(input_model).cuda()
        # 深度复制输入模型并将其移动到 CUDA 设备上，同时使用分布式数据并行
        ddp_model = copy.deepcopy(input_model).cuda()
        ddp_model = nn.parallel.DistributedDataParallel(
            ddp_model,
            bucket_cap_mb=1,  # 设置梯度分桶的容量限制为1MB
            gradient_as_bucket_view=use_bucket_view,  # 根据参数 use_bucket_view 决定是否以梯度视图的形式处理
            device_ids=[self.rank],  # 使用当前进程的设备 ID
            process_group=process_group,  # 设置进程组
            find_unused_parameters=find_unused_parameters,  # 是否查找未使用的参数
            static_graph=static_graph,  # 是否使用静态计算图
        )
        # 断言检查，验证是否正确记录了静态计算图的设置
        self.assertEqual(
            ddp_model._get_ddp_logging_data().get("static_graph", 0), static_graph
        )
        # 准备测试所需的输入数据
        input, ddp_input, target, ddp_target = self._prepare_dummy_data()
        # 定义损失函数为均方误差损失
        loss = nn.MSELoss()
        # 迭代次数
        n_iters = 5
        for i in range(n_iters):
            # 将模型参数梯度置零，不允许设置为 None
            model.zero_grad(set_to_none=False)
            ddp_model.zero_grad(set_to_none=False)
            # 使用单个模型进行训练
            self._train_model(
                model,
                input,
                target,
                loss,
                run_checkpoint=run_checkpoint,  # 是否运行检查点
                use_reentrant=use_reentrant,  # 是否使用可重入锁
            )
            # 使用分布式数据并行模型进行训练
            self._train_model(
                ddp_model,
                ddp_input,
                ddp_target,
                loss,
                run_checkpoint=run_checkpoint,
                use_reentrant=use_reentrant,
            )
            # 检查两个模型的参数梯度是否一致
            for i, j in zip(model.parameters(), ddp_model.parameters()):
                if not allow_none_grads:
                    # 如果不允许梯度为 None，则断言梯度不为 None
                    self.assertTrue(i.grad is not None)
                    self.assertTrue(j.grad is not None)
                # 使用相对和绝对误差容差检查两个模型的梯度是否相等
                self.assertEqual(i.grad, j.grad, rtol=1.3e-06, atol=5e-5)
    class CheckpointTwiceModule(CheckpointOnceModule):
        """
        Runs checkpoint for the same layer twice in a model. This simulates use
        cases such as pipeline parallel where the same layer can be checkpointed
        more than one time.
        """

        def __init__(self, use_reentrant=True):
            # 调用父类的初始化方法
            super().__init__(use_reentrant=use_reentrant)

        def forward(self, inp):
            # 使用第一层进行前向传播
            x = self.l1(inp)
            # 对第二层进行 checkpoint 操作
            x = checkpoint(self.l2, x, use_reentrant=self.use_reentrant)
            # 再次对第二层进行 checkpoint 操作
            x = checkpoint(self.l2, x, use_reentrant=self.use_reentrant)
            return x

    class CheckpointTwiceModuleWeightSharing(CheckpointTwiceModule):
        """
        Similar to CheckpointTwiceModule but the weights are shared.
        """

        def __init__(self, use_reentrant=True):
            # 调用父类的初始化方法
            super().__init__(use_reentrant=use_reentrant)
            # 共享权重
            self.l1.weight = self.l2.weight

        def forward(self, inp):
            # 使用第一层进行前向传播
            x = self.l1(inp)
            # 对第二层进行 checkpoint 操作
            x = checkpoint(self.l2, x, use_reentrant=self.use_reentrant)
            # 再次对第二层进行 checkpoint 操作
            x = checkpoint(self.l2, x, use_reentrant=self.use_reentrant)
            return x

    class DynamicCheckpointTwiceModule(CheckpointTwiceModule):
        def __init__(self, use_reentrant=True):
            # 调用父类的初始化方法
            super().__init__(use_reentrant=use_reentrant)
            # 初始化计数器
            self.count = 0

        def forward(self, inp):
            # 根据计数器选择不同的层进行 checkpoint 操作
            if self.count % 2:
                x = checkpoint(self.l1, inp, use_reentrant=self.use_reentrant)
            else:
                x = checkpoint(self.l2, inp, use_reentrant=self.use_reentrant)

            # 更新计数器
            self.count += 1
            return x

    class DynamicCheckpointTwiceModuleWeightSharing(DynamicCheckpointTwiceModule):
        def __init__(self, use_reentrant=True):
            # 调用父类的初始化方法
            super().__init__(use_reentrant=use_reentrant)
            # 共享权重
            self.l1.weight = self.l2.weight

    def _prepare_dummy_data(self):
        # 设置数据集大小
        ddp_bs = 16
        bs = ddp_bs * self.world_size
        # 生成随机输入和目标数据
        input = torch.rand((bs, 20), device="cuda", requires_grad=True)
        target = torch.randn((bs, 20), device="cuda")
        offset = self.rank * ddp_bs
        ddp_input = input[offset : offset + ddp_bs]
        ddp_target = target[offset : offset + ddp_bs]
        return input, ddp_input, target, ddp_target

    @skip_if_lt_x_gpu(2)
    @parametrize("use_reentrant", [True, False])
    def test_ddp_checkpointing_once(self, use_reentrant):
        """
        DDP works as expected when layer is checkpointed only once.
        """
        # 获取当前进程组
        process_group = self._get_process_group()
        # 遍历两种参数组合：use_bucket_view (True/False) 和 static_graph (True/False)
        for use_bucket_view, static_graph in product((False, True), (False, True)):
            # 调用 _test_ddp_checkpointing 方法，测试 DDP 在仅对层进行一次检查点时的行为
            self._test_ddp_checkpointing(
                self.CheckpointOnceModule(use_reentrant=use_reentrant),
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                static_graph=static_graph,
            )
            if static_graph:
                # 当 static_graph 为 True 时，find_unused_parameters 参数不起作用，因为对于静态图会被忽略
                # 调用 _test_ddp_checkpointing 方法，测试 DDP 在仅对层进行一次检查点时的行为
                self._test_ddp_checkpointing(
                    self.CheckpointOnceModule(),
                    process_group=process_group,
                    use_bucket_view=use_bucket_view,
                    static_graph=static_graph,
                    find_unused_parameters=True,
                )

    @skip_if_lt_x_gpu(2)
    @parametrize("use_reentrant", [True, False])
    def test_ddp_checkpointing_unused_params(self, use_reentrant):
        """
        With reentrant autograd checkpointing impl, DDP will fail when there are
        unused params in the model and no static graph training. With
        non-reentrant checkpointing implementation, this works as expected.
        """
        # 获取当前进程组
        process_group = self._get_process_group()
        # 遍历两种参数组合：use_bucket_view (True/False)
        for use_bucket_view in (True, False):
            # 如果不使用 reentrant，则使用 nullcontext 来禁用异常检查
            # 否则，使用 assertRaisesRegex 来捕获预期的 RuntimeError 异常
            err_ctx = (
                nullcontext()
                if not use_reentrant
                else self.assertRaisesRegex(
                    RuntimeError, "Expected to mark a variable ready only once."
                )
            )
            with err_ctx:
                # 调用 _test_ddp_checkpointing 方法，测试 DDP 在存在未使用的参数并且没有静态图训练时的行为
                model = self._test_ddp_checkpointing(
                    self.CheckpointOnceModule(use_reentrant=use_reentrant),
                    process_group=process_group,
                    use_bucket_view=use_bucket_view,
                    find_unused_parameters=True,
                )
            # 当 static_graph 为 True 时，测试通过
            model = self._test_ddp_checkpointing(
                self.CheckpointOnceModule(use_reentrant=use_reentrant),
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                find_unused_parameters=True,
                static_graph=True,
            )

    @skip_if_lt_x_gpu(2)
    @parametrize("use_reentrant", [True, False])
    @skip_if_lt_x_gpu(2)
    # 定义测试方法，用于验证动态模块在禁止重入的情况下可以多次进行检查点操作并共享权重
    def test_ddp_checkpointing_dynamic_weight_sharing(self):
        """
        Dynamic module can be checkpointed multiple times with weight sharing
        using non-reentrant checkpointing implementation.
        """
        # 获取当前进程组
        process_group = self._get_process_group()
        # 对于每种使用桶视图的情况（True 和 False）
        for use_bucket_view in (True, False):
            # 创建测试模型并调用 _test_ddp_checkpointing 方法进行验证
            model = self._test_ddp_checkpointing(
                self.DynamicCheckpointTwiceModuleWeightSharing(use_reentrant=False),
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                static_graph=False,
                find_unused_parameters=True,
                # 由于动态模块可能不使用所有参数，允许梯度为 None 的情况
                allow_none_grads=True,
            )

    # 如果层间存在权重共享，DDP 的工作方式符合预期
    @skip_if_lt_x_gpu(2)
    @parametrize("use_reentrant", [True, False])
    def test_ddp_checkpointing_weight_sharing(self, use_reentrant):
        """
        Test that checkpointing with weight sharing works.
        """
        # 获取当前进程组
        process_group = self._get_process_group()
        # 设置当前设备为 CUDA 设备的特定排名
        torch.cuda.set_device(self.rank)
        # 对于每种使用桶视图和静态图的组合
        for use_bucket_view, static_graph in product((False, True), (False, True)):
            # 设置随机种子
            torch.manual_seed(31415)
            # 创建两个线性层，并共享它们的权重
            l1 = nn.Linear(20, 20)
            l2 = nn.Linear(20, 20)
            l1.weight = l2.weight
            # 创建顺序模型
            model = nn.Sequential(l1, l2)
            # 调用 _test_ddp_checkpointing 方法进行验证
            self._test_ddp_checkpointing(
                model,
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                static_graph=static_graph,
                run_checkpoint=True,
                use_reentrant=use_reentrant,
            )

    @skip_if_lt_x_gpu(2)
    def test_ddp_checkpointing_twice_weight_sharing(self):
        """
        Checkpointing should work with static graph in the case of checkpointing
        same layer twice and having weights shared across layers.
        """
        # 获取当前进程组
        process_group = self._get_process_group()
        # 设置当前设备为 CUDA 设备的特定排名
        torch.cuda.set_device(self.rank)
        # 对于每种使用桶视图的情况（True 和 False）
        for use_bucket_view in (True, False):
            # 创建测试模型并调用 _test_ddp_checkpointing 方法进行验证
            model = self._test_ddp_checkpointing(
                self.CheckpointTwiceModuleWeightSharing(),
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                static_graph=True,
            )
    # 定义一个测试方法，用于测试无效的 PowerSGD 状态
    def test_invalid_powerSGD_state(self):
        # 使用 product 函数生成多个参数组合进行测试
        for start_powerSGD_iter, use_error_feedback, warm_start in product(
            [0, 1], [True, False], [True, False]
        ):
            # 如果既不使用误差反馈也不进行热启动，则跳过当前参数组合的测试
            if not use_error_feedback and not warm_start:
                continue
            # 使用 assertRaisesRegex 断言，验证是否会抛出 ValueError 异常，并检查异常信息
            with self.assertRaisesRegex(
                ValueError,
                "Expect `start_powerSGD_iter` > 1 if `use_error_feedback` or `warm_start` is enabled, "
                "because PowerSGD can only be applied after the first two iterations in DDP.",
            ):
                # 创建 PowerSGDState 对象，检查是否会抛出预期的异常
                state = powerSGD.PowerSGDState(
                    process_group=None,
                    matrix_approximation_rank=1,
                    start_powerSGD_iter=start_powerSGD_iter,
                    use_error_feedback=use_error_feedback,
                    warm_start=warm_start,
                )

    # 定义一个用于测试带有进程组的 DDP 的私有方法
    def _test_ddp_with_process_group(
        self,
        process_group,
        devices,
        device_ids,
        multi_device=False,
        gradient_as_bucket_view=False,
        """
        Note: we pass down `device_ids` all the way to DistributedDataParallel
        as part of the test. Below you find tests that either use a list of
        integers, a list of `torch.Device` instances, or an empty list.
        The `devices` argument is used to control placement of the model and
        must always be specified as list of `torch.Device` instances.
        """
        # 根据注释，将 `device_ids` 参数传递至 DistributedDataParallel 对象中
        # 作为测试的一部分。下面的测试使用整数列表、`torch.Device` 实例列表或空列表。
        # `devices` 参数用于控制模型的放置位置，必须始终指定为 `torch.Device` 实例的列表。

        local_batch_size = 1 if devices is None else len(devices)
        # 如果 `devices` 为 None，则本地批量大小为 1；否则为 `devices` 列表的长度
        global_batch_size = self.world_size * local_batch_size
        # 计算全局批量大小，为进程数乘以本地批量大小

        if multi_device:
            # 如果是多设备模式
            model, ddp_model, input, target = self._prepare_multi_device_module(
                process_group,
                devices,
                device_ids,
                global_batch_size,
                gradient_as_bucket_view,
            )
            # 准备多设备模型，并获取模型、DDP 模型、输入、目标
            ddp_logging_data = ddp_model._get_ddp_logging_data()
            self.assertTrue(ddp_logging_data.get("is_multi_device_module"))
            # 断言 DDP 日志数据中的 "is_multi_device_module" 属性为 True
        else:
            # 如果是单设备模式
            model, ddp_model, input, target = self._prepare_single_device_module(
                process_group,
                devices,
                device_ids,
                global_batch_size,
                gradient_as_bucket_view,
            )
            # 准备单设备模型，并获取模型、DDP 模型、输入、目标
            ddp_logging_data = ddp_model._get_ddp_logging_data()
            self.assertFalse(ddp_logging_data.get("is_multi_device_module"))
            # 断言 DDP 日志数据中的 "is_multi_device_module" 属性为 False

        def step_model(model, input, target):
            # 定义模型训练的步骤函数
            model.train()
            # 设置模型为训练模式
            output = model(input)
            # 将输入数据输入模型并获取输出
            loss = F.mse_loss(output, target.to(output.device))
            # 计算均方误差损失
            loss.backward()
            # 反向传播，计算梯度

        def update_parameters(model):
            # 更新模型参数函数
            for param in model.parameters():
                with torch.no_grad():
                    param -= param.grad
                param.grad = None

        # check two model parameters over 2 iterations
        # 检查两次迭代中的两个模型参数
        for iteration in range(2):
            # single cpu/gpu training
            # 单 CPU/GPU 训练
            step_model(model, input, target)

            # DDP training, DDP scatters subsets of input_cpu to nodes/GPUs
            # DDP 训练，DDP 将 input_cpu 的子集分散到节点/GPU
            step_model(
                ddp_model,
                input[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
                target[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
            )

            # Update weights and run a second iteration to shake out errors
            # 更新权重并运行第二次迭代以排除错误
            update_parameters(model)
            update_parameters(ddp_model)
            self.assertEqual(
                len(list(model.parameters())), len(list(ddp_model.parameters()))
            )
            # 断言模型参数的数量相同
            for i, j in zip(model.parameters(), ddp_model.parameters()):
                self.assertEqual(i, j, rtol=1.3e-06, atol=5e-5)
                # 逐个比较模型参数，相等性判断的相对/绝对容差设置为 1.3e-06 和 5e-5

            # Shuffle the input so that DDP input is different
            # 对输入进行随机排列，以确保 DDP 输入不同
            torch.manual_seed(1337 + iteration)
            input = input[torch.randperm(global_batch_size)]
    # 使用 DDP 通信钩子的 GPU 模型配置
    def _gpu_model_with_ddp_comm_hook(
        self, process_group, hook=None, gradient_as_bucket_view=False, state=None
    ):
        # 获取当前进程的 GPU 设备 ID
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        # 在指定的 GPU 设备上创建带有 DDP 的分布式数据并行模型
        gpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # 如果提供了通信钩子，则注册该钩子到 GPU 模型中
        if hook is not None:
            gpu_model.register_comm_hook(state, hook)

        # 返回配置好的 GPU 模型
        return gpu_model

    # 使用内置 DDP 通信钩子的 GPU 模型配置
    def _gpu_model_with_builtin_ddp_comm_hook(
        self, process_group, hook=None, gradient_as_bucket_view=False
    ):
        # 获取当前进程的 GPU 设备 ID
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        # 在指定的 GPU 设备上创建带有 DDP 的分布式数据并行模型
        gpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # 如果定义了内置 DDP 通信钩子，则将其注册到 GPU 模型中
        if hook is not None:
            gpu_model._register_builtin_comm_hook(hook)

        # 返回配置好的 GPU 模型
        return gpu_model

    # 运行模型并验证钩子效果
    def _run_and_verify_hook(self, model, input, expected_grad):
        # 执行模型的前向传播
        output = model(input, self.rank)

        # 执行模型的反向传播
        output.mean().backward()

        # 验证每个参数的梯度是否与期望的梯度一致
        [self.assertEqual(p.grad, expected_grad) for p in model.parameters()]

    # 简单的 DDP 通信钩子实现
    def _simple_hook(
        self, state: object, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        # 创建一个 Torch Future 对象
        fut = torch.futures.Future()
        # 设置 Future 对象的结果为与 GradBucket 缓冲区形状相同的全为1的张量
        fut.set_result(torch.ones_like(bucket.buffer()))

        # 定义 Future 完成后的处理函数
        def fut_then(fut):
            # 将 fut 的结果张量中的每个元素加1
            t = fut.value()
            return t + torch.ones_like(t)

        # 返回经过处理函数后的 Future 对象
        return fut.then(fut_then)

    # 测试模型输出不包含 NaN 值
    def _test_not_nan(self, model, x):
        # 使用模型进行前向传播
        y = model(x)
        # 断言模型输出不包含任何 NaN 值
        self.assertFalse(y.isnan().any().item())
        # 对模型的输出求和并执行反向传播
        y.sum().backward()
        # 断言每个参数的梯度不包含任何 NaN 值
        for p in model.parameters():
            self.assertFalse(p.grad.isnan().any().item())

    # 如果 GPU 数量小于 2，则跳过该测试
    @skip_if_lt_x_gpu(2)
    # 定义一个测试函数，用于测试同步批归一化在空输入时的行为
    def test_sync_batch_norm_only_empty_input(self):
        # 获取当前进程组
        pg = self._get_process_group()

        # 定义一个包含批归一化层的神经网络模型
        model = torch.nn.Sequential(
            nn.BatchNorm2d(2),  # 添加一个二维批归一化层
        ).to(device=self.rank)  # 将模型移动到指定的设备上
        model = DistributedDataParallel(
            model,
            device_ids=[self.rank],  # 设置设备ID列表
            process_group=pg,  # 设置进程组
        )
        # 将模型中的批归一化层转换为同步批归一化层
        model = nn.SyncBatchNorm.convert_sync_batchnorm(
            model,
            process_group=pg,  # 使用给定的进程组进行转换
        )

        model.train()  # 设置模型为训练模式

        # 只有 rank 0 接收到空输入
        x = torch.zeros(
            (1 if self.rank != 0 else 0, 2, 11, 13),  # 根据当前 rank 决定输入的形状
            dtype=torch.float32,
            device=self.rank,  # 将张量放在指定的设备上
        )

        # 设置输入张量需要梯度，这将触发反向传播过程中的集体通信
        x.requires_grad = True
        self._test_not_nan(model, x)  # 调用自定义测试方法，验证结果是否包含 NaN

        # 设置输入张量不需要梯度
        x.requires_grad = False
        self._test_not_nan(model, x)  # 再次调用测试方法，验证结果是否包含 NaN

        # 所有 rank 都接收到空输入
        x = torch.zeros((0, 2, 11, 13), dtype=torch.float32, device=self.rank)

        # 设置输入张量需要梯度，这将触发反向传播过程中的集体通信
        x.requires_grad = True
        self._test_not_nan(model, x)  # 调用自定义测试方法，验证结果是否包含 NaN

        # 设置输入张量不需要梯度
        x.requires_grad = False
        self._test_not_nan(model, x)  # 再次调用测试方法，验证结果是否包含 NaN

    # 如果 GPU 数量少于 2，则跳过该测试函数
    @skip_if_lt_x_gpu(2)
    def test_sync_batch_norm_empty_input(self):
        # 获取当前进程组
        pg = self._get_process_group()

        # 定义一个包含卷积层、批归一化层和线性层的神经网络模型
        model = torch.nn.Sequential(
            nn.Conv2d(2, 2, 3),  # 添加一个卷积层
            nn.BatchNorm2d(2),  # 添加一个二维批归一化层
            nn.Linear(28, 2),  # 添加一个线性层
        ).to(device=self.rank)  # 将模型移动到指定的设备上
        model = DistributedDataParallel(
            model,
            device_ids=[self.rank],  # 设置设备ID列表
            process_group=pg,  # 设置进程组
        )
        # 将模型中的批归一化层转换为同步批归一化层
        model = nn.SyncBatchNorm.convert_sync_batchnorm(
            model,
            process_group=pg,  # 使用给定的进程组进行转换
        )

        model.train()  # 设置模型为训练模式

        # 只有 rank 0 接收到空输入
        x = torch.zeros(
            (3 if self.rank != 0 else 0, 2, 30, 30),  # 根据当前 rank 决定输入的形状
            dtype=torch.float32,
            device=self.rank,  # 将张量放在指定的设备上
        )

        self._test_not_nan(model, x)  # 调用自定义测试方法，验证结果是否包含 NaN

        # 所有 rank 都接收到空输入
        x = torch.zeros((0, 2, 30, 30), dtype=torch.float32, device=self.rank)

        self._test_not_nan(model, x)  # 再次调用测试方法，验证结果是否包含 NaN

    # 定义一个数据类，包含两个可选的张量类型的成员变量
    @dataclass
    class CustomOutput:
        o1: Optional[torch.Tensor]  # 第一个成员变量，可选的张量类型
        o2: Dict[str, torch.Tensor]  # 第二个成员变量，字典类型，键为字符串，值为张量类型

    # 定义一个继承自 nn.Module 的模块类，用于生成自定义的数据类输出
    class DataclassOutputModule(nn.Module):
        def __init__(self, skip_o1):
            super().__init__()
            # 定义模块中的序列1，包含三个线性层
            self.seq1 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(3)])
            self.relu = nn.ReLU()  # 定义激活函数 ReLU
            # 定义模块中的序列2，包含三个线性层
            self.seq2 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(3)])
            self.skip_o1 = skip_o1  # 设置是否跳过 o1 的生成

        # 定义前向传播函数
        def forward(self, x):
            # 如果 skip_o1 为真，则 o1 为 None；否则，o1 为 seq1 的输出经过 ReLU 激活函数后的结果
            o1 = None if self.skip_o1 else self.relu(self.seq1(x))
            # o2 包含两个键值对："a" 对应 seq2 的输出，"b" 对应 seq2 的输出经过 ReLU 激活函数后的结果
            o2 = {"a": self.seq2(x), "b": self.relu(self.seq2(x))}
            # 返回自定义数据类的对象，包含 o1 和 o2
            return CommonDistributedDataParallelTest.CustomOutput(o1=o1, o2=o2)
    # 定义测试函数 `_test_dataclass_output`，用于测试数据类输出功能，接受参数 `skip_o1` 表示是否跳过 `o1` 属性
    def _test_dataclass_output(self, skip_o1):
        # 构造一个大小为 4x10 的张量列表，每个张量元素为 i 值，将其转移到当前进程的设备上
        net_x = torch.cat([torch.ones(4, 10) * i for i in range(self.world_size)]).to(
            self.rank
        )
        # 创建一个大小为 4x10 的张量，每个元素为当前进程的 rank 值，放置在当前进程的设备上
        ddp_x = torch.ones(4, 10, device=self.rank) * self.rank

        # 使用随机种子 `0` 来确保本地模型从相同的初始值开始
        torch.manual_seed(0)
        # 创建一个 DataclassOutputModule 实例 `net`，并将其放置在当前进程的设备上
        net = self.DataclassOutputModule(skip_o1=skip_o1).to(self.rank)
        # 使用分布式数据并行化策略，复制 `net`，将其放置在设备列表中包含当前进程 rank 值的设备上
        # 设置 `find_unused_parameters=True`，静态图为 `False`，使用自定义的进程组
        ddp = DistributedDataParallel(
            copy.deepcopy(net),
            device_ids=[self.rank],
            find_unused_parameters=True,
            static_graph=False,
            process_group=self._get_process_group(),
        )

        # 对 `net` 执行前向传播，得到输出 `net_out`
        net_out = net(net_x)
        # 对 `ddp` 执行前向传播，得到输出 `ddp_out`
        ddp_out = ddp(ddp_x)

        # 计算 `net_out` 的损失值，如果不跳过 `o1` 属性，则包括 `o1`、`o2["a"]` 和 `o2["b"]` 的和
        # 否则，仅计算 `o2["a"]` 和 `o2["b"]` 的和，使用当前进程的设备上的全 1 张量作为目标
        net_loss = F.mse_loss(
            net_out.o1 + net_out.o2["a"] + net_out.o2["b"]
            if not skip_o1
            else net_out.o2["a"] + net_out.o2["b"],
            torch.ones_like(net_out.o2["a"], device=self.rank),
        )
        
        # 计算 `ddp_out` 的损失值，逻辑与 `net_loss` 的计算类似
        ddp_loss = F.mse_loss(
            ddp_out.o1 + ddp_out.o2["a"] + ddp_out.o2["b"]
            if not skip_o1
            else ddp_out.o2["a"] + ddp_out.o2["b"],
            torch.ones_like(ddp_out.o2["a"], device=self.rank),
        )

        # 对 `net_loss` 进行反向传播
        net_loss.backward()
        # 对 `ddp_loss` 进行反向传播
        ddp_loss.backward()

        # 遍历 `net` 和 `ddp` 的参数，检查它们的梯度是否近似相等
        for p1, p2 in zip(net.parameters(), ddp.parameters()):
            if torch.is_tensor(p1.grad):
                self.assertTrue(p1.grad.allclose(p2.grad))
            else:
                self.assertEqual(p1.grad, p2.grad)

    # 装饰器函数 `skip_if_lt_x_gpu`，当 GPU 数量小于 2 时跳过测试
    @skip_if_lt_x_gpu(2)
    # 测试数据类输出功能，不跳过 `o1` 属性的情况
    def test_dataclass_output(self):
        self._test_dataclass_output(skip_o1=False)

    # 装饰器函数 `skip_if_lt_x_gpu`，当 GPU 数量小于 2 时跳过测试
    @skip_if_lt_x_gpu(2)
    # 测试数据类输出功能，跳过 `o1` 属性的情况
    def test_dataclass_output_unused_param(self):
        self._test_dataclass_output(skip_o1=True)
# 定义一个测试类，继承自 TestCase，用于测试 ComputeBucketAssignmentTest 的方法
class ComputeBucketAssignmentTest(TestCase):

    # 测试单一大小限制和单一数据类型的情况
    def test_single_limit_single_dtype(self):
        # 创建包含四个空张量的列表，每个张量包含不同数量的元素
        tensors = [
            torch.empty([100], dtype=torch.float),
            torch.empty([200], dtype=torch.float),
            torch.empty([100], dtype=torch.float),
            torch.empty([50], dtype=torch.float),
        ]
        # 调用 _compute_bucket_assignment_by_size 方法计算分桶分配结果及每个分桶的大小限制
        result, per_bucket_size_limits = dist._compute_bucket_assignment_by_size(
            tensors, [400]
        )
        # 断言每个分桶的大小限制都为 400
        self.assertTrue(all(size_lim == 400 for size_lim in per_bucket_size_limits))
        # 断言分桶分配结果符合预期
        self.assertEqual([[0], [1], [2], [3]], result)

    # 测试单一大小限制和多数据类型的情况
    def test_single_limit_multi_dtype(self):
        # 创建包含六个空张量的列表，包含不同数据类型和数量的元素
        tensors = [
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
        ]
        # 调用 _compute_bucket_assignment_by_size 方法计算分桶分配结果及每个分桶的大小限制
        result, per_bucket_size_limits = dist._compute_bucket_assignment_by_size(
            tensors, [400]
        )
        # 断言每个分桶的大小限制都为 400
        self.assertTrue(all(size_lim == 400 for size_lim in per_bucket_size_limits))
        # 断言分桶分配结果符合预期
        self.assertEqual([[0, 2], [1, 3], [4], [5]], result)

    # 测试多大小限制和单一数据类型的情况
    def test_multi_limit_single_dtype(self):
        # 创建包含四个空张量的列表，每个张量包含相同数量的元素
        tensors = [
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
        ]
        # 调用 _compute_bucket_assignment_by_size 方法计算分桶分配结果及每个分桶的大小限制
        result, per_bucket_size_limits = dist._compute_bucket_assignment_by_size(
            tensors, [40, 80]
        )
        # 断言每个分桶的大小限制符合预期
        self.assertEqual(per_bucket_size_limits, [40, 80, 80])
        # 断言分桶分配结果符合预期
        self.assertEqual([[0], [1, 2], [3]], result)

    # 测试多大小限制和多数据类型的情况
    def test_multi_limit_multi_dtype(self):
        # 创建包含六个空张量的列表，包含不同数据类型和数量的元素
        tensors = [
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
        ]
        # 调用 _compute_bucket_assignment_by_size 方法计算分桶分配结果及每个分桶的大小限制
        result, per_bucket_size_limits = dist._compute_bucket_assignment_by_size(
            tensors, [200, 400]
        )
        # 断言分桶分配结果符合预期
        self.assertEqual([[0], [1], [2, 4], [3, 5]], result)
        # 断言每个分桶的大小限制符合预期
        self.assertEqual(per_bucket_size_limits, [200, 200, 400, 400])


# 定义一个抽象通信测试类 AbstractCommTest
class AbstractCommTest:
    
    # 定义 op_timeout_sec 属性，返回超时时间为 1 秒
    @property
    def op_timeout_sec(self):
        return 1
    
    # 定义 world_size 属性，返回世界大小为 2
    @property
    def world_size(self):
        return 2
    
    # 定义 device 属性，抛出异常，提示需要测试子类重写该属性
    @property
    def device(self):
        self.fail("test subclass didn't override device")
    # 验证给定进程组 `pg` 中的序列号，并返回该序列号
    def _verify_sequence_number_across_pg(self, pg, verify_pg):
        # 获取进程组 `pg` 的序列号
        seq_num = pg._get_sequence_number_for_group()
        # 创建一个空列表 `obj_list`，长度为 `verify_pg` 中的进程数目，用于收集数据
        obj_list = [None for _ in range(dist.get_world_size(verify_pg))]
        # 使用 `verify_pg` 进程组进行全局收集操作，收集 `seq_num` 数据
        dist.all_gather_object(obj_list, seq_num, group=verify_pg)
        # 断言所有收集的数据都相同
        self.assertEqual(len(set(obj_list)), 1)
        # 返回收集的第一个对象，即序列号
        return obj_list[0]

    # 测试序列号是否递增
    def _test_sequence_num_incremented(self, process_group, ranks):
        # 使用单独的进程组 `verify_pg` 来验证初始序列号，以避免自身操作导致序列号增加
        verify_pg = dist.new_group(
            ranks=ranks,
            backend="gloo",
        )
        # 断言两个进程组中的进程数量相同
        assert dist.get_world_size(process_group) == dist.get_world_size(verify_pg)

        # 如果当前进程组 `process_group` 中的进程不是 `c10d` 中定义的组成员，则返回 `-1`
        initial_num = (
            self._verify_sequence_number_across_pg(
                pg=process_group, verify_pg=verify_pg
            )
            if not c10d._rank_not_in_group(process_group)
            else -1
        )

        # 验证序列号是否递增
        for i in range(10):
            # 创建一个包含单个元素的张量 `t`，在当前 CUDA 设备上
            t = torch.ones(1, device=torch.cuda.current_device())
            # 在 `process_group` 进程组上进行全局归约操作
            dist.all_reduce(t, group=process_group)
            # 如果当前进程不是 `c10d` 中定义的组成员，则跳过后续验证
            if not c10d._rank_not_in_group(process_group):
                # 获取当前进程组中的序列号 `seq_num`
                seq_num = self._verify_sequence_number_across_pg(
                    pg=process_group,
                    verify_pg=verify_pg,
                )
                # 断言当前序列号比初始序列号增加了 `i + 1`
                self.assertEqual(initial_num + i + 1, seq_num)

        # 当 `process_group` 中的进程数量大于 2 时进行以下测试
        if dist.get_world_size(process_group) > 2:
            # 如果当前进程不是 `c10d` 中定义的组成员，则跳过后续测试
            if dist.get_rank(process_group) not in [0, 2]:
                # 在 `process_group` 进程组上进行异步的全局归约操作
                dist.all_reduce(t, group=process_group, async_op=True)
            # 现在排名为 0 和 2 的进程应该比其他进程落后一个序列号
            if not c10d._rank_not_in_group(process_group):
                # 获取当前进程组的序列号 `seq_num`
                seq_num = process_group._get_sequence_number_for_group()
                # 获取当前进程的排名
                rank = dist.get_rank(process_group)
                # 创建一个空列表 `obj_list`，长度为 `verify_pg` 中的进程数目，用于收集数据
                obj_list = [None for _ in range(dist.get_world_size(verify_pg))]
                # 使用 `verify_pg` 进程组进行全局收集操作，收集 `(rank, seq_num)` 数据
                dist.all_gather_object(obj_list, (rank, seq_num), group=verify_pg)
                # 将收集到的数据转换成字典 `rank_to_seq_num`
                rank_to_seq_num = dict(obj_list)
                # 断言收集到的序列号值不重复
                self.assertEqual(len(set(rank_to_seq_num.values())), 2)
                # 断言排名为 0 和 2 的进程的序列号相同
                self.assertEqual(rank_to_seq_num[0], rank_to_seq_num[2])
                # 预期的相同值为除了排名为 0 和 2 的进程外的其他所有序列号值
                expected_same = {
                    rank_to_seq_num[i]
                    for i in rank_to_seq_num.keys()
                    if i not in [0, 2]
                }
                # 断言预期的相同值只有一个
                self.assertEqual(len(expected_same), 1)
                # 断言排名为 0 的进程的序列号比排名为 1 的进程的序列号增加了 1
                self.assertEqual(rank_to_seq_num[0] + 1, rank_to_seq_num[1])
    # 测试函数：设置默认组的序列号增加
    def _test_sequence_num_incremented_default_group(self, backend_name):
        # 设置当前 CUDA 设备为指定的 GPU
        torch.cuda.set_device(self.rank)
        # 创建一个文件存储对象，用于分布式通信
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用指定的后端名称、总进程数、当前进程的排名和文件存储对象
        dist.init_process_group(
            backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 执行默认组中序列号增加的测试，传入默认的进程组和所有进程的排名列表
        self._test_sequence_num_incremented(
            c10d._get_default_group(),
            ranks=list(range(dist.get_world_size())),
        )

    # 测试函数：设置子组的序列号增加
    def _test_sequence_num_incremented_subgroup(self, backend_name):
        # 设置当前 CUDA 设备为指定的 GPU
        torch.cuda.set_device(self.rank)
        # 创建一个文件存储对象，用于分布式通信
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用指定的后端名称、总进程数、当前进程的排名和文件存储对象
        dist.init_process_group(
            backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 创建子组，包含指定的进程排名
        subgroup_ranks = [0, 1, 2]
        subgroup = dist.new_group(subgroup_ranks)
        # 执行子组中序列号增加的测试，传入子组和子组的进程排名列表
        self._test_sequence_num_incremented(subgroup, subgroup_ranks)

    # 测试函数：设置默认进程组的序列号为全局唯一值
    def _test_sequence_num_set_default_pg(self, backend):
        # 创建一个文件存储对象，用于分布式通信
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用指定的后端名称、总进程数、当前进程的排名和文件存储对象
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 获取默认进程组对象
        default_pg = c10d._get_default_group()
        # 获取默认进程组的序列号
        seq_num = default_pg._get_sequence_number_for_group()
        # 创建对象列表，用于存储所有进程的序列号
        obj_list = [None for _ in range(dist.get_world_size())]
        # 使用全局通信收集所有进程的序列号到 obj_list 中
        dist.all_gather_object(obj_list, seq_num)
        # 断言所有收集的序列号集合长度为1，即所有进程的序列号相同
        self.assertEqual(len(set(obj_list)), 1)

    # 测试函数：设置新进程组的序列号为全局唯一值
    def _test_sequence_num_set_new_group(self, backend):
        # 创建一个文件存储对象，用于分布式通信
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用指定的后端名称、总进程数、当前进程的排名和文件存储对象
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 创建一个新的进程子组，包含进程排名为0和1的进程
        subgroup = dist.new_group([0, 1])

        # 如果当前进程在子组中
        if not c10d._rank_not_in_group(subgroup):
            # 获取子组的序列号
            subgroup_seq = subgroup._get_sequence_number_for_group()
            # 创建对象列表，用于存储子组中所有进程的序列号
            obj_list = [None for _ in range(dist.get_world_size(subgroup))]
            # 使用子组的全局通信收集所有进程的序列号到 obj_list 中
            dist.all_gather_object(obj_list, subgroup_seq, group=subgroup)
            # 断言所有收集的序列号集合长度为1，即子组中所有进程的序列号相同
            self.assertEqual(len(set(obj_list)), 1)
    # 测试函数，用于验证在不属于指定组的情况下，分布式操作是否会发出警告信息
    def _test_warn_not_in_group(self, backend):
        # 创建文件存储对象，用于分布式进程组通信
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化分布式进程组
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 生成属于指定组的进程的排名列表
        in_group_ranks = list(filter(lambda x: x % 2 == 0, range(self.world_size)))
        # 创建新的进程组对象
        group = dist.new_group(in_group_ranks)

        # 在不属于指定组的进程上执行分布式操作时，设置在发生警告时只触发一次的正则表达式消息
        x = torch.zeros(2, 2).cuda(self.rank)
        xs = [torch.zeros(2, 2).cuda(self.rank) for _ in range(len(in_group_ranks))]
        if self.rank not in in_group_ranks:
            msg = ".*{}.*does not belong to.*"
            with self.assertWarnsOnceRegex(UserWarning, msg.format("all_gather")):
                dist.all_gather(xs, x, group=group)
            with self.assertWarnsOnceRegex(UserWarning, msg.format("all_reduce")):
                dist.all_reduce(x, group=group)
            with self.assertWarnsOnceRegex(UserWarning, msg.format("barrier")):
                dist.barrier(group=group)
            with self.assertWarnsOnceRegex(UserWarning, msg.format("broadcast")):
                dist.broadcast(x, src=0, group=group)
        else:
            # 在属于指定组的进程上执行分布式操作
            dist.all_gather(xs, x, group=group)
            dist.all_reduce(x, group=group)
            dist.barrier(group=group)
            dist.broadcast(x, src=0, group=group)

    # 测试函数，用于验证分布式进程组的排名成员关系和异常情况
    def _test_rank_membership(self, backend):
        # 创建文件存储对象，用于分布式进程组通信
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化分布式进程组
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 确保世界大小大于1
        self.assertTrue(self.world_size > 1)

        # 创建包含单个进程的新进程组对象
        group = dist.new_group(ranks=[1])
        # 验证获取指定进程在进程组中的本地排名
        self.assertEqual(dist.get_group_rank(group, 1), 0)
        # 验证当指定进程不在进程组中时抛出值错误异常
        with self.assertRaisesRegex(ValueError, "not part of group"):
            dist.get_group_rank(group, 0)
        # 验证当传入未注册的虚拟进程组对象时抛出值错误异常
        with self.assertRaisesRegex(ValueError, "not registered"):
            dist.get_group_rank(DummyProcessGroup(self.rank, self.world_size), 0)

        # 验证获取指定进程在全局进程组中的全局排名
        self.assertEqual(dist.get_global_rank(group, 0), 1)
        # 验证当指定进程不在全局进程组中时抛出值错误异常
        with self.assertRaisesRegex(ValueError, "not part of group"):
            dist.get_global_rank(group, 1)
        # 验证当传入未注册的虚拟进程组对象时抛出值错误异常
        with self.assertRaisesRegex(ValueError, "not registered"):
            dist.get_global_rank(DummyProcessGroup(self.rank, self.world_size), 0)

        # 验证获取进程组中所有进程的排名列表
        self.assertEqual(dist.get_process_group_ranks(group), [1])
    # 定义一个测试方法，用于检查张量数据类型不匹配的情况，使用指定的后端
    def _test_tensor_dtype_mismatch(self, backend):
        # 创建一个文件存储对象，用于分布式访问，文件名为 self.file_name，总共 self.world_size 个节点
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，指定后端、总节点数、当前节点的排名、存储对象
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

        # 创建一个包含值为 7 的张量，形状为 2x2，位于 self.device 上
        tensor = torch.ones(2, 2, device=self.device) * 7
        # 将张量转换为半精度浮点数格式
        tensor_h = tensor.half()
        # 创建一个包含多个张量的列表，每个张量形状为 2x2，位于 self.device 上
        tensor_list = [
            torch.zeros(2, 2, device=self.device) for _ in range(self.world_size)
        ]
        # 复制张量列表，确保原列表不受影响
        tensor_list_h = list(tensor_list)
        # 将索引为 1 的张量转换为半精度浮点数格式
        tensor_list_h[1] = tensor_list_h[1].half()

        # 使用断言验证在进行全局聚合时是否会抛出值错误，错误信息为 "tensors with different dtypes"
        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_gather(tensor_list_h, tensor)

        # 使用断言验证在进行全局聚合时是否会抛出值错误，错误信息为 "tensors with different dtypes"
        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_gather(tensor_list, tensor_h)

        # 使用断言验证在进行集中的全局聚合时是否会抛出值错误，错误信息为 "tensors with different dtypes"
        dist.all_gather_coalesced([tensor_list_h], tensor_list)
        dist.all_gather_coalesced([tensor_list], tensor_list_h)

        # 使用断言验证在进行集中的全局聚合时是否会抛出值错误，错误信息为 "tensors with different dtypes"
        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_reduce_coalesced(tensor_list_h)

        # 使用断言验证在进行减少分散时是否会抛出值错误，错误信息为 "tensors with different dtypes"
        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.reduce_scatter(tensor, tensor_list_h)

        # 使用断言验证在进行减少分散时是否会抛出值错误，错误信息为 "tensors with different dtypes"
        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.reduce_scatter(tensor_h, tensor_list)

        # 使用断言验证在进行单个全对全时是否会抛出值错误，错误信息为 "tensors with different dtypes"
        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_to_all_single(tensor_h, tensor)

        # 使用断言验证在进行全对全时是否会抛出值错误，错误信息为 "tensors with different dtypes"
        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_to_all(tensor_list_h, tensor_list)

        # 使用断言验证在进行全对全时是否会抛出值错误，错误信息为 "tensors with different dtypes"
        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_to_all(tensor_list, tensor_list_h)

        # 使用断言验证在进行分散时是否会抛出值错误，错误信息为 "tensors with different dtypes"
        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.scatter(tensor, tensor_list_h)

        # 使用断言验证在进行收集时是否会抛出值错误，错误信息为 "tensors with different dtypes"
        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.gather(tensor_h, tensor_list)

        # 使用断言验证在进行收集时是否会抛出值错误，错误信息为 "tensors with different dtypes"
        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.gather(tensor, tensor_list_h)

        # 使用断言验证在进行分散时是否会抛出值错误，错误信息为 "tensors with different dtypes"
        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.scatter(tensor_h, tensor_list)
    # 测试复杂数据类型的张量操作，使用指定的后端
    def _test_tensor_dtype_complex(self, backend):
        # 创建一个文件存储对象，用于分布式存储
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用指定的后端和参数
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

        # 创建一个随机张量，指定设备
        tensor = torch.rand(2, device=self.device)
        # 将张量视为复数张量
        tensor_c = torch.view_as_complex(tensor)
        # 创建一个张量列表，包含多个随机张量
        tensor_list = [
            torch.rand(2, device=self.device) for _ in range(self.world_size)
        ]
        # 复制张量列表
        tensor_list_c = list(tensor_list)
        # 将第二个张量视为复数张量
        tensor_list_c[1] = torch.view_as_complex(tensor_list_c[1])

        # 在所有进程之间收集张量数据到列表中
        dist.all_gather(tensor_list, tensor)
        dist.all_gather(tensor_list, tensor_c)
        dist.all_gather(tensor_list_c, tensor)
        dist.all_gather(tensor_list_c, tensor_c)

    # 测试布尔类型张量的分布式操作，使用指定的后端
    def _test_bool_tensors(self, backend):
        # 创建一个文件存储对象，用于分布式存储
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用指定的后端和参数
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 根据后端类型确定设备类型
        device = "cuda" if backend == "nccl" else "cpu"
        
        # 创建一个布尔类型张量，指定设备
        tensor = torch.tensor([1, 0, 0, 1], dtype=torch.bool, device=device)
        zeros = torch.tensor([0, 0, 0, 0], dtype=torch.bool, device=device)
        # 根据进程的排名选择广播的起始张量
        outensor = zeros if self.rank > 0 else tensor
        # 对指定的张量进行广播操作，源为进程0
        dist.broadcast(outensor, src=0)
        # 断言广播后的张量与原始张量相等
        self.assertEqual(outensor, tensor)
# Variant of AbstractCommTest that expects world size of 4
class AbstractLargeCommTest:
    @property
    def op_timeout_sec(self):
        # 返回操作超时时间，这里设定为1秒
        return 1

    @property
    def world_size(self):
        # 返回世界大小，这里设定为4
        return 4

    @property
    def device(self):
        # 抛出错误，要求子类实现这个属性
        raise RuntimeError("Implement me")

    def _test_new_group_local_sync(self, backend):
        # 创建文件存储对象，用于进程间通信
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，指定后端、世界大小、当前进程的排名和存储对象
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 获取当前进程的排名
        rank = dist.get_rank()
        # 计算同步组的输入和输出排名列表
        ranks_in = [rank, (rank + 2) % self.world_size]
        ranks_out = [i for i in range(self.world_size) if i not in ranks_in]
        # 断言当前进程在输入排名列表中，不在输出排名列表中
        self.assertIn(rank, ranks_in)
        self.assertNotIn(rank, ranks_out)

        # 测试创建不带本地同步的新进程组
        self.assertIsNone(
            dist.new_group(ranks=ranks_out, use_local_synchronization=True)
        )

        # 创建带本地同步的新进程组，并断言其类型为ProcessGroup
        new_pg = dist.new_group(ranks=ranks_in, use_local_synchronization=True)
        self.assertIsInstance(new_pg, dist.ProcessGroup)

        # PTD在创建进程组前对排名进行排序，所以[3, 1]实际上被分配为[1, 0]
        ranks_in.sort()
        # 断言当前进程在新进程组中的排名
        self.assertEqual(dist.get_group_rank(new_pg, rank), ranks_in.index(rank))
        # 断言新进程组中的排名列表与预期相符
        self.assertEqual(
            ranks_in,
            dist.get_process_group_ranks(new_pg),
            f"expecting {ranks_in} but got {dist.get_process_group_ranks(new_pg)}",
        )

    def _test_new_group_local_sync_sanity_check(self, backend):
        # 创建文件存储对象，用于进程间通信
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，指定后端、世界大小、当前进程的排名和存储对象
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 获取当前进程的排名
        rank = dist.get_rank()

        # 将世界分为2个进程组
        pg_idx = rank // 2
        ranks_in = [pg_idx * 2, pg_idx * 2 + 1]
        # 创建带本地同步的新进程组
        new_pg = dist.new_group(ranks=ranks_in, use_local_synchronization=True)

        # 创建输入张量
        input_tensor = torch.tensor([pg_idx, rank], device=self.device)
        # 创建输出张量列表
        output_tensor_list = [
            torch.tensor(
                [-1, -1],
                device=self.device,
            )
            for _ in range(new_pg.size())
        ]
        # 在新进程组中进行全局收集
        dist.all_gather(output_tensor_list, input_tensor, group=new_pg)

        # 预期输出张量列表
        expected = [
            torch.tensor([pg_idx, ranks_in[0]], device=self.device),
            torch.tensor([pg_idx, ranks_in[1]], device=self.device),
        ]
        # 断言输出张量列表与预期相符
        self.assertEqual(output_tensor_list, expected)
    def _test_new_group_local_sync_duplicate_pg(self, backend):
        """
        We should support users create multiple PGs with the same set of
        members, and no conflict in group name
        """
        # 创建一个文件存储对象，用于分布式文件存储
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化分布式进程组
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 获取当前进程的排名
        rank = dist.get_rank()

        # 将全体进程分割成两个进程组
        rank = dist.get_rank()
        pg_idx = rank // 2
        ranks_in = [pg_idx * 2, pg_idx * 2 + 1]
        new_pgs = []
        # 创建两个新的进程组，并添加到列表中
        for _ in range(2):
            new_pgs.append(
                dist.new_group(ranks=ranks_in, use_local_synchronization=True)
            )

        # 创建输入张量
        input_tensor = torch.tensor([pg_idx, rank], device=self.device)
        # 对每个新的进程组执行全局收集操作
        for new_pg in new_pgs:
            output_tensor_list = [
                torch.tensor(
                    [-1, -1],
                    device=self.device,
                )
                for _ in range(new_pg.size())
            ]
            dist.all_gather(output_tensor_list, input_tensor, group=new_pg)

            # 预期的输出结果
            expected = [
                torch.tensor([pg_idx, ranks_in[0]], device=self.device),
                torch.tensor([pg_idx, ranks_in[1]], device=self.device),
            ]
            # 断言实际输出与预期输出相等
            self.assertEqual(output_tensor_list, expected)
class CommTest(AbstractCommTest, MultiProcessTestCase):
    # CommTest 类继承自 AbstractCommTest 和 MultiProcessTestCase
    def setUp(self):
        # 执行父类的 setUp 方法，初始化测试环境
        super().setUp()
        # 启动多进程
        self._spawn_processes()

    def tearDown(self):
        # 执行父类的 tearDown 方法，清理测试环境
        super().tearDown()
        try:
            # 尝试删除 self.file_name 指定的文件
            os.remove(self.file_name)
        except OSError:
            # 如果文件不存在，忽略 OSError 异常
            pass

    def test_debug_level(self):
        try:
            # 尝试删除环境变量 TORCH_DISTRIBUTED_DEBUG
            del os.environ["TORCH_DISTRIBUTED_DEBUG"]
        except KeyError:
            # 如果环境变量不存在，捕获 KeyError 异常并忽略
            pass

        # 设置调试级别基于环境变量
        dist.set_debug_level_from_env()
        # 默认情况应该是关闭状态
        default_debug_mode = dist.get_debug_level()
        self.assertEqual(default_debug_mode, dist.DebugLevel.OFF)
        mapping = {
            "OFF": dist.DebugLevel.OFF,
            "off": dist.DebugLevel.OFF,
            "oFf": dist.DebugLevel.OFF,
            "INFO": dist.DebugLevel.INFO,
            "info": dist.DebugLevel.INFO,
            "INfO": dist.DebugLevel.INFO,
            "DETAIL": dist.DebugLevel.DETAIL,
            "detail": dist.DebugLevel.DETAIL,
            "DeTaIl": dist.DebugLevel.DETAIL,
        }
        invalid_debug_modes = ["foo", 0, 1, -1]

        # 遍历 mapping 中的每种调试模式
        for mode in mapping.keys():
            # 设置环境变量 TORCH_DISTRIBUTED_DEBUG 为当前 mode
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = str(mode)
            # 根据环境变量设置调试级别
            dist.set_debug_level_from_env()
            set_debug_mode = dist.get_debug_level()
            # 断言当前设置的调试级别与预期的 mapping[mode] 相等
            self.assertEqual(
                set_debug_mode,
                mapping[mode],
                f"Expected {mode} to map to {mapping[mode]} but got {set_debug_mode}",
            )

        # 遍历无效的调试模式列表
        for mode in invalid_debug_modes:
            # 设置环境变量 TORCH_DISTRIBUTED_DEBUG 为当前 mode
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = str(mode)
            # 检查设置无效调试级别时是否引发 ValueError 异常
            with self.assertRaisesRegex(
                ValueError, "The value of TORCH_DISTRIBUTED_DEBUG must"
            ):
                # 调用设置调试级别的函数
                dist.set_debug_level_from_env()


class DummyWork(dist._Work):
    # DummyWork 类继承自 dist._Work 类
    def wait(self, timeout=5.0):
        # 如果 CUDA 可用，同步当前 CUDA 流
        if torch.cuda.is_available():
            torch.cuda.current_stream().synchronize()
        # 返回 True 表示等待完成
        return True


class DummyProcessGroup(dist.ProcessGroup):
    # DummyProcessGroup 类继承自 dist.ProcessGroup 类
    def getBackendName(self):
        # 返回字符串 "Dummy" 表示后端名称为 Dummy
        return "Dummy"

    def allgather(self, output_tensor_lists, input_tensor_list, opts=None):
        # 将 input_tensor_list 中的每个 input_tensor 复制到对应的 output_tensor_lists 中
        for output_tensor_list, input_tensor in zip(
            output_tensor_lists, input_tensor_list
        ):
            for output_tensor in output_tensor_list:
                output_tensor.copy_(input_tensor)

        # 返回 DummyWork() 实例表示 allgather 操作的工作完成
        return DummyWork()

    def allreduce(self, tensor_list, opts=None):
        # 对 tensor_list 中的每个 tensor 执行加 2 操作
        for tensor in tensor_list:
            tensor.add_(2)

        # 返回 DummyWork() 实例表示 allreduce 操作的工作完成
        return DummyWork()
    # 定义一个方法用于实现 barrier 操作，同步所有进程的执行
    def barrier(self, opts=None):
        # 获取默认的分布式存储对象
        store = c10d._get_default_store()
        # 定义用于同步的键名
        key = "TEST:DummyProcessGroup:barrier"
        
        # 如果当前进程的 rank 是 0
        if self.rank() == 0:
            # 初始工作进程计数器
            worker_count = 0
            # 默认情况下，TCPServer 位于 rank 0 上。因此，rank 0 需要确保在其他进程完成使用存储之前不会过早退出。
            # 注意，_store_based_barrier 不能解决这个问题，因为所有进程在退出之前都需要运行至少一次 store.add(key, 0)，但不能保证此时 rank 0 仍然存活。
            while worker_count < self.size() - 1:
                worker_count = store.add(key, 0)
        else:
            # 对于非 rank 0 进程，只需添加键值对表示它们完成了
            store.add(key, 1)
        
        # 返回一个虚拟的 DummyWork 对象
        return DummyWork()

    # 定义一个方法用于广播操作，将 tensor_list 中的每个张量值加一
    def broadcast(self, tensor_list, opts=None):
        for tensor in tensor_list:
            tensor.add_(1)
        
        # 返回一个虚拟的 DummyWork 对象
        return DummyWork()

    # 定义一个方法用于 reduce scatter 操作，从 input_tensor_lists 中选择当前 rank 对应的张量，复制到 output_tensor_list 中
    def reduce_scatter(self, output_tensor_list, input_tensor_lists, opts=None):
        for output_tensor, input_tensor_list in zip(output_tensor_list, input_tensor_lists):
            output_tensor.copy_(input_tensor_list[self.rank()])
        
        # 返回一个虚拟的 DummyWork 对象
        return DummyWork()

    # 定义一个方法用于发送操作，将 tensor_list 中的每个张量值加一，发送给指定的 dst 进程
    def send(self, tensor_list, dst, tag=0):
        for tensor in tensor_list:
            tensor.add_(1)
        
        # 返回一个虚拟的 DummyWork 对象
        return DummyWork()

    # 定义一个方法用于接收操作，将 tensor_list 中的每个张量值加二
    def recv(self, tensor_list, src, tag=0):
        for tensor in tensor_list:
            tensor.add_(2)
        
        # 返回一个虚拟的 DummyWork 对象
        return DummyWork()
# 定义 PythonProcessGroupExtensionTest 类，继承自 MultiProcessTestCase 类，用于测试多进程场景
class PythonProcessGroupExtensionTest(MultiProcessTestCase):

    # 在每个测试方法执行之前调用，设置测试环境
    def setUp(self):
        super().setUp()  # 调用父类的 setUp 方法，初始化测试环境
        self._spawn_processes()  # 调用 _spawn_processes 方法，启动测试所需的进程

    # 在每个测试方法执行之后调用，清理测试环境
    def tearDown(self):
        super().tearDown()  # 调用父类的 tearDown 方法，清理测试环境
        try:
            os.remove(self.file_name)  # 尝试删除指定的文件名 self.file_name
        except OSError:
            pass  # 如果文件不存在，捕获 OSError 异常并忽略

    # 测试获取后端名称的方法
    def test_get_backend_name(self):
        dpg = DummyProcessGroup(0, 1)  # 创建 DummyProcessGroup 实例
        self.assertEqual("Dummy", dpg.name())  # 断言 DummyProcessGroup 的名称为 "Dummy"

    # 测试后端类属性的方法
    def test_backend_class_attr(self):
        dist.Backend.register_backend(
            "dummy", PythonProcessGroupExtensionTest.create_dummy
        )  # 注册名为 "dummy" 的后端，使用 create_dummy 方法作为创建函数
        self.assertEqual(dist.Backend.DUMMY, "dummy")  # 断言 dist.Backend.DUMMY 的值为 "dummy"
        self.assertEqual(
            dist.Backend._plugins["DUMMY"].creator_fn,
            PythonProcessGroupExtensionTest.create_dummy,
        )  # 断言 dist.Backend._plugins["DUMMY"].creator_fn 等于 PythonProcessGroupExtensionTest.create_dummy

    # 测试检查后端是否可用的方法
    def test_is_backend_available(self):
        self.assertEqual(dist.is_ucc_available(), dist.is_backend_available("ucc"))
        # 断言 dist.is_ucc_available() 的返回值与 dist.is_backend_available("ucc") 相等
        self.assertFalse(dist.is_backend_available("dummy"))  # 断言 dist.is_backend_available("dummy") 返回 False
        dist.Backend.register_backend(
            "dummy", PythonProcessGroupExtensionTest.create_dummy
        )  # 注册名为 "dummy" 的后端，使用 create_dummy 方法作为创建函数
        self.assertTrue(dist.is_backend_available("dummy"))  # 断言 dist.is_backend_available("dummy") 返回 True

    # 测试后端配置的方法
    def test_backend_config(self):
        dist.Backend.register_backend(
            "dummy", PythonProcessGroupExtensionTest.create_dummy
        )  # 注册名为 "dummy" 的后端，使用 create_dummy 方法作为创建函数

        # 确保可以使用以下参数创建后端配置
        backend_config_strings_and_expected_values = [
            (dist.Backend.GLOO, "cpu:gloo,cuda:gloo"),
            (dist.Backend.NCCL, "cuda:nccl"),
            (dist.Backend.MPI, "cpu:mpi,cuda:mpi"),
            (dist.Backend.UCC, "cpu:ucc,cuda:ucc"),
            (dist.Backend.DUMMY, "cpu:dummy,cuda:dummy"),
            ("DUMMY", "cpu:dummy,cuda:dummy"),
            ("dummy", "cpu:dummy,cuda:dummy"),
            ("cpu:dummy,cuda:dummy", "cpu:dummy,cuda:dummy"),
            ("cpu:dummy,cuda:nccl", "cpu:dummy,cuda:nccl"),
            ("cpu:gloo,cuda:dummy", "cpu:gloo,cuda:dummy"),
            ("cpu:gloo,cuda:nccl", "cpu:gloo,cuda:nccl"),
        ]

        for config_str, expected_value in backend_config_strings_and_expected_values:
            with self.subTest(config_str):
                # 确保这些配置字符串有效且不会引发 ValueError 异常
                config = dist.BackendConfig(config_str)
                self.assertEqual(str(config), expected_value)

        # 确保以下参数创建后端配置时会引发 ValueError 异常
        invalid_backend_config_strings = [
            "cpu:gloo,cuda:nccl,",  # 末尾有逗号
            "cpu:gloo,cuda:nccl,cpu:dummy",  # 设备重复
        ]
        for config_str in invalid_backend_config_strings:
            with self.subTest(config_str):
                with self.assertRaises(ValueError):
                    dist.BackendConfig(config_str)
    def test_init_process_group_with_multiple_backends(self):
        # 注册名为 "dummy" 的后端，使用 PythonProcessGroupExtensionTest.create_dummy 方法创建
        dist.Backend.register_backend(
            "dummy", PythonProcessGroupExtensionTest.create_dummy
        )

        # 设置环境变量 MASTER_ADDR 和 MASTER_PORT
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"

        # 初始化进程组，使用 "cpu:dummy,cuda:dummy" 作为后端，指定当前进程的排名和总进程数
        dist.init_process_group(
            "cpu:dummy,cuda:dummy", rank=self.rank, world_size=self.world_size
        )

        # 测试 all_gather 函数
        input_tensor = torch.ones(2, 2) * 7
        output_tensor_list = [torch.zeros(2, 2) for _ in range(self.world_size)]
        dist.all_gather(output_tensor_list, input_tensor)

        # 进行进程间同步
        dist.barrier()

        # 销毁进程组
        dist.destroy_process_group()

    class Options:
        def __init__(self):
            pass

        def create(self):
            pass

    @staticmethod
    def create_dummy(store, group_rank, group_size, timeout):
        # 创建一个 DummyProcessGroup 对象，使用给定的组排名和组大小
        return DummyProcessGroup(group_rank, group_size)

    def test_collectives(self):
        # 注册名为 "dummy" 的后端，使用 PythonProcessGroupExtensionTest.create_dummy 方法创建
        dist.Backend.register_backend(
            "dummy", PythonProcessGroupExtensionTest.create_dummy
        )

        # 设置环境变量 MASTER_ADDR 和 MASTER_PORT
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"

        # 初始化进程组，使用 "dummy" 作为后端，指定当前进程的排名和总进程数
        dist.init_process_group("dummy", rank=self.rank, world_size=self.world_size)

        # 测试 all_gather 函数
        input_tensor = torch.ones(2, 2) * 7
        output_tensor_list = [torch.zeros(2, 2) for _ in range(self.world_size)]
        dist.all_gather(output_tensor_list, input_tensor)

        # 检查 all_gather 的输出是否与输入相等
        for tensor in output_tensor_list:
            self.assertEqual(tensor, input_tensor)

        # 测试 all_reduce 函数
        input_tensor = torch.ones(2, 2) * 7
        dist.all_reduce(input_tensor)
        self.assertEqual(input_tensor, torch.ones(2, 2) * 7 + 2)

        # 测试 broadcast 函数
        input_tensor = torch.zeros(2, 2)
        dist.broadcast(input_tensor, 0, async_op=True).wait()
        self.assertEqual(torch.ones(2, 2), input_tensor)

        # 测试 reduce_scatter 函数
        output_tensor = torch.zeros(2, 2)
        input_tensor_list = [torch.ones(2, 2) for _ in range(self.world_size)]
        dist.reduce_scatter(output_tensor, input_tensor_list)
        self.assertEqual(output_tensor, torch.zeros(2, 2) + 1)

        # 进行进程间同步
        dist.barrier()

        # 销毁进程组
        dist.destroy_process_group()
    # 定义测试方法，用于发送和接收张量的分布式测试
    def test_send_recv(self):
        # 注册名为 "dummy" 的后端，用于创建虚拟进程组
        dist.Backend.register_backend(
            "dummy", PythonProcessGroupExtensionTest.create_dummy
        )

        # 设置环境变量，指定主节点地址和端口号
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        
        # 初始化进程组，使用 "dummy" 后端，设置当前进程的排名和总进程数
        dist.init_process_group("dummy", rank=self.rank, world_size=self.world_size)

        # 测试发送功能
        input_tensor = torch.zeros(2, 2)
        dist.send(input_tensor, (self.rank + 1) % self.world_size)
        # 断言：验证发送后的张量值是否为全 1
        self.assertEqual(input_tensor, torch.zeros(2, 2) + 1)

        # 使用断言捕获 ValueError 异常，测试发送给自身的情况
        with self.assertRaises(ValueError):
            dist.send(input_tensor, dist.get_rank())

        # 测试接收功能
        input_tensor = torch.zeros(2, 2)
        dist.recv(input_tensor, (self.rank + 1) % self.world_size)
        # 断言：验证接收后的张量值是否为全 2
        self.assertEqual(input_tensor, torch.zeros(2, 2) + 2)

        # 同步所有进程，确保所有进程达到同步点
        dist.barrier()

        # 注意：此处故意不调用 `destroy_process_group`，因为不是所有的用户应用都会显式调用它。
# 使用给定的测试类实例化参数化测试
instantiate_parametrized_tests(CommonDistributedDataParallelTest)

# 定义一个测试类，继承自MultiProcessTestCase
class ProcessGroupWithDispatchedCollectivesTests(MultiProcessTestCase):

    # 返回世界大小为1的属性方法
    @property
    def world_size(self):
        return 1

    # 设置测试的前置条件
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    # 清理测试的后置条件
    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    # 测试初始化进程组（可选后端）
    def test_init_process_group_optional_backend(self):
        # 创建一个临时文件作为存储
        with tempfile.NamedTemporaryFile(delete=False) as f:
            store = dist.FileStore(f.name, self.world_size)
            # 如果系统支持 gloo 和 nccl 后端，则初始化进程组
            if dist.is_gloo_available() and dist.is_nccl_available():
                dist.init_process_group(
                    store=store,
                    rank=self.rank,
                    world_size=self.world_size,
                )
                # 销毁进程组
                dist.destroy_process_group()

    # 测试为所有后端初始化进程组
    def test_init_process_group_for_all_backends(self):
        # 遍历所有后端类型
        for backend in dist.Backend.backend_list:
            # 如果后端类型未定义，则跳过
            if backend == dist.Backend.UNDEFINED:
                continue
            # 如果是 MPI 后端且系统不支持 MPI，则跳过
            elif backend == dist.Backend.MPI:
                if not dist.is_mpi_available():
                    continue
            # 如果是 NCCL 后端且系统不支持 NCCL 或者没有 CUDA 设备，则跳过
            elif backend == dist.Backend.NCCL:
                if not dist.is_nccl_available() or not torch.cuda.is_available():
                    continue
            # 如果是 GLOO 后端且系统不支持 GLOO，则跳过
            elif backend == dist.Backend.GLOO:
                if not dist.is_gloo_available():
                    continue
            # 如果是 UCC 后端且系统不支持 UCC，则跳过
            elif backend == dist.Backend.UCC:
                if not dist.is_ucc_available():
                    continue

            # 创建一个临时文件作为存储
            with tempfile.NamedTemporaryFile(delete=False) as f:
                store = dist.FileStore(f.name, self.world_size)
                # 初始化指定后端的进程组
                dist.init_process_group(
                    backend=backend,
                    rank=self.rank,
                    world_size=self.world_size,
                    store=store,
                )
                # 获取默认的进程组，并断言其属性
                pg = c10d._get_default_group()
                self.assertEqual(pg.rank(), self.rank)
                self.assertEqual(pg.size(), self.world_size)
                self.assertEqual(pg.name(), str(backend))

                # 销毁进程组
                dist.destroy_process_group()
    # 使用不同的张量调用集合操作，确保张量能够正确分发
    def _call_collective_with_varying_tensors(self, backend, collective, *args):
        # 调用集合操作，确保张量能正确分发

        # TODO: 在将来将更新为不依赖特定后端的方式
        device = "cuda" if backend == "nccl" else "cpu"
        # 确保在调用分发时支持的设备（cpu、cuda）
        tensor = torch.zeros(2, 2, device=torch.device(device))
        # 多张量集合操作
        if collective == dist.barrier:
            collective()
        elif collective in (dist.all_gather, dist.gather):
            collective([tensor], tensor, *args)
        elif collective == dist.scatter:
            collective(tensor, [tensor], *args)
        elif collective in (dist.reduce_scatter, dist.all_to_all):
            # gloo 不支持 reduce_scatter 或 all_to_all
            if backend != "gloo":
                if collective == dist.reduce_scatter:
                    collective(tensor, [tensor], *args)
                else:
                    collective([tensor], [tensor], *args)
        else:
            collective(tensor, *args)

    # TODO: 后端将会被替换为非指定后端
    def _test_collectives(self, backend):
        # 使用文件存储初始化分布式进程组
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 集合操作及其参数列表
        collectives_and_args = [
            (dist.reduce, self.rank),
            (dist.broadcast, self.rank),
            (dist.all_reduce,),
            (dist.all_gather,),
            (dist.reduce_scatter,),
            (dist.barrier,),
            (dist.all_to_all,),
            (dist.scatter,),
        ]
        for collective, *args in collectives_and_args:
            with self.subTest(collective=collective, args=args):
                self._call_collective_with_varying_tensors(backend, collective, *args)

    def _test_allreduce_coalesced(self, backend):
        # 使用文件存储初始化分布式进程组
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # TODO: 在将来将更新为不依赖特定后端的方式
        device = "cuda" if backend == "nccl" else "cpu"
        tensors = [torch.ones(10, 10, device=torch.device(device))]
        # 执行所有张量的累积归约操作
        dist.all_reduce_coalesced(tensors, dist.ReduceOp.SUM)
        for tensor in tensors:
            self.assertEqual(tensor, torch.ones(10, 10) * self.world_size)
    # 定义一个测试函数，用于测试分布式环境下的 all_to_all_single 操作
    def _test_all_to_all_single(self, backend):
        # 创建一个文件存储对象，用于分布式进程间通信
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化分布式进程组，指定后端、总进程数、当前进程的排名和存储对象
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 根据后端选择设备，如果是 nccl 使用 CUDA 设备，否则使用 CPU 设备
        device = "cuda" if backend == "nccl" else "cpu"
        # 创建一个输入张量，全部元素初始化为 1，在指定设备上
        input_tensor = torch.ones(2, 2, device=torch.device(device))
        # 创建一个输出张量，全部元素初始化为 0，在指定设备上
        output_tensor = torch.zeros(2, 2, device=torch.device(device))
        # 执行分布式 all_to_all_single 操作，将输入张量内容分发到各个进程，并收集结果到输出张量
        dist.all_to_all_single(output_tensor, input_tensor)
class CompilerTest(MultiProcessTestCase):
    # CompilerTest 类，继承自 MultiProcessTestCase，用于测试编译器功能

    def setUp(self):
        # 设置测试环境
        super().setUp()
        self._spawn_processes()
        # 启动多进程

    def tearDown(self):
        # 清理测试环境
        super().tearDown()
        try:
            os.remove(self.file_name)
            # 尝试移除临时文件
        except OSError:
            pass
            # 如果文件不存在则忽略

    def _get_process_group(self):
        # 定义获取进程组的抽象方法
        raise NotImplementedError("To be implemented by subclass")
        # 抛出未实现异常，需要子类实现该方法

    def _test_work_wait(self, x: torch.Tensor, comm_fn: Callable):
        # 定义测试工作等待方法，接受 torch.Tensor 类型的输入 x 和 Callable 类型的通信函数 comm_fn

        pg = self._get_default_group()
        # 获取默认进程组

        def fn(x: torch.Tensor) -> torch.Tensor:
            # 定义内部函数 fn，接受 torch.Tensor 类型的输入 x，返回 torch.Tensor 类型的结果

            # N.B.: explicitly wrapping with CommTensor instead of updating
            # all_reduce Python implementation, as the later will need more
            # discussion.
            # 注意：显式使用 CommTensor 包装，而不是更新 all_reduce 的 Python 实现，后者需要更多讨论。
            y = CommTensor(x + x)
            # 使用 CommTensor 封装 x + x 的结果
            work, z = comm_fn(y, group=pg)
            # 调用 comm_fn 处理 y，并返回工作对象 work 和结果 z
            # this wait() will be ignored in tracing mode as
            # ProxyTorchDispatchMode only supports torch.Tensor, _ProxyTensor,
            # and torch.nn.Parameter objects
            # 在跟踪模式下，这个 wait() 方法将被忽略，因为 ProxyTorchDispatchMode 仅支持 torch.Tensor、_ProxyTensor 和 torch.nn.Parameter 对象
            work.wait()
            # 等待工作完成

            if isinstance(z, list):
                return [zz * 2 for zz in z]
                # 如果 z 是列表，则返回每个元素乘以 2 的结果
            elif isinstance(z, torch.Tensor):
                return z * 2
                # 如果 z 是 torch.Tensor，则返回 z 的每个元素乘以 2 的结果
            else:
                raise RuntimeError("Unexpected return type")
                # 抛出运行时错误，表示返回类型不符合预期

        xx = x.clone()
        # 克隆输入张量 x，赋值给 xx

        # trace fn into a GraphModule
        traced_fn = make_fx(fn)(xx)
        # 使用 make_fx 对 fn 进行跟踪，返回 GraphModule 对象 traced_fn
        traced_fn.graph.lint()
        # 对跟踪的图进行检查
        traced_fn.graph.eliminate_dead_code()
        # 消除死代码

        # make sure the mul op indeed waits for comm
        # 确保乘法操作确实等待通信

        for node in traced_fn.graph.nodes:
            # 遍历跟踪图中的节点
            if node.op == "call_function" and "mul.Tensor" in node.target.__name__:
                # 如果节点是调用函数且目标函数名称包含 "mul.Tensor"
                prev = node.args[0]
                curr = None
                waited = False
                commed = False
                while prev is not None and not commed:
                    curr = prev
                    waited |= all(
                        [
                            curr.op == "call_function",
                            curr.target == _wait_comm,
                        ]
                    )
                    commed |= all(
                        [
                            curr.op == "call_function",
                            CommTensor._is_supported(curr.target.__name__),
                        ]
                    )
                    prev = curr.args[0]

                self.assertTrue(waited)
                # 断言已经等待过通信
                self.assertTrue(commed)
                # 断言已经完成通信

        # Update input to make sure we are not recording it as constant during
        # tracing.
        # 更新输入以确保在跟踪过程中不将其记录为常量

        x += 1
        # 将输入 x 加 1
        xx += 1
        # 将输入 xx 加 1

        y = fn(x)
        # 调用 fn 处理 x，返回结果 y
        yy = traced_fn(xx)
        # 调用跟踪后的 traced_fn 处理 xx，返回结果 yy

        # check correctness
        # 检查正确性
        self.assertEqual(y, yy)
        # 断言 y 和 yy 相等

        xx += 1
        # 将输入 xx 再次加 1
        yy = traced_fn(xx)
        # 再次调用 traced_fn 处理 xx，返回结果 yy

        self.assertNotEqual(y, yy)
        # 断言 y 和 yy 不相等

    def _test_allreduce_work_wait(self, tensor):
        # 定义测试全局归约工作等待的方法，接受 tensor 参数

        def comm_fn(tensor, group=None):
            # 定义通信函数 comm_fn，接受 tensor 和 group 参数
            work = dist.all_reduce(tensor, group=group, async_op=True)
            # 使用分布式通信执行全局归约操作，返回异步工作对象 work
            return work, tensor

        self._test_work_wait(tensor, comm_fn=comm_fn)
        # 调用 _test_work_wait 方法，测试 tensor 上的 comm_fn 操作
    # 测试函数，用于测试分布式通信操作的异步工作和等待
    def _test_allgather_work_wait(self, tensor):
        # 定义通信函数，输入参数为 tensor 和可选的 group
        def comm_fn(tensor, group=None):
            # 创建一个与 tensor 相同形状的零张量列表，长度为 group 的大小
            out_tensors = [torch.zeros_like(tensor) for _ in range(group.size())]
            # 执行分布式 all_gather 操作，将 tensor 发送给所有组成员，异步操作
            work = dist.all_gather(out_tensors, tensor, group=group, async_op=True)
            # 等待 all_gather 操作的完成
            work.wait()

            # 返回执行的工作对象和聚合后的张量之和
            return work, sum(out_tensors)

        # 调用测试函数，传入 tensor 和定义的通信函数
        self._test_work_wait(tensor, comm_fn=comm_fn)

    def _test_allgather_into_tensor_work_wait(self, tensor):
        def comm_fn(tensor, group=None):
            # 创建一个与 tensor 相同形状的零张量列表，长度为 group 的大小
            out_tensors = [torch.zeros_like(tensor) for _ in range(group.size())]
            # 将 out_tensors 中的张量沿指定维度连接成一个输出张量
            output_tensor = torch.cat(out_tensors, dim=0)
            # 执行分布式 all_gather_into_tensor 操作，将 tensor 发送给所有组成员，异步操作
            work = dist.all_gather_into_tensor(
                output_tensor, tensor, group=group, async_op=True
            )
            # 等待 all_gather_into_tensor 操作的完成
            work.wait()

            # 返回执行的工作对象和聚合后的输出张量
            return work, output_tensor

        self._test_work_wait(tensor, comm_fn=comm_fn)

    def _test_reduce_scatter_work_wait(self, tensor):
        def comm_fn(tensor, group=None):
            # 创建包含多个复制 tensor 的输入张量列表，每个张量都加上对应的索引
            in_tensors = [tensor.clone() + i for i in range(group.size())]
            # 创建一个与 tensor 相同形状的零张量
            out_tensor = torch.zeros_like(tensor)
            # 执行分布式 reduce_scatter 操作，将输入张量列表分散到每个组成员，异步操作
            work = dist.reduce_scatter(
                out_tensor, in_tensors, group=group, async_op=True
            )
            # 返回执行的工作对象和输出张量
            return work, out_tensor

        self._test_work_wait(tensor, comm_fn=comm_fn)

    def _test_reduce_scatter_tensor_work_wait(self, tensor):
        def comm_fn(tensor, group=None):
            # 根据组成员数量将 tensor 切片，选择当前进程对应的切片
            out_tensor = torch.zeros_like(tensor).chunk(group.size(), dim=0)[self.rank]
            # 执行分布式 reduce_scatter_tensor 操作，将 tensor 分散到每个组成员，异步操作
            work = dist.reduce_scatter_tensor(
                out_tensor, tensor, group=group, async_op=True
            )
            # 返回执行的工作对象和输出张量
            return work, out_tensor

        self._test_work_wait(tensor, comm_fn=comm_fn)

    def _test_broadcast_work_wait(self, tensor):
        def comm_fn(tensor, group=None):
            # 执行分布式 broadcast 操作，从源进程 (src=0) 广播 tensor 到所有组成员，异步操作
            work = dist.broadcast(tensor, src=0, group=group, async_op=True)
            # 返回执行的工作对象和 tensor
            return work, tensor

        self._test_work_wait(tensor, comm_fn=comm_fn)

    def _test_scatter_work_wait(self, tensor):
        def comm_fn(tensor, group=None):
            # 如果当前进程是源进程 (src=0)，创建包含多个 tensor 的输入张量列表
            in_tensors = (
                [tensor + i for i in range(group.size())] if self.rank == 0 else None
            )
            # 创建一个与 tensor 相同形状的零张量
            out_tensor = torch.zeros_like(tensor)
            # 执行分布式 scatter 操作，从源进程 (src=0) 将输入张量列表发送到所有组成员，异步操作
            work = dist.scatter(
                out_tensor, in_tensors, src=0, group=group, async_op=True
            )
            # 返回执行的工作对象和输出张量
            return work, out_tensor

        self._test_work_wait(tensor, comm_fn=comm_fn)

    def _test_alltoall_work_wait(self, tensor):
        def comm_fn(tensor, group=None):
            # 创建一个与 tensor 相同形状的零张量列表，长度为 group 的大小
            out_tensors = [torch.zeros_like(tensor) for _ in range(group.size())]
            # 创建包含多个 tensor 的输入张量列表，每个元素都是 tensor
            in_tensors = [tensor for i in range(group.size())]
            # 执行分布式 all_to_all 操作，将输入张量列表发送到所有组成员并接收来自所有组成员的张量，异步操作
            work = dist.all_to_all(out_tensors, in_tensors, group=group, async_op=True)
            # 返回执行的工作对象和输出张量列表
            return work, out_tensors

        self._test_work_wait(tensor, comm_fn=comm_fn)
    # 定义一个测试方法，用于测试嵌套通信张量包装
    def _test_nested_comm_tensor_wrapping(self, tensor):
        # 定义通信函数，接收张量和通信组参数，对张量进行全局归约操作，并返回异步操作对象和原始张量
        def comm_fn(tensor, group=None):
            work = dist.all_reduce(CommTensor(tensor), group=group, async_op=True)
            return work, tensor

        # 调用测试方法，传入张量和定义好的通信函数
        self._test_work_wait(tensor, comm_fn=comm_fn)

    # 定义另一个测试方法，用于测试连续的通信操作和等待
    def _test_consecutive_comm_work_wait(self, tensor):
        # 定义通信函数，接收张量和通信组参数，首先执行异步全局归约操作，然后等待该操作完成，
        # 然后再执行第二次异步全局归约操作，并返回第二次操作的异步对象和原始张量
        def comm_fn(tensor, group=None):
            work1 = dist.all_reduce(tensor, group=group, async_op=True)
            work1.wait()
            work2 = dist.all_reduce(tensor, group=group, async_op=True)
            return work2, tensor

        # 调用测试方法，传入张量和定义好的通信函数
        self._test_work_wait(tensor, comm_fn=comm_fn)
# 定义 ReduceOpTest 类，用于测试 ReduceOp 相关功能
class ReduceOpTest(TestCase):

    # 测试函数，验证 ReduceOp 对象是否属于 c10d.ReduceOp 类型
    # Ref: https://github.com/pytorch/pytorch/issues/87191
    def test_op_isinstance_of_reduceop(self):
        # 遍历所有的 ReduceOp 枚举成员
        for reduce_op in (
            c10d.ReduceOp.SUM,
            c10d.ReduceOp.AVG,
            c10d.ReduceOp.PRODUCT,
            c10d.ReduceOp.MIN,
            c10d.ReduceOp.MAX,
            c10d.ReduceOp.BAND,
            c10d.ReduceOp.BOR,
            c10d.ReduceOp.BXOR,
        ):
            # 断言 reduce_op 是 c10d.ReduceOp 类型的实例
            self.assertTrue(isinstance(reduce_op, c10d.ReduceOp))

        # 遍历不同的 scale 值，验证生成的 nccl_premul_sum 是否为 ReduceOp 类型
        for scale in (torch.tensor(1.0), 2.0):
            self.assertTrue(
                isinstance(dist._make_nccl_premul_sum(scale), c10d.ReduceOp)
            )

    # 测试函数，验证 ReduceOp 对象是否支持复制操作
    # Ref: https://github.com/pytorch/pytorch/pull/87303#discussion_r1002879700
    def test_reduceop_copyable(self):
        # 遍历所有的 ReduceOp 枚举成员
        for reduce_op in (
            c10d.ReduceOp.SUM,
            c10d.ReduceOp.AVG,
            c10d.ReduceOp.PRODUCT,
            c10d.ReduceOp.MIN,
            c10d.ReduceOp.MAX,
            c10d.ReduceOp.BAND,
            c10d.ReduceOp.BOR,
            c10d.ReduceOp.BXOR,
        ):
            # 测试使用 copy.copy 和 copy.deepcopy 进行复制操作后是否相等
            self.assertEqual(copy.copy(reduce_op), reduce_op)
            self.assertEqual(copy.deepcopy(reduce_op), reduce_op)
            self.assertEqual(copy.copy(c10d.ReduceOp(reduce_op)), reduce_op)
            self.assertEqual(copy.deepcopy(c10d.ReduceOp(reduce_op)), reduce_op)

        # 遍历不同的 scale 值，验证生成的 nccl_premul_sum 是否支持复制操作
        for scale in (torch.tensor(1.0), 2.0):
            reduce_op = dist._make_nccl_premul_sum(scale)
            self.assertEqual(copy.copy(reduce_op), reduce_op)
            self.assertEqual(copy.deepcopy(reduce_op), reduce_op)

    # 测试函数，验证 ReduceOp 对象是否支持序列化和反序列化操作
    def test_reduceop_pickle(self):
        # 遍历所有的 ReduceOp 枚举成员
        for reduce_op in (
            c10d.ReduceOp.SUM,
            c10d.ReduceOp.AVG,
            c10d.ReduceOp.PRODUCT,
            c10d.ReduceOp.MIN,
            c10d.ReduceOp.MAX,
            c10d.ReduceOp.BAND,
            c10d.ReduceOp.BOR,
            c10d.ReduceOp.BXOR,
        ):
            # 使用 pickle 序列化和反序列化 reduce_op 对象
            pickle.loads(pickle.dumps(reduce_op))
            orig = c10d.ReduceOp(reduce_op)
            self.assertEqual(pickle.loads(pickle.dumps(orig)), orig)

        # 遍历不同的 scale 值，验证生成的 nccl_premul_sum 是否支持序列化和反序列化操作
        for scale in (torch.tensor(1.0), 2.0):
            reduce_op = dist._make_nccl_premul_sum(scale)
            self.assertEqual(pickle.loads(pickle.dumps(reduce_op)), reduce_op)

    # Ref: https://github.com/pytorch/pytorch/issues/90072
    # 定义一个测试方法，用于测试不同的 ReduceOp 操作
    def test_reduceop_equal(self):
        # 创建一个非 ReduceOp 类型的字符串
        not_reduceop = "abc"
        # 遍历所有的 ReduceOp 类型
        for reduce_op in (
            c10d.ReduceOp.SUM,
            c10d.ReduceOp.AVG,
            c10d.ReduceOp.PRODUCT,
            c10d.ReduceOp.MIN,
            c10d.ReduceOp.MAX,
            c10d.ReduceOp.BAND,
            c10d.ReduceOp.BOR,
            c10d.ReduceOp.BXOR,
        ):
            # 将 ReduceOp 类型转换为 ReduceOp 对象
            reduce_op_obj = c10d.ReduceOp(reduce_op)
            # 断言 ReduceOp 对象与自身相等
            self.assertEqual(reduce_op_obj, reduce_op_obj)
            # 断言 ReduceOp 对象与 ReduceOp 类型相等
            self.assertEqual(reduce_op_obj, reduce_op)
            # 断言 ReduceOp 对象与非 ReduceOp 类型不相等
            self.assertNotEqual(reduce_op_obj, not_reduceop)
            # 断言 ReduceOp 类型与非 ReduceOp 类型不相等
            self.assertNotEqual(reduce_op, not_reduceop)
            # TODO(crcrpar): 这里需要为关联性进行 `assertEqual`，即使
            # 比较 `RedOpType` 和 `ReduceOp` 比起 `ReduceOp` 和 `RedOptype`
            # 的比较更不太可能发生。
            # 调用 `RedOpType.__eq__(self, other)`
            self.assertNotEqual(reduce_op, reduce_op_obj)

            # 断言 None 不在 (reduce_op, reduce_op_obj) 中
            self.assertFalse(None in (reduce_op, reduce_op_obj))
            # 断言 not_reduceop 不在 (reduce_op, reduce_op_obj) 中
            self.assertFalse(not_reduceop in (reduce_op, reduce_op_obj))
class LocalRankTest(MultiProcessTestCase):
    @property
    def world_size(self):
        # 返回固定的进程数量 4
        return 4

    def setUp(self):
        # 执行父类的 setUp 方法
        super().setUp()
        # 启动多个子进程以进行测试
        self._spawn_processes()

    def tearDown(self):
        # 执行父类的 tearDown 方法
        super().tearDown()
        try:
            # 尝试删除指定的文件
            os.remove(self.file_name)
        except OSError:
            # 如果文件不存在则忽略
            pass

    def testWithoutEnv(self):
        # 测试当环境变量中没有 "LOCAL_RANK" 时是否抛出 RuntimeError 异常，异常信息包含 "LOCAL_RANK"
        with self.assertRaisesRegex(RuntimeError, "LOCAL_RANK"):
            dist.get_node_local_rank()

    def testWithoutEnvWithFallback(self):
        # 测试在没有 "LOCAL_RANK" 环境变量时，使用 fallback_rank 参数作为返回值
        self.assertEqual(dist.get_node_local_rank(fallback_rank=2), 2)

    def testNodeLocalRankOverridesFallback(self):
        # 设置环境变量 "LOCAL_RANK" 为当前进程的 rank
        os.environ["LOCAL_RANK"] = str(self.rank)
        # 测试 get_node_local_rank 方法在有 "LOCAL_RANK" 环境变量时是否返回正确的 rank 值，而不是使用 fallback_rank
        self.assertEqual(dist.get_node_local_rank(fallback_rank=123), self.rank)

    def testNodeLocalRank(self):
        # 设置环境变量 "LOCAL_RANK" 为当前进程的 rank
        os.environ["LOCAL_RANK"] = str(self.rank)
        # 测试 get_node_local_rank 方法在有 "LOCAL_RANK" 环境变量时是否返回正确的 rank 值
        self.assertEqual(dist.get_node_local_rank(), self.rank)


if __name__ == "__main__":
    # 断言当前没有初始化 CUDA 上下文，以确保在主进程上不会初始化 CUDA
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    # 运行所有的测试方法
    run_tests()
```