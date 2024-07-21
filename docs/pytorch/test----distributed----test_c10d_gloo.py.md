# `.\pytorch\test\distributed\test_c10d_gloo.py`

```
# Owner(s): ["oncall: distributed"]

import copy  # 导入深拷贝模块，用于复制对象
import logging  # 导入日志记录模块，用于输出日志信息
import math  # 导入数学函数模块，用于数学计算
import operator  # 导入操作符模块，用于进行操作符相关的操作
import os  # 导入操作系统模块，用于操作系统相关的功能
import random  # 导入随机数模块，用于生成随机数
import sys  # 导入系统模块，用于系统相关的功能
import tempfile  # 导入临时文件模块，用于创建临时文件和目录
from datetime import timedelta  # 从日期时间模块中导入时间间隔类
from functools import reduce  # 导入函数工具模块中的reduce函数，用于累积计算
from itertools import groupby  # 导入迭代工具模块中的groupby函数，用于分组迭代

import torch  # 导入PyTorch深度学习框架
import torch.distributed as c10d  # 导入PyTorch分布式模块

# 检查是否支持c10d并且Gloo后端可用，否则跳过测试
if not c10d.is_available() or not c10d.is_gloo_available():
    print("c10d GLOO not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import test_c10d_common  # 导入测试c10d常用函数模块
from test_c10d_common import (
    gpus_for_rank,
    LOOPBACK,
    ModuleForDdpCommHook,
    SparseGradientModule,
    Task,
)

import torch.distributed as dist  # 导入PyTorch分布式模块
import torch.nn.functional as F  # 导入PyTorch神经网络模块中的函数实现
import torch.testing._internal.common_utils as common  # 导入PyTorch内部测试通用工具模块
from torch import nn  # 从PyTorch中导入神经网络模块
from torch.distributed._shard.sharded_tensor import (  # 从分布式张量模块导入相关函数和类
    init_from_local_shards,
    Shard,
    ShardedTensor,
    ShardMetadata,
)
from torch.nn.parallel import DistributedDataParallel  # 从PyTorch中导入分布式数据并行模块
from torch.testing._internal.common_distributed import (  # 从PyTorch内部分布式测试模块导入相关函数
    create_device,
    MultiProcessTestCase,
    requires_gloo,
    simple_sparse_reduce_tests,
    skip_if_lt_x_gpu,
    skip_if_win32,
    verify_ddp_error_logged,
)
from torch.testing._internal.common_utils import (  # 从PyTorch内部测试通用工具模块导入相关函数
    retry_on_connect_failures,
    run_tests,
    skip_but_pass_in_sandcastle,
    TestCase,
)


def simple_reduce_tests(rank, world_size):
    tests = [
        (
            c10d.ReduceOp.SUM,  # 使用SUM操作符进行张量归约
            torch.tensor([rank + 1.0]),  # 创建张量，内容为rank + 1.0
            torch.tensor([float(world_size * (world_size + 1) / 2)]),  # 期望的归约结果张量
        ),
        (
            c10d.ReduceOp.PRODUCT,  # 使用PRODUCT操作符进行张量归约
            torch.tensor([rank + 1.0]),  # 创建张量，内容为rank + 1.0
            torch.tensor([float(math.factorial(world_size))]),  # 期望的归约结果张量
        ),
        (
            c10d.ReduceOp.MIN,  # 使用MIN操作符进行张量归约
            torch.tensor([rank + 1.0]),  # 创建张量，内容为rank + 1.0
            torch.tensor([1.0]),  # 期望的归约结果张量
        ),
        (
            c10d.ReduceOp.MAX,  # 使用MAX操作符进行张量归约
            torch.tensor([rank + 1.0]),  # 创建张量，内容为rank + 1.0
            torch.tensor([float(world_size)]),  # 期望的归约结果张量
        ),
    ]

    # 为BAND操作生成测试
    # 每次迭代中设置的位不同，以检查输出是否相应更改
    for i in range(4):
        vin = rank | (1 << i)  # 计算输入值vin，使用位运算设置i位
        vout = 1 << i  # 计算期望的输出值vout，使用位运算设置i位
        tests.append(
            (
                c10d.ReduceOp.BAND,  # 使用BAND操作符进行张量归约
                torch.tensor([vin], dtype=torch.int32),  # 创建整型张量，内容为vin
                torch.tensor([vout], dtype=torch.int32),  # 期望的归约结果整型张量
            ),
        )

    # 为BOR操作生成测试
    # 每次迭代中模拟较大的世界大小，每个排名贡献多个先前OR'ed的值
    for i in range(1, 5):
        vin = reduce(operator.or_, [rank * i + j for j in range(i)])  # 计算输入值vin，使用reduce和OR运算符
        vout = reduce(operator.or_, range(world_size * i))  # 计算期望的输出值vout，使用reduce和OR运算符
        tests.append(
            (
                c10d.ReduceOp.BOR,  # 使用BOR操作符进行张量归约
                torch.tensor([vin], dtype=torch.int32),  # 创建整型张量，内容为vin
                torch.tensor([vout], dtype=torch.int32),  # 期望的归约结果整型张量
            ),
        )

    # 为XOR操作生成测试
    # 每次迭代中模拟较大的世界大小，每个


这部分代码还没有完成，继续解释以下部分。
    # 循环遍历 i 从 1 到 4
    for i in range(1, 5):
        # 计算 vin，使用 reduce 和 operator.xor 对 rank * i + j 的值进行异或操作
        vin = reduce(operator.xor, [rank * i + j for j in range(i)])
        # 计算 vout，使用 reduce 和 operator.xor 对范围内的值进行异或操作，范围为 0 到 world_size * i - 1
        vout = reduce(operator.xor, range(world_size * i))
        # 将测试元组添加到 tests 列表中，包括 c10d.ReduceOp.BXOR、vin 的张量和 vout 的张量
        tests.append(
            (
                c10d.ReduceOp.BXOR,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    # 返回 tests 列表作为函数结果
    return tests
# 定义一个函数，用于生成一组简单的合并归约测试用例
def simple_coalesced_reduce_tests(rank, world_size):
    # 返回一个列表，包含多个元组，每个元组描述一个归约操作及其预期输入输出
    return [
        (
            c10d.ReduceOp.SUM,  # 归约操作为求和
            [torch.tensor([rank + 1.0]), torch.tensor([(rank + 1.0) ** 2])],  # 输入张量列表
            [
                torch.tensor([float(world_size * (world_size + 1) / 2)]),  # 预期输出张量列表的第一项
                torch.tensor(
                    [float(world_size * (world_size + 1) * (2 * world_size + 1) / 6)]
                ),  # 预期输出张量列表的第二项
            ],
        ),
        (
            c10d.ReduceOp.PRODUCT,  # 归约操作为乘积
            [torch.tensor([rank + 1.0]), torch.tensor([rank + 2.0])],  # 输入张量列表
            [
                torch.tensor([float(math.factorial(world_size))]),  # 预期输出张量列表的第一项
                torch.tensor([float(math.factorial(world_size + 1))]),  # 预期输出张量列表的第二项
            ],
        ),
        (
            c10d.ReduceOp.MIN,  # 归约操作为求最小值
            [torch.tensor([rank + x]) for x in [0.0, 1.0]],  # 输入张量列表
            [torch.tensor([0.0]), torch.tensor([1.0])],  # 预期输出张量列表
        ),
        (
            c10d.ReduceOp.MAX,  # 归约操作为求最大值
            [torch.tensor([rank + x]) for x in [1.0, 2.0]],  # 输入张量列表
            [torch.tensor([float(world_size)]), torch.tensor([world_size + 1.0])],  # 预期输出张量列表
        ),
    ]


# 定义一个函数，用于生成一组简单的多输入合并归约测试用例
def simple_multi_input_reduce_tests(rank, world_size):
    # 返回一个列表，包含多个元组，每个元组描述一个归约操作及其预期输入输出
    return [
        (
            c10d.ReduceOp.SUM,  # 归约操作为求和
            [torch.tensor([2 * rank + 0.0]), torch.tensor([2 * rank + 1.0])],  # 输入张量列表
            torch.tensor([float(world_size * (2 * world_size - 1))]),  # 预期输出张量
        ),
        (
            c10d.ReduceOp.PRODUCT,  # 归约操作为乘积
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],  # 输入张量列表
            torch.tensor([float(math.factorial(2 * world_size))]),  # 预期输出张量
        ),
        (
            c10d.ReduceOp.MIN,  # 归约操作为求最小值
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],  # 输入张量列表
            torch.tensor([1.0]),  # 预期输出张量
        ),
        (
            c10d.ReduceOp.MAX,  # 归约操作为求最大值
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],  # 输入张量列表
            torch.tensor([2.0 * world_size]),  # 预期输出张量
        ),
    ]


# 定义一个测试类 RendezvousEnvTest，继承自 TestCase 类
class RendezvousEnvTest(TestCase):
    # 测试方法，用于初始化日志记录器
    @requires_gloo()  # 标记需要使用 GLOO 后端
    @retry_on_connect_failures  # 如果连接失败则重试
    def test_logging_init(self):
        # 设置环境变量
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(common.find_free_port())
        os.environ["RANK"] = "0"

        previous_handlers = logging.root.handlers  # 获取初始化日志记录器前的处理程序列表

        # 初始化进程组，使用 GLOO 后端，初始化方法为环境变量方式
        c10d.init_process_group(backend="gloo", init_method="env://")

        current_handlers = logging.root.handlers  # 获取当前日志记录器的处理程序列表
        # 断言前后处理程序列表长度相同
        self.assertEqual(len(previous_handlers), len(current_handlers))
        # 逐一断言前后处理程序是否相同
        for current, previous in zip(current_handlers, previous_handlers):
            self.assertEqual(current, previous)

        c10d.destroy_process_group()  # 销毁进程组


# 定义一个测试类 TimeoutTest，继承自 AbstractTimeoutTest 和 TestCase 类
class TimeoutTest(test_c10d_common.AbstractTimeoutTest, TestCase):
    # 测试方法，用于测试默认存储超时时间（使用 GLOO 后端）
    @requires_gloo()  # 标记需要使用 GLOO 后端
    @retry_on_connect_failures  # 如果连接失败则重试
    def test_default_store_timeout_gloo(self):
        # 调用父类方法，测试默认存储超时时间
        self._test_default_store_timeout("gloo")


# 定义一个测试类 ProcessGroupGlooTest，继承自 MultiProcessTestCase 类
class ProcessGroupGlooTest(MultiProcessTestCase):
    # 这里应该还有一些测试方法，但给出的代码片段不完整，无法为其添加注释
    # 使用给定的存储、排名、总体大小和选项创建一个Gloo进程组对象
    pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, opts)
    # 在创建的进程组上执行 barrier 操作，确保所有进程都到达同步点
    dist.barrier(group=pg)
    # 返回创建的进程组对象
    return pg

    # 调用父类的 setUp 方法初始化测试环境
    super().setUp()
    # 调用 _spawn_processes 方法启动子进程
    self._spawn_processes()

    # 创建并返回一个ProcessGroupGloo选项对象，指定线程数和超时时间
    opts = c10d.ProcessGroupGloo._Options()
    opts._timeout = 50.0
    opts._devices = [create_device(interface=LOOPBACK)]
    opts._threads = threads
    return opts

@requires_gloo()
def test_multi_device_constructor(self):
    # 创建一个文件存储对象，用于ProcessGroupGloo的初始化
    store = c10d.FileStore(self.file_name, self.world_size)
    # 创建ProcessGroupGloo选项对象，并设置超时时间和设备列表
    opts = c10d.ProcessGroupGloo._Options()
    opts._timeout = 5.0
    opts._devices = [
        create_device(interface=LOOPBACK),
        create_device(interface=LOOPBACK),
    ]
    # 使用给定的存储、排名、总体大小和选项创建一个Gloo进程组对象
    pg = self._create_process_group_gloo(store, self.rank, self.world_size, opts)

    # 执行4次 allreduce 操作，确保每个设备都得到利用
    for fut in [pg.allreduce(torch.ones(i + 1)).get_future() for i in range(4)]:
        fut.wait()

@requires_gloo()
def test_empty_tensors(self):
    # 创建一个文件存储对象，用于ProcessGroupGloo的初始化
    store = c10d.FileStore(self.file_name, self.world_size)
    # 使用预设的 opts 方法返回ProcessGroupGloo选项对象
    pg = self._create_process_group_gloo(
        store, self.rank, self.world_size, self.opts()
    )

    # 创建一个空的 FloatTensor 列表
    xs = [torch.FloatTensor([])]
    # 在进程组中广播空的张量列表，并获取对应的 Future 对象
    fut = pg.broadcast(xs).get_future()
    # 等待广播操作完成
    fut.wait()
    # 获取广播结果
    output = fut.value()
    # 断言广播后的张量数量为零
    self.assertEqual(0, output[0].numel())
    # 断言广播后的张量与原始张量相等
    self.assertEqual(xs[0], output[0])

@requires_gloo()
    # 定义一个测试函数，用于测试广播操作的参数检查
    def test_broadcast_checks(self):
        # 创建一个基于文件的存储对象，用于进程组
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用Gloo后端创建一个进程组对象
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # 创建三个张量，分别用于测试广播时的不同情况
        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        # 测试当指定的根rank为负数时是否抛出RuntimeError异常
        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = -1
            opts.rootTensor = 0
            pg.broadcast([t1], opts)

        # 测试当指定的根rank超出世界大小时是否抛出RuntimeError异常
        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.world_size
            opts.rootTensor = 0
            pg.broadcast([t1], opts)

        # 测试当指定的根tensor索引为负数时是否抛出RuntimeError异常
        with self.assertRaisesRegex(RuntimeError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = -1
            pg.broadcast([t1], opts)

        # 测试当指定的根tensor索引超出提供的张量列表时是否抛出RuntimeError异常
        with self.assertRaisesRegex(RuntimeError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 1
            pg.broadcast([t1], opts)

        # 测试当没有提供任何张量时是否抛出RuntimeError异常
        with self.assertRaisesRegex(RuntimeError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([], opts)

        # 测试当提供的张量类型不同于根tensor类型时是否抛出RuntimeError异常
        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([t1, t2], opts)

        # 测试当提供的张量尺寸与根tensor尺寸不匹配时是否抛出RuntimeError异常
        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([t1, t3], opts)
    # 定义测试函数，用于测试广播基础功能，接受一个函数 fn 作为参数
    def _test_broadcast_basics(self, fn):
        # 创建一个文件存储，用于进程组
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用 Gloo 创建一个进程组
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # 定义广播函数，将数据 xs 广播到指定的根节点
        def broadcast(xs, rootRank, rootTensor):
            # 创建广播选项
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            # 发起广播操作，获取未来对象
            fut = pg.broadcast(xs, opts).get_future()
            fut.wait()  # 等待广播完成
            return fut.value()  # 返回广播结果

        # 每个进程都轮流作为根节点
        for i in range(self.world_size):
            # 使用输入张量运行函数 fn
            x = fn(torch.tensor([self.rank]))
            # 执行广播操作，广播单个输入张量
            output = broadcast([x], i, 0)
            # 断言广播后的输出与预期结果相等
            self.assertEqual(torch.tensor([i]), output[0])

            # 使用两个输入张量运行函数 fn
            num = 2
            for j in range(num):
                xs = [
                    fn(torch.tensor([self.rank * num + 0.0])),
                    fn(torch.tensor([self.rank * num + 1.0])),
                ]

                # 执行广播操作，广播两个输入张量
                output = broadcast(xs, i, j)
                # 断言广播后的输出与预期结果相等
                self.assertEqual(
                    torch.tensor([i * num + j], dtype=torch.float32), output[0]
                )
                self.assertEqual(
                    torch.tensor([i * num + j], dtype=torch.float32), output[1]
                )

        # 测试重载的方便函数
        x = torch.tensor([self.rank + 1.0])
        fut = pg.broadcast(x, root=0).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(torch.tensor([1.0]), result[0])

    # 标记为需要 Gloo 的测试函数
    @requires_gloo()
    def test_broadcast_basics(self):
        # 调用基础广播测试函数，传入一个克隆函数作为参数
        self._test_broadcast_basics(lambda t: t.clone())

    # 如果 GPU 小于 2，则跳过此测试
    @skip_if_lt_x_gpu(2)
    # 标记为需要 Gloo 的测试函数
    @requires_gloo()
    def test_broadcast_basics_cuda(self):
        # 调用基础广播测试函数，传入一个克隆并移至 GPU 的函数作为参数
        self._test_broadcast_basics(lambda t: t.clone().cuda())

    # 测试广播压力情况
    def _test_broadcast_stress(self, inputs):
        # 创建一个文件存储，用于进程组
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用 Gloo 创建一个进程组，并指定线程数为 8
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        # 创建广播操作句柄列表，将输入张量广播到对应的根节点
        work_handles = [
            pg.broadcast(inputs[i], root=(i % self.world_size))
            for i in range(len(inputs))
        ]
        # 遍历所有广播操作句柄，等待每个操作完成，并断言每个输入张量与预期结果相等
        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            self.assertEqual(
                torch.tensor([(i * self.world_size) + (i % self.world_size)]),
                inputs[i],
                msg=("Mismatch in iteration %d" % i),
            )

    # 标记为需要 Gloo 的测试函数
    @requires_gloo()
    def test_broadcast_stress(self):
        # 创建包含 1000 个张量的输入列表，每个张量包含特定值与进程 ID 相关联
        inputs = [torch.tensor([i * self.world_size + self.rank]) for i in range(1000)]
        # 调用广播压力测试函数
        self._test_broadcast_stress(inputs)

    # 如果 GPU 小于 2，则跳过此测试
    @skip_if_lt_x_gpu(2)
    # 标记为需要 Gloo 的测试函数
    @requires_gloo()
    def test_broadcast_stress_cuda(self):
        # 创建包含 1000 个张量的输入列表，每个张量包含特定值与进程 ID 相关联，并将它们移到 GPU
        inputs = [
            torch.tensor([i * self.world_size + self.rank]).cuda() for i in range(1000)
        ]
        # 调用 GPU 广播压力测试函数
        self._test_broadcast_stress(inputs)

    # 标记为需要 Gloo 的测试函数
    @requires_gloo()
    # 定义测试函数，用于检查 allreduce 方法的异常情况
    def test_allreduce_checks(self):
        # 创建文件存储对象，用于进程组
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用 Gloo 后端创建进程组对象
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # 创建不同类型和大小的张量用于测试
        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        # 测试空张量列表是否引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "requires non-empty tensor list"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([], opts)

        # 测试包含不同类型张量的列表是否引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t2], opts)

        # 测试包含不同大小张量的列表是否引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t3], opts)

    # 定义基础测试函数，用于测试 allreduce 方法的基本功能
    def _test_allreduce_basics(self, fn):
        # 创建文件存储对象，用于进程组
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用 Gloo 后端创建进程组对象
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # 单输入张量测试
        tests = simple_reduce_tests(self.rank, self.world_size)
        for op, input, expected in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensor = fn(input)
            fut = pg.allreduce([tensor], opts).get_future()
            fut.wait()
            result = fut.value()
            self.assertEqual(expected, result[0])

        # 多输入张量测试
        tests = simple_multi_input_reduce_tests(self.rank, self.world_size)
        for op, inputs, output in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensors = [fn(input) for input in inputs]
            fut = pg.allreduce(tensors, opts).get_future()
            fut.wait()
            result = fut.value()
            for tensor in result:
                self.assertEqual(output, tensor)

        # 测试重载的便捷函数（默认使用 sum 操作）
        x = fn(torch.tensor([self.rank + 1.0]))
        fut = pg.allreduce(x).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(
            torch.tensor([float(self.world_size * (self.world_size + 1) / 2)]),
            result[0],
        )

    # 使用 Gloo 后端测试基础功能的方法
    @requires_gloo()
    def test_allreduce_basics(self):
        self._test_allreduce_basics(lambda t: t.clone())

    # 如果 GPU 数量小于 2 则跳过测试，使用 Gloo 后端测试基础功能的 CUDA 版本
    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_allreduce_basics_cuda(self):
        self._test_allreduce_basics(lambda t: t.clone().cuda())
    # 定义一个方法用于测试 allreduce 操作在压力下的表现
    def _test_allreduce_stress(self, inputs):
        # 创建一个文件存储对象，用于进程组的通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建一个 Gloo 进程组对象
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        # 创建 future_handles 列表，存放每个 allreduce 操作的 future 对象
        future_handles = [
            pg.allreduce(inputs[i]).get_future() for i in range(len(inputs))
        ]
        # 遍历 future_handles 列表，等待每个 future 对象完成
        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            # 使用断言检查 allreduce 结果是否符合预期
            self.assertEqual(
                torch.tensor(
                    [
                        (i * self.world_size)
                        + (self.world_size * (self.world_size - 1) // 2)
                    ]
                ),
                future_handle.value()[0],
                msg=("Mismatch in iteration %d" % i),
            )

    # 标记需要 Gloo 支持的测试方法
    @requires_gloo()
    def test_allreduce_stress(self):
        # 创建输入张量列表，每个张量包含一个元素
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        # 调用 _test_allreduce_stress 方法进行测试
        self._test_allreduce_stress(inputs)

    # 标记需要 Gloo 支持的测试方法，使用 CUDA 加速版本
    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_allreduce_stress_cuda(self):
        # 创建输入张量列表，每个张量包含一个元素，并在 CUDA 上执行
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        # 调用 _test_allreduce_stress 方法进行测试
        self._test_allreduce_stress(inputs)

    # 标记需要 Gloo 支持的测试方法，测试 allreduce_coalesced 方法的各种检查
    @requires_gloo()
    def test_allreduce_coalesced_checks(self):
        # 创建一个文件存储对象，用于进程组的通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建一个 Gloo 进程组对象
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # 创建三个不同类型的张量用于测试
        t1 = torch.zeros(1, dtype=torch.float32)
        t2 = torch.zeros(1, dtype=torch.float64)
        t3 = torch.sparse_coo_tensor([[0]], [1], size=(1,))

        # 使用断言检查 allreduce_coalesced 方法处理空列表的情况
        with self.assertRaisesRegex(RuntimeError, "requires non-empty tensor list"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([], opts)

        # 使用断言检查 allreduce_coalesced 方法处理不同类型张量列表的情况
        with self.assertRaisesRegex(
            RuntimeError, "tensors must all have the same type"
        ):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1, t2], opts)

        # 使用断言检查 allreduce_coalesced 方法处理稀疏张量的情况
        with self.assertRaisesRegex(RuntimeError, "invalid tensor layout at index"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1, t3], opts)

        # 使用断言检查 allreduce_coalesced 方法处理不支持布局的情况
        with self.assertRaisesRegex(RuntimeError, "unsupported layout"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t3, t3.clone()], opts)

    # 标记需要至少一个 GPU 的支持，并且需要 Gloo 支持的测试方法，CUDA 版本
    @skip_if_lt_x_gpu(1)
    @requires_gloo()
    def test_allreduce_coalesced_checks_cuda(self):
        # 创建一个文件存储对象，用于进程组的通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建一个 Gloo 进程组对象
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # 创建一个浮点类型的张量在 CUDA 上执行
        t1 = torch.zeros(1, dtype=torch.float32)

        # 使用断言检查 allreduce_coalesced 方法处理不支持设备类型的情况
        with self.assertRaisesRegex(RuntimeError, "unsupported device type"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1.cuda(), t1.cuda()], opts)
    # 定义测试函数，测试 AllreduceCoalesced 的基本功能
    def _test_allreduce_coalesced_basics(self, fn):
        # 使用 FileStore 创建存储，传入文件名和进程组大小
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用 Gloo 创建进程组，传入存储、当前进程的排名、总进程数和其他选项
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # 获取简单的聚合减少测试用例，根据当前进程的排名和总进程数生成
        test_cases = simple_coalesced_reduce_tests(self.rank, self.world_size)
        # 对于每个测试用例，设置 AllreduceCoalesced 的选项
        for op, inputs, outputs in test_cases:
            opts = c10d.AllreduceCoalescedOptions()
            opts.reduceOp = op
            # 对输入列表中的每个输入张量应用函数 fn，生成张量列表
            tensors = [fn(x) for x in inputs]
            # 执行聚合减少操作，返回一个 Future 对象
            fut = pg.allreduce_coalesced(tensors, opts).get_future()
            fut.wait()  # 等待 Future 对象完成
            result = fut.value()  # 获取操作的结果
            # 检查每个结果张量与预期输出是否相等
            for result_tensor, expected in zip(result, outputs):
                self.assertEqual(result_tensor, expected)

    # 装饰器，要求使用 Gloo 后端
    @requires_gloo()
    # 测试 AllreduceCoalesced 的基本功能
    def test_allreduce_coalesced_basics(self):
        # 调用 _test_allreduce_coalesced_basics 函数，传入 lambda 表达式作为 fn 参数
        self._test_allreduce_coalesced_basics(lambda t: t.clone())

    # 定义函数，根据输入生成预期输出
    def _expected_output(self, i):
        ws = self.world_size
        return 2 * [torch.tensor([(i * ws) + (ws * (ws - 1) // 2)])]

    # 测试 AllreduceCoalesced 的压力情况
    def _test_allreduce_coalesced_stress(self, inputs):
        # 使用 FileStore 创建存储，传入文件名和进程组大小
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用 Gloo 创建进程组，传入存储、当前进程的排名、总进程数和其他选项
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        # 创建多个 Future 对象，每个对象执行一个输入的聚合减少操作
        future_handles = [
            pg.allreduce_coalesced(input).get_future() for input in inputs
        ]
        # 对每个 Future 对象进行等待和结果检查
        for i, future_handle in enumerate(future_handles):
            future_handle.wait()  # 等待 Future 对象完成
            result = future_handle.value()  # 获取操作的结果
            # 检查结果是否符合预期输出
            self.assertEqual(
                self._expected_output(i),
                result,
                msg=f"Mismatch in iteration {i}",
            )

    # 装饰器，要求使用 Gloo 后端
    @requires_gloo()
    # 测试 AllreduceCoalesced 的压力情况
    def test_allreduce_coalesced_stress(self):
        # 创建包含大量输入张量的列表
        inputs = [2 * [torch.tensor([i + self.rank])] for i in range(1000)]
        # 调用 _test_allreduce_coalesced_stress 函数，传入 inputs 作为参数
        self._test_allreduce_coalesced_stress(inputs)

    # 装饰器，要求使用 Gloo 后端
    @requires_gloo()
    # 测试异步执行的 AllreduceCoalesced 操作
    def test_allreduce_coalesced_async(self):
        # 使用 FileStore 创建存储，传入文件名和进程组大小
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用 Gloo 后端，传入当前进程的排名、总进程数、存储和其他选项
        c10d.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size, store=store
        )

        # 创建多个输入张量的列表
        xs = [2 * [torch.tensor([i + self.rank])] for i in range(2)]
        # 执行异步的 AllreduceCoalesced 操作，返回多个 Future 对象
        futs = [c10d.all_reduce_coalesced(x, async_op=True) for x in xs]
        # 等待所有 Future 对象完成
        torch.futures.wait_all(futs)
        # 对每个 Future 对象进行结果检查
        for i, fut in enumerate(futs):
            # 检查结果是否符合预期输出
            self.assertEqual(
                self._expected_output(i),
                fut.wait(),
                msg=f"Mismatch in iteration {i}",
            )

    # 装饰器，要求使用 Gloo 后端
    @requires_gloo()
    # 定义测试方法，用于验证稀疏张量全局归约的各种检查情况
    def test_sparse_allreduce_checks(self):
        # 创建基于文件的存储，指定文件名和进程组大小
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用Gloo后端创建进程组，指定存储、当前进程的排名、总进程数以及其他选项
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # 创建普通的全0张量
        t1 = torch.zeros([1])
        # 创建稀疏 COO 格式的张量，只有一个非零元素
        t2 = torch.sparse_coo_tensor([[0]], [1], size=(2,))
        # 创建稀疏 COO 格式的张量，只有一个非零元素
        t3 = torch.sparse_coo_tensor([[0]], [1], size=(4,))

        # 使用断言检查运行时错误，期望得到“requires non-empty tensor list”异常
        with self.assertRaisesRegex(RuntimeError, "requires non-empty tensor list"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([], opts)

        # 使用断言检查运行时错误，期望得到“invalid tensor layout”异常
        with self.assertRaisesRegex(RuntimeError, "invalid tensor layout"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t2], opts)

        # 使用断言检查运行时错误，期望得到“invalid tensor size”异常
        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t2, t3], opts)

        # 遍历不支持的归约操作，验证是否会触发“unsupported reduction operation”异常
        for op in [c10d.ReduceOp.PRODUCT, c10d.ReduceOp.MIN, c10d.ReduceOp.MAX]:
            with self.assertRaisesRegex(
                RuntimeError, "unsupported reduction operation"
            ):
                opts = c10d.AllreduceOptions()
                opts.reduceOp = op
                pg.allreduce([t3], opts)

    # 测试稀疏张量全局归约的基本功能
    def _test_sparse_allreduce_basics(self, fn):
        # 创建基于文件的存储，指定文件名和进程组大小
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用Gloo后端创建进程组，指定存储、当前进程的排名、总进程数以及其他选项
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # 针对每个进程数测试，生成简单的稀疏归约测试
        for num_inputs_per_rank in [1, 2]:
            tests = simple_sparse_reduce_tests(
                self.rank, self.world_size, num_inputs=num_inputs_per_rank
            )
            # 遍历每组输入和预期输出
            for inputs, outputs in tests:
                # 对输入列表中的每个输入应用给定的函数（在 lambda t: t 的情况下，即不修改）
                tensors = [fn(input) for input in inputs]
                # 执行全局归约操作，获取 future 对象
                fut = pg.allreduce(tensors).get_future()
                fut.wait()  # 等待操作完成
                result = fut.value()  # 获取归约结果
                # 使用断言验证归约后的结果与预期输出是否一致
                self.assertEqual(tensors, outputs)
                self.assertEqual(result, outputs)

    # 使用Gloo后端测试稀疏张量全局归约的基本功能
    @requires_gloo()
    def test_sparse_allreduce_basics(self):
        self._test_sparse_allreduce_basics(lambda t: t)

    # 如果GPU数量少于2，跳过测试；否则使用Gloo后端测试稀疏张量全局归约的基本功能（CUDA版本）
    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_sparse_allreduce_basics_cuda(self):
        self._test_sparse_allreduce_basics(lambda t: t.clone().cuda())

    # 如果GPU数量少于2，跳过测试；否则使用Gloo后端测试分发的CUDA版本稀疏张量全局归约
    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_sparse_allreduce_cuda_dispatched(self):
        # 创建基于文件的存储，指定文件名和进程组大小
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化使用Gloo后端的分布式进程组，指定存储、当前进程的排名、总进程数
        dist.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        # 生成简单的稀疏归约测试
        tests = simple_sparse_reduce_tests(self.rank, self.world_size, num_inputs=1)
        # 遍历每组输入和预期输出
        for inputs, outputs in tests:
            # 克隆最后一个输入张量，并将其复制到CUDA设备上
            tensors = inputs[-1].clone().cuda()
            # 执行全局归约操作，异步执行
            work = dist.all_reduce(tensors, async_op=True)
            work.wait()  # 等待操作完成
            # 使用断言验证归约后的结果是否与预期输出一致
            self.assertEqual([tensors], outputs)

    # 使用Gloo后端
    @requires_gloo()
    def test_allgather_into_tensor_coalesced(self):
        # 创建一个基于文件存储的分布式存储对象
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化分布式进程组，使用Gloo后端，指定存储、当前进程的等级和总进程数
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        # 设置随机种子为42
        torch.manual_seed(42)
        # 输入张量的形状列表
        in_shapes = [(5, 5), (10, 10), (15, 15)]
        # 输出张量的形状列表，每个张量的第一个维度为原始形状第一个维度乘以进程总数
        out_shapes = [(s[0] * self.world_size,) + s[1:] for s in in_shapes]

        # 创建空的输出张量列表
        outputs = [torch.empty(s) for s in out_shapes]
        # 创建随机填充的输入张量列表
        inputs = [torch.rand(s) for s in in_shapes]
        # 执行全局所有进程间张量的聚合操作，并将结果存储在输出张量中
        work = dist.group.WORLD.allgather_into_tensor_coalesced(outputs, inputs)
        # 等待操作完成
        work.wait()

        # 验证每个输出张量是否与其对应输入张量的重复拼接相等
        for output, input in zip(outputs, inputs):
            expect = torch.cat([input] * self.world_size)
            self.assertTrue(torch.allclose(output, expect))

    @requires_gloo()
    def test_reduce_scatter_tensor(self):
        # 创建一个基于文件存储的分布式存储对象
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化分布式进程组，使用Gloo后端，指定存储、当前进程的等级和总进程数
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        # 设置随机种子为42
        torch.manual_seed(42)
        # 输出张量的形状
        out_shape = (20, 20)
        # 输入张量的形状，第一个维度为输出张量第一个维度乘以进程总数
        in_shape = (out_shape[0] * self.world_size,) + out_shape[1:]

        # 创建空的输出张量
        output = torch.empty(out_shape)
        # 创建随机填充的输入张量
        input = torch.rand(in_shape)
        # 执行全局张量的分散归约操作，并将结果存储在输出张量中，异步操作
        work = dist.reduce_scatter_tensor(output, input, async_op=True)
        # 等待操作完成
        work.wait()

        # 计算期望的输出结果
        expect = (
            input.view(self.world_size, *out_shape).chunk(self.world_size)[self.rank]
            * self.world_size
        )
        # 验证输出张量是否与期望的结果相等
        self.assertTrue(torch.allclose(output, expect))

    @requires_gloo()
    def test_reduce_scatter_tensor_coalesced(self):
        # 创建一个基于文件存储的分布式存储对象
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化分布式进程组，使用Gloo后端，指定存储、当前进程的等级和总进程数
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        # 设置随机种子为42
        torch.manual_seed(42)
        # 输出张量的形状列表
        out_shapes = [(5, 5), (10, 10), (15, 15)]
        # 输入张量的形状列表，每个张量的第一个维度为输出张量第一个维度乘以进程总数
        in_shapes = [(s[0] * self.world_size,) + s[1:] for s in out_shapes]

        # 创建空的输出张量列表
        outputs = [torch.empty(s) for s in out_shapes]
        # 创建随机填充的输入张量列表
        inputs = [torch.rand(s) for s in in_shapes]
        # 执行全局张量的分散归约操作，并将结果存储在输出张量中
        work = dist.group.WORLD.reduce_scatter_tensor_coalesced(outputs, inputs)
        # 等待操作完成
        work.wait()

        # 验证每个输出张量是否与其对应输入张量的拼接结果乘以进程总数相等
        for output, input in zip(outputs, inputs):
            expect = (
                input.view(self.world_size, *output.shape).chunk(self.world_size)[
                    self.rank
                ]
                * self.world_size
            )
            self.assertTrue(torch.allclose(output, expect))
    # 定义测试函数，用于测试 scatter 操作的基本功能
    def _test_scatter_basics(self, fn):
        # 创建基于文件的存储对象，指定文件名和世界大小
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建基于 Gloo 后端的进程组对象，传入存储对象、当前进程的排名、世界大小和选项参数
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # 预先分配输入/输出张量
        input = [fn(torch.tensor([self.rank])) for _ in range(self.world_size)]
        outputs = [fn(torch.tensor([-1])) for _ in range(self.world_size)]

        # 依次作为 scatter 根节点，并积累工作项
        futures = []
        for i in range(self.world_size):
            opts = c10d.ScatterOptions()
            opts.rootRank = i
            if i == self.rank:
                # 当前进程为根节点时，将输入数据散射到其他进程
                futures.append(pg.scatter([outputs[i]], [input], opts).get_future())
            else:
                # 其他进程接收数据，不发送任何数据
                futures.append(pg.scatter([outputs[i]], [], opts).get_future())

        # 等待工作完成
        for i in range(self.world_size):
            futures[i].wait()
            result = futures[i].value()
            # 断言散射后的结果与预期一致
            self.assertEqual(torch.tensor([i]), result[0])

    @requires_gloo()
    # 测试 scatter 基本功能
    def test_scatter_basics(self):
        self._test_scatter_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    # 在 CUDA 上测试 scatter 基本功能
    def test_scatter_basics_cuda(self):
        self._test_scatter_basics(lambda t: t.clone().cuda())

    # 定义测试函数，用于测试 scatter 操作的压力情况
    def _test_scatter_stress(self, inputs, fn):
        # 创建基于文件的存储对象，指定文件名和世界大小
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建基于 Gloo 后端的进程组对象，传入存储对象、当前进程的排名、世界大小和选项参数（线程数为 8）
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        
        # 初始化输出列表，包含多个子列表，每个子列表包含各个进程的输出数据
        outputs = [
            [fn(torch.tensor([-1])) for _ in range(self.world_size)]
            for _ in range(len(inputs))
        ]
        future_handles = []
        # 遍历输入数据和根节点，进行 scatter 操作
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.ScatterOptions()
                opts.rootRank = root
                if root == self.rank:
                    # 当前进程为根节点时，发送输入数据给其他进程
                    fut = pg.scatter(
                        [outputs[i][root]], [[fn(e) for e in inputs[i]]], opts
                    ).get_future()
                else:
                    # 其他进程接收数据，不发送任何数据
                    fut = pg.scatter([outputs[i][root]], [], opts).get_future()
                future_handles.append(fut)

        # 等待所有 scatter 操作完成
        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            iter = i // self.world_size
            root = i % self.world_size
            result = future_handle.value()

            # 断言散射后的结果与预期一致
            self.assertEqual(
                torch.tensor([iter + root]),
                result[0],
                msg=("Mismatch in iteration %d for rank %d" % (iter, root)),
            )

    @requires_gloo()
    # 定义测试方法：设置 GLOO 过程组的超时时间
    def test_set_gloo_pg_timeout(self):
        # 创建基于文件的存储，用于 GLOO 过程组
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建 GLOO 过程组
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )
        # 执行全局归约操作
        pg.allreduce(torch.rand(10))
        # 断言 GLOO 过程组的超时时间为 50 秒
        self.assertEqual(pg.options._timeout, timedelta(seconds=50))
        # 设置 GLOO 过程组的默认超时时间为 23 秒
        pg._set_default_timeout(timedelta(seconds=23))
        # 再次断言 GLOO 过程组的超时时间为 23 秒
        self.assertEqual(pg.options._timeout, timedelta(seconds=23))

    # 标记为需要 GLOO 支持的测试方法：scatter 操作的压力测试
    def test_scatter_stress(self):
        # 构造输入数据列表
        inputs = [
            [torch.tensor([i + self.rank]) for _ in range(self.world_size)]
            for i in range(1000)
        ]
        # 调用内部方法进行 scatter 操作的压力测试
        self._test_scatter_stress(inputs, lambda t: t.clone())

    # 在 Sandcastle 中跳过执行的测试方法，因为存在问题
    @skip_but_pass_in_sandcastle(
        "Test is flaky, see https://github.com/pytorch/pytorch/issues/15963"
    )
    # 根据 GPU 数量跳过测试方法的装饰器：scatter 操作的 CUDA 版本压力测试
    @skip_if_lt_x_gpu(2)
    # 标记为需要 GLOO 支持的测试方法：scatter 操作的 CUDA 版本压力测试
    @requires_gloo()
    def test_scatter_stress_cuda(self):
        # 构造输入数据列表
        inputs = [
            [torch.tensor([i + self.rank]) for _ in range(self.world_size)]
            for i in range(1000)
        ]
        # 调用内部方法进行 scatter 操作的 CUDA 版本压力测试
        self._test_scatter_stress(inputs, lambda t: t.clone().cuda())

    # 标记为需要 GLOO 支持的测试方法：gather 基础操作测试
    @requires_gloo()
    def _test_gather_basics(self, fn):
        # 创建基于文件的存储，用于 GLOO 过程组
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建 GLOO 过程组
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # 预先分配输入/输出张量
        input = [fn(torch.tensor([self.rank]))]
        outputs = [fn(torch.tensor([-1])) for _ in range(self.world_size)]

        # 轮流作为 gather 的根节点并累积工作项
        futures = []
        for i in range(self.world_size):
            opts = c10d.GatherOptions()
            opts.rootRank = i
            if i == self.rank:
                futures.append(pg.gather([outputs], input, opts).get_future())
            else:
                futures.append(pg.gather([], input, opts).get_future())

        # 等待工作完成
        expected = [fn(torch.tensor([rank])) for rank in range(self.world_size)]
        for i in range(self.world_size):
            futures[i].wait()
            result = futures[i].value()
            if i == self.rank:
                # 断言预期结果与实际结果相等
                self.assertEqual(expected, result)

    # 标记为需要 GLOO 支持的测试方法：gather 基础操作测试
    @requires_gloo()
    def test_gather_basics(self):
        # 调用内部方法进行 gather 基础操作测试
        self._test_gather_basics(lambda t: t.clone())

    # 根据 GPU 数量跳过测试方法的装饰器：gather 基础操作的 CUDA 版本测试
    @skip_if_lt_x_gpu(2)
    # 标记为需要 GLOO 支持的测试方法：gather 基础操作的 CUDA 版本测试
    @requires_gloo()
    def test_gather_basics_cuda(self):
        # 调用内部方法进行 gather 基础操作的 CUDA 版本测试
        self._test_gather_basics(lambda t: t.clone().cuda())

    # 标记为需要 GLOO 支持的测试方法：gather 非连续输入测试
    @requires_gloo()
    def test_gather_noncontiguous_input(self):
        # 创建 2D 张量的列，以确保内存不连续
        self._test_gather_basics(lambda t: t.expand(2, 2).contiguous()[:, 0])
    # 定义一个用于测试 gather 操作的方法，输入参数包括输入数据和处理函数
    def _test_gather_stress(self, inputs, fn):
        # 创建一个基于文件的存储对象，用于进程组通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用 Gloo 后端创建一个进程组对象
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        # 存储将来的异步操作的句柄
        future_handles = []
        # 生成输出数据列表，其中每个元素都是一个二维列表
        outputs = [
            [[fn(torch.tensor([-1])) for _ in range(self.world_size)]]
            for _ in range(len(inputs))
        ]
        # 生成预期输出数据列表，其中每个元素也是一个二维列表
        expected_outputs = [
            [[torch.tensor([i + j]) for j in range(self.world_size)]]
            for i in range(len(inputs))
        ]
        # 遍历输入数据列表
        for i in range(len(inputs)):
            # 对于每个根进程，配置 gather 的选项
            for root in range(self.world_size):
                opts = c10d.GatherOptions()
                opts.rootRank = root
                # 如果当前进程是根进程，则使用根进程的 gather 方法
                if root == self.rank:
                    fut = pg.gather(outputs[i], [fn(inputs[i])], opts).get_future()
                # 如果当前进程不是根进程，则使用普通的 gather 方法
                else:
                    fut = pg.gather([], [fn(inputs[i])], opts).get_future()
                # 将异步操作的句柄存储到列表中
                future_handles.append(fut)

        # 遍历所有的异步操作句柄
        for i, future_handle in enumerate(future_handles):
            # 等待当前异步操作完成
            future_handle.wait()
            # 计算当前迭代和根进程的索引
            iter = i // self.world_size
            root = i % self.world_size
            # 如果当前进程是根进程，获取 gather 的结果，并进行断言检查
            if root == self.rank:
                result = future_handle.value()
                self.assertEqual(
                    expected_outputs[iter],
                    [result],
                    msg=("Mismatch in iteration %d for root %d" % (iter, root)),
                )

    # 标记使用 Gloo 后端的测试方法，用于测试 gather 操作的性能
    @requires_gloo()
    def test_gather_stress(self):
        # 创建包含 1000 个张量的输入列表
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        # 调用 _test_gather_stress 方法进行测试，处理函数为克隆当前张量
        self._test_gather_stress(inputs, lambda t: t.clone())

    # 标记使用 Gloo 后端和 CUDA 的测试方法，用于测试 CUDA 加速的 gather 操作性能
    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_gather_stress_cuda(self):
        # 创建包含 1000 个 CUDA 张量的输入列表
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        # 调用 _test_gather_stress 方法进行测试，处理函数为克隆并使用 CUDA
        self._test_gather_stress(inputs, lambda t: t.clone().cuda())
    # 定义测试函数，用于检查 allgather 方法的各种边界条件和异常情况
    def test_allgather_checks(self):
        # 创建一个文件存储对象，用于进程组通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建一个基于Gloo后端的进程组对象
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # 创建三个不同形状和数据类型的零张量
        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        # 检测输入空列表时是否会引发 RuntimeError
        with self.assertRaisesRegex(
            RuntimeError, "requires non-empty input tensor list"
        ):
            pg.allgather([], [])

        # 检测输入输出列表长度不同是否会引发 RuntimeError
        with self.assertRaisesRegex(
            RuntimeError, "requires input/output tensor lists to have the same length"
        ):
            pg.allgather([], [t1])

        # 检测输入输出列表长度不同是否会引发 RuntimeError，并使用多重张量列表
        with self.assertRaisesRegex(
            RuntimeError, "requires input/output tensor lists to have the same length"
        ):
            pg.allgather([[t1] * self.world_size, [t1] * self.world_size], [t1])

        # 检测输出张量列表长度不合法是否会引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "invalid output tensor list"):
            pg.allgather([[t1] * (self.world_size - 1)], [t1])

        # 检测输出张量列表长度不合法是否会引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "invalid output tensor list"):
            pg.allgather([[t1] * (self.world_size + 1)], [t1])

        # 检测输入张量类型不合法是否会引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            pg.allgather(
                [[t1, t1] * (self.world_size), [t1, t1] * (self.world_size)], [t1, t2]
            )

        # 检测输入张量大小不合法是否会引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            pg.allgather(
                [[t1, t1] * (self.world_size), [t1, t1] * (self.world_size)], [t1, t3]
            )

        # 检测输入张量类型不合法是否会引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "invalid tensor type"):
            pg.allgather([([t1, t2] * (self.world_size))[: self.world_size]], [t1])

        # 检测输入张量大小不合法是否会引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "invalid tensor size"):
            pg.allgather([([t1, t3] * (self.world_size))[: self.world_size]], [t1])

    # 测试 allgather 方法的基本功能，通过 lambda 函数传入功能函数 fn
    def _test_allgather_basics(self, fn):
        # 创建一个文件存储对象，用于进程组通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建一个基于Gloo后端的进程组对象
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # 遍历不同的输入张量数目
        for n in [1, 2, 3]:
            # 生成输入张量列表
            input = [fn(torch.tensor([n * self.rank + i])) for i in range(n)]
            # 生成输出张量列表，用于接收结果
            output = [
                [fn(torch.tensor([-1])) for _ in range(n * self.world_size)]
                for _ in range(n)
            ]
            # 生成预期输出张量列表
            expected_output = [
                [fn(torch.tensor([i])) for i in range(n * self.world_size)]
                for _ in range(n)
            ]
            # 调用 allgather 方法异步获取结果
            fut = pg.allgather(output, input).get_future()
            fut.wait()
            result = fut.value()
            if n == 1:
                result = [result]
            # 断言预期输出与实际输出相等
            self.assertEqual(expected_output, result)

    # 装饰器函数，标记依赖Gloo后端的测试函数
    @requires_gloo()
    def test_allgather_basics(self):
        # 调用 _test_allgather_basics 函数，使用 lambda 函数将复制函数传入
        self._test_allgather_basics(lambda t: t.clone())
    def test_allgather_basics_cuda(self):
        self._test_allgather_basics(lambda t: t.clone().cuda())

    @requires_gloo()
    def test_allgather_noncontiguous_input(self):
        # 从二维张量中取一个列，使得内存不连续
        # 然后测试基本的allgather功能
        self._test_allgather_basics(lambda t: t.expand(2, 2).contiguous()[:, 0])

    def _test_allgather_stress(self, inputs, fn):
        # 创建一个基于文件的存储，并使用指定的world_size
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建一个基于Gloo的进程组，并设置相关的参数
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        future_handles = []
        outputs = [
            [[fn(torch.tensor([-1])) for _ in range(self.world_size)]]
            for _ in range(len(inputs))
        ]
        expected_outputs = [
            [[torch.tensor([i + j]) for j in range(self.world_size)]]
            for i in range(len(inputs))
        ]
        input_holder = {}
        for i in range(len(inputs)):
            # 注意，这是为了解决在https://github.com/pytorch/pytorch/issues/75529中讨论的数据竞争问题，
            # 但是当这个竞争问题被修复后，我们应该能够直接将列表传递给allgather。
            input_holder[i] = [fn(inputs[i])]
            fut = pg.allgather(outputs[i], input_holder[i]).get_future()
            future_handles.append(fut)

        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            result = future_handle.value()
            self.assertEqual(
                expected_outputs[i],
                [result],
                msg=("Mismatch in iteration %d" % i),
            )

    @requires_gloo()
    def test_allgather_stress(self):
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        # 对输入数据进行大规模的allgather压力测试
        self._test_allgather_stress(inputs, lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_allgather_stress_cuda(self):
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        # 在CUDA环境下，对输入数据进行大规模的allgather压力测试
        self._test_allgather_stress(inputs, lambda t: t.clone().cuda())

    @requires_gloo()
    # 定义测试方法，用于测试 allgather_coalesced 函数的各种边界情况
    def test_allgather_coalesced_checks(self):
        # 创建一个基于文件的存储，用于进程组
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用 Gloo 后端创建一个进程组
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )
        # 创建一个包含一个零张量的输入列表
        dummy_input = [torch.zeros([1], dtype=torch.float32)]
        # 创建一个包含多个空列表的输出列表，数量等于世界大小
        dummy_output_lists = [
            [torch.zeros([1], dtype=torch.float32)] for _ in range(self.world_size)
        ]

        # 测试场景：其中一个输出张量与输入列表不匹配
        dummy_output_lists[0] = [torch.zeros([0], dtype=torch.float32)]
        # 断言捕获 RuntimeError，并验证错误消息中包含特定文本
        with self.assertRaisesRegex(
            RuntimeError, "invalid size of output tensor at index 0"
        ):
            # 调用 all_gather_coalesced 函数
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)

        # 测试场景：其中一个输出张量的类型与输入列表不匹配
        dummy_output_lists[0] = [torch.zeros([1], dtype=torch.float64)]
        # 断言捕获 RuntimeError，验证错误消息中包含特定文本
        with self.assertRaisesRegex(RuntimeError, "invalid tensor type at index 0"):
            # 再次调用 all_gather_coalesced 函数
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)

        # 测试场景：输出列表包含比世界大小还多的元素
        dummy_output_lists = [
            [torch.zeros([1], dtype=torch.float32)] for _ in range(self.world_size + 1)
        ]
        # 断言捕获 RuntimeError，验证错误消息中包含特定文本
        with self.assertRaisesRegex(
            RuntimeError, "output lists should be equal to world size"
        ):
            # 第三次调用 all_gather_coalesced 函数
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)

        # 测试场景：输出不是一个列表的列表
        dummy_output_lists = [torch.zeros([0], dtype=torch.float32)]
        # 断言捕获 TypeError，验证错误消息中包含特定文本
        with self.assertRaisesRegex(
            TypeError, "Invalid function argument.*output_tensor_lists"
        ):
            # 最后一次调用 all_gather_coalesced 函数
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)

    # 标记当前测试函数依赖于 Gloo 后端
    @requires_gloo()
    # 定义一个异步测试函数，用于测试 allgather_coalesced 方法
    def test_allgather_coalesced_async(self):
        # 创建一个基于文件的存储对象，用于进程组通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用 gloo 后端
        c10d.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size, store=store
        )

        # 创建输入张量列表 xxs，包含两个列表，每个列表包含两个张量
        xxs = [2 * [torch.tensor([i + self.rank])] for i in range(2)]
        # 创建输出张量列表 yys，每个元素是一个二维列表，初始化为与输入张量相同形状的零张量
        yys = [
            [[torch.zeros_like(x) for x in xx] for _ in range(self.world_size)]
            for xx in xxs
        ]
        # 使用 all_gather_coalesced 方法进行异步操作，返回 Future 列表
        futs = [
            c10d.all_gather_coalesced(yy, xx, async_op=True) for xx, yy in zip(xxs, yys)
        ]

        # 期望的输出结果列表 zzs，包含两个元素，每个元素是一个二维列表，包含两个张量列表
        zzs = [
            [2 * [torch.tensor([i + r])] for r in range(self.world_size)]
            for i in range(2)
        ]

        # 等待所有 Future 完成
        torch.futures.wait_all(futs)
        # 对比 yys 和 zzs 的每个元素，确保它们相等
        for yy, zz in zip(yys, zzs):
            # 遍历每个输出张量列表
            for y_out, z_out in zip(yy, zz):
                # 遍历每个输出张量列表中的张量
                for y, z in zip(y_out, z_out):
                    # 断言每个张量 y 和 z 相等
                    self.assertEqual(y, z)

        # 为了解决 https://github.com/pytorch/pytorch/issues/65231 添加的说明
        # 在失败的测试中，所有 assertEqual 在所有进程上都通过了。
        # 然而，其中一个进程在退出程序之前没有调用 ProcessGroupGloo 析构函数。
        # 这并不奇怪，因为 Python 仅保证垃圾收集可能在程序退出之前发生。
        # 如果垃圾收集没有发生，ProcessGroup 中的两个线程可能在被销毁之前被析构。
        # FIXME: 仍然不清楚为什么只有这个测试需要显式调用 destroy_process_group()
        c10d.destroy_process_group()

    @requires_gloo()
    # 测试 reduce 方法的各种检查点
    def test_reduce_checks(self):
        # 创建一个基于文件的存储对象，用于进程组
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用自定义方法创建一个 gloo 进程组对象
        pg = pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # 创建一个形状为 [1] 的零张量 t1
        t1 = torch.zeros([1], dtype=torch.float32)

        # 测试 rootRank 为负数时是否引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.ReduceOptions()
            opts.rootRank = -1
            opts.rootTensor = 0
            pg.reduce([t1], opts)

        # 测试 rootRank 超出 world_size 范围时是否引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "invalid root rank"):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.world_size
            opts.rootTensor = 0
            pg.reduce([t1], opts)

        # 测试 rootTensor 超出范围时是否引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, "invalid root tensor"):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 1
            pg.reduce([t1], opts)

        # 测试传入多于一个张量的列表时是否引发 RuntimeError
        with self.assertRaisesRegex(
            RuntimeError, "requires a single-element tensor list"
        ):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.reduce([t1, t1], opts)
    # 定义一个测试函数，用于测试基本的 reduce 操作
    def _test_reduce_basics(self, fn):
        # 创建一个 FileStore 对象，用于进程间通信，指定文件名和进程数
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建一个基于 Gloo 后端的进程组对象，传入 FileStore、当前进程的 rank、总进程数和其他选项
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )
        # 对于简单的 reduce 测试，遍历测试集中的每一组操作、输入和输出
        for op, input, output in simple_reduce_tests(self.rank, self.world_size):
            # 遍历每个可能作为根节点的进程
            for root in range(self.world_size):
                # 创建 ReduceOptions 对象，并设置操作类型和根节点的 rank
                opts = c10d.ReduceOptions()
                opts.reduceOp = op
                opts.rootRank = root
                # 对输入数据应用指定的转换函数，得到临时变量 tmp
                tmp = fn(input)
                # 发起 reduce 操作，并获取 Future 对象
                fut = pg.reduce([tmp], opts).get_future()
                # 等待 reduce 操作完成
                fut.wait()
                # 获取 reduce 操作的结果
                result = fut.value()
                # 如果当前进程是根节点，则验证输出结果是否与预期一致
                if root == self.rank:
                    self.assertEqual(output, result[0])

    # 标记使用 Gloo 后端的 reduce 基本功能测试
    @requires_gloo()
    def test_reduce_basics(self):
        # 执行基本 reduce 测试，传入 lambda 函数用于对输入数据进行克隆操作
        self._test_reduce_basics(lambda t: t.clone())

    # 标记仅在有两个或更多 GPU 的情况下使用的 reduce 基本功能测试
    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_reduce_basics_cuda(self):
        # 执行基本 reduce 测试，传入 lambda 函数用于对输入数据进行克隆并移至 GPU 操作
        self._test_reduce_basics(lambda t: t.clone().cuda())

    # 定义一个测试函数，用于测试 reduce 操作的高负荷场景
    def _test_reduce_stress(self, inputs):
        # 创建一个 FileStore 对象，用于进程间通信，指定文件名和进程数
        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建一个基于 Gloo 后端的进程组对象，传入 FileStore、当前进程的 rank、总进程数和其他选项（使用多线程）
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        # 初始化用于存放 Future 对象的列表
        future_handles = []
        # 初始化用于存放输出数据的列表
        outputs = []
        # 遍历输入数据列表
        for i in range(len(inputs)):
            # 遍历每个可能作为根节点的进程
            for root in range(self.world_size):
                # 创建 ReduceOptions 对象，并设置根节点的 rank
                opts = c10d.ReduceOptions()
                opts.rootRank = root
                # 对输入数据进行克隆操作，得到临时变量 tmp
                tmp = inputs[i].clone()
                # 将克隆后的数据添加到输出列表中
                outputs.append(tmp)
                # 发起 reduce 操作，并获取 Future 对象
                fut = pg.reduce([tmp], opts).get_future()
                # 将 Future 对象添加到列表中
                future_handles.append(fut)

        # 遍历 Future 对象列表
        for i, future_handle in enumerate(future_handles):
            # 等待 Future 对象的操作完成
            future_handle.wait()
            # 获取 reduce 操作的结果
            result = future_handle.value()
            # 计算当前迭代和根节点的关联信息
            iter = i // self.world_size
            root = i % self.world_size
            # 如果当前进程是根节点，则验证输出结果是否与预期一致
            if root == self.rank:
                self.assertEqual(
                    torch.tensor(
                        [
                            (iter * self.world_size)
                            + (self.world_size * (self.world_size - 1) // 2)
                        ]
                    ),
                    result[0],
                    msg=("Mismatch in iteration %d with root rank %d" % (iter, root)),
                )

    # 标记使用 Gloo 后端的 reduce 高负荷场景测试
    @requires_gloo()
    def test_reduce_stress(self):
        # 创建输入数据列表，每个元素为包含当前进程 rank 的张量
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        # 执行高负荷场景的 reduce 测试
        self._test_reduce_stress(inputs)

    # 标记仅在有两个或更多 GPU 的情况下使用的 reduce 高负荷场景测试
    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_reduce_stress_cuda(self):
        # 创建输入数据列表，每个元素为包含当前进程 rank 的 GPU 张量
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        # 执行高负荷场景的 reduce 测试
        self._test_reduce_stress(inputs)

    # 标记使用 Gloo 后端的测试函数结束
    @requires_gloo()
    def test_send_recv_all_to_all(self):
        # 使用指定的文件名和进程数量创建一个 FileStore 对象
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用 Gloo 后端创建一个 ProcessGroup 对象
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # 预先分配输入/输出张量
        inputs = [torch.tensor([self.rank]) for _ in range(self.world_size)]
        outputs = [torch.tensor([-1]) for _ in range(self.world_size)]

        # 发送操作
        send_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            # 将输入张量发送给指定进程，并返回发送操作的句柄
            send_work.append(pg.send([inputs[i]], i, 0))

        # 接收操作
        recv_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            # 接收来自指定进程的输出张量，并返回接收操作的句柄
            recv_work.append(pg.recv([outputs[i]], i, 0))

        # 等待发送操作完成
        for work in send_work:
            work.wait()
            # 断言发送操作已完成
            self.assertTrue(work.is_completed())

        # 等待接收操作完成
        for work in recv_work:
            work.wait()
            # 断言接收操作已完成
            self.assertTrue(work.is_completed())

        # 测试除了自身外的每个输出张量是否包含对应的进程排名
        for i in range(self.world_size):
            if i == self.rank:
                continue
            # 断言输出张量的值与其对应进程的排名相等
            self.assertEqual(torch.tensor([i]), outputs[i])

    @requires_gloo()
    def test_barrier_implies_wait(self):
        # 使用指定的文件名和进程数量创建一个 FileStore 对象
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用 Gloo 后端创建一个 ProcessGroup 对象
        pg = self._create_process_group_gloo(
            store, self.rank, self.world_size, self.opts()
        )

        # 启动 allreduce 操作
        size = (100, 100)
        num = 16
        tensors = [torch.full(size, float(i)) for i in range(num)]
        for tensor in tensors:
            # 执行 allreduce 操作，并忽略返回的工作句柄（leak the returned work handle）
            pg.allreduce(tensor)

        # barrier 操作确保所有先前的工作都已完成
        pg.barrier().get_future().wait()

        for i, tensor in enumerate(tensors):
            # 断言每个张量的值与其进程排名乘以进程总数的乘积相等
            self.assertEqual(torch.full(size, float(i * self.world_size)), tensor)

    @skip_if_win32()
    @requires_gloo()
    def test_round_robin(self):
        num_process_groups = 2
        # 使用指定的文件名和进程数量创建一个 FileStore 对象
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用 Gloo 后端初始化进程组
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        # 使用循环轮询方式创建多个进程组
        pg = c10d._round_robin_process_groups(
            [c10d.new_group(pg_options=self.opts()) for i in range(num_process_groups)]
        )

        # 执行几个集体操作，以便每个进程组都被调用到
        for _ in range(num_process_groups + 1):
            tensor = torch.full([100, 100], float(self.rank))
            # 对指定张量执行广播操作，将根进程的数据广播到其他进程
            pg.broadcast(tensor, root=0).wait()
            # 断言张量的值全部为 0.0
            self.assertEqual(torch.full([100, 100], 0.0), tensor)
    # 定义一个测试方法，用于测试 Round Robin 策略的创建和销毁
    def test_round_robin_create_destroy(self):
        # 创建一个基于文件的存储对象，用于进程组的通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用 Gloo 后端，指定存储、进程排名和总进程数
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )

        # 定义一个创建进程组的内部函数
        def create(num, prefix):
            # 使用 Round Robin 策略创建指定数量的进程组，并返回创建的进程组列表
            return c10d._round_robin_process_groups(
                [c10d.new_group(pg_options=self.opts()) for i in range(num)]
            )

        # 两次执行以下操作：创建、使用和销毁
        for i in range(2):
            # 每次创建两个进程组
            num_process_groups = 2
            pg = create(num=num_process_groups, prefix=i)
            # 循环三次，每次执行全reduce操作并验证结果
            for _ in range(3):
                tensor = torch.ones([10, 10])
                pg.allreduce(tensor).wait()
                # 断言每个元素都是当前进程数的浮点数值
                self.assertEqual(torch.full([10, 10], float(self.world_size)), tensor)
            # 删除进程组对象
            del pg
class DistributedDataParallelTest(
    test_c10d_common.CommonDistributedDataParallelTest, MultiProcessTestCase
):
    # 设置测试的准备工作，调用父类的setUp方法和生成多进程的函数
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    # 获取进程组的方法
    def _get_process_group(self):
        # 获取存储对象
        store = self._get_store()
        # 使用 gloo 后端初始化进程组，传入存储、当前进程的排名和总进程数
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        # 返回默认的分布式组
        return c10d.distributed_c10d._get_default_group()

    # 测试 gloo 后端的方法
    def _test_gloo_backend(
        self, devices, device_ids, multi_device=False, gradient_as_bucket_view=False
    ):
        # 创建文件存储对象，用于进程组的初始化
        store = c10d.FileStore(self.file_name, self.world_size)
        # 使用 gloo 后端初始化进程组，传入存储、当前进程的排名和总进程数
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        # 获取默认的分布式组
        process_group = c10d.distributed_c10d._get_default_group()
        # 获取最后一个设备
        device = devices[-1]
        # 根据设备获取进程组的后端，并创建设备
        backend = process_group._get_backend(device)
        backend.create_device(interface=LOOPBACK)
        # 调用方法测试 DDP（分布式数据并行）与给定进程组
        self._test_ddp_with_process_group(
            process_group, devices, device_ids, multi_device, gradient_as_bucket_view
        )

    # 装饰器，要求使用 gloo 后端
    @requires_gloo()
    def test_gloo_backend_cpu_module(self):
        # 测试单 CPU 设备的 gloo 后端
        self._test_gloo_backend([torch.device("cpu")], None)

    # 装饰器，要求使用 gloo 后端
    @requires_gloo()
    def test_gloo_backend_cpu_module_grad_is_view(self):
        # 测试单 CPU 设备的 gloo 后端，且梯度作为视图
        self._test_gloo_backend(
            [torch.device("cpu")], None, gradient_as_bucket_view=True
        )

    # 装饰器，要求使用 gloo 后端，且要求至少有两个 GPU 设备
    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_gloo_backend_1gpu_module_device_ids_integer_list(self):
        # 获取当前进程可用的 GPU 数量，并选取一个 GPU 设备
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        # 测试使用 gloo 后端的单 GPU 设备模式
        self._test_gloo_backend(devices, int_devices)

    # 装饰器，要求使用 gloo 后端，且要求至少有两个 GPU 设备
    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_gloo_backend_1gpu_module_device_ids_torch_device_list(self):
        # 获取当前进程可用的 GPU 数量，并选取一个 GPU 设备
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        # 测试使用 gloo 后端的单 GPU 设备模式，传入 Torch 设备列表
        self._test_gloo_backend(devices, devices)

    # 装饰器，要求使用 gloo 后端，且要求至少有四个 GPU 设备
    @requires_gloo()
    @skip_if_lt_x_gpu(4)
    def test_gloo_backend_2gpu_module(self):
        # 获取当前进程可用的 GPU 数量，并选取两个 GPU 设备
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        # 测试使用 gloo 后端的多 GPU 设备模式
        self._test_gloo_backend(devices, None, multi_device=True)

    # 装饰器，要求使用 gloo 后端，且要求至少有八个 GPU 设备
    @requires_gloo()
    @skip_if_lt_x_gpu(8)
    def test_gloo_backend_4gpu_module(self):
        # 获取当前进程可用的 GPU 数量，并选取四个 GPU 设备
        int_devices = gpus_for_rank(self.world_size)[self.rank][:4]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        # 测试使用 gloo 后端的多 GPU 设备模式
        self._test_gloo_backend(devices, None, multi_device=True)

    # 测试全局局部未使用的参数梯度的方法
    def _test_global_local_unused_params_grad(
        self, gradient_as_bucket_view=False, static_graph=False
    ):
    ):
`
    ):
        """
        By simulating a multi-task training, this test is to make sure:
        1) DDP does not touch the grad of globally unused parameters.
        2) DDP does update the grad of locally unused parameters.
        """

        # 定义一个神经网络模块，包含多个任务，某些参数在不同任务中被使用
        class GlobalLocalUnusedParamModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.t0 = Task()  # 定义任务 t0
                self.t1 = Task()  # 定义任务 t1
                self.task_unused = Task()  # 定义未使用的任务参数 task_unused

            def task_parameters(self):
                # 返回任务的参数集合
                return (self.t0.p, self.t1.p, self.task_unused.p)

            def forward(self, x, rank):
                # 根据 rank 返回不同任务的输出
                return self.t0(x) if rank == 0 else self.t1(x)

        # 定义一个函数来运行并验证梯度
        def run_and_verify_grad(model):
            # 执行前向传播
            output = model(8, self.rank)

            # 此时所有参数的梯度应该为 None
            t0_p, t1_p, task_unused_p = model.module.task_parameters()
            self.assertIsNone(t0_p.grad)
            self.assertIsNone(t1_p.grad)
            self.assertIsNone(task_unused_p.grad)

            # 执行反向传播
            output.mean().backward()

            # 本地未使用的参数应该在所有进程中更新梯度，但全局未使用的参数应该仍然是 None
            self.assertIsNotNone(t0_p.grad)
            self.assertIsNotNone(t1_p.grad)
            self.assertIsNone(task_unused_p.grad)

        process_group = self._get_process_group()  # 获取进程组

        # 在 CPU 上测试
        cpu_model = DistributedDataParallel(
            GlobalLocalUnusedParamModule().cpu(),
            process_group=process_group,
            find_unused_parameters=True,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )
        run_and_verify_grad(cpu_model)  # 运行并验证 CPU 模型的梯度

        # 在 GPU 上测试
        device_id = gpus_for_rank(self.world_size)[self.rank][0]  # 获取当前进程的 GPU 设备 ID
        gpu_model = DistributedDataParallel(
            GlobalLocalUnusedParamModule().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            find_unused_parameters=True,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )
        run_and_verify_grad(gpu_model)  # 运行并验证 GPU 模型的梯度

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad(self):
        self._test_global_local_unused_params_grad()

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad_with_grad_is_view(self):
        self._test_global_local_unused_params_grad(gradient_as_bucket_view=True)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad_with_static_graph(self):
        self._test_global_local_unused_params_grad(static_graph=True)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_find_unused_parameters_when_unused_parameters_empty(self):
        """
        An empty unused_parameters array does not imply find_unused_parameters =
        false. This test makes sure that DDP allreduces unused parameters
        accordingly where the forward pass in some process uses all parameters.
        This unit test creates a module that uses all parameters in rank = 0, and
        has unused parameters in other ranks.
        """

        # 定义一个继承自 nn.Module 的模块类 FindUnusedParamModule
        class FindUnusedParamModule(nn.Module):
            def __init__(self):
                super().__init__()
                # 创建两个 Task 实例作为模块的成员变量
                self.t0 = Task()
                self.t1 = Task()

            # 返回模块中两个 Task 实例的参数
            def task_parameters(self):
                return (self.t0.p, self.t1.p)

            # 模块的前向传播函数，根据 rank 决定使用哪些 Task
            def forward(self, x, rank):
                return self.t1(self.t0(x)) if rank == 0 else self.t1(x)

        # 定义一个函数，用于运行模型并验证梯度
        def run_and_verify_grad(model):
            # 运行前向传播
            output = model(8, self.rank)

            # 确保所有参数的梯度在此时为 None
            [self.assertIsNone(t_p.grad) for t_p in model.module.task_parameters()]

            # 运行反向传播
            output.mean().backward()

            # 现在所有未使用的参数应在所有进程中更新梯度
            [self.assertIsNotNone(t_p.grad) for t_p in model.module.task_parameters()]

        # 获取进程组
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
    def test_ignored_output(self):
        """
        Test that the output of a model can be ignored and that there is no
        implicit requirement that `backward` gets called.
        """
        # 获取进程组
        process_group = self._get_process_group()

        # 定义一个忽略输出的模型类
        class IgnoredOutput(nn.Module):
            def __init__(self):
                super().__init__()
                # 第一层全连接层，输入维度为2，输出维度为10，无偏置
                self.fc1 = nn.Linear(2, 10, bias=False)
                # 第二层全连接层，输入维度为10，输出维度为4，无偏置
                self.fc2 = nn.Linear(10, 4, bias=False)
                # ReLU 激活函数
                self.relu = nn.ReLU()

            def forward(self, x):
                # 前向传播过程
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                # 对输出进行 softmax 处理
                return F.softmax(x, dim=1)

        # 创建分布式数据并行模型，使用 IgnoredOutput 类的实例作为模型
        model = DistributedDataParallel(
            IgnoredOutput().float(),
            process_group=process_group,
        )

        # 设置批量大小
        batch_size = 4
        # 定义交叉熵损失函数
        criterion = nn.CrossEntropyLoss()
        # 生成随机输入数据，形状为 [batch_size, 2]
        input = torch.rand([batch_size, 2], dtype=torch.float)
        # 随机生成目标标签数据，长度为 batch_size
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])

        # 运行几次迭代，忽略输出结果
        for _ in range(4):
            output = model(input)
            del output

        # 运行几次迭代，使用输出结果进行反向传播
        for _ in range(4):
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

    @requires_gloo()
    def test_ignored_output_with_unused_parameters(self):
        """
        Test that the output of a model can be ignored and that there is no
        implicit requirement that `backward` gets called, if not all model
        parameters participated in computing the model output.
        """
        # 获取进程组
        process_group = self._get_process_group()

        # 定义一个带有未使用参数的忽略输出的模型类
        class IgnoredOutputWithUnusedParameters(nn.Module):
            def __init__(self):
                super().__init__()
                # 第一层全连接层，输入维度为2，输出维度为10，无偏置
                self.fc1 = nn.Linear(2, 10, bias=False)
                # 第二层全连接层，输入维度为10，输出维度为4，无偏置
                self.fc2 = nn.Linear(10, 4, bias=False)
                # 第三层全连接层，输入维度为4，输出维度为4，无偏置
                self.fc3 = nn.Linear(4, 4, bias=False)
                # ReLU 激活函数
                self.relu = nn.ReLU()

            def forward(self, x):
                # 前向传播过程
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                # 对输出进行 softmax 处理
                return F.softmax(x, dim=1)

        # 创建分布式数据并行模型，使用 IgnoredOutputWithUnusedParameters 类的实例作为模型
        model = DistributedDataParallel(
            IgnoredOutputWithUnusedParameters().float(),
            process_group=process_group,
            find_unused_parameters=True,
        )

        # 设置批量大小
        batch_size = 4
        # 定义交叉熵损失函数
        criterion = nn.CrossEntropyLoss()
        # 生成随机输入数据，形状为 [batch_size, 2]
        input = torch.rand([batch_size, 2], dtype=torch.float)
        # 随机生成目标标签数据，长度为 batch_size
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])

        # 运行几次迭代，忽略输出结果
        for _ in range(4):
            output = model(input)
            del output

        # 运行几次迭代，使用输出结果进行反向传播
        for _ in range(4):
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_ignored_sharded_tensor(self):
        # 定义一个测试函数，用于测试在包含ShardedTensor时忽略该参数的分布式数据并行模型构建和运行

        class MyModule(nn.Module):
            # 定义一个自定义的PyTorch模块，包含一个ShardedTensor作为参数
            def __init__(self, shard_tensor: ShardedTensor) -> None:
                super().__init__()
                # 初始化一个线性层，输入维度为2，输出维度为10，无偏置
                self.fc1 = nn.Linear(2, 10, bias=False)
                # 将ShardedTensor作为模块的参数
                self.st = nn.Parameter(shard_tensor)
                # 定义ReLU激活函数
                self.relu = nn.ReLU()

            def forward(self, x):
                # 前向传播函数，先通过线性层和ReLU激活函数处理输入x
                x = self.relu(self.fc1(x))
                # 对处理后的结果进行softmax操作
                return F.softmax(x, dim=1)

        # 初始化进程组，使用gloo后端，指定初始化方法和进程组大小等参数
        pg = dist.init_process_group(
            "gloo",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )
        # 根据rank选择合适的设备
        device = torch.device(f"cuda:{self.rank}")
        # 定义本地ShardedTensor的元数据，包括分片偏移、分片大小和放置信息
        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank % 2) * 5, 0],
            shard_sizes=[5, 10],
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )
        # 生成本地ShardedTensor的实例
        local_shards = [Shard(torch.randn(5, 10, device=device), local_shard_metadata)]
        # 从本地Shards初始化全局ShardedTensor
        st = init_from_local_shards(local_shards, [10, 10])
        # 创建MyModule实例
        m = MyModule(st)
        # 忽略模型中名为"st"的参数和缓冲区，使其不参与分布式数据并行的处理
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            module=m, params_and_buffers_to_ignore={"st"}
        )
        # 测试确保在模块包含ShardedTensor时，分布式数据并行构造函数不会失败
        DistributedDataParallel(
            m,
            device_ids=[device] if device.type == "gpu" else None,
            process_group=pg,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
            static_graph=True,
        )

    def _run_and_verify_sparse_gradients(self, vanilla_model, ddp_model):
        # 定义一个函数，用于运行和验证稀疏梯度

        mult = 2
        batch_size = mult * self.world_size
        criterion = nn.CrossEntropyLoss()
        input = torch.randint(0, 10, [batch_size, 2])
        target = torch.randint(0, 10, [batch_size])

        # 使用完整批次数据对单进程版本模型进行训练和反向传播
        criterion(vanilla_model(input), target).backward()

        # 使用部分批次数据对多进程版本模型进行训练和反向传播
        partial_input = input.split(mult)[self.rank]
        partial_target = target.split(mult)[self.rank]
        criterion(ddp_model(partial_input), partial_target).backward()

        # 检查梯度是否稀疏且相同
        vanilla_parameter = next(vanilla_model.parameters())
        ddp_parameter = next(ddp_model.parameters())
        self.assertEqual(
            vanilla_parameter.grad.coalesce(), ddp_parameter.grad.coalesce()
        )

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def _test_sparse_gradients(self, gradient_as_bucket_view=False):
        # 获取当前进程组
        process_group = self._get_process_group()

        # 确保跨进程间初始化的权重和输入是一致的
        torch.manual_seed(1337)

        # 创建原始的稀疏梯度模型
        vanilla_model = SparseGradientModule()

        # 使用分布式数据并行包装深拷贝的原始模型
        ddp_model = DistributedDataParallel(
            copy.deepcopy(vanilla_model),
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # 运行测试并验证稀疏梯度
        self._run_and_verify_sparse_gradients(vanilla_model, ddp_model)

    @requires_gloo()
    def test_sparse_gradients(self):
        # 调用 _test_sparse_gradients 函数进行测试
        self._test_sparse_gradients()

    @requires_gloo()
    def test_sparse_gradients_grad_is_view(self):
        # 使用 gradient_as_bucket_view=True 调用 _test_sparse_gradients 进行测试
        self._test_sparse_gradients(gradient_as_bucket_view=True)

    @requires_gloo()
    def test_ddp_comm_hook_future_passing_cpu(self):
        """
        This unit test verifies whether the Future object is passed properly.
        The callback function creates a Future object and sets a value to it.
        """
        # 创建文件存储，用于进程组
        store = c10d.FileStore(self.file_name, self.world_size)
        # 获取当前进程组
        process_group = self._get_process_group()

        # 在 CPU 上进行测试
        cpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().cpu(), process_group=process_group
        )

        # 注册 DDP 通信钩子
        cpu_model.register_comm_hook(None, self._simple_hook)

        # 检查梯度是否等于回调函数返回的值
        # 如果没有 comm_hook，结果将是 0.25 * torch.ones(2, 2)。
        self._run_and_verify_hook(cpu_model, 8, 2 * torch.ones(2, 2))

    def _gpu_model_with_ddp_comm_hook(
        self, process_group, hook=None, gradient_as_bucket_view=False, state=None
    ):
        # 获取当前设备 ID
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        # 在指定 GPU 设备上创建分布式数据并行模型
        gpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # 如果存在钩子，则注册 DDP 通信钩子
        if hook is not None:
            gpu_model.register_comm_hook(state, hook)

        return gpu_model

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_future_passing_gpu_gloo(self):
        """
        This unit test verifies whether the Future object is passed properly using gloo backend.
        The hook callback function creates a Future object and sets a value to it.
        """
        # 获取当前进程组
        process_group = self._get_process_group()

        # 获取带有 _simple_hook 注册的 GPU 模型
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, self._simple_hook)

        # 检查梯度是否等于 simple_hook 的回调函数返回的值
        # 如果没有 comm_hook，结果将是 0.25 * torch.ones(2, 2)。
        self._run_and_verify_hook(gpu_model, 8, 2 * torch.ones(2, 2))

    @requires_gloo()
    def test_ddp_invalid_comm_hook_init(self):
        """
        This unit test makes sure that register_comm_hook properly checks the format
        of hook defined by user. The Python hook must be callable. This test also
        checks whether bucket annotation checked properly if defined.
        """
        # 获取当前进程组
        process_group = self._get_process_group()

        # 创建一个 DistributedDataParallel 模型，使用自定义的通信钩子模块和进程组
        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        # 断言：注册通信钩子时，若钩子不可调用，抛出 TypeError 异常
        with self.assertRaisesRegex(TypeError, "Communication hook must be callable."):
            model.register_comm_hook(state=None, hook=1)

        # 断言：注册通信钩子时，若未正确指定 bucket 参数的类型为 dist.GradBucket，抛出 ValueError 异常
        with self.assertRaisesRegex(
            ValueError, "bucket annotation should be dist.GradBucket."
        ):

            def comm_hook(
                state: object, bucket: int
            ) -> torch.futures.Future[torch.Tensor]:
                return torch.futures.Future()

            model.register_comm_hook(state=None, hook=comm_hook)

    @requires_gloo()
    def test_ddp_invalid_comm_hook_return_type(self):
        """
        This test checks whether return annotation checked properly if defined. It also
        checks whether an internal error is thrown if return type is incorrect and user
        hasn't specified any return type annotation.
        """
        # 获取当前进程组
        process_group = self._get_process_group()

        # 创建一个 DistributedDataParallel 模型，使用自定义的通信钩子模块和进程组
        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        # 预期的错误消息
        expected_err = (
            "Communication hook: return annotation should be torch.futures.Future"
        )

        # 断言：注册通信钩子时，若返回类型注解错误，抛出 ValueError 异常
        with self.assertRaisesRegex(
            ValueError,
            expected_err,
        ):

            def comm_hook(state: object, bucket: dist.GradBucket) -> int:
                return torch.futures.Future()

            model.register_comm_hook(state=None, hook=comm_hook)

        # 验证是否记录了预期的 DDP 错误日志
        verify_ddp_error_logged(model, expected_err)

        # 断言：注册通信钩子时，若回调函数返回类型不是 torch.futures.Future 对象，抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError,
            "callback must return a torch.futures.Future object, but got",
        ):

            def comm_hook(state: object, bucket: dist.GradBucket):
                return 1

            # 注册通信钩子
            model.register_comm_hook(state=None, hook=comm_hook)

            # 运行前向传播
            output = model(8, self.rank)

            # 运行反向传播
            output.mean().backward()

    @requires_gloo()
    def test_ddp_comm_hook_register_just_once(self):
        """
        DDP通信钩子只能注册一次。此测试验证当多次调用register_comm_hook时是否正确抛出错误。
        """
        # 获取进程组
        process_group = self._get_process_group()

        # 创建一个用于DDP的模型，并指定进程组
        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        # 定义一个虚拟的通信钩子函数
        def dummy_hook(state, bucket):
            fut = torch.futures.Future()
            fut.set_result([bucket.buffer()])
            return fut

        # 第一次注册通信钩子函数
        model.register_comm_hook(None, dummy_hook)

        # 使用assertRaisesRegex检查是否正确抛出异常
        with self.assertRaisesRegex(
            RuntimeError,
            "register_comm_hook or register_builtin_comm_hook can only be called once.",
        ):
            # 第二次尝试注册通信钩子函数，预期会抛出异常
            model.register_comm_hook(None, dummy_hook)

    @requires_gloo()
    def test_ddp_comm_hook_sparse_gradients(self):
        """
        使用DDP通信钩子运行“test_sparse_gradients”单元测试。
        我们定义一个简单的钩子函数来进行全局归约，并在此测试中使用gloo后端。
        """
        # 获取进程组
        process_group = self._get_process_group()

        # 确保在所有进程中初始化权重和输入是相同的
        torch.manual_seed(1337)

        # 创建一个原始的模型实例
        vanilla_model = SparseGradientModule()

        # 使用DDP包装模型，指定进程组
        ddp_model = DistributedDataParallel(
            copy.deepcopy(vanilla_model),
            process_group=process_group,
        )

        # 定义一个使用gloo后端的全局归约钩子函数
        def allreduce_hook_gloo(
            state: object, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:
            # 定义一个函数，用于将结果除以2 * world_size
            def div_by_world_size(fut):
                return fut.wait()[0] / self.world_size

            # 使用进程组进行全局归约，返回一个Future对象
            fut = process_group.allreduce([bucket.buffer()]).get_future()
            # 将全局归约的结果传递给除法函数进行处理
            return fut.then(div_by_world_size)

        # 注册全局归约钩子函数到DDP模型中
        ddp_model.register_comm_hook(None, allreduce_hook_gloo)

        # 运行并验证稀疏梯度
        self._run_and_verify_sparse_gradients(vanilla_model, ddp_model)
class ReducerModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义第一个全连接层，输入维度为2，输出维度为10，无偏置
        self.fc1 = nn.Linear(2, 10, bias=False)
        # 定义第二个全连接层，输入维度为10，输出维度为4，无偏置
        self.fc2 = nn.Linear(10, 4, bias=False)
        # 定义第三个全连接层，输入维度为4，输出维度为4，无偏置
        self.fc3 = nn.Linear(4, 4, bias=False)
        # 定义激活函数ReLU
        self.relu = nn.ReLU()

    def forward(self, x, use_fc3=True):
        # 使用ReLU作为激活函数，先对输入x进行fc1层前向计算，然后应用ReLU激活
        x = self.relu(self.fc1(x)).float()
        # 继续使用ReLU作为激活函数，对前一层结果进行fc2层前向计算，再应用ReLU激活
        x = self.relu(self.fc2(x)).float()
        # 如果use_fc3为True，则对前一层结果进行fc3层前向计算，否则跳过
        if use_fc3:
            x = self.fc3(x).float()
        # 对输出结果进行softmax归一化，dim=1表示在第一个维度上进行归一化
        return F.softmax(x, dim=1)


class ReducerTest(TestCase):
    def setUp(self):
        # 创建临时文件以存储分布式进程组的状态
        self.file = tempfile.NamedTemporaryFile(delete=False)
        world_size = 1
        # 创建文件存储对象，用于分布式进程组的初始化
        self.store = c10d.FileStore(self.file.name, world_size)
        # 使用gloo后端初始化分布式进程组，rank为0表示当前进程组中的排名，world_size为进程组中的总进程数
        c10d.init_process_group(
            backend="gloo", store=self.store, rank=0, world_size=world_size
        )
        # 获取默认的分布式进程组
        self.process_group = c10d.distributed_c10d._get_default_group()

    def tearDown(self):
        # 销毁分布式进程组
        c10d.destroy_process_group()
        try:
            # 尝试删除临时文件
            os.remove(self.file.name)
        except OSError as e:
            print(str(e))
            pass

    @requires_gloo()
    def test_single_dtype_single_bucket(self):
        # 创建ReducerModule的实例model
        model = ReducerModule()
        # 获取模型参数列表
        parameters = list(model.parameters())
        # 创建包含所有参数索引的单个桶
        buckets = [list(range(len(parameters)))]
        # 使用分布式Reducer处理参数，指定桶大小和进程组
        dist.Reducer(
            parameters, buckets, [dist._DEFAULT_FIRST_BUCKET_BYTES], self.process_group
        )

    def _create_mixed_precision_model(self):
        # 创建混合精度模型ReducerModule的实例
        model = ReducerModule()
        # 将模型转换为float类型
        model.float()
        # 将fc1层参数转换为double类型
        model.fc1.double()
        return model

    @requires_gloo()
    def test_multi_dtype_single_bucket(self):
        # 创建混合精度模型
        model = self._create_mixed_precision_model()

        # 如果一个桶中存在多种数据类型，则抛出RuntimeError
        with self.assertRaises(RuntimeError):
            # 获取模型参数列表
            parameters = list(model.parameters())
            # 创建包含所有参数索引的单个桶
            buckets = [list(range(len(parameters)))]
            # 使用分布式Reducer处理参数，指定桶大小和进程组
            dist.Reducer(
                parameters,
                buckets,
                [dist._DEFAULT_FIRST_BUCKET_BYTES],
                self.process_group,
            )

    @requires_gloo()
    def test_multi_dtype_multi_bucket(self):
        # 创建混合精度模型
        model = self._create_mixed_precision_model()
        # 获取模型参数列表
        parameters = list(model.parameters())
        # 根据数据类型对参数索引进行分组
        group_by_dtype = groupby(
            range(len(parameters)), key=lambda i: parameters[i].dtype
        )
        # 创建多个桶，每个桶包含相同数据类型的参数索引
        buckets = [list(indices) for _, indices in group_by_dtype]
        # 使用分布式Reducer处理参数，指定每个桶的大小和进程组
        dist.Reducer(
            parameters,
            buckets,
            [dist._DEFAULT_FIRST_BUCKET_BYTES for _ in buckets],
            self.process_group,
        )
    # 为给定的模型列表创建一个分布式数据并行处理的Reducer对象
    def _create_reducer_for_models(self, models, find_unused_parameters=False):
        # 断言模型列表长度为1，确保只有一个模型
        self.assertEqual(len(models), 1)
        # 获取模型的所有参数
        parameters = list(models[0].parameters())
        # 根据参数的数据类型分组，返回一个迭代器，每个元素是一组参数索引
        group_by_dtype = groupby(
            range(len(parameters)), key=lambda i: parameters[i].dtype
        )
        # 将分组后的索引列表存入buckets中
        buckets = [list(indices) for _, indices in group_by_dtype]
        # 创建并返回一个Reducer对象，用于分布式数据并行处理
        return dist.Reducer(
            parameters,
            buckets,
            [dist._DEFAULT_FIRST_BUCKET_BYTES for _ in range(len(buckets))],
            self.process_group,
            find_unused_parameters=find_unused_parameters,
        )

    # 测试模型前向和反向传播
    @requires_gloo()
    def test_forward_backward(self):
        batch_size = 10
        # 创建混合精度模型
        model = self._create_mixed_precision_model()
        # 使用模型创建一个Reducer对象
        reducer = self._create_reducer_for_models([model])
        # 准备进行前向传播
        reducer.prepare_for_forward()
        # 定义损失函数
        loss = nn.CrossEntropyLoss()
        # 创建随机输入数据
        input = torch.rand([batch_size, 2], dtype=torch.double)
        # 创建随机目标数据
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
        # 执行模型前向传播计算损失
        output = loss(model(input), target)
        # 准备进行反向传播
        reducer.prepare_for_backward(output)
        # 执行反向传播
        output.backward()

    # 测试在存在未使用参数时的模型前向和反向传播
    @requires_gloo()
    def test_forward_backward_unused_parameters(self):
        batch_size = 10
        # 创建混合精度模型
        model = self._create_mixed_precision_model()
        # 使用模型创建一个Reducer对象，并设置find_unused_parameters=True
        reducer = self._create_reducer_for_models([model], find_unused_parameters=True)
        # 准备进行前向传播
        reducer.prepare_for_forward()
        # 定义损失函数
        loss = nn.CrossEntropyLoss()
        # 创建随机输入数据
        input = torch.rand([batch_size, 2], dtype=torch.double)
        # 创建随机目标数据
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
        # 执行模型前向传播计算损失，注意传入use_fc3=False
        output = loss(model(input, use_fc3=False), target)

        # 检查fc3的梯度是否为None
        self.assertEqual(None, model.fc3.weight.grad)

        # 准备进行反向传播
        reducer.prepare_for_backward(output)
        # 执行反向传播
        output.backward()

        # 由于fc3.weight在output的自动求导图中未出现，Reducer会标记fc3的梯度为ready
        # 但由于fc3.weight被全局认为是未使用的参数，它的梯度依然保持为None
        self.assertEqual(None, model.fc3.weight.grad)

    # 继续编写下一个测试函数之前的部分...
    # 定义一个测试函数，用于测试前向和后向优化器的功能
    def test_forward_backward_optimizer(self):
        # 定义批处理大小
        batch_size = 10
        # 创建混合精度模型
        model = self._create_mixed_precision_model()
        # 创建用于模型的参数减少器，指定了在查找未使用的参数时执行的操作
        reducer = self._create_reducer_for_models([model], find_unused_parameters=True)
        # 准备模型进行前向计算
        reducer.prepare_for_forward()
        # 定义损失函数为交叉熵损失
        loss = nn.CrossEntropyLoss()
        # 定义优化器为Adam优化器，优化模型的参数
        optimizer = torch.optim.Adam(model.parameters())

        # 循环执行三次训练迭代
        for i in range(3):
            # 生成随机输入数据张量，形状为[batch_size, 2]，数据类型为双精度浮点数
            input = torch.rand([batch_size, 2], dtype=torch.double)
            # 生成随机目标标签张量，数据类型为长整型，标签值范围为[0, 3]
            target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])

            # 调用优化器的zero_grad方法，将模型参数的梯度置零
            # 这里的操作确保在每次迭代之前梯度都被正确处理，避免梯度累积
            optimizer.zero_grad()

            # 用模型进行前向计算，计算输出与目标标签的交叉熵损失
            # 如果i大于0，则使用模型的额外参数use_fc3
            output = loss(model(input, use_fc3=(i > 0)), target)

            # 准备模型进行反向传播，传入输出值用于计算梯度
            reducer.prepare_for_backward(output)

            # 执行反向传播，计算模型参数的梯度
            output.backward()

            # 根据计算得到的梯度更新模型参数
            optimizer.step()
class CommTest(test_c10d_common.AbstractCommTest, MultiProcessTestCase):
    @property
    def device(self):
        # 返回设备为 CPU
        return "cpu"

    def setUp(self):
        # 调用父类的 setUp 方法初始化测试环境
        super().setUp()
        # 启动多进程
        self._spawn_processes()

    def tearDown(self):
        # 调用父类的 tearDown 方法清理测试环境
        super().tearDown()
        try:
            # 尝试删除测试过程中生成的文件
            os.remove(self.file_name)
        except OSError:
            # 如果文件不存在则捕获 OSError 异常
            pass

    def _test_broadcast_coalesced(self, process_group, device, root_rank):
        half = torch.float16

        # 如果设备为 CPU，则将 half 设置为 torch.float32，因为 CPU 不支持 torch.float16
        if device == torch.device("cpu"):
            half = torch.float32

        # 创建一个目标张量 target，其中包含多个分块，每个分块都是 torch.arange() 的结果
        target = torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float64, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)

        # 如果当前进程的 rank 等于 root_rank，则将 tensors 设置为 target 的克隆
        # 否则将 tensors 设置为与 target 维度相同的零张量
        if self.rank == root_rank:
            tensors = [tensor.clone() for tensor in target]
        else:
            tensors = [torch.zeros_like(tensor) for tensor in target]

        # 如果当前进程的 rank 不等于 root_rank，则断言 tensors 与 target 不相等
        if self.rank != root_rank:
            self.assertNotEqual(tensors, target)

        # 调用 c10d._broadcast_coalesced 方法进行广播
        c10d._broadcast_coalesced(
            process_group, tensors, buffer_size=256, src=root_rank
        )

        # 如果当前进程的 rank 不等于 root_rank，则断言 tensors 与 target 相等
        if self.rank != root_rank:
            self.assertEqual(tensors, target)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_broadcast_coalesced_gloo_cuda(self):
        # 创建一个文件存储对象，用于进程组通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化分布式进程组，使用 gloo 后端，传入文件存储、当前进程的 rank 和总进程数
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        # 获取默认的进程组对象
        process_group = c10d.distributed_c10d._get_default_group()
        # 设置设备为当前进程对应的 CUDA 设备
        device = torch.device("cuda:%d" % self.rank)
        # 获取当前设备对应的后端对象
        backend = process_group._get_backend(device)
        # 创建一个回环接口设备
        backend.create_device(interface=LOOPBACK)
        # 创建一个包含所有进程 rank 的列表
        ranks = list(range(self.world_size))
        # 对每个 root_rank 调用 _test_broadcast_coalesced 方法
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @requires_gloo()
    def test_broadcast_coalesced_gloo_cpu(self):
        # 创建一个文件存储对象，用于进程组通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化分布式进程组，使用 gloo 后端，传入文件存储、当前进程的 rank 和总进程数
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        # 获取默认的进程组对象
        process_group = c10d.distributed_c10d._get_default_group()
        # 设置设备为 CPU
        device = torch.device("cpu")
        # 获取当前设备对应的后端对象
        backend = process_group._get_backend(device)
        # 创建一个回环接口设备
        backend.create_device(interface=LOOPBACK)
        # 创建一个包含所有进程 rank 的列表
        ranks = list(range(self.world_size))
        # 对每个 root_rank 调用 _test_broadcast_coalesced 方法
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    # 测试设置序列号默认值使用pg_gloo后端
    def test_sequence_num_set_default_pg_gloo():
        self._test_sequence_num_set_default_pg(backend="gloo")
    
    # 装饰器：要求使用gloo后端
    # 装饰器：如果GPU数小于2则跳过测试
    # 测试设置序列号在gloo新群组中
    def test_sequence_num_set_gloo_new_group():
        self._test_sequence_num_set_new_group(backend="gloo")
    
    # 装饰器：如果GPU数小于2则跳过测试
    # 装饰器：要求使用gloo后端
    # 测试序列号在gloo默认组中递增
    def test_sequence_num_incremented_gloo_default():
        self._test_sequence_num_incremented_default_group("gloo")
    
    # 装饰器：如果GPU数小于4则跳过测试
    # 装饰器：要求使用gloo后端
    # 测试序列号在gloo子群组中递增
    def test_sequence_num_incremented_gloo_subgroup():
        if self.world_size < 4:
            return skip_but_pass_in_sandcastle("Test requires world_size of at least 4")
        self._test_sequence_num_incremented_subgroup("gloo")
    
    # 装饰器：如果GPU数小于2则跳过测试
    # 装饰器：要求使用gloo后端
    # 测试警告在非群组中使用gloo后端时发出
    def test_gloo_warn_not_in_group():
        self._test_warn_not_in_group(backend="gloo")
    
    # 装饰器：如果GPU数小于2则跳过测试
    # 装饰器：要求使用gloo后端
    # 测试gloo中的排名成员资格
    def test_gloo_rank_membership():
        self._test_rank_membership(backend="gloo")
    
    # 装饰器：如果GPU数小于2则跳过测试
    # 装饰器：要求使用gloo后端
    # 测试张量数据类型不匹配情况下的行为
    def test_tensor_dtype_mismatch():
        self._test_tensor_dtype_mismatch(backend="gloo")
    
    # 装饰器：如果GPU数小于2则跳过测试
    # 装饰器：要求使用gloo后端
    # 测试复杂数据类型的张量行为
    def test_tensor_dtype_complex():
        self._test_tensor_dtype_complex(backend="gloo")
    
    # 装饰器：要求使用gloo后端
    # 测试布尔张量的行为
    def test_bool_tensors():
        self._test_bool_tensors(backend="gloo")
# 定义一个测试类，继承自test_c10d_common.ProcessGroupWithDispatchedCollectivesTests
class GlooProcessGroupWithDispatchedCollectivesTests(
    test_c10d_common.ProcessGroupWithDispatchedCollectivesTests
):

    # 测试集合通信功能
    @requires_gloo()
    def test_collectives(self):
        # 使用"gloo"后端测试集合通信功能
        self._test_collectives(backend="gloo")

    # 测试集合通信中的全局归约操作
    @requires_gloo()
    def test_allreduce_coalesced(self):
        # 使用"gloo"后端测试全局归约操作
        self._test_allreduce_coalesced(backend="gloo")

    # 测试集合通信中的全局单对单交换操作
    @requires_gloo()
    def test_all_to_all_single(self):
        # 使用"gloo"后端测试全局单对单交换操作
        self._test_all_to_all_single(backend="gloo")

    # 测试集合通信中的全局聚集操作
    @requires_gloo()
    def test_allgather_coalesced(self):
        # 创建一个基于文件的存储对象
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用"gloo"后端
        dist.init_process_group(
            "gloo",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 创建一个输入张量，全为1.0，形状为10x10，数据类型为32位浮点数
        input_tensor = torch.ones(10, 10, dtype=torch.float32)
        # 创建一个输出张量列表，包含一个与输入张量形状相同的全0张量
        output_tensor_list = [torch.zeros_like(input_tensor)]
        # 执行全聚集操作
        dist.all_gather_coalesced([output_tensor_list], [input_tensor])
        # 断言输出张量列表与输入张量相同
        self.assertEqual(output_tensor_list, [input_tensor])

    # 测试监视型屏障操作
    @requires_gloo()
    def test_monitored_barrier(self):
        # 创建一个基于文件的存储对象
        store = dist.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用"gloo"后端
        dist.init_process_group(
            "gloo",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # 执行监视型屏障操作
        dist.monitored_barrier()


# 定义一个编译器测试类，继承自test_c10d_common.CompilerTest
class CompilerTest(test_c10d_common.CompilerTest):

    # 返回世界大小为2
    @property
    def world_size(self):
        return 2

    # 获取默认进程组
    def _get_default_group(self):
        # 创建一个基于文件的存储对象
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化进程组，使用"gloo"后端
        dist.init_process_group(
            backend="gloo",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        # 返回默认的分布式C10D进程组
        return dist.distributed_c10d._get_default_group()

    # 测试全局归约操作（CPU等待）
    def test_allreduce_work_wait_cpu(self):
        self._test_allreduce_work_wait(torch.ones(2, 2) * self.rank)

    # 如果GPU数目小于2，跳过测试全局归约操作（GPU等待）
    @skip_if_lt_x_gpu(2)
    def test_allreduce_work_wait_gpu(self):
        self._test_allreduce_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    # 测试全局聚集操作（CPU等待）
    def test_allgather_work_wait_cpu(self):
        self._test_allgather_work_wait(torch.ones(2, 2) * self.rank)

    # 如果GPU数目小于2，跳过测试全局聚集操作（GPU等待）
    @skip_if_lt_x_gpu(2)
    def test_allgather_work_wait_gpu(self):
        self._test_allgather_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    # 测试全局广播操作（CPU等待）
    def test_broadcast_work_wait_cpu(self):
        self._test_broadcast_work_wait(torch.ones(2, 2) * self.rank)

    # 如果GPU数目小于2，跳过测试全局广播操作（GPU等待）
    @skip_if_lt_x_gpu(2)
    def test_broadcast_work_wait_gpu(self):
        self._test_broadcast_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    # 测试全局分发操作（CPU等待）
    def test_scatter_work_wait_cpu(self):
        self._test_scatter_work_wait(torch.ones(2, 2) * self.rank)

    # 如果GPU数目小于2，跳过测试全局分发操作（GPU等待）
    @skip_if_lt_x_gpu(2)
    def test_scatter_work_wait_gpu(self):
        self._test_scatter_work_wait(torch.ones(2, 2, device=self.rank) * self.rank)

    # 测试嵌套通信张量封装操作
    def test_nested_comm_tensor_wrapping(self):
        self._test_nested_comm_tensor_wrapping(torch.ones(2, 2) * self.rank)
    # 定义一个测试方法，用于测试连续的通信、计算、等待操作在 CPU 上的表现
    def test_consecutive_comm_work_wait_cpu(self):
        # 调用内部方法 _test_consecutive_comm_work_wait，传入全为当前进程排名的 2x2 的张量
        self._test_consecutive_comm_work_wait(torch.ones(2, 2) * self.rank)
    
    # 使用装饰器 @skip_if_lt_x_gpu(2)，如果 GPU 数量小于 2，则跳过该测试方法
    def test_consecutive_comm_work_wait_gpu(self):
        # 调用内部方法 _test_consecutive_comm_work_wait，传入全为当前进程排名的 2x2 的张量，并指定设备为当前进程排名
        self._test_consecutive_comm_work_wait(
            torch.ones(2, 2, device=self.rank) * self.rank
        )
# 定义一个测试类，继承自 AbstractLargeCommTest 和 MultiProcessTestCase
class LargeCommTest(test_c10d_common.AbstractLargeCommTest, MultiProcessTestCase):
    
    # 在每个测试方法执行前调用，设置测试环境
    def setUp(self):
        super().setUp()
        # 启动多进程
        self._spawn_processes()

    # 在每个测试方法执行后调用，清理测试环境
    def tearDown(self):
        super().tearDown()
        try:
            # 尝试删除指定的文件
            os.remove(self.file_name)
        except OSError:
            # 如果文件不存在，捕获异常并忽略
            pass

    # 设定一个属性，返回 CPU 设备
    @property
    def device(self):
        return torch.device("cpu")

    # 测试方法修饰器，要求使用 gloo 后端
    @requires_gloo()
    def test_new_group_local_sync(self):
        # 调用内部方法进行本地同步测试，指定使用 gloo 后端
        self._test_new_group_local_sync(backend="gloo")

    # 测试方法修饰器，要求使用 gloo 后端
    @requires_gloo()
    def test_new_group_local_sync_sanity_check(self):
        # 调用内部方法进行本地同步的健全性检查，指定使用 gloo 后端
        self._test_new_group_local_sync_sanity_check(backend="gloo")

    # 测试方法修饰器，要求使用 gloo 后端
    @requires_gloo()
    def test_new_group_local_sync_duplicate_pg(self):
        # 调用内部方法进行本地同步的重复进程组测试，指定使用 gloo 后端
        self._test_new_group_local_sync_duplicate_pg(backend="gloo")

# 如果脚本作为主程序执行
if __name__ == "__main__":
    # 断言：确保在主进程上 CUDA 上下文未被初始化
    assert not torch.cuda._initialized, "test_distributed must not have initialized CUDA context on main process"

    # 运行测试
    run_tests()
```