# `.\pytorch\test\distributed\test_c10d_spawn.py`

```
# Owner(s): ["oncall: distributed"]

# 导入标准库和第三方库
import os
import sys
import tempfile

# 导入 PyTorch 相关模块
import torch
import torch.distributed as c10d
import torch.multiprocessing as mp
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import load_tests, NO_MULTIPROCESSING_SPAWN

# Torch distributed.nn 在 Windows 下不可用，检查导入错误
_torch_dist_nn_available = True
try:
    import torch.distributed.nn
except ImportError:
    _torch_dist_nn_available = False

# load_tests 函数用于在 sandcastle 上自动过滤分片测试，此行消除 flake 警告
load_tests = load_tests

# 如果 c10d 不可用，则跳过测试并输出消息到标准错误流
if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果 NO_MULTIPROCESSING_SPAWN 为真，则跳过测试并输出消息到标准错误流
if NO_MULTIPROCESSING_SPAWN:
    print("spawn not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 定义一个抽象类 AbstractProcessGroupShareTensorTest
class AbstractProcessGroupShareTensorTest:
    world_size = 2

    # _test_multiprocess 方法用于执行多进程测试
    def _test_multiprocess(self, f, shared_tensors, init_pg, n_output):
        ws = self.world_size
        # 创建一个临时文件对象，文件在删除时会被自动清除
        file = tempfile.NamedTemporaryFile(delete=False)
        # 获取 "spawn" 上下文
        ctx = mp.get_context("spawn")
        # 创建两个进程间通信的队列
        c2p = ctx.Queue(2)
        p2c = ctx.Queue(2)
        ps = []
        # 启动多个进程并添加到进程列表中
        for i in range(ws):
            p = ctx.Process(
                target=f, args=(i, file.name, shared_tensors, ws, init_pg, c2p, p2c)
            )
            p.start()
            ps.append(p)

        # 从通信队列中获取数据，进行断言比较
        for _ in range(ws * n_output):
            pid, expected, result = c2p.get()
            self.assertEqual(
                expected,
                result,
                msg=f"Expect rank {pid} to receive tensor {expected} but got {result}.",
            )

        # 向通信队列发送终止信号
        for _ in range(ws):
            p2c.put(0)

        # 等待所有进程结束
        for p in ps:
            p.join(2)

    # _test_broadcast_process 方法用于测试广播操作的进程函数
    @classmethod
    def _test_broadcast_process(
        cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c
    ):
        # 初始化进程组
        pg = init_pg(rank, filename, world_size)
        xs = [shared_tensors[rank]]
        # 执行广播操作
        pg.broadcast(xs).wait()
        # 将结果放入通信队列，以便后续断言比较
        c2p.put((rank, torch.zeros(2, 2), xs[0].to("cpu")))
        p2c.get()

    # _test_allreduce_process 方法用于测试全局归约操作的进程函数
    @classmethod
    def _test_allreduce_process(
        cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c
    ):
        # 初始化进程组
        pg = init_pg(rank, filename, world_size)
        xs = [shared_tensors[rank]]
        # 执行全局归约操作
        pg.allreduce(xs, op=c10d.ReduceOp.SUM).wait()
        # 将结果放入通信队列，以便后续断言比较
        c2p.put((rank, torch.ones(2, 2) * 2, xs[0].to("cpu")))
        p2c.get()

    # _test_allgather_process 方法（未完成的方法声明，缺少函数体）
    @classmethod
    def _test_allgather_process(
        cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c
        ):
            # 调用初始化进程组函数，返回进程组对象
            pg = init_pg(rank, filename, world_size)
            # 创建一个包含当前进程rank对应的共享张量的列表
            xs = [shared_tensors[rank]]
            # 创建一个包含与进程数相同个数的零张量列表
            ys = [[torch.zeros_like(xs[0]) for i in range(world_size)]]
            # 执行进程组的全局收集操作，将xs中的数据收集到ys中
            pg.allgather(ys, xs).wait()
            # 对每个进程执行以下操作
            for i in range(world_size):
                # 将rank、一个2x2张量以及ys中的第i个张量的CPU副本放入队列c2p中
                c2p.put((rank, torch.ones(2, 2) * i, ys[0][i].to("cpu")))

            # 从队列p2c中获取数据（这里假设是一个阻塞操作）
            p2c.get()
class TestDistributedNNFunctions(MultiProcessTestCase):
    # 继承自MultiProcessTestCase的测试类，用于测试分布式神经网络函数

    def setUp(self):
        # 在每个测试方法运行之前执行的设置方法
        super().setUp()
        self._spawn_processes()
        # 调用父类的setUp方法并生成子进程

    def tearDown(self):
        # 在每个测试方法运行之后执行的清理方法
        super().tearDown()
        try:
            os.remove(self.file_name)
            # 尝试删除self.file_name指定的文件
        except OSError:
            pass
            # 如果文件不存在则忽略

    @property
    def op_timeout_sec(self):
        # 返回操作超时时间（秒）
        return 1

    @property
    def world_size(self):
        # 返回世界大小（进程数量）
        return 2

    def _test_broadcast(self, backend):
        # 测试广播功能的私有方法，使用指定的后端

        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建c10d文件存储对象，用于进程间通信，使用self.file_name和self.world_size

        # 初始化进程组，以便使用分布式操作
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )

        device = torch.device(f"cuda:{self.rank}")
        # 在CUDA设备上创建张量，设备索引为当前进程的rank

        x = torch.ones(5, 5, device=device) + self.rank
        x.requires_grad = True
        # 创建一个需要梯度的张量x，其值为1加上当前进程的rank

        y = torch.distributed.nn.broadcast(x, 1)
        # 使用分布式函数广播张量x到所有进程，广播的根为rank为1的进程

        self.assertEqual(y, 1 + torch.ones(5, 5))
        # 断言广播后的结果y与预期的值相等

        z = y.sin().sum()
        z.backward()
        # 对z进行反向传播，计算梯度

        # 由于无法数值化地检查通信的梯度，因此需要进行一些计算

        if self.rank == 1:
            self.assertEqual(x.grad, 2 * torch.cos(x))
            # 如果当前进程的rank为1，则断言张量x的梯度等于2乘以cos(x)
        elif self.rank == 0:
            self.assertEqual(x.grad, torch.zeros(5, 5, device=device))
            # 如果当前进程的rank为0，则断言张量x的梯度为全零张量

    def _test_reduce(self, backend):
        # 测试归约功能的私有方法，使用指定的后端

        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建c10d文件存储对象，用于进程间通信，使用self.file_name和self.world_size

        # 初始化进程组，以便使用分布式操作
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )

        device = torch.device(f"cuda:{self.rank}")
        # 在CUDA设备上创建张量，设备索引为当前进程的rank

        x = torch.ones(5, 5, device=device) + self.rank
        x.requires_grad = True
        # 创建一个需要梯度的张量x，其值为1加上当前进程的rank

        y = torch.distributed.nn.reduce(x, 1, op=c10d.ReduceOp.SUM)
        # 使用分布式函数对张量x进行归约操作，使用SUM操作符，根为rank为1的进程

        if self.rank == 1:
            self.assertEqual(y, 3 * torch.ones(5, 5, device=device))
            # 如果当前进程的rank为1，则断言归约后的结果y等于3乘以全1张量
       
        z = y.sin().sum()
        z.backward()
        # 对z进行反向传播，计算梯度

        # 梯度会被广播到所有进程

        x_g = (3 * torch.ones(5, 5, device=device)).cos()
        self.assertEqual(x.grad, x_g)
        # 断言张量x的梯度等于x_g

    def _test_allreduce(self, backend):
        # 测试全归约功能的私有方法，使用指定的后端

        store = c10d.FileStore(self.file_name, self.world_size)
        # 创建c10d文件存储对象，用于进程间通信，使用self.file_name和self.world_size

        # 初始化进程组，以便使用分布式操作
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )

        device = torch.device(f"cuda:{self.rank}")
        # 在CUDA设备上创建张量，设备索引为当前进程的rank

        x = torch.ones(5, 5, device=device) + self.rank
        x.requires_grad = True
        # 创建一个需要梯度的张量x，其值为1加上当前进程的rank

        y = torch.distributed.nn.all_reduce(x, op=c10d.ReduceOp.SUM)
        # 使用分布式函数对张量x进行全归约操作，使用SUM操作符

        self.assertEqual(y, 3 * torch.ones(5, 5, device=device))
        # 断言全归约后的结果y等于3乘以全1张量

        z = y.sin().sum()
        z.backward()
        # 对z进行反向传播，计算梯度

        x_g = 2 * (3 * torch.ones(5, 5, device=device)).cos()
        self.assertEqual(x.grad, x_g)
        # 断言张量x的梯度等于x_g
    def _test_all_gather(self, backend):
        # 创建一个文件存储对象，用于进程间通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化进程组，必须在使用分布式操作前调用，设置存储、进程rank、总进程数和后端
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        # 根据当前进程的rank创建对应的CUDA设备
        device = torch.device(f"cuda:{self.rank}")
        # 创建一个大小为5x5的张量，每个元素值为当前进程rank加一
        x = torch.ones(5, 5, device=device) + self.rank
        x.requires_grad = True  # 设置张量需要梯度计算
        # 使用分布式通信收集所有进程的张量x到tensors列表中
        tensors = torch.distributed.nn.all_gather(x)
        # 遍历收集到的张量，确保每个张量的值符合预期（全1加当前rank）
        for i, t in enumerate(tensors):
            self.assertEqual(t, torch.ones(5, 5, device=device) + i)
        # 对收集到的张量进行堆叠并求和，然后对结果进行sin函数运算并求和
        y = torch.sum(torch.stack(tensors), axis=0)
        z = y.sin().sum()
        z.backward()  # 对z进行反向传播计算梯度

        # 计算预期的梯度值并验证
        x_s = 2 * (3 * torch.ones(5, 5, device=device)).cos()
        self.assertEqual(x.grad, x_s)

    def _test_all_to_all(self, backend):
        # 创建一个文件存储对象，用于进程间通信
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化进程组，必须在使用分布式操作前调用，设置存储、进程rank、总进程数和后端
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        # 根据当前进程的rank创建对应的CUDA设备
        device = torch.device(f"cuda:{self.rank}")
        # 创建两个大小为5x5的张量，每个元素值为当前进程rank乘以2加一
        x0 = torch.ones(5, 5, device=device) + 2 * self.rank
        x1 = torch.ones(5, 5, device=device) + 2 * self.rank
        x0.requires_grad = True  # 设置张量需要梯度计算
        x1.requires_grad = True  # 设置张量需要梯度计算
        # 创建空张量作为接收端
        y0 = torch.empty_like(x0)
        y1 = torch.empty_like(x1)
        # 使用分布式通信进行所有到所有的通信，将x0和x1发送到y0和y1，并返回结果到tensors
        tensors = torch.distributed.nn.all_to_all([y0, y1], [x0, x1])
        # 遍历结果张量列表，确保每个张量的值符合预期（全1加当前rank乘以2）
        for i, t in enumerate(tensors):
            self.assertEqual(t, torch.ones(5, 5, device=device) + 2 * i)
        # 对收集到的张量进行堆叠并求和，然后对结果进行sin函数运算并求和
        y = torch.sum(torch.stack(tensors), axis=0)
        z = y.sin().sum()
        z.backward()  # 对z进行反向传播计算梯度
        # 计算预期的梯度值并验证
        x_s = (4 * torch.ones(5, 5, device=device)).cos()
        self.assertEqual(x0.grad, x_s)
        self.assertEqual(x1.grad, x_s)
    # 定义一个测试函数，用于测试分布式环境下的 all_to_all_single 函数
    def _test_all_to_all_single(self, backend):
        # 创建一个基于文件的存储，用于分布式进程组的初始化
        store = c10d.FileStore(self.file_name, self.world_size)
        # 初始化分布式进程组，需要指定存储、进程在组中的排名、组中的总进程数和后端类型
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        # 根据当前进程的排名创建对应的 CUDA 设备
        device = torch.device(f"cuda:{self.rank}")
        # 计算需要生成的数据行数
        row = self.world_size * (self.rank + 1) * (self.world_size + 1) / 2
        # 在指定设备上创建一个填充为1的张量，并允许计算梯度
        x = torch.ones(int(row), 5, device=device) * (self.rank + 1)
        x.requires_grad = True
        # 创建一个和 x 具有相同大小的空张量 y
        y = torch.empty_like(x)
        # 设置输入和输出分割大小，用于 all_to_all_single 函数
        split_sizes = [(i + 1) * (self.rank + 1) for i in range(self.world_size)]
        # 调用分布式的 all_to_all_single 函数，进行数据交换
        y = torch.distributed.nn.all_to_all_single(
            y, x, output_split_sizes=split_sizes, input_split_sizes=split_sizes
        )
        # 根据输入的分割大小生成期望结果
        expected = []
        for idx, tensor in enumerate(torch.split(x, split_sizes)):
            expected.append(torch.full_like(tensor, (idx + 1)))
        expected = torch.cat(expected)
        # 断言计算得到的 y 和期望的结果 expected 相等
        self.assertEqual(y, expected)
        # 对 y 求 sin 函数的和，并进行反向传播
        z = y.sin().sum()
        z.backward()
        # 计算对 x 的梯度并设置期望的结果
        x_s = ((self.rank + 1) * torch.ones(int(row), 5, device=device)).cos()
        # 断言计算得到的梯度 x.grad 和期望的结果 x_s 相等
        self.assertEqual(x.grad, x_s)
```