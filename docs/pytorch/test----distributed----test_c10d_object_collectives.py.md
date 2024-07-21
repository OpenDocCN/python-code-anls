# `.\pytorch\test\distributed\test_c10d_object_collectives.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的库
import os
import sys
from functools import partial, wraps

import torch
import torch.distributed as dist

# 如果分布式不可用，则跳过测试并退出程序
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 导入测试相关的模块和常量
from torch.testing._internal.common_distributed import MultiProcessTestCase, TEST_SKIPS
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

# 如果使用 dev-asan 测试，输出相关信息并退出程序
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 确定使用的分布式后端
BACKEND = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO

# 确定世界大小，取值在2到4之间
WORLD_SIZE = min(4, max(2, torch.cuda.device_count()))


def with_comms(func=None):
    # 如果没有传入函数，则返回一个带有_comms修饰的函数
    if func is None:
        return partial(
            with_comms,
        )

    # 包装器函数，用于初始化通信环境，并在测试完成后销毁通信环境
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # 如果使用NCCL后端且GPU数量小于world_size，则退出测试
        if BACKEND == dist.Backend.NCCL and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)
        # 初始化分布式环境
        self.dist_init()
        # 执行测试函数
        func(self)
        # 销毁通信环境
        self.destroy_comms()

    return wrapper


class TestObjectCollectives(MultiProcessTestCase):
    # 测试类的设置方法，在每个测试方法执行前调用
    def setUp(self):
        super().setUp()
        # 设置环境变量WORLD_SIZE和BACKEND
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["BACKEND"] = BACKEND
        # 启动多进程测试
        self._spawn_processes()

    # 获取设备属性，根据后端选择设备
    @property
    def device(self):
        return (
            torch.device(self.rank)
            if BACKEND == dist.Backend.NCCL
            else torch.device("cpu")
        )

    # 返回世界大小
    @property
    def world_size(self):
        return WORLD_SIZE

    # 返回进程组
    @property
    def process_group(self):
        return dist.group.WORLD

    # 销毁通信环境，等待所有进程达到此处后再执行销毁
    def destroy_comms(self):
        dist.barrier()
        dist.destroy_process_group()

    # 初始化分布式环境，设置通信后端和初始化方法
    def dist_init(self):
        dist.init_process_group(
            backend=BACKEND,
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

        # 对于NCCL后端，设置设备
        if BACKEND == dist.Backend.NCCL:
            torch.cuda.set_device(self.rank)

    # 使用with_comms修饰器执行的测试方法，测试dist.all_gather_object方法
    @with_comms()
    def test_all_gather_object(self):
        output = [None] * dist.get_world_size()
        # 执行all_gather_object方法
        dist.all_gather_object(object_list=output, obj=self.rank)

        # 验证输出结果
        for i, v in enumerate(output):
            self.assertEqual(i, v, f"rank: {self.rank}")

    # 使用with_comms修饰器执行的测试方法，测试dist.gather_object方法
    @with_comms()
    def test_gather_object(self):
        output = [None] * dist.get_world_size() if self.rank == 0 else None
        # 执行gather_object方法
        dist.gather_object(obj=self.rank, object_gather_list=output)

        # 如果rank为0，验证输出结果
        if self.rank == 0:
            for i, v in enumerate(output):
                self.assertEqual(i, v, f"rank: {self.rank}")

    @with_comms()
    def test_send_recv_object_list(self):
        # 如果当前进程是 rank 为 0，则 val 为 99，否则为 None
        val = 99 if self.rank == 0 else None
        # 创建一个 object_list 列表，包含 dist.get_world_size() 个 val 元素
        object_list = [val] * dist.get_world_size()
        # 如果当前进程是 rank 为 0，则向 rank 为 1 的进程发送 object_list
        if self.rank == 0:
            dist.send_object_list(object_list, 1)
        # 如果当前进程是 rank 为 1，则从 rank 为 0 的进程接收 object_list
        if self.rank == 1:
            dist.recv_object_list(object_list, 0)

        # 如果当前进程的 rank 小于 2，则断言 object_list 的第一个元素为 99，否则为 None
        if self.rank < 2:
            self.assertEqual(99, object_list[0])
        else:
            self.assertEqual(None, object_list[0])

    @with_comms()
    def test_broadcast_object_list(self):
        # 如果当前进程是 rank 为 0，则 val 为 99，否则为 None
        val = 99 if self.rank == 0 else None
        # 创建一个 object_list 列表，包含 dist.get_world_size() 个 val 元素
        object_list = [val] * dist.get_world_size()
        # 使用 dist.broadcast_object_list 广播 object_list
        dist.broadcast_object_list(object_list=object_list)

        # 断言 object_list 的第一个元素为 99
        self.assertEqual(99, object_list[0])

    @with_comms()
    def test_scatter_object_list(self):
        # 如果当前进程是 rank 为 0，则创建一个包含 dist.get_world_size() 个元素的列表
        # 否则设置 input_list 为 None
        input_list = list(range(dist.get_world_size())) if self.rank == 0 else None
        # 创建一个长度为 1 的 output_list 列表，初始化为 [None]
        output_list = [None]
        # 使用 dist.scatter_object_list 将 input_list 散布到 output_list 中
        dist.scatter_object_list(
            scatter_object_output_list=output_list, scatter_object_input_list=input_list
        )

        # 断言 output_list 的第一个元素为当前进程的 rank
        self.assertEqual(self.rank, output_list[0])

    # Test Object Collectives With Sub Pg

    def setup_sub_pg(self):
        # 获取当前进程的 rank
        rank = dist.get_rank()
        # 计算当前进程的 base_rank，base_rank 是当前 rank 的偶数部分
        base_rank = rank - (rank % 2)
        # 创建一个包含当前进程和下一个进程 rank 的列表 ranks
        ranks = [base_rank, base_rank + 1]
        # 使用 ranks 创建一个新的 process group my_pg，并启用本地同步
        my_pg = dist.new_group(ranks, use_local_synchronization=True)
        return rank, ranks, my_pg

    @with_comms()
    def test_subpg_scatter_object(self):
        # 调用 setup_sub_pg 方法获取 rank, ranks, my_pg
        rank, ranks, my_pg = self.setup_sub_pg()
        # 创建一个长度为 1 的 out_list 列表，初始化为 [None]
        out_list = [None]
        # 使用 dist.scatter_object_list 将 out_list 散布到 ranks 中
        # 源地址是 ranks[0] 的进程，使用 my_pg 的进程组
        dist.scatter_object_list(out_list, ranks, src=ranks[0], group=my_pg)
        # 断言 out_list 的第一个元素为当前进程的 rank
        self.assertEqual(rank, out_list[0])

    @with_comms()
    def test_subpg_all_gather_object(self):
        # 调用 setup_sub_pg 方法获取 rank, ranks, my_pg
        rank, ranks, my_pg = self.setup_sub_pg()
        # 创建一个长度为 ranks 的元素个数的 out_list 列表，初始化为 [None]
        out_list = [None] * len(ranks)
        # 使用 dist.all_gather_object 将当前进程的 rank 收集到 out_list 中
        # 使用 my_pg 的进程组
        dist.all_gather_object(out_list, rank, group=my_pg)
        # 断言 out_list 应该和 ranks 相等
        self.assertEqual(ranks, out_list)

    @with_comms()
    def test_subpg_gather_object(self):
        # 调用 setup_sub_pg 方法获取 rank, ranks, my_pg
        rank, ranks, my_pg = self.setup_sub_pg()
        # 如果当前进程的 rank 是 ranks 的第一个进程，则创建一个长度为 ranks 的元素个数的 out_list 列表，初始化为 [None]
        # 否则设置 out_list 为 None
        out_list = [None] * len(ranks) if rank == ranks[0] else None
        # 使用 dist.gather_object 将当前进程的 rank 收集到 ranks[0] 进程的 out_list 中
        # 使用 my_pg 的进程组
        dist.gather_object(rank, out_list, dst=ranks[0], group=my_pg)
        # 如果当前进程的 rank 是 ranks 的第一个进程，则断言 out_list 应该和 ranks 相等
        if rank == ranks[0]:
            self.assertEqual(ranks, out_list)

    @with_comms()
    def test_subpg_broadcast_object(self):
        # 调用 setup_sub_pg 方法获取 rank, ranks, my_pg
        rank, ranks, my_pg = self.setup_sub_pg()
        # 创建一个长度为 1 的 out_list 列表，初始化为 [None]
        out_list = [None]
        # 如果当前进程的 rank 是 ranks 的第一个进程，则将 out_list 的第一个元素设置为当前进程的 rank
        if rank == ranks[0]:
            out_list[0] = rank
        # 使用 dist.broadcast_object_list 将 out_list 广播到 my_pg 的进程组
        # 源地址是 ranks[0] 的进程
        dist.broadcast_object_list(out_list, src=ranks[0], group=my_pg)
        # 断言 out_list 的第一个元素应该和 ranks[0] 的 rank 相等
        self.assertEqual(ranks[0], out_list[0])
# 如果当前脚本作为主程序执行（而不是被导入为模块），则运行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```