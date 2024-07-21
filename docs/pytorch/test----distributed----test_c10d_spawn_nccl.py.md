# `.\pytorch\test\distributed\test_c10d_spawn_nccl.py`

```py
# Owner(s): ["oncall: distributed"]
# 导入系统模块
import sys

# 导入测试模块和相关函数、类
import test_c10d_spawn
from test_c10d_spawn import _torch_dist_nn_available, TestDistributedNNFunctions

# 导入 PyTorch 相关模块
import torch
import torch.distributed as c10d

# 导入测试相关的辅助函数和类
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)

# 检查是否缺少 ProcessGroupNCCL 属性，设置 NO_NCCL 变量
NO_NCCL = not hasattr(c10d, "ProcessGroupNCCL")

# Python 版本小于 3.9 时执行以下代码
if sys.version_info < (3, 9):

    # 定义一个测试类 ProcessGroupShareTensorTest，继承自 AbstractProcessGroupShareTensorTest 和 TestCase
    class ProcessGroupShareTensorTest(
        test_c10d_spawn.AbstractProcessGroupShareTensorTest, TestCase
        ):
            # 定义一个类方法，用于初始化使用 NCCL 的进程组
            @classmethod
            def _init_pg_nccl(cls, rank, filename, world_size):
                # 创建一个 FileStore 对象，用于 NCCL 进程组
                store = c10d.FileStore(filename, world_size)
                # 返回一个 ProcessGroupNCCL 对象，用于指定 rank 的进程
                return c10d.ProcessGroupNCCL(store, rank, world_size)

            # 如果不满足 TEST_MULTIGPU 条件，则跳过此测试
            @skip_but_pass_in_sandcastle_if(
                not TEST_MULTIGPU, "At least 2 CUDA GPUS needed"
            )
            # 如果不满足 NO_NCCL 条件，则跳过此测试
            @skip_but_pass_in_sandcastle_if(NO_NCCL, "NCCL needed")
            # 定义测试方法，测试 NCCL 共享广播功能
            def test_shared_broadcast_nccl(self):
                # 调用 _test_multiprocess 方法进行多进程测试
                self._test_multiprocess(
                    # 使用 ProcessGroupShareTensorTest 的 _test_broadcast_process 方法作为测试函数
                    ProcessGroupShareTensorTest._test_broadcast_process,
                    # 创建一个列表，包含每个进程的共享张量
                    [torch.ones(2, 2).to(i) * i for i in range(self.world_size)],
                    # 使用 _init_pg_nccl 方法初始化 NCCL 进程组
                    ProcessGroupShareTensorTest._init_pg_nccl,
                    # 指定每个进程的数量
                    1,
                )

            # 如果不满足 TEST_MULTIGPU 条件，则跳过此测试
            @skip_but_pass_in_sandcastle_if(
                not TEST_MULTIGPU, "At least 2 CUDA GPUS needed"
            )
            # 如果不满足 NO_NCCL 条件，则跳过此测试
            @skip_but_pass_in_sandcastle_if(NO_NCCL, "NCCL needed")
            # 定义测试方法，测试 NCCL 共享全局归约功能
            def test_shared_allreduce_nccl(self):
                # 调用 _test_multiprocess 方法进行多进程测试
                self._test_multiprocess(
                    # 使用 ProcessGroupShareTensorTest 的 _test_allreduce_process 方法作为测试函数
                    ProcessGroupShareTensorTest._test_allreduce_process,
                    # 创建一个列表，包含每个进程的共享张量
                    [torch.ones(2, 2).to(i) for i in range(self.world_size)],
                    # 使用 _init_pg_nccl 方法初始化 NCCL 进程组
                    ProcessGroupShareTensorTest._init_pg_nccl,
                    # 指定每个进程的数量
                    1,
                )

            # 定义一个类方法，用于测试进程归约功能
            @classmethod
            def _test_reduce_process(
                cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c
            ):
                # 使用 init_pg 方法初始化进程组
                pg = init_pg(rank, filename, world_size)
                # 获取当前进程的共享张量
                x = shared_tensors[rank]
                # 对共享张量进行归约操作，求和并等待完成
                pg.reduce(x, root=0, op=c10d.ReduceOp.SUM).wait()
                # 如果当前进程是根进程（rank==0），将结果放入 c2p 队列中
                if rank == 0:
                    c2p.put((rank, torch.ones(2, 2) * 2, x.to("cpu")))
                else:
                    # 如果不是根进程，将结果放入 c2p 队列中
                    c2p.put((rank, torch.ones(2, 2), x.to("cpu")))
                # 从 p2c 队列获取数据
                p2c.get()

            # 如果不满足 TEST_MULTIGPU 条件，则跳过此测试
            @skip_but_pass_in_sandcastle_if(
                not TEST_MULTIGPU, "At least 2 CUDA GPUS needed"
            )
            # 如果不满足 NO_NCCL 条件，则跳过此测试
            @skip_but_pass_in_sandcastle_if(NO_NCCL, "NCCL needed")
            # 定义测试方法，测试 NCCL 共享进程归约功能
            def test_shared_reduce_nccl(self):
                # 调用 _test_multiprocess 方法进行多进程测试
                self._test_multiprocess(
                    # 使用 ProcessGroupShareTensorTest 的 _test_reduce_process 方法作为测试函数
                    ProcessGroupShareTensorTest._test_reduce_process,
                    # 创建一个列表，包含每个进程的共享张量
                    [torch.ones(2, 2).to(i) for i in range(self.world_size)],
                    # 使用 _init_pg_nccl 方法初始化 NCCL 进程组
                    ProcessGroupShareTensorTest._init_pg_nccl,
                    # 指定每个进程的数量
                    1,
                )

            # 如果不满足 TEST_MULTIGPU 条件，则跳过此测试
            @skip_but_pass_in_sandcastle_if(
                not TEST_MULTIGPU, "At least 2 CUDA GPUS needed"
            )
            # 如果不满足 NO_NCCL 条件，则跳过此测试
            @skip_but_pass_in_sandcastle_if(NO_NCCL, "NCCL needed")
            # 定义测试方法，测试 NCCL 共享全局收集功能
            def test_shared_allgather_nccl(self):
                # 调用 _test_multiprocess 方法进行多进程测试
                self._test_multiprocess(
                    # 使用 ProcessGroupShareTensorTest 的 _test_allgather_process 方法作为测试函数
                    ProcessGroupShareTensorTest._test_allgather_process,
                    # 创建一个列表，包含每个进程的共享张量
                    [torch.ones(2, 2).to(i) * i for i in range(self.world_size)],
                    # 使用 _init_pg_nccl 方法初始化 NCCL 进程组
                    ProcessGroupShareTensorTest._init_pg_nccl,
                    # 指定进程组的大小
                    self.world_size,
                )
# 如果未启用 dev-asan 调试（ASAN = AddressSanitizer），则执行以下代码块
if not TEST_WITH_DEV_DBG_ASAN:
    # 如果脚本作为主程序运行时（而不是被导入为模块），则执行以下代码块
    if __name__ == "__main__":
        # 运行测试函数
        run_tests()
```