# `.\pytorch\test\distributed\test_c10d_spawn_ucc.py`

```
# Owner(s): ["oncall: distributed"]

# 导入 sys 模块
import sys

# 导入测试相关模块和类
import test_c10d_spawn
from test_c10d_spawn import _torch_dist_nn_available, TestDistributedNNFunctions

# 导入 PyTorch 库
import torch
import torch.distributed as c10d
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import requires_ucc, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)

# 检查是否支持 UCC，如果不支持则标记为 NO_UCC
NO_UCC = not hasattr(c10d, "ProcessGroupUCC")

# 在 Python-3.9 上存在问题，详见 https://github.com/pytorch/pytorch/issues/51619
if sys.version_info < (3, 9):

    # 定义一个测试类，继承自 AbstractProcessGroupShareTensorTest 和 TestCase
    class ProcessGroupShareTensorTest(
        test_c10d_spawn.AbstractProcessGroupShareTensorTest, TestCase
    ):
        
        # 初始化 UCC 进程组
        @classmethod
        def _init_pg_ucc(cls, rank, filename, world_size):
            # 使用文件存储创建 c10d 进程组
            store = c10d.FileStore(filename, world_size)
            c10d.init_process_group(
                backend="ucc", store=store, rank=rank, world_size=world_size
            )
            # 返回默认的进程组
            return c10d.distributed_c10d._get_default_group()

        # 如果不支持多 GPU，则跳过测试
        @skip_but_pass_in_sandcastle_if(
            not TEST_MULTIGPU, "At least 2 CUDA GPUS needed"
        )
        # 如果不支持 UCC，则跳过测试
        @skip_but_pass_in_sandcastle_if(NO_UCC, "UCC needed")
        # 测试共享广播操作使用 UCC
        def test_shared_broadcast_ucc(self):
            self._test_multiprocess(
                ProcessGroupShareTensorTest._test_broadcast_process,
                [torch.ones(2, 2).to(i) * i for i in range(self.world_size)],
                ProcessGroupShareTensorTest._init_pg_ucc,
                1,
            )

        # 如果不支持多 GPU，则跳过测试
        @skip_but_pass_in_sandcastle_if(
            not TEST_MULTIGPU, "At least 2 CUDA GPUS needed"
        )
        # 如果不支持 UCC，则跳过测试
        @skip_but_pass_in_sandcastle_if(NO_UCC, "UCC needed")
        # 测试共享全局归约操作使用 UCC
        def test_shared_allreduce_ucc(self):
            self._test_multiprocess(
                ProcessGroupShareTensorTest._test_allreduce_process,
                [torch.ones(2, 2).to(i) for i in range(self.world_size)],
                ProcessGroupShareTensorTest._init_pg_ucc,
                1,
            )

        # 如果不支持多 GPU，则跳过测试
        @skip_but_pass_in_sandcastle_if(
            not TEST_MULTIGPU, "At least 2 CUDA GPUS needed"
        )
        # 如果不支持 UCC，则跳过测试
        @skip_but_pass_in_sandcastle_if(NO_UCC, "UCC needed")
        # 测试共享全局聚集操作使用 UCC
        def test_shared_allgather_ucc(self):
            self._test_multiprocess(
                ProcessGroupShareTensorTest._test_allgather_process,
                [torch.ones(2, 2).to(i) * i for i in range(self.world_size)],
                ProcessGroupShareTensorTest._init_pg_ucc,
                self.world_size,
            )

# 跳过在开发 ASAN 环境下的测试，因为 torch + multiprocessing spawn 存在已知问题
if not TEST_WITH_DEV_DBG_ASAN:
    class TestDistributedNNFunctionsUcc(TestDistributedNNFunctions):
        # Test Common Ops First.
        # 定义一个测试类 TestDistributedNNFunctionsUcc，继承自 TestDistributedNNFunctions
        
        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        # 标记为需要 UCC 支持的测试，并跳过如果 GPU 小于 2 个
        # 如果 _torch_dist_nn_available 为 False，则在沙盒环境中跳过测试，显示指定消息
        
        def test_broadcast(self):
            # 测试广播操作
            self._test_broadcast("ucc")
            # 调用内部方法 _test_broadcast，使用 UCC 实现
        
        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        # 标记为需要 UCC 支持的测试，并跳过如果 GPU 小于 2 个
        # 如果 _torch_dist_nn_available 为 False，则在沙盒环境中跳过测试，显示指定消息
        
        def test_reduce(self):
            # 测试减少（reduce）操作
            self._test_reduce("ucc")
            # 调用内部方法 _test_reduce，使用 UCC 实现
        
        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        # 标记为需要 UCC 支持的测试，并跳过如果 GPU 小于 2 个
        # 如果 _torch_dist_nn_available 为 False，则在沙盒环境中跳过测试，显示指定消息
        
        def test_allreduce(self):
            # 测试全局减少（allreduce）操作
            self._test_allreduce("ucc")
            # 调用内部方法 _test_allreduce，使用 UCC 实现
        
        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle(
            "runs into illegal memory access on first assertEqual check when run locally"
        )
        # 标记为需要 UCC 支持的测试，并跳过如果 GPU 小于 2 个
        # 在沙盒环境中跳过测试，显示指定消息：在本地运行时遇到非法内存访问
        
        def test_all_gather(self):
            # 测试全收集（all gather）操作
            self._test_all_gather("ucc")
            # 调用内部方法 _test_all_gather，使用 UCC 实现
        
        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        # 标记为需要 UCC 支持的测试，并跳过如果 GPU 小于 2 个
        # 如果 _torch_dist_nn_available 为 False，则在沙盒环境中跳过测试，显示指定消息
        
        def test_all_to_all(self):
            # 测试全到全（all to all）操作
            self._test_all_to_all("ucc")
            # 调用内部方法 _test_all_to_all，使用 UCC 实现
        
        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        # 标记为需要 UCC 支持的测试，并跳过如果 GPU 小于 2 个
        # 如果 _torch_dist_nn_available 为 False，则在沙盒环境中跳过测试，显示指定消息
        
        def test_all_to_all_single(self):
            # 测试单个全到全（all to all single）操作
            self._test_all_to_all_single("ucc")
            # 调用内部方法 _test_all_to_all_single，使用 UCC 实现
# 如果当前脚本被直接执行（而不是被导入为模块），则运行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```