# `.\pytorch\test\distributed\_shard\sharded_tensor\ops\test_binary_cmp.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import sys

import torch
import torch.distributed as dist

# 导入分布式张量相关模块和类
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed.distributed_c10d import _get_default_group

# 导入测试相关的辅助函数和类
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)

# 如果测试设置为使用开发调试 ASAN，输出信息并退出
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 定义测试用例类 TestShardedTensorBinaryOps，继承自 ShardedTensorTestBase
class TestShardedTensorBinaryOps(ShardedTensorTestBase):
    """Test base for binary comparison functions such as torch.equal, torch.allclose etc. for ShardedTensor"""

    seed = 42  # 设置随机种子

    # 获取随机生成的分片张量
    def get_random_tensors(
        self, spec1, spec2, *sizes, pg1=None, pg2=None, seed_offset=0
    ):
        pg1 = _get_default_group() if pg1 is None else pg1
        pg2 = _get_default_group() if pg2 is None else pg2

        # 设置随机种子并生成第一个分片张量
        torch.manual_seed(TestShardedTensorBinaryOps.seed)
        st1 = sharded_tensor.rand(spec1, sizes, process_group=pg1)

        # 设置不同的随机种子并生成第二个分片张量
        torch.manual_seed(TestShardedTensorBinaryOps.seed + seed_offset)
        st2 = sharded_tensor.rand(spec2, sizes, process_group=pg2)

        # 更新随机种子以保证每次调用都有不同的随机数生成
        TestShardedTensorBinaryOps.seed += 1

        return st1, st2

    # 获取 GPU 分片规格
    def get_gpu_specs(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        alt_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:1/cuda:1",
                "rank:0/cuda:0",
                "rank:3/cuda:3",
                "rank:2/cuda:2",
            ],
        )
        return spec, alt_spec
    # 定义一个私有方法，用于测试常见的失败情况，接受一个比较操作函数作为参数
    def _test_common_failures(self, cmp_op):
        # 获取当前 GPU 规格和备用 GPU 规格
        spec, alt_spec = self.get_gpu_specs()

        # 获取两个随机张量 st1 和 st2
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10)
        # 如果当前进程的 rank 是 0，则对 st1 的第一个本地分片的张量进行均匀初始化
        if self.rank == 0:
            torch.nn.init.uniform_(st1.local_shards()[0].tensor)
        # 使用给定的比较操作函数判断 st1 和 st2 是否不相等
        self.assertFalse(cmp_op(st1, st2))

        # 创建一个全为 1 的分片张量 st1 和一个形状不同的全为 1 的分片张量 st2
        st1 = sharded_tensor.ones(spec, 10, 10)
        st2 = sharded_tensor.ones(spec, 10, 5)
        # 使用给定的比较操作函数判断 st1 和 st2 是否不相等
        self.assertFalse(cmp_op(st1, st2))

        # 获取一个随机张量 st1 和一个使用备用 GPU 规格的随机张量 st2
        st1, st2 = self.get_random_tensors(spec, alt_spec, 10, 10)
        # 使用给定的比较操作函数判断 st1 和 st2 是否不相等
        self.assertFalse(cmp_op(st1, st2))

        # 创建一个全为 1 的分片张量 st1 和一个全为 0 的分片张量 st2
        st1 = sharded_tensor.ones(spec, 10, 10)
        st2 = sharded_tensor.zeros(spec, 10, 10)
        # 使用给定的比较操作函数判断 st1 和 st2 是否不相等
        self.assertFalse(cmp_op(st1, st2))

        # 创建一个全为 1 的分片张量 st1 和一个 dtype 为 torch.double 的全为 1 的分片张量 st2
        st1 = sharded_tensor.ones(spec, 10, 10)
        st2 = sharded_tensor.ones(spec, 10, 10, dtype=torch.double)
        # 使用给定的比较操作函数判断 st1 和 st2 是否不相等
        self.assertFalse(cmp_op(st1, st2))

        # 创建一个全为 1 的分片张量 st1 和一个 requires_grad=True 的全为 1 的分片张量 st2
        st1 = sharded_tensor.ones(spec, 10, 10)
        st2 = sharded_tensor.ones(spec, 10, 10, requires_grad=True)
        # 使用给定的比较操作函数判断 st1 和 st2 是否不相等
        self.assertFalse(cmp_op(st1, st2))

        # 创建一个指定了 CPU 分片规格的全为 1 的分片张量 st1 和一个指定了 pin_memory=True 的全为 1 的分片张量 st2
        cpu_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cpu",
                "rank:1/cpu",
                "rank:2/cpu",
                "rank:3/cpu",
            ],
        )
        st1 = sharded_tensor.ones(cpu_spec, 10, 10)
        st2 = sharded_tensor.ones(cpu_spec, 10, 10, pin_memory=True)
        # 使用给定的比较操作函数判断 st1 和 st2 是否不相等
        self.assertFalse(cmp_op(st1, st2))

        # 创建一个新的进程组 pg，包含指定的 ranks，并获取两个随机张量 st1 和 st2，使用 pg2 作为 ProcessGroup
        pg = dist.new_group([1, 0, 3, 2])
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10, pg2=pg)
        # 使用断言捕获 RuntimeError，检查其错误信息是否包含特定字符串
        with self.assertRaisesRegex(
            RuntimeError, "All distributed tensors should use the same ProcessGroup"
        ):
            cmp_op(st1, st2)

        # 创建一个新的进程组 pg，包含指定的 ranks，并获取两个随机张量 st1 和 st2，使用 pg2 作为 ProcessGroup
        pg = dist.new_group([0, 1, 2, 3])
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10, pg2=pg)
        # 使用断言捕获 RuntimeError，检查其错误信息是否包含特定字符串
        with self.assertRaisesRegex(
            RuntimeError, "All distributed tensors should use the same ProcessGroup"
        ):
            cmp_op(st1, st2)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    # 测试 torch.equal(ShardedTensor, ShardedTensor)
    def test_torch_equal_tensor_specs(self):
        self._test_common_failures(torch.equal)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    # 测试 torch.equal(ShardedTensor, ShardedTensor)，确保两个分片张量相等
    def test_torch_equal(self):
        """Test torch.equal(ShardedTensor, ShardedTensor)"""

        # 获取当前 GPU 规格和备用 GPU 规格
        spec, alt_spec = self.get_gpu_specs()
        # 获取两个随机张量 st1 和 st2
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10)
        # 使用 torch.equal 函数判断 st1 和 st2 是否相等
        self.assertTrue(torch.equal(st1, st2))

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    # 测试 torch.allclose(ShardedTensor, ShardedTensor)
    def test_torch_allclose_tensor_specs(self):
        self._test_common_failures(torch.allclose)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_torch_allclose(self):
        """Test torch.allclose(ShardedTensor, ShardedTensor)"""

        # 获取 GPU 规格和备用规格
        spec, alt_spec = self.get_gpu_specs()

        # 获取两个随机生成的 ShardedTensor 对象，形状为 (10, 10)，使用相同的规格
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10)

        # 断言 st1 和 st2 的所有元素在默认容差下相近
        self.assertTrue(torch.allclose(st1, st2))

        # 断言 st1 和 st2 的所有元素在容差为 0 时完全相等
        self.assertTrue(torch.allclose(st1, st2, atol=0))

        # 获取两个随机生成的 ShardedTensor 对象，形状为 (10, 10)，使用相同的规格和不同的种子偏移量
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10, seed_offset=1)

        # 断言 st1 和 st2 的所有元素在默认容差下不相近
        self.assertFalse(torch.allclose(st1, st2))

        # 断言 st1 和 st2 的所有元素在容差为 1 时相近，由于随机性，可能会有小的差异
        self.assertTrue(torch.allclose(st1, st2, atol=1))
# 如果当前脚本作为主程序执行（而不是作为模块导入），则运行 `run_tests()` 函数
if __name__ == "__main__":
    run_tests()
```