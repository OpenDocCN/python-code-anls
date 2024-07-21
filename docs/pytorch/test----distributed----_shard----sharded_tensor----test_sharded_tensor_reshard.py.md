# `.\pytorch\test\distributed\_shard\sharded_tensor\test_sharded_tensor_reshard.py`

```
# Owner(s): ["oncall: distributed"]

# 导入系统相关的模块和函数
import sys
# 导入 itertools 中的 product 函数，用于生成迭代器的笛卡尔积
from itertools import product

# 导入 PyTorch 库
import torch
# 导入分布式相关的模块和函数
from torch.distributed._shard import _shard_tensor, sharded_tensor
from torch.distributed._shard.sharding_spec import EnumerableShardingSpec, ShardMetadata
# 导入测试相关的模块和函数
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
# 导入分布式测试相关的模块
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_st_common import (
    _chunk_sharding_specs_list_for_test,
)

# 如果处于开发调试模式（ASAN），则打印消息并退出程序
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 定义一个测试类 TestReshard，继承自 ShardedTensorTestBase
class TestReshard(ShardedTensorTestBase):

    # 定义一个私有方法，用于运行 sharded_tensor 的 reshard 操作
    def _run_sharded_tensor_reshard(self, sharding_spec, reshard_spec, input_size):
        # 设置随机种子为 0
        torch.manual_seed(0)
        # 在当前 GPU 节点上生成指定大小的随机张量
        local_tensor = torch.rand(*input_size).cuda(self.rank)
        # 使用给定的 sharding_spec 对 local_tensor 进行分片
        st = _shard_tensor(local_tensor, sharding_spec)
        # 使用相同的 local_tensor 和 reshard_spec 创建另一个分片张量
        st_compare = _shard_tensor(local_tensor, reshard_spec)
        # 对第一个分片张量进行 reshard 操作
        st.reshard(reshard_spec)
        # 断言第一个分片张量只有一个本地分片
        self.assertEqual(1, len(st.local_shards()))
        # 断言第二个分片张量也只有一个本地分片
        self.assertEqual(1, len(st_compare.local_shards()))
        # 对第二个分片张量的元数据按照 placement.rank() 排序
        st_compare._metadata.shards_metadata.sort(
            key=lambda metadata: metadata.placement.rank()
        )
        # 断言两个分片张量的元数据相等
        self.assertEqual(st._metadata, st_compare._metadata)
        # 断言两个分片张量的本地张量数据相等
        self.assertEqual(st.local_tensor(), st_compare.local_tensor())
        # 断言两个分片张量的第一个本地分片的元数据相等
        self.assertEqual(
            st.local_shards()[0].metadata, st_compare.local_shards()[0].metadata
        )

    # 测试方法的装饰器，初始化 RPC 通信，跳过小于 4 个 GPU 的情况，需要 NCCL 支持
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    # 定义测试 sharded_tensor 的 reshard 操作的方法
    def test_sharded_tensor_reshard(self):
        # 定义维度列表
        dims = [0, 1]
        # 对 dims 中的维度进行两两组合
        for sharding_dim, reshard_dim in product(dims, dims):
            # 使用 seed=5 生成用于测试的 sharding_spec 和 reshard_spec
            specs = _chunk_sharding_specs_list_for_test(
                [sharding_dim, reshard_dim], seed=5
            )
            # 分别取出 spec 和 reshard_spec 进行测试
            spec, reshard_spec = specs[0], specs[1]
            self._run_sharded_tensor_reshard(spec, reshard_spec, [13, 21])
            self._run_sharded_tensor_reshard(spec, reshard_spec, [14, 23])
            self._run_sharded_tensor_reshard(spec, reshard_spec, [15, 26])
            self._run_sharded_tensor_reshard(spec, reshard_spec, [12, 24])

    # 测试方法的装饰器，初始化 RPC 通信，跳过小于 4 个 GPU 的情况，需要 NCCL 支持
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    # 定义测试方法，测试分片张量的重新分片时可能出现的错误情况
    def test_sharded_tensor_reshard_errors(self):
        # 生成用于测试的分片规格列表，种子为6
        specs = _chunk_sharding_specs_list_for_test([0, 1], seed=6)
        # 选择第一个和第二个规格作为测试的特定规格和重新分片规格
        spec, reshard_spec = specs[0], specs[1]
        
        # 定义一个可枚举的分片规格对象，包含两个分片元数据
        enumerable_sharding_spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
            ]
        )
        
        # 创建一个随机初始化的分片张量对象，使用给定的规格和维度
        st = sharded_tensor.rand(spec, 24, 12)
        
        # 断言在重新分片时抛出 NotImplementedError 异常，且异常消息指定只支持 ChunkShardingSpec
        with self.assertRaisesRegex(
            NotImplementedError, "Only ChunkShardingSpec supported for reshard."
        ):
            st.reshard(enumerable_sharding_spec)
        
        # 将分片张量的本地分片设置为第一个本地分片的复制，用于测试第二个错误情况
        st._local_shards = [st.local_shards()[0], st.local_shards()[0]]
        
        # 断言在重新分片时抛出 NotImplementedError 异常，且异常消息指定只支持单个本地分片
        with self.assertRaisesRegex(
            NotImplementedError, "Only single local shard supported for reshard."
        ):
            st.reshard(reshard_spec)
# 如果当前脚本作为主程序执行（而不是被导入到其他模块中），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```