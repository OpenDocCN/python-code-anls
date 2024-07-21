# `.\pytorch\test\distributed\_shard\sharded_tensor\ops\test_embedding.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入系统模块
import sys

# 导入PyTorch相关模块
import torch
import torch.distributed as dist
from torch.distributed._shard import shard_parameter
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    TEST_GPU_NUM,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_ops_common import (
    clone_module_parameter,
    generate_chunk_sharding_specs_for_test,
    generate_local_weight_sharding_params_for_test,
)

# 如果使用dev-asan模式，打印信息并退出
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 定义测试类 TestShardedEmbedding，继承自 ShardedTensorTestBase
class TestShardedEmbedding(ShardedTensorTestBase):
    
    # 定义内部方法 _run_sharded_embedding，用于运行分片嵌入测试
    def _run_sharded_embedding(
        self,
        spec,
        input_size,
        num_embeddings,
        embedding_dim,
        max_norm=None,
        norm_type=2.0,
        padding_idx=None,
        ):
        # 使用相同的种子初始化随机数生成器。
        torch.manual_seed(0)
        # 在当前进程内为本地嵌入创建 Embedding 层，将其放置在 GPU 上
        local_embedding = torch.nn.Embedding(
            num_embeddings,
            embedding_dim,
            max_norm=max_norm,
            norm_type=norm_type,
            padding_idx=padding_idx,
        ).cuda(self.rank)

        # 创建一个未放置在 GPU 上的分片 Embedding 层
        sharded_embedding = torch.nn.Embedding(
            num_embeddings,
            embedding_dim,
            max_norm=max_norm,
            norm_type=norm_type,
            padding_idx=padding_idx,
        )

        # 从本地嵌入层复制权重到分片嵌入层
        sharded_embedding.weight = clone_module_parameter(local_embedding, "weight")

        # 根据规格对分片嵌入层的权重进行分片
        shard_parameter(sharded_embedding, "weight", spec)

        # 运行分片计算
        torch.manual_seed(self.rank)  # 每个 rank 使用不同的输入
        inp = torch.randint(0, num_embeddings, tuple(input_size)).cuda(self.rank)
        sharded_output = sharded_embedding(inp)

        # 如果设置了 max_norm，确保在所有 rank 的输入上应用了 renorm
        if max_norm is not None:
            gathered_inputs = [torch.zeros_like(inp) for _ in range(TEST_GPU_NUM)]
            dist.all_gather(gathered_inputs, inp)
            unique_inp = torch.unique(torch.cat(gathered_inputs))
            local_embedding(unique_inp)

        # 运行本地计算
        local_output = local_embedding(inp)

        # 比较本地权重和分片权重，确保 renorm 正常应用
        if max_norm is not None:
            sharded_dim = spec.dim
            sharded_weight = sharded_embedding.weight.local_shards()[0].tensor
            (start_pos, chunk_size) = generate_local_weight_sharding_params_for_test(
                local_embedding.weight, sharded_dim, TEST_GPU_NUM, spec, self.rank
            )
            local_weight_narrowed = local_embedding.weight.narrow(
                sharded_dim, start_pos, chunk_size
            )
            self.assertEqual(local_weight_narrowed, sharded_weight)

        # 验证计算结果是否一致
        self.assertEqual(local_output, sharded_output)

        # 使用 torch.nn.functional.embedding 进行验证
        local_output = torch.nn.functional.embedding(
            inp,
            local_embedding.weight,
            max_norm=max_norm,
            norm_type=norm_type,
            padding_idx=padding_idx,
        )
        sharded_output = torch.nn.functional.embedding(
            inp,
            sharded_embedding.weight,
            max_norm=max_norm,
            norm_type=norm_type,
            padding_idx=padding_idx,
        )

        # 检查使用 functional.embedding 的计算结果是否一致
        self.assertEqual(local_output, sharded_output)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    # 定义测试方法，用于测试分片嵌入的列方向计算
    def test_sharded_embedding_colwise(self):
        # 对于生成的测试用例规格，循环执行以下操作
        for spec in generate_chunk_sharding_specs_for_test(1):
            # 使用指定的规格和参数运行分片嵌入方法，测试不同的输入参数组合
            self._run_sharded_embedding(spec, [5, 4], 17, 12)
            self._run_sharded_embedding(spec, [6, 7, 6], 21, 11)
            self._run_sharded_embedding(spec, [8, 6, 5, 4], 23, 13)
            self._run_sharded_embedding(spec, [8, 6, 5, 4, 7], 23, 16)
            self._run_sharded_embedding(spec, [4], 15, 14)
            self._run_sharded_embedding(spec, [34], 15, 14, padding_idx=10)
            self._run_sharded_embedding(spec, [8, 6, 5, 4], 23, 13, padding_idx=12)
            self._run_sharded_embedding(
                spec,
                [4, 5, 6],
                23,
                13,
                max_norm=2.5,
            )
            self._run_sharded_embedding(
                spec,
                [12, 7, 16],
                23,
                13,
                max_norm=2.5,
            )
            self._run_sharded_embedding(
                spec,
                [8, 16, 20],
                12,
                12,
                max_norm=1.25,
                norm_type=1.0,
            )
            self._run_sharded_embedding(spec, [30], 15, 14, max_norm=2.0)

    # 使用通信初始化装饰器，跳过没有足够 GPU 数量的测试，并要求支持 NCCL
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    # 定义测试方法，用于测试分片嵌入的行方向计算
    def test_sharded_embedding_rowwise(self):
        # 对于生成的测试用例规格，循环执行以下操作
        for spec in generate_chunk_sharding_specs_for_test(0):
            # 测试均匀分片
            self._run_sharded_embedding(spec, [5, 12], 16, 22)
            self._run_sharded_embedding(spec, [5, 4], 32, 12)
            self._run_sharded_embedding(spec, [6, 7, 6], 64, 11)
            self._run_sharded_embedding(
                spec,
                [5, 12],
                16,
                22,
                max_norm=2.5,
            )
            self._run_sharded_embedding(spec, [6, 7, 6], 64, 11, padding_idx=30)
            self._run_sharded_embedding(
                spec,
                [6, 5, 3],
                26,
                11,
                max_norm=2.0,
            )

            # 测试不均匀分片
            self._run_sharded_embedding(spec, [8, 6, 5, 4], 19, 11)
            self._run_sharded_embedding(spec, [6, 7, 6], 21, 11)
            self._run_sharded_embedding(spec, [4], 21, 11)
            self._run_sharded_embedding(spec, [8, 6, 5, 4], 21, 11, padding_idx=10)
            self._run_sharded_embedding(
                spec,
                [6, 5, 8],
                28,
                5,
                max_norm=2.0,
            )
            self._run_sharded_embedding(spec, [4], 14, 11, max_norm=2.5)
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```