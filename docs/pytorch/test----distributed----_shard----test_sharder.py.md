# `.\pytorch\test\distributed\_shard\test_sharder.py`

```py
# Owner(s): ["oncall: distributed"]
# 导入必要的库和模块
import copy  # 导入copy模块，用于对象的复制操作
import sys  # 导入sys模块，提供对解释器相关功能的访问

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch神经网络模块
from torch.distributed._shard import shard_module  # 导入分片模块函数
from torch.distributed._shard.sharded_tensor import ShardedTensor  # 导入分片张量类
from torch.distributed._shard.sharder import Sharder  # 导入分片器类
from torch.distributed._shard.sharding_plan import ShardingPlan  # 导入分片计划类
from torch.distributed._shard.sharding_spec import ChunkShardingSpec  # 导入分片规格类
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu  # 导入分布式测试相关函数

from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 导入测试相关工具函数和变量
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,  # 导入分片张量测试基类
    TEST_GPU_NUM,  # 导入测试用GPU数量
    with_comms,  # 导入用于通信测试的装饰器
)

if TEST_WITH_DEV_DBG_ASAN:
    # 如果处于开发ASAN模式，则输出警告信息并退出
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


# 定义一个简单的嵌入袋集合类
class CustomEmbeddingBagCollection(nn.Module):
    def __init__(self, num_bags, num_embeddings_per_bag, num_dims):
        super().__init__()
        self.num_bags = num_bags
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()

        # 根据参数创建多个嵌入袋（EmbeddingBag）模块，并加入到ModuleDict中
        for i in range(num_bags):
            self.embedding_bags[f"embedding_bag_{i}"] = nn.EmbeddingBag(
                num_embeddings_per_bag, num_dims, mode="sum"
            )

    def forward(self, inputs):
        outputs = []
        for bag in self.embedding_bags.values():
            outputs.append(bag(inputs))
        # 将所有嵌入袋的输出拼接起来并返回
        return torch.cat(outputs)


# 定义一个简单的分片版嵌入袋集合类
class CustomShardedEBC(nn.Module):
    def __init__(self, ebc, split_idx, specs):
        super().__init__()
        self.split_idx = split_idx
        row_spec, col_spec = specs

        # 根据规格创建嵌入袋，并根据分片索引对其进行分片
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()

        assert self.split_idx < ebc.num_bags  # 断言分片索引小于原始嵌入袋数量
        for i in range(ebc.num_bags):
            bag_key = f"embedding_bag_{i}"
            if i < self.split_idx:
                # 对嵌入袋模块进行行分片
                shard_module(
                    ebc,
                    plan=ShardingPlan(
                        plan={f"embedding_bags.{bag_key}.weight": row_spec}
                    ),
                )
            else:
                # 对嵌入袋模块进行列分片
                shard_module(
                    ebc,
                    plan=ShardingPlan(
                        plan={f"embedding_bags.{bag_key}.weight": col_spec}
                    ),
                )

            self.embedding_bags[bag_key] = ebc.embedding_bags[bag_key]


# 定义一个自定义的分片器类，继承自Sharder
class CustomSharder(Sharder):
    def __init__(self, devices, split_sharding_idx):
        self.devices = devices
        self.split_sharding_idx = split_sharding_idx
        # 创建行分片和列分片的规格对象
        self.rowwise_spec = ChunkShardingSpec(dim=0, placements=devices)
        self.colwise_spec = ChunkShardingSpec(dim=1, placements=devices)
    # 定义一个方法 `shard`，用于对嵌入袋（embedding bag）进行分片
    def shard(self, ebc: nn.Module) -> nn.Module:
        # 如果输入的 `ebc` 不是 `CustomEmbeddingBagCollection` 类型，抛出运行时错误
        if not isinstance(ebc, CustomEmbeddingBagCollection):
            raise RuntimeError(
                "The custom sharder only supports CustomEmbeddingBagCollection"
            )

        # 返回一个 `CustomShardedEBC` 对象，将 `ebc` 和分片索引 `(self.split_sharding_idx, (self.rowwise_spec, self.colwise_spec)` 作为参数传入
        return CustomShardedEBC(
            ebc, self.split_sharding_idx, (self.rowwise_spec, self.colwise_spec)
        )
    # 定义一个测试类 TestCustomSharder，继承自 ShardedTensorTestBase，用于测试自定义分片器功能
    class TestCustomSharder(ShardedTensorTestBase):
        
        # 装饰器：使用通信功能，但不初始化 RPC，跳过如果 GPU 数量小于 TEST_GPU_NUM
        @with_comms(init_rpc=False)
        @skip_if_lt_x_gpu(TEST_GPU_NUM)
        @requires_nccl()
        # 定义测试方法 test_custom_sharder
        def test_custom_sharder(self):
            # 定义内部类 MyModule，继承自 nn.Module，用于测试自定义模块
            class MyModule(nn.Module):
                # 初始化方法
                def __init__(self):
                    super().__init__()
                    # 创建自定义的 EmbeddingBag 集合对象，参数为 10, 10, 8
                    self.ebc = CustomEmbeddingBagCollection(10, 10, 8)

                # 前向传播方法
                def forward(self, inputs):
                    # 调用自定义 EmbeddingBag 集合对象的前向传播方法
                    return self.ebc(inputs)

            # 创建自定义的分片器对象 CustomSharder
            custom_sharder = CustomSharder(
                # 指定设备列表，每个设备为 "rank:i/cuda:i"，i 从 0 到 TEST_GPU_NUM-1
                devices=[f"rank:{i}/cuda:{i}" for i in range(TEST_GPU_NUM)],
                # 分片索引为 TEST_GPU_NUM 的一半
                split_sharding_idx=TEST_GPU_NUM // 2,
            )

            # 创建分片计划对象 ShardingPlan
            sharding_plan = ShardingPlan(
                # 指定分片计划，其中键为 "ebc"，值为 custom_sharder 分片器对象
                plan={
                    "ebc": custom_sharder,
                }
            )

            # 创建本地模型对象 MyModule 的实例，并移动到当前 GPU 的设备上
            local_model = MyModule().cuda(self.rank)
            # 深度复制本地模型对象，得到分片后的模型对象
            sharded_model = copy.deepcopy(local_model)

            # 使用给定的分片计划对模型进行分片
            shard_module(sharded_model, sharding_plan)

            # 检查分片后的模型是否已经正确分片
            emb_bags = sharded_model.ebc.embedding_bags
            # 断言 embedding_bag_0 的权重是否为 ShardedTensor 类型
            self.assertTrue(isinstance(emb_bags["embedding_bag_0"].weight, ShardedTensor))
            # 断言 embedding_bag_9 的权重是否为 ShardedTensor 类型
            self.assertTrue(isinstance(emb_bags["embedding_bag_9"].weight, ShardedTensor))
            # 断言 embedding_bag_0 的分片规格是否与 custom_sharder 的行规格相同
            self.assertEqual(
                emb_bags["embedding_bag_0"].weight.sharding_spec(),
                custom_sharder.rowwise_spec,
            )
            # 断言 embedding_bag_9 的分片规格是否与 custom_sharder 的列规格相同
            self.assertEqual(
                emb_bags["embedding_bag_9"].weight.sharding_spec(),
                custom_sharder.colwise_spec,
            )

            # 确保可以运行分片计算，并与本地模型版本的输出进行比较
            # 创建输入张量 input，并移动到当前 GPU 的设备上
            input = torch.arange(8).reshape((2, 4)).cuda(self.rank)
            # 在本地模型上运行输入并得到输出
            local_output = local_model(input)
            # 在分片模型上运行输入并得到输出
            sharded_output = sharded_model(input)

            # 断言本地输出与分片输出是否相等
            self.assertEqual(local_output, sharded_output)
    # 定义一个测试方法，用于测试自定义分片器的错误情况
    def test_custom_sharder_errors(self):
        # 创建一个自定义分片器对象，指定设备列表为多个GPU的设备地址，并设置分片索引
        custom_sharder = CustomSharder(
            devices=[f"rank:{i}/cuda:{i}" for i in range(TEST_GPU_NUM)],
            split_sharding_idx=TEST_GPU_NUM // 2,
        )

        # 创建一个分片计划对象，将空字符串映射到自定义分片器
        sharding_plan = ShardingPlan(
            plan={
                "": custom_sharder,
            }
        )

        # 创建一个定制的嵌入包集合对象，并将其移到指定的GPU上
        sharded_model = CustomEmbeddingBagCollection(10, 10, 8).cuda(self.rank)

        # 使用断言检查是否会抛出指定的 KeyError 异常，验证自定义分片器路径不能为空
        with self.assertRaisesRegex(
            KeyError, "path must not be empty for custom sharder!"
        ):
            # 使用提供的分片计划对模型进行分片
            shard_module(sharded_model, sharding_plan)

        # 测试冲突的分片计划情况
        # 创建一个分片规格对象，指定分片的维度和放置策略
        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:0", "rank:1/cuda:1"])
        # 创建另一个分片计划对象，将特定的权重路径映射到分片规格，同时将嵌入包映射到自定义分片器
        sharding_plan = ShardingPlan(
            plan={
                "embedding_bags.embedding_bag_0.weight": spec,
                "embedding_bags": custom_sharder,
            }
        )

        # 使用断言检查是否会抛出指定的 RuntimeError 异常，验证分片计划不应与子模块树冲突
        with self.assertRaisesRegex(
            RuntimeError, "should not conflict with the submodule tree"
        ):
            # 使用提供的分片计划对模型进行分片
            shard_module(sharded_model, sharding_plan)
# 如果当前脚本作为主程序执行（而非被导入其他模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```