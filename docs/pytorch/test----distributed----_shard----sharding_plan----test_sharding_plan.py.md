# `.\pytorch\test\distributed\_shard\sharding_plan\test_sharding_plan.py`

```
# 导入系统模块
import sys

# 导入 PyTorch 库及分布式模块
import torch
import torch.distributed as dist
import torch.nn as nn

# 导入分片模块及其相关组件
from torch.distributed._shard import shard_module
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_plan import ShardingPlan, ShardingPlanner
from torch.distributed._shard.sharding_spec import ChunkShardingSpec

# 导入测试相关模块及函数
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    TEST_GPU_NUM,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_ops_common import (
    generate_chunk_sharding_specs_for_test,
)
from torch.testing._internal.distributed._shard.test_common import SimpleMegatronLM

# 如果设置了测试标记 TEST_WITH_DEV_DBG_ASAN，则输出相应信息并退出
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


# 示例的分片策划器，将模块中的每个参数分片到所有可用设备上
class ChunkAllShardingPlanner(ShardingPlanner):
    dim = 0
    devices = []

    # 初始化方法，设定分片的维度和设备列表
    def __init__(self, chunk_dim=0, device_count=0):
        self.dim = chunk_dim
        self.devices = [f"rank:{i}/cuda:{i}" for i in range(device_count)]

    # 构建分片计划的方法，将模块中的参数按照设定的规则进行分片
    def build_plan(self, module: nn.Module) -> ShardingPlan:
        named_params = module.named_parameters()
        plan = {}
        for name, param in named_params:
            plan[name] = ChunkShardingSpec(self.dim, placements=self.devices)

        return ShardingPlan(plan=plan)


# 测试用例，继承自 ShardedTensorTestBase 类
class TestShardingPlan(ShardedTensorTestBase):

    # 装饰器函数，用于测试前的准备工作，包括通信初始化
    @with_comms(init_rpc=False)
    # 装饰器函数，如果 GPU 数量小于测试要求的数量 TEST_GPU_NUM，则跳过测试
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    # 装饰器函数，要求使用 NCCL 库
    @requires_nccl()
    def test_sharding_plan_errors(self):
        # 生成一个用于测试的行切分规格
        rowwise_sharding_spec = generate_chunk_sharding_specs_for_test(1)[0]
        
        # 创建一个具有错误计划的ShardingPlan对象
        sharding_plan_wrong_plan = ShardingPlan(
            plan={
                "fc1.weight": torch.randn(3, 4),
            },
            output_plan={"": rowwise_sharding_spec},
        )

        # 创建一个SimpleMegatronLM模型，并将其放置在指定的GPU上
        megatron_lm = SimpleMegatronLM([[17, 12], [12, 29]]).cuda(self.rank)

        # 使用断言确保在shard模块时捕获到特定类型的错误
        with self.assertRaisesRegex(
            TypeError, "Only `ShardingSpec` and `Sharder` are supported to shard"
        ):
            # 使用错误的sharding计划来shard模块
            shard_module(megatron_lm, sharding_plan_wrong_plan)

        # 创建一个具有错误输出计划的ShardingPlan对象
        sharding_plan_wrong_output_plan = ShardingPlan(
            plan={
                "fc1.weight": rowwise_sharding_spec,
            },
            output_plan={"": torch.randn(3, 4)},
        )

        # 使用断言确保在shard模块时捕获到特定类型的错误
        with self.assertRaisesRegex(
            TypeError, "Only `ShardingSpec` is supported as output_plan"
        ):
            # 使用错误的输出计划来shard模块
            shard_module(megatron_lm, sharding_plan_wrong_output_plan)

        # 创建一个具有错误模块路径的ShardingPlan对象
        sharding_plan_wrong_module_path = ShardingPlan(
            plan={
                "fc3.weight": rowwise_sharding_spec,
            },
        )
        # 使用断言确保在shard模块时捕获到特定类型的错误
        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            # 使用错误的模块路径来shard模块
            shard_module(megatron_lm, sharding_plan_wrong_module_path)

        # 创建一个具有错误参数路径的ShardingPlan对象
        sharding_plan_wrong_param_path = ShardingPlan(
            plan={
                "fc1.biass": rowwise_sharding_spec,
            },
        )
        # 使用断言确保在shard模块时捕获到特定类型的错误
        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            # 使用错误的参数路径来shard模块
            shard_module(megatron_lm, sharding_plan_wrong_param_path)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_custom_sharding_planner(self):
        # 创建一个SimpleMegatronLM模型，并将其放置在指定的GPU上
        megatron_lm = SimpleMegatronLM([[17, 12], [12, 29]], rank=self.rank).cuda(
            self.rank
        )
        # 创建一个ChunkAllShardingPlanner对象
        planner = ChunkAllShardingPlanner(device_count=TEST_GPU_NUM)
        # 构建模型的sharding计划
        sharding_plan = planner.build_plan(megatron_lm)

        # 使用sharding计划来shard模块
        shard_module(megatron_lm, sharding_plan)

        # 检查确保模块已经被sharded
        self.assertTrue(isinstance(megatron_lm.fc1.weight, ShardedTensor))
        self.assertTrue(isinstance(megatron_lm.fc2.weight, ShardedTensor))
        self.assertTrue(isinstance(megatron_lm.fc1.bias, ShardedTensor))
        self.assertTrue(isinstance(megatron_lm.fc2.bias, ShardedTensor))

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    # 定义一个测试函数，用于测试分片模块在子进程组中的行为
    def test_shard_module_sub_process_group(self):
        # 创建一个简单的 MegatronLM 模型实例，定义模型权重的分片方式，使用当前进程的 rank
        megatron_lm = SimpleMegatronLM([[17, 12], [12, 29]], rank=self.rank)
        
        # 定义按列分片的规格，指定分片的维度为0，以及每个分片的位置信息
        colwise_sharding_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        
        # 定义按行分片的规格，指定分片的维度为1，以及每个分片的位置信息
        rowwise_sharding_spec = ChunkShardingSpec(
            dim=1,
            placements=[
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        
        # 创建一个分片计划，指定每个需要分片的模型权重参数及其对应的分片规格
        sharding_plan = ShardingPlan(
            plan={
                "fc1.weight": colwise_sharding_spec,
                "fc2.weight": rowwise_sharding_spec,
            }
        )
        
        # 创建一个新的分布式进程组，包含 rank 2 和 3 的进程
        pg = dist.new_group([2, 3])

        # 如果当前进程的 rank 大于等于 2，则对 MegatronLM 模型进行分片处理
        if self.rank >= 2:
            shard_module(megatron_lm, sharding_plan, process_group=pg)
# 如果当前脚本被直接执行（而不是被导入为模块），则执行下面的代码块
if __name__ == "__main__":
    # 调用 run_tests() 函数，这通常用于执行单元测试或程序的主要功能
    run_tests()
```