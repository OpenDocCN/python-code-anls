# `.\pytorch\test\distributed\_shard\sharded_tensor\ops\test_embedding_bag.py`

```
# Owner(s): ["oncall: distributed"]  # 拥有者信息，指明代码归属的责任人员

import sys  # 导入系统模块

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式模块
from torch.distributed._shard import shard_parameter  # 导入分片参数函数
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu  # 导入测试相关的分布式工具函数
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 导入运行测试的工具函数和测试标志
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    TEST_GPU_NUM,
    with_comms,
)  # 导入分片张量测试基类、测试使用的GPU数量和通信装饰器
from torch.testing._internal.distributed._shard.sharded_tensor._test_ops_common import (
    clone_module_parameter,
    generate_chunk_sharding_specs_for_test,
    generate_local_weight_sharding_params_for_test,
)  # 导入模块参数克隆函数、用于测试的分块分片规格生成函数、用于测试的本地权重分片参数生成函数

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )  # 如果测试标志为True，打印相关跳过消息到标准错误流
    sys.exit(0)  # 退出程序，返回状态码0

class TestShardedEmbeddingBag(ShardedTensorTestBase):
    def _run_sharded_embedding_bag(
        self,
        spec,
        input_size,
        num_embeddings,
        embedding_dim,
        mode,
        include_last_offset=False,
        offset_size=None,
        max_norm=None,
        norm_type=2.0,
        padding_idx=None,
    ):  # 定义一个方法用于运行分片嵌入包测试，参数包括分片规格、输入尺寸、嵌入数量、嵌入维度、模式等

    @with_comms(init_rpc=False)  # 使用通信装饰器，禁止初始化RPC
    @skip_if_lt_x_gpu(TEST_GPU_NUM)  # 使用GPU数小于测试GPU数的跳过装饰器
    @requires_nccl()  # 使用需要NCCL的装饰器
    def test_sharded_embedding_bag_colwise(self):  # 定义列方向分片嵌入包测试方法
        for spec in generate_chunk_sharding_specs_for_test(1):  # 对于生成的测试用的块分片规格列表中的每个规格
            self._test_sharded_embedding_bag_with_test_cases(spec, 1)  # 调用测试分片嵌入包的具体测试案例方法，指定规格和1作为参数

    @with_comms(init_rpc=False)  # 使用通信装饰器，禁止初始化RPC
    @skip_if_lt_x_gpu(TEST_GPU_NUM)  # 使用GPU数小于测试GPU数的跳过装饰器
    @requires_nccl()  # 使用需要NCCL的装饰器
    def test_sharded_embedding_bag_rowwise(self):  # 定义行方向分片嵌入包测试方法
        for spec in generate_chunk_sharding_specs_for_test(0):  # 对于生成的测试用的块分片规格列表中的每个规格
            self._test_sharded_embedding_bag_with_test_cases(spec, 0)  # 调用测试分片嵌入包的具体测试案例方法，指定规格和0作为参数

if __name__ == "__main__":
    run_tests()  # 如果作为主程序运行，执行所有测试
```