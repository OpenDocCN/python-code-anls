# `.\pytorch\test\distributed\_shard\sharded_tensor\ops\test_init.py`

```
# Owner(s): ["oncall: distributed"]  # 代码维护者信息

import sys  # 导入sys模块，用于与Python解释器进行交互

import torch  # 导入PyTorch库

from torch.distributed._shard import sharded_tensor  # 导入sharded_tensor模块
from torch.distributed._shard.sharding_spec import ChunkShardingSpec  # 导入ChunkShardingSpec类
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu  # 导入测试相关的函数和装饰器
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 导入测试工具函数和标记
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,  # 导入ShardedTensorTestBase类
    with_comms,  # 导入with_comms装饰器
)

if TEST_WITH_DEV_DBG_ASAN:  # 如果测试标记TEST_WITH_DEV_DBG_ASAN为True
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,  # 打印警告信息到标准错误流
    )
    sys.exit(0)  # 终止程序运行，返回状态码0

class TestShardedTensorNNInit(ShardedTensorTestBase):
    """Testing torch.nn.init functions for ShardedTensor"""

    @with_comms  # 使用with_comms装饰器，设置通信环境
    @skip_if_lt_x_gpu(4)  # 如果GPU数量少于4，则跳过测试
    @requires_nccl()  # 需要使用NCCL库进行分布式训练
    def test_init_sharded_tensor_with_uniform(self):
        """Test torch.nn.init.uniform_(ShardedTensor, a, b)"""

        spec = ChunkShardingSpec(  # 创建ChunkShardingSpec对象，定义张量分片规格
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 8, 2  # 定义张量的高度和宽度
        expected_h = 2  # 预期的高度
        expected_device = torch.device(f"cuda:{self.rank}")  # 预期的设备
        a, b = 10, 20  # uniform分布的参数

        seed = 1234  # 随机种子
        dtype = torch.double  # 张量的数据类型

        st = sharded_tensor.empty(spec, h, w, dtype=dtype)  # 创建空的ShardedTensor对象
        self.assertEqual(1, len(st.local_shards()))  # 断言本地分片数量为1

        # Clone local tensor to ensure torch.nn.init starts from the same input
        local_tensor_clone = torch.clone(st.local_shards()[0].tensor)  # 克隆本地张量用于初始化
        torch.manual_seed(seed)  # 设置随机种子
        torch.nn.init.uniform_(st, a=a, b=b)  # 使用uniform分布初始化张量

        torch.manual_seed(seed)  # 重新设置相同的随机种子
        torch.nn.init.uniform_(local_tensor_clone, a=a, b=b)  # 使用uniform分布初始化克隆的本地张量
        self.assertEqual(local_tensor_clone, st.local_shards()[0].tensor)  # 断言克隆的张量与ShardedTensor的本地分片张量相等

    @with_comms  # 使用with_comms装饰器，设置通信环境
    @skip_if_lt_x_gpu(4)  # 如果GPU数量少于4，则跳过测试
    @requires_nccl()  # 需要使用NCCL库进行分布式训练
    def test_init_sharded_tensor_with_normal(self):
        """Test torch.nn.init.normal_(ShardedTensor, mean, std)"""

        spec = ChunkShardingSpec(  # 创建ChunkShardingSpec对象，定义张量分片规格
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 8, 2  # 定义张量的高度和宽度
        expected_h = 2  # 预期的高度
        expected_device = torch.device(f"cuda:{self.rank}")  # 预期的设备
        mean, std = 10, 5  # normal分布的参数

        seed = 1234  # 随机种子
        dtype = torch.double  # 张量的数据类型

        st = sharded_tensor.empty(spec, h, w, dtype=dtype)  # 创建空的ShardedTensor对象
        self.assertEqual(1, len(st.local_shards()))  # 断言本地分片数量为1

        # Clone local tensor to ensure torch.nn.init starts from the same input
        local_tensor_clone = torch.clone(st.local_shards()[0].tensor)  # 克隆本地张量用于初始化
        torch.manual_seed(seed)  # 设置随机种子
        torch.nn.init.normal_(st, mean=mean, std=std)  # 使用normal分布初始化张量

        torch.manual_seed(seed)  # 重新设置相同的随机种子
        torch.nn.init.normal_(local_tensor_clone, mean=mean, std=std)  # 使用normal分布初始化克隆的本地张量
        self.assertEqual(local_tensor_clone, st.local_shards()[0].tensor)  # 断言克隆的张量与ShardedTensor的本地分片张量相等

    @with_comms  # 使用with_comms装饰器，设置通信环境
    # 使用装饰器跳过如果 GPU 数量少于 4 的情况下运行此测试
    @skip_if_lt_x_gpu(4)
    # 要求使用 NCCL 库进行初始化设置
    @requires_nccl()
    # 定义测试函数，初始化带有 Kaiming 均匀分布的分片张量
    def test_init_sharded_tensor_with_kaiming_uniform(self):
        """Test torch.nn.init.kaiming_uniform_(ShardedTensor, a, mode, nonlinearity)"""

        # 定义分片规格
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 设置分片张量的高度和宽度
        h, w = 8, 2
        # 预期的高度为 2
        expected_h = 2
        # 期望的设备为当前 rank 对应的 CUDA 设备
        expected_device = torch.device(f"cuda:{self.rank}")
        # Kaiming 均匀分布的参数设置
        a, mode, nonlinearity = 0, "fan_in", "leaky_relu"

        # 设置随机种子
        seed = 1234
        # 设置张量类型为双精度
        dtype = torch.double

        # 创建一个空的分片张量对象
        st = sharded_tensor.empty(spec, h, w, dtype=dtype)
        # 断言本地分片数量为 1
        self.assertEqual(1, len(st.local_shards()))

        # 克隆本地张量，以确保 torch.nn.init 从相同的输入开始
        local_tensor_clone = torch.clone(st.local_shards()[0].tensor)
        # 设置随机种子
        torch.manual_seed(seed)
        # 对分片张量应用 Kaiming 均匀初始化
        torch.nn.init.kaiming_uniform_(st, a=a, mode=mode, nonlinearity=nonlinearity)

        # 重置随机种子
        torch.manual_seed(seed)
        # 对本地张量克隆应用 Kaiming 均匀初始化
        torch.nn.init.kaiming_uniform_(
            local_tensor_clone, a=a, mode=mode, nonlinearity=nonlinearity
        )
        # 断言本地张量克隆与更新后的分片张量相等
        self.assertEqual(local_tensor_clone, st.local_shards()[0].tensor)
# 如果当前脚本作为主程序执行（而不是被导入到其他模块中执行），则运行 run_tests 函数
if __name__ == "__main__":
    run_tests()
```