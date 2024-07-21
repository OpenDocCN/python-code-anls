# `.\pytorch\test\distributed\_shard\sharded_tensor\ops\test_tensor_ops.py`

```
# Owner(s): ["oncall: distributed"]

# 导入所需的模块和函数
import copy  # 导入深拷贝函数
import torch  # 导入PyTorch库
import torch.distributed._shard.sharded_tensor as sharded_tensor  # 导入分片张量相关模块

# 导入分片策略和测试相关函数
from torch.distributed._shard.sharding_spec import ChunkShardingSpec  # 导入分片规格类
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu  # 导入测试所需的装饰器
from torch.testing._internal.common_utils import run_tests  # 导入运行测试的函数

# 导入分片张量测试基类和测试所需的通信装饰器
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,  # 分片张量测试基类
    TEST_GPU_NUM,  # 测试所用的 GPU 数量
    with_comms,  # 通信装饰器
)


class TestTensorOps(ShardedTensorTestBase):
    @with_comms(init_rpc=False)  # 使用通信装饰器，初始化 RPC 为 False
    @skip_if_lt_x_gpu(TEST_GPU_NUM)  # 如果 GPU 数量小于指定数量，则跳过测试
    @requires_nccl()  # 需要 NCCL 库支持
    def test_deep_copy(self):
        # 定义分片策略：按第一维分片到不同的 CUDA 设备上
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 创建随机初始化的分片张量
        st = sharded_tensor.rand(spec, (12, 5))
        # 深拷贝分片张量
        copied_st = copy.deepcopy(st)
        # 断言拷贝后的对象类型与原对象类型相同
        self.assertTrue(type(copied_st) is type(st))
        # 断言拷贝后的本地张量与原对象的本地张量相等
        self.assertEqual(copied_st.local_tensor(), st.local_tensor())
        # 断言拷贝后的对象不是原对象的同一引用
        self.assertFalse(copied_st is st)

    @with_comms(init_rpc=False)  # 使用通信装饰器，初始化 RPC 为 False
    @skip_if_lt_x_gpu(TEST_GPU_NUM)  # 如果 GPU 数量小于指定数量，则跳过测试
    @requires_nccl()  # 需要 NCCL 库支持
    def test_inplace_copy(self):
        # 定义分片策略：按第一维分片到不同的 CUDA 设备上
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 创建随机初始化的分片张量
        st = sharded_tensor.rand(spec, (12, 5))
        # 创建全为1的分片张量
        ones_st = sharded_tensor.ones(spec, (12, 5))
        # 断言全为1的张量与随机张量不相等
        self.assertFalse(torch.equal(ones_st, st))
        # 将全为1的张量拷贝到随机张量中（原地操作）
        st.copy_(ones_st)
        # 断言两个张量相等
        self.assertTrue(torch.equal(st, ones_st))

        # 对于两个具有不同 requires_grad 属性的张量，使用无梯度上下文下的原地拷贝
        st_with_grad = sharded_tensor.rand(spec, (12, 5), requires_grad=True)
        self.assertTrue(st_with_grad.requires_grad)
        self.assertFalse(ones_st.requires_grad)
        with torch.no_grad():  # 使用无梯度上下文
            st_with_grad.copy_(ones_st)
            # 断言拷贝后的本地张量与全为1的本地张量相等
            self.assertEqual(st_with_grad.local_tensor(), ones_st.local_tensor())

    @with_comms(init_rpc=False)  # 使用通信装饰器，初始化 RPC 为 False
    @skip_if_lt_x_gpu(TEST_GPU_NUM)  # 如果 GPU 数量小于指定数量，则跳过测试
    @requires_nccl()  # 需要 NCCL 库支持
    def test_clone(self):
        # 定义分片策略：按第一维分片到不同的 CUDA 设备上
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 创建随机初始化的分片张量
        st = sharded_tensor.rand(spec, (12, 5))
        # 克隆分片张量
        copied_st = st.clone()
        # 断言克隆后的对象类型与原对象类型相同
        self.assertTrue(type(copied_st) is type(st))
        # 断言克隆后的本地张量与原对象的本地张量相等
        self.assertEqual(copied_st.local_tensor(), st.local_tensor())
        # 断言克隆后的对象不是原对象的同一引用
        self.assertFalse(copied_st is st)
    # 定义一个测试方法 `test_set_requires_grad`，用于测试设置梯度要求的功能
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_set_requires_grad(self):
        # 定义分片规格 `spec`，在第0维进行分片，并指定每个分片的放置位置
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 创建一个随机初始化的分片张量 `st`，使用 `spec` 定义的分片规格，形状为 (12, 5)
        st = sharded_tensor.rand(spec, (12, 5))
        # 获取 `st` 的所有本地分片
        local_shards = st.local_shards()
        # 在设置梯度要求之前，所有本地分片不应要求梯度
        for local_shard in local_shards:
            self.assertFalse(local_shard.tensor.requires_grad)

        # 将 `st` 设置为要求梯度
        st.requires_grad_()
        # 验证 `st` 确实要求梯度
        self.assertTrue(st.requires_grad)

        # 验证所有本地分片现在都要求梯度
        for local_shard in local_shards:
            self.assertTrue(local_shard.tensor.requires_grad)
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```