# `.\pytorch\test\distributed\checkpoint\test_utils.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import sys
from unittest.mock import MagicMock

import torch

# 导入分布式张量相关的类和函数
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    ShardMetadata,
)
from torch.distributed._shard.sharded_tensor.metadata import TensorProperties
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.utils import find_state_dict_object

# 导入测试工具函数和类
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)
from torch.testing._internal.distributed.distributed_utils import with_fake_comms

# 如果设置了测试标志 TEST_WITH_DEV_DBG_ASAN，输出相应信息并退出
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 定义创建分片张量的函数，根据给定的 rank、world_size 和 shards_per_rank 参数生成分片张量
def create_sharded_tensor(rank, world_size, shards_per_rank):
    # 初始化空列表来存储分片的元数据和本地分片
    shards_metadata = []
    local_shards = []
    
    # 循环创建所有分片的元数据和对应的本地分片
    for idx in range(0, world_size * shards_per_rank):
        # 计算当前分片所属的 rank
        shard_rank = idx // shards_per_rank
        # 创建分片的元数据对象
        shard_md = ShardMetadata(
            shard_offsets=[idx * 8], shard_sizes=[8], placement=f"rank:{shard_rank}/cpu"
        )
        shards_metadata.append(shard_md)
        
        # 如果当前分片属于当前 rank，则生成本地分片数据并添加到 local_shards 列表中
        if shard_rank == rank:
            shard = Shard.from_tensor_and_offsets(
                torch.rand(*shard_md.shard_sizes),
                shard_offsets=shard_md.shard_offsets,
                rank=rank,
            )
            local_shards.append(shard)

    # 创建分片张量的元数据对象
    sharded_tensor_md = ShardedTensorMetadata(
        shards_metadata=shards_metadata,
        size=torch.Size([8 * len(shards_metadata)]),
        tensor_properties=TensorProperties.create_from_tensor(torch.zeros(1)),
    )

    # 使用本地分片和全局元数据初始化 ShardedTensor 对象并返回
    return ShardedTensor._init_from_local_shards_and_global_metadata(
        local_shards=local_shards, sharded_tensor_metadata=sharded_tensor_md
    )


# 定义测试类 TestMedatadaIndex，继承自 TestCase 类
class TestMedatadaIndex(TestCase):
    # 测试初始化时的偏移转换
    def test_init_convert_offset(self):
        a = MetadataIndex("foo", [1, 2])  # 创建 MetadataIndex 对象 a
        b = MetadataIndex("foo", torch.Size([1, 2]))  # 创建 MetadataIndex 对象 b
        self.assertEqual(a, b)  # 断言 a 和 b 相等

    # 测试在相等比较时忽略 index 提示
    def test_index_hint_ignored_on_equals(self):
        a = MetadataIndex("foo")  # 创建 MetadataIndex 对象 a
        b = MetadataIndex("foo", index=99)  # 创建 MetadataIndex 对象 b，并指定 index
        self.assertEqual(a, b)  # 断言 a 和 b 相等

    # 测试在哈希比较时忽略 index 提示
    def test_index_hint_ignored_on_hash(self):
        a = MetadataIndex("foo")  # 创建 MetadataIndex 对象 a
        b = MetadataIndex("foo", index=99)  # 创建 MetadataIndex 对象 b，并指定 index
        self.assertEqual(hash(a), hash(b))  # 断言 a 和 b 的哈希值相等
    # 定义测试方法，用于测试在扁平化数据结构中查找对象的功能
    def test_flat_data(self):
        # 创建一个状态字典，包含键 "a" 和 "b"，分别对应一个随机张量和一个列表
        state_dict = {
            "a": torch.rand(10),  # 键 "a" 对应一个包含 10 个随机数的张量
            "b": [1, 2, 3],       # 键 "b" 对应一个包含整数 1, 2, 3 的列表
        }

        # 查找状态字典中键 "a" 对应的对象，应返回对应的张量
        a = find_state_dict_object(state_dict, MetadataIndex("a"))
        self.assertEqual(a, state_dict["a"])

        # 使用索引 [0] 查找状态字典中键 "a" 对应的对象，应返回对应的张量
        a = find_state_dict_object(state_dict, MetadataIndex("a", [0]))
        self.assertEqual(a, state_dict["a"])

        # 使用索引 99 查找状态字典中键 "a" 对应的对象，应返回对应的张量
        a = find_state_dict_object(state_dict, MetadataIndex("a", index=99))
        self.assertEqual(a, state_dict["a"])

        # 查找状态字典中键 "b" 对应的对象，应返回对应的列表
        b = find_state_dict_object(state_dict, MetadataIndex("b"))
        self.assertEqual(b, state_dict["b"])

        # 使用索引 1 查找状态字典中键 "b" 对应的对象，应返回对应的列表
        b = find_state_dict_object(state_dict, MetadataIndex("b", index=1))
        self.assertEqual(b, state_dict["b"])

        # 使用不存在的键 "c" 进行查找，预期会引发 ValueError 异常并提示 "FQN"
        with self.assertRaisesRegex(ValueError, "FQN"):
            find_state_dict_object(state_dict, MetadataIndex("c"))

        # 使用索引 [1] 查找键 "b" 对应的对象，预期会引发 ValueError 异常并提示 "ShardedTensor"
        with self.assertRaisesRegex(ValueError, "ShardedTensor"):
            find_state_dict_object(state_dict, MetadataIndex("b", [1]))

    # 使用装饰器定义测试方法，用于测试在分片张量查找中的对象查找功能
    @with_fake_comms(rank=0, world_size=2)
    def test_sharded_tensor_lookup(self):
        # 创建一个分片张量对象，分片数为 6 (2 个进程 * 每进程 3 个分片)
        st = create_sharded_tensor(rank=0, world_size=2, shards_per_rank=3)
        # 创建一个包含分片张量对象的状态字典
        state_dict = {"st": st}

        # 使用索引 [8] 查找状态字典中键 "st" 对应的对象，应返回分片张量的第二个本地分片
        obj = find_state_dict_object(state_dict, MetadataIndex("st", [8]))
        self.assertEqual(obj, st.local_shards()[1].tensor)

        # 使用索引 [8] 和提示索引 1 查找状态字典中键 "st" 对应的对象，应返回分片张量的第二个本地分片
        obj = find_state_dict_object(state_dict, MetadataIndex("st", [8], index=1))
        self.assertEqual(obj, st.local_shards()[1].tensor)

        # 使用索引 [8] 和不存在的提示索引 2 查找状态字典中键 "st" 对应的对象，应返回分片张量的第二个本地分片
        obj = find_state_dict_object(state_dict, MetadataIndex("st", [8], index=2))
        self.assertEqual(obj, st.local_shards()[1].tensor)

        # 使用索引 [8] 和不存在的提示索引 99 查找状态字典中键 "st" 对应的对象，应返回分片张量的第二个本地分片
        obj = find_state_dict_object(state_dict, MetadataIndex("st", [8], index=99))
        self.assertEqual(obj, st.local_shards()[1].tensor)

        # 使用未提供偏移量的索引查找键 "st" 对应的对象，预期会引发 ValueError 异常并提示 "no offset was provided"
        with self.assertRaisesRegex(ValueError, "no offset was provided"):
            find_state_dict_object(state_dict, MetadataIndex("st"))

        # 使用索引 [1] 查找键 "st" 对应的对象，预期会引发 ValueError 异常并提示 "Could not find shard"
        with self.assertRaisesRegex(ValueError, "Could not find shard"):
            find_state_dict_object(state_dict, MetadataIndex("st", [1]))
class TestTensorProperties(TestCase):
    # 定义一个测试类 TestTensorProperties，继承自 TestCase

    def test_create_from_tensor_correct_device(self):
        # 定义一个测试方法 test_create_from_tensor_correct_device

        # 创建一个形状为 [10, 2] 的随机张量，放置在 CPU 设备上
        t = torch.randn([10, 2], device="cpu")

        # 设置 t.is_pinned 方法为一个 MagicMock 对象，模拟返回 True
        t.is_pinned = MagicMock(return_value=True)

        # 调用 TensorProperties 类的静态方法 create_from_tensor，传入张量 t
        TensorProperties.create_from_tensor(t)

        # 断言 t.is_pinned 方法被调用，并且传入的参数是 torch.device("cpu")
        t.is_pinned.assert_called_with(device=torch.device("cpu"))


if __name__ == "__main__":
    # 如果当前脚本是主程序入口

    run_tests()
    # 运行测试
```