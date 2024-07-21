# `.\pytorch\test\distributed\checkpoint\test_planner.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入系统相关模块
import sys

# 导入PyTorch相关模块
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn

# 导入分布式相关的ShardedTensor和Shard类
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    ShardMetadata,
)

# 导入ShardedTensor相关的属性类
from torch.distributed._shard.sharded_tensor.metadata import (
    TensorProperties as TensorProperties_Shard,
)

# 导入分布式checkpoint相关的函数和异常
from torch.distributed.checkpoint._dedup_save_plans import dedup_save_plans
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.default_planner import (
    _create_default_local_metadata,
    create_default_global_save_plan,
    create_default_local_load_plan,
    create_default_local_save_plan,
    DefaultLoadPlanner,
)

# 导入checkpoint元数据相关的类
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)

# 导入checkpoint计划相关的枚举
from torch.distributed.checkpoint.planner import LoadItemType, WriteItemType

# 导入checkpoint计划帮助函数
from torch.distributed.checkpoint.planner_helpers import (
    create_read_items_for_chunk_list,
)

# 导入测试相关的函数和类
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)

# 导入测试中的临时目录管理函数
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir

# 导入分布式测试相关的函数
from torch.testing._internal.distributed.distributed_utils import (
    with_dist,
    with_fake_comms,
)

# 检查是否使用开发模式的ASAN
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 创建分片张量的函数，返回一个ShardedTensor对象
def create_sharded_tensor(rank, world_size, shards_per_rank, shard_size=8):
    shards_metadata = []
    local_shards = []

    # 遍历所有可能的分片索引
    for idx in range(0, world_size * shards_per_rank):
        shard_rank = idx // shards_per_rank
        # 创建分片元数据对象
        shard_md = ShardMetadata(
            shard_offsets=[idx * shard_size],
            shard_sizes=[shard_size],
            placement=f"rank:{shard_rank}/cpu",
        )
        shards_metadata.append(shard_md)

        # 如果当前分片属于当前rank，创建本地分片对象并加入local_shards列表
        if shard_rank == rank:
            shard = Shard.from_tensor_and_offsets(
                torch.rand(*shard_md.shard_sizes),
                shard_offsets=shard_md.shard_offsets,
                rank=rank,
            )
            local_shards.append(shard)

    # 创建ShardedTensor的元数据对象
    sharded_tensor_md = ShardedTensorMetadata(
        shards_metadata=shards_metadata,
        size=torch.Size([shard_size * len(shards_metadata)]),
        tensor_properties=TensorProperties_Shard.create_from_tensor(torch.zeros(1)),
    )

    # 从本地分片和全局元数据创建ShardedTensor对象并返回
    return ShardedTensor._init_from_local_shards_and_global_metadata(
        local_shards=local_shards, sharded_tensor_metadata=sharded_tensor_md
    )


# 测试类，继承自TestCase
class TestSavePlan(TestCase):
    # 使用装饰器设置虚拟通信环境，rank=1，world_size=4
    @with_fake_comms(rank=1, world_size=4)
    # 定义测试方法 test_local_plan(self)，用于测试本地保存计划的生成

    # 创建一个大小为 10 的随机张量 tensor
    tensor = torch.rand(10)

    # 定义一个包含整数列表 [1, 2, 3] 的变量 val
    val = [1, 2, 3]

    # 调用函数 create_sharded_tensor 创建一个分片张量 st，
    # 参数包括 rank=1（排名）、world_size=4（世界大小）、shards_per_rank=1（每排名分片数）
    st = create_sharded_tensor(rank=1, world_size=4, shards_per_rank=1)

    # 将 tensor、val 和 st 封装在一个字典 state_dict 中
    state_dict = {"tensor": tensor, "value": val, "st": st}

    # 调用函数 create_default_local_save_plan 生成默认的本地保存计划 plan，
    # 参数包括 state_dict 和 False（表示不包括复制的项）
    plan = create_default_local_save_plan(state_dict, False)

    # 断言计划中项的数量为 3
    self.assertEqual(3, len(plan.items))

    # 获取计划的第一个项 wi
    wi = plan.items[0]

    # 断言 wi 的索引为 MetadataIndex("tensor", [0])
    self.assertEqual(wi.index, MetadataIndex("tensor", [0]))

    # 断言 wi 的类型为 WriteItemType.TENSOR
    self.assertEqual(wi.type, WriteItemType.TENSOR)

    # 断言 wi 的张量数据大小与 tensor 相同
    self.assertEqual(wi.tensor_data.size, tensor.size())

    # 断言 wi 的张量属性与全零张量相同
    self.assertEqual(
        wi.tensor_data.properties,
        TensorProperties.create_from_tensor(torch.zeros(1)),
    )

    # 断言 wi 的张量分块偏移为 torch.Size([0])
    self.assertEqual(wi.tensor_data.chunk.offsets, torch.Size([0]))

    # 断言 wi 的张量分块大小为 torch.Size([10])
    self.assertEqual(wi.tensor_data.chunk.sizes, torch.Size([10]))

    # 获取计划的第三个项 st_wi
    st_wi = plan.items[2]

    # 断言 st_wi 的索引为 MetadataIndex("st", [8])
    self.assertEqual(st_wi.index, MetadataIndex("st", [8]))

    # 断言 st_wi 的类型为 WriteItemType.SHARD
    self.assertEqual(st_wi.type, WriteItemType.SHARD)

    # 断言 st_wi 的张量数据大小与 st 相同
    self.assertEqual(st_wi.tensor_data.size, st.size())

    # 断言 st_wi 的张量属性与全零张量相同
    self.assertEqual(
        st_wi.tensor_data.properties,
        TensorProperties.create_from_tensor(torch.zeros(1)),
    )

    # 断言 st_wi 的张量分块偏移为 torch.Size([8])
    self.assertEqual(st_wi.tensor_data.chunk.offsets, torch.Size([8]))

    # 断言 st_wi 的张量分块大小为 torch.Size([8])

    # 在协调器排名上，应该包括复制的项
    plan = create_default_local_save_plan(state_dict, True)

    # 断言计划中项的数量为 3
    self.assertEqual(3, len(plan.items))

    # 获取计划中类型为 WriteItemType.TENSOR 的项 tensor_wi
    tensor_wi = next(wi for wi in plan.items if wi.type == WriteItemType.TENSOR)

    # 断言 tensor_wi 的索引为 MetadataIndex("tensor", [0])
    self.assertEqual(tensor_wi.index, MetadataIndex("tensor", [0]))

    # 断言 tensor_wi 的张量数据大小与 tensor 相同
    self.assertEqual(tensor_wi.tensor_data.size, tensor.size())

    # 断言 tensor_wi 的张量属性与 tensor 相同
    self.assertEqual(
        tensor_wi.tensor_data.properties,
        TensorProperties.create_from_tensor(tensor),
    )

    # 断言 tensor_wi 的张量分块偏移为 torch.Size([0])
    self.assertEqual(tensor_wi.tensor_data.chunk.offsets, torch.Size([0]))

    # 断言 tensor_wi 的张量分块大小为 torch.Size([10])

    # 获取计划中类型为 WriteItemType.BYTE_IO 的项 bytes_wi
    bytes_wi = next(wi for wi in plan.items if wi.type == WriteItemType.BYTE_IO)

    # 断言 bytes_wi 的索引为 MetadataIndex("value")
    self.assertEqual(bytes_wi.index, MetadataIndex("value"))

    # 断言 bytes_wi 的张量数据为 None
    self.assertIsNone(bytes_wi.tensor_data)
    # 定义一个测试方法 test_global_plan，用于测试全局保存计划的创建和验证
    def test_global_plan(self):
        # 定义内部函数 create_data，根据给定的 rank 创建数据并返回本地保存计划
        def create_data(rank):
            # 使用 with_dist 上下文管理器，指定 rank 和 world_size 为 4
            with with_dist(rank=rank, world_size=4):
                # 创建一个包含随机数的 PyTorch 张量
                tensor = torch.rand(10)
                # 创建一个简单的列表
                val = [1, 2, 3]
                # 创建一个分片张量，使用给定的 rank、world_size 和 shards_per_rank 参数
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                # 构建状态字典，包括 tensor、value 和 st 三个条目
                state_dict = {"tensor": tensor, "value": val, "st": st}
                # 创建并返回默认的本地保存计划，如果 rank == 0 则为 True
                return create_default_local_save_plan(state_dict, rank == 0)

        # 使用 create_data 函数创建四个不同的保存计划，并将其存储在列表 all_plans 中
        all_plans = [create_data(0), create_data(1), create_data(2), create_data(3)]
        # 对保存计划列表进行去重处理
        all_plans = dedup_save_plans(all_plans)
        # 创建默认的全局保存计划，返回 final_plans 和 metadata 两个结果
        final_plans, metadata = create_default_global_save_plan(all_plans=all_plans)

        # 对 final_plans 和 all_plans 进行迭代，逐个比较每个保存计划的条目
        for new_plan, old_plan in zip(final_plans, all_plans):
            # 对每个条目进行比较，包括 index、type 和 tensor_data
            for new_item, old_item in zip(new_plan.items, old_plan.items):
                # 断言新旧条目的索引和类型相等
                self.assertEqual(new_item.index, old_item.index)
                self.assertEqual(new_item.type, old_item.type)
                self.assertEqual(new_item.tensor_data, old_item.tensor_data)
                
                # 断言新条目的全局唯一标识在 metadata 的 state_dict_metadata 中存在
                self.assertIn(new_item.index.fqn, metadata.state_dict_metadata)

                # 获取当前条目的元数据 item_md
                item_md = metadata.state_dict_metadata[new_item.index.fqn]
                # 根据条目类型判断元数据类型
                if new_item.type == WriteItemType.BYTE_IO:
                    self.assertTrue(isinstance(item_md, BytesStorageMetadata))
                else:
                    self.assertTrue(isinstance(item_md, TensorStorageMetadata))
                    # 断言张量数据的大小和属性与旧条目的相同
                    self.assertEqual(item_md.size, old_item.tensor_data.size)
                    self.assertEqual(
                        item_md.properties, old_item.tensor_data.properties
                    )

                    # 确保新条目的索引 index 不为空
                    self.assertIsNotNone(new_item.index.index)
                    # 确保元数据中对应的数据块（chunk）与旧条目的数据块相匹配
                    self.assertEqual(
                        item_md.chunks[new_item.index.index], old_item.tensor_data.chunk
                    )
    # 定义单元测试方法 `test_local_load_plan`，用于测试本地加载计划的生成
    def test_local_load_plan(self):
        # 定义内部函数 `create_state_dict`，用于创建包含不同数据类型的状态字典
        def create_state_dict(rank):
            # 使用分布式上下文管理器，指定分布式参数 `rank=rank`，全局大小为4
            with with_dist(rank=rank, world_size=4):
                # 创建一个大小为10的随机张量 `tensor`
                tensor = torch.rand(10)
                # 创建一个包含数值 [1, 2, 3] 的列表 `val`
                val = [1, 2, 3]
                # 创建一个分片张量 `st`，使用参数 `rank=rank, world_size=4, shards_per_rank=1`
                st = create_sharded_tensor(rank=rank, world_size=4, shards_per_rank=1)
                # 返回包含 tensor, val, st 的字典作为状态字典
                return {"tensor": tensor, "value": val, "st": st}

        # 使用 `create_state_dict` 创建 `rank=1` 的状态字典
        state_dict = create_state_dict(1)
        # 使用 `_create_default_local_metadata` 函数基于 `state_dict` 创建元数据
        metadata = _create_default_local_metadata(state_dict)

        # 使用 `create_default_local_load_plan` 函数创建默认的本地加载计划
        load_plan = create_default_local_load_plan(state_dict, metadata)
        # 断言加载计划条目数量为3
        self.assertEqual(3, len(load_plan.items))

        # 查找加载计划中目标索引为 "st" 的条目
        st_item = next(ri for ri in load_plan.items if ri.dest_index.fqn == "st")
        # 查找加载计划中目标索引为 "tensor" 的条目
        tensor_item = next(ri for ri in load_plan.items if ri.dest_index.fqn == "tensor")
        # 查找加载计划中目标索引为 "value" 的条目
        bytes_item = next(ri for ri in load_plan.items if ri.dest_index.fqn == "value")

        # 断言 `st_item` 的类型为 `LoadItemType.TENSOR`
        self.assertEqual(st_item.type, LoadItemType.TENSOR)
        # 断言 `st_item` 的目标索引与元数据索引 ("st", [8]) 完全一致
        self.assertEqual(st_item.dest_index, MetadataIndex("st", [8]))
        # 断言 `st_item` 的目标偏移为 torch.Size([0])
        self.assertEqual(st_item.dest_offsets, torch.Size([0]))
        # 断言 `st_item` 的存储索引与元数据索引 ("st", [8]) 完全一致
        self.assertEqual(st_item.storage_index, MetadataIndex("st", [8]))
        # 断言 `st_item` 的存储偏移为 torch.Size([0])
        self.assertEqual(st_item.storage_offsets, torch.Size([0]))
        # 断言 `st_item` 的长度为 torch.Size([8])
        self.assertEqual(st_item.lengths, torch.Size([8]))

        # 断言 `tensor_item` 的类型为 `LoadItemType.TENSOR`
        self.assertEqual(tensor_item.type, LoadItemType.TENSOR)
        # 断言 `tensor_item` 的目标索引与元数据索引 ("tensor", [0]) 完全一致
        self.assertEqual(tensor_item.dest_index, MetadataIndex("tensor", [0]))
        # 断言 `tensor_item` 的目标偏移为 torch.Size([0])
        self.assertEqual(tensor_item.dest_offsets, torch.Size([0]))
        # 断言 `tensor_item` 的存储索引与元数据索引 ("tensor", [0]) 完全一致
        self.assertEqual(tensor_item.storage_index, MetadataIndex("tensor", [0]))
        # 断言 `tensor_item` 的存储偏移为 torch.Size([0])
        self.assertEqual(tensor_item.storage_offsets, torch.Size([0]))
        # 断言 `tensor_item` 的长度为 torch.Size([10])
        self.assertEqual(tensor_item.lengths, torch.Size([10]))

        # 断言 `bytes_item` 的类型为 `LoadItemType.BYTE_IO`
        self.assertEqual(bytes_item.type, LoadItemType.BYTE_IO)
        # 断言 `bytes_item` 的目标索引为元数据索引 ("value")
        self.assertEqual(bytes_item.dest_index, MetadataIndex("value"))
    def test_load_with_resharding(self):
        # 定义内部函数，用于创建状态字典，模拟不同分片和排名
        def create_state_dict(rank, world_size):
            # 使用带有指定排名和世界大小的分布上下文
            with with_dist(rank=rank, world_size=world_size):
                # 创建分片张量的状态字典
                return {
                    "st": create_sharded_tensor(
                        rank=rank,
                        world_size=world_size,
                        shards_per_rank=1,
                        shard_size=128 // world_size,
                    )
                }

        # 创建世界大小为8的状态字典，排名为1
        world8_state_dict = create_state_dict(rank=1, world_size=8)
        # 使用默认的本地元数据创建世界大小为8的元数据
        world8_metadata = _create_default_local_metadata(world8_state_dict)

        # 创建世界大小为4的状态字典，排名为1
        world4_state_dict = create_state_dict(rank=1, world_size=4)
        # 使用默认的本地元数据创建世界大小为4的元数据
        world4_metadata = _create_default_local_metadata(world4_state_dict)

        # 第一个情景，从世界大小为8到世界大小为4，需要加载2个分片
        # 每个世界大小为4的分片有32个元素，因此需要加载2个分片
        load_plan = create_default_local_load_plan(world4_state_dict, world8_metadata)
        self.assertEqual(2, len(load_plan.items))
        # 找到目标偏移为 [0] 的加载项
        low_ri = next(
            ri for ri in load_plan.items if ri.dest_offsets == torch.Size([0])
        )
        # 找到目标偏移为 [16] 的加载项
        high_ri = next(
            ri for ri in load_plan.items if ri.dest_offsets == torch.Size([16])
        )

        self.assertEqual(low_ri.storage_index, MetadataIndex("st", [32]))
        self.assertEqual(low_ri.storage_offsets, torch.Size([0]))
        self.assertEqual(low_ri.dest_index, MetadataIndex("st", [32]))
        self.assertEqual(low_ri.dest_offsets, torch.Size([0]))
        self.assertEqual(low_ri.lengths, torch.Size([16]))

        self.assertEqual(high_ri.storage_index, MetadataIndex("st", [48]))
        self.assertEqual(high_ri.storage_offsets, torch.Size([0]))
        self.assertEqual(high_ri.dest_index, MetadataIndex("st", [32]))
        self.assertEqual(high_ri.dest_offsets, torch.Size([16]))
        self.assertEqual(high_ri.lengths, torch.Size([16]))

        # 第二个情景，从世界大小为4到世界大小为8，需要加载一个分片的上半部分
        # 在世界大小为8上，rank1需要加载世界大小为4中rank0的分片的上半部分
        load_plan = create_default_local_load_plan(world8_state_dict, world4_metadata)
        self.assertEqual(1, len(load_plan.items))
        ri = load_plan.items[0]
        self.assertEqual(ri.storage_index, MetadataIndex("st", [0]))
        self.assertEqual(ri.storage_offsets, torch.Size([16]))
        self.assertEqual(ri.dest_index, MetadataIndex("st", [16]))
        self.assertEqual(ri.dest_offsets, torch.Size([0]))
        self.assertEqual(ri.lengths, torch.Size([16]))
    # 定义一个测试方法，用于测试在不同世界大小下，当排名差一个时的加载情况
    def test_load_with_world_size_diff_by_one(self):
        # 定义一个内部函数，用于创建状态字典，根据给定的排名和世界大小
        def create_state_dict(rank, world_size):
            # 使用指定的排名和世界大小上下文管理分布式环境
            with with_dist(rank=rank, world_size=world_size):
                return {
                    "st": create_sharded_tensor(
                        rank=rank,
                        world_size=world_size,
                        shards_per_rank=1,
                        shard_size=120 // world_size,
                    )
                }

        # 创建世界大小为4时的状态字典，排名为1时的情况
        world4_state_dict = create_state_dict(rank=1, world_size=4)
        # 根据world4_state_dict创建默认的本地元数据
        world4_metadata = _create_default_local_metadata(world4_state_dict)

        # 创建世界大小为3时的状态字典，排名为1时的情况
        world3_state_dict = create_state_dict(rank=1, world_size=3)

        # 根据world3_state_dict和world4_metadata创建默认的本地加载计划
        load_plan = create_default_local_load_plan(world3_state_dict, world4_metadata)

        # 断言加载计划中项目的数量为2
        self.assertEqual(2, len(load_plan.items))

        # 查找加载计划中dest_offsets为[0]的项目，范围为[30, 60]，用于加载[40, 60]
        low_ri = next(
            ri for ri in load_plan.items if ri.dest_offsets == torch.Size([0])
        )
        # 查找加载计划中dest_offsets为[20]的项目，范围为[60, 90]，用于加载[60, 80]
        high_ri = next(
            ri for ri in load_plan.items if ri.dest_offsets == torch.Size([20])
        )

        # 断言low_ri的storage_index为"st", [30]
        self.assertEqual(low_ri.storage_index, MetadataIndex("st", [30]))
        # 断言low_ri的storage_offsets为[10]
        self.assertEqual(low_ri.storage_offsets, torch.Size([10]))
        # 断言low_ri的dest_index为"st", [40]
        self.assertEqual(low_ri.dest_index, MetadataIndex("st", [40]))
        # 断言low_ri的dest_offsets为[0]
        self.assertEqual(low_ri.dest_offsets, torch.Size([0]))
        # 断言low_ri的lengths为[20]
        self.assertEqual(low_ri.lengths, torch.Size([20]))

        # 断言high_ri的storage_index为"st", [60]
        self.assertEqual(high_ri.storage_index, MetadataIndex("st", [60]))
        # 断言high_ri的storage_offsets为[0]
        self.assertEqual(high_ri.storage_offsets, torch.Size([0]))
        # 断言high_ri的dest_index为"st", [40]
        self.assertEqual(high_ri.dest_index, MetadataIndex("st", [40]))
        # 断言high_ri的dest_offsets为[20]
        self.assertEqual(high_ri.dest_offsets, torch.Size([20]))
        # 断言high_ri的lengths为[20]
        self.assertEqual(high_ri.lengths, torch.Size([20]))
class TestPlannerHelpers(TestCase):
    def test_create_read_item_from_chunks(self):
        # 创建存储元数据对象 tensor_md，包括属性和大小信息，并定义块的存储元数据列表
        tensor_md = TensorStorageMetadata(
            properties=TensorProperties.create_from_tensor(torch.empty([16])),
            size=torch.Size([16]),
            chunks=[
                ChunkStorageMetadata(offsets=torch.Size([0]), sizes=torch.Size([8])),
                ChunkStorageMetadata(offsets=torch.Size([8]), sizes=torch.Size([8])),
            ],
        )

        # 创建一个新的块存储元数据对象 chunk
        chunk = ChunkStorageMetadata(offsets=torch.Size([4]), sizes=torch.Size([7]))
        # 根据给定的块列表创建读取项 read_items
        read_items = create_read_items_for_chunk_list("foo", tensor_md, [chunk])

        # 断言读取项的数量为 2
        self.assertEqual(2, len(read_items))
        # 断言第一个读取项的目标索引和偏移
        self.assertEqual(MetadataIndex("foo", [4]), read_items[0].dest_index)
        self.assertEqual(torch.Size([0]), read_items[0].dest_offsets)

        # 断言第一个读取项的存储索引和偏移
        self.assertEqual(MetadataIndex("foo", [0]), read_items[0].storage_index)
        self.assertEqual(torch.Size([4]), read_items[0].storage_offsets)

        # 断言第一个读取项的长度
        self.assertEqual(torch.Size([4]), read_items[0].lengths)

        # 断言第二个读取项的目标索引和偏移
        self.assertEqual(MetadataIndex("foo", [4]), read_items[1].dest_index)
        self.assertEqual(torch.Size([4]), read_items[1].dest_offsets)

        # 断言第二个读取项的存储索引和偏移
        self.assertEqual(MetadataIndex("foo", [8]), read_items[1].storage_index)
        self.assertEqual(torch.Size([0]), read_items[1].storage_offsets)

        # 断言第二个读取项的长度
        self.assertEqual(torch.Size([3]), read_items[1].lengths)


class TestLoadPlanner(TestCase):
    @with_temp_dir
    def test_strict(self):
        # 创建一个原始的神经网络模块 original_module
        original_module = nn.Linear(2, 2)
        # 将原始模块保存到指定的检查点目录 self.temp_dir
        dcp.save(state_dict={"module": original_module}, checkpoint_id=self.temp_dir)

        # 创建一个新的神经网络模块 new_module，并添加额外的参数
        new_module = nn.Linear(2, 2)
        new_module.extra_param = nn.Parameter(torch.randn(2, 2))
        # 使用 DefaultLoadPlanner 加载新模块的状态字典
        dcp.load(
            state_dict={"module": new_module},
            checkpoint_id=self.temp_dir,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )

        # 使用断言验证是否抛出 CheckpointException 异常，且异常消息包含 "Missing key in checkpoint"
        with self.assertRaisesRegex(CheckpointException, "Missing key in checkpoint"):
            dcp.load(
                state_dict={"module": new_module},
                checkpoint_id=self.temp_dir,
                planner=DefaultLoadPlanner(allow_partial_load=False),
            )


if __name__ == "__main__":
    run_tests()
```