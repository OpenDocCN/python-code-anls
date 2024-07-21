# `.\pytorch\test\distributed\_shard\sharding_spec\test_sharding_spec.py`

```
# 导入必要的模块和类
import copy
from dataclasses import dataclass
from typing import List, Union

import torch
# 从torch.distributed._shard模块导入_shard_tensor和sharded_tensor
from torch.distributed._shard import _shard_tensor, sharded_tensor
# 从torch.distributed._shard.sharded_tensor模块导入ShardedTensor、ShardedTensorMetadata和TensorProperties类
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    ShardedTensorMetadata,
    TensorProperties,
)
# 从torch.distributed._shard.sharding_spec模块导入多个类和函数
from torch.distributed._shard.sharding_spec import (
    _infer_sharding_spec_from_shards_metadata,
    ChunkShardingSpec,
    DevicePlacementSpec,
    EnumerableShardingSpec,
    ShardingSpec,
    ShardMetadata,
)
# 从torch.distributed._shard.sharding_spec._internals模块导入多个函数
from torch.distributed._shard.sharding_spec._internals import (
    check_tensor,
    get_chunk_sharding_params,
    get_chunked_dim_size,
    get_split_size,
    validate_non_overlapping_shards_metadata,
)
# 从torch.testing._internal.common_cuda模块导入TEST_MULTIGPU变量
from torch.testing._internal.common_cuda import TEST_MULTIGPU
# 从torch.testing._internal.common_distributed模块导入多个函数和类
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
# 从torch.testing._internal.common_utils模块导入多个函数和类
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TestCase,
)
# 从torch.testing._internal.distributed._shard.sharded_tensor模块导入ShardedTensorTestBase和with_comms函数
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
# 从torch.testing._internal.distributed._shard.sharded_tensor._test_st_common模块导入_chunk_sharding_specs_list_for_test函数
from torch.testing._internal.distributed._shard.sharded_tensor._test_st_common import (
    _chunk_sharding_specs_list_for_test,
)


class TestShardingSpec(TestCase):
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "2 CUDA GPUs are needed")
    # 测试设备放置规范
    def test_device_placement(self):
        # 验证有效设备
        DevicePlacementSpec("cuda:0")
        DevicePlacementSpec(torch.device(0))
        DevicePlacementSpec(torch.device("cuda:0"))
        DevicePlacementSpec("rank:0/cuda:0")
        DevicePlacementSpec("rank:0/cpu")
        DevicePlacementSpec("rank:0")

        # 验证无效设备
        with self.assertRaisesRegex(ValueError, "Could not parse remote_device"):
            DevicePlacementSpec("cuda:foo")
        with self.assertRaisesRegex(ValueError, "Could not parse remote_device"):
            DevicePlacementSpec("foo:0")
        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            DevicePlacementSpec("rank:0/cuda:foo")
        with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
            DevicePlacementSpec("rank:0/cpu2")

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "2 CUDA GPUs are needed")
    # 测试分块分片规范的有效性。

    # 创建 ChunkShardingSpec 实例，指定分片索引为 0，设备为 [torch.device(0), torch.device(1)]
    ChunkShardingSpec(0, [torch.device(0), torch.device(1)])
    # 创建 ChunkShardingSpec 实例，指定分片索引为 0，设备为 ["cuda:0", "cuda:1"]
    ChunkShardingSpec(0, [torch.device("cuda:0"), torch.device("cuda:1")])
    # 创建 ChunkShardingSpec 实例，指定分片索引为 -1，设备为 ["cuda:0", "cuda:1"]
    ChunkShardingSpec(-1, ["cuda:0", "cuda:1"])
    # 创建 ChunkShardingSpec 实例，指定分片索引为 0，设备为 ["rank:0/cuda:0", "rank:0/cuda:1"]
    ChunkShardingSpec(0, ["rank:0/cuda:0", "rank:0/cuda:1"])
    # 创建 ChunkShardingSpec 实例，指定分片索引为 0，设备为 ["rank:0", "rank:1"]
    ChunkShardingSpec(0, ["rank:0", "rank:1"])
    # 创建 ChunkShardingSpec 实例，指定分片索引为 0，设备为 ["rank:0/cpu", "rank:1/cpu"]
    ChunkShardingSpec(0, ["rank:0/cpu", "rank:1/cpu"])

    # 测试未实现错误情况
    with self.assertRaisesRegex(NotImplementedError, "not support named dimension"):
        # 尝试创建一个 ChunkShardingSpec 实例，指定分片索引为 "N"，设备为 ["cuda:0", "cuda:1"]
        ChunkShardingSpec("N", ["cuda:0", "cuda:1"])

    # 测试无效规范的情况
    with self.assertRaisesRegex(ValueError, "needs to be an integer"):
        # 尝试创建一个 ChunkShardingSpec 实例，分片索引为 None，设备为 ["cuda:0", "cuda:1"]
        ChunkShardingSpec(None, ["cuda:0", "cuda:1"])
    with self.assertRaisesRegex(ValueError, "needs to be an integer"):
        # 尝试创建一个 ChunkShardingSpec 实例，分片索引为 {}，设备为 ["cuda:0", "cuda:1"]
        ChunkShardingSpec({}, ["cuda:0", "cuda:1"])
    with self.assertRaisesRegex(ValueError, "Could not parse remote_device"):
        # 尝试创建一个 ChunkShardingSpec 实例，分片索引为 0，设备为 ["random:0", "cuda:1"]
        ChunkShardingSpec(0, ["random:0", "cuda:1"])
    with self.assertRaisesRegex(ValueError, "Could not parse remote_device"):
        # 尝试创建一个 ChunkShardingSpec 实例，分片索引为 0，设备为 ["cuda:foo", "cuda:1"]
        ChunkShardingSpec(0, ["cuda:foo", "cuda:1"])
    with self.assertRaisesRegex(ValueError, "Could not parse remote_device"):
        # 尝试创建一个 ChunkShardingSpec 实例，分片索引为 0，设备为 ["rank:foo", "cuda:1"]
        ChunkShardingSpec(0, ["rank:foo", "cuda:1"])
    with self.assertRaisesRegex(RuntimeError, "Expected one of"):
        # 尝试创建一个 ChunkShardingSpec 实例，分片索引为 0，设备为 ["rank:0/foo", "cuda:1"]
        ChunkShardingSpec(0, ["rank:0/foo", "cuda:1"])
    with self.assertRaisesRegex(RuntimeError, "Expected one of"):
        # 尝试创建一个 ChunkShardingSpec 实例，分片索引为 0，设备为 ["rank:0/random:0", "cuda:1"]
        ChunkShardingSpec(0, ["rank:0/random:0", "cuda:1"])
    with self.assertRaisesRegex(RuntimeError, "Invalid device string"):
        # 尝试创建一个 ChunkShardingSpec 实例，分片索引为 0，设备为 ["rank:0/cuda:foo", "cuda:1"]
        ChunkShardingSpec(0, ["rank:0/cuda:foo", "cuda:1"])
    # 测试获取分片参数的函数
    def test_get_chunk_sharding_params(self):
        # 定义节点排列列表
        ranks = [
            "rank:0/cuda:0",
            "rank:1/cuda:1",
            "rank:2/cuda:2",
            "rank:3/cuda:3",
        ]
        # 创建分片规格对象
        spec = ChunkShardingSpec(
            dim=0,
            placements=ranks,
        )
        # 调用函数计算分片参数并检查结果
        result = get_chunk_sharding_params(21, 4, spec, 1)
        self.assertEqual(6, result[0])
        self.assertEqual(6, result[1])
        # 再次调用函数计算分片参数并检查结果
        result = get_chunk_sharding_params(21, 4, spec, 3)
        self.assertEqual(18, result[0])
        self.assertEqual(3, result[1])
        # 调整排列列表中的顺序
        ranks[1], ranks[2] = ranks[2], ranks[1]
        ranks[0], ranks[3] = ranks[3], ranks[0]
        spec.placements = ranks
        # 再次调用函数计算分片参数并检查结果
        result = get_chunk_sharding_params(21, 4, spec, 1)
        self.assertEqual(12, result[0])
        self.assertEqual(6, result[1])
        # 再次调用函数计算分片参数并检查结果
        result = get_chunk_sharding_params(21, 4, spec, 3)
        self.assertEqual(0, result[0])
        self.assertEqual(6, result[1])

    # 推断枚举分片规格的情况
    def _infer_enum_sharding_spec_case(self):
        # 定义分片元数据列表
        shards_metadata = [
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[5, 5],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[5, 0],
                shard_sizes=[10, 5],
                placement="cuda:1",
            ),
        ]
        # 从分片元数据推断分片规格对象
        spec = _infer_sharding_spec_from_shards_metadata(shards_metadata)
        self.assertTrue(isinstance(spec, EnumerableShardingSpec))
        self.assertEqual(spec.shards, shards_metadata)

        # 定义另一组分片元数据列表
        shards_metadata = [
            ShardMetadata(
                shard_offsets=[0],
                shard_sizes=[16],
                placement="cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[16],
                shard_sizes=[9],
                placement="cuda:1",
            ),
        ]
        # 从分片元数据推断分片规格对象
        spec = _infer_sharding_spec_from_shards_metadata(shards_metadata)
        self.assertTrue(isinstance(spec, EnumerableShardingSpec))
        self.assertEqual(spec.shards, shards_metadata)

        # 定义另一组分片元数据列表
        shards_metadata = [
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
            ShardMetadata(
                shard_offsets=[0, 5],
                shard_sizes=[5, 5],
                placement="rank:2/cuda:2",
            ),
            ShardMetadata(
                shard_offsets=[5, 5],
                shard_sizes=[5, 5],
                placement="rank:3/cuda:3",
            ),
        ]
        # 从分片元数据推断分片规格对象
        spec = _infer_sharding_spec_from_shards_metadata(shards_metadata)
        self.assertTrue(isinstance(spec, EnumerableShardingSpec))
        self.assertEqual(spec.shards, shards_metadata)
    # 推断特定情况下的数据分片规格
    def _infer_chunk_sharding_spec_case(self, placements, sharding_dim, st_size):
        # 计算参与数据分片的进程数
        world_size = len(placements)
        # 根据数据维度的大小和进程数计算分片大小
        split_size = get_split_size(st_size[sharding_dim], world_size)
        # 初始化存储分片元数据的列表
        shards_metadata = [None] * world_size
        # 遍历每个进程及其对应的位置信息
        for idx, placement in enumerate(placements):
            # 深拷贝数据维度信息
            shard_size = copy.deepcopy(st_size)
            # 初始化偏移量列表
            offsets = [0] * len(st_size)
            # 根据分片大小和索引计算偏移量
            offsets[sharding_dim] = split_size * idx
            # 计算当前分片的维度大小
            shard_size[sharding_dim] = get_chunked_dim_size(
                st_size[sharding_dim], split_size, idx
            )
            # 将分片元数据存入对应进程的位置信息
            shards_metadata[placement.rank()] = ShardMetadata(
                shard_offsets=offsets,
                shard_sizes=shard_size,
                placement=placement,
            )

        # 从分片元数据推断数据分片规格
        spec = _infer_sharding_spec_from_shards_metadata(shards_metadata)
        # 断言推断出的规格是 ChunkShardingSpec 类型
        self.assertTrue(isinstance(spec, ChunkShardingSpec))
        # 断言推断出的规格维度与给定的分片维度相符
        self.assertEqual(spec.dim, sharding_dim)
        # 断言推断出的规格中的位置信息与给定的进程位置列表相符
        self.assertEqual(spec.placements, placements)

    # 测试从分片元数据推断数据分片规格的功能
    def test_infer_sharding_spec_from_shards_metadata(self):
        # 执行枚举分片规格推断的测试用例
        self._infer_enum_sharding_spec_case()
        # 生成用于测试的分片规格列表
        chunk_specs = _chunk_sharding_specs_list_for_test([0, 0, 1, 1], seed=31)
        # 遍历每个分片规格并进行推断分片规格的测试
        for spec in chunk_specs:
            self._infer_chunk_sharding_spec_case(spec.placements, 0, [4, 16])
            self._infer_chunk_sharding_spec_case(spec.placements, 0, [5, 15, 16])
            self._infer_chunk_sharding_spec_case(spec.placements, 1, [12, 16])
            self._infer_chunk_sharding_spec_case(spec.placements, 2, [4, 18, 15])
            self._infer_chunk_sharding_spec_case(spec.placements, 3, [7, 12, 16, 37])
            self._infer_chunk_sharding_spec_case(
                spec.placements, 4, [50, 4, 18, 15, 77]
            )
    # 定义测试函数 test_check_overlapping，用于验证非重叠的分片元数据
    def test_check_overlapping(self):
        # 定义分片元数据列表 shards，包含两个分片对象
        shards = [
            ShardMetadata(
                shard_offsets=[0, 0],  # 第一个分片的偏移量列表
                shard_sizes=[5, 5],    # 第一个分片的大小列表
                placement="cuda:0",    # 第一个分片的放置位置
            ),
            ShardMetadata(
                shard_offsets=[5, 0],  # 第二个分片的偏移量列表
                shard_sizes=[5, 5],    # 第二个分片的大小列表
                placement="cuda:1",    # 第二个分片的放置位置
            ),
        ]
        # 调用验证函数，确认分片元数据不存在重叠
        validate_non_overlapping_shards_metadata(shards)

        # 重新定义分片元数据列表 shards，这次包含有重叠的分片对象
        shards = [
            ShardMetadata(
                shard_offsets=[0, 0],  # 第一个分片的偏移量列表
                shard_sizes=[5, 5],    # 第一个分片的大小列表
                placement="cuda:0",    # 第一个分片的放置位置
            ),
            ShardMetadata(
                shard_offsets=[4, 0],  # 第二个分片的偏移量列表（存在重叠）
                shard_sizes=[5, 5],    # 第二个分片的大小列表
                placement="cuda:1",    # 第二个分片的放置位置
            ),
        ]
        # 使用断言检查是否引发值错误，并包含 "overlap" 字符串
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_non_overlapping_shards_metadata(shards)

        # 重新定义分片元数据列表 shards，这次包含有重叠的分片对象
        shards = [
            ShardMetadata(
                shard_offsets=[0, 0],  # 第一个分片的偏移量列表
                shard_sizes=[5, 5],    # 第一个分片的大小列表
                placement="cuda:0",    # 第一个分片的放置位置
            ),
            ShardMetadata(
                shard_offsets=[0, 4],  # 第二个分片的偏移量列表（存在重叠）
                shard_sizes=[5, 5],    # 第二个分片的大小列表
                placement="cuda:1",    # 第二个分片的放置位置
            ),
        ]
        # 使用断言检查是否引发值错误，并包含 "overlap" 字符串
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_non_overlapping_shards_metadata(shards)

        # 定义分片元数据列表 shards，包含三个分片对象
        shards = [
            ShardMetadata(
                shard_offsets=[5, 0, 5],  # 第一个分片的偏移量列表
                shard_sizes=[5, 5, 5],    # 第一个分片的大小列表
                placement="cuda:0",       # 第一个分片的放置位置
            ),
            ShardMetadata(
                shard_offsets=[5, 5, 5],  # 第二个分片的偏移量列表
                shard_sizes=[5, 5, 5],    # 第二个分片的大小列表
                placement="cuda:1",       # 第二个分片的放置位置
            ),
        ]
        # 调用验证函数，确认分片元数据不存在重叠
        validate_non_overlapping_shards_metadata(shards)

        # 重新定义分片元数据列表 shards，这次包含有重叠的分片对象
        shards = [
            ShardMetadata(
                shard_offsets=[5, 0, 5],  # 第一个分片的偏移量列表
                shard_sizes=[5, 5, 5],    # 第一个分片的大小列表
                placement="cuda:0",       # 第一个分片的放置位置
            ),
            ShardMetadata(
                shard_offsets=[5, 4, 5],  # 第二个分片的偏移量列表（存在重叠）
                shard_sizes=[5, 5, 5],    # 第二个分片的大小列表
                placement="cuda:1",       # 第二个分片的放置位置
            ),
        ]
        # 使用断言检查是否引发值错误，并包含 "overlap" 字符串
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_non_overlapping_shards_metadata(shards)

        # 重新定义分片元数据列表 shards，这次包含有重叠的分片对象
        shards = [
            ShardMetadata(
                shard_offsets=[5, 0, 5],   # 第一个分片的偏移量列表
                shard_sizes=[5, 5, 5],     # 第一个分片的大小列表
                placement="cuda:0",        # 第一个分片的放置位置
            ),
            ShardMetadata(
                shard_offsets=[5, 4, 9],   # 第二个分片的偏移量列表（存在重叠）
                shard_sizes=[5, 5, 5],     # 第二个分片的大小列表
                placement="cuda:1",        # 第二个分片的放置位置
            ),
        ]
        # 使用断言检查是否引发值错误，并包含 "overlap" 字符串
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_non_overlapping_shards_metadata(shards)
# 自定义网格分片规范，一个简单的示例来执行网格分片
@dataclass
class GridShardingSpec(ShardingSpec):
    grid_size: int  # 网格大小，用于定义分片的网格尺寸
    placements: List[Union[torch.distributed._remote_device, str]]  # 分片的放置位置列表，可以是远程设备或设备名称字符串的混合类型列表

    def __post_init__(self):
        # 初始化方法，在构造函数之后执行，确保所有的placements都是torch.distributed._remote_device类型
        for i, remote_device in enumerate(self.placements):
            if not isinstance(remote_device, torch.distributed._remote_device):
                self.placements[i] = torch.distributed._remote_device(remote_device)

    def build_metadata(
        self,
        tensor_sizes: torch.Size,
        tensor_properties: TensorProperties,
    ) -> ShardedTensorMetadata:
        # 构建分片元数据的方法
        tensor_num_dim = len(tensor_sizes)
        assert tensor_num_dim == 2, "only support 2-dim tensor for grid sharding"  # 断言确保仅支持二维张量进行网格分片
        shards_metadata = []

        def chunk_num(dim_size, grid_size):
            # 计算每个维度的分块数量的内部函数
            assert dim_size % grid_size == 0, "only support dim_size mod grid_size == 0"  # 断言确保维度大小能整除网格大小
            return dim_size // grid_size

        # 计算行和列的分片数量
        row_chunks = chunk_num(tensor_sizes[0], self.grid_size)
        col_chunks = chunk_num(tensor_sizes[1], self.grid_size)

        # 断言确保分片的数量与放置位置列表的长度一致
        assert row_chunks * col_chunks == len(self.placements)
        for row_idx in range(row_chunks):
            for col_idx in range(col_chunks):
                # 构建每个分片的元数据
                shards_metadata.append(
                    ShardMetadata(
                        shard_offsets=[
                            row_idx * self.grid_size,
                            col_idx * self.grid_size,
                        ],
                        shard_sizes=[self.grid_size, self.grid_size],
                        placement=self.placements[row_idx * row_chunks + col_idx],
                    )
                )
        # 返回ShardedTensorMetadata对象，包含所有分片的元数据
        return ShardedTensorMetadata(
            shards_metadata=shards_metadata,
            size=tensor_sizes,
            tensor_properties=tensor_properties,
        )

    def shard(
        self, tensor: torch.Tensor, src_rank: int = 0, process_group=None
    ) -> ShardedTensor:
        # 分片方法，抛出未实现错误，表示该方法尚未实现
        raise NotImplementedError("GridShardingSpec.shard not implemented yet!")


class TestCustomShardingSpec(ShardedTensorTestBase):
    def test_custom_sharding_spec(self):
        ranks = [
            "rank:0/cuda:0",
            "rank:1/cuda:1",
            "rank:2/cuda:2",
            "rank:3/cuda:3",
        ]

        grid_spec = GridShardingSpec(grid_size=4, placements=ranks)

        tensor_properties = TensorProperties(
            dtype=torch.get_default_dtype(),
            layout=torch.strided,
            requires_grad=False,
            memory_format=torch.contiguous_format,
            pin_memory=False,
        )

        # 构建元数据并验证分片的张量是否符合预期
        meta = grid_spec.build_metadata(torch.Size((8, 8)), tensor_properties)
        check_tensor(meta.shards_metadata, torch.Size((8, 8)))

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    # 定义一个测试函数，用于测试使用自定义网格分片规范的 sharded_tensor.ones(...) 方法
    def test_custom_sharding_spec_tensor_ctor(self):
        """Test sharded_tensor.ones(...) with the custom
        grid sharding spec.
        """

        # 定义四个节点的名称及其 GPU 设备
        ranks = [
            "rank:0/cuda:0",
            "rank:1/cuda:1",
            "rank:2/cuda:2",
            "rank:3/cuda:3",
        ]

        # 创建网格分片规范对象，指定网格大小为2，使用上述定义的节点名称
        grid_spec = GridShardingSpec(grid_size=2, placements=ranks)

        # 创建一个 sharded_tensor 对象，使用网格分片规范和指定的大小 (4, 4)，并初始化为全1张量
        st = sharded_tensor.ones(grid_spec, 4, 4)

        # 验证本地分片是否被 torch.ones 初始化
        local_shards = st.local_shards()
        self.assertEqual(1, len(local_shards))  # 确保本地分片只有一个
        local_shard = local_shards[0].tensor
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)  # 验证本地分片所在设备
        self.assertEqual((2, 2), local_shard.size())  # 验证本地分片的大小为 (2, 2)
        self.assertEqual(local_shard, torch.ones(2, 2))  # 验证本地分片是否为全1张量

    # 使用装饰器声明测试函数，测试自定义分片规范能否从 _shard_tensor 调用点被调用
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_custom_sharding_spec_shard_tensor(self):
        """Test custom spec can be invoked from the
        _shard_tensor callsite.
        """

        # 定义四个节点的名称及其 GPU 设备
        ranks = [
            "rank:0/cuda:0",
            "rank:1/cuda:1",
            "rank:2/cuda:2",
            "rank:3/cuda:3",
        ]

        # 创建网格分片规范对象，指定网格大小为2，使用上述定义的节点名称
        grid_spec = GridShardingSpec(grid_size=2, placements=ranks)

        # 使用断言验证在 _shard_tensor 调用点时，引发 NotImplementedError 异常并包含 "not implemented" 字符串
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            _shard_tensor(torch.randn(8, 8), grid_spec)
# 如果当前脚本作为主程序运行（而不是被导入到其他模块中），则执行 run_tests 函数
if __name__ == "__main__":
    run_tests()
```