# `.\pytorch\test\distributed\checkpoint\test_file_system_checkpoint_cpu.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入系统模块和临时文件模块
import sys
import tempfile
# 导入类型提示模块
from typing import Dict

# 导入PyTorch相关模块
import torch
import torch.distributed as dist
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharded_tensor import ShardedTensor, state_dict_hook
from torch.distributed._shard.sharding_spec import (
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardingSpec,
    ShardMetadata,
)

# 导入分布式检查点相关模块
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)

# 导入测试相关工具模块
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_st_common import (
    MyShardedModel1,
)

# 如果处于开发调试模式下，则跳过使用 dev-asan，因为torch + multiprocessing spawn存在已知问题
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 定义线程计数集合
_THREAD_COUNTS = {1, 2}

# 断言两个状态字典相等的方法
def assert_state_dict_equal(
    self: TestCase,
    state_dict_1: Dict[str, torch.Tensor],
    state_dict_2: Dict[str, torch.Tensor],
) -> bool:
    # 检查状态字典的长度是否相等
    self.assertEqual(
        len(state_dict_1), len(state_dict_2), "state_dict must be the same size"
    )
    # 检查状态字典的键集合是否相等
    self.assertEqual(
        set(state_dict_1.keys()),
        set(state_dict_2.keys()),
        "state_dict keys do not match",
    )

    # 遍历比较每个键值对应的值
    for key, value_1 in state_dict_1.items():
        value_2 = state_dict_2[key]
        if isinstance(value_1, ShardedTensor):
            # 如果值是 ShardedTensor 类型，则逐个比较本地分片
            for local_shard_1, local_shard_2 in zip(
                value_1.local_shards(), value_2.local_shards()
            ):
                self.assertTrue(
                    torch.equal(local_shard_1.tensor, local_shard_2.tensor),
                    f"Key {key}'s shard does not match",
                )
        elif isinstance(value_1, torch.Tensor):
            # 如果值是 torch.Tensor 类型，则直接比较张量内容
            self.assertTrue(
                torch.equal(value_1, value_2),
                f"Key {key}'s tensor does not match",
            )

    return True


# 自定义测试模块，包含多个神经网络层
class MyTestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 5)
        self.linear_2 = torch.nn.Linear(5, 1)
        self.emb = torch.nn.EmbeddingBag(5, 10)


# 自定义分布式模型，继承自 torch.nn.Module，使用 ShardedTensor
class MyShardedModel3(torch.nn.Module):
    def __init__(
        self,
        spec: ShardingSpec,
    ) -> None:
        super().__init__()
        # 创建一个随机初始化的 ShardedTensor 对象
        self.sharded_tensor: ShardedTensor = sharded_tensor.rand(
            spec, 10, 20, init_rrefs=False
        )


# 测试分布式状态字典的保存和加载
class TestDistributedStateDictSaveLoad(TestCase):
    @parametrize("thread_count", _THREAD_COUNTS)
    # 定义测试方法，用于测试仅读写张量的情况，接受线程数作为参数
    def test_read_write_only_tensor(self, thread_count) -> None:
        # 使用临时目录作为存储路径
        with tempfile.TemporaryDirectory() as path:
            # 创建测试模块的状态字典并准备保存
            state_dict_to_save = MyTestModule().state_dict()

            # 初始化文件系统写入器，指定路径和线程数
            fs_writer = FileSystemWriter(path=path, thread_count=thread_count)
            # 调用保存状态字典的函数，传入状态字典和文件系统写入器
            save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=fs_writer,
                no_dist=True,
            )

            # 创建新的测试模块状态字典，用于加载
            state_dict_to_load_to = MyTestModule().state_dict()

            # 使用断言检查加载后的状态字典与保存的是否相等，预期会触发 AssertionError
            with self.assertRaises(AssertionError):
                assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

            # 从文件中加载状态字典，不进行任何重分片操作
            fs_reader = FileSystemReader(path=path)
            # 调用加载状态字典的函数，传入状态字典和文件系统读取器
            load_state_dict(
                state_dict=state_dict_to_load_to,
                storage_reader=fs_reader,
                no_dist=True,
            )

            # 使用断言再次验证加载后的状态字典与保存的是否相等
            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)
class TestDistributedStateDictSaveLoadWithSharedTensor(ShardedTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms(init_rpc=False, backend="gloo")
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_read_write_shard_tensor(self, thread_count) -> None:
        # 创建临时目录列表，用于存储文件路径
        paths = [tempfile.mkdtemp()]
        # 使用分布式通信广播路径列表
        dist.broadcast_object_list(paths)

        # 获取第一个路径
        path = paths[0]

        # 创建分片规格对象
        # pyre-fixme [28]: 调用 `dist._sharding_spec.api.ChunkShardingSpec.__init__` 时意外的关键字参数 `dim`。
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0",
                "rank:1",
            ],
        )

        # 创建要保存的模型对象
        model_to_save = MyShardedModel1(spec, init_rrefs=False)

        # 测试保存状态
        # 注册状态字典钩子
        model_to_save._register_state_dict_hook(state_dict_hook)
        # 获取要保存的状态字典
        state_dict_to_save = model_to_save.state_dict()

        # 创建文件系统写入器
        fs_writer = FileSystemWriter(path=path, thread_count=thread_count)
        # 保存状态字典到文件系统
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        # 分布式通信等待所有进程完成
        dist.barrier()

        # 创建新的模型
        model_to_load = MyShardedModel1(spec, init_rrefs=False)
        # 这不是加载状态字典的正确钩子
        # model_to_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)
        # 注册状态字典钩子
        model_to_load._register_state_dict_hook(state_dict_hook)
        # 获取加载后的状态字典
        state_dict_to_load_to = model_to_load.state_dict()

        # 分布式通信等待所有进程完成
        dist.barrier()

        # 断言加载后的状态字典与保存的状态字典相等
        with self.assertRaises(AssertionError):
            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

        # 测试加载
        # 创建文件系统读取器
        fs_reader = FileSystemReader(path=path)
        # 加载状态字典
        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

        # 断言加载后的状态字典与保存的状态字典相等
        assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

        # 分布式通信等待所有进程完成
        dist.barrier()


class TestDistributedReshardOnLoad(ShardedTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    def get_file_path(self) -> str:
        # 如果进程是第0个进程，则创建临时目录，否则返回None
        paths = [tempfile.mkdtemp()] if dist.get_rank() == 0 else [None]
        # 使用分布式通信广播路径列表
        dist.broadcast_object_list(paths)
        # 返回第一个路径
        return paths[0]

    def load_tensor(self, tensor: ShardedTensor) -> torch.Tensor:
        # 如果进程是第0个进程，则创建与`tensor`相同形状的零张量，否则返回None
        res = torch.zeros(tensor.shape, device="cpu") if dist.get_rank() == 0 else None
        # 将张量收集到`res`中
        tensor.gather(out=res)
        # 返回结果张量
        return res

    @with_comms(init_rpc=False, backend="gloo")
    @parametrize("thread_count", _THREAD_COUNTS)
    @with_comms(init_rpc=False, backend="gloo")
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_load_rowwise_to_colwise(self, thread_count) -> None:
        # 获取文件路径
        path = self.get_file_path()
        # 断言当前进程数量与通信库中获取的世界大小相等
        self.assertEqual(self.world_size, dist.get_world_size())

        # 创建源数据分片规格对象
        src_spec = ChunkShardingSpec(
            dim=0,  # 指定分片维度为0
            placements=[  # 指定分片的放置策略
                "rank:0",
                "rank:1",
            ],
        )

        # 创建目标数据分片规格对象
        dst_spec = ChunkShardingSpec(
            dim=1,  # 指定分片维度为1
            placements=[  # 指定分片的放置策略
                "rank:0",
                "rank:1",
            ],
        )

        # 创建要保存的分片模型对象，并移到当前 GPU
        model_to_save = MyShardedModel3(src_spec).cuda(dist.get_rank())
        # 注册状态字典钩子
        model_to_save._register_state_dict_hook(state_dict_hook)
        # 获取要保存的状态字典
        state_dict_to_save = model_to_save.state_dict()

        # 创建文件系统写入器对象
        fs_writer = FileSystemWriter(path=path, thread_count=thread_count)
        # 保存状态字典到文件系统
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        # 创建要加载的分片模型对象，并移到当前 GPU
        model_to_load = MyShardedModel3(dst_spec).cuda(dist.get_rank())
        # 注册状态字典钩子
        model_to_load._register_state_dict_hook(state_dict_hook)
        # 获取要加载的目标状态字典
        state_dict_to_load_to = model_to_load.state_dict()

        # 创建文件系统读取器对象
        fs_reader = FileSystemReader(path=path)

        # 从文件系统加载状态字典
        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

        # 由于每个分片有不同的分片规格，不能使用 torch.allclose 进行比较
        store_tensor = self.load_tensor(model_to_save.sharded_tensor)
        load_tensor = self.load_tensor(model_to_load.sharded_tensor)

        # 如果是第一个进程，断言两个张量近似相等
        if dist.get_rank() == 0:
            self.assertTrue(torch.allclose(store_tensor, load_tensor))

    @with_comms(init_rpc=False, backend="gloo")
    @parametrize("thread_count", _THREAD_COUNTS)
    def test_save_load_bytes(self, thread_count) -> None:
        # 获取文件路径
        path = self.get_file_path()

        # 准备要保存的字节数据状态字典
        state_dict_to_save = {"bytes0": [1], "bytes1": "string"}

        # 创建文件系统写入器对象
        fs_writer = FileSystemWriter(path=path, thread_count=thread_count)
        # 保存状态字典到文件系统
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        # 准备要加载的字节数据状态字典
        state_dict_to_load = {"bytes0": [2], "bytes1": "other"}

        # 创建文件系统读取器对象
        fs_reader = FileSystemReader(path=path)
        # 从文件系统加载状态字典
        load_state_dict(state_dict=state_dict_to_load, storage_reader=fs_reader)

        # 断言加载后的字节数据与原始数据相等
        self.assertEqual([1], state_dict_to_load["bytes0"])
        self.assertEqual("string", state_dict_to_load["bytes1"])

    @with_comms(init_rpc=False, backend="gloo")
    @parametrize("thread_count", _THREAD_COUNTS)
# 调用函数 instantiate_parametrized_tests，传入 TestDistributedStateDictSaveLoad 类作为参数
instantiate_parametrized_tests(TestDistributedStateDictSaveLoad)

# 调用函数 instantiate_parametrized_tests，传入 TestDistributedStateDictSaveLoadWithSharedTensor 类作为参数
instantiate_parametrized_tests(TestDistributedStateDictSaveLoadWithSharedTensor)

# 调用函数 instantiate_parametrized_tests，传入 TestDistributedReshardOnLoad 类作为参数
instantiate_parametrized_tests(TestDistributedReshardOnLoad)

# 检查当前脚本是否作为主程序运行，如果是则调用 run_tests 函数
if __name__ == "__main__":
    run_tests()
```