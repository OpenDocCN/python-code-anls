# `.\pytorch\test\distributed\checkpoint\test_file_system_checkpoint.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的模块和库
import os  # 操作系统功能
import shutil  # 文件和目录操作
import sys  # 系统相关功能
import tempfile  # 创建临时文件和目录
from typing import Dict  # 类型提示，导入字典类型的支持

import torch  # PyTorch深度学习库
import torch.distributed as dist  # 分布式通信模块
from torch.distributed._shard import sharded_tensor  # 分布式张量支持
from torch.distributed._shard.sharded_tensor import ShardedTensor, state_dict_hook  # 分布式张量和状态字典钩子
from torch.distributed._shard.sharding_spec import (  # 分片规范相关模块
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardingSpec,
    ShardMetadata,
)

from torch.distributed.checkpoint import (  # 分布式检查点操作
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu  # 内部测试工具

from torch.testing._internal.common_utils import (  # 内部通用工具
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (  # 分布式张量测试相关模块
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_st_common import (  # 分布式张量通用测试工具
    MyShardedModel1,
)

# 如果开启了开发者ASAN模式，则打印消息并退出程序
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


def assert_state_dict_equal(  # 自定义函数：比较两个状态字典是否相等
    self: TestCase,  # 测试用例对象引用
    state_dict_1: Dict[str, torch.Tensor],  # 第一个状态字典
    state_dict_2: Dict[str, torch.Tensor],  # 第二个状态字典
) -> bool:
    # 断言两个状态字典的长度相等，否则抛出异常
    self.assertEqual(
        len(state_dict_1), len(state_dict_2), "state_dict must be the same size"
    )
    # 断言两个状态字典的键集合相等，否则抛出异常
    self.assertEqual(
        set(state_dict_1.keys()),
        set(state_dict_2.keys()),
        "state_dict keys do not match",
    )

    # 遍历第一个状态字典的键值对
    for key, value_1 in state_dict_1.items():
        value_2 = state_dict_2[key]  # 获取第二个状态字典对应键的值
        # 如果值是 ShardedTensor 类型，则比较其本地分片是否相等
        if isinstance(value_1, ShardedTensor):
            for local_shard_1, local_shard_2 in zip(
                value_1.local_shards(), value_2.local_shards()
            ):
                self.assertTrue(
                    torch.equal(local_shard_1.tensor, local_shard_2.tensor),
                    f"Key {key}'s shard does not match",
                )
        # 如果值是 torch.Tensor 类型，则比较其值是否相等
        elif isinstance(value_1, torch.Tensor):
            self.assertTrue(
                torch.equal(value_1, value_2),
                f"Key {key}'s tensor does not match",
            )

    return True


class MyTestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 5)  # 创建线性层
        self.linear_2 = torch.nn.Linear(5, 1)  # 创建线性层
        self.emb = torch.nn.EmbeddingBag(5, 10)  # 创建EmbeddingBag层


# 以下的 ShardedModels 类定义是从 test/distributed/_sharded_tensor/test_sharded_tensor.py 中借用的
class MyShardedModel3(torch.nn.Module):
    def __init__(
        self,
        spec: ShardingSpec,  # 分片规范对象
    ) -> None:
        super().__init__()
        self.sharded_tensor: ShardedTensor = sharded_tensor.rand(
            spec, 10, 20, init_rrefs=False
        )  # 创建随机初始化的分布式张量对象


class TestDistributedStateDictSaveLoad(TestCase):  # 分布式状态字典保存和加载测试用例类
    # 定义单元测试方法，测试读取和写入仅张量的功能
    def test_read_write_only_tensor(self) -> None:
        # 使用临时目录进行测试
        with tempfile.TemporaryDirectory() as path:
            # 准备要保存的模块状态字典
            state_dict_to_save = MyTestModule().state_dict()

            # 创建文件系统写入器对象，指定保存路径
            fs_writer = FileSystemWriter(path=path)
            # 调用保存状态字典的函数，写入文件系统
            save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=fs_writer,
                no_dist=True,
            )

            # 准备要加载的模块状态字典
            state_dict_to_load_to = MyTestModule().state_dict()

            # 断言加载后的状态字典与保存前不相等，抛出断言错误
            with self.assertRaises(AssertionError):
                assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

            # 使用文件系统读取器对象，从文件加载状态字典
            fs_reader = FileSystemReader(path=path)
            # 调用加载状态字典的函数，从文件系统读取并加载
            load_state_dict(
                state_dict=state_dict_to_load_to,
                storage_reader=fs_reader,
                no_dist=True,
            )

            # 断言加载后的状态字典与保存前相等
            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

        # 使用另一个临时目录进行测试，包含单个文件的情况
        with tempfile.TemporaryDirectory() as path:
            # 准备要保存的模块状态字典
            state_dict_to_save = MyTestModule().state_dict()

            # 创建文件系统写入器对象，指定保存路径并设置单个文件每个等级
            fs_writer = FileSystemWriter(path=path, single_file_per_rank=True)
            # 调用保存状态字典的函数，写入文件系统
            save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=fs_writer,
                no_dist=True,
            )

            # 准备要加载的模块状态字典
            state_dict_to_load_to = MyTestModule().state_dict()

            # 断言加载后的状态字典与保存前不相等，抛出断言错误
            with self.assertRaises(AssertionError):
                assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

            # 使用文件系统读取器对象，从文件加载状态字典
            fs_reader = FileSystemReader(path=path)
            # 调用加载状态字典的函数，从文件系统读取并加载
            load_state_dict(
                state_dict=state_dict_to_load_to,
                storage_reader=fs_reader,
                no_dist=True,
            )

            # 断言加载后的状态字典与保存前相等
            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)
# 定义一个测试类，用于测试带有共享张量的分布式状态字典的保存和加载
class TestDistributedStateDictSaveLoadWithSharedTensor(ShardedTensorTestBase):

    # 返回世界大小，这里固定返回2，表示有两个进程
    @property
    def world_size(self) -> int:
        return 2

    # 测试函数，用于测试读写分片张量
    @with_comms(init_rpc=False)  # 使用通信环境，但不初始化 RPC
    @skip_if_lt_x_gpu(2)  # 如果 GPU 少于2个，则跳过测试
    @requires_nccl()  # 要求使用 NCCL 通信
    def test_read_write_shard_tensor(self) -> None:
        # 创建临时目录的路径列表
        paths = [tempfile.mkdtemp()]
        # 将路径列表广播给所有进程
        dist.broadcast_object_list(paths)

        # 取第一个路径作为存储路径
        path = paths[0]

        # 创建分片规范，指定维度为0，指定各分片的放置方式
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        )

        # 创建要保存的模型对象，使用指定的分片规范和禁止初始化 RRefs
        model_to_save = MyShardedModel1(spec, init_rrefs=False)

        # 测试保存状态
        model_to_save._register_state_dict_hook(state_dict_hook)  # 注册状态字典钩子
        state_dict_to_save = model_to_save.state_dict()  # 获取要保存的状态字典

        fs_writer = FileSystemWriter(path=path)  # 创建文件系统写入器
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)  # 保存状态字典到文件系统

        dist.barrier()  # 等待所有进程完成

        # 创建一个新的模型对象
        model_to_load = MyShardedModel1(spec, init_rrefs=False)

        # 注释掉下一行是不正确的加载状态字典的钩子
        # model_to_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)

        model_to_load._register_state_dict_hook(state_dict_hook)  # 注册状态字典钩子
        state_dict_to_load_to = model_to_load.state_dict()  # 获取加载后的状态字典

        dist.barrier()  # 等待所有进程完成

        # 使用断言检查加载后的状态字典与保存的状态字典是否相等，预期会抛出 AssertionError
        with self.assertRaises(AssertionError):
            assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)

        # 测试加载状态字典
        fs_reader = FileSystemReader(path=path)  # 创建文件系统读取器
        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)  # 从文件系统加载状态字典

        assert_state_dict_equal(self, state_dict_to_load_to, state_dict_to_save)  # 使用断言检查加载后的状态字典与保存的状态字典是否相等
        dist.barrier()  # 等待所有进程完成
    # 定义测试方法，用于测试行到列的加载
    def test_load_rowwise_to_colwise(self) -> None:
        # 获取文件路径
        path = self.get_file_path()
        # 断言当前进程数与全局进程数相等
        self.assertEqual(self.world_size, dist.get_world_size())

        # 创建源数据分片规格对象
        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        src_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        )

        # 创建目标数据分片规格对象
        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        dst_spec = ChunkShardingSpec(
            dim=1,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        )

        # 如果当前进程是第一个进程
        if dist.get_rank() == 0:
            # 递归删除路径下所有内容（如果存在），忽略错误
            shutil.rmtree(path, ignore_errors=True)
            # 创建目录路径
            os.makedirs(path)

        # 创建要保存的分片模型对象，并将其移到当前 GPU
        model_to_save = MyShardedModel3(src_spec).cuda(dist.get_rank())
        # 注册状态字典钩子函数
        model_to_save._register_state_dict_hook(state_dict_hook)
        # 获取要保存的状态字典
        state_dict_to_save = model_to_save.state_dict()

        # 创建文件系统写入器对象，指定路径
        fs_writer = FileSystemWriter(path=path)
        # 保存状态字典到文件系统中
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        # 创建要加载的分片模型对象，并将其移到当前 GPU
        model_to_load = MyShardedModel3(dst_spec).cuda(dist.get_rank())
        # 注册状态字典钩子函数
        model_to_load._register_state_dict_hook(state_dict_hook)
        # 创建用于加载的状态字典对象
        state_dict_to_load_to = model_to_load.state_dict()

        # 创建文件系统读取器对象，指定路径
        fs_reader = FileSystemReader(path=path)
        # 从文件系统中加载状态字典
        load_state_dict(state_dict=state_dict_to_load_to, storage_reader=fs_reader)

        # 由于每个分片模型有不同的分片规格，因此无法使用 torch.allclose 进行比较
        # 加载存储的张量数据到 store_tensor
        store_tensor = self.load_tensor(model_to_save.sharded_tensor)
        # 加载加载的张量数据到 load_tensor
        load_tensor = self.load_tensor(model_to_load.sharded_tensor)

        # 如果当前进程是第一个进程
        if dist.get_rank() == 0:
            # 断言 store_tensor 与 load_tensor 在数值上近似相等
            self.assertTrue(torch.allclose(store_tensor, load_tensor))

    # 标记为通信测试，初始化 RPC 功能为关闭状态
    @with_comms(init_rpc=False)
    # 如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 要求使用 NCCL 通信库
    @requires_nccl()
    # 定义测试方法，用于测试字节数据的保存与加载
    def test_save_load_bytes(self) -> None:
        # 获取文件路径
        path = self.get_file_path()

        # 要保存的状态字典对象，包含两个条目
        state_dict_to_save = {"bytes0": [1], "bytes1": "string"}

        # 创建文件系统写入器对象，指定路径
        fs_writer = FileSystemWriter(path=path)
        # 保存状态字典到文件系统中
        save_state_dict(state_dict=state_dict_to_save, storage_writer=fs_writer)

        # 要加载的状态字典对象，修改了条目的值
        state_dict_to_load = {"bytes0": [2], "bytes1": "other"}

        # 创建文件系统读取器对象，指定路径
        fs_reader = FileSystemReader(path=path)
        # 从文件系统中加载状态字典
        load_state_dict(state_dict=state_dict_to_load, storage_reader=fs_reader)

        # 断言加载后的状态字典中 "bytes0" 的值为 [1]
        self.assertEqual([1], state_dict_to_load["bytes0"])
        # 断言加载后的状态字典中 "bytes1" 的值为 "string"
        self.assertEqual("string", state_dict_to_load["bytes1"])

    # 标记为通信测试，初始化 RPC 功能为关闭状态
    @with_comms(init_rpc=False)
    # 如果 GPU 数量小于 2，则跳过测试
    @skip_if_lt_x_gpu(2)
    # 要求使用 NCCL 通信库
    @requires_nccl()
# 如果这个脚本被直接执行（而不是被导入到其他模块中执行），则运行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```