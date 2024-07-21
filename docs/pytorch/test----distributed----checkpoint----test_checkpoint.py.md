# `.\pytorch\test\distributed\checkpoint\test_checkpoint.py`

```
# Owner(s): ["oncall: distributed"]

import os  # 导入操作系统模块
import sys  # 导入系统模块
from typing import cast, List, Optional, Union  # 导入类型提示相关的模块

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式模块
import torch.futures  # 导入PyTorch异步模块
import torch.nn  # 导入PyTorch神经网络模块

from torch.distributed._shard import sharded_tensor  # 导入分片张量相关模块

from torch.distributed._shard.sharded_tensor import ShardedTensor, state_dict_hook  # 导入分片张量和状态字典钩子
from torch.distributed._shard.sharding_spec import ChunkShardingSpec  # 导入分片规范模块

from torch.distributed.checkpoint import (  # 导入分布式检查点相关模块
    CheckpointException,  # 导入检查点异常类
    load_state_dict,  # 导入加载状态字典函数
    save_state_dict,  # 导入保存状态字典函数
    StorageReader,  # 导入存储读取器
    StorageWriter,  # 导入存储写入器
)

from torch.distributed.checkpoint.default_planner import _create_default_local_metadata  # 导入创建默认本地元数据函数

from torch.distributed.checkpoint.metadata import (  # 导入元数据相关模块
    BytesStorageMetadata,  # 导入字节存储元数据类
    Metadata,  # 导入元数据类
    TensorStorageMetadata,  # 导入张量存储元数据类
)

from torch.distributed.checkpoint.planner import (  # 导入检查点计划相关模块
    LoadPlan,  # 导入加载计划类
    LoadPlanner,  # 导入加载计划器类
    SavePlan,  # 导入保存计划类
    SavePlanner,  # 导入保存计划器类
)
from torch.distributed.checkpoint.storage import WriteResult  # 导入写入结果类
from torch.futures import Future  # 导入异步任务类
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu  # 导入测试相关模块

from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 导入通用测试函数和调试模式检查标记
from torch.testing._internal.distributed._shard.sharded_tensor import (  # 导入分片张量测试基类和通信上下文装饰器
    ShardedTensorTestBase,
    with_comms,
)

if TEST_WITH_DEV_DBG_ASAN:  # 如果处于开发调试模式下
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",  # 打印消息：跳过dev-asan，因为torch + multiprocessing spawn有已知问题
        file=sys.stderr,  # 输出到标准错误流
    )
    sys.exit(0)  # 退出程序，返回状态码0


class TestModule(torch.nn.Module):  # 定义测试模块类，继承自PyTorch的模块基类
    def __init__(self) -> None:  # 初始化方法
        super().__init__()  # 调用父类初始化方法
        self.sharded: ShardedTensor = sharded_tensor.zeros(self.spec(), 4, 4)  # 创建分片张量对象
        self.regular = torch.nn.Parameter(torch.ones(4, 4))  # 创建普通的PyTorch参数对象
        self.extra_sharded: Optional[ShardedTensor] = None  # 额外的分片张量对象，默认为空
        self.extra_param: Optional[torch.nn.Parameter] = None  # 额外的PyTorch参数对象，默认为空
        self._register_state_dict_hook(state_dict_hook)  # 注册状态字典钩子函数

    def spec(self) -> ChunkShardingSpec:  # 定义方法spec，返回分片规范对象
        # pyre-fixme [28]: Unexpected keyword argument `dim` to call `dist._sharding_spec.api.ChunkShardingSpec.__init__`.
        return ChunkShardingSpec(  # 返回分片规范对象
            dim=0,  # 指定分片的维度为0
            placements=[  # 分片的放置位置列表
                "rank:0/cuda:0",  # 放置在rank 0的cuda 0上
                "rank:1/cuda:1",  # 放置在rank 1的cuda 1上
            ],
        )


class TestDistributedCheckpointing(ShardedTensorTestBase):  # 定义分布式检查点测试类，继承自分片张量测试基类
    @property
    def world_size(self) -> int:  # 定义属性world_size，返回整数类型
        return 2  # 返回值为2，表示分布式环境的世界大小为2

    @with_comms(init_rpc=False)  # 使用通信上下文装饰器，初始化RPC为假
    @skip_if_lt_x_gpu(2)  # 如果GPU数小于2，则跳过测试
    @requires_nccl()  # 要求使用NCCL库
    def test_tensor_metadata_with_missing_rank_spec(self) -> None:  # 定义测试方法test_tensor_metadata_with_missing_rank_spec，无返回值
        spec = ChunkShardingSpec(  # 创建分片规范对象
            dim=0,  # 指定分片的维度为0
            placements=[  # 分片的放置位置列表
                "rank:1/cuda:1",  # 放置在rank 1的cuda 1上
            ],
        )

        st = sharded_tensor.zeros(spec, 4, 4, dtype=torch.float64)  # 创建具有特定规范的分片张量
        mapping = {}  # 创建空字典对象，用于映射

        md = _create_default_local_metadata({"st": st})  # 创建默认的本地元数据

        st_md = md.state_dict_metadata["st"]  # 获取分片张量的状态字典元数据
        self.assertEqual(1, len(st_md.chunks))  # 断言检查分片数是否为1
    # 定义一个测试方法，测试默认的元数据生成函数 _create_default_local_metadata
    def test_default_metadata(self) -> None:
        # 根据分布式环境的排名获取设备，例如 "cuda:0" 或 "cuda:1"
        device = f"cuda:{dist.get_rank()}"
        # 创建一个块分片规格对象，指定维度为0，包含两个放置信息
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
            ],
        )

        # 构建状态字典，包括不同类型的数据
        state_dict = {
            "sharded": sharded_tensor.rand(  # 在分片张量中生成随机数据
                spec,
                (
                    10,
                    10,
                ),
            ),
            "replicated": torch.rand(4, device=device),  # 在指定设备上生成随机张量
            "bytes": [1, 2, 3, 4],  # 字节列表
        }

        # 调用元数据生成函数，获取默认的本地元数据对象
        metadata = _create_default_local_metadata(state_dict)

        # 断言检查生成的元数据中是否包含 "bytes" 字段
        self.assertTrue("bytes" in metadata.state_dict_metadata)
        # 断言检查 "bytes" 字段对应的元数据类型是否为 BytesStorageMetadata
        self.assertIsInstance(
            metadata.state_dict_metadata["bytes"], BytesStorageMetadata
        )

        # 断言检查生成的元数据中是否包含 "replicated" 字段
        self.assertTrue("replicated" in metadata.state_dict_metadata)
        # 断言检查 "replicated" 字段对应的元数据类型是否为 TensorStorageMetadata
        self.assertIsInstance(
            metadata.state_dict_metadata["replicated"], TensorStorageMetadata
        )
        # 获取 "replicated" 字段对应的元数据对象
        md = metadata.state_dict_metadata["replicated"]
        # 断言检查元数据中张量大小是否与原始状态字典中的张量大小相同
        self.assertEqual(md.size, state_dict["replicated"].size())
        # 断言检查元数据中张量属性的数据类型是否为 torch.float32
        self.assertEqual(md.properties.dtype, torch.float32)
        # 断言检查元数据中块数是否为1
        self.assertEqual(1, len(md.chunks))

        # 断言检查生成的元数据中是否包含 "sharded" 字段
        self.assertTrue("sharded" in metadata.state_dict_metadata)
        # 断言检查 "sharded" 字段对应的元数据类型是否为 TensorStorageMetadata
        self.assertIsInstance(
            metadata.state_dict_metadata["sharded"], TensorStorageMetadata
        )
        # 获取 "sharded" 字段对应的元数据对象
        md = metadata.state_dict_metadata["sharded"]
        # 断言检查元数据中张量属性的数据类型是否为 torch.float32
        self.assertEqual(md.properties.dtype, torch.float32)
        # 断言检查元数据中张量大小是否与原始状态字典中的张量大小相同
        self.assertEqual(md.size, state_dict["sharded"].size())
        # 断言检查元数据中块数是否为2
        self.assertEqual(2, len(md.chunks))
# 定义一个测试存储基类，继承自TestStorageBase和StorageWriter，用于模拟有故障情况的存储写入操作
class FaultyStorageWriter(TestStorageBase, StorageWriter):
    def __init__(self, fail_conf):
        super().__init__(fail_conf)  # 调用父类TestStorageBase和StorageWriter的构造函数进行初始化

    # 重置操作，不执行任何动作
    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        return

    # 设置存储写入器的初始化过程，模拟失败条件并抛出异常
    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        self._fail_rank("fail_set_up_storage_writer")

    # 准备本地存储计划，如果出现失败条件则抛出异常
    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        self._fail_rank("fail_prepare_local_plan")
        return plan

    # 准备全局存储计划，如果出现失败条件则抛出异常
    def prepare_global_plan(self, plans: List[SavePlan]) -> List[SavePlan]:
        self._fail_rank("fail_prepare_global_plan")
        return plans

    # 写入数据，如果出现失败条件则返回一个异步异常Future
    def write_data(
        self, plan: SavePlan, planner: SavePlanner
    ) -> Future[List[WriteResult]]:
        self._fail_rank("fail_write_data")
        return self._fail_rank_async("fail_write_data_async", [])

    # 结束操作，如果出现失败条件则抛出异常
    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        self._fail_rank("fail_finish")

    # 静态方法，验证检查点ID的有效性，总是返回True
    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return True


# 定义一个测试存储读取基类，继承自TestStorageBase和StorageReader，用于模拟有故障情况的存储读取操作
class FaultyStorageReader(TestStorageBase, StorageReader):
    def __init__(self, metadata, fail_conf):
        super().__init__(fail_conf)  # 调用父类TestStorageBase和StorageReader的构造函数进行初始化
        self.metadata = metadata  # 存储元数据

    # 重置操作，不执行任何动作
    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        return

    # 设置存储读取器的初始化过程，模拟失败条件并抛出异常
    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        self._fail_rank("fail_set_up_storage_reader")

    # 准备本地读取计划，如果出现失败条件则抛出异常
    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        self._fail_rank("fail_prepare_local_plan")
        return plan

    # 准备全局读取计划，如果出现失败条件则抛出异常
    def prepare_global_plan(self, plans: List[LoadPlan]) -> List[LoadPlan]:
        self._fail_rank("fail_prepare_global_plan")
        return plans

    # 读取数据，如果出现失败条件则返回一个异步异常Future
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        self._fail_rank("fail_read_data")
        return self._fail_rank_async("fail_read_data_async")

    # 读取元数据，如果出现失败条件则返回存储的元数据
    def read_metadata(self) -> Metadata:
        self._fail_rank("fail_read_metadata")
        return self.metadata
    # 定义一个类方法，用于验证检查点 ID 是否有效
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        # 简单地返回 True，表示检查点 ID 总是有效
        return True
class TestDistributedFailure(ShardedTensorTestBase):
    # TestDistributedFailure 类继承自 ShardedTensorTestBase 类

    def get_spec(self):
        # 返回一个 ChunkShardingSpec 对象，指定了数据分片的规格
        return ChunkShardingSpec(
            dim=0,
            placements=[f"rank:{r}/cuda:{r}" for r in range(dist.get_world_size())],
        )

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_dummy_writer_works(self) -> None:
        # 定义一个包含不同数据类型的 state_dict 对象
        state_dict = {
            "sharded": sharded_tensor.rand(self.get_spec(), 20, 20),
            "replicated": torch.rand(10, 10),
            "bytes": [1, 2, 3, 4],
        }

        # 调用 save_state_dict 函数，将 state_dict 保存到 FaultyStorageWriter 中
        save_state_dict(state_dict, FaultyStorageWriter({}))

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_dummy_reader_works(self) -> None:
        # 定义一个包含不同数据类型的 state_dict 对象
        state_dict = {
            "sharded": sharded_tensor.rand(self.get_spec(), 20, 20),
            "replicated": torch.rand(10, 10),
            "bytes": [1, 2, 3, 4],
        }

        # 使用 state_dict 创建默认的本地 metadata
        metadata = _create_default_local_metadata(state_dict)

        # 调用 load_state_dict 函数，从 FaultyStorageReader 中加载 state_dict
        load_state_dict(state_dict, FaultyStorageReader(metadata, {}))

    def _test_dist_failure(self, callback, kwargs):
        # 如果 kwargs 不为空，获取第一个参数的值作为 bad_ranks
        bad_ranks = next(iter(kwargs.values())) if len(kwargs) > 0 else []

        # 如果 bad_ranks 为空列表，则调用 callback 函数
        if len(bad_ranks) == 0:
            callback()
        else:
            # 否则，使用 self.assertRaises 来捕获 CheckpointException 异常
            with self.assertRaises(CheckpointException) as cm:
                callback()
            e = cast(CheckpointException, cm.exception)
            # 遍历异常中的失败项，并验证是否符合预期的异常类型
            for rank, wrapped_ex in e.failures.items():
                ex = wrapped_ex[0]
                self.assertTrue(rank in bad_ranks, msg=f"{rank} did not fail")
                if not kwargs.get("ignore_exception_type", False):
                    self.assertEqual(ValueError, type(ex), str(ex))

            failed_ranks = e.failures.keys()
            # 验证所有预期失败的 rank 是否都在失败列表中
            for rank in bad_ranks:
                self.assertTrue(
                    rank in failed_ranks,
                    msg=f"{rank} was supposed to fail was fine",
                )

    def _test_save(self, state_dict, coordinator=0, **kwargs):
        # 检查当前是否未初始化分布式环境
        no_dist = not dist.is_initialized()

        def _save():
            # 定义保存函数 _save，调用 save_state_dict 函数
            save_state_dict(
                state_dict,
                storage_writer=FaultyStorageWriter(kwargs),
                coordinator_rank=coordinator,
                no_dist=no_dist,
            )

        # 调用 _test_dist_failure 函数，传入 _save 函数和额外的 kwargs 参数
        self._test_dist_failure(_save, kwargs)

    def _test_load(self, state_dict, coordinator=0, **kwargs):
        # 检查当前是否未初始化分布式环境
        no_dist = not dist.is_initialized()

        def _load():
            # 定义加载函数 _load，创建默认的本地 metadata，并调用 load_state_dict 函数
            metadata = _create_default_local_metadata(state_dict)
            load_state_dict(
                state_dict,
                storage_reader=FaultyStorageReader(metadata, kwargs),
                coordinator_rank=coordinator,
                no_dist=no_dist,
            )

        # 调用 _test_dist_failure 函数，传入 _load 函数和额外的 kwargs 参数
        self._test_dist_failure(_load, kwargs)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    # 定义一个测试方法，用于测试保存过程中的错误处理
    def test_save_error_handling(self) -> None:
        # 创建包含不同数据类型的状态字典
        state_dict = {
            "sharded": sharded_tensor.rand(self.get_spec(), 20, 20),  # 创建分片张量数据
            "replicated": torch.rand(10, 10),  # 创建普通张量数据
            "bytes": [1, 2, 3, 4],  # 创建整数列表
        }

        # 测试保存方法，模拟存储写入时设置存储写入失败的情况
        self._test_save(state_dict, fail_set_up_storage_writer=[0])
        # 测试保存方法，模拟存储写入完成时失败的情况
        self._test_save(state_dict, fail_finish=[0])
        # 测试保存方法，模拟准备全局计划时失败的情况
        self._test_save(state_dict, fail_prepare_global_plan=[0])

        # 测试保存方法，模拟准备局部计划时失败的情况
        self._test_save(state_dict, fail_prepare_local_plan=[0])
        # 测试保存方法，模拟写入数据时失败的情况
        self._test_save(state_dict, fail_write_data=[2])
        # 测试保存方法，模拟异步写入数据时失败的情况
        self._test_save(state_dict, fail_write_data_async=[3])

        # 测试保存方法，模拟协调器为1时设置存储写入失败的情况
        self._test_save(state_dict, coordinator=1, fail_set_up_storage_writer=[1])
        # 测试保存方法，模拟协调器为1时存储写入完成时失败的情况
        self._test_save(state_dict, coordinator=1, fail_finish=[1])

    # 定义一个测试方法，用于测试在没有分布式环境下的保存过程中的错误处理
    def test_save_error_handling_no_dist(self) -> None:
        # 创建包含不同数据类型的状态字典
        state_dict = {"replicated": torch.rand(10, 10), "bytes": [1, 2, 3, 4]}

        # 断言当前不是处于分布式初始化状态
        self.assertFalse(dist.is_initialized())

        # 测试保存方法，模拟存储写入时设置存储写入失败的情况
        self._test_save(state_dict, fail_set_up_storage_writer=[0])
        # 测试保存方法，模拟存储写入完成时失败的情况
        self._test_save(state_dict, fail_finish=[0])
        # 测试保存方法，模拟准备全局计划时失败的情况
        self._test_save(state_dict, fail_prepare_global_plan=[0])

        # 测试保存方法，模拟准备局部计划时失败的情况
        self._test_save(state_dict, fail_prepare_local_plan=[0])
        # 测试保存方法，模拟写入数据时失败的情况
        self._test_save(state_dict, fail_write_data=[0])
        # 测试保存方法，模拟异步写入数据时失败的情况
        self._test_save(state_dict, fail_write_data_async=[0])

    # 用于装饰测试加载过程中的错误处理方法，包括初始化 RPC，需要至少4个 GPU，并要求支持 NCCL
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_load_error_handling(self) -> None:
        # 创建包含不同数据类型的状态字典
        state_dict = {
            "sharded": sharded_tensor.rand(self.get_spec(), 20, 20),  # 创建分片张量数据
            "replicated": torch.rand(10, 10),  # 创建普通张量数据
            "bytes": [1, 2, 3, 4],  # 创建整数列表
        }

        # 测试加载方法，正常加载状态字典
        self._test_load(state_dict)
        # 测试加载方法，模拟设置存储读取失败的情况
        self._test_load(state_dict, fail_set_up_storage_reader=[0])
        # 测试加载方法，模拟准备全局计划时失败的情况
        self._test_load(state_dict, fail_prepare_global_plan=[0])
        # 测试加载方法，模拟读取元数据时失败的情况
        self._test_load(state_dict, fail_read_metadata=[0])
        # 测试加载方法，模拟准备局部计划时失败的情况
        self._test_load(state_dict, fail_prepare_local_plan=[1])
        # 测试加载方法，模拟读取数据时失败的情况
        self._test_load(state_dict, fail_read_data=[3])
        # 测试加载方法，模拟异步读取数据时失败的情况
        self._test_load(state_dict, fail_read_data_async=[1])

        # 测试加载方法，模拟协调器为3时设置存储读取失败的情况
        self._test_load(state_dict, coordinator=3, fail_set_up_storage_reader=[0])
        # 测试加载方法，模拟协调器为1时读取元数据时失败的情况
        self._test_load(state_dict, coordinator=1, fail_read_metadata=[3])
        # 测试加载方法，模拟协调器为2时读取数据时失败的情况
        self._test_load(state_dict, coordinator=2, fail_read_data=[0])
        # 测试加载方法，模拟协调器为3时异步读取数据时失败的情况
        self._test_load(state_dict, coordinator=3, fail_read_data_async=[2])
        # 测试加载方法，模拟协调器为1时准备全局计划时失败的情况
        self._test_load(state_dict, coordinator=1, fail_prepare_global_plan=[1])

    # 用于测试在没有分布式环境下加载过程中的错误处理方法
    def test_load_error_handling_no_dist(self) -> None:
        # 创建包含不同数据类型的状态字典
        state_dict = {"replicated": torch.rand(10, 10), "bytes": [1, 2, 3, 4]}
        # 测试加载方法，正常加载状态字典
        self._test_load(state_dict)
        # 测试加载方法，模拟设置存储读取失败的情况
        self._test_load(state_dict, fail_set_up_storage_reader=[0])
        # 测试加载方法，模拟读取元数据时失败的情况
        self._test_load(state_dict, fail_read_metadata=[0])
        # 测试加载方法，模拟准备局部计划时失败的情况
        self._test_load(state_dict, fail_prepare_local_plan=[0])
        # 测试加载方法，模拟准备全局计划时失败的情况
        self._test_load(state_dict, fail_prepare_global_plan=[0])
        # 测试加载方法，模拟读取数据时失败的情况
        self._test_load(state_dict, fail_read_data=[0])
        # 测试加载方法，模拟异步读取数据时失败的情况
        self._test_load(state_dict, fail_read_data_async=[0])
# 如果当前脚本作为主程序执行（而不是被导入到其他脚本中执行），则运行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```