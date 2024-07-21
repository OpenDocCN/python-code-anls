# `.\pytorch\test\distributed\_shard\sharded_tensor\test_sharded_tensor.py`

```py
# Owner(s): ["oncall: distributed"]
# 导入必要的模块和库
import copy                     # 导入深拷贝函数
import io                       # 导入处理IO流的库
import itertools                # 导入迭代工具函数
import math                     # 导入数学函数
import pickle                   # 导入序列化和反序列化函数
import sys                      # 导入系统相关功能
from typing import List         # 导入类型提示

import torch                    # 导入PyTorch库
import torch.distributed as dist  # 导入分布式训练相关功能
from torch.distributed import distributed_c10d, rpc  # 导入分布式通信模块
from torch.distributed._shard import sharded_tensor  # 导入分片张量相关功能
from torch.distributed._shard.api import (  # 导入分片张量的API
    _collect_local_shard,
    _reshard_output,
    _shard_tensor,
    load_with_process_group,
    shard_parameter,
)
from torch.distributed._shard.sharded_tensor import (  # 导入分片张量类和函数
    custom_sharded_op_impl,
    pre_load_state_dict_hook,
    Shard,
    ShardedTensor,
    ShardedTensorBase,
    ShardedTensorMetadata,
    state_dict_hook,
)
from torch.distributed._shard.sharded_tensor.api import (  # 导入分片张量的API函数
    _create_tensor_from_params,
    TensorProperties,
)
from torch.distributed._shard.sharded_tensor.utils import (  # 导入分片张量的实用工具函数
    _parse_and_validate_remote_device,
)
from torch.distributed._shard.sharding_spec import (  # 导入分片规格相关功能
    ChunkShardingSpec,
    EnumerableShardingSpec,
    ShardMetadata,
)
from torch.distributed.remote_device import _remote_device  # 导入远程设备模块
from torch.testing._internal.common_distributed import (  # 导入分布式测试相关功能
    requires_nccl,
    skip_if_lt_x_gpu,
    spawn_threads_and_init_comms,
    tp_transports,
)
from torch.testing._internal.common_utils import (  # 导入通用测试工具函数
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_CUDA,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (  # 导入分片张量测试相关类
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_st_common import (  # 导入分片张量测试的共用函数
    _chunk_sharding_specs_list_for_test,
    MyShardedModel1,
)

if TEST_WITH_DEV_DBG_ASAN:  # 如果开启了ASAN调试模式，则跳过该测试
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 定义测试类 TestShardedTensorMetadata，继承自 TestCase
class TestShardedTensorMetadata(TestCase):
    # 定义一个测试方法，用于序列化和反序列化验证
    def test_serialize_and_deserialize(self):
        # 创建一组分片元数据对象列表
        shard_metadatas = [
            ShardMetadata(
                shard_offsets=[0, 0],
                shard_sizes=[5, 5],
                placement="rank:0/cuda:0",
            ),
            ShardMetadata(
                shard_offsets=[0, 5],
                shard_sizes=[5, 5],
                placement="rank:1/cuda:1",
            ),
            ShardMetadata(
                shard_offsets=[5, 0],
                shard_sizes=[5, 5],
                placement="rank:2/cuda:2",
            ),
            ShardMetadata(
                shard_offsets=[5, 5],
                shard_sizes=[5, 5],
                placement="rank:3/cuda:3",
            ),
        ]

        # 定义一组张量数据类型
        dtypes = [
            torch.float,
            torch.double,
            torch.cfloat,
            torch.cdouble,
            torch.half,
            torch.bfloat16,
            torch.uint8,
            torch.int8,
            torch.short,
            torch.int,
            torch.long,
            torch.bool,
        ]

        # 定义一组张量布局
        layouts = [torch.strided, torch.sparse_coo]
        
        # 定义是否需要梯度的布尔值
        requires_grads = [True, False]
        
        # 定义内存格式
        memory_formats = [
            torch.contiguous_format,
            torch.channels_last,
            torch.preserve_format,
        ]
        
        # 定义是否使用固定内存的布尔值
        pin_memories = [True, False]

        # 遍历所有可能的张量属性组合
        for tensor_properties_input in itertools.product(
            dtypes, layouts, requires_grads, memory_formats, pin_memories
        ):
            (
                dtype,
                layout,
                requires_grad,
                memory_format,
                pin_memory,
            ) = tensor_properties_input

            # 创建预期的分布式张量元数据对象
            expected_st_metadata = sharded_tensor.ShardedTensorMetadata(
                shard_metadatas,
                (10, 10),
                TensorProperties(
                    dtype, layout, requires_grad, memory_format, pin_memory
                ),
            )

            # 序列化预期的分布式张量元数据对象
            pickled_obj = pickle.dumps(expected_st_metadata)
            
            # 反序列化对象
            st_metadata = pickle.loads(pickled_obj)
            
            # 使用断言检查预期的分布式张量元数据对象与反序列化得到的对象是否相等
            self.assertEqual(expected_st_metadata, st_metadata)
# 定义一个测试类 TestShardParameter，继承自 ShardedTensorTestBase
class TestShardParameter(ShardedTensorTestBase):

    # 用装饰器 with_comms(init_rpc=False) 包装，表明在测试中不需要初始化 RPC
    @with_comms(init_rpc=False)
    # 用装饰器 skip_if_lt_x_gpu(4) 包装，如果 GPU 少于 4 个则跳过测试
    @skip_if_lt_x_gpu(4)
    # 用装饰器 requires_nccl() 包装，表明测试需要使用 NCCL
    def test_shard_parameter(self):
        # 定义 ChunkShardingSpec 对象 spec，指定维度为0，分布在四个 GPU 上
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        # 创建一个具有12个输入和12个输出的线性层，并将其放在当前 GPU 上
        fc = torch.nn.Linear(12, 12).cuda(self.rank)
        # 克隆线性层的权重张量，命名为 weight_og
        weight_og = fc.weight.clone()
        # 根据规范 spec 对线性层的权重进行分片
        shard_parameter(fc, "weight", spec)

        # 验证分片操作是否成功
        # 检查 fc.weight 是否是 ShardedTensor 类型
        self.assertTrue(isinstance(fc.weight, ShardedTensor))
        # 获取本地的分片列表
        local_shards = fc.weight.local_shards()
        # 断言本地分片列表长度为1
        self.assertEqual(1, len(local_shards))
        # 断言第一个分片的张量大小为 [3, 12]
        self.assertEqual(torch.Size([3, 12]), local_shards[0].tensor.size())
        # 断言第一个分片的张量在第一维度上的大小为 3
        self.assertEqual(3, local_shards[0].tensor.size(0))
        # 断言第一个分片的张量在第二维度上的大小为 12
        self.assertEqual(12, local_shards[0].tensor.size(1))
        # 断言第一个分片的张量与原始权重张量 weight_og 在指定维度上的对应部分相等
        self.assertEqual(
            torch.narrow(weight_og, 0, 3 * self.rank, 3), local_shards[0].tensor
        )
    # 定义测试函数，用于测试参数错误的情况
    def test_shard_parameter_errors(self):
        # 创建分片规范对象，指定维度为0，并设置分片位置列表
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        # 在指定设备上创建一个线性层对象，并指定CUDA设备
        fc = torch.nn.Linear(12, 12).cuda(self.rank)
        
        # 使用断言检查是否引发值错误，并匹配特定错误消息
        with self.assertRaisesRegex(ValueError, "does not match with src_rank"):
            # 调用函数进行参数分片操作
            shard_parameter(fc, "weight", spec, src_rank=self.rank)

        # 使用断言检查是否引发属性错误，并匹配特定错误消息
        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            # 调用函数进行参数分片操作
            shard_parameter(fc, "foo", spec)

        # 使用断言检查是否引发值错误，并匹配特定错误消息
        with self.assertRaisesRegex(
            ValueError, "Expected Linear.bias to be a Tensor, but found str"
        ):
            # 删除偏置项并设置为字符串，然后调用函数进行参数分片操作
            del fc.bias
            fc.bias = "foo"
            shard_parameter(fc, "bias", spec)

        # 使用断言检查是否引发值错误，并匹配特定错误消息
        with self.assertRaisesRegex(ValueError, "not a contiguous Tensor"):
            # 将偏置项设置为非连续张量，然后调用函数进行参数分片操作
            fc.bias = torch.rand(10, 10).cuda(self.rank).t()
            shard_parameter(fc, "bias", spec)

        # 更新分片规范对象，设置当前设备特定的分片位置
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:{self.rank}/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 使用断言检查是否引发值错误，并匹配特定错误消息
        with self.assertRaisesRegex(ValueError, "does not match with sharding_spec"):
            # 调用函数进行参数分片操作
            shard_parameter(fc, "weight", spec)

        # 创建枚举分片规范对象，指定分片元数据列表
        spec = EnumerableShardingSpec(
            [
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
            ]
        )
        # 使用断言检查是否引发未实现错误，并匹配特定错误消息
        with self.assertRaisesRegex(NotImplementedError, "not implemented yet!"):
            # 调用函数进行参数分片操作
            shard_parameter(fc, "weight", spec)
# 定义一个测试类 TestShardTensor，继承自 ShardedTensorTestBase
class TestShardTensor(ShardedTensorTestBase):

    # 声明一个测试方法 test_shard_tensor，使用了装饰器函数 with_comms、skip_if_lt_x_gpu、requires_nccl
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_shard_tensor(self):
        # 定义数据分片规格 spec，使用 ChunkShardingSpec 类，指定维度为0，四个分片的放置位置
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 生成一个大小为 (12, 12) 的随机张量 tensor，将其放置在当前进程的 GPU 上
        tensor = torch.rand(12, 12).cuda(self.rank)
        # 对张量 tensor 进行分片，返回一个 ShardedTensor 对象 st
        st = _shard_tensor(tensor, spec)

        # 验证分片操作是否正确
        self.assertTrue(isinstance(st, sharded_tensor.ShardedTensor))
        local_shard = st.local_tensor()  # 获取本地分片的张量
        self.assertEqual(1, len(st.local_shards()))  # 确保本地分片数量为1
        self.assertEqual(torch.Size([3, 12]), local_shard.size())  # 验证本地分片张量的形状
        self.assertEqual(torch.narrow(tensor, 0, 3 * self.rank, 3), local_shard)  # 验证本地分片的数据是否正确

    # 声明一个测试方法 test_shard_tensor_with_empty_shard，使用了装饰器函数 with_comms、skip_if_lt_x_gpu、requires_nccl
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_shard_tensor_with_empty_shard(self):
        # 定义数据分片规格 spec，使用 ChunkShardingSpec 类，指定维度为0，四个分片的放置位置
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 生成一个大小为 (9, 12) 的随机张量 tensor，将其放置在当前进程的 GPU 上
        tensor = torch.rand(9, 12).cuda(self.rank)
        # 对张量 tensor 进行分片，返回一个 ShardedTensor 对象 st
        st = _shard_tensor(tensor, spec)

        # 验证分片操作是否正确
        self.assertTrue(isinstance(st, sharded_tensor.ShardedTensor))
        sms = st.metadata().shards_metadata  # 获取分片的元数据信息
        self.assertEqual(len(sms), 4)  # 确保分片数为4
        for sm in sms:
            # 遍历每个分片的元数据，确保分片偏移量和分片大小不超过张量的维度
            self.assertTrue(sm.shard_offsets[0] + sm.shard_sizes[0] <= tensor.size(0))

        local_shard = st.local_tensor()  # 获取本地分片的张量
        self.assertEqual(1, len(st.local_shards()))  # 确保本地分片数量为1
        if dist.get_rank() < 3:
            self.assertEqual(torch.Size([3, 12]), local_shard.size())  # 验证本地分片张量的形状
            self.assertEqual(torch.narrow(tensor, 0, 3 * self.rank, 3), local_shard)  # 验证本地分片的数据是否正确
        else:
            self.assertEqual(torch.Size([0, 12]), local_shard.size())  # 如果当前进程排名大于等于3，本地分片应为空张量
    # 定义测试方法，用于测试分片张量时的异常情况处理
    def test_shard_tensor_errors(self):
        # 创建一个分片规格对象，指定维度为0，及其分布位置
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 创建一个在GPU上随机生成的张量，分片维度由当前rank确定
        tensor = torch.rand(12, 12).cuda(self.rank)

        # 使用断言检查是否引发了指定的 ValueError 异常，检查是否与源rank匹配
        with self.assertRaisesRegex(ValueError, "does not match with src_rank"):
            _shard_tensor(tensor, spec, src_rank=self.rank)

        # 使用断言检查是否引发了指定的 ValueError 异常，检查张量是否是连续的
        with self.assertRaisesRegex(ValueError, "not a contiguous Tensor"):
            # 创建一个不连续的张量，并尝试对其进行分片操作
            tensor_t = torch.rand(12, 12).cuda(self.rank).t()
            _shard_tensor(tensor_t, spec)

        # 更新分片规格对象，当前rank位置更新为f字符串表达式，其余保持不变
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                f"rank:{self.rank}/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 使用断言检查是否引发了指定的 ValueError 异常，检查是否与分片规格匹配
        with self.assertRaisesRegex(ValueError, "does not match with sharding_spec"):
            _shard_tensor(tensor, spec)

        # 创建一个枚举分片规格对象，指定每个分片的元数据
        spec = EnumerableShardingSpec(
            [
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
            ]
        )
        # 使用断言检查是否引发了指定的 NotImplementedError 异常，表明功能尚未实现
        with self.assertRaisesRegex(NotImplementedError, "not implemented yet!"):
            _shard_tensor(tensor, spec)
class TestModuleHookApi(ShardedTensorTestBase):
    # 定义一个测试类，继承自ShardedTensorTestBase，用于测试模块钩子API

    class DummyNNModule(torch.nn.Module):
        # 定义一个虚拟的神经网络模块类，继承自torch.nn.Module

        def __init__(self, spec, tensor_size):
            # 初始化方法，接受分片规格和张量大小作为参数

            super().__init__()
            # 调用父类的初始化方法

            self.st = sharded_tensor.rand(spec, *tensor_size)
            # 使用给定的分片规格和张量大小创建一个分片张量对象，并赋值给实例变量self.st

        def forward(self):
            # 前向传播方法，返回当前模块的分片张量对象

            return self.st

    @with_comms(init_rpc=False)
    # 使用with_comms装饰器，禁用RPC初始化的上下文

    @skip_if_lt_x_gpu(4)
    # 使用skip_if_lt_x_gpu装饰器，如果GPU数目少于4，则跳过测试

    @requires_nccl()
    # 使用requires_nccl装饰器，确保NCCL库可用

    def test_reshard_output(self):
        # 测试方法：测试重新分片输出

        specs = _chunk_sharding_specs_list_for_test([0, 1], seed=5)
        # 生成用于测试的分片规格列表，种子为5

        spec, reshard_spec = specs[0], specs[1]
        # 从生成的规格列表中获取第一个规格作为spec，第二个规格作为reshard_spec

        test_module = self.DummyNNModule(spec, [24, 12])
        # 创建一个DummyNNModule的实例test_module，传入spec和张量大小[24, 12]

        st = test_module()
        # 调用test_module的__call__方法，执行前向传播，获取分片张量对象st

        local_shard = st.local_tensor()
        # 获取st的本地张量分片

        pg = dist.distributed_c10d._get_default_group()
        # 获取默认的进程组

        st_compare = ShardedTensor._init_from_local_shards(
            copy.deepcopy(st.local_shards()),
            st.size(),
            process_group=pg,
        )
        # 使用本地分片初始化一个ShardedTensor对象st_compare，保持分片对象的一致性

        st_compare._sharding_spec = copy.deepcopy(spec)
        # 将st_compare的分片规格设置为spec的深拷贝

        st_compare.reshard(reshard_spec)
        # 对st_compare执行重新分片操作，使用reshard_spec

        test_module = _reshard_output(test_module, reshard_spec)
        # 对test_module执行输出重新分片操作，使用reshard_spec

        st = test_module()
        # 调用重新分片后的test_module的__call__方法，获取分片张量对象st

        local_shard = st.local_tensor()
        # 获取st的本地张量分片

        local_shard_compare = st_compare.local_tensor()
        # 获取st_compare的本地张量分片

        self.assertEqual(local_shard, local_shard_compare)
        # 断言两个本地张量分片的内容是否相等

        self.assertEqual(local_shard.size(0), 24)
        # 断言本地张量分片的第一个维度大小为24

        self.assertEqual(local_shard.size(1), 3)
        # 断言本地张量分片的第二个维度大小为3

    @with_comms(init_rpc=False)
    # 使用with_comms装饰器，禁用RPC初始化的上下文

    @skip_if_lt_x_gpu(4)
    # 使用skip_if_lt_x_gpu装饰器，如果GPU数目少于4，则跳过测试

    @requires_nccl()
    # 使用requires_nccl装饰器，确保NCCL库可用

    def test_collect_local_shard(self):
        # 测试方法：测试收集本地分片

        specs = _chunk_sharding_specs_list_for_test([0], seed=5)
        # 生成用于测试的分片规格列表，种子为5

        spec = specs[0]
        # 从生成的规格列表中获取第一个规格作为spec

        test_module = self.DummyNNModule(spec, [23, 15])
        # 创建一个DummyNNModule的实例test_module，传入spec和张量大小[23, 15]

        st = test_module()
        # 调用test_module的__call__方法，执行前向传播，获取分片张量对象st

        local_shard = st.local_tensor()
        # 获取st的本地张量分片

        test_module = _collect_local_shard(test_module)
        # 对test_module执行收集本地分片的操作

        output = test_module()
        # 调用收集本地分片后的test_module的__call__方法，获取输出结果output

        self.assertTrue(isinstance(output, torch.Tensor))
        # 断言输出结果output是torch.Tensor类型的对象

        self.assertEqual(local_shard, output)
        # 断言本地张量分片local_shard与输出结果output相等


class TestLocalTensor(ShardedTensorTestBase):
    # 定义一个测试类，继承自ShardedTensorTestBase，用于测试本地张量相关功能

    @with_comms(init_rpc=False)
    # 使用with_comms装饰器，禁用RPC初始化的上下文

    @skip_if_lt_x_gpu(4)
    # 使用skip_if_lt_x_gpu装饰器，如果GPU数目少于4，则跳过测试

    @requires_nccl()
    # 使用requires_nccl装饰器，确保NCCL库可用

    def test_local_tensor(self):
        # 测试方法：测试获取本地张量

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 创建一个分片规格对象ChunkShardingSpec，指定分片的维度为0，放置在四个GPU上

        st = sharded_tensor.rand(spec, 24, 12)
        # 使用给定的分片规格和张量大小创建一个分片张量对象st，大小为24x12

        local_shard = st.local_tensor()
        # 获取st的本地张量分片

        self.assertEqual(torch.Size([6, 12]), local_shard.size())
        # 断言本地张量分片的大小为[6, 12]

        self.assertEqual(st.local_tensor(), local_shard)
        # 断言再次获取的本地张量分片与之前获取的本地张量分片对象相等
    # 定义一个测试方法，用于测试本地张量错误情况
    def test_local_tensor_error(self):
        # 创建一个块分片规范对象，指定在维度0上进行分片，并指定每个分片的位置
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:1/cuda:1",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:2/cuda:2",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
                "rank:3/cuda:3",
            ],
        )
        # 使用指定的分片规范创建一个随机分布的分片张量，形状为(24, 12)
        st = sharded_tensor.rand(spec, 24, 12)
        # 使用断言检测是否抛出预期的 NotImplementedError 异常，异常信息为"Only single local shard is supported."
        with self.assertRaisesRegex(
            NotImplementedError, "Only single local shard is supported."
        ):
            # 尝试获取本地分片，预期会抛出 NotImplementedError 异常
            local_shard = st.local_tensor()
# 定义一个测试类 TestShardedTensorChunked，继承自 ShardedTensorTestBase
class TestShardedTensorChunked(ShardedTensorTestBase):
    
    # 使用装饰器，确保测试函数在通信环境下运行
    @with_comms
    # 如果 GPU 少于 4 个则跳过该测试函数
    @skip_if_lt_x_gpu(4)
    # 需要使用 NCCL 库
    @requires_nccl()
    # 测试 sharded_tensor_metadata 方法
    def test_sharded_tensor_metadata(self):
        
        # 定义分片规格 spec，沿第 0 维度分片，分别在四个 GPU 上
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        
        # 创建一个空的分片张量 st，形状为 [10, 20]，并初始化远程引用
        st = sharded_tensor.empty(spec, 10, 20, init_rrefs=True)
        # 获取分片张量的元数据
        st_metadata = st.metadata()
        
        # 断言分片张量的形状为 [10, 20]
        self.assertEqual(torch.Size([10, 20]), st_metadata.size)
        # 断言分片张量的形状为 [10, 20]
        self.assertEqual(torch.Size([10, 20]), st.size())
        # 断言分片张量的数据类型为 torch.float
        self.assertEqual(torch.float, st.dtype)
        # 断言分片张量的布局为 torch.strided
        self.assertEqual(torch.strided, st.layout)
        # 断言分片张量不需要梯度计算
        self.assertEqual(False, st.requires_grad)
        # 断言分片张量是连续的
        self.assertTrue(st.is_contiguous())
        # 断言分片张量不是固定在内存中的
        self.assertFalse(st.is_pinned())

        # 创建一个需要梯度计算的分片张量 st
        st = sharded_tensor.empty(spec, 10, 20, requires_grad=True, init_rrefs=True)
        # 断言分片张量需要梯度计算
        self.assertEqual(True, st.requires_grad)

        # 创建一个数据类型为 torch.double 的分片张量 st
        st = sharded_tensor.empty(spec, 10, 20, dtype=torch.double, init_rrefs=True)
        # 断言分片张量的数据类型为 torch.double
        self.assertEqual(torch.double, st.dtype)

        # 需要在 CPU 上执行以进行 pin_memory 测试
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cpu",
                "rank:1/cpu",
                "rank:2/cpu",
                "rank:3/cpu",
            ],
        )

        # 创建一个固定在内存中的分片张量 st
        st = sharded_tensor.empty(spec, 10, 20, pin_memory=True, init_rrefs=True)
        # 断言分片张量固定在内存中
        self.assertEqual(True, st.is_pinned())

        # 测试只读属性，由于不能简单地改变全局元数据而改变底层分片的属性
        with self.assertRaisesRegex(RuntimeError, "torch function '__set__'"):
            # 尝试设置分片张量的 requires_grad 属性，预期会抛出 RuntimeError
            st.requires_grad = True
    # 定义一个测试方法，用于测试完整的世界大小
    def test_complete_world_size(self):
        # 针对每个维度进行迭代测试
        for dim in [0, -2]:
            # 创建分片规格对象，指定维度和放置策略
            spec = ChunkShardingSpec(
                dim=dim,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                    "rank:2/cuda:2",
                    "rank:3/cuda:3",
                ],
            )
            # 使用指定规格创建一个空的分布式张量，初始化分布式引用
            st = sharded_tensor.empty(spec, 10, 20, init_rrefs=True)

            # 验证本地分片
            local_shards = st.local_shards()
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            if self.rank == 3:
                self.assertEqual((1, 20), local_shard.size())
            else:
                self.assertEqual((3, 20), local_shard.size())

            # 验证全局元数据
            st_metadata = st.metadata()
            shards_metadata = st_metadata.shards_metadata
            self.assertEqual(4, len(shards_metadata))

            # 遍历每个分片的元数据进行验证
            for rank, shard_metadata in enumerate(shards_metadata):
                self.assertEqual([rank * 3, 0], shard_metadata.shard_offsets)
                if rank == 3:
                    self.assertEqual([1, 20], shard_metadata.shard_sizes)
                else:
                    self.assertEqual([3, 20], shard_metadata.shard_sizes)
                self.assertEqual(
                    f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement)
                )

            # 验证远程分片
            remote_shards = st.remote_shards()
            self.assertEqual(3, len(remote_shards))

            # 遍历远程分片进行验证
            for rpc_rank, shards in remote_shards.items():
                self.assertEqual(1, len(shards))
                for remote_shard in shards:
                    self.assertEqual(rpc_rank, remote_shard.owner().id)
                    shard = remote_shard.to_here()
                    self.assertEqual(
                        f"rank:{rpc_rank}/cuda:{rpc_rank}",
                        str(shard.metadata.placement),
                    )
                    if rpc_rank == 3:
                        self.assertEqual((1, 20), shard.tensor.size())
                    else:
                        self.assertEqual((3, 20), shard.tensor.size())

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_create_sharded_tensor_with_ones(self):
        """Test sharded_tensor.ones(...)"""

        # 定义分片规格，指定分片维度和每个分片的放置位置
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 10, 20
        # 使用指定规格创建一个全是1的分片张量
        st = sharded_tensor.ones(spec, h, w)

        # 验证本地分片是否使用 torch.ones 进行了初始化
        local_shards = st.local_shards()
        self.assertEqual(1, len(local_shards))
        local_shard = local_shards[0].tensor
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
        # 计算预期的本地分片形状，对于 rank!=3，ceil(h/4)=3，对于 rank=3，为1
        expected_h = 1 if self.rank == 3 else math.ceil(h / 4)
        self.assertEqual((expected_h, w), local_shard.size())
        self.assertEqual(local_shard, torch.ones(expected_h, w))

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_gather_even(self) -> None:
        """Test _sharded_tensor.gather(...) with evenly distributed._shards"""

        # 定义分片规格，指定分片维度和每个分片的放置位置
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 10, 20
        # 使用指定规格创建一个全是1的分片张量
        st = sharded_tensor.ones(spec, h, w)

        full_tensor = None
        dst = 1
        # 如果当前进程的 rank 等于目标 dst，则创建一个全0张量用于收集结果
        if self.rank == dst:
            full_tensor = torch.zeros(
                h,
                w,
                device=torch.device(f"cuda:{dst}"),
            )
        # 执行 gather 操作，将分片张量 st 的数据收集到 full_tensor 中
        st.gather(dst, full_tensor)

        # 如果当前进程的 rank 等于目标 dst，则验证 full_tensor 是否为全1张量
        if self.rank == dst:
            self.assertEqual(full_tensor, torch.ones(h, w))
        else:
            self.assertIsNone(full_tensor)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_gather_uneven(self) -> None:
        """Test _sharded_tensor.gather(...) with unevenly distributed._shards"""

        # 定义分片规格，指定分片维度和每个分片的放置位置（这里有些分片重复）
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
            ],
        )
        h, w = 10, 20
        # 使用指定规格创建一个全是1的分片张量
        st = sharded_tensor.ones(spec, h, w)

        full_tensor = None
        dst = 1
        # 如果当前进程的 rank 等于目标 dst，则创建一个全0张量用于收集结果
        if self.rank == dst:
            full_tensor = torch.zeros(
                h,
                w,
                device=torch.device(f"cuda:{dst}"),
            )
        # 执行 gather 操作，将分片张量 st 的数据收集到 full_tensor 中
        st.gather(dst, full_tensor)

        # 如果当前进程的 rank 等于目标 dst，则验证 full_tensor 是否为全1张量
        if self.rank == dst:
            self.assertEqual(full_tensor, torch.ones(h, w))
        else:
            self.assertIsNone(full_tensor)
    def test_create_sharded_tensor_with_zeros(self):
        """Test sharded_tensor.zeros(...)"""

        # 定义分片规格，将数据在不同设备上分片存储
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 10, 20
        # 使用 sharded_tensor.zeros 创建一个分布式张量
        st = sharded_tensor.zeros(spec, h, w)

        # 验证本地分片是否用 torch.zeros 初始化
        local_shards = st.local_shards()
        self.assertEqual(1, len(local_shards))
        local_shard = local_shards[0].tensor
        # 验证本地分片的设备是否正确
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
        # 计算预期的分片大小
        expected_h = 1 if self.rank == 3 else math.ceil(h / 4)
        self.assertEqual((expected_h, w), local_shard.size())
        # 验证本地分片是否与 torch.zeros 的期望值相等
        self.assertEqual(local_shard, torch.zeros(expected_h, w))

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_create_sharded_tensor_with_rand(self):
        """Test sharded_tensor.rand(...)/randn(...)"""

        # 定义分片规格，将数据在不同设备上分片存储
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 8, 2
        seed = 1234

        expected_h = 2
        expected_device = torch.device(f"cuda:{self.rank}")
        dtype = torch.double
        torch.manual_seed(seed)
        # 测试 sharded_tensor.rand 的创建
        expected = torch.rand(expected_h, w, device=expected_device, dtype=dtype)
        # 重置种子以确保生成相同的随机数
        torch.manual_seed(seed)
        st = sharded_tensor.rand(spec, h, w, dtype=dtype)

        # 验证本地分片是否用 torch.rand 初始化
        local_shards = st.local_shards()
        self.assertEqual(1, len(local_shards))
        local_shard = local_shards[0].tensor
        # 验证本地分片的设备是否正确
        self.assertEqual(expected_device, local_shard.device)
        self.assertEqual((expected_h, w), local_shard.size())
        self.assertEqual(expected, local_shard)

        # 测试 sharded_tensor.randn 的创建
        torch.manual_seed(seed)
        expected_randn = torch.randn(expected_h, w, device=expected_device, dtype=dtype)
        # 重置种子以确保生成相同的随机数
        torch.manual_seed(seed)
        st_randn = sharded_tensor.randn(spec, h, w, dtype=dtype)

        # 验证本地分片是否用 torch.randn 初始化
        local_shards = st_randn.local_shards()
        self.assertEqual(1, len(local_shards))
        local_shard = local_shards[0].tensor
        # 验证本地分片的设备是否正确
        self.assertEqual(expected_device, local_shard.device)
        self.assertEqual((expected_h, w), local_shard.size())
        self.assertEqual(expected_randn, local_shard)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    # 定义测试函数，用于测试 sharded_tensor.full(...) 方法
    def test_create_sharded_tensor_with_full(self):
        """Test sharded_tensor.full(...)"""

        # 定义分片规格，指定维度为0，分布在四个 GPU 上
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        
        # 定义高度和宽度
        h, w = 10, 20
        
        # 填充值
        fill_value = 1234
        
        # 使用 sharded_tensor.full 创建分片张量
        st = sharded_tensor.full(
            spec, size=(h, w), fill_value=fill_value, dtype=torch.int32
        )

        # 验证本地分片是否使用 torch.full 初始化
        local_shards = st.local_shards()
        
        # 断言本地分片数量为1
        self.assertEqual(1, len(local_shards))
        
        # 获取本地分片的张量
        local_shard = local_shards[0].tensor
        
        # 断言本地分片的设备为当前进程的 GPU 设备
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
        
        # 计算预期的本地分片高度
        # 当 rank != 3 时，ceil(h / 4) = 3；当 rank == 3 时，预期高度为1
        expected_h = 1 if self.rank == 3 else math.ceil(h / 4)
        
        # 断言本地分片的形状为 (expected_h, w)
        self.assertEqual((expected_h, w), local_shard.size())
        
        # 断言本地分片的值与使用 torch.full 方法生成的张量相等
        self.assertEqual(
            local_shard,
            torch.full(size=(expected_h, w), fill_value=fill_value, dtype=torch.int32),
        )

    # 使用装饰器配置通信环境
    @with_comms
    # 跳过 GPU 数量小于 4 的情况
    @skip_if_lt_x_gpu(4)
    # 要求使用 NCCL
    @requires_nccl()
    # 定义一个测试方法，用于测试创建类似张量的操作，如 torch.zeros_like(...)、torch.full_like 等
    def test_create_sharded_tensor_like(self):
        """Test tensor like methods, i.e. torch.zeros_like(...), torch.full_like, etc."""

        # 定义分片规格，指定在第0维度上分片，每个分片的位置和设备
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 定义张量的高度和宽度
        h, w = 8, 8
        # 预期的高度（分片后的高度）
        expected_h = 2
        # 设置随机种子
        seed = 1234
        # 张量的数据类型
        dtype = torch.double
        # 预期的设备，根据当前对象的 rank 属性确定 CUDA 设备
        expected_device = torch.device(f"cuda:{self.rank}")
        # 使用分片规格和指定的高度、宽度生成一个随机分片张量
        st = sharded_tensor.rand(spec, (h, w), dtype=dtype)
        # 定义张量类似操作的字典，包括 torch.zeros_like、torch.full_like 等
        tensor_like_ops = {
            torch.zeros_like: torch.zeros,
            torch.ones_like: torch.ones,
            torch.rand_like: torch.rand,
            torch.randn_like: torch.randn,
            torch.empty_like: torch.empty,
            torch.full_like: torch.full,
        }
        # 遍历每个张量类似操作及其对应的预期本地操作
        for op, expect_local_op in tensor_like_ops.items():
            if op == torch.full_like:
                # 对于 torch.full_like，需要额外的 fill_value 参数
                expect_tensor = expect_local_op(
                    (expected_h, w), 8.8, device=expected_device, dtype=dtype
                )
                # 调用 op 方法生成新的分片张量 new_op_st，并断言其本地张量与预期张量相等
                new_op_st = op(st, 8.8, dtype=dtype)
                self.assertEqual(new_op_st.local_tensor(), expect_tensor)
            elif op == torch.empty_like:
                # 对于 torch.empty_like，仅比较形状是否相等
                expect_tensor = expect_local_op(
                    expected_h, w, device=expected_device, dtype=dtype
                )
                # 调用 op 方法生成新的分片张量 new_op_st，并断言其本地张量的形状与预期张量的形状相等
                new_op_st = op(st, dtype=dtype)
                self.assertEqual(new_op_st.local_tensor().shape, expect_tensor.shape)
            else:
                # 其他情况下，设置随机种子，生成预期张量，再次设置随机种子生成新的分片张量，断言它们相等
                torch.manual_seed(seed)
                expect_tensor = expect_local_op(
                    expected_h, w, device=expected_device, dtype=dtype
                )
                torch.manual_seed(seed)
                new_op_st = op(st, dtype=dtype)
                self.assertEqual(new_op_st.local_tensor(), expect_tensor)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    # 定义测试方法，用于验证部分世界大小功能
    def test_partial_world_size(self):
        # 创建分块分片规格对象，指定分块的维度为0
        spec = ChunkShardingSpec(
            dim=0,
            # 指定不同分片的放置规则，每条规则格式为"rank:<rank>/cuda:<cuda>"
            placements=[
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 使用给定规格创建一个空的分布式张量对象，尺寸为10x20，初始化分布式引用
        st = sharded_tensor.empty(spec, 10, 20, init_rrefs=True)

        # 验证本地分片
        local_shards = st.local_shards()
        # 如果当前进程的 rank 大于等于2
        if self.rank >= 2:
            # 断言本地分片数量为1
            self.assertEqual(1, len(local_shards))
            # 获取本地分片的张量数据
            local_shard = local_shards[0].tensor
            # 断言本地分片的设备为当前 rank 对应的 CUDA 设备
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            # 断言本地分片的尺寸为(5, 20)
            self.assertEqual((5, 20), local_shard.size())
        else:
            # 如果当前进程的 rank 小于2，断言本地分片数量为0
            self.assertEqual(0, len(local_shards))

        # 验证全局元数据
        st_metadata = st.metadata()
        # 获取分片的元数据列表
        shards_metadata = st_metadata.shards_metadata
        # 断言分片元数据列表长度为2
        self.assertEqual(2, len(shards_metadata))

        # 遍历分片元数据列表，验证每个分片的偏移、尺寸和放置规则
        for shard_rank, shard_metadata in enumerate(shards_metadata):
            # 断言分片偏移为 [shard_rank * 5, 0]
            self.assertEqual([shard_rank * 5, 0], shard_metadata.shard_offsets)
            # 断言分片尺寸为 [5, 20]
            self.assertEqual([5, 20], shard_metadata.shard_sizes)
            # 断言分片放置规则为 "rank:<shard_rank + 2>/cuda:<shard_rank + 2>"
            self.assertEqual(
                f"rank:{shard_rank + 2}/cuda:{shard_rank + 2}",
                str(shard_metadata.placement),
            )

        # 验证远程分片
        remote_shards = st.remote_shards()
        # 如果当前进程的 rank 大于等于2
        if self.rank >= 2:
            # 断言远程分片数量为1
            self.assertEqual(1, len(remote_shards))
        else:
            # 如果当前进程的 rank 小于2，断言远程分片数量为2
            self.assertEqual(2, len(remote_shards))

        # 遍历远程分片字典，验证每个 RPC 进程的远程分片
        for rpc_rank, shards in remote_shards.items():
            # 断言每个 RPC 进程的远程分片数量为1
            self.assertEqual(1, len(shards))
            # 遍历每个远程分片，验证其所有者及其元数据的放置规则和尺寸
            for remote_shard in shards:
                # 断言远程分片的所有者为当前 RPC 进程的 rank
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                # 将远程分片拉取到本地
                shard = remote_shard.to_here()
                # 断言远程分片的放置规则为 "rank:<rpc_rank>/cuda:<rpc_rank>"
                self.assertEqual(
                    f"rank:{rpc_rank}/cuda:{rpc_rank}", str(shard.metadata.placement)
                )
                # 断言远程分片的张量尺寸为(5, 20)
                self.assertEqual((5, 20), shard.tensor.size())

    # 使用通信装饰器包装测试方法
    @with_comms
    # 如果 GPU 数量少于4，跳过此测试
    @skip_if_lt_x_gpu(4)
    # 要求 NCCL 库可用
    @requires_nccl()
    # 定义一个测试方法，用于测试分布式数据并行环境下的新组功能
    def test_new_group(self):
        # 定义分片分布规范，指定分片的维度为0，以及各个分片的部署位置
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        # 创建一个新的分布式进程组，包含 ranks=[1, 2, 3] 中的进程
        pg = dist.new_group(ranks=[1, 2, 3])
        
        # 创建一个空的分片张量，指定分片规范、张量大小为 (10, 20)，使用上述进程组和初始化的远程引用
        st = sharded_tensor.empty(spec, 10, 20, process_group=pg, init_rrefs=True)

        # 验证本地分片
        local_shards = st.local_shards()
        if self.rank >= 2:
            # 如果当前进程的 rank 大于等于 2，则期望有一个本地分片
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            self.assertEqual((5, 20), local_shard.size())
        else:
            # 否则期望没有本地分片
            self.assertEqual(0, len(local_shards))

        # 验证全局元数据
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        # 期望有两个分片的元数据
        self.assertEqual(2, len(shards_metadata))

        # 遍历验证各个分片的元数据
        for shard_rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual([shard_rank * 5, 0], shard_metadata.shard_offsets)
            self.assertEqual([5, 20], shard_metadata.shard_sizes)
            # 验证分片的部署位置信息
            self.assertEqual(
                f"rank:{shard_rank + 2}/cuda:{shard_rank + 2}",
                str(shard_metadata.placement),
            )

        # 验证远程分片
        remote_shards = st.remote_shards()
        if self.rank >= 2:
            # 如果当前进程的 rank 大于等于 2，则期望有一个远程分片
            self.assertEqual(1, len(remote_shards))
        else:
            # 否则期望有两个远程分片
            self.assertEqual(2, len(remote_shards))

        # 遍历验证各个远程分片的信息
        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(1, len(shards))
            for remote_shard in shards:
                # 获取远程分片数据，并验证其属性
                shard = remote_shard.to_here()
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                self.assertEqual(
                    f"rank:{rpc_rank}/cuda:{rpc_rank}", str(shard.metadata.placement)
                )
                self.assertEqual((5, 20), shard.tensor.size())

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_multiple_local_shards(self):
        # 定义分片规格，指定在第一维度进行分片，以及分片的放置情况
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 创建一个空的分片张量，使用给定的分片规格，初始形状为 (16, 20)，并初始化远程引用
        st = sharded_tensor.empty(spec, 16, 20, init_rrefs=True)

        # 验证本地分片
        local_shards = st.local_shards()
        self.assertEqual(2, len(local_shards))  # 确保本地分片数量为 2
        for local_shard in local_shards:
            # 确保本地分片所在设备为当前进程的 CUDA 设备
            self.assertEqual(
                torch.device(f"cuda:{self.rank}"), local_shard.tensor.device
            )
            # 确保本地分片的形状为 (2, 20)
            self.assertEqual((2, 20), local_shard.tensor.size())

        # 验证全局元数据
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(8, len(shards_metadata))  # 确保分片元数据条目数为 8

        for shard_idx, shard_metadata in enumerate(shards_metadata):
            # 确保分片的偏移为 [shard_idx * 2, 0]
            self.assertEqual([shard_idx * 2, 0], shard_metadata.shard_offsets)
            # 确保分片的形状为 [2, 20]
            self.assertEqual([2, 20], shard_metadata.shard_sizes)
            # 确保分片的放置情况字符串符合预期格式
            self.assertEqual(
                f"rank:{shard_idx % 4}/cuda:{shard_idx % 4}",
                str(shard_metadata.placement),
            )

        # 验证远程分片
        remote_shards = st.remote_shards()
        self.assertEqual(3, len(remote_shards))  # 确保远程分片数量为 3
        owners = {}
        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(2, len(shards))  # 确保每个 RPC 等级有 2 个远程分片
            for remote_shard in shards:
                # 将远程分片拉取到本地，确保其形状为 (2, 20)
                shard = remote_shard.to_here()
                self.assertEqual((2, 20), shard.tensor.size())
                # 确保远程分片的所有者是当前 RPC 等级
                self.assertEqual(rpc_rank, remote_shard.owner().id)

    @skip_if_lt_x_gpu(4)  # 如果当前 GPU 数量小于 4，则跳过该测试用例
    @requires_nccl()  # 需要 NCCL 库支持
    # 定义一个测试方法，用于测试分片列功能
    def test_sharding_columns(self):
        # 初始化 PostgreSQL 数据库连接
        self.init_pg()

        # 遍历维度列表进行测试，维度包括正向和反向
        for dim in [1, -1]:
            # 创建分片规格对象，指定维度和分布位置
            spec = ChunkShardingSpec(
                dim=dim,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                    "rank:2/cuda:2",
                    "rank:3/cuda:3",
                ],
            )

            # 使用分片规格对象创建空的分片张量
            st = sharded_tensor.empty(spec, 10, 32)

            # 验证本地分片
            local_shards = st.local_shards()
            # 断言本地分片数量为1
            self.assertEqual(1, len(local_shards))
            # 获取本地分片张量
            local_shard = local_shards[0].tensor
            # 断言本地分片张量的设备是当前 rank 对应的 CUDA 设备
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            # 断言本地分片张量的形状为 (10, 8)
            self.assertEqual((10, 8), local_shard.size())

            # 验证全局元数据
            st_metadata = st.metadata()
            # 获取分片元数据列表
            shards_metadata = st_metadata.shards_metadata
            # 断言分片元数据列表长度为4
            self.assertEqual(4, len(shards_metadata))

            # 遍历分片元数据列表，验证每个分片的偏移、大小和放置位置
            for rank, shard_metadata in enumerate(shards_metadata):
                # 断言分片偏移
                self.assertEqual([0, rank * 8], shard_metadata.shard_offsets)
                # 断言分片大小
                self.assertEqual([10, 8], shard_metadata.shard_sizes)
                # 断言分片放置位置字符串表示
                self.assertEqual(
                    f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement)
                )
    # 定义测试无效分片的方法，这里的 self 是测试类的实例
    def test_invalid_sharding(self):
        # 初始化虚拟的分布式进程组
        self.init_pg()

        # 确保在创建 ChunkShardingSpec 对象时，抛出 NotImplementedError 异常，提示不支持命名维度
        with self.assertRaisesRegex(
            NotImplementedError, "does not support named dimension"
        ):
            # 创建具有指定维度和位置的 ChunkShardingSpec 对象，并调用 sharded_tensor.empty 方法
            spec = ChunkShardingSpec(dim="H", placements=["rank:1/cuda:1"])
            sharded_tensor.empty(spec, 10, 20)

        # 遍历多个维度值，确保在维度值无效时，抛出 ValueError 异常，提示无效的分片维度
        for dim in [2, 3, 4, -3, -4, -5]:
            spec = ChunkShardingSpec(dim=dim, placements=["rank:1/cuda:1"])
            with self.assertRaisesRegex(ValueError, "Invalid sharding dim"):
                sharded_tensor.empty(spec, 10, 20)

        # 创建一个具有全局排名 5 的 ChunkShardingSpec 对象，并确保在进程组中不存在该全局排名时，抛出 ValueError 异常
        spec = ChunkShardingSpec(dim=0, placements=["rank:5/cuda:1"])
        with self.assertRaisesRegex(
            ValueError, "Global rank 5 does not exist in input process group"
        ):
            sharded_tensor.empty(spec, 10, 20)

        # 创建一个具有全局排名 0 的 ChunkShardingSpec 对象，并确保在初始化 ShardedTensor 时调用了 torch.add 方法时，抛出 RuntimeError 异常
        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:1"])
        st = sharded_tensor.empty(spec, 10, 20)
        tensor = torch.empty(10, 20)
        with self.assertRaisesRegex(
            RuntimeError, r".*not supported for ShardedTensor!$"
        ):
            torch.add(st, tensor)

        # 创建一个具有全局排名 0 的 ChunkShardingSpec 对象，并确保在指定了非稠密布局时，抛出 ValueError 异常
        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:1"])
        with self.assertRaisesRegex(
            ValueError, "Only torch.strided layout is currently supported"
        ):
            sharded_tensor.empty(spec, 10, 20, layout=torch.sparse_coo)

        # 创建一个具有全局排名 0 的 ChunkShardingSpec 对象，并确保在指定了非连续内存格式时，抛出 ValueError 异常
        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:1"])
        with self.assertRaisesRegex(
            ValueError,
            "Only torch.contiguous_format memory_format is currently supported",
        ):
            sharded_tensor.empty(spec, 10, 20, memory_format=torch.channels_last)

        # 创建一个具有 worker0 名称的 ChunkShardingSpec 对象，并确保在 RPC 框架未初始化时，抛出 RuntimeError 异常
        spec = ChunkShardingSpec(dim=0, placements=["worker0/cuda:1"])
        with self.assertRaisesRegex(
            RuntimeError, "RPC framework needs to be initialized"
        ):
            sharded_tensor.empty(spec, 10, 20)

        # 创建一个具有全局排名 0 的 ChunkShardingSpec 对象，并确保在初始化 ShardedTensor 时调用了 torch.add 方法时，抛出 RuntimeError 异常
        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:1"])
        with self.assertRaisesRegex(
            RuntimeError, "RPC Framework needs to be initialized"
        ):
            st = sharded_tensor.empty(spec, 10, 20, init_rrefs=True)

        # 确保在未初始化远程参考时，试图调用 remote_shards 方法时，抛出 RuntimeError 异常
        with self.assertRaisesRegex(
            RuntimeError, "ShardedTensor created with init_rrefs=False"
        ):
            st = sharded_tensor.empty(spec, 10, 20)
            st.remote_shards()

        # 初始化 RPC 框架
        self.init_rpc()

        # 创建一个具有 workerfoo 名称的 ChunkShardingSpec 对象，并确保在指定了无效工作器名称时，抛出 ValueError 异常
        spec = ChunkShardingSpec(dim=0, placements=["workerfoo/cuda:1"])
        with self.assertRaisesRegex(ValueError, "Invalid worker name"):
            sharded_tensor.empty(spec, 10, 20, init_rrefs=True)
    def test_invalid_pg_rpc_ranks(self):
        self.init_pg()

        # 初始化 RPC，使用不同的 ranks。
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            _transports=tp_transports()
        )
        # 设置初始化方法为文件路径格式
        rpc_backend_options.init_method = f"file://{self.file_name}"
        # 计算新的 rank，确保在 world_size 范围内循环
        rank = (self.rank + 1) % self.world_size
        # 初始化 RPC
        rpc.init_rpc(
            name=f"worker{rank}",
            rank=rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )

        # 创建分片规格
        spec = ChunkShardingSpec(dim=0, placements=["rank:1/cuda:1"])
        # 验证是否抛出特定异常
        with self.assertRaisesRegex(
            ValueError, "Default ProcessGroup and RPC ranks must be the same"
        ):
            # 创建空的分片张量
            sharded_tensor.empty(spec, 10, 20, init_rrefs=True)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_insufficient_sharding_dims(self):
        self.init_pg()

        # 创建分片规格
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 创建空的分片张量
        st = sharded_tensor.empty(spec, 2, 20)

        # 验证本地分片
        local_shards = st.local_shards()
        if self.rank <= 1:
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            self.assertEqual((1, 20), local_shard.size())
        else:
            self.assertEqual(1, len(local_shards))
            local_shard = local_shards[0].tensor
            self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.device)
            self.assertEqual(local_shard.numel(), 0)

        # 验证全局元数据
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(4, len(shards_metadata))

        for shard_rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual([shard_rank, 0], shard_metadata.shard_offsets)
            self.assertEqual(
                f"rank:{shard_rank}/cuda:{shard_rank}", str(shard_metadata.placement)
            )
            if shard_rank <= 1:
                self.assertEqual([1, 20], shard_metadata.shard_sizes)
            else:
                self.assertEqual([0, 20], shard_metadata.shard_sizes)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    # 定义测试方法，用于验证分片张量的大小设置是否正确
    def test_sharded_tensor_sizes(self):
        # 定义分片设置规范，指定分片的维度为0，并指定每个分片的位置
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        # 使用不同方式调用sharded_tensor.empty方法进行测试，传入 *args 形式的参数
        st = sharded_tensor.empty(spec, 10, 20, init_rrefs=True)
        # 验证分片张量的大小是否符合预期
        self.assertEqual(torch.Size([10, 20]), st.size())

        # 使用单个 *args 形式的参数进行测试
        st = sharded_tensor.empty(spec, 10, init_rrefs=True)
        # 验证分片张量的大小是否符合预期
        self.assertEqual(torch.Size([10]), st.size())

        # 使用列表形式的参数进行测试
        st = sharded_tensor.empty(spec, [10, 20], init_rrefs=True)
        # 验证分片张量的大小是否符合预期
        self.assertEqual(torch.Size([10, 20]), st.size())

        # 使用元组形式的参数进行测试
        st = sharded_tensor.empty(spec, (10, 20), init_rrefs=True)
        # 验证分片张量的大小是否符合预期
        self.assertEqual(torch.Size([10, 20]), st.size())

        # 测试通过索引访问张量的行大小
        st = sharded_tensor.empty(spec, (10, 20), init_rrefs=True)
        # 验证分片张量的第0维的大小是否符合预期
        self.assertEqual(st.size(0), 10)

        # 测试通过索引访问张量的列大小
        st = sharded_tensor.empty(spec, (10, 20), init_rrefs=True)
        # 验证分片张量的第1维的大小是否符合预期
        self.assertEqual(st.size(1), 20)

        # 测试通过负索引访问张量的大小
        st = sharded_tensor.empty(spec, (10, 20), init_rrefs=True)
        # 验证分片张量的倒数第1维的大小是否符合预期
        self.assertEqual(st.size(-1), 20)

        # 测试通过dim方法获取张量的维度
        self.assertEqual(st.dim(), 2)
        # 测试通过ndim属性获取张量的维度
        self.assertEqual(st.ndim, 2)

        # 测试对于无效输入是否抛出正确的异常
        st = sharded_tensor.empty(spec, (10, 20), init_rrefs=True)
        # 验证当索引超出张量维度范围时是否抛出IndexError异常
        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            st.size(-3)
        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            st.size(2)

        # 测试对于非法类型的输入是否抛出TypeError异常
        with self.assertRaises(TypeError):
            st = sharded_tensor.empty(spec, "foo")

    # 使用装饰器定义的测试环境，要求至少有4个GPU
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_state_dict(self):
        # 定义分片规格对象，指定分片维度为0，并指定各设备的放置方式
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        # 创建一个基于给定规格的分片模型对象
        m = MyShardedModel1(spec)

        # 测试保存状态字典
        # 注册状态字典钩子函数
        m._register_state_dict_hook(state_dict_hook)
        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 获取当前模型的状态字典
        mod_state_dict = m.state_dict()
        # 获取状态字典中的键集合
        mod_state_keys = mod_state_dict.keys()
        # 断言验证特定键是否存在于状态字典的键集合中
        self.assertTrue("sharded_tensor1" in mod_state_keys)
        self.assertTrue("submodule.sharded_tensor2" in mod_state_keys)
        # 将状态字典保存到字节流缓冲区中
        torch.save(mod_state_dict, buffer)

        # 测试加载状态字典
        # 创建一个新的分片模型对象，无需传入任何规格参数
        module_load = MyShardedModel1()
        # 注册加载状态字典前钩子函数
        module_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)

        # 将缓冲区指针移动到字节流开头
        buffer.seek(0)
        # 从字节流中加载状态字典
        state_dict_deser = torch.load(buffer)
        # 使用加载的状态字典来装载模型状态
        module_load.load_state_dict(state_dict_deser, strict=False)

        # 再次注册状态字典钩子函数
        module_load._register_state_dict_hook(state_dict_hook)
        # 获取加载后模型的状态字典键集合
        loaded_dict_keys = module_load.state_dict().keys()
        # 断言验证特定键是否存在于加载后模型的状态字典中
        self.assertTrue("sharded_tensor1" in loaded_dict_keys)
        self.assertTrue("submodule.sharded_tensor2" in loaded_dict_keys)
        # 在加载后进行验证
        # 断言验证两个分片张量是否相等
        self.assertTrue(torch.equal(m.sharded_tensor1, module_load.sharded_tensor1))
        # 断言验证子模块中的分片张量是否相等
        self.assertTrue(
            torch.equal(
                m.submodule.sharded_tensor2, module_load.submodule.sharded_tensor2
            )
        )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_state_dict_new_group(self):
        # 定义新的分片规格对象，指定分片维度为0，并指定不同的设备放置方式
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:2/cuda:0",
                "rank:3/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        # 创建一个基于给定规格和进程组的分片模型对象
        pg = dist.new_group([2, 3])
        m = MyShardedModel1(spec, pg)

        # 测试保存状态字典
        # 注册状态字典钩子函数
        m._register_state_dict_hook(state_dict_hook)
        # 创建一个字节流缓冲区
        buffer = io.BytesIO()
        # 将当前模型的状态字典保存到字节流缓冲区中
        torch.save(m.state_dict(), buffer)

        # 测试加载状态字典
        # 创建一个新的分片模型对象，不传入任何规格参数，但使用相同的进程组
        module_load = MyShardedModel1(spec=None, group=pg)
        # 注册加载状态字典前钩子函数
        module_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)

        # 将缓冲区指针移动到字节流开头
        buffer.seek(0)
        # 使用相同的进程组加载状态字典
        with load_with_process_group(pg):
            state_dict_deser = torch.load(buffer)
            module_load.load_state_dict(state_dict_deser, strict=False)

        # 在加载后进行验证
        # 断言验证两个分片张量是否相等
        self.assertTrue(torch.equal(m.sharded_tensor1, module_load.sharded_tensor1))
        # 断言验证子模块中的分片张量是否相等
        self.assertTrue(
            torch.equal(
                m.submodule.sharded_tensor2, module_load.submodule.sharded_tensor2
            )
        )
    # 测试不含分片张量的状态字典
    def test_state_dict_no_sharded_tensors(self):
        # 创建一个线性模型，输入和输出均为10
        m = torch.nn.Linear(10, 10)

        # 测试保存操作
        # 获取保存前的状态字典
        state_dict_before = m.state_dict()
        # 注册状态字典钩子
        m._register_state_dict_hook(state_dict_hook)
        # 创建一个字节流缓冲区，并将模型的状态字典保存到其中
        buffer = io.BytesIO()
        torch.save(m.state_dict(), buffer)
        # 断言保存前后状态字典一致
        self.assertEqual(state_dict_before, m.state_dict())

        # 测试加载操作
        # 创建另一个线性模型用于加载
        module_load = torch.nn.Linear(10, 10)
        # 注册加载状态字典前钩子
        module_load._register_load_state_dict_pre_hook(pre_load_state_dict_hook, True)

        # 将缓冲区指针位置移动到起始处
        buffer.seek(0)
        # 从缓冲区加载状态字典
        state_dict_deser = torch.load(buffer)
        # 使用加载的状态字典更新模型参数，允许非严格加载
        module_load.load_state_dict(state_dict_deser, strict=False)

        # 验证加载后的模型参数是否一致
        self.assertEqual(m.weight, module_load.weight)
        self.assertEqual(m.bias, module_load.bias)

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_load_state_dict_errors(self):
        # 初始化 RPC
        self.init_rpc()

        # 初始化分布式进程组
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

        # 创建分片规格
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        # 创建分片模型
        m = MyShardedModel1(spec)

        # 测试保存操作
        # 注册状态字典钩子
        m._register_state_dict_hook(state_dict_hook)
        # 创建字节流缓冲区并保存模型状态字典
        buffer = io.BytesIO()
        torch.save(m.state_dict(), buffer)

        # 创建新的进程组
        pg = dist.new_group(ranks=[0, 2, 3])

        # 将缓冲区指针位置移动到起始处
        buffer.seek(0)
        # 如果当前进程不是rank为0，则断言捕获运行时错误
        if self.rank != 0:
            with self.assertRaisesRegex(RuntimeError, "Local rank at save time was"):
                with load_with_process_group(pg):
                    state_dict_deser = torch.load(buffer)
        else:
            # 否则，断言捕获运行时错误
            with self.assertRaisesRegex(
                RuntimeError, "Local world size at save time was"
            ):
                with load_with_process_group(pg):
                    state_dict_deser = torch.load(buffer)

        # 销毁进程组
        dist.destroy_process_group()

        # 将缓冲区指针位置移动到起始处
        buffer.seek(0)
        # 断言捕获运行时错误，需要初始化默认进程组
        with self.assertRaisesRegex(
            RuntimeError, "Need to initialize default process group"
        ):
            state_dict_deser = torch.load(buffer)
        # 关闭 RPC
        rpc.shutdown()

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_cleanup(self):
        # 创建张量函数
        def create_tensors():
            # 创建分片规格
            spec = ChunkShardingSpec(
                dim=0,
                placements=[
                    "rank:0/cuda:0",
                    "rank:1/cuda:1",
                    "rank:2/cuda:2",
                    "rank:3/cuda:3",
                ],
            )
            # 创建空的分片张量和未初始化的分片张量
            st1 = sharded_tensor.empty(spec, 10, 20, init_rrefs=True)
            st2 = sharded_tensor.empty(spec, 10, 20)

        # 调用创建张量函数
        create_tensors()
        # 断言确保_sharded_tensor_map中没有张量对象
        self.assertEqual(0, len(sharded_tensor.api._sharded_tensor_map))
# 定义一个测试类，继承自ShardedTensorTestBase，用于测试ShardedTensor的可枚举性
class TestShardedTensorEnumerable(ShardedTensorTestBase):
    
    # 标记为需要通信支持的测试方法
    @with_comms
    # 如果GPU数少于4，则跳过测试
    @skip_if_lt_x_gpu(4)
    # 要求使用NCCL进行通信
    @requires_nccl()
    def test_sharded_tensor_metadata(self):
        # 定义一个可枚举分片规范的对象spec
        spec = EnumerableShardingSpec(
            [
                # 第一个分片的元数据
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                # 第二个分片的元数据
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
                # 第三个分片的元数据
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:2/cuda:2",
                ),
                # 第四个分片的元数据
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="rank:3/cuda:3",
                ),
            ]
        )

        # 创建一个空的ShardedTensor对象st，根据spec定义的分片规范
        st = sharded_tensor.empty(spec, 10, 10, init_rrefs=True)
        # 获取并检查ShardedTensor对象的元数据st_metadata
        st_metadata = st.metadata()
        self.assertEqual(torch.Size([10, 10]), st_metadata.size)
        self.assertEqual(torch.float, st.dtype)
        self.assertEqual(torch.strided, st.layout)
        self.assertEqual(False, st.requires_grad)
        self.assertTrue(st.is_contiguous())
        self.assertFalse(st.is_pinned())

        # 创建一个带有requires_grad=True的空的ShardedTensor对象st
        st = sharded_tensor.empty(spec, 10, 10, requires_grad=True, init_rrefs=True)
        self.assertEqual(True, st.requires_grad)

        # 创建一个指定dtype=torch.double的空的ShardedTensor对象st
        st = sharded_tensor.empty(spec, 10, 10, dtype=torch.double, init_rrefs=True)
        self.assertEqual(torch.double, st.dtype)

        # 需要CPU支持以进行pin_memory
        # 定义一个新的可枚举分片规范的对象spec
        spec = EnumerableShardingSpec(
            [
                # 第一个分片的元数据
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cpu",
                ),
                # 第二个分片的元数据
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="rank:1/cpu",
                ),
                # 第三个分片的元数据
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:2/cpu",
                ),
                # 第四个分片的元数据
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="rank:3/cpu",
                ),
            ]
        )

        # 创建一个带有pin_memory=True的空的ShardedTensor对象st
        st = sharded_tensor.empty(spec, 10, 10, pin_memory=True, init_rrefs=True)
        self.assertTrue(st.is_pinned())

    # 标记为需要通信支持的测试方法
    @with_comms
    # 如果GPU数少于4，则跳过测试
    @skip_if_lt_x_gpu(4)
    # 要求使用NCCL进行通信
    @requires_nccl()
    # 定义测试函数 test_grid_sharding，用于验证网格分片功能
    def test_grid_sharding(self):
        # 创建一个分片规范对象 spec，包含四个 ShardMetadata 元数据对象
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:2/cuda:2",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="rank:3/cuda:3",
                ),
            ]
        )

        # 使用 spec 创建一个空的分片张量 st，形状为 (10, 10)，并初始化远程引用
        st = sharded_tensor.empty(spec, 10, 10, init_rrefs=True)
        # 断言张量 st 的形状为 (10, 10)
        self.assertEqual((10, 10), st.size())
        # 断言本地分片的数量为 1
        self.assertEqual(1, len(st.local_shards()))

        # 验证本地分片。
        local_shard = st.local_shards()[0]
        # 断言本地分片的设备为 "cuda:{self.rank}"
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.tensor.device)
        # 断言本地分片的形状为 (5, 5)
        self.assertEqual((5, 5), local_shard.tensor.size())

        # 验证本地分片元数据。
        self.assertEqual(
            (self.rank // 2 * 5, (self.rank % 2) * 5),
            local_shard.metadata.shard_offsets,
        )
        # 断言本地分片的尺寸为 (5, 5)
        self.assertEqual((5, 5), local_shard.metadata.shard_sizes)
        # 断言本地分片的放置位置字符串为 "rank:{self.rank}/cuda:{self.rank}"
        self.assertEqual(
            f"rank:{self.rank}/cuda:{self.rank}", str(local_shard.metadata.placement)
        )

        # 验证全局元数据。
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        # 断言分片元数据列表的长度为 4
        self.assertEqual(4, len(shards_metadata))
        for rank, shard_metadata in enumerate(shards_metadata):
            # 断言每个分片的偏移量
            self.assertEqual(
                (rank // 2 * 5, (rank % 2) * 5), shard_metadata.shard_offsets
            )
            # 断言每个分片的尺寸为 (5, 5)
            self.assertEqual((5, 5), shard_metadata.shard_sizes)
            # 断言每个分片的放置位置字符串为 "rank:{rank}/cuda:{rank}"
            self.assertEqual(f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement))

        # 验证远程分片。
        remote_shards = st.remote_shards()
        # 断言远程分片字典的长度为 3
        self.assertEqual(3, len(remote_shards))

        for rpc_rank, shards in remote_shards.items():
            # 对于每个 RPC 排序，断言其分片列表的长度为 1
            self.assertEqual(1, len(shards))
            for remote_shard in shards:
                # 断言远程分片的所有者 ID 与 rpc_rank 相符
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                shard = remote_shard.to_here()
                # 断言远程拉取的分片的形状为 (5, 5)
                self.assertEqual((5, 5), shard.tensor.size())

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_create_sharded_tensor_with_ones(self):
        """Test sharded_tensor.ones(...)"""

        # 定义分片规格，描述分片的元数据和放置位置
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:2/cuda:2",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="rank:3/cuda:3",
                ),
            ]
        )

        # 创建一个全是1的分布式张量
        st = sharded_tensor.ones(spec, 10, 10, init_rrefs=True)
        self.assertEqual((10, 10), st.size())
        self.assertEqual(1, len(st.local_shards()))

        # 验证本地分片是否使用 torch.ones 进行初始化
        local_shard = st.local_shards()[0]
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.tensor.device)
        self.assertEqual((5, 5), local_shard.tensor.size())
        self.assertEqual(local_shard.tensor, torch.ones(5, 5))

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_gather_even(self) -> None:
        """Test _sharded_tensor.gather(...) with evenly distributed._shards"""

        # 定义分片规格，描述分片的元数据和放置位置
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:2/cuda:2",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="rank:3/cuda:3",
                ),
            ]
        )

        h, w = 10, 10
        # 创建一个全是1的分布式张量
        st = sharded_tensor.ones(spec, h, w, init_rrefs=True)

        full_tensor = None
        dst = 0
        if self.rank == dst:
            # 如果当前进程是目标进程，创建一个全是0的张量
            full_tensor = torch.zeros(h, w, device=torch.device(f"cuda:{dst}"))
        # 将分片收集到目标进程的 full_tensor 中
        st.gather(dst, full_tensor)

        if self.rank == dst:
            # 如果当前进程是目标进程，验证收集到的张量是否全是1
            self.assertEqual(full_tensor, torch.ones(h, w))
        else:
            # 如果当前进程不是目标进程，full_tensor 应为 None
            self.assertIsNone(full_tensor)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    # 定义测试函数，用于测试在不均匀分布的_shards情况下的_sharded_tensor.gather(...)方法
    def test_gather_uneven(self) -> None:
        """Test _sharded_tensor.gather(...) with unevenly distributed _shards"""

        # 定义一个分片规范，描述了多个Shard的元数据和位置信息
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="rank:3/cuda:3",
                ),
            ]
        )

        # 设置高度和宽度变量
        h, w = 10, 10

        # 创建一个sharded_tensor实例，所有元素初始化为1，根据给定的分片规范和大小
        st = sharded_tensor.ones(spec, h, w, init_rrefs=True)

        # 初始化一个空的完整张量
        full_tensor = None

        # 设置目标设备编号
        dst = 0

        # 如果当前进程的rank等于目标设备编号
        if self.rank == dst:
            # 创建一个全零张量，设备是cuda:0
            full_tensor = torch.zeros(h, w, device=torch.device(f"cuda:{dst}"))

        # 将分片的张量收集到full_tensor中
        st.gather(dst, full_tensor)

        # 如果当前进程的rank等于目标设备编号
        if self.rank == dst:
            # 断言full_tensor的内容与全1张量相等
            self.assertEqual(full_tensor, torch.ones(h, w))
        else:
            # 否则，确保full_tensor为空
            self.assertIsNone(full_tensor)

    # 标记函数使用通信组件装饰器
    @with_comms
    # 如果GPU数少于4，则跳过此测试
    @skip_if_lt_x_gpu(4)
    # 要求使用NCCL库
    @requires_nccl()
    # 标记函数使用通信组件装饰器
    @with_comms
    # 如果GPU数少于4，则跳过此测试
    @skip_if_lt_x_gpu(4)
    # 要求使用NCCL库
    @requires_nccl()
    # 定义一个测试方法，用于测试将分片张量转移到 CUDA 设备上的情况
    def test_sharded_tensor_to_cuda(self):
        # 定义一个用于 CPU 的分片规格对象，指定在不同的 CPU 上进行分片
        cpu_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cpu",
                "rank:1/cpu",
                "rank:2/cpu",
                "rank:3/cpu",
            ],
        )
        # 定义一个用于 CUDA 的分片规格对象，指定在不同的 CUDA 设备上进行分片
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 定义张量的高度和宽度
        h, w = 10, 20
        
        # 创建一个在 CUDA 上的分片张量，返回新的 ShardedTensor，但本地分片保持不变（无移动）
        st_cuda = sharded_tensor.zeros(spec, h, w)
        # 将 st_cuda 移动到 CUDA 设备上，返回新的 ShardedTensor
        new_st_cuda = st_cuda.cuda()
        # 断言 st_cuda 和 new_st_cuda 不是同一个对象
        self.assertTrue(st_cuda is not new_st_cuda)
        # 断言 st_cuda 的本地张量与 new_st_cuda 的本地张量是同一个对象
        self.assertTrue(st_cuda.local_tensor() is new_st_cuda.local_tensor())

        # 创建一个使用 gloo 后端的新分组对象
        gloo_pg = dist.new_group(backend="gloo")

        # 创建一个在 CPU 上的分片张量，并指定使用 gloo_pg 进程组
        st_cpu = sharded_tensor.zeros(cpu_spec, h, w, process_group=gloo_pg)
        # 将 st_cpu 移动到 CUDA 设备上
        new_st_gpu = st_cpu.cuda()
        
        # 检查移动后的分片规格是否仍然是 ChunkShardingSpec 类型
        spec_after_move = new_st_gpu.sharding_spec()
        self.assertIsInstance(spec_after_move, ChunkShardingSpec)
        
        # 检查移动前后的分片规格几乎相同，除了设备位置不同
        spec_before_move = st_cpu.sharding_spec()
        self.assertEqual(spec_before_move.dim, spec_after_move.dim)
        self.assertEqual(len(spec_before_move.placements), len(spec_after_move.placements))
        for i, remote_device_after in enumerate(spec_after_move.placements):
            remote_device_before = spec_before_move.placements[i]
            self.assertEqual(remote_device_before.rank(), remote_device_after.rank())
            self.assertEqual(str(remote_device_before.device().type), "cpu")
            self.assertEqual(str(remote_device_after.device().type), "cuda")
        
        # 确保元数据也已经更改为 CUDA 设备
        metas = new_st_gpu.metadata().shards_metadata
        for meta in metas:
            self.assertEqual(str(meta.placement.device().type), "cuda")

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    # 定义测试方法，用于测试分片张量的转换功能
    def test_sharded_tensor_to_test(self):
        # 创建分片分布规格对象，指定按第0维分片，各分片的位置
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
        h, w = 10, 20
        # 创建一个全零的分片张量对象 st，按照指定规格和尺寸创建
        st = sharded_tensor.zeros(spec, h, w)
        # 测试在 CUDA 设备上转换分片张量，返回同一分片张量（无移动）
        st_self = st.to(dtype=st.dtype, device="cuda")
        self.assertTrue(st_self is st)

        # 测试数据类型转换为 float16
        st_16 = st.to(torch.float16)
        self.assertFalse(st_16 is st)
        self.assertEqual(st_16.dtype, torch.float16)

        # 测试设备转换为 CPU
        st_cpu = st.to(device=torch.device("cpu"))
        self.assertFalse(st_cpu is st)
        self.assertEqual(st_cpu.local_tensor().device.type, "cpu")

        # 测试设备转换为 CUDA
        st_cuda = st_cpu.to(device=torch.device("cuda"))
        self.assertEqual(st_cuda.local_tensor().device.type, "cuda")

        # 无关键字参数的设备转换为 CUDA
        st_cuda = st_cpu.to(torch.device("cuda"))
        self.assertEqual(st_cuda.local_tensor().device.type, "cuda")

        # 测试设备转换为 CPU 后再转回 CUDA
        st_cpu = st_cuda.to(torch.device("cpu"))
        self.assertEqual(st_cpu.local_tensor().device.type, "cpu")

        # 字符串形式的设备转换测试
        st_cpu = st_cuda.to("cpu")
        self.assertEqual(st_cpu.local_tensor().device.type, "cpu")
        st_cuda = st_cpu.to("cuda")
        self.assertEqual(st_cuda.local_tensor().device.type, "cuda")

        # 整数形式的设备转换测试
        st_cpu = st_cuda.to("cpu")
        self.assertEqual(st_cpu.local_tensor().device.type, "cpu")
        st_cuda = st_cpu.to(self.rank)
        self.assertEqual(st_cuda.local_tensor().device.type, "cuda")

        # 测试从张量对象转换
        cuda_tensor = torch.randn(3, 4, dtype=torch.float16, device="cuda")
        st_cuda = st.to(cuda_tensor)
        self.assertFalse(st_cuda is st)
        self.assertEqual(st_cuda.dtype, torch.float16)

        cuda_tensor = torch.randn(3, 4, dtype=torch.float16, device="cuda:2")
        st_cuda = st.to(cuda_tensor)
        self.assertEqual(st_cuda.dtype, torch.float16)

        # 测试同时转换数据类型和设备
        st_cpu_16 = st.to("cpu", torch.float16)
        self.assertEqual(st_cpu_16.dtype, torch.float16)
        self.assertEqual(st_cpu_16.local_tensor().device.type, "cpu")

        st_cuda_32 = st_cpu_16.to("cuda", torch.float32)
        self.assertEqual(st_cuda_32.dtype, torch.float32)
        self.assertEqual(st_cuda_32.local_tensor().device.type, "cuda")

        # 测试传递额外的进程组参数
        gloo_pg = dist.new_group(backend="gloo")
        st_gloo = st.to(device="cpu", process_group=gloo_pg)
        self.assertFalse(st_gloo is st)
        self.assertEqual(st_gloo.local_tensor().device.type, "cpu")
        self.assertEqual(st_gloo._process_group, gloo_pg)

    @with_comms
    # 装饰器，如果 GPU 数量少于 4，则跳过此测试
    @skip_if_lt_x_gpu(4)
    # 要求使用 NCCL 库
    @requires_nccl()
    # 定义测试方法，测试分片张量的设备分配情况
    def test_sharded_tensor_device(self):
        # 定义分片规格，指定在第 0 维分片的位置和设备
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 定义分片张量的高度和宽度
        h, w = 10, 20
        # 创建一个 CUDA 分片张量，保持本地分片不变
        st = sharded_tensor.zeros(spec, h, w)
        # 获取当前 CUDA 设备
        current_device = torch.device(torch.cuda.current_device())
        # 断言当前分片张量的设备与当前设备一致
        self.assertEqual(current_device, st.device)

        # 测试转换到 CPU 后，设备是否正确变更
        cpu_device = torch.device("cpu")
        # 将分片张量转换到 CPU 设备
        st_cpu = st.to(device=cpu_device)
        # 断言转换后的分片张量设备为 CPU
        self.assertEqual(st_cpu.device, cpu_device)
    # 定义一个测试方法，用于验证非均匀分片的情况
    def test_uneven_shards(self):
        # 初始化分布式进程组
        self.init_pg()

        # 定义分片规格，包含了四个分片的元数据信息
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],  # 分片偏移量
                    shard_sizes=[2, 4],    # 分片尺寸
                    placement="rank:0/cuda:0",  # 分片的放置信息
                ),
                ShardMetadata(
                    shard_offsets=[0, 4],
                    shard_sizes=[4, 2],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[2, 0],
                    shard_sizes=[4, 4],
                    placement="rank:2/cuda:2",
                ),
                ShardMetadata(
                    shard_offsets=[4, 4],
                    shard_sizes=[2, 2],
                    placement="rank:3/cuda:3",
                ),
            ]
        )

        # 使用分片规格创建一个空的分布式张量，尺寸为 6x6
        st = sharded_tensor.empty(spec, 6, 6)
        # 验证张量的尺寸是否为 (6, 6)
        self.assertEqual((6, 6), st.size())
        # 验证本地分片数量是否为 1
        self.assertEqual(1, len(st.local_shards()))

        # 定义一个验证方法，用于验证每个分片的尺寸是否符合预期
        def verify_size(rank, tensor_dims):
            if rank == 0:
                self.assertEqual((2, 4), tensor_dims)
            elif rank == 1:
                self.assertEqual((4, 2), tensor_dims)
            elif rank == 2:
                self.assertEqual((4, 4), tensor_dims)
            elif rank == 3:
                self.assertEqual((2, 2), tensor_dims)

        # 定义一个验证方法，用于验证每个分片的偏移量是否符合预期
        def verify_offsets(rank, offsets):
            if rank == 0:
                self.assertEqual((0, 0), offsets)
            elif rank == 1:
                self.assertEqual((0, 4), offsets)
            elif rank == 2:
                self.assertEqual((2, 0), offsets)
            elif rank == 3:
                self.assertEqual((4, 4), offsets)

        # 验证本地分片
        local_shard = st.local_shards()[0]
        # 验证本地分片的设备是否为当前进程的 CUDA 设备
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.tensor.device)
        # 验证本地分片的张量尺寸是否符合预期
        verify_size(self.rank, local_shard.tensor.size())

        # 验证本地分片的元数据信息
        verify_offsets(self.rank, local_shard.metadata.shard_offsets)
        verify_size(self.rank, local_shard.metadata.shard_sizes)
        # 验证本地分片的放置信息是否正确
        self.assertEqual(
            f"rank:{self.rank}/cuda:{self.rank}", str(local_shard.metadata.placement)
        )

        # 验证全局元数据信息
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        # 验证全局分片数量为 4
        self.assertEqual(4, len(shards_metadata))
        # 遍历每个全局分片的元数据信息，并进行验证
        for rank, shard_metadata in enumerate(shards_metadata):
            verify_offsets(rank, shard_metadata.shard_offsets)
            verify_size(rank, shard_metadata.shard_sizes)
            # 验证全局分片的放置信息是否正确
            self.assertEqual(f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement))

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    # 定义测试函数，用于验证部分世界大小的情况
    def test_partial_world_size(self):
        # 创建可枚举分片规范对象
        spec = EnumerableShardingSpec(
            [
                # 第一个分片的元数据
                ShardMetadata(
                    shard_offsets=[0, 0],  # 分片偏移量
                    shard_sizes=[5, 5],    # 分片大小
                    placement="rank:0/cuda:0",  # 分片放置信息
                ),
                # 第二个分片的元数据
                ShardMetadata(
                    shard_offsets=[5, 0],    # 分片偏移量
                    shard_sizes=[5, 5],      # 分片大小
                    placement="rank:1/cuda:1",  # 分片放置信息
                ),
            ]
        )

        # 使用给定规范和初始化的远程引用创建一个空的分片张量
        st = sharded_tensor.empty(spec, 10, 5, init_rrefs=True)
        # 验证分片张量的大小
        self.assertEqual((10, 5), st.size())
        # 如果当前进程的排名小于或等于1，则验证本地分片的存在
        if self.rank <= 1:
            self.assertEqual(1, len(st.local_shards()))
        else:
            self.assertEqual(0, len(st.local_shards()))

        # 如果当前进程的排名小于或等于1，则进一步验证本地分片的内容和元数据
        if self.rank <= 1:
            # 验证本地分片
            local_shard = st.local_shards()[0]
            self.assertEqual(
                torch.device(f"cuda:{self.rank}"), local_shard.tensor.device
            )
            self.assertEqual((5, 5), local_shard.tensor.size())

            # 验证本地分片的元数据
            self.assertEqual((self.rank * 5, 0), local_shard.metadata.shard_offsets)
            self.assertEqual((5, 5), local_shard.metadata.shard_sizes)
            self.assertEqual(
                f"rank:{self.rank}/cuda:{self.rank}",
                str(local_shard.metadata.placement),
            )

        # 验证全局元数据
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(2, len(shards_metadata))
        # 遍历每个排名和分片元数据，验证其偏移量、大小和放置信息
        for rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual((rank * 5, 0), shard_metadata.shard_offsets)
            self.assertEqual((5, 5), shard_metadata.shard_sizes)
            self.assertEqual(f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement))

        # 验证远程分片
        remote_shards = st.remote_shards()
        # 如果当前进程的排名小于或等于1，则只有一个远程分片，否则有两个远程分片
        if self.rank <= 1:
            self.assertEqual(1, len(remote_shards))
        else:
            self.assertEqual(2, len(remote_shards))

        # 遍历远程分片的所有排名和分片，验证其归属和张量大小
        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(1, len(shards))

            for remote_shard in shards:
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                shard = remote_shard.to_here()
                self.assertEqual((5, 5), shard.tensor.size())

    # 使用通信装饰器、GPU数量不小于4和需要NCCL库的装饰器进行标记
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    # 定义一个测试方法，用于测试创建新的进程组和分布式张量
    def test_new_group(self):
        # 创建一个分片规格对象，描述两个分片的元数据和放置位置
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:3/cuda:3",
                ),
            ]
        )

        # 使用指定的进程组创建一个新的分布式进程组对象
        pg = dist.new_group(ranks=[1, 2, 3])

        # 创建一个空的分布式张量对象，基于给定的分片规格和进程组
        st = sharded_tensor.empty(spec, 10, 5, process_group=pg, init_rrefs=True)
        self.assertEqual((10, 5), st.size())  # 断言张量的大小为 (10, 5)

        # 如果当前进程的排名是 1 或 3，则执行以下验证操作
        if self.rank == 1 or self.rank == 3:
            # 验证本地分片
            local_shard = st.local_shards()[0]
            self.assertEqual(
                torch.device(f"cuda:{self.rank}"), local_shard.tensor.device
            )  # 断言本地分片的设备是 cuda:self.rank
            self.assertEqual((5, 5), local_shard.tensor.size())  # 断言本地分片的大小为 (5, 5)

            # 验证本地分片的元数据
            self.assertEqual(
                (self.rank // 2 * 5, 0), local_shard.metadata.shard_offsets
            )  # 断言本地分片的偏移量
            self.assertEqual((5, 5), local_shard.metadata.shard_sizes)  # 断言本地分片的大小
            self.assertEqual(
                f"rank:{self.rank}/cuda:{self.rank}",
                str(local_shard.metadata.placement),
            )  # 断言本地分片的放置位置字符串表示

        # 验证全局元数据
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        self.assertEqual(2, len(shards_metadata))  # 断言分片元数据列表的长度为 2
        for rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual((rank * 5, 0), shard_metadata.shard_offsets)
            self.assertEqual((5, 5), shard_metadata.shard_sizes)
            self.assertEqual(
                f"rank:{rank * 2 + 1}/cuda:{rank * 2 + 1}",
                str(shard_metadata.placement),
            )  # 断言每个分片的放置位置字符串表示

        # 验证远程分片
        remote_shards = st.remote_shards()
        if self.rank == 1 or self.rank == 3:
            self.assertEqual(1, len(remote_shards))  # 断言远程分片的数量为 1
        else:
            self.assertEqual(2, len(remote_shards))  # 断言远程分片的数量为 2

        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(1, len(shards))  # 断言每个 RPC 排名的远程分片列表长度为 1

            for remote_shard in shards:
                self.assertEqual(rpc_rank, remote_shard.owner().id)  # 断言远程分片的所有者排名
                shard = remote_shard.to_here()
                self.assertEqual((5, 5), shard.tensor.size())  # 断言远程分片的大小
    # 定义一个测试方法，用于验证 EnumerableShardingSpec 的功能
    def test_with_rpc_names(self):
        # 创建一个包含四个 ShardMetadata 对象的 EnumerableShardingSpec 对象
        spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="worker0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[0, 5],
                    shard_sizes=[5, 5],
                    placement="worker1/cuda:1",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="worker2/cuda:2",
                ),
                ShardMetadata(
                    shard_offsets=[5, 5],
                    shard_sizes=[5, 5],
                    placement="worker3/cuda:3",
                ),
            ]
        )

        # 使用 spec 创建一个空的 sharded_tensor 对象，大小为 10x10，初始化远程引用
        st = sharded_tensor.empty(spec, 10, 10, init_rrefs=True)
        # 断言 sharded_tensor 的大小为 (10, 10)
        self.assertEqual((10, 10), st.size())
        # 断言本地分片的数量为 1
        self.assertEqual(1, len(st.local_shards()))

        # 验证本地分片
        local_shard = st.local_shards()[0]
        # 断言本地分片的设备为当前进程的 CUDA 设备
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.tensor.device)
        # 断言本地分片的大小为 (5, 5)
        self.assertEqual((5, 5), local_shard.tensor.size())

        # 验证本地分片的元数据
        self.assertEqual(
            (self.rank // 2 * 5, (self.rank % 2) * 5),
            local_shard.metadata.shard_offsets,
        )
        self.assertEqual((5, 5), local_shard.metadata.shard_sizes)
        self.assertEqual(
            f"worker{self.rank}/cuda:{self.rank}", str(local_shard.metadata.placement)
        )

        # 验证全局元数据
        st_metadata = st.metadata()
        shards_metadata = st_metadata.shards_metadata
        # 断言分片元数据的数量为 4
        self.assertEqual(4, len(shards_metadata))
        for rank, shard_metadata in enumerate(shards_metadata):
            # 验证每个分片的偏移量
            self.assertEqual(
                (rank // 2 * 5, (rank % 2) * 5), shard_metadata.shard_offsets
            )
            # 验证每个分片的大小为 (5, 5)
            self.assertEqual((5, 5), shard_metadata.shard_sizes)
            # 验证每个分片的放置位置字符串
            self.assertEqual(f"worker{rank}/cuda:{rank}", str(shard_metadata.placement))

        # 验证远程分片
        remote_shards = st.remote_shards()
        # 断言远程分片的数量为 3
        self.assertEqual(3, len(remote_shards))

        for rpc_rank, shards in remote_shards.items():
            # 每个远程 RPC 排序的远程分片数量为 1
            self.assertEqual(1, len(shards))
            for remote_shard in shards:
                # 验证远程分片的所有者 ID 与 RPC 排序一致
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                # 将远程分片拉取到当前节点，并验证其大小为 (5, 5)
                shard = remote_shard.to_here()
                self.assertEqual((5, 5), shard.tensor.size())
# 定义一个测试类，从 ShardedTensorTestBase 继承而来
class TestShardedTensorFromLocalTensor(ShardedTensorTestBase):

    # 从本地张量生成分片张量的方法
    def _generate_st_from_chunk_local_tensor(self, st_size, sharding_spec):
        # 使用分片规格构建张量元数据
        tensor_meta = sharding_spec.build_metadata(st_size, TensorProperties())
        
        # 获取默认的分布式组
        pg = dist.distributed_c10d._get_default_group()

        local_tensor = None  # 本地张量初始化为空
        local_shard_metadata = None  # 本地分片元数据初始化为空
        rank_to_metadata = {}  # 用于存储排名到元数据的映射字典

        # 遍历张量元数据中的每个分片元数据
        for shard_metadata in tensor_meta.shards_metadata:
            # 解析和验证远程设备信息，获取排名和设备
            rank, device = _parse_and_validate_remote_device(pg, shard_metadata.placement)
            rank_to_metadata[rank] = shard_metadata  # 将排名和元数据存入映射字典
            if rank == self.rank:
                # 如果排名与当前对象的排名相同，则创建并初始化本地张量
                local_tensor = torch.rand(shard_metadata.shard_sizes).cuda(device)
                local_shard_metadata = shard_metadata  # 设置本地分片元数据为当前分片元数据

        # TODO: 确定当某些排名没有分片时 API 应该如何行为
        # 参考 https://github.com/pytorch/pytorch/issues/73133
        assert local_tensor is not None  # 断言确保本地张量不为空

        # 使用本地张量初始化 ShardedTensor 对象
        st = ShardedTensor._init_from_local_tensor(
            local_tensor,
            sharding_spec,
            st_size,
            init_rrefs=True,
        )

        # 断言确保初始化后的 ShardedTensor 的尺寸与给定尺寸相同
        self.assertEqual(tuple(st_size), st.size())
        # 断言确保 ShardedTensor 的本地分片数量为 1
        self.assertEqual(1, len(st.local_shards()))

        # 验证本地分片
        local_shard = st.local_shards()[0]
        # 断言确保 ShardedTensor 的本地张量与初始生成的本地张量相同
        self.assertEqual(st.local_tensor(), local_tensor)
        # 断言确保本地分片的设备与当前对象的设备相同
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.tensor.device)

        # 验证本地分片的元数据
        self.assertEqual(
            local_shard_metadata.shard_offsets, local_shard.metadata.shard_offsets
        )
        self.assertEqual(
            local_shard_metadata.shard_sizes, local_shard.metadata.shard_sizes
        )
        self.assertEqual(local_shard_metadata.placement, local_shard.metadata.placement)

        # 验证全局元数据
        st_shards_metadata = st.metadata().shards_metadata
        # 断言确保 ShardedTensor 的分片元数据数量与世界大小相同
        self.assertEqual(self.world_size, len(st_shards_metadata))
        # 断言确保 ShardedTensor 的分片元数据与初始张量元数据相同
        self.assertEqual(tensor_meta.shards_metadata, st_shards_metadata)

        # 验证远程分片
        remote_shards = st.remote_shards()
        # 断言确保远程分片的数量与世界大小减一相同
        self.assertEqual(self.world_size - 1, len(remote_shards))
        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(1, len(shards))
            for remote_shard in shards:
                # 断言确保远程分片的所有者排名与当前远程分片的排名相同
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                # 如果远程分片不存在，to_here() 将抛出异常
                if tensor_meta.shards_metadata[rpc_rank]:
                    shard = remote_shard.to_here()
                    # 断言确保远程分片的张量尺寸与对应排名的元数据相同
                    self.assertEqual(
                        rank_to_metadata[rpc_rank].shard_sizes, shard.tensor.size()
                    )
    # 定义测试方法，用于从本地张量初始化分片张量
    def test_init_from_local_tensor(self):
        # 使用特定的分片规格列表生成测试用的分片规格
        chunk_specs = _chunk_sharding_specs_list_for_test([0, 1, 1, 0], seed=31)
        # 遍历每个分片规格
        for spec in chunk_specs:
            # 调用方法生成从本地张量到分片张量的转换，生成不同大小的分片
            self._generate_st_from_chunk_local_tensor([20, 10], spec)
            self._generate_st_from_chunk_local_tensor([21, 11], spec)
            self._generate_st_from_chunk_local_tensor([23, 16], spec)
            self._generate_st_from_chunk_local_tensor([44, 16, 8], spec)

    # 使用装饰器包装的测试方法，用于测试从本地张量初始化分片张量时的错误情况
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_from_local_tensor_errors(self):
        # 创建一个可枚举的分片规格对象，包含两个分片的详细信息
        enumerable_sharding_spec = EnumerableShardingSpec(
            [
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
            ]
        )
        # 定义张量的大小
        st_size = [24, 12]
        # 在指定的 GPU 上生成一个随机张量
        local_tensor = torch.rand(*st_size).cuda(self.rank)
        # 使用上下文管理器验证是否抛出值错误异常，检查张量是否覆盖整个张量
        with self.assertRaisesRegex(ValueError, "do not cover the entire tensor"):
            # 调用方法从本地张量初始化分片张量，期望抛出指定的异常信息
            ShardedTensor._init_from_local_tensor(
                local_tensor,
                enumerable_sharding_spec,
                st_size,
            )
        # 使用特定的分片规格生成另一个分片规格列表
        chunk_specs = _chunk_sharding_specs_list_for_test([0], seed=31)
        # 使用上下文管理器验证是否抛出值错误异常，检查非连续张量
        with self.assertRaisesRegex(
            ValueError, "local_tensor is not a contiguous Tensor."
        ):
            # 调用方法从本地张量的转置初始化分片张量，期望抛出指定的异常信息
            ShardedTensor._init_from_local_tensor(
                local_tensor.t(),
                chunk_specs[0],
                st_size,
            )
class TestShardedTensorFromLocalShards(ShardedTensorTestBase):
    # 定义一个测试类，测试从本地分片创建分片张量的功能，继承自ShardedTensorTestBase基类

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_local_shards(self):
        # 测试本地分片功能的方法

        # 计算本地分片的偏移量，根据当前进程的排名（rank）计算得到
        shard_offsets = [(self.rank // 2) * 5, (self.rank % 2) * 5]

        # 定义本地分片的元数据，包括分片偏移量、分片大小、以及张量的放置信息
        local_shard_metadata = ShardMetadata(
            shard_offsets=shard_offsets,
            shard_sizes=[5, 5],
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )

        # 在CUDA设备上创建一个5x5的随机张量
        local_tensor = torch.randn(5, 5, device=f"cuda:{self.rank}")

        # 使用本地张量和元数据创建本地分片对象
        local_shard = sharded_tensor.Shard(local_tensor, local_shard_metadata)

        # 根据给定的偏移量和排名创建本地分片对象
        local_shard_from_offsets = sharded_tensor.Shard.from_tensor_and_offsets(
            local_tensor, shard_offsets=shard_offsets, rank=self.rank
        )

        # 断言本地分片对象的元数据与从偏移量创建的本地分片对象的元数据相等
        self.assertEqual(local_shard.metadata, local_shard_from_offsets.metadata)

        # 创建一个不正确的本地分片元数据（分片大小不匹配），验证是否抛出值错误异常
        wrong_local_shard_metadata = ShardMetadata(
            shard_offsets=shard_offsets,
            shard_sizes=[6, 5],
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )
        with self.assertRaisesRegex(ValueError, "Shard tensor size does not match"):
            local_shard_from_wrong_meta = sharded_tensor.Shard(
                local_tensor,
                metadata=wrong_local_shard_metadata,
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    # 定义一个测试函数，用于测试从本地分片初始化分布式张量
    def test_init_from_local_shards(self):
        # 创建本地分片的元数据对象
        local_shard_metadata = ShardMetadata(
            # 设置分片偏移量，基于当前进程排名（rank）来确定
            shard_offsets=[(self.rank // 2) * 5, (self.rank % 2) * 5],
            # 设置每个分片的大小为5x5
            shard_sizes=[5, 5],
            # 设置本地分片的放置位置，使用当前进程的rank和CUDA设备
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )

        # 创建包含一个本地分片的列表
        local_shards = [
            # 使用随机生成的5x5张量，并与本地分片元数据相关联
            sharded_tensor.Shard(
                torch.randn(5, 5, device=f"cuda:{self.rank}"), local_shard_metadata
            )
        ]

        # 从本地分片初始化分布式张量
        st = sharded_tensor.init_from_local_shards(
            local_shards, [10, 10], init_rrefs=True
        )
        
        # 断言分布式张量的大小为(10, 10)
        self.assertEqual((10, 10), st.size())
        # 断言分布式张量的本地分片数量为1
        self.assertEqual(1, len(st.local_shards()))

        # 验证本地分片的张量属性
        local_shard = st.local_shards()[0]
        self.assertEqual(torch.device(f"cuda:{self.rank}"), local_shard.tensor.device)
        self.assertEqual((5, 5), local_shard.tensor.size())

        # 验证本地分片的元数据属性
        self.assertEqual(
            (self.rank // 2 * 5, (self.rank % 2) * 5),
            local_shard.metadata.shard_offsets,
        )
        self.assertEqual((5, 5), local_shard.metadata.shard_sizes)
        self.assertEqual(
            f"rank:{self.rank}/cuda:{self.rank}", str(local_shard.metadata.placement)
        )

        # 验证全局元数据的属性
        shards_metadata = st.metadata().shards_metadata
        self.assertEqual(4, len(shards_metadata))
        for rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual(
                (rank // 2 * 5, (rank % 2) * 5), shard_metadata.shard_offsets
            )
            self.assertEqual((5, 5), shard_metadata.shard_sizes)
            self.assertEqual(f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement))

        # 验证远程分片的属性
        remote_shards = st.remote_shards()
        self.assertEqual(3, len(remote_shards))

        for rpc_rank, shards in remote_shards.items():
            self.assertEqual(1, len(shards))
            for remote_shard in shards:
                self.assertEqual(rpc_rank, remote_shard.owner().id)
                shard = remote_shard.to_here()
                self.assertEqual((5, 5), shard.tensor.size())

    # 如果GPU数量小于4，则跳过该测试
    @skip_if_lt_x_gpu(4)
    # 定义一个测试方法，用于测试从本地分片和全局元数据初始化的基础分片张量
    def test_st_base_init_from_local_shards_and_global_metadata(self):
        # 设置世界大小为4
        world_size = 4
        # 初始化分片元数据列表和分片列表
        shards_metadata = []
        shards = []
        # 遍历每个rank（0到3）
        for rank in range(world_size):
            # 创建本地分片的元数据对象
            local_shard_metadata = ShardMetadata(
                shard_offsets=[(rank // 2) * 5, (rank % 2) * 5],
                shard_sizes=[5, 5],
                placement=f"rank:{rank}/cuda:{rank}",
            )
            # 将本地分片的元数据对象加入列表
            shards_metadata.append(local_shard_metadata)
            # 创建本地分片张量对象并加入分片列表
            shards.append(
                sharded_tensor.Shard(
                    torch.randn(5, 5, device=f"cuda:{rank}"), local_shard_metadata
                )
            )

        # 设置张量属性对象
        tensor_properties = TensorProperties(
            dtype=torch.get_default_dtype(),
            layout=torch.strided,
            requires_grad=False,
            memory_format=torch.contiguous_format,
            pin_memory=False,
        )

        # 设置分片张量元数据对象
        sharded_tensor_metadata = sharded_tensor.ShardedTensorMetadata(
            shards_metadata=shards_metadata,
            size=torch.Size([10, 10]),
            tensor_properties=tensor_properties,
        )

        # 使用本地分片和全局元数据初始化基础分片张量对象
        st_base = sharded_tensor.ShardedTensorBase._init_from_local_shards_and_global_metadata(
            shards, sharded_tensor_metadata=sharded_tensor_metadata
        )
        
        # 断言本地分片的数量为4
        self.assertEqual(4, len(st_base.local_shards()))

        # 验证第一个本地分片的设备
        local_shard = st_base.local_shards()[0]
        self.assertEqual(torch.device("cuda:0"), local_shard.tensor.device)
        self.assertEqual((5, 5), local_shard.tensor.size())

        # 验证第一个本地分片的元数据
        self.assertEqual(
            (0, 0),
            local_shard.metadata.shard_offsets,
        )
        self.assertEqual((5, 5), local_shard.metadata.shard_sizes)
        self.assertEqual("rank:0/cuda:0", str(local_shard.metadata.placement))

        # 验证全局元数据
        shards_metadata = st_base.metadata().shards_metadata
        self.assertEqual(4, len(shards_metadata))
        for rank, shard_metadata in enumerate(shards_metadata):
            self.assertEqual(
                (rank // 2 * 5, (rank % 2) * 5), shard_metadata.shard_offsets
            )
            self.assertEqual((5, 5), shard_metadata.shard_sizes)
            self.assertEqual(f"rank:{rank}/cuda:{rank}", str(shard_metadata.placement))
    # 定义一个测试方法，用于测试从本地分片创建新组的初始化
    def test_init_from_local_shards_new_group(self):
        # 创建一个新的分布式组，包括排名为1、2、3的进程
        new_pg = dist.new_group(ranks=[1, 2, 3])

        # 如果当前进程的排名不是0，则执行以下代码块
        if self.rank != 0:
            # 创建本地分片的元数据对象
            local_shard_metadata = ShardMetadata(
                # 设置分片的偏移量和大小
                shard_offsets=[5 * (self.rank - 1), 0],
                shard_sizes=[5, 5],
                # 设置本地分片的放置位置
                placement=f"rank:{self.rank}/cuda:{self.rank}",
            )
            # 创建本地分片对象列表
            local_shards = [
                sharded_tensor.Shard(
                    torch.randn(5, 5, device=f"cuda:{self.rank}"), local_shard_metadata
                )
            ]

            # 使用本地分片初始化一个分布式张量
            st = sharded_tensor.init_from_local_shards(
                local_shards, [15, 5], process_group=new_pg
            )

            # 验证本地分片
            local_shard = st.local_shards()[0]
            self.assertEqual(
                torch.device(f"cuda:{self.rank}"), local_shard.tensor.device
            )
            self.assertEqual((5, 5), local_shard.tensor.size())

            # 验证本地分片的元数据
            self.assertEqual(
                ((self.rank - 1) * 5, 0), local_shard.metadata.shard_offsets
            )
            self.assertEqual((5, 5), local_shard.metadata.shard_sizes)
            self.assertEqual(
                f"rank:{self.rank}/cuda:{self.rank}",
                str(local_shard.metadata.placement),
            )

            # 验证全局元数据
            st_metadata = st.metadata()
            shards_metadata = st_metadata.shards_metadata
            self.assertEqual(3, len(shards_metadata))
            for rank, shard_metadata in enumerate(shards_metadata):
                self.assertEqual((rank * 5, 0), shard_metadata.shard_offsets)
                self.assertEqual((5, 5), shard_metadata.shard_sizes)
                self.assertEqual(
                    f"rank:{rank + 1}/cuda:{rank + 1}", str(shard_metadata.placement)
                )

    # 使用通信装饰器，跳过小于4个GPU的情况，并要求使用NCCL
    # 测试函数，用于验证从本地分片初始化时处理无效的本地分片情况
    def test_init_from_local_shards_invalid_local_shards(self):
        # 创建本地分片的元数据对象
        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank // 2) * 5, (self.rank % 2) * 5],
            shard_sizes=[5, 5],
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )

        # 创建稀疏张量的索引和值
        indices = [[0, 1, 1], [2, 0, 2]]
        values = [3.2, 4.5, 5.8]
        # 使用稀疏张量的索引和值创建稀疏 COO 张量
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, (5, 5), device=f"cuda:{self.rank}"
        )

        # 创建空的本地分片列表，用于测试抛出异常
        empty_local_shards = []
        # 验证在所有排名上没有本地分片的情况下会抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, "have no local shards on all ranks"):
            st = sharded_tensor.init_from_local_shards(
                empty_local_shards, [10, 10], init_rrefs=True
            )

        # 创建错误布局的本地分片列表，用于测试抛出异常
        wrong_layout_shards = [
            sharded_tensor.Shard(sparse_tensor, local_shard_metadata)
        ]
        # 验证只支持 torch.strided 布局，其他布局会抛出 ValueError 异常
        with self.assertRaisesRegex(
            ValueError, "Only torch.strided layout is currently supported"
        ):
            st = sharded_tensor.init_from_local_shards(
                wrong_layout_shards, [10, 10], init_rrefs=True
            )

        # 创建错误内存格式的本地分片列表，用于测试抛出异常
        wrong_memory_format_shards = [
            sharded_tensor.Shard(
                torch.randn(5, 5, device=f"cuda:{self.rank}").t(), local_shard_metadata
            )
        ]
        # 验证只支持 torch.contiguous_format 内存格式，其他格式会抛出 ValueError 异常
        with self.assertRaisesRegex(
            ValueError,
            "Only torch.contiguous_format memory_format is currently supported",
        ):
            st = sharded_tensor.init_from_local_shards(
                wrong_memory_format_shards, [10, 10], init_rrefs=True
            )

        # 创建错误尺寸的本地分片列表，用于测试抛出异常
        with self.assertRaisesRegex(ValueError, "Shard tensor size does not match"):
            wrong_size_shards = [
                sharded_tensor.Shard(
                    torch.randn(2, 3, device=f"cuda:{self.rank}"), local_shard_metadata
                )
            ]

        # 创建错误设备的本地分片列表，用于测试抛出异常
        with self.assertRaisesRegex(
            ValueError, "Local shard tensor device does not match"
        ):
            wrong_device_shards = [
                sharded_tensor.Shard(torch.randn(5, 5), local_shard_metadata)
            ]
    # 定义一个测试方法，用于测试从本地碎片初始化 ShardedTensor 时的无效属性跨排问题
    def test_init_from_local_shards_invalid_property_cross_ranks(self):
        # 创建本地碎片的元数据对象，包括碎片偏移、大小和放置信息
        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank // 2) * 5, (self.rank % 2) * 5],
            shard_sizes=[5, 5],
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )
        # 如果当前进程的 rank 是 0，则整个张量的大小为 [10, 10]，否则为 [10, 5]
        tensor_overall_size = [10, 10] if self.rank == 0 else [10, 5]
        # 创建一个包含错误数据类型的碎片列表
        wrong_dtype_shards = [
            sharded_tensor.Shard(
                torch.ones(5, 5, device=f"cuda:{self.rank}"), local_shard_metadata
            )
        ]
        # 断言初始化 ShardedTensor 时引发 ValueError，提示全局大小属性不匹配于不同的 rank 之间
        with self.assertRaisesRegex(
            ValueError,
            "ShardedTensor global_size property does not match from different ranks!",
        ):
            st = sharded_tensor.init_from_local_shards(
                wrong_dtype_shards, tensor_overall_size, init_rrefs=True
            )

        # 如果当前进程的 rank 是 0，则张量的数据类型为 torch.int，否则为 torch.float32
        tensor_dtype = torch.int if self.rank == 0 else torch.float32
        # 创建一个包含错误数据类型的碎片列表
        wrong_dtype_shards = [
            sharded_tensor.Shard(
                torch.ones(5, 5, device=f"cuda:{self.rank}", dtype=tensor_dtype),
                local_shard_metadata,
            )
        ]
        # 断言初始化 ShardedTensor 时引发 ValueError，提示数据类型属性不匹配于不同的 rank 之间
        with self.assertRaisesRegex(
            ValueError,
            "ShardedTensor dtype property does not match from different ranks!",
        ):
            st = sharded_tensor.init_from_local_shards(
                wrong_dtype_shards, [10, 10], init_rrefs=True
            )

        # 如果当前进程的 rank 是 0，则张量需要梯度为 True，否则为 False
        tensor_requires_grad = True if self.rank == 0 else False
        # 创建一个包含错误梯度属性的碎片列表
        wrong_requires_grad_shards = [
            sharded_tensor.Shard(
                torch.randn(
                    5, 5, device=f"cuda:{self.rank}", requires_grad=tensor_requires_grad
                ),
                local_shard_metadata,
            )
        ]
        # 断言初始化 ShardedTensor 时引发 ValueError，提示梯度属性不匹配于不同的 rank 之间
        with self.assertRaisesRegex(
            ValueError,
            "ShardedTensor requires_grad property does not match from different ranks!",
        ):
            st = sharded_tensor.init_from_local_shards(
                wrong_requires_grad_shards, [10, 10], init_rrefs=True
            )

        # 更新本地碎片的元数据对象，改变放置信息为在 CPU 上
        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank // 2) * 5, (self.rank % 2) * 5],
            shard_sizes=[5, 5],
            placement=f"rank:{self.rank}/cpu",
        )
    def test_init_from_local_shards_invalid_pin_memory(self):
        # pin memory can only be on dense cpu
        # 定义本地分片的元数据对象
        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank // 2) * 5, (self.rank % 2) * 5],
            shard_sizes=[5, 5],
            placement=f"rank:{self.rank}/cpu",
        )
        # 创建错误的 pin_memory 设置的本地分片列表
        wrong_pin_memory_local_shards = [
            sharded_tensor.Shard(
                torch.randn(5, 5, pin_memory=True), local_shard_metadata
            ),
            sharded_tensor.Shard(
                torch.randn(5, 5, pin_memory=False), local_shard_metadata
            ),
        ]
        # 断言：本地分片的 pin_memory 属性需要一致
        with self.assertRaisesRegex(
            ValueError, "Local shards' tensor pin_memory property need to be the same"
        ):
            st = sharded_tensor.init_from_local_shards(
                wrong_pin_memory_local_shards, [10, 10], init_rrefs=True
            )

        # 根据当前进程的 rank 确定 tensor_pin_memory 的设置
        tensor_pin_memory = True if self.rank == 0 else False
        # 创建跨多个进程的错误 pin_memory 设置的分片列表
        wrong_pin_memory_shards_cross_ranks = [
            sharded_tensor.Shard(
                torch.randn(5, 5, pin_memory=tensor_pin_memory), local_shard_metadata
            )
        ]
        # 断言：跨多个进程的 ShardedTensor 的 pin_memory 属性需要一致
        with self.assertRaisesRegex(
            ValueError,
            "ShardedTensor pin_memory property does not match from different ranks!",
        ):
            st = sharded_tensor.init_from_local_shards(
                wrong_pin_memory_shards_cross_ranks, [10, 10], init_rrefs=True
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_from_local_shards_invalid_shards_overlap(self):
        # 根据当前进程的 rank 定义本地分片的大小
        local_shard_size = [5, 5] if self.rank != 0 else [6, 6]
        # 定义本地分片的元数据对象
        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank // 2) * 5, (self.rank % 2) * 5],
            shard_sizes=local_shard_size,
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )

        # 创建本地分片列表，每个分片都位于当前进程的指定 GPU 上
        local_shards = [
            sharded_tensor.Shard(
                torch.randn(local_shard_size, device=f"cuda:{self.rank}"),
                local_shard_metadata,
            )
        ]

        # 断言：初始化 ShardedTensor 时发现重叠的分片
        with self.assertRaisesRegex(ValueError, "overlap"):
            sharded_tensor.init_from_local_shards(
                local_shards, [10, 10], init_rrefs=True
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    # 定义一个测试方法，用于初始化分布式张量时处理无效的本地分片间隙情况
    def test_init_from_local_shards_invalid_shards_gaps(self):
        # 根据当前进程的排名确定本地分片的大小，若排名不为0，则分片大小为[5, 5]，否则为[4, 4]
        local_shard_size = [5, 5] if self.rank != 0 else [4, 4]
        # 创建本地分片的元数据对象，包括分片偏移、分片大小和部署信息
        local_shard_metadata = ShardMetadata(
            shard_offsets=[(self.rank // 2) * 5, (self.rank % 2) * 5],
            shard_sizes=local_shard_size,
            placement=f"rank:{self.rank}/cuda:{self.rank}",
        )

        # 创建本地分片列表，每个分片是一个Shard对象，包括张量数据和元数据
        local_shards = [
            sharded_tensor.Shard(
                torch.randn(local_shard_size, device=f"cuda:{self.rank}"),
                local_shard_metadata,
            )
        ]

        # 使用断言上下文管理器，期望引发值错误并包含指定的错误消息
        with self.assertRaisesRegex(ValueError, "does not match tensor volume"):
            # 调用函数init_from_local_shards，试图初始化分布式张量，若出错则抛出值错误
            sharded_tensor.init_from_local_shards(
                local_shards, [10, 10], init_rrefs=True
            )

    # 应用装饰器，确保本测试在使用通信时生效
    @with_comms
    # 跳过GPU少于4个的环境下的测试
    @skip_if_lt_x_gpu(4)
    # 确保环境支持NCCL通信
    @requires_nccl()
class TestShardedTensorCustomOps(ShardedTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_custom_op(self):
        # 定义一个自定义的分片操作，使用torch.asin函数作为实现
        @custom_sharded_op_impl(torch.asin)
        def my_sharded_asin(types, args, kwargs, process_group):
            return torch.asin(args[0].local_shards()[0].tensor)

        # 定义一个ChunkShardingSpec对象，指定维度和分布情况
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        # 创建一个随机初始化的分片张量
        st = sharded_tensor.rand(spec, 10, 10)
        # 对分片张量应用torch.asin函数
        res = torch.asin(st)
        # 断言操作结果与直接应用到本地分片的结果相同
        self.assertEqual(res, torch.asin(st.local_shards()[0].tensor))

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_custom_op_override(self):
        # 在指定GPU设备上创建一个随机初始化的张量
        t = torch.rand(10, 10).cuda(self.rank)

        # 导入自定义分片规范操作函数
        from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op

        # 定义一个自定义的分片规范操作，使用torch.nn.functional.linear函数作为实现
        @custom_sharding_spec_op(ChunkShardingSpec, torch.nn.functional.linear)
        def my_sharded_linear(types, args, kwargs, process_group):
            return t

        # 定义一个ChunkShardingSpec对象，指定维度和分布情况
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        # 在指定GPU设备上创建一个线性层
        m = torch.nn.Linear(32, 16).cuda(self.rank)
        # 分片线性层的权重参数
        shard_parameter(m, "weight", spec)

        # 对线性层应用随机张量
        result = m(torch.rand(15, 32).cuda(self.rank))
        # 断言操作结果与随机张量相同
        self.assertEqual(t, result)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_custom_op_errors(self):
        # 检查是否能够捕获到TypeError并包含指定错误信息的异常

        # 测试自定义分片操作的错误处理，缺少参数TypeError
        with self.assertRaisesRegex(TypeError, "expects signature"):
            @custom_sharded_op_impl(torch.nn.functional.linear)
            def my_op1(types, args, kwargs, process_group, random_param):
                pass

        # 测试自定义分片操作的错误处理，缺少参数TypeError
        with self.assertRaisesRegex(TypeError, "expects signature"):
            @custom_sharded_op_impl(torch.nn.functional.linear)
            def my_op2(types):
                pass


class TestShardMetadata(ShardedTensorTestBase):
    @with_comms
    @requires_nccl()
    def test_shard_metadata_init(self):
        # 获取默认进程组
        pg = dist.distributed_c10d._get_default_group()

        # 创建ShardMetadata对象，初始化分片元数据
        md = ShardMetadata([10], [0])
        # 确认placement为空
        self.assertIsNone(md.placement)
        # 使用异常检查验证远程设备为None的情况
        with self.assertRaisesRegex(ValueError, "remote device is None"):
            _parse_and_validate_remote_device(pg, md.placement)

        # 使用字符串placement创建ShardMetadata对象
        md = ShardMetadata([10], [0], "rank:0/cpu")
        # 确认placement已经转换为远程设备对象
        self.assertEqual(md.placement, _remote_device("rank:0/cpu"))
        # 解析和验证远程设备的rank和设备类型
        rank, device = _parse_and_validate_remote_device(pg, md.placement)
        self.assertEqual(0, rank)
        self.assertEqual(device, torch.device("cpu"))

    @with_comms
    @requires_nccl()
    # 定义一个测试函数，用于测试在没有指定放置信息的情况下创建分片对象
    def test_create_shard_with_no_placement(self):
        # 创建一个包含放置索引和分片长度的元数据对象
        md = ShardMetadata([0], [10])
        # 使用长度为 10 的零张量创建一个分片对象，并使用上面创建的元数据
        shard = Shard(torch.zeros(10), md)
        # 断言分片对象的元数据中的放置信息为空
        self.assertIsNone(shard.metadata.placement)
class TestShardedTensorSubGroupInit(TestCase):
    # 使用装饰器生成线程并初始化通信，设置世界大小为4
    @spawn_threads_and_init_comms(world_size=4)
    def test_sub_process_group_sharded_tensor_init(self):
        # 获取世界组中的成员
        world_pg = dist.GroupMember.WORLD
        # 获取当前进程的排名
        rank = dist.get_rank()

        # 定义子组大小
        sub_group_sz = 2
        # 根据当前进程的排名计算子组的排名列表
        sub_pg_ranks = [r for r in range(4) if r % sub_group_sz == rank % sub_group_sz]
        # 创建新的子组
        sub_pg = dist.new_group(
            sub_pg_ranks,
            backend=dist.get_backend(world_pg),
            use_local_synchronization=True,
        )
        # 在子组内进行屏障同步
        dist.barrier(sub_pg)

        # 从本地分片初始化分片张量
        ShardedTensor._init_from_local_shards(
            [
                Shard(
                    tensor=torch.tensor([1, 2, 3], device="meta"),
                    metadata=ShardMetadata(
                        shard_offsets=[3 * (rank // sub_group_sz)],
                        shard_sizes=[3],
                        placement=f"rank:{rank}/meta",
                    ),
                )
            ],
            6,
            process_group=sub_pg,
        )

    # 使用装饰器生成线程并初始化通信，设置世界大小为4
    @spawn_threads_and_init_comms(world_size=4)
    def test_sub_process_group_placement_validation(self):
        # 获取世界组中的成员
        world_pg = dist.GroupMember.WORLD
        self.assertIsNotNone(world_pg)
        # 获取当前进程的排名
        rank = dist.get_rank()

        # 定义子组大小
        sub_group_sz = 2
        # 根据当前进程的排名计算子组的排名列表
        sub_pg_ranks = [r for r in range(4) if r % sub_group_sz == rank % sub_group_sz]
        # 创建新的子组
        sub_pg = dist.new_group(
            sub_pg_ranks,
            backend=dist.get_backend(world_pg),
            use_local_synchronization=True,
        )
        # 在子组内进行屏障同步
        dist.barrier(sub_pg)

        # 遍历子组内的排名，解析和验证远程设备
        for r in sub_pg_ranks:
            _parse_and_validate_remote_device(
                sub_pg, _remote_device(f"rank:{r}/cuda:{r % sub_group_sz}")
            )


class TestCreateTensorNoProcessGroupMode(TestCase):
    # 测试从本地分片和全局元数据初始化分片张量
    def test_init_from_local_shards_and_global_metadata(self):
        # 定义分片张量的元数据
        st_metadata: ShardedTensorMetadata = ShardedTensorMetadata(
            shards_metadata=[
                ShardMetadata(
                    shard_offsets=[0, 0], shard_sizes=[2, 2], placement="rank:0/cpu"
                ),
                ShardMetadata(
                    shard_offsets=[2, 0], shard_sizes=[2, 2], placement="rank:1/cpu"
                ),
            ],
            size=torch.Size([4, 2]),
        )
        # 初始化本地分片列表
        st_local_shards: List[Shard] = []
        for shard_metadata in st_metadata.shards_metadata:
            st_local_shards.append(
                Shard(
                    tensor=torch.zeros(
                        shard_metadata.shard_sizes,
                        device=shard_metadata.placement.device(),
                    ),
                    metadata=shard_metadata,
                )
            )

        # 从本地分片和全局元数据初始化分片张量基类
        ShardedTensorBase._init_from_local_shards_and_global_metadata(
            local_shards=st_local_shards,
            sharded_tensor_metadata=st_metadata,
        )
    # 定义一个测试函数，用于测试非连续的本地分片
    def test_non_contiguous_local_shards(self):
        # 创建一个 ShardedTensorMetadata 对象，描述分片张量的元数据
        st_metadata: ShardedTensorMetadata = ShardedTensorMetadata(
            shards_metadata=[
                # 第一个分片的元数据，包括偏移量、大小和部署位置信息
                ShardMetadata(
                    shard_offsets=[0, 0], shard_sizes=[2, 2], placement="rank:0/cpu"
                ),
                # 第二个分片的元数据，包括偏移量、大小和部署位置信息
                ShardMetadata(
                    shard_offsets=[2, 0], shard_sizes=[2, 2], placement="rank:1/cpu"
                ),
            ],
            size=torch.Size([4, 2]),  # 描述整体张量的大小
        )
        # 初始化一个空列表，用于存储本地分片对象
        st_local_shards: List[Shard] = []
        # 创建一个 4x2 的张量作为数据源
        src = torch.randn(4, 2)
        # 遍历每个分片的元数据
        for shard_metadata in st_metadata.shards_metadata:
            # 获取当前分片的偏移量和大小
            offsets = shard_metadata.shard_offsets
            sizes = shard_metadata.shard_sizes
            # 根据偏移量和大小从数据源中切割出相应的子张量，并创建一个 Shard 对象
            st_local_shards.append(
                Shard(
                    tensor=src[
                        offsets[0] : offsets[0] + sizes[0],
                        offsets[1] : offsets[1] + sizes[1],
                    ],
                    metadata=shard_metadata,  # 将当前分片的元数据绑定到 Shard 对象
                )
            )

        # 调用 ShardedTensorBase 类的方法，使用本地分片和全局元数据来初始化分片张量
        ShardedTensorBase._init_from_local_shards_and_global_metadata(
            local_shards=st_local_shards,
            sharded_tensor_metadata=st_metadata,
        )
# 如果当前脚本作为主程序运行，执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```