# `.\pytorch\torch\distributed\_tensor\examples\torchrec_sharding_example.py`

```
"""
# mypy: allow-untyped-defs
"""
"""
The following example demonstrates how to represent torchrec's embedding
sharding with the DTensor API.
"""

import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统相关模块
from functools import cached_property  # 导入缓存属性装饰器
from typing import List, TYPE_CHECKING  # 导入类型提示相关模块

import torch  # 导入PyTorch库
from torch.distributed._tensor import (
    DeviceMesh,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)  # 导入PyTorch分布式张量相关模块
from torch.distributed._tensor.debug.visualize_sharding import visualize_sharding  # 导入可视化分片相关模块
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    TensorProperties,
    TensorStorageMetadata,
)  # 导入分布式训练检查点元数据相关模块

if TYPE_CHECKING:
    from torch.distributed._tensor.placement_types import Placement  # 类型检查时导入的特定类型

def get_device_type():
    """
    根据系统环境返回设备类型，如果CUDA可用且有至少4个GPU则返回'cuda'，否则返回'cpu'
    """
    return (
        "cuda"
        if torch.cuda.is_available() and torch.cuda.device_count() >= 4
        else "cpu"
    )

aten = torch.ops.aten  # 获取PyTorch的ATen操作接口
supported_ops = [aten.view.default, aten._to_copy.default]  # 支持的操作列表

# this torch.Tensor subclass is a wrapper around all local shards associated
# with a single sharded embedding table.
class LocalShardsWrapper(torch.Tensor):
    """
    这个torch.Tensor子类是封装了与单个分片嵌入表相关联的所有本地分片。
    """
    local_shards: List[torch.Tensor]  # 本地分片列表
    storage_meta: TensorStorageMetadata  # 张量存储元数据

    @staticmethod
    def __new__(
        cls, local_shards: List[torch.Tensor], offsets: List[torch.Size]
    ) -> "LocalShardsWrapper":
        """
        创建一个新的LocalShardsWrapper实例。

        参数：
        - local_shards: 包含本地分片的列表
        - offsets: 分片偏移量列表

        返回：
        - 创建的LocalShardsWrapper实例
        """
        assert len(local_shards) > 0  # 确保本地分片列表不为空
        assert len(local_shards) == len(offsets)  # 确保本地分片和偏移量列表长度相同
        assert local_shards[0].ndim == 2  # 确保第一个本地分片是二维的

        # 计算拼接后的张量形状，第二个张量维度上的拼接
        cat_tensor_shape = list(local_shards[0].shape)
        if len(local_shards) > 1:  # 如果有多个分片，则进行列向分片
            for shard_size in [s.shape for s in local_shards[1:]]:
                cat_tensor_shape[1] += shard_size[1]

        # 根据第一个本地分片创建包装器属性和形状
        wrapper_properties = TensorProperties.create_from_tensor(local_shards[0])
        wrapper_shape = torch.Size(cat_tensor_shape)

        # 创建每个分片的存储元数据
        chunks_meta = [
            ChunkStorageMetadata(o, s.shape) for s, o in zip(local_shards, offsets)
        ]

        # 使用torch.Tensor._make_wrapper_subclass方法创建子类实例
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            wrapper_shape,
        )
        r.shards = local_shards  # 设置分片属性
        r.storage_meta = TensorStorageMetadata(
            properties=wrapper_properties,
            size=wrapper_shape,
            chunks=chunks_meta,
        )  # 设置张量存储元数据

        return r

    # necessary for ops dispatching from this subclass to its local shards
    @classmethod
    # 定义特殊方法 __torch_dispatch__，用于根据函数和类型分发操作
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # 如果 kwargs 为 None，则初始化为空字典
        kwargs = kwargs or {}

        # 如果 func 在支持的操作列表中
        if func in supported_ops:
            # 对于第一个参数的每个分片，调用 func 函数并传递其他参数和关键字参数
            res_shards_list = [
                func(shard, *args[1:], **kwargs) for shard in args[0].shards
            ]
            # 返回 LocalShardsWrapper 对象，包含结果分片列表和原始分片的偏移量
            return LocalShardsWrapper(res_shards_list, args[0].shard_offsets)
        else:
            # 如果 func 不在支持的操作列表中，则抛出 NotImplementedError 异常
            raise NotImplementedError(
                f"{func} is not supported for LocalShardsWrapper!"
            )

    # 定义属性 shards，返回本地分片列表
    @property
    def shards(self) -> List[torch.Tensor]:
        return self.local_shards

    # 定义属性 shards 的 setter 方法，设置本地分片列表
    @shards.setter
    def shards(self, local_shards: List[torch.Tensor]):
        self.local_shards = local_shards

    # 定义属性 shard_sizes，返回存储元数据的各分片大小列表
    @cached_property
    def shard_sizes(self) -> List[torch.Size]:
        return [chunk.sizes for chunk in self.storage_meta.chunks]

    # 定义属性 shard_offsets，返回存储元数据的各分片偏移量列表
    @cached_property
    def shard_offsets(self) -> List[torch.Size]:
        return [chunk.offsets for chunk in self.storage_meta.chunks]
def run_torchrec_row_wise_even_sharding_example(rank, world_size):
    # row-wise even sharding example:
    #   One table is evenly sharded by rows within the global ProcessGroup.
    #   In our example, the table's num_embedding is 8, and the embedding dim is 16
    #   The global ProcessGroup has 4 ranks, so each rank will have one 2 by 16 local
    #   shard.

    # device mesh is a representation of the worker ranks
    # create a 1-D device mesh that includes every rank
    device_type = get_device_type()
    device = torch.device(device_type)
    device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(world_size,))

    # manually create the embedding table's local shards
    num_embeddings = 8
    embedding_dim = 16
    emb_table_shape = torch.Size([num_embeddings, embedding_dim])
    # tensor shape
    local_shard_shape = torch.Size(
        [num_embeddings // world_size, embedding_dim]  # (local_rows, local_cols)
    )
    # tensor offset
    local_shard_offset = torch.Size((rank * 2, embedding_dim))
    # tensor
    local_tensor = torch.randn(local_shard_shape, device=device)
    # row-wise sharding: one shard per rank
    # create the local shards wrapper
    local_shards_wrapper = LocalShardsWrapper(
        local_shards=[local_tensor],
        offsets=[local_shard_offset],
    )

    ###########################################################################
    # example 1: transform local_shards into DTensor
    # usage in TorchRec:
    #   ShardedEmbeddingCollection stores model parallel params in
    #   _model_parallel_name_to_sharded_tensor which is initialized in
    #   _initialize_torch_state() and torch.Tensor params are transformed
    #   into ShardedTensor by ShardedTensor._init_from_local_shards().
    #
    #   This allows state_dict() to always return ShardedTensor objects.

    # this is the sharding placement we use in DTensor to represent row-wise sharding
    # row_wise_sharding_placements means that the global tensor is sharded by first dim
    # over the 1-d mesh.
    row_wise_sharding_placements: List[Placement] = [Shard(0)]

    # create a DTensor from the local shard
    dtensor = DTensor.from_local(
        local_shards_wrapper, device_mesh, row_wise_sharding_placements, run_check=False
    )

    # display the DTensor's sharding
    visualize_sharding(dtensor, header="Row-wise even sharding example in DTensor")

    ###########################################################################
    # example 2: transform DTensor into local_shards
    # usage in TorchRec:
    #   In ShardedEmbeddingCollection's load_state_dict pre hook
    #   _pre_load_state_dict_hook, if the source param is a ShardedTensor
    #   then we need to transform it into its local_shards.

    # transform DTensor into LocalShardsWrapper
    dtensor_local_shards = dtensor.to_local()
    assert isinstance(dtensor_local_shards, LocalShardsWrapper)
    shard_tensor = dtensor_local_shards.shards[0]
    # 使用 torch 的 equal 方法来比较两个张量 shard_tensor 和 local_tensor 是否相等
    assert torch.equal(shard_tensor, local_tensor)
    
    # 检查 dtensor_local_shards 对象的第一个分片大小是否等于 local_shard_shape
    assert dtensor_local_shards.shard_sizes[0] == local_shard_shape  # unwrap shape
    
    # 检查 dtensor_local_shards 对象的第一个分片偏移量是否等于 local_shard_offset
    assert dtensor_local_shards.shard_offsets[0] == local_shard_offset  # unwrap offset
def run_torchrec_row_wise_uneven_sharding_example(rank, world_size):
    # row-wise uneven sharding example:
    #   One table is unevenly sharded by rows within the global ProcessGroup.
    #   In our example, the table's num_embedding is 8, and the embedding dim is 16
    #   The global ProcessGroup has 4 ranks, and each rank will have the local shard
    #   of shape:
    #       rank 0: [1, 16]
    #       rank 1: [3, 16]
    #       rank 2: [1, 16]
    #       rank 3: [3, 16]

    # device mesh is a representation of the worker ranks
    # create a 1-D device mesh that includes every rank
    device_type = get_device_type()  # 获取设备类型
    device = torch.device(device_type)  # 根据设备类型创建设备对象
    device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(world_size,))  # 初始化设备网格

    # manually create the embedding table's local shards
    num_embeddings = 8  # 嵌入表的数量
    embedding_dim = 16  # 嵌入维度
    emb_table_shape = torch.Size([num_embeddings, embedding_dim])  # 嵌入表的形状
    # tensor shape
    local_shard_shape = (
        torch.Size([1, embedding_dim])  # 如果 rank % 2 == 0，则本地分片形状为 [1, 16]
        if rank % 2 == 0
        else torch.Size([3, embedding_dim])  # 如果 rank % 2 != 0，则本地分片形状为 [3, 16]
    )
    # tensor offset
    local_shard_offset = torch.Size((rank // 2 * 4 + rank % 2 * 1, embedding_dim))  # 计算本地分片的偏移量
    # tensor
    local_tensor = torch.randn(local_shard_shape, device=device)  # 在设备上创建随机张量
    # local shards
    # row-wise sharding: one shard per rank
    # create the local shards wrapper
    local_shards_wrapper = LocalShardsWrapper(
        local_shards=[local_tensor],  # 本地分片列表
        offsets=[local_shard_offset],  # 分片的偏移量列表
    )

    ###########################################################################
    # example 1: transform local_shards into DTensor
    # create the DTensorMetadata which torchrec should provide
    row_wise_sharding_placements: List[Placement] = [Shard(0)]  # 指定行分片的位置信息

    # note: for uneven sharding, we need to specify the shape and stride because
    # DTensor would assume even sharding and compute shape/stride based on the
    # assumption. Torchrec needs to pass in this information explicitely.
    # shape/stride are global tensor's shape and stride
    dtensor = DTensor.from_local(
        local_shards_wrapper,  # 从本地分片创建 DTensor
        device_mesh,  # 设备网格
        row_wise_sharding_placements,  # 行分片的位置信息列表
        run_check=False,  # 是否运行检查
        shape=emb_table_shape,  # 全局张量的形状，用于不均匀分片
        stride=(embedding_dim, 1),  # 全局张量的步长
    )
    # so far visualize_sharding() cannot print correctly for unevenly sharded DTensor
    # because it relies on offset computation which assumes even sharding.
    visualize_sharding(dtensor, header="Row-wise uneven sharding example in DTensor")  # 可视化不均匀分片的 DTensor
    # check the dtensor has the correct shape and stride on all ranks
    assert dtensor.shape == emb_table_shape  # 检查 DTensor 的形状是否正确
    assert dtensor.stride() == (embedding_dim, 1)  # 检查 DTensor 的步长是否正确

    ###########################################################################
    # example 2: transform DTensor into local_shards
    # note: DTensor.to_local() always returns a LocalShardsWrapper
    # 将分布式张量转换为本地分片对象
    dtensor_local_shards = dtensor.to_local()
    # 断言确保转换后的对象是 LocalShardsWrapper 类型
    assert isinstance(dtensor_local_shards, LocalShardsWrapper)
    # 获取第一个分片张量
    shard_tensor = dtensor_local_shards.shards[0]
    # 断言确保第一个分片张量与本地张量相等
    assert torch.equal(shard_tensor, local_tensor)
    # 断言确保第一个分片的大小与本地分片形状相同
    assert dtensor_local_shards.shard_sizes[0] == local_shard_shape  # unwrap shape
    # 断言确保第一个分片的偏移与本地分片偏移相同
    assert dtensor_local_shards.shard_offsets[0] == local_shard_offset  # unwrap offset
def run_torchrec_table_wise_sharding_example(rank, world_size):
    # table-wise example:
    #   each rank in the global ProcessGroup holds one different table.
    #   In our example, the table's num_embedding is 8, and the embedding dim is 16
    #   The global ProcessGroup has 4 ranks, so each rank will have one 8 by 16 complete
    #   table as its local shard.

    device_type = get_device_type()
    device = torch.device(device_type)
    # note: without initializing this mesh, the following local_tensor will be put on
    # device cuda:0.
    device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(world_size,))

    # manually create the embedding table's local shards
    num_embeddings = 8
    embedding_dim = 16
    emb_table_shape = torch.Size([num_embeddings, embedding_dim])

    # for table i, if the current rank holds the table, then the local shard is
    # a LocalShardsWrapper containing the tensor; otherwise the local shard is
    # an empty torch.Tensor
    table_to_shards = {}  # map {table_id: local shard of table_id}
    table_to_local_tensor = {}  # map {table_id: local tensor of table_id}
    # create 4 embedding tables and place them on different ranks
    # each rank will hold one complete table, and the dict will store
    # the corresponding local shard.
    for i in range(world_size):
        # tensor
        local_tensor = (
            torch.randn(*emb_table_shape, device=device)
            if rank == i
            else torch.empty(0, device=device)
        )
        table_to_local_tensor[i] = local_tensor
        # tensor shape
        local_shard_shape = local_tensor.shape
        # tensor offset
        local_shard_offset = torch.Size((0, 0))
        # wrap local shards into a wrapper
        local_shards_wrapper = (
            LocalShardsWrapper(
                local_shards=[local_tensor],
                offsets=[local_shard_offset],
            )
            if rank == i
            else local_tensor
        )
        table_to_shards[i] = local_shards_wrapper

    ###########################################################################
    # example 1: transform local_shards into DTensor
    table_to_dtensor = {}  # same purpose as _model_parallel_name_to_sharded_tensor
    table_wise_sharding_placements = [Replicate()]  # table-wise sharding
    # 遍历每个表格的 ID 和其对应的本地分片列表
    for table_id, local_shards in table_to_shards.items():
        # 创建一个子网格，仅包含我们放置表格的 rank
        # 注意，我们不能使用 ``init_device_mesh'' 来创建子网格
        # 因此选择使用 `DeviceMesh` API 直接创建 DeviceMesh
        device_submesh = DeviceMesh(
            device_type=device_type,
            mesh=torch.tensor(
                [table_id], dtype=torch.int64
            ),  # 表格 ``table_id`` 被放置在 rank ``table_id``
        )
        # 从当前表格的本地分片创建一个 DTensor
        # 注意：对于不均匀的分片，我们需要指定形状和步长，因为 DTensor 会假设均匀分片并根据这个假设计算形状/步长。
        # Torchrec 需要显式传入这些信息。
        dtensor = DTensor.from_local(
            local_shards,
            device_submesh,
            table_wise_sharding_placements,
            run_check=False,
            shape=emb_table_shape,  # 这对于不均匀的分片是必需的
            stride=(embedding_dim, 1),
        )
        # 将生成的 DTensor 存储到表格到 DTensor 的映射中
        table_to_dtensor[table_id] = dtensor

    # 打印每个表格的分片情况
    for table_id, dtensor in table_to_dtensor.items():
        visualize_sharding(
            dtensor,
            header=f"Table-wise sharding example in DTensor for Table {table_id}",
        )
        # 检查每个 DTensor 的形状和步长在所有 rank 上是否正确
        assert dtensor.shape == emb_table_shape
        assert dtensor.stride() == (embedding_dim, 1)

    ###########################################################################
    # 示例 2：将 DTensor 转换为 torch.Tensor
    for table_id, local_tensor in table_to_local_tensor.items():
        # 注意：DTensor.to_local() 总是返回一个空的 torch.Tensor，
        # 不管 DTensor._local_tensor 中传入了什么。
        dtensor_local_shards = table_to_dtensor[table_id].to_local()
        if rank == table_id:
            assert isinstance(dtensor_local_shards, LocalShardsWrapper)
            shard_tensor = dtensor_local_shards.shards[0]
            # 断言以验证张量是否正确展开
            assert torch.equal(shard_tensor, local_tensor)
            # 断言以验证分片大小是否正确展开
            assert dtensor_local_shards.shard_sizes[0] == emb_table_shape
            # 断言以验证分片偏移是否正确展开
            assert dtensor_local_shards.shard_offsets[0] == torch.Size(
                (0, 0)
            )
        else:
            # 断言以验证在非目标 rank 上，本地分片为空
            assert dtensor_local_shards.numel() == 0
# 定义一个函数，用于运行指定的示例代码，接受三个参数：rank（进程的排名），world_size（总进程数），example_name（示例代码的名称）
def run_example(rank, world_size, example_name):
    # 存储示例代码的字典，将示例名称映射到对应的函数
    name_to_example_code = {
        "row-wise-even": run_torchrec_row_wise_even_sharding_example,
        "row-wise-uneven": run_torchrec_row_wise_uneven_sharding_example,
        "table-wise": run_torchrec_table_wise_sharding_example,
    }
    # 如果指定的示例名称不存在于字典中，则打印错误信息并返回
    if example_name not in name_to_example_code:
        print(f"example for {example_name} does not exist!")
        return

    # 获取指定名称的示例函数
    example_func = name_to_example_code[example_name]

    # 设置随机种子
    torch.manual_seed(0)

    # 运行指定的示例函数，传入进程排名和总进程数作为参数
    example_func(rank, world_size)


if __name__ == "__main__":
    # 此脚本由 torchrun 启动，torchrun 会自动管理 ProcessGroup
    rank = int(os.environ["RANK"])  # 获取环境变量中的进程排名
    world_size = int(os.environ["WORLD_SIZE"])  # 获取环境变量中的总进程数
    assert world_size == 4  # 确保总进程数为 4，因为我们的示例使用了 4 个工作进程
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="torchrec sharding examples",  # 设置命令行参数的描述信息
        formatter_class=argparse.RawTextHelpFormatter,  # 使用原始文本格式显示帮助信息
    )
    example_prompt = (
        "choose one sharding example from below:\n"  # 提示用户从下面选择一个分片示例
        "\t1. row-wise-even;\n"
        "\t2. row-wise-uneven\n"
        "\t3. table-wise\n"
        "e.g. you want to try the row-wise even sharding example, please input 'row-wise-even'\n"
    )
    parser.add_argument("-e", "--example", help=example_prompt, required=True)  # 添加一个必选的示例参数
    args = parser.parse_args()  # 解析命令行参数
    run_example(rank, world_size, args.example)  # 运行示例函数，传入解析得到的示例名称作为参数
```