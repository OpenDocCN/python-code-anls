# `.\pytorch\torch\distributed\_tensor\_collective_utils.py`

```
# mypy: allow-untyped-defs
# 引入日志模块
import logging
# 引入数学模块
import math
# 引入数据类装饰器
from dataclasses import dataclass
# 引入LRU缓存装饰器
from functools import lru_cache
# 引入类型提示
from typing import List, Optional

# 引入PyTorch库
import torch
# 引入PyTorch分布式功能集合模块
import torch.distributed._functional_collectives as funcol
# 引入PyTorch张量位置类型模块
import torch.distributed._tensor.placement_types as placement_types
# 引入PyTorch设备网格资源管理模块
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
# 引入PyTorch分布式通信模块
from torch.distributed.distributed_c10d import (
    _get_group_size_by_name,
    broadcast,
    get_global_rank,
    get_rank,
    GroupMember,
    ProcessGroup,
    scatter,
    Work,
)

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 如果当前不是在torch deploy环境下运行
if not torch._running_with_deploy():

    @torch.library.register_fake("_dtensor::shard_dim_alltoall")
    def _shard_dim_alltoall_meta(input, gather_dim, shard_dim, group_name):
        # 根据组名获取组大小
        group_size = _get_group_size_by_name(group_name)
        # 创建与输入张量相同形状的空张量列表
        stacked_list = [torch.empty_like(input) for _ in range(group_size)]
        # 在指定维度上连接张量列表，并按照指定维度分块
        return torch.cat(stacked_list, dim=gather_dim).chunk(group_size, dim=shard_dim)

else:
    # 如果是在torch deploy环境下运行，发出警告
    import warnings

    warnings.warn(
        "PyTorch Distributed functional collectives do not work with torch::deploy."
    )

# 定义函数：在指定维度上进行分块分散操作
def shard_dim_alltoall(input, gather_dim, shard_dim, mesh, mesh_dim):
    # 如果设备网格的设备类型是CPU
    if mesh.device_type == "cpu":
        # Gloo不支持alltoall操作，因此使用allgather + chunk进行回退操作

        # TODO: 这里的日志记录过于频繁
        # 发出警告：CPU进程组暂不支持alltoall操作，使用allgather + chunk进行回退！
        logger.warning(
            "CPU process group does not support alltoall yet, falling back with allgather + chunk!"
        )
        # 执行allgather张量操作，并在指定维度上切分结果
        out = funcol.all_gather_tensor(input, gather_dim, (mesh, mesh_dim))
        # 如果输出是异步集体张量类型，等待操作完成
        if isinstance(out, funcol.AsyncCollectiveTensor):
            out = out.wait()
        # 按照设备网格的本地排名从切分的结果中选择张量块
        out = torch.chunk(out, mesh.size(mesh_dim), dim=shard_dim)[
            mesh.get_local_rank(mesh_dim)
        ]
        # 如果张量不是连续的，则进行连续化处理
        return out.contiguous() if not out.is_contiguous() else out

    # 解析组名以获取组名
    group_name = funcol._resolve_group_name((mesh, mesh_dim))
    # TODO: 启用分块分散操作的异步操作
    return torch.ops._dtensor.shard_dim_alltoall(
        input, gather_dim, shard_dim, group_name
    )


# 定义函数：在设备网格上进行张量分散操作
def mesh_scatter(
    output: torch.Tensor,
    scatter_list: List[torch.Tensor],
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
) -> Optional[Work]:
    """
    scatter a list of tensors to a device mesh dimension. We by default
    use the first rank of the mesh dimension as the source of truth, i.e
    for a 2d mesh [[0, 1], [2, 3]], if we scatter on mesh_dim = 1, we will
    scatter the tensor list on rank 0 to rank 0/1, and tensor list on rank
    2 to rank 2/3.
    """
    # 如果需要异步操作，则返回Work对象
    if async_op:
        return Work()

    # TODO: 补充详细注释
    return None
    """
    Args:
        output (torch.Tensor): 接收分散列表的张量。
        scatter_list (List[torch.Tensor]): 要分散的张量列表。
        mesh_dim (int, optional): 指定在哪个网格维度上进行分散，默认选择网格维度的第一个排名作为真值来源。

    Returns:
        A :class:`Work` object

    """
    # TODO: 理想情况下，我们应该使用元张量的方式
    # （注册一个元内核用于集体操作），这样可以避免通信。
    # 一旦完成，需要删除下面的检查。
    # 如果输出是元张量，则返回空
    if output.is_meta:
        return None
    
    # 获取网格维度对应的进程组
    dim_group = mesh.get_group(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)
    
    # 源应为全局排名
    src_for_dim = 0
    
    # 如果网格维度不是全局组，则获取该维度的全局排名
    if dim_group is not GroupMember.WORLD:
        src_for_dim = get_global_rank(dim_group, 0)
    
    # 如果当前进程是该维度的源进程，执行异步分散操作
    if src_for_dim == get_rank():
        fut = scatter(
            output,
            scatter_list=scatter_list,
            src=src_for_dim,
            group=dim_group,
            async_op=async_op,
        )
    else:
        # 如果当前进程不是源进程，执行分散操作但不传递scatter_list
        fut = scatter(
            output,
            scatter_list=None,
            src=src_for_dim,
            group=dim_group,
            async_op=async_op,
        )

    # 返回异步操作的future对象
    return fut
# 定义函数，将给定的张量广播到设备网格的指定维度
def mesh_broadcast(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
) -> Optional[Work]:
    """
    broadcast the tensor to a device mesh dimension. We by default
    use the first rank of the mesh dimension as the source of truth, i.e
    for a 2d mesh [[0, 1], [2, 3]], if we broadcast on mesh_dim = 1, we will
    broadcast the tensor on rank 0 to rank 0/1, and tensor on rank 2
    to rank 2/3.

    Args:
        tensor (torch.Tensor): tensor to broadcast.
        mesh (DeviceMesh): the device mesh topology.
        mesh_dim (int, optional): indicate which mesh dimension we want
            to scatter on, we by default choose the first rank on the
            mesh dimension as source of truth.
        async_op (bool, optional): whether the operation is asynchronous.

    Returns:
        Optional[Work]: A Work object representing the broadcast operation.
    """
    # TODO: Ideally we should use the meta tensor way
    # (to register a meta kernel for the collective op)
    # so that it would avoid the communication. Need to
    # remove the check below once that is done.
    # 检查张量是否为元张量，若是则返回空，表示不需要广播
    if tensor.is_meta:
        return None
    # 获取指定维度的通信组
    dim_group = mesh.get_group(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)
    # 确定源头的全局排名
    src_for_dim = 0
    if dim_group is not GroupMember.WORLD:
        src_for_dim = get_global_rank(dim_group, 0)

    # 调用广播函数，将张量广播到指定维度的所有设备
    return broadcast(tensor, src=src_for_dim, group=dim_group, async_op=async_op)


# 定义函数，对张量进行填充以增加指定维度的大小
def pad_tensor(tensor: torch.Tensor, pad_dim: int, pad_size: int) -> torch.Tensor:
    if pad_size == 0:
        return tensor
    # 构建填充参数列表
    pad = [0, 0] * (tensor.ndim - pad_dim)
    pad[-1] = pad_size
    # 使用PyTorch函数对张量进行填充
    return torch.nn.functional.pad(tensor, pad)


# 定义函数，对张量进行去除填充以减少指定维度的大小
def unpad_tensor(tensor: torch.Tensor, pad_dim: int, pad_size: int) -> torch.Tensor:
    if pad_size == 0:
        return tensor
    # 使用narrow函数对张量进行截取，去除指定维度上的填充
    return tensor.narrow(
        pad_dim,
        start=0,
        length=tensor.size(pad_dim) - pad_size,
    )


# 定义函数，将空张量填充到片段列表中
def fill_empty_tensor_to_shards(
    shards: List[torch.Tensor], shard_dim: int, num_empty_tensors: int
) -> List[torch.Tensor]:
    if num_empty_tensors == 0:
        return shards
    # 获取第一个片段的大小，并将指定维度的大小设置为0
    tensor_size = list(shards[0].size())
    tensor_size = [
        size if idx != shard_dim else 0 for idx, size in enumerate(tensor_size)
    ]
    # 创建全零张量，作为填充的空张量
    tensor = shards[0].new_zeros(tensor_size)
    # 将空张量添加到片段列表中指定次数
    for _ in range(num_empty_tensors):
        shards.append(tensor)
    return shards


# 定义函数，将指定张量规格转换为字节大小
def spec_to_bytes(spec: "placement_types.DTensorSpec") -> int:
    # 断言张量规格的元数据不为空
    assert spec.tensor_meta is not None, "spec should have tensor meta defined!"
    # 计算张量所占的总字节数，即数据类型的字节大小乘以张量形状的总元素个数
    return spec.tensor_meta.dtype.itemsize * math.prod(spec.shape)


@dataclass
class MeshTopoInfo:
    """
    Mesh information for collective cost estimation
    """

    mesh: DeviceMesh
    mesh_dim_devices: List[int]
    mesh_dim_bandwidth: List[float]
    mesh_dim_latency: List[float]

    @staticmethod
    @lru_cache(None)
    # ...
    # 根据给定的设备网格构建网格拓扑信息对象
    def build_from_mesh(mesh: DeviceMesh) -> "MeshTopoInfo":
        # 为主机内部和主机间通信模式生成网格拓扑信息
        # 注意，我们为简化起见做出了一些假设：
        # 1. 假设网格是同质的，且采用 GPU/NCCL 模型
        # 2. 假设 GPU 架构为 Ampere 或 Hopper
        # 3. 假设目前所有集合操作都采用环形算法
        num_devices_per_host = _mesh_resources.num_devices_per_host(mesh.device_type)
        # 基本带宽数值（节点内部），单位为 GB/s
        base_bw = 87.7
        mesh_dim_bandwidth = [base_bw] * mesh.ndim
        # 延迟以微秒表示（节点内部，NV-link）
        mesh_dim_latency = [0.6] * mesh.ndim
        mesh_dim_devices = [1] * mesh.ndim

        total_num_devices = 1
        for mesh_dim in reversed(range(mesh.ndim)):
            num_devices = mesh.size(mesh_dim)
            mesh_dim_devices[mesh_dim] = num_devices
            total_num_devices *= num_devices
            if total_num_devices > num_devices_per_host:
                # 魔数，用于主机间通信带宽/延迟因子
                # 此数值假设最新的 GPU 架构，即 Ampere 或 Hopper
                # TODO: 查看是否需要调整此值或提供用户指定带宽/延迟的方式
                mesh_dim_bandwidth[mesh_dim] *= 0.22
                # 设置为主机间以太网的延迟
                mesh_dim_latency[mesh_dim] = 2.7

        return MeshTopoInfo(
            mesh, mesh_dim_devices, mesh_dim_bandwidth, mesh_dim_latency
        )
# 计算在网格拓扑结构中执行全局收集操作的通信成本
def allgather_cost(bytes_gb: float, mesh_topo: MeshTopoInfo, mesh_dim: int) -> float:
    num_devices_on_mesh_dim = mesh_topo.mesh_dim_devices[mesh_dim]  # 获取指定维度上的设备数量
    mesh_dim_bandwidth = mesh_topo.mesh_dim_bandwidth[mesh_dim]  # 获取指定维度上的带宽
    num_hops = num_devices_on_mesh_dim - 1  # 计算在指定维度上的跳数
    # 基础延迟 + 通信延迟
    latency = 6.6 + num_hops * mesh_topo.mesh_dim_latency[mesh_dim]  # 计算延迟，单位为微秒
    bw = (bytes_gb * num_hops / num_devices_on_mesh_dim) / mesh_dim_bandwidth  # 计算带宽，单位为秒
    return latency + bw * 1e6  # 返回总成本，将带宽转换为微秒


# 计算在网格拓扑结构中执行全局归约操作的通信成本
def allreduce_cost(bytes_gb: float, mesh_topo: MeshTopoInfo, mesh_dim: int) -> float:
    num_devices_on_mesh_dim = mesh_topo.mesh_dim_devices[mesh_dim]  # 获取指定维度上的设备数量
    mesh_dim_bandwidth = mesh_topo.mesh_dim_bandwidth[mesh_dim]  # 获取指定维度上的带宽
    # 归约操作的通信量几乎是全收集和分散归约的两倍
    num_hops = 2 * num_devices_on_mesh_dim - 1  # 计算归约操作中的跳数

    latency = 6.6 + num_hops * mesh_topo.mesh_dim_latency[mesh_dim]  # 计算延迟，单位为微秒
    bw = (bytes_gb * num_hops / num_devices_on_mesh_dim) / mesh_dim_bandwidth  # 计算带宽，单位为秒
    return latency + bw * 1e6  # 返回总成本，将带宽转换为微秒


# 计算在网格拓扑结构中执行分散归约操作的通信成本
def reduce_scatter_cost(
    bytes_gb: float,
    mesh_topo: MeshTopoInfo,
    mesh_dim: int,
) -> float:
    num_devices_on_mesh_dim = mesh_topo.mesh_dim_devices[mesh_dim]  # 获取指定维度上的设备数量
    mesh_dim_bandwidth = mesh_topo.mesh_dim_bandwidth[mesh_dim]  # 获取指定维度上的带宽
    num_hops = num_devices_on_mesh_dim - 1  # 计算在指定维度上的跳数
    # 基础延迟 + 通信延迟
    latency = 6.6 + num_hops * mesh_topo.mesh_dim_latency[mesh_dim]  # 计算延迟，单位为微秒
    bw = (bytes_gb * num_hops / num_devices_on_mesh_dim) / mesh_dim_bandwidth  # 计算带宽，单位为秒
    return latency + bw * 1e6  # 返回总成本，将带宽转换为微秒


# 计算从当前到目标 DTensorSpec 的重新分布成本
def redistribute_cost(
    current_spec: "placement_types.DTensorSpec",
    target_spec: "placement_types.DTensorSpec",
) -> float:
    """
    This function returns the cost of redistribute from current to target DTensorSpec.

    NOTE:
    1. Only consider communication cost here, since computation costs for redistribute
       are quite trival (i.e. we only need to narrow or simple division)
    2. Only consider redistribute cost on same mesh, cross mesh communication cost is
       not quite needed for operator strategy estimation/selection.
    """
    if current_spec.mesh != target_spec.mesh:
        # 如果网格不同，返回无穷大的成本，表示不支持不同网格的通信
        # TODO: 看看是否要支持跨网格通信
        return float("inf")

    if current_spec.is_replicated():
        # 快速返回：
        # 如果当前规格已经完全复制，则通信成本为 0
        return 0.0

    mesh_topo = MeshTopoInfo.build_from_mesh(current_spec.mesh)  # 根据当前规格的网格构建拓扑信息
    cost = 0.0
    comm_bytes_gb = (
        spec_to_bytes(current_spec) / current_spec.num_shards / 1024 / 1024 / 1024
    )  # 计算通信字节数，单位为 GB

    # 考虑重新分布成本的转换:
    # 1. 全局收集 2. 全部互相传输
    # 3. 全局归约 4. 分散归约
    for i, (current, target) in enumerate(
        zip(current_spec.placements, target_spec.placements)
        # 遍历当前规格和目标规格的放置方式
    ):
        # 如果当前节点和目标节点相同，则跳过当前循环，继续下一个节点的计算
        if current == target:
            continue

        # 获取当前维度上的设备数量
        num_devices_on_mesh_dim = mesh_topo.mesh_dim_devices[i]

        # 如果当前节点是分片（shard），目标节点是复制（replicate）
        if current.is_shard() and target.is_replicate():
            # 使用 allgather 通信时，通信字节数扩展
            comm_bytes_gb *= num_devices_on_mesh_dim
            # 增加 allgather 通信代价到总代价中
            cost += allgather_cost(comm_bytes_gb, mesh_topo, i)

        # 如果当前节点和目标节点都是分片（shard）
        elif current.is_shard() and target.is_shard():
            # 由于我们尚未实现 alltoall 通信，因此增加一个惩罚以偏向 allgather
            cost += allgather_cost(comm_bytes_gb, mesh_topo, i) + 1.0

        # 如果当前节点是部分（partial），目标节点是复制（replicate）
        elif current.is_partial() and target.is_replicate():
            # 增加 allreduce 通信代价到总代价中
            cost += allreduce_cost(comm_bytes_gb, mesh_topo, i)

        # 如果当前节点是部分（partial），目标节点是分片（shard）
        elif current.is_partial() and target.is_shard():
            # 增加 reduce_scatter 通信代价到总代价中
            cost += reduce_scatter_cost(comm_bytes_gb, mesh_topo, i)
            # 在 reduce_scatter 之后，通信字节数减半以供进一步的集合操作使用
            comm_bytes_gb /= num_devices_on_mesh_dim

        # 如果当前节点是分片（shard），目标节点是部分（partial）
        elif current.is_shard() and target.is_partial():
            # 禁止分片到部分的通信，因为这种重新分配操作是无意义的
            return float("inf")

    # 返回计算得到的总通信代价
    return cost
```