# `.\pytorch\torch\_inductor\comm_analysis.py`

```
import functools  # 导入 functools 模块，用于支持函数式编程工具
import math  # 导入 math 模块，提供数学函数
from enum import IntEnum  # 导入 IntEnum 枚举类型，用于定义整数枚举


import sympy  # 导入 sympy 库，用于符号计算

import torch  # 导入 PyTorch 库
from . import ir  # 导入当前目录下的 ir 模块

from .utils import get_dtype_size, sympy_product  # 从当前目录下的 utils 模块导入函数和类
from .virtualized import V  # 从当前目录下的 virtualized 模块导入 V 对象


class NCCL_COLL(IntEnum):  # 定义 NCCL_COLL 枚举类型，表示集合通信操作类型
    ALL_REDUCE = 0  # 所有归约操作
    ALL_GATHER = 1  # 所有聚集操作
    REDUCE_SCATTER = 2  # 归约散射操作


class NVIDIA_GPU_TYPE(IntEnum):  # 定义 NVIDIA_GPU_TYPE 枚举类型，表示 NVIDIA GPU 类型
    VOLTA = 0  # Volta 架构
    AMPERE = 1  # Ampere 架构
    HOPPER = 2  # Hopper 架构


@functools.lru_cache  # 使用 functools.lru_cache 进行结果缓存
def get_gpu_type() -> NVIDIA_GPU_TYPE:  # 定义函数 get_gpu_type，返回 NVIDIA_GPU_TYPE 枚举类型
    gpu_info = torch.utils.collect_env.get_gpu_info(torch.utils.collect_env.run) or ""  # 获取 GPU 信息
    if "V100" in gpu_info:  # 如果 GPU 信息中包含 "V100"
        return NVIDIA_GPU_TYPE.VOLTA  # 返回 Volta 架构
    elif "A100" in gpu_info:  # 如果 GPU 信息中包含 "A100"
        return NVIDIA_GPU_TYPE.AMPERE  # 返回 Ampere 架构
    elif "H100" in gpu_info:  # 如果 GPU 信息中包含 "H100"
        return NVIDIA_GPU_TYPE.HOPPER  # 返回 Hopper 架构
    else:
        # 如果未识别到特定 GPU 类型，默认为 Ampere 架构
        return NVIDIA_GPU_TYPE.AMPERE  # 返回 Ampere 架构


def get_collective_type(node: ir.IRNode) -> NCCL_COLL:  # 定义函数 get_collective_type，返回 NCCL_COLL 枚举类型
    if not isinstance(node, ir._CollectiveKernel):  # 如果 node 不是 _CollectiveKernel 类型的实例
        raise ValueError(f"node is not a collective kernel: {node}")  # 抛出异常，指示 node 不是集合内核

    kernel_name = node.python_kernel_name  # 获取节点的 Python 内核名称
    assert kernel_name is not None  # 确保内核名称不为空
    if "all_reduce" in kernel_name:  # 如果内核名称包含 "all_reduce"
        return NCCL_COLL.ALL_REDUCE  # 返回 ALL_REDUCE 类型
    elif "all_gather" in kernel_name:  # 如果内核名称包含 "all_gather"
        return NCCL_COLL.ALL_GATHER  # 返回 ALL_GATHER 类型
    elif "reduce_scatter" in kernel_name:  # 如果内核名称包含 "reduce_scatter"
        return NCCL_COLL.REDUCE_SCATTER  # 返回 REDUCE_SCATTER 类型
    else:
        raise ValueError(f"Unsupported collective kernel: {kernel_name}")  # 抛出异常，指示不支持的集合内核


def get_collective_input_size_bytes(node: ir.IRNode) -> int:  # 定义函数 get_collective_input_size_bytes，返回整数
    sz_bytes = 0  # 初始化字节数为 0
    for inp in node.inputs:  # 遍历节点的输入
        numel = sympy_product(inp.layout.size)  # 计算输入张量的元素数量
        if isinstance(numel, sympy.Integer):  # 如果元素数量是 sympy.Integer 类型
            # For ease of testing
            numel = int(numel)  # 转换为整数
        else:
            numel = V.graph.sizevars.size_hint(numel, fallback=0)  # 获取大小提示

        sz_bytes += numel * get_dtype_size(inp.layout.dtype)  # 计算输入张量的总字节数

    return sz_bytes  # 返回总字节数


def get_collective_group_size(node: ir.IRNode) -> int:  # 定义函数 get_collective_group_size，返回整数
    if type(node) == ir._CollectiveKernel:  # 如果节点是 _CollectiveKernel 类型
        from torch.distributed.distributed_c10d import _get_group_size_by_name  # 导入函数

        return _get_group_size_by_name(node.constant_args[-1])  # 获取组大小
    else:
        raise TypeError(f"Unsupported collective type: {node}")  # 抛出异常，指示不支持的集合类型


####################################################################################################################
# The following code and constants are adapted from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc #
####################################################################################################################


class NCCL_HW(IntEnum):  # 定义 NCCL_HW 枚举类型，表示 NCCL 硬件类型
    NVLINK = 0  # NVLink 硬件
    PCI = 1  # PCI 硬件
    NET = 2  # 网络硬件


class NCCL_ALGO(IntEnum):  # 定义 NCCL_ALGO 枚举类型，表示 NCCL 算法类型
    TREE = 0  # 树形算法
    RING = 1  # 环形算法


class NCCL_PROTO(IntEnum):  # 定义 NCCL_PROTO 枚举类型，表示 NCCL 协议类型
    # The ordering and enum values here matches original in
    # https://github.com/NVIDIA/nccl/blob/0b083e52096c387bad7a5c5c65b26a9dca54de8c/src/include/devcomm.h#L28
    # For difference between these protocols, see https://github.com/NVIDIA/nccl/issues/281#issuecomment-571816990
    LL = 0  # 低延迟协议
    # LL128 = 1   # 低延迟 128 字节协议
    # SIMPLE = 2
# Latencies in us
# len(NCCL_ALGO) x len(NCCL_PROTO)
# NOTE: use array instead of tensor to prevent incompatibility with fake mode
baseLat = [
    # Tree
    [
        6.8,  # Latency for Tree algorithm with Low-Latency protocol
    ],
    # Ring
    [
        6.6,  # Latency for Ring algorithm with Low-Latency protocol
    ],
]

# Latencies in us
# len(NCCL_HW) x len(NCCL_ALGO) x len(NCCL_PROTO)
hwLat = [
    # NVLINK
    [
        [0.6],  # Latency for NVLINK with Tree algorithm and Low-Latency protocol
        [0.6],  # Latency for NVLINK with Ring algorithm and Low-Latency protocol
    ],
    # PCI
    [
        [1.0],  # Latency for PCI with Tree algorithm and Low-Latency protocol
        [1.0],  # Latency for PCI with Ring algorithm and Low-Latency protocol
    ],
    # NET
    [
        [5.0],  # Latency for NET with Tree algorithm and Low-Latency protocol
        [2.7],  # Latency for NET with Ring algorithm and Low-Latency protocol
    ],
]

# LL128 max BW per channel
llMaxBws = [
    # Volta-N1/Intel-N2/Intel-N4
    [
        39.0,   # Max bandwidth for LL128 per channel on Volta-N1/Intel-N2/Intel-N4
        39.0,   # Max bandwidth for LL128 per channel on Volta-N1/Intel-N2/Intel-N4
        20.4,   # Max bandwidth for LL128 per channel on Volta-N1/Intel-N2/Intel-N4
    ],
    # Ampere-N1/AMD-N2/AMD-N4
    [
        87.7,   # Max bandwidth for LL128 per channel on Ampere-N1/AMD-N2/AMD-N4
        22.5,   # Average of max bandwidth for LL128 per channel on Ring and Tree algorithms for Ampere-N1/AMD-N2/AMD-N4
        19.0,   # Max bandwidth for LL128 per channel on Ampere-N1/AMD-N2/AMD-N4
    ],
    # Hopper-N1/AMD-N2/AMD-N4
    [
        87.7,   # Max bandwidth for LL128 per channel on Hopper-N1/AMD-N2/AMD-N4
        22.5,   # Average of max bandwidth for LL128 per channel on Ring and Tree algorithms for Hopper-N1/AMD-N2/AMD-N4
        19.0,   # Max bandwidth for LL128 per channel on Hopper-N1/AMD-N2/AMD-N4
    ],
]

def estimate_nccl_collective_runtime(node: ir.IRNode) -> float:
    """
    Returns estimated NCCL collective runtime in nanoseconds (ns).

    The following heuristics are copied from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc.
    We aim to estimate the runtime as accurately as possible.

    Assumptions:
    - only ring algorithm (NCCL_ALGO_RING) is used
    - only Low-Latency protocol (NCCL_PROTO_LL) is used, i.e. Simple or LL128 is not used
    - 8 gpus per node  # TODO: Need to find a way to get accurate "gpus per node" and "# nodes" info.
    - collective is one of: allreduce, reducescatter, allgather
    """
    tensor_storage_size_bytes = get_collective_input_size_bytes(node)
    # Convert bytes to GB
    tensor_storage_size_GB = tensor_storage_size_bytes / 1024 / 1024 / 1024

    # Currently assumes each node has 8 gpus. And when >1 node is used, assumes each node uses all 8 gpus.
    # TODO: Need to find a way to get accurate "gpus per node" and "# nodes" info.
    num_gpus_per_node = 8
    group_size = get_collective_group_size(node)
    nNodes = math.ceil(group_size / num_gpus_per_node)
    nRanks = group_size  # this is total # of gpus globally that participate in this collective op

    if nRanks <= 1:
        return 0

    # Assumes ring algorithm
    nccl_algo = NCCL_ALGO.RING
    nccl_proto = NCCL_PROTO.LL
    coll = get_collective_type(node)

    # =============== bandwidth computation ===============
    # First compute bandwidth in GB/s; then at the end, convert it to GB/ns

    bwIntra = torch._inductor.config.intra_node_bw
    bwInter = torch._inductor.config.inter_node_bw

    compCapIndex = get_gpu_type()
    index2 = nNodes - 1 if nNodes <= 2 else 2
    # LL: for single node, we look at GPU type; for multi-node, we look at CPU type
    index1 = compCapIndex if nNodes == 1 else 0
    llMaxBw = llMaxBws[index1][index2]

    # NOTE: each step of ring algorithm is synchronized,
    # and is bottlenecked by the slowest link which is the inter-node interconnect.
    # hence when nNodes >= 2, bw is inter-node bandwidth.
    # NOTE: the original code in https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc
    # have this as `if nNodes <= 2` which seems wrong. Corrected it here.
    # 根据条件选择内部带宽或者跨节点带宽
    bw = bwIntra if nNodes == 1 else bwInter
    nChannels = 2  # 假设通道数为2
    # 计算总线带宽
    busBw = nChannels * bw

    # Various model refinements
    # 根据不同的条件调整总线带宽
    busBw = min(
        llMaxBw,
        busBw
        * (1.0 / 4.0 if (nNodes > 1 or coll == NCCL_COLL.ALL_REDUCE) else 1.0 / 3.0),
    )

    if coll == NCCL_COLL.ALL_REDUCE:
        # 计算所有约简操作的步骤数
        nsteps = 2 * (nRanks - 1)
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER):
        # 根据不同的集合通信操作计算步骤数
        nsteps = nRanks - 1

    # Convert bus BW to algorithm BW (tensor bytes / algoBW = actual execution time)
    # 计算带宽比例因子
    ratio = (1.0 * nRanks) / nsteps  # type: ignore[possibly-undefined]
    bandwidth = busBw * ratio
    # Convert GB/s to GB/ns
    # 将带宽从GB/s转换为GB/ns
    bandwidth_GB_per_ns = bandwidth / 1e9

    # =============== latency computation ===============
    intraHw = NCCL_HW.NVLINK

    if coll == NCCL_COLL.ALL_REDUCE:
        if nNodes > 1:
            # 计算跨节点约简操作的步骤数
            nInterSteps = 2 * nNodes
        else:
            nInterSteps = 0
    elif coll in (NCCL_COLL.REDUCE_SCATTER, NCCL_COLL.ALL_GATHER):
        # 根据不同的集合通信操作计算跨节点步骤数
        nInterSteps = nNodes - 1

    # First compute latency in us; then at the end, convert it to ns
    # 计算总延迟，单位为微秒，最后转换为纳秒
    latency = baseLat[nccl_algo][nccl_proto]
    intraLat = hwLat[intraHw][nccl_algo][nccl_proto]
    interLat = hwLat[NCCL_HW.NET][nccl_algo][nccl_proto]

    # Inter-node rings still have to launch nsteps * net overhead.
    # 如果有多个节点，计算网络开销
    netOverhead = 0.0
    if nNodes > 1:
        netOverhead = 1.0  # getNetOverhead(comm);
    intraLat = max(intraLat, netOverhead)
    # 计算总延迟，包括内部和跨节点延迟
    latency += (nsteps - nInterSteps) * intraLat + nInterSteps * interLat  # type: ignore[possibly-undefined]
    # Convert us to ns
    # 将延迟从微秒转换为纳秒
    latency_ns = latency * 1e3

    # =============== final result ===============
    # 计算传输时间
    transport_ns = tensor_storage_size_GB / bandwidth_GB_per_ns
    return transport_ns + latency_ns
# 上述代码和常量来自 https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc 的适配版本
################################################################################################################
```