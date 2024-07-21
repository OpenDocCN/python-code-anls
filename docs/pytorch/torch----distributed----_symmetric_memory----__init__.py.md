# `.\pytorch\torch\distributed\_symmetric_memory\__init__.py`

```
import socket  # 导入socket模块，用于获取主机名
import uuid  # 导入uuid模块，用于生成唯一标识符

from contextlib import contextmanager  # 导入contextmanager，支持上下文管理器
from functools import partial  # 导入partial，用于创建partial函数
from typing import Callable, Dict, Generator, List, Optional, Tuple  # 导入类型提示相关的模块

import torch  # 导入PyTorch库
import torch.distributed._functional_collectives as funcol  # 导入PyTorch分布式功能集合模块
import torch.distributed.distributed_c10d as c10d  # 导入PyTorch分布式C10d模块
from torch._C._distributed_c10d import _SymmetricMemory  # 导入对称内存相关的C模块

_group_name_to_store: Dict[str, c10d.Store] = {}  # 定义一个空字典，用于存储进程组名称到存储对象的映射


def enable_symm_mem_for_group(group_name: str) -> None:
    """
    Enables symmetric memory for a process group.

    Args:
        group_name (str): the name of the process group.
    """
    if group_name in _group_name_to_store:  # 检查进程组是否已经存在于存储字典中
        return

    group = c10d._resolve_process_group(group_name)  # 解析指定名称的进程组
    store = c10d.PrefixStore(
        "symmetric_memory",
        c10d._get_process_group_store(group),
    )  # 创建一个前缀存储对象，用于对称内存
    # 使用基于存储的广播来从进程中引导一个文件存储，并同时验证所有排名是否位于同一主机上
    hostname = socket.gethostname()  # 获取当前主机名
    if group.rank() == 0:  # 如果是排名为0的进程
        uid = str(uuid.uuid4())  # 生成一个UUID作为唯一标识符
        msg = f"{hostname}/{uid}"
        store.set("init", msg)  # 在存储中设置初始化消息
    else:  # 对于其他排名的进程
        msg = store.get("init").decode("utf-8")  # 从存储中获取并解码初始化消息
        tokens = msg.split("/")  # 拆分消息
        assert len(tokens) == 2, tokens  # 断言消息包含两部分
        rank_0_hostname, uid = tokens
        if hostname != rank_0_hostname:  # 如果当前主机名与排名0的主机名不匹配
            raise RuntimeError(
                "init_symmetric_memory_for_process_group() failed for "
                f'group "{group_name}". Rank 0 and rank {group.rank()} '
                f"are on different hosts ({rank_0_hostname} and {hostname})"
            )  # 抛出运行时错误，指示排名0和当前排名位于不同主机上
    store = torch._C._distributed_c10d.FileStore(f"/tmp/{uid}", group.size())  # 创建一个文件存储对象
    # TODO: check device connectiivity
    _group_name_to_store[group_name] = store  # 将存储对象添加到进程组名称到存储字典中
    _SymmetricMemory.set_group_info(
        group_name,
        group.rank(),
        group.size(),
        store,
    )  # 设置对称内存组信息


_is_test_mode: bool = False  # 定义一个全局变量，表示当前是否处于测试模式


@contextmanager
def _test_mode() -> Generator[None, None, None]:
    """
    Forces ``is_symm_mem_enabled_for_group()`` to return ``True`` and the ops
    defined in the ``symm_mem`` namespace to use fallback implementations.

    The context manager is not thread safe.
    """
    global _is_test_mode  # 声明使用全局的测试模式变量
    prev = _is_test_mode  # 保存当前的测试模式状态
    try:
        _is_test_mode = True  # 设置测试模式为True
        yield  # 进入上下文管理器
    finally:
        _is_test_mode = prev  # 恢复之前的测试模式状态


def is_symm_mem_enabled_for_group(group_name: str) -> bool:
    """
    Check if symmetric memory is enabled for a process group.

    Args:
        group_name (str): the name of the process group.
    """
    return _is_test_mode or group_name in _group_name_to_store  # 检查对称内存是否已经为指定的进程组启用


_group_name_to_workspace_tensor: Dict[str, Optional[torch.Tensor]] = {}  # 定义一个空字典，用于存储进程组名称到工作区张量的映射


def get_symm_mem_workspace(group_name: str, min_size: int) -> _SymmetricMemory:
    """
    Get the symmetric memory workspace associated with the process group. If
    ``min_size`` is greater than the workspace associated with ``group_name``,
    the workspace will be re-allocated and re-rendezvous'd.
    """
    # 根据组名从映射中获取对应的张量对象
    tensor = _group_name_to_workspace_tensor.get(group_name)
    # 计算张量的总元素数乘以每个元素的字节大小，如果张量为None则大小为0
    size = tensor.numel() * tensor.element_size() if tensor is not None else 0
    # 如果张量为None或者大小小于最小要求的大小
    if tensor is None or size < min_size:
        # 创建一个新的空白对称内存块，要求大小为给定的最大大小或者最小要求大小
        tensor = _SymmetricMemory.empty_strided_p2p(
            (max(size, min_size),),                # 指定张量的大小
            [1],                                   # 指定张量的步幅
            torch.uint8,                           # 指定张量的数据类型为torch.uint8
            torch.device(f"cuda:{torch.cuda.current_device()}"),  # 指定张量的计算设备
            group_name                             # 传递组名作为参数
        )
        # 将新创建的张量存储到组名到张量的映射中
        _group_name_to_workspace_tensor[group_name] = tensor
    # 调用SymmetricMemory类的rendezvous方法，返回处理后的张量
    return _SymmetricMemory.rendezvous(tensor)
# 定义一个全局变量 _backend_stream，用于存储 CUDA 流对象（可选类型为 torch.cuda.Stream）
_backend_stream: Optional[torch.cuda.Stream] = None


# 获取后端流对象的函数，如果未初始化则创建并返回该对象
def _get_backend_stream() -> torch.cuda.Stream:
    global _backend_stream
    if _backend_stream is None:
        _backend_stream = torch.cuda.Stream()
    return _backend_stream


# 执行微流水线计算和通信的函数
def _pipelined_all_gather_and_consume(
    shard: torch.Tensor,
    shard_consumer: Callable[[torch.Tensor, int], None],
    ag_out: torch.Tensor,
    group_name: str,
) -> None:
    """
    使用微流水线计算和通信执行以下逻辑：

        tensor = all_gather_tensor(shard, gather_dim=1, group=group)
        chunks = tensor.chunk(group.size())
        for src_rank, chunk in enumerate(chunks):
            shard_consumer(chunk, src_rank)

    注意：
    - 传递给 shard_consumer 的 shard 总是连续的。
    """
    # 计算需要的对称内存空间大小
    p2p_workspace_size_req = shard.numel() * shard.element_size()
    # 获取对称内存工作空间对象
    symm_mem = get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    # 获取组的大小和当前进程的排名
    group_size = symm_mem.world_size
    rank = symm_mem.rank

    # 获取后端流对象并等待当前 CUDA 流完成
    backend_stream = _get_backend_stream()
    backend_stream.wait_stream(torch.cuda.current_stream())
    # 获取本地 P2P 缓冲区
    local_p2p_buf = symm_mem.get_buffer(rank, shard.shape, shard.dtype)

    # 将 ag_out 切分成多个 chunk
    chunks = ag_out.chunk(group_size)

    # 在消费本地 shard 的同时，将其复制到本地 P2P 缓冲区中另一个流中
    shard_consumer(shard, rank)
    chunks[rank].copy_(shard)

    # 使用后端流进行操作
    with torch.cuda.stream(backend_stream):
        local_p2p_buf.copy_(shard)
        symm_mem.barrier(channel=0)
    torch.cuda.current_stream().wait_stream(backend_stream)

    # 此时，所有进程已将其本地 shard 复制到其本地 P2P 缓冲区中，
    # 现在每个进程可以复制和消费远程的 shard。
    for step in range(1, group_size):
        if step % 2 == 0:
            stream = torch.cuda.current_stream()
        else:
            stream = backend_stream
        remote_rank = (step + rank) % group_size
        remote_p2p_buf = symm_mem.get_buffer(remote_rank, shard.shape, shard.dtype)
        with torch.cuda.stream(stream):
            chunks[remote_rank].copy_(remote_p2p_buf)
            shard_consumer(chunks[remote_rank], remote_rank)

    # 使用后端流进行操作
    with torch.cuda.stream(backend_stream):
        symm_mem.barrier(channel=group_size % 2)
    torch.cuda.current_stream().wait_stream(backend_stream)


# 执行微流水线生产和 all-to-all 通信的函数
def _pipelined_produce_and_all2all(
    chunk_producer: Callable[[int, torch.Tensor], None],
    output: torch.Tensor,
    group_name: str,
) -> None:
    """
    使用微流水线计算和通信执行以下逻辑：

        chunks = [
            chunk_producer(dst_rank, chunks[dst_rank])
            for dst_rank in range(group_size):
        ]
        dist.all_to_all_single(output=output, input=torch.cat(chunks))
    """
    # 将输出张量分成多个 chunk
    out_chunks = output.chunk(c10d._get_group_size_by_name(group_name))
    # 计算所需的 P2P 工作空间大小
    p2p_workspace_size_req = out_chunks[0].numel() * out_chunks[0].element_size() * 2
    # 从指定的分布式存储区获取对称内存工作空间
    symm_mem = get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    # 获取当前分组的大小（进程数）
    group_size = symm_mem.world_size
    # 获取当前进程的排名
    rank = symm_mem.rank

    # 获取与当前后端相关的 CUDA 流
    backend_stream = _get_backend_stream()
    # 等待当前 CUDA 流上的操作完成
    backend_stream.wait_stream(torch.cuda.current_stream())

    def get_p2p_buf(rank: int, idx: int) -> torch.Tensor:
        # 断言确保索引 idx 只能为 0 或 1
        assert idx in (0, 1)
        # 如果 idx 为 0，则偏移量为 0；如果 idx 为 1，则偏移量为 out_chunks[0] 的元素数量
        offset = 0 if idx == 0 else out_chunks[0].numel()
        # 获取对称内存中的缓冲区
        return symm_mem.get_buffer(
            rank, out_chunks[0].shape, out_chunks[0].dtype, offset
        )

    # 准备两个本地的 P2P 缓冲区，以便远程进程可以从一个缓冲区中拉取步骤 [i] 的结果，
    # 而本地进程可以计算步骤 [i+1] 的结果并直接写入另一个 P2P 缓冲区。
    local_p2p_buf_0 = get_p2p_buf(rank, 0)
    local_p2p_buf_1 = get_p2p_buf(rank, 1)

    for step in range(1, group_size):
        # 计算远程进程的排名，以实现循环移位
        remote_rank = (rank - step) % group_size
        if step % 2 == 0:
            # 如果步骤为偶数，使用当前 CUDA 流
            stream = torch.cuda.current_stream()
            other_stream = backend_stream
            p2p_buf = local_p2p_buf_1
            remote_p2p_buf = get_p2p_buf(remote_rank, 1)
        else:
            # 如果步骤为奇数，使用后端 CUDA 流
            stream = backend_stream
            other_stream = torch.cuda.current_stream()
            p2p_buf = local_p2p_buf_0
            remote_p2p_buf = get_p2p_buf(remote_rank, 0)
        with torch.cuda.stream(stream):
            # 在指定的 CUDA 流上执行 chunk_producer 函数
            chunk_producer((rank + step) % group_size, p2p_buf)
            # 在对称内存中进行屏障同步，通道选择取决于步骤的奇偶性
            symm_mem.barrier(channel=step % 2)
            # 使另一个 CUDA 流在当前流上的屏障完成之后再执行 chunk_producer，
            # 以避免计算延迟屏障的影响。
            other_stream.wait_stream(stream)
            # 将远程进程的 P2P 缓冲区的内容复制到 out_chunks 数组中的相应位置
            out_chunks[remote_rank].copy_(remote_p2p_buf)

    # 最后一步：当前进程自身执行 chunk_producer 函数
    chunk_producer(rank, out_chunks[rank])
    # 等待当前 CUDA 流上的操作完成
    torch.cuda.current_stream().wait_stream(backend_stream)
lib = torch.library.Library("symm_mem", "DEF")  # noqa: TOR901
lib.define(
    "fused_all_gather_matmul(Tensor A, Tensor[] Bs, int gather_dim, str group_name) -> (Tensor, Tensor[])"
)
lib.define(
    "fused_matmul_reduce_scatter(Tensor A, Tensor B, str reduce_op, int scatter_dim, str group_name) -> Tensor"
)

@torch.library.impl(lib, "fused_all_gather_matmul", "Meta")
def _fused_all_gather_matmul_fallback(
    A_shard: torch.Tensor,
    Bs: List[torch.Tensor],
    gather_dim: int,
    group_name: str,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Fallback implementation of fused_all_gather_matmul using all_gather_into_tensor
    from _c10d_functional. Performs all-gather operation on A_shard and then
    computes matmul with each tensor in Bs.
    """
    group_size = c10d._get_group_size_by_name(group_name)
    A = torch.ops._c10d_functional.all_gather_into_tensor(
        A_shard.contiguous(), group_size, group_name
    )
    A = torch.ops._c10d_functional.wait_tensor(A)
    A = A.view(group_size, *A_shard.shape).movedim(gather_dim + 1, 1).flatten(0, 1)
    return A.movedim(0, gather_dim), [
        torch.matmul(A, B).movedim(0, gather_dim) for B in Bs
    ]

@torch.library.impl(lib, "fused_all_gather_matmul", "CUDA")
def _fused_all_gather_matmul(
    A_shard: torch.Tensor,
    Bs: List[torch.Tensor],
    gather_dim: int,
    group_name: str,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    CUDA implementation of fused_all_gather_matmul.
    """
    if _is_test_mode:
        return _fused_all_gather_matmul_fallback(A_shard, Bs, gather_dim, group_name)
    if A_shard.dim() < 2:
        raise ValueError("A_shard must be a matrix")
    for B in Bs:
        if B.dim() != 2:
            raise ValueError("B must be a matrix")
    if gather_dim < 0 or gather_dim >= A_shard.dim():
        raise ValueError("Invalid gather_dim")

    group = c10d._resolve_process_group(group_name)
    with torch.profiler.record_function("fused_all_gather_matmul"):
        # 使用 Torch Profiler 记录函数执行时间，标记为 "fused_all_gather_matmul"

        # 将 gather_dim 维度移动到最前面，并将张量展平为二维矩阵。
        # 展平后的张量不需要连续存储（为了计算效率），因为 _pipelined_all_gather_and_consume
        # 保证传递给 shard_consumer 的分片是连续的。
        x = A_shard.movedim(gather_dim, 0)
        leading_dims = [group.size()] + list(x.shape[:-1])
        x = x.flatten(0, -2)

        # 用于恢复上述变换的辅助函数
        def unflatten(t: torch.Tensor) -> torch.Tensor:
            return t.view(*leading_dims, -1).flatten(0, 1).movedim(0, gather_dim)

        # 创建一个与 x 形状相同的空张量 ag_out
        ag_out = x.new_empty(
            x.shape[0] * group.size(),
            x.shape[1],
        )
        
        # 创建一个列表 outputs，其中每个元素是一个与 x 形状相同的空张量
        outputs = [
            x.new_empty(
                x.shape[0] * group.size(),
                B.shape[1],
            )
            for B in Bs
        ]

        # 将 outputs 中的每个张量分块，以备后续输出
        output_shards = [output.chunk(group.size()) for output in outputs]

        # 定义分片消费函数，计算 A 的第一维度上的块状矩阵乘法
        def shard_consumer(shard: torch.Tensor, rank: int) -> None:
            for idx, B in enumerate(Bs):
                torch.mm(shard, B, out=output_shards[idx][rank])

        # 执行管道式全局聚集和消费操作
        _pipelined_all_gather_and_consume(
            x,
            shard_consumer,
            ag_out,
            group_name,
        )

        # 返回恢复后的 ag_out 和 outputs 列表中每个输出的恢复版本
        return unflatten(ag_out), [unflatten(output) for output in outputs]
# 重新排列张量 `t`，使得 `t.permute(perm)` 成为连续的
def make_contiguous_for_perm(
    t: torch.Tensor,
    perm: List[int],
) -> torch.Tensor:
    """
    Restride `t` such that `t.permute(perm)` is contiguous.
    """
    # 创建逆置换列表，用于将排列后的张量重新排列为连续的
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return t.permute(perm).contiguous().permute(inv_perm)


# 重新排列 `t` 的 `scatter_dim` 维度，以优化性能
def restride_A_shard_for_fused_all_gather_matmul(
    t: torch.Tensor,
    scatter_dim: int,
) -> torch.Tensor:
    """
    Restride the `A_shard` arg of `fused_all_gather_matmul` for optimal perf.
    See the doc for `fused_all_gather_matmul` for detail.
    """
    # 创建新的排列顺序，将 `scatter_dim` 维度移到最前面
    perm = list(range(len(t.shape)))
    perm.insert(0, perm.pop(scatter_dim))
    return make_contiguous_for_perm(t, perm)


# 使用微流水线计算和通信执行矩阵乘法和归约散步操作
@torch.library.impl(lib, "fused_matmul_reduce_scatter", "Meta")
def _fused_matmul_reduce_scatter_fallback(
    A: torch.Tensor,
    B: torch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        reduce_scatter_tensor(A @ B, reduce_op, scatter_dim, group_name)

    Optimal stride order for A - if A.movedim(scatter_dim, 0) is contiguous, no
    extra copy is required for input layout transformation. Otherwise A needs
    to be copied once.

    NOTE:
    - The K dim across ranks are currently accumulated with bf16 which results
      in accuracy loss.
    """
    if _is_test_mode:
        # 如果处于测试模式，则回退到非优化实现
        return _fused_matmul_reduce_scatter_fallback(
            A, B, reduce_op, scatter_dim, group_name
        )
    if A.dim() < 2:
        # 如果 A 的维度小于 2，则抛出异常
        raise ValueError("A_shard must be a matrix")
    if scatter_dim < 0 or scatter_dim >= A.dim():
        # 如果 scatter_dim 不在有效范围内，则抛出异常
        raise ValueError("Invalid gather_dim")
    if B.dim() != 2:
        # 如果 B 的维度不为 2，则抛出异常
        raise ValueError("B must be a matrix")
    if reduce_op == "sum":
        # 如果 reduce_op 是 sum，则使用求和函数进行归约
        reduce_fn = partial(torch.sum, dim=0)
    elif reduce_op == "avg":
        # 如果 reduce_op 是 avg，则使用平均值函数进行归约
        reduce_fn = partial(torch.mean, dim=0)
    else:
        # 如果 reduce_op 不是 sum 或 avg，则抛出异常
        raise ValueError("reduce_op must be sum or avg")

    # 解析进程组名称，获取相应的进程组对象
    group = c10d._resolve_process_group(group_name)
    # 计算输出张量的形状，保持 A 的维度，但在 scatter_dim 维度上除以进程组的大小
    out_shape = [*A.shape[:-1], B.shape[1]]
    out_shape[scatter_dim] //= group.size()
    with torch.profiler.record_function("fused_matmul_reduce_scatter"):
        # 使用 Torch Profiler 记录性能数据，该操作是融合的矩阵乘积和 reduce_scatter 操作

        # 将 scatter_dim 维度移到最前面，并将张量展平成一个二维矩阵
        x = A.movedim(scatter_dim, 0)
        
        # 计算前导维度，将 group 的大小作为第一个维度，并将 x 的形状前 N-1 维加入列表中
        leading_dims = [group.size()] + list(x.shape[:-1])
        leading_dims[1] //= group.size()  # 将第二个维度除以 group 的大小
        x = x.flatten(0, -2)  # 将 x 沿着第 0 到倒数第二个维度展平
        shards = x.chunk(group.size())  # 将 x 分成 group.size() 个块

        # 定义生成器函数，用于计算每个块的矩阵乘积
        def chunk_producer(rank: int, out: torch.Tensor) -> None:
            torch.matmul(shards[rank], B, out=out)

        # 创建一个与 x 形状相同的新空张量
        stacked_partials = x.new_empty(x.shape[0], B.shape[1])

        # 执行流水线化生成和 all2all 通信操作
        _pipelined_produce_and_all2all(
            chunk_producer,
            stacked_partials,
            group_name,
        )

        # 确保转置和减少操作在单个减少核中产生连续的结果
        return reduce_fn(
            stacked_partials.view(*leading_dims, -1)
            .movedim(1, scatter_dim + 1)
            .movedim(0, scatter_dim),
            dim=scatter_dim,
        )
# 重新排列张量 `t` 的维度，以优化 `fused_matmul_reduce_scatter` 函数中 `A_shard` 参数的性能
def restride_A_for_fused_matmul_reduce_scatter(
    t: torch.Tensor,
    gather_dim: int,
) -> torch.Tensor:
    """
    Restride the `A_shard` arg of `fused_matmul_reduce_scatter` for optimal
    perf. See the doc for `fused_matmul_reduce_scatter` for detail.
    """
    # 创建一个包含当前张量 `t` 维度顺序的列表
    perm = list(range(len(t.shape)))
    # 将指定的 `gather_dim` 维度移到列表的最前面
    perm.insert(0, perm.pop(gather_dim))
    # 根据新的维度顺序对张量 `t` 进行内存连续性操作
    return make_contiguous_for_perm(t, perm)
```