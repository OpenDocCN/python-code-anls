# `.\pytorch\torch\distributed\_composable\fsdp\_fsdp_collectives.py`

```py
# 从 typing 模块中导入需要的类型声明
from typing import List, NamedTuple, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
import torch._dynamo.compiled_autograd as ca
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.distributed_c10d import ReduceOp

# 导入自定义的模块和函数
from ._fsdp_common import (
    _get_dim0_padded_size,
    _raise_assert_with_print,
    _to_dtype_if_needed,
)
from ._fsdp_param import FSDPParam

# 定义一个命名元组用于存储 all-gather 操作的结果
class AllGatherResult(NamedTuple):
    all_gather_output: torch.Tensor
    all_gather_event: Optional[torch.cuda.Event]
    all_gather_work: Optional[dist.distributed_c10d.Work]
    # 每个参数的 all-gather 输入的数据类型列表
    param_all_gather_input_dtypes: List[List[torch.dtype]]
    # 每个参数的 all-gather 输入的元素数量列表
    param_all_gather_input_numels: List[List[int]]
    # param_all_gather_input_numels 的扁平化版本，用于避免重新计算时的 CPU 开销
    all_gather_input_split_sizes: List[int]

# 创建一个 torch.library.Library 实例，命名为 lib，表示 FSDP 库中的一个片段
lib = torch.library.Library("fsdp", "FRAGMENT")  # noqa: TOR901

# 定义一个名为 all_gather_copy_in 的函数，用于元编程，实现数据的 all-gather 操作
lib.define(
    """
    all_gather_copy_in(
        Tensor[] all_gather_inputs,
        SymInt[] inp_split_sizes,
        SymInt all_gather_input_numel,
        SymInt world_size,
        SymInt rank,
        ScalarType dtype,
        Device device
    ) -> (Tensor, Tensor)
    """
)

# 使用 torch.library.impl 装饰器将 all_gather_copy_in 函数实现为元编程的 Meta 版本
@torch.library.impl(lib, "all_gather_copy_in", "Meta")
def all_gather_copy_in_meta(
    all_gather_inputs: List[torch.Tensor],
    inp_split_sizes: List[int],
    all_gather_input_numel: int,
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 创建一个空的 Tensor 用于存储 all-gather 操作的结果，数据类型为 dtype，设备为 "meta"
    all_gather_output = torch.empty(
        (all_gather_input_numel * world_size,), dtype=dtype, device="meta"
    )
    # 从 all_gather_output 中选择当前进程的部分数据作为输入
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    # 返回 all_gather 操作的输入和输出 Tensor
    return all_gather_input, all_gather_output

# 使用 torch.library.impl 装饰器分别实现 all_gather_copy_in 函数的 CUDA 和 CPU 版本
@torch.library.impl(lib, "all_gather_copy_in", "CUDA")
@torch.library.impl(lib, "all_gather_copy_in", "CPU")
def all_gather_copy_in_cuda(
    all_gather_inputs: List[torch.Tensor],
    inp_split_sizes: List[int],
    all_gather_input_numel: int,
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 创建一个空的 Tensor 用于存储 all-gather 操作的结果，数据类型为 dtype，设备为指定的 device
    all_gather_output = torch.empty(
        (all_gather_input_numel * world_size,), dtype=dtype, device=device
    )
    # 从 all_gather_output 中选择当前进程的部分数据作为输入
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    # 将 all_gather_input 按照 inp_split_sizes 切分，并将 all_gather_inputs 复制到相应的位置
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    with torch.no_grad():
        torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs)
    # 返回 all_gather 操作的输入和输出 Tensor
    return all_gather_input, all_gather_output

# 定义一个名为 split_with_sizes_copy 的函数，用于分割 Tensor，并复制到指定位置
lib.define(
    "split_with_sizes_copy(Tensor all_gather_output, SymInt[] all_gather_input_split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()"
)

# 使用 torch.library.impl 装饰器将 split_with_sizes_copy 函数实现为元编程的 Meta 版本
@torch.library.impl(lib, "split_with_sizes_copy", "Meta")
# 将函数声明为 torch 库的实现，绑定到 "split_with_sizes_copy" 函数，在 CUDA 设备上执行
@torch.library.impl(lib, "split_with_sizes_copy", "CUDA")
# 将函数声明为 torch 库的实现，绑定到 "split_with_sizes_copy" 函数，在 CPU 上执行
@torch.library.impl(lib, "split_with_sizes_copy", "CPU")
# 定义一个函数，用于按指定大小分割张量，并将结果存储在给定的输出列表中
def split_with_sizes_copy(
    all_gather_output: torch.Tensor,  # 输入的张量，需要分割的对象
    all_gather_input_split_sizes: List[int],  # 每个分割后的张量大小列表
    dim: int,  # 沿着哪个维度进行分割
    out: List[torch.Tensor],  # 存储分割后结果的列表
) -> None:
    # 调用 torch 库中的 split_with_sizes_copy 函数来执行实际的分割操作
    torch.split_with_sizes_copy(
        all_gather_output, all_gather_input_split_sizes, dim=dim, out=out
    )


# 定义一个名为 "chunk_cat" 的库函数，用于按维度和指定数量拼接张量，并将结果存储在指定的输出张量中
lib.define(
    "chunk_cat(Tensor[] tensors, int dim, int num_chunks, *, Tensor(a!) out) -> ()"
)


# 将函数声明为 torch 库的实现，绑定到 "chunk_cat" 函数，执行元信息处理
@torch.library.impl(lib, "chunk_cat", "Meta")
# 将函数声明为 torch 库的实现，绑定到 "chunk_cat" 函数，在 CUDA 设备上执行
@torch.library.impl(lib, "chunk_cat", "CUDA")
# 将函数声明为 torch 库的实现，绑定到 "chunk_cat" 函数，在 CPU 上执行
@torch.library.impl(lib, "chunk_cat", "CPU")
# 定义一个函数，用于将列表中的多个张量按指定维度和数量进行拼接，并将结果存储在指定的输出张量中
def chunk_cat(
    tensors: List[torch.Tensor],  # 待拼接的张量列表
    dim: int,  # 拼接的维度
    num_chunks: int,  # 拼接的块数
    out: torch.Tensor,  # 存储拼接结果的张量
) -> None:
    # 调用 torch 库中的 _chunk_cat 函数来执行实际的拼接操作
    torch._chunk_cat(tensors, dim, num_chunks, out=out)


# 定义一个函数，用于在不计算梯度的情况下执行全部聚合操作，接收参数和输出并返回全部聚合结果
@torch.no_grad()
def foreach_all_gather(
    fsdp_params: List[FSDPParam],  # 包含 FSDPParam 对象的列表
    group: dist.ProcessGroup,  # 分布式进程组
    async_op: bool,  # 是否异步执行操作
    all_gather_copy_in_stream: torch.cuda.Stream,  # 复制输入流
    all_gather_stream: torch.cuda.Stream,  # 聚合流
    device: torch.device,  # 设备类型
) -> Optional[AllGatherResult]:
    # 获取分布式进程组的大小和当前进程的排名
    world_size, rank = group.size(), group.rank()
    # 在复制输入流上下文中执行以下代码块
    with torch.cuda.stream(all_gather_copy_in_stream):
        # 从 fsdp_params 中获取所有聚合输入，并保存到 param_all_gather_inputs 列表中
        param_all_gather_inputs: List[List[torch.Tensor]] = [
            fsdp_param.all_gather_inputs for fsdp_param in fsdp_params
        ]
        # 获取所有聚合输入的元数据，包括数据类型和数据大小
        (
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            dtype,
        ) = _get_all_gather_input_metadatas(param_all_gather_inputs)
        # 如果数据类型为 torch.uint8，则将所有聚合输入转换为 torch.uint8 类型的张量视图
        if dtype == torch.uint8:
            all_gather_inputs = [
                t.view(torch.uint8) for ts in param_all_gather_inputs for t in ts
            ]
        else:
            # 否则直接使用原始的所有聚合输入张量列表
            all_gather_inputs = [t for ts in param_all_gather_inputs for t in ts]
        # 计算每个输入张量的大小并存储在 inp_split_sizes 列表中
        inp_split_sizes = [t.numel() for t in all_gather_inputs]
        # 计算所有聚合输入张量的总大小
        all_gather_input_numel = sum(inp_split_sizes)
        # 调用 torch 库中的 all_gather_copy_in 函数执行复制输入的所有聚合操作
        all_gather_input, all_gather_output = torch.ops.fsdp.all_gather_copy_in(
            all_gather_inputs,
            inp_split_sizes,
            all_gather_input_numel,
            world_size,
            rank,
            dtype,
            device,
        )
        # 删除 param_all_gather_inputs 以释放内存
        del param_all_gather_inputs
    # 等待复制输入流的操作完成
    all_gather_stream.wait_stream(all_gather_copy_in_stream)
    # 在聚合流上下文中执行以下代码块
    with torch.cuda.stream(all_gather_stream):
        # 执行分布式 all_gather 操作，将结果存储在 all_gather_output 中
        all_gather_work = dist.all_gather_into_tensor(
            output_tensor=all_gather_output,
            input_tensor=all_gather_input,
            group=group,
            async_op=async_op,
        )
        # 记录聚合流的事件
        all_gather_event = all_gather_stream.record_event()
        # 返回全部聚合的结果对象
        return AllGatherResult(
            all_gather_output,
            all_gather_event,
            all_gather_work,
            param_all_gather_input_dtypes,
            param_all_gather_input_numels,
            inp_split_sizes,
        )


# 定义一个函数，用于在不计算梯度的情况下执行全部聚合操作的结果复制输出
@torch.no_grad()
def foreach_all_gather_copy_out(
    all_gather_result: AllGatherResult,  # 全部聚合操作的结果对象
    fsdp_params: List[FSDPParam],  # 包含 FSDPParam 对象的列表
    group: dist.ProcessGroup,  # 分布式进程组
) -> None:
    # 省略函数体，因为此处不需要添加注释
    (
        all_gather_output,
        all_gather_event,
        all_gather_work,
        param_all_gather_input_dtypes,
        param_all_gather_input_numels,
        all_gather_input_split_sizes,
    ) = all_gather_result
    # 解包 all_gather_result 中的各个元素：all_gather_output, all_gather_event, all_gather_work,
    # param_all_gather_input_dtypes, param_all_gather_input_numels, all_gather_input_split_sizes

    if all_gather_event is not None:  # sync op
        # 如果 all_gather_event 不为 None，表示是同步操作，等待 CUDA 当前流程中的事件完成
        torch.cuda.current_stream().wait_event(all_gather_event)

    if isinstance(all_gather_work, dist.distributed_c10d.Work):  # async op
        # 如果 all_gather_work 是 dist.distributed_c10d.Work 的实例，表示是异步操作，需要等待其完成
        all_gather_work.wait()

    world_size, device = group.size(), all_gather_output.device
    # 获取 group 的大小作为 world_size，获取 all_gather_output 的设备作为 device

    for all_gather_input_numels, all_gather_input_dtypes, fsdp_param in zip(
        param_all_gather_input_numels, param_all_gather_input_dtypes, fsdp_params
    ):
        # 迭代 param_all_gather_input_numels, param_all_gather_input_dtypes, fsdp_params 的元素
        if ca.compiled_autograd_enabled:
            # 如果 compiled_autograd_enabled 开启
            fsdp_param.init_all_gather_outputs(
                all_gather_input_numels,
                all_gather_input_dtypes,
                world_size,
                device,
                # 注意：在编译模式下，确保我们始终重新创建 all_gather_outputs
                # 每次 AllGather 操作。参见 [Note: Invariants for torch.compile Traceable FSDP2]。
                force_recreate=True,
            )
        else:
            # 如果 compiled_autograd_enabled 没有开启
            fsdp_param.init_all_gather_outputs(
                all_gather_input_numels, all_gather_input_dtypes, world_size, device
            )  # 第一次调用后成为 no-op
            fsdp_param.alloc_all_gather_outputs()
            # 分配 all_gather_outputs 的空间

    all_gather_output = all_gather_output.view(world_size, -1)
    # 将 all_gather_output 重新视图为 (world_size, -1) 形状

    gen = (t for fsdp_param in fsdp_params for t in fsdp_param.all_gather_outputs)
    # 生成器表达式，用于遍历 fsdp_params 中每个 fsdp_param 的 all_gather_outputs

    if all_gather_output.dtype == torch.uint8:
        # 如果 all_gather_output 的数据类型是 torch.uint8
        out = [t.view(world_size, -1).view(torch.uint8) for t in gen]
        # 对 gen 中的每个张量 t，视图为 (world_size, -1)，然后视图为 torch.uint8
    else:
        out = [t.view(world_size, -1) for t in gen]
        # 对 gen 中的每个张量 t，视图为 (world_size, -1)

    torch.ops.fsdp.split_with_sizes_copy(
        all_gather_output, all_gather_input_split_sizes, dim=1, out=out
    )
    # 调用 torch.ops.fsdp.split_with_sizes_copy 进行操作，将 all_gather_output 按指定大小拆分，并复制到 out 中
@torch.no_grad()
def foreach_reduce(
    fsdp_params: List[FSDPParam],  # 参数列表，包含FSDPParam类型的元素
    unsharded_grads: List[torch.Tensor],  # 非分片的梯度张量列表
    reduce_scatter_group: dist.ProcessGroup,  # reduce scatter操作使用的进程组
    reduce_scatter_stream: torch.cuda.Stream,  # reduce scatter操作使用的CUDA流
    orig_dtype: torch.dtype,  # 原始数据类型
    reduce_dtype: Optional[torch.dtype],  # reduce操作使用的数据类型，可选
    device: torch.device,  # 设备类型，表示操作在哪个设备上进行
    all_reduce_group: Optional[dist.ProcessGroup],  # all reduce操作使用的进程组，仅在HSDP时不为None
    all_reduce_stream: torch.cuda.Stream,  # all reduce操作使用的CUDA流
    all_reduce_grads: bool,  # 是否进行全局reduce操作的标志
    partial_reduce_output: Optional[torch.Tensor],  # 部分reduce输出，仅在HSDP时使用
) -> Tuple[torch.Tensor, torch.cuda.Event, torch.cuda.Event, Optional[torch.Tensor]]:
    """
    ``unsharded_grads`` owns the references to the gradients computed by
    autograd, so clearing the list frees the gradients.
    """
    grad_dtypes = {grad.dtype for grad in unsharded_grads}  # 收集梯度张量的数据类型集合
    if len(grad_dtypes) != 1:
        # 检查是否所有梯度张量的数据类型一致，如果不一致则引发异常
        _raise_assert_with_print(
            f"FSDP reduce-scatter expects uniform gradient dtype but got {grad_dtypes}"
        )
    grad_dtype = unsharded_grads[0].dtype  # 获取梯度张量的数据类型
    reduce_dtype = reduce_dtype or grad_dtype  # 确定reduce操作使用的数据类型，默认为梯度张量的数据类型
    predivide_factor, postdivide_factor = _get_gradient_divide_factors(
        reduce_scatter_group, all_reduce_group, reduce_dtype
    )  # 获取梯度分割因子
    world_size = reduce_scatter_group.size()  # 获取reduce scatter操作使用的进程组大小
    padded_unsharded_sizes = tuple(
        _get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads
    )  # 计算每个梯度张量的第一个维度填充后的大小
    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)  # 计算reduce scatter操作的输入元素数
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size  # 计算reduce scatter操作的输出元素数
    reduce_scatter_input = torch.empty(
        (reduce_scatter_input_numel,), dtype=reduce_dtype, device=device
    )  # 创建reduce scatter操作的输入张量
    foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)
    current_stream = torch.cuda.current_stream()
    # 只有在复制输入完成后，才能释放梯度张量的引用
    unsharded_grads.clear()
    reduce_scatter_stream.wait_stream(current_stream)  # 等待reduce scatter操作的CUDA流完成
    # 使用给定的 CUDA 流进行 reduce-scatter 操作
    with torch.cuda.stream(reduce_scatter_stream):
        # 创建一个新的空张量用于接收 reduce-scatter 的输出
        reduce_output = reduce_scatter_input.new_empty((reduce_scatter_output_numel,))
        
        # 如果需要，对输入进行除法操作
        _div_if_needed(reduce_scatter_input, predivide_factor)
        
        # 执行 reduce-scatter 操作，将结果写入 reduce_output 中
        dist.reduce_scatter_tensor(
            output=reduce_output,
            input=reduce_scatter_input,
            group=reduce_scatter_group,
            op=ReduceOp.AVG if predivide_factor is None else ReduceOp.SUM,
        )
        
        # 记录 reduce-scatter 操作的事件
        reduce_scatter_event = reduce_scatter_stream.record_event()
        
        # 将 post_reduce_stream 设置为 reduce-scatter 流
        post_reduce_stream = reduce_scatter_stream
        
        # 如果存在 all_reduce_group（HSDP），则执行以下操作
        if all_reduce_group is not None:  # HSDP
            # 累加操作必须在 reduce-scatter 流中运行
            if not all_reduce_grads:
                # 如果 partial_reduce_output 存在，则累加 reduce_output 到其中
                if partial_reduce_output is not None:
                    partial_reduce_output += reduce_output
                else:
                    partial_reduce_output = reduce_output
                
                # 返回 reduce_scatter_input, reduce_scatter_event, post_reduce_stream 记录的事件和 partial_reduce_output
                return (
                    reduce_scatter_input,
                    reduce_scatter_event,
                    post_reduce_stream.record_event(),
                    partial_reduce_output,
                )
            
            # 如果 partial_reduce_output 存在，则累加到 reduce_output 中
            if partial_reduce_output is not None:
                reduce_output += partial_reduce_output
            
            # 将 post_reduce_stream 设置为 all_reduce_stream
            post_reduce_stream = all_reduce_stream
            
            # 等待 reduce-scatter 流完成，然后在 all_reduce 流中执行下列操作
            all_reduce_stream.wait_stream(reduce_scatter_stream)
            with torch.cuda.stream(all_reduce_stream):
                # 执行全局 all-reduce 操作，根据 predivide_factor 的值选择 AVG 或 SUM 操作
                dist.all_reduce(
                    reduce_output,
                    group=all_reduce_group,
                    op=ReduceOp.AVG if predivide_factor is None else ReduceOp.SUM,
                )
    # 使用指定的 CUDA 流执行下面的代码块
    with torch.cuda.stream(post_reduce_stream):
        # 如果需要的话，对 reduce_output 进行除法操作
        _div_if_needed(reduce_output, postdivide_factor)
        # 将 reduce_output 转换为原始数据类型（如果需要）
        reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
        
        # 遍历填充后的非分片大小和 FSDP 参数列表
        flat_grad_offset = 0  # 表示在 reduce_scatter_output_numel 范围内的偏移量
        for padded_unsharded_size, fsdp_param in zip(
            padded_unsharded_sizes, fsdp_params
        ):
            # 创建新的分片梯度张量，用于累积分片梯度
            new_sharded_grad = torch.as_strided(
                reduce_output,
                size=fsdp_param.sharded_size,
                stride=fsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            
            # 检查是否需要累积梯度
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            
            # 如果需要将梯度传输到 CPU
            if fsdp_param.offload_to_cpu:
                # 如果不需要累积梯度，则仅重叠 D2H 复制（复制到固定内存）
                non_blocking = fsdp_param.pin_memory and not to_accumulate_grad
                
                # 将 GPU 分片梯度复制到 CPU，可以选择异步操作
                new_sharded_grad = new_sharded_grad.to(
                    torch.device("cpu"), non_blocking=non_blocking
                )
                
                # 如果采用异步操作，则记录一个事件，以便在 CPU 线程上阻塞，
                # 确保 D2H 复制在优化器之前完成
                if non_blocking:
                    fsdp_param.grad_offload_event = reduce_scatter_stream.record_event()
            
            # 如果需要累积梯度
            if to_accumulate_grad:
                # 确保 fsdp_param.sharded_param.grad 是 DTensor 类型
                assert isinstance(fsdp_param.sharded_param.grad, DTensor)
                # 累积新的分片梯度到原始梯度上
                fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
            else:
                # 将新的分片梯度转换为分片 DTensor，并赋值给梯度
                new_sharded_dtensor_grad = fsdp_param.to_sharded_dtensor(
                    new_sharded_grad
                )
                fsdp_param.sharded_param.grad = new_sharded_dtensor_grad
            
            # 如果未启用编译后自动求导，则对每个钩子执行后累积梯度操作
            if not ca.compiled_autograd_enabled:
                for hook in (
                    getattr(fsdp_param.sharded_param, "_post_accumulate_grad_hooks", {})
                    or {}
                ).values():
                    hook(fsdp_param.sharded_param)
            
            # 计算填充后的分片元素数量，并更新偏移量
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            flat_grad_offset += padded_sharded_numel
        
        # 记录后处理缩减流的事件
        post_reduce_event = post_reduce_stream.record_event()
    
    # 返回缩减分散输入、缩减事件、后处理缩减事件和空值
    return reduce_scatter_input, reduce_scatter_event, post_reduce_event, None
def foreach_reduce_scatter_copy_in(
    unsharded_grads: List[torch.Tensor],
    reduce_scatter_input: torch.Tensor,
    world_size: int,
) -> None:
    # 将 reduce_scatter_input 转换为 shape 为 (world_size, -1) 的张量
    reduce_scatter_input = reduce_scatter_input.view(world_size, -1)
    # 使用 fsdp 模块的 chunk_cat 函数将 unsharded_grads 沿着 dim=0 的维度分块拼接到 reduce_scatter_input 上
    torch.ops.fsdp.chunk_cat(
        unsharded_grads, dim=0, num_chunks=world_size, out=reduce_scatter_input
    )


def _get_all_gather_input_metadatas(
    param_all_gather_inputs: List[List[torch.Tensor]],
) -> Tuple[List[List[torch.dtype]], List[List[int]], torch.dtype]:
    # 初始化两个空列表用于存储参数的数据类型和元素数量
    param_all_gather_input_dtypes: List[List[torch.dtype]] = []
    param_all_gather_input_numels: List[List[int]] = []
    # 获取第一个参数的数据类型作为 all_gather 操作的数据类型
    all_gather_dtype = param_all_gather_inputs[0][0].dtype
    # 遍历每个参数的 all_gather 输入
    for all_gather_inputs in param_all_gather_inputs:
        input_dtypes: List[torch.dtype] = []
        input_numels: List[int] = []
        # 遍历每个 all_gather 输入的张量
        for all_gather_input in all_gather_inputs:
            # 如果当前输入的数据类型与 all_gather_dtype 不同，则将 all_gather_dtype 设置为 torch.uint8
            if all_gather_input.dtype != all_gather_dtype:
                all_gather_dtype = torch.uint8
            # 记录当前输入的数据类型和元素数量
            input_dtypes.append(all_gather_input.dtype)
            input_numels.append(all_gather_input.numel())
        # 将当前参数的数据类型列表和元素数量列表添加到对应的总列表中
        param_all_gather_input_dtypes.append(input_dtypes)
        param_all_gather_input_numels.append(input_numels)
    # 返回参数的数据类型列表、元素数量列表和 all_gather 操作的数据类型
    return (
        param_all_gather_input_dtypes,
        param_all_gather_input_numels,
        all_gather_dtype,
    )


def _get_gradient_divide_factors(
    reduce_scatter_group: dist.ProcessGroup,
    all_reduce_group: Optional[dist.ProcessGroup],
    reduce_dtype: torch.dtype,
) -> Union[Tuple[None, None], Tuple[float, float]]:
    # 对于 float32 或 bfloat16 数据类型，不需要担心溢出/下溢，使用 NCCL 内置的除法以避免使用单独的除法核
    if reduce_dtype in (torch.float32, torch.bfloat16):
        return None, None
    # 计算数据并行大小
    data_parallel_size = reduce_scatter_group.size()
    if all_reduce_group is not None:
        data_parallel_size *= all_reduce_group.size()
    # 对于 fp16，由于其动态范围较小，为避免溢出/下溢，我们在归约前后除以约 sqrt(N)
    factor: int = 1
    while data_parallel_size % factor == 0 and data_parallel_size / factor > factor:
        factor *= 2
    factor = float(factor)
    # 返回除数因子和归约参与者数量除以因子后的结果
    return (factor, data_parallel_size / factor)


def _div_if_needed(tensor: torch.Tensor, div_factor: Optional[float]) -> None:
    # 如果除数因子不为 None 且大于 1，则对张量执行就地除法操作
    if div_factor is not None and div_factor > 1:
        tensor.div_(div_factor)
```