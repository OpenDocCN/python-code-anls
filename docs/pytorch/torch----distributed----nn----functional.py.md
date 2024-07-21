# `.\pytorch\torch\distributed\nn\functional.py`

```py
# mypy: allow-untyped-defs
# 引入 PyTorch 库
import torch
# 引入 PyTorch 分布式通信模块
import torch.distributed as dist
# 从 PyTorch 自动求导模块中引入 Function 类
from torch.autograd import Function

# 下面两个导入语句取决于 USE_DISTRIBUTED 编译标志的存在性，如果不存在则会抛出导入错误
# 引入 torch.distributed 中的 group 和 ReduceOp 类
from torch.distributed import group, ReduceOp


def broadcast(tensor, src, group=group.WORLD):
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Arguments:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process.
        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Received tensor from the broadcast op.

    """
    # 调用自定义的 _Broadcast.apply 方法进行广播操作
    return _Broadcast.apply(src, group, tensor)


def gather(tensor, dst=0, group=group.WORLD):
    """
    Gathers a list of tensors in a single process.

    Arguments:
        tensor (Tensor): Input tensor.
        dst (int, optional): Destination rank (default is 0).
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple[Tensor]: List of appropriately-sized tensors with the gathered data.
    """
    # 调用自定义的 _Gather.apply 方法进行聚集操作
    return _Gather.apply(dst, group, tensor)


def scatter(tensors, src=0, group=group.WORLD):
    """
    Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Arguments:
        tensors (list[Tensor]): List of tensors to scatter on the source rank.
            Receivers must pass ``None``.
        src (int, optional): Source rank (default is 0).
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output tensor from the scatter operation.

    """
    # 调用自定义的 _Scatter.apply 方法进行分散操作
    return _Scatter.apply(src, group, *tensors)


def reduce(tensor, dst, op=ReduceOp.SUM, group=group.WORLD):
    """
    Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Arguments:
        tensor (Tensor): Input of the collective.
        dst (int): Destination rank.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective.

    """
    # 调用自定义的 _Reduce.apply 方法进行归约操作
    return _Reduce.apply(dst, op, group, tensor)


def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=group.WORLD):
    """
    Reduces, then scatters a list of tensors to all processes in a group.

    ```python
    # Not implemented yet
    ```py

    """
    """
    Perform reduce and scatter operation across distributed tensors.

    Arguments:
        output (Tensor): Output tensor.
            The tensor where the result of the operation will be stored.
        input_list (list[Tensor]): List of tensors to reduce and scatter.
            List of tensors that will be reduced and scattered across processes.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum. Specifies an operation used for element-wise reductions.
            Defaults to the operation specified by the reduction operation enum.
        group (ProcessGroup, optional): The process group to work on.
            Optional parameter specifying the group of processes to perform
            collective operations with. Defaults to the default process group.

    Returns:
        Tensor: Output of the collective operation.
            The tensor resulting from the reduction and scattering of input tensors
            across processes, stored in the `output` tensor.

    """
    return _Reduce_Scatter.apply(op, group, output, *input_list)
# 将给定张量从整个组中聚集到一个列表中。
def all_gather(tensor, group=group.WORLD):
    # 调用自定义的 _AllGather 类的 apply 方法，执行聚集操作
    return _AllGather.apply(group, tensor)


# 单个张量的全局聚集。从所有排名中聚集单个张量，并将它们放入单个输出张量中。
def _all_gather_base(output_tensor, input_tensor, group=group.WORLD):
    # 调用自定义的 _AllGatherBase 类的 apply 方法，执行单个张量的全局聚集操作
    return _AllGatherBase.apply(output_tensor, input_tensor, group)


# 将输入张量列表散布给组中的所有进程，并返回聚集的张量列表。
def all_to_all(output_tensor_list, input_tensor_list, group=group.WORLD):
    # 调用自定义的 _AlltoAll 类的 apply 方法，执行输入张量列表的散布和输出张量列表的聚集操作
    return _AlltoAll.apply(group, output_tensor_list, *input_tensor_list)


# 将输入张量拆分，并将拆分列表散布给组中的所有进程。
# 然后将来自组中所有进程的接收张量连接起来，并返回单个输出张量。
def all_to_all_single(
    output,
    input,
    output_split_sizes=None,
    input_split_sizes=None,
    group=group.WORLD,
):
    # 此函数未完成，注释未提供完整的函数实现信息
    pass
    # 调用自定义的_AlltoAllSingle函数，执行分布式收集和散布操作
    return _AlltoAllSingle.apply(
        group, output, output_split_sizes, input_split_sizes, input
    )
def all_reduce(tensor, op=ReduceOp.SUM, group=group.WORLD):
    """
    Reduces the tensor data across all machines in such a way that all get the final result.

    After the call the returned tensor is going to be bitwise
    identical in all processes.

    Arguments:
        tensor (Tensor): Input of the collective.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective

    """
    return _AllReduce.apply(op, group, tensor)


class _Broadcast(Function):
    @staticmethod
    def forward(ctx, src, group, tensor):
        ctx.src = src
        ctx.group = group
        ctx.rank = dist.get_rank(group=group)
        # torch.distributed makes all the calls in place
        # we allocate new tensors to avoid this
        tensor = tensor.clone()  # 克隆输入张量，以防止in-place修改
        dist.broadcast(tensor, src, group=group)  # 在指定的进程组中进行广播操作
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        gx = _Reduce.apply(ctx.src, ReduceOp.SUM, ctx.group, grad_output)  # 对梯度进行全局reduce操作
        if ctx.src != ctx.rank:
            gx.zero_()  # 如果当前进程不是广播源进程，则将梯度置零
        return (None, None, gx)  # 返回梯度


class _Gather(Function):
    @staticmethod
    def forward(ctx, dst, group, tensor):
        ctx.dst = dst
        ctx.group = group
        # Need to create a list of tensors here to do the
        # aggregation, get it from the group size
        # tensor should be correctly sized for the method
        # gathering
        tensor_list = [
            torch.zeros_like(tensor) for i in range(dist.get_world_size(group=group))
        ]  # 根据进程组大小创建张量列表，用于收集操作

        tensor = tensor.contiguous()  # 确保张量是连续的
        if dist.get_rank(group=group) == dst:
            dist.gather(tensor, tensor_list, dst, group=group)  # 执行收集操作到指定进程
        else:
            dist.gather(tensor, None, dst, group=group)  # 对于非目标进程，不进行收集操作
        return tuple(tensor_list)  # 返回收集到的张量列表

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None) + (_Scatter.apply(ctx.dst, ctx.group, *grad_outputs),)  # 向上流动梯度，执行分发操作


class _Scatter(Function):
    @staticmethod
    def forward(ctx, src, group, *tensors):
        ctx.src = src
        ctx.group = group
        assert all(t.size() == tensors[0].size() for t in tensors)  # 断言所有张量的大小一致
        output = torch.zeros_like(tensors[0])  # 创建与输入张量同样大小的输出张量
        if dist.get_rank(group=group) == src:
            dist.scatter(output, list(tensors), src, group=group)  # 在指定进程组中执行分发操作
        else:
            dist.scatter(output, None, src, group=group)  # 对于非源进程，不进行分发操作
        return output  # 返回分发后的张量

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None) + _Gather.apply(ctx.src, ctx.group, grad_output)  # 向上流动梯度，执行收集操作


class _Reduce(Function):
    @staticmethod
    def forward(ctx, src, op, group, tensor):
        ctx.src = src
        ctx.group = group
        tensor = tensor.clone()  # 克隆输入张量，以防止in-place修改
        dist.reduce(tensor, src, op=op, group=group)  # 在指定进程组中进行reduce操作
        return tensor

    @staticmethod
    # 定义一个函数 backward，接收两个参数 ctx 和 grad_output
    def backward(ctx, grad_output):
        # 返回一个元组，包含四个 None 对象和 _Broadcast.apply 的返回值
        return (None, None, None) + (_Broadcast.apply(ctx.src, ctx.group, grad_output),)
# 定义一个名为 _Reduce_Scatter 的类，继承自 Function 类
class _Reduce_Scatter(Function):
    # 静态方法：前向传播方法，用于执行 reduce_scatter 操作
    @staticmethod
    def forward(ctx, op, group, tensor, *input_tensor_list):
        # 将 group 存储在上下文中
        ctx.group = group
        # 需要连续的张量进行集合操作
        tensor = tensor.contiguous()
        # 将输入张量列表中的每个张量都转换为连续的张量
        input_tensor_list = tuple(t.contiguous() for t in input_tensor_list)
        # 调用 dist.reduce_scatter 执行 reduce_scatter 操作
        dist.reduce_scatter(tensor, list(input_tensor_list), op=op, group=group)
        # 返回张量作为前向传播的结果
        return tensor

    # 静态方法：反向传播方法
    @staticmethod
    def backward(ctx, grad_output):
        # 返回一个元组，包含三个 None，和通过 _AllGather.apply 方法计算的梯度
        return (None, None, None) + _AllGather.apply(ctx.group, grad_output)


# 定义一个名为 _AllGather 的类，继承自 Function 类
class _AllGather(Function):
    # 静态方法：前向传播方法，用于执行 all_gather 操作
    @staticmethod
    def forward(ctx, group, tensor):
        # 需要连续的张量进行集合操作
        tensor = tensor.contiguous()

        # 将 group 存储在上下文中
        ctx.group = group
        # 创建一个与 tensor 同样大小的空张量列表
        out_tensor_list = [
            torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))
        ]

        # 调用 dist.all_gather 执行 all_gather 操作
        dist.all_gather(out_tensor_list, tensor, group=group)
        # 返回一个元组，包含所有收集到的张量列表
        return tuple(out_tensor_list)

    # 静态方法：反向传播方法
    @staticmethod
    def backward(ctx, *grad_outputs):
        # 如果使用的是 NCCL 后端
        if dist.get_backend(group=ctx.group) is dist.Backend.NCCL:
            # 获取当前进程的排名
            rank = dist.get_rank(group=ctx.group)
            # 创建一个与 grad_outputs[rank] 同样大小的空张量 gx
            gx = torch.empty_like(grad_outputs[rank])
            # 调用 _Reduce_Scatter.apply 执行 reduce_scatter 操作，求梯度
            gx = _Reduce_Scatter.apply(ReduceOp.SUM, ctx.group, gx, *grad_outputs)
        else:
            # 如果不支持 ReduceScatter，使用 AlltoAll 和 .sum() 来模拟 ReduceScatter 的行为
            tensor_list = [torch.empty_like(tensor) for tensor in grad_outputs]
            gxs = _AlltoAll.apply(ctx.group, tensor_list, *grad_outputs)
            gx = torch.sum(torch.stack(gxs), dim=0)
        # 返回一个元组，包含两个 None 和计算得到的梯度 gx
        return (None, gx)


# 定义一个名为 _AllGatherBase 的类，继承自 Function 类
class _AllGatherBase(Function):
    # 静态方法：前向传播方法，用于执行 _all_gather_base 操作
    @staticmethod
    def forward(ctx, output_tensor, input_tensor, group):
        # 将 group 存储在上下文中
        ctx.group = group
        # 调用 dist._all_gather_base 执行 _all_gather_base 操作
        dist._all_gather_base(output_tensor, input_tensor.contiguous(), group=group)
        # 返回输出张量作为前向传播的结果
        return output_tensor

    # 静态方法：反向传播方法
    @staticmethod
    def backward(ctx, grad_output):
        # 如果使用的是 NCCL 后端
        if dist.get_backend(group=ctx.group) is dist.Backend.NCCL:
            # 获取当前进程组的世界大小
            world_size = dist.get_world_size(group=ctx.group)
            # 获取 grad_output 的大小
            out_size = list(grad_output.size())
            # 如果第一维度不能被 world_size 整除，抛出异常
            if out_size[0] % world_size != 0:
                raise RuntimeError(
                    f"Tensor with dimensions: {out_size} does "
                    f"not have first dimension divisible by world_size: {world_size}"
                )
            # 修改 out_size 的第一维度
            out_size[0] = out_size[0] // dist.get_world_size(group=ctx.group)
            # 创建一个与 grad_output 大小和设备类型相同的空张量 gx
            gx = torch.empty(
                out_size, device=grad_output.device, dtype=grad_output.dtype
            )
            # 调用 dist._reduce_scatter_base 执行 _reduce_scatter_base 操作
            dist._reduce_scatter_base(gx, grad_output, ReduceOp.SUM, ctx.group)
        else:
            # 如果后端不支持，则抛出异常
            raise RuntimeError("Backend not supported!")
        # 返回一个元组，包含三个 None 和计算得到的梯度 gx
        return (None, gx, None)


class _AlltoAll(Function):
    @staticmethod
    def forward(ctx, group, out_tensor_list, *tensors):
        # 设置上下文的分组信息
        ctx.group = group
        # 记录输入张量的大小列表，每个张量的大小由其尺寸函数返回
        ctx.input_tensor_size_list = [
            tensors[i].size() for i in range(dist.get_world_size(group=group))
        ]
        # 获取当前进程在指定组中的排名
        my_rank = dist.get_rank(group=group)
        # 保证所有输入张量是连续的，以确保内存布局符合要求
        tensors = tuple(t.contiguous() for t in tensors)
        # 如果使用GLOO后端
        if dist.get_backend(group=group) is dist.Backend.GLOO:
            # 遍历组中的所有进程
            for i in range(dist.get_world_size(group=group)):
                to_send = None
                # 如果当前进程是第i个进程，则将tensors作为数据发送
                if i == my_rank:
                    to_send = list(tensors)
                # 使用scatter函数将数据发送给各个进程
                dist.scatter(out_tensor_list[i], to_send, i, group=group)
        else:
            # 使用all_to_all函数进行全局的所有到所有通信
            dist.all_to_all(
                out_tensor_list,
                list(tensors),
                group=group,
            )
        # 返回输出张量列表
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # 根据输入张量大小列表创建张量列表
        tensor_list = [
            torch.empty(
                size, device=grad_outputs[0].device, dtype=grad_outputs[0].dtype
            )
            for size in ctx.input_tensor_size_list
        ]
        # 返回None作为梯度输入，以及生成的张量列表作为梯度输出
        return (None, None) + _AlltoAll.apply(ctx.group, tensor_list, *grad_outputs)
class _AlltoAllSingle(Function):
    @staticmethod
    def forward(ctx, group, output, output_split_sizes, input_split_sizes, input):
        # 保存上下文信息，用于反向传播
        ctx.group = group
        ctx.input_size = input.size()
        ctx.output_split_sizes = input_split_sizes
        ctx.input_split_sizes = output_split_sizes
        # 调用分布式通信库的 all_to_all_single 函数进行数据交换
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        # 返回输出张量
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 创建一个新的张量，用于存储反向传播梯度
        tensor = torch.empty(
            ctx.input_size, device=grad_output.device, dtype=grad_output.dtype
        )
        # 返回梯度信息和下一层的反向传播结果
        return (None, None, None, None) + (
            _AlltoAllSingle.apply(
                ctx.group,
                tensor,
                ctx.output_split_sizes,
                ctx.input_split_sizes,
                grad_output.contiguous(),
            ),
        )


class _AllReduce(Function):
    @staticmethod
    def forward(ctx, op, group, tensor):
        # 保存操作类型和分组信息到上下文
        ctx.group = group
        ctx.op = op
        # 克隆输入张量，以避免修改原始数据
        tensor = tensor.clone()
        # 调用分布式通信库的 all_reduce 函数进行张量规约操作
        dist.all_reduce(tensor, op=op, group=group)
        # 返回处理后的张量
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # 返回梯度信息和下一层的反向传播结果
        return (None, None) + (_AllReduce.apply(ctx.op, ctx.group, grad_output),)
```