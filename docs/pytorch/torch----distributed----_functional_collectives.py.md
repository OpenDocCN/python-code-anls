# `.\pytorch\torch\distributed\_functional_collectives.py`

```py
# mypy: allow-untyped-defs
# 引入系统模块和警告模块
import sys
import warnings
# 引入类型提示相关模块和函数
from typing import cast, List, Optional, Tuple, TYPE_CHECKING, Union

# 引入PyTorch主模块和分布式模块
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.distributed.device_mesh import DeviceMesh
from torch.fx.experimental.proxy_tensor import get_innermost_proxy_mode

# 引入自定义的函数实现模块
from . import _functional_collectives_impl as fun_col_impl

# 尝试导入C++扩展模块中的tree_map_only函数，若导入失败则导入Python实现的tree_map_only函数
try:
    from torch.utils._cxx_pytree import tree_map_only
except ImportError:
    from torch.utils._pytree import tree_map_only  # type: ignore[no-redef]

# 如果当前是在使用Torch Deploy环境中运行
if torch._running_with_deploy():

    def is_torchdynamo_compiling():
        """在Torch Deploy环境中无法导入torchdynamo。"""
        return False

# 如果不是在Torch Deploy环境中，则尝试导入torch.compiler模块中的is_dynamo_compiling函数
else:
    try:
        from torch.compiler import is_dynamo_compiling as is_torchdynamo_compiling
    # 若导入异常，则给出警告并定义is_torchdynamo_compiling函数返回False
    except Exception:
        warnings.warn(
            "Unable to import torchdynamo util `is_torchdynamo_compiling`, so won't support torchdynamo correctly"
        )

        def is_torchdynamo_compiling():
            return False

"""
New traceable, functional collectives.
RFC: https://github.com/pytorch/pytorch/issues/93173

  compiler: trace these ops with plain-old-data schemas, then choose how to lower them.
  eager: execute these 'functional' ops which in eager return AsyncCollectiveTensor subclasses,
         automatically calling .wait() on underlying/hidden async 'work' obj only when fed to
         a downstream op.

Issues:
* Where should these ops live? Couldn't `import torch` if putting these ops in existing torch.distributed files
* Proper support for eager requires inplace ops. We should explore having it as an option for the API.
"""

"""
Functional collectives are asynchronous only and we perform implicit stream synchronization
on behalf of the user.

We use AsyncCollectiveTensor to wrap the result tensor of a collective and it lets us witness
first usage of the tensor and insert cross stream sync at the right place.

The above are the easy bits, the hard one is how we match the Work object returned by
c10d and the tensor AsyncCollectiveTensor wraps. We alloc the tensor inside the collective
op implementation (see ``clone()`` call in ``_all_reduce``) and then it's handled by the
dispatcher which might call other implementations that are allowed to change the returned
tensor - even return a tensor with a different shape (see ``torch.vmap``).

This means the caller of our ops receives a Tensor that is not guaranteed to be the same
allocated by our implementations and that makes pairing The AsyncTensor to the original
tensor a lot harder. This pairing is needed so we can lookup the Work object to use.

Originally, we tried WeakKeyDictionary to map from Tensor to Work, but because Tensor's
identity is not stable across dispatch, the op caller would end up with a different Tensor
instance that would not match any in the dictionary.

With Tensor identity out of the question, we decided use the tensor data pointer, which
"""
"""
should be stable across all the Tensor changes done during dispatch.

We have a dictionary of tensor::data_ptr -> Work that we insert right after we call into c10d.

We use this dictionary when AsyncCollectiveTensor is used to invoke Work::wait()

Finally, we setup a finalizer against the tensor wrapper to observe it getting collected so we
can clean up stale entries in the dictionary.

To eliminate the possibility of races we have a global version counter that is used by the finalizer.

As a wise man said once: Don't cross the streams (https://www.youtube.com/watch?v=wyKQe_i9yyo)

"""

"""
Functional collectives can accept any of these types to describe the ranks participating in collectives.

The different types will be desugared to a canonical format
"""
RANK_TYPES = Union[
    List[int],                           # 可接受整数列表描述参与集体操作的排名
    List[List[int]],                     # 可接受整数列表的列表描述参与集体操作的排名
    dist.ProcessGroup,                   # 可接受分布式进程组对象描述参与集体操作的排名
    DeviceMesh,                          # 可接受设备网格对象描述参与集体操作的排名
    Tuple["dist._tensor.DeviceMesh", int], # 可接受元组描述参与集体操作的排名和整数
    str,                                 # 可接受字符串描述参与集体操作的排名
]


"""
User facing APIs for functional collectives
-------------------------------------------

These apis are called by user code and expected to work both in eager execution and compilation,
but there are significant differences to how the two modes are implemented underneath.

Eager execution is 'optimized' using a tensor subclass that schedules the synchronization (via wait_tensor() op)
just before the tensor is first used.  Compiled tracing currently relies on the compiler to perform this optimization,
and cannot yet correctly trace the AsyncTensor wrapper class.  In the future, these paths may be unified
if sufficient subclass support is added in dynamo.

Example: all_reduce is an entrypoint API, and other collectives follow a similar pattern.

Here's how it works under torch.compile/dynamo:
all_reduce(...)
  |--> _expand_group(...)               - 将 processgroup 展开为规范化/可追踪格式
  |--> c10d_functional.all_reduce(...)  - dynamo 捕获此操作调用，不会深入追踪
  |--> _maybe_wrap_tensor(...)          - 立即调用 wait_tensor() 操作，不需要 AsyncTensor 子类

And under eager execution:
all_reduce(...)
  |--> _expand_group(...)               - 与上述相同，但对于 eager 模式不那么关键
  |--> c10d_functional.all_reduce(...)  - 调度实际核函数或记录追踪中的操作
  |--> _maybe_wrap_tensor(...)          - 对返回的张量应用 AsyncTensor 包装器，在首次使用时发出 wait_tensor()

"""


def wait_tensor(tensor):
    """
    Wait on a tensor returned by the collectives ops.

    Waiting follows device semantics, which means blocking on CPU and synchronizing streams on CUDA.
    """
    return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]


def broadcast(self: torch.Tensor, src: int, group: RANK_TYPES, tag: str = ""):
    """
    Broadcasts the tensor to all processes in the given process group.
    """
    # 将张量广播到给定进程组中的所有进程
    # 使用给定的源等级(src)和进程组(group)进行广播操作，发送数据到指定进程组中的所有进程
    group_name = _resolve_group_name(group, tag)
    # 调用底层的 C++ 函数进行广播操作，将 self（当前对象）广播给指定的进程组(group_name)
    tensor = torch.ops._c10d_functional.broadcast(self, src, group_name)
    # 可能对返回的张量(tensor)进行包装，根据需要执行一些额外的操作
    return _maybe_wrap_tensor(tensor)
# 在所有机器上对张量数据进行归约操作，确保所有机器都获得最终结果
def all_reduce(self: torch.Tensor, reduceOp: str, group: RANK_TYPES, tag: str = ""):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    The input tensor is left unmodified.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    # 解析并获取组名，用于确定执行归约操作的机器组
    group_name = _resolve_group_name(group, tag)
    # 调用底层 C++ 函数执行张量归约操作
    tensor = torch.ops._c10d_functional.all_reduce(self, reduceOp.lower(), group_name)
    # 将结果进行包装处理，确保返回值的类型与输入张量一致
    return _maybe_wrap_tensor(tensor)


# 在所有机器上收集张量数据并在指定维度上进行拼接
def all_gather_tensor(
    self: torch.Tensor,
    gather_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    """
    Gather tensor data across from all machines and concatenate over ``gather_dim``.

    Note that it currently only supports gather_dim = 0.

    The input tensor is left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    # 断言张量是连续的，以确保能够正确进行收集操作
    assert self.is_contiguous()
    # 解析并获取组名，用于确定执行收集操作的机器组
    group_name = _resolve_group_name(group, tag)
    # 获取组的大小，即参与收集的机器数量
    group_size = c10d._get_group_size_by_name(group_name)
    # 调用底层 C++ 函数执行张量收集操作
    tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        self, group_size, group_name
    )
    # 对结果进行可能的包装处理，确保返回值的类型与输入张量一致
    res = _maybe_wrap_tensor(tensor)
    # 如果指定的收集维度不是0，需要进行拼接操作
    if gather_dim != 0:
        # 如果结果是异步集合张量类型，则需要等待其完成收集操作
        if isinstance(res, AsyncCollectiveTensor):
            res = res.wait()  # type: ignore[attr-defined]
        # 在指定维度上进行分块操作并拼接结果
        res = torch.cat(torch.chunk(res, group_size, dim=0), dim=gather_dim)
    return res


# 在所有机器上使用自动求导功能收集张量数据并在指定维度上进行拼接
def all_gather_tensor_autograd(
    self: torch.Tensor,
    gather_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    """
    Gather tensor data across from all machines and concatenate over ``gather_dim``.

    Note that it currently only supports gather_dim = 0.
    """
    # 函数功能与 all_gather_tensor 相同，在代码逻辑上没有额外的区别，因此不再添加重复的注释
    assert self.is_contiguous()
    group_name = _resolve_group_name(group, tag)
    group_size = c10d._get_group_size_by_name(group_name)
    tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        self, group_size, group_name
    )
    res = _maybe_wrap_tensor(tensor)
    if gather_dim != 0:
        if isinstance(res, AsyncCollectiveTensor):
            res = res.wait()  # type: ignore[attr-defined]
        res = torch.cat(torch.chunk(res, group_size, dim=0), dim=gather_dim)
    return res
    """
    This function is the same as all_gather_tensor but will propagate the
    backwards gradient across workers.

    See all_gather_tensor for more details on usage.
    """
    # 解析并获取最终的通信组名称
    group_name = _resolve_group_name(group, tag)
    # 通过组名称获取通信组的大小
    group_size = c10d._get_group_size_by_name(group_name)

    # 调用底层 C++ 函数进行张量的全局收集
    tensor = torch.ops._c10d_functional_autograd.all_gather_into_tensor(
        self, group_size, group_name
    )
    # 将收集到的张量应用到 _FromTorchTensor 自定义函数中
    res = _FromTorchTensor.apply(tensor)
    
    # TODO this should be done inside AsyncCollectiveTensor to delay the wait() call
    # 如果 gather_dim 不为 0，则进行以下操作
    if gather_dim != 0:
        # 如果 res 是 AsyncCollectiveTensor 类型，则等待其完成异步操作
        if isinstance(res, AsyncCollectiveTensor):
            res = res.wait()  # type: ignore[attr-defined]
        # 按照指定维度进行分块并拼接张量
        res = torch.cat(torch.chunk(res, group_size, dim=0), dim=gather_dim)
    
    # 返回最终的结果张量
    return res
# 定义一个方法，用于在多台机器上以某种方式减少张量数据，并将结果分散到对应的排名中
def reduce_scatter_tensor(
    self: torch.Tensor,
    reduceOp: str,
    scatter_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result, then scatter the results to corresponding ranks.


    The input tensor is left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh
    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    # 解析组名称，根据给定的组和标签
    group_name = _resolve_group_name(group, tag)
    # 获取组大小
    group_size = c10d._get_group_size_by_name(group_name)

    # 确保输入张量在指定的维度上可以被组大小整除
    assert (
        self.size(scatter_dim) % group_size == 0
    ), f"input dimension {scatter_dim} ({self.size(scatter_dim)}) must be a multiple of group_size {group_size}"
    
    # 如果 scatter_dim 不为 0，则按照 scatter_dim 维度分割张量
    if scatter_dim != 0:
        tensor_list = torch.chunk(self, group_size, dim=scatter_dim)
        self = torch.cat(tensor_list)

    # 调用底层 C++ 函数执行张量的 reduce scatter 操作
    tensor = torch.ops._c10d_functional.reduce_scatter_tensor(
        self,
        reduceOp.lower(),
        group_size,
        group_name,  # type: ignore[possibly-undefined]
    )
    # 将结果张量进行封装
    res = _maybe_wrap_tensor(tensor)
    return res


def reduce_scatter_tensor_autograd(
    self: torch.Tensor,
    reduceOp: str,
    scatter_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result, then scatter the results to corresponding ranks.

    This function is the same as reduce_scatter_tensor but will propagate the
    backwards gradient across workers.

    Currently only the "sum" reduceOp is supported.

    See reduce_scatter_tensor for more details on usage.
    """
    # 解析组名称，根据给定的组和标签
    group_name = _resolve_group_name(group, tag)
    # 获取组大小
    group_size = c10d._get_group_size_by_name(group_name)

    # 确保输入张量在指定的维度上可以被组大小整除
    assert (
        self.size(scatter_dim) % group_size == 0
    ), f"input dimension {scatter_dim} ({self.size(scatter_dim)}) must be a multiple of group_size {group_size}"
    
    # 如果 scatter_dim 不为 0，则按照 scatter_dim 维度分割张量
    if scatter_dim != 0:
        tensor_list = torch.chunk(self, group_size, dim=scatter_dim)
        self = torch.cat(tensor_list)

    # 调用带有自动微分功能的底层 C++ 函数执行张量的 reduce scatter 操作
    tensor = torch.ops._c10d_functional_autograd.reduce_scatter_tensor(
        self,
        reduceOp.lower(),
        group_size,
        group_name,  # type: ignore[possibly-undefined]
    )
    # 应用自定义的 Torch Tensor 包装器
    res = _FromTorchTensor.apply(tensor)
    return res


def all_reduce_coalesced(
    self: List[torch.Tensor], reduceOp: str, group: RANK_TYPES, tag: str = ""
) -> List[torch.Tensor]:
    """
    Reduces a list of tensors across all machines in such a way that all get
    # 根据给定的分组信息（group）和标签（tag），解析出分组名称
    group_name = _resolve_group_name(group, tag)
    # 调用底层 C10D 函数库实现的聚合操作，将所有张量进行汇总归约
    tensor_list = torch.ops._c10d_functional.all_reduce_coalesced(  # type: ignore[attr-defined]
        self,  # 当前对象引用
        reduceOp.lower(),  # 使用指定的归约操作（reduceOp），并转换为小写
        group_name,  # 使用解析得到的分组名称
    )
    # 对结果列表中的每个张量进行可能的包装（处理），然后返回结果列表
    return list(map(_maybe_wrap_tensor, tensor_list))
def all_gather_into_tensor_coalesced(
    self: List[torch.Tensor], group: RANK_TYPES, tag: str = ""
) -> List[torch.Tensor]:
    """
    Gather a list of tensors across from all machines.

    Note that it currently only supports gather_dim = 0.

    The input tensor is left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    # 解析并获取分组名称，基于给定的组类型和标签
    group_name = _resolve_group_name(group, tag)
    # 获取指定组名的组大小（成员数量）
    group_size = c10d._get_group_size_by_name(group_name)
    # 调用底层 C++ 函数进行张量收集操作，返回收集到的张量列表
    tensor_list = torch.ops._c10d_functional.all_gather_into_tensor_coalesced(  # type: ignore[attr-defined]
        self,
        group_size,
        group_name,
    )
    # 对收集到的张量列表应用可能的张量包装处理，并转换为 Python 列表返回
    return list(map(_maybe_wrap_tensor, tensor_list))


def reduce_scatter_tensor_coalesced(
    inputs: List[torch.Tensor],
    reduceOp: str,
    scatter_dim: List[int],
    group: RANK_TYPES,
    tag: str = "",
) -> List[torch.Tensor]:
    """
    Reduces a list of tensors across all machines in such a way that all get
    the final result, then scatter the results to corresponding ranks.

    The input tensors are left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    # 解析并获取分组名称，基于给定的组类型和标签
    group_name = _resolve_group_name(group, tag)
    # 获取指定组名的组大小（成员数量）
    group_size = c10d._get_group_size_by_name(group_name)

    # 确保 scatter_dim 的长度与输入张量列表的长度相同
    assert len(scatter_dim) == len(inputs)
    # 遍历输入的 scatter_dim 和 inputs 的组合，确保每个指定维度上的大小是组大小的整数倍
    for idx, (dim, tensor) in enumerate(zip(scatter_dim, inputs)):
        assert (
            tensor.size(dim) % group_size == 0
        ), f"input dimension {dim} ({tensor.size(dim)} must be a multiple of group_size {group_size} for tensor at index {idx}"
        # 如果 dim 不等于 0，则按照指定的 dim 维度对张量进行切块，并将结果拼接回 inputs 中的原始张量位置
        if dim != 0:
            tensor_list = torch.chunk(tensor, group_size, dim=dim)
            inputs[idx] = torch.cat(tensor_list)
    tensor_list = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(  # 调用 Torch C10D 模块的函数 reduce_scatter_tensor_coalesced 进行张量的聚合和分散操作
        inputs,  # 输入张量列表，用于聚合和分散操作
        reduceOp.lower(),  # 通过 reduceOp 对象调用 lower 方法，获取聚合操作的类型
        group_size,  # 分组大小，用于指定聚合操作的分组尺寸
        group_name,  # 分组名称，可能未定义，用于指定执行聚合操作的分组
    )

    return list(map(_maybe_wrap_tensor, tensor_list))  # 将 tensor_list 中的每个张量应用 _maybe_wrap_tensor 函数，并返回包含处理后张量的列表
# 检查目标操作是否为视图操作
def _is_view_op(tgt):
    assert isinstance(tgt, torch._ops.OpOverload)
    # 获取操作的模式定义
    schema = tgt._schema
    if len(schema.arguments) > 0:
        # 获取操作的第一个参数
        first_arg = schema.arguments[0]
        # 检查是否为视图操作
        return first_arg.alias_info is not None and not first_arg.alias_info.is_write

# 执行所有到所有通信的单步操作
def all_to_all_single(
    self: torch.Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
    group: RANK_TYPES,
    tag: str = "",
) -> torch.Tensor:
    """
    每个进程将输入张量分割，然后将分割列表分散到组中的所有进程。
    然后将组中所有进程接收到的张量连接起来，并返回单个输出张量。

    Group 可以是以下之一：
        List[int]: 参与集体通信的进程排名列表。
        List[List[int]]: MPMD 中参与此集体通信的二维网格排名。
        ProcessGroup: 使用进程组的排名和标签执行集体通信。
        DeviceMesh: 在网格的所有排名上执行 SPMD 集体通信。
        (DeviceMesh, int): 在 DeviceMesh 的一个维度上执行 MPMD 集体通信。

    :: 注意：如果传递 PG 或 1D 列表以执行 MPMD 集体通信，编译器将无法恢复该信息并执行集体代数优化。
    使用其他形式的输入进行操作。
    """
    if output_split_sizes is not None:
        assert all(
            isinstance(size, (int, torch.SymInt)) for size in output_split_sizes
        ), output_split_sizes
    if input_split_sizes is not None:
        assert all(
            isinstance(size, (int, torch.SymInt)) for size in input_split_sizes
        ), input_split_sizes
    # 解析组名称
    group_name = _resolve_group_name(group, tag)
    # 获取组的大小
    group_size = c10d._get_group_size_by_name(group_name)
    if output_split_sizes is None or input_split_sizes is None:
        assert output_split_sizes is None and input_split_sizes is None, (
            "output_split_sizes and input_split_sizes must either be "
            "specified together or both set to None"
        )
        # 如果未指定输出和输入的分割大小，则默认为每个组大小平均分割
        output_split_sizes = [self.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes
    # 调用底层 C++ 函数执行所有到所有通信
    tensor = torch.ops._c10d_functional.all_to_all_single(  # type: ignore[attr-defined]
        self,
        output_split_sizes,
        input_split_sizes,
        group_name,
    )
    return _maybe_wrap_tensor(tensor)

# 支持自动求导的 all_to_all_single 函数的变体
def all_to_all_single_autograd(
    self: torch.Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
    group: RANK_TYPES,
    tag: str = "",
) -> torch.Tensor:
    """
    和 all_to_all_single 相同，但支持自动求导。
    """
    if output_split_sizes is not None:
        assert all(
            isinstance(size, (int, torch.SymInt)) for size in output_split_sizes
        ), output_split_sizes
    # 如果输入的 input_split_sizes 不为 None，则进行断言检查：
    # 检查所有元素是否为整数或 torch.SymInt 类型
    assert all(
        isinstance(size, (int, torch.SymInt)) for size in input_split_sizes
    ), input_split_sizes

    # 解析 group 参数并获取完整的分组名称
    group_name = _resolve_group_name(group, tag)
    
    # 根据分组名称获取分组的大小
    group_size = c10d._get_group_size_by_name(group_name)
    
    # 如果 output_split_sizes 或 input_split_sizes 任一为 None：
    assert output_split_sizes is None and input_split_sizes is None, (
        "output_split_sizes and input_split_sizes must either be "
        "specified together or both set to None"
    )
    
    # 根据分组大小计算默认的 output_split_sizes 和 input_split_sizes
    output_split_sizes = [self.shape[0] // group_size] * group_size
    input_split_sizes = output_split_sizes
    
    # 调用 Torch 的 C++ 扩展函数 all_to_all_single 进行数据交换
    tensor = torch.ops._c10d_functional_autograd.all_to_all_single(  # type: ignore[attr-defined]
        self,
        output_split_sizes,
        input_split_sizes,
        group_name,
    )
    
    # 返回经过 _FromTorchTensor 类处理后的张量对象
    return _FromTorchTensor.apply(tensor)
# 定义一个方法，用于对张量进行置换，根据给定的源/目标对。`src_dst` 应该被定义为 src_dst[m] == n 意味着 m 发送到 n。

class AsyncCollectiveTensor(torch.Tensor):
    """
    一个张量包装的子类，用于在首次使用底层张量之前触发等待。
    在像以下这样的功能性集体PyTorch包装器中使用它：
    def functional_collective(self, group, tag):
        tag, rankset, group_size = _expand_group(group, tag)
        tensor = torch.ops.c10d_functional.{collective}(self, tag, rankset, group_size)
        return _maybe_wrap_tensor(tensor)
    """

    elem: torch.Tensor  # 包装的元素张量
    completed: bool  # 标识是否完成等待

    __slots__ = ["elem", "completed"]

    @staticmethod
    def __new__(cls, elem: torch.Tensor):
        # 创建一个新的张量子类包装器
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
        )
        r.elem = elem  # 设置包装的元素
        r.completed = False  # 初始化未完成等待状态
        return r

    def __tensor_flatten__(self):
        return ["elem"], None  # 展平时仅包含 `elem`

    def tolist(self):
        return self.trigger_wait().tolist()  # 调用 `trigger_wait()` 等待后转换为列表

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        assert meta is None
        elem = inner_tensors["elem"]
        return AsyncCollectiveTensor(elem)  # 返回一个新的 AsyncCollectiveTensor 实例，使用 `elem`

    def __repr__(self):
        return f"AsyncCollectiveTensor({self.trigger_wait()})"  # 返回包含等待结果的字符串表示

    def trigger_wait(self):
        if not self.completed:
            out = wait_tensor(self.elem)  # 如果尚未完成等待，则等待张量
            self.completed = True  # 设置为已完成等待
            return out
        else:
            return self.elem  # 否则直接返回张量本身

    def wait(self) -> torch.Tensor:
        return wait_tensor(self.elem)  # 等待底层张量的函数
    # 返回对象内部的元素作为 ACS 的底层张量
    def _get_acs_underlying_tensor(self):
        """This method enables  _functional_collectives_impl to test if a tensor is an ACS"""
        # 直接返回对象内部的元素
        return self.elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # 检查是否为 torch.ops.aten.view.default 函数
        if func == torch.ops.aten.view.default:
            # 快速处理 aten.view 操作，因为很多与 view 相关的操作都会经过 aten.view
            # 这样可以避免 pytree 的性能下降
            res = func(args[0].elem, args[1])
            # 将结果包装成 AsyncCollectiveTensor 对象
            wrapper_res = AsyncCollectiveTensor(res)
            return wrapper_res

        # 检查 func 是否为 view 操作
        is_view_op = _is_view_op(func)

        def unwrap(e: AsyncCollectiveTensor):
            # wait_tensor 是幂等的，只会进行一次流同步
            if not is_view_op:
                return e.trigger_wait()
            return e.elem

        def wrap(e: torch.Tensor):
            # wait_tensor 是幂等的，只会进行一次流同步
            assert not isinstance(e, AsyncCollectiveTensor)
            res = AsyncCollectiveTensor(e)
            return res

        # 使用 tree_map_only 将 AsyncCollectiveTensor 对象的 unwrap 函数应用到 args 和 kwargs 上
        unwrapped_args = tree_map_only(AsyncCollectiveTensor, unwrap, args)
        unwrapped_kwargs = tree_map_only(AsyncCollectiveTensor, unwrap, kwargs)

        # 调用 func 函数，并传入解包后的参数
        out = func(*unwrapped_args, **unwrapped_kwargs)

        # 如果是 view 操作，则需要重新包装输出
        if is_view_op:
            out = tree_map_only(torch.Tensor, wrap, out)

        return out

    # 将 ACS 对象转换为 numpy 数组
    def numpy(self):
        # 等待对象完成操作并将结果转换为 numpy 数组
        return self.wait().numpy()
"""
Utils and infrastructure for tracing support
"""


def _expand_group(group: RANK_TYPES, tag: str = "") -> Tuple[str, List[int], int]:
    """
    _expand_group desugars the different RANK_TYPES types into a canonical format that is traceable.

    By having this be part of the explicit eager codepath, we avoid having to specialize behavior inside
    torchdynamo and can still interoperate with processgroup objects or other untraceable forms.
    """
    # 如果是类型检查阶段（TYPE_CHECKING为True），定义类型转换函数
    if TYPE_CHECKING:

        def cast_listlistint(x):
            return cast(List[List[int]], x)

        def cast_listint(x):
            return cast(List[int], x)

    else:
        # 在运行时定义虚拟的类型转换操作，因为 dynamo 不支持真实的类型转换
        # dynamo 不喜欢遇到 'typing' 对象 ()
        def cast_listlistint(x):
            return x

        def cast_listint(x):
            return x

    rankset: List[int]
    # 根据不同的输入类型，展开和标准化组对象
    if isinstance(group, list):
        if isinstance(group[0], list):
            # 如果是嵌套列表，则进行类型转换并展开rankset
            nested_list = cast_listlistint(group)
            rankset = []
            group_size = -1
            for rs in nested_list:
                rankset.extend(rs)
                if group_size != -1 and group_size != len(rs):
                    raise ValueError(
                        f"group sizes must be identical found {group_size} and {len(rs)}"
                    )
                group_size = len(rs)
        else:
            # 如果是列表，则进行类型转换并获取rankset
            rankset = cast_listint(group)
            group_size = len(rankset)
    elif isinstance(group, dist.ProcessGroup):
        # 如果是进程组对象，获取其包含的rankset和组大小，并设置标签
        rankset = dist.get_process_group_ranks(group)
        group_size = len(rankset)
        tag = tag or c10d._get_group_tag(group)
    elif isinstance(group, DeviceMesh):
        # 如果是设备网格对象，检查维度是否为1，并从中获取标签、rankset和组大小
        assert (
            group.ndim == 1
        ), "Only 1D mesh is supported, pass in (DeviceMesh, int) together if mesh > 1D"
        # TODO: 应在整个网格上运行集体操作，而不是在维度0上运行
        tag, rankset, _ = group._dim_group_infos[0]
        group_size = len(rankset)
    elif isinstance(group, tuple):
        # 如果是元组，检查其结构是否符合 (DeviceMesh, int)，并从中获取标签、rankset和组大小
        if (
            len(group) == 2
            and isinstance(group[0], DeviceMesh)
            and isinstance(group[1], int)
        ):
            dmesh = group[0]
            dim = group[1]
            tag, rankset, _ = dmesh._dim_group_infos[dim]
            group_size = len(rankset)
        else:
            raise ValueError("Invalid tuple for group must be (DeviceMesh, int)")
    else:
        # 如果是其他类型，则抛出异常
        raise ValueError(
            "Invalid type for group, must be one of List, Processgroup, DeviceMesh or (DeviceMesh, int)."
        )

    return (tag, rankset, group_size)


def _resolve_group_name(group: RANK_TYPES, tag: str = "") -> str:
    """
    _resolve_group_name resolves the name of the group based on the given RANK_TYPES type.

    It returns a string representing the resolved group name.
    """
    # 根据给定的 group 返回相应的组名。
    """
    # `tag` 将被弃用。详细信息请参见：
    # https://github.com/pytorch/pytorch/issues/93173#issuecomment-1907095208
    # 检查 group 的类型，返回相应的组名
    if isinstance(group, dist.ProcessGroup):
        return group.group_name
    # 如果 group 是字符串类型，则直接返回
    elif isinstance(group, str):
        return group
    # 如果 group 是 DeviceMesh 类型，则检查维度是否为 1，并返回相关信息
    elif isinstance(group, DeviceMesh):
        assert (
            group.ndim == 1
        ), "Only 1D mesh is supported, pass in (DeviceMesh, int) together if mesh > 1D"
        return group._dim_group_infos[0][2]
    # 如果 group 是元组类型，且符合 (DeviceMesh, int) 的形式，则返回对应维度的信息
    elif isinstance(group, tuple):
        if (
            len(group) == 2
            and isinstance(group[0], DeviceMesh)
            and isinstance(group[1], int)
        ):
            dmesh = group[0]
            dim = group[1]
            return dmesh._dim_group_infos[dim][2]
        else:
            raise ValueError("Invalid tuple for group must be (DeviceMesh, int)")
    # 如果 group 是列表类型，则进行相关警告，并返回由 ranks 和 tag 决定的组名
    elif isinstance(group, list):
        if not is_torchdynamo_compiling():
            warnings.warn(
                "The combination of ranks + tag as process group "
                "identifier has been deprecated. Please switch to "
                "using ProcessGroup, DeviceMesh, or group name instead.",
                FutureWarning,
                stacklevel=3,
            )
        return c10d._resolve_group_name_by_ranks_and_tag(cast(List[int], group), tag)
    # 如果 group 类型不被支持，则抛出 ValueError 异常
    else:
        raise ValueError(f"Unsupported group type: {type(group)}, {group}")
class _FromTorchTensor(torch.autograd.Function):
    """
    _FromTorchTensor allows autograd to propagate from a normal Tensor to an
    AsyncCollectiveTensor.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,  # pyre-ignore[2]: Parameter must be annotated.
        input: torch.Tensor,
    ) -> torch.Tensor:
        # 正向传播函数，将输入的普通张量转换为 AsyncCollectiveTensor
        return _maybe_wrap_tensor(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # 反向传播函数，直接返回梯度输出
        return grad_output


def _are_we_tracing() -> bool:
    if is_torchdynamo_compiling():
        # 如果正在编译 TorchDynamo，则返回 True
        return True
    # 如果功能化被打开，我们几乎肯定在编译/追踪中
    # (特别是 AOTAutograd 以功能化打开一次追踪模型，但关闭了代理追踪，因此这是我们检测它的方式)。
    if (
        torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL)
        is not None
    ):
        return True
    mode = get_innermost_proxy_mode()
    if mode is None:
        # 如果模式为 None，则返回 False
        return False
    # 返回模式中的追踪器是否为 None
    return mode.tracer is not None


def _maybe_wrap_tensor(self) -> torch.Tensor:
    if _are_we_tracing():
        # 如果正在追踪，则等待张量并返回
        return wait_tensor(self)
    # 否则，创建 AsyncCollectiveTensor 对象并返回
    res = AsyncCollectiveTensor(self)
    return cast(torch.Tensor, res)


def _all_gather_into_tensor_coalesced_meta(self, tag, rankset, group_size):
    def mk_out_tensor(shard):
        out_size = list(shard.size())
        out_size[0] *= group_size
        # 创建一个空的输出张量，其大小根据分片的大小和组大小来确定
        out_tensor = shard.new_empty(out_size)
        return out_tensor

    # 对 self 中的每个张量分片，调用 mk_out_tensor 创建对应的输出张量，并返回列表
    return [mk_out_tensor(t) for t in self]


# 现在我们注册元内核来处理追踪
def _broadcast_meta(self, *args):
    # 返回与输入张量相同大小的空张量
    return torch.empty_like(self)


def _all_reduce_meta(self, *args):
    # 返回与输入张量相同大小的空张量
    return torch.empty_like(self)


def _wait_tensor_meta(self, *args):
    # 返回与输入张量相同大小的空张量
    return torch.empty_like(self)


def _all_gather_into_tensor_meta(shard, tag, rankset, group_size):
    # 创建一个空的输出张量，其大小根据分片的大小和组大小来确定
    out_size = list(shard.size())
    out_size[0] *= group_size
    return shard.new_empty(out_size)


def _reduce_scatter_tensor_meta(input, reduce_op, tag, rankset, group_size):
    # 创建一个空的输出张量，其大小根据输入张量的大小和组大小来确定
    out_size = list(input.size())
    out_size[0] //= group_size
    return input.new_empty(out_size)


def _all_reduce_coalesced_meta(self, *args):
    # 返回一个列表，其中每个元素是与输入张量相同大小的空张量
    return [torch.empty_like(t) for t in self]


def _all_reduce__meta(inp, *args):
    # 直接返回输入张量
    return inp


def _broadcast__meta(inp, *args):
    # 直接返回输入张量
    return inp


def _all_reduce_coalesced__meta(inputs, *args):
    # 返回输入列表中每个张量的副本
    return inputs


def _reduce_scatter_tensor_coalesced_meta(inputs, reduceOp, tag, rankset, group_size):
    def mk_out_tensor(input):
        # 创建一个空的输出张量，其大小根据输入张量的大小和组大小来确定
        out_size = list(input.size())
        out_size[0] //= group_size
        return input.new_empty(out_size)

    # 对输入列表中的每个张量调用 mk_out_tensor，并返回列表
    return [mk_out_tensor(t) for t in inputs]


# 注意：我们经常说 all_to_all 具有动态输出大小，但这不完全正确：
# 实际上，通常是您提前手动通信输出分割大小（这是动态的），
# 如果没有在 torch 部署模式下运行
if not torch._running_with_deploy():
    # 创建一个名为 "_c10d_functional"，类型为 "IMPL" 的 Library 实例，用于一些功能的实现
    lib_impl = torch.library.Library("_c10d_functional", "IMPL")
    
    # 将各种功能函数注册到 Library 实例中，用于在分布式环境下使用
    lib_impl.impl("all_reduce", _all_reduce_meta, "Meta")
    lib_impl.impl("all_reduce_", _all_reduce__meta, "Meta")
    lib_impl.impl("all_reduce_coalesced", _all_reduce_coalesced_meta, "Meta")
    lib_impl.impl("all_reduce_coalesced_", _all_reduce_coalesced__meta, "Meta")
    lib_impl.impl("wait_tensor", _wait_tensor_meta, "Meta")
    lib_impl.impl("all_gather_into_tensor_out", _all_gather_into_tensor_out_native_meta, "Meta")
    lib_impl.impl("all_gather_into_tensor", _all_gather_into_tensor_native_meta, "Meta")
    lib_impl.impl("all_gather_into_tensor_coalesced", _all_gather_into_tensor_coalesced_native_meta, "Meta")
    lib_impl.impl("reduce_scatter_tensor", _reduce_scatter_tensor_native_meta, "Meta")
    lib_impl.impl("reduce_scatter_tensor_coalesced", _reduce_scatter_tensor_coalesced_native_meta, "Meta")
    lib_impl.impl("all_to_all_single", _all_to_all_single_meta, "Meta")
    lib_impl.impl("broadcast", _broadcast_meta, "Meta")
    lib_impl.impl("broadcast_", _broadcast__meta, "Meta")

    # 为了向后兼容，注册旧版本的操作
    # 在功能集体的 beta 版本中删除这些
    # TODO(yifu): 在功能集体的 beta 版本中移除这些
    # 创建名为 'c10d_functional' 的遗留库对象，用于函数定义 (DEF 模式)
    legacy_lib = torch.library.Library("c10d_functional", "DEF")
    # 创建名为 'c10d_functional' 的遗留库对象，用于函数实现 (IMPL 模式)
    legacy_lib_impl = torch.library.Library("c10d_functional", "IMPL")

    # 定义一组操作的字符串定义列表
    ops_defs = [
        "broadcast(Tensor self, int src, str tag, int[] ranks, int group_size) -> Tensor",
        "all_reduce(Tensor self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor",
        "all_reduce_coalesced(Tensor[] self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor[]",
        "wait_tensor(Tensor self) -> Tensor",
        "all_gather_into_tensor(Tensor shard, str tag, int[] ranks, int group_size) -> Tensor",
        "all_gather_into_tensor_coalesced(Tensor[] input, str tag, int[] ranks, int group_size) -> Tensor[]",
        "reduce_scatter_tensor(Tensor input, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor",
        "reduce_scatter_tensor_coalesced(Tensor[] inputs, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor[]",
        "all_to_all_single(Tensor input, SymInt[]? output_split_sizes, SymInt[]? input_split_sizes, str tag, int[] ranks, int group_size) -> Tensor",  # noqa: B950
    ]

    # 获取当前模块对象
    my_module = sys.modules[__name__]

    # 遍历操作定义列表
    for op_def in ops_defs:
        # 提取操作的名称，即第一个括号之前的部分
        op_name = op_def[0: op_def.index("(")]
        # 获取对应操作的后端实现函数对象
        backend_impl = getattr(fun_col_impl, f"_{op_name}")
        # 在遗留库中定义操作，使用 pt2_compliant_tag 标签
        legacy_lib.define(op_def, tags=torch.Tag.pt2_compliant_tag)
        # 将操作名称、后端实现函数和自动求导模式注册到实现库中
        legacy_lib_impl.impl(op_name, backend_impl, "CompositeImplicitAutograd")
else:
    warnings.warn(
        "PyTorch Distributed functional collectives do not work with torch::deploy."
    )



# 否则情况下，给出警告信息，指出 PyTorch 分布式功能集合与 torch::deploy 不兼容。
else:
    warnings.warn(
        "PyTorch Distributed functional collectives do not work with torch::deploy."
    )




"""
Dynamo Remappings allow seamless translation from non-functional collectives of supportable form into
functional collective calls followed by inplace copy ops, allowing them to be traced into a functional graph.

We implement this by writing a decomposition and teaching dynamo how to associate it to a corresponding op via
the mapping dict below.

These schemas intentionally match torch.distributed.distributed_c10d.* ops that we are trying to remap from
"""



# Dynamo Remappings 允许将支持形式上的非功能性集合转换为功能性集合调用，然后是就地复制操作，
# 使它们可以被追踪到功能图中。
#
# 通过编写分解并教导 Dynamo 如何通过下面的映射字典将其关联到相应的操作来实现这一点。
#
# 这些模式有意匹配我们试图从中重新映射的 torch.distributed.distributed_c10d.* 操作。




def all_gather_tensor_inplace(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group,  # TODO add a type,
    async_op: bool = False,
    tag: str = "",
    gather_dim: int = 0,
):



# 执行就地操作的全局聚合张量函数，将输入张量的数据在给定维度上全局聚合到输出张量中。
# 参数：
#   - output_tensor: 输出张量，聚合结果将会存储在这里
#   - input_tensor: 输入张量，要进行聚合的数据来源
#   - group: 分组对象，表示进行聚合的组
#   - async_op: 是否异步操作，默认为 False，不支持异步操作
#   - tag: 操作标签，用于标识聚合操作
#   - gather_dim: 聚合操作的维度，默认为 0




    assert (
        not async_op
    ), "Can't remap async version of inplace op to functional collective"



# 断言不支持异步操作，因为无法将异步版本的就地操作映射到功能性集合操作中。
    assert (
        not async_op
    ), "Can't remap async version of inplace op to functional collective"




    group = group or dist.group.WORLD
    assert group is not None



# 确保分组对象存在，如果未提供则默认使用 dist.group.WORLD。
    group = group or dist.group.WORLD
    assert group is not None




    return output_tensor.copy_(all_gather_tensor(input_tensor, gather_dim, group, tag))



# 返回在全局聚合操作中输出张量的就地拷贝版本，其值由输入张量在给定维度上的全局聚合得出。
    return output_tensor.copy_(all_gather_tensor(input_tensor, gather_dim, group, tag))




def reduce_scatter_tensor_inplace(
    output: torch.Tensor,
    input: torch.Tensor,
    op: str = "sum",  # TODO type is actually c10d ReduceOp. is this ok?
    group=None,  # TODO add a type
    async_op: bool = False,
    scatter_dim: int = 0,
    tag: str = "",
):



# 执行就地操作的张量分散约简函数，将输入张量的数据通过指定操作符在给定维度上分散约简到输出张量中。
# 参数：
#   - output: 输出张量，约简的结果将会存储在这里
#   - input: 输入张量，要进行分散约简的数据来源
#   - op: 约简操作符，默认为 "sum"，实际上应该是 c10d ReduceOp 的类型。这样做可以吗？
#   - group: 分组对象，表示进行约简的组
#   - async_op: 是否异步操作，默认为 False，不支持异步操作
#   - scatter_dim: 约简操作的维度，默认为 0
#   - tag: 操作标签，用于标识约简操作




    assert (
        not async_op
    ), "Can't remap async version of inplace op to functional collective"



# 断言不支持异步操作，因为无法将异步版本的就地操作映射到功能性集合操作中。
    assert (
        not async_op
    ), "Can't remap async version of inplace op to functional collective"




    group = group or dist.group.WORLD
    assert group is not None



# 确保分组对象存在，如果未提供则默认使用 dist.group.WORLD。
    group = group or dist.group.WORLD
    assert group is not None




    return output.copy_(reduce_scatter_tensor(input, op, scatter_dim, group, tag))



# 返回在张量分散约简操作中输出张量的就地拷贝版本，其值由输入张量通过指定操作符在给定维度上分散约简得出。
    return output.copy_(reduce_scatter_tensor(input, op, scatter_dim, group, tag))




REDUCE_OP_TO_STR = {
    dist.ReduceOp.SUM: "sum",
    dist.ReduceOp.AVG: "avg",
    dist.ReduceOp.PRODUCT: "product",
    dist.ReduceOp.MIN: "min",
    dist.ReduceOp.MAX: "max",
    dist.ReduceOp.BAND: "band",
    dist.ReduceOp.BOR: "bor",
    dist.ReduceOp.BXOR: "bxor",
}



# 减少操作符到字符串的映射字典，用于将 c10d ReduceOp 转换为易读的字符串形式。
REDUCE_OP_TO_STR = {
    dist.ReduceOp.SUM: "sum",
    dist.ReduceOp.AVG: "avg",
    dist.ReduceOp.PRODUCT: "product",
    dist.ReduceOp.MIN: "min",
    dist.ReduceOp.MAX: "max",
    dist.ReduceOp.BAND: "band",
    dist.ReduceOp.BOR: "bor",
    dist.ReduceOp.BXOR: "bxor",
}




def all_reduce_inplace(
    tensor: torch.Tensor,
    op: str = "sum",
    group=None,
    async_op: bool = False,
    tag: str = "",
):



# 执行就地操作的全局张量全部约简函数，将输入张量的数据通过指定操作符在全局范围内全部约简到输出张量中。
# 参数：
#   - tensor: 输入和输出张量，约简的结果将会存储在这里
#   - op: 约简操作符，默认为 "sum"
#   - group: 分组对象，表示进行约简的组
#   - async_op: 是否异步操作，默认为 False，不支持异步操作
#   - tag: 操作标签，用于标识约简操作




    assert (
        not async_op
    ), "Can't remap async version of inplace op to functional collective"



# 断言不支持异步操作，因为无法将异步版本的就地操作映射到功能性集合操作中。
    assert (
        not async_op
    ), "Can't remap async version of inplace op to functional collective"




    group = group or dist.group.WORLD
    assert group is not None



# 确保分组对象存在，如果未提供则默认使用 dist.group.WORLD。
    group = group or dist.group.WORLD
    assert group is not None




    return tensor.copy_(all_reduce(tensor, op, group, tag))



# 返回在全局全部约简操作中张量的就地拷贝版本，其值由输入张量通过指定操作符在全局范围内全部约简得出。
    return tensor.copy_(all_reduce(tensor, op, group, tag))




def all_to_all
    # 确保异步操作的版本不能映射到功能集合操作中
    ), "Can't remap async version of inplace op to functional collective"
    
    # 断言所有张量的第一个维度大小与第一个张量相同，用于检查变量大小的重映射是否支持
    assert all(
        t.size(0) == tensor.size(0) for t in tensor_list
    ), "Remapping variable size all_gather is not yet supported"

    # 如果未指定分组，将分组设为默认的全局分组
    group = group or dist.group.WORLD
    # 断言分组不为 None
    assert group is not None

    # 调用 all_gather_tensor 函数，收集所有节点上的张量数据到 output 中
    output = all_gather_tensor(tensor, 0, group, tag)

    # 使用 aten.slice 替代 aten.split，因为后者在 SymInt 的情况下会将 tensor.shape(0) 不必要地固定下来。
    output_splits = []
    offset = 0
    # 遍历 tensor_list 中的每个张量 t，将 output 中对应位置的数据切片存入 output_splits
    for t in tensor_list:
        output_splits.append(output[offset : offset + t.size(0)])
        offset += t.size(0)
    
    # 将 output_splits 中的切片数据复制回 tensor_list 中的每个张量 dst
    for dst, src in zip(tensor_list, output_splits):
        dst.copy_(src)
    
    # 返回更新后的 tensor_list
    return tensor_list
# 从 torch.distributed.distributed_c10d 中导入所需的函数和模块
from torch.distributed.distributed_c10d import (
    _all_gather_base as legacy_all_gather_base,          # 导入 _all_gather_base 并重命名为 legacy_all_gather_base
    _reduce_scatter_base as legacy_reduce_scatter_base,  # 导入 _reduce_scatter_base 并重命名为 legacy_reduce_scatter_base
    all_gather as legacy_all_gather,                    # 导入 all_gather 并重命名为 legacy_all_gather
    all_gather_into_tensor as legacy_allgather,         # 导入 all_gather_into_tensor 并重命名为 legacy_allgather
    all_reduce as legacy_allreduce,                     # 导入 all_reduce 并重命名为 legacy_allreduce
    all_to_all_single as legacy_all_to_all_single,      # 导入 all_to_all_single 并重命名为 legacy_all_to_all_single
    reduce_scatter_tensor as legacy_reducescatter,      # 导入 reduce_scatter_tensor 并重命名为 legacy_reducescatter
)

# 这个字典应该包含 dynamo 可以重新映射的函数集合。
# 这些集合中的函数应该以与它们映射的函数相同的方式接受参数和关键字参数。
traceable_collective_remaps = {
    legacy_allgather: all_gather_tensor_inplace,        # 将 legacy_allgather 映射到 all_gather_tensor_inplace
    legacy_reducescatter: reduce_scatter_tensor_inplace, # 将 legacy_reducescatter 映射到 reduce_scatter_tensor_inplace
    legacy_allreduce: all_reduce_inplace,               # 将 legacy_allreduce 映射到 all_reduce_inplace
    legacy_all_to_all_single: all_to_all_inplace,       # 将 legacy_all_to_all_single 映射到 all_to_all_inplace
    legacy_all_gather: all_gather_inplace,              # 将 legacy_all_gather 映射到 all_gather_inplace
    legacy_reduce_scatter_base: reduce_scatter_tensor_inplace,  # 将 legacy_reduce_scatter_base 映射到 reduce_scatter_tensor_inplace
    legacy_all_gather_base: all_gather_tensor_inplace,  # 将 legacy_all_gather_base 映射到 all_gather_tensor_inplace
}
```