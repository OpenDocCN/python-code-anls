# `.\pytorch\torch\nn\parallel\distributed.py`

```py
# mypy: allow-untyped-defs
# 引入必要的模块和库
import copy  # 复制对象的模块
import functools  # 提供高阶函数的工具，如partial函数
import inspect  # 提供检查代码的工具，如获取对象信息
import itertools  # 提供创建和操作迭代器的工具
import logging  # 提供日志记录功能
import os  # 提供与操作系统交互的功能
import sys  # 提供与Python解释器交互的功能
import warnings  # 提供警告控制功能
import weakref  # 提供弱引用对象的工具
from collections import defaultdict, deque  # 提供额外的集合类型：defaultdict和deque
from contextlib import contextmanager  # 提供上下文管理工具
from dataclasses import dataclass, fields, is_dataclass  # 提供数据类的支持
from enum import auto, Enum  # 提供枚举类型的支持
from typing import Any, Callable, List, Optional, Tuple, Type, TYPE_CHECKING  # 提供类型提示支持

import torch  # 导入PyTorch库
import torch.distributed as dist  # 提供分布式支持
from torch._utils import _get_device_index  # 提供内部工具函数：获取设备索引
from torch.autograd import Function, Variable  # 提供自动求导支持
from torch.distributed.algorithms.join import Join, Joinable, JoinHook  # 提供分布式算法支持
from torch.utils._pytree import tree_flatten, tree_unflatten  # 提供内部工具函数：树结构展开与重建

from ..modules import Module  # 导入自定义模块：模型
from .scatter_gather import gather, scatter_kwargs  # 导入自定义模块：数据分发与聚集

RPC_AVAILABLE = False  # 初始化RPC可用性标志为False
if dist.is_available():  # 如果分布式模块可用
    from torch.distributed.distributed_c10d import (  # 导入分布式C10d模块的相关函数
        _get_default_group,  # 获取默认分组
        _rank_not_in_group,  # 检查排名是否不在分组中
        ReduceOp,  # 用于指定分布式约简操作的枚举
    )
    from torch.distributed.utils import (  # 导入分布式工具函数
        _alloc_storage,  # 分配存储
        _cast_forward_inputs,  # 强制转换前向输入
        _free_storage,  # 释放存储
        _sync_module_states,  # 同步模块状态
        _to_kwargs,  # 转换为关键字参数
        _verify_param_shape_across_processes,  # 验证跨进程的参数形状
    )
if dist.rpc.is_available():  # 如果RPC模块可用
    RPC_AVAILABLE = True  # 设置RPC可用性标志为True
    from torch.distributed.rpc import RRef  # 导入RPC引用对象

if TYPE_CHECKING:  # 如果是类型检查模式
    from torch.utils.hooks import RemovableHandle  # 导入可移除句柄对象

__all__ = ["DistributedDataParallel"]  # 指定导出的公共符号列表

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


@dataclass
class _MixedPrecision:
    """
    This configures DDP-native mixed precision training.

    Attributes:
        param_dtype (torch.dtype): This specifies the dtype for model
            parameters, inputs (when ``cast_forward_inputs`` is set to
            ``True``), and therefore the dtype for computation.
            However, outside the forward and backward passes, parameters are in
            full precision. Model checkpointing always happens in full
            precision.
        reduce_dtype (torch.dtype): This specifies the dtype for gradient
            reduction, which is permitted to differ from ``param_dtype``.
        buffer_dtype (torch.dtype): This specifies the dtype for buffers.

    .. note:: This API is experimental and subject to change.

    .. note:: Only floating point tensors are cast to their specified dtypes.

    .. note:: ``state_dict`` checkpoints parameters and buffers in full
        precision.

    .. note:: Each low precision dtype must be specified explicitly. For
        example, ``_MixedPrecision(reduce_dtype=torch.float16)`` only specifies
        the reduction dtype to be low precision, and DDP will not cast
        parameters or buffers.

    .. note:: If a ``reduce_dtype`` is not specified, then gradient reduction
        happens in ``param_dtype`` if specified or the original parameter dtype
        otherwise. For example, ``_MixedPrecision(param_dtype=torch.float16)``
        would result in communication occurring in fp16.
    """

    param_dtype: Optional[torch.dtype] = None  # 模型参数的数据类型，默认为None
    reduce_dtype: Optional[torch.dtype] = None  # 梯度约简的数据类型，默认为None
    # 定义一个可选的 torch 数据类型变量，用于存储缓冲区数据类型，默认为 None
    buffer_dtype: Optional[torch.dtype] = None
    
    # TODO (rohan-varma): 保持低精度梯度的设置：bool = False
    # TODO (rohan-varma): 添加允许用户在完整精度下运行 batchnorm 和 layernorm 的 API。
    # 对于分布式数据并行（DDP），可以通过不对 BN 和 LN 单元进行参数类型转换来实现。
# 将混合精度配置应用于根模块中的所有缓冲区
def _cast_buffers(mixed_precision_config, root_module):
    # 遍历根模块中的所有缓冲区
    for buf in root_module.buffers():
        # 如果缓冲区被 DDP 忽略且已标记为被忽略，则跳过
        if hasattr(buf, "_ddp_ignored") and buf._ddp_ignored:
            continue

        # 将缓冲区数据类型转换为指定的混合精度缓冲区数据类型
        buf.data = buf.to(dtype=mixed_precision_config.buffer_dtype)


# 设置混合精度参数并为其分配和释放存储空间
def _setup_mixed_precision_params(mixed_precision_config, root_module):
    # 遍历根模块中的所有参数
    for param in root_module.parameters():
        # 不为 DDP 忽略的参数设置混合精度
        if hasattr(param, "_ddp_ignored") and param._ddp_ignored:
            continue

        # 如果参数没有被标记为混合精度参数，则创建相应的混合精度参数
        if not hasattr(param, "_mp_param"):
            param._mp_param = torch.zeros_like(
                param,
                device=param.device,
                dtype=mixed_precision_config.param_dtype,
                requires_grad=param.requires_grad,
            )
            # 释放混合精度参数的存储空间
            _free_storage(param._mp_param)
            # _fp_param 将指向完整精度的参数，以便在前向/后向过程中可以切换回来
            param._fp_param = param.data


# 将输出展平为张量列表，并标识是否输出为 RRef
def _tree_flatten_with_rref(output):
    output_is_rref = RPC_AVAILABLE and isinstance(output, RRef)
    if output_is_rref:
        output_tensor_list, treespec = tree_flatten(output.local_value())
    else:
        output_tensor_list, treespec = tree_flatten(output)
    # 需要返回展平的张量列表、重组它们的 spec，以及输出是否是 RRef 类型
    return output_tensor_list, treespec, output_is_rref


# 根据 spec 重组输出，如有必要重新构建为 RRef
def _tree_unflatten_with_rref(output, treespec, output_is_rref):
    # 根据 spec 重组输出
    output = tree_unflatten(output, treespec)
    # 如果输出为 RRef 类型，则重新构建为 RRef
    if output_is_rref:
        output = RRef(output)
    return output


# 递归查找对象中包含的所有张量
def _find_tensors(obj):
    # 如果支持 RPC 并且对象是 RRef 类型
    if RPC_AVAILABLE and isinstance(obj, RRef):
        # 如果当前节点是 RRef 的所有者，则解开它并尝试查找张量
        # TODO: 扩展到远程 RRefs 的情况
        if obj.is_owner():
            return _find_tensors(obj.local_value())
    # 如果对象是张量，则返回它
    if isinstance(obj, torch.Tensor):
        return [obj]
    # 如果对象是列表或元组，则递归查找每个元素中的张量
    if isinstance(obj, (list, tuple)):
        return itertools.chain.from_iterable(map(_find_tensors, obj))
    # 如果对象是字典，则递归查找每个值中的张量
    if isinstance(obj, dict):
        return itertools.chain.from_iterable(map(_find_tensors, obj.values()))
    # 如果对象是数据类，则递归查找每个字段中的张量
    if is_dataclass(obj):
        return itertools.chain.from_iterable(
            map(_find_tensors, (getattr(obj, f.name) for f in fields(obj)))
        )

    # 对于其他类型的对象，默认返回空迭代器
    return []
    # 定义包含相关环境变量名称的列表
    relevant_env_vars = [
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_PORT",
        "MASTER_ADDR",
        "CUDA_VISIBLE_DEVICES",
        "GLOO_SOCKET_IFNAME",
        "GLOO_DEVICE_TRANSPORT",
        "NCCL_SOCKET_IFNAME",
        "TORCH_NCCL_BLOCKING_WAIT",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "NCCL_IB_DISABLE",
        # 更多的 NCCL 环境变量:
        "NCCL_P2P_DISABLE",
        "NCCL_P2P_LEVEL",
        "NCCL_SHM_DISABLE",
        "NCCL_SOCKET_NTHREADS",
        "NCCL_NSOCKS_PERTHREAD",
        "NCCL_BUFFSIZE",
        "NCCL_NTHREADS",
        "NCCL_RINGS",
        "NCCL_MAX_NCHANNELS",
        "NCCL_MIN_NCHANNELS",
        "NCCL_CHECKS_DISABLE",
        "NCCL_CHECK_POINTERS",
        "NCCL_LAUNCH_MODE",
        "NCCL_IB_HCA",
        "NCCL_IB_TIMEOUT",
        "NCCL_IB_RETRY_CNT",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_SL",
        "NCCL_IB_TC",
        "NCCL_IB_AR_THRESHOLD",
        "NCCL_IB_CUDA_SUPPORT",
        "NCCL_NET_GDR_LEVEL",
        "NCCL_NET_GDR_READ",
        "NCCL_SINGLE_RING_THRESHOLD",
        "NCCL_LL_THRESHOLD",
        "NCCL_TREE_THRESHOLD",
        "NCCL_ALGO",
        "NCCL_PROTO",
        "NCCL_IGNORE_CPU_AFFINITY",
        "NCCL_DEBUG_FILE",
        "NCCL_COLLNET_ENABLE",
        "NCCL_TOPO_FILE",
        "NCCL_TOPO_DUMP_FILE",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    ]
    
    # 初始化格式化输出字符串
    formatted_output = ""
    
    # 遍历每个环境变量名称
    for var in relevant_env_vars:
        # 如果环境变量存在，则获取其值；否则使用 "N/A"
        value = os.environ[var] if var in os.environ else "N/A"
        # 将环境变量名称和对应值格式化后添加到输出字符串
        formatted_output += f"env:{var}={value}\n"
    
    # 打印格式化后的输出字符串
    print(formatted_output)
# 定义枚举类_BufferCommHookLocation，包含两个枚举值：PRE_FORWARD和POST_FORWARD
class _BufferCommHookLocation(Enum):
    PRE_FORWARD = auto()
    POST_FORWARD = auto()


# 使用dataclass装饰器定义_BufferCommHook类，用于存储缓冲通信钩子相关的信息
@dataclass
class _BufferCommHook:
    # 缓冲通信钩子函数，可以是任意可调用对象
    buffer_comm_hook: Callable
    # 缓冲通信钩子的状态，可以是任意类型的数据
    buffer_comm_hook_state: Any
    # 缓冲通信钩子的位置，使用_BufferCommHookLocation枚举来表示
    buffer_comm_hook_location: _BufferCommHookLocation


# 定义_DDPSink类，继承自Function，用于在反向传播开始时添加DDPSink来执行各种函数
class _DDPSink(Function):
    @staticmethod
    def forward(ctx, ddp_weakref, *inputs):
        # 禁用梯度填充，确保None梯度保持为None，不会被填充为0
        ctx.set_materialize_grads(False)
        ctx.ddp_weakref = ddp_weakref
        ret = inputs
        # 如果存在DDPSink的克隆，将输入张量克隆出来，否则直接返回原输入
        if ddp_weakref()._ddp_sink_clone:
            ret = tuple(
                inp.clone() if isinstance(inp, torch.Tensor) else inp for inp in inputs
            )
        return ret

    @staticmethod
    def backward(ctx, *grad_outputs):
        # 在静态图训练的第一次迭代时，延迟排队所有约简操作
        ddp_weakref = ctx.ddp_weakref()
        reducer = ddp_weakref.reducer
        static_graph = ddp_weakref.static_graph
        delay_ar_enqueued = (
            static_graph and ddp_weakref._static_graph_delay_allreduce_enqueued
        )
        if static_graph and not delay_ar_enqueued:
            Variable._execution_engine.queue_callback(  # type: ignore[call-arg,misc]
                reducer._delay_all_reduce
            )
            ddp_weakref._static_graph_delay_allreduce_enqueued = True

        # 返回梯度的反向传播，第一个梯度为None，其余梯度与输出梯度相同
        return (None, *grad_outputs)


# 定义_DDPJoinHook类，继承自JoinHook类，用于在DDP执行时执行特定的连接钩子
class _DDPJoinHook(JoinHook):
    def __init__(self, ddp, divide_by_initial_world_size):
        """为内部使用设置配置变量。"""
        assert isinstance(ddp, DistributedDataParallel), (
            "DDP join hook requires passing in a DistributedDataParallel "
            "instance as the state"
        )
        assert ddp.logger is not None
        # 设置DDP实例的不均匀输入连接标志
        ddp.logger._set_uneven_input_join()
        self.ddp = ddp
        # 设置是否按初始世界大小划分
        self.ddp._divide_by_initial_world_size = divide_by_initial_world_size
        super().__init__()
    # 主钩子函数，用于在前向和后向传播中替代DDP集体通信操作
    def main_hook(self):
        """Shadow the DDP collective communication operations in the forward and backward passes."""
        # 获取当前对象的DDP模块
        ddp = self.ddp
        # 重新构建桶，在训练周期内仅重建一次
        ddp.reducer._rebuild_buckets()

        # 如果在前向传播中需要同步模块缓冲区，则安排一个广播
        # TODO: 使DDP不均匀输入上下文管理器支持缓冲区通信钩子
        # （https://github.com/pytorch/pytorch/issues/65436）
        ddp._check_and_sync_module_buffers()

        # 检查是否需要在后向传播中同步
        should_sync_backwards = ddp._check_global_requires_backward_grad_sync(
            is_joined_rank=True
        )
        # 如果当前迭代不需要同步梯度，则在下一次迭代中禁用前向参数同步
        # 因此相应地设置`require_forward_param_sync`
        ddp.require_forward_param_sync = should_sync_backwards
        if not should_sync_backwards:
            return

        # 安排每个梯度桶的一个allreduce，以匹配后向传播的allreduce
        ddp._match_all_reduce_for_bwd_pass()

        # 检查是否需要allreduce本地未使用的参数
        if ddp.find_unused_parameters:
            ddp._match_unused_params_allreduce()

        # 重建参数仅在训练周期内推送一次
        ddp.reducer._push_all_rebuilt_params()

    # 后钩子函数，确保最终模型在所有进程中一致
    def post_hook(self, is_last_joiner: bool):
        """Sync the final model to ensure that the model is the same across all processes."""
        # 同步最终模型以确保在所有进程中模型一致
        self.ddp._sync_final_model(is_last_joiner)
# 定义一个类，实现基于 torch.distributed 的模块级分布式数据并行处理
class DistributedDataParallel(Module, Joinable):
    r"""Implement distributed data parallelism based on ``torch.distributed`` at module level.

    This container provides data parallelism by synchronizing gradients
    across each model replica. The devices to synchronize across are
    specified by the input ``process_group``, which is the entire world
    by default. Note that ``DistributedDataParallel`` does not chunk or
    otherwise shard the input across participating GPUs; the user is
    responsible for defining how to do so, for example through the use
    of a :class:`DistributedSampler`.

    See also: :ref:`distributed-basics` and :ref:`cuda-nn-ddp-instead`.
    The same constraints on input as in :class:`torch.nn.DataParallel` apply.

    Creation of this class requires that ``torch.distributed`` to be already
    initialized, by calling :func:`torch.distributed.init_process_group`.

    ``DistributedDataParallel`` is proven to be significantly faster than
    :class:`torch.nn.DataParallel` for single-node multi-GPU data
    parallel training.

    To use ``DistributedDataParallel`` on a host with N GPUs, you should spawn
    up ``N`` processes, ensuring that each process exclusively works on a single
    GPU from 0 to N-1. This can be done by either setting
    ``CUDA_VISIBLE_DEVICES`` for every process or by calling:

        >>> # xdoctest: +SKIP("undefined variables")
        >>> torch.cuda.set_device(i)

    where i is from 0 to N-1. In each process, you should refer the following
    to construct this module:

        >>> # xdoctest: +SKIP("undefined variables")
        >>> torch.distributed.init_process_group(
        >>>     backend='nccl', world_size=N, init_method='...'
        >>> )
        >>> model = DistributedDataParallel(model, device_ids=[i], output_device=i)

    In order to spawn up multiple processes per node, you can use either
    ``torch.distributed.launch`` or ``torch.multiprocessing.spawn``.

    .. note::
        Please refer to `PyTorch Distributed Overview <https://pytorch.org/tutorials/beginner/dist_overview.html>`__
        for a brief introduction to all features related to distributed training.

    .. note::
        ``DistributedDataParallel`` can be used in conjunction with
        :class:`torch.distributed.optim.ZeroRedundancyOptimizer` to reduce
        per-rank optimizer states memory footprint. Please refer to
        `ZeroRedundancyOptimizer recipe <https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html>`__
        for more details.

    .. note:: ``nccl`` backend is currently the fastest and highly recommended
        backend when using GPUs. This applies to both single-node and
        multi-node distributed training.
    # 这个模块也支持混合精度分布式训练。
    # 这意味着你的模型可以包含不同类型的参数，例如混合的 `fp16` 和 `fp32`，对这些混合类型参数的梯度归约会正常工作。
    
    # 如果你在一个进程上使用 `torch.save` 来保存模块的检查点，
    # 并在其他进程上使用 `torch.load` 来恢复它，请确保每个进程都正确配置了 `map_location`。
    # 如果没有正确配置 `map_location`，`torch.load` 可能会将模块恢复到与保存时不同的设备上。
    
    # 当模型在 `M` 个节点上使用 `batch=N` 进行训练时，
    # 如果损失在批次中的实例之间是求和（而不是通常的求平均），则相比于在单节点上使用 `batch=M*N` 进行训练时，
    # 梯度会小 `M` 倍（因为不同节点之间的梯度被平均了）。
    # 当你想要获得与本地训练对等的数学上等效的训练过程时，你应该考虑这一点。
    # 但在大多数情况下，你可以将使用 DistributedDataParallel 包装的模型、使用 DataParallel 包装的模型和单 GPU 上的普通模型视为相同的（例如，对于等效的批次大小使用相同的学习率）。
    
    # 参数在进程之间永远不会广播。
    # 模块对梯度执行全局归约步骤，并假定这些梯度将在所有进程中以相同的方式被优化器修改。
    # 缓冲区（例如 BatchNorm 的统计数据）从排名为 0 的进程中的模块广播到系统中的所有其他副本中，每次迭代都会进行广播。
    # 使用 DistributedDataParallel 与分布式 RPC 框架时，建议使用 torch.distributed.autograd.backward 计算梯度，
    # 并使用 torch.distributed.optim.DistributedOptimizer 优化参数。

    # 导入必要的库和模块
    >>> import torch.distributed.autograd as dist_autograd
    >>> from torch.nn.parallel import DistributedDataParallel as DDP
    >>> import torch
    >>> from torch import optim
    >>> from torch.distributed.optim import DistributedOptimizer
    >>> import torch.distributed.rpc as rpc
    >>> from torch.distributed.rpc import RRef

    # 创建两个需要梯度的张量
    >>> t1 = torch.rand((3, 3), requires_grad=True)
    >>> t2 = torch.rand((3, 3), requires_grad=True)

    # 在远程 worker 上执行 torch.add 操作
    >>> rref = rpc.remote("worker1", torch.add, args=(t1, t2))

    # 使用 DistributedDataParallel 包装模型
    >>> ddp_model = DDP(my_model)

    # 设置优化器的参数列表
    >>> optimizer_params = [rref]
    >>> for param in ddp_model.parameters():
    >>>     optimizer_params.append(RRef(param))

    # 创建 DistributedOptimizer 实例，使用 SGD 作为优化器
    >>> dist_optim = DistributedOptimizer(
    >>>     optim.SGD,
    >>>     optimizer_params,
    >>>     lr=0.05,
    >>> )

    # 使用 dist_autograd.context() 进入分布式自动求导上下文
    >>> with dist_autograd.context() as context_id:
    >>>     # 在 DDP 模型上执行前向传播
    >>>     pred = ddp_model(rref.to_here())
    >>>     # 计算损失值
    >>>     loss = loss_func(pred, target)
    >>>     # 执行反向传播
    >>>     dist_autograd.backward(context_id, [loss])
    >>>     # 执行优化步骤
    >>>     dist_optim.step(context_id)

    # DistributedDataParallel 目前对 torch.utils.checkpoint 的梯度检查点功能支持有限。
    # 推荐使用 use_reentrant=False 进行检查点操作，这样 DDP 可以正常工作而无需限制。
    # 如果使用 use_reentrant=True（默认值），则 DDP 在模型中没有未使用的参数并且每层最多只检查点一次时会正常工作。
    # 确保不要向 DDP 传递 find_unused_parameters=True，否则目前不支持多次检查点同一层或检查点模型中存在未使用的参数的情况。

    # 如果要让非 DDP 模型从 DDP 模型的状态字典中加载状态，需要先使用 consume_prefix_in_state_dict_if_present
    # 方法去除状态字典中的 "module." 前缀。
    >>> To let a non-DDP model load a state dict from a DDP model,
    >>> :meth:`~torch.nn.modules.utils.consume_prefix_in_state_dict_if_present`
    >>> needs to be applied to strip the prefix "module." in the DDP state dict before loading.
    # 警告：本模块的构造函数、前向方法以及对输出（或该模块输出函数）的求导是分布式同步点。
    # 如果不同进程可能执行不同的代码，请考虑这一点。
    
    # 警告：该模块假设所有参数在创建模块时都已注册到模型中。后续不应添加或删除参数，也不应更改缓冲区。
    
    # 警告：该模块假设所有分布式进程中模型的参数注册顺序相同。
    # 模块会按照模型参数的相反顺序执行梯度的全局归约（allreduce）操作。
    # 因此，用户需确保每个分布式进程拥有完全相同的模型及参数注册顺序。
    
    # 警告：该模块允许使用非行优先连续（non-rowmajor-contiguous）步幅的参数。
    # 例如，您的模型可能包含一些参数的内存格式为 torch.contiguous_format，
    # 其他参数的格式为 torch.channels_last。
    # 然而，不同进程中对应的参数必须具有相同的步幅。
    
    # 警告：该模块不兼容 torch.autograd.grad 函数。
    # 它只能在参数的 .grad 属性中累积梯度。
    
    # 警告：如果您计划将该模块与使用多个工作进程的 DataLoader 一起使用，
    # 并且使用的是 nccl 或 gloo 后端（使用 Infiniband），
    # 请将多进程启动方法更改为 "forkserver"（仅适用于 Python 3）或 "spawn"。
    # 不幸的是，Gloo（使用 Infiniband）和 NCCL2 不是进程安全的，
    # 如果不更改此设置，可能会遇到死锁问题。
    
    # 警告：在使用 DistributedDataParallel 包装模型后，不应尝试更改模型的参数。
    # 因为在包装模型时，DistributedDataParallel 的构造函数会在模型的所有参数上注册额外的梯度归约函数。
    # 如果后续更改模型的参数，梯度归约函数将不再匹配正确的参数集。
    
    # 警告：使用 DistributedDataParallel 与分布式 RPC 框架的结合是实验性的，并可能会更改。
    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
        >>> net = torch.nn.parallel.DistributedDataParallel(model)
    """

    # 用于追踪当前线程是否在 torchdynamo 目的下的 DDP 前向传播中
    _active_ddp_module: Optional["DistributedDataParallel"] = None

    def __init__(
        self,
        module,
        device_ids=None,
        output_device=None,
        dim=0,
        broadcast_buffers=True,
        process_group=None,
        bucket_cap_mb=None,
        find_unused_parameters=False,
        check_reduction=False,
        gradient_as_bucket_view=False,
        static_graph=False,
        delay_all_reduce_named_params=None,
        param_to_hook_all_reduce=None,
        mixed_precision: Optional[_MixedPrecision] = None,
        device_mesh=None,
    ):
        # 初始化 DistributedDataParallel 对象
        super().__init__(
            module,
            device_ids=device_ids,
            output_device=output_device,
            dim=dim,
            broadcast_buffers=broadcast_buffers,
        )
        # 是否延迟执行所有约简操作的参数
        self.delay_all_reduce_named_params = delay_all_reduce_named_params
        # 用于 Hook 所有约简操作的参数
        self.param_to_hook_all_reduce = param_to_hook_all_reduce

    def _register_accum_grad_hook(self):
        # 导入 torch.distributed._functional_collectives 作为 fcol
        import torch.distributed._functional_collectives as fcol

        # 编译累积梯度 Hook
        def compiled_accum_grad_hook(
            param,
            *,
            param_index: int,
        ):
            # 如果不需要在后向传播中同步梯度，则直接返回
            if not self.require_backward_grad_sync:
                return

            # 如果梯度为 None，则直接返回
            if param.grad is None:
                return

            # 如果有通信 Hook，则依次调用每个 Hook 处理梯度
            if self._comm_hooks:
                for hook, state in self._comm_hooks:
                    hook(state, (param.grad, param))
            else:
                # 计算分布式平均梯度并复制到 param.grad
                gradient = param.grad / self.process_group.size()
                gradient = fcol.all_reduce(gradient, "sum", self.process_group)
                param.grad.copy_(gradient)

        # 遍历所有模块参数，注册后累积梯度 Hook
        for index, param in enumerate(self._module_parameters):
            if not param.requires_grad:
                continue
            self._accum_grad_hooks.append(
                param.register_post_accumulate_grad_hook(
                    functools.partial(
                        compiled_accum_grad_hook,
                        param_index=index,
                    )
                )
            )

    def _delayed_all_reduce_hook(self, grad):
        # 获取进程组的世界大小
        world_size = dist.get_world_size(self.process_group)

        # 将延迟梯度缓冲区除以世界大小
        self._delay_grad_buffer.div_(world_size)  # type: ignore[union-attr]
        # 执行所有约简操作并返回梯度
        _ = dist.all_reduce(
            self._delay_grad_buffer, group=self.process_group, async_op=True
        )
        return grad

    def _register_delay_all_reduce_hook(
        self,
        bucket_cap_mb,
        param_to_hook_all_reduce,
        device_ids,
    ):
        # 1. 创建梯度缓冲区
        #    如果没有指定设备 ID，则将设备设置为 CPU；否则使用第一个设备 ID
        device = torch.device("cpu") if device_ids is None else device_ids[0]
        #    使用所有需要延迟全局归约的参数的总元素数量创建零张量，并指定设备
        self._delay_grad_buffer = torch.zeros(
            sum(p.numel() for p in self._delay_all_reduce_params),
            device=device,
        )

        # 2. 广播参数
        #    将所有需要延迟归约的参数分离并广播到分布式进程组中
        detached_params = [p.detach() for p in self._delay_all_reduce_params]
        dist._broadcast_coalesced(self.process_group, detached_params, bucket_cap_mb, 0)

        # 3. 钩子所有归约到指定参数
        #    将延迟归约钩子注册到指定的参数上
        param_to_hook_all_reduce.register_hook(self._delayed_all_reduce_hook)

        # 4. 为梯度构建张量视图
        offset = 0
        #    遍历所有需要延迟归约的参数
        for param in self._delay_all_reduce_params:
            #    从梯度缓冲区中获取当前参数的梯度视图，并按照参数的形状进行视图构建
            grad_view = self._delay_grad_buffer[offset : (offset + param.numel())].view(
                param.shape
            )
            #    将梯度视图添加到延迟梯度视图列表中
            self._delay_grad_views.append(grad_view)
            offset = offset + param.numel()

        # 5. 检查是否所有需要梯度的参数的归约都被延迟了
        #    遍历模块及其参数，检查是否有需要梯度的参数未被延迟归约
        for module_name, module in self.module.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    full_name = f"{module_name}.{param_name}"
                    #    如果该参数的完整名称不在忽略列表中，则不设置延迟归约所有参数为真
                    if full_name not in self.parameters_to_ignore:
                        #    存在至少一个参数其归约不会被延迟
                        #    在这种情况下，我们不应将 self._delay_all_reduce_all_params 设置为 True
                        return
        #    所有需要梯度的参数的归约都已被延迟
        self._delay_all_reduce_all_params = True
    # 设置用于反向优化器的配置
    def _setup_in_backward_optimizers(self):
        # 检查用户是否使用了 apply_optim_in_backward 方法来重叠优化器步骤和 DDP 后向传播
        # 当前的限制：
        # 1. 目前仅支持 allreduce，不支持自定义通信。
        # 2. 对于由 DDP 管理的参数，在其后向优化器运行后，它们的梯度会被设置为 ``None``。
        #    如果您的用例要求 DDP 参数的梯度在其后向优化器运行后不被设置为 ``None``，请访问
        #    https://github.com/pytorch/pytorch/issues/90052。
        # 注意：我们使用 self._module_parameters 而不是 .parameters()，因为前者排除了被忽略（非由 DDP 管理）的参数。
        if any(hasattr(p, "_in_backward_optimizers") for p in self._module_parameters):
            torch._C._log_api_usage_once("ddp.optimizer_in_backward")
            # 移除 apply_optim_in_backward 注册的钩子，因为 DDP 自定义了优化器如何与后向传播重叠，以实现 allreduce。
            param_to_handle_map = (
                dist.optim.apply_optimizer_in_backward.param_to_optim_hook_handle_map
            )
            for p in self._module_parameters:
                for handle in param_to_handle_map.get(p, []):
                    handle.remove()

            # 需要一个对 DDP 实例的弱引用来运行 all_reduce（来自 reducer）并获取管理的 DDP 参数。
            ddp_weakref = weakref.ref(self)
            # 注意：在函数中导入，否则会导致循环导入。
            from torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks import (
                _apply_optim_in_backward_hook,
            )

            # 注册通信钩子以重叠优化器与后向传播
            self.register_comm_hook(
                ddp_weakref,
                _apply_optim_in_backward_hook(
                    gradient_is_bucket_view=self.gradient_as_bucket_view
                ),
            )

            # 设置 reducer 中的 optimizer_in_backward 方法
            self.reducer._set_optimizer_in_backward()  # type: ignore[attr-defined]

    # 触发 reducer 的自动求导钩子，以对 Reducer 桶中的参数进行 allreduce
    # 注意，这仅在混合精度训练期间使用，因为在低精度参数设置下，构造时安装的 Reducer 钩子不会被调用。
    def _fire_reducer_autograd_hook(self, idx, *unused):
        self.reducer._autograd_hook(idx)  # type: ignore[attr-defined]
    def _root_copy_hook(self, *args: Any, **kwargs: Any) -> None:
        """
        For DDP mixed precision, put low precision copies on separate stream and create events to wait for them.

        When training with DDP mixed precision, this root pre-forward hook kicks
        off low precision copies on a separate stream and creates respective
        events to wait for them.
        """
        # Clear out previous iteration submodule to event. This is because we
        # may have populated some events for modules that didn't end up being
        # used.
        # 初始化一个 defaultdict，用于存储子模块到事件队列的映射关系
        self._submodule_to_event = defaultdict(deque)  # type: ignore[var-annotated]
        
        # 使用指定的 CUDA 流来执行以下操作
        with torch.cuda.stream(self._mp_stream):
            # 遍历模型的所有模块
            for submodule in self.module.modules():
                # 遍历当前模块的参数，不递归进入子模块
                for param in submodule.parameters(recurse=False):
                    # 如果参数被 DDP 忽略，则跳过
                    if hasattr(param, "_ddp_ignored") and param._ddp_ignored:
                        continue
                    # 为低精度参数分配存储空间
                    _alloc_storage(param._mp_param, param.size())
                    
                    # 将参数的数据复制到低精度参数中
                    with torch.no_grad():
                        param._mp_param.copy_(param.data)
                        
                        # TODO: 当 zero_grad(set_to_none=False) 或者梯度累积情况下，
                        # 累积梯度可能是 fp32 类型，导致在 DDP 反向传播时出错，
                        # 因为传入的梯度类型与累积梯度类型不匹配。目前手动将累积梯度转为低精度，
                        # 未来可能采用 FSDP 风格的梯度累积管理，其中累积梯度保存，
                        # .grad 字段设为 None，从而避免此问题。
                        if param.grad is not None:
                            param.grad.data = param.grad.to(
                                self.mixed_precision.param_dtype  # type: ignore[union-attr]
                            )
                    
                    # 将参数的数据更新为低精度参数
                    param.data = param._mp_param
                
                # 创建 CUDA 事件并记录
                copy_event = torch.cuda.Event()
                copy_event.record()
                # 将事件添加到对应的子模块事件队列中
                self._submodule_to_event[submodule].append(copy_event)
    ) -> None:
        """在执行计算之前，等待适当的事件以确保低精度副本已完成。"""
        try:
            # 从队列中取出子模块对应的事件
            event = self._submodule_to_event[module].popleft()
        except IndexError:
            # 拷贝事件已经被等待过了
            return

        # 等待事件完成，使用当前 CUDA 流
        event.wait(stream=torch.cuda.current_stream())
        # 遍历模块的参数，不包括子模块的参数
        for p in module.parameters(recurse=False):
            # 如果参数不需要梯度或者被忽略了，则不注册钩子
            if not p.requires_grad or (hasattr(p, "_ddp_ignored") and p._ddp_ignored):
                continue
            # 在这里注册自动求导钩子，而不是在 DDP 的构造函数中
            # 因为我们在处理低精度参数。通过获取梯度累加器来注册钩子。
            tmp = p.expand_as(p)
            grad_acc = tmp.grad_fn.next_functions[0][0]

            # 注册钩子
            hook = grad_acc.register_hook(
                functools.partial(self._fire_reducer_autograd_hook, p._idx)
            )
            # 设置参数的 DDP 多进程钩子状态
            p._ddp_mp_hook_state = (grad_acc, hook)

    def _log_and_throw(self, err_type, err_msg):
        # 如果有 logger，则设置错误并记录日志
        if self.logger is not None:
            self.logger.set_error_and_log(f"{str(err_type)}: {err_msg}")
        # 抛出异常
        raise err_type(err_msg)

    def _ddp_init_helper(
        self,
        parameters,
        expect_sparse_gradient,
        param_to_name_mapping,
        static_graph,
    def __getattr__(self, name: str) -> Any:
        """将缺失的属性转发给包装的模块。"""
        try:
            # 委托给 nn.Module 的逻辑
            return super().__getattr__(name)
        except AttributeError:
            # 返回包装模块的属性
            return getattr(self.module, name)

    def __getstate__(self):
        # 检查默认的进程组
        self._check_default_group()
        # 复制对象的字典属性
        attrs = copy.copy(self.__dict__)
        # 删除不需要被序列化的属性
        del attrs["process_group"]
        del attrs["reducer"]
        del attrs["logger"]
        return attrs

    def __setstate__(self, state):
        # 如果可序列化，则进程组应为默认组
        self.process_group = _get_default_group()
        # 调用父类的设置状态方法
        super().__setstate__(state)
        # 设置默认值
        self.__dict__.setdefault("require_forward_param_sync", True)
        self.__dict__.setdefault("require_backward_grad_sync", True)
        # 为 reducer 构建参数和期望的稀疏梯度
        parameters, expect_sparse_gradient = self._build_params_for_reducer()
        # 在调试模式下，构建参数索引到参数名的映射
        param_to_name_mapping = self._build_debug_param_to_name_mapping(parameters)
        # 构建 reducer
        self._ddp_init_helper(
            parameters,
            expect_sparse_gradient,
            param_to_name_mapping,
            self.static_graph,
        )
        # 如果是静态图，则设置 reducer 的静态图标志，并确保 logger 不为空
        if self.static_graph:
            self.reducer._set_static_graph()
            assert self.logger is not None
            self.logger._set_static_graph()
    # 构建用于 reducer 的参数列表
    def _build_params_for_reducer(self):
        # 生成包含 (模块名, 参数) 的元组列表，仅包括需要梯度的参数
        modules_and_parameters = [
            (module, parameter)
            for module_name, module in self.module.named_modules()  # 遍历模型的所有命名模块
            for parameter in [
                param
                # 注意我们使用 module.named_parameters 而不是 parameters(module)。
                # parameters(module) 仅在单进程多设备情况下需要，用于通过 _former_parameters 访问复制的参数。
                for param_name, param in module.named_parameters(recurse=False)  # 获取当前模块的命名参数
                if param.requires_grad  # 只选择需要梯度计算的参数
                and f"{module_name}.{param_name}" not in self.parameters_to_ignore  # 排除忽略列表中的参数
            ]
        ]

        # 去除可能在子模块间共享的重复参数
        memo = set()
        modules_and_parameters = [
            (m, p)
            for m, p in modules_and_parameters
            if p not in memo and not memo.add(p)  # 检查并添加到 memo 集合中，以确保参数唯一性
        ]

        # 构建参数列表
        parameters = [parameter for _, parameter in modules_and_parameters]

        # 检查模块是否会产生稀疏梯度
        def produces_sparse_gradient(module):
            if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):  # 检查是否是嵌入层或嵌入包层
                return module.sparse  # 返回是否稀疏
            return False

        # 构建布尔列表，指示每个参数是否预期具有稀疏梯度
        expect_sparse_gradient = [
            produces_sparse_gradient(module) for module, _ in modules_and_parameters
        ]

        # 分配模块缓冲区
        self._assign_modules_buffers()

        # 返回参数列表和稀疏梯度期望列表
        return parameters, expect_sparse_gradient
    # 将 self.module.named_buffers 分配给 self.modules_buffers
    def _assign_modules_buffers(self):
        """
        Assign self.module.named_buffers to self.modules_buffers.
    
        Assigns module buffers to self.modules_buffers which are then used to
        broadcast across ranks when broadcast_buffers=True. Note that this
        must be called every time buffers need to be synced because buffers can
        be reassigned by user module,
        see https://github.com/pytorch/pytorch/issues/63916.
        """
        # 收集模块的缓冲区，过滤掉应忽略的缓冲区
        named_module_buffers = [
            (buffer, buffer_name)
            for buffer_name, buffer in self.module.named_buffers()
            if buffer_name not in self.parameters_to_ignore
        ]
        # 将缓冲区列表赋值给 self.modules_buffers
        self.modules_buffers = [
            buffer for (buffer, buffer_name) in named_module_buffers
        ]
        # 创建字典，表示未被 DDP 忽略的模块缓冲区
        self.named_module_buffers = {
            buffer_name: buffer for (buffer, buffer_name) in named_module_buffers
        }
    
    # 构建调试参数到名称映射
    def _build_debug_param_to_name_mapping(self, parameters):
        # 创建参数到参数索引的映射
        param_to_param_index = {parameters[i]: i for i in range(len(parameters))}
        # 创建参数集合
        param_set = set(parameters)
        # 创建参数索引到完全限定名称的映射字典
        param_index_to_param_fqn = {}
        # 遍历模块及其子模块的名称和模块对象
        for module_name, module in self.module.named_modules():
            # 遍历模块中的参数及其名称
            for param_name, param in module.named_parameters(recurse=False):
                # 构造完全限定名称
                fqn = f"{module_name}.{param_name}"
                # 如果参数不在忽略列表中且要求梯度
                if fqn not in self.parameters_to_ignore and param.requires_grad:
                    # 如果参数不在参数集合中，抛出异常
                    if param not in param_set:
                        self._log_and_throw(
                            ValueError,
                            f"Param with name {fqn} found in module parameters, but not DDP parameters."
                            " This indicates a bug in DDP, please report an issue to PyTorch.",
                        )
                    # 获取参数索引
                    param_index = param_to_param_index[param]
                    # 将参数索引和完全限定名称映射存入字典中
                    param_index_to_param_fqn[param_index] = fqn
    
        # 确保覆盖了所有参数
        if len(param_set) != len(param_index_to_param_fqn):
            self._log_and_throw(
                ValueError,
                (
                    "Expected param to name mapping to cover all parameters, but"
                    f" got conflicting lengths: {len(param_set)} vs "
                    f"{len(param_index_to_param_fqn)}. This indicates a bug in DDP"
                    ", please report an issue to PyTorch."
                ),
            )
    
        # 返回参数索引到完全限定名称的映射字典
        return param_index_to_param_fqn
    # 返回给定模块 m 及其子模块的参数生成器
    def _get_parameters(self, m, recurse=True):
        """Return a generator of module parameters."""

        def model_parameters(m):
            # 如果模块 m 有 _former_parameters 属性，则使用其值作为参数集合
            ps = (
                m._former_parameters.values()
                if hasattr(m, "_former_parameters")
                # 否则调用 m 的 parameters 方法获取参数集合（不递归）
                else m.parameters(recurse=False)
            )
            yield from ps

        # 如果 recurse 为 True，则遍历 m 及其所有子模块；否则只遍历 m 自身
        for mod in m.modules() if recurse else [m]:
            yield from model_parameters(mod)

    # 检查当前的默认进程组是否与 DDP 的默认进程组不匹配，若不匹配则抛出异常
    def _check_default_group(self):
        pickle_not_supported = False
        try:
            # 检查当前实例的 process_group 是否与默认的进程组不同
            if self.process_group != _get_default_group():
                pickle_not_supported = True
        except RuntimeError:
            # 若运行时出现异常，则标记 pickle_not_supported 为 True
            pickle_not_supported = True

        # 若 pickle_not_supported 为 True，则记录日志并抛出 RuntimeError 异常
        if pickle_not_supported:
            self._log_and_throw(
                RuntimeError,
                "DDP Pickling/Unpickling are only supported "
                "when using DDP with the default process "
                "group. That is, when you have called "
                "init_process_group and have not passed "
                "process_group argument to DDP constructor",
            )

    # 上下文管理器：禁用 DDP 进程间的梯度同步
    @contextmanager
    def no_sync(self):
        r"""
        Context manager to disable gradient synchronizations across DDP processes.

        Within this context, gradients will be accumulated on module
        variables, which will later be synchronized in the first
        forward-backward pass exiting the context.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> ddp = torch.nn.parallel.DistributedDataParallel(model, pg)
            >>> with ddp.no_sync():
            >>>     for input in inputs:
            >>>         ddp(input).backward()  # no synchronization, accumulate grads
            >>> ddp(another_input).backward()  # synchronize grads

        .. warning::
            The forward pass should be included inside the context manager, or
            else gradients will still be synchronized.
        """
        # 保存旧的 require_backward_grad_sync 值，并将其设置为 False
        old_require_backward_grad_sync = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            # 恢复 require_backward_grad_sync 值为旧值
            self.require_backward_grad_sync = old_require_backward_grad_sync

    # 类方法：获取当前活动的 DDP 模块
    @classmethod
    def _get_active_ddp_module(cls):
        """`TorchDynamo` requires DDP's status and module for cooperative optimization."""
        return cls._active_ddp_module

    # 上下文管理器：在 DDP 前向传播过程中使用的上下文管理器，用于标记当前活动的 DDP 模块
    # 注意：此 ctxmgr 函数在 torchdynamo 中标记为 'skip'，仅对 'module_to_run' 生效
    # 详见 torch._dynamo/eval_frame.py TorchPatcher.patch 获取更多详情
    @contextmanager
    @torch._disable_dynamo(recursive=False)
    def _inside_ddp_forward(self):
        # 将当前实例设置为活动的 DDP 模块
        DistributedDataParallel._active_ddp_module = self
        try:
            yield
        finally:
            # 清空活动的 DDP 模块
            DistributedDataParallel._active_ddp_module = None
    # 在分布式数据并行（DDP）模式下运行前向传播的方法。根据self._use_python_reducer决定使用Python reducer还是CPP reducer。
    def _run_ddp_forward(self, *inputs, **kwargs):
        if self._use_python_reducer:
            return self.module(*inputs, **kwargs)  # type: ignore[index]
        else:
            # 在DDP环境中执行前向传播，并在其内部运行
            with self._inside_ddp_forward():
                return self.module(*inputs, **kwargs)  # type: ignore[index]

    # 清空梯度缓冲区的方法。假设梯度累积是在自动求导引擎中原地进行的。
    def _clear_grad_buffer(self):
        if self._delay_grad_buffer is not None:
            # 当所有参数的梯度为None时，通过重置整个梯度缓冲区来批量清零所有参数的梯度。
            all_param_grad_none = all(
                param.grad is None for param in self._delay_all_reduce_params
            )

            for index, param in enumerate(self._delay_all_reduce_params):
                if param.grad is None:
                    param.grad = self._delay_grad_views[index]
                    if not all_param_grad_none:
                        param.grad.zero_()

            # 当所有参数的梯度都为None时，重置延迟梯度缓冲区的内容。
            if all_param_grad_none:
                self._delay_grad_buffer.zero_()

    # DDP的延迟初始化方法，在首次前向传播之前执行。
    def _lazy_init(self):
        # 设置在反向传播时的优化器
        self._setup_in_backward_optimizers()
        self._lazy_init_ran = True

    # 判断是否应该禁用CPP reducer的方法。
    def _should_disable_cpp_reducer(self) -> bool:
        return self._use_python_reducer and (
            torch._utils.is_compiling() or self._force_to_disable_cpp_reducer
        )

    # 前向传播方法，记录分布式数据并行的前向传播过程。
    def forward(self, *inputs, **kwargs):
        with torch.autograd.profiler.record_function("DistributedDataParallel.forward"):
            # 预处理输入参数
            inputs, kwargs = self._pre_forward(*inputs, **kwargs)
            # 根据self._delay_all_reduce_all_params决定使用哪种方式进行前向传播
            output = (
                self.module.forward(*inputs, **kwargs)
                if self._delay_all_reduce_all_params
                else self._run_ddp_forward(*inputs, **kwargs)
            )
            # 后处理前向传播的输出结果
            return self._post_forward(output)

    # 分散输入数据的方法，调用scatter_kwargs函数进行实际操作。
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    # 将输入数据和参数转换为关键字参数的方法，调用_to_kwargs函数实现。
    def to_kwargs(self, inputs, kwargs, device_id):
        # 兼容性保留，根据设备类型和ID创建torch.device对象，并设置相关参数。
        return _to_kwargs(
            inputs,
            kwargs,
            torch.device(self.device_type, device_id),
            self.use_side_stream_for_tensor_copies,
        )

    # 收集输出数据的方法，调用gather函数进行实际操作。
    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

    # 设置模型为训练模式的方法，调用父类的train方法并返回self。
    def train(self, mode=True):
        super().train(mode)
        return self

    # 当运行在join模式下时，安排一个allreduce操作来通知加入的进程，本次迭代是否运行反向传播同步。
    # 检查是否需要在反向传播中同步全局梯度
    def _check_global_requires_backward_grad_sync(self, is_joined_rank):
        # 如果不是加入的等级并且需要后向梯度同步
        if not is_joined_rank and self.require_backward_grad_sync:
            # 创建一个张量，值为1，设备为self.device
            requires_sync_tensor = torch.ones(1, device=self.device)
        else:
            # 创建一个张量，值为0，设备为self.device
            requires_sync_tensor = torch.zeros(1, device=self.device)

        # 执行异步的全局归约操作，将requires_sync_tensor的值在进程组中归约
        work = dist.all_reduce(
            requires_sync_tensor, group=self.process_group, async_op=True
        )

        # 如果是加入的等级
        if is_joined_rank:
            # 等待归约操作完成
            work.wait()
            # 检查是否需要在反向传播中同步梯度
            should_sync_backwards = requires_sync_tensor.item() != 0
            return should_sync_backwards
        else:
            # 不返回任何值，返回值不应该被使用
            return None

    # 当运行在加入模式时，检查并同步模块缓冲区
    def _check_and_sync_module_buffers(self):
        # 如果需要在前向传播中同步缓冲区
        if self._check_sync_bufs_pre_fwd():
            # 找到一个共同的等级作为权威等级
            authoritative_rank = self._find_common_rank(self._distributed_rank, False)
            # 同步模块缓冲区
            self._sync_module_buffers(authoritative_rank)

    # 当运行在加入模式时，同意一个共同的等级并广播模型参数到所有其他等级
    def _sync_final_model(self, is_last_joiner):
        # 确定将作为权威模型副本的进程
        self._authoritative_rank = self._find_common_rank(
            self._distributed_rank, is_last_joiner
        )
        # 同步模块状态
        _sync_module_states(
            module=self.module,
            process_group=self.process_group,
            broadcast_bucket_size=self.broadcast_bucket_size,
            src=self._authoritative_rank,
            params_and_buffers_to_ignore=self.parameters_to_ignore,
            broadcast_buffers=self.broadcast_buffers,
        )

    # 调度通信操作以匹配在Reducer的反向传播中调度的操作
    # 定义一个私有方法，用于在反向传播过程中执行所有通信操作
    def _match_all_reduce_for_bwd_pass(self):
        # 初始化一个空列表，用于存储通信任务
        comm_work = []
        # 根据 Reducer 调度的顺序安排通信任务，保证在连接模式下（如动态重建桶顺序时）保持相同的顺序。
        
        # 获取由零张量代替实际张量形状的梯度桶列表
        grad_buckets = self.reducer._get_zeros_like_grad_buckets()
        # 遍历每个梯度桶
        for grad_bucket in grad_buckets:
            # 使用 Reducer 执行通信钩子，并将返回的工作对象添加到 comm_work 列表中
            work = self.reducer._run_comm_hook(grad_bucket)
            comm_work.append(work)
        # 等待所有通信任务完成
        for work in comm_work:
            work.wait()

    # 对未使用的参数映射在所有进程中进行全局求和
    def _match_unused_params_allreduce(self):
        # 获取本地使用的参数映射
        locally_used_param_map = self.reducer._get_local_used_map()
        # 使用 process_group 对参数映射进行全局归约操作
        self.process_group.allreduce(locally_used_param_map)

    # DDP（分布式数据并行）的加入钩子，通过在前向和反向传播中镜像通信来支持在不均匀输入上训练
    def join(
        self,
        divide_by_initial_world_size: bool = True,
        enable: bool = True,
        throw_on_early_termination: bool = False,
    ):
        r"""
        DDP join hook enables training on uneven inputs by mirroring communications in forward and backward passes.

        Arguments:
            kwargs (dict): a :class:`dict` containing any keyword arguments
                to modify the behavior of the join hook at run time; all
                :class:`Joinable` instances sharing the same join context
                manager are forwarded the same value for ``kwargs``.

        The hook supports the following keyword arguments:
            divide_by_initial_world_size (bool, optional):
                If ``True``, then gradients are divided by the initial world
                size that DDP was launched with.
                If ``False``, then gradients are divided by the effective world
                size (i.e. the number of non-joined processes), meaning that
                the uneven inputs contribute more toward the global gradient.
                Typically, this should be set to ``True`` if the degree of
                unevenness is small but can be set to ``False`` in extreme
                cases for possibly better results.
                Default is ``True``.
        """
        # 获取关键字参数中的 divide_by_initial_world_size，若不存在则默认为 True
        divide_by_initial_world_size = kwargs.get("divide_by_initial_world_size", True)
        # 返回一个 _DDPJoinHook 对象，用于加入 DDP 训练
        return _DDPJoinHook(
            self, divide_by_initial_world_size=divide_by_initial_world_size
        )

    # 返回加入过程中使用的设备
    @property
    def join_device(self):
        return self.device

    # 返回加入过程中使用的进程组
    @property
    def join_process_group(self):
        return self.process_group
    def _register_buffer_comm_hook(
        self,
        state,
        hook: Callable,
        comm_hook_location=_BufferCommHookLocation.POST_FORWARD,
    ):
        r"""
        Allow custom registration of hooks that define how buffer are synchronized across ranks.

        The hook takes in an optional state and is passed in a Dict[str, Tensor]
        corresponding to buffer names and the buffers, and can run arbitrary reductions
        on buffers as opposed to DDP's default broadcast from rank 0. This is useful for
        example if a counter needs to be summed or averaged across ranks every iteration.

        Args:
            state (Any): Optional state that is passed to the hook.
            hook (Callable): Callable with the following signature:
                         ``hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]``
            comm_hook_location (_BufferCommHookLocation): Enum value indicating
                            where to run the hook.
                            _BufferCommHookLocation.PRE_FORWARD means that the
                            hook will run _before_ the forward pass, and
                            _BufferCommHookLocation.POST_FORWARD means that the
                            hook will run _after_ the forward pass.

            NOTE: To maximize performance, users can return a
                List[torch.futures.Future] from their hook, and DDP will
                install and await these hooks appropriately at the end of
                the backward pass. This will ensure all buffers are
                synchronized by the end of the backward pass. If this
                setting is used, it is recommended to pass
                comm_hook_location=_BufferCommHookLocation.POST_FORWARD,
                which will trigger the hook after the forward pass.
                If _BufferCommHookLocation.PRE_FORWARD is used, users must
                ensure appropriate synchronization when manipulating GPU
                buffers in the forward pass.
        """
        assert callable(hook)
        # 设置对象的缓冲通信钩子，用于自定义缓冲区在不同进程间同步的方式
        self.buffer_hook = _BufferCommHook(
            buffer_comm_hook=hook,
            buffer_comm_hook_state=state,
            buffer_comm_hook_location=comm_hook_location,
        )
    def _register_builtin_comm_hook(self, comm_hook_type):
        r"""
        注册内置的通信钩子，用于指定 DDP 如何跨多个 worker 聚合梯度。

        内置的钩子旨在为某些钩子提供高效的 C++ 实现，如果使用 Python 通信钩子在 Python 中实现可能效率不高。

        Args:
            comm_hook_type (dist.BuiltinCommHookType): 通信钩子的类型，如 ALLREDUCE, FP16_COMPRESS 等。

        .. warning ::
            DDP 通信钩子只能注册一次，并且应在调用 backward 之前注册。

        Example::
            下面是一个 FP16 压缩的示例，其中梯度在 allreduce 之前被压缩成 16 位浮点数，
            然后在 allreduce 后解压缩。

            >>> # xdoctest: +SKIP('undefined name')
            >>> ddp._register_builtin_comm_hook(dist.BuiltinCommHookType.FP16_COMPRESS)

        """
        assert self.logger is not None
        self.logger._set_comm_hook_name(str(comm_hook_type))
        dist._register_builtin_comm_hook(self.reducer, comm_hook_type)

    def _distributed_broadcast_coalesced(
        self, tensors, buffer_size, authoritative_rank=0
    ):
        """
        在分布式环境中广播并合并张量。

        Args:
            tensors (list of torch.Tensor): 待广播的张量列表。
            buffer_size (int): 缓冲区大小。
            authoritative_rank (int): 授权的排名，默认为 0。

        """
        dist._broadcast_coalesced(
            self.process_group, tensors, buffer_size, authoritative_rank
        )

    def _check_sync_bufs_post_fwd(self):
        """
        检查是否需要在前向传播后同步模块缓冲区。

        Returns:
            bool: 如果需要在前向传播后同步模块缓冲区，则为 True，否则为 False。

        """
        return (
            self.will_sync_module_buffers()
            and hasattr(self, "buffer_hook")
            and self.buffer_hook.buffer_comm_hook_location
            == _BufferCommHookLocation.POST_FORWARD
        )

    def _check_sync_bufs_pre_fwd(self):
        """
        检查是否需要在前向传播前同步模块缓冲区。

        Returns:
            bool: 如果需要在前向传播前同步模块缓冲区，则为 True，否则为 False。

        """
        return self.will_sync_module_buffers() and (
            not hasattr(self, "buffer_hook")
            or self.buffer_hook.buffer_comm_hook_location
            == _BufferCommHookLocation.PRE_FORWARD
        )

    def will_sync_module_buffers(self):
        """
        检查是否需要同步模块缓冲区。

        Returns:
            bool: 如果需要同步模块缓冲区，则为 True，否则为 False。

        """
        return (
            self.require_forward_param_sync
            and self.broadcast_buffers
            and len(self.modules_buffers) > 0
        )

    def _find_common_rank(self, input_rank, rank_cond):
        """
        查找共同的排名。

        Args:
            input_rank (int): 输入的排名。
            rank_cond (bool): 排名条件是否满足。

        Returns:
            int: 共同的排名。

        """
        # -1 表示此排名不考虑作为共同排名
        rank_to_use = torch.tensor(
            [input_rank if rank_cond else -1],
            device=self.device,
        )
        dist.all_reduce(rank_to_use, op=ReduceOp.MAX, group=self.process_group)
        if rank_to_use.item() == -1:
            self._log_and_throw(
                ValueError,
                "BUG! Expected rank_cond to be true for at least one process."
                " This indicates a bug in PyTorch, please report an issue.",
            )
        return rank_to_use.item()
    def _sync_buffers(self):
        with torch.no_grad():
            # module buffer sync
            # 模块缓冲区同步
            # Synchronize buffers across processes.
            # 在多进程中同步缓冲区。
            # If we are running DDP with the join manager, we have to agree
            # upon a rank to sync module buffers from, since rank 0 may
            # already have been joined and have stale module buffers.
            # 如果我们使用联合管理器运行 DDP，则必须就要从中同步模块缓冲区的秩达成一致，
            # 因为秩 0 可能已经加入并具有过时的模块缓冲区。
            if self._join_config.enable:
                authoritative_rank = self._find_common_rank(
                    self._distributed_rank, True
                )
            else:
                # The process with rank 0 is considered the authoritative copy.
                # 秩为 0 的进程被认为是权威副本。
                authoritative_rank = 0
            # Update self.modules_buffers incase any buffers were
            # reassigned.
            # 如果有任何缓冲区被重新分配，更新 self.modules_buffers
            self._assign_modules_buffers()
            self._sync_module_buffers(authoritative_rank)

    def _sync_module_buffers(self, authoritative_rank):
        if not hasattr(self, "buffer_hook"):
            # If buffer_hook attribute is not present, use default broadcast coalesced method.
            # 如果不存在 buffer_hook 属性，则使用默认的合并广播方法。
            self._default_broadcast_coalesced(authoritative_rank=authoritative_rank)
        else:
            # Use custom buffer communication hook and state.
            # 使用自定义的缓冲区通信钩子和状态。
            hook = self.buffer_hook.buffer_comm_hook
            state = self.buffer_hook.buffer_comm_hook_state
            # Invoke the hook to communicate buffers and get futures.
            # 调用钩子来通信缓冲区并获取 future 对象。
            futs = hook(state, self.named_module_buffers)
            if futs is not None:
                # Install post-backward futures for reduction.
                # 为了减少，安装后向 future 对象。
                self.reducer._install_post_backward_futures(futs)

    def _default_broadcast_coalesced(
        self, bufs=None, bucket_size=None, authoritative_rank=0
    ):
        """
        Broadcasts buffers from rank 0 to rest of workers.

        If bufs, bucket_size are None, default values self.modules_buffers
        and self.broadcast_bucket_size are used instead.
        """
        # 广播缓冲区从秩 0 到其他的 worker。
        # 如果 bufs 和 bucket_size 是 None，则使用 self.modules_buffers 和 self.broadcast_bucket_size 的默认值。
        if bufs is None:
            bufs = self.modules_buffers
        if bucket_size is None:
            bucket_size = self.broadcast_bucket_size

        self._distributed_broadcast_coalesced(bufs, bucket_size, authoritative_rank)

    def _passing_sync_batchnorm_handle(self, module):
        for layer in module.modules():
            if isinstance(layer, torch.nn.modules.SyncBatchNorm):
                if self.device_type == "cpu":
                    self._log_and_throw(
                        ValueError,
                        "SyncBatchNorm layers only work with GPU modules",
                    )
    # 检查通信钩子是否可调用，如果不可调用则抛出类型错误异常
    def _check_comm_hook(self, hook):
        if not callable(hook):
            self._log_and_throw(TypeError, "Communication hook must be callable.")

        # 获取钩子函数的签名信息
        sig = inspect.signature(hook)
        
        # 检查钩子函数参数中的 bucket 注释，应为 dist.GradBucket 类型
        if (
            sig.parameters["bucket"].annotation != inspect._empty
            and sig.parameters["bucket"].annotation != dist.GradBucket
        ):
            self._log_and_throw(
                ValueError,
                "Communication hook: bucket annotation should be dist.GradBucket.",
            )

        # 检查钩子函数返回值注释，应为 torch.futures.Future[torch.Tensor] 类型
        if (
            sig.return_annotation != inspect._empty
            and sig.return_annotation != torch.futures.Future[torch.Tensor]
        ):
            self._log_and_throw(
                ValueError,
                "Communication hook: return annotation should be torch.futures.Future[torch.Tensor].",
            )

        # 检查特定名称的钩子函数是否可用，并且检查 CUDA 和 NCCL 版本是否符合要求
        if hook.__name__ in [
            "bf16_compress_hook",
            "bf16_compress_wrapper_hook",
        ] and (
            (torch.version.cuda is None and torch.version.hip is None)
            or (
                torch.version.cuda is not None
                and int(torch.version.cuda.split(".")[0]) < 11
            )
            or not dist.is_available()
            or not dist.is_nccl_available()
            or torch.cuda.nccl.version() < (2, 10)
        ):
            self._log_and_throw(
                TypeError,
                "BF16 all reduce communication hook required CUDA 11+ and NCCL 2.10+.",
            )

    # 返回当前进程在分布式设置中的排名
    @property
    def _distributed_rank(self):
        return dist.get_rank(self.process_group)

    # 静态方法：返回由给定 DDP 单元管理的参数的生成器，可以选择是否返回带名称的参数
    @staticmethod
    def _get_data_parallel_params(module, named_params=False):
        """Return a generator of parameters managed by a given DDP unit."""
        for param in (
            module.parameters() if not named_params else module.named_parameters()
        ):
            # 忽略具有 "_ddp_ignored" 属性的参数
            if not hasattr(param, "_ddp_ignored"):
                yield param

    # 静态方法：设置要在模型中忽略的参数和缓冲区
    @staticmethod
    def _set_params_and_buffers_to_ignore_for_model(
        module, params_and_buffers_to_ignore
    ):
    ):
        """
        Set parameters and buffers to be ignored by DDP.

        Expected format for parameters is the fully qualified name: {module_name}.{param_name}, and
        similarly, {module_name}.{buffer_name} for buffers. For example:
        params_to_ignore = []
        # NB: model here is vanilla PyTorch module, not yet wrapped with DDP.
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if should_ignore(param):
                    # Create expected format
                    fqn = f"{module_name}.{param_name}"
                    params_to_ignore.append(fqn)
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            model,
            params_to_ignore
        )
        """
        # This is a workaround to set parameters and buffers DDP should ignore
        # during synchronization. It will be removed when the API is finalized
        # as part of addressing https://github.com/pytorch/pytorch/issues/43690.
        
        # 将要忽略的参数和缓冲设置到模块的 _ddp_params_and_buffers_to_ignore 属性中
        module._ddp_params_and_buffers_to_ignore = params_and_buffers_to_ignore
        
        # 遍历模块中的命名参数，如果在 params_and_buffers_to_ignore 中则标记为 DDP 忽略
        for name, param in module.named_parameters():
            if name in params_and_buffers_to_ignore:
                param._ddp_ignored = True
        
        # 遍历模块中的命名缓冲，如果在 params_and_buffers_to_ignore 中则标记为 DDP 忽略
        for name, buffer in module.named_buffers():
            if name in params_and_buffers_to_ignore:
                buffer._ddp_ignored = True

    def _get_ddp_logging_data(self):
        r"""
        Return a dictionary of logging data for debugging and analysis.

        This interface can be called after DistributedDataParallel() is
        constructed. It returns a dictionary of logging data. It could help
        for debugging and analysis. The logging data includes DistributedDataParallel
        constructor input parameters, some internal states of DistributedDataParallel
        and performance metrics. Simply print the dictionary and see what
        these metrics are.
        This is a prototype interface and subject to change in the future.
        """
        assert self.logger is not None
        # 获取分布式数据并行对象的日志数据，用于调试和分析
        ddp_logging_data = self.logger._get_ddp_logging_data()
        return {**ddp_logging_data.strs_map, **ddp_logging_data.ints_map}
    def _set_ddp_runtime_logging_sample_rate(self, sample_rate):
        r"""
        设置采集运行时统计信息的采样率。

        此接口允许用户设置采集运行时统计信息的采样率。在默认情况下，会在前10次迭代中记录运行时统计信息，
        超过10次迭代后，将每 "sample_rate" 次训练迭代记录一次运行时统计信息。
        如果 sample_rate 小于1，则抛出 ValueError 异常。

        这是一个原型接口，将来可能会发生变化。
        """
        if sample_rate < 1:
            self._log_and_throw(
                ValueError,
                "DDP runtime logging sample rate should be equal or greater than 1",
            )
        self.reducer._set_ddp_runtime_logging_sample_rate(sample_rate)

    def _set_static_graph(self):
        """
        设置静态图模式用于 DDP。

        建议在 DDP 构造函数中设置静态图，该私有 API 将在内部调用。
        """
        # 如果已经设置了 self.static_graph，则不需要再次设置
        if self.static_graph:
            warnings.warn(
                "You've set static_graph to be True, no need to set it again."
            )
            return
        self.static_graph = True
        self._static_graph_delay_allreduce_enqueued = False
        self.reducer._set_static_graph()
        assert self.logger is not None
        self.logger._set_static_graph()
        if self.find_unused_parameters:
            warnings.warn(
                "You passed find_unused_parameters=true to DistributedDataParallel, "
                "`_set_static_graph` will detect unused parameters automatically, so "
                "you do not need to set find_unused_parameters=true, just be sure these "
                "unused parameters will not change during training loop while calling "
                "`_set_static_graph`."
            )

    def _remove_autograd_hooks(self):
        """移除由 reducer 在模型参数上注册的自动求导钩子。"""
        self.reducer._remove_autograd_hooks()

    def _check_reducer_finalized(self):
        """
        检查 reducer 是否处理了所有桶并适当地完成了反向传播。

        在训练循环中调用 .backward() 后，调用此方法有助于避免由于 reducer 未能完成反向传播而导致的后续难以调试的错误。
        """
        self.reducer._check_reducer_finalized()

    def _set_sparse_metadata(self, global_unique_ids):
        """
        设置稀疏元数据信息。

        这个方法会将全局唯一标识作为参数传递给 reducer 的 _set_sparse_metadata 方法。
        """
        self.reducer._set_sparse_metadata(global_unique_ids)
    def _update_process_group(self, new_process_group):
        """
        Dynamically updates the process group for DDP so that we can shrink/expand DDP
        world size without having to reinitialize DDP.

        NOTE: If you are using custom communications hooks via, register_comm_hook,
        you need to update the process groups for those hooks separately.
        """
        # 强制重新构建新的进程组的 buckets。这确保所有的 rank 在重新构建 buckets 时是同步的，
        # 同时重新评估之前基于可能已更改的 world size 的 buckets 的假设。
        self._has_rebuilt_buckets = False  # 标记为 False，需要重新构建 buckets
        self.reducer._reset_state()  # 重置状态，以便重新建立与新进程组相关的 reducer 状态

        if not _rank_not_in_group(new_process_group):
            # 如果新进程组包含当前 rank
            self.process_group = new_process_group  # 更新当前对象的进程组
            self.reducer._update_process_group(new_process_group)  # 更新 reducer 对象的进程组

    def _set_ddp_sink_clone(self, val: bool):
        """
        Sets whether or not DDPSink should clone the output tensors or not.
        The default is True since if the loss is modified in place we run
        into the view is modified in-place error.

        Although, cloning the tensors can add significant memory and
        performance hit if the number and size of tensors are large. As
        a result, this can be set to False if you are not modifying the
        loss in place.
        """
        # 设置 DDPSink 是否应克隆输出张量。
        # 默认为 True，因为如果在原地修改损失，则可能会遇到“视图被原地修改”的错误。
        self._ddp_sink_clone = val  # 更新是否克隆输出张量的标志
```