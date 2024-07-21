# `.\pytorch\torch\distributed\_composable\fsdp\_fsdp_param_group.py`

```py
# 引入 mypy: allow-untyped-defs，允许未类型化的定义
import contextlib  # 提供对上下文管理器的支持
import logging  # 日志记录模块
from typing import Any, cast, Dict, List, NamedTuple, Optional, Set, Tuple  # 引入类型提示

import torch  # PyTorch 深度学习库
import torch._dynamo.compiled_autograd as ca  # 动态图自动求导的编译支持
import torch.distributed as dist  # 分布式通信支持
import torch.nn as nn  # PyTorch 神经网络模块
from torch.distributed.fsdp._common_utils import _named_parameters_with_duplicates  # 导入特定函数
from torch.profiler import record_function  # 性能分析记录函数
from torch.utils._pytree import tree_flatten, tree_unflatten  # 树结构的扁平化和反扁平化操作
from torch.utils.hooks import RemovableHandle  # 可移除的钩子句柄

from ._fsdp_api import MixedPrecisionPolicy, OffloadPolicy  # 导入自定义模块
from ._fsdp_collectives import (  # 导入自定义的集合操作
    AllGatherResult,
    foreach_all_gather,
    foreach_all_gather_copy_out,
    foreach_reduce,
)
from ._fsdp_common import FSDPMeshInfo, HSDPMeshInfo, TrainingState  # 导入自定义的共享信息类
from ._fsdp_param import FSDPParam, ParamModuleInfo, ShardedState  # 导入自定义的参数和状态类

logger = logging.getLogger("torch.distributed._composable.fsdp")  # 获取分布式 FSDP 模块的日志记录器

_ModuleToHandleDict = Dict[nn.Module, RemovableHandle]  # 用于状态字典的类型定义


"""
[Note: Overlapping all-gather copy-in and all-gather]
For implicit forward prefetching, we want to overlap the next copy-in with the
current all-gather. We do so using a separate copy-in stream. However, since
we have the all-gather input as a view into the output, we must make sure to
copy into different memory from the current all-gather's output. Thus, we keep
a reference to the current all-gather's output and have the next FSDP parameter
group free it after its copy-in. Finally, we have the last FSDP state flush the
reference to avoid holding onto memory after forward.
"""


class FSDPCommContext:
    """This has the communication state shared across FSDP states/parameter groups."""
    # 定义 FSDP 通信上下文类，用于共享跨 FSDP 状态/参数组的通信状态
    def lazy_init(self):
        # 检查是否有可用的 CUDA 设备，否则抛出错误
        if not torch.cuda.is_available():
            raise RuntimeError("FSDP requires CUDA for streams")
        
        # 设置全聚合和减少-分散流的优先级较高，有助于避免由于复制延迟而导致的计算阻塞
        high_priority = -1
        
        # 创建全聚合复制输入流，允许在前向传播中重叠下一个复制操作与当前全聚合操作
        self.all_gather_copy_in_stream = torch.cuda.Stream(priority=high_priority)
        
        # 创建全聚合流，允许在前向传播中重叠下一个全聚合操作与当前前向计算
        self.all_gather_stream = torch.cuda.Stream(priority=high_priority)
        
        # 创建减少-分散流，为后向逻辑（如梯度前后的划分和减少-分散）提供独立的执行“线程”
        self.reduce_scatter_stream = torch.cuda.Stream(priority=high_priority)
        
        # 运行 HSDP 全约减操作与全聚合/减少-分散操作并发执行，因为集体操作使用不同的网络资源，可以在典型的节点内分片/节点间复制情况下重叠
        self.all_reduce_stream = torch.cuda.Stream()
        
        # 全聚合/减少-分散状态保持对在一个流中生成并在另一个流中使用的集体张量的引用，并伴随 CUDA 事件进行同步
        self.all_gather_state: Optional[AllGatherState] = None
        self.reduce_scatter_state: Optional[ReduceScatterState] = None
        
        # 前向传播后的顺序，用于显式后向预取
        self.post_forward_order: List[FSDPParamGroup] = []  # 这会导致引用循环

    def get_all_gather_streams(
        self, training_state: TrainingState
    ) -> Tuple[torch.cuda.Stream, torch.cuda.Stream]:
        if training_state in (TrainingState.FORWARD, TrainingState.PRE_BACKWARD):
            # 在隐式预取时使用单独的流
            return self.all_gather_copy_in_stream, self.all_gather_stream
        
        # 返回当前 CUDA 流，用于其它情况
        current_stream = torch.cuda.current_stream()
        return current_stream, current_stream
# 定义一个名为 AllGatherState 的命名元组，用于保存 all-gather 操作的结果和 CUDA 事件对象
class AllGatherState(NamedTuple):
    all_gather_result: AllGatherResult
    event: torch.cuda.Event  # all-gather copy-out

# 定义一个名为 ReduceScatterState 的命名元组，用于保存 reduce-scatter 操作的输入张量和 CUDA 事件对象
class ReduceScatterState(NamedTuple):
    reduce_scatter_input: torch.Tensor
    event: torch.cuda.Event  # reduce-scatter event

class FSDPParamGroup:
    """This class represents a parameter group to communicate together."""

    _orig_dtype: torch.dtype  # 原始数据类型
    _reduce_dtype: Optional[torch.dtype]  # 可选的 reduce 数据类型

    def __init__(
        self,
        params: List[nn.Parameter],  # 参数列表
        module: nn.Module,  # 模块对象
        mesh_info: FSDPMeshInfo,  # FSDP 网络信息
        post_forward_mesh_info: Optional[FSDPMeshInfo],  # 可选的后向传播 FSDP 网络信息
        device: torch.device,  # 设备类型
        mp_policy: MixedPrecisionPolicy,  # 混合精度策略
        offload_policy: OffloadPolicy,  # 卸载策略
        ):
            self.module = module  # 允许引用循环，因为生命周期是一对一的
            # 根据参数和模块获取参数模块信息
            param_module_infos = _get_param_module_infos(params, module)
            # 使用参数信息和其他相关信息创建FSDPParam对象列表
            self.fsdp_params = [
                FSDPParam(
                    param,
                    module_info,
                    mesh_info,
                    post_forward_mesh_info,
                    device,
                    mp_policy,
                    offload_policy,
                )
                for param, module_info in zip(params, param_module_infos)
            ]
            # 设置mesh_info，用于后向传播之后的信息
            self.mesh_info = mesh_info
            # 设置post_forward_mesh_info，用于后向传播之后的信息
            self.post_forward_mesh_info = post_forward_mesh_info
            # 设置设备信息
            self.device = device
            # 设置mp_policy，用于并行处理策略
            self.mp_policy = mp_policy
            # 设置训练状态为IDLE
            self._training_state = TrainingState.IDLE
            # Group的分片状态始终与其参数的分片状态匹配
            self._sharded_state = ShardedState.SHARDED
            # 设置模块的全限定名，从根模块前缀化
            self._module_fqn: Optional[str] = None

            # - Hook state
            # 用于保存模块到预保存状态字典挂钩句柄的字典
            self._module_to_pre_save_state_dict_hook_handle: _ModuleToHandleDict = {}
            # 用于加载模块到预加载状态字典挂钩句柄的字典
            self._module_to_pre_load_state_dict_hook_handle: _ModuleToHandleDict = {}

            # - Communication and communication/computation overlap
            # 初始化FSDPCommContext对象，用于通信上下文
            self.comm_ctx = FSDPCommContext()
            # Group在共享的后向顺序中的索引列表
            self._post_forward_indices: List[int] = []
            # 是否减少梯度（无论是对于FSDP还是HSDP）
            self.reduce_grads: bool = True
            # 是否对HSDP进行全局梯度归约；仅在self.reduce_grads为true时使用，设置为false表示reduce-scatter但不进行all-reduce
            self.all_reduce_grads: bool = True
            # 是否在后向传播后重新分片参数（仅对梯度累积有用）
            self.reshard_after_backward: bool = True

            # - CUDA events for stream synchronization
            # 保存全聚集输出缓冲区、同步对象和元数据的可选对象
            self._all_gather_result: Optional[AllGatherResult] = None
            # 保存在组的后向过程中标记结束的reduce-scatter/all-reduce view-out CUDA事件，在后向传播结束时应等待
            self._post_reduce_event: Optional[torch.cuda.Event] = None
            # 保存在转向不同world大小时重新分片后的前向CUDA事件，在下一个unshard时应等待
            self._reshard_after_forward_event: Optional[torch.cuda.Event] = None

            # 仅对HSDP有效，如果在没有全局梯度归约的情况下累积梯度，则保存部分reduce输出（仅reduce-scattered但未进行all-reduced）
            self._partial_reduce_output: Optional[torch.Tensor] = None

        # 初始化 #
    # 初始化多处理数据类型属性
    def _init_mp_dtypes(self) -> None:
        # 遍历所有的FSDP参数，调用init_dtype_attrs方法初始化数据类型属性
        for fsdp_param in self.fsdp_params:
            fsdp_param.init_dtype_attrs(self.mp_policy)
        
        # 收集所有原始数据类型，确保只有一种
        orig_dtypes = {fsdp_param.orig_dtype for fsdp_param in self.fsdp_params}
        if len(orig_dtypes) != 1:
            # 如果不止一种原始数据类型，抛出断言错误
            raise AssertionError(
                f"FSDP expects uniform original parameter dtype but got {orig_dtypes}"
            )
        
        # 设置对象的原始数据类型为唯一的那种类型
        self._orig_dtype = next(iter(orig_dtypes))
        
        # 收集所有reduce数据类型，确保只有一种
        reduce_dtypes = {fsdp_param.reduce_dtype for fsdp_param in self.fsdp_params}
        if len(reduce_dtypes) != 1:
            # 如果不止一种reduce数据类型，抛出断言错误
            raise AssertionError(
                f"FSDP expects uniform reduce dtype but got {reduce_dtypes}"
            )
        
        # 设置对象的reduce数据类型为唯一的那种类型
        self._reduce_dtype = next(iter(reduce_dtypes))

    def lazy_init(self):
        # 懒初始化应该是幂等的（多次调用效果相同）
        
        # 获取所有在"meta"设备上的参数名列表
        param_names_on_meta = [
            fsdp_param._param_fqn
            for fsdp_param in self.fsdp_params
            if fsdp_param.sharded_param.device.type == "meta"
        ]
        
        # 如果存在在"meta"设备上的参数，则抛出运行时错误
        if param_names_on_meta:
            raise RuntimeError(
                "FSDP parameters should be materialized from meta device before training, "
                f"but the following were still on meta device: {param_names_on_meta}\n"
                "For example, call module.to_empty(device) to materialize to device and "
                "call module.reset_parameters() on each module to initialize values."
            )
        
        # 在构造之后但在前向传播之前，懒初始化混合精度属性
        self._init_mp_dtypes()  # 初始化多处理数据类型
        self._register_state_dict_hooks()  # 注册状态字典钩子函数

    # 运行时 #
    def unshard(self, async_op: bool = False):
        # 如果已经调用过_unshard_result，则返回
        if self._all_gather_result is not None:
            return
        
        # 如果已经是未分片状态，则返回（无操作）
        if self.is_unsharded:
            return
        
        # 如果存在_reshard_after_forward_event，则等待所有聚合流处理完成
        if self._reshard_after_forward_event is not None:
            # 释放所有聚合流事件后的事件
            self._wait_all_gather_streams_on_event(self._reshard_after_forward_event)
            self._reshard_after_forward_event = None
        
        # 使用record_function记录“FSDP::all_gather”操作
        with record_function(self._with_fqn("FSDP::all_gather")):
            # 对每个FSDP参数执行foreach_all_gather操作
            self._all_gather_result = foreach_all_gather(
                self.fsdp_params,
                self._all_gather_process_group,
                async_op,
                *self.comm_ctx.get_all_gather_streams(self._training_state),
                self.device,
            )
    def wait_for_unshard(self):
        """
        1. In forward with implicit prefetching, to overlap the current copy-out
           with the next all-gather, we save a reference to the current all-gather
           result to free after the next copy-out.
        2. Otherwise (explicit prefetching or in backward), we free the
           all-gather result immediately after the current copy-out since we can
           already overlap the current copy-out with the previous reduce-scatter.
        """
        # 如果没有前置的 unshard 操作，则直接返回
        if not self._all_gather_result:
            return
        # 如果当前处于 FORWARD 训练状态，进行隐式预取
        if self._training_state == TrainingState.FORWARD:  # implicit prefetch
            # 如果存在之前的 all-gather 状态
            if prev_all_gather_state := self.comm_ctx.all_gather_state:
                # 等待前一个 all-gather 的事件完成
                self._wait_all_gather_streams_on_event(prev_all_gather_state.event)
                # 释放前一个 all-gather 的结果
                self.comm_ctx.all_gather_state = None
        # 记录函数调用的性能数据
        with record_function(self._with_fqn("FSDP::all_gather_copy_out")):
            # 对每个 all-gather 的结果进行复制输出
            foreach_all_gather_copy_out(
                self._all_gather_result,
                self.fsdp_params,
                self._all_gather_process_group,
            )
        # 初始化每个 fsdp_param 的未分片参数
        for fsdp_param in self.fsdp_params:
            fsdp_param.init_unsharded_param()
        # 将对象转换为未分片状态
        self._to_unsharded()
        # 创建 CUDA 事件并记录
        all_gather_copy_out_event = torch.cuda.Event()
        all_gather_copy_out_event.record()
        # 如果当前处于 FORWARD 训练状态，保存当前 all-gather 的状态
        if self._training_state == TrainingState.FORWARD:
            self.comm_ctx.all_gather_state = AllGatherState(
                self._all_gather_result, all_gather_copy_out_event
            )
        else:
            # 否则，等待所有 all-gather 流处理完当前事件
            self._wait_all_gather_streams_on_event(all_gather_copy_out_event)
        # 清空 all-gather 的结果，除非已保存在 `all_gather_state` 中
        self._all_gather_result = None

    def _wait_all_gather_streams_on_event(self, event: torch.cuda.Event):
        # 在延迟初始化之前调用 `unshard` 意味着流尚未初始化
        if hasattr(self.comm_ctx, "all_gather_copy_in_stream"):
            # 等待所有聚合复制输入流的事件完成
            self.comm_ctx.all_gather_copy_in_stream.wait_event(event)
        if hasattr(self.comm_ctx, "all_gather_stream"):
            # 等待所有聚合流的事件完成
            self.comm_ctx.all_gather_stream.wait_event(event)

    def reshard(self):
        # 如果当前处于 FORWARD 训练状态，并且不需要在 FORWARD 后重新分片，则直接返回
        if self._training_state == TrainingState.FORWARD:
            if not self._reshard_after_forward:
                return
            # 如果使用后向后重新分片，则转换为后向后的分片状态，并记录事件
            if self._use_post_forward_mesh:
                self._to_sharded_post_forward()
                self._reshard_after_forward_event = torch.cuda.Event()
                self._reshard_after_forward_event.record()
                return
        # 否则，转换为分片状态
        self._to_sharded()

    def pre_forward(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
        ):
        # 准备进行 FORWARD 操作前的处理
        # （此处未提供完整代码，因此无需进一步注释）
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        # 如果未启用编译自动微分，则记录调试信息
        if not ca.compiled_autograd_enabled:
            logger.debug("%s", self._with_fqn("FSDP::pre_forward"))
        # 记录函数调用
        with record_function(self._with_fqn("FSDP::pre_forward")):
            # 设置训练状态为前向传播
            self._training_state = TrainingState.FORWARD
            # 取消分片操作
            self.unshard()
            # 等待取消分片完成
            self.wait_for_unshard()
            # 注册后向传播钩子
            args, kwargs = self._register_post_backward_hook(args, kwargs)
            # 返回参数和关键字参数
            return args, kwargs

    def post_forward(self, module: nn.Module, input: Any, output: Any):
        # 如果未启用编译自动微分，则记录调试信息
        if not ca.compiled_autograd_enabled:
            logger.debug("%s", self._with_fqn("FSDP::post_forward"))
        # 记录函数调用
        with record_function(self._with_fqn("FSDP::post_forward")):
            # 重新分片操作
            self.reshard()
            # 记录后向传播
            self._record_post_forward()
            # 设置训练状态为空闲
            self._training_state = TrainingState.IDLE
            # 返回输出结果
            return output

    def _record_post_forward(self) -> None:
        # 由于每个前向调用之前都有一个预后向分片组，我们记录每次使用（带有重复）
        post_forward_index = len(self.comm_ctx.post_forward_order)
        # 将当前对象添加到后向传播顺序中
        self.comm_ctx.post_forward_order.append(self)
        # 记录后向传播索引
        self._post_forward_indices.append(post_forward_index)

    def pre_backward(self, default_prefetch: bool, *unused: Any):
        # 如果训练状态为预后向，则直接返回
        if self._training_state == TrainingState.PRE_BACKWARD:
            return
        # 如果未启用编译自动微分，则记录调试信息
        if not ca.compiled_autograd_enabled:
            logger.debug("%s", self._with_fqn("FSDP::pre_backward"))
        # 记录函数调用
        with record_function(self._with_fqn("FSDP::pre_backward")):
            # 设置训练状态为预后向
            self._training_state = TrainingState.PRE_BACKWARD
            # 取消分片操作（如果未预取）
            self.unshard()
            # 等待取消分片完成
            self.wait_for_unshard()
            # 如果默认预取，则进行后向预取
            if default_prefetch:
                self._backward_prefetch()

    def finalize_backward(self):
        # 如果存在后减少事件，则等待其完成
        if self._post_reduce_event is not None:
            torch.cuda.current_stream().wait_event(self._post_reduce_event)
            self._post_reduce_event = None
        # 对于每个 FSDP 参数，如果存在梯度偏移事件，则同步并清空
        for fsdp_param in self.fsdp_params:
            if fsdp_param.grad_offload_event is not None:
                fsdp_param.grad_offload_event.synchronize()
                fsdp_param.grad_offload_event = None
        # 清空后向传播索引列表
        self._post_forward_indices.clear()

    def _backward_prefetch(self) -> None:
        # 如果训练状态为预后向且后向索引非空，则执行预取操作
        if self._training_state == TrainingState.PRE_BACKWARD:
            if not self._post_forward_indices:
                # 如果运行多个 `backward` 时索引为空，则直接返回
                return
            # 弹出当前后向索引
            curr_index = self._post_forward_indices.pop()
            # 计算目标索引
            if (target_index := curr_index - 1) < 0:
                return
            # 根据反向后向顺序，简单地预取，可能会导致部分模块未在此次反向中使用
            target_fsdp_param_group = self.comm_ctx.post_forward_order[target_index]
            self._prefetch_unshard(target_fsdp_param_group, "backward")
    def _prefetch_unshard(
        target_fsdp_param_group: "FSDPParamGroup", pass_type: str
    ) -> None:
        # 根据传入的 pass_type 确定训练状态
        if pass_type == "backward":
            training_state = TrainingState.PRE_BACKWARD
        elif pass_type == "forward":
            training_state = TrainingState.FORWARD
        else:
            raise ValueError(f"Unknown pass type: {pass_type}")
        # 获取目标 FSDPParamGroup 的全限定名称
        target_fqn = target_fsdp_param_group._module_fqn
        # 使用记录函数记录当前操作的信息
        with record_function(
            f"FSDP::{pass_type}_prefetch for {target_fqn}"
        ), target_fsdp_param_group.use_training_state(training_state):
            # 执行 unshard 操作
            target_fsdp_param_group.unshard()

    # Utilities #
    def _to_sharded(self):
        # 如果当前对象未处于 sharded 状态，则将每个 fsdp_param 转换为 sharded 状态
        if not self.is_sharded:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_sharded()
            self._sharded_state = ShardedState.SHARDED

    def _to_sharded_post_forward(self):
        # 如果当前对象未处于 sharded_post_forward 状态，则将每个 fsdp_param 转换为 sharded_post_forward 状态
        if not self.is_sharded_post_forward:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_sharded_post_forward()
            self._sharded_state = ShardedState.SHARDED_POST_FORWARD

    def _to_unsharded(self):
        # 如果当前对象未处于 unsharded 状态，则将每个 fsdp_param 转换为 unsharded 状态
        if not self.is_unsharded:
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_unsharded()
            self._sharded_state = ShardedState.UNSHARDED

    @property
    def is_sharded(self) -> bool:
        # 返回当前对象是否处于 sharded 状态
        return self._sharded_state == ShardedState.SHARDED

    @property
    def is_sharded_post_forward(self) -> bool:
        # 返回当前对象是否处于 sharded_post_forward 状态
        return self._sharded_state == ShardedState.SHARDED_POST_FORWARD

    @property
    def is_unsharded(self) -> bool:
        # 返回当前对象是否处于 unsharded 状态
        return self._sharded_state == ShardedState.UNSHARDED

    @contextlib.contextmanager
    def use_training_state(self, training_state: TrainingState):
        # 临时设置对象的训练状态，并在上下文结束后恢复原来的训练状态
        old_training_state = self._training_state
        self._training_state = training_state
        try:
            yield
        finally:
            self._training_state = old_training_state

    # Hook Registration #
    def _register_post_backward_hook(
        self, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        # 当编译开启时，依赖于 `root_post_backward_callback` 来调用每个 `FSDPParamGroup.post_backward`
        if ca.compiled_autograd_enabled:
            # 如果编译的自动求导已启用，则直接返回参数 args 和 kwargs
            return args, kwargs
        # 如果梯度未启用，则直接返回参数 args 和 kwargs
        if not torch.is_grad_enabled():
            return args, kwargs
        # 将 args 和 kwargs 展平成列表和对应的结构信息
        args_list, args_spec = tree_flatten(args)
        kwargs_list, kwargs_spec = tree_flatten(kwargs)
        # 将 args 和 kwargs 的列表合并成一个总列表
        args_kwargs_list = list(args_list) + list(kwargs_list)
        # 初始化用于存储输入张量索引和张量本身的列表
        inp_tensor_indices: List[int] = []
        inp_tensors: List[torch.Tensor] = []
        # 遍历 args_kwargs_list，找出所有需要梯度的张量，并记录它们的索引和张量本身
        for i, obj in enumerate(args_kwargs_list):
            if torch.is_tensor(obj) and obj.requires_grad:
                inp_tensor_indices.append(i)
                inp_tensors.append(obj)
        # 如果没有找到需要梯度的张量，则直接返回原始的 args 和 kwargs
        if len(inp_tensors) == 0:
            return args, kwargs  # 没有需要梯度的张量
        # 调用 RegisterPostBackwardFunction.apply 方法，注册后向传播的函数
        inp_tensors = RegisterPostBackwardFunction.apply(self, *inp_tensors)
        # 将注册后的张量替换回 args_kwargs_list 中原来的位置
        for inp_tensor_idx, inp_tensor in zip(inp_tensor_indices, inp_tensors):
            args_kwargs_list[inp_tensor_idx] = inp_tensor
        # 根据处理后的 args_list 和 kwargs_list，重新构建 args 和 kwargs
        args_list = args_kwargs_list[: len(args_list)]
        kwargs_list = args_kwargs_list[len(args_list) :]
        args = tree_unflatten(args_list, args_spec)
        kwargs = tree_unflatten(kwargs_list, kwargs_spec)
        # 返回重新构建后的 args 和 kwargs
        return args, kwargs

    def _register_state_dict_hooks(self) -> None:
        # 检查预保存状态字典钩子和预加载状态字典钩子的数量是否一致，若不一致则抛出异常
        num_pre_save_hooks = len(self._module_to_pre_save_state_dict_hook_handle)
        num_pre_load_hooks = len(self._module_to_pre_load_state_dict_hook_handle)
        assert (
            num_pre_save_hooks == num_pre_load_hooks
        ), f"Pre-save: {num_pre_save_hooks} pre-load: {num_pre_load_hooks}"
        # 如果已经注册了预保存状态字典钩子，则直接返回，无需再次注册
        if num_pre_save_hooks > 0:
            return  # already registered
        # 获取包含 FSDP 参数的模块集合
        modules_with_fsdp_params: Set[nn.Module] = {
            fsdp_param._module_info.module for fsdp_param in self.fsdp_params
        }

        def to_sharded_hook(*args: Any, **kwargs: Any) -> None:
            self._to_sharded()

        # 遍历具有 FSDP 参数的模块集合，为每个模块注册预保存和预加载状态字典钩子
        for module in modules_with_fsdp_params:
            self._module_to_pre_save_state_dict_hook_handle[
                module
            ] = module.register_state_dict_pre_hook(to_sharded_hook)
            self._module_to_pre_load_state_dict_hook_handle[
                module
            ] = module._register_load_state_dict_pre_hook(to_sharded_hook)

    # Properties #
    @property
    def _reshard_after_forward(self) -> bool:
        # 返回是否在前向传播后进行重分片操作的布尔值
        return self.post_forward_mesh_info is not None

    @property
    def _use_post_forward_mesh(self) -> bool:
        # 返回是否使用后向传播后的网格信息的布尔值
        return (
            self._reshard_after_forward
            and self.mesh_info != self.post_forward_mesh_info
        )

    @property
    def _is_hsdp(self) -> bool:
        # 返回当前 mesh_info 是否是 HSDPMeshInfo 类型的布尔值
        return isinstance(self.mesh_info, HSDPMeshInfo)

    @property
    # 返回当前对象的后向传播后的网格信息，如果启用了分片后向传播则返回，否则返回正常的网格信息
    def _all_gather_process_group(self) -> dist.ProcessGroup:
        mesh_info = (
            cast(FSDPMeshInfo, self.post_forward_mesh_info)
            if self.is_sharded_post_forward
            else self.mesh_info
        )
        # 断言mesh_info是FSDPMeshInfo类型的实例
        assert isinstance(mesh_info, FSDPMeshInfo)
        # 返回网格信息的分片进程组
        return mesh_info.shard_process_group

    @property
    # 返回当前对象的减少散步进程组
    def _reduce_scatter_process_group(self) -> dist.ProcessGroup:
        # 断言mesh_info是FSDPMeshInfo类型的实例
        assert isinstance(self.mesh_info, FSDPMeshInfo)
        # 返回网格信息的分片进程组
        return self.mesh_info.shard_process_group

    @property
    # 返回当前对象的全局减少进程组
    def _all_reduce_process_group(self) -> dist.ProcessGroup:
        # 断言mesh_info是HSDPMeshInfo类型的实例
        assert isinstance(self.mesh_info, HSDPMeshInfo)
        # 返回网格信息的复制进程组
        return self.mesh_info.replicate_process_group

    # 返回带有完全限定名称的标签字符串
    def _with_fqn(self, label: str) -> str:
        if self._module_fqn:
            # 如果存在模块的完全限定名称，则返回带有完全限定名称的标签字符串
            return f"{label} ({self._module_fqn})"
        # 否则，返回原始的标签字符串
        return label
def _get_param_module_infos(
    params: List[nn.Parameter], module: nn.Module
) -> List[ParamModuleInfo]:
    """
    Shared parameter: lin1.weight = lin2.weight
    Shared module: mlp.lin1 = mlp.lin2
    We do not remove duplicates when traversing both modules and parameters to
    find shared modules' parameters and shared parameters within a module.
    """
    # 将参数列表转换为集合，以便快速查找
    params_set = set(params)
    # 创建空字典，用于存储参数到 ParamModuleInfo 对象的映射关系
    param_to_module_info: Dict[nn.Parameter, ParamModuleInfo] = {}
    # 遍历模块及其子模块，查找共享的参数和共享参数所在的模块
    for _, submodule in module.named_modules(remove_duplicate=False):
        # 获取当前子模块中带有重复参数的命名参数生成器
        for param_name, param in _named_parameters_with_duplicates(
            submodule, recurse=False
        ):
            # 如果参数在给定的参数集合中
            if param in params_set:
                # 如果参数尚未在映射字典中，则将其添加进去
                if param not in param_to_module_info:
                    param_to_module_info[param] = ParamModuleInfo(submodule, param_name)
                else:
                    # 如果参数已经在映射字典中，说明这个参数在多个模块中共享
                    # 将当前子模块和参数名分别添加到共享模块和共享参数名列表中
                    param_to_module_info[param].shared_modules.append(submodule)
                    param_to_module_info[param].shared_param_names.append(param_name)
    # 如果映射字典的长度不等于参数列表的长度，说明有参数没有在模块树中找到对应的模块
    if len(param_to_module_info) != len(params):
        raise AssertionError(f"Some parameters are not in the module tree of {module}")
    # 返回按参数顺序排列的 ParamModuleInfo 列表
    return [param_to_module_info[param] for param in params]


class RegisterPostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param_group: FSDPParamGroup, *inputs: torch.Tensor):
        # All tensors in `inputs` should require gradient
        # 将 param_group 存储在上下文中，并直接返回输入张量列表
        ctx.param_group = param_group
        return inputs

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        # 调用 param_group 的 post_backward 方法
        ctx.param_group.post_backward()
        # 返回一个 None 和输入梯度张量列表的元组作为反向传播的结果
        return (None,) + grads
```