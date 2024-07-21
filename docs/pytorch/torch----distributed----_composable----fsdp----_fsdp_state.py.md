# `.\pytorch\torch\distributed\_composable\fsdp\_fsdp_state.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和类
import functools  # 导入 functools 模块，用于装饰器功能
import logging  # 导入 logging 模块，用于记录日志
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING  # 导入类型提示相关的类和函数

import torch  # 导入 PyTorch 模块
import torch._dynamo.compiled_autograd as ca  # 导入编译后自动求导的模块
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch.autograd import Variable  # 导入 PyTorch 变量自动求导模块
from torch.distributed._composable_state import (  # 导入可组合状态模块相关函数和类
    _get_module_state,
    _insert_module_state,
    _State,
)
from torch.distributed.utils import _to_kwargs  # 导入转换为关键字参数的函数
from torch.utils._pytree import tree_flatten, tree_map  # 导入树结构处理相关函数

from ._fsdp_api import MixedPrecisionPolicy  # 导入混合精度策略类
from ._fsdp_common import _cast_fp_tensor, TrainingState  # 导入浮点数张量类型转换函数和训练状态枚举类
from ._fsdp_param_group import FSDPCommContext, FSDPParamGroup  # 导入通信上下文和参数组类

if TYPE_CHECKING:
    from ._fsdp_param import FSDPParam  # 在类型检查模式下导入 FSDPParam 类

logger = logging.getLogger("torch.distributed._composable.fsdp")  # 获取名为 "torch.distributed._composable.fsdp" 的 logger 对象


class FSDPStateContext:
    """This has state shared across FSDP states."""

    def __init__(self):
        # 所有根状态模块树中的 FSDP 状态列表
        self.all_states: List[FSDPState] = []
        # 迭代的前向根仅运行一次每次前向的逻辑；这个根可能不是由延迟初始化设置的整体根，
        # 在仅子模块运行前向（例如仅用于评估的编码器）的情况下
        self.iter_forward_root: Optional[FSDPState] = None
        # 后向过程仅应在这个后向的最终回调中排队一次
        self.post_backward_final_callback_queued: bool = False
        # 是否在这个后向的最终回调中完成后向过程
        self.is_last_backward: bool = True
        # 可选的由用户提供的事件，在优化器之后记录用于在根前向时所有聚合流等待的事件
        self.post_optim_event: Optional[torch.cuda.Event] = None


def disable_if_config_true(func):
    @functools.wraps(func)
    def fsdp_hook_wrapper(*args, **kwargs):
        if torch._dynamo.config.skip_fsdp_hooks:
            # 如果配置要求跳过 FSDP 钩子，则禁用函数并返回
            return torch._dynamo.disable(func, recursive=True)(*args, **kwargs)
        else:
            # 否则，正常调用函数
            return func(*args, **kwargs)

    return fsdp_hook_wrapper


class FSDPState(_State):
    def __init__(self):
        super().__init__()
        # FSDP 参数组对象，初始为 None
        self._fsdp_param_group: Optional[FSDPParamGroup] = None
        # 根状态在延迟初始化期间设置
        self._is_root: Optional[bool] = None
        # FSDP 状态上下文对象
        self._state_ctx = FSDPStateContext()
        # FSDP 通信上下文对象
        self._comm_ctx = FSDPCommContext()
        # 训练状态，初始为 IDLE
        self._training_state: TrainingState = TrainingState.IDLE
        # 待预取的前向状态列表
        self._states_to_forward_prefetch: List[FSDPState] = []
        # 待预取的后向状态列表
        self._states_to_backward_prefetch: List[FSDPState] = []

    # 定义一个独立的初始化函数，因为 `__init__` 在合约中被调用
    def init(
        self, module: nn.Module, device: torch.device, mp_policy: MixedPrecisionPolicy
    # 将模块状态插入到模块列表中
    _insert_module_state(module, self)
    # 设置当前对象的模块
    self._module = module
    # 设置当前对象的设备
    self._device = device
    # 设置当前对象的多进程策略
    self._mp_policy = mp_policy
    # 注册前向传播的预处理钩子，并指定为前置钩子，携带关键字参数
    self._pre_forward_hook_handle = module.register_forward_pre_hook(
        self._pre_forward, prepend=True, with_kwargs=True
    )
    # 注册前向传播的后处理钩子，并指定为非前置钩子
    self._post_forward_hook_handle = module.register_forward_hook(
        self._post_forward, prepend=False
    )

def _root_pre_forward(
    self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    # 惰性初始化当前对象
    self._lazy_init()
    # 如果已经存在迭代根节点，则直接返回参数
    if self._state_ctx.iter_forward_root is not None:
        return args, kwargs
    # 如果未启用编译自动梯度，则记录调试信息
    if not ca.compiled_autograd_enabled:
        logger.debug("FSDP::root_pre_forward")
    # 将当前对象设置为迭代根节点
    self._state_ctx.iter_forward_root = self
    # 使用 Torch Profiler 记录函数执行时间："FSDP::root_pre_forward"
    with torch.profiler.record_function("FSDP::root_pre_forward"):
        # 如果存在优化器后事件，则等待隐式预取的全收集操作
        if (event := self._state_ctx.post_optim_event) is not None:
            self._comm_ctx.all_gather_copy_in_stream.wait_event(event)
            self._comm_ctx.all_gather_stream.wait_event(event)
            self._state_ctx.post_optim_event = None
        else:
            current_stream = torch.cuda.current_stream()
            self._comm_ctx.all_gather_copy_in_stream.wait_stream(current_stream)
            self._comm_ctx.all_gather_stream.wait_stream(current_stream)
        # 如果设备类型为 "cuda"，则使用 Torch Profiler 记录函数执行时间："FSDP::inputs_to_device"
        if self._device.type == "cuda":
            with torch.profiler.record_function("FSDP::inputs_to_device"):
                # 将参数和关键字参数转换为设备对应的形式
                args_tuple, kwargs_tuple = _to_kwargs(
                    args, kwargs, self._device, False
                )  # same as DDP
            args, kwargs = args_tuple[0], kwargs_tuple[0]
    # 返回处理后的参数和关键字参数
    return args, kwargs
    def _lazy_init(self) -> None:
        """
        Lazy initialization represents when all modules' parallelisms have
        finalized (e.g. FSDP has been applied to all desired modules). This
        means that we can determine which state is the root, and we do so by
        the 1st state to run forward.
        """
        # 如果已经初始化过，则直接返回，无需执行任何操作
        if self._is_root is not None:
            return  # no-op: already initialized
        
        # 将当前实例标记为根状态
        self._is_root = True
        
        # 获取根模块
        root_module = self._module
        
        # 遍历根模块中的所有命名模块
        for module_name, module in root_module.named_modules():
            # 获取模块对应的FSDP状态
            if (state := _get_module_fsdp_state(module)) is None:
                continue
            
            # 如果当前模块不是根模块
            if module is not root_module:
                # 如果该模块的状态已经被初始化过，则抛出运行时异常
                if state._is_root is not None:
                    raise RuntimeError(
                        "FSDP state has already been lazily initialized for "
                        f"{module_name}\nFSDP requires running forward through "
                        "the root module first"
                    )
                # 否则将该模块的状态标记为非根状态
                state._is_root = False
            
            # 将当前状态添加到所有状态的上下文中
            self._state_ctx.all_states.append(state)
        
        # 如果存在FSDP参数组，则设置后向传播网格信息为None，以确保参数在训练后被释放和全局聚合
        if self._fsdp_param_group:
            self._fsdp_param_group.post_forward_mesh_info = None
        
        # 初始化FQNs（全限定名）
        self._init_fqns()
        
        # 初始化共享状态
        self._init_shared_state()
        
        # 对于每个状态，运行参数组的惰性初始化，以改善错误消息
        for state in self._state_ctx.all_states:
            if state._fsdp_param_group:
                state._fsdp_param_group.lazy_init()

    def _init_shared_state(self) -> None:
        # 初始化通信上下文的惰性初始化
        self._comm_ctx.lazy_init()
        
        # 遍历所有状态，并将状态上下文和通信上下文分配给各个状态
        for state in self._state_ctx.all_states:
            state._state_ctx = self._state_ctx
            state._comm_ctx = self._comm_ctx
            
            # 如果状态存在FSDP参数组，则将通信上下文分配给参数组的通信上下文
            if fsdp_param_group := state._fsdp_param_group:
                fsdp_param_group.comm_ctx = self._comm_ctx

    def _init_fqns(self) -> None:
        """Sets module and parameter FQN attributes for debugging."""
        # 确保当前实例为根
        assert self._is_root
        
        # 获取根模块
        root_module = self._module
        
        # 初始化参数到FSDP参数和模块到FSDP参数组的映射字典
        param_to_fsdp_param: Dict[nn.Parameter, FSDPParam] = {}
        module_to_fsdp_param_group: Dict[nn.Module, FSDPParamGroup] = {}
        
        # 遍历所有状态
        for state in self._state_ctx.all_states:
            # 如果状态存在FSDP参数组
            if fsdp_param_group := state._fsdp_param_group:
                # 遍历FSDP参数组中的所有FSDP参数，并将参数映射到FSDP参数
                for fsdp_param in fsdp_param_group.fsdp_params:
                    param_to_fsdp_param[fsdp_param.sharded_param] = fsdp_param
                # 将模块映射到FSDP参数组
                module_to_fsdp_param_group[fsdp_param_group.module] = fsdp_param_group
        
        # 遍历根模块中的所有命名参数
        for param_name, param in root_module.named_parameters():
            # 如果参数存在于参数到FSDP参数映射中，则设置参数的参数全限定名属性
            if param in param_to_fsdp_param:
                param_to_fsdp_param[param]._param_fqn = param_name
        
        # 遍历根模块中的所有命名模块
        for module_name, module in root_module.named_modules():
            # 如果模块存在于模块到FSDP参数组映射中，则设置模块的模块全限定名属性
            if module in module_to_fsdp_param_group:
                module_to_fsdp_param_group[module]._module_fqn = module_name
    @disable_if_config_true
    def _pre_forward(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        # 如果当前处于预反向传播状态，直接返回传入的参数
        if self._training_state == TrainingState.PRE_BACKWARD:
            return args, kwargs
        # 将训练状态设置为前向传播
        self._training_state = TrainingState.FORWARD
        # 调用根前向传播方法对输入进行处理
        args, kwargs = self._root_pre_forward(module, args, kwargs)
        # 如果需要，将输入参数转换为指定的数据类型
        if self._mp_policy.cast_forward_inputs and self._mp_policy.param_dtype:
            with torch.profiler.record_function("FSDP::cast_forward_inputs"):
                cast_fn = functools.partial(
                    _cast_fp_tensor, self._mp_policy.param_dtype
                )
                args, kwargs = tree_map(cast_fn, args), tree_map(cast_fn, kwargs)
        # 如果存在参数组，调用其前向传播方法
        if self._fsdp_param_group:
            args, kwargs = self._fsdp_param_group.pre_forward(module, args, kwargs)
        # 对于需要提前预取数据的状态，执行数据预取操作
        for fsdp_state in self._states_to_forward_prefetch:
            if (target_param_group := fsdp_state._fsdp_param_group) is not None:
                FSDPParamGroup._prefetch_unshard(target_param_group, "forward")
        # 返回处理后的参数和关键字参数
        return args, kwargs

    @disable_if_config_true
    def _post_forward(self, module: nn.Module, input: Any, output: Any) -> Any:
        # 如果当前处于预反向传播状态，直接返回输出
        if self._training_state == TrainingState.PRE_BACKWARD:
            return output
        # 如果存在参数组，调用其后向传播方法
        if self._fsdp_param_group:
            output = self._fsdp_param_group.post_forward(module, input, output)
        # 注册前向传播钩子
        output = self._register_pre_backward_hook(output)
        # 将训练状态设置为空闲
        self._training_state = TrainingState.IDLE
        # 如果当前对象是迭代前向根对象
        if self._state_ctx.iter_forward_root is self:
            # 如果需要，释放最后的全聚合结果
            if all_gather_state := self._comm_ctx.all_gather_state:
                # 等待全聚合复制事件完成
                self._comm_ctx.all_gather_copy_in_stream.wait_event(
                    all_gather_state.event
                )
                # 等待全聚合流事件完成
                self._comm_ctx.all_gather_stream.wait_event(all_gather_state.event)
                # 释放全聚合结果
                self._comm_ctx.all_gather_state = None
            # 清空迭代前向根对象
            self._state_ctx.iter_forward_root = None
        # 如果需要，将输出转换为指定的数据类型
        if self._mp_policy.output_dtype is not None:
            with torch.profiler.record_function("FSDP::cast_forward_outputs"):
                output = tree_map(
                    functools.partial(_cast_fp_tensor, self._mp_policy.output_dtype),
                    output,
                )
        # 返回处理后的输出
        return output
    # 设置训练状态为预反向传播阶段
    def _pre_backward(self, grad: torch.Tensor) -> torch.Tensor:
        self._training_state = TrainingState.PRE_BACKWARD
        # 注册根后向传播最终回调函数
        self._register_root_post_backward_final_callback()
        # 如果存在FSDP参数组，则执行预反向传播
        if self._fsdp_param_group:
            # 默认预取所有需要反向传播的状态
            default_prefetch = len(self._states_to_backward_prefetch) == 0
            self._fsdp_param_group.pre_backward(default_prefetch)
        # 针对需要预取的状态执行反向传播
        for fsdp_state in self._states_to_backward_prefetch:
            # 如果存在目标参数组，则进行未分片的预取操作
            if (target_param_group := fsdp_state._fsdp_param_group) is not None:
                FSDPParamGroup._prefetch_unshard(target_param_group, "backward")
        # 返回梯度
        return grad

    # 注册根后向传播最终回调函数
    def _root_post_backward_final_callback(self) -> None:
        # 如果未启用编译的自动求导，则记录调试信息
        if not ca.compiled_autograd_enabled:
            logger.debug("FSDP::root_post_backward")
        # 使用Torch Profiler记录函数执行时间
        with torch.profiler.record_function("FSDP::root_post_backward_callback"):
            # 遍历所有状态上下文
            for state in self._state_ctx.all_states:
                # 如果状态具有FSDP参数组并且未分片，则执行后向传播
                if state._fsdp_param_group and state._fsdp_param_group.is_unsharded:
                    state._fsdp_param_group.post_backward()
                # 将状态设置为IDLE（空闲）
                state._training_state = TrainingState.IDLE
                # 如果状态具有FSDP参数组，则将其训练状态设置为空闲
                if state._fsdp_param_group:
                    state._fsdp_param_group._training_state = TrainingState.IDLE
                # 如果是最后一次反向传播，则执行最终化后向传播
                if self._state_ctx.is_last_backward:
                    state._finalize_backward()
            # 如果是最后一次反向传播，则清空后向传播队列和通信上下文的状态
            if self._state_ctx.is_last_backward:
                self._comm_ctx.post_forward_order.clear()
                self._comm_ctx.reduce_scatter_state = None
            # 标记后向传播最终回调已排队完成
            self._state_ctx.post_backward_final_callback_queued = False

    # 最终化后向传播
    def _finalize_backward(self) -> None:
        # 如果存在FSDP参数组，则执行最终化后向传播
        if self._fsdp_param_group:
            self._fsdp_param_group.finalize_backward()

    # 注册预反向传播钩子
    def _register_pre_backward_hook(self, output: Any) -> Any:
        # 如果未启用梯度，则直接返回输出
        if not torch.is_grad_enabled():
            return output
        # 扁平化输出和空间
        flat_outputs, _ = tree_flatten(output)
        # 提取所有需要梯度的张量
        tensors = tuple(
            t for t in flat_outputs if (torch.is_tensor(t) and t.requires_grad)
        )
        # 如果存在张量，则为每个张量注册预反向传播钩子
        if tensors:
            for tensor in tensors:
                tensor.register_hook(self._pre_backward)
        # 返回输出
        return output

    # 注册根后向传播最终回调函数
    def _register_root_post_backward_final_callback(self):
        # 如果已经排队后向传播最终回调，则直接返回
        if self._state_ctx.post_backward_final_callback_queued:
            return
        # 标记已经排队后向传播最终回调
        self._state_ctx.post_backward_final_callback_queued = True
        # 使用执行引擎队列回调函数
        Variable._execution_engine.queue_callback(
            self._root_post_backward_final_callback
        )
# 获取给定神经网络模块的 FSDPState 状态对象（如果存在的话）
def _get_module_fsdp_state(module: nn.Module) -> Optional[FSDPState]:
    # 获取给定模块的状态
    state = _get_module_state(module)
    # 如果状态对象是 FSDPState 类型，则返回该状态对象
    if isinstance(state, FSDPState):
        return state
    # 如果状态对象不是 FSDPState 类型，则返回 None
    return None
```