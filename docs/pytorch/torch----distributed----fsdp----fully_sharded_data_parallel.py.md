# `.\pytorch\torch\distributed\fsdp\fully_sharded_data_parallel.py`

```
# 忽略 mypy 类型检查错误
# 导入必要的库和模块
import contextlib  # 提供上下文管理器的工具函数
import copy  # 提供对象拷贝操作
import functools  # 提供高阶函数操作工具
import math  # 提供数学函数
import traceback  # 提供异常跟踪功能
import warnings  # 提供警告管理功能
from contextlib import contextmanager  # 提供上下文管理器装饰器
from enum import auto, Enum  # 提供枚举类型的支持
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)  # 导入各种类型提示

import torch  # PyTorch 深度学习库
import torch.distributed as dist  # 分布式通信模块
import torch.distributed.fsdp._traversal_utils as traversal_utils  # FSDP 分布式通信工具
import torch.nn as nn  # PyTorch 神经网络模块
from torch.distributed._tensor import DeviceMesh  # 分布式张量支持
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    _CHECKPOINT_WRAPPED_MODULE,
    ActivationWrapper,
)  # 分布式算法的检查点包装器
from torch.distributed.algorithms._comm_hooks import LOW_PRECISION_HOOKS  # 低精度通信钩子
from torch.distributed.fsdp._common_utils import (
    _FSDPState,
    _get_param_to_fqns,
    FSDP_PREFIX,
    FSDP_WRAPPED_MODULE,
    HandleTrainingState,
    TrainingState,
)  # FSDP 分布式通信的公共工具
from torch.distributed.fsdp._dynamo_utils import _annotate_modules_for_dynamo  # FSDP 动力学工具
from torch.distributed.fsdp._init_utils import (
    _check_orig_params_flattened,
    _init_buffer_state,
    _init_core_state,
    _init_device_handle,
    _init_extension,
    _init_ignored_module_states,
    _init_param_handle_from_module,
    _init_prefetching_state,
    _init_process_group_state,
    _init_runtime_state,
    _init_state_dict_state,
    HYBRID_SHARDING_STRATEGIES,
    ProcessGroupType,
)  # FSDP 初始化工具
from torch.distributed.fsdp._runtime_utils import (
    _get_fsdp_root_states,
    _is_fsdp_root,
    _lazy_init,
    _post_forward,
    _post_forward_reshard,
    _pre_forward,
    _pre_forward_unshard,
    _root_pre_forward,
    _unshard,
    _wait_for_computation_stream,
)  # FSDP 运行时工具
from torch.distributed.fsdp._wrap_utils import _auto_wrap  # FSDP 包装工具
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    LocalOptimStateDictConfig,
    LocalStateDictConfig,
    MixedPrecision,
    OptimStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictConfig,
    StateDictSettings,
    StateDictType,
)  # FSDP API 接口
from torch.distributed.utils import _p_assert  # 分布式工具

from ._flat_param import FlatParameter, FlatParamHandle  # 扁平参数和参数句柄
from ._optim_utils import (
    _flatten_optim_state_dict,
    _get_param_id_to_param_from_optim_input,
    _get_param_key_to_param,
    _get_param_to_param_id_from_optim_input,
    _get_param_to_param_key,
    _optim_state_dict,
    _rekey_sharded_optim_state_dict,
    _set_optim_use_dtensor,
)  # 优化器工具函数
from ._state_dict_utils import _register_all_state_dict_hooks  # 状态字典工具
from ._unshard_param_utils import (
    _deregister_orig_params,
    _register_flat_param,
    _register_orig_params,
    _unshard_params,
    _unshard_params_for_summon,
)  # 参数解碎工具
from .wrap import CustomPolicy, ModuleWrapPolicy  # 自定义包装策略

__all__ = [
    "FullyShardedDataParallel",
    "OptimStateKeyType",
]  # 导出模块中的公共接口名称

FLAT_PARAM = "_flat_param"  # 扁平参数名称

class OptimStateKeyType(Enum):
    """代表优化器状态字典中键的类型。"""
    # 定义一个枚举成员，名称为 PARAM_NAME，自动分配值
    PARAM_NAME = auto()
    # 定义另一个枚举成员，名称为 PARAM_ID，自动分配值（自动递增）
    PARAM_ID = auto()
class FullyShardedDataParallel(nn.Module, _FSDPState):
    """A wrapper for sharding module parameters across data parallel workers.

    This is inspired by `Xu et al.`_ as well as the ZeRO Stage 3 from DeepSpeed_.
    FullyShardedDataParallel is commonly shortened to FSDP.

    .. _`Xu et al.`: https://arxiv.org/abs/2004.13336
    .. _DeepSpeed: https://www.deepspeed.ai/

    To understand FSDP internals, refer to the
    :ref:`fsdp_notes`.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> import torch
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> torch.cuda.set_device(device_id)
        >>> sharded_module = FSDP(my_module)
        >>> optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        >>> x = sharded_module(x, y=3, z=torch.Tensor([1]))
        >>> loss = x.sum()
        >>> loss.backward()
        >>> optim.step()

    Using FSDP involves wrapping your module and then initializing your
    optimizer after. This is required since FSDP changes the parameter
    variables.

    When setting up FSDP, you need to consider the destination CUDA
    device. If the device has an ID (``dev_id``), you have three options:

    * Place the module on that device
    * Set the device using ``torch.cuda.set_device(dev_id)``
    * Pass ``dev_id`` into the ``device_id`` constructor argument.

    This ensures that the FSDP instance's compute device is the
    destination device. For option 1 and 3, the FSDP initialization
    always occurs on GPU. For option 2, the FSDP initialization
    happens on module's current device, which may be a CPU.

    If you're using the ``sync_module_states=True`` flag, you need to
    ensure that the module is on a GPU or use the ``device_id``
    argument to specify a CUDA device that FSDP will move the module
    to in the FSDP constructor. This is necessary because
    ``sync_module_states=True`` requires GPU communication.

    FSDP also takes care of moving input tensors to the forward method
    to the GPU compute device, so you don't need to manually move them
    from CPU.

    For ``use_orig_params=True``,
    ``ShardingStrategy.SHARD_GRAD_OP`` exposes the unsharded
    parameters, not the sharded parameters after forward, unlike
    ``ShardingStrategy.FULL_SHARD``. If you want
    to inspect the gradients, you can use the ``summon_full_params``
    method with ``with_grads=True``.

    With ``limit_all_gathers=True``, you may see a gap in the FSDP
    pre-forward where the CPU thread is not issuing any kernels. This is
    intentional and shows the rate limiter in effect. Synchronizing the CPU
    thread in that way prevents over-allocating memory for subsequent
    all-gathers, and it should not actually delay GPU kernel execution.

    FSDP replaces managed modules' parameters with ``torch.Tensor``
    views during forward and backward computation for autograd-related
    """

    def __init__(self, module, **kwargs):
        """
        Initialize the FullyShardedDataParallel wrapper for a given module.

        Args:
            module (nn.Module): The module to be wrapped and parallelized.
            **kwargs: Additional keyword arguments for configuration.
        """
        super().__init__()
        # Initialize the FSDP state
        _FSDPState.__init__(self, module, **kwargs)

    def _register_grad_hook(self, param):
        """
        Register a gradient hook for a parameter to handle gradient accumulation.

        Args:
            param (torch.Tensor): The parameter tensor to register the hook for.
        """
        if param.requires_grad:
            # Register a hook to accumulate gradients across all ranks
            param.register_hook(lambda grad: grad.clone(memory_format=torch.contiguous_format))
    reasons. If your module's forward relies on saved references to
    the parameters instead of reacquiring the references each
    iteration, then it will not see FSDP's newly created views,
    and autograd will not work correctly.

    Finally, when using ``sharding_strategy=ShardingStrategy.HYBRID_SHARD``
    with the sharding process group being intra-node and the
    replication process group being inter-node, setting
    ``NCCL_CROSS_NIC=1`` can help improve the all-reduce times over
    the replication process group for some cluster setups.

    **Limitations**

    There are several limitations to be aware of when using FSDP:

    * FSDP currently does not support gradient accumulation outside
      ``no_sync()`` when using CPU offloading. This is because FSDP
      uses the newly-reduced gradient instead of accumulating with any
      existing gradient, which can lead to incorrect results.

    * FSDP does not support running the forward pass of a submodule
      that is contained in an FSDP instance. This is because the
      submodule's parameters will be sharded, but the submodule itself
      is not an FSDP instance, so its forward pass will not all-gather
      the full parameters appropriately.

    * FSDP does not work with double backwards due to the way it
      registers backward hooks.

    * FSDP has some constraints when freezing parameters.
      For ``use_orig_params=False``, each FSDP instance must manage
      parameters that are all frozen or all non-frozen. For
      ``use_orig_params=True``, FSDP supports mixing frozen and
      non-frozen parameters, but it's recommended to avoid doing so to
      prevent higher than expected gradient memory usage.

    * As of PyTorch 1.12, FSDP offers limited support for shared
      parameters. If enhanced shared parameter support is needed for
      your use case, please post in
      `this issue <https://github.com/pytorch/pytorch/issues/77724>`__.

    * You should avoid modifying the parameters between forward and
      backward without using the ``summon_full_params`` context, as
      the modifications may not persist.
    def __init__(
        self,
        module: nn.Module,
        process_group: ProcessGroupType = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
        cpu_offload: Optional[CPUOffload] = None,
        auto_wrap_policy: Optional[
            Union[Callable, ModuleWrapPolicy, CustomPolicy]
        ] = None,
        backward_prefetch: Optional[BackwardPrefetch] = BackwardPrefetch.BACKWARD_PRE,
        mixed_precision: Optional[MixedPrecision] = None,
        ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
        param_init_fn: Optional[Callable[[nn.Module], None]] = None,
        device_id: Optional[Union[int, torch.device]] = None,
        sync_module_states: bool = False,
        forward_prefetch: bool = False,
        limit_all_gathers: bool = True,
        use_orig_params: bool = False,
        ignored_states: Union[
            Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]
        ] = None,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        """
        Initialize FSDP (Fully Sharded Data Parallel) wrapper.

        Args:
            module (nn.Module): The module to be wrapped.
            process_group (ProcessGroupType, optional): Process group for communication.
            sharding_strategy (Optional[ShardingStrategy], optional): Sharding strategy.
            cpu_offload (Optional[CPUOffload], optional): CPU offload settings.
            auto_wrap_policy (Optional[Union[Callable, ModuleWrapPolicy, CustomPolicy]], optional):
                Automatic wrapping policy.
            backward_prefetch (Optional[BackwardPrefetch], optional): Backward prefetch strategy.
            mixed_precision (Optional[MixedPrecision], optional): Mixed precision settings.
            ignored_modules (Optional[Iterable[torch.nn.Module]], optional): Ignored modules.
            param_init_fn (Optional[Callable[[nn.Module], None]], optional): Parameter initialization function.
            device_id (Optional[Union[int, torch.device]], optional): Device ID for placement.
            sync_module_states (bool, optional): Flag to sync module states.
            forward_prefetch (bool, optional): Flag for forward prefetching.
            limit_all_gathers (bool, optional): Flag to limit all_gathers.
            use_orig_params (bool, optional): Flag to use original parameters.
            ignored_states (Union[Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]], optional):
                Ignored states.
            device_mesh (Optional[DeviceMesh], optional): Device mesh for distributed setup.
        """

@property
    def module(self) -> nn.Module:
        """Return the wrapped module."""
        # FSDP's `.module` must refer to the innermost wrapped module when
        # composing with other module wrappers in order for state dict to work
        if isinstance(self._fsdp_wrapped_module, ActivationWrapper):
            return getattr(self._fsdp_wrapped_module, _CHECKPOINT_WRAPPED_MODULE)
        return self._fsdp_wrapped_module

@property
    def _has_params(self) -> bool:
        """Returns whether this FSDP instance manages any parameters."""
        return hasattr(self, "_handle") and self._handle is not None

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self._fsdp_wrapped_module, name)

    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is an ``nn.Sequential``."""
        if hasattr(self, FSDP_WRAPPED_MODULE):
            return self._fsdp_wrapped_module.__getitem__(key)  # type: ignore[operator]
        return super().__getitem__(key)

    def check_is_root(self) -> bool:
        """Check if this instance is a root FSDP module."""
        return _is_fsdp_root(self, self)

    @staticmethod
    def fsdp_modules(
        module: nn.Module,
        root_only: bool = False,
    ):
        """
        Static method to retrieve all modules within the FSDP hierarchy.

        Args:
            module (nn.Module): The module to start traversing from.
            root_only (bool, optional): Flag to restrict to root modules only.
        """
    def _get_all_fsdp_instances(
        self, module: torch.nn.Module, root_only: bool = False
    ) -> List["FullyShardedDataParallel"]:
        """Return all nested FSDP instances.

        This method traverses through the input module and collects all instances
        of FullyShardedDataParallel (FSDP) modules. If root_only is set to True,
        only the root FSDP modules are returned.

        Args:
            module (torch.nn.Module): Root module to start the traversal.
            root_only (bool): Flag indicating whether to return only root FSDP modules.
                (Default: False)

        Returns:
            List[FullyShardedDataParallel]: List of FSDP modules nested within the
            input module.
        """
        if root_only:
            return _get_fsdp_root_states(module)
        return traversal_utils._get_fsdp_states(module)

    def apply(self, fn: Callable[[nn.Module], None]) -> "FullyShardedDataParallel":
        r"""Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

        This method applies the given function ``fn`` recursively to every submodule
        within the current module instance, including itself. It ensures that the
        function is also applied to all nested modules.

        Typical use includes initializing the parameters of a model (see also :ref:`nn-init-doc`).

        Compared to ``torch.nn.Module.apply``, this version additionally gathers
        the full parameters before applying ``fn``. It should not be called from
        within another ``summon_full_params`` context.

        Args:
            fn (:class:`Module` -> None): Function to be applied to each submodule.

        Returns:
            FullyShardedDataParallel: The current instance after applying the function.
        """
        uninitialized = self._is_root is None
        self._assert_state(TrainingState.IDLE)
        # Use `_unshard_params_for_summon()` with `recurse=False` instead of
        # `_unshard_fsdp_state_params()` directly to perform lazy
        # initialization, which is needed to initialize `FlatParameter`
        # parameter attributes as required by the unshard logic
        with _unshard_params_for_summon(
            self,
            self,
            writeback=True,
            rank0_only=False,
            offload_to_cpu=False,
            with_grads=False,
        ):
            ret = super().apply(fn)

        # Reset lazy init called in `_unshard_params_for_summon()` since
        # `apply()` may have been called on FSDP instance that is not truly a
        # root, in which case it will be incorrectly marked as one.
        if uninitialized and self._is_root:
            for module in traversal_utils._get_fsdp_states(self):
                module._reset_lazy_init()

        return ret

    def _mixed_precision_enabled_for_buffers(self) -> bool:
        """Return whether the user explicitly enabled buffer mixed precision.

        This method checks whether buffer mixed precision has been explicitly
        enabled at the instance level of FullyShardedDataParallel (FSDP). Unlike
        parameters and gradient reduction, buffer mixed precision applies to
        buffers (non-parameter tensors) managed by FSDP instances.

        NOTE: Unlike parameters and gradient reduction, buffer mixed precision
        is applied at the FSDP instance level, not the ``FlatParameter`` level,
        which may be different for the composable code path.
        
        Returns:
            bool: True if buffer mixed precision is explicitly enabled, False otherwise.
        """
        return self.mixed_precision.buffer_dtype is not None
    # 检查是否已注册低精度钩子
    def _low_precision_hook_enabled(self) -> bool:
        """Whether a low precision hook is registered or not."""
        return self._comm_hook is not None and self._comm_hook in LOW_PRECISION_HOOKS

    # 重置实例状态，以便在下次前向传播时运行 `_lazy_init`
    def _reset_lazy_init(self) -> None:
        """Reset instance so :func:`_lazy_init` will run on the next forward."""
        self._is_root: Optional[bool] = None

    # 设置给定模块的状态字典类型及相关配置
    @staticmethod
    def set_state_dict_type(
        module: nn.Module,
        state_dict_type: StateDictType,
        state_dict_config: Optional[StateDictConfig] = None,
        optim_state_dict_config: Optional[OptimStateDictConfig] = None,
    ):
        pass

    # 获取根据模块 `module` 的 FSDP 模块获取的状态字典类型及对应的配置
    @staticmethod
    def get_state_dict_type(module: nn.Module) -> StateDictSettings:
        """Get the state_dict_type and the corresponding configurations for the FSDP modules rooted at ``module``.

        The target module does not have to be an FSDP module.

        Returns:
            A ``StateDictSettings`` containing the state_dict_type and
            state_dict / optim_state_dict configs that are currently set.

        Raises:
            ``AssertionError`` if the ``StateDictSettings`` for different
            FSDP submodules differ.
        """
        state_dict_settings: Optional[StateDictSettings] = None
        # 遍历所有的 FSDP 模块
        for submodule in FullyShardedDataParallel.fsdp_modules(module):
            if state_dict_settings is None:
                # 第一个子模块的状态字典设置
                state_dict_settings = StateDictSettings(
                    state_dict_type=submodule._state_dict_type,
                    state_dict_config=submodule._state_dict_config,
                    optim_state_dict_config=submodule._optim_state_dict_config,
                )
                # 设置优化器是否使用 dtensor
                _set_optim_use_dtensor(submodule, state_dict_settings)
            else:
                # 后续子模块的状态字典设置
                submodule_settings = StateDictSettings(
                    submodule._state_dict_type,
                    submodule._state_dict_config,
                    submodule._optim_state_dict_config,
                )
                # 断言所有 FSDP 模块的状态字典设置必须相同
                assert state_dict_settings == submodule_settings, (
                    "All FSDP modules must have the same state dict settings."
                    f"Got {submodule_settings} and {state_dict_settings}."
                )
                # 设置优化器是否使用 dtensor
                _set_optim_use_dtensor(submodule, submodule_settings)
        return state_dict_settings

    # 上下文管理器，用于设置给定模块的状态字典类型及相关配置
    @staticmethod
    @contextlib.contextmanager
    def state_dict_type(
        module: nn.Module,
        state_dict_type: StateDictType,
        state_dict_config: Optional[StateDictConfig] = None,
        optim_state_dict_config: Optional[OptimStateDictConfig] = None,
    ):
        pass
    # 定义一个静态方法，用于设置目标模块及其所有后代 FSDP 模块的 state_dict_type。
    
    def set_state_dict_type(
        module: nn.Module,
        state_dict_type: StateDictType,
        state_dict_config: Optional[StateDictConfig],
        optim_state_dict_config: Optional[OptimStateDictConfig],
    ) -> Generator:
        """
        Set the ``state_dict_type`` of all the descendant FSDP modules of the target module.
    
        This context manager has the same functions as :meth:`set_state_dict_type`. Read the document of
        :meth:`set_state_dict_type` for the detail.
    
        Example::
    
            >>> # xdoctest: +SKIP("undefined variables")
            >>> model = DDP(FSDP(...))
            >>> with FSDP.state_dict_type(
            >>>     model,
            >>>     StateDictType.SHARDED_STATE_DICT,
            >>> ):
            >>>     checkpoint = model.state_dict()
    
        Args:
            module (torch.nn.Module): Root module.
            state_dict_type (StateDictType): the desired ``state_dict_type`` to set.
            state_dict_config (Optional[StateDictConfig]): the model ``state_dict``
                configuration for the target ``state_dict_type``.
            optim_state_dict_config (Optional[OptimStateDictConfig]): the optimizer
               ``state_dict`` configuration for the target ``state_dict_type``.
        """
        # 调用 FullyShardedDataParallel 的静态方法设置 state_dict_type，并保存之前的设置
        prev_state_dict_settings = FullyShardedDataParallel.set_state_dict_type(
            module,
            state_dict_type,
            state_dict_config,
            optim_state_dict_config,
        )
        # 返回一个生成器对象
        yield
        # 恢复之前的 state_dict_type 设置
        FullyShardedDataParallel.set_state_dict_type(
            module,
            prev_state_dict_settings.state_dict_type,
            prev_state_dict_settings.state_dict_config,
            prev_state_dict_settings.optim_state_dict_config,
        )
    
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the forward pass for the wrapped module, inserting FSDP-specific pre- and post-forward sharding logic."""
        # 获取句柄对象
        handle = self._handle
        # 使用 Torch 自动求导分析器记录函数 "FullyShardedDataParallel.forward"
        with torch.autograd.profiler.record_function(
            "FullyShardedDataParallel.forward"
        ):
            # 执行前向传播前的 FSDP 特定逻辑
            args, kwargs = _root_pre_forward(self, self, args, kwargs)
            unused = None
            # 执行前向传播前的预处理逻辑
            args, kwargs = _pre_forward(
                self,
                handle,
                _pre_forward_unshard,
                self._fsdp_wrapped_module,
                args,
                kwargs,
            )
            # 如果有句柄对象，则进行断言验证
            if handle:
                _p_assert(
                    handle.flat_param.device == self.compute_device,
                    "Expected `FlatParameter` to be on the compute device "
                    f"{self.compute_device} but got {handle.flat_param.device}",
                )
            # 调用被 FSDP 包装后的模块进行前向传播
            output = self._fsdp_wrapped_module(*args, **kwargs)
            # 执行前向传播后的后处理逻辑
            return _post_forward(
                self, handle, _post_forward_reshard, self, unused, output
            )
    
    
    @staticmethod
    @contextlib.contextmanager
    def summon_full_params(
        module: nn.Module,
        recurse: bool = True,
        writeback: bool = True,
        rank0_only: bool = False,
        offload_to_cpu: bool = False,
        with_grads: bool = False,
    ):
    def _deregister_orig_params_ctx(self):
        """Deregister the original parameters and expose the :class:`FlatParameter`.

        If a :class:`FlatParameter` is sharded, then
        this refreshes the sharded views before exiting. This method should
        only be called when using the original parameters.
        """
        # 断言是否在使用原始参数，如果不是则抛出异常
        _p_assert(
            self._use_orig_params,
            "`_deregister_orig_params_ctx()` should only be called when "
            "`_use_orig_params=True`",
        )
        # 遍历所有 FSDP 模块状态，并取消注册原始参数
        for fsdp_module in traversal_utils._get_fsdp_states(self):
            _deregister_orig_params(fsdp_module, fsdp_module)
        try:
            # 返回一个上下文管理器的生成器
            yield
        finally:
            # 遍历所有 FSDP 模块状态，并重新注册原始参数
            for fsdp_module in traversal_utils._get_fsdp_states(self):
                _register_orig_params(fsdp_module, fsdp_module)

    def _apply(self, *args, **kwargs):
        """Deregister the original parameters and expose the :class:`FlatParameter` s before calling ``_apply()``."""
        # 当使用原始参数时：因为 (1) `FlatParameter` 拥有存储，
        # (2) `_apply()` 是诸如 `to()` 和 `cuda()` 等最常见的更改存储操作的子程序，
        # 我们重写 `_apply()` 以便直接在 `FlatParameter` 上执行存储更改，而不是应用于原始参数，然后写回到 `FlatParameter`。
        context = (
            self._deregister_orig_params_ctx()
            if self._use_orig_params
            else contextlib.nullcontext()
        )
        with context:
            # 调用父类的 `_apply()` 方法，并传递所有参数和关键字参数
            return super()._apply(*args, **kwargs)

    def named_buffers(
        self,
        *args,
        **kwargs,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Return an iterator over module buffers, yielding both the name of the buffer and the buffer itself.

        Intercepts buffer names and removes all occurrences of the FSDP-specific flattened buffer prefix
        when inside the :meth:`summon_full_params` context manager.
        """
        # 当处于 `summon_full_params` 上下文管理器中时，应清除缓冲区名中的 FSDP 特定前缀的所有出现
        should_clean_name = self.training_state == TrainingState.SUMMON_FULL_PARAMS
        # 遍历父类的命名缓冲区方法返回的迭代器
        for buffer_name, buffer in super().named_buffers(*args, **kwargs):
            if should_clean_name:
                # 移除缓冲区名中的 FSDP 特定前缀的所有实例；在嵌套 FSDP 模块的情况下可能会有多个
                buffer_name = buffer_name.replace(FSDP_PREFIX, "")
            yield (buffer_name, buffer)

    def named_parameters(
        self,
        *args,
        **kwargs,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Return an iterator over module parameters, yielding both the name of the parameter and the parameter itself."""
        # 调用父类的命名参数方法，返回一个迭代器，产生参数名和参数本身
        return super().named_parameters(*args, **kwargs)
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        """Return an iterator over module parameters, yielding both the name of the parameter and the parameter itself.

        Intercepts parameter names and removes all occurrences of the FSDP-specific flattened parameter prefix
        when inside the :meth:`summon_full_params` context manager.
        """
        # 检查当前是否应该清理参数名（仅在处于 SUMMON_FULL_PARAMS 训练状态时）
        should_clean_name = self.training_state == TrainingState.SUMMON_FULL_PARAMS
        # 遍历父类的命名参数，并返回参数名及其对应的参数对象
        for param_name, param in super().named_parameters(*args, **kwargs):
            if should_clean_name:
                # 如果应该清理参数名，则移除所有 FSDP 特定的前缀
                param_name = param_name.replace(FSDP_PREFIX, "")
            yield (param_name, param)

    def _assert_state(self, state: Union[TrainingState, List[TrainingState]]) -> None:
        """Assert we are in the given state."""
        # 因为 assert 可能被关闭，而这里的错误检查非常重要，所以我们使用显式的错误检查，如果需要的话会抛出 ValueError 异常。
        if isinstance(state, TrainingState):
            state = [state]
        # 检查当前对象的训练状态是否在给定状态列表中
        if self.training_state not in state:
            msg = (
                f"expected to be in states {state} but current state "
                f"is {self.training_state}"
            )
            # 如果在 autograd hook 的上下文中失败，断言可能不会生成有用的消息，因此打印消息以确保可见性。
            if self.rank == 0:
                print(f"Asserting FSDP instance is: {self}")
                print(f"ERROR: {msg}")
                traceback.print_stack()
            raise ValueError(msg)

    @contextmanager


这段代码包含两个方法的实现以及一个上下文管理器的声明，分别进行了详细的注释说明。
    def no_sync(self) -> Generator:
        """Disable gradient synchronizations across FSDP instances.

        Within this context, gradients will be accumulated in module
        variables, which will later be synchronized in the first
        forward-backward pass after exiting the context. This should only be
        used on the root FSDP instance and will recursively apply to all
        children FSDP instances.

        .. note:: This likely results in higher memory usage because FSDP will
            accumulate the full model gradients (instead of gradient shards)
            until the eventual sync.

        .. note:: When used with CPU offloading, the gradients will not be
            offloaded to CPU when inside the context manager. Instead, they
            will only be offloaded right after the eventual sync.
        """
        # 初始化 FSDP 实例
        _lazy_init(self, self)
        # 如果不是根节点，则抛出运行时错误
        if not self._is_root:
            raise RuntimeError(
                "`no_sync()` on inner FSDP instances is not supported. Please call `no_sync()` on root FSDP module."
            )
        # 断言当前状态为 IDLE
        self._assert_state(TrainingState.IDLE)
        # 存储旧的同步标志
        old_flags = []
        # 遍历所有模块
        for m in self.modules():
            # 如果模块是 FullyShardedDataParallel 类型
            if isinstance(m, FullyShardedDataParallel):
                # 存储当前的同步标志，并将其设置为 False
                old_flags.append((m, m._sync_gradients))
                m._sync_gradients = False
        try:
            # 返回一个生成器对象，用于执行后续操作
            yield
        finally:
            # 恢复各模块的同步标志，并进行断言检查
            for m, old_flag in old_flags:
                assert not m._sync_gradients, (
                    "`_sync_gradients` was incorrectly set to "
                    "`True` while in the `no_sync()` context manager"
                )
                m._sync_gradients = old_flag

    @torch.no_grad()
    def clip_grad_norm_(
        self, max_norm: Union[float, int], norm_type: Union[float, int] = 2.0
    ):
        """Clip the gradients of all model parameters in-place.

        Args:
            max_norm (float or int): The maximum allowed value of the norm of the gradients.
            norm_type (float or int): The type of norm. Default is 2.0, which refers to the L2 norm.

        This function modifies the gradients of each parameter in place. It computes the norm
        of gradients and if the norm exceeds `max_norm`, it scales the gradients so that their
        norm becomes `max_norm`.

        .. note:: This function operates in-place, modifying the gradients of the parameters.
        """
        pass  # Placeholder for implementation

    @staticmethod
    def _warn_optim_input(optim_input, *, stacklevel: int = 1):
        """Issue a warning about the deprecated `optim_input` argument.

        Args:
            optim_input: Deprecated argument that will be removed after PyTorch 1.13.
            stacklevel (int): How many steps above the `_warn_optim_input` call to issue the warning.

        This static method issues a FutureWarning about the deprecated `optim_input` argument
        in PyTorch optimization functions. It advises users to remove this argument from their
        code to ensure compatibility with future versions of PyTorch.
        """
        if optim_input is not None:
            # 发出关于 `optim_input` 参数已弃用的警告
            warnings.warn(
                "The `optim_input` argument is deprecated and will be removed after PyTorch 1.13. "
                "You may remove it from your code without changing its functionality.",
                FutureWarning,
                stacklevel=stacklevel + 1,
            )

    @staticmethod
    def _is_using_optim_input(optim_input, optim) -> bool:
        """Check if the optimizer should use `optim_input` argument.

        Args:
            optim_input: The input argument for the optimizer.
            optim: The optimizer object.

        Returns:
            bool: True if `optim_input` should be used, False otherwise.

        This static method determines whether the optimizer should use the `optim_input` argument
        based on its presence or absence. If `optim_input` is None and `optim` is None, the default
        behavior is used. If `optim_input` is provided, it takes precedence over `optim`.

        """
        if optim_input is None and optim is None:
            # 使用 `optim_input` 的默认行为
            return True
        if optim_input is not None:
            # 使用 `optim_input` 的代码路径
            return True
        # 使用 `optim` 的代码路径
        return False
    def _warn_legacy_optim_state_dict(curr: str, new: str, *, stacklevel: int = 1):
        """发出警告信息，指示某个过时的优化器状态字典的替代方式。

        Args:
            curr (str): 当前过时优化器状态字典的名称。
            new (str): 替代当前优化器状态字典的名称。
            stacklevel (int, optional): 警告信息的堆栈级别，默认为 1。
        """
        warnings.warn(
            f"``FullyShardedDataParallel.{curr}``is being deprecated and is "
            f"replaced by ``FullyShardedDataParallel.{new}``. "
            f"``FullyShardedDataParallel.{curr}`` may be removed after PyTorch 2.2.",
            FutureWarning,
            stacklevel=stacklevel + 1,
        )

    @staticmethod
    def _optim_state_dict_impl(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_state_dict: Dict[str, Any],
        optim_input: Optional[
            Union[
                List[Dict[str, Any]],
                Iterable[torch.nn.Parameter],
            ]
        ] = None,
        rank0_only: bool = True,
        full_state_dict: bool = True,
        group: Optional[dist.ProcessGroup] = None,
        cpu_offload: bool = True,
        *,
        _stacklevel: int = 1,
    ) -> Dict[str, Any]:
        """转换带有分片模型的优化器状态字典。

        这是内部 API，用于所有优化器状态字典实现。给定模型、优化器、原始优化器状态字典，
        此 API 会从优化器状态字典中移除 FSDP 内部信息和内部分片。

        Args:
            model (torch.nn.Module): 被优化的模型。
            optim (torch.optim.Optimizer): 使用的优化器。
            optim_state_dict (Dict[str, Any]): 原始优化器状态字典。
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]], optional):
                优化器的输入参数。默认为 None。
            rank0_only (bool, optional): 是否仅处理排名为 0 的节点。默认为 True。
            full_state_dict (bool, optional): 是否处理完整的状态字典。默认为 True。
            group (Optional[dist.ProcessGroup], optional): 分布式处理组。默认为 None。
            cpu_offload (bool, optional): 是否进行 CPU 卸载。默认为 True。
            _stacklevel (int, optional): 警告信息的堆栈级别。默认为 1。

        Returns:
            Dict[str, Any]: 转换后的优化器状态字典。
        """
        if full_state_dict:
            FullyShardedDataParallel._warn_optim_input(
                optim_input, stacklevel=_stacklevel + 1
            )
            using_optim_input = FullyShardedDataParallel._is_using_optim_input(
                optim_input,
                optim,
            )
        else:
            using_optim_input = False
            assert optim_input is None and not rank0_only

        use_orig_params = FullyShardedDataParallel.fsdp_modules(model)[
            0
        ]._use_orig_params
        assert all(
            use_orig_params == m._use_orig_params
            for m in FullyShardedDataParallel.fsdp_modules(model)
        ), "Not all FSDP modules have the same _use_orig_params value"

        return _optim_state_dict(
            model=model,
            optim=optim,
            optim_state_dict=optim_state_dict,
            optim_input=optim_input,
            rank0_only=rank0_only,
            shard_state=not full_state_dict,
            group=group,
            using_optim_input=using_optim_input,
            use_orig_params=use_orig_params,
            cpu_offload=cpu_offload,
        )

    @staticmethod
    def _optim_state_dict_to_load_impl(
        optim_state_dict: Dict[str, Any],
        model: torch.nn.Module,
        optim_input: Optional[
            Union[
                List[Dict[str, Any]],
                Iterable[torch.nn.Parameter],
            ]
        ] = None,
        optim: Optional[torch.optim.Optimizer] = None,
        full_state_dict: bool = True,
        rank0_only: bool = False,
        is_named_optimizer: bool = False,
        group: Optional[dist.ProcessGroup] = None,
        *,
        _stacklevel: int = 1,
    ) -> Dict[str, Any]:
        """转换用于加载的优化器状态字典，对应于带有分片模型的情况。

        Args:
            optim_state_dict (Dict[str, Any]): 原始优化器状态字典。
            model (torch.nn.Module): 模型。
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]], optional):
                优化器的输入参数。默认为 None。
            optim (Optional[torch.optim.Optimizer], optional): 使用的优化器。默认为 None。
            full_state_dict (bool, optional): 是否处理完整的状态字典。默认为 True。
            rank0_only (bool, optional): 是否仅处理排名为 0 的节点。默认为 False。
            is_named_optimizer (bool, optional): 是否为命名优化器。默认为 False。
            group (Optional[dist.ProcessGroup], optional): 分布式处理组。默认为 None。
            _stacklevel (int, optional): 警告信息的堆栈级别。默认为 1。

        Returns:
            Dict[str, Any]: 转换后的用于加载的优化器状态字典。
        """
    ) -> Dict[str, Any]:
        """
        Convert an optimizer state-dict so that it can be loaded into the optimizer associated with the FSDP model.

        This is the internal API that is used by all the load optim_state_dict implementations.
        Given model, optim, and the saved optim_state_dict, this API adds the FSDP
        internal information and internal sharding to the optim_state_dict.
        """
        # 如果需要完整的状态字典
        if full_state_dict:
            # 警告优化器输入
            FullyShardedDataParallel._warn_optim_input(optim_input)
            # 检查是否正在使用优化器输入
            using_optim_input = FullyShardedDataParallel._is_using_optim_input(
                optim_input,
                optim,
            )
        else:
            # 如果不需要完整的状态字典，确保优化器输入为None且不仅限于rank0
            using_optim_input = False
            assert optim_input is None and not rank0_only

        # 使用原始参数
        use_orig_params = FullyShardedDataParallel.fsdp_modules(model)[
            0
        ]._use_orig_params
        # 确保所有FSDP模块具有相同的_use_orig_params值
        assert all(
            use_orig_params == m._use_orig_params
            for m in FullyShardedDataParallel.fsdp_modules(model)
        ), "Not all FSDP modules have the same _use_orig_params value"

        # 如果仅限于rank0并且当前进程组的rank大于0，则优化状态字典为空
        if rank0_only and dist.get_rank(group) > 0:
            optim_state_dict = {}
        # 展开优化状态字典
        sharded_osd = _flatten_optim_state_dict(
            optim_state_dict,
            model=model,
            use_orig_params=use_orig_params,
            optim=(optim if is_named_optimizer else None),
            rank0_only=rank0_only,
            group=group,
        )
        # 重新调整分片的优化状态字典
        return _rekey_sharded_optim_state_dict(
            sharded_osd,
            model=model,
            optim=optim,
            optim_input=optim_input,
            using_optim_input=using_optim_input,
            is_named_optimizer=is_named_optimizer,
        )

    @staticmethod
    def full_optim_state_dict(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_input: Optional[
            Union[
                List[Dict[str, Any]],
                Iterable[torch.nn.Parameter],
            ]
        ] = None,
        rank0_only: bool = True,
        group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Generate a full optimizer state-dict from the model and optimizer.

        Args:
            model: The PyTorch model.
            optim: The PyTorch optimizer.
            optim_input: Optional input parameters for optimizer.
            rank0_only: Flag indicating if only rank 0 should contribute.
            group: Optional process group for distributed training.

        Returns:
            Dict: The full optimizer state-dict.
        """

    @staticmethod
    def sharded_optim_state_dict(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Generate a sharded optimizer state-dict from the model and optimizer.

        Args:
            model: The PyTorch model.
            optim: The PyTorch optimizer.
            group: Optional process group for distributed training.

        Returns:
            Dict: The sharded optimizer state-dict.
        """
    ) -> Dict[str, Any]:
        """
        返回优化器状态字典的分片形式。

        此 API 类似于 :meth:`full_optim_state_dict`，但此 API 将所有非零维度的状态均分片到 :class:`ShardedTensor` 以节省内存。
        只有在模型 `state_dict` 是使用上下文管理器 `with state_dict_type(SHARDED_STATE_DICT):` 派生的情况下才应使用此 API。

        有关详细用法，请参阅 :meth:`full_optim_state_dict`。

        .. warning:: 返回的状态字典包含 `ShardedTensor`，不能直接被常规的 `optim.load_state_dict` 使用。
        """
        FullyShardedDataParallel._warn_legacy_optim_state_dict(
            "sharded_optim_state_dict",
            "optim_state_dict",
            stacklevel=2,
        )
        return FullyShardedDataParallel._optim_state_dict_impl(
            model=model,
            optim=optim,
            optim_state_dict=optim.state_dict(),
            optim_input=None,
            rank0_only=False,
            full_state_dict=False,
            group=group,
            _stacklevel=2,
        )

    @staticmethod
    def shard_full_optim_state_dict(
        full_optim_state_dict: Dict[str, Any],
        model: torch.nn.Module,
        optim_input: Optional[
            Union[
                List[Dict[str, Any]],
                Iterable[torch.nn.Parameter],
            ]
        ] = None,
        optim: Optional[torch.optim.Optimizer] = None,
    ):
        """
        将完整的优化器状态字典分片。

        Args:
            full_optim_state_dict (Dict[str, Any]): 完整的优化器状态字典。
            model (torch.nn.Module): PyTorch 模型。
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]], optional): 优化器的输入。默认为 None。
            optim (Optional[torch.optim.Optimizer], optional): PyTorch 优化器。默认为 None。
        
        Returns:
            Dict[str, Any]: 分片后的优化器状态字典。
        """
        ...

    @staticmethod
    def flatten_sharded_optim_state_dict(
        sharded_optim_state_dict: Dict[str, Any],
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
    ):
        """
        展平分片的优化器状态字典。

        Args:
            sharded_optim_state_dict (Dict[str, Any]): 分片的优化器状态字典。
            model (torch.nn.Module): PyTorch 模型。
            optim (torch.optim.Optimizer): PyTorch 优化器。
        
        Returns:
            None
        """
        ...
    ) -> Dict[str, Any]:
        """Flatten a sharded optimizer state-dict.

        将分片的优化器状态字典展平。

        The API is similar to :meth:`shard_full_optim_state_dict`. The only
        difference is that the input ``sharded_optim_state_dict`` should be
        returned from :meth:`sharded_optim_state_dict`. Therefore, there will
        be all-gather calls on each rank to gather ``ShardedTensor`` s.

        Args:
            sharded_optim_state_dict (Dict[str, Any]): Optimizer state dict
                corresponding to the unflattened parameters and holding the
                sharded optimizer state.
            model (torch.nn.Module):
                Refer to :meth:`shard_full_optim_state_dict`.
            optim (torch.optim.Optimizer): Optimizer for ``model`` 's
                parameters.

        Returns:
            Refer to :meth:`shard_full_optim_state_dict`.
        """
        FullyShardedDataParallel._warn_legacy_optim_state_dict(
            "flatten_sharded_optim_state_dict",
            "optim_state_dict_to_load",
            stacklevel=2,
        )
        return FullyShardedDataParallel._optim_state_dict_to_load_impl(
            optim_state_dict=sharded_optim_state_dict,
            model=model,
            optim_input=None,
            optim=optim,
            full_state_dict=False,
            is_named_optimizer=False,
        )

    @staticmethod
    def scatter_full_optim_state_dict(
        full_optim_state_dict: Optional[Dict[str, Any]],
        model: torch.nn.Module,
        optim_input: Optional[
            Union[
                List[Dict[str, Any]],
                Iterable[torch.nn.Parameter],
            ]
        ] = None,
        optim: Optional[torch.optim.Optimizer] = None,
        group: Optional[Any] = None,
    ):
        """Scatter the full optimizer state dict.

        Args:
            full_optim_state_dict (Optional[Dict[str, Any]]): Full optimizer state dict.
            model (torch.nn.Module): Model instance.
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
                Input for the optimizer.
            optim (Optional[torch.optim.Optimizer]): Optimizer for model's parameters.
            group (Optional[Any]): Process group for distributed training.

        Returns:
            None
        """

    @staticmethod
    def rekey_optim_state_dict(
        optim_state_dict: Dict[str, Any],
        optim_state_key_type: OptimStateKeyType,
        model: torch.nn.Module,
        optim_input: Optional[
            Union[
                List[Dict[str, Any]],
                Iterable[torch.nn.Parameter],
            ]
        ] = None,
        optim: Optional[torch.optim.Optimizer] = None,
    ):
        """Rekey the optimizer state dict.

        Args:
            optim_state_dict (Dict[str, Any]): Optimizer state dict to rekey.
            optim_state_key_type (OptimStateKeyType): Type of key for rekeying.
            model (torch.nn.Module): Model instance.
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
                Input for the optimizer.
            optim (Optional[torch.optim.Optimizer]): Optimizer for model's parameters.

        Returns:
            None
        """

    @staticmethod
    def optim_state_dict(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_state_dict: Optional[Dict[str, Any]] = None,
        group: Optional[dist.ProcessGroup] = None,
    ):
        """Get the optimizer state dict.

        Args:
            model (torch.nn.Module): Model instance.
            optim (torch.optim.Optimizer): Optimizer for model's parameters.
            optim_state_dict (Optional[Dict[str, Any]]): Existing optimizer state dict to update.
            group (Optional[dist.ProcessGroup]): Process group for distributed training.

        Returns:
            Dict[str, Any]: Optimizer state dict.
        """

    @staticmethod
    def optim_state_dict_to_load(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        optim_state_dict: Dict[str, Any],
        is_named_optimizer: bool = False,
        load_directly: bool = False,
        group: Optional[dist.ProcessGroup] = None,
    ):
        """Prepare the optimizer state dict for loading.

        Args:
            model (torch.nn.Module): Model instance.
            optim (torch.optim.Optimizer): Optimizer for model's parameters.
            optim_state_dict (Dict[str, Any]): Optimizer state dict to prepare for loading.
            is_named_optimizer (bool): Whether the optimizer is named.
            load_directly (bool): Whether to load the optimizer state dict directly.
            group (Optional[dist.ProcessGroup]): Process group for distributed training.

        Returns:
            None
        """
    # 定义一个内部方法 `_unshard`，用于执行参数解聚操作，支持异步操作
    def _unshard(self, async_op: bool = False):
        
        # 内部类 `UnshardHandle`，用于管理解聚操作的处理
        class UnshardHandle:
            
            # 初始化方法，接收平坦参数句柄和解聚事件
            def __init__(
                self,
                flat_param_handle: Optional[FlatParamHandle],
                unshard_event: torch.cuda.Event,
            ):
                self._flat_param_handle = flat_param_handle  # 平坦参数句柄
                self._unshard_event = unshard_event  # 解聚事件

            # 等待方法，如果存在平坦参数句柄，则等待解聚事件完成
            def wait(self):
                if self._flat_param_handle is not None:
                    current_stream = (
                        self._flat_param_handle._device_handle.current_stream()
                    )
                    current_stream.wait_event(self._unshard_event)
                    self._flat_param_handle = None

        # 如果存在处理句柄 `_handle`
        if self._handle:
            # 使用训练状态管理上下文，设置为前向和前向后训练状态
            with self._use_training_state(
                TrainingState.FORWARD_BACKWARD, HandleTrainingState.FORWARD
            ):
                # 调用 `_unshard` 方法执行解聚操作
                _unshard(
                    self, self._handle, self._unshard_stream, self._pre_unshard_stream
                )
                # 记录解聚流的事件
                self._unshard_event = self._unshard_stream.record_event()
            # 设置处理句柄的预取标志为真
            self._handle._prefetched = True
        
        # 创建解聚处理句柄实例
        unshard_handle = UnshardHandle(self._handle, self._unshard_stream)
        
        # 如果是异步操作，直接返回解聚处理句柄
        if async_op:
            return unshard_handle
        
        # 否则，等待解聚处理完成
        unshard_handle.wait()
        
        # 返回空值
        return None

    # 内部方法 `_wait_unshard_streams_on_current_stream`，等待解聚流在当前流上的计算完成
    def _wait_unshard_streams_on_current_stream(self):
        _wait_for_computation_stream(
            self._device_handle.current_stream(),
            self._unshard_stream,
            self._pre_unshard_stream,
        )

    # 上下文管理器 `_use_training_state`，用于管理训练状态的切换
    @contextlib.contextmanager
    def _use_training_state(
        self, training_state: TrainingState, handle_training_state: HandleTrainingState
    ):
        # 保存当前训练状态
        prev_training_state = self.training_state
        self.training_state = training_state
        
        # 如果存在处理句柄 `_handle`，保存之前的处理句柄训练状态
        if self._handle:
            prev_handle_training_state = self._handle._training_state
            self._handle._training_state = handle_training_state
        
        try:
            # 执行上下文管理器中的代码块
            yield
        
        finally:
            # 最终恢复之前的训练状态
            self.training_state = prev_training_state
            
            # 如果存在处理句柄 `_handle`，恢复之前的处理句柄训练状态
            if self._handle:
                self._handle._training_state = prev_handle_training_state
def _get_grad_norm(
    params: Iterable[nn.Parameter],
    norm_type: float,
) -> torch.Tensor:
    """
    返回参数 ``params`` 的梯度范数，其中梯度被视为单个向量。

    返回的范数始终是 FP32，即使参数/梯度使用低精度也是如此。这是因为返回值在后续的使用中会跨多个处理单元进行归约。
    """
    # 获取具有梯度的参数列表
    params_with_grad = [param for param in params if param.grad is not None]
    if len(params_with_grad) == 0:
        return torch.tensor(0.0)
    # 提取梯度列表
    grads = [param.grad for param in params_with_grad]
    # 检查所有梯度的数据类型是否一致
    grad_dtypes = {grad.dtype for grad in grads}
    if len(grad_dtypes) != 1:
        raise ValueError(
            f"Requires uniform dtype across all gradients but got {grad_dtypes}"
        )
    # 计算 FP32 下的梯度范数，将梯度视为单个向量
    grad_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                torch.linalg.vector_norm(grad.detach(), norm_type, dtype=torch.float32)
                for grad in grads
            ],
        ),
        norm_type,
        dtype=torch.float32,
    )
    return grad_norm


def _get_param_to_fqn(
    model: torch.nn.Module,
) -> Dict[torch.nn.Parameter, str]:
    """
    构建从参数到其参数名的映射。

    ``model`` 不应包含任何 :class:`FullyShardedDataParallel` 实例，这意味着没有参数应该是 ``FlatParameter``。
    因此，与 :meth:`_get_param_to_fqns` 相比，映射的值可能会从包含的单一列表展平为包含的参数名本身。

    Args:
        model (torch.nn.Module): 根模块，不应包含任何 :class:`FullyShardedDataParallel` 实例。
    """
    # 调用 _get_param_to_fqns 函数获取参数到参数名列表的映射
    param_to_param_names = _get_param_to_fqns(model)
    # 对于每一个参数名列表，确保列表不为空
    for param_names in param_to_param_names.values():
        assert (
            len(param_names) > 0
        ), "`_get_param_to_fqns()` should not construct empty lists"
        # 如果列表长度大于1，抛出异常
        if len(param_names) > 1:
            raise RuntimeError(
                "Each parameter should only map to one parameter name but got "
                f"{len(param_names)}: {param_names}"
            )
    # 构建从参数到其参数名的映射，取每个参数对应的第一个参数名
    param_to_param_name = {
        param: param_names[0] for param, param_names in param_to_param_names.items()
    }
    return param_to_param_name


def _get_fqn_to_param(
    model: torch.nn.Module,
) -> Dict[str, torch.nn.Parameter]:
    """构建 :meth:`_get_param_to_fqn` 的逆映射。"""
    # 获取从参数到参数名的映射
    param_to_param_name = _get_param_to_fqn(model)
    # 构建从参数名到参数的映射，即逆映射
    return dict(zip(param_to_param_name.values(), param_to_param_name.keys()))
```