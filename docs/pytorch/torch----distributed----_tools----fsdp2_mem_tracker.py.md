# `.\pytorch\torch\distributed\_tools\fsdp2_mem_tracker.py`

```
from copy import deepcopy  # 导入深拷贝函数 deepcopy
from datetime import timedelta  # 导入时间间隔模块中的 timedelta 类
from functools import partial, wraps  # 导入偏函数和装饰器相关函数
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union  # 导入类型提示相关的类型

import torch  # 导入 PyTorch 库
import torch.distributed as dist  # 导入 PyTorch 分布式模块
from torch import nn, optim  # 导入神经网络和优化器相关模块
from torch._guards import active_fake_mode  # 导入内部保护模块
from torch.distributed._composable.fsdp import FSDPModule  # 导入 FSDP 模块
from torch.distributed._composable.fsdp._fsdp_param_group import FSDPParamGroup  # 导入 FSDP 参数组模块
from torch.distributed._tools.mem_tracker import _RefType, _State, MemTracker  # 导入内存追踪相关模块
from torch.distributed.distributed_c10d import (  # 导入分布式 C10d 模块
    _IllegalWork,
    ProcessGroup,
    ReduceOp,
    Work,
)
from torch.futures import Future  # 导入异步 Future 模块
from torch.utils._python_dispatch import TorchDispatchMode  # 导入 Python 分发模式
from torch.utils._pytree import tree_map_only  # 导入 PyTree 模块
from torch.utils.weak import WeakIdKeyDictionary, weakref  # 导入弱引用相关模块

_TOTAL_KEY = "Total"  # 定义总键名 "_TOTAL_KEY"

__all__ = ["FSDPMemTracker"]  # 模块导出的所有对象列表，包括 "FSDPMemTracker"

class _FSDPRefType(_RefType):
    """
    Enumerates categories of memory usage in FSDP modules, including parameters, gradients, activations,
    and optimizer states.

    Attributes:
        SHARDED_PARAM (str): Memory usage of sharded parameters.
        UNSHARDED_PARAM (str): Memory usage of unsharded parameters.
        BUFFER (str): Memory usage of buffer tensors.
        SHARDED_GRAD (str): Memory usage of sharded gradients corresponding to the sharded parameters.
        UNSHARDED_GRAD (str): Memory usage of unsharded gradients corresponding to the unsharded parameters.
        ACT (str): Memory usage of activations and tensors from forward and AC recomputation.
        TEMP (str): Memory usage of temporary tensors during the backward pass including gradients of activations.
        ALL_GATHER (str): Memory usage of all_gather output tensor.
        REDUCE_SCATTER (str): Memory usage of reduce_scatter input tensor.
        OPT (str): Memory usage of tensors storing optimizer states.
        INP (str): Memory usage of input tensors.
    """
    SHARDED_PARAM = "Sharded Param"  # 分片参数的内存使用
    UNSHARDED_PARAM = "Unsharded Param"  # 非分片参数的内存使用
    BUFFER = "Buffer"  # 缓冲张量的内存使用
    SHARDED_GRAD = "Sharded Grad"  # 分片参数对应的梯度的内存使用
    UNSHARDED_GRAD = "Unsharded Grad"  # 非分片参数对应的梯度的内存使用
    ACT = "Activation"  # 激活和正向计算重组张量的内存使用
    TEMP = "Temp"  # 反向传播期间临时张量的内存使用，包括激活梯度
    ALL_GATHER = "All Gather"  # all_gather 输出张量的内存使用
    REDUCE_SCATTER = "Reduce Scatter"  # reduce_scatter 输入张量的内存使用
    OPT = "OptState"  # 存储优化器状态张量的内存使用
    INP = "Inputs"  # 输入张量的内存使用


class _SavedFSDPMethods(NamedTuple):
    pre_backward: Callable  # 保存的 FSDP 方法：反向传播前方法
    post_backward: Callable  # 保存的 FSDP 方法：反向传播后方法


class _SavedCollectives(NamedTuple):
    all_gather_into_tensor: Callable  # 保存的 FSDP 集体操作：all_gather 到张量方法
    reduce_scatter_tensor: Callable  # 保存的 FSDP 集体操作：reduce_scatter 张量方法
    all_reduce: Callable  # 保存的 FSDP 集体操作：all_reduce 方法
    barrier: Callable  # 保存的 FSDP 集体操作：barrier 方法


class _FSDPModState(_State):
    """
    Enumerates the states of FSDP modules during the forward and backward passes.
    """

    BEF_PRE_FW = "Before Pre-Forward"  # 前向传播前状态：预正向
    AFT_PRE_FW = "After Pre-Forward"  # 前向传播后状态：预正向
    BEF_POST_FW = "Before Post-Forward"  # 前向传播前状态：后正向
    AFT_POST_FW = "After Post-Forward"  # 前向传播后状态：后正向
    BEF_PRE_BW = "Before Pre-Backward"  # 反向传播前状态：预反向
    AFT_PRE_BW = "After Pre-Backward"  # 反向传播后状态：预反向
    BEF_POST_BW = "Before Post-Backward"  # 反向传播前状态：后反向
    AFT_POST_BW = "After Post-Backward"  # 反向传播后状态：后反向
    PRE_FW_AC = "Pre-Forward AC"  # 正向计算前状态：AC 预正向
    POST_FW_AC = "Post-Forward AC"  # 正向计算后状态：AC 后正向
    PEAK_FW = "Peak Forward"  # 正向计算的峰值状态
    PEAK_BW = "Peak Backward"


# 将字符串"Peak Backward"赋值给变量PEAK_BW
class _FSDPModMemStats:
    """
    A class to store the memory statistics of an FSDP module.

    Args:
        mod_fqn (str): The fully qualified name of the FSDP module.

    Attributes:
        snapshots (Dict[_FSDPModState, Dict[torch.device, Dict[str, int]]]): A dictionary of memory snapshots
        of the module at different states as defined by ``_FSDPModState``. Each key is a device, and
        each value is another dictionary with keys as memory reference types defined by ``_FSDPRefType`` and
        values as the memory consumed in bytes.

    """

    def __init__(self, mod_fqn: str) -> None:
        # 初始化函数，接收FSDP模块的完全限定名称作为参数
        self.mod_fqn = mod_fqn
        # 用于存储每个设备的局部峰值内存消耗的字典
        self.local_peak: Dict[torch.device, int] = {}
        # 存储模块不同状态下的内存快照，每个状态对应一个字典，其中键为设备，值为按_FSDPRefType定义的内存引用类型和消耗的字节
        self.snapshots: Dict[
            _FSDPModState, List[Dict[torch.device, Dict[str, int]]]
        ] = {}


class FSDPMemTracker(MemTracker):
    """
    A ``TorchDispatchMode`` based context manager that extends ``torch.distributed._tools.mem_tracker.MemTracker`` to track
    and categorize the peak memory and module-wise memory usage of FSDP modules.

    It tracks the peak memory usage across all the devices of all the FSDP modules in the module tree and categorizes
    the tensor memory usage as defined by ``_FSDPRefType``. Further, it captures memory `snapshots` at different stages of
    the module execution defined by ``_FSDPModState``.

    Attributes:
        memory_tracking: A weakref key dictionary to store the memory statistics of each module. Each key is a reference
        to a module, and each value is a ``_FSDPModMemStats`` object that stores the memory statistics of the module.

    Args:
        mod (torch.nn.Module): The root FSDP module to be tracked.
        optm (torch.optim.Optimizer, optional): The optimizer to be tracked.

    Note: Please refer to ``torch.distributed._tools.mem_tracker.MemTracker`` to learn about the limitations.

    Example usage

    .. code-block:: python

        module = ...
        optimizer = ...
        inp = ...
        fmt = FSDPMemTracker(module, optimizer)
        fmt.track_inputs((inp,))
        with fmt:
            optimizer.zero_grad()
            loss = module(inp)
            print("After Forward:")
            fmt.display_snapshot("current")
            loss.backward()
            optimizer.step()
        fmt.display_snapshot("peak")
        fmt.display_modulewise_snapshots(depth = 3, units = "MB")

    """

    def __init__(
        self,
        mod: torch.nn.Module,
        optm: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        # 初始化函数，继承自MemTracker类，用于跟踪和分类FSDP模块的峰值内存和模块内存使用情况
        super().__init__()
        # 断言确保mod是FSDPModule类型，因为FSDPMemTracker只支持FSDP模块
        assert isinstance(mod, FSDPModule), "FSDPMemTracker only supports FSDP modules"
        # 存储根FSDP模块
        self._root_mod = mod
        # 要跟踪的优化器（可选）
        self._optm = optm
        # 是否处于假模式
        self._in_fake_mode: bool = False
        # 存储FSDP模块到保存方法的弱引用字典
        self._fsdp_mod_to_saved_methods: WeakIdKeyDictionary = WeakIdKeyDictionary()
        # 保存的收集操作
        self._saved_collectives: _SavedCollectives
        # 引用类（_RefType类型）
        self._ref_class: Type[_RefType] = _FSDPRefType
    # 跟踪在初始化后的分片参数和梯度
    def _instrument_fsdp_sharded_params_grads(
        self, fsdp_param_group: FSDPParamGroup
    ) -> None:
        for fsdp_param in fsdp_param_group.fsdp_params:
            # 更新或创建分片参数的窗口信息
            self._update_and_maybe_create_winfos(
                fsdp_param.sharded_param,
                _FSDPRefType.SHARDED_PARAM,
            )
            # 获取分片梯度
            sharded_grad = fsdp_param.sharded_param.grad
            if sharded_grad is not None:
                # 更新或创建分片梯度的窗口信息
                self._update_and_maybe_create_winfos(
                    sharded_grad,
                    _FSDPRefType.SHARDED_GRAD,
                )

    # 在调用 FSDPState._post_forward 之前的状态捕获内存快照
    # 如果 reshard_after_forward 不是 False，则捕获状态重组的情况
    # 有两种情况：
    # 情况 1：在反向传播中调用，表示在 AC 区域内。如果这是 AC 区域中的顶级模块，则将 _in_ac 标志设置为 False。
    # 情况 2：在前向传播中调用。
    def _fsdp_state_pre_forward(
        self,
        fsdp_mod: FSDPModule,
        orig_fsdp_state_pre_fw: Callable,
    ) -> None:
        @wraps(orig_fsdp_state_pre_fw)
        def inner(*args: Any, **kwargs: Any) -> Any:
            mod_stat = self.memory_tracking[fsdp_mod]
            if self._mod_tracker.is_bw:
                state = _FSDPModState.POST_FW_AC
                if self._ac_mod is not None and self._ac_mod() is fsdp_mod:
                    self._ac_mod = None
                    self._in_ac = False
            else:
                state = _FSDPModState.BEF_POST_FW
            mod_stat.snapshots.setdefault(state, []).append(self.get_tracker_snapshot())

            output = orig_fsdp_state_pre_fw(*args, **kwargs)

            if not self._mod_tracker.is_bw:
                mod_stat.snapshots.setdefault(_FSDPModState.AFT_POST_FW, []).append(
                    self.get_tracker_snapshot()
                )
            return output

        return inner

    # 在调用 FSDPState._post_forward 之后的状态捕获内存快照
    # 如果 reshard_after_forward 不是 False，则捕获状态重组的情况
    # 有两种情况：
    # 情况 1：在反向传播中调用，表示在 AC 区域内。如果这是 AC 区域中的顶级模块，则将 _in_ac 标志设置为 False。
    # 情况 2：在前向传播中调用。
    def _fsdp_state_post_forward(
        self,
        fsdp_mod: FSDPModule,
        orig_fsdp_state_post_fw: Callable,
    ) -> Callable:
        @wraps(orig_fsdp_state_post_fw)
        def inner(*args: Any, **kwargs: Any) -> Any:
            mod_stat = self.memory_tracking[fsdp_mod]
            if self._mod_tracker.is_bw:
                state = _FSDPModState.POST_FW_AC
                if self._ac_mod is not None and self._ac_mod() is fsdp_mod:
                    self._ac_mod = None
                    self._in_ac = False
            else:
                state = _FSDPModState.BEF_POST_FW
            mod_stat.snapshots.setdefault(state, []).append(self.get_tracker_snapshot())

            output = orig_fsdp_state_post_fw(*args, **kwargs)

            if not self._mod_tracker.is_bw:
                mod_stat.snapshots.setdefault(_FSDPModState.AFT_POST_FW, []).append(
                    self.get_tracker_snapshot()
                )
            return output

        return inner

    # 在调用 FSDPParamGroup.pre_backward 之前的状态捕获内存快照
    def _fsdp_param_group_pre_backward(
        self,
        fsdp_mod: FSDPModule,
        orig_fsdp_param_group_pre_backward: Callable,
    ) -> Callable:
        # 返回一个装饰器函数，用于包装原始的 FSDPParamGroup.pre_backward 方法
        @wraps(orig_fsdp_param_group_pre_backward)
        def inner(*args: Any, **kwargs: Any) -> None:
            # 获取当前 FSDPModule 对象的内存统计信息
            mod_stat = self.memory_tracking[fsdp_mod]
            # 获取当前的内存快照
            snapshot = self.get_tracker_snapshot()
            # 记录本地峰值内存使用情况到 mod_stat.local_peak 字典中
            mod_stat.local_peak = {
                dev: dev_snap[_TOTAL_KEY] for dev, dev_snap in snapshot.items()
            }
            # 将当前快照存储到 mod_stat.snapshots 字典的 PEAK_BW 键下
            mod_stat.snapshots.setdefault(_FSDPModState.PEAK_BW, []).append(snapshot)
            # 将当前快照存储到 mod_stat.snapshots 字典的 BEF_PRE_BW 键下（深拷贝）
            mod_stat.snapshots.setdefault(_FSDPModState.BEF_PRE_BW, []).append(
                deepcopy(snapshot)
            )
            # 调用原始的 FSDPParamGroup.pre_backward 方法
            orig_fsdp_param_group_pre_backward(*args, **kwargs)

            # 将 FSDPParamGroup.pre_backward 方法执行后的内存快照存储到 AFT_PRE_BW 键下
            mod_stat.snapshots.setdefault(_FSDPModState.AFT_PRE_BW, []).append(
                self.get_tracker_snapshot()
            )

        return inner



    def _fsdp_param_group_post_backward(
        self,
        fsdp_mod: FSDPModule,
        orig_fsdp_param_group_post_backward: Callable,
    ) -> Callable:
        # 返回一个装饰器函数，用于包装原始的 FSDPParamGroup.post_backward 方法
        @wraps(orig_fsdp_param_group_post_backward)
        def inner(*args: Any, **kwargs: Any) -> None:
            # 获取当前 FSDPModule 对象的状态
            fsdp_state = fsdp_mod._get_fsdp_state()
            # 如果存在 fsdp_param_group，则处理其中的参数
            if fsdp_param_group := fsdp_state._fsdp_param_group:
                # 遍历 fsdp_param_group 中的每个 fsdp_param
                for fsdp_param in fsdp_param_group.fsdp_params:
                    # 获取未分片的梯度（unsharded_grad）
                    unsharded_grad = fsdp_param._unsharded_param.grad
                    # 如果未分片的梯度不为空，则更新或创建相关的窗口信息
                    if unsharded_grad is not None:
                        self._update_and_maybe_create_winfos(
                            unsharded_grad,
                            _FSDPRefType.UNSHARDED_GRAD,
                            update_existing=True,
                        )

            # 获取当前 FSDPModule 对象的内存统计信息
            mod_stat = self.memory_tracking[fsdp_mod]
            # 将当前快照存储到 mod_stat.snapshots 字典的 BEF_POST_BW 键下
            mod_stat.snapshots.setdefault(_FSDPModState.BEF_POST_BW, []).append(
                self.get_tracker_snapshot()
            )

            # 调用原始的 FSDPParamGroup.post_backward 方法
            orig_fsdp_param_group_post_backward(*args, **kwargs)

            # 如果存在 fsdp_param_group，则处理其中的参数
            if fsdp_param_group := fsdp_state._fsdp_param_group:
                # 遍历 fsdp_param_group 中的每个 fsdp_param
                for fsdp_param in fsdp_param_group.fsdp_params:
                    # 获取分片后的梯度（sharded_grad）
                    sharded_grad = fsdp_param.sharded_param.grad
                    # 如果分片后的梯度不为空，则更新或创建相关的窗口信息
                    if sharded_grad is not None:
                        self._update_and_maybe_create_winfos(
                            sharded_grad,
                            _FSDPRefType.SHARDED_GRAD,
                        )

            # 将当前快照存储到 mod_stat.snapshots 字典的 AFT_POST_BW 键下
            mod_stat.snapshots.setdefault(_FSDPModState.AFT_POST_BW, []).append(
                self.get_tracker_snapshot()
            )

        return inner
    def _instrument_fsdp_module(self) -> None:
        # 遍历根模块中的所有子模块
        for module in self._root_mod.modules():
            # 检查当前模块是否为 FSDPModule 类型
            if isinstance(module, FSDPModule):
                # 获取当前 FSDPModule 的 FSDPState 对象
                fsdp_state = module._get_fsdp_state()
                # 如果存在 FSDPParamGroup 对象，则进行下列操作
                if fsdp_param_group := fsdp_state._fsdp_param_group:
                    # 调用方法处理 FSDPParamGroup 的参数梯度
                    self._instrument_fsdp_sharded_params_grads(fsdp_param_group)
                    
                    # 移除现有的 _pre_forward 和 _post_forward 钩子，并注册新的钩子
                    fsdp_state._pre_forward_hook_handle.remove()
                    fsdp_state._post_forward_hook_handle.remove()
                    fsdp_state._pre_forward_hook_handle = (
                        # 注册新的前向传播前钩子
                        module.register_forward_pre_hook(
                            self._fsdp_state_pre_forward(
                                module, fsdp_state._pre_forward
                            ),
                            prepend=True,
                            with_kwargs=True,
                        )
                    )
                    fsdp_state._post_forward_hook_handle = (
                        # 注册新的前向传播后钩子
                        module.register_forward_hook(
                            self._fsdp_state_post_forward(module, fsdp_state._post_forward),
                            prepend=False,
                            always_call=True,
                        )
                    )
                    
                    # 保存当前 FSDPParamGroup 的方法
                    self._fsdp_mod_to_saved_methods[module] = _SavedFSDPMethods(
                        fsdp_param_group.pre_backward,
                        fsdp_param_group.post_backward,
                    )
                    
                    # 修改 FSDPParamGroup 的 pre_backward 和 post_backward 方法
                    fsdp_param_group.pre_backward = self._fsdp_param_group_pre_backward(  # type: ignore[assignment]
                        module, fsdp_param_group.pre_backward
                    )
                    fsdp_param_group.post_backward = (
                        self._fsdp_param_group_post_backward(
                            module, fsdp_param_group.post_backward
                        )
                    )

        # 遍历根模块中的所有缓冲区
        for buffer in self._root_mod.buffers():
            # 更新或创建缓冲区相关的信息
            self._update_and_maybe_create_winfos(
                buffer,
                _FSDPRefType.BUFFER,
            )
    def _instrument_optimizer(self) -> None:
        # 如果存在优化器对象，则注册钩子函数以跟踪优化器状态。
        # 预钩子函数用于将标志 ``_in_opt`` 设置为 True。
        # 后钩子函数取消标志的设置，并跟踪在优化器步骤期间创建的任何优化器状态。
        if self._optm is not None:
            self._track_optimizer_states(_FSDPRefType.OPT, self._optm)

            def _opt_step_pre_hook(
                optimizer: optim.Optimizer, args: Any, kwargs: Any
            ) -> None:
                self._in_opt = True

            def _opt_step_post_hook(
                optimizer: optim.Optimizer, args: Any, kwargs: Any
            ) -> None:
                self._track_optimizer_states(_FSDPRefType.OPT, optimizer)
                self._in_opt = False

            # 注册优化器步骤的预钩子和后钩子
            self._optimizer_hook_handles = (
                self._optm.register_step_pre_hook(_opt_step_pre_hook),
                self._optm.register_step_post_hook(_opt_step_post_hook),
            )

    def _register_module_and_optimizer_hooks(self) -> None:
        # 注册模块和优化器的钩子函数
        self._instrument_fsdp_module()
        self._instrument_optimizer()

    def _deregister_module_and_optimizer_hooks(self) -> None:
        # 反注册模块和优化器的钩子函数
        for (
            fsdp_mod,
            saved_methods,
        ) in self._fsdp_mod_to_saved_methods.items():
            fsdp_state = fsdp_mod._get_fsdp_state()
            # 移除前向传播前后钩子
            fsdp_state._pre_forward_hook_handle.remove()
            fsdp_state._post_forward_hook_handle.remove()
            # 重新注册前向传播前后钩子
            fsdp_state._pre_forward_hook_handle = fsdp_mod.register_forward_pre_hook(
                fsdp_state._pre_forward, prepend=True, with_kwargs=True
            )
            fsdp_state._post_forward_hook_handle = fsdp_mod.register_forward_hook(
                fsdp_state._post_forward, prepend=False
            )
            # 如果存在 FSDP 参数组，则恢复保存的反向传播方法
            if fsdp_param_group := fsdp_state._fsdp_param_group:
                fsdp_param_group.pre_backward = saved_methods.pre_backward
                fsdp_param_group.post_backward = saved_methods.post_backward
        self._fsdp_mod_to_saved_methods.clear()

        # 移除优化器钩子函数
        if self._optimizer_hook_handles is not None:
            for handle in self._optimizer_hook_handles:
                handle.remove()
            self._optimizer_hook_handles = None

    def _restore_collectives(self) -> None:
        # 恢复保存的集合操作函数
        dist.all_gather_into_tensor = self._saved_collectives.all_gather_into_tensor
        dist.reduce_scatter_tensor = self._saved_collectives.reduce_scatter_tensor
        dist.all_reduce = self._saved_collectives.all_reduce
        dist.barrier = self._saved_collectives.barrier
        # 删除保存的集合操作对象
        del self._saved_collectives
    def track_inputs(self, inputs: Tuple[Any, ...]) -> None:
        """
        This is used to track the input tensors to the model and annotate them as ``Inputs``.
        Args:
            inputs (Tuple[Any]): A tuple containing the input data. This can include tensors
                        as well as other data types. Only tensors will be tracked.
        """

        # 定义内部函数 _track_inputs，用于更新输入数据的状态
        def _track_inputs(t: torch.Tensor) -> None:
            self._update_and_maybe_create_winfos(
                t,
                _FSDPRefType.INP,
            )

        # 针对输入中的每个 torch.Tensor 调用 _track_inputs 函数进行状态更新
        tree_map_only(torch.Tensor, _track_inputs, inputs)

    def track_external(
        self, *external: Union[nn.Module, optim.Optimizer, torch.Tensor]
    ) -> None:
        """This is no-op for ``FSDPMemTracker``"""
        # 对于 FSDPMemTracker 类的 track_external 方法，什么也不做，故而是空操作
        pass

    def __enter__(self) -> "FSDPMemTracker":
        # 进入上下文管理器时执行的操作，设置模拟模式、注册模块和优化器钩子、以及设置内存跟踪等
        self._in_fake_mode = True if active_fake_mode() else False
        self._register_module_and_optimizer_hooks()
        self._instrument_and_maybe_bypass_collectives()
        self._track_resize()
        # 获取当前内存快照，设置内存峰值快照和内存峰值字典
        self._peak_mem_snap = self.get_tracker_snapshot()
        self._peak_mem = {
            dev: dev_snap[_TOTAL_KEY] for dev, dev_snap in self._peak_mem_snap.items()
        }
        # 进入模块追踪器上下文
        self._mod_tracker.__enter__()
        # 进入 TorchDispatchMode 上下文
        TorchDispatchMode.__enter__(self)
        return self

    def __exit__(self, *args: Any) -> None:
        # 退出上下文管理器时执行的操作，包括取消模块和优化器钩子的注册、恢复通信集合操作、以及恢复内存调整等
        self._deregister_module_and_optimizer_hooks()
        self._restore_collectives()
        self._restore_resize()
        # 退出 TorchDispatchMode 上下文
        TorchDispatchMode.__exit__(self, *args)
        # 退出模块追踪器上下文
        self._mod_tracker.__exit__(*args)

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):  # type: ignore[no-untyped-def]
        # 执行 TorchDispatch 时调用的方法，根据当前状态更新参考类型，并更新峰值状态统计信息
        res = func(*args, **kwargs or {})
        if self._in_opt:
            reftype = _FSDPRefType.OPT
        elif self._mod_tracker.is_bw and not self._in_ac:
            reftype = _FSDPRefType.TEMP
        else:
            reftype = _FSDPRefType.ACT
        # 针对结果中的每个 torch.Tensor 调用 _track 方法进行状态更新
        tree_map_only(torch.Tensor, partial(self._track, reftype), res)
        # 根据当前模块状态更新峰值状态统计信息
        peak_state = (
            _FSDPModState.PEAK_BW if self._mod_tracker.is_bw else _FSDPModState.PEAK_FW
        )
        self._update_peak_stats(peak_state)
        return res
```