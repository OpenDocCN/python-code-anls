# `.\pytorch\torch\distributed\_composable\fsdp\fully_shard.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和函数
import functools  # 导入 functools 模块，用于高阶函数（higher-order functions）
from typing import Any, cast, Iterable, List, NoReturn, Optional, Union  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 深度学习框架
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch.distributed._composable import contract  # 导入 torch 分布式模块的 contract 装饰器
from torch.distributed._tensor import DeviceMesh  # 导入 torch 分布式模块的 DeviceMesh 类

from ._fsdp_api import MixedPrecisionPolicy, OffloadPolicy  # 导入 FSDP API 相关模块
from ._fsdp_common import FSDPMeshInfo, HSDPMeshInfo  # 导入 FSDP 公共模块
from ._fsdp_init import (
    _get_device_from_mesh,  # 导入初始化模块的获取设备函数
    _get_managed_modules,   # 导入初始化模块的获取管理模块函数
    _get_managed_states,    # 导入初始化模块的获取管理状态函数
    _get_post_forward_mesh_info,  # 导入初始化模块的获取后向传播后网格信息函数
    _init_default_fully_shard_mesh,  # 导入初始化模块的默认完全分片网格初始化函数
    _move_states_to_device,  # 导入初始化模块的状态移动到设备函数
)
from ._fsdp_param_group import FSDPParamGroup  # 导入 FSDP 参数组相关模块
from ._fsdp_state import _get_module_fsdp_state, FSDPState  # 导入 FSDP 状态相关模块


# 该装饰器将一个状态对象添加到 `module`，可以通过 `fully_shard.state(module)` 访问该状态对象。状态对象和模块是一对一的关系。
@contract(state_cls=FSDPState)  # 使用 contract 装饰器，指定参数类型为 FSDPState 类型
def fully_shard(
    module: nn.Module,  # 输入参数 module 是一个 nn.Module 类的对象
    *,
    mesh: Optional[DeviceMesh] = None,  # 可选参数 mesh，用于指定设备网格信息，默认为 None
    reshard_after_forward: Union[bool, int] = True,  # 可选参数 reshard_after_forward，指定是否在每次前向传播后重新分片，默认为 True
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),  # 混合精度策略对象，默认为 MixedPrecisionPolicy() 实例
    offload_policy: OffloadPolicy = OffloadPolicy(),  # 卸载策略对象，默认为 OffloadPolicy() 实例
):
    """
    将模块的参数在数据并行工作器之间进行分片。

    该函数应用完全分片数据并行（FSDP）或其变体于 ``module`` 上，这是一种在减少内存占用的同时增加通信成本的技术。
    参数在 ``mesh`` 上进行分片，并且它们的梯度和优化器状态也相应地进行分片。

    分片后的参数通过全收集来构造未分片的参数，用于前向或后向计算。计算后，未分片的参数被释放以节省内存。
    梯度通过网格进行减少，并且除以网格大小以实现数据并行。优化器步骤在分片参数上运行。

    每次调用 ``fully_shard`` 都会构建一个包含 ``module.parameters()`` 中参数的通信组，除非这些参数已经分配给来自嵌套调用的组。
    模型中构建多个组（例如“逐层”）允许达到内存的峰值节省和通信/计算重叠。

    在实现上，分片的参数表示为 :class:`DTensor`，在 dim-0 上进行分片，未分片的参数表示为 :class:`Tensor`。
    一个模块前向预钩子进行全收集参数，一个模块前向钩子释放它们。类似的后向钩子收集参数，然后释放参数/减少梯度。
    """
    """
    Args:
        mesh (Optional[DeviceMesh]): This data parallel mesh defines the
            sharding and device. If 1D, then parameters are fully sharded
            across the 1D mesh (FSDP). If 2D, then parameters are sharded
            across the 0th dim and replicated across the 1st dim (HSDP). The
            mesh's device type gives the device type used for communication;
            if a CUDA or CUDA-like device type, then we use the current device.
        reshard_after_forward (Union[bool, int]): This controls the parameter
            behavior after forward and can trade off memory and communication:
            - If ``True``, then this reshards parameters after forward and
            all-gathers in backward.
            - If ``False``, then this keeps the unsharded parameters in memory
            after forward and avoids the all-gather in backward.
            - If an ``int``, then this represents the world size to reshard to
            after forward. It should be a non-trivial divisor of the ``mesh``
            shard dim size (i.e. excluding 1 and the dim size itself). A choice
            may be the intra-node size (e.g. ``torch.cuda.device_count()``).
            This allows the all-gather in backward to be over a smaller world
            size at the cost of higher memory usage than setting to ``True``.
            - The root FSDP state has its value specially set to ``False`` as a
            heuristic since its parameters would typically be immediately
            all-gathered for backward.
            - After forward, the parameters registered to the module depend on
            to this: The registered parameters are the sharded parameters if
            ``True``; unsharded parameters if ``False``; and the paramters
            resharded to the smaller mesh otherwise. To modify the parameters
            between forward and backward, the registered parameters must be the
            sharded parameters. For ``False`` or an ``int``, this can be done
            by manually resharding via :meth:`reshard`.
        mp_policy (MixedPrecisionPolicy): This controls the mixed precision
            policy, which offers parameter/reduction mixed precision for this
            module. See :class:`MixedPrecisionPolicy` for details.
        offload_policy (OffloadPolicy): This controls the offloading policy,
            which offers parameter/gradient/optimizer state offloading. See
            :class:`OffloadPolicy` and its subclasses for details.
    """
    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        raise ValueError(
            f"fully_shard does not support containers that do not implement forward: {module}"
        )
    mesh = mesh or _init_default_fully_shard_mesh()
    if mesh.ndim not in (1, 2):
        raise ValueError(f"fully_shard expects a 1D or 2D DeviceMesh but got {mesh}")
    elif mesh.ndim == 1:
        mesh_info = FSDPMeshInfo(mesh, shard_mesh_dim=0)
    else:
        # 如果不需要进行分片，使用默认的 HSDPMeshInfo 初始化
        mesh_info = HSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
    
    # 根据 mesh 获取设备信息
    device = _get_device_from_mesh(mesh)
    
    # 获取后向传播时的网格信息
    post_forward_mesh_info = _get_post_forward_mesh_info(
        reshard_after_forward, mesh_info
    )
    
    # 获取模块的完全分片状态
    state = fully_shard.state(module)
    
    # 初始化模块的状态信息
    state.init(module, device, mp_policy)
    
    # 获取受管理的模块列表
    managed_modules = _get_managed_modules(module)
    
    # 获取受管理状态的参数和缓冲区
    params, buffers = _get_managed_states(managed_modules)
    
    # 将参数和缓冲区移到指定设备上
    _move_states_to_device(params, buffers, device)
    
    # 如果存在参数，则创建 FSDPParamGroup 对象
    if params:
        state._fsdp_param_group = FSDPParamGroup(
            params,
            module,
            mesh_info,
            post_forward_mesh_info,
            device,
            mp_policy,
            offload_policy,
        )
    
    # 对于 Dynamo
    for managed_module in managed_modules:
        # 设置为 FSDP 管理的模块
        managed_module._is_fsdp_managed_module = True  # type: ignore[assignment]
        # 使用原始参数进行 FSDP 管理
        managed_module._fsdp_use_orig_params = True  # type: ignore[assignment]
    
    # 将 FSDP 放在方法解析顺序的最左侧，以获得最高优先级
    cls = module.__class__
    dct = {"__deepcopy__": unimplemented_deepcopy}
    new_cls = type(f"FSDP{cls.__name__}", (FSDPModule, cls), dct)
    module.__class__ = new_cls
    
    # 返回修改后的模块对象
    return module
def unimplemented_deepcopy(*args: Any, **kwargs: Any) -> NoReturn:
    # 抛出断言错误，因为 FSDP 不支持深拷贝，建议使用状态字典进行序列化
    raise AssertionError(
        "FSDP does not support deepcopy. Please use state dict for serialization."
    )


class FSDPModule:
    def __new__(cls, *args, **kwargs):
        """
        重写 __new__ 方法以移除 FSDP 类并直接构造原始类，例如索引到容器模块中的情况。
        """
        # 使用索引 2，因为索引 0 是动态构造的 `FSDP<...>` 类
        # 索引 1 是 `FSDPModule` 类本身
        orig_cls = cls.__mro__[2]
        self = orig_cls.__new__(orig_cls, *args, **kwargs)
        self.__init__(*args, **kwargs)
        return self

    def reshard(self) -> None:
        """
        重新分配模块的参数，将分片参数注册到模块，并在需要时释放非分片参数。此方法不递归。
        """
        state = self._get_fsdp_state()
        if fsdp_param_group := state._fsdp_param_group:
            fsdp_param_group.reshard()

    def unshard(self, async_op: bool = False) -> Optional["UnshardHandle"]:
        """
        通过分配内存并收集所有参数来取消分片模块的参数。此方法不递归。

        Args:
            async_op (bool): 如果为 ``True``，则返回一个具有 :meth:`wait` 方法的 :class:`UnshardHandle`，
                用于等待取消分片操作。如果为 ``False``，则返回 ``None`` 并在函数内部等待句柄。

        .. warning:: 此方法是实验性的，可能会更改。

        .. note:: 如果 ``async_op=True``，则用户无需在返回的句柄上调用 :meth:`wait`，
            因为 FSDP 将在预前向传播中自动等待挂起的取消分片操作。
        """
        state = self._get_fsdp_state()
        fsdp_param_group = state._fsdp_param_group
        if fsdp_param_group is not None:
            fsdp_param_group.lazy_init()
            fsdp_param_group.unshard(async_op=async_op)
        handle = UnshardHandle(fsdp_param_group)
        if async_op:
            return handle
        handle.wait()
        return None

    def set_is_last_backward(self, is_last_backward: bool) -> None:
        """
        设置下一个反向传播是否为最后一个，这意味着 FSDP 应该等待梯度归约完成，并清除用于显式预取的内部数据结构。
        """
        state = self._get_fsdp_state()
        state._state_ctx.is_last_backward = is_last_backward

    def set_requires_gradient_sync(
        self, requires_gradient_sync: bool, *, recurse: bool = True
    ):
        """
        设置是否需要梯度同步，如果需要，则递归设置其子模块。
        """
    ) -> None:
        """
        设置模块是否同步梯度。这可以用于实现梯度累积而无需通信。对于 HSDP，这同时控制了
        reduce-scatter 和 all-reduce。

        Args:
            requires_gradient_sync (bool): 是否减少模块参数的梯度。
            recurse (bool): 是否设置所有子模块或仅传入的模块。
        """
        # 将 self 强制转换为 nn.Module 类型
        self_module = cast(nn.Module, self)
        # 如果 recurse 为 True，则获取所有子模块；否则只包含 self_module
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            # 检查当前模块是否为 FSDPModule 类型
            if isinstance(module, FSDPModule):
                # 获取 FSDP 模块的状态信息
                state = module._get_fsdp_state()
                # 检查是否存在 FSDP 参数组
                if fsdp_param_group := state._fsdp_param_group:
                    # 设置参数组的 reduce_grads 和 all_reduce_grads 属性
                    fsdp_param_group.reduce_grads = requires_gradient_sync
                    fsdp_param_group.all_reduce_grads = requires_gradient_sync

    def set_requires_all_reduce(
        self, requires_all_reduce: bool, *, recurse: bool = True
    ) -> None:
        """
        设置模块是否应该进行全局梯度归约。这可以用于仅使用 reduce-scatter 而不是
        all-reduce 进行梯度累积。

        Args:
            requires_all_reduce (bool): 是否进行全局梯度归约。
            recurse (bool): 是否设置所有子模块或仅传入的模块。
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    # 设置参数组的 all_reduce_grads 属性
                    fsdp_param_group.all_reduce_grads = requires_all_reduce

    def set_reshard_after_backward(
        self, reshard_after_backward: bool, *, recurse: bool = True
    ) -> None:
        """
        设置模块是否在反向传播后重新划分参数。这可以在梯度累积期间用来在内存和通信之间
        进行权衡。

        Args:
            reshard_after_backward (bool): 是否在反向传播后重新划分参数。
            recurse (bool): 是否设置所有子模块或仅传入的模块。
        """
        self_module = cast(nn.Module, self)
        modules = list(self_module.modules()) if recurse else [self_module]
        for module in modules:
            if isinstance(module, FSDPModule):
                state = module._get_fsdp_state()
                if fsdp_param_group := state._fsdp_param_group:
                    # 设置参数组的 reshard_after_backward 属性
                    fsdp_param_group.reshard_after_backward = reshard_after_backward
    # 设置需要在前向传播中显式预取所有聚合的 FSDP 模块。
    # 预取操作在该模块的聚合拷贝之后运行。
    # 当传递一个包含下一个 FSDP 模块的单例列表时，会产生与默认重叠行为相同的聚合操作重叠行为，但预取聚合操作会从 CPU 提前发起。
    # 当传递长度至少为二的列表时，会采用更积极的重叠方式，并使用更多的保留内存。
    def set_modules_to_forward_prefetch(self, modules: List["FSDPModule"]) -> None:
        """
        Sets the FSDP modules for which this FSDP module should explicitly
        prefetch all-gathers in forward. The prefetching runs after this
        module's all-gather copy-out.

        Passing a singleton list containing the next FSDP module gives the same
        all-gather overlap behavior as the default overlap behavior, except the
        prefetched all-gather is issued earlier from the CPU. Passing a list
        with at least length two is required for more aggressive overlap and
        will use more reserved memory.

        Args:
            modules (List[FSDPModule]): FSDP modules to prefetch.
        """
        _assert_all_fsdp_modules(modules)
        # 将每个模块的 FSDP 状态添加到待预取的状态列表中
        self._get_fsdp_state()._states_to_forward_prefetch = [
            module._get_fsdp_state() for module in modules
        ]

    # 设置需要在反向传播中显式预取所有聚合的 FSDP 模块。
    # 这会覆盖默认的反向预取实现，后者基于逆向的后-前顺序预取下一个 FSDP 模块。
    # 当传递一个包含前一个 FSDP 模块的单例列表时，会产生与默认重叠行为相同的聚合操作重叠行为。
    # 当传递长度至少为二的列表时，会采用更积极的重叠方式，并使用更多的保留内存。
    def set_modules_to_backward_prefetch(self, modules: List["FSDPModule"]) -> None:
        """
        Sets the FSDP modules for which this FSDP module should explicitly
        prefetch all-gathers in backward. This overrides the default backward
        pretching implementation that prefetches the next FSDP module based on
        the reverse post-forward order.

        Passing a singleton list containing the previous FSDP module gives the
        same all-gather overlap behavior as the default overlap behavior.
        Passing a list with at least length two is required for more aggressive
        overlap and will use more reserved memory.

        Args:
            modules (List[FSDPModule]): FSDP modules to prefetch.
        """
        _assert_all_fsdp_modules(modules)
        # 将每个模块的 FSDP 状态添加到待预取的状态列表中
        self._get_fsdp_state()._states_to_backward_prefetch = [
            module._get_fsdp_state() for module in modules
        ]

    # 设置一个后优化步骤事件，用于根 FSDP 模块等待所有聚合流的事件。
    # 默认情况下，根 FSDP 模块会在当前流上等待所有聚合流，以确保优化步骤完成后再进行聚合。
    # 然而，如果在优化步骤之后存在不相关的计算，这可能引入虚假的依赖关系。
    # 此 API 允许用户提供自己的事件进行等待。
    # 在根模块等待事件后，事件会被丢弃，因此应在每次迭代中使用新的事件调用此 API。
    def set_post_optim_event(self, event: torch.cuda.Event) -> None:
        """
        Sets a post-optimizer-step event for the root FSDP module to wait the
        all-gather streams on.

        By default, the root FSDP module waits the all-gather streams on the
        current stream to ensure that the optimizer step has finished before
        all-gathering. However, this may introduce false dependencies if
        there is unrelated computation after the optimizer step. This API
        allows the user to provide their own event to wait on. After the root
        waits on the event, the event is discarded, so this API should be
        called with a new event each iteration.

        Args:
            event (torch.cuda.Event): Event recorded after the optimizer step
                to wait all-gather streams on.
        """
        # 将事件设置为根 FSDP 模块等待的后优化步骤事件
        self._get_fsdp_state()._state_ctx.post_optim_event = event

    # 返回当前模块的 FSDP 状态。
    def _get_fsdp_state(self) -> FSDPState:
        if (state := _get_module_fsdp_state(cast(nn.Module, self))) is None:
            raise AssertionError(f"No FSDP state found on {self}")
        return state
    # 确保分片参数已注册，重新分片
    self.reshard()
    # 调用父类的_apply方法，并返回结果
    ret = super()._apply(*args, **kwargs)  # type: ignore[misc]
    # 获取当前对象的FSDP状态
    state = self._get_fsdp_state()
    # 如果没有FSDP参数组，直接返回之前的结果
    if not (fsdp_param_group := state._fsdp_param_group):
        return ret
    # 在没有梯度跟踪的情况下，对每个FSDP参数进行重置其分片参数
    # 这段逻辑是为了解决一个问题，在DTensor能够填充本地张量之后会被移除
    with torch.no_grad():
        for fsdp_param in fsdp_param_group.fsdp_params:
            fsdp_param.reset_sharded_param()
    return ret
class UnshardHandle:
    """
    A handle to wait on the unshard op.

    Args:
        fsdp_param_group (FSDPParamGroup, optional): FSDP parameter group to
            unshard. This should be ``None`` iff the FSDP module does not
            manage any parameters, meaning the unshard is a no-op.
    """

    def __init__(self, fsdp_param_group: Optional[FSDPParamGroup]):
        # 初始化方法，接受一个 FSDPParamGroup 类型的参数，用于管理参数的分片
        self._fsdp_param_group = fsdp_param_group

    def wait(self):
        """
        Waits on the unshard op.

        This ensures that the current stream can use the unsharded parameters,
        which are now registered to the module.
        """
        # 等待参数的 unshard 操作完成
        if self._fsdp_param_group is not None:
            self._fsdp_param_group.wait_for_unshard()
            # 避免保持对参数组的引用，以便垃圾回收
            self._fsdp_param_group = None


def register_fsdp_forward_method(module: nn.Module, method_name: str) -> None:
    """
    Registers a method on ``module`` to be a forward method for FSDP.

    FSDP only knows to run its pre-forward and post-forward hooks on the
    default :meth:`nn.Module.forward` method. This function patches a user
    specified method to run the pre/post-forward hooks before/after the method,
    respectively. If ``module`` is not an :class:`FSDPModule`, then this is a
    no-op.

    Args:
        module (nn.Module): Module to register the forward method on.
        method_name (str): Name of the forward method.
    """
    # 注册一个方法作为 FSDP 的前向方法
    if not isinstance(module, FSDPModule):
        # 如果 module 不是 FSDPModule 类型，则不执行任何操作，允许在使用和不使用 FSDP 时都包含该函数
        return
    if not hasattr(module, method_name):
        # 如果 module 没有指定的方法名，抛出 ValueError
        raise ValueError(f"{type(module)} does not have a method {method_name}")
    orig_method = getattr(module, method_name)

    @functools.wraps(orig_method)
    def wrapped_method(self, *args, **kwargs):
        # 创建一个包装函数，用于在指定方法的前后运行 FSDP 的前向和后向钩子
        fsdp_state = self._get_fsdp_state()
        args, kwargs = fsdp_state._pre_forward(self, args, kwargs)
        out = orig_method(*args, **kwargs)
        return fsdp_state._post_forward(self, args, out)

    # 使用 `__get__` 方法将 `wrapped_method` 变成实例方法
    setattr(
        module,
        method_name,
        wrapped_method.__get__(module, type(module)),  # type:ignore[attr-defined]
    )


def _assert_all_fsdp_modules(modules: Iterable[Any]) -> None:
    """
    Asserts that all modules in the iterable are instances of FSDPModule.

    Args:
        modules (Iterable[Any]): Iterable of modules to check.
    """
    for module in modules:
        if not isinstance(module, FSDPModule):
            # 如果 module 不是 FSDPModule 类型，抛出 ValueError
            raise ValueError(f"Expects FSDPModule but got {type(module)}: {module}")
```