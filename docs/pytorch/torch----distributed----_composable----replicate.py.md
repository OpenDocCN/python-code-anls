# `.\pytorch\torch\distributed\_composable\replicate.py`

```
# mypy: allow-untyped-defs
# 引入弱引用模块，用于管理对象的弱引用
import weakref
# 引入类型提示模块
from typing import Any, cast, Dict, Iterable, List, NoReturn, Optional, Set, Tuple

# 引入 PyTorch 深度学习框架
import torch
# 引入神经网络模块
import torch.nn as nn
# 引入分布式训练相关状态管理模块
from torch.distributed._composable_state import _State
# 引入分布式数据并行模块
from torch.nn.parallel import DistributedDataParallel

# 引入本地合同模块
from .contract import _get_registry, contract

# 全局变量，根模块前缀为空字符串
_ROOT_MODULE_PREFIX = ""

# _ReplicateState 类，继承自 _State 类
class _ReplicateState(_State):
    # _ReplicateState 类的初始化函数
    def __init__(self) -> None:
        super().__init__()
        # module 属性作为 nn.Module 对象列表
        self.module: nn.Module = nn.ParameterList()
        # 标志是否已初始化
        self.has_initialized: bool = False
        # _param_list 属性作为 nn.ParameterList 对象
        self._param_list: nn.ParameterList = nn.ParameterList()
        # TODO(@fegin): 用于测试的变量，可能会移除
        self._orig_module = self.module
        # _param_names 属性作为字符串列表
        self._param_names: List[str] = []
        # _no_sync 属性标志是否同步
        self._no_sync: bool = False
        # _init_args 属性为可选的元组参数
        self._init_args: Optional[Tuple[Any, ...]] = None
        # _init_kwargs 属性为字典类型参数
        self._init_kwargs: Dict[str, Any] = {}
        # _comm_hook_args 属性为任意类型参数列表
        self._comm_hook_args: List[Any] = []

    # _collect_params 方法，收集模块参数
    def _collect_params(
        self,
        module: nn.Module,
        ignored_modules: Set[nn.Module],
        ignored_params: Set[nn.Parameter],
        prefix: str = _ROOT_MODULE_PREFIX,
    ) -> None:
        # 如果模块由 fully_sharded API 管理，则跳过
        if _is_fully_sharded(module):
            return

        # 如果模块在忽略模块集合中，则跳过该模块及其所有后代
        if module in ignored_modules:
            return

        # 递归前缀处理
        recurse_prefix = (
            f"{prefix}." if prefix != _ROOT_MODULE_PREFIX else _ROOT_MODULE_PREFIX
        )

        # 遍历模块的命名参数
        for n, p in module.named_parameters(recurse=False):
            # 如果参数不在忽略参数集合中，则加入参数列表
            if p not in ignored_params:
                self._param_list.append(p)
                self._param_names.append(f"{recurse_prefix}{n}")

        # 遍历模块的子模块
        for name, child_module in module.named_children():
            # 递归调用 _collect_params 方法
            self._collect_params(
                child_module,
                ignored_modules,
                ignored_params,
                prefix=f"{recurse_prefix}{name}",
            )

    # lazy_init 方法，延迟初始化操作
    def lazy_init(self) -> None:
        # 使用 torch._disable_dynamo 装饰器进行动态禁用
        @torch._disable_dynamo(recursive=True)
        def _lazy_init():
            # 断言初始化参数不为空
            assert self._init_args is not None
            # 调用 init 方法进行初始化
            self.init(*self._init_args, **self._init_kwargs)
            # 注册通信钩子
            self.register_comm_hook()
            # 清空初始化参数和关键字参数
            self._init_args = tuple()
            self._init_kwargs = {}

        # 执行延迟初始化函数
        _lazy_init()

    # init 方法，初始化模块及其参数
    def init(
        self,
        module: nn.Module,
        ignored_modules: Set[nn.Module],
        **kwargs,
    ) -> None:
        # 如果已经初始化过，则直接返回，避免重复初始化
        if self.has_initialized:
            return

        # 标记为已经初始化
        self.has_initialized = True

        # 从关键字参数中获取设备网格信息，默认为 None
        device_mesh = kwargs.get("device_mesh", None)
        # 将传入的 module 参数保存到对象属性中
        self.module = module
        # 初始化一个集合，用于存放需要忽略的参数
        ignored_params = {p for m in ignored_modules for p in m.parameters()}
        # 导入局部函数 _localize_dtensor
        from torch.distributed.tensor.parallel.ddp import _localize_dtensor

        # 调用 _localize_dtensor 函数，处理传入的 module
        _localize_dtensor(module)
        # 调用对象的私有方法 _collect_params，收集 module 的参数信息
        self._collect_params(module, ignored_modules, ignored_params)

        # 如果 kwargs 中包含 "device_id" 键
        if "device_id" in kwargs:
            # replicate() 支持一种小的用户友好性增强，即用户可以传入 Union[int, torch.device]
            # 类型的 device_id，即使是 CPU 设备，用户也不需要为 CPU/GPU 运行改变代码。
            # 我们根据传入的 device_id 推导出 DDP 需要的正确 device_ids。
            if kwargs["device_id"] is not None:
                device_id = kwargs["device_id"]
                # 将 device_id 转换为 DDP 需要的 device_ids 格式
                if isinstance(device_id, torch.device) and device_id.type == "cpu":
                    # CPU 模块接收 device_ids 为 None
                    kwargs["device_ids"] = None
                else:
                    # GPU 模块期望 device_ids=[cuda_device]
                    kwargs["device_ids"] = [device_id]
            else:
                kwargs["device_ids"] = None
            # 删除 kwargs 中的 "device_id" 键
            kwargs.pop("device_id")

        # 使用给定的参数初始化 DistributedDataParallel 对象，并保存到 self._ddp
        self._ddp = DistributedDataParallel(self._param_list, **kwargs)
        # 弱引用 DDP 实例，目前仅用于测试目的
        replicate.state(self.module)._ddp_weakref = weakref.ref(self._ddp)

    def register_comm_hook(self) -> None:
        # 遍历存储的通信 hook 参数列表，并注册到 self._ddp
        for comm_args, comm_kwargs in self._comm_hook_args:
            self._ddp.register_comm_hook(*comm_args, **comm_kwargs)
        # 清空已注册的通信 hook 参数列表
        self._comm_hook_args.clear()

    def record_init_args(self, *args, **kwargs) -> None:
        # 记录初始化时的位置参数和关键字参数
        self._init_args = args
        self._init_kwargs = kwargs

    def forward_pre_hook(
        self, module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        # 如果存在初始化参数，则进行延迟初始化
        if self._init_args or self._init_kwargs:
            self.lazy_init()
        # 根据 self._no_sync 的值设置 self._ddp.require_backward_grad_sync 属性
        self._ddp.require_backward_grad_sync = not self._no_sync
        # 调用 DDP 对象的 _pre_forward 方法，并返回其结果
        return self._ddp._pre_forward(*args, **kwargs)

    def forward_post_hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        # 调用 DDP 对象的 _post_forward 方法，并返回其结果
        return self._ddp._post_forward(output)
# 定义一个未实现的深拷贝函数，抛出断言错误
def unimplemented_deepcopy(*args: Any, **kwargs: Any) -> NoReturn:
    raise AssertionError(
        "DDP does not support deepcopy. Please use state dict for serialization."
    )


# Follow the same pattern as FSDP/fully_shard
class DDP:
    def __new__(cls, *args, **kwargs):
        """
        重写 ``__new__`` 方法，以移除 DDP 类并直接构造原始类，
        用于像索引到容器模块的情况。
        """
        # 使用索引2，因为索引0是动态构造的 `DDP<...>` 类，
        # 索引1是 `DDP` 类本身
        orig_cls = cls.__mro__[2]
        return orig_cls.__new__(orig_cls, *args, **kwargs)

    def set_requires_gradient_sync(self, requires_gradient_sync: bool) -> None:
        """
        设置模块是否应该同步梯度。这可用于实现梯度累积而无需通信。

        Args:
            requires_gradient_sync (bool): 是否减少模块参数的梯度。
        """
        replicate.state(self)._no_sync = not requires_gradient_sync

    def register_comm_hook(self, *args, **kwargs) -> None:
        """
        注册通信钩子函数。

        Args:
            *args: 位置参数
            **kwargs: 关键字参数
        """
        replicate.state(self)._comm_hook_args.append((args, kwargs))


@contract(state_cls=_ReplicateState)
def replicate(
    module: nn.Module,
    ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
    **kwargs,
) -> nn.Module:
    r"""Replicates a module

    Args:
        module (torch.nn.Module): 需要复制的模块

    Example::
        >>> # xdoctest: +REQUIRES(module:torch._C._distributed_c10d)
        >>> module = nn.Linear(3, 3)
        >>> replicate(module)
    """
    torch._C._log_api_usage_once("torch.distributed.replicate")

    # TODO(fegin): using kwargs is not a good idea if we would like to make
    # replicate a formal API to replace DDP.
    if "device_id" in kwargs:
        if not isinstance(kwargs["device_id"], (int, torch.device)):
            raise RuntimeError(
                "Expected device_id to be int or torch.device, "
                f"but got {type(kwargs['device_id'])}"
            )

    # 检查是否已经完全分片
    if _is_fully_sharded(module):
        raise RuntimeError(
            "Cannot apply `replicate()` on a Module already managed by `fully_shard`"
        )

    # 初始化忽略的模块集合
    if ignored_modules is None:
        ignored_modules = {}
    else:
        ignored_modules = set(ignored_modules)

    # 获取模块的状态
    state = cast(_ReplicateState, replicate.state(module))
    # 注册前向预处理钩子
    module.register_forward_pre_hook(state.forward_pre_hook, with_kwargs=True)
    # 获取设备网格参数
    device_mesh = kwargs.get("device_mesh", None)
    # 如果设备网格不为空，则进行以下操作
    if device_mesh is not None:
        # 从torch.distributed.device_mesh中导入_mesh_resources模块
        from torch.distributed.device_mesh import _mesh_resources
        
        # 获取设备网格的父级网格，如果存在则执行以下操作
        if _mesh_resources.get_parent_mesh(device_mesh) is not None:
            # TODO: 这是一个临时的解决方法，以启用DDP + TP。
            # 我们应该在DDP中完成逻辑，使得2D实现是可靠的，并且state_dict可以直接使用。
            #
            # 这不会与DDP类中的操作冲突，因为传递给replicate的模块并非原始模块。
            # 从torch.distributed.tensor.parallel.ddp中导入_localize_dtensor和_reconstruct_dtensor函数
            from torch.distributed.tensor.parallel.ddp import (
                _localize_dtensor,
                _reconstruct_dtensor,
            )
            
            # 向模块注册_forward_pre_hook，用_reconstruct_dtensor作为回调函数
            module.register_forward_pre_hook(_reconstruct_dtensor)
            # 向模块注册_forward_hook，用_localize_dtensor作为回调函数
            module.register_forward_hook(_localize_dtensor)
    
    # 向模块注册state.forward_post_hook作为forward hook
    module.register_forward_hook(state.forward_post_hook)  # type: ignore[arg-type]
    
    # 使用state.record_init_args函数记录模块的初始化参数和其他传递的关键字参数
    state.record_init_args(module, ignored_modules, **kwargs)
    
    # 将DDP类置于模块类的左侧，以确保在方法解析顺序中具有最高优先级
    cls = module.__class__
    dct = {"__deepcopy__": unimplemented_deepcopy}
    # 创建新的类，继承自DDP和原始模块类，包含未实现的深拷贝方法__deepcopy__
    new_cls = type(f"DDP{cls.__name__}", (DDP, cls), dct)
    # 将模块的__class__属性设置为新创建的类new_cls
    module.__class__ = new_cls
    
    # 返回修改后的模块
    return module
# 检查给定的模块是否被标记为 fully_shard
def _is_fully_sharded(module: nn.Module) -> bool:
    # 获取模块的注册信息
    registry = _get_registry(module)
    # 如果没有注册信息，则返回 False
    if registry is None:
        return False
    # 检查注册信息中是否包含 "fully_shard" 标记
    return "fully_shard" in registry
```