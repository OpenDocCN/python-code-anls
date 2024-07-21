# `.\pytorch\torch\distributed\algorithms\_optimizer_overlap\optimizer_overlap.py`

```
# mypy: allow-untyped-defs
import inspect  # 导入inspect模块，用于获取对象的信息
from abc import ABC, abstractmethod  # 导入ABC和abstractmethod，用于定义抽象基类和抽象方法
from typing import Dict, Type  # 导入Dict和Type类型，用于类型提示

from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook  # 导入allreduce_hook函数
from torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks import (
    _hook_then_optimizer,
    _OptimizerHookState,
)  # 导入_hook_then_optimizer和_OptimizerHookState函数

from torch.distributed.fsdp import FullyShardedDataParallel  # 导入FullyShardedDataParallel类
from torch.distributed.optim import as_functional_optim  # 导入as_functional_optim函数
from torch.nn.parallel import DistributedDataParallel  # 导入DistributedDataParallel类
from torch.optim import Optimizer  # 导入Optimizer类


# Contains the mappings between the regular and overlapped optimizer types.
_registered_overlapped_optims: Dict[Type, Type] = {}  # 创建一个空字典，用于存储普通优化器和重叠优化器类型之间的映射关系


def register_overlapped(optim_cls):
    def decorator(target_overlapped_optim_cls):
        if target_overlapped_optim_cls in _registered_overlapped_optims:
            raise ValueError(
                f"{target_overlapped_optim_cls} already registered with optim_cls "
                f"{_registered_overlapped_optims[optim_cls]} {optim_cls}, trying to"
                f"re-register it for {optim_cls} is not supported."
            )
        _registered_overlapped_optims[optim_cls] = target_overlapped_optim_cls
        return target_overlapped_optim_cls

    return decorator


class OverlappedOptimizer(ABC):
    def __init__(self, optim_cls: Type) -> None:
        """
        Initialize the OverlappedOptimizer.

        Overlappedoptimizer is a base class that child classes can implement to
        specify how different optimizers will register themselves with DDP.
        """
        self.optim_cls = optim_cls  # 初始化优化器类属性为给定的优化器类型

    @abstractmethod
    def register_ddp(self, ddp: DistributedDataParallel) -> None:
        """Registers the overlapped optimizer with DDP."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support overlapped DDP."
        )  # 抽象方法，子类需实现在DistributedDataParallel中注册重叠优化器的逻辑

    @abstractmethod
    def register_fsdp(self, fsdp: FullyShardedDataParallel) -> None:
        """Registers the overlapped optimizer with FSDP."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support overlapped FSDP."
        )  # 抽象方法，子类需实现在FullyShardedDataParallel中注册重叠优化器的逻辑


@register_overlapped(Optimizer)
class _OverlappedStandardOptimizer(OverlappedOptimizer):
    """Overlaps a regular ``Optimizer``."""

    def __init__(self, optim_cls: Type, params, *optim_args, **optim_kwargs) -> None:
        super().__init__(optim_cls)  # 调用父类构造函数初始化优化器类属性
        f_optim = as_functional_optim(self.optim_cls, *optim_args, **optim_kwargs)  # 将优化器转换为函数式优化器
        self._opt_hook_state = _OptimizerHookState(f_optim, params)  # 初始化优化器钩子状态对象

    def register_ddp(self, ddp_inst: DistributedDataParallel):
        # NOTE: using a custom communication hook and fused optimizer is not
        # yet supported.
        ddp_inst.register_comm_hook(  # 在DistributedDataParallel实例中注册通信钩子
            None,  # 封装后的钩子状态
            _hook_then_optimizer(allreduce_hook, self._opt_hook_state),  # 将allreduce_hook和优化器钩子状态组合注册
        )

    # TODO: register_fsdp once FSDP supports communication hook.
    # 定义一个方法，用于将重叠优化器注册到FSDP（FullyShardedDataParallel）中
    def register_fsdp(self, fsdp: FullyShardedDataParallel) -> None:
        """Register the overlapped optimizer with FSDP."""
        # 抛出一个未实现的错误，指示当前类不支持重叠的FSDP
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support overlapped FSDP."
        )
def _as_overlapped_optim(optim_cls: Type, params, *args, **kwargs):
    """Return a new ``OverlappedOptimizer`` instance that supports ``optim_cls``."""
    # 遍历优化器类及其父类的方法解析顺序
    for clz in inspect.getmro(optim_cls):
        try:
            # 尝试使用注册的重叠优化器类来创建实例
            return _registered_overlapped_optims[clz](
                optim_cls, params, *args, **kwargs
            )
        except KeyError:
            pass

    # 如果未找到注册的重叠优化器类，回退到标准的重叠优化器
    # 这会在用户尝试使用不受支持的优化器时引发错误
    return _OverlappedStandardOptimizer(optim_cls, params, *args, **kwargs)
```