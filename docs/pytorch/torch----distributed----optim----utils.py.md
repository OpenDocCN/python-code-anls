# `.\pytorch\torch\distributed\optim\utils.py`

```
# mypy: allow-untyped-defs
# 导入类型相关的模块
from typing import Type

# 导入 torch 的优化器模块
from torch import optim

# 导入各种功能优化器的具体实现类
from .functional_adadelta import _FunctionalAdadelta
from .functional_adagrad import _FunctionalAdagrad
from .functional_adam import _FunctionalAdam
from .functional_adamax import _FunctionalAdamax
from .functional_adamw import _FunctionalAdamW
from .functional_rmsprop import _FunctionalRMSprop
from .functional_rprop import _FunctionalRprop
from .functional_sgd import _FunctionalSGD

# dict，将用户传入的优化器类映射到对应的功能优化器类
functional_optim_map = {
    optim.Adagrad: _FunctionalAdagrad,
    optim.Adam: _FunctionalAdam,
    optim.AdamW: _FunctionalAdamW,
    optim.SGD: _FunctionalSGD,
    optim.Adadelta: _FunctionalAdadelta,
    optim.RMSprop: _FunctionalRMSprop,
    optim.Rprop: _FunctionalRprop,
    optim.Adamax: _FunctionalAdamax,
}

# 注册一个新的功能优化器到 functional_optim_map 中的接口
def register_functional_optim(key, optim):
    """
    Interface to insert a new functional optimizer to functional_optim_map
    ``fn_optim_key`` and ``fn_optimizer`` are user defined. The optimizer and key
    need not be of :class:`torch.optim.Optimizer` (e.g. for custom optimizers)
    Example::
        >>> # import the new functional optimizer
        >>> # xdoctest: +SKIP
        >>> from xyz import fn_optimizer
        >>> from torch.distributed.optim.utils import register_functional_optim
        >>> fn_optim_key = "XYZ_optim"
        >>> register_functional_optim(fn_optim_key, fn_optimizer)
    """
    # 如果 key 不在 functional_optim_map 中，则将新的功能优化器插入其中
    if key not in functional_optim_map:
        functional_optim_map[key] = optim

# 将标准优化器转换为功能优化器的接口函数
def as_functional_optim(optim_cls: Type, *args, **kwargs):
    try:
        # 获取对应的功能优化器类
        functional_cls = functional_optim_map[optim_cls]
    except KeyError as e:
        # 如果找不到对应的功能优化器类，则抛出异常
        raise ValueError(
            f"Optimizer {optim_cls} does not have a functional " f"counterpart!"
        ) from e

    # 使用功能优化器类创建实例并返回
    return _create_functional_optim(functional_cls, *args, **kwargs)

# 创建功能优化器实例的内部函数
def _create_functional_optim(functional_optim_cls: Type, *args, **kwargs):
    # 调用功能优化器类的构造函数创建实例
    return functional_optim_cls(
        [],
        *args,
        **kwargs,
        _allow_empty_param_list=True,
    )
```