# `.\pytorch\torch\distributed\optim\__init__.py`

```py
"""
:mod:`torch.distributed.optim` exposes DistributedOptimizer, which takes a list
of remote parameters (:class:`~torch.distributed.rpc.RRef`) and runs the
optimizer locally on the workers where the parameters live.  The distributed
optimizer can use any of the local optimizer :ref:`optimizer-algorithms` to
apply the gradients on each worker.
"""
# 导入警告模块，用于捕获和处理警告信息
import warnings

# 导入 PyTorch 库
import torch
from torch import optim

# 导入各种优化器的功能实现模块
from .apply_optimizer_in_backward import (
    _apply_optimizer_in_backward,
    _get_in_backward_optimizers,
)
from .functional_adadelta import _FunctionalAdadelta
from .functional_adagrad import _FunctionalAdagrad
from .functional_adam import _FunctionalAdam
from .functional_adamax import _FunctionalAdamax
from .functional_adamw import _FunctionalAdamW
from .functional_rmsprop import _FunctionalRMSprop
from .functional_rprop import _FunctionalRprop
from .functional_sgd import _FunctionalSGD
from .named_optimizer import _NamedOptimizer
from .utils import as_functional_optim

# 使用警告过滤器，确保始终显示警告信息
with warnings.catch_warnings():
    warnings.simplefilter("always")
    # 发出警告，说明 TorchScript 对功能性优化器的支持已弃用，并在将来的 PyTorch 版本中将被移除
    warnings.warn(
        "`TorchScript` support for functional optimizers is deprecated "
        "and will be removed in a future PyTorch release. "
        "Consider using the `torch.compile` optimizer instead.",
        DeprecationWarning,
        stacklevel=2,
    )

# 如果 torch._C 模块有 "_rpc_init" 属性，导入分布式优化器
if hasattr(torch._C, "_rpc_init"):
    from .optimizer import DistributedOptimizer

# 导入其他优化器类
from .post_localSGD_optimizer import PostLocalSGDOptimizer
from .zero_redundancy_optimizer import ZeroRedundancyOptimizer

# 定义模块中公开的符号列表
__all__ = [
    "as_functional_optim",
    "DistributedOptimizer",
    "PostLocalSGDOptimizer",
    "ZeroRedundancyOptimizer",
]
```