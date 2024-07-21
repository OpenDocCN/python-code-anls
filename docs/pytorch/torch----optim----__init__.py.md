# `.\pytorch\torch\optim\__init__.py`

```
"""
:mod:`torch.optim` is a package implementing various optimization algorithms.

Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can also be easily integrated in the
future.
"""

# 导入需要的优化器和相关工具模块
from torch.optim import lr_scheduler, swa_utils
# 导入各种优化器类
from torch.optim.adadelta import Adadelta
from torch.optim.adagrad import Adagrad
from torch.optim.adam import Adam
from torch.optim.adamax import Adamax
from torch.optim.adamw import AdamW
from torch.optim.asgd import ASGD
from torch.optim.lbfgs import LBFGS
from torch.optim.nadam import NAdam
from torch.optim.optimizer import Optimizer
from torch.optim.radam import RAdam
from torch.optim.rmsprop import RMSprop
from torch.optim.rprop import Rprop
from torch.optim.sgd import SGD
from torch.optim.sparse_adam import SparseAdam

# 删除不需要的优化器定义，用于避免未使用的变量警告
del adadelta  # type: ignore[name-defined] # noqa: F821
del adagrad  # type: ignore[name-defined] # noqa: F821
del adam  # type: ignore[name-defined] # noqa: F821
del adamw  # type: ignore[name-defined] # noqa: F821
del sparse_adam  # type: ignore[name-defined] # noqa: F821
del adamax  # type: ignore[name-defined] # noqa: F821
del asgd  # type: ignore[name-defined] # noqa: F821
del sgd  # type: ignore[name-defined] # noqa: F821
del radam  # type: ignore[name-defined] # noqa: F821
del rprop  # type: ignore[name-defined] # noqa: F821
del rmsprop  # type: ignore[name-defined] # noqa: F821
del optimizer  # type: ignore[name-defined] # noqa: F821
del nadam  # type: ignore[name-defined] # noqa: F821
del lbfgs  # type: ignore[name-defined] # noqa: F821
```