# `.\pytorch\torch\optim\_multi_tensor\__init__.py`

```py
"""
:mod:`torch.optim._multi_tensor` is a package implementing various optimization algorithms.

Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""
# 导入 partialmethod 函数，用于创建带有预设参数的方法
from functools import partialmethod

# 导入 PyTorch 中的优化器模块
from torch import optim


# 定义一个函数 partialclass，用于创建具有预设参数的优化器类
def partialclass(cls, *args, **kwargs):  # noqa: D103
    # 创建一个新的类 NewCls，继承自参数传入的 cls 类
    class NewCls(cls):
        # 重写 __init__ 方法，使用 partialmethod 将传入的参数绑定到原始 __init__ 方法上
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


# 使用 partialclass 函数创建多个优化器类的变体，这些类将 foreach 参数设置为 True
Adam = partialclass(optim.Adam, foreach=True)
AdamW = partialclass(optim.AdamW, foreach=True)
NAdam = partialclass(optim.NAdam, foreach=True)
SGD = partialclass(optim.SGD, foreach=True)
RAdam = partialclass(optim.RAdam, foreach=True)
RMSprop = partialclass(optim.RMSprop, foreach=True)
Rprop = partialclass(optim.Rprop, foreach=True)
ASGD = partialclass(optim.ASGD, foreach=True)
Adamax = partialclass(optim.Adamax, foreach=True)
Adadelta = partialclass(optim.Adadelta, foreach=True)
Adagrad = partialclass(optim.Adagrad, foreach=True)
```