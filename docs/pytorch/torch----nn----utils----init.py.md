# `.\pytorch\torch\nn\utils\init.py`

```py
# mypy: allow-untyped-defs
# 导入 inspect 模块，用于检查函数签名
import inspect

# 导入 torch 库，用于神经网络模块
import torch


# 定义一个函数 skip_init，用于实例化模块对象但不初始化其参数和缓冲区
def skip_init(module_cls, *args, **kwargs):
    r"""
    给定一个模块类对象和 args / kwargs，实例化模块但不初始化其参数和缓冲区。

    这在初始化较慢或需要进行自定义初始化时非常有用，使得默认初始化变得不必要。但由于
    该函数的实现方式，存在一些注意事项：

    1. 模块在构造函数中必须接受一个 `device` 参数，该参数将传递给任何在构造过程中创建的参数
    或缓冲区。

    2. 模块在其构造函数中不能对参数进行任何计算，除了初始化（即从 :mod:`torch.nn.init` 中的函数）。

    如果满足这些条件，模块将以参数和缓冲区未初始化的状态实例化，就像使用 :func:`torch.empty` 创建一样。

    Args:
        module_cls: 类对象；应该是 :class:`torch.nn.Module` 的子类
        args: 传递给模块构造函数的 args
        kwargs: 传递给模块构造函数的 kwargs

    Returns:
        以未初始化参数和缓冲区的实例化模块

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> import torch
        >>> m = torch.nn.utils.skip_init(torch.nn.Linear, 5, 1)
        >>> m.weight
        Parameter containing:
        tensor([[0.0000e+00, 1.5846e+29, 7.8307e+00, 2.5250e-29, 1.1210e-44]],
               requires_grad=True)
        >>> m2 = torch.nn.utils.skip_init(torch.nn.Linear, in_features=6, out_features=1)
        >>> m2.weight
        Parameter containing:
        tensor([[-1.4677e+24,  4.5915e-41,
```