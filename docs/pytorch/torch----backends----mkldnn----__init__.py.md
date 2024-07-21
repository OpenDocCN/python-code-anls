# `.\pytorch\torch\backends\mkldnn\__init__.py`

```py
# mypy: allow-untyped-defs
# 导入系统模块
import sys
# 导入上下文管理器模块
from contextlib import contextmanager

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 导入PyTorch库
import torch
# 导入PyTorch的后端模块
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule


def is_available():
    r"""Return whether PyTorch is built with MKL-DNN support."""
    # 检查PyTorch是否构建了MKL-DNN支持，返回布尔值
    return torch._C._has_mkldnn


VERBOSE_OFF = 0
VERBOSE_ON = 1
VERBOSE_ON_CREATION = 2


class verbose:
    """
    On-demand oneDNN (former MKL-DNN) verbosing functionality.

    To make it easier to debug performance issues, oneDNN can dump verbose
    messages containing information like kernel size, input data size and
    execution duration while executing the kernel. The verbosing functionality
    can be invoked via an environment variable named `DNNL_VERBOSE`. However,
    this methodology dumps messages in all steps. Those are a large amount of
    verbose messages. Moreover, for investigating the performance issues,
    generally taking verbose messages for one single iteration is enough.
    This on-demand verbosing functionality makes it possible to control scope
    for verbose message dumping. In the following example, verbose messages
    will be dumped out for the second inference only.

    .. highlight:: python
    .. code-block:: python

        import torch
        model(data)
        with torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON):
            model(data)

    Args:
        level: Verbose level
            - ``VERBOSE_OFF``: Disable verbosing
            - ``VERBOSE_ON``:  Enable verbosing
            - ``VERBOSE_ON_CREATION``: Enable verbosing, including oneDNN kernel creation
    """

    def __init__(self, level):
        # 初始化函数，设置verbose等级
        self.level = level

    def __enter__(self):
        # 进入上下文管理器时执行的操作
        if self.level == VERBOSE_OFF:
            return
        # 设置MKLDNN为verbose模式，并检查是否设置成功
        st = torch._C._verbose.mkldnn_set_verbose(self.level)
        assert (
            st
        ), "Failed to set MKLDNN into verbose mode. Please consider to disable this verbose scope."
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出上下文管理器时执行的操作，关闭MKLDNN的verbose模式
        torch._C._verbose.mkldnn_set_verbose(VERBOSE_OFF)
        return False


def set_flags(_enabled, _deterministic=None):
    # 设置MKLDNN的标志位，并返回原始的标志位元组
    orig_flags = (torch._C._get_mkldnn_enabled(), torch._C._get_mkldnn_deterministic())
    torch._C._set_mkldnn_enabled(_enabled)
    if _deterministic is not None:
        torch._C._set_mkldnn_deterministic(_deterministic)
    return orig_flags


@contextmanager
def flags(enabled=False, deterministic=False):
    # 使用上下文管理器设置MKLDNN的标志位，并返回原始的标志位
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled, deterministic)
    try:
        yield
    finally:
        # 在上下文结束时恢复MKLDNN的原始标志位
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


class MkldnnModule(PropModule):
    def __init__(self, m, name):
        # 初始化MkldnnModule类，继承自PropModule
        super().__init__(m, name)

    enabled = ContextProp(torch._C._get_mkldnn_enabled, torch._C._set_mkldnn_enabled)
    # 创建一个上下文属性对象，用于控制 MKLDNN（Math Kernel Library for Deep Neural Networks）的确定性设置。
    # 第一个参数是获取 MKLDNN 的确定性设置的函数。
    # 第二个参数是设置 MKLDNN 的确定性设置的函数。
    deterministic = ContextProp(
        torch._C._get_mkldnn_deterministic, torch._C._set_mkldnn_deterministic
    )
# 如果类型检查开启，则定义两个变量 `enabled` 和 `deterministic`，它们的类型为 `ContextProp`
if TYPE_CHECKING:
    enabled: ContextProp
    deterministic: ContextProp

# 将当前模块替换为经过 MkldnnModule 包装后的模块，其中传入的参数为当前模块本身和当前模块的名称
sys.modules[__name__] = MkldnnModule(sys.modules[__name__], __name__)
```