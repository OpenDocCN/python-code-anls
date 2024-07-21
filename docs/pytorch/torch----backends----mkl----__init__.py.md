# `.\pytorch\torch\backends\mkl\__init__.py`

```
# 引入 torch 库，用于操作 PyTorch 相关功能
# mypy: allow-untyped-defs
import torch

# 定义函数，判断当前 PyTorch 是否编译了 MKL 支持
def is_available():
    r"""Return whether PyTorch is built with MKL support."""
    return torch._C.has_mkl

# 定义常量，用于控制详细输出的开关状态
VERBOSE_OFF = 0
VERBOSE_ON = 1

# 定义上下文管理器类 verbose，用于控制一MKL的详细输出功能
class verbose:
    """
    On-demand oneMKL verbosing functionality.

    To make it easier to debug performance issues, oneMKL can dump verbose
    messages containing execution information like duration while executing
    the kernel. The verbosing functionality can be invoked via an environment
    variable named `MKL_VERBOSE`. However, this methodology dumps messages in
    all steps. Those are a large amount of verbose messages. Moreover, for
    investigating the performance issues, generally taking verbose messages
    for one single iteration is enough. This on-demand verbosing functionality
    makes it possible to control scope for verbose message dumping. In the
    following example, verbose messages will be dumped out for the second
    inference only.

    .. highlight:: python
    .. code-block:: python

        import torch
        model(data)
        with torch.backends.mkl.verbose(torch.backends.mkl.VERBOSE_ON):
            model(data)

    Args:
        level: Verbose level
            - ``VERBOSE_OFF``: Disable verbosing
            - ``VERBOSE_ON``:  Enable verbosing
    """

    # 初始化方法，接受一个参数 enable，用于指定详细输出是否启用
    def __init__(self, enable):
        self.enable = enable

    # 进入上下文时调用的方法
    def __enter__(self):
        # 如果详细输出被禁用，则直接返回
        if self.enable == VERBOSE_OFF:
            return
        # 调用 torch._C._verbose.mkl_set_verbose 方法设置 MKL 的详细输出模式
        st = torch._C._verbose.mkl_set_verbose(self.enable)
        # 断言是否成功设置 MKL 为详细输出模式，若失败则抛出异常
        assert (
            st
        ), "Failed to set MKL into verbose mode. Please consider to disable this verbose scope."
        return self

    # 退出上下文时调用的方法
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 在退出时将 MKL 设置为详细输出模式关闭
        torch._C._verbose.mkl_set_verbose(VERBOSE_OFF)
        # 返回 False，以便可能的异常继续被抛出
        return False
```