# `.\pytorch\torch\__config__.py`

```py
# mypy: allow-untyped-defs
# 导入 torch 模块
import torch

# 定义函数 show，返回一个可读的字符串，描述 PyTorch 的配置信息
def show():
    """
    Return a human-readable string with descriptions of the
    configuration of PyTorch.
    """
    # 调用 torch._C._show_config() 函数，返回 PyTorch 配置信息的字符串
    return torch._C._show_config()


# TODO: In principle, we could provide more structured version/config
# information here. For now only CXX_FLAGS is exposed, as Timer
# uses them.

# 定义函数 _cxx_flags，返回构建 PyTorch 时使用的 CXX_FLAGS
def _cxx_flags():
    """Returns the CXX_FLAGS used when building PyTorch."""
    # 调用 torch._C._cxx_flags() 函数，返回用于构建 PyTorch 的 CXX_FLAGS
    return torch._C._cxx_flags()


# 定义函数 parallel_info，返回详细的字符串，描述并行设置
def parallel_info():
    r"""Returns detailed string with parallelization settings"""
    # 调用 torch._C._parallel_info() 函数，返回包含并行设置详细信息的字符串
    return torch._C._parallel_info()
```