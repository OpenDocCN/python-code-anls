# `.\pytorch\torch\nn\attention\__init__.py`

```
# mypy: allow-untyped-defs
""" This module contains functions and classes that alter the behavior of torch.nn.functional.scaled_dot_product_attention """
# 引入上下文管理器
import contextlib
# 引入类型注解
from typing import List, Union
# 引入警告模块中的warn函数
from warnings import warn

# 引入具体的模块和函数，用于定义 SDPBackend
from torch._C import _SDPBackend as SDPBackend
from torch.backends.cuda import (
    can_use_efficient_attention,
    can_use_flash_attention,
    cudnn_sdp_enabled,
    enable_cudnn_sdp,
    enable_flash_sdp,
    enable_math_sdp,
    enable_mem_efficient_sdp,
    flash_sdp_enabled,
    math_sdp_enabled,
    mem_efficient_sdp_enabled,
    SDPAParams,
)

# 定义公开的模块成员列表
__all__: List[str] = ["SDPBackend", "sdpa_kernel", "WARN_FOR_UNFUSED_KERNELS"]

# Note: [SDPA warnings]
# TODO: Consider using this for sdpa regardless of subclasses
# This only effects users of bias subclasses
# If this is set to True, we will warn the user if they are not using the fused kernels
# As well, it will raise warnings for all the reasons why the fused kernels can't be run.
# To set this to True, run
# torch.nn.attention.WARN_FOR_UNFUSED_KERNELS = True
# 控制是否警告未融合的内核使用情况的全局标志
WARN_FOR_UNFUSED_KERNELS = False


# Hacks for Sphinx documentation:
# https://stackoverflow.com/questions/38765577/overriding-sphinx-autodoc-alias-of-for-import-of-private-class
# 为了Sphinx文档生成，重新定义 SDPBackend
SDPBackend = SDPBackend
r"""An enum-like class that contains the different backends for scaled dot product attention.
    This backend class is designed to be used with the sdpa_kernel context manager.

    The following Enums are available:
        - ERROR: An error occurred when trying to determine the backend.
        - MATH: The math backend for scaled dot product attention.
        - FLASH_ATTENTION: The flash attention backend for scaled dot product attention.
        - EFFICIENT_ATTENTION: The efficient attention backend for scaled dot product attention.
        - CUDNN_ATTENTION: The cuDNN backend for scaled dot product attention.

    See :func:`torch.nn.attention.sdpa_kernel` for more details.

    .. warning:: This class is in beta and subject to change.
"""
# 设置 SDPBackend 的模块和名称属性，以便文档生成
SDPBackend.__module__ = __name__
SDPBackend.__name__ = "SDPBackend"


# 定义函数，用于引发内核警告
def _raise_kernel_warnings(params: SDPAParams) -> None:
    """
    If WARN_FOR_UNFUSED_KERNELS is set to True, this will raise warnings
    for all the reasons why the fused kernels can't be run. If using subclasses
    """
    # 如果 WARN_FOR_UNFUSED_KERNELS 为真，则根据条件引发警告
    if WARN_FOR_UNFUSED_KERNELS:
        # 检查是否可以使用高效的注意力机制
        if not can_use_efficient_attention(params):
            warn("Efficient attention can't be used because:")
            can_use_efficient_attention(params, True)
        # 检查是否可以使用闪光注意力机制
        if not can_use_flash_attention(params):
            warn("Flash attention can't be used because:")
            can_use_flash_attention(params, True)


# 上下文管理器，用于选择缩放点积注意力的后端
@contextlib.contextmanager
def sdpa_kernel(backends: Union[List[SDPBackend], SDPBackend]):
    r"""
    Context manager to select which backend to use for scaled dot product attention.

    .. warning:: This function is beta and subject to change.
"""
    """
    This context manager allows selection of backend(s) for scaled dot product attention.
    Upon exiting the context, it restores the previous state of the backend flags.
    
    Args:
        backend (Union[List[SDPBackend], SDPBackend]): A single backend or a list of backends for scaled dot product attention.
    
    Example:
    
    .. code-block:: python
    
        from torch.nn.functional import scaled_dot_product_attention
        from torch.nn.attention import SDPBackend, sdpa_kernel
    
        # Only enable flash attention backend
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            scaled_dot_product_attention(...)
    
        # Enable the Math or Efficient attention backends
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            scaled_dot_product_attention(...)
    
    This context manager dynamically configures the backends based on the provided argument(s).
    
    """
    
    # Validate the type of `backends` argument
    assert isinstance(
        backends, (list, SDPBackend)
    ), "Backend must be an instance of SDPBackend or a list of SDPBackend instances"
    
    # Convert a single backend into a list if necessary
    if isinstance(backends, SDPBackend):
        backends = [backends]
    
    # Convert the list of backends into a set to ensure uniqueness
    backends = set(backends)
    
    # Store the current states of the backend flags
    previous_cudnn: bool = cudnn_sdp_enabled()
    previous_flash: bool = flash_sdp_enabled()
    previous_mem_efficient: bool = mem_efficient_sdp_enabled()
    previous_math: bool = math_sdp_enabled()
    
    try:
        # Determine which backends to enable based on the provided list
        enable_cudnn = SDPBackend.CUDNN_ATTENTION in backends
        enable_flash = SDPBackend.FLASH_ATTENTION in backends
        enable_mem_efficient = SDPBackend.EFFICIENT_ATTENTION in backends
        enable_math = SDPBackend.MATH in backends
    
        # Enable or disable each backend accordingly
        enable_cudnn_sdp(enable_cudnn)
        enable_flash_sdp(enable_flash)
        enable_mem_efficient_sdp(enable_mem_efficient)
        enable_math_sdp(enable_math)
    
        # Yield an empty dictionary as the return value of the context manager
        yield {}
    
    finally:
        # Restore the previous states of the backend flags upon exiting the context manager
        enable_cudnn_sdp(previous_cudnn)
        enable_flash_sdp(previous_flash)
        enable_mem_efficient_sdp(previous_mem_efficient)
        enable_math_sdp(previous_math)
# 定义一个函数 _get_flash_version，返回一个字符串类型的值
def _get_flash_version() -> str:
    """This returns the closest matching tag for the flash attention backend"""
    # 返回固定的字符串 "2.5.6"，表示 Flash 注意力后端的最接近匹配标签
    return "2.5.6"
```