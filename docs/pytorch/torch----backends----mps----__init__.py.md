# `.\pytorch\torch\backends\mps\__init__.py`

```py
# mypy: allow-untyped-defs
# 导入 functools 模块中的 lru_cache，并起别名为 _lru_cache
from functools import lru_cache as _lru_cache

# 导入 typing 模块中的 Optional 类型
from typing import Optional

# 导入 torch 库
import torch

# 导入相对路径的库 ...library 中的 Library 类，并起别名为 _Library
from ...library import Library as _Library

# 定义模块的公开接口列表
__all__ = ["is_built", "is_available", "is_macos13_or_newer", "is_macos_or_newer"]


# 函数：判断 PyTorch 是否支持 MPS
def is_built() -> bool:
    r"""Return whether PyTorch is built with MPS support.

    Note that this doesn't necessarily mean MPS is available; just that
    if this PyTorch binary were run a machine with working MPS drivers
    and devices, we would be able to use it.
    """
    return torch._C._has_mps


# 装饰器函数：判断 MPS 是否可用，使用缓存功能
@_lru_cache
def is_available() -> bool:
    r"""Return a bool indicating if MPS is currently available."""
    return torch._C._mps_is_available()


# 装饰器函数：判断当前 MacOS 是否是给定版本或更新版本
@_lru_cache
def is_macos_or_newer(major: int, minor: int) -> bool:
    r"""Return a bool indicating whether MPS is running on given MacOS or newer."""
    return torch._C._mps_is_on_macos_or_newer(major, minor)


# 装饰器函数：判断当前 MacOS 是否是 MacOS 13 或更新版本
@_lru_cache
def is_macos13_or_newer(minor: int = 0) -> bool:
    r"""Return a bool indicating whether MPS is running on MacOS 13 or newer."""
    return torch._C._mps_is_on_macos_or_newer(13, minor)


# 可选类型变量 _lib 初始化为 None
_lib: Optional[_Library] = None


# 函数：注册 prims 作为 var_mean 和 group_norm 的实现
def _init():
    r"""Register prims as implementation of var_mean and group_norm."""
    global _lib
    # 如果 PyTorch 没有构建 MPS 支持或者 _lib 已经被初始化，则返回
    if is_built() is False or _lib is not None:
        return
    # 导入相关的库和函数
    from ..._decomp.decompositions import (
        native_group_norm_backward as _native_group_norm_backward,
    )
    from ..._refs import native_group_norm as _native_group_norm

    # 创建 _Library 实例，并将 native_group_norm 和 native_group_norm_backward 注册为 MPS 实现
    _lib = _Library("aten", "IMPL")
    _lib.impl("native_group_norm", _native_group_norm, "MPS")
    _lib.impl("native_group_norm_backward", _native_group_norm_backward, "MPS")
```