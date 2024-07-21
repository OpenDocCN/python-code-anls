# `.\pytorch\torch\_functorch\deprecated.py`

```
"""
The APIs in this file are exposed as `functorch.*`. They are thin wrappers
around the torch.func.* APIs that have deprecation warnings -- we're trying
to move people to the torch.func.* equivalents.

NB: We don't use *args, **kwargs in the signatures because that changes the
documentation.
"""

import textwrap                              # 导入文本包装模块，用于处理文档字符串格式
import warnings                              # 导入警告模块，用于发出警告信息
from typing import Any, Callable, Optional, Tuple, Union  # 导入类型提示相关模块

import torch._functorch.apis as apis         # 导入functorch的API接口
import torch._functorch.eager_transforms as _impl   # 导入functorch的急切转换函数
import torch._functorch.make_functional as _nn_impl  # 导入functorch的函数化模块
import torch.nn as nn                        # 导入PyTorch的神经网络模块
from torch._functorch.eager_transforms import argnums_t   # 导入functorch的参数编号类型
from torch._functorch.vmap import in_dims_t, out_dims_t   # 导入functorch的vmap映射输入和输出维度类型


def get_warning(api, new_api=None, replace_newlines=False):
    """
    根据提供的API名称生成警告信息。

    Args:
    - api: 被弃用的API名称
    - new_api: 新API的名称，默认为`torch.func.{api}`
    - replace_newlines: 是否替换警告信息中的换行符，默认为False

    Returns:
    - 警告信息字符串
    """
    if new_api is None:
        new_api = f"torch.func.{api}"
    warning = (
        f"We've integrated functorch into PyTorch. As the final step of the \n"
        f"integration, `functorch.{api}` is deprecated as of PyTorch \n"
        f"2.0 and will be deleted in a future version of PyTorch >= 2.3. \n"
        f"Please use `{new_api}` instead; see the PyTorch 2.0 release notes \n"
        f"and/or the `torch.func` migration guide for more details \n"
        f"https://pytorch.org/docs/main/func.migrating.html"
    )
    if replace_newlines:
        warning = warning.replace("\n", "")
    return warning


def warn_deprecated(api, new_api=None):
    """
    发出关于API弃用的警告信息。

    Args:
    - api: 被弃用的API名称
    - new_api: 新API的名称，默认为`torch.func.{api}`
    """
    warning = get_warning(api, new_api, replace_newlines=True)
    warnings.warn(warning, FutureWarning, stacklevel=3)


def setup_docs(functorch_api, torch_func_api=None, new_api_name=None):
    """
    设置API文档，包括警告信息。

    Args:
    - functorch_api: functorch中的API对象
    - torch_func_api: 对应的torch.func中的API对象，默认为functorch中同名对象
    - new_api_name: 新API的名称，默认为`torch.func.{functorch_api.__name__}`
    """
    api_name = functorch_api.__name__
    if torch_func_api is None:
        torch_func_api = getattr(_impl, api_name)
    # See https://docs.python.org/3/using/cmdline.html#cmdoption-OO
    if torch_func_api.__doc__ is None:
        return

    warning = get_warning(api_name, new_api_name)
    warning_note = "\n.. warning::\n\n" + textwrap.indent(warning, "    ")
    warning_note = textwrap.indent(warning_note, "    ")
    functorch_api.__doc__ = torch_func_api.__doc__ + warning_note


def vmap(
    func: Callable,
    in_dims: in_dims_t = 0,
    out_dims: out_dims_t = 0,
    randomness: str = "error",
    *,
    chunk_size=None,
) -> Callable:
    """
    包装了vmap函数，发出了关于其弃用的警告信息。

    Args:
    - func: 要映射的函数
    - in_dims: 输入维度描述
    - out_dims: 输出维度描述
    - randomness: 随机性处理方式，默认为"error"
    - chunk_size: 分块大小参数

    Returns:
    - 包装后的函数对象
    """
    warn_deprecated("vmap", "torch.vmap")
    return apis.vmap(func, in_dims, out_dims, randomness, chunk_size=chunk_size)


def grad(func: Callable, argnums: argnums_t = 0, has_aux: bool = False) -> Callable:
    """
    包装了grad函数，发出了关于其弃用的警告信息。

    Args:
    - func: 要求梯度的函数
    - argnums: 参数编号
    - has_aux: 是否有辅助参数，默认为False

    Returns:
    - 包装后的函数对象
    """
    warn_deprecated("grad")
    return apis.grad(func, argnums, has_aux)


def grad_and_value(
    func: Callable, argnums: argnums_t = 0, has_aux: bool = False
) -> Callable:
    """
    包装了grad_and_value函数，发出了关于其弃用的警告信息。

    Args:
    - func: 要求梯度和值的函数
    - argnums: 参数编号
    - has_aux: 是否有辅助参数，默认为False

    Returns:
    - 包装后的函数对象
    """
    warn_deprecated("grad_and_value")
    return apis.grad_and_value(func, argnums, has_aux)


def vjp(func: Callable, *primals, has_aux: bool = False):
    """
    包装了vjp函数，发出了关于其弃用的警告信息。

    Args:
    - func: 要求Jacobian向量积的函数
    - *primals: 主要参数
    - has_aux: 是否有辅助参数，默认为False

    Returns:
    - vjp计算的结果
    """
    warn_deprecated("vjp")
    return _impl.vjp(func, *primals, has_aux=has_aux)


def jvp(
    func: Callable,
    primals: Any,
    tangents: Any,
    *,
    strict: bool = False,
    # 声明一个变量 `has_aux`，其类型为布尔型，并初始化为 `False`
    has_aux: bool = False,
def jvp(
    func: Callable,
    primals,
    tangents,
    *,
    strict=False,
    has_aux=False,
):
    warn_deprecated("jvp")
    return _impl.jvp(func, primals, tangents, strict=strict, has_aux=has_aux)
# 警告已废弃，使用"jvp"。调用底层实现的jvp函数，计算函数func在primals和tangents处的Jacobian向量积。

def jacrev(
    func: Callable,
    argnums: Union[int, Tuple[int]] = 0,
    *,
    has_aux=False,
    chunk_size: Optional[int] = None,
    _preallocate_and_copy=False,
):
    warn_deprecated("jacrev")
    return _impl.jacrev(
        func,
        argnums,
        has_aux=has_aux,
        chunk_size=chunk_size,
        _preallocate_and_copy=_preallocate_and_copy,
    )
# 警告已废弃，使用"jacrev"。调用底层实现的jacrev函数，计算函数func关于指定参数argnums的反向 Jacobian 矩阵。

def jacfwd(
    func: Callable,
    argnums: argnums_t = 0,
    has_aux: bool = False,
    *,
    randomness: str = "error",
):
    warn_deprecated("jacfwd")
    return _impl.jacfwd(func, argnums, has_aux, randomness=randomness)
# 警告已废弃，使用"jacfwd"。调用底层实现的jacfwd函数，计算函数func关于指定参数argnums的前向 Jacobian 矩阵。

def hessian(func, argnums=0):
    warn_deprecated("hessian")
    return _impl.hessian(func, argnums=argnums)
# 警告已废弃，使用"hessian"。调用底层实现的hessian函数，计算函数func关于指定参数argnums的 Hessian 矩阵。

def functionalize(func: Callable, *, remove: str = "mutations") -> Callable:
    warn_deprecated("functionalize")
    return _impl.functionalize(func, remove=remove)
# 警告已废弃，使用"functionalize"。调用底层实现的functionalize函数，将函数func转换为无状态的函数（去除mutations）。

def make_functional(model: nn.Module, disable_autograd_tracking: bool = False):
    warn_deprecated("make_functional", "torch.func.functional_call")
    return _nn_impl.make_functional(model, disable_autograd_tracking)
# 警告已废弃，使用"make_functional"，建议使用torch.func.functional_call。调用底层实现的make_functional函数，将PyTorch模型转换为无状态的函数形式。

def make_functional_with_buffers(
    model: nn.Module, disable_autograd_tracking: bool = False
):
    warn_deprecated("make_functional_with_buffers", "torch.func.functional_call")
    return _nn_impl.make_functional_with_buffers(model, disable_autograd_tracking)
# 警告已废弃，使用"make_functional_with_buffers"，建议使用torch.func.functional_call。调用底层实现的make_functional_with_buffers函数，将PyTorch模型及其缓冲区转换为无状态的函数形式。

def combine_state_for_ensemble(models):
    warn_deprecated("combine_state_for_ensemble", "torch.func.stack_module_state")
    return _nn_impl.combine_state_for_ensemble(models)
# 警告已废弃，使用"combine_state_for_ensemble"，建议使用torch.func.stack_module_state。调用底层实现的combine_state_for_ensemble函数，将多个模型的状态合并为一个用于集成的状态。

setup_docs(vmap, apis.vmap, "torch.vmap")
setup_docs(grad, apis.grad)
setup_docs(grad_and_value, apis.grad_and_value)
setup_docs(vjp)
setup_docs(jvp)
setup_docs(jacrev)
setup_docs(jacfwd)
setup_docs(hessian)
setup_docs(functionalize)
setup_docs(make_functional, _nn_impl.make_functional, "torch.func.functional_call")
setup_docs(
    make_functional_with_buffers, _nn_impl.make_functional, "torch.func.functional_call"
)
setup_docs(
    combine_state_for_ensemble,
    _nn_impl.combine_state_for_ensemble,
    "torch.func.stack_module_state",
)
# 设置文档，为各个函数设置文档信息。
```