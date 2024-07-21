# `.\pytorch\torch\jit\_builtins.py`

```
# mypy: allow-untyped-defs
# 导入复数数学运算模块
import cmath
# 导入标准数学运算模块
import math
# 导入警告模块
import warnings

# 导入有序字典数据结构
from collections import OrderedDict
# 导入类型提示模块
from typing import Dict, Optional

# 导入 PyTorch 深度学习框架
import torch
# 导入 PyTorch 的 CUDA 加速模块
import torch.backends.cudnn as cudnn

# 导入自定义的工具函数
from ..nn.modules.utils import _list_with_default, _pair, _quadruple, _single, _triple

# 可选的内置表，字典的键是整数，值是字符串或空值
_builtin_table: Optional[Dict[int, str]] = None

# 包含内置操作的模块集合，用于类型检查时忽略定义的属性错误
_modules_containing_builtins = (
    torch, torch._C._nn, torch._C._fft, torch._C._linalg,
    torch._C._nested, torch._C._sparse, torch._C._special
)  # type: ignore[attr-defined] # noqa: B950

# 内置操作列表，每个元素都是一个包含函数和操作名的元组
_builtin_ops = [
    (_pair, "aten::_pair"),
    (_quadruple, "aten::_quadruple"),
    (_single, "aten::_single"),
    (_triple, "aten::_triple"),
    (_list_with_default, "aten::list_with_default"),
    (OrderedDict, "aten::dict"),
    (dict, "aten::dict"),
    (cudnn.is_acceptable, "aten::cudnn_is_acceptable"),
    (math.ceil, "aten::ceil"),
    (math.copysign, "aten::copysign"),
    (math.erf, "aten::erf"),
    (math.erfc, "aten::erfc"),
    (math.exp, "aten::exp"),
    (math.expm1, "aten::expm1"),
    (math.fabs, "aten::fabs"),
    (math.floor, "aten::floor"),
    (math.gamma, "aten::gamma"),
    (math.lgamma, "aten::lgamma"),
    (math.log, "aten::log"),
    (math.log10, "aten::log10"),
    (math.log1p, "aten::log1p"),
    (math.pow, "aten::pow"),
    (math.sqrt, "aten::sqrt"),
    (math.isnan, "aten::isnan"),
    (math.asinh, "aten::asinh"),
    (math.atanh, "aten::atanh"),
    (math.cosh, "aten::cosh"),
    (math.sinh, "aten::sinh"),
    (math.tanh, "aten::tanh"),
    (math.acos, "aten::acos"),
    (math.asin, "aten::asin"),
    (math.atan, "aten::atan"),
    (math.atan2, "aten::atan2"),
    (math.cos, "aten::cos"),
    (math.sin, "aten::sin"),
    (math.tan, "aten::tan"),
    (math.asinh, "aten::asinh"),
    (math.atanh, "aten::atanh"),
    (math.acosh, "aten::acosh"),
    (math.fmod, "aten::fmod"),
    (math.modf, "aten::modf"),
    (math.factorial, "aten::factorial"),
    (math.frexp, "aten::frexp"),
    (math.isinf, "aten::isinf"),
    (math.degrees, "aten::degrees"),
    (math.radians, "aten::radians"),
    (cmath.isnan, "aten::isnan"),
    (cmath.isfinite, "aten::isfinite"),
    (cmath.isinf, "aten::isinf"),
    (cmath.phase, "aten::angle"),
    (cmath.rect, "aten::polar"),
    (cmath.log, "aten::log"),
    (cmath.log10, "aten::log10"),
    (cmath.sqrt, "aten::sqrt"),
    (cmath.exp, "aten::exp"),
    (cmath.sin, "aten::sin"),
    (cmath.tan, "aten::tan"),
    (cmath.cos, "aten::cos"),
    (cmath.asin, "aten::asin"),
    (cmath.acos, "aten::acos"),
    (cmath.atan, "aten::atan"),
    (cmath.sinh, "aten::sinh"),
    (cmath.cosh, "aten::cosh"),
    (cmath.tanh, "aten::tanh"),
    (cmath.asinh, "aten::asinh"),
    (cmath.acosh, "aten::acosh"),
    (cmath.atanh, "aten::atanh"),
    (math.ldexp, "aten::ldexp"),
    (torch._assert, "aten::_assert"),
    (torch.autograd.grad, "aten::grad"),
    (torch.autograd.backward, "aten::backward"),
    (torch._C._infer_size, "aten::_infer_size"),
    # 将 torch.nn.functional._no_grad_embedding_renorm_ 映射到对应的 ATen 函数 aten::_no_grad_embedding_renorm_
    (torch.nn.functional._no_grad_embedding_renorm_, "aten::_no_grad_embedding_renorm_"),  # type: ignore[attr-defined]
    
    # 将 torch.nn.functional.assert_int_or_pair 映射到对应的 ATen 函数 aten::_assert_int_or_pair
    (torch.nn.functional.assert_int_or_pair, "aten::_assert_int_or_pair"),
    
    # 将 torch.nn.init._no_grad_fill_ 映射到对应的 ATen 函数 aten::_no_grad_fill_
    (torch.nn.init._no_grad_fill_, "aten::_no_grad_fill_"),
    
    # 将 torch.nn.init._no_grad_normal_ 映射到对应的 ATen 函数 aten::_no_grad_normal_
    (torch.nn.init._no_grad_normal_, "aten::_no_grad_normal_"),
    
    # 将 torch.nn.init._no_grad_uniform_ 映射到对应的 ATen 函数 aten::_no_grad_uniform_
    (torch.nn.init._no_grad_uniform_, "aten::_no_grad_uniform_"),
    
    # 将 torch.nn.init._no_grad_zero_ 映射到对应的 ATen 函数 aten::_no_grad_zero_
    (torch.nn.init._no_grad_zero_, "aten::_no_grad_zero_"),
    
    # 将 torch._C._get_tracing_state 映射到对应的 ATen 函数 aten::_get_tracing_state
    (torch._C._get_tracing_state, "aten::_get_tracing_state"),
    
    # 将 torch._C._get_cpu_capability 映射到对应的 ATen 函数 aten::_get_cpu_capability
    (torch._C._get_cpu_capability, "aten::_get_cpu_capability"),
    
    # 将 warnings.warn 映射到对应的 ATen 函数 aten::warn
    (warnings.warn, "aten::warn"),
    
    # 将 torch._VF.stft 映射到对应的 ATen 函数 aten::stft
    (torch._VF.stft, "aten::stft"),  # type: ignore[attr-defined]
    
    # 将 torch._VF.istft 映射到对应的 ATen 函数 aten::istft
    (torch._VF.istft, "aten::istft"),  # type: ignore[attr-defined]
    
    # 将 torch._VF.cdist 映射到对应的 ATen 函数 aten::cdist
    (torch._VF.cdist, "aten::cdist"),  # type: ignore[attr-defined]
    
    # 将 torch._VF.norm 映射到对应的 ATen 函数 aten::norm
    (torch._VF.norm, "aten::norm"),  # type: ignore[attr-defined]
    
    # 将 torch._VF.unique_dim 映射到对应的 ATen 函数 aten::unique_dim
    (torch._VF.unique_dim, "aten::unique_dim"),
    
    # 将 torch._VF.unique_consecutive 映射到对应的 ATen 函数 aten::unique_consecutive
    (torch._VF.unique_consecutive, "aten::unique_consecutive"),  # type: ignore[attr-defined]
    
    # 将 torch._VF.nuclear_norm 映射到对应的 ATen 函数 aten::nuclear_norm
    (torch._VF.nuclear_norm, "aten::nuclear_norm"),
    
    # 将 torch._VF.frobenius_norm 映射到对应的 ATen 函数 aten::frobenius_norm
    (torch._VF.frobenius_norm, "aten::frobenius_norm"),
    
    # 将 torch._VF.tensordot 映射到对应的 ATen 函数 aten::tensordot
    (torch._VF.tensordot, "aten::tensordot"),  # type: ignore[attr-defined]
# ops in torch.functional are bound to torch
# in these cases, we want to resolve the function to their python implementation
# instead looking up a builtin "aten::" schema
# torch.functional 中的操作与 torch 绑定在一起
# 在这些情况下，我们希望将函数解析为它们的 Python 实现，而不是查找内置的 "aten::" 模式

def _gen_torch_functional_registered_ops():
    # eventually ops should encompass all of torch/functional.py, (torch.functional.__all__)
    # but we are currently only able to compile some of the functions. additionally,
    # some functions directly map to their aten:: implementations.
    # TODO: add support for more ops
    # 最终，ops 应该包含 torch/functional.py 中的所有内容（torch.functional.__all__）
    # 但目前我们只能编译其中的一些函数。此外，一些函数直接映射到它们的 aten:: 实现。
    # TODO: 添加对更多操作的支持
    ops = [
        "stft",
        "istft",
        "lu",
        "cdist",
        "norm",
        "unique",
        "unique_consecutive",
        "tensordot",
    ]
    return {getattr(torch.functional, name) for name in ops}

_functional_registered_ops = _gen_torch_functional_registered_ops()


def _is_special_functional_bound_op(fn):
    return fn in _functional_registered_ops
    # 检查函数是否为特殊的 torch.functional 绑定操作的函数

# lazily built to ensure the correct initialization order
# 延迟构建以确保正确的初始化顺序
def _get_builtin_table():
    global _builtin_table
    if _builtin_table is not None:
        return _builtin_table
    _builtin_table = {}

    def register_all(mod):
        for name in dir(mod):
            v = getattr(mod, name)
            if (
                callable(v)
                and not _is_special_functional_bound_op(v)
                and v is not torch.no_grad
                and v is not torch.autocast
            ):
                # Fixup inconsistency in segment_reduce
                if name == "_segment_reduce":
                    name = name[1:]
                _builtin_ops.append((v, "aten::" + name))

    for mod in _modules_containing_builtins:
        register_all(mod)

    _builtin_ops.append((math.gcd, "aten::gcd"))
    _builtin_ops.append((math.isfinite, "aten::isfinite"))
    _builtin_ops.append((math.remainder, "aten::mathremainder"))  # type: ignore[attr-defined]

    import torch.distributed.autograd as dist_autograd

    if dist_autograd.is_available():
        _builtin_ops.append((dist_autograd.get_gradients, "aten::get_gradients"))
        _builtin_ops.append((dist_autograd.backward, "aten::dist_backward"))

    # populate the _builtin_table from _builtin_ops
    # 从 _builtin_ops 中填充 _builtin_table
    for builtin, aten_op in _builtin_ops:
        _builtin_table[id(builtin)] = aten_op

    return _builtin_table


def _register_builtin(fn, op):
    _get_builtin_table()[id(fn)] = op
    # 将函数与对应的 aten:: 操作注册到 _builtin_table 中

def _find_builtin(fn):
    return _get_builtin_table().get(id(fn))
    # 查找给定函数在 _builtin_table 中对应的 aten:: 操作
```