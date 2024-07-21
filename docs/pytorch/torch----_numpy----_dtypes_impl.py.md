# `.\pytorch\torch\_numpy\_dtypes_impl.py`

```
# mypy: ignore-errors
# 忽略类型检查错误

"""Dtypes/scalar type implementaions with torch dtypes.

Here `dtype` is always a torch.dtype, this module knows nothing about
scalar types, wrapper dtypes or anything like that. PyTorch only.
"""
# 从 collections 模块导入 namedtuple 类
from collections import namedtuple

# 导入 torch 库
import torch

# defaults : mimic NumPy, allow user control
# 定义一个命名元组 DefaultDTypes，包含浮点数、复数和整数的默认 dtype
DefaultDTypes = namedtuple(
    "DefaultDTypes", ["float_dtype", "complex_dtype", "int_dtype"]
)

# a global state
# 全局状态变量 _default_dtypes，用于存储默认 dtype
# 第一次调用 default_dtypes() 时设置该变量，避免导入 torch._dynamo.config 并创建循环引用
_default_dtypes = None


def default_dtypes():
    global _default_dtypes
    # 如果 _default_dtypes 为 None，则进行初始化
    if _default_dtypes is None:
        import torch._dynamo.config as config

        # 使用 config 模块中的配置创建 DefaultDTypes 命名元组
        _default_dtypes = DefaultDTypes(
            float_dtype=getattr(torch, config.numpy_default_float),
            complex_dtype=getattr(torch, config.numpy_default_complex),
            int_dtype=getattr(torch, config.numpy_default_int),
        )
        # 断言确保各 dtype 都是 torch 的 dtype 类型
        assert isinstance(_default_dtypes.float_dtype, torch.dtype)
        assert isinstance(_default_dtypes.complex_dtype, torch.dtype)
        assert isinstance(_default_dtypes.int_dtype, torch.dtype)
    return _default_dtypes


def get_default_dtype_for(dtype):
    """Default scalar type given sctype category."""
    # 根据输入的 dtype 返回默认的标量类型
    if dtype == torch.bool:
        return dtype
    if dtype.is_complex:
        return default_dtypes().complex_dtype
    if dtype.is_floating_point:
        return default_dtypes().float_dtype
    # else, it must be (some) integer
    # 若不是布尔类型、复数类型或浮点类型，则默认为整数类型
    return default_dtypes().int_dtype


# 从 _casting_dicts 模块导入 can_cast_impl 函数
from . import _casting_dicts as _cd


def can_cast_impl(from_torch_dtype, to_torch_dtype, casting):
    # 调用 _cd 模块中的 _can_cast_dict 字典，判断是否能够从 from_torch_dtype 转换到 to_torch_dtype
    return _cd._can_cast_dict[casting][from_torch_dtype][to_torch_dtype]


def result_type_impl(*tensors):
    # NB: torch dtypes here
    # 注意：这里的输入是 torch 的 dtype 类型
    # 获取第一个张量的 dtype
    dtyp = tensors[0].dtype
    # 如果只有一个张量，直接返回其 dtype
    if len(tensors) == 1:
        return dtyp

    # 遍历剩余张量，根据 _result_type_dict 字典更新 dtyp 的值
    for curr in tensors[1:]:
        dtyp = _cd._result_type_dict[dtyp][curr.dtype]

    return dtyp


def python_type_for_torch(dtyp):
    """Get a python scalar type a torch dtype"""
    # 根据 torch 的 dtype 返回相应的 Python 标量类型
    if dtyp.is_floating_point:
        typ = float
    elif dtyp.is_complex:
        typ = complex
    elif dtyp == torch.bool:
        typ = bool
    else:
        typ = int
    return typ


# ### NEP 50 helpers ###

# 定义标量类型元组 _SCALAR_TYPES，包含 int、bool、float 和 complex 类型
_SCALAR_TYPES = (int, bool, float, complex)

# 定义标量和符号类型元组 _SCALAR_AND_SYMBOLIC_TYPES，包含 _SCALAR_TYPES 和 torch 的 SymInt、SymFloat、SymBool 类型
_SCALAR_AND_SYMBOLIC_TYPES = (
    *_SCALAR_TYPES,
    torch.SymInt,
    torch.SymFloat,
    torch.SymBool,
)

# 定义仅适用于张量的 NEP 50 助手函数元组 _NEP50_FUNCS_TENSOR_ONLY，包含一些函数名字符串
_NEP50_FUNCS_TENSOR_ONLY = (
    "minimum",
    "maximum",
    "logaddexp",
    "logaddexp2",
    "lcm",
    "gcd",
    "hypot",
    "heaviside",
    "fmod",
    "fmin",
    "fmax",
    "copysign",
    "arctan2",
)


def is_scalar(x):
    # 判断 x 是否为标量类型（int、bool、float、complex 中的一种）
    return isinstance(x, _SCALAR_TYPES)


def is_scalar_or_symbolic(x):
    # 判断 x 是否为标量或符号类型（_SCALAR_AND_SYMBOLIC_TYPES 中的一种）
    return isinstance(x, _SCALAR_AND_SYMBOLIC_TYPES)


def _dtype_for_scalar(py_type):
    # 辅助函数，根据 Python 的标量类型返回对应的 torch dtype
    # 此函数在代码中未被使用，可能用于其他部分的实现
    # 返回一个字典，根据给定的 py_type 返回相应的 torch 类型
    return {
        bool: torch.bool,             # 如果 py_type 是 bool，返回 torch.bool 类型
        torch.SymBool: torch.bool,    # 如果 py_type 是 torch.SymBool，也返回 torch.bool 类型
        int: torch.int64,             # 如果 py_type 是 int，返回 torch.int64 类型
        torch.SymInt: torch.int64,    # 如果 py_type 是 torch.SymInt，返回 torch.int64 类型
        float: torch.float64,         # 如果 py_type 是 float，返回 torch.float64 类型
        torch.SymFloat: torch.float64,# 如果 py_type 是 torch.SymFloat，返回 torch.float64 类型
        complex: torch.complex128,    # 如果 py_type 是 complex，返回 torch.complex128 类型
    }[py_type]                        # 返回字典中与 py_type 对应的值
# 根据输入的对象 x 的类型，返回其数据类型，如果是 torch.Tensor 则返回其 dtype
def _dtype_for_scalar_or_tensor(x):
    return x.dtype if isinstance(x, torch.Tensor) else _dtype_for_scalar(type(x))


# 检查输入的对象 x 是否是浮点数或者浮点数张量
def is_float_or_fp_tensor(x):
    return _dtype_for_scalar_or_tensor(x).is_floating_point


# 检查输入的对象 x 是否是复数或者复数张量
def is_complex_or_complex_tensor(x):
    return _dtype_for_scalar_or_tensor(x).is_complex


# 根据给定的 dtype 返回一个类别码，用于分类数据类型
def _category(dtype):
    return {
        torch.bool: 0,
        torch.SymBool: 0,
        # int
        torch.uint8: 1,
        torch.int8: 1,
        torch.int16: 1,
        torch.int32: 1,
        torch.int64: 1,
        torch.SymInt: 1,
        # float
        torch.float16: 2,
        torch.float32: 2,
        torch.float64: 2,
        torch.SymFloat: 2,
        # complex
        torch.complex64: 3,
        torch.complex128: 3,
    }[dtype]


# 如果 x1 或者 x2 是 Python 标量，则根据 NEP 50 将其类型提升为张量
def nep50_to_tensors(x1, x2, handle_weaks, function_name):
    """If either of inputs is a python scalar, type-promote with NEP 50."""
    
    # 将输入的标量 scalar 转换为张量，如果没有指定 dtype，则使用标量的类型来推断
    def to_tensor(scalar, dtype=None):
        if dtype is None:
            dtype = _dtype_for_scalar(type(scalar))
            dtype = get_default_dtype_for(dtype)
        return torch.as_tensor(scalar, dtype=dtype)

    # 判断 x1 和 x2 是否是弱类型，即非张量
    x1_is_weak = not isinstance(x1, torch.Tensor)
    x2_is_weak = not isinstance(x2, torch.Tensor)
    
    # 如果不处理弱类型或者 x1、x2 都是弱类型，则直接返回转换后的 x1 和 x2
    if not handle_weaks or (x1_is_weak and x2_is_weak):
        x1 = to_tensor(x1) if x1_is_weak else x1
        x2 = to_tensor(x2) if x2_is_weak else x2
        return x1, x2

    # 如果一个是弱类型，一个不是，按照 NEP 50 的规定进行类型提升
    assert x1_is_weak != x2_is_weak

    weak, not_weak = (x1, x2) if x1_is_weak else (x2, x1)

    # 查找弱类型的数据类型
    weak_dtype = _dtype_for_scalar(type(weak))

    # 获取类别码
    cat_weak = _category(weak_dtype)
    cat_not_weak = _category(not_weak.dtype)

    dt = not_weak.dtype if cat_weak <= cat_not_weak else None

    # 处理特殊情况：复数 + float32
    if weak_dtype.is_complex and not_weak.dtype == torch.float32:
        dt = torch.complex64

    # 检查溢出：在 PyTorch 中，uint8(-1) 会环绕到 255，但 NEP50 要求引发异常
    if cat_weak == 1 and cat_not_weak == 1:
        iinfo = torch.iinfo(not_weak.dtype)
        if not (iinfo.min <= weak <= iinfo.max):
            raise OverflowError(
                f"Python integer {weak} out of bounds for {not_weak.dtype}"
            )

    # 如果弱类型的数据类型不匹配或者函数名在 _NEP50_FUNCS_TENSOR_ONLY 中，则将其转换为张量
    if weak_dtype != dt or function_name in _NEP50_FUNCS_TENSOR_ONLY:
        weak = to_tensor(weak, dt)

    return (weak, not_weak) if x1_is_weak else (not_weak, weak)
```