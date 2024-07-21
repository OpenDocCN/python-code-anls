# `.\pytorch\torch\testing\_internal\common_dtype.py`

```
# mypy: ignore-errors

# 引入 List 类型用于类型提示
from typing import List

# 引入 PyTorch 库
import torch


# 用于验证每个给定的 dtype 是否为 torch.dtype
def _validate_dtypes(*dtypes):
    for dtype in dtypes:
        assert isinstance(dtype, torch.dtype)
    return dtypes


# 表示一个与 PyTorch dispatch 宏对应的元组的类
class _dispatch_dtypes(tuple):
    def __add__(self, other):
        assert isinstance(other, tuple)
        return _dispatch_dtypes(tuple.__add__(self, other))


# 空类型的实例
_empty_types = _dispatch_dtypes(())


# 返回空类型实例的函数
def empty_types():
    return _empty_types


# 浮点类型的实例，包括 float32 和 float64
_floating_types = _dispatch_dtypes((torch.float32, torch.float64))


# 返回浮点类型实例的函数
def floating_types():
    return _floating_types


# 浮点类型及其半精度（half）的实例
_floating_types_and_half = _floating_types + (torch.half,)


# 返回浮点类型及其半精度的函数
def floating_types_and_half():
    return _floating_types_and_half


# 浮点类型及其它指定 dtypes 的实例
def floating_types_and(*dtypes):
    return _floating_types + _validate_dtypes(*dtypes)


# 浮点类型和复数类型的实例
_floating_and_complex_types = _floating_types + (torch.cfloat, torch.cdouble)


# 返回浮点类型和复数类型实例的函数
def floating_and_complex_types():
    return _floating_and_complex_types


# 浮点类型和复数类型及其它指定 dtypes 的实例
def floating_and_complex_types_and(*dtypes):
    return _floating_and_complex_types + _validate_dtypes(*dtypes)


# 双精度浮点类型的实例
_double_types = _dispatch_dtypes((torch.float64, torch.complex128))


# 返回双精度浮点类型实例的函数
def double_types():
    return _double_types


# 积分类型，不包含 uint16/uint32/uint64，为了向后兼容性
_integral_types = _dispatch_dtypes(
    (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
)


# 返回积分类型实例的函数
def integral_types():
    return _integral_types


# 积分类型及其它指定 dtypes 的实例
def integral_types_and(*dtypes):
    return _integral_types + _validate_dtypes(*dtypes)


# 所有支持的类型，包括浮点和积分类型
_all_types = _floating_types + _integral_types


# 返回所有支持类型实例的函数
def all_types():
    return _all_types


# 所有支持类型及其它指定 dtypes 的实例
def all_types_and(*dtypes):
    return _all_types + _validate_dtypes(*dtypes)


# 复数类型的实例
_complex_types = _dispatch_dtypes((torch.cfloat, torch.cdouble))


# 返回复数类型实例的函数
def complex_types():
    return _complex_types


# 复数类型及其它指定 dtypes 的实例
def complex_types_and(*dtypes):
    return _complex_types + _validate_dtypes(*dtypes)


# 所有支持类型及复数类型的实例
_all_types_and_complex = _all_types + _complex_types


# 返回所有支持类型及复数类型实例的函数
def all_types_and_complex():
    return _all_types_and_complex


# 所有支持类型、复数类型及其它指定 dtypes 的实例
def all_types_and_complex_and(*dtypes):
    return _all_types_and_complex + _validate_dtypes(*dtypes)


# 所有支持类型及半精度类型的实例
_all_types_and_half = _all_types + (torch.half,)


# 返回所有支持类型及半精度类型实例的函数
def all_types_and_half():
    return _all_types_and_half


# 自定义 dtypes 的函数，创建包含任意 dtypes 的列表
def custom_types(*dtypes):
    return _empty_types + _validate_dtypes(*dtypes)


# 以下函数用于方便测试套件，并没有对应的 C++ dispatch 宏


# 参考 AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS。
# 返回所有可能的 dtypes 列表
def get_all_dtypes(
    include_half=True,
    include_bfloat16=True,
    include_bool=True,
    include_complex=True,
    include_complex32=False,
    include_qint=False
) -> List[torch.dtype]:
    # 获取所有整数类型的数据类型列表
    dtypes = get_all_int_dtypes() + get_all_fp_dtypes(
        include_half=include_half, include_bfloat16=include_bfloat16
    )
    # 如果需要包含布尔类型，则将 torch.bool 加入到数据类型列表中
    if include_bool:
        dtypes.append(torch.bool)
    # 如果需要包含复数类型，则将所有复数数据类型加入到数据类型列表中
    if include_complex:
        dtypes += get_all_complex_dtypes(include_complex32)
    # 如果需要包含量化整数类型，则将所有量化整数数据类型加入到数据类型列表中
    if include_qint:
        dtypes += get_all_qint_dtypes()
    # 返回最终的数据类型列表
    return dtypes
# 返回所有与数学相关的 PyTorch 数据类型列表
def get_all_math_dtypes(device) -> List[torch.dtype]:
    # 调用获取所有整数数据类型的函数
    return (
        get_all_int_dtypes()
        # 加上获取所有浮点数数据类型的函数的返回值
        + get_all_fp_dtypes(
            include_half=device.startswith("cuda"), include_bfloat16=False
        )
        # 再加上获取所有复数数据类型的函数的返回值
        + get_all_complex_dtypes()
    )


# 返回所有复数数据类型的列表
def get_all_complex_dtypes(include_complex32=False) -> List[torch.dtype]:
    # 如果包括 torch.complex32，则返回包含 torch.complex32, torch.complex64 和 torch.complex128 的列表
    return (
        [torch.complex32, torch.complex64, torch.complex128]
        if include_complex32
        # 否则返回包含 torch.complex64 和 torch.complex128 的列表
        else [torch.complex64, torch.complex128]
    )


# 返回所有整数数据类型的列表
def get_all_int_dtypes() -> List[torch.dtype]:
    return [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]


# 返回所有浮点数数据类型的列表
def get_all_fp_dtypes(include_half=True, include_bfloat16=True) -> List[torch.dtype]:
    dtypes = [torch.float32, torch.float64]
    # 如果包括半精度浮点数类型
    if include_half:
        dtypes.append(torch.float16)
    # 如果包括 BF16 浮点数类型
    if include_bfloat16:
        dtypes.append(torch.bfloat16)
    return dtypes


# 返回所有量化整数数据类型的列表
def get_all_qint_dtypes() -> List[torch.dtype]:
    return [torch.qint8, torch.quint8, torch.qint32, torch.quint4x2, torch.quint2x4]


# 映射浮点数数据类型到对应的复数数据类型的字典
float_to_corresponding_complex_type_map = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}
```