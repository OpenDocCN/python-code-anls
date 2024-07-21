# `.\pytorch\torch\_refs\_conversions.py`

```
# mypy: allow-untyped-defs
# 导入 torch 库
import torch
# 导入 torch._prims_common 模块，并重命名为 utils
import torch._prims_common as utils

# 从 torch._decomp 模块中导入 register_decomposition 函数
from torch._decomp import register_decomposition

# 从 torch._prims_common 模块中导入 TensorLikeType 类型
from torch._prims_common import TensorLikeType
# 从 torch._prims_common.wrappers 模块中导入 out_wrapper 函数
from torch._prims_common.wrappers import out_wrapper
# 导入 torch._refs 模块中的 _broadcast_shapes 函数
from torch._refs import _broadcast_shapes

# Data conversion references.
#
# 注意：此模块打破了通常的 torch 命名方案，其中 _refs.foo.bar 是 torch.foo.bar 的引用。
# 下面的定义不是 _refs/__init__.py 的一部分，以避免与 Python 内置类型（如 int）发生名称冲突。

__all__ = [
    # 数据类型
    "bfloat16",
    "bool",
    "byte",
    "cdouble",
    "cfloat",
    "chalf",
    "char",
    "double",
    "float",
    "half",
    "int",
    "long",
    "short",
    # 杂项
    "complex",
    "polar",
]


# 创建数据类型转换方法的内部函数
def _make_conversion_method(name: str, dtype: torch.dtype):
    # 定义转换方法，将 self 转换为指定的数据类型 dtype
    def fn(
        self: TensorLikeType, memory_format: torch.memory_format = torch.preserve_format
    ) -> TensorLikeType:
        # 调用 self 的 to 方法，将数据类型转换为 dtype，并保留内存格式
        return self.to(dtype, memory_format=memory_format)  # type: ignore[call-overload]

    # 设置函数名
    fn.__name__ = name
    return fn


# 定义各种数据类型转换方法
bfloat16 = _make_conversion_method("bfloat16", torch.bfloat16)
bool = _make_conversion_method("bool", torch.bool)
byte = _make_conversion_method("byte", torch.uint8)
cdouble = _make_conversion_method("cdouble", torch.cdouble)
cfloat = _make_conversion_method("cfloat", torch.cfloat)
chalf = _make_conversion_method("chalf", torch.complex32)
char = _make_conversion_method("char", torch.int8)
double = _make_conversion_method("double", torch.double)
float = _make_conversion_method("float", torch.float)
half = _make_conversion_method("half", torch.half)
int = _make_conversion_method("int", torch.int)
long = _make_conversion_method("long", torch.long)
short = _make_conversion_method("short", torch.short)


# 注册 complex 函数的分解操作
@register_decomposition(torch._ops.ops.aten.complex)
# 注意：由于语义不同，complex 函数禁用类型推断测试。
# exact_dtype 用于与核心中的 complex_check_dtype 兼容。
@out_wrapper(exact_dtype=True)
# 定义 complex 函数，接受实部 real 和虚部 imag 作为参数，返回复数张量
def complex(real: TensorLikeType, imag: TensorLikeType) -> TensorLikeType:
    # 允许的数据类型包括 float32、float64 和 float16
    allowed_dtypes = (torch.float32, torch.float64, torch.float16)
    # 检查 real 和 imag 的数据类型是否在允许的范围内
    torch._check(
        real.dtype in allowed_dtypes and imag.dtype in allowed_dtypes,
        lambda: (
            f"Expected both inputs to be Half, Float or Double tensors but got "
            f"{real.dtype} and {imag.dtype}"
        ),
    )
    # 检查 real 和 imag 的数据类型是否一致
    torch._check(
        real.dtype == imag.dtype,
        lambda: (
            f"Expected object of scalar type {real.dtype} but got "
            f"scalar type {imag.dtype} for second argument"
        ),
    )
    # 计算结果的数据类型，根据 real 的数据类型确定对应的复数数据类型
    result_dtype = utils.corresponding_complex_dtype(real.dtype)  # type: ignore[arg-type]
    # 计算 real 和 imag 张量的广播形状
    common_shape = _broadcast_shapes(real.shape, imag.shape)
    # 使用 real 对象的属性创建一个新的空数组，具有指定的形状和数据类型
    result = real.new_empty(
        common_shape,        # 公共的数组形状作为参数传递
        dtype=result_dtype,  # 结果数组的数据类型
        layout=real.layout,  # 使用 real 对象的布局属性
        device=real.device,  # 使用 real 对象的设备属性
        # pin_memory=real.is_pinned(),  # NYI
    )
    # 将实部数据（real）赋值给 result 对象的 real 属性
    result.real = real
    # 将虚部数据（imag）赋值给 result 对象的 imag 属性
    result.imag = imag
    # 返回创建的 result 对象作为函数的结果
    return result
# 将函数注册为 torch._ops.ops.aten.polar 的分解函数
@register_decomposition(torch._ops.ops.aten.polar)

# 注解: polar 函数由于语义不同，禁用了类型提升测试
# exact_dtype 是为了与 core 中的 complex_check_dtype 兼容
# 在返回结果时确保与输入张量的精确数据类型相匹配
@out_wrapper(exact_dtype=True)
def polar(abs: TensorLikeType, angle: TensorLikeType) -> TensorLikeType:
    # 创建一个复数张量，其实部为 abs，虚部为 angle
    result = torch.complex(abs, angle)
    # 设置结果的实部为 abs * cos(angle)
    result.real = abs * torch.cos(angle)
    # 设置结果的虚部为 abs * sin(angle)
    result.imag = abs * torch.sin(angle)
    # 返回创建的复数张量结果
    return result
```