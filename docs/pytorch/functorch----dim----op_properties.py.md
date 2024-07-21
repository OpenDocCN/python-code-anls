# `.\pytorch\functorch\dim\op_properties.py`

```py
# 导入 torch 库，用于进行张量操作和神经网络功能
import torch

# 定义可通过更快路径执行的逐点运算方法列表
tensor_magic_methods = ["add", ""]
# 包含可以逆向执行的逐点运算方法列表
pointwise_magic_methods_with_reverse = (
    "add",
    "sub",
    "mul",
    "floordiv",
    "div",
    "truediv",
    "mod",
    "pow",
    "lshift",
    "rshift",
    "and",
    "or",
    "xor",
)
# 包含所有逐点运算方法的列表，包括逆向执行和其他方法
pointwise_magic_methods = (
    *(x for m in pointwise_magic_methods_with_reverse for x in (m, "r" + m)),
    "eq",
    "gt",
    "le",
    "lt",
    "ge",
    "gt",
    "ne",
    "neg",
    "pos",
    "abs",
    "invert",
    "iadd",
    "isub",
    "imul",
    "ifloordiv",
    "idiv",
    "itruediv",
    "imod",
    "ipow",
    "ilshift",
    "irshift",
    "iand",
    "ior",
    "ixor",
    "int",
    "long",
    "float",
    "complex",
)

# 创建包含所有逐点方法的特殊方法列表
pointwise_methods = (*(f"__{m}__" for m in pointwise_magic_methods),)

# 定义包含 torch.Tensor 类中所有逐点方法的元组
pointwise = (
    *(getattr(torch.Tensor, m) for m in pointwise_methods),
    torch.nn.functional.dropout,
    torch.where,
    torch.Tensor.abs,
    torch.abs,
    torch.Tensor.acos,
    torch.acos,
    torch.Tensor.acosh,
    torch.acosh,
    torch.Tensor.add,
    torch.add,
    torch.Tensor.addcdiv,
    torch.addcdiv,
    torch.Tensor.addcmul,
    torch.addcmul,
    torch.Tensor.addr,
    torch.addr,
    torch.Tensor.angle,
    torch.angle,
    torch.Tensor.asin,
    torch.asin,
    torch.Tensor.asinh,
    torch.asinh,
    torch.Tensor.atan,
    torch.atan,
    torch.Tensor.atan2,
    torch.atan2,
    torch.Tensor.atanh,
    torch.atanh,
    torch.Tensor.bitwise_and,
    torch.bitwise_and,
    torch.Tensor.bitwise_left_shift,
    torch.bitwise_left_shift,
    torch.Tensor.bitwise_not,
    torch.bitwise_not,
    torch.Tensor.bitwise_or,
    torch.bitwise_or,
    torch.Tensor.bitwise_right_shift,
    torch.bitwise_right_shift,
    torch.Tensor.bitwise_xor,
    torch.bitwise_xor,
    torch.Tensor.ceil,
    torch.ceil,
    torch.celu,
    torch.nn.functional.celu,
    torch.Tensor.clamp,
    torch.clamp,
    torch.Tensor.clamp_max,
    torch.clamp_max,
    torch.Tensor.clamp_min,
    torch.clamp_min,
    torch.Tensor.copysign,
    torch.copysign,
    torch.Tensor.cos,
    torch.cos,
    torch.Tensor.cosh,
    torch.cosh,
    torch.Tensor.deg2rad,
    torch.deg2rad,
    torch.Tensor.digamma,
    torch.digamma,
    torch.Tensor.div,
    torch.div,
    torch.dropout,
    torch.nn.functional.dropout,
    torch.nn.functional.elu,
    torch.Tensor.eq,
    torch.eq,
    torch.Tensor.erf,
    torch.erf,
    torch.Tensor.erfc,
    torch.erfc,
    torch.Tensor.erfinv,
    torch.erfinv,
    torch.Tensor.exp,
    torch.exp,
    torch.Tensor.exp2,
    torch.exp2,
    torch.Tensor.expm1,
    torch.expm1,
    torch.feature_dropout,
    torch.Tensor.float_power,
    torch.float_power,
    torch.Tensor.floor,
    torch.floor,
    # 返回 torch.Tensor 类的 floor_divide 方法
    torch.Tensor.floor_divide,
    
    # 返回 torch 模块的 floor_divide 函数
    torch.floor_divide,
    
    # 返回 torch.Tensor 类的 fmod 方法
    torch.Tensor.fmod,
    
    # 返回 torch 模块的 fmod 函数
    torch.fmod,
    
    # 返回 torch.Tensor 类的 frac 方法
    torch.Tensor.frac,
    
    # 返回 torch 模块的 frac 函数
    torch.frac,
    
    # 返回 torch.Tensor 类的 frexp 方法
    torch.Tensor.frexp,
    
    # 返回 torch 模块的 frexp 函数
    torch.frexp,
    
    # 返回 torch.Tensor 类的 gcd 方法
    torch.Tensor.gcd,
    
    # 返回 torch 模块的 gcd 函数
    torch.gcd,
    
    # 返回 torch.Tensor 类的 ge 方法
    torch.Tensor.ge,
    
    # 返回 torch 模块的 ge 函数
    torch.ge,
    
    # 返回 torch.nn.functional 模块的 gelu 函数
    torch.nn.functional.gelu,
    
    # 返回 torch.nn.functional 模块的 glu 函数
    torch.nn.functional.glu,
    
    # 返回 torch.Tensor 类的 gt 方法
    torch.Tensor.gt,
    
    # 返回 torch 模块的 gt 函数
    torch.gt,
    
    # 返回 torch.Tensor 类的 hardshrink 方法
    torch.Tensor.hardshrink,
    
    # 返回 torch 模块的 hardshrink 函数
    torch.hardshrink,
    
    # 返回 torch.nn.functional 模块的 hardshrink 函数
    torch.nn.functional.hardshrink,
    
    # 返回 torch.nn.functional 模块的 hardsigmoid 函数
    torch.nn.functional.hardsigmoid,
    
    # 返回 torch.nn.functional 模块的 hardswish 函数
    torch.nn.functional.hardswish,
    
    # 返回 torch.nn.functional 模块的 hardtanh 函数
    torch.nn.functional.hardtanh,
    
    # 返回 torch.Tensor 类的 heaviside 方法
    torch.Tensor.heaviside,
    
    # 返回 torch 模块的 heaviside 函数
    torch.heaviside,
    
    # 返回 torch.Tensor 类的 hypot 方法
    torch.Tensor.hypot,
    
    # 返回 torch 模块的 hypot 函数
    torch.hypot,
    
    # 返回 torch.Tensor 类的 i0 方法
    torch.Tensor.i0,
    
    # 返回 torch 模块的 i0 函数
    torch.i0,
    
    # 返回 torch.Tensor 类的 igamma 方法
    torch.Tensor.igamma,
    
    # 返回 torch 模块的 igamma 函数
    torch.igamma,
    
    # 返回 torch.Tensor 类的 igammac 方法
    torch.Tensor.igammac,
    
    # 返回 torch 模块的 igammac 函数
    torch.igammac,
    
    # 返回 torch.Tensor 类的 isclose 方法
    torch.Tensor.isclose,
    
    # 返回 torch 模块的 isclose 函数
    torch.isclose,
    
    # 返回 torch.Tensor 类的 isfinite 方法
    torch.Tensor.isfinite,
    
    # 返回 torch 模块的 isfinite 函数
    torch.isfinite,
    
    # 返回 torch.Tensor 类的 isinf 方法
    torch.Tensor.isinf,
    
    # 返回 torch 模块的 isinf 函数
    torch.isinf,
    
    # 返回 torch.Tensor 类的 isnan 方法
    torch.Tensor.isnan,
    
    # 返回 torch 模块的 isnan 函数
    torch.isnan,
    
    # 返回 torch.Tensor 类的 isneginf 方法
    torch.Tensor.isneginf,
    
    # 返回 torch 模块的 isneginf 函数
    torch.isneginf,
    
    # 返回 torch.Tensor 类的 isposinf 方法
    torch.Tensor.isposinf,
    
    # 返回 torch 模块的 isposinf 函数
    torch.isposinf,
    
    # 返回 torch.Tensor 类的 isreal 方法
    torch.Tensor.isreal,
    
    # 返回 torch 模块的 isreal 函数
    torch.isreal,
    
    # 返回 torch.Tensor 类的 kron 方法
    torch.Tensor.kron,
    
    # 返回 torch 模块的 kron 函数
    torch.kron,
    
    # 返回 torch.Tensor 类的 lcm 方法
    torch.Tensor.lcm,
    
    # 返回 torch 模块的 lcm 函数
    torch.lcm,
    
    # 返回 torch.Tensor 类的 ldexp 方法
    torch.Tensor.ldexp,
    
    # 返回 torch 模块的 ldexp 函数
    torch.ldexp,
    
    # 返回 torch.Tensor 类的 le 方法
    torch.Tensor.le,
    
    # 返回 torch 模块的 le 函数
    torch.le,
    
    # 返回 torch.nn.functional 模块的 leaky_relu 函数
    torch.nn.functional.leaky_relu,
    
    # 返回 torch.Tensor 类的 lerp 方法
    torch.Tensor.lerp,
    
    # 返回 torch 模块的 lerp 函数
    torch.lerp,
    
    # 返回 torch.Tensor 类的 lgamma 方法
    torch.Tensor.lgamma,
    
    # 返回 torch 模块的 lgamma 函数
    torch.lgamma,
    
    # 返回 torch.Tensor 类的 log 方法
    torch.Tensor.log,
    
    # 返回 torch 模块的 log 函数
    torch.log,
    
    # 返回 torch.Tensor 类的 log10 方法
    torch.Tensor.log10,
    
    # 返回 torch 模块的 log10 函数
    torch.log10,
    
    # 返回 torch.Tensor 类的 log1p 方法
    torch.Tensor.log1p,
    
    # 返回 torch 模块的 log1p 函数
    torch.log1p,
    
    # 返回 torch.Tensor 类的 log2 方法
    torch.Tensor.log2,
    
    # 返回 torch 模块的 log2 函数
    torch.log2,
    
    # 返回 torch.nn.functional 模块的 logsigmoid 函数
    torch.nn.functional.logsigmoid,
    
    # 返回 torch.Tensor 类的 logical_and 方法
    torch.Tensor.logical_and,
    
    # 返回 torch 模块的 logical_and 函数
    torch.logical_and,
    
    # 返回 torch.Tensor 类的 logical_not 方法
    torch.Tensor.logical_not,
    
    # 返回 torch 模块的 logical_not 函数
    torch.logical_not,
    
    # 返回 torch.Tensor 类的 logical_or 方法
    torch.Tensor.logical_or,
    
    # 返回 torch 模块的 logical_or 函数
    torch.logical_or,
    
    # 返回 torch.Tensor 类的 logical_xor 方法
    torch.Tensor.logical_xor,
    
    # 返回 torch 模块的 logical_xor 函数
    torch.logical_xor,
    
    # 返回 torch.Tensor 类的 logit 方法
    torch.Tensor.logit,
    
    # 返回 torch 模块的 logit 函数
    torch.logit,
    
    # 返回 torch.Tensor 类的 lt 方法
    torch.Tensor.lt,
    
    # 返回 torch 模块的 lt 函数
    torch.lt,
    
    # 返回 torch.Tensor 类的 maximum 方法
    torch.Tensor.maximum,
    
    # 返回 torch 模块的 maximum 函数
    torch.maximum,
    
    # 返回 torch.Tensor 类的 minimum 方法
    torch.Tensor.minimum,
    
    # 返回 torch 模块的 minimum 函数
    torch.minimum,
    
    # 返回 torch.nn.functional 模块的 mish 函数
    torch.nn.functional.mish,
    
    # 返回 torch.Tensor 类的 mvlgamma 方法
    torch.Tensor.mvlgamma,
    
    # 返回 torch 模块的 mvlgamma 函数
    torch.mvlgamma,
    
    # 返回 torch.Tensor 类的 nan_to_num 方法
    torch.Tensor.nan_to_num,
    
    # 返回 torch 模块的 nan_to_num 函数
    torch.nan_to_num,
    
    # 返回 torch.Tensor 类的 ne 方法
    torch.Tensor.ne,
    
    # 返回 torch 模块的 ne 函数
    torch.ne,
    
    # 返回 torch.Tensor 类的 neg 方法
    torch.Tensor.neg,
    
    # 返回 torch 模块的 neg 函数
    torch.neg,
    
    # 返回 torch.Tensor 类的 nextafter 方法
    torch.Tensor.nextafter,
    
    # 返回 torch 模块的 nextafter 函数
    torch.nextafter,
    
    # 返回 torch.Tensor 类的 outer 方法
    torch.Tensor.outer,
    
    # 返回 torch 模块的 outer 函数
    torch.outer,
    
    # 返回 torch 模块的 polar 函数
    torch.polar,
    
    # 返回 torch.Tensor 类的 polygamma 方法
    torch.Tensor.polygamma,
    
    # 返回 torch 模块的 polygamma 函数
    torch.polygamma,
    
    # 返回 torch.Tensor 类的 positive 方法
    torch.Tensor.positive,
    
    # 返回 torch 模块的 positive 函数
    torch.positive,
    
    # 返回 torch.Tensor 类的 pow 方法
    torch.Tensor.pow,
    
    # 返回 torch 模块的 pow 函数
    torch.pow,
    
    # 返回 torch.Tensor 类的 prelu 方法
    torch.Tensor.prelu,
    
    # 返回 torch 模块的 prelu 函数
    torch.prelu,
    
    # 返回 torch.nn.functional 模块的 prelu 函数
    torch.nn.functional.prelu,
    
    # 返回 torch.Tensor 类的 rad2deg 方法
    torch.Tensor.rad2deg,
    
    # 返回 torch 模块的 rad2deg 函数
    torch.rad2deg,
    
    # 返回 torch.Tensor 类的 reciprocal 方法
    torch.Tensor.reciprocal,
    
    # 返回 torch 模块的 reciprocal 函数
    torch.reciprocal,
    
    # 返回 torch.Tensor 类的 relu 方法
    torch.Tensor.relu,
    
    # 返回 torch 模块的 relu 函数
    torch.relu,
    
    # 返回 torch.nn.functional 模块的 relu 函数
    torch.nn.functional.relu,
    
    # 返回 torch.nn.functional 模块的 relu6 函数
    torch.nn.functional.relu6,
    
    # 返回 torch.Tensor 类的 remainder 方法
    torch.Tensor.remainder,
    
    # 返回 torch 模块的 remainder 函数
    torch.remainder,
    
    # 返回 torch.Tensor 类的 round 方法
    torch.Tensor.round,
    
    # 返回 torch 模块的 round 函数
    torch.round,
    
    # 返回 torch.nn.functional 模块的 rrelu 函数
    torch.rrelu,
    
    # 返回 torch.nn.functional 模块的 rrelu 函数
    torch.nn.functional.rrelu,
    
    # 返回 torch.Tensor 类的 rsqrt 方法
    torch.Tensor.rsqrt,
    
    # 返回 torch 模块的 rsqrt 函数
    torch.rsqrt,
    
    # 返回 torch.Tensor 类的 rsub 方法
    torch.Tensor
    # 返回一个张量，其中每个元素表示对应元素的符号位（是否为负数）
    torch.Tensor.signbit,
    
    # 返回一个张量，其中每个元素表示对应元素的符号位（是否为负数）
    torch.signbit,
    
    # 返回一个张量，应用Sigmoid-Linear单元激活函数（SiLU）
    torch.nn.functional.silu,
    
    # 返回一个张量，其中每个元素表示对应元素的正弦值
    torch.Tensor.sin,
    
    # 返回一个张量，其中每个元素表示对应元素的正弦值
    torch.sin,
    
    # 返回一个张量，其中每个元素表示对应元素的sinc函数值
    torch.Tensor.sinc,
    
    # 返回一个张量，其中每个元素表示对应元素的sinc函数值
    torch.sinc,
    
    # 返回一个张量，其中每个元素表示对应元素的双曲正弦值
    torch.Tensor.sinh,
    
    # 返回一个张量，其中每个元素表示对应元素的双曲正弦值
    torch.sinh,
    
    # 返回一个张量，应用Softplus函数
    torch.nn.functional.softplus,
    
    # 返回一个张量，应用Softshrink函数
    torch.nn.functional.softshrink,
    
    # 返回一个张量，其中每个元素表示对应元素的平方根
    torch.Tensor.sqrt,
    
    # 返回一个张量，其中每个元素表示对应元素的平方根
    torch.sqrt,
    
    # 返回一个张量，其中每个元素表示对应元素的平方
    torch.Tensor.square,
    
    # 返回一个张量，其中每个元素表示对应元素的平方
    torch.square,
    
    # 返回一个张量，其中每个元素表示对应元素的差（减法）
    torch.Tensor.sub,
    
    # 返回一个张量，其中每个元素表示对应元素的差（减法）
    torch.sub,
    
    # 返回一个张量，其中每个元素表示对应元素的正切值
    torch.Tensor.tan,
    
    # 返回一个张量，其中每个元素表示对应元素的正切值
    torch.tan,
    
    # 返回一个张量，其中每个元素表示对应元素的双曲正切值
    torch.Tensor.tanh,
    
    # 返回一个张量，其中每个元素表示对应元素的双曲正切值
    torch.tanh,
    
    # 返回一个张量，应用双曲正切函数
    torch.nn.functional.tanh,
    
    # 返回一个张量，其元素小于阈值时设置为0，大于等于阈值时保持不变
    torch.threshold,
    
    # 返回一个张量，应用阈值函数
    torch.nn.functional.threshold,
    
    # 返回一个张量的积分
    torch.trapz,
    
    # 返回一个张量，其中每个元素表示对应元素的真实除法结果
    torch.Tensor.true_divide,
    
    # 返回一个张量，其中每个元素表示对应元素的真实除法结果
    torch.true_divide,
    
    # 返回一个张量，其中每个元素表示对应元素的截断值
    torch.Tensor.trunc,
    
    # 返回一个张量，其中每个元素表示对应元素的截断值
    torch.trunc,
    
    # 返回一个张量，其中每个元素表示对应元素的x * log(y)的值
    torch.Tensor.xlogy,
    
    # 返回一个张量，其中每个元素表示对应元素的x * log(y)的值
    torch.xlogy,
    
    # 返回一个与输入张量形状相同的随机张量，值在[0,1)范围内
    torch.rand_like,
)
```