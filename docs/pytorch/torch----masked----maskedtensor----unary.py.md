# `.\pytorch\torch\masked\maskedtensor\unary.py`

```
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

# 导入 torch 模块
import torch

# 从当前目录下的 core 模块导入 _map_mt_args_kwargs 和 _wrap_result 函数
from .core import _map_mt_args_kwargs, _wrap_result

# 声明 __all__ 列表，类型标注忽略
__all__ = []  # type: ignore[var-annotated]

# 定义 UNARY_NAMES 列表，包含各种一元操作函数的名称
UNARY_NAMES = [
    "abs",
    "absolute",
    "acos",
    "arccos",
    "acosh",
    "arccosh",
    "angle",
    "asin",
    "arcsin",
    "asinh",
    "arcsinh",
    "atan",
    "arctan",
    "atanh",
    "arctanh",
    "bitwise_not",
    "ceil",
    "clamp",
    "clip",
    "conj_physical",
    "cos",
    "cosh",
    "deg2rad",
    "digamma",
    "erf",
    "erfc",
    "erfinv",
    "exp",
    "exp2",
    "expm1",
    "fix",
    "floor",
    "frac",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "logit",
    "i0",
    "isnan",
    "nan_to_num",
    "neg",
    "negative",
    "positive",
    "pow",
    "rad2deg",
    "reciprocal",
    "round",
    "rsqrt",
    "sigmoid",
    "sign",
    "sgn",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "trunc",
]

# 根据 UNARY_NAMES 创建 INPLACE_UNARY_NAMES 列表，包含各种一元操作函数名称后接下划线的形式
INPLACE_UNARY_NAMES = [
    n + "_"
    for n in (list(set(UNARY_NAMES) - {"angle", "positive", "signbit", "isnan"}))
]

# 显示声明当前不支持的一元操作函数列表
UNARY_NAMES_UNSUPPORTED = [
    "atan2",
    "arctan2",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "copysign",
    "float_power",
    "fmod",
    "frexp",
    "gradient",
    "imag",
    "ldexp",
    "lerp",
    "logical_not",
    "hypot",
    "igamma",
    "igammac",
    "mvlgamma",
    "nextafter",
    "polygamma",
    "real",
    "remainder",
    "true_divide",
    "xlogy",
]

# 定义内部辅助函数 _unary_helper，用于处理一元操作
def _unary_helper(fn, args, kwargs, inplace):
    # 如果 kwargs 的长度不为 0，则抛出 ValueError 异常
    if len(kwargs) != 0:
        raise ValueError(
            "MaskedTensor unary ops require that len(kwargs) == 0. "
            "If you need support for this, please open an issue on Github."
        )
    
    # 检查除第一个参数外的所有参数是否为 Tensor 类型，如果是则抛出 TypeError 异常
    for a in args[1:]:
        if torch.is_tensor(a):
            raise TypeError(
                "MaskedTensor unary ops do not support additional Tensor arguments"
            )

    # 根据 args 和 kwargs 映射处理 mask 和 data，分别存储在 mask_args 和 mask_kwargs，data_args 和 data_kwargs 中
    mask_args, mask_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x._masked_mask)
    data_args, data_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x._masked_data)

    # 根据输入参数的布局类型执行不同的操作
    if args[0].layout == torch.sparse_coo:
        # 如果布局是稀疏的 COO 格式，则对数据进行处理
        data_args[0] = data_args[0].coalesce()  # 合并稀疏数据
        s = data_args[0].size()  # 获取数据的尺寸
        i = data_args[0].indices()  # 获取数据的索引
        data_args[0] = data_args[0].coalesce().values()  # 获取数据的值
        v = fn(*data_args)  # 对数据进行一元操作
        result_data = torch.sparse_coo_tensor(i, v, size=s)  # 生成稀疏 COO 张量

    elif args[0].layout == torch.sparse_csr:
        # 如果布局是稀疏的 CSR 格式，则对数据进行处理
        crow = data_args[0].crow_indices()  # 获取行索引
        col = data_args[0].col_indices()  # 获取列索引
        data_args[0] = data_args[0].values()  # 获取数据的值
        v = fn(*data_args)  # 对数据进行一元操作
        result_data = torch.sparse_csr_tensor(crow, col, v)  # 生成稀疏 CSR 张量

    else:
        # 如果不是稀疏布局，则直接对数据进行一元操作
        result_data = fn(*data_args)
    # 如果 inplace 参数为 True，则在第一个参数对象上设置数据掩码并返回该对象自身
    if inplace:
        args[0]._set_data_mask(result_data, mask_args[0])
        return args[0]
    # 如果 inplace 参数为 False，则将结果数据和掩码作为参数传递给 _wrap_result 函数并返回其结果
    else:
        return _wrap_result(result_data, mask_args[0])
# 根据给定的函数名，获取 torch.ops.aten 模块中对应的函数对象
def _torch_unary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)

    # 定义一个闭包函数 unary_fn，用于执行单目操作，不支持原地操作
    def unary_fn(*args, **kwargs):
        return _unary_helper(fn, args, kwargs, inplace=False)

    return unary_fn


# 根据给定的函数名，获取 torch.ops.aten 模块中对应的函数对象
def _torch_inplace_unary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)

    # 定义一个闭包函数 unary_fn，用于执行单目操作，支持原地操作
    def unary_fn(*args, **kwargs):
        return _unary_helper(fn, args, kwargs, inplace=True)

    return unary_fn


# 根据 UNARY_NAMES 列表中的函数名创建一个映射，将函数对象映射到 _torch_unary 包装后的函数
NATIVE_UNARY_MAP = {
    getattr(torch.ops.aten, name): _torch_unary(name) for name in UNARY_NAMES
}

# 根据 INPLACE_UNARY_NAMES 列表中的函数名创建一个映射，将函数对象映射到 _torch_inplace_unary 包装后的函数
NATIVE_INPLACE_UNARY_MAP = {
    getattr(torch.ops.aten, name): _torch_inplace_unary(name)
    for name in INPLACE_UNARY_NAMES
}

# 获取 NATIVE_UNARY_MAP 中所有的函数对象，并将其组成列表
NATIVE_UNARY_FNS = list(NATIVE_UNARY_MAP.keys())

# 获取 NATIVE_INPLACE_UNARY_MAP 中所有的函数对象，并将其组成列表
NATIVE_INPLACE_UNARY_FNS = list(NATIVE_INPLACE_UNARY_MAP.keys())


# 判断给定的函数对象 fn 是否属于 NATIVE_UNARY_FNS 或 NATIVE_INPLACE_UNARY_FNS 中的一员
def _is_native_unary(fn):
    return fn in NATIVE_UNARY_FNS or fn in NATIVE_INPLACE_UNARY_FNS


# 根据给定的函数对象 fn，调用其对应的映射函数进行单目操作
def _apply_native_unary(fn, *args, **kwargs):
    if fn in NATIVE_UNARY_FNS:
        return NATIVE_UNARY_MAP[fn](*args, **kwargs)
    if fn in NATIVE_INPLACE_UNARY_FNS:
        return NATIVE_INPLACE_UNARY_MAP[fn](*args, **kwargs)
    return NotImplemented
```