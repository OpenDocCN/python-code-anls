# `.\pytorch\torch\masked\maskedtensor\binary.py`

```py
# 引入 torch 库，用于张量操作
import torch

# 从当前目录的 core 模块中导入特定函数和类
from .core import (
    _map_mt_args_kwargs,
    _masks_match,
    _tensors_match,
    _wrap_result,
    is_masked_tensor,
)

# __all__ 列表，用于声明在 import * 时可被导入的符号，这里标注为不做类型注解
__all__ = []  # type: ignore[var-annotated]

# 支持二进制操作的函数名列表
BINARY_NAMES = [
    "add",
    "atan2",
    "arctan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "div",
    "divide",
    "floor_divide",
    "fmod",
    "logaddexp",
    "logaddexp2",
    "mul",
    "multiply",
    "nextafter",
    "remainder",
    "sub",
    "subtract",
    "true_divide",
    "eq",
    "ne",
    "le",
    "ge",
    "greater",
    "greater_equal",
    "gt",
    "less_equal",
    "lt",
    "less",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    "not_equal",
]

# 原位二进制操作的函数名列表
INPLACE_BINARY_NAMES = [
    n + "_"
    for n in (
        list(
            set(BINARY_NAMES)
            - {
                "logaddexp",
                "logaddexp2",
                "equal",
                "fmin",
                "minimum",
                "maximum",
                "fmax",
            }
        )
    )
]


# 获取至少一个 MaskedTensor 的掩码
def _get_at_least_one_mask(a, b):
    if not is_masked_tensor(a) and not is_masked_tensor(b):
        raise TypeError("At least one of `a` and `b` must be a MaskedTensor")
    if not _masks_match(a, b):
        raise ValueError("a and b must have matching masks")
    if is_masked_tensor(a):
        return a.get_mask()
    return b.get_mask()


# 二进制操作的辅助函数
def _binary_helper(fn, args, kwargs, inplace):
    # kwargs 的长度必须为 0
    if len(kwargs) != 0:
        raise ValueError("len(kwargs) must equal 0")
    
    # 检查除了左右操作数外的其他张量参数是否为 Tensor 类型，如果是则抛出异常
    for a in args[2:]:
        if torch.is_tensor(a):
            raise TypeError(
                "MaskedTensor binary ops do not support Tensor arguments aside from the lhs and rhs"
            )

    # 检查输入的两个操作数的掩码是否匹配
    if not _masks_match(*args[:2]):
        raise ValueError(
            "Input masks must match. If you need support for this, please open an issue on Github."
        )

    # 将参数和关键字参数中的数据和掩码映射成两个列表
    data_args, data_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.get_data())
    mask_args, mask_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.get_mask())

    # 获取第一个数据参数的布局信息
    args0_layout = data_args[0].layout
    
    # 检查第二个数据参数是否为张量或者 MaskedTensor，并且其布局与第一个数据参数的布局相同
    same_layout = (
        torch.is_tensor(data_args[1]) or is_masked_tensor(data_args[1])
    ) and (args0_layout == data_args[1].layout)
    # 如果第一个参数的布局是稀疏 COO 格式
    if args0_layout == torch.sparse_coo:
        # 如果要求相同布局
        if same_layout:
            # 检查第一个和第二个输入的索引是否匹配
            if not _tensors_match(data_args[0].indices(), data_args[1].indices()):
                raise ValueError(
                    "sparse_coo indices must match. If you need support for this, please open an issue on Github."
                )
            # 检查输入的大小是否相同
            if data_args[0].size() != data_args[1].size():
                raise ValueError(
                    "input1 and input2 must have the same size for binary functions."
                )

            # 将第二个输入的数据转换为值（values）
            data_args[1] = data_args[1].values()

        # 获取第一个输入的索引
        i = data_args[0].indices()
        # 获取第一个输入的大小
        size = data_args[0].size()
        # 将第一个输入的数据转换为值（values）
        data_args[0] = data_args[0].values()
        # 调用指定的函数（fn），并传入所有数据参数
        v = fn(*data_args)
        # 创建稀疏 COO 张量，使用索引 i、值 v 和大小 size
        result_data = torch.sparse_coo_tensor(i, v, size)

    # 如果第一个参数的布局是稀疏 CSR 格式
    elif args0_layout == torch.sparse_csr:
        # 如果要求相同布局
        if same_layout:
            # 检查第一个和第二个输入的行压缩索引和列索引是否匹配
            if not (
                _tensors_match(data_args[0].crow_indices(), data_args[1].crow_indices())
                and _tensors_match(
                    data_args[0].col_indices(), data_args[1].col_indices()
                )
            ):
                raise ValueError(
                    "sparse_csr indices must match. If you need support for this, please open an issue on Github."
                )
            # 将第二个输入的数据转换为值（values）
            data_args[1] = data_args[1].values()

        # 获取第一个输入的行压缩索引和列索引
        crow = data_args[0].crow_indices()
        col = data_args[0].col_indices()
        # 将第一个输入的数据转换为值（values）
        data_args[0] = data_args[0].values()
        # 调用指定的函数（fn），并传入所有数据参数
        v = fn(*data_args)
        # 创建稀疏 CSR 张量，使用行压缩索引 crow、列索引 col 和值 v
        result_data = torch.sparse_csr_tensor(crow, col, v)

    # 如果不是稀疏 COO 或 CSR 格式，直接调用指定的函数（fn），并传入所有数据参数
    else:
        result_data = fn(*data_args)

    # 如果是原地操作
    if inplace:
        # 设置结果数据和掩码，并返回第一个输入的对象
        args[0]._set_data_mask(result_data, mask_args[0])
        return args[0]
    else:
        # 获取至少一个掩码
        result_mask = _get_at_least_one_mask(*args[:2])
        # 对于稀疏张量，仅在布局为 strided 时才能扩展掩码
        if args0_layout == torch.strided:
            result_mask = result_mask.expand_as(result_data)
        # 封装结果数据和掩码，返回结果
        return _wrap_result(result_data, result_mask)
# 根据函数名获取对应的 Torch 操作函数，并创建一个非原地操作的函数
def _torch_binary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)

    # 定义一个非原地操作的函数，调用 _binary_helper 辅助函数进行处理
    def binary_fn(*args, **kwargs):
        return _binary_helper(fn, args, kwargs, inplace=False)

    return binary_fn


# 根据函数名获取对应的 Torch 操作函数，并创建一个原地操作的函数
def _torch_inplace_binary(fn_name):
    fn = getattr(torch.ops.aten, fn_name)

    # 定义一个原地操作的函数，调用 _binary_helper 辅助函数进行处理
    def binary_fn(*args, **kwargs):
        return _binary_helper(fn, args, kwargs, inplace=True)

    return binary_fn


# 创建一个映射字典，将 Torch 操作函数映射到相应的非原地操作函数
NATIVE_BINARY_MAP = {
    getattr(torch.ops.aten, name): _torch_binary(name) for name in BINARY_NAMES
}

# 创建一个映射字典，将 Torch 操作函数映射到相应的原地操作函数
NATIVE_INPLACE_BINARY_MAP = {
    getattr(torch.ops.aten, name): _torch_inplace_binary(name)
    for name in INPLACE_BINARY_NAMES
}

# 获取非原地操作函数的列表
NATIVE_BINARY_FNS = list(NATIVE_BINARY_MAP.keys())

# 获取原地操作函数的列表
NATIVE_INPLACE_BINARY_FNS = list(NATIVE_INPLACE_BINARY_MAP.keys())


# 检查给定函数是否为 Torch 的原生二元操作函数
def _is_native_binary(fn):
    return fn in NATIVE_BINARY_FNS or fn in NATIVE_INPLACE_BINARY_FNS


# 根据函数及参数调用相应的 Torch 原生二元操作函数
def _apply_native_binary(fn, *args, **kwargs):
    # 如果函数在非原地操作函数列表中，则调用相应的非原地操作函数
    if fn in NATIVE_BINARY_FNS:
        return NATIVE_BINARY_MAP[fn](*args, **kwargs)
    # 如果函数在原地操作函数列表中，则调用相应的原地操作函数
    if fn in NATIVE_INPLACE_BINARY_FNS:
        return NATIVE_INPLACE_BINARY_MAP[fn](*args, **kwargs)
    # 如果函数不在以上两个列表中，则返回 NotImplemented
    return NotImplemented
```