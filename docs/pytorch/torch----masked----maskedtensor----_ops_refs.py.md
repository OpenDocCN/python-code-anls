# `.\pytorch\torch\masked\maskedtensor\_ops_refs.py`

```py
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

from functools import partial                   # 导入 partial 函数，用于创建偏函数
from typing import Any, Callable, Dict, TYPE_CHECKING   # 导入类型提示相关模块

import torch                                   # 导入 PyTorch 库
from .binary import _apply_native_binary, NATIVE_BINARY_FNS, NATIVE_INPLACE_BINARY_FNS   # 导入二元操作相关函数
from .core import (
    _get_data,                                # 导入从 MaskedTensor 获取数据的函数
    _masks_match,                             # 导入判断两个 MaskedTensor 是否匹配的函数
    _maybe_get_mask,                          # 导入获取 MaskedTensor 的掩码的函数
    is_masked_tensor,                         # 导入判断是否为 MaskedTensor 的函数
    MaskedTensor,                             # 导入 MaskedTensor 类
)
from .passthrough import _apply_pass_through_fn, PASSTHROUGH_FNS   # 导入传递函数相关函数
from .reductions import (
    _apply_reduction,                         # 导入执行减少操作的函数
    NATIVE_REDUCE_FNS,                        # 导入本地减少函数列表
    TENSOR_REDUCE_FNS,                        # 导入张量减少函数列表
    TORCH_REDUCE_FNS,                         # 导入 Torch 减少函数列表
)
from .unary import _apply_native_unary, NATIVE_INPLACE_UNARY_FNS, NATIVE_UNARY_FNS   # 导入一元操作相关函数


if TYPE_CHECKING:
    from torch._ops import OpOverload         # 导入 OpOverload 类型，仅在类型检查时有效


__all__ = []  # type: ignore[var-annotated]   # 初始化导出列表，用于忽略类型注解


def _check_args_kwargs_length(
    args, kwargs, error_prefix, len_args=None, len_kwargs=None
):
    if len_args is not None and len_args != len(args):   # 检查参数列表长度是否匹配
        raise ValueError(
            f"{error_prefix}: len(args) must be {len_args} but got {len(args)}"
        )
    if len_kwargs is not None and len_kwargs != len(kwargs):   # 检查关键字参数长度是否匹配
        raise ValueError(
            f"{error_prefix}: len(kwargs) must be {len_kwargs} but got {len(kwargs)}"
        )


class _MaskedContiguous(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if not is_masked_tensor(input):             # 检查输入是否为 MaskedTensor
            raise ValueError("MaskedContiguous forward: input must be a MaskedTensor.")

        if input.is_contiguous():                   # 检查输入是否为连续张量
            return input

        data = input.get_data()                    # 获取输入的数据部分
        mask = input.get_mask()                    # 获取输入的掩码部分

        return MaskedTensor(data.contiguous(), mask.contiguous())   # 返回连续的 MaskedTensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output                         # 反向传播，直接返回梯度输出


class _MaskedToDense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        if not is_masked_tensor(input):             # 检查输入是否为 MaskedTensor
            raise ValueError("MaskedToDense forward: input must be a MaskedTensor.")

        if input.layout == torch.strided:           # 检查输入的布局是否为 strided
            return input

        ctx.layout = input.layout                   # 记录输入的布局
        data = input.get_data()                    # 获取输入的数据部分
        mask = input.get_mask()                    # 获取输入的掩码部分

        return MaskedTensor(data.to_dense(), mask.to_dense())   # 返回密集表示的 MaskedTensor

    @staticmethod
    def backward(ctx, grad_output):
        layout = ctx.layout                         # 获取记录的布局信息

        if layout == torch.sparse_coo:              # 根据布局选择合适的反向传播方法
            return grad_output.to_sparse_coo()
        elif layout == torch.sparse_csr:
            return grad_output.to_sparse_csr()
        elif layout == torch.strided:
            return grad_output.to_dense()
        raise ValueError("to_dense: Unsupported input layout: ", layout)   # 报错，不支持的输入布局


class _MaskedToSparse(torch.autograd.Function):
    @staticmethod
    # 定义静态方法 forward，用于执行前向传播
    def forward(ctx, input):
        # 检查输入是否为 MaskedTensor 类型，若不是则抛出数值错误异常
        if not is_masked_tensor(input):
            raise ValueError("MaskedToSparse forward: input must be a MaskedTensor.")

        # 根据稀疏张量的惯例，如果输入的布局已经是稀疏 COO 格式，则直接返回输入
        if input.layout == torch.sparse_coo:
            return input

        # 获取输入数据和掩码
        data = input.get_data()
        mask = input.get_mask()
        
        # 将掩码转换为稀疏 COO 格式，并且进行合并操作
        sparse_mask = mask.to_sparse_coo().coalesce()
        
        # 根据稀疏掩码对数据进行稀疏化处理
        sparse_data = data.sparse_mask(sparse_mask)

        # 返回处理后的 MaskedTensor 对象，包括稀疏化后的数据和掩码
        return MaskedTensor(sparse_data, sparse_mask)

    # 定义静态方法 backward，用于执行反向传播
    @staticmethod
    def backward(ctx, grad_output):
        # 将梯度输出 grad_output 转换为密集张量并返回
        return grad_output.to_dense()
class _MaskedToSparseCsr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 检查输入是否为 MaskedTensor 类型
        if not is_masked_tensor(input):
            raise ValueError("MaskedToSparseCsr forward: input must be a MaskedTensor.")

        # 检查输入数据维度是否为二维
        if input._masked_data.ndim != 2:
            raise ValueError(
                f"Only 2D tensors can be converted to the SparseCsr layout but got shape: {input._masked_data.size()}"
            )

        # 如果输入已经是稀疏 CSR 格式，则直接返回
        if input.layout == torch.sparse_csr:
            return input

        # 获取输入数据和掩码
        data = input.get_data()
        mask = input.get_mask()
        # 将掩码转换为稀疏 CSR 格式
        sparse_mask = mask.to_sparse_csr()
        # 根据稀疏掩码对数据进行稀疏化处理
        sparse_data = data.sparse_mask(sparse_mask)

        # 返回一个新的 MaskedTensor 对象，其数据为稀疏化后的数据，掩码为稀疏 CSR 格式的掩码
        return MaskedTensor(sparse_data, sparse_mask)

    @staticmethod
    def backward(ctx, grad_output):
        # 对梯度输出进行转换为密集张量
        return grad_output.to_dense()


class _MaskedWhere(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cond, self, other):
        # 在前向传播中，标记条件张量为不可微分
        ctx.mark_non_differentiable(cond)
        # 保存条件张量以备后用
        ctx.save_for_backward(cond)
        # 调用 torch.ops.aten.where 函数执行 where 操作
        return torch.ops.aten.where(cond, self, other)

    @staticmethod
    def backward(ctx, grad_output):
        # 在反向传播中，获取保存的条件张量
        (cond,) = ctx.saved_tensors

        # 定义一个函数，返回与输入张量形状相同的 MaskedTensor 对象
        def masked_out_like(mt):
            return MaskedTensor(mt.get_data(), torch.zeros_like(mt.get_mask()).bool())

        # 返回 where 操作的梯度计算结果，分别对应不同的条件
        return (
            None,  # cond 参数没有梯度
            torch.ops.aten.where(cond, grad_output, masked_out_like(grad_output)),  # 对 self 的梯度
            torch.ops.aten.where(cond, masked_out_like(grad_output), grad_output),  # 对 other 的梯度
        )


_MASKEDTENSOR_FUNCTION_TABLE = {}

_function_fn_apply_map = {
    (
        tuple(NATIVE_REDUCE_FNS),
        tuple(TORCH_REDUCE_FNS),
        tuple(TENSOR_REDUCE_FNS),
    ): _apply_reduction,
}

# 构建函数映射表 _MASKEDTENSOR_FUNCTION_TABLE
for fn_map_list, apply_fn in _function_fn_apply_map.items():
    for fn_map in fn_map_list:
        for fn in fn_map:
            # 使用偏函数 partial 绑定 apply_fn 到函数 fn，并将其注册到函数表中
            _MASKEDTENSOR_FUNCTION_TABLE[fn] = partial(apply_fn, fn)


def register_function_func(ops):
    """
    Used for registering a new __torch_function__ function to MaskedTensor
    Called via _MASKEDTENSOR_FUNCTION_TABLE[func](*args, **kwargs)

    The code to register a new function looks like:

    @register_function_func(list_of_ops)
    def foo(func, *args, **kwargs):
        <implementation>
    """

    def wrapper(func):
        for op in ops:
            # 使用偏函数 partial 绑定 func 到操作 op，并将其注册到函数表中
            _MASKEDTENSOR_FUNCTION_TABLE[op] = partial(func, op)

    return wrapper


@register_function_func(NATIVE_REDUCE_FNS + TORCH_REDUCE_FNS + TENSOR_REDUCE_FNS)
def _general_function_reductions(func, *args, **kwargs):
    # 执行通用函数减少操作
    return _apply_reduction(func, *args, **kwargs)


@register_function_func([torch.Tensor.where, torch.where])
def _function_where(func, *args, **kwargs):
    # 检查参数的长度
    _check_args_kwargs_length(
        args, kwargs, "__torch_function__, torch.where", len_args=3, len_kwargs=0
    )
    # 调用 MaskedWhere 的 forward 方法进行 where 操作
    return _MaskedWhere.apply(*args)


@register_function_func([torch.Tensor.contiguous])
def _function_contiguous(func, *args, **kwargs):
    # 调用 MaskedContiguous 的 apply 方法进行连续化操作
    return _MaskedContiguous.apply(args[0])


@register_function_func([torch.Tensor.to_dense])
# 注册一个新的 dispatch 函数到 _MASKEDTENSOR_DISPATCH_TABLE 中，以便 MaskedTensor 使用
def register_dispatch_func(aten_ops):
    """
    用于将新的 __torch_dispatch__ 函数注册到 MaskedTensor 中
    被调用方式为 _MASKEDTENSOR_DISPATCH_TABLE[func](*args, **kwargs)

    注册新函数的代码示例：

    @register_dispatch_func(list_of_ops)
    def foo(func, *args, **kwargs):
        <实现内容>
    """

    def wrapper(func):
        # 将给定的 aten_ops 中的每个操作注册到 _MASKEDTENSOR_DISPATCH_TABLE 中
        for aten_op in aten_ops:
            _MASKEDTENSOR_DISPATCH_TABLE[aten_op] = partial(func, aten_op)

    return wrapper


# 注册一个通用的 reduction 函数到 _MASKEDTENSOR_DISPATCH_TABLE 中
@register_dispatch_func(NATIVE_REDUCE_FNS + TORCH_REDUCE_FNS + TENSOR_REDUCE_FNS)
def _general_reduction(func, *args, **kwargs):
    return _apply_reduction(func, *args, **kwargs)


# 注册一个通用的 passthrough 函数到 _MASKEDTENSOR_DISPATCH_TABLE 中
@register_dispatch_func(PASSTHROUGH_FNS)
def _general_passthrough(func, *args, **kwargs):
    return _apply_pass_through_fn(func, *args, **kwargs)


# 注册一个通用的 unary 函数到 _MASKEDTENSOR_DISPATCH_TABLE 中
@register_dispatch_func(NATIVE_UNARY_FNS + NATIVE_INPLACE_UNARY_FNS)
def _general_unary(func, *args, **kwargs):
    return _apply_native_unary(func, *args, **kwargs)


# 注册一个通用的 binary 函数到 _MASKEDTENSOR_DISPATCH_TABLE 中
@register_dispatch_func(NATIVE_BINARY_FNS + NATIVE_INPLACE_BINARY_FNS)
def _general_binary(func, *args, **kwargs):
    return _apply_native_binary(func, *args, **kwargs)


# 注册一个 stride 函数到 _MASKEDTENSOR_DISPATCH_TABLE 中
@register_dispatch_func([torch.ops.aten.stride])
def stride(func, *args, **kwargs):
    return None


# 注册一个 sym_stride 函数到 _MASKEDTENSOR_DISPATCH_TABLE 中
@register_dispatch_func([torch.ops.aten.sym_stride])
def sym_stride(func, *args, **kwargs):
    return None


# 注册一个 layout 函数到 _MASKEDTENSOR_DISPATCH_TABLE 中
@register_dispatch_func([torch.ops.prim.layout])
def layout(func, *args, **kwargs):
    return _get_data(args[0]).layout


# 注册一个 is_contiguous 函数到 _MASKEDTENSOR_DISPATCH_TABLE 中
@register_dispatch_func([torch.ops.aten.is_contiguous])
def is_contiguous(func, *args, **kwargs):
    data = _get_data(args[0])
    # 如果数据是稀疏的，则抛出异常
    if data.is_sparse:
        raise ValueError("MaskedTensors with sparse data do not have is_contiguous")
    # 否则调用原始的 is_contiguous 函数
    return func(data, *args[1:], **kwargs)


# 注册一个 is_strides_like_format 函数到 _MASKEDTENSOR_DISPATCH_TABLE 中
@register_dispatch_func([torch.ops.aten.is_strides_like_format])
def is_strides_like_format(func, *args, **kwargs):
    data = _get_data(args[0])
    # 如果数据是稀疏的，则抛出异常
    if data.is_sparse:
        raise ValueError(
            "MaskedTensors with sparse data do not have is_strides_like_format"
        )
    # 否则调用原始的 is_strides_like_format 函数
    return func(data, *args[1:], **kwargs)


# 注册一个 is_non_overlapping_and_dense 函数到 _MASKEDTENSOR_DISPATCH_TABLE 中
@register_dispatch_func([torch.ops.aten.is_non_overlapping_and_dense])
def is_non_overlapping_and_dense(func, *args, **kwargs):
    data = _get_data(args[0])
    # 如果数据是稀疏的，则抛出异常
    if data.is_sparse:
        raise ValueError(
            "MaskedTensors with sparse data do not have is_non_overlapping_and_dense"
        )
    # 否则调用原始的 is_non_overlapping_and_dense 函数
    return func(data, *args[1:], **kwargs)


# 注册一个 contiguous 函数到 _MASKEDTENSOR_DISPATCH_TABLE 中
@register_dispatch_func([torch.ops.aten.contiguous])
# 注册一个函数为分派函数，处理需要连续存储的操作
def contiguous(func, *args, **kwargs):
    # 检查第一个参数是否为稀疏数据，如果是则引发错误，因为稀疏数据的 MaskedTensors 不能保证连续存储
    if _get_data(args[0]).is_sparse:
        raise ValueError("MaskedTensors with sparse data do not have contiguous")
    # 调用自定义的 _MaskedContiguous.apply 方法来确保数据连续性
    return _MaskedContiguous.apply(args[0])


# 注册一个分派函数，处理 torch.ops.aten.new_empty_strided 操作
@register_dispatch_func([torch.ops.aten.new_empty_strided])
def new_empty_strided(func, *args, **kwargs):
    # 检查参数和关键字参数的长度，确保匹配
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=3)
    # 获取 args[0] 的数据部分
    data = _get_data(args[0])
    # 获取可能存在的 mask
    mask = _maybe_get_mask(args[0])
    # 检查 args[1] 是否与 data.size() 的形状相同
    if tuple(args[1]) != tuple(data.size()):
        raise ValueError(
            f"__torch_dispatch__, {func}: args[1] expected to be the same as data.size()"
        )
    # 检查 args[2] 是否与 data.stride() 的形状相同
    if tuple(args[2]) != tuple(data.stride()):
        raise ValueError(
            f"__torch_dispatch__, {func}: args[2] expected to be the same as data.stride()"
        )
    # 调用原始的 func 方法，生成新的 MaskedTensor 对象并返回
    return MaskedTensor(func(data, args[1], args[2], **kwargs), mask)


# 注册一个分派函数，处理 torch.ops.aten._local_scalar_dense 操作
@register_dispatch_func([torch.ops.aten._local_scalar_dense])
def _local_scalar_dense(func, *args, **kwargs):
    # 如果 args[0] 没有 mask，则引发错误，因为预期是一个带有 mask 的张量
    if not _maybe_get_mask(args[0]):
        raise ValueError(f"__torch_dispatch__, {func}: expected a mask tensor")
    # 调用 torch.ops.aten._local_scalar_dense 处理 args[0] 的数据部分
    return torch.ops.aten._local_scalar_dense(_get_data(args[0]))


# 注册一个分派函数，处理 torch.ops.aten.detach 和 torch.ops.aten.clone 操作
@register_dispatch_func([torch.ops.aten.detach, torch.ops.aten.clone])
def _apply_fn_on_data(func, *args, **kwargs):
    # 对 args[0] 的数据部分调用 func 函数，并生成一个新的 MaskedTensor 对象返回
    return MaskedTensor(func(_get_data(args[0])), _maybe_get_mask(args[0]))


# 注册一个分派函数，处理 torch.ops.aten._to_copy 操作
@register_dispatch_func([torch.ops.aten._to_copy])
def _to_copy(func, *args, **kwargs):
    # 调用 func 方法处理 args[0] 的数据部分，并生成一个新的 MaskedTensor 对象返回
    new_data = func(_get_data(args[0]), *args[1:], **kwargs)
    return MaskedTensor(new_data, _maybe_get_mask(args[0]))


# 注册一个分派函数，处理 torch.ops.aten._softmax 操作
@register_dispatch_func([torch.ops.aten._softmax])
def _softmax(func, *args, **kwargs):
    # 检查参数和关键字参数的长度，确保匹配
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=3, len_kwargs=0
    )
    # 获取 args[0] 的数据部分和可能存在的 mask
    data = _get_data(args[0])
    mask = _maybe_get_mask(args[0])
    # 调用 torch.ops.aten._masked_softmax 处理 data，并传入反转的 mask，args[1]，以及维度参数 2
    result_data = torch.ops.aten._masked_softmax(data, ~mask, args[1], 2)
    # 返回生成的 MaskedTensor 对象
    return MaskedTensor(result_data, mask)


# 注册一个分派函数，处理 torch.ops.aten.ones_like 操作
@register_dispatch_func([torch.ops.aten.ones_like])
def ones_like(func, *args, **kwargs):
    # 检查参数和关键字参数的长度，确保匹配
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=1)
    # 调用 func 方法处理 args[0] 的数据部分，并生成一个新的 MaskedTensor 对象返回
    result_data = func(_get_data(args[0]), **kwargs)
    return MaskedTensor(result_data, _maybe_get_mask(args[0]))


# 注册一个分派函数，处理 torch.ops.aten._softmax_backward_data 操作
@register_dispatch_func([torch.ops.aten._softmax_backward_data])
def _softmax_backward_data(func, *args, **kwargs):
    # 检查参数和关键字参数的长度，确保匹配
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=4)
    # 直接返回 func 方法处理后的结果，作为返回值
    grad, output, dim, input_dtype = args
    # 检查输入的梯度和输出是否都是MaskedTensor类型，并且它们的掩码匹配
    if is_masked_tensor(grad) and is_masked_tensor(output):
        # 如果梯度和输出的掩码不匹配，抛出数值错误异常
        if not _masks_match(grad, output):
            raise ValueError(
                "__torch_dispatch__, {func}: expected the masks of grad and output to match"
            )
        # 获取梯度数据
        grad_data = _get_data(grad)
        # 调用底层的PyTorch操作执行带掩码的softmax的反向传播
        new_grad_data = torch.ops.aten._masked_softmax_backward(
            grad_data,
            _get_data(output),
            ~_maybe_get_mask(grad),  # 获取梯度的反掩码
            dim % grad_data.ndim,    # 计算梯度数据的维度
        )
        # 创建新的MaskedTensor对象，其中包含更新后的梯度数据和原始的掩码
        res = MaskedTensor(new_grad_data, _maybe_get_mask(grad))
        # 返回结果MaskedTensor对象
        return res
    else:
        # 如果输入的梯度和输出不是MaskedTensor类型，则抛出数值错误异常
        raise ValueError(
            f"__torch_dispatch__, {func}: grad and output must both be MaskedTensors"
        )
@register_dispatch_func([torch.ops.aten.copy_])
def copy_(func, *args, **kwargs):
    # 检查参数和关键字参数的长度，确保与函数签名匹配
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=2)
    # 检查第一个和第二个参数的掩码是否匹配，如果不匹配则抛出数值错误
    if not _masks_match(_maybe_get_mask(args[0]), _maybe_get_mask(args[1])):
        raise ValueError("args[0] mask and args[1] mask must match but do not")
    # 获取第一个和第二个参数的数据，并调用给定的函数 func 进行处理
    func(_get_data(args[0]), _get_data(args[1]))
    # 返回第一个参数作为结果
    return args[0]


@register_dispatch_func([torch.ops.aten.where])
def where(func, *args, **kwargs):
    # 检查参数和关键字参数的长度，确保与函数签名匹配
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=3, len_kwargs=0
    )
    # 检查第一个参数是否为张量，如果不是则抛出数值错误
    if not torch.is_tensor(args[0]):
        raise ValueError("__torch_dispatch__, {func}: expected args[0] to be a tensor")
    # 将第二个和第三个参数分别赋值给 mx 和 my
    mx = args[1]
    my = args[2]
    # 如果 mx 不是 MaskedTensor，则创建一个默认掩码为全 True 的 MaskedTensor 对象
    if not is_masked_tensor(mx):
        mx = MaskedTensor(mx, torch.ones_like(mx, dtype=torch.bool))
    # 如果 my 不是 MaskedTensor，则创建一个默认掩码为全 True 的 MaskedTensor 对象
    if not is_masked_tensor(my):
        my = MaskedTensor(my, torch.ones_like(my, dtype=torch.bool))
    # 使用给定的函数 func 处理 args[0]、mx 的数据以及 my 的数据，并返回新数据
    new_data = func(args[0], mx.get_data(), my.get_data())
    # 使用给定的函数 func 处理 args[0]、mx 的掩码以及 my 的掩码，并返回新掩码
    new_mask = func(args[0], mx.get_mask(), my.get_mask())
    # 返回一个新的 MaskedTensor 对象，包含新的数据和新的掩码
    return MaskedTensor(new_data, new_mask)


@register_dispatch_func([torch.ops.aten._to_sparse])
def _to_sparse(func, *args, **kwargs):
    # 检查参数和关键字参数的长度，确保与函数签名匹配
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0
    )
    # 检查第一个参数是否为张量，如果不是则抛出类型错误
    if not torch.is_tensor(args[0]):
        raise TypeError("__torch_dispatch__, {func}: expected args[0] to be a tensor")
    # 将第一个参数赋值给 mt
    mt = args[0]
    # 如果 mt 不是 MaskedTensor，则创建一个默认掩码为全 True 的 MaskedTensor 对象
    if not is_masked_tensor(mt):
        mt = MaskedTensor(mt, torch.ones_like(mt, dtype=torch.bool))
    # 如果 mt 已经是稀疏 COO 格式，则直接返回
    if mt.is_sparse_coo():
        return mt
    # 使用给定的函数 func 处理 mt 的掩码，并对其进行合并操作
    new_mask = func(_maybe_get_mask(args[0])).coalesce()
    # 使用原始数据 mt 的稀疏掩码 new_mask，生成新的稀疏数据 new_data
    new_data = _get_data(args[0]).sparse_mask(new_mask)
    # 返回一个新的 MaskedTensor 对象，包含新的数据和新的掩码
    return MaskedTensor(new_data, new_mask)


@register_dispatch_func([torch.ops.aten._to_sparse_csr])
def _to_sparse_csr(func, *args, **kwargs):
    # 检查参数和关键字参数的长度，确保与函数签名匹配
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0
    )
    # 检查第一个参数是否为张量，如果不是则抛出数值错误
    if not torch.is_tensor(args[0]):
        raise ValueError("__torch_dispatch__, {func}: expected args[0] to be a tensor")
    # 将第一个参数赋值给 mt
    mt = args[0]
    # 如果 mt 不是 MaskedTensor，则创建一个默认掩码为全 True 的 MaskedTensor 对象
    if not is_masked_tensor(mt):
        mt = MaskedTensor(mt, torch.ones_like(mt).bool())
    # 如果 mt 已经是稀疏 CSR 格式，则直接返回
    if mt.is_sparse_csr():
        return mt
    # 使用给定的函数 func 处理 mt 的掩码，并生成新的稀疏掩码 new_mask
    new_mask = func(_maybe_get_mask(args[0]))
    # 使用原始数据 mt 的稀疏掩码 new_mask，生成新的稀疏数据 new_data
    new_data = _get_data(args[0]).sparse_mask(new_mask)
    # 返回一个新的 MaskedTensor 对象，包含新的数据和新的掩码
    return MaskedTensor(new_data, new_mask)


@register_dispatch_func([torch.ops.aten._to_dense])
def _to_dense(func, *args, **kwargs):
    # 检查参数和关键字参数的长度，确保与函数签名匹配
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0
    )
    # 检查第一个参数是否为张量，如果不是则抛出数值错误
    if not torch.is_tensor(args[0]):
        raise ValueError("__torch_dispatch__, {func}: expected args[0] to be a tensor")
    # 将第一个参数赋值给 mt
    mt = args[0]
    # 如果 mt 不是 MaskedTensor，则创建一个默认掩码为全 True 的 MaskedTensor 对象
    if not is_masked_tensor(mt):
        mt = MaskedTensor(mt, torch.ones_like(mt).bool())
    # 使用给定的函数 func 分别处理 mt 的数据和掩码，生成新的数据和掩码
    new_data = func(_get_data(args[0]))
    new_mask = func(_maybe_get_mask(args[0]))
    # 返回一个新的 MaskedTensor 对象，包含新的数据和新的掩码
    return MaskedTensor(new_data, new_mask)
# 注册一个函数来处理 torch.ops.aten._indices 的调度，使其返回一个 MaskedTensor 对象
@register_dispatch_func([torch.ops.aten._indices])
def _indices(func, *args, **kwargs):
    # 检查参数和关键字参数的长度是否符合预期
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0
    )
    # 获取第一个参数的数据，并获取其稀疏张量的索引
    data = _get_data(args[0]).indices()
    # 用全是 True 的张量创建一个 MaskedTensor 对象
    return MaskedTensor(data, torch.ones_like(data).bool())


# 注册一个函数来处理 torch.ops.aten._values 的调度，使其返回一个 MaskedTensor 对象
@register_dispatch_func([torch.ops.aten._values])
def _values(func, *args, **kwargs):
    # 检查参数和关键字参数的长度是否符合预期
    _check_args_kwargs_length(
        args, kwargs, f"__torch_dispatch__, {func}", len_args=1, len_kwargs=0
    )
    # 获取第一个参数的数据，并获取其稀疏张量的值
    data = _get_data(args[0]).values()
    # 用全是 True 的张量创建一个 MaskedTensor 对象
    return MaskedTensor(data, torch.ones_like(data).bool())


# 注册一个函数来处理 torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors 的调度，使其返回一个 MaskedTensor 对象
@register_dispatch_func([torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors])
def _sparse_coo_tensor_with_dims_and_tensors(func, *args, **kwargs):
    # 将参数列表转换为可变列表
    new_args = list(args)
    # 如果倒数第一个参数是 MaskedTensor，则获取其数据部分
    if is_masked_tensor(args[-1]):
        new_args[-1] = args[-1].get_data()
    # 如果倒数第二个参数是 MaskedTensor，则获取其数据部分
    if is_masked_tensor(args[-2]):
        new_args[-2] = args[-2].get_data()

    # 调用原始函数并获取新的数据
    new_data = func(*new_args, **kwargs)
    # 用全是 True 的张量创建一个 MaskedTensor 对象作为数据
    new_args[-1] = torch.ones_like(new_args[-1])
    # 再次调用原始函数获取新的掩码数据，并将其转换为布尔类型
    new_mask = func(*new_args, **kwargs).bool()

    # 返回用新数据和新掩码创建的 MaskedTensor 对象
    return MaskedTensor(new_data, new_mask)


# 注册一个函数来处理 torch.ops.aten.is_same_size 的调度，比较两个参数的数据部分是否尺寸相同
@register_dispatch_func([torch.ops.aten.is_same_size])
def is_same_size(func, *args, **kwargs):
    # 检查参数和关键字参数的长度是否符合预期
    _check_args_kwargs_length(args, kwargs, f"__torch_dispatch__, {func}", len_args=2)
    # 比较第一个参数的数据部分和第二个参数的数据部分是否尺寸相同
    return _get_data(args[0]).is_same_size(_get_data(args[1]))
```