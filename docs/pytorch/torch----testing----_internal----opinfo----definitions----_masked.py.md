# `.\pytorch\torch\testing\_internal\opinfo\definitions\_masked.py`

```
# 忽略 mypy 的错误提示

# 导入所需的库
import unittest
from collections.abc import Sequence
from functools import partial
from typing import List

import numpy as np

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import tol, toleranceOverride
from torch.testing._internal.common_dtype import (
    all_types_and,
    all_types_and_complex_and,
    complex_types,
    floating_and_complex_types_and,
    floating_types_and,
    integral_types,
)
from torch.testing._internal.opinfo.core import (
    DecorateInfo,
    gradcheck_wrapper_masked_operation,
    gradcheck_wrapper_masked_pointwise_operation,
    M,
    OpInfo,
    ReductionOpInfo,
    S,
    sample_inputs_reduction,
    SampleInput,
)
from torch.testing._internal.opinfo.utils import prod_numpy, reference_reduction_numpy

# 用于 log_softmax、softmax、softmin 的输入样本生成函数
def sample_inputs_softmax_variant(
    op_info,
    device,
    dtype,
    requires_grad,
    with_dtype=False,
    use_zero_dimensions=True,
    **kwargs,
):
    # 创建张量的部分函数
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    # 不同的输入形状和维度
    cases = [
        ((S,), (0,)),
        ((S, S), (0,)),
        ((S, S), (1,)),
        ((S, S), (-1,)),
        ((S, M, S), (2,)),
        *([((S, 0, 0), (-1,))] if use_zero_dimensions else []),
    ]
    # 如果需要指定数据类型，则设置为 torch.float64
    kwargs = dict(dtype=torch.float64) if with_dtype else None

    # 当设备不是 XLA 时，添加一个空维度的测试用例
    if torch.device(device).type != "xla":
        cases.append(((), (0,)))

    # 返回生成的输入样本
    return (
        SampleInput(make_arg(shape), args=dim, kwargs=kwargs) for shape, dim in cases
    )

# 生成带有掩码的操作的掩码
def _generate_masked_op_mask(input_shape, device, **kwargs):
    # 创建布尔类型张量的部分函数
    make_arg = partial(
        make_tensor, dtype=torch.bool, device=device, requires_grad=False
    )
    # 生成不同的掩码情况
    yield None
    yield make_arg(input_shape)
    if len(input_shape) > 2:
        # 广播最后一个掩码维度
        yield make_arg(input_shape[:-1] + (1,))
        # 广播中间的掩码维度
        yield make_arg(input_shape[:1] + (1,) + input_shape[2:])
        # 广播第一个掩码维度
        yield make_arg((1,) + input_shape[1:])
        # 掩码维度小于输入维度
        yield make_arg(input_shape[1:])
        # 掩码维度为 1
        yield make_arg(input_shape[-1:])
        # 不支持需要广播输入的掩码（掩码维度大于输入维度），但如果有需求，可以重新考虑这种情况

# 为带有掩码的归约操作生成输入样本
def sample_inputs_masked_reduction(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for masked reduction operators.

    Masked reduction operator is a reduction operator with trailing
    mask optional argument. A mask is a bool tensor with the same
    shape as input or a shape that is broadcastable to input shape.
    """
    # 将 op_info.supports_multiple_dims 的值存入 kwargs 字典中的 supports_multiple_dims 键
    kwargs["supports_multiple_dims"] = op_info.supports_multiple_dims

    # 使用 sample_inputs_reduction 函数生成的样本输入进行迭代
    for sample_input in sample_inputs_reduction(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        # 使用 _generate_masked_op_mask 函数生成针对当前输入形状的掩码进行迭代
        for mask in _generate_masked_op_mask(
            sample_input.input.shape, device, **kwargs
        ):
            # 准备 SampleInput 的参数和关键字参数
            sample_input_args, sample_input_kwargs = sample_input.args, dict(
                mask=mask, **sample_input.kwargs
            )
            # 生成一个新的 SampleInput 对象，并通过 yield 返回
            yield SampleInput(
                sample_input.input.detach().requires_grad_(requires_grad),
                args=sample_input_args,
                kwargs=sample_input_kwargs,
            )
            # 如果满足条件，则对输入的特定位置赋予特定的值
            if (
                not requires_grad
                and dtype.is_floating_point
                and sample_input.input.ndim == 2
                and mask is not None
                and mask.shape == sample_input.input.shape
            ):
                # 对输入的对角线位置填充无穷大、负无穷大和 NaN 值
                for v in [torch.inf, -torch.inf, torch.nan]:
                    t = sample_input.input.detach()
                    t.diagonal(0, -2, -1).fill_(v)
                    # 生成一个新的 SampleInput 对象，并通过 yield 返回
                    yield SampleInput(
                        t.requires_grad_(requires_grad),
                        args=sample_input_args,
                        kwargs=sample_input_kwargs,
                    )
def sample_inputs_sparse_coo_masked_reduction(
    op_info, device, dtype, requires_grad, **kwargs
):
    """Sample inputs for masked reduction operators that support inputs
    with sparse coo layouts.
    """
    # 如果操作信息支持稀疏格式
    if op_info.supports_sparse:
        # 取操作名并去掉"masked."前缀
        op_name = op_info.name.replace("masked.", "")
        # 遍历通过masked reduction函数生成的样本输入
        for sample_input in sample_inputs_masked_reduction(
            op_info, device, dtype, requires_grad, **kwargs
        ):
            # 获取样本输入中的掩码（mask）
            mask = sample_input.kwargs.get("mask")
            # 如果掩码存在
            if mask is not None:
                # 复制样本输入的关键字参数，并更新为稀疏格式的掩码
                sample_input_kwargs = sample_input.kwargs.copy()
                sample_input_kwargs.update(mask=mask.to_sparse())
                # 生成稀疏格式输入的样本
                yield SampleInput(
                    sample_input.input.to_sparse(),
                    args=sample_input.args,
                    kwargs=sample_input_kwargs,
                )
            else:
                # 如果掩码不存在且操作名属于{"prod", "amax", "amin"}
                if op_name in {"prod", "amax", "amin"}:
                    # FIXME: 目前不支持带有非零减少标识并且未指定掩码的稀疏COO张量的减少操作，
                    # 参见torch.masked.prod的实现细节。
                    # 继续下一个循环
                    continue
                # 生成稀疏格式输入的样本
                yield SampleInput(
                    sample_input.input.to_sparse(),
                    args=sample_input.args,
                    kwargs=sample_input.kwargs,
                )


def sample_inputs_sparse_csr_masked_reduction(
    op_info, device, dtype, requires_grad, **kwargs
):
    """Sample inputs for masked reduction operators that support inputs
    with sparse csr layouts.
    """
    # 检查操作是否支持稀疏 CSR (Compressed Sparse Row) 格式
    if op_info.supports_sparse_csr:
        # 从操作信息中提取操作名称，移除可能的前缀 "masked."
        op_name = op_info.name.replace("masked.", "")
        
        # 遍历生成用于掩码归约操作的样本输入
        for sample_input in sample_inputs_masked_reduction(
            op_info, device, dtype, requires_grad, **kwargs
        ):
            # 如果样本输入的维度不为2或者未设置 keepdim 参数为 True，则跳过当前循环
            if not (
                sample_input.input.ndim == 2 and sample_input.kwargs.get("keepdim")
            ):
                # - 稀疏 CSR 张量始终为2维张量
                # - 只有在 keepdim 为 True 时才定义 CSR 张量的掩码归约操作
                continue
            
            # 获取样本输入中的掩码（mask）
            mask = sample_input.kwargs.get("mask")
            
            # 如果存在掩码，更新样本输入的 kwargs，并创建新的样本输入对象
            if mask is not None:
                sample_input_kwargs = sample_input.kwargs.copy()
                sample_input_kwargs.update(mask=mask.to_sparse_csr())
                new_sample = SampleInput(
                    sample_input.input.to_sparse_csr(),
                    args=sample_input.args,
                    kwargs=sample_input_kwargs,
                )
            else:
                # 如果不存在掩码，并且操作名称在 ["prod", "amax", "amin", "mean"] 中，
                # 则跳过当前循环，因为这些归约操作不支持非零归约身份和未指定的掩码。
                if op_name in ["prod", "amax", "amin", "mean"]:
                    # 对于稀疏 CSR 张量，不支持具有非零归约身份和未指定掩码的归约操作，
                    # 参见 torch.masked.prod 的实现细节。
                    continue
                
                # 否则，创建新的样本输入对象
                new_sample = SampleInput(
                    sample_input.input.to_sparse_csr(),
                    args=sample_input.args,
                    kwargs=sample_input.kwargs,
                )
            
            # 生成新的样本输入对象
            yield new_sample
            
            # 如果样本输入的 kwargs 中的 dim 为 0，
            # 则生成一个额外的样本输入对象，用于测试 CSR 实现的最小化归约操作。
            if sample_input.kwargs["dim"] == 0:
                # CSR 张量的归约操作在内部和/或外部维度上使用不同的实现方式。
                # 因此，作为CSR实现的最小测试，必须生成以下 kwargs：
                #   dict(dim=0, keepdim=True)
                #   dict(dim=1, keepdim=True)
                #   dict(dim=(0, 1), keepdim=True)
                # 这里从 dim=0 的情况生成 dim=1 的情况。
                sample_input_kwargs = new_sample.kwargs.copy()
                sample_input_kwargs.update(dim=1)
                yield SampleInput(
                    new_sample.input.clone(),
                    args=sample_input.args,
                    kwargs=sample_input_kwargs,
                )
# 为 masked_norm 函数生成样本输入
def sample_inputs_masked_norm(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for masked norm."""
    # 遍历不同的范数值，包括 2.0, 1, 无穷大, 负无穷大, 0
    for ord in [2.0, 1, float("inf"), float("-inf"), 0]:
        # 对于每个范数值，生成使用 masked_reduction 函数得到的样本输入
        for sample_input in sample_inputs_masked_reduction(
            op_info, device, dtype, requires_grad, **kwargs
        ):
            # 准备 sample_input 的参数和关键字参数
            sample_input_args, sample_input_kwargs = (
                ord,  # 范数值作为第一个参数
            ) + sample_input.args, sample_input.kwargs.copy()
            # 生成 SampleInput 对象，其中的输入使用 clone 方法进行深拷贝，并设定是否需要梯度
            yield SampleInput(
                sample_input.input.clone().requires_grad_(requires_grad),
                args=sample_input_args,
                kwargs=sample_input_kwargs,
            )


# 生成参考函数 reference_masked_std_var，基于给定的 numpy_fn 函数
def reference_masked_std_var(
    numpy_fn,
):
    # 使用 reference_reduction_numpy 函数创建参考函数 ref
    ref = reference_reduction_numpy(numpy_fn)

    # 将无偏或校正参数转换为 ddof（自由度）参数
    def func(
        input,
        dim=None,
        unbiased=None,
        *,
        correction=None,
        **kwargs,
    ):
        ddof = 1  # 默认自由度为 1
        # 根据 unbiased 参数设定 ddof
        if unbiased is not None:
            ddof = 1 if unbiased else 0
        # 根据 correction 参数设定 ddof
        if correction is not None:
            ddof = correction

        # 如果 dim 是 Sequence 类型，则转换为元组
        if isinstance(dim, Sequence):
            dim = tuple(dim)

        # 调用 ref 函数进行计算，传入 input, dim, ddof 和其他关键字参数
        return ref(input, dim, ddof=ddof, **kwargs)

    return func


# 为 masked_std_var 函数生成样本输入
def sample_inputs_masked_std_var(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for masked std/var."""
    # 设置 kwargs 中的 supports_multiple_dims 参数为 op_info 的属性 supports_multiple_dims
    kwargs["supports_multiple_dims"] = op_info.supports_multiple_dims
    # 导入 sample_inputs_std_var 函数
    from torch.testing._internal.common_methods_invocations import sample_inputs_std_var
    # 定义生成经过掩码处理的样本的生成器函数
    def masked_samples():
        # 对于由 sample_inputs_std_var 生成的每个样本输入
        for sample_input in sample_inputs_std_var(
            op_info, device, dtype, requires_grad, **kwargs
        ):
            # 如果 sample_input.args 非空且第一个参数是布尔类型，则跳过此样本
            if len(sample_input.args) and isinstance(sample_input.args[0], bool):
                continue  # masked.{std, var} doesn't support `.var(unbiased)`

            # 生成掩码操作所需的掩码数组
            for mask in _generate_masked_op_mask(
                sample_input.input.shape, device, **kwargs
            ):
                # 准备用于生成样本输入的参数和关键字参数
                sample_input_args, sample_input_kwargs = sample_input.args, dict(
                    mask=mask, **sample_input.kwargs
                )
                # 生成一个新的 SampleInput 对象，包括输入数据并设置梯度是否需要
                yield SampleInput(
                    sample_input.input.detach().requires_grad_(requires_grad),
                    args=sample_input_args,
                    kwargs=sample_input_kwargs,
                )
                # 如果不需要梯度，并且数据类型是浮点型，输入数据维度为二，且掩码不为空且与输入数据形状相同
                if (
                    not requires_grad
                    and dtype.is_floating_point
                    and sample_input.input.ndim == 2
                    and mask is not None
                    and mask.shape == sample_input.input.shape
                ):
                    # 对于一些特殊值进行填充，以测试 NaN 的情况
                    for v in [torch.inf, -torch.inf, torch.nan]:
                        t = sample_input.input.detach()
                        # 将对角线上的元素填充为特殊值 v
                        t.diagonal(0, -2, -1).fill_(v)
                        # 生成新的 SampleInput 对象，包括填充后的数据
                        yield SampleInput(
                            t.requires_grad_(requires_grad),
                            args=sample_input_args,
                            kwargs=sample_input_kwargs,
                        )

    # 遍历经过掩码处理的样本生成器函数生成的每个样本
    for sample_input in masked_samples():
        # 获取修正值 correction
        correction = sample_input.kwargs.get("correction")
        if correction is None:
            correction = int(sample_input.kwargs.get("unbiased", True))

        # 获取维度参数 dim
        dim = sample_input.kwargs.get("dim", None)

        # 如果掩码参数为 None，则计算原始计数，使用 torch.masked.sum 进行计算
        if sample_input.kwargs.get("mask") is None:
            orig_count = torch.masked.sum(
                torch.ones(sample_input.input.shape, dtype=torch.int64),
                dim,
                keepdim=True,
            )
        else:
            # 否则，计算输入数据的掩码，并计算原始计数
            inmask = torch.masked._input_mask(
                sample_input.input, *sample_input.args, **sample_input.kwargs
            )
            orig_count = torch.masked.sum(
                inmask.new_ones(sample_input.input.shape, dtype=torch.int64),
                dim,
                keepdim=True,
                mask=inmask,
            )
        
        # 如果原始计数的最小值小于等于修正值加一，则跳过此样本
        if orig_count.min() <= correction + 1:
            # 跳过会导致在方差计算中产生 NaN 的样本
            continue

        # 返回当前样本输入
        yield sample_input
# 为masked softmax、log_softmax和softmin操作生成样本输入

def sample_inputs_masked_softmax(
    op_info, device, dtype, requires_grad, with_dtype=False, **kwargs
):
    """Sample inputs for masked softmax, log_softmax, and softmin.

    Masked normalization operator is a reduction operator with
    trailing mask optional argument. A mask is a bool tensor with the
    same shape as input or a shape that is broadcastable to input
    shape.
    """
    # 遍历由sample_inputs_softmax_variant函数生成的样本输入
    for sample_input in sample_inputs_softmax_variant(
        op_info, device, dtype, requires_grad, with_dtype=with_dtype, **kwargs
    ):
        # 生成用于操作的遮罩(mask)
        for mask in _generate_masked_op_mask(
            sample_input.input.shape, device, **kwargs
        ):
            # 生成SampleInput对象，包括输入数据的克隆、是否需要梯度等信息，以及遮罩
            yield SampleInput(
                sample_input.input.clone().requires_grad_(requires_grad),
                *sample_input.args,
                mask=mask,
                **sample_input.kwargs,
            )


# 为masked cumsum和cumprod操作生成样本输入

def sample_inputs_masked_cumops(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for masked cumsum and cumprod."""
    inputs: List[SampleInput] = []
    # 遍历由sample_inputs_softmax_variant函数生成的样本输入
    for sample_input in sample_inputs_softmax_variant(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        # 生成用于操作的遮罩(mask)
        for mask in _generate_masked_op_mask(
            sample_input.input.shape, device, **kwargs
        ):
            # 如果遮罩类型不是torch.Tensor，则跳过当前循环
            if type(mask) != torch.Tensor:
                continue
            # 准备SampleInput的参数和关键字参数，包括遮罩和其他关键字参数
            sample_input_args, sample_input_kwargs = sample_input.args, dict(
                mask=mask, **sample_input.kwargs
            )
            # 如果关键字参数中包含"keepdim"，则移除它
            if "keepdim" in sample_input_kwargs:
                sample_input_kwargs.pop("keepdim")
            # 确保维度信息是必须的
            # 如果有传递参数，则使用第一个参数作为维度
            if sample_input_args:
                dim = sample_input.args[0]
            else:
                # 否则，从关键字参数中获取"dim"，如果不存在则继续下一个循环
                if "dim" not in sample_input_kwargs:
                    continue
                dim = sample_input_kwargs.pop("dim")
                sample_input_args = (dim,)
            # 生成SampleInput对象，包括输入数据的克隆、是否需要梯度等信息，以及遮罩和维度信息
            yield SampleInput(
                sample_input.input.clone().requires_grad_(requires_grad),
                *sample_input_args,
                **sample_input_kwargs,
            )


# 为masked logaddexp操作生成样本输入

def sample_inputs_masked_logaddexp(op_info, device, dtype, requires_grad, **kwargs):
    """Sample inputs for masked logaddexp."""
    # 定义不同形状的样本输入形状
    shapes = [(S,), (S, S), (S, M, S)]
    # 生成输入遮罩列表，用于不同形状的输入
    input_mask_lists = [
        list(_generate_masked_op_mask(shape, device, **kwargs)) for shape in shapes
    ]
    # 生成其他遮罩列表，同样用于不同形状的输入
    other_mask_lists = [
        list(_generate_masked_op_mask(shape, device, **kwargs)) for shape in shapes
    ]

    # 使用make_tensor函数创建张量的部分参数，包括数据类型、设备和是否需要梯度
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )
    # 遍历不同形状的样本输入、输入遮罩列表和其他遮罩列表
    for shape, input_masks, other_masks in zip(
        shapes, input_mask_lists, other_mask_lists
    ):
        # 生成SampleInput对象，包括形状为shape的输入数据、输入遮罩和其他遮罩
        for input_mask, other_mask in zip(input_masks, other_masks):
            yield SampleInput(
                make_arg(shape),
                make_arg(shape),
                input_mask=input_mask,
                other_mask=other_mask,
            )
def sample_inputs_masked_normalize(op_info, device, dtype, requires_grad, **kwargs):
    """生成用于带遮罩标准化的样本输入."""
    # 遍历不同的范数值
    for ord in [2.0, 1, float("inf"), float("-inf"), 0]:
        # 调用 sample_inputs_softmax_variant 函数生成样本输入
        for sample_input in sample_inputs_softmax_variant(
            op_info, device, dtype, requires_grad, use_zero_dimensions=False, **kwargs
        ):
            # 使用生成的样本输入创建 SampleInput 对象，并作为生成器的输出
            yield SampleInput(
                sample_input.input.clone().requires_grad_(requires_grad),
                ord,
                *sample_input.args,
                **sample_input.kwargs,
            )


op_db: List[OpInfo] = [
    # 创建一个 ReductionOpInfo 对象，表示一个特定的降维操作信息
    ReductionOpInfo(
        "masked.sum",  # 操作名称为 "masked.sum"
        ref=reference_reduction_numpy(np.sum),  # 参考的 NumPy 中的 sum 函数
        method_variant=None,  # 操作的方法变体为空
        identity=0,  # 操作的单位元是数字 0
        nan_policy="propagate",  # NaN 策略设定为 "propagate"，即传播 NaN 值
        supports_out=False,  # 不支持输出参数
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和后向梯度
        supports_sparse=True,  # 支持稀疏张量输入
        supports_sparse_csr=True,  # 支持稀疏 CSR 格式的输入
        promotes_int_to_int64=True,  # 推广整数类型为 int64
        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),  # 支持的数据类型包括布尔型、float16 和 bfloat16
        skips=(
            DecorateInfo(
                unittest.skip("Failing on some jobs"),  # 装饰信息，跳过某些测试任务
                "TestReductions",  # 测试类名为 "TestReductions"
                "test_reference_masked",  # 测试方法名为 "test_reference_masked"
                dtypes=(torch.bool, torch.int8, torch.int16, torch.int32),  # 支持的数据类型包括布尔型、int8、int16 和 int32
            ),
            DecorateInfo(
                unittest.expectedFailure,  # 装饰信息，期望测试失败
                "TestNormalizeOperators",  # 测试类名为 "TestNormalizeOperators"
                "test_normalize_operator_exhaustive",  # 测试方法名为 "test_normalize_operator_exhaustive"
            ),
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.expectedFailure, "TestReductions", "test_dim_empty"),  # 装饰信息，期望测试失败，测试类名为 "TestReductions"，测试方法名为 "test_dim_empty"
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_dim_empty_keepdim"
            ),  # 装饰信息，期望测试失败，测试类名为 "TestReductions"，测试方法名为 "test_dim_empty_keepdim"
            # RuntimeError: undefined value tensor
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),  # 装饰信息，期望测试失败，测试类名为 "TestJit"，测试方法名为 "test_variant_consistency_jit"
        ),
        decorators=[
            DecorateInfo(
                toleranceOverride(  # 装饰信息，覆盖容差设置
                    {
                        torch.bfloat16: tol(atol=1e-03, rtol=5e-2),  # 对于 torch.bfloat16 类型，设置绝对容差和相对容差
                        torch.float16: tol(atol=1e-03, rtol=5e-3),  # 对于 torch.float16 类型，设置绝对容差和相对容差
                    }
                ),
                "TestReductions",  # 测试类名为 "TestReductions"
                "test_reference_masked",  # 测试方法名为 "test_reference_masked"
            ),
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-02, rtol=1e-03)}),  # 装饰信息，覆盖容差设置，针对 torch.float16 类型
                "TestReductions",  # 测试类名为 "TestReductions"
                "test_ref_small_input",  # 测试方法名为 "test_ref_small_input"
            ),
            DecorateInfo(
                toleranceOverride(  # 装饰信息，覆盖容差设置
                    {
                        torch.bfloat16: tol(atol=0.1, rtol=0.1),  # 对于 torch.bfloat16 类型，设置绝对容差和相对容差
                        torch.float16: tol(atol=5e-3, rtol=5e-3),  # 对于 torch.float16 类型，设置绝对容差和相对容差
                    }
                ),
                "TestMasked",  # 测试类名为 "TestMasked"
                "test_mask_layout",  # 测试方法名为 "test_mask_layout"
            ),
        ],
        sample_inputs_func=sample_inputs_masked_reduction,  # 样本输入函数为 sample_inputs_masked_reduction
        sample_inputs_sparse_coo_func=sample_inputs_sparse_coo_masked_reduction,  # 稀疏 COO 格式样本输入函数为 sample_inputs_sparse_coo_masked_reduction
        sample_inputs_sparse_csr_func=sample_inputs_sparse_csr_masked_reduction,  # 稀疏 CSR 格式样本输入函数为 sample_inputs_sparse_csr_masked_reduction
    ),
    #`
        # 创建一个 ReductionOpInfo 对象，用于描述一个特定的归约操作
        ReductionOpInfo(
            "masked.prod",  # 操作的名称为 "masked.prod"
            ref=prod_numpy,  # 参考实现使用的是 prod_numpy 函数
            method_variant=None,  # 方法变体为 None，表示没有特定的方法变体
            identity=1,  # 归约操作的单位元为 1
            nan_policy="propagate",  # 处理 NaN 的策略为 "propagate"
            # 设置用于解决问题的快速梯度检查模式，参见 https://github.com/pytorch/pytorch/issues/80411
            gradcheck_fast_mode=True,
            supports_out=False,  # 不支持输出张量 out 参数
            supports_forward_ad=True,  # 支持前向自动微分
            supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
            supports_sparse=True,  # 支持稀疏张量
            supports_sparse_csr=True,  # 支持稀疏 CSR 格式张量
            promotes_int_to_int64=True,  # 促进整数类型提升到 int64
            dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),  # 支持的数据类型包括所有类型、复数类型、torch.bool、torch.float16 和 torch.bfloat16
            skips=(
                # 跳过以下测试用例
                DecorateInfo(
                    unittest.expectedFailure,
                    "TestNormalizeOperators",
                    "test_normalize_operator_exhaustive",
                ),
                DecorateInfo(
                    unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
                ),
                DecorateInfo(
                    unittest.skip("Failing on some jobs"),
                    "TestReductions",
                    "test_reference_masked",
                    dtypes=(torch.bool, torch.int8, torch.int16, torch.int32),
                ),
                DecorateInfo(
                    "TestReductions",
                    "test_ref_small_input",
                    dtypes=(torch.int8, torch.int16, torch.int32),
                ),
                # FIXME: "cuda_scatter_gather_base_kernel_func" not implemented for ... (used for sparse_coo inputs)
                DecorateInfo(
                    unittest.skip("Skipped!"),
                    "TestMasked",
                    "test_mask_layout",
                    device_type="cuda",
                    dtypes=(torch.bool, *integral_types(), *complex_types()),
                ),
            ),
            decorators=[
                # 设置容差覆盖以及其对应的测试用例和方法
                DecorateInfo(
                    toleranceOverride({torch.float16: tol(atol=1e-03, rtol=1e-02)}),
                    "TestReductions",
                    "test_reference_masked",
                ),
                DecorateInfo(
                    toleranceOverride({torch.float16: tol(atol=1e-03, rtol=1e-03)}),
                    "TestReductions",
                    "test_ref_duplicate_values",
                ),
                DecorateInfo(
                    toleranceOverride({torch.float16: tol(atol=1e-03, rtol=1e-03)}),
                    "TestReductions",
                    "test_ref_small_input",
                ),
                DecorateInfo(
                    toleranceOverride({torch.float16: tol(atol=1e-02, rtol=1.5e-03)}),
                    "TestMasked",
                    "test_mask_layout",
                    device_type="cpu",
                ),
            ],
            sample_inputs_func=sample_inputs_masked_reduction,  # 提供样本输入函数用于掩码归约操作
            sample_inputs_sparse_coo_func=sample_inputs_sparse_coo_masked_reduction,  # 提供稀疏 COO 格式样本输入函数
            sample_inputs_sparse_csr_func=sample_inputs_sparse_csr_masked_reduction,  # 提供稀疏 CSR 格式样本输入函数
        ),
    OpInfo(
        "masked.cumsum",  # 定义操作名称为 "masked.cumsum"
        dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),  # 定义适用的数据类型范围
        method_variant=None,  # 指定方法变体为 None
        gradcheck_fast_mode=True,  # 使用快速模式进行梯度检查，因为在慢速模式下运行较慢
        supports_out=False,  # 不支持输出参数
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度与后向梯度
        skips=(  # 定义跳过的测试用例集合
            DecorateInfo(
                unittest.expectedFailure,  # 标记预期失败的装饰器
                "TestNormalizeOperators",  # 测试类名
                "test_normalize_operator_exhaustive",  # 测试方法名
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 标记跳过的装饰器，并注明跳过原因
                "TestJit",  # 测试类名
                "test_variant_consistency_jit"  # 测试方法名
            ),
        ),
        sample_inputs_func=sample_inputs_masked_cumops,  # 提供生成样本输入的函数
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,  # 梯度检查的包装函数
    ),
    OpInfo(
        "masked.cumprod",  # 定义操作名称为 "masked.cumprod"
        dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),  # 定义适用的数据类型范围
        method_variant=None,  # 指定方法变体为 None
        gradcheck_fast_mode=True,  # 使用快速模式进行梯度检查，因为在慢速模式下运行较慢
        supports_out=False,  # 不支持输出参数
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度与后向梯度
        skips=(  # 定义跳过的测试用例集合
            DecorateInfo(
                unittest.expectedFailure,  # 标记预期失败的装饰器
                "TestNormalizeOperators",  # 测试类名
                "test_normalize_operator_exhaustive",  # 测试方法名
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 标记跳过的装饰器，并注明跳过原因
                "TestJit",  # 测试类名
                "test_variant_consistency_jit"  # 测试方法名
            ),
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=1e-5, rtol=1e-5)}),  # 设置容忍度修正器，指定浮点数类型的绝对误差和相对误差
                "TestCompositeCompliance",  # 测试类名
                "test_backward",  # 测试方法名
                device_type="cuda",  # 指定设备类型为 CUDA
            ),
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=2e-3, rtol=2e-3)}),  # 设置容忍度修正器，指定浮点数类型的绝对误差和相对误差
                "TestInductorOpInfo",  # 测试类名
                "test_comprehensive",  # 测试方法名
                device_type="cuda",  # 指定设备类型为 CUDA
            ),
        ),
        sample_inputs_func=sample_inputs_masked_cumops,  # 提供生成样本输入的函数
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,  # 梯度检查的包装函数
    ),
    # 创建一个 ReductionOpInfo 对象，用于描述一种特定的减少操作 "masked.amax"
    ReductionOpInfo(
        "masked.amax",
        nan_policy="propagate",  # 指定处理 NaN 值的策略为 "propagate"
        supports_out=False,  # 表示不支持输出参数
        dtypes=all_types_and(torch.float16, torch.bfloat16),  # 支持的数据类型包括所有类型以及 torch.float16 和 torch.bfloat16
        supports_sparse=True,  # 支持稀疏张量
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        supports_sparse_csr=True,  # 支持稀疏 CSR 格式
        ref=reference_reduction_numpy(np.amax),  # 参考实现使用 NumPy 的 np.amax 函数
        skips=(
            DecorateInfo(
                unittest.expectedFailure,  # 标记为预期失败的测试
                "TestNormalizeOperators",  # 所属测试类名为 "TestNormalizeOperators"
                "test_normalize_operator_exhaustive",  # 具体测试方法名为 "test_normalize_operator_exhaustive"
            ),
            DecorateInfo(
                unittest.expectedFailure,  # 标记为预期失败的测试
                "TestReductions",  # 所属测试类名为 "TestReductions"
                "test_dim_empty",  # 具体测试方法名为 "test_dim_empty"
            ),
            DecorateInfo(
                unittest.expectedFailure,  # 标记为预期失败的测试
                "TestReductions",  # 所属测试类名为 "TestReductions"
                "test_dim_empty_keepdim",  # 具体测试方法名为 "test_dim_empty_keepdim"
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 标记为跳过的测试，注释为 "Skipped!"
                "TestJit",  # 所属测试类名为 "TestJit"
                "test_variant_consistency_jit",  # 具体测试方法名为 "test_variant_consistency_jit"
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 标记为跳过的测试，注释为 "Skipped!"
                "TestMasked",  # 所属测试类名为 "TestMasked"
                "test_mask_layout",  # 具体测试方法名为 "test_mask_layout"
                dtypes=(torch.bool, *integral_types(), *complex_types()),  # 测试的数据类型包括 torch.bool 以及整数和复数类型
            ),
        ),
        sample_inputs_func=sample_inputs_masked_reduction,  # 提供样本输入函数 sample_inputs_masked_reduction
        sample_inputs_sparse_coo_func=sample_inputs_sparse_coo_masked_reduction,  # 提供稀疏 COO 格式输入的样本输入函数
        sample_inputs_sparse_csr_func=sample_inputs_sparse_csr_masked_reduction,  # 提供稀疏 CSR 格式输入的样本输入函数
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,  # 梯度检查的包装器函数为 gradcheck_wrapper_masked_operation
    ),
    ReductionOpInfo(
        "masked.amin",  # 定义一个用于最小值计算的操作信息对象，操作名为"masked.amin"
        nan_policy="propagate",  # 设置处理 NaN 值的策略为"propagate"
        supports_out=False,  # 不支持输出参数
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向和后向梯度
        dtypes=all_types_and(torch.float16, torch.bfloat16),  # 支持所有数据类型以及 torch.float16 和 torch.bfloat16
        supports_sparse=True,  # 支持稀疏张量
        supports_sparse_csr=True,  # 支持稀疏 CSR 格式的张量
        ref=reference_reduction_numpy(np.amin),  # 参考 NumPy 中 np.amin 函数的实现作为参考
        skips=(
            DecorateInfo(
                unittest.expectedFailure,  # 标记为预期失败的测试
                "TestNormalizeOperators",  # 测试用例类名为 "TestNormalizeOperators"
                "test_normalize_operator_exhaustive",  # 具体测试方法名为 "test_normalize_operator_exhaustive"
            ),
            # FIXME: 当 dim=[] 时，amax 函数会对所有维度进行降维操作
            DecorateInfo(unittest.expectedFailure, "TestReductions", "test_dim_empty"),
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_dim_empty_keepdim"
            ),
            # 运行时错误：未知的内置操作：aten::iinfo
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # FIXME: "cuda_scatter_gather_base_kernel_func" 在稀疏 COO 输入时未实现
            # FIXME: "_segment_reduce_lengths_cpu/cuda" 在稀疏 CSR 输入时未实现
            DecorateInfo(
                unittest.skip("Skipped!"),  # 跳过测试，并附加消息 "Skipped!"
                "TestMasked",  # 测试用例类名为 "TestMasked"
                "test_mask_layout",  # 具体测试方法名为 "test_mask_layout"
                dtypes=(torch.bool, *integral_types(), *complex_types()),  # 测试使用的数据类型列表
            ),
        ),
        sample_inputs_func=sample_inputs_masked_reduction,  # 用于生成输入样本的函数
        sample_inputs_sparse_coo_func=sample_inputs_sparse_coo_masked_reduction,  # 用于生成稀疏 COO 格式输入样本的函数
        sample_inputs_sparse_csr_func=sample_inputs_sparse_csr_masked_reduction,  # 用于生成稀疏 CSR 格式输入样本的函数
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,  # 梯度检查包装器函数
    ),
    ReductionOpInfo(
        "masked.argmax",  # 定义一个用于最大值计算的操作信息对象，操作名为"masked.argmax"
        supports_out=False,  # 不支持输出参数
        supports_multiple_dims=False,  # 不支持多维度计算
        supports_autograd=False,  # 不支持自动求导
        dtypes=all_types_and(torch.float16, torch.bfloat16),  # 支持所有数据类型以及 torch.float16 和 torch.bfloat16
        ref=reference_reduction_numpy(np.argmax, supports_keepdims=False),  # 参考 NumPy 中 np.argmax 函数的实现，不支持 keepdims 参数
        skips=(
            DecorateInfo(
                unittest.expectedFailure,  # 标记为预期失败的测试
                "TestNormalizeOperators",  # 测试用例类名为 "TestNormalizeOperators"
                "test_normalize_operator_exhaustive",  # 具体测试方法名为 "test_normalize_operator_exhaustive"
            ),
            # 初始值不是 argmax 函数的关键字参数
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_reference_masked"
            ),
            # NotSupportedError: 编译函数无法使用带有默认值的关键字参数
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
        ),
        sample_inputs_func=sample_inputs_masked_reduction,  # 用于生成输入样本的函数
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,  # 梯度检查包装器函数
    ),
    ReductionOpInfo(
        "masked.argmin",  # 定义一个名为 "masked.argmin" 的操作信息对象
        supports_out=False,  # 不支持输出参数
        supports_multiple_dims=False,  # 不支持多维度操作
        supports_autograd=False,  # 不支持自动求导
        dtypes=all_types_and(torch.float16, torch.bfloat16),  # 支持所有类型以及 torch.float16 和 torch.bfloat16 类型
        ref=reference_reduction_numpy(np.argmin, supports_keepdims=False),  # 参考 numpy 的 np.argmin 函数作为参考实现
        skips=(  # 跳过以下测试用例
            DecorateInfo(
                unittest.expectedFailure,  # 预期测试失败
                "TestNormalizeOperators",  # 测试类名为 "TestNormalizeOperators"
                "test_normalize_operator_exhaustive",  # 测试方法名为 "test_normalize_operator_exhaustive"
            ),
            DecorateInfo(
                unittest.expectedFailure,  # 预期测试失败
                "TestReductions",  # 测试类名为 "TestReductions"
                "test_reference_masked"  # 测试方法名为 "test_reference_masked"
            ),
            DecorateInfo(
                unittest.expectedFailure,  # 预期测试失败
                "TestJit",  # 测试类名为 "TestJit"
                "test_variant_consistency_jit"  # 测试方法名为 "test_variant_consistency_jit"
            ),
        ),
        sample_inputs_func=sample_inputs_masked_reduction,  # 使用 sample_inputs_masked_reduction 函数生成示例输入
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,  # 使用 gradcheck_wrapper_masked_operation 进行梯度检查
    ),
    OpInfo(
        "masked.median",  # 定义一个名为 "masked.median" 的操作信息对象
        dtypes=floating_types_and(torch.bfloat16, torch.float16),  # 支持浮点类型以及 torch.bfloat16 和 torch.float16 类型
        method_variant=None,  # 方法变体为 None
        supports_out=False,  # 不支持输出参数
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向和后向梯度计算
        skips=(  # 跳过以下测试用例
            DecorateInfo(
                unittest.expectedFailure,  # 预期测试失败
                "TestNormalizeOperators",  # 测试类名为 "TestNormalizeOperators"
                "test_normalize_operator_exhaustive"  # 测试方法名为 "test_normalize_operator_exhaustive"
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 跳过测试，注释为 "Skipped!"
                "TestJit",  # 测试类名为 "TestJit"
                "test_variant_consistency_jit"  # 测试方法名为 "test_variant_consistency_jit"
            ),
        ),
        sample_inputs_func=partial(
            sample_inputs_masked_softmax, use_zero_dimensions=False  # 使用部分应用的 sample_inputs_masked_softmax 函数，使用零维度为 False
        ),
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,  # 使用 gradcheck_wrapper_masked_operation 进行梯度检查
    ),
    # 创建一个 ReductionOpInfo 对象，描述了 "masked.norm" 操作的规约信息
    ReductionOpInfo(
        "masked.norm",  # 操作的标识符为 "masked.norm"
        identity=0,  # 标识元素为 0
        method_variant=None,  # 没有指定特定的方法变体
        nan_policy="propagate",  # 处理 NaN 值时传播 NaN
        supports_out=False,  # 不支持输出参数
        promotes_int_to_float=True,  # 支持将整数提升为浮点数
        dtypes=floating_types_and(torch.float16, torch.bfloat16),  # 支持的数据类型为浮点数类型和 torch.float16、torch.bfloat16
        skips=(
            DecorateInfo(
                unittest.expectedFailure,  # 标记为预期失败的测试
                "TestNormalizeOperators",  # 测试类为 "TestNormalizeOperators"
                "test_normalize_operator_exhaustive",  # 测试方法为 "test_normalize_operator_exhaustive"
            ),
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(
                unittest.expectedFailure,  # 标记为预期失败的测试
                "TestReductions",  # 测试类为 "TestReductions"
                "test_dim_empty",  # 测试方法为 "test_dim_empty"
            ),
            DecorateInfo(
                unittest.expectedFailure,  # 标记为预期失败的测试
                "TestReductions",  # 测试类为 "TestReductions"
                "test_dim_empty_keepdim"  # 测试方法为 "test_dim_empty_keepdim"
            ),
            # torch.jit.frontend.NotSupportedError: Compiled functions
            # can't take variable number of arguments or use
            # keyword-only arguments with defaults
            DecorateInfo(
                unittest.expectedFailure,  # 标记为预期失败的测试
                "TestJit",  # 测试类为 "TestJit"
                "test_variant_consistency_jit"  # 测试方法为 "test_variant_consistency_jit"
            ),
        ),
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向和后向自动求导
        sample_inputs_func=sample_inputs_masked_norm,  # 用于获取样本输入的函数为 sample_inputs_masked_norm
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,  # 用于梯度检查的包装器函数为 gradcheck_wrapper_masked_operation
    ),
    # 创建一个 ReductionOpInfo 对象，用于描述某个归约操作的信息
    ReductionOpInfo(
        "masked.var",  # 归约操作的名称
        ref=reference_masked_std_var(np.var)  # 根据 numpy 版本选择参考的实现函数
        if np.lib.NumpyVersion(np.__version__) >= "1.20.2"  # 根据 numpy 版本选择参考函数的条件判断
        else None,  # 如果条件不满足，则参考函数为 None
        method_variant=None,  # 方法变体为 None
        nan_policy="propagate",  # 处理 NaN 的策略为传播
        supports_out=False,  # 不支持输出参数
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        # 查看 https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,  # 关闭批处理前向梯度的检查
        promotes_int_to_float=True,  # 提升整数为浮点数
        dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),  # 支持的数据类型
        skips=(
            # conj 和 torch 调度存在问题，参见 https://github.com/pytorch/pytorch/issues/82479
            DecorateInfo(
                unittest.skip("Skipped!"),  # 跳过该测试
                "TestSchemaCheckModeOpInfo",  # 测试类名
                "test_schema_correctness",  # 测试方法名
                dtypes=(torch.complex64, torch.complex128),  # 测试的数据类型
            ),
            DecorateInfo(
                unittest.expectedFailure,  # 预期该测试会失败
                "TestNormalizeOperators",  # 测试类名
                "test_normalize_operator_exhaustive",  # 测试方法名
            ),
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.expectedFailure, "TestReductions", "test_dim_empty"),  # 预期该测试会失败
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_dim_empty_keepdim"
            ),  # 预期该测试会失败
            # RuntimeError: undefined value tensor
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),  # 预期该测试会失败
        ),
        decorators=[
            DecorateInfo(
                toleranceOverride(
                    {
                        torch.float16: tol(atol=1e-02, rtol=1e-02),  # 设置容差
                        torch.bfloat16: tol(atol=1e-03, rtol=1e-03),
                    }
                ),
                "TestReductions",  # 测试类名
                "test_reference_masked",  # 测试方法名
            ),
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-02, rtol=1e-02)}),
                "TestReductions",  # 测试类名
                "test_ref_small_input",  # 测试方法名
            ),
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-02, rtol=1e-02)}),
                "TestMasked",  # 测试类名
                "test_reference_masked",  # 测试方法名
            ),
            DecorateInfo(
                toleranceOverride(
                    {
                        torch.float16: tol(atol=1e-02, rtol=1e-02),  # 设置容差
                        torch.bfloat16: tol(atol=1e-03, rtol=1e-03),
                    }
                ),
                "TestMasked",  # 测试类名
                "test_reference_masked",  # 测试方法名
            ),
        ],
        sample_inputs_func=sample_inputs_masked_std_var,  # 设置样本输入函数
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,  # 设置梯度检查包装器
        check_batched_grad=True,  # 检查批处理梯度
    ),
    ReductionOpInfo(
        "masked.std",  # 创建一个 ReductionOpInfo 对象，用于计算带掩码的标准差
        ref=reference_masked_std_var(np.std)  # 如果 NumPy 版本 >= 1.20.2，则使用 reference_masked_std_var(np.std) 作为参考值函数，否则为 None
        if np.lib.NumpyVersion(np.__version__) >= "1.20.2" else None,
        method_variant=None,  # 方法变体设置为 None
        nan_policy="propagate",  # 在计算中遇到 NaN 值时，传播该值
        gradcheck_fast_mode=True,  # 在慢速 gradcheck 中运行时选择快速模式
        supports_out=False,  # 不支持输出参数
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向和后向自动微分
        check_batched_forward_grad=False,  # 不检查批量前向梯度
        promotes_int_to_float=True,  # 将整数提升为浮点数
        dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),  # 数据类型包括所有类型、复数类型、torch.half 和 torch.bfloat16
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),  # 标记为跳过测试，并注明原因
                "TestSchemaCheckModeOpInfo",  # 测试类名
                "test_schema_correctness",  # 测试方法名
                dtypes=(torch.complex64, torch.complex128),  # 测试的数据类型为 torch.complex64 和 torch.complex128
            ),
            DecorateInfo(
                unittest.expectedFailure,  # 标记为预期失败的测试
                "TestNormalizeOperators",  # 测试类名
                "test_normalize_operator_exhaustive",  # 测试方法名
            ),
            DecorateInfo(
                unittest.expectedFailure,  # 标记为预期失败的测试
                "TestReductions",  # 测试类名
                "test_dim_empty",  # 测试方法名
            ),
            DecorateInfo(
                unittest.expectedFailure,  # 标记为预期失败的测试
                "TestReductions",  # 测试类名
                "test_dim_empty_keepdim"  # 测试方法名
            ),
            DecorateInfo(
                unittest.expectedFailure,  # 标记为预期失败的测试
                "TestJit",  # 测试类名
                "test_variant_consistency_jit"  # 测试方法名
            ),
        ),
        decorators=[
            DecorateInfo(
                toleranceOverride(  # 设置容差覆盖函数，用于设置不同数据类型的容差
                    {
                        torch.bfloat16: tol(atol=1e-02, rtol=1e-02),  # torch.bfloat16 数据类型的容差设置
                        torch.float16: tol(atol=1e-02, rtol=1e-02),  # torch.float16 数据类型的容差设置
                    }
                ),
                "TestReductions",  # 测试类名
                "test_reference_masked",  # 测试方法名
            ),
            DecorateInfo(
                toleranceOverride({torch.float16: tol(atol=1e-02, rtol=1e-02)}),  # 设置 torch.float16 数据类型的容差
                "TestReductions",  # 测试类名
                "test_ref_small_input",  # 测试方法名
            ),
            DecorateInfo(
                toleranceOverride(  # 设置容差覆盖函数，用于设置不同数据类型的容差
                    {
                        torch.float16: tol(atol=1e-02, rtol=1e-02),  # torch.float16 数据类型的容差设置
                        torch.bfloat16: tol(atol=5e-03, rtol=5e-04),  # torch.bfloat16 数据类型的容差设置
                    }
                ),
                "TestMasked",  # 测试类名
                "test_reference_masked",  # 测试方法名
            ),
        ],
        sample_inputs_func=sample_inputs_masked_std_var,  # 设置样本输入函数
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,  # 设置 gradcheck 包装器函数
        check_batched_grad=True,  # 检查批量梯度计算
    ),
    OpInfo(
        "masked.softmax",  # 定义操作名称为 "masked.softmax"
        method_variant=None,  # 方法变体为空
        dtypes=floating_types_and(torch.half, torch.bfloat16),  # 数据类型为浮点类型以及 torch.half 和 torch.bfloat16
        sample_inputs_func=sample_inputs_masked_softmax,  # 样本输入函数为 sample_inputs_masked_softmax
        skips=(  # 跳过的测试装饰器列表开始
            DecorateInfo(
                unittest.expectedFailure,  # 预期失败装饰器
                "TestNormalizeOperators",  # 测试类名为 "TestNormalizeOperators"
                "test_normalize_operator_exhaustive",  # 测试方法名为 "test_normalize_operator_exhaustive"
            ),
            DecorateInfo(
                unittest.expectedFailure,  # 预期失败装饰器
                "TestJit",  # 测试类名为 "TestJit"
                "test_variant_consistency_jit"  # 测试方法名为 "test_variant_consistency_jit"
            ),
        ),  # 跳过的测试装饰器列表结束
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,  # 梯度检查包装器为 gradcheck_wrapper_masked_operation
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和后向梯度
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        "masked.log_softmax",  # 定义操作名称为 "masked.log_softmax"
        method_variant=None,  # 方法变体为空
        dtypes=floating_types_and(torch.half, torch.bfloat16),  # 数据类型为浮点类型以及 torch.half 和 torch.bfloat16
        sample_inputs_func=sample_inputs_masked_softmax,  # 样本输入函数为 sample_inputs_masked_softmax
        skips=(  # 跳过的测试装饰器列表开始
            DecorateInfo(
                unittest.expectedFailure,  # 预期失败装饰器
                "TestNormalizeOperators",  # 测试类名为 "TestNormalizeOperators"
                "test_normalize_operator_exhaustive",  # 测试方法名为 "test_normalize_operator_exhaustive"
            ),
            DecorateInfo(
                unittest.expectedFailure,  # 预期失败装饰器
                "TestJit",  # 测试类名为 "TestJit"
                "test_variant_consistency_jit"  # 测试方法名为 "test_variant_consistency_jit"
            ),
        ),  # 跳过的测试装饰器列表结束
        decorators=[  # 装饰器列表开始
            DecorateInfo(
                toleranceOverride({torch.bfloat16: tol(atol=1e-02, rtol=1e-02)}),  # 覆盖容差为 torch.bfloat16 类型的容差为 atol=1e-02, rtol=1e-02
                "TestMasked",  # 测试类名为 "TestMasked"
                "test_reference_masked",  # 测试方法名为 "test_reference_masked"
            ),
        ],  # 装饰器列表结束
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,  # 梯度检查包装器为 gradcheck_wrapper_masked_operation
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和后向梯度
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        "masked.softmin",  # 定义操作名称为 "masked.softmin"
        method_variant=None,  # 方法变体为空
        dtypes=floating_types_and(torch.half, torch.bfloat16),  # 数据类型为浮点类型以及 torch.half 和 torch.bfloat16
        sample_inputs_func=sample_inputs_masked_softmax,  # 样本输入函数为 sample_inputs_masked_softmax
        skips=(  # 跳过的测试装饰器列表开始
            DecorateInfo(
                unittest.expectedFailure,  # 预期失败装饰器
                "TestNormalizeOperators",  # 测试类名为 "TestNormalizeOperators"
                "test_normalize_operator_exhaustive",  # 测试方法名为 "test_normalize_operator_exhaustive"
            ),
            DecorateInfo(
                unittest.expectedFailure,  # 预期失败装饰器
                "TestJit",  # 测试类名为 "TestJit"
                "test_variant_consistency_jit"  # 测试方法名为 "test_variant_consistency_jit"
            ),
        ),  # 跳过的测试装饰器列表结束
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,  # 梯度检查包装器为 gradcheck_wrapper_masked_operation
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和后向梯度
        supports_out=False,  # 不支持输出
    ),
    OpInfo(
        "masked.normalize",
        method_variant=None,
        dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_masked_normalize,
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
        ),
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,
        # 在慢速的梯度检查中运行非常缓慢 - 或者减少输入大小作为替代方法
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
    ),


    OpInfo(
        "masked.logaddexp",
        dtypes=floating_types_and(torch.float16, torch.bfloat16),
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_forward_grad=False,
        skips=(
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # NotSupportedError: Compiled functions can't ... use keyword-only arguments with defaults
            DecorateInfo(
                unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit"
            ),
            DecorateInfo(
                unittest.skip("Skipped!"), "TestFwdGradients", "test_fn_gradgrad"
            ),
            DecorateInfo(
                unittest.skip("Skipped!"), "TestBwdGradients", "test_fn_gradgrad"
            ),
        ),
        sample_inputs_func=sample_inputs_masked_logaddexp,
        gradcheck_wrapper=gradcheck_wrapper_masked_pointwise_operation,
    ),
    # 创建一个 ReductionOpInfo 对象，描述了一个被掩码的对数求和操作
    ReductionOpInfo(
        "masked.logsumexp",  # 操作的名称为 "masked.logsumexp"
        dtypes=all_types_and(torch.half, torch.bfloat16),  # 支持的数据类型包括半精度和 bfloat16
        method_variant=None,  # 操作的方法变体为空
        nan_policy="propagate",  # 处理 NaN 的策略是传播（propagate）
        supports_out=False,  # 不支持输出参数
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        skips=(
            # 被装饰为跳过的测试信息
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
            # FIXME: 当维度为 [] 时，减少所有维度
            DecorateInfo(unittest.skip("Skipped!"), "TestReductions", "test_dim_empty"),
            # 被装饰为跳过的测试信息，当维度为 [] 时保持维度
            DecorateInfo(
                unittest.skip("Skipped!"), "TestReductions", "test_dim_empty_keepdim"
            ),
            # 如果身份是 -torch.inf 则会发生溢出
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestReductions",
                "test_empty_tensor_empty_slice",
            ),
            # NotSupportedError: 编译函数不能...使用带默认值的关键字参数
            DecorateInfo(
                unittest.skip("Skipped!"), "TestJit", "test_variant_consistency_jit"
            ),
            # 所有的值都相同，除了 -inf 和 nan
            DecorateInfo(unittest.skip("Skipped!"), "TestDecomp", "test_comprehensive"),
        ),
        sample_inputs_func=sample_inputs_masked_reduction,  # 用于生成输入样本的函数
        gradcheck_wrapper=gradcheck_wrapper_masked_operation,  # 梯度检查的包装器函数
    ),
]


注释：


# 这是一个单独的方括号，不在任何语法结构中使用，需要检查是否意外或缺失了代码段。
```