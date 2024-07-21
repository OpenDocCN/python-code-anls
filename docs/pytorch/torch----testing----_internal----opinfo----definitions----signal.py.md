# `.\pytorch\torch\testing\_internal\opinfo\definitions\signal.py`

```py
# 忽略 mypy 类型检查的错误，通常用于声明在类型检查中不需要处理的情况
# 导入单元测试模块 unittest
import unittest
# 导入 functools 模块的 partial 函数，用于部分应用函数参数
from functools import partial
# 导入 itertools 模块的 product 函数，用于生成迭代器中的笛卡尔积
from itertools import product
# 导入 typing 模块中的 Callable、List 和 Tuple 类型
from typing import Callable, List, Tuple

# 导入 numpy 库
import numpy

# 导入 torch 库
import torch
# 从 torch.testing._internal.common_dtype 中导入 floating_types
from torch.testing._internal.common_dtype import floating_types
# 从 torch.testing._internal.common_utils 中导入 TEST_SCIPY
from torch.testing._internal.common_utils import TEST_SCIPY
# 从 torch.testing._internal.opinfo.core 中导入 DecorateInfo、ErrorInput、OpInfo、SampleInput 类
from torch.testing._internal.opinfo.core import (
    DecorateInfo,
    ErrorInput,
    OpInfo,
    SampleInput,
)

# 如果 TEST_SCIPY 为真，则导入 scipy.signal 模块
if TEST_SCIPY:
    import scipy.signal


# 定义 sample_inputs_window 函数，生成窗口操作的样本输入
def sample_inputs_window(op_info, device, dtype, requires_grad, *args, **kwargs):
    r"""Base function used to create sample inputs for windows.

    For additional required args you should use *args, as well as **kwargs for
    additional keyword arguments.
    """

    # 测试窗口大小最多为 5 个样本。
    for size, sym in product(range(6), (True, False)):
        # 生成 SampleInput 对象，包含窗口大小、符号、设备、数据类型和梯度需求等信息
        yield SampleInput(
            size,
            *args,
            sym=sym,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            **kwargs,
        )


# 定义 reference_inputs_window 函数，生成窗口操作的参考输入
def reference_inputs_window(op_info, device, dtype, requires_grad, *args, **kwargs):
    r"""Reference inputs function to use for windows which have a common signature, i.e.,
    window size and sym only.

    Implement other special functions for windows that have a specific signature.
    See exponential and gaussian windows for instance.
    """
    # 使用 sample_inputs_window 函数生成基本的样本输入
    yield from sample_inputs_window(
        op_info, device, dtype, requires_grad, *args, **kwargs
    )

    # 定义特定窗口大小的测试用例
    cases = (8, 16, 32, 64, 128, 256)

    # 遍历测试用例，生成相应的 SampleInput 对象
    for size in cases:
        yield SampleInput(size, sym=False)
        yield SampleInput(size, sym=True)


# 定义 reference_inputs_exponential_window 函数，生成指数窗口操作的参考输入
def reference_inputs_exponential_window(
    op_info, device, dtype, requires_grad, **kwargs
):
    # 使用 sample_inputs_window 函数生成基本的样本输入
    yield from sample_inputs_window(op_info, device, dtype, requires_grad, **kwargs)

    # 定义特定的指数窗口大小和参数
    cases = (
        (8, {"center": 4, "tau": 0.5}),
        (16, {"center": 8, "tau": 2.5}),
        (32, {"center": 16, "tau": 43.5}),
        (64, {"center": 20, "tau": 3.7}),
        (128, {"center": 62, "tau": 99}),
        (256, {"tau": 10}),
    )

    # 遍历测试用例，生成相应的 SampleInput 对象
    for size, kw in cases:
        yield SampleInput(size, sym=False, **kw)
        kw["center"] = None
        yield SampleInput(size, sym=True, **kw)


# 定义 reference_inputs_gaussian_window 函数，生成高斯窗口操作的参考输入
def reference_inputs_gaussian_window(op_info, device, dtype, requires_grad, **kwargs):
    # 使用 sample_inputs_window 函数生成基本的样本输入
    yield from sample_inputs_window(op_info, device, dtype, requires_grad, **kwargs)

    # 定义特定的高斯窗口大小和参数
    cases = (
        (8, {"std": 0.1}),
        (16, {"std": 1.2}),
        (32, {"std": 2.1}),
        (64, {"std": 3.9}),
        (128, {"std": 4.5}),
        (256, {"std": 10}),
    )

    # 遍历测试用例，生成相应的 SampleInput 对象
    for size, kw in cases:
        yield SampleInput(size, sym=False, **kw)
        yield SampleInput(size, sym=True, **kw)


# 定义 reference_inputs_kaiser_window 函数，生成凯泽窗口操作的参考输入
def reference_inputs_kaiser_window(op_info, device, dtype, requires_grad, **kwargs):
    # 使用 sample_inputs_window 函数生成基本的样本输入
    yield from sample_inputs_window(op_info, device, dtype, requires_grad, **kwargs)
    # 定义测试用例，每个元组包含一个整数和一个参数字典
    cases = (
        (8, {"beta": 2}),
        (16, {"beta": 12}),
        (32, {"beta": 30}),
        (64, {"beta": 35}),
        (128, {"beta": 41.2}),
        (256, {"beta": 100}),
    )

    # 遍历测试用例集合
    for size, kw in cases:
        # 生成一个不对称的 SampleInput 对象，并返回
        yield SampleInput(size, sym=False, **kw)
        # 生成一个对称的 SampleInput 对象，并返回
        yield SampleInput(size, sym=True, **kw)
# 生成通用余弦窗口的参考输入
def reference_inputs_general_cosine_window(
    op_info, device, dtype, requires_grad, **kwargs
):
    # 调用函数生成窗口样本输入
    yield from sample_inputs_window(op_info, device, dtype, requires_grad, **kwargs)

    # 定义不同窗口大小和参数的测试案例
    cases = (
        (8, {"a": [0.5, 0.5]}),
        (16, {"a": [0.46, 0.54]}),
        (32, {"a": [0.46, 0.23, 0.31]}),
        (64, {"a": [0.5]}),
        (128, {"a": [0.1, 0.8, 0.05, 0.05]}),
        (256, {"a": [0.2, 0.2, 0.2, 0.2, 0.2]}),
    )

    # 遍历测试案例，生成对应的样本输入
    for size, kw in cases:
        yield SampleInput(size, sym=False, **kw)
        yield SampleInput(size, sym=True, **kw)


# 生成通用汉明窗口的参考输入
def reference_inputs_general_hamming_window(
    op_info, device, dtype, requires_grad, **kwargs
):
    # 调用函数生成窗口样本输入
    yield from sample_inputs_window(op_info, device, dtype, requires_grad, **kwargs)

    # 定义不同窗口大小和参数的测试案例
    cases = (
        (8, {"alpha": 0.54}),
        (16, {"alpha": 0.5}),
        (32, {"alpha": 0.23}),
        (64, {"alpha": 0.8}),
        (128, {"alpha": 0.9}),
        (256, {"alpha": 0.05}),
    )

    # 遍历测试案例，生成对应的样本输入
    for size, kw in cases:
        yield SampleInput(size, sym=False, **kw)
        yield SampleInput(size, sym=True, **kw)


# 生成窗口错误输入
def error_inputs_window(op_info, device, *args, **kwargs):
    # 测试负大小窗口的情况
    yield ErrorInput(
        SampleInput(-1, *args, dtype=torch.float32, device=device, **kwargs),
        error_type=ValueError,
        error_regex="requires non-negative window length, got M=-1",
    )

    # 测试非 torch.strided 张量的窗口，例如 torch.sparse_coo
    yield ErrorInput(
        SampleInput(
            3,
            *args,
            layout=torch.sparse_coo,
            device=device,
            dtype=torch.float32,
            **kwargs,
        ),
        error_type=ValueError,
        error_regex="is implemented for strided tensors only, got: torch.sparse_coo",
    )

    # 测试非浮点数数据类型的窗口张量，例如 torch.long
    yield ErrorInput(
        SampleInput(3, *args, dtype=torch.long, device=device, **kwargs),
        error_type=ValueError,
        error_regex="expects float32 or float64 dtypes, got: torch.int64",
    )

    # 测试 bfloat16 数据类型的窗口张量
    yield ErrorInput(
        SampleInput(3, *args, dtype=torch.bfloat16, device=device, **kwargs),
        error_type=ValueError,
        error_regex="expects float32 or float64 dtypes, got: torch.bfloat16",
    )

    # 测试 float16 数据类型的窗口张量
    yield ErrorInput(
        SampleInput(3, *args, dtype=torch.float16, device=device, **kwargs),
        error_type=ValueError,
        error_regex="expects float32 or float64 dtypes, got: torch.float16",
    )


# 指数窗口的错误输入
def error_inputs_exponential_window(op_info, device, **kwargs):
    # 调用窗口错误输入生成器生成常见的错误输入
    yield from error_inputs_window(op_info, device, **kwargs)

    # 测试负衰减值的情况
    # 使用 yield 返回一个 ErrorInput 对象，表示一个测试用例，检查输入参数 tau 为负数的情况
    yield ErrorInput(
        SampleInput(3, tau=-1, dtype=torch.float32, device=device, **kwargs),
        error_type=ValueError,  # 错误类型为 ValueError
        error_regex="Tau must be positive, got: -1 instead.",  # 错误消息的正则表达式
    )

    # 使用 yield 返回另一个 ErrorInput 对象，表示一个测试用例，检查对称窗口和给定中心值的情况
    yield ErrorInput(
        SampleInput(3, center=1, sym=True, dtype=torch.float32, device=device),
        error_type=ValueError,  # 错误类型为 ValueError
        error_regex="Center must be None for symmetric windows",  # 错误消息的正则表达式
    )
# 定义一个生成常见高斯窗口错误输入的函数
def error_inputs_gaussian_window(op_info, device, **kwargs):
    # 调用 error_inputs_window 函数生成常见窗口错误输入
    yield from error_inputs_window(op_info, device, std=0.5, **kwargs)

    # 测试负标准差的情况
    yield ErrorInput(
        SampleInput(3, std=-1, dtype=torch.float32, device=device, **kwargs),
        error_type=ValueError,
        error_regex="Standard deviation must be positive, got: -1 instead.",
    )


# 定义一个生成常见 Kaiser 窗口错误输入的函数
def error_inputs_kaiser_window(op_info, device, **kwargs):
    # 调用 error_inputs_window 函数生成常见窗口错误输入
    yield from error_inputs_window(op_info, device, beta=12, **kwargs)

    # 测试负 beta 值的情况
    yield ErrorInput(
        SampleInput(3, beta=-1, dtype=torch.float32, device=device, **kwargs),
        error_type=ValueError,
        error_regex="beta must be non-negative, got: -1 instead.",
    )


# 定义一个生成通用余弦窗口错误输入的函数
def error_inputs_general_cosine_window(op_info, device, **kwargs):
    # 调用 error_inputs_window 函数生成常见窗口错误输入，指定余弦窗口系数 a
    yield from error_inputs_window(op_info, device, a=[0.54, 0.46], **kwargs)

    # 测试系数 a 为 None 的情况
    yield ErrorInput(
        SampleInput(3, a=None, dtype=torch.float32, device=device, **kwargs),
        error_type=TypeError,
        error_regex="Coefficients must be a list/tuple",
    )

    # 测试系数 a 为空列表的情况
    yield ErrorInput(
        SampleInput(3, a=[], dtype=torch.float32, device=device, **kwargs),
        error_type=ValueError,
        error_regex="Coefficients cannot be empty",
    )


# 定义一个装饰 scipy 信号窗口函数的包装器
def reference_signal_window(fn: Callable):
    r"""Wrapper for scipy signal window references.

    Discards keyword arguments for window reference functions that don't have a matching signature with
    torch, e.g., gaussian window.
    """

    def _fn(
        *args,
        dtype=numpy.float64,
        device=None,
        layout=torch.strided,
        requires_grad=False,
        **kwargs,
    ):
        r"""The unused arguments are defined to disregard those values"""
        # 调用给定的窗口函数 fn，并将结果转换为指定的 dtype 类型
        return fn(*args, **kwargs).astype(dtype)

    return _fn


# 定义一个创建与不同窗口相关的 OpInfo 对象的辅助函数
def make_signal_windows_opinfo(
    name: str,
    ref: Callable,
    sample_inputs_func: Callable,
    reference_inputs_func: Callable,
    error_inputs_func: Callable,
    *,
    skips: Tuple[DecorateInfo, ...] = (),
):
    r"""Helper function to create OpInfo objects related to different windows."""
    # 返回一个 OpInfo 对象，其中包含以下信息：
    # - name: 操作的名称
    # - ref: 如果 TEST_SCIPY 为真，则包含 ref，否则为 None
    # - dtypes: 浮点类型的列表，通过 floating_types 函数获取
    # - dtypesIfCUDA: 浮点类型的列表，通过 floating_types 函数获取
    # - sample_inputs_func: 样本输入函数的引用
    # - reference_inputs_func: 参考输入函数的引用
    # - error_inputs_func: 错误输入函数的引用
    # - supports_out: 是否支持输出（此处为 False）
    # - supports_autograd: 是否支持自动求导（此处为 False）
    # - skips: 一个装饰信息（DecorateInfo）对象的元组，表示跳过的测试信息列表，包括以下项目：
    #   - unittest.expectedFailure 装饰的测试 "TestOperatorSignatures" 的 "test_get_torch_func_signature_exhaustive"
    #   - unittest.expectedFailure 装饰的测试 "TestJit" 的 "test_variant_consistency_jit"
    #   - unittest.skip 装饰的测试 "TestCommon" 的 "test_noncontiguous_samples"
    #   - unittest.skip 装饰的测试 "TestCommon" 的 "test_variant_consistency_eager"
    #   - unittest.skip 装饰的测试 "TestMathBits" 的 "test_conj_view"
    #   - unittest.skip 装饰的测试 "TestMathBits" 的 "test_neg_conj_view"
    #   - unittest.skip 装饰的测试 "TestMathBits" 的 "test_neg_view"
    #   - unittest.skip 装饰的测试 "TestVmapOperatorsOpInfo" 的 "test_vmap_exhaustive"
    #   - unittest.skip 装饰的测试 "TestVmapOperatorsOpInfo" 的 "test_op_has_batch_rule"
    #   - unittest.skip 装饰的测试 "TestCommon" 的 "test_numpy_ref_mps"，并包括从外部传入的 skips 元素
    return OpInfo(
        name=name,
        ref=ref if TEST_SCIPY else None,
        dtypes=floating_types(),
        dtypesIfCUDA=floating_types(),
        sample_inputs_func=sample_inputs_func,
        reference_inputs_func=reference_inputs_func,
        error_inputs_func=error_inputs_func,
        supports_out=False,
        supports_autograd=False,
        skips=(
            # TODO: same as this?
            # https://github.com/pytorch/pytorch/issues/81774
            # also see: arange, new_full
            # fails to match any schemas despite working in the interpreter
            DecorateInfo(
                unittest.expectedFailure,
                "TestOperatorSignatures",
                "test_get_torch_func_signature_exhaustive",
            ),
            # fails to match any schemas despite working in the interpreter
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # skip these tests since we have non tensor input
            DecorateInfo(
                unittest.skip("Skipped!"), "TestCommon", "test_noncontiguous_samples"
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestMathBits", "test_conj_view"),
            DecorateInfo(
                unittest.skip("Skipped!"), "TestMathBits", "test_neg_conj_view"
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestMathBits", "test_neg_view"),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestVmapOperatorsOpInfo",
                "test_vmap_exhaustive",
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestVmapOperatorsOpInfo",
                "test_op_has_batch_rule",
            ),
            DecorateInfo(
                unittest.skip("Buggy on MPS for now (mistakenly promotes to float64)"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
            *skips,
        ),
    )
op_db: List[OpInfo] = [
    # 创建一个名为 'signal.windows.hamming' 的操作信息对象
    make_signal_windows_opinfo(
        name="signal.windows.hamming",
        # 如果 TEST_SCIPY 为真，则设置参考对象为 scipy.signal.windows.hamming 的参考信号窗口函数，否则为 None
        ref=reference_signal_window(scipy.signal.windows.hamming)
        if TEST_SCIPY
        else None,
        # 设置样本输入函数为 sample_inputs_window
        sample_inputs_func=sample_inputs_window,
        # 设置参考输入函数为 reference_inputs_window
        reference_inputs_func=reference_inputs_window,
        # 设置错误输入函数为 error_inputs_window
        error_inputs_func=error_inputs_window,
    ),
    # 创建一个名为 'signal.windows.hann' 的操作信息对象
    make_signal_windows_opinfo(
        name="signal.windows.hann",
        # 如果 TEST_SCIPY 为真，则设置参考对象为 scipy.signal.windows.hann 的参考信号窗口函数，否则为 None
        ref=reference_signal_window(scipy.signal.windows.hann) if TEST_SCIPY else None,
        # 设置样本输入函数为 sample_inputs_window
        sample_inputs_func=sample_inputs_window,
        # 设置参考输入函数为 reference_inputs_window
        reference_inputs_func=reference_inputs_window,
        # 设置错误输入函数为 error_inputs_window
        error_inputs_func=error_inputs_window,
    ),
    # 创建一个名为 'signal.windows.bartlett' 的操作信息对象
    make_signal_windows_opinfo(
        name="signal.windows.bartlett",
        # 如果 TEST_SCIPY 为真，则设置参考对象为 scipy.signal.windows.bartlett 的参考信号窗口函数，否则为 None
        ref=reference_signal_window(scipy.signal.windows.bartlett)
        if TEST_SCIPY
        else None,
        # 设置样本输入函数为 sample_inputs_window
        sample_inputs_func=sample_inputs_window,
        # 设置参考输入函数为 reference_inputs_window
        reference_inputs_func=reference_inputs_window,
        # 设置错误输入函数为 error_inputs_window
        error_inputs_func=error_inputs_window,
    ),
    # 创建一个名为 'signal.windows.blackman' 的操作信息对象
    make_signal_windows_opinfo(
        name="signal.windows.blackman",
        # 如果 TEST_SCIPY 为真，则设置参考对象为 scipy.signal.windows.blackman 的参考信号窗口函数，否则为 None
        ref=reference_signal_window(scipy.signal.windows.blackman)
        if TEST_SCIPY
        else None,
        # 设置样本输入函数为 sample_inputs_window
        sample_inputs_func=sample_inputs_window,
        # 设置参考输入函数为 reference_inputs_window
        reference_inputs_func=reference_inputs_window,
        # 设置错误输入函数为 error_inputs_window
        error_inputs_func=error_inputs_window,
    ),
    # 创建一个名为 'signal.windows.cosine' 的操作信息对象
    make_signal_windows_opinfo(
        name="signal.windows.cosine",
        # 如果 TEST_SCIPY 为真，则设置参考对象为 scipy.signal.windows.cosine 的参考信号窗口函数，否则为 None
        ref=reference_signal_window(scipy.signal.windows.cosine)
        if TEST_SCIPY
        else None,
        # 设置样本输入函数为 sample_inputs_window
        sample_inputs_func=sample_inputs_window,
        # 设置参考输入函数为 reference_inputs_window
        reference_inputs_func=reference_inputs_window,
        # 设置错误输入函数为 error_inputs_window
        error_inputs_func=error_inputs_window,
    ),
    # 创建一个名为 'signal.windows.exponential' 的操作信息对象
    make_signal_windows_opinfo(
        name="signal.windows.exponential",
        # 如果 TEST_SCIPY 为真，则设置参考对象为 scipy.signal.windows.exponential 的参考信号窗口函数，否则为 None
        ref=reference_signal_window(scipy.signal.windows.exponential)
        if TEST_SCIPY
        else None,
        # 设置样本输入函数为 sample_inputs_window，同时设定参数 tau 为 2.78
        sample_inputs_func=partial(sample_inputs_window, tau=2.78),
        # 设置参考输入函数为 reference_inputs_exponential_window，同时设定参数 tau 为 2.78
        reference_inputs_func=partial(reference_inputs_exponential_window, tau=2.78),
        # 设置错误输入函数为 error_inputs_exponential_window
        error_inputs_func=error_inputs_exponential_window,
    ),
    # 创建一个名为 'signal.windows.gaussian' 的操作信息对象
    make_signal_windows_opinfo(
        name="signal.windows.gaussian",
        # 如果 TEST_SCIPY 为真，则设置参考对象为 scipy.signal.windows.gaussian 的参考信号窗口函数，否则为 None
        ref=reference_signal_window(scipy.signal.windows.gaussian)
        if TEST_SCIPY
        else None,
        # 设置样本输入函数为 sample_inputs_window，同时设定参数 std 为 1.92
        sample_inputs_func=partial(sample_inputs_window, std=1.92),
        # 设置参考输入函数为 reference_inputs_gaussian_window，同时设定参数 std 为 1.92
        reference_inputs_func=partial(reference_inputs_gaussian_window, std=1.92),
        # 设置错误输入函数为 error_inputs_gaussian_window
        error_inputs_func=error_inputs_gaussian_window,
        # 设定跳过装饰信息，包括一个跳过装饰器对象和相关信息
        skips=(
            DecorateInfo(
                unittest.skip("Buggy on MPS for now (mistakenly promotes to float64)"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
        ),
    ),
]
    make_signal_windows_opinfo(
        name="signal.windows.kaiser",
        # 如果 TEST_SCIPY 为真，则使用 scipy.signal.windows.kaiser 函数作为参考函数
        ref=reference_signal_window(scipy.signal.windows.kaiser)
        if TEST_SCIPY
        else None,
        # 使用 partial 函数生成带有特定参数的函数，用于生成示例输入
        sample_inputs_func=partial(sample_inputs_window, beta=12.0),
        # 使用 partial 函数生成带有特定参数的函数，用于生成参考输入
        reference_inputs_func=partial(reference_inputs_kaiser_window, beta=12.0),
        # 指定用于生成错误输入的函数
        error_inputs_func=error_inputs_kaiser_window,
    ),
    make_signal_windows_opinfo(
        name="signal.windows.general_cosine",
        # 如果 TEST_SCIPY 为真，则使用 scipy.signal.windows.general_cosine 函数作为参考函数
        ref=reference_signal_window(scipy.signal.windows.general_cosine)
        if TEST_SCIPY
        else None,
        # 使用 partial 函数生成带有特定参数的函数，用于生成示例输入
        sample_inputs_func=partial(sample_inputs_window, a=[0.54, 0.46]),
        # 使用 partial 函数生成带有特定参数的函数，用于生成参考输入
        reference_inputs_func=partial(
            reference_inputs_general_cosine_window, a=[0.54, 0.46]
        ),
        # 指定用于生成错误输入的函数
        error_inputs_func=error_inputs_general_cosine_window,
    ),
    make_signal_windows_opinfo(
        name="signal.windows.general_hamming",
        # 如果 TEST_SCIPY 为真，则使用 scipy.signal.windows.general_hamming 函数作为参考函数
        ref=reference_signal_window(scipy.signal.windows.general_hamming)
        if TEST_SCIPY
        else None,
        # 使用 partial 函数生成带有特定参数的函数，用于生成示例输入
        sample_inputs_func=partial(sample_inputs_window, alpha=0.54),
        # 使用 partial 函数生成带有特定参数的函数，用于生成参考输入
        reference_inputs_func=partial(
            reference_inputs_general_hamming_window, alpha=0.54
        ),
        # 指定用于生成错误输入的函数
        error_inputs_func=error_inputs_window,
    ),
    make_signal_windows_opinfo(
        name="signal.windows.nuttall",
        # 如果 TEST_SCIPY 为真，则使用 scipy.signal.windows.nuttall 函数作为参考函数
        ref=reference_signal_window(scipy.signal.windows.nuttall)
        if TEST_SCIPY
        else None,
        # 指定用于生成示例输入的函数
        sample_inputs_func=sample_inputs_window,
        # 指定用于生成参考输入的函数
        reference_inputs_func=reference_inputs_window,
        # 指定用于生成错误输入的函数
        error_inputs_func=error_inputs_window,
    ),
]
```