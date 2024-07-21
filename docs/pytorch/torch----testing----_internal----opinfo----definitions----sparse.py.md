# `.\pytorch\torch\testing\_internal\opinfo\definitions\sparse.py`

```
# 忽略类型检查错误，通常用于类型检查工具mypy
# Ignore type-checking errors, typically used for type-checking tool mypy
mypy: ignore-errors

# 导入操作系统相关功能
# Import OS related functionalities
import os

# 导入PyTorch库
# Import PyTorch library
import torch

# 从torch.testing中导入make_tensor函数，用于生成张量
# Import make_tensor function from torch.testing for tensor creation
from torch.testing import make_tensor  # noqa: F401

# 从torch.testing._internal.opinfo.core中导入以下函数和类，用于操作信息的核心功能
# Import functions and classes from torch.testing._internal.opinfo.core for core operation information
from torch.testing._internal.opinfo.core import (
    BinaryUfuncInfo,
    ErrorInput,
    generate_elementwise_binary_tensors,
    ReductionOpInfo,
    sample_inputs_reduction,
    SampleInput,
)


def _check_validate(op_info, sample):
    # 内部函数，用于检查操作信息和样本是否匹配，检查是否抛出预期的异常
    # Internal function to validate operation information against sample,
    # checking if the expected exception is raised
    def _check_fail(sample):
        try:
            op_info(
                sample.sample_input.input,
                *sample.sample_input.args,
                **sample.sample_input.kwargs,
            )
        except sample.error_type:
            pass
        except Exception as msg:
            raise AssertionError(  # noqa: B904
                f"{op_info.name} on {sample.sample_input=} expected exception "
                f"{sample.error_type}: {sample.error_regex}, got {type(msg).__name__}: {msg}"
            )
        else:
            raise AssertionError(
                f"{op_info.name} on {sample.sample_input=} expected exception "
                f"{sample.error_type}: {sample.error_regex}, got none."
            )

    # 内部函数，用于检查操作信息和样本是否匹配，检查是否成功执行
    # Internal function to validate operation information against sample,
    # checking if execution succeeds
    def _check_success(sample):
        try:
            op_info(sample.input, *sample.args, **sample.kwargs)
        except Exception as msg:
            raise AssertionError(  # noqa: B904
                f"{op_info.name} on {sample=} expected to succeed "
                f", got {type(msg).__name__}: {msg}"
            )

    # 如果样本是ErrorInput类型，则执行检查失败函数；否则执行检查成功函数
    # If sample is of ErrorInput type, execute _check_fail function; otherwise, execute _check_success function
    if isinstance(sample, ErrorInput):
        _check_fail(sample)
    else:
        _check_success(sample)


def _sample_inputs_sparse(
    sample_inputs,
    maybe_failing_sample_inputs,
    validate_sample_input,
    op_info,
    *args,
    **kwargs,
):
    # 检查环境变量PYTORCH_TEST_CHECK_VALIDATE_SPARSE_SAMPLES是否设置为"1"，决定是否进行检查验证
    # Check if environment variable PYTORCH_TEST_CHECK_VALIDATE_SPARSE_SAMPLES is set to "1" to decide validation check
    check_validate = (
        os.environ.get("PYTORCH_TEST_CHECK_VALIDATE_SPARSE_SAMPLES", "0") == "1"
    )

    # 遍历sample_inputs生成器返回的样本，对每个样本进行验证和可能的处理
    # Iterate over samples returned by sample_inputs generator, validate and handle each sample
    for sample in sample_inputs(op_info, *args, **kwargs):
        # 调用validate_sample_input函数验证样本
        # Call validate_sample_input function to validate the sample
        sample = validate_sample_input(op_info, sample, check_validate=check_validate)
        # 如果样本是SampleInput类型，则产生该样本
        # If sample is of SampleInput type, yield the sample
        if isinstance(sample, SampleInput):
            yield sample
        # 错误输入在_error_inputs_sparse函数中处理

    # 遍历maybe_failing_sample_inputs生成器返回的样本，对每个样本进行验证和可能的处理
    # Iterate over samples returned by maybe_failing_sample_inputs generator, validate and handle each sample
    for sample in maybe_failing_sample_inputs(op_info, *args, **kwargs):
        # 调用validate_sample_input函数验证样本
        # Call validate_sample_input function to validate the sample
        sample = validate_sample_input(op_info, sample, check_validate=check_validate)
        # 如果样本是SampleInput类型，则产生该样本
        # If sample is of SampleInput type, yield the sample
        if isinstance(sample, SampleInput):
            yield sample


def _error_inputs_sparse(
    maybe_failing_sample_inputs, validate_sample_input, op_info, *args, **kwargs
):
    # 检查环境变量PYTORCH_TEST_CHECK_VALIDATE_SPARSE_SAMPLES是否设置为"1"，决定是否进行检查验证
    # Check if environment variable PYTORCH_TEST_CHECK_VALIDATE_SPARSE_SAMPLES is set to "1" to decide validation check
    check_validate = (
        os.environ.get("PYTORCH_TEST_CHECK_VALIDATE_SPARSE_SAMPLES", "0") == "1"
    )

    # 遍历maybe_failing_sample_inputs生成器返回的样本，对每个样本进行验证和可能的处理
    # Iterate over samples returned by maybe_failing_sample_inputs generator, validate and handle each sample
    for sample in maybe_failing_sample_inputs(op_info, *args, **kwargs):
        # 调用validate_sample_input函数验证样本
        # Call validate_sample_input function to validate the sample
        sample = validate_sample_input(op_info, sample, check_validate=check_validate)
        # 如果样本是ErrorInput类型，则产生该样本
        # If sample is of ErrorInput type, yield the sample
        if isinstance(sample, ErrorInput):
            yield sample
        # Sample inputs are handled in sample_inputs_sparse


def _apply_requires_grad_to_samples(sample_inputs):
    """Decorator to _maybe_failing_sample_inputs_... generator functions
    that clones and sets requires_grad argument to tensors in sample
    """
    # 用于修饰_maybe_failing_sample_inputs_...生成器函数的装饰器
    # 该装饰器克隆样本中的张量并设置requires_grad参数
    # Decorator for _maybe_failing_sample_inputs_... generator functions
    # that clones and sets requires_grad argument to tensors in sample
    """
    Generate and yield sample inputs for a given operation.

    This function wraps around the `sample_inputs` function and applies
    certain transformations based on input arguments. This is useful
    when handling tensor instances that need to share the same tensors.

    Args:
        op_info (object): Information about the operation.
        device (torch.device): The device on which tensors will be allocated.
        dtype (torch.dtype): Data type for the tensors.
        requires_grad (bool): Whether tensors require gradient computation.
        layout (torch.layout): Layout descriptor for tensors.
        **kwargs: Additional keyword arguments for custom configurations.

    Returns:
        function: A generator function (`wrapper`) that yields transformed
                  sample inputs based on the provided arguments.
    """

    def wrapper(op_info, device, dtype, requires_grad, layout, **kwargs):
        """
        Inner function (`wrapper`) that generates sample inputs and applies
        `apply_requires_grad` transformation based on the `requires_grad` flag.

        Args:
            op_info (object): Information about the operation.
            device (torch.device): The device on which tensors will be allocated.
            dtype (torch.dtype): Data type for the tensors.
            requires_grad (bool): Whether tensors require gradient computation.
            layout (torch.layout): Layout descriptor for tensors.
            **kwargs: Additional keyword arguments for custom configurations.

        Yields:
            torch.Tensor: Transformed sample inputs based on `apply_requires_grad`.

        """
        def apply_requires_grad(x):
            """
            Applies `requires_grad` condition to a given tensor `x`.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Transformed tensor based on `requires_grad`.

            """
            if (
                not isinstance(x, torch.Tensor)
                or x.requires_grad
                or not requires_grad
                or not (x.is_floating_point() or x.is_complex())
            ):
                return x
            return x.detach().clone().requires_grad_(requires_grad)

        if requires_grad:
            # Yield transformed sample inputs
            for sample_input in sample_inputs(
                op_info, device, dtype, requires_grad, layout, **kwargs
            ):
                yield sample_input.transform(apply_requires_grad)
        else:
            # Yield sample inputs without any transformation
            yield from sample_inputs(
                op_info, device, dtype, requires_grad, layout, **kwargs
            )

    return wrapper
def sample_inputs_sparse_reduction(
    op_info, device, dtype, requires_grad, layout, blocksize=None, **kwargs
):
    """生成稀疏张量的归约操作的样本输入。"""
    # 从布局中提取布局名称，并确保正确格式化以匹配支持的操作
    layout_name = str(layout).split(".", 1)[-1].rsplit("_coo", 1)[0]
    op_supports_layout = getattr(op_info, "supports_" + layout_name)
    if not op_supports_layout:
        # 如果操作不支持给定的布局类型，则直接返回
        return

    # 遍历归约操作的样本输入集合
    for sample_input in sample_inputs_reduction(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        if sample_input.input.ndim == 0:
            # 不支持标量稀疏张量，跳过处理
            continue

        if layout in {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            if sample_input.input.ndim < 2:
                # 需要至少二维张量才能转换为稀疏压缩张量
                continue
            if sample_input.input.ndim > 2 and (sample_input.input == 0).any():
                # 跳过批处理的稀疏压缩样本，其中包含显式的零值，因为to_sparse(layout=..)会失败
                # 详见 gh-98495。TODO: 在 gh-98495 修复后删除此 if 块。
                continue

        if layout in {torch.sparse_bsr, torch.sparse_bsc} and blocksize is None:
            # 对于块大小未指定的块稀疏格式，设置默认块大小为 (1, 1)
            blocksize = (1, 1)

        # 生成并返回处理后的稀疏张量样本输入
        yield SampleInput(
            sample_input.input.detach()
            .to_sparse(layout=layout, blocksize=blocksize)
            .requires_grad_(requires_grad),
            args=sample_input.args,
            kwargs=sample_input.kwargs,
        )

        if layout is torch.sparse_coo and (dtype.is_floating_point or dtype.is_complex):
            # 非压缩的稀疏样本
            inp = sample_input.input.detach().to_sparse(layout=layout)
            inp = torch.sparse_coo_tensor(
                inp.indices().repeat(1, 2),
                inp.values().repeat(2),
                inp.shape,
                dtype=inp.dtype,
                device=inp.device,
            )
            assert not inp.is_coalesced()
            # 生成并返回处理后的非压缩稀疏张量样本输入
            yield SampleInput(
                inp.requires_grad_(requires_grad),
                args=sample_input.args,
                kwargs=sample_input.kwargs,
            )

        if sample_input.input.ndim > 2:
            # 混合样本
            yield SampleInput(
                sample_input.input.detach()
                .to_sparse(
                    layout=layout,
                    blocksize=blocksize,
                    dense_dim=sample_input.input.ndim - 2,
                )
                .requires_grad_(requires_grad),
                args=sample_input.args,
                kwargs=sample_input.kwargs,
            )


def _validate_sample_input_sparse_reduction(op_info, sample, check_validate=False):
    """在样本有效且被支持时返回指定的样本。"""
    # 函数未完整提供，没有给出具体实现，因此不添加更多的代码注释。
    # 创建一个特殊的未指定对象，用于后续条件判断中的比较
    UNSPECIFIED = object()
    
    # 如果操作的名称是 "sum"，则调用特定的函数验证输入的稀疏数据样本
    if op_info.name == "sum":
        sample = _validate_sample_input_sparse_reduction_sum(sample)
    
    # 如果操作的名称是 "masked.sum" 中的任何一个，则进行一系列条件检查和错误处理
    if op_info.name in {"masked.sum"}:
        # 从样本的关键字参数中获取 "mask"，如果未指定则使用 UNSPECIFIED
        mask = sample.kwargs.get("mask", UNSPECIFIED)
        
        # 如果 mask 不是 None 或 UNSPECIFIED，并且满足以下条件之一：
        # - mask 的维度大于2
        # - mask 的布局是 torch.strided
        # - mask 中有任何值为 0
        # 则将 sample 转换为 ErrorInput 实例，提示特定的错误信息
        if (
            mask not in {None, UNSPECIFIED}
            and mask.ndim > 2
            and mask.layout is torch.strided
            and (mask == 0).any()
        ):
            # TODO: 在修复 gh-98495 后移除此 if 块
            sample = ErrorInput(
                sample,
                error_regex="Expect the same number of specified elements per batch.",
            )
        
        # 如果未设置保持维度参数 keepdim，则将 sample 转换为 ErrorInput 实例，提示错误信息
        elif not sample.kwargs.get("keepdim"):
            sample = ErrorInput(
                sample,
                error_type=(AssertionError, RuntimeError),
                error_regex="reduction operations on (CSR|CSC) tensors with keepdim=False is unsupported",
            )
        
        # 如果 mask 是 UNSPECIFIED，则将 sample 转换为 ErrorInput 实例，提示错误信息
        elif mask is UNSPECIFIED:
            sample = ErrorInput(
                sample,
                error_type=ValueError,
                error_regex="masked (.*) expects explicit mask for sparse_csr tensor input",
            )
        
        # 如果样本的输入维度大于2，则将 sample 转换为 ErrorInput 实例，提示错误信息
        elif sample.input.ndim > 2:
            sample = ErrorInput(
                sample,
                error_regex="crow_indices is supposed to be a vector, but got 3 dimensional tensor.",
            )
    
    # 如果需要验证，则调用 _check_validate 函数验证操作信息和样本
    if check_validate:
        _check_validate(op_info, sample)
    
    # 返回处理后的样本，可能包含错误信息
    return sample
# 定义一个函数用于验证稀疏张量的减少和求和操作的输入样本的有效性
def _validate_sample_input_sparse_reduction_sum(sample, check_validate=False):
    # 注意: 当修复一个失败的样本案例时，移除对应的 if 块
    t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
    # 从关键字参数中获取维度信息
    dim = t_kwargs.get("dim")
    # 从关键字参数中获取 keepdim 是否保持维度的标志
    keepdim = t_kwargs.get("keepdim")
    # 获取输入张量的布局信息
    layout = t_inp.layout

    # 如果 dim 是整数、列表或元组类型之一
    if isinstance(dim, (int, list, tuple)):
        # 如果布局是 CSR、CSC、BSR 或 BSC 中的一种
        if layout in {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            # 如果布局是 CSC、BSR 或 BSC 中的一种，报告错误
            if layout in {torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}:
                return ErrorInput(
                    sample,
                    error_regex=(
                        "Currently the only compressed sparse format supported for sum.dim_IntList is CSR, but got layout"
                    ),
                )
            # 如果布局是 CSR 或 CSC 且 keepdim=False，报告错误
            if layout in {torch.sparse_csr, torch.sparse_csc} and not keepdim:
                return ErrorInput(
                    sample,
                    error_regex=(
                        "reduction operations on CSR tensors with keepdim=False is unsupported"
                    ),
                )
            # 如果输入张量的维度不等于 2，报告错误
            if t_inp.dim() != 2:
                return ErrorInput(
                    sample,
                    error_regex=("input_dim == 2 INTERNAL ASSERT"),
                )
            # 如果布局是 CSR，根据数据类型报告不支持的错误
            if layout == torch.sparse_csr:
                if t_inp.dtype == torch.bool:
                    return ErrorInput(
                        sample,
                        error_regex=("_sparse_csr_sum_cpu not implemented for 'Bool'"),
                    )
                if t_inp.dtype == torch.complex32:
                    return ErrorInput(
                        sample,
                        error_regex=(
                            "_sparse_csr_sum_cuda not implemented for 'ComplexHalf'"
                        ),
                    )

    # 如果没有任何错误，返回样本本身
    return sample


def _maybe_failing_sample_inputs_sparse_reduction_sum(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    """Generator of samples that are known to fail or that were failing in past."""
    # 注意: 当修复一个失败的案例时，移除异常注释，但保留 `yield sample` 语句。
    # 如果布局是 CSR 或 CSC
    if layout in [
        torch.sparse_csr,
        torch.sparse_csc,
        # 继续添加其它稀疏张量布局类型
        # ...
    ]:
    ]:
        # NotImplementedError: Could not run 'aten::sum.IntList_out' with arguments from the 'SparseCsrCPU' backend.
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)  # 创建一个二维张量
            .to_sparse(layout=layout)  # 将张量转换为稀疏张量，使用给定的布局
            .requires_grad_(requires_grad),  # 标记张量需要梯度计算
            kwargs=dict(dim=0, keepdim=True),  # 额外的参数字典，指定维度和保持维度信息
        )
        yield SampleInput(
            torch.tensor([[[0, 1]], [[2, 3]]], dtype=dtype)  # 创建一个三维张量
            .to_sparse(layout=layout, dense_dim=1)  # 将张量转换为稀疏张量，指定稠密维度为1
            .requires_grad_(requires_grad),  # 标记张量需要梯度计算
            kwargs=dict(dim=0),  # 额外的参数字典，指定维度
        )
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)  # 创建一个二维张量
            .to_sparse(layout=layout)  # 将张量转换为稀疏张量，使用给定的布局
            .requires_grad_(requires_grad),  # 标记张量需要梯度计算
            kwargs=dict(dim=(0,)),  # 额外的参数字典，指定维度为元组
        )
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)  # 创建一个二维张量
            .to_sparse(layout=layout)  # 将张量转换为稀疏张量，使用给定的布局
            .requires_grad_(requires_grad),  # 标记张量需要梯度计算
            kwargs=dict(dim=(0,), keepdim=True),  # 额外的参数字典，指定维度为元组和保持维度信息
        )
        yield SampleInput(
            torch.tensor([[[0, 1]], [[2, 3]]], dtype=dtype)  # 创建一个三维张量
            .to_sparse(layout=layout, dense_dim=1)  # 将张量转换为稀疏张量，指定稠密维度为1
            .requires_grad_(requires_grad),  # 标记张量需要梯度计算
            kwargs=dict(dim=(0,)),  # 额外的参数字典，指定维度为元组
        )

        # RuntimeError: torch.empty: Only batched sparse compressed (non-block) tensors are supported, but got size [2]
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype)  # 创建一个二维张量
            .to_sparse(layout=layout)  # 将张量转换为稀疏张量，使用给定的布局
            .requires_grad_(requires_grad),  # 标记张量需要梯度计算
            kwargs=dict(dim=0),  # 额外的参数字典，指定维度
        )

    if layout in [
        torch.sparse_bsr,  # 检查给定的布局是否在支持的稀疏布局中
        torch.sparse_bsc,
    # 生成稀疏张量样本输入，用于测试
    yield SampleInput(
        # 创建二维稀疏张量，数据类型为给定的dtype
        torch.tensor([[0, 1], [2, 3]], dtype=dtype)
        # 转换为稀疏张量，指定布局为layout，块大小为(2, 2)
        .to_sparse(layout=layout, blocksize=(2, 2))
        # 设置是否需要梯度
        .requires_grad_(requires_grad),
        # 附加的关键字参数，维度为0，保持维度为True
        kwargs=dict(dim=0, keepdim=True),
    )
    
    yield SampleInput(
        # 创建三维稀疏张量，数据类型为给定的dtype
        torch.tensor([[[0, 1]], [[2, 3]]], dtype=dtype)
        # 转换为稀疏张量，指定布局为layout，稠密维度为1，块大小为(1, 1)
        .to_sparse(layout=layout, dense_dim=1, blocksize=(1, 1))
        # 设置是否需要梯度
        .requires_grad_(requires_grad),
        # 附加的关键字参数，维度为0
        kwargs=dict(dim=0),
    )
    
    yield SampleInput(
        # 创建二维稀疏张量，数据类型为给定的dtype
        torch.tensor([[0, 1], [2, 3]], dtype=dtype)
        # 转换为稀疏张量，指定布局为layout，块大小为(1, 1)
        .to_sparse(layout=layout, blocksize=(1, 1))
        # 设置是否需要梯度
        .requires_grad_(requires_grad),
        # 附加的关键字参数，维度为(0,)
        kwargs=dict(dim=(0,)),
    )
    
    yield SampleInput(
        # 创建二维稀疏张量，数据类型为给定的dtype
        torch.tensor([[0, 1], [2, 3]], dtype=dtype)
        # 转换为稀疏张量，指定布局为layout，块大小为(1, 1)
        .to_sparse(layout=layout, blocksize=(1, 1))
        # 设置是否需要梯度
        .requires_grad_(requires_grad),
        # 附加的关键字参数，维度为(0,)，保持维度为True
        kwargs=dict(dim=(0,), keepdim=True),
    )
    
    yield SampleInput(
        # 创建三维稀疏张量，数据类型为给定的dtype
        torch.tensor([[[0, 1]], [[2, 3]]], dtype=dtype)
        # 转换为稀疏张量，指定布局为layout，块大小为(1, 1)，稠密维度为1
        .to_sparse(layout=layout, blocksize=(1, 1), dense_dim=1)
        # 设置是否需要梯度
        .requires_grad_(requires_grad),
        # 附加的关键字参数，维度为(0,)
        kwargs=dict(dim=(0,)),
    )
    
    # RuntimeError: torch.empty: Only batched sparse compressed (non-block) tensors are supported, but got size [2]
    yield SampleInput(
        # 创建二维稀疏张量，数据类型为给定的dtype
        torch.tensor([[0, 1], [2, 3]], dtype=dtype)
        # 转换为稀疏张量，指定布局为layout，块大小为(1, 1)
        .to_sparse(layout=layout, blocksize=(1, 1))
        # 设置是否需要梯度
        .requires_grad_(requires_grad),
        # 附加的关键字参数，维度为0
        kwargs=dict(dim=0),
    )
# 生成稀疏张量求和操作的示例输入
def sample_inputs_sparse_reduction_sum(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    """Sample inputs for sum on sparse tensors."""
    # 使用 _sample_inputs_sparse 函数生成稀疏张量的示例输入
    yield from _sample_inputs_sparse(
        sample_inputs_sparse_reduction,
        _maybe_failing_sample_inputs_sparse_reduction_sum,
        _validate_sample_input_sparse_reduction,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


# 生成稀疏张量求和操作的错误输入
def error_inputs_sparse_reduction_sum(op_info, device, layout, **kwargs):
    """Error inputs for sum on sparse tensors."""
    # 设置数据类型为 torch.float64 和不需要梯度
    dtype = torch.float64
    requires_grad = False
    # 使用 _error_inputs_sparse 函数生成稀疏张量求和操作的错误输入
    yield from _error_inputs_sparse(
        _maybe_failing_sample_inputs_sparse_reduction_sum,
        _validate_sample_input_sparse_reduction,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


# 生成稀疏张量元素级二元操作的示例输入
def sample_inputs_sparse_elementwise_binary_operation(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    """Sample inputs for elementwise binary operations on sparse tensors.

    The samples include regular, zero-sized, batched, and hybrid
    sparse tensors as well as rhs scalars. All tensors are full tensors.
    """

    # 将输入张量转换为稀疏张量的函数
    def _to_sparse(tensor, **kwargs):
        return tensor.detach().to_sparse(**kwargs).requires_grad_(requires_grad)

    # 使用 generate_elementwise_binary_tensors 函数生成元素级二元操作的张量示例输入
    for sample_input in generate_elementwise_binary_tensors(
        op_info,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        exclude_zero=True,
        **kwargs,
    ):
        yield _to_sparse(sample_input, layout=layout)
    ):
        # 解构 sample_input，获取输入数据和第一个参数
        lhs, rhs = sample_input.input, sample_input.args[0]
        # 最小稠密维度设为0
        min_dense_dim = 0
        # 最大稠密维度为 lhs 的维度数减1
        max_dense_dim = lhs.ndim - 1
        # 如果 layout 是稀疏格式之一，则进入条件
        if layout in {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            # 如果 lhs 的维度小于2，则稀疏压缩张量的稠密维度必须为2，跳过当前循环
            if lhs.ndim < 2:
                continue
            # 更新最大稠密维度为 lhs 的维度数减2
            max_dense_dim = lhs.ndim - 2

        # 遍历从 min_dense_dim 到 max_dense_dim 的稠密维度
        for dense_dim in range(min_dense_dim, max_dense_dim + 1):
            # 如果 layout 是 torch.sparse_bsr 或 torch.sparse_bsc，则设置块大小列表
            if layout in {torch.sparse_bsr, torch.sparse_bsc}:
                # 初始化块大小列表，初始添加 (1, 1)
                blocksizes = [(1, 1)]
                # 如果 lhs 元素数大于0，则添加实际的块大小
                if lhs.numel() > 0:
                    blocksizes.append(
                        (
                            lhs.shape[lhs.ndim - 2 - dense_dim],
                            lhs.shape[lhs.ndim - 1 - dense_dim],
                        )
                    )
            else:
                # 否则，块大小设为 [None]
                blocksizes = [None]
            
            # 遍历块大小列表
            for blocksize in blocksizes:
                # 构造转换为稀疏张量的参数字典
                to_sparse_kwargs = dict(
                    layout=layout, dense_dim=dense_dim, blocksize=blocksize
                )
                # 将 lhs 转换为稀疏张量
                lhs_sparse = _to_sparse(lhs, **to_sparse_kwargs)
                # 将 rhs 转换为稀疏张量
                rhs_sparse = _to_sparse(rhs, **to_sparse_kwargs)
                # 生成操作 (sparse, sparse) 的样本输入
                yield SampleInput(
                    lhs_sparse,
                    args=(rhs_sparse, *sample_input.args[1:]),
                    kwargs=sample_input.kwargs,
                )
                # 生成操作 (sparse, scalar) 的样本输入
                yield SampleInput(
                    lhs_sparse,
                    args=(
                        make_tensor(
                            (), dtype=dtype, device=device, requires_grad=requires_grad
                        ),
                        *sample_input.args[1:],
                    ),
                    kwargs=sample_input.kwargs,
                )
def _validate_sample_input_elementwise_binary_sparse_mul(sample):
    # 当修复失败的样本案例时，删除相应的 if 块
    t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
    # 计算批处理维度，排除稠密维度和稀疏维度后的维度数
    batch_dim = t_inp.dim() - t_inp.dense_dim() - t_inp.sparse_dim()
    # 获取输入张量的布局信息
    layout = t_inp.layout
    # 获取输入张量的数据类型
    dtype = t_inp.dtype

    # 检查条件：布局为 torch.sparse_csr，批处理维度大于0，并且第一个参数的维度大于0
    if layout is torch.sparse_csr and batch_dim > 0 and t_args[0].ndim > 0:
        return ErrorInput(
            sample,
            error_regex=(
                "coo_to_sparse_csr: conversion from Sparse to SparseCsr for input"
                " tensors with sparse_dim[(][)]!=2 is not supported"
            ),
        )
    # 检查条件：布局为 torch.sparse_csc，并且第一个参数的维度大于0
    elif layout is torch.sparse_csc and t_args[0].ndim > 0:
        return ErrorInput(
            sample, error_regex="Expected result Tensor to be of format CSR"
        )
    # 检查条件：布局为 torch.sparse_bsr，并且第一个参数的维度大于0
    elif layout is torch.sparse_bsr and t_args[0].ndim > 0:
        return ErrorInput(
            sample,
            error_regex="empty_sparse_compressed expected sparse compressed [(]non-block[)] tensor layout but got SparseBsr",
        )
    # 检查条件：布局为 torch.sparse_bsc，并且第一个参数的维度大于0
    elif layout is torch.sparse_bsc and t_args[0].ndim > 0:
        return ErrorInput(
            sample,
            error_regex="empty_sparse_compressed expected sparse compressed [(]non-block[)] tensor layout but got SparseBsc",
        )
    # 检查条件：布局为 torch.sparse_coo，数据类型为 torch.bool，并且第一个参数的维度大于0，且输入张量在CPU上，并且元素数大于0，并且稠密维度大于0
    elif (
        layout is torch.sparse_coo
        and dtype is torch.bool
        and t_args[0].ndim > 0
        and t_inp.is_cpu
        and t_inp.numel() > 0
        and t_inp.dense_dim() > 0
    ):
        return ErrorInput(
            sample, error_regex="\"addcmul_cpu_out\" not implemented for 'Bool'"
        )
    # 检查条件：布局为 torch.sparse_coo 或 torch.sparse_csr，数据类型为 torch.bool，并且输入张量的非零元素数大于0，并且第一个参数的维度大于0，且输入张量在CPU上，并且元素数大于0
    elif (
        layout in {torch.sparse_coo, torch.sparse_csr}
        and dtype is torch.bool
        and t_inp._nnz() > 0
        and t_args[0].ndim > 0
        and t_inp.is_cpu
        and t_inp.numel() > 0
    ):
        return ErrorInput(
            sample, error_regex="\"mul_out_sparse\" not implemented for 'Bool'"
        )
    # 检查条件：布局为 torch.sparse_csr，并且第一个参数的布局为 torch.strided，且第一个参数的维度在输入张量的维度之间
    elif (
        layout is torch.sparse_csr
        and t_args[0].layout is torch.strided
        and 0 < t_args[0].ndim
        and t_args[0].ndim < t_inp.ndim
    ):
        return ErrorInput(
            sample, error_regex="sparse_mask_sparse_csr expects self to be 2D"
        )
    # 检查条件：布局为 torch.sparse_csr，并且输入张量的稠密维度大于0，并且输入张量的非零元素数大于0，并且输入张量在CPU上，并且数据类型为 torch.float16，并且第一个参数的维度大于0
    elif (
        layout is torch.sparse_csr
        and t_inp.dense_dim() > 0
        and t_inp._nnz() > 0
        and t_inp.is_cpu
        and dtype is torch.float16
        and t_args[0].ndim > 0
    ):
        # 返回错误输入，指定错误正则表达式
        return ErrorInput(
            sample,
            error_regex=(
                "expects sparse inputs with equal dimensionality, number of sparse dimensions,"
                " and shape of sparse dimensions"
            ),
        )
    ):
        # 如果样本数据中包含 'Half' 类型数据，返回一个错误输入对象
        return ErrorInput(
            sample, error_regex="\"addcmul_cpu_out\" not implemented for 'Half'"
        )
    # 返回处理后的样本数据
    return sample
# 应用装饰器，将函数标记为适用于样本的需要梯度的操作
@_apply_requires_grad_to_samples
def _maybe_failing_sample_inputs_sparse_elementwise_binary_mul(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    """Generator of samples that are known to fail or that were failing in past."""
    # 当修复一个失败的情况时，删除异常注释，但保留 `yield sample` 语句。

    # 根据布局选择合适的块大小
    blocksize = (1, 1) if layout in {torch.sparse_bsr, torch.sparse_bsc} else None
    # 创建稀疏张量 `regular`
    regular = torch.tensor([[1, 2], [3, 4]], device=device, dtype=dtype).to_sparse(
        layout=layout, dense_dim=0, blocksize=blocksize
    )
    # 创建稀疏张量 `batch`
    batch = torch.tensor(
        [[[1, 2], [3, 4]], [[4, 5], [6, 7]]], device=device, dtype=dtype
    ).to_sparse(layout=layout, dense_dim=0, blocksize=blocksize
    )
    # 创建稀疏张量 `hybrid`
    hybrid = torch.tensor(
        [[[1], [2]], [[3], [4]]], device=device, dtype=dtype
    ).to_sparse(layout=layout, dense_dim=1, blocksize=blocksize)

    # 处理布局为 `torch.sparse_csr` 的情况
    if layout is torch.sparse_csr:
        # RuntimeError: crow_indices is supposed to be a vector, but got 2 dimensional tensor
        yield SampleInput(batch, args=(batch,))
        # RuntimeError: Only tensors with two sparse dimensions can be
        # converted to the SparseCsr layout, got self with 3 sparse
        # dimensions.
        yield SampleInput(
            torch.zeros_like(hybrid).requires_grad_(requires_grad),
            args=(torch.zeros_like(hybrid).requires_grad_(requires_grad),),
        )
        # 处理数据类型为 `torch.complex32` 的情况
        if dtype is torch.complex32:
            # RuntimeError: "mul_out_sparse" not implemented for 'ComplexHalf'
            yield SampleInput(regular, args=(regular,))
        # 处理数据类型为 `torch.bool` 且 `regular` 在 CPU 上的情况
        if dtype is torch.bool and regular.is_cpu:
            # RuntimeError: "mul_out_sparse" not implemented for 'Bool'
            yield SampleInput(regular, args=(regular,))
    
    # 处理布局为 `torch.sparse_csc` 的情况
    if layout is torch.sparse_csc:
        # RuntimeError: Expected result Tensor to be of format CSR
        yield SampleInput(regular, args=(regular,))
    
    # 处理布局为 `torch.sparse_bsr` 的情况
    if layout is torch.sparse_bsr:
        # RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor layout but got SparseBsr
        yield SampleInput(regular, args=(regular,))
    
    # 处理布局为 `torch.sparse_bsc` 的情况
    if layout is torch.sparse_bsc:
        # RuntimeError: empty_sparse_compressed expected sparse compressed (non-block) tensor layout but got SparseBsc
        yield SampleInput(regular, args=(regular,))
    
    # 处理布局为 `torch.sparse_coo` 的情况
    if layout is torch.sparse_coo:
        # 处理数据类型为 `torch.complex32` 的情况
        if dtype is torch.complex32:
            # RuntimeError: "mul_out_sparse" not implemented for 'ComplexHalf'
            yield SampleInput(regular, args=(regular,))
        # 处理数据类型为 `torch.bool` 且 `regular` 在 CPU 上的情况
        if dtype is torch.bool and regular.is_cpu:
            # RuntimeError: "mul_out_sparse" not implemented for 'Bool'
            yield SampleInput(regular, args=(regular,))
        # 处理数据类型为 `torch.bool` 或 `torch.float16` 且 `hybrid` 在 CPU 上的情况
        if dtype in {torch.bool, torch.float16} and regular.is_cpu:
            # RuntimeError: "addcmul_cpu_out" not implemented for '(Bool|Half)'
            yield SampleInput(hybrid, args=(hybrid,))
# 验证稀疏张量元素级二进制操作的样本输入的有效性，根据操作信息和样本输入参数来进行验证
def _validate_sample_input_sparse_elementwise_binary_operation(
    op_info, sample, check_validate=False
):
    # 如果操作名称为 "mul"，则调用 _validate_sample_input_elementwise_binary_sparse_mul 函数处理样本
    if op_info.name == "mul":
        sample = _validate_sample_input_elementwise_binary_sparse_mul(sample)

    # 如果需要进行验证，则调用 _check_validate 函数验证操作信息和样本
    if check_validate:
        _check_validate(op_info, sample)
    
    # 返回经过可能的处理和验证后的样本
    return sample


# 生成稀疏张量上的乘法操作的输入样本
def sample_inputs_sparse_mul(op_info, device, dtype, requires_grad, layout, **kwargs):
    """Sample inputs for mul operation on sparse tensors."""
    # 使用 _sample_inputs_sparse 函数生成稀疏元素级二进制操作的输入样本
    yield from _sample_inputs_sparse(
        sample_inputs_sparse_elementwise_binary_operation,
        _maybe_failing_sample_inputs_sparse_elementwise_binary_mul,
        _validate_sample_input_sparse_elementwise_binary_operation,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


# 生成稀疏张量上乘法操作的错误输入
def error_inputs_sparse_mul(op_info, device, layout, **kwargs):
    """Error inputs for mul operation on sparse tensors."""
    # 定义数据类型为 torch.float64，不需要梯度计算
    dtype = torch.float64
    requires_grad = False
    # 使用 _error_inputs_sparse 函数生成稀疏元素级二进制乘法操作的错误输入
    yield from _error_inputs_sparse(
        _maybe_failing_sample_inputs_sparse_elementwise_binary_mul,
        _validate_sample_input_sparse_elementwise_binary_operation,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


# 生成与给定张量具有相似性质的稀疏张量函数的输入样本
def _sample_inputs_sparse_like_fns(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    # 导入 TestCase 类，用于生成简单的输入样本
    from torch.testing._internal.common_utils import TestCase

    # 使用 TestCase().generate_simple_inputs 方法生成简单的输入样本
    for tensor in TestCase().generate_simple_inputs(
        layout,
        device=device,
        dtype=dtype,
        enable_batch=True,
        enable_hybrid=True,
        enable_zero_sized=True,
        enable_non_contiguous_indices=False,
        enable_non_contiguous_values=False,
    ):
        # 生成 SampleInput 对象，其中参数和关键字参数为空
        yield SampleInput(tensor, args=(), kwargs={})
        # 生成 SampleInput 对象，关键字参数包含设备、数据类型和布局信息
        yield SampleInput(
            tensor, args=(), kwargs=dict(device=device, dtype=dtype, layout=layout)
        )

        # 如果数据类型不是 torch.float64，则生成相应的 SampleInput 对象
        if dtype is not torch.float64:
            yield SampleInput(tensor, args=(), kwargs=dict(dtype=torch.float64))

        # 如果支持 CUDA，生成在另一个设备上的 SampleInput 对象
        if torch.cuda.is_available():
            other_device = "cuda" if tensor.device.type == "cpu" else "cpu"
            yield SampleInput(tensor, args=(), kwargs=dict(device=other_device))

        # 根据布局选择不同的 SampleInput 对象，用于测试不同稀疏布局
        if layout is torch.sparse_csr:
            other_layout = torch.sparse_csc
        elif layout is torch.sparse_csc:
            other_layout = torch.sparse_csr
        elif layout is torch.sparse_bsr:
            other_layout = torch.sparse_bsc
        elif layout is torch.sparse_bsc:
            other_layout = torch.sparse_bsr
        else:
            other_layout = torch.strided
        yield SampleInput(tensor, args=(), kwargs=dict(layout=other_layout))

        # 如果布局不是 torch.sparse_coo，则生成相应的 SampleInput 对象
        if layout is not torch.sparse_coo:
            yield SampleInput(tensor, args=(), kwargs=dict(layout=torch.sparse_coo))


# 验证与给定稀疏张量函数相似的输入样本的有效性
def _validate_sample_input_sparse_like_fns(op_info, sample, check_validate=False):
    # 检查输入的布局是否为稀疏格式之一，并且操作名不是 "zeros_like"
    if sample.input.layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    } and op_info.name not in {"zeros_like"}:
        # 检查是否指定了与输入布局不同的布局类型，如果是则返回错误信息
        if sample.kwargs.get("layout", sample.input.layout) != sample.input.layout:
            return ErrorInput(
                sample,
                error_regex=(
                    "empty_like with different sparse layout is not supported"
                    " \\(self is Sparse(Csc|Csr|Bsc|Bsr) but you requested Sparse(Csr|Csc|Bsr|Bsc)\\)"
                ),
            )
    # 如果输入的布局为 torch.sparse_coo，则返回错误信息
    if sample.input.layout is torch.sparse_coo:
        return ErrorInput(
            sample,
            error_regex=(
                "Could not run 'aten::normal_' with arguments from the 'Sparse(CPU|CUDA)' backend."
            ),
        )
    # 如果需要验证（check_validate），则调用 _check_validate 函数对操作信息和样本进行验证
    if check_validate:
        _check_validate(op_info, sample)
    # 返回样本数据
    return sample
# 定义一个函数，用于生成类似稀疏张量函数的样本输入
def _maybe_failing_sample_inputs_sparse_like_fns(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    # 如果CUDA可用且布局不是稀疏COO
    if torch.cuda.is_available() and layout is not torch.sparse_coo:
        # 根据设备类型选择另一个设备
        other_device = "cuda" if torch.device(device).type == "cpu" else "cpu"
        # 根据当前布局选择另一个布局
        if layout is torch.sparse_csr:
            other_layout = torch.sparse_csc
        elif layout is torch.sparse_csc:
            other_layout = torch.sparse_csr
        elif layout is torch.sparse_bsr:
            other_layout = torch.sparse_bsc
        elif layout is torch.sparse_bsc:
            other_layout = torch.sparse_bsr
        else:
            other_layout = torch.strided

        # 如果是块状布局，则设置块大小为(1, 1)，否则为None
        blocksize = (1, 1) if layout in {torch.sparse_bsr, torch.sparse_bsc} else None

        # 生成一个样本输入，包括稀疏张量和其他设备信息
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype, device=device).to_sparse(
                layout=layout, blocksize=blocksize
            ),
            kwargs=dict(device=other_device),
        )

        # 生成一个样本输入，包括稀疏张量和其他布局信息
        yield SampleInput(
            torch.tensor([[0, 1], [2, 3]], dtype=dtype, device=device).to_sparse(
                layout=layout, blocksize=blocksize
            ),
            kwargs=dict(layout=other_layout),
        )


# 定义一个函数，用于生成稀疏张量函数的样本输入
def sample_inputs_sparse_like_fns(
    op_info, device, dtype, requires_grad, layout, **kwargs
):
    """Sample inputs for like-functions on sparse tensors."""
    # 从可能失败的样本输入函数生成稀疏张量函数的样本输入
    yield from _sample_inputs_sparse(
        _sample_inputs_sparse_like_fns,
        _maybe_failing_sample_inputs_sparse_like_fns,
        _validate_sample_input_sparse_like_fns,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


# 定义一个函数，用于生成稀疏张量函数的错误输入
def error_inputs_sparse_like_fns(op_info, device, layout, **kwargs):
    """Error inputs for like-functions on sparse tensors."""
    # 指定数据类型为torch.float64，不需要梯度计算
    dtype = torch.float64
    requires_grad = False
    # 从可能失败的样本输入函数生成稀疏张量函数的错误输入
    yield from _error_inputs_sparse(
        _maybe_failing_sample_inputs_sparse_like_fns,
        _validate_sample_input_sparse_like_fns,
        op_info,
        device,
        dtype,
        requires_grad,
        layout,
        **kwargs,
    )


# 定义一个函数，用于验证默认稀疏样本输入
def _validate_sample_input_sparse_default(op_info, sample, check_validate=False):
    # 如果操作名称是"to_sparse"
    if op_info.name == "to_sparse":
        # 如果输入的布局在稀疏CSR、CSC、BSR或BSC之中，并且参数列表长度为1且参数类型是整数，且参数不等于2
        if (
            sample.input.layout
            in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}
            and len(sample.args) == 1
            and isinstance(sample.args[0], int)
            and sample.args[0] != 2
        ):
            # 返回一个错误输入，指示稀疏维度参数必须为2
            sample = ErrorInput(
                sample,
                error_regex="sparse dim argument must be 2 for sparse_compressed_to_sparse",
            )

    # 如果需要验证，则执行验证函数
    if check_validate:
        _check_validate(op_info, sample)
    # 返回验证后的样本输入
    return sample


# 定义一个函数，用于验证稀疏样本输入
def validate_sample_input_sparse(op_info, sample, check_validate=False):
    """Return the specified sample when it is valid and supported by the
    operation. Otherwise, return the sample as ErrorInput instance.
    """
    # 如果 op_info 是 ReductionOpInfo 类型的实例，则调用对应的验证函数来验证稀疏输入样本
    if isinstance(op_info, ReductionOpInfo):
        return _validate_sample_input_sparse_reduction(
            op_info, sample, check_validate=check_validate
        )
    # 如果 op_info 是 BinaryUfuncInfo 类型的实例，则调用对应的验证函数来验证稀疏输入样本
    elif isinstance(op_info, BinaryUfuncInfo):
        return _validate_sample_input_sparse_elementwise_binary_operation(
            op_info, sample, check_validate=check_validate
        )
    # 对于其他类型的 op_info，默认调用通用的稀疏输入验证函数
    else:
        return _validate_sample_input_sparse_default(
            op_info, sample, check_validate=check_validate
        )
```