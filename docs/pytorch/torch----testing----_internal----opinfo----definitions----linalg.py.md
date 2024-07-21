# `.\pytorch\torch\testing\_internal\opinfo\definitions\linalg.py`

```
# 忽略 mypy 类型检查时可能产生的错误
# 导入 itertools、random、unittest 和 partial 函数
import itertools
import random
import unittest
from functools import partial
# 导入 chain 和 product 函数，以及 Iterable、List 和 Tuple 类型
from itertools import chain, product
from typing import Iterable, List, Tuple

# 导入 numpy 库，并将 inf 函数直接导入当前命名空间
import numpy as np
from numpy import inf

# 导入 torch 库
import torch

# 导入 torch.testing.make_tensor 函数
from torch.testing import make_tensor
# 导入 torch.testing._internal.common_cuda 中的若干函数和变量
from torch.testing._internal.common_cuda import (
    _get_magma_version,
    _get_torch_cuda_version,
    with_tf32_off,
)
# 导入 torch.testing._internal.common_device_type 中的若干函数和变量
from torch.testing._internal.common_device_type import (
    has_cusolver,
    skipCPUIfNoLapack,
    skipCUDAIf,
    skipCUDAIfNoCusolver,
    skipCUDAIfNoMagma,
    skipCUDAIfNoMagmaAndNoCusolver,
    skipCUDAIfNoMagmaAndNoLinalgsolver,
    skipCUDAIfRocm,
    tol,
    toleranceOverride,
)
# 导入 torch.testing._internal.common_dtype 中的若干函数和变量
from torch.testing._internal.common_dtype import (
    all_types_and_complex,
    all_types_and_complex_and,
    floating_and_complex_types,
    floating_and_complex_types_and,
    get_all_complex_dtypes,
)
# 导入 torch.testing._internal.common_utils 中的若干函数和变量
from torch.testing._internal.common_utils import (
    GRADCHECK_NONDET_TOL,
    IS_MACOS,
    make_fullrank_matrices_with_distinct_singular_values,
    skipIfSlowGradcheckEnv,
    slowTest,
    TEST_WITH_ROCM,
)
# 导入 torch.testing._internal.opinfo.core 中的若干函数和变量
from torch.testing._internal.opinfo.core import (
    clone_sample,
    DecorateInfo,
    ErrorInput,
    gradcheck_wrapper_hermitian_input,
    L,
    M,
    OpInfo,
    ReductionOpInfo,
    S,
    SampleInput,
)
# 导入 torch.testing._internal.opinfo.refs 中的若干函数和变量
from torch.testing._internal.opinfo.refs import PythonRefInfo, ReductionPythonRefInfo


# 定义函数 sample_kwargs_vector_norm，接受张量 t 和任意关键字参数 kwargs
def sample_kwargs_vector_norm(t, **kwargs):
    # 内部定义函数 ords，用于返回不同的范数顺序
    def ords():
        # 具有标识（identity）的范数顺序
        has_id = (6, 4, 2, 1, 0, 0.9)
        # 不具有标识的范数顺序
        no_id = (inf, -2.1, -inf)
        # 如果张量 t 的元素数为 0
        if t.numel() == 0:
            dim = kwargs.get("dim")
            # 如果未指定维度，则返回具有标识的范数顺序
            if dim is None:
                return has_id
            # 如果指定了维度，并且维度不是可迭代的，则将其转为可迭代对象
            if not isinstance(dim, Iterable):
                dim = (dim,)
            # 遍历每个维度，如果指定维度的大小为 0，则返回具有标识的范数顺序
            for d in dim:
                if t.size(d) == 0:
                    return has_id
        # 返回包含标识和非标识范数顺序的完整顺序列表
        return has_id + no_id

    # 返回生成器表达式，为每个范数顺序创建一个字典，其中键为 ()，值为包含 ord 键的字典
    return (((), dict(ord=o)) for o in ords())


# 定义函数 sample_inputs_svd，用于生成奇异值分解（SVD）操作的样本输入
def sample_inputs_svd(op_info, device, dtype, requires_grad=False, **kwargs):
    # 使用 make_fullrank_matrices_with_distinct_singular_values 函数生成全秩矩阵
    make_fullrank = make_fullrank_matrices_with_distinct_singular_values
    make_arg = partial(
        make_fullrank, dtype=dtype, device=device, requires_grad=requires_grad
    )

    # 检查操作信息的名称是否包含 "linalg.svd"
    is_linalg_svd = "linalg.svd" in op_info.name
    # 定义不同的批次和维度
    batches = [(), (0,), (3,)]
    ns = [0, 3, 5]

    # 内部定义函数 uniformize，用于标准化奇异值分解的输出
    def uniformize(usv):
        S = usv[1]
        k = S.shape[-1]
        U = usv[0][..., :k]
        Vh = usv[2] if is_linalg_svd else usv[2].mH
        Vh = Vh[..., :k, :]
        return U, S, Vh

    # 定义几个函数，用于从奇异值分解的输出中提取 U、S 和 Vh
    def fn_U(usv):
        U, _, _ = uniformize(usv)
        return U.abs()

    def fn_S(usv):
        return uniformize(usv)[1]

    def fn_Vh(usv):
        _, S, Vh = uniformize(usv)
        return S, Vh.abs()

    def fn_UVh(usv):
        U, S, Vh = uniformize(usv)
        return U @ Vh, S

    # 将这些函数组合成元组
    fns = (fn_U, fn_S, fn_Vh, fn_UVh)

    # 定义 fullmat 变量，根据是否为 linalg.svd 进行选择
    fullmat = "full_matrices" if is_linalg_svd else "some"
    # 使用 product 函数生成多个迭代器的笛卡尔积，生成所有可能的组合
    for batch, n, k, fullmat_val, fn in product(batches, ns, ns, (True, False), fns):
        # 计算当前组合的形状，形状由 batch、n 和 k 决定
        shape = batch + (n, k)
        # 生成 SampleInput 对象，调用 make_arg 函数创建输入参数，设置输出处理函数和参数
        yield SampleInput(
            make_arg(*shape), kwargs={fullmat: fullmat_val}, output_process_fn_grad=fn
        )
# 生成交叉样本输入，用于测试函数的多个输入情况
def sample_inputs_cross(op_info, device, dtype, requires_grad, **kwargs):
    # 偏函数，创建张量的辅助函数，指定数据类型、设备和是否需要梯度
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )
    # 生成样本输入，形状为 (S, 3)，作为位置参数传递
    yield SampleInput(make_arg((S, 3)), args=(make_arg((S, 3)),))
    # 生成样本输入，形状为 (S, 3, S)，作为位置参数传递，使用关键字参数指定维度为 1
    yield SampleInput(
        make_arg((S, 3, S)), args=(make_arg((S, 3, S)),), kwargs=dict(dim=1)
    )
    # 生成样本输入，形状为 (1, 3) 和 (S, 3)，使用关键字参数指定维度为 -1
    yield SampleInput(make_arg((1, 3)), args=(make_arg((S, 3)),), kwargs=dict(dim=-1))


# 生成错误输入，用于测试函数抛出错误的情况
def error_inputs_cross(op_info, device, **kwargs):
    # 偏函数，创建张量的辅助函数，指定设备和数据类型为 torch.float32
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)

    # 创建样本输入，形状为 (S, 3) 和 (S, 1)，用于测试抛出 RuntimeError 错误，期望错误信息为 "inputs dimension -1 must have length 3"
    sample = SampleInput(input=make_arg((S, 3)), args=(make_arg((S, 1)),))
    err = "inputs dimension -1 must have length 3"
    yield ErrorInput(sample, error_regex=err, error_type=RuntimeError)

    # 创建样本输入，形状为 (5, S, 3) 和 (S, 3)，用于测试抛出 RuntimeError 错误，期望错误信息为 "inputs must have the same number of dimensions"
    sample = SampleInput(input=make_arg((5, S, 3)), args=(make_arg((S, 3)),))
    err = "inputs must have the same number of dimensions"
    yield ErrorInput(sample, error_regex=err, error_type=RuntimeError)

    # 创建样本输入，形状为 (S, 2) 和 (S, 2)，用于测试抛出 RuntimeError 错误，期望错误信息为 "must have length 3"
    sample = SampleInput(input=make_arg((S, 2)), args=(make_arg((S, 2)),))
    err = "must have length 3"
    yield ErrorInput(sample, error_regex=err, error_type=RuntimeError)

    # 创建样本输入，形状为 (S, 2) 和 (S, 2)，使用关键字参数指定维度为 2，用于测试抛出 IndexError 错误，期望错误信息为 "Dimension out of range"
    sample = SampleInput(
        input=make_arg((S, 2)), args=(make_arg((S, 2)),), kwargs=dict(dim=2)
    )
    err = "Dimension out of range"
    yield ErrorInput(sample, error_regex=err, error_type=IndexError)


def sample_inputs_householder_product(op_info, device, dtype, requires_grad, **kwargs):
    """
    This function generates input for torch.linalg.householder_product (torch.orgqr).
    The first argument should be a square matrix or batch of square matrices, the second argument is a vector or batch of vectors.
    Empty, square, rectangular, batched square and batched rectangular input is generated.
    """
    # 偏函数，创建张量的辅助函数，指定设备、数据类型和是否需要梯度，数值范围为 [-2, 2]
    make_arg = partial(
        make_tensor,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        low=-2,
        high=2,
    )
    # 生成样本输入：形状为 (S, S) 的矩阵和形状为 (S,) 的向量
    yield SampleInput(make_arg((S, S)), make_arg((S,)))
    # 生成样本输入：形状为 (S+1, S) 的矩阵和形状为 (S,) 的向量
    yield SampleInput(make_arg((S + 1, S)), make_arg((S,)))
    # 生成样本输入：形状为 (2, 1, S, S) 的矩阵和形状为 (2, 1, S) 的向量
    yield SampleInput(make_arg((2, 1, S, S)), make_arg((2, 1, S)))
    # 生成样本输入：形状为 (2, 1, S+1, S) 的矩阵和形状为 (2, 1, S) 的向量
    yield SampleInput(make_arg((2, 1, S + 1, S)), make_arg((2, 1, S)))
    # 生成样本输入：形状为 (0, 0) 的矩阵和形状为 (0,) 的向量，不指定数值范围
    yield SampleInput(
        make_arg((0, 0), low=None, high=None),
        make_arg((0,), low=None, high=None),
    )
    # 生成样本输入：形状为 (S, S) 的矩阵和形状为 (0,) 的向量，不指定数值范围
    yield SampleInput(make_arg((S, S)), make_arg((0,), low=None, high=None))
    # 生成样本输入：形状为 (S, S) 的矩阵和形状为 (S-2,) 的向量，不指定数值范围
    yield SampleInput(make_arg((S, S)), make_arg((S - 2,), low=None, high=None))
    # 生成样本输入：形状为 (S, S-1) 的矩阵和形状为 (S-2,) 的向量，不指定数值范围
    yield SampleInput(make_arg((S, S - 1)), make_arg((S - 2,), low=None, high=None))
def sample_inputs_linalg_det_singular(op_info, device, dtype, requires_grad, **kwargs):
    # 创建一个部分应用了make_tensor的函数make_arg，用于生成具有特定设备和数据类型的张量
    make_arg = partial(make_tensor, device=device, dtype=dtype)

    # 定义生成奇异矩阵批次的函数make_singular_matrix_batch_base
    def make_singular_matrix_batch_base(size, rank):
        # 断言矩阵维度的最后两维是相等的
        assert size[-1] == size[-2]
        # 断言秩大于0且小于最后一个维度的大小
        assert rank > 0 and rank < size[-1]

        # 获取矩阵的大小
        n = size[-1]
        # 生成大小为(size[:-2] + (n, rank))的张量a和(size[:-2] + (rank, n))的张量b，并缩放为原来的1/10
        a = make_arg(size[:-2] + (n, rank)) / 10
        b = make_arg(size[:-2] + (rank, n)) / 10
        # 计算矩阵乘积x = a @ b
        x = a @ b
        # 对x进行LU分解并获取LU分解的结果lu、置换向量pivs以及误差
        lu, pivs, _ = torch.linalg.lu_factor_ex(x)
        # 根据LU分解的结果lu重建矩阵L和U，并获取U的对角线绝对值的张量u_diag_abs
        p, l, u = torch.lu_unpack(lu, pivs)
        u_diag_abs = u.diagonal(0, -2, -1).abs()
        # 获取绝对值最大的对角线元素，并保持其维度
        u_diag_abs_largest = u_diag_abs.max(dim=-1, keepdim=True).values
        # 找出最小的对角线元素的索引
        u_diag_abs_smallest_idxs = torch.topk(
            u_diag_abs, k=(n - rank), largest=False
        ).indices
        # 将U的对角线元素除以最大绝对值的对角线元素，并在最小对角线元素处加入dtype类型的最小值
        u.diagonal(0, -2, -1).div_(u_diag_abs_largest)
        u.diagonal(0, -2, -1)[..., u_diag_abs_smallest_idxs] = torch.finfo(dtype).eps
        # 构建矩阵p @ l @ u
        matrix = p @ l @ u

        # 如果需要梯度，则设置matrix为需要梯度的张量
        matrix.requires_grad_(requires_grad)
        return matrix

    # 遍历空元组、(2,)和(2, 2)的batch和size的笛卡尔积
    for batch, size in product(((), (2,), (2, 2)), range(6)):
        shape = batch + (size, size)
        # 遍历秩从1到size-1的范围
        for rank in range(1, size):
            # 返回SampleInput对象，其值为make_singular_matrix_batch_base生成的矩阵
            yield SampleInput(make_singular_matrix_batch_base(shape, rank))


def sample_inputs_linalg_matrix_power(op_info, device, dtype, requires_grad, **kwargs):
    # 使用make_fullrank_matrices_with_distinct_singular_values创建make_fullrank函数
    make_fullrank = make_fullrank_matrices_with_distinct_singular_values
    # 创建部分应用make_tensor函数的make_arg，用于生成具有特定设备、数据类型和梯度需求的张量
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )
    # 创建部分应用make_fullrank函数的make_arg_fullrank，用于生成具有特定设备、数据类型和梯度需求的张量
    make_arg_fullrank = partial(
        make_fullrank, dtype=dtype, device=device, requires_grad=requires_grad
    )
    # 定义测试矩阵大小和批次大小的列表test_sizes
    test_sizes = [
        (1, ()),
        (2, (0,)),
        (2, (2,)),
    ]

    # 遍历test_sizes列表
    for matrix_size, batch_sizes in test_sizes:
        size = batch_sizes + (matrix_size, matrix_size)
        # 遍历n为0、3和5的范围
        for n in (0, 3, 5):
            # 返回SampleInput对象，其值为make_arg生成的张量，带有参数n
            yield SampleInput(make_arg(size), args=(n,))
        # 遍历n为-4、-2和-1的范围
        for n in [-4, -2, -1]:
            # 返回SampleInput对象，其值为make_arg_fullrank生成的张量，带有参数n
            yield SampleInput(make_arg_fullrank(*size), args=(n,))


def sample_inputs_linalg_det_logdet_slogdet(
    op_info, device, dtype, requires_grad, **kwargs
):
    # 使用make_fullrank_matrices_with_distinct_singular_values创建make_fullrank函数
    make_fullrank = make_fullrank_matrices_with_distinct_singular_values
    # 创建部分应用make_fullrank函数的make_arg，用于生成具有特定设备、数据类型和梯度需求的张量
    make_arg = partial(
        make_fullrank, dtype=dtype, device=device, requires_grad=requires_grad
    )
    # 定义批次大小的列表batches和矩阵维度n的列表ns
    batches = [(), (0,), (3,)]
    ns = [0, 1, 5]

    # 判断op_info.name是否为"logdet"
    is_logdet = op_info.name == "logdet"

    # 遍历batches和ns的笛卡尔积
    for (
        batch,
        n,
    ) in product(batches, ns):
        shape = batch + (n, n)
        # 生成具有形状shape的张量A
        A = make_arg(*shape)
        # 如果是logdet操作且A不是复数类型并且元素数量大于0
        if is_logdet and not A.is_complex() and A.numel() > 0:
            # 获取A的slogdet的符号并将A乘以其行列式的符号以改变行列式的符号
            s = torch.linalg.slogdet(A).sign
            A = A * s.unsqueeze(-1).unsqueeze(-1)
            # 如果需要梯度，则设置A为需要梯度的张量
            A.requires_grad_(requires_grad)
        # 返回SampleInput对象，其值为张量A
        yield SampleInput(A)
    """Samples the inputs for both linalg.lu_solve and lu_solve"""
    # 定义函数 make_fn 为 make_fullrank_matrices_with_distinct_singular_values 的偏函数
    make_fn = make_fullrank_matrices_with_distinct_singular_values
    # 定义 make_a 和 make_b 为 make_fn 和 make_tensor 的偏函数，指定 dtype 和 device
    make_a = partial(make_fn, dtype=dtype, device=device)
    make_b = partial(make_tensor, dtype=dtype, device=device)

    def clone(X, requires_grad):
        # 克隆张量 X，并设置是否需要梯度
        Y = X.clone()
        Y.requires_grad_(requires_grad)
        return Y

    # 判断当前操作是否为 linalg.lu_solve
    is_linalg_lu_solve = op_info.name == "linalg.lu_solve"

    # 定义不同的批次和维度大小的组合
    batches = ((), (0,), (2,))
    ns = (3, 1, 0)
    nrhs = (4, 1, 0)

    # 遍历所有可能的维度组合
    for n, batch, rhs in product(ns, batches, nrhs):
        # 创建矩阵 A，维度为 (batch + (n, n))
        A = make_a(*(batch + (n, n)))
        # 对矩阵 A 进行 LU 分解，得到 LU 分解结果 LU 和置换矩阵 pivots
        LU, pivots = torch.linalg.lu_factor(A)

        # 创建张量 B，维度为 (batch + (n, rhs))
        B = make_b(batch + (n, rhs))

        # 判断是否需要计算梯度
        grads = (False,) if not requires_grad else (True, False)
        # 针对每个输入是否需要梯度的组合进行遍历
        # 当 requires_grad == True 时，至少一个输入需要启用 requires_grad
        for LU_grad, B_grad in product(grads, grads):
            if requires_grad and not LU_grad and not B_grad:
                continue

            if is_linalg_lu_solve:
                # 遍历 adjoint 和 left 的所有组合
                for adjoint, left in product((True, False), repeat=2):
                    # 生成 SampleInput 对象，传入克隆后的 LU、B 张量及其他参数
                    yield SampleInput(
                        clone(LU, LU_grad),
                        args=(pivots, clone(B if left else B.mT, B_grad)),
                        kwargs=dict(adjoint=adjoint, left=left),
                    )
            else:
                # 生成 SampleInput 对象，传入克隆后的 B、LU 张量及其他参数
                yield SampleInput(clone(B, B_grad), args=(clone(LU, LU_grad), pivots))
# 定义一个函数，生成包含多个线性代数运算测试用例的输入数据
def sample_inputs_linalg_multi_dot(op_info, device, dtype, requires_grad, **kwargs):
    # 每个测试用例包含一个矩阵乘法链的大小
    # 例如 [2, 3, 4, 5] 生成矩阵 (2, 3) @ (3, 4) @ (4, 5)
    test_cases = [
        [1, 2, 1],
        [2, 0, 2],
        [0, 2, 2],
        [2, 2, 2, 2],
        [2, 3, 4, 5],
        [5, 4, 0, 2],
        [2, 4, 3, 5, 3, 2],
    ]

    # 遍历每个测试用例的矩阵尺寸
    for sizes in test_cases:
        tensors = []
        # 遍历每对相邻的尺寸，生成对应的张量
        for size in zip(sizes[:-1], sizes[1:]):
            t = make_tensor(
                size, dtype=dtype, device=device, requires_grad=requires_grad
            )
            tensors.append(t)
        # 生成一个 SampleInput 对象，包含生成的张量作为输入
        yield SampleInput(tensors)


# 定义一个函数，生成包含矩阵范数运算测试用例的输入数据
def sample_inputs_linalg_matrix_norm(op_info, device, dtype, requires_grad, **kwargs):
    low_precision_dtypes = (torch.float16, torch.bfloat16, torch.complex32)
    # 部分应用 make_tensor 函数，固定设备、数据类型和梯度属性
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )

    sizes = ((2, 2), (2, 3, 2))
    if dtype in low_precision_dtypes:
        # 低精度数据类型不支持 svdvals
        ords = ("fro", inf, -inf, 1, -1)
    else:
        ords = ("fro", "nuc", inf, -inf, 1, -1, 2, -2)
    dims = ((-2, -1), (-1, 0))

    # 使用 product 函数生成所有尺寸、阶数、维度和 keepdim 值的组合
    for size, ord, dim, keepdim in product(sizes, ords, dims, [True, False]):
        # 生成一个 SampleInput 对象，包含 make_arg 生成的张量作为输入
        yield SampleInput(make_arg(size), args=(ord, dim, keepdim))


# 定义一个函数，生成包含向量范数运算测试用例的输入数据
def sample_inputs_linalg_norm(
    op_info, device, dtype, requires_grad, *, variant=None, **kwargs
):
    # 如果提供了 variant 参数但不符合预期，抛出异常
    if variant is not None and variant not in ("subgradient_at_zero",):
        raise ValueError(
            f"Unsupported variant, expected variant to be 'subgradient_at_zero' but got: {variant}"
        )

    # 定义测试用例中可能的张量尺寸
    test_sizes = [
        (S,),
        (0,),
        (S, S),
        (0, 0),
        (S, 0),
        (0, S),
        (S, S, S),
        (0, S, S),
        (S, 0, S),
        (0, 0, 0),
    ]

    # 定义可能的向量阶数
    vector_ords = (None, 0, 0.5, 1, 2, 3.5, inf, -0.5, -1, -2, -3.5, -inf)
    if dtype in {torch.float16, torch.bfloat16, torch.complex32}:
        # 低精度数据类型不支持 svdvals
        matrix_ords = ("fro", inf, -inf, 1, -1)
    else:
        matrix_ords = (None, "fro", "nuc", inf, -inf, 1, -1, 2, -2)

    # 部分应用 make_tensor 函数，固定数据类型、设备和梯度属性，而 low 和 high 参数不设定
    make_arg = partial(
        make_tensor,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        low=None,
        high=None,
    )
    for test_size in test_sizes:
        # 检查当前测试大小是否为向量范数（长度为1）或矩阵范数（长度为2）
        is_vector_norm = len(test_size) == 1
        is_matrix_norm = len(test_size) == 2

        # 检查是否对 p=2 范数有效，要么是向量范数，要么最后两个维度大小均不为0
        is_valid_for_p2 = is_vector_norm or (test_size[-1] != 0 and test_size[-2] != 0)

        for keepdim in [False, True]:
            # 如果变体不是 "subgradient_at_zero" 并且对 p=2 范数有效，则生成样本输入
            if variant != "subgradient_at_zero" and is_valid_for_p2:
                yield SampleInput(make_arg(test_size), keepdim=keepdim)

            # 如果既不是向量范数也不是矩阵范数，则继续下一个迭代
            if not (is_vector_norm or is_matrix_norm):
                continue

            # 根据是向量范数还是矩阵范数选择相应的范数列表
            ords = vector_ords if is_vector_norm else matrix_ords

            for ord in ords:
                # 如果是向量范数且最后一个维度大小为0，则跳过
                if is_vector_norm and test_size[-1] == 0:
                    if ord == np.inf or (ord is not None and ord < 0):
                        # 如果是 np.inf 或 ord 小于0，则跳过当前迭代
                        continue
                elif is_matrix_norm:
                    # 根据不同的 ord 值选择相应需要检查的维度
                    dims_to_check = {
                        None: (0,),
                        np.inf: (0,),
                        2: (0, 1),
                        1: (1,),
                        -1: (1,),
                        -2: (0, 1),
                        -np.inf: (0,),
                    }.get(ord, ())

                    # 如果任何需要检查的维度大小为0，则跳过当前迭代
                    if any(test_size[d] == 0 for d in dims_to_check):
                        continue

                # 如果变体是 "subgradient_at_zero"，则生成以零张量为输入的样本
                if variant == "subgradient_at_zero":
                    yield SampleInput(
                        torch.zeros(
                            test_size,
                            dtype=dtype,
                            device=device,
                            requires_grad=requires_grad,
                        ),
                        ord,
                        keepdim=keepdim,
                    )
                else:
                    # 否则生成使用 make_arg 函数生成的输入样本
                    yield SampleInput(make_arg(test_size), ord, keepdim=keepdim)

                    # 如果 ord 是 "nuc" 或 "fro"，则生成使用指定维度的输入样本
                    if ord in ["nuc", "fro"]:
                        yield SampleInput(
                            make_arg(test_size), ord=ord, keepdim=keepdim, dim=(0, 1)
                        )
def sample_inputs_linalg_vecdot(op_info, device, dtype, requires_grad, **kwargs):
    # Partial function to create tensors with specified device, dtype, and requires_grad flag
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    # Various combinations of batches and sizes for tensor shapes
    batches = ((), (0,), (1,), (5,))
    # Different sizes for tensor dimensions
    ns = (0, 1, 3, 5)
    # Iterate over all combinations of batches and sizes
    for b, n in product(batches, ns):
        # Form the shape tuple
        shape = b + (n,)
        # Yield a SampleInput object with tensor and its creation arguments
        yield SampleInput(make_arg(shape), args=(make_arg(shape),))
        # Iterate over dimensions of the shape
        for i in range(len(shape)):
            # Yield a SampleInput with tensor, creation arguments, and dimension specified
            yield SampleInput(
                make_arg(shape), args=(make_arg(shape),), kwargs=dict(dim=i)
            )


def sample_inputs_linalg_invertible(
    op_info, device, dtype, requires_grad=False, **kwargs
):
    """
    This function generates invertible inputs for linear algebra ops
    The input is generated as the itertools.product of 'batches' and 'ns'.
    In total this function generates 8 SampleInputs
    'batches' cases include:
        () - single input,
        (0,) - zero batched dimension,
        (2,) - batch of two matrices,
        (1, 1) - 1x1 batch of matrices
    'ns' gives 0x0 and 5x5 matrices.
    Zeros in dimensions are edge cases in the implementation and important to test for in order to avoid unexpected crashes.
    """
    # Function to generate matrices with distinct singular values
    make_fn = make_fullrank_matrices_with_distinct_singular_values
    # Partial function to create tensors with specified dtype, device, and requires_grad flag
    make_arg = partial(make_fn, dtype=dtype, device=device, requires_grad=requires_grad)

    # Various combinations of batch sizes and matrix dimensions
    batches = [(), (0,), (2,), (1, 1)]
    # Matrix dimensions for testing invertibility
    ns = [5, 0]

    # Iterate over all combinations of batches and matrix dimensions
    for batch, n in product(batches, ns):
        # Yield a SampleInput with an invertible matrix as input
        yield SampleInput(make_arg(*batch, n, n))


def sample_inputs_matrix_rank(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function produces inputs for matrix rank that test
    all possible combinations for atol and rtol
    """

    # Function to create atol or rtol argument based on type
    def make_tol_arg(kwarg_type, inp):
        if kwarg_type == "none":
            return None
        if kwarg_type == "float":
            return 1.0
        assert kwarg_type == "tensor"
        return torch.ones(inp.shape[:-2], device=device)

    # Iterate over float and tensor types for atol and rtol
    for tol_type in ["float", "tensor"]:
        # Iterate over combinations of atol and rtol types
        for atol_type, rtol_type in product(["none", tol_type], repeat=2):
            # Skip the default behavior cases where neither atol nor rtol are specified
            if not atol_type and not rtol_type:
                continue
            # Iterate over SampleInputs generated from invertible matrix inputs
            for sample in sample_inputs_linalg_invertible(
                op_info, device, dtype, requires_grad
            ):
                # Ensure no additional kwargs are initially present
                assert sample.kwargs == {}
                # Set kwargs for atol and rtol based on specified types
                sample.kwargs = {
                    "atol": make_tol_arg(atol_type, sample.input),
                    "rtol": make_tol_arg(rtol_type, sample.input),
                }
                # Yield the modified SampleInput with specified kwargs
                yield sample

    # Yield SampleInputs with default kwargs for invertible matrix inputs
    yield from sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad)


def sample_inputs_linalg_pinv_singular(
    op_info, device, dtype, requires_grad=False, **kwargs
):
    """
    This function produces factors `a` and `b` to generate inputs of the form `a @ b.t()` to
    """
    # Functionally similar to sample_inputs_linalg_invertible but tailored for singular matrix inputs
    test the backward method of `linalg_pinv`. That way we always preserve the rank of the
    input no matter the perturbations applied to it by the gradcheck.
    Note that `pinv` is Frechet-differentiable in a rank-preserving neighborhood.
    """
    batches = [(), (0,), (2,), (1, 1)]
    # 定义多个不同大小的批次和矩阵维度
    size = [0, 3, 50]

    # 对每一组批次和矩阵大小的组合进行迭代
    for batch, m, n in product(batches, size, size):
        # 对于每一个迭代组合，取最小的3、m、n中的值
        for k in range(min(3, m, n)):
            # 通过随机生成的矩阵创建 `a`，使其正交化，并且标记为需要梯度计算
            a = (
                torch.rand(*batch, m, k, device=device, dtype=dtype)
                .qr()  # 对生成的随机矩阵进行QR分解，得到正交矩阵Q
                .Q.requires_grad_(requires_grad)  # 标记Q需要计算梯度
            )
            # 通过随机生成的矩阵创建 `b`，使其正交化，并且标记为需要梯度计算
            b = (
                torch.rand(*batch, n, k, device=device, dtype=dtype)
                .qr()  # 对生成的随机矩阵进行QR分解，得到正交矩阵Q
                .Q.requires_grad_(requires_grad)  # 标记Q需要计算梯度
            )
            # 生成一个 `SampleInput` 实例，作为生成器的输出
            yield SampleInput(a, args=(b,))
def sample_inputs_linalg_cond(op_info, device, dtype, requires_grad=False, **kwargs):
    # 偏函数，用于创建具有指定数据类型、设备和梯度属性的张量
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )

    # autograd 不支持元素数量为零的输入
    shapes = (
        (S, S),         # 形状为 (S, S) 的张量
        (2, S, S),      # 形状为 (2, S, S) 的张量
        (2, 1, S, S),   # 形状为 (2, 1, S, S) 的张量
    )

    # 遍历不同形状的张量，并生成 SampleInput 对象
    for shape in shapes:
        yield SampleInput(make_arg(shape))


def sample_inputs_linalg_vander(op_info, device, dtype, requires_grad=False, **kwargs):
    # 偏函数，用于创建具有指定数据类型、设备和梯度属性的张量
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )

    shapes = (
        (),         # 空形状的张量
        (1,),       # 形状为 (1,) 的张量
        (S,),       # 形状为 (S,) 的张量
        (2, S),     # 形状为 (2, S) 的张量
    )

    # 遍历不同形状的张量，并根据条件生成 SampleInput 对象
    for shape in shapes:
        if len(shape) > 0 and shape[-1] > 1:
            yield SampleInput(make_arg(shape))
        n = shape[-1] if len(shape) > 0 else 1
        for i in range(3):
            # 生成 n-1, n, n+1 三个不同的 N 值
            N = n + i - 1
            if N < 2:
                continue
            yield SampleInput(make_arg(shape), kwargs=dict(N=N))


def np_vander_batched(x, N=None):
    # 包装器，支持对一维张量批处理的 np.vander 调用（适合测试需求）
    if x.ndim == 0:
        x = x[np.newaxis]
    if x.ndim == 1:
        y = np.vander(x, N=N, increasing=True)
        return y
    else:
        if N is None:
            N = x.shape[-1]
        y = np.vander(x.ravel(), N=N, increasing=True).reshape((*x.shape, N))
        return y


def sample_inputs_linalg_cholesky_inverse(
    op_info, device, dtype, requires_grad=False, **kwargs
):
    from torch.testing._internal.common_utils import random_well_conditioned_matrix

    # Cholesky 分解适用于正定矩阵
    single_well_conditioned_matrix = random_well_conditioned_matrix(
        S, S, dtype=dtype, device=device
    )
    batch_well_conditioned_matrices = random_well_conditioned_matrix(
        2, S, S, dtype=dtype, device=device
    )
    single_pd = single_well_conditioned_matrix @ single_well_conditioned_matrix.mH
    batch_pd = batch_well_conditioned_matrices @ batch_well_conditioned_matrices.mH

    # 不同类型的输入矩阵
    inputs = (
        torch.zeros(0, 0, dtype=dtype, device=device),  # 0x0 矩阵
        torch.zeros(0, 2, 2, dtype=dtype, device=device),  # 零批量矩阵
        single_pd,      # 单个正定矩阵
        batch_pd,       # 批量正定矩阵
    )

    # 为每个输入矩阵生成 Cholesky 分解的测试用例
    test_cases = (torch.linalg.cholesky(a, upper=False) for a in inputs)
    for l in test_cases:
        # 生成下三角样本
        l.requires_grad = requires_grad
        yield SampleInput(l)  # 默认为 upper=False
        yield SampleInput(
            l.detach().clone().requires_grad_(requires_grad), kwargs=dict(upper=False)
        )

        # 生成上三角输入
        u = l.detach().clone().mT.contiguous().requires_grad_(requires_grad)
        yield SampleInput(u, kwargs=dict(upper=True))


def sample_inputs_linalg_ldl_factor(
    op_info, device, dtype, requires_grad=False, **kwargs
):
    # 从 torch.testing._internal.common_utils 导入所需的函数
    from torch.testing._internal.common_utils import (
        random_hermitian_pd_matrix,
        random_symmetric_pd_matrix,
    )

    # 将设备转换为 torch.device 对象
    device = torch.device(device)

    # 对称矩阵输入
    # 生成一个随机对称正定矩阵，返回一个 SampleInput 对象，包含单个矩阵
    yield SampleInput(
        random_symmetric_pd_matrix(S, dtype=dtype, device=device),
        kwargs=dict(hermitian=False),
    )  # single matrix
    # 生成一个随机对称正定矩阵，返回一个 SampleInput 对象，包含批量矩阵
    yield SampleInput(
        random_symmetric_pd_matrix(S, 2, dtype=dtype, device=device),
        kwargs=dict(hermitian=False),
    )  # batch of matrices
    # 生成一个零矩阵，返回一个 SampleInput 对象，包含单个矩阵
    yield SampleInput(
        torch.zeros(0, 0, dtype=dtype, device=device), kwargs=dict(hermitian=False)
    )  # 0x0 matrix
    # 生成一个零矩阵，返回一个 SampleInput 对象，包含批量矩阵
    yield SampleInput(
        torch.zeros(0, 2, 2, dtype=dtype, device=device), kwargs=dict(hermitian=False)
    )  # zero batch of matrices

    # 共轭转置矩阵输入
    # 对于复数输入在 CUDA 上 hermitian=True 仅支持 MAGMA 2.5.4+
    magma_254_available = device.type == "cuda" and _get_magma_version() >= (2, 5, 4)
    if dtype.is_complex and (device.type == "cpu" or magma_254_available):
        # 生成一个随机共轭转置正定矩阵，返回一个 SampleInput 对象，包含单个矩阵
        yield SampleInput(
            random_hermitian_pd_matrix(S, dtype=dtype, device=device),
            kwargs=dict(hermitian=True),
        )  # single matrix
        # 生成一个随机共轭转置正定矩阵，返回一个 SampleInput 对象，包含批量矩阵
        yield SampleInput(
            random_hermitian_pd_matrix(S, 2, dtype=dtype, device=device),
            kwargs=dict(hermitian=True),
        )  # batch of matrices
# 定义函数 `sample_inputs_linalg_ldl_solve`，生成用于测试 `torch.linalg.ldl_factor_ex` 函数的输入样本
def sample_inputs_linalg_ldl_solve(
    op_info, device, dtype, requires_grad=False, **kwargs
):
    # 导入生成随机 Hermitian 和 symmetric 正定矩阵的工具函数
    from torch.testing._internal.common_utils import (
        random_hermitian_pd_matrix,
        random_symmetric_pd_matrix,
    )

    # 将设备字符串转换为 torch 设备对象
    device = torch.device(device)
    
    # 定义 symmetric_inputs，包括单个矩阵、批量矩阵和零矩阵的元组
    symmetric_inputs = (
        random_symmetric_pd_matrix(S, dtype=dtype, device=device),  # 单个矩阵
        random_symmetric_pd_matrix(
            S, 2, dtype=dtype, device=device
        ),  # 批量矩阵
        torch.zeros(0, 0, dtype=dtype, device=device),  # 0x0 矩阵
        torch.zeros(0, 2, 2, dtype=dtype, device=device),  # 零批量矩阵
    )
    
    # 根据设备和数据类型，定义 hermitian_inputs，仅在 CPU 和复数类型时包括
    hermitian_inputs = (
        (
            random_hermitian_pd_matrix(S, dtype=dtype, device=device),
            random_hermitian_pd_matrix(S, 2, dtype=dtype, device=device),
        )
        if device.type == "cpu" and dtype.is_complex
        else ()
    )
    
    # 对 symmetric_inputs 中的每个矩阵调用 `torch.linalg.ldl_factor_ex`，返回生成器对象 test_cases1
    test_cases1 = (
        torch.linalg.ldl_factor_ex(a, hermitian=False) for a in symmetric_inputs
    )
    
    # 对 hermitian_inputs 中的每个矩阵调用 `torch.linalg.ldl_factor_ex`，返回生成器对象 test_cases2
    test_cases2 = (
        torch.linalg.ldl_factor_ex(a, hermitian=True) for a in hermitian_inputs
    )

    # 对于 symmetric_inputs 中的每个测试用例进行迭代
    make_arg = partial(
        make_tensor, device=device, dtype=dtype, requires_grad=requires_grad
    )
    for test_case in test_cases1:
        # 解包 test_case 结果
        factors, pivots, _ = test_case
        # 标记 factors 是否需要梯度
        factors.requires_grad = requires_grad
        # 对于 B_batch_shape 的每种形状，生成 B 张量并返回 SampleInput
        for B_batch_shape in ((), factors.shape[:-2]):
            B = make_arg((*B_batch_shape, factors.shape[-1], S))
            yield SampleInput(factors, args=(pivots, B), kwargs=dict(hermitian=False))
            # 克隆 factors，并返回带梯度信息的 SampleInput
            clone_factors = factors.detach().clone().requires_grad_(requires_grad)
            yield SampleInput(
                clone_factors, args=(pivots, B), kwargs=dict(hermitian=False)
            )

    # 对于 hermitian_inputs 中的每个测试用例进行迭代
    for test_case in test_cases2:
        # 解包 test_case 结果
        factors, pivots, _ = test_case
        # 标记 factors 是否需要梯度
        factors.requires_grad = requires_grad
        # 对于 B_batch_shape 的每种形状，生成 B 张量并返回 SampleInput
        for B_batch_shape in ((), factors.shape[:-2]):
            B = make_arg((*B_batch_shape, factors.shape[-1], S))
            yield SampleInput(factors, args=(pivots, B), kwargs=dict(hermitian=True))
            # 克隆 factors，并返回带梯度信息的 SampleInput
            clone_factors = factors.detach().clone().requires_grad_(requires_grad)
            yield SampleInput(
                clone_factors, args=(pivots, B), kwargs=dict(hermitian=True)
            )


``````
# 定义函数 `sample_inputs_linalg_lstsq`，生成用于测试线性最小二乘问题的输入样本
def sample_inputs_linalg_lstsq(op_info, device, dtype, requires_grad=False, **kwargs):
    # 导入生成随机良条件矩阵的工具函数
    from torch.testing._internal.common_utils import random_well_conditioned_matrix

    # 将设备字符串转换为 torch 设备对象
    device = torch.device(device)

    # 根据设备类型选择使用的求解驱动程序
    drivers: Tuple[str, ...]
    if device.type == "cuda":
        drivers = ("gels",)
    else:
        drivers = ("gels", "gelsy", "gelss", "gelsd")

    # 定义 delta 值的元组，根据设备是否为 CPU 或具有 cusolver 决策
    deltas: Tuple[int, ...]
    if device.type == "cpu" or has_cusolver():
        deltas = (-1, 0, +1)
    # 如果 Cusolver 不可用，仅使用方阵系统
    # 如果不是处理正向问题（正常情况），则设置 deltas 为一个空元组
    else:
        deltas = (0,)

    # 使用 product 函数生成多个三元组的组合，每个三元组包含 batch、driver 和 delta
    # batch 可能是空元组 ()、包含一个元素的元组 (3,) 或包含两个元素的元组 (3, 3)
    # drivers 是一个迭代器，包含多个驱动对象
    # deltas 是一个元组，包含一个或零个整数，表示矩阵维数的增量
    for batch, driver, delta in product(((), (3,), (3, 3)), drivers, deltas):
        # 根据当前的 batch、delta 组合确定矩阵的形状 shape
        shape = batch + (3 + delta, 3)
        # 创建一个随机且条件良好的矩阵 a，形状为 shape，数据类型为 dtype，存储设备为 device
        a = random_well_conditioned_matrix(*shape, dtype=dtype, device=device)
        # 设置是否需要计算梯度
        a.requires_grad_(requires_grad)
        # 创建一个张量 b，形状为 shape，数据类型为 dtype，存储设备为 device
        # b 的值在 low 和 high 范围内随机生成，是否需要计算梯度根据 requires_grad 决定
        b = make_tensor(
            shape,
            dtype=dtype,
            device=device,
            low=None,
            high=None,
            requires_grad=requires_grad,
        )
        # 生成一个 SampleInput 对象，包含矩阵 a、张量 b 和驱动对象 driver
        yield SampleInput(a, b, driver=driver)
def error_inputs_lstsq(op_info, device, **kwargs):
    # 创建一个标量张量，形状为空元组，随机生成，指定设备
    zero_d = torch.randn((), device=device)
    # 返回一个 ErrorInput 对象生成器，包含一个 SampleInput 对象
    yield ErrorInput(
        SampleInput(zero_d, args=(zero_d,)),
        error_type=RuntimeError,
        error_regex="at least 2 dimensions",
    )


def error_inputs_lstsq_grad_oriented(op_info, device, **kwargs):
    # 创建一个标量张量，形状为空元组，随机生成，指定设备
    zero_d = torch.randn((), device=device)
    # 返回一个 ErrorInput 对象生成器，包含一个 SampleInput 对象，参数为 (zero_d, None)
    yield ErrorInput(
        SampleInput(zero_d, args=(zero_d, None)),
        error_type=RuntimeError,
        error_regex="at least 2 dimensions",
    )


def sample_inputs_diagonal_diag_embed(op_info, device, dtype, requires_grad, **kwargs):
    # 创建一个部分应用了 make_tensor 函数的函数 make_arg
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )

    # 定义用于 2D 张量的形状
    shapes_2d = ((S, S), (3, 5), (5, 3))

    # 定义用于 3D 张量的形状
    shapes_3d = ((S, S, S),)

    # 定义用于 2D 张量的关键字参数
    kwargs_2d = (dict(), dict(offset=2), dict(offset=2), dict(offset=1))

    # 定义用于 3D 张量的关键字参数
    kwargs_3d = (
        dict(offset=1, dim1=1, dim2=2),
        dict(offset=2, dim1=0, dim2=1),
        dict(offset=-2, dim1=0, dim2=1),
    )

    # 遍历 2D 和 3D 形状及其关键字参数的笛卡尔积，并返回 SampleInput 对象生成器
    for shape, kwarg in chain(
        product(shapes_2d, kwargs_2d), product(shapes_3d, kwargs_3d)
    ):
        yield SampleInput(make_arg(shape), kwargs=kwarg)


def error_inputs_diagonal_diag_embed(op_info, device, **kwargs):
    # 创建一个部分应用了 make_tensor 函数的函数 make_arg，指定设备为输入设备，数据类型为 float32
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)

    # 定义用于 1D 张量的形状
    shapes1d = (0, 1, (0,), (1,))

    # 定义用于 2D 张量的形状
    shapes2d = ((M, L),)

    # 定义用于 3D 张量的形状
    shapes3d = ((M, S, L),)

    # 定义用于 1D 张量的关键字参数
    kwargs1d = {}

    # 定义用于 2D 张量的关键字参数
    kwargs2d = (
        # dim1 == dim2 不被允许
        dict(dim1=1, dim2=1),
        # 超出边界的维度不被允许
        dict(dim1=10000),
        dict(dim2=10000),
    )

    # 与 2D 张量的关键字参数相同，用于 3D 张量的关键字参数
    kwargs3d = kwargs2d

    # 返回形状和关键字参数的笛卡尔积，作为 SampleInput 对象生成器
    samples1d = product(shapes1d, kwargs1d)
    samples2d = product(shapes2d, kwargs2d)
    samples3d = product(shapes3d, kwargs3d)
    # 从三个样本生成器中依次取出形状和参数
    for shape, kwargs in chain(samples1d, samples2d, samples3d):
        # 根据形状创建输入参数对象
        arg = make_arg(shape)
        # 创建包含输入参数和关键字参数的样本输入对象
        sample = SampleInput(input=arg, kwargs=kwargs)

        # 从关键字参数中获取维度信息
        dim1 = kwargs.get("dim1")
        dim2 = kwargs.get("dim2")

        # 根据操作名检查是否包含 "diagonal"
        if "diagonal" in op_info.name:
            # 如果操作名包含 "diagonal"，获取参数的维度
            num_dim = arg.dim()
        elif op_info.name in ("diag_embed", "_refs.diag_embed"):
            # 如果操作名是 "diag_embed" 或 "_refs.diag_embed"，检查特定的形状条件
            if shape in ((0,), (1,)):
                # 如果形状为 (0,) 或 (1,)，跳过当前循环
                continue
            # 否则，计算参数的维度加一
            num_dim = arg.dim() + 1
        else:
            # 如果未匹配到任何操作名，抛出运行时错误
            raise RuntimeError("should be unreachable")

        # 计算维度的边界范围
        bound1 = -num_dim
        bound2 = num_dim - 1
        dim_range = range(bound1, bound2 + 1)

        # 检查维度1和维度2是否超出范围
        dim1_cond = dim1 and dim1 not in dim_range
        dim2_cond = dim2 and dim2 not in dim_range

        # 如果维度1和维度2相同，抛出错误
        if dim1 == dim2:
            err = f"diagonal dimensions cannot be identical {dim1}, {dim2}"
            yield ErrorInput(sample, error_regex=err, error_type=RuntimeError)
        # 如果维度1或维度2超出范围，抛出索引错误
        elif dim1_cond or dim2_cond:
            err_dim = dim1 if dim1_cond else dim2
            err = (
                r"Dimension out of range \(expected to be in range of "
                rf"\[{bound1}, {bound2}\], but got {err_dim}\)"
            )
            yield ErrorInput(sample, error_regex=err, error_type=IndexError)
        else:
            # 如果未匹配到任何条件，抛出运行时错误
            raise RuntimeError("should be unreachable")
def sample_inputs_linalg_cholesky(
    op_info, device, dtype, requires_grad=False, **kwargs
):
    """
    This function generates always positive-definite input for torch.linalg.cholesky using
    random_hermitian_pd_matrix.
    The input is generated as the itertools.product of 'batches' and 'ns'.
    In total this function generates 8 SampleInputs
    'batches' cases include:
        () - single input,
        (0,) - zero batched dimension,
        (2,) - batch of two matrices,
        (1, 1) - 1x1 batch of matrices
    'ns' gives 0x0 and 5x5 matrices.
    Zeros in dimensions are edge cases in the implementation and important to test for in order to avoid unexpected crashes.
    """
    from torch.testing._internal.common_utils import random_hermitian_pd_matrix

    # Define different batch sizes and matrix sizes to iterate over
    batches = [(), (0,), (2,), (1, 1)]
    ns = [5, 0]
    # Iterate over the product of batches, ns, and upper triangle flag
    for batch, n, upper in product(batches, ns, [True, False]):
        # Generate a random positive-definite Hermitian matrix of size n x n
        a = random_hermitian_pd_matrix(n, *batch, dtype=dtype, device=device)
        # Set requires_grad attribute based on the input argument
        a.requires_grad = requires_grad
        # Yield a SampleInput object with the generated matrix 'a' and upper triangle flag 'upper'
        yield SampleInput(a, upper=upper)


def sample_inputs_linalg_eig(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates input for torch.linalg.eig
    """

    # Output processing function to return the eigenvalues and absolute values of eigenvectors
    def out_fn(output):
        return output[0], abs(output[1])

    # Generate samples using sample_inputs_linalg_invertible function
    samples = sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad)
    # Iterate over generated samples
    for sample in samples:
        # Assign the output processing function to handle gradients
        sample.output_process_fn_grad = out_fn
        # Yield each processed sample
        yield sample


def sample_inputs_linalg_eigh(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates input for torch.linalg.eigh/eigvalsh with UPLO="U" or "L" keyword argument.
    """

    # Output processing function based on the type of output (eigh or eigvalsh)
    def out_fn(output):
        if isinstance(output, tuple):
            # Handle output from eigh function (eigenvalues and eigenvectors)
            return output[0], abs(output[1])
        else:
            # Handle output from eigvalsh function (only eigenvalues)
            return output

    # Generate samples using sample_inputs_linalg_invertible function
    samples = sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad)
    # Iterate over generated samples
    for sample in samples:
        # Set UPLO keyword argument randomly to "L" or "U"
        sample.kwargs = {"UPLO": random.choice(["L", "U"])}
        # Assign the output processing function to handle gradients
        sample.output_process_fn_grad = out_fn
        # Yield each processed sample
        yield sample


def sample_inputs_linalg_pinv(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates input for torch.linalg.pinv with hermitian=False keyword argument.
    """
    # Iterate over invertible samples generated by sample_inputs_linalg_invertible function
    for o in sample_inputs_linalg_invertible(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        # Determine real_dtype based on whether dtype is complex
        real_dtype = o.input.real.dtype if dtype.is_complex else dtype
        # Iterate over different relative tolerance (rtol) values
        for rtol in (None, 1.0, torch.tensor(1.0, dtype=real_dtype, device=device)):
            # Clone the current sample
            o = clone_sample(o)
            # Set rtol keyword argument
            o.kwargs = {"rtol": rtol}
            # Yield the modified sample
            yield o


def sample_inputs_linalg_pinv_hermitian(
    op_info, device, dtype, requires_grad=False, **kwargs


    # 接受多个参数：op_info, device, dtype，并且可以选择性地接受 requires_grad 和其他关键字参数
def sample_inputs_linalg_pinv_with_hermitian_keyword_argument(
    op_info, device, dtype, requires_grad=False, **kwargs
):
    """
    This function generates input for torch.linalg.pinv with hermitian=True keyword argument.
    """
    # Generate sample inputs using the sample_inputs_linalg_invertible function
    for o in sample_inputs_linalg_invertible(
        op_info, device, dtype, requires_grad, **kwargs
    ):
        # Set kwargs to include 'hermitian': True
        o.kwargs = {"hermitian": True}
        # Yield the modified SampleInput object
        yield o


def sample_inputs_linalg_solve(
    op_info, device, dtype, requires_grad=False, vector_rhs_allowed=True, **kwargs
):
    """
    This function generates always solvable input for torch.linalg.solve
    We sample a fullrank square matrix (i.e. invertible) A
    The first input to torch.linalg.solve is generated as the itertools.product of 'batches' and 'ns'.
    The second input is generated as the product of 'batches', 'ns' and 'nrhs'.
    In total this function generates 18 SampleInputs
    'batches' cases include:
        () - single input,
        (0,) - zero batched dimension,
        (2,) - batch of two matrices.
    'ns' gives 0x0 and 5x5 matrices.
    and 'nrhs' controls the number of vectors to solve for:
        () - using 1 as the number of vectors implicitly
        (1,) - same as () but explicit
        (3,) - solve for 3 vectors.
    Zeros in dimensions are edge cases in the implementation and important to test for in order to avoid unexpected crashes.
    'vector_rhs_allowed' controls whether to include nrhs = () to the list of SampleInputs.
    torch.solve / triangular_solve / cholesky_solve (opposed to torch.linalg.solve) do not allow
    1D tensors (vectors) as the right-hand-side.
    Once torch.solve / triangular_solve / cholesky_solve and its testing are removed,
    'vector_rhs_allowed' may be removed here as well.
    """
    # Define helper functions for creating matrices and tensors
    make_fullrank = make_fullrank_matrices_with_distinct_singular_values
    make_a = partial(
        make_fullrank, dtype=dtype, device=device, requires_grad=requires_grad
    )
    make_b = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )

    # Define possible values for 'batches', 'ns', and 'nrhs'
    batches = [(), (0,), (2,)]
    ns = [5, 0]
    if vector_rhs_allowed:
        nrhs = [(), (1,), (3,)]
    else:
        nrhs = [(1,), (3,)]

    # Generate combinations of 'ns', 'batches', and 'nrhs'
    for n, batch, rhs in product(ns, batches, nrhs):
        # Yield SampleInput objects with matrices generated using make_a and tensors using make_b
        yield SampleInput(make_a(*batch, n, n), args=(make_b(batch + (n,) + rhs),))


def sample_inputs_linalg_solve_triangular(
    op_info, device, dtype, requires_grad=False, **kwargs
):
    """
    This function generates inputs for torch.linalg.solve_triangular
    'bs' includes batch dimensions,
    'ns' controls matrix size including edge cases like 0x0 matrix,
    and 'ks' controls the matrix dimensions in the product 'left', 'upper', 'uni' are used to include upper, lower, and unit triangular matrices.
    """
    # Partial function to create tensors with specific dtype and device
    make_arg = partial(make_tensor, dtype=dtype, device=device)
    # Define possible values for batch sizes, matrix sizes, and triangular matrix types
    bs = (1, 2, 0)
    ns = (3, 0)
    ks = (1, 3, 0)

    # Generate combinations of 'bs', 'ns', 'ks', and (left, upper, uni)
    for b, n, k, (left, upper, uni) in product(
        bs, ns, ks, product((True, False), repeat=3)
    ):
        # Yield SampleInput objects with tensors generated using make_arg
        yield SampleInput(make_arg(b, n, k, left, upper, uni))
    ):
        if b == 1:
            A = make_arg((n, n)) if left else make_arg((k, k))
            B = make_arg((n, k))
        else:
            A = make_arg((b, n, n)) if left else make_arg((b, k, k))
            B = make_arg((b, n, k))
        if uni:
            # 对角线元素设为1，保持一致性而写
            A.diagonal(0, -2, -1).fill_(1.0)
        else:
            d = A.diagonal(0, -2, -1)
            d[d.abs() < 1e-6] = 1.0  # 绝对值小于1e-6的对角线元素设为1.0
        if upper:
            A.triu_()  # 上三角矩阵化
        else:
            A.tril_()  # 下三角矩阵化
        kwargs = {"upper": upper, "left": left, "unitriangular": uni}
        if requires_grad:
            for grad_A, grad_B in product((True, False), repeat=2):
                # A或B至少有一个需要梯度
                if not grad_A and not grad_B:
                    continue
                yield SampleInput(
                    A.clone().requires_grad_(grad_A),  # 克隆A并设置是否需要梯度
                    args=(B.clone().requires_grad_(grad_B),),  # 克隆B并设置是否需要梯度
                    kwargs=kwargs,
                )
        else:
            yield SampleInput(A, args=(B,), kwargs=kwargs)
# 生成适用于旧版求解函数的始终可解输入数据
# （即不在 torch.linalg 模块中的函数）。
# 与 sample_inputs_linalg_solve 不同，这里的 A x = b 方程中的 b 必须满足 b.ndim >= 2，不允许向量。
# 同时参数顺序与其相反。
def sample_inputs_legacy_solve(op_info, device, dtype, requires_grad=False, **kwargs):
    # 调用 sample_inputs_linalg_solve 生成的输入数据，禁止向量类型的右侧 b
    out = sample_inputs_linalg_solve(
        op_info, device, dtype, requires_grad=requires_grad, vector_rhs_allowed=False
    )

    # 定义输出处理函数，返回输出的第一个元素
    def out_fn(output):
        return output[0]

    # 反转张量的顺序
    for sample in out:
        # 交换输入和参数的顺序，使得 sample.input 变为第一个元素，sample.args 变为其余参数的元组
        sample.input, sample.args = sample.args[0], (sample.input,)
        # 如果 op_info.name 为 "solve"，则设置 sample.output_process_fn_grad 为 out_fn 函数
        if op_info.name == "solve":
            sample.output_process_fn_grad = out_fn
        # 生成每个修改后的 sample
        yield sample


# 生成适用于 linalg.lu_factor 的输入数据
# full_rank 变量表示是否为 full rank 的 LU 分解
def sample_inputs_linalg_lu(op_info, device, dtype, requires_grad=False, **kwargs):
    full_rank = op_info.name == "linalg.lu_factor"
    # 根据 full_rank 的不同选择生成张量的函数
    make_fn = (
        make_tensor
        if not full_rank
        else make_fullrank_matrices_with_distinct_singular_values
    )
    # 部分应用 make_fn 函数，固定 dtype、device 和 requires_grad 参数
    make_arg = partial(make_fn, dtype=dtype, device=device, requires_grad=requires_grad)

    # 定义输出处理函数，根据 op_info.name 的不同返回不同的输出元素
    def out_fn(output):
        if op_info.name == "linalg.lu":
            return output[1], output[2]
        else:
            return output

    # 针对不同的 batch_shapes、pivot 和 delta 参数组合生成输入数据
    batch_shapes = ((), (3,), (3, 3))
    # 在 CUDA 设备上支持 pivot=False
    pivots = (True, False) if torch.device(device).type == "cuda" else (True,)
    deltas = (-2, -1, 0, +1, +2)
    for batch_shape, pivot, delta in product(batch_shapes, pivots, deltas):
        # 根据给定的 batch_shape、pivot 和 delta 生成具体的 shape 参数
        shape = batch_shape + (S + delta, S)
        # 根据 shape 调用 make_arg 生成张量 A
        A = make_arg(shape) if not full_rank else make_arg(*shape)
        # 生成 SampleInput，传入参数 pivot 和 output_process_fn_grad=out_fn
        yield SampleInput(A, kwargs={"pivot": pivot}, output_process_fn_grad=out_fn)


# 生成适用于 linalg_svdvals 的输入数据
def sample_inputs_linalg_svdvals(op_info, device, dtype, requires_grad=False, **kwargs):
    # 部分应用 make_tensor 函数，固定 dtype、device 和 requires_grad 参数
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    # 定义不同的 batches 和 ns 参数组合
    batches = [(), (0,), (2,), (1, 1)]
    ns = [5, 2, 0]

    # 针对 batches 和 ns 的组合生成输入数据
    for batch, m, n in product(batches, ns, ns):
        # 根据给定的 batch、m 和 n 生成具体的张量
        yield SampleInput(make_arg(batch + (m, n)))


# 生成适用于 linalg_qr_geqrf 的输入数据
def sample_inputs_linalg_qr_geqrf(op_info, device, dtype, requires_grad=False, **kwargs):
    # 仅当矩阵为 full rank 时 QR 分解才有意义
    make_fullrank = make_fullrank_matrices_with_distinct_singular_values
    # 部分应用 make_fullrank 函数，固定 dtype、device 和 requires_grad 参数
    make_arg = partial(make_fullrank, dtype=dtype, device=device, requires_grad=requires_grad)

    # 定义不同的 batches 和 ns 参数组合
    batches = [(), (0,), (2,), (1, 1)]
    ns = [5, 2, 0]

    # 针对 batches 和 ns 的组合生成输入数据
    for batch, (m, n) in product(batches, product(ns, ns)):
        shape = batch + (m, n)
        # 根据给定的 shape 生成具体的张量 A
        yield SampleInput(make_arg(*shape))


# 生成适用于 tensorsolve 的输入数据
def sample_inputs_tensorsolve(op_info, device, dtype, requires_grad, **kwargs):
    # 定义不同的 a_shapes 参数组合
    a_shapes = [(2, 3, 6), (3, 4, 4, 3)]
    # 目前 NumPy 不支持零维张量，因此跳过零维张量的生成
    # 对于每个 a_shape，生成相应的张量输入数据
    # NumPy is used in reference check tests.
    # See https://github.com/numpy/numpy/pull/20482 for tracking NumPy bugfix.
    # 定义了一个固定的列表，用于测试参考检查。
    dimss = [None, (0, 2)]

    # 使用偏函数创建一个函数 make_arg，用于创建张量。
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, requires_grad=requires_grad
    )

    # 使用 itertools.product 对 a_shapes 和 dimss 的所有组合进行迭代。
    for a_shape, dims in itertools.product(a_shapes, dimss):
        # 使用 make_arg 创建张量 a，其形状为 a_shape。
        a = make_arg(a_shape)
        # 使用 make_arg 创建张量 b，其形状为 a_shape 的前两个维度。
        b = make_arg(a_shape[:2])
        # 生成一个 SampleInput 实例，包括张量 a、b 和维度 dims。
        yield SampleInput(a, b, dims=dims)
# 定义函数 sample_inputs_tensorinv，生成用于测试的输入数据
def sample_inputs_tensorinv(op_info, device, dtype, requires_grad, **kwargs):
    # make_arg 是一个函数 make_fullrank_matrices_with_distinct_singular_values 的别名
    make_arg = make_fullrank_matrices_with_distinct_singular_values

    # 定义内部函数 make_input，返回一个根据参数生成的张量
    def make_input():
        return make_arg(12, 12, device=device, dtype=dtype, requires_grad=requires_grad)

    # 定义输入数据的形状列表，每个元素包含左右张量的形状
    shapes = [
        ((2, 2, 3), (12, 1)),   # 第一组形状
        ((4, 3), (6, 1, 2)),    # 第二组形状
    ]

    # 遍历形状列表中的每一个形状对
    for shape_lhs, shape_rhs in shapes:
        # 生成输入张量，并按照给定形状重新整形，然后断开与计算图的连接
        inp = make_input().reshape(*shape_lhs, *shape_rhs).detach()
        # 根据 requires_grad 设置是否需要梯度
        inp.requires_grad_(requires_grad)
        # 使用 yield 返回一个 SampleInput 对象，包含 inp 和形状列表中左张量的索引
        yield SampleInput(inp, ind=len(shape_lhs))


# 定义 op_db 变量，包含 OpInfo 对象的列表
op_db: List[OpInfo] = [
    # 第一个 OpInfo 对象，代表 linalg.cross 操作
    OpInfo(
        "linalg.cross",  # 操作名称
        ref=lambda x, y, dim=-1: np.cross(x, y, axis=dim),  # 参考实现
        op=torch.linalg.cross,  # PyTorch 中对应的操作
        dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),  # 支持的数据类型
        aten_name="linalg_cross",  # 对应的 ATen 函数名称
        sample_inputs_func=sample_inputs_cross,  # 生成示例输入的函数
        error_inputs_func=error_inputs_cross,  # 生成错误输入的函数
        supports_out=True,  # 是否支持输出张量
        supports_fwgrad_bwgrad=True,  # 是否支持前向和反向梯度
        supports_forward_ad=True,  # 是否支持自动微分的前向传播
        skips=(  # 被跳过的测试信息
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),  # 跳过的具体信息
                "TestCommon",  # 测试的类名
                "test_numpy_ref_mps",  # 测试函数名
            ),
        ),
    ),
    # 第二个 OpInfo 对象，代表 linalg.det 操作
    OpInfo(
        "linalg.det",  # 操作名称
        aten_name="linalg_det",  # 对应的 ATen 函数名称
        op=torch.linalg.det,  # PyTorch 中对应的操作
        aliases=("det",),  # 别名列表
        dtypes=floating_and_complex_types(),  # 支持的数据类型
        supports_forward_ad=True,  # 是否支持自动微分的前向传播
        supports_fwgrad_bwgrad=True,  # 是否支持前向和反向梯度
        sample_inputs_func=sample_inputs_linalg_det_logdet_slogdet,  # 生成示例输入的函数
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],  # 装饰器列表
        check_batched_gradgrad=False,  # 是否检查批量梯度的二阶梯度
    ),
]
    OpInfo(
        "linalg.det",  # 操作名称为 "linalg.det"
        aten_name="linalg_det",  # 对应的 ATen 函数名称为 "linalg_det"
        op=torch.linalg.det,  # 使用 torch.linalg.det 函数进行操作
        variant_test_name="singular",  # 测试的变体名称为 "singular"
        aliases=("det",),  # 别名包括 "det"
        dtypes=floating_and_complex_types(),  # 支持浮点数和复数类型的数据
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向和后向自动微分
        check_batched_gradgrad=False,  # 不检查批处理梯度
        sample_inputs_func=sample_inputs_linalg_det_singular,  # 使用特定的函数生成示例输入
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],  # 使用的装饰器列表
        skips=(  # 跳过的测试列表
            DecorateInfo(
                unittest.skip("The backward may give different results"),  # 跳过的原因：后向可能会给出不同的结果
                "TestCommon",  # 测试类名
                "test_noncontiguous_samples",  # 测试方法名
            ),
            DecorateInfo(
                unittest.skip("Gradients are incorrect on macos"),  # 跳过的原因：在 macOS 上梯度不正确
                "TestBwdGradients",  # 测试类名
                "test_fn_grad",  # 测试方法名
                device_type="cpu",  # 适用于 CPU 设备
                dtypes=(torch.float64,),  # 数据类型为 torch.float64
                active_if=IS_MACOS,  # 仅在 macOS 上生效
            ),
            DecorateInfo(
                unittest.skip("Gradients are incorrect on macos"),  # 跳过的原因：在 macOS 上梯度不正确
                "TestFwdGradients",  # 测试类名
                "test_forward_mode_AD",  # 测试方法名
                device_type="cpu",  # 适用于 CPU 设备
                dtypes=(torch.float64,),  # 数据类型为 torch.float64
                active_if=IS_MACOS,  # 仅在 macOS 上生效
            ),
            # 下面的 DecorateInfo 都类似，提供了相同的结构和注释方式，描述了不同的跳过测试情况
            DecorateInfo(
                unittest.expectedFailure,
                "TestBwdGradients",
                "test_fn_gradgrad",
                dtypes=(torch.complex128,),
            ),
            DecorateInfo(
                unittest.expectedFailure,
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                dtypes=(torch.complex128,),
            ),
            DecorateInfo(
                unittest.skip("Skipped, see https://github.com//issues/84192"),
                "TestBwdGradients",
                "test_fn_gradgrad",
                device_type="cuda",
            ),
            DecorateInfo(
                unittest.skip("Skipped, see https://github.com//issues/84192"),
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                device_type="cuda",
            ),
            DecorateInfo(
                unittest.skip(
                    "Flaky on ROCm https://github.com/pytorch/pytorch/issues/93044"
                ),
                "TestBwdGradients",
                "test_fn_grad",
                device_type="cuda",
                dtypes=get_all_complex_dtypes(),
                active_if=TEST_WITH_ROCM,
            ),
            DecorateInfo(
                unittest.skip(
                    "Flaky on ROCm https://github.com/pytorch/pytorch/issues/93045"
                ),
                "TestFwdGradients",
                "test_forward_mode_AD",
                device_type="cuda",
                dtypes=get_all_complex_dtypes(),
                active_if=TEST_WITH_ROCM,
            ),
        ),
    ),
    OpInfo(
        "linalg.diagonal",
        aten_name="linalg_diagonal",
        aten_backward_name="diagonal_backward",
        dtypes=all_types_and_complex_and(
            torch.bool, torch.bfloat16, torch.float16, torch.chalf
        ),
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_diagonal_diag_embed,
        error_inputs_func=error_inputs_diagonal_diag_embed,
    ),
    OpInfo(
        "linalg.cholesky",
        aten_name="linalg_cholesky",
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        sample_inputs_func=sample_inputs_linalg_cholesky,
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.cholesky_ex",
        aten_name="linalg_cholesky_ex",
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        sample_inputs_func=sample_inputs_linalg_cholesky,
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.vecdot",
        aten_name="linalg_vecdot",
        ref=lambda x, y, *, dim=-1: (x.conj() * y).sum(dim),
        dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_linalg_vecdot,
        check_batched_forward_grad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        skips=(
            # Issue with conj and torch dispatch, see https://github.com/pytorch/pytorch/issues/82479
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestSchemaCheckModeOpInfo",
                "test_schema_correctness",
                dtypes=(torch.complex64, torch.complex128),
            ),
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
        ),
    ),



    OpInfo(
        "linalg.diagonal",
        # 定义操作名称为 "linalg.diagonal"
        aten_name="linalg_diagonal",
        # 设置 ATen 中对应的名称为 "linalg_diagonal"
        aten_backward_name="diagonal_backward",
        # 设置反向传播时的 ATen 函数名为 "diagonal_backward"
        dtypes=all_types_and_complex_and(
            torch.bool, torch.bfloat16, torch.float16, torch.chalf
        ),
        # 指定操作支持的数据类型，包括所有标准类型和复数类型
        supports_out=False,
        # 指示操作不支持输出参数
        supports_forward_ad=True,
        # 指示操作支持前向自动微分
        supports_fwgrad_bwgrad=True,
        # 指示操作支持前向和后向梯度传播
        sample_inputs_func=sample_inputs_diagonal_diag_embed,
        # 设置用于生成示例输入的函数为 sample_inputs_diagonal_diag_embed
        error_inputs_func=error_inputs_diagonal_diag_embed,
        # 设置用于生成错误输入的函数为 error_inputs_diagonal_diag_embed
    ),
    OpInfo(
        "linalg.cholesky",
        # 定义操作名称为 "linalg.cholesky"
        aten_name="linalg_cholesky",
        # 设置 ATen 中对应的名称为 "linalg_cholesky"
        dtypes=floating_and_complex_types(),
        # 指定操作支持的数据类型为浮点数和复数类型
        supports_forward_ad=True,
        # 指示操作支持前向自动微分
        supports_fwgrad_bwgrad=True,
        # 指示操作支持前向和后向梯度传播
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        # 关闭批处理前向梯度检查
        sample_inputs_func=sample_inputs_linalg_cholesky,
        # 设置用于生成示例输入的函数为 sample_inputs_linalg_cholesky
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        # 设置用于梯度检查的包装器函数为 gradcheck_wrapper_hermitian_input
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        # 添加装饰器列表，条件是如果没有 CUDA 的 Magma 和 Cusolver 或者没有 CPU 的 Lapack，则跳过
    ),
    OpInfo(
        "linalg.cholesky_ex",
        # 定义操作名称为 "linalg.cholesky_ex"
        aten_name="linalg_cholesky_ex",
        # 设置 ATen 中对应的名称为 "linalg_cholesky_ex"
        dtypes=floating_and_complex_types(),
        # 指定操作支持的数据类型为浮点数和复数类型
        supports_forward_ad=True,
        # 指示操作支持前向自动微分
        supports_fwgrad_bwgrad=True,
        # 指示操作支持前向和后向梯度传播
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        # 关闭批处理前向梯度检查
        sample_inputs_func=sample_inputs_linalg_cholesky,
        # 设置用于生成示例输入的函数为 sample_inputs_linalg_cholesky
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        # 设置用于梯度检查的包装器函数为 gradcheck_wrapper_hermitian_input
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        # 添加装饰器列表，条件是如果没有 CUDA 的 Magma 和 Cusolver 或者没有 CPU 的 Lapack，则跳过
    ),
    OpInfo(
        "linalg.vecdot",
        # 定义操作名称为 "linalg.vecdot"
        aten_name="linalg_vecdot",
        # 设置 ATen 中对应的名称为 "linalg_vecdot"
        ref=lambda x, y, *, dim=-1: (x.conj() * y).sum(dim),
        # 定义操作的参考实现为计算向量内积
        dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16),
        # 指定操作支持的数据类型为浮点数、复数以及 torch.half 和 torch.bfloat16
        sample_inputs_func=sample_inputs_linalg_vecdot,
        # 设置用于生成示例输入的函数为 sample_inputs_linalg_vecdot
        check_batched_forward_grad=False,
        # 关闭批处理前向梯度检查
        supports_forward_ad=True,
        # 指示操作支持前向自动微分
        supports_fwgrad_bwgrad=True,
        # 指示操作支持前向和后向梯度传播
        skips=(
            # Issue with conj and torch dispatch, see https://github.com/pytorch/pytorch/issues/82479
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestSchemaCheckModeOpInfo",
                "test_schema_correctness",
                dtypes=(torch.complex64, torch.complex128),
            ),
            # 添加跳过的装饰信息，用于指定在某些条件下跳过测试
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
        ),
    ),
    OpInfo(
        "linalg.cond",  # 操作名称为 "linalg.cond"
        aten_name="linalg_cond",  # 对应的 ATen 操作名称为 "linalg_cond"
        dtypes=floating_and_complex_types(),  # 数据类型为浮点数和复数类型的集合
        sample_inputs_func=sample_inputs_linalg_cond,  # 获取样本输入的函数为 sample_inputs_linalg_cond
        check_batched_gradgrad=False,  # 不检查批处理梯度梯度
        check_batched_forward_grad=False,  # 不检查批处理前向传播梯度
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向梯度与后向梯度
        gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,  # 梯度检查的非确定性容差为 GRADCHECK_NONDET_TOL
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],  # 使用的装饰器列表
        skips=(  # 跳过测试的装饰信息列表
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip 函数跳过测试
                "TestFakeTensor",  # 测试类名为 "TestFakeTensor"
                "test_fake_crossref_backward_amp",  # 测试方法名为 "test_fake_crossref_backward_amp"
                device_type="cuda",  # 设备类型为 "cuda"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
                active_if=TEST_WITH_ROCM,  # 在 TEST_WITH_ROCM 激活时生效
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip 函数跳过测试
                "TestFakeTensor",  # 测试类名为 "TestFakeTensor"
                "test_fake_crossref_backward_no_amp",  # 测试方法名为 "test_fake_crossref_backward_no_amp"
                device_type="cuda",  # 设备类型为 "cuda"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
                active_if=TEST_WITH_ROCM,  # 在 TEST_WITH_ROCM 激活时生效
            ),
        ),
    ),
    OpInfo(
        "linalg.eig",  # 操作名称为 "linalg.eig"
        aten_name="linalg_eig",  # 对应的 ATen 操作名称为 "linalg_eig"
        op=torch.linalg.eig,  # 使用 torch.linalg.eig 作为操作函数
        dtypes=floating_and_complex_types(),  # 数据类型为浮点数和复数类型的集合
        sample_inputs_func=sample_inputs_linalg_eig,  # 获取样本输入的函数为 sample_inputs_linalg_eig
        check_batched_forward_grad=False,  # 不检查批处理前向传播梯度
        check_batched_grad=False,  # 不检查批处理梯度
        check_batched_gradgrad=False,  # 不检查批处理梯度梯度
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向梯度与后向梯度
        skips=(  # 跳过测试的装饰信息列表
            # AssertionError: Scalars are not equal!
            DecorateInfo(
                unittest.expectedFailure,  # 预期测试失败
                "TestCommon",  # 测试类名为 "TestCommon"
                "test_out",  # 测试方法名为 "test_out"
                device_type="cpu",  # 设备类型为 "cpu"
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip 函数跳过测试
                "TestCommon",  # 测试类名为 "TestCommon"
                "test_out",  # 测试方法名为 "test_out"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip 函数跳过测试
                "TestCommon",  # 测试类名为 "TestCommon"
                "test_variant_consistency_eager",  # 测试方法名为 "test_variant_consistency_eager"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip 函数跳过测试
                "TestJit",  # 测试类名为 "TestJit"
                "test_variant_consistency_jit",  # 测试方法名为 "test_variant_consistency_jit"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
        ),
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack, with_tf32_off],  # 使用的装饰器列表
    ),
    OpInfo(
        "linalg.eigvals",  # 操作名称为 'linalg.eigvals'
        aten_name="linalg_eigvals",  # ATen函数名称为 'linalg_eigvals'
        op=torch.linalg.eigvals,  # 使用 torch.linalg.eigvals() 函数
        dtypes=floating_and_complex_types(),  # 数据类型为浮点数和复数类型的集合
        sample_inputs_func=sample_inputs_linalg_invertible,  # 用于生成样本输入的函数为 sample_inputs_linalg_invertible
        check_batched_forward_grad=False,  # 不检查批处理前向梯度
        check_batched_grad=False,  # 不检查批处理梯度
        check_batched_gradgrad=False,  # 不检查批处理二阶梯度
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],  # 装饰器包括 skipCUDAIfNoMagma 和 skipCPUIfNoLapack
        skips=(  # 跳过以下测试
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip("Skipped!") 跳过测试
                "TestCommon",  # 测试类名为 "TestCommon"
                "test_out",  # 测试方法名为 "test_out"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip("Skipped!") 跳过测试
                "TestCommon",  # 测试类名为 "TestCommon"
                "test_variant_consistency_eager",  # 测试方法名为 "test_variant_consistency_eager"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip("Skipped!") 跳过测试
                "TestJit",  # 测试类名为 "TestJit"
                "test_variant_consistency_jit",  # 测试方法名为 "test_variant_consistency_jit"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
        ),
    ),
    OpInfo(
        "linalg.eigh",  # 操作名称为 'linalg.eigh'
        aten_name="linalg_eigh",  # ATen函数名称为 'linalg_eigh'
        dtypes=floating_and_complex_types(),  # 数据类型为浮点数和复数类型的集合
        sample_inputs_func=sample_inputs_linalg_eigh,  # 用于生成样本输入的函数为 sample_inputs_linalg_eigh
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,  # 使用 gradcheck_wrapper_hermitian_input 包装的梯度检查器
        check_batched_forward_grad=False,  # 不检查批处理前向梯度
        check_batched_grad=False,  # 不检查批处理梯度
        check_batched_gradgrad=False,  # 不检查批处理二阶梯度
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack, with_tf32_off],  # 装饰器包括 skipCUDAIfNoMagma, skipCPUIfNoLapack 和 with_tf32_off
        skips=(  # 跳过以下测试
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip("Skipped!") 跳过测试
                "TestCommon",  # 测试类名为 "TestCommon"
                "test_out",  # 测试方法名为 "test_out"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip("Skipped!") 跳过测试
                "TestCommon",  # 测试类名为 "TestCommon"
                "test_variant_consistency_eager",  # 测试方法名为 "test_variant_consistency_eager"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip("Skipped!") 跳过测试
                "TestJit",  # 测试类名为 "TestJit"
                "test_variant_consistency_jit",  # 测试方法名为 "test_variant_consistency_jit"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
        ),
    ),
    OpInfo(
        "linalg.eigvalsh",  # 定义操作信息对象，指定操作名为 "linalg.eigvalsh"
        aten_name="linalg_eigvalsh",  # 设置 ATen 名称为 "linalg_eigvalsh"
        dtypes=floating_and_complex_types(),  # 操作支持的数据类型为浮点数和复数类型
        sample_inputs_func=sample_inputs_linalg_eigh,  # 获取示例输入函数为 sample_inputs_linalg_eigh
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,  # 使用 gradcheck_wrapper_hermitian_input 进行梯度检查包装
        check_batched_forward_grad=False,  # 禁用批处理前向梯度检查
        check_batched_grad=False,  # 禁用批处理梯度检查
        check_batched_gradgrad=False,  # 禁用批处理梯度二阶检查
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和后向梯度
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],  # 使用修饰器进行条件跳过检查
        skips=(  # 跳过的测试条件列表
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest 跳过测试
                "TestCommon",  # 测试类为 "TestCommon"
                "test_out",  # 测试方法为 "test_out"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest 跳过测试
                "TestCommon",  # 测试类为 "TestCommon"
                "test_variant_consistency_eager",  # 测试方法为 "test_variant_consistency_eager"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest 跳过测试
                "TestJit",  # 测试类为 "TestJit"
                "test_variant_consistency_jit",  # 测试方法为 "test_variant_consistency_jit"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
        ),
    ),
    OpInfo(
        "linalg.householder_product",  # 定义操作信息对象，指定操作名为 "linalg.householder_product"
        aten_name="linalg_householder_product",  # 设置 ATen 名称为 "linalg_householder_product"
        op=torch.linalg.householder_product,  # 操作为 torch.linalg.householder_product
        aliases=("orgqr",),  # 别名为 "orgqr"
        dtypes=floating_and_complex_types(),  # 操作支持的数据类型为浮点数和复数类型
        gradcheck_fast_mode=True,  # 使用快速模式进行梯度检查
        check_batched_grad=False,  # 禁用批处理梯度检查
        check_batched_gradgrad=False,  # 禁用批处理梯度二阶检查
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和后向梯度
        check_batched_forward_grad=False,  # 禁用批处理前向梯度检查
        sample_inputs_func=sample_inputs_householder_product,  # 获取示例输入函数为 sample_inputs_householder_product
        decorators=[  # 使用修饰器进行条件跳过检查
            skipCUDAIfNoCusolver,
            skipCPUIfNoLapack,
            DecorateInfo(
                toleranceOverride({torch.complex64: tol(atol=1e-3, rtol=1e-3)}),  # 设置 torch.complex64 类型的容差
            ),
            DecorateInfo(
                unittest.skip("Skipped! Flaky"),  # 使用 unittest 跳过测试，注明测试不稳定
                "TestFwdGradients",  # 测试类为 "TestFwdGradients"
                "test_fn_fwgrad_bwgrad",  # 测试方法为 "test_fn_fwgrad_bwgrad"
                device_type="cpu",  # 设备类型为 "cpu"
                dtypes=(torch.complex128,),  # 数据类型为 torch.complex128
            ),
        ],
    ),
    OpInfo(
        "linalg.ldl_factor",  # 定义操作信息对象，指定操作名为 "linalg.ldl_factor"
        aten_name="linalg_ldl_factor",  # 设置 ATen 名称为 "linalg_ldl_factor"
        dtypes=floating_and_complex_types(),  # 操作支持的数据类型为浮点数和复数类型
        supports_autograd=False,  # 不支持自动求导
        sample_inputs_func=sample_inputs_linalg_ldl_factor,  # 获取示例输入函数为 sample_inputs_linalg_ldl_factor
        decorators=[skipCUDAIfNoMagmaAndNoLinalgsolver, skipCPUIfNoLapack],  # 使用修饰器进行条件跳过检查
    ),
    OpInfo(
        "linalg.ldl_factor_ex",  # 定义操作信息对象，指定操作名为 "linalg.ldl_factor_ex"
        aten_name="linalg_ldl_factor_ex",  # 设置 ATen 名称为 "linalg_ldl_factor_ex"
        dtypes=floating_and_complex_types(),  # 操作支持的数据类型为浮点数和复数类型
        supports_autograd=False,  # 不支持自动求导
        sample_inputs_func=sample_inputs_linalg_ldl_factor,  # 获取示例输入函数为 sample_inputs_linalg_ldl_factor
        decorators=[skipCUDAIfNoMagmaAndNoLinalgsolver, skipCPUIfNoLapack],  # 使用修饰器进行条件跳过检查
    ),
    OpInfo(
        "linalg.ldl_solve",  # 定义操作名称为'linalg.ldl_solve'
        aten_name="linalg_ldl_solve",  # 设置 ATen 函数名称为 'linalg_ldl_solve'
        dtypes=floating_and_complex_types(),  # 指定支持的数据类型为浮点数和复数类型
        supports_autograd=False,  # 不支持自动微分
        sample_inputs_func=sample_inputs_linalg_ldl_solve,  # 设置样本输入函数为 sample_inputs_linalg_ldl_solve
        decorators=[
            skipCUDAIf(  # 添加装饰器列表，用于条件跳过测试
                _get_torch_cuda_version() < (11, 4), "not available before CUDA 11.3.1"
            ),
            skipCUDAIfNoCusolver,  # 跳过没有 Cusolver 的 CUDA 环境
            skipCUDAIfRocm,  # 跳过在 ROCm 环境下的测试
            skipCPUIfNoLapack,  # 跳过没有 Lapack 的 CPU 环境
        ],
    ),
    OpInfo(
        "linalg.lstsq",  # 定义操作名称为 'linalg.lstsq'
        aten_name="linalg_lstsq",  # 设置 ATen 函数名称为 'linalg_lstsq'
        dtypes=floating_and_complex_types(),  # 指定支持的数据类型为浮点数和复数类型
        supports_out=True,  # 支持输出参数
        sample_inputs_func=sample_inputs_linalg_lstsq,  # 设置样本输入函数为 sample_inputs_linalg_lstsq
        error_inputs_func=error_inputs_lstsq,  # 设置错误输入函数为 error_inputs_lstsq
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],  # 添加装饰器列表，用于条件跳过测试
        skips=(
            # 跳过梯度检查，因为它们在 'grad_oriented' 变体中测试过
            DecorateInfo(unittest.skip("Skipped!"), "TestFwdGradients"),
            DecorateInfo(unittest.skip("Skipped!"), "TestBwdGradients"),
            # 对于属性 'shape' 的值不匹配的情况，跳过测试
            DecorateInfo(unittest.skip("Skipped!"), "TestCommon", "test_out"),
            # 在 'mps' 设备上使用 'torch.float32' 数据类型的情况下跳过测试
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            # 在 'mps' 设备上使用 'torch.float32' 数据类型的情况下跳过测试
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            # 在 'mps' 设备上使用 'torch.float32' 数据类型的情况下跳过测试
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.lstsq",  # 操作名称为 linalg.lstsq
        aten_name="linalg_lstsq",  # 对应的 ATen 函数名称为 linalg_lstsq
        variant_test_name="grad_oriented",  # 使用 grad_oriented 变体进行测试
        # 定义操作：使用 torch.linalg.lstsq 计算并返回结果的第一个元素
        op=lambda a, b, driver: torch.linalg.lstsq(a, b, driver=driver)[0],
        supports_out=False,  # 不支持输出参数
        dtypes=floating_and_complex_types(),  # 支持浮点数和复数类型的数据
        sample_inputs_func=sample_inputs_linalg_lstsq,  # 使用 sample_inputs_linalg_lstsq 函数生成示例输入
        error_inputs_func=error_inputs_lstsq_grad_oriented,  # 使用 error_inputs_lstsq_grad_oriented 函数生成错误输入
        gradcheck_fast_mode=True,  # 在慢速 gradcheck 下运行速度非常慢，可以减少输入大小来替代
        supports_autograd=True,  # 支持自动微分
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向和后向自动微分
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],  # 装饰器列表，根据条件跳过 CUDA 或 CPU 测试
        skips=(
            # 下面两个测试不适用于使用 lambda 作为操作的情况
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            DecorateInfo(
                unittest.expectedFailure,
                "TestOperatorSignatures",
                "test_get_torch_func_signature_exhaustive",
            ),
        ),
    ),
    OpInfo(
        "linalg.matrix_power",  # 操作名称为 linalg.matrix_power
        aliases=("matrix_power",),  # 别名为 matrix_power
        aten_name="linalg_matrix_power",  # 对应的 ATen 函数名称为 linalg_matrix_power
        dtypes=floating_and_complex_types(),  # 支持浮点数和复数类型的数据
        # 链接到的 GitHub 问题页面：https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,  # 在慢速 gradcheck 下运行速度非常慢，可以减少输入大小来替代
        supports_inplace_autograd=False,  # 不支持原地自动微分
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向和后向自动微分
        check_batched_grad=False,  # 不检查批处理梯度
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],  # 装饰器列表，根据条件跳过 CUDA 或 CPU 测试
        sample_inputs_func=sample_inputs_linalg_matrix_power,  # 使用 sample_inputs_linalg_matrix_power 函数生成示例输入
    ),
    OpInfo(
        "linalg.multi_dot",
        # gradcheck 函数无法处理 TensorList 输入，因此需要使用此 lambda 函数
        aten_name="linalg_multi_dot",
        # 所有类型以及复数类型，包括 torch.half 和 torch.bfloat16
        dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
        # CUDA 环境下支持的浮点数和复数类型，包括 torch.half 和 torch.bfloat16
        dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
        supports_inplace_autograd=False,
        # 对空输入张量进行批次化梯度检查时会失败
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # https://github.com/pytorch/pytorch/issues/66357
        # 对批次化前向梯度进行检查时会失败
        check_batched_forward_grad=False,
        # 生成样本输入函数为 sample_inputs_linalg_multi_dot
        sample_inputs_func=sample_inputs_linalg_multi_dot,
        gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
        skips=(
            # https://github.com/pytorch/pytorch/issues/67470
            # 跳过 DecorateInfo 中指定的测试用例，原因是 Issue 67470
            DecorateInfo(
                unittest.skip("67470!"), "TestCommon", "test_noncontiguous_samples"
            ),
            # 在 XLA 上会失败
            # AssertionError: False is not true : Tensors failed to compare as equal!
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestOpInfo",
                device_type="xla",
                dtypes=(torch.long,),
            ),
            # https://github.com/pytorch/pytorch/issues/71774
            # 在 CPU 上的特定情况下会跳过此测试用例
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestNNCOpInfo",
                "test_nnc_correctness",
                device_type="cpu",
                dtypes=(torch.long,),
            ),
        ),
    ),
    # 注意：linalg.norm 有两个变体，因此可以使用不同的跳过装饰器来处理不同的样本输入
    OpInfo(
        "linalg.norm",
        aten_name="linalg_norm",
        op=torch.linalg.norm,
        # 浮点数和复数类型，包括 torch.float16 和 torch.bfloat16
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
        # 装饰器包括 skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        # 生成样本输入函数为 sample_inputs_linalg_norm
        sample_inputs_func=sample_inputs_linalg_norm,
        supports_forward_ad=True,
        # 对批次化前向梯度进行检查时会失败
        check_batched_forward_grad=False,
        supports_fwgrad_bwgrad=True,
        skips=(
            # 预期失败的测试用例
            DecorateInfo(
                unittest.expectedFailure, "TestBwdGradients", "test_fn_gradgrad"
            ),
            # 在 CUDA 上的特定情况下会跳过此测试用例
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
            # 在 CUDA 上的特定情况下会跳过此测试用例
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_no_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
        ),
    ),
    OpInfo(
        "linalg.norm",  # 操作名称是 "linalg.norm"
        op=torch.linalg.norm,  # 使用 torch 中的 linalg.norm 操作函数
        variant_test_name="subgradients_at_zero",  # 测试变体名称为 "subgradients_at_zero"
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),  # 数据类型包括浮点数和复数类型以及 torch.float16 和 torch.bfloat16
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],  # 装饰器列表，用于条件跳过测试
        sample_inputs_func=partial(
            sample_inputs_linalg_norm, variant="subgradient_at_zero"  # 使用指定变体生成样本输入的函数
        ),
        aten_name="linalg_norm",  # 对应的 ATen 操作名称是 "linalg_norm"
        supports_forward_ad=True,  # 支持正向自动求导
        check_batched_forward_grad=False,  # 不检查批处理的前向梯度
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        skips=(  # 跳过的测试用例信息列表
            DecorateInfo(
                unittest.expectedFailure, "TestBwdGradients", "test_fn_gradgrad"  # 预期失败的反向梯度测试用例
            ),
            DecorateInfo(
                unittest.expectedFailure, "TestFwdGradients", "test_fn_fwgrad_bwgrad"  # 预期失败的前向和反向梯度测试用例
            ),
            DecorateInfo(
                unittest.expectedFailure, "TestFwdGradients", "test_forward_mode_AD"  # 预期失败的前向自动求导测试用例
            ),
            DecorateInfo(
                unittest.expectedFailure, "TestBwdGradients", "test_fn_grad"  # 预期失败的反向梯度测试用例
            ),
        ),
    ),
    
    OpInfo(
        "linalg.matrix_norm",  # 操作名称是 "linalg.matrix_norm"
        aten_name="linalg_matrix_norm",  # 对应的 ATen 操作名称是 "linalg_matrix_norm"
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),  # 数据类型包括浮点数和复数类型以及 torch.float16 和 torch.bfloat16
        supports_forward_ad=True,  # 支持正向自动求导
        check_batched_forward_grad=False,  # 不检查批处理的前向梯度
        check_batched_gradgrad=False,  # 不检查批处理的梯度-梯度梯度
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],  # 装饰器列表，用于条件跳过测试
        sample_inputs_func=sample_inputs_linalg_matrix_norm,  # 使用指定函数生成样本输入
        skips=(  # 跳过的测试用例信息列表
            DecorateInfo(
                unittest.skip("Skipped!"),  # 跳过该测试用例，显示 "Skipped!"
                "TestFakeTensor",  # 测试类名为 "TestFakeTensor"
                "test_fake_crossref_backward_amp",  # 测试方法名为 "test_fake_crossref_backward_amp"
                device_type="cuda",  # 设备类型为 "cuda"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
                active_if=TEST_WITH_ROCM,  # 仅在使用 ROCm 测试时激活
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 跳过该测试用例，显示 "Skipped!"
                "TestFakeTensor",  # 测试类名为 "TestFakeTensor"
                "test_fake_crossref_backward_no_amp",  # 测试方法名为 "test_fake_crossref_backward_no_amp"
                device_type="cuda",  # 设备类型为 "cuda"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
                active_if=TEST_WITH_ROCM,  # 仅在使用 ROCm 测试时激活
            ),
        ),
    ),
    
    OpInfo(
        "linalg.qr",  # 操作名称是 "linalg.qr"
        aten_name="linalg_qr",  # 对应的 ATen 操作名称是 "linalg_qr"
        op=torch.linalg.qr,  # 使用 torch 中的 linalg.qr 操作函数
        dtypes=floating_and_complex_types(),  # 数据类型包括浮点数和复数类型
        supports_forward_ad=True,  # 支持正向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        check_batched_gradgrad=False,  # 不检查批处理的梯度-梯度梯度
        sample_inputs_func=sample_inputs_linalg_qr_geqrf,  # 使用指定函数生成样本输入
        decorators=[skipCUDAIfNoCusolver, skipCPUIfNoLapack],  # 装饰器列表，用于条件跳过测试
    ),
    OpInfo(
        "linalg.slogdet",  # 操作名称为 linalg.slogdet
        aten_name="linalg_slogdet",  # ATen 函数名为 linalg_slogdet
        op=torch.linalg.slogdet,  # 使用 torch.linalg.slogdet 实现该操作
        dtypes=floating_and_complex_types(),  # 支持浮点数和复数类型的数据
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        sample_inputs_func=sample_inputs_linalg_det_logdet_slogdet,  # 使用 sample_inputs_linalg_det_logdet_slogdet 函数生成示例输入
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],  # 装饰器列表，用于条件跳过测试
    ),
    OpInfo(
        "linalg.vander",  # 操作名称为 linalg.vander
        aten_name="linalg_vander",  # ATen 函数名为 linalg_vander
        ref=np_vander_batched,  # 参考实现为 np_vander_batched
        op=torch.linalg.vander,  # 使用 torch.linalg.vander 实现该操作
        dtypes=all_types_and_complex(),  # 支持所有类型和复数类型的数据
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        supports_out=False,  # 不支持输出参数
        sample_inputs_func=sample_inputs_linalg_vander,  # 使用 sample_inputs_linalg_vander 函数生成示例输入
        skips=(
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),  # 跳过条件：目前不支持 MPS
                "TestCommon",  # 测试类名
                "test_numpy_ref_mps",  # 测试方法名
            ),
        ),
    ),
    ReductionOpInfo(
        "linalg.vector_norm",  # 操作名称为 linalg.vector_norm
        op=torch.linalg.vector_norm,  # 使用 torch.linalg.vector_norm 实现该操作
        identity=0,  # 标识元素为 0
        nan_policy="propagate",  # NaN 策略为传播
        supports_multiple_dims=True,  # 支持多维度计算
        complex_to_real=True,  # 复数转换为实数
        supports_forward_ad=True,  # 支持前向自动求导
        check_batched_forward_grad=False,  # 关闭批处理前向梯度检查
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),  # 支持浮点数、复数以及 torch.float16 和 torch.bfloat16 类型
        generate_args_kwargs=sample_kwargs_vector_norm,  # 生成 vector_norm 操作的参数和关键字参数
        aten_name="linalg_vector_norm",  # ATen 函数名为 linalg_vector_norm
        skips=(
            DecorateInfo(
                unittest.expectedFailure,  # 预期测试失败
                "TestReductions",  # 测试类名
                "test_dim_empty",  # 测试方法名
            ),
            DecorateInfo(
                unittest.expectedFailure,  # 预期测试失败
                "TestReductions",  # 测试类名
                "test_dim_empty_keepdim"  # 测试方法名
            ),
        ),
    ),
    OpInfo(
        "linalg.lu_factor",  # 操作名称为 linalg.lu_factor
        aten_name="linalg_lu_factor",  # ATen 函数名为 linalg_lu_factor
        op=torch.linalg.lu_factor,  # 使用 torch.linalg.lu_factor 实现该操作
        dtypes=floating_and_complex_types(),  # 支持浮点数和复数类型的数据
        gradcheck_fast_mode=True,  # 使用快速模式进行梯度检查
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        sample_inputs_func=sample_inputs_linalg_lu,  # 使用 sample_inputs_linalg_lu 函数生成示例输入
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],  # 装饰器列表，用于条件跳过测试
        skips=(
            DecorateInfo(
                unittest.expectedFailure,  # 预期测试失败
                "TestCommon",  # 测试类名
                "test_compare_cpu",  # 测试方法名
            ),
        ),
    ),
    OpInfo(
        "linalg.lu_factor_ex",  # 定义操作信息对象，对应 torch.linalg.lu_factor_ex 函数
        aten_name="linalg_lu_factor_ex",  # 对应的 ATen 函数名
        op=torch.linalg.lu_factor_ex,  # 实际执行的 PyTorch 操作函数
        dtypes=floating_and_complex_types(),  # 支持的浮点数和复数类型
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,  # 使用快速梯度检查模式
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        sample_inputs_func=sample_inputs_linalg_lu,  # 获取示例输入的函数
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],  # 装饰器列表，条件跳过测试
        skips=(
            # linalg.lu_factor: LU without pivoting is not implemented on the CPU
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_compare_cpu"),  # 标记为预期失败的测试用例信息
        ),
    ),
    OpInfo(
        "linalg.lu",  # 定义操作信息对象，对应 torch.linalg.lu 函数
        aten_name="linalg_lu",  # 对应的 ATen 函数名
        op=torch.linalg.lu,  # 实际执行的 PyTorch 操作函数
        dtypes=floating_and_complex_types(),  # 支持的浮点数和复数类型
        # https://github.com/pytorch/pytorch/issues/80411
        # Runs very slowly on slow-gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,  # 使用快速梯度检查模式
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        sample_inputs_func=sample_inputs_linalg_lu,  # 获取示例输入的函数
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],  # 装饰器列表，条件跳过测试
        skips=(
            # linalg.lu_factor: LU without pivoting is not implemented on the CPU
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_compare_cpu"),  # 标记为预期失败的测试用例信息
        ),
    ),
    OpInfo(
        "linalg.lu_solve",  # 定义操作信息对象，对应 torch.linalg.lu_solve 函数
        op=torch.linalg.lu_solve,  # 实际执行的 PyTorch 操作函数
        aten_name="linalg_lu_solve",  # 对应的 ATen 函数名
        dtypes=floating_and_complex_types(),  # 支持的浮点数和复数类型
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,  # 使用快速梯度检查模式
        supports_forward_ad=True,  # 支持前向自动微分
        check_batched_forward_grad=False,  # 不检查批处理的前向梯度
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        sample_inputs_func=sample_inputs_lu_solve,  # 获取示例输入的函数
        skips=(
            DecorateInfo(
                unittest.skip("Tests different backward paths"),  # 跳过测试，原因为测试不同的反向路径
                "TestCommon",  # 测试所在模块
                "test_floating_inputs_are_differentiable",  # 测试用例名称
            ),
        ),
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],  # 装饰器列表，条件跳过测试
    ),
    OpInfo(
        "linalg.inv",  # 操作名称为 "linalg.inv"
        aten_name="linalg_inv",  # 对应的 ATen 函数名为 "linalg_inv"
        op=torch.linalg.inv,  # 使用 torch.linalg.inv 函数作为操作
        aliases=("inverse",),  # 别名为 "inverse"
        dtypes=floating_and_complex_types(),  # 支持的数据类型为浮点数和复数类型
        sample_inputs_func=sample_inputs_linalg_invertible,  # 获取样本输入函数为 sample_inputs_linalg_invertible
        check_batched_gradgrad=False,  # 不检查批处理的梯度梯度
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向梯度反向梯度
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],  # 使用的装饰器列表
        skips=(  # 跳过的测试用例列表
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip 标记为跳过
                "TestCommon",  # 测试类名为 "TestCommon"
                "test_out",  # 测试方法名为 "test_out"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip 标记为跳过
                "TestCommon",  # 测试类名为 "TestCommon"
                "test_variant_consistency_eager",  # 测试方法名为 "test_variant_consistency_eager"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip 标记为跳过
                "TestJit",  # 测试类名为 "TestJit"
                "test_variant_consistency_jit",  # 测试方法名为 "test_variant_consistency_jit"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
        ),
    ),
    OpInfo(
        "linalg.inv_ex",  # 操作名称为 "linalg.inv_ex"
        aten_name="linalg_inv_ex",  # 对应的 ATen 函数名为 "linalg_inv_ex"
        op=torch.linalg.inv_ex,  # 使用 torch.linalg.inv_ex 函数作为操作
        dtypes=floating_and_complex_types(),  # 支持的数据类型为浮点数和复数类型
        sample_inputs_func=sample_inputs_linalg_invertible,  # 获取样本输入函数为 sample_inputs_linalg_invertible
        check_batched_gradgrad=False,  # 不检查批处理的梯度梯度
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向梯度反向梯度
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],  # 使用的装饰器列表
        skips=(  # 跳过的测试用例列表
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip 标记为跳过
                "TestCommon",  # 测试类名为 "TestCommon"
                "test_out",  # 测试方法名为 "test_out"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip 标记为跳过
                "TestCommon",  # 测试类名为 "TestCommon"
                "test_variant_consistency_eager",  # 测试方法名为 "test_variant_consistency_eager"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip 标记为跳过
                "TestJit",  # 测试类名为 "TestJit"
                "test_variant_consistency_jit",  # 测试方法名为 "test_variant_consistency_jit"
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
        ),
    ),
    OpInfo(
        "linalg.solve",
        aten_name="linalg_solve",
        op=torch.linalg.solve,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_solve,
        # 在梯度检查速度较慢时运行非常缓慢 - 可以通过减少输入大小来替代
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            # 跳过以下测试用例
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.solve_ex",
        aten_name="linalg_solve_ex",
        op=torch.linalg.solve_ex,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_solve,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            # 跳过以下测试用例
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.solve_triangular",
        aten_name="linalg_solve_triangular",
        op=torch.linalg.solve_triangular,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_solve_triangular,
        supports_fwgrad_bwgrad=True,
        skips=(skipCPUIfNoLapack,),
        # 由于调用了 out.copy_(result)，linalg.solve_triangular 无法进行批处理
        supports_forward_ad=True,
    ),
    OpInfo(
        "linalg.matrix_rank",  # 操作信息对象，指定执行 linalg.matrix_rank 操作
        aten_name="linalg_matrix_rank",  # 对应的 ATen 操作名
        dtypes=floating_and_complex_types(),  # 支持的数据类型，浮点数和复数类型
        supports_autograd=False,  # 不支持自动求导
        sample_inputs_func=sample_inputs_matrix_rank,  # 用于获取示例输入的函数
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],  # 装饰器列表，条件为没有 Magma 和 Cusolver 或没有 Lapack
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),  # 跳过测试的装饰信息，给出跳过的原因
                "TestCommon",  # 测试类名
                "test_out",  # 测试方法名
                device_type="mps",  # 设备类型为 mps
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 跳过测试的装饰信息，给出跳过的原因
                "TestCommon",  # 测试类名
                "test_variant_consistency_eager",  # 测试方法名
                device_type="mps",  # 设备类型为 mps
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            # jit 不接受张量输入进行矩阵秩的测试
            DecorateInfo(
                unittest.skip("Skipped!"),  # 跳过测试的装饰信息，给出跳过的原因
                "TestJit",  # 测试类名
                "test_variant_consistency_jit",  # 测试方法名
                dtypes=[torch.complex64, torch.float32],  # 数据类型为 torch.complex64 和 torch.float32
            ),
        ),
    ),
    OpInfo(
        "linalg.matrix_rank",  # 操作信息对象，指定执行 linalg.matrix_rank 操作
        aten_name="linalg_matrix_rank",  # 对应的 ATen 操作名
        variant_test_name="hermitian",  # 变体测试名称为 hermitian
        dtypes=floating_and_complex_types(),  # 支持的数据类型，浮点数和复数类型
        supports_autograd=False,  # 不支持自动求导
        sample_inputs_func=sample_inputs_linalg_pinv_hermitian,  # 用于获取示例输入的函数
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],  # 装饰器列表，条件为没有 Magma 和 Cusolver 或没有 Lapack
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),  # 跳过测试的装饰信息，给出跳过的原因
                "TestCommon",  # 测试类名
                "test_out",  # 测试方法名
                device_type="mps",  # 设备类型为 mps
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 跳过测试的装饰信息，给出跳过的原因
                "TestJit",  # 测试类名
                "test_variant_consistency_jit",  # 测试方法名
                device_type="mps",  # 设备类型为 mps
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
        ),
    ),
    OpInfo(
        "linalg.pinv",  # 操作信息对象，指定执行 linalg.pinv 操作
        aten_name="linalg_pinv",  # 对应的 ATen 操作名
        op=torch.linalg.pinv,  # 操作函数为 torch.linalg.pinv
        dtypes=floating_and_complex_types(),  # 支持的数据类型，浮点数和复数类型
        gradcheck_fast_mode=True,  # 在快速梯度检查模式下运行，因为在慢速梯度检查时速度很慢
        check_batched_grad=False,  # 不检查批次化梯度
        check_batched_gradgrad=False,  # 不检查批次化梯度的梯度
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        sample_inputs_func=sample_inputs_linalg_pinv,  # 用于获取示例输入的函数
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],  # 装饰器列表，条件为没有 Magma 和 Cusolver 或没有 Lapack
        skips=(
            # 在 CUDA 设备上存在 "leaked XXXX bytes CUDA memory on device 0" 的错误
            DecorateInfo(
                unittest.skip("Skipped!"),  # 跳过测试的装饰信息，给出跳过的原因
                "TestJit",  # 测试类名
                "test_variant_consistency_jit",  # 测试方法名
                device_type="cuda",  # 设备类型为 cuda
            ),
        ),
    ),
    OpInfo(
        "linalg.pinv",
        aten_name="linalg_pinv",
        variant_test_name="singular",
        # pinv is Frechet-differentiable in a rank-preserving neighborhood,
        # so we feed inputs that are the products of two full-rank factors,
        # to avoid any rank changes caused by the perturbations in the gradcheck
        op=lambda a, b: torch.linalg.pinv(a @ b.mT),  # 定义操作，计算 torch.linalg.pinv(a @ b.mT)
        dtypes=floating_and_complex_types(),  # 设置支持的数据类型为浮点数和复数类型
        supports_out=False,  # 不支持直接指定输出张量
        check_batched_grad=False,  # 不检查批处理梯度
        check_batched_gradgrad=False,  # 不检查批处理梯度的二阶导数
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        sample_inputs_func=sample_inputs_linalg_pinv_singular,  # 获取样本输入函数
        # Only large tensors show issues with implicit backward used prior to
        # explicit backward implementation.
        decorators=[slowTest, skipCUDAIfNoCusolver, skipCPUIfNoLapack],  # 使用装饰器修饰测试函数
        skips=(
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"  # 标记预期失败的测试用例
            ),
            # CUDA runs out of memory
            DecorateInfo(
                unittest.skip("Skipped!"),  # 标记为跳过测试
                "TestFwdGradients",  # 测试前向梯度
                "test_fn_fwgrad_bwgrad",  # 测试前向梯度和反向梯度
                device_type="cuda",  # 指定设备类型为 CUDA
                dtypes=[torch.cdouble],  # 指定数据类型为双精度复数张量
            ),
            # This test takes almost 2 hours to run!
            DecorateInfo(
                unittest.skip("Skipped!"),  # 标记为跳过测试
                "TestBwdGradients",  # 测试反向梯度
                "test_fn_gradgrad",  # 测试二阶导数
                device_type="cuda",  # 指定设备类型为 CUDA
                dtypes=[torch.cdouble],  # 指定数据类型为双精度复数张量
            ),
        ),
    ),
    OpInfo(
        "linalg.pinv",  # 操作的名称是 "linalg.pinv"
        aten_name="linalg_pinv",  # ATen 中对应的名称是 "linalg_pinv"
        variant_test_name="hermitian",  # 使用 "hermitian" 变体进行测试
        dtypes=floating_and_complex_types(),  # 支持浮点数和复数类型的数据
        check_batched_grad=False,  # 不检查批量梯度
        check_batched_gradgrad=False,  # 不检查批量梯度的梯度
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和后向梯度
        # 参考 https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,  # 不检查批量前向梯度
        sample_inputs_func=sample_inputs_linalg_pinv_hermitian,  # 使用 sample_inputs_linalg_pinv_hermitian 函数来生成样本输入
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,  # 使用 gradcheck_wrapper_hermitian_input 函数进行梯度检查
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],  # 使用 skipCUDAIfNoMagma 和 skipCPUIfNoLapack 修饰器
        skips=(  # 跳过以下测试用例
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip("Skipped!") 跳过测试
                "TestCommon",  # 测试所属的模块是 "TestCommon"
                "test_out",  # 测试方法名为 "test_out"
                device_type="mps",  # 在 "mps" 设备上运行测试
                dtypes=[torch.float32],  # 测试时使用 torch.float32 数据类型
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip("Skipped!") 跳过测试
                "TestCommon",  # 测试所属的模块是 "TestCommon"
                "test_variant_consistency_eager",  # 测试方法名为 "test_variant_consistency_eager"
                device_type="mps",  # 在 "mps" 设备上运行测试
                dtypes=[torch.float32],  # 测试时使用 torch.float32 数据类型
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip("Skipped!") 跳过测试
                "TestJit",  # 测试所属的模块是 "TestJit"
                "test_variant_consistency_jit",  # 测试方法名为 "test_variant_consistency_jit"
                device_type="mps",  # 在 "mps" 设备上运行测试
                dtypes=[torch.float32],  # 测试时使用 torch.float32 数据类型
            ),
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=1e-5, rtol=1e-5)}),  # 使用指定的容差进行测试
                "TestCommon",  # 测试所属的模块是 "TestCommon"
                "test_noncontiguous_samples",  # 测试方法名为 "test_noncontiguous_samples"
                device_type="cuda",  # 在 "cuda" 设备上运行测试
            ),
            # 该测试在慢速梯度检查下容易出现问题，可能由于舍入误差引起
            DecorateInfo(
                skipIfSlowGradcheckEnv,  # 在慢速梯度检查环境下跳过测试
                "TestFwdGradients",  # 测试所属的模块是 "TestFwdGradients"
                "test_fn_fwgrad_bwgrad",  # 测试方法名为 "test_fn_fwgrad_bwgrad"
                device_type="cuda",  # 在 "cuda" 设备上运行测试
            ),
        ),
    ),
    # 创建 OpInfo 对象，用于描述 "linalg.svd" 操作的信息
    OpInfo(
        "linalg.svd",  # 操作名称为 "linalg.svd"
        op=torch.linalg.svd,  # 使用 torch.linalg.svd 函数实现
        aten_name="linalg_svd",  # ATen 中的名称为 "linalg_svd"
        decomp_aten_name="_linalg_svd",  # 分解操作在 ATen 中的名称为 "_linalg_svd"
        dtypes=floating_and_complex_types(),  # 使用浮点数和复数类型作为数据类型
        # 在慢速梯度检查上运行非常缓慢 - 或者可以减少输入大小以加快检查
        gradcheck_fast_mode=True,  # 使用快速梯度检查模式
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        supports_forward_ad=True,  # 支持前向自动微分
        check_batched_forward_grad=False,  # 不检查批处理前向梯度
        # 我们使用 at::allclose，它没有批处理规则
        check_batched_grad=False,  # 不检查批处理梯度
        check_batched_gradgrad=False,  # 不检查批处理梯度的梯度
        sample_inputs_func=sample_inputs_svd,  # 使用 sample_inputs_svd 函数生成示例输入
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],  # 使用装饰器进行条件跳过
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip 标记为跳过
                "TestCommon",
                "test_out",
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip 标记为跳过
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip 标记为跳过
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",  # 设备类型为 "mps"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip 标记为跳过
                "TestFakeTensor",
                "test_fake_crossref_backward_amp",
                device_type="cuda",  # 设备类型为 "cuda"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
                active_if=TEST_WITH_ROCM,  # 如果在 ROCm 下激活
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip 标记为跳过
                "TestFakeTensor",
                "test_fake_crossref_backward_no_amp",
                device_type="cuda",  # 设备类型为 "cuda"
                dtypes=[torch.float32],  # 数据类型为 torch.float32
                active_if=TEST_WITH_ROCM,  # 如果在 ROCm 下激活
            ),
        ),
    ),
    OpInfo(
        "linalg.svdvals",  # 操作名称，表示计算奇异值的函数
        op=torch.linalg.svdvals,  # PyTorch 中实现该操作的函数
        aten_name="linalg_svdvals",  # ATen 中对应的操作名称
        decomp_aten_name="_linalg_svd",  # 分解操作在 ATen 中的名称
        dtypes=floating_and_complex_types(),  # 支持的数据类型，包括浮点数和复数类型
        check_batched_forward_grad=False,  # 不检查批处理前向梯度
        supports_fwgrad_bwgrad=True,  # 支持前向和反向梯度
        supports_forward_ad=True,  # 支持自动微分的前向传播
        # 使用 at::allclose，它没有批处理规则
        check_batched_gradgrad=False,  # 不检查批处理梯度梯度
        sample_inputs_func=sample_inputs_linalg_svdvals,  # 生成用于测试的输入样本的函数
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],  # 修饰器列表，用于条件跳过测试
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),  # 标记为跳过的装饰器信息
                "TestFakeTensor",  # 测试类名称
                "test_fake_crossref_backward_amp",  # 测试函数名称
                device_type="cuda",  # 在 CUDA 设备上执行测试
                dtypes=[torch.float32],  # 测试的数据类型为 float32
                active_if=TEST_WITH_ROCM,  # 仅在 ROCm 环境下激活
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),  # 标记为跳过的装饰器信息
                "TestFakeTensor",  # 测试类名称
                "test_fake_crossref_backward_no_amp",  # 测试函数名称
                device_type="cuda",  # 在 CUDA 设备上执行测试
                dtypes=[torch.float32],  # 测试的数据类型为 float32
                active_if=TEST_WITH_ROCM,  # 仅在 ROCm 环境下激活
            ),
        ),
    ),
    OpInfo(
        "linalg.tensorinv",  # 操作名称，表示张量的逆运算
        ref=np.linalg.tensorinv,  # NumPy 中对应的参考实现
        dtypes=floating_and_complex_types(),  # 支持的数据类型，包括浮点数和复数类型
        sample_inputs_func=sample_inputs_tensorinv,  # 生成用于测试的输入样本的函数
        supports_forward_ad=True,  # 支持自动微分的前向传播
        supports_fwgrad_bwgrad=True,  # 支持前向和反向梯度
        # 查看 https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,  # 不检查批处理前向梯度
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],  # 修饰器列表，用于条件跳过测试
        skips=(
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),  # 标记为跳过的装饰器信息
                "TestCommon",  # 测试类名称
                "test_numpy_ref_mps",  # 测试函数名称
            ),
        ),
    ),
    OpInfo(
        "linalg.tensorsolve",  # 操作名称，表示张量的求解运算
        ref=lambda a, b, dims=None: np.linalg.tensorsolve(a, b, axes=dims),  # 使用 Lambda 函数引用 NumPy 中的参考实现
        dtypes=floating_and_complex_types(),  # 支持的数据类型，包括浮点数和复数类型
        sample_inputs_func=sample_inputs_tensorsolve,  # 生成用于测试的输入样本的函数
        supports_forward_ad=True,  # 支持自动微分的前向传播
        supports_fwgrad_bwgrad=True,  # 支持前向和反向梯度
        decorators=[
            skipCUDAIfNoMagmaAndNoCusolver,  # 条件修饰器：如果没有 Magma 和 Cusolver，跳过测试
            skipCPUIfNoLapack,  # 条件修饰器：如果没有 Lapack，跳过测试
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=1e-03, rtol=1e-03)}),  # 覆盖容差的修饰器信息
                "TestCommon",  # 测试类名称
                "test_noncontiguous_samples",  # 测试函数名称
                device_type="cuda",  # 在 CUDA 设备上执行测试
            ),
        ],
        skips=(
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),  # 标记为跳过的装饰器信息
                "TestCommon",  # 测试类名称
                "test_numpy_ref_mps",  # 测试函数名称
            ),
        ),
    ),
python_ref_db: List[OpInfo] = [
    # 定义一个列表 python_ref_db，包含多个 OpInfo 对象，用于存储 Python 函数参考信息

    #
    # torch.linalg
    #

    # 创建一个 PythonRefInfo 对象，表示 torch.linalg.cross 函数的参考信息
    PythonRefInfo(
        "_refs.linalg.cross",
        torch_opinfo_name="linalg.cross",
        supports_out=True,
        op_db=op_db,
        skips=(
            # TODO: 这个装饰器真的需要吗？
            DecorateInfo(
                unittest.expectedFailure, "TestCommon", "test_python_ref_errors"
            ),
        ),
    ),

    # 创建一个 PythonRefInfo 对象，表示 torch.linalg.diagonal 函数的参考信息
    PythonRefInfo(
        "_refs.linalg.diagonal",
        torch_opinfo_name="linalg.diagonal",
        supports_out=False,
        op_db=op_db,
    ),

    # 创建一个 PythonRefInfo 对象，表示 torch.linalg.vecdot 函数的参考信息
    PythonRefInfo(
        "_refs.linalg.vecdot",
        torch_opinfo_name="linalg.vecdot",
        op_db=op_db,
    ),

    # 创建一个 ReductionPythonRefInfo 对象，表示 torch.linalg.vector_norm 函数的参考信息
    ReductionPythonRefInfo(
        "_refs.linalg.vector_norm",
        torch_opinfo_name="linalg.vector_norm",
        supports_out=True,
        op_db=op_db,
        skips=(
            # FIXME: 当 dim=[] 时，sum 函数会减少所有维度
            DecorateInfo(unittest.expectedFailure, "TestReductions", "test_dim_empty"),
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_dim_empty_keepdim"
            ),
        ),
    ),

    # 创建一个 PythonRefInfo 对象，表示 torch.linalg.matrix_norm 函数的参考信息
    PythonRefInfo(
        "_refs.linalg.matrix_norm",
        torch_opinfo_name="linalg.matrix_norm",
        supports_out=True,
        # 使用 vector_norm 函数内部，而 vector_norm 受到 https://github.com/pytorch/pytorch/issues/77216 影响
        validate_view_consistency=False,
        op_db=op_db,
    ),

    # 创建一个 PythonRefInfo 对象，表示 torch.linalg.norm 函数的参考信息
    PythonRefInfo(
        "_refs.linalg.norm",
        torch_opinfo_name="linalg.norm",
        supports_out=True,
        # 使用 vector_norm 函数内部，而 vector_norm 受到 https://github.com/pytorch/pytorch/issues/77216 影响
        validate_view_consistency=False,
        op_db=op_db,
    ),

    # 创建一个 PythonRefInfo 对象，表示 torch.linalg.svd 函数的参考信息
    PythonRefInfo(
        "_refs.linalg.svd",
        torch_opinfo_name="linalg.svd",
        supports_out=True,
        op_db=op_db,
    ),

    # 创建一个 PythonRefInfo 对象，表示 torch.linalg.svdvals 函数的参考信息
    PythonRefInfo(
        "_refs.linalg.svdvals",
        torch_opinfo_name="linalg.svdvals",
        supports_out=True,
        op_db=op_db,
    ),
]
```