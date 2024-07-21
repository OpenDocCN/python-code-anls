# `.\pytorch\torch\_lowrank.py`

```py
"""Implement various linear algebra algorithms for low rank matrices.
"""

__all__ = ["svd_lowrank", "pca_lowrank"]

from typing import Optional, Tuple

import torch
from torch import _linalg_utils as _utils, Tensor
from torch.overrides import handle_torch_function, has_torch_function


def get_approximate_basis(
    A: Tensor,
    q: int,
    niter: Optional[int] = 2,
    M: Optional[Tensor] = None,
) -> Tensor:
    """Return tensor :math:`Q` with :math:`q` orthonormal columns such
    that :math:`Q Q^H A` approximates :math:`A`. If :math:`M` is
    specified, then :math:`Q` is such that :math:`Q Q^H (A - M)`
    approximates :math:`A - M`. without instantiating any tensors
    of the size of :math:`A` or :math:`M`.

    .. note:: The implementation is based on the Algorithm 4.4 from
              Halko et al., 2009.

    .. note:: For an adequate approximation of a k-rank matrix
              :math:`A`, where k is not known in advance but could be
              estimated, the number of :math:`Q` columns, q, can be
              choosen according to the following criteria: in general,
              :math:`k <= q <= min(2*k, m, n)`. For large low-rank
              matrices, take :math:`q = k + 5..10`.  If k is
              relatively small compared to :math:`min(m, n)`, choosing
              :math:`q = k + 0..2` may be sufficient.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Args::
        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int): the dimension of subspace spanned by :math:`Q`
                 columns.

        niter (int, optional): the number of subspace iterations to
                               conduct; ``niter`` must be a
                               nonnegative integer. In most cases, the
                               default value 2 is more than enough.

        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, m, n)`.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """

    # Set default value for niter if not provided
    niter = 2 if niter is None else niter

    # Determine the appropriate data type based on A's properties
    dtype = _utils.get_floating_dtype(A) if not A.is_complex() else A.dtype

    # Alias for matrix multiplication function
    matmul = _utils.matmul

    # Generate a random matrix R of shape (A.shape[-1], q)
    R = torch.randn(A.shape[-1], q, dtype=dtype, device=A.device)

    # Compute X = A @ R or (A - M) @ R if M is provided
    X = matmul(A, R)
    if M is not None:
        X = X - matmul(M, R)

    # Compute the QR decomposition of X and extract Q
    Q = torch.linalg.qr(X).Q
    # 对于给定的迭代次数，执行以下操作
    for i in range(niter):
        # 计算矩阵 A 的共轭转置与矩阵 Q 的乘积
        X = matmul(A.mH, Q)
        # 如果矩阵 M 不为 None，则从 X 中减去矩阵 M 的共轭转置与矩阵 Q 的乘积
        if M is not None:
            X = X - matmul(M.mH, Q)
        # 对矩阵 X 进行 QR 分解，并取其正交矩阵 Q
        Q = torch.linalg.qr(X).Q
        # 计算矩阵 A 与矩阵 Q 的乘积
        X = matmul(A, Q)
        # 如果矩阵 M 不为 None，则从 X 中减去矩阵 M 与矩阵 Q 的乘积
        if M is not None:
            X = X - matmul(M, Q)
        # 再次对矩阵 X 进行 QR 分解，并取其正交矩阵 Q
        Q = torch.linalg.qr(X).Q
    # 返回经过多次迭代后得到的正交矩阵 Q
    return Q
# 如果当前环境不是 Torch 脚本环境（即不在 Torch 的 JIT 脚本中），则执行以下代码块
if not torch.jit.is_scripting():
    # 创建包含输入张量 A 和 M 的元组 tensor_ops
    tensor_ops = (A, M)
    # 如果 tensor_ops 中的所有元素类型都是 torch.Tensor 或 None 类型，并且这些对象支持 Torch 函数重载机制
    if not set(map(type, tensor_ops)).issubset(
        (torch.Tensor, type(None))
    ) and has_torch_function(tensor_ops):
        # 调用 Torch 函数重载处理器，处理 svd_lowrank 函数的调用
        return handle_torch_function(
            svd_lowrank, tensor_ops, A, q=q, niter=niter, M=M
        )
# 如果不满足上述条件，则调用 _svd_lowrank 函数进行后续处理
return _svd_lowrank(A, q=q, niter=niter, M=M)
    # 如果 q 为 None，则将其设为 6，否则保持不变
    q = 6 if q is None else q
    # 获取矩阵 A 的最后两个维度的形状
    m, n = A.shape[-2:]
    # 获取 matmul 函数的引用
    matmul = _utils.matmul
    # 如果 M 不为 None，则将其广播到 A 的大小
    if M is not None:
        M = M.broadcast_to(A.size())

    # 假设 A 是"高"的矩阵（即行数大于等于列数）
    if m < n:
        # 对 A 取 Hermitian（共轭转置）
        A = A.mH
        # 如果 M 不为 None，则对 M 也取 Hermitian
        if M is not None:
            M = M.mH

    # 获取 A 的近似基础 Q
    Q = get_approximate_basis(A, q, niter=niter, M=M)
    # 计算 B = Q 的 Hermitian 乘以 A
    B = matmul(Q.mH, A)
    # 如果 M 不为 None，则计算 B = B - Q 的 Hermitian 乘以 M
    if M is not None:
        B = B - matmul(Q.mH, M)
    # 对 B 进行奇异值分解，得到 U, S, Vh
    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    # 对 Vh（奇异值分解的右奇异矩阵的 Hermitian）取 Hermitian
    V = Vh.mH
    # 计算 U = Q 的乘以 U
    U = Q.matmul(U)

    # 如果 A 是"扁平"的矩阵（即行数小于列数）
    if m < n:
        # 交换 U 和 V
        U, V = V, U

    # 返回 U, S, V
    return U, S, V
# 检查是否处于脚本模式，如果不是，则检查输入是否具有 torch 函数处理，如果有，则调用处理函数处理输入并返回结果
if not torch.jit.is_scripting():
    if type(A) is not torch.Tensor and has_torch_function((A,)):
        return handle_torch_function(
            pca_lowrank, (A,), A, q=q, center=center, niter=niter
        )

# 获取输入张量 A 的形状 (m, n)
(m, n) = A.shape[-2:]

# 如果未提供 q 参数，则设置 q 为 min(6, m, n)
if q is None:
    q = min(6, m, n)
# 如果 q 不在合法范围内，抛出 ValueError 异常
elif not (q >= 0 and q <= min(m, n)):
    raise ValueError(
        f"q(={q}) must be non-negative integer and not greater than min(m, n)={min(m, n)}"
    )

# 如果 niter 小于 0，抛出 ValueError 异常
if not (niter >= 0):
    raise ValueError(f"niter(={niter}) must be non-negative integer")
    # 根据输入矩阵 A 推断浮点数数据类型
    dtype = _utils.get_floating_dtype(A)
    
    # 如果不进行中心化操作，则直接调用 _svd_lowrank 函数并返回结果
    if not center:
        return _svd_lowrank(A, q, niter=niter, M=None)
    
    # 如果输入矩阵 A 是稀疏矩阵
    if _utils.is_sparse(A):
        # 检查输入矩阵 A 的维度，确保是二维张量
        if len(A.shape) != 2:
            raise ValueError("pca_lowrank input is expected to be 2-dimensional tensor")
        
        # 计算稀疏矩阵 A 按列求和，得到列均值向量 c
        c = torch.sparse.sum(A, dim=(-2,)) / m
        
        # 重新整形 c，准备创建稀疏张量
        column_indices = c.indices()[0]
        indices = torch.zeros(
            2,
            len(column_indices),
            dtype=column_indices.dtype,
            device=column_indices.device,
        )
        indices[0] = column_indices
        C_t = torch.sparse_coo_tensor(
            indices, c.values(), (n, 1), dtype=dtype, device=A.device
        )
    
        # 创建稀疏张量 ones_m1_t，形状与 A 的前两个维度一致
        ones_m1_t = torch.ones(A.shape[:-2] + (1, m), dtype=dtype, device=A.device)
        
        # 计算 M = C_t * ones_m1_t^T，其中 ^T 表示转置
        M = torch.sparse.mm(C_t, ones_m1_t).mT
        
        # 调用 _svd_lowrank 函数，传入 M，并返回结果
        return _svd_lowrank(A, q, niter=niter, M=M)
    else:
        # 计算 A 沿着倒数第二维的均值，得到 C
        C = A.mean(dim=(-2,), keepdim=True)
        
        # 调用 _svd_lowrank 函数，传入 A 减去 C 的结果，并返回结果
        return _svd_lowrank(A - C, q, niter=niter, M=None)
```