# `.\pytorch\torch\_lobpcg.py`

```py
# mypy: allow-untyped-defs
"""Locally Optimal Block Preconditioned Conjugate Gradient methods.
"""
# 作者：Pearu Peterson
# 创建日期：2020年2月

# 导入必要的类型注解
from typing import Dict, Optional, Tuple

# 导入 torch 库及其子模块
import torch
from torch import _linalg_utils as _utils, Tensor
from torch.overrides import handle_torch_function, has_torch_function

# 定义可以公开访问的函数和类
__all__ = ["lobpcg"]

# 定义一个函数用于计算逆向传播的完整特征空间
def _symeig_backward_complete_eigenspace(D_grad, U_grad, A, D, U):
    # 计算矩阵 F，其中 F_ij = (d_j - d_i)^{-1} (i ≠ j)，F_ii = 0
    F = D.unsqueeze(-2) - D.unsqueeze(-1)
    F.diagonal(dim1=-2, dim2=-1).fill_(float("inf"))  # 将对角线上的元素填充为无穷大
    F.pow_(-1)  # 对 F 中的每个元素取倒数

    # 计算 A 的梯度，A.grad = U (D.grad + (U^T U.grad * F)) U^T
    Ut = U.mT.contiguous()  # 获取 U 的转置，并确保其在内存中是连续存储的
    res = torch.matmul(
        U, torch.matmul(torch.diag_embed(D_grad) + torch.matmul(Ut, U_grad) * F, Ut)
    )  # 使用上述公式计算结果

    return res  # 返回计算结果


# 定义一个函数，根据多项式的根计算多项式的系数
def _polynomial_coefficients_given_roots(roots):
    """
    给定多项式的根 `roots`，计算多项式的系数。

    如果 roots = (r_1, ..., r_n)，则该方法返回系数 (a_0, a_1, ..., a_n (== 1))，
    使得 p(x) = (x - r_1) * ... * (x - r_n)
         = x^n + a_{n-1} * x^{n-1} + ... a_1 * x_1 + a_0

    注意：为了更好的性能，需要编写一个低级别的核心函数。
    """
    poly_order = roots.shape[-1]  # 获取多项式的阶数
    poly_coeffs_shape = list(roots.shape)
    # 假设 p(x) = x^n + a_{n-1} * x^{n-1} + ... + a_1 * x + a_0，
    # 所以 poly_coeffs = {a_0, ..., a_n, a_{n+1}(== 1)}，
    # 但我们插入一个额外的系数以便下面更好的向量化操作
    poly_coeffs_shape[-1] += 2
    poly_coeffs = roots.new_zeros(poly_coeffs_shape)  # 创建一个全零的系数数组
    poly_coeffs[..., 0] = 1  # 设置最高次项的系数为 1
    poly_coeffs[..., -1] = 1  # 设置常数项的系数为 1

    # 使用霍纳法则计算多项式系数
    for i in range(1, poly_order + 1):
        # 注意，对于这种方法来说，计算其反向传播是计算上的难点，
        # 因为给定系数，需要找到根或者根据 Vieta's 定理计算灵敏度。
        # 因此下面的代码尝试绕过显式的根查找，通过内存复制操作来模拟霍纳法则的递归过程。
        # 这些内存复制是为了构造计算图中的节点，利用霍纳法则的显式（非原位，每一步都有独立的节点）
        # 递归方法。
        # 需要更多的内存，O(... * k^2)，但只有 O(... * k^2) 的复杂度。
        poly_coeffs_new = poly_coeffs.clone() if roots.requires_grad else poly_coeffs
        out = poly_coeffs_new.narrow(-1, poly_order - i, i + 1)
        out -= roots.narrow(-1, i - 1, 1) * poly_coeffs.narrow(
            -1, poly_order - i + 1, i + 1
        )
        poly_coeffs = poly_coeffs_new

    return poly_coeffs.narrow(-1, 1, poly_order + 1)  # 返回计算得到的多项式系数


# 定义一个函数，使用霍纳法则计算多项式在给定点 x 处的值
def _polynomial_value(poly, x, zero_power, transition):
    """
    一个通用的方法，使用霍纳法则计算多项式 poly(x) 的值。
    """
    # 使用 Horner's Rule 对多项式进行求值
    def horner_eval(poly, x, zero_power, transition):
        # 初始化结果为 zero_power 的克隆，即多项式求和的起始点
        res = zero_power.clone()
        # 从多项式的最高次数依次向最低次数遍历
        for k in range(poly.size(-1) - 2, -1, -1):
            # 使用 transition 函数进行 Horner's Rule 的一步迭代
            # 将当前的 res、x 和多项式的系数 poly[..., k] 传递给 transition 函数
            res = transition(res, x, poly[..., k])
        # 返回最终的求和结果
        return res
```python`
# 定义函数 `_matrix_polynomial_value`，用于计算矩阵输入 `x` 的多项式 `poly(x)` 的值
def _matrix_polynomial_value(poly, x, zero_power=None):
    """
    Evaluates `poly(x)` for the (batched) matrix input `x`.
    Check out `_polynomial_value` function for more details.
    """

    # 定义矩阵感知的 Horner 算法迭代函数
    def transition(curr_poly_val, x, poly_coeff):
        # 使用矩阵乘法计算当前多项式值与输入矩阵 x 的乘积
        res = x.matmul(curr_poly_val)
        # 在对角线上添加多项式系数 poly_coeff
        res.diagonal(dim1=-2, dim2=-1).add_(poly_coeff.unsqueeze(-1))
        return res

    # 如果未提供 zero_power 参数，则创建单位矩阵，与输入矩阵 x 的维度相同
    if zero_power is None:
        zero_power = torch.eye(
            x.size(-1), x.size(-1), dtype=x.dtype, device=x.device
        ).view(*([1] * len(list(x.shape[:-2]))), x.size(-1), x.size(-1))

    # 调用 `_polynomial_value` 函数，返回多项式计算结果
    return _polynomial_value(poly, x, zero_power, transition)


# 定义函数 `_vector_polynomial_value`，用于计算向量输入 `x` 的多项式 `poly(x)` 的值
def _vector_polynomial_value(poly, x, zero_power=None):
    """
    Evaluates `poly(x)` for the (batched) vector input `x`.
    Check out `_polynomial_value` function for more details.
    """

    # 定义向量感知的 Horner 算法迭代函数
    def transition(curr_poly_val, x, poly_coeff):
        # 使用向量乘法和加法计算多项式的值
        res = torch.addcmul(poly_coeff.unsqueeze(-1), x, curr_poly_val)
        return res

    # 如果未提供 zero_power 参数，则创建元素全为 1 的张量，与输入 x 的形状相同
    if zero_power is None:
        zero_power = x.new_ones(1).expand(x.shape)

    # 调用 `_polynomial_value` 函数，返回多项式计算结果
    return _polynomial_value(poly, x, zero_power, transition)


# 定义函数 `_symeig_backward_partial_eigenspace`，用于计算特征值反向传播的部分特征空间
def _symeig_backward_partial_eigenspace(D_grad, U_grad, A, D, U, largest):
    # 计算投影算子，投影到由矩阵 U 的列张成的正交子空间上
    Ut = U.mT.contiguous()
    proj_U_ortho = -U.matmul(Ut)
    proj_U_ortho.diagonal(dim1=-2, dim2=-1).add_(1)

    # 计算 U 的正交基 U_ortho，该基是 U 的列张成子空间的正交补
    #
    # 为确保结果的确定性，使用指定设备上的随机生成器
    gen = torch.Generator(A.device)

    # U 的正交补
    U_ortho = proj_U_ortho.matmul(
        torch.randn(
            (*A.shape[:-1], A.size(-1) - D.size(-1)),
            dtype=A.dtype,
            device=A.device,
            generator=gen,
        )
    )
    U_ortho_t = U_ortho.mT.contiguous()

    # 计算张量 D 的特征多项式系数
    chr_poly_D = _polynomial_coefficients_given_roots(D)

    # 下面的代码解决 Sylvester 方程的显式解，
    # 用于计算并整合存储在 `res` 变量中的整个梯度。
    #
    # 等同于以下简单实现：
    # res = A.new_zeros(A.shape)
    # p_res = A.new_zeros(*A.shape[:-1], D.size(-1))
    # for k in range(1, chr_poly_D.size(-1)):
    #     p_res.zero_()
    #     for i in range(0, k):
    #         p_res += (A.matrix_power(k - 1 - i) @ U_grad) * D.pow(i).unsqueeze(-2)
    #     res -= chr_poly_D[k] * (U_ortho @ poly_D_at_A.inverse() @ U_ortho_t @  p_res @ U.t())
    #
    # Note that dX is a differential, so the gradient contribution comes from the backward sensitivity
    # Tr(f(U_grad, D_grad, A, U, D)^T dX) = Tr(g(U_grad, A, U, D)^T dA) for some functions f and g,
    # and we need to compute g(U_grad, A, U, D)
    #
    # The naive implementation is based on the paper
    # Hu, Qingxi, and Daizhan Cheng.
    # "The polynomial solution to the Sylvester matrix equation."
    # Applied mathematics letters 19.9 (2006): 859-864.
    #
    # We can modify the computation of `p_res` from above in a more efficient way
    # p_res =   U_grad * (chr_poly_D[1] * D.pow(0) + ... + chr_poly_D[k] * D.pow(k)).unsqueeze(-2)
    #       + A U_grad * (chr_poly_D[2] * D.pow(0) + ... + chr_poly_D[k] * D.pow(k - 1)).unsqueeze(-2)
    #       + ...
    #       + A.matrix_power(k - 1) U_grad * chr_poly_D[k]
    # Note that this saves us from redundant matrix products with A (elimination of matrix_power)
    U_grad_projected = U_grad
    series_acc = U_grad_projected.new_zeros(U_grad_projected.shape)
    for k in range(1, chr_poly_D.size(-1)):
        poly_D = _vector_polynomial_value(chr_poly_D[..., k:], D)
        series_acc += U_grad_projected * poly_D.unsqueeze(-2)
        U_grad_projected = A.matmul(U_grad_projected)

    # compute chr_poly_D(A) which essentially is:
    #
    # chr_poly_D_at_A = A.new_zeros(A.shape)
    # for k in range(chr_poly_D.size(-1)):
    #     chr_poly_D_at_A += chr_poly_D[k] * A.matrix_power(k)
    #
    # Note, however, for better performance we use the Horner's rule
    chr_poly_D_at_A = _matrix_polynomial_value(chr_poly_D, A)

    # compute the action of `chr_poly_D_at_A` restricted to U_ortho_t
    chr_poly_D_at_A_to_U_ortho = torch.matmul(
        U_ortho_t, torch.matmul(chr_poly_D_at_A, U_ortho)
    )
    # we need to invert 'chr_poly_D_at_A_to_U_ortho`, for that we compute its
    # Cholesky decomposition and then use `torch.cholesky_solve` for better stability.
    # Cholesky decomposition requires the input to be positive-definite.
    # Note that `chr_poly_D_at_A_to_U_ortho` is positive-definite if
    # 1. `largest` == False, or
    # 2. `largest` == True and `k` is even
    # under the assumption that `A` has distinct eigenvalues.
    #
    # check if `chr_poly_D_at_A_to_U_ortho` is positive-definite or negative-definite
    chr_poly_D_at_A_to_U_ortho_sign = -1 if (largest and (k % 2 == 1)) else +1
    chr_poly_D_at_A_to_U_ortho_L = torch.linalg.cholesky(
        chr_poly_D_at_A_to_U_ortho_sign * chr_poly_D_at_A_to_U_ortho
    )

    # compute the gradient part in span(U)
    res = _symeig_backward_complete_eigenspace(D_grad, U_grad, A, D, U)

    # incorporate the Sylvester equation solution into the full gradient
    # it resides in span(U_ortho)
    res -= U_ortho.matmul(
        chr_poly_D_at_A_to_U_ortho_sign
        * torch.cholesky_solve(
            U_ortho_t.matmul(series_acc), chr_poly_D_at_A_to_U_ortho_L
        )
    ).matmul(Ut)

    return res
def _symeig_backward(D_grad, U_grad, A, D, U, largest):
    # 如果 `U` 是方阵，则 `U` 的列是完整的特征空间
    if U.size(-1) == U.size(-2):
        # 返回完整特征空间的反向传播结果
        return _symeig_backward_complete_eigenspace(D_grad, U_grad, A, D, U)
    else:
        # 返回部分特征空间的反向传播结果
        return _symeig_backward_partial_eigenspace(D_grad, U_grad, A, D, U, largest)


class LOBPCGAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        A: Tensor,
        k: Optional[int] = None,
        B: Optional[Tensor] = None,
        X: Optional[Tensor] = None,
        n: Optional[int] = None,
        iK: Optional[Tensor] = None,
        niter: Optional[int] = None,
        tol: Optional[float] = None,
        largest: Optional[bool] = None,
        method: Optional[str] = None,
        tracker: None = None,
        ortho_iparams: Optional[Dict[str, int]] = None,
        ortho_fparams: Optional[Dict[str, float]] = None,
        ortho_bparams: Optional[Dict[str, bool]] = None,
    ) -> Tuple[Tensor, Tensor]:
        # 确保输入是连续的，以提高效率。
        # 注意：自动求导当前不支持稀疏输入的稠密梯度。
        A = A.contiguous() if (not A.is_sparse) else A
        if B is not None:
            B = B.contiguous() if (not B.is_sparse) else B

        # 调用 _lobpcg 函数计算 LOBPCG 方法的结果 D 和 U
        D, U = _lobpcg(
            A,
            k,
            B,
            X,
            n,
            iK,
            niter,
            tol,
            largest,
            method,
            tracker,
            ortho_iparams,
            ortho_fparams,
            ortho_bparams,
        )

        # 保存用于反向传播的参数和计算的结果
        ctx.save_for_backward(A, B, D, U)
        ctx.largest = largest  # 保存最大特征值或最小特征值的标志

        # 返回 LOBPCG 方法的计算结果 D 和 U
        return D, U

    @staticmethod
    # 定义反向传播函数，接收上下文 ctx、D_grad 和 U_grad 作为输入
    def backward(ctx, D_grad, U_grad):
        # 初始化 A_grad 和 B_grad 为 None
        A_grad = B_grad = None
        # 初始化一个长度为 14 的 None 列表
        grads = [None] * 14

        # 从上下文中获取保存的张量 A, B, D, U 和 largest
        A, B, D, U = ctx.saved_tensors
        largest = ctx.largest

        # 检查 lobpcg.backward 的输入是否有不支持的情况
        if A.is_sparse or (B is not None and B.is_sparse and ctx.needs_input_grad[2]):
            # 如果 A 是稀疏张量，或者 B 是稀疏张量且需要对第三个输入进行梯度计算，则抛出异常
            raise ValueError(
                "lobpcg.backward does not support sparse input yet."
                "Note that lobpcg.forward does though."
            )
        if (
            A.dtype in (torch.complex64, torch.complex128)
            or B is not None
            and B.dtype in (torch.complex64, torch.complex128)
        ):
            # 如果 A 或者 B 的数据类型是复数类型，则抛出异常
            raise ValueError(
                "lobpcg.backward does not support complex input yet."
                "Note that lobpcg.forward does though."
            )
        if B is not None:
            # 如果 B 不为 None，则抛出异常，lobpcg.backward 目前不支持 B != I 的情况
            raise ValueError(
                "lobpcg.backward does not support backward with B != I yet."
            )

        if largest is None:
            # 如果 largest 为 None，则设置为 True
            largest = True

        # 对称特征值分解的反向传播过程
        if B is None:
            # 如果 B 为 None，则调用 _symeig_backward 函数计算 A_grad 和 B_grad
            A_grad = _symeig_backward(D_grad, U_grad, A, D, U, largest)

        # 将 A_grad 和 B_grad 分别赋值给 grads 列表的第 0 和第 2 个位置
        grads[0] = A_grad  # A 在 grads 中的索引为 0
        grads[2] = B_grad  # B 在 grads 中的索引为 2
        # 返回 grads 列表作为输出
        return tuple(grads)
# 使用LOBPCG方法解决对称正定广义特征值问题，找到最大（或最小）的k个特征值及其对应的特征向量
def lobpcg(
    A: Tensor,  # 输入参数A，可以是密集矩阵、稀疏矩阵或密集矩阵的批次
    k: Optional[int] = None,  # 要找到的特征值个数，默认为None，即找所有特征值
    B: Optional[Tensor] = None,  # 可选的右手边矩阵B，默认为None
    X: Optional[Tensor] = None,  # 可选的初始矩阵X，默认为None
    n: Optional[int] = None,  # 可选的矩阵维度n，默认为None
    iK: Optional[Tensor] = None,  # 可选的预处理矩阵iK，默认为None
    niter: Optional[int] = None,  # 可选的迭代次数，默认为None
    tol: Optional[float] = None,  # 可选的收敛容差，默认为None
    largest: Optional[bool] = None,  # 可选的特征值大小排序方式，默认为None
    method: Optional[str] = None,  # 可选的LOBPCG算法类型，默认为None
    tracker: None = None,  # 不使用跟踪器
    ortho_iparams: Optional[Dict[str, int]] = None,  # 可选的整数参数字典用于正交化过程，默认为None
    ortho_fparams: Optional[Dict[str, float]] = None,  # 可选的浮点数参数字典用于正交化过程，默认为None
    ortho_bparams: Optional[Dict[str, bool]] = None,  # 可选的布尔参数字典用于正交化过程，默认为None
) -> Tuple[Tensor, Tensor]:  # 返回值是特征值张量E和特征向量张量X的元组

    """Find the k largest (or smallest) eigenvalues and the corresponding
    eigenvectors of a symmetric positive definite generalized
    eigenvalue problem using matrix-free LOBPCG methods.

    This function is a front-end to the following LOBPCG algorithms
    selectable via `method` argument:

      `method="basic"` - the LOBPCG method introduced by Andrew
      Knyazev, see [Knyazev2001]. A less robust method, may fail when
      Cholesky is applied to singular input.

      `method="ortho"` - the LOBPCG method with orthogonal basis
      selection [StathopoulosEtal2002]. A robust method.

    Supported inputs are dense, sparse, and batches of dense matrices.

    .. note:: In general, the basic method spends least time per
      iteration. However, the robust methods converge much faster and
      are more stable. So, the usage of the basic method is generally
      not recommended but there exist cases where the usage of the
      basic method may be preferred.

    .. warning:: The backward method does not support sparse and complex inputs.
      It works only when `B` is not provided (i.e. `B == None`).
      We are actively working on extensions, and the details of
      the algorithms are going to be published promptly.

    .. warning:: While it is assumed that `A` is symmetric, `A.grad` is not.
      To make sure that `A.grad` is symmetric, so that `A - t * A.grad` is symmetric
      in first-order optimization routines, prior to running `lobpcg`
      we do the following symmetrization map: `A -> (A + A.t()) / 2`.
      The map is performed only when the `A` requires gradients.

    Returns:

      E (Tensor): tensor of eigenvalues of size :math:`(*, k)`

      X (Tensor): tensor of eigenvectors of size :math:`(*, m, k)`
    """
    # 检查是否处于非脚本化状态（即非 Torch 脚本环境）
    if not torch.jit.is_scripting():
        # 定义包含张量操作的元组
        tensor_ops = (A, B, X, iK)
        # 检查 tensor_ops 中的对象类型是否都是 torch.Tensor 或 None 类型，并且具有 Torch 函数
        if not set(map(type, tensor_ops)).issubset(
            (torch.Tensor, type(None))
        ) and has_torch_function(tensor_ops):
            # 如果满足条件，则调用 handle_torch_function 处理 torch 函数
            return handle_torch_function(
                lobpcg,
                tensor_ops,
                A,
                k=k,
                B=B,
                X=X,
                n=n,
                iK=iK,
                niter=niter,
                tol=tol,
                largest=largest,
                method=method,
                tracker=tracker,
                ortho_iparams=ortho_iparams,
                ortho_fparams=ortho_fparams,
                ortho_bparams=ortho_bparams,
            )

    # 如果处于 Torch 脚本环境
    if not torch._jit_internal.is_scripting():
        # 检查矩阵 A 是否需要梯度或者矩阵 B 是否不为 None 且需要梯度
        if A.requires_grad or (B is not None and B.requires_grad):
            # 对称化操作，确保 A 和 B 是对称的，以便于一阶优化方法使用
            A_sym = (A + A.mT) / 2
            B_sym = (B + B.mT) / 2 if (B is not None) else None

            # 调用 LOBPCGAutogradFunction.apply 处理对称化后的 A_sym 和 B_sym
            return LOBPCGAutogradFunction.apply(
                A_sym,
                k,
                B_sym,
                X,
                n,
                iK,
                niter,
                tol,
                largest,
                method,
                tracker,
                ortho_iparams,
                ortho_fparams,
                ortho_bparams,
            )
    else:
        # 如果处于 Torch 脚本环境且 A 或 B 需要梯度，则抛出运行时错误
        if A.requires_grad or (B is not None and B.requires_grad):
            raise RuntimeError(
                "Script and require grads is not supported atm."
                "If you just want to do the forward, use .detach()"
                "on A and B before calling into lobpcg"
            )
    # 调用 _lobpcg 函数，进行特征值求解
    return _lobpcg(
        A,                  # 系统矩阵 A
        k,                  # 要计算的特征值数量
        B,                  # 可选的预处理矩阵 B
        X,                  # 初始特征向量矩阵 X
        n,                  # 矩阵 A 的维度
        iK,                 # 可选的预处理逆矩阵
        niter,              # 最大迭代次数
        tol,                # 允许的误差容限
        largest,            # 是否计算最大的特征值
        method,             # 解法选择
        tracker,            # 迭代跟踪器
        ortho_iparams,      # 正交化内部参数
        ortho_fparams,      # 正交化前处理参数
        ortho_bparams,      # 正交化后处理参数
    )
# 定义一个私有函数 `_lobpcg`，用于执行部分特定的特征值问题求解（LOBPCG算法）。
def _lobpcg(
    A: Tensor,  # 输入参数 A，要求是一个张量（Tensor），必须是方阵
    k: Optional[int] = None,  # 可选参数：要计算的特征对数量
    B: Optional[Tensor] = None,  # 可选参数：另一个张量 B，如果给定，则要求与 A 形状相同
    X: Optional[Tensor] = None,  # 可选参数：初始向量 X
    n: Optional[int] = None,  # 可选参数：特征对数量
    iK: Optional[Tensor] = None,  # 可选参数：未指定用途的张量 iK
    niter: Optional[int] = None,  # 可选参数：迭代次数
    tol: Optional[float] = None,  # 可选参数：容差值
    largest: Optional[bool] = None,  # 可选参数：是否计算最大的特征值对
    method: Optional[str] = None,  # 可选参数：求解方法，默认为 "ortho"
    tracker: None = None,  # 可选参数：追踪器，默认为 None
    ortho_iparams: Optional[Dict[str, int]] = None,  # 可选参数：正交方法的整数参数字典
    ortho_fparams: Optional[Dict[str, float]] = None,  # 可选参数：正交方法的浮点数参数字典
    ortho_bparams: Optional[Dict[str, bool]] = None,  # 可选参数：正交方法的布尔参数字典
) -> Tuple[Tensor, Tensor]:  # 返回值为两个张量的元组

    # A 必须是方阵:
    assert A.shape[-2] == A.shape[-1], A.shape

    if B is not None:
        # 如果给定 B，则要求 A 和 B 具有相同的形状:
        assert A.shape == B.shape, (A.shape, B.shape)

    # 获取 A 的浮点数类型
    dtype = _utils.get_floating_dtype(A)
    # 获取 A 的设备信息
    device = A.device

    # 如果未指定容差值 tol，则根据 dtype 设置默认值
    if tol is None:
        feps = {torch.float32: 1.2e-07, torch.float64: 2.23e-16}[dtype]
        tol = feps**0.5

    # 获取 A 的列数
    m = A.shape[-1]
    # 如果 k 未指定，则根据 X 是否为 None 设置默认值
    k = (1 if X is None else X.shape[-1]) if k is None else k
    # 如果 n 未指定，则根据 X 是否为 None 设置默认值
    n = (k if n is None else n) if X is None else X.shape[-1]

    # 如果 A 的行数小于 3 * n，则抛出错误
    if m < 3 * n:
        raise ValueError(
            f"LPBPCG algorithm is not applicable when the number of A rows (={m})"
            f" is smaller than 3 x the number of requested eigenpairs (={n})"
        )

    # 如果未指定求解方法，则设置为 "ortho"
    method = "ortho" if method is None else method

    # 设置整数参数字典 iparams
    iparams = {
        "m": m,
        "n": n,
        "k": k,
        "niter": 1000 if niter is None else niter,
    }

    # 设置浮点数参数字典 fparams
    fparams = {
        "tol": tol,
    }

    # 设置布尔参数字典 bparams
    bparams = {"largest": True if largest is None else largest}

    # 如果方法为 "ortho"，则更新参数字典
    if method == "ortho":
        if ortho_iparams is not None:
            iparams.update(ortho_iparams)
        if ortho_fparams is not None:
            fparams.update(ortho_fparams)
        if ortho_bparams is not None:
            bparams.update(ortho_bparams)
        iparams["ortho_i_max"] = iparams.get("ortho_i_max", 3)
        iparams["ortho_j_max"] = iparams.get("ortho_j_max", 3)
        fparams["ortho_tol"] = fparams.get("ortho_tol", tol)
        fparams["ortho_tol_drop"] = fparams.get("ortho_tol_drop", tol)
        fparams["ortho_tol_replace"] = fparams.get("ortho_tol_replace", tol)
        bparams["ortho_use_drop"] = bparams.get("ortho_use_drop", False)

    # 如果不是 Torch 的脚本模式，则设置 LOBPCG.call_tracker 为 LOBPCG_call_tracker
    if not torch.jit.is_scripting():
        LOBPCG.call_tracker = LOBPCG_call_tracker  # type: ignore[method-assign]
    # 检查张量 A 的维度是否大于 2
    if len(A.shape) > 2:
        # 计算除最后两个维度外其余维度的元素个数 N
        N = int(torch.prod(torch.tensor(A.shape[:-2])))
        # 将 A 重塑为 (N, 最后两个维度的形状)，并命名为 bA
        bA = A.reshape((N,) + A.shape[-2:])
        # 如果 B 不为 None，则将 B 重塑为 (N, A 的倒数第二和倒数第一维度的形状)，并命名为 bB
        bB = B.reshape((N,) + A.shape[-2:]) if B is not None else None
        # 如果 X 不为 None，则将 X 重塑为 (N, X 的倒数第二和倒数第一维度的形状)，并命名为 bX
        bX = X.reshape((N,) + X.shape[-2:]) if X is not None else None
        # 创建一个形状为 (N, k) 的空张量 bE，指定数据类型和设备
        bE = torch.empty((N, k), dtype=dtype, device=device)
        # 创建一个形状为 (N, m, k) 的空张量 bXret，指定数据类型和设备
        bXret = torch.empty((N, m, k), dtype=dtype, device=device)

        # 遍历 N 个子张量
        for i in range(N):
            # 获取当前子张量 A_
            A_ = bA[i]
            # 如果 bB 不为 None，则获取当前子张量 B_
            B_ = bB[i] if bB is not None else None
            # 如果 bX 不为 None，则获取当前子张量 X_
            X_ = (
                torch.randn((m, n), dtype=dtype, device=device) if bX is None else bX[i]
            )
            # 断言当前子张量 X_ 的形状为二维且为 (m, n)
            assert len(X_.shape) == 2 and X_.shape == (m, n), (X_.shape, (m, n))
            # 设置 iparams 中的 "batch_index" 键为当前索引 i
            iparams["batch_index"] = i
            # 创建 LOBPCG 的 worker 实例并运行
            worker = LOBPCG(A_, B_, X_, iK, iparams, fparams, bparams, method, tracker)
            worker.run()
            # 将 worker 的前 k 个特征值存入 bE 的第 i 行
            bE[i] = worker.E[:k]
            # 将 worker 的前 k 列特征向量存入 bXret 的第 i 个张量
            bXret[i] = worker.X[:, :k]

        # 如果不处于 Torch 脚本模式，则重置 LOBPCG.call_tracker 为 LOBPCG_call_tracker_orig
        if not torch.jit.is_scripting():
            LOBPCG.call_tracker = LOBPCG_call_tracker_orig  # type: ignore[method-assign]

        # 返回重塑后的结果张量 bE 和 bXret
        return bE.reshape(A.shape[:-2] + (k,)), bXret.reshape(A.shape[:-2] + (m, k))

    # 如果 A 的维度不大于 2，则执行以下代码

    # 如果 X 为 None，则创建一个形状为 (m, n) 的随机张量并指定数据类型和设备，否则使用给定的 X
    X = torch.randn((m, n), dtype=dtype, device=device) if X is None else X
    # 断言 X 的形状为二维且为 (m, n)
    assert len(X.shape) == 2 and X.shape == (m, n), (X.shape, (m, n))

    # 创建 LOBPCG 的 worker 实例
    worker = LOBPCG(A, B, X, iK, iparams, fparams, bparams, method, tracker)

    # 运行 LOBPCG worker
    worker.run()

    # 如果不处于 Torch 脚本模式，则重置 LOBPCG.call_tracker 为 LOBPCG_call_tracker_orig
    if not torch.jit.is_scripting():
        LOBPCG.call_tracker = LOBPCG_call_tracker_orig  # type: ignore[method-assign]

    # 返回 LOBPCG worker 的前 k 个特征值和前 k 列特征向量
    return worker.E[:k], worker.X[:, :k]
class LOBPCG:
    """LOBPCG方法的工作类。"""

    def __init__(
        self,
        A: Optional[Tensor],
        B: Optional[Tensor],
        X: Tensor,
        iK: Optional[Tensor],
        iparams: Dict[str, int],
        fparams: Dict[str, float],
        bparams: Dict[str, bool],
        method: str,
        tracker: None,
    ) -> None:
        # 常量参数初始化
        self.A = A  # 存储参数 A
        self.B = B  # 存储参数 B
        self.iK = iK  # 存储参数 iK
        self.iparams = iparams  # 存储整型参数字典 iparams
        self.fparams = fparams  # 存储浮点型参数字典 fparams
        self.bparams = bparams  # 存储布尔型参数字典 bparams
        self.method = method  # 存储方法名称
        self.tracker = tracker  # 存储跟踪器，通常为 None
        m = iparams["m"]  # 从 iparams 中获取 m 的值
        n = iparams["n"]  # 从 iparams 中获取 n 的值

        # 变量参数初始化
        self.X = X  # 存储参数 X
        self.E = torch.zeros((n,), dtype=X.dtype, device=X.device)  # 初始化全零向量 E
        self.R = torch.zeros((m, n), dtype=X.dtype, device=X.device)  # 初始化全零矩阵 R
        self.S = torch.zeros((m, 3 * n), dtype=X.dtype, device=X.device)  # 初始化全零矩阵 S
        self.tvars: Dict[str, Tensor] = {}  # 空字典，用于存储张量类型的临时变量
        self.ivars: Dict[str, int] = {"istep": 0}  # 包含一个整型变量 istep 的字典
        self.fvars: Dict[str, float] = {"_": 0.0}  # 包含一个浮点型变量 _ 的字典
        self.bvars: Dict[str, bool] = {"_": False}  # 包含一个布尔型变量 _ 的字典

    def __str__(self):
        lines = ["LOPBCG:"]
        lines += [f"  iparams={self.iparams}"]  # 打印 iparams 字典
        lines += [f"  fparams={self.fparams}"]  # 打印 fparams 字典
        lines += [f"  bparams={self.bparams}"]  # 打印 bparams 字典
        lines += [f"  ivars={self.ivars}"]  # 打印 ivars 字典
        lines += [f"  fvars={self.fvars}"]  # 打印 fvars 字典
        lines += [f"  bvars={self.bvars}"]  # 打印 bvars 字典
        lines += [f"  tvars={self.tvars}"]  # 打印 tvars 字典
        lines += [f"  A={self.A}"]  # 打印参数 A
        lines += [f"  B={self.B}"]  # 打印参数 B
        lines += [f"  iK={self.iK}"]  # 打印参数 iK
        lines += [f"  X={self.X}"]  # 打印参数 X
        lines += [f"  E={self.E}"]  # 打印变量 E
        r = ""
        for line in lines:
            r += line + "\n"
        return r

    def update(self):
        """设置并更新迭代变量。"""
        if self.ivars["istep"] == 0:
            X_norm = float(torch.norm(self.X))  # 计算参数 X 的范数并转换为浮点数
            iX_norm = X_norm**-1  # 计算 X 范数的倒数
            A_norm = float(torch.norm(_utils.matmul(self.A, self.X))) * iX_norm  # 计算 A * X 的范数
            B_norm = float(torch.norm(_utils.matmul(self.B, self.X))) * iX_norm  # 计算 B * X 的范数
            self.fvars["X_norm"] = X_norm  # 存储 X 的范数
            self.fvars["A_norm"] = A_norm  # 存储 A * X 的范数
            self.fvars["B_norm"] = B_norm  # 存储 B * X 的范数
            self.ivars["iterations_left"] = self.iparams["niter"]  # 设置剩余迭代次数
            self.ivars["converged_count"] = 0  # 设置收敛计数器为 0
            self.ivars["converged_end"] = 0  # 设置收敛结束标志为 0

        if self.method == "ortho":
            self._update_ortho()  # 调用内部方法 _update_ortho 更新迭代
        else:
            self._update_basic()  # 调用内部方法 _update_basic 更新迭代

        self.ivars["iterations_left"] = self.ivars["iterations_left"] - 1  # 更新剩余迭代次数
        self.ivars["istep"] = self.ivars["istep"] + 1  # 更新迭代步数

    def update_residual(self):
        """从 A, B, X, E 更新残差 R。"""
        mm = _utils.matmul
        self.R = mm(self.A, self.X) - mm(self.B, self.X) * self.E
    def update_converged_count(self):
        """Determine the number of converged eigenpairs using backward stable
        convergence criterion, see discussion in Sec 4.3 of [DuerschEtal2018].

        Users may redefine this method for custom convergence criteria.
        """
        # 获取先前收敛的特征对数量
        prev_count = self.ivars["converged_count"]
        # 获取收敛容差阈值
        tol = self.fparams["tol"]
        # 获取矩阵 A 的范数
        A_norm = self.fvars["A_norm"]
        # 获取矩阵 B 的范数
        B_norm = self.fvars["B_norm"]
        # 获取当前的特征值、特征向量和残差
        E, X, R = self.E, self.X, self.R
        # 计算相对残差 rerr
        rerr = (
            torch.norm(R, 2, (0,))
            * (torch.norm(X, 2, (0,)) * (A_norm + E[: X.shape[-1]] * B_norm)) ** -1
        )
        # 判断哪些特征对已经收敛
        converged = rerr < tol
        # 计算已收敛的特征对数量
        count = 0
        for b in converged:
            if not b:
                # 忽略接下来的特征对的收敛以确保特征对的严格排序
                break
            count += 1
        # 断言已收敛的特征对数量不会减少
        assert (
            count >= prev_count
        ), f"the number of converged eigenpairs (was {prev_count}, got {count}) cannot decrease"
        # 更新收敛的特征对数量
        self.ivars["converged_count"] = count
        # 存储相对残差 rerr
        self.tvars["rerr"] = rerr
        # 返回已收敛的特征对数量
        return count

    def stop_iteration(self):
        """Return True to stop iterations.

        Note that tracker (if defined) can force-stop iterations by
        setting ``worker.bvars['force_stop'] = True``.
        """
        # 如果强制停止标志被设置，或者迭代次数已用完，或者已达到收敛特征对的数量上限，则返回 True 停止迭代
        return (
            self.bvars.get("force_stop", False)
            or self.ivars["iterations_left"] == 0
            or self.ivars["converged_count"] >= self.iparams["k"]
        )

    def run(self):
        """Run LOBPCG iterations.

        Use this method as a template for implementing LOBPCG
        iteration scheme with custom tracker that is compatible with
        TorchScript.
        """
        # 更新状态
        self.update()

        # 如果不在 TorchScript 模式且定义了追踪器，则调用追踪器
        if not torch.jit.is_scripting() and self.tracker is not None:
            self.call_tracker()

        # 当未达到停止条件时持续迭代
        while not self.stop_iteration():
            # 更新状态
            self.update()

            # 如果不在 TorchScript 模式且定义了追踪器，则调用追踪器
            if not torch.jit.is_scripting() and self.tracker is not None:
                self.call_tracker()

    @torch.jit.unused
    def call_tracker(self):
        """Interface for tracking iteration process in Python mode.

        Tracking the iteration process is disabled in TorchScript
        mode. In fact, one should specify tracker=None when JIT
        compiling functions using lobpcg.
        """
        # 在 TorchScript 模式下不执行任何操作
        # 在 Python 模式下用于追踪迭代过程的接口
        pass

    # Internal methods
    def _update_basic(self):
        """
        Update or initialize iteration variables when `method == "basic"`.
        """
        # 定义矩阵乘法函数为 mm，简化后续代码
        mm = torch.matmul
        # 获取已收敛的结束位置和计数
        ns = self.ivars["converged_end"]
        nc = self.ivars["converged_count"]
        # 获取问题尺寸 n 和是否找到最大特征值 largest
        n = self.iparams["n"]
        largest = self.bparams["largest"]

        # 若当前迭代步骤为 0
        if self.ivars["istep"] == 0:
            # 获取 Rayleigh-Ritz 变换 Ri
            Ri = self._get_rayleigh_ritz_transform(self.X)
            # 构造对称矩阵 M = X^T * A * Ri
            M = _utils.qform(_utils.qform(self.A, self.X), Ri)
            # 求解对称特征值问题 E, Z = eig(M)，Z 是特征向量矩阵
            E, Z = _utils.symeig(M, largest)
            # 更新 X = X * Ri * Z
            self.X[:] = mm(self.X, mm(Ri, Z))
            # 更新特征值数组 E
            self.E[:] = E
            # 更新残差向量
            self.update_residual()
            # 更新已收敛计数
            nc = self.update_converged_count()
            # 更新 S 的前 n 列为 X
            self.S[..., :n] = self.X

            # 计算 W = iK * R 并更新已收敛结束位置
            W = _utils.matmul(self.iK, self.R)
            self.ivars["converged_end"] = ns = n + np + W.shape[-1]
            # 更新 S 的列范围为 n+np 到 ns 的部分为 W
            self.S[:, n + np : ns] = W
        else:
            # 从 S 中取出列范围为 nc 到 ns 的子矩阵 S_
            S_ = self.S[:, nc:ns]
            # 获取 Rayleigh-Ritz 变换 Ri
            Ri = self._get_rayleigh_ritz_transform(S_)
            # 构造对称矩阵 M = S_^T * A * Ri
            M = _utils.qform(_utils.qform(self.A, S_), Ri)
            # 求解对称特征值问题 E_, Z = eig(M)，Z 是特征向量矩阵
            E_, Z = _utils.symeig(M, largest)
            # 更新 X 的列范围为 nc 到末尾为 S_ * Ri * Z 的前 n-nc 列
            self.X[:, nc:] = mm(S_, mm(Ri, Z[:, : n - nc]))
            # 更新特征值数组 E 的子数组
            self.E[nc:] = E_[: n - nc]
            # 计算 S_ * Ri * Z 的列范围为 n 到 2n-nc 的部分 P
            P = mm(S_, mm(Ri, Z[:, n : 2 * n - nc]))
            # 计算 P 的列数 np
            np = P.shape[-1]

            # 更新残差向量
            self.update_residual()
            # 更新已收敛计数
            nc = self.update_converged_count()
            # 更新 S 的前 n 列为 X
            self.S[..., :n] = self.X
            # 更新 S 的列范围为 n 到 n+np 的部分为 P
            self.S[:, n : n + np] = P
            # 计算 W = iK * R 的列范围为 nc 到末尾
            W = _utils.matmul(self.iK, self.R[:, nc:])

            # 更新已收敛结束位置
            self.ivars["converged_end"] = ns = n + np + W.shape[-1]
            # 更新 S 的列范围为 n+np 到 ns 的部分为 W
            self.S[:, n + np : ns] = W
    def _update_ortho(self):
        """
        Update or initialize iteration variables when `method == "ortho"`.
        """
        # 导入 torch 中的矩阵乘法函数 mm
        mm = torch.matmul
        # 获取已收敛的结束索引和计数
        ns = self.ivars["converged_end"]
        nc = self.ivars["converged_count"]
        # 获取迭代参数中的 n
        n = self.iparams["n"]
        # 获取基础参数中的 largest
        largest = self.bparams["largest"]

        # 如果当前步骤数为 0
        if self.ivars["istep"] == 0:
            # 计算 Rayleigh-Ritz 变换 Ri
            Ri = self._get_rayleigh_ritz_transform(self.X)
            # 计算 M = A*X*Ri*Ri
            M = _utils.qform(_utils.qform(self.A, self.X), Ri)
            # 对 M 进行对角化得到特征值 E 和特征向量 Z
            E, Z = _utils.symeig(M, largest)
            # 更新 X = X*Ri*Z
            self.X = mm(self.X, mm(Ri, Z))
            # 更新残差
            self.update_residual()
            # 更新已收敛计数
            np = 0
            nc = self.update_converged_count()
            # 将前 n 列设为新的特征向量
            self.S[:, :n] = self.X
            # 获取正交基 W
            W = self._get_ortho(self.R, self.X)
            # 更新已收敛结束索引 ns
            ns = self.ivars["converged_end"] = n + np + W.shape[-1]
            # 将新的正交基 W 加入 S 中
            self.S[:, n + np : ns] = W

        else:
            # 否则，取出 S 中已收敛的部分 S_
            S_ = self.S[:, nc:ns]
            # 使用 Rayleigh-Ritz 过程对 S_ 进行操作
            E_, Z = _utils.symeig(_utils.qform(self.A, S_), largest)

            # 更新 E, X, P
            self.X[:, nc:] = mm(S_, Z[:, : n - nc])
            self.E[nc:] = E_[: n - nc]
            # 计算 P = S_*Z*Z.T*basis(Z[:,:])
            P = mm(
                S_,
                mm(
                    Z[:, n - nc :],
                    _utils.basis(Z[: n - nc, n - nc :].mT),
                ),
            )
            # 计算 P 的维度
            np = P.shape[-1]

            # 检查收敛性
            self.update_residual()
            # 更新已收敛计数
            nc = self.update_converged_count()

            # 更新 S
            self.S[:, :n] = self.X
            self.S[:, n : n + np] = P
            # 获取新的正交基 W
            W = self._get_ortho(self.R[:, nc:], self.S[:, : n + np])
            # 更新已收敛结束索引 ns
            ns = self.ivars["converged_end"] = n + np + W.shape[-1]
            # 将新的正交基 W 加入 S 中
            self.S[:, n + np : ns] = W
    def _get_rayleigh_ritz_transform(self, S):
        """Return a transformation matrix that is used in Rayleigh-Ritz
        procedure for reducing a general eigenvalue problem :math:`(S^TAS)
        C = (S^TBS) C E` to a standard eigenvalue problem :math: `(Ri^T
        S^TAS Ri) Z = Z E` where `C = Ri Z`.

        .. note:: In the original Rayleigh-Ritz procedure in
          [DuerschEtal2018], the problem is formulated as follows::

            SAS = S^T A S
            SBS = S^T B S
            D = (<diagonal matrix of SBS>) ** -1/2
            R^T R = Cholesky(D SBS D)
            Ri = D R^-1
            solve symeig problem Ri^T SAS Ri Z = Theta Z
            C = Ri Z

          To reduce the number of matrix products (denoted by empty
          space between matrices), here we introduce element-wise
          products (denoted by symbol `*`) so that the Rayleigh-Ritz
          procedure becomes::

            SAS = S^T A S
            SBS = S^T B S
            d = (<diagonal of SBS>) ** -1/2    # this is 1-d column vector
            dd = d d^T                         # this is 2-d matrix
            R^T R = Cholesky(dd * SBS)
            Ri = R^-1 * d                      # broadcasting
            solve symeig problem Ri^T SAS Ri Z = Theta Z
            C = Ri Z

          where `dd` is 2-d matrix that replaces matrix products `D M
          D` with one element-wise product `M * dd`; and `d` replaces
          matrix product `D M` with element-wise product `M *
          d`. Also, creating the diagonal matrix `D` is avoided.

        Args:
        S (Tensor): the matrix basis for the search subspace, size is
                    :math:`(m, n)`.

        Returns:
        Ri (tensor): upper-triangular transformation matrix of size
                     :math:`(n, n)`.

        """
        B = self.B  # 获取类成员变量 B
        mm = torch.matmul  # 定义 torch 中的矩阵乘法函数的别名 mm
        SBS = _utils.qform(B, S)  # 计算 SBS = S^T B S，使用外部函数 _utils.qform
        d_row = SBS.diagonal(0, -2, -1) ** -0.5  # 计算 SBS 的对角元素向量的平方根倒数
        d_col = d_row.reshape(d_row.shape[0], 1)  # 将 d_row 变形为列向量
        # TODO use torch.linalg.cholesky_solve once it is implemented
        R = torch.linalg.cholesky((SBS * d_row) * d_col, upper=True)  # 计算 Cholesky 分解结果 R
        return torch.linalg.solve_triangular(
            R, d_row.diag_embed(), upper=True, left=False
        )  # 使用三角求解器解线性方程组，返回变换矩阵 Ri

    def _get_svqb(
        self, U: Tensor, drop: bool, tau: float  # Tensor  # bool  # float
    ) -> Tensor:
        """Return B-orthonormal U.

        .. note:: When `drop` is `False` then `svqb` is based on the
                  Algorithm 4 from [DuerschPhD2015] that is a slight
                  modification of the corresponding algorithm
                  introduced in [StathopolousWu2002].

        Args:

          U (Tensor) : initial approximation, size is (m, n)
          drop (bool) : when True, drop columns that
                     contribution to the `span([U])` is small.
          tau (float) : positive tolerance

        Returns:

          U (Tensor) : B-orthonormal columns (:math:`U^T B U = I`), size
                       is (m, n1), where `n1 = n` if `drop` is `False,
                       otherwise `n1 <= n`.

        """
        # Check if U is empty, if so, return U as it is
        if torch.numel(U) == 0:
            return U
        
        # Compute UBU = U^T * B * U using the qform utility function
        UBU = _utils.qform(self.B, U)
        
        # Extract the diagonal elements of UBU
        d = UBU.diagonal(0, -2, -1)

        # Detect and drop exact zero columns from U.
        # This section ensures robustness against exact zero columns
        # in U, preventing potential failures in subsequent operations.
        nz = torch.where(abs(d) != 0.0)
        assert len(nz) == 1, nz
        if len(nz[0]) < len(d):
            # Drop columns from U that are exact zeros
            U = U[:, nz[0]]
            if torch.numel(U) == 0:
                return U
            # Recompute UBU and d after dropping columns
            UBU = _utils.qform(self.B, U)
            d = UBU.diagonal(0, -2, -1)
            nz = torch.where(abs(d) != 0.0)
            assert len(nz[0]) == len(d)

        # Proceed with Algorithm 4 from [DuerschPhD2015].
        
        # Compute d_col = d^-0.5, reshaping it to match dimensions of d
        d_col = (d**-0.5).reshape(d.shape[0], 1)
        
        # Construct DUBUD = UBU * d_col * d_col^T
        DUBUD = UBU * d_col * d_col.T
        
        # Compute eigenvalues and eigenvectors of DUBUD
        E, Z = _utils.symeig(DUBUD)
        
        # Compute threshold value t based on tau and the maximum absolute value in E
        t = tau * abs(E).max()
        
        # Optionally drop columns based on the drop parameter and threshold t
        if drop:
            keep = torch.where(E > t)
            assert len(keep) == 1, keep
            E = E[keep[0]]
            Z = Z[:, keep[0]]
            d_col = d_col[keep[0]]
        else:
            E[(torch.where(E < t))[0]] = t

        # Return the final result of the orthogonalization process
        return torch.matmul(U * d_col.T, Z * E**-0.5)
# 将原始的 LOBPCG 调用跟踪器函数引用保存在变量中，以便稍后使用
LOBPCG_call_tracker_orig = LOBPCG.call_tracker

# 定义 LOBPCG_call_tracker 函数，用于替代 LOBPCG 类中的 call_tracker 方法
def LOBPCG_call_tracker(self):
    # 调用存储在 self.tracker 中的回调函数，并传入当前对象 self 作为参数
    self.tracker(self)
```