# `.\pytorch\torch\linalg\__init__.py`

```py
import sys  # 导入sys模块，用于访问系统相关的功能

import torch  # 导入torch模块，PyTorch深度学习库的核心包
from torch._C import _add_docstr, _linalg  # 导入torch._C模块中的_add_docstr和_linalg，用于文档字符串添加和线性代数操作

LinAlgError = torch._C._LinAlgError  # 将torch._C._LinAlgError赋值给LinAlgError，用于处理线性代数错误

Tensor = torch.Tensor  # 将torch.Tensor赋值给Tensor，表示Tensor类型的别名

common_notes = {  # 定义一个字典common_notes，包含常见的文档说明
    "experimental_warning": """This function is "experimental" and it may change in a future PyTorch release.""",
    "sync_note": "When inputs are on a CUDA device, this function synchronizes that device with the CPU.",
    "sync_note_ex": r"When the inputs are on a CUDA device, this function synchronizes only when :attr:`check_errors`\ `= True`.",
    "sync_note_has_ex": ("When inputs are on a CUDA device, this function synchronizes that device with the CPU. "
                         "For a version of this function that does not synchronize, see :func:`{}`.")
}


# Note: This not only adds doc strings for functions in the linalg namespace, but
# also connects the torch.linalg Python namespace to the torch._C._linalg builtins.

cross = _add_docstr(_linalg.linalg_cross, r"""
linalg.cross(input, other, *, dim=-1, out=None) -> Tensor


Computes the cross product of two 3-dimensional vectors.

Supports input of float, double, cfloat and cdouble dtypes. Also supports batches
of vectors, for which it computes the product along the dimension :attr:`dim`.
It broadcasts over the batch dimensions.

Args:
    input (Tensor): the first input tensor.
    other (Tensor): the second input tensor.
    dim  (int, optional): the dimension along which to take the cross-product. Default: `-1`.

Keyword args:
    out (Tensor, optional): the output tensor. Ignored if `None`. Default: `None`.

Example:
    >>> a = torch.randn(4, 3)
    >>> a
    tensor([[-0.3956,  1.1455,  1.6895],
            [-0.5849,  1.3672,  0.3599],
            [-1.1626,  0.7180, -0.0521],
            [-0.1339,  0.9902, -2.0225]])
    >>> b = torch.randn(4, 3)
    >>> b
    tensor([[-0.0257, -1.4725, -1.2251],
            [-1.1479, -0.7005, -1.9757],
            [-1.3904,  0.3726, -1.1836],
            [-0.9688, -0.7153,  0.2159]])
    >>> torch.linalg.cross(a, b)
    tensor([[ 1.0844, -0.5281,  0.6120],
            [-2.4490, -1.5687,  1.9792],
            [-0.8304, -1.3037,  0.5650],
            [-1.2329,  1.9883,  1.0551]])
    >>> a = torch.randn(1, 3)  # a is broadcast to match shape of b
    >>> a
    tensor([[-0.9941, -0.5132,  0.5681]])
    >>> torch.linalg.cross(a, b)
    tensor([[ 1.4653, -1.2325,  1.4507],
            [ 1.4119, -2.6163,  0.1073],
            [ 0.3957, -1.9666, -1.0840],
            [ 0.2956, -0.3357,  0.2139]])
""")  # 添加文档字符串到torch.linalg.cross函数，描述其计算两个3维向量的叉积操作

cholesky = _add_docstr(_linalg.linalg_cholesky, r"""
linalg.cholesky(A, *, upper=False, out=None) -> Tensor

Computes the Cholesky decomposition of a complex Hermitian or real symmetric positive-definite matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **Cholesky decomposition** of a complex Hermitian or real symmetric positive-definite matrix
:math:`A \in \mathbb{K}^{n \times n}` is defined as

.. math::
    A = LL^* \quad \text{or} \quad A = U^*U

where :math:`L` is a lower triangular matrix with real and positive diagonal entries,
and :math:`U` is an upper triangular matrix with real and positive diagonal entries.

Args:
    A (Tensor): the input matrix to be decomposed.
    upper (bool, optional): whether to compute the upper or lower triangular Cholesky factor. Default: `False`.
    out (Tensor, optional): the output tensor. Ignored if `None`. Default: `None`.

Returns:
    Tensor: the Cholesky factor of :attr:`A`.

Raises:
    RuntimeError: if :attr:`A` is not positive-definite.

Example:
    >>> A = torch.tensor([[10., 2.], [2., 10.]])
    >>> torch.linalg.cholesky(A)
    tensor([[3.1623, 0.0000],
            [0.6325, 3.0777]])
""")  # 添加文档字符串到torch.linalg.cholesky函数，描述其计算正定实对称矩阵的Cholesky分解
    A = LL^{\text{H}}\mathrlap{\qquad L \in \mathbb{K}^{n \times n}}


    # 定义矩阵 A 为 L 和其共轭转置 L^H 的乘积
    A = LL^{\text{H}}\mathrlap{\qquad L \in \mathbb{K}^{n \times n}}
where :math:`L` is a lower triangular matrix with real positive diagonal (even in the complex case) and
:math:`L^{\text{H}}` is the conjugate transpose when :math:`L` is complex, and the transpose when :math:`L` is real-valued.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

""" + fr"""
.. note:: {common_notes["sync_note_has_ex"].format("torch.linalg.cholesky_ex")}
""" + r"""

.. seealso::

        :func:`torch.linalg.cholesky_ex` for a version of this operation that
        skips the (slow) error checking by default and instead returns the debug
        information. This makes it a faster way to check if a matrix is
        positive-definite.

        :func:`torch.linalg.eigh` for a different decomposition of a Hermitian matrix.
        The eigenvalue decomposition gives more information about the matrix but it
        slower to compute than the Cholesky decomposition.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of symmetric or Hermitian positive-definite matrices.

Keyword args:
    upper (bool, optional): whether to return an upper triangular matrix.
        The tensor returned with upper=True is the conjugate transpose of the tensor
        returned with upper=False.
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if the :attr:`A` matrix or any matrix in a batched :attr:`A` is not Hermitian
                  (resp. symmetric) positive-definite. If :attr:`A` is a batch of matrices,
                  the error message will include the batch index of the first matrix that fails
                  to meet this condition.

Examples::

    >>> A = torch.randn(2, 2, dtype=torch.complex128)
    >>> A = A @ A.T.conj() + torch.eye(2) # creates a Hermitian positive-definite matrix
    >>> A
    tensor([[2.5266+0.0000j, 1.9586-2.0626j],
            [1.9586+2.0626j, 9.4160+0.0000j]], dtype=torch.complex128)
    >>> L = torch.linalg.cholesky(A)
    >>> L
    tensor([[1.5895+0.0000j, 0.0000+0.0000j],
            [1.2322+1.2976j, 2.4928+0.0000j]], dtype=torch.complex128)
    >>> torch.dist(L @ L.T.conj(), A)
    tensor(4.4692e-16, dtype=torch.float64)

    >>> A = torch.randn(3, 2, 2, dtype=torch.float64)
    >>> A = A @ A.mT + torch.eye(2)  # batch of symmetric positive-definite matrices
    >>> L = torch.linalg.cholesky(A)
    >>> torch.dist(L @ L.mT, A)
    tensor(5.8747e-16, dtype=torch.float64)
"""

cholesky_ex = _add_docstr(_linalg.linalg_cholesky_ex, r"""
linalg.cholesky_ex(A, *, upper=False, check_errors=False, out=None) -> (Tensor, Tensor)

Computes the Cholesky decomposition of a complex Hermitian or real
symmetric positive-definite matrix.

This function skips the (slow) error checking and error message construction
inv = _add_docstr(_linalg.linalg_inv, r"""
linalg.inv(A, *, out=None) -> Tensor

Computes the inverse of a square matrix if it exists.
Throws a `RuntimeError` if the matrix is not invertible.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
for a matrix :math:`A \in \mathbb{K}^{n \times n}`,
its **inverse matrix** :math:`A^{-1} \in \mathbb{K}^{n \times n}` (if it exists) is defined as

.. math::

    A^{-1}A = AA^{-1} = \mathrm{I}_n

where :math:`\mathrm{I}_n` is the `n`-dimensional identity matrix.

""" + fr"""
.. note:: {common_notes["sync_note_ex"]}

.. warning:: {common_notes["experimental_warning"]}
""" + r"""

.. seealso::
        :func:`torch.linalg.cholesky` is a NumPy compatible variant that always checks for errors.

Args:
    A (Tensor): the Hermitian `n \times n` matrix or the batch of such matrices of size
                    `(*, n, n)` where `*` is one or more batch dimensions.

Keyword args:
    out (Tensor, optional): the output tensor for the inverse matrices. Ignored if `None`.
""" + r"""

Examples::

    >>> A = torch.randn(2, 2, dtype=torch.float64)
    >>> A
    tensor([[ 0.1336, -1.2027],
            [-0.7883,  0.5706]], dtype=torch.float64)
    >>> A_inv = torch.linalg.inv(A)
    >>> A_inv
    tensor([[ 3.6001,  2.6727],
            [ 4.7787,  1.6687]], dtype=torch.float64)
""")
"""
The inverse matrix exists if and only if :math:`A` is `invertible`_. In this case,
the inverse is unique.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices
then the output has the same batch dimensions.

""" + fr"""
.. note:: {common_notes["sync_note_has_ex"].format("torch.linalg.inv_ex")}
""" + r"""

.. note::
    Consider using :func:`torch.linalg.solve` if possible for multiplying a matrix on the left by
    the inverse, as::

        linalg.solve(A, B) == linalg.inv(A) @ B  # When B is a matrix

    It is always preferred to use :func:`~solve` when possible, as it is faster and more
    numerically stable than computing the inverse explicitly.

.. seealso::

        :func:`torch.linalg.pinv` computes the pseudoinverse (Moore-Penrose inverse) of matrices
        of any shape.

        :func:`torch.linalg.solve` computes :attr:`A`\ `.inv() @ \ `:attr:`B` with a
        numerically stable algorithm.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of invertible matrices.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if the matrix :attr:`A` or any matrix in the batch of matrices :attr:`A` is not invertible.

Examples::

    >>> A = torch.randn(4, 4)
    >>> Ainv = torch.linalg.inv(A)
    >>> torch.dist(A @ Ainv, torch.eye(4))
    tensor(1.1921e-07)

    >>> A = torch.randn(2, 3, 4, 4)  # Batch of matrices
    >>> Ainv = torch.linalg.inv(A)
    >>> torch.dist(A @ Ainv, torch.eye(4))
    tensor(1.9073e-06)

    >>> A = torch.randn(4, 4, dtype=torch.complex128)  # Complex matrix
    >>> Ainv = torch.linalg.inv(A)
    >>> torch.dist(A @ Ainv, torch.eye(4))
    tensor(7.5107e-16, dtype=torch.float64)

.. _invertible:
    https://en.wikipedia.org/wiki/Invertible_matrix#The_invertible_matrix_theorem
"""

solve_ex = _add_docstr(_linalg.linalg_solve_ex, r"""
linalg.solve_ex(A, B, *, left=True, check_errors=False, out=None) -> (Tensor, Tensor)

A version of :func:`~solve` that does not perform error checks unless :attr:`check_errors`\ `= True`.
It also returns the :attr:`info` tensor returned by `LAPACK's getrf`_.

""" + fr"""
.. note:: {common_notes["sync_note_ex"]}

.. warning:: {common_notes["experimental_warning"]}
""" + r"""

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.

Keyword args:
    left (bool, optional): whether to solve the system :math:`AX=B` or :math:`XA = B`. Default: `True`.
    check_errors (bool, optional): controls whether to check the content of ``infos`` and raise
                                   an error if it is non-zero. Default: `False`.
    out (tuple, optional): tuple of two tensors to write the output to. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(result, info)`.

Examples::

    >>> A = torch.randn(3, 3)
"""
    # 使用 torch.linalg.solve_ex 函数求解线性代数方程 A * X = I，返回解矩阵 Ainv 和信息对象 info
    >>> Ainv, info = torch.linalg.solve_ex(A)
    # 计算 A 的逆矩阵和 Ainv 之间的 Frobenius 范数，理论上应该接近零
    >>> torch.dist(torch.linalg.inv(A), Ainv)
    tensor(0.)
    # info 对象应为整数类型的张量，表示求解过程中的状态信息
    >>> info
    tensor(0, dtype=torch.int32)
# 将字符串作为内部文档字符串添加到 _linalg.linalg_inv_ex 函数上
inv_ex = _add_docstr(_linalg.linalg_inv_ex, r"""
linalg.inv_ex(A, *, check_errors=False, out=None) -> (Tensor, Tensor)

计算方阵的逆（如果可逆）。

返回一个命名元组 ``(inverse, info)``。``inverse`` 包含对 :attr:`A` 求逆的结果，``info`` 存储 LAPACK 错误代码。

如果 :attr:`A` 不是可逆矩阵，或者如果它是一批矩阵且其中一个或多个不可逆，那么 ``info`` 对应的值为正整数。
正整数表示输入矩阵 LU 分解的对角线元素恰好为零。
``info`` 全为零表示求逆成功。如果 ``check_errors=True`` 且 ``info`` 包含正整数，则会引发 RuntimeError。

支持 float、double、cfloat 和 cdouble 数据类型的输入。
还支持矩阵批处理，如果 :attr:`A` 是矩阵批处理，则输出具有相同的批处理维度。

""" + fr"""
.. note:: {common_notes["sync_note_ex"]}

.. warning:: {common_notes["experimental_warning"]}
""" + r"""

.. seealso::

        :func:`torch.linalg.inv` 是一个与 NumPy 兼容的变体，始终会检查错误。

Args:
    A (Tensor): 形状为 `(*, n, n)` 的张量，其中 `*` 是零个或多个批处理维度，包含方阵。
    check_errors (bool, optional): 控制是否检查 ``info`` 的内容。默认为 `False`。

Keyword args:
    out (tuple, optional): 写入输出的两个张量组成的元组。如果为 `None`，则忽略。默认为 `None`。

Examples::

    >>> A = torch.randn(3, 3)
    >>> Ainv, info = torch.linalg.inv_ex(A)
    >>> torch.dist(torch.linalg.inv(A), Ainv)
    tensor(0.)
    >>> info
    tensor(0, dtype=torch.int32)

""")

# 将字符串作为内部文档字符串添加到 _linalg.linalg_det 函数上
det = _add_docstr(_linalg.linalg_det, r"""
linalg.det(A, *, out=None) -> Tensor

计算方阵的行列式。

支持 float、double、cfloat 和 cdouble 数据类型的输入。
还支持矩阵批处理，如果 :attr:`A` 是矩阵批处理，则输出具有相同的批处理维度。

.. seealso::

        :func:`torch.linalg.slogdet` 计算方阵的符号和行列式绝对值的自然对数。

Args:
    A (Tensor): 形状为 `(*, n, n)` 的张量，其中 `*` 是零个或多个批处理维度的张量。

Keyword args:
    out (Tensor, optional): 输出张量。如果为 `None`，则忽略。默认为 `None`。

Examples::

    >>> A = torch.randn(3, 3)
    >>> torch.linalg.det(A)
    tensor(0.0934)

    >>> A = torch.randn(3, 2, 2)
    >>> torch.linalg.det(A)
    tensor([1.1990, 0.4099, 0.7386])
""")

# 将字符串作为内部文档字符串添加到 _linalg.linalg_slogdet 函数上
slogdet = _add_docstr(_linalg.linalg_slogdet, r"""
linalg.slogdet(A, *, out=None) -> (Tensor, Tensor)

计算方阵的行列式的符号和绝对值的自然对数。

""")
eig = _add_docstr(_linalg.linalg_eig, r"""
linalg.eig(A, *, out=None) -> (Tensor, Tensor)

计算方阵的特征值分解（如果存在）。

假设 :math:`\mathbb{K}` 是 :math:`\mathbb{R}` 或 :math:`\mathbb{C}`，
对于一个方阵 :math:`A \in \mathbb{K}^{n \times n}`，特征值分解（如果存在）定义为

.. math::

    A = V \operatorname{diag}(\Lambda) V^{-1}\mathrlap{\qquad V \in \mathbb{C}^{n \times n}, \Lambda \in \mathbb{C}^n}

此分解仅当 :math:`A` 是可对角化的时成立。
当所有特征值不同的时候，方阵是可对角化的。

支持 float、double、cfloat 和 cdouble 数据类型的输入。
还支持矩阵批处理，如果 :attr:`A` 是矩阵批处理，则输出具有相同的批处理维度。

返回的特征值没有特定顺序保证。

.. note:: 实矩阵的特征值和特征向量可能是复数。

""" + fr"""
.. note:: {common_notes["sync_note"]}
""" + r"""

.. warning:: 此函数假设 :attr:`A` 是可对角化的（例如，当所有特征值都不同的时候）。
             如果不能对角化，返回的特征值是正确的，但是 :math:`A \neq V \operatorname{diag}(\Lambda)V^{-1}`。

""")
# 警告：返回的特征向量已经被归一化为具有单位范数 `1`。
# 即便如此，矩阵的特征向量并不唯一，且与矩阵 :attr:`A` 不连续。
# 由于这种不唯一性，不同的硬件和软件可能会计算不同的特征向量。

# 这种非唯一性是由于将特征向量乘以 :math:`e^{i \phi}, \phi \in \mathbb{R}` 会产生另一组有效的特征向量。
# 因此，损失函数不应依赖于特征向量的相位，因为这个量并没有明确定义。
# 在计算此函数的梯度时会检查这一点。因此，在输入位于 CUDA 设备上时，计算此函数的梯度会同步该设备与 CPU。

# 警告：使用 `eigenvectors` 张量计算的梯度仅当矩阵 :attr:`A` 具有不同的特征值时才是有限的。
# 此外，如果任意两个特征值之间的距离接近零，则梯度将是数值不稳定的，
# 因为它依赖于特征值 :math:`\lambda_i` 通过计算 :math:`\frac{1}{\min_{i \neq j} \lambda_i - \lambda_j}`。

# 参见：

# - :func:`torch.linalg.eigvals` 仅计算特征值。
#   与 :func:`torch.linalg.eig` 不同，:func:`~eigvals` 的梯度始终是数值稳定的。

# - :func:`torch.linalg.eigh` 是一个（更快速）函数，用于计算埃尔米特和对称矩阵的特征值分解。

# - :func:`torch.linalg.svd` 是一个计算另一种可以用于任何形状矩阵的谱分解的函数。

# - :func:`torch.linalg.qr` 是另一个（更快速）的分解函数，可以用于任何形状的矩阵。

Args:
    A (Tensor): 形状为 `(*, n, n)` 的张量，其中 `*` 是零个或多个批量维度，
                包含可对角化的矩阵。

Keyword args:
    out (tuple, optional): 两个张量的输出元组。如果为 `None` 则忽略。默认为 `None`。

Returns:
    一个命名元组 `(eigenvalues, eigenvectors)`，对应于上述的 :math:`\Lambda` 和 :math:`V`。

    `eigenvalues` 和 `eigenvectors` 总是复数，即使 :attr:`A` 是实数。特征向量由 `eigenvectors` 的列给出。

示例：

>>> A = torch.randn(2, 2, dtype=torch.complex128)
>>> A
tensor([[ 0.9828+0.3889j, -0.4617+0.3010j],
        [ 0.1662-0.7435j, -0.6139+0.0562j]], dtype=torch.complex128)
>>> L, V = torch.linalg.eig(A)
>>> L
tensor([ 1.1226+0.5738j, -0.7537-0.1286j], dtype=torch.complex128)
>>> V
tensor([[ 0.9218+0.0000j,  0.1882-0.2220j],
        [-0.0270-0.3867j,  0.9567+0.0000j]], dtype=torch.complex128)
    # 计算特征值分解后的重构误差，验证特征值分解的准确性
    >>> torch.dist(V @ torch.diag(L) @ torch.linalg.inv(V), A)
    tensor(7.7119e-16, dtype=torch.float64)
    
    # 创建一个随机的张量 A，形状为 (3, 2, 2)，数据类型为 torch.float64
    >>> A = torch.randn(3, 2, 2, dtype=torch.float64)
    # 对张量 A 进行特征值分解，返回特征值 L 和特征向量 V
    >>> L, V = torch.linalg.eig(A)
    # 计算重构误差，验证特征值分解的准确性
    >>> torch.dist(V @ torch.diag_embed(L) @ torch.linalg.inv(V), A)
    tensor(3.2841e-16, dtype=torch.float64)
.. _diagonalizable:
    https://en.wikipedia.org/wiki/Diagonalizable_matrix#Definition
"""
"""

eigvals = _add_docstr(_linalg.linalg_eigvals, r"""
linalg.eigvals(A, *, out=None) -> Tensor

Computes the eigenvalues of a square matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **eigenvalues** of a square matrix :math:`A \in \mathbb{K}^{n \times n}` are defined
as the roots (counted with multiplicity) of the polynomial `p` of degree `n` given by

.. math::

    p(\lambda) = \operatorname{det}(A - \lambda \mathrm{I}_n)\mathrlap{\qquad \lambda \in \mathbb{C}}

where :math:`\mathrm{I}_n` is the `n`-dimensional identity matrix.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

The returned eigenvalues are not guaranteed to be in any specific order.

.. note:: The eigenvalues of a real matrix may be complex, as the roots of a real polynomial may be complex.

          The eigenvalues of a matrix are always well-defined, even when the matrix is not diagonalizable.

""" + fr"""
.. note:: {common_notes["sync_note"]}
""" + r"""

.. seealso::

        :func:`torch.linalg.eig` computes the full eigenvalue decomposition.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Returns:
    A complex-valued tensor containing the eigenvalues even when :attr:`A` is real.

Examples::

    >>> A = torch.randn(2, 2, dtype=torch.complex128)
    >>> L = torch.linalg.eigvals(A)
    >>> L
    tensor([ 1.1226+0.5738j, -0.7537-0.1286j], dtype=torch.complex128)

    >>> torch.dist(L, torch.linalg.eig(A).eigenvalues)
    tensor(2.4576e-07)
""")

eigh = _add_docstr(_linalg.linalg_eigh, r"""
linalg.eigh(A, UPLO='L', *, out=None) -> (Tensor, Tensor)

Computes the eigenvalue decomposition of a complex Hermitian or real symmetric matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **eigenvalue decomposition** of a complex Hermitian or real symmetric matrix
:math:`A \in \mathbb{K}^{n \times n}` is defined as

.. math::

    A = Q \operatorname{diag}(\Lambda) Q^{\text{H}}\mathrlap{\qquad Q \in \mathbb{K}^{n \times n}, \Lambda \in \mathbb{R}^n}

where :math:`Q^{\text{H}}` is the conjugate transpose when :math:`Q` is complex, and the transpose when :math:`Q` is real-valued.
:math:`Q` is orthogonal in the real case and unitary in the complex case.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

:attr:`A` is assumed to be Hermitian (resp. symmetric), but this is not checked internally, instead:

- If :attr:`UPLO`\ `= 'L'` (default), only the lower triangular part of the matrix is used in the computation.

"""
"""
- If :attr:`UPLO`\ `= 'U'`, only the upper triangular part of the matrix is used.
"""

"""
The eigenvalues are returned in ascending order.
"""

"""
.. note:: {common_notes["sync_note"]}
"""

"""
.. note:: The eigenvalues of real symmetric or complex Hermitian matrices are always real.
"""

"""
.. warning:: The eigenvectors of a symmetric matrix are not unique, nor are they continuous with
             respect to :attr:`A`. Due to this lack of uniqueness, different hardware and
             software may compute different eigenvectors.

             This non-uniqueness is caused by the fact that multiplying an eigenvector by
             `-1` in the real case or by :math:`e^{i \phi}, \phi \in \mathbb{R}` in the complex
             case produces another set of valid eigenvectors of the matrix.
             For this reason, the loss function shall not depend on the phase of the eigenvectors, as
             this quantity is not well-defined.
             This is checked for complex inputs when computing the gradients of this function. As such,
             when inputs are complex and are on a CUDA device, the computation of the gradients
             of this function synchronizes that device with the CPU.
"""

"""
.. warning:: Gradients computed using the `eigenvectors` tensor will only be finite when
             :attr:`A` has distinct eigenvalues.
             Furthermore, if the distance between any two eigenvalues is close to zero,
             the gradient will be numerically unstable, as it depends on the eigenvalues
             :math:`\lambda_i` through the computation of
             :math:`\frac{1}{\min_{i \neq j} \lambda_i - \lambda_j}`.
"""

"""
.. warning:: User may see pytorch crashes if running `eigh` on CUDA devices with CUDA versions before 12.1 update 1
             with large ill-conditioned matrices as inputs.
             Refer to :ref:`Linear Algebra Numerical Stability<Linear Algebra Stability>` for more details.
             If this is the case, user may (1) tune their matrix inputs to be less ill-conditioned,
             or (2) use :func:`torch.backends.cuda.preferred_linalg_library` to
             try other supported backends.
"""
eigvalsh = _add_docstr(_linalg.linalg_eigvalsh, r"""
linalg.eigvalsh(A, UPLO='L', *, out=None) -> Tensor

Computes the eigenvalues of a complex Hermitian or real symmetric matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **eigenvalues** of a complex Hermitian or real symmetric matrix :math:`A \in \mathbb{K}^{n \times n}`
are defined as the roots (counted with multiplicity) of the polynomial `p` of degree `n` given by

.. math::

    det(A - \lambda I) = 0,

where :math:`\lambda` are the eigenvalues of :math:`A`.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of symmetric or Hermitian matrices.
    UPLO ('L', 'U', optional): controls whether to use the upper or lower triangular part
                               of :attr:`A` in the computations. Default: `'L'`.

Keyword args:
    out (tuple, optional): output tuple of two tensors. Ignored if `None`. Default: `None`.

Returns:
    Tensor: the eigenvalues of matrix :attr:`A`.

Examples::
    >>> A = torch.randn(2, 2, dtype=torch.complex128)
    >>> A = A + A.T.conj()  # creates a Hermitian matrix
    >>> A
    tensor([[2.9228+0.0000j, 0.2029-0.0862j],
            [0.2029+0.0862j, 0.3464+0.0000j]], dtype=torch.complex128)
    >>> L = torch.linalg.eigvalsh(A)
    >>> L
    tensor([0.3277, 2.9415], dtype=torch.float64)
""")
    p(\lambda) = \operatorname{det}(A - \lambda \mathrm{I}_n)\mathrlap{\qquad \lambda \in \mathbb{R}}


计算多项式的特征方程，其中 A 是一个 n × n 的矩阵，\lambda 是特征值。det 表示行列式，\mathrm{I}_n 表示 n 阶单位矩阵。这个方程描述了特征值 \lambda 在实数域 \mathbb{R} 上的情况。
householder_product = _add_docstr(_linalg.linalg_householder_product, r"""
householder_product(A, tau, *, out=None) -> Tensor

Computes the first `n` columns of a product of Householder matrices.

Let :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`, and
let :math:`V \in \mathbb{K}^{m \times n}` be a matrix with columns :math:`v_i \in \mathbb{K}^m`
for :math:`i=1,\ldots,m` with :math:`m \geq n`. Denote by :math:`w_i` the vector resulting from
zeroing out the first :math:`i-1` components of :math:`v_i` and setting to `1` the :math:`i`-th.
For a vector :math:`\tau \in \mathbb{K}^k` with :math:`k \leq n`, this function computes the
first :math:`n` columns of the matrix

.. math::

    H_1H_2 ... H_k \qquad\text{with}\qquad H_i = \mathrm{I}_m - \tau_i w_i w_i^{\text{H}}

where :math:`\mathrm{I}_m` is the `m`-dimensional identity matrix and :math:`w^{\text{H}}` is the
conjugate transpose when :math:`w` is complex, and the transpose when :math:`w` is real-valued.

Args:
    A (Tensor): input tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    tau (Tensor): tensor of Householder coefficients of shape `(*, k)`.
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Returns:
    Tensor: tensor containing the first `n` columns of the Householder product.

Note:
    This function does not perform checks on the shapes of `A` or `tau`. It assumes they are
    correctly shaped to perform the operation as described.
""")
# 使用与输入矩阵 :attr:`A` 相同大小的输出矩阵。

# 查看“正交或酉矩阵的表示”详细信息。

# 支持 float、double、cfloat 和 cdouble 数据类型的输入。
# 还支持矩阵批处理，如果输入是矩阵批处理，则输出具有相同的批处理维度。

# 参见:

#     :func:`torch.geqrf` 可与此函数一起使用以形成从 `qr` 分解中获得的 `Q` 矩阵。

#     :func:`torch.ormqr` 是一个相关函数，计算一个由 Householder 矩阵乘以另一个矩阵的乘积。
#     然而，该函数不支持自动求导。

# 警告:
#     只有当 :math:`tau_i \neq \frac{1}{||v_i||^2}` 时，梯度计算才是良定义的。
#     如果不满足此条件，不会抛出错误，但产生的梯度可能包含 `NaN`。

# 参数:
#     A (Tensor): 形状为 `(*, m, n)` 的张量，其中 `*` 是零个或多个批处理维度。
#     tau (Tensor): 形状为 `(*, k)` 的张量，其中 `*` 是零个或多个批处理维度。

# 关键字参数:
#     out (Tensor, optional): 输出张量。如果为 `None`，则忽略。默认值为 `None`。

# 异常:
#     RuntimeError: 如果 :attr:`A` 不满足要求 `m >= n`，
#                   或 :attr:`tau` 不满足要求 `n >= k`。
"""
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

""" + fr"""
.. note:: {common_notes["sync_note_has_ex"].format("torch.linalg.ldl_factor_ex")}
""" + r"""

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of symmetric or Hermitian matrices.

Keyword args:
    hermitian (bool, optional): whether to consider the input to be Hermitian or symmetric.
                                For real-valued matrices, this switch has no effect. Default: `False`.
    out (tuple, optional): tuple of two tensors to write the output to. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(LD, pivots)`.

Examples::

    >>> A = torch.randn(3, 3)
    >>> A = A @ A.mT # make symmetric
    >>> A
    tensor([[7.2079, 4.2414, 1.9428],
            [4.2414, 3.4554, 0.3264],
            [1.9428, 0.3264, 1.3823]])
    >>> LD, pivots = torch.linalg.ldl_factor(A)
    >>> LD
    tensor([[ 7.2079,  0.0000,  0.0000],
            [ 0.5884,  0.9595,  0.0000],
            [ 0.2695, -0.8513,  0.1633]])
    >>> pivots
    tensor([1, 2, 3], dtype=torch.int32)

.. _LAPACK's sytrf:
    https://www.netlib.org/lapack/explore-html/d3/db6/group__double_s_ycomputational_gad91bde1212277b3e909eb6af7f64858a.html
"""

ldl_factor_ex = _add_docstr(_linalg.linalg_ldl_factor_ex, r"""
linalg.ldl_factor_ex(A, *, hermitian=False, check_errors=False, out=None) -> (Tensor, Tensor, Tensor)

This is a version of :func:`~ldl_factor` that does not perform error checks unless :attr:`check_errors`\ `= True`.
It also returns the :attr:`info` tensor returned by `LAPACK's sytrf`_.
``info`` stores integer error codes from the backend library.
A positive integer indicates the diagonal element of :math:`D` that is zero.
Division by 0 will occur if the result is used for solving a system of linear equations.
``info`` filled with zeros indicates that the factorization was successful.
If ``check_errors=True`` and ``info`` contains positive integers, then a `RuntimeError` is thrown.

""" + fr"""
.. note:: {common_notes["sync_note_ex"]}

.. warning:: {common_notes["experimental_warning"]}
""" + r"""

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions
                consisting of symmetric or Hermitian matrices.

Keyword args:
    hermitian (bool, optional): whether to consider the input to be Hermitian or symmetric.
                                For real-valued matrices, this switch has no effect. Default: `False`.
    check_errors (bool, optional): controls whether to check the content of ``info`` and raise
                                   an error if it is non-zero. Default: `False`.
    out (tuple, optional): tuple of three tensors to write the output to. Ignored if `None`. Default: `None`.

Returns:
    A named tuple `(LD, pivots, info)`.

Examples::

    >>> A = torch.randn(3, 3)
"""
    # 计算矩阵 A 的转置与自身的乘积，并使其变为对称矩阵
    >>> A = A @ A.mT # make symmetric
    
    # 打印输出矩阵 A 的当前值
    >>> A
    tensor([[7.2079, 4.2414, 1.9428],
            [4.2414, 3.4554, 0.3264],
            [1.9428, 0.3264, 1.3823]])
    
    # 使用 torch.linalg.ldl_factor_ex 函数对矩阵 A 进行 LDL 分解
    >>> LD, pivots, info = torch.linalg.ldl_factor_ex(A)
    
    # 打印输出 LDL 分解得到的下三角矩阵 LD
    >>> LD
    tensor([[ 7.2079,  0.0000,  0.0000],
            [ 0.5884,  0.9595,  0.0000],
            [ 0.2695, -0.8513,  0.1633]])
    
    # 打印输出 LDL 分解得到的置换向量 pivots
    >>> pivots
    tensor([1, 2, 3], dtype=torch.int32)
    
    # 打印输出 LDL 分解的信息指示变量 info
    >>> info
    tensor(0, dtype=torch.int32)
# 定义一个 Sphinx 链接到 LAPACK's sytrf 文档的引用
.. _LAPACK's sytrf:
    https://www.netlib.org/lapack/explore-html/d3/db6/group__double_s_ycomputational_gad91bde1212277b3e909eb6af7f64858a.html
"""

# 使用 _add_docstr 函数为 linalg_ldl_solve 函数添加文档字符串
ldl_solve = _add_docstr(_linalg.linalg_ldl_solve, r"""
linalg.ldl_solve(LD, pivots, B, *, hermitian=False, out=None) -> Tensor

Computes the solution of a system of linear equations using the LDL factorization.

:attr:`LD` and :attr:`pivots` are the compact representation of the LDL factorization and
are expected to be computed by :func:`torch.linalg.ldl_factor_ex`.
:attr:`hermitian` argument to this function should be the same
as the corresponding arguments in :func:`torch.linalg.ldl_factor_ex`.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

""" + fr"""
.. warning:: {common_notes["experimental_warning"]}
""" + r"""

Args:
    LD (Tensor): the `n \times n` matrix or the batch of such matrices of size
                      `(*, n, n)` where `*` is one or more batch dimensions.
    pivots (Tensor): the pivots corresponding to the LDL factorization of :attr:`LD`.
    B (Tensor): right-hand side tensor of shape `(*, n, k)`.

Keyword args:
    hermitian (bool, optional): whether to consider the decomposed matrix to be Hermitian or symmetric.
                                For real-valued matrices, this switch has no effect. Default: `False`.
    out (tuple, optional): output tensor. `B` may be passed as `out` and the result is computed in-place on `B`.
                           Ignored if `None`. Default: `None`.

Examples::

    >>> A = torch.randn(2, 3, 3)
    >>> A = A @ A.mT # make symmetric
    >>> LD, pivots, info = torch.linalg.ldl_factor_ex(A)
    >>> B = torch.randn(2, 3, 4)
    >>> X = torch.linalg.ldl_solve(LD, pivots, B)
    >>> torch.linalg.norm(A @ X - B)
    >>> tensor(0.0001)
""")

# 使用 _add_docstr 函数为 linalg_lstsq 函数添加文档字符串
lstsq = _add_docstr(_linalg.linalg_lstsq, r"""
torch.linalg.lstsq(A, B, rcond=None, *, driver=None) -> (Tensor, Tensor, Tensor, Tensor)

Computes a solution to the least squares problem of a system of linear equations.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **least squares problem** for a linear system :math:`AX = B` with
:math:`A \in \mathbb{K}^{m \times n}, B \in \mathbb{K}^{m \times k}` is defined as

.. math::

    \min_{X \in \mathbb{K}^{n \times k}} \|AX - B\|_F

where :math:`\|-\|_F` denotes the Frobenius norm.

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if the inputs are batches of matrices then
the output has the same batch dimensions.

:attr:`driver` chooses the backend function that will be used.
For CPU inputs the valid values are `'gels'`, `'gelsy'`, `'gelsd`, `'gelss'`.
To choose the best driver on CPU consider:
"""
- If :attr:`A` is well-conditioned (its `condition number`_ is not too large), or you do not mind some precision loss.

  - For a general matrix: `'gelsy'` (QR with pivoting) (default)
    如果 :attr:`A` 矩阵条件良好（其 `条件数`_ 不是很大），或者您不介意一些精度损失。
    对于一般矩阵，使用 `'gelsy'` 算法（QR 分解并进行枢轴选择）（默认）。

  - If :attr:`A` is full-rank: `'gels'` (QR)
    如果 :attr:`A` 是满秩的，则使用 `'gels'` 算法（QR 分解）。

- If :attr:`A` is not well-conditioned.
  如果 :attr:`A` 的条件不佳。

  - `'gelsd'` (tridiagonal reduction and SVD)
    使用 `'gelsd'` 算法（三对角矩阵约简和奇异值分解）。
  
  - But if you run into memory issues: `'gelss'` (full SVD).
    但如果遇到内存问题，则使用 `'gelss'` 算法（完全奇异值分解）。

For CUDA input, the only valid driver is `'gels'`, which assumes that :attr:`A` is full-rank.
对于 CUDA 输入，唯一有效的驱动程序是 `'gels'`，假定 :attr:`A` 是满秩的。

See also the `full description of these drivers`_
详细了解这些驱动程序，请参阅`这些驱动程序的完整描述`_。

:attr:`rcond` is used to determine the effective rank of the matrices in :attr:`A`
when :attr:`driver` is one of (`'gelsy'`, `'gelsd'`, `'gelss'`).
In this case, if :math:`\sigma_i` are the singular values of `A` in decreasing order,
:math:`\sigma_i` will be rounded down to zero if :math:`\sigma_i \leq \text{rcond} \cdot \sigma_1`.
If :attr:`rcond`\ `= None` (default), :attr:`rcond` is set to the machine precision of the dtype of :attr:`A` times `max(m, n)`.

:attr:`rcond` 用于确定 :attr:`A` 矩阵的有效秩，当 :attr:`driver` 是 (`'gelsy'`, `'gelsd'`, `'gelss'`) 中的一个时。
在这种情况下，如果 :math:`\sigma_i` 是按降序排列的 `A` 的奇异值，则如果 :math:`\sigma_i \leq \text{rcond} \cdot \sigma_1`，
则 :math:`\sigma_i` 将舍入为零。如果 :attr:`rcond`\ `= None`（默认值），则 :attr:`rcond` 设置为 :attr:`A` 的 dtype 的机器精度乘以 `max(m, n)`。

This function returns the solution to the problem and some extra information in a named tuple of
four tensors `(solution, residuals, rank, singular_values)`. For inputs :attr:`A`, :attr:`B`
of shape `(*, m, n)`, `(*, m, k)` respectively, it contains

- `solution`: the least squares solution. It has shape `(*, n, k)`.
  解决问题的最小二乘解。其形状为 `(*, n, k)`。

- `residuals`: the squared residuals of the solutions, that is, :math:`\|AX - B\|_F^2`.
  It has shape equal to the batch dimensions of :attr:`A`.
  它是解的残差平方和，即 :math:`\|AX - B\|_F^2`。其形状与 :attr:`A` 的批量维度相同。
  当 `m > n` 且每个矩阵在 :attr:`A` 中都是满秩时才计算，否则返回空张量。
  如果 :attr:`A` 是一批矩阵并且任何一个矩阵不是满秩，则返回一个空张量。这种行为可能在未来的 PyTorch 发布中更改。

- `rank`: tensor of ranks of the matrices in :attr:`A`.
  It has shape equal to the batch dimensions of :attr:`A`.
  当 :attr:`driver` 是 (`'gelsy'`, `'gelsd'`, `'gelss'`) 中的一个时计算，否则返回一个空张量。

- `singular_values`: tensor of singular values of the matrices in :attr:`A`.
  It has shape `(*, min(m, n))`.
  当 :attr:`driver` 是 (`'gelsd'`, `'gelss'`) 中的一个时计算，否则返回一个空张量。

.. note::
    This function computes `X = \ `:attr:`A`\ `.pinverse() @ \ `:attr:`B` in a faster and
    more numerically stable way than performing the computations separately.
    此函数计算 `X = \ `:attr:`A`\ `.pinverse() @ \ `:attr:`B`，比单独执行计算更快且数值更稳定。

.. warning::
    The default value of :attr:`rcond` may change in a future PyTorch release.
    It is therefore recommended to use a fixed value to avoid potential
    breaking changes.
    :attr:`rcond` 的默认值可能会在未来的 PyTorch 发布中更改。
    因此建议使用固定值以避免潜在的兼容性问题。

Args:
    A (Tensor): lhs tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
               形状为 `(*, m, n)` 的左手边张量，`*` 表示零个或多个批量维度。
    B (Tensor): rhs tensor of shape `(*, m, k)` where `*` is zero or more batch dimensions.
               形状为 `(*, m, k)` 的右手边张量，`*` 表示零个或多个批量维度。
"""
    rcond (float, optional): used to determine the effective rank of :attr:`A`.
                             # rcond参数，用于确定属性A的有效秩。
                             If :attr:`rcond`\ `= None`, :attr:`rcond` is set to the machine
                             # 如果rcond参数为None，则将rcond设置为属性A的数据类型的机器精度乘以max(m, n)。
                             precision of the dtype of :attr:`A` times `max(m, n)`. Default: `None`.
                             # 精度为属性A的数据类型的机器精度乘以max(m, n)。默认值为None。
# 定义函数签名和参数文档
Keyword args:
    driver (str, optional): name of the LAPACK/MAGMA method to be used.
        If `None`, `'gelsy'` is used for CPU inputs and `'gels'` for CUDA inputs.
        Default: `None`.
        
Returns:
    A named tuple `(solution, residuals, rank, singular_values)`.

Examples::

    >>> A = torch.randn(1,3,3)
    >>> A
    tensor([[[-1.0838,  0.0225,  0.2275],
         [ 0.2438,  0.3844,  0.5499],
         [ 0.1175, -0.9102,  2.0870]]])
    >>> B = torch.randn(2,3,3)
    >>> B
    tensor([[[-0.6772,  0.7758,  0.5109],
         [-1.4382,  1.3769,  1.1818],
         [-0.3450,  0.0806,  0.3967]],
        [[-1.3994, -0.1521, -0.1473],
         [ 1.9194,  1.0458,  0.6705],
         [-1.1802, -0.9796,  1.4086]]])
    >>> X = torch.linalg.lstsq(A, B).solution # A is broadcasted to shape (2, 3, 3)
    >>> torch.dist(X, torch.linalg.pinv(A) @ B)
    tensor(1.5152e-06)

    >>> S = torch.linalg.lstsq(A, B, driver='gelsd').singular_values
    >>> torch.dist(S, torch.linalg.svdvals(A))
    tensor(2.3842e-07)

    >>> A[:, 0].zero_()  # Decrease the rank of A
    >>> rank = torch.linalg.lstsq(A, B).rank
    >>> rank
    tensor([2])

.. _condition number:
    https://pytorch.org/docs/main/linalg.html#torch.linalg.cond
.. _full description of these drivers:
    https://www.netlib.org/lapack/lug/node27.html
"""

# 定义函数 linalg_matrix_power 的文档字符串
matrix_power = _add_docstr(_linalg.linalg_matrix_power, r"""
matrix_power(A, n, *, out=None) -> Tensor

Computes the `n`-th power of a square matrix for an integer `n`.

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

If :attr:`n`\ `= 0`, it returns the identity matrix (or batch) of the same shape
as :attr:`A`. If :attr:`n` is negative, it returns the inverse of each matrix
(if invertible) raised to the power of `abs(n)`.

.. note::
    Consider using :func:`torch.linalg.solve` if possible for multiplying a matrix on the left by
    a negative power as, if :attr:`n`\ `> 0`::

        torch.linalg.solve(matrix_power(A, n), B) == matrix_power(A, -n)  @ B

    It is always preferred to use :func:`~solve` when possible, as it is faster and more
    numerically stable than computing :math:`A^{-n}` explicitly.

.. seealso::

        :func:`torch.linalg.solve` computes :attr:`A`\ `.inverse() @ \ `:attr:`B` with a
        numerically stable algorithm.

Args:
    A (Tensor): tensor of shape `(*, m, m)` where `*` is zero or more batch dimensions.
    n (int): the exponent.

Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if :attr:`n`\ `< 0` and the matrix :attr:`A` or any matrix in the
                  batch of matrices :attr:`A` is not invertible.

Examples::

    >>> A = torch.randn(3, 3)
    >>> torch.linalg.matrix_power(A, 0)
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    >>> torch.linalg.matrix_power(A, 3)

""")
    # 定义一个三行三列的张量，表示一个3x3的矩阵
    tensor([[ 1.0756,  0.4980,  0.0100],
            [-1.6617,  1.4994, -1.9980],
            [-0.4509,  0.2731,  0.8001]])
    >>> torch.linalg.matrix_power(A.expand(2, -1, -1), -2)
    # 对张量 A 进行扩展，使其形状变为两个3x3的矩阵，然后计算其逆矩阵的平方
    tensor([[[ 0.2640,  0.4571, -0.5511],
            [-1.0163,  0.3491, -1.5292],
            [-0.4899,  0.0822,  0.2773]],
            [[ 0.2640,  0.4571, -0.5511],
            [-1.0163,  0.3491, -1.5292],
            [-0.4899,  0.0822,  0.2773]]])
# 导入需要的函数 _add_docstr 和 _linalg.linalg_matrix_rank
matrix_rank = _add_docstr(_linalg.linalg_matrix_rank, r"""
linalg.matrix_rank(A, *, atol=None, rtol=None, hermitian=False, out=None) -> Tensor

Computes the numerical rank of a matrix.

The matrix rank is computed as the number of singular values
(or eigenvalues in absolute value when :attr:`hermitian`\ `= True`)
that are greater than :math:`\max(\text{atol}, \sigma_1 * \text{rtol})` threshold,
where :math:`\sigma_1` is the largest singular value (or eigenvalue).

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

If :attr:`hermitian`\ `= True`, :attr:`A` is assumed to be Hermitian if complex or
symmetric if real, but this is not checked internally. Instead, just the lower
triangular part of the matrix is used in the computations.

If :attr:`rtol` is not specified and :attr:`A` is a matrix of dimensions `(m, n)`,
the relative tolerance is set to be :math:`\max(m, n) \varepsilon`
and :math:`\varepsilon` is the epsilon value for the dtype of :attr:`A` (see :class:`.finfo`).
If :attr:`rtol` is not specified and :attr:`atol` is specified to be larger than zero then
:attr:`rtol` is set to zero.

If :attr:`atol` or :attr:`rtol` is a :class:`torch.Tensor`, its shape must be broadcastable to that
of the singular values of :attr:`A` as returned by :func:`torch.linalg.svdvals`.

.. note::
    This function has NumPy compatible variant `linalg.matrix_rank(A, tol, hermitian=False)`.
    However, use of the positional argument :attr:`tol` is deprecated in favor of :attr:`atol` and :attr:`rtol`.

.. note:: The matrix rank is computed using a singular value decomposition
          :func:`torch.linalg.svdvals` if :attr:`hermitian`\ `= False` (default) and the eigenvalue
          decomposition :func:`torch.linalg.eigvalsh` when :attr:`hermitian`\ `= True`.
          {common_notes["sync_note"]}

Args:
    A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.
    tol (float, Tensor, optional): [NumPy Compat] Alias for :attr:`atol`. Default: `None`.

Keyword args:
    atol (float, Tensor, optional): the absolute tolerance value. When `None` it's considered to be zero.
                                    Default: `None`.
    rtol (float, Tensor, optional): the relative tolerance value. See above for the value it takes when `None`.
                                    Default: `None`.
    hermitian(bool): indicates whether :attr:`A` is Hermitian if complex
                     or symmetric if real. Default: `False`.
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Examples::

    >>> A = torch.eye(10)
    >>> torch.linalg.matrix_rank(A)
    tensor(10)
    >>> B = torch.eye(10)
    >>> B[0, 0] = 0
    >>> torch.linalg.matrix_rank(B)
    tensor(9)

    >>> A = torch.randn(4, 3, 2)
    >>> torch.linalg.matrix_rank(A)
"""
    # 创建一个包含四个整数 2 的张量
    tensor([2, 2, 2, 2])
    
    # 创建一个形状为 (2, 4, 2, 3) 的随机张量 A，并计算其矩阵秩
    >>> A = torch.randn(2, 4, 2, 3)
    >>> torch.linalg.matrix_rank(A)
    返回一个张量，其中每个元素都是 A 对应位置上的矩阵秩
    
    # 创建一个形状为 (2, 4, 3, 3)、数据类型为复数的随机张量 A，并计算其矩阵秩
    >>> A = torch.randn(2, 4, 3, 3, dtype=torch.complex64)
    >>> torch.linalg.matrix_rank(A)
    返回一个张量，其中每个元素都是 A 对应位置上的矩阵秩
    
    # 继续使用 A 计算矩阵秩，但假定 A 是厄米特矩阵（复共轭转置等于自身）
    >>> torch.linalg.matrix_rank(A, hermitian=True)
    返回一个张量，其中每个元素都是 A 对应位置上的厄米特矩阵的矩阵秩
    
    # 使用自定义的容差值（atol=1.0, rtol=0.0）计算 A 的矩阵秩
    >>> torch.linalg.matrix_rank(A, atol=1.0, rtol=0.0)
    返回一个张量，其中每个元素是根据给定容差值计算出的 A 的矩阵秩
    
    # 继续使用自定义的容差值计算 A 的厄米特矩阵的矩阵秩
    >>> torch.linalg.matrix_rank(A, atol=1.0, rtol=0.0, hermitian=True)
    返回一个张量，其中每个元素是根据给定容差值计算出的 A 的厄米特矩阵的矩阵秩
# 导入标准库中的 _add_docstr 和 _linalg.linalg_norm 函数
norm = _add_docstr(_linalg.linalg_norm, r"""
linalg.norm(A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None) -> Tensor

Computes a vector or matrix norm.

Supports input of float, double, cfloat and cdouble dtypes.

Whether this function computes a vector or matrix norm is determined as follows:

- If :attr:`dim` is an `int`, the vector norm will be computed.
- If :attr:`dim` is a `2`-`tuple`, the matrix norm will be computed.
- If :attr:`dim`\ `= None` and :attr:`ord`\ `= None`,
  :attr:`A` will be flattened to 1D and the `2`-norm of the resulting vector will be computed.
- If :attr:`dim`\ `= None` and :attr:`ord` `!= None`, :attr:`A` must be 1D or 2D.

:attr:`ord` defines the norm that is computed. The following norms are supported:

======================     =========================  ========================================================
:attr:`ord`                norm for matrices          norm for vectors
======================     =========================  ========================================================
`None` (default)           Frobenius norm             `2`-norm (see below)
`'fro'`                    Frobenius norm             -- not supported --
`'nuc'`                    nuclear norm               -- not supported --
`inf`                      `max(sum(abs(x), dim=1))`  `max(abs(x))`
`-inf`                     `min(sum(abs(x), dim=1))`  `min(abs(x))`
`0`                        -- not supported --        `sum(x != 0)`
`1`                        `max(sum(abs(x), dim=0))`  as below
`-1`                       `min(sum(abs(x), dim=0))`  as below
`2`                        largest singular value     as below
`-2`                       smallest singular value    as below
other `int` or `float`     -- not supported --        `sum(abs(x)^{ord})^{(1 / ord)}`
======================     =========================  ========================================================

where `inf` refers to `float('inf')`, NumPy's `inf` object, or any equivalent object.

.. seealso::

        :func:`torch.linalg.vector_norm` computes a vector norm.

        :func:`torch.linalg.matrix_norm` computes a matrix norm.

        The above functions are often clearer and more flexible than using :func:`torch.linalg.norm`.
        For example, `torch.linalg.norm(A, ord=1, dim=(0, 1))` always
        computes a matrix norm, but with `torch.linalg.vector_norm(A, ord=1, dim=(0, 1))` it is possible
        to compute a vector norm over the two dimensions.

Args:
    A (Tensor): tensor of shape `(*, n)` or `(*, m, n)` where `*` is zero or more batch dimensions
    ord (int, float, inf, -inf, 'fro', 'nuc', optional): order of norm. Default: `None`
    dim (int, Tuple[int], optional): dimensions over which to compute
        the vector or matrix norm. See above for the behavior when :attr:`dim`\ `= None`.
        Default: `None`
""")
    keepdim (bool, optional): 如果设置为 `True`，则在结果中保留被减少的维度作为尺寸为一的维度。默认值为 `False`

# `2`            `2`-norm (see below)
# `inf`          `max(abs(x))`
# `-inf`         `min(abs(x))`
# `0`            `sum(x != 0)`
# other `int` or `float`   `sum(abs(x)^{ord})^{(1 / ord)}`
# ======================   ===============================

# where `inf` refers to `float('inf')`, NumPy's `inf` object, or any equivalent object.

# :attr:`dtype` may be used to perform the computation in a more precise dtype.
# It is semantically equivalent to calling ``linalg.vector_norm(x.to(dtype))``
# but it is faster in some cases.

# .. seealso::
# 
#    :func:`torch.linalg.matrix_norm` computes a matrix norm.
# 
# Args:
#     x (Tensor): tensor, flattened by default, but this behavior can be
#         controlled using :attr:`dim`.
#     ord (int, float, inf, -inf, 'fro', 'nuc', optional): order of norm. Default: `2`
#     dim (int, Tuple[int], optional): dimensions over which to compute
#         the norm. See above for the behavior when :attr:`dim`\ `= None`.
#         Default: `None`
#     keepdim (bool, optional): If set to `True`, the reduced dimensions are retained
#         in the result as dimensions with size one. Default: `False`
# 
# Keyword args:
#     out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.
#     dtype (:class:`torch.dtype`, optional): type used to perform the accumulation and the return.
#         If specified, :attr:`x` is cast to :attr:`dtype` before performing the operation,
#         and the returned tensor's type will be :attr:`dtype` if real and of its real counterpart if complex.
#         :attr:`dtype` may be complex if :attr:`x` is complex, otherwise it must be real.
#         :attr:`x` should be convertible without narrowing to :attr:`dtype`. Default: None
# 
# Returns:
#     A real-valued tensor, even when :attr:`x` is complex.
# 
# Examples::
# 
#     >>> from torch import linalg as LA
#     >>> a = torch.arange(9, dtype=torch.float) - 4
#     >>> a
#     tensor([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
#     >>> B = a.reshape((3, 3))
#     >>> B
#     tensor([[-4., -3., -2.],
#             [-1.,  0.,  1.],
#             [ 2.,  3.,  4.]])
#     >>> LA.vector_norm(a, ord=3.5)
#     tensor(5.4345)
#     >>> LA.vector_norm(B, ord=3.5)
#     tensor(5.4345)



# matrix_norm = _add_docstr(_linalg.linalg_matrix_norm, r"""
# linalg.matrix_norm(A, ord='fro', dim=(-2, -1), keepdim=False, *, dtype=None, out=None) -> Tensor
# 
# Computes a matrix norm.
# 
# If :attr:`A` is complex valued, it computes the norm of :attr:`A`\ `.abs()`
# 
# Support input of float, double, cfloat and cdouble dtypes.
# Also supports batches of matrices: the norm will be computed over the
# dimensions specified by the 2-tuple :attr:`dim` and the other dimensions will
# be treated as batch dimensions. The output will have the same batch dimensions.
# 
# :attr:`ord` defines the matrix norm that is computed. The following norms are supported:
# 
# ======================   ========================================================
# :attr:`ord`              matrix norm
# ```
matmul = _add_docstr(_linalg.linalg_matmul, r"""
linalg.matmul(input, other, *, out=None) -> Tensor

Alias for :func:`torch.matmul`
""")

diagonal = _add_docstr(_linalg.linalg_diagonal, r"""
linalg.diagonal(A, *, offset=0, dim1=-2, dim2=-1) -> Tensor

Alias for :func:`torch.diagonal` with defaults :attr:`dim1`\ `= -2`, :attr:`dim2`\ `= -1`.
""")

multi_dot = _add_docstr(_linalg.linalg_multi_dot, r"""
linalg.multi_dot(tensors, *, out=None)

Efficiently multiplies two or more matrices by reordering the multiplications so that
the fewest arithmetic operations are performed.

Supports inputs of float, double, cfloat and cdouble dtypes.
This function does not support batched inputs.

Every tensor in :attr:`tensors` must be 2D, except for the first and last which
# 定义了一个函数 `_add_docstr`，用于为函数添加文档字符串，文档字符串可以用于描述函数的作用和参数信息
svd = _add_docstr(_linalg.linalg_svd, r"""
# linalg.svd 函数签名，说明了函数接受的参数和返回的结果
linalg.svd(A, full_matrices=True, *, driver=None, out=None) -> (Tensor, Tensor, Tensor)

# 计算矩阵的奇异值分解（SVD）

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **full SVD** of a matrix
:math:`A \in \mathbb{K}^{m \times n}`, if `k = min(m,n)`, is defined as

# 对于实数或复数域 :math:`\mathbb{K}`，矩阵 :math:`A \in \mathbb{K}^{m \times n}` 的全 SVD 定义如下
.. math::

    A = U \operatorname{diag}(S) V^{\text{H}}
    \mathrlap{\qquad U \in \mathbb{K}^{m \times m}, S \in \mathbb{R}^k, V \in \mathbb{K}^{n \times n}}

# 这里，:math:`\operatorname{diag}(S)` 表示对角矩阵，其维度为 :math:`\mathbb{K}^{m \times n}`
# 当 :math:`V` 为复数时，:math:`V^{\text{H}}` 表示共轭转置；当 :math:`V` 为实数时，表示转置。

"""
# 矩阵 :math:`U`, :math:`V` (因此 :math:`V^{\text{H}}`) 在实数情况下是正交的，在复数情况下是酉的。
# 当 `m > n`（或 `m < n`）时，可以丢弃 `U`（或 `V`）的最后 `m - n`（或 `n - m`）列，形成**减少的奇异值分解**：

# .. math::

#     A = U \operatorname{diag}(S) V^{\text{H}}
#     \mathrlap{\qquad U \in \mathbb{K}^{m \times k}, S \in \mathbb{R}^k, V \in \mathbb{K}^{k \times n}}

# 其中 :math:`\operatorname{diag}(S) \in \mathbb{K}^{k \times k}`。
# 在这种情况下，:math:`U` 和 :math:`V` 也具有标准正交列。

# 支持 float、double、cfloat 和 cdouble 数据类型的输入。
# 也支持矩阵的批处理，如果 :attr:`A` 是矩阵的批处理，则输出具有相同的批处理维度。

# 返回的分解是一个命名元组 `(U, S, Vh)`，对应于上述的 :math:`U`, :math:`S`, :math:`V^{\text{H}}`。

# 奇异值按降序返回。

# 参数 :attr:`full_matrices` 选择完整（默认）或减少的奇异值分解。

# 在 CUDA 中，参数 :attr:`driver` 可以与 cuSOLVER 后端一起使用，选择计算 SVD 的算法。
# 选择驱动程序是精度和速度之间的权衡。

# - 如果 :attr:`A` 很好条件化（其 `条件数`_ 不太大），或者您不介意一些精度损失。

#   - 对于一般矩阵：`'gesvdj'`（雅可比方法）
#   - 如果 :attr:`A` 高或宽（`m >> n` 或 `m << n`）：`'gesvda'`（近似方法）

# - 如果 :attr:`A` 不是很好条件化或精度很重要：`'gesvd'`（基于 QR 分解）

# 默认情况下（:attr:`driver`\ `= None`），我们调用 `'gesvdj'`，如果失败，则回退到 `'gesvd'`。

# 与 `numpy.linalg.svd` 的区别：

# - 与 `numpy.linalg.svd` 不同，此函数始终返回三个张量的元组，并且不支持 `compute_uv` 参数。
# 请使用 :func:`torch.linalg.svdvals`，它只计算奇异值，而不是 `compute_uv=False`。

# .. note:: 当 :attr:`full_matrices`\ `= True` 时，对于 `U[..., :, min(m, n):]` 和 `Vh[..., min(m, n):, :]` 的梯度将被忽略，
# 因为这些向量可以是相应子空间的任意基向量。
# 警告：返回的张量 `U` 和 `V` 不是唯一的，也不与 :attr:`A` 相连续。
# 由于缺乏唯一性，不同的硬件和软件可能计算出不同的奇异向量。

# 这种不唯一性是由于任意乘以奇异向量对 `(u_k, v_k)` 中的任意一对乘以 `-1`（实数情况下）或乘以
# :math:`e^{i \phi}, \phi \in \mathbb{R}`（复数情况下）会产生矩阵的另外两个有效的奇异向量。
# 因此，损失函数不应依赖于这个 :math:`e^{i \phi}` 量，因为它没有明确定义。
# 当计算此函数的梯度时，检查复数输入时是否存在这种情况。因此，当输入是复数且在CUDA设备上时，
# 计算此函数的梯度会将该设备与CPU同步。

# 警告：使用 `U` 或 `Vh` 计算的梯度仅在 :attr:`A` 没有重复奇异值时才是有限的。
# 如果 :attr:`A` 是矩形的，则此外，零也不能是其奇异值之一。
# 此外，如果任意两个奇异值之间的距离接近零，则梯度将在数值上不稳定，
# 因为它依赖于奇异值 :math:`\sigma_i` 的计算，通过 :math:`\frac{1}{\min_{i \neq j} \sigma_i^2 - \sigma_j^2}` 进行。
# 在矩形情况下，如果 :attr:`A` 具有较小的奇异值，则梯度也会在数值上不稳定，
# 因为它还依赖于 :math:`\frac{1}{\sigma_i}` 的计算。

# 参见：

# :func:`torch.linalg.svdvals` 仅计算奇异值。
# 与 :func:`torch.linalg.svd` 不同，:func:`~svdvals` 的梯度始终是数值稳定的。

# :func:`torch.linalg.eig` 用于计算矩阵的另一种类型的谱分解函数。
# 特征分解仅适用于方阵。

# :func:`torch.linalg.eigh` 用于快速计算对称和厄米特矩阵的特征值分解的函数。

# :func:`torch.linalg.qr` 用于在一般矩阵上进行另一种（更快速）的分解。

def torch.svd(A, full_matrices=True, driver=None):
    pass
    out (tuple, optional): output tuple of three tensors. Ignored if `None`.
cond = _add_docstr(_linalg.linalg_cond, r"""
linalg.cond(A, p=None, *, out=None) -> Tensor

计算矩阵相对于矩阵范数的条件数。

设 :math:`\mathbb{K}` 是 :math:`\mathbb{R}` 或 :math:`\mathbb{C}`，
矩阵的 **条件数** :math:`\kappa` 定义为

""")
# 定义一个名为 `A` 的矩阵，其元素属于数域 `K`，大小为 `n x n`
:math:`A \in \mathbb{K}^{n \times n}` is defined as

# 计算矩阵 `A` 的条件数 `kappa(A)`，其定义为矩阵 `A` 的某种范数与其逆的某种范数的乘积
.. math::

    \kappa(A) = \|A\|_p\|A^{-1}\|_p

# 矩阵 `A` 的条件数 `kappa(A)` 衡量线性系统 `AX = B` 在数值上的稳定性，相对于一个矩阵范数而言。

# 支持 `float`、`double`、`cfloat` 和 `cdouble` 数据类型的输入。
Supports input of float, double, cfloat and cdouble dtypes.
# 还支持批量输入的矩阵，如果 `A` 是一个批量的矩阵，则输出的结果维度与 `A` 相同。
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

# 参数 `p` 定义要计算的矩阵范数。支持以下范数：
:attr:`p` defines the matrix norm that is computed. The following norms are supported:

=========    =================================
:attr:`p`    matrix norm
=========    =================================
`None`       `2`-norm (largest singular value)
`'fro'`      Frobenius norm
`'nuc'`      nuclear norm
`inf`        `max(sum(abs(x), dim=1))`
`-inf`       `min(sum(abs(x), dim=1))`
`1`          `max(sum(abs(x), dim=0))`
`-1`         `min(sum(abs(x), dim=0))`
`2`          largest singular value
`-2`         smallest singular value
=========    =================================

# 其中 `inf` 表示 `float('inf')`、NumPy 的 `inf` 对象或任何等价对象。

# 当 `p` 是 `('fro', 'nuc', inf, -inf, 1, -1)` 中的一个时，此函数使用 `torch.linalg.norm` 和 `torch.linalg.inv` 进行计算。
# 因此，在这种情况下，矩阵（或批量中的每个矩阵） `A` 必须是方阵且可逆的。
For :attr:`p` is one of `('fro', 'nuc', inf, -inf, 1, -1)`, this function uses
:func:`torch.linalg.norm` and :func:`torch.linalg.inv`.
As such, in this case, the matrix (or every matrix in the batch) :attr:`A` has to be square
and invertible.

# 当 `p` 是 `(2, -2)` 时，此函数可以根据奇异值 `sigma_1 \geq \ldots \geq \sigma_n` 计算条件数。
# 在这些情况下，使用 `torch.linalg.svdvals` 进行计算。对于这些范数，矩阵（或批量中的每个矩阵） `A` 可以具有任何形状。
For :attr:`p` in `(2, -2)`, this function can be computed in terms of the singular values
:math:`\sigma_1 \geq \ldots \geq \sigma_n`

.. math::

    \kappa_2(A) = \frac{\sigma_1}{\sigma_n}\qquad \kappa_{-2}(A) = \frac{\sigma_n}{\sigma_1}

# 当输入在 CUDA 设备上时，如果 `p` 是 `('fro', 'nuc', inf, -inf, 1, -1)` 中的一个，则此函数将同步该设备与 CPU。
.. note :: When inputs are on a CUDA device, this function synchronizes that device with the CPU
           if :attr:`p` is one of `('fro', 'nuc', inf, -inf, 1, -1)`.

# 参见：

# - :func:`torch.linalg.solve` 解决方程组的函数，其中矩阵是方阵。
# - :func:`torch.linalg.lstsq` 用于解决一般矩阵线性系统的函数。

# Args:
#     A (Tensor): 形状为 `(*, m, n)` 的张量，其中 `*` 表示零个或多个批量维度，
#                     当 `p` 在 `(2, -2)` 中时，以及形状为 `(*, n, n)` 的张量，其中每个矩阵
#                     对于 `p` 在 `('fro', 'nuc', inf, -inf, 1, -1)` 中。
Args:
    A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions
                    for :attr:`p` in `(2, -2)`, and of shape `(*, n, n)` where every matrix
                    is invertible for :attr:`p` in `('fro', 'nuc', inf, -inf, 1, -1)`.

#     p (int, inf, -inf, 'fro', 'nuc', optional):
#         要在计算中使用的矩阵范数类型（参见上文）。默认为 `None`
    p (int, inf, -inf, 'fro', 'nuc', optional):
        the type of the matrix norm to use in the computations (see above). Default: `None`

# Keyword args:
#     out (Tensor, optional): 输出张量。如果为 `None`，则忽略。默认为 `None`。
Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

# Returns:
#     返回一个实值张量，即使输入的 `A` 是复数类型。
Returns:
    A real-valued tensor, even when :attr:`A` is complex.

# Raises:
#     RuntimeError:
#         如果 `p` 是 `('fro', 'nuc', inf, -inf, 1, -1)` 中的一个，并且矩阵 `A` 或批量 `A` 中的任何矩阵不是方阵或可逆的。
Raises:
    RuntimeError:
        if :attr:`p` is one of `('fro', 'nuc', inf, -inf, 1, -1)`
        and the :attr:`A` matrix or any matrix in the batch :attr:`A` is not square
        or invertible.

# Examples::

#     >>> A = torch.randn(3, 4, 4, dtype=torch.complex64)
#     >>> torch.linalg.cond(A)
Examples::

    >>> A = torch.randn(3, 4, 4, dtype=torch.complex64)
    >>> torch.linalg.cond(A)
    >>> A = torch.tensor([[1., 0, -1], [0, 1, 0], [1, 0, 1]])
    >>> torch.linalg.cond(A)
    tensor([1.4142])
    
    计算矩阵 A 的条件数，默认使用 2-范数。
    
    
    >>> torch.linalg.cond(A, 'fro')
    tensor(3.1623)
    
    计算矩阵 A 的条件数，使用 Frobenius 范数。
    
    
    >>> torch.linalg.cond(A, 'nuc')
    tensor(9.2426)
    
    计算矩阵 A 的核范数条件数。
    
    
    >>> torch.linalg.cond(A, float('inf'))
    tensor(2.)
    
    计算矩阵 A 的无穷范数条件数。
    
    
    >>> torch.linalg.cond(A, float('-inf'))
    tensor(1.)
    
    计算矩阵 A 的负无穷范数条件数。
    
    
    >>> torch.linalg.cond(A, 1)
    tensor(2.)
    
    计算矩阵 A 的 1-范数条件数。
    
    
    >>> torch.linalg.cond(A, -1)
    tensor(1.)
    
    计算矩阵 A 的 -1-范数条件数。
    
    
    >>> torch.linalg.cond(A, 2)
    tensor([1.4142])
    
    计算矩阵 A 的 2-范数条件数。
    
    
    >>> torch.linalg.cond(A, -2)
    tensor([0.7071])
    
    计算矩阵 A 的 -2-范数条件数。
    
    
    >>> A = torch.randn(2, 3, 3)
    >>> torch.linalg.cond(A)
    tensor([[9.5917],
            [3.2538]])
    
    对一个形状为 (2, 3, 3) 的随机张量 A 计算条件数，返回每个张量的条件数。
    
    
    >>> A = torch.randn(2, 3, 3, dtype=torch.complex64)
    >>> torch.linalg.cond(A)
    tensor([[4.6245],
            [4.5671]])
    
    对一个形状为 (2, 3, 3) 的随机复数类型张量 A 计算条件数，返回每个张量的条件数。
""")
# 使用`_add_docstr`函数，为`linalg_pinv`函数添加文档字符串
pinv = _add_docstr(_linalg.linalg_pinv, r"""
# linalg.pinv函数的签名和作用说明
linalg.pinv(A, *, atol=None, rtol=None, hermitian=False, out=None) -> Tensor

# 计算矩阵的伪逆（Moore-Penrose逆）

# 伪逆可以通过代数定义，但通过奇异值分解（SVD）来理解更为方便

# 支持float，double，cfloat和cdouble数据类型的输入
# 同时支持矩阵的批处理，如果A是矩阵的批处理，则输出具有相同的批处理维度

# 如果hermitian=True，则假定A是复数时的Hermitian矩阵或实数时的对称矩阵，
# 但这在内部不进行检查。相反，计算中只使用矩阵的下三角部分。

# 那些低于max(atol, σ₁ * rtol)阈值的奇异值（或当hermitian=True时的特征值范数）
# 被视为零并在计算中被丢弃，其中σ₁是最大的奇异值（或特征值）。

# 如果rtol未指定且A是维度为（m，n）的矩阵，则相对容差被设置为rtol = max(m, n)ε，
# 其中ε是A的dtype的ε值（参见.finfo）。
# 如果未指定rtol且atol被指定为大于零，则rtol被设置为零。

# 如果atol或rtol是torch.Tensor，则其形状必须可以广播到由torch.linalg.svd返回的A的奇异值的形状。

# 注意：如果hermitian=False，则此函数使用torch.linalg.svd，如果hermitian=True，则使用torch.linalg.eigh。
# 对于CUDA输入，此函数将该设备与CPU同步。

# 注意：如果可能，考虑使用torch.linalg.lstsq来将矩阵左乘伪逆，如：
# torch.linalg.lstsq(A, B).solution == A.pinv() @ B
# 总是优先使用lstsq，因为它比显式计算伪逆更快且数值上更稳定。

# 注意：此函数有一个与NumPy兼容的变体linalg.pinv(A, rcond, hermitian=False)。
# 然而，使用位置参数rcond已弃用，推荐使用rtol。

# 警告：此函数在内部使用torch.linalg.svd（或当hermitian=True时使用torch.linalg.eigh），
# 因此其导数具有与这些函数相同的问题。有关详细信息，请参阅torch.linalg.svd和torch.linalg.eigh中的警告。

# 参见：
#   torch.linalg.inv计算方阵的逆。
#   torch.linalg.lstsq使用稳定的算法计算A.pinv() @ B。

Args:
    A (Tensor): 形状为（*，m，n）的张量，其中*是零个或多个批处理维度。
""")
    rcond (float, Tensor, optional): [NumPy Compat]. Alias for :attr:`rtol`. Default: `None`.
Keyword args:
    atol (float, Tensor, optional): 绝对容差值。当为 `None` 时，被视为零。
                                    默认值：`None`。
    rtol (float, Tensor, optional): 相对容差值。参见上文中对 `None` 值的处理。
                                    默认值：`None`。
    hermitian(bool, optional): 如果为复数，则指示 :attr:`A` 是否为共轭转置（Hermitian）矩阵，如果为实数则指示是否为对称矩阵。默认值：`False`。
    out (Tensor, optional): 输出张量。如果为 `None` 则被忽略。默认值：`None`。

Examples::

    >>> A = torch.randn(3, 5)
    >>> A
    tensor([[ 0.5495,  0.0979, -1.4092, -0.1128,  0.4132],
            [-1.1143, -0.3662,  0.3042,  1.6374, -0.9294],
            [-0.3269, -0.5745, -0.0382, -0.5922, -0.6759]])
    >>> torch.linalg.pinv(A)
    tensor([[ 0.0600, -0.1933, -0.2090],
            [-0.0903, -0.0817, -0.4752],
            [-0.7124, -0.1631, -0.2272],
            [ 0.1356,  0.3933, -0.5023],
            [-0.0308, -0.1725, -0.5216]])

    >>> A = torch.randn(2, 6, 3)
    >>> Apinv = torch.linalg.pinv(A)
    >>> torch.dist(Apinv @ A, torch.eye(3))
    tensor(8.5633e-07)

    >>> A = torch.randn(3, 3, dtype=torch.complex64)
    >>> A = A + A.T.conj()  # 创建一个共轭转置矩阵（Hermitian matrix）
    >>> Apinv = torch.linalg.pinv(A, hermitian=True)
    >>> torch.dist(Apinv @ A, torch.eye(3))
    tensor(1.0830e-06)

.. _defined algebraically:
    https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Existence_and_uniqueness
.. _through the SVD:
    https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Singular_value_decomposition_(SVD)
""")

matrix_exp = _add_docstr(_linalg.linalg_matrix_exp, r"""
linalg.matrix_exp(A) -> Tensor

计算方阵的指数函数。

设 :math:`\mathbb{K}` 为 :math:`\mathbb{R}` 或 :math:`\mathbb{C}`，
此函数计算方阵 :math:`A \in \mathbb{K}^{n \times n}` 的 **矩阵指数函数**，定义如下：

.. math::
    \mathrm{matrix\_exp}(A) = \sum_{k=0}^\infty \frac{1}{k!}A^k \in \mathbb{K}^{n \times n}.

如果矩阵 :math:`A` 的特征值为 :math:`\lambda_i \in \mathbb{C}`，
则矩阵 :math:`\mathrm{matrix\_exp}(A)` 的特征值为 :math:`e^{\lambda_i} \in \mathbb{C}`。

支持输入的数据类型为 bfloat16, float, double, cfloat 和 cdouble。
也支持批量处理的矩阵，如果 :attr:`A` 是批量的矩阵，则输出也具有相同的批量维度。

Args:
    A (Tensor): 形状为 `(*, n, n)` 的张量，其中 `*` 是零个或多个批量维度。

Example::

    >>> A = torch.empty(2, 2, 2)
    >>> A[0, :, :] = torch.eye(2, 2)
    >>> A[1, :, :] = 2 * torch.eye(2, 2)
    >>> A
    tensor([[[1., 0.],
             [0., 1.]],

            [[2., 0.],
             [0., 2.]]])
    >>> torch.linalg.matrix_exp(A)
    tensor([[[2.7183, 0.0000],
             [0.0000, 2.7183]],

             [[7.3891, 0.0000],
              [0.0000, 7.3891]]])

    >>> import math
    # 创建一个张量 A，它是一个反对称矩阵，即A.transpose() == -A
    A = torch.tensor([[0, math.pi/3], [-math.pi/3, 0]])
    
    # 计算矩阵指数函数 exp(A)，返回结果是一个张量
    # 对于反对称矩阵 A，exp(A) 的结果是一个旋转矩阵
    # 具体来说，exp(A) = [[cos(pi/3), sin(pi/3)], [-sin(pi/3), cos(pi/3)]]
    result = torch.linalg.matrix_exp(A)
    
    # 打印计算结果
    print(result)
# 定义 linalg.solve 函数的文档字符串，描述其功能和用法
solve = _add_docstr(_linalg.linalg_solve, r"""
linalg.solve(A, B, *, left=True, out=None) -> Tensor

Computes the solution of a square system of linear equations with a unique solution.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
this function computes the solution :math:`X \in \mathbb{K}^{n \times k}` of the **linear system** associated to
:math:`A \in \mathbb{K}^{n \times n}, B \in \mathbb{K}^{n \times k}`, which is defined as

.. math:: AX = B

If :attr:`left`\ `= False`, this function returns the matrix :math:`X \in \mathbb{K}^{n \times k}` that solves the system

.. math::

    XA = B\mathrlap{\qquad A \in \mathbb{K}^{k \times k}, B \in \mathbb{K}^{n \times k}.}

This system of linear equations has one solution if and only if :math:`A` is `invertible`_.
This function assumes that :math:`A` is invertible.

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if the inputs are batches of matrices then
the output has the same batch dimensions.

Letting `*` be zero or more batch dimensions,

- If :attr:`A` has shape `(*, n, n)` and :attr:`B` has shape `(*, n)` (a batch of vectors) or shape
  `(*, n, k)` (a batch of matrices or "multiple right-hand sides"), this function returns `X` of shape
  `(*, n)` or `(*, n, k)` respectively.
- Otherwise, if :attr:`A` has shape `(*, n, n)` and  :attr:`B` has shape `(n,)`  or `(n, k)`, :attr:`B`
  is broadcasted to have shape `(*, n)` or `(*, n, k)` respectively.
  This function then returns the solution of the resulting batch of systems of linear equations.

.. note::
    This function computes `X = \ `:attr:`A`\ `.inverse() @ \ `:attr:`B` in a faster and
    more numerically stable way than performing the computations separately.

.. note::
    It is possible to compute the solution of the system :math:`XA = B` by passing the inputs
    :attr:`A` and :attr:`B` transposed and transposing the output returned by this function.

""" + fr"""
.. note:: {common_notes["sync_note_has_ex"].format("torch.linalg.solve_ex")}
""" + r"""

.. seealso::

        :func:`torch.linalg.solve_triangular` computes the solution of a triangular system of linear
        equations with a unique solution.

Args:
    A (Tensor): tensor of shape `(*, n, n)` where `*` is zero or more batch dimensions.
    B (Tensor): right-hand side tensor of shape `(*, n)` or  `(*, n, k)` or `(n,)` or `(n, k)`
                according to the rules described above

Keyword args:
    left (bool, optional): whether to solve the system :math:`AX=B` or :math:`XA = B`. Default: `True`.
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.

Raises:
    RuntimeError: if the :attr:`A` matrix is not invertible or any matrix in a batched :attr:`A`
                  is not invertible.

Examples::

    >>> A = torch.randn(3, 3)
    >>> b = torch.randn(3)
    >>> x = torch.linalg.solve(A, b)
    >>> torch.allclose(A @ x, b)
    True

""")
    # 创建一个形状为 (2, 3, 3) 的随机张量 A
    >>> A = torch.randn(2, 3, 3)
    # 创建一个形状为 (2, 3, 4) 的随机张量 B
    >>> B = torch.randn(2, 3, 4)
    # 使用 torch.linalg.solve 求解方程 A @ X = B，其中 X 是未知的张量
    >>> X = torch.linalg.solve(A, B)
    # 打印 X 的形状，应为 (2, 3, 4)
    >>> X.shape
    torch.Size([2, 3, 4])
    # 验证 A @ X 是否接近于 B
    >>> torch.allclose(A @ X, B)
    True

    # 创建一个形状为 (2, 3, 3) 的随机张量 A
    >>> A = torch.randn(2, 3, 3)
    # 创建一个形状为 (3, 1) 的随机张量 b
    >>> b = torch.randn(3, 1)
    # 使用 torch.linalg.solve 求解方程 A @ x = b，其中 b 被广播到尺寸 (2, 3, 1)
    >>> x = torch.linalg.solve(A, b)
    # 打印 x 的形状，应为 (2, 3, 1)
    >>> x.shape
    torch.Size([2, 3, 1])
    # 验证 A @ x 是否接近于 b
    >>> torch.allclose(A @ x, b)
    True

    # 创建一个形状为 (2, 3, 3) 的随机张量 A
    >>> A = torch.randn(2, 3, 3)
    # 创建一个形状为 (3,) 的随机张量 b
    >>> b = torch.randn(3)
    # 使用 torch.linalg.solve 求解方程 A @ x = b，其中 b 被广播到尺寸 (2, 3)
    >>> x = torch.linalg.solve(A, b)
    # 打印 x 的形状，应为 (2, 3)
    >>> x.shape
    torch.Size([2, 3])
    # 计算 A @ x，并将 x 在最后一个维度上增加一维以匹配 A 的形状
    >>> Ax = A @ x.unsqueeze(-1)
    # 验证 Ax 是否接近于 b 扩展到与 Ax 相同形状
    >>> torch.allclose(Ax, b.unsqueeze(-1).expand_as(Ax))
    True
# 将 solve_triangular 函数重命名为 _linalg.linalg_solve_triangular，并添加文档字符串
solve_triangular = _add_docstr(_linalg.linalg_solve_triangular, r"""
linalg.solve_triangular(A, B, *, upper, left=True, unitriangular=False, out=None) -> Tensor

计算具有唯一解的三角线性方程组的解。

假设 :math:`\mathbb{K}` 是 :math:`\mathbb{R}` 或 :math:`\mathbb{C}`，
此函数计算与三角矩阵 :math:`A \in \mathbb{K}^{n \times n}`（对角线上没有零，即可逆矩阵）和矩形矩阵 :math:`B \in \mathbb{K}^{n \times k}` 关联的线性系统的解 :math:`X \in \mathbb{K}^{n \times k}`，
即解方程

.. math:: AX = B

参数 :attr:`upper` 表示矩阵 :math:`A` 是上三角还是下三角。

如果 :attr:`left`\ `= False`，则返回矩阵 :math:`X \in \mathbb{K}^{n \times k}`，解决方程

.. math::

    XA = B\mathrlap{\qquad A \in \mathbb{K}^{k \times k}, B \in \mathbb{K}^{n \times k}.}

如果 :attr:`upper`\ `= True`（或 `False`），将仅访问 :attr:`A` 的上（或下）三角部分。
主对角线下方的元素将被视为零并且不会被访问。

如果 :attr:`unitriangular`\ `= True`，假定 :attr:`A` 的对角线元素为1，不会被访问。

如果对角线元素包含零或非常接近零，并且 :attr:`unitriangular`\ `= False`（默认），
或者输入矩阵具有非常小的特征值，则结果可能包含 `NaN`。

支持 float、double、cfloat 和 cdouble 数据类型的输入。
还支持批处理的矩阵输入，如果输入是矩阵的批处理，则输出具有相同的批处理维度。

.. seealso::

    :func:`torch.linalg.solve` 计算具有唯一解的一般方阵线性系统的解。

参数:
    A (Tensor): 形状为 `(*, n, n)`（或 `(*, k, k)` 如果 :attr:`left`\ `= True`）的张量，
                其中 `*` 是零个或多个批处理维度。
    B (Tensor): 形状为 `(*, n, k)` 的右侧张量。

关键字参数:
    upper (bool): :attr:`A` 是否是上三角矩阵。
    left (bool, 可选): 是否解决系统 :math:`AX=B` 或 :math:`XA = B`。默认值为 `True`。
    unitriangular (bool, 可选): 如果为 `True`，假设 :attr:`A` 的对角线元素全部等于 `1`。默认值为 `False`。
    out (Tensor, 可选): 输出张量。可以将 `B` 作为 `out` 传递，并在 `B` 上原地计算结果。如果为 `None`，则忽略。默认值为 `None`。

示例::

    >>> A = torch.randn(3, 3).triu_()
    >>> B = torch.randn(3, 4)
    >>> X = torch.linalg.solve_triangular(A, B, upper=True)
    >>> torch.allclose(A @ X, B)
    True

    >>> A = torch.randn(2, 3, 3).tril_()
    >>> B = torch.randn(2, 3, 4)
""")
    # 使用 torch.linalg.solve_triangular 解决线性方程组 A @ X = B，其中 A 是下三角矩阵，返回 X
    >>> X = torch.linalg.solve_triangular(A, B, upper=False)
    
    # 检查矩阵乘积 A @ X 是否接近于 B，返回布尔值
    >>> torch.allclose(A @ X, B)
    True
    
    # 创建一个形状为 (2, 4, 4) 的随机张量 A，并将其转换为下三角矩阵
    >>> A = torch.randn(2, 4, 4).tril_()
    
    # 创建一个形状为 (2, 3, 4) 的随机张量 B
    >>> B = torch.randn(2, 3, 4)
    
    # 使用 torch.linalg.solve_triangular 解决线性方程组 X @ A = B，其中 A 是下三角矩阵，返回 X
    >>> X = torch.linalg.solve_triangular(A, B, upper=False, left=False)
    
    # 检查矩阵乘积 X @ A 是否接近于 B，返回布尔值
    >>> torch.allclose(X @ A, B)
    True
lu_factor = _add_docstr(_linalg.linalg_lu_factor, r"""
linalg.lu_factor(A, *, bool pivot=True, out=None) -> (Tensor, Tensor)

Computes a compact representation of the LU factorization with partial pivoting of a matrix.

This function computes a compact representation of the decomposition given by :func:`torch.linalg.lu`.
If the matrix is square, this representation may be used in :func:`torch.linalg.lu_solve`
to solve system of linear equations that share the matrix :attr:`A`.

The returned decomposition is represented as a named tuple `(LU, pivots)`.
The ``LU`` matrix has the same shape as the input matrix ``A``. Its upper and lower triangular
parts encode the non-constant elements of ``L`` and ``U`` of the LU decomposition of ``A``.

The returned permutation matrix is represented by a 1-indexed vector. `pivots[i] == j` represents
that in the `i`-th step of the algorithm, the `i`-th row was permuted with the `j-1`-th row.

On CUDA, one may use :attr:`pivot`\ `= False`. In this case, this function returns the LU
decomposition without pivoting if it exists.

Supports inputs of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if the inputs are batches of matrices then
the output has the same batch dimensions.

""" + fr"""
.. note:: {common_notes["sync_note_has_ex"].format("torch.linalg.lu_factor_ex")}
""" + r"""
.. warning:: The LU decomposition is almost never unique, as often there are different permutation
             matrices that can yield different LU decompositions.
             As such, different platforms, like SciPy, or inputs on different devices,
             may produce different valid decompositions.

             Gradient computations are only supported if the input matrix is full-rank.
             If this condition is not met, no error will be thrown, but the gradient may not be finite.
             This is because the LU decomposition with pivoting is not differentiable at these points.

.. seealso::

        :func:`torch.linalg.lu_solve` solves a system of linear equations given the output of this
        function provided the input matrix was square and invertible.

        :func:`torch.lu_unpack` unpacks the tensors returned by :func:`~lu_factor` into the three
        matrices `P, L, U` that form the decomposition.

        :func:`torch.linalg.lu` computes the LU decomposition with partial pivoting of a possibly
        non-square matrix. It is a composition of :func:`~lu_factor` and :func:`torch.lu_unpack`.

        :func:`torch.linalg.solve` solves a system of linear equations. It is a composition
        of :func:`~lu_factor` and :func:`~lu_solve`.

Args:
    A (Tensor): tensor of shape `(*, m, n)` where `*` is zero or more batch dimensions.

Keyword args:

""")  # 将给定函数 _linalg.linalg_lu_factor 用 _add_docstr 添加文档字符串
    # 设置参数 pivot，指定是否计算带部分选主元的 LU 分解（Partial Pivoting LU Decomposition），或普通的 LU 分解
    # 设定默认值为 True，表示计算带部分选主元的 LU 分解。如果设置为 False，在 CPU 上不支持。
    pivot (bool, optional): Whether to compute the LU decomposition with partial pivoting, or the regular LU
                            decomposition. :attr:`pivot`\ `= False` not supported on CPU. Default: `True`.
    # 设置参数 out，指定一个包含两个张量的元组，用于写入输出结果。如果为 None，则忽略此参数。默认值为 None。
    out (tuple, optional): tuple of two tensors to write the output to. Ignored if `None`. Default: `None`.
# 返回一个命名元组 `(LU, pivots)`。

# 如果矩阵 :attr:`A` 不可逆或者批处理中任何一个矩阵 :attr:`A` 不可逆，则抛出 RuntimeError 异常。
# 示例：
# 创建一个大小为 2x3x3 的张量 A
>>> A = torch.randn(2, 3, 3)
# 创建两个大小为 2x3x4 和 2x3x7 的张量 B1 和 B2
>>> B1 = torch.randn(2, 3, 4)
>>> B2 = torch.randn(2, 3, 7)
# 对 A 进行 LU 分解，返回 LU 分解的结果 LU 和 pivots
>>> LU, pivots = torch.linalg.lu_factor(A)
# 使用 LU 分解的结果解决线性方程组 AX = B1，返回解 X1
>>> X1 = torch.linalg.lu_solve(LU, pivots, B1)
# 使用 LU 分解的结果解决线性方程组 AX = B2，返回解 X2
>>> X2 = torch.linalg.lu_solve(LU, pivots, B2)
# 检查 A @ X1 是否接近于 B1
>>> torch.allclose(A @ X1, B1)
True
# 检查 A @ X2 是否接近于 B2
>>> torch.allclose(A @ X2, B2)
True

# 定义链接到可逆矩阵相关概念的超链接
.. _invertible:
    https://en.wikipedia.org/wiki/Invertible_matrix#The_invertible_matrix_theorem


lu_factor_ex = _add_docstr(_linalg.linalg_lu_factor_ex, r"""
linalg.lu_factor_ex(A, *, pivot=True, check_errors=False, out=None) -> (Tensor, Tensor, Tensor)

这是一个不进行错误检查（除非 :attr:`check_errors`\ `= True`）的 :func:`~lu_factor` 版本。
它还返回 `LAPACK's getrf`_ 返回的 :attr:`info` 张量。

.. note:: {common_notes["sync_note_ex"]}

.. warning:: {common_notes["experimental_warning"]}

Args:
    A (Tensor): 形状为 `(*, m, n)` 的张量，其中 `*` 是零个或多个批处理维度。

Keyword args:
    pivot (bool, 可选): 是否使用部分主元素选取进行 LU 分解，或者正常的 LU 分解。在 CPU 上不支持 :attr:`pivot`\ `= False`。默认值：`True`。
    check_errors (bool, 可选): 控制是否检查 ``infos`` 的内容，如果非零则抛出错误。默认值：`False`。
    out (tuple, 可选): 三个张量的元组，用于写入输出。如果为 `None`，则忽略。默认值：`None`。

Returns:
    一个命名元组 `(LU, pivots, info)`。

.. _LAPACK's getrf:
    https://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga0019443faea08275ca60a734d0593e60.html
""")

lu_solve = _add_docstr(_linalg.linalg_lu_solve, r"""
linalg.lu_solve(LU, pivots, B, *, left=True, adjoint=False, out=None) -> Tensor

使用 LU 分解计算具有唯一解的方程组的解。

设 :math:`\mathbb{K}` 为 :math:`\mathbb{R}` 或 :math:`\mathbb{C}`，
此函数计算与 :math:`A \in \mathbb{K}^{n \times n}, B \in \mathbb{K}^{n \times k}` 相关的 **线性系统** 的解
:math:`X \in \mathbb{K}^{n \times k}`，其中定义为

.. math:: AX = B

其中 :math:`A` 被作为 :func:`~lu_factor` 返回的分解因子化。

如果 :attr:`left`\ `= False`，此函数返回解决系统

.. math::

    XA = B\mathrlap{\qquad A \in \mathbb{K}^{k \times k}, B \in \mathbb{K}^{n \times k}.}

如果 :attr:`adjoint`\ `= True`（且 :attr:`left`\ `= True`），给定 :math:`A` 的 LU 分解，
此函数返回解决系统的 :math:`X \in \mathbb{K}^{n \times k}`

.. math::
    A^{\text{H}}X = B\mathrlap{\qquad A \in \mathbb{K}^{k \times k}, B \in \mathbb{K}^{n \times k}.}


    # 计算共轭转置矩阵 A 的 Hermitian 乘积 X，结果等于矩阵 B
    A^{\text{H}}X = B
    # 其中矩阵 A 的维度为 k × k，矩阵 B 的维度为 n × k
    \qquad A \in \mathbb{K}^{k \times k}, B \in \mathbb{K}^{n \times k}.
lu = _add_docstr(_linalg.linalg_lu, r"""
lu(A, *, pivot=True, out=None) -> (Tensor, Tensor, Tensor)

Computes the LU decomposition with partial pivoting of a matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **LU decomposition with partial pivoting** of a matrix
:math:`A \in \mathbb{K}^{m \times n}` is defined as

.. math::

    A = PLU\mathrlap{\qquad P \in \mathbb{K}^{m \times m}, L \in \mathbb{K}^{m \times k}, U \in \mathbb{K}^{k \times n}}

where `k = min(m,n)`, :math:`P` is a `permutation matrix`_, :math:`L` is lower triangular with ones on the diagonal
and :math:`U` is upper triangular.

If :attr:`pivot`\ `= False` and :attr:`A` is on GPU, then the **LU decomposition without pivoting** is computed

.. math::

    A = LU\mathrlap{\qquad L \in \mathbb{K}^{m \times k}, U \in \mathbb{K}^{k \times n}}

When :attr:`pivot`\ `= False`, the returned matrix :attr:`P` will be empty.
The LU decomposition without pivoting `may not exist`_ if any of the principal minors of :attr:`A` is singular.
""")



# Computes the LU decomposition of a matrix A with optional partial pivoting.

Letting :math:`\mathbb{K}` be the field of real or complex numbers,
the function decomposes :math:`A \in \mathbb{K}^{m \times n}` into matrices
P (permutation matrix), L (lower triangular with ones on diagonal), and U (upper triangular).

Args:
    A (Tensor): Input matrix of shape `(*, m, n)`.
    pivot (bool, optional): Whether to perform partial pivoting. Default: `True`.
    out (Tensor, optional): Output tensor. Ignored if `None`. Default: `None`.

Returns:
    Tuple containing tensors P (permutation matrix), L (lower triangular matrix),
    and U (upper triangular matrix) such that A = PLU.

Raises:
    RuntimeError: If LU decomposition without pivoting fails due to singularity.

Notes:
    - Supports batched inputs; maintains batch dimensions in outputs.
    - If `pivot=False` and A is on GPU, computes LU decomposition without pivoting.

See Also:
    `Invertible Matrix Theorem <https://en.wikipedia.org/wiki/Invertible_matrix#The_invertible_matrix_theorem>`_

"""
tensorinv = _add_docstr(_linalg.linalg_tensorinv, r"""
linalg.tensorinv(A, ind=2, *, out=None) -> Tensor

计算 :func:`torch.tensordot` 的乘法逆。

如果 `m` 是 :attr:`A` 的前 :attr:`ind` 维度的乘积，`n` 是剩余维度的乘积，该函数期望 `m` 和 `n` 相等。
在这种情况下，它计算一个张量 `X`，使得 `tensordot(\ `:attr:`A`\ `, X, \ `:attr:`ind`\ `)` 在维度 `m` 上是单位矩阵。
`X` 的形状与 :attr:`A` 相同，但将前 :attr:`ind` 维度推到末尾。

.. code:: text

    X.shape == A.shape[ind:] + A.shape[:ind]

支持 float、double、cfloat 和 cdouble 类型的输入。
""")
# 定义函数 `tensorsolve`，用于解决方程 `torch.tensordot(A, X) = B`，返回解 `X`。
tensorsolve = _add_docstr(_linalg.linalg_tensorsolve, r"""
linalg.tensorsolve(A, B, dims=None, *, out=None) -> Tensor

# 计算系统 `torch.tensordot(A, X) = B` 的解 `X`。

# 如果 `m` 是 `A` 的前 `B.ndim` 维度的乘积，而 `n` 是其余维度的乘积，该函数要求 `m` 和 `n` 相等。
# 返回的张量 `x` 满足 `tensordot(A, x, dims=x.ndim) == B`。
# `x` 的形状为 `A[B.ndim:]`。

# 如果指定了 `dims`，`A` 将被重塑为：
# A = movedim(A, dims, range(len(dims) - A.ndim + 1, 0))

# 支持 float、double、cfloat 和 cdouble 数据类型的输入。

.. seealso::

        :func:`torch.linalg.tensorinv` 计算 `torch.tensordot` 的乘法逆。

Args:
    A (Tensor): 需要解的张量。其形状必须满足 `prod(A.shape[:B.ndim]) == prod(A.shape[B.ndim:])`。
    B (Tensor): 形状为 `A.shape[:B.ndim]` 的张量。

"""
)
    dims (Tuple[int], optional): dimensions of :attr:`A` to be moved.
        If `None`, no dimensions are moved. Default: `None`.
Keyword args:
    out (Tensor, optional): output tensor. Ignored if `None`. Default: `None`.
        输出张量，可选参数。如果为 `None`，则忽略。默认为 `None`。

Raises:
    RuntimeError: if the reshaped :attr:`A`\ `.view(m, m)` with `m` as above  is not
                  invertible or the product of the first :attr:`ind` dimensions is not equal
                  to the product of the rest of the dimensions.
        当重塑的 :attr:`A`\ `.view(m, m)` 不可逆或者第一个 :attr:`ind` 维度的乘积不等于剩余维度的乘积时，抛出 `RuntimeError` 异常。

Examples::

    >>> A = torch.eye(2 * 3 * 4).reshape((2 * 3, 4, 2, 3, 4))
    >>> B = torch.randn(2 * 3, 4)
    >>> X = torch.linalg.tensorsolve(A, B)
    >>> X.shape
    torch.Size([2, 3, 4])
    >>> torch.allclose(torch.tensordot(A, X, dims=X.ndim), B)
    True
        示例代码，展示了如何使用 `torch.linalg.tensorsolve` 函数求解张量方程。

    >>> A = torch.randn(6, 4, 4, 3, 2)
    >>> B = torch.randn(4, 3, 2)
    >>> X = torch.linalg.tensorsolve(A, B, dims=(0, 2))
    >>> X.shape
    torch.Size([6, 4])
    >>> A = A.permute(1, 3, 4, 0, 2)
    >>> A.shape[B.ndim:]
    torch.Size([6, 4])
    >>> torch.allclose(torch.tensordot(A, X, dims=X.ndim), B, atol=1e-6)
    True
        更多示例代码，展示了如何使用 `torch.linalg.tensorsolve` 函数解决具有不同维度的张量方程。


qr = _add_docstr(_linalg.linalg_qr, r"""
qr(A, mode='reduced', *, out=None) -> (Tensor, Tensor)

Computes the QR decomposition of a matrix.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **full QR decomposition** of a matrix
:math:`A \in \mathbb{K}^{m \times n}` is defined as

.. math::

    A = QR\mathrlap{\qquad Q \in \mathbb{K}^{m \times m}, R \in \mathbb{K}^{m \times n}}

where :math:`Q` is orthogonal in the real case and unitary in the complex case,
and :math:`R` is upper triangular with real diagonal (even in the complex case).

When `m > n` (tall matrix), as `R` is upper triangular, its last `m - n` rows are zero.
In this case, we can drop the last `m - n` columns of `Q` to form the
**reduced QR decomposition**:

.. math::

    A = QR\mathrlap{\qquad Q \in \mathbb{K}^{m \times n}, R \in \mathbb{K}^{n \times n}}

The reduced QR decomposition agrees with the full QR decomposition when `n >= m` (wide matrix).

Supports input of float, double, cfloat and cdouble dtypes.
Also supports batches of matrices, and if :attr:`A` is a batch of matrices then
the output has the same batch dimensions.

The parameter :attr:`mode` chooses between the full and reduced QR decomposition.
If :attr:`A` has shape `(*, m, n)`, denoting `k = min(m, n)`

- :attr:`mode`\ `= 'reduced'` (default): Returns `(Q, R)` of shapes `(*, m, k)`, `(*, k, n)` respectively.
  It is always differentiable.
- :attr:`mode`\ `= 'complete'`: Returns `(Q, R)` of shapes `(*, m, m)`, `(*, m, n)` respectively.
  It is differentiable for `m <= n`.
- :attr:`mode`\ `= 'r'`: Computes only the reduced `R`. Returns `(Q, R)` with `Q` empty and `R` of shape `(*, k, n)`.
  It is never differentiable.

Differences with `numpy.linalg.qr`:

- :attr:`mode`\ `= 'raw'` is not implemented.
- Unlike `numpy.linalg.qr`, this function always returns a tuple of two tensors.
  When :attr:`mode`\ `= 'r'`, the `Q` tensor is an empty tensor.


注释：
vander = _add_docstr(_linalg.linalg_vander, r"""
vander(x, N=None) -> Tensor

生成一个范德蒙矩阵。

返回范德蒙矩阵 :math:`V`

.. math::

    V = \begin{pmatrix}
            1 & x_1 & x_1^2 & \dots & x_1^{N-1}\\
            1 & x_2 & x_2^2 & \dots & x_2^{N-1}\\
            1 & x_3 & x_3^2 & \dots & x_3^{N-1}\\
            \vdots & \vdots & \vdots & \ddots &\vdots \\
            1 & x_n & x_n^2 & \dots & x_n^{N-1}
        \end{pmatrix}.

当 `N > 1` 时。
如果 :attr:`N`\ `= None`，则 `N = x.size(-1)`，这样输出将是一个方阵。

支持 float、double、cfloat、cdouble 和整数类型的输入。
还支持向量批次输入，如果 :attr:`x` 是向量批次，则输出具有相同的批次维度。

与 `numpy.vander` 的区别：

- 与 `numpy.vander` 不同，此函数返回 :attr:`x` 的幂按升序排列。
  若要按相反顺序获取它们，请调用 ``linalg.vander(x, N).flip(-1)``。

Args:
    x (Tensor): 输入向量或向量批次
    N (int, optional): 范德蒙矩阵的列数。默认为 `None`，即 `N = x.size(-1)`

Returns:
    Tensor: 范德蒙矩阵 :math:`V`
""")
    x (Tensor): tensor的形状为 `(*, n)`，其中 `*` 表示零个或多个批次维度，每个维度包含一个向量。
# 导入 torch 库中的 linalg 模块
import torch.linalg as _linalg

# 函数签名和说明文档
vecdot = _add_docstr(_linalg.linalg_vecdot, r"""
linalg.vecdot(x, y, *, dim=-1, out=None) -> Tensor

计算两批向量沿指定维度的点积。

符号表示，该函数计算：

\[
\sum_{i=1}^n \overline{x_i}y_i.
\]

其中对于复向量，\(\overline{x_i}\) 表示共轭，对于实向量，它是自身。

支持半精度、BF16、单精度、双精度、复数单精度、复数双精度和整数数据类型。
支持广播操作。

Args:
    x (Tensor): 第一批形状为 `(*, n)` 的向量。
    y (Tensor): 第二批形状为 `(*, n)` 的向量。

Keyword args:
    dim (int): 计算点积的维度。默认为 `-1`。
    out (Tensor, optional): 输出张量。如果为 `None`，则忽略。默认为 `None`。

Examples::

    >>> v1 = torch.randn(3, 2)
    >>> v2 = torch.randn(3, 2)
    >>> linalg.vecdot(v1, v2)
    tensor([ 0.3223,  0.2815, -0.1944])
    >>> torch.vdot(v1[0], v2[0])
    tensor(0.3223)
""")
```