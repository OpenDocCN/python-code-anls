# `D:\src\scipysrc\scipy\scipy\linalg\interpolative.py`

```
# Python module for interfacing with `id_dist`.

r"""
======================================================================
Interpolative matrix decomposition (:mod:`scipy.linalg.interpolative`)
======================================================================

.. moduleauthor:: Kenneth L. Ho <klho@stanford.edu>

.. versionadded:: 0.13

.. currentmodule:: scipy.linalg.interpolative

An interpolative decomposition (ID) of a matrix :math:`A \in
\mathbb{C}^{m \times n}` of rank :math:`k \leq \min \{ m, n \}` is a
factorization

.. math::
  A \Pi =
  \begin{bmatrix}
   A \Pi_{1} & A \Pi_{2}
  \end{bmatrix} =
  A \Pi_{1}
  \begin{bmatrix}
   I & T
  \end{bmatrix},

where :math:`\Pi = [\Pi_{1}, \Pi_{2}]` is a permutation matrix with
:math:`\Pi_{1} \in \{ 0, 1 \}^{n \times k}`, i.e., :math:`A \Pi_{2} =
A \Pi_{1} T`. This can equivalently be written as :math:`A = BP`,
where :math:`B = A \Pi_{1}` and :math:`P = [I, T] \Pi^{\mathsf{T}}`
are the *skeleton* and *interpolation matrices*, respectively.

If :math:`A` does not have exact rank :math:`k`, then there exists an
approximation in the form of an ID such that :math:`A = BP + E`, where
:math:`\| E \| \sim \sigma_{k + 1}` is on the order of the :math:`(k +
1)`-th largest singular value of :math:`A`. Note that :math:`\sigma_{k
# 导入 scipy.linalg.interpolative 模块，别名为 sli
import scipy.linalg.interpolative as sli

# 导入 scipy.linalg 模块中的 hilbert 函数
from scipy.linalg import hilbert

# 设定 Hilbert 矩阵的维度为 1000
n = 1000
# 导入 numpy 库，用于创建和处理数组
>>> import numpy as np
# 设定矩阵的大小为 n x n，并以 Fortran 风格的顺序创建空数组 A
>>> n = 1000
>>> A = np.empty((n, n), order='F')
# 使用嵌套循环为矩阵 A 的每个元素赋值，以使其成为 Hilbert 矩阵的逆
>>> for j in range(n):
...     for i in range(n):
...         A[i,j] = 1. / (i + j + 1)

# 导入 scipy.sparse.linalg 库中的 aslinearoperator 函数，将矩阵 A 转换为线性操作器 L
>>> from scipy.sparse.linalg import aslinearoperator
>>> L = aslinearoperator(A)

# 此操作自动设置描述矩阵及其共轭在向量上操作的方法

# 计算插值分解（ID）
# ------------------

# 我们有几种算法选择来计算插值分解（ID），主要根据以下两个对立面：

# 1. 矩阵的表示方式，即通过其条目或通过其在向量上的作用；
# 2. 是否将其近似到固定的相对精度或固定的秩。

# 依次讨论每种选择。

# 在所有情况下，ID 由三个参数表示：

# 1. 秩 k；
# 2. 索引数组 idx；
# 3. 插值系数 proj。

# ID 由以下关系指定：np.dot(A[:,idx[:k]], proj) == A[:,idx[k:]]。

# 从矩阵条目
# ...............

# 首先考虑以其条目形式给出的矩阵。

# 要计算到固定精度的 ID，请输入：
>>> eps = 1e-3
>>> k, idx, proj = sli.interp_decomp(A, eps)

# 其中 eps < 1 是期望的精度。

# 要计算到固定秩的 ID，请使用：
>>> idx, proj = sli.interp_decomp(A, k)

# 其中 k >= 1 是所需的秩。

# 这两种算法使用随机抽样，通常比相应的旧的确定性算法更快，可以通过以下命令访问：
>>> k, idx, proj = sli.interp_decomp(A, eps, rand=False)

# 和：
>>> idx, proj = sli.interp_decomp(A, k, rand=False)

# 分别进行访问。

# 从矩阵的作用
# ...............

# 现在考虑一个以其在向量上的作用作为 :class:`scipy.sparse.linalg.LinearOperator` 给出的矩阵。

# 要计算到固定精度的 ID，请输入：
>>> k, idx, proj = sli.interp_decomp(L, eps)

# 要计算到固定秩的 ID，请使用：
>>> idx, proj = sli.interp_decomp(L, k)

# 这些算法是随机的。

# 重构 ID
# --------------------

# 上述 ID 程序不会显式输出骨架和插值矩阵，而是以更紧凑（有时更有用）的形式返回相关信息。要构建这些矩阵，请写：

# 用于骨架矩阵的函数：
>>> B = sli.reconstruct_skel_matrix(A, k, idx)

# 用于插值矩阵的函数：
>>> P = sli.reconstruct_interp_matrix(idx, proj)

# 然后可以计算 ID 近似值为：
>>> C = np.dot(B, P)

# 还可以直接使用以下方式构建：
>>> C = sli.reconstruct_matrix_from_id(B, idx, proj)

# 而无需先计算 P。
# Alternatively, this can be done explicitly as well using:
# 另外，也可以显式地使用以下方法来完成：

>>> B = A[:,idx[:k]]
# 从矩阵 A 中选择列索引 idx[:k] 对应的列，形成新的矩阵 B

>>> P = np.hstack([np.eye(k), proj])[:,np.argsort(idx)]
# 创建一个矩阵 P，由单位矩阵 np.eye(k) 和 proj 水平堆叠而成，并按照索引 idx 进行排序

>>> C = np.dot(B, P)
# 计算矩阵乘积 C = B * P

Computing an SVD
----------------
# 计算奇异值分解（SVD）

An ID can be converted to an SVD via the command:
# 可以通过以下命令将一个 ID 转换为 SVD：

>>> U, S, V = sli.id_to_svd(B, idx, proj)
# 使用 sli.id_to_svd 函数，将标识 ID 转换为奇异值分解 U, S, V

The SVD approximation is then:
# 然后得到的奇异值分解近似为：

>>> approx = U @ np.diag(S) @ V.conj().T
# 近似值 approx = U * S * V 的共轭转置

The SVD can also be computed "fresh" by combining both the ID and conversion
steps into one command. Following the various ID algorithms above, there are
correspondingly various SVD algorithms that one can employ.
# 也可以通过将 ID 和转换步骤合并为一个命令来直接计算 SVD。根据上述各种 ID 算法，可以相应地使用各种 SVD 算法。

From matrix entries
...................
# 从矩阵的条目入手

We consider first SVD algorithms for a matrix given in terms of its entries.
# 我们首先考虑给定矩阵条目的 SVD 算法。

To compute an SVD to a fixed precision, type:
# 要计算一个到固定精度的 SVD，请输入：

>>> U, S, V = sli.svd(A, eps)
# 使用 sli.svd 函数，计算矩阵 A 的 SVD，达到指定精度 eps

To compute an SVD to a fixed rank, use:
# 要计算一个到固定秩的 SVD，请使用：

>>> U, S, V = sli.svd(A, k)
# 使用 sli.svd 函数，计算矩阵 A 的 SVD，达到指定秩 k

Both algorithms use random sampling; for the deterministic versions, issue the
keyword ``rand=False`` as above.
# 这两种算法都使用随机抽样；对于确定性版本，请像上面一样使用关键字 ``rand=False``。

From matrix action
..................
# 从矩阵作用入手

Now consider a matrix given in terms of its action on a vector.
# 现在考虑一个矩阵，根据其在向量上的作用来描述。

To compute an SVD to a fixed precision, type:
# 要计算一个到固定精度的 SVD，请输入：

>>> U, S, V = sli.svd(L, eps)
# 使用 sli.svd 函数，计算矩阵 L 的 SVD，达到指定精度 eps

To compute an SVD to a fixed rank, use:
# 要计算一个到固定秩的 SVD，请使用：

>>> U, S, V = sli.svd(L, k)
# 使用 sli.svd 函数，计算矩阵 L 的 SVD，达到指定秩 k

Utility routines
----------------
# 实用程序例程

Several utility routines are also available.
# 还有几个实用程序例程可用。

To estimate the spectral norm of a matrix, use:
# 要估计矩阵的谱范数，请使用：

>>> snorm = sli.estimate_spectral_norm(A)
# 使用 sli.estimate_spectral_norm 函数，估计矩阵 A 的谱范数

This algorithm is based on the randomized power method and thus requires only
matrix-vector products. The number of iterations to take can be set using the
keyword ``its`` (default: ``its=20``). The matrix is interpreted as a
:class:`scipy.sparse.linalg.LinearOperator`, but it is also valid to supply it
as a :class:`numpy.ndarray`, in which case it is trivially converted using
:func:`scipy.sparse.linalg.aslinearoperator`.
# 此算法基于随机功率方法，因此仅需要矩阵-向量乘积。可以使用关键字 ``its`` 设置迭代次数（默认为 ``its=20``）。该矩阵被解释为一个 :class:`scipy.sparse.linalg.LinearOperator`，但也可以作为一个 :class:`numpy.ndarray` 提供，此时可以使用 :func:`scipy.sparse.linalg.aslinearoperator` 来轻松转换。

The same algorithm can also estimate the spectral norm of the difference of two
matrices ``A1`` and ``A2`` as follows:
# 同样的算法也可以估计两个矩阵 ``A1`` 和 ``A2`` 差的谱范数，如下所示：

>>> A1, A2 = A**2, A
>>> diff = sli.estimate_spectral_norm_diff(A1, A2)
# 使用 sli.estimate_spectral_norm_diff 函数，估计矩阵 A1 和 A2 的谱范数差

This is often useful for checking the accuracy of a matrix approximation.
# 这通常用于检查矩阵逼近的准确性。

Some routines in :mod:`scipy.linalg.interpolative` require estimating the rank
of a matrix as well. This can be done with either:
# :mod:`scipy.linalg.interpolative` 中的一些例程还需要估计矩阵的秩。可以使用以下方法之一来完成：

>>> k = sli.estimate_rank(A, eps)
# 使用 sli.estimate_rank 函数，估计矩阵 A 的秩，精度由参数 eps 控制

or:
# 或者：

>>> k = sli.estimate_rank(L, eps)
# 使用 sli.estimate_rank 函数，估计矩阵 L 的秩，精度由参数 eps 控制

depending on the representation. The parameter ``eps`` controls the definition
of the numerical rank.
# 根据表示方式而定。参数 ``eps`` 控制数值秩的定义。

Finally, the random number generation required for all randomized routines can
be controlled via :func:`scipy.linalg.interpolative.seed`. To reset the seed
values to their original values, use:
# 最后，所有随机例程所需的随机数生成可以通过 :func:`scipy.linalg.interpolative.seed` 进行控制。要将种子值重置为其原始值，请使用：

>>> sli.seed('default')
# 使用 sli.seed 函数，将种子值重置为默认值

To specify the seed values, use:
# 要指定种子值，请使用：

>>> s = 42
>>> sli.seed(s)
# 使用 sli.seed 函数，将种子值设置为 s

where ``s`` must be an integer or array of 55 floats. If an integer, the array
of floats is obtained by using ``numpy.random.rand`` with the given integer
seed.
# 这里的 ``s`` 必须是一个整数或者包含 55 个浮点数的数组。如果是整数，则使用给定的整数种子生成一个浮点数数组。

To simply generate some random numbers, type:
# 要生成一些随机数，请输入：

>>> arr = sli.rand(n)
# 使用 sli.rand 函数，生成 n 个随机数
"""
The above functions all automatically detect the appropriate interface and work
with both real and complex data types, passing input arguments to the proper
backend routine.

"""

import scipy.linalg._interpolative_backend as _backend  # 导入 Scipy 的内部库 _interpolative_backend
import numpy as np  # 导入 NumPy 库
import sys  # 导入系统相关的模块

__all__ = [  # 定义可以导出的函数列表
    'estimate_rank',
    'estimate_spectral_norm',
    'estimate_spectral_norm_diff',
    'id_to_svd',
    'interp_decomp',
    'rand',
    'reconstruct_interp_matrix',
    'reconstruct_matrix_from_id',
    'reconstruct_skel_matrix',
    'seed',
    'svd',
]

_DTYPE_ERROR = ValueError("invalid input dtype (input must be float64 or complex128)")  # 定义数据类型错误异常
_TYPE_ERROR = TypeError("invalid input type (must be array or LinearOperator)")  # 定义输入类型错误异常
_32BIT_ERROR = ValueError("interpolative decomposition on 32-bit systems "
                          "with complex128 is buggy")  # 定义32位系统下使用complex128会出现的错误
_IS_32BIT = (sys.maxsize < 2**32)  # 检测当前系统是否是32位系统


def _is_real(A):
    """
    Check if the input array `A` is of real type (float64) or complex.

    Parameters
    ----------
    A : array-like
        Input array to check.

    Returns
    -------
    bool
        True if `A` is real (float64), False if `A` is complex (complex128).

    Raises
    ------
    _DTYPE_ERROR
        If `A` is not float64 or complex128.
    _TYPE_ERROR
        If `A` is not an array-like object.

    """
    try:
        if A.dtype == np.complex128:  # 检查数组的数据类型是否为 complex128
            return False
        elif A.dtype == np.float64:  # 检查数组的数据类型是否为 float64
            return True
        else:
            raise _DTYPE_ERROR  # 抛出数据类型错误异常
    except AttributeError as e:
        raise _TYPE_ERROR from e  # 抛出输入类型错误异常


def seed(seed=None):
    """
    Seed the internal random number generator used in this ID package.

    The generator is a lagged Fibonacci method with 55-element internal state.

    Parameters
    ----------
    seed : int, sequence, 'default', optional
        If 'default', the random seed is reset to a default value.

        If `seed` is a sequence containing 55 floating-point numbers
        in range [0,1], these are used to set the internal state of
        the generator.

        If the value is an integer, the internal state is obtained
        from `numpy.random.RandomState` (MT19937) with the integer
        used as the initial seed.

        If `seed` is omitted (None), ``numpy.random.rand`` is used to
        initialize the generator.

    """
    # For details, see :func:`_backend.id_srand`, :func:`_backend.id_srandi`,
    # and :func:`_backend.id_srando`.

    if isinstance(seed, str) and seed == 'default':  # 检查是否使用默认种子
        _backend.id_srando()  # 使用默认种子初始化随机数生成器
    elif hasattr(seed, '__len__'):  # 检查种子是否为长度可迭代对象
        state = np.asfortranarray(seed, dtype=float)  # 将种子转换为 Fortran 风格的数组
        if state.shape != (55,):  # 检查数组形状是否为 (55,)
            raise ValueError("invalid input size")  # 抛出值错误异常
        elif state.min() < 0 or state.max() > 1:  # 检查数组值是否在 [0, 1] 范围内
            raise ValueError("values not in range [0,1]")  # 抛出值错误异常
        _backend.id_srandi(state)  # 使用给定的种子初始化随机数生成器
    elif seed is None:  # 如果种子未指定
        _backend.id_srandi(np.random.rand(55))  # 使用随机生成的种子初始化随机数生成器
    else:  # 如果种子是整数
        rnd = np.random.RandomState(seed)  # 创建一个随机状态生成器对象
        _backend.id_srandi(rnd.rand(55))  # 使用生成的种子初始化随机数生成器


def rand(*shape):
    """
    Generate standard uniform pseudorandom numbers via a very efficient lagged
    Fibonacci method.

    This routine is used for all random number generation in this package and
    can affect ID and SVD results.

    Parameters
    ----------
    *shape
        Shape of output array

    """
    # For details, see :func:`_backend.id_srand`, and :func:`_backend.id_srando`.
    # 使用 numpy.prod 计算 shape 中所有元素的乘积，然后使用 _backend.id_srand 进行伪随机数种子初始化
    # 将结果 reshape 成指定的 shape
    return _backend.id_srand(np.prod(shape)).reshape(shape)
def interp_decomp(A, eps_or_k, rand=True):
    """
    Compute ID of a matrix.

    An ID of a matrix `A` is a factorization defined by a rank `k`, a column
    index array `idx`, and interpolation coefficients `proj` such that::

        numpy.dot(A[:,idx[:k]], proj) = A[:,idx[k:]]

    The original matrix can then be reconstructed as::

        numpy.hstack([A[:,idx[:k]],
                      numpy.dot(A[:,idx[:k]], proj)]
                    )[:,numpy.argsort(idx)]

    or via the routine :func:`reconstruct_matrix_from_id`. This can
    equivalently be written as::

        numpy.dot(A[:,idx[:k]],
                  numpy.hstack([numpy.eye(k), proj])
                  )[:,np.argsort(idx)]

    in terms of the skeleton and interpolation matrices::

        B = A[:,idx[:k]]

    and::

        P = numpy.hstack([numpy.eye(k), proj])[:,np.argsort(idx)]

    respectively. See also :func:`reconstruct_interp_matrix` and
    :func:`reconstruct_skel_matrix`.

    The ID can be computed to any relative precision or rank (depending on the
    value of `eps_or_k`). If a precision is specified (`eps_or_k < 1`), then
    this function has the output signature::

        k, idx, proj = interp_decomp(A, eps_or_k)

    Otherwise, if a rank is specified (`eps_or_k >= 1`), then the output
    signature is::

        idx, proj = interp_decomp(A, eps_or_k)

    ..  This function automatically detects the form of the input parameters
        and passes them to the appropriate backend. For details, see
        :func:`_backend.iddp_id`, :func:`_backend.iddp_aid`,
        :func:`_backend.iddp_rid`, :func:`_backend.iddr_id`,
        :func:`_backend.iddr_aid`, :func:`_backend.iddr_rid`,
        :func:`_backend.idzp_id`, :func:`_backend.idzp_aid`,
        :func:`_backend.idzp_rid`, :func:`_backend.idzr_id`,
        :func:`_backend.idzr_aid`, and :func:`_backend.idzr_rid`.

    Parameters
    ----------
    A : :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator` with `rmatvec`
        Matrix to be factored
    eps_or_k : float or int
        Relative error (if ``eps_or_k < 1``) or rank (if ``eps_or_k >= 1``) of
        approximation.
    rand : bool, optional
        Whether to use random sampling if `A` is of type :class:`numpy.ndarray`
        (randomized algorithms are always used if `A` is of type
        :class:`scipy.sparse.linalg.LinearOperator`).

    Returns
    -------
    k : int
        Rank required to achieve specified relative precision if
        ``eps_or_k < 1``.
    idx : :class:`numpy.ndarray`
        Column index array.
    proj : :class:`numpy.ndarray`
        Interpolation coefficients.
    """  # numpy/numpydoc#87  # noqa: E501

    # 导入必要的模块或函数
    from scipy.sparse.linalg import LinearOperator

    # 判断矩阵 A 是否为实数类型
    real = _is_real(A)
    # 如果输入的 A 是 numpy 数组
    if isinstance(A, np.ndarray):
        # 如果 eps_or_k 小于 1，则将其作为 eps 处理
        if eps_or_k < 1:
            eps = eps_or_k
            # 如果需要随机化（rand=True），且需要使用实数（real=True）
            if rand:
                if real:
                    # 调用后端函数 iddp_aid 进行处理
                    k, idx, proj = _backend.iddp_aid(eps, A)
                else:
                    # 如果是虚数且运行在 32 位环境下，抛出 _32BIT_ERROR 异常
                    if _IS_32BIT:
                        raise _32BIT_ERROR
                    # 否则调用 idzp_aid 处理
                    k, idx, proj = _backend.idzp_aid(eps, A)
            else:
                # 如果不需要随机化，但需要使用实数
                if real:
                    # 调用 iddp_id 处理
                    k, idx, proj = _backend.iddp_id(eps, A)
                else:
                    # 否则调用 idzp_id 处理
                    k, idx, proj = _backend.idzp_id(eps, A)
            # 返回 k、idx 减一（Python 索引从零开始）、proj
            return k, idx - 1, proj
        else:
            # 否则，将 eps_or_k 强制转换为整数 k
            k = int(eps_or_k)
            # 如果需要随机化
            if rand:
                # 如果需要使用实数
                if real:
                    # 调用 iddr_aid 处理
                    idx, proj = _backend.iddr_aid(A, k)
                else:
                    # 如果是虚数且运行在 32 位环境下，抛出 _32BIT_ERROR 异常
                    if _IS_32BIT:
                        raise _32BIT_ERROR
                    # 否则调用 idzr_aid 处理
                    idx, proj = _backend.idzr_aid(A, k)
            else:
                # 如果不需要随机化
                if real:
                    # 调用 iddr_id 处理
                    idx, proj = _backend.iddr_id(A, k)
                else:
                    # 否则调用 idzr_id 处理
                    idx, proj = _backend.idzr_id(A, k)
            # 返回 idx 减一（Python 索引从零开始）、proj
            return idx - 1, proj
    # 如果输入的 A 是 LinearOperator 对象
    elif isinstance(A, LinearOperator):
        # 获取 LinearOperator 对象 A 的形状 m、n
        m, n = A.shape
        # 获取 A 的右向乘方法（rmatvec）
        matveca = A.rmatvec
        # 如果 eps_or_k 小于 1，则将其作为 eps 处理
        if eps_or_k < 1:
            eps = eps_or_k
            # 如果需要使用实数
            if real:
                # 调用 iddp_rid 处理
                k, idx, proj = _backend.iddp_rid(eps, m, n, matveca)
            else:
                # 如果是虚数且运行在 32 位环境下，抛出 _32BIT_ERROR 异常
                if _IS_32BIT:
                    raise _32BIT_ERROR
                # 否则调用 idzp_rid 处理
                k, idx, proj = _backend.idzp_rid(eps, m, n, matveca)
            # 返回 k、idx 减一（Python 索引从零开始）、proj
            return k, idx - 1, proj
        else:
            # 否则，将 eps_or_k 强制转换为整数 k
            k = int(eps_or_k)
            # 如果需要使用实数
            if real:
                # 调用 iddr_rid 处理
                idx, proj = _backend.iddr_rid(m, n, matveca, k)
            else:
                # 如果是虚数且运行在 32 位环境下，抛出 _32BIT_ERROR 异常
                if _IS_32BIT:
                    raise _32BIT_ERROR
                # 否则调用 idzr_rid 处理
                idx, proj = _backend.idzr_rid(m, n, matveca, k)
            # 返回 idx 减一（Python 索引从零开始）、proj
            return idx - 1, proj
    else:
        # 如果输入的 A 类型既不是 numpy 数组也不是 LinearOperator 对象，抛出 _TYPE_ERROR 异常
        raise _TYPE_ERROR
def reconstruct_matrix_from_id(B, idx, proj):
    """
    Reconstruct matrix from its ID.

    A matrix `A` with skeleton matrix `B` and ID indices and coefficients `idx`
    and `proj`, respectively, can be reconstructed as::

        numpy.hstack([B, numpy.dot(B, proj)])[:,numpy.argsort(idx)]

    See also :func:`reconstruct_interp_matrix` and
    :func:`reconstruct_skel_matrix`.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`_backend.idd_reconid` and
        :func:`_backend.idz_reconid`.

    Parameters
    ----------
    B : :class:`numpy.ndarray`
        Skeleton matrix.
        输入参数 B：骨架矩阵 B。

    idx : :class:`numpy.ndarray`
        Column index array.
        输入参数 idx：列索引数组。

    proj : :class:`numpy.ndarray`
        Interpolation coefficients.
        输入参数 proj：插值系数。

    Returns
    -------
    :class:`numpy.ndarray`
        Reconstructed matrix.
        返回值：重建后的矩阵。
    """
    if _is_real(B):
        return _backend.idd_reconid(B, idx + 1, proj)
    else:
        return _backend.idz_reconid(B, idx + 1, proj)


def reconstruct_interp_matrix(idx, proj):
    """
    Reconstruct interpolation matrix from ID.

    The interpolation matrix can be reconstructed from the ID indices and
    coefficients `idx` and `proj`, respectively, as::

        P = numpy.hstack([numpy.eye(proj.shape[0]), proj])[:,numpy.argsort(idx)]

    The original matrix can then be reconstructed from its skeleton matrix `B`
    via::

        numpy.dot(B, P)

    See also :func:`reconstruct_matrix_from_id` and
    :func:`reconstruct_skel_matrix`.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`_backend.idd_reconint` and
        :func:`_backend.idz_reconint`.

    Parameters
    ----------
    idx : :class:`numpy.ndarray`
        Column index array.
        输入参数 idx：列索引数组。

    proj : :class:`numpy.ndarray`
        Interpolation coefficients.
        输入参数 proj：插值系数。

    Returns
    -------
    :class:`numpy.ndarray`
        Interpolation matrix.
        返回值：插值矩阵。
    """
    if _is_real(proj):
        return _backend.idd_reconint(idx + 1, proj)
    else:
        return _backend.idz_reconint(idx + 1, proj)


def reconstruct_skel_matrix(A, k, idx):
    """
    Reconstruct skeleton matrix from ID.

    The skeleton matrix can be reconstructed from the original matrix `A` and its
    ID rank and indices `k` and `idx`, respectively, as::

        B = A[:,idx[:k]]

    The original matrix can then be reconstructed via::

        numpy.hstack([B, numpy.dot(B, proj)])[:,numpy.argsort(idx)]

    See also :func:`reconstruct_matrix_from_id` and
    :func:`reconstruct_interp_matrix`.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`_backend.idd_copycols` and
        :func:`_backend.idz_copycols`.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Original matrix.
        输入参数 A：原始矩阵。

    k : int
        Rank of ID.
        输入参数 k：ID 的秩。

    idx : :class:`numpy.ndarray`
        Column index array.
        输入参数 idx：列索引数组。

    Returns
    -------
    :class:`numpy.ndarray`
        Skeleton matrix.
        返回值：骨架矩阵。
    """
    -------
    :class:`numpy.ndarray`
        Skeleton matrix.
    """
    # 如果 A 是实数类型的数组，则调用 _backend.idd_copycols 函数进行列复制操作
    if _is_real(A):
        return _backend.idd_copycols(A, k, idx + 1)
    # 如果 A 不是实数类型的数组，则调用 _backend.idz_copycols 函数进行列复制操作
    else:
        return _backend.idz_copycols(A, k, idx + 1)
def id_to_svd(B, idx, proj):
    """
    Convert ID to SVD.

    The SVD reconstruction of a matrix with skeleton matrix `B` and ID indices and
    coefficients `idx` and `proj`, respectively, is::

        U, S, V = id_to_svd(B, idx, proj)
        A = numpy.dot(U, numpy.dot(numpy.diag(S), V.conj().T))

    See also :func:`svd`.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`_backend.idd_id2svd` and
        :func:`_backend.idz_id2svd`.

    Parameters
    ----------
    B : :class:`numpy.ndarray`
        Skeleton matrix.
    idx : :class:`numpy.ndarray`
        Column index array.
    proj : :class:`numpy.ndarray`
        Interpolation coefficients.

    Returns
    -------
    U : :class:`numpy.ndarray`
        Left singular vectors.
    S : :class:`numpy.ndarray`
        Singular values.
    V : :class:`numpy.ndarray`
        Right singular vectors.
    """
    # 根据输入矩阵类型选择相应的后端实现进行 SVD 分解
    if _is_real(B):
        # 如果输入矩阵为实数类型，调用实数类型的 SVD 分解函数
        U, V, S = _backend.idd_id2svd(B, idx + 1, proj)
    else:
        # 如果输入矩阵为复数类型，调用复数类型的 SVD 分解函数
        U, V, S = _backend.idz_id2svd(B, idx + 1, proj)
    # 返回左奇异向量 U，奇异值 S 和右奇异向量 V
    return U, S, V


def estimate_spectral_norm(A, its=20):
    """
    Estimate spectral norm of a matrix by the randomized power method.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`_backend.idd_snorm` and
        :func:`_backend.idz_snorm`.

    Parameters
    ----------
    A : :class:`scipy.sparse.linalg.LinearOperator`
        Matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with the
        `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
    its : int, optional
        Number of power method iterations.

    Returns
    -------
    float
        Spectral norm estimate.
    """
    # 将输入矩阵 A 转换为线性操作器
    from scipy.sparse.linalg import aslinearoperator
    A = aslinearoperator(A)
    m, n = A.shape
    # 定义用于应用矩阵和其伴随的函数
    def matvec(x):
        return A.matvec(x)
    def matveca(x):
        return A.rmatvec(x)
    # 根据输入矩阵类型选择相应的后端实现进行谱范数估计
    if _is_real(A):
        return _backend.idd_snorm(m, n, matveca, matvec, its=its)
    else:
        return _backend.idz_snorm(m, n, matveca, matvec, its=its)


def estimate_spectral_norm_diff(A, B, its=20):
    """
    Estimate spectral norm of the difference of two matrices by the randomized
    power method.

    ..  This function automatically detects the matrix data type and calls the
        appropriate backend. For details, see :func:`_backend.idd_diffsnorm` and
        :func:`_backend.idz_diffsnorm`.

    Parameters
    ----------
    A : :class:`scipy.sparse.linalg.LinearOperator`
        First matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with the
        `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
    B : :class:`scipy.sparse.linalg.LinearOperator`
        Second matrix given as a :class:`scipy.sparse.linalg.LinearOperator` with
        the `matvec` and `rmatvec` methods (to apply the matrix and its adjoint).
    """
    # 根据输入矩阵类型选择相应的后端实现进行差值矩阵谱范数估计
    if _is_real(A):
        return _backend.idd_diffsnorm(A.shape[0], A.shape[1], A.rmatvec, A.matvec, B.rmatvec, B.matvec, its=its)
    else:
        return _backend.idz_diffsnorm(A.shape[0], A.shape[1], A.rmatvec, A.matvec, B.rmatvec, B.matvec, its=its)
    its : int, optional
        Number of power method iterations.  # 参数 its：幂法迭代的次数

    Returns
    -------
    float
        Spectral norm estimate of matrix difference.  # 返回一个浮点数，表示矩阵差的谱范数估计
    """
    from scipy.sparse.linalg import aslinearoperator
    A = aslinearoperator(A)  # 将 A 转换为线性操作符
    B = aslinearoperator(B)  # 将 B 转换为线性操作符
    m, n = A.shape  # 获取 A 的行数 m 和列数 n

    def matvec1(x):
        return A.matvec(x)  # 定义 A 的矩阵向量乘积操作

    def matveca1(x):
        return A.rmatvec(x)  # 定义 A 的共轭转置矩阵向量乘积操作

    def matvec2(x):
        return B.matvec(x)  # 定义 B 的矩阵向量乘积操作

    def matveca2(x):
        return B.rmatvec(x)  # 定义 B 的共轭转置矩阵向量乘积操作

    if _is_real(A):
        return _backend.idd_diffsnorm(
            m, n, matveca1, matveca2, matvec1, matvec2, its=its)  # 如果 A 是实数类型，使用实数版本的差异谱范数计算
    else:
        return _backend.idz_diffsnorm(
            m, n, matveca1, matveca2, matvec1, matvec2, its=its)  # 如果 A 是复数类型，使用复数版本的差异谱范数计算
# 导入必要的库函数
def svd(A, eps_or_k, rand=True):
    """
    Compute SVD of a matrix via an ID.

    An SVD of a matrix `A` is a factorization::

        A = numpy.dot(U, numpy.dot(numpy.diag(S), V.conj().T))

    where `U` and `V` have orthonormal columns and `S` is nonnegative.

    The SVD can be computed to any relative precision or rank (depending on the
    value of `eps_or_k`).

    See also :func:`interp_decomp` and :func:`id_to_svd`.

    ..  This function automatically detects the form of the input parameters and
        passes them to the appropriate backend. For details, see
        :func:`_backend.iddp_svd`, :func:`_backend.iddp_asvd`,
        :func:`_backend.iddp_rsvd`, :func:`_backend.iddr_svd`,
        :func:`_backend.iddr_asvd`, :func:`_backend.iddr_rsvd`,
        :func:`_backend.idzp_svd`, :func:`_backend.idzp_asvd`,
        :func:`_backend.idzp_rsvd`, :func:`_backend.idzr_svd`,
        :func:`_backend.idzr_asvd`, and :func:`_backend.idzr_rsvd`.

    Parameters
    ----------
    A : :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator`
        Matrix to be factored, given as either a :class:`numpy.ndarray` or a
        :class:`scipy.sparse.linalg.LinearOperator` with the `matvec` and
        `rmatvec` methods (to apply the matrix and its adjoint).
    eps_or_k : float or int
        Relative error (if ``eps_or_k < 1``) or rank (if ``eps_or_k >= 1``) of
        approximation.
    rand : bool, optional
        Whether to use random sampling if `A` is of type :class:`numpy.ndarray`
        (randomized algorithms are always used if `A` is of type
        :class:`scipy.sparse.linalg.LinearOperator`).

    Returns
    -------
    U : :class:`numpy.ndarray`
        Left singular vectors.
    S : :class:`numpy.ndarray`
        Singular values.
    V : :class:`numpy.ndarray`
        Right singular vectors.
    """
    # 从 scipy.sparse.linalg 模块中导入 LinearOperator 类
    from scipy.sparse.linalg import LinearOperator

    # 调用 _is_real 函数，判断矩阵 A 是否为实数类型
    real = _is_real(A)
    # 检查输入的矩阵 A 是否为 numpy 数组
    if isinstance(A, np.ndarray):
        # 如果 eps_or_k 小于 1，则将其作为 eps 参数处理
        if eps_or_k < 1:
            eps = eps_or_k
            # 如果 rand 参数为 True
            if rand:
                # 如果 real 参数为 True
                if real:
                    # 调用特定后端函数进行基于增量双分解的奇异值分解
                    U, V, S = _backend.iddp_asvd(eps, A)
                else:
                    # 如果系统为 32 位，则引发错误
                    if _IS_32BIT:
                        raise _32BIT_ERROR
                    # 否则调用特定后端函数进行增量双分解的零平面奇异值分解
                    U, V, S = _backend.idzp_asvd(eps, A)
            else:
                # 如果 real 参数为 True
                if real:
                    # 调用特定后端函数进行基于增量双分解的奇异值分解
                    U, V, S = _backend.iddp_svd(eps, A)
                else:
                    # 调用特定后端函数进行增量零平面奇异值分解
                    U, V, S = _backend.idzp_svd(eps, A)
        else:
            # 将 eps_or_k 强制转换为整数 k
            k = int(eps_or_k)
            # 如果 k 超过矩阵 A 的最小维度
            if k > min(A.shape):
                # 抛出值错误，提示近似秩超过最小维度
                raise ValueError(f"Approximation rank {k} exceeds min(A.shape) = "
                                 f" {min(A.shape)} ")
            # 如果 rand 参数为 True
            if rand:
                # 如果 real 参数为 True
                if real:
                    # 调用特定后端函数进行基于增量随机化双分解的奇异值分解
                    U, V, S = _backend.iddr_asvd(A, k)
                else:
                    # 如果系统为 32 位，则引发错误
                    if _IS_32BIT:
                        raise _32BIT_ERROR
                    # 否则调用特定后端函数进行增量零平面随机化奇异值分解
                    U, V, S = _backend.idzr_asvd(A, k)
            else:
                # 如果 real 参数为 True
                if real:
                    # 调用特定后端函数进行基于增量双分解的奇异值分解
                    U, V, S = _backend.iddr_svd(A, k)
                else:
                    # 调用特定后端函数进行增量零平面奇异值分解
                    U, V, S = _backend.idzr_svd(A, k)
    # 如果 A 是线性操作符（LinearOperator）的实例
    elif isinstance(A, LinearOperator):
        # 获取线性操作符 A 的形状
        m, n = A.shape
        # 定义向量乘法函数 matvec
        def matvec(x):
            return A.matvec(x)
        # 定义右乘转置向量乘法函数 matveca
        def matveca(x):
            return A.rmatvec(x)
        # 如果 eps_or_k 小于 1，则将其作为 eps 参数处理
        if eps_or_k < 1:
            eps = eps_or_k
            # 如果 real 参数为 True
            if real:
                # 调用特定后端函数进行基于增量随机化随机投影奇异值分解
                U, V, S = _backend.iddp_rsvd(eps, m, n, matveca, matvec)
            else:
                # 如果系统为 32 位，则引发错误
                if _IS_32BIT:
                    raise _32BIT_ERROR
                # 否则调用特定后端函数进行增量零平面随机投影奇异值分解
                U, V, S = _backend.idzp_rsvd(eps, m, n, matveca, matvec)
        else:
            # 将 eps_or_k 强制转换为整数 k
            k = int(eps_or_k)
            # 如果 real 参数为 True
            if real:
                # 调用特定后端函数进行基于增量双分解的随机投影奇异值分解
                U, V, S = _backend.iddr_rsvd(m, n, matveca, matvec, k)
            else:
                # 如果系统为 32 位，则引发错误
                if _IS_32BIT:
                    raise _32BIT_ERROR
                # 否则调用特定后端函数进行增量零平面随机投影奇异值分解
                U, V, S = _backend.idzr_rsvd(m, n, matveca, matvec, k)
    else:
        # 如果 A 的类型不是预期的 np.ndarray 或 LinearOperator，则引发类型错误
        raise _TYPE_ERROR
    # 返回计算得到的 U, S, V
    return U, S, V
def estimate_rank(A, eps):
    """
    Estimate matrix rank to a specified relative precision using randomized
    methods.

    The matrix `A` can be given as either a :class:`numpy.ndarray` or a
    :class:`scipy.sparse.linalg.LinearOperator`, with different algorithms used
    for each case. If `A` is of type :class:`numpy.ndarray`, then the output
    rank is typically about 8 higher than the actual numerical rank.

    ..  This function automatically detects the form of the input parameters and
        passes them to the appropriate backend. For details,
        see :func:`_backend.idd_estrank`, :func:`_backend.idd_findrank`,
        :func:`_backend.idz_estrank`, and :func:`_backend.idz_findrank`.

    Parameters
    ----------
    A : :class:`numpy.ndarray` or :class:`scipy.sparse.linalg.LinearOperator`
        Matrix whose rank is to be estimated, given as either a
        :class:`numpy.ndarray` or a :class:`scipy.sparse.linalg.LinearOperator`
        with the `rmatvec` method (to apply the matrix adjoint).
    eps : float
        Relative error for numerical rank definition.

    Returns
    -------
    int
        Estimated matrix rank.
    """
    from scipy.sparse.linalg import LinearOperator  # 导入线性运算符类

    real = _is_real(A)  # 判断矩阵是否为实数类型

    if isinstance(A, np.ndarray):  # 如果 A 是 numpy 数组
        if real:
            rank = _backend.idd_estrank(eps, A)  # 使用实数算法估计矩阵的秩
        else:
            rank = _backend.idz_estrank(eps, A)  # 使用复数算法估计矩阵的秩
        if rank == 0:
            # 对于几乎完全秩满的情况，返回最小维度作为估计的秩
            rank = min(A.shape)
        return rank  # 返回估计的矩阵秩
    elif isinstance(A, LinearOperator):  # 如果 A 是线性运算符
        m, n = A.shape  # 获取线性运算符的形状信息
        matveca = A.rmatvec  # 获取线性运算符的右向特征向量方法
        if real:
            return _backend.idd_findrank(eps, m, n, matveca)  # 使用实数算法查找秩
        else:
            return _backend.idz_findrank(eps, m, n, matveca)  # 使用复数算法查找秩
    else:
        raise _TYPE_ERROR  # 抛出类型错误异常，说明无法处理的类型
```