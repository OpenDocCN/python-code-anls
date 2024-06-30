# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_eigen\_svds.py`

```
import math
import numpy as np

from .arpack import _arpack  # type: ignore[attr-defined]
from . import eigsh

from scipy._lib._util import check_random_state
from scipy.sparse.linalg._interface import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.lobpcg import lobpcg  # type: ignore[no-redef]
from scipy.sparse.linalg._svdp import _svdp
from scipy.linalg import svd

# 定义 `arpack_int` 为 `_arpack` 模块中的 `timing.nbx.dtype`
arpack_int = _arpack.timing.nbx.dtype
# 定义 `__all__` 列表，指定了 `svds` 作为导出模块的公共接口
__all__ = ['svds']

# 定义函数 `_herm`，返回矩阵的共轭转置
def _herm(x):
    return x.T.conj()

# 定义函数 `_iv`，该函数处理求解特征值问题的输入验证和标准化
def _iv(A, k, ncv, tol, which, v0, maxiter,
        return_singular, solver, random_state):

    # 对 `solver` 参数进行输入验证和标准化
    # 由于其他参数需要用到 `solver`，因此此处先验证 `solver`
    solver = str(solver).lower()
    solvers = {"arpack", "lobpcg", "propack"}
    if solver not in solvers:
        raise ValueError(f"solver must be one of {solvers}.")

    # 对 `A` 参数进行输入验证和标准化，将其转换为线性操作符
    A = aslinearoperator(A)  # 这里包含了一些输入验证
    # `A` 的数据类型必须是浮点数或复数类型
    if not (np.issubdtype(A.dtype, np.complexfloating)
            or np.issubdtype(A.dtype, np.floating)):
        message = "`A` must be of floating or complex floating data type."
        raise ValueError(message)
    # `A` 的形状必须非空
    if math.prod(A.shape) == 0:
        message = "`A` must not be empty."
        raise ValueError(message)

    # 对 `k` 参数进行输入验证和标准化
    # 如果使用 `propack` 求解器，`k` 的最大值为 `min(A.shape) - 1`
    kmax = min(A.shape) if solver == 'propack' else min(A.shape) - 1
    if int(k) != k or not (0 < k <= kmax):
        message = "`k` must be an integer satisfying `0 < k < min(A.shape)`."
        raise ValueError(message)
    k = int(k)

    # 对 `ncv` 参数进行输入验证和标准化
    if solver == "arpack" and ncv is not None:
        if int(ncv) != ncv or not (k < ncv < min(A.shape)):
            message = ("`ncv` must be an integer satisfying "
                       "`k < ncv < min(A.shape)`.")
            raise ValueError(message)
        ncv = int(ncv)

    # 对 `tol` 参数进行输入验证和标准化
    if tol < 0 or not np.isfinite(tol):
        message = "`tol` must be a non-negative floating point value."
        raise ValueError(message)
    tol = float(tol)

    # 对 `which` 参数进行输入验证和标准化
    which = str(which).upper()
    whichs = {'LM', 'SM'}
    if which not in whichs:
        raise ValueError(f"`which` must be in {whichs}.")

    # 对 `v0` 参数进行输入验证和标准化
    if v0 is not None:
        v0 = np.atleast_1d(v0)
        # `v0` 的数据类型必须是浮点数或复数类型
        if not (np.issubdtype(v0.dtype, np.complexfloating)
                or np.issubdtype(v0.dtype, np.floating)):
            message = ("`v0` must be of floating or complex floating "
                       "data type.")
            raise ValueError(message)
        # `v0` 的形状必须与 `propack` 求解器或者 `min(A.shape)` 匹配
        shape = (A.shape[0],) if solver == 'propack' else (min(A.shape),)
        if v0.shape != shape:
            message = f"`v0` must have shape {shape}."
            raise ValueError(message)

    # 对 `maxiter` 参数进行输入验证和标准化
    # 如果 `maxiter` 不为 None 并且不是正整数或小于等于零，则抛出数值错误异常
    if maxiter is not None and (int(maxiter) != maxiter or maxiter <= 0):
        message = "`maxiter` must be a positive integer."
        raise ValueError(message)

    # 将 `maxiter` 转换为整数，如果 `maxiter` 为 None，则保持为 None
    maxiter = int(maxiter) if maxiter is not None else maxiter

    # 对 `return_singular_vectors` 进行输入验证和标准化
    # 对此不进行灵活处理；过于复杂且收益微乎其微
    rs_options = {True, False, "vh", "u"}
    # 如果 `return_singular_vectors` 不在预定义的选项集合 `rs_options` 中，则抛出数值错误异常
    if return_singular not in rs_options:
        raise ValueError(f"`return_singular_vectors` must be in {rs_options}.")

    # 检查并返回随机状态对象
    random_state = check_random_state(random_state)

    # 返回元组 (A, k, ncv, tol, which, v0, maxiter, return_singular, solver, random_state)
    return (A, k, ncv, tol, which, v0, maxiter,
            return_singular, solver, random_state)
# 执行部分奇异值分解（SVD）的函数，针对稀疏矩阵进行计算

def svds(A, k=6, ncv=None, tol=0, which='LM', v0=None,
         maxiter=None, return_singular_vectors=True,
         solver='arpack', random_state=None, options=None):
    """
    Partial singular value decomposition of a sparse matrix.

    Compute the largest or smallest `k` singular values and corresponding
    singular vectors of a sparse matrix `A`. The order in which the singular
    values are returned is not guaranteed.

    In the descriptions below, let ``M, N = A.shape``.

    Parameters
    ----------
    A : ndarray, sparse matrix, or LinearOperator
        Matrix to decompose of a floating point numeric dtype.
    k : int, default: 6
        Number of singular values and singular vectors to compute.
        Must satisfy ``1 <= k <= kmax``, where ``kmax=min(M, N)`` for
        ``solver='propack'`` and ``kmax=min(M, N) - 1`` otherwise.
    ncv : int, optional
        When ``solver='arpack'``, this is the number of Lanczos vectors
        generated. See :ref:`'arpack' <sparse.linalg.svds-arpack>` for details.
        When ``solver='lobpcg'`` or ``solver='propack'``, this parameter is
        ignored.
    tol : float, optional
        Tolerance for singular values. Zero (default) means machine precision.
    which : {'LM', 'SM'}
        Which `k` singular values to find: either the largest magnitude ('LM')
        or smallest magnitude ('SM') singular values.
    v0 : ndarray, optional
        The starting vector for iteration; see method-specific
        documentation (:ref:`'arpack' <sparse.linalg.svds-arpack>`,
        :ref:`'lobpcg' <sparse.linalg.svds-lobpcg>`), or
        :ref:`'propack' <sparse.linalg.svds-propack>` for details.
    maxiter : int, optional
        Maximum number of iterations; see method-specific
        documentation (:ref:`'arpack' <sparse.linalg.svds-arpack>`,
        :ref:`'lobpcg' <sparse.linalg.svds-lobpcg>`), or
        :ref:`'propack' <sparse.linalg.svds-propack>` for details.
    return_singular_vectors : {True, False, "u", "vh"}
        Singular values are always computed and returned; this parameter
        controls the computation and return of singular vectors.

        - ``True``: return singular vectors.
        - ``False``: do not return singular vectors.
        - ``"u"``: if ``M <= N``, compute only the left singular vectors and
          return ``None`` for the right singular vectors. Otherwise, compute
          all singular vectors.
        - ``"vh"``: if ``M > N``, compute only the right singular vectors and
          return ``None`` for the left singular vectors. Otherwise, compute
          all singular vectors.

        If ``solver='propack'``, the option is respected regardless of the
        matrix shape.
    """

    # 实现对稀疏矩阵 `A` 的部分奇异值分解
    # 计算最大或最小的 `k` 个奇异值和相应的奇异向量
    # 返回的奇异值的顺序没有保证
    # 在下面的描述中，设 ``M, N = A.shape``
    
    # 参数详解
    # ----------
    # A : ndarray, sparse matrix, or LinearOperator
    #     要分解的矩阵，浮点数数值数据类型。
    # k : int, default: 6
    #     要计算的奇异值和奇异向量的数量。
    #     必须满足 ``1 <= k <= kmax``，其中 ``kmax=min(M, N)`` 对于 ``solver='propack'``，
    #     否则为 ``kmax=min(M, N) - 1``。
    # ncv : int, optional
    #     当 ``solver='arpack'`` 时，这是生成的 Lanczos 向量的数量。
    #     参见 :ref:`'arpack' <sparse.linalg.svds-arpack>` 获取详细信息。
    #     当 ``solver='lobpcg'`` 或 ``solver='propack'`` 时，此参数被忽略。
    # tol : float, optional
    #     奇异值的容差。零（默认）表示机器精度。
    # which : {'LM', 'SM'}
    #     要查找的 `k` 个奇异值：最大幅值（'LM'）或最小幅值（'SM'）。
    # v0 : ndarray, optional
    #     迭代的起始向量；参见特定方法的文档（:ref:`'arpack' <sparse.linalg.svds-arpack>`，
    #     :ref:`'lobpcg' <sparse.linalg.svds-lobpcg>`），或
    #     :ref:`'propack' <sparse.linalg.svds-propack>` 获取详细信息。
    # maxiter : int, optional
    #     最大迭代次数；参见特定方法的文档（:ref:`'arpack' <sparse.linalg.svds-arpack>`，
    #     :ref:`'lobpcg' <sparse.linalg.svds-lobpcg>`），或
    #     :ref:`'propack' <sparse.linalg.svds-propack>` 获取详细信息。
    # return_singular_vectors : {True, False, "u", "vh"}
    #     总是计算并返回奇异值；此参数控制奇异向量的计算和返回。
    #
    #     - ``True``: 返回奇异向量。
    #     - ``False``: 不返回奇异向量。
    #     - ``"u"``: 如果 ``M <= N``，仅计算左奇异向量，并且对右奇异向量返回 ``None``。
    #                否则，计算所有奇异向量。
    #     - ``"vh"``: 如果 ``M > N``，仅计算右奇异向量，并且对左奇异向量返回 ``None``。
    #                否则，计算所有奇异向量。
    #
    #     如果 ``solver='propack'``，则无论矩阵形状如何，都会尊重此选项。
    solver :  {'arpack', 'propack', 'lobpcg'}, optional
            # 参数 solver: 指定使用的求解器类型，可选的值有 'arpack', 'lobpcg', 'propack'
            The solver used.
            # 使用的求解器类型。
            :ref:`'arpack' <sparse.linalg.svds-arpack>`,
            # 参考链接指向 ARPACK 求解器的文档
            :ref:`'lobpcg' <sparse.linalg.svds-lobpcg>`, and
            # 参考链接指向 LOBPCG 求解器的文档
            :ref:`'propack' <sparse.linalg.svds-propack>` are supported.
            # 参考链接指向 PROPACK 求解器的文档，这些都是支持的选项。
            Default: `'arpack'`.
            # 默认值为 `'arpack'`。
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional
            # 参数 random_state: 用于生成重采样的伪随机数生成器状态
        Pseudorandom number generator state used to generate resamples.
            # 用于生成重采样的伪随机数生成器的状态。
        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
            # 如果 `random_state` 为 None 或者 `np.random`，则使用 `numpy.random.RandomState` 单例。
        If `random_state` is an int, a new ``RandomState`` instance is used,
            # 如果 `random_state` 是整数，则使用一个新的 `RandomState` 实例，
        seeded with `random_state`.
            # 并使用 `random_state` 进行种子初始化。
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used.
            # 如果 `random_state` 已经是一个 `Generator` 或 `RandomState` 实例，则直接使用该实例。
    options : dict, optional
        A dictionary of solver-specific options. No solver-specific options
            # 参数 options: 求解器特定选项的字典，目前未支持任何求解器特定选项，此参数预留以供将来使用。
        are currently supported; this parameter is reserved for future use.
            # 目前未支持任何求解器特定选项；此参数保留供将来使用。

    Returns
    -------
    u : ndarray, shape=(M, k)
        # 返回值 u: 形状为 (M, k) 的 ndarray，包含左奇异向量作为列。
        Unitary matrix having left singular vectors as columns.
        # 单位矩阵，其左奇异向量作为列。

    s : ndarray, shape=(k,)
        # 返回值 s: 形状为 (k,) 的 ndarray，包含奇异值。
        The singular values.
        # 奇异值。

    vh : ndarray, shape=(k, N)
        # 返回值 vh: 形状为 (k, N) 的 ndarray，包含右奇异向量作为行。
        Unitary matrix having right singular vectors as rows.
        # 单位矩阵，其右奇异向量作为行。

    Notes
    -----
    This is a naive implementation using ARPACK or LOBPCG as an eigensolver
            # 这是一个使用 ARPACK 或 LOBPCG 作为特征值求解器的简单实现
    on the matrix ``A.conj().T @ A`` or ``A @ A.conj().T``, depending on
            # 对矩阵 ``A.conj().T @ A`` 或 ``A @ A.conj().T`` 的处理，取决于哪一个尺寸更小，
    which one is smaller size, followed by the Rayleigh-Ritz method
            # 随后使用 Rayleigh-Ritz 方法
    as postprocessing; see
            # 作为后处理；参见
    Using the normal matrix, in Rayleigh-Ritz method, (2022, Nov. 19),
            # 在 Rayleigh-Ritz 方法中使用正规矩阵 (2022年11月19日)
    Wikipedia, https://w.wiki/4zms.

    Alternatively, the PROPACK solver can be called.
            # 或者，可以调用 PROPACK 求解器。

    Choices of the input matrix `A` numeric dtype may be limited.
            # 输入矩阵 `A` 的数值类型选择可能受限。
    Only ``solver="lobpcg"`` supports all floating point dtypes
            # 只有 ``solver="lobpcg"`` 支持所有浮点数类型
    real: 'np.float32', 'np.float64', 'np.longdouble' and
    complex: 'np.complex64', 'np.complex128', 'np.clongdouble'.
            # 实数：'np.float32', 'np.float64', 'np.longdouble' 和
    The ``solver="arpack"`` supports only
            # 复数：'np.complex64', 'np.complex128', 'np.clongdouble'。
    'np.float32', 'np.float64', and 'np.complex128'.
            # ``solver="arpack"`` 仅支持 'np.float32', 'np.float64', 和 'np.complex128'。

    Examples
    --------
    Construct a matrix `A` from singular values and vectors.
            # 构建一个由奇异值和奇异向量组成的矩阵 `A`。

    >>> import numpy as np
    >>> from scipy import sparse, linalg, stats
    >>> from scipy.sparse.linalg import svds, aslinearoperator, LinearOperator

    Construct a dense matrix `A` from singular values and vectors.
            # 从奇异值和奇异向量构建一个稠密矩阵 `A`。

    >>> rng = np.random.default_rng(258265244568965474821194062361901728911)
    >>> orthogonal = stats.ortho_group.rvs(10, random_state=rng)
    >>> s = [1e-3, 1, 2, 3, 4]  # non-zero singular values
            # 生成一个正交矩阵 `orthogonal`，并设定随机种子
    >>> u = orthogonal[:, :5]         # left singular vectors
            # 提取左奇异向量
    >>> vT = orthogonal[:, 5:].T      # right singular vectors
            # 提取右奇异向量的转置
    >>> A = u @ np.diag(s) @ vT
            # 构建矩阵 `A`，使用奇异值和奇异向量

    With only four singular values/vectors, the SVD approximates the original
    matrix.
            # 使用仅有的四个奇异值/奇异向量，SVD 近似于原始矩阵。

    >>> u4, s4, vT4 = svds(A, k=4)
    >>> A4 = u4 @ np.diag(s4) @ vT4
    >>> np.allclose(A4, A, atol=1e-3)
    True
    With all five non-zero singular values/vectors, we can reproduce
    the original matrix more accurately.

    >>> u5, s5, vT5 = svds(A, k=5)
    # 计算矩阵的前 5 个奇异向量和奇异值
    >>> A5 = u5 @ np.diag(s5) @ vT5
    # 重构原始矩阵 A，使用计算得到的奇异向量和奇异值
    >>> np.allclose(A5, A)
    True

    The singular values match the expected singular values.

    >>> np.allclose(s5, s)
    True

    Since the singular values are not close to each other in this example,
    every singular vector matches as expected up to a difference in sign.

    >>> (np.allclose(np.abs(u5), np.abs(u)) and
    ...  np.allclose(np.abs(vT5), np.abs(vT)))
    True

    The singular vectors are also orthogonal.

    >>> (np.allclose(u5.T @ u5, np.eye(5)) and
    ...  np.allclose(vT5 @ vT5.T, np.eye(5)))
    True

    If there are (nearly) multiple singular values, the corresponding
    individual singular vectors may be unstable, but the whole invariant
    subspace containing all such singular vectors is computed accurately
    as can be measured by angles between subspaces via 'subspace_angles'.

    >>> rng = np.random.default_rng(178686584221410808734965903901790843963)
    >>> s = [1, 1 + 1e-6]  # non-zero singular values
    # 定义非零奇异值 s
    >>> u, _ = np.linalg.qr(rng.standard_normal((99, 2)))
    # 生成一个 99x2 的正交矩阵 u
    >>> v, _ = np.linalg.qr(rng.standard_normal((99, 2)))
    # 生成一个 99x2 的正交矩阵 v
    >>> vT = v.T
    # 计算 v 的转置
    >>> A = u @ np.diag(s) @ vT
    # 构造原始矩阵 A，使用生成的 u、s、vT
    >>> A = A.astype(np.float32)
    # 将矩阵 A 转换为单精度浮点数类型
    >>> u2, s2, vT2 = svds(A, k=2, random_state=rng)
    # 计算 A 的前 2 个奇异向量和奇异值
    >>> np.allclose(s2, s)
    True

    The angles between the individual exact and computed singular vectors
    may not be so small. To check use:

    >>> (linalg.subspace_angles(u2[:, :1], u[:, :1]) +
    ...  linalg.subspace_angles(u2[:, 1:], u[:, 1:]))
    array([0.06562513])  # may vary
    >>> (linalg.subspace_angles(vT2[:1, :].T, vT[:1, :].T) +
    ...  linalg.subspace_angles(vT2[1:, :].T, vT[1:, :].T))
    array([0.06562507])  # may vary

    As opposed to the angles between the 2-dimensional invariant subspaces
    that these vectors span, which are small for rights singular vectors

    >>> linalg.subspace_angles(u2, u).sum() < 1e-6
    True

    as well as for left singular vectors.

    >>> linalg.subspace_angles(vT2.T, vT.T).sum() < 1e-6
    True

    The next example follows that of 'sklearn.decomposition.TruncatedSVD'.

    >>> rng = np.random.RandomState(0)
    # 使用随机种子初始化随机数生成器
    >>> X_dense = rng.random(size=(100, 100))
    # 生成一个 100x100 的随机密集矩阵 X_dense
    >>> X_dense[:, 2 * np.arange(50)] = 0
    # 将矩阵 X_dense 中的某些列置零
    >>> X = sparse.csr_matrix(X_dense)
    # 转换为稀疏矩阵 X
    >>> _, singular_values, _ = svds(X, k=5, random_state=rng)
    # 计算稀疏矩阵 X 的前 5 个奇异值和奇异向量
    >>> print(singular_values)
    [ 4.3293...  4.4491...  4.5420...  4.5987... 35.2410...]
    # 打印计算得到的奇异值

    The function can be called without the transpose of the input matrix
    ever explicitly constructed.

    >>> rng = np.random.default_rng(102524723947864966825913730119128190974)
    # 使用随机种子初始化随机数生成器
    >>> G = sparse.rand(8, 9, density=0.5, random_state=rng)
    # 生成一个稀疏随机矩阵 G
    >>> Glo = aslinearoperator(G)
    # 将 G 转换为线性操作符 Glo
    >>> _, singular_values_svds, _ = svds(Glo, k=5, random_state=rng)
    # 计算线性操作符 Glo 的前 5 个奇异值和奇异向量
    >>> _, singular_values_svd, _ = linalg.svd(G.toarray())
    # 对稀疏矩阵 G 的密集表示计算所有奇异值和奇异向量
    >>> np.allclose(singular_values_svds, singular_values_svd[-4::-1])
    True
    
    The code checks if all elements in `singular_values_svds` are close to the elements in `singular_values_svd` reversed starting from the fourth last element.
    
    The most memory efficient scenario is where neither
    the original matrix, nor its transpose, is explicitly constructed.
    Our example computes the smallest singular values and vectors
    of 'LinearOperator' constructed from the numpy function 'np.diff' used
    column-wise to be consistent with 'LinearOperator' operating on columns.
    
    >>> diff0 = lambda a: np.diff(a, axis=0)
    
    Defines a lambda function `diff0` that computes differences along the first axis of an array `a`.
    
    Let us create the matrix from 'diff0' to be used for validation only.
    
    >>> n = 5  # The dimension of the space.
    
    Defines an integer `n` with value 5.
    
    >>> M_from_diff0 = diff0(np.eye(n))
    >>> print(M_from_diff0.astype(int))
    [[-1  1  0  0  0]
     [ 0 -1  1  0  0]
     [ 0  0 -1  1  0]
     [ 0  0  0 -1  1]]
    
    Computes `M_from_diff0` using the `diff0` function on a 5x5 identity matrix and prints it as integers.
    
    The matrix 'M_from_diff0' is bi-diagonal and could be alternatively
    created directly by
    
    >>> M = - np.eye(n - 1, n, dtype=int)
    >>> np.fill_diagonal(M[:,1:], 1)
    >>> np.allclose(M, M_from_diff0)
    True
    
    Constructs matrix `M` as a bi-diagonal matrix and verifies it against `M_from_diff0`.
    
    Its transpose
    
    >>> print(M.T)
    [[-1  0  0  0]
     [ 1 -1  0  0]
     [ 0  1 -1  0]
     [ 0  0  1 -1]
     [ 0  0  0  1]]
    
    Prints the transpose of matrix `M`.
    
    can be viewed as the incidence matrix; see
    Incidence matrix, (2022, Nov. 19), Wikipedia, https://w.wiki/5YXU,
    of a linear graph with 5 vertices and 4 edges. The 5x5 normal matrix
    ``M.T @ M`` thus is
    
    >>> print(M.T @ M)
    [[ 1 -1  0  0  0]
     [-1  2 -1  0  0]
     [ 0 -1  2 -1  0]
     [ 0  0 -1  2 -1]
     [ 0  0  0 -1  1]]
    
    Calculates and prints the result of `M.T @ M`, representing the graph Laplacian.
    
    the graph Laplacian, while the actually used in 'svds' smaller size
    4x4 normal matrix ``M @ M.T``
    
    >>> print(M @ M.T)
    [[ 2 -1  0  0]
     [-1  2 -1  0]
     [ 0 -1  2 -1]
     [ 0  0 -1  2]]
    
    Calculates and prints the result of `M @ M.T`, representing the edge-based Laplacian.
    
    is the so-called edge-based Laplacian; see
    Symmetric Laplacian via the incidence matrix, in Laplacian matrix,
    (2022, Nov. 19), Wikipedia, https://w.wiki/5YXW.
    
    Provides a reference to the edge-based Laplacian in graph theory.
    
    The 'LinearOperator' setup needs the options 'rmatvec' and 'rmatmat'
    of multiplication by the matrix transpose ``M.T``, but we want to be
    matrix-free to save memory, so knowing how ``M.T`` looks like, we
    manually construct the following function to be
    used in ``rmatmat=diff0t``.
    
    >>> def diff0t(a):
    ...     if a.ndim == 1:
    ...         a = a[:,np.newaxis]  # Turn 1D into 2D array
    ...     d = np.zeros((a.shape[0] + 1, a.shape[1]), dtype=a.dtype)
    ...     d[0, :] = - a[0, :]
    ...     d[1:-1, :] = a[0:-1, :] - a[1:, :]
    ...     d[-1, :] = a[-1, :]
    ...     return d
    
    Defines a function `diff0t` that computes the transpose of `diff0` efficiently handling different dimensions of `a`.
    
    We check that our function 'diff0t' for the matrix transpose is valid.
    
    >>> np.allclose(M.T, diff0t(np.eye(n-1)))
    True
    
    Verifies that `diff0t` produces the same result as `M.T` using a 4x4 identity matrix.
    
    Now we setup our matrix-free 'LinearOperator' called 'diff0_func_aslo'
    and for validation the matrix-based 'diff0_matrix_aslo'.
    
    >>> def diff0_func_aslo_def(n):
    ...     return LinearOperator(matvec=diff0,
    ...                           matmat=diff0,
    ...                           rmatvec=diff0t,
    ...                           rmatmat=diff0t,
    """
    args = _iv(A, k, ncv, tol, which, v0, maxiter, return_singular_vectors,
               solver, random_state)
    (A, k, ncv, tol, which, v0, maxiter,
     return_singular_vectors, solver, random_state) = args
    """

    # 调用函数 _iv，获取参数列表并解包赋值给相应的变量
    args = _iv(A, k, ncv, tol, which, v0, maxiter, return_singular_vectors,
               solver, random_state)
    (A, k, ncv, tol, which, v0, maxiter,
     return_singular_vectors, solver, random_state) = args

    # 检查要求的特征值是否为最大值
    largest = (which == 'LM')

    # 获取矩阵 A 的形状
    n, m = A.shape

    # 根据矩阵 A 的形状选择不同的操作方式
    if n >= m:
        # 如果行数大于等于列数，则使用正常的矩阵向量乘法和矩阵乘法
        X_dot = A.matvec
        X_matmat = A.matmat
        XH_dot = A.rmatvec
        XH_mat = A.rmatmat
        transpose = False
    else:
        # 如果行数小于列数，则需要转置操作，同时获取数据类型
        X_dot = A.rmatvec
        X_matmat = A.rmatmat
        XH_dot = A.matvec
        XH_mat = A.matmat
        transpose = True

        dtype = getattr(A, 'dtype', None)
        if dtype is None:
            dtype = A.dot(np.zeros([m, 1])).dtype

    # 定义函数 matvec_XH_X 和 matmat_XH_X，用于计算 XH_X
    def matvec_XH_X(x):
        return XH_dot(X_dot(x))

    def matmat_XH_X(x):
        return XH_mat(X_matmat(x))

    # 创建 LinearOperator 对象 XH_X，用于表示 XH_X 矩阵
    XH_X = LinearOperator(matvec=matvec_XH_X, dtype=A.dtype,
                          matmat=matmat_XH_X,
                          shape=(min(A.shape), min(A.shape)))

    # 如果使用 lobpcg 求解器，进行特征值分解
    if solver == 'lobpcg':

        # 根据 k 的值和初始向量 v0，初始化 X
        if k == 1 and v0 is not None:
            X = np.reshape(v0, (-1, 1))
        else:
            X = random_state.standard_normal(size=(min(A.shape), k))

        # 使用 lobpcg 方法求解特征值问题，返回特征向量 eigvec
        _, eigvec = lobpcg(XH_X, X, tol=tol ** 2, maxiter=maxiter,
                           largest=largest)
    # 如果使用 PROPACK 求解器
    elif solver == 'propack':
        # 确定是否需要返回左奇异向量
        jobu = return_singular_vectors in {True, 'u'}
        # 确定是否需要返回右奇异向量
        jobv = return_singular_vectors in {True, 'vh'}
        # 判断是否为 'SM' 模式
        irl_mode = (which == 'SM')
        # 调用 _svdp 函数进行奇异值分解
        res = _svdp(A, k=k, tol=tol**2, which=which, maxiter=None,
                    compute_u=jobu, compute_v=jobv, irl_mode=irl_mode,
                    kmax=maxiter, v0=v0, random_state=random_state)

        # 解包结果，忽略最后一个输出 bnd
        u, s, vh, _ = res

        # PROPACK 的奇异值顺序似乎是从大到小。`svds` 的输出顺序不保证一致，
        # 但对于 ARPACK 和 LOBPCG，它们实际上是按从小到大排序的，因此需要反转以保持一致性。
        s = s[::-1]  # 反转奇异值向量 s
        u = u[:, ::-1]  # 反转左奇异向量矩阵 u 的列
        vh = vh[::-1]  # 反转右奇异向量矩阵 vh

        # 根据 jobu 和 jobv 确定是否返回左右奇异向量
        u = u if jobu else None
        vh = vh if jobv else None

        if return_singular_vectors:
            return u, s, vh  # 返回左右奇异向量和奇异值
        else:
            return s  # 只返回奇异值

    # 如果使用 ARPACK 求解器或者 solver 为 None
    elif solver == 'arpack' or solver is None:
        # 如果 v0 为 None，则使用随机初始化的 v0
        if v0 is None:
            v0 = random_state.standard_normal(size=(min(A.shape),))
        # 调用 eigsh 函数计算特征值和特征向量
        _, eigvec = eigsh(XH_X, k=k, tol=tol ** 2, maxiter=maxiter,
                          ncv=ncv, which=which, v0=v0)
        # 对特征向量进行 QR 分解，确保其正交性
        eigvec, _ = np.linalg.qr(eigvec)

    # 此时 eigenvectors eigvec 必须是正交的；参见 gh-16712
    # 计算 Av = X @ eigvec
    Av = X_matmat(eigvec)

    # 如果不需要返回奇异向量
    if not return_singular_vectors:
        # 计算 Av 的奇异值，不计算左右奇异向量
        s = svd(Av, compute_uv=False, overwrite_a=True)
        return s[::-1]  # 返回反转后的奇异值向量

    # 计算 Av 的奇异值分解，包括左右奇异向量
    u, s, vh = svd(Av, full_matrices=False, overwrite_a=True)
    u = u[:, ::-1]  # 反转左奇异向量矩阵 u 的列
    s = s[::-1]  # 反转奇异值向量 s
    vh = vh[::-1]  # 反转右奇异向量矩阵 vh

    # 再次确定是否需要返回左右奇异向量
    jobu = return_singular_vectors in {True, 'u'}
    jobv = return_singular_vectors in {True, 'vh'}

    # 如果需要转置
    if transpose:
        # 计算左奇异向量 u_tmp 和右奇异向量 vh
        u_tmp = eigvec @ _herm(vh) if jobu else None
        vh = _herm(u) if jobv else None
        u = u_tmp
    else:
        # 如果不需要左奇异向量 u，则置为 None
        if not jobu:
            u = None
        # 计算右奇异向量 vh
        vh = vh @ _herm(eigvec) if jobv else None

    return u, s, vh  # 返回左奇异向量、奇异值、右奇异向量
```