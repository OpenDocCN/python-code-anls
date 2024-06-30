# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_eigen\_svds_doc.py`

```
# 定义名为 _svds_arpack_doc 的函数，使用 ARPACK 进行稀疏矩阵的部分奇异值分解
def _svds_arpack_doc(A, k=6, ncv=None, tol=0, which='LM', v0=None,
                     maxiter=None, return_singular_vectors=True,
                     solver='arpack', random_state=None):
    """
    Partial singular value decomposition of a sparse matrix using ARPACK.

    Compute the largest or smallest `k` singular values and corresponding
    singular vectors of a sparse matrix `A`. The order in which the singular
    values are returned is not guaranteed.

    In the descriptions below, let ``M, N = A.shape``.

    Parameters
    ----------
    A : sparse matrix or LinearOperator
        Matrix to decompose.
    k : int, optional
        Number of singular values and singular vectors to compute.
        Must satisfy ``1 <= k <= min(M, N) - 1``.
        Default is 6.
    ncv : int, optional
        The number of Lanczos vectors generated.
        The default is ``min(n, max(2*k + 1, 20))``.
        If specified, must satistify ``k + 1 < ncv < min(M, N)``; ``ncv > 2*k``
        is recommended.
    tol : float, optional
        Tolerance for singular values. Zero (default) means machine precision.
    which : {'LM', 'SM'}
        Which `k` singular values to find: either the largest magnitude ('LM')
        or smallest magnitude ('SM') singular values.
    v0 : ndarray, optional
        The starting vector for iteration:
        an (approximate) left singular vector if ``N > M`` and a right singular
        vector otherwise. Must be of length ``min(M, N)``.
        Default: random
    maxiter : int, optional
        Maximum number of Arnoldi update iterations allowed;
        default is ``min(M, N) * 10``.
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

    solver :  {'arpack', 'propack', 'lobpcg'}, optional
            This is the solver-specific documentation for ``solver='arpack'``.
            :ref:`'lobpcg' <sparse.linalg.svds-lobpcg>` and
            :ref:`'propack' <sparse.linalg.svds-propack>`
            are also supported.
    """
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional
    # 参数 random_state 控制伪随机数生成器的状态，用于生成重采样的数据。
    # 可选取值有：None、整数、`numpy.random.Generator`、`numpy.random.RandomState`。
    # - 如果为 None 或 np.random，则使用 numpy.random.RandomState 单例。
    # - 如果为整数，则使用以 random_state 为种子创建的新 RandomState 实例。
    # - 如果已经是 Generator 或 RandomState 实例，则直接使用该实例。

    options : dict, optional
        A dictionary of solver-specific options. No solver-specific options
        are currently supported; this parameter is reserved for future use.
    # 参数 options 是一个字典，用于传递特定求解器的选项。
    # 目前没有支持的特定求解器选项；这个参数是为将来使用而保留的。

    Returns
    -------
    u : ndarray, shape=(M, k)
        Unitary matrix having left singular vectors as columns.
    # 返回一个 ndarray u，形状为 (M, k)，其中每列是左奇异向量组成的单位矩阵。

    s : ndarray, shape=(k,)
        The singular values.
    # 返回一个 ndarray s，形状为 (k,)，包含奇异值。

    vh : ndarray, shape=(k, N)
        Unitary matrix having right singular vectors as rows.
    # 返回一个 ndarray vh，形状为 (k, N)，其中每行是右奇异向量组成的单位矩阵。

    Notes
    -----
    This is a naive implementation using ARPACK as an eigensolver
    on ``A.conj().T @ A`` or ``A @ A.conj().T``, depending on which one is more
    efficient.
    # 这是一个使用 ARPACK 作为特征值求解器的简单实现，应用于 ``A.conj().T @ A`` 或 ``A @ A.conj().T``，
    # 具体取决于哪个更有效率。

    Examples
    --------
    Construct a matrix ``A`` from singular values and vectors.
    # 构建一个由奇异值和奇异向量构成的矩阵 A 的示例。

    >>> import numpy as np
    >>> from scipy.stats import ortho_group
    >>> from scipy.sparse import csc_matrix, diags
    >>> from scipy.sparse.linalg import svds
    >>> rng = np.random.default_rng()
    >>> orthogonal = csc_matrix(ortho_group.rvs(10, random_state=rng))
    >>> s = [0.0001, 0.001, 3, 4, 5]  # singular values
    >>> u = orthogonal[:, :5]         # left singular vectors
    >>> vT = orthogonal[:, 5:].T      # right singular vectors
    >>> A = u @ diags(s) @ vT
    # 使用奇异值和奇异向量构建矩阵 A。

    With only three singular values/vectors, the SVD approximates the original
    matrix.
    # 使用三个奇异值/向量，SVD近似原始矩阵。

    >>> u2, s2, vT2 = svds(A, k=3, solver='arpack')
    >>> A2 = u2 @ np.diag(s2) @ vT2
    >>> np.allclose(A2, A.toarray(), atol=1e-3)
    True
    # 使用 svds 进行 SVD 分解，k=3，求解器为 'arpack'。
    # 检查重建的 A2 是否与原始矩阵 A 的数组形式非常接近。

    With all five singular values/vectors, we can reproduce the original
    matrix.
    # 使用所有五个奇异值/向量，可以完全重建原始矩阵。

    >>> u3, s3, vT3 = svds(A, k=5, solver='arpack')
    >>> A3 = u3 @ np.diag(s3) @ vT3
    >>> np.allclose(A3, A.toarray())
    True
    # 使用 svds 进行 SVD 分解，k=5，求解器为 'arpack'。
    # 检查重建的 A3 是否与原始矩阵 A 的数组形式完全一致。

    The singular values match the expected singular values, and the singular
    vectors are as expected up to a difference in sign.
    # 奇异值与预期的奇异值匹配，奇异向量在符号差异上也符合预期。

    >>> (np.allclose(s3, s) and
    ...  np.allclose(np.abs(u3), np.abs(u.toarray())) and
    ...  np.allclose(np.abs(vT3), np.abs(vT.toarray())))
    True
    # 检查奇异值 s3 是否与预期的 s 接近；
    # 检查绝对值后的奇异向量 u3 是否与 u 的数组形式的绝对值接近；
    # 检查绝对值后的奇异向量 vT3 是否与 vT 的数组形式的绝对值接近。

    The singular vectors are also orthogonal.
    # 奇异向量也是正交的。

    >>> (np.allclose(u3.T @ u3, np.eye(5)) and
    ...  np.allclose(vT3 @ vT3.T, np.eye(5)))
    True
    # 检查 u3 的转置乘以 u3 是否接近单位矩阵；
    # 检查 vT3 乘以 vT3 的转置是否接近单位矩阵。
# 使用 LOBPCG 方法对稀疏矩阵进行部分奇异值分解（SVD）

def _svds_lobpcg_doc(A, k=6, ncv=None, tol=0, which='LM', v0=None,
                     maxiter=None, return_singular_vectors=True,
                     solver='lobpcg', random_state=None):
    """
    Partial singular value decomposition of a sparse matrix using LOBPCG.

    Compute the largest or smallest `k` singular values and corresponding
    singular vectors of a sparse matrix `A`. The order in which the singular
    values are returned is not guaranteed.

    In the descriptions below, let ``M, N = A.shape``.

    Parameters
    ----------
    A : sparse matrix or LinearOperator
        Matrix to decompose.
    k : int, default: 6
        Number of singular values and singular vectors to compute.
        Must satisfy ``1 <= k <= min(M, N) - 1``.
    ncv : int, optional
        Ignored.
    tol : float, optional
        Tolerance for singular values. Zero (default) means machine precision.
    which : {'LM', 'SM'}
        Which `k` singular values to find: either the largest magnitude ('LM')
        or smallest magnitude ('SM') singular values.
    v0 : ndarray, optional
        If `k` is 1, the starting vector for iteration:
        an (approximate) left singular vector if ``N > M`` and a right singular
        vector otherwise. Must be of length ``min(M, N)``.
        Ignored otherwise.
        Default: random
    maxiter : int, default: 20
        Maximum number of iterations.
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

    solver :  {'arpack', 'propack', 'lobpcg'}, optional
        Solver to use for the decomposition. 'lobpcg' is chosen for this function.

        This is the solver-specific documentation for ``solver='lobpcg'``.
        :ref:`'arpack' <sparse.linalg.svds-arpack>` and
        :ref:`'propack' <sparse.linalg.svds-propack>`
        are also supported.
        
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional
        Pseudorandom number generator state used to generate resamples.

        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used.
    """
    options : dict, optional
        一个字典，包含特定求解器的选项。当前未支持任何特定求解器选项；此参数预留以供将来使用。

    Returns
    -------
    u : ndarray, shape=(M, k)
        左奇异向量构成的酉矩阵，列为向量。
    s : ndarray, shape=(k,)
        奇异值。
    vh : ndarray, shape=(k, N)
        右奇异向量构成的酉矩阵，行为向量。

    Notes
    -----
    这是一个使用LOBPCG作为特征求解器，在``A.conj().T @ A``或``A @ A.conj().T``上的朴素实现，具体取决于哪一个更高效。

    Examples
    --------
    构造一个矩阵 ``A``，由奇异值和向量组成。

    >>> import numpy as np
    >>> from scipy.stats import ortho_group
    >>> from scipy.sparse import csc_matrix, diags
    >>> from scipy.sparse.linalg import svds
    >>> rng = np.random.default_rng()
    >>> orthogonal = csc_matrix(ortho_group.rvs(10, random_state=rng))
    >>> s = [0.0001, 0.001, 3, 4, 5]  # 奇异值
    >>> u = orthogonal[:, :5]         # 左奇异向量
    >>> vT = orthogonal[:, 5:].T      # 右奇异向量
    >>> A = u @ diags(s) @ vT

    仅使用三个奇异值/向量，SVD近似原始矩阵。

    >>> u2, s2, vT2 = svds(A, k=3, solver='lobpcg')
    >>> A2 = u2 @ np.diag(s2) @ vT2
    >>> np.allclose(A2, A.toarray(), atol=1e-3)
    True

    使用所有五个奇异值/向量，可以重构原始矩阵。

    >>> u3, s3, vT3 = svds(A, k=5, solver='lobpcg')
    >>> A3 = u3 @ np.diag(s3) @ vT3
    >>> np.allclose(A3, A.toarray())
    True

    奇异值与预期奇异值匹配，奇异向量与预期相符，可能存在符号差异。

    >>> (np.allclose(s3, s) and
    ...  np.allclose(np.abs(u3), np.abs(u.todense())) and
    ...  np.allclose(np.abs(vT3), np.abs(vT.todense())))
    True

    奇异向量也是正交的。

    >>> (np.allclose(u3.T @ u3, np.eye(5)) and
    ...  np.allclose(vT3 @ vT3.T, np.eye(5)))
    True

    """
    pass
# 定义函数 _svds_propack_doc，用于部分奇异值分解（SVD）的稀疏矩阵处理，使用 PROPACK 算法
def _svds_propack_doc(A, k=6, ncv=None, tol=0, which='LM', v0=None,
                      maxiter=None, return_singular_vectors=True,
                      solver='propack', random_state=None):
    """
    Partial singular value decomposition of a sparse matrix using PROPACK.

    Compute the largest or smallest `k` singular values and corresponding
    singular vectors of a sparse matrix `A`. The order in which the singular
    values are returned is not guaranteed.

    In the descriptions below, let ``M, N = A.shape``.

    Parameters
    ----------
    A : sparse matrix or LinearOperator
        Matrix to decompose. If `A` is a ``LinearOperator``
        object, it must define both ``matvec`` and ``rmatvec`` methods.
    k : int, default: 6
        Number of singular values and singular vectors to compute.
        Must satisfy ``1 <= k <= min(M, N)``.
    ncv : int, optional
        Ignored.
    tol : float, optional
        The desired relative accuracy for computed singular values.
        Zero (default) means machine precision.
    which : {'LM', 'SM'}
        Which `k` singular values to find: either the largest magnitude ('LM')
        or smallest magnitude ('SM') singular values. Note that choosing
        ``which='SM'`` will force the ``irl`` option to be set ``True``.
    v0 : ndarray, optional
        Starting vector for iterations: must be of length ``A.shape[0]``.
        If not specified, PROPACK will generate a starting vector.
    maxiter : int, optional
        Maximum number of iterations / maximal dimension of the Krylov
        subspace. Default is ``10 * k``.
    return_singular_vectors : {True, False, "u", "vh"}
        Singular values are always computed and returned; this parameter
        controls the computation and return of singular vectors.

        - ``True``: return singular vectors.
        - ``False``: do not return singular vectors.
        - ``"u"``: compute only the left singular vectors; return ``None`` for
          the right singular vectors.
        - ``"vh"``: compute only the right singular vectors; return ``None``
          for the left singular vectors.

    solver :  {'arpack', 'propack', 'lobpcg'}, optional
            This is the solver-specific documentation for ``solver='propack'``.
            :ref:`'arpack' <sparse.linalg.svds-arpack>` and
            :ref:`'lobpcg' <sparse.linalg.svds-lobpcg>`
            are also supported.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        Pseudorandom number generator state used to generate resamples.

        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used.
    """
    # 实现部分奇异值分解（SVD）的稀疏矩阵处理，使用 PROPACK 算法
    pass
    options : dict, optional
        一个包含特定求解器选项的字典。当前不支持任何特定求解器选项；此参数保留用于将来使用。

    Returns
    -------
    u : ndarray, shape=(M, k)
        具有左奇异向量作为列的酉矩阵。
    s : ndarray, shape=(k,)
        奇异值。
    vh : ndarray, shape=(k, N)
        具有右奇异向量作为行的酉矩阵。

    Notes
    -----
    这是与Fortran库PROPACK的接口 [1]_。
    当前默认情况下，除非寻找最小的奇异值/向量（``which='SM'``），否则禁用IRL模式。

    References
    ----------

    .. [1] Larsen, Rasmus Munk. "PROPACK-Software for large and sparse SVD
       calculations." Available online. URL
       http://sun.stanford.edu/~rmunk/PROPACK (2004): 2008-2009.

    Examples
    --------
    构造一个由奇异值和向量构成的矩阵 ``A``。

    >>> import numpy as np
    >>> from scipy.stats import ortho_group
    >>> from scipy.sparse import csc_matrix, diags
    >>> from scipy.sparse.linalg import svds
    >>> rng = np.random.default_rng()
    >>> orthogonal = csc_matrix(ortho_group.rvs(10, random_state=rng))
    >>> s = [0.0001, 0.001, 3, 4, 5]  # 奇异值
    >>> u = orthogonal[:, :5]         # 左奇异向量
    >>> vT = orthogonal[:, 5:].T      # 右奇异向量
    >>> A = u @ diags(s) @ vT

    仅使用三个奇异值/向量，SVD可以近似原始矩阵。

    >>> u2, s2, vT2 = svds(A, k=3, solver='propack')
    >>> A2 = u2 @ np.diag(s2) @ vT2
    >>> np.allclose(A2, A.todense(), atol=1e-3)
    True

    使用所有五个奇异值/向量，可以重现原始矩阵。

    >>> u3, s3, vT3 = svds(A, k=5, solver='propack')
    >>> A3 = u3 @ np.diag(s3) @ vT3
    >>> np.allclose(A3, A.todense())
    True

    奇异值与预期的奇异值匹配，奇异向量在符号差异上与预期相符。

    >>> (np.allclose(s3, s) and
    ...  np.allclose(np.abs(u3), np.abs(u.toarray())) and
    ...  np.allclose(np.abs(vT3), np.abs(vT.toarray())))
    True

    奇异向量也是正交的。

    >>> (np.allclose(u3.T @ u3, np.eye(5)) and
    ...  np.allclose(vT3 @ vT3.T, np.eye(5)))
    True
```