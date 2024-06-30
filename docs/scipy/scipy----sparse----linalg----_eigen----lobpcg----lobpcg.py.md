# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_eigen\lobpcg\lobpcg.py`

```
"""
Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG).

References
----------
.. [1] A. V. Knyazev (2001),
       Toward the Optimal Preconditioned Eigensolver: Locally Optimal
       Block Preconditioned Conjugate Gradient Method.
       SIAM Journal on Scientific Computing 23, no. 2,
       pp. 517-541. :doi:`10.1137/S1064827500366124`

.. [2] A. V. Knyazev, I. Lashuk, M. E. Argentati, and E. Ovchinnikov (2007),
       Block Locally Optimal Preconditioned Eigenvalue Xolvers (BLOPEX)
       in hypre and PETSc.  :arxiv:`0705.2626`

.. [3] A. V. Knyazev's C and MATLAB implementations:
       https://github.com/lobpcg/blopex
"""

# 引入警告模块，用于可能的警告信息
import warnings
# 引入NumPy库，用于数值计算
import numpy as np
# 从SciPy线性代数模块中引入特定函数
from scipy.linalg import (inv, eigh, cho_factor, cho_solve,
                          cholesky, LinAlgError)
# 从SciPy稀疏矩阵模块中引入特定函数和类
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import issparse

# 模块级别的导出列表，只导出lobpcg函数
__all__ = ["lobpcg"]


def _report_nonhermitian(M, name):
    """
    Report if `M` is not a Hermitian matrix given its type.
    """
    # 从SciPy线性代数模块中引入norm函数
    from scipy.linalg import norm

    # 计算矩阵M与其共轭转置之差的一范数
    md = M - M.T.conj()
    # 计算差的一范数
    nmd = norm(md, 1)
    # 计算机器精度的估计
    tol = 10 * np.finfo(M.dtype).eps
    # 更新容差值，取较大的一个
    tol = max(tol, tol * norm(M, 1))
    # 如果差的一范数超过容差值，则发出警告
    if nmd > tol:
        warnings.warn(
              f"Matrix {name} of the type {M.dtype} is not Hermitian: "
              f"condition: {nmd} < {tol} fails.",
              UserWarning, stacklevel=4
         )

def _as2d(ar):
    """
    If the input array is 2D return it, if it is 1D, append a dimension,
    making it a column vector.
    """
    # 如果输入数组是二维的则直接返回，否则将一维数组转换成列向量
    if ar.ndim == 2:
        return ar
    else:  # 假设为一维数组
        # 将输入数组转换成NumPy数组
        aux = np.asarray(ar)
        # 调整数组形状为列向量形式
        aux.shape = (ar.shape[0], 1)
        return aux


def _makeMatMat(m):
    # 如果m为None，则返回None；如果m是可调用的函数，则返回一个lambda函数；否则返回一个矩阵乘法函数
    if m is None:
        return None
    elif callable(m):
        return lambda v: m(v)
    else:
        return lambda v: m @ v


def _matmul_inplace(x, y, verbosityLevel=0):
    """Perform 'np.matmul' in-place if possible.

    If some sufficient conditions for inplace matmul are met, do so.
    Otherwise try inplace update and fall back to overwrite if that fails.
    """
    # 如果x是C连续的，且x与y具有兼容的数据类型和形状，则进行原地矩阵乘法
    if x.flags["CARRAY"] and x.shape[1] == y.shape[1] and x.dtype == y.dtype:
        np.matmul(x, y, out=x)
    else:
        # 否则尝试原地更新，如果失败则回退到覆盖操作
        try:
            np.matmul(x, y, out=x)
        except Exception:
            if verbosityLevel:
                # 如果verbosityLevel非零，则发出警告信息
                warnings.warn(
                    "Inplace update of x = x @ y failed, "
                    "x needs to be overwritten.",
                    UserWarning, stacklevel=3
                )
            x = x @ y
    return x
def _applyConstraints(blockVectorV, factYBY, blockVectorBY, blockVectorY):
    """Changes blockVectorV in-place."""
    # 计算 YBV = B^H * V
    YBV = blockVectorBY.T.conj() @ blockVectorV
    # 解线性方程组得到 tmp = (YB Y)^(-1) * YBV
    tmp = cho_solve(factYBY, YBV)
    # 更新 blockVectorV = V - Y * tmp
    blockVectorV -= blockVectorY @ tmp


def _b_orthonormalize(B, blockVectorV, blockVectorBV=None,
                      verbosityLevel=0):
    """in-place B-orthonormalize the given block vector using Cholesky."""
    # 如果 blockVectorBV 未提供，则根据 B 对 blockVectorV 进行变换
    if blockVectorBV is None:
        if B is None:
            blockVectorBV = blockVectorV
        else:
            try:
                # 尝试用 B 对 blockVectorV 进行乘法变换
                blockVectorBV = B(blockVectorV)
            except Exception as e:
                # 若失败则警告并返回 None
                if verbosityLevel:
                    warnings.warn(
                        f"Secondary MatMul call failed with error\n"
                        f"{e}\n",
                        UserWarning, stacklevel=3
                    )
                    return None, None, None
            # 检查变换后的矩阵形状是否保持一致
            if blockVectorBV.shape != blockVectorV.shape:
                raise ValueError(
                    f"The shape {blockVectorV.shape} "
                    f"of the orthogonalized matrix not preserved\n"
                    f"and changed to {blockVectorBV.shape} "
                    f"after multiplying by the secondary matrix.\n"
                )

    # 计算 VBV = V^H * BV
    VBV = blockVectorV.T.conj() @ blockVectorBV
    try:
        # 尝试对 VBV 进行 Cholesky 分解
        VBV = cholesky(VBV, overwrite_a=True)
        # 计算 VBV 的逆
        VBV = inv(VBV, overwrite_a=True)
        # 对 blockVectorV 进行原地乘法操作 V = V * VBV
        blockVectorV = _matmul_inplace(
            blockVectorV, VBV,
            verbosityLevel=verbosityLevel
        )
        # 如果有 B，则对 blockVectorBV 进行相同的乘法操作 BV = BV * VBV
        if B is not None:
            blockVectorBV = _matmul_inplace(
                blockVectorBV, VBV,
                verbosityLevel=verbosityLevel
            )
        return blockVectorV, blockVectorBV, VBV
    except LinAlgError:
        # 若 Cholesky 分解失败则警告并返回 None
        if verbosityLevel:
            warnings.warn(
                "Cholesky has failed.",
                UserWarning, stacklevel=3
            )
        return None, None, None


def _get_indx(_lambda, num, largest):
    """Get `num` indices into `_lambda` depending on `largest` option."""
    # 对 _lambda 进行排序，获取索引
    ii = np.argsort(_lambda)
    # 根据 largest 参数选择返回最大值或最小值的索引
    if largest:
        ii = ii[:-num - 1:-1]
    else:
        ii = ii[:num]

    return ii


def _handle_gramA_gramB_verbosity(gramA, gramB, verbosityLevel):
    # 如果 verbosityLevel 非零，则报告 gramA 和 gramB 的非厄米特性
    if verbosityLevel:
        _report_nonhermitian(gramA, "gramA")
        _report_nonhermitian(gramB, "gramB")
    # A: {sparse matrix, ndarray, LinearOperator, callable object}
    #    问题的 Hermitian 线性算子，通常由稀疏矩阵给出。常称为“刚度矩阵”。
    # X: ndarray, float32 or float64
    #    对 ``k`` 个特征向量的初始近似（非稀疏）。如果 `A` 的 ``shape=(n,n)``，则 `X` 必须具有 ``shape=(n,k)``。
    # B: {sparse matrix, ndarray, LinearOperator, callable object}, optional
    #    默认为 ``B = None``，相当于单位矩阵。在广义特征值问题中作为右手边算子。常称为“质量矩阵”。必须是 Hermitian 正定的。
    # M: {sparse matrix, ndarray, LinearOperator, callable object}, optional
    #    默认为 ``M = None``，相当于单位矩阵。旨在加速收敛的预条件器。
    # Y: ndarray, float32 or float64, default: None
    #    ``n-by-sizeY`` 的约束数组，其中 ``sizeY < n``。迭代将在 `Y` 的列空间的 ``B``-正交补空间中进行。若存在，`Y` 必须是满秩的。
    # tol: scalar, optional
    #    默认为 ``tol=n*sqrt(eps)``。解算器停止标准的容差。
    # maxiter: int, default: 20
    #    最大迭代次数。
    # largest: bool, default: True
    #    当为 True 时，求解最大特征值，否则求解最小特征值。
    # verbosityLevel: int, optional
    #    默认为 ``verbosityLevel=0``，没有输出。控制解算器的标准/屏幕输出。
    # retLambdaHistory: bool, default: False
    #    是否返回迭代的特征值历史。
    # retResidualNormsHistory: bool, default: False
    #    是否返回迭代残差范数的历史。
    # restartControl: int, optional
    #    如果残差相对于 ``retResidualNormsHistory`` 中最小记录的值增加 ``2**restartControl`` 倍，则进行迭代重启。
    #    默认为 ``restartControl=20``，对于向后兼容性，使重启较少发生。

    # 返回值
    # -------
    # lambda: ndarray, shape ``(k, )``
    #    长度为 ``k`` 的近似特征值数组。
    # v: ndarray, 形状与 ``X.shape`` 相同
    #    长度为 ``k`` 的近似特征向量数组。
    # lambdaHistory: ndarray, optional
    #    如果 `retLambdaHistory` 为 ``True``，则返回特征值的历史。
    # ResidualNormsHistory: ndarray, optional
    #    如果 `retResidualNormsHistory` 为 ``True``，则返回残差范数的历史。

    # 注意事项
    # ------
    # 迭代循环最多运行 ``maxit=maxiter`` 次迭代（如果 ``maxit=None``，则为 20 次），并在达到容差时提前结束。
    # 与先前版本不兼容，LOBPCG 现在返回最精确的迭代向量块，而不是最后一个迭代的向量块，作为可能发散的疗法。
    # 如果 ``X.dtype == np.float32`` 并且用户提供的操作/乘法对 `A`, `B` 和 `M` 保留了 ``np.float32`` 数据类型，
    # 所有的计算和输出结果都将使用 ``np.float32`` 数据类型。
    
    # 迭代历史记录的大小等于最佳迭代次数的数量（限制为 `maxit`）加上3：初始、最终和后处理。
    
    # 如果 `retLambdaHistory` 和 `retResidualNormsHistory` 都为 ``True``，
    # 返回的元组格式为 ``(lambda, V, lambda history, residual norms history)``。
    
    # 在接下来的说明中，``n`` 表示矩阵的大小，``k`` 表示需要的特征值（最小或最大）。
    
    # LOBPCG 算法在每次迭代中通过调用密集特征值求解器 `eigh` 来解决大小为 ``3k`` 的特征值问题，
    # 因此如果 ``k`` 相对于 ``n`` 不够小，调用 LOBPCG 算法就没有意义。
    # 此外，如果调用 LOBPCG 算法时 ``5k > n``，它可能会在内部中断，因此代码会调用标准函数 `eigh`。
    # LOBPCG 算法并不要求 ``n`` 很大才能工作，而是要求比值 ``n / k`` 很大。
    # 当 ``k=1`` 且 ``n=10`` 时，LOBPCG 可以工作，尽管 ``n`` 很小。该方法适用于 ``n / k`` 极大的情况。
    
    # 收敛速度基本上取决于三个因素：
    
    # 1. 初始近似解 `X` 的质量，用于寻找特征向量。
    #    如果没有更好的选择，随机分布在原点附近的向量效果很好。
    
    # 2. 所需特征值与其余特征值的相对分离程度。可以通过调整 ``k`` 来改善分离度。
    
    # 3. 适当的预条件处理以收缩谱间隔。
    #    例如，对于大 ``n`` 的杆振动测试问题，如果不使用有效的预条件处理，收敛速度将很慢。
    #    对于这种特定问题，一个简单而有效的预条件处理函数是对 `A` 进行线性求解，因为 `A` 是三对角的。
    
    # 参考文献
    # ----------
    # .. [1] A. V. Knyazev (2001),
    #        Toward the Optimal Preconditioned Eigensolver: Locally Optimal
    #        Block Preconditioned Conjugate Gradient Method.
    #        SIAM Journal on Scientific Computing 23, no. 2,
    #        pp. 517-541. :doi:`10.1137/S1064827500366124`
    #
    # .. [2] A. V. Knyazev, I. Lashuk, M. E. Argentati, and E. Ovchinnikov
    #        (2007), Block Locally Optimal Preconditioned Eigenvalue Solvers
    #        (BLOPEX) in hypre and PETSc. :arxiv:`0705.2626`
    #
    # .. [3] A. V. Knyazev's C and MATLAB implementations:
    #        https://github.com/lobpcg/blopex
    
    # 示例
    # --------
    # 我们的第一个示例非常简单 - 通过解决非广义特征值问题 ``A x = lambda x`` 来找到对角矩阵的最大特征值，
    # 没有约束或预条件处理。
    >>> import numpy as np
    >>> from scipy.sparse import spdiags
    >>> from scipy.sparse.linalg import LinearOperator, aslinearoperator
    >>> from scipy.sparse.linalg import lobpcg
    
    The square matrix size is
    
    >>> n = 100
    
    and its diagonal entries are 1, ..., 100 defined by
    
    >>> vals = np.arange(1, n + 1).astype(np.int16)
    
    The first mandatory input parameter in this test is
    the sparse diagonal matrix `A`
    of the eigenvalue problem ``A x = lambda x`` to solve.
    
    >>> A = spdiags(vals, 0, n, n)   # 创建稀疏对角矩阵A，对角线元素为vals数组
    >>> A = A.astype(np.int16)       # 将矩阵A的数据类型转换为np.int16
    >>> A.toarray()                  # 将稀疏矩阵A转换为密集矩阵并显示
    array([[  1,   0,   0, ...,   0,   0,   0],
           [  0,   2,   0, ...,   0,   0,   0],
           [  0,   0,   3, ...,   0,   0,   0],
           ...,
           [  0,   0,   0, ...,  98,   0,   0],
           [  0,   0,   0, ...,   0,  99,   0],
           [  0,   0,   0, ...,   0,   0, 100]], dtype=int16)
    
    The second mandatory input parameter `X` is a 2D array with the
    row dimension determining the number of requested eigenvalues.
    `X` is an initial guess for targeted eigenvectors.
    `X` must have linearly independent columns.
    If no initial approximations available, randomly oriented vectors
    commonly work best, e.g., with components normally distributed
    around zero or uniformly distributed on the interval [-1 1].
    Setting the initial approximations to dtype ``np.float32``
    forces all iterative values to dtype ``np.float32`` speeding up
    the run while still allowing accurate eigenvalue computations.
    
    >>> k = 1
    >>> rng = np.random.default_rng()
    >>> X = rng.normal(size=(n, k))   # 创建大小为(n, k)的正态分布随机矩阵X作为初值
    >>> X = X.astype(np.float32)      # 将矩阵X的数据类型转换为np.float32
    
    >>> eigenvalues, _ = lobpcg(A, X, maxiter=60)   # 使用lobpcg求解特征值问题
    >>> eigenvalues                               # 输出求解得到的特征值
    array([100.])
    >>> eigenvalues.dtype                         # 输出特征值数组的数据类型
    dtype('float32')
    
    `lobpcg` needs only access the matrix product with `A` rather
    then the matrix itself. Since the matrix `A` is diagonal in
    this example, one can write a function of the matrix product
    ``A @ X`` using the diagonal values ``vals`` only, e.g., by
    element-wise multiplication with broadcasting in the lambda-function
    
    >>> A_lambda = lambda X: vals[:, np.newaxis] * X   # 定义lambda函数A_lambda，实现矩阵-向量乘法
    
    or the regular function
    
    >>> def A_matmat(X):
    ...     return vals[:, np.newaxis] * X            # 定义函数A_matmat，实现矩阵-矩阵乘法
    
    and use the handle to one of these callables as an input
    
    >>> eigenvalues, _ = lobpcg(A_lambda, X, maxiter=60)   # 使用A_lambda作为输入求解特征值问题
    >>> eigenvalues                                       # 输出求解得到的特征值
    array([100.])
    >>> eigenvalues, _ = lobpcg(A_matmat, X, maxiter=60)   # 使用A_matmat作为输入求解特征值问题
    >>> eigenvalues                                       # 输出求解得到的特征值
    array([100.])
    
    The traditional callable `LinearOperator` is no longer
    necessary but still supported as the input to `lobpcg`.
    Specifying ``matmat=A_matmat`` explicitly improves performance.
    
    >>> A_lo = LinearOperator((n, n), matvec=A_matmat, matmat=A_matmat, dtype=np.int16)   # 创建线性操作符A_lo
    >>> eigenvalues, _ = lobpcg(A_lo, X, maxiter=80)   # 使用A_lo作为输入求解特征值问题
    >>> eigenvalues                                   # 输出求解得到的特征值
    array([100.])
    
    The least efficient callable option is `aslinearoperator`:
    # 将 lobpcg 函数应用于线性操作符 A，使用矩阵 X 进行计算，最多进行 80 次迭代
    eigenvalues, _ = lobpcg(aslinearoperator(A), X, maxiter=80)
    # 输出计算得到的特征值数组
    eigenvalues
    # 创建一个 n 行 k 列的随机正态分布矩阵 X
    X = np.random.default_rng().normal(size=(n, k))

    # 利用 lobpcg 函数计算矩阵 A 的最小的三个特征值，设置 largest=False，最多进行 90 次迭代
    eigenvalues, _ = lobpcg(A, X, largest=False, maxiter=90)
    # 打印计算得到的特征值数组
    print(eigenvalues)

    # 创建一个 n 行 3 列的单位矩阵 Y，用作约束条件的输入参数
    Y = np.eye(n, 3)

    # 定义一个 lambda 函数 M，实现对输入矩阵 X 的预处理
    inv_vals = 1./vals
    inv_vals = inv_vals.astype(np.float32)
    M = lambda X: inv_vals[:, np.newaxis] * X

    # 利用 lobpcg 函数计算矩阵 A 的最小的三个特征值，设置 largest=False，最多进行 80 次迭代，不进行预处理
    eigenvalues, _ = lobpcg(A_matmat, X, Y=Y, largest=False, maxiter=80)
    # 输出计算得到的特征值数组
    eigenvalues

    # 输出计算得到的特征值数组的数据类型
    eigenvalues.dtype

    # 利用 lobpcg 函数计算矩阵 A 的最小的三个特征值，设置 largest=False，最多进行 20 次迭代，使用预处理 M
    eigenvalues, _ = lobpcg(A_matmat, X, Y=Y, M=M, largest=False, maxiter=20)
    # 输出计算得到的特征值数组
    eigenvalues

    # 如果 blockVectorY 不为 None，则检查其维度是否为二维，若不是则发出警告并将其设为 None
    sizeY = 0
    if blockVectorY is not None:
        if len(blockVectorY.shape) != 2:
            warnings.warn(
                f"Expected rank-2 array for argument Y, instead got "
                f"{len(blockVectorY.shape)}, "
                f"so ignore it and use no constraints.",
                UserWarning, stacklevel=2
            )
            blockVectorY = None
        else:
            sizeY = blockVectorY.shape[1]

    # 如果 blockVectorX 为 None，则抛出 ValueError
    # 否则，将 blockVectorX 赋值给 bestblockVectorX
    blockVectorX = X
    bestblockVectorX = blockVectorX

    # 将 blockVectorY 赋值给 blockVectorY，并设定残差容限为 tol
    blockVectorY = Y
    residualTolerance = tol

    # 如果 maxiter 为 None，则将其设定为 20
    if maxiter is None:
        maxiter = 20

    # 将 maxiter 设为 bestIterationNumber
    bestIterationNumber = maxiter

    # 如果 blockVectorY 不为 None，则检查其维度是否为二维，若不是则发出警告并将其设为 None
    sizeY = 0
    if blockVectorY is not None:
        if len(blockVectorY.shape) != 2:
            warnings.warn(
                f"Expected rank-2 array for argument Y, instead got "
                f"{len(blockVectorY.shape)}, "
                f"so ignore it and use no constraints.",
                UserWarning, stacklevel=2
            )
            blockVectorY = None
        else:
            sizeY = blockVectorY.shape[1]

    # 如果 blockVectorX 为 None，则抛出 ValueError
    # 否则，将 blockVectorX 赋值给 bestblockVectorX
    blockVectorX = X
    bestblockVectorX = blockVectorX

    # 将 blockVectorY 赋值给 blockVectorY，并设定残差容限为 tol
    blockVectorY = Y
    residualTolerance = tol

    # 如果 maxiter 为 None，则将其设定为 20
    if maxiter is None:
        maxiter = 20

    # 将 maxiter 设为 bestIterationNumber
    bestIterationNumber = maxiter

    # 如果 blockVectorY 不为 None，则检查其维度是否为二维，若不是则发出警告并将其设为 None
    sizeY = 0
    if blockVectorY is not None:
        if len(blockVectorY.shape) != 2:
            warnings.warn(
                f"Expected rank-2 array for argument Y, instead got "
                f"{len(blockVectorY.shape)}, "
                f"so ignore it and use no constraints.",
                UserWarning, stacklevel=2
            )
            blockVectorY = None
        else:
            sizeY = blockVectorY.shape[1]

    # 如果 blockVectorX 为 None，则抛出 ValueError
    # 否则，将 blockVectorX 赋值给 bestblockVectorX
    blockVectorX = X
    bestblockVectorX = blockVectorX

    # 将 blockVectorY 赋值给 blockVectorY，并设定残差容限为 tol
    blockVectorY = Y
    residualTolerance = tol

    # 如果 maxiter 为 None，则将其设定为 20
    if maxiter is None:
        maxiter = 20

    # 将 maxiter 设为 bestIterationNumber
    bestIterationNumber = maxiter

    # 如果 blockVectorY 不为 None，则检查其维度是否为二维，若不是则发出警告并将其设为 None
    # 检查 blockVectorX 的形状是否为二维数组，如果不是则抛出数值错误异常
    if len(blockVectorX.shape) != 2:
        raise ValueError("expected rank-2 array for argument X")

    # 获取 blockVectorX 的形状信息
    n, sizeX = blockVectorX.shape

    # 检查 blockVectorX 的数据类型是否为浮点数类型，如果不是则发出警告，并转换为 np.float32 类型
    if not np.issubdtype(blockVectorX.dtype, np.inexact):
        warnings.warn(
            f"Data type for argument X is {blockVectorX.dtype}, "
            f"which is not inexact, so casted to np.float32.",
            UserWarning, stacklevel=2
        )
        blockVectorX = np.asarray(blockVectorX, dtype=np.float32)

    # 如果需要记录 lambda 的历史，则创建一个用于存储 lambda 值的零矩阵
    if retLambdaHistory:
        lambdaHistory = np.zeros((maxiter + 3, sizeX),
                                 dtype=blockVectorX.dtype)

    # 如果需要记录残差范数的历史，则创建一个用于存储残差范数的零矩阵
    if retResidualNormsHistory:
        residualNormsHistory = np.zeros((maxiter + 3, sizeX),
                                        dtype=blockVectorX.dtype)

    # 如果 verbosityLevel 非零，则打印关于求解过程的详细信息
    if verbosityLevel:
        aux = "Solving "
        if B is None:
            aux += "standard"
        else:
            aux += "generalized"
        aux += " eigenvalue problem with"
        if M is None:
            aux += "out"
        aux += " preconditioning\n\n"
        aux += "matrix size %d\n" % n
        aux += "block size %d\n\n" % sizeX
        if blockVectorY is None:
            aux += "No constraints\n\n"
        else:
            if sizeY > 1:
                aux += "%d constraints\n\n" % sizeY
            else:
                aux += "%d constraint\n\n" % sizeY
        print(aux)
    # 检查问题的尺寸是否满足要求，如果不满足则发出警告并采用密集特征值求解器
    if (n - sizeY) < (5 * sizeX):
        warnings.warn(
            f"The problem size {n} minus the constraints size {sizeY} "
            f"is too small relative to the block size {sizeX}. "
            f"Using a dense eigensolver instead of LOBPCG iterations."
            f"No output of the history of the iterations.",
            UserWarning, stacklevel=2
        )

        # 调整 sizeX，确保其不超过问题的尺寸
        sizeX = min(sizeX, n)

        # 如果存在约束条件 blockVectorY，则抛出未实现错误
        if blockVectorY is not None:
            raise NotImplementedError(
                "The dense eigensolver does not support constraints."
            )

        # 定义要返回的特征值索引的闭区间范围
        if largest:
            eigvals = (n - sizeX, n - 1)
        else:
            eigvals = (0, sizeX - 1)

        try:
            # 尝试根据 A 的类型调整成可用的矩阵形式
            if isinstance(A, LinearOperator):
                A = A(np.eye(n, dtype=int))
            elif callable(A):
                A = A(np.eye(n, dtype=int))
                if A.shape != (n, n):
                    raise ValueError(
                        f"The shape {A.shape} of the primary matrix\n"
                        f"defined by a callable object is wrong.\n"
                    )
            elif issparse(A):
                A = A.toarray()
            else:
                A = np.asarray(A)
        except Exception as e:
            # 如果 A 转换失败，抛出异常
            raise Exception(
                f"Primary MatMul call failed with error\n"
                f"{e}\n")

        # 如果 B 存在，尝试将其调整成可用的矩阵形式
        if B is not None:
            try:
                if isinstance(B, LinearOperator):
                    B = B(np.eye(n, dtype=int))
                elif callable(B):
                    B = B(np.eye(n, dtype=int))
                    if B.shape != (n, n):
                        raise ValueError(
                            f"The shape {B.shape} of the secondary matrix\n"
                            f"defined by a callable object is wrong.\n"
                        )
                elif issparse(B):
                    B = B.toarray()
                else:
                    B = np.asarray(B)
            except Exception as e:
                # 如果 B 转换失败，抛出异常
                raise Exception(
                    f"Secondary MatMul call failed with error\n"
                    f"{e}\n")

        try:
            # 调用密集特征值求解器求解特征值和特征向量
            vals, vecs = eigh(A,
                              B,
                              subset_by_index=eigvals,
                              check_finite=False)
            if largest:
                # 如果求解最大特征值，则反转顺序以与 'LM' 模式中的 eigs() 兼容
                vals = vals[::-1]
                vecs = vecs[:, ::-1]

            return vals, vecs
        except Exception as e:
            # 如果密集特征值求解失败，抛出异常
            raise Exception(
                f"Dense eigensolver failed with error\n"
                f"{e}\n"
            )

    # 如果 residualTolerance 未设置或小于等于 0，则设置一个默认值
    if (residualTolerance is None) or (residualTolerance <= 0.0):
        residualTolerance = np.sqrt(np.finfo(blockVectorX.dtype).eps) * n

    # 将 A、B、M 转换为矩阵-矩阵乘法形式
    A = _makeMatMat(A)
    B = _makeMatMat(B)
    M = _makeMatMat(M)

    # 对 X 应用约束条件
    # 如果 blockVectorY 不为 None，则进行以下操作
    if blockVectorY is not None:

        # 如果 B 不为 None，则对 blockVectorY 进行 B-变换，得到 blockVectorBY
        if B is not None:
            blockVectorBY = B(blockVectorY)
            # 检查变换后的形状是否与原始 blockVectorY 的形状相同，否则抛出 ValueError 异常
            if blockVectorBY.shape != blockVectorY.shape:
                raise ValueError(
                    f"The shape {blockVectorY.shape} "
                    f"of the constraint not preserved\n"
                    f"and changed to {blockVectorBY.shape} "
                    f"after multiplying by the secondary matrix.\n"
                )
        else:
            # 如果 B 为 None，则直接将 blockVectorBY 设置为 blockVectorY
            blockVectorBY = blockVectorY

        # 计算 gramYBY，这是一个密集数组
        gramYBY = blockVectorY.T.conj() @ blockVectorBY
        try:
            # 尝试对 gramYBY 进行 Cholesky 分解，这里覆盖原始数组
            gramYBY = cho_factor(gramYBY, overwrite_a=True)
        except LinAlgError as e:
            # 如果 Cholesky 分解失败，则抛出 ValueError 异常
            raise ValueError("Linearly dependent constraints") from e

        # 应用约束条件到 blockVectorX
        _applyConstraints(blockVectorX, gramYBY, blockVectorBY, blockVectorY)

    ##
    # 对 blockVectorX 进行 B-正交化处理
    blockVectorX, blockVectorBX, _ = _b_orthonormalize(
        B, blockVectorX, verbosityLevel=verbosityLevel)
    if blockVectorX is None:
        # 如果处理后的 blockVectorX 为 None，则抛出 ValueError 异常
        raise ValueError("Linearly dependent initial approximations")

    ##
    # 计算初始的 Ritz 向量：解决特征值问题
    blockVectorAX = A(blockVectorX)
    if blockVectorAX.shape != blockVectorX.shape:
        # 检查计算后的 blockVectorAX 的形状是否与原始 blockVectorX 的形状相同，否则抛出 ValueError 异常
        raise ValueError(
            f"The shape {blockVectorX.shape} "
            f"of the initial approximations not preserved\n"
            f"and changed to {blockVectorAX.shape} "
            f"after multiplying by the primary matrix.\n"
        )

    # 计算 gramXAX
    gramXAX = blockVectorX.T.conj() @ blockVectorAX

    # 求解特征值问题，返回特征值 _lambda 和特征向量 eigBlockVector
    _lambda, eigBlockVector = eigh(gramXAX, check_finite=False)
    ii = _get_indx(_lambda, sizeX, largest)
    _lambda = _lambda[ii]
    if retLambdaHistory:
        lambdaHistory[0, :] = _lambda

    # 选择特征向量的子集
    eigBlockVector = np.asarray(eigBlockVector[:, ii])

    # 将特征向量与 blockVectorX 进行矩阵乘法，覆盖原始 blockVectorX
    blockVectorX = _matmul_inplace(
        blockVectorX, eigBlockVector,
        verbosityLevel=verbosityLevel
    )

    # 将特征向量与 blockVectorAX 进行矩阵乘法，覆盖原始 blockVectorAX
    blockVectorAX = _matmul_inplace(
        blockVectorAX, eigBlockVector,
        verbosityLevel=verbosityLevel
    )

    # 如果 B 不为 None，则将特征向量与 blockVectorBX 进行矩阵乘法，覆盖原始 blockVectorBX
    if B is not None:
        blockVectorBX = _matmul_inplace(
            blockVectorBX, eigBlockVector,
            verbosityLevel=verbosityLevel
        )

    ##
    # 活跃索引集合初始化
    activeMask = np.ones((sizeX,), dtype=bool)

    ##
    # 主迭代循环开始

    # blockVectorP 将在迭代过程中设置
    blockVectorP = None  # 在迭代过程中设置
    blockVectorAP = None
    blockVectorBP = None

    # 初始化最小残差范数
    smallestResidualNorm = np.abs(np.finfo(blockVectorX.dtype).max)

    # 迭代编号初始化
    iterationNumber = -1

    # 是否重新启动迭代
    restart = True

    # 是否强制重新启动
    forcedRestart = False

    # 显式 Gram 矩阵标志位
    explicitGramFlag = False

    # 如果 B 不为 None，则计算 aux = blockVectorBX * _lambda[np.newaxis, :]
    # 否则计算 aux = blockVectorX * _lambda[np.newaxis, :]
    if B is not None:
        aux = blockVectorBX * _lambda[np.newaxis, :]
    else:
        aux = blockVectorX * _lambda[np.newaxis, :]

    # 计算残差向量 blockVectorR
    blockVectorR = blockVectorAX - aux

    # 计算残差的范数
    aux = np.sum(blockVectorR.conj() * blockVectorR, 0)
    residualNorms = np.sqrt(np.abs(aux))
    # 在提前退出循环的情况下使用旧的 lambda 值。
    # 如果需要返回 lambda 历史记录，则将当前迭代的 lambda 存入 lambdaHistory 中
    if retLambdaHistory:
        lambdaHistory[iterationNumber + 1, :] = _lambda

    # 如果需要返回残差范数历史记录，则将当前迭代的残差范数存入 residualNormsHistory 中
    if retResidualNormsHistory:
        residualNormsHistory[iterationNumber + 1, :] = residualNorms

    # 计算当前迭代的平均残差范数
    residualNorm = np.sum(np.abs(residualNorms)) / sizeX

    # 如果当前平均残差范数小于最小残差范数，则更新最小残差范数及其对应的迭代信息和向量
    if residualNorm < smallestResidualNorm:
        smallestResidualNorm = residualNorm
        bestIterationNumber = iterationNumber + 1
        bestblockVectorX = blockVectorX

    # 如果最大残差范数超过预设的残差容限值，则发出警告提示
    if np.max(np.abs(residualNorms)) > residualTolerance:
        warnings.warn(
            f"Exited at iteration {iterationNumber} with accuracies \n"
            f"{residualNorms}\n"
            f"not reaching the requested tolerance {residualTolerance}.\n"
            f"Use iteration {bestIterationNumber} instead with accuracy \n"
            f"{smallestResidualNorm}.\n",
            UserWarning, stacklevel=2
        )

    # 如果设置了详细输出级别，则打印最终迭代得到的特征值和残差范数
    if verbosityLevel:
        print(f"Final iterative eigenvalue(s):\n{_lambda}")
        print(f"Final iterative residual norm(s):\n{residualNorms}")

    # 将 blockVectorX 更新为最优的 blockVectorX，以确保满足 blockVectorY 的约束
    blockVectorX = bestblockVectorX

    # 如果存在 blockVectorY，则调用函数 _applyConstraints，确保满足 blockVectorY 的约束
    if blockVectorY is not None:
        _applyConstraints(blockVectorX,
                          gramYBY,
                          blockVectorBY,
                          blockVectorY)

    # 将 blockVectorX 乘以 A，得到 blockVectorAX，用于确保满足最终的正交化要求
    blockVectorAX = A(blockVectorX)

    # 检查 blockVectorAX 的形状是否与 blockVectorX 相同，如果不同则引发 ValueError 异常
    if blockVectorAX.shape != blockVectorX.shape:
        raise ValueError(
            f"The shape {blockVectorX.shape} "
            f"of the postprocessing iterate not preserved\n"
            f"and changed to {blockVectorAX.shape} "
            f"after multiplying by the primary matrix.\n"
        )

    # 计算 blockVectorX 和 blockVectorAX 的 Gram 矩阵 gramXAX
    gramXAX = np.dot(blockVectorX.T.conj(), blockVectorAX)

    # 将 blockVectorBX 初始化为 blockVectorX，如果定义了 B，则将 blockVectorX 乘以 B 得到 blockVectorBX
    blockVectorBX = blockVectorX
    if B is not None:
        blockVectorBX = B(blockVectorX)

        # 检查 blockVectorBX 的形状是否与 blockVectorX 相同，如果不同则引发 ValueError 异常
        if blockVectorBX.shape != blockVectorX.shape:
            raise ValueError(
                f"The shape {blockVectorX.shape} "
                f"of the postprocessing iterate not preserved\n"
                f"and changed to {blockVectorBX.shape} "
                f"after multiplying by the secondary matrix.\n"
            )

    # 计算 blockVectorX 和 blockVectorBX 的 Gram 矩阵 gramXBX
    gramXBX = np.dot(blockVectorX.T.conj(), blockVectorBX)

    # 处理 gramXAX 和 gramXBX 的详细输出级别
    _handle_gramA_gramB_verbosity(gramXAX, gramXBX, verbosityLevel)

    # 对 gramXAX 和 gramXBX 进行对称化处理
    gramXAX = (gramXAX + gramXAX.T.conj()) / 2
    gramXBX = (gramXBX + gramXBX.T.conj()) / 2

    # 尝试使用 eigh 函数计算特征值和特征向量
    try:
        _lambda, eigBlockVector = eigh(gramXAX,
                                       gramXBX,
                                       check_finite=False)
    except LinAlgError as e:
        # 如果 eigh 函数抛出 LinAlgError 异常，则引发 ValueError 异常
        raise ValueError("eigh has failed in lobpcg postprocessing") from e

    # 根据特征值的索引 ii 对 _lambda 和 eigBlockVector 进行筛选
    ii = _get_indx(_lambda, sizeX, largest)
    _lambda = _lambda[ii]
    eigBlockVector = np.asarray(eigBlockVector[:, ii])

    # 更新 blockVectorX 为其与 eigBlockVector 的乘积
    blockVectorX = np.dot(blockVectorX, eigBlockVector)

    # 更新 blockVectorAX 为其与 eigBlockVector 的乘积
    blockVectorAX = np.dot(blockVectorAX, eigBlockVector)
    # 如果 B 不为空，则使用特征块向量对 blockVectorBX 进行变换
    if B is not None:
        blockVectorBX = np.dot(blockVectorBX, eigBlockVector)
        # 计算辅助变量 aux，为 blockVectorBX 与 _lambda 的逐元素乘积
        aux = blockVectorBX * _lambda[np.newaxis, :]
    else:
        # 否则，直接计算辅助变量 aux，为 blockVectorX 与 _lambda 的逐元素乘积
        aux = blockVectorX * _lambda[np.newaxis, :]

    # 计算残差向量 blockVectorR，为 blockVectorAX 减去 aux
    blockVectorR = blockVectorAX - aux

    # 计算残差范数 residualNorms，为 blockVectorR 的各列向量的二范数
    aux = np.sum(blockVectorR.conj() * blockVectorR, 0)
    residualNorms = np.sqrt(np.abs(aux))

    # 如果需要返回 lambda 历史记录，则在 lambdaHistory 中记录当前 _lambda
    if retLambdaHistory:
        lambdaHistory[bestIterationNumber + 1, :] = _lambda
    # 如果需要返回残差范数历史记录，则在 residualNormsHistory 中记录当前 residualNorms
    if retResidualNormsHistory:
        residualNormsHistory[bestIterationNumber + 1, :] = residualNorms

    # 如果需要返回 lambda 历史记录，则截取 lambdaHistory 到当前迭代步数
    if retLambdaHistory:
        lambdaHistory = lambdaHistory[: bestIterationNumber + 2, :]
    # 如果需要返回残差范数历史记录，则截取 residualNormsHistory 到当前迭代步数
    if retResidualNormsHistory:
        residualNormsHistory = residualNormsHistory[: bestIterationNumber + 2, :]

    # 如果最大的残差范数大于预设的残差容差 residualTolerance，则发出警告
    if np.max(np.abs(residualNorms)) > residualTolerance:
        warnings.warn(
            f"Exited postprocessing with accuracies \n"
            f"{residualNorms}\n"
            f"not reaching the requested tolerance {residualTolerance}.",
            UserWarning, stacklevel=2
        )

    # 如果设置了输出详细信息的 verbosityLevel，则打印最终的特征值和残差范数
    if verbosityLevel:
        print(f"Final postprocessing eigenvalue(s):\n{_lambda}")
        print(f"Final residual norm(s):\n{residualNorms}")

    # 如果需要返回 lambda 历史记录，则将 lambdaHistory 拆分为单独的数组列表
    if retLambdaHistory:
        lambdaHistory = np.vsplit(lambdaHistory, np.shape(lambdaHistory)[0])
        lambdaHistory = [np.squeeze(i) for i in lambdaHistory]
    # 如果需要返回残差范数历史记录，则将 residualNormsHistory 拆分为单独的数组列表
    if retResidualNormsHistory:
        residualNormsHistory = np.vsplit(residualNormsHistory,
                                         np.shape(residualNormsHistory)[0])
        residualNormsHistory = [np.squeeze(i) for i in residualNormsHistory]

    # 根据返回设置返回相应的结果
    if retLambdaHistory:
        if retResidualNormsHistory:
            return _lambda, blockVectorX, lambdaHistory, residualNormsHistory
        else:
            return _lambda, blockVectorX, lambdaHistory
    else:
        if retResidualNormsHistory:
            return _lambda, blockVectorX, residualNormsHistory
        else:
            return _lambda, blockVectorX
```