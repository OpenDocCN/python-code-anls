# `D:\src\scipysrc\scikit-learn\sklearn\utils\extmath.py`

```
# 导入警告模块，部分函数导入，整数类型的验证
import warnings
from functools import partial
from numbers import Integral

# 导入科学计算库
import numpy as np
from scipy import linalg, sparse

# 导入参数验证相关工具
from ..utils._param_validation import Interval, StrOptions, validate_params
# 导入过时警告相关工具
from ..utils.deprecation import deprecated
# 导入数组操作相关函数
from ._array_api import _is_numpy_namespace, device, get_namespace
# 导入稀疏矩阵相关函数
from .sparsefuncs_fast import csr_row_norms
# 导入数据验证相关函数
from .validation import check_array, check_random_state


def squared_norm(x):
    """计算向量或矩阵 x 的平方欧几里得范数（Frobenius 范数）。

    比使用 norm(x) ** 2 更快速。

    Parameters
    ----------
    x : array-like
        输入数组，可以是向量或二维矩阵。

    Returns
    -------
    float
        当 x 是向量时返回欧几里得范数，当 x 是矩阵时返回 Frobenius 范数。
    """
    # 将 x 按顺序展平成一维数组
    x = np.ravel(x, order="K")
    # 如果 x 的数据类型是整数，发出警告
    if np.issubdtype(x.dtype, np.integer):
        warnings.warn(
            (
                "Array type is integer, np.dot may overflow. "
                "Data should be float type to avoid this issue"
            ),
            UserWarning,
        )
    # 计算 x 与自身的点积，即平方欧几里得范数
    return np.dot(x, x)


def row_norms(X, squared=False):
    """计算矩阵 X 的每行的（平方）欧几里得范数。

    等价于 np.sqrt((X * X).sum(axis=1))，但支持稀疏矩阵，并且不会创建临时数组。

    不执行输入验证。

    Parameters
    ----------
    X : array-like
        输入数组。
    squared : bool, default=False
        如果为 True，则返回平方范数。

    Returns
    -------
    array-like
        矩阵 X 的每行的（平方）欧几里得范数。
    """
    # 如果 X 是稀疏矩阵
    if sparse.issparse(X):
        # 转换为 CSR 格式，计算行范数
        X = X.tocsr()
        norms = csr_row_norms(X)
        # 如果不是返回平方范数，则取平方根
        if not squared:
            norms = np.sqrt(norms)
    else:
        # 获取命名空间和设备信息
        xp, _ = get_namespace(X)
        # 如果是 NumPy 命名空间
        if _is_numpy_namespace(xp):
            # 将 X 转换为 NumPy 数组，计算每行的元素平方和
            X = np.asarray(X)
            norms = np.einsum("ij,ij->i", X, X)
            # 转换为相应命名空间的数组
            norms = xp.asarray(norms)
        else:
            # 计算每行元素的平方和
            norms = xp.sum(xp.multiply(X, X), axis=1)
        # 如果不是返回平方范数，则取平方根
        if not squared:
            norms = xp.sqrt(norms)
    return norms


def fast_logdet(A):
    """计算方阵 A 的行列式的自然对数。

    当 det(A) 非负且定义良好时，返回方阵的行列式的自然对数。
    如果行列式为零或负，则返回 -Inf。

    等价于 np.log(np.det(A))，但更加健壮。

    Parameters
    ----------
    A : array_like of shape (n, n)
        方阵。

    Returns
    -------
    logdet : float
        当 det(A) 严格正时，返回 log(det(A))。
        当 det(A) 非正或未定义时，返回 -inf。

    See Also
    --------
    numpy.linalg.slogdet : 计算数组的符号和行列式的自然对数。

    Examples
    --------
    ```
    --------
    >>> import numpy as np
    >>> from sklearn.utils.extmath import fast_logdet
    >>> a = np.array([[5, 1], [2, 8]])
    >>> fast_logdet(a)
    3.6375861597263857
    """
    # 导入所需的库函数：导入numpy库并重命名为np，从sklearn.utils.extmath中导入fast_logdet函数
    xp, _ = get_namespace(A)
    # 调用get_namespace函数，返回结果分配给xp变量，忽略第二个返回值
    sign, ld = xp.linalg.slogdet(A)
    # 使用xp（可能是numpy或cupy）的linalg模块的slogdet函数计算输入矩阵A的符号和其行列式的自然对数值
    if not sign > 0:
        # 如果行列式的符号不大于0（即非正数），返回负无穷
        return -xp.inf
    # 否则返回计算得到的行列式的自然对数值
    return ld
def randomized_range_finder(
    A, *, size, n_iter, power_iteration_normalizer="auto", random_state=None
):
    """Compute an orthonormal matrix whose range approximates the range of A.

    Parameters
    ----------
    A : 2D array
        The input data matrix.

    size : int
        Size of the return array.

    n_iter : int
        Number of power iterations used to stabilize the result.

    power_iteration_normalizer : {'auto', 'QR', None}, default='auto'
        Whether to use QR decomposition to normalize the power iterations.

    random_state : int, RandomState instance or None, default=None
        Random number generator seed control.

    Returns
    -------
    Q : 2D array
        A matrix with orthonormal rows, approximating the range of A.

    Notes
    -----
    This function computes an approximation to the range of A using the
    randomized power iteration method. The number of iterations `n_iter`
    affects the approximation accuracy. Larger `n_iter` improves accuracy
    but increases computation time.

    References
    ----------
    Halko, N., Martinsson, P. G., & Tropp, J. A. (2011).
    Finding structure with randomness: Probabilistic algorithms for
    constructing approximate matrix decompositions. SIAM Review, 53(2), 217-288.
    """
    if power_iteration_normalizer == "auto":
        # Determine the normalization method based on matrix dimensions
        power_iteration_normalizer = 'QR' if A.shape[0] >= A.shape[1] else None
    
    # Initialize a random number generator
    rng = np.random.RandomState(random_state)
    
    # Generate a random Gaussian matrix
    G = rng.normal(size=(A.shape[1], size))
    
    # Perform power iterations
    for _ in range(n_iter):
        # Perform matrix multiplication with A and G
        G = safe_sparse_dot(A.T, A @ G, dense_output=G.dtype == np.float64)
        
        if power_iteration_normalizer == 'QR':
            # Apply QR decomposition for normalization
            Q, _ = np.linalg.qr(G, mode='reduced')
        elif power_iteration_normalizer is None:
            # Directly apply column-wise normalization
            Q, _ = np.linalg.qr(G, mode='economic')
        else:
            raise ValueError(f"Unknown normalizer '{power_iteration_normalizer}'")
    
    return Q
    xp, is_array_api_compliant = get_namespace(A)
    # 获取数组 A 的命名空间和是否符合数组 API 的标志

    random_state = check_random_state(random_state)
    # 根据 random_state 参数获取随机数生成器实例

    # 生成形状为 (A.shape[1], size) 的正态分布随机向量
    # XXX: 如果可能，未来可以直接从 xp 生成随机数。
    Q = xp.asarray(random_state.normal(size=(A.shape[1], size)))
    # 如果 A 有 dtype 属性且是实数浮点类型，则使用 float32 计算和组件
    if hasattr(A, "dtype") and xp.isdtype(A.dtype, kind="real floating"):
        Q = xp.astype(Q, A.dtype, copy=False)

    # 如果符合数组 API，将 Q 移动到设备（如 GPU），在必要时先转换为 float32 以避免在设备上分配不必要的内存。
    if is_array_api_compliant:
        Q = xp.asarray(Q, device=device(A))

    # 处理 "auto" 模式
    # 如果 power_iteration_normalizer 参数为 "auto"
    if power_iteration_normalizer == "auto":
        # 如果迭代次数 n_iter 小于等于 2
        if n_iter <= 2:
            # 设置 power_iteration_normalizer 为 "none"
            power_iteration_normalizer = "none"
        # 如果满足数组 API 兼容性条件
        elif is_array_api_compliant:
            # 发出警告，说明数组 API 不支持 LU 分解，将使用 QR 分解
            warnings.warn(
                "Array API does not support LU factorization, falling back to QR"
                " instead. Set `power_iteration_normalizer='QR'` explicitly to silence"
                " this warning."
            )
            # 设置 power_iteration_normalizer 为 "QR"
            power_iteration_normalizer = "QR"
        else:
            # 否则设置 power_iteration_normalizer 为 "LU"
            power_iteration_normalizer = "LU"
    
    # 如果 power_iteration_normalizer 参数为 "LU" 并且数组 API 兼容性条件满足
    elif power_iteration_normalizer == "LU" and is_array_api_compliant:
        # 抛出数值错误，说明数组 API 不支持 LU 分解，建议设置为使用 QR 分解
        raise ValueError(
            "Array API does not support LU factorization. Set "
            "`power_iteration_normalizer='QR'` instead."
        )

    # 如果满足数组 API 兼容性条件
    if is_array_api_compliant:
        # 使用 xp.linalg.qr 函数进行 QR 分解，返回一个偏函数 qr_normalizer
        qr_normalizer = partial(xp.linalg.qr, mode="reduced")
    else:
        # 否则使用 scipy.linalg.qr 函数进行 QR 分解，返回一个偏函数 qr_normalizer
        # 在没有显式使用数组 API 时使用 scipy.linalg 而非 numpy.linalg
        qr_normalizer = partial(linalg.qr, mode="economic", check_finite=False)

    # 根据 power_iteration_normalizer 参数选择正常化方法
    if power_iteration_normalizer == "QR":
        # 如果为 "QR"，使用 qr_normalizer 函数
        normalizer = qr_normalizer
    elif power_iteration_normalizer == "LU":
        # 如果为 "LU"，使用 scipy.linalg.lu 函数，并设置参数
        normalizer = partial(linalg.lu, permute_l=True, check_finite=False)
    else:
        # 否则使用一个匿名函数，返回输入向量及空值
        normalizer = lambda x: (x, None)

    # 执行 n_iter 次幂迭代，用于进一步“印记”矩阵 A 的顶部奇异向量在 Q 中
    for _ in range(n_iter):
        # 对 A @ Q 进行 normalizer 正常化操作，并返回结果给 Q
        Q, _ = normalizer(A @ Q)
        # 对 A.T @ Q 进行 normalizer 正常化操作，并返回结果给 Q
        Q, _ = normalizer(A.T @ Q)

    # 使用 Q 通过线性投影来抽样 A 的范围
    # 提取一个正交基
    Q, _ = qr_normalizer(A @ Q)

    # 返回 Q，作为函数结果
    return Q
# 使用装饰器 validate_params 对 randomized_svd 函数进行参数验证，确保参数类型和取值范围符合要求
@validate_params(
    {
        "M": [np.ndarray`
@validate_params(
    {
        "M": [np.ndarray, "sparse matrix"],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "n_oversamples": [Interval(Integral, 0, None, closed="left")],
        "n_iter": [Interval(Integral, 0, None, closed="left"), StrOptions({"auto"})],
        "power_iteration_normalizer": [StrOptions({"auto", "QR", "LU", "none"})],
        "transpose": ["boolean", StrOptions({"auto"})],
        "flip_sign": ["boolean"],
        "random_state": ["random_state"],
        "svd_lapack_driver": [StrOptions({"gesdd", "gesvd"})],
    },
    prefer_skip_nested_validation=True,
)
def randomized_svd(
    M,
    n_components,
    *,
    n_oversamples=10,
    n_iter="auto",
    power_iteration_normalizer="auto",
    transpose="auto",
    flip_sign=True,
    random_state=None,
    svd_lapack_driver="gesdd",
):
    """Compute a truncated randomized SVD.

    This method solves the fixed-rank approximation problem described in [1]_
    (problem (1.5), p5).

    Parameters
    ----------
    M : {ndarray, sparse matrix}
        Matrix to decompose.

    n_components : int
        Number of singular values and vectors to extract.

    n_oversamples : int, default=10
        Additional number of random vectors to sample the range of `M` so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of `M` is `n_components + n_oversamples`. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values. Users might wish
        to increase this parameter up to `2*k - n_components` where k is the
        effective rank, for large matrices, noisy problems, matrices with
        slowly decaying spectrums, or to increase precision accuracy. See [1]_
        (pages 5, 23 and 26).

    n_iter : int or 'auto', default='auto'
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) in which case `n_iter` is set to 7.
        This improves precision with few components. Note that in general
        users should rather increase `n_oversamples` before increasing `n_iter`
        as the principle of the randomized method is to avoid usage of these
        more costly power iterations steps. When `n_components` is equal
        or greater to the effective matrix rank and the spectrum does not
        present a slow decay, `n_iter=0` or `1` should even work fine in theory
        (see [1]_ page 9).

        .. versionchanged:: 0.18

    power_iteration_normalizer : str, {'auto', 'QR', 'LU', 'none'}, default='auto'
        Method to normalize the power iterations. 'auto' chooses between
        'QR' and 'LU' decompositions based on which is more efficient.

    transpose : bool or 'auto', default='auto'
        Whether to transpose the input matrix `M`. 'auto' determines this based
        on the shape of `M`.

    flip_sign : bool, default=True
        Whether to flip the sign of components. This is done to ensure
        consistency in the output and does not affect the accuracy.

    random_state : {None, int, numpy.random.RandomState}, default=None
        Determines random number generation for initializing random vectors.

    svd_lapack_driver : str, {'gesdd', 'gesvd'}, default='gesdd'
        Which LAPACK driver to use. 'gesdd' is generally faster and more robust
        for most scenarios.

    Raises
    ------
    ValueError
        If the input matrix `M` is not compatible with the expected types.

    Returns
    -------
    U : ndarray
        Left singular vectors.

    Sigma : ndarray
        Singular values.

    Vt : ndarray
        Right singular vectors.

    Notes
    -----
    This function provides a method for computing a truncated singular value
    decomposition using randomized algorithms. It is efficient for large-scale
    problems where the full decomposition is impractical.

    References
    ----------
    .. [1] Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure
           with randomness: Probabilistic algorithms for constructing approximate
           matrix decompositions. SIAM review, 53(2), 217-288.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from sklearn.utils.extmath import randomized_svd
    >>> M = np.random.rand(100, 100)
    >>> U, Sigma, Vt = randomized_svd(M, n_components=5)

    """
    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        # 定义参数 power_iteration_normalizer，表示幂迭代的规范化方式
        Whether the power iterations are normalized with step-by-step
        # 指示是否使用逐步QR分解来规范化幂迭代（最慢但最精确）
        QR factorization (the slowest but most accurate), 'none'
        # 'none' 表示不进行规范化（最快，但当 n_iter 较大时数值上不稳定）
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.

        .. versionadded:: 0.18
        # 说明此参数在版本 0.18 中被引入

    transpose : bool or 'auto', default='auto'
        # 定义参数 transpose，表示是否对 M 的转置进行运算
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        # 'auto' 模式下，如果 M.shape[1] > M.shape[0]，则自动对 M 进行转置
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.

        .. versionchanged:: 0.18
        # 说明此参数在版本 0.18 中发生了变化

    flip_sign : bool, default=True
        # 定义参数 flip_sign，表示是否翻转左奇异向量的符号以解决符号不确定性
        The output of a singular value decomposition is only unique up to a
        permutation of the signs of the singular vectors. If `flip_sign` is
        set to `True`, the sign ambiguity is resolved by making the largest
        loadings for each component in the left singular vectors positive.

    random_state : int, RandomState instance or None, default='warn'
        # 定义参数 random_state，表示用于初始化随机向量的伪随机数生成器的种子
        The seed of the pseudo random number generator to use when
        shuffling the data, i.e. getting the random vectors to initialize
        the algorithm. Pass an int for reproducible results across multiple
        function calls. See :term:`Glossary <random_state>`.

        .. versionchanged:: 1.2
            The default value changed from 0 to None.
        # 说明此参数在版本 1.2 中发生了变化，默认值从 0 变为 None

    svd_lapack_driver : {"gesdd", "gesvd"}, default="gesdd"
        # 定义参数 svd_lapack_driver，表示使用的SVD计算方法选择
        Whether to use the more efficient divide-and-conquer approach
        (`"gesdd"`) or more general rectangular approach (`"gesvd"`) to compute
        the SVD of the matrix B, which is the projection of M into a low
        dimensional subspace, as described in [1]_.

        .. versionadded:: 1.2
        # 说明此参数在版本 1.2 中被引入

    Returns
    -------
    u : ndarray of shape (n_samples, n_components)
        # 返回值 u，表示具有翻转符号的左奇异向量组成的单位矩阵
        Unitary matrix having left singular vectors with signs flipped as columns.
    s : ndarray of shape (n_components,)
        # 返回值 s，表示奇异值，按非增序排列
        The singular values, sorted in non-increasing order.
    vh : ndarray of shape (n_components, n_features)
        # 返回值 vh，表示具有翻转符号的右奇异向量组成的单位矩阵
        Unitary matrix having right singular vectors with signs flipped as rows.

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision). To increase the precision it is recommended to
    increase `n_oversamples`, up to `2*k-n_components` where k is the
    # 算法通过随机化加速计算，找到（通常非常好的）截断奇异值分解的近似解。
    # 在大矩阵上特别快速，适用于只需提取少量主成分的情况。
    # 为了进一步加速，`n_iter` 可以设置为 <=2（代价是精度损失）。
    # 增加 `n_oversamples` 可以增加精度，最多可增加到 `2*k-n_components`，其中 k 是
    # ...
    if sparse.issparse(M) and M.format in ("lil", "dok"):
        # 如果输入的矩阵 M 是稀疏矩阵，并且格式是 lil 或 dok，则发出警告
        warnings.warn(
            "Calculating SVD of a {} is expensive. "
            "csr_matrix is more efficient.".format(type(M).__name__),
            sparse.SparseEfficiencyWarning,
        )

    random_state = check_random_state(random_state)
    # 确保随机数生成器处于可用状态
    n_random = n_components + n_oversamples
    # 计算随机向量的数量，包括主成分数和额外采样数
    n_samples, n_features = M.shape
    # 获取矩阵 M 的样本数和特征数

    if n_iter == "auto":
        # 如果未显式指定迭代次数
        # 调整 n_iter。对于 PCA，经验上发现 7 是一个良好的折衷值。参见 issue #5299
        n_iter = 7 if n_components < 0.1 * min(M.shape) else 4

    if transpose == "auto":
        # 如果未显式指定是否转置
        transpose = n_samples < n_features
        # 根据样本数和特征数判断是否进行转置操作
    if transpose:
        # 如果需要进行转置
        # 对于较小的 shape[1]，这种实现稍微快一些
        M = M.T

    Q = randomized_range_finder(
        M,
        size=n_random,
        n_iter=n_iter,
        power_iteration_normalizer=power_iteration_normalizer,
        random_state=random_state,
    )
    # 使用随机范围查找器计算基于随机向量的基础向量集合 Q

    # 使用基础向量集合 Q 投影 M 到 (k + p) 维空间
    B = Q.T @ M

    # 在细矩阵上执行 SVD：宽度为 (k + p)
    xp, is_array_api_compliant = get_namespace(B)
    if is_array_api_compliant:
        # 如果数组兼容 API
        Uhat, s, Vt = xp.linalg.svd(B, full_matrices=False)
    else:
        # 当 array_api_dispatch 禁用时，依赖 scipy.linalg 而不是 numpy.linalg
        # 以避免引入与 scikit-learn 之前版本行为不一致的变化
        Uhat, s, Vt = linalg.svd(
            B, full_matrices=False, lapack_driver=svd_lapack_driver
        )
    del B
    # 删除临时变量 B

    U = Q @ Uhat
    # 计算最终的主成分 U

    if flip_sign:
        # 如果需要翻转符号
        if not transpose:
            # 如果没有进行转置，则根据 U 和 Vt 翻转
            U, Vt = svd_flip(U, Vt)
        else:
            # 如果进行了转置，则基于 U 而不是 Vt 进行翻转决策
            U, Vt = svd_flip(U, Vt, u_based_decision=False)
    # 如果 transpose 参数为真，则按照输入约定对结果进行转置
    if transpose:
        # 返回转置后的 Vt 的前 n_components 行，s 的前 n_components 个元素，以及转置后的 U 的前 n_components 列
        return Vt[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        # 否则返回 U 的前 n_components 列，s 的前 n_components 个元素，以及 Vt 的前 n_components 行
        return U[:, :n_components], s[:n_components], Vt[:n_components, :]
def _randomized_eigsh(
    M,
    n_components,
    *,
    n_oversamples=10,
    n_iter="auto",
    power_iteration_normalizer="auto",
    selection="module",
    random_state=None,
):
    """Computes a truncated eigendecomposition using randomized methods

    This method solves the fixed-rank approximation problem described in the
    Halko et al paper.

    The choice of which components to select can be tuned with the `selection`
    parameter.

    .. versionadded:: 0.24

    Parameters
    ----------
    M : ndarray or sparse matrix
        Matrix to decompose, it should be real symmetric square or complex
        hermitian

    n_components : int
        Number of eigenvalues and vectors to extract.

    n_oversamples : int, default=10
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of eigenvectors and eigenvalues. Users might wish
        to increase this parameter up to `2*k - n_components` where k is the
        effective rank, for large matrices, noisy problems, matrices with
        slowly decaying spectrums, or to increase precision accuracy. See Halko
        et al (pages 5, 23 and 26).

    n_iter : int or 'auto', default='auto'
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) in which case `n_iter` is set to 7.
        This improves precision with few components. Note that in general
        users should rather increase `n_oversamples` before increasing `n_iter`
        as the principle of the randomized method is to avoid usage of these
        more costly power iterations steps. When `n_components` is equal
        or greater to the effective matrix rank and the spectrum does not
        present a slow decay, `n_iter=0` or `1` should even work fine in theory
        (see Halko et al paper, page 9).

    power_iteration_normalizer : {'auto', 'QR', 'LU', 'none'}, default='auto'
        Whether the power iterations are normalized with step-by-step
        QR factorization (the slowest but most accurate), 'none'
        (the fastest but numerically unstable when `n_iter` is large, e.g.
        typically 5 or larger), or 'LU' factorization (numerically stable
        but can lose slightly in accuracy). The 'auto' mode applies no
        normalization if `n_iter` <= 2 and switches to LU otherwise.
    """
    # 实现使用随机方法的截断特征分解

    # 解决在 Halko 等人的论文中描述的固定秩逼近问题

    # 可以通过 `selection` 参数调整选择哪些成分

    # 添加版本 0.24 中的新增内容

    # 输入参数解释：

    # M：ndarray 或稀疏矩阵
    #     要分解的矩阵，应为实对称方阵或复共轭的 Hermitian 矩阵

    # n_components：int
    #     要提取的特征值和特征向量的数量

    # n_oversamples：int，默认为 10
    #     采样 M 的范围所需的额外随机向量数量，以确保适当的条件数。用于找到 M 范围的总随机向量数量为 n_components + n_oversamples。
    #     较小的数字可以提高速度，但可能会对特征向量和特征值的近似质量产生负面影响。
    #     用户可能希望将此参数增加到 `2*k - n_components`，其中 k 是有效秩，适用于大矩阵、噪声问题、特征谱缓慢衰减的矩阵或提高精度的情况。
    #     参见 Halko 等人的论文（第 5、23 和 26 页）。

    # n_iter：int 或 'auto'，默认为 'auto'
    #     幂迭代的次数。可用于处理非常嘈杂的问题。当设置为 'auto' 时，如果 `n_components` 较小（< .1 * min(X.shape)），
    #     则将其设置为 7，否则设置为 4。这可以提高在少量成分的情况下的精度。
    #     一般来说，用户应该在增加 `n_iter` 之前增加 `n_oversamples`，因为随机方法的原则是避免使用这些更昂贵的幂迭代步骤。
    #     当 `n_components` 等于或大于有效矩阵秩，并且频谱不显示缓慢衰减时，理论上 `n_iter=0` 或 `1` 甚至可以工作得很好（参见 Halko 等人的论文，第 9 页）。

    # power_iteration_normalizer：{'auto', 'QR', 'LU', 'none'}，默认为 'auto'
    #     幂迭代是否通过逐步 QR 分解（最慢但最精确）、'none'（当 `n_iter` 较大时最快但数值上不稳定，例如通常为 5 或更大）或 'LU' 分解（数值上稳定但可能略微损失精度）来进行标准化。
    #     'auto' 模式如果 `n_iter` <= 2 则不应用标准化，并在其他情况下切换到 LU。
    selection : {'value', 'module'}, default='module'
        # 参数 `selection` 控制选择哪种策略来获取特征分解的组件。当 `selection` 为 `'value'` 时（尚未实现，将来可能成为默认选项），返回对应于最大特征值的 n 个组件。
        # 当 `selection` 为 `'module'` 时，返回对应于模最大的 n 个特征值的组件。

    random_state : int, RandomState instance, default=None
        # 用于初始化伪随机数生成器的种子，用于对数据进行洗牌，即获取随机向量来初始化算法。
        # 传递一个整数可以确保在多次函数调用中产生可重复的结果。参见“术语表”。

    Notes
    -----
    # 此算法使用随机方法来找到一个（通常非常好的）近似截断的特征分解，以加快计算速度。

    # 这种方法在希望从大型矩阵中提取少量组件时特别快速。为了进一步加快速度，可以将 `n_iter` 设置为 <=2（以损失精度为代价）。
    # 要增加精度，建议增加 `n_oversamples`，直到 `2*k-n_components`，其中 k 是有效秩。
    # 通常 `n_components` 应选择大于 k，因此增加 `n_oversamples` 到 `n_components` 应足够。

    Strategy 'value': not implemented yet.
        # 策略 'value'：尚未实现。
        # Halko 等人的文献中的算法 5.3、5.4 和 5.5 应该是未来实现的良好候选方案。

    Strategy 'module':
        # 策略 'module'：
        # 原理是对角化矩阵的奇异值和特征值相关联：如果 t 是 A 的特征值，则 |t| 是 A 的奇异值。
        # 此方法依赖于随机化的奇异值分解来找到与模最大的 n 个奇异值对应的 n 个奇异组件，并使用奇异向量的符号来确定 t 的真实符号：如果左右奇异向量的符号不同，则相应的特征值为负数。

    Returns
    -------
    eigvals : 1D array of shape (n_components,) containing the `n_components`
        # 包含选择的 `n_components` 特征值的一维数组（见 `selection` 参数）。

    eigvecs : 2D array of shape (M.shape[0], n_components) containing the
        # 包含与 `eigvals` 对应的 `n_components` 特征向量的二维数组，按相应顺序排列。
        # 注意，这遵循 `scipy.linalg.eigh` 的约定。

    See Also
    --------
    :func:`randomized_svd`

    References
    ----------
    * :arxiv:`"Finding structure with randomness:
      Stochastic algorithms for constructing approximate matrix decompositions"
      (Algorithm 4.3 for strategy 'module') <0909.4061>`
      Halko, et al. (2009)
    """
    if selection == "value":  # pragma: no cover
        # 待完成：可以在 Halko 等人的文献中找到一个算法。
        # 抛出未实现错误，因为尚未实现对 'value' 策略的支持。
        raise NotImplementedError()
    elif selection == "module":
        # 如果选择是 "module"，执行以下操作：

        # 使用随机化SVD分解矩阵M，返回左奇异向量(U)，奇异值(S)，右奇异向量的转置(Vt)
        U, S, Vt = randomized_svd(
            M,
            n_components=n_components,
            n_oversamples=n_oversamples,
            n_iter=n_iter,
            power_iteration_normalizer=power_iteration_normalizer,
            flip_sign=False,  # 不需要确定性的U和Vt（flip_sign=True），因为之后只使用UVt的点积
            random_state=random_state,
        )

        # 提取前n_components列的左奇异向量作为特征向量（eigvecs）
        eigvecs = U[:, :n_components]

        # 提取前n_components个奇异值作为特征值（eigvals）
        eigvals = S[:n_components]

        # 将奇异值转换为特征值的形式：
        # 对于任意特征值t，对应的奇异值是|t|。
        # 因此，如果有一个负的特征值t，对应的奇异值将是-t，
        # 并且左奇异向量（U）和右奇异向量（V）将具有相反的符号。
        # 最快的方法参见：<https://stackoverflow.com/a/61974002/7262247>
        diag_VtU = np.einsum("ji,ij->j", Vt[:n_components, :], U[:, :n_components])
        signs = np.sign(diag_VtU)
        eigvals = eigvals * signs  # 根据奇异值的符号调整特征值的符号

    else:  # pragma: no cover
        # 如果选择不是 "module"，抛出异常
        raise ValueError("Invalid `selection`: %r" % selection)

    # 返回特征值和特征向量
    return eigvals, eigvecs
def weighted_mode(a, w, *, axis=0):
    """Return an array of the weighted modal (most common) value in the passed array.

    If there is more than one such value, only the first is returned.
    The bin-count for the modal bins is also returned.

    This is an extension of the algorithm in scipy.stats.mode.

    Parameters
    ----------
    a : array-like of shape (n_samples,)
        Array of which values to find mode(s).
    w : array-like of shape (n_samples,)
        Array of weights for each value.
    axis : int, default=0
        Axis along which to operate. Default is 0, i.e. the first axis.

    Returns
    -------
    vals : ndarray
        Array of modal values.
    score : ndarray
        Array of weighted counts for each mode.

    See Also
    --------
    scipy.stats.mode: Calculates the Modal (most common) value of array elements
        along specified axis.

    Examples
    --------
    >>> from sklearn.utils.extmath import weighted_mode
    >>> x = [4, 1, 4, 2, 4, 2]
    >>> weights = [1, 1, 1, 1, 1, 1]
    >>> weighted_mode(x, weights)
    (array([4.]), array([3.]))

    The value 4 appears three times: with uniform weights, the result is
    simply the mode of the distribution.

    >>> weights = [1, 3, 0.5, 1.5, 1, 2]  # deweight the 4's
    >>> weighted_mode(x, weights)
    (array([2.]), array([3.5]))

    The value 2 has the highest score: it appears twice with weights of
    1.5 and 2: the sum of these is 3.5.
    """
    # If axis is None, flatten both 'a' and 'w' arrays and set axis to 0
    if axis is None:
        a = np.ravel(a)
        w = np.ravel(w)
        axis = 0
    else:
        # Convert 'a' and 'w' to numpy arrays
        a = np.asarray(a)
        w = np.asarray(w)

    # If shapes of 'a' and 'w' arrays are not identical, broadcast 'w' to match 'a'
    if a.shape != w.shape:
        w = np.full(a.shape, w, dtype=w.dtype)

    # Get all unique values in 'a' array
    scores = np.unique(np.ravel(a))
    # Create a template array of zeros with shape of 'a'
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)
    # Iterate through unique scores
    for score in scores:
        # Create a template array of zeros and update it where 'a' equals 'score'
        template = np.zeros(a.shape)
        ind = a == score
        template[ind] = w[ind]
        # Compute sums along specified axis and expand dimensions
        counts = np.expand_dims(np.sum(template, axis), axis)
        # Update 'mostfrequent' where 'counts' is greater than 'oldcounts'
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent
    # Return the most frequent value(s) and their counts
    return mostfrequent, oldcounts
    # 将输入的数组转换为 NumPy 数组的列表
    arrays = [np.asarray(x) for x in arrays]
    
    # 获取每个数组的长度，组成一个生成器对象
    shape = (len(x) for x in arrays)
    
    # 使用 np.indices 函数生成一个索引数组，形状由 shape 定义
    ix = np.indices(shape)
    
    # 将索引数组重新形状为二维数组，行数为数组个数，列数根据数组长度决定
    ix = ix.reshape(len(arrays), -1).T
    
    # 如果输出数组 out 为 None，则根据数组的数据类型创建一个同样形状的空数组
    if out is None:
        dtype = np.result_type(*arrays)  # 找到最宽松的数据类型以便兼容所有输入数组
        out = np.empty_like(ix, dtype=dtype)
    
    # 遍历数组列表，根据索引数组 ix 逐列填充输出数组 out
    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]
    
    # 返回填充好的输出数组
    return out
# 对 SVD 结果进行翻转以确保确定性输出

def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    If u_based_decision is False, then the same sign correction is applied to
    so that the rows in v that are largest in absolute value are always
    positive.

    Parameters
    ----------
    u : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
        u can be None if `u_based_decision` is False.

    v : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`. The input v should
        really be called vt to be consistent with scipy's output.
        v can be None if `u_based_decision` is True.

    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u_adjusted : ndarray
        Array u with adjusted columns and the same dimensions as u.

    v_adjusted : ndarray
        Array v with adjusted rows and the same dimensions as v.
    """
    # 获取计算所需的运算库和设备信息
    xp, _ = get_namespace(*[a for a in [u, v] if a is not None])

    if u_based_decision:
        # 如果基于 u 进行决策，则调整 u 的列和 v 的行
        # 找到 u.T 每列中绝对值最大的元素的索引
        max_abs_u_cols = xp.argmax(xp.abs(u.T), axis=1)
        # 生成一个与 u.T 形状相同的设备数组
        shift = xp.arange(u.T.shape[0], device=device(u))
        # 计算要修改的元素的索引
        indices = max_abs_u_cols + shift * u.T.shape[1]
        # 获取这些元素的符号
        signs = xp.sign(xp.take(xp.reshape(u.T, (-1,)), indices, axis=0))
        # 对 u 进行符号修正
        u *= signs[np.newaxis, :]
        # 如果 v 不为 None，则对 v 进行符号修正
        if v is not None:
            v *= signs[:, np.newaxis]
    else:
        # 如果基于 v 进行决策，则调整 v 的行和 u 的列
        # 找到 v 每行中绝对值最大的元素的索引
        max_abs_v_rows = xp.argmax(xp.abs(v), axis=1)
        # 生成一个与 v 形状相同的设备数组
        shift = xp.arange(v.shape[0], device=device(v))
        # 计算要修改的元素的索引
        indices = max_abs_v_rows + shift * v.shape[1]
        # 获取这些元素的符号
        signs = xp.sign(xp.take(xp.reshape(v, (-1,)), indices, axis=0))
        # 如果 u 不为 None，则对 u 进行符号修正
        if u is not None:
            u *= signs[np.newaxis, :]
        # 对 v 进行符号修正
        v *= signs[:, np.newaxis]
    return u, v


# TODO(1.6): remove
@deprecated(  # type: ignore
    "The function `log_logistic` is deprecated and will be removed in 1.6. "
    "Use `-np.logaddexp(0, -x)` instead."
)
def log_logistic(X, out=None):
    """Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.

    This implementation is numerically stable and uses `-np.logaddexp(0, -x)`.

    For the ordinary logistic function, use ``scipy.special.expit``.

    Parameters
    ----------
    X : array-like of shape (M, N) or (M,)
        Argument to the logistic function.
    """
    # 计算 logistic 函数的对数，使用稳定的数值方法 `-np.logaddexp(0, -x)`
    # 对于普通的 logistic 函数，使用 `scipy.special.expit`

    This implementation is numerically stable and uses `-np.logaddexp(0, -x)`.

    For the ordinary logistic function, use ``scipy.special.expit``.

    Parameters
    ----------
    X : array-like of shape (M, N) or (M,)
        Argument to the logistic function.
    """
    # 计算 logistic 函数的对数，使用稳定的数值方法 `-np.logaddexp(0, -x)`
    # 对于普通的 logistic 函数，使用 `scipy.special.expit`
    X = check_array(X, dtype=np.float64, ensure_2d=False)

# 使用 check_array 函数检查并转换输入数组 X 的数据类型为 np.float64，并确保其为二维数组或更高维度。


    if out is None:
        out = np.empty_like(X)

# 如果参数 out 为 None，则创建一个形状与 X 相同的空数组 out。


    np.logaddexp(0, -X, out=out)

# 计算元素级别的 logistic 函数的对数，使用 np.logaddexp 函数，将结果存储在 out 数组中。


    out *= -1

# 将 out 数组中的每个元素乘以 -1，以获得 logistic 函数的对数的负值。


    return out

# 返回计算结果数组 out，其中包含每个输入元素的 logistic 函数的对数的负值。
# 定义 softmax 函数，用于计算 softmax 激活函数
def softmax(X, copy=True):
    """
    Calculate the softmax function.

    The softmax function is calculated by
    np.exp(X) / np.sum(np.exp(X), axis=1)

    This will cause overflow when large values are exponentiated.
    Hence the largest value in each row is subtracted from each data
    point to prevent this.

    Parameters
    ----------
    X : array-like of float of shape (M, N)
        Argument to the logistic function.

    copy : bool, default=True
        Copy X or not.

    Returns
    -------
    out : ndarray of shape (M, N)
        Softmax function evaluated at every point in x.
    """
    # 获取适当的数组命名空间和其数组 API 兼容性信息
    xp, is_array_api_compliant = get_namespace(X)
    # 如果需要复制输入数据，则进行复制
    if copy:
        X = xp.asarray(X, copy=True)
    # 计算每行的最大值，用于处理指数函数溢出问题
    max_prob = xp.reshape(xp.max(X, axis=1), (-1, 1))
    X -= max_prob

    if _is_numpy_namespace(xp):
        # 优化处理 NumPy 数组的指数计算
        np.exp(X, out=np.asarray(X))
    else:
        # 对于不支持 `out=` 的数组 API，使用 exp 函数计算
        X = xp.exp(X)

    # 计算每行的 softmax 分母，确保分母不会为零
    sum_prob = xp.reshape(xp.sum(X, axis=1), (-1, 1))
    X /= sum_prob
    return X


# 确保输入数组的最小值不小于给定的阈值，使其非负化
def make_nonnegative(X, min_value=0):
    """Ensure `X.min()` >= `min_value`.

    Parameters
    ----------
    X : array-like
        The matrix to make non-negative.
    min_value : float, default=0
        The threshold value.

    Returns
    -------
    array-like
        The thresholded array.

    Raises
    ------
    ValueError
        When X is sparse.
    """
    # 获取数组 X 的最小值
    min_ = X.min()
    # 如果最小值小于设定的阈值，则进行调整
    if min_ < min_value:
        # 如果 X 是稀疏矩阵，抛出数值错误
        if sparse.issparse(X):
            raise ValueError(
                "Cannot make the data matrix"
                " nonnegative because it is sparse."
                " Adding a value to every entry would"
                " make it no longer sparse."
            )
        # 将 X 中所有元素加上调整值，使得最小值达到阈值
        X = X + (min_value - min_)
    return X


# 针对浮点输入，为 numpy 累加器函数提供 float64 类型，以避免精度问题
# 参考：https://github.com/numpy/numpy/issues/9393
def _safe_accumulator_op(op, x, *args, **kwargs):
    """
    This function provides numpy accumulator functions with a float64 dtype
    when used on a floating point input. This prevents accumulator overflow on
    smaller floating point dtypes.

    Parameters
    ----------
    op : function
        A numpy accumulator function such as np.mean or np.sum.
    x : ndarray
        A numpy array to apply the accumulator function.
    *args : positional arguments
        Positional arguments passed to the accumulator function after the
        input x.
    **kwargs : keyword arguments
        Keyword arguments passed to the accumulator function.

    Returns
    -------
    result
        The output of the accumulator function passed to this function.
    """
    # 如果输入 x 是浮点类型且其数据类型大小小于 8 字节，则使用 float64 类型进行计算
    if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
        result = op(x, *args, **kwargs, dtype=np.float64)
    else:
        result = op(x, *args, **kwargs)
    return result


def _incremental_mean_and_var(
    X, last_mean, last_variance, last_sample_count, sample_weight=None


    # 定义函数的参数列表，包括 X，上一次计算的均值、方差、样本计数及样本权重，默认为 None
    # X: 数据集
    # last_mean: 上一次计算的均值
    # last_variance: 上一次计算的方差
    # last_sample_count: 上一次计算的样本计数
    # sample_weight: 可选参数，用于加权计算的权重，默认为 None
# old = stats until now
# new = the current increment
# updated = the aggregated stats
last_sum = last_mean * last_sample_count
X_nan_mask = np.isnan(X)  # 创建一个布尔数组，表示 X 中的 NaN 值位置
if np.any(X_nan_mask):  # 检查 X 是否包含 NaN 值
    sum_op = np.nansum  # 如果存在 NaN 值，使用 np.nansum 处理求和操作
else:
    sum_op = np.sum  # 如果不存在 NaN 值，使用 np.sum 处理求和操作
if sample_weight is not None:
    # 如果指定了 sample_weight，则进行加权求和操作
    # 等价于 np.nansum(X * sample_weight, axis=0)，通过矩阵乘法计算加权和，处理 NaN 值
    # 更安全，因为 np.float64(X*W) != np.float64(X)*np.float64(W)
    new_sum = _safe_accumulator_op(
        np.matmul, sample_weight, np.where(X_nan_mask, 0, X)
    )
    # 计算新的样本数，考虑了 NaN 值对应的权重
    new_sample_count = _safe_accumulator_op(
        np.sum, sample_weight[:, None] * (~X_nan_mask), axis=0
    )
else:
    # 如果未指定 sample_weight，则进行普通求和操作
    new_sum = _safe_accumulator_op(sum_op, X, axis=0)
    n_samples = X.shape[0]  # 获取样本数量
    # 计算新的样本数，减去了每列中的 NaN 值的数量
    new_sample_count = n_samples - np.sum(X_nan_mask, axis=0)

updated_sample_count = last_sample_count + new_sample_count  # 更新样本数统计

updated_mean = (last_sum + new_sum) / updated_sample_count  # 更新均值统计

if last_variance is None:
    updated_variance = None  # 如果之前没有计算过方差，则更新方差为 None
    else:
        # 计算新的均值 T
        T = new_sum / new_sample_count
        # 计算偏差 temp
        temp = X - T
        if sample_weight is not None:
            # 如果存在样本权重，则计算加权平方差
            # 相当于 np.nansum((X-T)**2 * sample_weight, axis=0)
            # 使用 _safe_accumulator_op 来避免精度问题，因为 np.float64(X*W) != np.float64(X)*np.float64(W)
            correction = _safe_accumulator_op(
                np.matmul, sample_weight, np.where(X_nan_mask, 0, temp)
            )
            # 计算 temp 的平方
            temp **= 2
            # 计算新的非归一化方差
            new_unnormalized_variance = _safe_accumulator_op(
                np.matmul, sample_weight, np.where(X_nan_mask, 0, temp)
            )
        else:
            # 如果没有样本权重，则直接使用 sum_op 计算加和
            correction = _safe_accumulator_op(sum_op, temp, axis=0)
            # 计算 temp 的平方
            temp **= 2
            # 计算新的非归一化方差
            new_unnormalized_variance = _safe_accumulator_op(sum_op, temp, axis=0)

        # 计算修正的二次传递算法的修正项
        # 参见 "Algorithms for computing the sample variance: analysis
        # and recommendations", by Chan, Golub, and LeVeque.
        new_unnormalized_variance -= correction**2 / new_sample_count

        # 计算上次方差的非归一化形式
        last_unnormalized_variance = last_variance * last_sample_count

        # 使用 np.errstate 进行错误处理，忽略除以零和无效操作的警告
        with np.errstate(divide="ignore", invalid="ignore"):
            # 计算上次样本数与新样本数的比率
            last_over_new_count = last_sample_count / new_sample_count
            # 更新非归一化方差
            updated_unnormalized_variance = (
                last_unnormalized_variance
                + new_unnormalized_variance
                + last_over_new_count
                / updated_sample_count
                * (last_sum / last_over_new_count - new_sum) ** 2
            )

        # 将上一次样本数为零的位置的更新非归一化方差设置为新的非归一化方差
        zeros = last_sample_count == 0
        updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
        # 计算最终更新后的方差
        updated_variance = updated_unnormalized_variance / updated_sample_count

    # 返回更新后的均值、方差和样本数
    return updated_mean, updated_variance, updated_sample_count
# 修改向量的符号以确保结果的可复现性
def _deterministic_vector_sign_flip(u):
    # 找到每个向量中绝对值最大的元素的索引
    max_abs_rows = np.argmax(np.abs(u), axis=1)
    # 获取这些元素的符号
    signs = np.sign(u[range(u.shape[0]), max_abs_rows])
    # 将每个向量按照符号进行符号翻转
    u *= signs[:, np.newaxis]
    # 返回符号翻转后的向量数组
    return u


# 计算累积和时使用高精度，并检查最终值是否与预期的总和匹配
def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    # 计算累积和，使用 float64 类型
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    # 计算预期的总和
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    # 如果累积和的最后一个元素与预期总和不匹配（在指定的容差范围内），发出警告
    if not np.allclose(
        out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
    ):
        warnings.warn(
            (
                "cumsum was found to be unstable: "
                "its last element does not correspond to sum"
            ),
            RuntimeWarning,
        )
    # 返回累积和数组
    return out


# 计算带权重的平均值，忽略 NaN 值
def _nanaverage(a, weights=None):
    # 如果数组为空，返回 NaN
    if len(a) == 0:
        return np.nan
    
    # 创建 NaN 掩码
    mask = np.isnan(a)
    # 如果所有值都是 NaN，则返回 NaN
    if mask.all():
        return np.nan
    
    # 如果未提供权重，则使用 np.nanmean 计算平均值
    if weights is None:
        return np.nanmean(a)
    
    # 否则，使用提供的权重进行加权平均
    weights = np.asarray(weights)
    a, weights = a[~mask], weights[~mask]
    try:
        return np.average(a, weights=weights)
    ```
    except ZeroDivisionError:
        # 处理除零错误异常，通常发生在所有权重为零的情况下，这时候忽略它们
        return np.average(a)
# 对输入的数组或稀疏矩阵进行逐元素平方操作，支持就地计算或复制操作。
def safe_sqr(X, *, copy=True):
    """Element wise squaring of array-likes and sparse matrices.

    Parameters
    ----------
    X : {array-like, ndarray, sparse matrix}
        输入的数组或稀疏矩阵。

    copy : bool, default=True
        是否创建输入的副本进行操作，或者进行原地计算（默认行为）。

    Returns
    -------
    X ** 2 : element wise square
        返回输入的逐元素平方结果。

    Examples
    --------
    >>> from sklearn.utils import safe_sqr
    >>> safe_sqr([1, 2, 3])
    array([1, 4, 9])
    """
    # 检查并转换输入的数组或稀疏矩阵为接受的格式（稀疏矩阵类型为 "csr", "csc", "coo"）
    X = check_array(X, accept_sparse=["csr", "csc", "coo"], ensure_2d=False)
    
    # 如果输入是稀疏矩阵
    if sparse.issparse(X):
        # 如果选择复制操作
        if copy:
            X = X.copy()  # 复制稀疏矩阵
        X.data **= 2  # 对稀疏矩阵的数据进行逐元素平方
    else:
        # 如果选择复制操作
        if copy:
            X = X**2  # 创建输入的副本并进行平方操作
        else:
            X **= 2  # 原地逐元素平方操作
    
    # 返回计算结果
    return X


def _approximate_mode(class_counts, n_draws, rng):
    """Computes approximate mode of multivariate hypergeometric.

    This is an approximation to the mode of the multivariate
    hypergeometric given by class_counts and n_draws.
    It shouldn't be off by more than one.

    It is the mostly likely outcome of drawing n_draws many
    samples from the population given by class_counts.

    Parameters
    ----------
    class_counts : ndarray of int
        Population per class.

    n_draws : int
        Number of draws (samples to draw) from the overall population.

    rng : random state
        Used to break ties.

    Returns
    -------
    sampled_classes : ndarray of int
        Number of samples drawn from each class.
        np.sum(sampled_classes) == n_draws

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.extmath import _approximate_mode
    >>> _approximate_mode(class_counts=np.array([4, 2]), n_draws=3, rng=0)
    array([2, 1])
    >>> _approximate_mode(class_counts=np.array([5, 2]), n_draws=4, rng=0)
    array([3, 1])
    >>> _approximate_mode(class_counts=np.array([2, 2, 2, 1]),
    ...                   n_draws=2, rng=0)
    array([0, 1, 1, 0])
    >>> _approximate_mode(class_counts=np.array([2, 2, 2, 1]),
    ...                   n_draws=2, rng=42)
    array([1, 1, 0, 0])
    """
    rng = check_random_state(rng)
    
    # 这里计算的是对多变量超几何分布的估计模式
    continuous = class_counts / class_counts.sum() * n_draws
    # floored 表示我们不会超过 n_draws，但可能会不够
    floored = np.floor(continuous)
    # 根据它们剩余的概率添加样本，直到达到 n_draws
    need_to_add = int(n_draws - floored.sum())
    # 如果需要添加的数量大于0，则进入以下逻辑
    if need_to_add > 0:
        # 计算 continuous 与 floored 的差值 remainder
        remainder = continuous - floored
        # 对 remainder 进行排序并去重，然后逆序排列
        values = np.sort(np.unique(remainder))[::-1]
        
        # 根据 remainder 添加值，但是为了避免偏差，随机打破平局
        for value in values:
            # 找到 remainder 中值为 value 的索引
            (inds,) = np.where(remainder == value)
            
            # 如果需要添加的数量小于等于 inds 中的元素数量，
            # 则从中随机选择需要添加的数量
            # 如果需要添加的数量大于 inds 中的元素数量，
            # 则全部添加并进入下一个值的处理
            add_now = min(len(inds), need_to_add)
            inds = rng.choice(inds, size=add_now, replace=False)
            
            # 将 floored 中选定的索引处的值加一
            floored[inds] += 1
            # 减去已经添加的数量
            need_to_add -= add_now
            
            # 如果已经添加的数量达到需要添加的总量，跳出循环
            if need_to_add == 0:
                break
    
    # 将 floored 转换为整数类型并返回
    return floored.astype(int)
```