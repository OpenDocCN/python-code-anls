# `D:\src\scipysrc\scikit-learn\sklearn\manifold\_locally_linear.py`

```
# 导入必要的模块和函数
from numbers import Integral, Real  # 导入整数和实数类型
import numpy as np  # 导入 NumPy 库
from scipy.linalg import eigh, qr, solve, svd  # 导入线性代数相关函数
from scipy.sparse import csr_matrix, eye, lil_matrix  # 导入稀疏矩阵相关函数
from scipy.sparse.linalg import eigsh  # 导入稀疏矩阵特征值计算函数

# 导入模块和函数
from ..base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
    _UnstableArchMixin,
)
from ..neighbors import NearestNeighbors  # 导入最近邻模块中的 NearestNeighbors 类
from ..utils import check_array, check_random_state  # 导入数据验证和随机状态检查函数
from ..utils._arpack import _init_arpack_v0  # 导入 ARPACK 相关函数
from ..utils._param_validation import Interval, StrOptions, validate_params  # 导入参数验证函数
from ..utils.extmath import stable_cumsum  # 导入数学扩展函数
from ..utils.validation import FLOAT_DTYPES, check_is_fitted  # 导入数据类型验证和拟合状态检查函数


def barycenter_weights(X, Y, indices, reg=1e-3):
    """计算从 Y 到 X 的重心权重

    我们估计每个 Y[indices] 中的点分配给 X[i] 的权重。重心权重总和为 1。

    Parameters
    ----------
    X : array-like, shape (n_samples, n_dim)
        目标数据集

    Y : array-like, shape (n_samples, n_dim)
        源数据集

    indices : array-like, shape (n_samples, n_dim)
        用于计算重心的 Y 中的点的索引

    reg : float, default=1e-3
        正则化参数，用于处理 n_neighbors > n_dim 时的问题

    Returns
    -------
    B : array-like, shape (n_samples, n_neighbors)
        计算得到的重心权重矩阵

    Notes
    -----
    详见开发者文档获取更多信息。
    """
    X = check_array(X, dtype=FLOAT_DTYPES)  # 检查并转换 X 的数据类型
    Y = check_array(Y, dtype=FLOAT_DTYPES)  # 检查并转换 Y 的数据类型
    indices = check_array(indices, dtype=int)  # 检查并转换 indices 的数据类型为整数类型

    n_samples, n_neighbors = indices.shape  # 获取 indices 的形状信息
    assert X.shape[0] == n_samples  # 断言 X 的行数与样本数相等

    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)  # 创建一个空的数组 B，用于存储结果
    v = np.ones(n_neighbors, dtype=X.dtype)  # 创建一个全为 1 的向量 v

    # 如果 G 是奇异矩阵并且迹为零，可能会引发 LinalgError
    for i, ind in enumerate(indices):
        A = Y[ind]  # 从 Y 中取出索引为 ind 的数据赋值给 A
        C = A - X[i]  # 计算偏移量 C，利用广播特性
        G = np.dot(C, C.T)  # 计算协方差矩阵 G
        trace = np.trace(G)  # 计算 G 的迹
        if trace > 0:
            R = reg * trace  # 如果迹大于零，计算正则化参数 R
        else:
            R = reg  # 否则，使用默认的正则化参数
        G.flat[:: n_neighbors + 1] += R  # 在 G 的对角线上增加正则化项
        w = solve(G, v, assume_a="pos")  # 解线性方程 Gw = v，假设 G 是正定的
        B[i, :] = w / np.sum(w)  # 计算归一化的重心权重并存储到 B 中

    return B  # 返回计算得到的重心权重矩阵


def barycenter_kneighbors_graph(X, n_neighbors, reg=1e-3, n_jobs=None):
    """计算 X 中每个样本的 k-近邻的重心加权图

    Parameters
    ----------
    X : {array-like, NearestNeighbors}
        样本数据，形状为 (n_samples, n_features)，可以是 NumPy 数组或 NearestNeighbors 对象

    n_neighbors : int
        每个样本的近邻数量

    reg : float, default=1e-3
        在解最小二乘问题时的正则化参数。仅在 mode='barycenter' 时有效。如果为 None，则使用默认值。

    n_jobs : int or None, optional (default=None)
        并行运行的作业数。默认为 None，表示不并行运行。
    # 设置并行运行的邻居搜索作业数量。如果为 None，则默认为 1，除非在 joblib.parallel_backend 上下文中。
    # 如果为 -1，则使用所有可用的处理器。详见“Glossary <n_jobs>”了解更多细节。
    n_jobs : int or None, default=None
        
    # 返回一个稀疏矩阵，采用 CSR 格式，形状为 [n_samples, n_samples]。
    # A[i, j] 被赋予连接 i 到 j 的边的权重。
    Returns
    -------
    
    # 查看相关函数
    See Also
    --------
    sklearn.neighbors.kneighbors_graph
    sklearn.neighbors.radius_neighbors_graph
    """
    
    # 使用 n_neighbors+1 个最近邻创建 KNN 模型，使用给定的并行作业数量 n_jobs 来拟合数据 X。
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs).fit(X)
    
    # 获取 KNN 模型中的拟合数据 X
    X = knn._fit_X
    
    # 获取拟合数据的样本数
    n_samples = knn.n_samples_fit_
    
    # 获取每个样本的 k 个最近邻的索引，排除自身
    ind = knn.kneighbors(X, return_distance=False)[:, 1:]
    
    # 计算每个样本的重心权重
    data = barycenter_weights(X, X, ind, reg=reg)
    
    # 生成 CSR 格式稀疏矩阵的行指针数组
    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    
    # 返回稀疏矩阵，其中数据为 data.ravel()，列索引为 ind.ravel()，行指针为 indptr，形状为 (n_samples, n_samples)
    return csr_matrix((data.ravel(), ind.ravel(), indptr), shape=(n_samples, n_samples))
# 定义一个函数，用于找到矩阵 M 的零空间。

def null_space(
    M, k, k_skip=1, eigen_solver="arpack", tol=1e-6, max_iter=100, random_state=None
):
    """
    Find the null space of a matrix M.

    Parameters
    ----------
    M : {array, matrix, sparse matrix, LinearOperator}
        Input covariance matrix: should be symmetric positive semi-definite

    k : int
        Number of eigenvalues/vectors to return

    k_skip : int, default=1
        Number of low eigenvalues to skip.

    eigen_solver : {'auto', 'arpack', 'dense'}, default='arpack'
        auto : algorithm will attempt to choose the best method for input data
        arpack : use arnoldi iteration in shift-invert mode.
                    For this method, M may be a dense matrix, sparse matrix,
                    or general linear operator.
                    Warning: ARPACK can be unstable for some problems.  It is
                    best to try several random seeds in order to check results.
        dense  : use standard dense matrix operations for the eigenvalue
                    decomposition.  For this method, M must be an array
                    or matrix type.  This method should be avoided for
                    large problems.

    tol : float, default=1e-6
        Tolerance for 'arpack' method.
        Not used if eigen_solver=='dense'.

    max_iter : int, default=100
        Maximum number of iterations for 'arpack' method.
        Not used if eigen_solver=='dense'

    random_state : int, RandomState instance, default=None
        Determines the random number generator when ``solver`` == 'arpack'.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    """

    # 根据输入的矩阵大小和需要返回的特征向量数量，自动选择合适的特征值求解方法
    if eigen_solver == "auto":
        if M.shape[0] > 200 and k + k_skip < 10:
            eigen_solver = "arpack"
        else:
            eigen_solver = "dense"

    # 使用 ARPACK 方法进行特征值分解
    if eigen_solver == "arpack":
        # 初始化 ARPACK 方法的初始向量 v0
        v0 = _init_arpack_v0(M.shape[0], random_state)
        try:
            # 调用 ARPACK 方法求解特征值和特征向量
            eigen_values, eigen_vectors = eigsh(
                M, k + k_skip, sigma=0.0, tol=tol, maxiter=max_iter, v0=v0
            )
        except RuntimeError as e:
            # 如果 ARPACK 方法出现错误，建议使用密集矩阵方法，并提供错误信息
            raise ValueError(
                "Error in determining null-space with ARPACK. Error message: "
                "'%s'. Note that eigen_solver='arpack' can fail when the "
                "weight matrix is singular or otherwise ill-behaved. In that "
                "case, eigen_solver='dense' is recommended. See online "
                "documentation for more information." % e
            ) from e

        # 返回从第 k_skip 到末尾的特征向量和特征值之和
        return eigen_vectors[:, k_skip:], np.sum(eigen_values[k_skip:])

    # 使用密集矩阵方法进行特征值分解
    elif eigen_solver == "dense":
        # 如果 M 具有 toarray 方法，则转换为密集矩阵
        if hasattr(M, "toarray"):
            M = M.toarray()
        # 使用密集矩阵方法进行特征值分解，并按索引排序特征值
        eigen_values, eigen_vectors = eigh(
            M, subset_by_index=(k_skip, k + k_skip - 1), overwrite_a=True
        )
        index = np.argsort(np.abs(eigen_values))
        # 返回按排序后索引排列的特征向量和特征值之和
        return eigen_vectors[:, index], np.sum(eigen_values)
    # 如果以上条件都不满足，则抛出值错误异常，指明未识别的特征值求解器名称
    else:
        raise ValueError("Unrecognized eigen_solver '%s'" % eigen_solver)
    # 创建一个最近邻对象，用于寻找每个样本的最近邻居
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    # 将数据拟合到最近邻对象中
    nbrs.fit(X)
    # 获取拟合后的数据，这是为了解决 NearestNeighbors 对象的内部数据引用问题
    X = nbrs._fit_X

    # 获取输入数据的样本数量 N 和维度 d_in
    N, d_in = X.shape

    # 检查输出的降维维度是否小于等于输入的维度
    if n_components > d_in:
        raise ValueError(
            "output dimension must be less than or equal to input dimension"
        )
    # 检查最近邻的数量是否小于样本数量，以确保算法能正常工作
    if n_neighbors >= N:
        raise ValueError(
            "Expected n_neighbors <= n_samples,  but n_samples = %d, n_neighbors = %d"
            % (N, n_neighbors)
        )

    # 根据 eigen_solver 类型确定 M 是否为稀疏矩阵
    M_sparse = eigen_solver != "dense"
    # 根据稀疏矩阵类型选择合适的容器构造器
    M_container_constructor = lil_matrix if M_sparse else np.zeros

    # 根据不同的方法来构造权重矩阵 W
    if method == "standard":
        # 使用标准方法计算权重矩阵 W
        W = barycenter_kneighbors_graph(
            nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs
        )

        # 计算 M = (I-W)'(I-W)
        # 根据 solver 的类型，采用不同的计算方法
        if M_sparse:
            M = eye(*W.shape, format=W.format) - W
            M = M.T * M
        else:
            M = (W.T * W - W.T - W).toarray()
            # 将对角线元素置为 1，相当于 W = W - I
            M.flat[:: M.shape[0] + 1] += 1

    elif method == "hessian":
        # 计算 Hessian 方法需要的降维空间维度
        dp = n_components * (n_components + 1) // 2

        # 检查邻居数量是否满足 Hessian 方法的要求
        if n_neighbors <= n_components + dp:
            raise ValueError(
                "for method='hessian', n_neighbors must be "
                "greater than "
                "[n_components * (n_components + 3) / 2]"
            )

        # 获取每个样本的最近邻居索引
        neighbors = nbrs.kneighbors(
            X, n_neighbors=n_neighbors + 1, return_distance=False
        )
        neighbors = neighbors[:, 1:]

        # 创建一个空的 Yi 矩阵，用于存储降维后的特征
        Yi = np.empty((n_neighbors, 1 + n_components + dp), dtype=np.float64)
        Yi[:, 0] = 1

        # 根据稀疏矩阵类型创建 M 矩阵的容器
        M = M_container_constructor((N, N), dtype=np.float64)

        # 根据邻居数据计算 Gi 并进行平均中心化处理
        use_svd = n_neighbors > d_in
        for i in range(N):
            Gi = X[neighbors[i]]
            Gi -= Gi.mean(0)

            # 构建 Hessian 估计器
            if use_svd:
                U = svd(Gi, full_matrices=0)[0]
            else:
                Ci = np.dot(Gi, Gi.T)
                U = eigh(Ci)[1][:, ::-1]

            # 将计算的特征加入到 Yi 矩阵中
            Yi[:, 1 : 1 + n_components] = U[:, :n_components]

            j = 1 + n_components
            for k in range(n_components):
                Yi[:, j : j + n_components - k] = U[:, k : k + 1] * U[:, k:n_components]
                j += n_components - k

            # 对 Q 和 R 进行 QR 分解
            Q, R = qr(Yi)

            # 计算权重向量 w，并进行归一化处理
            w = Q[:, n_components + 1 :]
            S = w.sum(0)
            S[np.where(abs(S) < hessian_tol)] = 1
            w /= S

            # 根据权重更新 M 矩阵
            nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
            M[nbrs_x, nbrs_y] += np.dot(w, w.T)
    # 如果方法选择为 "ltsa"，执行以下操作
    elif method == "ltsa":
        # 计算每个样本点的近邻，返回近邻的索引（排除自身），作为邻居索引数组
        neighbors = nbrs.kneighbors(
            X, n_neighbors=n_neighbors + 1, return_distance=False
        )
        # 由于返回的邻居数组包含自身，需要去除第一个元素（自身索引）
        neighbors = neighbors[:, 1:]

        # 构造一个容器 M，用于存储 LTSA 算法中的中间结果，大小为 (N, N)，数据类型为 np.float64
        M = M_container_constructor((N, N), dtype=np.float64)

        # 根据邻居数是否大于输入维度来确定是否使用 SVD 分解
        use_svd = n_neighbors > d_in

        # 遍历每个样本点
        for i in range(N):
            # 取出当前样本点的邻居点集合 Xi，并将其减去均值（即中心化）
            Xi = X[neighbors[i]]
            Xi -= Xi.mean(0)

            # 如果使用 SVD 进行分解
            if use_svd:
                # 对中心化后的邻居点集合 Xi 进行 SVD 分解，返回的 v 是特征向量
                v = svd(Xi, full_matrices=True)[0]
            else:
                # 否则，计算 Xi * Xi^T 的特征值分解，返回的 v 是特征向量
                Ci = np.dot(Xi, Xi.T)
                v = eigh(Ci)[1][:, ::-1]

            # 初始化 Gi 矩阵，大小为 (n_neighbors, n_components + 1)
            Gi = np.zeros((n_neighbors, n_components + 1))
            # 将特征向量 v 的前 n_components 列存入 Gi 中
            Gi[:, 1:] = v[:, :n_components]
            # Gi 的第一列设为 1.0 / sqrt(n_neighbors)
            Gi[:, 0] = 1.0 / np.sqrt(n_neighbors)

            # 计算 Gi * Gi^T 得到 GiGiT
            GiGiT = np.dot(Gi, Gi.T)

            # 构造邻居点的索引网格
            nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
            # 在 M 中减去 GiGiT
            M[nbrs_x, nbrs_y] -= GiGiT

            # M 中邻居点自身的位置加上 n_neighbors 个单位矩阵
            M[neighbors[i], neighbors[i]] += np.ones(shape=n_neighbors)

    # 如果 M_sparse 为 True，则将 M 转换为稀疏矩阵格式（CSR 格式）
    if M_sparse:
        M = M.tocsr()

    # 调用 null_space 函数，计算 M 的零空间，返回降维后的表示
    return null_space(
        M,
        n_components,
        k_skip=1,
        eigen_solver=eigen_solver,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
    )
# 使用装饰器 validate_params 进行参数验证，确保函数输入符合指定的格式和条件
@validate_params(
    {
        "X": ["array-like", NearestNeighbors],  # X 参数应为类似数组或 NearestNeighbors 对象
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],  # n_neighbors 参数为大于等于1的整数
        "n_components": [Interval(Integral, 1, None, closed="left")],  # n_components 参数为大于等于1的整数
        "reg": [Interval(Real, 0, None, closed="left")],  # reg 参数为大于等于0的实数
        "eigen_solver": [StrOptions({"auto", "arpack", "dense"})],  # eigen_solver 参数为 "auto", "arpack", 或 "dense" 中的一个字符串
        "tol": [Interval(Real, 0, None, closed="left")],  # tol 参数为大于等于0的实数
        "max_iter": [Interval(Integral, 1, None, closed="left")],  # max_iter 参数为大于等于1的整数
        "method": [StrOptions({"standard", "hessian", "modified", "ltsa"})],  # method 参数为 "standard", "hessian", "modified", 或 "ltsa" 中的一个字符串
        "hessian_tol": [Interval(Real, 0, None, closed="left")],  # hessian_tol 参数为大于等于0的实数
        "modified_tol": [Interval(Real, 0, None, closed="left")],  # modified_tol 参数为大于等于0的实数
        "random_state": ["random_state"],  # random_state 参数为随机状态对象
        "n_jobs": [None, Integral],  # n_jobs 参数可以为 None 或整数类型
    },
    prefer_skip_nested_validation=True,  # 设置优先跳过嵌套验证
)
# locally_linear_embedding 函数定义，执行局部线性嵌入分析
def locally_linear_embedding(
    X,
    *,
    n_neighbors,
    n_components,
    reg=1e-3,
    eigen_solver="auto",
    tol=1e-6,
    max_iter=100,
    method="standard",
    hessian_tol=1e-4,
    modified_tol=1e-12,
    random_state=None,
    n_jobs=None,
):
    """Perform a Locally Linear Embedding analysis on the data.

    Read more in the :ref:`User Guide <locally_linear_embedding>`.

    Parameters
    ----------
    X : {array-like, NearestNeighbors}
        Sample data, shape = (n_samples, n_features), in the form of a
        numpy array or a NearestNeighbors object.

    n_neighbors : int
        Number of neighbors to consider for each point.

    n_components : int
        Number of coordinates for the manifold.

    reg : float, default=1e-3
        Regularization constant, multiplies the trace of the local covariance
        matrix of the distances.

    eigen_solver : {'auto', 'arpack', 'dense'}, default='auto'
        auto : algorithm will attempt to choose the best method for input data

        arpack : use arnoldi iteration in shift-invert mode.
                    For this method, M may be a dense matrix, sparse matrix,
                    or general linear operator.
                    Warning: ARPACK can be unstable for some problems.  It is
                    best to try several random seeds in order to check results.

        dense  : use standard dense matrix operations for the eigenvalue
                    decomposition.  For this method, M must be an array
                    or matrix type.  This method should be avoided for
                    large problems.

    tol : float, default=1e-6
        Tolerance for 'arpack' method
        Not used if eigen_solver=='dense'.

    max_iter : int, default=100
        Maximum number of iterations for the arpack solver.
    method : {'standard', 'hessian', 'modified', 'ltsa'}, default='standard'
        standard : 使用标准的局部线性嵌入算法。
                   参见文献 [1]_
        hessian  : 使用 Hessian 特征映射方法。该方法要求
                   n_neighbors > n_components * (1 + (n_components + 1) / 2。
                   参见文献 [2]_
        modified : 使用修改后的局部线性嵌入算法。
                   参见文献 [3]_
        ltsa     : 使用局部切线空间对齐算法。
                   参见文献 [4]_

    hessian_tol : float, default=1e-4
        Hessian 特征映射方法的容差值。
        仅在 method == 'hessian' 时使用。

    modified_tol : float, default=1e-12
        修改后的局部线性嵌入方法的容差值。
        仅在 method == 'modified' 时使用。

    random_state : int, RandomState instance, default=None
        当 solver == 'arpack' 时，确定随机数生成器。
        设置为 int 可实现多次函数调用的可重复结果。
        参见 :term:`Glossary <random_state>`。

    n_jobs : int or None, default=None
        用于邻居搜索的并行作业数。
        ``None`` 表示 1，除非在 :obj:`joblib.parallel_backend` 上下文中。
        ``-1`` 表示使用所有处理器。更多详细信息参见 :term:`Glossary <n_jobs>`。

    Returns
    -------
    Y : ndarray of shape (n_samples, n_components)
        嵌入向量。

    squared_error : float
        嵌入向量的重构误差。等同于 ``norm(Y - W Y, 'fro')**2``，其中 W 是重构权重。

    References
    ----------

    .. [1] Roweis, S. & Saul, L. Nonlinear dimensionality reduction
        by locally linear embedding.  Science 290:2323 (2000).
    .. [2] Donoho, D. & Grimes, C. Hessian eigenmaps: Locally
        linear embedding techniques for high-dimensional data.
        Proc Natl Acad Sci U S A.  100:5591 (2003).
    .. [3] `Zhang, Z. & Wang, J. MLLE: Modified Locally Linear
        Embedding Using Multiple Weights.
        <https://citeseerx.ist.psu.edu/doc_view/pid/0b060fdbd92cbcc66b383bcaa9ba5e5e624d7ee3>`_
    .. [4] Zhang, Z. & Zha, H. Principal manifolds and nonlinear
        dimensionality reduction via tangent space alignment.
        Journal of Shanghai Univ.  8:406 (2004)

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import locally_linear_embedding
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding, _ = locally_linear_embedding(X[:100],n_neighbors=5, n_components=2)
    >>> embedding.shape
    (100, 2)
    # 调用 _locally_linear_embedding 函数进行局部线性嵌入计算，并返回结果
    return _locally_linear_embedding(
        X=X,  # 输入数据矩阵
        n_neighbors=n_neighbors,  # 近邻数目
        n_components=n_components,  # 输出的嵌入空间维度
        reg=reg,  # 正则化系数
        eigen_solver=eigen_solver,  # 特征值求解器的选择
        tol=tol,  # 特征值分解的容差
        max_iter=max_iter,  # 迭代最大次数
        method=method,  # 解算方法
        hessian_tol=hessian_tol,  # Hessian 特征值的容差
        modified_tol=modified_tol,  # 稀疏特征值的容差
        random_state=random_state,  # 随机数种子或随机状态
        n_jobs=n_jobs,  # 并行运行的作业数
    )
class LocallyLinearEmbedding(
    ClassNamePrefixFeaturesOutMixin,  # 继承自 ClassNamePrefixFeaturesOutMixin，提供了一些特定功能和输出的混合类
    TransformerMixin,  # 继承自 TransformerMixin，用于数据转换的Mixin类
    _UnstableArchMixin,  # 继承自 _UnstableArchMixin，可能是与不稳定架构相关的Mixin类
    BaseEstimator,  # 继承自 BaseEstimator，是所有估计器的基类
):
    """Locally Linear Embedding.

    Read more in the :ref:`User Guide <locally_linear_embedding>`.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to consider for each point.

    n_components : int, default=2
        Number of coordinates for the manifold.

    reg : float, default=1e-3
        Regularization constant, multiplies the trace of the local covariance
        matrix of the distances.

    eigen_solver : {'auto', 'arpack', 'dense'}, default='auto'
        The solver used to compute the eigenvectors. The available options are:

        - `'auto'` : algorithm will attempt to choose the best method for input
          data.
        - `'arpack'` : use arnoldi iteration in shift-invert mode. For this
          method, M may be a dense matrix, sparse matrix, or general linear
          operator.
        - `'dense'`  : use standard dense matrix operations for the eigenvalue
          decomposition. For this method, M must be an array or matrix type.
          This method should be avoided for large problems.

        .. warning::
           ARPACK can be unstable for some problems.  It is best to try several
           random seeds in order to check results.

    tol : float, default=1e-6
        Tolerance for 'arpack' method
        Not used if eigen_solver=='dense'.

    max_iter : int, default=100
        Maximum number of iterations for the arpack solver.
        Not used if eigen_solver=='dense'.

    method : {'standard', 'hessian', 'modified', 'ltsa'}, default='standard'
        - `standard`: use the standard locally linear embedding algorithm. see
          reference [1]_
        - `hessian`: use the Hessian eigenmap method. This method requires
          ``n_neighbors > n_components * (1 + (n_components + 1) / 2``. see
          reference [2]_
        - `modified`: use the modified locally linear embedding algorithm.
          see reference [3]_
        - `ltsa`: use local tangent space alignment algorithm. see
          reference [4]_

    hessian_tol : float, default=1e-4
        Tolerance for Hessian eigenmapping method.
        Only used if ``method == 'hessian'``.

    modified_tol : float, default=1e-12
        Tolerance for modified LLE method.
        Only used if ``method == 'modified'``.

    neighbors_algorithm : {'auto', 'brute', 'kd_tree', 'ball_tree'}, \
                          default='auto'
        Algorithm to use for nearest neighbors search, passed to
        :class:`~sklearn.neighbors.NearestNeighbors` instance.

    random_state : int, RandomState instance, default=None
        Determines the random number generator when
        ``eigen_solver`` == 'arpack'. Pass an int for reproducible results
        across multiple function calls. See :term:`Glossary <random_state>`.
    """
    _parameter_constraints: dict = {
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "reg": [Interval(Real, 0, None, closed="left")],
        "eigen_solver": [StrOptions({"auto", "arpack", "dense"})],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "method": [StrOptions({"standard", "hessian", "modified", "ltsa"})],
        "hessian_tol": [Interval(Real, 0, None, closed="left")],
        "modified_tol": [Interval(Real, 0, None, closed="left")],
        "neighbors_algorithm": [StrOptions({"auto", "brute", "kd_tree", "ball_tree"})],
        "random_state": ["random_state"],
        "n_jobs": [None, Integral],
    }



    _parameter_constraints: dict = {
        # 参数约束字典，定义了 LocallyLinearEmbedding 类的各参数的取值范围和类型约束
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        # 邻居数，必须为大于等于1的整数
        "n_components": [Interval(Integral, 1, None, closed="left")],
        # 嵌入的维度，必须为大于等于1的整数
        "reg": [Interval(Real, 0, None, closed="left")],
        # 正则化参数，必须为大于等于0的实数
        "eigen_solver": [StrOptions({"auto", "arpack", "dense"})],
        # 特征值求解器，可选项为{"auto", "arpack", "dense"}中的一个字符串
        "tol": [Interval(Real, 0, None, closed="left")],
        # 迭代收敛的容差，必须为大于等于0的实数
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        # 最大迭代次数，必须为大于等于1的整数
        "method": [StrOptions({"standard", "hessian", "modified", "ltsa"})],
        # 计算方法，可选项为{"standard", "hessian", "modified", "ltsa"}中的一个字符串
        "hessian_tol": [Interval(Real, 0, None, closed="left")],
        # Hessian 特征值计算的容差，必须为大于等于0的实数
        "modified_tol": [Interval(Real, 0, None, closed="left")],
        # Modified LLE 特征值计算的容差，必须为大于等于0的实数
        "neighbors_algorithm": [StrOptions({"auto", "brute", "kd_tree", "ball_tree"})],
        # 邻居搜索算法，可选项为{"auto", "brute", "kd_tree", "ball_tree"}中的一个字符串
        "random_state": ["random_state"],
        # 随机数种子或者 None，用于控制随机性
        "n_jobs": [None, Integral],
        # 并行作业数量，可以是 None 或者一个整数
    }
    }

    # 初始化方法，设置各种参数并将其保存在对象中
    def __init__(
        self,
        *,
        n_neighbors=5,                    # 邻居数量，默认为5
        n_components=2,                   # 嵌入空间的维度，默认为2
        reg=1e-3,                         # 正则化参数，默认为1e-3
        eigen_solver="auto",              # 特征值求解器的选择，默认为"auto"
        tol=1e-6,                         # 迭代求解的容差，默认为1e-6
        max_iter=100,                     # 最大迭代次数，默认为100
        method="standard",                # LLE方法，默认为"standard"
        hessian_tol=1e-4,                 # Hessian矩阵的容差，默认为1e-4
        modified_tol=1e-12,               # 改进的Graßmann条件的容差，默认为1e-12
        neighbors_algorithm="auto",       # 邻居搜索算法的选择，默认为"auto"
        random_state=None,                # 随机数种子，默认为None
        n_jobs=None,                      # 并行作业数，默认为None
    ):
        self.n_neighbors = n_neighbors    # 将参数保存到对象的属性中
        self.n_components = n_components  # 将参数保存到对象的属性中
        self.reg = reg                    # 将参数保存到对象的属性中
        self.eigen_solver = eigen_solver  # 将参数保存到对象的属性中
        self.tol = tol                    # 将参数保存到对象的属性中
        self.max_iter = max_iter          # 将参数保存到对象的属性中
        self.method = method              # 将参数保存到对象的属性中
        self.hessian_tol = hessian_tol    # 将参数保存到对象的属性中
        self.modified_tol = modified_tol  # 将参数保存到对象的属性中
        self.random_state = random_state  # 将参数保存到对象的属性中
        self.neighbors_algorithm = neighbors_algorithm  # 将参数保存到对象的属性中
        self.n_jobs = n_jobs              # 将参数保存到对象的属性中

    # 私有方法，用于拟合和转换数据，计算嵌入向量并估计重构误差
    def _fit_transform(self, X):
        # 使用最近邻居算法拟合数据
        self.nbrs_ = NearestNeighbors(
            n_neighbors=self.n_neighbors,             # 设置最近邻居的数量
            algorithm=self.neighbors_algorithm,       # 设置邻居搜索算法
            n_jobs=self.n_jobs,                       # 设置并行作业数
        )

        # 检查随机数种子，并根据需要设置
        random_state = check_random_state(self.random_state)
        # 验证数据X的格式，并转换为浮点型数据
        X = self._validate_data(X, dtype=float)
        # 对数据进行最近邻拟合
        self.nbrs_.fit(X)
        # 计算局部线性嵌入并估计重构误差
        self.embedding_, self.reconstruction_error_ = _locally_linear_embedding(
            X=self.nbrs_,                           # 最近邻对象作为输入数据
            n_neighbors=self.n_neighbors,           # 最近邻的数量
            n_components=self.n_components,         # 嵌入空间的维度
            eigen_solver=self.eigen_solver,         # 特征值求解器的选择
            tol=self.tol,                           # 求解迭代的容差
            max_iter=self.max_iter,                 # 最大迭代次数
            method=self.method,                     # LLE方法的选择
            hessian_tol=self.hessian_tol,           # Hessian矩阵的容差
            modified_tol=self.modified_tol,         # 改进的Graßmann条件的容差
            random_state=random_state,              # 随机数种子
            reg=self.reg,                           # 正则化参数
            n_jobs=self.n_jobs,                     # 并行作业数
        )
        # 设置输出的特征数量
        self._n_features_out = self.embedding_.shape[1]

    # 装饰器，用于控制拟合过程
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Compute the embedding vectors for data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training set.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted `LocallyLinearEmbedding` class instance.
        """
        # 调用内部方法进行拟合和转换
        self._fit_transform(X)
        return self

    # 装饰器，用于控制拟合和转换过程
    @_fit_context(prefer_skip_nested_validation=True)
    def fit_transform(self, X, y=None):
        """Compute the embedding vectors for data X and transform X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training set.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Returns the instance itself.
        """
        # 调用内部方法进行拟合和转换，并返回嵌入向量
        self._fit_transform(X)
        return self.embedding_
    # 定义一个方法，用于将新数据点转换到嵌入空间中

    """
    Transform new points into embedding space.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        训练集，包含n_samples个样本和n_features个特征。

    Returns
    -------
    X_new : ndarray of shape (n_samples, n_components)
        返回转换后的数据集。

    Notes
    -----
    由于该方法执行了缩放操作，不建议与不具有缩放不变性的方法（如SVMs）一同使用。
    """
    
    # 检查当前实例是否已经拟合过，确保能够进行转换
    check_is_fitted(self)

    # 使用内部方法验证输入数据X，不重置数据
    X = self._validate_data(X, reset=False)
    
    # 计算每个样本点的最近邻居索引
    ind = self.nbrs_.kneighbors(
        X, n_neighbors=self.n_neighbors, return_distance=False
    )
    
    # 计算使用质心权重的权重向量
    weights = barycenter_weights(X, self.nbrs_._fit_X, ind, reg=self.reg)
    
    # 创建一个空的数组，用于存储转换后的数据
    X_new = np.empty((X.shape[0], self.n_components))
    
    # 针对每个样本点，使用其最近邻居的嵌入空间来计算转换后的值
    for i in range(X.shape[0]):
        X_new[i] = np.dot(self.embedding_[ind[i]].T, weights[i])
    
    # 返回转换后的数据集
    return X_new
```