# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\linsolve.py`

```
# 从警告模块中导入warn、catch_warnings和simplefilter函数
from warnings import warn, catch_warnings, simplefilter

# 导入numpy库，并从中导入asarray函数
import numpy as np
from numpy import asarray

# 从scipy.sparse子模块中导入issparse、SparseEfficiencyWarning、csc_matrix、eye和diags函数
from scipy.sparse import (issparse,
                          SparseEfficiencyWarning, csc_matrix, eye, diags)

# 从scipy.sparse._sputils模块中导入is_pydata_spmatrix和convert_pydata_sparse_to_scipy函数
from scipy.sparse._sputils import is_pydata_spmatrix, convert_pydata_sparse_to_scipy

# 从scipy.linalg模块中导入LinAlgError异常类
from scipy.linalg import LinAlgError

# 导入copy模块
import copy

# 从当前包中导入_superlu模块
from . import _superlu

# 初始化noScikit变量为False
noScikit = False
# 尝试导入scikits.umfpack模块，如果失败则将noScikit设为True
try:
    import scikits.umfpack as umfpack
except ImportError:
    noScikit = True

# 根据是否成功导入scikits.umfpack模块，设置useUmfpack变量为相应值
useUmfpack = not noScikit

# 定义__all__列表，包含公开的函数和类名
__all__ = ['use_solver', 'spsolve', 'splu', 'spilu', 'factorized',
           'MatrixRankWarning', 'spsolve_triangular']

# 定义MatrixRankWarning类，继承自UserWarning，用于矩阵秩相关警告
class MatrixRankWarning(UserWarning):
    pass

# 定义use_solver函数，用于选择默认的稀疏直接求解器
def use_solver(**kwargs):
    """
    Select default sparse direct solver to be used.

    Parameters
    ----------
    useUmfpack : bool, optional
        Use UMFPACK [1]_, [2]_, [3]_, [4]_. over SuperLU. Has effect only
        if ``scikits.umfpack`` is installed. Default: True
    assumeSortedIndices : bool, optional
        Allow UMFPACK to skip the step of sorting indices for a CSR/CSC matrix.
        Has effect only if useUmfpack is True and ``scikits.umfpack`` is
        installed. Default: False

    Notes
    -----
    The default sparse solver is UMFPACK when available
    (``scikits.umfpack`` is installed). This can be changed by passing
    useUmfpack = False, which then causes the always present SuperLU
    based solver to be used.

    UMFPACK requires a CSR/CSC matrix to have sorted column/row indices. If
    sure that the matrix fulfills this, pass ``assumeSortedIndices=True``
    to gain some speed.

    References
    ----------
    .. [1] T. A. Davis, Algorithm 832:  UMFPACK - an unsymmetric-pattern
           multifrontal method with a column pre-ordering strategy, ACM
           Trans. on Mathematical Software, 30(2), 2004, pp. 196--199.
           https://dl.acm.org/doi/abs/10.1145/992200.992206

    .. [2] T. A. Davis, A column pre-ordering strategy for the
           unsymmetric-pattern multifrontal method, ACM Trans.
           on Mathematical Software, 30(2), 2004, pp. 165--195.
           https://dl.acm.org/doi/abs/10.1145/992200.992205

    .. [3] T. A. Davis and I. S. Duff, A combined unifrontal/multifrontal
           method for unsymmetric sparse matrices, ACM Trans. on
           Mathematical Software, 25(1), 1999, pp. 1--19.
           https://doi.org/10.1145/305658.287640

    .. [4] T. A. Davis and I. S. Duff, An unsymmetric-pattern multifrontal
           method for sparse LU factorization, SIAM J. Matrix Analysis and
           Computations, 18(1), 1997, pp. 140--158.
           https://doi.org/10.1137/S0895479894246905T.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse.linalg import use_solver, spsolve
    >>> from scipy.sparse import csc_matrix
    >>> R = np.random.randn(5, 5)
    >>> A = csc_matrix(R)
    >>> b = np.random.randn(5)
    >>> use_solver(useUmfpack=False) # enforce superLU over UMFPACK

    """
    # 使用稀疏矩阵求解器 `spsolve` 计算方程组 `A * x = b` 的解 `x`
    >>> x = spsolve(A, b)
    # 验证通过矩阵乘法检查解 `x` 是否满足方程 `A * x = b`，并返回布尔值
    >>> np.allclose(A.dot(x), b)
    True
    # 设置 umfPack 使用参数 `useUmfpack` 为默认值
    >>> use_solver(useUmfpack=True) 
    """
    # 如果关键字参数中包含 `useUmfpack`，则更新全局变量 `useUmfpack` 的值
    if 'useUmfpack' in kwargs:
        globals()['useUmfpack'] = kwargs['useUmfpack']
    # 如果当前 `useUmfpack` 为真（即已启用 umfPack 解算器），并且关键字参数中包含 `assumeSortedIndices`
    if useUmfpack and 'assumeSortedIndices' in kwargs:
        # 配置 umfPack 解算器的参数 `assumeSortedIndices` 为关键字参数中指定的值
        umfpack.configure(assumeSortedIndices=kwargs['assumeSortedIndices'])
# 根据稀疏矩阵的数据类型获取 UMFPACK 的系列字符串表示。
def _get_umf_family(A):
    # 定义不同数据类型组合对应的 UMFPACK 系列字符串
    _families = {
        (np.float64, np.int32): 'di',
        (np.complex128, np.int32): 'zi',
        (np.float64, np.int64): 'dl',
        (np.complex128, np.int64): 'zl'
    }

    # 获取矩阵 A 的浮点数类型（np.float64 或 np.complex128）
    f_type = getattr(np, A.dtype.name)
    # 获取矩阵 A 的索引类型（np.int32 或 np.int64）
    i_type = getattr(np, A.indices.dtype.name)

    try:
        # 根据矩阵的数据类型组合，获取对应的 UMFPACK 系列字符串
        family = _families[(f_type, i_type)]

    except KeyError as e:
        # 抛出错误，说明不支持该数据类型组合
        msg = ('only float64 or complex128 matrices with int32 or int64 '
               f'indices are supported! (got: matrix: {f_type}, indices: {i_type})')
        raise ValueError(msg) from e

    # 修正 UMFPACK 系列字符串的最后一个字符为 'l'
    family = family[0] + "l"
    # 复制矩阵 A，将其索引指针和索引都转换为 np.int64 类型
    A_new = copy.copy(A)
    A_new.indptr = np.asarray(A.indptr, dtype=np.int64)
    A_new.indices = np.asarray(A.indices, dtype=np.int64)

    return family, A_new

# 检查并安全地将索引类型降级为 np.intc
def _safe_downcast_indices(A):
    # 检查可安全降级的最大值
    max_value = np.iinfo(np.intc).max

    # 检查最后一个指针是否超出了最大值
    if A.indptr[-1] > max_value:  # 因为 indptr 总是排序的，所以最后一个值是最大值
        raise ValueError("indptr values too large for SuperLU")

    # 只有当数组足够大时才检查
    if max(*A.shape) > max_value:
        # 如果任意索引值超出最大值，抛出错误
        if np.any(A.indices > max_value):
            raise ValueError("indices values too large for SuperLU")

    # 将索引和指针转换为 np.intc 类型并返回
    indices = A.indices.astype(np.intc, copy=False)
    indptr = A.indptr.astype(np.intc, copy=False)
    return indices, indptr

# 解决稀疏线性方程组 Ax=b 的函数
def spsolve(A, b, permc_spec=None, use_umfpack=True):
    """Solve the sparse linear system Ax=b, where b may be a vector or a matrix.

    Parameters
    ----------
    A : ndarray or sparse matrix
        The square matrix A will be converted into CSC or CSR form
    b : ndarray or sparse matrix
        The matrix or vector representing the right hand side of the equation.
        If a vector, b.shape must be (n,) or (n, 1).
    permc_spec : str, optional
        How to permute the columns of the matrix for sparsity preservation.
        (default: 'COLAMD')

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering [1]_, [2]_.

    use_umfpack : bool, optional
        if True (default) then use UMFPACK for the solution [3]_, [4]_, [5]_,
        [6]_ . This is only referenced if b is a vector and
        ``scikits.umfpack`` is installed.

    Returns
    -------
    x : ndarray or sparse matrix
        the solution of the sparse linear equation.
        If b is a vector, then x is a vector of size A.shape[1]
        If b is a matrix, then x is a matrix of size (A.shape[1], b.shape[1])
    """
    """
    Notes
    -----
    For solving the matrix expression AX = B, this solver assumes the resulting
    matrix X is sparse, as is often the case for very sparse inputs.  If the
    resulting X is dense, the construction of this sparse result will be
    relatively expensive.  In that case, consider converting A to a dense
    matrix and using scipy.linalg.solve or its variants.

    References
    ----------
    .. [1] T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, Algorithm 836:
           COLAMD, an approximate column minimum degree ordering algorithm,
           ACM Trans. on Mathematical Software, 30(3), 2004, pp. 377--380.
           :doi:`10.1145/1024074.1024080`

    .. [2] T. A. Davis, J. R. Gilbert, S. Larimore, E. Ng, A column approximate
           minimum degree ordering algorithm, ACM Trans. on Mathematical
           Software, 30(3), 2004, pp. 353--376. :doi:`10.1145/1024074.1024079`

    .. [3] T. A. Davis, Algorithm 832:  UMFPACK - an unsymmetric-pattern
           multifrontal method with a column pre-ordering strategy, ACM
           Trans. on Mathematical Software, 30(2), 2004, pp. 196--199.
           https://dl.acm.org/doi/abs/10.1145/992200.992206

    .. [4] T. A. Davis, A column pre-ordering strategy for the
           unsymmetric-pattern multifrontal method, ACM Trans.
           on Mathematical Software, 30(2), 2004, pp. 165--195.
           https://dl.acm.org/doi/abs/10.1145/992200.992205

    .. [5] T. A. Davis and I. S. Duff, A combined unifrontal/multifrontal
           method for unsymmetric sparse matrices, ACM Trans. on
           Mathematical Software, 25(1), 1999, pp. 1--19.
           https://doi.org/10.1145/305658.287640

    .. [6] T. A. Davis and I. S. Duff, An unsymmetric-pattern multifrontal
           method for sparse LU factorization, SIAM J. Matrix Analysis and
           Computations, 18(1), 1997, pp. 140--158.
           https://doi.org/10.1137/S0895479894246905T.


    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import spsolve
    >>> A = csc_matrix([[3, 2, 0], [1, -1, 0], [0, 5, 1]], dtype=float)
    >>> B = csc_matrix([[2, 0], [-1, 0], [2, 0]], dtype=float)
    >>> x = spsolve(A, B)
    >>> np.allclose(A.dot(x).toarray(), B.toarray())
    True
    """

    # 检查输入是否为 PyData 稀疏矩阵，并获取其类信息
    is_pydata_sparse = is_pydata_spmatrix(b)
    pydata_sparse_cls = b.__class__ if is_pydata_sparse else None

    # 将输入矩阵 A 和向量 b 转换为 scipy 稀疏矩阵格式
    A = convert_pydata_sparse_to_scipy(A)
    b = convert_pydata_sparse_to_scipy(b)

    # 如果 A 不是稀疏矩阵或者格式不是 "csc" 或 "csr"，则转换为 csc 格式
    if not (issparse(A) and A.format in ("csc", "csr")):
        A = csc_matrix(A)
        warn('spsolve requires A be CSC or CSR matrix format',
             SparseEfficiencyWarning, stacklevel=2)

    # 如果 b 不是稀疏矩阵，则转换为数组
    b_is_sparse = issparse(b)
    if not b_is_sparse:
        b = asarray(b)

    # 检查 b 是否为向量（即形状为 (n,) 或 (n, 1)）
    b_is_vector = ((b.ndim == 1) or (b.ndim == 2 and b.shape[1] == 1))

    # 对于非规范格式，对 A 进行重复项求和处理
    A.sum_duplicates()
    A = A._asfptype()  # 将 A 转换为浮点数格式（如果不是的话）
    result_dtype = np.promote_types(A.dtype, b.dtype)  # 获取 A 和 b 的最广泛的数据类型
    if A.dtype != result_dtype:
        A = A.astype(result_dtype)  # 如果 A 的数据类型不是 result_dtype，则转换为 result_dtype
    if b.dtype != result_dtype:
        b = b.astype(result_dtype)  # 如果 b 的数据类型不是 result_dtype，则转换为 result_dtype

    # 验证输入的形状
    M, N = A.shape  # 获取 A 的行数 M 和列数 N
    if M != N:
        raise ValueError(f"matrix must be square (has shape {(M, N)})")  # 如果 A 不是方阵，则抛出 ValueError

    if M != b.shape[0]:
        raise ValueError(f"matrix - rhs dimension mismatch ({A.shape} - {b.shape[0]})")  # 如果 A 和 b 的行数不匹配，则抛出 ValueError

    use_umfpack = use_umfpack and useUmfpack  # 更新 use_umfpack 变量为 use_umfpack 和 useUmfpack 的逻辑与

    if b_is_vector and use_umfpack:  # 如果 b 是向量并且 use_umfpack 为 True
        if b_is_sparse:
            b_vec = b.toarray()  # 如果 b 是稀疏矩阵，则转换为稠密数组
        else:
            b_vec = b  # 否则直接使用 b
        b_vec = asarray(b_vec, dtype=A.dtype).ravel()  # 将 b_vec 转换为 A 的数据类型的数组，并展平为一维数组

        if noScikit:
            raise RuntimeError('Scikits.umfpack not installed.')  # 如果没有安装 Scikits.umfpack，则抛出 RuntimeError

        if A.dtype.char not in 'dD':
            raise ValueError("convert matrix data to double, please, using"
                  " .astype(), or set linsolve.useUmfpack = False")  # 如果 A 的数据类型不是双精度浮点型，则抛出 ValueError

        umf_family, A = _get_umf_family(A)  # 获取 A 的 UMFPACK 家族类型和更新后的 A
        umf = umfpack.UmfpackContext(umf_family)  # 创建 UMFPACK 上下文对象
        x = umf.linsolve(umfpack.UMFPACK_A, A, b_vec,
                         autoTranspose=True)  # 使用 UMFPACK 解线性方程组 A*x = b_vec
    else:
        # 如果 b 不是稀疏向量且是稀疏矩阵，则将 b 转换为密集数组，并更新标志
        if b_is_vector and b_is_sparse:
            b = b.toarray()
            b_is_sparse = False

        # 如果 b 不是稀疏向量，则根据 A 的格式确定 flag
        if not b_is_sparse:
            if A.format == "csc":
                flag = 1  # 使用 CSC 格式
            else:
                flag = 0  # 使用 CSR 格式

            # 将 A 的索引和指针转换为 np.intc 类型，以便传递给 _superlu.gssv
            indices = A.indices.astype(np.intc, copy=False)
            indptr = A.indptr.astype(np.intc, copy=False)
            options = dict(ColPerm=permc_spec)
            # 调用 _superlu.gssv 进行求解线性方程组
            x, info = _superlu.gssv(N, A.nnz, A.data, indices, indptr,
                                    b, flag, options=options)
            # 如果求解信息 info 不为 0，说明矩阵 A 是奇异的，发出警告并将结果填充为 NaN
            if info != 0:
                warn("Matrix is exactly singular", MatrixRankWarning, stacklevel=2)
                x.fill(np.nan)
            # 如果 b 是向量，则将 x 展平
            if b_is_vector:
                x = x.ravel()
        else:
            # 如果 b 是稀疏矩阵
            Afactsolve = factorized(A)

            # 如果 b 不是 CSC 格式或不是 PyData 稀疏矩阵，发出效率警告并将 b 转换为 CSC 格式
            if not (b.format == "csc" or is_pydata_spmatrix(b)):
                warn('spsolve is more efficient when sparse b '
                     'is in the CSC matrix format',
                     SparseEfficiencyWarning, stacklevel=2)
                b = csc_matrix(b)

            # 创建稀疏输出矩阵，通过重复应用稀疏因子分解来解决 b 的列
            data_segs = []
            row_segs = []
            col_segs = []
            for j in range(b.shape[1]):
                # TODO: 替换为 bj = b[:, j].toarray().ravel()，一旦支持 1D 稀疏数组
                bj = b[:, [j]].toarray().ravel()
                xj = Afactsolve(bj)
                w = np.flatnonzero(xj)
                segment_length = w.shape[0]
                row_segs.append(w)
                col_segs.append(np.full(segment_length, j, dtype=int))
                data_segs.append(np.asarray(xj[w], dtype=A.dtype))
            # 合并所有段的数据、行和列以创建稀疏矩阵 x
            sparse_data = np.concatenate(data_segs)
            sparse_row = np.concatenate(row_segs)
            sparse_col = np.concatenate(col_segs)
            x = A.__class__((sparse_data, (sparse_row, sparse_col)),
                           shape=b.shape, dtype=A.dtype)

            # 如果是 PyData 稀疏矩阵，则转换为对应的类型
            if is_pydata_sparse:
                x = pydata_sparse_cls.from_scipy_sparse(x)

    return x
    """
    Compute the LU decomposition of a sparse, square matrix.

    Parameters
    ----------
    A : sparse matrix
        Sparse matrix to factorize. Most efficient when provided in CSC
        format. Other formats will be converted to CSC before factorization.
    permc_spec : str, optional
        How to permute the columns of the matrix for sparsity preservation.
        (default: 'COLAMD')

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering

    diag_pivot_thresh : float, optional
        Threshold used for a diagonal entry to be an acceptable pivot.
        See SuperLU user's guide for details [1]_
    relax : int, optional
        Expert option for customizing the degree of relaxing supernodes.
        See SuperLU user's guide for details [1]_
    panel_size : int, optional
        Expert option for customizing the panel size.
        See SuperLU user's guide for details [1]_
    options : dict, optional
        Dictionary containing additional expert options to SuperLU.
        See SuperLU user guide [1]_ (section 2.4 on the 'Options' argument)
        for more details. For example, you can specify
        ``options=dict(Equil=False, IterRefine='SINGLE'))``
        to turn equilibration off and perform a single iterative refinement.

    Returns
    -------
    invA : scipy.sparse.linalg.SuperLU
        Object, which has a ``solve`` method.

    See also
    --------
    spilu : incomplete LU decomposition

    Notes
    -----
    This function uses the SuperLU library.

    References
    ----------
    .. [1] SuperLU https://portal.nersc.gov/project/sparse/superlu/

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import splu
    >>> A = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
    >>> B = splu(A)
    >>> x = np.array([1., 2., 3.], dtype=float)
    >>> B.solve(x)
    array([ 1. , -3. , -1.5])
    >>> A.dot(B.solve(x))
    array([ 1.,  2.,  3.])
    >>> B.solve(A.dot(x))
    array([ 1.,  2.,  3.])
    """

    # Check if A is a pydata sparse matrix; if so, convert it to scipy CSC format
    if is_pydata_spmatrix(A):
        def csc_construct_func(*a, cls=type(A)):
            return cls.from_scipy_sparse(csc_matrix(*a))
        A = A.to_scipy_sparse().tocsc()
    else:
        csc_construct_func = csc_matrix

    # Convert A to CSC format if it's not already in CSC format
    if not (issparse(A) and A.format == "csc"):
        A = csc_matrix(A)
        warn('splu converted its input to CSC format',
             SparseEfficiencyWarning, stacklevel=2)

    # Ensure that duplicate entries are summed for non-canonical formats
    A.sum_duplicates()
    
    # Upcast matrix A to a floating point format if necessary
    A = A._asfptype()

    # Get the dimensions of matrix A
    M, N = A.shape
    # 检查矩阵的维度是否相等，如果不相等则抛出值错误异常
    if (M != N):
        raise ValueError("can only factor square matrices")  # 这是真的吗？

    # 将稀疏矩阵 A 转换为适合传递给超级 LU 分解的索引和指针形式
    indices, indptr = _safe_downcast_indices(A)

    # 设置超级 LU 分解的选项，包括对角元素枢轴阈值、列置换策略、面板尺寸和松弛因子
    _options = dict(DiagPivotThresh=diag_pivot_thresh, ColPerm=permc_spec,
                    PanelSize=panel_size, Relax=relax)
    
    # 如果提供了额外的选项，则更新默认选项
    if options is not None:
        _options.update(options)

    # 确保不对列进行任何置换
    if (_options["ColPerm"] == "NATURAL"):
        _options["SymmetricMode"] = True

    # 调用超级 LU 的 gstrf 函数进行矩阵分解
    return _superlu.gstrf(N, A.nnz, A.data, indices, indptr,
                          csc_construct_func=csc_construct_func,
                          ilu=False, options=_options)
def spilu(A, drop_tol=None, fill_factor=None, drop_rule=None, permc_spec=None,
          diag_pivot_thresh=None, relax=None, panel_size=None, options=None):
    """
    Compute an incomplete LU decomposition for a sparse, square matrix.

    The resulting object is an approximation to the inverse of `A`.

    Parameters
    ----------
    A : (N, N) array_like
        Sparse matrix to factorize. Most efficient when provided in CSC format.
        Other formats will be converted to CSC before factorization.
    drop_tol : float, optional
        Drop tolerance (0 <= tol <= 1) for an incomplete LU decomposition.
        (default: 1e-4)
    fill_factor : float, optional
        Specifies the fill ratio upper bound (>= 1.0) for ILU. (default: 10)
    drop_rule : str, optional
        Comma-separated string of drop rules to use.
        Available rules: ``basic``, ``prows``, ``column``, ``area``,
        ``secondary``, ``dynamic``, ``interp``. (Default: ``basic,area``)

        See SuperLU documentation for details.

    Remaining other options
        Same as for `splu`

    Returns
    -------
    invA_approx : scipy.sparse.linalg.SuperLU
        Object, which has a ``solve`` method.

    See also
    --------
    splu : complete LU decomposition

    Notes
    -----
    To improve the better approximation to the inverse, you may need to
    increase `fill_factor` AND decrease `drop_tol`.

    This function uses the SuperLU library.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import spilu
    >>> A = csc_matrix([[1., 0., 0.], [5., 0., 2.], [0., -1., 0.]], dtype=float)
    >>> B = spilu(A)
    >>> x = np.array([1., 2., 3.], dtype=float)
    >>> B.solve(x)
    array([ 1. , -3. , -1.5])
    >>> A.dot(B.solve(x))
    array([ 1.,  2.,  3.])
    >>> B.solve(A.dot(x))
    array([ 1.,  2.,  3.])
    """

    if is_pydata_spmatrix(A):
        # 如果 A 是 PyData sparse matrix，则定义一个函数将其转换为 CSC 格式
        def csc_construct_func(*a, cls=type(A)):
            return cls.from_scipy_sparse(csc_matrix(*a))
        # 将 A 转换为 CSC 格式
        A = A.to_scipy_sparse().tocsc()
    else:
        # 否则，默认的构造函数是 csc_matrix
        csc_construct_func = csc_matrix

    if not (issparse(A) and A.format == "csc"):
        # 如果 A 不是稀疏矩阵或者不是 CSC 格式，则转换为 CSC 格式
        A = csc_matrix(A)
        warn('spilu converted its input to CSC format',
             SparseEfficiencyWarning, stacklevel=2)

    # 对于非标准格式，合并重复元素
    A.sum_duplicates()
    # 将 A 转换为浮点数格式
    A = A._asfptype()

    M, N = A.shape
    if (M != N):
        # 如果矩阵不是方阵，则抛出 ValueError
        raise ValueError("can only factor square matrices")  # is this true?

    # 安全下降转换索引
    indices, indptr = _safe_downcast_indices(A)

    # 设置选项字典，包括 ILU 的参数和用户提供的额外选项
    _options = dict(ILU_DropRule=drop_rule, ILU_DropTol=drop_tol,
                    ILU_FillFactor=fill_factor,
                    DiagPivotThresh=diag_pivot_thresh, ColPerm=permc_spec,
                    PanelSize=panel_size, Relax=relax)
    if options is not None:
        _options.update(options)

    # 确保不应用列置换
    # 如果选项字典中的 "ColPerm" 键对应的值为 "NATURAL"
    if (_options["ColPerm"] == "NATURAL"):
        # 将选项字典中的 "SymmetricMode" 键设为 True
        _options["SymmetricMode"] = True
    
    # 调用 _superlu.gstrf 函数进行稀疏矩阵分解
    return _superlu.gstrf(N, A.nnz, A.data, indices, indptr,
                          # 传递稀疏矩阵的构造函数
                          csc_construct_func=csc_construct_func,
                          # 设置 ILU 预处理为 True
                          ilu=True,
                          # 将选项字典作为参数传递
                          options=_options)
# 返回一个函数，用于解决稀疏线性系统，输入的矩阵 A 需要预先分解
def factorized(A):
    # 检查输入的矩阵 A 是否为 PyData sparse matrix 类型，如果是则转换为 scipy.sparse.csc_matrix 格式
    if is_pydata_spmatrix(A):
        A = A.to_scipy_sparse().tocsc()

    # 如果使用 UMFPACK 解算器
    if useUmfpack:
        # 如果未安装 Scikits.umfpack 报错
        if noScikit:
            raise RuntimeError('Scikits.umfpack not installed.')

        # 如果 A 不是 CSR 格式的稀疏矩阵，转换为 CSC 格式，并发出警告
        if not (issparse(A) and A.format == "csc"):
            A = csc_matrix(A)
            warn('splu converted its input to CSC format',
                 SparseEfficiencyWarning, stacklevel=2)

        # 将 A 转换为浮点型
        A = A._asfptype()

        # 如果 A 的数据类型不是双精度浮点型，则报错
        if A.dtype.char not in 'dD':
            raise ValueError("convert matrix data to double, please, using"
                  " .astype(), or set linsolve.useUmfpack = False")

        # 获取 UMFPACK 求解器类型及其上下文
        umf_family, A = _get_umf_family(A)
        umf = umfpack.UmfpackContext(umf_family)

        # 执行 LU 分解
        umf.numeric(A)

        # 定义用于解决线性方程组的函数 solve
        def solve(b):
            with np.errstate(divide="ignore", invalid="ignore"):
                # 忽略除以零和无效操作的警告，使用 UMFPACK 解算器求解线性方程组
                result = umf.solve(umfpack.UMFPACK_A, A, b, autoTranspose=True)

            return result

        return solve
    else:
        # 如果不使用 UMFPACK 解算器，则使用 scipy.sparse.linalg.splu 求解并返回结果
        return splu(A).solve


# 解决方程组 A * x = b，其中 A 是三角矩阵
def spsolve_triangular(A, b, lower=True, overwrite_A=False, overwrite_b=False,
                       unit_diagonal=False):
    """
    Parameters
    ----------
    A : (M, M) sparse matrix
        A sparse square triangular matrix. Should be in CSR or CSC format.
    b : (M,) or (M, N) array_like
        Right-hand side matrix in ``A x = b``
    lower : bool, optional
        Whether `A` is a lower or upper triangular matrix.
        Default is lower triangular matrix.
    overwrite_A : bool, optional
        Allow changing `A`.
        Enabling gives a performance gain. Default is False.
    overwrite_b : bool, optional
        Allow overwriting data in `b`.
        Enabling gives a performance gain. Default is False.
        If `overwrite_b` is True, it should be ensured that
        `b` has an appropriate dtype to be able to store the result.
    """
    if is_pydata_spmatrix(A):
        A = A.to_scipy_sparse().tocsc()

如果 `A` 是 PyData 稀疏矩阵，则将其转换为 SciPy 的 csc 格式稀疏矩阵。


    trans = "N"
    if issparse(A) and A.format == "csr":
        A = A.T
        trans = "T"
        lower = not lower

如果 `A` 是稀疏矩阵且格式为 "csr"，则将其转置为 "csc" 格式，并设置 `trans` 为 "T"，同时根据 `lower` 参数更新 `lower` 为相反的布尔值。


    if not (issparse(A) and A.format == "csc"): 
        warn('CSC or CSR matrix format is required. Converting to CSC matrix.',
             SparseEfficiencyWarning, stacklevel=2)
        A = csc_matrix(A)
    elif not overwrite_A:
        A = A.copy()

如果 `A` 不是稀疏矩阵或者格式不是 "csc"，则发出警告并将其转换为 "csc" 格式的稀疏矩阵。如果 `overwrite_A` 参数为 False，则复制 `A`。


    M, N = A.shape
    if M != N:
        raise ValueError(
            f'A must be a square matrix but its shape is {A.shape}.')

获取矩阵 `A` 的形状，并检查其是否为方阵，如果不是则引发 ValueError。


    if unit_diagonal:
        with catch_warnings():
            simplefilter('ignore', SparseEfficiencyWarning)
            A.setdiag(1)
    else:
        diag = A.diagonal()
        if np.any(diag == 0):
            raise LinAlgError(
                'A is singular: zero entry on diagonal.')
        invdiag = 1/diag
        if trans == "N":
            A = A @ diags(invdiag)
        else:
            A = (A.T @ diags(invdiag)).T

如果 `unit_diagonal` 为 True，则将 `A` 的对角线元素设置为 1；否则，计算 `A` 的逆对角线元素，并根据 `trans` 的值应用到 `A` 上。


    # sum duplicates for non-canonical format
    A.sum_duplicates()

处理非规范格式的重复元素求和。


    b = np.asanyarray(b)

    if b.ndim not in [1, 2]:
        raise ValueError(
            f'b must have 1 or 2 dims but its shape is {b.shape}.')
    if M != b.shape[0]:
        raise ValueError(
            'The size of the dimensions of A must be equal to '
            'the size of the first dimension of b but the shape of A is '
            f'{A.shape} and the shape of b is {b.shape}.'
        )

将 `b` 转换为 NumPy 数组，并检查其维度是否为 1 或 2。然后检查 `A` 和 `b` 的形状是否匹配。


    result_dtype = np.promote_types(np.promote_types(A.dtype, np.float32), b.dtype)
    if A.dtype != result_dtype:
        A = A.astype(result_dtype)
    if b.dtype != result_dtype:
        b = b.astype(result_dtype)
    elif not overwrite_b:
        b = b.copy()

确定结果的数据类型，并根据需要将 `A` 和 `b` 转换为该数据类型。如果 `overwrite_b` 参数为 False，则复制 `b`。


    if lower:
        L = A
        U = csc_matrix((N, N), dtype=result_dtype)
    else:
        L = eye(N, dtype=result_dtype, format='csc')
        U = A
        U.setdiag(0)

根据 `lower` 参数选择创建 `L` 和 `U`。如果 `lower` 为 True，则 `L` 等于 `A`，`U` 是一个空的 csc 矩阵。否则，`L` 是一个单位对角线的 csc 矩阵，`U` 等于 `A` 并将其对角线置为 0。
    # 调用 _superlu 模块的 gstrs 函数进行超级 LU 分解并求解线性方程组
    x, info = _superlu.gstrs(trans,
                             N, L.nnz, L.data, L.indices, L.indptr,
                             N, U.nnz, U.data, U.indices, U.indptr,
                             b)
    # 检查 info 是否为非零值，如果是则抛出线性代数错误异常
    if info:
        raise LinAlgError('A is singular.')

    # 如果不是单位对角线矩阵，对 x 进行对角线逆矩阵的逐元素乘法
    if not unit_diagonal:
        # 将 invdiag 调整为与 x 形状兼容的形状
        invdiag = invdiag.reshape(-1, *([1] * (len(x.shape) - 1)))
        x = x * invdiag

    # 返回求解得到的 x 向量
    return x
```