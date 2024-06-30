# `D:\src\scipysrc\sympy\sympy\matrices\solvers.py`

```
from sympy.core.function import expand_mul
from sympy.core.symbol import Dummy, uniquely_named_symbol, symbols
from sympy.utilities.iterables import numbered_symbols

from .exceptions import ShapeError, NonSquareMatrixError, NonInvertibleMatrixError
from .eigen import _fuzzy_positive_definite
from .utilities import _get_intermediate_simp, _iszero


def _diagonal_solve(M, rhs):
    """Solves ``Ax = B`` efficiently, where A is a diagonal Matrix,
    with non-zero diagonal entries.

    Examples
    ========

    >>> from sympy import Matrix, eye
    >>> A = eye(2)*2
    >>> B = Matrix([[1, 2], [3, 4]])
    >>> A.diagonal_solve(B) == B/2
    True

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if not M.is_diagonal():
        raise TypeError("Matrix should be diagonal")
    if rhs.rows != M.rows:
        raise TypeError("Size mismatch")

    return M._new(
        rhs.rows, rhs.cols, lambda i, j: rhs[i, j] / M[i, i])


def _lower_triangular_solve(M, rhs):
    """Solves ``Ax = B``, where A is a lower triangular matrix.

    See Also
    ========

    upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    from .dense import MutableDenseMatrix

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if rhs.rows != M.rows:
        raise ShapeError("Matrices size mismatch.")
    if not M.is_lower:
        raise ValueError("Matrix must be lower triangular.")

    dps = _get_intermediate_simp()
    X   = MutableDenseMatrix.zeros(M.rows, rhs.cols)

    for j in range(rhs.cols):
        for i in range(M.rows):
            if M[i, i] == 0:
                raise TypeError("Matrix must be non-singular.")

            X[i, j] = dps((rhs[i, j] - sum(M[i, k]*X[k, j]
                                        for k in range(i))) / M[i, i])

    return M._new(X)

def _lower_triangular_solve_sparse(M, rhs):
    """Solves ``Ax = B``, where A is a lower triangular matrix.

    See Also
    ========

    upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if rhs.rows != M.rows:
        raise ShapeError("Matrices size mismatch.")
    if not M.is_lower:
        raise ValueError("Matrix must be lower triangular.")

    dps  = _get_intermediate_simp()
    rows = [[] for i in range(M.rows)]

    for i, j, v in M.row_list():
        if i > j:
            rows[i].append((j, v))

    X = rhs.as_mutable()

    # The following loop iterates over non-zero entries of the lower triangular matrix M
    # and updates the solution X for each column of rhs.
    for i in range(M.rows):
        for j, v in rows[i]:
            X[i, :] -= v * X[j, :]

    return M._new(X)
    # 遍历 rhs 矩阵的列
    for j in range(rhs.cols):
        # 遍历 rhs 矩阵的行
        for i in range(rhs.rows):
            # 遍历 rows[i] 中的每对 u, v
            for u, v in rows[i]:
                # 更新 X[i, j]，减去 v * X[u, j]
                X[i, j] -= v * X[u, j]

            # 将 X[i, j] 除以 M[i, i] 的值，然后应用 dps 函数处理结果
            X[i, j] = dps(X[i, j] / M[i, i])

    # 返回用 X 创建的新矩阵 M
    return M._new(X)
def _upper_triangular_solve(M, rhs):
    """Solves ``Ax = B``, where A is an upper triangular matrix.

    See Also
    ========

    lower_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    from .dense import MutableDenseMatrix  # 导入MutableDenseMatrix类

    if not M.is_square:  # 检查矩阵M是否为方阵
        raise NonSquareMatrixError("Matrix must be square.")
    if rhs.rows != M.rows:  # 检查rhs的行数与矩阵M的行数是否匹配
        raise ShapeError("Matrix size mismatch.")
    if not M.is_upper:  # 检查矩阵M是否为上三角矩阵
        raise TypeError("Matrix is not upper triangular.")

    dps = _get_intermediate_simp()  # 获取中间简化的精度
    X   = MutableDenseMatrix.zeros(M.rows, rhs.cols)  # 创建零矩阵X，行数与M相同，列数与rhs相同

    for j in range(rhs.cols):  # 遍历rhs的列
        for i in reversed(range(M.rows)):  # 逆序遍历M的行
            if M[i, i] == 0:  # 检查M的对角元素是否为零
                raise ValueError("Matrix must be non-singular.")

            # 解方程 X[i, j] = (rhs[i, j] - sum(M[i, k]*X[k, j] for k in range(i + 1, M.rows))) / M[i, i]
            X[i, j] = dps((rhs[i, j] - sum(M[i, k]*X[k, j] for k in range(i + 1, M.rows))) / M[i, i])

    return M._new(X)  # 返回由X构建的新矩阵M


def _upper_triangular_solve_sparse(M, rhs):
    """Solves ``Ax = B``, where A is an upper triangular matrix.

    See Also
    ========

    lower_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if not M.is_square:  # 检查矩阵M是否为方阵
        raise NonSquareMatrixError("Matrix must be square.")
    if rhs.rows != M.rows:  # 检查rhs的行数与矩阵M的行数是否匹配
        raise ShapeError("Matrix size mismatch.")
    if not M.is_upper:  # 检查矩阵M是否为上三角矩阵
        raise TypeError("Matrix is not upper triangular.")

    dps  = _get_intermediate_simp()  # 获取中间简化的精度
    rows = [[] for i in range(M.rows)]  # 创建一个空列表的列表，长度为M的行数

    for i, j, v in M.row_list():  # 遍历M的行表达式列表
        if i < j:
            rows[i].append((j, v))  # 将非零元素添加到对应的行中

    X = rhs.as_mutable()  # 将rhs转换为可变形式

    for j in range(rhs.cols):  # 遍历rhs的列
        for i in reversed(range(rhs.rows)):  # 逆序遍历rhs的行
            for u, v in reversed(rows[i]):  # 逆序遍历行i中的非零元素
                X[i, j] -= v*X[u, j]  # 更新X[i, j]的值

            X[i, j] = dps(X[i, j] / M[i, i])  # 使用中间简化精度更新X[i, j]

    return M._new(X)  # 返回由X构建的新矩阵M


def _cholesky_solve(M, rhs):
    """Solves ``Ax = B`` using Cholesky decomposition,
    for a general square non-singular matrix.
    For a non-square matrix with rows > cols,
    the least squares solution is returned.

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv_solve
    cramer_solve
    """

    if M.rows < M.cols:  # 如果矩阵M的行数小于列数
        raise NotImplementedError(
            'Under-determined System. Try M.gauss_jordan_solve(rhs)')  # 抛出未实现错误

    hermitian = True  # 假设矩阵M是Hermitian对称的
    reform    = False  # 不需要进行重构

    if M.is_symmetric():  # 如果矩阵M是对称的
        hermitian = False  # 矩阵M不是Hermitian对称的
    elif not M.is_hermitian:  # 如果矩阵M不是Hermitian对称的
        reform = True  # 需要重构

    if reform or _fuzzy_positive_definite(M) is False:  # 如果需要重构或者矩阵M不是模糊正定的
        H         = M.H  # 获取矩阵M的共轭转置
        M         = H.multiply(M)  # 计算H乘以M
        rhs       = H.multiply(rhs)  # 计算H乘以rhs
        hermitian = not M.is_symmetric()  # 更新Hermitian对称性

    L = M.cholesky(hermitian=hermitian)  # 计算矩阵M的Cholesky分解
    # 使用下三角矩阵 L 解方程 Y = L.lower_triangular_solve(rhs)
    Y = L.lower_triangular_solve(rhs)

    # 如果矩阵 L 是 Hermitian 对称的，则返回 L 的共轭转置的上三角矩阵与 Y 的解
    if hermitian:
        return (L.H).upper_triangular_solve(Y)
    # 如果矩阵 L 不是 Hermitian 对称的，则返回 L 的转置的上三角矩阵与 Y 的解
    else:
        return (L.T).upper_triangular_solve(Y)
# 解决线性系统 Ax = B，使用 LDL 分解，适用于一般的方阵且非奇异的情况。
# 对于行数大于列数的非方阵，返回最小二乘解。
def _LDLsolve(M, rhs):
    hermitian = True  # 假设矩阵是对称的
    reform = False  # 不需要重构矩阵

    # 检查矩阵是否对称
    if M.is_symmetric():
        hermitian = False
    elif not M.is_hermitian:
        reform = True  # 需要重构矩阵

    # 如果需要重构或者矩阵不是模糊正定的，进行重构
    if reform or _fuzzy_positive_definite(M) is False:
        H = M.H  # 取 M 的共轭转置
        M = H.multiply(M)  # 计算 H*M
        rhs = H.multiply(rhs)  # 计算 H*rhs
        hermitian = not M.is_symmetric()  # 更新 hermitian 标志

    # 进行 LDL 分解
    L, D = M.LDLdecomposition(hermitian=hermitian)
    Y = L.lower_triangular_solve(rhs)  # 解下三角方程 LY = rhs
    Z = D.diagonal_solve(Y)  # 解对角方程 DZ = Y

    # 根据矩阵是否对称选择求解方式并返回结果
    if hermitian:
        return (L.H).upper_triangular_solve(Z)  # 返回 (L的共轭转置)*Z
    else:
        return (L.T).upper_triangular_solve(Z)  # 返回 L 的转置*Z


# 解决线性系统 Ax = rhs，其中 A = M。
# 这适用于符号矩阵，对于实数或复数矩阵，使用 mpmath.lu_solve 或 mpmath.qr_solve。
def _LUsolve(M, rhs, iszerofunc=_iszero):
    # 检查矩阵和右侧向量的行数是否相等
    if rhs.rows != M.rows:
        raise ShapeError("``M`` and ``rhs`` must have the same number of rows.")

    m = M.rows  # 矩阵 M 的行数
    n = M.cols  # 矩阵 M 的列数

    # 如果行数小于列数，抛出未实现错误
    if m < n:
        raise NotImplementedError("Underdetermined systems not supported.")

    try:
        # 使用 LU 分解求解线性系统
        A, perm = M.LUdecomposition_Simple(iszerofunc=iszerofunc, rankcheck=True)
    except ValueError:
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

    dps = _get_intermediate_simp()  # 获取中间简化的精度
    b = rhs.permute_rows(perm).as_mutable()  # 对 rhs 进行行置换

    # 前向代换，确保所有对角线条目都被缩放为 1
    for i in range(m):
        for j in range(min(i, n)):
            scale = A[i, j]  # 取出对角线元素
            # 更新 b[i, j]，使得其减去另一个乘以 A[i, j]的值
            b.zip_row_op(i, j, lambda x, y: dps(x - y * scale))

    # 对于过定的系统进行一致性检查
    if m > n:
        for i in range(n, m):
            for j in range(b.cols):
                if not iszerofunc(b[i, j]):
                    raise ValueError("The system is inconsistent.")

        b = b[0:n, :]  # 如果一致，截断零行
    # 向后代入法（Backward Substitution）
    
    # 从最后一行开始向上遍历
    for i in range(n - 1, -1, -1):
        # 对当前行 i 进行操作，处理列索引大于 i 的元素
        for j in range(i + 1, n):
            # 获取 A[i, j] 的值作为缩放因子
            scale = A[i, j]
            # 调用 b 对象的 zip_row_op 方法，将行 i 和行 j 进行线性组合
            b.zip_row_op(i, j, lambda x, y: dps(x - y * scale))
    
        # 获取 A[i, i] 的值作为缩放因子
        scale = A[i, i]
        # 调用 b 对象的 row_op 方法，将行 i 进行缩放操作，将其元素除以 scale
        b.row_op(i, lambda x, _: dps(x / scale))
    
    # 返回 rhs 对象的一个新实例，其内容由 b 对象生成
    return rhs.__class__(b)
def _QRsolve(M, b):
    """解线性系统 ``Ax = b`` 的方程。

    ``M`` 是矩阵 ``A``，方法的参数是向量 ``b``。该方法返回解向量 ``x``。
    如果 ``b`` 是一个矩阵，则对每列 ``b`` 解决系统，并返回与 ``b`` 相同形状的矩阵。

    与 LUsolve 方法相比，该方法速度较慢（大约慢两倍），但对浮点运算更稳定。
    然而，LUsolve 通常使用精确算法，因此您不需要使用 QRsolve。

    主要用于教育目的和符号矩阵，对于实数（或复数）矩阵，请使用 mpmath.qr_solve。

    参见
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    pinv_solve
    QRdecomposition
    cramer_solve
    """

    # 获取中间简化表达式的位数
    dps = _get_intermediate_simp(expand_mul, expand_mul)
    
    # 对 M 进行 QR 分解
    Q, R = M.QRdecomposition()
    
    # 计算 y = Q^T * b
    y = Q.T * b

    # 回代法解决 R*x = y：
    # 我们在向量 'x' 中以反向方式构建结果，最后才将其反转。
    x = []
    n = R.rows

    for j in range(n - 1, -1, -1):
        tmp = y[j, :]

        for k in range(j + 1, n):
            tmp -= R[j, k] * x[n - 1 - k]

        # 使用 dps 函数简化 tmp
        tmp = dps(tmp)

        # 计算并添加 x[j]
        x.append(tmp / R[j, j])

    # 将结果垂直堆叠为矩阵并返回
    return M.vstack(*x[::-1])



def _gauss_jordan_solve(M, B, freevar=False):
    """
    使用 Gauss Jordan 消元法解 ``Ax = B`` 方程。

    可能存在零个、一个或无限多个解。如果存在一个解，将返回该解。
    如果存在无限多个解，将以参数形式返回。如果不存在解，则会引发 ValueError。

    参数
    ==========

    B : Matrix
        要解的方程的右手边。必须与矩阵 A 具有相同的行数。

    freevar : boolean, optional
        标志，当设置为 `True` 时，将返回解中自由变量的索引（列矩阵），
        用于系统不定（例如 A 的列数多于行数），可能存在无限多个解的情况，
        以任意自由变量的值表示。默认为 `False`。

    返回
    =======

    x : Matrix
        满足 ``Ax = B`` 的矩阵。将具有与矩阵 A 相同的列数，并与矩阵 B 相同的列数。

    params : Matrix
        如果系统不定（例如 A 的列数多于行数），则可能存在无限多个解，
        以任意参数返回这些参数。这些任意参数作为 params 矩阵返回。
    """
    # 定义一个可选参数 `free_var_index`，用于存储自由变量的索引列表
    free_var_index : List, optional
        If the system is underdetermined (e.g. A has more columns than
        rows), infinite solutions are possible, in terms of arbitrary
        values of free variables. Then the indices of the free variables
        in the solutions (column Matrix) are returned by free_var_index,
        if the flag `freevar` is set to `True`.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 1, 1], [1, 2, 2, -1], [2, 4, 0, 6]])
    >>> B = Matrix([7, 12, 4])
    >>> sol, params = A.gauss_jordan_solve(B)
    >>> sol
    Matrix([
    [-2*tau0 - 3*tau1 + 2],
    [                 tau0],
    [           2*tau1 + 5],
    [                 tau1]])
    >>> params
    Matrix([
    [tau0],
    [tau1]])
    >>> taus_zeroes = { tau:0 for tau in params }
    >>> sol_unique = sol.xreplace(taus_zeroes)
    >>> sol_unique
        Matrix([
    [2],
    [0],
    [5],
    [0]])


    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    >>> B = Matrix([3, 6, 9])
    >>> sol, params = A.gauss_jordan_solve(B)
    >>> sol
    Matrix([
    [-1],
    [ 2],
    [ 0]])
    >>> params
    Matrix(0, 1, [])

    >>> A = Matrix([[2, -7], [-1, 4]])
    >>> B = Matrix([[-21, 3], [12, -2]])
    >>> sol, params = A.gauss_jordan_solve(B)
    >>> sol
    Matrix([
    [0, -2],
    [3, -1]])
    >>> params
    Matrix(0, 2, [])


    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 1, 1], [1, 2, 2, -1], [2, 4, 0, 6]])
    >>> B = Matrix([7, 12, 4])
    >>> sol, params, freevars = A.gauss_jordan_solve(B, freevar=True)
    >>> sol
    Matrix([
    [-2*tau0 - 3*tau1 + 2],
    [                 tau0],
    [           2*tau1 + 5],
    [                 tau1]])
    >>> params
    Matrix([
    [tau0],
    [tau1]])
    >>> freevars
    [1, 3]

    # 显示通过高斯约当消元法求解线性方程组，并得到特解及参数化解

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gaussian_elimination

    """

    from sympy.matrices import Matrix, zeros

    # 获取 Matrix 类的引用
    cls      = M.__class__
    # 将增广矩阵构造为 M 与 B 水平叠加的结果
    aug      = M.hstack(M.copy(), B.copy())
    # B 的列数
    B_cols   = B.cols
    # 计算增广矩阵的行数和列数
    row, col = aug[:, :-B_cols].shape

    # 使用简化的行阶梯形式求解线性方程组
    A, pivots = aug.rref(simplify=True)
    # 将 A 矩阵与解向量分开
    A, v      = A[:, :-B_cols], A[:, -B_cols:]
    # 找出主元列的索引
    pivots    = list(filter(lambda p: p < col, pivots))
    # 计算主元的个数（即方程组的秩）
    rank      = len(pivots)

    # 获取自由符号的索引列表
    # 非主元列即为自由变量
    free_var_index = [c for c in range(A.cols) if c not in pivots]

    # 将矩阵转换为块形式
    permutation = Matrix(pivots + free_var_index).T

    # 检查解是否存在
    # 增广矩阵的秩应该等于系数矩阵的秩
    if not v[rank:, :].is_zero_matrix:
        raise ValueError("Linear system has no solution")

    # 自由参数
    # 获取当前未编号的自由符号名称列表
    name = uniquely_named_symbol('tau', [aug],
            compare=lambda i: str(i).rstrip('1234567890'),
            modify=lambda s: '_' + s).name
    
    # 创建一个生成器对象，生成带编号的符号名称
    gen  = numbered_symbols(name)
    
    # 使用生成器创建一个矩阵，填充元素为按顺序生成的符号
    tau  = Matrix([next(gen) for k in range((col - rank)*B_cols)]).reshape(
            col - rank, B_cols)

    # 构建完整的参数化解
    V        = A[:rank, free_var_index]  # 从矩阵 A 中选取特定行列的子矩阵 V
    vt       = v[:rank, :]               # 从向量 v 中选取特定行的子向量 vt
    free_sol = tau.vstack(vt - V * tau, tau)  # 使用 tau 构建一个垂直堆叠的解向量

    # 恢复原始排列顺序
    sol = zeros(col, B_cols)  # 创建一个零矩阵，用于存储最终的解

    for k in range(col):
        sol[permutation[k], :] = free_sol[k,:]  # 将 free_sol 中的解按照 permutation 的顺序复制到 sol 中

    # 将解和 tau 封装成特定的类对象
    sol, tau = cls(sol), cls(tau)

    # 如果需要自由变量的索引，则返回解和 tau 以及自由变量的索引；否则只返回解和 tau
    if freevar:
        return sol, tau, free_var_index
    else:
        return sol, tau
# 使用Moore-Penrose伪逆解决“Ax = B”形式的线性方程组问题

def _pinv_solve(M, B, arbitrary_matrix=None):
    """Solve ``Ax = B`` using the Moore-Penrose pseudoinverse.

    There may be zero, one, or infinite solutions.  If one solution
    exists, it will be returned.  If infinite solutions exist, one will
    be returned based on the value of arbitrary_matrix.  If no solutions
    exist, the least-squares solution is returned.

    Parameters
    ==========

    M : Matrix
        The coefficient matrix A in the equation Ax = B.
    B : Matrix
        The right hand side of the equation to be solved for.  Must have
        the same number of rows as matrix A.
    arbitrary_matrix : Matrix, optional
        If the system is underdetermined (e.g. A has more columns than
        rows), infinite solutions are possible, in terms of an arbitrary
        matrix.  This parameter may be set to a specific matrix to use
        for that purpose; if so, it must be the same shape as x, with as
        many rows as matrix A has columns, and as many columns as matrix
        B.  If left as None, an appropriate matrix containing dummy
        symbols in the form of ``wn_m`` will be used, with n and m being
        row and column position of each symbol.

    Returns
    =======

    x : Matrix
        The matrix that will satisfy ``Ax = B``.  Will have as many rows as
        matrix A has columns, and as many columns as matrix B.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
    >>> B = Matrix([7, 8])
    >>> A.pinv_solve(B)
    Matrix([
    [ _w0_0/6 - _w1_0/3 + _w2_0/6 - 55/18],
    [-_w0_0/3 + 2*_w1_0/3 - _w2_0/3 + 1/9],
    [ _w0_0/6 - _w1_0/3 + _w2_0/6 + 59/18]])
    >>> A.pinv_solve(B, arbitrary_matrix=Matrix([0, 0, 0]))
    Matrix([
    [-55/18],
    [   1/9],
    [ 59/18]])

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.lower_triangular_solve
    sympy.matrices.dense.DenseMatrix.upper_triangular_solve
    gauss_jordan_solve
    cholesky_solve
    diagonal_solve
    LDLsolve
    LUsolve
    QRsolve
    pinv

    Notes
    =====

    This may return either exact solutions or least squares solutions.
    To determine which, check ``A * A.pinv() * B == B``.  It will be
    True if exact solutions exist, and False if only a least-squares
    solution exists.  Be aware that the left hand side of that equation
    may need to be simplified to correctly compare to the right hand
    side.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Moore-Penrose_pseudoinverse#Obtaining_all_solutions_of_a_linear_system

    """

    from sympy.matrices import eye

    # 将输入的M作为系数矩阵A
    A      = M
    # 计算矩阵A的Moore-Penrose伪逆
    A_pinv = M.pinv()

    # 如果未提供arbitrary_matrix，则生成一个包含dummy符号的适当矩阵
    if arbitrary_matrix is None:
        # 获取矩阵A和B的行数和列数
        rows, cols       = A.cols, B.cols
        # 生成符号w，用作dummy符号的标识
        w                = symbols('w:{}_:{}'.format(rows, cols), cls=Dummy)
        # 构造一个与矩阵A的列数相同的矩阵，用于表示任意矩阵
        arbitrary_matrix = M.__class__(cols, rows, w).T

    # 返回解x，根据Moore-Penrose公式计算
    return A_pinv.multiply(B) + (eye(A.cols) -
            A_pinv.multiply(A)).multiply(arbitrary_matrix)
    """Solves system of linear equations using Cramer's rule.
    
    This method is relatively inefficient compared to other methods.
    However it only uses a single division, assuming a division-free determinant
    method is provided. This is helpful to minimize the chance of divide-by-zero
    cases in symbolic solutions to linear systems.
    
    Parameters
    ==========
    M : Matrix
        The matrix representing the left hand side of the equation.
    rhs : Matrix
        The matrix representing the right hand side of the equation.
    det_method : str or callable
        The method to use to calculate the determinant of the matrix.
        The default is ``'laplace'``.  If a callable is passed, it should take a
        single argument, the matrix, and return the determinant of the matrix.
    
    Returns
    =======
    x : Matrix
        The matrix that will satisfy ``Ax = B``.  Will have as many rows as
        matrix A has columns, and as many columns as matrix B.
    
    Examples
    ========
    >>> from sympy import Matrix
    >>> A = Matrix([[0, -6, 1], [0, -6, -1], [-5, -2, 3]])
    >>> B = Matrix([[-30, -9], [-18, -27], [-26, 46]])
    >>> x = A.cramer_solve(B)
    >>> x
    Matrix([
    [ 0, -5],
    [ 4,  3],
    [-6,  9]])
    
    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/Cramer%27s_rule#Explicit_formulas_for_small_systems
    
    """
    # 导入必要的模块和函数
    from .dense import zeros
    
    # 定义一个函数，用于获取矩阵中特定位置的元素
    def entry(i, j):
        return rhs[i, sol] if j == col else M[i, j]
    
    # 根据传入的 det_method 参数选择不同的 determinant 计算方法
    if det_method == "bird":
        # 如果选择了 "bird" 方法，则导入相应的函数
        from .determinant import _det_bird
        det = _det_bird
    elif det_method == "laplace":
        # 如果选择了 "laplace" 方法，则导入相应的函数
        from .determinant import _det_laplace
        det = _det_laplace
    elif isinstance(det_method, str):
        # 如果 det_method 是字符串，使用 lambda 函数调用 matrix.det(method=det_method) 方法
        det = lambda matrix: matrix.det(method=det_method)
    else:
        # 否则，直接使用传入的 det_method 函数
        det = det_method
    
    # 计算矩阵 M 的 determinant
    det_M = det(M)
    
    # 初始化结果矩阵 x，使用与 rhs 相同的形状
    x = zeros(*rhs.shape)
    
    # 遍历 rhs 矩阵的每一列
    for sol in range(rhs.shape[1]):
        # 遍历 rhs 矩阵的每一行
        for col in range(rhs.shape[0]):
            # 计算 x[col, sol] 的值，应用 Cramer's rule
            x[col, sol] = det(M.__class__(*M.shape, entry)) / det_M
    
    # 返回结果矩阵，使用与 M 相同的类来创建新的矩阵
    return M.__class__(x)
def _solve(M, rhs, method='GJ'):
    """Solves linear equations for a unique solution.

    Parameters
    ==========

    M : Matrix
        Coefficient matrix of the linear equations.

    rhs : Matrix
        Vector representing the right-hand side of the linear equation.

    method : string, optional
        Specifies the method to use for solving the linear equations:
        
        - If 'GJ' or 'GE', Gauss-Jordan elimination will be used via 
          `M.gauss_jordan_solve(rhs)`.
        
        - If 'LU', solves using LU decomposition (`M.LUsolve(rhs)`).
        
        - If 'CH', solves using Cholesky decomposition (`M.cholesky_solve(rhs)`).
        
        - If 'QR', solves using QR decomposition (`M.QRsolve(rhs)`).
        
        - If 'LDL', solves using LDL decomposition (`M.LDLsolve(rhs)`).
        
        - If 'PINV', solves using pseudo-inverse (`M.pinv_solve(rhs)`).
        
        - If 'CRAMER', solves using Cramer's rule (`M.cramer_solve(rhs)`).
        
        - For other methods, computes solution via inverse (`M.inv(method=method).multiply(rhs)`).

        Each method has specific requirements and behavior, detailed in respective functions.

    Returns
    =======

    solutions : Matrix
        Vector representing the solution to the linear equations.

    Raises
    ======

    ValueError
        If the matrix M is not square or if the system does not have a unique solution.
        In such cases, appropriate error messages are raised.

    NonInvertibleMatrixError
        Raised when the matrix is found to be non-invertible during computation,
        specifically when using 'GJ' or 'GE' methods.

        The error message suggests using `M.gauss_jordan_solve(rhs)` to obtain a parametric solution
        if the determinant of M is zero.

    """

    if method in ('GJ', 'GE'):
        try:
            # Attempt to solve using Gauss-Jordan elimination
            soln, param = M.gauss_jordan_solve(rhs)

            if param:
                raise NonInvertibleMatrixError("Matrix det == 0; not invertible. "
                "Try ``M.gauss_jordan_solve(rhs)`` to obtain a parametric solution.")

        except ValueError:
            raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

        return soln

    elif method == 'LU':
        # Solve using LU decomposition
        return M.LUsolve(rhs)
    elif method == 'CH':
        # Solve using Cholesky decomposition
        return M.cholesky_solve(rhs)
    elif method == 'QR':
        # Solve using QR decomposition
        return M.QRsolve(rhs)
    elif method == 'LDL':
        # Solve using LDL decomposition
        return M.LDLsolve(rhs)
    elif method == 'PINV':
        # Solve using pseudo-inverse
        return M.pinv_solve(rhs)
    elif method == 'CRAMER':
        # Solve using Cramer's rule
        return M.cramer_solve(rhs)
    else:
        # Solve using specified inverse method
        return M.inv(method=method).multiply(rhs)


def _solve_least_squares(M, rhs, method='CH'):
    """Return the least-square fit to the data.

    Parameters
    ==========

    M : Matrix
        Coefficient matrix for the least squares problem.

    rhs : Matrix
        Vector representing the right-hand side of the least squares problem.

    method : string or boolean, optional
        Specifies the method to use for solving the least squares problem:

        - If 'CH', solves using Cholesky decomposition (`M.cholesky_solve(rhs)`).
        
        - If 'LDL', solves using LDL decomposition (`M.LDLsolve(rhs)`).
        
        - If 'QR', solves using QR decomposition (`M.QRsolve(rhs)`).
        
        - If 'PINV', solves using pseudo-inverse (`M.pinv_solve(rhs)`).
        
        - Otherwise, the conjugate of M is used to create a system of equations,
          passed to `solve` with the hint defined by `method`.

    Returns
    =======

    solutions : Matrix
        Vector representing the solution to the least squares problem.
    """
    # 如果方法为 'CH'，使用 Cholesky 分解法求解线性方程组 Mx = rhs
    if method == 'CH':
        return M.cholesky_solve(rhs)
    # 如果方法为 'QR'，使用 QR 分解法求解线性方程组 Mx = rhs
    elif method == 'QR':
        return M.QRsolve(rhs)
    # 如果方法为 'LDL'，使用 LDL 分解法求解线性方程组 Mx = rhs
    elif method == 'LDL':
        return M.LDLsolve(rhs)
    # 如果方法为 'PINV'，使用广义逆求解线性方程组 Mx = rhs
    elif method == 'PINV':
        return M.pinv_solve(rhs)
    else:
        # 否则，计算 M 的共轭转置并解线性方程组 (M^H * M)x = M^H * rhs，方法由参数指定
        t = M.H
        return (t * M).solve(t * rhs, method=method)
```