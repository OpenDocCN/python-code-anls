# `D:\src\scipysrc\sympy\sympy\matrices\inverse.py`

```
# 从 sympy.polys.matrices.exceptions 模块中导入 DMNonInvertibleMatrixError 异常
# 从 sympy.polys.domains 模块中导入 EX
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError
from sympy.polys.domains import EX

# 从当前包中导入 MatrixError, NonSquareMatrixError, NonInvertibleMatrixError 异常
# 从当前包中导入 _iszero 函数
from .exceptions import MatrixError, NonSquareMatrixError, NonInvertibleMatrixError
from .utilities import _iszero

# 定义一个名为 _pinv_full_rank 的函数，用于处理完全行或列秩矩阵的伪逆运算
def _pinv_full_rank(M):
    """Subroutine for full row or column rank matrices.

    For full row rank matrices, inverse of ``A * A.H`` Exists.
    For full column rank matrices, inverse of ``A.H * A`` Exists.

    This routine can apply for both cases by checking the shape
    and have small decision.
    """

    # 如果矩阵 M 是零矩阵，则返回其转置
    if M.is_zero_matrix:
        return M.H

    # 如果矩阵 M 的行数大于等于列数，则计算 (M.H * M).inv() * M.H
    if M.rows >= M.cols:
        return M.H.multiply(M).inv().multiply(M.H)
    else:
        # 否则，计算 M.H * M * (M.H * M).inv()
        return M.H.multiply(M.multiply(M.H).inv())

# 定义一个名为 _pinv_rank_decomposition 的函数，用于处理秩分解的伪逆运算
def _pinv_rank_decomposition(M):
    """Subroutine for rank decomposition

    With rank decompositions, `A` can be decomposed into two full-
    rank matrices, and each matrix can take pseudoinverse
    individually.
    """

    # 如果矩阵 M 是零矩阵，则返回其转置
    if M.is_zero_matrix:
        return M.H

    # 对矩阵 M 进行秩分解得到 B 和 C
    B, C = M.rank_decomposition()

    # 分别计算 B 和 C 的完全秩伪逆
    Bp = _pinv_full_rank(B)
    Cp = _pinv_full_rank(C)

    # 返回 C 的完全秩伪逆乘以 B 的完全秩伪逆
    return Cp.multiply(Bp)

# 定义一个名为 _pinv_diagonalization 的函数，用于使用对角化方法的伪逆运算
def _pinv_diagonalization(M):
    """Subroutine using diagonalization

    This routine can sometimes fail if SymPy's eigenvalue
    computation is not reliable.
    """

    # 如果矩阵 M 是零矩阵，则返回其转置
    if M.is_zero_matrix:
        return M.H

    # 将 M 赋值给 A，M 的转置赋值给 AH
    A  = M
    AH = M.H

    try:
        # 如果矩阵 M 的行数大于等于列数，则进行 A.H * A 的对角化
        if M.rows >= M.cols:
            P, D   = AH.multiply(A).diagonalize(normalize=True)
            D_pinv = D.applyfunc(lambda x: 0 if _iszero(x) else 1 / x)

            # 返回 P * D_pinv * P.H * AH
            return P.multiply(D_pinv).multiply(P.H).multiply(AH)

        else:
            # 否则，进行 A * A.H 的对角化
            P, D   = A.multiply(AH).diagonalize(normalize=True)
            D_pinv = D.applyfunc(lambda x: 0 if _iszero(x) else 1 / x)

            # 返回 AH * P * D_pinv * P.H
            return AH.multiply(P).multiply(D_pinv).multiply(P.H)

    except MatrixError:
        # 如果出现 MatrixError 异常，则抛出 NotImplementedError
        raise NotImplementedError(
            'pinv for rank-deficient matrices where '
            'diagonalization of A.H*A fails is not supported yet.')

# 定义一个名为 _pinv 的函数，用于计算矩阵的 Moore-Penrose 伪逆
def _pinv(M, method='RD'):
    """Calculate the Moore-Penrose pseudoinverse of the matrix.

    The Moore-Penrose pseudoinverse exists and is unique for any matrix.
    If the matrix is invertible, the pseudoinverse is the same as the
    inverse.

    Parameters
    ==========

    method : String, optional
        Specifies the method for computing the pseudoinverse.

        If ``'RD'``, Rank-Decomposition will be used.

        If ``'ED'``, Diagonalization will be used.

    Examples
    ========

    Computing pseudoinverse by rank decomposition :

    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2, 3], [4, 5, 6]])
    >>> A.pinv()
    Matrix([
    [-17/18,  4/9],
    [  -1/9,  1/9],
    [ 13/18, -2/9]])

    Computing pseudoinverse by diagonalization :

    >>> B = A.pinv(method='ED')
    >>> B.simplify()
    >>> B
    Matrix([
    [-17/18,  4/9],
    [  -1/9,  1/9],
    [ 13/18, -2/9]])

    See Also
    ========

    inv
    pinv_solve

    References
    """

    # 如果选择使用秩分解方法计算伪逆
    if method == 'RD':
        return _pinv_rank_decomposition(M)
    # 如果选择使用对角化方法计算伪逆
    elif method == 'ED':
        return _pinv_diagonalization(M)
    else:
        raise ValueError("Unknown method '{}' for calculating pseudoinverse.".format(method))
    # 处理 Moore-Penrose 伪逆的函数，参考维基百科[1]中的定义和方法
    
    # 特殊情况：零矩阵的伪逆是其转置矩阵。
    if M.is_zero_matrix:
        # 返回矩阵 M 的共轭转置作为伪逆结果
        return M.H
    
    # 根据指定的方法计算矩阵的伪逆
    if method == 'RD':
        # 使用分解法计算矩阵 M 的伪逆
        return _pinv_rank_decomposition(M)
    elif method == 'ED':
        # 使用对角化法计算矩阵 M 的伪逆
        return _pinv_diagonalization(M)
    else:
        # 若方法参数不合法，抛出数值错误异常
        raise ValueError('invalid pinv method %s' % repr(method))
# 初步检查矩阵是否可逆，提取行列式用于 _inv_ADJ 函数
def _verify_invertible(M, iszerofunc=_iszero):
    """Initial check to see if a matrix is invertible. Raises or returns
    determinant for use in _inv_ADJ."""

    # 如果矩阵不是方阵，则抛出异常
    if not M.is_square:
        raise NonSquareMatrixError("A Matrix must be square to invert.")

    # 计算矩阵行列式（使用 berkowitz 方法）
    d    = M.det(method='berkowitz')
    # 判断行列式是否为零
    zero = d.equals(0)

    # 如果 equals() 方法无法判断，尝试进行行简化
    if zero is None: # if equals() can't decide, will rref be able to?
        # 对矩阵进行行简化，并获取简化后的矩阵
        ok   = M.rref(simplify=True)[0]
        # 检查对角线元素是否存在为零的情况
        zero = any(iszerofunc(ok[j, j]) for j in range(ok.rows))

    # 如果行列式为零，则抛出异常
    if zero:
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

    # 返回矩阵的行列式
    return d

# 使用伴随矩阵和行列式计算逆矩阵
def _inv_ADJ(M, iszerofunc=_iszero):
    """Calculates the inverse using the adjugate matrix and a determinant.

    See Also
    ========

    inv
    inverse_GE
    inverse_LU
    inverse_CH
    inverse_LDL
    """

    # 首先进行矩阵可逆性检查，并获取矩阵行列式
    d = _verify_invertible(M, iszerofunc=iszerofunc)

    # 返回伴随矩阵除以行列式的结果作为逆矩阵
    return M.adjugate() / d

# 使用高斯消元法计算逆矩阵
def _inv_GE(M, iszerofunc=_iszero):
    """Calculates the inverse using Gaussian elimination.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_LU
    inverse_CH
    inverse_LDL
    """

    # 导入密集矩阵类
    from .dense import Matrix

    # 如果矩阵不是方阵，则抛出异常
    if not M.is_square:
        raise NonSquareMatrixError("A Matrix must be square to invert.")

    # 将原矩阵与单位矩阵水平堆叠
    big = Matrix.hstack(M.as_mutable(), Matrix.eye(M.rows))
    # 对堆叠后的矩阵进行行简化
    red = big.rref(iszerofunc=iszerofunc, simplify=True)[0]

    # 检查简化后的矩阵对角线元素是否存在为零的情况
    if any(iszerofunc(red[j, j]) for j in range(red.rows)):
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

    # 返回简化后矩阵的右侧部分作为逆矩阵
    return M._new(red[:, big.rows:])

# 使用LU分解计算逆矩阵
def _inv_LU(M, iszerofunc=_iszero):
    """Calculates the inverse using LU decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_CH
    inverse_LDL
    """

    # 如果矩阵不是方阵，则抛出异常
    if not M.is_square:
        raise NonSquareMatrixError("A Matrix must be square to invert.")
    # 如果矩阵包含符号变量，则进行可逆性检查
    if M.free_symbols:
        _verify_invertible(M, iszerofunc=iszerofunc)

    # 使用LU分解解线性方程组得到逆矩阵
    return M.LUsolve(M.eye(M.rows), iszerofunc=_iszero)

# 使用Cholesky分解计算逆矩阵
def _inv_CH(M, iszerofunc=_iszero):
    """Calculates the inverse using cholesky decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_LU
    inverse_LDL
    """

    # 进行矩阵可逆性检查
    _verify_invertible(M, iszerofunc=iszerofunc)

    # 使用Cholesky分解解线性方程组得到逆矩阵
    return M.cholesky_solve(M.eye(M.rows))

# 使用LDL分解计算逆矩阵
def _inv_LDL(M, iszerofunc=_iszero):
    """Calculates the inverse using LDL decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_LU
    inverse_CH
    """

    # 进行矩阵可逆性检查
    _verify_invertible(M, iszerofunc=iszerofunc)

    # 使用LDL分解解线性方程组得到逆矩阵
    return M.LDLsolve(M.eye(M.rows))

# 使用QR分解计算逆矩阵
def _inv_QR(M, iszerofunc=_iszero):
    """Calculates the inverse using QR decomposition.

    See Also
    ========

    inv
    inverse_ADJ
    inverse_GE
    inverse_CH
    inverse_LDL
    """

    # 进行矩阵可逆性检查
    _verify_invertible(M, iszerofunc=iszerofunc)

    # 使用QR分解解线性方程组得到逆矩阵
    return M.QRsolve(M.eye(M.rows))

# 尝试将矩阵转换为域矩阵（DomainMatrix）
def _try_DM(M, use_EX=False):
    """Try to convert a matrix to a ``DomainMatrix``."""
    # 将矩阵转换为域矩阵
    dM = M.to_DM()
    # 获取域
    K = dM.domain
    # 如果不使用 EX 并且 K.is_EXRAW 为真，则返回 None
    if not use_EX and K.is_EXRAW:
        return None
    # 如果 K.is_EXRAW 为真，则将 dM 转换为 EX 后返回
    elif K.is_EXRAW:
        return dM.convert_to(EX)
    # 如果上述条件都不满足，则返回 dM
    else:
        return dM
# 使用 DomainMatrix 计算给定矩阵 dM 的逆矩阵。
def _inv_DM(dM, cancel=True):
    # 获取矩阵 dM 的行数 m 和列数 n
    m, n = dM.shape
    # 如果矩阵不是方阵，则抛出异常
    if m != n:
        raise NonSquareMatrixError("A Matrix must be square to invert.")

    try:
        # 尝试使用 dM.inv_den() 方法计算逆矩阵 dMi 和分母 den
        dMi, den = dM.inv_den()
    except DMNonInvertibleMatrixError:
        # 如果矩阵不可逆，则抛出异常
        raise NonInvertibleMatrixError("Matrix det == 0; not invertible.")

    if cancel:
        # 如果需要取消分母，则将 dMi 转换为域，并与 den 取商，然后转换为 Matrix 对象
        if not dMi.domain.is_Field:
            dMi = dMi.to_field()
        Mi = (dMi / den).to_Matrix()
    else:
        # 如果不取消分母，则直接将 dMi 转换为 Matrix，并除以 den
        Mi = dMi.to_Matrix() / dMi.domain.to_sympy(den)

    # 返回计算得到的逆矩阵 Mi
    return Mi

# 使用 BLOCKWISE 方法计算给定矩阵 M 的逆矩阵。
def _inv_block(M, iszerofunc=_iszero):
    # 导入 BlockMatrix 类
    from sympy.matrices.expressions.blockmatrix import BlockMatrix
    # 获取矩阵 M 的行数 i
    i = M.shape[0]
    # 如果矩阵行数小于等于 20，则使用 LU 方法计算逆矩阵
    if i <= 20:
        return M.inv(method="LU", iszerofunc=_iszero)
    # 分块拆分矩阵 M
    A = M[:i // 2, :i // 2]
    B = M[:i // 2, i // 2:]
    C = M[i // 2:, :i // 2]
    D = M[i // 2:, i // 2:]
    try:
        # 递归计算 D 块的逆矩阵 D_inv
        D_inv = _inv_block(D)
    except NonInvertibleMatrixError:
        # 如果 D 块不可逆，则使用 LU 方法计算整体的逆矩阵
        return M.inv(method="LU", iszerofunc=_iszero)
    # 计算 B*D_inv
    B_D_i = B * D_inv
    # 计算 B*D_inv*C
    BDC = B_D_i * C
    # 计算 A - B*D_inv*C，并递归计算其逆矩阵 A_n
    A_n = A - BDC
    try:
        A_n = _inv_block(A_n)
    except NonInvertibleMatrixError:
        # 如果 A_n 不可逆，则使用 LU 方法计算整体的逆矩阵
        return M.inv(method="LU", iszerofunc=_iszero)
    # 计算 -A_n*B_D_inv
    B_n = -A_n * B_D_i
    # 计算 D_inv*C*(-A_n)
    dc = D_inv * C
    C_n = -dc * A_n
    # 计算 D_inv + D_inv*C*(-B_n)
    D_n = D_inv + dc * -B_n
    # 构建新的 BlockMatrix nn，并转换为明确的 Matrix 对象
    nn = BlockMatrix([[A_n, B_n], [C_n, D_n]]).as_explicit()
    # 返回计算得到的逆矩阵 nn
    return nn

# 根据指定的方法计算矩阵 M 的逆矩阵。
def _inv(M, method=None, iszerofunc=_iszero, try_block_diag=False):
    """
    根据指定的方法返回矩阵的逆矩阵。默认情况下使用 DM 方法（如果找到合适的域），
    否则对于密集矩阵使用 GE，对于稀疏矩阵使用 LDL。

    Parameters
    ==========

    method : ('DM', 'DMNC', 'GE', 'LU', 'ADJ', 'CH', 'LDL', 'QR')

    iszerofunc : function, optional
        用于零元素测试的函数。

    try_block_diag : bool, optional
        如果为 True，则尝试形成块对角矩阵，使用 get_diag_blocks() 方法分别求逆，
        然后重建完整的逆矩阵。

    Examples
    ========

    >>> from sympy import SparseMatrix, Matrix
    >>> A = SparseMatrix([
    ... [ 2, -1,  0],
    ... [-1,  2, -1],
    ... [ 0,  0,  2]])
    >>> A.inv('CH')
    Matrix([
    [2/3, 1/3, 1/6],
    [1/3, 2/3, 1/3],
    [  0,   0, 1/2]])
    >>> A.inv(method='LDL') # 'method=' 是可选的
    Matrix([
    [2/3, 1/3, 1/6],
    [1/3, 2/3, 1/3],
    [  0,   0, 1/2]])
    >>> A * _
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> A = Matrix(A)
    >>> A.inv('CH')
    Matrix([
    [2/3, 1/3, 1/6],
    [1/3, 2/3, 1/3],
    """
    Calculate the inverse of a matrix `M` using various methods based on the `method` keyword.

    Parameters
    ==========
    M : Matrix
        The matrix to be inverted.
    method : str, optional
        Method to use for inversion. Default is determined by matrix type.
    iszerofunc : callable, optional
        Function to determine if a value is zero.

    Returns
    =======
    Matrix
        The inverted matrix.

    Raises
    ======
    NonSquareMatrixError
        If `M` is not a square matrix.
    ValueError
        If the determinant of `M` is zero or the inversion method is unrecognized.

    Notes
    =====
    According to the `method` keyword, it calls the appropriate inversion method:

        DM .... Use DomainMatrix `inv_den` method
        DMNC .... Use DomainMatrix `inv_den` method without cancellation
        GE .... inverse_GE(); default for dense matrices
        LU .... inverse_LU()
        ADJ ... inverse_ADJ()
        CH ... inverse_CH()
        LDL ... inverse_LDL(); default for sparse matrices
        QR ... inverse_QR()

    Note, the GE and LU methods may require the matrix to be simplified
    before it is inverted in order to properly detect zeros during
    pivoting. In difficult cases a custom zero detection function can
    be provided by setting the `iszerofunc` argument to a function that
    should return True if its argument is zero. The ADJ routine computes
    the determinant and uses that to detect singular matrices in addition
    to testing for zeros on the diagonal.

    See Also
    ========
    inverse_ADJ
    inverse_GE
    inverse_LU
    inverse_CH
    inverse_LDL
    """

    from sympy.matrices import diag, SparseMatrix

    # Check if the matrix is square
    if not M.is_square:
        raise NonSquareMatrixError("A Matrix must be square to invert.")

    # Attempt to use block diagonal inversion if requested
    if try_block_diag:
        blocks = M.get_diag_blocks()
        r = []

        # Invert each block and construct a diagonal matrix
        for block in blocks:
            r.append(block.inv(method=method, iszerofunc=iszerofunc))

        return diag(*r)

    # Determine the inversion method based on the given parameters
    if method is None and iszerofunc is _iszero:
        dM = _try_DM(M, use_EX=False)
        if dM is not None:
            method = 'DM'
    elif method in ("DM", "DMNC"):
        dM = _try_DM(M, use_EX=True)

    # If no method is explicitly provided, choose default methods based on matrix type
    if method is None:
        if isinstance(M, SparseMatrix):
            method = 'LDL'
        else:
            method = 'GE'

    # Perform matrix inversion based on the selected method
    if method == "DM":
        rv = _inv_DM(dM)
    elif method == "DMNC":
        rv = _inv_DM(dM, cancel=False)
    elif method == "GE":
        rv = M.inverse_GE(iszerofunc=iszerofunc)
    elif method == "LU":
        rv = M.inverse_LU(iszerofunc=iszerofunc)
    elif method == "ADJ":
        rv = M.inverse_ADJ(iszerofunc=iszerofunc)
    elif method == "CH":
        rv = M.inverse_CH(iszerofunc=iszerofunc)
    elif method == "LDL":
        rv = M.inverse_LDL(iszerofunc=iszerofunc)
    elif method == "QR":
        rv = M.inverse_QR(iszerofunc=iszerofunc)
    elif method == "BLOCK":
        rv = M.inverse_BLOCK(iszerofunc=iszerofunc)
    else:
        raise ValueError("Inversion method unrecognized")

    # Return the inverted matrix
    return M._new(rv)
```