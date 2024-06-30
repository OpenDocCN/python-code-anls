# `D:\src\scipysrc\sympy\sympy\matrices\decompositions.py`

```
import copy  # 导入 copy 模块，用于复制对象

from sympy.core import S  # 导入 sympy 的 S 对象
from sympy.core.function import expand_mul  # 导入 sympy 的 expand_mul 函数
from sympy.functions.elementary.miscellaneous import Min, sqrt  # 导入 sympy 的 Min 和 sqrt 函数
from sympy.functions.elementary.complexes import sign  # 导入 sympy 的 sign 函数

from .exceptions import NonSquareMatrixError, NonPositiveDefiniteMatrixError  # 从当前包导入自定义异常类
from .utilities import _get_intermediate_simp, _iszero  # 从当前包导入辅助函数
from .determinant import _find_reasonable_pivot_naive  # 从当前包导入特定功能模块的函数


def _rank_decomposition(M, iszerofunc=_iszero, simplify=False):
    r"""Returns a pair of matrices (`C`, `F`) with matching rank
    such that `A = C F`.

    Parameters
    ==========

    iszerofunc : Function, optional
        A function used for detecting whether an element can
        act as a pivot.  ``lambda x: x.is_zero`` is used by default.

    simplify : Bool or Function, optional
        A function used to simplify elements when looking for a
        pivot. By default SymPy's ``simplify`` is used.

    Returns
    =======

    (C, F) : Matrices
        `C` and `F` are full-rank matrices with rank as same as `A`,
        whose product gives `A`.

        See Notes for additional mathematical details.

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([
    ...     [1, 3, 1, 4],
    ...     [2, 7, 3, 9],
    ...     [1, 5, 3, 1],
    ...     [1, 2, 0, 8]
    ... ])
    >>> C, F = A.rank_decomposition()
    >>> C
    Matrix([
    [1, 3, 4],
    [2, 7, 9],
    [1, 5, 1],
    [1, 2, 8]])
    >>> F
    Matrix([
    [1, 0, -2, 0],
    [0, 1,  1, 0],
    [0, 0,  0, 1]])
    >>> C * F == A
    True

    Notes
    =====

    Obtaining `F`, an RREF of `A`, is equivalent to creating a
    product

    .. math::
        E_n E_{n-1} ... E_1 A = F

    where `E_n, E_{n-1}, \dots, E_1` are the elimination matrices or
    permutation matrices equivalent to each row-reduction step.

    The inverse of the same product of elimination matrices gives
    `C`:

    .. math::
        C = \left(E_n E_{n-1} \dots E_1\right)^{-1}

    It is not necessary, however, to actually compute the inverse:
    the columns of `C` are those from the original matrix with the
    same column indices as the indices of the pivot columns of `F`.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Rank_factorization

    .. [2] Piziak, R.; Odell, P. L. (1 June 1999).
        "Full Rank Factorization of Matrices".
        Mathematics Magazine. 72 (3): 193. doi:10.2307/2690882

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.rref
    """

    F, pivot_cols = M.rref(simplify=simplify, iszerofunc=iszerofunc,
            pivots=True)  # 对输入矩阵 M 进行行最简形式 (RREF) 分解，返回 RREF 和主元列索引列表
    rank = len(pivot_cols)  # 计算主元列的数量，即矩阵的秩

    C = M.extract(range(M.rows), pivot_cols)  # 从原始矩阵 M 中提取与主元列相对应的列，构成矩阵 C
    F = F[:rank, :]  # 截取 RREF 矩阵 F 的前 rank 行，保留与主元列相对应的部分

    return C, F


def _liupc(M):
    """Liu's algorithm, for pre-determination of the Elimination Tree of
    the given matrix, used in row-based symbolic Cholesky factorization.

    Examples
    ========

    >>> from sympy import SparseMatrix
    >>> S = SparseMatrix([
    ... [1, 0, 3, 2],
    # 定义一个稀疏矩阵的索引结构，描述其非零元素的位置关系
    ... [0, 0, 1, 0],
    ... [4, 0, 0, 5],
    ... [0, 6, 7, 0]])

    # 调用对象 S 的 liupc 方法，计算稀疏矩阵 M 的下标和主元素的描述
    >>> S.liupc()

    # 引用文献参考文献 [1]，指向符号稀疏Cholesky分解的消除树，作者 Jeroen Van Grondelle (1999)
    # 参考链接：https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.39.7582
    """
    # 算法 2.4，第17页参考

    # 获取矩阵 M 中在或者在对角线以下的非零元素的索引
    R = [[] for r in range(M.rows)]

    # 遍历稀疏矩阵 M 的非零元素的行列表，将列索引加入到 R 中对应的行中
    for r, c, _ in M.row_list():
        if c <= r:
            R[r].append(c)

    # inf 被赋值为 R 的长度，假设不会有索引超过这个长度
    inf     = len(R)  # nothing will be this large
    # parent 和 virtual 初始化为 M 的行数长度，用于构建消除树
    parent  = [inf]*M.rows
    virtual = [inf]*M.rows

    # 遍历稀疏矩阵 M 的每一行 r
    for r in range(M.rows):
        # 遍历行 r 中的每一列 c，除了最后一个元素
        for c in R[r][:-1]:
            # 当 virtual[c] 小于 r 时，交换 virtual[c] 和 r，直到 virtual[c] >= r
            while virtual[c] < r:
                t          = virtual[c]
                virtual[c] = r
                c          = t

            # 如果 virtual[c] 为 inf，将 parent[c] 和 virtual[c] 设置为 r
            if virtual[c] == inf:
                parent[c] = virtual[c] = r

    # 返回稀疏矩阵 M 的索引结构 R 和消除树的父节点列表 parent
    return R, parent
# 执行符号化的 Cholesky 分解，用于预测 Cholesky 分解的非零结构
def _row_structure_symbolic_cholesky(M):
    # 调用 SparseMatrix 的 liupc 方法，返回 R 和 parent 数组
    R, parent = M.liupc()
    # 设置 inf 作为 R 的长度，用作无穷大的标记
    inf       = len(R)  # this acts as infinity
    # 复制 R 到 Lrow，深拷贝确保不改变原始数据
    Lrow      = copy.deepcopy(R)

    # 对每行 k 进行迭代
    for k in range(M.rows):
        # 对于每个在 R[k] 中的 j
        for j in R[k]:
            # 当 j 不是 inf 且不等于 k 时，执行以下循环
            while j != inf and j != k:
                # 将 j 添加到 Lrow[k] 中，并更新 j 为其父节点 parent[j]
                Lrow[k].append(j)
                j = parent[j]

        # 对 Lrow[k] 中的元素进行排序，并去除重复项，确保每行中只有唯一的索引
        Lrow[k] = sorted(set(Lrow[k]))

    # 返回最终的 Lrow，即每行对应的非零元素索引列表
    return Lrow


# 返回矩阵 A 的 Cholesky 分解 L，满足 L * L.H == A（如果 hermitian 为 True）
# 或 L * L.T == A（如果 hermitian 为 False）
def _cholesky(M, hermitian=True):
    # 导入 MutableDenseMatrix 类
    from .dense import MutableDenseMatrix

    # 如果 M 不是方阵，则抛出异常
    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    # 如果 hermitian 为 True 且 M 不是 Hermitian 矩阵，则抛出异常
    if hermitian and not M.is_hermitian:
        raise ValueError("Matrix must be Hermitian.")
    # 如果 hermitian 为 False 且 M 不是对称矩阵，则抛出异常
    if not hermitian and not M.is_symmetric():
        raise ValueError("Matrix must be symmetric.")

    # 创建一个 M.rows × M.rows 的零矩阵 L
    L   = MutableDenseMatrix.zeros(M.rows, M.rows)
    # 如果 hermitian 为真，则执行下面的代码块
    if hermitian:
        # 遍历矩阵 M 的行
        for i in range(M.rows):
            # 遍历当前行 i 之前的列 j
            for j in range(i):
                # 计算 Cholesky 分解中的 L[i, j] 元素
                L[i, j] = ((1 / L[j, j]) * (M[i, j] -
                    sum(L[i, k] * L[j, k].conjugate() for k in range(j))))

            # 计算对角线元素 L[i, i] 的平方和
            Lii2 = (M[i, i] -
                sum(L[i, k] * L[i, k].conjugate() for k in range(i)))

            # 检查 Lii2 是否为正数，如果不是则抛出异常
            if Lii2.is_positive is False:
                raise NonPositiveDefiniteMatrixError(
                    "Matrix must be positive-definite")

            # 计算并赋值对角线元素 L[i, i]
            L[i, i] = sqrt(Lii2)

    # 如果 hermitian 为假，则执行下面的代码块
    else:
        # 遍历矩阵 M 的行
        for i in range(M.rows):
            # 遍历当前行 i 之前的列 j
            for j in range(i):
                # 计算 Cholesky 分解中的 L[i, j] 元素
                L[i, j] = ((1 / L[j, j]) * (M[i, j] -
                    sum(L[i, k] * L[j, k] for k in range(j))))

            # 计算并赋值对角线元素 L[i, i]
            L[i, i] = sqrt(M[i, i] -
                sum(L[i, k] ** 2 for k in range(i)))

    # 返回 Cholesky 分解后的矩阵 M
    return M._new(L)
    # 导入密集矩阵的MutableDenseMatrix类
    from .dense import MutableDenseMatrix

    # 检查矩阵M是否为方阵，若不是则引发异常
    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    # 如果要求矩阵为Hermitian，并且M不满足Hermitian性质，则引发异常
    if hermitian and not M.is_hermitian:
        raise ValueError("Matrix must be Hermitian.")
    # 如果不要求矩阵为Hermitian，并且M不满足对称性质，则引发异常
    if not hermitian and not M.is_symmetric():
        raise ValueError("Matrix must be symmetric.")

    # 获取中间表达式简化函数
    dps = _get_intermediate_simp(expand_mul, expand_mul)
    # 使用符号Cholesky方法获取M的行结构
    Crowstruc = M.row_structure_symbolic_cholesky()
    # 创建全零的MutableDenseMatrix对象C，形状与M相同
    C = MutableDenseMatrix.zeros(M.rows)
    # 遍历 Crowstruc 列表的索引范围
    for i in range(len(Crowstruc)):
        # 遍历 Crowstruc[i] 中的每个元素 j
        for j in Crowstruc[i]:
            # 如果 i 不等于 j，则执行以下操作
            if i != j:
                # 将 M[i, j] 赋值给 C[i, j]
                C[i, j] = M[i, j]
                # 初始化 summ 为 0
                summ = 0

                # 遍历 Crowstruc[i] 中小于 j 的每个元素 p1
                for p1 in Crowstruc[i]:
                    if p1 < j:
                        # 遍历 Crowstruc[j] 中小于 j 的每个元素 p2
                        for p2 in Crowstruc[j]:
                            if p2 < j:
                                # 如果 p1 等于 p2，则执行以下操作
                                if p1 == p2:
                                    # 如果 hermitian 为真，则将 C[i, p1]*C[j, p1].conjugate() 加到 summ
                                    if hermitian:
                                        summ += C[i, p1] * C[j, p1].conjugate()
                                    else:
                                        summ += C[i, p1] * C[j, p1]
                            else:
                                break
                        else:
                            break

                # 计算更新后的 C[i, j] 的值
                C[i, j] = dps((C[i, j] - summ) / C[j, j])

            else:  # 如果 i 等于 j
                # 将 M[j, j] 赋值给 C[j, j]
                C[j, j] = M[j, j]
                # 初始化 summ 为 0
                summ = 0

                # 遍历 Crowstruc[j] 中小于 j 的每个元素 k
                for k in Crowstruc[j]:
                    if k < j:
                        # 如果 hermitian 为真，则将 C[j, k]*C[j, k].conjugate() 加到 summ
                        if hermitian:
                            summ += C[j, k] * C[j, k].conjugate()
                        else:
                            summ += C[j, k] ** 2
                    else:
                        break

                # 计算更新后的 C[j, j] 的值
                Cjj2 = dps(C[j, j] - summ)

                # 如果 hermitian 为真且 Cjj2 不是正数，则抛出异常
                if hermitian and Cjj2.is_positive is False:
                    raise NonPositiveDefiniteMatrixError(
                        "Matrix must be positive-definite")

                # 将 C[j, j] 更新为 Cjj2 的平方根
                C[j, j] = sqrt(Cjj2)

    # 返回更新后的矩阵 M 的副本
    return M._new(C)
def _LDLdecomposition_sparse(M, hermitian=True):
    """
    Returns the LDL Decomposition (matrices ``L`` and ``D``) of matrix
    ``A``, such that ``L * D * L.T == A``. ``A`` must be a square,
    symmetric, positive-definite and non-singular.

    This method eliminates the use of square root and ensures that all
    the diagonal entries of L are 1.

    Examples
    ========

    >>> from sympy import SparseMatrix
    >>> A = SparseMatrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    >>> L, D = A.LDLdecomposition()
    >>> L
    Matrix([



    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")
    if hermitian and not M.is_hermitian:
        raise ValueError("Matrix must be Hermitian.")
    if not hermitian and not M.is_symmetric():
        raise ValueError("Matrix must be symmetric.")

    # Initialize matrices L and D
    D = MutableDenseMatrix.zeros(M.rows, M.rows)
    L = MutableDenseMatrix.eye(M.rows)

    if hermitian:
        # Compute L and D for a Hermitian matrix
        for i in range(M.rows):
            for j in range(i):
                # Compute entries of L
                L[i, j] = (1 / D[j, j]) * (M[i, j] - sum(
                    L[i, k] * L[j, k].conjugate() * D[k, k] for k in range(j)))

            # Compute diagonal entries of D
            D[i, i] = (M[i, i] - sum(
                L[i, k] * L[i, k].conjugate() * D[k, k] for k in range(i)))

            # Check for positive-definiteness
            if D[i, i].is_positive is False:
                raise NonPositiveDefiniteMatrixError(
                    "Matrix must be positive-definite")

    else:
        # Compute L and D for a symmetric matrix
        for i in range(M.rows):
            for j in range(i):
                # Compute entries of L
                L[i, j] = (1 / D[j, j]) * (M[i, j] - sum(
                    L[i, k] * L[j, k] * D[k, k] for k in range(j)))

            # Compute diagonal entries of D
            D[i, i] = M[i, i] - sum(L[i, k]**2 * D[k, k] for k in range(i))

    # Return matrices L and D
    return M._new(L), M._new(D)



    >>> D
    Matrix([



    [25,  0,  0],
    [ 0,  9,  0],
    [ 0,  0,  9]])
    >>> L*D*L.T == A
    True

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.cholesky
    sympy.matrices.matrixbase.MatrixBase.LUdecomposition
    QRdecomposition
    """
    [   1,   0, 0],  # 创建一个包含整数的列表，表示矩阵的第一行
    [ 3/5,   1, 0],  # 继续创建矩阵的第二行，包含分数
    [-1/5, 1/3, 1]])  # 完成矩阵的创建，包含负分数和分数

    >>> D  # 打印矩阵 D 的内容
    Matrix([  # 创建一个 Matrix 对象，展示以下行
    [25, 0, 0],  # 第一行包含整数
    [ 0, 9, 0],  # 第二行包含整数
    [ 0, 0, 9]])  # 第三行包含整数
    >>> L * D * L.T == A  # 检查等式 L * D * L 转置是否等于 A
    True  # 输出结果为 True，表示等式成立

    """

    from .dense import MutableDenseMatrix  # 导入 MutableDenseMatrix 类

    if not M.is_square:  # 如果矩阵 M 不是方阵，抛出异常
        raise NonSquareMatrixError("Matrix must be square.")
    if hermitian and not M.is_hermitian:  # 如果要求是 Hermitian 矩阵但 M 不是，抛出异常
        raise ValueError("Matrix must be Hermitian.")
    if not hermitian and not M.is_symmetric():  # 如果不要求是 Hermitian 但 M 不对称，抛出异常
        raise ValueError("Matrix must be symmetric.")

    dps       = _get_intermediate_simp(expand_mul, expand_mul)  # 调用 _get_intermediate_simp 函数，存储在 dps 变量中
    Lrowstruc = M.row_structure_symbolic_cholesky()  # 调用 M 的 row_structure_symbolic_cholesky 方法，返回 L 的行结构
    L         = MutableDenseMatrix.eye(M.rows)  # 创建一个 M.rows x M.rows 的单位矩阵 L
    D         = MutableDenseMatrix.zeros(M.rows, M.cols)  # 创建一个 M.rows x M.cols 的零矩阵 D

    for i in range(len(Lrowstruc)):  # 循环遍历 Lrowstruc 的长度
        for j in Lrowstruc[i]:  # 遍历 Lrowstruc[i] 中的每个元素 j
            if i != j:  # 如果 i 不等于 j
                L[i, j] = M[i, j]  # 设置 L[i, j] 等于 M[i, j]
                summ    = 0  # 初始化 summ 变量为 0

                for p1 in Lrowstruc[i]:  # 遍历 Lrowstruc[i] 中的每个元素 p1
                    if p1 < j:  # 如果 p1 小于 j
                        for p2 in Lrowstruc[j]:  # 遍历 Lrowstruc[j] 中的每个元素 p2
                            if p2 < j:  # 如果 p2 小于 j
                                if p1 == p2:  # 如果 p1 等于 p2
                                    if hermitian:  # 如果要求是 Hermitian 矩阵
                                        summ += L[i, p1]*L[j, p1].conjugate()*D[p1, p1]  # 更新 summ
                                    else:
                                        summ += L[i, p1]*L[j, p1]*D[p1, p1]  # 更新 summ
                            else:
                                break  # 结束内部循环
                    else:
                        break  # 结束外部循环

                L[i, j] = dps((L[i, j] - summ) / D[j, j])  # 设置 L[i, j] 使用 dps 进行更新

            else:  # 如果 i 等于 j
                D[i, i] = M[i, i]  # 设置 D[i, i] 等于 M[i, i]
                summ    = 0  # 初始化 summ 变量为 0

                for k in Lrowstruc[i]:  # 遍历 Lrowstruc[i] 中的每个元素 k
                    if k < i:  # 如果 k 小于 i
                        if hermitian:  # 如果要求是 Hermitian 矩阵
                            summ += L[i, k]*L[i, k].conjugate()*D[k, k]  # 更新 summ
                        else:
                            summ += L[i, k]**2*D[k, k]  # 更新 summ
                    else:
                        break  # 结束循环

                D[i, i] = dps(D[i, i] - summ)  # 设置 D[i, i] 使用 dps 进行更新

                if hermitian and D[i, i].is_positive is False:  # 如果要求是 Hermitian 矩阵且 D[i, i] 不是正数，抛出异常
                    raise NonPositiveDefiniteMatrixError(
                        "Matrix must be positive-definite")

    return M._new(L), M._new(D)  # 返回一个包含 L 和 D 的新 M 对象
# 使用LU分解将矩阵M分解为下三角矩阵L、上三角矩阵U和行置换索引对列表perm。
def _LUdecomposition(M, iszerofunc=_iszero, simpfunc=None, rankcheck=False):
    """Returns (L, U, perm) where L is a lower triangular matrix with unit
    diagonal, U is an upper triangular matrix, and perm is a list of row
    swap index pairs. If A is the original matrix, then
    ``A = (L*U).permuteBkwd(perm)``, and the row permutation matrix P such
    that $P A = L U$ can be computed by ``P = eye(A.rows).permuteFwd(perm)``.

    See documentation for LUCombined for details about the keyword argument
    rankcheck, iszerofunc, and simpfunc.

    Parameters
    ==========

    rankcheck : bool, optional
        Determines if this function should detect the rank
        deficiency of the matrixis and should raise a
        ``ValueError``.

    iszerofunc : function, optional
        A function which determines if a given expression is zero.

        The function should be a callable that takes a single
        SymPy expression and returns a 3-valued boolean value
        ``True``, ``False``, or ``None``.

        It is internally used by the pivot searching algorithm.
        See the notes section for a more information about the
        pivot searching algorithm.

    simpfunc : function or None, optional
        A function that simplifies the input.

        If this is specified as a function, this function should be
        a callable that takes a single SymPy expression and returns
        an another SymPy expression that is algebraically
        equivalent.

        If ``None``, it indicates that the pivot search algorithm
        should not attempt to simplify any candidate pivots.

        It is internally used by the pivot searching algorithm.
        See the notes section for a more information about the
        pivot searching algorithm.

    Examples
    ========

    >>> from sympy import Matrix
    >>> a = Matrix([[4, 3], [6, 3]])
    >>> L, U, _ = a.LUdecomposition()
    >>> L
    Matrix([
    [  1, 0],
    [3/2, 1]])
    >>> U
    Matrix([
    [4,    3],
    [0, -3/2]])

    See Also
    ========

    sympy.matrices.dense.DenseMatrix.cholesky
    sympy.matrices.dense.DenseMatrix.LDLdecomposition
    QRdecomposition
    LUdecomposition_Simple
    LUdecompositionFF
    LUsolve
    """

    # 使用LUdecomposition_Simple方法进行LU分解，返回合并后的矩阵combined和行置换信息p。
    combined, p = M.LUdecomposition_Simple(iszerofunc=iszerofunc,
        simpfunc=simpfunc, rankcheck=rankcheck)

    # L是大小为M.rows x M.rows的下三角矩阵
    # U是大小为M.rows x M.cols的上三角矩阵
    # L具有单位对角线。对于combined的每一列，其下对角线部分由L共享。
    # 如果L的列数大于combined，则L剩余的下对角线部分为零。
    # L和combined的上三角部分是相等的。
    # 定义函数 entry_L(i, j)，用于计算矩阵 L 的元素值
    def entry_L(i, j):
        # 如果 i 小于 j，返回矩阵 M 的零元素
        if i < j:
            return M.zero
        # 如果 i 等于 j，返回矩阵 M 的单位元素
        elif i == j:
            return M.one
        # 如果 j 小于 combined 的列数，则返回 combined[i, j] 的值
        elif j < combined.cols:
            return combined[i, j]
    
        # 否则返回矩阵 M 的零元素，表示在 L 中存在没有对应项的次对角线元素
        # 在 combined 中没有对应的条目
        return M.zero
    
    # 定义函数 entry_U(i, j)，用于计算矩阵 U 的元素值
    def entry_U(i, j):
        # 如果 i 大于 j，返回矩阵 M 的零元素；否则返回 combined[i, j] 的值
        return M.zero if i > j else combined[i, j]
    
    # 创建矩阵 L，其大小为 combined 的行数乘以行数，元素值由 entry_L 函数确定
    L = M._new(combined.rows, combined.rows, entry_L)
    # 创建矩阵 U，其大小为 combined 的行数乘以列数，元素值由 entry_U 函数确定
    U = M._new(combined.rows, combined.cols, entry_U)
    
    # 返回计算得到的矩阵 L、U，以及变换矩阵 p
    return L, U, p
def _LUdecomposition_Simple(M, iszerofunc=_iszero, simpfunc=None,
        rankcheck=False):
    r"""Compute the PLU decomposition of the matrix.

    Parameters
    ==========

    rankcheck : bool, optional
        Determines if this function should detect the rank
        deficiency of the matrix and should raise a
        ``ValueError``.

    iszerofunc : function, optional
        A function which determines if a given expression is zero.

        The function should be a callable that takes a single
        SymPy expression and returns a 3-valued boolean value
        ``True``, ``False``, or ``None``.

        It is internally used by the pivot searching algorithm.
        See the notes section for more information about the
        pivot searching algorithm.

    simpfunc : function or None, optional
        A function that simplifies the input.

        If specified as a function, this function should be
        a callable that takes a single SymPy expression and returns
        another SymPy expression that is algebraically
        equivalent.

        If ``None``, it indicates that the pivot search algorithm
        should not attempt to simplify any candidate pivots.

        It is internally used by the pivot searching algorithm.
        See the notes section for more information about the
        pivot searching algorithm.

    Returns
    =======

    (lu, row_swaps) : (Matrix, list)
        If the original matrix is a $m, n$ matrix:

        *lu* is a $m, n$ matrix, which contains the result of the
        decomposition in a compressed form. See the notes section
        to see how the matrix is compressed.

        *row_swaps* is a $m$-element list where each element is a
        pair of row exchange indices.

        ``A = (L*U).permute_backward(perm)``, and the row
        permutation matrix $P$ from the formula $P A = L U$ can be
        computed by ``P=eye(A.row).permute_forward(perm)``.

    Raises
    ======

    ValueError
        Raised if ``rankcheck=True`` and the matrix is found to
        be rank deficient during the computation.

    Notes
    =====

    About the PLU decomposition:

    PLU decomposition is a generalization of an LU decomposition
    which can be extended for rank-deficient matrices.

    It can further be generalized for non-square matrices, and this
    is the notation that SymPy is using.

    PLU decomposition is a decomposition of a $m, n$ matrix $A$ in
    the form of $P A = L U$ where

    * $L$ is a $m, m$ lower triangular matrix with unit diagonal
        entries.
    * $U$ is a $m, n$ upper triangular matrix.
    * $P$ is a $m, m$ permutation matrix.

    So, for a square matrix, the decomposition would look like:
    # 这部分是文档中的数学公式示例，展示了LU分解的不同情况。
    # 对于行数大于列数的矩阵，LU分解的结果如下所示：
    # L 是单位下三角矩阵，U 是上三角矩阵。L 的右上方和 U 的下方是零。
    .. math::
        L = \begin{bmatrix}
        1 & 0 & 0 & \cdots & 0 \\
        L_{1, 0} & 1 & 0 & \cdots & 0 \\
        L_{2, 0} & L_{2, 1} & 1 & \cdots & 0 \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        L_{n-1, 0} & L_{n-1, 1} & L_{n-1, 2} & \cdots & 1
        \end{bmatrix}
    
    # U 是上三角矩阵，其左下方是零。
    .. math::
        U = \begin{bmatrix}
        U_{0, 0} & U_{0, 1} & U_{0, 2} & \cdots & U_{0, n-1} \\
        0 & U_{1, 1} & U_{1, 2} & \cdots & U_{1, n-1} \\
        0 & 0 & U_{2, 2} & \cdots & U_{2, n-1} \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & 0 & \cdots & U_{n-1, n-1}
        \end{bmatrix}
    
    # 对于行数小于列数的矩阵，LU分解的结果如下所示：
    # L 是单位下三角矩阵，U 是上三角矩阵。L 的右上方和 U 的下方是零。
    .. math::
        L = \begin{bmatrix}
        1 & 0 & 0 & \cdots & 0 \\
        L_{1, 0} & 1 & 0 & \cdots & 0 \\
        L_{2, 0} & L_{2, 1} & 1 & \cdots & 0 \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        L_{m-1, 0} & L_{m-1, 1} & L_{m-1, 2} & \cdots & 1
        \end{bmatrix}
    
    # U 是上三角矩阵，其左下方是零。
    .. math::
        U = \begin{bmatrix}
        U_{0, 0} & U_{0, 1} & U_{0, 2} & \cdots & U_{0, m-1} & \cdots & U_{0, n-1} \\
        0 & U_{1, 1} & U_{1, 2} & \cdots & U_{1, m-1} & \cdots & U_{1, n-1} \\
        0 & 0 & U_{2, 2} & \cdots & U_{2, m-1} & \cdots & U_{2, n-1} \\
        \vdots & \vdots & \vdots & \ddots & \vdots & \cdots & \vdots \\
        0 & 0 & 0 & \cdots & U_{m-1, m-1} & \cdots & U_{m-1, n-1} \\
        \end{bmatrix}
    
    # 关于压缩的LU存储方式的说明：
    # LU分解的结果通常以压缩形式存储，而不是分别返回 L 和 U 矩阵。
    # 这种方式虽然不太直观，但在实际应用中非常常见。
    # 在很多情况下，存储方式更加高效。
    About the compressed LU storage:
    
    # The results of the decomposition are often stored in compressed
    # forms rather than returning $L$ and $U$ matrices individually.
    
    # It may be less intiuitive, but it is commonly used for a lot of
    # 作为说明，不是具体的代码实现，因此不需要额外的注释。
    numeric libraries because of the efficiency.



    The storage matrix is defined as following for this specific
    method:



    * The subdiagonal elements of $L$ are stored in the subdiagonal
        portion of $LU$, that is $LU_{i, j} = L_{i, j}$ whenever
        $i > j$.



    * The elements on the diagonal of $L$ are all 1, and are not
        explicitly stored.



    * $U$ is stored in the upper triangular portion of $LU$, that is
        $LU_{i, j} = U_{i, j}$ whenever $i <= j$.



    * For a case of $m > n$, the right side of the $L$ matrix is
        trivial to store.



    * For a case of $m < n$, the below side of the $U$ matrix is
        trivial to store.



    So, for a square matrix, the compressed output matrix would be:

    .. math::
        LU = \begin{bmatrix}
        U_{0, 0} & U_{0, 1} & U_{0, 2} & \cdots & U_{0, n-1} \\
        L_{1, 0} & U_{1, 1} & U_{1, 2} & \cdots & U_{1, n-1} \\
        L_{2, 0} & L_{2, 1} & U_{2, 2} & \cdots & U_{2, n-1} \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        L_{n-1, 0} & L_{n-1, 1} & L_{n-1, 2} & \cdots & U_{n-1, n-1}
        \end{bmatrix}



    For a matrix with more rows than the columns, the compressed
    output matrix would be:

    .. math::
        LU = \begin{bmatrix}
        U_{0, 0} & U_{0, 1} & U_{0, 2} & \cdots & U_{0, n-1} \\
        L_{1, 0} & U_{1, 1} & U_{1, 2} & \cdots & U_{1, n-1} \\
        L_{2, 0} & L_{2, 1} & U_{2, 2} & \cdots & U_{2, n-1} \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        L_{n-1, 0} & L_{n-1, 1} & L_{n-1, 2} & \cdots
        & U_{n-1, n-1} \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        L_{m-1, 0} & L_{m-1, 1} & L_{m-1, 2} & \cdots
        & L_{m-1, n-1} \\
        \end{bmatrix}



    For a matrix with more columns than the rows, the compressed
    output matrix would be:

    .. math::
        LU = \begin{bmatrix}
        U_{0, 0} & U_{0, 1} & U_{0, 2} & \cdots & U_{0, m-1}
        & \cdots & U_{0, n-1} \\
        L_{1, 0} & U_{1, 1} & U_{1, 2} & \cdots & U_{1, m-1}
        & \cdots & U_{1, n-1} \\
        L_{2, 0} & L_{2, 1} & U_{2, 2} & \cdots & U_{2, m-1}
        & \cdots & U_{2, n-1} \\
        \vdots & \vdots & \vdots & \ddots & \vdots
        & \cdots & \vdots \\
        L_{m-1, 0} & L_{m-1, 1} & L_{m-1, 2} & \cdots & U_{m-1, m-1}
        & \cdots & U_{m-1, n-1} \\
        \end{bmatrix}



    About the pivot searching algorithm:

    When a matrix contains symbolic entries, the pivot search algorithm
    differs from the case where every entry can be categorized as zero or
    nonzero.



    The algorithm searches column by column through the submatrix whose
    top left entry coincides with the pivot position.



    If it exists, the pivot is the first entry in the current search
    column that iszerofunc guarantees is nonzero.



    If no such candidate exists, then each candidate pivot is simplified
    if simpfunc is not None.



    The search is repeated, with the difference that a candidate may be
    """
    Perform LU decomposition of a matrix M.

    This function computes the LU decomposition of the input matrix M.
    It optionally performs rank checking and handles special cases where
    the matrix has zero entries or is smaller than expected.

    Parameters
    ==========
    M : Matrix
        Input matrix to be decomposed.

    rankcheck : bool, optional
        Flag indicating whether to perform rank checking during decomposition.

    Returns
    =======
    lu : Matrix
        Lower and upper triangular factors of the input matrix M.

    row_swaps : list
        List of row swaps performed during decomposition.

    Notes
    =====
    If rankcheck is True, the function verifies if the matrix rank is
    sufficient for LU decomposition. If the matrix contains zero entries,
    it returns a matrix of zeros with the same dimensions. The algorithm
    uses an intermediate function to simplify numerical operations (dps).
    The pivot column for decomposition starts from index 0.

    See Also
    ========
    sympy.matrices.matrixbase.MatrixBase.LUdecomposition
    LUdecompositionFF
    LUsolve
    """

    if rankcheck:
        # Check if the diagonal element of the bottom-right submatrix is zero
        if iszerofunc(
                lu[Min(lu.rows, lu.cols) - 1, Min(lu.rows, lu.cols) - 1]):
            # Raise an error if the matrix rank is insufficient
            raise ValueError("Rank of matrix is strictly less than"
                                " number of rows or columns."
                                " Pass keyword argument"
                                " rankcheck=False to compute"
                                " the LU decomposition of this matrix.")

    if S.Zero in M.shape:
        # Handle case where matrix has no entries by returning a matrix of zeros
        return M.zeros(M.rows, M.cols), []

    # Get intermediate simplification precision
    dps = _get_intermediate_simp()
    # Make a mutable copy of input matrix M
    lu = M.as_mutable()
    # Initialize list to track row swaps
    row_swaps = []

    # Start pivot column index
    pivot_col = 0

    # Return decomposed matrix and row swap information
    return lu, row_swaps
def _LUdecompositionFF(M):
    """Compute a fraction-free LU decomposition.

    Returns 4 matrices P, L, D, U such that PA = L D**-1 U.
    If the elements of the matrix belong to some integral domain I, then all
    elements of L, D and U are guaranteed to belong to I.

    See Also
    ========

    sympy.matrices.matrixbase.MatrixBase.LUdecomposition
    LUdecomposition_Simple
    LUsolve

    References
    ==========

    .. [1] W. Zhou & D.J. Jeffrey, "Fraction-free matrix factors: new forms
        for LU and QR factors". Frontiers in Computer Science in China,
        Vol 2, no. 1, pp. 67-80, 2008.
    """

    from sympy.matrices import SparseMatrix  # 导入 SparseMatrix 类

    zeros    = SparseMatrix.zeros  # 创建一个用于生成零矩阵的函数
    eye      = SparseMatrix.eye    # 创建一个单位矩阵的函数
    n, m     = M.rows, M.cols      # 获取矩阵 M 的行数和列数
    U, L, P  = M.as_mutable(), eye(n), eye(n)  # 初始化 U, L, P 矩阵
    DD       = zeros(n, n)         # 创建一个 n x n 的零矩阵 DD
    oldpivot = 1                   # 初始化旧主元为 1

    for k in range(n - 1):
        if U[k, k] == 0:  # 如果当前主元为 0
            for kpivot in range(k + 1, n):  # 寻找一个非零主元进行行交换
                if U[kpivot, k]:
                    break
            else:
                raise ValueError("Matrix is not full rank")  # 如果不存在非零主元，则抛出异常

            # 执行行交换操作，更新 L, P, U 矩阵
            U[k, k:], U[kpivot, k:] = U[kpivot, k:], U[k, k:]
            L[k, :k], L[kpivot, :k] = L[kpivot, :k], L[k, :k]
            P[k, :], P[kpivot, :]   = P[kpivot, :], P[k, :]

        L[k, k] = Ukk = U[k, k]  # 更新 L 矩阵的对角元素
        DD[k, k] = oldpivot * Ukk  # 更新 DD 矩阵的对角元素

        for i in range(k + 1, n):
            L[i, k] = Uik = U[i, k]

            for j in range(k + 1, m):
                U[i, j] = (Ukk * U[i, j] - U[k, j] * Uik) / oldpivot

            U[i, k] = 0

        oldpivot = Ukk  # 更新旧主元为当前主元

    DD[n - 1, n - 1] = oldpivot  # 更新 DD 矩阵的最后一个对角元素为最后一个主元

    return P, L, DD, U  # 返回四个矩阵 P, L, DD, U
    # 获取矩阵 A 的共轭转置
    AH = A.H
    # 获取矩阵 A 的行数 m 和列数 n
    m, n = A.shape
    # 如果 m 大于等于 n，则执行以下操作
    if m >= n:
        # 对矩阵 AH * A 进行对角化，得到特征向量矩阵 V 和特征值矩阵 S
        V, S = (AH * A).diagonalize()

        # 初始化一个空列表用于存储非零特征值对应的索引
        ranked = []
        # 遍历 S 的对角线元素
        for i, x in enumerate(S.diagonal()):
            # 如果特征值不为零，则将其索引 i 添加到 ranked 列表中
            if not x.is_zero:
                ranked.append(i)

        # 根据 ranked 列表，从 V 中选择对应的列向量
        V = V[:, ranked]

        # 计算非零特征值的平方根，形成特征值列表
        Singular_vals = [sqrt(S[i, i]) for i in range(S.rows) if i in ranked]

        # 构造新的对角矩阵 S，其对角线元素为 Singular_vals
        S = S.diag(*Singular_vals)

        # 对 V 进行 QR 分解，得到正交矩阵 V 和一个未使用的变量
        V, _ = V.QRdecomposition()

        # 计算 U 矩阵，其为 A * V * S^(-1)
        U = A * V * S.inv()

    # 如果 m 小于 n，则执行以下操作
    else:
        # 对矩阵 A * AH 进行对角化，得到特征向量矩阵 U 和特征值矩阵 S
        U, S = (A * AH).diagonalize()

        # 初始化一个空列表用于存储非零特征值对应的索引
        ranked = []
        # 遍历 S 的对角线元素
        for i, x in enumerate(S.diagonal()):
            # 如果特征值不为零，则将其索引 i 添加到 ranked 列表中
            if not x.is_zero:
                ranked.append(i)

        # 根据 ranked 列表，从 U 中选择对应的列向量
        U = U[:, ranked]

        # 计算非零特征值的平方根，形成特征值列表
        Singular_vals = [sqrt(S[i, i]) for i in range(S.rows) if i in ranked]

        # 构造新的对角矩阵 S，其对角线元素为 Singular_vals
        S = S.diag(*Singular_vals)

        # 对 U 进行 QR 分解，得到正交矩阵 U 和一个未使用的变量
        U, _ = U.QRdecomposition()

        # 计算 V 矩阵，其为 AH * U * S^(-1)
        V = AH * U * S.inv()

    # 返回计算得到的 U, S, V 三个矩阵
    return U, S, V
def _QRdecomposition_optional(M, normalize=True):
    # 定义向量的内积函数
    def dot(u, v):
        return u.dot(v, hermitian=True)

    # 获取中间简化的表达式
    dps = _get_intermediate_simp(expand_mul, expand_mul)

    # 将矩阵 M 转换为可变类型
    A = M.as_mutable()
    # 存储列排名的列表
    ranked = []

    # 初始化 Q 矩阵为 A，R 矩阵为与 A 列数相同的零矩阵
    Q = A
    R = A.zeros(A.cols)

    # 开始进行 QR 分解
    for j in range(A.cols):
        for i in range(j):
            # 如果第 i 列是零矩阵，则跳过
            if Q[:, i].is_zero_matrix:
                continue

            # 计算 R 矩阵的元素 R[i, j]
            R[i, j] = dot(Q[:, i], Q[:, j]) / dot(Q[:, i], Q[:, i])
            # 简化 R[i, j] 的表达式
            R[i, j] = dps(R[i, j])
            # 更新 Q 矩阵的第 j 列
            Q[:, j] -= Q[:, i] * R[i, j]

        # 简化 Q 矩阵的第 j 列
        Q[:, j] = dps(Q[:, j])
        # 如果 Q 矩阵的第 j 列不是零矩阵
        if Q[:, j].is_zero_matrix is not True:
            # 将 j 加入排名列表
            ranked.append(j)
            # 设置 R 矩阵的对角元素 R[j, j] 为 M 的单位元素
            R[j, j] = M.one

    # 提取 Q 矩阵中指定行和列的子矩阵
    Q = Q.extract(range(Q.rows), ranked)
    # 提取 R 矩阵中指定行和列的子矩阵
    R = R.extract(ranked, range(R.cols))

    # 如果需要归一化
    if normalize:
        # 归一化 Q 矩阵的列向量
        for i in range(Q.cols):
            norm = Q[:, i].norm()
            Q[:, i] /= norm
            R[i, :] *= norm

    # 返回 M 类型的 Q 和 R 矩阵
    return M.__class__(Q), M.__class__(R)
    # QR分解可能返回一个矩阵Q，其可能是矩形的。
    # 在这种情况下，正交性条件可能满足为 $\mathbb{I} = Q.H*Q$，但不满足反向乘积 $\mathbb{I} = Q * Q.H$。
    
    >>> Q.H * Q
    Matrix([
    [1, 0],
    [0, 1]])
    >>> Q * Q.H
    Matrix([
    [27261/30625,   348/30625, -1914/6125],
    [  348/30625, 30589/30625,   198/6125],
    [ -1914/6125,    198/6125,   136/1225]])
    
    # 如果要扩展结果为完全正交分解，应该将 $Q$ 增加另一个正交列。
    
    # 可以附加一个单位矩阵，并且可以运行Gram-Schmidt过程使它们成为扩展的正交基。
    
    >>> Q_aug = Q.row_join(Matrix.eye(3))
    >>> Q_aug = Q_aug.QRdecomposition()[0]
    >>> Q_aug
    Matrix([
    [ 6/7, -69/175, 58/175],
    [ 3/7, 158/175, -6/175],
    [-2/7,    6/35,  33/35]])
    >>> Q_aug.H * Q_aug
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> Q_aug * Q_aug.H
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    
    # 将 $R$ 矩阵扩展为带有零行是直接的。
    
    >>> R_aug = R.col_join(Matrix([[0, 0, 0]]))
    >>> R_aug
    Matrix([
    [14,  21, 0],
    [ 0, 175, 0],
    [ 0,   0, 0]])
    >>> Q_aug * R_aug == A
    True
    
    # 一个零矩阵的示例：
    
    >>> from sympy import Matrix
    >>> A = Matrix.zeros(3, 4)
    >>> Q, R = A.QRdecomposition()
    
    # 它们可能返回带有零行和列的矩阵。
    
    >>> Q
    Matrix(3, 0, [])
    >>> R
    Matrix(0, 4, [])
    >>> Q*R
    Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]])
    
    # 如上所述，$Q$ 可以用单位矩阵的列扩展，$R$ 可以用零矩阵的行扩展。
    
    >>> Q_aug = Q.row_join(Matrix.eye(3))
    >>> R_aug = R.col_join(Matrix.zeros(3, 4))
    >>> Q_aug * Q_aug.T
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> R_aug
    Matrix([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]])
    >>> Q_aug * R_aug == A
    True
    
    # 参见
    # ========
    # sympy.matrices.dense.DenseMatrix.cholesky
    # sympy.matrices.dense.DenseMatrix.LDLdecomposition
    # sympy.matrices.matrixbase.MatrixBase.LUdecomposition
    # QRsolve
def _upper_hessenberg_decomposition(A):
    """Converts a matrix into Hessenberg matrix H.

    Returns 2 matrices H, P s.t.
    $P H P^{T} = A$, where H is an upper hessenberg matrix
    and P is an orthogonal matrix

    Examples
    ========

    >>> from sympy import Matrix
    >>> A = Matrix([
    ...     [1,2,3],
    ...     [-3,5,6],
    ...     [4,-8,9],
    ... ])
    >>> H, P = A.upper_hessenberg_decomposition()
    >>> H
    Matrix([
    [1,    6/5,    17/5],
    [5, 213/25, -134/25],
    [0, 216/25,  137/25]])
    >>> P
    Matrix([
    [1,    0,   0],
    [0, -3/5, 4/5],
    [0,  4/5, 3/5]])
    >>> P * H * P.H == A
    True


    References
    ==========

    .. [#] https://mathworld.wolfram.com/HessenbergDecomposition.html
    """

    # Make a mutable copy of the input matrix A
    M = A.as_mutable()

    # Check if the matrix is square, raise an error if not
    if not M.is_square:
        raise NonSquareMatrixError("Matrix must be square.")

    # Get the number of columns (or rows, since it's square)
    n = M.cols

    # Initialize P as the identity matrix of size n
    P = M.eye(n)

    # Initialize H as a copy of M
    H = M

    # Perform Hessenberg decomposition
    for j in range(n - 2):
        # Extract the subvector u from H
        u = H[j + 1:, j]

        # Continue if the submatrix below the first element of u is zero
        if u[1:, :].is_zero_matrix:
            continue

        # Modify the first element of u
        if sign(u[0]) != 0:
            u[0] = u[0] + sign(u[0]) * u.norm()
        else:
            u[0] = u[0] + u.norm()

        # Normalize u to get v
        v = u / u.norm()

        # Update H using Householder transformations
        H[j + 1:, :] = H[j + 1:, :] - 2 * v * (v.H * H[j + 1:, :])
        H[:, j + 1:] = H[:, j + 1:] - (H[:, j + 1:] * (2 * v)) * v.H
        P[:, j + 1:] = P[:, j + 1:] - (P[:, j + 1:] * (2 * v)) * v.H

    # Return the upper Hessenberg matrix H and the orthogonal matrix P
    return H, P
```