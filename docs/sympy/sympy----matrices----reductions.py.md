# `D:\src\scipysrc\sympy\sympy\matrices\reductions.py`

```
    # 导入需要的模块和函数
    from types import FunctionType
    from sympy.polys.polyerrors import CoercionFailed
    from sympy.polys.domains import ZZ, QQ
    # 导入本模块中定义的工具函数和方法
    from .utilities import _get_intermediate_simp, _iszero, _dotprodsimp, _simplify
    from .determinant import _find_reasonable_pivot

def _row_reduce_list(mat, rows, cols, one, iszerofunc, simpfunc,
                normalize_last=True, normalize=True, zero_above=True):
    """Row reduce a flat list representation of a matrix and return a tuple
    (rref_matrix, pivot_cols, swaps) where ``rref_matrix`` is a flat list,
    ``pivot_cols`` are the pivot columns and ``swaps`` are any row swaps that
    were used in the process of row reduction.

    Parameters
    ==========

    mat : list
        list of matrix elements, must be ``rows`` * ``cols`` in length

    rows, cols : integer
        number of rows and columns in flat list representation

    one : SymPy object
        represents the value one, from ``Matrix.one``

    iszerofunc : determines if an entry can be used as a pivot

    simpfunc : used to simplify elements and test if they are
        zero if ``iszerofunc`` returns `None`

    normalize_last : indicates where all row reduction should
        happen in a fraction-free manner and then the rows are
        normalized (so that the pivots are 1), or whether
        rows should be normalized along the way (like the naive
        row reduction algorithm)

    normalize : whether pivot rows should be normalized so that
        the pivot value is 1

    zero_above : whether entries above the pivot should be zeroed.
        If ``zero_above=False``, an echelon matrix will be returned.
    """

    # 定义内部函数，用于获取矩阵的列
    def get_col(i):
        return mat[i::cols]

    # 定义内部函数，用于交换矩阵的两行
    def row_swap(i, j):
        mat[i*cols:(i + 1)*cols], mat[j*cols:(j + 1)*cols] = \
            mat[j*cols:(j + 1)*cols], mat[i*cols:(i + 1)*cols]

    # 定义内部函数，用于执行行消除操作
    def cross_cancel(a, i, b, j):
        """Does the row op row[i] = a*row[i] - b*row[j]"""
        q = (j - i)*cols
        for p in range(i*cols, (i + 1)*cols):
            mat[p] = isimp(a*mat[p] - b*mat[p + q])

    # 获取用于简化操作的中间函数
    isimp = _get_intermediate_simp(_dotprodsimp)
    # 初始化主元行和列的索引
    piv_row, piv_col = 0, 0
    # 初始化主元列列表和行交换列表
    pivot_cols = []
    swaps = []

    # 使用无分数方法，将每个主元上下都置零
    # 当主列指针小于列数并且主行指针小于行数时，执行循环
    while piv_col < cols and piv_row < rows:
        # 调用 _find_reasonable_pivot 函数找到合适的主元素位置、值及其他信息
        pivot_offset, pivot_val, \
        assumed_nonzero, newly_determined = _find_reasonable_pivot(
                get_col(piv_col)[piv_row:], iszerofunc, simpfunc)

        # _find_reasonable_pivot 可能会在过程中简化一些内容，让我们不要浪费它们
        # 将新确定的值更新到矩阵中相应的位置
        for (offset, val) in newly_determined:
            offset += piv_row
            mat[offset*cols + piv_col] = val

        # 如果找不到合适的主元素位置，增加主列指针并继续下一轮循环
        if pivot_offset is None:
            piv_col += 1
            continue

        # 将当前主列指针添加到主列列表中
        pivot_cols.append(piv_col)
        # 如果主元素不在主对角线上，交换当前行和主元素所在行
        if pivot_offset != 0:
            row_swap(piv_row, pivot_offset + piv_row)
            swaps.append((piv_row, pivot_offset + piv_row))

        # 如果不是在最后进行归一化，进行归一化处理
        if normalize_last is False:
            i, j = piv_row, piv_col
            # 将主元素所在位置归一化为 1，同时归一化当前行其他元素
            mat[i*cols + j] = one
            for p in range(i*cols + j + 1, (i + 1)*cols):
                mat[p] = isimp(mat[p] / pivot_val)
            # 归一化后，主元素的值变为 1
            pivot_val = one

        # 将主元素上下方的其他行归零
        for row in range(rows):
            # 跳过当前行
            if row == piv_row:
                continue
            # 如果不归零上方的行，并且当前行在主元素上方，跳过
            if zero_above is False and row < piv_row:
                continue
            # 如果当前位置已经是零，不进行处理
            val = mat[row*cols + piv_col]
            if iszerofunc(val):
                continue

            # 使用交叉消去法将当前行的元素归零
            cross_cancel(pivot_val, row, val, piv_row)
        
        # 主行指针向下移动一行
        piv_row += 1

    # 如果设置了在最后进行归一化，并且允许归一化，则对每一行进行归一化处理
    if normalize_last is True and normalize is True:
        for piv_i, piv_j in enumerate(pivot_cols):
            pivot_val = mat[piv_i*cols + piv_j]
            # 将当前行的主元素归一化为 1，同时归一化当前行其他元素
            mat[piv_i*cols + piv_j] = one
            for p in range(piv_i*cols + piv_j + 1, (piv_i + 1)*cols):
                mat[p] = isimp(mat[p] / pivot_val)

    # 返回处理后的矩阵、主列列表和交换元组
    return mat, tuple(pivot_cols), tuple(swaps)
# This functions is a candidate for caching if it gets implemented for matrices.
# 如果为矩阵实现了缓存功能，这个函数将是一个候选项。

def _row_reduce(M, iszerofunc, simpfunc, normalize_last=True,
                normalize=True, zero_above=True):
    # 调用 _row_reduce_list 函数对矩阵 M 进行行简化操作，返回简化后的矩阵 mat、主元列 pivot_cols 和交换信息 swaps
    mat, pivot_cols, swaps = _row_reduce_list(list(M), M.rows, M.cols, M.one,
            iszerofunc, simpfunc, normalize_last=normalize_last,
            normalize=normalize, zero_above=zero_above)
    
    # 返回一个新的矩阵对象，包括简化后的矩阵 mat、主元列 pivot_cols 和交换信息 swaps
    return M._new(M.rows, M.cols, mat), pivot_cols, swaps


def _is_echelon(M, iszerofunc=_iszero):
    """Returns `True` if the matrix is in echelon form. That is, all rows of
    zeros are at the bottom, and below each leading non-zero in a row are
    exclusively zeros."""
    # 如果矩阵行数或列数小于等于 0，直接返回 True
    if M.rows <= 0 or M.cols <= 0:
        return True
    
    # 检查除第一行外的所有行第一列元素是否都为零
    zeros_below = all(iszerofunc(t) for t in M[1:, 0])
    
    # 如果第一行第一列元素为零，则递归检查去除第一行和第一列后的子矩阵
    if iszerofunc(M[0, 0]):
        return zeros_below and _is_echelon(M[:, 1:], iszerofunc)
    
    # 否则，递归检查去除第一行和第一列后的子矩阵
    return zeros_below and _is_echelon(M[1:, 1:], iszerofunc)


def _echelon_form(M, iszerofunc=_iszero, simplify=False, with_pivots=False):
    """Returns a matrix row-equivalent to ``M`` that is in echelon form. Note
    that echelon form of a matrix is *not* unique, however, properties like the
    row space and the null space are preserved.
    
    返回一个与 M 行等价的、处于梯形形式的矩阵。注意，矩阵的梯形形式并不是唯一的，但其行空间和零空间等特性是保持不变的。

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> M.echelon_form()
    Matrix([
    [1,  2],
    [0, -2]])
    """
    # 如果 simplify 是函数类型，则将其作为简化函数，否则使用内部默认的简化函数 _simplify
    simpfunc = simplify if isinstance(simplify, FunctionType) else _simplify
    
    # 调用 _row_reduce 函数对矩阵 M 进行行简化操作，获取简化后的矩阵 mat 和主元列 pivots
    mat, pivots, _ = _row_reduce(M, iszerofunc, simpfunc,
            normalize_last=True, normalize=False, zero_above=False)
    
    # 如果需要返回主元列 pivots，则返回简化后的矩阵 mat 和主元列 pivots
    if with_pivots:
        return mat, pivots
    
    # 否则，只返回简化后的矩阵 mat
    return mat


# This functions is a candidate for caching if it gets implemented for matrices.
# 如果为矩阵实现了缓存功能，这个函数将是一个候选项。
def _rank(M, iszerofunc=_iszero, simplify=False):
    """Returns the rank of a matrix.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> m = Matrix([[1, 2], [x, 1 - 1/x]])
    >>> m.rank()
    2
    >>> n = Matrix(3, 3, range(1, 10))
    >>> n.rank()
    2
    """
    
    def _permute_complexity_right(M, iszerofunc):
        """Permute columns with complicated elements as
        far right as they can go.  Since the ``sympy`` row reduction
        algorithms start on the left, having complexity right-shifted
        speeds things up.

        Returns a tuple (mat, perm) where perm is a permutation
        of the columns to perform to shift the complex columns right, and mat
        is the permuted matrix."""
        
        def complexity(i):
            # 判断一列中有多少元素的零判断结果无法确定
            return sum(1 if iszerofunc(e) is None else 0 for e in M[:, i])

        # 对复杂列进行右移，以加快``sympy``行约简算法的速度
        complex = [(complexity(i), i) for i in range(M.cols)]
        perm    = [j for (i, j) in sorted(complex)]

        return (M.permute(perm, orientation='cols'), perm)

    # 如果 simplify 是函数类型，则将其作为简化函数，否则使用内部默认的简化函数 _simplify
    simpfunc = simplify if isinstance(simplify, FunctionType) else _simplify
    # 如果矩阵行数或列数小于等于零，直接返回秩为零
    if M.rows <= 0 or M.cols <= 0:
        return 0

    # 如果矩阵行数或列数为1，或者两者之一为1，检查矩阵每个元素是否为零
    # 如果存在非零元素，返回秩为1
    if M.rows <= 1 or M.cols <= 1:
        zeros = [iszerofunc(x) for x in M]

        if False in zeros:
            return 1

    # 如果矩阵为2x2，检查矩阵每个元素是否为零
    if M.rows == 2 and M.cols == 2:
        zeros = [iszerofunc(x) for x in M]

        # 如果所有元素都为零，返回秩为0
        if False not in zeros and None not in zeros:
            return 0

        # 计算矩阵的行列式值
        d = M.det()

        # 如果行列式为零且存在非零元素，返回秩为1
        if iszerofunc(d) and False in zeros:
            return 1
        # 如果行列式不为零，返回秩为2
        if iszerofunc(d) is False:
            return 2

    # 对矩阵进行初等行变换和化简，计算主元的数量
    mat, _       = _permute_complexity_right(M, iszerofunc=iszerofunc)
    _, pivots, _ = _row_reduce(mat, iszerofunc, simpfunc, normalize_last=True,
            normalize=False, zero_above=False)

    # 返回矩阵的秩，即主元的数量
    return len(pivots)
def _to_DM_ZZ_QQ(M):
    # 检查对象 M 是否具有属性 '_rep'，如果没有则返回 None
    # 这是因为某些测试可能会因为缺少 '_rep' 属性而失败，如出现
    # "AttributeError: 'SubspaceOnlyMatrix' object has no attribute '_rep'."
    # 如果不做此检查，可能会影响部分测试用例的通过，但实际上这些测试用例
    # 似乎并没有太大意义。假设是有人试图通过继承某些 Matrix 类的方式创建
    # 了一个新的类，但没有继承到标准 Matrix 类所使用的全部属性，这种情况
    # 下会导致多种问题。
    if not hasattr(M, '_rep'):
        return None

    # 获取 M 的 _rep 属性
    rep = M._rep
    # 获取 rep 的域 K
    K = rep.domain

    # 如果域 K 是整数环 ZZ，则直接返回 rep
    if K.is_ZZ:
        return rep
    # 如果域 K 是有理数域 QQ，则尝试将 rep 转换为整数环 ZZ
    elif K.is_QQ:
        try:
            return rep.convert_to(ZZ)
        except CoercionFailed:
            return rep
    # 对于其他情况，检查 M 中的元素是否全为有理数，如果不是则返回 None
    else:
        if not all(e.is_Rational for e in M):
            return None
        # 尝试将 rep 转换为整数环 ZZ，如果失败则尝试转换为有理数域 QQ
        try:
            return rep.convert_to(ZZ)
        except CoercionFailed:
            return rep.convert_to(QQ)


def _rref_dm(dM):
    """计算域矩阵的简化行阶梯形式。"""
    # 获取域矩阵 dM 的域 K
    K = dM.domain

    # 如果域 K 是整数环 ZZ，则计算简化行阶梯形式，返回结果、最小公倍数和主元素位置
    if K.is_ZZ:
        dM_rref, den, pivots = dM.rref_den(keep_domain=False)
        dM_rref = dM_rref.to_field() / den
    # 如果域 K 是有理数域 QQ，则计算简化行阶梯形式，返回结果和主元素位置
    elif K.is_QQ:
        dM_rref, pivots = dM.rref()
    else:
        # 如果域 K 不是整数环 ZZ 也不是有理数域 QQ，则断言失败，应该不会被覆盖到
        assert False  # pragma: no cover

    # 将域矩阵的简化行阶梯形式转换为一般矩阵形式并返回
    M_rref = dM_rref.to_Matrix()

    return M_rref, pivots


def _rref(M, iszerofunc=_iszero, simplify=False, pivots=True,
        normalize_last=True):
    """返回矩阵的简化行阶梯形式以及主元素的索引。

    参数
    ==========

    iszerofunc : Function
        用于检测一个元素是否可以作为主元素的函数，默认为 ``lambda x: x.is_zero``.

    simplify : Function
        用于简化元素以便寻找主元素的函数，默认使用 SymPy 的 ``simplify`` 函数.

    pivots : True 或 False
        如果为 ``True``，返回包含简化行阶梯形式矩阵和主元素索引的元组。
        如果为 ``False``，则仅返回简化行阶梯形式矩阵.

    normalize_last : True 或 False
        如果为 ``True``，在所有主元素上下条目都归零之前，不对主元素进行归一化为 `1`。
        这意味着在最后一步之前，行约简算法是不含分数的。
        如果为 ``False``，则使用朴素的行约简过程，即在使用行操作将每个主元素归零之前，
        先将主元素归一化为 `1`。

    示例
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> m = Matrix([[1, 2], [x, 1 - 1/x]])
    >>> m.rref()
    (Matrix([
    [1, 0],
    [0, 1]]), (0, 1))
    >>> rref_matrix, rref_pivots = m.rref()
    >>> rref_matrix
    Matrix([
    [1, 0],
    [0, 1]])
    >>> rref_pivots
    (0, 1)

    ``iszerofunc`` 可以纠正带有浮点值的矩阵中的舍入误差。在以下示例中，
    调用 ``rref()`` 导致
    ```
    floating point errors, incorrectly row reducing the matrix.
    ``iszerofunc= lambda x: abs(x) < 1e-9`` sets sufficiently small numbers
    to zero, avoiding this error.

    >>> m = Matrix([[0.9, -0.1, -0.2, 0], [-0.8, 0.9, -0.4, 0], [-0.1, -0.8, 0.6, 0]])
    >>> m.rref()
    (Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]]), (0, 1, 2))
    >>> m.rref(iszerofunc=lambda x:abs(x)<1e-9)
    (Matrix([
    [1, 0, -0.301369863013699, 0],
    [0, 1, -0.712328767123288, 0],
    [0, 0,         0,          0]]), (0, 1))

    Notes
    =====

    The default value of ``normalize_last=True`` can provide significant
    speedup to row reduction, especially on matrices with symbols.  However,
    if you depend on the form row reduction algorithm leaves entries
    of the matrix, set ``normalize_last=False``
    """
    # 尝试使用 DomainMatrix 处理整数或有理数矩阵 ZZ 或 QQ
    dM = _to_DM_ZZ_QQ(M)

    if dM is not None:
        # 使用 DomainMatrix 处理整数或有理数矩阵 ZZ 或 QQ
        mat, pivot_cols = _rref_dm(dM)
    else:
        # 使用通用的 Matrix 行简化算法
        if isinstance(simplify, FunctionType):
            simpfunc = simplify
        else:
            simpfunc = _simplify

        # 调用通用行简化函数进行行简化操作
        mat, pivot_cols, _ = _row_reduce(M, iszerofunc, simpfunc,
                normalize_last, normalize=True, zero_above=True)

    if pivots:
        return mat, pivot_cols
    else:
        return mat
```