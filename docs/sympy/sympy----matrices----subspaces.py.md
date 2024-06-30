# `D:\src\scipysrc\sympy\sympy\matrices\subspaces.py`

```
# 导入自定义的_iszero函数，用于判断是否为零
from .utilities import _iszero

# 定义函数 _columnspace，返回矩阵 M 的列空间的基向量列表
def _columnspace(M, simplify=False):
    """Returns a list of vectors (Matrix objects) that span columnspace of ``M``

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])
    >>> M
    Matrix([
    [ 1,  3, 0],
    [-2, -6, 0],
    [ 3,  9, 6]])
    >>> M.columnspace()
    [Matrix([
    [ 1],
    [-2],
    [ 3]]), Matrix([
    [0],
    [0],
    [6]])]

    See Also
    ========

    nullspace
    rowspace
    """

    # 计算化简行阶梯形式并返回主列和化简矩阵
    reduced, pivots = M.echelon_form(simplify=simplify, with_pivots=True)

    # 返回主列对应的列向量列表
    return [M.col(i) for i in pivots]


# 定义函数 _nullspace，返回矩阵 M 的零空间的基向量列表
def _nullspace(M, simplify=False, iszerofunc=_iszero):
    """Returns list of vectors (Matrix objects) that span nullspace of ``M``

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])
    >>> M
    Matrix([
    [ 1,  3, 0],
    [-2, -6, 0],
    [ 3,  9, 6]])
    >>> M.nullspace()
    [Matrix([
    [-3],
    [ 1],
    [ 0]])]

    See Also
    ========

    columnspace
    rowspace
    """

    # 计算行简化阶梯形式和主列，并获取自由变量索引
    reduced, pivots = M.rref(iszerofunc=iszerofunc, simplify=simplify)
    free_vars = [i for i in range(M.cols) if i not in pivots]
    basis = []

    # 对每个自由变量生成基向量
    for free_var in free_vars:
        vec = [M.zero] * M.cols
        vec[free_var] = M.one

        # 后向替换法求解线性方程组
        for piv_row, piv_col in enumerate(pivots):
            vec[piv_col] -= reduced[piv_row, free_var]

        basis.append(vec)

    # 返回基向量列表
    return [M._new(M.cols, 1, b) for b in basis]


# 定义函数 _rowspace，返回矩阵 M 的行空间的基向量列表
def _rowspace(M, simplify=False):
    """Returns a list of vectors that span the row space of ``M``.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [1, 3, 0, -2, -6, 0, 3, 9, 6])
    >>> M
    Matrix([
    [ 1,  3, 0],
    [-2, -6, 0],
    [ 3,  9, 6]])
    >>> M.rowspace()
    [Matrix([[1, 3, 0]]), Matrix([[0, 0, 6]])]
    """

    # 计算化简行阶梯形式并返回每个主列对应的行向量列表
    reduced, pivots = M.echelon_form(simplify=simplify, with_pivots=True)

    # 返回行向量列表
    return [reduced.row(i) for i in range(len(pivots))]


# 定义函数 _orthogonalize，对给定的向量列表进行 Gram-Schmidt 正交化处理
def _orthogonalize(cls, *vecs, normalize=False, rankcheck=False):
    """Apply the Gram-Schmidt orthogonalization procedure
    to vectors supplied in ``vecs``.

    Parameters
    ==========

    vecs
        vectors to be made orthogonal

    normalize : bool
        If ``True``, return an orthonormal basis.

    rankcheck : bool
        If ``True``, the computation does not stop when encountering
        linearly dependent vectors.

        If ``False``, it will raise ``ValueError`` when any zero
        or linearly dependent vectors are found.

    Returns
    =======

    list
        List of orthogonal (or orthonormal) basis vectors.

    Examples
    ========

    >>> from sympy import I, Matrix
    >>> v = [Matrix([1, I]), Matrix([1, -I])]
    >>> Matrix.orthogonalize(*v)
    [Matrix([
    [1],
    ...
    [0]]), Matrix([
    [0],
    ...
    [0]])]
    """

    # 实现 Gram-Schmidt 正交化过程，生成正交化后的向量列表
    """
    Gram-Schmidt正交化过程，将输入的向量集合进行正交化处理，并返回正交化后的向量集合。

    Parameters
    ==========
    vecs : list of Matrix
        要进行正交化处理的向量列表

    normalize : bool, optional
        是否对正交化后的向量进行归一化，默认为True

    rankcheck : bool, optional
        是否检查向量集合的线性独立性，默认为False

    Returns
    =======
    list of Matrix
        正交化处理后的向量列表

    Raises
    ======
    ValueError
        如果向量集合不是线性独立的，则抛出该异常

    See Also
    ========
    MatrixBase.QRdecomposition

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    """
    from .decompositions import _QRdecomposition_optional

    if not vecs:
        return []

    all_row_vecs = (vecs[0].rows == 1)

    # 将向量列表中的每个向量转换为列向量
    vecs = [x.vec() for x in vecs]
    # 将所有列向量水平连接形成一个矩阵M
    M = cls.hstack(*vecs)
    # 对矩阵M进行可选的QR分解，并返回Q为正交矩阵，R为上三角矩阵
    Q, R = _QRdecomposition_optional(M, normalize=normalize)

    # 如果启用了rankcheck且Q的列数小于向量列表的长度，则抛出异常
    if rankcheck and Q.cols < len(vecs):
        raise ValueError("GramSchmidt: vector set not linearly independent")

    ret = []
    # 遍历Q的列，根据all_row_vecs的值构造列向量或行向量，并加入结果列表ret中
    for i in range(Q.cols):
        if all_row_vecs:
            col = cls(Q[:, i].T)
        else:
            col = cls(Q[:, i])
        ret.append(col)
    # 返回正交化处理后的向量列表ret
    return ret
```