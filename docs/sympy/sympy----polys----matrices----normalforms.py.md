# `D:\src\scipysrc\sympy\sympy\polys\matrices\normalforms.py`

```
# 引入必要的模块和类
from collections import defaultdict
from .domainmatrix import DomainMatrix
from .exceptions import DMDomainError, DMShapeError
from sympy.ntheory.modular import symmetric_residue
from sympy.polys.domains import QQ, ZZ

# TODO（未来工作）：
# Smith 和 Hermite 正则形式存在更快的算法，我们应该实现。
# 参考 Kannan-Bachem 算法：https://www.researchgate.net/publication/220617516_Polynomial_Algorithms_for_Computing_the_Smith_and_Hermite_Normal_Forms_of_an_Integer_Matrix

def smith_normal_form(m):
    '''
    返回矩阵 `m` 在环 `domain` 上的 Smith 正则形式。
    只有在环是主理想整环时才有效。

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> from sympy.polys.matrices.normalforms import smith_normal_form
    >>> m = DomainMatrix([[ZZ(12), ZZ(6), ZZ(4)],
    ...                   [ZZ(3), ZZ(9), ZZ(6)],
    ...                   [ZZ(2), ZZ(16), ZZ(14)]], (3, 3), ZZ)
    >>> print(smith_normal_form(m).to_Matrix())
    Matrix([[1, 0, 0], [0, 10, 0], [0, 0, -30]])

    '''
    # 计算矩阵的不变因子
    invs = invariant_factors(m)
    # 构造由不变因子构成的对角矩阵
    smf = DomainMatrix.diag(invs, m.domain, m.shape)
    return smf


def add_columns(m, i, j, a, b, c, d):
    '''
    将 m 的第 i 列替换为 a*m[:, i] + b*m[:, j]
    将 m 的第 j 列替换为 c*m[:, i] + d*m[:, j]
    '''
    for k in range(len(m)):
        e = m[k][i]
        m[k][i] = a*e + b*m[k][j]
        m[k][j] = c*e + d*m[k][j]


def invariant_factors(m):
    '''
    返回矩阵 `m` 的阿贝尔不变因子的元组（即 Smith 正则形式中的结果）

    References
    ==========

    [1] https://en.wikipedia.org/wiki/Smith_normal_form#Algorithm
    [2] https://web.archive.org/web/20200331143852/https://sierra.nmsu.edu/morandi/notes/SmithNormalForm.pdf

    '''
    # 确定矩阵的定义域
    domain = m.domain
    # 如果定义域不是主理想整环，则抛出异常
    if not domain.is_PID:
        msg = "矩阵条目必须在主理想整环上"
        raise ValueError(msg)

    # 如果矩阵形状为零，则返回空元组
    if 0 in m.shape:
        return ()

    rows, cols = shape = m.shape
    # 将稀疏矩阵转换为稠密矩阵，并转换为列表形式
    m = list(m.to_dense().rep.to_ddm())

    def add_rows(m, i, j, a, b, c, d):
        '''
        将 m 的第 i 行替换为 a*m[i, :] + b*m[j, :]
        将 m 的第 j 行替换为 c*m[i, :] + d*m[j, :]
        '''
        for k in range(cols):
            e = m[i][k]
            m[i][k] = a*e + b*m[j][k]
            m[j][k] = c*e + d*m[j][k]
    def clear_column(m):
        # 通过行和列操作使得 m[1:, 0] 变为零
        if m[0][0] == 0:
            return m  # 如果 m[0][0] 为零，直接返回，覆盖率检测注释
        pivot = m[0][0]  # 设定主元为 m[0][0]
        for j in range(1, rows):
            if m[j][0] == 0:
                continue
            d, r = domain.div(m[j][0], pivot)  # 使用 domain 的除法计算商和余数
            if r == 0:
                add_rows(m, 0, j, 1, 0, -d, 1)  # 添加行以消除 m[j][0] 的影响
            else:
                a, b, g = domain.gcdex(pivot, m[j][0])  # 使用 domain 的扩展欧几里得算法计算最大公因数和系数
                d_0 = domain.div(m[j][0], g)[0]  # 计算 m[j][0] 与最大公因数的商
                d_j = domain.div(pivot, g)[0]  # 计算 pivot 与最大公因数的商
                add_rows(m, 0, j, a, b, d_0, -d_j)  # 添加行以消除 m[j][0] 的影响，使用系数
                pivot = g  # 更新主元为最大公因数
        return m  # 返回处理后的矩阵
    
    def clear_row(m):
        # 通过行和列操作使得 m[0, 1:] 变为零
        if m[0][0] == 0:
            return m  # 如果 m[0][0] 为零，直接返回，覆盖率检测注释
        pivot = m[0][0]  # 设定主元为 m[0][0]
        for j in range(1, cols):
            if m[0][j] == 0:
                continue
            d, r = domain.div(m[0][j], pivot)  # 使用 domain 的除法计算商和余数
            if r == 0:
                add_columns(m, 0, j, 1, 0, -d, 1)  # 添加列以消除 m[0][j] 的影响
            else:
                a, b, g = domain.gcdex(pivot, m[0][j])  # 使用 domain 的扩展欧几里得算法计算最大公因数和系数
                d_0 = domain.div(m[0][j], g)[0]  # 计算 m[0][j] 与最大公因数的商
                d_j = domain.div(pivot, g)[0]  # 计算 pivot 与最大公因数的商
                add_columns(m, 0, j, a, b, d_0, -d_j)  # 添加列以消除 m[0][j] 的影响，使用系数
                pivot = g  # 更新主元为最大公因数
        return m  # 返回处理后的矩阵
    
    # 重排行和列，直到 m[0,0] 可能不为零
    ind = [i for i in range(rows) if m[i][0] != 0]  # 找出第一列非零元素的行索引列表
    if ind and ind[0] != 0:
        m[0], m[ind[0]] = m[ind[0]], m[0]  # 如果第一个非零元素不在第一行，交换行
    else:
        ind = [j for j in range(cols) if m[0][j] != 0]  # 找出第一行非零元素的列索引列表
        if ind and ind[0] != 0:
            for row in m:
                row[0], row[ind[0]] = row[ind[0]], row[0]  # 如果第一个非零元素不在第一列，交换列
    
    # 使得第一行和第一列以外的元素都为零
    while (any(m[0][i] != 0 for i in range(1, cols)) or
           any(m[i][0] != 0 for i in range(1, rows))):
        m = clear_column(m)  # 清除列
        m = clear_row(m)  # 清除行
    
    if 1 in shape:
        invs = ()
    else:
        lower_right = DomainMatrix([r[1:] for r in m[1:]], (rows-1, cols-1), domain)  # 取出右下角的子矩阵
        invs = invariant_factors(lower_right)  # 计算子矩阵的不变因子
    
    if m[0][0]:
        result = [m[0][0]]
        result.extend(invs)
        # 如果 m[0] 不能整除其余矩阵的不变因子
        for i in range(len(result)-1):
            if result[i] and domain.div(result[i+1], result[i])[1] != 0:
                g = domain.gcd(result[i+1], result[i])  # 计算最大公因数
                result[i+1] = domain.div(result[i], g)[0] * result[i+1]  # 更新不变因子
                result[i] = g  # 更新最大公因数
            else:
                break
    else:
        result = invs + (m[0][0],)  # 如果 m[0][0] 为零，返回不变因子和该元素
    
    return tuple(result)  # 返回结果元组
# 计算扩展欧几里得算法的结果，用于支持计算 Hermite 正规形式的函数。
def _gcdex(a, b):
    # 调用 ZZ.gcdex(a, b)，返回值为 x, y, g，其中 x*a + y*b = g。
    x, y, g = ZZ.gcdex(a, b)
    # 如果 a 不为零且 b 能被 a 整除，则设置 y 为零，x 为 -1（如果 a < 0）或 1。
    if a != 0 and b % a == 0:
        y = 0
        x = -1 if a < 0 else 1
    # 返回计算结果 x, y, g。
    return x, y, g


# 计算 DomainMatrix A 的 Hermite 正规形式，要求 A 的域为 ZZ。
def _hermite_normal_form(A):
    # 如果 A 的域不是 ZZ，则抛出 DMDomainError 异常。
    if not A.domain.is_ZZ:
        raise DMDomainError('Matrix must be over domain ZZ.')
    
    # 获取矩阵 A 的行数 m 和列数 n。
    m, n = A.shape
    # 将 A 转换为稠密矩阵，并复制为双重域矩阵形式。
    A = A.to_dense().rep.to_ddm().copy()
    
    # 我们的目标是将主元素放在最右边的列。
    # 不变量：在处理每一行之前，k 应该是到目前为止我们放置主元素的最左列的索引。
    k = n
    for i in range(m - 1, -1, -1):
        # 从最后一行向第一行遍历，逆序处理每一行
        if k == 0:
            # 当 k 为 0 时，表示已经找到了所需的 n 个主元（pivot），此时不需要再考虑更多的行，
            # 因为这已经是可能的主元数量的最大值。
            break
        k -= 1
        # k 现在指向我们要放置主元的列。
        # 我们希望在主元列左侧的所有条目都为零。
        for j in range(k - 1, -1, -1):
            if A[i][j] != 0:
                # 用列 j 和列 k 的线性组合替换它们，以便在第 i 行中，列 j 为 0，而列 k 为它们行 i 条目的最大公约数。
                # 这确保了列 k 中有非零条目。
                u, v, d = _gcdex(A[i][k], A[i][j])
                r, s = A[i][k] // d, A[i][j] // d
                add_columns(A, k, j, u, v, -s, r)
        b = A[i][k]
        # 不希望主元条目为负数。
        if b < 0:
            # 如果主元为负数，则通过线性组合确保主元为正数。
            add_columns(A, k, k, -1, 0, -1, 0)
            b = -b
        # 如果主元条目为零，则说明从主元列向左到末尾的行全为零。
        # 在这种情况下，我们仍在处理同一主元列的下一行。因此：
        if b == 0:
            k += 1
        # 如果主元条目非零，则希望在其右侧的所有条目符合除法算法的意义，即它们都是相对于主元的余数。
        else:
            for j in range(k + 1, n):
                q = A[i][j] // b
                add_columns(A, j, k, 1, -q, 0, 1)
    # 最后，HNF 由 A 矩阵中成功设置了非零主元的列组成。
    return DomainMatrix.from_rep(A.to_dfm_or_ddm())[:, k:]
# 定义一个函数，执行模 *D* Hermite正规形式归约算法在 :py:class:`~.DomainMatrix` *A* 上
def _hermite_normal_form_modulo_D(A, D):
    r"""
    Perform the mod *D* Hermite Normal Form reduction algorithm on
    :py:class:`~.DomainMatrix` *A*.

    Explanation
    ===========

    If *A* is an $m \times n$ matrix of rank $m$, having Hermite Normal Form
    $W$, and if *D* is any positive integer known in advance to be a multiple
    of $\det(W)$, then the HNF of *A* can be computed by an algorithm that
    works mod *D* in order to prevent coefficient explosion.

    Parameters
    ==========

    A : :py:class:`~.DomainMatrix` over :ref:`ZZ`
        $m \times n$ matrix, having rank $m$.
        输入参数 A 是一个 :py:class:`~.DomainMatrix` 类型，其定义在整数环 :ref:`ZZ` 上，维度为 $m \times n$，并且其秩为 $m$。

    D : :ref:`ZZ`
        Positive integer, known to be a multiple of the determinant of the
        HNF of *A*.
        输入参数 D 是一个正整数，预先已知是 *A* 的 Hermite 正规形式的行列式的倍数。

    Returns
    =======

    :py:class:`~.DomainMatrix`
        The HNF of matrix *A*.
        返回值是矩阵 *A* 的 Hermite 正规形式。

    Raises
    ======

    DMDomainError
        If the domain of the matrix is not :ref:`ZZ`, or
        if *D* is given but is not in :ref:`ZZ`.
        如果矩阵的域不是 :ref:`ZZ`，或者给定的 *D* 不在 :ref:`ZZ` 中，则引发异常 DMDomainError。

    DMShapeError
        If the matrix has more rows than columns.
        如果矩阵的行数多于列数，则引发异常 DMShapeError。

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithm 2.4.8.)

    """

    # 检查矩阵 A 的域是否为整数环 ZZ
    if not A.domain.is_ZZ:
        raise DMDomainError('Matrix must be over domain ZZ.')
    
    # 检查输入的 D 是否为正整数且属于整数环 ZZ
    if not ZZ.of_type(D) or D < 1:
        raise DMDomainError('Modulus D must be positive element of domain ZZ.')

    # 定义一个函数，用于在模 R 下添加两列的矩阵操作
    def add_columns_mod_R(m, R, i, j, a, b, c, d):
        # replace m[:, i] by (a*m[:, i] + b*m[:, j]) % R
        # and m[:, j] by (c*m[:, i] + d*m[:, j]) % R
        # 循环处理矩阵 m 的每一行
        for k in range(len(m)):
            e = m[k][i]
            # 更新第 i 列为 (a*m[:, i] + b*m[:, j]) % R
            m[k][i] = symmetric_residue((a * e + b * m[k][j]) % R, R)
            # 更新第 j 列为 (c*m[:, i] + d*m[:, j]) % R
            m[k][j] = symmetric_residue((c * e + d * m[k][j]) % R, R)

    # 创建一个空的默认字典 W
    W = defaultdict(dict)

    # 获取矩阵 A 的形状
    m, n = A.shape
    
    # 如果列数小于行数，抛出异常 DMShapeError
    if n < m:
        raise DMShapeError('Matrix must have at least as many columns as rows.')

    # 将矩阵 A 转换为稠密矩阵，再转换为可分块对角矩阵，最后复制一份
    A = A.to_dense().rep.to_ddm().copy()

    # 初始化变量 k 为列数 n，初始化模数 R 为 D
    k = n
    R = D

    # 从行数 m-1 循环到 0，逐步计算 Hermite 正规形式
    for i in range(m - 1, -1, -1):
        k -= 1
        # 从列数 k-1 循环到 0
        for j in range(k - 1, -1, -1):
            # 如果 A[i][j] 不为零
            if A[i][j] != 0:
                # 使用扩展欧几里得算法计算 u, v, d
                u, v, d = _gcdex(A[i][k], A[i][j])
                r, s = A[i][k] // d, A[i][j] // d
                # 调用函数 add_columns_mod_R 更新矩阵 A 的列 i 和 j
                add_columns_mod_R(A, R, k, j, u, v, -s, r)
        
        # 获取当前 A[i][k] 的值
        b = A[i][k]
        # 如果 b 等于零，将其更新为 R
        if b == 0:
            A[i][k] = b = R
        # 使用扩展欧几里得算法计算 u, v, d
        u, v, d = _gcdex(b, R)
        
        # 更新字典 W 中的值
        for ii in range(m):
            W[ii][i] = u*A[ii][k] % R
        
        # 如果 W[i][i] 等于零，将其更新为 R
        if W[i][i] == 0:
            W[i][i] = R
        
        # 从 i+1 行循环到 m，更新 W 中的值
        for j in range(i + 1, m):
            q = W[i][j] // W[i][i]
            add_columns(W, j, i, 1, -q, 0, 1)
        
        # 将 R 除以 d
        R //= d

    # 返回 DomainMatrix 对象，使用 W 构造其形状为 (m, m)，并在整数环 ZZ 上转换为稠密矩阵
    return DomainMatrix(W, (m, m), ZZ).to_dense()
    # 检查矩阵 A 的域是否为整数环 ZZ，若不是则抛出异常
    if not A.domain.is_ZZ:
        raise DMDomainError('Matrix must be over domain ZZ.')
    
    # 如果提供了 D 并且没有要求检查矩阵的秩，或者 A 的秩等于其行数，则使用模 D 的算法
    if D is not None and (not check_rank or A.convert_to(QQ).rank() == A.shape[0]):
        # 返回通过模 D 算法得到的 Hermite 正则形式
        return _hermite_normal_form_modulo_D(A, D)
    else:
        # 否则，使用普通的 Hermite 正则形式算法
        return _hermite_normal_form(A)


这段代码是一个函数的实现，根据输入的参数和条件选择不同的算法来计算矩阵 A 的 Hermite 正则形式（Hermite Normal Form）。
```