# `D:\src\scipysrc\sympy\sympy\polys\matrices\dense.py`

```
# 导入必要的异常类
from .exceptions import (
    DMShapeError,
    DMDomainError,
    DMNonInvertibleMatrixError,
    DMNonSquareMatrixError,
)
# 导入类型变量
from typing import Sequence, TypeVar
from sympy.polys.matrices._typing import RingElement

# 类型变量，表示矩阵元素的类型 T
T = TypeVar('T')

# 类型变量，表示环中的矩阵元素的类型 R，继承自 RingElement
R = TypeVar('R', bound=RingElement)

# 矩阵转置函数，将给定矩阵 matrix 转置
def ddm_transpose(matrix: Sequence[Sequence[T]]) -> list[list[T]]:
    """matrix transpose"""
    return list(map(list, zip(*matrix)))

# 矩阵加法，将矩阵 a 与矩阵 b 相加，结果保存在 a 中
def ddm_iadd(a: list[list[R]], b: Sequence[Sequence[R]]) -> None:
    """a += b"""
    for ai, bi in zip(a, b):
        for j, bij in enumerate(bi):
            ai[j] += bij

# 矩阵减法，将矩阵 a 减去矩阵 b，结果保存在 a 中
def ddm_isub(a: list[list[R]], b: Sequence[Sequence[R]]) -> None:
    """a -= b"""
    for ai, bi in zip(a, b):
        for j, bij in enumerate(bi):
            ai[j] -= bij

# 矩阵取负，将矩阵 a 中的每个元素取负
def ddm_ineg(a: list[list[R]]) -> None:
    """a <-- -a"""
    for ai in a:
        for j, aij in enumerate(ai):
            ai[j] = -aij

# 矩阵乘法，将矩阵 a 中的每个元素乘以标量 b
def ddm_imul(a: list[list[R]], b: R) -> None:
    """a <-- a*b"""
    for ai in a:
        for j, aij in enumerate(ai):
            ai[j] = aij * b

# 右乘标量，将矩阵 a 中的每个元素右乘标量 b
def ddm_irmul(a: list[list[R]], b: R) -> None:
    """a <-- b*a"""
    for ai in a:
        for j, aij in enumerate(ai):
            ai[j] = b * aij

# 矩阵乘法，将矩阵 a 加上矩阵 b 与矩阵 c 的乘积
def ddm_imatmul(
    a: list[list[R]], b: Sequence[Sequence[R]], c: Sequence[Sequence[R]]
) -> None:
    """a += b @ c"""
    # 计算 c 的转置
    cT = list(zip(*c))

    # 对每一行的元素进行累加乘积操作
    for bi, ai in zip(b, a):
        for j, cTj in enumerate(cT):
            ai[j] = sum(map(mul, bi, cTj), ai[j])
# 定义一个函数，计算矩阵的原地简化行阶梯形式（简化行阶梯形式是将矩阵变为行最简形的一种方法）。
def ddm_irref(a, _partial_pivot=False):
    """In-place reduced row echelon form of a matrix.

    Compute the reduced row echelon form of $a$. Modifies $a$ in place and
    returns a list of the pivot columns.

    Uses naive Gauss-Jordan elimination in the ground domain which must be a
    field.

    This routine is only really suitable for use with simple field domains like
    :ref:`GF(p)`, :ref:`QQ` and :ref:`QQ(a)` although even for :ref:`QQ` with
    larger matrices it is possibly more efficient to use fraction free
    approaches.

    This method is not suitable for use with rational function fields
    (:ref:`K(x)`) because the elements will blowup leading to costly gcd
    operations. In this case clearing denominators and using fraction free
    approaches is likely to be more efficient.

    For inexact numeric domains like :ref:`RR` and :ref:`CC` pass
    ``_partial_pivot=True`` to use partial pivoting to control rounding errors.

    Examples
    ========

    >>> from sympy.polys.matrices.dense import ddm_irref
    >>> from sympy import QQ
    >>> M = [[QQ(1), QQ(2), QQ(3)], [QQ(4), QQ(5), QQ(6)]]
    >>> pivots = ddm_irref(M)
    >>> M
    [[1, 0, -1], [0, 1, 2]]
    >>> pivots
    [0, 1]

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.rref
        Higher level interface to this routine.
    ddm_irref_den
        The fraction free version of this routine.
    sdm_irref
        A sparse version of this routine.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Row_echelon_form#Reduced_row_echelon_form

    # 在下面的代码部分，我们计算 aij 的逆，并在内部循环中使用乘法而不是除法。
    # 这里的域是一个字段，因此这两种操作都是定义良好的。然而，对于某些域来说性能上有显著差异。
    # 例如在 QQ 或 QQ(x) 中，求逆是免费的，但乘法和除法的成本是相同的，因此性能差异不大。
    # 而在 GF(p), QQ<sqrt(2)>, RR 或 CC 中，乘法比除法更快，因此重复使用预先计算的逆来进行多次乘法可以大大加快速度。
    # 在 QQ<a> 中，当 deg(minpoly(a)) 很大时，这样做的收益最大。

    # 对于像 QQ(x) 这样的域，这种方法可能表现不佳的原因有很多。
    # 通常初始矩阵具有简单的分母，而 fraction-free 方法使用 exquo (ddm_irref_den) 会在整个过程中保留该属性。
    # 而这里的方法会导致分母的增长，从而导致中间表达式中昂贵的 gcd 操作。
    # 对于像 QQ(x,y,z,...) 这样有多个生成元的情况，这种方法的性能非常糟糕。

    # TODO: 使用一种非平凡的枢轴策略来控制中间表达式的增长。
    # 重新排列行和/或列可以推迟最复杂的元素直到最后。
    # 如果第一个枢轴是一个复杂/大的元素，那么第一轮的简化将会
    # 立即引入整个矩阵的膨胀表达式。

    # a 是 (m x n) 的矩阵
    m = len(a)  # 获取矩阵 a 的行数 m
    if not m:
        return []  # 如果矩阵为空，则返回空列表
    n = len(a[0])  # 获取矩阵 a 的列数 n

    i = 0  # 初始化行索引 i
    pivots = []  # 存储主元列索引的列表

    for j in range(n):
        # 对于性能原因，应该在所有域中使用适当的主元素，但仅在 RR 和 CC（可能还有类似 RR(x) 的其他域）中是绝对必要的。
        # 如果域是 RR 或 CC，则 DDM.rref() 使用基于主元候选值绝对值的部分（行）主元素。这条路径由 DDM.rref() 使用。
        if _partial_pivot:
            # 在当前列 j 中找到绝对值最大的行 ip
            ip = max(range(i, m), key=lambda ip: abs(a[ip][j]))
            # 将第 ip 行和第 i 行交换
            a[i], a[ip] = a[ip], a[i]

        # 获取主元素值
        aij = a[i][j]

        # 处理零主元素
        if not aij:
            # 在当前列 j 中寻找非零主元素所在的行 ip
            for ip in range(i+1, m):
                aij = a[ip][j]
                # 如果找到非零主元素，则交换第 ip 行和第 i 行
                if aij:
                    a[i], a[ip] = a[ip], a[i]
                    break
            else:
                # 如果未找到非零主元素，则继续到下一列
                continue

        # 对第 i 行进行归一化处理
        ai = a[i]
        aijinv = aij**-1
        for l in range(j, n):
            ai[l] *= aijinv  # ai[j] 变为 1

        # 消除右侧上方和下方的元素
        for k, ak in enumerate(a):
            if k == i or not ak[j]:
                continue
            akj = ak[j]
            ak[j] -= akj  # ak[j] 变为 0
            for l in range(j+1, n):
                ak[l] -= akj * ai[l]

        # 处理下一行
        pivots.append(j)
        i += 1

        # 如果没有更多行了，则结束
        if i >= m:
            break

    return pivots  # 返回主元素列索引列表
    #
    # A simpler presentation of this algorithm is given in [1]:
    #
    # Given an n x n matrix A and n x 1 matrix b:
    #
    #   for i in range(n):
    #       if i != 0:
    #           d = a[i-1][i-1]
    #       for j in range(n):
    #           if j == i:
    #               continue
    #           b[j] = a[i][i]*b[j] - a[j][i]*b[i]
    #           for k in range(n):
    #               a[j][k] = a[i][i]*a[j][k] - a[j][i]*a[i][k]
    #               if i != 0:
    #                   a[j][k] /= d
    #
    # Our version here is a bit more complicated because:
    #
    #  1. We use row-swaps to avoid zero pivots.
    #  2. We allow for some columns to be missing pivots.
    #  3. We avoid a lot of redundant arithmetic.
    #
    # TODO: Use a non-trivial pivoting strategy. Even just row swapping makes a
    # big difference to performance if e.g. the upper-left entry of the matrix
    # is a huge polynomial.
    
    # a is (m x n)
    m = len(a)  # 获取矩阵a的行数m
    if not m:  # 如果矩阵a的行数为0，返回单位元素和空列表
        return K.one, []
    n = len(a[0])  # 获取矩阵a的列数n

    d = None  # 初始化d为None，用于存储主元
    pivots = []  # 初始化pivots列表，用于存储主元列的索引
    no_pivots = []  # 初始化no_pivots列表，用于存储没有主元的列的索引
    # i, j will be the row and column indices of the current pivot
    i = 0
    # Iterate over each column index j from 0 to n-1
    for j in range(n):
        # next pivot?
        # Get the element at the current pivot position (i, j)
        aij = a[i][j]

        # swap rows if zero
        # If the pivot element is zero, find a non-zero element in the same column below and swap rows
        if not aij:
            for ip in range(i+1, m):
                aij = a[ip][j]
                # row-swap
                # If a non-zero element is found in column j below the current row i, swap rows i and ip
                if aij:
                    a[i], a[ip] = a[ip], a[i]
                    break
            else:
                # go to next column
                # If no non-zero element is found in column j below row i, mark column j as having no pivot
                no_pivots.append(j)
                continue

        # Now aij is the pivot and i,j are the row and column. We need to clear
        # the column above and below but we also need to keep track of the
        # denominator of the RREF which means also multiplying everything above
        # and to the left by the current pivot aij and dividing by d (which we
        # multiplied everything by in the previous iteration so this is an
        # exact division).
        #
        # First handle the upper left corner which is usually already diagonal
        # with all diagonal entries equal to the current denominator but there
        # can be other non-zero entries in any column that has no pivot.

        # Update previous pivots in the matrix
        # If there are previous pivots, update the entries in those columns
        if pivots:
            pivot_val = aij * a[0][pivots[0]]
            # Divide out the common factor
            if d is not None:
                pivot_val = K.exquo(pivot_val, d)

            # Update the values in rows corresponding to previous pivots
            for ip, jp in enumerate(pivots):
                a[ip][jp] = pivot_val

        # Update columns without pivots
        # Update entries in columns that do not have a pivot
        for jnp in no_pivots:
            for ip in range(i):
                aijp = a[ip][jnp]
                if aijp:
                    aijp *= aij
                    if d is not None:
                        aijp = K.exquo(aijp, d)
                    a[ip][jnp] = aijp

        # Eliminate above, below and to the right as in ordinary division free
        # Gauss-Jordan elimination except also dividing out d from every entry.

        # Iterate over all rows jp and their corresponding row entries aj in matrix a
        for jp, aj in enumerate(a):

            # Skip the current row
            # Skip the row that is the current pivot row i
            if jp == i:
                continue

            # Eliminate to the right in all rows
            # For each row jp ≠ i, eliminate entries to the right of column j
            for kp in range(j+1, n):
                ajk = aij * aj[kp] - aj[j] * a[i][kp]
                if d is not None:
                    ajk = K.exquo(ajk, d)
                aj[kp] = ajk

            # Set to zero above and below the pivot
            # Set the entries above and below the pivot column j in row jp to zero
            aj[j] = K.zero

        # next row
        # Mark column j as a pivot column
        pivots.append(j)
        # Move to the next row i
        i += 1

        # no more rows left?
        # If there are no more rows left to process, exit the loop
        if i >= m:
            break

        # Update the current denominator d if the current pivot aij is not 1
        if not K.is_one(aij):
            d = aij
        else:
            d = None

    # Determine the final denominator denom
    if not pivots:
        denom = K.one
    else:
        denom = a[0][pivots[0]]

    # Return the final denominator and list of pivot columns pivots
    return denom, pivots
# 定义函数 ddm_iinv，计算矩阵 a 在域 K 上的逆矩阵并存储在 ainv 中
def ddm_iinv(ainv, a, K):
    """ainv  <--  inv(a)

    Compute the inverse of a matrix $a$ over a field $K$ using Gauss-Jordan
    elimination. The result is stored in $ainv$.

    使用高斯-约旦消元法计算矩阵 $a$ 在域 $K$ 上的逆矩阵。结果存储在 $ainv$ 中。

    Uses division in the ground domain which should be an exact field.
    使用域 $K$ 中的除法，该域应为精确域。

    Examples
    ========

    >>> from sympy.polys.matrices.ddm import ddm_iinv, ddm_imatmul
    >>> from sympy import QQ
    >>> a = [[QQ(1), QQ(2)], [QQ(3), QQ(4)]]
    >>> ainv = [[None, None], [None, None]]
    >>> ddm_iinv(ainv, a, QQ)
    >>> ainv
    [[-2, 1], [3/2, -1/2]]
    >>> result = [[QQ(0), QQ(0)], [QQ(0), QQ(0)]]
    >>> ddm_imatmul(result, a, ainv)
    >>> result
    [[1, 0], [0, 1]]

    See Also
    ========

    ddm_irref: the underlying routine.
    """
    # 如果 K 不是一个域，则抛出异常
    if not K.is_Field:
        raise DMDomainError('Not a field')

    # a 的行数 m
    m = len(a)
    if not m:
        return
    # a 的列数 n
    n = len(a[0])
    # 如果矩阵 a 不是方阵，则抛出异常
    if m != n:
        raise DMNonSquareMatrixError

    # 创建单位矩阵 eye，用于后续高斯-约旦消元
    eye = [[K.one if i==j else K.zero for j in range(n)] for i in range(n)]
    # 构造增广矩阵 Aaug，将单位矩阵 eye 附加到矩阵 a 的右侧
    Aaug = [row + eyerow for row, eyerow in zip(a, eye)]
    # 使用 ddm_irref 函数计算增广矩阵 Aaug 的主元位置列表
    pivots = ddm_irref(Aaug)
    # 如果主元位置列表不等于 [0, 1, ..., n-1]，则抛出非可逆矩阵错误异常
    if pivots != list(range(n)):
        raise DMNonInvertibleMatrixError('Matrix det == 0; not invertible.')
    # 将增广矩阵 Aaug 的右侧部分（表示矩阵的逆）复制给 ainv 列表
    ainv[:] = [row[n:] for row in Aaug]
# 根据 LU 分解求解线性方程组 L*U*x = swaps(b)，返回解 x
def ddm_ilu_solve(x, L, U, swaps, b):
    # 初始化 x 为 b 的副本
    x[:] = b[:]
    m = len(U)
    # 如果 U 矩阵为空，直接返回
    if not m:
        return
    n = len(U[0])
    
    # 根据 LU 分解的置换 swaps 重排 b 的顺序
    for i, ip in reversed(swaps):
        x[i], x[ip] = x[ip], x[i]
    
    # 回代求解 U*x = y
    for i in range(n - 1, -1, -1):
        x[i] /= U[i][i]
        for j in range(i):
            x[j] -= x[i] * U[j][i]
    
    # 前代求解 L*y = x
    for i in range(m):
        for j in range(i):
            x[i] -= x[j] * L[i][j]
    
    return x
    # 计算矩阵 U 的行数
    m = len(U)
    if not m:
        return
    # 计算矩阵 U 的列数
    n = len(U[0])

    # 检查向量 b 的行数
    m2 = len(b)
    if not m2:
        raise DMShapeError("Shape mismtch")
    # 检查向量 b 的列数
    o = len(b[0])

    # 检查矩阵 U 和向量 b 的形状是否匹配
    if m != m2:
        raise DMShapeError("Shape mismtch")
    # 检查是否为欠定系统
    if m < n:
        raise NotImplementedError("Underdetermined")

    # 如果存在置换，对向量 b 进行置换
    if swaps:
        # 深拷贝向量 b
        b = [row[:] for row in b]
        # 根据置换对向量 b 进行交换
        for i1, i2 in swaps:
            b[i1], b[i2] = b[i2], b[i1]

    # 解 Ly = b，计算向量 y
    y = [[None] * o for _ in range(m)]
    for k in range(o):
        for i in range(m):
            rhs = b[i][k]
            for j in range(i):
                rhs -= L[i][j] * y[j][k]
            y[i][k] = rhs

    # 如果 m > n，检查矩阵 L 是否为奇异矩阵
    if m > n:
        for i in range(n, m):
            for j in range(o):
                if y[i][j]:
                    raise DMNonInvertibleMatrixError

    # 解 Ux = y，计算向量 x
    for k in range(o):
        for i in reversed(range(n)):
            # 如果对角元素为零，则矩阵 U 为奇异矩阵
            if not U[i][i]:
                raise DMNonInvertibleMatrixError
            rhs = y[i][k]
            for j in range(i+1, n):
                rhs -= U[i][j] * x[j][k]
            x[i][k] = rhs / U[i][i]
# 使用 Berkowitz 算法计算矩阵的特征多项式

def ddm_berk(M, K):
    """
    Berkowitz algorithm for computing the characteristic polynomial.

    Explanation
    ===========

    The Berkowitz algorithm is a division-free algorithm for computing the
    characteristic polynomial of a matrix over any commutative ring using only
    arithmetic in the coefficient ring.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.polys.matrices.dense import ddm_berk
    >>> from sympy.polys.domains import ZZ
    >>> M = [[ZZ(1), ZZ(2)], [ZZ(3), ZZ(4)]]
    >>> ddm_berk(M, ZZ)
    [[1], [-5], [-2]]
    >>> Matrix(M).charpoly()
    PurePoly(lambda**2 - 5*lambda - 2, lambda, domain='ZZ')

    See Also
    ========

    sympy.polys.matrices.domainmatrix.DomainMatrix.charpoly
        The high-level interface to this function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Samuelson%E2%80%93Berkowitz_algorithm
    """
    m = len(M)  # 获取矩阵 M 的行数
    if not m:
        return [[K.one]]  # 如果矩阵 M 为空，返回一个包含 K.one 的列表

    n = len(M[0])  # 获取矩阵 M 的列数

    if m != n:
        raise DMShapeError("Not square")  # 如果矩阵 M 不是方阵，抛出错误

    if n == 1:
        return [[K.one], [-M[0][0]]]  # 如果矩阵 M 是 1x1 的，返回对应的特征多项式系数列表

    a = M[0][0]  # 获取矩阵 M 的左上角元素
    R = [M[0][1:]]  # 获取矩阵 M 第一行除去第一个元素的部分作为 R
    C = [[row[0]] for row in M[1:]]  # 获取矩阵 M 第一列除去第一个元素的部分作为 C
    A = [row[1:] for row in M[1:]]  # 获取矩阵 M 去掉第一行和第一列后的子矩阵作为 A

    q = ddm_berk(A, K)  # 递归调用 Berkowitz 算法计算子矩阵 A 的特征多项式

    T = [[K.zero] * n for _ in range(n + 1)]  # 初始化一个 (n+1) x n 的零矩阵
    for i in range(n):
        T[i][i] = K.one
        T[i + 1][i] = -a  # 填充 T 矩阵的第一列和第二列

    for i in range(2, n + 1):
        if i == 2:
            AnC = C
        else:
            C = AnC
            AnC = [[K.zero] for row in C]
            ddm_imatmul(AnC, A, C)  # 使用函数 ddm_imatmul 计算 AnC = A * C

        RAnC = [[K.zero]]
        ddm_imatmul(RAnC, R, AnC)  # 使用函数 ddm_imatmul 计算 RAnC = R * AnC

        for j in range(0, n + 1 - i):
            T[i + j][j] = -RAnC[0][0]  # 填充 T 矩阵的剩余部分

    qout = [[K.zero] for _ in range(n + 1)]  # 初始化一个 (n+1) x 1 的零矩阵
    ddm_imatmul(qout, T, q)  # 使用函数 ddm_imatmul 计算 qout = T * q

    return qout
```