# `D:\src\scipysrc\sympy\sympy\matrices\determinant.py`

```
# 导入所需模块和函数类型
from types import FunctionType

# 导入缓存相关的装饰器
from sympy.core.cache import cacheit

# 导入数值类型相关模块
from sympy.core.numbers import Float, Integer

# 导入单例对象
from sympy.core.singleton import S

# 导入用于生成唯一命名符号的函数
from sympy.core.symbol import uniquely_named_symbol

# 导入乘法相关的模块
from sympy.core.mul import Mul

# 导入多项式相关的模块和函数
from sympy.polys import PurePoly, cancel

# 导入组合数学函数
from sympy.functions.combinatorial.numbers import nC

# 导入多项式矩阵相关模块
from sympy.polys.matrices.domainmatrix import DomainMatrix

# 导入行列式分解矩阵相关模块
from sympy.polys.matrices.ddm import DDM

# 导入自定义异常类
from .exceptions import NonSquareMatrixError

# 导入自定义工具函数模块
from .utilities import (
    _get_intermediate_simp, _get_intermediate_simp_bool,
    _iszero, _is_zero_after_expand_mul, _dotprodsimp, _simplify
)


def _find_reasonable_pivot(col, iszerofunc=_iszero, simpfunc=_simplify):
    """ Find the lowest index of an item in ``col`` that is
    suitable for a pivot.  If ``col`` consists only of
    Floats, the pivot with the largest norm is returned.
    Otherwise, the first element where ``iszerofunc`` returns
    False is used.  If ``iszerofunc`` does not return false,
    items are simplified and retested until a suitable
    pivot is found.

    Returns a 4-tuple
        (pivot_offset, pivot_val, assumed_nonzero, newly_determined)
    where pivot_offset is the index of the pivot, pivot_val is
    the (possibly simplified) value of the pivot, assumed_nonzero
    is True if an assumption that the pivot was non-zero
    was made without being proved, and newly_determined are
    elements that were simplified during the process of pivot
    finding."""

    # 初始化新确定的元素列表
    newly_determined = []
    
    # 将列转换为列表形式
    col = list(col)
    
    # 如果列中所有元素都是 Float 或 Integer 类型，并且至少有一个是 Float
    if all(isinstance(x, (Float, Integer)) for x in col) and any(
            isinstance(x, Float) for x in col):
        # 计算列中每个元素的绝对值
        col_abs = [abs(x) for x in col]
        # 找到绝对值最大的元素
        max_value = max(col_abs)
        # 如果 iszerofunc 判断最大值为零
        if iszerofunc(max_value):
            # 如果最大值不为零，将所有非零元素置为零并添加到新确定的元素列表中
            if max_value != 0:
                newly_determined = [(i, 0) for i, x in enumerate(col) if x != 0]
            # 返回空作为索引和值，False 作为假设非零标志，并返回新确定的元素列表
            return (None, None, False, newly_determined)
        # 找到最大值在列中的索引并返回索引、该元素的值、假设非零标志 False，以及新确定的元素列表
        index = col_abs.index(max_value)
        return (index, col[index], False, newly_determined)

    # PASS 1 (直接使用 iszerofunc)
    possible_zeros = []
    for i, x in enumerate(col):
        # 调用 iszerofunc 判断当前元素是否为零
        is_zero = iszerofunc(x)
        # 如果找到了非零元素，则返回其索引、值、假设非零标志 False，以及新确定的元素列表
        if is_zero == False:
            return (i, x, False, newly_determined)
        # 将 iszerofunc 的结果添加到可能的零列表中
        possible_zeros.append(is_zero)

    # 如果没有找到确定非零元素，返回空作为索引和值，False 作为假设非零标志，以及新确定的元素列表
    # 如果所有可能的零都是真的，我们没有主元
    if all(possible_zeros):
        # 如果所有的元素都肯定是零，那么没有主元
        return (None, None, False, newly_determined)

    # PASS 2 (iszerofunc 在简化之后)
    # 我们还没有找到肯定不是零的元素，所以
    # 遍历 iszerofunc 无法判断的元素，并尝试
    # 简化以查看是否找到了一些信息
    for i, x in enumerate(col):
        if possible_zeros[i] is not None:
            continue
        # 对 x 进行简化
        simped = simpfunc(x)
        # 判断简化后的结果是否为零
        is_zero = iszerofunc(simped)
        # 如果 is_zero 是 True 或者 False，则表示新发现了一些信息
        if is_zero in (True, False):
            newly_determined.append((i, simped))
        # 如果发现了一个肯定不是零的元素，则返回结果
        if is_zero == False:
            return (i, simped, False, newly_determined)
        possible_zeros[i] = is_zero

    # 简化之后，一些被识别为零的元素可能确实是零
    if all(possible_zeros):
        # 如果所有的元素都肯定是零，那么没有主元
        return (None, None, False, newly_determined)

    # PASS 3 (.equals(0))
    # 一些表达式无法简化为零，但 ``.equals(0)`` 为 True。
    # 作为最后的尝试，对这些表达式应用 ``.equals`` 方法
    for i, x in enumerate(col):
        if possible_zeros[i] is not None:
            continue
        # 检查 x 是否等于零
        if x.equals(S.Zero):
            # 当 ``.equals(0)`` 返回 True 时，将其视为已证明为零
            possible_zeros[i] = True
            newly_determined.append((i, S.Zero))

    if all(possible_zeros):
        # 如果所有的元素都肯定是零，那么没有主元
        return (None, None, False, newly_determined)

    # 在这一点上，没有任何元素可以肯定是主元。
    # 为了保持与现有行为的兼容性，我们假设未确定的元素不是零。
    # 在这种情况下，我们可能应该发出一个警告。
    i = possible_zeros.index(None)
    return (i, col[i], True, newly_determined)
def _find_reasonable_pivot_naive(col, iszerofunc=_iszero, simpfunc=None):
    """
    Helper that computes the pivot value and location from a
    sequence of contiguous matrix column elements. As a side effect
    of the pivot search, this function may simplify some of the elements
    of the input column. A list of these simplified entries and their
    indices are also returned.
    This function mimics the behavior of _find_reasonable_pivot(),
    but does less work trying to determine if an indeterminate candidate
    pivot simplifies to zero. This more naive approach can be much faster,
    with the trade-off that it may erroneously return a pivot that is zero.

    ``col`` is a sequence of contiguous column entries to be searched for
    a suitable pivot.
    ``iszerofunc`` is a callable that returns a Boolean that indicates
    if its input is zero, or None if no such determination can be made.
    ``simpfunc`` is a callable that simplifies its input. It must return
    its input if it does not simplify its input. Passing in
    ``simpfunc=None`` indicates that the pivot search should not attempt
    to simplify any candidate pivots.

    Returns a 4-tuple:
    (pivot_offset, pivot_val, assumed_nonzero, newly_determined)
    ``pivot_offset`` is the sequence index of the pivot.
    ``pivot_val`` is the value of the pivot.
    pivot_val and col[pivot_index] are equivalent, but will be different
    when col[pivot_index] was simplified during the pivot search.
    ``assumed_nonzero`` is a boolean indicating if the pivot cannot be
    guaranteed to be zero. If assumed_nonzero is true, then the pivot
    may or may not be non-zero. If assumed_nonzero is false, then
    the pivot is non-zero.
    ``newly_determined`` is a list of index-value pairs of pivot candidates
    that were simplified during the pivot search.
    """

    # indeterminates holds the index-value pairs of each pivot candidate
    # that is neither zero or non-zero, as determined by iszerofunc().
    # If iszerofunc() indicates that a candidate pivot is guaranteed
    # non-zero, or that every candidate pivot is zero then the contents
    # of indeterminates are unused.
    # Otherwise, the only viable candidate pivots are symbolic.
    # In this case, indeterminates will have at least one entry,
    # and all but the first entry are ignored when simpfunc is None.
    indeterminates = []
    for i, col_val in enumerate(col):
        col_val_is_zero = iszerofunc(col_val)
        if col_val_is_zero == False:
            # This pivot candidate is non-zero.
            return i, col_val, False, []
        elif col_val_is_zero is None:
            # The candidate pivot's comparison with zero
            # is indeterminate.
            indeterminates.append((i, col_val))

    if len(indeterminates) == 0:
        # All candidate pivots are guaranteed to be zero, i.e. there is
        # no pivot.
        return None, None, False, []
    # 如果未传入简化函数（simpfunc），则假设第一个不定候选枢轴非零。
    if simpfunc is None:
        # 调用者未传入一个简化函数，这个函数可能确定一个不定候选枢轴是否非零，
        # 因此假设第一个不定候选枢轴是非零的。
        return indeterminates[0][0], indeterminates[0][1], True, []

    # newly_determined 用于存储在寻找非零枢轴期间简化的候选枢轴的索引-值对。
    newly_determined = []
    # 遍历不定候选枢轴列表
    for i, col_val in indeterminates:
        # 对当前候选枢轴值 col_val 进行简化
        tmp_col_val = simpfunc(col_val)
        # 如果简化函数 simpfunc 返回的对象与原始对象不同（即简化有效）
        if id(col_val) != id(tmp_col_val):
            # simpfunc() 简化了这个候选枢轴。
            newly_determined.append((i, tmp_col_val))
            # 如果简化后的值 tmp_col_val 不为零
            if iszerofunc(tmp_col_val) == False:
                # 候选枢轴简化为保证非零值。
                return i, tmp_col_val, False, newly_determined

    # 如果没有找到非零的候选枢轴，则返回第一个不定候选枢轴的索引和值，并标记为真和新简化的候选枢轴。
    return indeterminates[0][0], indeterminates[0][1], True, newly_determined
# This functions is a candidate for caching if it gets implemented for matrices.
def _berkowitz_toeplitz_matrix(M):
    """Return (A,T) where T the Toeplitz matrix used in the Berkowitz algorithm
    corresponding to ``M`` and A is the first principal submatrix.
    """

    # the 0 x 0 case is trivial
    if M.rows == 0 and M.cols == 0:
        return M._new(1,1, [M.one])

    # Partition M = [ a_11  R ]
    #              [  C    A ]
    #
    # Splitting the matrix M into components for the Berkowitz algorithm
    a, R = M[0,0],   M[0, 1:]  # Extracting first row elements a_11 and R
    C, A = M[1:, 0], M[1:,1:]  # Extracting first column elements C and A

    # The Toeplitz matrix construction
    #
    # Constructing the Toeplitz matrix T used in Berkowitz algorithm
    # [ 1                                     ]
    # [ -a         1                          ]
    # [ -RC       -a        1                 ]
    # [ -RAC     -RC       -a       1         ]
    # [ -RA**2C -RAC      -RC      -a       1 ]
    # etc.

    # Compute the diagonal entries of the Toeplitz matrix
    diags = [C]
    for i in range(M.rows - 2):
        diags.append(A.multiply(diags[i], dotprodsimp=None))  # Recursive computation of -R * A**n * C
    diags = [(-R).multiply(d, dotprodsimp=None)[0, 0] for d in diags]  # Final adjustment of diagonal elements
    diags = [M.one, -a] + diags  # Prepend 1 and -a to complete the diagonal entries

    # Define a function to retrieve entries of the Toeplitz matrix
    def entry(i,j):
        if j > i:
            return M.zero
        return diags[i - j]

    # Create the Toeplitz matrix using the computed entries
    toeplitz = M._new(M.cols + 1, M.rows, entry)
    return (A, toeplitz)



# This functions is a candidate for caching if it gets implemented for matrices.
def _berkowitz_vector(M):
    """ Run the Berkowitz algorithm and return a vector whose entries
        are the coefficients of the characteristic polynomial of ``M``.

        Given N x N matrix, efficiently compute
        coefficients of characteristic polynomials of ``M``
        without division in the ground domain.

        This method is particularly useful for computing determinant,
        principal minors and characteristic polynomial when ``M``
        has complicated coefficients e.g. polynomials. Semi-direct
        usage of this algorithm is also important in computing
        efficiently sub-resultant PRS.

        Assuming that M is a square matrix of dimension N x N and
        I is N x N identity matrix, then the Berkowitz vector is
        an N x 1 vector whose entries are coefficients of the
        polynomial

                        charpoly(M) = det(t*I - M)

        As a consequence, all polynomials generated by Berkowitz
        algorithm are monic.

        For more information on the implemented algorithm refer to:

        [1] S.J. Berkowitz, On computing the determinant in small
            parallel time using a small number of processors, ACM,
            Information Processing Letters 18, 1984, pp. 147-150

        [2] M. Keber, Division-Free computation of sub-resultants
            using Bezout matrices, Tech. Report MPI-I-2006-1-006,
            Saarbrucken, 2006
    """

    # handle the trivial cases
    # This function is currently incomplete and lacks further implementation.
    # 如果矩阵 M 的行数和列数都为 0
    if M.rows == 0 and M.cols == 0:
        # 返回一个新的矩阵，大小为 1x1，内容为 [M.one]
        return M._new(1, 1, [M.one])
    
    # 如果矩阵 M 的行数和列数都为 1
    elif M.rows == 1 and M.cols == 1:
        # 返回一个新的矩阵，大小为 2x1，内容为 [M.one, -M[0,0]]
        return M._new(2, 1, [M.one, -M[0,0]])

    # 调用 _berkowitz_toeplitz_matrix 函数，获取子矩阵 submat 和 Toeplitz 矩阵 toeplitz
    submat, toeplitz = _berkowitz_toeplitz_matrix(M)

    # 返回 Toeplitz 矩阵 toeplitz 与 _berkowitz_vector(submat) 的乘积
    return toeplitz.multiply(_berkowitz_vector(submat), dotprodsimp=None)
# 计算矩阵的伴随矩阵，即其代数余子式矩阵的转置

def _adjugate(M, method="berkowitz"):
    """Returns the adjugate, or classical adjoint, of
    a matrix.  That is, the transpose of the matrix of cofactors.

    https://en.wikipedia.org/wiki/Adjugate

    Parameters
    ==========

    method : string, optional
        Method to use to find the cofactors, can be "bareiss", "berkowitz",
        "bird", "laplace" or "lu".

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> M.adjugate()
    Matrix([
    [ 4, -2],
    [-3,  1]])

    See Also
    ========

    cofactor_matrix
    sympy.matrices.matrixbase.MatrixBase.transpose
    """

    # 调用 Matrix 对象的 cofactor_matrix 方法计算代数余子式矩阵，并返回其转置
    return M.cofactor_matrix(method=method).transpose()


# This functions is a candidate for caching if it gets implemented for matrices.
def _charpoly(M, x='lambda', simplify=_simplify):
    """Computes characteristic polynomial det(x*I - M) where I is
    the identity matrix.

    A PurePoly is returned, so using different variables for ``x`` does
    not affect the comparison or the polynomials:

    Parameters
    ==========

    x : string, optional
        Name for the "lambda" variable, defaults to "lambda".

    simplify : function, optional
        Simplification function to use on the characteristic polynomial
        calculated. Defaults to ``simplify``.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x, y
    >>> M = Matrix([[1, 3], [2, 0]])
    >>> M.charpoly()
    PurePoly(lambda**2 - lambda - 6, lambda, domain='ZZ')
    >>> M.charpoly(x) == M.charpoly(y)
    True
    >>> M.charpoly(x) == M.charpoly(y)
    True

    Specifying ``x`` is optional; a symbol named ``lambda`` is used by
    default (which looks good when pretty-printed in unicode):

    >>> M.charpoly().as_expr()
    lambda**2 - lambda - 6

    And if ``x`` clashes with an existing symbol, underscores will
    be prepended to the name to make it unique:

    >>> M = Matrix([[1, 2], [x, 0]])
    >>> M.charpoly(x).as_expr()
    _x**2 - _x - 2*x

    Whether you pass a symbol or not, the generator can be obtained
    with the gen attribute since it may not be the same as the symbol
    that was passed:

    >>> M.charpoly(x).gen
    _x
    >>> M.charpoly(x).gen == x
    False

    Notes
    =====

    The Samuelson-Berkowitz algorithm is used to compute
    the characteristic polynomial efficiently and without any
    division operations.  Thus the characteristic polynomial over any
    commutative ring without zero divisors can be computed.

    If the determinant det(x*I - M) can be found out easily as
    in the case of an upper or a lower triangular matrix, then
    instead of Samuelson-Berkowitz algorithm, eigenvalues are computed
    and the characteristic polynomial with their help.

    See Also
    ========

    det
    """

    if not M.is_square:
        raise NonSquareMatrixError()

    # Use DomainMatrix. We are already going to convert this to a Poly so there
    # 这里应该有进一步的代码，但是未提供完整
    # 将矩阵 M 转换为具有扩展域的矩阵 dM，确保不涉及除法或零检测问题，因此使用 EXRAW 可行。
    #
    # 如果 M.to_DM() 需要时会回退到 EXRAW 而不是 EX。EXRAW 在基本算术上更快，因为它不会为每个操作调用 cancel，
    # 但生成的未简化结果在后续的 simplify 调用中会变慢。总体上使用 EX 更快，但在某些情况下，EXRAW+simplify 可以得到更简单的结果，
    # 因此我们暂时保留 charpoly 的现有行为...

    # 将矩阵 M 转换为扩展域矩阵 dM
    dM = M.to_DM()

    # 获取域 K（通常是 EXRAW 或其他自定义的域）
    K = dM.domain

    # 计算 dM 的特征多项式
    cp = dM.charpoly()

    # 为符号 x 分配唯一的命名，用于多项式表示
    x = uniquely_named_symbol(x, [M], modify=lambda s: '_' + s)

    # 根据域的类型选择如何构造多项式 p
    if K.is_EXRAW or simplify is not _simplify:
        # XXX: 将域中的系数转换为 SymPy 表达式是昂贵的。仅当调用者提供了自定义的 simplify 函数以保持向后兼容性，
        # 或者域为 EX 时才这样做。对于其他任何域，在这个阶段进行简化都没有好处，因为 Poly 将一切都放入规范形式中。
        
        # 将域中的系数转换为 SymPy 表达式
        berk_vector = [K.to_sympy(c) for c in cp]

        # 对系数进行简化
        berk_vector = [simplify(a) for a in berk_vector]

        # 使用简化后的系数构造纯多项式
        p = PurePoly(berk_vector, x)

    else:
        # 直接从域元素列表构造多项式
        p = PurePoly(cp, x, domain=K)

    # 返回构造的多项式 p
    return p
# 计算给定矩阵 M 中指定元素的余子式（cofactor）。
# 如果矩阵 M 不是方阵或者行数小于 1，则引发非方阵矩阵错误。
def _cofactor(M, i, j, method="berkowitz"):
    if not M.is_square or M.rows < 1:
        raise NonSquareMatrixError()

    # 返回 (-1)^(i + j) * M.minor(i, j, method)，即指定元素的余子式
    return S.NegativeOne**((i + j) % 2) * M.minor(i, j, method)


# 返回一个矩阵，其元素为给定矩阵 M 中每个元素的余子式。
# 如果矩阵 M 不是方阵，则引发非方阵矩阵错误。
def _cofactor_matrix(M, method="berkowitz"):
    if not M.is_square:
        raise NonSquareMatrixError()

    # 使用 lambda 函数生成一个新的矩阵，每个元素是通过指定方法计算的余子式
    return M._new(M.rows, M.cols,
            lambda i, j: M.cofactor(i, j, method))


# 计算给定矩阵 M 的永久（permanent），适用于方阵和非方阵。
# 对于 m x n 矩阵（其中 m <= n），永久定义为在 [1, 2, ..., n] 上大小不超过 m 的排列 s 的总和，
# 每个排列 s 的乘积从 i = 1 到 m 的 M[i, s[i]]。
# 如果矩阵 M 不是方阵，则先对其转置以确保满足 m <= n 的条件。
def _per(M):
    import itertools

    m, n = M.shape
    if m > n:
        M = M.T
        m, n = n, m
    s = list(range(n))

    subsets = []
    for i in range(1, m + 1):
        subsets += list(map(list, itertools.combinations(s, i)))

    perm = 0
    # 对于每个子集进行迭代计算排列式的每一项
    for subset in subsets:
        # 初始化乘积为1
        prod = 1
        # 获取当前子集的长度
        sub_len = len(subset)
        # 对于矩阵的每一行进行迭代
        for i in range(m):
             # 计算当前子集中列索引的元素之和，并乘到乘积中
             prod *= sum(M[i, j] for j in subset)
        # 计算当前子集的贡献，包括乘积、符号和组合数
        perm += prod * S.NegativeOne**sub_len * nC(n - sub_len, m - sub_len)
    # 最终结果乘以 (-1)^m，简化结果并返回
    perm *= S.NegativeOne**m
    return perm.simplify()
def _det_DOM(M):
    # 使用 DomainMatrix 类从给定的 Matrix 对象 M 中创建一个域矩阵 DOM
    DOM = DomainMatrix.from_Matrix(M, field=True, extension=True)
    # 获取 DOM 的定义域 K
    K = DOM.domain
    # 将域 K 转换为 SymPy 中的表示，并计算 DOM 的行列式
    return K.to_sympy(DOM.det())

# This functions is a candidate for caching if it gets implemented for matrices.
def _det(M, method="bareiss", iszerofunc=None):
    """Computes the determinant of a matrix if ``M`` is a concrete matrix object
    otherwise return an expressions ``Determinant(M)`` if ``M`` is a
    ``MatrixSymbol`` or other expression.

    Parameters
    ==========

    method : string, optional
        Specifies the algorithm used for computing the matrix determinant.

        If the matrix is at most 3x3, a hard-coded formula is used and the
        specified method is ignored. Otherwise, it defaults to
        ``'bareiss'``.

        Also, if the matrix is an upper or a lower triangular matrix, determinant
        is computed by simple multiplication of diagonal elements, and the
        specified method is ignored.

        If it is set to ``'domain-ge'``, then Gaussian elimination method will
        be used via using DomainMatrix.

        If it is set to ``'bareiss'``, Bareiss' fraction-free algorithm will
        be used.

        If it is set to ``'berkowitz'``, Berkowitz' algorithm will be used.

        If it is set to ``'bird'``, Bird's algorithm will be used [1]_.

        If it is set to ``'laplace'``, Laplace's algorithm will be used.

        Otherwise, if it is set to ``'lu'``, LU decomposition will be used.

        .. note::
            For backward compatibility, legacy keys like "bareis" and
            "det_lu" can still be used to indicate the corresponding
            methods.
            And the keys are also case-insensitive for now. However, it is
            suggested to use the precise keys for specifying the method.

    iszerofunc : FunctionType or None, optional
        If it is set to ``None``, it will be defaulted to ``_iszero`` if the
        method is set to ``'bareiss'``, and ``_is_zero_after_expand_mul`` if
        the method is set to ``'lu'``.

        It can also accept any user-specified zero testing function, if it
        is formatted as a function which accepts a single symbolic argument
        and returns ``True`` if it is tested as zero and ``False`` if it
        tested as non-zero, and also ``None`` if it is undecidable.

    Returns
    =======

    det : Basic
        Result of determinant.

    Raises
    ======

    ValueError
        If unrecognized keys are given for ``method`` or ``iszerofunc``.

    NonSquareMatrixError
        If attempted to calculate determinant from a non-square matrix.

    Examples
    ========

    >>> from sympy import Matrix, eye, det
    >>> I3 = eye(3)
    >>> det(I3)
    1
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> det(M)
    -2
    >>> det(M) == M.det()
    True
    >>> M.det(method="domain-ge")
    -2

    References
    ==========

    """
    """
    [1] Bird, R. S. (2011). A simple division-free algorithm for computing
           determinants. Inf. Process. Lett., 111(21), 1072-1074. doi:
           10.1016/j.ipl.2011.08.006
    """

    # 将 `method` 参数转换为小写
    method = method.lower()

    # 如果 `method` 为 "bareis"，则更正为 "bareiss"
    if method == "bareis":
        method = "bareiss"
    # 如果 `method` 为 "det_lu"，则更正为 "lu"
    elif method == "det_lu":
        method = "lu"

    # 检查 `method` 是否为预定义的方法之一，若不是则抛出异常
    if method not in ("bareiss", "berkowitz", "lu", "domain-ge", "bird",
                      "laplace"):
        raise ValueError("Determinant method '%s' unrecognized" % method)

    # 如果未提供 `iszerofunc` 函数，根据 `method` 分配默认的零判定函数
    if iszerofunc is None:
        if method == "bareiss":
            iszerofunc = _is_zero_after_expand_mul
        elif method == "lu":
            iszerofunc = _iszero

    # 如果提供的 `iszerofunc` 不是函数类型，则抛出异常
    elif not isinstance(iszerofunc, FunctionType):
        raise ValueError("Zero testing method '%s' unrecognized" % iszerofunc)

    # 获取矩阵 `M` 的行数
    n = M.rows

    # 如果矩阵是方阵，则执行相应的快速计算
    if n == M.cols: # 方阵检查由各个方法函数单独执行
        if n == 0:
            return M.one
        elif n == 1:
            return M[0, 0]
        elif n == 2:
            # 对于2阶方阵，直接计算行列式的值
            m = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            return _get_intermediate_simp(_dotprodsimp)(m)
        elif n == 3:
            # 对于3阶方阵，应用展开法计算行列式的值
            m =  (M[0, 0] * M[1, 1] * M[2, 2]
                + M[0, 1] * M[1, 2] * M[2, 0]
                + M[0, 2] * M[1, 0] * M[2, 1]
                - M[0, 2] * M[1, 1] * M[2, 0]
                - M[0, 0] * M[1, 2] * M[2, 1]
                - M[0, 1] * M[1, 0] * M[2, 2])
            return _get_intermediate_simp(_dotprodsimp)(m)

    # 存储每个强连通分量的行列式值
    dets = []
    for b in M.strongly_connected_components():
        # 根据 `method` 调用相应的行列式计算方法
        if method == "domain-ge": # 使用 DomainMatrix 计算行列式
            det = _det_DOM(M[b, b])
        elif method == "bareiss":
            det = M[b, b]._eval_det_bareiss(iszerofunc=iszerofunc)
        elif method == "berkowitz":
            det = M[b, b]._eval_det_berkowitz()
        elif method == "lu":
            det = M[b, b]._eval_det_lu(iszerofunc=iszerofunc)
        elif method == "bird":
            det = M[b, b]._eval_det_bird()
        elif method == "laplace":
            det = M[b, b]._eval_det_laplace()
        dets.append(det)
    
    # 返回所有强连通分量行列式的乘积作为最终行列式的值
    return Mul(*dets)
# This function calculates the determinant of a matrix using Bareiss' fraction-free
# algorithm, an extension of Gaussian elimination. It is particularly suitable for dense
# symbolic matrices, minimizing the occurrence of fractions and reducing term rewriting.
# 
# Parameters
# ==========
# 
# iszerofunc : function, optional
#     Function to determine zeros during LU decomposition. Defaults to `lambda x: x.is_zero`.
# 
# TODO: Implement algorithm for sparse matrices (SFF),
# http://www.eecis.udel.edu/~saunders/papers/sffge/it5.ps.
def _det_bareiss(M, iszerofunc=_is_zero_after_expand_mul):
    """Compute matrix determinant using Bareiss' fraction-free
    algorithm which is an extension of the well known Gaussian
    elimination method. This approach is best suited for dense
    symbolic matrices and will result in a determinant with
    minimal number of fractions. It means that less term
    rewriting is needed on resulting formulae.

    Parameters
    ==========

    iszerofunc : function, optional
        The function to use to determine zeros when doing an LU decomposition.
        Defaults to ``lambda x: x.is_zero``.

    TODO: Implement algorithm for sparse matrices (SFF),
    http://www.eecis.udel.edu/~saunders/papers/sffge/it5.ps.
    """

    # Recursively implemented Bareiss' algorithm as per Deanna Richelle Leggett's
    # thesis http://www.math.usm.edu/perry/Research/Thesis_DRL.pdf
    def bareiss(mat, cumm=1):
        if mat.rows == 0:
            return mat.one
        elif mat.rows == 1:
            return mat[0, 0]

        # find a pivot and extract the remaining matrix
        # With the default iszerofunc, _find_reasonable_pivot slows down
        # the computation by the factor of 2.5 in one test.
        # Relevant issues: #10279 and #13877.
        pivot_pos, pivot_val, _, _ = _find_reasonable_pivot(mat[:, 0], iszerofunc=iszerofunc)
        if pivot_pos is None:
            return mat.zero

        # if we have a valid pivot, we'll do a "row swap", so keep the
        # sign of the det
        sign = (-1) ** (pivot_pos % 2)

        # we want every row but the pivot row and every column
        rows = [i for i in range(mat.rows) if i != pivot_pos]
        cols = list(range(mat.cols))
        tmp_mat = mat.extract(rows, cols)

        def entry(i, j):
            ret = (pivot_val*tmp_mat[i, j + 1] - mat[pivot_pos, j + 1]*tmp_mat[i, 0]) / cumm
            if _get_intermediate_simp_bool(True):
                return _dotprodsimp(ret)
            elif not ret.is_Atom:
                return cancel(ret)
            return ret

        return sign*bareiss(M._new(mat.rows - 1, mat.cols - 1, entry), pivot_val)

    if not M.is_square:
        raise NonSquareMatrixError()

    if M.rows == 0:
        return M.one
        # sympy/matrices/tests/test_matrices.py contains a test that
        # suggests that the determinant of a 0 x 0 matrix is one, by
        # convention.

    return bareiss(M)


# This function computes the determinant of a matrix using the Berkowitz algorithm.
def _det_berkowitz(M):
    """ Use the Berkowitz algorithm to compute the determinant."""

    if not M.is_square:
        raise NonSquareMatrixError()

    if M.rows == 0:
        return M.one
        # sympy/matrices/tests/test_matrices.py contains a test that
        # suggests that the determinant of a 0 x 0 matrix is one, by
        # convention.

    berk_vector = _berkowitz_vector(M)
    return (-1)**(len(berk_vector) - 1) * berk_vector[-1]
# This functions is a candidate for caching if it gets implemented for matrices.
def _det_LU(M, iszerofunc=_iszero, simpfunc=None):
    """ Computes the determinant of a matrix from its LU decomposition.
    This function uses the LU decomposition computed by
    LUDecomposition_Simple().

    The keyword arguments iszerofunc and simpfunc are passed to
    LUDecomposition_Simple().
    iszerofunc is a callable that returns a boolean indicating if its
    input is zero, or None if it cannot make the determination.
    simpfunc is a callable that simplifies its input.
    The default is simpfunc=None, which indicate that the pivot search
    algorithm should not attempt to simplify any candidate pivots.
    If simpfunc fails to simplify its input, then it must return its input
    instead of a copy.

    Parameters
    ==========

    iszerofunc : function, optional
        The function to use to determine zeros when doing an LU decomposition.
        Defaults to ``lambda x: x.is_zero``.

    simpfunc : function, optional
        The simplification function to use when looking for zeros for pivots.
    """

    if not M.is_square:
        raise NonSquareMatrixError()

    if M.rows == 0:
        return M.one
        # sympy/matrices/tests/test_matrices.py contains a test that
        # suggests that the determinant of a 0 x 0 matrix is one, by
        # convention.

    # Perform LU decomposition of matrix M
    lu, row_swaps = M.LUdecomposition_Simple(iszerofunc=iszerofunc,
            simpfunc=simpfunc)
    # P*A = L*U => det(A) = det(L)*det(U)/det(P) = det(P)*det(U).
    # Lower triangular factor L encoded in lu has unit diagonal => det(L) = 1.
    # P is a permutation matrix => det(P) in {-1, 1} => 1/det(P) = det(P).
    # LUdecomposition_Simple() returns a list of row exchange index pairs, rather
    # than a permutation matrix, but det(P) = (-1)**len(row_swaps).

    # Avoid forming the potentially time consuming  product of U's diagonal entries
    # if the product is zero.
    # Bottom right entry of U is 0 => det(A) = 0.
    # It may be impossible to determine if this entry of U is zero when it is symbolic.
    if iszerofunc(lu[lu.rows-1, lu.rows-1]):
        return M.zero

    # Compute det(P)
    det = -M.one if len(row_swaps)%2 else M.one

    # Compute det(U) by calculating the product of U's diagonal entries.
    # The upper triangular portion of lu is the upper triangular portion of the
    # U factor in the LU decomposition.
    for k in range(lu.rows):
        det *= lu[k, k]

    # return det(P)*det(U)
    return det


@cacheit
def __det_laplace(M):
    """Compute the determinant of a matrix using Laplace expansion.

    This is a recursive function, and it should not be called directly.
    Use _det_laplace() instead. The reason for splitting this function
    into two is to allow caching of determinants of submatrices. While
    one could also define this function inside _det_laplace(), that
    would remove the advantage of using caching in Cramer Solve.
    """
    # 获取矩阵 M 的行数
    n = M.shape[0]
    # 如果矩阵 M 只有一行，直接返回该元素
    if n == 1:
        return M[0]
    # 如果矩阵 M 是 2x2 的，计算并返回行列式的值
    elif n == 2:
        return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    else:
        # 对于大于 2x2 的矩阵，使用拉普拉斯展开计算行列式
        return sum((-1) ** i * M[0, i] *
                   __det_laplace(M.minor_submatrix(0, i)) for i in range(n))
# 计算使用拉普拉斯展开法求矩阵的行列式

def _det_laplace(M):
    """Compute the determinant of a matrix using Laplace expansion.

    While Laplace expansion is not the most efficient method of computing
    a determinant, it is a simple one, and it has the advantage of
    being division free. To improve efficiency, this function uses
    caching to avoid recomputing determinants of submatrices.
    """
    # 检查矩阵是否为方阵
    if not M.is_square:
        raise NonSquareMatrixError()
    
    # 如果矩阵的行数为0，按照约定返回单位矩阵的值为1
    if M.shape[0] == 0:
        return M.one
        # sympy/matrices/tests/test_matrices.py 中有一个测试表明，0 x 0 矩阵的行列式按照约定为1。

    # 调用内部函数计算行列式
    return __det_laplace(M.as_immutable())


# 计算使用 Bird 算法求矩阵的行列式

def _det_bird(M):
    r"""Compute the determinant of a matrix using Bird's algorithm.

    Bird's algorithm is a simple division-free algorithm for computing, which
    is of lower order than the Laplace's algorithm. It is described in [1]_.

    References
    ==========

    .. [1] Bird, R. S. (2011). A simple division-free algorithm for computing
           determinants. Inf. Process. Lett., 111(21), 1072-1074. doi:
           10.1016/j.ipl.2011.08.006
    """
    # 定义内部函数 mu，用于 Bird 算法的计算
    def mu(X):
        n = X.shape[0]
        zero = X.domain.zero

        total = zero
        diag_sums = [zero]
        # 计算对角线元素的累加和
        for i in reversed(range(1, n)):
            total -= X[i][i]
            diag_sums.append(total)
        diag_sums = diag_sums[::-1]

        # 创建新的矩阵，用于 Bird 算法的中间步骤
        elems = [[zero] * i + [diag_sums[i]] + X_i[i + 1:] for i, X_i in
                 enumerate(X)]
        return DDM(elems, X.shape, X.domain)

    # 将输入矩阵转换为 DDM 格式
    Mddm = M._rep.to_ddm()
    n = M.shape[0]
    # 如果矩阵行数为0，按照约定返回单位矩阵的值为1
    if n == 0:
        return M.one
        # sympy/matrices/tests/test_matrices.py 中有一个测试表明，0 x 0 矩阵的行列式按照约定为1。

    # 初始化 Fn1 为 Mddm
    Fn1 = Mddm
    # 使用 Bird 算法迭代计算结果
    for _ in range(n - 1):
        Fn1 = mu(Fn1).matmul(Mddm)
    detA = Fn1[0][0]
    # 如果矩阵行数为偶数，调整结果的符号
    if n % 2 == 0:
        detA = -detA

    # 将计算结果从 DDM 格式转换为 sympy 格式并返回
    return Mddm.domain.to_sympy(detA)


# 返回指定矩阵的 (i,j) 小行列式

def _minor(M, i, j, method="berkowitz"):
    """Return the (i,j) minor of ``M``.  That is,
    return the determinant of the matrix obtained by deleting
    the `i`th row and `j`th column from ``M``.

    Parameters
    ==========

    i, j : int
        The row and column to exclude to obtain the submatrix.

    method : string, optional
        Method to use to find the determinant of the submatrix, can be
        "bareiss", "berkowitz", "bird", "laplace" or "lu".

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> M.minor(1, 1)
    -12

    See Also
    ========

    minor_submatrix
    cofactor
    det
    """

    # 检查矩阵是否为方阵
    if not M.is_square:
        raise NonSquareMatrixError()

    # 调用矩阵对象的 minor_submatrix 方法计算子矩阵的行列式
    return M.minor_submatrix(i, j).det(method=method)


# 返回移除指定行和列后的子矩阵

def _minor_submatrix(M, i, j):
    """Return the submatrix obtained by removing the `i`th row
    and `j`th column from ``M`` (works with Pythonic negative indices).

    Parameters
    ==========

    i, j : int
        The row and column to exclude to obtain the submatrix.
    """
    # i, j : int
    #     The row and column to exclude to obtain the submatrix.
    # 指定要排除的行和列，以获取子矩阵。

    # Examples
    # ========
    #
    # >>> from sympy import Matrix
    # >>> M = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # >>> M.minor_submatrix(1, 1)
    # Matrix([
    # [1, 3],
    # [7, 9]])
    #
    # See Also
    # ========
    #
    # minor
    # cofactor
    # """
    
    # 如果 i 小于 0，则将其转换为 M 的行数加上 i
    if i < 0:
        i += M.rows
    # 如果 j 小于 0，则将其转换为 M 的列数加上 j
    if j < 0:
        j += M.cols

    # 如果 i 不在有效的行范围内，或者 j 不在有效的列范围内，则引发 ValueError
    if not 0 <= i < M.rows or not 0 <= j < M.cols:
        raise ValueError("`i` and `j` must satisfy 0 <= i < ``M.rows`` "
                         "(%d)" % M.rows + "and 0 <= j < ``M.cols`` (%d)." % M.cols)

    # 获取剩余行和列的索引列表，这些索引不包括 i 和 j 对应的行和列
    rows = [a for a in range(M.rows) if a != i]
    cols = [a for a in range(M.cols) if a != j]

    # 提取并返回由剩余行和列组成的子矩阵
    return M.extract(rows, cols)
```