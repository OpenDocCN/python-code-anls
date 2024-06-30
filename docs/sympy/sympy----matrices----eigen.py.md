# `D:\src\scipysrc\sympy\sympy\matrices\eigen.py`

```
# 从types模块导入FunctionType类，用于检查对象是否函数类型
from types import FunctionType
# 从collections模块导入Counter类，用于计数可迭代对象中元素的出现次数
from collections import Counter

# 从mpmath库导入mp和workprec函数
from mpmath import mp, workprec
# 从mpmath.libmp.libmpf模块导入prec_to_dps函数
from mpmath.libmp.libmpf import prec_to_dps

# 从sympy.core.sorting模块导入default_sort_key函数
from sympy.core.sorting import default_sort_key
# 从sympy.core.evalf模块导入DEFAULT_MAXPREC和PrecisionExhausted类
from sympy.core.evalf import DEFAULT_MAXPREC, PrecisionExhausted
# 从sympy.core.logic模块导入fuzzy_and和fuzzy_or函数
from sympy.core.logic import fuzzy_and, fuzzy_or
# 从sympy.core.numbers模块导入Float类
from sympy.core.numbers import Float
# 从sympy.core.sympify模块导入_sympify函数
from sympy.core.sympify import _sympify
# 从sympy.functions.elementary.miscellaneous模块导入sqrt函数
from sympy.functions.elementary.miscellaneous import sqrt
# 从sympy.polys模块导入roots、CRootOf、ZZ、QQ、EX类
from sympy.polys import roots, CRootOf, ZZ, QQ, EX
# 从sympy.polys.matrices模块导入DomainMatrix类
from sympy.polys.matrices import DomainMatrix
# 从sympy.polys.matrices.eigen模块导入dom_eigenvects和dom_eigenvects_to_sympy函数
from sympy.polys.matrices.eigen import dom_eigenvects, dom_eigenvects_to_sympy
# 从sympy.polys.polytools模块导入gcd函数
from sympy.polys.polytools import gcd

# 从当前目录的exceptions模块导入MatrixError和NonSquareMatrixError异常类
from .exceptions import MatrixError, NonSquareMatrixError
# 从当前目录的determinant模块导入_find_reasonable_pivot函数
from .determinant import _find_reasonable_pivot
# 从当前目录的utilities模块导入_iszero和_simplify函数
from .utilities import _iszero, _simplify

# 设置doctest需要的外部依赖，当测试指定函数时需要matplotlib库
__doctest_requires__ = {
    ('_is_indefinite',
     '_is_negative_definite',
     '_is_negative_semidefinite',
     '_is_positive_definite',
     '_is_positive_semidefinite'): ['matplotlib'],
}


def _eigenvals_eigenvects_mpmath(M):
    """Compute eigenvalues and eigenvectors using mpmath."""
    # 定义计算向量范数的lambda函数norm2
    norm2 = lambda v: mp.sqrt(sum(i**2 for i in v))

    # 初始化v1为None
    v1 = None
    # 计算M中所有浮点数对象的最大精度
    prec = max(x._prec for x in M.atoms(Float))
    # 计算eps为2的负prec次方
    eps = 2**-prec

    # 当prec小于DEFAULT_MAXPREC时进行循环
    while prec < DEFAULT_MAXPREC:
        # 使用当前精度设置mpmath的工作精度
        with workprec(prec):
            # 将M转换为mpmath的矩阵对象A，并计算其特征值和特征向量
            A = mp.matrix(M.evalf(n=prec_to_dps(prec)))
            E, ER = mp.eig(A)
            # 计算当前特征值向量范数v2
            v2 = norm2([i for e in E for i in (mp.re(e), mp.im(e))])
            # 如果v1不为None且v1与v2之间的差小于eps，则返回计算结果
            if v1 is not None and mp.fabs(v1 - v2) < eps:
                return E, ER
            # 更新v1为当前的v2值
            v1 = v2
        # 将精度倍增
        prec *= 2

    # 当循环结束时，抛出精度耗尽的异常
    raise PrecisionExhausted


def _eigenvals_mpmath(M, multiple=False):
    """Compute eigenvalues using mpmath."""
    # 调用_eigenvals_eigenvects_mpmath计算特征值和特征向量
    E, _ = _eigenvals_eigenvects_mpmath(M)
    # 将特征值列表转换为Sympy对象列表
    result = [_sympify(x) for x in E]
    # 如果multiple为True，则直接返回结果列表，否则返回计数后的字典
    if multiple:
        return result
    return dict(Counter(result))


def _eigenvects_mpmath(M):
    """Compute eigenvectors using mpmath."""
    # 调用_eigenvals_eigenvects_mpmath计算特征值和特征向量
    E, ER = _eigenvals_eigenvects_mpmath(M)
    # 初始化结果列表
    result = []
    # 遍历M的行数，计算每个特征值对应的特征向量并加入结果列表
    for i in range(M.rows):
        eigenval = _sympify(E[i])
        eigenvect = _sympify(ER[:, i])
        result.append((eigenval, 1, [eigenvect]))

    return result


# This function is a candidate for caching if it gets implemented for matrices.
def _eigenvals(
    M, error_when_incomplete=True, *, simplify=False, multiple=False,
    rational=False, **flags):
    r"""Compute eigenvalues of the matrix.

    Parameters
    ==========

    error_when_incomplete : bool, optional
        If it is set to ``True``, it will raise an error if not all
        eigenvalues are computed. This is caused by ``roots`` not returning
        a full list of eigenvalues.
    # 如果矩阵 M 为空，根据 multiple 的设置返回空列表或空字典
    if not M:
        if multiple:
            return []
        return {}

    # 如果矩阵 M 不是方阵，抛出 NonSquareMatrixError 异常
    if not M.is_square:
        raise NonSquareMatrixError("{} must be a square matrix.".format(M))

    # 如果矩阵 M 的元素类型不是整数或有理数，且包含浮点数，则调用 _eigenvals_mpmath 函数处理
    if M._rep.domain not in (ZZ, QQ):
        # 跳过 ZZ/QQ 类型的检查，因为它可能会很慢
        if all(x.is_number for x in M) and M.has(Float):
            return _eigenvals_mpmath(M, multiple=multiple)

    # 如果 rational 为 True，则对矩阵 M 中的浮点数进行有理数替换
    if rational:
        from sympy.simplify import nsimplify
        M = M.applyfunc(
            lambda x: nsimplify(x, rational=True) if x.has(Float) else x)

    # 如果 multiple 为 True，则调用 _eigenvals_list 函数返回多个特征值的列表形式
    if multiple:
        return _eigenvals_list(
            M, error_when_incomplete=error_when_incomplete, simplify=simplify,
            **flags)
    
    # 如果 multiple 为 False，则调用 _eigenvals_dict 函数返回特征值的字典形式
    return _eigenvals_dict(
        M, error_when_incomplete=error_when_incomplete, simplify=simplify,
        **flags)
# 定义错误消息字符串，说明不能用根式表达大于5x5尺寸的矩阵的特征值，只支持有理数域
eigenvals_error_message = \
"It is not always possible to express the eigenvalues of a matrix " + \
"of size 5x5 or higher in radicals. " + \
"We have CRootOf, but domains other than the rationals are not " + \
"currently supported. " + \
"If there are no symbols in the matrix, " + \
"it should still be possible to compute numeric approximations " + \
"of the eigenvalues using " + \
"M.evalf().eigenvals() or M.charpoly().nroots()."

# 定义函数 _eigenvals_list，计算矩阵的特征值列表
def _eigenvals_list(
    M, error_when_incomplete=True, simplify=False, **flags):
    # 计算强连通分量
    iblocks = M.strongly_connected_components()
    # 存储所有特征值的列表
    all_eigs = []
    # 判断矩阵的表示域是否为整数环或有理数域
    is_dom = M._rep.domain in (ZZ, QQ)
    # 遍历每个强连通分量
    for b in iblocks:

        # 对于1x1大小的块的快速路径：
        if is_dom and len(b) == 1:
            index = b[0]
            val = M[index, index]
            all_eigs.append(val)
            continue

        # 获取当前块的子矩阵
        block = M[b, b]

        # 根据是否提供简化函数选择使用的方法生成特征多项式
        if isinstance(simplify, FunctionType):
            charpoly = block.charpoly(simplify=simplify)
        else:
            charpoly = block.charpoly()

        # 计算特征多项式的所有根
        eigs = roots(charpoly, multiple=True, **flags)

        # 如果根数与块的行数不匹配，则尝试获取所有根的列表
        if len(eigs) != block.rows:
            try:
                eigs = charpoly.all_roots(multiple=True)
            except NotImplementedError:
                # 如果无法获取完整的根，则根据设置决定是否引发异常
                if error_when_incomplete:
                    raise MatrixError(eigenvals_error_message)
                else:
                    eigs = []

        # 将当前块的特征值添加到总列表中
        all_eigs += eigs

    # 如果不需要简化，则直接返回所有特征值的列表
    if not simplify:
        return all_eigs
    # 如果简化函数不是函数类型，则使用内置的简化函数
    if not isinstance(simplify, FunctionType):
        simplify = _simplify
    # 使用简化函数对所有特征值进行简化处理，并返回结果列表
    return [simplify(value) for value in all_eigs]


# 定义函数 _eigenvals_dict，计算矩阵的特征值字典
def _eigenvals_dict(
    M, error_when_incomplete=True, simplify=False, **flags):
    # 计算强连通分量
    iblocks = M.strongly_connected_components()
    # 存储所有特征值的字典
    all_eigs = {}
    # 判断矩阵的表示域是否为整数环或有理数域
    is_dom = M._rep.domain in (ZZ, QQ)
    # 遍历每个强连通分量
    for b in iblocks:

        # 对于1x1大小的块的快速路径：
        if is_dom and len(b) == 1:
            index = b[0]
            val = M[index, index]
            # 将特征值及其出现次数添加到字典中
            all_eigs[val] = all_eigs.get(val, 0) + 1
            continue

        # 获取当前块的子矩阵
        block = M[b, b]

        # 根据是否提供简化函数选择使用的方法生成特征多项式
        if isinstance(simplify, FunctionType):
            charpoly = block.charpoly(simplify=simplify)
        else:
            charpoly = block.charpoly()

        # 计算特征多项式的所有根
        eigs = roots(charpoly, multiple=False, **flags)

        # 如果根的数量与块的行数不匹配，则尝试获取所有根的字典
        if sum(eigs.values()) != block.rows:
            try:
                eigs = dict(charpoly.all_roots(multiple=False))
            except NotImplementedError:
                # 如果无法获取完整的根，则根据设置决定是否引发异常
                if error_when_incomplete:
                    raise MatrixError(eigenvals_error_message)
                else:
                    eigs = {}

        # 将当前块的特征值及其出现次数添加到总字典中
        for k, v in eigs.items():
            if k in all_eigs:
                all_eigs[k] += v
            else:
                all_eigs[k] = v

    # 如果不需要简化，则直接返回所有特征值的字典
    if not simplify:
        return all_eigs
    # 如果简化函数不是函数类型，则使用内置的简化函数
    if not isinstance(simplify, FunctionType):
        simplify = _simplify
    # 使用简化函数对所有特征值的键进行简化处理，并返回结果字典
    return {simplify(key): value for key, value in all_eigs.items()}
    """Get a basis for the eigenspace for a particular eigenvalue"""
    # 构建矩阵 m，将对角线上减去给定特征值的单位矩阵，用于计算特征向量空间的基
    m   = M - M.eye(M.rows) * eigenval
    # 使用 nullspace 方法计算矩阵 m 的零空间（特征向量空间），使用自定义的零判定函数 iszerofunc
    ret = m.nullspace(iszerofunc=iszerofunc)

    # 对于实特征值，其零空间应该非平凡（即不只包含零向量）
    # 如果未找到特征向量，且允许简化操作，则再次尝试更严格的条件
    if len(ret) == 0 and simplify:
        ret = m.nullspace(iszerofunc=iszerofunc, simplify=True)
    # 如果仍未找到特征向量，则抛出未实现错误，说明无法计算给定特征值的特征向量
    if len(ret) == 0:
        raise NotImplementedError(
            "Can't evaluate eigenvector for eigenvalue {}".format(eigenval))
    # 返回找到的特征向量空间的基
    return ret
# 从给定的矩阵 M 创建一个域矩阵对象 DOM，指定 field=True 和 extension=True
# 域矩阵是指可以处理多项式和扩展对象的矩阵
DOM = DomainMatrix.from_Matrix(M, field=True, extension=True)

# 将域矩阵 DOM 转换为密集矩阵
DOM = DOM.to_dense()

# 检查域矩阵的域是否与全局符号 EX 相同
if DOM.domain != EX:
    # 如果不相同，计算 DOM 的有理数和代数数特征向量
    rational, algebraic = dom_eigenvects(DOM)

    # 将 DOM 的特征向量转换为 SymPy 的表示形式，使用给定的参数 kwargs
    eigenvects = dom_eigenvects_to_sympy(
        rational, algebraic, M.__class__, **kwargs)

    # 对特征向量列表按照默认排序键排序
    eigenvects = sorted(eigenvects, key=lambda x: default_sort_key(x[0]))

    # 返回排序后的特征向量列表
    return eigenvects
# 如果域矩阵的域与全局符号 EX 相同，则返回空值
return None


# 计算矩阵 M 的特征值，忽略有理数的返回，只返回代数数的特征值
eigenvals = M.eigenvals(rational=False, **flags)

# 确保所有特征值都以根式形式表示
for x in eigenvals:
    if x.has(CRootOf):
        raise MatrixError(
            "Eigenvector computation is not implemented if the matrix have "
            "eigenvalues in CRootOf form")

# 对特征值项按照默认排序键排序，并转换为列表形式
eigenvals = sorted(eigenvals.items(), key=default_sort_key)

# 初始化结果列表
ret = []

# 对每个特征值及其重数进行迭代
for val, mult in eigenvals:
    # 计算特征值 val 对应的特征空间的特征向量
    vects = _eigenspace(M, val, iszerofunc=iszerofunc, simplify=simplify)

    # 将特征值 val、重数 mult 和特征向量列表 vects 组成元组，添加到结果列表 ret 中
    ret.append((val, mult, vects))

# 返回特征值、重数和特征向量的列表
return ret


# 这个函数如果为矩阵实现了缓存功能，可以进行缓存
def _eigenvects(M, error_when_incomplete=True, iszerofunc=_iszero, *, chop=False, **flags):
    """计算矩阵的特征向量。

    Parameters
    ==========

    error_when_incomplete : bool, optional
        当未计算出所有特征值时是否引发错误。这是由于 `roots` 没有返回完整的特征值列表引起的。

    iszerofunc : function, optional
        指定用于 `rref` 的零测试函数。

        默认值是 `_iszero`，它使用 SymPy 的简单且快速的默认假设处理器。

        如果格式化为一个接受单个符号参数并根据测试为零返回 `True`，测试为非零返回 `False`，
        不可决返回 `None` 的任何用户指定的零测试函数也可以接受。

    simplify : bool or function, optional
        如果为 `True`，将使用 `as_content_primitive()` 来清理归一化的工件。

        它也将被 `nullspace` 程序使用。

    chop : bool or positive number, optional
        如果矩阵包含任何浮点数，它们将被更改为计算目的的有理数，但答案将在使用 `evalf` 后返回。
        `chop` 标志传递给 `evalf`。
        当 `chop=True` 时，将使用默认精度；一个数字将被解释为所需的精度级别。

    Returns
    =======

    """
    # 获取参数中的 simplify 和 primitive 标志，如果不存在则默认为 True 和 False
    simplify = flags.get('simplify', True)
    primitive = flags.get('simplify', False)
    # 从 flags 字典中移除 'simplify' 和 'multiple' 键
    flags.pop('simplify', None)  # 如果存在 'simplify' 键则移除
    flags.pop('multiple', None)  # 如果存在 'multiple' 键则移除

    # 如果 simplify 不是 FunctionType 类型，则根据其值决定使用哪种简化函数
    if not isinstance(simplify, FunctionType):
        simpfunc = _simplify if simplify else lambda x: x

    # 检查矩阵 M 是否包含浮点数
    has_floats = M.has(Float)
    if has_floats:
        # 如果 M 中的所有元素都是数字，则使用 _eigenvects_mpmath 进行计算
        if all(x.is_number for x in M):
            return _eigenvects_mpmath(M)
        # 导入 nsimplify 函数并使用它来处理 M 中的每个元素
        from sympy.simplify import nsimplify
        M = M.applyfunc(lambda x: nsimplify(x, rational=True))

    # 使用 _eigenvects_DOM 函数计算 M 的特征向量
    ret = _eigenvects_DOM(M)
    # 如果 _eigenvects_DOM 返回 None，则使用 _eigenvects_sympy 函数计算特征向量
    if ret is None:
        ret = _eigenvects_sympy(M, iszerofunc, simplify=simplify, **flags)

    # 如果设置了 primitive 标志，则对特征向量列表中的每个向量进行去除公共整数分母操作
    if primitive:
        # 定义 denom_clean 函数，用于去除每个向量的公共整数分母
        def denom_clean(l):
            return [(v / gcd(list(v))).applyfunc(simpfunc) for v in l]

        # 对 ret 中的每个元素应用 denom_clean 函数
        ret = [(val, mult, denom_clean(es)) for val, mult, es in ret]

    # 如果原始矩阵 M 包含浮点数，则将返回的特征值和特征向量转换为浮点数
    if has_floats:
        # 使用 evalf 方法将特征值和每个特征向量的元素转换为浮点数
        ret = [(val.evalf(chop=chop), mult, [v.evalf(chop=chop) for v in es])
               for val, mult, es in ret]

    # 返回计算得到的特征值、重数和特征空间的列表 ret
    return ret
# 根据给定的矩阵 M 进行 Householder 双对角分解
def _bidiagonal_decmp_hholder(M):
    m = M.rows  # 获取矩阵 M 的行数
    n = M.cols  # 获取矩阵 M 的列数
    A = M.as_mutable()  # 将矩阵 M 转换为可变类型

    # 初始化单位矩阵 U 和 V
    U, V = A.eye(m), A.eye(n)
    # 对于每一个 i 在范围内，执行以下操作，最多进行 min(m, n) 次循环
    for i in range(min(m, n)):
        # 通过 _householder_vector 函数计算得到 Householder 变换的向量 v 和系数 bet
        v, bet = _householder_vector(A[i:, i])
        
        # 计算 Householder 矩阵 hh_mat
        hh_mat = A.eye(m - i) - bet * v * v.H
        
        # 更新 A[i:, i:]，将 hh_mat 应用到 A[i:, i:] 上
        A[i:, i:] = hh_mat * A[i:, i:]
        
        # 创建单位矩阵 temp，更新其子矩阵部分
        temp = A.eye(m)
        temp[i:, i:] = hh_mat
        
        # 更新 U 矩阵
        U = U * temp
        
        # 如果 i + 1 小于等于 n - 2，则执行以下操作
        if i + 1 <= n - 2:
            # 通过 _householder_vector 函数计算得到 Householder 变换的向量 v 和系数 bet
            v, bet = _householder_vector(A[i, i+1:].T)
            
            # 计算 Householder 矩阵 hh_mat
            hh_mat = A.eye(n - i - 1) - bet * v * v.H
            
            # 更新 A[i:, i+1:]，将 A[i:, i+1:] 应用到 hh_mat 上
            A[i:, i+1:] = A[i:, i+1:] * hh_mat
            
            # 创建单位矩阵 temp，更新其子矩阵部分
            temp = A.eye(n)
            temp[i+1:, i+1:] = hh_mat
            
            # 更新 V 矩阵
            V = temp * V
    
    # 返回计算结果 U, A, V
    return U, A, V
# 定义函数_eval_bidiag_hholder，用于计算输入矩阵的双对角化形式
def _eval_bidiag_hholder(M):
    # 获取矩阵的行数m和列数n
    m = M.rows
    n = M.cols
    # 将输入矩阵转换为可变形式
    A = M.as_mutable()
    # 循环直到达到m和n中的较小值
    for i in range(min(m, n)):
        # 计算 Householder 向量及其 beta 值
        v, bet = _householder_vector(A[i:, i])
        # 计算 Householder 变换矩阵
        hh_mat = A.eye(m-i) - bet * v * v.H
        # 更新矩阵A的对角线及其以上部分
        A[i:, i:] = hh_mat * A[i:, i:]
        # 如果i + 1小于n - 2，则继续执行下面的操作
        if i + 1 <= n - 2:
            # 计算 Householder 向量及其 beta 值
            v, bet = _householder_vector(A[i, i+1:].T)
            # 计算 Householder 变换矩阵
            hh_mat = A.eye(n - i - 1) - bet * v * v.H
            # 更新矩阵A的次对角线部分
            A[i:, i+1:] = A[i:, i+1:] * hh_mat
    # 返回双对角化后的矩阵A
    return A


# 定义函数_bidiagonal_decomposition，用于计算矩阵的双对角分解（upper为True时）
def _bidiagonal_decomposition(M, upper=True):
    """
    Returns $(U,B,V.H)$ for

    $$A = UBV^{H}$$

    where $A$ is the input matrix, and $B$ is its Bidiagonalized form

    Note: Bidiagonal Computation can hang for symbolic matrices.

    Parameters
    ==========

    upper : bool. Whether to do upper bidiagnalization or lower.
                True for upper and False for lower.

    References
    ==========

    .. [1] Algorithm 5.4.2, Matrix computations by Golub and Van Loan, 4th edition
    .. [2] Complex Matrix Bidiagonalization, https://github.com/vslobody/Householder-Bidiagonalization

    """

    # 检查upper是否为布尔值
    if not isinstance(upper, bool):
        raise ValueError("upper must be a boolean")

    # 如果upper为True，则调用_eval_bidiag_hholder函数进行上双对角化计算
    if upper:
        return _eval_bidiag_hholder(M)

    # 如果upper为False，则对M的共轭转置进行上双对角化计算
    X = _bidiagonal_decmp_hholder(M.H)
    return X[2].H, X[1].H, X[0].H


# 定义函数_bidiagonalize，用于计算输入矩阵的双对角化形式
def _bidiagonalize(M, upper=True):
    """
    Returns $B$, the Bidiagonalized form of the input matrix.

    Note: Bidiagonal Computation can hang for symbolic matrices.

    Parameters
    ==========

    upper : bool. Whether to do upper bidiagnalization or lower.
                True for upper and False for lower.

    References
    ==========

    .. [1] Algorithm 5.4.2, Matrix computations by Golub and Van Loan, 4th edition
    .. [2] Complex Matrix Bidiagonalization : https://github.com/vslobody/Householder-Bidiagonalization

    """

    # 检查upper是否为布尔值
    if not isinstance(upper, bool):
        raise ValueError("upper must be a boolean")

    # 如果upper为True，则调用_eval_bidiag_hholder函数进行上双对角化计算
    if upper:
        return _eval_bidiag_hholder(M)

    # 如果upper为False，则对M的共轭转置进行上双对角化计算，并返回其共轭转置
    return _eval_bidiag_hholder(M.H).H


# 定义函数_diagonalize，用于计算输入矩阵的对角化形式及其伴随矩阵
def _diagonalize(M, reals_only=False, sort=False, normalize=False):
    """
    Return (P, D), where D is diagonal and

        D = P^-1 * M * P

    where M is current matrix.

    Parameters
    ==========

    reals_only : bool. Whether to throw an error if complex numbers are need
                    to diagonalize. (Default: False)

    sort : bool. Sort the eigenvalues along the diagonal. (Default: False)

    normalize : bool. If True, normalize the columns of P. (Default: False)

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix(3, 3, [1, 2, 0, 0, 3, 0, 2, -4, 2])
    >>> M
    Matrix([
    [1,  2, 0],
    [0,  3, 0],
    [2, -4, 2]])
    >>> (P, D) = M.diagonalize()
    >>> D
    Matrix([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 3]])
    >>> P
    Matrix([
    [-1, 0, -1],
    [ 0, 0, -1],
    [ 2, 1,  2]])
    >>> P.inv() * M * P
    Matrix([
    [1, 0, 0],
    [0, 2, 0],
    [0, 0, 3]])

    See Also
    ========


    """

    # 返回M的对角化形式和伴随矩阵
    return M.diagonalize(reals_only=reals_only, sort=sort, normalize=normalize)
    # 检查矩阵 M 是否为方阵，如果不是则引发 NonSquareMatrixError 异常
    if not M.is_square:
        raise NonSquareMatrixError()

    # 调用 _is_diagonalizable_with_eigen 函数判断矩阵 M 是否可对角化，并获取其结果及特征向量
    is_diagonalizable, eigenvecs = _is_diagonalizable_with_eigen(M,
                reals_only=reals_only)

    # 如果矩阵 M 不可对角化，则引发 MatrixError 异常
    if not is_diagonalizable:
        raise MatrixError("Matrix is not diagonalizable")

    # 如果 sort 标志为真，则对特征向量 eigenvecs 按照默认排序键进行排序
    if sort:
        eigenvecs = sorted(eigenvecs, key=default_sort_key)

    # 初始化空列表 p_cols 和 diag，用于存储特征向量的列和对角元素
    p_cols, diag = [], []

    # 遍历 eigenvecs 中的每个特征值 val 及其重数 mult，以及对应的特征向量组成的基
    for val, mult, basis in eigenvecs:
        # 将 val 重复 mult 次添加到 diag 列表中
        diag += [val] * mult
        # 将 basis 添加到 p_cols 列表中
        p_cols += basis

    # 如果 normalize 标志为真，则对 p_cols 中的每个向量进行归一化处理
    if normalize:
        p_cols = [v / v.norm() for v in p_cols]

    # 构建并返回由特征向量列和对角元素组成的新矩阵
    return M.hstack(*p_cols), M.diag(*diag)
def _fuzzy_positive_definite(M):
    # 检查矩阵 M 是否具有正对角元素
    positive_diagonals = M._has_positive_diagonals()
    if positive_diagonals is False:
        return False

    # 如果矩阵具有正对角元素且强对角线占优，则返回 True
    if positive_diagonals and M.is_strongly_diagonally_dominant:
        return True

    # 否则返回 None
    return None


def _fuzzy_positive_semidefinite(M):
    # 检查矩阵 M 是否具有非负对角元素
    nonnegative_diagonals = M._has_nonnegative_diagonals()
    if nonnegative_diagonals is False:
        return False

    # 如果矩阵具有非负对角元素且弱对角线占优，则返回 True
    if nonnegative_diagonals and M.is_weakly_diagonally_dominant:
        return True

    # 否则返回 None
    return None


def _is_positive_definite(M):
    # 如果矩阵 M 不是厄米特矩阵，则将其转换为对称矩阵
    if not M.is_hermitian:
        if not M.is_square:
            return False
        M = M + M.H

    # 使用模糊判定函数检测 M 的正定性
    fuzzy = _fuzzy_positive_definite(M)
    if fuzzy is not None:
        return fuzzy

    # 使用高斯消元方法判定 M 的正定性
    return _is_positive_definite_GE(M)


def _is_positive_semidefinite(M):
    # 如果矩阵 M 不是厄米特矩阵，则将其转换为对称矩阵
    if not M.is_hermitian:
        if not M.is_square:
            return False
        M = M + M.H

    # 使用模糊判定函数检测 M 的半正定性
    fuzzy = _fuzzy_positive_semidefinite(M)
    if fuzzy is not None:
        return fuzzy

    # 使用 Cholesky 分解方法判定 M 的半正定性
    return _is_positive_semidefinite_cholesky(M)


def _is_negative_definite(M):
    # 检测矩阵 -M 的正定性
    return _is_positive_definite(-M)


def _is_negative_semidefinite(M):
    # 检测矩阵 -M 的半正定性
    return _is_positive_semidefinite(-M)


def _is_indefinite(M):
    # 如果矩阵 M 是厄米特矩阵
    if M.is_hermitian:
        # 获取 M 的特征值
        eigen = M.eigenvals()
        args1        = [x.is_positive for x in eigen.keys()]
        any_positive = fuzzy_or(args1)
        args2        = [x.is_negative for x in eigen.keys()]
        any_negative = fuzzy_or(args2)

        # 判断 M 是否是不定的
        return fuzzy_and([any_positive, any_negative])

    # 如果矩阵 M 是方阵，则检测 M + M.H 是否不定
    elif M.is_square:
        return (M + M.H).is_indefinite

    # 否则返回 False
    return False


def _is_positive_definite_GE(M):
    """一个无需除法的高斯消元法用于测试正定性。"""
    M = M.as_mutable()
    size = M.rows

    # 对 M 进行高斯消元
    for i in range(size):
        # 检查主对角元素是否为正
        is_positive = M[i, i].is_positive
        if is_positive is not True:
            return is_positive
        for j in range(i+1, size):
            # 使用高斯消元法处理剩余部分
            M[j, i+1:] = M[i, i] * M[j, i+1:] - M[j, i] * M[i, i+1:]
    # 如果所有元素都符合正定性条件，则返回 True
    return True


def _is_positive_semidefinite_cholesky(M):
    """使用完全主元的 Cholesky 分解方法判定半正定性。

    参考文献
    ==========

    .. [1] http://eprints.ma.man.ac.uk/1199/1/covered/MIMS_ep2008_116.pdf

    .. [2] https://www.value-at-risk.net/cholesky-factorization/
    """
    M = M.as_mutable()
    # 遍历矩阵 M 的行数
    for k in range(M.rows):
        # 获取从主对角线开始到最后一个对角线的值组成的列表
        diags = [M[i, i] for i in range(k, M.rows)]
        # 调用函数 _find_reasonable_pivot 查找合适的主元
        pivot, pivot_val, nonzero, _ = _find_reasonable_pivot(diags)

        # 如果存在非零元素，则返回 None
        if nonzero:
            return None

        # 如果找不到主元，检查子矩阵右下角是否全为零
        if pivot is None:
            # 检查子矩阵右下角是否有非零元素，如果有则返回 None
            for i in range(k+1, M.rows):
                for j in range(k, M.cols):
                    iszero = M[i, j].is_zero
                    if iszero is None:
                        return None
                    elif iszero is False:
                        return False
            # 如果子矩阵右下角全为零，则返回 True
            return True

        # 检查主对角线元素或找到的主元是否为负数，如果是则返回 False
        if M[k, k].is_negative or pivot_val.is_negative:
            return False
        # 检查主对角线元素和找到的主元是否非负，如果不是则返回 None
        elif not (M[k, k].is_nonnegative and pivot_val.is_nonnegative):
            return None

        # 如果主元位置大于零，则交换矩阵 M 的列和行
        if pivot > 0:
            M.col_swap(k, k+pivot)
            M.row_swap(k, k+pivot)

        # 将主对角线元素开方
        M[k, k] = sqrt(M[k, k])
        # 主对角线元素右侧的元素除以主对角线元素的值
        M[k, k+1:] /= M[k, k]
        # 子矩阵右下角的元素减去主对角线元素右侧的元素的共轭转置乘以主对角线元素右侧的元素
        M[k+1:, k+1:] -= M[k, k+1:].H * M[k, k+1:]

    # 返回矩阵 M 最右下角元素是否非负
    return M[-1, -1].is_nonnegative
# 定义一个字符串变量，包含了关于矩阵定性的文档字符串
_doc_positive_definite = \
    r"""Finds out the definiteness of a matrix.

    Explanation
    ===========

    A square real matrix $A$ is:

    - A positive definite matrix if $x^T A x > 0$
      for all non-zero real vectors $x$.
    - A positive semidefinite matrix if $x^T A x \geq 0$
      for all non-zero real vectors $x$.
    - A negative definite matrix if $x^T A x < 0$
      for all non-zero real vectors $x$.
    - A negative semidefinite matrix if $x^T A x \leq 0$
      for all non-zero real vectors $x$.
    - An indefinite matrix if there exists non-zero real vectors
      $x, y$ with $x^T A x > 0 > y^T A y$.

    A square complex matrix $A$ is:

    - A positive definite matrix if $\text{re}(x^H A x) > 0$
      for all non-zero complex vectors $x$.
    - A positive semidefinite matrix if $\text{re}(x^H A x) \geq 0$
      for all non-zero complex vectors $x$.
    - A negative definite matrix if $\text{re}(x^H A x) < 0$
      for all non-zero complex vectors $x$.
    - A negative semidefinite matrix if $\text{re}(x^H A x) \leq 0$
      for all non-zero complex vectors $x$.
    - An indefinite matrix if there exists non-zero complex vectors
      $x, y$ with $\text{re}(x^H A x) > 0 > \text{re}(y^H A y)$.

    A matrix need not be symmetric or hermitian to be positive definite.

    - A real non-symmetric matrix is positive definite if and only if
      $\frac{A + A^T}{2}$ is positive definite.
    - A complex non-hermitian matrix is positive definite if and only if
      $\frac{A + A^H}{2}$ is positive definite.

    And this extension can apply for all the definitions above.

    However, for complex cases, you can restrict the definition of
    $\text{re}(x^H A x) > 0$ to $x^H A x > 0$ and require the matrix
    to be hermitian.
    But we do not present this restriction for computation because you
    can check ``M.is_hermitian`` independently with this and use
    the same procedure.

    Examples
    ========

    An example of symmetric positive definite matrix:

    .. plot::
        :context: reset
        :format: doctest
        :include-source: True

        >>> from sympy import Matrix, symbols
        >>> from sympy.plotting import plot3d
        >>> a, b = symbols('a b')
        >>> x = Matrix([a, b])

        >>> A = Matrix([[1, 0], [0, 1]])
        >>> A.is_positive_definite
        True
        >>> A.is_positive_semidefinite
        True

        >>> p = plot3d((x.T*A*x)[0, 0], (a, -1, 1), (b, -1, 1))

    An example of symmetric positive semidefinite matrix:

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> A = Matrix([[1, -1], [-1, 1]])
        >>> A.is_positive_definite
        False
        >>> A.is_positive_semidefinite
        True

        >>> p = plot3d((x.T*A*x)[0, 0], (a, -1, 1), (b, -1, 1))

    An example of symmetric negative definite matrix:

    """
    # 示例：定义一个负定矩阵
    >>> A = Matrix([[-1, 0], [0, -1]])
    # 检查矩阵是否是负定的
    >>> A.is_negative_definite
    True
    # 检查矩阵是否是负半定的
    >>> A.is_negative_semidefinite
    True
    # 检查矩阵是否不定
    >>> A.is_indefinite
    False
    
    # 绘制一个三维图像，展示矩阵的效果
    >>> p = plot3d((x.T*A*x)[0, 0], (a, -1, 1), (b, -1, 1))
    
    An example of symmetric indefinite matrix:
    
    # 示例：定义一个非对称不定矩阵
    >>> A = Matrix([[1, 2], [2, -1]])
    # 检查矩阵是否是不定的
    >>> A.is_indefinite
    True
    
    # 绘制一个三维图像，展示矩阵的效果
    >>> p = plot3d((x.T*A*x)[0, 0], (a, -1, 1), (b, -1, 1))
    
    An example of non-symmetric positive definite matrix.
    
    # 示例：定义一个非对称正定矩阵
    >>> A = Matrix([[1, 2], [-2, 1]])
    # 检查矩阵是否是正定的
    >>> A.is_positive_definite
    True
    # 检查矩阵是否是正半定的
    >>> A.is_positive_semidefinite
    True
    
    # 绘制一个三维图像，展示矩阵的效果
    >>> p = plot3d((x.T*A*x)[0, 0], (a, -1, 1), (b, -1, 1))
    
    Notes
    =====
    
    Although some people trivialize the definition of positive definite
    matrices only for symmetric or hermitian matrices, this restriction
    is not correct because it does not classify all instances of
    positive definite matrices from the definition $x^T A x > 0$ or
    $\text{re}(x^H A x) > 0$.
    
    For instance, ``Matrix([[1, 2], [-2, 1]])`` presented in
    the example above is an example of real positive definite matrix
    that is not symmetric.
    
    However, since the following formula holds true;
    
    # 虽然一些人将正定矩阵的定义局限于对称或共轭转置矩阵，
    # 但这种限制并不正确，因为它不能将所有正定矩阵的实例从
    # 定义 $x^T A x > 0$ 或 $\text{re}(x^H A x) > 0$ 中分类出来。
    
    .. math::
        \text{re}(x^H A x) > 0 \iff
        \text{re}(x^H \frac{A + A^H}{2} x) > 0
    
    We can classify all positive definite matrices that may or may not
    be symmetric or hermitian by transforming the matrix to
    $\frac{A + A^T}{2}$ or $\frac{A + A^H}{2}$
    (which is guaranteed to be always real symmetric or complex
    hermitian) and we can defer most of the studies to symmetric or
    hermitian positive definite matrices.
    
    But it is a different problem for the existence of Cholesky
    decomposition. Because even though a non symmetric or a non
    hermitian matrix can be positive definite, Cholesky or LDL
    decomposition does not exist because the decompositions require the
    matrix to be symmetric or hermitian.
    
    References
    ==========
    
    .. [1] https://en.wikipedia.org/wiki/Definiteness_of_a_matrix#Eigenvalues
    
    .. [2] https://mathworld.wolfram.com/PositiveDefiniteMatrix.html
    
    .. [3] Johnson, C. R. "Positive Definite Matrices." Amer.
        Math. Monthly 77, 259-264 1970.
# 将 `_doc_positive_definite` 的文档字符串赋值给 `_is_positive_definite` 对象的文档字符串
_is_positive_definite.__doc__     = _doc_positive_definite
# 将 `_doc_positive_definite` 的文档字符串赋值给 `_is_positive_semidefinite` 对象的文档字符串
_is_positive_semidefinite.__doc__ = _doc_positive_definite
# 将 `_doc_positive_definite` 的文档字符串赋值给 `_is_negative_definite` 对象的文档字符串
_is_negative_definite.__doc__     = _doc_positive_definite
# 将 `_doc_positive_definite` 的文档字符串赋值给 `_is_negative_semidefinite` 对象的文档字符串
_is_negative_semidefinite.__doc__ = _doc_positive_definite
# 将 `_doc_positive_definite` 的文档字符串赋值给 `_is_indefinite` 对象的文档字符串
_is_indefinite.__doc__            = _doc_positive_definite


def _jordan_form(M, calc_transform=True, *, chop=False):
    """Return $(P, J)$ where $J$ is a Jordan block
    matrix and $P$ is a matrix such that $M = P J P^{-1}$

    Parameters
    ==========

    calc_transform : bool
        If ``False``, then only $J$ is returned.

    chop : bool
        All matrices are converted to exact types when computing
        eigenvalues and eigenvectors.  As a result, there may be
        approximation errors.  If ``chop==True``, these errors
        will be truncated.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[ 6,  5, -2, -3], [-3, -1,  3,  3], [ 2,  1, -2, -3], [-1,  1,  5,  5]])
    >>> P, J = M.jordan_form()
    >>> J
    Matrix([
    [2, 1, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 2, 1],
    [0, 0, 0, 2]])

    See Also
    ========

    jordan_block
    """

    # 检查输入矩阵是否为方阵，若不是则引发异常
    if not M.is_square:
        raise NonSquareMatrixError("Only square matrices have Jordan forms")

    # 将输入矩阵赋值给局部变量 mat
    mat        = M
    # 检查矩阵中是否包含浮点数类型元素
    has_floats = M.has(Float)

    # 如果矩阵中包含浮点数，则尝试获取矩阵中所有浮点数元素的最大精度
    if has_floats:
        try:
            max_prec = max(term._prec for term in M.values() if isinstance(term, Float))
        except ValueError:
            # 如果矩阵中没有明确为浮点数的元素，设置默认的最大精度为 53
            max_prec = 53

        # 将最大精度转换为最大十进制精度（dps），设置最小的最大十进制精度为 15，以防止在包含未评估表达式的矩阵中丢失精度
        max_dps = max(prec_to_dps(max_prec), 15)

    def restore_floats(*args):
        """If ``has_floats`` is `True`, cast all ``args`` as
        matrices of floats."""

        # 如果矩阵中包含浮点数，则将所有参数转换为具有浮点数的矩阵
        if has_floats:
            args = [m.evalf(n=max_dps, chop=chop) for m in args]
        # 如果参数只有一个，则返回该参数
        if len(args) == 1:
            return args[0]

        # 如果参数大于一个，则返回参数列表
        return args

    # 缓存一些计算以提高速度
    mat_cache = {}

    def eig_mat(val, pow):
        """Cache computations of ``(M - val*I)**pow`` for quick
        retrieval"""

        # 如果计算已缓存，则直接返回计算结果
        if (val, pow) in mat_cache:
            return mat_cache[(val, pow)]

        # 如果计算未缓存，则根据乘幂值进行矩阵计算，并缓存结果
        if (val, pow - 1) in mat_cache:
            mat_cache[(val, pow)] = mat_cache[(val, pow - 1)].multiply(
                    mat_cache[(val, 1)], dotprodsimp=None)
        else:
            mat_cache[(val, pow)] = (mat - val*M.eye(M.rows)).pow(pow)

        return mat_cache[(val, pow)]

    # 辅助函数
    def nullity_chain(val, algebraic_multiplicity):
        """Calculate the sequence  [0, nullity(E), nullity(E**2), ...]
        until it is constant where ``E = M - val*I``"""
        
        # mat.rank() is faster than computing the null space,
        # so use the rank-nullity theorem
        cols    = M.cols
        ret     = [0]
        nullity = cols - eig_mat(val, 1).rank()
        i       = 2

        while nullity != ret[-1]:
            ret.append(nullity)

            if nullity == algebraic_multiplicity:
                break

            nullity  = cols - eig_mat(val, i).rank()
            i       += 1

            # Due to issues like #7146 and #15872, SymPy sometimes
            # gives the wrong rank. In this case, raise an error
            # instead of returning an incorrect matrix
            if nullity < ret[-1] or nullity > algebraic_multiplicity:
                raise MatrixError(
                    "SymPy had encountered an inconsistent "
                    "result while computing Jordan block: "
                    "{}".format(M))

        return ret


```    
    def blocks_from_nullity_chain(d):
        """Return a list of the size of each Jordan block.
        If d_n is the nullity of E**n, then the number
        of Jordan blocks of size n is

            2*d_n - d_(n-1) - d_(n+1)"""

        # d[0] is always the number of columns, so skip past it
        mid = [2*d[n] - d[n - 1] - d[n + 1] for n in range(1, len(d) - 1)]
        # d is assumed to plateau with "d[ len(d) ] == d[-1]", so
        # 2*d_n - d_(n-1) - d_(n+1) == d_n - d_(n-1)
        end = [d[-1] - d[-2]] if len(d) > 1 else [d[0]]

        return mid + end



    def pick_vec(small_basis, big_basis):
        """Picks a vector from big_basis that isn't in
        the subspace spanned by small_basis"""

        if len(small_basis) == 0:
            return big_basis[0]

        for v in big_basis:
            _, pivots = M.hstack(*(small_basis + [v])).echelon_form(
                    with_pivots=True)

            if pivots[-1] == len(small_basis):
                return v



    # roots doesn't like Floats, so replace them with Rationals
    if has_floats:
        from sympy.simplify import nsimplify
        mat = mat.applyfunc(lambda x: nsimplify(x, rational=True))



    # first calculate the jordan block structure
    eigs = mat.eigenvals()



    # Make sure that we have all roots in radical form
    for x in eigs:
        if x.has(CRootOf):
            raise MatrixError(
                "Jordan normal form is not implemented if the matrix have "
                "eigenvalues in CRootOf form")



    # most matrices have distinct eigenvalues
    # and so are diagonalizable.  In this case, don't
    # do extra work!
    # 如果特征值的数量等于矩阵的列数，则执行以下操作
    if len(eigs.keys()) == mat.cols:
        # 按照默认排序键对特征值进行排序，得到块的顺序
        blocks     = sorted(eigs.keys(), key=default_sort_key)
        # 根据特征值构造对角矩阵
        jordan_mat = mat.diag(*blocks)

        # 如果不需要计算变换矩阵，则返回浮点数修正后的约当矩阵
        if not calc_transform:
            return restore_floats(jordan_mat)

        # 计算每个特征值对应的特征矩阵的零空间的一维子空间的基
        jordan_basis = [eig_mat(eig, 1).nullspace()[0]
                for eig in blocks]
        # 将基组成矩阵
        basis_mat    = mat.hstack(*jordan_basis)

        # 返回浮点数修正后的基矩阵和约当矩阵
        return restore_floats(basis_mat, jordan_mat)

    # 存储块结构的空列表
    block_structure = []

    # 遍历排序后的特征值列表
    for eig in sorted(eigs.keys(), key=default_sort_key):
        # 获取代数重数
        algebraic_multiplicity = eigs[eig]
        # 计算特征值对应的零空间链
        chain = nullity_chain(eig, algebraic_multiplicity)
        # 从零空间链中获取块大小
        block_sizes = blocks_from_nullity_chain(chain)

        # 如果块大小列表为 [a, b, c, ...]，则大小为 1 的约当块有 a 个，大小为 2 的有 b 个，以此类推
        # 创建一个数组，每个块对应一个 (特征值, 块大小) 的条目
        size_nums = [(i+1, num) for i, num in enumerate(block_sizes)]

        # 我们期望较大的约当块出现在前面
        size_nums.reverse()

        # 将 (特征值, 块大小) 的条目扩展到块结构列表中
        block_structure.extend(
            [(eig, size) for size, num in size_nums for _ in range(num)])

    # 计算约当形式的总块大小
    jordan_form_size = sum(size for eig, size in block_structure)

    # 如果约当形式的总块大小不等于矩阵的行数，则抛出矩阵错误异常
    if jordan_form_size != M.rows:
        raise MatrixError(
            "SymPy 在计算约当块时遇到了不一致的结果：{}".format(M))

    # 生成器表达式，根据块结构生成约当块
    blocks     = (mat.jordan_block(size=size, eigenvalue=eig) for eig, size in block_structure)
    # 构造约当矩阵
    jordan_mat = mat.diag(*blocks)

    # 如果不需要计算变换矩阵，则返回浮点数修正后的约当矩阵
    if not calc_transform:
        return restore_floats(jordan_mat)

    # 初始化约当基组列表
    jordan_basis = []

    # 对于每个广义特征空间，计算其基础
    # 我们从 null( (A - eig*I)**n ) 中寻找一个向量，
    # 它不在 null( (A - eig*I)**(n-1) ) 中，其中 n 是约当块的大小
    #
    # 理想情况下，我们会遍历 block_structure 并计算每个广义特征空间。
    # 然而，这会导致大量不必要的计算。相反，我们分别处理每个特征值，
    # 因为我们知道它们的广义特征空间必须有线性独立的基。
    jordan_basis = []
    # 对特征值按照指定的排序函数进行排序，依次处理每个特征值
    for eig in sorted(eigs.keys(), key=default_sort_key):
        # 初始化特征值的基础向量集合
        eig_basis = []

        # 遍历块结构中的每个块特征值及其大小
        for block_eig, size in block_structure:
            # 如果块特征值不等于当前处理的特征值，则跳过此块
            if block_eig != eig:
                continue

            # 计算大块特征值对应的零空间和小块特征值对应的零空间
            null_big = (eig_mat(eig, size)).nullspace()
            null_small = (eig_mat(eig, size - 1)).nullspace()

            # 选择一个向量，它在大块特征值的基础上，但不在小块特征值的基础上，并且与其他共享相同特征值的广义特征向量无关
            vec = pick_vec(null_small + eig_basis, null_big)

            # 生成新的向量列表，扩展特征值基础向量集合和约当基础向量集合
            new_vecs = [eig_mat(eig, i).multiply(vec, dotprodsimp=None)
                        for i in range(size)]
            eig_basis.extend(new_vecs)
            jordan_basis.extend(reversed(new_vecs))

    # 将约当基础向量按列拼接成矩阵
    basis_mat = mat.hstack(*jordan_basis)

    # 返回浮点数形式的基础矩阵，同时恢复之前保存的浮点数精度
    return restore_floats(basis_mat, jordan_mat)
def _left_eigenvects(M, **flags):
    """Returns left eigenvectors and eigenvalues.

    This function returns the list of triples (eigenval, multiplicity,
    basis) for the left eigenvectors. Options are the same as for
    eigenvects(), i.e. the ``**flags`` arguments gets passed directly to
    eigenvects().

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[0, 1, 1], [1, 0, 0], [1, 1, 1]])
    >>> M.eigenvects()
    [(-1, 1, [Matrix([
    [-1],
    [ 1],
    [ 0]])]), (0, 1, [Matrix([
    [ 0],
    [-1],
    [ 1]])]), (2, 1, [Matrix([
    [2/3],
    [1/3],
    [  1]])])]
    >>> M.left_eigenvects()
    [(-1, 1, [Matrix([[-2, 1, 1]])]), (0, 1, [Matrix([[-1, -1, 1]])]), (2,
    1, [Matrix([[1, 1, 1]])])]

    """

    # Compute the transpose of matrix M and then find its eigenvectors
    eigs = M.transpose().eigenvects(**flags)

    # Rearrange the eigenvalues, multiplicities, and bases for left eigenvectors
    return [(val, mult, [l.transpose() for l in basis]) for val, mult, basis in eigs]


def _singular_values(M):
    """Compute the singular values of a Matrix

    Examples
    ========

    >>> from sympy import Matrix, Symbol
    >>> x = Symbol('x', real=True)
    >>> M = Matrix([[0, 1, 0], [0, x, 0], [-1, 0, 0]])
    >>> M.singular_values()
    [sqrt(x**2 + 1), 1, 0]

    See Also
    ========

    condition_number
    """

    # Check if M has more rows than columns or equal
    if M.rows >= M.cols:
        # Compute eigenvalues of M.H * M (Hermitian conjugate * M)
        valmultpairs = M.H.multiply(M).eigenvals()
    else:
        # Compute eigenvalues of M * M.H (M * Hermitian conjugate)
        valmultpairs = M.multiply(M.H).eigenvals()

    # Initialize an empty list for singular values
    vals = []

    # Expand eigenvalues with their multiplicities into a simple list
    for k, v in valmultpairs.items():
        vals += [sqrt(k)] * v  # Repeat sqrt(k) v times for each eigenvalue

    # Pad with zeros if the number of singular values is less than M's number of columns
    if len(vals) < M.cols:
        vals += [M.zero] * (M.cols - len(vals))

    # Sort singular values in descending order
    vals.sort(reverse=True, key=default_sort_key)

    return vals
```