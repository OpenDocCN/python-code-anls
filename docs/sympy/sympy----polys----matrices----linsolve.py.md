# `D:\src\scipysrc\sympy\sympy\polys\matrices\linsolve.py`

```
#
# sympy.polys.matrices.linsolve module
#
# This module defines the _linsolve function which is the internal workhorse
# used by linsolve. This computes the solution of a system of linear equations
# using the SDM sparse matrix implementation in sympy.polys.matrices.sdm. This
# is a replacement for solve_lin_sys in sympy.polys.solvers which is
# inefficient for large sparse systems due to the use of a PolyRing with many
# generators:
#
#     https://github.com/sympy/sympy/issues/20857
#
# The implementation of _linsolve here handles:
#
# - Extracting the coefficients from the Expr/Eq input equations.
# - Constructing a domain and converting the coefficients to
#   that domain.
# - Using the SDM.rref, SDM.nullspace etc methods to generate the full
#   solution working with arithmetic only in the domain of the coefficients.
#
# The routines here are particularly designed to be efficient for large sparse
# systems of linear equations although as well as dense systems. It is
# possible that for some small dense systems solve_lin_sys which uses the
# dense matrix implementation DDM will be more efficient. With smaller systems
# though the bulk of the time is spent just preprocessing the inputs and the
# relative time spent in rref is too small to be noticeable.
#

from collections import defaultdict

from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.singleton import S

from sympy.polys.constructor import construct_domain
from sympy.polys.solvers import PolyNonlinearError

from .sdm import (
    SDM,
    sdm_irref,
    sdm_particular_from_rref,
    sdm_nullspace_from_rref
)

from sympy.utilities.misc import filldedent


def _linsolve(eqs, syms):
    """Solve a linear system of equations.

    Examples
    ========

    Solve a linear system with a unique solution:

    >>> from sympy import symbols, Eq
    >>> from sympy.polys.matrices.linsolve import _linsolve
    >>> x, y = symbols('x, y')
    >>> eqs = [Eq(x + y, 1), Eq(x - y, 2)]
    >>> _linsolve(eqs, [x, y])
    {x: 3/2, y: -1/2}

    In the case of underdetermined systems the solution will be expressed in
    terms of the unknown symbols that are unconstrained:

    >>> _linsolve([Eq(x + y, 0)], [x, y])
    {x: -y, y: y}

    """
    # Number of unknowns (columns in the non-augmented matrix)
    nsyms = len(syms)

    # Convert to sparse augmented matrix (len(eqs) x (nsyms+1))
    eqsdict, const = _linear_eq_to_dict(eqs, syms)
    Aaug = sympy_dict_to_dm(eqsdict, const, syms)
    K = Aaug.domain

    # sdm_irref has issues with float matrices. This uses the ddm_rref()
    # function. When sdm_rref() can handle float matrices reasonably this
    # should be removed...
    if K.is_RealField or K.is_ComplexField:
        Aaug = Aaug.to_ddm().rref()[0].to_sdm()

    # Compute reduced-row echelon form (RREF)
    Arref, pivots, nzcols = sdm_irref(Aaug)

    # No solution:
    if pivots and pivots[-1] == nsyms:
        return None
    # 使用已简化的行阶梯形式矩阵 Arref 解非齐次线性系统的特解 P
    P = sdm_particular_from_rref(Arref, nsyms+1, pivots)

    # 使用 Arref 求解齐次线性系统的零空间，得到通解 V 和非主元列索引 nonpivots
    # 注意：使用 nsyms 而非 nsyms+1，以忽略最后一列
    V, nonpivots = sdm_nullspace_from_rref(Arref, K.one, nsyms, pivots, nzcols)

    # 将特解和通解的项收集到一个 defaultdict 中
    sol = defaultdict(list)
    for i, v in P.items():
        sol[syms[i]].append(K.to_sympy(v))
    for npi, Vi in zip(nonpivots, V):
        sym = syms[npi]
        for i, v in Vi.items():
            sol[syms[i]].append(sym * K.to_sympy(v))

    # 对每个符号的项使用 Add 函数进行合并
    sol = {s: Add(*terms) for s, terms in sol.items()}

    # 填充缺失的符号项为零
    zero = S.Zero
    for s in set(syms) - set(sol):
        sol[s] = zero

    # 所有步骤完成，返回最终的解 sol
    return sol
# 将符号表达式组成的方程系统转换为稀疏增广矩阵
def sympy_dict_to_dm(eqs_coeffs, eqs_rhs, syms):
    """Convert a system of dict equations to a sparse augmented matrix"""
    # 提取所有方程中的元素，并构建域K及其元素映射elems_K
    elems = set(eqs_rhs).union(*(e.values() for e in eqs_coeffs))
    K, elems_K = construct_domain(elems, field=True, extension=True)
    # 创建元素到elems_K的映射字典
    elem_map = dict(zip(elems, elems_K))
    # 获取方程的数量和符号的数量
    neqs = len(eqs_coeffs)
    nsyms = len(syms)
    # 创建符号到索引的映射字典
    sym2index = dict(zip(syms, range(nsyms)))
    # 初始化空列表用于存储每个方程的字典表示
    eqsdict = []
    # 遍历每个方程及其右侧常数项
    for eq, rhs in zip(eqs_coeffs, eqs_rhs):
        # 构建当前方程的符号到系数的映射字典eqdict
        eqdict = {sym2index[s]: elem_map[c] for s, c in eq.items()}
        # 如果右侧常数项非零，则添加到eqdict中
        if rhs:
            eqdict[nsyms] = -elem_map[rhs]
        # 如果eqdict非空，则将其添加到eqsdict中
        if eqdict:
            eqsdict.append(eqdict)
    # 使用稀疏增广矩阵SDM类创建并返回增广矩阵sdm_aug
    sdm_aug = SDM(enumerate(eqsdict), (neqs, nsyms + 1), K)
    return sdm_aug


# 将线性表达式/方程转换为字典形式，返回系数字典及独立于符号的项的列表
def _linear_eq_to_dict(eqs, syms):
    """Convert a system Expr/Eq equations into dict form, returning
    the coefficient dictionaries and a list of syms-independent terms
    from each expression in ``eqs```.

    Examples
    ========

    >>> from sympy.polys.matrices.linsolve import _linear_eq_to_dict
    >>> from sympy.abc import x
    >>> _linear_eq_to_dict([2*x + 3], {x})
    ([{x: 2}], [3])
    """
    # 初始化空列表coeffs用于存储每个方程的符号到系数的映射字典
    coeffs = []
    # 初始化空列表ind用于存储每个方程的独立于符号的项
    ind = []
    # 将符号集syms转换为集合symset
    symset = set(syms)
    # 遍历每个表达式e
    for e in eqs:
        # 如果e是一个等式
        if e.is_Equality:
            # 将左侧和右侧表达式转换为字典形式
            coeff, terms = _lin_eq2dict(e.lhs, symset)
            cR, tR = _lin_eq2dict(e.rhs, symset)
            # 执行系数的相减操作
            coeff -= cR
            # 处理右侧项的线性组合
            for k, v in tR.items():
                if k in terms:
                    terms[k] -= v
                else:
                    terms[k] = -v
            # 删除系数为0的项
            terms = {k: v for k, v in terms.items() if v}
            # 将coeff和terms分别赋值给c和d
            c, d = coeff, terms
        else:
            # 将表达式转换为字典形式
            c, d = _lin_eq2dict(e, symset)
        # 将字典形式的terms添加到coeffs中
        coeffs.append(d)
        # 将系数c添加到ind中
        ind.append(c)
    # 返回coeffs和ind
    return coeffs, ind


# 将表达式a转换为字典形式，返回其独立于符号的部分和符号到系数的映射字典
def _lin_eq2dict(a, symset):
    """return (c, d) where c is the sym-independent part of ``a`` and
    ``d`` is an efficiently calculated dictionary mapping symbols to
    their coefficients. A PolyNonlinearError is raised if non-linearity
    is detected.

    The values in the dictionary will be non-zero.

    Examples
    ========

    >>> from sympy.polys.matrices.linsolve import _lin_eq2dict
    >>> from sympy.abc import x, y
    >>> _lin_eq2dict(x + 2*y + 3, {x, y})
    (3, {x: 1, y: 2})
    """
    # 如果a是符号集symset中的一个符号，则返回(0, {a: 1})
    if a in symset:
        return S.Zero, {a: S.One}
    # 如果a是加法表达式
    elif a.is_Add:
        # 初始化terms_list为defaultdict(list)
        terms_list = defaultdict(list)
        # 初始化coeff_list为空列表
        coeff_list = []
        # 遍历a的每个项ai
        for ai in a.args:
            # 递归调用_lin_eq2dict，获取ci和ti
            ci, ti = _lin_eq2dict(ai, symset)
            # 将ci添加到coeff_list中
            coeff_list.append(ci)
            # 将ti中每一项的系数cij添加到terms_list[mij]中
            for mij, cij in ti.items():
                terms_list[mij].append(cij)
        # 将coeff_list中的所有项相加得到coeff
        coeff = Add(*coeff_list)
        # 将terms_list中的每一项符号的系数列表相加得到terms
        terms = {sym: Add(*coeffs) for sym, coeffs in terms_list.items()}
        # 返回coeff和terms
        return coeff, terms
    # 如果表达式 a 是乘法类型
    elif a.is_Mul:
        # 初始化变量
        terms = terms_coeff = None
        coeff_list = []

        # 遍历 a 中的每一个因子 ai
        for ai in a.args:
            # 将 ai 转换为字典表示的线性方程，ci 是系数，ti 是包含符号的字典
            ci, ti = _lin_eq2dict(ai, symset)
            
            # 如果 ti 为空，则将 ci 添加到系数列表中
            if not ti:
                coeff_list.append(ci)
            # 如果 terms 还未初始化，则将当前 ti 赋值给 terms，ci 赋值给 terms_coeff
            elif terms is None:
                terms = ti
                terms_coeff = ci
            else:
                # 如果 ti 不为空且 terms 已经初始化，则说明存在交叉项
                raise PolyNonlinearError(filldedent('''
                    nonlinear cross-term: %s''' % a))

        # 将系数列表中的所有项相乘得到最终的系数
        coeff = Mul._from_args(coeff_list)

        # 如果 terms 仍为 None，则返回计算出的系数和空字典
        if terms is None:
            return coeff, {}
        else:
            # 否则，将 terms 中的每个符号 sym 乘以 coeff * c，得到完整的项表示
            terms = {sym: coeff * c for sym, c in terms.items()}
            return coeff * terms_coeff, terms

    # 如果 a 中不包含符号集 symset 中的自由变量
    elif not a.has_xfree(symset):
        # 返回 a 本身和空字典，表示其为线性
        return a, {}
    
    # 如果以上条件均不满足，则抛出多项式非线性错误，指明非线性项的表达式 a
    else:
        raise PolyNonlinearError('nonlinear term: %s' % a)
```