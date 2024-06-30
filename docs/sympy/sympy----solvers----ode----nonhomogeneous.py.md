# `D:\src\scipysrc\sympy\sympy\solvers\ode\nonhomogeneous.py`

```
r"""
This File contains helper functions for nth_linear_constant_coeff_undetermined_coefficients,
nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients,
nth_linear_constant_coeff_variation_of_parameters,
and nth_linear_euler_eq_nonhomogeneous_variation_of_parameters.

All the functions in this file are used by more than one solvers so, instead of creating
instances in other classes for using them it is better to keep it here as separate helpers.

"""
from collections import defaultdict
from sympy.core import Add, S
from sympy.core.function import diff, expand, _mexpand, expand_mul
from sympy.core.relational import Eq
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Wild
from sympy.functions import exp, cos, cosh, im, log, re, sin, sinh, \
    atan2, conjugate
from sympy.integrals import Integral
from sympy.polys import (Poly, RootOf, rootof, roots)
from sympy.simplify import collect, simplify, separatevars, powsimp, trigsimp # type: ignore
from sympy.utilities import numbered_symbols
from sympy.solvers.solvers import solve
from sympy.matrices import wronskian
from .subscheck import sub_func_doit
from sympy.solvers.ode.ode import get_numbered_constants


def _test_term(coeff, func, order):
    r"""
    Linear Euler ODEs have the form  K*x**order*diff(y(x), x, order) = F(x),
    where K is independent of x and y(x), order>= 0.
    So we need to check that for each term, coeff == K*x**order from
    some K.  We have a few cases, since coeff may have several
    different types.
    """
    # 获取函数的自变量 x
    x = func.args[0]
    # 获取函数的类型 f
    f = func.func

    # 检查阶数是否合法
    if order < 0:
        raise ValueError("order should be greater than 0")
    
    # 如果系数为零，直接返回真
    if coeff == 0:
        return True
    
    # 对于零阶导数的情况
    if order == 0:
        # 检查系数中是否包含自变量 x
        if x in coeff.free_symbols:
            return False
        return True
    
    # 对于系数为乘积的情况
    if coeff.is_Mul:
        # 检查系数是否包含函数 f(x)
        if coeff.has(f(x)):
            return False
        # 检查是否为 x 的阶数次幂
        return x**order in coeff.args
    
    # 对于系数为幂次的情况
    elif coeff.is_Pow:
        return coeff.as_base_exp() == (x, order)
    
    # 对于一阶导数的情况
    elif order == 1:
        return x == coeff
    
    # 其他情况返回假
    return False


def _get_euler_characteristic_eq_sols(eq, func, match_obj):
    r"""
    Returns the solution of homogeneous part of the linear euler ODE and
    the list of roots of characteristic equation.

    The parameter ``match_obj`` is a dict of order:coeff terms, where order is the order
    of the derivative on each term, and coeff is the coefficient of that derivative.

    """
    # 获取函数的自变量 x
    x = func.args[0]
    # 获取函数的类型 f
    f = func.func

    # 首先设置特征方程
    chareq, symbol = S.Zero, Dummy('x')

    # 构建特征方程
    for i in match_obj:
        if i >= 0:
            chareq += (match_obj[i]*diff(x**symbol, x, i)*x**-symbol).expand()

    # 将特征方程转化为多项式
    chareq = Poly(chareq, symbol)
    # 计算特征方程的根
    chareqroots = [rootof(chareq, k) for k in range(chareq.degree())]
    collectterms = []

    # 生成常数的生成器
    constants = list(get_numbered_constants(eq, num=chareq.degree()*2))
    constants.reverse()

    # 创建根到其重数或者特征根的字典
    # 使用 defaultdict 创建一个计数器，用于统计每个根的出现次数
    charroots = defaultdict(int)
    # 遍历 chareqroots 中的每个根，并计数
    for root in chareqroots:
        charroots[root] += 1
    # 初始化 gsol 为零
    gsol = S.Zero
    # 将 log 函数赋给 ln 变量
    ln = log
    # 遍历 charroots 中的每个根及其重数
    for root, multiplicity in charroots.items():
        # 根据根的类型不同执行不同的操作
        for i in range(multiplicity):
            # 如果根是 RootOf 类型
            if isinstance(root, RootOf):
                # 计算 gsol 的贡献，使用常量列表中的值
                gsol += (x**root) * constants.pop()
                # 如果重数不为1，抛出 ValueError 异常
                if multiplicity != 1:
                    raise ValueError("Value should be 1")
                # 将 (0, root, 0) 添加到 collectterms 列表的开头
                collectterms = [(0, root, 0)] + collectterms
            # 如果根是实数
            elif root.is_real:
                # 计算 gsol 的贡献，使用 ln(x) 的幂次和常量列表中的值
                gsol += ln(x)**i*(x**root) * constants.pop()
                # 将 (i, root, 0) 添加到 collectterms 列表的开头
                collectterms = [(i, root, 0)] + collectterms
            # 如果根是复数
            else:
                # 分别取出根的实部和虚部
                reroot = re(root)
                imroot = im(root)
                # 计算 gsol 的贡献，使用 ln(x) 的幂次和常量列表中的值
                gsol += ln(x)**i * (x**reroot) * (
                    constants.pop() * sin(abs(imroot)*ln(x))
                    + constants.pop() * cos(imroot*ln(x)))
                # 将 (i, reroot, imroot) 添加到 collectterms 列表的开头
                collectterms = [(i, reroot, imroot)] + collectterms

    # 将 gsol 转化为 f(x) = gsol 的方程形式
    gsol = Eq(f(x), gsol)

    # 初始化生成解的列表
    gensols = []
    # 遍历 collectterms 中的每个元组 (i, reroot, imroot)
    # 根据 imroot 的值选择使用 sin 或 cos
    for i, reroot, imroot in collectterms:
        # 如果 imroot 为零，将 ln(x)**i * x**reroot 添加到 gensols 中
        if imroot == 0:
            gensols.append(ln(x)**i*x**reroot)
        else:
            # 计算 sin 形式和 cos 形式
            sin_form = ln(x)**i*x**reroot*sin(abs(imroot)*ln(x))
            # 如果 sin 形式已经在 gensols 中，添加 cos 形式；否则添加 sin 形式
            if sin_form in gensols:
                cos_form = ln(x)**i*x**reroot*cos(imroot*ln(x))
                gensols.append(cos_form)
            else:
                gensols.append(sin_form)
    # 返回方程 gsol 和生成解列表 gensols
    return gsol, gensols
# 定义一个辅助函数，用于变分参数法和非齐次 Euler 方程的解法
def _solve_variation_of_parameters(eq, func, roots, homogen_sol, order, match_obj, simplify_flag=True):
    r"""
    Helper function for the method of variation of parameters and nonhomogeneous euler eq.

    See the
    :py:meth:`~sympy.solvers.ode.single.NthLinearConstantCoeffVariationOfParameters`
    docstring for more information on this method.

    The parameter are ``match_obj`` should be a dictionary that has the following
    keys:

    ``list``
    A list of solutions to the homogeneous equation.

    ``sol``
    The general solution.

    """
    # 获取待求解函数和自变量
    f = func.func
    x = func.args[0]
    r = match_obj
    psol = 0
    # 计算 Wronskian 行列式
    wr = wronskian(roots, x)

    if simplify_flag:
        # 简化 Wronskian，以解决一些特定的常微分方程简化问题，如 issue 4662 所述
        wr = simplify(wr)  # 我们需要更好的简化方法
                           # 用于某些常微分方程。例如，查看 issue 4662。
        # 将常见的 sin(x)**2 + cos(x)**2 简化为 1
        wr = trigsimp(wr, deep=True, recursive=True)
    if not wr:
        # 如果 Wronskian 为零，则说明解不线性无关
        raise NotImplementedError("Cannot find " + str(order) +
        " solutions to the homogeneous equation necessary to apply " +
        "variation of parameters to " + str(eq) + " (Wronskian == 0)")
    if len(roots) != order:
        # 如果给定的根数与方程阶数不符合，则无法应用变分参数法
        raise NotImplementedError("Cannot find " + str(order) +
        " solutions to the homogeneous equation necessary to apply " +
        "variation of parameters to " +
        str(eq) + " (number of terms != order)")
    negoneterm = S.NegativeOne**(order)
    # 应用变分参数法的主要循环
    for i in roots:
        psol += negoneterm*Integral(wronskian([sol for sol in roots if sol != i], x)*r[-1]/wr, x)*i/r[order]
        negoneterm *= -1

    if simplify_flag:
        # 对得到的特解进行简化
        psol = simplify(psol)
        psol = trigsimp(psol, deep=True)
    # 返回方程的通解，包括齐次解和特解
    return Eq(f(x), homogen_sol.rhs + psol)


def _get_const_characteristic_eq_sols(r, func, order):
    r"""
    Returns the roots of characteristic equation of constant coefficient
    linear ODE and list of collectterms which is later on used by simplification
    to use collect on solution.

    The parameter `r` is a dict of order:coeff terms, where order is the order of the
    derivative on each term, and coeff is the coefficient of that derivative.

    """
    x = func.args[0]
    # 首先，设置常系数线性常微分方程的特征方程
    chareq, symbol = S.Zero, Dummy('x')

    for i in r.keys():
        if isinstance(i, str) or i < 0:
            pass
        else:
            chareq += r[i]*symbol**i

    chareq = Poly(chareq, symbol)
    # 不能简单地调用 roots，因为对于无法解决的多项式它不返回 rootof
    chareqroots = roots(chareq, multiple=True)
    if len(chareqroots) != order:
        # 如果返回的根数与方程阶数不符，则尝试通过 rootof 函数获取根
        chareqroots = [rootof(chareq, k) for k in range(chareq.degree())]

    # 检查特征方程是否复数解
    chareq_is_complex = not all(i.is_real for i in chareq.all_coeffs())

    # 创建一个字典，记录每个根的重数或特征根
    charroots = defaultdict(int)
    for root in chareqroots:
        charroots[root] += 1
    # 我们需要跟踪项以便在最后运行 collect()。
    # 这对于 constantsimp 的正确工作至关重要。
    collectterms = []  # 用于存储收集的项
    gensols = []        # 存储一般解的列表
    conjugate_roots = []  # 用于防止共轭根重复使用

    # 按照 chareqroots 提供的顺序循环遍历根
    for root in chareqroots:
        # 如果 root 不在 charroots 中，则跳过
        if root not in charroots:
            continue
        
        # 获取 root 的重数
        multiplicity = charroots.pop(root)
        
        # 根据重数循环处理根
        for i in range(multiplicity):
            # 如果方程是复数特征方程
            if chareq_is_complex:
                gensols.append(x**i * exp(root * x))
                collectterms = [(i, root, 0)] + collectterms
                continue
            
            # 分离实部和虚部
            reroot = re(root)
            imroot = im(root)
            
            # 如果实部和虚部都有 atan2 函数
            if imroot.has(atan2) and reroot.has(atan2):
                # 当 re 和 im 停止返回圆形 atan2 用法时移除此条件
                gensols.append(x**i * exp(root * x))
                collectterms = [(i, root, 0)] + collectterms
            else:
                # 如果 root 已经是共轭根中的一个
                if root in conjugate_roots:
                    collectterms = [(i, reroot, imroot)] + collectterms
                    continue
                
                # 如果虚部为 0
                if imroot == 0:
                    gensols.append(x**i * exp(reroot * x))
                    collectterms = [(i, reroot, 0)] + collectterms
                    continue
                
                # 将 root 加入到共轭根列表中
                conjugate_roots.append(conjugate(root))
                
                # 添加正弦和余弦项到通解中
                gensols.append(x**i * exp(reroot * x) * sin(abs(imroot) * x))
                gensols.append(x**i * exp(reroot * x) * cos(imroot * x))

                # 重要的顺序安排
                collectterms = [(i, reroot, imroot)] + collectterms
    
    # 返回生成的解列表和收集的项列表
    return gensols, collectterms
# Ideally these kind of simplification functions shouldn't be part of solvers.
# odesimp should be improved to handle these kind of specific simplifications.
# 定义了一个辅助函数，用于在收集项上获取简化的解。理想情况下，应该由 odesimp 处理这些特定的简化。
def _get_simplified_sol(sol, func, collectterms):
    r"""
    Helper function which collects the solution on
    collectterms. Ideally this should be handled by odesimp.It is used
    only when the simplify is set to True in dsolve.

    The parameter ``collectterms`` is a list of tuple (i, reroot, imroot) where `i` is
    the multiplicity of the root, reroot is real part and imroot being the imaginary part.

    """
    # 获取函数对象和自变量
    f = func.func
    x = func.args[0]
    # 对收集项按照默认排序键排序，并反转排序顺序
    collectterms.sort(key=default_sort_key)
    collectterms.reverse()
    # 确保解只有一个，并且左侧是 f(x)
    assert len(sol) == 1 and sol[0].lhs == f(x)
    # 将解赋值为右侧表达式
    sol = sol[0].rhs
    # 对解进行展开乘法
    sol = expand_mul(sol)
    # 遍历收集项中的每个元组 (i, reroot, imroot)
    for i, reroot, imroot in collectterms:
        # 收集包含 x^i * exp(reroot*x) * sin(abs(imroot)*x) 的项
        sol = collect(sol, x**i*exp(reroot*x)*sin(abs(imroot)*x))
        # 收集包含 x^i * exp(reroot*x) * cos(imroot*x) 的项
        sol = collect(sol, x**i*exp(reroot*x)*cos(imroot*x))
    # 再次遍历收集项，这次只收集 x^i * exp(reroot*x) 的项
    for i, reroot, imroot in collectterms:
        sol = collect(sol, x**i*exp(reroot*x))
    # 对解进行幂简化
    sol = powsimp(sol)
    # 返回一个方程 Eq(f(x), sol)，表示 f(x) = sol
    return Eq(f(x), sol)


def _undetermined_coefficients_match(expr, x, func=None, eq_homogeneous=S.Zero):
    r"""
    Returns a trial function match if undetermined coefficients can be applied
    to ``expr``, and ``None`` otherwise.

    A trial expression can be found for an expression for use with the method
    of undetermined coefficients if the expression is an
    additive/multiplicative combination of constants, polynomials in `x` (the
    independent variable of expr), `\sin(a x + b)`, `\cos(a x + b)`, and
    `e^{a x}` terms (in other words, it has a finite number of linearly
    independent derivatives).

    Note that you may still need to multiply each term returned here by
    sufficient `x` to make it linearly independent with the solutions to the
    homogeneous equation.

    This is intended for internal use by ``undetermined_coefficients`` hints.

    SymPy currently has no way to convert `\sin^n(x) \cos^m(y)` into a sum of
    only `\sin(a x)` and `\cos(b x)` terms, so these are not implemented.  So,
    for example, you will need to manually convert `\sin^2(x)` into `[1 +
    \cos(2 x)]/2` to properly apply the method of undetermined coefficients on
    it.

    Examples
    ========

    >>> from sympy import log, exp
    >>> from sympy.solvers.ode.nonhomogeneous import _undetermined_coefficients_match
    >>> from sympy.abc import x
    >>> _undetermined_coefficients_match(9*x*exp(x) + exp(-x), x)
    {'test': True, 'trialset': {x*exp(x), exp(-x), exp(x)}}
    >>> _undetermined_coefficients_match(log(x), x)
    {'test': False}

    """
    # 使用指定参数合并指数表达式（如果有的话）
    a = Wild('a', exclude=[x])
    b = Wild('b', exclude=[x])
    expr = powsimp(expr, combine='exp')  # exp(x)*exp(2*x + 1) => exp(3*x + 1)
    # 初始化返回的字典
    retdict = {}
    # 定义一个函数 `_test_term`，用于检测表达式是否符合待定系数的适当形式，返回布尔值
    def _test_term(expr, x) -> bool:
        # 如果表达式中不包含变量 x，则返回 True
        if not expr.has(x):
            return True
        # 如果表达式是加法类型，则递归检查每个子表达式
        if expr.is_Add:
            return all(_test_term(i, x) for i in expr.args)
        # 如果表达式是乘法类型
        if expr.is_Mul:
            # 如果表达式包含 sin 或 cos 函数
            if expr.has(sin, cos):
                foundtrig = False
                # 确保在参数中只有一个三角函数，详见文档字符串说明
                for i in expr.args:
                    if i.has(sin, cos):
                        if foundtrig:
                            return False
                        else:
                            foundtrig = True
            # 对乘法表达式的每个因子递归进行检查
            return all(_test_term(i, x) for i in expr.args)
        # 如果表达式是函数类型
        if expr.is_Function:
            # 函数为 sin, cos, exp, sinh, cosh 中的一个，并且参数符合 a*x + b 的形式
            return expr.func in (sin, cos, exp, sinh, cosh) and \
                   bool(expr.args[0].match(a*x + b))
        # 如果表达式是幂次类型，且底数是符号并且指数是非负整数
        if expr.is_Pow and expr.base.is_Symbol and expr.exp.is_Integer and \
                expr.exp >= 0:
            return True
        # 如果表达式是幂次类型，且底数是数字，检查指数是否符合 a*x + b 的形式
        if expr.is_Pow and expr.base.is_number:
            return bool(expr.exp.match(a*x + b))
        # 对于其它情况，表达式应为符号或者是数字
        return expr.is_Symbol or bool(expr.is_number)
    def _get_trial_set(expr, x, exprs=set()):
        r"""
        Returns a set of trial terms for undetermined coefficients.

        The idea behind undetermined coefficients is that the terms expression
        repeat themselves after a finite number of derivatives, except for the
        coefficients (they are linearly dependent).  So if we collect these,
        we should have the terms of our trial function.
        """
        def _remove_coefficient(expr, x):
            r"""
            Returns the expression without a coefficient.

            Similar to expr.as_independent(x)[1], except it only works
            multiplicatively.
            """
            term = S.One
            if expr.is_Mul:
                for i in expr.args:
                    if i.has(x):
                        term *= i
            elif expr.has(x):
                term = expr
            return term

        # Expand the expression to handle all multiplication terms
        expr = expand_mul(expr)
        # If the expression is an addition of terms
        if expr.is_Add:
            for term in expr.args:
                # Check if the term without coefficient is already in exprs
                if _remove_coefficient(term, x) in exprs:
                    pass
                else:
                    # Add the term without coefficient to exprs
                    exprs.add(_remove_coefficient(term, x))
                    # Recursively call _get_trial_set for the current term
                    exprs = exprs.union(_get_trial_set(term, x, exprs))
        else:
            # If the expression is not an addition of terms
            term = _remove_coefficient(expr, x)
            tmpset = exprs.union({term})
            oldset = set()
            while tmpset != oldset:
                # Iteratively differentiate and remove coefficients
                oldset = tmpset.copy()
                expr = expr.diff(x)
                term = _remove_coefficient(expr, x)
                if term.is_Add:
                    # If term is an addition, recursively call _get_trial_set
                    tmpset = tmpset.union(_get_trial_set(term, x, tmpset))
                else:
                    tmpset.add(term)
            exprs = tmpset
        # Return the set of trial terms
        return exprs

    def is_homogeneous_solution(term):
        r""" This function checks whether the given trialset contains any root
            of homogeneous equation"""
        # Check if the expanded substituted function is zero
        return expand(sub_func_doit(eq_homogeneous, func, term)).is_zero

    # Perform a test on the expression to determine a trial term
    retdict['test'] = _test_term(expr, x)
    # Check if the test was successful
    if retdict['test']:
        # Try to generate a list of trial solutions that will have the
        # undetermined coefficients. Note that if any of these are not linearly
        # independent with any of the solutions to the homogeneous equation,
        # then they will need to be multiplied by sufficient x to make them so.
        # This function DOES NOT do that (it doesn't even look at the
        # homogeneous equation).
        temp_set = set()
        # Iterate over each term of the expression
        for i in Add.make_args(expr):
            # Generate a set of trial terms for the current term
            act = _get_trial_set(i, x)
            # Check if there exists a homogeneous solution for any term
            if eq_homogeneous is not S.Zero:
                while any(is_homogeneous_solution(ts) for ts in act):
                    # Multiply each term by x until no homogeneous solution exists
                    act = {x*ts for ts in act}
            # Union the generated trial terms with the temporary set
            temp_set = temp_set.union(act)

        # Store the set of trial solutions in retdict
        retdict['trialset'] = temp_set
    # Return the updated dictionary
    return retdict
# 定义解决非确定系数方法的辅助函数，用于求解常系数线性微分方程
def _solve_undetermined_coefficients(eq, func, order, match, trialset):
    r"""
    Helper function for the method of undetermined coefficients.

    See the
    :py:meth:`~sympy.solvers.ode.single.NthLinearConstantCoeffUndeterminedCoefficients`
    docstring for more information on this method.

    The parameter ``trialset`` is the set of trial functions as returned by
    ``_undetermined_coefficients_match()['trialset']``.

    The parameter ``match`` should be a dictionary that has the following
    keys:

    ``list``
    A list of solutions to the homogeneous equation.

    ``sol``
    The general solution.

    """
    r = match  # 将匹配结果存储在变量 r 中
    coeffs = numbered_symbols('a', cls=Dummy)  # 创建一个以'a'开头的符号生成器
    coefflist = []  # 存储系数列表的空列表
    gensols = r['list']  # 从匹配结果中获取齐次方程的解列表
    gsol = r['sol']  # 从匹配结果中获取齐次方程的通解
    f = func.func  # 获取函数对象
    x = func.args[0]  # 获取自变量对象

    # 如果齐次方程的解的数量不等于给定的阶数，抛出未实现的错误
    if len(gensols) != order:
        raise NotImplementedError("Cannot find " + str(order) +
        " solutions to the homogeneous equation necessary to apply" +
        " undetermined coefficients to " + str(eq) +
        " (number of terms != order)")

    trialfunc = 0  # 初始化试探函数为零
    # 遍历试探函数集合，并为每个试探函数分配一个系数
    for i in trialset:
        c = next(coeffs)  # 从系数生成器中获取下一个系数符号
        coefflist.append(c)  # 将生成的系数符号添加到系数列表中
        trialfunc += c*i  # 构建试探函数的表达式

    # 将试探函数代入微分方程，生成一组方程
    eqs = sub_func_doit(eq, f(x), trialfunc)

    coeffsdict = dict(list(zip(trialset, [0]*(len(trialset) + 1))))  # 创建系数字典，初始化系数值为零

    eqs = _mexpand(eqs)  # 对方程组进行展开和简化处理

    # 遍历方程组中的每一项，将其分离为符号和系数，并将结果存储到系数字典中
    for i in Add.make_args(eqs):
        s = separatevars(i, dict=True, symbols=[x])  # 将表达式分离为变量和系数
        if coeffsdict.get(s[x]):
            coeffsdict[s[x]] += s['coeff']  # 更新系数字典中变量对应的系数值
        else:
            coeffsdict[s[x]] = s['coeff']  # 添加新的变量及其系数到系数字典中

    # 解方程组，求解出系数值
    coeffvals = solve(list(coeffsdict.values()), coefflist)

    # 如果无法解出系数值，抛出未实现的错误
    if not coeffvals:
        raise NotImplementedError(
            "Could not solve `%s` using the "
            "method of undetermined coefficients "
            "(unable to solve for coefficients)." % eq)

    # 计算得到的试探函数的特解，并将其与齐次方程的通解相加，得到最终的解
    psol = trialfunc.subs(coeffvals)

    return Eq(f(x), gsol.rhs + psol)  # 返回微分方程的解
```