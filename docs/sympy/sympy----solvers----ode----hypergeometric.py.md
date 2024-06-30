# `D:\src\scipysrc\sympy\sympy\solvers\ode\hypergeometric.py`

```
r'''
This module contains the implementation of the 2nd_hypergeometric hint for
dsolve. This is an incomplete implementation of the algorithm described in [1].
The algorithm solves 2nd order linear ODEs of the form

.. math:: y'' + A(x) y' + B(x) y = 0\text{,}

where `A` and `B` are rational functions. The algorithm should find any
solution of the form

.. math:: y = P(x) _pF_q(..; ..;\frac{\alpha x^k + \beta}{\gamma x^k + \delta})\text{,}

where pFq is any of 2F1, 1F1 or 0F1 and `P` is an "arbitrary function".
Currently only the 2F1 case is implemented in SymPy but the other cases are
described in the paper and could be implemented in future (contributions
welcome!).

References
==========

.. [1] L. Chan, E.S. Cheb-Terrab, Non-Liouvillian solutions for second order
       linear ODEs, (2004).
       https://arxiv.org/abs/math-ph/0402063
'''

from sympy.core import S, Pow
from sympy.core.function import expand
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol, Wild
from sympy.functions import exp, sqrt, hyper
from sympy.integrals import Integral
from sympy.polys import roots, gcd
from sympy.polys.polytools import cancel, factor
from sympy.simplify import collect, simplify, logcombine # type: ignore
from sympy.simplify.powsimp import powdenest
from sympy.solvers.ode.ode import get_numbered_constants


def match_2nd_hypergeometric(eq, func):
    # Extract the independent variable from the function
    x = func.args[0]
    # Differentiate the function with respect to x
    df = func.diff(x)
    # Define wildcards for coefficients of the second order ODE
    a3 = Wild('a3', exclude=[func, func.diff(x), func.diff(x, 2)])
    b3 = Wild('b3', exclude=[func, func.diff(x), func.diff(x, 2)])
    c3 = Wild('c3', exclude=[func, func.diff(x), func.diff(x, 2)])
    # Define the differential equation in its general form
    deq = a3*(func.diff(x, 2)) + b3*df + c3*func
    # Match the equation with the differential equation form
    r = collect(eq, [func.diff(x, 2), func.diff(x), func]).match(deq)
    # Check if all matched values are polynomials; if not, expand and try matching again
    if r and not all(val.is_polynomial() for val in r.values()):
        n, d = eq.as_numer_denom()
        eq = expand(n)
        r = collect(eq, [func.diff(x, 2), func.diff(x), func]).match(deq)

    # If a valid match is found and the coefficient a3 is nonzero, compute and return A and B
    if r and r[a3] != 0:
        A = cancel(r[b3] / r[a3])
        B = cancel(r[c3] / r[a3])
        return [A, B]
    else:
        return []


def equivalence_hypergeometric(A, B, func):
    # This method for finding the equivalence is only for 2F1 type.
    # We can extend it for 1F1 and 0F1 type also.
    x = func.args[0]

    # Compute the "shifted invariant" J1 of the equation
    I1 = factor(cancel(A.diff(x)/2 + A**2/4 - B))
    J1 = factor(cancel(x**2 * I1 + S(1)/4))

    # Separate numerator and denominator of J1
    num, dem = J1.as_numer_denom()

    # Apply power simplification to expand powers of x in J1
    num = powdenest(expand(num))
    dem = powdenest(expand(dem))

    # Compute the different powers of x in J1 to find the value of k
    # k is such that J1 = x**k * J0(x**k), where J0 has integer powers
    # This computation assists in determining the form of the hypergeometric function
    # 定义一个函数 _power_counting，用于统计给定数值或表达式中的幂次
    def _power_counting(num):
        # 初始化一个集合 _pow，用于存储不重复的幂次值，初始包含 0
        _pow = {0}
        # 遍历 num 中的每一个值或表达式
        for val in num:
            # 检查当前值 val 是否包含变量 x
            if val.has(x):
                # 如果 val 是 Pow 对象，并且其基数（底数）与 x 相同，则添加其指数到 _pow
                if isinstance(val, Pow) and val.as_base_exp()[0] == x:
                    _pow.add(val.as_base_exp()[1])
                # 如果 val 等于 x，则添加其指数到 _pow
                elif val == x:
                    _pow.add(val.as_base_exp()[1])
                # 否则递归调用 _power_counting 函数，更新 _pow
                else:
                    _pow.update(_power_counting(val.args))
        # 返回统计完成的 _pow 集合
        return _pow

    # 计算 num 和 dem 中的幂次
    pow_num = _power_counting((num, ))
    pow_dem = _power_counting((dem, ))
    # 将 pow_num 中的幂次更新到 pow_dem 中
    pow_dem.update(pow_num)

    # 将 pow_dem 赋值给 _pow
    _pow = pow_dem
    # 计算 _pow 的最大公约数，并赋值给 k
    k = gcd(_pow)

    # 计算给定方程的 I0
    I0 = powdenest(simplify(factor(((J1/k**2) - S(1)/4)/((x**k)**2))), force=True)
    # 使用 x**(1/k) 替换 x 后，进一步简化并因式分解 I0
    I0 = factor(cancel(powdenest(I0.subs(x, x**(S(1)/k)), force=True)))

    # 在这一点之前，I0 和 J1 可能是 x 的函数，但替换 x 为 x**(1/k) 后，I0 应为 x 的有理函数，
    # 否则不能使用超几何求解器。注意 k 可以是非整数有理数，如 2/7。
    if not I0.is_rational_function(x):
        return None

    # 将 I0 分解为分子和分母
    num, dem = I0.as_numer_denom()

    # 计算 num 中的最大幂次
    max_num_pow = max(_power_counting((num, )))
    # 获取 dem 的参数列表
    dem_args = dem.args
    # 初始化存放奇点的列表和幂次的列表
    sing_point = []
    dem_pow = []
    # 计算 I0 的奇点
    for arg in dem_args:
        # 如果参数 arg 包含 x
        if arg.has(x):
            # 如果 arg 是 Pow 类型，则将其指数添加到 dem_pow，并获取其根的第一个键作为奇点
            if isinstance(arg, Pow):
                dem_pow.append(arg.as_base_exp()[1])
                sing_point.append(list(roots(arg.as_base_exp()[0], x).keys())[0])
            # 否则，arg 是 (x-a) 类型，将其指数添加到 dem_pow，并获取其根的第一个键作为奇点
            else:
                dem_pow.append(arg.as_base_exp()[1])
                sing_point.append(list(roots(arg, x).keys())[0])

    # 对 dem_pow 进行排序
    dem_pow.sort()

    # 检查最大幂次与 dem_pow 是否等价，如果是 "2F1"，则返回相关信息，否则返回 None
    if equivalence(max_num_pow, dem_pow) == "2F1":
        return {'I0':I0, 'k':k, 'sing_point':sing_point, 'type':"2F1"}
    else:
        return None
# 匹配第二类超几何函数 2F1 的通用形式，返回相关参数的字典
def match_2nd_2F1_hypergeometric(I, k, sing_point, func):
    # 获取函数中的自变量 x
    x = func.args[0]
    # 定义用于匹配的通配符
    a = Wild("a")
    b = Wild("b")
    c = Wild("c")
    t = Wild("t")
    s = Wild("s")
    r = Wild("r")
    alpha = Wild("alpha")
    beta = Wild("beta")
    gamma = Wild("gamma")
    delta = Wild("delta")
    
    # 定义标准 2F1 方程的 I0
    I0 = ((a-b+1)*(a-b-1)*x**2 + 2*((1-a-b)*c + 2*a*b)*x + c*(c-2))/(4*x**2*(x-1)**2)
    
    # 如果奇点不是 [0, 1]，则生成用于寻找 Mobius 变换的方程
    if sing_point != [0, 1]:
        eqs = []
        sing_eqs = [-beta/alpha, -delta/gamma, (delta-beta)/(alpha-gamma)]
        # 为寻找 Mobius 变换生成方程
        for i in range(3):
            if i<len(sing_point):
                eqs.append(Eq(sing_eqs[i], sing_point[i]))
            else:
                eqs.append(Eq(1/sing_eqs[i], 0))
        
        # 解上述方程以获得 Mobius 变换
        _beta = -alpha*sing_point[0]
        _delta = -gamma*sing_point[1]
        _gamma = alpha
        if len(sing_point) == 3:
            _gamma = (_beta + sing_point[2]*alpha)/(sing_point[2] - sing_point[1])
        
        # 计算和简化 Mobius 变换
        mob = (alpha*x + beta)/(gamma*x + delta)
        mob = mob.subs(beta, _beta)
        mob = mob.subs(delta, _delta)
        mob = mob.subs(gamma, _gamma)
        mob = cancel(mob)
        
        # 计算另一个变量 t
        t = (beta - delta*x)/(gamma*x - alpha)
        t = cancel(((t.subs(beta, _beta)).subs(delta, _delta)).subs(gamma, _gamma))
    else:
        mob = x
        t = x
    
    # 应用 Mobius 变换将 I 转换为 I0
    I = I.subs(x, t)
    I = I*(t.diff(x))**2
    I = factor(I)
    
    # 初始化一个字典来存储参数
    dict_I = {x**2:0, x:0, 1:0}
    
    # 将 I0 分子和分母分开
    I0_num, I0_dem = I0.as_numer_denom()
    
    # 收集标准方程 (x**2, x) 的系数
    dict_I0 = {x**2:s**2 - 1, x:(2*(1-r)*c + (r+s)*(r-s)), 1:c*(c-2)}
    
    # 收集给定方程的 I0 的 (x**2, x) 的系数
    dict_I.update(collect(expand(cancel(I*I0_dem)), [x**2, x], evaluate=False))
    
    eqs = []
    
    # 比较不同 x 的幂次的系数，以找到标准方程的参数值
    for key in [x**2, x, 1]:
        eqs.append(Eq(dict_I[key], dict_I0[key]))
    
    # 计算参数 a, b, c 的可能根
    _c = 1 - factor(sqrt(1+eqs[2].lhs))
    if not _c.has(Symbol):
        _c = min(list(roots(eqs[2], c)))
    _s = factor(sqrt(eqs[0].lhs + 1))
    _r = _c - factor(sqrt(_c**2 + _s**2 + eqs[1].lhs - 2*_c))
    _a = (_r + _s)/2
    _b = (_r - _s)/2
    
    # 构建参数字典
    rn = {'a':simplify(_a), 'b':simplify(_b), 'c':simplify(_c), 'k':k, 'mobius':mob, 'type':"2F1"}
    return rn


# 此函数用于检查与 2F1 类型方程的等价性
def equivalence(max_num_pow, dem_pow):
    # max_num_pow 是分子中 x 的最大幂次的值
    # dem_pow 是形式为 (a*x + b) 的不同因子的幂次列表
    # 参考文献为 L. Chan 和 E.S. Cheb-Terrab 的论文《Non-Liouvillian solutions for second order linear ODEs》中的表格1。
    # 我们可以扩展到 1F1 和 0F1 类型。
    
    # 如果 max_num_pow 等于 2：
    if max_num_pow == 2:
        # 如果 dem_pow 是 [[2, 2], [2, 2, 2]] 中的一种：
        if dem_pow in [[2, 2], [2, 2, 2]]:
            # 返回字符串 "2F1"
            return "2F1"
    
    # 如果 max_num_pow 等于 1：
    elif max_num_pow == 1:
        # 如果 dem_pow 是 [[1, 2, 2], [2, 2, 2], [1, 2], [2, 2]] 中的一种：
        if dem_pow in [[1, 2, 2], [2, 2, 2], [1, 2], [2, 2]]:
            # 返回字符串 "2F1"
            return "2F1"
    
    # 如果 max_num_pow 等于 0：
    elif max_num_pow == 0:
        # 如果 dem_pow 是 [[1, 1, 2], [2, 2], [1, 2, 2], [1, 1], [2], [1, 2], [2, 2]] 中的一种：
        if dem_pow in [[1, 1, 2], [2, 2], [1, 2, 2], [1, 1], [2], [1, 2], [2, 2]]:
            # 返回字符串 "2F1"
            return "2F1"
    
    # 如果以上条件均不满足，则返回 None
    return None
# 定义一个函数，用于求解超几何函数的解
def get_sol_2F1_hypergeometric(eq, func, match_object):
    # 提取函数的自变量
    x = func.args[0]
    # 导入超级展开函数和多项式工具
    from sympy.simplify.hyperexpand import hyperexpand
    from sympy.polys.polytools import factor
    # 获取方程中的两个编号常数
    C0, C1 = get_numbered_constants(eq, num=2)
    # 提取匹配对象中的参数
    a = match_object['a']
    b = match_object['b']
    c = match_object['c']
    A = match_object['A']

    # 初始化解为 None
    sol = None

    # 根据 c 是否为整数来判断不同情况下的解
    if c.is_integer == False:
        # 情况1：c 不是整数
        sol = C0*hyper([a, b], [c], x) + C1*hyper([a-c+1, b-c+1], [2-c], x)*x**(1-c)
    elif c == 1:
        # 情况2：c 等于1
        y2 = Integral(exp(Integral((-(a+b+1)*x + c)/(x**2-x), x))/(hyperexpand(hyper([a, b], [c], x))**2), x)*hyper([a, b], [c], x)
        sol = C0*hyper([a, b], [c], x) + C1*y2
    elif (c-a-b).is_integer == False:
        # 情况3：c-a-b 不是整数
        sol = C0*hyper([a, b], [1+a+b-c], 1-x) + C1*hyper([c-a, c-b], [1+c-a-b], 1-x)*(1-x)**(c-a-b)

    # 如果有解
    if sol:
        # 对解进行变换
        subs = match_object['mobius']
        dtdx = simplify(1/(subs.diff(x)))
        _B = ((a + b + 1)*x - c).subs(x, subs)*dtdx
        _B = factor(_B + ((x**2 -x).subs(x, subs))*(dtdx.diff(x)*dtdx))
        _A = factor((x**2 - x).subs(x, subs)*(dtdx**2))
        e = exp(logcombine(Integral(cancel(_B/(2*_A)), x), force=True))
        sol = sol.subs(x, match_object['mobius'])
        sol = sol.subs(x, x**match_object['k'])
        e = e.subs(x, x**match_object['k'])

        # 如果 A 不为零
        if not A.is_zero:
            e1 = Integral(A/2, x)
            e1 = exp(logcombine(e1, force=True))
            sol = cancel((e/e1)*x**((-match_object['k']+1)/2))*sol
            sol = Eq(func, sol)
            return sol

        # 如果 A 为零
        sol = cancel((e)*x**((-match_object['k']+1)/2))*sol
        sol = Eq(func, sol)
    
    # 返回最终的解
    return sol
```