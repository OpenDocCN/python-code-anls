# `D:\src\scipysrc\sympy\sympy\solvers\deutils.py`

```
"""Utility functions for classifying and solving
ordinary and partial differential equations.

Contains
========
_preprocess
ode_order
_desolve

"""
from sympy.core import Pow  # 导入 Pow 类
from sympy.core.function import Derivative, AppliedUndef  # 导入 Derivative 和 AppliedUndef 类
from sympy.core.relational import Equality  # 导入 Equality 类
from sympy.core.symbol import Wild  # 导入 Wild 类

def _preprocess(expr, func=None, hint='_Integral'):
    """Prepare expr for solving by making sure that differentiation
    is done so that only func remains in unevaluated derivatives and
    (if hint does not end with _Integral) that doit is applied to all
    other derivatives. If hint is None, do not do any differentiation.
    (Currently this may cause some simple differential equations to
    fail.)

    In case func is None, an attempt will be made to autodetect the
    function to be solved for.

    >>> from sympy.solvers.deutils import _preprocess
    >>> from sympy import Derivative, Function
    >>> from sympy.abc import x, y, z
    >>> f, g = map(Function, 'fg')

    If f(x)**p == 0 and p>0 then we can solve for f(x)=0
    >>> _preprocess((f(x).diff(x)-4)**5, f(x))
    (Derivative(f(x), x) - 4, f(x))

    Apply doit to derivatives that contain more than the function
    of interest:

    >>> _preprocess(Derivative(f(x) + x, x))
    (Derivative(f(x), x) + 1, f(x))

    Do others if the differentiation variable(s) intersect with those
    of the function of interest or contain the function of interest:

    >>> _preprocess(Derivative(g(x), y, z), f(y))
    (0, f(y))
    >>> _preprocess(Derivative(f(y), z), f(y))
    (0, f(y))

    Do others if the hint does not end in '_Integral' (the default
    assumes that it does):

    >>> _preprocess(Derivative(g(x), y), f(x))
    (Derivative(g(x), y), f(x))
    >>> _preprocess(Derivative(f(x), y), f(x), hint='')
    (0, f(x))

    Do not do any derivatives if hint is None:

    >>> eq = Derivative(f(x) + 1, x) + Derivative(f(x), y)
    >>> _preprocess(eq, f(x), hint=None)
    (Derivative(f(x) + 1, x) + Derivative(f(x), y), f(x))

    If it's not clear what the function of interest is, it must be given:

    >>> eq = Derivative(f(x) + g(x), x)
    >>> _preprocess(eq, g(x))
    (Derivative(f(x), x) + Derivative(g(x), x), g(x))
    >>> try: _preprocess(eq)
    ... except ValueError: print("A ValueError was raised.")
    A ValueError was raised.

    """
    if isinstance(expr, Pow):
        # 如果表达式是幂次方，则尝试解析指数大于零的情况
        if (expr.exp).is_positive:
            expr = expr.base
    derivs = expr.atoms(Derivative)  # 获取表达式中所有的 Derivative 对象
    if not func:
        funcs = set().union(*[d.atoms(AppliedUndef) for d in derivs])
        # 自动检测要解决的函数
        if len(funcs) != 1:
            raise ValueError('The function cannot be '
                'automatically detected for %s.' % expr)
        func = funcs.pop()
    fvars = set(func.args)  # 获取函数的参数集合
    if hint is None:
        return expr, func
    reps = [(d, d.doit()) for d in derivs if not hint.endswith('_Integral') or
            d.has(func) or set(d.variables) & fvars]
    # 返回替换列表，其中包含对于提示 hint 未以 '_Integral' 结尾或者包含函数相关变量的 Derivative 对象
   `
    # 使用 subs 方法将表达式 expr 中的符号替换为 reps 中的值
    eq = expr.subs(reps)
    # 返回替换后的表达式 eq 和函数 func
    return eq, func
# 计算给定微分方程关于给定函数的阶数

def ode_order(expr, func):
    """
    Returns the order of a given differential
    equation with respect to func.

    This function is implemented recursively.

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.solvers.deutils import ode_order
    >>> from sympy.abc import x
    >>> f, g = map(Function, ['f', 'g'])
    >>> ode_order(f(x).diff(x, 2) + f(x).diff(x)**2 +
    ... f(x).diff(x), f(x))
    2
    >>> ode_order(f(x).diff(x, 2) + g(x).diff(x, 3), f(x))
    2
    >>> ode_order(f(x).diff(x, 2) + g(x).diff(x, 3), g(x))
    3

    """
    # 创建一个通配符 Wild 对象，排除 func
    a = Wild('a', exclude=[func])
    # 如果表达式能够匹配到通配符 a，则返回阶数 0
    if expr.match(a):
        return 0

    # 如果表达式是 Derivative 类型
    if isinstance(expr, Derivative):
        # 如果 Derivative 对象的第一个参数是 func
        if expr.args[0] == func:
            # 返回导数的变量数作为阶数
            return len(expr.variables)
        else:
            # 否则，递归地计算所有参数的最大阶数
            args = expr.args[0].args
            rv = len(expr.variables)
            if args:
                rv += max(ode_order(_, func) for _ in args)
            return rv
    else:
        # 对表达式中所有参数进行递归调用，返回最大阶数
        return max(ode_order(_, func) for _ in expr.args) if expr.args else 0


# 这是一个辅助函数，用于在 ode 和 pde 模块中的 dsolve 和 pdsolve 函数中使用

def _desolve(eq, func=None, hint="default", ics=None, simplify=True, *, prep=True, **kwargs):
    """This is a helper function to dsolve and pdsolve in the ode
    and pde modules.

    If the hint provided to the function is "default", then a dict with
    the following keys are returned

    'func'    - It provides the function for which the differential equation
                has to be solved. This is useful when the expression has
                more than one function in it.

    'default' - The default key as returned by classifier functions in ode
                and pde.py

    'hint'    - The hint given by the user for which the differential equation
                is to be solved. If the hint given by the user is 'default',
                then the value of 'hint' and 'default' is the same.

    'order'   - The order of the function as returned by ode_order

    'match'   - It returns the match as given by the classifier functions, for
                the default hint.

    If the hint provided to the function is not "default" and is not in
    ('all', 'all_Integral', 'best'), then a dict with the above mentioned keys
    is returned along with the keys which are returned when dict in
    classify_ode or classify_pde is set True

    If the hint given is in ('all', 'all_Integral', 'best'), then this function
    returns a nested dict, with the keys, being the set of classified hints
    returned by classifier functions, and the values being the dict of form
    as mentioned above.

    Key 'eq' is a common key to all the above mentioned hints which returns an
    expression if eq given by user is an Equality.

    See Also
    ========
    classify_ode(ode.py)
    classify_pde(pde.py)
    """
    # 如果输入是一个等式，将其转换为左侧减右侧的形式
    if isinstance(eq, Equality):
        eq = eq.lhs - eq.rhs

    # 预处理方程式并且如果未提供函数，则查找函数
    if prep or func is None:
        eq, func = _preprocess(eq, func)
        prep = False
    # 从 kwargs 中获取 'type' 参数，用于标识调用 solve 函数时是普通微分方程还是偏微分方程
    type = kwargs.get('type', None)
    # 从 kwargs 中获取 'xi' 和 'eta' 参数
    xi = kwargs.get('xi')
    eta = kwargs.get('eta')
    # 从 kwargs 中获取 'x0' 参数，默认为 0
    x0 = kwargs.get('x0', 0)
    # 从 kwargs 中获取 'n' 参数，表示项数
    terms = kwargs.get('n')

    # 如果 type 为 'ode'，则引入求解普通微分方程的相关函数和分类器
    if type == 'ode':
        from sympy.solvers.ode import classify_ode, allhints
        classifier = classify_ode
        string = 'ODE '
        dummy = ''

    # 如果 type 为 'pde'，则引入求解偏微分方程的相关函数和分类器
    elif type == 'pde':
        from sympy.solvers.pde import classify_pde, allhints
        classifier = classify_pde
        string = 'PDE '
        dummy = 'p'

    # 如果 kwargs 中没有明确指定不进行分类，则调用分类器函数进行分类
    if kwargs.get('classify', True):
        # 使用分类器对方程进行分类，并返回分类的提示和信息
        hints = classifier(eq, func, dict=True, ics=ics, xi=xi, eta=eta,
                           n=terms, x0=x0, hint=hint, prep=prep)

    else:
        # 如果 kwargs 指定不进行分类，则返回指定的分类提示和信息
        hints = kwargs.get('hint',
                           {'default': hint,
                            hint: kwargs['match'],
                            'order': kwargs['order']})

    # 如果没有默认提示，则抛出值错误异常
    if not hints['default']:
        # 如果提示不在 allhints 中，或者提示不是 'default'，则抛出值错误异常
        if hint not in allhints and hint != 'default':
            raise ValueError("Hint not recognized: " + hint)
        # 如果提示不在有序提示列表中，并且提示不是 'default'，则抛出值错误异常
        elif hint not in hints['ordered_hints'] and hint != 'default':
            raise ValueError(string + str(eq) + " does not match hint " + hint)
        # 如果方程的阶数为 0，则抛出值错误异常
        elif hints['order'] == 0:
            raise ValueError(
                str(eq) + " is not a solvable differential equation in " + str(func))
        # 否则，抛出未实现错误异常
        else:
            raise NotImplementedError(dummy + "solve" + ": Cannot solve " + str(eq))

    # 如果提示为 'default'，则调用 _desolve() 函数进行求解
    if hint == 'default':
        return _desolve(eq, func, ics=ics, hint=hints['default'], simplify=simplify,
                        prep=prep, x0=x0, classify=False, order=hints['order'],
                        match=hints[hints['default']], xi=xi, eta=eta, n=terms, type=type)
    # 如果提示为 'all', 'all_Integral', 'best' 中的一种，则执行以下代码块
    retdict = {}  # 初始化空字典，用于存储解的结果
    gethints = set(hints) - {'order', 'default', 'ordered_hints'}  # 获取所有提示的集合，排除 'order', 'default', 'ordered_hints'

    # 如果提示为 'all_Integral'，则进入特殊处理逻辑
    if hint == 'all_Integral':
        # 去除以 '_Integral' 结尾的提示后，重新更新 gethints 集合
        for i in hints:
            if i.endswith('_Integral'):
                gethints.remove(i[:-len('_Integral')])
        
        # 特殊情况下需要排除的提示列表
        for k in ["1st_homogeneous_coeff_best", "1st_power_series",
                  "lie_group", "2nd_power_series_ordinary", "2nd_power_series_regular"]:
            # 如果 k 存在于 gethints 中，则移除它
            if k in gethints:
                gethints.remove(k)
    
    # 遍历剩余的提示集合，调用 _desolve 函数解决方程
    for i in gethints:
        sol = _desolve(eq, func, ics=ics, hint=i, x0=x0, simplify=simplify, prep=prep,
                       classify=False, n=terms, order=hints['order'], match=hints[i], type=type)
        retdict[i] = sol  # 将解添加到结果字典中对应的提示键下
    
    retdict['all'] = True  # 设置 'all' 键为 True，表示包含所有解
    retdict['eq'] = eq  # 将输入方程存储在结果字典中
    return retdict  # 返回包含解和其他信息的字典作为结果

elif hint not in allhints:  # 如果提示不在已知的提示集合中
    raise ValueError("Hint not recognized: " + hint)  # 抛出值错误，提示未被识别

elif hint not in hints:  # 如果提示不在给定的提示列表中
    raise ValueError(string + str(eq) + " does not match hint " + hint)  # 抛出值错误，显示方程与提示不匹配

else:
    hints['hint'] = hint  # 如果以上条件都不满足，则将提示键加入提示字典以标识所需的提示类型

hints.update({'func': func, 'eq': eq})  # 更新提示字典，添加函数和方程键
return hints  # 返回更新后的提示字典作为结果
```