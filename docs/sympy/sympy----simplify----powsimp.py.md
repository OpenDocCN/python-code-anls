# `D:\src\scipysrc\sympy\sympy\simplify\powsimp.py`

```
from collections import defaultdict
from functools import reduce
from math import prod

from sympy.core.function import expand_log, count_ops, _coeff_isneg
from sympy.core import sympify, Basic, Dummy, S, Add, Mul, Pow, expand_mul, factor_terms
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.numbers import Integer, Rational
from sympy.core.mul import _keep_coeff
from sympy.core.rules import Transform
from sympy.functions import exp_polar, exp, log, root, polarify, unpolarify
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys import lcm, gcd
from sympy.ntheory.factor_ import multiplicity

# 导入必要的模块和函数

def powsimp(expr, deep=False, combine='all', force=False, measure=count_ops):
    """
    Reduce expression by combining powers with similar bases and exponents.

    Explanation
    ===========

    If ``deep`` is ``True`` then powsimp() will also simplify arguments of
    functions. By default ``deep`` is set to ``False``.

    If ``force`` is ``True`` then bases will be combined without checking for
    assumptions, e.g. sqrt(x)*sqrt(y) -> sqrt(x*y) which is not true
    if x and y are both negative.

    You can make powsimp() only combine bases or only combine exponents by
    changing combine='base' or combine='exp'.  By default, combine='all',
    which does both.  combine='base' will only combine::

         a   a          a                          2x      x
        x * y  =>  (x*y)   as well as things like 2   =>  4

    and combine='exp' will only combine
    ::

         a   b      (a + b)
        x * x  =>  x

    combine='exp' will strictly only combine exponents in the way that used
    to be automatic.  Also use deep=True if you need the old behavior.

    When combine='all', 'exp' is evaluated first.  Consider the first
    example below for when there could be an ambiguity relating to this.
    This is done so things like the second example can be completely
    combined.  If you want 'base' combined first, do something like
    powsimp(powsimp(expr, combine='base'), combine='exp').

    Examples
    ========

    >>> from sympy import powsimp, exp, log, symbols
    >>> from sympy.abc import x, y, z, n
    >>> powsimp(x**y*x**z*y**z, combine='all')
    x**(y + z)*y**z
    >>> powsimp(x**y*x**z*y**z, combine='exp')
    x**(y + z)*y**z
    >>> powsimp(x**y*x**z*y**z, combine='base', force=True)
    x**y*(x*y)**z

    >>> powsimp(x**z*x**y*n**z*n**y, combine='all', force=True)
    (n*x)**(y + z)
    >>> powsimp(x**z*x**y*n**z*n**y, combine='exp')
    n**(y + z)*x**(y + z)
    >>> powsimp(x**z*x**y*n**z*n**y, combine='base', force=True)
    (n*x)**y*(n*x)**z

    >>> x, y = symbols('x y', positive=True)
    >>> powsimp(log(exp(x)*exp(y)))
    log(exp(x)*exp(y))
    >>> powsimp(log(exp(x)*exp(y)), deep=True)
    x + y

    Radicals with Mul bases will be combined if combine='exp'

    >>> from sympy import sqrt
    >>> x, y = symbols('x y')

    Two radicals are automatically joined through Mul:
    ```
    >>> a=sqrt(x*sqrt(y))

这行代码计算一个复合表达式，其中a被定义为x乘以y的平方根再开平方根。


    >>> a*a**3 == a**4
    True

这行代码检查是否a的四次方等于a乘以a的立方的结果。


    But if an integer power of that radical has been
    autoexpanded then Mul does not join the resulting factors:

这段注释指出，如果根式的整数次幂已经被自动展开，则乘法操作不会合并生成的因子。


    >>> a**4 # auto expands to a Mul, no longer a Pow
    x**2*y

这行代码展示了a的四次方被自动展开为一个Mul对象，而不再是Pow对象，结果是x的平方乘以y。


    >>> _*a # so Mul doesn't combine them
    x**2*y*sqrt(x*sqrt(y))

这行代码展示了乘法操作不会合并这些因子。


    >>> powsimp(_) # but powsimp will
    (x*sqrt(y))**(5/2)

这行代码展示了使用powsimp函数后，这些因子被合并为一个表达式，结果是(x乘以y的开平方)的5/2次方。


    >>> powsimp(x*y*a) # but won't when doing so would violate assumptions
    x*y*sqrt(x*sqrt(y))

这行代码展示了在违反假设时，powsimp函数不会合并这些因子。

"""
def recurse(arg, **kwargs):
    _deep = kwargs.get('deep', deep)
    _combine = kwargs.get('combine', combine)
    _force = kwargs.get('force', force)
    _measure = kwargs.get('measure', measure)
    return powsimp(arg, _deep, _combine, _force, _measure)

这是一个递归函数`recurse`，它接受一个参数`arg`和一些关键字参数，然后调用`powsimp`函数来简化`arg`。


expr = sympify(expr)

将输入的表达式`expr`转换为符号表达式。


if (not isinstance(expr, Basic) or isinstance(expr, MatrixSymbol) or (
        expr.is_Atom or expr in (exp_polar(0), exp_polar(1)))):
    return expr

如果`expr`不是`Basic`的实例，或者是`MatrixSymbol`的实例，或者是原子表达式，或者是`exp_polar(0)`或`exp_polar(1)`，则直接返回`expr`。


if deep or expr.is_Add or expr.is_Mul and _y not in expr.args:
    expr = expr.func(*[recurse(w) for w in expr.args])

如果`deep`为真，或者`expr`是加法表达式，或者`expr`是乘法表达式且`_y`不在`expr.args`中，则对`expr.args`中的每个元素应用`recurse`函数。


if expr.is_Pow:
    return recurse(expr*_y, deep=False)/_y

如果`expr`是幂运算表达式，则返回对`expr*_y`应用`recurse`函数后再除以`_y`。


if not expr.is_Mul:
    return expr

如果`expr`不是乘法表达式，则直接返回`expr`。


# handle the Mul
else:
    raise ValueError("combine must be one of ('all', 'exp', 'base').")

如果`expr`是乘法表达式，但不符合预期的情况，则抛出值错误异常，指示`combine`必须是`('all', 'exp', 'base')`中的一个选项。
# 定义函数 powdenest，用于处理幂次中指数的收集和简化
def powdenest(eq, force=False, polar=False):
    """
    根据允许的假设收集幂次中的指数。

    Explanation
    ===========

    给定 ``(bb**be)**e``，可以根据以下条件简化：
        * 如果 ``bb`` 是正数，或者
        * ``e`` 是整数，或者
        * ``|be| < 1``，则可以简化为 ``bb**(be*e)``

    给定幂次的乘积被提升到指数的情况 ``(bb1**be1 * bb2**be2...)**e``，简化方式如下：

    - 如果 e 是正数，则所有 bei 的最大公约数可以与 e 结合；
    - 所有非负的 bb 可以与负数分开，并且它们的最大公约数可以与 e 结合；自动化简已经处理了此分离。
    - 幂次中具有分母为整数的指数的整数因子可以从任何项中移除，并且这些整数的最大公约数可以与 e 结合。

    将 `force` 设置为 `True` 将使未明确为负的符号表现得像正数一样，从而实现更多的展开。

    将 `polar` 设置为 `True` 将在对数的黎曼曲面上进行简化，也会导致更多的展开。

    当 exp() 中存在对数的和时，可能会得到幂次的乘积，例如 ``exp(3*(log(a) + 2*log(b)))`` -> ``a**3*b**6``。

    Examples
    ========

    >>> from sympy.abc import a, b, x, y, z
    >>> from sympy import Symbol, exp, log, sqrt, symbols, powdenest

    >>> powdenest((x**(2*a/3))**(3*x))
    (x**(2*a/3))**(3*x)
    >>> powdenest(exp(3*x*log(2)))
    2**(3*x)

    假设可能会阻止展开：

    >>> powdenest(sqrt(x**2))
    sqrt(x**2)

    >>> p = symbols('p', positive=True)
    >>> powdenest(sqrt(p**2))
    p

    不会进行其他的展开。

    >>> i, j = symbols('i,j', integer=True)
    >>> powdenest((x**x)**(i + j)) # -X-> (x**x)**i*(x**x)**j
    x**(x*(i + j))

    但是 exp() 将通过将所有非对数项移至函数的外部来进行展开；这可能会导致将 exp 折叠为具有不同底数的幂次：

    >>> powdenest(exp(3*y*log(x)))
    x**(3*y)
    >>> powdenest(exp(y*(log(a) + log(b))))
    (a*b)**y
    >>> powdenest(exp(3*(log(a) + log(b))))
    a**3*b**3

    如果假设允许，符号也可以移到最外层的指数中：

    >>> i = Symbol('i', integer=True)
    >>> powdenest(((x**(2*i))**(3*y))**x)
    ((x**(2*i))**(3*y))**x
    >>> powdenest(((x**(2*i))**(3*y))**x, force=True)
    x**(6*i*x*y)

    >>> powdenest(((x**(2*a/3))**(3*y/i))**x)
    ((x**(2*a/3))**(3*y/i))**x
    >>> powdenest((x**(2*i)*y**(4*i))**z, force=True)
    (x*y**2)**(2*i*z)

    >>> n = Symbol('n', negative=True)

    >>> powdenest((x**i)**y, force=True)
    x**(i*y)
    >>> powdenest((n**i)**x, force=True)
    (n**i)**x

    """
    from sympy.simplify.simplify import posify
    # 如果 force 参数为 True，则执行以下逻辑
    if force:
        # 定义内部函数 _denest，用于将幂函数展开
        def _denest(b, e):
            # 如果 b 不是 Pow 或 exp 对象，则返回其是否为正数和 Pow 对象
            if not isinstance(b, (Pow, exp)):
                return b.is_positive, Pow(b, e, evaluate=False)
            # 否则递归调用 _denest 函数直至找到基础非 Pow 或 exp 的对象
            return _denest(b.base, b.exp * e)
        
        # 初始化一个空列表 reps，用于存储替换规则
        reps = []
        
        # 遍历方程式 eq 中所有的 Pow 和 exp 对象
        for p in eq.atoms(Pow, exp):
            # 如果 p.base 是 Pow 或 exp 对象，则进行深度展开操作
            if isinstance(p.base, (Pow, exp)):
                # 调用 _denest 函数尝试展开当前对象 p，并得到展开后的结果 dp
                ok, dp = _denest(*p.args)
                # 如果展开结果 ok 不是 False，则将当前对象 p 和展开结果 dp 加入到 reps 中
                if ok is not False:
                    reps.append((p, dp))
        
        # 如果 reps 列表不为空，则对方程式 eq 执行替换操作
        if reps:
            eq = eq.subs(reps)
        
        # 对方程式 eq 进行正数化处理，并更新 reps
        eq, reps = posify(eq)
        
        # 对方程式 eq 进行幂函数化简操作，关闭 force 参数，并处理极坐标 polar
        return powdenest(eq, force=False, polar=polar).xreplace(reps)

    # 如果 polar 参数为 True，则执行以下逻辑
    if polar:
        # 将方程式 eq 转化为极坐标形式，并得到变换 rep
        eq, rep = polarify(eq)
        # 先解极坐标化方程式 eq，然后再进行幂函数化简操作，并应用 rep 变换
        return unpolarify(powdenest(unpolarify(eq, exponents_only=True)), rep)

    # 对方程式 eq 进行幂函数化简操作
    new = powsimp(eq)
    # 对化简后的结果 new 执行变换操作，过滤出所有的 Pow 或 exp 对象
    return new.xreplace(Transform(
        _denest_pow, filter=lambda m: m.is_Pow or isinstance(m, exp)))
_y = Dummy('y')  # 创建一个名为'y'的符号变量


def _denest_pow(eq):
    """
    Denest powers.

    This is a helper function for powdenest that performs the actual
    transformation.
    """
    from sympy.simplify.simplify import logcombine  # 导入logcombine函数

    b, e = eq.as_base_exp()  # 提取表达式eq的基数和指数部分
    if b.is_Pow or isinstance(b, exp) and e != 1:
        new = b._eval_power(e)  # 尝试对基数b应用指数e的幂运算
        if new is not None:
            eq = new  # 如果成功，则更新eq为新表达式
            b, e = new.as_base_exp()  # 更新b和e为新表达式的基数和指数

    # denest exp with log terms in exponent
    if b is S.Exp1 and e.is_Mul:
        logs = []
        other = []
        for ei in e.args:
            if any(isinstance(ai, log) for ai in Add.make_args(ei)):
                logs.append(ei)  # 收集e的乘积项中包含对数的部分
            else:
                other.append(ei)  # 收集e的乘积项中不包含对数的部分
        logs = logcombine(Mul(*logs))  # 将收集到的对数项进行合并
        return Pow(exp(logs), Mul(*other))  # 返回以指数为乘积项和非对数项的新幂表达式

    _, be = b.as_base_exp()
    if be is S.One and not (b.is_Mul or
                            b.is_Rational and b.q != 1 or
                            b.is_positive):
        return eq  # 如果基数b是1，并且不是乘积、有理数且分母不为1、或者不是正数，则返回原始表达式eq

    # denest eq which is either pos**e or Pow**e or Mul**e or
    # Mul(b1**e1, b2**e2)

    # handle polar numbers specially
    polars, nonpolars = [], []
    for bb in Mul.make_args(b):
        if bb.is_polar:
            polars.append(bb.as_base_exp())  # 将极坐标形式的基数部分收集起来
        else:
            nonpolars.append(bb)  # 将非极坐标形式的基数部分收集起来
    if len(polars) == 1 and not polars[0][0].is_Mul:
        return Pow(polars[0][0], polars[0][1]*e)*powdenest(Mul(*nonpolars)**e)
    elif polars:
        return Mul(*[powdenest(bb**(ee*e)) for (bb, ee) in polars]) \
            *powdenest(Mul(*nonpolars)**e)

    if b.is_Integer:
        # use log to see if there is a power here
        logb = expand_log(log(b))  # 对整数基数b应用对数扩展
        if logb.is_Mul:
            c, logb = logb.args
            e *= c
            base = logb.args[0]
            return Pow(base, e)  # 返回以扩展后的对数为指数的幂表达式

    # if b is not a Mul or any factor is an atom then there is nothing to do
    if not b.is_Mul or any(s.is_Atom for s in Mul.make_args(b)):
        return eq  # 如果基数b不是乘积，或者任何因子是原子，则返回原始表达式eq

    # let log handle the case of the base of the argument being a Mul, e.g.
    # sqrt(x**(2*i)*y**(6*i)) -> x**i*y**(3**i) if x and y are positive; we
    # will take the log, expand it, and then factor out the common powers that
    # now appear as coefficient. We do this manually since terms_gcd pulls out
    # fractions, terms_gcd(x+x*y/2) -> x*(y + 2)/2 and we don't want the 1/2;
    # gcd won't pull out numerators from a fraction: gcd(3*x, 9*x/2) -> x but
    # we want 3*x. Neither work with noncommutatives.

    def nc_gcd(aa, bb):
        a, b = [i.as_coeff_Mul() for i in [aa, bb]]
        c = gcd(a[0], b[0]).as_numer_denom()[0]
        g = Mul(*(a[1].args_cnc(cset=True)[0] & b[1].args_cnc(cset=True)[0]))
        return _keep_coeff(c, g)

    glogb = expand_log(log(b))  # 对基数b应用对数扩展
    if glogb.is_Add:
        args = glogb.args
        g = reduce(nc_gcd, args)  # 对扩展后的对数表达式进行最大公约数计算
        if g != 1:
            cg, rg = g.as_coeff_Mul()
            glogb = _keep_coeff(cg, rg*Add(*[a/g for a in args]))  # 重新组合扩展后的对数表达式

    # now put the log back together again
    # 如果 glogb 是 log 类型或者不是 Mul 类型
    if isinstance(glogb, log) or not glogb.is_Mul:
        # 如果 glogb 的第一个参数是 Pow 类型或者是 exp 类型
        if glogb.args[0].is_Pow or isinstance(glogb.args[0], exp):
            # 调用 _denest_pow 函数处理 glogb 的第一个参数
            glogb = _denest_pow(glogb.args[0])
            # 如果 glogb 的指数的绝对值小于 1
            if (abs(glogb.exp) < 1) == True:
                # 返回 Pow(glogb.base, glogb.exp*e)
                return Pow(glogb.base, glogb.exp*e)
        # 返回 eq
        return eq

    # 如果 log(b) 是 Mul 类型，则通过 logcombine 函数合并任何加法项
    add = []
    other = []
    for a in glogb.args:
        # 如果 a 是 Add 类型，则将其添加到 add 列表中
        if a.is_Add:
            add.append(a)
        # 否则将其添加到 other 列表中
        else:
            other.append(a)
    # 使用 logcombine 函数合并 add 列表中的项，然后应用 exp 函数再次取幂
    return Pow(exp(logcombine(Mul(*add))), e*Mul(*other))
```