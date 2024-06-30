# `D:\src\scipysrc\sympy\sympy\series\order.py`

```
from sympy.core import S, sympify, Expr, Dummy, Add, Mul
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.function import Function, PoleError, expand_power_base, expand_log
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.exponential import exp, log
from sympy.sets.sets import Complement
from sympy.utilities.iterables import uniq, is_sequence

# 导入 SymPy 库中需要的模块和类

class Order(Expr):
    r""" Represents the limiting behavior of some function.

    Explanation
    ===========

    The order of a function characterizes the function based on the limiting
    behavior of the function as it goes to some limit. Only taking the limit
    point to be a number is currently supported. This is expressed in
    big O notation [1]_.

    The formal definition for the order of a function `g(x)` about a point `a`
    is such that `g(x) = O(f(x))` as `x \rightarrow a` if and only if there
    exists a `\delta > 0` and an `M > 0` such that `|g(x)| \leq M|f(x)|` for
    `|x-a| < \delta`.  This is equivalent to `\limsup_{x \rightarrow a}
    |g(x)/f(x)| < \infty`.

    Let's illustrate it on the following example by taking the expansion of
    `\sin(x)` about 0:

    .. math ::
        \sin(x) = x - x^3/3! + O(x^5)

    where in this case `O(x^5) = x^5/5! - x^7/7! + \cdots`. By the definition
    of `O`, there is a `\delta > 0` and an `M` such that:

    .. math ::
        |x^5/5! - x^7/7! + ....| <= M|x^5| \text{ for } |x| < \delta

    or by the alternate definition:

    .. math ::
        \lim_{x \rightarrow 0} | (x^5/5! - x^7/7! + ....) / x^5| < \infty

    which surely is true, because

    .. math ::
        \lim_{x \rightarrow 0} | (x^5/5! - x^7/7! + ....) / x^5| = 1/5!


    As it is usually used, the order of a function can be intuitively thought
    of representing all terms of powers greater than the one specified. For
    example, `O(x^3)` corresponds to any terms proportional to `x^3,
    x^4,\ldots` and any higher power. For a polynomial, this leaves terms
    proportional to `x^2`, `x` and constants.

    Examples
    ========

    >>> from sympy import O, oo, cos, pi
    >>> from sympy.abc import x, y

    >>> O(x + x**2)
    O(x)
    >>> O(x + x**2, (x, 0))
    O(x)
    >>> O(x + x**2, (x, oo))
    O(x**2, (x, oo))

    >>> O(1 + x*y)
    O(1, x, y)
    >>> O(1 + x*y, (x, 0), (y, 0))
    O(1, x, y)
    >>> O(1 + x*y, (x, oo), (y, oo))
    O(x*y, (x, oo), (y, oo))

    >>> O(1) in O(1, x)
    True
    >>> O(1, x) in O(1)
    False
    >>> O(x) in O(1, x)
    True
    >>> O(x**2) in O(x)
    True

    >>> O(x)*x
    O(x**2)
    >>> O(x) - O(x)
    O(x)
    >>> O(cos(x))
    O(1)
    >>> O(cos(x), (x, pi/2))
    O(x - pi/2, (x, pi/2))

    References
    ==========

    .. [1] `Big O notation <https://en.wikipedia.org/wiki/Big_O_notation>`_

    Notes
    =====

    In ``O(f(x), x)`` the expression ``f(x)`` is assumed to have a leading
    term.  ``O(f(x), x)`` is automatically transformed to
    """
    ``O(f(x).as_leading_term(x),x)``.

        ``O(expr*f(x), x)`` is ``O(f(x), x)``

        ``O(expr, x)`` is ``O(1)``

        ``O(0, x)`` is 0.

    Multivariate O is also supported:

        ``O(f(x, y), x, y)`` is transformed to
        ``O(f(x, y).as_leading_term(x,y).as_leading_term(y), x, y)``

    In the multivariate case, it is assumed the limits w.r.t. the various
    symbols commute.

    If no symbols are passed then all symbols in the expression are used
    and the limit point is assumed to be zero.
    """

    # 标记这个类为一个 Order 类型
    is_Order = True

    # 定义一个空的 __slots__，以优化内存使用
    __slots__ = ()

    # 缓存修饰器，用于对 _eval_nseries 方法的结果进行缓存
    @cacheit
    def _eval_nseries(self, x, n, logx, cdir=0):
        return self  # 直接返回当前对象，不进行级数展开

    # 获取表达式部分
    @property
    def expr(self):
        return self.args[0]

    # 获取所有变量
    @property
    def variables(self):
        if self.args[1:]:
            return tuple(x[0] for x in self.args[1:])
        else:
            return ()

    # 获取点（限制点）
    @property
    def point(self):
        if self.args[1:]:
            return tuple(x[1] for x in self.args[1:])
        else:
            return ()

    # 获取自由符号
    @property
    def free_symbols(self):
        return self.expr.free_symbols | set(self.variables)

    # 定义 _eval_power 方法，用于处理 Order 对象的幂运算
    def _eval_power(b, e):
        if e.is_Number and e.is_nonnegative:
            return b.func(b.expr ** e, *b.args[1:])
        if e == O(1):
            return b
        return

    # 将 Order 对象转换为表达式和变量的元组
    def as_expr_variables(self, order_symbols):
        if order_symbols is None:
            order_symbols = self.args[1:]
        else:
            if (not all(o[1] == order_symbols[0][1] for o in order_symbols) and
                    not all(p == self.point[0] for p in self.point)):
                raise NotImplementedError('Order at points other than 0 '
                                          'or oo not supported, got %s as a point.' % self.point)
            if order_symbols and order_symbols[0][1] != self.point[0]:
                raise NotImplementedError(
                    "Multiplying Order at different points is not supported.")
            order_symbols = dict(order_symbols)
            for s, p in dict(self.args[1:]).items():
                if s not in order_symbols.keys():
                    order_symbols[s] = p
            order_symbols = sorted(order_symbols.items(), key=lambda x: default_sort_key(x[0]))
        return self.expr, tuple(order_symbols)

    # 移除 Order 部分，返回零
    def removeO(self):
        return S.Zero

    # 返回当前 Order 对象
    def getO(self):
        return self

    # 缓存修饰器，用于检查其他对象是否包含在当前 Order 对象中
    @cacheit
    def __contains__(self, other):
        result = self.contains(other)
        if result is None:
            raise TypeError('contains did not evaluate to a bool')
        return result
    # 对象方法：用于对表达式中的变量进行替换
    def _eval_subs(self, old, new):
        # 如果旧变量存在于当前对象的变量列表中
        if old in self.variables:
            # 使用新表达式替换旧变量后的新表达式
            newexpr = self.expr.subs(old, new)
            # 获取旧变量在变量列表中的索引
            i = self.variables.index(old)
            # 创建变量和点的副本
            newvars = list(self.variables)
            newpt = list(self.point)
            # 如果新变量是符号
            if new.is_symbol:
                # 将新变量替换旧变量的位置
                newvars[i] = new
            else:
                # 获取新表达式中自由符号集合
                syms = new.free_symbols
                # 如果自由符号集合只有一个元素或者旧变量在集合中
                if len(syms) == 1 or old in syms:
                    # 如果旧变量在符号集合中，则选取该变量作为替换变量
                    if old in syms:
                        var = self.variables[i]
                    else:
                        var = syms.pop()
                    # 尝试用当前点的值替换新表达式中的变量，以检查是否为固定点
                    point = new.subs(var, self.point[i])
                    # 如果点的值发生变化
                    if point != self.point[i]:
                        # 导入解方程的工具
                        from sympy.solvers.solveset import solveset
                        # 创建一个虚拟变量
                        d = Dummy()
                        # 解方程 old - new.subs(var, d) = 0
                        sol = solveset(old - new.subs(var, d), d)
                        # 如果解是补集形式，取其第一个元素
                        if isinstance(sol, Complement):
                            e1 = sol.args[0]
                            e2 = sol.args[1]
                            sol = set(e1) - set(e2)
                        # 将解包装成字典形式
                        res = [dict(zip((d, ), sol))]
                        # 对虚拟变量 d 进行极限运算
                        point = d.subs(res[0]).limit(old, self.point[i])
                    # 更新变量列表和点的列表
                    newvars[i] = var
                    newpt[i] = point
                # 如果旧变量不在符号集合中
                elif old not in syms:
                    # 删除变量列表和点的列表中的旧变量
                    del newvars[i], newpt[i]
                    # 如果符号集合为空且新表达式等于当前点的值
                    if not syms and new == self.point[i]:
                        newvars.extend(syms)
                        newpt.extend([S.Zero]*len(syms))
                else:
                    return
            # 返回用新变量和点生成的 Order 对象
            return Order(newexpr, *zip(newvars, newpt))

    # 对象方法：计算表达式的共轭
    def _eval_conjugate(self):
        # 计算表达式的共轭
        expr = self.expr._eval_conjugate()
        # 如果计算得到了共轭表达式
        if expr is not None:
            # 返回一个新的对象，将共轭表达式传递给构造函数
            return self.func(expr, *self.args[1:])

    # 对象方法：计算表达式对指定变量的导数
    def _eval_derivative(self, x):
        # 计算表达式对指定变量 x 的导数
        return self.func(self.expr.diff(x), *self.args[1:]) or self

    # 对象方法：计算表达式的转置
    def _eval_transpose(self):
        # 计算表达式的转置
        expr = self.expr._eval_transpose()
        # 如果计算得到了转置表达式
        if expr is not None:
            # 返回一个新的对象，将转置表达式传递给构造函数
            return self.func(expr, *self.args[1:])

    # 对象方法：表达式的负运算
    def __neg__(self):
        # 返回表达式的负数
        return self
# 创建一个名为 O 的变量，并将其指向 Order 对象
O = Order
```