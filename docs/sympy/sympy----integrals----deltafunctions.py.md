# `D:\src\scipysrc\sympy\sympy\integrals\deltafunctions.py`

```
from sympy.core.mul import Mul  # 导入 Mul 类，用于乘法表达式
from sympy.core.singleton import S  # 导入 S 单例，用于表示特殊常量
from sympy.core.sorting import default_sort_key  # 导入排序函数 default_sort_key
from sympy.functions import DiracDelta, Heaviside  # 导入 DiracDelta 和 Heaviside 函数
from .integrals import Integral, integrate  # 导入本地的 Integral 和 integrate 函数


def change_mul(node, x):
    """change_mul(node, x)

       重新排列乘积的操作数，将任何简单的 DiracDelta 表达式移到前面。

       Explanation
       ===========

       如果未找到简单的 DiracDelta 表达式，则简化所有的 DiracDelta 表达式
       (使用 DiracDelta.expand(diracdelta=True, wrt=x))。

       Return: (dirac, new node)
       Where:
         o dirac 要么是一个简单的 DiracDelta 表达式，要么是 None（如果未找到简单的表达式）;
         o new node 要么是简化后的 DiracDelta 表达式，要么是 None（如果无法简化）。

       Examples
       ========

       >>> from sympy import DiracDelta, cos
       >>> from sympy.integrals.deltafunctions import change_mul
       >>> from sympy.abc import x, y
       >>> change_mul(x*y*DiracDelta(x)*cos(x), x)
       (DiracDelta(x), x*y*cos(x))
       >>> change_mul(x*y*DiracDelta(x**2 - 1)*cos(x), x)
       (None, x*y*cos(x)*DiracDelta(x - 1)/2 + x*y*cos(x)*DiracDelta(x + 1)/2)
       >>> change_mul(x*y*DiracDelta(cos(x))*cos(x), x)
       (None, None)

       See Also
       ========

       sympy.functions.special.delta_functions.DiracDelta
       deltaintegrate
    """

    new_args = []  # 初始化空列表 new_args
    dirac = None  # 初始化 dirac 变量为 None

    #Sorting is needed so that we consistently collapse the same delta;
    #However, we must preserve the ordering of non-commutative terms
    c, nc = node.args_cnc()  # 将节点按照可交换和不可交换的项分组
    sorted_args = sorted(c, key=default_sort_key)  # 对可交换项按照默认排序键排序
    sorted_args.extend(nc)  # 将不可交换项添加到排序后的列表中

    for arg in sorted_args:
        if arg.is_Pow and isinstance(arg.base, DiracDelta):
            new_args.append(arg.func(arg.base, arg.exp - 1))
            arg = arg.base
        if dirac is None and (isinstance(arg, DiracDelta) and arg.is_simple(x)):
            dirac = arg
        else:
            new_args.append(arg)
    if not dirac:  # 如果没有简单的 DiracDelta 表达式
        new_args = []
        for arg in sorted_args:
            if isinstance(arg, DiracDelta):
                new_args.append(arg.expand(diracdelta=True, wrt=x))  # 扩展 DiracDelta 表达式
            elif arg.is_Pow and isinstance(arg.base, DiracDelta):
                new_args.append(arg.func(arg.base.expand(diracdelta=True, wrt=x), arg.exp))
            else:
                new_args.append(arg)
        if new_args != sorted_args:
            nnode = Mul(*new_args).expand()  # 构造新的乘法表达式并展开
        else:  # 如果节点没有改变，则无需操作
            nnode = None
        return (None, nnode)
    return (dirac, Mul(*new_args))


def deltaintegrate(f, x):
    """
    deltaintegrate(f, x)

    Explanation
    ===========

    The idea for integration is the following:

    ...
    """
    # 如果表达式 f 中不包含 DiracDelta，则返回 None
    if not f.has(DiracDelta):
        return None
    
    # 如果 f 是 DiracDelta 函数的实例
    if f.func == DiracDelta:
        # 尝试对 f 进行展开，使得其中的 DiracDelta 表达式更简单，展开变量是 x
        h = f.expand(diracdelta=True, wrt=x)
        # 如果无法简化表达式
        if h == f:
            # 表示无法简化表达式
            # FIXME: 第二项指明这是 DiracDelta 还是 Derivative
            # 对于 DiracDelta 的导数，需要使用链式法则进行积分
            # 如果 f 在变量 x 上是简单的
            if f.is_simple(x):
                # 如果 f 的参数个数小于等于1或者第二个参数为0
                if (len(f.args) <= 1 or f.args[1] == 0):
                    # 返回 Heaviside 函数，参数为 f 的第一个参数
                    return Heaviside(f.args[0])
                else:
                    # 返回 DiracDelta(f.args[0], f.args[1] - 1) 除以 f.args[0] 的首项系数
                    return (DiracDelta(f.args[0], f.args[1] - 1) /
                            f.args[0].as_poly().LC())
        else:  # 尝试积分简化后的表达式
            fh = integrate(h, x)
            return fh
    elif f.is_Mul or f.is_Pow:  # 如果 f 是乘法或者幂运算： g(x) = a*b*c*f(DiracDelta(h(x)))*d*e
        g = f.expand()  # 对 f 进行展开
        if f != g:  # 如果展开有效
            fh = integrate(g, x)  # 对展开后的 g(x) 进行积分
            if fh is not None and not isinstance(fh, Integral):  # 如果积分结果不为空且不是积分对象
                return fh  # 返回积分结果
        else:
            # 没有进行展开，尝试提取一个简单的 DiracDelta 项
            deltaterm, rest_mult = change_mul(f, x)

            if not deltaterm:  # 如果没有提取到 DiracDelta 项
                if rest_mult:  # 如果还有剩余乘积
                    fh = integrate(rest_mult, x)  # 对剩余乘积进行积分
                    return fh  # 返回积分结果
            else:
                from sympy.solvers import solve
                deltaterm = deltaterm.expand(diracdelta=True, wrt=x)  # 对提取的 DiracDelta 项进行展开
                if deltaterm.is_Mul:  # 如果是乘法形式的 DiracDelta 项，取出任何提取的因子
                    deltaterm, rest_mult_2 = change_mul(deltaterm, x)
                    rest_mult = rest_mult * rest_mult_2
                point = solve(deltaterm.args[0], x)[0]  # 解方程找到点 x 的值

                # 返回通过重复分部积分后留下的最大超实数项。例如，
                #
                #   integrate(y*DiracDelta(x, 1),x) == y*DiracDelta(x,0),  not 0
                #
                # 这样做是为了使 Integral(y*DiracDelta(x).diff(x),x).doit()
                # 返回 y*DiracDelta(x) 而不是 0 或者 DiracDelta(x)，
                # 虽然它们在定义域内的值都是正确的，但对于嵌套积分会给出错误答案。
                n = (0 if len(deltaterm.args)==1 else deltaterm.args[1])  # 确定 n 的值
                m = 0
                while n >= 0:
                    r = S.NegativeOne**n * rest_mult.diff(x, n).subs(x, point)  # 计算积分项
                    if r.is_zero:  # 如果结果为零
                        n -= 1
                        m += 1
                    else:
                        if m == 0:
                            return r * Heaviside(x - point)  # 返回结果乘以 Heaviside 函数
                        else:
                            return r * DiracDelta(x, m-1)  # 返回结果乘以 DiracDelta 函数
                # 在某种非常微弱的意义上，x=0 仍然是一个奇点，
                # 但我们希望它在实际应用中不会有任何影响。
                return S.Zero  # 返回零
    return None  # 默认返回空值
```