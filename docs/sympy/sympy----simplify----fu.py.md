# `D:\src\scipysrc\sympy\sympy\simplify\fu.py`

```
from collections import defaultdict  # 导入 defaultdict 类用于创建默认字典

from sympy.core.add import Add  # 导入 Add 类用于代表加法表达式
from sympy.core.expr import Expr  # 导入 Expr 类用于代表表达式
from sympy.core.exprtools import Factors, gcd_terms, factor_terms  # 导入因子相关工具函数
from sympy.core.function import expand_mul  # 导入 expand_mul 函数用于展开乘法
from sympy.core.mul import Mul  # 导入 Mul 类用于代表乘法表达式
from sympy.core.numbers import pi, I  # 导入 pi 和虚数单位 I
from sympy.core.power import Pow  # 导入 Pow 类用于代表幂表达式
from sympy.core.singleton import S  # 导入 S 单例，用于表示一个符号
from sympy.core.sorting import ordered  # 导入 ordered 函数用于排序
from sympy.core.symbol import Dummy  # 导入 Dummy 类用于创建虚拟符号
from sympy.core.sympify import sympify  # 导入 sympify 函数用于将字符串转换为符号表达式
from sympy.core.traversal import bottom_up  # 导入 bottom_up 函数用于从底向上遍历表达式
from sympy.functions.combinatorial.factorials import binomial  # 导入二项式系数函数
from sympy.functions.elementary.hyperbolic import (  # 导入双曲函数
    cosh, sinh, tanh, coth, sech, csch, HyperbolicFunction)
from sympy.functions.elementary.trigonometric import (  # 导入三角函数
    cos, sin, tan, cot, sec, csc, sqrt, TrigonometricFunction)
from sympy.ntheory.factor_ import perfect_power  # 导入 perfect_power 函数判断是否完全幂
from sympy.polys.polytools import factor  # 导入 factor 函数用于因式分解
from sympy.strategies.tree import greedy  # 导入 greedy 策略用于树搜索
from sympy.strategies.core import identity, debug  # 导入 identity 和 debug 策略

from sympy import SYMPY_DEBUG  # 导入 SYMPY_DEBUG 用于控制调试模式

# ================== Fu-like tools ===========================

def TR0(rv):
    """简化有理多项式，尝试简化表达式，例如合并像 3*x + 2*x 这样的内容"""
    # 虽然使用 cancel 函数可以很好地处理可交换对象，但无法处理非可交换对象
    return rv.normal().factor().expand()


def TR1(rv):
    """用 1/cos, 1/sin 替换 sec, csc"""
    def f(rv):
        if isinstance(rv, sec):
            a = rv.args[0]
            return S.One/cos(a)
        elif isinstance(rv, csc):
            a = rv.args[0]
            return S.One/sin(a)
        return rv

    return bottom_up(rv, f)


def TR2(rv):
    """用 sin/cos, cos/sin 替换 tan, cot"""
    def f(rv):
        if isinstance(rv, tan):
            a = rv.args[0]
            return sin(a)/cos(a)
        elif isinstance(rv, cot):
            a = rv.args[0]
            return cos(a)/sin(a)
        return rv

    return bottom_up(rv, f)


def TR2i(rv, half=False):
    """将涉及 sin 和 cos 的比率转换如下::
        sin(x)/cos(x) -> tan(x)
        如果 half=True，则 sin(x)/(cos(x) + 1) -> tan(x/2)

    虚数单位及幂的规则也可以识别

    """
    # 转换仅在符合假设条件下进行
    (i.e. the base must be positive or the exponent must be an integer
    for both numerator and denominator)

    >>> TR2i(sin(x)**a/(cos(x) + 1)**a)
    sin(x)**a/(cos(x) + 1)**a

    """


注释：


    # 这段代码段是一个文档字符串（docstring），用于描述函数或方法的功能和使用方法
    # 文档字符串通常包含函数的输入参数、返回值及其它相关信息
    # 在这里，文档字符串可能用于描述函数 TR2i 的用法或者一些关键信息
    def f(rv):
        # 如果不是乘法表达式，直接返回原始值
        if not rv.is_Mul:
            return rv
    
        # 将分子和分母分离出来
        n, d = rv.as_numer_denom()
        # 如果分子或分母是原子表达式，直接返回原始值
        if n.is_Atom or d.is_Atom:
            return rv
    
        def ok(k, e):
            # 初始因子过滤
            return (
                (e.is_integer or k.is_positive) and (
                k.func in (sin, cos) or (half and
                k.is_Add and
                len(k.args) >= 2 and
                any(any(isinstance(ai, cos) or ai.is_Pow and ai.base is cos
                for ai in Mul.make_args(a)) for a in k.args))))
    
        # 将分子转换为幂次字典
        n = n.as_powers_dict()
        # 从分子中移除不符合条件的因子
        ndone = [(k, n.pop(k)) for k in list(n.keys()) if not ok(k, n[k])]
        # 如果分子为空字典，则返回原始值
        if not n:
            return rv
    
        # 将分母转换为幂次字典
        d = d.as_powers_dict()
        # 从分母中移除不符合条件的因子
        ddone = [(k, d.pop(k)) for k in list(d.keys()) if not ok(k, d[k])]
        # 如果分母为空字典，则返回原始值
        if not d:
            return rv
    
        # 如果需要，进行因式分解
        def factorize(d, ddone):
            newk = []
            for k in d:
                if k.is_Add and len(k.args) > 1:
                    knew = factor(k) if half else factor_terms(k)
                    if knew != k:
                        newk.append((k, knew))
            if newk:
                for i, (k, knew) in enumerate(newk):
                    del d[k]
                    newk[i] = knew
                newk = Mul(*newk).as_powers_dict()
                for k in newk:
                    v = d[k] + newk[k]
                    if ok(k, v):
                        d[k] = v
                    else:
                        ddone.append((k, v))
                del newk
        factorize(n, ndone)
        factorize(d, ddone)
    
        # 合并处理结果
        t = []
        for k in n:
            if isinstance(k, sin):
                a = cos(k.args[0], evaluate=False)
                if a in d and d[a] == n[k]:
                    t.append(tan(k.args[0])**n[k])
                    n[k] = d[a] = None
                elif half:
                    a1 = 1 + a
                    if a1 in d and d[a1] == n[k]:
                        t.append((tan(k.args[0]/2))**n[k])
                        n[k] = d[a1] = None
            elif isinstance(k, cos):
                a = sin(k.args[0], evaluate=False)
                if a in d and d[a] == n[k]:
                    t.append(tan(k.args[0])**-n[k])
                    n[k] = d[a] = None
            elif half and k.is_Add and k.args[0] is S.One and \
                    isinstance(k.args[1], cos):
                a = sin(k.args[1].args[0], evaluate=False)
                if a in d and d[a] == n[k] and (d[a].is_integer or \
                        a.is_positive):
                    t.append(tan(a.args[0]/2)**-n[k])
                    n[k] = d[a] = None
    
        # 如果存在处理结果，则重新构建表达式
        if t:
            rv = Mul(*(t + [b**e for b, e in n.items() if e]))/\
                Mul(*[b**e for b, e in d.items() if e])
            rv *= Mul(*[b**e for b, e in ndone])/Mul(*[b**e for b, e in ddone])
    
        # 返回最终结果
        return rv
    
    # 对给定表达式应用底向上的变换函数f，并返回结果
    return bottom_up(rv, f)
# 定义函数 TR3，用于对表达式进行简化，特别处理三角函数中的负数和特定角度范围的处理
def TR3(rv):
    from sympy.simplify.simplify import signsimp  # 导入符号简化函数 signsimp

    # 内部函数 f，用于处理三角函数的参数
    def f(rv):
        # 如果参数不是三角函数，直接返回参数
        if not isinstance(rv, TrigonometricFunction):
            return rv
        # 对参数进行符号简化
        rv = rv.func(signsimp(rv.args[0]))
        # 如果简化后参数不是三角函数，直接返回参数
        if not isinstance(rv, TrigonometricFunction):
            return rv
        # 处理特定范围的角度，将其转换为相应的角度
        if (rv.args[0] - S.Pi/4).is_positive is (S.Pi/2 - rv.args[0]).is_positive is True:
            # 创建函数映射表，用于角度转换
            fmap = {cos: sin, sin: cos, tan: cot, cot: tan, sec: csc, csc: sec}
            rv = fmap[type(rv)](S.Pi/2 - rv.args[0])  # 根据类型进行转换
        return rv

    # 将 rv 中的三角函数参数替换，使其自动更新
    rv = rv.replace(
        lambda x: isinstance(x, TrigonometricFunction),  # 检查是否为三角函数
        lambda x: x.replace(
            lambda n: n.is_number and n.is_Mul,  # 检查是否为数值和乘法
            lambda n: n.func(*n.args)))  # 进行函数操作

    return bottom_up(rv, f)  # 应用 f 函数进行底向上处理


# 定义函数 TR4，用于识别特殊角度的三角函数值
def TR4(rv):
    # 特殊角度 0, pi/6, pi/4, pi/3, pi/2 的处理已经完成
    return rv.replace(
        lambda x:
            isinstance(x, TrigonometricFunction) and  # 检查是否为三角函数
            (r:=x.args[0]/pi).is_Rational and r.q in (1, 2, 3, 4, 6),  # 检查角度是否为特定有理数
        lambda x:
            x.func(x.args[0].func(*x.args[0].args)))  # 替换三角函数的参数


# 定义辅助函数 _TR56，用于在 TR5 和 TR6 中替换 f**2 为 h(g**2) 的形式
def _TR56(rv, f, g, h, max, pow):
    # max 控制可以出现在 f 上的指数大小
    # 例如，如果 max=4，则 f**4 将被更改为 h(g**2)**2。
    """
        pow : controls whether the exponent must be a perfect power of 2
              控制指数是否必须是2的幂
              e.g. if pow=True (and max >= 6) then f**6 will not be changed
              例如，如果 pow=True（且 max >= 6），那么 f**6 将不会被改变
              but f**8 will be changed to h(g**2)**4
              但是 f**8 将会被改变为 h(g**2)**4
    
        >>> from sympy.simplify.fu import _TR56 as T
        导入符号计算模块中的 _TR56 函数作为 T
    
        >>> from sympy.abc import x
        导入符号变量 x
    
        >>> from sympy import sin, cos
        导入 sin 和 cos 函数
    
        >>> h = lambda x: 1 - x
        定义 lambda 函数 h，其输入为 x，输出为 1 - x
    
        >>> T(sin(x)**3, sin, cos, h, 4, False)
        对 sin(x)**3 应用 T 函数，其中 sin、cos、h 是函数参数，4 是 max，False 是 pow
        输出为 (1 - cos(x)**2)*sin(x)
    
        >>> T(sin(x)**6, sin, cos, h, 6, False)
        对 sin(x)**6 应用 T 函数，其中 sin、cos、h 是函数参数，6 是 max，False 是 pow
        输出为 (1 - cos(x)**2)**3
    
        >>> T(sin(x)**6, sin, cos, h, 6, True)
        对 sin(x)**6 应用 T 函数，其中 sin、cos、h 是函数参数，6 是 max，True 是 pow
        输出为 sin(x)**6
    
        >>> T(sin(x)**8, sin, cos, h, 10, True)
        对 sin(x)**8 应用 T 函数，其中 sin、cos、h 是函数参数，10 是 max，True 是 pow
        输出为 (1 - cos(x)**2)**4
        """
    
        def _f(rv):
            # I'm not sure if this transformation should target all even powers
            # or only those expressible as powers of 2. Also, should it only
            # make the changes in powers that appear in sums -- making an isolated
            # change is not going to allow a simplification as far as I can tell.
            # 我不确定此转换是否应该针对所有偶数次幂，还是仅限于可表示为2的幂的次幂。
            # 此外，它是否应该只在出现在和式中的幂上进行更改 - 单独的更改不会实现简化，据我所知。
            
            # 如果 rv 不是 f 函数的指数形式，则返回 rv 本身
            if not (rv.is_Pow and rv.base.func == f):
                return rv
            # 如果指数部分不是实数，则返回 rv 本身
            if not rv.exp.is_real:
                return rv
    
            # 如果指数小于 0，则返回 rv 本身
            if (rv.exp < 0) == True:
                return rv
            # 如果指数大于 max，则返回 rv 本身
            if (rv.exp > max) == True:
                return rv
            # 如果指数为 1，则返回 rv 本身
            if rv.exp == 1:
                return rv
            # 如果指数为 2，则返回 h(g(rv.base.args[0])**2)
            if rv.exp == 2:
                return h(g(rv.base.args[0])**2)
            else:
                # 对于其他情况：
                if rv.exp % 2 == 1:  # 如果指数为奇数
                    e = rv.exp//2
                    return f(rv.base.args[0])*h(g(rv.base.args[0])**2)**e
                elif rv.exp == 4:  # 如果指数为 4
                    e = 2
                elif not pow:  # 如果 pow 不为 True
                    if rv.exp % 2:  # 如果指数为奇数
                        return rv
                    e = rv.exp//2
                else:  # 如果 pow 为 True
                    p = perfect_power(rv.exp)
                    if not p:
                        return rv
                    e = rv.exp//2
                return h(g(rv.base.args[0])**2)**e
    
        return bottom_up(rv, _f)
# 定义函数 TR5，用于将 sin**2 替换为 1 - cos(x)**2
def TR5(rv, max=4, pow=False):
    """Replacement of sin**2 with 1 - cos(x)**2.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR5
    >>> from sympy.abc import x
    >>> from sympy import sin
    >>> TR5(sin(x)**2)
    1 - cos(x)**2
    >>> TR5(sin(x)**-2)  # unchanged
    sin(x)**(-2)
    >>> TR5(sin(x)**4)
    (1 - cos(x)**2)**2
    """
    # 调用 _TR56 函数，使用 sin 和 cos 函数及 lambda 表达式进行转换
    return _TR56(rv, sin, cos, lambda x: 1 - x, max=max, pow=pow)


# 定义函数 TR6，用于将 cos**2 替换为 1 - sin(x)**2
def TR6(rv, max=4, pow=False):
    """Replacement of cos**2 with 1 - sin(x)**2.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR6
    >>> from sympy.abc import x
    >>> from sympy import cos
    >>> TR6(cos(x)**2)
    1 - sin(x)**2
    >>> TR6(cos(x)**-2)  # unchanged
    cos(x)**(-2)
    >>> TR6(cos(x)**4)
    (1 - sin(x)**2)**2
    """
    # 调用 _TR56 函数，使用 cos 和 sin 函数及 lambda 表达式进行转换
    return _TR56(rv, cos, sin, lambda x: 1 - x, max=max, pow=pow)


# 定义函数 TR7，用于降低 cos(x)**2 的幂次
def TR7(rv):
    """Lowering the degree of cos(x)**2.

    Examples
    ========

    >>> from sympy.simplify.fu import TR7
    >>> from sympy.abc import x
    >>> from sympy import cos
    >>> TR7(cos(x)**2)
    cos(2*x)/2 + 1/2
    >>> TR7(cos(x)**2 + 1)
    cos(2*x)/2 + 3/2

    """
    # 定义内部函数 f，如果输入的表达式不是 cos(x)**2，则返回原表达式，否则进行转换
    def f(rv):
        if not (rv.is_Pow and rv.base.func == cos and rv.exp == 2):
            return rv
        return (1 + cos(2*rv.base.args[0]))/2
    
    # 应用 bottom_up 函数，对输入的 rv 应用 f 函数进行转换
    return bottom_up(rv, f)


# 定义函数 TR8，用于将 cos 和 sin 的乘积转换为它们的和或差
def TR8(rv, first=True):
    """Converting products of ``cos`` and/or ``sin`` to a sum or
    difference of ``cos`` and or ``sin`` terms.

    Examples
    ========

    >>> from sympy.simplify.fu import TR8
    >>> from sympy import cos, sin
    >>> TR8(cos(2)*cos(3))
    cos(5)/2 + cos(1)/2
    >>> TR8(cos(2)*sin(3))
    sin(5)/2 + sin(1)/2
    >>> TR8(sin(2)*sin(3))
    -cos(5)/2 + cos(1)/2
    """
    # 这里的函数体未提供，示例中的函数体应该实现将 cos 和 sin 的乘积转换为和或差的逻辑
    # 但实际示例中未给出，因此在此留空
    def f(rv):
        # 如果 rv 不是乘法或者是幂运算且底数是 cos 或 sin 函数，并且指数是整数或者底数是正数，则不做处理，直接返回 rv
        if not (
            rv.is_Mul or
            rv.is_Pow and
            rv.base.func in (cos, sin) and
            (rv.exp.is_integer or rv.base.is_positive)):
            return rv

        # 如果是第一次调用函数
        if first:
            # 将 rv 拆分为分子和分母，然后分别展开每部分
            n, d = [expand_mul(i) for i in rv.as_numer_denom()]
            # 对分子和分母分别进行 TR8 转换
            newn = TR8(n, first=False)
            newd = TR8(d, first=False)
            # 如果分子或分母有变化，则重新计算最大公约数
            if newn != n or newd != d:
                rv = gcd_terms(newn/newd)
                # 如果结果是乘法且第一个参数是有理数且有两个参数且第二个参数是加法，则重新调整为乘法形式
                if rv.is_Mul and rv.args[0].is_Rational and \
                        len(rv.args) == 2 and rv.args[1].is_Add:
                    rv = Mul(*rv.as_coeff_Mul())
            return rv

        # 初始化函数参数列表，分为 cos、sin 和其他
        args = {cos: [], sin: [], None: []}
        # 将 rv 中的每个因子分配到相应的列表中
        for a in Mul.make_args(rv):
            if a.func in (cos, sin):
                args[type(a)].append(a.args[0])
            elif (a.is_Pow and a.exp.is_Integer and a.exp > 0 and \
                    a.base.func in (cos, sin)):
                # 如果是指数是正整数的幂运算且底数是 cos 或 sin 函数，则展开处理
                args[type(a.base)].extend([a.base.args[0]]*a.exp)
            else:
                args[None].append(a)
        
        # 提取 cos 和 sin 的列表
        c = args[cos]
        s = args[sin]
        # 如果没有 cos 或 sin 的因子或者其中一个列表的长度大于 1，则不做处理，直接返回 rv
        if not (c and s or len(c) > 1 or len(s) > 1):
            return rv
        
        # 处理其他因子的列表
        args = args[None]
        # 取 cos 和 sin 列表长度的最小值
        n = min(len(c), len(s))
        # 将对应位置的 cos 和 sin 因子进行合并操作，结果添加到 args 列表中
        for i in range(n):
            a1 = s.pop()
            a2 = c.pop()
            args.append((sin(a1 + a2) + sin(a1 - a2))/2)
        
        # 处理剩余的 cos 因子
        while len(c) > 1:
            a1 = c.pop()
            a2 = c.pop()
            args.append((cos(a1 + a2) + cos(a1 - a2))/2)
        if c:
            args.append(cos(c.pop()))
        
        # 处理剩余的 sin 因子
        while len(s) > 1:
            a1 = s.pop()
            a2 = s.pop()
            args.append((-cos(a1 + a2) + cos(a1 - a2))/2)
        if s:
            args.append(sin(s.pop()))
        
        # 将处理后的参数列表作为 Mul 的参数，再进行 TR8 转换，并展开乘法
        return TR8(expand_mul(Mul(*args)))
    
    # 对 rv 应用 bottom_up 函数，并使用 f 函数进行转换处理
    return bottom_up(rv, f)
# 定义一个函数 TR9，用于将一系列 cos 或 sin 项表示为 cos 或 sin 项的乘积之和
"""Sum of ``cos`` or ``sin`` terms as a product of ``cos`` or ``sin``.

Examples
========
示例

>>> from sympy.simplify.fu import TR9
>>> from sympy import cos, sin
>>> TR9(cos(1) + cos(2))
2*cos(1/2)*cos(3/2)
>>> TR9(cos(1) + 2*sin(1) + 2*sin(2))
cos(1) + 4*sin(3/2)*cos(1/2)

If no change is made by TR9, no re-arrangement of the
expression will be made. For example, though factoring
of common term is attempted, if the factored expression
was not changed, the original expression will be returned:

如果 TR9 没有进行任何更改，则不会重新排列表达式。
例如，尽管尝试因式分解公共项，但如果因式分解的表达式
未更改，则将返回原始表达式：

"""
    def f(rv):
        # 如果 rv 不是一个加法表达式，则直接返回 rv
        if not rv.is_Add:
            return rv

        def do(rv, first=True):
            # 如果 rv 不是一个加法表达式，则直接返回 rv
            if not rv.is_Add:
                return rv

            args = list(ordered(rv.args))
            # 如果参数个数不是两个，则尝试合并可组合成的三角函数和的项
            if len(args) != 2:
                hit = False
                # 遍历所有可能的参数组合
                for i in range(len(args)):
                    ai = args[i]
                    if ai is None:
                        continue
                    for j in range(i + 1, len(args)):
                        aj = args[j]
                        if aj is None:
                            continue
                        was = ai + aj
                        # 递归调用 do 函数处理可以简化的项
                        new = do(was)
                        if new != was:
                            args[i] = new  # 在原地更新
                            args[j] = None
                            hit = True
                            break  # 跳出当前循环，继续下一个 i 的处理
                # 如果有合并操作发生，则重新构造一个加法表达式
                if hit:
                    rv = Add(*[_f for _f in args if _f])
                    # 如果 rv 仍然是一个加法表达式，则继续递归处理
                    if rv.is_Add:
                        rv = do(rv)

                return rv

            # 如果参数个数为两个，则调用 trig_split 函数进行处理
            split = trig_split(*args)
            # 如果 split 返回空，则返回原始 rv
            if not split:
                return rv
            gcd, n1, n2, a, b, iscos = split

            # 根据规则应用对应的三角函数合并公式
            if iscos:
                if n1 == n2:
                    return gcd*n1*2*cos((a + b)/2)*cos((a - b)/2)
                if n1 < 0:
                    a, b = b, a
                return -2*gcd*sin((a + b)/2)*sin((a - b)/2)
            else:
                if n1 == n2:
                    return gcd*n1*2*sin((a + b)/2)*cos((a - b)/2)
                if n1 < 0:
                    a, b = b, a
                return 2*gcd*cos((a + b)/2)*sin((a - b)/2)

        # 调用 process_common_addends 函数处理公共项，并传入 do 函数作为参数
        return process_common_addends(rv, do)  # 不要按自由符号筛选

    # 调用 bottom_up 函数，将 rv 和 f 函数作为参数传入
    return bottom_up(rv, f)
def TR10(rv, first=True):
    """Separate sums in ``cos`` and ``sin``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR10
    >>> from sympy.abc import a, b, c
    >>> from sympy import cos, sin
    >>> TR10(cos(a + b))
    -sin(a)*sin(b) + cos(a)*cos(b)
    >>> TR10(sin(a + b))
    sin(a)*cos(b) + sin(b)*cos(a)
    >>> TR10(sin(a + b + c))
    (-sin(a)*sin(b) + cos(a)*cos(b))*sin(c) + \
    (sin(a)*cos(b) + sin(b)*cos(a))*cos(c)
    """

    def f(rv):
        # 如果 rv 的函数不是 cos 或 sin，则返回原始 rv
        if rv.func not in (cos, sin):
            return rv

        f = rv.func
        arg = rv.args[0]
        # 如果 rv 的参数是一个加法表达式
        if arg.is_Add:
            if first:
                # 按顺序获取加法表达式中的参数
                args = list(ordered(arg.args))
            else:
                args = list(arg.args)
            # 弹出一个参数作为 a
            a = args.pop()
            # 剩余的参数组成一个新的加法表达式 b
            b = Add._from_args(args)
            # 如果 b 也是一个加法表达式
            if b.is_Add:
                if f == sin:
                    # 返回 sin(a)*TR10(cos(b), first=False) + cos(a)*TR10(sin(b), first=False)
                    return sin(a)*TR10(cos(b), first=False) + \
                        cos(a)*TR10(sin(b), first=False)
                else:
                    # 返回 cos(a)*TR10(cos(b), first=False) - sin(a)*TR10(sin(b), first=False)
                    return cos(a)*TR10(cos(b), first=False) - \
                        sin(a)*TR10(sin(b), first=False)
            else:
                if f == sin:
                    # 返回 sin(a)*cos(b) + cos(a)*sin(b)
                    return sin(a)*cos(b) + cos(a)*sin(b)
                else:
                    # 返回 cos(a)*cos(b) - sin(a)*sin(b)
                    return cos(a)*cos(b) - sin(a)*sin(b)
        # 如果 rv 的参数不是加法表达式，则返回原始 rv
        return rv

    # 调用 bottom_up 函数，将 f 应用到 rv 上并返回结果
    return bottom_up(rv, f)


def TR10i(rv):
    """Sum of products to function of sum.

    Examples
    ========

    >>> from sympy.simplify.fu import TR10i
    >>> from sympy import cos, sin, sqrt
    >>> from sympy.abc import x

    >>> TR10i(cos(1)*cos(3) + sin(1)*sin(3))
    cos(2)
    >>> TR10i(cos(1)*sin(3) + sin(1)*cos(3) + cos(3))
    cos(3) + sin(4)
    >>> TR10i(sqrt(2)*cos(x)*x + sqrt(6)*sin(x)*x)
    2*sqrt(2)*x*sin(x + pi/6)

    """
    global _ROOT2, _ROOT3, _invROOT3
    # 如果 _ROOT2 是 None，则调用 _roots 函数初始化全局变量
    if _ROOT2 is None:
        _roots()

    # 调用 bottom_up 函数，将 f 应用到 rv 上并返回结果
    return bottom_up(rv, f)


def TR11(rv, base=None):
    """Function of double angle to product. The ``base`` argument can be used
    to indicate what is the un-doubled argument, e.g. if 3*pi/7 is the base
    then cosine and sine functions with argument 6*pi/7 will be replaced.

    Examples
    ========

    >>> from sympy.simplify.fu import TR11
    >>> from sympy import cos, sin, pi
    >>> from sympy.abc import x
    >>> TR11(sin(2*x))
    2*sin(x)*cos(x)
    >>> TR11(cos(2*x))
    -sin(x)**2 + cos(x)**2
    >>> TR11(sin(4*x))
    4*(-sin(x)**2 + cos(x)**2)*sin(x)*cos(x)
    >>> TR11(sin(4*x/3))
    4*(-sin(x/3)**2 + cos(x/3)**2)*sin(x/3)*cos(x/3)

    If the arguments are simply integers, no change is made
    unless a base is provided:

    >>> TR11(cos(2))
    cos(2)
    >>> TR11(cos(4), 2)
    -sin(2)**2 + cos(2)**2

    There is a subtle issue here in that autosimplification will convert
    some higher angles to lower angles

    >>> cos(6*pi/7) + cos(3*pi/7)
    -cos(pi/7) + cos(3*pi/7)

    The 6*pi/7 angle is now pi/7 but can be targeted with TR11 by supplying
    the 3*pi/7 base:

    >>> TR11(_, 3*pi/7)

    """
    # 调用 bottom_up 函数，将 f 应用到 rv 上并返回结果
    return bottom_up(rv, f)
    # 计算表达式中的三角函数值，使用pi和常见的三角函数（sin、cos）
    -sin(3*pi/7)**2 + cos(3*pi/7)**2 + cos(3*pi/7)

    """
    
    # 定义一个函数f，接受一个参数rv
    def f(rv):
        # 如果rv的函数不是sin或cos，则直接返回rv
        if rv.func not in (cos, sin):
            return rv
        
        # 如果存在基础值base
        if base:
            # 将rv的函数赋给变量f，并计算f(base*2)的值给变量t
            f = rv.func
            t = f(base*2)
            co = S.One
            # 如果t是乘法形式，将其拆分为系数和非系数部分
            if t.is_Mul:
                co, t = t.as_coeff_Mul()
            # 如果t的函数不是sin或cos，则返回rv
            if t.func not in (cos, sin):
                return rv
            # 如果rv的参数等于t的参数
            if rv.args[0] == t.args[0]:
                c = cos(base)
                s = sin(base)
                # 如果rv的函数是cos，则返回(c^2 - s^2)/co；否则返回2*c*s/co
                if f is cos:
                    return (c**2 - s**2)/co
                else:
                    return 2*c*s/co
            return rv
        
        # 如果rv的参数的第一个元素不是数字
        elif not rv.args[0].is_Number:
            # 如果系数的分子能被2整除，则进行下面的变化
            c, m = rv.args[0].as_coeff_Mul(rational=True)
            if c.p % 2 == 0:
                # 计算新的参数arg，并创建TR11对象的cos和sin值
                arg = c.p//2*m/c.q
                c = TR11(cos(arg))
                s = TR11(sin(arg))
                # 如果rv的函数是sin，则将rv设置为2*s*c；否则设置为c^2 - s^2
                if rv.func == sin:
                    rv = 2*s*c
                else:
                    rv = c**2 - s**2
        # 返回处理后的rv
        return rv

    # 使用bottom_up函数递归地应用函数f到rv上，并返回结果
    return bottom_up(rv, f)
def _TR11(rv):
    """
    Helper for TR11 to find half-arguments for sin in factors of
    num/den that appear in cos or sin factors in the den/num.

    Examples
    ========

    >>> from sympy.simplify.fu import TR11, _TR11
    >>> from sympy import cos, sin
    >>> from sympy.abc import x
    >>> TR11(sin(x/3)/(cos(x/6)))
    sin(x/3)/cos(x/6)
    >>> _TR11(sin(x/3)/(cos(x/6)))
    2*sin(x/6)
    >>> TR11(sin(x/6)/(sin(x/3)))
    sin(x/6)/sin(x/3)
    >>> _TR11(sin(x/6)/(sin(x/3)))
    1/(2*cos(x/6))

    """
    def f(rv):
        # 如果 rv 不是表达式类型，则直接返回 rv
        if not isinstance(rv, Expr):
            return rv

        def sincos_args(flat):
            # 找出在 flat 的参数中作为 sin 和 cos 的基础出现的参数，
            # 并且其指数为整数
            args = defaultdict(set)
            for fi in Mul.make_args(flat):
                b, e = fi.as_base_exp()
                if e.is_Integer and e > 0:
                    if b.func in (cos, sin):
                        args[type(b)].add(b.args[0])
            return args
        
        # 将 rv 分解为分子和分母中 sin 和 cos 的参数集合
        num_args, den_args = map(sincos_args, rv.as_numer_denom())

        def handle_match(rv, num_args, den_args):
            # 遍历分子中的 sin 参数，查找其一半在分母中是否存在
            # 如果存在，则传递这个半角给 TR11 处理 rv
            for narg in num_args[sin]:
                half = narg/2
                if half in den_args[cos]:
                    func = cos
                elif half in den_args[sin]:
                    func = sin
                else:
                    continue
                rv = TR11(rv, half)
                den_args[func].remove(half)
            return rv
        
        # 处理分子中的 sin，分母中的 sin 或 cos
        rv = handle_match(rv, num_args, den_args)
        # 处理分母中的 sin，分子中的 sin 或 cos
        rv = handle_match(rv, den_args, num_args)
        return rv

    # 应用 bottom_up 函数来逐层应用变换函数 f 到 rv
    return bottom_up(rv, f)


def TR12(rv, first=True):
    """Separate sums in ``tan``.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import tan
    >>> from sympy.simplify.fu import TR12
    >>> TR12(tan(x + y))
    (tan(x) + tan(y))/(-tan(x)*tan(y) + 1)
    """

    def f(rv):
        # 如果 rv 不是 tan 函数，则直接返回 rv
        if not rv.func == tan:
            return rv

        arg = rv.args[0]
        if arg.is_Add:
            # 如果 tan 的参数是一个加法表达式
            if first:
                args = list(ordered(arg.args))  # 排序加法表达式的参数
            else:
                args = list(arg.args)
            a = args.pop()
            b = Add._from_args(args)
            if b.is_Add:
                tb = TR12(tan(b), first=False)  # 递归地应用 TR12 变换到 b
            else:
                tb = tan(b)
            # 返回 tan(a) + tan(b) 的结果
            return (tan(a) + tb)/(1 - tan(a)*tb)
        return rv

    # 应用 bottom_up 函数来逐层应用变换函数 f 到 rv
    return bottom_up(rv, f)


def TR12i(rv):
    """Combine tan arguments as
    (tan(y) + tan(x))/(tan(x)*tan(y) - 1) -> -tan(x + y).

    Examples
    ========

    >>> from sympy.simplify.fu import TR12i
    >>> from sympy import tan
    >>> from sympy.abc import a, b, c

    """
    # 对变量 a, b, c 分别求其 tan 值，并构建一个包含这些值的列表
    >>> ta, tb, tc = [tan(i) for i in (a, b, c)]
    # 计算表达式 (ta + tb)/(-ta*tb + 1)，并传递给函数 TR12i 进行处理
    >>> TR12i((ta + tb)/(-ta*tb + 1))
    # 计算 tan(a + b)
    tan(a + b)
    # 计算表达式 (ta + tb)/(ta*tb - 1)，并传递给函数 TR12i 进行处理
    >>> TR12i((ta + tb)/(ta*tb - 1))
    # 计算 -tan(a + b)
    -tan(a + b)
    # 计算表达式 (-ta - tb)/(ta*tb - 1)，并传递给函数 TR12i 进行处理
    >>> TR12i((-ta - tb)/(ta*tb - 1))
    # 计算 tan(a + b)
    tan(a + b)
    # 计算表达式 eq，并传递给函数 TR12i 进行处理
    >>> eq = (ta + tb)/(-ta*tb + 1)**2*(-3*ta - 3*tc)/(2*(ta*tc - 1))
    # 对 eq 进行展开并传递给函数 TR12i 进行处理
    >>> TR12i(eq.expand())
    # 计算 -3*tan(a + b)*tan(a + c)/(2*(tan(a) + tan(b) - 1))
    -3*tan(a + b)*tan(a + c)/(2*(tan(a) + tan(b) - 1))
    """
    # 返回函数 bottom_up(rv, f) 的计算结果
    return bottom_up(rv, f)
def TRmorrie(rv):
    """Transforms a product of cosines into a quotient involving sines.

    Examples
    ========

    >>> from sympy.simplify.fu import TRmorrie, TR8, TR3
    >>> from sympy.abc import x
    >>> from sympy import Mul, cos, pi
    >>> TRmorrie(cos(x)*cos(2*x))
    sin(4*x)/(4*sin(x))
    >>> TRmorrie(7*Mul(*[cos(x) for x in range(10)]))
    7*sin(12)*sin(16)*cos(5)*cos(7)*cos(9)/(64*sin(1)*sin(3))

    Sometimes autosimplification will cause a power to be
    not recognized. e.g. in the following, cos(4*pi/7) automatically
    simplifies to -cos(3*pi/7) so only 2 of the 3 terms are
    recognized:

    >>> TRmorrie(cos(pi/7)*cos(2*pi/7)*cos(4*pi/7))
    -sin(3*pi/7)*cos(3*pi/7)/(4*sin(pi/7))

    A touch by TR8 resolves the expression to a Rational

    >>> TR8(_)
    -1/8

    In this case, if eq is unsimplified, the answer is obtained
    directly:

    >>> eq = cos(pi/9)*cos(2*pi/9)*cos(3*pi/9)*cos(4*pi/9)
    >>> TRmorrie(eq)
    1/16

    But if angles are made canonical with TR3 then the answer
    is not simplified without further work:

    >>> TR3(eq)
    sin(pi/18)*cos(pi/9)*cos(2*pi/9)/2
    >>> TRmorrie(_)
    sin(pi/18)*sin(4*pi/9)/(8*sin(pi/9))
    >>> TR8(_)
    cos(7*pi/18)/(16*sin(pi/9))
    >>> TR3(_)
    1/16

    The original expression would have resolved to 1/16 directly with TR8,
    however:

    >>> TR8(eq)
    1/16

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Morrie%27s_law

    """

    def f(rv):
        if not rv.is_Mul:
            return rv

        # Separate cosine and non-cosine factors
        args = {cos: [], None: []}
        for a in Mul.make_args(rv):
            if a.func == cos:
                args[cos].append(a.args[0])
            else:
                args[None].append(a)

        cos_factors = args[cos]
        non_cos_factors = args[None]

        # If there are fewer than two cosine factors, return original expression
        if len(cos_factors) < 2:
            return rv

        # Process cosine factors into transformed expression involving sines
        while len(cos_factors) > 1:
            angle1 = cos_factors.pop()
            angle2 = cos_factors.pop()
            non_cos_factors.append(sin(angle1 + angle2) / (2 * sin(angle1)))
        
        if cos_factors:
            non_cos_factors.append(sin(cos_factors.pop()))

        return Mul(*non_cos_factors)

    return bottom_up(rv, f)
    # 定义函数 f，接受参数 rv 和 first（默认为 True）
    def f(rv, first=True):
        # 如果 rv 不是 Mul 类型，则直接返回 rv
        if not rv.is_Mul:
            return rv
        # 如果 first 为真，将 rv 分解为分子和分母，分别递归处理后返回它们的商
        if first:
            n, d = rv.as_numer_denom()
            return f(n, 0)/f(d, 0)

        # 使用 defaultdict 创建 args 字典，用于存储余弦函数系数的列表
        args = defaultdict(list)
        # 创建空字典 coss，用于存储余弦函数和它们的指数
        coss = {}
        # 创建空列表 other，用于存储非余弦函数的表达式
        other = []

        # 遍历 rv 的每个元素 c
        for c in rv.args:
            # 将 c 分解为底数 b 和指数 e
            b, e = c.as_base_exp()
            # 如果 e 是整数并且 b 是 cos 函数
            if e.is_Integer and isinstance(b, cos):
                # 将 b 的系数和参数 a 分解出来，将系数添加到 args 字典对应的列表中
                co, a = b.args[0].as_coeff_Mul()
                args[a].append(co)
                # 将余弦函数 b 记录到 coss 字典中，其指数记录为 e
                coss[b] = e
            else:
                # 如果不是余弦函数，则添加到 other 列表中
                other.append(c)

        # 创建空列表 new，用于存储新生成的表达式
        new = []
        # 遍历 args 字典中的每个参数 a
        for a in args:
            # 获取系数列表 c 并进行排序
            c = args[a]
            c.sort()
            # 循环处理每个系数
            while c:
                k = 0
                cc = ci = c[0]
                # 计算连续相同系数 cc 的个数 k
                while cc in c:
                    k += 1
                    cc *= 2
                # 如果 k 大于 1，生成新的表达式 newarg
                if k > 1:
                    newarg = sin(2**k*ci*a)/2**k/sin(ci*a)
                    # 确定可以应用的次数 take
                    take = None
                    ccs = []
                    # 循环 k 次，计算更新 coss 字典中余弦函数的指数
                    for i in range(k):
                        cc /= 2
                        key = cos(a*cc, evaluate=False)
                        ccs.append(cc)
                        take = min(coss[key], take or coss[key])
                    # 更新余弦函数的指数
                    for i in range(k):
                        cc = ccs.pop()
                        key = cos(a*cc, evaluate=False)
                        coss[key] -= take
                        # 如果指数为 0，则移除该系数 cc
                        if not coss[key]:
                            c.remove(cc)
                    # 将新表达式的 take 次方加入 new 列表
                    new.append(newarg**take)
                else:
                    # 否则生成新的 cos(b*a) 表达式并加入 other 列表
                    b = cos(c.pop(0)*a)
                    other.append(b**coss[b])

        # 如果 new 列表不为空，重新构建 rv 表达式并返回
        if new:
            rv = Mul(*(new + other + [
                cos(k*a, evaluate=False) for a in args for k in args[a]]))

        # 调用 bottom_up 函数，将 rv 和 f 函数作为参数返回结果
        return bottom_up(rv, f)

    # 返回调用 bottom_up 函数的结果
    return bottom_up(rv, f)
# 将给定的表达式中的双曲函数幂次简化为更简单的形式。
def TR14(rv, first=True):
    """Convert factored powers of sin and cos identities into simpler
    expressions.

    Examples
    ========

    >>> from sympy.simplify.fu import TR14
    >>> from sympy.abc import x, y
    >>> from sympy import cos, sin
    >>> TR14((cos(x) - 1)*(cos(x) + 1))
    -sin(x)**2
    >>> TR14((sin(x) - 1)*(sin(x) + 1))
    -cos(x)**2
    >>> p1 = (cos(x) + 1)*(cos(x) - 1)
    >>> p2 = (cos(y) - 1)*2*(cos(y) + 1)
    >>> p3 = (3*(cos(y) - 1))*(3*(cos(y) + 1))
    >>> TR14(p1*p2*p3*(x - 1))
    -18*(x - 1)*sin(x)**2*sin(y)**4

    """

    # 应用底向上的方法将函数应用到表达式上
    return bottom_up(rv, f)


# 将 sin(x)**-2 转换为 1 + cot(x)**2 的形式
def TR15(rv, max=4, pow=False):
    """Convert sin(x)**-2 to 1 + cot(x)**2.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR15
    >>> from sympy.abc import x
    >>> from sympy import sin
    >>> TR15(1 - 1/sin(x)**2)
    -cot(x)**2

    """

    def f(rv):
        if not (isinstance(rv, Pow) and isinstance(rv.base, sin)):
            return rv

        e = rv.exp
        if e % 2 == 1:
            return TR15(rv.base**(e + 1))/rv.base

        ia = 1/rv
        a = _TR56(ia, sin, cot, lambda x: 1 + x, max=max, pow=pow)
        if a != ia:
            rv = a
        return rv

    # 应用底向上的方法将函数应用到表达式上
    return bottom_up(rv, f)


# 将 cos(x)**-2 转换为 1 + tan(x)**2 的形式
def TR16(rv, max=4, pow=False):
    """Convert cos(x)**-2 to 1 + tan(x)**2.

    See _TR56 docstring for advanced use of ``max`` and ``pow``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR16
    >>> from sympy.abc import x
    >>> from sympy import cos
    >>> TR16(1 - 1/cos(x)**2)
    -tan(x)**2

    """

    def f(rv):
        if not (isinstance(rv, Pow) and isinstance(rv.base, cos)):
            return rv

        e = rv.exp
        if e % 2 == 1:
            return TR15(rv.base**(e + 1))/rv.base

        ia = 1/rv
        a = _TR56(ia, cos, tan, lambda x: 1 + x, max=max, pow=pow)
        if a != ia:
            rv = a
        return rv

    # 应用底向上的方法将函数应用到表达式上
    return bottom_up(rv, f)


# 将 f(x)**-i 转换为 g(x)**i 的形式，其中 i 是整数或基数为正且 f, g 是：tan, cot; sin, csc; 或 cos, sec。
def TR111(rv):
    """Convert f(x)**-i to g(x)**i where either ``i`` is an integer
    or the base is positive and f, g are: tan, cot; sin, csc; or cos, sec.

    Examples
    ========

    >>> from sympy.simplify.fu import TR111
    >>> from sympy.abc import x
    >>> from sympy import tan
    >>> TR111(1 - 1/tan(x)**2)
    1 - cot(x)**2

    """

    def f(rv):
        if not (
            isinstance(rv, Pow) and
            (rv.base.is_positive or rv.exp.is_integer and rv.exp.is_negative)):
            return rv

        if isinstance(rv.base, tan):
            return cot(rv.base.args[0])**-rv.exp
        elif isinstance(rv.base, sin):
            return csc(rv.base.args[0])**-rv.exp
        elif isinstance(rv.base, cos):
            return sec(rv.base.args[0])**-rv.exp
        return rv

    # 应用底向上的方法将函数应用到表达式上
    return bottom_up(rv, f)


# 将 tan(x)**2 转换为 sec(x)**2 - 1，cot(x)**2 转换为 csc(x)**2 - 1
    # 文档字符串，说明函数的高级使用情况，特别是在处理 max 和 pow 时，请参阅 _TR56 的文档字符串。

    # 定义函数 f，接受一个参数 rv
    def f(rv):
        # 如果 rv 不是 Pow 类型或者 rv 的基数函数不是 cot 或 tan，则直接返回 rv
        if not (isinstance(rv, Pow) and rv.base.func in (cot, tan)):
            return rv

        # 对 rv 分别调用 _TR56 函数，处理 tan 的情况，将 tan 转换为 sec，指数函数减一，使用 max 和 pow 参数
        rv = _TR56(rv, tan, sec, lambda x: x - 1, max=max, pow=pow)
        # 对 rv 分别调用 _TR56 函数，处理 cot 的情况，将 cot 转换为 csc，指数函数减一，使用 max 和 pow 参数
        rv = _TR56(rv, cot, csc, lambda x: x - 1, max=max, pow=pow)
        # 返回转换后的 rv
        return rv

    # 调用 bottom_up 函数，传入参数 rv 和函数 f，返回处理后的结果
    return bottom_up(rv, f)
# 定义函数 TRpower，用于将带有正整数幂的 sin(x) 和 cos(x) 转换为求和形式的表达式
def TRpower(rv):
    """Convert sin(x)**n and cos(x)**n with positive n to sums.

    Examples
    ========

    >>> from sympy.simplify.fu import TRpower
    >>> from sympy.abc import x
    >>> from sympy import cos, sin
    >>> TRpower(sin(x)**6)
    -15*cos(2*x)/32 + 3*cos(4*x)/16 - cos(6*x)/32 + 5/16
    >>> TRpower(sin(x)**3*cos(2*x)**4)
    (3*sin(x)/4 - sin(3*x)/4)*(cos(4*x)/2 + cos(8*x)/8 + 3/8)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Power-reduction_formulae

    """

    # 定义内部函数 f(rv)，用于递归地转换表达式中的 sin(x)**n 和 cos(x)**n
    def f(rv):
        # 如果 rv 不是 Pow 类型或者其基数不是 sin 或 cos 函数，则返回原始 rv
        if not (isinstance(rv, Pow) and isinstance(rv.base, (sin, cos))):
            return rv
        # 将 rv 分解为基数和指数
        b, n = rv.as_base_exp()
        x = b.args[0]
        # 如果指数 n 是正整数
        if n.is_Integer and n.is_positive:
            # 处理 cos(x)**n，其中 n 是奇数
            if n.is_odd and isinstance(b, cos):
                rv = 2**(1-n)*Add(*[binomial(n, k)*cos((n - 2*k)*x)
                    for k in range((n + 1)//2)])
            # 处理 sin(x)**n，其中 n 是奇数
            elif n.is_odd and isinstance(b, sin):
                rv = 2**(1-n)*S.NegativeOne**((n-1)//2)*Add(*[binomial(n, k)*
                    S.NegativeOne**k*sin((n - 2*k)*x) for k in range((n + 1)//2)])
            # 处理 cos(x)**n，其中 n 是偶数
            elif n.is_even and isinstance(b, cos):
                rv = 2**(1-n)*Add(*[binomial(n, k)*cos((n - 2*k)*x)
                    for k in range(n//2)])
            # 处理 sin(x)**n，其中 n 是偶数
            elif n.is_even and isinstance(b, sin):
                rv = 2**(1-n)*S.NegativeOne**(n//2)*Add(*[binomial(n, k)*
                    S.NegativeOne**k*cos((n - 2*k)*x) for k in range(n//2)])
            # 如果 n 是偶数，还需添加一个额外项
            if n.is_even:
                rv += 2**(-n)*binomial(n, n//2)
        return rv

    # 应用函数 f 到输入 rv 的所有子表达式，使用 bottom_up 函数逐层应用
    return bottom_up(rv, f)


# 定义函数 L，用于计算表达式中三角函数的数量
def L(rv):
    """Return count of trigonometric functions in expression.

    Examples
    ========

    >>> from sympy.simplify.fu import L
    >>> from sympy.abc import x
    >>> from sympy import cos, sin
    >>> L(cos(x)+sin(x))
    2
    """
    # 使用 sympy 的 count 方法计算表达式中三角函数的个数，并返回结果
    return S(rv.count(TrigonometricFunction))


# ============== end of basic Fu-like tools =====================

# 如果 SYMPY_DEBUG 为真，则对以下变量应用 debug 函数，并赋值给相应变量
if SYMPY_DEBUG:
    (TR0, TR1, TR2, TR3, TR4, TR5, TR6, TR7, TR8, TR9, TR10, TR11, TR12, TR13,
    TR2i, TRmorrie, TR14, TR15, TR16, TR12i, TR111, TR22
    )= list(map(debug,
    (TR0, TR1, TR2, TR3, TR4, TR5, TR6, TR7, TR8, TR9, TR10, TR11, TR12, TR13,
    TR2i, TRmorrie, TR14, TR15, TR16, TR12i, TR111, TR22)))


# 元组表示链式转换 -- (f, g) -> lambda x: g(f(x))
# 列表表示选择式转换 -- [f, g] -> lambda x: min(f(x), g(x), key=objective)

# 定义 CTR1，表示选择式转换，包含两个可能的转换序列和一个恒等变换
CTR1 = [(TR5, TR0), (TR6, TR0), identity]

# 定义 CTR2，表示选择式转换，使用 TR11 或者两个可能的转换序列和一个恒等变换
CTR2 = (TR11, [(TR5, TR0), (TR6, TR0), TR0])

# 定义 CTR3，表示选择式转换，包含两个可能的转换序列和一个恒等变换
CTR3 = [(TRmorrie, TR8, TR0), (TRmorrie, TR8, TR10i, TR0), identity]

# 定义 CTR4，表示选择式转换，包含两个可能的转换序列和一个恒等变换
CTR4 = [(TR4, TR10i), identity]

# 定义 RL1，表示链式转换序列
RL1 = (TR4, TR3, TR4, TR12, TR4, TR13, TR4, TR0)


# XXX 不太清楚如何实现这个转换
# 参见 Fu 论文第 7 页的引用，Union 符号是指什么？
# 图表显示所有这些是一条转换链，但文本指出它们独立应用
# 如果 L 开始增加，没有实现断开
RL2 = [
    # 创建一个包含多个元组的列表，每个元组中包含不同数量的元素
    (TR4, TR3, TR10, TR4, TR3, TR11),
    # 创建一个元组，包含4个元素
    (TR5, TR7, TR11, TR4),
    # 创建一个元组，包含8个元素，其中有重复的TR9出现多次
    (CTR3, CTR1, TR9, CTR2, TR4, TR9, TR9, CTR4),
    # 使用identity函数作为列表的最后一个元素
    identity,
    ]
def fu(rv, measure=lambda x: (L(x), x.count_ops())):
    """Attempt to simplify expression by using transformation rules given
    in the algorithm by Fu et al.

    :func:`fu` will try to minimize the objective function ``measure``.
    By default this first minimizes the number of trig terms and then minimizes
    the number of total operations.

    Examples
    ========

    >>> from sympy.simplify.fu import fu
    >>> from sympy import cos, sin, tan, pi, S, sqrt
    >>> from sympy.abc import x, y, a, b

    >>> fu(sin(50)**2 + cos(50)**2 + sin(pi/6))
    3/2
    >>> fu(sqrt(6)*cos(x) + sqrt(2)*sin(x))
    2*sqrt(2)*sin(x + pi/3)

    CTR1 example

    >>> eq = sin(x)**4 - cos(y)**2 + sin(y)**2 + 2*cos(x)**2
    >>> fu(eq)
    cos(x)**4 - 2*cos(y)**2 + 2

    CTR2 example

    >>> fu(S.Half - cos(2*x)/2)
    sin(x)**2

    CTR3 example

    >>> fu(sin(a)*(cos(b) - sin(b)) + cos(a)*(sin(b) + cos(b)))
    sqrt(2)*sin(a + b + pi/4)

    CTR4 example

    >>> fu(sqrt(3)*cos(x)/2 + sin(x)/2)
    sin(x + pi/3)

    Example 1

    >>> fu(1-sin(2*x)**2/4-sin(y)**2-cos(x)**4)
    -cos(x)**2 + cos(y)**2

    Example 2

    >>> fu(cos(4*pi/9))
    sin(pi/18)
    >>> fu(cos(pi/9)*cos(2*pi/9)*cos(3*pi/9)*cos(4*pi/9))
    1/16

    Example 3

    >>> fu(tan(7*pi/18)+tan(5*pi/18)-sqrt(3)*tan(5*pi/18)*tan(7*pi/18))
    -sqrt(3)

    Objective function example

    >>> fu(sin(x)/cos(x))  # default objective function
    tan(x)
    >>> fu(sin(x)/cos(x), measure=lambda x: -x.count_ops()) # maximize op count
    sin(x)/cos(x)

    References
    ==========

    .. [1] https://www.sciencedirect.com/science/article/pii/S0895717706001609
    """
    fRL1 = greedy(RL1, measure)  # 使用 RL1 策略函数进行贪婪优化
    fRL2 = greedy(RL2, measure)  # 使用 RL2 策略函数进行贪婪优化

    was = rv  # 保存原始表达式
    rv = sympify(rv)  # 将输入的表达式转换为 SymPy 的表达式对象
    if not isinstance(rv, Expr):  # 如果 rv 不是 SymPy 表达式对象
        return rv.func(*[fu(a, measure=measure) for a in rv.args])  # 递归应用 fu 函数到 rv 的每个参数
    rv = TR1(rv)  # 应用 TR1 变换规则到 rv
    if rv.has(tan, cot):  # 如果 rv 中包含 tan 或 cot 函数
        rv1 = fRL1(rv)  # 使用 fRL1 进行优化
        if (measure(rv1) < measure(rv)):  # 如果 rv1 的度量值比 rv 小
            rv = rv1  # 更新 rv 为 rv1
        if rv.has(tan, cot):  # 如果 rv 中仍然包含 tan 或 cot 函数
            rv = TR2(rv)  # 应用 TR2 变换规则到 rv
    if rv.has(sin, cos):  # 如果 rv 中包含 sin 或 cos 函数
        rv1 = fRL2(rv)  # 使用 fRL2 进行优化
        rv2 = TR8(TRmorrie(rv1))  # 应用 TRmorrie 和 TR8 变换规则到 rv1
        rv = min([was, rv, rv1, rv2], key=measure)  # 从 was, rv, rv1, rv2 中选择度量值最小的表达式
    return min(TR2i(rv), rv, key=measure)  # 返回度量值最小的 TR2i(rv), rv 中的表达式


def process_common_addends(rv, do, key2=None, key1=True):
    """Apply ``do`` to addends of ``rv`` that (if ``key1=True``) share at least
    a common absolute value of their coefficient and the value of ``key2`` when
    applied to the argument. If ``key1`` is False ``key2`` must be supplied and
    will be the only key applied.
    """

    # collect by absolute value of coefficient and key2
    absc = defaultdict(list)
    if key1:
        for a in rv.args:
            c, a = a.as_coeff_Mul()  # 提取参数 a 的系数和内容
            if c < 0:
                c = -c  # 将系数转为正数
                a = -a  # 将内容转为相反数，保持符号一致性
            absc[(c, key2(a) if key2 else 1)].append(a)  # 根据系数和 key2(a) 将参数 a 归类到 absc 字典中
    elif key2:
        for a in rv.args:
            absc[(S.One, key2(a))].append(a)  # 根据 key2(a) 将参数 a 归类到 absc 字典中
    else:
        # 如果字典 absc 是空的，抛出数值错误异常，要求至少有一个键
        raise ValueError('must have at least one key')

    args = []
    hit = False
    # 遍历字典 absc 的键值对
    for k in absc:
        # 获取键 k 对应的值 v
        v = absc[k]
        # 解构赋值，将 k 拆分为 c 和 _
        c, _ = k
        # 如果 v 的长度大于1
        if len(v) > 1:
            # 创建一个加法表达式对象 e，将 v 中的元素作为参数，evaluate=False 表示不立即求值
            e = Add(*v, evaluate=False)
            # 对 e 执行处理函数 do，获取处理后的新对象 new
            new = do(e)
            # 如果处理后的新对象 new 不等于原对象 e，则更新 e，并标记 hit 为 True
            if new != e:
                e = new
                hit = True
            # 将计算结果 c*e 添加到参数列表 args 中
            args.append(c * e)
        else:
            # 将计算结果 c*v[0] 添加到参数列表 args 中
            args.append(c * v[0])
    # 如果标记 hit 为 True，则创建一个新的加法表达式对象 rv，参数为 args 中的所有元素
    if hit:
        rv = Add(*args)

    # 返回最终的计算结果 rv
    return rv
# 定义一个字符串，包含多个字符串“TR0”到“TR22”作为键，对应于当前作用域中这些字符串的值
fufuncs = '''
    TR0 TR1 TR2 TR3 TR4 TR5 TR6 TR7 TR8 TR9 TR10 TR10i TR11
    TR12 TR13 L TR2i TRmorrie TR12i
    TR14 TR15 TR16 TR111 TR22'''.split()

# 创建一个字典 FU，将每个字符串键映射到其在当前作用域中的值
FU = dict(list(zip(fufuncs, list(map(locals().get, fufuncs)))))

# 定义一个函数 _roots，用于设置全局变量 _ROOT2, _ROOT3 和 _invROOT3
def _roots():
    global _ROOT2, _ROOT3, _invROOT3
    # 设置 _ROOT2 和 _ROOT3 为 2 和 3 的平方根
    _ROOT2, _ROOT3 = sqrt(2), sqrt(3)
    # 设置 _invROOT3 为 3 的倒数
    _invROOT3 = 1/_ROOT3

# 初始化 _ROOT2 为 None
_ROOT2 = None

# 定义函数 trig_split，用于计算三角函数组合的一些复杂情况
def trig_split(a, b, two=False):
    """Return the gcd, s1, s2, a1, a2, bool where

    If two is False (default) then::
        a + b = gcd*(s1*f(a1) + s2*f(a2)) where f = cos if bool else sin
    else:
        if bool, a + b was +/- cos(a1)*cos(a2) +/- sin(a1)*sin(a2) and equals
            n1*gcd*cos(a - b) if n1 == n2 else
            n1*gcd*cos(a + b)
        else a + b was +/- cos(a1)*sin(a2) +/- sin(a1)*cos(a2) and equals
            n1*gcd*sin(a + b) if n1 = n2 else
            n1*gcd*sin(b - a)

    Examples
    ========

    >>> from sympy.simplify.fu import trig_split
    >>> from sympy.abc import x, y, z
    >>> from sympy import cos, sin, sqrt

    >>> trig_split(cos(x), cos(y))
    (1, 1, 1, x, y, True)
    >>> trig_split(2*cos(x), -2*cos(y))
    (2, 1, -1, x, y, True)
    >>> trig_split(cos(x)*sin(y), cos(y)*sin(y))
    (sin(y), 1, 1, x, y, True)

    >>> trig_split(cos(x), -sqrt(3)*sin(x), two=True)
    (2, 1, -1, x, pi/6, False)
    >>> trig_split(cos(x), sin(x), two=True)
    (sqrt(2), 1, 1, x, pi/4, False)
    >>> trig_split(cos(x), -sin(x), two=True)
    (sqrt(2), 1, -1, x, pi/4, False)
    >>> trig_split(sqrt(2)*cos(x), -sqrt(6)*sin(x), two=True)
    (2*sqrt(2), 1, -1, x, pi/6, False)
    >>> trig_split(-sqrt(6)*cos(x), -sqrt(2)*sin(x), two=True)
    (-2*sqrt(2), 1, 1, x, pi/3, False)
    >>> trig_split(cos(x)/sqrt(6), sin(x)/sqrt(2), two=True)
    (sqrt(6)/3, 1, 1, x, pi/6, False)
    >>> trig_split(-sqrt(6)*cos(x)*sin(y), -sqrt(2)*sin(x)*sin(y), two=True)
    (-2*sqrt(2)*sin(y), 1, 1, x, pi/3, False)

    >>> trig_split(cos(x), sin(x))
    >>> trig_split(cos(x), sin(z))
    >>> trig_split(2*cos(x), -sin(x))
    >>> trig_split(cos(x), -sqrt(3)*sin(x))
    >>> trig_split(cos(x)*cos(y), sin(x)*sin(z))
    >>> trig_split(cos(x)*cos(y), sin(x)*sin(y))
    >>> trig_split(-sqrt(6)*cos(x), sqrt(2)*sin(x)*sin(y), two=True)
    """
    global _ROOT2, _ROOT3, _invROOT3
    # 如果 _ROOT2 是 None，则调用 _roots 函数初始化全局变量
    if _ROOT2 is None:
        _roots()

    # 将 a 和 b 转换为 Factors 对象
    a, b = [Factors(i) for i in (a, b)]
    # 获得归一化后的 Factors 对象及其最大公约数
    ua, ub = a.normal(b)
    gcd = a.gcd(b).as_expr()
    n1 = n2 = 1
    # 处理特殊因子 -1 的情况
    if S.NegativeOne in ua.factors:
        ua = ua.quo(S.NegativeOne)
        n1 = -n1
    elif S.NegativeOne in ub.factors:
        ub = ub.quo(S.NegativeOne)
        n2 = -n2
    # 将 ua 和 ub 转换为表达式
    a, b = [i.as_expr() for i in (ua, ub)]
    def pow_cos_sin(a, two):
        """Return ``a`` as a tuple (r, c, s) such that
        ``a = (r or 1)*(c or 1)*(s or 1)``.

        Three arguments are returned (radical, c-factor, s-factor) as
        long as the conditions set by ``two`` are met; otherwise None is
        returned. If ``two`` is True there will be one or two non-None
        values in the tuple: c and s or c and r or s and r or s or c with c
        being a cosine function (if possible) else a sine, and s being a sine
        function (if possible) else oosine. If ``two`` is False then there
        will only be a c or s term in the tuple.

        ``two`` also require that either two cos and/or sin be present (with
        the condition that if the functions are the same the arguments are
        different or vice versa) or that a single cosine or a single sine
        be present with an optional radical.

        If the above conditions dictated by ``two`` are not met then None
        is returned.
        """
        # 初始化 c 和 s 为 None
        c = s = None
        # 将常数系数设置为 S.One
        co = S.One

        # 如果 a 是乘法表达式
        if a.is_Mul:
            # 尝试将 a 分解为常数系数和剩余部分
            co, a = a.as_coeff_Mul()
            # 如果剩余部分的参数数量大于2或者 two 参数为 False，则返回 None
            if len(a.args) > 2 or not two:
                return None
            # 如果剩余部分是乘法表达式
            if a.is_Mul:
                args = list(a.args)
            else:
                args = [a]
            # 将第一个参数弹出
            a = args.pop(0)
            # 判断第一个参数的类型并赋值给 c 或 s
            if isinstance(a, cos):
                c = a
            elif isinstance(a, sin):
                s = a
            elif a.is_Pow and a.exp is S.Half:  # autoeval doesn't allow -1/2
                co *= a
            else:
                return None
            # 如果还有剩余参数
            if args:
                b = args[0]
                # 判断第二个参数的类型并赋值给 c 或 s
                if isinstance(b, cos):
                    if c:
                        s = b
                    else:
                        c = b
                elif isinstance(b, sin):
                    if s:
                        c = b
                    else:
                        s = b
                elif b.is_Pow and b.exp is S.Half:
                    co *= b
                else:
                    return None
            # 返回 co，如果 co 是 S.One 则返回 None，否则返回 co, c, s
            return co if co is not S.One else None, c, s
        # 如果 a 是 cos 函数
        elif isinstance(a, cos):
            c = a
        # 如果 a 是 sin 函数
        elif isinstance(a, sin):
            s = a
        # 如果 c 和 s 都是 None，则返回 None
        if c is None and s is None:
            return
        # 如果 co 是 S.One，则设为 None
        co = co if co is not S.One else None
        # 返回 co, c, s
        return co, c, s

    # 获取 a 和 b 的符号分解
    m = pow_cos_sin(a, two)
    # 如果 m 为 None，则返回 None
    if m is None:
        return
    # 将 m 中的值分别赋给 coa, ca, sa
    coa, ca, sa = m
    # 获取 b 的符号分解
    m = pow_cos_sin(b, two)
    # 如果 m 为 None，则返回 None
    if m is None:
        return
    # 将 m 中的值分别赋给 cob, cb, sb
    cob, cb, sb = m

    # 检查它们
    # 如果 ca 是 None 并且 cb 不是 None，或者 ca 是 sin 类型
    if (not ca) and cb or ca and isinstance(ca, sin):
        # 交换 coa, ca, sa 和 cob, cb, sb 的值
        coa, ca, sa, cob, cb, sb = cob, cb, sb, coa, ca, sa
        # 交换 n1 和 n2 的值
        n1, n2 = n2, n1
    # 如果 two 是 False，需要 cos(x) 和 cos(y) 或 sin(x) 和 sin(y)
    if not two:
        # 设置 c 和 s 的值
        c = ca or sa
        s = cb or sb
        # 如果 c 不是 s 的函数类型，则返回 None
        if not isinstance(c, s.func):
            return None
        # 返回 gcd, n1, n2, c 的参数，s 的参数，以及 c 是否是 cos 类型的布尔值
        return gcd, n1, n2, c.args[0], s.args[0], isinstance(c, cos)
    # 如果不满足条件，则直接返回，不做任何操作
    else:
        # 如果两个条件均为假，则继续执行
        if not coa and not cob:
            # 如果ca和cb均为真，并且sa和sb均为真，则继续执行
            if (ca and cb and sa and sb):
                # 如果ca的类型不等于sa的函数类型，则返回
                if isinstance(ca, sa.func) is not isinstance(cb, sb.func):
                    return
                # 将ca和sa的参数放入集合args中
                args = {j.args for j in (ca, sa)}
                # 如果cb和sb的参数不都在args中，则返回
                if not all(i.args in args for i in (cb, sb)):
                    return
                # 返回gcd, n1, n2, ca的第一个参数, sa的第一个参数, 判断ca是否为sa的函数类型
                return gcd, n1, n2, ca.args[0], sa.args[0], isinstance(ca, sa.func)
        # 如果ca和sa任一为真，或者cb和sb任一为真，或者two为真并且ca和sa均为空或cb和sb均为空，则返回
        if ca and sa or cb and sb or \
            two and (ca is None and sa is None or cb is None and sb is None):
            return
        # 如果ca为真，则将c赋值为ca；如果sa为真，则将s赋值为sb
        c = ca or sa
        s = cb or sb
        # 如果c的参数不等于s的参数，则返回
        if c.args != s.args:
            return
        # 如果coa为假，则将其赋值为S.One
        if not coa:
            coa = S.One
        # 如果cob为假，则将其赋值为S.One
        if not cob:
            cob = S.One
        # 如果coa等于cob，则更新gcd的值，返回gcd, n1, n2, c的第一个参数, pi/4, False
        if coa is cob:
            gcd *= _ROOT2
            return gcd, n1, n2, c.args[0], pi/4, False
        # 如果coa除以cob等于_ROOT3，则更新gcd的值，返回gcd, n1, n2, c的第一个参数, pi/3, False
        elif coa/cob == _ROOT3:
            gcd *= 2*cob
            return gcd, n1, n2, c.args[0], pi/3, False
        # 如果coa除以cob等于_invROOT3，则更新gcd的值，返回gcd, n1, n2, c的第一个参数, pi/6, False
        elif coa/cob == _invROOT3:
            gcd *= 2*coa
            return gcd, n1, n2, c.args[0], pi/6, False
# 如果表达式 ``e`` 可以写成 ``g*(a + s)`` 的形式，其中 ``s`` 是 ``+/-1``，返回 ``g``、``a`` 和 ``s``，
# 其中 ``a`` 不带有前导负系数。

def as_f_sign_1(e):
    """If ``e`` is a sum that can be written as ``g*(a + s)`` where
    ``s`` is ``+/-1``, return ``g``, ``a``, and ``s`` where ``a`` does
    not have a leading negative coefficient.

    Examples
    ========

    >>> from sympy.simplify.fu import as_f_sign_1
    >>> from sympy.abc import x
    >>> as_f_sign_1(x + 1)
    (1, x, 1)
    >>> as_f_sign_1(x - 1)
    (1, x, -1)
    >>> as_f_sign_1(-x + 1)
    (-1, x, -1)
    >>> as_f_sign_1(-x - 1)
    (-1, x, 1)
    >>> as_f_sign_1(2*x + 2)
    (2, x, 1)
    """
    # 如果表达式不是加法或者加法项的数量不是2，则返回空
    if not e.is_Add or len(e.args) != 2:
        return

    # 精确匹配
    a, b = e.args
    if a in (S.NegativeOne, S.One):
        g = S.One
        # 如果 b 是乘法并且第一个参数是数值并且小于0，则将 a 和 b 取反，并将 g 取反
        if b.is_Mul and b.args[0].is_Number and b.args[0] < 0:
            a, b = -a, -b
            g = -g
        return g, b, a

    # gcd 匹配
    a, b = [Factors(i) for i in e.args]
    ua, ub = a.normal(b)
    gcd = a.gcd(b).as_expr()
    if S.NegativeOne in ua.factors:
        ua = ua.quo(S.NegativeOne)
        n1 = -1
        n2 = 1
    elif S.NegativeOne in ub.factors:
        ub = ub.quo(S.NegativeOne)
        n1 = 1
        n2 = -1
    else:
        n1 = n2 = 1
    a, b = [i.as_expr() for i in (ua, ub)]
    if a is S.One:
        a, b = b, a
        n1, n2 = n2, n1
    if n1 == -1:
        gcd = -gcd
        n2 = -n2

    if b is S.One:
        return gcd, a, n2


# 将所有双曲函数替换为使用 Osborne 规则的三角函数
def _osborne(e, d):
    """Replace all hyperbolic functions with trig functions using
    the Osborne rule.

    Notes
    =====

    ``d`` is a dummy variable to prevent automatic evaluation
    of trigonometric/hyperbolic functions.


    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    """

    def f(rv):
        if not isinstance(rv, HyperbolicFunction):
            return rv
        a = rv.args[0]
        # 如果 a 不是加法，则将 a 乘以 d；否则，将 a 的每个项乘以 d
        a = a*d if not a.is_Add else Add._from_args([i*d for i in a.args])
        if isinstance(rv, sinh):
            return I*sin(a)
        elif isinstance(rv, cosh):
            return cos(a)
        elif isinstance(rv, tanh):
            return I*tan(a)
        elif isinstance(rv, coth):
            return cot(a)/I
        elif isinstance(rv, sech):
            return sec(a)
        elif isinstance(rv, csch):
            return csc(a)/I
        else:
            raise NotImplementedError('unhandled %s' % rv.func)

    return bottom_up(e, f)


# 将所有三角函数替换为使用 Osborne 规则的双曲函数
def _osbornei(e, d):
    """Replace all trig functions with hyperbolic functions using
    the Osborne rule.

    Notes
    =====

    ``d`` is a dummy variable to prevent automatic evaluation
    of trigonometric/hyperbolic functions.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    """
    # 定义函数 f，接受参数 rv
    def f(rv):
        # 检查 rv 是否不是 TrigonometricFunction 的实例，若不是则直接返回 rv
        if not isinstance(rv, TrigonometricFunction):
            return rv
        # 从 rv 的参数中分离出常数和变量 x，as_Add=True 表示将结果作为 Add 对象返回
        const, x = rv.args[0].as_independent(d, as_Add=True)
        # 将变量 d 替换为 S.One，并加上常数 const 乘以虚数单位 I，得到复数 a
        a = x.xreplace({d: S.One}) + const*I
        # 根据 rv 的具体类型进行不同的处理
        if isinstance(rv, sin):
            # 若 rv 是 sin 类型，则返回 sinh(a) 除以虚数单位 I
            return sinh(a)/I
        elif isinstance(rv, cos):
            # 若 rv 是 cos 类型，则返回 cosh(a)
            return cosh(a)
        elif isinstance(rv, tan):
            # 若 rv 是 tan 类型，则返回 tanh(a) 除以虚数单位 I
            return tanh(a)/I
        elif isinstance(rv, cot):
            # 若 rv 是 cot 类型，则返回 coth(a) 乘以虚数单位 I
            return coth(a)*I
        elif isinstance(rv, sec):
            # 若 rv 是 sec 类型，则返回 sech(a)
            return sech(a)
        elif isinstance(rv, csc):
            # 若 rv 是 csc 类型，则返回 csch(a) 乘以虚数单位 I
            return csch(a)*I
        else:
            # 如果 rv 是未处理的其他类型，则引发 NotImplementedError 异常
            raise NotImplementedError('unhandled %s' % rv.func)

    # 对表达式 e 应用自底向上的转换，使用函数 f 进行转换
    return bottom_up(e, f)
# 将给定表达式中的双曲函数用三角函数表示，并返回转换后的表达式及反向转换函数
def hyper_as_trig(rv):
    """Return an expression containing hyperbolic functions in terms
    of trigonometric functions. Any trigonometric functions initially
    present are replaced with Dummy symbols and the function to undo
    the masking and the conversion back to hyperbolics is also returned. It
    should always be true that::

        t, f = hyper_as_trig(expr)
        expr == f(t)

    Examples
    ========

    >>> from sympy.simplify.fu import hyper_as_trig, fu
    >>> from sympy.abc import x
    >>> from sympy import cosh, sinh
    >>> eq = sinh(x)**2 + cosh(x)**2
    >>> t, f = hyper_as_trig(eq)
    >>> f(fu(t))
    cosh(2*x)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hyperbolic_function
    """
    from sympy.simplify.simplify import signsimp
    from sympy.simplify.radsimp import collect

    # 从输入表达式中找到所有三角函数
    trigs = rv.atoms(TrigonometricFunction)
    # 使用 Dummy 符号来替换所有的三角函数
    reps = [(t, Dummy()) for t in trigs]
    masked = rv.xreplace(dict(reps))  # 用 Dummy 符号替换三角函数

    # 准备反向替换，以便将 Dummy 符号转换回原始的三角函数
    reps = [(v, k) for k, v in reps]

    d = Dummy()

    # 返回转换后的表达式及反向转换函数
    return _osborne(masked, d), lambda x: collect(signsimp(
        _osbornei(x, d).xreplace(dict(reps))), S.ImaginaryUnit)


# 将表达式中的 sin 和 cos 的乘积及幂次转换为和的形式
def sincos_to_sum(expr):
    """Convert products and powers of sin and cos to sums.

    Explanation
    ===========

    Applied power reduction TRpower first, then expands products, and
    converts products to sums with TR8.

    Examples
    ========

    >>> from sympy.simplify.fu import sincos_to_sum
    >>> from sympy.abc import x
    >>> from sympy import cos, sin
    >>> sincos_to_sum(16*sin(x)**3*cos(2*x)**2)
    7*sin(x) - 5*sin(3*x) + 3*sin(5*x) - sin(7*x)
    """

    # 如果表达式中没有 sin 或 cos 函数，直接返回原表达式
    if not expr.has(cos, sin):
        return expr
    else:
        # 先应用幂次降低函数 TRpower，然后展开乘积，并用 TR8 转换乘积为和
        return TR8(expand_mul(TRpower(expr)))
```