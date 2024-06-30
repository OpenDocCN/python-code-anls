# `D:\src\scipysrc\sympy\sympy\core\add.py`

```
# 引入类型别名 Tuple 作为 tTuple
from typing import Tuple as tTuple
# 引入 defaultdict，一个默认值为列表的字典
from collections import defaultdict
# 引入 reduce 函数，用于对序列进行累积操作
from functools import reduce
# 引入 attrgetter 函数，用于获取对象的属性值
from operator import attrgetter
# 从当前包中导入 _args_sortkey 函数
from .basic import _args_sortkey
# 从当前包中导入 global_parameters，全局参数对象
from .parameters import global_parameters
# 从当前包中导入 _fuzzy_group, fuzzy_or, fuzzy_not 函数
from .logic import _fuzzy_group, fuzzy_or, fuzzy_not
# 从当前包中导入 S，单例模式
from .singleton import S
# 从当前包中导入 AssocOp, AssocOpDispatcher 类
from .operations import AssocOp, AssocOpDispatcher
# 从当前包中导入 cacheit 装饰器
from .cache import cacheit
# 从当前包中导入 equal_valued 函数
from .numbers import equal_valued
# 从当前包中导入 ilcm, igcd 函数
from .intfunc import ilcm, igcd
# 从当前包中导入 Expr 类
from .expr import Expr
# 从当前包中导入 UndefinedKind 类
from .kind import UndefinedKind
# 从 sympy.utilities.iterables 中导入 is_sequence, sift 函数
from sympy.utilities.iterables import is_sequence, sift


def _could_extract_minus_sign(expr):
    # 假设 expr 是类似于 Add 的对象
    # 统计 expr.args 中可以提取负号的参数数量
    negative_args = sum(1 for i in expr.args
        if i.could_extract_minus_sign())
    # 计算 expr.args 中非负号参数的数量
    positive_args = len(expr.args) - negative_args
    if positive_args > negative_args:
        return False
    elif positive_args < negative_args:
        return True
    # 如果负号参数数量等于非负号参数数量，则通过 sort_key() 方法选择较优的顺序
    return bool(expr.sort_key() < (-expr).sort_key())


def _addsort(args):
    # 对 args 进行原地排序，使用 _args_sortkey 函数作为排序键
    args.sort(key=_args_sortkey)


def _unevaluated_Add(*args):
    """返回一个良好形式的未评估 Add 对象：将数字收集并放在 slot 0 中，并对参数进行排序。当 args 发生变化但仍要返回未评估的 Add 时使用。

    Examples
    ========

    >>> from sympy.core.add import _unevaluated_Add as uAdd
    >>> from sympy import S, Add
    >>> from sympy.abc import x, y
    >>> a = uAdd(*[S(1.0), x, S(2)])
    >>> a.args[0]
    3.00000000000000
    >>> a.args[1]
    x

    除了数字在 slot 0 中，没有其他参数顺序的保证，因为它们是通过哈希排序的。因此，用于测试目的的输出只能与此函数的输出或作为多个选项之一进行测试：

    >>> opts = (Add(x, y, evaluate=False), Add(y, x, evaluate=False))
    >>> a = uAdd(x, y)
    >>> assert a in opts and a == uAdd(x, y)
    >>> uAdd(x + 1, x + 2)
    x + x + 3
    """
    args = list(args)
    newargs = []
    co = S.Zero
    while args:
        a = args.pop()
        if a.is_Add:
            # 这样可以避免嵌套增加，从而确保 x + (x + 1) -> x + x + 1 (3 个参数)
            args.extend(a.args)
        elif a.is_Number:
            co += a
        else:
            newargs.append(a)
    _addsort(newargs)
    if co:
        newargs.insert(0, co)
    # 使用 Add._from_args() 方法创建 Add 对象，返回未评估的 Add 对象
    return Add._from_args(newargs)


class Add(Expr, AssocOp):
    """
    表示代数群的加法运算的表达式。

    .. deprecated:: 1.7

       在核心运算符（:class:`~.Mul`, :class:`~.Add`, :class:`~.Pow`）中使用非 :class:`~.Expr` 的参数已弃用。详细信息请参见 :ref:`non-expr-args-deprecated`。

    """
    pass
    # 每个 ``Add()`` 的参数必须是 ``Expr`` 类型的表达式。SymPy 中的大多数标量对象在使用中缀运算符 ``+`` 时调用此类。
    # 
    # ``Add()`` 的另一种用法是表示抽象加法的结构，使得可以对其参数进行替换以返回不同的类。参考示例部分了解更多信息。
    # 
    # 当不传递 ``evaluate=False`` 参数时，``Add()`` 对其参数进行求值。
    # 求值逻辑包括：
    # 
    # 1. 展平
    #    ``Add(x, Add(y, z))`` -> ``Add(x, y, z)``
    # 
    # 2. 去除恒等元素
    #    ``Add(x, 0, y)`` -> ``Add(x, y)``
    # 
    # 3. 通过 ``.as_coeff_Mul()`` 收集系数
    #    ``Add(x, 2*x)`` -> ``Mul(3, x)``
    # 
    # 4. 项排序
    #    ``Add(y, x, 2)`` -> ``Add(2, x, y)``
    # 
    # 如果没有传递参数，则返回恒等元素 0。如果传递单个元素，则返回该元素。
    # 
    # 注意，``Add(*args)`` 比 ``sum(args)`` 更高效，因为它展平了参数。``sum(a, b, c, ...)`` 递归地将参数添加为 ``a + (b + (c + ...))``，具有二次复杂度。
    # 另一方面，``Add(a, b, c, d)`` 不假设嵌套结构，使得复杂度是线性的。
    # 
    # 由于加法是群操作，每个参数应该有相同的 :obj:`sympy.core.kind.Kind()`。
    # 
    # 示例
    # ========
    # 
    # >>> from sympy import Add, I
    # >>> from sympy.abc import x, y
    # >>> Add(x, 1)
    # x + 1
    # >>> Add(x, x)
    # 2*x
    # >>> 2*x**2 + 3*x + I*y + 2*y + 2*x/5 + 1.0*y + 1
    # 2*x**2 + 17*x/5 + 3.0*y + I*y + 1
    # 
    # 如果传递 ``evaluate=False``，则结果不会被求值。
    # 
    # >>> Add(1, 2, evaluate=False)
    # 1 + 2
    # >>> Add(x, x, evaluate=False)
    # x + x
    # 
    # ``Add()`` 也表示加法操作的一般结构。
    # 
    # >>> from sympy import MatrixSymbol
    # >>> A,B = MatrixSymbol('A', 2,2), MatrixSymbol('B', 2,2)
    # >>> expr = Add(x,y).subs({x:A, y:B})
    # >>> expr
    # A + B
    # >>> type(expr)
    # <class 'sympy.matrices.expressions.matadd.MatAdd'>
    # 
    # 注意打印器不按参数顺序显示。
    # 
    # >>> Add(x, 1)
    # x + 1
    # >>> Add(x, 1).args
    # (1, x)
    # 
    # 参见
    # ========
    # 
    # MatAdd
    # 返回一个元组 (coeff, args)，其中 self 被视为一个加法表达式，coeff 是数字项，args 是除数字项外的所有其他项组成的元组。

    def as_coeff_add(self, *deps):
        """
        Returns a tuple (coeff, args) where self is treated as an Add and coeff
        is the Number term and args is a tuple of all other terms.

        Examples
        ========

        >>> from sympy.abc import x
        >>> (7 + 3*x).as_coeff_add()
        (7, (3*x,))
        >>> (7*x).as_coeff_add()
        (0, (7*x,))
        """
        
        # 如果传入了 deps 参数
        if deps:
            # 使用 sift 函数将 self.args 拆分成两个列表 l1 和 l2，
            # l1 中的元素满足 x.has_free(*deps)，l2 中的元素是其余的元素
            l1, l2 = sift(self.args, lambda x: x.has_free(*deps), binary=True)
            # 返回经过 _new_rawargs 处理的 l2 元组和 l1 元组
            return self._new_rawargs(*l2), tuple(l1)
        
        # 否则从 self.args[0] 中提取 coeff 和 notrat
        coeff, notrat = self.args[0].as_coeff_add()
        
        # 如果 coeff 不是零，则返回 coeff 和 notrat 加上 self.args[1:] 组成的元组
        if coeff is not S.Zero:
            return coeff, notrat + self.args[1:]
        
        # 否则返回零和 self.args 组成的元组
        return S.Zero, self.args

    # 返回一个元组 (coeff, args)，其中 coeff 是一个加法表达式的系数。
    # 如果 rational=True 或者 coeff 是有理数，则返回 coeff 和除去 coeff 外的其他项组成的新表达式。
    # 否则返回零和 self 本身组成的元组。
    def as_coeff_Add(self, rational=False, deps=None):
        """
        Efficiently extract the coefficient of a summation.
        """
        
        # 将 self.args[0] 作为系数，将 self.args[1:] 作为其他项
        coeff, args = self.args[0], self.args[1:]

        # 如果 coeff 是数字，并且 rational=False 或者 coeff 是有理数
        if coeff.is_Number and not rational or coeff.is_Rational:
            # 返回 coeff 和经过 _new_rawargs 处理的 args 组成的新表达式
            return coeff, self._new_rawargs(*args)
        
        # 否则返回零和 self 组成的元组
        return S.Zero, self

    # 注意，我们故意没有实现 Add.as_coeff_mul() 方法。
    # 相反，我们让 Expr.as_coeff_mul() 方法对于 Add 类型的表达式始终返回 (S.One, self)。
    # 参见 issue 5524。
    # 定义一个方法来计算幂运算的表达式
    def _eval_power(self, e):
        # 导入需要的函数和类
        from .evalf import pure_complex
        from .relational import is_eq
        
        # 如果表达式中有两个参数且其中至少一个是无限的
        if len(self.args) == 2 and any(_.is_infinite for _ in self.args):
            # 如果 e 不等于零且不等于 1
            if e.is_zero is False and is_eq(e, S.One) is False:
                # 寻找形如 a + I*b 的文字 a 和 b
                a, b = self.args
                # 如果 a 中包含虚数单位 I
                if a.coeff(S.ImaginaryUnit):
                    # 交换 a 和 b 的位置
                    a, b = b, a
                # 提取 b 中的虚数单位系数
                ico = b.coeff(S.ImaginaryUnit)
                # 如果虚数单位系数存在且为扩展实数，同时 a 也是扩展实数
                if ico and ico.is_extended_real and a.is_extended_real:
                    # 如果 e 是扩展负数，则返回零
                    if e.is_extended_negative:
                        return S.Zero
                    # 如果 e 是扩展正数，则返回复数无穷大
                    if e.is_extended_positive:
                        return S.ComplexInfinity
            return
        
        # 如果 e 是有理数且 self 是数字
        if e.is_Rational and self.is_number:
            # 纯复数 ri 的判断
            ri = pure_complex(self)
            if ri:
                r, i = ri
                # 如果 e 的分母为 2
                if e.q == 2:
                    from sympy.functions.elementary.miscellaneous import sqrt
                    # 计算模长 D
                    D = sqrt(r**2 + i**2)
                    # 如果 D 是有理数
                    if D.is_Rational:
                        from .exprtools import factor_terms
                        from sympy.functions.elementary.complexes import sign
                        from .function import expand_multinomial
                        # (r, i, D) 是勾股数三元组
                        root = sqrt(factor_terms((D - r)/2))**e.p
                        return root * expand_multinomial((
                            # 主值
                            (D + r)/abs(i) + sign(i)*S.ImaginaryUnit)**e.p)
                # 如果 e 等于 -1
                elif e == -1:
                    return _unevaluated_Mul(
                        r - i*S.ImaginaryUnit,
                        1/(r**2 + i**2))
        
        # 如果 e 是数字且其绝对值不等于 1
        elif e.is_Number and abs(e) != 1:
            # 处理浮点数情况：(2.0 + 4*x)**e -> 4**e*(0.5 + x)**e
            c, m = zip(*[i.as_coeff_Mul() for i in self.args])
            # 如果其中任何一个参数是浮点数
            if any(i.is_Float for i in c):
                big = -1
                # 找出绝对值最大的浮点数
                for i in c:
                    if abs(i) >= big:
                        big = abs(i)
                # 如果最大的浮点数大于零且不等于 1
                if big > 0 and not equal_valued(big, 1):
                    from sympy.functions.elementary.complexes import sign
                    # 构造 big 的正负值
                    bigs = (big, -big)
                    c = [sign(i) if i in bigs else i/big for i in c]
                    # 计算 addpow 的幂
                    addpow = Add(*[c*m for c, m in zip(c, m)])**e
                    return big**e * addpow

    # 缓存装饰器，用于优化函数调用结果的性能
    @cacheit
    def _eval_derivative(self, s):
        # 返回对当前对象各参数求导后的新对象
        return self.func(*[a.diff(s) for a in self.args])

    # 对当前对象进行 n 级数展开
    def _eval_nseries(self, x, n, logx, cdir=0):
        # 对每个参数进行 n 级数展开，并返回新的对象
        terms = [t.nseries(x, n=n, logx=logx, cdir=cdir) for t in self.args]
        return self.func(*terms)

    # 对当前对象进行简单匹配
    def _matches_simple(self, expr, repl_dict):
        # 处理如 (w+3).matches('x+5') 的情况，返回匹配结果字典
        coeff, terms = self.as_coeff_add()
        if len(terms) == 1:
            return terms[0].matches(expr - coeff, repl_dict)
        return
    # 定义一个方法 `matches`，用于比较表达式 `expr` 是否与当前对象匹配
    def matches(self, expr, repl_dict=None, old=False):
        # 调用内部方法 `_matches_commutative` 进行比较，并返回结果
        return self._matches_commutative(expr, repl_dict, old)

    @staticmethod
    # 定义一个静态方法 `_combine_inverse`，用于计算 lhs - rhs
    def _combine_inverse(lhs, rhs):
        """
        Returns lhs - rhs, but treats oo like a symbol so oo - oo
        returns 0, instead of a nan.
        """
        # 导入符号化简模块中的 `signsimp` 函数
        from sympy.simplify.simplify import signsimp
        # 定义无穷大常量的元组
        inf = (S.Infinity, S.NegativeInfinity)
        # 如果 lhs 或 rhs 包含无穷大常量
        if lhs.has(*inf) or rhs.has(*inf):
            # 导入符号模块中的 `Dummy` 类
            from .symbol import Dummy
            # 创建虚拟符号 oo 作为无穷大的替代
            oo = Dummy('oo')
            # 定义替换字典和反向替换字典
            reps = {
                S.Infinity: oo,
                S.NegativeInfinity: -oo
            }
            ireps = {v: k for k, v in reps.items()}
            # 替换 lhs 和 rhs 中的无穷大常量
            eq = lhs.xreplace(reps) - rhs.xreplace(reps)
            # 如果结果包含 oo，则替换为其基础符号
            if eq.has(oo):
                eq = eq.replace(
                    lambda x: x.is_Pow and x.base is oo,
                    lambda x: x.base)
            # 反向替换回原始符号
            rv = eq.xreplace(ireps)
        else:
            # 否则直接计算 lhs - rhs
            rv = lhs - rhs
        # 对结果进行符号化简
        srv = signsimp(rv)
        # 如果简化后是一个数值，则返回简化结果，否则返回未简化的结果
        return srv if srv.is_Number else rv

    @cacheit
    # 定义一个装饰器方法 `as_two_terms`，用于获取表达式的头部和尾部
    def as_two_terms(self):
        """Return head and tail of self.

        This is the most efficient way to get the head and tail of an
        expression.

        - if you want only the head, use self.args[0];
        - if you want to process the arguments of the tail then use
          self.as_coef_add() which gives the head and a tuple containing
          the arguments of the tail when treated as an Add.
        - if you want the coefficient when self is treated as a Mul
          then use self.as_coeff_mul()[0]

        >>> from sympy.abc import x, y
        >>> (3*x - 2*y + 5).as_two_terms()
        (5, 3*x - 2*y)
        """
        # 返回表达式的第一个参数作为头部，以及剩余参数作为尾部
        return self.args[0], self._new_rawargs(*self.args[1:])
    # 利用符号表达式的原始方法，将表达式分解为分子和分母部分
    def as_numer_denom(self):
        """
        Decomposes an expression to its numerator part and its
        denominator part.

        Examples
        ========

        >>> from sympy.abc import x, y, z
        >>> (x*y/z).as_numer_denom()
        (x*y, z)
        >>> (x*(y + 1)/y**7).as_numer_denom()
        (x*(y + 1), y**7)

        See Also
        ========

        sympy.core.expr.Expr.as_numer_denom
        """
        # 将表达式转化为其基本部分
        content, expr = self.primitive()
        # 如果表达式不是加法，则返回重新构造的乘法对象的分子分母形式
        if not isinstance(expr, Add):
            return Mul(content, expr, evaluate=False).as_numer_denom()
        # 否则，获取内容的分子和分母
        ncon, dcon = content.as_numer_denom()

        # 收集各项的分子和分母
        nd = defaultdict(list)
        for f in expr.args:
            ni, di = f.as_numer_denom()
            nd[di].append(ni)

        # 如果只有一个分母，直接返回构造的新表达式
        if len(nd) == 1:
            d, n = nd.popitem()
            return self.func(
                *[_keep_coeff(ncon, ni) for ni in n]), _keep_coeff(dcon, d)

        # 合并具有相同分母的项
        for d, n in nd.items():
            if len(n) == 1:
                nd[d] = n[0]
            else:
                nd[d] = self.func(*n)

        # 组合成单一的分子和分母
        denoms, numers = [list(i) for i in zip(*iter(nd.items()))]
        n, d = self.func(*[Mul(*(denoms[:i] + [numers[i]] + denoms[i + 1:]))
                   for i in range(len(numers))]), Mul(*denoms)

        return _keep_coeff(ncon, n), _keep_coeff(dcon, d)

    # 判断表达式是否是多项式的内部方法
    def _eval_is_polynomial(self, syms):
        return all(term._eval_is_polynomial(syms) for term in self.args)

    # 判断表达式是否是有理函数的内部方法
    def _eval_is_rational_function(self, syms):
        return all(term._eval_is_rational_function(syms) for term in self.args)

    # 判断表达式是否是亚解析的内部方法
    def _eval_is_meromorphic(self, x, a):
        return _fuzzy_group((arg.is_meromorphic(x, a) for arg in self.args),
                            quick_exit=True)

    # 判断表达式是否是代数表达式的内部方法
    def _eval_is_algebraic_expr(self, syms):
        return all(term._eval_is_algebraic_expr(syms) for term in self.args)

    # 下面是关于表达式属性的假设方法

    # 判断表达式是否是实数的快速聚合方法
    _eval_is_real = lambda self: _fuzzy_group(
        (a.is_real for a in self.args), quick_exit=True)
    # 判断表达式是否是扩展实数的快速聚合方法
    _eval_is_extended_real = lambda self: _fuzzy_group(
        (a.is_extended_real for a in self.args), quick_exit=True)
    # 判断表达式是否是复数的快速聚合方法
    _eval_is_complex = lambda self: _fuzzy_group(
        (a.is_complex for a in self.args), quick_exit=True)
    # 判断表达式是否是反厄米的快速聚合方法
    _eval_is_antihermitian = lambda self: _fuzzy_group(
        (a.is_antihermitian for a in self.args), quick_exit=True)
    # 判断表达式是否是有限的快速聚合方法
    _eval_is_finite = lambda self: _fuzzy_group(
        (a.is_finite for a in self.args), quick_exit=True)
    # 判断表达式是否是厄米的快速聚合方法
    _eval_is_hermitian = lambda self: _fuzzy_group(
        (a.is_hermitian for a in self.args), quick_exit=True)
    # 判断表达式是否是整数的快速聚合方法
    _eval_is_integer = lambda self: _fuzzy_group(
        (a.is_integer for a in self.args), quick_exit=True)
    # 使用一个 lambda 函数来评估是否所有参数都是有理数，并进行模糊分组
    _eval_is_rational = lambda self: _fuzzy_group(
        (a.is_rational for a in self.args), quick_exit=True)

    # 使用一个 lambda 函数来评估是否所有参数都是代数的，并进行模糊分组
    _eval_is_algebraic = lambda self: _fuzzy_group(
        (a.is_algebraic for a in self.args), quick_exit=True)

    # 使用一个 lambda 函数来评估是否所有参数都是可交换的
    _eval_is_commutative = lambda self: _fuzzy_group(
        a.is_commutative for a in self.args)

    # 评估对象是否为无穷的，遍历所有参数并检查
    def _eval_is_infinite(self):
        sawinf = False
        for a in self.args:
            ainf = a.is_infinite
            if ainf is None:
                return None
            elif ainf is True:
                # 如果之前已经发现了无穷，再次发现无穷将返回不确定
                if sawinf is True:
                    return None
                sawinf = True
        return sawinf

    # 评估对象是否为虚数，根据参数类型进行分析
    def _eval_is_imaginary(self):
        nz = []  # 存储非零的实数部分
        im_I = []  # 存储虚数部分
        for a in self.args:
            if a.is_extended_real:
                if a.is_zero:
                    pass
                elif a.is_zero is False:
                    nz.append(a)
                else:
                    return
            elif a.is_imaginary:
                im_I.append(a*S.ImaginaryUnit)  # 将虚数乘以虚数单位进行标记
            elif a.is_Mul and S.ImaginaryUnit in a.args:
                coeff, ai = a.as_coeff_mul(S.ImaginaryUnit)
                if ai == (S.ImaginaryUnit,) and coeff.is_extended_real:
                    im_I.append(-coeff)  # 虚数单位的系数取反加入虚数部分
                else:
                    return
            else:
                return
        b = self.func(*nz)  # 构造只包含非零实数部分的对象
        if b != self:
            if b.is_zero:
                return fuzzy_not(self.func(*im_I).is_zero)  # 如果非零实数部分为零，则判断虚数部分是否全为零
            elif b.is_zero is False:
                return False

    # 评估对象是否为零，根据参数类型进行详细分析
    def _eval_is_zero(self):
        if self.is_commutative is False:
            # 如果对象不是可交换的，则无法确定是否为零
            # issue 10528: 无法确定非可交换符号是否为零
            return
        nz = []  # 存储非零的实数部分
        z = 0  # 记录零的数量
        im_or_z = False  # 是否包含虚数或零
        im = 0  # 记录虚数的数量
        for a in self.args:
            if a.is_extended_real:
                if a.is_zero:
                    z += 1
                elif a.is_zero is False:
                    nz.append(a)
                else:
                    return
            elif a.is_imaginary:
                im += 1
            elif a.is_Mul and S.ImaginaryUnit in a.args:
                coeff, ai = a.as_coeff_mul(S.ImaginaryUnit)
                if ai == (S.ImaginaryUnit,) and coeff.is_extended_real:
                    im_or_z = True  # 存在虚数单位乘以实数系数的情况
                else:
                    return
            else:
                return
        if z == len(self.args):
            return True  # 所有参数都是零
        if len(nz) in [0, len(self.args)]:
            return None  # 所有参数要么都是非零实数，要么都是零
        b = self.func(*nz)  # 构造只包含非零实数部分的对象
        if b.is_zero:
            if not im_or_z:
                if im == 0:
                    return True  # 没有虚数部分，且非零实数部分为零
                elif im == 1:
                    return False  # 只有一个虚数部分
        if b.is_zero is False:
            return False  # 非零实数部分不为零
    # 检查表达式中是否包含奇数项，返回 True 或 None
    def _eval_is_odd(self):
        # 从 self.args 中筛选出非偶数项组成列表 l
        l = [f for f in self.args if not (f.is_even is True)]
        # 如果 l 为空，则返回 False
        if not l:
            return False
        # 如果 l 的第一项为奇数，则递归调用 _new_rawargs 方法并检查剩余项是否为偶数
        if l[0].is_odd:
            return self._new_rawargs(*l[1:]).is_even

    # 检查表达式是否为无理数，返回 True、None 或 False
    def _eval_is_irrational(self):
        # 遍历 self.args 中的每个表达式 t
        for t in self.args:
            # 检查 t 是否为无理数
            a = t.is_irrational
            if a:
                # 创建 self.args 的副本，并移除当前项 t
                others = list(self.args)
                others.remove(t)
                # 如果剩余项都是有理数，则返回 True；否则返回 None
                if all(x.is_rational is True for x in others):
                    return True
                return None
            # 如果 a 为 None，则返回
            if a is None:
                return
        # 如果所有项都不是无理数，则返回 False
        return False

    # 检查表达式中的所有项是否全为非负数或全为非正数，返回 True 或 False
    def _all_nonneg_or_nonppos(self):
        nn = np = 0
        # 遍历 self.args 中的每个表达式 a
        for a in self.args:
            if a.is_nonnegative:
                # 如果 a 是非负数，并且之前已经出现过非正数，则返回 False
                if np:
                    return False
                nn = 1
            elif a.is_nonpositive:
                # 如果 a 是非正数，并且之前已经出现过非负数，则返回 False
                if nn:
                    return False
                np = 1
            else:
                break
        else:
            # 如果所有项都是非负数或非正数，则返回 True
            return True

    # 检查表达式是否为扩展正数，返回 True、None 或 False
    def _eval_is_extended_positive(self):
        if self.is_number:
            # 如果 self 是一个数字，则调用父类的 _eval_is_extended_positive 方法
            return super()._eval_is_extended_positive()
        # 将 self 表达式表示为系数 c 和余项 a
        c, a = self.as_coeff_Add()
        # 如果系数 c 不为零
        if not c.is_zero:
            # 导入 _monotonic_sign 函数
            from .exprtools import _monotonic_sign
            # 计算余项 a 的单调符号 v
            v = _monotonic_sign(a)
            # 如果 v 不为 None
            if v is not None:
                # 计算 s = v + c
                s = v + c
                # 如果 s 不等于 self，并且 s 和 a 满足扩展正数的条件，则返回 True
                if s != self and s.is_extended_positive and a.is_extended_nonnegative:
                    return True
                # 如果 self 的自由符号只有一个，则再次计算其单调符号 v
                if len(self.free_symbols) == 1:
                    v = _monotonic_sign(self)
                    # 如果 v 不等于 self，并且 v 是扩展正数，则返回 True
                    if v is not None and v != self and v.is_extended_positive:
                        return True
        # 初始化变量 pos、nonneg、nonpos 和 unknown_sign 为 False
        pos = nonneg = nonpos = unknown_sign = False
        # 初始化 saw_INF 为空集合
        saw_INF = set()
        # 筛选出不为零的所有项组成 args 列表
        args = [a for a in self.args if not a.is_zero]
        # 如果 args 为空，则返回 False
        if not args:
            return False
        # 遍历 args 中的每个表达式 a
        for a in args:
            # 判断 a 是否为扩展正数和无穷大
            ispos = a.is_extended_positive
            infinite = a.is_infinite
            # 如果 a 是无穷大
            if infinite:
                # 将模糊逻辑或的结果添加到 saw_INF 集合中
                saw_INF.add(fuzzy_or((ispos, a.is_extended_nonnegative)))
                # 如果集合 saw_INF 中既有 True 又有 False，则返回
                if True in saw_INF and False in saw_INF:
                    return
            # 如果 a 是扩展正数
            if ispos:
                pos = True
                continue
            # 如果 a 是非负数
            elif a.is_extended_nonnegative:
                nonneg = True
                continue
            # 如果 a 是非正数
            elif a.is_extended_nonpositive:
                nonpos = True
                continue

            # 如果 a 既不是无穷大也不是无穷小，则返回
            if infinite is None:
                return
            unknown_sign = True

        # 如果 saw_INF 集合不为空
        if saw_INF:
            # 如果 saw_INF 集合中的元素个数大于 1，则返回
            if len(saw_INF) > 1:
                return
            # 返回 saw_INF 集合中的唯一元素
            return saw_INF.pop()
        # 如果存在未知符号，则返回
        elif unknown_sign:
            return
        # 如果不存在非正数和非负数项，并且存在正数项，则返回 True
        elif not nonpos and not nonneg and pos:
            return True
        # 如果不存在非正数项，并且存在正数项，则返回 True
        elif not nonpos and pos:
            return True
        # 如果不存在正数项，并且不存在非负数项，则返回 False
        elif not pos and not nonneg:
            return False
    # 检查当前表达式是否是一个数值，如果不是，则进行下面的计算
    def _eval_is_extended_nonnegative(self):
        # 将表达式分解为常数部分 c 和余项 a
        if not self.is_number:
            c, a = self.as_coeff_Add()
            # 如果常数部分 c 不为零且余项 a 是扩展非负的
            if not c.is_zero and a.is_extended_nonnegative:
                # 导入表达式工具模块中的 _monotonic_sign 函数
                from .exprtools import _monotonic_sign
                # 计算余项 a 的单调符号
                v = _monotonic_sign(a)
                # 如果计算结果不为 None
                if v is not None:
                    # 计算新的表达式 s = v + c
                    s = v + c
                    # 如果新表达式 s 不等于原始表达式且是扩展非负的
                    if s != self and s.is_extended_nonnegative:
                        # 返回 True
                        return True
                    # 如果自由符号数量为 1
                    if len(self.free_symbols) == 1:
                        # 计算原始表达式的单调符号
                        v = _monotonic_sign(self)
                        # 如果计算结果不为 None 且新表达式是扩展非负的
                        if v is not None and v != self and v.is_extended_nonnegative:
                            # 返回 True
                            return True

    # 检查当前表达式是否是一个数值，如果不是，则进行下面的计算
    def _eval_is_extended_nonpositive(self):
        # 将表达式分解为常数部分 c 和余项 a
        if not self.is_number:
            c, a = self.as_coeff_Add()
            # 如果常数部分 c 不为零且余项 a 是扩展非正的
            if not c.is_zero and a.is_extended_nonpositive:
                # 导入表达式工具模块中的 _monotonic_sign 函数
                from .exprtools import _monotonic_sign
                # 计算余项 a 的单调符号
                v = _monotonic_sign(a)
                # 如果计算结果不为 None
                if v is not None:
                    # 计算新的表达式 s = v + c
                    s = v + c
                    # 如果新表达式 s 不等于原始表达式且是扩展非正的
                    if s != self and s.is_extended_nonpositive:
                        # 返回 True
                        return True
                    # 如果自由符号数量为 1
                    if len(self.free_symbols) == 1:
                        # 计算原始表达式的单调符号
                        v = _monotonic_sign(self)
                        # 如果计算结果不为 None 且新表达式是扩展非正的
                        if v is not None and v != self and v.is_extended_nonpositive:
                            # 返回 True
                            return True
    # 判断是否为扩展负数表达式的评估函数，继承自基类
    def _eval_is_extended_negative(self):
        # 如果表达式是一个数字，调用父类的扩展负数判断函数
        if self.is_number:
            return super()._eval_is_extended_negative()
        
        # 将表达式分解为系数和加法项
        c, a = self.as_coeff_Add()
        
        # 如果系数不为零，导入表达式工具模块中的 _monotonic_sign 函数
        if not c.is_zero:
            from .exprtools import _monotonic_sign
            # 对加法项应用 _monotonic_sign 函数
            v = _monotonic_sign(a)
            # 如果返回值不为 None
            if v is not None:
                # 计算带系数的表达式
                s = v + c
                # 如果结果与原表达式不同且结果为扩展负数且加法项为扩展非正数，则返回 True
                if s != self and s.is_extended_negative and a.is_extended_nonpositive:
                    return True
                # 如果自由符号个数为1
                if len(self.free_symbols) == 1:
                    # 对表达式应用 _monotonic_sign 函数
                    v = _monotonic_sign(self)
                    # 如果返回值不为 None 且不等于原表达式且结果为扩展负数，则返回 True
                    if v is not None and v != self and v.is_extended_negative:
                        return True
        
        # 初始化标志变量
        neg = nonpos = nonneg = unknown_sign = False
        # 记录出现的无限值情况
        saw_INF = set()
        # 过滤掉加法项中的零
        args = [a for a in self.args if not a.is_zero]
        # 如果过滤后没有有效加法项，则返回 False
        if not args:
            return False
        
        # 遍历加法项
        for a in args:
            # 判断加法项是否为扩展负数
            isneg = a.is_extended_negative
            # 判断加法项是否为无限值
            infinite = a.is_infinite
            # 如果是无限值
            if infinite:
                # 将看到的无限值情况添加到集合中
                saw_INF.add(fuzzy_or((isneg, a.is_extended_nonpositive)))
                # 如果集合中同时包含 True 和 False，则返回
                if True in saw_INF and False in saw_INF:
                    return
                
            # 如果加法项为扩展负数
            if isneg:
                neg = True
                continue
            # 如果加法项为扩展非正数
            elif a.is_extended_nonpositive:
                nonpos = True
                continue
            # 如果加法项为扩展非负数
            elif a.is_extended_nonnegative:
                nonneg = True
                continue
            
            # 如果加法项既非负也非正
            if infinite is None:
                return
            
            # 如果无法确定符号
            unknown_sign = True
        
        # 如果看到无限值情况
        if saw_INF:
            # 如果集合中元素多于1个，则返回
            if len(saw_INF) > 1:
                return
            # 返回集合中的元素
            return saw_INF.pop()
        # 如果无法确定符号
        elif unknown_sign:
            return
        # 如果既非非负又非非正且存在负数加法项
        elif not nonneg and not nonpos and neg:
            return True
        # 如果既非非负且存在负数加法项
        elif not nonneg and neg:
            return True
        # 如果不存在负数且既非非正
        elif not neg and not nonpos:
            return False
    # 定义一个方法 `_eval_subs`，用于处理符号表达式中的替换操作
    def _eval_subs(self, old, new):
        # 如果旧表达式不是加法类型，特别处理无穷大的情况
        if not old.is_Add:
            if old is S.Infinity and -old in self.args:
                # 处理无穷大的替换为相反数的情况，如 foo - oo 被内部处理为 foo + (-oo)
                return self.xreplace({-old: -new})
            return None

        # 将自身表达式分解为系数和项
        coeff_self, terms_self = self.as_coeff_Add()
        # 将旧表达式分解为系数和项
        coeff_old, terms_old = old.as_coeff_Add()

        # 如果自身和旧表达式的系数都是有理数
        if coeff_self.is_Rational and coeff_old.is_Rational:
            if terms_self == terms_old:   # (2 + a).subs( 3 + a, y) -> -1 + y
                # 返回一个新的表达式，用新值替换旧值的结果
                return self.func(new, coeff_self, -coeff_old)
            if terms_self == -terms_old:  # (2 + a).subs(-3 - a, y) -> -1 - y
                # 返回一个新的表达式，用新值替换旧值的结果
                return self.func(-new, coeff_self, coeff_old)

        # 如果自身和旧表达式的系数相等或者都是同一表达式
        if coeff_self.is_Rational and coeff_old.is_Rational \
                or coeff_self == coeff_old:
            # 将项转换为列表
            args_old, args_self = self.func.make_args(
                terms_old), self.func.make_args(terms_self)
            # 如果旧表达式的项比自身的项少，执行替换操作
            if len(args_old) < len(args_self):  # (a+b+c).subs(b+c,x) -> a+x
                self_set = set(args_self)
                old_set = set(args_old)

                # 如果旧表达式的项都在自身表达式中，执行替换操作
                if old_set < self_set:
                    ret_set = self_set - old_set
                    # 返回一个新的表达式，用新值替换旧值的结果
                    return self.func(new, coeff_self, -coeff_old,
                               *[s._subs(old, new) for s in ret_set])

                # 否则将旧表达式的项取反，再执行替换操作
                args_old = self.func.make_args(
                    -terms_old)     # (a+b+c+d).subs(-b-c,x) -> a-x+d
                old_set = set(args_old)
                if old_set < self_set:
                    ret_set = self_set - old_set
                    # 返回一个新的表达式，用新值替换旧值的结果
                    return self.func(-new, coeff_self, coeff_old,
                               *[s._subs(old, new) for s in ret_set])

    # 定义一个方法 `removeO`，用于从表达式中移除所有的阶符号
    def removeO(self):
        # 从参数列表中筛选出不是阶符号的参数
        args = [a for a in self.args if not a.is_Order]
        # 返回一个新的表达式，只包含非阶符号的参数
        return self._new_rawargs(*args)

    # 定义一个方法 `getO`，用于获取表达式中的所有阶符号
    def getO(self):
        # 从参数列表中筛选出是阶符号的参数
        args = [a for a in self.args if a.is_Order]
        # 如果有阶符号存在，返回一个新的表达式，只包含阶符号参数
        if args:
            return self._new_rawargs(*args)

    # 使用装饰器 `cacheit` 对下一个方法进行缓存处理
    @cacheit
    def extract_leading_order(self, symbols, point=None):
        """
        返回多项式的主导项及其阶数。

        Examples
        ========

        >>> from sympy.abc import x
        >>> (x + 1 + 1/x**5).extract_leading_order(x)
        ((x**(-5), O(x**(-5))),)
        >>> (1 + x).extract_leading_order(x)
        ((1, O(1)),)
        >>> (x + x**2).extract_leading_order(x)
        ((x, O(x)),)

        """
        # 导入 Order 类
        from sympy.series.order import Order
        lst = []
        # 将 symbols 转换为列表，如果不是序列则转换成单元素列表
        symbols = list(symbols if is_sequence(symbols) else [symbols])
        # 如果 point 为 None，则设定为与 symbols 长度相等的全零列表
        if not point:
            point = [0]*len(symbols)
        # 创建包含每个 self.args 元素及其对应 Order 对象的序列
        seq = [(f, Order(f, *zip(symbols, point))) for f in self.args]
        # 遍历 seq 中的每个 (f, Order(f, ...)) 对
        for ef, of in seq:
            # 遍历 lst 中的每个 (e, o) 对
            for e, o in lst:
                # 如果当前的 o 包含 of 且 o 不等于 of，则将 of 设为 None 并终止循环
                if o.contains(of) and o != of:
                    of = None
                    break
            # 如果 of 为 None，则跳过当前循环
            if of is None:
                continue
            # 将当前 (ef, of) 对加入到新的 lst 中
            new_lst = [(ef, of)]
            # 再次遍历 lst 中的每个 (e, o) 对
            for e, o in lst:
                # 如果 of 包含 o 且 o 不等于 of，则继续下一次循环
                if of.contains(o) and o != of:
                    continue
                # 将当前 (e, o) 对加入到新的 lst 中
                new_lst.append((e, o))
            # 更新 lst 为新的 lst
            lst = new_lst
        # 返回 lst 转换为元组后的结果
        return tuple(lst)

    def as_real_imag(self, deep=True, **hints):
        """
        返回表示复数的元组。

        Examples
        ========

        >>> from sympy import I
        >>> (7 + 9*I).as_real_imag()
        (7, 9)
        >>> ((1 + I)/(1 - I)).as_real_imag()
        (0, 1)
        >>> ((1 + 2*I)*(1 + 3*I)).as_real_imag()
        (-5, 5)
        """
        # 获取 self.args
        sargs = self.args
        re_part, im_part = [], []
        # 遍历 sargs 中的每个 term
        for term in sargs:
            # 调用 term 的 as_real_imag 方法，获取其实部和虚部
            re, im = term.as_real_imag(deep=deep)
            # 将实部添加到 re_part 列表中，将虚部添加到 im_part 列表中
            re_part.append(re)
            im_part.append(im)
        # 构造并返回包含 re_part 和 im_part 的元组
        return (self.func(*re_part), self.func(*im_part))
    # 定义一个方法来计算作为主导项的表达式，这里的参数包括 x、logx 和 cdir
    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        # 导入必要的符号和函数
        from sympy.core.symbol import Dummy, Symbol
        from sympy.series.order import Order
        from sympy.functions.elementary.exponential import log
        from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
        from .function import expand_mul

        # 获取当前对象的高阶项
        o = self.getO()
        if o is None:
            o = Order(0)
        # 移除高阶项
        old = self.removeO()

        # 如果旧表达式中包含 Piecewise 函数，则将其折叠为简单形式
        if old.has(Piecewise):
            old = piecewise_fold(old)

        # 如果当前对象中包含对数函数，按照特定标志进行展开
        if any(isinstance(a, log) for a in self.args):
            logflags = {"deep": True, "log": True, "mul": False, "power_exp": False,
                "power_base": False, "multinomial": False, "basic": False, "force": False,
                "factor": False}
            old = old.expand(**logflags)
        
        # 对旧表达式进行乘法展开
        expr = expand_mul(old)

        # 如果表达式不是加法形式，则直接返回其作为主导项的结果
        if not expr.is_Add:
            return expr.as_leading_term(x, logx=logx, cdir=cdir)

        # 找出表达式中的无限项
        infinite = [t for t in expr.args if t.is_infinite]

        # 如果 logx 为 None，则创建一个新的符号 logx
        _logx = Dummy('logx') if logx is None else logx
        # 计算每个子项的主导项
        leading_terms = [t.as_leading_term(x, logx=_logx, cdir=cdir) for t in expr.args]

        # 初始化最小阶数和新表达式
        min, new_expr = Order(0), 0

        # 尝试计算主导项
        try:
            for term in leading_terms:
                order = Order(term, x)
                if not min or order not in min:
                    min = order
                    new_expr = term
                elif min in order:
                    new_expr += term

        # 如果出现类型错误，则返回原始表达式
        except TypeError:
            return expr

        # 如果 logx 为 None，则用 log(x) 替换 _logx
        if logx is None:
            new_expr = new_expr.subs(_logx, log(x))

        # 检查新表达式是否为零
        is_zero = new_expr.is_zero
        if is_zero is None:
            # 简化新表达式并取消常见项
            new_expr = new_expr.trigsimp().cancel()
            is_zero = new_expr.is_zero
        # 如果新表达式为零，则通过级数计算主导项
        if is_zero is True:
            try:
                # 尝试获取最小阶数
                n0 = min.getn()
            except NotImplementedError:
                n0 = S.One
            if n0.has(Symbol):
                n0 = S.One
            res = Order(1)
            incr = S.One
            # 逐步增加级数展开的阶数，直到不再是阶数形式
            while res.is_Order:
                res = old._eval_nseries(x, n=n0+incr, logx=logx, cdir=cdir).cancel().powsimp().trigsimp()
                incr *= 2
            return res.as_leading_term(x, logx=logx, cdir=cdir)

        # 如果新表达式为 NaN，则返回无限项和高阶项的和
        elif new_expr is S.NaN:
            return old.func._from_args(infinite) + o

        # 否则返回计算得到的新表达式
        else:
            return new_expr

    # 返回对象的共轭
    def _eval_adjoint(self):
        return self.func(*[t.adjoint() for t in self.args])

    # 返回对象的共轭
    def _eval_conjugate(self):
        return self.func(*[t.conjugate() for t in self.args])

    # 返回对象的转置
    def _eval_transpose(self):
        return self.func(*[t.transpose() for t in self.args])
    def primitive(self):
        """
        Return ``(R, self/R)`` where ``R``` is the Rational GCD of ``self```.

        ``R`` is collected only from the leading coefficient of each term.

        Examples
        ========

        >>> from sympy.abc import x, y

        >>> (2*x + 4*y).primitive()
        (2, x + 2*y)

        >>> (2*x/3 + 4*y/9).primitive()
        (2/9, 3*x + 2*y)

        >>> (2*x/3 + 4.2*y).primitive()
        (1/3, 2*x + 12.6*y)

        No subprocessing of term factors is performed:

        >>> ((2 + 2*x)*x + 2).primitive()
        (1, x*(2*x + 2) + 2)

        Recursive processing can be done with the ``as_content_primitive()``
        method:

        >>> ((2 + 2*x)*x + 2).as_content_primitive()
        (2, x*(x + 1) + 1)

        See also: primitive() function in polytools.py

        """

        # Initialize an empty list to store processed terms
        terms = []
        # Flag to check if ComplexInfinity is encountered
        inf = False

        # Iterate through each term in self.args
        for a in self.args:
            # Separate the term into coefficient and multiplicand
            c, m = a.as_coeff_Mul()
            # If coefficient c is not Rational, reset it to 1
            if not c.is_Rational:
                c = S.One
                m = a
            # Check if term contains ComplexInfinity
            inf = inf or m is S.ComplexInfinity
            # Append the tuple (numerator of c, denominator of c, multiplicand m) to terms
            terms.append((c.p, c.q, m))

        # Calculate the GCD of all numerators and the LCM of all denominators in terms
        if not inf:
            ngcd = reduce(igcd, [t[0] for t in terms], 0)
            dlcm = reduce(ilcm, [t[1] for t in terms], 1)
        else:
            ngcd = reduce(igcd, [t[0] for t in terms if t[1]], 0)
            dlcm = reduce(ilcm, [t[1] for t in terms if t[1]], 1)

        # If both ngcd and dlcm are 1, return 1 and self
        if ngcd == dlcm == 1:
            return S.One, self

        # Adjust coefficients of terms based on ngcd and dlcm
        if not inf:
            for i, (p, q, term) in enumerate(terms):
                terms[i] = _keep_coeff(Rational((p//ngcd)*(dlcm//q)), term)
        else:
            for i, (p, q, term) in enumerate(terms):
                if q:
                    terms[i] = _keep_coeff(Rational((p//ngcd)*(dlcm//q)), term)
                else:
                    terms[i] = _keep_coeff(Rational(p, q), term)

        # Sort terms based on the leading coefficient
        if terms[0].is_Number or terms[0] is S.ComplexInfinity:
            c = terms.pop(0)
        else:
            c = None
        _addsort(terms)
        if c:
            terms.insert(0, c)

        # Return the Rational ngcd/dlcm and reconstruct the expression with adjusted terms
        return Rational(ngcd, dlcm), self._new_rawargs(*terms)
    # 返回包含两个元素的元组 (R, self/R)，其中 R 是从 self 中提取的正有理数。
    # 如果 radical 为 True（默认为 False），则将常见根号作为原始表达式的因子之一移除。
    def as_content_primitive(self, radical=False, clear=True):
        """Return the tuple (R, self/R) where R is the positive Rational
        extracted from self. If radical is True (default is False) then
        common radicals will be removed and included as a factor of the
        primitive expression.

        Examples
        ========

        >>> from sympy import sqrt
        >>> (3 + 3*sqrt(2)).as_content_primitive()
        (3, 1 + sqrt(2))

        Radical content can also be factored out of the primitive:

        >>> (2*sqrt(2) + 4*sqrt(10)).as_content_primitive(radical=True)
        (2, sqrt(2)*(1 + 2*sqrt(5)))

        See docstring of Expr.as_content_primitive for more examples.
        """
        # 对 self 中的每个参数进行 as_content_primitive 操作，并保持系数
        con, prim = self.func(*[_keep_coeff(*a.as_content_primitive(
            radical=radical, clear=clear)) for a in self.args]).primitive()
        
        # 如果 clear 为 False 且 con 不是整数且 prim 是加法表达式
        if not clear and not con.is_Integer and prim.is_Add:
            con, d = con.as_numer_denom()
            _p = prim/d
            # 如果 _p 中有任何一个参数的系数是整数，则更新 prim
            if any(a.as_coeff_Mul()[0].is_Integer for a in _p.args):
                prim = _p
            else:
                con /= d
        
        # 如果 radical 为 True 并且 prim 是加法表达式
        if radical and prim.is_Add:
            # 查找可以移除的常见根号
            args = prim.args
            rads = []
            common_q = None
            for m in args:
                term_rads = defaultdict(list)
                for ai in Mul.make_args(m):
                    if ai.is_Pow:
                        b, e = ai.as_base_exp()
                        # 如果指数是有理数且底数是整数，则将其加入 term_rads
                        if e.is_Rational and b.is_Integer:
                            term_rads[e.q].append(abs(int(b))**e.p)
                if not term_rads:
                    break
                if common_q is None:
                    common_q = set(term_rads.keys())
                else:
                    common_q = common_q & set(term_rads.keys())
                    if not common_q:
                        break
                rads.append(term_rads)
            else:
                # 处理根号
                # 只保留 common_q 中存在的根号
                for r in rads:
                    for q in list(r.keys()):
                        if q not in common_q:
                            r.pop(q)
                    for q in r:
                        r[q] = Mul(*r[q])
                # 找到每个 q 的底数的最大公因数（gcd）
                G = []
                for q in common_q:
                    g = reduce(igcd, [r[q] for r in rads], 0)
                    if g != 1:
                        G.append(g**Rational(1, q))
                if G:
                    G = Mul(*G)
                    args = [ai/G for ai in args]
                    prim = G*prim.func(*args)

        return con, prim

    @property
    def _sorted_args(self):
        # 导入默认排序键并返回参数按此键排序后的元组
        from .sorting import default_sort_key
        return tuple(sorted(self.args, key=default_sort_key))
    # 导入 difference_delta 函数来计算差分 delta
    def _eval_difference_delta(self, n, step):
        from sympy.series.limitseq import difference_delta as dd
        # 使用 difference_delta 函数计算每个参数的差分 delta，然后将结果传递给 self.func
        return self.func(*[dd(a, n, step) for a in self.args])

    # 将对象转换为 mpmath mpc（复数）类型，如果可能的话
    @property
    def _mpc_(self):
        # 从 .numbers 模块导入 Float 类
        from .numbers import Float
        # 将表达式 self 分解为实部和剩余部分
        re_part, rest = self.as_coeff_Add()
        # 将剩余部分分解为虚部和虚数单位
        im_part, imag_unit = rest.as_coeff_Mul()
        # 如果虚数单位不是 S.ImaginaryUnit，则抛出 AttributeError
        if not imag_unit == S.ImaginaryUnit:
            raise AttributeError("Cannot convert Add to mpc. Must be of the form Number + Number*I")

        # 将实部和虚部分别转换为 Float 类型的 mpmath 格式
        return (Float(re_part)._mpf_, Float(im_part)._mpf_)

    # 对象的负数运算符重载
    def __neg__(self):
        # 如果全局参数 global_parameters.distribute 不为真，则调用父类的 __neg__ 方法
        if not global_parameters.distribute:
            return super().__neg__()
        # 否则，返回乘法运算的结果，即 S.NegativeOne 乘以 self
        return Mul(S.NegativeOne, self)
# 导入 AssocOpDispatcher 类，并初始化 add 变量
add = AssocOpDispatcher('add')

# 从当前目录的 mul 模块中导入 Mul、_keep_coeff、_unevaluated_Mul 三个类或函数
from .mul import Mul, _keep_coeff, _unevaluated_Mul

# 从当前目录的 numbers 模块中导入 Rational 类
from .numbers import Rational
```