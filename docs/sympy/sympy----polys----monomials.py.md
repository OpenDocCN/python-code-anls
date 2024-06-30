# `D:\src\scipysrc\sympy\sympy\polys\monomials.py`

```
# 导入必要的模块和函数：从itertools导入combinations_with_replacement和product，从textwrap导入dedent函数
from itertools import combinations_with_replacement, product
from textwrap import dedent

# 从sympy核心模块导入Mul, S, Tuple, sympify函数；从sympy.polys.polyerrors导入ExactQuotientFailed异常；从sympy.polys.polyutils导入PicklableWithSlots, dict_from_expr函数；从sympy.utilities中导入public函数；从sympy.utilities.iterables中导入is_sequence, iterable函数
from sympy.core import Mul, S, Tuple, sympify
from sympy.polys.polyerrors import ExactQuotientFailed
from sympy.polys.polyutils import PicklableWithSlots, dict_from_expr
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence, iterable

# 使用public装饰器声明itermonomials函数为公共接口
@public
def itermonomials(variables, max_degrees, min_degrees=None):
    r"""
    ``max_degrees`` and ``min_degrees`` are either both integers or both lists.
    Unless otherwise specified, ``min_degrees`` is either ``0`` or
    ``[0, ..., 0]``.

    A generator of all monomials ``monom`` is returned, such that
    either
    ``min_degree <= total_degree(monom) <= max_degree``,
    or
    ``min_degrees[i] <= degree_list(monom)[i] <= max_degrees[i]``,
    for all ``i``.

    Case I. ``max_degrees`` and ``min_degrees`` are both integers
    =============================================================

    Given a set of variables $V$ and a min_degree $N$ and a max_degree $M$
    generate a set of monomials of degree less than or equal to $N$ and greater
    than or equal to $M$. The total number of monomials in commutative
    variables is huge and is given by the following formula if $M = 0$:

        .. math::
            \frac{(\#V + N)!}{\#V! N!}

    For example if we would like to generate a dense polynomial of
    a total degree $N = 50$ and $M = 0$, which is the worst case, in 5
    variables, assuming that exponents and all of coefficients are 32-bit long
    and stored in an array we would need almost 80 GiB of memory! Fortunately
    most polynomials, that we will encounter, are sparse.

    Consider monomials in commutative variables $x$ and $y$
    and non-commutative variables $a$ and $b$::

        >>> from sympy import symbols
        >>> from sympy.polys.monomials import itermonomials
        >>> from sympy.polys.orderings import monomial_key
        >>> from sympy.abc import x, y

        >>> sorted(itermonomials([x, y], 2), key=monomial_key('grlex', [y, x]))
        [1, x, y, x**2, x*y, y**2]

        >>> sorted(itermonomials([x, y], 3), key=monomial_key('grlex', [y, x]))
        [1, x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3]

        >>> a, b = symbols('a, b', commutative=False)
        >>> set(itermonomials([a, b, x], 2))
        {1, a, a**2, b, b**2, x, x**2, a*b, b*a, x*a, x*b}

        >>> sorted(itermonomials([x, y], 2, 1), key=monomial_key('grlex', [y, x]))
        [x, y, x**2, x*y, y**2]

    Case II. ``max_degrees`` and ``min_degrees`` are both lists
    ===========================================================

    If ``max_degrees = [d_1, ..., d_n]`` and
    ``min_degrees = [e_1, ..., e_n]``, the number of monomials generated
    is:

    .. math::
        (d_1 - e_1 + 1) (d_2 - e_2 + 1) \cdots (d_n - e_n + 1)

    Let us generate all monomials ``monom`` in variables $x$ and $y$
    n = len(variables)
    # 计算变量的数量
    if is_sequence(max_degrees):
        # 检查最大度数是否为序列
        if len(max_degrees) != n:
            raise ValueError('Argument sizes do not match')
        # 检查最大度数列表长度是否与变量数量相同
        if min_degrees is None:
            min_degrees = [0]*n
        elif not is_sequence(min_degrees):
            raise ValueError('min_degrees is not a list')
        else:
            if len(min_degrees) != n:
                raise ValueError('Argument sizes do not match')
            if any(i < 0 for i in min_degrees):
                raise ValueError("min_degrees cannot contain negative numbers")
        # 初始化总度数为假
        total_degree = False
    else:
        max_degree = max_degrees
        # 将单一的最大度数赋给max_degree
        if max_degree < 0:
            raise ValueError("max_degrees cannot be negative")
        if min_degrees is None:
            min_degree = 0
        else:
            if min_degrees < 0:
                raise ValueError("min_degrees cannot be negative")
            min_degree = min_degrees
        # 初始化总度数为真
        total_degree = True
    # 如果是总度数模式
    if total_degree:
        if min_degree > max_degree:
            return
        if not variables or max_degree == 0:
            # 如果没有变量或者最大度数为0，则返回单项式1
            yield S.One
            return
        # 强制转换变量为列表，以防传递元组或其他不兼容的集合
        variables = list(variables) + [S.One]
        # 如果所有变量都是可交换的
        if all(variable.is_commutative for variable in variables):
            monomials_list_comm = []
            # 对可交换变量组合产生的项进行处理
            for item in combinations_with_replacement(variables, max_degree):
                powers = dict.fromkeys(variables, 0)
                for variable in item:
                    if variable != 1:
                        powers[variable] += 1
                if sum(powers.values()) >= min_degree:
                    monomials_list_comm.append(Mul(*item))
            # 使用集合确保唯一性，并返回生成器
            yield from set(monomials_list_comm)
        else:
            monomials_list_non_comm = []
            # 对不可交换变量组合产生的项进行处理
            for item in product(variables, repeat=max_degree):
                powers = dict.fromkeys(variables, 0)
                for variable in item:
                    if variable != 1:
                        powers[variable] += 1
                if sum(powers.values()) >= min_degree:
                    monomials_list_non_comm.append(Mul(*item))
            # 使用集合确保唯一性，并返回生成器
            yield from set(monomials_list_non_comm)
    else:
        # 检查每个变量的最小度数是否小于等于最大度数，若有不符合条件的则抛出数值错误异常
        if any(min_degrees[i] > max_degrees[i] for i in range(n)):
            raise ValueError('min_degrees[i] must be <= max_degrees[i] for all i')
        
        # 初始化一个空列表用来存储每个变量的幂次组合列表
        power_lists = []
        
        # 遍历变量、最小度数和最大度数，并生成每个变量的幂次列表，存储在power_lists中
        for var, min_d, max_d in zip(variables, min_degrees, max_degrees):
            power_lists.append([var**i for i in range(min_d, max_d + 1)])
        
        # 使用product函数生成所有变量幂次列表的笛卡尔积，并逐个生成乘积表达式
        for powers in product(*power_lists):
            yield Mul(*powers)
def monomial_count(V, N):
    r"""
    Computes the number of monomials.

    The number of monomials is given by the following formula:

    .. math::

        \frac{(\#V + N)!}{\#V! N!}

    where `N` is a total degree and `V` is a set of variables.

    Examples
    ========

    >>> from sympy.polys.monomials import itermonomials, monomial_count
    >>> from sympy.polys.orderings import monomial_key
    >>> from sympy.abc import x, y

    >>> monomial_count(2, 2)
    6

    >>> M = list(itermonomials([x, y], 2))

    >>> sorted(M, key=monomial_key('grlex', [y, x]))
    [1, x, y, x**2, x*y, y**2]
    >>> len(M)
    6

    """
    # 导入阶乘函数
    from sympy.functions.combinatorial.factorials import factorial
    # 计算并返回单项式的数量
    return factorial(V + N) / factorial(V) / factorial(N)

def monomial_mul(A, B):
    """
    Multiplication of tuples representing monomials.

    Examples
    ========

    Lets multiply `x**3*y**4*z` with `x*y**2`::

        >>> from sympy.polys.monomials import monomial_mul

        >>> monomial_mul((3, 4, 1), (1, 2, 0))
        (4, 6, 1)

    which gives `x**4*y**6*z`.

    """
    # 返回两个单项式元组的对应元素相加后的元组
    return tuple([ a + b for a, b in zip(A, B) ])

def monomial_div(A, B):
    """
    Division of tuples representing monomials.

    Examples
    ========

    Lets divide `x**3*y**4*z` by `x*y**2`::

        >>> from sympy.polys.monomials import monomial_div

        >>> monomial_div((3, 4, 1), (1, 2, 0))
        (2, 2, 1)

    which gives `x**2*y**2*z`. However::

        >>> monomial_div((3, 4, 1), (1, 2, 2)) is None
        True

    `x*y**2*z**2` does not divide `x**3*y**4*z`.

    """
    # 调用monomial_ldiv计算两个单项式元组的商，如果所有元素都非负则返回结果，否则返回None
    C = monomial_ldiv(A, B)

    if all(c >= 0 for c in C):
        return tuple(C)
    else:
        return None

def monomial_ldiv(A, B):
    """
    Division of tuples representing monomials.

    Examples
    ========

    Lets divide `x**3*y**4*z` by `x*y**2`::

        >>> from sympy.polys.monomials import monomial_ldiv

        >>> monomial_ldiv((3, 4, 1), (1, 2, 0))
        (2, 2, 1)

    which gives `x**2*y**2*z`.

        >>> monomial_ldiv((3, 4, 1), (1, 2, 2))
        (2, 2, -1)

    which gives `x**2*y**2*z**-1`.

    """
    # 返回两个单项式元组的对应元素相减后的元组
    return tuple([ a - b for a, b in zip(A, B) ])

def monomial_pow(A, n):
    """Return the n-th pow of the monomial. """
    # 返回单项式元组中每个元素乘以n后的元组
    return tuple([ a*n for a in A ])

def monomial_gcd(A, B):
    """
    Greatest common divisor of tuples representing monomials.

    Examples
    ========

    Lets compute GCD of `x*y**4*z` and `x**3*y**2`::

        >>> from sympy.polys.monomials import monomial_gcd

        >>> monomial_gcd((1, 4, 1), (3, 2, 0))
        (1, 2, 0)

    which gives `x*y**2`.

    """
    # 返回两个单项式元组中对应元素的最小值构成的元组
    return tuple([ min(a, b) for a, b in zip(A, B) ])

def monomial_lcm(A, B):
    """
    Least common multiple of tuples representing monomials.

    Examples
    ========

    Lets compute LCM of `x*y**4*z` and `x**3*y**2`::

        >>> from sympy.polys.monomials import monomial_lcm

        >>> monomial_lcm((1, 4, 1), (3, 2, 0))
        (3, 4, 1)

    """
    # 返回两个单项式元组中对应元素的最大值构成的元组
    return tuple([ max(a, b) for a, b in zip(A, B) ])
    # 定义一个函数，接受两个参数 A 和 B，返回一个元组
    def comp_max(A, B):
        # 使用列表推导式生成一个包含 A 和 B 中每个对应位置最大值的列表，并将其转换为元组返回
        return tuple([ max(a, b) for a, b in zip(A, B) ])
# 定义函数 monomial_divides，判断是否存在一个单项式 X，使得 XA == B
def monomial_divides(A, B):
    # 使用 Python 内置函数 zip 将 A 和 B 中的元素逐对组合成元组，all 函数确保所有元组中的元素满足 a <= b 的条件
    return all(a <= b for a, b in zip(A, B))

# 定义函数 monomial_max，返回一组单项式中每个变量的最大次数
def monomial_max(*monoms):
    # 将第一个单项式 monoms[0] 的副本赋值给 M
    M = list(monoms[0])
    
    # 遍历 monoms 中的其余单项式 N
    for N in monoms[1:]:
        # 使用 enumerate 函数获取索引 i 和元素 n，在 M[i] 和 n 之间取最大值，更新 M 中的值
        for i, n in enumerate(N):
            M[i] = max(M[i], n)
    
    # 返回 M 转换为元组的结果
    return tuple(M)

# 定义函数 monomial_min，返回一组单项式中每个变量的最小次数
def monomial_min(*monoms):
    # 将第一个单项式 monoms[0] 的副本赋值给 M
    M = list(monoms[0])
    
    # 遍历 monoms 中的其余单项式 N
    for N in monoms[1:]:
        # 使用 enumerate 函数获取索引 i 和元素 n，在 M[i] 和 n 之间取最小值，更新 M 中的值
        for i, n in enumerate(N):
            M[i] = min(M[i], n)
    
    # 返回 M 转换为元组的结果
    return tuple(M)

# 定义函数 monomial_deg，返回单项式的总次数
def monomial_deg(M):
    # 使用内置函数 sum 对单项式 M 中的元素求和，得到总次数
    return sum(M)

# 定义函数 term_div，用于在环/域上对两个项进行除法运算
def term_div(a, b, domain):
    # 分别从 a 和 b 中获取领头单项式和领头系数
    a_lm, a_lc = a
    b_lm, b_lc = b
    
    # 判断领头单项式 a_lm 是否能整除 b_lm
    monom = monomial_div(a_lm, b_lm)
    
    # 如果域是一个字段（即具有乘法逆元），且 monom 不为 None
    if domain.is_Field:
        # 返回 monom 和 a_lc 与 b_lc 在域 domain 上的商
        if monom is not None:
            return monom, domain.quo(a_lc, b_lc)
        else:
            return None
    else:
        # 如果不是字段，或者 monom 是 None，或者 a_lc 不能整除 b_lc
        if not (monom is None or a_lc % b_lc):
            return monom, domain.quo(a_lc, b_lc)
        else:
            return None

# 定义类 MonomialOps，用于生成快速单项式算术函数的代码生成器
class MonomialOps:
    """Code generator of fast monomial arithmetic functions. """

    # 初始化方法，接受变量数目 ngens
    def __init__(self, ngens):
        self.ngens = ngens

    # 私有方法 _build，用于执行生成的代码，并返回生成的函数
    def _build(self, code, name):
        ns = {}
        exec(code, ns)
        return ns[name]

    # 私有方法 _vars，用于生成变量名列表，如 ["a0", "a1", "a2"]，长度由 ngens 决定
    def _vars(self, name):
        return [ "%s%s" % (name, i) for i in range(self.ngens) ]

    # 公共方法 mul，生成并返回单项式乘法函数 monomial_mul 的代码
    def mul(self):
        name = "monomial_mul"
        # 定义模板字符串，用于生成 monomial_mul 函数的代码
        template = dedent("""\
        def %(name)s(A, B):
            (%(A)s,) = A
            (%(B)s,) = B
            return (%(AB)s,)
        """)
        # 使用 _vars 方法生成变量列表 A 和 B，并生成相应的乘积列表 AB
        A = self._vars("a")
        B = self._vars("b")
        AB = [ "%s + %s" % (a, b) for a, b in zip(A, B) ]
        # 使用模板字符串填充参数，生成代码字符串
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "AB": ", ".join(AB)}
        # 调用 _build 方法执行生成的代码，返回生成的 monomial_mul 函数
        return self._build(code, name)
    # 定义一个方法 pow，用于生成一元多项式的幂函数
    def pow(self):
        # 设定函数名为 "monomial_pow"
        name = "monomial_pow"
        # 定义模板字符串，生成指定名称的幂函数代码
        template = dedent("""\
        def %(name)s(A, k):
            (%(A)s,) = A
            return (%(Ak)s,)
        """)
        # 获取一元多项式中的变量列表 A
        A = self._vars("a")
        # 构建每个变量乘以 k 后的表达式列表 Ak
        Ak = [ "%s*k" % a for a in A ]
        # 将模板中的变量替换为实际值，生成完整的函数代码
        code = template % {"name": name, "A": ", ".join(A), "Ak": ", ".join(Ak)}
        # 调用 _build 方法，根据生成的代码和函数名构建函数对象并返回
        return self._build(code, name)

    # 定义一个方法 mulpow，用于生成一元多项式的乘幂函数
    def mulpow(self):
        # 设定函数名为 "monomial_mulpow"
        name = "monomial_mulpow"
        # 定义模板字符串，生成指定名称的乘幂函数代码
        template = dedent("""\
        def %(name)s(A, B, k):
            (%(A)s,) = A
            (%(B)s,) = B
            return (%(ABk)s,)
        """)
        # 获取一元多项式中的变量列表 A 和 B
        A = self._vars("a")
        B = self._vars("b")
        # 构建每对变量相加并乘以 k 后的表达式列表 ABk
        ABk = [ "%s + %s*k" % (a, b) for a, b in zip(A, B) ]
        # 将模板中的变量替换为实际值，生成完整的函数代码
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "ABk": ", ".join(ABk)}
        # 调用 _build 方法，根据生成的代码和函数名构建函数对象并返回
        return self._build(code, name)

    # 定义一个方法 ldiv，用于生成一元多项式的左除函数
    def ldiv(self):
        # 设定函数名为 "monomial_ldiv"
        name = "monomial_ldiv"
        # 定义模板字符串，生成指定名称的左除函数代码
        template = dedent("""\
        def %(name)s(A, B):
            (%(A)s,) = A
            (%(B)s,) = B
            return (%(AB)s,)
        """)
        # 获取一元多项式中的变量列表 A 和 B
        A = self._vars("a")
        B = self._vars("b")
        # 构建每对变量相减后的表达式列表 AB
        AB = [ "%s - %s" % (a, b) for a, b in zip(A, B) ]
        # 将模板中的变量替换为实际值，生成完整的函数代码
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "AB": ", ".join(AB)}
        # 调用 _build 方法，根据生成的代码和函数名构建函数对象并返回
        return self._build(code, name)

    # 定义一个方法 div，用于生成一元多项式的除法函数
    def div(self):
        # 设定函数名为 "monomial_div"
        name = "monomial_div"
        # 定义模板字符串，生成指定名称的除法函数代码
        template = dedent("""\
        def %(name)s(A, B):
            (%(A)s,) = A
            (%(B)s,) = B
            %(RAB)s
            return (%(R)s,)
        """)
        # 获取一元多项式中的变量列表 A 和 B
        A = self._vars("a")
        B = self._vars("b")
        # 生成每对变量相减后的表达式列表 AB 和条件判断语句 RAB
        RAB = [ "r%(i)s = a%(i)s - b%(i)s\n    if r%(i)s < 0: return None" % {"i": i} for i in range(self.ngens) ]
        # 获取变量列表 R
        R = self._vars("r")
        # 将模板中的变量替换为实际值，生成完整的函数代码
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "RAB": "\n    ".join(RAB), "R": ", ".join(R)}
        # 调用 _build 方法，根据生成的代码和函数名构建函数对象并返回
        return self._build(code, name)

    # 定义一个方法 lcm，用于生成一元多项式的最小公倍数函数
    def lcm(self):
        # 设定函数名为 "monomial_lcm"
        name = "monomial_lcm"
        # 定义模板字符串，生成指定名称的最小公倍数函数代码
        template = dedent("""\
        def %(name)s(A, B):
            (%(A)s,) = A
            (%(B)s,) = B
            return (%(AB)s,)
        """)
        # 获取一元多项式中的变量列表 A 和 B
        A = self._vars("a")
        B = self._vars("b")
        # 生成每对变量的最小值表达式列表 AB
        AB = [ "%s if %s >= %s else %s" % (a, a, b, b) for a, b in zip(A, B) ]
        # 将模板中的变量替换为实际值，生成完整的函数代码
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "AB": ", ".join(AB)}
        # 调用 _build 方法，根据生成的代码和函数名构建函数对象并返回
        return self._build(code, name)

    # 定义一个方法 gcd，用于生成一元多项式的最大公约数函数
    def gcd(self):
        # 设定函数名为 "monomial_gcd"
        name = "monomial_gcd"
        # 定义模板字符串，生成指定名称的最大公约数函数代码
        template = dedent("""\
        def %(name)s(A, B):
            (%(A)s,) = A
            (%(B)s,) = B
            return (%(AB)s,)
        """)
        # 获取一元多项式中的变量列表 A 和 B
        A = self._vars("a")
        B = self._vars("b")
        # 生成每对变量的最大值表达式列表 AB
        AB = [ "%s if %s <= %s else %s" % (a, a, b, b) for a, b in zip(A, B) ]
        # 将模板中的变量替换为实际值，生成完整的函数代码
        code = template % {"name": name, "A": ", ".join(A), "B": ", ".join(B), "AB": ", ".join(AB)}
        # 调用 _build 方法，根据生成的代码和函数名构建函数对象并返回
        return self._build(code, name)
# 定义一个公开的类 Monomial，继承自 PicklableWithSlots 类
@public
class Monomial(PicklableWithSlots):
    """Class representing a monomial, i.e. a product of powers. """

    # 限制实例只能具有 'exponents' 和 'gens' 两个属性，以节省内存空间
    __slots__ = ('exponents', 'gens')

    # 初始化方法，创建一个单项式对象
    def __init__(self, monom, gens=None):
        # 检查 monom 是否可迭代
        if not iterable(monom):
            # 如果不可迭代，将其转换成 SymPy 表达式的字典表示
            rep, gens = dict_from_expr(sympify(monom), gens=gens)
            # 如果字典长度为1且唯一值为1，则将 monom 更新为唯一键
            if len(rep) == 1 and list(rep.values())[0] == 1:
                monom = list(rep.keys())[0]
            else:
                # 否则抛出异常，表示期望的是单项式
                raise ValueError("Expected a monomial got {}".format(monom))

        # 将单项式的指数部分转换为元组，并初始化属性
        self.exponents = tuple(map(int, monom))
        self.gens = gens

    # 重新构建方法，返回一个新的单项式实例
    def rebuild(self, exponents, gens=None):
        return self.__class__(exponents, gens or self.gens)

    # 返回单项式的长度，即指数的个数
    def __len__(self):
        return len(self.exponents)

    # 返回一个迭代器，迭代单项式的指数部分
    def __iter__(self):
        return iter(self.exponents)

    # 获取单项式指定位置的指数
    def __getitem__(self, item):
        return self.exponents[item]

    # 计算单项式的哈希值
    def __hash__(self):
        return hash((self.__class__.__name__, self.exponents, self.gens))

    # 返回单项式的字符串表示形式
    def __str__(self):
        if self.gens:
            # 如果存在生成器，返回带有生成器和指数的乘积形式
            return "*".join([ "%s**%s" % (gen, exp) for gen, exp in zip(self.gens, self.exponents) ])
        else:
            # 否则返回单项式类名和指数的表示形式
            return "%s(%s)" % (self.__class__.__name__, self.exponents)

    # 将单项式转换为 SymPy 表达式
    def as_expr(self, *gens):
        """Convert a monomial instance to a SymPy expression. """
        gens = gens or self.gens

        # 如果不存在生成器，则无法转换成表达式，抛出异常
        if not gens:
            raise ValueError(
                "Cannot convert %s to an expression without generators" % self)

        # 使用生成器和指数构建 SymPy 表达式
        return Mul(*[ gen**exp for gen, exp in zip(gens, self.exponents) ])

    # 判断两个单项式是否相等
    def __eq__(self, other):
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            return False

        return self.exponents == exponents

    # 判断两个单项式是否不相等
    def __ne__(self, other):
        return not self == other

    # 定义单项式乘法运算
    def __mul__(self, other):
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise NotImplementedError

        # 返回乘积后的新单项式对象
        return self.rebuild(monomial_mul(self.exponents, exponents))

    # 定义单项式除法运算（整数除法）
    def __truediv__(self, other):
        if isinstance(other, Monomial):
            exponents = other.exponents
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            raise NotImplementedError

        # 计算除法结果
        result = monomial_div(self.exponents, exponents)

        if result is not None:
            return self.rebuild(result)
        else:
            # 如果无法精确除尽，则抛出异常
            raise ExactQuotientFailed(self, Monomial(other))

    # 整数除法与真除法相同
    __floordiv__ = __truediv__

    # 定义单项式的乘方运算
    def __pow__(self, other):
        n = int(other)
        if n < 0:
            raise ValueError("a non-negative integer expected, got %s" % other)
        # 返回乘方后的新单项式对象
        return self.rebuild(monomial_pow(self.exponents, n))
    # 计算单项式的最大公约数（Greatest Common Divisor, GCD）
    def gcd(self, other):
        """Greatest common divisor of monomials. """
        # 如果参数 `other` 是 Monomial 类的实例，则使用其指数
        if isinstance(other, Monomial):
            exponents = other.exponents
        # 如果参数 `other` 是元组（tuple）或 Tuple 类型，则直接使用其值
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            # 如果参数 `other` 类型不符合预期，抛出类型错误异常
            raise TypeError(
                "an instance of Monomial class expected, got %s" % other)

        # 调用 `monomial_gcd` 函数计算指数的最大公约数，并使用 `self.rebuild` 方法重建单项式
        return self.rebuild(monomial_gcd(self.exponents, exponents))

    # 计算单项式的最小公倍数（Least Common Multiple, LCM）
    def lcm(self, other):
        """Least common multiple of monomials. """
        # 如果参数 `other` 是 Monomial 类的实例，则使用其指数
        if isinstance(other, Monomial):
            exponents = other.exponents
        # 如果参数 `other` 是元组（tuple）或 Tuple 类型，则直接使用其值
        elif isinstance(other, (tuple, Tuple)):
            exponents = other
        else:
            # 如果参数 `other` 类型不符合预期，抛出类型错误异常
            raise TypeError(
                "an instance of Monomial class expected, got %s" % other)

        # 调用 `monomial_lcm` 函数计算指数的最小公倍数，并使用 `self.rebuild` 方法重建单项式
        return self.rebuild(monomial_lcm(self.exponents, exponents))
```