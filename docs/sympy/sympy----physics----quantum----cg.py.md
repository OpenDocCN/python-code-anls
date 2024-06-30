# `D:\src\scipysrc\sympy\sympy\physics\quantum\cg.py`

```
#TODO:
# -Implement Clebsch-Gordan symmetries
# -Improve simplification method
# -Implement new simplifications
"""Clebsch-Gordon Coefficients."""

# 导入必要的符号计算库函数
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.printing.pretty.stringpict import prettyForm, stringPict

# 导入特殊函数，例如 KroneckerDelta 和 Wigner 系列函数
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j, wigner_9j
from sympy.printing.precedence import PRECEDENCE

# 导出的符号列表
__all__ = [
    'CG',
    'Wigner3j',
    'Wigner6j',
    'Wigner9j',
    'cg_simp'
]

#-----------------------------------------------------------------------------
# CG Coefficients
#-----------------------------------------------------------------------------


class Wigner3j(Expr):
    """Class for the Wigner-3j symbols.

    Explanation
    ===========

    Wigner 3j-symbols are coefficients determined by the coupling of
    two angular momenta. When created, they are expressed as symbolic
    quantities that, for numerical parameters, can be evaluated using the
    ``.doit()`` method [1]_.

    Parameters
    ==========

    j1, m1, j2, m2, j3, m3 : Number, Symbol
        Terms determining the angular momentum of coupled angular momentum
        systems.

    Examples
    ========

    Declare a Wigner-3j coefficient and calculate its value

        >>> from sympy.physics.quantum.cg import Wigner3j
        >>> w3j = Wigner3j(6,0,4,0,2,0)
        >>> w3j
        Wigner3j(6, 0, 4, 0, 2, 0)
        >>> w3j.doit()
        sqrt(715)/143

    See Also
    ========

    CG: Clebsch-Gordan coefficients

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """

    is_commutative = True

    def __new__(cls, j1, m1, j2, m2, j3, m3):
        args = map(sympify, (j1, m1, j2, m2, j3, m3))
        return Expr.__new__(cls, *args)

    @property
    def j1(self):
        return self.args[0]

    @property
    def m1(self):
        return self.args[1]

    @property
    def j2(self):
        return self.args[2]

    @property
    def m2(self):
        return self.args[3]

    @property
    def j3(self):
        return self.args[4]

    @property
    def m3(self):
        return self.args[5]

    @property
    def is_symbolic(self):
        return not all(arg.is_number for arg in self.args)

    # This is modified from the _print_Matrix method
    # 生成美观的输出格式，包含三行每行两个元素的元组，元素为打印后的 j 和 m 值
    m = ((printer._print(self.j1), printer._print(self.m1)),
         (printer._print(self.j2), printer._print(self.m2)),
         (printer._print(self.j3), printer._print(self.m3)))

    # 设定水平和垂直分隔符的大小
    hsep = 2
    vsep = 1

    # 初始化最大宽度列表，包含三个元素，初始值为 -1
    maxw = [-1]*3

    # 遍历每一行，计算每个元素的最大宽度
    for j in range(3):
        maxw[j] = max(m[j][i].width() for i in range(2))

    # 初始化输出的表格形式对象为 None
    D = None

    # 对每一列进行处理
    for i in range(2):
        D_row = None

        # 对每一行进行处理
        for j in range(3):
            # 获取当前元素的打印后对象
            s = m[j][i]

            # 计算当前元素与最大宽度之间的差异
            wdelta = maxw[j] - s.width()
            wleft = wdelta // 2  # 左侧填充空格数量
            wright = wdelta - wleft  # 右侧填充空格数量

            # 右侧填充空格
            s = prettyForm(*s.right(' ' * wright))

            # 左侧填充空格
            s = prettyForm(*s.left(' ' * wleft))

            # 如果当前行的 D_row 为空，则直接赋值当前元素
            if D_row is None:
                D_row = s
                continue

            # 在当前行的 D_row 后添加水平分隔符
            D_row = prettyForm(*D_row.right(' ' * hsep))

            # 将当前元素添加到 D_row 中
            D_row = prettyForm(*D_row.right(s))

        # 如果 D 为空，则直接赋值当前行的 D_row
        if D is None:
            D = D_row
            continue

        # 在 D 和当前行之间添加垂直分隔符
        for _ in range(vsep):
            D = prettyForm(*D.below(' '))

        # 将当前行的 D_row 添加到 D 中
        D = prettyForm(*D.below(D_row))

    # 将整体结果用括号包围起来
    D = prettyForm(*D.parens())

    # 返回生成的美观输出格式
    return D
class CG(Wigner3j):
    r"""Class for Clebsch-Gordan coefficient.

    Explanation
    ===========

    Clebsch-Gordan coefficients describe the angular momentum coupling between
    two systems. The coefficients give the expansion of a coupled total angular
    momentum state and an uncoupled tensor product state. The Clebsch-Gordan
    coefficients are defined as [1]_:

    .. math ::
        C^{j_3,m_3}_{j_1,m_1,j_2,m_2} = \left\langle j_1,m_1;j_2,m_2 | j_3,m_3\right\rangle

    Parameters
    ==========

    j1, m1, j2, m2 : Number, Symbol
        Angular momenta of states 1 and 2.

    j3, m3: Number, Symbol
        Total angular momentum of the coupled system.

    Examples
    ========

    Define a Clebsch-Gordan coefficient and evaluate its value

        >>> from sympy.physics.quantum.cg import CG
        >>> from sympy import S
        >>> cg = CG(S(3)/2, S(3)/2, S(1)/2, -S(1)/2, 1, 1)
        >>> cg
        CG(3/2, 3/2, 1/2, -1/2, 1, 1)
        >>> cg.doit()
        sqrt(3)/2
        >>> CG(j1=S(1)/2, m1=-S(1)/2, j2=S(1)/2, m2=+S(1)/2, j3=1, m3=0).doit()
        sqrt(2)/2


    Compare [2]_.

    See Also
    ========

    Wigner3j: Wigner-3j symbols

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    .. [2] `Clebsch-Gordan Coefficients, Spherical Harmonics, and d Functions
        <https://pdg.lbl.gov/2020/reviews/rpp2020-rev-clebsch-gordan-coefs.pdf>`_
        in P.A. Zyla *et al.* (Particle Data Group), Prog. Theor. Exp. Phys.
        2020, 083C01 (2020).
    """
    # 设置运算符优先级，这里是幂运算的优先级减一
    precedence = PRECEDENCE["Pow"] - 1

    def doit(self, **hints):
        # 如果是符号表示，抛出数值错误
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        # 调用 clebsch_gordan 函数计算 Clebsch-Gordan 系数的值并返回
        return clebsch_gordan(self.j1, self.j2, self.j3, self.m1, self.m2, self.m3)

    def _pretty(self, printer, *args):
        # 打印美化后的表达式
        bot = printer._print_seq(
            (self.j1, self.m1, self.j2, self.m2), delimiter=',')
        top = printer._print_seq((self.j3, self.m3), delimiter=',')

        pad = max(top.width(), bot.width())
        bot = prettyForm(*bot.left(' '))
        top = prettyForm(*top.left(' '))

        if not pad == bot.width():
            bot = prettyForm(*bot.right(' '*(pad - bot.width())))
        if not pad == top.width():
            top = prettyForm(*top.right(' '*(pad - top.width())))
        s = stringPict('C' + ' '*pad)
        s = prettyForm(*s.below(bot))
        s = prettyForm(*s.above(top))
        return s

    def _latex(self, printer, *args):
        # 打印 LaTeX 格式的表达式
        label = map(printer._print, (self.j3, self.m3, self.j1,
                    self.m1, self.j2, self.m2))
        return r'C^{%s,%s}_{%s,%s,%s,%s}' % tuple(label)


class Wigner6j(Expr):
    """Class for the Wigner-6j symbols

    See Also
    ========

    Wigner3j: Wigner-3j symbols

    """
    def __new__(cls, j1, j2, j12, j3, j, j23):
        args = map(sympify, (j1, j2, j12, j3, j, j23))
        return Expr.__new__(cls, *args)

    @property
    # 返回第一个参数 self.args[0]
    def j1(self):
        return self.args[0]

    # 返回第二个参数 self.args[1]，作为属性访问
    @property
    def j2(self):
        return self.args[1]

    # 返回第三个参数 self.args[2]，作为属性访问
    @property
    def j12(self):
        return self.args[2]

    # 返回第四个参数 self.args[3]，作为属性访问
    @property
    def j3(self):
        return self.args[3]

    # 返回第五个参数 self.args[4]，作为属性访问
    @property
    def j(self):
        return self.args[4]

    # 返回第六个参数 self.args[5]，作为属性访问
    @property
    def j23(self):
        return self.args[5]

    # 判断对象是否为符号表达式，如果有任意一个参数不是数值类型则返回 True
    @property
    def is_symbolic(self):
        return not all(arg.is_number for arg in self.args)

    # 从 _print_Matrix 方法修改而来，生成美观的输出格式
    def _pretty(self, printer, *args):
        # 构建矩阵 m，包含打印每个参数的结果
        m = ((printer._print(self.j1), printer._print(self.j3)),
            (printer._print(self.j2), printer._print(self.j)),
            (printer._print(self.j12), printer._print(self.j23)))
        # 设置水平和垂直分隔符
        hsep = 2
        vsep = 1
        # 计算每列的最大宽度
        maxw = [-1]*3
        for j in range(3):
            maxw[j] = max(m[j][i].width() for i in range(2))
        # 初始化输出结果
        D = None
        # 遍历矩阵行
        for i in range(2):
            D_row = None
            # 遍历矩阵列
            for j in range(3):
                s = m[j][i]
                # 计算左右填充空格的数量，使得每列对齐
                wdelta = maxw[j] - s.width()
                wleft = wdelta // 2
                wright = wdelta - wleft

                # 在右侧填充空格
                s = prettyForm(*s.right(' ' * wright))
                # 在左侧填充空格
                s = prettyForm(*s.left(' ' * wleft))

                # 如果当前行为空，直接赋值为 s
                if D_row is None:
                    D_row = s
                    continue
                # 在当前行右侧添加水平分隔符
                D_row = prettyForm(*D_row.right(' ' * hsep))
                # 在当前行右侧添加 s
                D_row = prettyForm(*D_row.right(s))
            # 如果结果 D 为空，直接赋值为当前行 D_row
            if D is None:
                D = D_row
                continue
            # 在 D 和 D_row 之间添加垂直分隔符
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))
            # 在 D 的下方添加 D_row
            D = prettyForm(*D.below(D_row))
        # 在结果外围加上花括号，返回最终美观输出
        D = prettyForm(*D.parens(left='{', right='}'))
        return D

    # 将对象转换为 LaTeX 格式的字符串
    def _latex(self, printer, *args):
        # 使用打印器打印每个参数的结果，生成 LaTeX 字符串
        label = map(printer._print, (self.j1, self.j2, self.j12,
                    self.j3, self.j, self.j23))
        # 返回格式化后的 LaTeX 字符串
        return r'\left\{\begin{array}{ccc} %s & %s & %s \\ %s & %s & %s \end{array}\right\}' % \
            tuple(label)

    # 执行计算，如果有符号参数则引发 ValueError 异常
    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        # 调用 wigner_6j 函数计算结果并返回
        return wigner_6j(self.j1, self.j2, self.j12, self.j3, self.j, self.j23)
class Wigner9j(Expr):
    """Class for the Wigner-9j symbols
    
    See Also
    ========
    
    Wigner3j: Wigner-3j symbols
    
    """
    # 初始化方法，创建一个新的 Wigner9j 对象
    def __new__(cls, j1, j2, j12, j3, j4, j34, j13, j24, j):
        # 将输入参数符号化，并调用基类 Expr 的 __new__ 方法进行实例化
        args = map(sympify, (j1, j2, j12, j3, j4, j34, j13, j24, j))
        return Expr.__new__(cls, *args)

    # 返回属性 j1 的值
    @property
    def j1(self):
        return self.args[0]

    # 返回属性 j2 的值
    @property
    def j2(self):
        return self.args[1]

    # 返回属性 j12 的值
    @property
    def j12(self):
        return self.args[2]

    # 返回属性 j3 的值
    @property
    def j3(self):
        return self.args[3]

    # 返回属性 j4 的值
    @property
    def j4(self):
        return self.args[4]

    # 返回属性 j34 的值
    @property
    def j34(self):
        return self.args[5]

    # 返回属性 j13 的值
    @property
    def j13(self):
        return self.args[6]

    # 返回属性 j24 的值
    @property
    def j24(self):
        return self.args[7]

    # 返回属性 j 的值
    @property
    def j(self):
        return self.args[8]

    # 返回是否包含符号的布尔值
    @property
    def is_symbolic(self):
        return not all(arg.is_number for arg in self.args)

    # 用于美化输出 LaTeX 格式的方法，基于 _print_Matrix 方法修改
    def _pretty(self, printer, *args):
        # 创建一个包含符号的矩阵 m
        m = (
            (printer._print(self.j1), printer._print(self.j3), printer._print(self.j13)),
            (printer._print(self.j2), printer._print(self.j4), printer._print(self.j24)),
            (printer._print(self.j12), printer._print(self.j34), printer._print(self.j)))
        hsep = 2  # 水平分隔
        vsep = 1  # 垂直分隔
        maxw = [-1]*3  # 存放每列的最大宽度
        # 计算每列的最大宽度
        for j in range(3):
            maxw[j] = max(m[j][i].width() for i in range(3))
        D = None
        # 构建输出矩阵 D
        for i in range(3):
            D_row = None
            for j in range(3):
                s = m[j][i]
                wdelta = maxw[j] - s.width()
                wleft = wdelta // 2
                wright = wdelta - wleft

                s = prettyForm(*s.right(' ' * wright))
                s = prettyForm(*s.left(' ' * wleft))

                if D_row is None:
                    D_row = s
                    continue
                D_row = prettyForm(*D_row.right(' ' * hsep))
                D_row = prettyForm(*D_row.right(s))
            if D is None:
                D = D_row
                continue
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))
            D = prettyForm(*D.below(D_row))
        D = prettyForm(*D.parens(left='{', right='}'))  # 加上大括号
        return D

    # 输出 LaTeX 格式的方法
    def _latex(self, printer, *args):
        # 将参数符号化为字符串
        label = map(printer._print, (self.j1, self.j2, self.j12, self.j3,
                self.j4, self.j34, self.j13, self.j24, self.j))
        # 返回 LaTeX 格式的表达式
        return r'\left\{\begin{array}{ccc} %s & %s & %s \\ %s & %s & %s \\ %s & %s & %s \end{array}\right\}' % \
            tuple(label)

    # 进行计算的方法，如果包含符号则引发 ValueError 异常
    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return wigner_9j(self.j1, self.j2, self.j12, self.j3, self.j4, self.j34, self.j13, self.j24, self.j)
    # 如果输入的表达式 e 是加法表达式（Add对象），则调用_cg_simp_add函数简化
    if isinstance(e, Add):
        return _cg_simp_add(e)
    # 如果输入的表达式 e 是求和表达式（Sum对象），则调用_cg_simp_sum函数简化
    elif isinstance(e, Sum):
        return _cg_simp_sum(e)
    # 如果输入的表达式 e 是乘法表达式（Mul对象），则对每个乘法因子调用cg_simp函数进行简化，并返回乘法的结果
    elif isinstance(e, Mul):
        return Mul(*[cg_simp(arg) for arg in e.args])
    # 如果输入的表达式 e 是幂运算表达式（Pow对象），则对幂的底数和指数分别调用cg_simp函数进行简化，并返回简化后的幂运算结果
    elif isinstance(e, Pow):
        return Pow(cg_simp(e.base), e.exp)
    # 对于其他类型的输入表达式 e，直接返回表达式 e，不做任何简化
    else:
        return e
def _cg_simp_add(e):
    #TODO: Improve simplification method
    """Takes a sum of terms involving Clebsch-Gordan coefficients and
    simplifies the terms.

    Explanation
    ===========

    First, we create two lists, cg_part, which contains terms involving CG
    coefficients, and other_part, which contains all other terms. We then iterate
    through each term in the expression 'e'. If the term involves CG coefficients,
    we categorize it into cg_part; otherwise, into other_part. After categorization,
    we apply simplification methods to terms in cg_part and accumulate any new terms
    into other_part.
    """
    cg_part = []
    other_part = []

    e = expand(e)
    for arg in e.args:
        if arg.has(CG):
            if isinstance(arg, Sum):
                other_part.append(_cg_simp_sum(arg))
            elif isinstance(arg, Mul):
                terms = 1
                for term in arg.args:
                    if isinstance(term, Sum):
                        terms *= _cg_simp_sum(term)
                    else:
                        terms *= term
                if terms.has(CG):
                    cg_part.append(terms)
                else:
                    other_part.append(terms)
            else:
                cg_part.append(arg)
        else:
            other_part.append(arg)

    cg_part, other = _check_varsh_871_1(cg_part)
    other_part.append(other)
    cg_part, other = _check_varsh_871_2(cg_part)
    other_part.append(other)
    cg_part, other = _check_varsh_872_9(cg_part)
    other_part.append(other)
    return Add(*cg_part) + Add(*other_part)


def _check_varsh_871_1(term_list):
    # Sum( CG(a,alpha,b,0,a,alpha), (alpha, -a, a)) == KroneckerDelta(b,0)
    a, alpha, b, lt = map(Wild, ('a', 'alpha', 'b', 'lt'))
    expr = lt*CG(a, alpha, b, 0, a, alpha)
    simp = (2*a + 1)*KroneckerDelta(b, 0)
    sign = lt/abs(lt)
    build_expr = 2*a + 1
    index_expr = a + alpha
    return _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, b, lt), (a, b), build_expr, index_expr)


def _check_varsh_871_2(term_list):
    # Sum((-1)**(a-alpha)*CG(a,alpha,a,-alpha,c,0),(alpha,-a,a))
    a, alpha, c, lt = map(Wild, ('a', 'alpha', 'c', 'lt'))
    expr = lt*CG(a, alpha, a, -alpha, c, 0)
    simp = sqrt(2*a + 1)*KroneckerDelta(c, 0)
    sign = (-1)**(a - alpha)*lt/abs(lt)
    build_expr = 2*a + 1
    index_expr = a + alpha
    return _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, c, lt), (a, c), build_expr, index_expr)


def _check_varsh_872_9(term_list):
    # Sum( CG(a,alpha,b,beta,c,gamma)*CG(a,alpha',b,beta',c,gamma), (gamma, -c, c), (c, abs(a-b), a+b))
    a, alpha, alphap, b, beta, betap, c, gamma, lt = map(Wild, (
        'a', 'alpha', 'alphap', 'b', 'beta', 'betap', 'c', 'gamma', 'lt'))
    # Case alpha==alphap, beta==betap

    # For numerical alpha,beta
    expr = lt*CG(a, alpha, b, beta, c, gamma)**2
    simp = S.One
    sign = lt/abs(lt)
    x = abs(a - b)
    y = abs(alpha + beta)
    build_expr = a + b + 1 - Piecewise((x, x > y), (0, Eq(x, y)), (y, y > x))
    index_expr = a + b - c
    # 调用 _check_cg_simp 函数，处理简化符号表达式
    term_list, other1 = _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, b, beta, c, gamma, lt), (a, alpha, b, beta), build_expr, index_expr)

    # 用于处理符号 alpha 和 beta 的情况
    x = abs(a - b)
    y = a + b
    build_expr = (y + 1 - x)*(x + y + 1)
    index_expr = (c - x)*(x + c) + c + gamma
    # 再次调用 _check_cg_simp 函数，处理简化后的表达式
    term_list, other2 = _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, b, beta, c, gamma, lt), (a, alpha, b, beta), build_expr, index_expr)

    # 处理 alpha 不等于 alphap 或 beta 不等于 betap 的情况
    # 注意：仅当主导项为 1 时有效，模式匹配无法处理带有通配符主导项的情况
    # 用于处理数值类型的 alpha、alphap、beta、betap
    expr = CG(a, alpha, b, beta, c, gamma)*CG(a, alphap, b, betap, c, gamma)
    simp = KroneckerDelta(alpha, alphap)*KroneckerDelta(beta, betap)
    sign = S.One
    x = abs(a - b)
    y = abs(alpha + beta)
    build_expr = a + b + 1 - Piecewise((x, x > y), (0, Eq(x, y)), (y, y > x))
    index_expr = a + b - c
    # 再次调用 _check_cg_simp 函数，处理简化后的表达式
    term_list, other3 = _check_cg_simp(expr, simp, sign, S.One, term_list, (a, alpha, alphap, b, beta, betap, c, gamma), (a, alpha, alphap, b, beta, betap), build_expr, index_expr)

    # 处理符号类型的 alpha、alphap、beta、betap
    x = abs(a - b)
    y = a + b
    build_expr = (y + 1 - x)*(x + y + 1)
    index_expr = (c - x)*(x + c) + c + gamma
    # 最后一次调用 _check_cg_simp 函数，处理简化后的表达式，并返回处理结果
    term_list, other4 = _check_cg_simp(expr, simp, sign, S.One, term_list, (a, alpha, alphap, b, beta, betap, c, gamma), (a, alpha, alphap, b, beta, betap), build_expr, index_expr)

    # 返回 term_list 和计算的其他结果之和
    return term_list, other1 + other2 + other4
# 检查并简化给定的费曼图项，返回简化后的项列表和由简化生成的任何项

def _check_cg_simp(expr, simp, sign, lt, term_list, variables, dep_variables, build_index_expr, index_expr):
    """ Checks for simplifications that can be made, returning a tuple of the
    simplified list of terms and any terms generated by simplification.

    Parameters
    ==========

    expr: expression
        The expression with Wild terms that will be matched to the terms in
        the sum

    simp: expression
        The expression with Wild terms that is substituted in place of the CG
        terms in the case of simplification

    sign: expression
        The expression with Wild terms denoting the sign that is on expr that
        must match

    lt: expression
        The expression with Wild terms that gives the leading term of the
        matched expr

    term_list: list
        A list of all of the terms is the sum to be simplified

    variables: list
        A list of all the variables that appears in expr

    dep_variables: list
        A list of the variables that must match for all the terms in the sum,
        i.e. the dependent variables

    build_index_expr: expression
        Expression with Wild terms giving the number of elements in cg_index

    index_expr: expression
        Expression with Wild terms giving the index terms have when storing
        them to cg_index

    """
    other_part = 0  # 初始化其他部分的计数
    i = 0  # 初始化循环计数器
    while i < len(term_list):  # 循环直到处理完所有项
        sub_1 = _check_cg(term_list[i], expr, len(variables))  # 检查当前项是否匹配表达式
        if sub_1 is None:  # 如果不匹配，继续下一项
            i += 1
            continue
        if not build_index_expr.subs(sub_1).is_number:  # 如果构建索引表达式不是数值，继续下一项
            i += 1
            continue
        sub_dep = [(x, sub_1[x]) for x in dep_variables]  # 提取依赖变量的子表达式
        cg_index = [None]*build_index_expr.subs(sub_1)  # 初始化CG索引列表
        for j in range(i, len(term_list)):  # 遍历剩余的项
            sub_2 = _check_cg(term_list[j], expr.subs(sub_dep), len(variables) - len(dep_variables), sign=(sign.subs(sub_1), sign.subs(sub_dep)))  # 检查是否匹配第二个表达式
            if sub_2 is None:  # 如果不匹配，继续下一项
                continue
            if not index_expr.subs(sub_dep).subs(sub_2).is_number:  # 如果索引表达式不是数值，继续下一项
                continue
            cg_index[index_expr.subs(sub_dep).subs(sub_2)] = j, expr.subs(lt, 1).subs(sub_dep).subs(sub_2), lt.subs(sub_2), sign.subs(sub_dep).subs(sub_2)  # 将匹配的项加入CG索引
        if not any(i is None for i in cg_index):  # 如果所有项都匹配
            min_lt = min(*[ abs(term[2]) for term in cg_index ])  # 找到最小领头项的绝对值
            indices = [ term[0] for term in cg_index]
            indices.sort()  # 排序索引列表
            indices.reverse()  # 反转索引列表
            [ term_list.pop(j) for j in indices ]  # 删除匹配项
            for term in cg_index:  # 对CG索引中的每个项
                if abs(term[2]) > min_lt:  # 如果领头项的绝对值大于最小领头项
                    term_list.append( (term[2] - min_lt*term[3])*term[1] )  # 添加调整后的项
            other_part += min_lt*(sign*simp).subs(sub_1)  # 计算其他部分的贡献
        else:  # 如果有项不匹配
            i += 1  # 继续下一项
    return term_list, other_part  # 返回简化后的项列表和其他部分的总和


def _check_cg(cg_term, expr, length, sign=None):
    """Checks whether a term matches the given expression"""
    # TODO: Check for symmetries
    matches = cg_term.match(expr)  # 尝试匹配费曼图项和给定表达式
    if matches is None:  # 如果没有匹配，返回None
        return
    # 如果参数 sign 不为 None，则执行下面的条件判断
    if sign is not None:
        # 如果 sign 不是元组类型，则抛出类型错误异常
        if not isinstance(sign, tuple):
            raise TypeError('sign must be a tuple')
        # 如果 sign 的第一个元素不等于第二个元素通过 matches 变量替换后的结果，则返回
        if not sign[0] == (sign[1]).subs(matches):
            return
    # 如果 matches 列表的长度等于 length 变量的值，则返回 matches 列表
    if len(matches) == length:
        return matches
# 对给定表达式 e 应用变换 _check_varsh_sum_871_1，返回变换后的结果
def _cg_simp_sum(e):
    e = _check_varsh_sum_871_1(e)
    # 对变换后的表达式再次应用 _check_varsh_sum_871_2，返回结果
    e = _check_varsh_sum_871_2(e)
    # 对变换后的表达式再次应用 _check_varsh_sum_872_4，返回结果
    e = _check_varsh_sum_872_4(e)
    # 返回最终变换后的表达式结果
    return e


# 匹配表达式 e 中的 CG 符号并执行变换
def _check_varsh_sum_871_1(e):
    # 定义符号变量和通配符
    a = Wild('a')
    alpha = symbols('alpha')
    b = Wild('b')
    # 尝试匹配表达式 e 中的模式 Sum(CG(...), (alpha, -a, a))
    match = e.match(Sum(CG(a, alpha, b, 0, a, alpha), (alpha, -a, a)))
    # 如果匹配成功且匹配项长度为 2
    if match is not None and len(match) == 2:
        # 应用替换公式 ((2*a + 1)*KroneckerDelta(b, 0))，返回替换后的结果
        return ((2*a + 1)*KroneckerDelta(b, 0)).subs(match)
    # 如果匹配失败或者匹配项长度不为 2，则返回原始表达式 e
    return e


# 匹配表达式 e 中的 CG 符号并执行变换
def _check_varsh_sum_871_2(e):
    # 定义符号变量和通配符
    a = Wild('a')
    alpha = symbols('alpha')
    c = Wild('c')
    # 尝试匹配表达式 e 中的模式 Sum((-1)**(a - alpha)*CG(...), (alpha, -a, a))
    match = e.match(
        Sum((-1)**(a - alpha)*CG(a, alpha, a, -alpha, c, 0), (alpha, -a, a)))
    # 如果匹配成功且匹配项长度为 2
    if match is not None and len(match) == 2:
        # 应用替换公式 (sqrt(2*a + 1)*KroneckerDelta(c, 0))，返回替换后的结果
        return (sqrt(2*a + 1)*KroneckerDelta(c, 0)).subs(match)
    # 如果匹配失败或者匹配项长度不为 2，则返回原始表达式 e
    return e


# 匹配表达式 e 中的 CG 符号并执行变换
def _check_varsh_sum_872_4(e):
    # 定义符号变量和通配符
    alpha = symbols('alpha')
    beta = symbols('beta')
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    cp = Wild('cp')
    gamma = Wild('gamma')
    gammap = Wild('gammap')
    # 创建 CG 对象
    cg1 = CG(a, alpha, b, beta, c, gamma)
    cg2 = CG(a, alpha, b, beta, cp, gammap)
    # 尝试匹配表达式 e 中的模式 Sum(cg1*cg2, (alpha, -a, a), (beta, -b, b))
    match1 = e.match(Sum(cg1*cg2, (alpha, -a, a), (beta, -b, b)))
    # 如果匹配成功且匹配项长度为 6
    if match1 is not None and len(match1) == 6:
        # 应用替换公式 (KroneckerDelta(c, cp)*KroneckerDelta(gamma, gammap))，返回替换后的结果
        return (KroneckerDelta(c, cp)*KroneckerDelta(gamma, gammap)).subs(match1)
    # 尝试匹配表达式 e 中的模式 Sum(cg1**2, (alpha, -a, a), (beta, -b, b))
    match2 = e.match(Sum(cg1**2, (alpha, -a, a), (beta, -b, b)))
    # 如果匹配成功且匹配项长度为 4
    if match2 is not None and len(match2) == 4:
        # 返回 S.One，即数值 1
        return S.One
    # 如果匹配失败或者匹配项长度不为 6 或 4，则返回原始表达式 e
    return e


# 将表达式 term 转换为 CG 对象的列表，返回 CG 对象列表、系数和正负号
def _cg_list(term):
    # 如果 term 是 CG 对象，则返回单个元组和系数 1、正负号 1
    if isinstance(term, CG):
        return (term,), 1, 1
    # 否则，term 必须是 Add、Mul 或 Pow 类型，否则抛出 NotImplementedError 异常
    cg = []
    coeff = 1
    if not isinstance(term, (Mul, Pow)):
        raise NotImplementedError('term must be CG, Add, Mul or Pow')
    # 如果 term 是 Pow 类型且指数是数字
    if isinstance(term, Pow) and term.exp.is_number:
        # 将 term.base 添加到 cg 中 exp 次
        [ cg.append(term.base) for _ in range(term.exp) ]
    else:
        # 否则返回 term 本身作为单个元组，系数为 1，正负号为 1
        return (term,), 1, 1
    # 如果 term 是 Mul 类型
    if isinstance(term, Mul):
        # 遍历 term 的参数
        for arg in term.args:
            # 如果参数是 CG 对象，则添加到 cg 列表中，否则更新 coeff
            if isinstance(arg, CG):
                cg.append(arg)
            else:
                coeff *= arg
        # 返回 CG 对象列表、系数和正负号
        return cg, coeff, coeff/abs(coeff)
```