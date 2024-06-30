# `D:\src\scipysrc\sympy\sympy\polys\numberfields\galois_resolvents.py`

```
"""
Galois resolvents

Each of the functions in ``sympy.polys.numberfields.galoisgroups`` that
computes Galois groups for a particular degree $n$ uses resolvents. Given the
polynomial $T$ whose Galois group is to be computed, a resolvent is a
polynomial $R$ whose roots are defined as functions of the roots of $T.

One way to compute the coefficients of $R$ is by approximating the roots of $T$
to sufficient precision. This module defines a :py:class:`~.Resolvent` class
that handles this job, determining the necessary precision, and computing $R$.

In some cases, the coefficients of $R$ are symmetric in the roots of $T$,
meaning they are equal to fixed functions of the coefficients of $T$. Therefore
another approach is to compute these functions once and for all, and record
them in a lookup table. This module defines code that can compute such tables.
The tables for polynomials $T$ of degrees 4 through 6, produced by this code,
are recorded in the resolvent_lookup.py module.
"""

from sympy.core.evalf import (
    evalf, fastlog, _evalf_with_bounded_error, quad_to_mpmath,
)  # 导入数值评估相关函数

from sympy.core.symbol import symbols, Dummy  # 导入符号和虚拟符号类

from sympy.polys.densetools import dup_eval  # 导入多项式的快速评估工具
from sympy.polys.domains import ZZ  # 导入整数环
from sympy.polys.numberfields.resolvent_lookup import resolvent_coeff_lambdas  # 导入预解决系数函数
from sympy.polys.orderings import lex  # 导入词典序
from sympy.polys.polyroots import preprocess_roots  # 导入多项式根预处理函数
from sympy.polys.polytools import Poly  # 导入多项式类
from sympy.polys.rings import xring  # 导入多项式环类
from sympy.polys.specialpolys import symmetric_poly  # 导入对称多项式函数
from sympy.utilities.lambdify import lambdify  # 导入 lambdify 函数

from mpmath import MPContext  # 导入多精度上下文
from mpmath.libmp.libmpf import prec_to_dps  # 导入精度转换函数

class GaloisGroupException(Exception):
    ...  # Galois 分组异常类，继承自异常类

class ResolventException(GaloisGroupException):
    ...  # 解决多项式异常类，继承自 Galois 分组异常类

class Resolvent:
    r"""
    If $G$ is a subgroup of the symmetric group $S_n$,
    $F$ a multivariate polynomial in $\mathbb{Z}[X_1, \ldots, X_n]$,
    $H$ the stabilizer of $F$ in $G$ (i.e. the permutations $\sigma$ such that
    $F(X_{\sigma(1)}, \ldots, X_{\sigma(n)}) = F(X_1, \ldots, X_n)$), and $s$
    a set of left coset representatives of $H$ in $G$, then the resolvent
    polynomial $R(Y)$ is the product over $\sigma \in s$ of
    $Y - F(X_{\sigma(1)}, \ldots, X_{\sigma(n)})$.

    For example, consider the resolvent for the form
    $$F = X_0 X_2 + X_1 X_3$$
    and the group $G = S_4$. In this case, the stabilizer $H$ is the dihedral
    group $D4 = < (0123), (02) >$, and a set of representatives of $G/H$ is
    $\{I, (01), (03)\}$. The resolvent can be constructed as follows:

    >>> from sympy.combinatorics.permutations import Permutation
    >>> from sympy.core.symbol import symbols
    >>> from sympy.polys.numberfields.galoisgroups import Resolvent
    >>> X = symbols('X0 X1 X2 X3')
    >>> F = X[0]*X[2] + X[1]*X[3]
    >>> s = [Permutation([0, 1, 2, 3]), Permutation([1, 0, 2, 3]),
    ... Permutation([3, 1, 2, 0])]
    >>> R = Resolvent(F, X, s)
    """
    pass  # 空的 Resolvent 类定义，用于计算解决多项式 R(Y)
    This resolvent has three roots, which are the conjugates of ``F`` under the
    three permutations in ``s``:

# 这个解决多项式有三个根，它们是 ``F`` 在 ``s`` 的三个置换下的共轭根。

    >>> R.root_lambdas[0](*X)
    X0*X2 + X1*X3

# 使用解决多项式对象 R 的第一个根函数，对变量 X 进行替换并计算结果。

    >>> R.root_lambdas[1](*X)
    X0*X3 + X1*X2

# 使用解决多项式对象 R 的第二个根函数，对变量 X 进行替换并计算结果。

    >>> R.root_lambdas[2](*X)
    X0*X1 + X2*X3

# 使用解决多项式对象 R 的第三个根函数，对变量 X 进行替换并计算结果。

    Resolvents are useful for computing Galois groups. Given a polynomial $T$
    of degree $n$, we will use a resolvent $R$ where $Gal(T) \leq G \leq S_n$.
    We will then want to substitute the roots of $T$ for the variables $X_i$
    in $R$, and study things like the discriminant of $R$, and the way $R$
    factors over $\mathbb{Q}$.

# 解决多项式对计算 Galois 群很有用。给定一个次数为 $n$ 的多项式 $T$，我们会使用一个解决多项式 $R$，其中 $Gal(T) \leq G \leq S_n$。
# 我们将使用 $T$ 的根替换 $R$ 中的变量 $X_i$，并研究 $R$ 的判别式以及 $R$ 在有理数域上的分解方式。

    From the symmetry in $R$'s construction, and since $Gal(T) \leq G$, we know
    from Galois theory that the coefficients of $R$ must lie in $\mathbb{Z}$.
    This allows us to compute the coefficients of $R$ by approximating the
    roots of $T$ to sufficient precision, plugging these values in for the
    variables $X_i$ in the coefficient expressions of $R$, and then simply
    rounding to the nearest integer.

# 由于 $R$ 的构造对称性，并且由于 $Gal(T) \leq G$，根据 Galois 理论，$R$ 的系数必须属于 $\mathbb{Z}$。
# 这使得我们可以通过对 $T$ 的根进行足够精度的近似来计算 $R$ 的系数，将这些值代入 $R$ 的系数表达式中的变量 $X_i$ 中，然后简单地四舍五入到最近的整数。

    In order to determine a sufficient precision for the roots of $T$, this
    ``Resolvent`` class imposes certain requirements on the form ``F``. It
    could be possible to design a different ``Resolvent`` class, that made
    different precision estimates, and different assumptions about ``F``.

# 为了确定 $T$ 的根的足够精度，这个 ``Resolvent`` 类对形式 ``F`` 施加了一些要求。
# 设计一个不同的 ``Resolvent`` 类可能会做出不同的精度估计，并对 ``F`` 做出不同的假设。

    ``F`` must be homogeneous, and all terms must have unit coefficient.
    Furthermore, if $r$ is the number of terms in ``F``, and $t$ the total
    degree, and if $m$ is the number of conjugates of ``F``, i.e. the number
    of permutations in ``s``, then we require that $m < r 2^t$. Again, it is
    not impossible to work with forms ``F`` that violate these assumptions, but
    this ``Resolvent`` class requires them.

# ``F`` 必须是齐次的，且所有项的系数必须为单位系数。
# 此外，如果 $r$ 是 ``F`` 中的项数，$t$ 是总次数，$m$ 是 ``F`` 的共轭个数（即 ``s`` 中的置换数），则要求 $m < r 2^t$。
# 虽然也可以处理违反这些假设的 ``F`` 形式，但这个 ``Resolvent`` 类要求它们。

    Since determining the integer coefficients of the resolvent for a given
    polynomial $T$ is one of the main problems this class solves, we take some
    time to explain the precision bounds it uses.

# 由于确定给定多项式 $T$ 的解决多项式的整数系数是这个类解决的主要问题之一，我们花一些时间解释它使用的精度界限。

    The general problem is:
    Given a multivariate polynomial $P \in \mathbb{Z}[X_1, \ldots, X_n]$, and a
    bound $M \in \mathbb{R}_+$, compute an $\varepsilon > 0$ such that for any
    complex numbers $a_1, \ldots, a_n$ with $|a_i| < M$, if the $a_i$ are
    approximated to within an accuracy of $\varepsilon$ by $b_i$, that is,
    $|a_i - b_i| < \varepsilon$ for $i = 1, \ldots, n$, then
    $|P(a_1, \ldots, a_n) - P(b_1, \ldots, b_n)| < 1/2$. In other words, if it
    is known that $P(a_1, \ldots, a_n) = c$ for some $c \in \mathbb{Z}$, then
    $P(b_1, \ldots, b_n)$ can be rounded to the nearest integer in order to
    determine $c$.

# 一般问题是：
# 给定一个多变量多项式 $P \in \mathbb{Z}[X_1, \ldots, X_n]$，以及一个上界 $M \in \mathbb{R}_+$，计算一个 $\varepsilon > 0$，
# 对于任意复数 $a_1, \ldots, a_n$ 满足 $|a_i| < M$，如果 $a_i$ 被精度 $\varepsilon$ 近似为 $b_i$，即对于 $i = 1, \ldots, n$，
# $|a_i - b_i| < \varepsilon$，那么 $|P(a_1, \ldots, a_n) - P(b_1, \ldots, b_n)| < 1/2$。
# 换句话说，如果已知 $P(a_1, \ldots, a_n) = c$，其中 $c \in \mathbb{Z}$，那么 $P(b_1, \ldots, b_n)$ 可以四舍五入到最近的整数来确定 $c$。

    To derive our error bound, consider the monomial $xyz$. Defining
    $d_i = b_i - a_i$, our error is
    $|(a_1 + d_1)(a_2 + d_2)(a_3 + d_3) - a_1 a_2 a_3|$, which is bounded
    above by $|(M + \varepsilon)^3 - M^3|$. Passing to a general monomial of
    total degree $t$, this expression is bounded by
    $M^{t-1}\varepsilon(t + 2^t\varepsilon/M)$ provided $\varepsilon < M$,

# 为了推导我们的误差界限，考虑单项式 $xyz$。定义 $d_i = b_i - a_i$，我们的误差是 $|(a_1 + d_1)(a_2 + d_2)(a_3 + d_3) - a_1 a_2 a_3|$，
# 其上界为 $|(M + \varepsilon)^3 - M^3|$。对于总次数为 $t$ 的一般单项式，假设 $\varepsilon < M$，该表达式由
    # 在这段注释中，作者描述了如何估计多项式 $R(Y)$ 的系数的误差上界。多项式 $R(Y)$ 是一个具有 $r$ 项且总次数为 $t$ 的齐次多项式，其系数均为 $\pm 1$。作者使用递归的方式分析了每个系数的误差上界，并推导出了一个通用的上界公式。对于多项式 $R(Y)$ 的每个系数 $j$，其误差上界为 $\binom{m}{j} r^j (jt + 1) M^{jt - 1} \varepsilon$，其中 $\varepsilon$ 是误差限，$m$ 是共轭多项式的数量，$M$ 是一个足够大的常数，$t$ 是多项式 $F$ 的次数。作者指出，误差上界在 $j$ 增大到 $m/2$ 时达到最大值，因为此时组合数 $\binom{m}{j}$ 开始减小，而其他因子继续增加。因此，对于多项式 $R(Y)$ 所有系数的评估，应选择这些误差上界中的最大值，即 $r^m (mt + 1) M^{mt - 1} \varepsilon$。
    # 论文引用了 Cohen 的著作，特别是关于解决方案的定义。
    def __init__(self, F, X, s):
        r"""
        Parameters
        ==========

        F : :py:class:`~.Expr`
            polynomial in the symbols in *X*
        X : list of :py:class:`~.Symbol`
            list of symbols representing variables
        s : list of :py:class:`~.Permutation`
            representing the cosets of the stabilizer of *F* in
            some subgroup $G$ of $S_n$, where $n$ is the length of *X*.
        """
        self.F = F
        self.X = X
        self.s = s

        # Number of conjugates:
        self.m = len(s)  # 计算稳定子群中 *F* 的共轭数量
        # Total degree of F (computed below):
        self.t = None  # *F* 的总次数，初始为 None
        # Number of terms in F (computed below):
        self.r = 0  # *F* 的项数，初始为 0

        for monom, coeff in Poly(F).terms():
            if abs(coeff) != 1:
                raise ResolventException('Resolvent class expects forms with unit coeffs')
            t = sum(monom)  # 计算单项式的总次数
            if t != self.t and self.t is not None:
                raise ResolventException('Resolvent class expects homogeneous forms')
            self.t = t  # 更新 *F* 的总次数
            self.r += 1  # 计算 *F* 的项数

        m, t, r = self.m, self.t, self.r
        if not m < r * 2**t:
            raise ResolventException('Resolvent class expects m < r*2^t')
        M = symbols('M')
        # Precision sufficient for computing the coeffs of the resolvent:
        self.coeff_prec_func = Poly(r**m*(m*t + 1)*M**(m*t - 1))
        # Precision sufficient for checking whether any of the roots of the
        # resolvent are integers:
        self.root_prec_func = Poly(r*(t + 1)*M**(t - 1))

        # The conjugates of F are the roots of the resolvent.
        # For evaluating these to required numerical precisions, we need
        # lambdified versions.
        # Note: for a given permutation sigma, the conjugate (sigma F) is
        # equivalent to lambda [sigma^(-1) X]: F.
        self.root_lambdas = [
            lambdify((~s[j])(X), F)
            for j in range(self.m)
        ]  # 创建计算每个共轭的 lambda 函数列表

        # For evaluating the coeffs, we'll also need lambdified versions of
        # the elementary symmetric functions for degree m.
        Y = symbols('Y')
        R = symbols(' '.join(f'R{i}' for i in range(m)))
        f = 1
        for r in R:
            f *= (Y - r)
        C = Poly(f, Y).coeffs()
        self.esf_lambdas = [lambdify(R, c) for c in C]  # 创建计算 m 阶基本对称函数的 lambda 函数列表
    def get_prec(self, M, target='coeffs'):
        r"""
        For a given upper bound *M* on the magnitude of the complex numbers to
        be plugged in for this resolvent's symbols, compute a sufficient
        precision for evaluating those complex numbers, such that the
        coefficients, or the integer roots, of the resolvent can be determined.

        Parameters
        ==========

        M : real number
            Upper bound on magnitude of the complex numbers to be plugged in.

        target : str, 'coeffs' or 'roots', default='coeffs'
            Name the task for which a sufficient precision is desired.
            This is either determining the coefficients of the resolvent
            ('coeffs') or determining its possible integer roots ('roots').
            The latter may require significantly lower precision.

        Returns
        =======

        int $m$
            such that $2^{-m}$ is a sufficient upper bound on the
            error in approximating the complex numbers to be plugged in.

        """
        # As explained in the docstring for this class, our precision estimates
        # require that M be at least 2.
        M = max(M, 2)  # 确保 M 至少为 2，以满足精度要求
        # 根据目标类型选择适当的函数来计算所需的精度
        f = self.coeff_prec_func if target == 'coeffs' else self.root_prec_func
        # 调用 evalf 函数计算所需的精度 r
        r, _, _, _ = evalf(2*f(M), 1, {})
        # 返回结果，fastlog(r) + 1 表示计算得到的精度 m
        return fastlog(r) + 1
    def approximate_roots_of_poly(self, T, target='coeffs'):
        """
        Approximate the roots of a given polynomial *T* to sufficient precision
        in order to evaluate this resolvent's coefficients, or determine
        whether the resolvent has an integer root.

        Parameters
        ==========

        T : :py:class:`~.Poly`
            Polynomial object representing the polynomial whose roots are to be approximated.

        target : str, 'coeffs' or 'roots', default='coeffs'
            Set the approximation precision to be sufficient for the desired
            task, which is either determining the coefficients of the resolvent
            ('coeffs') or determining its possible integer roots ('roots').
            The latter may require significantly lower precision.

        Returns
        =======

        list of elements of :ref:`CC`
            List of approximate roots of the polynomial T.

        """
        ctx = MPContext()
        # Extract leading coefficient and preprocess polynomial T to handle roots efficiently
        coeff, T = preprocess_roots(T)
        coeff = ctx.mpf(str(coeff))  # Convert coefficient to arbitrary-precision floating-point number

        scaled_roots = T.all_roots(radicals=False)  # Compute all roots of polynomial T

        # Compute initial rough approximations of roots scaled by the leading coefficient
        approx0 = [coeff * quad_to_mpmath(_evalf_with_bounded_error(r, m=0)) for r in scaled_roots]
        # Compute an upper bound on the magnitude of roots considering initial approximation errors
        M = max(abs(b) for b in approx0) + 1
        # Determine the required precision for root approximation based on the target (coefficients or roots)
        m = self.get_prec(M, target=target)
        n = fastlog(M._mpf_) + 1
        p = m + n + 1
        ctx.prec = p  # Set the precision context for further computations
        d = prec_to_dps(p)  # Convert precision to decimal places

        # Refine the approximations of roots using the determined precision context
        approx1 = [r.eval_approx(d, return_mpmath=True) for r in scaled_roots]
        approx1 = [coeff * ctx.mpc(r) for r in approx1]  # Scale refined approximations by the coefficient

        return approx1

    @staticmethod
    def round_mpf(a):
        if isinstance(a, int):
            return a  # Return input directly if it's already an integer
        # Use mpmath's nint function to round the arbitrary-precision float 'a' to the nearest integer
        # Convert the rounded result to SymPy's integer type ZZ (Integer)
        return ZZ(int(a.context.nint(a)))
    def round_roots_to_integers_for_poly(self, T):
        """
        对于给定的多项式 *T*，将其共轭根四舍五入为最接近的整数。

        Explanation
        ===========

        这个方法返回的整数不一定是共轭根的根；然而，如果共轭根有任何整数根（对于给定的多项式 *T*），那么它们必定在这些整数中。

        如果也需要解析多项式的系数，则不应使用此方法。而是使用 ``eval_for_poly`` 方法。此方法可能比 ``eval_for_poly`` 更快。

        Parameters
        ==========

        T : :py:class:`~.Poly`
            给定的多项式对象 *T*。

        Returns
        =======

        dict
            键是 ``self.s`` 中这些排列的索引，使得对应的根四舍五入为有理整数。

            值是 :ref:`ZZ`。


        """
        # 使用多项式 *T* 的近似根作为目标，获取多项式的根
        approx_roots_of_T = self.approximate_roots_of_poly(T, target='roots')
        # 使用根 lambda 函数列表，计算 self 的近似根
        approx_roots_of_self = [r(*approx_roots_of_T) for r in self.root_lambdas]
        # 返回一个字典，其中键是索引 i，值是将 r.real 四舍五入为整数后的结果，前提是 r.imag 四舍五入为 0
        return {
            i: self.round_mpf(r.real)
            for i, r in enumerate(approx_roots_of_self)
            if self.round_mpf(r.imag) == 0
        }
    def eval_for_poly(self, T, find_integer_root=False):
        r"""
        Compute the integer values of the coefficients of this resolvent, when
        plugging in the roots of a given polynomial.

        Parameters
        ==========

        T : :py:class:`~.Poly`
            The polynomial object whose roots are used for evaluation.

        find_integer_root : ``bool``, default ``False``
            If ``True``, then also determine whether the resolvent has an
            integer root, and return the first one found, along with its
            index, i.e. the index of the permutation ``self.s[i]`` it
            corresponds to.

        Returns
        =======

        Tuple ``(R, a, i)``

            ``R`` is this resolvent as a dense univariate polynomial over
            :ref:`ZZ`, i.e. a list of :ref:`ZZ`.

            If *find_integer_root* was ``True``, then ``a`` and ``i`` are the
            first integer root found, and its index, if one exists.
            Otherwise ``a`` and ``i`` are both ``None``.

        """
        approx_roots_of_T = self.approximate_roots_of_poly(T, target='coeffs')
        # Evaluate the root lambdas of self at approximate roots of T
        approx_roots_of_self = [r(*approx_roots_of_T) for r in self.root_lambdas]
        # Evaluate the esf lambdas of self at the evaluated roots of self
        approx_coeffs_of_self = [c(*approx_roots_of_self) for c in self.esf_lambdas]

        R = []
        # Iterate over the evaluated coefficients of self
        for c in approx_coeffs_of_self:
            if self.round_mpf(c.imag) != 0:
                # Raise an exception if the imaginary part of c is non-zero
                raise ResolventException(f"Got non-integer coeff for resolvent: {c}")
            # Round the real part of c to handle floating-point precision
            R.append(self.round_mpf(c.real))

        a0, i0 = None, None

        if find_integer_root:
            # Iterate over the indices and roots of self
            for i, r in enumerate(approx_roots_of_self):
                if self.round_mpf(r.imag) != 0:
                    continue  # Skip roots with non-zero imaginary part
                a = self.round_mpf(r.real)
                # Check if a is an integer root of the polynomial R
                if not dup_eval(R, a, ZZ):
                    a0, i0 = a, i
                    break  # Exit loop if first integer root is found

        return R, a0, i0
# 定义函数，用于对文本进行按行包装，以便在达到指定宽度后换行。
def wrap(text, width=80):
    """Line wrap a polynomial expression. """
    out = ''
    col = 0
    # 遍历文本中的每个字符
    for c in text:
        # 如果字符是空格且当前列数超过指定宽度，则将字符替换为换行符，列数重置为0
        if c == ' ' and col > width:
            c, col = '\n', 0
        else:
            col += 1
        out += c
    return out


# 定义函数，生成符号 s1, s2, ..., sn 用于表示元素对称多项式。
def s_vars(n):
    """Form the symbols s1, s2, ..., sn to stand for elem. symm. polys. """
    return symbols([f's{i + 1}' for i in range(n)])


# 定义函数，计算与给定多项式相关的解析式的系数作为多项式函数。
def sparse_symmetrize_resolvent_coeffs(F, X, s, verbose=False):
    """
    Compute the coefficients of a resolvent as functions of the coefficients of
    the associated polynomial.

    F must be a sparse polynomial.
    """
    import time, sys
    # 使用每个排列生成与 X 变量相关的多变量形式的解析式的根。
    root_forms = [
        F.compose(list(zip(X, sigma(X))))
        for sigma in s
    ]

    # 计算解析式的系数（除了首项系数为 1）作为 X 变量的对称形式。
    Y = [Dummy(f'Y{i}') for i in range(len(s))]
    coeff_forms = []
    # 遍历每个阶数 i，生成对应的对称多项式。
    for i in range(1, len(s) + 1):
        if verbose:
            print('----')
            print(f'Computing symmetric poly of degree {i}...')
            sys.stdout.flush()
        t0 = time.time()
        G = symmetric_poly(i, *Y)
        t1 = time.time()
        if verbose:
            print(f'took {t1 - t0} seconds')
            print('lambdifying...')
            sys.stdout.flush()
        t0 = time.time()
        C = lambdify(Y, (-1)**i*G)
        t1 = time.time()
        if verbose:
            print(f'took {t1 - t0} seconds')
            sys.stdout.flush()
        coeff_forms.append(C)

    coeffs = []
    # 对每个系数形式 f，将解析式的根形式作为参数进行计算。
    for i, f in enumerate(coeff_forms):
        if verbose:
            print('----')
            print(f'Plugging root forms into elem symm poly {i+1}...')
            sys.stdout.flush()
        t0 = time.time()
        g = f(*root_forms)
        t1 = time.time()
        coeffs.append(g)
        if verbose:
            print(f'took {t1 - t0} seconds')
            sys.stdout.flush()

    # 现在对这些系数进行对称化处理。这意味着将它们重新转换为关于 X 变量的元对称多项式。
    symmetrized = []
    symmetrization_times = []
    ss = s_vars(len(X))
    # 对每个系数 A 进行处理，确保对称性，并记录处理时间。
    for i, A in list(enumerate(coeffs)):
        if verbose:
            print('-----')
            print(f'Coeff {i+1}...')
            sys.stdout.flush()
        t0 = time.time()
        B, rem, _ = A.symmetrize()
        t1 = time.time()
        # 如果存在非零余项，引发异常。
        if rem != 0:
            msg = f"Got nonzero remainder {rem} for resolvent (F, X, s) = ({F}, {X}, {s})"
            raise ResolventException(msg)
        # 将 B 转换为表达式字符串，基于 ss 变量。
        B_str = str(B.as_expr(*ss))
        symmetrized.append(B_str)
        symmetrization_times.append(t1 - t0)
        if verbose:
            print(wrap(B_str))
            print(f'took {t1 - t0} seconds')
            sys.stdout.flush()

    return symmetrized, symmetrization_times


# 定义函数，用于为度数在 4 到 6 之间的多项式 T 定义所有的解析式。
def define_resolvents():
    """Define all the resolvents for polys T of degree 4 through 6. """
    from sympy.combinatorics.galois import PGL2F5
    # 导入符号计算模块中的置换类 Permutation
    from sympy.combinatorics.permutations import Permutation

    # 创建带有变量 X0, X1, X2, X3 的多项式环 R4，并使用 lex 排序
    R4, X4 = xring("X0,X1,X2,X3", ZZ, lex)
    X = X4

    # 定义度数为 4 的一元多项式 F40 作为一个解析式
    F40 = X[0]*X[1]**2 + X[1]*X[2]**2 + X[2]*X[3]**2 + X[3]*X[0]**2
    # 定义用于 F40 的置换列表 s40
    s40 = [
        Permutation(3),            # 置换 (0 1 2)
        Permutation(3)(0, 1),      # 置换 (0 1)
        Permutation(3)(0, 2),      # 置换 (0 2)
        Permutation(3)(0, 3),      # 置换 (0 3)
        Permutation(3)(1, 2),      # 置换 (1 2)
        Permutation(3)(2, 3),      # 置换 (2 3)
    ]

    # 定义度数为 4 的另一个一元多项式 F41
    F41 = X[0]*X[2] + X[1]*X[3]
    # 定义用于 F41 的置换列表 s41
    s41 = [
        Permutation(3),            # 置换 (0 1 2)
        Permutation(3)(0, 1),      # 置换 (0 1)
        Permutation(3)(0, 3)       # 置换 (0 3)
    ]

    # 创建带有变量 X0, X1, X2, X3, X4 的多项式环 R5，并使用 lex 排序
    R5, X5 = xring("X0,X1,X2,X3,X4", ZZ, lex)
    X = X5

    # 定义度数为 5 的一元多项式 F51
    F51 = (  X[0]**2*(X[1]*X[4] + X[2]*X[3])
           + X[1]**2*(X[2]*X[0] + X[3]*X[4])
           + X[2]**2*(X[3]*X[1] + X[4]*X[0])
           + X[3]**2*(X[4]*X[2] + X[0]*X[1])
           + X[4]**2*(X[0]*X[3] + X[1]*X[2]))
    # 定义用于 F51 的置换列表 s51
    s51 = [
        Permutation(4),            # 置换 (0 1 2 3)
        Permutation(4)(0, 1),      # 置换 (0 1)
        Permutation(4)(0, 2),      # 置换 (0 2)
        Permutation(4)(0, 3),      # 置换 (0 3)
        Permutation(4)(0, 4),      # 置换 (0 4)
        Permutation(4)(1, 4)       # 置换 (1 4)
    ]

    # 创建带有变量 X0, X1, X2, X3, X4, X5 的多项式环 R6，并使用 lex 排序
    R6, X6 = xring("X0,X1,X2,X3,X4,X5", ZZ, lex)
    X = X6

    # 定义度数为 6 的一元多项式 F61
    H = PGL2F5()
    term0 = X[0]**2*X[5]**2*(X[1]*X[4] + X[2]*X[3])
    # 使用 H 中元素的置换 s(X) 组成 terms 集合，然后求和得到 F61
    terms = {term0.compose(list(zip(X, s(X)))) for s in H.elements}
    F61 = sum(terms)
    # 定义用于 F61 的置换列表 s61
    s61 = [Permutation(5)] + [Permutation(5)(0, n) for n in range(1, 6)]

    # 定义度数为 6 的另一个一元多项式 F62
    F62 = X[0]*X[1]*X[2] + X[3]*X[4]*X[5]
    # 定义用于 F62 的置换列表 s62
    s62 = [Permutation(5)] + [
        Permutation(5)(i, j + 3) for i in range(3) for j in range(3)
    ]

    # 返回包含所有定义的多项式和置换的字典
    return {
        (4, 0): (F40, X4, s40),    # 返回度数为 4 的多项式 F40、环 X4 和置换 s40 的元组
        (4, 1): (F41, X4, s41),    # 返回度数为 4 的多项式 F41、环 X4 和置换 s41 的元组
        (5, 1): (F51, X5, s51),    # 返回度数为 5 的多项式 F51、环 X5 和置换 s51 的元组
        (6, 1): (F61, X6, s61),    # 返回度数为 6 的多项式 F61、环 X6 和置换 s61 的元组
        (6, 2): (F62, X6, s62),    # 返回度数为 6 的多项式 F62、环 X6 和置换 s62 的元组
    }
def generate_lambda_lookup(verbose=False, trial_run=False):
    """
    Generate the whole lookup table of coeff lambdas, for all resolvents.
    生成所有解析函数的系数lambda查找表。

    Parameters
    ==========

    verbose : bool, optional
        Whether to print verbose output. Defaults to False.
        是否输出详细信息，默认为False。

    trial_run : bool, optional
        Whether to run a trial (generating only a subset of data). Defaults to False.
        是否运行试验（仅生成部分数据）。默认为False。
    """

    # 定义解析函数的工作集合
    jobs = define_resolvents()

    # 存储解析函数系数lambda的列表
    lambda_lists = {}

    # 总计算时间
    total_time = 0

    # 记录(6, 1)解析函数的时间
    time_for_61 = 0

    # 记录(6, 1)解析函数最后一个的时间
    time_for_61_last = 0

    # 遍历每个解析函数任务
    for k, (F, X, s) in jobs.items():
        # 对解析函数系数进行稀疏对称化处理，返回处理后的结果及时间信息
        symmetrized, times = sparse_symmetrize_resolvent_coeffs(F, X, s, verbose=verbose)

        # 统计总的处理时间
        total_time += sum(times)

        # 如果是解析函数(6, 1)
        if k == (6, 1):
            time_for_61 = sum(times)
            time_for_61_last = times[-1]

        # 生成系数lambda的变量
        sv = s_vars(len(X))
        head = f'lambda {", ".join(str(v) for v in sv)}:'

        # 构建lambda列表的字符串表示
        lambda_lists[k] = ',\n        '.join([
            f'{head} ({wrap(f)})'
            for f in symmetrized
        ])

        # 如果是试验运行模式，则中断循环
        if trial_run:
            break

    # 构建查找表的字符串表示
    table = (
         "# This table was generated by a call to\n"
         "# `sympy.polys.numberfields.galois_resolvents.generate_lambda_lookup()`.\n"
        f"# The entire job took {total_time:.2f}s.\n"
        f"# Of this, Case (6, 1) took {time_for_61:.2f}s.\n"
        f"# The final polynomial of Case (6, 1) alone took {time_for_61_last:.2f}s.\n"
         "resolvent_coeff_lambdas = {\n")

    # 将lambda列表添加到查找表字符串中
    for k, L in lambda_lists.items():
        table += f"    {k}: [\n"
        table +=  "        " + L + '\n'
        table +=  "    ],\n"
    table += "}\n"
    return table


def get_resolvent_by_lookup(T, number):
    """
    Use the lookup table, to return a resolvent (as dup) for a given
    polynomial *T*.

    Parameters
    ==========

    T : Poly
        The polynomial whose resolvent is needed
        需要解析的多项式。

    number : int
        For some degrees, there are multiple resolvents.
        Use this to indicate which one you want.
        对于某些阶数，可能存在多个解析函数。使用此参数指定需要的解析函数。

    Returns
    =======

    dup
    返回解析函数（作为dup）。
    """

    # 获取多项式的阶数
    degree = T.degree()

    # 从查找表中获取对应阶数和编号的解析函数系数列表
    L = resolvent_coeff_lambdas[(degree, number)]

    # 获取多项式的系数列表（去除首项1）
    T_coeffs = T.rep.to_list()[1:]

    # 构造解析函数结果列表
    return [ZZ(1)] + [c(*T_coeffs) for c in L]


# Use
#   (.venv) $ python -m sympy.polys.numberfields.galois_resolvents
# to reproduce the table found in resolvent_lookup.py
if __name__ == "__main__":
    import sys
    verbose = '-v' in sys.argv[1:]
    trial_run = '-t' in sys.argv[1:]
    table = generate_lambda_lookup(verbose=verbose, trial_run=trial_run)
    print(table)
```