# `D:\src\scipysrc\sympy\sympy\polys\subresultants_qq_zz.py`

```
"""
This module contains functions for the computation
of Euclidean, (generalized) Sturmian, (modified) subresultant
polynomial remainder sequences (prs's) of two polynomials;
included are also three functions for the computation of the
resultant of two polynomials.

Except for the function res_z(), which computes the resultant
of two polynomials, the pseudo-remainder function prem()
of sympy is _not_ used by any of the functions in the module.

Instead of prem() we use the function

rem_z().

Included is also the function quo_z().

An explanation of why we avoid prem() can be found in the
references stated in the docstring of rem_z().

1. Theoretical background:
==========================
Consider the polynomials f, g in Z[x] of degrees deg(f) = n and
deg(g) = m with n >= m.

Definition 1:
=============
The sign sequence of a polynomial remainder sequence (prs) is the
sequence of signs of the leading coefficients of its polynomials.

Sign sequences can be computed with the function:

sign_seq(poly_seq, x)

Definition 2:
=============
A polynomial remainder sequence (prs) is called complete if the
degree difference between any two consecutive polynomials is 1;
otherwise, it called incomplete.

It is understood that f, g belong to the sequences mentioned in
the two definitions above.

1A. Euclidean and subresultant prs's:
=====================================
The subresultant prs of f, g is a sequence of polynomials in Z[x]
analogous to the Euclidean prs, the sequence obtained by applying
on f, g Euclid's algorithm for polynomial greatest common divisors
(gcd) in Q[x].

The subresultant prs differs from the Euclidean prs in that the
coefficients of each polynomial in the former sequence are determinants
--- also referred to as subresultants --- of appropriately selected
sub-matrices of sylvester1(f, g, x), Sylvester's matrix of 1840 of
dimensions (n + m) * (n + m).

Recall that the determinant of sylvester1(f, g, x) itself is
called the resultant of f, g and serves as a criterion of whether
the two polynomials have common roots or not.

In SymPy the resultant is computed with the function
resultant(f, g, x). This function does _not_ evaluate the
determinant of sylvester(f, g, x, 1); instead, it returns
the last member of the subresultant prs of f, g, multiplied
(if needed) by an appropriate power of -1; see the caveat below.

In this module we use three functions to compute the
resultant of f, g:
a) res(f, g, x) computes the resultant by evaluating
the determinant of sylvester(f, g, x, 1);
b) res_q(f, g, x) computes the resultant recursively, by
performing polynomial divisions in Q[x] with the function rem();
c) res_z(f, g, x) computes the resultant recursively, by
performing polynomial divisions in Z[x] with the function prem().

Caveat: If Df = degree(f, x) and Dg = degree(g, x), then:

resultant(f, g, x) = (-1)**(Df*Dg) * resultant(g, f, x).

For complete prs's the sign sequence of the Euclidean prs of f, g
"""

# Importing necessary functions from sympy
from sympy import Matrix, S, diff, zeros

# Helper function to compute polynomial division
def prem(f, g, x):
    """
    Compute the pseudo-remainder of f and g in Z[x] using
    the function rem_z().
    """
    pass

# Helper function to compute resultant recursively in Q[x]
def res_q(f, g, x):
    """
    Compute the resultant recursively by performing polynomial
    divisions in Q[x] with the function rem().
    """
    pass

# Helper function to compute resultant recursively in Z[x]
def res_z(f, g, x):
    """
    Compute the resultant recursively by performing polynomial
    divisions in Z[x] with the function prem().
    """
    pass
# 调用 sylvester 函数计算 Sylvester 矩阵的第一种方法
sylvester(f, g, x, method=1)
sylvester(f, g, x, method=2)
bezout(f, g, x, method='prs')

# The following identity holds:
# 
# bezout(f, g, x, method='prs') =
# backward_eye(deg(f))*bezout(f, g, x, method='bz')*backward_eye(deg(f))

# 2B. Subresultant and modified subresultant prs's by
# ===================================================
# determinant evaluations:
# =======================
# We use the Sylvester matrices of 1840 and 1853 to
# compute, respectively, subresultant and modified
# subresultant polynomial remainder sequences. However,
# for large matrices this approach takes a lot of time.
# 
# Instead of utilizing the Sylvester matrices, we can
# employ the Bezout matrix which is of smaller dimensions.
# 
# subresultants_sylv(f, g, x)
# modified_subresultants_sylv(f, g, x)
# subresultants_bezout(f, g, x)
# modified_subresultants_bezout(f, g, x)

# 2C. Subresultant prs's by ONE determinant evaluation:
# =====================================================
# All three functions in this section evaluate one determinant
# per remainder polynomial; this is the determinant of an
# appropriately selected sub-matrix of sylvester1(f, g, x),
# Sylvester's matrix of 1840.
# 
# To compute the remainder polynomials the function
# subresultants_rem(f, g, x) employs rem(f, g, x).
# By contrast, the other two functions implement Van Vleck's ideas
# of 1900 and compute the remainder polynomials by triangularizing
# sylvester2(f, g, x), Sylvester's matrix of 1853.
# 
# subresultants_rem(f, g, x)
# subresultants_vv(f, g, x)
# subresultants_vv_2(f, g, x).

# 2E. Euclidean, Sturmian prs's in Q[x]:
# ======================================
# euclid_q(f, g, x)
# sturm_q(f, g, x)

# 2F. Euclidean, Sturmian and (modified) subresultant prs's P-G:
# ==============================================================
# All functions in this section are based on the Pell-Gordon (P-G)
# theorem of 1917.
# Computations are done in Q[x], employing the function rem(f, g, x)
# for the computation of the remainder polynomials.
# 
# euclid_pg(f, g, x)
# sturm_pg(f, g, x)
# subresultants_pg(f, g, x)
# modified_subresultants_pg(f, g, x)

# 2G. Euclidean, Sturmian and (modified) subresultant prs's A-M-V:
# =================================================================
# All functions in this section are based on the Akritas-Malaschonok-
# Vigklas (A-M-V) theorem of 2015.
# Computations are done in Z[x], employing the function rem_z(f, g, x)
# for the computation of the remainder polynomials.
# 
# euclid_amv(f, g, x)
# sturm_amv(f, g, x)
# subresultants_amv(f, g, x)
# modified_subresultants_amv(f, g, x)

# 2Ga. Exception:
# ===============
# subresultants_amv_q(f, g, x)
# 
# This function employs rem(f, g, x) for the computation of
# the remainder polynomials, despite the fact that it implements
# the A-M-V Theorem.
# 
# It is included in our module in order to show that theorems P-G
# and A-M-V can be implemented utilizing either the function
# rem(f, g, x) or the function rem_z(f, g, x).
# 
# For clearly historical reasons --- since the Collins-Brown-Traub
# coefficients-reduction factor beta_i was not available in 1917 ---
    '''
    We have implemented the Pell-Gordon theorem with the function
    rem(f, g, x) and the A-M-V Theorem with the function rem_z(f, g, x).

    2H. Resultants:
    ===============
    res(f, g, x)
    res_q(f, g, x)
    res_z(f, g, x)
    '''

    # 导入所需的库和模块
    from sympy.concrete.summations import summation
    from sympy.core.function import expand
    from sympy.core.numbers import nan
    from sympy.core.singleton import S
    from sympy.core.symbol import Dummy as var
    from sympy.functions.elementary.complexes import Abs, sign
    from sympy.functions.elementary.integers import floor
    from sympy.matrices.dense import eye, Matrix, zeros
    from sympy.printing.pretty.pretty import pretty_print as pprint
    from sympy.simplify.simplify import simplify
    from sympy.polys.domains import QQ
    from sympy.polys.polytools import degree, LC, Poly, pquo, quo, prem, rem
    from sympy.polys.polyerrors import PolynomialError

    def sylvester(f, g, x, method = 1):
        '''
        The input polynomials f, g are in Z[x] or in Q[x]. Let m = degree(f, x),
        n = degree(g, x) and mx = max(m, n).

        a. If method = 1 (default), computes sylvester1, Sylvester's matrix of 1840
           of dimension (m + n) x (m + n). The determinants of properly chosen
           submatrices of this matrix (a.k.a. subresultants) can be
           used to compute the coefficients of the Euclidean PRS of f, g.

        b. If method = 2, computes sylvester2, Sylvester's matrix of 1853
           of dimension (2*mx) x (2*mx). The determinants of properly chosen
           submatrices of this matrix (a.k.a. "modified" subresultants) can be
           used to compute the coefficients of the Sturmian PRS of f, g.

        Applications of these Matrices can be found in the references below.
        Especially, for applications of sylvester2, see the first reference!!

        References
        ==========
        1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: "On a Theorem
        by Van Vleck Regarding Sturm Sequences. Serdica Journal of Computing,
        Vol. 7, No 4, 101-134, 2013.

        2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: "Sturm Sequences
        and Modified Subresultant Polynomial Remainder Sequences."
        Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.
        '''

        # 获取多项式 f, g 的次数
        m, n = degree(Poly(f, x), x), degree(Poly(g, x), x)

        # 特殊情况处理：
        # A:: 当 m = n < 0 时（即两个多项式都为0）
        if m == n and n < 0:
            return Matrix([])

        # B:: 当 m = n = 0 时（即两个多项式都是常数）
        if m == n and n == 0:
            return Matrix([])

        # C:: 当 m == 0 且 n < 0 或 m < 0 且 n == 0 时
        # （即一个多项式为常数，另一个为0）
        if m == 0 and n < 0:
            return Matrix([])
        elif m < 0 and n == 0:
            return Matrix([])

        # D:: 当 m >= 1 且 n < 0 或 m < 0 且 n >= 1 时
        # （即一个多项式的次数 >= 1，另一个为0）
        if m >= 1 and n < 0:
            return Matrix([0])
        elif m < 0 and n >= 1:
            return Matrix([0])

        # 获取多项式 f 的所有系数
        fp = Poly(f, x).all_coeffs()
    gp = Poly(g, x).all_coeffs()

    # 计算 Sylvester 矩阵的方法 1840 (默认; 又称 sylvester1)
    if method <= 1:
        # 初始化一个全零矩阵，维度为 m + n
        M = zeros(m + n)
        k = 0
        # 填充矩阵的前 n 行，使用 fp 的系数
        for i in range(n):
            j = k
            for coeff in fp:
                M[i, j] = coeff
                j = j + 1
            k = k + 1
        k = 0
        # 填充矩阵的后 m 行，使用 gp 的系数
        for i in range(n, m + n):
            j = k
            for coeff in gp:
                M[i, j] = coeff
                j = j + 1
            k = k + 1
        # 返回构建的 Sylvester 矩阵 M
        return M

    # 计算 Sylvester 矩阵的方法 1853 (又称 sylvester2)
    if method >= 2:
        # 调整 fp 和 gp 的长度，使其相等
        if len(fp) < len(gp):
            h = []
            for i in range(len(gp) - len(fp)):
                h.append(0)
            fp[ : 0] = h
        else:
            h = []
            for i in range(len(fp) - len(gp)):
                h.append(0)
            gp[ : 0] = h
        # 计算矩阵 M 的维度
        mx = max(m, n)
        dim = 2 * mx
        M = zeros(dim)
        k = 0
        # 填充矩阵的每一行，使用 fp 和 gp 的系数交替填充
        for i in range(mx):
            j = k
            for coeff in fp:
                M[2*i, j] = coeff
                j = j + 1
            j = k
            for coeff in gp:
                M[2*i + 1, j] = coeff
                j = j + 1
            k = k + 1
        # 返回构建的 Sylvester 矩阵 M
        return M


这些注释将帮助理解每行代码的具体作用，包括矩阵构建的细节和条件逻辑的处理方式。
# 确保多项式 f 和 g 非零
if f == 0 or g == 0:
    # 如果其中一个多项式为零，直接返回它们作为列表
    return [f, g]

# 计算多项式 f 和 g 的次数
n = degF = degree(f, x)
m = degG = degree(g, x)

# 确保多项式次数的正确性
if n == 0 and m == 0:
    # 如果两个多项式的次数都是零，返回它们作为列表
    return [f, g]
if n < m:
    # 如果 f 的次数小于 g 的次数，交换 f 和 g，同时交换它们的次数和度数
    n, m, degF, degG, f, g = m, n, degG, degF, g, f
if n > 0 and m == 0:
    # 如果 n 大于 0 且 m 等于 0，返回 f 和 g 作为列表
    return [f, g]

SR_L = [f, g]  # 子剩余列表，初始包含输入的 f 和 g

# 构造斯尔维斯特矩阵 sylvester(f, g, x, 1)
S = sylvester(f, g, x, 1)

# 选择适当的 S 的子矩阵并构造子剩余多项式
j = m - 1

while j > 0:
    Sp = S[:, :]  # S 的副本
    # 删除 g 的最后 j 行系数
    for ind in range(m + n - j, m + n):
        Sp.row_del(m + n - j)
    # 删除 f 的最后 j 行系数
    for ind in range(m - j, m):
        Sp.row_del(m - j)

    # 计算行列式并形成系数列表
    coeff_L, k, l = [], Sp.rows, 0
    while l <= j:
        coeff_L.append(Sp[:, 0:k].det())
        Sp.col_swap(k - 1, k + l)
        l += 1

    # 构造多项式并将其附加到 SR_L
    SR_L.append(Poly(coeff_L, x).as_expr())
    j -= 1

# 处理 j = 0 的情况，即最后一个子剩余多项式
SR_L.append(S.det())

# 调用 process_matrix_output 函数处理 SR_L 列表，并返回处理后的结果
return process_matrix_output(SR_L, x)
    """
    The input polynomials f, g are in Z[x] or in Q[x]. It is assumed
    that deg(f) >= deg(g).

    Computes the modified subresultant polynomial remainder sequence (prs)
    of f, g by evaluating determinants of appropriately selected
    submatrices of sylvester(f, g, x, 2). The dimensions of the
    latter are (2*deg(f)) x (2*deg(f)).

    Each coefficient is computed by evaluating the determinant of the
    corresponding submatrix of sylvester(f, g, x, 2).

    If the modified subresultant prs is complete, then the output coincides
    with the Sturmian sequence of the polynomials f, g.

    References:
    ===========
    1. A. G. Akritas,G.I. Malaschonok and P.S. Vigklas:
    Sturm Sequences and Modified Subresultant Polynomial Remainder
    Sequences. Serdica Journal of Computing, Vol. 8, No 1, 29--46, 2014.

    """

    # make sure neither f nor g is 0
    如果 f 或者 g 是 0，则直接返回它们的列表
    if f == 0 or g == 0:
        return [f, g]

    n = degF = degree(f, x)  # 计算 f 的次数
    m = degG = degree(g, x)  # 计算 g 的次数

    # make sure proper degrees
    # 确保 f 和 g 的次数符合预期
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, degF, degG, f, g = m, n, degG, degF, g, f
    if n > 0 and m == 0:
        return [f, g]

    SR_L = [f, g]      # modified subresultant list

    # form matrix sylvester(f, g, x, 2)
    # 构造 Sylvester 矩阵
    S = sylvester(f, g, x, 2)

    # pick appropriate submatrices of S
    # and form modified subresultant polys
    j = m - 1

    while j > 0:
        # delete last 2*j rows of pairs of coeffs of f, g
        # 复制 S 的前 2*n - 2*j 行，形成 Sp
        Sp = S[0:2*n - 2*j, :]

        # evaluate determinants and form coefficients list
        # 计算行列式并形成系数列表
        coeff_L, k, l = [], Sp.rows, 0
        while l <= j:
            coeff_L.append(Sp[:, 0:k].det())  # 计算行列式并添加到 coeff_L
            Sp.col_swap(k - 1, k + l)
            l += 1

        # form poly and append to SP_L
        # 构造多项式并添加到 SR_L
        SR_L.append(Poly(coeff_L, x).as_expr())
        j -= 1

    # j = 0
    # 将 S 的行列式添加到 SR_L
    SR_L.append(S.det())

    return process_matrix_output(SR_L, x)
# 定义函数 res，计算多项式 f 和 g 的结果式
def res(f, g, x):
    """
    The input polynomials f, g are in Z[x] or in Q[x].

    The output is the resultant of f, g computed by evaluating
    the determinant of the matrix sylvester(f, g, x, 1).

    References:
    ===========
    1. J. S. Cohen: Computer Algebra and Symbolic Computation
     - Mathematical Methods. A. K. Peters, 2003.

    """
    # 如果 f 或 g 为零多项式，则抛出异常
    if f == 0 or g == 0:
         raise PolynomialError("The resultant of %s and %s is not defined" % (f, g))
    else:
        # 返回 sylvester(f, g, x, 1) 的行列式，即多项式 f 和 g 的结果式
        return sylvester(f, g, x, 1).det()

# 定义函数 res_q，递归计算在有理数域 Q[x] 中多项式 f 和 g 的结果式
def res_q(f, g, x):
    """
    The input polynomials f, g are in Z[x] or in Q[x].

    The output is the resultant of f, g computed recursively
    by polynomial divisions in Q[x], using the function rem.
    See Cohen's book p. 281.

    References:
    ===========
    1. J. S. Cohen: Computer Algebra and Symbolic Computation
     - Mathematical Methods. A. K. Peters, 2003.
    """
    # 计算多项式 f 和 g 的次数
    m = degree(f, x)
    n = degree(g, x)
    # 如果 f 的次数小于 g 的次数，交换 f 和 g 并取符号反转的结果式
    if m < n:
        return (-1)**(m*n) * res_q(g, f, x)
    elif n == 0:  # 如果 g 是常数多项式
        return g**m
    else:
        # 使用 rem 函数计算 f 除以 g 的余数
        r = rem(f, g, x)
        if r == 0:
            return 0
        else:
            # 计算余数 r 的次数和 g 的首项系数 LC(g, x)
            s = degree(r, x)
            l = LC(g, x)
            # 返回结果式
            return (-1)**(m*n) * l**(m-s) * res_q(g, r, x)

# 定义函数 res_z，递归计算在整数环 Z[x] 中多项式 f 和 g 的结果式
def res_z(f, g, x):
    """
    The input polynomials f, g are in Z[x] or in Q[x].

    The output is the resultant of f, g computed recursively
    by polynomial divisions in Z[x], using the function prem().
    See Cohen's book p. 283.

    References:
    ===========
    1. J. S. Cohen: Computer Algebra and Symbolic Computation
     - Mathematical Methods. A. K. Peters, 2003.
    """
    # 计算多项式 f 和 g 的次数
    m = degree(f, x)
    n = degree(g, x)
    # 如果 f 的次数小于 g 的次数，交换 f 和 g 并取符号反转的结果式
    if m < n:
        return (-1)**(m*n) * res_z(g, f, x)
    elif n == 0:  # 如果 g 是常数多项式
        return g**m
    else:
        # 使用 prem 函数计算 f 除以 g 的余数
        r = prem(f, g, x)
        if r == 0:
            return 0
        else:
            # 计算 delta, w, s, l, k，并返回结果式
            delta = m - n + 1
            w = (-1)**(m*n) * res_z(g, r, x)
            s = degree(r, x)
            l = LC(g, x)
            k = delta * n - m + s
            return quo(w, l**k, x)

# 定义函数 sign_seq，返回多项式序列中每个多项式首项系数的符号序列
def sign_seq(poly_seq, x):
    """
    Given a sequence of polynomials poly_seq, it returns
    the sequence of signs of the leading coefficients of
    the polynomials in poly_seq.

    """
    # 遍历多项式序列，返回每个多项式的首项系数的符号
    return [sign(LC(poly_seq[i], x)) for i in range(len(poly_seq))]

# 定义函数 bezout，计算多项式 p 和 q 的 Bezout 矩阵或 Sylvester 矩阵
def bezout(p, q, x, method='bz'):
    """
    The input polynomials p, q are in Z[x] or in Q[x]. Let
    mx = max(degree(p, x), degree(q, x)).

    The default option bezout(p, q, x, method='bz') returns Bezout's
    symmetric matrix of p and q, of dimensions (mx) x (mx). The
    determinant of this matrix is equal to the determinant of sylvester2,
    Sylvester's matrix of 1853, whose dimensions are (2*mx) x (2*mx);
    however the subresultants of these two matrices may differ.

    The other option, bezout(p, q, x, 'prs'), is of interest to us
    in this module because it returns a matrix equivalent to sylvester2.

    """
    # 计算多项式 p 和 q 的最高次数
    mx = max(degree(p, x), degree(q, x))
    # 根据不同的方法计算并返回 Bezout 矩阵或 Sylvester 矩阵
    if method == 'bz':
        return bezout_bz(p, q, x, mx)
    elif method == 'prs':
        return bezout_prs(p, q, x, mx)
    # 获取多项式 p 和 q 的次数
    m, n = degree(Poly(p, x), x), degree(Poly(q, x), x)
    
    # 特殊情况处理:
    # A:: 如果 m = n < 0 (即两个多项式均为零)
    if m == n and n < 0:
        # 返回一个空的矩阵
        return Matrix([])
    
    # B:: 如果 m = n = 0 (即两个多项式均为常数)
    if m == n and n == 0:
        # 返回一个空的矩阵
        return Matrix([])
    
    # C:: 如果 m == 0 并且 n < 0 或者 m < 0 并且 n == 0
    # (即其中一个多项式为常数，另一个为零)
    if m == 0 and n < 0:
        # 返回一个空的矩阵
        return Matrix([])
    elif m < 0 and n == 0:
        # 返回一个空的矩阵
        return Matrix([])
    
    # D:: 如果 m >= 1 并且 n < 0 或者 m < 0 并且 n >= 1
    # (即其中一个多项式的次数 >= 1，另一个为零)
    if m >= 1 and n < 0:
        # 返回一个包含一个零的矩阵
        return Matrix([0])
    elif m < 0 and n >= 1:
        # 返回一个包含一个零的矩阵
        return Matrix([0])
    
    # 定义变量 y
    y = var('y')
    
    # 计算表达式 expr 在 x = y 处的值
    expr = p * q.subs({x: y}) - p.subs({x: y}) * q
    
    # 因此 expr 恰好可以被 x - y 整除
    # 将 expr 除以 x - y 并生成多项式对象
    poly = Poly(quo(expr, x - y), x, y)
    
    # 形成 Bezout 矩阵并存储在 B 中，根据 method 的值
    # 在第一行或者最后一行存储每个多项式的首项系数
    mx = max(m, n)
    B = zeros(mx)
    for i in range(mx):
        for j in range(mx):
            if method == 'prs':
                # 如果 method 是 'prs'，则存储在倒数第 i 行和倒数第 j 列
                B[mx - 1 - i, mx - 1 - j] = poly.nth(i, j)
            else:
                # 否则存储在第 i 行和第 j 列
                B[i, j] = poly.nth(i, j)
    
    # 返回构建的 Bezout 矩阵 B
    return B
def backward_eye(n):
    '''
    Returns the backward identity matrix of dimensions n x n.

    Needed to "turn" the Bezout matrices
    so that the leading coefficients are first.
    See docstring of the function bezout(p, q, x, method='bz').
    '''
    # Create an identity matrix of size n x n
    M = eye(n)  # identity matrix of order n

    # Swap rows to reverse the order of the identity matrix
    for i in range(int(M.rows / 2)):
        M.row_swap(0 + i, M.rows - 1 - i)

    return M


def subresultants_bezout(p, q, x):
    """
    The input polynomials p, q are in Z[x] or in Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant polynomial remainder sequence
    of p, q by evaluating determinants of appropriately selected
    submatrices of bezout(p, q, x, 'prs'). The dimensions of the
    latter are deg(p) x deg(p).

    Each coefficient is computed by evaluating the determinant of the
    corresponding submatrix of bezout(p, q, x, 'prs').

    bezout(p, q, x, 'prs) is used instead of sylvester(p, q, x, 1),
    Sylvester's matrix of 1840, because the dimensions of the latter
    are (deg(p) + deg(q)) x (deg(p) + deg(q)).

    If the subresultant prs is complete, then the output coincides
    with the Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. G.M.Diaz-Toca,L.Gonzalez-Vega: Various New Expressions for Subresultants
    and Their Applications. Appl. Algebra in Engin., Communic. and Comp.,
    Vol. 15, 233-266, 2004.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    f, g = p, q
    n = degF = degree(f, x)
    m = degG = degree(g, x)

    # make sure proper degrees
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, degF, degG, f, g = m, n, degG, degF, g, f
    if n > 0 and m == 0:
        return [f, g]

    SR_L = [f, g]      # subresultant list
    F = LC(f, x)**(degF - degG)

    # form the bezout matrix
    B = bezout(f, g, x, 'prs')

    # pick appropriate submatrices of B
    # and form subresultant polys
    if degF > degG:
        j = 2
    if degF == degG:
        j = 1
    while j <= degF:
        M = B[0:j, :]
        k, coeff_L = j - 1, []
        while k <= degF - 1:
            coeff_L.append(M[:, 0:j].det())
            if k < degF - 1:
                M.col_swap(j - 1, k + 1)
            k = k + 1

        # apply Theorem 2.1 in the paper by Toca & Vega 2004
        # to get correct signs
        SR_L.append(int((-1)**(j*(j-1)/2)) * (Poly(coeff_L, x) / F).as_expr())
        j = j + 1

    return process_matrix_output(SR_L, x)


def modified_subresultants_bezout(p, q, x):
    """
    The input polynomials p, q are in Z[x] or in Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the modified subresultant polynomial remainder sequence
    of p, q by evaluating determinants of appropriately selected
    submatrices of bezout(p, q, x, 'prs'). The dimensions of the
    latter are deg(p) x deg(p).

    Each coefficient is computed by evaluating the determinant of the
    corresponding submatrix of bezout(p, q, x, 'prs').

    bezout(p, q, x, 'prs) is used instead of sylvester(p, q, x, 1),
    Sylvester's matrix of 1840, because the dimensions of the latter
    are (deg(p) + deg(q)) x (deg(p) + deg(q)).

    If the subresultant prs is complete, then the output coincides
    with the Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. G.M.Diaz-Toca,L.Gonzalez-Vega: Various New Expressions for Subresultants
    and Their Applications. Appl. Algebra in Engin., Communic. and Comp.,
    Vol. 15, 233-266, 2004.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    f, g = p, q
    n = degF = degree(f, x)
    m = degG = degree(g, x)

    # make sure proper degrees
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, degF, degG, f, g = m, n, degG, degF, g, f
    if n > 0 and m == 0:
        return [f, g]

    SR_L = [f, g]      # subresultant list
    F = LC(f, x)**(degF - degG)

    # form the bezout matrix
    B = bezout(f, g, x, 'prs')

    # pick appropriate submatrices of B
    # and form subresultant polys
    if degF > degG:
        j = 2
    if degF == degG:
        j = 1
    while j <= degF:
        M = B[0:j, :]
        k, coeff_L = j - 1, []
        while k <= degF - 1:
            coeff_L.append(M[:, 0:j].det())
            if k < degF - 1:
                M.col_swap(j - 1, k + 1)
            k = k + 1

        # apply Theorem 2.1 in the paper by Toca & Vega 2004
        # to get correct signs
        SR_L.append(int((-1)**(j*(j-1)/2)) * (Poly(coeff_L, x) / F).as_expr())
        j = j + 1

    return process_matrix_output(SR_L, x)
    """
    # 确保 p 和 q 都不为零
    if p == 0 or q == 0:
        return [p, q]

    # 初始化 f 和 g 为 p 和 q
    f, g = p, q
    # 计算多项式 f 和 g 的次数
    n = degF = degree(f, x)
    m = degG = degree(g, x)

    # 确保多项式次数合适
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, degF, degG, f, g = m, n, degG, degF, g, f
    if n > 0 and m == 0:
        return [f, g]

    # 初始化子结果多项式列表
    SR_L = [f, g]      # subresultant list

    # 构造 Bézout 矩阵
    B = bezout(f, g, x, 'prs')

    # 选择适当的 B 的子矩阵
    # 并形成子结果多项式
    if degF > degG:
        j = 2
    if degF == degG:
        j = 1
    while j <= degF:
        M = B[0:j, :]
        k, coeff_L = j - 1, []
        while k <= degF - 1:
            coeff_L.append(M[:, 0:j].det())
            if k < degF - 1:
                M.col_swap(j - 1, k + 1)
            k = k + 1

        ## 根据 Toca & Vega 2004 年的论文中的定理 2.1，在这种情况下不需要
        ## 因为 Bézout 矩阵等价于 Sylvester2 矩阵
        SR_L.append((Poly(coeff_L, x)).as_expr())
        j = j + 1

    # 处理输出矩阵 SR_L 并返回结果
    return process_matrix_output(SR_L, x)
    """
# 确保 p 和 q 都不为零多项式，若其中任何一个为零则直接返回 [p, q]
if p == 0 or q == 0:
    return [p, q]

# 确保多项式 p 和 q 的次数设置正确
d0 =  degree(p, x)
d1 =  degree(q, x)
if d0 == 0 and d1 == 0:
    return [p, q]
if d1 > d0:
    d0, d1 = d1, d0
    p, q = q, p
if d0 > 0 and d1 == 0:
    return [p, q]

# 确保 LC(p)（p 的首项系数）大于 0
flag = 0
if  LC(p,x) < 0:
    flag = 1
    p = -p
    q = -q

# 初始化变量
lcf = LC(p, x)**(d0 - d1)               # 计算 LC(p)^(d0 - d1)，用于后续的修正子剩余
a0, a1 = p, q                           # 将 p 和 q 赋值给 a0 和 a1
sturm_seq = [a0, a1]                    # 初始化 Sturm 序列的列表，起始包含 a0 和 a1
    del0 = d0 - d1                          # 计算多项式的度差
    rho1 =  LC(a1, x)                       # 计算多项式 a1 的主导系数
    exp_deg = d1 - 1                        # 计算多项式 a2 的预期次数
    a2 = - rem(a0, a1, domain=QQ)           # 计算多项式 a0 除以 a1 的第一个余项
    rho2 =  LC(a2,x)                        # 计算多项式 a2 的主导系数
    d2 =  degree(a2, x)                     # 计算多项式 a2 的实际次数
    deg_diff_new = exp_deg - d2             # 预期次数与实际次数的差
    del1 = d1 - d2                          # 计算多项式的度差

    # mul_fac is the factor by which a2 is multiplied to
    # get integer coefficients
    mul_fac_old = rho1**(del0 + del1 - deg_diff_new)

    # 根据 a2 被乘以的系数，进行追加处理
    if method == 0:
        sturm_seq.append( simplify(lcf * a2 *  Abs(mul_fac_old)))
    else:
        sturm_seq.append( simplify( a2 *  Abs(mul_fac_old)))

    # 主循环
    deg_diff_old = deg_diff_new
    while d2 > 0:
        a0, a1, d0, d1 = a1, a2, d1, d2      # 更新多项式和次数
        del0 = del1                          # 更新多项式的度差
        exp_deg = d1 - 1                     # 新的预期次数
        a2 = - rem(a0, a1, domain=QQ)        # 新的余项
        rho3 =  LC(a2, x)                    # 计算多项式 a2 的主导系数
        d2 =  degree(a2, x)                  # 计算多项式 a2 的实际次数
        deg_diff_new = exp_deg - d2          # 更新预期次数与实际次数的差
        del1 = d1 - d2                       # 更新多项式的度差

        # 考虑到被"遗漏"的 rho1**deg_diff_old 的影响
        expo_old = deg_diff_old               # rho1 的这个次方
        expo_new = del0 + del1 - deg_diff_new # rho2 的这个次方

        # 更新变量并追加到序列中
        mul_fac_new = rho2**(expo_new) * rho1**(expo_old) * mul_fac_old
        deg_diff_old, mul_fac_old = deg_diff_new, mul_fac_new
        rho1, rho2 = rho2, rho3
        if method == 0:
            sturm_seq.append( simplify(lcf * a2 *  Abs(mul_fac_old)))
        else:
            sturm_seq.append( simplify( a2 *  Abs(mul_fac_old)))

    if flag:          # 改变序列的符号
        sturm_seq = [-i for i in sturm_seq]

    # 是否存在次数大于 0 的最大公约数？
    m = len(sturm_seq)
    if sturm_seq[m - 1] == nan or sturm_seq[m - 1] == 0:
        sturm_seq.pop(m - 1)

    return sturm_seq
def sturm_q(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the (generalized) Sturm sequence of p and q in Q[x].
    Polynomial divisions in Q[x] are performed, using the function rem(p, q, x).

    The coefficients of the polynomials in the Sturm sequence can be uniquely
    determined from the corresponding coefficients of the polynomials found
    either in:

        (a) the ``modified'' subresultant prs, (references 1, 2)

    or in

        (b) the subresultant prs (reference 3).

    References
    ==========
    1. Pell A. J., R. L. Gordon. The Modified Remainders Obtained in Finding
    the Highest Common Factor of Two Polynomials. Annals of MatheMatics,
    Second Series, 18 (1917), No. 4, 188-193.

    2 Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
    and Modified Subresultant Polynomial Remainder Sequences.''
    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    d0 =  degree(p, x)   # 获取多项式 p 在变量 x 中的次数
    d1 =  degree(q, x)   # 获取多项式 q 在变量 x 中的次数
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        d0, d1 = d1, d0
        p, q = q, p
    if d0 > 0 and d1 == 0:
        return [p,q]

    # make sure LC(p) > 0
    flag = 0
    if  LC(p,x) < 0:     # 检查 p 的主导系数是否小于零
        flag = 1         # 如果是，设置标志以便后续更改符号
        p = -p           # 取 p 的相反数
        q = -q           # 取 q 的相反数

    # initialize
    a0, a1 = p, q                           # 初始化输入多项式
    sturm_seq = [a0, a1]                    # 初始化输出的 Sturm 序列列表
    a2 = -rem(a0, a1, domain=QQ)            # 计算第一个余数
    d2 =  degree(a2, x)                     # 计算第一个余数的次数
    sturm_seq.append( a2 )

    # main loop
    while d2 > 0:
        a0, a1, d0, d1 = a1, a2, d1, d2      # 更新多项式和次数
        a2 = -rem(a0, a1, domain=QQ)         # 计算新的余数
        d2 =  degree(a2, x)                  # 计算新余数的次数
        sturm_seq.append( a2 )

    if flag:          # 如果之前更改过符号，将整个序列取反
        sturm_seq = [-i for i in sturm_seq]

    # gcd is of degree > 0 ?
    m = len(sturm_seq)
    if sturm_seq[m - 1] == nan or sturm_seq[m - 1] == 0:
        sturm_seq.pop(m - 1)

    return sturm_seq

def sturm_amv(p, q, x, method=0):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the (generalized) Sturm sequence of p and q in Z[x] or Q[x].
    If q = diff(p, x, 1) it is the usual Sturm sequence.

    A. If method == 0, default, the remainder coefficients of the
       sequence are (in absolute value) ``modified'' subresultants, which
       for non-monic polynomials are greater than the coefficients of the
       corresponding subresultants by the factor Abs(LC(p)**( deg(p)- deg(q))).
    """
    # 保证 p 和 q 都不是零多项式
    if p == 0 or q == 0:
        return [p, q]

    # 保证多项式次数的正确性
    d0 =  degree(p, x)   # 获取多项式 p 在变量 x 中的次数
    d1 =  degree(q, x)   # 获取多项式 q 在变量 x 中的次数
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        d0, d1 = d1, d0
        p, q = q, p
    if d0 > 0 and d1 == 0:
        return [p,q]

    # 根据 method 参数选择不同的计算方式
    if method == 0:
        pass  # 暂时没有实现 method == 0 的具体计算方式
    # 计算欧几里得序列
    prs = euclid_amv(p, q, x)

    # 防御性检查：如果 prs 是空列表或长度为 2，直接返回 prs
    if prs == [] or len(prs) == 2:
        return prs

    # prs 中的系数是子剩余式，相比于对应的子剩余式要小
    # 因子为 Abs( LC(prs[0])**( degree(prs[0], x) - degree(prs[1], x) ) )
    # 参考第二个参考文献中的定理 2
    lcf = Abs( LC(prs[0])**( degree(prs[0], x) - degree(prs[1], x) ) )

    # 序列中的前两个多项式保持不变
    sturm_seq = [prs[0], prs[1]]

    # 根据 "-, -, +, +, -, -, +, +,..." 模式更改符号
    # 如果需要，乘以 lcf
    flag = 0
    m = len(prs)
    i = 2
    # 当 i 小于等于 m-1 时进行循环
    while i <= m-1:
        # 如果 flag 等于 0，则执行以下操作
        if flag == 0:
            # 将 -prs[i] 加入到 Sturm 序列中
            sturm_seq.append( - prs[i] )
            # 增加 i 的值
            i = i + 1
            # 如果 i 等于 m，则跳出循环
            if i == m:
                break
            # 将 -prs[i] 加入到 Sturm 序列中
            sturm_seq.append( - prs[i] )
            # 增加 i 的值
            i = i + 1
            # 设置 flag 为 1
            flag = 1
        # 如果 flag 等于 1，则执行以下操作
        elif flag == 1:
            # 将 prs[i] 加入到 Sturm 序列中
            sturm_seq.append( prs[i] )
            # 增加 i 的值
            i = i + 1
            # 如果 i 等于 m，则跳出循环
            if i == m:
                break
            # 将 prs[i] 加入到 Sturm 序列中
            sturm_seq.append( prs[i] )
            # 增加 i 的值
            i = i + 1
            # 设置 flag 为 0

    # 如果 method 等于 0 并且 lcf 大于 1，则判断是否需要修改子结果或修改后的子结果
    if method == 0 and lcf > 1:
        # 复制 Sturm 序列的前两个元素到辅助序列 aux_seq 中
        aux_seq = [sturm_seq[0], sturm_seq[1]]
        # 对于 Sturm 序列中的每个元素（从索引 2 开始），进行修改后的处理并加入到 aux_seq 中
        for i in range(2, m):
            aux_seq.append(simplify(sturm_seq[i] * lcf ))
        # 将辅助序列 aux_seq 赋值给 Sturm 序列 sturm_seq
        sturm_seq = aux_seq

    # 返回最终的 Sturm 序列
    return sturm_seq
# 定义一个函数用于计算多项式 p 和 q 在环 Z[x] 或 Q[x] 上的欧几里德序列
def euclid_pg(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the Euclidean sequence of p and q in Z[x] or Q[x].

    If the Euclidean sequence is complete the coefficients of the polynomials
    in the sequence are subresultants. That is, they are  determinants of
    appropriately selected submatrices of sylvester1, Sylvester's matrix of 1840.
    In this case the Euclidean sequence coincides with the subresultant prs
    of the polynomials p, q.

    If the Euclidean sequence is incomplete the signs of the coefficients of the
    polynomials in the sequence may differ from the signs of the coefficients of
    the corresponding polynomials in the subresultant prs; however, the absolute
    values are the same.

    To compute the Euclidean sequence, no determinant evaluation takes place.
    We first compute the (generalized) Sturm sequence  of p and q using
    sturm_pg(p, q, x, 1), in which case the coefficients are (in absolute value)
    equal to subresultants. Then we change the signs of the remainders in the
    Sturm sequence according to the pattern "-, -, +, +, -, -, +, +,..." ;
    see Lemma 1 in the 1st reference or Theorem 3 in the 2nd reference as well as
    the function sturm_pg(p, q, x).

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On the Remainders
    Obtained in Finding the Greatest Common Divisor of Two Polynomials.'' Serdica
    Journal of Computing 9(2) (2015), 123-138.

    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial
    Remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''
    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.
    """
    # 使用 Pell-Gordon (或 AMV) 定理计算斯图姆序列，其中 prs 的系数（绝对值）是子剩余式
    prs = sturm_pg(p, q, x, 1)  ## any other method would do

    # 防御性检查
    if prs == [] or len(prs) == 2:
        return prs

    # 欧几里德序列的前两个多项式的符号保持不变
    euclid_seq = [prs[0], prs[1]]

    # 根据 "-, -, +, +, -, -, +, +,..." 模式改变余数的符号
    flag = 0
    m = len(prs)
    i = 2
    while i <= m-1:
        if  flag == 0:
            euclid_seq.append(- prs[i] )
            i = i + 1
            if i == m:
                break
            euclid_seq.append(- prs[i] )
            i = i + 1
            flag = 1
        elif flag == 1:
            euclid_seq.append(prs[i] )
            i = i + 1
            if i == m:
                break
            euclid_seq.append(prs[i] )
            i = i + 1
            flag = 0

    return euclid_seq
    # 确保 p 和 q 均不为零
    if p == 0 or q == 0:
        return [p, q]
    
    # 确保 p 和 q 的次数是正确的
    d0 =  degree(p, x)    # 计算多项式 p 在变量 x 中的次数
    d1 =  degree(q, x)    # 计算多项式 q 在变量 x 中的次数
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        d0, d1 = d1, d0    # 交换 p 和 q 的次序
        p, q = q, p
    if d0 > 0 and d1 == 0:
        return [p,q]
    
    # 确保 LC(p) > 0
    flag = 0
    if  LC(p,x) < 0:      # 检查多项式 p 在变量 x 中的首项系数是否小于零
        flag = 1
        p = -p            # 将 p 取反
        q = -q            # 将 q 取反
    
    # 初始化
    a0, a1 = p, q                            # 将 p 和 q 分配给 a0 和 a1，作为输入多项式
    euclid_seq = [a0, a1]                    # 初始化欧几里得序列的输出列表
    a2 = rem(a0, a1, domain=QQ)              # 计算使用 QQ 域的多项式除法得到的第一个余数
    d2 =  degree(a2, x)                      # 计算余数 a2 在变量 x 中的次数
    euclid_seq.append( a2 )
    
    # 主循环
    while d2 > 0:
        a0, a1, d0, d1 = a1, a2, d1, d2       # 更新多项式和次数
        a2 = rem(a0, a1, domain=QQ)           # 计算新的余数
        d2 =  degree(a2, x)                   # 计算新余数 a2 在变量 x 中的次数
        euclid_seq.append( a2 )
    
    if flag:            # 如果之前将 p 和 q 取反了，现在需要将整个序列取反
        euclid_seq = [-i for i in euclid_seq]
    
    # 检查 gcd 是否是次数大于零的多项式
    m = len(euclid_seq)
    if euclid_seq[m - 1] == nan or euclid_seq[m - 1] == 0:   # 检查最后一个多项式是否为 NaN 或者 0
        euclid_seq.pop(m - 1)
    
    return euclid_seq    # 返回欧几里得序列
def euclid_amv(f, g, x):
    """
    f, g are polynomials in Z[x] or Q[x]. It is assumed
    that degree(f, x) >= degree(g, x).

    Computes the Euclidean sequence of p and q in Z[x] or Q[x].

    If the Euclidean sequence is complete the coefficients of the polynomials
    in the sequence are subresultants. That is, they are  determinants of
    appropriately selected submatrices of sylvester1, Sylvester's matrix of 1840.
    In this case the Euclidean sequence coincides with the subresultant prs,
    of the polynomials p, q.

    If the Euclidean sequence is incomplete the signs of the coefficients of the
    polynomials in the sequence may differ from the signs of the coefficients of
    the corresponding polynomials in the subresultant prs; however, the absolute
    values are the same.

    To compute the coefficients, no determinant evaluation takes place.
    Instead, polynomial divisions in Z[x] or Q[x] are performed, using
    the function rem_z(f, g, x);  the coefficients of the remainders
    computed this way become subresultants with the help of the
    Collins-Brown-Traub formula for coefficient reduction.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial
    remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''
    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.

    """
    # make sure neither f nor g is 0
    if f == 0 or g == 0:
        return [f, g]

    # make sure proper degrees
    d0 =  degree(f, x)  # 获取 f 的多项式阶数
    d1 =  degree(g, x)  # 获取 g 的多项式阶数
    if d0 == 0 and d1 == 0:  # 如果 f 和 g 的阶数都为 0，则直接返回它们
        return [f, g]
    if d1 > d0:  # 如果 g 的阶数大于 f 的阶数，则交换 f 和 g 以保证 d0 >= d1
        d0, d1 = d1, d0
        f, g = g, f
    if d0 > 0 and d1 == 0:  # 如果 f 的阶数大于 0 且 g 的阶数为 0，则直接返回它们
        return [f, g]

    # initialize 初始化变量
    a0 = f
    a1 = g
    euclid_seq = [a0, a1]  # 初始化欧几里得序列，起始为 a0 和 a1
    deg_dif_p1, c = degree(a0, x) - degree(a1, x) + 1, -1  # 初始化 deg_dif_p1 和 c

    # compute the first polynomial of the prs 计算 prs 的第一个多项式
    i = 1
    a2 = rem_z(a0, a1, x) / Abs( (-1)**deg_dif_p1 )     # 计算第一个余数
    euclid_seq.append( a2 )  # 将 a2 加入欧几里得序列
    d2 =  degree(a2, x)                              # 计算 a2 的阶数

    # main loop 主循环
    while d2 >= 1:
        a0, a1, d0, d1 = a1, a2, d1, d2       # 更新多项式和阶数
        i += 1
        sigma0 = -LC(a0)
        c = (sigma0**(deg_dif_p1 - 1)) / (c**(deg_dif_p1 - 2))
        deg_dif_p1 = degree(a0, x) - d2 + 1
        a2 = rem_z(a0, a1, x) / Abs( (c**(deg_dif_p1 - 1)) * sigma0 )
        euclid_seq.append( a2 )  # 将新的余数 a2 加入欧几里得序列
        d2 =  degree(a2, x)                   # 计算 a2 的阶数

    # gcd is of degree > 0 ? 检查 gcd 的阶数是否大于 0
    m = len(euclid_seq)
    if euclid_seq[m - 1] == nan or euclid_seq[m - 1] == 0:
        euclid_seq.pop(m - 1)

    return euclid_seq


def modified_subresultants_pg(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    """
    """
    Computes the ``modified'' subresultant prs of p and q in Z[x] or Q[x];
    the coefficients of the polynomials in the sequence are
    ``modified'' subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester2, Sylvester's matrix of 1853.

    To compute the coefficients, no determinant evaluation takes place. Instead,
    polynomial divisions in Q[x] are performed, using the function rem(p, q, x);
    the coefficients of the remainders computed this way become ``modified''
    subresultants with the help of the Pell-Gordon Theorem of 1917.

    If the ``modified'' subresultant prs is complete, and LC( p ) > 0, it coincides
    with the (generalized) Sturm sequence of the polynomials p, q.

    References
    ==========
    1. Pell A. J., R. L. Gordon. The Modified Remainders Obtained in Finding
    the Highest Common Factor of Two Polynomials. Annals of MatheMatics,
    Second Series, 18 (1917), No. 4, 188-193.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
    and Modified Subresultant Polynomial Remainder Sequences.''
    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    d0 =  degree(p,x)                   # degree of polynomial p
    d1 =  degree(q,x)                   # degree of polynomial q
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        d0, d1 = d1, d0                 # swap degrees if d1 > d0
        p, q = q, p                     # swap polynomials accordingly
    if d0 > 0 and d1 == 0:
        return [p,q]

    # initialize
    k =  var('k')                       # symbolic variable for index in summation formula
    u_list = []                         # list to store (-1)**u_i values
    subres_l = [p, q]                   # list to store modified subresultant prs
    a0, a1 = p, q                       # the input polynomials
    del0 = d0 - d1                      # degree difference
    degdif = del0                       # save degree difference
    rho_1 = LC(a0)                      # leading coefficient of a0

    # Initialize Pell-Gordon variables
    rho_list_minus_1 =  sign( LC(a0, x))      # sign of LC(a0)
    rho1 =  LC(a1, x)                         # leading coefficient of a1
    rho_list = [ sign(rho1)]                  # list of signs
    p_list = [del0]                           # list of degree differences
    u =  summation(k, (k, 1, p_list[0]))      # summation result for u
    u_list.append(u)                          # append u value to u_list
    v = sum(p_list)                           # calculate v value

    # first remainder
    exp_deg = d1 - 1                          # expected degree of a2
    a2 = - rem(a0, a1, domain=QQ)             # compute first remainder
    rho2 =  LC(a2, x)                         # leading coefficient of a2
    d2 =  degree(a2, x)                       # actual degree of a2
    deg_diff_new = exp_deg - d2               # expected - actual degree
    del1 = d1 - d2                            # degree difference

    # mul_fac is the factor by which a2 is multiplied to
    # get integer coefficients
    # 计算 mul_fac_old，根据给定的公式 rho1**(del0 + del1 - deg_diff_new)
    mul_fac_old = rho1**(del0 + del1 - deg_diff_new)

    # 更新 Pell-Gordon 变量，将 1 + deg_diff_new 添加到 p_list 中
    p_list.append(1 + deg_diff_new)              # deg_diff_new 为完整序列时为 0

    # 应用 Pell-Gordon 公式 (7)，参考第二个引用
    num = 1                                     # 分数的分子
    for u in u_list:
        num *= (-1)**u
    num = num * (-1)**v

    # 分母取决于完整/不完整序列
    if deg_diff_new == 0:                        # 完整序列
        den = 1
        for k in range(len(rho_list)):
            den *= rho_list[k]**(p_list[k] + p_list[k + 1])
        den = den * rho_list_minus_1
    else:                                       # 不完整序列
        den = 1
        for k in range(len(rho_list)-1):
            den *= rho_list[k]**(p_list[k] + p_list[k + 1])
        den = den * rho_list_minus_1
        expo = (p_list[len(rho_list) - 1] + p_list[len(rho_list)] - deg_diff_new)
        den = den * rho_list[len(rho_list) - 1]**expo

    # 行列式的符号取决于 sg(num / den)
    if sign(num / den) > 0:
        subres_l.append( simplify(rho_1**degdif*a2* Abs(mul_fac_old) ) )
    else:
        subres_l.append(- simplify(rho_1**degdif*a2* Abs(mul_fac_old) ) )

    # 更新 Pell-Gordon 变量
    k = var('k')
    rho_list.append( sign(rho2))
    u =  summation(k, (k, 1, p_list[len(p_list) - 1]))
    u_list.append(u)
    v = sum(p_list)
    deg_diff_old=deg_diff_new

    # 主循环
    while d2 > 0:
        a0, a1, d0, d1 = a1, a2, d1, d2      # 更新多项式和次数
        del0 = del1                          # 更新次数差
        exp_deg = d1 - 1                     # 新的期望次数
        a2 = - rem(a0, a1, domain=QQ)        # 新的余数
        rho3 =  LC(a2, x)                    # a2 的首项系数
        d2 =  degree(a2, x)                  # a2 的实际次数
        deg_diff_new = exp_deg - d2          # 期望次数减去实际次数
        del1 = d1 - d2                       # 次数差

        # 考虑“遗漏”的 rho1**deg_diff_old 的幂
        expo_old = deg_diff_old               # rho1 提升到这个幂
        expo_new = del0 + del1 - deg_diff_new # rho2 提升到这个幂

        mul_fac_new = rho2**(expo_new) * rho1**(expo_old) * mul_fac_old

        # 更新变量
        deg_diff_old, mul_fac_old = deg_diff_new, mul_fac_new
        rho1, rho2 = rho2, rho3

        # 更新 Pell-Gordon 变量
        p_list.append(1 + deg_diff_new)       # 对完整序列来说，deg_diff_new 为 0

        # 应用第二个参考中的 Pell-Gordon 公式（7）
        num = 1                              # 分子
        for u in u_list:
            num *= (-1)**u
        num = num * (-1)**v

        # 分母取决于完整/不完整序列
        if deg_diff_new == 0:                 # 完整序列
            den = 1
            for k in range(len(rho_list)):
                den *= rho_list[k]**(p_list[k] + p_list[k + 1])
            den = den * rho_list_minus_1
        else:                                # 不完整序列
            den = 1
            for k in range(len(rho_list)-1):
                den *= rho_list[k]**(p_list[k] + p_list[k + 1])
            den = den * rho_list_minus_1
            expo = (p_list[len(rho_list) - 1] + p_list[len(rho_list)] - deg_diff_new)
            den = den * rho_list[len(rho_list) - 1]**expo

        # 行列式的符号取决于 sg(num / den)
        if  sign(num / den) > 0:
            subres_l.append( simplify(rho_1**degdif*a2* Abs(mul_fac_old) ) )
        else:
            subres_l.append(- simplify(rho_1**degdif*a2* Abs(mul_fac_old) ) )

        # 更新 Pell-Gordon 变量
        k = var('k')
        rho_list.append( sign(rho2))
        u =  summation(k, (k, 1, p_list[len(p_list) - 1]))
        u_list.append(u)
        v = sum(p_list)

    # gcd 的次数 > 0 ?
    m = len(subres_l)
    if subres_l[m - 1] == nan or subres_l[m - 1] == 0:
        subres_l.pop(m - 1)

    # LC( p ) < 0
    m = len(subres_l)   # 由于 deg(gcd ) > 0，列表可能变短了
    if LC( p ) < 0:
        aux_seq = [subres_l[0], subres_l[1]]
        for i in range(2, m):
            aux_seq.append(simplify(subres_l[i] * (-1) ))
        subres_l = aux_seq

    return  subres_l
# p, q are polynomials in Z[x] or Q[x]. It is assumed that degree(p, x) >= degree(q, x).
# Computes the subresultant prs of p and q in Z[x] or Q[x], based on modified subresultants.
# The coefficients of the polynomials in these sequences differ only in sign and a specific factor as per Theorem 2 of the reference.
# The output sequence contains subresultants, which are determinants of selected submatrices of sylvester1, Sylvester's matrix of 1840.
# If complete, the subresultant prs coincides with the Euclidean sequence of p, q.
def subresultants_pg(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant prs of p and q in Z[x] or Q[x], from
    the modified subresultant prs of p and q.

    The coefficients of the polynomials in these two sequences differ only
    in sign and the factor LC(p)**( deg(p)- deg(q)) as stated in
    Theorem 2 of the reference.

    The coefficients of the polynomials in the output sequence are
    subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester1, Sylvester's matrix of 1840.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: "On the Remainders
    Obtained in Finding the Greatest Common Divisor of Two Polynomials."
    Serdica Journal of Computing 9(2) (2015), 123-138.

    """
    # compute the modified subresultant prs
    lst = modified_subresultants_pg(p,q,x)  ## any other method would do

    # defensive
    if lst == [] or len(lst) == 2:
        return lst

    # the coefficients in lst are modified subresultants and, hence, are
    # greater than those of the corresponding subresultants by the factor
    # LC(lst[0])**( deg(lst[0]) - deg(lst[1])); see Theorem 2 in reference.
    lcf = LC(lst[0])**( degree(lst[0], x) - degree(lst[1], x) )

    # Initialize the subresultant prs list
    subr_seq = [lst[0], lst[1]]

    # compute the degree sequences m_i and j_i of Theorem 2 in reference.
    deg_seq = [degree(Poly(poly, x), x) for poly in lst]
    deg = deg_seq[0]
    deg_seq_s = deg_seq[1:-1]
    m_seq = [m-1 for m in deg_seq_s]
    j_seq = [deg - m for m in m_seq]

    # compute the AMV factors of Theorem 2 in reference.
    fact = [(-1)**( j*(j-1)/S(2) ) for j in j_seq]

    # shortened list without the first two polys
    lst_s = lst[2:]

    # poly lst_s[k] is multiplied times fact[k], divided by lcf
    # and appended to the subresultant prs list
    m = len(fact)
    for k in range(m):
        if sign(fact[k]) == -1:
            subr_seq.append(-lst_s[k] / lcf)
        else:
            subr_seq.append(lst_s[k] / lcf)

    return subr_seq


# p, q are polynomials in Z[x] or Q[x]. It is assumed that degree(p, x) >= degree(q, x).
# Computes the subresultant prs of p and q in Q[x];
# the coefficients of the polynomials in the sequence are subresultants,
# determinants of appropriately selected submatrices of sylvester1, Sylvester's matrix of 1840.
# To compute the coefficients, no determinant evaluation takes place.
# Instead, polynomial divisions in Q[x] are performed, using the function rem(p, q, x);
# the coefficients of the remainders computed this way become subresultants with the help of the
# Akritas-Malaschonok-Vigklas Theorem of 2015.
def subresultants_amv_q(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant prs of p and q in Q[x];
    the coefficients of the polynomials in the sequence are
    subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester1, Sylvester's matrix of 1840.

    To compute the coefficients, no determinant evaluation takes place.
    Instead, polynomial divisions in Q[x] are performed, using the
    function rem(p, q, x);  the coefficients of the remainders
    computed this way become subresultants with the help of the
    Akritas-Malaschonok-Vigklas Theorem of 2015.

    """
    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial
    remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''
    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.

    """
    # make sure neither p nor q is 0
    确保 p 和 q 都不为 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    确保多项式 p 和 q 的次数合适
    d0 =  degree(p, x)
    d1 =  degree(q, x)
    如果 d0 和 d1 都为 0，则直接返回 [p, q]
    if d0 == 0 and d1 == 0:
        return [p, q]
    如果 d1 大于 d0，则交换它们以便 d0 始终大于等于 d1
    if d1 > d0:
        d0, d1 = d1, d0
        p, q = q, p
    如果 d0 大于 0 且 d1 等于 0，则直接返回 [p, q]
    if d0 > 0 and d1 == 0:
        return [p, q]

    # initialize
    初始化变量
    i, s = 0, 0                              # counters for remainders & odd elements
    i 和 s 分别用于计数余项和奇数元素
    p_odd_index_sum = 0                      # contains the sum of p_1, p_3, etc
    p_odd_index_sum 用于存储 p_1, p_3 等的和
    subres_l = [p, q]                        # subresultant prs output list
    subres_l 用于存储子剩余多项式 prs 的输出列表
    a0, a1 = p, q                            # the input polys
    a0 和 a1 分别表示输入的多项式
    sigma1 =  LC(a1, x)                      # leading coeff of a1
    sigma1 是多项式 a1 的首项系数
    p0 = d0 - d1                             # degree difference
    p0 表示 d0 和 d1 之间的次数差
    if p0 % 2 == 1:
        s += 1
    如果 p0 是奇数，则增加 s 的计数
    phi = floor( (s + 1) / 2 )
    phi 是 (s + 1) / 2 的整数部分
    mul_fac = 1
    mul_fac 是初始化为 1 的乘法因子
    d2 = d1

    # main loop
    主循环开始
    while d2 > 0:
        i += 1
        增加 i 的计数
        a2 = rem(a0, a1, domain= QQ)          # new remainder
        a2 是通过 a0 除以 a1 后得到的新的余项
        if i == 1:
            sigma2 =  LC(a2, x)
        如果 i 等于 1，则计算 sigma2
        else:
            sigma3 =  LC(a2, x)
            sigma1, sigma2 = sigma2, sigma3
        否则，更新 sigma1 和 sigma2
        d2 =  degree(a2, x)
        计算 a2 的次数
        p1 = d1 - d2
        计算 p1 作为 d1 和 d2 的差
        psi = i + phi + p_odd_index_sum
        psi 是公式（9）中第一个分数的符号

        # new mul_fac
        计算新的乘法因子
        mul_fac = sigma1**(p0 + 1) * mul_fac

        ## compute the sign of the first fraction in formula (9) of the paper
        计算公式（9）中第一个分数的符号
        # numerator
        num = (-1)**psi
        计算分子
        # denominator
        den = sign(mul_fac)
        计算分母

        # the sign of the determinant depends on sign( num / den ) != 0
        判断行列式的符号是否不等于 0
        if  sign(num / den) > 0:
            subres_l.append( simplify(expand(a2* Abs(mul_fac))))
        如果大于 0，则将新的子剩余多项式加入 subres_l
        else:
            subres_l.append(- simplify(expand(a2* Abs(mul_fac))))
        否则，加入负数形式的子剩余多项式

        ## bring into mul_fac the missing power of sigma if there was a degree gap
        如果存在次数差，则将缺失的 sigma 幂引入乘法因子
        if p1 - 1 > 0:
            mul_fac = mul_fac * sigma1**(p1 - 1)

       # update AMV variables
        更新 AMV 变量
        a0, a1, d0, d1 = a1, a2, d1, d2
        更新 a0, a1, d0, d1
        p0 = p1
        更新 p0
        if p0 % 2 ==1:
            s += 1
        如果 p0 是奇数，则增加 s 的计数
        phi = floor( (s + 1) / 2 )
        更新 phi
        if i%2 == 1:
            p_odd_index_sum += p0             # p_i has odd index
        如果 i 是奇数，则更新 p_odd_index_sum

    # gcd is of degree > 0 ?
    检查最终的 gcd 的次数是否大于 0
    m = len(subres_l)
    计算 subres_l 的长度
    if subres_l[m - 1] == nan or subres_l[m - 1] == 0:
        如果最后一个元素是 nan 或者 0，则移除它
        subres_l.pop(m - 1)

    return  subres_l
    返回计算得到的子剩余多项式列表
# 计算给定 base 的 expo 次幂的符号，而不计算幂本身
def compute_sign(base, expo):
    # 获取 base 的符号
    sb = sign(base)
    # 如果 base 的符号为正，则结果为 1
    if sb == 1:
        return 1
    # 计算 expo 对 2 取余数
    pe = expo % 2
    # 如果 expo 为偶数，则结果为 -sb（base 的相反符号）
    if pe == 0:
        return -sb
    else:
        # 如果 expo 为奇数，则结果为 base 的符号 sb
        return sb

# 计算多项式 p 除以 q 的余数，确保余数也在整数多项式环 Z[x] 中
def rem_z(p, q, x):
    # 检查 p 和 q 是否是 Z[x] 中的多项式，并且确保它们有相同的生成器
    if (p.as_poly().is_univariate and q.as_poly().is_univariate and
            p.as_poly().gens == q.as_poly().gens):
        # 计算 delta，为 p 的次数减去 q 的次数加 1
        delta = (degree(p, x) - degree(q, x) + 1)
        # 返回将 p 乘以 q 的主导系数的绝对值的 delta 次幂后的结果，再使用 rem 函数进行多项式除法
        return rem(Abs(LC(q, x))**delta  *  p, q, x)
    else:
        # 如果 p 和 q 不符合 Z[x] 的要求，则使用 prem 函数
        return prem(p, q, x)

# 计算多项式 p 除以 q 的商，确保商也在整数多项式环 Z[x] 中
def quo_z(p, q, x):
    # 检查 p 和 q 是否是 Z[x] 中的多项式，并且确保它们有相同的生成器
    if (p.as_poly().is_univariate and q.as_poly().is_univariate and
            p.as_poly().gens == q.as_poly().gens):
        # 计算 delta，为 p 的次数减去 q 的次数加 1
        delta = (degree(p, x) - degree(q, x) + 1)
        # 返回将 p 乘以 q 的主导系数的绝对值的 delta 次幂后的结果，再使用 quo 函数进行多项式除法
        return quo(Abs(LC(q, x))**delta  *  p, q, x)
    else:
        # 如果 p 和 q 不符合 Z[x] 的要求，则使用 pquo 函数
        return pquo(p, q, x)
    # 确保 f 和 g 都不为零
    if f == 0 or g == 0:
        return [f, g]

    # 确保 f 的次数大于等于 g 的次数
    d0 =  degree(f, x)
    d1 =  degree(g, x)
    if d0 == 0 and d1 == 0:
        return [f, g]
    if d1 > d0:
        d0, d1 = d1, d0
        f, g = g, f
    if d0 > 0 and d1 == 0:
        return [f, g]

    # 初始化
    a0 = f
    a1 = g
    subres_l = [a0, a1]  # 存放子结果多项式的列表
    deg_dif_p1, c = degree(a0, x) - degree(a1, x) + 1, -1  # deg_dif_p1 表示多项式差的次数加一，c 未使用

    # 初始化 AMV 计算中用到的变量
    sigma1 =  LC(a1, x)                      # a1 的首项系数
    i, s = 0, 0                              # 计数器，用于余数和奇数元素
    p_odd_index_sum = 0                      # 包含 p_1、p_3 等的和
    p0 = deg_dif_p1 - 1
    if p0 % 2 == 1:
        s += 1
    phi = floor( (s + 1) / 2 )               # phi 是 s 和 1 的一半的向下取整

    # 计算 prs 的第一个多项式
    i += 1
    a2 = rem_z(a0, a1, x) / Abs( (-1)**deg_dif_p1 )     # 第一个余数
    sigma2 =  LC(a2, x)                       # a2 的首项系数
    d2 =  degree(a2, x)                       # a2 的次数
    p1 = d1 - d2                              # 次数差

    # sgn_den 是公式 (9) 中第一个分数的分母，用来使 a2 的系数变为整数
    sgn_den = compute_sign( sigma1, p0 + 1 )

    ## 计算公式 (9) 中第一个分数的符号
    # 分子
    psi = i + phi + p_odd_index_sum
    num = (-1)**psi
    # 分母
    den = sgn_den

    # 行列式的符号取决于 sign(num / den) 是否大于零
    if  sign(num / den) > 0:
        subres_l.append( a2 )  # 将 a2 加入子结果多项式列表
    else:
        subres_l.append( -a2 )  # 将 -a2 加入子结果多项式列表
    # 更新 AMV 变量
    if p1 % 2 == 1:
        s += 1

    # 如果存在间隙，引入 sigma 的缺失幂
    if p1 - 1 > 0:
        sgn_den = sgn_den * compute_sign(sigma1, p1 - 1)

    # 主循环
    while d2 >= 1:
        # 计算 phi
        phi = floor((s + 1) / 2)
        if i % 2 == 1:
            p_odd_index_sum += p1  # p_i 的索引是奇数时累加

        # 更新多项式和度数
        a0, a1, d0, d1 = a1, a2, d1, d2
        p0 = p1  # 更新度数差异
        i += 1
        sigma0 = -LC(a0)

        # 计算系数 c
        c = (sigma0**(deg_dif_p1 - 1)) / (c**(deg_dif_p1 - 2))
        deg_dif_p1 = degree(a0, x) - d2 + 1
        a2 = rem_z(a0, a1, x) / Abs((c**(deg_dif_p1 - 1)) * sigma0)
        sigma3 = LC(a2, x)  # a2 的主导系数
        d2 = degree(a2, x)  # a2 的实际度数
        p1 = d1 - d2  # 更新度数差异
        psi = i + phi + p_odd_index_sum

        # 更新变量
        sigma1, sigma2 = sigma2, sigma3

        # 计算新的 sgn_den
        sgn_den = compute_sign(sigma1, p0 + 1) * sgn_den

        # 计算公式（9）中第一个分数的符号
        # 分子
        num = (-1)**psi
        # 分母
        den = sgn_den

        # 行列式的符号依赖于 sign(num / den) != 0
        if sign(num / den) > 0:
            subres_l.append(a2)
        else:
            subres_l.append(-a2)

        # 更新 AMV 变量
        if p1 % 2 == 1:
            s += 1

        # 如果存在间隙，引入 sigma 的缺失幂
        if p1 - 1 > 0:
            sgn_den = sgn_den * compute_sign(sigma1, p1 - 1)

    # 检查 gcd 的度数是否大于 0
    m = len(subres_l)
    if subres_l[m - 1] == nan or subres_l[m - 1] == 0:
        subres_l.pop(m - 1)

    return subres_l
# 计算修改后的子结果式 AMV
def modified_subresultants_amv(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the modified subresultant prs of p and q in Z[x] or Q[x],
    from the subresultant prs of p and q.
    The coefficients of the polynomials in the two sequences differ only
    in sign and the factor LC(p)**( deg(p)- deg(q)) as stated in
    Theorem 2 of the reference.

    The coefficients of the polynomials in the output sequence are
    modified subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester2, Sylvester's matrix of 1853.

    If the modified subresultant prs is complete, and LC( p ) > 0, it coincides
    with the (generalized) Sturm's sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: "On the Remainders
    Obtained in Finding the Greatest Common Divisor of Two Polynomials."
    Serdica Journal of Computing, Serdica Journal of Computing, 9(2) (2015), 123-138.

    """

    # 计算子结果式 prs
    lst = subresultants_amv(p,q,x)     ## any other method would do

    # 防御性代码
    if lst == [] or len(lst) == 2:
        return lst

    # lst 中的系数是子结果式，因此比相应修改后的子结果式中的系数小
    # 因子为 LC(lst[0])**( deg(lst[0]) - deg(lst[1])); 参见定理 2
    lcf = LC(lst[0])**( degree(lst[0], x) - degree(lst[1], x) )

    # 初始化修改后的子结果式列表
    subr_seq = [lst[0], lst[1]]

    # 计算定理 2 的度数序列 m_i 和 j_i
    deg_seq = [degree(Poly(poly, x), x) for poly in lst]
    deg = deg_seq[0]
    deg_seq_s = deg_seq[1:-1]
    m_seq = [m-1 for m in deg_seq_s]
    j_seq = [deg - m for m in m_seq]

    # 计算定理 2 的 AMV 因子
    fact = [(-1)**( j*(j-1)/S(2) ) for j in j_seq]

    # 简化后的列表去掉前两个多项式
    lst_s = lst[2:]

    # 多项式 lst_s[k] 乘以 fact[k] 和 lcf，然后附加到修改后的子结果式列表中
    m = len(fact)
    for k in range(m):
        if sign(fact[k]) == -1:
            subr_seq.append( simplify(-lst_s[k] * lcf) )
        else:
            subr_seq.append( simplify(lst_s[k] * lcf) )

    return subr_seq


# 用于各种子结果式 prs 算法的辅助函数
def correct_sign(deg_f, deg_g, s1, rdel, cdel):
    """
    Used in various subresultant prs algorithms.

    Evaluates the determinant, (a.k.a. subresultant) of a properly selected
    submatrix of s1, Sylvester's matrix of 1840, to get the correct sign
    and value of the leading coefficient of a given polynomial remainder.

    deg_f, deg_g are the degrees of the original polynomials p, q for which the
    matrix s1 = sylvester(p, q, x, 1) was constructed.

    rdel denotes the expected degree of the remainder; it is the number of
    rows to be deleted from each group of rows in s1 as described in the
    reference below.
    """
    cdel denotes the expected degree minus the actual degree of the remainder;
    it is the number of columns to be deleted --- starting with the last column
    forming the square matrix --- from the matrix resulting after the row deletions.

    References
    ==========
    Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
    and Modified Subresultant Polynomial Remainder Sequences.''
    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    """
    M = s1[:, :]   # copy of matrix s1

    # eliminate rdel rows from the first deg_g rows
    for i in range(M.rows - deg_f - 1, M.rows - deg_f - rdel - 1, -1):
        M.row_del(i)
    # 从前 deg_f 行中删除 rdel 行

    # eliminate rdel rows from the last deg_f rows
    for i in range(M.rows - 1, M.rows - rdel - 1, -1):
        M.row_del(i)
    # 从最后 deg_f 行中删除 rdel 行

    # eliminate cdel columns
    for i in range(cdel):
        M.col_del(M.rows - 1)
    # 删除 cdel 列

    # define submatrix
    Md = M[:, 0: M.rows]
    # 定义子矩阵 Md，取 M 的前 M.rows 行和所有列组成的子矩阵

    return Md.det()
    # 返回 Md 的行列式的值
# 计算两个多项式 p, q 在整数环或有理数环上的子剩余结果序列（subresultant polynomial remainder sequence，PRS）。
def subresultants_rem(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant prs of p and q in Z[x] or Q[x];
    the coefficients of the polynomials in the sequence are
    subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester1, Sylvester's matrix of 1840.

    To compute the coefficients polynomial divisions in Q[x] are
    performed, using the function rem(p, q, x). The coefficients
    of the remainders computed this way become subresultants by evaluating
    one subresultant per remainder --- that of the leading coefficient.
    This way we obtain the correct sign and value of the leading coefficient
    of the remainder and we easily ``force'' the rest of the coefficients
    to become subresultants.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G.:``Three New Methods for Computing Subresultant
    Polynomial Remainder Sequences (PRS's).'' Serdica Journal of Computing 9(1) (2015), 1-26.

    """
    # 确保 p 和 q 都不是零多项式
    if p == 0 or q == 0:
        return [p, q]

    # 确保多项式 p 的次数大于等于多项式 q 的次数
    f, g = p, q
    n = deg_f = degree(f, x)
    m = deg_g = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, deg_f, deg_g, f, g = m, n, deg_g, deg_f, g, f
    if n > 0 and m == 0:
        return [f, g]

    # 初始化
    s1 = sylvester(f, g, x, 1)   # 计算 Sylvester 矩阵
    sr_list = [f, g]            # 子剩余结果列表

    # 主循环
    while deg_g > 0:
        r = rem(p, q, x)         # 计算 p 除以 q 的余数
        d = degree(r, x)
        if d < 0:
            return sr_list

        # 计算系数使其成为子剩余结果，评估一个行列式
        exp_deg = deg_g - 1          # 预期的次数
        sign_value = correct_sign(n, m, s1, exp_deg, exp_deg - d)
        r = simplify((r / LC(r, x)) * sign_value)

        # 将具有子剩余结果系数的多项式追加到列表中
        sr_list.append(r)

        # 更新多项式的次数和多项式本身
        deg_f, deg_g = deg_g, d
        p, q = q, r

    # 如果最后一个子剩余结果的次数大于 0，则移除
    m = len(sr_list)
    if sr_list[m - 1] == nan or sr_list[m - 1] == 0:
        sr_list.pop(m - 1)

    return sr_list


# 将指定矩阵 M 的第 (i, j) 元素作为主元素，应用 Dodgson-Bareiss 整数保持变换，将该主元素下方所有元素置零。
def pivot(M, i, j):
    '''
    M is a matrix, and M[i, j] specifies the pivot element.

    All elements below M[i, j], in the j-th column, will
    be zeroed, if they are not already 0, according to
    Dodgson-Bareiss' integer preserving transformations.

    References
    ==========
    1. Akritas, A. G.: ``A new method for computing polynomial greatest
    common divisors and polynomial remainder sequences.''
    Numerische MatheMatik 52, 119-127, 1988.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On a Theorem
    by Van Vleck Regarding Sturm Sequences.''
    Serdica Journal of Computing, 7, No 4, 101-134, 2013.

    '''
    # 创建矩阵 M 的副本 ma
    ma = M[:, :]  # copy of matrix M
    # 获取副本矩阵 ma 的行数
    rs = ma.rows  # No. of rows
    # 获取副本矩阵 ma 的列数
    cs = ma.cols  # No. of cols
    
    # 对副本矩阵 ma 进行行操作，从第 i+1 行开始到末尾
    for r in range(i+1, rs):
        # 如果当前行 r、列 j 处的元素不为零
        if ma[r, j] != 0:
            # 对于从 j+1 到末尾的每一列 c
            for c in range(j + 1, cs):
                # 根据矩阵消元公式更新矩阵元素 ma[r, c]
                ma[r, c] = ma[i, j] * ma[r, c] - ma[i, c] * ma[r, j]
            # 将当前行 r、列 j 处的元素置零
            ma[r, j] = 0
    
    # 返回更新后的副本矩阵 ma
    return ma
# 定义一个函数，用于将列表 L 向右旋转 k 个位置。L 可以是矩阵的行或列表。
def rotate_r(L, k):
    '''
    Rotates right by k. L is a row of a matrix or a list.

    右旋转列表 L k 个位置。L 可以是矩阵的行或列表。
    '''
    # 将 L 转换为列表 ll
    ll = list(L)
    # 如果 ll 为空列表，直接返回空列表
    if ll == []:
        return []
    # 依次向右旋转 k 次
    for i in range(k):
        # 弹出列表 ll 最后一个元素
        el = ll.pop(len(ll) - 1)
        # 将 el 插入到列表 ll 的开头
        ll.insert(0, el)
    # 如果 L 是列表，则返回旋转后的列表 ll；否则，返回包含 ll 的 Matrix 对象
    return ll if isinstance(L, list) else Matrix([ll])

# 定义一个函数，用于将列表 L 向左旋转 k 个位置。L 可以是矩阵的行或列表。
def rotate_l(L, k):
    '''
    Rotates left by k. L is a row of a matrix or a list.

    左旋转列表 L k 个位置。L 可以是矩阵的行或列表。
    '''
    # 将 L 转换为列表 ll
    ll = list(L)
    # 如果 ll 为空列表，直接返回空列表
    if ll == []:
        return []
    # 依次向左旋转 k 次
    for i in range(k):
        # 弹出列表 ll 的第一个元素
        el = ll.pop(0)
        # 将 el 插入到列表 ll 的倒数第二个位置
        ll.insert(len(ll) - 1, el)
    # 如果 L 是列表，则返回旋转后的列表 ll；否则，返回包含 ll 的 Matrix 对象
    return ll if isinstance(L, list) else Matrix([ll])

# 定义一个函数，将矩阵的行 row 转换为一个指定次数 deg 和变量 x 的多项式。
def row2poly(row, deg, x):
    '''
    Converts the row of a matrix to a poly of degree deg and variable x.
    Some entries at the beginning and/or at the end of the row may be zero.

    将矩阵的行 row 转换为次数为 deg 且变量为 x 的多项式。
    行的开头和/或结尾可能包含零项。
    '''
    # 初始化 k 为 0
    k = 0
    # 初始化空的多项式 poly
    poly = []
    # 获取行的长度 leng
    leng = len(row)

    # 找到多项式的起始点，即行中第一个非零元素的位置
    while row[k] == 0:
        k = k + 1

    # 将多项式的下一个 deg + 1 个元素添加到 poly 中
    for j in range( deg + 1):
        if k + j <= leng:
            poly.append(row[k + j])

    # 返回多项式对象 Poly，使用变量 x
    return Poly(poly, x)

# 定义一个函数，创建一个“小”矩阵 M，以便进行三角化。
def create_ma(deg_f, deg_g, row1, row2, col_num):
    '''
    Creates a ``small'' matrix M to be triangularized.

    创建一个“小”矩阵 M，用于进行三角化。

    deg_f, deg_g 分别为被除数和除数多项式的次数，其中 deg_g > deg_f。
    row1 中的元素为除数多项式的系数，row2 中的元素为被除数多项式的系数。
    col_num 定义矩阵 M 的列数。
    '''
    # 如果 deg_g - deg_f >= 1，则打印消息并返回
    if deg_g - deg_f >= 1:
        print('Reverse degrees')
        return

    # 创建一个 deg_f - deg_g + 2 行，col_num 列的零矩阵 m
    m = zeros(deg_f - deg_g + 2, col_num)

    # 填充矩阵 m 的前 deg_f - deg_g + 1 行
    for i in range(deg_f - deg_g + 1):
        m[i, :] = rotate_r(row1, i)
    # 将 row2 填充到矩阵 m 的最后一行
    m[deg_f - deg_g + 1, :] = row2

    # 返回创建的矩阵 m
    return m

# 定义一个函数，找到由 create_ma() 创建的“小”矩阵 M 在三角化后对应的多项式的次数。
def find_degree(M, deg_f):
    '''
    Finds the degree of the poly corresponding (after triangularization)
    to the _last_ row of the ``small'' matrix M, created by create_ma().

    查找与 create_ma() 创建的“小”矩阵 M 的最后一行（三角化后）对应的多项式的次数。

    deg_f 是被除数多项式的次数。
    如果最后一行全为 0，则返回 None。
    '''
    # 初始化 j 为 deg_f
    j = deg_f
    # 遍历 M 的列
    for i in range(0, M.cols):
        # 如果 M 的最后一行第 i 列为 0，则 j 减 1
        if M[M.rows - 1, i] == 0:
            j = j - 1
        else:
            # 否则，返回 j（即最后一个非零项的次数），如果 j 小于 0，则返回 0
            return j if j >= 0 else 0

# 定义一个函数，用于进行最后的修正操作。
def final_touches(s2, r, deg_g):
    """
    s2 is sylvester2, r is the row pointer in s2,
    deg_g is the degree of the poly last inserted in s2.

    After a gcd of degree > 0 has been found with Van Vleck's
    method, and was inserted into s2, if its last term is not
    in the last column of s2, then it is inserted as many
    times as needed, rotated right by one each time, until
    the condition is met.

    最终的修正操作函数。

    s2 是 sylvester2 矩阵，r 是 s2 中的行指针，
    deg_g 是最后插入 s2 中的多项式的次数。

    使用 Van Vleck 方法找到次数大于 0 的 gcd，并将其插入 s2 后，
    如果其最后一项不在 s2 的最后一列中，则旋转插入所需次数，
    每次向右旋转一次，直到满足条件。

    """
    # 获取 s2 的第 r-1 行
    R = s2.row(r-1)

    # 找到第一个非零项的位置 i
    for i in range(s2.cols):
        if R[0,i] == 0:
            continue
        else:
            break

    # 计算缺少的行数，直到最后一项在 s2 的最后一列
    mr = s2.cols - (i + deg_g + 1)

    # 将缺少的行插入，替换行中现有的条目
    i = 0
    # 当 `mr` 不等于零且 `r + i` 小于 `s2` 的行数时，执行循环
    while mr != 0 and r + i < s2.rows:
        # 将 `s2` 的第 `r + i` 行设为调用 `rotate_r(R, i + 1)` 后的结果
        s2[r + i, :] = rotate_r(R, i + 1)
        # 增加 `i` 的值，减少 `mr` 的值，以控制循环条件
        i += 1
        mr -= 1

    # 返回修改后的 `s2`
    return s2
# 定义一个函数 subresultants_vv，用于计算多项式 p, q 在整数或有理数系数下的子结果式 prs
def subresultants_vv(p, q, x, method = 0):
    """
    p, q are polynomials in Z[x] (intended) or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant prs of p, q by triangularizing,
    in Z[x] or in Q[x], all the smaller matrices encountered in the
    process of triangularizing sylvester2, Sylvester's matrix of 1853;
    see references 1 and 2 for Van Vleck's method. With each remainder,
    sylvester2 gets updated and is prepared to be printed if requested.

    If sylvester2 has small dimensions and you want to see the final,
    triangularized matrix use this version with method=1; otherwise,
    use either this version with method=0 (default) or the faster version,
    subresultants_vv_2(p, q, x), where sylvester2 is used implicitly.

    Sylvester's matrix sylvester1  is also used to compute one
    subresultant per remainder; namely, that of the leading
    coefficient, in order to obtain the correct sign and to
    force the remainder coefficients to become subresultants.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    If the final, triangularized matrix s2 is printed, then:
        (a) if deg(p) - deg(q) > 1 or deg( gcd(p, q) ) > 0, several
            of the last rows in s2 will remain unprocessed;
        (b) if deg(p) - deg(q) == 0, p will not appear in the final matrix.

    References
    ==========
    1. Akritas, A. G.: ``A new method for computing polynomial greatest
    common divisors and polynomial remainder sequences.''
    Numerische MatheMatik 52, 119-127, 1988.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On a Theorem
    by Van Vleck Regarding Sturm Sequences.''
    Serdica Journal of Computing, 7, No 4, 101-134, 2013.

    3. Akritas, A. G.:``Three New Methods for Computing Subresultant
    Polynomial Remainder Sequences (PRS's).'' Serdica Journal of Computing 9(1) (2015), 1-26.

    """
    # 确保 p 和 q 均不为零
    if p == 0 or q == 0:
        return [p, q]

    # 确保多项式 p 和 q 的次数设置正确
    f, g = p, q
    n = deg_f = degree(f, x)
    m = deg_g = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, deg_f, deg_g, f, g = m, n, deg_g, deg_f, g, f
    if n > 0 and m == 0:
        return [f, g]

    # 初始化
    s1 = sylvester(f, g, x, 1)  # 计算 Sylvester 矩阵 s1
    s2 = sylvester(f, g, x, 2)  # 计算 Sylvester 矩阵 s2
    sr_list = [f, g]            # 存储子结果式列表
    col_num = 2 * n             # s2 矩阵的列数

    # 创建两行多项式系数的矩阵 (row0, row1)
    row0 = Poly(f, x, domain = QQ).all_coeffs()  # 获取多项式 f 的所有系数
    leng0 = len(row0)
    for i in range(col_num - leng0):
        row0.append(0)  # 补充零系数，使行长度与矩阵列数相等
    row0 = Matrix([row0])  # 将系数列表转换为矩阵行向量
    row1 = Poly(g,x, domain = QQ).all_coeffs()  # 获取多项式 g 的所有系数
    leng1 = len(row1)
    for i in range(col_num - leng1):
        row1.append(0)  # 补充零系数，使行长度与矩阵列数相等
    row1 = Matrix([row1])  # 将系数列表转换为矩阵行向量

    # 对于 deg_f - deg_g == 1 的情况，初始化行指针 r；可能在后续重置
    r = 2

    # 根据多项式的次数修改 s2 矩阵的第一行
    # 如果 deg_f - deg_g 大于 1，则执行以下操作
    if deg_f - deg_g > 1:
        # 初始化 r 为 1
        r = 1
        # 替换 s2 中已有的条目，并且在行中插入 row0，插入次数为 deg_f - deg_g - 1，每次插入时进行旋转
        for i in range(deg_f - deg_g - 1):
            s2[r + i, :] = rotate_r(row0, i + 1)
        # 更新 r，增加 deg_f - deg_g - 1
        r = r + deg_f - deg_g - 1
        # 在 s2 中插入 row1，插入次数为 deg_f - deg_g，每次插入时进行旋转
        for i in range(deg_f - deg_g):
            s2[r + i, :] = rotate_r(row1, r + i)
        # 更新 r，增加 deg_f - deg_g
        r = r + deg_f - deg_g

    # 如果 deg_f - deg_g 等于 0
    if deg_f - deg_g == 0:
        # 初始化 r 为 0
        r = 0

    # 主循环
    while deg_g > 0:
        # 创建一个小矩阵 M，并将其三角化
        M = create_ma(deg_f, deg_g, row1, row0, col_num)
        # 只需要 M 的第一行和最后一行
        for i in range(deg_f - deg_g + 1):
            M1 = pivot(M, i, i)
            M = M1[:, :]

        # 将 M 的最后一行视为多项式，并找到其次数
        d = find_degree(M, deg_f)
        if d is None:
            break
        exp_deg = deg_g - 1

        # 计算一个行列式并生成系数的子结果
        sign_value = correct_sign(n, m, s1, exp_deg, exp_deg - d)
        poly = row2poly(M[M.rows - 1, :], d, x)
        temp2 = LC(poly, x)
        poly = simplify((poly / temp2) * sign_value)

        # 根据需要更新 s2，插入 M 的第一行
        row0 = M[0, :]
        for i in range(deg_g - d):
            s2[r + i, :] = rotate_r(row0, r + i)
        r = r + deg_g - d

        # 根据需要更新 s2，插入 M 的最后一行
        row1 = rotate_l(M[M.rows - 1, :], deg_f - d)
        row1 = (row1 / temp2) * sign_value
        for i in range(deg_g - d):
            s2[r + i, :] = rotate_r(row1, r + i)
        r = r + deg_g - d

        # 更新 degrees
        deg_f, deg_g = deg_g, d

        # 将 poly 与子结果系数附加到 sr_list 中
        sr_list.append(poly)

    # 最后处理以打印 s2 矩阵
    if method != 0 and s2.rows > 2:
        s2 = final_touches(s2, r, deg_g)
        pprint(s2)
    elif method != 0 and s2.rows == 2:
        s2[1, :] = rotate_r(s2.row(1), 1)
        pprint(s2)

    # 返回 sr_list
    return sr_list
# 定义一个函数用于计算两个多项式 p, q 的 Van Vleck 方法下的子结果列。
def subresultants_vv_2(p, q, x):
    """
    p, q are polynomials in Z[x] (intended) or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant prs of p, q by triangularizing,
    in Z[x] or in Q[x], all the smaller matrices encountered in the
    process of triangularizing sylvester2, Sylvester's matrix of 1853;
    see references 1 and 2 for Van Vleck's method.

    If the sylvester2 matrix has big dimensions use this version,
    where sylvester2 is used implicitly. If you want to see the final,
    triangularized matrix sylvester2, then use the first version,
    subresultants_vv(p, q, x, 1).

    sylvester1, Sylvester's matrix of 1840, is also used to compute
    one subresultant per remainder; namely, that of the leading
    coefficient, in order to obtain the correct sign and to
    ``force'' the remainder coefficients to become subresultants.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G.: ``A new method for computing polynomial greatest
    common divisors and polynomial remainder sequences.''
    Numerische MatheMatik 52, 119-127, 1988.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On a Theorem
    by Van Vleck Regarding Sturm Sequences.''
    Serdica Journal of Computing, 7, No 4, 101-134, 2013.

    3. Akritas, A. G.:``Three New Methods for Computing Subresultant
    Polynomial Remainder Sequences (PRS's).'' Serdica Journal of Computing 9(1) (2015), 1-26.

    """
    # 确保 p 和 q 均不为零多项式，否则直接返回它们
    if p == 0 or q == 0:
        return [p, q]

    # 确保 p 的次数大于等于 q 的次数
    f, g = p, q
    n = deg_f = degree(f, x)
    m = deg_g = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, deg_f, deg_g, f, g = m, n, deg_g, deg_f, g, f
    if n > 0 and m == 0:
        return [f, g]

    # 初始化
    s1 = sylvester(f, g, x, 1)  # 计算 Sylvester 矩阵 s1
    sr_list = [f, g]  # 子结果列初始化，包含 f 和 g
    col_num = 2 * n  # sylvester2 中的列数为 2 * n

    # 创建两行多项式系数的矩阵表示 (row0, row1)
    row0 = Poly(f, x, domain=QQ).all_coeffs()  # 将多项式 f 的系数转换为列表
    leng0 = len(row0)
    for i in range(col_num - leng0):
        row0.append(0)  # 如果长度不足 col_num，则补齐零
    row0 = Matrix([row0])  # 转换为 SymPy 的 Matrix 对象
    row1 = Poly(g, x, domain=QQ).all_coeffs()  # 将多项式 g 的系数转换为列表
    leng1 = len(row1)
    for i in range(col_num - leng1):
        row1.append(0)  # 如果长度不足 col_num，则补齐零
    row1 = Matrix([row1])  # 转换为 SymPy 的 Matrix 对象

    # 主循环开始
    while deg_g > 0:
        # 当 deg_g 大于 0 时执行以下循环，进行多项式的辗转相除过程

        # 创建一个大小适当的矩阵 M，并将其三角化
        M = create_ma(deg_f, deg_g, row1, row0, col_num)

        # 对于每个 i 在范围内，进行主元选取操作
        for i in range(deg_f - deg_g + 1):
            M1 = pivot(M, i, i)
            M = M1[:, :]

        # 将 M 的最后一行视为多项式，找到其次数
        d = find_degree(M, deg_f)
        if d is None:
            return sr_list

        # 计算一个行列式并生成系数的子剩余
        sign_value = correct_sign(n, m, s1, exp_deg, exp_deg - d)
        poly = row2poly(M[M.rows - 1, :], d, x)
        poly = simplify((poly / LC(poly, x)) * sign_value)

        # 将生成的子剩余多项式添加到 sr_list 中
        sr_list.append(poly)

        # 更新多项式的次数和矩阵的行数
        deg_f, deg_g = deg_g, d
        row0 = row1
        row1 = Poly(poly, x, domain=QQ).all_coeffs()
        leng1 = len(row1)
        for i in range(col_num - leng1):
            row1.append(0)
        row1 = Matrix([row1])

    # 返回生成的子剩余列表 sr_list
    return sr_list
```