# `D:\src\scipysrc\sympy\sympy\polys\numberfields\tests\test_primes.py`

```
# 从 math 模块导入 prod 函数
from math import prod

# 从 sympy 模块导入 QQ 和 ZZ 对象
from sympy import QQ, ZZ

# 从 sympy.abc 模块导入 x 和 theta 符号
from sympy.abc import x, theta

# 从 sympy.ntheory 模块导入 factorint 和 n_order 函数
from sympy.ntheory import factorint
from sympy.ntheory.residue_ntheory import n_order

# 从 sympy.polys 模块导入 Poly 和 cyclotomic_poly 函数
from sympy.polys import Poly, cyclotomic_poly

# 从 sympy.polys.matrices 模块导入 DomainMatrix 类
from sympy.polys.matrices import DomainMatrix

# 从 sympy.polys.numberfields.basis 模块导入 round_two 函数
from sympy.polys.numberfields.basis import round_two

# 从 sympy.polys.numberfields.exceptions 模块导入 StructureError 异常类
from sympy.polys.numberfields.exceptions import StructureError

# 从 sympy.polys.numberfields.modules 模块导入 PowerBasis 和 to_col 函数
from sympy.polys.numberfields.modules import PowerBasis, to_col

# 从 sympy.polys.numberfields.primes 模块导入 prime_decomp 和 _two_elt_rep 函数
from sympy.polys.numberfields.primes import (
    prime_decomp, _two_elt_rep,
    _check_formal_conditions_for_maximal_order,
)

# 从 sympy.testing.pytest 模块导入 raises 函数
from sympy.testing.pytest import raises


# 定义测试函数，用于检查 _check_formal_conditions_for_maximal_order 函数的行为
def test_check_formal_conditions_for_maximal_order():
    # 创建一个 cyclotomic_poly 多项式 T，其中参数为 5
    T = Poly(cyclotomic_poly(5, x))
    # 创建 T 的 PowerBasis
    A = PowerBasis(T)
    # 创建 A 的子模块 B，该子模块基于 DomainMatrix 的单位矩阵乘以 2
    B = A.submodule_from_matrix(2 * DomainMatrix.eye(4, ZZ))
    # 创建 B 的子模块 C，该子模块基于 DomainMatrix 的单位矩阵乘以 3
    C = B.submodule_from_matrix(3 * DomainMatrix.eye(4, ZZ))
    # 创建 A 的子模块 D，该子模块基于 DomainMatrix 的单位矩阵的前 3 列
    D = A.submodule_from_matrix(DomainMatrix.eye(4, ZZ)[:, :-1])
    
    # 对于 B，检测是否引发 StructureError 异常
    raises(StructureError, lambda: _check_formal_conditions_for_maximal_order(B))
    # 对于 C，检测是否引发 StructureError 异常
    raises(StructureError, lambda: _check_formal_conditions_for_maximal_order(C))
    # 对于 D，检测是否引发 StructureError 异常
    raises(StructureError, lambda: _check_formal_conditions_for_maximal_order(D))


# 定义测试函数，用于测试 _two_elt_rep 函数
def test_two_elt_rep():
    # 定义素数 ell
    ell = 7
    # 创建一个 cyclotomic_poly 多项式 T，其参数为 ell
    T = Poly(cyclotomic_poly(ell))
    # 调用 round_two 函数获取 ZK 和 dK
    ZK, dK = round_two(T)
    
    # 遍历一组素数
    for p in [29, 13, 11, 5]:
        # 对于素数 p，计算其在 T 中的素分解
        P = prime_decomp(p, T)
        # 遍历每个 Pi
        for Pi in P:
            # 创建 H，作为 p*ZK + Pi.alpha*ZK 的表示
            H = p*ZK + Pi.alpha*ZK
            # 获取 H 的基向量
            gens = H.basis_element_pullbacks()
            # 使用 gens 和 ZK 计算 _two_elt_rep
            b = _two_elt_rep(gens, ZK, p)
            # 断言结果 b 与 Pi.alpha 相等
            if b != Pi.alpha:
                # 如果不相等，再次构建 H2，并断言 H2 与 H 相等
                H2 = p*ZK + b*ZK
                assert H2 == H


# 定义测试函数，用于测试 prime_decomp 函数中的 valuation 方法
def test_valuation_at_prime_ideal():
    # 定义素数 p
    p = 7
    # 创建一个 cyclotomic_poly 多项式 T，其参数为 p
    T = Poly(cyclotomic_poly(p))
    # 调用 round_two 函数获取 ZK 和 dK
    ZK, dK = round_two(T)
    # 计算 p 在 T 中的素分解
    P = prime_decomp(p, T, dK=dK, ZK=ZK)
    # 断言分解列表 P 的长度为 1
    assert len(P) == 1
    # 获取分解结果中的第一个元素 P0
    P0 = P[0]
    # 使用 valuation 方法计算 P0 在 p*ZK 上的估值
    v = P0.valuation(p*ZK)
    # 断言计算结果 v 等于 P0 的 e 属性
    assert v == P0.e
    # 测试 easy 0 情况：
    # 断言 P0 在 5*ZK 上的估值为 0
    assert P0.valuation(5*ZK) == 0


# 定义测试函数，用于测试 prime_decomp 函数的行为
def test_decomp_1():
    # 在所有 cyclotomic 字段中，所有素数分解都处于 "easy case"，
    # 因为其指数为单位。
    # 在这里我们检查一个分裂素数的情况。
    # 创建一个 cyclotomic_poly 多项式 T，其参数为 7
    T = Poly(cyclotomic_poly(7))
    # 使用 lambda 函数检测是否引发 ValueError 异常
    raises(ValueError, lambda: prime_decomp(7))
    # 计算 T 的素数分解
    P = prime_decomp(7, T)
    # 断言分解列表 P 的长度为 1
    assert len(P) == 1
    # 获取分解结果中的第一个元素 P0
    P0 = P[0]
    # 断言 P0 的 e 属性等于 6
    assert P0.e == 6
    # 断言 P0 的 f 属性等于 1
    assert P0.f == 1
    # 测试幂运算：
    # 断言 P0 的 0 次幂等于 P0.ZK
    assert P0**0 == P0.ZK
    # 断言 P0 的一次方等于 P0 本身
    assert P0**1 == P0
    # 断言 P0 的六次方等于 7 乘以 P0 的 ZK 属性
    assert P0**6 == 7 * P0.ZK
def test_decomp_2():
    # 更简单的旋转幂系数情况，但在这里我们检查非分裂的素数。
    ell = 7
    # 构造一个多项式对象，表示旋转幂多项式
    T = Poly(cyclotomic_poly(ell))
    # 对每个素数 p 进行循环
    for p in [29, 13, 11, 5]:
        # 计算 n_order(p, ell) 的结果，作为指数 f_exp
        f_exp = n_order(p, ell)
        # 计算 (ell - 1) // f_exp 的结果，作为指数 g_exp
        g_exp = (ell - 1) // f_exp
        # 对素数 p 进行分解，返回一个列表 P
        P = prime_decomp(p, T)
        # 断言列表 P 的长度等于 g_exp
        assert len(P) == g_exp
        # 对列表 P 中的每个 Pi 进行断言，确保 Pi.e 等于 1
        for Pi in P:
            assert Pi.e == 1
            # 断言 Pi.f 等于 f_exp
            assert Pi.f == f_exp


def test_decomp_3():
    # 创建一个多项式对象 T，表示 x^2 - 35
    T = Poly(x ** 2 - 35)
    # 创建一个空字典 rad
    rad = {}
    # 调用 round_two 函数，返回 ZK 和 dK 的值
    ZK, dK = round_two(T, radicals=rad)
    # 35 除以 4 余 3，因此域的判别式为 4*5*7，理论上每个有理素数 2、5、7 应该是一个素理想的平方。
    # 对每个素数 p 进行循环
    for p in [2, 5, 7]:
        # 对素数 p 进行分解，返回一个列表 P
        P = prime_decomp(p, T, dK=dK, ZK=ZK, radical=rad.get(p))
        # 断言列表 P 的长度为 1
        assert len(P) == 1
        # 断言 P[0].e 等于 2
        assert P[0].e == 2
        # 断言 P[0]**2 等于 p*ZK
        assert P[0]**2 == p*ZK


def test_decomp_4():
    # 创建一个多项式对象 T，表示 x^2 - 21
    T = Poly(x ** 2 - 21)
    # 创建一个空字典 rad
    rad = {}
    # 调用 round_two 函数，返回 ZK 和 dK 的值
    ZK, dK = round_two(T, radicals=rad)
    # 21 除以 4 余 1，因此域的判别式为 3*7，理论上每个有理素数 3、7 应该是一个素理想的平方。
    # 对每个素数 p 进行循环
    for p in [3, 7]:
        # 对素数 p 进行分解，返回一个列表 P
        P = prime_decomp(p, T, dK=dK, ZK=ZK, radical=rad.get(p))
        # 断言列表 P 的长度为 1
        assert len(P) == 1
        # 断言 P[0].e 等于 2
        assert P[0].e == 2
        # 断言 P[0]**2 等于 p*ZK
        assert P[0]**2 == p*ZK


def test_decomp_5():
    # 这是我们首次测试素数分解的“困难情况”。
    # 我们在一个二次扩展 Q(sqrt(d)) 中工作，其中 d 为 1 mod 4，
    # 我们考虑有理素数 2 的分解，它除以指数。
    # 理论上说 p 的分解形式取决于 d 对 8 的余数，因此我们考虑 d = 1 mod 8 和 d = 5 mod 8 两种情况。
    for d in [-7, -3]:
        # 创建一个多项式对象 T，表示 x^2 - d
        T = Poly(x ** 2 - d)
        # 创建一个空字典 rad
        rad = {}
        # 调用 round_two 函数，返回 ZK 和 dK 的值
        ZK, dK = round_two(T, radicals=rad)
        p = 2
        # 对素数 p 进行分解，返回一个列表 P
        P = prime_decomp(p, T, dK=dK, ZK=ZK, radical=rad.get(p))
        if d % 8 == 1:
            # 如果 d 对 8 取余为 1，断言列表 P 的长度为 2
            assert len(P) == 2
            # 断言列表 P 中每个 Pi 的 e 和 f 都等于 1
            assert all(P[i].e == 1 and P[i].f == 1 for i in range(2))
            # 断言 P 中每个 Pi 的 Pi**Pi.e 的乘积等于 p * ZK
            assert prod(Pi**Pi.e for Pi in P) == p * ZK
        else:
            # 如果 d 对 8 取余不为 1，断言 d 对 8 取余为 5
            assert d % 8 == 5
            # 断言列表 P 的长度为 1
            assert len(P) == 1
            # 断言 P[0].e 等于 1
            assert P[0].e == 1
            # 断言 P[0].f 等于 2
            assert P[0].f == 2
            # 断言 P[0].as_submodule() 等于 p * ZK
            assert P[0].as_submodule() == p * ZK


def test_decomp_6():
    # 另一个情况是 2 除以指数的案例。这是戴德金的一个必要判别因子的例子。（参见科恩，练习 6.10。）
    # 创建一个多项式对象 T，表示 x^3 + x^2 - 2*x + 8
    T = Poly(x ** 3 + x ** 2 - 2 * x + 8)
    # 创建一个空字典 rad
    rad = {}
    # 调用 round_two 函数，返回 ZK 和 dK 的值
    ZK, dK = round_two(T, radicals=rad)
    p = 2
    # 对素数 p 进行分解，返回一个列表 P
    P = prime_decomp(p, T, dK=dK, ZK=ZK, radical=rad.get(p))
    # 断言列表 P 的长度为 3
    assert len(P) == 3
    # 断言列表 P 中每个 Pi 的 e 和 f 都等于 1
    assert all(Pi.e == 1 and Pi.f == 1 for Pi in P)
    # 断言 P 中每个 Pi 的 Pi**Pi.e 的乘积等于 p * ZK
    assert prod(Pi**Pi.e for Pi in P) == p * ZK


def test_decomp_7():
    # 尝试通过代数域进行操作
    # 创建一个多项式对象 T，表示 x^3 + x^2 - 2*x + 8
    T = Poly(x ** 3 + x ** 2 - 2 * x + 8)
    # 从多项式 T 构造一个代数域 K
    K = QQ.alg_field_from_poly(T)
    p = 2
    # 获取素数 p 在 K 中的所有理想
    P = K.primes_above(p)
    # 获取 K 的最大秩 ZK
    ZK = K.maximal_order()
    # 断言列表 P 的长度为 3
    assert len(P) == 3
    # 断言列表 P 中每个 Pi 的 e 和 f 都等于 1
    assert all(Pi.e == 1 and Pi.f == 1 for Pi in P)
    # 断言 P 中每个 Pi 的 Pi**Pi.e 的乘积等于 p * ZK
    assert prod(Pi**Pi.e for Pi in P) == p * ZK


def test_decomp_8():
    # 在这里添加下
    # This time we consider various cubics, and try factoring all primes
    # dividing the index.
    cases = (
        x ** 3 + 3 * x ** 2 - 4 * x + 4,   # Define a tuple of cubic polynomials to analyze
        x ** 3 + 3 * x ** 2 + 3 * x - 3,   # Each polynomial is a function of variable x
        x ** 3 + 5 * x ** 2 - x + 3,       # These expressions are to be evaluated for their roots
        x ** 3 + 5 * x ** 2 - 5 * x - 5,   # and factorization properties.
        x ** 3 + 3 * x ** 2 + 5,           # They represent different forms of cubic equations.
        x ** 3 + 6 * x ** 2 + 3 * x - 1,
        x ** 3 + 6 * x ** 2 + 4,
        x ** 3 + 7 * x ** 2 + 7 * x - 7,
        x ** 3 + 7 * x ** 2 - x + 5,
        x ** 3 + 7 * x ** 2 - 5 * x + 5,
        x ** 3 + 4 * x ** 2 - 3 * x + 7,
        x ** 3 + 8 * x ** 2 + 5 * x - 1,
        x ** 3 + 8 * x ** 2 - 2 * x + 6,
        x ** 3 + 6 * x ** 2 - 3 * x + 8,
        x ** 3 + 9 * x ** 2 + 6 * x - 8,
        x ** 3 + 15 * x ** 2 - 9 * x + 13,
    )

    def display(T, p, radical, P, I, J):
        """Useful for inspection, when running test manually."""
        # Print various details for inspection purposes
        print('=' * 20)
        print(T, p, radical)
        for Pi in P:
            print(f'  ({Pi!r})')
        print("I: ", I)
        print("J: ", J)
        print(f'Equal: {I == J}')

    inspect = False  # Flag to control whether to display detailed inspection information
    for g in cases:
        T = Poly(g)  # Create a polynomial object T from each expression g
        rad = {}     # Initialize an empty dictionary for radicals
        ZK, dK = round_two(T, radicals=rad)  # Compute ZK and dK using round_two function
        dT = T.discriminant()  # Compute discriminant of polynomial T
        f_squared = dT // dK   # Calculate f_squared as the quotient of discriminant and dK
        F = factorint(f_squared)  # Factorize f_squared into its prime factors
        for p in F:
            radical = rad.get(p)  # Retrieve radical value from dictionary rad if exists
            P = prime_decomp(p, T, dK=dK, ZK=ZK, radical=radical)  # Decompose prime p into components
            I = prod(Pi**Pi.e for Pi in P)  # Calculate product I from components Pi in P
            J = p * ZK  # Calculate J as p times ZK
            if inspect:
                display(T, p, radical, P, I, J)  # Optionally display detailed inspection information
            assert I == J  # Assert that I equals J
def test_PrimeIdeal_eq():
    # `==` should fail on objects of different types, so even a completely
    # inert PrimeIdeal should test unequal to the rational prime it divides.
    T = Poly(cyclotomic_poly(7))
    # Decompose the prime number 5 in the polynomial T and select the first prime ideal
    P0 = prime_decomp(5, T)[0]
    # Assert that the residue degree f of P0 is 6
    assert P0.f == 6
    # Assert that P0 as a submodule equals 5 times the integral closure of ZK
    assert P0.as_submodule() == 5 * P0.ZK
    # Assert that P0 is not equal to the integer 5
    assert P0 != 5


def test_PrimeIdeal_add():
    T = Poly(cyclotomic_poly(7))
    # Decompose the prime number 7 in the polynomial T and select the first prime ideal
    P0 = prime_decomp(7, T)[0]
    # Adding ideals computes their GCD, so adding the ramified prime dividing
    # 7 to 7 itself should reproduce this prime (as a submodule).
    assert P0 + 7 * P0.ZK == P0.as_submodule()


def test_str():
    # Without alias:
    k = QQ.alg_field_from_poly(Poly(x**2 + 7))
    # Get the prime above 2 in field k and select the first one
    frp = k.primes_above(2)[0]
    # Assert the string representation of frp
    assert str(frp) == '(2, 3*_x/2 + 1/2)'

    # Get the prime above 3 in field k and select the first one
    frp = k.primes_above(3)[0]
    # Assert the string representation of frp
    assert str(frp) == '(3)'

    # With alias:
    k = QQ.alg_field_from_poly(Poly(x ** 2 + 7), alias='alpha')
    # Get the prime above 2 in field k and select the first one
    frp = k.primes_above(2)[0]
    # Assert the string representation of frp with alias alpha
    assert str(frp) == '(2, 3*alpha/2 + 1/2)'

    # Get the prime above 3 in field k and select the first one
    frp = k.primes_above(3)[0]
    # Assert the string representation of frp with alias alpha
    assert str(frp) == '(3)'


def test_repr():
    T = Poly(x**2 + 7)
    ZK, dK = round_two(T)
    # Decompose the prime number 2 in the polynomial T with additional parameters
    P = prime_decomp(2, T, dK=dK, ZK=ZK)
    # Assert the representation of the first element of P
    assert repr(P[0]) == '[ (2, (3*x + 1)/2) e=1, f=1 ]'
    # Assert the representation of the first element of P with a specified field generator
    assert P[0].repr(field_gen=theta) == '[ (2, (3*theta + 1)/2) e=1, f=1 ]'
    # Assert the representation of the first element of P with a specified field generator and just the generators
    assert P[0].repr(field_gen=theta, just_gens=True) == '(2, (3*theta + 1)/2)'


def test_PrimeIdeal_reduce():
    k = QQ.alg_field_from_poly(Poly(x ** 3 + x ** 2 - 2 * x + 8))
    Zk = k.maximal_order()
    # Get the primes above 2 in field k and select the third one
    P = k.primes_above(2)
    frp = P[2]

    # reduce_element
    # Create an algebraic number from a vector using the parent method
    a = Zk.parent(to_col([23, 20, 11]), denom=6)
    # Expected reduced element using the parent method with different vector values
    a_bar_expected = Zk.parent(to_col([11, 5, 2]), denom=6)
    # Reduce a using frp's reduce_element method and assert equality with a_bar_expected
    a_bar = frp.reduce_element(a)
    assert a_bar == a_bar_expected

    # reduce_ANP
    # Create an algebraic number from a vector of rational numbers
    a = k([QQ(11, 6), QQ(20, 6), QQ(23, 6)])
    # Expected reduced algebraic number
    a_bar_expected = k([QQ(2, 6), QQ(5, 6), QQ(11, 6)])
    # Reduce a using frp's reduce_ANP method and assert equality with a_bar_expected
    a_bar = frp.reduce_ANP(a)
    assert a_bar == a_bar_expected

    # reduce_alg_num
    # Convert a to an algebraic number
    a = k.to_alg_num(a)
    # Convert a_bar_expected to an algebraic number
    a_bar_expected = k.to_alg_num(a_bar_expected)
    # Reduce a using frp's reduce_alg_num method and assert equality with a_bar_expected
    a_bar = frp.reduce_alg_num(a)
    assert a_bar == a_bar_expected


def test_issue_23402():
    k = QQ.alg_field_from_poly(Poly(x ** 3 + x ** 2 - 2 * x + 8))
    # Get the primes above 3 in field k and select the first one
    P = k.primes_above(3)
    # Assert that the alpha attribute of the first element of P is equivalent to 0
    assert P[0].alpha.equiv(0)
```