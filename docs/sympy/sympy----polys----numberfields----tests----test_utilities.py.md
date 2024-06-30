# `D:\src\scipysrc\sympy\sympy\polys\numberfields\tests\test_utilities.py`

```
from sympy.abc import x
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.domains import FF, QQ
from sympy.polys.matrices import DomainMatrix, DM
from sympy.polys.matrices.exceptions import DMRankError
from sympy.polys.numberfields.utilities import (
    AlgIntPowers, coeff_search, extract_fundamental_discriminant,
    isolate, supplement_a_subspace,
)
from sympy.printing.lambdarepr import IntervalPrinter
from sympy.testing.pytest import raises

# 定义测试函数 test_AlgIntPowers_01
def test_AlgIntPowers_01():
    # 创建一个多项式 T，表示第五个旋转多项式
    T = Poly(cyclotomic_poly(5))
    # 创建 AlgIntPowers 对象，用于计算 T 的代数整数幂
    zeta_pow = AlgIntPowers(T)
    # 检查索引为负的操作是否引发 ValueError 异常
    raises(ValueError, lambda: zeta_pow[-1])
    # 对前 10 个指数进行循环
    for e in range(10):
        a = e % 5
        if a < 4:
            # 计算 zeta_pow[e]，并验证其符合期望的特定形式
            c = zeta_pow[e]
            assert c[a] == 1 and all(c[i] == 0 for i in range(4) if i != a)
        else:
            # 当 a 大于等于 4 时，zeta_pow[e] 应为 [-1, -1, -1, -1]
            assert zeta_pow[e] == [-1] * 4

# 定义测试函数 test_AlgIntPowers_02
def test_AlgIntPowers_02():
    # 创建多项式 T = x^3 + 2*x^2 + 3*x + 4
    T = Poly(x**3 + 2*x**2 + 3*x + 4)
    # 设置整数 m = 7
    m = 7
    # 创建 AlgIntPowers 对象，用于计算 T 的代数整数幂，取模 m
    theta_pow = AlgIntPowers(T, m)
    # 对前 10 个指数进行循环
    for e in range(10):
        # 计算 theta_pow[e] 的值
        computed = theta_pow[e]
        # 计算预期的结果 expected
        coeffs = (Poly(x)**e % T + Poly(x**3)).rep.to_list()[1:]
        expected = [c % m for c in reversed(coeffs)]
        # 断言 computed 和 expected 相等
        assert computed == expected

# 定义测试函数 test_coeff_search
def test_coeff_search():
    # 初始化空列表 C
    C = []
    # 调用 coeff_search 函数，查找满足条件的前 13 个系数对
    search = coeff_search(2, 1)
    for i, c in enumerate(search):
        C.append(c)
        # 当 i 达到 12 时停止循环
        if i == 12:
            break
    # 断言 C 等于预期的系数对列表
    assert C == [[1, 1], [1, 0], [1, -1], [0, 1], [2, 2], [2, 1], [2, 0], [2, -1], [2, -2], [1, 2], [1, -2], [0, 2], [3, 3]]

# 定义测试函数 test_extract_fundamental_discriminant
def test_extract_fundamental_discriminant():
    # 测试抛出 ValueError 异常，当输入参数不满足条件时
    raises(ValueError, lambda: extract_fundamental_discriminant(2))
    raises(ValueError, lambda: extract_fundamental_discriminant(3))
    # 测试多种情况，验证 extract_fundamental_discriminant 函数的输出是否符合预期
    cases = (
        (0, {}, {0: 1}),
        (1, {}, {}),
        (8, {2: 3}, {}),
        (-8, {2: 3, -1: 1}, {}),
        (12, {2: 2, 3: 1}, {}),
        (36, {}, {2: 1, 3: 1}),
        (45, {5: 1}, {3: 1}),
        (48, {2: 2, 3: 1}, {2: 1}),
        (1125, {5: 1}, {3: 1, 5: 1}),
    )
    for a, D_expected, F_expected in cases:
        # 调用 extract_fundamental_discriminant 函数
        D, F = extract_fundamental_discriminant(a)
        # 断言返回的 D 和 F 与预期结果相等
        assert D == D_expected
        assert F == F_expected

# 定义测试函数 test_supplement_a_subspace_1
def test_supplement_a_subspace_1():
    # 创建一个有理数域上的域矩阵 M
    M = DM([[1, 7, 0], [2, 3, 4]], QQ).transpose()
    # 在有理数域 QQ 上补充子空间
    B = supplement_a_subspace(M)
    # 断言补充后的矩阵 B 符合预期
    assert B[:, :2] == M
    assert B[:, 2] == DomainMatrix.eye(3, QQ).to_dense()[:, 0]

    # 将矩阵 M 转换为有限域 FF(7) 上的矩阵
    M = M.convert_to(FF(7))
    # 在有限域 FF(7) 上补充子空间
    B = supplement_a_subspace(M)
    # 断言补充后的矩阵 B 符合预期
    assert B[:, :2] == M
    # 当在模 7 的情况下工作时，M 的第一列变为 [1, 0, 0]
    # 因此补充向量不等于第一列，而是等于第二个标准基向量
    assert B[:, 2] == DomainMatrix.eye(3, FF(7)).to_dense()[:, 1]

# 定义测试函数 test_supplement_a_subspace_2
def test_supplement_a_subspace_2():
    # 该函数尚未实现，留待后续补充
    pass
    # 创建一个有理数域 QQ 上的矩阵 M，其值为 [[1, 0, 0], [2, 0, 0]] 的转置
    M = DM([[1, 0, 0], [2, 0, 0]], QQ).transpose()
    
    # 使用 pytest 的 raises 函数检查是否会抛出 DMRankError 异常
    with raises(DMRankError):
        # 调用 supplement_a_subspace 函数，期望它在给定的矩阵 M 上抛出 DMRankError 异常
        supplement_a_subspace(M)
# 定义测试函数 test_IntervalPrinter，用于测试 IntervalPrinter 类的功能
def test_IntervalPrinter():
    # 创建 IntervalPrinter 类的实例对象 ip
    ip = IntervalPrinter()
    # 断言调用 ip 对象的 doprint 方法对 x**Rational(1, 3) 进行处理得到的字符串为 "x**(mpi('1/3'))"
    assert ip.doprint(x**Rational(1, 3)) == "x**(mpi('1/3'))"
    # 断言调用 ip 对象的 doprint 方法对 sqrt(x) 进行处理得到的字符串为 "x**(mpi('1/2'))"
    assert ip.doprint(sqrt(x)) == "x**(mpi('1/2'))"

# 定义测试函数 test_isolate，用于测试 isolate 函数的功能
def test_isolate():
    # 断言调用 isolate 函数处理参数 1 得到的结果是 (1, 1)
    assert isolate(1) == (1, 1)
    # 断言调用 isolate 函数处理参数 S.Half 得到的结果是 (S.Half, S.Half)
    assert isolate(S.Half) == (S.Half, S.Half)

    # 断言调用 isolate 函数处理参数 sqrt(2) 得到的结果是 (1, 2)
    assert isolate(sqrt(2)) == (1, 2)
    # 断言调用 isolate 函数处理参数 -sqrt(2) 得到的结果是 (-2, -1)
    assert isolate(-sqrt(2)) == (-2, -1)

    # 断言调用 isolate 函数处理参数 sqrt(2)，设置精度 eps=Rational(1, 100)，得到的结果是 (Rational(24, 17), Rational(17, 12))
    assert isolate(sqrt(2), eps=Rational(1, 100)) == (Rational(24, 17), Rational(17, 12))
    # 断言调用 isolate 函数处理参数 -sqrt(2)，设置精度 eps=Rational(1, 100)，得到的结果是 (Rational(-17, 12), Rational(-24, 17))
    assert isolate(-sqrt(2), eps=Rational(1, 100)) == (Rational(-17, 12), Rational(-24, 17))

    # 断言调用 lambda 匿名函数捕获 isolate 函数对参数 I 的处理会抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: isolate(I))
```