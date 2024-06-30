# `D:\src\scipysrc\sympy\sympy\polys\tests\test_rootisolation.py`

```
"""Tests for real and complex root isolation and refinement algorithms. """

# 导入需要的模块和函数
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, ZZ_I, EX
from sympy.polys.polyerrors import DomainError, RefinementFailed, PolynomialError
from sympy.polys.rootisolation import (
    dup_cauchy_upper_bound, dup_cauchy_lower_bound,
    dup_mignotte_sep_bound_squared,
)
# 导入测试框架中的 raises 函数
from sympy.testing.pytest import raises

# 定义测试函数 test_dup_sturm
def test_dup_sturm():
    # 创建有理数域 QQ 上的多项式环 R，并定义变量 x
    R, x = ring("x", QQ)

    # 断言语句，验证 dup_sturm 函数对整数和变量的返回结果
    assert R.dup_sturm(5) == [1]
    assert R.dup_sturm(x) == [x, 1]

    # 定义多项式 f，并验证 dup_sturm 函数对多项式的返回结果
    f = x**3 - 2*x**2 + 3*x - 5
    assert R.dup_sturm(f) == [f, 3*x**2 - 4*x + 3, -QQ(10,9)*x + QQ(13,3), -QQ(3303,100)]


# 定义测试函数 test_dup_cauchy_upper_bound
def test_dup_cauchy_upper_bound():
    # 测试空列表和单项式时是否会抛出异常
    raises(PolynomialError, lambda: dup_cauchy_upper_bound([], QQ))
    raises(PolynomialError, lambda: dup_cauchy_upper_bound([QQ(1)], QQ))
    # 测试包含虚数的列表时是否会抛出异常
    raises(DomainError, lambda: dup_cauchy_upper_bound([ZZ_I(1), ZZ_I(1)], ZZ_I))

    # 断言语句，验证 dup_cauchy_upper_bound 函数的返回结果
    assert dup_cauchy_upper_bound([QQ(1), QQ(0), QQ(0)], QQ) == QQ.zero
    assert dup_cauchy_upper_bound([QQ(1), QQ(0), QQ(-2)], QQ) == QQ(3)


# 定义测试函数 test_dup_cauchy_lower_bound
def test_dup_cauchy_lower_bound():
    # 测试空列表和单项式时是否会抛出异常
    raises(PolynomialError, lambda: dup_cauchy_lower_bound([], QQ))
    raises(PolynomialError, lambda: dup_cauchy_lower_bound([QQ(1)], QQ))
    raises(PolynomialError, lambda: dup_cauchy_lower_bound([QQ(1), QQ(0), QQ(0)], QQ))
    # 测试包含虚数的列表时是否会抛出异常
    raises(DomainError, lambda: dup_cauchy_lower_bound([ZZ_I(1), ZZ_I(1)], ZZ_I))

    # 断言语句，验证 dup_cauchy_lower_bound 函数的返回结果
    assert dup_cauchy_lower_bound([QQ(1), QQ(0), QQ(-2)], QQ) == QQ(2, 3)


# 定义测试函数 test_dup_mignotte_sep_bound_squared
def test_dup_mignotte_sep_bound_squared():
    # 测试空列表和单项式时是否会抛出异常
    raises(PolynomialError, lambda: dup_mignotte_sep_bound_squared([], QQ))
    raises(PolynomialError, lambda: dup_mignotte_sep_bound_squared([QQ(1)], QQ))

    # 断言语句，验证 dup_mignotte_sep_bound_squared 函数的返回结果
    assert dup_mignotte_sep_bound_squared([QQ(1), QQ(0), QQ(-2)], QQ) == QQ(3, 5)


# 定义测试函数 test_dup_refine_real_root
def test_dup_refine_real_root():
    # 创建整数域 ZZ 上的多项式环 R，并定义变量 x
    R, x = ring("x", ZZ)
    f = x**2 - 2

    # 断言语句，验证 dup_refine_real_root 函数在不同步数下的返回结果
    assert R.dup_refine_real_root(f, QQ(1), QQ(1), steps=1) == (QQ(1), QQ(1))
    assert R.dup_refine_real_root(f, QQ(1), QQ(1), steps=9) == (QQ(1), QQ(1))

    # 验证函数在给定边界外的情况下是否会抛出 ValueError 异常
    raises(ValueError, lambda: R.dup_refine_real_root(f, QQ(-2), QQ(2)))

    s, t = QQ(1, 1), QQ(2, 1)

    # 断言语句，验证 dup_refine_real_root 函数在不同步数下的返回结果
    assert R.dup_refine_real_root(f, s, t, steps=0) == (QQ(1, 1), QQ(2, 1))
    assert R.dup_refine_real_root(f, s, t, steps=1) == (QQ(1, 1), QQ(3, 2))
    assert R.dup_refine_real_root(f, s, t, steps=2) == (QQ(4, 3), QQ(3, 2))
    assert R.dup_refine_real_root(f, s, t, steps=3) == (QQ(7, 5), QQ(3, 2))
    assert R.dup_refine_real_root(f, s, t, steps=4) == (QQ(7, 5), QQ(10, 7))

    s, t = QQ(1, 1), QQ(3, 2)

    # 断言语句，验证 dup_refine_real_root 函数在不同步数下的返回结果
    assert R.dup_refine_real_root(f, s, t, steps=0) == (QQ(1, 1), QQ(3, 2))
    assert R.dup_refine_real_root(f, s, t, steps=1) == (QQ(4, 3), QQ(3, 2))
    assert R.dup_refine_real_root(f, s, t, steps=2) == (QQ(7, 5), QQ(3, 2))
    assert R.dup_refine_real_root(f, s, t, steps=3) == (QQ(7, 5), QQ(10, 7))
    assert R.dup_refine_real_root(f, s, t, steps=4) == (QQ(7, 5), QQ(17, 12))

    s, t = QQ(1, 1), QQ(5, 3)
    # 检查使用函数 `dup_refine_real_root` 对给定函数 `f` 和区间 `(s, t)` 进行根的精细化求解，期望结果是 (QQ(1, 1), QQ(5, 3))
    assert R.dup_refine_real_root(f, s, t, steps=0) == (QQ(1, 1), QQ(5, 3))
    
    # 通过增加一次精细化步骤，检查函数 `dup_refine_real_root` 的返回结果是否是 (QQ(1, 1), QQ(3, 2))
    assert R.dup_refine_real_root(f, s, t, steps=1) == (QQ(1, 1), QQ(3, 2))
    
    # 通过增加两次精细化步骤，检查函数 `dup_refine_real_root` 的返回结果是否是 (QQ(7, 5), QQ(3, 2))
    assert R.dup_refine_real_root(f, s, t, steps=2) == (QQ(7, 5), QQ(3, 2))
    
    # 通过增加三次精细化步骤，检查函数 `dup_refine_real_root` 的返回结果是否是 (QQ(7, 5), QQ(13, 9))
    assert R.dup_refine_real_root(f, s, t, steps=3) == (QQ(7, 5), QQ(13, 9))
    
    # 通过增加四次精细化步骤，检查函数 `dup_refine_real_root` 的返回结果是否是 (QQ(7, 5), QQ(27, 19))
    assert R.dup_refine_real_root(f, s, t, steps=4) == (QQ(7, 5), QQ(27, 19))
    
    # 设置 s 和 t 为 QQ(-1, 1) 和 QQ(-2, 1)
    s, t = QQ(-1, 1), QQ(-2, 1)
    
    # 检查使用函数 `dup_refine_real_root` 对给定函数 `f` 和区间 `(s, t)` 进行根的精细化求解，期望结果是 (-QQ(2, 1), -QQ(1, 1))
    assert R.dup_refine_real_root(f, s, t, steps=0) == (-QQ(2, 1), -QQ(1, 1))
    
    # 通过增加一次精细化步骤，检查函数 `dup_refine_real_root` 的返回结果是否是 (-QQ(3, 2), -QQ(1, 1))
    assert R.dup_refine_real_root(f, s, t, steps=1) == (-QQ(3, 2), -QQ(1, 1))
    
    # 通过增加两次精细化步骤，检查函数 `dup_refine_real_root` 的返回结果是否是 (-QQ(3, 2), -QQ(4, 3))
    assert R.dup_refine_real_root(f, s, t, steps=2) == (-QQ(3, 2), -QQ(4, 3))
    
    # 通过增加三次精细化步骤，检查函数 `dup_refine_real_root` 的返回结果是否是 (-QQ(3, 2), -QQ(7, 5))
    assert R.dup_refine_real_root(f, s, t, steps=3) == (-QQ(3, 2), -QQ(7, 5))
    
    # 通过增加四次精细化步骤，检查函数 `dup_refine_real_root` 的返回结果是否是 (-QQ(10, 7), -QQ(7, 5))
    assert R.dup_refine_real_root(f, s, t, steps=4) == (-QQ(10, 7), -QQ(7, 5))
    
    # 检查当指定求解区间不正确时，函数 `dup_refine_real_root` 是否会引发 `RefinementFailed` 异常
    raises(RefinementFailed, lambda: R.dup_refine_real_root(f, QQ(0), QQ(1)))
    
    # 设置 s, t, u, v, w 分别为 QQ(1), QQ(2), QQ(24, 17), QQ(17, 12), QQ(7, 5)
    s, t, u, v, w = QQ(1), QQ(2), QQ(24, 17), QQ(17, 12), QQ(7, 5)
    
    # 检查使用函数 `dup_refine_real_root` 对给定函数 `f` 和区间 `(s, t)` 进行根的精细化求解，期望结果是 (u, v)
    assert R.dup_refine_real_root(f, s, t, eps=QQ(1, 100)) == (u, v)
    
    # 检查使用函数 `dup_refine_real_root` 对给定函数 `f` 和区间 `(s, t)` 进行根的精细化求解，期望结果是 (u, v)
    assert R.dup_refine_real_root(f, s, t, steps=6) == (u, v)
    
    # 检查使用函数 `dup_refine_real_root` 对给定函数 `f` 和区间 `(s, t)` 进行根的精细化求解，期望结果是 (w, v)
    assert R.dup_refine_real_root(f, s, t, eps=QQ(1, 100), steps=5) == (w, v)
    
    # 检查使用函数 `dup_refine_real_root` 对给定函数 `f` 和区间 `(s, t)` 进行根的精细化求解，期望结果是 (u, v)
    assert R.dup_refine_real_root(f, s, t, eps=QQ(1, 100), steps=6) == (u, v)
    
    # 检查使用函数 `dup_refine_real_root` 对给定函数 `f` 和区间 `(s, t)` 进行根的精细化求解，期望结果是 (u, v)
    assert R.dup_refine_real_root(f, s, t, eps=QQ(1, 100), steps=7) == (u, v)
    
    # 设置 s, t, u, v 分别为 QQ(-2), QQ(-1), QQ(-3, 2), QQ(-4, 3)
    s, t, u, v = QQ(-2), QQ(-1), QQ(-3, 2), QQ(-4, 3)
    
    # 检查使用函数 `dup_refine_real_root` 对给定函数 `f` 和区间 `(s, t)` 进行根的精细化求解，期望结果是 (s, t)
    assert R.dup_refine_real_root(f, s, t, disjoint=QQ(-5)) == (s, t)
    
    # 检查使用函数 `dup_refine_real_root` 对给定函数 `f` 和区间 `(s, t)` 进行根的精细化求解，期望结果是 (s, t)
    assert R.dup_refine_real_root(f, s, t, disjoint=-v) == (s, t)
    
    # 检查使用函数 `dup_refine_real_root` 对给定函数 `f` 和区间 `(s, t)` 进行根的精细化求解，期望结果是 (u, v)
    assert R.dup_refine_real_root(f, s, t, disjoint=v) == (u, v)
    
    # 设置 s, t, u, v 分别为 QQ(1), QQ(2), QQ(4, 3), QQ(3, 2)
    s, t, u, v = QQ(1), QQ(2), QQ(4, 3), QQ(3, 2)
    
    # 检查使用函数 `dup_refine_real_root` 对给定函数 `f` 和区间 `(s, t)` 进行根的精细化求解，期望结果是 (s, t)
    assert R.dup_refine_real_root(f, s, t, disjoint=QQ(5)) == (s, t)
    
    # 检查使用函数 `dup_refine_real_root` 对给定函数 `f` 和区间 `(s, t)` 进行根的精细化求解，期望结果是 (s, t)
    assert R.dup_refine_real_root(f, s, t, disjoint=-u) == (s, t)
    
    # 检查使用函数 `dup_refine_real_root` 对给定函数 `f` 和区间 `(s, t)` 进行根的精细化求解，期望结果是 (u, v)
    assert R.dup_refine_real_root(f, s, t, disjoint=u) == (u, v)
# 定义一个多项式环 R 和一个变量 x，使用整数环 ZZ
def test_dup_isolate_real_roots_sqf():
    R, x = ring("x", ZZ)

    # 断言：对于常数多项式 0，返回空列表
    assert R.dup_isolate_real_roots_sqf(0) == []
    # 断言：对于常数多项式 5，返回空列表
    assert R.dup_isolate_real_roots_sqf(5) == []

    # 断言：对于 x^2 + x 这个二次方程，返回实根区间 [(-1, -1), (0, 0)]
    assert R.dup_isolate_real_roots_sqf(x**2 + x) == [(-1, -1), (0, 0)]
    # 断言：对于 x^2 - x 这个二次方程，返回实根区间 [(0, 0), (1, 1)]
    assert R.dup_isolate_real_roots_sqf(x**2 - x) == [(0, 0), (1, 1)]

    # 断言：对于 x^4 + x + 1 这个四次方程，返回空列表，表示无实根
    assert R.dup_isolate_real_roots_sqf(x**4 + x + 1) == []

    # 定义一个实根区间列表 I
    I = [(-2, -1), (1, 2)]
    # 断言：对于 x^2 - 2 这个二次方程，返回实根区间 I
    assert R.dup_isolate_real_roots_sqf(x**2 - 2) == I
    # 断言：对于 -x^2 + 2 这个二次方程，返回实根区间 I，说明多项式的符号变化不影响实根
    assert R.dup_isolate_real_roots_sqf(-x**2 + 2) == I

    # 断言：对于 x - 1 这个一次方程，返回实根区间 [(1, 1)]
    assert R.dup_isolate_real_roots_sqf(x - 1) == [(1, 1)]
    # 断言：对于 x^2 - 3*x + 2 这个二次方程，返回实根区间 [(1, 1), (2, 2)]
    assert R.dup_isolate_real_roots_sqf(x**2 - 3*x + 2) == [(1, 1), (2, 2)]
    # 断言：对于 x^3 - 6*x^2 + 11*x - 6 这个三次方程，返回实根区间 [(1, 1), (2, 2), (3, 3)]
    assert R.dup_isolate_real_roots_sqf(x**3 - 6*x**2 + 11*x - 6) == [(1, 1), (2, 2), (3, 3)]
    # 断言：对于 x^4 - 10*x^3 + 35*x^2 - 50*x + 24 这个四次方程，返回实根区间 [(1, 1), (2, 2), (3, 3), (4, 4)]
    assert R.dup_isolate_real_roots_sqf(x**4 - 10*x**3 + 35*x**2 - 50*x + 24) == [(1, 1), (2, 2), (3, 3), (4, 4)]
    # 断言：对于 x^5 - 15*x^4 + 85*x^3 - 225*x^2 + 274*x - 120 这个五次方程，返回实根区间 [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    assert R.dup_isolate_real_roots_sqf(x**5 - 15*x**4 + 85*x**3 - 225*x**2 + 274*x - 120) == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]

    # 断言：对于 x - 10 这个一次方程，返回实根区间 [(10, 10)]
    assert R.dup_isolate_real_roots_sqf(x - 10) == [(10, 10)]
    # 断言：对于 x^2 - 30*x + 200 这个二次方程，返回实根区间 [(10, 10), (20, 20)]
    assert R.dup_isolate_real_roots_sqf(x**2 - 30*x + 200) == [(10, 10), (20, 20)]
    # 断言：对于 x^3 - 60*x^2 + 1100*x - 6000 这个三次方程，返回实根区间 [(10, 10), (20, 20), (30, 30)]
    assert R.dup_isolate_real_roots_sqf(x**3 - 60*x**2 + 1100*x - 6000) == [(10, 10), (20, 20), (30, 30)]
    # 断言：对于 x^4 - 100*x^3 + 3500*x^2 - 50000*x + 240000 这个四次方程，返回实根区间 [(10, 10), (20, 20), (30, 30), (40, 40)]
    assert R.dup_isolate_real_roots_sqf(x**4 - 100*x**3 + 3500*x**2 - 50000*x + 240000) == [(10, 10), (20, 20), (30, 30), (40, 40)]
    # 断言：对于 x^5 - 150*x^4 + 8500*x^3 - 225000*x^2 + 2740000*x - 12000000 这个五次方程，返回实根区间 [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]
    assert R.dup_isolate_real_roots_sqf(x**5 - 150*x**4 + 8500*x**3 - 225000*x**2 + 2740000*x - 12000000) == [(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]

    # 断言：对于 x + 1 这个一次方程，返回实根区间 [(-1, -1)]
    assert R.dup_isolate_real_roots_sqf(x + 1) == [(-1, -1)]
    # 断言：对于 x^2 + 3*x + 2 这个二次方程，返回实根区间 [(-2, -2), (-1, -1)]
    assert R.dup_isolate_real_roots_sqf(x**2 + 3*x + 2) == [(-2, -2), (-1, -1)]
    # 断言：对于 x^3 + 6*x^2 + 11*x + 6 这个三次方程，返回实根区间 [(-3, -3), (-2, -2), (-1, -1)]
    assert R.dup_isolate_real_roots_sqf(x**3 + 6*x**2 + 11*x + 6) == [(-3, -3), (-2, -2), (-1, -1)]
    # 断言：对于 x^4 + 10*x^3 + 35*x^2 + 50*x + 24 这个四次方程，返回实根区间 [(-4, -4), (-3, -3), (-2, -2), (-1, -1)]
    assert R.dup_isolate_real_roots_sqf(x**4 + 10*x**3 + 35*x**2 + 50*x + 24) == [(-4, -4), (-3, -3), (-2, -2), (-1, -1)]
    # 断言：对于 x^5 + 15*x^4 + 85*x^3 + 225*x^2 + 274*x + 120 这个五次方程，返回实根区间 [(-5, -5), (-4, -4), (-3, -3), (-2, -2), (-1, -1)]
    assert R.dup_isolate_real_roots_sqf(x**5 + 15*x**4 + 85*x**3 + 225*x**2 + 274*x + 120) == [(-5, -5), (-4, -4), (-3, -3), (-2, -2), (-1, -1)]

    # 断言
    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 x**4 - 5，并断言返回结果为 [(-2, -1), (1, 2)]
    assert R.dup_isolate_real_roots_sqf(x**4 - 5) == [(-2, -1), (1, 2)]
    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 x**5 - 5，并断言返回结果为 [(1, 2)]
    assert R.dup_isolate_real_roots_sqf(x**5 - 5) == [(1, 2)]
    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 x**6 - 5，并断言返回结果为 [(-2, -1), (1, 2)]
    assert R.dup_isolate_real_roots_sqf(x**6 - 5) == [(-2, -1), (1, 2)]
    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 x**7 - 5，并断言返回结果为 [(1, 2)]
    assert R.dup_isolate_real_roots_sqf(x**7 - 5) == [(1, 2)]
    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 x**8 - 5，并断言返回结果为 [(-2, -1), (1, 2)]
    assert R.dup_isolate_real_roots_sqf(x**8 - 5) == [(-2, -1), (1, 2)]
    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 x**9 - 5，并断言返回结果为 [(1, 2)]
    assert R.dup_isolate_real_roots_sqf(x**9 - 5) == [(1, 2)]

    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 x**2 - 1，并断言返回结果为 [(-1, -1), (1, 1)]
    assert R.dup_isolate_real_roots_sqf(x**2 - 1) == \
        [(-1, -1), (1, 1)]
    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 x**3 + 2*x**2 - x - 2，并断言返回结果为 [(-2, -2), (-1, -1), (1, 1)]
    assert R.dup_isolate_real_roots_sqf(x**3 + 2*x**2 - x - 2) == \
        [(-2, -2), (-1, -1), (1, 1)]
    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 x**4 - 5*x**2 + 4，并断言返回结果为 [(-2, -2), (-1, -1), (1, 1), (2, 2)]
    assert R.dup_isolate_real_roots_sqf(x**4 - 5*x**2 + 4) == \
        [(-2, -2), (-1, -1), (1, 1), (2, 2)]
    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 x**5 + 3*x**4 - 5*x**3 - 15*x**2 + 4*x + 12，并断言返回结果为 [(-3, -3), (-2, -2), (-1, -1), (1, 1), (2, 2)]
    assert R.dup_isolate_real_roots_sqf(x**5 + 3*x**4 - 5*x**3 - 15*x**2 + 4*x + 12) == \
        [(-3, -3), (-2, -2), (-1, -1), (1, 1), (2, 2)]
    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 x**6 - 14*x**4 + 49*x**2 - 36，并断言返回结果为 [(-3, -3), (-2, -2), (-1, -1), (1, 1), (2, 2), (3, 3)]
    assert R.dup_isolate_real_roots_sqf(x**6 - 14*x**4 + 49*x**2 - 36) == \
        [(-3, -3), (-2, -2), (-1, -1), (1, 1), (2, 2), (3, 3)]
    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 2*x**7 + x**6 - 28*x**5 - 14*x**4 + 98*x**3 + 49*x**2 - 72*x - 36，并断言返回结果为 [(-3, -3), (-2, -2), (-1, -1), (-1, 0), (1, 1), (2, 2), (3, 3)]
    assert R.dup_isolate_real_roots_sqf(2*x**7 + x**6 - 28*x**5 - 14*x**4 + 98*x**3 + 49*x**2 - 72*x - 36) == \
        [(-3, -3), (-2, -2), (-1, -1), (-1, 0), (1, 1), (2, 2), (3, 3)]
    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 4*x**8 - 57*x**6 + 210*x**4 - 193*x**2 + 36，并断言返回结果为 [(-3, -3), (-2, -2), (-1, -1), (-1, 0), (0, 1), (1, 1), (2, 2), (3, 3)]
    assert R.dup_isolate_real_roots_sqf(4*x**8 - 57*x**6 + 210*x**4 - 193*x**2 + 36) == \
        [(-3, -3), (-2, -2), (-1, -1), (-1, 0), (0, 1), (1, 1), (2, 2), (3, 3)]

    # 将 f 设为 9*x**2 - 2
    f = 9*x**2 - 2
    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 f，并断言返回结果为 [(-1, 0), (0, 1)]
    assert R.dup_isolate_real_roots_sqf(f) == \
        [(-1, 0), (0, 1)]

    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 f 和 eps=QQ(1, 10)，并断言返回结果为 [(QQ(-1, 2), QQ(-3, 7)), (QQ(3, 7), QQ(1, 2))]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 10)) == \
        [(QQ(-1, 2), QQ(-3, 7)), (QQ(3, 7), QQ(1, 2))]
    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 f 和 eps=QQ(1, 100)，并断言返回结果为 [(QQ(-9, 19), QQ(-8, 17)), (QQ(8, 17), QQ(9, 19))]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 100)) == \
        [(QQ(-9, 19), QQ(-8, 17)), (QQ(8, 17), QQ(9, 19))]
    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 f 和 eps=QQ(1, 1000)，并断言返回结果为 [(QQ(-33, 70), QQ(-8, 17)), (QQ(8, 17), QQ(33, 70))]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 1000)) == \
        [(QQ(-33, 70), QQ(-8, 17)), (QQ(8, 17), QQ(33, 70))]
    # 调用 R 对象的 dup_isolate_real_roots_sqf 方法，传入 f 和 eps=QQ(1, 10000)，并断言返回结果为 [(QQ(-33, 70), QQ(-107, 227)), (QQ(107, 227), QQ(33, 70))]
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 10000)) == \
        [(QQ(-33, 70), QQ(-107, 227)), (QQ(107,
    # 断言：验证 f 的重复孤立实根的二次因子的结果是否等于指定的列表
    assert R.dup_isolate_real_roots_sqf(f) == \
        [(-a, -a), (-1, 0), (0, 1), (d, d)]

    # 断言：验证带有自定义精度的 f 的重复孤立实根的二次因子的结果是否等于指定的列表
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 100000000000)) == \
        [(-QQ(a), -QQ(a)), (-QQ(1, b), -QQ(1, b)), (QQ(1, c), QQ(1, c)), (QQ(d), QQ(d))]

    # 解包结果，快速计算 f 的重复孤立实根的二次因子
    (u, v), B, C, (s, t) = R.dup_isolate_real_roots_sqf(f, fast=True)

    # 断言：验证快速计算模式下 f 的重复孤立实根的二次因子的结果是否满足指定条件
    assert u < -a < v and B == (-QQ(1), QQ(0)) and C == (QQ(0), QQ(1)) and s < d < t

    # 断言：验证带有自定义极高精度的快速计算模式下 f 的重复孤立实根的二次因子的结果是否等于指定的列表
    assert R.dup_isolate_real_roots_sqf(f, fast=True, eps=QQ(1, 100000000000000000000000000000)) == \
        [(-QQ(a), -QQ(a)), (-QQ(1, b), -QQ(1, b)), (QQ(1, c), QQ(1, c)), (QQ(d), QQ(d))]

    # 设置新的多项式 f
    f = -10*x**4 + 8*x**3 + 80*x**2 - 32*x - 160

    # 断言：验证新的 f 的重复孤立实根的二次因子的结果是否等于指定的列表
    assert R.dup_isolate_real_roots_sqf(f) == \
        [(-2, -2), (-2, -1), (2, 2), (2, 3)]

    # 断言：验证带有自定义精度的新的 f 的重复孤立实根的二次因子的结果是否等于指定的列表
    assert R.dup_isolate_real_roots_sqf(f, eps=QQ(1, 100)) == \
        [(-QQ(2), -QQ(2)), (-QQ(23, 14), -QQ(18, 11)), (QQ(2), QQ(2)), (QQ(39, 16), QQ(22, 9))]

    # 设置新的 f
    f = x - 1

    # 断言：验证在指定区间 inf=2 下，f 的重复孤立实根的二次因子的结果是否为空列表
    assert R.dup_isolate_real_roots_sqf(f, inf=2) == []
    # 断言：验证在指定区间 sup=0 下，f 的重复孤立实根的二次因子的结果是否为空列表
    assert R.dup_isolate_real_roots_sqf(f, sup=0) == []

    # 断言：验证 f 的重复孤立实根的二次因子的结果是否等于指定的列表
    assert R.dup_isolate_real_roots_sqf(f) == [(1, 1)]
    # 断言：验证带有 inf=1 参数的 f 的重复孤立实根的二次因子的结果是否等于指定的列表
    assert R.dup_isolate_real_roots_sqf(f, inf=1) == [(1, 1)]
    # 断言：验证带有 sup=1 参数的 f 的重复孤立实根的二次因子的结果是否等于指定的列表
    assert R.dup_isolate_real_roots_sqf(f, sup=1) == [(1, 1)]
    # 断言：验证带有 inf=1 和 sup=1 参数的 f 的重复孤立实根的二次因子的结果是否等于指定的列表
    assert R.dup_isolate_real_roots_sqf(f, inf=1, sup=1) == [(1, 1)]

    # 设置新的 f
    f = x**2 - 2

    # 断言：验证在指定区间 inf=QQ(7, 4) 下，f 的重复孤立实根的二次因子的结果是否为空列表
    assert R.dup_isolate_real_roots_sqf(f, inf=QQ(7, 4)) == []
    # 断言：验证在指定区间 inf=QQ(7, 5) 下，f 的重复孤立实根的二次因子的结果是否等于指定的列表
    assert R.dup_isolate_real_roots_sqf(f, inf=QQ(7, 5)) == [(QQ(7, 5), QQ(3, 2))]
    # 断言：验证在指定区间 sup=QQ(7, 5) 下，f 的重复孤立实根的二次因子的结果是否等于指定的列表
    assert R.dup_isolate_real_roots_sqf(f, sup=QQ(7, 5)) == [(-2, -1)]
    # 断言：验证在指定区间 sup=QQ(7, 4) 下，f 的重复孤立实根的二次因子的结果是否等于指定的列表
    assert R.dup_isolate_real_roots_sqf(f, sup=QQ(7, 4)) == [(-2, -1), (1, QQ(3, 2))]
    # 断言：验证在指定区间 sup=-QQ(7, 4) 下，f 的重复孤立实根的二次因子的结果是否为空列表
    assert R.dup_isolate_real_roots_sqf(f, sup=-QQ(7, 4)) == []
    # 断言：验证在指定区间 sup=-QQ(7, 5) 下，f 的重复孤立实根的二次因子的结果是否等于指定的列表
    assert R.dup_isolate_real_roots_sqf(f, sup=-QQ(7, 5)) == [(-QQ(3, 2), -QQ(7, 5))]
    # 断言：验证在指定区间 inf=-QQ(7, 5) 下，f 的重复孤立实根的二次因子的结果是否等于指定的列表
    assert R.dup_isolate_real_roots_sqf(f, inf=-QQ(7, 5)) == [(1, 2)]
    # 断言：验证在指定区间 inf=-QQ(7, 4) 下，f 的重复孤立实根的二次因子的结果是否等于指定的列表
    assert R.dup_isolate_real_roots_sqf(f, inf=-QQ(7, 4)) == [(-QQ(3, 2), -1), (1, 2)]

    # 设置区间 I
    I = [(-2, -1), (1, 2)]

    # 断言：验证在指定区间 inf=-2 下，f 的重复孤立实根的二次因子的结果是否等于预定义的区间 I
    assert R.dup_isolate_real_roots_sqf(f, inf=-2) == I
    # 断言：验证在指定区间 sup=+2 下，f 的重复孤立实根的二次因子的结果是否等于预定义的区间 I
    assert R.dup_isolate_real_roots_sqf(f, sup=+2) == I

    # 断言：验证在指定区间 inf=-2 和 sup=2 下，f 的重复孤立实根的二次因子的结果是否等于预定义的区间 I
    assert R.dup_isolate_real_roots_sqf(f, inf=-2, sup=2) == I

    # 设置新的多项式 f 和环 R
    R, x = ring("x", QQ)
    f = QQ(8, 5)*x**2 - QQ(87374, 3855)*x - QQ(17, 771)

    # 断言：验证 f 的重复孤立实根的二次因子的结果是否等于指定的列表
    assert R.dup
# 定义一个测试函数，用于测试多项式环 R 上的 dup_isolate_real_roots 方法
def test_dup_isolate_real_roots():
    # 创建多项式环 R 和变量 x
    R, x = ring("x", ZZ)

    # 断言调用 dup_isolate_real_roots 方法对多项式 0 的结果为空列表
    assert R.dup_isolate_real_roots(0) == []
    # 断言调用 dup_isolate_real_roots 方法对多项式 3 的结果为空列表
    assert R.dup_isolate_real_roots(3) == []

    # 断言调用 dup_isolate_real_roots 方法对多项式 5*x 的结果为 [((0, 0), 1)]
    assert R.dup_isolate_real_roots(5*x) == [((0, 0), 1)]
    # 断言调用 dup_isolate_real_roots 方法对多项式 7*x**4 的结果为 [((0, 0), 4)]
    assert R.dup_isolate_real_roots(7*x**4) == [((0, 0), 4)]

    # 断言调用 dup_isolate_real_roots 方法对多项式 x**2 + x 的结果为 [((-1, -1), 1), ((0, 0), 1)]
    assert R.dup_isolate_real_roots(x**2 + x) == [((-1, -1), 1), ((0, 0), 1)]
    # 断言调用 dup_isolate_real_roots 方法对多项式 x**2 - x 的结果为 [((0, 0), 1), ((1, 1), 1)]
    assert R.dup_isolate_real_roots(x**2 - x) == [((0, 0), 1), ((1, 1), 1)]

    # 断言调用 dup_isolate_real_roots 方法对多项式 x**4 + x + 1 的结果为空列表
    assert R.dup_isolate_real_roots(x**4 + x + 1) == []

    # 定义区间 I
    I = [((-2, -1), 1), ((1, 2), 1)]

    # 断言调用 dup_isolate_real_roots 方法对多项式 x**2 - 2 的结果为 I
    assert R.dup_isolate_real_roots(x**2 - 2) == I
    # 断言调用 dup_isolate_real_roots 方法对多项式 -x**2 + 2 的结果为 I
    assert R.dup_isolate_real_roots(-x**2 + 2) == I

    # 定义多项式 f 和对其进行因式分解得到的 g
    f = 16*x**14 - 96*x**13 + 24*x**12 + 936*x**11 - 1599*x**10 - 2880*x**9 + 9196*x**8 \
      + 552*x**7 - 21831*x**6 + 13968*x**5 + 21690*x**4 - 26784*x**3 - 2916*x**2 + 15552*x - 5832
    g = R.dup_sqf_part(f)

    # 断言调用 dup_isolate_real_roots 方法对多项式 f 的结果为指定的区间列表
    assert R.dup_isolate_real_roots(f) == \
        [((-QQ(2), -QQ(3, 2)), 2), ((-QQ(3, 2), -QQ(1, 1)), 3), ((QQ(1), QQ(3, 2)), 3),
         ((QQ(3, 2), QQ(3, 2)), 4), ((QQ(5, 3), QQ(2)), 2)]

    # 断言调用 dup_isolate_real_roots_sqf 方法对多项式 g 的结果为指定的区间列表
    assert R.dup_isolate_real_roots_sqf(g) == \
        [(-QQ(2), -QQ(3, 2)), (-QQ(3, 2), -QQ(1, 1)), (QQ(1), QQ(3, 2)),
         (QQ(3, 2), QQ(3, 2)), (QQ(3, 2), QQ(2))]
    # 断言调用 dup_isolate_real_roots 方法对多项式 g 的结果为指定的区间列表
    assert R.dup_isolate_real_roots(g) == \
        [((-QQ(2), -QQ(3, 2)), 1), ((-QQ(3, 2), -QQ(1, 1)), 1), ((QQ(1), QQ(3, 2)), 1),
         ((QQ(3, 2), QQ(3, 2)), 1), ((QQ(3, 2), QQ(2)), 1)]

    # 重新定义多项式 f
    f = x - 1

    # 断言调用 dup_isolate_real_roots 方法对多项式 f 在 inf=2 时的结果为空列表
    assert R.dup_isolate_real_roots(f, inf=2) == []
    # 断言调用 dup_isolate_real_roots 方法对多项式 f 在 sup=0 时的结果为空列表
    assert R.dup_isolate_real_roots(f, sup=0) == []

    # 断言调用 dup_isolate_real_roots 方法对多项式 f 的结果为 [((1, 1), 1)]
    assert R.dup_isolate_real_roots(f) == [((1, 1), 1)]
    # 断言调用 dup_isolate_real_roots 方法对多项式 f 在 inf=1 时的结果为 [((1, 1), 1)]
    assert R.dup_isolate_real_roots(f, inf=1) == [((1, 1), 1)]
    # 断言调用 dup_isolate_real_roots 方法对多项式 f 在 sup=1 时的结果为 [((1, 1), 1)]
    assert R.dup_isolate_real_roots(f, sup=1) == [((1, 1), 1)]
    # 断言调用 dup_isolate_real_roots 方法对多项式 f 在 inf=1, sup=1 时的结果为 [((1, 1), 1)]
    assert R.dup_isolate_real_roots(f, inf=1, sup=1) == [((1, 1), 1)]

    # 重新定义多项式 f
    f = x**4 - 4*x**2 + 4

    # 断言调用 dup_isolate_real_roots 方法对多项式 f 在 inf=QQ(7, 4) 时的结果为空列表
    assert R.dup_isolate_real_roots(f, inf=QQ(7, 4)) == []
    # 断言调用 dup_isolate_real_roots 方法对多项式 f 在 inf=QQ(7, 5) 时的结果为 [((QQ(7, 5), QQ(3, 2)), 2)]
    assert R.dup_isolate_real_roots(f, inf=QQ(7, 5)) == [((QQ(7, 5), QQ(3, 2)), 2)]
    # 断言调用 dup_isolate_real_roots 方法对多项式 f 在 sup=QQ(7, 5) 时的结果为 [((-2, -1), 2)]
    assert R.dup_isolate_real_roots(f, sup=QQ(7, 5)) == [((-2, -1), 2)]
    # 断言调用 dup_isolate_real_roots 方法对多项式 f 在 sup=QQ(7, 4) 时的结果为 [((-2, -1), 2), ((1, QQ(3, 2)), 2)]
    assert R.dup_isolate_real_roots(f, sup=QQ(7, 4)) == [((-2, -1), 2), ((1, QQ(3, 2)), 2)]
    # 断言调用 dup_isolate_real_roots 方法对多项式 f 在 sup=-QQ(7, 4) 时的结果为空列表
    assert R.dup_isolate_real_roots(f, sup=-QQ(7, 4)) == []
    # 断言调用 dup_isolate_real_roots 方法对多项式 f 在 sup=-QQ(7, 5) 时的结果为 [((-QQ(3, 2), -QQ(7, 5)), 2)]
    assert R.dup_isolate_real_roots(f, sup=-QQ(7, 5)) == [((-QQ(3, 2), -QQ(7, 5)), 2)]
    # 断言调用 dup_isolate_real_roots 方法对多项式 f 在 inf=-QQ(7, 5) 时的结果为 [((1, 2), 2)]
    assert R.dup_isolate_real_roots(f, inf=-QQ(7, 5)) == [((1,
    # 计算多项式 f(x) = x^45 - 45*x^44 + 990*x^43 - 1
    f = (x**45 - 45*x**44 + 990*x**43 - 1)
    # 计算多项式 g(x)，这是一个长多项式，包含了大量的高次幂项
    g = (x**46 - 15180*x**43 + 9366819*x**40 - 53524680*x**39 + 260932815*x**38 - 1101716330*x**37 + 4076350421*x**36 - 13340783196*x**35 + 38910617655*x**34 - 101766230790*x**33 + 239877544005*x**32 - 511738760544*x**31 + 991493848554*x**30 - 1749695026860*x**29 + 2818953098830*x**28 - 4154246671960*x**27 + 5608233007146*x**26 - 6943526580276*x**25 + 7890371113950*x**24 - 8233430727600*x**23 + 7890371113950*x**22 - 6943526580276*x**21 + 5608233007146*x**20 - 4154246671960*x**19 + 2818953098830*x**18 - 1749695026860*x**17 + 991493848554*x**16 - 511738760544*x**15 + 239877544005*x**14 - 101766230790*x**13 + 38910617655*x**12 - 13340783196*x**11 + 4076350421*x**10 - 1101716330*x**9 + 260932815*x**8 - 53524680*x**7 + 9366819*x**6 - 1370754*x**5 + 163185*x**4 - 15180*x**3 + 1035*x**2 - 47*x + 1)

    # 断言验证多项式 f(x)*g(x) 的实根分解结果是否等于给定的列表
    assert R.dup_isolate_real_roots(f*g) == \
        [((0, QQ(1, 2)), 1), ((QQ(2, 3), QQ(3, 4)), 1), ((QQ(3, 4), 1), 1), ((6, 7), 1), ((24, 25), 1)]

    # 在环 R 中定义变量 x，并尝试对 x + 3 进行实根分解
    R, x = ring("x", EX)
    # 使用 lambda 函数尝试执行对 x + 3 的实根分解，预期会抛出 DomainError 异常
    raises(DomainError, lambda: R.dup_isolate_real_roots(x + 3))
def test_dup_isolate_real_roots_list():
    R, x = ring("x", ZZ)  # 创建一个整数环 R 和符号变量 x

    assert R.dup_isolate_real_roots_list([x**2 + x, x]) == \
        [((-1, -1), {0: 1}), ((0, 0), {0: 1, 1: 1})]
    assert R.dup_isolate_real_roots_list([x**2 - x, x]) == \
        [((0, 0), {0: 1, 1: 1}), ((1, 1), {0: 1})]

    assert R.dup_isolate_real_roots_list([x + 1, x + 2, x - 1, x + 1, x - 1, x - 1]) == \
        [((-QQ(2), -QQ(2)), {1: 1}), ((-QQ(1), -QQ(1)), {0: 1, 3: 1}), ((QQ(1), QQ(1)), {2: 1, 4: 1, 5: 1})]

    assert R.dup_isolate_real_roots_list([x + 1, x + 2, x - 1, x + 1, x - 1, x + 2]) == \
        [((-QQ(2), -QQ(2)), {1: 1, 5: 1}), ((-QQ(1), -QQ(1)), {0: 1, 3: 1}), ((QQ(1), QQ(1)), {2: 1, 4: 1})]

    f, g = x**4 - 4*x**2 + 4, x - 1  # 定义两个多项式 f 和 g

    assert R.dup_isolate_real_roots_list([f, g], inf=QQ(7, 4)) == []  # 使用 inf 参数进行实数根分离
    assert R.dup_isolate_real_roots_list([f, g], inf=QQ(7, 5)) == \
        [((QQ(7, 5), QQ(3, 2)), {0: 2})]
    assert R.dup_isolate_real_roots_list([f, g], sup=QQ(7, 5)) == \
        [((-2, -1), {0: 2}), ((1, 1), {1: 1})]
    assert R.dup_isolate_real_roots_list([f, g], sup=QQ(7, 4)) == \
        [((-2, -1), {0: 2}), ((1, 1), {1: 1}), ((1, QQ(3, 2)), {0: 2})]
    assert R.dup_isolate_real_roots_list([f, g], sup=-QQ(7, 4)) == []
    assert R.dup_isolate_real_roots_list([f, g], sup=-QQ(7, 5)) == \
        [((-QQ(3, 2), -QQ(7, 5)), {0: 2})]
    assert R.dup_isolate_real_roots_list([f, g], inf=-QQ(7, 5)) == \
        [((1, 1), {1: 1}), ((1, 2), {0: 2})]
    assert R.dup_isolate_real_roots_list([f, g], inf=-QQ(7, 4)) == \
        [((-QQ(3, 2), -1), {0: 2}), ((1, 1), {1: 1}), ((1, 2), {0: 2})]

    f, g = 2*x**2 - 1, x**2 - 2  # 定义新的多项式 f 和 g

    assert R.dup_isolate_real_roots_list([f, g]) == \
        [((-QQ(2), -QQ(1)), {1: 1}), ((-QQ(1), QQ(0)), {0: 1}),
         ((QQ(0), QQ(1)), {0: 1}), ((QQ(1), QQ(2)), {1: 1})]
    assert R.dup_isolate_real_roots_list([f, g], strict=True) == \
        [((-QQ(3, 2), -QQ(4, 3)), {1: 1}), ((-QQ(1), -QQ(2, 3)), {0: 1}),
         ((QQ(2, 3), QQ(1)), {0: 1}), ((QQ(4, 3), QQ(3, 2)), {1: 1})]

    f, g = x**2 - 2, x**3 - x**2 - 2*x + 2  # 定义新的多项式 f 和 g

    assert R.dup_isolate_real_roots_list([f, g]) == \
        [((-QQ(2), -QQ(1)), {1: 1, 0: 1}), ((QQ(1), QQ(1)), {1: 1}), ((QQ(1), QQ(2)), {1: 1, 0: 1})]

    f, g = x**3 - 2*x, x**5 - x**4 - 2*x**3 + 2*x**2  # 定义新的多项式 f 和 g

    assert R.dup_isolate_real_roots_list([f, g]) == \
        [((-QQ(2), -QQ(1)), {1: 1, 0: 1}), ((QQ(0), QQ(0)), {0: 1, 1: 2}),
         ((QQ(1), QQ(1)), {1: 1}), ((QQ(1), QQ(2)), {1: 1, 0: 1})]

    f, g = x**9 - 3*x**8 - x**7 + 11*x**6 - 8*x**5 - 8*x**4 + 12*x**3 - 4*x**2, x**5 - 2*x**4 + 3*x**3 - 4*x**2 + 2*x  # 定义新的多项式 f 和 g

    assert R.dup_isolate_real_roots_list([f, g], basis=False) == \
        [((-2, -1), {0: 2}), ((0, 0), {0: 2, 1: 1}), ((1, 1), {0: 3, 1: 2}), ((1, 2), {0: 2})]
    assert R.dup_isolate_real_roots_list([f, g], basis=True) == \
        [((-2, -1), {0: 2}, [1, 0, -2]), ((0, 0), {0: 2, 1: 1}, [1, 0]),
         ((1, 1), {0: 3, 1: 2}, [1, -1]), ((1, 2), {0: 2}, [1, 0, -2])]

    R, x = ring("x", EX)  # 切换到扩展环 EX
    # 使用 raises 函数验证调用 R.dup_isolate_real_roots_list([x + 3]) 是否会引发 DomainError 异常
    raises(DomainError, lambda: R.dup_isolate_real_roots_list([x + 3]))
# 定义一个测试函数，用于测试多项式的实数根的隔离列表
def test_dup_isolate_real_roots_list_QQ():
    # 创建一个整数环 R 和一个变量 x
    R, x = ring("x", ZZ)

    # 定义两个多项式 f 和 g
    f = x**5 - 200
    g = x**5 - 201

    # 断言计算多项式列表 [f, g] 的实数根的隔离列表是否等于期望值
    assert R.dup_isolate_real_roots_list([f, g]) == \
        [((QQ(75, 26), QQ(101, 35)), {0: 1}), ((QQ(309, 107), QQ(26, 9)), {1: 1})]

    # 切换环为有理数环 QQ 和变量 x
    R, x = ring("x", QQ)

    # 重新定义两个有理数多项式 f 和 g
    f = -QQ(1, 200)*x**5 + 1
    g = -QQ(1, 201)*x**5 + 1

    # 再次断言计算多项式列表 [f, g] 的实数根的隔离列表是否等于期望值
    assert R.dup_isolate_real_roots_list([f, g]) == \
        [((QQ(75, 26), QQ(101, 35)), {0: 1}), ((QQ(309, 107), QQ(26, 9)), {1: 1})]


# 定义一个测试函数，用于测试多项式的实数根个数计算
def test_dup_count_real_roots():
    # 创建一个整数环 R 和一个变量 x
    R, x = ring("x", ZZ)

    # 断言计算多项式 0 的实数根个数是否为 0
    assert R.dup_count_real_roots(0) == 0
    # 断言计算多项式 7 的实数根个数是否为 0
    assert R.dup_count_real_roots(7) == 0

    # 定义一个一次多项式 f = x - 1
    f = x - 1
    # 断言计算多项式 f 的实数根个数是否为 1
    assert R.dup_count_real_roots(f) == 1
    # 断言在区间 [0, 1] 上计算多项式 f 的实数根个数是否为 1
    assert R.dup_count_real_roots(f, inf=0, sup=1) == 1
    # 断言在区间 [0, 2] 上计算多项式 f 的实数根个数是否为 1
    assert R.dup_count_real_roots(f, inf=0, sup=2) == 1

    # 定义一个二次多项式 f = x**2 - 2
    f = x**2 - 2
    # 断言计算多项式 f 的实数根个数是否为 2
    assert R.dup_count_real_roots(f) == 2
    # 断言在区间 [-1, 1] 上计算多项式 f 的实数根个数是否为 0
    assert R.dup_count_real_roots(f, inf=-1, sup=1) == 0


# 定义测试函数的参数，用于测试多项式的复数根个数计算
a, b = (-QQ(1), -QQ(1)), (QQ(1), QQ(1))
c, d = (QQ(0), QQ(0)), (QQ(1), QQ(1))

# 定义一个测试函数，用于测试多项式的复数根个数计算（第一组测试）
def test_dup_count_complex_roots_1():
    # 创建一个整数环 R 和一个变量 x
    R, x = ring("x", ZZ)

    # 定义一个一次多项式 f = x - 1
    f = x - 1
    # 断言计算多项式 f 的复数根个数是否为 1，使用参数 a, b
    assert R.dup_count_complex_roots(f, a, b) == 1
    # 断言计算多项式 f 的复数根个数是否为 1，使用参数 c, d
    assert R.dup_count_complex_roots(f, c, d) == 1

    # 定义一个一次多项式 f = x + 1
    f = x + 1
    # 断言计算多项式 f 的复数根个数是否为 1，使用参数 a, b
    assert R.dup_count_complex_roots(f, a, b) == 1
    # 断言计算多项式 f 的复数根个数是否为 0，使用参数 c, d
    assert R.dup_count_complex_roots(f, c, d) == 0


# 定义一个测试函数，用于测试多项式的复数根个数计算（第二组测试）
def test_dup_count_complex_roots_2():
    # 创建一个整数环 R 和一个变量 x
    R, x = ring("x", ZZ)

    # 定义一个二次多项式 f = x**2 - x
    f = x**2 - x
    # 断言计算多项式 f 的复数根个数是否为 2，使用参数 a, b
    assert R.dup_count_complex_roots(f, a, b) == 2
    # 断言计算多项式 f 的复数根个数是否为 2，使用参数 c, d
    assert R.dup_count_complex_roots(f, c, d) == 2

    # 定义一个二次多项式 f = -x**2 + x
    f = -x**2 + x
    # 断言计算多项式 f 的复数根个数是否为 2，使用参数 a, b
    assert R.dup_count_complex_roots(f, a, b) == 2
    # 断言计算多项式 f 的复数根个数是否为 2，使用参数 c, d
    assert R.dup_count_complex_roots(f, c, d) == 2

    # 定义一个二次多项式 f = x**2 + x
    f = x**2 + x
    # 断言计算多项式 f 的复数根个数是否为 2，使用参数 a, b
    assert R.dup_count_complex_roots(f, a, b) == 2
    # 断言计算多项式 f 的复数根个数是否为 1，使用参数 c, d
    assert R.dup_count_complex_roots(f, c, d) == 1

    # 定义一个二次多项式 f = -x**2 - x
    f = -x**2 - x
    # 断言计算多项式 f 的复数根个数是否为 2，使用参数 a, b
    assert R.dup_count_complex_roots(f, a, b) == 2
    # 断言计算多项式 f 的复数根个数是否为 1，使用参数 c, d
    assert R.dup_count_complex_roots(f, c, d) == 1


# 定义一个测试函数，用于测试多项式的复数根个数计算（第三组测试）
def test_dup_count_complex_roots_3():
    # 创建一个整数环 R 和一个变量 x
    R, x = ring("x", ZZ)

    # 定义一个二次多项式 f = x**2 - 1
    f = x**2 - 1
    # 断言计算多项式 f 的复数根个数是否为 2，使用参数 a, b
    assert R.dup_count_complex_roots(f, a, b) == 2
    # 断言计算多项式 f 的复数根个数是否为 1，使用参数 c, d
    assert R.dup_count_complex_roots(f, c, d) == 1

    # 定义一个三次多项式 f = x**3 - x
    f = x**3 - x
    # 断言计算多项式 f 的复数根个数是否为 3，使用参数 a, b
    assert R.dup_count_complex_roots(f, a, b) == 3
    # 断言计算多项式 f 的复数根个数是否为 2，使用参数 c, d
    assert R.dup_count_complex_roots(f, c, d) == 2
    # 确保函数 R.dup_count_complex_roots 在给定函数 f 和复数区间 [a, b] 上返回 3
    assert R.dup_count_complex_roots(f, a, b) == 3
    
    # 确保函数 R.dup_count_complex_roots 在给定函数 f 和复数区间 [c, d] 上返回 2
    assert R.dup_count_complex_roots(f, c, d) == 2
    
    # 定义多项式 f = -x**3 - x，其根为 (z-I)*(z+I)*(-z)
    f = -x**3 - x
    # 确保函数 R.dup_count_complex_roots 在给定函数 f 和复数区间 [a, b] 上返回 3
    assert R.dup_count_complex_roots(f, a, b) == 3
    # 确保函数 R.dup_count_complex_roots 在给定函数 f 和复数区间 [c, d] 上返回 2
    assert R.dup_count_complex_roots(f, c, d) == 2
    
    # 定义多项式 f = x**3 - x**2 + x - 1，其根为 (z-I)*(z+I)*(z-1)
    f = x**3 - x**2 + x - 1
    # 确保函数 R.dup_count_complex_roots 在给定函数 f 和复数区间 [a, b] 上返回 3
    assert R.dup_count_complex_roots(f, a, b) == 3
    # 确保函数 R.dup_count_complex_roots 在给定函数 f 和复数区间 [c, d] 上返回 2
    assert R.dup_count_complex_roots(f, c, d) == 2
    
    # 定义多项式 f = x**4 - x**3 + x**2 - x，其根为 (z-I)*(z+I)*(z-1)*(z)
    f = x**4 - x**3 + x**2 - x
    # 确保函数 R.dup_count_complex_roots 在给定函数 f 和复数区间 [a, b] 上返回 4
    assert R.dup_count_complex_roots(f, a, b) == 4
    # 确保函数 R.dup_count_complex_roots 在给定函数 f 和复数区间 [c, d] 上返回 3
    assert R.dup_count_complex_roots(f, c, d) == 3
    
    # 定义多项式 f = -x**4 + x**3 - x**2 + x，其根为 (z-I)*(z+I)*(z-1)*(-z)
    f = -x**4 + x**3 - x**2 + x
    # 确保函数 R.dup_count_complex_roots 在给定函数 f 和复数区间 [a, b] 上返回 4
    assert R.dup_count_complex_roots(f, a, b) == 4
    # 确保函数 R.dup_count_complex_roots 在给定函数 f 和复数区间 [c, d] 上返回 3
    assert R.dup_count_complex_roots(f, c, d) == 3
    
    # 定义多项式 f = x**4 - 1，其根为 (z-I)*(z+I)*(z-1)*(z+1)
    f = x**4 - 1
    # 确保函数 R.dup_count_complex_roots 在给定函数 f 和复数区间 [a, b] 上返回 4
    assert R.dup_count_complex_roots(f, a, b) == 4
    # 确保函数 R.dup_count_complex_roots 在给定函数 f 和复数区间 [c, d] 上返回 2
    assert R.dup_count_complex_roots(f, c, d) == 2
    
    # 定义多项式 f = x**5 - x，其根为 (z-I)*(z+I)*(z-1)*(z+1)*(z)
    f = x**5 - x
    # 确保函数 R.dup_count_complex_roots 在给定函数 f 和复数区间 [a, b] 上返回 5
    assert R.dup_count_complex_roots(f, a, b) == 5
    # 确保函数 R.dup_count_complex_roots 在给定函数 f 和复数区间 [c, d] 上返回 3
    assert R.dup_count_complex_roots(f, c, d) == 3
    
    # 定义多项式 f = -x**5 + x，其根为 (z-I)*(z+I)*(z-1)*(z+1)*(-z)
    f = -x**5 + x
    # 确保函数 R.dup_count_complex_roots 在给定函数 f 和复数区间 [a, b] 上返回 5
    assert R.dup_count_complex_roots(f, a, b) == 5
    # 确保函数 R.dup_count_complex_roots 在给定函数 f 和复数区间 [c, d] 上返回 3
    assert R.dup_count_complex_roots(f, c, d) == 3
def test_dup_count_complex_roots_5():
    R, x = ring("x", ZZ)

    # 定义多项式 f = x^2 + 2x + 2
    f = x**2 + 2*x + 2
    # 断言 f 在区间 (a, b) 内的复数根重数为 2
    assert R.dup_count_complex_roots(f, a, b) == 2
    # 断言 f 在区间 (c, d) 内的复数根重数为 0
    assert R.dup_count_complex_roots(f, c, d) == 0

    # 定义多项式 f = x^3 + x^2 - 2
    f = x**3 + x**2 - 2
    # 断言 f 在区间 (a, b) 内的复数根重数为 3
    assert R.dup_count_complex_roots(f, a, b) == 3
    # 断言 f 在区间 (c, d) 内的复数根重数为 1
    assert R.dup_count_complex_roots(f, c, d) == 1

    # 定义多项式 f = x^4 + x^3 - 2x
    f = x**4 + x**3 - 2*x
    # 断言 f 在区间 (a, b) 内的复数根重数为 4
    assert R.dup_count_complex_roots(f, a, b) == 4
    # 断言 f 在区间 (c, d) 内的复数根重数为 2
    assert R.dup_count_complex_roots(f, c, d) == 2

    # 定义多项式 f = x^3 + 3x^2 + 4x + 2
    f = x**3 + 3*x**2 + 4*x + 2
    # 断言 f 在区间 (a, b) 内的复数根重数为 3
    assert R.dup_count_complex_roots(f, a, b) == 3
    # 断言 f 在区间 (c, d) 内的复数根重数为 0
    assert R.dup_count_complex_roots(f, c, d) == 0

    # 定义多项式 f = x^4 + 3x^3 + 4x^2 + 2x
    f = x**4 + 3*x**3 + 4*x**2 + 2*x
    # 断言 f 在区间 (a, b) 内的复数根重数为 4
    assert R.dup_count_complex_roots(f, a, b) == 4
    # 断言 f 在区间 (c, d) 内的复数根重数为 1
    assert R.dup_count_complex_roots(f, c, d) == 1

    # 定义多项式 f = x^4 + 2x^3 + x^2 - 2x - 2
    f = x**4 + 2*x**3 + x**2 - 2*x - 2
    # 断言 f 在区间 (a, b) 内的复数根重数为 4
    assert R.dup_count_complex_roots(f, a, b) == 4
    # 断言 f 在区间 (c, d) 内的复数根重数为 1
    assert R.dup_count_complex_roots(f, c, d) == 1

    # 定义多项式 f = x^5 + 2x^4 + x^3 - 2x^2 - 2x
    f = x**5 + 2*x**4 + x**3 - 2*x**2 - 2*x
    # 断言 f 在区间 (a, b) 内的复数根重数为 5
    assert R.dup_count_complex_roots(f, a, b) == 5
    # 断言 f 在区间 (c, d) 内的复数根重数为 2
    assert R.dup_count_complex_roots(f, c, d) == 2


def test_dup_count_complex_roots_6():
    R, x = ring("x", ZZ)

    # 定义多项式 f = x^2 - 2x + 2
    f = x**2 - 2*x + 2
    # 断言 f 在区间 (a, b) 内的复数根重数为 2
    assert R.dup_count_complex_roots(f, a, b) == 2
    # 断言 f 在区间 (c, d) 内的复数根重数为 1
    assert R.dup_count_complex_roots(f, c, d) == 1

    # 定义多项式 f = x^3 - 3x^2 + 4x - 2
    f = x**3 - 3*x**2 + 4*x - 2
    # 断言 f 在区间 (a, b) 内的复数根重数为 3
    assert R.dup_count_complex_roots(f, a, b) == 3
    # 断言 f 在区间 (c, d) 内的复数根重数为 2
    assert R.dup_count_complex_roots(f, c, d) == 2

    # 定义多项式 f = x^4 - 3x^3 + 4x^2 - 2x
    f = x**4 - 3*x**3 + 4*x**2 - 2*x
    # 断言 f 在区间 (a, b) 内的复数根重数为 4
    assert R.dup_count_complex_roots(f, a, b) == 4
    # 断言 f 在区间 (c, d) 内的复数根重数为 3
    assert R.dup_count_complex_roots(f, c, d) == 3

    # 定义多项式 f = x^3 - x^2 + 2
    f = x**3 - x**2 + 2
    # 断言 f 在区间 (a, b) 内的复数根重数为 3
    assert R.dup_count_complex_roots(f, a, b) == 3
    # 断言 f 在区间 (c, d) 内的复数根重数为 1
    assert R.dup_count_complex_roots(f, c, d) == 1

    # 定义多项式 f = x^4 - x^3 + 2x
    f = x**4 - x**3 + 2*x
    # 断言 f 在区间 (a, b) 内的复数根重数为 4
    assert R.dup_count_complex_roots(f, a, b) == 4
    # 断言 f 在区间 (c, d) 内的复数根重数为 2
    assert R.dup_count_complex_roots(f, c, d) == 2

    # 定义多项式 f = x^4 - 2x^3 + x^2 + 2x - 2
    f = x**4 - 2*x**3 + x**2 + 2*x - 2
    # 断言 f 在区间 (a, b) 内的复数根重数为 4
    assert R.dup_count_complex_roots(f, a, b) == 4
    # 断言 f 在区间 (c, d) 内的复数根重数为 2
    assert R.dup_count_complex_roots(f, c, d) == 2

    # 定义多项式 f = x^5 - 2x^4 + x^3 + 2x^2 - 2x
    f = x**5 - 2*x**4 + x**3 + 2*x**2 - 2*x
    # 断言 f 在区间 (a, b) 内的复数根重数为 5
    assert R.dup_count_complex_roots(f, a, b) == 5
    # 断言 f 在区间 (c, d) 内的复数根重数为 3
    assert R.dup_count_complex_roots(f, c, d) == 3


def test_dup_count_complex_roots_7():
    R, x = ring("x", ZZ)

    # 定义多项式 f = x^4 + 4
    f = x**4 + 4
    # 断言 f 在区间 (a, b) 内的复数根重数为 4
    assert R.dup_count_complex_roots(f, a, b) == 4
    #
    # 断言函数调用，验证 f 函数在给定参数 c, d 下复数根的重复次数是否为 1
    assert R.dup_count_complex_roots(f, c, d) == 1
    
    # 定义多项式 f = (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z-1)，其次数为 5
    f = x**5 - x**4 + 4*x - 4
    # 断言函数调用，验证 f 函数在给定参数 a, b 下复数根的重复次数是否为 5
    assert R.dup_count_complex_roots(f, a, b) == 5
    # 断言函数调用，验证 f 函数在给定参数 c, d 下复数根的重复次数是否为 2
    assert R.dup_count_complex_roots(f, c, d) == 2
    
    # 定义多项式 f = (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z-1)*z，其次数为 6
    f = x**6 - x**5 + 4*x**2 - 4*x
    # 断言函数调用，验证 f 函数在给定参数 a, b 下复数根的重复次数是否为 6
    assert R.dup_count_complex_roots(f, a, b) == 6
    # 断言函数调用，验证 f 函数在给定参数 c, d 下复数根的重复次数是否为 3
    assert R.dup_count_complex_roots(f, c, d) == 3
    
    # 定义多项式 f = (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z+1)，其次数为 5
    f = x**5 + x**4 + 4*x + 4
    # 断言函数调用，验证 f 函数在给定参数 a, b 下复数根的重复次数是否为 5
    assert R.dup_count_complex_roots(f, a, b) == 5
    # 断言函数调用，验证 f 函数在给定参数 c, d 下复数根的重复次数是否为 1
    assert R.dup_count_complex_roots(f, c, d) == 1
    
    # 定义多项式 f = (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z+1)*z，其次数为 6
    f = x**6 + x**5 + 4*x**2 + 4*x
    # 断言函数调用，验证 f 函数在给定参数 a, b 下复数根的重复次数是否为 6
    assert R.dup_count_complex_roots(f, a, b) == 6
    # 断言函数调用，验证 f 函数在给定参数 c, d 下复数根的重复次数是否为 2
    assert R.dup_count_complex_roots(f, c, d) == 2
    
    # 定义多项式 f = (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z-1)*(z+1)，其次数为 6
    f = x**6 - x**4 + 4*x**2 - 4
    # 断言函数调用，验证 f 函数在给定参数 a, b 下复数根的重复次数是否为 6
    assert R.dup_count_complex_roots(f, a, b) == 6
    # 断言函数调用，验证 f 函数在给定参数 c, d 下复数根的重复次数是否为 2
    assert R.dup_count_complex_roots(f, c, d) == 2
    
    # 定义多项式 f = (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z-1)*(z+1)*z，其次数为 7
    f = x**7 - x**5 + 4*x**3 - 4*x
    # 断言函数调用，验证 f 函数在给定参数 a, b 下复数根的重复次数是否为 7
    assert R.dup_count_complex_roots(f, a, b) == 7
    # 断言函数调用，验证 f 函数在给定参数 c, d 下复数根的重复次数是否为 3
    assert R.dup_count_complex_roots(f, c, d) == 3
    
    # 定义多项式 f = (z-I-1)*(z+I-1)*(z-I+1)*(z+I+1)*(z-1)*(z+1)*(z-I)*(z+I)，其次数为 8
    f = x**8 + 3*x**4 - 4
    # 断言函数调用，验证 f 函数在给定参数 a, b 下复数根的重复次数是否为 8
    assert R.dup_count_complex_roots(f, a, b) == 8
    # 断言函数调用，验证 f 函数在给定参数 c, d 下复数根的重复次数是否为 3
    assert R.dup_count_complex_roots(f, c, d) == 3
def test_dup_count_complex_roots_8():
    # 创建有理数环 R 和变量 x
    R, x = ring("x", ZZ)

    # 定义多项式 f = x^9 + 3*x^5 - 4*x
    f = x**9 + 3*x**5 - 4*x
    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (a, b) 内的复根数为 9
    assert R.dup_count_complex_roots(f, a, b) == 9
    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (c, d) 内的复根数为 4
    assert R.dup_count_complex_roots(f, c, d) == 4

    # 定义另一个多项式 f = x^11 - 2*x^9 + 3*x^7 - 6*x^5 - 4*x^3 + 8*x
    f = x**11 - 2*x**9 + 3*x**7 - 6*x**5 - 4*x**3 + 8*x
    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (a, b) 内的复根数为 9
    assert R.dup_count_complex_roots(f, a, b) == 9
    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (c, d) 内的复根数为 4
    assert R.dup_count_complex_roots(f, c, d) == 4


def test_dup_count_complex_roots_implicit():
    # 创建有理数环 R 和变量 x
    R, x = ring("x", ZZ)

    # 定义多项式 f = x^5 - x
    f = x**5 - x

    # 断言函数 R.dup_count_complex_roots 计算多项式 f 的复根数为 5
    assert R.dup_count_complex_roots(f) == 5

    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在包含 (0, 0) 的上界区间内的复根数为 3
    assert R.dup_count_complex_roots(f, sup=(0, 0)) == 3
    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在包含 (0, 0) 的下界区间内的复根数为 3
    assert R.dup_count_complex_roots(f, inf=(0, 0)) == 3


def test_dup_count_complex_roots_exclude():
    # 创建有理数环 R 和变量 x
    R, x = ring("x", ZZ)

    # 定义多项式 f = x^5 - x
    f = x**5 - x

    # 定义区间 (a, b)
    a, b = (-QQ(1), QQ(0)), (QQ(1), QQ(1))

    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (a, b) 内的复根数为 4
    assert R.dup_count_complex_roots(f, a, b) == 4

    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (a, b) 内，排除类型为 'S' 的复根后的复根数为 3
    assert R.dup_count_complex_roots(f, a, b, exclude=['S']) == 3
    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (a, b) 内，排除类型为 'N' 的复根后的复根数为 3
    assert R.dup_count_complex_roots(f, a, b, exclude=['N']) == 3

    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (a, b) 内，排除类型为 'S' 和 'N' 的复根后的复根数为 2
    assert R.dup_count_complex_roots(f, a, b, exclude=['S', 'N']) == 2

    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (a, b) 内，排除类型为 'E' 的复根后的复根数为 4
    assert R.dup_count_complex_roots(f, a, b, exclude=['E']) == 4
    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (a, b) 内，排除类型为 'W' 的复根后的复根数为 4
    assert R.dup_count_complex_roots(f, a, b, exclude=['W']) == 4

    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (a, b) 内，排除类型为 'E' 和 'W' 的复根后的复根数为 4
    assert R.dup_count_complex_roots(f, a, b, exclude=['E', 'W']) == 4

    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (a, b) 内，排除类型为 'N'、'S'、'E'、'W' 的复根后的复根数为 2
    assert R.dup_count_complex_roots(f, a, b, exclude=['N', 'S', 'E', 'W']) == 2

    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (a, b) 内，排除类型为 'SW' 的复根后的复根数为 3
    assert R.dup_count_complex_roots(f, a, b, exclude=['SW']) == 3
    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (a, b) 内，排除类型为 'SE' 的复根后的复根数为 3
    assert R.dup_count_complex_roots(f, a, b, exclude=['SE']) == 3

    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (a, b) 内，排除类型为 'SW' 和 'SE' 的复根后的复根数为 2
    assert R.dup_count_complex_roots(f, a, b, exclude=['SW', 'SE']) == 2
    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (a, b) 内，排除类型为 'SW'、'SE' 和 'S' 的复根后的复根数为 1
    assert R.dup_count_complex_roots(f, a, b, exclude=['SW', 'SE', 'S']) == 1
    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (a, b) 内，排除类型为 'SW'、'SE'、'S' 和 'N' 的复根后的复根数为 0
    assert R.dup_count_complex_roots(f, a, b, exclude=['SW', 'SE', 'S', 'N']) == 0

    # 重新定义区间 (a, b)
    a, b = (QQ(0), QQ(0)), (QQ(1), QQ(1))

    # 断言函数 R.dup_count_complex_roots 计算多项式 f 在区间 (a, b) 内，排除所有复根后的复根数为 1
    assert R.dup_count_complex_roots(f, a, b, exclude=True) == 1


def test_dup_isolate_complex_roots_sqf():
    # 创建有理数环 R 和变量 x
    R, x = ring("x", ZZ)

    # 定义多项式 f = x^2 - 2*x + 3
    f = x**2 - 2*x + 3

    # 断言函数 R.dup_isolate_complex_roots_sqf 计算多项式 f 的复根的区间列表
    assert R.dup_isolate_complex_roots_sqf(f) == \
        [((0, -6), (6, 0)), ((0, 0), (6, 6))]
    # 断言函数 R.dup_isolate_complex_roots_sqf 计算多项式 f 的复根的区间列表（使用 blackbox=True）
    assert [ r.as_tuple() for r in R.dup_isolate_complex_roots_sqf(f, blackbox=True) ] == \
        [((0, -6), (6, 0)), ((0, 0), (6, 6))]
    # 创建一个有理数环 R，并定义一个多项式环，变量为 'x'
    R, x = ring("x", ZZ)
    # 定义一个多项式 f，具体为 4*x**4 - x**3 + 2*x**2 + 5*x
    f = 4*x**4 - x**3 + 2*x**2 + 5*x

    # 断言：使用 R.dup_isolate_all_roots_sqf 函数对多项式 f 进行因式分解，并断言结果符合以下预期值
    assert R.dup_isolate_all_roots_sqf(f) == \
        # 期望的结果包含两部分：根的列表和因子的列表
        ([(-1, 0), (0, 0)],  # 根的列表，每个元组包含一个根的区间
         [((0, -QQ(5, 2)), (QQ(5, 2), 0)), ((0, 0), (QQ(5, 2), QQ(5, 2)))])  # 因子的列表，每对元组表示一个因子的区间

    # 断言：使用 R.dup_isolate_all_roots_sqf 函数对多项式 f 进行因式分解，设置精度参数为 QQ(1, 10)，并断言结果符合以下预期值
    assert R.dup_isolate_all_roots_sqf(f, eps=QQ(1, 10)) == \
        # 期望的结果包含两部分：根的列表和因子的列表
        ([(QQ(-7, 8), QQ(-6, 7)), (0, 0)],  # 根的列表，每个元组包含一个根的区间
         [((QQ(35, 64), -QQ(35, 32)), (QQ(5, 8), -QQ(65, 64))), ((QQ(35, 64), QQ(65, 64)), (QQ(5, 8), QQ(35, 32)))])  # 因子的列表，每对元组表示一个因子的区间
# 定义一个测试函数，用于测试在整数环上多项式的根的隔离操作
def test_dup_isolate_all_roots():
    # 创建一个整数环 R 和变量 x
    R, x = ring("x", ZZ)
    # 定义一个多项式 f
    f = 4*x**4 - x**3 + 2*x**2 + 5*x

    # 断言：调用 R.dup_isolate_all_roots(f) 应返回如下结果
    assert R.dup_isolate_all_roots(f) == \
        ([((-1, 0), 1), ((0, 0), 1)],  # 包含两个根的区间列表
         [(((0, -QQ(5, 2)), (QQ(5, 2), 0)), 1),  # 包含两个根的复数根列表
          (((0, 0), (QQ(5, 2), QQ(5, 2))), 1)])  # 包含两个根的复数根列表

    # 断言：调用 R.dup_isolate_all_roots(f, eps=QQ(1, 10)) 应返回如下结果
    assert R.dup_isolate_all_roots(f, eps=QQ(1, 10)) == \
        ([((QQ(-7, 8), QQ(-6, 7)), 1), ((0, 0), 1)],  # 包含两个根的区间列表
         [(((QQ(35, 64), -QQ(35, 32)), (QQ(5, 8), -QQ(65, 64))), 1),  # 包含两个根的复数根列表
          (((QQ(35, 64), QQ(65, 64)), (QQ(5, 8), QQ(35, 32))), 1)])  # 包含两个根的复数根列表

    # 定义另一个多项式 f
    f = x**5 + x**4 - 2*x**3 - 2*x**2 + x + 1
    # 断言：调用 R.dup_isolate_all_roots(f) 应引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: R.dup_isolate_all_roots(f))
```