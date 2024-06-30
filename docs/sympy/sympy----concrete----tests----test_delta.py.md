# `D:\src\scipysrc\sympy\sympy\concrete\tests\test_delta.py`

```
# 从 sympy.concrete 模块导入 Sum 类
from sympy.concrete import Sum
# 从 sympy.concrete.delta 模块导入 deltaproduct 和 deltasummation 函数，并导入 _extract_delta 函数
from sympy.concrete.delta import deltaproduct as dp, deltasummation as ds, _extract_delta
# 从 sympy.core 模块导入 Eq, S, symbols 和 oo
from sympy.core import Eq, S, symbols, oo
# 从 sympy.functions 模块导入 KroneckerDelta 别名为 KD, Piecewise 和 piecewise_fold 函数
from sympy.functions import KroneckerDelta as KD, Piecewise, piecewise_fold
# 从 sympy.logic 模块导入 And 函数
from sympy.logic import And
# 从 sympy.testing.pytest 模块导入 raises 函数

# 创建整数符号 i, j, k, l, m 和非交换符号 x, y 的符号对象
i, j, k, l, m = symbols("i j k l m", integer=True, finite=True)
x, y = symbols("x y", commutative=False)

# 定义测试函数 test_deltaproduct_trivial，测试 deltaproduct 函数的基本情况
def test_deltaproduct_trivial():
    # 断言 deltaproduct(x, (j, 1, 0)) 的结果为 1
    assert dp(x, (j, 1, 0)) == 1
    # 断言 deltaproduct(x, (j, 1, 3)) 的结果为 x 的三次方
    assert dp(x, (j, 1, 3)) == x**3
    # 断言 deltaproduct(x + y, (j, 1, 3)) 的结果为 (x + y) 的三次方
    assert dp(x + y, (j, 1, 3)) == (x + y)**3
    # 断言 deltaproduct(x*y, (j, 1, 3)) 的结果为 (x*y) 的三次方
    assert dp(x*y, (j, 1, 3)) == (x*y)**3
    # 断言 deltaproduct(KD(i, j), (k, 1, 3)) 的结果为 KD(i, j)
    assert dp(KD(i, j), (k, 1, 3)) == KD(i, j)
    # 断言 deltaproduct(x*KD(i, j), (k, 1, 3)) 的结果为 x 的三次方乘以 KD(i, j)
    assert dp(x*KD(i, j), (k, 1, 3)) == x**3*KD(i, j)
    # 断言 deltaproduct(x*y*KD(i, j), (k, 1, 3)) 的结果为 (x*y) 的三次方乘以 KD(i, j)
    assert dp(x*y*KD(i, j), (k, 1, 3)) == (x*y)**3*KD(i, j)

# 定义测试函数 test_deltaproduct_basic，测试 deltaproduct 函数在基本情况下的行为
def test_deltaproduct_basic():
    # 断言 deltaproduct(KD(i, j), (j, 1, 3)) 的结果为 0
    assert dp(KD(i, j), (j, 1, 3)) == 0
    # 断言 deltaproduct(KD(i, j), (j, 1, 1)) 的结果为 KD(i, 1)
    assert dp(KD(i, j), (j, 1, 1)) == KD(i, 1)
    # 断言 deltaproduct(KD(i, j), (j, 2, 2)) 的结果为 KD(i, 2)
    assert dp(KD(i, j), (j, 2, 2)) == KD(i, 2)
    # 断言 deltaproduct(KD(i, j), (j, 3, 3)) 的结果为 KD(i, 3)
    assert dp(KD(i, j), (j, 3, 3)) == KD(i, 3)
    # 断言 deltaproduct(KD(i, j), (j, 1, k)) 的结果为 KD(i, 1)*KD(k, 1) + KD(k, 0)
    assert dp(KD(i, j), (j, 1, k)) == KD(i, 1)*KD(k, 1) + KD(k, 0)
    # 断言 deltaproduct(KD(i, j), (j, k, 3)) 的结果为 KD(i, 3)*KD(k, 3) + KD(k, 4)
    assert dp(KD(i, j), (j, k, 3)) == KD(i, 3)*KD(k, 3) + KD(k, 4)
    # 断言 deltaproduct(KD(i, j), (j, k, l)) 的结果为 KD(i, l)*KD(k, l) + KD(k, l + 1)
    assert dp(KD(i, j), (j, k, l)) == KD(i, l)*KD(k, l) + KD(k, l + 1)

# 定义测试函数 test_deltaproduct_mul_x_kd，测试 deltaproduct 函数在 x*KD(i, j) 的情况下的行为
def test_deltaproduct_mul_x_kd():
    # 断言 deltaproduct(x*KD(i, j), (j, 1, 3)) 的结果为 0
    assert dp(x*KD(i, j), (j, 1, 3)) == 0
    # 断言 deltaproduct(x*KD(i, j), (j, 1, 1)) 的结果为 x*KD(i, 1)
    assert dp(x*KD(i, j), (j, 1, 1)) == x*KD(i, 1)
    # 断言 deltaproduct(x*KD(i, j), (j, 2, 2)) 的结果为 x*KD(i, 2)
    assert dp(x*KD(i, j), (j, 2, 2)) == x*KD(i, 2)
    # 断言 deltaproduct(x*KD(i, j), (j, 3, 3)) 的结果为 x*KD(i, 3)
    assert dp(x*KD(i, j), (j, 3, 3)) == x*KD(i, 3)
    # 断言 deltaproduct(x*KD(i, j), (j, 1, k)) 的结果为 x*KD(i, 1)*KD(k, 1) + KD(k, 0)
    assert dp(x*KD(i, j), (j, 1, k)) == x*KD(i, 1)*KD(k, 1) + KD(k, 0)
    # 断言 deltaproduct(x*KD(i, j), (j, k, 3)) 的结果为 x*KD(i, 3)*KD(k, 3) + KD(k, 4)
    assert dp(x*KD(i, j), (j, k, 3)) == x*KD(i, 3)*KD(k, 3) + KD(k, 4)
    # 断言 deltaproduct(x*KD(i, j), (j, k, l)) 的结果为 x*KD(i, l)*KD(k, l) + KD(k, l + 1)
    assert dp(x*KD(i, j), (j, k, l)) == x*KD(i, l)*KD(k, l) + KD(k, l + 1)

# 定义测试函数 test_deltaproduct_mul_add_x_y_kd，测试 deltaproduct 函数在 (x + y)*KD(i, j) 的情况下的行为
def test_deltaproduct_mul_add_x_y_kd():
    # 断言 deltaproduct((x + y)*KD(i, j), (j, 1, 3)) 的结果为 0
    assert dp((x + y)*KD(i, j), (j, 1, 3)) == 0
    # 断言 deltaproduct((x + y)*KD(i, j), (j, 1, 1)) 的结果为 (x + y)*KD(i, 1)
    assert dp((x + y)*KD(i, j), (j, 1, 1)) == (x + y)*KD(i, 1)
    # 断言 deltaproduct((x + y)*KD(i, j), (j, 2, 2)) 的结果为 (x + y)*KD(i, 2)
    assert dp((x + y)*KD(i, j), (j, 2, 2))
    # 断言语句，验证表达式是否为真
    assert dp(KD(i, k) + KD(j, k), (k, l, m)) == \
        # 计算 KD(l, m + 1) 和以下四个项的总和
        KD(l, m + 1) + \
        KD(i, m) * KD(l, m) + \
        KD(j, m) * KD(l, m) + \
        KD(i, m) * KD(j, m - 1) * KD(l, m - 1) + \
        KD(i, m - 1) * KD(j, m) * KD(l, m - 1)
# 定义测试函数 test_deltaproduct_mul_x_add_kd_kd，用于测试带有乘法和 Kronecker δ 函数的 delta product 函数 dp 的行为
def test_deltaproduct_mul_x_add_kd_kd():
    # 断言：当 k 在 1 到 3 范围内时，x*(KD(i, k) + KD(j, k)) 的 delta product dp 结果应为 0
    assert dp(x*(KD(i, k) + KD(j, k)), (k, 1, 3)) == 0
    # 断言：当 k 在 1 到 1 范围内时，x*(KD(i, k) + KD(j, k)) 的 delta product dp 结果应为 x*(KD(i, 1) + KD(j, 1))
    assert dp(x*(KD(i, k) + KD(j, k)), (k, 1, 1)) == x*(KD(i, 1) + KD(j, 1))
    # 断言：当 k 在 2 到 2 范围内时，x*(KD(i, k) + KD(j, k)) 的 delta product dp 结果应为 x*(KD(i, 2) + KD(j, 2))
    assert dp(x*(KD(i, k) + KD(j, k)), (k, 2, 2)) == x*(KD(i, 2) + KD(j, 2))
    # 断言：当 k 在 3 到 3 范围内时，x*(KD(i, k) + KD(j, k)) 的 delta product dp 结果应为 x*(KD(i, 3) + KD(j, 3))
    assert dp(x*(KD(i, k) + KD(j, k)), (k, 3, 3)) == x*(KD(i, 3) + KD(j, 3))
    # 断言：当 k 在 1 到 l 范围内时，x*(KD(i, k) + KD(j, k)) 的 delta product dp 结果应为一系列项的和
    assert dp(x*(KD(i, k) + KD(j, k)), (k, 1, l)) == KD(l, 0) + \
        x*KD(i, 1)*KD(l, 1) + x*KD(j, 1)*KD(l, 1) + \
        x**2*KD(i, 1)*KD(j, 2)*KD(l, 2) + x**2*KD(j, 1)*KD(i, 2)*KD(l, 2)
    # 断言：当 k 在 l 到 3 范围内时，x*(KD(i, k) + KD(j, k)) 的 delta product dp 结果应为一系列项的和
    assert dp(x*(KD(i, k) + KD(j, k)), (k, l, 3)) == KD(l, 4) + \
        x*KD(i, 3)*KD(l, 3) + x*KD(j, 3)*KD(l, 3) + \
        x**2*KD(i, 2)*KD(j, 3)*KD(l, 2) + x**2*KD(i, 3)*KD(j, 2)*KD(l, 2)
    # 断言：当 k 在 l 到 m 范围内时，x*(KD(i, k) + KD(j, k)) 的 delta product dp 结果应为一系列项的和
    assert dp(x*(KD(i, k) + KD(j, k)), (k, l, m)) == KD(l, m + 1) + \
        x*KD(i, m)*KD(l, m) + x*KD(j, m)*KD(l, m) + \
        x**2*KD(i, m - 1)*KD(j, m)*KD(l, m - 1) + \
        x**2*KD(i, m)*KD(j, m - 1)*KD(l, m - 1)


# 定义测试函数 test_deltaproduct_mul_add_x_y_add_kd_kd，用于测试带有加法、乘法和 Kronecker δ 函数的 delta product 函数 dp 的行为
def test_deltaproduct_mul_add_x_y_add_kd_kd():
    # 断言：当 k 在 1 到 3 范围内时，(x + y)*(KD(i, k) + KD(j, k)) 的 delta product dp 结果应为 0
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, 1, 3)) == 0
    # 断言：当 k 在 1 到 1 范围内时，(x + y)*(KD(i, k) + KD(j, k)) 的 delta product dp 结果应为 (x + y)*(KD(i, 1) + KD(j, 1))
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, 1, 1)) == \
        (x + y)*(KD(i, 1) + KD(j, 1))
    # 断言：当 k 在 2 到 2 范围内时，(x + y)*(KD(i, k) + KD(j, k)) 的 delta product dp 结果应为 (x + y)*(KD(i, 2) + KD(j, 2))
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, 2, 2)) == \
        (x + y)*(KD(i, 2) + KD(j, 2))
    # 断言：当 k 在 3 到 3 范围内时，(x + y)*(KD(i, k) + KD(j, k)) 的 delta product dp 结果应为 (x + y)*(KD(i, 3) + KD(j, 3))
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, 3, 3)) == \
        (x + y)*(KD(i, 3) + KD(j, 3))
    # 断言：当 k 在 1 到 l 范围内时，(x + y)*(KD(i, k) + KD(j, k)) 的 delta product dp 结果应为一系列项的和
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, 1, l)) == KD(l, 0) + \
        (x + y)*KD(i, 1)*KD(l, 1) + (x + y)*KD(j, 1)*KD(l, 1) + \
        (x + y)**2*KD(i, 1)*KD(j, 2)*KD(l, 2) + \
        (x + y)**2*KD(j, 1)*KD(i, 2)*KD(l, 2)
    # 断言：当 k 在 l 到 3 范围内时，(x + y)*(KD(i, k) + KD(j, k)) 的 delta product dp 结果应为一系列项的和
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, l, 3)) == KD(l, 4) + \
        (x + y)*KD(i, 3)*KD(l, 3) + (x + y)*KD(j, 3)*KD(l, 3) + \
        (x + y)**2*KD(i, 2)*KD(j, 3)*KD(l, 2) + \
        (x + y)**2*KD(i, 3)*KD(j, 2)*KD(l, 2)
    # 断言：当 k 在 l 到 m 范围内时，(x + y)*(KD(i, k) + KD(j, k)) 的 delta product dp 结果应为一系列项的和
    assert dp((x + y)*(KD(i, k) + KD(j, k)), (k, l, m)) == KD(l, m + 1) + \
        (x + y)*KD(i, m)*KD(l, m) + (x + y)*KD(j, m)*KD(l, m) + \
        (x + y)**2*KD(i, m - 1)*KD(j, m)*KD(l, m - 1) + \
        (x + y)**2*KD(i, m)*KD(j, m - 1)*KD(l, m - 1)


# 定义测试函数 test_deltaproduct_add_mul_x_y_mul_x_kd，用于测试带有加法、乘法和 Kronecker δ 函数的 delta product 函数 dp 的行为
def test_deltaproduct_add_mul_x_y_mul_x_kd():
    # 断言：当 j 在 1 到 3 范围内时，x
# 定义一个测试函数，用于测试带有特定形式的 delta product 表达式
def test_deltaproduct_mul_x_add_y_kd():
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp(x*(y + KD(i, j)), (j, 1, 3)) == (x*y)**3 + \
        x*(x*y)**2*KD(i, 1) + (x*y)*x*(x*y)*KD(i, 2) + (x*y)**2*x*KD(i, 3)
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp(x*(y + KD(i, j)), (j, 1, 1)) == x*(y + KD(i, 1))
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp(x*(y + KD(i, j)), (j, 2, 2)) == x*(y + KD(i, 2))
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp(x*(y + KD(i, j)), (j, 3, 3)) == x*(y + KD(i, 3))
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp(x*(y + KD(i, j)), (j, 1, k)) == \
        (x*y)**k + Piecewise(
            ((x*y)**(i - 1)*x*(x*y)**(k - i), And(1 <= i, i <= k)),
            (0, True)
        ).expand()
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp(x*(y + KD(i, j)), (j, k, 3)) == \
        ((x*y)**(-k + 4) + Piecewise(
            ((x*y)**(i - k)*x*(x*y)**(3 - i), And(k <= i, i <= 3)),
            (0, True)
        )).expand()
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp(x*(y + KD(i, j)), (j, k, l)) == \
        ((x*y)**(-k + l + 1) + Piecewise(
            ((x*y)**(i - k)*x*(x*y)**(l - i), And(k <= i, i <= l)),
            (0, True)
        )).expand()


# 定义另一个测试函数，测试带有双倍 KD 的 delta product 表达式
def test_deltaproduct_mul_x_add_y_twokd():
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp(x*(y + 2*KD(i, j)), (j, 1, 3)) == (x*y)**3 + \
        2*x*(x*y)**2*KD(i, 1) + 2*x*y*x*x*y*KD(i, 2) + 2*(x*y)**2*x*KD(i, 3)
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp(x*(y + 2*KD(i, j)), (j, 1, 1)) == x*(y + 2*KD(i, 1))
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp(x*(y + 2*KD(i, j)), (j, 2, 2)) == x*(y + 2*KD(i, 2))
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp(x*(y + 2*KD(i, j)), (j, 3, 3)) == x*(y + 2*KD(i, 3))
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp(x*(y + 2*KD(i, j)), (j, 1, k)) == \
        (x*y)**k + Piecewise(
            (2*(x*y)**(i - 1)*x*(x*y)**(k - i), And(1 <= i, i <= k)),
            (0, True)
        ).expand()
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp(x*(y + 2*KD(i, j)), (j, k, 3)) == \
        ((x*y)**(-k + 4) + Piecewise(
            (2*(x*y)**(i - k)*x*(x*y)**(3 - i), And(k <= i, i <= 3)),
            (0, True)
        )).expand()
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp(x*(y + 2*KD(i, j)), (j, k, l)) == \
        ((x*y)**(-k + l + 1) + Piecewise(
            (2*(x*y)**(i - k)*x*(x*y)**(l - i), And(k <= i, i <= l)),
            (0, True)
        )).expand()


# 定义另一个测试函数，测试带有多项式和 KD 的 delta product 表达式
def test_deltaproduct_mul_add_x_y_add_y_kd():
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp((x + y)*(y + KD(i, j)), (j, 1, 3)) == ((x + y)*y)**3 + \
        (x + y)*((x + y)*y)**2*KD(i, 1) + \
        (x + y)*y*(x + y)**2*y*KD(i, 2) + \
        ((x + y)*y)**2*(x + y)*KD(i, 3)
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp((x + y)*(y + KD(i, j)), (j, 1, 1)) == (x + y)*(y + KD(i, 1))
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp((x + y)*(y + KD(i, j)), (j, 2, 2)) == (x + y)*(y + KD(i, 2))
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp((x + y)*(y + KD(i, j)), (j, 3, 3)) == (x + y)*(y + KD(i, 3))
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp((x + y)*(y + KD(i, j)), (j, 1, k)) == \
        ((x + y)*y)**k + Piecewise(
            (((x + y)*y)**(-1)*((x + y)*y)**i*(x + y)*((x + y)*y)**k*((x + y)*y)**(-i), (i >= 1) & (i <= k)),
            (0, True)
        )
    # 断言测试 dp 函数对于特定参数的计算结果
    assert dp((x + y)*(y + KD(i, j)), (j, k, 3)) == (
        (x + y)*y)**4*((x + y)*y)**(-k) + Piecewise((((x + y)*y)**i*((x + y)*y)**(-k)*(x + y)*((x + y)*y)**3*((x + y)*y)**(-i),
        (i >= k) & (i <= 3)), (0, True))
    # 断言语句，验证 dp 函数在给定参数 (x + y)*(y + KD(i, j)) 时的返回值是否等于下面的表达式
    assert dp((x + y)*(y + KD(i, j)), (j, k, l)) == \
        # 下面是表达式的具体形式，包含两部分的和：一部分是常数项，另一部分是 Piecewise 函数
        (x + y)*y*((x + y)*y)**l*((x + y)*y)**(-k) + Piecewise(
            # 第一分支：当条件 (i >= k) & (i <= l) 满足时，返回以下表达式
            (((x + y)*y)**i*((x + y)*y)**(-k)*(x + y)*((x + y)*y)**l*((x + y)*y)**(-i), (i >= k) & (i <= l)),
            # 第二分支：默认情况下返回 0
            (0, True))
# 定义一个测试函数，用于测试特定的数学表达式与期望结果是否相等
def test_deltaproduct_mul_add_x_kd_add_y_kd():
    # 断言测试表达式的求和形式与预期结果相等
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, 1, 3)) == \
        KD(i, 1)*(KD(i, k) + x)*((KD(i, k) + x)*y)**2 + \
        KD(i, 2)*(KD(i, k) + x)*y*(KD(i, k) + x)**2*y + \
        KD(i, 3)*((KD(i, k) + x)*y)**2*(KD(i, k) + x) + \
        ((KD(i, k) + x)*y)**3
    
    # 断言测试表达式的求和形式与预期结果相等
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, 1, 1)) == \
        (x + KD(i, k))*(y + KD(i, 1))
    
    # 断言测试表达式的求和形式与预期结果相等
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, 2, 2)) == \
        (x + KD(i, k))*(y + KD(i, 2))
    
    # 断言测试表达式的求和形式与预期结果相等
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, 3, 3)) == \
        (x + KD(i, k))*(y + KD(i, 3))
    
    # 断言测试表达式的求和形式与预期结果相等
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, 1, k)) == \
        ((KD(i, k) + x)*y)**k + Piecewise(
        (((KD(i, k) + x)*y)**(-1)*((KD(i, k) + x)*y)**i*(KD(i, k) + x)*
        ((KD(i, k) + x)*y)**k*((KD(i, k) + x)*y)**(-i), (i >= 1) & (i <= k)), (0, True))
    
    # 断言测试表达式的求和形式与预期结果相等
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, k, 3)) == (
        (KD(i, k) + x)*y)**4*((KD(i, k) + x)*y)**(-k) + Piecewise(
        (((KD(i, k) + x)*y)**i*((KD(i, k) + x)*y)**(-k)*(KD(i, k) + x)*
        ((KD(i, k) + x)*y)**3*((KD(i, k) + x)*y)**(-i),
        (i >= k) & (i <= 3)), (0, True))
    
    # 断言测试表达式的求和形式与预期结果相等
    assert dp((x + KD(i, k))*(y + KD(i, j)), (j, k, l)) == (
        KD(i, k) + x)*y*((KD(i, k) + x)*y)**l*((KD(i, k) + x)*y)**(-k) + Piecewise(
        (((KD(i, k) + x)*y)**i*((KD(i, k) + x)*y)**(-k)*(KD(i, k) + x)*
        ((KD(i, k) + x)*y)**l*((KD(i, k) + x)*y)**(-i), (i >= k) & (i <= l)), (0, True))


# 定义一个测试函数，用于测试特定的数学表达式与期望结果是否相等
def test_deltasummation_trivial():
    # 断言测试表达式与预期结果相等
    assert ds(x, (j, 1, 0)) == 0
    
    # 断言测试表达式与预期结果相等
    assert ds(x, (j, 1, 3)) == 3*x
    
    # 断言测试表达式与预期结果相等
    assert ds(x + y, (j, 1, 3)) == 3*(x + y)
    
    # 断言测试表达式与预期结果相等
    assert ds(x*y, (j, 1, 3)) == 3*x*y
    
    # 断言测试表达式与预期结果相等
    assert ds(KD(i, j), (k, 1, 3)) == 3*KD(i, j)
    
    # 断言测试表达式与预期结果相等
    assert ds(x*KD(i, j), (k, 1, 3)) == 3*x*KD(i, j)
    
    # 断言测试表达式与预期结果相等
    assert ds(x*y*KD(i, j), (k, 1, 3)) == 3*x*y*KD(i, j)


# 定义一个测试函数，用于测试特定的数学表达式与期望结果是否相等
def test_deltasummation_basic_numerical():
    # 声明一个整数符号 n
    n = symbols('n', integer=True, nonzero=True)
    
    # 断言测试表达式与预期结果相等
    assert ds(KD(n, 0), (n, 1, 3)) == 0
    
    # 断言测试表达式与预期结果相等，返回未评估状态，直到实现
    assert ds(KD(i**2, j**2), (j, -oo, oo)) == \
        Sum(KD(i**2, j**2), (j, -oo, oo))
    
    # 断言测试表达式与预期结果相等
    assert Piecewise((KD(i, k), And(1 <= i, i <= 3)), (0, True)) == \
        ds(KD(i, j)*KD(j, k), (j, 1, 3)) == \
        ds(KD(j, k)*KD(i, j), (j, 1, 3))
    
    # 断言测试表达式与预期结果相等
    assert ds(KD(i, k), (k, -oo, oo)) == 1
    
    # 断言测试表达式与预期结果相等
    assert ds(KD(i, k), (k, 0, oo)) == Piecewise((1, S.Zero <= i), (0, True))
    
    # 断言测试表达式与预期结果相等
    assert ds(KD(i, k), (k, 1, 3)) == \
        Piecewise((1, And(1 <= i, i <= 3)), (0, True))
    
    # 断言测试表达式与预期结果相等
    assert ds(k*KD(i, j)*KD(j, k), (k, -oo, oo)) == j*KD(i, j)
    
    # 断言测试表达式与预期结果相等
    assert ds(j*KD(i, j), (j, -oo, oo)) == i
    
    # 断言测试表达式与预期结果相等
    assert ds(i*KD(i, j), (i, -oo, oo)) == j
    
    # 断言测试表达式与预期结果相等
    assert ds(x, (i, 1, 3)) == 3*x
    
    # 断言测试表达式与预期结果相等
    assert ds((i + j)*KD(i, j), (j, -oo, oo)) == 2*i


# 定义一个测试函数，用于测试特定的数学表达式与期望结果是否相等
def test_deltasummation_basic_symbolic():
    # 断言测试表达式与预期结果相等
    assert ds(KD(i, j), (j, 1, 3)) == \
        Piecewise((1, And(1 <= i, i <= 3)), (0, True))
    
    # 断言测试表达式与预期结果相等
    assert ds(KD(i, j), (j, 1, 1)) == Piecewise((1, Eq(i, 1)), (0, True))
    # 对称导数的断言，验证对称导数函数 `ds` 的返回结果是否符合预期
    
    # 验证 ds(KD(i, j), (j, 2, 2)) 的返回结果是否等于 Piecewise((1, Eq(i, 2)), (0, True))
    assert ds(KD(i, j), (j, 2, 2)) == Piecewise((1, Eq(i, 2)), (0, True))
    
    # 验证 ds(KD(i, j), (j, 3, 3)) 的返回结果是否等于 Piecewise((1, Eq(i, 3)), (0, True))
    assert ds(KD(i, j), (j, 3, 3)) == Piecewise((1, Eq(i, 3)), (0, True))
    
    # 验证 ds(KD(i, j), (j, 1, k)) 的返回结果是否等于 Piecewise((1, And(1 <= i, i <= k)), (0, True))
    assert ds(KD(i, j), (j, 1, k)) == Piecewise((1, And(1 <= i, i <= k)), (0, True))
    
    # 验证 ds(KD(i, j), (j, k, 3)) 的返回结果是否等于 Piecewise((1, And(k <= i, i <= 3)), (0, True))
    assert ds(KD(i, j), (j, k, 3)) == Piecewise((1, And(k <= i, i <= 3)), (0, True))
    
    # 验证 ds(KD(i, j), (j, k, l)) 的返回结果是否等于 Piecewise((1, And(k <= i, i <= l)), (0, True))
    assert ds(KD(i, j), (j, k, l)) == Piecewise((1, And(k <= i, i <= l)), (0, True))
# 定义函数 test_deltasummation_mul_x_kd，用于测试 ds 函数对 x*KD(i, j) 表达式的求导和分段函数的正确性
def test_deltasummation_mul_x_kd():
    # 断言表达式 ds(x*KD(i, j), (j, 1, 3)) 的返回值应符合 Piecewise((x, And(1 <= i, i <= 3)), (0, True)) 的条件
    assert ds(x*KD(i, j), (j, 1, 3)) == \
        Piecewise((x, And(1 <= i, i <= 3)), (0, True))
    # 断言表达式 ds(x*KD(i, j), (j, 1, 1)) 的返回值应符合 Piecewise((x, Eq(i, 1)), (0, True)) 的条件
    assert ds(x*KD(i, j), (j, 1, 1)) == Piecewise((x, Eq(i, 1)), (0, True))
    # 断言表达式 ds(x*KD(i, j), (j, 2, 2)) 的返回值应符合 Piecewise((x, Eq(i, 2)), (0, True)) 的条件
    assert ds(x*KD(i, j), (j, 2, 2)) == Piecewise((x, Eq(i, 2)), (0, True))
    # 断言表达式 ds(x*KD(i, j), (j, 3, 3)) 的返回值应符合 Piecewise((x, Eq(i, 3)), (0, True)) 的条件
    assert ds(x*KD(i, j), (j, 3, 3)) == Piecewise((x, Eq(i, 3)), (0, True))
    # 断言表达式 ds(x*KD(i, j), (j, 1, k)) 的返回值应符合 Piecewise((x, And(1 <= i, i <= k)), (0, True)) 的条件
    assert ds(x*KD(i, j), (j, 1, k)) == \
        Piecewise((x, And(1 <= i, i <= k)), (0, True))
    # 断言表达式 ds(x*KD(i, j), (j, k, 3)) 的返回值应符合 Piecewise((x, And(k <= i, i <= 3)), (0, True)) 的条件
    assert ds(x*KD(i, j), (j, k, 3)) == \
        Piecewise((x, And(k <= i, i <= 3)), (0, True))
    # 断言表达式 ds(x*KD(i, j), (j, k, l)) 的返回值应符合 Piecewise((x, And(k <= i, i <= l)), (0, True)) 的条件
    assert ds(x*KD(i, j), (j, k, l)) == \
        Piecewise((x, And(k <= i, i <= l)), (0, True))


# 定义函数 test_deltasummation_mul_add_x_y_kd，用于测试 ds 函数对 (x + y)*KD(i, j) 表达式的求导和分段函数的正确性
def test_deltasummation_mul_add_x_y_kd():
    # 断言表达式 ds((x + y)*KD(i, j), (j, 1, 3)) 的返回值应符合 Piecewise((x + y, And(1 <= i, i <= 3)), (0, True)) 的条件
    assert ds((x + y)*KD(i, j), (j, 1, 3)) == \
        Piecewise((x + y, And(1 <= i, i <= 3)), (0, True))
    # 断言表达式 ds((x + y)*KD(i, j), (j, 1, 1)) 的返回值应符合 Piecewise((x + y, Eq(i, 1)), (0, True)) 的条件
    assert ds((x + y)*KD(i, j), (j, 1, 1)) == \
        Piecewise((x + y, Eq(i, 1)), (0, True))
    # 断言表达式 ds((x + y)*KD(i, j), (j, 2, 2)) 的返回值应符合 Piecewise((x + y, Eq(i, 2)), (0, True)) 的条件
    assert ds((x + y)*KD(i, j), (j, 2, 2)) == \
        Piecewise((x + y, Eq(i, 2)), (0, True))
    # 断言表达式 ds((x + y)*KD(i, j), (j, 3, 3)) 的返回值应符合 Piecewise((x + y, Eq(i, 3)), (0, True)) 的条件
    assert ds((x + y)*KD(i, j), (j, 3, 3)) == \
        Piecewise((x + y, Eq(i, 3)), (0, True))
    # 断言表达式 ds((x + y)*KD(i, j), (j, 1, k)) 的返回值应符合 Piecewise((x + y, And(1 <= i, i <= k)), (0, True)) 的条件
    assert ds((x + y)*KD(i, j), (j, 1, k)) == \
        Piecewise((x + y, And(1 <= i, i <= k)), (0, True))
    # 断言表达式 ds((x + y)*KD(i, j), (j, k, 3)) 的返回值应符合 Piecewise((x + y, And(k <= i, i <= 3)), (0, True)) 的条件
    assert ds((x + y)*KD(i, j), (j, k, 3)) == \
        Piecewise((x + y, And(k <= i, i <= 3)), (0, True))
    # 断言表达式 ds((x + y)*KD(i, j), (j, k, l)) 的返回值应符合 Piecewise((x + y, And(k <= i, i <= l)), (0, True)) 的条件
    assert ds((x + y)*KD(i, j), (j, k, l)) == \
        Piecewise((x + y, And(k <= i, i <= l)), (0, True))


# 定义函数 test_deltasummation_add_kd_kd，用于测试 ds 函数对 KD(i, k) + KD(j, k) 表达式的求导和分段函数的正确性
def test_deltasummation_add_kd_kd():
    # 断言表达式 ds(KD(i, k) + KD(j, k), (k, 1, 3)) 的返回值应符合 piecewise_fold(Piecewise((1, And(1 <= i, i <= 3)), (0, True)) + Piecewise((1, And(1 <= j, j <= 3)), (0, True))) 的条件
    assert ds(KD(i, k) + KD(j, k), (k, 1, 3)) == piecewise_fold(
        Piecewise((1, And(1 <= i, i <= 3)), (0, True)) +
        Piecewise((1, And(1 <= j, j <= 3)), (0, True)))
    # 断言表达式 ds(KD(i, k) + KD(j, k), (k, 1, 1)) 的返回值应符合 piecewise_fold(Piecewise((1, Eq(i, 1)), (0, True)) + Piecewise((1, Eq(j, 1)), (0, True))) 的条件
    assert ds(KD(i, k) + KD(j, k), (k, 1, 1)) == piecewise_fold(
        Piecewise((1, Eq(i, 1)), (0, True)) +
        Piecewise((1, Eq(j, 1)), (0, True)))
    # 断言表达式 ds(KD(i, k) + KD(j, k), (k, 2, 2)) 的返回值应符合 piecewise_fold(Piecewise((1, Eq(i, 2)), (0, True)) + Piecewise((1, Eq(j, 2)), (0, True))) 的条件
    assert ds(KD(i, k) + KD(j, k), (k, 2, 2)) == piecewise_fold(
        Piecewise((1, Eq(i, 2)), (0, True)) +
        Piecewise((1, Eq(j, 2)), (0, True)))
    # 断言表达式 ds(KD(i, k) + KD(j, k), (k, 3, 3)) 的返回值应符合 piecewise_fold(Piecewise((1, Eq(i, 3)), (0, True)) + Piecewise((1, Eq(j, 3)), (0, True))) 的条件
    assert ds(KD(i,
    # 验证偏微分的等式是否成立，针对不同的 k 范围进行断言
    
    assert ds(x*KD(i, k) + KD(j, k), (k, 1, 1)) == piecewise_fold(
        # 对于 k = 1 的情况，根据 i 和 j 的值选择合适的分段函数结果
        Piecewise((x, Eq(i, 1)), (0, True)) +
        Piecewise((1, Eq(j, 1)), (0, True)))
    
    assert ds(x*KD(i, k) + KD(j, k), (k, 2, 2)) == piecewise_fold(
        # 对于 k = 2 的情况，根据 i 和 j 的值选择合适的分段函数结果
        Piecewise((x, Eq(i, 2)), (0, True)) +
        Piecewise((1, Eq(j, 2)), (0, True)))
    
    assert ds(x*KD(i, k) + KD(j, k), (k, 3, 3)) == piecewise_fold(
        # 对于 k = 3 的情况，根据 i 和 j 的值选择合适的分段函数结果
        Piecewise((x, Eq(i, 3)), (0, True)) +
        Piecewise((1, Eq(j, 3)), (0, True)))
    
    assert ds(x*KD(i, k) + KD(j, k), (k, 1, l)) == piecewise_fold(
        # 对于 1 <= k <= l 的情况，根据 i 和 j 的值选择合适的分段函数结果
        Piecewise((x, And(1 <= i, i <= l)), (0, True)) +
        Piecewise((1, And(1 <= j, j <= l)), (0, True)))
    
    assert ds(x*KD(i, k) + KD(j, k), (k, l, 3)) == piecewise_fold(
        # 对于 l <= k <= 3 的情况，根据 i 和 j 的值选择合适的分段函数结果
        Piecewise((x, And(l <= i, i <= 3)), (0, True)) +
        Piecewise((1, And(l <= j, j <= 3)), (0, True)))
    
    assert ds(x*KD(i, k) + KD(j, k), (k, l, m)) == piecewise_fold(
        # 对于 l <= k <= m 的情况，根据 i 和 j 的值选择合适的分段函数结果
        Piecewise((x, And(l <= i, i <= m)), (0, True)) +
        Piecewise((1, And(l <= j, j <= m)), (0, True)))
# 定义一个函数用于测试 delta_summation 的乘法和加法操作
def test_deltasummation_mul_x_add_kd_kd():
    # 断言：对 x*(KD(i, k) + KD(j, k)) 进行 delta_summation 操作
    assert ds(x*(KD(i, k) + KD(j, k)), (k, 1, 3)) == piecewise_fold(
        # 对结果进行分段函数折叠
        Piecewise((x, And(1 <= i, i <= 3)), (0, True)) +
        Piecewise((x, And(1 <= j, j <= 3)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, 1, 1)) == piecewise_fold(
        Piecewise((x, Eq(i, 1)), (0, True)) +
        Piecewise((x, Eq(j, 1)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, 2, 2)) == piecewise_fold(
        Piecewise((x, Eq(i, 2)), (0, True)) +
        Piecewise((x, Eq(j, 2)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, 3, 3)) == piecewise_fold(
        Piecewise((x, Eq(i, 3)), (0, True)) +
        Piecewise((x, Eq(j, 3)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, 1, l)) == piecewise_fold(
        Piecewise((x, And(1 <= i, i <= l)), (0, True)) +
        Piecewise((x, And(1 <= j, j <= l)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, l, 3)) == piecewise_fold(
        Piecewise((x, And(l <= i, i <= 3)), (0, True)) +
        Piecewise((x, And(l <= j, j <= 3)), (0, True)))
    assert ds(x*(KD(i, k) + KD(j, k)), (k, l, m)) == piecewise_fold(
        Piecewise((x, And(l <= i, i <= m)), (0, True)) +
        Piecewise((x, And(l <= j, j <= m)), (0, True)))


# 定义一个函数用于测试 delta_summation 的加法、乘法和加法操作
def test_deltasummation_mul_add_x_y_add_kd_kd():
    # 断言：对 (x + y)*(KD(i, k) + KD(j, k)) 进行 delta_summation 操作
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, 1, 3)) == piecewise_fold(
        Piecewise((x + y, And(1 <= i, i <= 3)), (0, True)) +
        Piecewise((x + y, And(1 <= j, j <= 3)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, 1, 1)) == piecewise_fold(
        Piecewise((x + y, Eq(i, 1)), (0, True)) +
        Piecewise((x + y, Eq(j, 1)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, 2, 2)) == piecewise_fold(
        Piecewise((x + y, Eq(i, 2)), (0, True)) +
        Piecewise((x + y, Eq(j, 2)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, 3, 3)) == piecewise_fold(
        Piecewise((x + y, Eq(i, 3)), (0, True)) +
        Piecewise((x + y, Eq(j, 3)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, 1, l)) == piecewise_fold(
        Piecewise((x + y, And(1 <= i, i <= l)), (0, True)) +
        Piecewise((x + y, And(1 <= j, j <= l)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, l, 3)) == piecewise_fold(
        Piecewise((x + y, And(l <= i, i <= 3)), (0, True)) +
        Piecewise((x + y, And(l <= j, j <= 3)), (0, True)))
    assert ds((x + y)*(KD(i, k) + KD(j, k)), (k, l, m)) == piecewise_fold(
        Piecewise((x + y, And(l <= i, i <= m)), (0, True)) +
        Piecewise((x + y, And(l <= j, j <= m)), (0, True)))


# 定义一个函数用于测试 delta_summation 的加法和乘法操作
def test_deltasummation_add_mul_x_y_mul_x_kd():
    # 断言：对 x*y + x*KD(i, j) 进行 delta_summation 操作
    assert ds(x*y + x*KD(i, j), (j, 1, 3)) == \
        # 返回结果分段函数
        Piecewise((3*x*y + x, And(1 <= i, i <= 3)), (3*x*y, True))
    assert ds(x*y + x*KD(i, j), (j, 1, 1)) == \
        Piecewise((x*y + x, Eq(i, 1)), (x*y, True))
    assert ds(x*y + x*KD(i, j), (j, 2, 2)) == \
        Piecewise((x*y + x, Eq(i, 2)), (x*y, True))
    # 对于表达式 ds(x*y + x*KD(i, j), (j, 3, 3)) 进行断言，验证其求导结果是否等于给定的分段函数
    assert ds(x*y + x*KD(i, j), (j, 3, 3)) == \
        Piecewise((x*y + x, Eq(i, 3)), (x*y, True))
    
    # 对于表达式 ds(x*y + x*KD(i, j), (j, 1, k)) 进行断言，验证其求导结果是否等于给定的分段函数
    assert ds(x*y + x*KD(i, j), (j, 1, k)) == \
        Piecewise((k*x*y + x, And(1 <= i, i <= k)), (k*x*y, True))
    
    # 对于表达式 ds(x*y + x*KD(i, j), (j, k, 3)) 进行断言，验证其求导结果是否等于给定的分段函数
    assert ds(x*y + x*KD(i, j), (j, k, 3)) == \
        Piecewise(((4 - k)*x*y + x, And(k <= i, i <= 3)), ((4 - k)*x*y, True))
    
    # 对于表达式 ds(x*y + x*KD(i, j), (j, k, l)) 进行断言，验证其求导结果是否等于给定的分段函数
    assert ds(x*y + x*KD(i, j), (j, k, l)) == Piecewise(
        ((l - k + 1)*x*y + x, And(k <= i, i <= l)), ((l - k + 1)*x*y, True))
def test_deltasummation_mul_x_add_y_kd():
    assert ds(x*(y + KD(i, j)), (j, 1, 3)) == \
        Piecewise((3*x*y + x, And(1 <= i, i <= 3)), (3*x*y, True))
    assert ds(x*(y + KD(i, j)), (j, 1, 1)) == \
        Piecewise((x*y + x, Eq(i, 1)), (x*y, True))
    assert ds(x*(y + KD(i, j)), (j, 2, 2)) == \
        Piecewise((x*y + x, Eq(i, 2)), (x*y, True))
    assert ds(x*(y + KD(i, j)), (j, 3, 3)) == \
        Piecewise((x*y + x, Eq(i, 3)), (x*y, True))
    assert ds(x*(y + KD(i, j)), (j, 1, k)) == \
        Piecewise((k*x*y + x, And(1 <= i, i <= k)), (k*x*y, True))
    assert ds(x*(y + KD(i, j)), (j, k, 3)) == \
        Piecewise(((4 - k)*x*y + x, And(k <= i, i <= 3)), ((4 - k)*x*y, True))
    assert ds(x*(y + KD(i, j)), (j, k, l)) == Piecewise(
        ((l - k + 1)*x*y + x, And(k <= i, i <= l)), ((l - k + 1)*x*y, True))


注释：


# 测试 delta summation 函数在 x * (y + KD(i, j)) 形式下的计算结果
def test_deltasummation_mul_x_add_y_kd():
    # 第一个断言：计算 ds(x*(y + KD(i, j)), (j, 1, 3))，并验证其结果
    assert ds(x*(y + KD(i, j)), (j, 1, 3)) == \
        Piecewise((3*x*y + x, And(1 <= i, i <= 3)), (3*x*y, True))
    
    # 第二个断言：计算 ds(x*(y + KD(i, j)), (j, 1, 1))，并验证其结果
    assert ds(x*(y + KD(i, j)), (j, 1, 1)) == \
        Piecewise((x*y + x, Eq(i, 1)), (x*y, True))
    
    # 第三个断言：计算 ds(x*(y + KD(i, j)), (j, 2, 2))，并验证其结果
    assert ds(x*(y + KD(i, j)), (j, 2, 2)) == \
        Piecewise((x*y + x, Eq(i, 2)), (x*y, True))
    
    # 第四个断言：计算 ds(x*(y + KD(i, j)), (j, 3, 3))，并验证其结果
    assert ds(x*(y + KD(i, j)), (j, 3, 3)) == \
        Piecewise((x*y + x, Eq(i, 3)), (x*y, True))
    
    # 第五个断言：计算 ds(x*(y + KD(i, j)), (j, 1, k))，并验证其结果
    assert ds(x*(y + KD(i, j)), (j, 1, k)) == \
        Piecewise((k*x*y + x, And(1 <= i, i <= k)), (k*x*y, True))
    
    # 第六个断言：计算 ds(x*(y + KD(i, j)), (j, k, 3))，并验证其结果
    assert ds(x*(y + KD(i, j)), (j, k, 3)) == \
        Piecewise(((4 - k)*x*y + x, And(k <= i, i <= 3)), ((4 - k)*x*y, True))
    
    # 第七个断言：计算 ds(x*(y + KD(i, j)), (j, k, l))，并验证其结果
    assert ds(x*(y + KD(i, j)), (j, k, l)) == Piecewise(
        ((l - k + 1)*x*y + x, And(k <= i, i <= l)), ((l - k + 1)*x*y, True))



def test_deltasummation_mul_add_x_y_twokd():
    assert ds(x*(y + 2*KD(i, j)), (j, 1, 3)) == \
        Piecewise((3*x*y + 2*x, And(1 <= i, i <= 3)), (3*x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, 1, 1)) == \
        Piecewise((x*y + 2*x, Eq(i, 1)), (x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, 2, 2)) == \
        Piecewise((x*y + 2*x, Eq(i, 2)), (x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, 3, 3)) == \
        Piecewise((x*y + 2*x, Eq(i, 3)), (x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, 1, k)) == \
        Piecewise((k*x*y + 2*x, And(1 <= i, i <= k)), (k*x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, k, 3)) == Piecewise(
        ((4 - k)*x*y + 2*x, And(k <= i, i <= 3)), ((4 - k)*x*y, True))
    assert ds(x*(y + 2*KD(i, j)), (j, k, l)) == Piecewise(
        ((l - k + 1)*x*y + 2*x, And(k <= i, i <= l)), ((l - k + 1)*x*y, True))


注释：


# 测试 delta summation 函数在 x * (y + 2*KD(i, j)) 形式下的计算结果
def test_deltasummation_mul_add_x_y_twokd():
    # 第一个断言：计算 ds(x*(y + 2*KD(i, j)), (j, 1, 3))，并验证其结果
    assert ds(x*(y + 2*KD(i, j)), (j, 1, 3)) == \
        Piecewise((3*x*y + 2*x, And(1 <= i, i <= 3)), (3*x*y, True))
    
    # 第二个断言：计算 ds(x*(y + 2*KD(i, j)), (j, 1, 1))，并验证其结果
    assert ds(x*(y + 2*KD(i, j)), (j, 1, 1)) == \
        Piecewise((x*y + 2*x, Eq(i, 1)), (x*y, True))
    
    # 第三个断言：计算 ds(x*(y + 2*KD(i, j)), (j, 2, 2))，并验证其结果
    assert ds(x*(y + 2*KD(i, j)), (j, 2, 2)) == \
        Piecewise((x*y + 2*x, Eq(i, 2)), (x*y, True))
    
    # 第四个断言：计算 ds(x*(y + 2*KD(i, j)), (j, 3, 3))，并验证其结果
    assert ds(x*(y + 2*KD(i, j)), (j, 3, 3)) == \
        Piecewise((x*y + 2*x, Eq(i, 3)), (x*y, True))
    
    # 第五个断言：计算 ds(x*(y + 2*KD(i, j)), (j, 1, k))，并验证其结果
    assert ds(x*(y + 2*KD(i, j)), (j, 1, k)) == \
        Piecewise((k*x*y + 2*x, And(1 <= i, i <= k)), (k*x*y, True))
    
    # 第六个断言：计算 ds(x*(y + 2*KD(i, j)), (j, k, 3))，并验证其结果
    assert ds(x*(y + 2*KD(i, j)), (j, k, 3)) == Piecewise(
        ((4 - k)*x*y + 2*x, And(k <= i, i <= 3)), ((4 - k)*x*y, True))
    
    # 第七个断言：计算 ds(x*(y + 2*KD(i, j)), (j, k, l))，并验证其结果
    assert ds(x
    # 断言：对偏导数 ds((x + KD(i, k))*(y + KD(i, j)), (j, 1, 1)) 进行检查
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, 1, 1)) == piecewise_fold(
        # 使用分段函数将结果进行折叠
        Piecewise((KD(i, k) + x, Eq(i, 1)), (0, True)) +
        # 计算结果的第一部分
        (KD(i, k) + x)*y)

    # 断言：对偏导数 ds((x + KD(i, k))*(y + KD(i, j)), (j, 2, 2)) 进行检查
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, 2, 2)) == piecewise_fold(
        # 使用分段函数将结果进行折叠
        Piecewise((KD(i, k) + x, Eq(i, 2)), (0, True)) +
        # 计算结果的第一部分
        (KD(i, k) + x)*y)

    # 断言：对偏导数 ds((x + KD(i, k))*(y + KD(i, j)), (j, 3, 3)) 进行检查
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, 3, 3)) == piecewise_fold(
        # 使用分段函数将结果进行折叠
        Piecewise((KD(i, k) + x, Eq(i, 3)), (0, True)) +
        # 计算结果的第一部分
        (KD(i, k) + x)*y)

    # 断言：对偏导数 ds((x + KD(i, k))*(y + KD(i, j)), (j, 1, k)) 进行检查
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, 1, k)) == piecewise_fold(
        # 使用分段函数将结果进行折叠
        Piecewise((KD(i, k) + x, And(1 <= i, i <= k)), (0, True)) +
        # 计算结果的第一部分
        k*(KD(i, k) + x)*y)

    # 断言：对偏导数 ds((x + KD(i, k))*(y + KD(i, j)), (j, k, 3)) 进行检查
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, k, 3)) == piecewise_fold(
        # 使用分段函数将结果进行折叠
        Piecewise((KD(i, k) + x, And(k <= i, i <= 3)), (0, True)) +
        # 计算结果的第一部分
        (4 - k)*(KD(i, k) + x)*y)

    # 断言：对偏导数 ds((x + KD(i, k))*(y + KD(i, j)), (j, k, l)) 进行检查
    assert ds((x + KD(i, k))*(y + KD(i, j)), (j, k, l)) == piecewise_fold(
        # 使用分段函数将结果进行折叠
        Piecewise((KD(i, k) + x, And(k <= i, i <= l)), (0, True)) +
        # 计算结果的第一部分
        (l - k + 1)*(KD(i, k) + x)*y)
# 定义一个测试函数，用于测试 _extract_delta 函数对于特定输入的行为
def test_extract_delta():
    # 断言测试，验证调用 _extract_delta 函数时传入特定参数是否会引发 ValueError 异常
    raises(ValueError, lambda: _extract_delta(KD(i, j) + KD(k, l), i))
```