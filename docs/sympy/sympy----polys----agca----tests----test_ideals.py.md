# `D:\src\scipysrc\sympy\sympy\polys\agca\tests\test_ideals.py`

```
"""Test ideals.py code."""

# 从 sympy.polys 导入 QQ 和 ilex
from sympy.polys import QQ, ilex
# 从 sympy.abc 导入 x, y, z
from sympy.abc import x, y, z
# 从 sympy.testing.pytest 导入 raises 函数
from sympy.testing.pytest import raises


# 定义测试理想运算的函数
def test_ideal_operations():
    # 创建有理数多项式环 QQ.old_poly_ring，包含变量 x 和 y
    R = QQ.old_poly_ring(x, y)
    # 创建理想 I，包含变量 x
    I = R.ideal(x)
    # 创建理想 J，包含变量 y
    J = R.ideal(y)
    # 创建理想 S，包含变量 x*y
    S = R.ideal(x*y)
    # 创建理想 T，包含变量 x 和 y
    T = R.ideal(x, y)

    # 断言 I 和 J 不相等
    assert not (I == J)
    # 断言 I 等于 I 自身
    assert I == I

    # 断言 I 和 J 的并集等于 T
    assert I.union(J) == T
    # 断言 I 和 J 的加法操作等于 T
    assert I + J == T
    # 断言 I 和 T 的加法操作等于 T
    assert I + T == T

    # 断言 I 不是 T 的子集
    assert not I.subset(T)
    # 断言 T 是 I 的子集
    assert T.subset(I)

    # 断言 I 和 J 的乘法操作等于 S
    assert I.product(J) == S
    # 断言 I 乘以 J 等于 S
    assert I*J == S
    # 断言 x 乘以 J 等于 S
    assert x*J == S
    # 断言 I 乘以 y 等于 S
    assert I*y == S
    # 断言 R.convert(x) 乘以 J 等于 S
    assert R.convert(x)*J == S
    # 断言 I 乘以 R.convert(y) 等于 S
    assert I*R.convert(y) == S

    # 断言 I 不是零理想
    assert not I.is_zero()
    # 断言 J 不是整环
    assert not J.is_whole_ring()

    # 断言理想 R.ideal(x**2 + 1, x) 是整环
    assert R.ideal(x**2 + 1, x).is_whole_ring()
    # 断言空理想等于 R.ideal(0)
    assert R.ideal() == R.ideal(0)
    # 断言空理想是零理想
    assert R.ideal().is_zero()

    # 断言 T 包含 x*y
    assert T.contains(x*y)
    # 断言 T 是由 [x, y] 生成的子集
    assert T.subset([x, y])

    # 断言 T 用 x 表示的生成元等于 [R(1), R(0)]
    assert T.in_terms_of_generators(x) == [R(1), R(0)]

    # 断言 T 的 0 次幂等于 R.ideal(1)
    assert T**0 == R.ideal(1)
    # 断言 T 的 1 次幂等于 T
    assert T**1 == T
    # 断言 T 的 2 次幂等于 R.ideal(x**2, y**2, x*y)
    assert T**2 == R.ideal(x**2, y**2, x*y)
    # 断言 I 的 5 次幂等于 R.ideal(x**5)
    assert I**5 == R.ideal(x**5)


# 定义异常测试函数
def test_exceptions():
    # 创建 QQ.old_poly_ring(x) 中的理想 I，包含变量 x
    I = QQ.old_poly_ring(x).ideal(x)
    # 创建 QQ.old_poly_ring(y) 中的理想 J，包含常数 1
    J = QQ.old_poly_ring(y).ideal(1)
    # 断言调用 I.union(x) 会抛出 ValueError 异常
    raises(ValueError, lambda: I.union(x))
    # 断言调用 I + J 会抛出 ValueError 异常
    raises(ValueError, lambda: I + J)
    # 断言调用 I * J 会抛出 ValueError 异常
    raises(ValueError, lambda: I * J)
    # 断言调用 I.union(J) 会抛出 ValueError 异常
    raises(ValueError, lambda: I.union(J))
    # 断言 I 不等于 J
    assert (I == J) is False
    # 断言 I 不等于 J
    assert I != J


# 定义测试包含全局理想的函数
def test_nontriv_global():
    # 创建有理数多项式环 QQ.old_poly_ring，包含变量 x, y, z
    R = QQ.old_poly_ring(x, y, z)

    # 定义包含函数 contains
    def contains(I, f):
        # 返回理想 R.ideal(*I) 是否包含 f
        return R.ideal(*I).contains(f)

    # 断言包含函数 contains([x, y], x) 返回 True
    assert contains([x, y], x)
    # 断言包含函数 contains([x, y], x + y) 返回 True
    assert contains([x, y], x + y)
    # 断言包含函数 contains([x, y], 1) 返回 False
    assert not contains([x, y], 1)
    # 断言包含函数 contains([x, y], z) 返回 False
    assert not contains([x, y], z)
    # 断言包含函数 contains([x**2 + y, x**2 + x], x - y) 返回 True
    assert contains([x**2 + y, x**2 + x], x - y)
    # 断言包含函数 contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2) 返回 False
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    # 断言包含函数 contains([x*(1 + x + y), y*(1 + z)], x) 返回 True
    assert contains([x*(1 + x + y), y*(1 + z)], x)
    # 断言包含函数 contains([x*(1 + x + y), y*(1 + z)], x + y) 返回 True
    assert contains([x*(1 + x + y), y*(1 + z)], x + y)


# 定义测试包含局部顺序的函数
def test_nontriv_local():
    # 创建有理数多项式环 QQ.old_poly_ring，包含变量 x, y, z，并指定顺序为 ilex
    R = QQ.old_poly_ring(x, y, z, order=ilex)

    # 定义包含函数 contains
    def contains(I, f):
        # 返回理想 R.ideal(*I) 是否包含 f
        return R.ideal(*I).contains(f)

    # 断言包含函数 contains([x, y], x) 返回 True
    assert contains([x, y], x)
    # 断言包含函数 contains([x, y], x + y) 返回 True
    assert contains([x, y], x + y)
    # 断言包含函数 contains([x, y], 1) 返回 False
    assert not contains([x, y], 1)
    # 断言包含函数 contains([x, y], z) 返回 False
    assert not contains([x, y], z)
    # 断言包含函数 contains([x**2 + y, x**2 + x], x - y) 返回 True
    assert contains([x**2 + y, x**2 + x], x - y)
    # 断言包含函数 contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2) 返回 False
    assert not contains([x + y + z, x*y + x*z + y*z, x*y*z], x**2)
    # 断言包含函数 contains([x*(1 + x + y), y*(1 + z)], x) 返回 True
    assert contains([x*(1 + x + y), y*(1 + z)], x)
    # 断言包含函数 contains([x*(1 + x + y), y*(1 + z)], x + y) 返回 True
    # 使用 QQ 对象创建一个多项式环 R，包括变量 x, y, z
    R = QQ.old_poly_ring(x, y, z)
    
    # 断言：验证 R 中由两个理想的交集是否等于给定的理想
    assert R.ideal(x, y).intersect(R.ideal(y**2, z)) == R.ideal(y**2, y*z, x*z)
    
    # 断言：验证 R 中由一个理想和空理想的交集是否为零理想
    assert R.ideal(x, y).intersect(R.ideal()).is_zero()
    
    # 使用 QQ 对象创建一个多项式环 R，包括变量 x, y, z，并指定变量排序顺序为 "ilex"
    R = QQ.old_poly_ring(x, y, z, order="ilex")
    
    # 断言：验证 R 中由两个复杂的理想的交集是否等于给定的理想
    assert R.ideal(x, y).intersect(R.ideal(y**2 + y**2*z, z + z*x**3*y)) == \
        R.ideal(y**2, y*z, x*z)
# 定义一个测试函数，用于测试多项式环的理想与元素的约化操作
def test_quotient():
    # 导入多项式环 QQ 中的旧版本多项式环函数
    R = QQ.old_poly_ring(x, y, z)
    # 断言：对于 R 中的理想 (x, y) 模去理想 (y**2, z) 的商等于理想 (x, y)
    assert R.ideal(x, y).quotient(R.ideal(y**2, z)) == R.ideal(x, y)


def test_reduction():
    # 从 sympy.polys.distributedmodules 模块中导入 sdm_nf_buchberger_reduced 函数
    from sympy.polys.distributedmodules import sdm_nf_buchberger_reduced
    # 在 QQ 中创建一个旧版本多项式环 R，包含变量 x 和 y
    R = QQ.old_poly_ring(x, y)
    # 创建理想 I，包含元素 x**5 和 y
    I = R.ideal(x**5, y)
    # 创建元素 e，将 x**3 + y**2 转换为 R 中的元素
    e = R.convert(x**3 + y**2)
    # 断言：理想 I 对元素 e 的约化结果等于 e 自身
    assert I.reduce_element(e) == e
    # 断言：使用 sdm_nf_buchberger_reduced 算法约化理想 I 对元素 e 的结果等于 R 中 x**3 的转换
    assert I.reduce_element(e, NF=sdm_nf_buchberger_reduced) == R.convert(x**3)
```