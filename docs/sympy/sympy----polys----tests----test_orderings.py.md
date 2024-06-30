# `D:\src\scipysrc\sympy\sympy\polys\tests\test_orderings.py`

```
"""Tests of monomial orderings. """

# 导入所需的模块和函数
from sympy.polys.orderings import (
    monomial_key, lex, grlex, grevlex, ilex, igrlex,
    LexOrder, InverseOrder, ProductOrder, build_product_order,
)

# 导入符号变量 x, y, z, t 和核心模块 S
from sympy.abc import x, y, z, t
from sympy.core import S
# 导入 pytest 的 raises 函数用于测试
from sympy.testing.pytest import raises

# 测试 lex 顺序的函数
def test_lex_order():
    # 断言 lex((1, 2, 3)) 返回值为 (1, 2, 3)
    assert lex((1, 2, 3)) == (1, 2, 3)
    # 断言 str(lex) 返回值为 'lex'
    assert str(lex) == 'lex'

    # 断言 lex((1, 2, 3)) 等于 lex((1, 2, 3))
    assert lex((1, 2, 3)) == lex((1, 2, 3))

    # 断言 lex((2, 2, 3)) 大于 lex((1, 2, 3))
    assert lex((2, 2, 3)) > lex((1, 2, 3))
    # 断言 lex((1, 3, 3)) 大于 lex((1, 2, 3))
    assert lex((1, 3, 3)) > lex((1, 2, 3))
    # 断言 lex((1, 2, 4)) 大于 lex((1, 2, 3))
    assert lex((1, 2, 4)) > lex((1, 2, 3))

    # 断言 lex((0, 2, 3)) 小于 lex((1, 2, 3))
    assert lex((0, 2, 3)) < lex((1, 2, 3))
    # 断言 lex((1, 1, 3)) 小于 lex((1, 2, 3))
    assert lex((1, 1, 3)) < lex((1, 2, 3))
    # 断言 lex((1, 2, 2)) 小于 lex((1, 2, 3))
    assert lex((1, 2, 2)) < lex((1, 2, 3))

    # 断言 lex.is_global 为 True
    assert lex.is_global is True
    # 断言 lex 等于 LexOrder() 函数的返回值
    assert lex == LexOrder()
    # 断言 lex 不等于 grlex
    assert lex != grlex

# 测试 grlex 顺序的函数
def test_grlex_order():
    # 断言 grlex((1, 2, 3)) 返回值为 (6, (1, 2, 3))
    assert grlex((1, 2, 3)) == (6, (1, 2, 3))
    # 断言 str(grlex) 返回值为 'grlex'
    assert str(grlex) == 'grlex'

    # 断言 grlex((1, 2, 3)) 等于 grlex((1, 2, 3))
    assert grlex((1, 2, 3)) == grlex((1, 2, 3))

    # 断言 grlex((2, 2, 3)) 大于 grlex((1, 2, 3))
    assert grlex((2, 2, 3)) > grlex((1, 2, 3))
    # 断言 grlex((1, 3, 3)) 大于 grlex((1, 2, 3))
    assert grlex((1, 3, 3)) > grlex((1, 2, 3))
    # 断言 grlex((1, 2, 4)) 大于 grlex((1, 2, 3))
    assert grlex((1, 2, 4)) > grlex((1, 2, 3))

    # 断言 grlex((0, 2, 3)) 小于 grlex((1, 2, 3))
    assert grlex((0, 2, 3)) < grlex((1, 2, 3))
    # 断言 grlex((1, 1, 3)) 小于 grlex((1, 2, 3))
    assert grlex((1, 1, 3)) < grlex((1, 2, 3))
    # 断言 grlex((1, 2, 2)) 小于 grlex((1, 2, 3))
    assert grlex((1, 2, 2)) < grlex((1, 2, 3))

    # 进一步的比较测试
    assert grlex((2, 2, 3)) > grlex((1, 2, 4))
    assert grlex((1, 3, 3)) > grlex((1, 2, 4))
    assert grlex((0, 2, 3)) < grlex((1, 2, 2))
    assert grlex((1, 1, 3)) < grlex((1, 2, 2))
    assert grlex((0, 1, 1)) > grlex((0, 0, 2))
    assert grlex((0, 3, 1)) < grlex((2, 2, 1))

    # 断言 grlex.is_global 为 True
    assert grlex.is_global is True

# 测试 grevlex 顺序的函数
def test_grevlex_order():
    # 断言 grevlex((1, 2, 3)) 返回值为 (6, (-3, -2, -1))
    assert grevlex((1, 2, 3)) == (6, (-3, -2, -1))
    # 断言 str(grevlex) 返回值为 'grevlex'
    assert str(grevlex) == 'grevlex'

    # 断言 grevlex((1, 2, 3)) 等于 grevlex((1, 2, 3))
    assert grevlex((1, 2, 3)) == grevlex((1, 2, 3))

    # 断言 grevlex((2, 2, 3)) 大于 grevlex((1, 2, 3))
    assert grevlex((2, 2, 3)) > grevlex((1, 2, 3))
    # 断言 grevlex((1, 3, 3)) 大于 grevlex((1, 2, 3))
    assert grevlex((1, 3, 3)) > grevlex((1, 2, 3))
    # 断言 grevlex((1, 2, 4)) 大于 grevlex((1, 2, 3))
    assert grevlex((1, 2, 4)) > grevlex((1, 2, 3))

    # 断言 grevlex((0, 2, 3)) 小于 grevlex((1, 2, 3))
    assert grevlex((0, 2, 3)) < grevlex((1, 2, 3))
    # 断言 grevlex((1, 1, 3)) 小于 grevlex((1, 2, 3))
    assert grevlex((1, 1, 3)) < grevlex((1, 2, 3))
    # 断言 grevlex((1, 2, 2)) 小于 grevlex((1, 2, 3))
    assert grevlex((1, 2, 2)) < grevlex((1, 2, 3))

    # 进一步的比较测试
    assert grevlex((2, 2, 3)) > grevlex((1, 2, 4))
    assert grevlex((1, 3, 3)) > grevlex((1, 2, 4))
    assert grevlex((0, 2, 3)) < grevlex((1, 2, 2))
    assert grevlex((1, 1, 3)) < grevlex((1, 2, 2))
    assert grevlex((0, 1, 1)) > grevlex((0, 0, 2))
    assert grevlex((0, 3, 1)) < grevlex((2, 2, 1))

    # 断言 grevlex.is_global 为 True
    assert grevlex.is_global is True

# 测试 In
    # 断言：使用 ProductOrder 类对 (grlex, None) 和 (ilex, None) 进行初始化，并检查其 is_global 属性是否为 None
    assert ProductOrder((grlex, None), (ilex, None)).is_global is None
    
    # 断言：使用 ProductOrder 类对 (igrlex, None) 和 (ilex, None) 进行初始化，并检查其 is_global 属性是否为 False
    assert ProductOrder((igrlex, None), (ilex, None)).is_global is False
# 定义函数 test_monomial_key，用于测试 monomial_key 函数的各种用法和参数组合
def test_monomial_key():
    # 断言默认调用 monomial_key 函数返回 lex 函数对象
    assert monomial_key() == lex

    # 测试传入参数 'lex'，期望返回 lex 函数对象
    assert monomial_key('lex') == lex
    # 测试传入参数 'grlex'，期望返回 grlex 函数对象
    assert monomial_key('grlex') == grlex
    # 测试传入参数 'grevlex'，期望返回 grevlex 函数对象
    assert monomial_key('grevlex') == grevlex

    # 测试传入无效参数 'foo'，期望引发 ValueError 异常
    raises(ValueError, lambda: monomial_key('foo'))
    # 测试传入非字符串参数 1，期望引发 ValueError 异常
    raises(ValueError, lambda: monomial_key(1))

    # 创建测试用的多项式列表 M
    M = [x, x**2*z**2, x*y, x**2, S.One, y**2, x**3, y, z, x*y**2*z, x**2*y**2]
    # 使用 'lex' 排序方式和指定变量顺序 [z, y, x] 对 M 进行排序，验证结果是否正确
    assert sorted(M, key=monomial_key('lex', [z, y, x])) == \
        [S.One, x, x**2, x**3, y, x*y, y**2, x**2*y**2, z, x*y**2*z, x**2*z**2]
    # 使用 'grlex' 排序方式和指定变量顺序 [z, y, x] 对 M 进行排序，验证结果是否正确
    assert sorted(M, key=monomial_key('grlex', [z, y, x])) == \
        [S.One, x, y, z, x**2, x*y, y**2, x**3, x**2*y**2, x*y**2*z, x**2*z**2]
    # 使用 'grevlex' 排序方式和指定变量顺序 [z, y, x] 对 M 进行排序，验证结果是否正确
    assert sorted(M, key=monomial_key('grevlex', [z, y, x])) == \
        [S.One, x, y, z, x**2, x*y, y**2, x**3, x**2*y**2, x**2*z**2, x*y**2*z]

# 定义函数 test_build_product_order，用于测试 build_product_order 函数的用法和参数组合
def test_build_product_order():
    # 使用两个排序元组 (("grlex", x, y), ("grlex", z, t)) 和变量顺序 [x, y, z, t]，构建排序函数并应用于元组 (4, 5, 6, 7)，验证结果是否正确
    assert build_product_order((("grlex", x, y), ("grlex", z, t)), [x, y, z, t])((4, 5, 6, 7)) == \
        ((9, (4, 5)), (13, (6, 7)))

    # 断言两次调用相同参数的 build_product_order 返回相同的对象
    assert build_product_order((("grlex", x, y), ("grlex", z, t)), [x, y, z, t]) == \
        build_product_order((("grlex", x, y), ("grlex", z, t)), [x, y, z, t])
```