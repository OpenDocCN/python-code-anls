# `D:\src\scipysrc\scipy\scipy\interpolate\tests\test_pade.py`

```
# 从 numpy.testing 模块导入 assert_array_equal 和 assert_array_almost_equal 函数
from numpy.testing import (assert_array_equal, assert_array_almost_equal)
# 从 scipy.interpolate 模块导入 pade 函数
from scipy.interpolate import pade

# 定义测试函数 test_pade_trivial，用于测试 pade 函数在简单情况下的行为
def test_pade_trivial():
    # 对于输入 [1.0]，计算 Pade 近似的分子和分母多项式
    nump, denomp = pade([1.0], 0)
    # 断言分子多项式的系数与预期值相等
    assert_array_equal(nump.c, [1.0])
    # 断言分母多项式的系数与预期值相等
    assert_array_equal(denomp.c, [1.0])

    # 重复上述步骤，测试另一种输入方式
    nump, denomp = pade([1.0], 0, 0)
    assert_array_equal(nump.c, [1.0])
    assert_array_equal(denomp.c, [1.0])


# 定义测试函数 test_pade_4term_exp，用于测试 pade 函数在给定的 Taylor 级数下的行为
def test_pade_4term_exp():
    # 定义 exp(x) 的前四个 Taylor 级数系数
    an = [1.0, 1.0, 0.5, 1.0/6]

    # 对于给定的 Taylor 级数系数和阶数，计算 Pade 近似的分子和分母多项式
    nump, denomp = pade(an, 0)
    # 断言分子多项式的系数与预期值近似相等
    assert_array_almost_equal(nump.c, [1.0/6, 0.5, 1.0, 1.0])
    # 断言分母多项式的系数与预期值近似相等
    assert_array_almost_equal(denomp.c, [1.0])

    nump, denomp = pade(an, 1)
    assert_array_almost_equal(nump.c, [1.0/6, 2.0/3, 1.0])
    assert_array_almost_equal(denomp.c, [-1.0/3, 1.0])

    nump, denomp = pade(an, 2)
    assert_array_almost_equal(nump.c, [1.0/3, 1.0])
    assert_array_almost_equal(denomp.c, [1.0/6, -2.0/3, 1.0])

    nump, denomp = pade(an, 3)
    assert_array_almost_equal(nump.c, [1.0])
    assert_array_almost_equal(denomp.c, [-1.0/6, 0.5, -1.0, 1.0])

    # 测试包含可选参数的情况
    nump, denomp = pade(an, 0, 3)
    assert_array_almost_equal(nump.c, [1.0/6, 0.5, 1.0, 1.0])
    assert_array_almost_equal(denomp.c, [1.0])

    nump, denomp = pade(an, 1, 2)
    assert_array_almost_equal(nump.c, [1.0/6, 2.0/3, 1.0])
    assert_array_almost_equal(denomp.c, [-1.0/3, 1.0])

    nump, denomp = pade(an, 2, 1)
    assert_array_almost_equal(nump.c, [1.0/3, 1.0])
    assert_array_almost_equal(denomp.c, [1.0/6, -2.0/3, 1.0])

    nump, denomp = pade(an, 3, 0)
    assert_array_almost_equal(nump.c, [1.0])
    assert_array_almost_equal(denomp.c, [-1.0/6, 0.5, -1.0, 1.0])

    # 测试数组的减少情况
    nump, denomp = pade(an, 0, 2)
    assert_array_almost_equal(nump.c, [0.5, 1.0, 1.0])
    assert_array_almost_equal(denomp.c, [1.0])

    nump, denomp = pade(an, 1, 1)
    assert_array_almost_equal(nump.c, [1.0/2, 1.0])
    assert_array_almost_equal(denomp.c, [-1.0/2, 1.0])

    nump, denomp = pade(an, 2, 0)
    assert_array_almost_equal(nump.c, [1.0])
    assert_array_almost_equal(denomp.c, [1.0/2, -1.0, 1.0])


# 定义测试函数 test_pade_ints，用于测试 pade 函数在整数和浮点数序列下的一致性
def test_pade_ints():
    # 简单的测试序列（一个整数序列，一个浮点数序列）
    an_int = [1, 2, 3, 4]
    an_flt = [1.0, 2.0, 3.0, 4.0]

    # 确保整数序列和相同值的浮点数序列给出相同的结果
    for i in range(0, len(an_int)):
        for j in range(0, len(an_int) - i):

            # 对给定阶数的整数和浮点数序列分别计算 Pade 近似的分子和分母多项式
            nump_int, denomp_int = pade(an_int, i, j)
            nump_flt, denomp_flt = pade(an_flt, i, j)

            # 断言它们的分子多项式和分母多项式系数相同
            assert_array_equal(nump_int.c, nump_flt.c)
            assert_array_equal(denomp_int.c, denomp_flt.c)


# 定义测试函数 test_pade_complex，用于测试 pade 函数在复杂情况下的行为
def test_pade_complex():
    # 测试具有已知解的序列（参考文献见 IEEE 论文）
    # 这里暂时保留空白，后续可能需要添加具体的测试内容
    pass
    # 定义复数变量 x，这些测试将适用于任何复数。
    x = 0.2 + 0.6j
    
    # 计算给定复数 x 对应的一组系数
    an = [
        1.0,
        x,
        -x * x.conjugate(),
        x.conjugate() * (x**2) + x * (x.conjugate()**2),
        -(x**3) * x.conjugate() - 3 * (x * x.conjugate())**2 - x * (x.conjugate()**3)
    ]
    
    # 调用 pade 函数计算 Pade 近似的分子和分母多项式，阶数分别为 1 和 1
    nump, denomp = pade(an, 1, 1)
    
    # 断言分子多项式的系数近似等于指定值
    assert_array_almost_equal(nump.c, [x + x.conjugate(), 1.0])
    
    # 断言分母多项式的系数近似等于指定值
    assert_array_almost_equal(denomp.c, [x.conjugate(), 1.0])
    
    # 再次调用 pade 函数计算 Pade 近似的分子和分母多项式，阶数分别为 1 和 2
    nump, denomp = pade(an, 1, 2)
    
    # 断言分子多项式的系数近似等于指定值
    assert_array_almost_equal(nump.c, [x**2, 2*x + x.conjugate(), 1.0])
    
    # 断言分母多项式的系数近似等于指定值
    assert_array_almost_equal(denomp.c, [x + x.conjugate(), 1.0])
    
    # 再次调用 pade 函数计算 Pade 近似的分子和分母多项式，阶数分别为 2 和 2
    nump, denomp = pade(an, 2, 2)
    
    # 断言分子多项式的系数近似等于指定值
    assert_array_almost_equal(
        nump.c,
        [x**2 + x*x.conjugate() + x.conjugate()**2, 2*(x + x.conjugate()), 1.0]
    )
    
    # 断言分母多项式的系数近似等于指定值
    assert_array_almost_equal(denomp.c, [x.conjugate()**2, x + 2*x.conjugate(), 1.0])
```