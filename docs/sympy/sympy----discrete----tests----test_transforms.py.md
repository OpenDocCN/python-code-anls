# `D:\src\scipysrc\sympy\sympy\discrete\tests\test_transforms.py`

```
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, Symbol, symbols, I, Rational
from sympy.discrete import (fft, ifft, ntt, intt, fwht, ifwht,
    mobius_transform, inverse_mobius_transform)
from sympy.testing.pytest import raises

# 定义一个测试函数，用于测试 FFT 和 IFFT 算法的功能
def test_fft_ifft():
    # 断言空列表和包含一个有理数的列表经过 fft 和 ifft 处理后结果不变
    assert all(tf(ls) == ls for tf in (fft, ifft)
                            for ls in ([], [Rational(5, 3)]))

    # 定义一个列表 ls
    ls = list(range(6))
    # 预期的 fft 结果 fls
    fls = [15, -7*sqrt(2)/2 - 4 - sqrt(2)*I/2 + 2*I, 2 + 3*I,
             -4 + 7*sqrt(2)/2 - 2*I - sqrt(2)*I/2, -3,
             -4 + 7*sqrt(2)/2 + sqrt(2)*I/2 + 2*I,
              2 - 3*I, -7*sqrt(2)/2 - 4 - 2*I + sqrt(2)*I/2]

    # 断言 fft(ls) 等于 fls
    assert fft(ls) == fls
    # 断言 ifft(fls) 等于 ls 加上两个零元素
    assert ifft(fls) == ls + [S.Zero]*2

    # 定义一个复数列表 ls
    ls = [1 + 2*I, 3 + 4*I, 5 + 6*I]
    # 预期的 ifft 结果 ifls
    ifls = [Rational(9, 4) + 3*I, I*Rational(-7, 4), Rational(3, 4) + I, -2 - I/4]

    # 断言 ifft(ls) 等于 ifls
    assert ifft(ls) == ifls
    # 断言 fft(ifls) 等于 ls 加上一个零元素
    assert fft(ifls) == ls + [S.Zero]

    # 定义一个实数符号 x
    x = Symbol('x', real=True)
    # 断言 fft(x) 引发 TypeError 异常
    raises(TypeError, lambda: fft(x))
    # 断言 ifft([x, 2*x, 3*x**2, 4*x**3]) 引发 ValueError 异常
    raises(ValueError, lambda: ifft([x, 2*x, 3*x**2, 4*x**3]))


# 定义一个测试函数，用于测试 NTT 和 INTT 算法的功能
def test_ntt_intt():
    # 定义一个素数形式的模数 p 和长度为 1 或 2 的序列 q
    p = 7*17*2**23 + 1
    q = 2*500000003 + 1  # 只适用于长度为 1 或 2 的序列
    r = 2*3*5*7  # 复合模数

    # 断言空列表和包含一个元素 5 的列表经过 ntt 和 intt 处理后结果不变
    assert all(tf(ls, p) == ls for tf in (ntt, intt)
                                for ls in ([], [5]))

    # 定义一个列表 ls
    ls = list(range(6))
    # 预期的 ntt 结果 nls
    nls = [15, 801133602, 738493201, 334102277, 998244350, 849020224,
            259751156, 12232587]

    # 断言 ntt(ls, p) 等于 nls
    assert ntt(ls, p) == nls
    # 断言 intt(nls, p) 等于 ls 加上两个零元素
    assert intt(nls, p) == ls + [0]*2

    # 定义一个复数列表 ls
    ls = [1 + 2*I, 3 + 4*I, 5 + 6*I]
    # 定义一个整数类型的符号 x
    x = Symbol('x', integer=True)

    # 断言 ntt(x, p) 引发 TypeError 异常
    raises(TypeError, lambda: ntt(x, p))
    # 断言 intt([x, 2*x, 3*x**2, 4*x**3], p) 引发 ValueError 异常
    raises(ValueError, lambda: intt([x, 2*x, 3*x**2, 4*x**3], p))
    # 断言 intt(ls, p) 引发 ValueError 异常
    raises(ValueError, lambda: intt(ls, p))
    # 断言 ntt([1.2, 2.1, 3.5], p) 引发 ValueError 异常
    raises(ValueError, lambda: ntt([1.2, 2.1, 3.5], p))
    # 断言 ntt([3, 5, 6], q) 引发 ValueError 异常
    raises(ValueError, lambda: ntt([3, 5, 6], q))
    # 断言 ntt([4, 5, 7], r) 引发 ValueError 异常
    raises(ValueError, lambda: ntt([4, 5, 7], r))
    # 断言 ntt([1.0, 2.0, 3.0], p) 引发 ValueError 异常
    raises(ValueError, lambda: ntt([1.0, 2.0, 3.0], p))


# 定义一个测试函数，用于测试 FWHT 和 IFWHT 算法的功能
def test_fwht_ifwht():
    # 断言空列表和包含一个有理数的列表经过 fwht 和 ifwht 处理后结果不变
    assert all(tf(ls) == ls for tf in (fwht, ifwht) \
                        for ls in ([], [Rational(7, 4)]))

    # 定义一个列表 ls
    ls = [213, 321, 43235, 5325, 312, 53]
    # 预期的 fwht 结果 fls
    fls = [49459, 38061, -47661, -37759, 48729, 37543, -48391, -38277]

    # 断言 fwht(ls) 等于 fls
    assert fwht(ls) == fls
    # 断言 ifwht(fls) 等于 ls 加上两个零元素
    assert ifwht(fls) == ls + [S.Zero]*2

    # 定义一个复数列表 ls
    ls = [S.Half + 2*I, Rational(3, 7) + 4*I, Rational(5, 6) + 6*I, Rational(7, 3), Rational(9, 4)]
    # 预期的 ifwht 结果 ifls
    ifls = [Rational(533, 672) + I*3/2, Rational(23, 224) + I/2, Rational(1, 672), Rational(107, 224) - I,
        Rational(155, 672) + I*3/2, Rational(-103, 224) + I/2, Rational(-377, 672), Rational(-19, 224) - I]

    # 断言 ifwht(ls) 等于 ifls
    assert ifwht(ls) == ifls
    # 断言 fwht(ifls) 等于 ls 加上三个零元素
    assert fwht(ifls) == ls + [S.Zero]*3

    # 定义符号 x 和 y
    x, y = symbols('x y')

    # 断言 fwht(x) 引发 TypeError 异常
    raises(TypeError, lambda: fwht(x))

    # 定义一个列表 ls
    ls = [x, 2*x, 3*x**2, 4*x**3]
    # 预期的 ifls
    ifls = [x**3 + 3*x**2/4 + x*Rational(3, 4),
        -x**3 + 3*x**2/4 - x/4,
        -x**3 - 3*x**2/4 + x*Rational(3, 4),
        x**3 - 3*x**2/4 - x/4]

    # 断言 ifwht(ls) 等于 ifls
    assert ifwht(ls
    # 断言检查 ifwht 函数对 ls 的输出是否等于 ifls
    assert ifwht(ls) == ifls
    
    # 断言检查 fwht 函数对 ifls 的输出是否等于 ls
    assert fwht(ifls) == ls
    
    # 创建包含 x, y, x**2, y**2, x*y 的列表 ls
    ls = [x, y, x**2, y**2, x*y]
    
    # 创建包含多个表达式的列表 fls
    fls = [
        x**2 + x*y + x + y**2 + y,
        x**2 + x*y + x - y**2 - y,
        -x**2 + x*y + x - y**2 + y,
        -x**2 + x*y + x + y**2 - y,
        x**2 - x*y + x + y**2 + y,
        x**2 - x*y + x - y**2 - y,
        -x**2 - x*y + x - y**2 + y,
        -x**2 - x*y + x + y**2 - y
    ]
    
    # 断言检查 fwht 函数对 ls 的输出是否等于 fls
    assert fwht(ls) == fls
    
    # 断言检查 ifwht 函数对 fls 的输出是否等于 ls 加上三个 S.Zero 元素的列表
    assert ifwht(fls) == ls + [S.Zero]*3
    
    # 将 ls 设置为包含数字 0 到 5 的列表
    ls = list(range(6))
    
    # 断言检查 fwht 函数对 ls 的输出是否等于 ifwht 函数对 ls 的输出中每个元素乘以 8 的结果
    assert fwht(ls) == [x*8 for x in ifwht(ls)]
# 定义一个用于测试 Möbius 变换函数的函数
def test_mobius_transform():
    # 断言：对于空列表和包含单个有理数的列表，分别测试是否所有的 Möbius 变换和逆变换函数都返回原始列表
    assert all(tf(ls, subset=subset) == ls
                for ls in ([], [Rational(7, 4)]) for subset in (True, False)
                for tf in (mobius_transform, inverse_mobius_transform))

    # 创建符号变量 w, x, y, z
    w, x, y, z = symbols('w x y z')

    # 断言：对于 [x, y] 输入，Möbius 变换应返回 [x, x + y]
    assert mobius_transform([x, y]) == [x, x + y]
    # 断言：对于 [x, x + y] 输入，逆 Möbius 变换应返回 [x, y]
    assert inverse_mobius_transform([x, x + y]) == [x, y]
    # 断言：对于 [x, y] 输入，设置 subset=False，Möbius 变换应返回 [x + y, y]
    assert mobius_transform([x, y], subset=False) == [x + y, y]
    # 断言：对于 [x + y, y] 输入，设置 subset=False，逆 Möbius 变换应返回 [x, y]
    assert inverse_mobius_transform([x + y, y], subset=False) == [x, y]

    # 断言：对于 [w, x, y, z] 输入，Möbius 变换应返回 [w, w + x, w + y, w + x + y + z]
    assert mobius_transform([w, x, y, z]) == [w, w + x, w + y, w + x + y + z]
    # 断言：对于 [w, w + x, w + y, w + x + y + z] 输入，逆 Möbius 变换应返回 [w, x, y, z]
    assert inverse_mobius_transform([w, w + x, w + y, w + x + y + z]) == \
            [w, x, y, z]
    # 断言：对于 [w, x, y, z] 输入，设置 subset=False，Möbius 变换应返回 [w + x + y + z, x + z, y + z, z]
    assert mobius_transform([w, x, y, z], subset=False) == \
            [w + x + y + z, x + z, y + z, z]
    # 断言：对于 [w + x + y + z, x + z, y + z, z] 输入，设置 subset=False，逆 Möbius 变换应返回 [w, x, y, z]
    assert inverse_mobius_transform([w + x + y + z, x + z, y + z, z], subset=False) == \
            [w, x, y, z]

    # 创建一个包含有理数和复数的列表 ls
    ls = [Rational(2, 3), Rational(6, 7), Rational(5, 8), 9, Rational(5, 3) + 7*I]
    # 创建经 Möbius 变换后期望的列表 mls
    mls = [Rational(2, 3), Rational(32, 21), Rational(31, 24), Rational(1873, 168),
            Rational(7, 3) + 7*I, Rational(67, 21) + 7*I, Rational(71, 24) + 7*I,
            Rational(2153, 168) + 7*I]

    # 断言：对于列表 ls，应用 Möbius 变换后应返回 mls
    assert mobius_transform(ls) == mls
    # 断言：对于列表 mls，应用逆 Möbius 变换后应返回原始列表 ls 加上三个零
    assert inverse_mobius_transform(mls) == ls + [S.Zero]*3

    # 更新 mls，添加额外的数据
    mls = [Rational(2153, 168) + 7*I, Rational(69, 7), Rational(77, 8), 9, Rational(5, 3) + 7*I, 0, 0, 0]

    # 断言：对于列表 ls，设置 subset=False，应用 Möbius 变换后应返回更新后的 mls
    assert mobius_transform(ls, subset=False) == mls
    # 断言：对于更新后的 mls，设置 subset=False，应用逆 Möbius 变换后应返回原始列表 ls 加上三个零
    assert inverse_mobius_transform(mls, subset=False) == ls + [S.Zero]*3

    # 缩短 ls 列表，移除最后一个元素
    ls = ls[:-1]
    # 更新 mls，移除最后一个元素
    mls = [Rational(2, 3), Rational(32, 21), Rational(31, 24), Rational(1873, 168)]

    # 断言：对于缩短后的 ls，应用 Möbius 变换后应返回更新后的 mls
    assert mobius_transform(ls) == mls
    # 断言：对于更新后的 mls，应用逆 Möbius 变换后应返回原始列表 ls
    assert inverse_mobius_transform(mls) == ls

    # 更新 mls，移除复数部分
    mls = [Rational(1873, 168), Rational(69, 7), Rational(77, 8), 9]

    # 断言：对于缩短后的 ls，设置 subset=False，应用 Möbius 变换后应返回更新后的 mls
    assert mobius_transform(ls, subset=False) == mls
    # 断言：对于更新后的 mls，设置 subset=False，应用逆 Möbius 变换后应返回原始列表 ls
    assert inverse_mobius_transform(mls, subset=False) == ls

    # 断言：对于单个符号变量 x，应用 Möbius 变换，会引发 TypeError 异常
    raises(TypeError, lambda: mobius_transform(x, subset=True))
    # 断言：对于单个符号变量 y，设置 subset=False，应用逆 Möbius 变换，会引发 TypeError 异常
    raises(TypeError, lambda: inverse_mobius_transform(y, subset=False))
```