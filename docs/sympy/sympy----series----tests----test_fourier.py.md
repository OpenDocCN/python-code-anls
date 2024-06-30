# `D:\src\scipysrc\sympy\sympy\series\tests\test_fourier.py`

```
from sympy.core.add import Add
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin, sinc, tan)
from sympy.series.fourier import fourier_series
from sympy.series.fourier import FourierSeries
from sympy.testing.pytest import raises
from functools import lru_cache

# 定义符号变量 x, y, z
x, y, z = symbols('x y z')

# 使用 functools 模块中的 lru_cache 装饰器，缓存 _get_examples 函数的结果
@lru_cache()
def _get_examples():
    # 计算 x 在区间 (-pi, pi) 上的 Fourier 级数
    fo = fourier_series(x, (x, -pi, pi))
    # 计算 x^2 在区间 (-pi, pi) 上的 Fourier 级数
    fe = fourier_series(x**2, (-pi, pi))
    # 计算分段函数 Piecewise((0, x < 0), (pi, True)) 在区间 (-pi, pi) 上的 Fourier 级数
    fp = fourier_series(Piecewise((0, x < 0), (pi, True)), (x, -pi, pi))
    return fo, fe, fp

# 定义测试函数 test_FourierSeries
def test_FourierSeries():
    # 调用 _get_examples 获取 Fourier 级数示例 fo, fe, fp
    fo, fe, fp = _get_examples()

    # 断言 1 在区间 (-pi, pi) 上的 Fourier 级数为 1
    assert fourier_series(1, (-pi, pi)) == 1
    # 断言分段函数 (Piecewise((0, x < 0), (pi, True))) 的 Fourier 级数截断结果与 fp 的截断结果相同
    assert (Piecewise((0, x < 0), (pi, True)).
            fourier_series((x, -pi, pi)).truncate()) == fp.truncate()
    # 断言 fo 是 FourierSeries 类的实例
    assert isinstance(fo, FourierSeries)
    # 断言 fo 的函数部分为 x
    assert fo.function == x
    # 断言 fo 的自变量为 x
    assert fo.x == x
    # 断言 fo 的周期为 (-pi, pi)
    assert fo.period == (-pi, pi)

    # 断言 fo 的第三项系数为 2*sin(3*x) / 3
    assert fo.term(3) == 2*sin(3*x) / 3
    # 断言 fe 的第三项系数为 -4*cos(3*x) / 9
    assert fe.term(3) == -4*cos(3*x) / 9
    # 断言 fp 的第三项系数为 2*sin(3*x) / 3
    assert fp.term(3) == 2*sin(3*x) / 3

    # 断言 fo 在 x 处的主导项为 2*sin(x)
    assert fo.as_leading_term(x) == 2*sin(x)
    # 断言 fe 在 x 处的主导项为 pi**2 / 3
    assert fe.as_leading_term(x) == pi**2 / 3
    # 断言 fp 在 x 处的主导项为 pi / 2
    assert fp.as_leading_term(x) == pi / 2

    # 断言 fo 的截断结果为 2*sin(x) - sin(2*x) + (2*sin(3*x) / 3)
    assert fo.truncate() == 2*sin(x) - sin(2*x) + (2*sin(3*x) / 3)
    # 断言 fe 的截断结果为 -4*cos(x) + cos(2*x) + pi**2 / 3
    assert fe.truncate() == -4*cos(x) + cos(2*x) + pi**2 / 3
    # 断言 fp 的截断结果为 2*sin(x) + (2*sin(3*x) / 3) + pi / 2
    assert fp.truncate() == 2*sin(x) + (2*sin(3*x) / 3) + pi / 2

    # 获取 fo 的截断结果 fot，期望的序列 s
    fot = fo.truncate(n=None)
    s = [0, 2*sin(x), -sin(2*x)]
    # 对 fot 进行遍历验证
    for i, t in enumerate(fot):
        if i == 3:
            break
        # 断言 fot 的第 i 项与 s[i] 相等
        assert s[i] == t

    # 定义内部函数 _check_iter，验证 f 的前 i 项与 f[ind] 相等
    def _check_iter(f, i):
        for ind, t in enumerate(f):
            assert t == f[ind]
            if ind == i:
                break

    # 分别对 fo, fe, fp 进行前 3 项的验证
    _check_iter(fo, 3)
    _check_iter(fe, 3)
    _check_iter(fp, 3)

    # 断言 fo 在 x=x^2 时不变
    assert fo.subs(x, x**2) == fo

    # 断言调用 fourier_series 函数时抛出 ValueError 异常
    raises(ValueError, lambda: fourier_series(x, (0, 1, 2)))
    raises(ValueError, lambda: fourier_series(x, (x, 0, oo)))
    raises(ValueError, lambda: fourier_series(x*y, (0, oo)))

# 定义测试函数 test_FourierSeries_2
def test_FourierSeries_2():
    # 定义分段函数 p
    p = Piecewise((0, x < 0), (x, True))
    # 计算 p 在区间 (-2, 2) 上的 Fourier 级数
    f = fourier_series(p, (x, -2, 2))

    # 断言 f 的第三项系数为 (2*sin(3*pi*x / 2) / (3*pi) - 4*cos(3*pi*x / 2) / (9*pi**2))
    assert f.term(3) == (2*sin(3*pi*x / 2) / (3*pi) -
                         4*cos(3*pi*x / 2) / (9*pi**2))
    # 断言 f 的截断结果为 (2*sin(pi*x / 2) / pi - sin(pi*x) / pi - 4*cos(pi*x / 2) / pi**2 + S.Half)

# 定义测试函数 test_square_wave
def test_square_wave():
    """Test if fourier_series approximates discontinuous function correctly."""
    # 定义方波函数 square_wave
    square_wave = Piecewise((1, x < pi), (-1, True))
    # 计算 square_wave 在区间 (0, 2*pi) 上的 Fourier 级数
    s = fourier_series(square_wave, (x, 0, 2*pi))

    # 断言 s 的前 3 项截断结果为 4 / pi * sin(x) + 4 / (3 * pi) * sin(3 * x) + 4 / (5 * pi) * sin(5 * x)
    # 断言语句：验证 s 对象的 sigma_approximation 方法在参数为 4 时的返回值是否等于以下表达式的结果
    assert s.sigma_approximation(4) == 4 / pi * sin(x) * sinc(pi / 4) + \
        4 / (3 * pi) * sin(3 * x) * sinc(3 * pi / 4)
# 定义测试函数，用于验证傅里叶级数的截断结果
def test_sawtooth_wave():
    # 对给定的周期函数计算其傅里叶级数，并断言截断后的结果
    s = fourier_series(x, (x, 0, pi))
    assert s.truncate(4) == \
        pi/2 - sin(2*x) - sin(4*x)/2 - sin(6*x)/3
    # 对另一个定义域内的周期函数计算其傅里叶级数，并断言截断后的结果
    s = fourier_series(x, (x, 0, 1))
    assert s.truncate(4) == \
        S.Half - sin(2*pi*x)/pi - sin(4*pi*x)/(2*pi) - sin(6*pi*x)/(3*pi)


# 定义测试傅里叶级数对象的操作函数
def test_FourierSeries__operations():
    # 获取示例函数
    fo, fe, fp = _get_examples()

    # 对傅里叶级数对象进行缩放和平移，并断言截断后的结果
    fes = fe.scale(-1).shift(pi**2)
    assert fes.truncate() == 4*cos(x) - cos(2*x) + 2*pi**2 / 3

    # 对傅里叶级数对象进行水平平移，并断言截断后的结果
    assert fp.shift(-pi/2).truncate() == (2*sin(x) + (2*sin(3*x) / 3) +
                                          (2*sin(5*x) / 5))

    # 对傅里叶级数对象进行缩放，并断言截断后的结果
    fos = fo.scale(3)
    assert fos.truncate() == 6*sin(x) - 3*sin(2*x) + 2*sin(3*x)

    # 对傅里叶级数对象在 x 轴方向进行缩放和平移，并断言截断后的结果
    fx = fe.scalex(2).shiftx(1)
    assert fx.truncate() == -4*cos(2*x + 2) + cos(4*x + 4) + pi**2 / 3

    # 对傅里叶级数对象进行复合操作，并断言截断后的结果
    fl = fe.scalex(3).shift(-pi).scalex(2).shiftx(1).scale(4)
    assert fl.truncate() == (-16*cos(6*x + 6) + 4*cos(12*x + 12) -
                             4*pi + 4*pi**2 / 3)

    # 断言特定操作引发 ValueError 异常
    raises(ValueError, lambda: fo.shift(x))
    raises(ValueError, lambda: fo.shiftx(sin(x)))
    raises(ValueError, lambda: fo.scale(x*y))
    raises(ValueError, lambda: fo.scalex(x**2))


# 定义测试傅里叶级数对象的取负操作函数
def test_FourierSeries__neg():
    # 获取示例函数
    fo, fe, fp = _get_examples()

    # 断言傅里叶级数对象取负后的截断结果
    assert (-fo).truncate() == -2*sin(x) + sin(2*x) - (2*sin(3*x) / 3)
    assert (-fe).truncate() == +4*cos(x) - cos(2*x) - pi**2 / 3


# 定义测试傅里叶级数对象的加法和减法操作函数
def test_FourierSeries__add__sub():
    # 获取示例函数
    fo, fe, fp = _get_examples()

    # 断言傅里叶级数对象加自身的结果等于缩放系数为2的结果
    assert fo + fo == fo.scale(2)
    # 断言傅里叶级数对象减自身的结果等于0
    assert fo - fo == 0
    # 断言傅里叶级数对象取负两次的结果等于其缩放系数为-2的结果
    assert -fe - fe == fe.scale(-2)

    # 断言傅里叶级数对象之间的加法和减法操作的截断结果
    assert (fo + fe).truncate() == 2*sin(x) - sin(2*x) - 4*cos(x) + cos(2*x) \
        + pi**2 / 3
    assert (fo - fe).truncate() == 2*sin(x) - sin(2*x) + 4*cos(x) - cos(2*x) \
        - pi**2 / 3

    # 断言对傅里叶级数对象加常数引发 TypeError 异常
    assert isinstance(fo + 1, Add)
    raises(ValueError, lambda: fo + fourier_series(x, (x, 0, 2)))


# 定义测试有限傅里叶级数的函数
def test_FourierSeries_finite():

    # 断言对正弦函数的傅里叶级数截断结果
    assert fourier_series(sin(x)).truncate(1) == sin(x)
    # 断言对复合函数的傅里叶级数截断结果的类型为傅里叶级数对象
    # assert type(fourier_series(sin(x)*log(x))).truncate() == FourierSeries
    # assert type(fourier_series(sin(x**2+6))).truncate() == FourierSeries
    # 断言对带有多个变量的函数的傅里叶级数截断结果
    assert fourier_series(sin(x)*log(y)*exp(z),(x,pi,-pi)).truncate() == sin(x)*log(y)*exp(z)
    # 断言对高次幂正弦函数的傅里叶级数截断结果
    assert fourier_series(sin(x)**6).truncate(oo) == -15*cos(2*x)/32 + 3*cos(4*x)/16 - cos(6*x)/32 \
           + Rational(5, 16)
    assert fourier_series(sin(x) ** 6).truncate() == -15 * cos(2 * x) / 32 + 3 * cos(4 * x) / 16 \
           + Rational(5, 16)
    # 断言对复合三角函数的傅里叶级数截断结果
    assert fourier_series(sin(4*x+3) + cos(3*x+4)).truncate(oo) ==  -sin(4)*sin(3*x) + sin(4*x)*cos(3) \
           + cos(4)*cos(3*x) + sin(3)*cos(4*x)
    # 断言对包含三角函数和其它函数的傅里叶级数截断结果
    assert fourier_series(sin(x)+cos(x)*tan(x)).truncate(oo) == 2*sin(x)
    # 断言对余弦函数的傅里叶级数截断结果
    assert fourier_series(cos(pi*x), (x, -1, 1)).truncate(oo) == cos(pi*x)
    # 断言对复合余弦函数的傅里叶级数截断结果
    assert fourier_series(cos(3*pi*x + 4) - sin(4*pi*x)*log(pi*y), (x, -1, 1)).truncate(oo) == -log(pi*y)*sin(4*pi*x)\
           - sin(4)*sin(3*pi*x) + cos(4)*cos(3*pi*x)
```