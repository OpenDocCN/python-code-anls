# `D:\src\scipysrc\sympy\sympy\sets\tests\test_ordinals.py`

```
from sympy.sets.ordinals import Ordinal, OmegaPower, ord0, omega  # 导入必要的类和函数
from sympy.testing.pytest import raises  # 导入异常处理函数

def test_string_ordinals():
    assert str(omega) == 'w'  # 检查omega对象的字符串表示是否为'w'
    assert str(Ordinal(OmegaPower(5, 3), OmegaPower(3, 2))) == 'w**5*3 + w**3*2'  # 检查指定组合的Ordinal对象的字符串表示
    assert str(Ordinal(OmegaPower(5, 3), OmegaPower(0, 5))) == 'w**5*3 + 5'  # 检查指定组合的Ordinal对象的字符串表示
    assert str(Ordinal(OmegaPower(1, 3), OmegaPower(0, 5))) == 'w*3 + 5'  # 检查指定组合的Ordinal对象的字符串表示
    assert str(Ordinal(OmegaPower(omega + 1, 1), OmegaPower(3, 2))) == 'w**(w + 1) + w**3*2'  # 检查指定组合的Ordinal对象的字符串表示

def test_addition_with_integers():
    assert 3 + Ordinal(OmegaPower(5, 3)) == Ordinal(OmegaPower(5, 3))  # 检查整数和Ordinal对象的加法运算
    assert Ordinal(OmegaPower(5, 3))+3 == Ordinal(OmegaPower(5, 3), OmegaPower(0, 3))  # 检查Ordinal对象和整数的加法运算
    assert Ordinal(OmegaPower(5, 3), OmegaPower(0, 2))+3 == \
        Ordinal(OmegaPower(5, 3), OmegaPower(0, 5))  # 检查Ordinal对象和整数的加法运算

def test_addition_with_ordinals():
    assert Ordinal(OmegaPower(5, 3), OmegaPower(3, 2)) + Ordinal(OmegaPower(3, 3)) == \
        Ordinal(OmegaPower(5, 3), OmegaPower(3, 5))  # 检查Ordinal对象之间的加法运算
    assert Ordinal(OmegaPower(5, 3), OmegaPower(3, 2)) + Ordinal(OmegaPower(4, 2)) == \
        Ordinal(OmegaPower(5, 3), OmegaPower(4, 2))  # 检查Ordinal对象之间的加法运算
    assert Ordinal(OmegaPower(omega, 2), OmegaPower(3, 2)) + Ordinal(OmegaPower(4, 2)) == \
        Ordinal(OmegaPower(omega, 2), OmegaPower(4, 2))  # 检查Ordinal对象之间的加法运算

def test_comparison():
    assert Ordinal(OmegaPower(5, 3)) > Ordinal(OmegaPower(4, 3), OmegaPower(2, 1))  # 检查Ordinal对象之间的比较
    assert Ordinal(OmegaPower(5, 3), OmegaPower(3, 2)) < Ordinal(OmegaPower(5, 4))  # 检查Ordinal对象之间的比较
    assert Ordinal(OmegaPower(5, 4)) < Ordinal(OmegaPower(5, 5), OmegaPower(4, 1))  # 检查Ordinal对象之间的比较

    assert Ordinal(OmegaPower(5, 3), OmegaPower(3, 2)) == \
        Ordinal(OmegaPower(5, 3), OmegaPower(3, 2))  # 检查Ordinal对象之间的相等比较
    assert not Ordinal(OmegaPower(5, 3), OmegaPower(3, 2)) == Ordinal(OmegaPower(5, 3))  # 检查Ordinal对象之间的不相等比较
    assert Ordinal(OmegaPower(omega, 3)) > Ordinal(OmegaPower(5, 3))  # 检查Ordinal对象之间的比较

def test_multiplication_with_integers():
    w = omega
    assert 3*w == w  # 检查整数和Ordinal对象的乘法运算
    assert w*9 == Ordinal(OmegaPower(1, 9))  # 检查Ordinal对象和整数的乘法运算

def test_multiplication():
    w = omega
    assert w*(w + 1) == w*w + w  # 检查Ordinal对象之间的乘法运算
    assert (w + 1)*(w + 1) ==  w*w + w + 1  # 检查Ordinal对象之间的乘法运算
    assert w*1 == w  # 检查Ordinal对象和整数的乘法运算
    assert 1*w == w  # 检查整数和Ordinal对象的乘法运算
    assert w*ord0 == ord0  # 检查Ordinal对象和ord0的乘法运算
    assert ord0*w == ord0  # 检查ord0和Ordinal对象的乘法运算
    assert w**w == w * w**w  # 检查Ordinal对象之间的乘法运算
    assert (w**w)*w*w == w**(w + 2)  # 检查Ordinal对象之间的乘法运算

def test_exponentiation():
    w = omega
    assert w**2 == w*w  # 检查Ordinal对象的幂运算
    assert w**3 == w*w*w  # 检查Ordinal对象的幂运算
    assert w**(w + 1) == Ordinal(OmegaPower(omega + 1, 1))  # 检查Ordinal对象的幂运算
    assert (w**w)*(w**w) == w**(w*2)  # 检查Ordinal对象的幂运算

def test_compare_not_instance():
    w = OmegaPower(omega + 1, 1)
    assert(not (w == None))  # 检查对象是否不是None
    assert(not (w < 5))  # 检查对象是否不小于5
    raises(TypeError, lambda: w < 6.66)  # 检查对象是否引发TypeError异常

def test_is_successort():
    w = Ordinal(OmegaPower(5, 1))
    assert not w.is_successor_ordinal  # 检查对象的is_successor_ordinal属性是否为False
```