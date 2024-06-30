# `D:\src\scipysrc\sympy\sympy\core\tests\test_truediv.py`

```
# 导入 SymPy 的相关模块，用于测试 SymPy 在真除法开启时的工作情况
from sympy.core.numbers import (Float, Rational)
from sympy.core.symbol import Symbol


# 定义测试真除法的函数
def test_truediv():
    # 断言：1/2 不等于 0，验证真除法是否正确工作
    assert 1/2 != 0
    # 断言：使用有理数 Rational(1) 进行真除法操作，不等于 0
    assert Rational(1)/2 != 0


# 定义测试函数 dotest，接受一个参数 s
def dotest(s):
    # 创建符号变量 x 和 y
    x = Symbol("x")
    y = Symbol("y")
    # 列表 l 包含不同类型的数值和表达式
    l = [
        Rational(2),      # 有理数 2
        Float("1.3"),     # 浮点数 1.3
        x,                # 符号变量 x
        y,                # 符号变量 y
        pow(x, y)*y,      # 表达式 pow(x, y)*y
        5,                # 整数 5
        5.5               # 浮点数 5.5
    ]
    # 对列表 l 中的每对元素 (x, y) 调用函数 s
    for x in l:
        for y in l:
            s(x, y)
    return True


# 定义基本运算测试函数 test_basic
def test_basic():
    # 定义函数 s(a, b)，用于测试基本运算
    def s(a, b):
        x = a        # 将参数 a 赋值给变量 x
        x = +a       # 取变量 a 的正值
        x = -a       # 取变量 a 的负值
        x = a + b    # 变量 a 与 b 的加法运算
        x = a - b    # 变量 a 与 b 的减法运算
        x = a*b      # 变量 a 与 b 的乘法运算
        x = a/b      # 变量 a 与 b 的真除法运算
        x = a**b     # 变量 a 的 b 次幂运算
        del x        # 删除变量 x
    # 断言：调用 dotest 函数并传入 s 函数作为参数
    assert dotest(s)


# 定义增强赋值运算测试函数 test_ibasic
def test_ibasic():
    # 定义函数 s(a, b)，用于测试增强赋值运算
    def s(a, b):
        x = a        # 将参数 a 赋值给变量 x
        x += b       # 变量 x 自增变量 b
        x = a        # 将参数 a 赋值给变量 x
        x -= b       # 变量 x 自减变量 b
        x = a        # 将参数 a 赋值给变量 x
        x *= b       # 变量 x 自乘变量 b
        x = a        # 将参数 a 赋值给变量 x
        x /= b       # 变量 x 自除以变量 b
    # 断言：调用 dotest 函数并传入 s 函数作为参数
    assert dotest(s)
```