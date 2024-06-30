# `D:\src\scipysrc\sympy\sympy\core\benchmarks\bench_numbers.py`

```
# 导入必要的类和常量
from sympy.core.numbers import Integer, Rational, pi, oo
from sympy.core.intfunc import integer_nthroot, igcd
from sympy.core.singleton import S

# 创建一个整数对象，值为 3
i3 = Integer(3)
# 创建一个整数对象，值为 4
i4 = Integer(4)
# 创建一个有理数对象，值为 3/4
r34 = Rational(3, 4)
# 创建一个有理数对象，值为 4/5
q45 = Rational(4, 5)

# 定义一个函数，用于测试创建整数对象的运行时间
def timeit_Integer_create():
    Integer(2)

# 定义一个函数，用于测试将整数对象转换为 int 类型的运行时间
def timeit_Integer_int():
    int(i3)

# 定义一个函数，用于测试取单例对象 S.One 的负数的运行时间
def timeit_neg_one():
    -S.One

# 定义一个函数，用于测试将整数对象取负数的运行时间
def timeit_Integer_neg():
    -i3

# 定义一个函数，用于测试取整数对象的绝对值的运行时间
def timeit_Integer_abs():
    abs(i3)

# 定义一个函数，用于测试整数对象之间的减法运行时间
def timeit_Integer_sub():
    i3 - i3

# 定义一个函数，用于测试取圆周率的绝对值的运行时间
def timeit_abs_pi():
    abs(pi)

# 定义一个函数，用于测试取负无穷大的运行时间
def timeit_neg_oo():
    -oo

# 定义一个函数，用于测试整数对象与整数常量相加的运行时间
def timeit_Integer_add_i1():
    i3 + 1

# 定义一个函数，用于测试整数对象之间相加的运行时间
def timeit_Integer_add_ij():
    i3 + i4

# 定义一个函数，用于测试整数对象与有理数对象相加的运行时间
def timeit_Integer_add_Rational():
    i3 + r34

# 定义一个函数，用于测试整数对象乘以整数常量的运行时间
def timeit_Integer_mul_i4():
    i3 * 4

# 定义一个函数，用于测试整数对象之间相乘的运行时间
def timeit_Integer_mul_ij():
    i3 * i4

# 定义一个函数，用于测试整数对象与有理数对象相乘的运行时间
def timeit_Integer_mul_Rational():
    i3 * r34

# 定义一个函数，用于测试整数对象与整数常量是否相等的运行时间
def timeit_Integer_eq_i3():
    i3 == 3

# 定义一个函数，用于测试整数对象与有理数对象是否相等的运行时间
def timeit_Integer_ed_Rational():
    i3 == r34

# 定义一个函数，用于测试计算整数的平方根的运行时间
def timeit_integer_nthroot():
    integer_nthroot(100, 2)

# 定义一个函数，用于测试计算两个整数的最大公约数的运行时间
def timeit_number_igcd_23_17():
    igcd(23, 17)

# 定义一个函数，用于测试计算两个整数的最大公约数的运行时间
def timeit_number_igcd_60_3600():
    igcd(60, 3600)

# 定义一个函数，用于测试有理数对象与整数常量相加的运行时间
def timeit_Rational_add_r1():
    r34 + 1

# 定义一个函数，用于测试有理数对象之间相加的运行时间
def timeit_Rational_add_rq():
    r34 + q45
```