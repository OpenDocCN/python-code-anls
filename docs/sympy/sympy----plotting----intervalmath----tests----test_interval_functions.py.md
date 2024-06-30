# `D:\src\scipysrc\sympy\sympy\plotting\intervalmath\tests\test_interval_functions.py`

```
# 从 sympy.external 模块中导入 import_module 函数
# 从 sympy.plotting.intervalmath 模块中导入一系列数学函数和操作符
from sympy.external import import_module
from sympy.plotting.intervalmath import (
    Abs, acos, acosh, And, asin, asinh, atan, atanh, ceil, cos, cosh,
    exp, floor, imax, imin, interval, log, log10, Or, sin, sinh, sqrt,
    tan, tanh,
)

# 使用 import_module 函数导入 'numpy' 模块，并将其赋值给 np
np = import_module('numpy')
# 如果未成功导入 numpy 模块，则设置 disabled 变量为 True
if not np:
    disabled = True


# 定义一个函数 test_interval_pow，用于测试区间幂运算
def test_interval_pow():
    # 测试 2 的区间 [1, 2] 的幂是否等于区间 [2, 4]
    a = 2**interval(1, 2) == interval(2, 4)
    assert a == (True, True)
    # 测试区间 [1, 2] 的幂的区间 [1, 2] 是否等于区间 [1, 4]
    a = interval(1, 2)**interval(1, 2) == interval(1, 4)
    assert a == (True, True)
    # 测试区间 [-1, 1] 的幂的区间 [0.5, 2]，预期结果是无效区间
    a = interval(-1, 1)**interval(0.5, 2)
    assert a.is_valid is None
    # 测试区间 [-2, -1] 的幂的区间 [1, 2]，预期结果是无效区间
    a = interval(-2, -1) ** interval(1, 2)
    assert a.is_valid is False
    # 测试区间 [-2, -1] 的幂的单个数值 0.5，预期结果是无效区间
    a = interval(-2, -1) ** (1.0 / 2)
    assert a.is_valid is False
    # 测试区间 [-1, 1] 的幂的单个数值 0.5，预期结果是无效区间
    a = interval(-1, 1)**(1.0 / 2)
    assert a.is_valid is None
    # 测试区间 [-1, 1] 的幂的单个数值 1/3，是否等于区间 [-1, 1]
    a = interval(-1, 1)**(1.0 / 3) == interval(-1, 1)
    assert a == (True, True)
    # 测试区间 [-1, 1] 的幂的单个数值 2，是否等于区间 [0, 1]
    a = interval(-1, 1)**2 == interval(0, 1)
    assert a == (True, True)
    # 测试区间 [-1, 1] 的幂的单个数值 1/29，是否等于区间 [-1, 1]
    a = interval(-1, 1) ** (1.0 / 29) == interval(-1, 1)
    assert a == (True, True)
    # 测试 -2 的区间 [1, 1] 的幂是否等于区间 [-2, -2]
    a = -2**interval(1, 1) == interval(-2, -2)
    assert a == (True, True)

    # 测试区间 [1, 2] 的 is_valid 设置为 False 后的幂运算，预期结果是无效区间
    a = interval(1, 2, is_valid=False)**2
    assert a.is_valid is False

    # 测试区间 [-3, -3] 的幂的区间 [1, 2]，预期结果是无效区间
    a = (-3)**interval(1, 2)
    assert a.is_valid is False
    # 测试单个数值 -4 的区间 [0.5, 0.5] 的幂，预期结果是无效区间
    a = (-4)**interval(0.5, 0.5)
    assert a.is_valid is False
    # 测试单个数值 -3 的区间 [1, 1] 的幂是否等于区间 [-3, -3]
    assert ((-3)**interval(1, 1) == interval(-3, -3)) == (True, True)

    # 测试区间 [8, 64] 的幂的单个数值 2/3，验证结果是否接近 [4, 16]
    a = interval(8, 64)**(2.0 / 3)
    assert abs(a.start - 4) < 1e-10  # eps
    assert abs(a.end - 16) < 1e-10
    # 测试区间 [-8, 64] 的幂的单个数值 2/3，验证结果是否接近 [4, 16]
    a = interval(-8, 64)**(2.0 / 3)
    assert abs(a.start - 4) < 1e-10  # eps
    assert abs(a.end - 16) < 1e-10


# 定义一个函数 test_exp，用于测试区间的指数函数
def test_exp():
    # 对区间 [-∞, 0] 应用指数函数，验证结果的起始和结束值是否正确
    a = exp(interval(-np.inf, 0))
    assert a.start == np.exp(-np.inf)
    assert a.end == np.exp(0)
    # 对区间 [1, 2] 应用指数函数，验证结果的起始和结束值是否正确
    a = exp(interval(1, 2))
    assert a.start == np.exp(1)
    assert a.end == np.exp(2)
    # 对单个数值 1 应用指数函数，验证结果的起始和结束值是否正确
    a = exp(1)
    assert a.start == np.exp(1)
    assert a.end == np.exp(1)


# 定义一个函数 test_log，用于测试区间的自然对数函数
def test_log():
    # 对区间 [1, 2] 应用自然对数函数，验证结果的起始和结束值是否正确
    a = log(interval(1, 2))
    assert a.start == 0
    assert a.end == np.log(2)
    # 对区间 [-1, 1] 应用自然对数函数，预期结果是无效区间
    a = log(interval(-1, 1))
    assert a.is_valid is None
    # 对区间 [-3, -1] 应用自然对数函数，预期结果是无效区间
    a = log(interval(-3, -1))
    assert a.is_valid is False
    # 对单个数值 -3 应用自然对数函数，预期结果是无效区间
    a = log(-3)
    assert a.is_valid is False
    # 对单个数值 2 应用自然对数函数，验证结果的起始和结束值是否正确
    a = log(2)
    assert a.start == np.log(2)
    assert a.end == np.log(2)


# 定义一个函数 test_log10，用于测试区间的以10为底的对数函数
def test_log10():
    # 对区间 [1, 2] 应用以10为底的对数函数，验证结果的起始和结束值是否正确
    a = log10(interval(1, 2))
    assert a.start == 0
    assert a.end == np.log10(2)
    # 对区间 [-1, 1] 应用以10为底的对数函数，预期结果是无效区间
    a = log10(interval(-1, 1))
    assert a.is_valid is None
    # 对区间 [-3, -1] 应用以10为底的对数函数，预期结果是无效区间
    a = log10(interval(-3, -1))
    assert a.is_valid is False
    # 对单个数值 -3 应用以10为底的对数函数，预期结果是无效区间
    a = log10(-3)
    assert a.is_valid is False
    # 对单个数值 2 应用以10为底的对数函数，验证结果的起始和结束值是否正确
    a = log10(2)
    assert a.start == np.log10(2)
    assert a.end == np.log
    # 使用 interval 函数创建一个区间对象，并对其应用 sin 函数
    a = sin(interval(-np.pi / 4, np.pi / 4))
    # 断言区间起始点的值等于 np.sin(-np.pi / 4)
    assert a.start == np.sin(-np.pi / 4)
    # 断言区间结束点的值等于 np.sin(np.pi / 4)
    assert a.end == np.sin(np.pi / 4)

    # 使用 interval 函数创建一个区间对象，并对其应用 sin 函数
    a = sin(interval(np.pi / 4, 3 * np.pi / 4))
    # 断言区间起始点的值等于 np.sin(np.pi / 4)
    assert a.start == np.sin(np.pi / 4)
    # 断言区间结束点的值等于 1
    assert a.end == 1

    # 使用 interval 函数创建一个区间对象，并对其应用 sin 函数
    a = sin(interval(7 * np.pi / 6, 7 * np.pi / 4))
    # 断言区间起始点的值等于 -1
    assert a.start == -1
    # 断言区间结束点的值等于 np.sin(7 * np.pi / 6)
    assert a.end == np.sin(7 * np.pi / 6)

    # 使用 interval 函数创建一个区间对象，并对其应用 sin 函数
    a = sin(interval(0, 3 * np.pi))
    # 断言区间起始点的值等于 -1
    assert a.start == -1
    # 断言区间结束点的值等于 1
    assert a.end == 1

    # 使用 interval 函数创建一个区间对象，并对其应用 sin 函数
    a = sin(interval(np.pi / 3, 7 * np.pi / 4))
    # 断言区间起始点的值等于 -1
    assert a.start == -1
    # 断言区间结束点的值等于 1
    assert a.end == 1

    # 对单个数值应用 sin 函数
    a = sin(np.pi / 4)
    # 断言结果为单个数值的起始和结束点的值相同，均为 np.sin(np.pi / 4)
    assert a.start == np.sin(np.pi / 4)
    assert a.end == np.sin(np.pi / 4)

    # 使用 interval 函数创建一个区间对象，is_valid 参数设置为 False
    a = sin(interval(1, 2, is_valid=False))
    # 断言该区间对象的 is_valid 属性为 False
    assert a.is_valid is False
# 定义一个测试函数 test_cos，用于测试 cos 函数在给定区间的表现
def test_cos():
    # 在区间 [0, π/4] 上调用 cos 函数
    a = cos(interval(0, np.pi / 4))
    # 断言起始值等于 cos(π/4)，即区间的左端点对应的余弦值
    assert a.start == np.cos(np.pi / 4)
    # 断言结束值为 1
    assert a.end == 1

    # 在区间 [-π/4, π/4] 上调用 cos 函数
    a = cos(interval(-np.pi / 4, np.pi / 4))
    # 断言起始值等于 cos(-π/4)，即区间的左端点对应的余弦值
    assert a.start == np.cos(-np.pi / 4)
    # 断言结束值为 1
    assert a.end == 1

    # 在区间 [π/4, 3π/4] 上调用 cos 函数
    a = cos(interval(np.pi / 4, 3 * np.pi / 4))
    # 断言起始值等于 cos(3π/4)，即区间的左端点对应的余弦值
    assert a.start == np.cos(3 * np.pi / 4)
    # 断言结束值等于 cos(π/4)，即区间的右端点对应的余弦值
    assert a.end == np.cos(np.pi / 4)

    # 在区间 [3π/4, 5π/4] 上调用 cos 函数
    a = cos(interval(3 * np.pi / 4, 5 * np.pi / 4))
    # 断言起始值等于 -1
    assert a.start == -1
    # 断言结束值等于 cos(3π/4)，即区间的左端点对应的余弦值
    assert a.end == np.cos(3 * np.pi / 4)

    # 在区间 [0, 3π] 上调用 cos 函数
    a = cos(interval(0, 3 * np.pi))
    # 断言起始值等于 -1
    assert a.start == -1
    # 断言结束值等于 1
    assert a.end == 1

    # 在区间 [-π/3, 5π/4] 上调用 cos 函数
    a = cos(interval(-np.pi / 3, 5 * np.pi / 4))
    # 断言起始值等于 -1
    assert a.start == -1
    # 断言结束值等于 1
    assert a.end == 1

    # 在区间 [1, 2] 上调用 cos 函数，并标记结果为无效
    a = cos(interval(1, 2, is_valid=False))
    # 断言结果的有效性为 False
    assert a.is_valid is False



# 定义一个测试函数 test_tan，用于测试 tan 函数在给定区间的表现
def test_tan():
    # 在区间 [0, π/4] 上调用 tan 函数
    a = tan(interval(0, np.pi / 4))
    # 断言起始值为 0
    assert a.start == 0
    # 根据 lib_interval 定义的 tan 函数，断言结束值应为 sin(π/4)/cos(π/4)
    assert a.end == np.sin(np.pi / 4) / np.cos(np.pi / 4)

    # 在区间 [π/4, 3π/4] 上调用 tan 函数
    a = tan(interval(np.pi / 4, 3 * np.pi / 4))
    # 断言结果的有效性为 None，即 tan 在此区间上存在间断点
    assert a.is_valid is None



# 定义一个测试函数 test_sqrt，用于测试 sqrt 函数在给定区间的表现
def test_sqrt():
    # 在区间 [1, 4] 上调用 sqrt 函数
    a = sqrt(interval(1, 4))
    # 断言起始值为 1
    assert a.start == 1
    # 断言结束值为 2
    assert a.end == 2

    # 在区间 [0.01, 1] 上调用 sqrt 函数
    a = sqrt(interval(0.01, 1))
    # 断言起始值为 sqrt(0.01)
    assert a.start == np.sqrt(0.01)
    # 断言结束值为 1
    assert a.end == 1

    # 在区间 [-1, 1] 上调用 sqrt 函数
    a = sqrt(interval(-1, 1))
    # 断言结果的有效性为 None，即 sqrt 函数不能处理负数区间
    assert a.is_valid is None

    # 调用 sqrt 函数传入一个数值 4
    a = sqrt(4)
    # 断言结果等于区间 [2, 2]
    assert (a == interval(2, 2)) == (True, True)

    # 调用 sqrt 函数传入一个负数 -3
    a = sqrt(-3)
    # 断言结果的有效性为 False，即 sqrt 函数不能处理负数
    assert a.is_valid is False



# 定义一个测试函数 test_imin，用于测试 imin 函数在给定区间的表现
def test_imin():
    # 在区间 [1, 3], [2, 5], [-1, 3] 上调用 imin 函数
    a = imin(interval(1, 3), interval(2, 5), interval(-1, 3))
    # 断言起始值为 -1
    assert a.start == -1
    # 断言结束值为 3
    assert a.end == 3

    # 在区间 [-2, 2], [1, 4] 上调用 imin 函数
    a = imin(-2, interval(1, 4))
    # 断言起始值为 -2
    assert a.start == -2
    # 断言结束值为 -2
    assert a.end == -2

    # 在区间 [3, 4], [-2, 2] 上调用 imin 函数，并标记区间 [-2, 2] 为无效
    a = imin(5, interval(3, 4), interval(-2, 2, is_valid=False))
    # 断言起始值为 3
    assert a.start == 3
    # 断言结束值为 4
    assert a.end == 4



# 定义一个测试函数 test_imax，用于测试 imax 函数在给定区间的表现
def test_imax():
    # 在区间 [-2, 2], [2, 7], [-3, 9] 上调用 imax 函数
    a = imax(interval(-2, 2), interval(2, 7), interval(-3, 9))
    # 断言起始值为 2
    assert a.start == 2
    # 断言结束值为 9
    assert a.end == 9

    # 在区间 [1, 4], 8 上调用 imax 函数
    a = imax(8, interval(1, 4))
    # 断言起始值为 8
    assert a.start == 8
    # 断言结束值为 8
    assert a.end == 8

    # 在区间 [1,
    # 使用区间[-1.5, 1.5]调用asin函数，返回区间对象a
    a = asin(interval(-1.5, 1.5))
    # 断言a对象的有效性属性为None
    assert a.is_valid is None
    
    # 使用区间[-2, -1.5]调用asin函数，返回区间对象a
    a = asin(interval(-2, -1.5))
    # 断言a对象的有效性属性为False
    assert a.is_valid is False
    
    # 使用区间[0, 2]调用asin函数，返回区间对象a
    a = asin(interval(0, 2))
    # 断言a对象的有效性属性为None
    assert a.is_valid is None
    
    # 使用区间[2, 5]调用asin函数，返回区间对象a
    a = asin(interval(2, 5))
    # 断言a对象的有效性属性为False
    assert a.is_valid is False
    
    # 使用值0.5调用asin函数，返回值a
    a = asin(0.5)
    # 断言a对象的起始值等于numpy中0.5的反正弦值
    assert a.start == np.arcsin(0.5)
    # 断言a对象的结束值等于numpy中0.5的反正弦值
    assert a.end == np.arcsin(0.5)
    
    # 使用值1.5调用asin函数，返回值a
    a = asin(1.5)
    # 断言a对象的有效性属性为False
    assert a.is_valid is False
# 定义测试函数 test_acos，用于测试 acos 函数的返回值
def test_acos():
    # 对 interval(-0.5, 0.5) 调用 acos 函数，并验证其起始点和终止点是否与 np.arccos(0.5) 一致
    a = acos(interval(-0.5, 0.5))
    assert a.start == np.arccos(0.5)
    assert a.end == np.arccos(-0.5)

    # 对 interval(-1.5, 1.5) 调用 acos 函数，验证其 is_valid 属性是否为 None
    a = acos(interval(-1.5, 1.5))
    assert a.is_valid is None

    # 对 interval(-2, -1.5) 调用 acos 函数，验证其 is_valid 属性是否为 False
    a = acos(interval(-2, -1.5))
    assert a.is_valid is False

    # 对 interval(0, 2) 调用 acos 函数，验证其 is_valid 属性是否为 None
    a = acos(interval(0, 2))
    assert a.is_valid is None

    # 对 interval(2, 5) 调用 acos 函数，验证其 is_valid 属性是否为 False
    a = acos(interval(2, 5))
    assert a.is_valid is False

    # 对单个数值 0.5 调用 acos 函数，并验证其起始点和终止点是否相同，均为 np.arccos(0.5)
    a = acos(0.5)
    assert a.start == np.arccos(0.5)
    assert a.end == np.arccos(0.5)

    # 对单个数值 1.5 调用 acos 函数，验证其 is_valid 属性是否为 False
    a = acos(1.5)
    assert a.is_valid is False


# 定义测试函数 test_ceil，用于测试 ceil 函数的返回值
def test_ceil():
    # 对 interval(0.2, 0.5) 调用 ceil 函数，验证其起始点和终止点是否为 1
    a = ceil(interval(0.2, 0.5))
    assert a.start == 1
    assert a.end == 1

    # 对 interval(0.5, 1.5) 调用 ceil 函数，验证其起始点和终止点是否为 1 和 2，同时验证 is_valid 属性是否为 None
    a = ceil(interval(0.5, 1.5))
    assert a.start == 1
    assert a.end == 2
    assert a.is_valid is None

    # 对 interval(-5, 5) 调用 ceil 函数，验证其 is_valid 属性是否为 None
    a = ceil(interval(-5, 5))
    assert a.is_valid is None

    # 对单个数值 5.4 调用 ceil 函数，并验证其起始点和终止点是否为 6
    a = ceil(5.4)
    assert a.start == 6
    assert a.end == 6


# 定义测试函数 test_floor，用于测试 floor 函数的返回值
def test_floor():
    # 对 interval(0.2, 0.5) 调用 floor 函数，验证其起始点和终止点是否为 0
    a = floor(interval(0.2, 0.5))
    assert a.start == 0
    assert a.end == 0

    # 对 interval(0.5, 1.5) 调用 floor 函数，验证其起始点和终止点是否为 0 和 1，同时验证 is_valid 属性是否为 None
    a = floor(interval(0.5, 1.5))
    assert a.start == 0
    assert a.end == 1
    assert a.is_valid is None

    # 对 interval(-5, 5) 调用 floor 函数，验证其 is_valid 属性是否为 None
    a = floor(interval(-5, 5))
    assert a.is_valid is None

    # 对单个数值 5.4 调用 floor 函数，并验证其起始点和终止点是否为 5
    a = floor(5.4)
    assert a.start == 5
    assert a.end == 5


# 定义测试函数 test_asinh，用于测试 asinh 函数的返回值
def test_asinh():
    # 对 interval(1, 2) 调用 asinh 函数，并验证其起始点和终止点是否与 np.arcsinh(1) 和 np.arcsinh(2) 一致
    a = asinh(interval(1, 2))
    assert a.start == np.arcsinh(1)
    assert a.end == np.arcsinh(2)

    # 对单个数值 0.5 调用 asinh 函数，并验证其起始点和终止点是否与 np.arcsinh(0.5) 一致
    a = asinh(0.5)
    assert a.start == np.arcsinh(0.5)
    assert a.end == np.arcsinh(0.5)


# 定义测试函数 test_acosh，用于测试 acosh 函数的返回值
def test_acosh():
    # 对 interval(3, 5) 调用 acosh 函数，并验证其起始点和终止点是否与 np.arccosh(3) 和 np.arccosh(5) 一致
    a = acosh(interval(3, 5))
    assert a.start == np.arccosh(3)
    assert a.end == np.arccosh(5)

    # 对 interval(0, 3) 调用 acosh 函数，验证其 is_valid 属性是否为 None
    a = acosh(interval(0, 3))
    assert a.is_valid is None

    # 对 interval(-3, 0.5) 调用 acosh 函数，验证其 is_valid 属性是否为 False
    a = acosh(interval(-3, 0.5))
    assert a.is_valid is False

    # 对单个数值 0.5 调用 acosh 函数，验证其 is_valid 属性是否为 False
    a = acosh(0.5)
    assert a.is_valid is False

    # 对单个数值 2 调用 acosh 函数，并验证其起始点和终止点是否与 np.arccosh(2) 一致
    a = acosh(2)
    assert a.start == np.arccosh(2)
    assert a.end == np.arccosh(2)


# 定义测试函数 test_atanh，用于测试 atanh 函数的返回值
def test_atanh():
    # 对 interval(-0.5, 0.5) 调用 atanh 函数，并验证其起始点和终止点是否与 np.arctanh(-0.5) 和 np.arctanh(0.5) 一致
    a = atanh(interval(-0.5, 0.5))
    assert a.start == np.arctanh(-0.5)
    assert a.end == np.arctanh(0.5)

    # 对 interval(0, 3) 调用 atanh 函数，验证其 is_valid 属性是否为 None
    a = atanh(interval(0, 3))
    assert a.is_valid is None

    # 对 interval(-3, -2) 调用 atanh 函数，验证其 is_valid 属性是否为 False
    a = atanh(interval(-3, -2))
    assert a.is_valid is False

    # 对单个数值 0.5 调用 atanh 函数，并验证其起始点和终止点是否与 np.arctanh(0.5) 一致
    a = atanh(0.5)
    assert a.start == np.arctanh(0.5)
    assert a.end == np.arctanh(0.5)

    # 对单个数值 1.5 调用 atanh 函数，验证其 is_valid 属性是否为 False
    a = atanh(1.5)
    assert a.is_valid is False


# 定义测试函数 test_Abs，用于测试 Abs 函数的返回值
def test_Abs():
    # 对 interval(-0.5, 0.5) 调用 Abs 函数，验证其结果是否与 interval(0, 0.5) 相同，并且返回值均为 True
    assert (Abs(interval(-0.5, 0.5)) == interval(0, 0.5)) == (True, True)

    # 对 interval(-3, -2) 调用 Abs 函数，验证其结果是否与 interval(2, 3
```