# `D:\src\scipysrc\sympy\sympy\core\tests\test_random.py`

```
import random  # 导入random模块，用于生成随机数
from sympy.core.random import random as rand, seed, shuffle, _assumptions_shuffle  # 从sympy.core.random模块导入random函数和其他函数
from sympy.core.symbol import Symbol, symbols  # 从sympy.core.symbol模块导入Symbol和symbols函数
from sympy.functions.elementary.trigonometric import sin, acos  # 从sympy.functions.elementary.trigonometric模块导入sin和acos函数
from sympy.abc import x  # 从sympy.abc模块导入变量x


def test_random():
    random.seed(42)  # 设置随机数种子为42
    a = random.random()  # 生成一个随机浮点数并赋给a
    random.seed(42)  # 重新设置随机数种子为42
    Symbol('z').is_finite  # 创建一个Symbol对象，并检查其是否有限

    b = random.random()  # 再次生成一个随机浮点数并赋给b
    assert a == b  # 断言a与b相等，验证随机数生成的一致性

    got = set()  # 创建一个空集合
    for i in range(2):
        random.seed(28)  # 设置随机数种子为28
        m0, m1 = symbols('m_0 m_1', real=True)  # 创建两个实数符号m0和m1
        _ = acos(-m0/m1)  # 计算acos(-m0/m1)，但未使用结果
        got.add(random.uniform(0,1))  # 生成一个0到1之间的随机浮点数，并加入到集合got中
    assert len(got) == 1  # 断言集合got中元素的个数为1，验证生成的随机数是一致的

    random.seed(10)  # 设置随机数种子为10
    y = 0  # 初始化y为0
    for i in range(4):
        y += sin(random.uniform(-10,10) * x)  # 生成一个在[-10, 10]之间的随机数乘以x，然后计算sin的和并加到y中
    random.seed(10)  # 重新设置随机数种子为10
    z = 0  # 初始化z为0
    for i in range(4):
        z += sin(random.uniform(-10,10) * x)  # 生成一个在[-10, 10]之间的随机数乘以x，然后计算sin的和并加到z中
    assert y == z  # 断言y与z相等，验证随机数生成的一致性


def test_seed():
    assert rand() < 1  # 断言rand()函数生成的随机数小于1
    seed(1)  # 设置随机数种子为1
    a = rand()  # 生成一个随机数并赋给a
    b = rand()  # 再次生成一个随机数并赋给b
    seed(1)  # 重新设置随机数种子为1
    c = rand()  # 再次生成一个随机数并赋给c
    d = rand()  # 再次生成一个随机数并赋给d
    assert a == c  # 断言a与c相等，验证随机数生成的一致性
    if not c == d:  # 如果c不等于d
        assert a != b  # 断言a不等于b
    else:  # 否则
        assert a == b  # 断言a等于b，验证随机数生成的一致性

    abc = 'abc'  # 创建一个字符串'abc'
    first = list(abc)  # 将字符串转换为列表赋给first
    second = list(abc)  # 将字符串转换为列表赋给second
    third = list(abc)  # 将字符串转换为列表赋给third

    seed(123)  # 设置随机数种子为123
    shuffle(first)  # 对列表first进行原地乱序操作

    seed(123)  # 重新设置随机数种子为123
    shuffle(second)  # 对列表second进行原地乱序操作
    _assumptions_shuffle(third)  # 对列表third进行乱序操作

    assert first == second == third  # 断言first、second、third三者相等，验证乱序操作的一致性
```