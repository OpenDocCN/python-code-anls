# `D:\src\scipysrc\sympy\sympy\strategies\tests\test_rl.py`

```
# 导入需要的符号和函数
from sympy.core.singleton import S
from sympy.strategies.rl import (
    rm_id, glom, flatten, unpack, sort, distribute, subs, rebuild)
from sympy.core.basic import Basic  # 导入基本的符号和表达式类
from sympy.core.add import Add  # 导入加法表达式类
from sympy.core.mul import Mul  # 导入乘法表达式类
from sympy.core.symbol import symbols  # 导入符号定义函数
from sympy.abc import x  # 导入符号 x


def test_rm_id():
    # 创建一个用于移除特定元素的函数
    rmzeros = rm_id(lambda x: x == 0)
    # 断言移除零后的基本表达式的正确性
    assert rmzeros(Basic(S(0), S(1))) == Basic(S(1))
    assert rmzeros(Basic(S(0), S(0))) == Basic(S(0))
    assert rmzeros(Basic(S(2), S(1))) == Basic(S(2), S(1))


def test_glom():
    # 定义用于分组和重构的函数
    def key(x):
        return x.as_coeff_Mul()[1]

    def count(x):
        return x.as_coeff_Mul()[0]

    def newargs(cnt, arg):
        return cnt * arg

    rl = glom(key, count, newargs)  # 创建一个重构表达式的函数
    # 测试用例
    result = rl(Add(x, -x, 3 * x, 2, 3, evaluate=False))
    expected = Add(3 * x, 5)
    # 断言重构后的表达式参数集合是否与预期相同
    assert set(result.args) == set(expected.args)


def test_flatten():
    # 测试表达式扁平化函数
    assert flatten(Basic(S(1), S(2), Basic(S(3), S(4)))) == \
        Basic(S(1), S(2), S(3), S(4))


def test_unpack():
    # 测试表达式解包函数
    assert unpack(Basic(S(2))) == 2
    assert unpack(Basic(S(2), S(3))) == Basic(S(2), S(3))


def test_sort():
    # 测试表达式排序函数
    assert sort(str)(Basic(S(3), S(1), S(2))) == Basic(S(1), S(2), S(3))


def test_distribute():
    # 定义用于分配的自定义类
    class T1(Basic):
        pass

    class T2(Basic):
        pass

    distribute_t12 = distribute(T1, T2)  # 创建一个分配函数
    # 测试用例
    assert distribute_t12(T1(S(1), S(2), T2(S(3), S(4)), S(5))) == \
        T2(T1(S(1), S(2), S(3), S(5)), T1(S(1), S(2), S(4), S(5)))
    assert distribute_t12(T1(S(1), S(2), S(3))) == T1(S(1), S(2), S(3))


def test_distribute_add_mul():
    x, y = symbols('x, y')
    expr = Mul(2, Add(x, y), evaluate=False)
    expected = Add(Mul(2, x), Mul(2, y))
    distribute_mul = distribute(Mul, Add)  # 创建一个乘法和加法的分配函数
    assert distribute_mul(expr) == expected


def test_subs():
    rl = subs(1, 2)  # 创建一个替换函数
    assert rl(1) == 2
    assert rl(3) == 3


def test_rebuild():
    expr = Basic.__new__(Add, S(1), S(2))  # 创建一个新的基本表达式对象
    assert rebuild(expr) == 3  # 重构并断言结果正确
```