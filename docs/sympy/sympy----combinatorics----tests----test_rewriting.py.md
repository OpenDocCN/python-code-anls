# `D:\src\scipysrc\sympy\sympy\combinatorics\tests\test_rewriting.py`

```
from sympy.combinatorics.fp_groups import FpGroup
from sympy.combinatorics.free_groups import free_group
from sympy.testing.pytest import raises

# 定义一个测试函数，用于测试重写系统的功能
def test_rewriting():
    # 创建自由群 F，以及生成元 a 和 b
    F, a, b = free_group("a, b")
    # 创建 FpGroup 对象 G，其中包含一个关于生成元 a 和 b 的关系
    G = FpGroup(F, [a*b*a**-1*b**-1])
    # 重新获取生成元 a 和 b
    a, b = G.generators
    # 获取重写系统 R
    R = G._rewriting_system
    # 断言重写系统 R 是可合流的
    assert R.is_confluent

    # 使用 reduce 方法测试正规形式的计算
    assert G.reduce(b**-1*a) == a*b**-1
    assert G.reduce(b**3*a**4*b**-2*a) == a**5*b
    assert G.equals(b**2*a**-1*b, b**4*a**-1*b**-1)

    # 使用 reduce_using_automaton 方法测试自动机的规约
    assert R.reduce_using_automaton(b*a*a**2*b**-1) == a**3
    assert R.reduce_using_automaton(b**3*a**4*b**-2*a) == a**5*b
    assert R.reduce_using_automaton(b**-1*a) == a*b**-1

    # 使用新的关系创建 FpGroup 对象 G
    G = FpGroup(F, [a**3, b**3, (a*b)**2])
    R = G._rewriting_system
    # 使得重写系统 R 可合流
    R.make_confluent()
    # 断言重写系统 R 现在是可合流的
    assert R.is_confluent
    # 同时确保系统实际上是可合流的
    assert R._check_confluence()
    assert G.reduce(b*a**-1*b**-1*a**3*b**4*a**-1*b**-15) == a**-1*b**-1
    # 检查自动机规约
    assert R.reduce_using_automaton(b*a**-1*b**-1*a**3*b**4*a**-1*b**-15) == a**-1*b**-1

    # 使用新的关系创建 FpGroup 对象 G
    G = FpGroup(F, [a**2, b**3, (a*b)**4])
    R = G._rewriting_system
    # 使用 reduce 方法测试正规形式的计算
    assert G.reduce(a**2*b**-2*a**2*b) == b**-1
    assert R.reduce_using_automaton(a**2*b**-2*a**2*b) == b**-1
    assert G.reduce(a**3*b**-2*a**2*b) == a**-1*b**-1
    assert R.reduce_using_automaton(a**3*b**-2*a**2*b) == a**-1*b**-1
    # 在添加规则后进行检查
    R.add_rule(a**2, b)
    assert R.reduce_using_automaton(a**2*b**-2*a**2*b) == b**-1
    assert R.reduce_using_automaton(a**4*b**-2*a**2*b**3) == b

    # 设置最大限制为 15，测试是否会引发 RuntimeError
    R.set_max(15)
    raises(RuntimeError, lambda:  R.add_rule(a**-3, b))
    # 再次设置最大限制为 20，成功添加规则
    R.set_max(20)
    R.add_rule(a**-3, b)

    # 断言添加规则 a -> a 的结果是空集
    assert R.add_rule(a, a) == set()
```