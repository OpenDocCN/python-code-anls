# `D:\src\scipysrc\sympy\sympy\assumptions\tests\test_wrapper.py`

```
# 导入需要的模块和函数
from sympy.assumptions.ask import Q
from sympy.assumptions.wrapper import (AssumptionsWrapper, is_infinite,
    is_extended_real)
from sympy.core.symbol import Symbol
from sympy.core.assumptions import _assume_defined


# 测试所有预定义谓词的存在
def test_all_predicates():
    # 遍历预定义的谓词列表
    for fact in _assume_defined:
        # 构造谓词方法名
        method_name = f'_eval_is_{fact}'
        # 断言在AssumptionsWrapper类中存在该谓词方法
        assert hasattr(AssumptionsWrapper, method_name)


# 测试AssumptionsWrapper类的功能
def test_AssumptionsWrapper():
    # 创建带有正属性的符号x
    x = Symbol('x', positive=True)
    # 创建没有特定属性的符号y
    y = Symbol('y')
    # 断言：对于带有正属性的符号x，is_positive应为True
    assert AssumptionsWrapper(x).is_positive
    # 断言：对于没有特定属性的符号y，is_positive应为None
    assert AssumptionsWrapper(y).is_positive is None
    # 断言：对于带有正属性的符号y，通过Q.positive(y)，is_positive应为True
    assert AssumptionsWrapper(y, Q.positive(y)).is_positive


# 测试is_infinite函数
def test_is_infinite():
    # 创建带有无穷属性的符号x
    x = Symbol('x', infinite=True)
    # 创建不带有无穷属性的符号y
    y = Symbol('y', infinite=False)
    # 创建没有特定属性的符号z
    z = Symbol('z')
    # 断言：对于带有无穷属性的符号x，is_infinite应为True
    assert is_infinite(x)
    # 断言：对于不带有无穷属性的符号y，is_infinite应为False
    assert not is_infinite(y)
    # 断言：对于没有特定属性的符号z，is_infinite应为None
    assert is_infinite(z) is None
    # 断言：对于带有无穷属性的符号z，通过Q.infinite(z)，is_infinite应为True
    assert is_infinite(z, Q.infinite(z))


# 测试is_extended_real函数
def test_is_extended_real():
    # 创建带有扩展实数属性的符号x
    x = Symbol('x', extended_real=True)
    # 创建不带有扩展实数属性的符号y
    y = Symbol('y', extended_real=False)
    # 创建没有特定属性的符号z
    z = Symbol('z')
    # 断言：对于带有扩展实数属性的符号x，is_extended_real应为True
    assert is_extended_real(x)
    # 断言：对于不带有扩展实数属性的符号y，is_extended_real应为False
    assert not is_extended_real(y)
    # 断言：对于没有特定属性的符号z，is_extended_real应为None
    assert is_extended_real(z) is None
    # 断言：对于带有扩展实数属性的符号z，通过Q.extended_real(z)，is_extended_real应为True
    assert is_extended_real(z, Q.extended_real(z))
```