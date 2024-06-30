# `D:\src\scipysrc\sympy\sympy\functions\elementary\tests\test_interface.py`

```
# 导入需要的模块和函数
from sympy.core.function import Function
from sympy.core.sympify import sympify
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.limits import limit
from sympy.abc import x

# 定义一个测试函数，用于测试自定义函数的级数展开和极限计算
def test_function_series1():
    """Create our new "sin" function."""
    
    # 定义自定义的函数类，继承自 sympy 的 Function 类
    class my_function(Function):
        
        # 定义函数的一阶导数
        def fdiff(self, argindex=1):
            return cos(self.args[0])

        # 定义函数的求值方法
        @classmethod
        def eval(cls, arg):
            arg = sympify(arg)
            if arg == 0:
                return sympify(0)

    # 测试自定义函数的泰勒级数展开是否与标准 sin 函数的级数展开一致
    assert my_function(x).series(x, 0, 10) == sin(x).series(x, 0, 10)
    # 测试自定义函数在 x 趋近于 0 时的极限是否为 1
    assert limit(my_function(x)/x, x, 0) == 1

# 定义第二个测试函数，用于测试另一个自定义函数的级数展开
def test_function_series2():
    """Create our new "cos" function."""
    
    # 定义另一个自定义的函数类，同样继承自 sympy 的 Function 类
    class my_function2(Function):
        
        # 定义函数的一阶导数
        def fdiff(self, argindex=1):
            return -sin(self.args[0])

        # 定义函数的求值方法
        @classmethod
        def eval(cls, arg):
            arg = sympify(arg)
            if arg == 0:
                return sympify(1)

    # 测试自定义函数的泰勒级数展开是否与标准 cos 函数的级数展开一致
    assert my_function2(x).series(x, 0, 10) == cos(x).series(x, 0, 10)

# 定义第三个测试函数，用于测试另一个自定义函数的级数展开和函数求值
def test_function_series3():
    """
    Test our easy "tanh" function.

    This test tests two things:
      * that the Function interface works as expected and it's easy to use
      * that the general algorithm for the series expansion works even when the
        derivative is defined recursively in terms of the original function,
        since tanh(x).diff(x) == 1-tanh(x)**2
    """
    
    # 定义一个计算双曲正切函数 tanh 的自定义函数类
    class mytanh(Function):
        
        # 定义函数的一阶导数
        def fdiff(self, argindex=1):
            return 1 - mytanh(self.args[0])**2

        # 定义函数的求值方法
        @classmethod
        def eval(cls, arg):
            arg = sympify(arg)
            if arg == 0:
                return sympify(0)

    # 创建标准的 tanh(x) 和自定义的 mytanh(x)
    e = tanh(x)
    f = mytanh(x)
    # 测试自定义函数的泰勒级数展开是否与标准 tanh 函数的级数展开一致
    assert e.series(x, 0, 6) == f.series(x, 0, 6)
```