# `D:\src\scipysrc\sympy\sympy\core\tests\test_priority.py`

```
# 导入所需模块和函数装饰器
from sympy.core.decorators import call_highest_priority
from sympy.core.expr import Expr
from sympy.core.mod import Mod
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.functions.elementary.integers import floor

# 定义一个继承自Integer的类Higher，代表整数1，优先级_op_priority设为20
class Higher(Integer):
    '''
    Integer of value 1 and _op_priority 20

    Operations handled by this class return 1 and reverse operations return 2
    '''

    _op_priority = 20.0
    result = 1

    # 定义构造函数，初始化p为1
    def __new__(cls):
        obj = Expr.__new__(cls)
        obj.p = 1
        return obj

    # 定义乘法操作，使用call_highest_priority装饰器调用__rmul__，返回结果1
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return self.result

    # 定义反向乘法操作，使用call_highest_priority装饰器调用__mul__，返回结果2
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return 2*self.result

    # 同上，定义加法、反向加法、减法、反向减法、乘方、反向乘方、真除法、反向真除法、求余、反向求余、整除和反向整除的操作和反向操作
    @call_highest_priority('__radd__')
    def __add__(self, other):
        return self.result

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return 2*self.result

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return self.result

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return 2*self.result

    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        return self.result

    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        return 2*self.result

    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return self.result

    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        return 2*self.result

    @call_highest_priority('__rmod__')
    def __mod__(self, other):
        return self.result

    @call_highest_priority('__mod__')
    def __rmod__(self, other):
        return 2*self.result

    @call_highest_priority('__rfloordiv__')
    def __floordiv__(self, other):
        return self.result

    @call_highest_priority('__floordiv__')
    def __rfloordiv__(self, other):
        return 2*self.result


# 定义一个继承自Higher的类Lower，代表整数-1，优先级_op_priority设为5
class Lower(Higher):
    '''
    Integer of value -1 and _op_priority 5

    Operations handled by this class return -1 and reverse operations return -2
    '''

    _op_priority = 5.0
    result = -1

    # 定义构造函数，初始化p为-1
    def __new__(cls):
        obj = Expr.__new__(cls)
        obj.p = -1
        return obj


# 创建一个符号变量x
x = Symbol('x')

# 创建Higher类的实例h和Lower类的实例l
h = Higher()
l = Lower()


# 定义测试函数test_mul，测试乘法操作
def test_mul():
    assert h*l == h*x == 1
    assert l*h == x*h == 2
    assert x*l == l*x == -x


# 定义测试函数test_add，测试加法操作
def test_add():
    assert h + l == h + x == 1
    assert l + h == x + h == 2
    assert x + l == l + x == x - 1


# 定义测试函数test_sub，测试减法操作
def test_sub():
    assert h - l == h - x == 1
    assert l - h == x - h == 2
    assert x - l == -(l - x) == x + 1


# 定义测试函数test_pow，测试乘方操作
def test_pow():
    assert h**l == h**x == 1
    assert l**h == x**h == 2
    assert (x**l).args == (1/x).args and (x**l).is_Pow
    assert (l**x).args == ((-1)**x).args and (l**x).is_Pow


# 定义测试函数test_div，测试真除法操作
def test_div():
    assert h/l == h/x == 1
    assert l/h == x/h == 2
    assert x/l == 1/(l/x) == -x


# 定义测试函数test_mod，测试求余操作
def test_mod():
    assert h%l == h%x == 1
    assert l%h == x%h == 2
    #`
    # 断言 x 对 l 取模的结果等于使用 Mod 类对 x 取模 (-1) 的结果
    assert x % l == Mod(x, -1)
    
    # 断言 l 对 x 取模的结果等于使用 Mod 类对 (-1) 取模 x 的结果
    assert l % x == Mod(-1, x)
# 定义一个测试函数，用于测试整数除法运算（取整操作）

def test_floordiv():
    # 断言：h 除以 l 的整数部分等于 h 除以 x 的整数部分，都应该等于 1
    assert h//l == h//x == 1
    # 断言：l 除以 h 的整数部分等于 x 除以 h 的整数部分，都应该等于 2
    assert l//h == x//h == 2
    # 断言：x 除以 l 的整数部分等于 floor(-x) 的结果
    assert x//l == floor(-x)
    # 断言：l 除以 x 的整数部分等于 floor(-1/x) 的结果
    assert l//x == floor(-1/x)
```