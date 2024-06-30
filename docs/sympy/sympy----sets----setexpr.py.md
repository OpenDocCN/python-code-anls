# `D:\src\scipysrc\sympy\sympy\sets\setexpr.py`

```
from sympy.core import Expr
from sympy.core.decorators import call_highest_priority, _sympifyit
from .fancysets import ImageSet
from .sets import set_add, set_sub, set_mul, set_div, set_pow, set_function

class SetExpr(Expr):
    """An expression that can take on values of a set.

    Examples
    ========

    >>> from sympy import Interval, FiniteSet
    >>> from sympy.sets.setexpr import SetExpr

    >>> a = SetExpr(Interval(0, 5))
    >>> b = SetExpr(FiniteSet(1, 10))
    >>> (a + b).set
    Union(Interval(1, 6), Interval(10, 15))
    >>> (2*a + b).set
    Interval(1, 20)
    """
    _op_priority = 11.0  # 设置操作的优先级

    def __new__(cls, setarg):
        return Expr.__new__(cls, setarg)  # 调用父类Expr的构造方法创建新的实例

    set = property(lambda self: self.args[0])  # 获取SetExpr实例的set属性，即存储在args[0]中的集合对象

    def _latex(self, printer):
        return r"SetExpr\left({}\right)".format(printer._print(self.set))  # 返回SetExpr对象的Latex表示形式

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other):
        return _setexpr_apply_operation(set_add, self, other)  # 调用_setexpr_apply_operation进行加法操作

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__add__')
    def __radd__(self, other):
        return _setexpr_apply_operation(set_add, other, self)  # 调用_setexpr_apply_operation进行反向加法操作

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return _setexpr_apply_operation(set_mul, self, other)  # 调用_setexpr_apply_operation进行乘法操作

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return _setexpr_apply_operation(set_mul, other, self)  # 调用_setexpr_apply_operation进行反向乘法操作

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return _setexpr_apply_operation(set_sub, self, other)  # 调用_setexpr_apply_operation进行减法操作

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return _setexpr_apply_operation(set_sub, other, self)  # 调用_setexpr_apply_operation进行反向减法操作

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        return _setexpr_apply_operation(set_pow, self, other)  # 调用_setexpr_apply_operation进行乘方操作

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        return _setexpr_apply_operation(set_pow, other, self)  # 调用_setexpr_apply_operation进行反向乘方操作

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return _setexpr_apply_operation(set_div, self, other)  # 调用_setexpr_apply_operation进行真除操作

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        return _setexpr_apply_operation(set_div, other, self)  # 调用_setexpr_apply_operation进行反向真除操作

    def _eval_func(self, func):
        # TODO: this could be implemented straight into `imageset`:
        res = set_function(func, self.set)  # 调用set_function函数对self.set应用func函数
        if res is None:
            return SetExpr(ImageSet(func, self.set))  # 如果res为None，则返回ImageSet(func, self.set)的SetExpr形式
        return SetExpr(res)  # 否则返回res的SetExpr形式


def _setexpr_apply_operation(op, x, y):
    if isinstance(x, SetExpr):
        x = x.set  # 如果x是SetExpr实例，则将其转换为其内部的set对象
    if isinstance(y, SetExpr):
        y = y.set  # 如果y是SetExpr实例，则将其转换为其内部的set对象
    out = op(x, y)  # 对x和y执行op操作，返回结果
    # 返回一个 SetExpr 类的实例化对象，并将 out 作为参数传递给构造函数
    return SetExpr(out)
```