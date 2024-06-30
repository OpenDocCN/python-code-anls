# `D:\src\scipysrc\sympy\sympy\codegen\scipy_nodes.py`

```
from sympy.core.function import Add, ArgumentIndexError, Function
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import cos, sin

# 定义一个函数 _cosm1，计算 cos(x) - 1
def _cosm1(x, *, evaluate=True):
    # 返回 cos(x) - 1 的结果
    return Add(cos(x, evaluate=evaluate), -S.One, evaluate=evaluate)

# 定义一个函数类 cosm1，表示 cos(x) - 1 函数
class cosm1(Function):
    """ Minus one plus cosine of x, i.e. cos(x) - 1. For use when x is close to zero.

    Helper class for use with e.g. scipy.special.cosm1
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.cosm1.html
    """
    nargs = 1  # 指定函数的参数个数为 1

    # 返回该函数的第一阶导数
    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            # 如果参数索引为 1，返回 -sin(x)
            return -sin(*self.args)
        else:
            # 抛出参数索引错误
            raise ArgumentIndexError(self, argindex)

    # 将函数重写为 cos(x) - 1 形式
    def _eval_rewrite_as_cos(self, x, **kwargs):
        return _cosm1(x)

    # 对函数进行数值估算
    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(cos).evalf(*args, **kwargs)

    # 对函数进行简化操作
    def _eval_simplify(self, **kwargs):
        x, = self.args
        candidate = _cosm1(x.simplify(**kwargs))
        # 如果简化后的结果与未评估的结果不同，则返回简化后的结果
        if candidate != _cosm1(x, evaluate=False):
            return candidate
        else:
            return cosm1(x)

# 定义一个函数 _powm1，计算 x**y - 1
def _powm1(x, y, *, evaluate=True):
    # 返回 x**y - 1 的结果
    return Add(Pow(x, y, evaluate=evaluate), -S.One, evaluate=evaluate)

# 定义一个函数类 powm1，表示 x**y - 1 函数
class powm1(Function):
    """ Minus one plus x to the power of y, i.e. x**y - 1. For use when x is close to one or y is close to zero.

    Helper class for use with e.g. scipy.special.powm1
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.powm1.html
    """
    nargs = 2  # 指定函数的参数个数为 2

    # 返回该函数的第一阶导数
    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            # 如果参数索引为 1，返回 x**y * y / x
            return Pow(self.args[0], self.args[1])*self.args[1]/self.args[0]
        elif argindex == 2:
            # 如果参数索引为 2，返回 log(x) * x**y
            return log(self.args[0])*Pow(*self.args)
        else:
            # 抛出参数索引错误
            raise ArgumentIndexError(self, argindex)

    # 将函数重写为 Pow 形式
    def _eval_rewrite_as_Pow(self, x, y, **kwargs):
        return _powm1(x, y)

    # 对函数进行数值估算
    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(Pow).evalf(*args, **kwargs)

    # 对函数进行简化操作
    def _eval_simplify(self, **kwargs):
        x, y = self.args
        candidate = _powm1(x.simplify(**kwargs), y.simplify(**kwargs))
        # 如果简化后的结果与未评估的结果不同，则返回简化后的结果
        if candidate != _powm1(x, y, evaluate=False):
            return candidate
        else:
            return powm1(x, y)
```