# `D:\src\scipysrc\sympy\sympy\codegen\numpy_nodes.py`

```
# 从 sympy 库中导入特定模块和类
from sympy.core.function import Add, ArgumentIndexError, Function
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.exponential import exp, log

# 定义函数 _logaddexp，计算两个指数函数的和的对数
def _logaddexp(x1, x2, *, evaluate=True):
    return log(Add(exp(x1, evaluate=evaluate), exp(x2, evaluate=evaluate), evaluate=evaluate))

# 定义常量 _two，表示常数 2
_two = S.One*2

# 计算 ln(2) 的值，即常数 2 的自然对数
_ln2 = log(_two)

# 定义函数 _lb，计算给定数的对数在以 2 为底的对数
def _lb(x, *, evaluate=True):
    return log(x, evaluate=evaluate)/_ln2

# 定义函数 _exp2，计算常数 2 的给定幂次方
def _exp2(x, *, evaluate=True):
    return Pow(_two, x, evaluate=evaluate)

# 定义函数 _logaddexp2，计算以 2 为底的两个指数函数的和的对数
def _logaddexp2(x1, x2, *, evaluate=True):
    return _lb(Add(_exp2(x1, evaluate=evaluate),
                   _exp2(x2, evaluate=evaluate), evaluate=evaluate))

# 定义类 logaddexp，表示对输入进行指数函数和的对数操作的函数
class logaddexp(Function):
    """ Logarithm of the sum of exponentiations of the inputs.

    Helper class for use with e.g. numpy.logaddexp

    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html
    """
    nargs = 2

    # 构造函数，对输入参数进行排序并返回函数实例
    def __new__(cls, *args):
        return Function.__new__(cls, *sorted(args, key=default_sort_key))

    # 返回该函数的一阶导数
    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        # 根据参数索引选择对应的参数，并返回导数表达式
        if argindex == 1:
            wrt, other = self.args
        elif argindex == 2:
            other, wrt = self.args
        else:
            raise ArgumentIndexError(self, argindex)
        return S.One/(S.One + exp(other-wrt))

    # 重写函数，将其表达为对数函数的操作
    def _eval_rewrite_as_log(self, x1, x2, **kwargs):
        return _logaddexp(x1, x2)

    # 对函数进行数值评估，返回数值结果
    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(log).evalf(*args, **kwargs)

    # 简化函数表达式，返回简化后的结果
    def _eval_simplify(self, *args, **kwargs):
        a, b = (x.simplify(**kwargs) for x in self.args)
        candidate = _logaddexp(a, b)
        if candidate != _logaddexp(a, b, evaluate=False):
            return candidate
        else:
            return logaddexp(a, b)

# 定义类 logaddexp2，表示对输入进行以 2 为底的指数函数和的对数操作的函数
class logaddexp2(Function):
    """ Logarithm of the sum of exponentiations of the inputs in base-2.

    Helper class for use with e.g. numpy.logaddexp2

    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html
    """
    nargs = 2

    # 构造函数，对输入参数进行排序并返回函数实例
    def __new__(cls, *args):
        return Function.__new__(cls, *sorted(args, key=default_sort_key))

    # 返回该函数的一阶导数
    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        # 根据参数索引选择对应的参数，并返回导数表达式
        if argindex == 1:
            wrt, other = self.args
        elif argindex == 2:
            other, wrt = self.args
        else:
            raise ArgumentIndexError(self, argindex)
        return S.One/(S.One + _exp2(other-wrt))

    # 重写函数，将其表达为以 2 为底的对数函数的操作
    def _eval_rewrite_as_log(self, x1, x2, **kwargs):
        return _logaddexp2(x1, x2)

    # 对函数进行数值评估，返回数值结果
    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(log).evalf(*args, **kwargs)
    # 定义一个方法 `_eval_simplify`，该方法接受任意位置参数 `args` 和关键字参数 `kwargs`
    a, b = (x.simplify(**kwargs).factor() for x in self.args)
    # 使用生成器表达式对 `self.args` 中的每个元素 `x` 进行简化和因式分解，将结果分配给 `a` 和 `b`
    candidate = _logaddexp2(a, b)
    # 调用 `_logaddexp2` 函数，将 `a` 和 `b` 作为参数传递给它，并将结果赋给 `candidate`
    if candidate != _logaddexp2(a, b, evaluate=False):
        # 如果 `_logaddexp2` 函数的默认计算结果不等于禁用评估模式下的结果
        return candidate
        # 返回 `candidate`
    else:
        # 如果上述条件不满足
        return logaddexp2(a, b)
        # 返回调用 `logaddexp2` 函数时 `a` 和 `b` 作为参数的结果
```