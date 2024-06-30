# `D:\src\scipysrc\sympy\sympy\integrals\singularityfunctions.py`

```
# 从 sympy 库中导入 SingularityFunction 和 DiracDelta 函数
from sympy.functions import SingularityFunction, DiracDelta
# 从 sympy 库中导入 integrate 函数
from sympy.integrals import integrate


# 定义 singularityintegrate 函数，处理奇异函数的不定积分
def singularityintegrate(f, x):
    """
    This function handles the indefinite integrations of Singularity functions.
    The ``integrate`` function calls this function internally whenever an
    instance of SingularityFunction is passed as argument.

    Explanation
    ===========

    The idea for integration is the following:

    - If we are dealing with a SingularityFunction expression,
      i.e. ``SingularityFunction(x, a, n)``, we just return
      ``SingularityFunction(x, a, n + 1)/(n + 1)`` if ``n >= 0`` and
      ``SingularityFunction(x, a, n + 1)`` if ``n < 0``.

    - If the node is a multiplication or power node having a
      SingularityFunction term we rewrite the whole expression in terms of
      Heaviside and DiracDelta and then integrate the output. Lastly, we
      rewrite the output of integration back in terms of SingularityFunction.

    - If none of the above case arises, we return None.

    Examples
    ========

    >>> from sympy.integrals.singularityfunctions import singularityintegrate
    >>> from sympy import SingularityFunction, symbols, Function
    >>> x, a, n, y = symbols('x a n y')
    >>> f = Function('f')
    >>> singularityintegrate(SingularityFunction(x, a, 3), x)
    SingularityFunction(x, a, 4)/4
    >>> singularityintegrate(5*SingularityFunction(x, 5, -2), x)
    5*SingularityFunction(x, 5, -1)
    >>> singularityintegrate(6*SingularityFunction(x, 5, -1), x)
    6*SingularityFunction(x, 5, 0)
    >>> singularityintegrate(x*SingularityFunction(x, 0, -1), x)
    0
    >>> singularityintegrate(SingularityFunction(x, 1, -1) * f(x), x)
    f(1)*SingularityFunction(x, 1, 0)

    """

    # 如果 f 中不包含 SingularityFunction，则返回 None
    if not f.has(SingularityFunction):
        return None

    # 如果 f 是 SingularityFunction 的实例
    if isinstance(f, SingularityFunction):
        x, a, n = f.args
        # 如果 n 是正数或零，返回 SingularityFunction(x, a, n + 1)/(n + 1)
        if n.is_positive or n.is_zero:
            return SingularityFunction(x, a, n + 1)/(n + 1)
        # 如果 n 是 -1 到 -4 之间的负数，则返回 SingularityFunction(x, a, n + 1)
        elif n in (-1, -2, -3, -4):
            return SingularityFunction(x, a, n + 1)

    # 如果 f 是乘法或幂运算
    if f.is_Mul or f.is_Pow:
        # 将 f 用 DiracDelta 重写，并对其进行积分
        expr = f.rewrite(DiracDelta)
        expr = integrate(expr, x)
        # 将积分后的表达式重写为 SingularityFunction 形式
        return expr.rewrite(SingularityFunction)

    # 如果以上条件都不满足，则返回 None
    return None
```