# `D:\src\scipysrc\sympy\sympy\functions\special\mathieu_functions.py`

```
""" This module contains the Mathieu functions.
"""

# 导入所需模块
from sympy.core.function import Function, ArgumentIndexError
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, cos

# 定义 MathieuBase 类，作为 Mathieu 函数的抽象基类
class MathieuBase(Function):
    """
    Abstract base class for Mathieu functions.

    This class is meant to reduce code duplication.
    """

    unbranched = True

    # 定义共轭函数的求值
    def _eval_conjugate(self):
        a, q, z = self.args
        return self.func(a.conjugate(), q.conjugate(), z.conjugate())

# 定义 Mathieu Sine 函数 mathieus
class mathieus(MathieuBase):
    r"""
    The Mathieu Sine function $S(a,q,z)$.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\prime\prime} + (a - 2 q \cos(2 x)) y(x) = 0

    The other solution is the Mathieu Cosine function.

    Examples
    ========

    >>> from sympy import diff, mathieus
    >>> from sympy.abc import a, q, z

    >>> mathieus(a, q, z)
    mathieus(a, q, z)

    >>> mathieus(a, 0, z)
    sin(sqrt(a)*z)

    >>> diff(mathieus(a, q, z), z)
    mathieusprime(a, q, z)

    See Also
    ========

    mathieuc: Mathieu cosine function.
    mathieusprime: Derivative of Mathieu sine function.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    .. [4] https://functions.wolfram.com/MathieuandSpheroidalFunctions/MathieuS/
    """

    # 定义对 z 的偏导数
    def fdiff(self, argindex=1):
        if argindex == 3:
            a, q, z = self.args
            return mathieusprime(a, q, z)
        else:
            raise ArgumentIndexError(self, argindex)

    # 类方法，用于求值
    @classmethod
    def eval(cls, a, q, z):
        if q.is_Number and q.is_zero:
            return sin(sqrt(a)*z)
        # 尝试提取 -1 的因子
        if z.could_extract_minus_sign():
            return -cls(a, q, -z)

# 定义 Mathieu Cosine 函数 mathieuc
class mathieuc(MathieuBase):
    r"""
    The Mathieu Cosine function $C(a,q,z)$.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\prime\prime} + (a - 2 q \cos(2 x)) y(x) = 0

    The other solution is the Mathieu Sine function.

    Examples
    ========

    >>> from sympy import diff, mathieuc
    >>> from sympy.abc import a, q, z

    >>> mathieuc(a, q, z)
    mathieuc(a, q, z)

    >>> mathieuc(a, 0, z)
    cos(sqrt(a)*z)

    >>> diff(mathieuc(a, q, z), z)
    mathieucprime(a, q, z)

    See Also
    ========

    mathieus: Mathieu sine function
    mathieusprime: Derivative of Mathieu sine function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    """

    # 定义对 z 的偏导数
    def fdiff(self, argindex=1):
        if argindex == 3:
            a, q, z = self.args
            return mathieucprime(a, q, z)
        else:
            raise ArgumentIndexError(self, argindex)
    """
    # 定义一个 fdiff 方法，处理对象的参数索引，返回对应的求导结果
    def fdiff(self, argindex=1):
        # 如果参数索引为3，解构对象的参数并调用 MathieuC' 函数
        if argindex == 3:
            a, q, z = self.args
            return mathieucprime(a, q, z)
        # 否则，抛出参数索引错误
        else:
            raise ArgumentIndexError(self, argindex)

    # 定义一个类方法 eval，计算 MathieuC 函数的值
    @classmethod
    def eval(cls, a, q, z):
        # 如果 q 是一个数字且为零，返回 cos(sqrt(a)*z)
        if q.is_Number and q.is_zero:
            return cos(sqrt(a)*z)
        
        # 尝试提取 z 的负号因子
        if z.could_extract_minus_sign():
            # 返回一个新的 MathieuC 对象，参数为 a, q, -z
            return cls(a, q, -z)
class mathieusprime(MathieuBase):
    r"""
    The derivative $S^{\prime}(a,q,z)$ of the Mathieu Sine function.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\prime\prime} + (a - 2 q \cos(2 x)) y(x) = 0

    The other solution is the Mathieu Cosine function.

    Examples
    ========

    >>> from sympy import diff, mathieusprime
    >>> from sympy.abc import a, q, z

    >>> mathieusprime(a, q, z)
    mathieusprime(a, q, z)

    >>> mathieusprime(a, 0, z)
    sqrt(a)*cos(sqrt(a)*z)

    >>> diff(mathieusprime(a, q, z), z)
    (-a + 2*q*cos(2*z))*mathieus(a, q, z)

    See Also
    ========

    mathieus: Mathieu sine function
    mathieuc: Mathieu cosine function
    mathieucprime: Derivative of Mathieu cosine function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    .. [4] https://functions.wolfram.com/MathieuandSpheroidalFunctions/MathieuSPrime/

    """

    def fdiff(self, argindex=1):
        # 判断参数索引是否为3，即是否对 z 求导数
        if argindex == 3:
            # 解析参数 a, q, z
            a, q, z = self.args
            # 返回 Mathieu Sine 函数的导数表达式
            return (2*q*cos(2*z) - a)*mathieus(a, q, z)
        else:
            # 抛出参数索引错误
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, q, z):
        # 如果 q 是数值且为零
        if q.is_Number and q.is_zero:
            # 返回 sqrt(a)*cos(sqrt(a)*z)
            return sqrt(a)*cos(sqrt(a)*z)
        # 尝试提取负号因子
        if z.could_extract_minus_sign():
            # 返回使用相反数 z 调用类方法 cls(a, q, -z)
            return cls(a, q, -z)


class mathieucprime(MathieuBase):
    r"""
    The derivative $C^{\prime}(a,q,z)$ of the Mathieu Cosine function.

    Explanation
    ===========

    This function is one solution of the Mathieu differential equation:

    .. math ::
        y(x)^{\prime\prime} + (a - 2 q \cos(2 x)) y(x) = 0

    The other solution is the Mathieu Sine function.

    Examples
    ========

    >>> from sympy import diff, mathieucprime
    >>> from sympy.abc import a, q, z

    >>> mathieucprime(a, q, z)
    mathieucprime(a, q, z)

    >>> mathieucprime(a, 0, z)
    -sqrt(a)*sin(sqrt(a)*z)

    >>> diff(mathieucprime(a, q, z), z)
    (-a + 2*q*cos(2*z))*mathieuc(a, q, z)

    See Also
    ========

    mathieus: Mathieu sine function
    mathieuc: Mathieu cosine function
    mathieusprime: Derivative of Mathieu sine function

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathieu_function
    .. [2] https://dlmf.nist.gov/28
    .. [3] https://mathworld.wolfram.com/MathieuFunction.html
    .. [4] https://functions.wolfram.com/MathieuandSpheroidalFunctions/MathieuCPrime/

    """

    def fdiff(self, argindex=1):
        # 判断参数索引是否为3，即是否对 z 求导数
        if argindex == 3:
            # 解析参数 a, q, z
            a, q, z = self.args
            # 返回 Mathieu Cosine 函数的导数表达式
            return (2*q*cos(2*z) - a)*mathieuc(a, q, z)
        else:
            # 抛出参数索引错误
            raise ArgumentIndexError(self, argindex)

    @classmethod
    def eval(cls, a, q, z):
        # 如果 q 是数值且为零
        if q.is_Number and q.is_zero:
            # 返回 -sqrt(a)*sin(sqrt(a)*z)
            return -sqrt(a)*sin(sqrt(a)*z)
        # 尝试提取负号因子
        if z.could_extract_minus_sign():
            # 返回使用相反数 z 调用类方法 cls(a, q, -z)
            return cls(a, q, -z)
    # 定义一个类方法 `eval`，参数包括 `cls`（类本身）、`a`、`q`、`z`
    def eval(cls, a, q, z):
        # 检查 q 是否为数字并且是否为零
        if q.is_Number and q.is_zero:
            # 如果 q 是零，则返回表达式 -sqrt(a) * sin(sqrt(a) * z)
            return -sqrt(a) * sin(sqrt(a) * z)
        
        # 尝试提取 -1 的因子
        if z.could_extract_minus_sign():
            # 如果可以提取负号，则返回 -cls(a, q, -z)，其中调用类自身的构造函数
            return -cls(a, q, -z)
```