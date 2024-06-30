# `D:\src\scipysrc\sympy\sympy\holonomic\holonomicerrors.py`

```
""" Common Exceptions for `holonomic` module. """

# 定义基本的 Holonomic 模块异常类
class BaseHolonomicError(Exception):

    # 定义 new 方法，但未实现，抛出未实现错误
    def new(self, *args):
        raise NotImplementedError("abstract base class")

# 定义非幂级数异常类，继承自 BaseHolonomicError
class NotPowerSeriesError(BaseHolonomicError):

    # 初始化方法，接受 holonomic 对象和 x0 参数
    def __init__(self, holonomic, x0):
        self.holonomic = holonomic
        self.x0 = x0

    # 字符串表示方法，描述不具有幂级数的错误
    def __str__(self):
        s = 'A Power Series does not exists for '
        s += str(self.holonomic)
        s += ' about %s.' % self.x0
        return s

# 定义非 Holonomic 错误类，继承自 BaseHolonomicError
class NotHolonomicError(BaseHolonomicError):

    # 初始化方法，接受 m 参数
    def __init__(self, m):
        self.m = m

    # 字符串表示方法，返回异常信息
    def __str__(self):
        return self.m

# 定义奇异性错误类，继承自 BaseHolonomicError
class SingularityError(BaseHolonomicError):

    # 初始化方法，接受 holonomic 对象和 x0 参数
    def __init__(self, holonomic, x0):
        self.holonomic = holonomic
        self.x0 = x0

    # 字符串表示方法，描述在 x0 处存在奇异性的错误
    def __str__(self):
        s = str(self.holonomic)
        s += ' has a singularity at %s.' % self.x0
        return s

# 定义非超几何级数异常类，继承自 BaseHolonomicError
class NotHyperSeriesError(BaseHolonomicError):

    # 初始化方法，接受 holonomic 对象和 x0 参数
    def __init__(self, holonomic, x0):
        self.holonomic = holonomic
        self.x0 = x0

    # 字符串表示方法，描述在 x0 处不是超几何级数的错误
    def __str__(self):
        s = 'Power series expansion of '
        s += str(self.holonomic)
        s += ' about %s is not hypergeometric' % self.x0
        return s
```