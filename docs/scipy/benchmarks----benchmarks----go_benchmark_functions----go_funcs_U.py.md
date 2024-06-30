# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_U.py`

```
# 导入需要的函数和模块：abs, sin, cos, pi, sqrt
from numpy import abs, sin, cos, pi, sqrt
# 从.go_benchmark模块中导入Benchmark类
from .go_benchmark import Benchmark


class Ursem01(Benchmark):
    """
    Ursem 1 objective function.

    This class defines the Ursem 1 [1]_ global optimization problem. This is a
    unimodal minimization problem defined as follows:

    .. math::

        f_{\text{Ursem01}}(x) = - \sin(2x_1 - 0.5 \pi) - 3 \cos(x_2) - 0.5 x_1

    with :math:`x_1 \in [-2.5, 3]` and :math:`x_2 \in [-2, 2]`.

    *Global optimum*: :math:`f(x) = -4.81681406371` for
    :math:`x = [1.69714, 0.0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用Benchmark类的构造函数初始化
        Benchmark.__init__(self, dimensions)

        # 设置变量范围
        self._bounds = [(-2.5, 3.0), (-2.0, 2.0)]

        # 设置全局最优解
        self.global_optimum = [[1.69714, 0.0]]
        # 设置全局最优值
        self.fglob = -4.81681406371

    def fun(self, x, *args):
        # 增加函数评估次数
        self.nfev += 1

        # 计算目标函数值
        return -sin(2 * x[0] - 0.5 * pi) - 3.0 * cos(x[1]) - 0.5 * x[0]


class Ursem03(Benchmark):
    """
    Ursem 3 objective function.

    This class defines the Ursem 3 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Ursem03}}(x) = - \sin(2.2 \pi x_1 + 0.5 \pi) 
                                \frac{2 - \lvert x_1 \rvert}{2}
                                \frac{3 - \lvert x_1 \rvert}{2}
                                - \sin(2.2 \pi x_2 + 0.5 \pi)
                                \frac{2 - \lvert x_2 \rvert}{2}
                                \frac{3 - \lvert x_2 \rvert}{2}

    with :math:`x_1 \in [-2, 2]`, :math:`x_2 \in [-1.5, 1.5]`.

    *Global optimum*: :math:`f(x) = -3` for :math:`x = [0, 0]`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO Gavana and Jamil #157 disagree on the formulae here. Jamil squares the
    x[1] term in the sine expression. Gavana doesn't.  Go with Gavana here.
    """

    def __init__(self, dimensions=2):
        # 调用Benchmark类的构造函数初始化
        Benchmark.__init__(self, dimensions)

        # 设置变量范围
        self._bounds = [(-2, 2), (-1.5, 1.5)]

        # 设置全局最优解
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        # 设置全局最优值
        self.fglob = -3.0

    def fun(self, x, *args):
        # 增加函数评估次数
        self.nfev += 1

        # 计算目标函数值
        u = -(sin(2.2 * pi * x[0] + 0.5 * pi)
              * ((2.0 - abs(x[0])) / 2.0) * ((3.0 - abs(x[0])) / 2))
        v = -(sin(2.2 * pi * x[1] + 0.5 * pi)
              * ((2.0 - abs(x[1])) / 2) * ((3.0 - abs(x[1])) / 2))
        return u + v


class Ursem04(Benchmark):
    """
    Ursem 4 objective function.

    This class defines the Ursem 4 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Ursem04}}(x) = -3 \sin(0.5 \pi x_1 + 0.5 \pi)
                                \frac{2 - \sqrt{x_1^2 + x_2 ^ 2}}{4}

    with :math:`x_i \in [-2, 2]` for :math:`i = 1, 2`.
    """
    *Global optimum*: :math:`f(x) = -1.5` for :math:`x = [0, 0]` for
    :math:`i = 1, 2`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """



    # 定义一个类，继承自Benchmark类，用于实现一个具体的优化问题的基准测试
    def __init__(self, dimensions=2):
        # 调用父类Benchmark的初始化方法，设定问题的维度
        Benchmark.__init__(self, dimensions)

        # 设定问题变量的取值范围为[-2.0, 2.0]^N
        self._bounds = list(zip([-2.0] * self.N, [2.0] * self.N))

        # 设定全局最优解，对于该问题是一个零向量列表
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        
        # 设定全局最优解对应的函数值
        self.fglob = -1.5

    # 定义优化问题的目标函数，对给定的变量x计算函数值
    def fun(self, x, *args):
        # 记录函数调用次数
        self.nfev += 1

        # 计算目标函数的值，这里是一个二维函数的具体实现
        return (-3 * sin(0.5 * pi * x[0] + 0.5 * pi)
                * (2 - sqrt(x[0] ** 2 + x[1] ** 2)) / 4)
class UrsemWaves(Benchmark):
    r"""
    Ursem Waves objective function.

    This class defines the Ursem Waves [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{UrsemWaves}}(x) = -0.9x_1^2 + (x_2^2 - 4.5x_2^2)x_1x_2
                                   + 4.7 \cos \left[ 2x_1 - x_2^2(2 + x_1)
                                   \right ] \sin(2.5 \pi x_1)

    with :math:`x_1 \in [-0.9, 1.2]`, :math:`x_2 \in [-1.2, 1.2]`.

    *Global optimum*: :math:`f(x) = -8.5536` for :math:`x = [1.2, 1.2]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Jamil #159, has an x_2^2 - 4.5 x_2^2 in the brackets. Why wasn't this
    rationalised to -5.5 x_2^2? This makes me wonder if the equation  is listed
    correctly?
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设定维度
        Benchmark.__init__(self, dimensions)

        # 定义变量范围
        self._bounds = [(-0.9, 1.2), (-1.2, 1.2)]

        # 设置全局最优解和对应的函数值
        self.global_optimum = [[1.2 for _ in range(self.N)]]
        self.fglob = -8.5536

    def fun(self, x, *args):
        # 增加函数评估计数器
        self.nfev += 1

        # 计算函数的三部分
        u = -0.9 * x[0] ** 2
        v = (x[1] ** 2 - 4.5 * x[1] ** 2) * x[0] * x[1]
        w = 4.7 * cos(3 * x[0] - x[1] ** 2 * (2 + x[0])) * sin(2.5 * pi * x[0])
        
        # 返回函数值
        return u + v + w
```