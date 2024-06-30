# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_X.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
from numpy import abs, sum, sin, cos, pi, exp, arange, prod, sqrt  # 从 NumPy 中导入一些数学函数

from .go_benchmark import Benchmark  # 从当前包的 go_benchmark 模块导入 Benchmark 类


class XinSheYang01(Benchmark):

    r"""
    Xin-She Yang 1 objective function.

    This class defines the Xin-She Yang 1 [1]_ global optimization problem.
    This is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{XinSheYang01}}(x) = \sum_{i=1}^{n} \epsilon_i \lvert x_i 
                                     \rvert^i

    The variable :math:`\epsilon_i, (i = 1, ..., n)` is a random variable
    uniformly distributed in :math:`[0, 1]`.

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-5, 5]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)  # 调用父类 Benchmark 的初始化方法，设定维度

        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))  # 设置变量边界为 [-5, 5] 的列表
        self.custom_bounds = ([-2, 2], [-2, 2])  # 自定义边界为 [-2, 2] 的元组列表

        self.global_optimum = [[0 for _ in range(self.N)]]  # 设置全局最优解为维度为 self.N 的零向量
        self.fglob = 0.0  # 设置全局最优解的函数值为 0.0
        self.change_dimensionality = True  # 标志位，表示可以改变维度

    def fun(self, x, *args):
        self.nfev += 1  # 计算函数调用次数加一

        i = arange(1.0, self.N + 1.0)  # 创建一个从 1 到 self.N 的数组
        return sum(np.random.random(self.N) * (abs(x) ** i))  # 返回随机数组乘以 x 的绝对值的幂的和


class XinSheYang02(Benchmark):

    r"""
    Xin-She Yang 2 objective function.

    This class defines the Xin-She Yang 2 [1]_ global optimization problem.
    This is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{XinSheYang02}}(\x) = \frac{\sum_{i=1}^{n} \lvert{x_{i}}\rvert}
                                      {e^{\sum_{i=1}^{n} \sin\left(x_{i}^{2.0}
                                      \right)}}

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-2\pi, 2\pi]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)  # 调用父类 Benchmark 的初始化方法，设定维度

        self._bounds = list(zip([-2 * pi] * self.N,
                           [2 * pi] * self.N))  # 设置变量边界为 [-2π, 2π] 的列表

        self.global_optimum = [[0 for _ in range(self.N)]]  # 设置全局最优解为维度为 self.N 的零向量
        self.fglob = 0.0  # 设置全局最优解的函数值为 0.0
        self.change_dimensionality = True  # 标志位，表示可以改变维度

    def fun(self, x, *args):
        self.nfev += 1  # 计算函数调用次数加一

        return sum(abs(x)) * exp(-sum(sin(x ** 2.0)))  # 返回绝对值和的指数乘积的和的负值


class XinSheYang03(Benchmark):

    r"""
    Xin-She Yang 3 objective function.

    This class defines the Xin-She Yang 3 [1]_ global optimization problem.
    This is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{XinSheYang03}}(x) = e^{-\sum_{i=1}^{n} (x_i/\beta)^{2m}}
                                     - 2e^{-\sum_{i=1}^{n} x_i^2}
                                     \prod_{i=1}^{n} \cos^2(x_i)


    Where, in this exercise, :math:`\beta = 15` and :math:`m = 3`.

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-20, 20]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -1` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    # 定义一个 Benchmark 类的子类，代表 XinSheYang03 函数的实现
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的构造函数并传入维度参数
        Benchmark.__init__(self, dimensions)

        # 设定变量的取值范围，对每个维度都是 [-20.0, 20.0]
        self._bounds = list(zip([-20.0] * self.N, [20.0] * self.N))

        # 设定全局最优解，即每个维度上都是 0 的点
        self.global_optimum = [[0 for _ in range(self.N)]]
        
        # 设定全局最优解对应的函数值
        self.fglob = -1.0
        
        # 标志表示是否改变维度
        self.change_dimensionality = True

    # 定义评估函数 fun，计算 XinSheYang03 函数在给定参数 x 上的取值
    def fun(self, x, *args):
        # 增加函数评估计数器
        self.nfev += 1

        # 设定参数 beta 和 m
        beta, m = 15.0, 5.0
        
        # 计算 u、v、w 三个中间变量
        u = sum((x / beta) ** (2 * m))
        v = sum(x ** 2)
        w = prod(cos(x) ** 2)

        # 计算并返回 XinSheYang03 函数的值
        return exp(-u) - 2 * exp(-v) * w
class XinSheYang04(Benchmark):
    """
    Xin-She Yang 4 objective function.

    This class defines the Xin-She Yang 4 [1]_ global optimization problem.
    This is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{XinSheYang04}}(x) = \left[ \sum_{i=1}^{n} \sin^2(x_i)
                                     - e^{-\sum_{i=1}^{n} x_i^2} \right ]
                                     e^{-\sum_{i=1}^{n} \sin^2 \sqrt{ \lvert
                                     x_i \rvert }}

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -1` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用父类构造函数，设置维度
        Benchmark.__init__(self, dimensions)

        # 设置变量的取值范围
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))

        # 设置全局最优解
        self.global_optimum = [[0 for _ in range(self.N)]]
        
        # 设置全局最优解的函数值
        self.fglob = -1.0
        
        # 标记维度是否改变
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 计算函数中的三个部分
        u = sum(sin(x) ** 2)
        v = sum(x ** 2)
        w = sum(sin(sqrt(abs(x))) ** 2)
        
        # 返回函数值
        return (u - exp(-v)) * exp(-w)


class Xor(Benchmark):
    """
    Xor objective function.

    This class defines the Xor [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Xor}}(x) = \left[ 1 + \exp \left( - \frac{x_7}{1 +
        \exp(-x_1 - x_2 - x_5)} - \frac{x_8}{1 + \exp(-x_3 - x_4 - x_6)}
        - x_9 \right ) \right ]^{-2} \\
        + \left [ 1 + \exp \left( -\frac{x_7}{1 + \exp(-x_5)}
        - \frac{x_8}{1 + \exp(-x_6)} - x_9 \right ) \right] ^{-2} \\
        + \left [1 - \left\{1 + \exp \left(-\frac{x_7}{1 + \exp(-x_1 - x_5)}
        - \frac{x_8}{1 + \exp(-x_3 - x_6)} - x_9 \right ) \right\}^{-1}
        \right ]^2 \\
        + \left [1 - \left\{1 + \exp \left(-\frac{x_7}{1 + \exp(-x_2 - x_5)}
        - \frac{x_8}{1 + \exp(-x_4 - x_6)} - x_9 \right ) \right\}^{-1}
        \right ]^2


    with :math:`x_i \in [-1, 1]` for :math:`i=1,...,9`.

    *Global optimum*: :math:`f(x) = 0.9597588` for
    :math:`\x = [1, -1, 1, -1, -1, 1, 1, -1, 0.421134]`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=9):
        # 调用父类构造函数，设置维度
        Benchmark.__init__(self, dimensions)

        # 设置变量的取值范围
        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))

        # 设置全局最优解
        self.global_optimum = [[1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 0.421134]]
        
        # 设置全局最优解的函数值
        self.fglob = 0.9597588
    # 定义一个方法 `fun`，接受参数 `self`（指向当前对象），`x`（一个列表），`*args`（可变数量的额外参数）
    def fun(self, x, *args):
        # 增加函数评估计数器
        self.nfev += 1

        # 计算 F11
        F11 = x[6] / (1.0 + exp(-x[0] - x[1] - x[4]))
        # 计算 F12
        F12 = x[7] / (1.0 + exp(-x[2] - x[3] - x[5]))
        # 计算 F1
        F1 = (1.0 + exp(-F11 - F12 - x[8])) ** (-2)

        # 计算 F21
        F21 = x[6] / (1.0 + exp(-x[4]))
        # 计算 F22
        F22 = x[7] / (1.0 + exp(-x[5]))
        # 计算 F2
        F2 = (1.0 + exp(-F21 - F22 - x[8])) ** (-2)

        # 计算 F31
        F31 = x[6] / (1.0 + exp(-x[0] - x[4]))
        # 计算 F32
        F32 = x[7] / (1.0 + exp(-x[2] - x[5]))
        # 计算 F3
        F3 = (1.0 - (1.0 + exp(-F31 - F32 - x[8])) ** (-1)) ** 2

        # 计算 F41
        F41 = x[6] / (1.0 + exp(-x[1] - x[4]))
        # 计算 F42
        F42 = x[7] / (1.0 + exp(-x[3] - x[5]))
        # 计算 F4
        F4 = (1.0 - (1.0 + exp(-F41 - F42 - x[8])) ** (-1)) ** 2

        # 返回计算得到的结果 F1 + F2 + F3 + F4
        return F1 + F2 + F3 + F4
```