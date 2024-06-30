# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_K.py`

```
from numpy import asarray, atleast_2d, arange, sin, sqrt, prod, sum, round
from .go_benchmark import Benchmark

class Katsuura(Benchmark):
    r"""
    Katsuura objective function.

    This class defines the Katsuura [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Katsuura}}(x) = \prod_{i=0}^{n-1} \left [ 1 +
        (i+1) \sum_{k=1}^{d} \lfloor (2^k x_i) \rfloor 2^{-k} \right ]

    Where, in this exercise, :math:`d = 32`.

    Here, :math:`n` represents the number of dimensions and 
    :math:`x_i \in [0, 100]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 1` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`.

    .. [1] Adorio, E. MVF - "Multivariate Test Functions Library in C for
    Unconstrained Global Optimization", 2005
    .. [2] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO: Adorio has wrong global minimum.  Adorio uses round, Gavana docstring
    uses floor, but Gavana code uses round.  We'll use round...
    """

    def __init__(self, dimensions=2):
        # 调用父类构造函数初始化维度
        Benchmark.__init__(self, dimensions)

        # 设定变量范围
        self._bounds = list(zip([0.0] * self.N, [100.0] * self.N))

        # 设定全局最优解
        self.global_optimum = [[0.0 for _ in range(self.N)]]

        # 设定自定义范围
        self.custom_bounds = [(0, 1), (0, 1)]

        # 设定全局最优值
        self.fglob = 1.0

        # 指示可以改变维度
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 设定常量
        d = 32

        # 生成从1到d的一维数组并转置为列向量
        k = atleast_2d(arange(1, d + 1)).T

        # 生成从0到self.N-1的一维数组
        i = arange(0., self.N * 1.)

        # 计算内部表达式
        inner = round(2 ** k * x) * (2. ** (-k))

        # 返回函数值
        return prod(sum(inner, axis=0) * (i + 1) + 1)


class Keane(Benchmark):
    r"""
    Keane objective function.

    This class defines the Keane [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Keane}}(x) = \frac{\sin^2(x_1 - x_2)\sin^2(x_1 + x_2)}
        {\sqrt{x_1^2 + x_2^2}}


    with :math:`x_i \in [0, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0.0` for 
    :math:`x = [7.85396153, 7.85396135]`.

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: Jamil #69, there is no way that the function can have a negative
    value.  Everything is squared.  I think that they have the wrong solution.
    """

    def __init__(self, dimensions=2):
        # 调用父类构造函数初始化维度
        Benchmark.__init__(self, dimensions)

        # 设定变量范围
        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))

        # 设定全局最优解
        self.global_optimum = [[7.85396153, 7.85396135]]

        # 设定自定义范围
        self.custom_bounds = [(-1, 0.34), (-1, 0.34)]

        # 设定全局最优值
        self.fglob = 0.

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 计算函数值
        val = sin(x[0] - x[1]) ** 2 * sin(x[0] + x[1]) ** 2
        return val / sqrt(x[0] ** 2 + x[1] ** 2)
    """
    Kowalik objective function.

    This class defines the Kowalik global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Kowalik}}(x) = \sum_{i=0}^{10} \left [ a_i
        - \frac{x_1 (b_i^2 + b_i x_2)} {b_i^2 + b_i x_3 + x_4} \right ]^2

    Where:

    .. math::

        \begin{matrix}
        a = [4, 2, 1, 1/2, 1/4, 1/6, 1/8, 1/10, 1/12, 1/14, 1/16] \\
        b = [0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627,
             0.0456, 0.0342, 0.0323, 0.0235, 0.0246]\\
        \end{matrix}

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in 
    [-5, 5]` for :math:`i = 1, ..., 4`.

    *Global optimum*: :math:`f(x) = 0.00030748610` for :math:`x = 
    [0.192833, 0.190836, 0.123117, 0.135766]`.

    Reference:
    [1] https://www.itl.nist.gov/div898/strd/nls/data/mgh09.shtml
    """

    # 初始化函数，设置问题的维度并初始化边界、全局最优和全局最优值
    def __init__(self, dimensions=4):
        # 调用父类 Benchmark 的初始化方法
        Benchmark.__init__(self, dimensions)
        
        # 设置搜索空间的边界，每个维度都是 [-5.0, 5.0]
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        
        # 设置全局最优解，这是一个二维列表
        self.global_optimum = [[0.192833, 0.190836, 0.123117, 0.135766]]
        
        # 设置全局最优值
        self.fglob = 0.00030748610
        
        # 设置 Kowalik 函数中的系数 a 和 b
        self.a = asarray([4.0, 2.0, 1.0, 1 / 2.0, 1 / 4.0, 1 / 6.0, 1 / 8.0,
                          1 / 10.0, 1 / 12.0, 1 / 14.0, 1 / 16.0])
        self.b = asarray([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627,
                          0.0456, 0.0342, 0.0323, 0.0235, 0.0246])

    # 计算 Kowalik 函数的值，即目标函数的实现
    def fun(self, x, *args):
        # 增加函数评估次数的计数器
        self.nfev += 1

        # 计算 Kowalik 函数的向量形式
        vec = self.b - (x[0] * (self.a ** 2 + self.a * x[1])
                   / (self.a ** 2 + self.a * x[2] + x[3]))
        
        # 返回向量的平方和作为函数值
        return sum(vec ** 2)
```