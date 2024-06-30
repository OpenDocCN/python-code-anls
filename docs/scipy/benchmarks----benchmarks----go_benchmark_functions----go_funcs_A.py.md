# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_A.py`

```
from numpy import abs, cos, exp, pi, prod, sin, sqrt, sum  # 导入所需的数学函数
from .go_benchmark import Benchmark  # 导入自定义的基准类 Benchmark


class Ackley01(Benchmark):

    r"""
    Ackley01 objective function.

    The Ackley01 [1]_ global optimization problem is a multimodal minimization
    problem defined as follows:

    .. math::

        f_{\text{Ackley01}}(x) = -20 e^{-0.2 \sqrt{\frac{1}{n} \sum_{i=1}^n
         x_i^2}} - e^{\frac{1}{n} \sum_{i=1}^n \cos(2 \pi x_i)} + 20 + e


    Here, :math:`n` represents the number of dimensions and :math:`x_i \in
    [-35, 35]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Adorio, E. MVF - "Multivariate Test Functions Library in C for
    Unconstrained Global Optimization", 2005

    TODO: the -0.2 factor in the exponent of the first term is given as
    -0.02 in Jamil et al.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)  # 调用父类 Benchmark 的初始化方法

        self._bounds = list(zip([-35.0] * self.N, [35.0] * self.N))  # 定义变量范围边界
        self.global_optimum = [[0 for _ in range(self.N)]]  # 设定全局最优解
        self.fglob = 0.0  # 初始化全局最小值
        self.change_dimensionality = True  # 设定维度变化标志为真

    def fun(self, x, *args):
        self.nfev += 1  # 增加函数评估计数器
        u = sum(x ** 2)  # 计算平方和 u
        v = sum(cos(2 * pi * x))  # 计算余弦和 v
        return (-20. * exp(-0.2 * sqrt(u / self.N))  # 返回 Ackley01 函数的值
                - exp(v / self.N) + 20. + exp(1.))


class Ackley02(Benchmark):

    r"""
    Ackley02 objective function.

    The Ackley02 [1]_ global optimization problem is a multimodal minimization
    problem defined as follows:

    .. math::

        f_{\text{Ackley02}(x) = -200 e^{-0.02 \sqrt{x_1^2 + x_2^2}}


    with :math:`x_i \in [-32, 32]` for :math:`i=1, 2`.

    *Global optimum*: :math:`f(x) = -200` for :math:`x = [0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    """
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)  # 调用父类 Benchmark 的初始化方法

        self._bounds = list(zip([-32.0] * self.N, [32.0] * self.N))  # 定义变量范围边界
        self.global_optimum = [[0 for _ in range(self.N)]]  # 设定全局最优解
        self.fglob = -200.  # 设定全局最小值为 -200

    def fun(self, x, *args):
        self.nfev += 1  # 增加函数评估计数器
        return -200 * exp(-0.02 * sqrt(x[0] ** 2 + x[1] ** 2))  # 返回 Ackley02 函数的值


class Ackley03(Benchmark):

    r"""
    Ackley03 [1]_ objective function.

    The Ackley03 global optimization problem is a multimodal minimization
    problem defined as follows:

    .. math::

        f_{\text{Ackley03}}(x) = -200 e^{-0.02 \sqrt{x_1^2 + x_2^2}} +
            5e^{\cos(3x_1) + \sin(3x_2)}


    with :math:`x_i \in [-32, 32]` for :math:`i=1, 2`.

    *Global optimum*: :math:`f(x) = -195.62902825923879` for :math:`x
    = [-0.68255758, -0.36070859]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling

    """
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)  # 调用父类 Benchmark 的初始化方法

        self._bounds = list(zip([-32.0] * self.N, [32.0] * self.N))  # 定义变量范围边界
        self.global_optimum = [[-0.68255758, -0.36070859]]  # 设定全局最优解
        self.fglob = -195.62902825923879  # 设定全局最小值为 -195.62902825923879
    """
    Class implementing the Eggholder function benchmark.

    This function is derived from [1]_ and is a multimodal function that presents
    challenges to optimization algorithms due to its multiple local minima.

    Parameters
    ----------
    dimensions : int, optional
        Number of dimensions of the problem (default is 2).

    Attributes
    ----------
    _bounds : list of tuples
        List containing lower and upper bounds for each dimension of the problem.
    global_optimum : list
        Coordinates of the global optimum point.
    fglob : float
        Value of the function at the global optimum.

    Methods
    -------
    fun(x, *args)
        Compute the value of the Eggholder function at point x.

    Notes
    -----
    This function is defined as:

        f(x1, x2) = - (a + b)

    where:
        a = 200 * exp(-0.02 * sqrt(x1^2 + x2^2))
        b = 5 * exp(cos(3 * x1) + sin(3 * x2))

    References
    ----------
    .. [1] A. Hedar, Y. Fukushima, "Tabu Search technique for unconstrained
           and Numerical Optimisation, 2013, 4, 150-194.

    TODO: I think the minus sign is missing in front of the first term in eqn3
      in [1]_.  This changes the global minimum
    """

    def __init__(self, dimensions=2):
        # Initialize the Benchmark superclass with the specified dimensions
        Benchmark.__init__(self, dimensions)

        # Define the bounds for each dimension of the problem
        self._bounds = list(zip([-32.0] * self.N, [32.0] * self.N))
        
        # Define the coordinates of the global optimum point
        self.global_optimum = [[-0.68255758, -0.36070859]]
        
        # Define the value of the function at the global optimum
        self.fglob = -195.62902825923879

    def fun(self, x, *args):
        # Increment the function evaluation counter
        self.nfev += 1
        
        # Compute the value of the Eggholder function at point x
        a = -200 * exp(-0.02 * sqrt(x[0] ** 2 + x[1] ** 2))
        a += 5 * exp(cos(3 * x[0]) + sin(3 * x[1]))
        return a
class Adjiman(Benchmark):
    r"""
    Adjiman objective function.

    The Adjiman [1]_ global optimization problem is a multimodal minimization
    problem defined as follows:

    .. math::

        f_{\text{Adjiman}}(x) = \cos(x_1)\sin(x_2) - \frac{x_1}{(x_2^2 + 1)}

    with, :math:`x_1 \in [-1, 2]` and :math:`x_2 \in [-1, 1]`.

    *Global optimum*: :math:`f(x) = -2.02181` for :math:`x = [2.0, 0.10578]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用父类构造函数初始化维度
        Benchmark.__init__(self, dimensions)

        # 设定变量的取值范围
        self._bounds = ([-1.0, 2.0], [-1.0, 1.0])
        # 设定全局最优解
        self.global_optimum = [[2.0, 0.10578]]
        # 设定全局最优解的函数值
        self.fglob = -2.02180678

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1
        # 返回 Adjiman 函数的值
        return cos(x[0]) * sin(x[1]) - x[0] / (x[1] ** 2 + 1)


class Alpine01(Benchmark):
    r"""
    Alpine01 objective function.

    The Alpine01 [1]_ global optimization problem is a multimodal minimization
    problem defined as follows:

    .. math::

        f_{\text{Alpine01}}(x) = \sum_{i=1}^{n} \lvert {x_i \sin \left( x_i
        \right) + 0.1 x_i} \rvert

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in
    [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用父类构造函数初始化维度
        Benchmark.__init__(self, dimensions)

        # 设定变量的取值范围
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        # 设定全局最优解
        self.global_optimum = [[0 for _ in range(self.N)]]
        # 设定全局最优解的函数值
        self.fglob = 0.0
        # 设置维度变化标志为真
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 返回 Alpine01 函数的值
        return sum(abs(x * sin(x) + 0.1 * x))


class Alpine02(Benchmark):
    r"""
    Alpine02 objective function.

    The Alpine02 [1]_ global optimization problem is a multimodal minimization
    problem defined as follows:

    .. math::

        f_{\text{Alpine02}}(x) = \prod_{i=1}^{n} \sqrt{x_i} \sin(x_i)

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [0,
    10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -6.1295` for :math:`x =
    [7.91705268, 4.81584232]` for :math:`i = 1, 2`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: eqn 7 in [1]_ has the wrong global minimum value.
    """
    # 初始化函数，用于设置对象的初始状态，dimensions 参数默认为 2
    def __init__(self, dimensions=2):
        # 调用 Benchmark 类的初始化方法，传入 dimensions 参数
        Benchmark.__init__(self, dimensions)

        # 设置对象的边界属性 _bounds，以元组形式表示每个维度的取值范围
        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))
        
        # 设置全局最优解的初始值，一个包含一个二维坐标的列表
        self.global_optimum = [[7.91705268, 4.81584232]]
        
        # 设置全局最优值的初始值
        self.fglob = -6.12950
        
        # 设置改变维度的标志，默认为 True，表示支持改变维度
        self.change_dimensionality = True

    # 函数 fun，用于计算给定向量 x 的函数值，并更新函数评估次数 nfev
    def fun(self, x, *args):
        # 增加函数评估次数计数器 nfev
        self.nfev += 1

        # 返回 x 各元素平方根与正弦函数值的乘积的乘积（prod 函数）
        return prod(sqrt(x) * sin(x))
class AMGM(Benchmark):

    r"""
    AMGM objective function.

    The AMGM (Arithmetic Mean - Geometric Mean Equality) global optimization
    problem is a multimodal minimization problem defined as follows

    .. math::

        f_{\text{AMGM}}(x) = \left ( \frac{1}{n} \sum_{i=1}^{n} x_i -
         \sqrt[n]{ \prod_{i=1}^{n} x_i} \right )^2


    Here, :math:`n` represents the number of dimensions and :math:`x_i \in
    [0, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_1 = x_2 = ... = x_n` for
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO, retrieved 2015

    TODO: eqn 7 in [1]_ has the wrong global minimum value.
    """

    def __init__(self, dimensions=2):
        # 调用 Benchmark 类的初始化方法，设定维度
        Benchmark.__init__(self, dimensions)

        # 设定变量的上下界
        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))

        # 设定全局最优解和对应的函数值
        self.global_optimum = [[1, 1]]
        self.fglob = 0.0

        # 标记维度变化的状态
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估的次数计数
        self.nfev += 1

        # 计算 f1 和 f2 的值
        f1 = sum(x)  # 计算向量 x 各元素的和
        f2 = prod(x)  # 计算向量 x 各元素的乘积（prod 函数未明确给出，可能指向 numpy.prod）

        # 计算 AMGM 函数值
        f1 = f1 / self.N  # 计算算术平均值
        f2 = f2 ** (1.0 / self.N)  # 计算几何平均值
        f = (f1 - f2) ** 2  # 计算 AMGM 函数的平方

        return f
```