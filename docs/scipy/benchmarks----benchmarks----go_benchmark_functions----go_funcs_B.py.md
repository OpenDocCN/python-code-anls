# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_B.py`

```
# 导入必要的数学函数和数据结构
from numpy import abs, cos, exp, log, arange, pi, sin, sqrt, sum
from .go_benchmark import Benchmark

# 创建 BartelsConn 类，继承 Benchmark 类
class BartelsConn(Benchmark):

    r"""
    Bartels-Conn objective function.

    The BartelsConn [1]_ global optimization problem is a multimodal
    minimization problem defined as follows:

    .. math::

        f_{\text{BartelsConn}}(x) = \lvert {x_1^2 + x_2^2 + x_1x_2} \rvert +
         \lvert {\sin(x_1)} \rvert + \lvert {\cos(x_2)} \rvert


    with :math:`x_i \in [-500, 500]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 1` for :math:`x = [0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    """

    # 初始化函数，设置问题维度，默认为2维
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置问题的搜索范围边界
        self._bounds = list(zip([-500.] * self.N, [500.] * self.N))
        
        # 设置全局最优解，即使当前问题的全局最优解
        self.global_optimum = [[0 for _ in range(self.N)]]
        
        # 设置全局最优解对应的函数值
        self.fglob = 1.0

    # 定义目标函数 fun，计算 BartelsConn 函数在给定点 x 处的函数值
    def fun(self, x, *args):
        # 每调用一次 fun 函数，增加一次函数评估次数计数器
        self.nfev += 1

        # 计算 BartelsConn 函数的值
        return (abs(x[0] ** 2.0 + x[1] ** 2.0 + x[0] * x[1]) + abs(sin(x[0]))
                + abs(cos(x[1])))


# 创建 Beale 类，继承 Benchmark 类
class Beale(Benchmark):

    r"""
    Beale objective function.

    The Beale [1]_ global optimization problem is a multimodal
    minimization problem defined as follows:

    .. math::

        f_{\text{Beale}}(x) = \left(x_1 x_2 - x_1 + 1.5\right)^{2} +
        \left(x_1 x_2^{2} - x_1 + 2.25\right)^{2} + \left(x_1 x_2^{3} - x_1 +
        2.625\right)^{2}


    with :math:`x_i \in [-4.5, 4.5]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x=[3, 0.5]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    # 初始化函数，设置问题维度，默认为2维
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置问题的搜索范围边界
        self._bounds = list(zip([-4.5] * self.N, [4.5] * self.N))
        
        # 设置全局最优解，即 Beale 函数的全局最优解
        self.global_optimum = [[3.0, 0.5]]
        
        # 设置全局最优解对应的函数值
        self.fglob = 0.0

    # 定义目标函数 fun，计算 Beale 函数在给定点 x 处的函数值
    def fun(self, x, *args):
        # 每调用一次 fun 函数，增加一次函数评估次数计数器
        self.nfev += 1

        # 计算 Beale 函数的值
        return ((1.5 - x[0] + x[0] * x[1]) ** 2
                + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
                + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2)


# 创建 BiggsExp02 类，继承 Benchmark 类
class BiggsExp02(Benchmark):

    r"""
    BiggsExp02 objective function.

    The BiggsExp02 [1]_ global optimization problem is a multimodal minimization
    problem defined as follows

    .. math::

        \begin{matrix}
        f_{\text{BiggsExp02}}(x) = \sum_{i=1}^{10} (e^{-t_i x_1}
           - 5 e^{-t_i x_2} - y_i)^2 \\
        t_i = 0.1 i\\
        y_i = e^{-t_i} - 5 e^{-10t_i}\\
        \end{matrix}


    with :math:`x_i \in [0, 20]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 10]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions

    """

    # 初始化函数，设置问题维度，默认为2维
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置问题的搜索范围边界
        self._bounds = list(zip([0.] * self.N, [20.] * self.N))
        
        # 设置全局最优解，即 BiggsExp02 函数的全局最优解
        self.global_optimum = [[1.0, 10.0]]
        
        # 设置全局最优解对应的函数值
        self.fglob = 0.0
    """
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    # 定义一个 Benchmark 类的子类，用于处理特定维度的优化问题
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设置维度
        Benchmark.__init__(self, dimensions)

        # 设置问题的搜索空间边界，这里是一个二维空间范围
        self._bounds = list(zip([0] * 2, [20] * 2))

        # 设置全局最优解，这里是一个二维列表
        self.global_optimum = [[1., 10.]]

        # 初始化全局最优值
        self.fglob = 0

    # 定义求解目标函数的方法
    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 定义 t 的取值范围
        t = arange(1, 11.) * 0.1

        # 计算目标函数的值
        y = exp(-t) - 5 * exp(-10 * t)
        vec = (exp(-t * x[0]) - 5 * exp(-t * x[1]) - y) ** 2

        # 返回目标函数值的总和作为结果
        return sum(vec)
class BiggsExp03(Benchmark):
    r"""
    BiggsExp03 objective function.

    The BiggsExp03 [1]_ global optimization problem is a multimodal minimization
    problem defined as follows

    .. math::

        \begin{matrix}\ f_{\text{BiggsExp03}}(x) = \sum_{i=1}^{10}
        (e^{-t_i x_1} - x_3e^{-t_i x_2} - y_i)^2\\
        t_i = 0.1i\\
        y_i = e^{-t_i} - 5e^{-10 t_i}\\
        \end{matrix}


    with :math:`x_i \in [0, 20]` for :math:`i = 1, 2, 3`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 10, 5]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    """

    def __init__(self, dimensions=3):
        # 初始化Benchmark父类，设定维度
        Benchmark.__init__(self, dimensions)

        # 设定变量边界
        self._bounds = list(zip([0] * 3,
                                [20] * 3))
        # 设定全局最优解
        self.global_optimum = [[1., 10., 5.]]
        # 设定全局最优函数值
        self.fglob = 0

    def fun(self, x, *args):
        # 增加评估函数调用次数计数
        self.nfev += 1

        # 生成 t 数组
        t = arange(1., 11.) * 0.1
        # 计算 y 数组
        y = exp(-t) - 5 * exp(-10 * t)
        # 计算目标函数向量
        vec = (exp(-t * x[0]) - x[2] * exp(-t * x[1]) - y) ** 2

        # 返回目标函数值
        return sum(vec)


class BiggsExp04(Benchmark):
    r"""
    BiggsExp04 objective function.

    The BiggsExp04 [1]_ global optimization problem is a multimodal
    minimization problem defined as follows

    .. math::

        \begin{matrix}\ f_{\text{BiggsExp04}}(x) = \sum_{i=1}^{10}
        (x_3 e^{-t_i x_1} - x_4 e^{-t_i x_2} - y_i)^2\\
        t_i = 0.1i\\
        y_i = e^{-t_i} - 5 e^{-10 t_i}\\
        \end{matrix}


    with :math:`x_i \in [0, 20]` for :math:`i = 1, ..., 4`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 10, 1, 5]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    """

    def __init__(self, dimensions=4):
        # 初始化Benchmark父类，设定维度
        Benchmark.__init__(self, dimensions)

        # 设定变量边界
        self._bounds = list(zip([0.] * 4,
                                [20.] * 4))
        # 设定全局最优解
        self.global_optimum = [[1., 10., 1., 5.]]
        # 设定全局最优函数值
        self.fglob = 0

    def fun(self, x, *args):
        # 增加评估函数调用次数计数
        self.nfev += 1

        # 生成 t 数组
        t = arange(1, 11.) * 0.1
        # 计算 y 数组
        y = exp(-t) - 5 * exp(-10 * t)
        # 计算目标函数向量
        vec = (x[2] * exp(-t * x[0]) - x[3] * exp(-t * x[1]) - y) ** 2

        # 返回目标函数值
        return sum(vec)


class BiggsExp05(Benchmark):
    r"""
    BiggsExp05 objective function.

    The BiggsExp05 [1]_ global optimization problem is a multimodal minimization
    problem defined as follows

    .. math::

        \begin{matrix}\ f_{\text{BiggsExp05}}(x) = \sum_{i=1}^{11}
        (x_3 e^{-t_i x_1} - x_4 e^{-t_i x_2} + 3 e^{-t_i x_5} - y_i)^2\\
        t_i = 0.1i\\
        y_i = e^{-t_i} - 5e^{-10 t_i} + 3e^{-4 t_i}\\
        \end{matrix}


    with :math:`x_i \in [0, 20]` for :math:`i=1, ..., 5`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 10, 1, 5, 4]`
    """
    [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """
    
    class HockSchittkowski(Benchmark):
        """
        初始化 HockSchittkowski 类，设置维度默认为 5。
    
        Parameters:
        dimensions (int): 控制优化问题的维度，默认为 5
    
        Notes:
        - 继承自 Benchmark 类
    
        """
        def __init__(self, dimensions=5):
            # 调用父类 Benchmark 的初始化方法，设定维度
            Benchmark.__init__(self, dimensions)
    
            # 设置变量边界，每个维度的范围从 0 到 20
            self._bounds = list(zip([0.] * 5,
                                    [20.] * 5))
            # 设置全局最优解，一个包含 5 个值的列表
            self.global_optimum = [[1., 10., 1., 5., 4.]]
            # 初始化全局最优值为 0
            self.fglob = 0
    
        def fun(self, x, *args):
            # 增加函数评估计数
            self.nfev += 1
            # 生成 t 数组，范围从 0.1 到 1.1，步长为 0.1
            t = arange(1, 12.) * 0.1
            # 计算函数 y 的值
            y = exp(-t) - 5 * exp(-10 * t) + 3 * exp(-4 * t)
            # 计算向量 vec 的值，根据给定的公式
            vec = (x[2] * exp(-t * x[0]) - x[3] * exp(-t * x[1])
                   + 3 * exp(-t * x[4]) - y) ** 2
    
            # 返回向量 vec 的总和作为函数值
            return sum(vec)
class Bird(Benchmark):
    
    r"""
    Bird objective function.

    The Bird global optimization problem is a multimodal minimization
    problem defined as follows

    .. math::

        f_{\text{Bird}}(x) = \left(x_1 - x_2\right)^{2} + e^{\left[1 -
         \sin\left(x_1\right) \right]^{2}} \cos\left(x_2\right) + e^{\left[1 -
          \cos\left(x_2\right)\right]^{2}} \sin\left(x_1\right)

    with :math:`x_i \in [-2\pi, 2\pi]`

    *Global optimum*: :math:`f(x) = -106.7645367198034` for :math:`x
    = [4.701055751981055, 3.152946019601391]` or :math:`x =
    [-1.582142172055011, -3.130246799635430]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设置维度
        Benchmark.__init__(self, dimensions)

        # 定义变量的上下界
        self._bounds = list(zip([-2.0 * pi] * self.N,
                                [2.0 * pi] * self.N))
        
        # 设定全局最优解列表
        self.global_optimum = [[4.701055751981055, 3.152946019601391],
                               [-1.582142172055011, -3.130246799635430]]
        
        # 设定全局最优解对应的函数值
        self.fglob = -106.7645367198034

    def fun(self, x, *args):
        # 每次调用该函数，增加计数器 nfev 的值
        self.nfev += 1
        
        # 计算 Bird 函数的值并返回
        return (sin(x[0]) * exp((1 - cos(x[1])) ** 2)
                + cos(x[1]) * exp((1 - sin(x[0])) ** 2) + (x[0] - x[1]) ** 2)


class Bohachevsky1(Benchmark):
    
    r"""
    Bohachevsky 1 objective function.

    The Bohachevsky 1 [1]_ global optimization problem is a multimodal
    minimization problem defined as follows

        .. math::

        f_{\text{Bohachevsky}}(x) = \sum_{i=1}^{n-1}\left[x_i^2 + 2 x_{i+1}^2 -
        0.3 \cos(3 \pi x_i) - 0.4 \cos(4 \pi x_{i + 1}) + 0.7 \right]

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in
    [-15, 15]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for :math:`i = 1,
    ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: equation needs to be fixed up in the docstring. see Jamil#17
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设置维度
        Benchmark.__init__(self, dimensions)

        # 定义变量的上下界
        self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))
        
        # 设定全局最优解列表，所有维度为0
        self.global_optimum = [[0 for _ in range(self.N)]]
        
        # 设定全局最优解对应的函数值为0
        self.fglob = 0.0

    def fun(self, x, *args):
        # 每次调用该函数，增加计数器 nfev 的值
        self.nfev += 1
        
        # 计算 Bohachevsky 1 函数的值并返回
        return (x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * cos(3 * pi * x[0])
                - 0.4 * cos(4 * pi * x[1]) + 0.7)


class Bohachevsky2(Benchmark):
    
    r"""
    Bohachevsky 2 objective function.

    The Bohachevsky 2 [1]_ global optimization problem is a multimodal
    # 定义 Bohachevsky 函数作为最小化问题

        .. math::

        f_{\text{Bohachevsky}}(x) = \sum_{i=1}^{n-1}\left[x_i^2 + 2 x_{i+1}^2 -
        0.3 \cos(3 \pi x_i) - 0.4 \cos(4 \pi x_{i + 1}) + 0.7 \right]


    这里，:math:`n` 表示维度的数量，:math:`x_i \in
    [-15, 15]` 对于 :math:`i = 1, ..., n`。

    *全局最优解*: :math:`f(x) = 0` 对于 :math:`x_i = 0`，对于 :math:`i = 1,
    ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: 文档字符串中的方程需要修正。Jamil 的论点也是错误的。
    在 cos 项前面不应该有 0.4 的因子
    """
    
    # 定义 Bohachevsky 函数类
    def __init__(self, dimensions=2):
        # 调用 Benchmark 类的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置变量边界为 [-100.0, 100.0] 的列表
        self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))
        # 设置全局最优解为所有维度为 0 的列表
        self.global_optimum = [[0 for _ in range(self.N)]]
        # 设置全局最优值为 0.0
        self.fglob = 0.0

    # 定义 Bohachevsky 函数的计算方法
    def fun(self, x, *args):
        # 增加函数评估次数的计数器
        self.nfev += 1

        # 返回 Bohachevsky 函数的计算结果
        return (x[0] ** 2 + 2 * x[1] ** 2 - 0.3 * cos(3 * pi * x[0])
                 * cos(4 * pi * x[1]) + 0.3)
class Bohachevsky3(Benchmark):
    
    r"""
    Bohachevsky 3 objective function.

    The Bohachevsky 3 [1]_ global optimization problem is a multimodal
    minimization problem defined as follows

        .. math::

        f_{\text{Bohachevsky}}(x) = \sum_{i=1}^{n-1}\left[x_i^2 + 2 x_{i+1}^2 -
        0.3 \cos(3 \pi x_i) - 0.4 \cos(4 \pi x_{i + 1}) + 0.7 \right]


    Here, :math:`n` represents the number of dimensions and :math:`x_i \in
    [-15, 15]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for :math:`i = 1,
    ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: equation needs to be fixed up in the docstring. Jamil#19
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法
        Benchmark.__init__(self, dimensions)

        # 定义搜索空间边界
        self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))
        
        # 设置全局最优解
        self.global_optimum = [[0 for _ in range(self.N)]]
        
        # 设置全局最优值
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估次数
        self.nfev += 1
        
        # 计算 Bohachevsky 3 函数值
        return (x[0] ** 2 + 2 * x[1] ** 2
                - 0.3 * cos(3 * pi * x[0] + 4 * pi * x[1]) + 0.3)


class BoxBetts(Benchmark):
    
    r"""
    BoxBetts objective function.

    The BoxBetts global optimization problem is a multimodal
    minimization problem defined as follows

    .. math::

        f_{\text{BoxBetts}}(x) = \sum_{i=1}^k g(x_i)^2


    Where, in this exercise:

    .. math::

        g(x) = e^{-0.1i x_1} - e^{-0.1i x_2} - x_3\left[e^{-0.1i}
        - e^{-i}\right]


    And :math:`k = 10`.

    Here, :math:`x_1 \in [0.9, 1.2], x_2 \in [9, 11.2], x_3 \in [0.9, 1.2]`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 10, 1]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=3):
        # 调用父类 Benchmark 的初始化方法
        Benchmark.__init__(self, dimensions)

        # 定义搜索空间边界
        self._bounds = ([0.9, 1.2], [9.0, 11.2], [0.9, 1.2])
        
        # 设置全局最优解
        self.global_optimum = [[1.0, 10.0, 1.0]]
        
        # 设置全局最优值
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估次数
        self.nfev += 1

        # 定义指数序列 i
        i = arange(1, 11)
        
        # 计算 BoxBetts 函数值
        g = (exp(-0.1 * i * x[0]) - exp(-0.1 * i * x[1])
             - (exp(-0.1 * i) - exp(-i)) * x[2])
        
        return sum(g**2)


class Branin01(Benchmark):
    
    r"""
    Branin01  objective function.

    The Branin01 global optimization problem is a multimodal minimization
    problem defined as follows

    .. math::

        f_{\text{Branin01}}(x) = \left(- 1.275 \frac{x_1^{2}}{\pi^{2}} + 5
        \frac{x_1}{\pi} + x_2 -6\right)^{2} + \left(10 -\frac{5}{4 \pi} \right)
        \cos\left(x_1\right) + 10


    with :math:`x_1 \in [-5, 10], x_2 \in [0, 15]`

    *Global optimum*: :math:`f(x) = 0.39788735772973816` for :math:`x =
    # 定义一个 BenchmarkFunction 类的子类，用于表示 Rosenbrock 函数
    class Rosenbrock(Benchmark):

        # 初始化函数，设置维度默认为 2
        def __init__(self, dimensions=2):
            # 调用父类 Benchmark 的初始化函数
            Benchmark.__init__(self, dimensions)

            # 定义 Rosenbrock 函数的定义域边界
            self._bounds = [(-5., 10.), (0., 15.)]

            # 设置 Rosenbrock 函数的全局最优解和全局最优值
            self.global_optimum = [[-pi, 12.275], [pi, 2.275], [3 * pi, 2.475]]
            self.fglob = 0.39788735772973816

        # 定义 Rosenbrock 函数的计算方法
        def fun(self, x, *args):
            # 增加函数评估次数计数器
            self.nfev += 1

            # 计算 Rosenbrock 函数的值
            return ((x[1] - (5.1 / (4 * pi ** 2)) * x[0] ** 2
                     + 5 * x[0] / pi - 6) ** 2
                    + 10 * (1 - 1 / (8 * pi)) * cos(x[0]) + 10)
class Branin02(Benchmark):
    r"""
    Branin02 objective function.

    The Branin02 global optimization problem is a multimodal minimization
    problem defined as follows

    .. math::

        f_{\text{Branin02}}(x) = \left(- 1.275 \frac{x_1^{2}}{\pi^{2}}
        + 5 \frac{x_1}{\pi} + x_2 - 6 \right)^{2} + \left(10 - \frac{5}{4 \pi}
        \right) \cos\left(x_1\right) \cos\left(x_2\right)
        + \log(x_1^2+x_2^2 + 1) + 10


    with :math:`x_i \in [-5, 15]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 5.559037` for :math:`x = [-3.2, 12.53]`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=2):
        # 调用父类的初始化方法，设置维度
        Benchmark.__init__(self, dimensions)

        # 设定变量的取值范围
        self._bounds = [(-5.0, 15.0), (-5.0, 15.0)]

        # 设定全局最优解
        self.global_optimum = [[-3.1969884, 12.52625787]]
        # 设定全局最优值
        self.fglob = 5.5589144038938247

    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 计算Branin02函数的值
        return ((x[1] - (5.1 / (4 * pi ** 2)) * x[0] ** 2
                + 5 * x[0] / pi - 6) ** 2
                + 10 * (1 - 1 / (8 * pi)) * cos(x[0]) * cos(x[1])
                + log(x[0] ** 2.0 + x[1] ** 2.0 + 1.0) + 10)


class Brent(Benchmark):
    r"""
    Brent objective function.

    The Brent [1]_ global optimization problem is a multimodal minimization
    problem defined as follows:

    .. math::

        f_{\text{Brent}}(x) = (x_1 + 10)^2 + (x_2 + 10)^2 + e^{(-x_1^2 -x_2^2)}


    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [-10, -10]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO solution is different to Jamil#24
    """

    def __init__(self, dimensions=2):
        # 调用父类的初始化方法，设置维度
        Benchmark.__init__(self, dimensions)

        # 设定变量的取值范围
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.custom_bounds = ([-10, 2], [-10, 2])

        # 设定全局最优解
        self.global_optimum = [[-10.0, -10.0]]
        # 设定全局最优值
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1
        # 计算Brent函数的值
        return ((x[0] + 10.0) ** 2.0 + (x[1] + 10.0) ** 2.0
                + exp(-x[0] ** 2.0 - x[1] ** 2.0))


class Brown(Benchmark):
    r"""
    Brown objective function.

    The Brown [1]_ global optimization problem is a multimodal minimization
    problem defined as follows:

    .. math::

        f_{\text{Brown}}(x) = \sum_{i=1}^{n-1}\left[
        \left(x_i^2\right)^{x_{i + 1}^2 + 1}
        + \left(x_{i + 1}^2\right)^{x_i^2 + 1}\right]


    with :math:`x_i \in [-1, 4]` for :math:`i=1,...,n`.

    *Global optimum*: :math:`f(x_i) = 0` for :math:`x_i = 0` for
    :math:`i=1,...,n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    """
    # 初始化方法，设置对象的维度（默认为2）
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，传入维度参数
        Benchmark.__init__(self, dimensions)

        # 设置对象的边界为 [-1.0, 4.0] 的列表，元素个数为维度数 N
        self._bounds = list(zip([-1.0] * self.N, [4.0] * self.N))
        
        # 自定义边界设置为固定值列表 [-1.0, 1.0]，重复出现两次
        self.custom_bounds = ([-1.0, 1.0], [-1.0, 1.0])
        
        # 设置全局最优解为一个 N 维度的零列表
        self.global_optimum = [[0 for _ in range(self.N)]]
        
        # 设置全局最优解的函数值为 0.0
        self.fglob = 0.0
        
        # 设置维度变换标志为 True
        self.change_dimensionality = True

    # 定义计算函数值的方法，接受参数 x 和可变参数 args
    def fun(self, x, *args):
        # 每调用一次 fun 方法，增加一次函数评估计数
        self.nfev += 1

        # 提取 x 的前 N-1 维到 x0
        x0 = x[:-1]
        # 提取 x 的后 N-1 维到 x1
        x1 = x[1:]
        
        # 计算并返回函数值，这里使用了数学表达式进行计算
        return sum((x0 ** 2.0) ** (x1 ** 2.0 + 1.0)
                   + (x1 ** 2.0) ** (x0 ** 2.0 + 1.0))
class Bukin02(Benchmark):
    """
    Bukin02 objective function.

    The Bukin02 [1]_ global optimization problem is a multimodal minimization
    problem defined as follows:

    .. math::

        f_{\text{Bukin02}}(x) = 100 (x_2^2 - 0.01x_1^2 + 1)
        + 0.01(x_1 + 10)^2

    with :math:`x_1 \in [-15, -5], x_2 \in [-3, 3]`

    *Global optimum*: :math:`f(x) = -124.75` for :math:`x = [-15, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: I think that Gavana and Jamil are wrong on this function. In both
    sources the x[1] term is not squared. As such there will be a minimum at
    the smallest value of x[1].
    """

    def __init__(self, dimensions=2):
        """
        Initialize the Bukin02 benchmark function.

        Parameters:
        dimensions : int, optional
            Number of dimensions of the function (default is 2)
        """
        Benchmark.__init__(self, dimensions)

        # Define the search space bounds for each dimension
        self._bounds = [(-15.0, -5.0), (-3.0, 3.0)]

        # Define the global optimum point(s) for this function
        self.global_optimum = [[-15.0, 0.0]]

        # Define the global optimum function value
        self.fglob = -124.75

    def fun(self, x, *args):
        """
        Evaluate the Bukin02 function at point x.

        Parameters:
        x : array-like
            Point at which to evaluate the function

        Returns:
        float
            Value of the Bukin02 function at point x
        """

        # Increment the function evaluation count
        self.nfev += 1

        # Compute the Bukin02 function value at point x
        return (100 * (x[1] ** 2 - 0.01 * x[0] ** 2 + 1.0)
                + 0.01 * (x[0] + 10.0) ** 2.0)


class Bukin04(Benchmark):
    """
    Bukin04 objective function.

    The Bukin04 [1]_ global optimization problem is a multimodal minimization
    problem defined as follows:

    .. math::

        f_{\text{Bukin04}}(x) = 100 x_2^{2} + 0.01 \lvert{x_1 + 10}
        \rvert

    with :math:`x_1 \in [-15, -5], x_2 \in [-3, 3]`

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [-10, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        """
        Initialize the Bukin04 benchmark function.

        Parameters:
        dimensions : int, optional
            Number of dimensions of the function (default is 2)
        """
        Benchmark.__init__(self, dimensions)

        # Define the search space bounds for each dimension
        self._bounds = [(-15.0, -5.0), (-3.0, 3.0)]

        # Define the global optimum point(s) for this function
        self.global_optimum = [[-10.0, 0.0]]

        # Define the global optimum function value
        self.fglob = 0.0

    def fun(self, x, *args):
        """
        Evaluate the Bukin04 function at point x.

        Parameters:
        x : array-like
            Point at which to evaluate the function

        Returns:
        float
            Value of the Bukin04 function at point x
        """

        # Increment the function evaluation count
        self.nfev += 1

        # Compute the Bukin04 function value at point x
        return 100 * x[1] ** 2 + 0.01 * abs(x[0] + 10)


class Bukin06(Benchmark):
    """
    Bukin06 objective function.

    The Bukin06 [1]_ global optimization problem is a multimodal minimization
    problem defined as follows:

    .. math::

        f_{\text{Bukin06}}(x) = 100 \sqrt{ \lvert{x_2 - 0.01 x_1^{2}}
        \rvert} + 0.01 \lvert{x_1 + 10} \rvert

    with :math:`x_1 \in [-15, -5], x_2 \in [-3, 3]`

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [-10, 1]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        """
        Initialize the Bukin06 benchmark function.

        Parameters:
        dimensions : int, optional
            Number of dimensions of the function (default is 2)
        """
        Benchmark.__init__(self, dimensions)

        # Define the search space bounds for each dimension
        self._bounds = [(-15.0, -5.0), (-3.0, 3.0)]

        # Define the global optimum point(s) for this function
        self.global_optimum = [[-10.0, 1.0]]

        # Define the global optimum function value
        self.fglob = 0.0
    # 定义一个方法 `fun`，接受 `self` 作为隐式参数，以及 `x` 和可变数量的额外参数 `args`
    def fun(self, x, *args):
        # 将对象属性 `nfev` 自增 1，用于计算方法调用次数
        self.nfev += 1
        # 返回函数值，根据给定公式计算得出
        return 100 * sqrt(abs(x[1] - 0.01 * x[0] ** 2)) + 0.01 * abs(x[0] + 10)
```