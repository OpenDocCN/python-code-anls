# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_L.py`

```
# 从 numpy 模块中导入所需函数 sum, cos, exp, pi, arange, sin
from numpy import sum, cos, exp, pi, arange, sin
# 从当前目录下的 go_benchmark 模块中导入 Benchmark 类
from .go_benchmark import Benchmark

# 定义 Langermann 类，继承自 Benchmark 类
class Langermann(Benchmark):

    r"""
    Langermann objective function.

    This class defines the Langermann [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Langermann}}(x) = - \sum_{i=1}^{5} 
        \frac{c_i \cos\left\{\pi \left[\left(x_{1}- a_i\right)^{2}
        + \left(x_{2} - b_i \right)^{2}\right]\right\}}{e^{\frac{\left( x_{1}
        - a_i\right)^{2} + \left( x_{2} - b_i\right)^{2}}{\pi}}}

    Where:

    .. math::

        \begin{matrix}
        a = [3, 5, 2, 1, 7]\\
        b = [5, 2, 1, 4, 9]\\
        c = [1, 2, 5, 2, 3] \\
        \end{matrix}

    Here :math:`x_i \in [0, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -5.1621259`
    for :math:`x = [2.00299219, 1.006096]`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO: Langermann from Gavana is _not the same_ as Jamil #68.
    """

    # 初始化方法，设置维度为 2
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间边界为每个维度 [0.0, 10.0]
        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))

        # 设置全局最优解
        self.global_optimum = [[2.00299219, 1.006096]]
        # 设置全局最优值
        self.fglob = -5.1621259

    # 定义目标函数 fun，计算 Langermann 函数值
    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 定义 Langermann 函数中的参数 a, b, c
        a = [3, 5, 2, 1, 7]
        b = [5, 2, 1, 4, 9]
        c = [1, 2, 5, 2, 3]

        # 计算并返回 Langermann 函数的值
        return (-sum(c * exp(-(1 / pi) * ((x[0] - a) ** 2 +
                    (x[1] - b) ** 2)) * cos(pi * ((x[0] - a) ** 2
                                            + (x[1] - b) ** 2))))
    """
    这段代码定义了一个类，实现了一种特定的优化算法的基准函数。
    """

    def __init__(self, dimensions=6):
        """
        初始化函数，设置类的初始状态和属性。

        Parameters:
        dimensions (int): 维度数，默认为6，必须在6到60之间。

        Raises:
        ValueError: 如果dimensions不在6到60之间，抛出数值错误异常。

        """
        # dimensions is in [6:60]
        # max dimensions is going to be 60.
        if dimensions not in range(6, 61):
            raise ValueError("LJ dimensions must be in (6, 60)")

        # 调用父类Benchmark的初始化方法，设置维度数
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间的边界，默认为每个维度 [-4.0, 4.0]
        self._bounds = list(zip([-4.0] * self.N, [4.0] * self.N))

        # 初始化全局最优解为空列表
        self.global_optimum = [[]]

        # 设置已知的一组局部最小值列表
        self.minima = [-1.0, -3.0, -6.0, -9.103852, -12.712062,
                       -16.505384, -19.821489, -24.113360, -28.422532,
                       -32.765970, -37.967600, -44.326801, -47.845157,
                       -52.322627, -56.815742, -61.317995, -66.530949,
                       -72.659782, -77.1777043]

        # 根据维度计算并设置全局最优解的初始值
        k = int(dimensions / 3)
        self.fglob = self.minima[k - 2]

        # 标记维度是否已更改
        self.change_dimensionality = True

    def change_dimensions(self, ndim):
        """
        更改优化问题的维度。

        Parameters:
        ndim (int): 新的维度数

        Raises:
        ValueError: 如果ndim不在6到60之间，抛出数值错误异常。

        """
        if ndim not in range(6, 61):
            raise ValueError("LJ dimensions must be in (6, 60)")

        # 调用父类Benchmark的方法，更改维度数
        Benchmark.change_dimensions(self, ndim)

        # 根据新的维度计算并设置全局最优解
        self.fglob = self.minima[int(self.N / 3) - 2]

    def fun(self, x, *args):
        """
        定义了优化问题的目标函数。

        Parameters:
        x (list): 输入变量向量
        *args: 额外的参数（这里未使用）

        Returns:
        float: 计算得到的目标函数值

        """
        # 每次调用目标函数计数增加
        self.nfev += 1

        # 计算 k 值，k = N / 3
        k = int(self.N / 3)
        s = 0.0

        # 循环遍历计算目标函数的主体部分
        for i in range(k - 1):
            for j in range(i + 1, k):
                a = 3 * i
                b = 3 * j
                xd = x[a] - x[b]
                yd = x[a + 1] - x[b + 1]
                zd = x[a + 2] - x[b + 2]
                ed = xd * xd + yd * yd + zd * zd
                ud = ed * ed * ed
                if ed > 0.0:
                    s += (1.0 / ud - 2.0) / ud

        # 返回目标函数值
        return s
class Leon(Benchmark):
    r"""
    Leon objective function.

    This class defines the Leon [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Leon}}(\mathbf{x}) = \left(1 - x_{1}\right)^{2} 
        + 100 \left(x_{2} - x_{1}^{2} \right)^{2}


    with :math:`x_i \in [-1.2, 1.2]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 1]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的构造函数，初始化维度
        Benchmark.__init__(self, dimensions)

        # 设置变量的取值范围
        self._bounds = list(zip([-1.2] * self.N, [1.2] * self.N))

        # 设置全局最优解和全局最优值
        self.global_optimum = [[1 for _ in range(self.N)]]
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 计算 Leon 函数的值
        return 100. * (x[1] - x[0] ** 2.0) ** 2.0 + (1 - x[0]) ** 2.0


class Levy03(Benchmark):
    r"""
    Levy 3 objective function.

    This class defines the Levy 3 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Levy03}}(\mathbf{x}) =
        \sin^2(\pi y_1)+\sum_{i=1}^{n-1}(y_i-1)^2[1+10\sin^2(\pi y_{i+1})]+(y_n-1)^2

    Where, in this exercise:

    .. math::

        y_i=1+\frac{x_i-1}{4}


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i=1,...,n`.

    *Global optimum*: :math:`f(x_i) = 0` for :math:`x_i = 1` for :math:`i=1,...,n`

    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005

    TODO: not clear what the Levy function definition is.  Gavana, Mishra,
    Adorio have different forms. Indeed Levy 3 docstring from Gavana
    disagrees with the Gavana code!  The following code is from the Mishra
    listing of Levy08.
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的构造函数，初始化维度
        Benchmark.__init__(self, dimensions)

        # 设置变量的取值范围
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        # 设置自定义的变量范围
        self.custom_bounds = [(-5, 5), (-5, 5)]

        # 设置全局最优解和全局最优值
        self.global_optimum = [[1 for _ in range(self.N)]]
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 将变量 x 转换为 y
        y = 1 + (x - 1) / 4
        # 计算 Levy 3 函数的值
        v = sum((y[:-1] - 1) ** 2 * (1 + 10 * sin(pi * y[1:]) ** 2))
        z = (y[-1] - 1) ** 2
        return sin(pi * y[0]) ** 2 + v + z


class Levy05(Benchmark):
    r"""
    Levy 5 objective function.

    This class defines the Levy 5 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:
    """
    .. math::

        f_{\text{Levy05}}(\mathbf{x}) =
        \sum_{i=1}^{5} i \cos \left[(i-1)x_1 + i \right] \times \sum_{j=1}^{5} j
        \cos \left[(j+1)x_2 + j \right] + (x_1 + 1.42513)^2 + (x_2 + 0.80032)^2

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i=1,...,n`.

    *Global optimum*: :math:`f(x_i) = -176.1375779` for
    :math:`\mathbf{x} = [-1.30685, -1.42485]`.

    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005
    """

    # 定义一个名为 Levy05 的基准函数类
    def __init__(self, dimensions=2):
        # 调用基类 Benchmark 的初始化方法，设定维度
        Benchmark.__init__(self, dimensions)

        # 设定函数自变量 x 的取值范围
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        # 设定自定义的边界范围
        self.custom_bounds = ([-2.0, 2.0], [-2.0, 2.0])

        # 设定全局最优解
        self.global_optimum = [[-1.30685, -1.42485]]
        # 设定全局最优解的函数值
        self.fglob = -176.1375779

    # 定义基准函数 Levy05 的具体实现
    def fun(self, x, *args):
        # 计算函数评估次数加一
        self.nfev += 1

        # 定义序列 i，从 1 到 5
        i = arange(1, 6)
        # 计算 a 的值，利用序列 i 和函数 x[0] 的值
        a = i * cos((i - 1) * x[0] + i)
        # 计算 b 的值，利用序列 i 和函数 x[1] 的值
        b = i * cos((i + 1) * x[1] + i)

        # 返回函数的计算结果，包括两个和式以及额外的平方项
        return sum(a) * sum(b) + (x[0] + 1.42513) ** 2 + (x[1] + 0.80032) ** 2
class Levy13(Benchmark):
    r"""
    Levy13 objective function.

    This class defines the Levy13 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Levy13}}(x) = \left(x_{1} -1\right)^{2} \left[\sin^{2}
        \left(3 \pi x_{2}\right) + 1\right] + \left(x_{2} 
        - 1\right)^{2} \left[\sin^{2}\left(2 \pi x_{2}\right)
        + 1\right] + \sin^{2}\left(3 \pi x_{1}\right)


    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 1]`

    .. [1] Mishra, S. Some new test functions for global optimization and
    performance of repulsive particle swarm method.
    Munich Personal RePEc Archive, 2006, 2718
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设定维度
        Benchmark.__init__(self, dimensions)

        # 定义变量 _bounds，表示每个维度的取值范围为 [-10.0, 10.0]
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        
        # 自定义的边界条件，限制每个维度的取值范围为 [-5, 5]
        self.custom_bounds = [(-5, 5), (-5, 5)]

        # 全局最优解设定为 x = [1, 1]
        self.global_optimum = [[1 for _ in range(self.N)]]
        
        # 设定全局最小值为 0.0
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数计算次数的计数器
        self.nfev += 1

        # 计算 u = sin(3 * pi * x1)^2
        u = sin(3 * pi * x[0]) ** 2
        
        # 计算 v = (x1 - 1)^2 * [1 + sin^2(3 * pi * x2)]
        v = (x[0] - 1) ** 2 * (1 + (sin(3 * pi * x[1])) ** 2)
        
        # 计算 w = (x2 - 1)^2 * [1 + sin^2(2 * pi * x2)]
        w = (x[1] - 1) ** 2 * (1 + (sin(2 * pi * x[1])) ** 2)
        
        # 返回目标函数值 f(x) = u + v + w
        return u + v + w
```