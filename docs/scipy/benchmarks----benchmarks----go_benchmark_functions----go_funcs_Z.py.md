# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_Z.py`

```
from numpy import abs, sum, sign, arange  # 导入所需的函数和模块
from .go_benchmark import Benchmark  # 导入本地模块go_benchmark中的Benchmark类

class Zacharov(Benchmark):  # 定义一个名为Zacharov的类，继承自Benchmark类

    r"""
    Zacharov objective function.

    This class defines the Zacharov [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Zacharov}}(x) = \sum_{i=1}^{n} x_i^2 + \left ( \frac{1}{2}
                                 \sum_{i=1}^{n} i x_i \right )^2
                                 + \left ( \frac{1}{2} \sum_{i=1}^{n} i x_i 
                                 \right )^4

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-5, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)  # 调用父类Benchmark的初始化方法

        self._bounds = list(zip([-5.0] * self.N, [10.0] * self.N))  # 设定变量取值范围
        self.custom_bounds = ([-1, 1], [-1, 1])  # 自定义变量的额外取值范围

        self.global_optimum = [[0 for _ in range(self.N)]]  # 全局最优解
        self.fglob = 0.0  # 全局最优解的函数值
        self.change_dimensionality = True  # 维度改变标志位

    def fun(self, x, *args):
        self.nfev += 1  # 计算函数调用次数

        u = sum(x ** 2)  # 计算第一项
        v = sum(arange(1, self.N + 1) * x)  # 计算第二项
        return u + (0.5 * v) ** 2 + (0.5 * v) ** 4  # 返回函数值


class ZeroSum(Benchmark):  # 定义一个名为ZeroSum的类，继承自Benchmark类

    r"""
    ZeroSum objective function.

    This class defines the ZeroSum [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{ZeroSum}}(x) = \begin{cases}
                                0 & \textrm{if} \sum_{i=1}^n x_i = 0 \\
                                1 + \left(10000 \left |\sum_{i=1}^n x_i\right|
                                \right)^{0.5} & \textrm{otherwise}
                                \end{cases}

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` where :math:`\sum_{i=1}^n x_i = 0`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)  # 调用父类Benchmark的初始化方法

        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))  # 设定变量取值范围

        self.global_optimum = [[]]  # 全局最优解
        self.fglob = 0.0  # 全局最优解的函数值
        self.change_dimensionality = True  # 维度改变标志位

    def fun(self, x, *args):
        self.nfev += 1  # 计算函数调用次数

        if abs(sum(x)) < 3e-16:  # 判断是否满足全局最优条件
            return 0.0
        return 1.0 + (10000.0 * abs(sum(x))) ** 0.5  # 返回函数值
    """
    .. math::

       f_{\text{Zettl}}(x) = \frac{1}{4} x_{1} + \left(x_{1}^{2} - 2 x_{1}
                             + x_{2}^{2}\right)^{2}

    定义了 Zettl 函数，用于全局优化问题的基准函数之一。函数形式如上所示，依赖于两个变量 x1 和 x2。

    with :math:`x_i \in [-1, 5]` for :math:`i = 1, 2`.

    定义了函数的定义域，x1 和 x2 分别在 [-1, 5] 区间内取值。

    *Global optimum*: :math:`f(x) = -0.0037912` for :math:`x = [-0.029896, 0.0]`

    给出了函数的全局最优解和相应的最优函数值。

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    引用了文献 [1]，详细描述了全局优化问题中的基准函数及其应用背景。
    """

    # 初始化 Zettl 函数的类
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设置维度信息
        Benchmark.__init__(self, dimensions)

        # 设置函数定义域的边界
        self._bounds = list(zip([-5.0] * self.N, [10.0] * self.N))

        # 设置函数的全局最优解
        self.global_optimum = [[-0.02989597760285287, 0.0]]

        # 设置函数的全局最优值
        self.fglob = -0.003791237220468656

    # 定义 Zettl 函数的计算方法
    def fun(self, x, *args):
        # 增加函数计算的调用次数计数器
        self.nfev += 1

        # 计算 Zettl 函数在给定点 x 的函数值
        return (x[0] ** 2 + x[1] ** 2 - 2 * x[0]) ** 2 + 0.25 * x[0]
class Zimmerman(Benchmark):
    
    r"""
    Zimmerman objective function.

    This class defines the Zimmerman [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Zimmerman}}(x) = \max \left[Zh1(x), Zp(Zh2(x))
                                  \textrm{sgn}(Zh2(x)), Zp(Zh3(x))
                                  \textrm{sgn}(Zh3(x)),
                                  Zp(-x_1)\textrm{sgn}(x_1),
                                  Zp(-x_2)\textrm{sgn}(x_2) \right]

    Where, in this exercise:

    .. math::

        \begin{cases}
        Zh1(x) = 9 - x_1 - x_2 \\
        Zh2(x) = (x_1 - 3)^2 + (x_2 - 2)^2 \\
        Zh3(x) = x_1x_2 - 14 \\
        Zp(t) = 100(1 + t)
        \end{cases}

    Where :math:`x` is a vector and :math:`t` is a scalar.

    Here, :math:`x_i \in [0, 100]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [7, 2]`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO implementation from Gavana
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        # 设置问题的变量边界，每个维度都是 [0.0, 100.0]
        self._bounds = list(zip([0.0] * self.N, [100.0] * self.N))
        
        # 自定义的边界设置为 ([0.0, 8.0], [0.0, 8.0])
        self.custom_bounds = ([0.0, 8.0], [0.0, 8.0])

        # 全局最优解设定为 [[7.0, 2.0]]
        self.global_optimum = [[7.0, 2.0]]
        
        # 全局最优解的函数值设定为 0.0
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估计数器
        self.nfev += 1

        # 计算多个函数值中的最大值作为返回值
        return max(Zh1(x),
                   Zp(Zh2(x)) * sign(Zh2(x)),
                   Zp(Zh3(x)) * sign(Zh3(x)),
                   Zp(-x[0]) * sign(x[0]),
                   Zp(-x[1]) * sign(x[1]))


class Zirilli(Benchmark):
    
    r"""
    Zettl objective function.

    This class defines the Zirilli [1]_ global optimization problem. This is a
    unimodal minimization problem defined as follows:

    .. math::

        f_{\text{Zirilli}}(x) = 0.25x_1^4 - 0.5x_1^2 + 0.1x_1 + 0.5x_2^2

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -0.3523` for :math:`x = [-1.0465, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        # 设置问题的变量边界，每个维度都是 [-10.0, 10.0]
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        
        # 自定义的边界设置为 ([-2.0, 2.0], [-2.0, 2.0])
        self.custom_bounds = ([-2.0, 2.0], [-2.0, 2.0])

        # 全局最优解设定为 [[-1.0465, 0.0]]
        self.global_optimum = [[-1.0465, 0.0]]
        
        # 全局最优解的函数值设定为 -0.35238603
        self.fglob = -0.35238603

    def fun(self, x, *args):
        # 增加函数评估计数器
        self.nfev += 1

        # 计算 Zirilli 函数的值并返回
        return 0.25 * x[0] ** 4 - 0.5 * x[0] ** 2 + 0.1 * x[0] + 0.5 * x[1] ** 2
```