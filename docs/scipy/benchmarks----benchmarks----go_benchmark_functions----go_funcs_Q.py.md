# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_Q.py`

```
from numpy import abs, sum, arange, sqrt

from .go_benchmark import Benchmark  # 导入 Benchmark 类

class Qing(Benchmark):
    r"""
    Qing objective function.

    This class defines the Qing [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Qing}}(x) = \sum_{i=1}^{n} (x_i^2 - i)^2

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-500, 500]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = \pm \sqrt(i)` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)  # 调用父类 Benchmark 的初始化方法

        self._bounds = list(zip([-500.0] * self.N,
                           [500.0] * self.N))  # 定义默认的搜索空间边界
        self.custom_bounds = [(-2, 2), (-2, 2)]  # 自定义搜索空间边界
        self.global_optimum = [[sqrt(_) for _ in range(1, self.N + 1)]]  # 全局最优解
        self.fglob = 0  # 目标函数的全局最小值
        self.change_dimensionality = True  # 标志是否改变维度

    def fun(self, x, *args):
        self.nfev += 1  # 计算函数评估次数

        i = arange(1, self.N + 1)  # 创建一个从 1 到 self.N 的数组
        return sum((x ** 2.0 - i) ** 2.0)  # 返回目标函数的计算结果


class Quadratic(Benchmark):
    r"""
    Quadratic objective function.

    This class defines the Quadratic [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Quadratic}}(x) = -3803.84 - 138.08x_1 - 232.92x_2 + 128.08x_1^2
        + 203.64x_2^2 + 182.25x_1x_2

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -3873.72418` for
    :math:`x = [0.19388, 0.48513]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)  # 调用父类 Benchmark 的初始化方法

        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))  # 定义默认的搜索空间边界
        self.custom_bounds = [(0, 1), (0, 1)]  # 自定义搜索空间边界
        self.global_optimum = [[0.19388, 0.48513]]  # 全局最优解
        self.fglob = -3873.72418  # 目标函数的全局最小值
        self.change_dimensionality = True  # 标志是否改变维度

    def fun(self, x, *args):
        self.nfev += 1  # 计算函数评估次数

        return (-3803.84 - 138.08 * x[0] - 232.92 * x[1] + 128.08 * x[0] ** 2.0
                + 203.64 * x[1] ** 2.0 + 182.25 * x[0] * x[1])  # 返回目标函数的计算结果


class Quintic(Benchmark):
    r"""
    Quintic objective function.

    This class defines the Quintic [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Quintic}}(x) = \sum_{i=1}^{n} \left|{x_{i}^{5} - 3 x_{i}^{4}
        + 4 x_{i}^{3} + 2 x_{i}^{2} - 10 x_{i} -4}\right|

    Here, :math:`n` represents the number of dimensions and
    # 定义一个 BenchmarkFunction 类的子类，表示特定的基准函数
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设置维度
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间的边界，每个维度的取值范围为 [-10.0, 10.0]
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        
        # 设置自定义的边界条件，这里指定了两个维度的边界为 (-2, 2)
        self.custom_bounds = [(-2, 2), (-2, 2)]

        # 设置全局最优解，对于每个维度设置为 -1.0
        self.global_optimum = [[-1.0 for _ in range(self.N)]]
        
        # 设置全局最优解的函数值为 0
        self.fglob = 0
        
        # 标记是否改变了问题的维度，这里设为 True
        self.change_dimensionality = True

    # 定义求解函数值的方法，输入参数 x 是待求解的变量，*args 是额外的参数（未使用）
    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 计算基准函数的值，这里是一个多项式的值求和
        return sum(abs(x ** 5 - 3 * x ** 4 + 4 * x ** 3 + 2 * x ** 2
                       - 10 * x - 4))
```