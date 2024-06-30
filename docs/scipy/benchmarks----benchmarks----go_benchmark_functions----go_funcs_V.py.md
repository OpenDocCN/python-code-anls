# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_V.py`

```
from numpy import sum, cos, sin, log
from .go_benchmark import Benchmark

class VenterSobiezcczanskiSobieski(Benchmark):
    """
    Venter Sobiezcczanski-Sobieski objective function.

    This class defines the Venter Sobiezcczanski-Sobieski global optimization
    problem. This is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{VenterSobiezcczanskiSobieski}}(x) = x_1^2 - 100 \cos^2(x_1)
                                                      - 100 \cos(x_1^2/30)
                                                      + x_2^2 - 100 \cos^2(x_2)
                                                      - 100 \cos(x_2^2/30)

    with :math:`x_i \in [-50, 50]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -400` for :math:`x = [0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Jamil #160 hasn't written the equation very well. Normally a cos
    squared term is written as cos^2(x) rather than cos(x)^2
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设定维度
        Benchmark.__init__(self, dimensions)

        # 定义搜索空间边界
        self._bounds = list(zip([-50.0] * self.N, [50.0] * self.N))
        # 自定义边界
        self.custom_bounds = ([-10, 10], [-10, 10])

        # 全局最优解
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = -400

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 计算目标函数中的各个部分
        u = x[0] ** 2.0 - 100.0 * cos(x[0]) ** 2.0
        v = -100.0 * cos(x[0] ** 2.0 / 30.0) + x[1] ** 2.0
        w = - 100.0 * cos(x[1]) ** 2.0 - 100.0 * cos(x[1] ** 2.0 / 30.0)
        # 返回目标函数值
        return u + v + w


class Vincent(Benchmark):
    """
    Vincent objective function.

    This class defines the Vincent global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Vincent}}(x) = - \sum_{i=1}^{n} \sin(10 \log(x))

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [0.25, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -n` for :math:`x_i = 7.70628098`
    for :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设定维度
        Benchmark.__init__(self, dimensions)

        # 定义搜索空间边界
        self._bounds = list(zip([0.25] * self.N, [10.0] * self.N))

        # 全局最优解
        self.global_optimum = [[7.70628098 for _ in range(self.N)]]
        self.fglob = -float(self.N)
        # 标记需要改变维度
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 计算目标函数值
        return -sum(sin(10.0 * log(x)))
```