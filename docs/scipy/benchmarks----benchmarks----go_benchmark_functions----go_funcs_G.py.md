# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_G.py`

```
import numpy as np  # 导入 NumPy 库，用于科学计算
from numpy import abs, sin, cos, exp, floor, log, arange, prod, sqrt, sum  # 从 NumPy 库导入特定函数

from .go_benchmark import Benchmark  # 导入自定义的 Benchmark 类


class Gear(Benchmark):  # 定义 Gear 类，继承自 Benchmark 类

    r"""
    Gear objective function.

    This class defines the Gear [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Gear}}({x}) = \left \{ \frac{1.0}{6.931}
       - \frac{\lfloor x_1\rfloor \lfloor x_2 \rfloor }
       {\lfloor x_3 \rfloor \lfloor x_4 \rfloor } \right\}^2


    with :math:`x_i \in [12, 60]` for :math:`i = 1, ..., 4`.

    *Global optimum*: :math:`f(x) = 2.7 \cdot 10^{-12}` for :math:`x =
    [16, 19, 43, 49]`, where the various :math:`x_i` may be permuted.

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=4):  # 初始化函数，设置默认维度为4
        Benchmark.__init__(self, dimensions)  # 调用父类 Benchmark 的初始化函数

        self._bounds = list(zip([12.0] * self.N, [60.0] * self.N))  # 设置变量范围的上下界
        self.global_optimum = [[16, 19, 43, 49]]  # 全局最优解
        self.fglob = 2.7e-12  # 全局最优解对应的函数值

    def fun(self, x, *args):  # 定义目标函数 fun，接受参数 x 和其他可选参数
        self.nfev += 1  # 增加函数评估的计数器值

        return (1. / 6.931
                - floor(x[0]) * floor(x[1]) / floor(x[2]) / floor(x[3])) ** 2  # 返回 Gear 函数的计算结果


class Giunta(Benchmark):  # 定义 Giunta 类，继承自 Benchmark 类

    r"""
    Giunta objective function.

    This class defines the Giunta [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Giunta}}({x}) = 0.6 + \sum_{i=1}^{n} \left[\sin^{2}\left(1
        - \frac{16}{15} x_i\right) - \frac{1}{50} \sin\left(4
        - \frac{64}{15} x_i\right) - \sin\left(1
        - \frac{16}{15} x_i\right)\right]


    with :math:`x_i \in [-1, 1]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0.06447042053690566` for
    :math:`x = [0.4673200277395354, 0.4673200169591304]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Jamil has the wrong fglob.  I think there is a lower value.
    """

    def __init__(self, dimensions=2):  # 初始化函数，设置默认维度为2
        Benchmark.__init__(self, dimensions)  # 调用父类 Benchmark 的初始化函数

        self._bounds = list(zip([-1.0] * self.N, [1.0] * self.N))  # 设置变量范围的上下界
        self.global_optimum = [[0.4673200277395354, 0.4673200169591304]]  # 全局最优解
        self.fglob = 0.06447042053690566  # 全局最优解对应的函数值

    def fun(self, x, *args):  # 定义目标函数 fun，接受参数 x 和其他可选参数
        self.nfev += 1  # 增加函数评估的计数器值

        arg = 16 * x / 15.0 - 1  # 计算 Giunta 函数中的参数
        return 0.6 + sum(sin(arg) + sin(arg) ** 2 + sin(4 * arg) / 50.)  # 返回 Giunta 函数的计算结果


class GoldsteinPrice(Benchmark):  # 定义 GoldsteinPrice 类，继承自 Benchmark 类

    r"""
    Goldstein-Price objective function.

    This class defines the Goldstein-Price [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:
    # 初始化 Goldstein-Price 函数类，继承自 Benchmark 类，设定默认维度为 2
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置变量的取值范围，对每个维度均为 [-2.0, 2.0]
        self._bounds = list(zip([-2.0] * self.N, [2.0] * self.N))

        # 设置全局最优解，对 Goldstein-Price 函数而言，全局最优解是 [0, -1]
        self.global_optimum = [[0., -1.]]
        
        # 设置全局最优值，Goldstein-Price 函数在全局最优解处的函数值为 3.0
        self.fglob = 3.0

    # 实现 Goldstein-Price 函数的计算
    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 计算 Goldstein-Price 函数的第一个部分 a
        a = (1 + (x[0] + x[1] + 1) ** 2
             * (19 - 14 * x[0] + 3 * x[0] ** 2
             - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2))
        
        # 计算 Goldstein-Price 函数的第二个部分 b
        b = (30 + (2 * x[0] - 3 * x[1]) ** 2
             * (18 - 32 * x[0] + 12 * x[0] ** 2
             + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))
        
        # 返回 Goldstein-Price 函数的最终结果
        return a * b
class Griewank(Benchmark):
    r"""
    Griewank objective function.

    This class defines the Griewank global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Griewank}}(x) = \frac{1}{4000}\sum_{i=1}^n x_i^2
        - \prod_{i=1}^n\cos\left(\frac{x_i}{\sqrt{i}}\right) + 1

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-600, 600]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法
        Benchmark.__init__(self, dimensions)

        # 定义变量的取值范围
        self._bounds = list(zip([-100.0] * self.N, [100.0] * self.N))
        
        # 自定义的变量取值范围
        self.custom_bounds = [(-50, 50), (-50, 50)]

        # 全局最优解
        self.global_optimum = [[0 for _ in range(self.N)]]
        
        # 全局最优解对应的函数值
        self.fglob = 0.0
        
        # 是否改变维度的标志
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估计数器
        self.nfev += 1

        # 定义索引数组 i
        i = arange(1., np.size(x) + 1.)
        
        # Griewank 函数表达式
        return sum(x ** 2 / 4000) - prod(cos(x / sqrt(i))) + 1


class Gulf(Benchmark):
    r"""
    Gulf objective function.

    This class defines the Gulf [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Gulf}}(x) = \sum_{i=1}^99 \left( e^{-\frac{\lvert y_i
        - x_2 \rvert^{x_3}}{x_1}}  - t_i \right)


    Where, in this exercise:

    .. math::

       t_i = i/100 \\
       y_i = 25 + [-50 \log(t_i)]^{2/3}


    with :math:`x_i \in [0, 60]` for :math:`i = 1, 2, 3`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [50, 25, 1.5]`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO Gavana has absolute of (u - x[1]) term. Jamil doesn't... Leaving it in.
    """

    def __init__(self, dimensions=3):
        # 调用父类 Benchmark 的初始化方法
        Benchmark.__init__(self, dimensions)

        # 定义变量的取值范围
        self._bounds = list(zip([0.0] * self.N, [50.0] * self.N))

        # 全局最优解
        self.global_optimum = [[50.0, 25.0, 1.5]]
        
        # 全局最优解对应的函数值
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估计数器
        self.nfev += 1

        # 定义常数 m 和索引数组 i
        m = 99.
        i = arange(1., m + 1)
        
        # 计算辅助变量 u
        u = 25 + (-50 * log(i / 100.)) ** (2 / 3.)
        
        # Gulf 函数表达式
        vec = (exp(-((abs(u - x[1])) ** x[2] / x[0])) - i / 100.)
        return sum(vec ** 2)
```