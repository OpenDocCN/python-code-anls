# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_R.py`

```
from numpy import abs, sum, sin, cos, asarray, arange, pi, exp, log, sqrt
from scipy.optimize import rosen
from .go_benchmark import Benchmark

# 定义 Rana 类，继承 Benchmark 类
class Rana(Benchmark):

    r"""
    Rana objective function.

    This class defines the Rana [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Rana}}(x) = \sum_{i=1}^{n} \left[x_{i}
        \sin\left(\sqrt{\lvert{x_{1} - x_{i} + 1}\rvert}\right)
        \cos\left(\sqrt{\lvert{x_{1} + x_{i} + 1}\rvert}\right) +
        \left(x_{1} + 1\right) \sin\left(\sqrt{\lvert{x_{1} + x_{i} +
        1}\rvert}\right) \cos\left(\sqrt{\lvert{x_{1} - x_{i} +
        1}\rvert}\right)\right]

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in
    [-500.0, 500.0]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x_i) = -928.5478` for
    :math:`x = [-300.3376, 500]`.

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: homemade global minimum here.
    """

    # 初始化 Rana 类
    def __init__(self, dimensions=2):
        # 调用 Benchmark 类的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置边界
        self._bounds = list(zip([-500.000001] * self.N, [500.000001] * self.N))

        # 设置全局最优解和最优值
        self.global_optimum = [[-300.3376, 500.]]
        self.fglob = -500.8021602966615
        self.change_dimensionality = True

    # 定义 Rana 函数
    def fun(self, x, *args):
        self.nfev += 1

        t1 = sqrt(abs(x[1:] + x[: -1] + 1))
        t2 = sqrt(abs(x[1:] - x[: -1] + 1))
        v = (x[1:] + 1) * cos(t2) * sin(t1) + x[:-1] * cos(t1) * sin(t2)
        return sum(v)


# 定义 Rastrigin 类，继承 Benchmark 类
class Rastrigin(Benchmark):

    r"""
    Rastrigin objective function.

    This class defines the Rastrigin [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Rastrigin}}(x) = 10n \sum_{i=1}^n \left[ x_i^2
        - 10 \cos(2\pi x_i) \right]

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-5.12, 5.12]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    # 初始化 Rastrigin 类
    def __init__(self, dimensions=2):
        # 调用 Benchmark 类的初始化方法
        Benchmark.__init__(self, dimensions)
        # 设置边界
        self._bounds = list(zip([-5.12] * self.N, [5.12] * self.N))

        # 设置全局最优解和最优值
        self.global_optimum = [[0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    # 定义 Rastrigin 函数
    def fun(self, x, *args):
        self.nfev += 1

        return 10.0 * self.N + sum(x ** 2.0 - 10.0 * cos(2.0 * pi * x))


# 定义 Ratkowsky01 类，继承 Benchmark 类
class Ratkowsky01(Benchmark):

    """
    Ratkowsky objective function.

    .. [1] https://www.itl.nist.gov/div898/strd/nls/data/ratkowsky3.shtml
    """

    # TODO, this is a NIST regression standard dataset
    # 初始化函数，设置默认维度为4，并调用父类 Benchmark 的初始化方法
    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)

        # 设置边界值列表，每个维度的上下边界分别是 [0., 1., 0., 0.1] 和 [1000, 20., 3., 6.]
        self._bounds = list(zip([0., 1., 0., 0.1],
                           [1000, 20., 3., 6.]))

        # 全局最优解的列表，包含一个长度为4的列表
        self.global_optimum = [[6.996415127e2, 5.2771253025, 7.5962938329e-1,
                                1.2792483859]]

        # 全局最优解对应的函数值
        self.fglob = 8.786404908e3

        # 一维数组 a 包含15个数值
        self.a = asarray([16.08, 33.83, 65.80, 97.20, 191.55, 326.20, 386.87,
                          520.53, 590.03, 651.92, 724.93, 699.56, 689.96,
                          637.56, 717.41])

        # 一维数组 b 包含从1到15的数值
        self.b = arange(1, 16.)

    # 定义函数 fun，计算给定参数 x 下的目标函数值
    def fun(self, x, *args):
        # 每调用一次 fun 函数，自增计数器 nfev
        self.nfev += 1

        # 根据公式计算向量 vec 的值
        vec = x[0] / ((1 + exp(x[1] - x[2] * self.b)) ** (1 / x[3]))

        # 计算目标函数值，返回 sum((a - vec) ** 2)
        return sum((self.a - vec) ** 2)
class Ratkowsky02(Benchmark):
    """
    Ratkowsky02 objective function.

    This class defines the Ratkowsky 2 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::
        f_{\text{Ratkowsky02}}(x) = \sum_{m=1}^{9}(a_m - x[0] / (1 + exp(x[1]
        - b_m x[2]))^2

    where
    
    .. math::
        
        \begin{cases}
        a=[8.93, 10.8, 18.59, 22.33, 39.35, 56.11, 61.73, 64.62, 67.08]\\
        b=[9., 14., 21., 28., 42., 57., 63., 70., 79.]\\
        \end{cases}       
        
        
    Here :math:`x_1 \in [1, 100]`, :math:`x_2 \in [0.1, 5]` and
    :math:`x_3 \in [0.01, 0.5]`

    *Global optimum*: :math:`f(x) = 8.0565229338` for
    :math:`x = [7.2462237576e1, 2.6180768402, 6.7359200066e-2]`

    .. [1] https://www.itl.nist.gov/div898/strd/nls/data/ratkowsky2.shtml
    """

    def __init__(self, dimensions=3):
        # 初始化Benchmark基类
        Benchmark.__init__(self, dimensions)

        # 设定搜索空间边界
        self._bounds = list(zip([10, 0.5, 0.01],
                           [200, 5., 0.5]))
        # 设置全局最优解
        self.global_optimum = [[7.2462237576e1, 2.6180768402, 6.7359200066e-2]]
        # 设置全局最优值
        self.fglob = 8.0565229338
        # 设定参数a
        self.a = asarray([8.93, 10.8, 18.59, 22.33, 39.35, 56.11, 61.73, 64.62,
                          67.08])
        # 设定参数b
        self.b = asarray([9., 14., 21., 28., 42., 57., 63., 70., 79.])

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 计算公式中的分母部分
        vec = x[0] / (1 + exp(x[1] - x[2] * self.b))
        # 计算目标函数值
        return sum((self.a - vec) ** 2)


class Ripple01(Benchmark):
    """
    Ripple 1 objective function.

    This class defines the Ripple 1 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Ripple01}}(x) = \sum_{i=1}^2 -e^{-2 \log 2 
        (\frac{x_i-0.1}{0.8})^2} \left[\sin^6(5 \pi x_i)
        + 0.1\cos^2(500 \pi x_i) \right]


    with :math:`x_i \in [0, 1]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -2.2` for :math:`x_i = 0.1` for
    :math:`i = 1, 2`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 初始化Benchmark基类
        Benchmark.__init__(self, dimensions)
        # 设定搜索空间边界
        self._bounds = list(zip([0.0] * self.N, [1.0] * self.N))
        # 设置全局最优解
        self.global_optimum = [[0.1 for _ in range(self.N)]]
        # 设置全局最优值
        self.fglob = -2.2

    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 计算第一个部分的值
        u = -2.0 * log(2.0) * ((x - 0.1) / 0.8) ** 2.0
        # 计算第二个部分的值
        v = sin(5.0 * pi * x) ** 6.0 + 0.1 * cos(500.0 * pi * x) ** 2.0
        # 计算目标函数值
        return sum(-exp(u) * v)
    """
    .. math::

        f_{\text{Ripple25}}(x) = \sum_{i=1}^2 -e^{-2 
        \log 2 (\frac{x_i-0.1}{0.8})^2}
        \left[\sin^6(5 \pi x_i) \right]

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [0, 1]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = -2` for :math:`x_i = 0.1` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    # 定义一个名为 Ripple25 的函数类，继承自 Benchmark 类
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法
        Benchmark.__init__(self, dimensions)
        # 设置边界为每个维度都是 [0.0, 1.0]
        self._bounds = list(zip([0.0] * self.N, [1.0] * self.N))

        # 设置全局最优解为每个维度都是 0.1
        self.global_optimum = [[0.1 for _ in range(self.N)]]
        # 设置全局最优值为 -2.0
        self.fglob = -2.0

    # 定义函数 fun，计算 Ripple25 函数的值
    def fun(self, x, *args):
        # 增加函数评估计数器
        self.nfev += 1

        # 计算 u 部分的值
        u = -2.0 * log(2.0) * ((x - 0.1) / 0.8) ** 2.0
        # 计算 v 部分的值
        v = sin(5.0 * pi * x) ** 6.0
        # 计算整体函数值并返回
        return sum(-exp(u) * v)
class Rosenbrock(Benchmark):
    r"""
    Rosenbrock objective function.

    This class defines the Rosenbrock [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Rosenbrock}}(x) = \sum_{i=1}^{n-1} [100(x_i^2
       - x_{i+1})^2 + (x_i - 1)^2]

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-5, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 1` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # 调用 Benchmark 类的初始化方法，设置维度
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间边界
        self._bounds = list(zip([-30.] * self.N, [30.0] * self.N))
        # 设置自定义边界
        self.custom_bounds = [(-2, 2), (-2, 2)]

        # 设置全局最优解
        self.global_optimum = [[1 for _ in range(self.N)]]
        # 设置全局最优值
        self.fglob = 0.0
        # 标记是否改变维度的标志位
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估次数
        self.nfev += 1

        # 调用 rosen 函数计算 Rosenbrock 函数值并返回
        return rosen(x)


class RosenbrockModified(Benchmark):
    r"""
    Modified Rosenbrock objective function.

    This class defines the Modified Rosenbrock [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{RosenbrockModified}}(x) = 74 + 100(x_2 - x_1^2)^2
       + (1 - x_1)^2 - 400 e^{-\frac{(x_1+1)^2 + (x_2 + 1)^2}{0.1}}

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-2, 2]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 34.04024310` for
    :math:`x = [-0.90955374, -0.95057172]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: We have different global minimum compared to Jamil #106. This is
    possibly because of the (1-x) term is using the wrong parameter.
    """

    def __init__(self, dimensions=2):
        # 调用 Benchmark 类的初始化方法，设置维度
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间边界
        self._bounds = list(zip([-2.0] * self.N, [2.0] * self.N))
        # 设置自定义边界
        self.custom_bounds = ([-1.0, 0.5], [-1.0, 1.0])

        # 设置全局最优解
        self.global_optimum = [[-0.90955374, -0.95057172]]
        # 设置全局最优值
        self.fglob = 34.040243106640844

    def fun(self, x, *args):
        # 增加函数评估次数
        self.nfev += 1

        # 计算 Modified Rosenbrock 函数值并返回
        a = 74 + 100. * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
        a -= 400 * exp(-((x[0] + 1.) ** 2 + (x[1] + 1.) ** 2) / 0.1)
        return a


class RotatedEllipse01(Benchmark):
    r"""
    Rotated Ellipse 1 objective function.

    This class defines the Rotated Ellipse 1 [1]_ global optimization problem. This
    is a unimodal minimization problem defined as follows:

    .. math::

       f_{\text{RotatedEllipse01}}(x) = 7x_1^2 - 6 \sqrt{3} x_1x_2 + 13x_2^2
    ```
    """
    这部分是一个文档字符串，通常用来描述代码文件的整体内容和相关信息。
    包含了对优化问题的描述、全局最优解的定义、参考文献等信息。
    """

    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，继承父类的维度参数
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间的边界，对每个维度设置范围 [-500.0, 500.0]
        self._bounds = list(zip([-500.0] * self.N,
                           [500.0] * self.N))
        
        # 设置自定义的搜索空间边界，仅限于两个维度，范围为 [-2.0, 2.0]
        self.custom_bounds = ([-2.0, 2.0], [-2.0, 2.0])

        # 设置全局最优解，对于此函数是 [[0.0, 0.0]]
        self.global_optimum = [[0.0, 0.0]]
        
        # 设置全局最优值，函数的全局最优值为 0.0
        self.fglob = 0.0

    def fun(self, x, *args):
        # 每次调用函数 fun()，增加函数评估计数器 self.nfev 的值
        self.nfev += 1

        # 返回 benchmark 函数的计算结果，此处是具体的优化函数表达式
        return (7.0 * x[0] ** 2.0 - 6.0 * sqrt(3) * x[0] * x[1]
                + 13 * x[1] ** 2.0)
# 继承 Benchmark 类，实现 Rotated Ellipse 2 目标函数的类
class RotatedEllipse02(Benchmark):

    r"""
    Rotated Ellipse 2 objective function.

    This class defines the Rotated Ellipse 2 [1]_ global optimization problem. This
    is a unimodal minimization problem defined as follows:

    .. math::

       f_{\text{RotatedEllipse02}}(x) = x_1^2 - x_1 x_2 + x_2^2

    with :math:`x_i \in [-500, 500]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    # 初始化函数，设置维度为 2
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法
        Benchmark.__init__(self, dimensions)

        # 定义搜索空间的边界为 [-500, 500] x [-500, 500]
        self._bounds = list(zip([-500.0] * self.N,
                           [500.0] * self.N))
        
        # 自定义的边界，限制在 [-2, 2] x [-2, 2]
        self.custom_bounds = ([-2.0, 2.0], [-2.0, 2.0])

        # 全局最优解为 [0, 0]
        self.global_optimum = [[0.0, 0.0]]
        
        # 全局最优解对应的函数值为 0
        self.fglob = 0.0

    # 目标函数 fun，计算 Rotated Ellipse 2 函数的值
    def fun(self, x, *args):
        # 增加函数评估次数计数
        self.nfev += 1

        # 计算 Rotated Ellipse 2 函数的值
        return x[0] ** 2.0 - x[0] * x[1] + x[1] ** 2.0
```