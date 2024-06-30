# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_H.py`

```
# 导入numpy库，简化代码中对其函数和类的引用
import numpy as np
# 从numpy库中导入特定函数，例如abs, arctan2等
from numpy import abs, arctan2, asarray, cos, exp, arange, pi, sin, sqrt, sum
# 从go_benchmark模块中导入Benchmark类
from .go_benchmark import Benchmark

# Hansen类，继承自Benchmark类，表示Hansen目标函数
class Hansen(Benchmark):

    r"""
    Hansen objective function.

    This class defines the Hansen [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Hansen}}(x) = \left[ \sum_{i=0}^4(i+1)\cos(ix_1+i+1)\right ]
        \left[\sum_{j=0}^4(j+1)\cos[(j+2)x_2+j+1])\right ]


    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -176.54179` for
    :math:`x = [-7.58989583, -7.70831466]`.

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Jamil #61 is missing the starting value of i.
    """

    # 初始化方法，设定维度为2
    def __init__(self, dimensions=2):
        # 调用父类Benchmark的初始化方法
        Benchmark.__init__(self, dimensions)
        
        # 设定变量范围为[-10.0, 10.0]
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))

        # 全局最优解
        self.global_optimum = [[-7.58989583, -7.70831466]]
        # 全局最优值
        self.fglob = -176.54179

    # 目标函数方法，计算Hansen函数值
    def fun(self, x, *args):
        # 计算函数评估次数
        self.nfev += 1
        
        # 构建数组i，包含[0, 1, 2, 3, 4]
        i = arange(5.)
        # 计算a部分的值
        a = (i + 1) * cos(i * x[0] + i + 1)
        # 计算b部分的值
        b = (i + 1) * cos((i + 2) * x[1] + i + 1)

        # 返回函数值，为a和b的乘积的和
        return sum(a) * sum(b)


class Hartmann3(Benchmark):

    r"""
    Hartmann3 objective function.

    This class defines the Hartmann3 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Hartmann3}}(x) = -\sum\limits_{i=1}^{4} c_i 
        e^{-\sum\limits_{j=1}^{n}a_{ij}(x_j - p_{ij})^2}


    Where, in this exercise:

    .. math::

        \begin{array}{l|ccc|c|ccr}
        \hline
        i & & a_{ij}&  & c_i & & p_{ij} &  \\
        \hline
        1 & 3.0 & 10.0 & 30.0 & 1.0 & 0.3689  & 0.1170 & 0.2673 \\
        2 & 0.1 & 10.0 & 35.0 & 1.2 & 0.4699 & 0.4387 & 0.7470 \\
        3 & 3.0 & 10.0 & 30.0 & 3.0 & 0.1091 & 0.8732 & 0.5547 \\
        4 & 0.1 & 10.0 & 35.0 & 3.2 & 0.03815 & 0.5743 & 0.8828 \\
        \hline
        \end{array}


    with :math:`x_i \in [0, 1]` for :math:`i = 1, 2, 3`.

    *Global optimum*: :math:`f(x) = -3.8627821478` 
    for :math:`x = [0.11461292,  0.55564907,  0.85254697]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Jamil #62 has an incorrect coefficient. p[1, 1] should be 0.4387
    """
    # 初始化函数，用于设置基准的维度，默认为三维
    def __init__(self, dimensions=3):
        # 调用 Benchmark 类的初始化函数，传入维度参数
        Benchmark.__init__(self, dimensions)
    
        # 设置搜索空间的边界，默认为每个维度范围从 0.0 到 1.0
        self._bounds = list(zip([0.0] * self.N, [1.0] * self.N))
    
        # 设置全局最优解，这里给出一个三维的坐标点
        self.global_optimum = [[0.11461292, 0.55564907, 0.85254697]]
        
        # 设置全局最优解对应的函数值
        self.fglob = -3.8627821478
    
        # 设置控制函数的系数矩阵 a，是一个 4x3 的数组
        self.a = asarray([[3.0, 10., 30.],
                          [0.1, 10., 35.],
                          [3.0, 10., 30.],
                          [0.1, 10., 35.]])
    
        # 设置控制函数的参数矩阵 p，是一个 4x3 的数组
        self.p = asarray([[0.3689, 0.1170, 0.2673],
                          [0.4699, 0.4387, 0.7470],
                          [0.1091, 0.8732, 0.5547],
                          [0.03815, 0.5743, 0.8828]])
    
        # 设置控制函数的系数 c，是一个长度为 4 的数组
        self.c = asarray([1., 1.2, 3., 3.2])
    
    # 定义函数 fun，用于计算给定参数 x 下的目标函数值
    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1
    
        # 将输入的 x 转换为至少是二维的数组
        XX = np.atleast_2d(x)
        
        # 计算控制函数中的距离度量 d
        d = sum(self.a * (XX - self.p) ** 2, axis=1)
        
        # 计算目标函数值，是对控制函数值的加权和的负数
        return -sum(self.c * exp(-d))
class Hartmann6(Benchmark):
    r"""
    Hartmann6 objective function.

    This class defines the Hartmann6 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Hartmann6}}(x) = -\sum\limits_{i=1}^{4} c_i
        e^{-\sum\limits_{j=1}^{n}a_{ij}(x_j - p_{ij})^2}

    Where, in this exercise:

    .. math::

        \begin{array}{l|cccccc|r}
        \hline
        i & &   &   a_{ij} &  &  & & c_i  \\
        \hline
        1 & 10.0  & 3.0  & 17.0 & 3.50  & 1.70  & 8.00  & 1.0 \\
        2 & 0.05  & 10.0 & 17.0 & 0.10  & 8.00  & 14.00 & 1.2 \\
        3 & 3.00  & 3.50 & 1.70 & 10.0  & 17.00 & 8.00  & 3.0 \\
        4 & 17.00 & 8.00 & 0.05 & 10.00 & 0.10  & 14.00 & 3.2 \\
        \hline
        \end{array}

        \newline
        \
        \newline

        \begin{array}{l|cccccr}
        \hline
        i &  &   & p_{ij} &  & & \\
        \hline
        1 & 0.1312 & 0.1696 & 0.5569 & 0.0124 & 0.8283 & 0.5886 \\
        2 & 0.2329 & 0.4135 & 0.8307 & 0.3736 & 0.1004 & 0.9991 \\
        3 & 0.2348 & 0.1451 & 0.3522 & 0.2883 & 0.3047 & 0.6650 \\
        4 & 0.4047 & 0.8828 & 0.8732 & 0.5743 & 0.1091 & 0.0381 \\
        \hline
        \end{array}

    with :math:`x_i \in [0, 1]` for :math:`i = 1, ..., 6`.

    *Global optimum*: :math:`f(x_i) = -3.32236801141551` for
    :math:`{x} = [0.20168952, 0.15001069, 0.47687398, 0.27533243, 0.31165162,
    0.65730054]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=6):
        Benchmark.__init__(self, dimensions)

        # 设置变量边界范围，从0到1
        self._bounds = list(zip([0.0] * self.N, [1.0] * self.N))

        # 设置全局最优解
        self.global_optimum = [[0.20168952, 0.15001069, 0.47687398, 0.27533243,
                                0.31165162, 0.65730054]]

        # 设置全局最优函数值
        self.fglob = -3.32236801141551

        # 设置函数中使用的系数矩阵 a
        self.a = asarray([[10., 3., 17., 3.5, 1.7, 8.],
                          [0.05, 10., 17., 0.1, 8., 14.],
                          [3., 3.5, 1.7, 10., 17., 8.],
                          [17., 8., 0.05, 10., 0.1, 14.]])

        # 设置函数中使用的坐标偏移矩阵 p
        self.p = asarray([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                          [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                          [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665],
                          [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])

        # 设置函数中使用的常数向量 c
        self.c = asarray([1.0, 1.2, 3.0, 3.2])

    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 确保 x 至少是一个二维数组
        XX = np.atleast_2d(x)

        # 计算 d 向量，是系数矩阵与坐标偏移矩阵之间的加权平方差
        d = sum(self.a * (XX - self.p) ** 2, axis=1)

        # 计算并返回 Hartmann6 函数的值
        return -sum(self.c * exp(-d))
    """
    .. math::

        f_{\text{HelicalValley}}({x}) = 100{[z-10\Psi(x_1,x_2)]^2
        +(\sqrt{x_1^2+x_2^2}-1)^2}+x_3^2

    Where, in this exercise:

    .. math::

        2\pi\Psi(x,y) =  \begin{cases} \arctan(y/x) & \textrm{for} x > 0 \\
        \pi + \arctan(y/x) & \textrm{for } x < 0 \end{cases}

    with :math:`x_i \in [-100, 100]` for :math:`i = 1, 2, 3`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 0, 0]`

    .. [1] Fletcher, R. & Powell, M. A Rapidly Convergent Descent Method for
    Minimization, Computer Journal, 1963, 62, 163-168

    TODO: Jamil equation is different to original reference. The above paper
    can be obtained from
    http://galton.uchicago.edu/~lekheng/courses/302/classics/
    fletcher-powell.pdf
    """

    # 定义一个类 HelicalValley，继承自 Benchmark 类，用于描述特定的优化问题
    def __init__(self, dimensions=3):
        # 调用父类 Benchmark 的初始化方法
        Benchmark.__init__(self, dimensions)

        # 定义变量 _bounds，表示变量 x 的取值范围为 [-10., 10.]
        self._bounds = list(zip([-10.] * self.N, [10.] * self.N))

        # 设置全局最优解为 [1.0, 0.0, 0.0]
        self.global_optimum = [[1.0, 0.0, 0.0]]

        # 设置全局最优解对应的函数值为 0.0
        self.fglob = 0.0

    # 定义函数 fun，用于计算 Helical Valley 函数的值
    def fun(self, x, *args):
        # 增加函数评估计数器值
        self.nfev += 1

        # 计算极径 r
        r = sqrt(x[0] ** 2 + x[1] ** 2)

        # 计算角度 theta
        theta = 1 / (2. * pi) * arctan2(x[1], x[0])

        # 计算 Helical Valley 函数的值并返回
        return x[2] ** 2 + 100 * ((x[2] - 10 * theta) ** 2 + (r - 1) ** 2)
class HimmelBlau(Benchmark):

    r"""
    HimmelBlau objective function.

    This class defines the HimmelBlau [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{HimmelBlau}}({x}) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2


    with :math:`x_i \in [-6, 6]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [3, 2]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        # 设置变量边界
        self._bounds = list(zip([-5.] * self.N, [5.] * self.N))

        # 设置全局最优解和其函数值
        self.global_optimum = [[3.0, 2.0]]
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估计数
        self.nfev += 1

        # 返回 HimmelBlau 函数的计算结果
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


class HolderTable(Benchmark):

    r"""
    HolderTable objective function.

    This class defines the HolderTable [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{HolderTable}}({x}) = - \left|{e^{\left|{1
        - \frac{\sqrt{x_{1}^{2} + x_{2}^{2}}}{\pi} }\right|}
        \sin\left(x_{1}\right) \cos\left(x_{2}\right)}\right|


    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -19.20850256788675` for
    :math:`x_i = \pm 9.664590028909654` for :math:`i = 1, 2`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO: Jamil #146 equation is wrong - should be squaring the x1 and x2
    terms, but isn't. Gavana does.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        # 设置变量边界
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))

        # 设置全局最优解和其函数值
        self.global_optimum = [(8.055023472141116, 9.664590028909654),
                               (-8.055023472141116, 9.664590028909654),
                               (8.055023472141116, -9.664590028909654),
                               (-8.055023472141116, -9.664590028909654)]
        self.fglob = -19.20850256788675

    def fun(self, x, *args):
        # 增加函数评估计数
        self.nfev += 1

        # 返回 HolderTable 函数的计算结果
        return -abs(sin(x[0]) * cos(x[1])
                    * exp(abs(1 - sqrt(x[0] ** 2 + x[1] ** 2) / pi)))


class Hosaki(Benchmark):

    r"""
    Hosaki objective function.

    This class defines the Hosaki [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Hosaki}}(x) = \left ( 1 - 8 x_1 + 7 x_1^2 - \frac{7}{3} x_1^3
        + \frac{1}{4} x_1^4 \right ) x_2^2 e^{-x_1}


    with :math:`x_i \in [0, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -2.3458115` for :math:`x = [4, 2]`.

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        # 设置变量边界
        self._bounds = list(zip([0.0] * self.N, [10.0] * self.N))

        # 设置全局最优解和其函数值
        self.global_optimum = [[4.0, 2.0]]
        self.fglob = -2.3458115

    def fun(self, x, *args):
        # 增加函数评估计数
        self.nfev += 1

        # 返回 Hosaki 函数的计算结果
        return (1 - 8 * x[0] + 7 * x[0] ** 2 - (7 / 3) * x[0] ** 3
                + (1 / 4) * x[0] ** 4) * x[1] ** 2 * exp(-x[0])
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    # 定义一个 Benchmark 的子类，用于特定优化问题的函数评估
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化方法，设置问题维度
        Benchmark.__init__(self, dimensions)

        # 设置变量边界范围
        self._bounds = ([0., 5.], [0., 6.])
        # 设置自定义的边界范围
        self.custom_bounds = [(0, 5), (0, 5)]

        # 设置全局最优解
        self.global_optimum = [[4, 2]]
        # 设置全局最优解对应的函数值
        self.fglob = -2.3458115

    # 定义优化问题的目标函数
    def fun(self, x, *args):
        # 增加函数评估的计数器
        self.nfev += 1

        # 计算目标函数的值
        val = (1 - 8 * x[0] + 7 * x[0] ** 2 - 7 / 3. * x[0] ** 3
               + 0.25 * x[0] ** 4)
        # 返回目标函数值乘以第二个变量的平方和指数函数的结果
        return val * x[1] ** 2 * exp(-x[1])
```