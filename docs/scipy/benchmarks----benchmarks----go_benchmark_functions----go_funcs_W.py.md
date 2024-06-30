# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_W.py`

```
from numpy import atleast_2d, arange, sum, cos, exp, pi
from .go_benchmark import Benchmark

class Watson(Benchmark):
    r"""
    Watson objective function.

    This class defines the Watson [1]_ global optimization problem. This is a
    unimodal minimization problem defined as follows:

    .. math::

        f_{\text{Watson}}(x) = \sum_{i=0}^{29} \left\{
                               \sum_{j=0}^4 ((j + 1)a_i^j x_{j+1})
                               - \left[ \sum_{j=0}^5 a_i^j
                               x_{j+1} \right ]^2 - 1 \right\}^2
                               + x_1^2

    Where, in this exercise, :math:`a_i = i/29`.

    with :math:`x_i \in [-5, 5]` for :math:`i = 1, ..., 6`.

    *Global optimum*: :math:`f(x) = 0.002288` for
    :math:`x = [-0.0158, 1.012, -0.2329, 1.260, -1.513, 0.9928]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Jamil #161 writes equation using (j - 1).  According to code in Adorio
    and Gavana it should be (j+1). However the equations in those papers
    contain (j - 1) as well.  However, I've got the right global minimum!!!
    """

    def __init__(self, dimensions=6):
        Benchmark.__init__(self, dimensions)

        # 设置变量_bounds为一个元组列表，定义了每个维度的取值范围[-5.0, 5.0]
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))

        # 定义全局最优解和其对应的函数值
        self.global_optimum = [[-0.0158, 1.012, -0.2329, 1.260, -1.513,
                                0.9928]]
        self.fglob = 0.002288

    # 定义函数fun，计算Watson函数在给定的点x处的函数值
    def fun(self, x, *args):
        # 自增函数评估计数器
        self.nfev += 1

        # 生成30行1列的二维数组，每行都是从0到29的整数
        i = atleast_2d(arange(30.)).T
        # 根据i计算每个元素a_i = i / 29
        a = i / 29.
        # 生成一个包含5个元素的一维数组，值为0到4的整数
        j = arange(5.)
        # 生成一个包含6个元素的一维数组，值为0到5的整数
        k = arange(6.)

        # 计算Watson函数的第一部分
        t1 = sum((j + 1) * a ** j * x[1:], axis=1)
        # 计算Watson函数的第二部分
        t2 = sum(a ** k * x, axis=1)

        # 计算内部表达式 (t1 - t2^2 - 1)^2
        inner = (t1 - t2 ** 2 - 1) ** 2

        # 返回Watson函数的最终结果
        return sum(inner) + x[0] ** 2


class Wavy(Benchmark):
    r"""
    Wavy objective function.

    This class defines the W / Wavy [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Wavy}}(x) = 1 - \frac{1}{n} \sum_{i=1}^{n}
                             \cos(kx_i)e^{-\frac{x_i^2}{2}}

    Where, in this exercise, :math:`k = 10`. The number of local minima is
    :math:`kn` and :math:`(k + 1)n` for odd and even :math:`k` respectively.

    Here, :math:`x_i \in [-\pi, \pi]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        # 设置变量_bounds为一个元组列表，定义了每个维度的取值范围[-π, π]
        self._bounds = list(zip([-pi] * self.N, [pi] * self.N))

        # 定义全局最优解和其对应的函数值
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True
    # 定义一个方法 `fun`，接受 `self`，一个参数 `x` 和任意数量的额外参数 `args`
    def fun(self, x, *args):
        # 将对象的属性 `nfev` 增加 1
        self.nfev += 1

        # 返回一个数值，计算方式为 1.0 减去以下表达式的结果：
        # (1.0 / self.N) 乘以 cos(10 * x) 和 exp(-x ** 2.0 / 2.0) 函数在 x 上的和
        return 1.0 - (1.0 / self.N) * sum(cos(10 * x) * exp(-x ** 2.0 / 2.0))
class WayburnSeader01(Benchmark):
    """
    Wayburn and Seader 1 objective function.

    This class defines the Wayburn and Seader 1 global optimization
    problem. This is a unimodal minimization problem defined as follows:

    .. math::

        f_{\text{WayburnSeader01}}(x) = (x_1^6 + x_2^4 - 17)^2
                                        + (2x_1 + x_2 - 4)^2


    with :math:`x_i \in [-5, 5]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [1, 2]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # Initialize the benchmark superclass with specified dimensions
        Benchmark.__init__(self, dimensions)

        # Define the search space boundaries for each dimension
        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))

        # Define custom bounds for specific scenarios
        self.custom_bounds = ([-2, 2], [-2, 2])

        # Define the global optimum location(s)
        self.global_optimum = [[1.0, 2.0]]

        # Define the global optimum function value
        self.fglob = 0.0

    def fun(self, x, *args):
        # Increment the function evaluation counter
        self.nfev += 1

        # Calculate the Wayburn and Seader 1 objective function value
        return (x[0] ** 6 + x[1] ** 4 - 17) ** 2 + (2 * x[0] + x[1] - 4) ** 2


class WayburnSeader02(Benchmark):
    """
    Wayburn and Seader 2 objective function.

    This class defines the Wayburn and Seader 2 global optimization
    problem. This is a unimodal minimization problem defined as follows:

    .. math::

        f_{\text{WayburnSeader02}}(x) = \left[ 1.613 - 4(x_1 - 0.3125)^2
                                        - 4(x_2 - 1.625)^2 \right]^2
                                        + (x_2 - 1)^2


    with :math:`x_i \in [-500, 500]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0.2, 1]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        # Initialize the benchmark superclass with specified dimensions
        Benchmark.__init__(self, dimensions)

        # Define the search space boundaries for each dimension
        self._bounds = list(zip([-500.0] * self.N, [500.0] * self.N))

        # Define custom bounds for specific scenarios
        self.custom_bounds = ([-1, 2], [-1, 2])

        # Define the global optimum location(s)
        self.global_optimum = [[0.2, 1.0]]

        # Define the global optimum function value
        self.fglob = 0.0

    def fun(self, x, *args):
        # Increment the function evaluation counter
        self.nfev += 1

        # Calculate the Wayburn and Seader 2 objective function value
        u = (1.613 - 4 * (x[0] - 0.3125) ** 2 - 4 * (x[1] - 1.625) ** 2) ** 2
        v = (x[1] - 1) ** 2
        return u + v


class Weierstrass(Benchmark):
    """
    Weierstrass objective function.

    This class defines the Weierstrass global optimization problem.
    This is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Weierstrass}}(x) = \sum_{i=1}^{n} \left [
                                   \sum_{k=0}^{kmax} a^k \cos 
                                   \left( 2 \pi b^k (x_i + 0.5) \right) - n
                                   \sum_{k=0}^{kmax} a^k \cos(\pi b^k) \right ]


    Where, in this exercise, :math:`kmax = 20`, :math:`a = 0.5` and
    :math:`b = 3`.
    """
    """
    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-0.5, 0.5]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 4` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005

    TODO line 1591.
    TODO Jamil, Gavana have got it wrong.  The second term is not supposed to
    be included in the outer sum. Mishra code has it right as does the
    reference referred to in Jamil#166.
    """

    # 定义一个类，继承自Benchmark类，用于多维优化问题的基准函数
    def __init__(self, dimensions=2):
        # 调用父类Benchmark的初始化方法
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间的边界，每个维度的取值范围为[-0.5, 0.5]
        self._bounds = list(zip([-0.5] * self.N, [0.5] * self.N))

        # 全局最优解为所有维度上都是0的情况，对应的函数值为4
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = 0  # 全局最优值的函数值设为0
        self.change_dimensionality = True  # 标记是否改变维度的布尔值

    # 定义评价函数fun，计算给定输入x的函数值
    def fun(self, x, *args):
        self.nfev += 1  # 每调用一次fun函数，增加一次函数评估计数器nfev

        kmax = 20
        a, b = 0.5, 3.0

        # 生成一个列向量k，包含从0到kmax的整数
        k = atleast_2d(arange(kmax + 1.)).T
        # 计算第一个部分的值，包含a的幂次、b的幂次、和余弦函数的计算
        t1 = a ** k * cos(2 * pi * b ** k * (x + 0.5))
        # 计算第二个部分的值，这是一个标量，用于后续的计算
        t2 = self.N * sum(a ** k.T * cos(pi * b ** k.T))

        # 返回函数值，将两部分相加并减去第二部分的值
        return sum(sum(t1, axis=0)) - t2
class Whitley(Benchmark):
    """
    Whitley objective function.

    This class defines the Whitley [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Whitley}}(x) = \sum_{i=1}^n \sum_{j=1}^n
                                \left[\frac{(100(x_i^2-x_j)^2
                                + (1-x_j)^2)^2}{4000} - \cos(100(x_i^2-x_j)^2
                                + (1-x_j)^2)+1 \right]


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10.24, 10.24]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 1` for
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015

    TODO Jamil#167 has '+ 1' inside the cos term, when it should be outside it.
    """

    def __init__(self, dimensions=2):
        # 调用 Benchmark 类的构造函数初始化维度
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间边界
        self._bounds = list(zip([-10.24] * self.N,
                           [10.24] * self.N))
        # 自定义的边界设置
        self.custom_bounds = ([-1, 2], [-1, 2])

        # 全局最优解
        self.global_optimum = [[1.0 for _ in range(self.N)]]
        # 全局最优值
        self.fglob = 0.0
        # 改变维度标志
        self.change_dimensionality = True

    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        XI = x
        XJ = atleast_2d(x).T

        # 计算临时变量
        temp = 100.0 * ((XI ** 2.0) - XJ) + (1.0 - XJ) ** 2.0
        # 计算内部函数
        inner = (temp ** 2.0 / 4000.0) - cos(temp) + 1.0
        # 返回内部函数的总和
        return sum(sum(inner, axis=0))


class Wolfe(Benchmark):
    """
    Wolfe objective function.

    This class defines the Wolfe [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

       f_{\text{Wolfe}}(x) = \frac{4}{3}(x_1^2 + x_2^2 - x_1x_2)^{0.75} + x_3


    with :math:`x_i \in [0, 2]` for :math:`i = 1, 2, 3`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=3):
        # 调用 Benchmark 类的构造函数初始化维度
        Benchmark.__init__(self, dimensions)

        # 设置搜索空间边界
        self._bounds = list(zip([0.0] * self.N, [2.0] * self.N))

        # 设置全局最优解
        self.global_optimum = [[0.0 for _ in range(self.N)]]
        # 设置全局最优值
        self.fglob = 0.0

    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 计算目标函数
        return 4 / 3 * (x[0] ** 2 + x[1] ** 2 - x[0] * x[1]) ** 0.75 + x[2]
```