# `D:\src\scipysrc\scipy\benchmarks\benchmarks\go_benchmark_functions\go_funcs_N.py`

```
from numpy import cos, sqrt, sin, abs  # 导入 numpy 库中的 cos、sqrt、sin、abs 函数
from .go_benchmark import Benchmark  # 导入当前目录下的 go_benchmark 模块中的 Benchmark 类


class NeedleEye(Benchmark):  # 定义 NeedleEye 类，继承自 Benchmark 类

    r"""
    NeedleEye objective function.

    This class defines the Needle-Eye [1]_ global optimization problem. This is a
    a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{NeedleEye}}(x) =
            \begin{cases}
            1 & \textrm{if }\hspace{5pt} \lvert x_i \rvert  <  eye \hspace{5pt}
            \forall i \\
            \sum_{i=1}^n (100 + \lvert x_i \rvert) & \textrm{if } \hspace{5pt}
            \lvert x_i \rvert > eye \\
            0 & \textrm{otherwise}\\
            \end{cases}


    Where, in this exercise, :math:`eye = 0.0001`.

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 1` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Gavana, A. Global Optimization Benchmarks and AMPGO retrieved 2015
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)  # 调用父类 Benchmark 的初始化方法

        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))  # 设置搜索空间的边界，每个维度为 [-10.0, 10.0]

        self.global_optimum = [[0.0 for _ in range(self.N)]]  # 全局最优解，每个维度为 0.0
        self.fglob = 1.0  # 全局最小值为 1.0
        self.change_dimensionality = True  # 标志位，表示可以改变维度


    def fun(self, x, *args):
        self.nfev += 1  # 增加函数评估次数计数器

        f = fp = 0.0  # 初始化目标函数值和一个辅助标志
        eye = 0.0001  # 定义阈值 eye 为 0.0001

        for val in x:  # 遍历输入的向量 x 中的每个元素
            if abs(val) >= eye:  # 如果元素的绝对值大于等于阈值 eye
                fp = 1.0  # 设置辅助标志为 1.0
                f += 100.0 + abs(val)  # 目标函数值增加 100 + |val|
            else:
                f += 1.0  # 否则，目标函数值增加 1.0

        if fp < 1e-6:  # 如果辅助标志 fp 小于 1e-6
            f = f / self.N  # 将目标函数值除以维度数 self.N

        return f  # 返回计算得到的目标函数值


class NewFunction01(Benchmark):  # 定义 NewFunction01 类，继承自 Benchmark 类

    r"""
    NewFunction01 objective function.

    This class defines the NewFunction01 [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

       f_{\text{NewFunction01}}(x) = \left | {\cos\left(\sqrt{\left|{x_{1}^{2}
       + x_{2}}\right|}\right)} \right |^{0.5} + (x_{1} + x_{2})/100


    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -0.18459899925` for
    :math:`x = [-8.46669057, -9.99982177]`

    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005

    TODO line 355
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)  # 调用父类 Benchmark 的初始化方法

        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))  # 设置搜索空间的边界，每个维度为 [-10.0, 10.0]

        self.global_optimum = [[-8.46668984648, -9.99980944557]]  # 全局最优解
        self.fglob = -0.184648852475  # 全局最小值

    def fun(self, x, *args):
        self.nfev += 1  # 增加函数评估次数计数器

        return ((abs(cos(sqrt(abs(x[0] ** 2 + x[1]))))) ** 0.5
                + 0.01 * (x[0] + x[1]))  # 计算并返回目标函数值


class NewFunction02(Benchmark):  # 定义 NewFunction02 类，继承自 Benchmark 类

    r"""
    NewFunction02 objective function.

    This class defines the NewFunction02 global optimization problem. This is a
    multimodal minimization problem defined as follows:
    """
    .. math::

       f_{\text{NewFunction02}}(x) = \left | {\sin\left(\sqrt{\lvert{x_{1}^{2}
       + x_{2}}\rvert}\right)} \right |^{0.5} + (x_{1} + x_{2})/100

    定义了一个数学公式作为目标函数 f_{\text{NewFunction02}}(x)，用于优化

    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    定义了变量 x_i 的取值范围为 [-10, 10]，其中 i = 1, 2

    *Global optimum*: :math:`f(x) = -0.19933159253` for
    :math:`x = [-9.94103375, -9.99771235]`

    给出了全局最优解的数值和对应的变量取值

    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005

    引用了文献 [1]，介绍了一些优化方法

    TODO Line 368
    标记需要关注的代码行 368

    TODO WARNING, minimum value is estimated from running many optimisations and
    choosing the best.
    给出警告，最小值估计基于多次优化运行并选择最佳结果得出

    """

    # 定义一个 Benchmark 类的子类，初始化函数
    def __init__(self, dimensions=2):
        # 调用父类 Benchmark 的初始化函数
        Benchmark.__init__(self, dimensions)

        # 设置变量的取值范围
        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))

        # 设置全局最优解的变量取值
        self.global_optimum = [[-9.94114736324, -9.99997128772]]

        # 设置全局最优解的目标函数值
        self.fglob = -0.199409030092

    # 定义目标函数 fun，接受变量 x 和额外参数 args
    def fun(self, x, *args):
        # 增加函数评估次数计数器
        self.nfev += 1

        # 计算并返回目标函数的值
        return ((abs(sin(sqrt(abs(x[0] ** 2 + x[1]))))) ** 0.5
                + 0.01 * (x[0] + x[1]))
# 定义一个名为 "Mishra05" 的新函数，其来源是 Gavana 的第三个函数
```