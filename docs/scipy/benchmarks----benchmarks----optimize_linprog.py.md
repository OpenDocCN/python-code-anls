# `D:\src\scipysrc\scipy\benchmarks\benchmarks\optimize_linprog.py`

```
# 导入操作系统相关模块
import os

# 导入 NumPy 库，并指定别名 np
import numpy as np

# 导入 suppress_warnings 函数，用于在测试中抑制警告
from numpy.testing import suppress_warnings

# 从当前包中导入 Benchmark 类、is_xslow 函数以及 safe_import 函数
from .common import Benchmark, is_xslow, safe_import

# 使用 safe_import 上下文管理器导入 scipy.optimize 中的 linprog 函数和 OptimizeWarning 异常类
with safe_import():
    from scipy.optimize import linprog, OptimizeWarning

# 使用 safe_import 上下文管理器导入 scipy.optimize.tests.test_linprog 中的 lpgen_2d 和 magic_square 函数
with safe_import():
    from scipy.optimize.tests.test_linprog import lpgen_2d, magic_square

# 使用 safe_import 上下文管理器导入 scipy.linalg 中的 toeplitz 函数
with safe_import():
    from scipy.linalg import toeplitz

# 定义 methods 列表，包含两个元组，每个元组表示一个方法的名称和空字典参数
methods = [("highs-ipm", {}),
           ("highs-ds", {})]

# 定义 problems 列表，包含多个字符串，每个字符串表示一个优化问题的名称
problems = ['25FV47', '80BAU3B', 'ADLITTLE', 'AFIRO', 'AGG', 'AGG2', 'AGG3',
            'BANDM', 'BEACONFD', 'BLEND', 'BNL1', 'BNL2', 'BORE3D', 'BRANDY',
            'CAPRI', 'CYCLE', 'CZPROB', 'D2Q06C', 'D6CUBE', 'DEGEN2', 'DEGEN3',
            'DFL001', 'E226', 'ETAMACRO', 'FFFFF800', 'FINNIS', 'FIT1D',
            'FIT1P', 'FIT2D', 'FIT2P', 'GANGES', 'GFRD-PNC', 'GREENBEA',
            'GREENBEB', 'GROW15', 'GROW22', 'GROW7', 'ISRAEL', 'KB2', 'LOTFI',
            'MAROS', 'MAROS-R7', 'MODSZK1', 'PEROLD', 'PILOT', 'PILOT4',
            'PILOT87', 'PILOT-JA', 'PILOTNOV', 'PILOT-WE', 'QAP8', 'QAP12',
            'QAP15', 'RECIPE', 'SC105', 'SC205', 'SC50A', 'SC50B', 'SCAGR25',
            'SCAGR7', 'SCFXM1', 'SCFXM2', 'SCFXM3', 'SCORPION', 'SCRS8',
            'SCSD1', 'SCSD6', 'SCSD8', 'SCTAP1', 'SCTAP2', 'SCTAP3', 'SHARE1B',
            'SHARE2B', 'SHELL', 'SHIP04L', 'SHIP04S', 'SHIP08L', 'SHIP08S',
            'SHIP12L', 'SHIP12S', 'SIERRA', 'STAIR', 'STANDATA', 'STANDMPS',
            'STOCFOR1', 'STOCFOR2', 'STOCFOR3', 'TRUSS', 'TUFF', 'VTP-BASE',
            'WOOD1P', 'WOODW']

# 定义 infeasible_problems 列表，包含多个字符串，每个字符串表示一个不可行问题的名称
infeasible_problems = ['bgdbg1', 'bgetam', 'bgindy', 'bgprtr', 'box1',
                       'ceria3d', 'chemcom', 'cplex1', 'cplex2', 'ex72a',
                       'ex73a', 'forest6', 'galenet', 'gosh', 'gran',
                       'itest2', 'itest6', 'klein1', 'klein2', 'klein3',
                       'mondou2', 'pang', 'pilot4i', 'qual', 'reactor',
                       'refinery', 'vol1', 'woodinfe']

# 根据 is_xslow() 函数的返回值确定 enabled_problems 和 enabled_infeasible_problems 列表的内容
if not is_xslow():
    # 如果 is_xslow() 返回 False，则使用以下列表作为 enabled_problems
    enabled_problems = ['ADLITTLE', 'AFIRO', 'BLEND', 'BEACONFD', 'GROW7',
                        'LOTFI', 'SC105', 'SCTAP1', 'SHARE2B', 'STOCFOR1']
    # 如果 is_xslow() 返回 False，则使用以下列表作为 enabled_infeasible_problems
    enabled_infeasible_problems = ['bgdbg1', 'bgprtr', 'box1', 'chemcom',
                                   'cplex2', 'ex72a', 'ex73a', 'forest6',
                                   'galenet', 'itest2', 'itest6', 'klein1',
                                   'refinery', 'woodinfe']
else:
    # 如果 is_xslow() 返回 True，则使用 problems 和 infeasible_problems 列表作为 enabled_problems 和 enabled_infeasible_problems
    enabled_problems = problems
    enabled_infeasible_problems = infeasible_problems


def klee_minty(D):
    # 构造 Klee-Minty 问题的系数矩阵 A_ub
    A_1 = np.array([2**(i + 1) if i > 0 else 1 for i in range(D)])
    A1_ = np.zeros(D)
    A1_[0] = 1
    A_ub = toeplitz(A_1, A1_)
    
    # 构造 Klee-Minty 问题的不等式约束右侧向量 b_ub
    b_ub = np.array([5**(i + 1) for i in range(D)])
    
    # 构造 Klee-Minty 问题的线性目标函数系数向量 c
    c = -np.array([2**(D - i - 1) for i in range(D)])
    
    # 构造 Klee-Minty 问题的最优解向量 xf
    xf = np.zeros(D)
    xf[-1] = 5**D
    
    # 计算 Klee-Minty 问题的目标函数值 obj
    obj = c @ xf
    
    return c, A_ub, b_ub, xf, obj


# Benchmark 类的一个子类 MagicSquare，包含一组解决方案
class MagicSquare(Benchmark):
    
    # solutions 属性，包含多个元组，每个元组表示一个魔方阵问题的维度和已知的最优解
    solutions = [(3, 1.7305505947214375), (4, 1.5485271031586025),
                 (5, 1.807494583582637), (6, 1.747266446858304)]
    # 初始化参数列表，包括方法和解决方案
    params = [methods, solutions]
    # 定义参数名称列表，分别是方法和问题维度以及目标函数描述
    param_names = ['method', '(dimensions, objective)']

    # 设置方法和问题的初始化函数
    def setup(self, meth, prob):
        # 如果不是在特别慢的环境下运行
        if not is_xslow():
            # 如果问题的第一个维度大于4，则抛出未实现错误
            if prob[0] > 4:
                raise NotImplementedError("skipped")

        # 解包问题的维度和目标
        dims, obj = prob
        # 使用魔法函数生成魔法方的等式约束和其他变量
        self.A_eq, self.b_eq, self.c, numbers, _ = magic_square(dims)
        # 将函数对象置为空
        self.fun = None

    # 测试魔法方函数执行时间的函数
    def time_magic_square(self, meth, prob):
        # 解包方法和选项
        method, options = meth
        # 使用上下文管理器去除警告
        with suppress_warnings() as sup:
            # 过滤优化警告中关于 A_eq 未出现的警告
            sup.filter(OptimizeWarning, "A_eq does not appear")
            # 使用线性规划求解器计算最优化问题
            res = linprog(c=self.c, A_eq=self.A_eq, b_eq=self.b_eq,
                          bounds=(0, 1), method=method, options=options)
            # 保存结果的函数值
            self.fun = res.fun

    # 跟踪魔法方误差的函数
    def track_magic_square(self, meth, prob):
        # 解包问题的维度和目标
        dims, obj = prob
        # 如果函数值为空，则计算时间魔方函数
        if self.fun is None:
            self.time_magic_square(meth, prob)
        # 计算绝对误差和相对误差
        self.abs_error = np.abs(self.fun - obj)
        self.rel_error = np.abs((self.fun - obj)/obj)
        # 返回最小的误差值
        return min(self.abs_error, self.rel_error)
class KleeMinty(Benchmark):
    # 定义一个继承自Benchmark类的KleeMinty类

    params = [
        methods,  # 参数化设置：方法列表，通过外部传入
        [3, 6, 9]  # 参数化设置：维度列表，固定为3, 6, 9
    ]
    param_names = ['method', 'dimensions']  # 参数名称定义：方法和维度

    def setup(self, meth, dims):
        # 初始化函数，设置实例属性，准备测试数据
        self.c, self.A_ub, self.b_ub, self.xf, self.obj = klee_minty(dims)
        self.fun = None  # 初始化fun属性为None

    def time_klee_minty(self, meth, dims):
        # 测试函数：执行Klee-Minty问题求解
        method, options = meth  # 解构方法参数
        res = linprog(c=self.c, A_ub=self.A_ub, b_ub=self.b_ub,
                      method=method, options=options)  # 调用线性规划求解器
        self.fun = res.fun  # 记录求解结果的目标函数值
        self.x = res.x  # 记录求解结果的最优解向量

    def track_klee_minty(self, meth, prob):
        # 跟踪函数：监控Klee-Minty问题求解的误差
        if self.fun is None:
            self.time_klee_minty(meth, prob)  # 如果fun尚未计算，则重新计算
        self.abs_error = np.abs(self.fun - self.obj)  # 计算绝对误差
        self.rel_error = np.abs((self.fun - self.obj)/self.obj)  # 计算相对误差
        return min(self.abs_error, self.rel_error)  # 返回最小误差


class LpGen(Benchmark):
    # 定义一个继承自Benchmark类的LpGen类
    params = [
        methods,  # 参数化设置：方法列表，通过外部传入
        range(20, 100, 20),  # 参数化设置：m的范围从20到80，步长20
        range(20, 100, 20)   # 参数化设置：n的范围从20到80，步长20
    ]
    param_names = ['method', 'm', 'n']  # 参数名称定义：方法、m和n

    def setup(self, meth, m, n):
        # 初始化函数，设置实例属性，准备测试数据
        self.A, self.b, self.c = lpgen_2d(m, n)  # 生成二维线性规划问题的系数矩阵

    def time_lpgen(self, meth, m, n):
        # 测试函数：执行二维线性规划问题求解
        method, options = meth  # 解构方法参数
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "scipy.linalg.solve\nIll-conditioned")
            linprog(c=self.c, A_ub=self.A, b_ub=self.b,
                    method=method, options=options)  # 调用线性规划求解器


class Netlib(Benchmark):
    # 定义一个继承自Benchmark类的Netlib类
    params = [
        methods,  # 参数化设置：方法列表，通过外部传入
        problems  # 参数化设置：问题列表，通过外部传入
    ]
    param_names = ['method', 'problems']  # 参数名称定义：方法和问题

    def setup(self, meth, prob):
        # 初始化函数，设置实例属性，准备测试数据
        if prob not in enabled_problems:
            raise NotImplementedError("skipped")  # 如果问题未启用，则抛出NotImplementedError

        dir_path = os.path.dirname(os.path.realpath(__file__))  # 获取当前文件所在目录路径
        datafile = os.path.join(dir_path, "linprog_benchmark_files",
                                prob + ".npz")  # 构建数据文件的完整路径
        data = np.load(datafile, allow_pickle=True)  # 加载数据文件
        self.c = data["c"]  # 从数据中获取系数向量c
        self.A_eq = data["A_eq"]  # 从数据中获取等式约束矩阵A_eq
        self.A_ub = data["A_ub"]  # 从数据中获取不等式约束矩阵A_ub
        self.b_ub = data["b_ub"]  # 从数据中获取不等式约束向量b_ub
        self.b_eq = data["b_eq"]  # 从数据中获取等式约束向量b_eq
        self.bounds = np.squeeze(data["bounds"])  # 从数据中获取变量边界约束
        self.obj = float(data["obj"].flatten()[0])  # 从数据中获取目标函数的值，并转换为浮点数
        self.fun = None  # 初始化fun属性为None

    def time_netlib(self, meth, prob):
        # 测试函数：执行Netlib问题求解
        method, options = meth  # 解构方法参数
        res = linprog(c=self.c,
                      A_ub=self.A_ub,
                      b_ub=self.b_ub,
                      A_eq=self.A_eq,
                      b_eq=self.b_eq,
                      bounds=self.bounds,
                      method=method,
                      options=options)  # 调用线性规划求解器
        self.fun = res.fun  # 记录求解结果的目标函数值

    def track_netlib(self, meth, prob):
        # 跟踪函数：监控Netlib问题求解的误差
        if self.fun is None:
            self.time_netlib(meth, prob)  # 如果fun尚未计算，则重新计算
        self.abs_error = np.abs(self.fun - self.obj)  # 计算绝对误差
        self.rel_error = np.abs((self.fun - self.obj)/self.obj)  # 计算相对误差
        return min(self.abs_error, self.rel_error)  # 返回最小误差


class Netlib_infeasible(Benchmark):
    # 定义一个继承自Benchmark类的Netlib_infeasible类
    params = [
        methods,  # 参数化设置：方法列表，通过外部传入
        infeasible_problems  # 参数化设置：不可行问题列表，通过外部传入
    ]
    param_names = ['method', 'problems']  # 参数名称定义：方法和问题
    # 设置测试用例的初始化方法，接受方法和问题名称作为参数
    def setup(self, meth, prob):
        # 如果问题名称不在可用的无解问题列表中，则抛出未实现错误
        if prob not in enabled_infeasible_problems:
            raise NotImplementedError("skipped")

        # 获取当前文件所在目录的路径
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # 构建数据文件的完整路径
        datafile = os.path.join(dir_path, "linprog_benchmark_files",
                                "infeasible", prob + ".npz")
        # 加载数据文件，允许使用pickle
        data = np.load(datafile, allow_pickle=True)
        # 初始化测试用例的参数
        self.c = data["c"]
        self.A_eq = data["A_eq"]
        self.A_ub = data["A_ub"]
        self.b_ub = data["b_ub"]
        self.b_eq = data["b_eq"]
        self.bounds = np.squeeze(data["bounds"])
        self.status = None

    # 计算netlib无解问题的运行时间
    def time_netlib_infeasible(self, meth, prob):
        method, options = meth
        # 调用linprog函数计算结果
        res = linprog(c=self.c,
                      A_ub=self.A_ub,
                      b_ub=self.b_ub,
                      A_eq=self.A_eq,
                      b_eq=self.b_eq,
                      bounds=self.bounds,
                      method=method,
                      options=options)
        # 更新测试用例的状态
        self.status = res.status

    # 跟踪netlib无解问题的状态
    def track_netlib_infeasible(self, meth, prob):
        # 如果状态为空，则调用计算时间方法
        if self.status is None:
            self.time_netlib_infeasible(meth, prob)
        # 返回测试用例的状态
        return self.status
```