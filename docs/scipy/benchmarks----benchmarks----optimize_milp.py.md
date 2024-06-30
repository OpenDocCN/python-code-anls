# `D:\src\scipysrc\scipy\benchmarks\benchmarks\optimize_milp.py`

```
# 导入必要的模块
import os  # 导入操作系统模块

import numpy as np  # 导入NumPy库
from numpy.testing import assert_allclose  # 导入NumPy测试模块中的断言函数

from .common import Benchmark, safe_import  # 从当前包中导入Benchmark类和safe_import函数

# 使用safe_import上下文管理器安全导入scipy.optimize.milp模块
with safe_import():
    from scipy.optimize import milp

# 使用safe_import上下文管理器安全导入scipy.optimize.tests.test_linprog中的magic_square函数
with safe_import():
    from scipy.optimize.tests.test_linprog import magic_square

# MIPLIB 2017 benchmarks included with permission of the authors
# The MIPLIB benchmark problem set was downloaded from https://miplib.zib.de/.
# An MPS converter (scikit-glpk) was used to load the data into Python. The
# arrays were arranged to the format required by `milp` and saved to `npz`
# format using `np.savez`.
# 定义MIPLIB 2017基准测试问题列表
milp_problems = ["piperout-27"]


class MilpMiplibBenchmarks(Benchmark):
    # 参数化基准测试用例，参数为milp_problems列表中的问题
    params = [milp_problems]
    param_names = ['problem']

    def setup(self, prob):
        # 如果self对象没有data属性，则加载基准测试数据
        if not hasattr(self, 'data'):
            # 获取当前文件所在目录的路径
            dir_path = os.path.dirname(os.path.realpath(__file__))
            # 构建数据文件的完整路径
            datafile = os.path.join(dir_path, "linprog_benchmark_files",
                                    "milp_benchmarks.npz")
            # 使用NumPy加载数据文件，允许使用pickle加载
            self.data = np.load(datafile, allow_pickle=True)

        # 从self.data中获取指定问题prob的数据
        c, A_ub, b_ub, A_eq, b_eq, bounds, integrality = self.data[prob]

        # 从bounds中提取下界和上界
        lb = [li for li, ui in bounds]
        ub = [ui for li, ui in bounds]

        # 初始化约束列表
        cons = []
        # 如果A_ub不为None，则添加不等式约束(A_ub, -inf, b_ub)
        if A_ub is not None:
            cons.append((A_ub, -np.inf, b_ub))
        # 如果A_eq不为None，则添加等式约束(A_eq, b_eq, b_eq)
        if A_eq is not None:
            cons.append((A_eq, b_eq, b_eq))

        # 将数据设置为实例属性
        self.c = c
        self.constraints = cons
        self.bounds = (lb, ub)
        self.integrality = integrality

    def time_milp(self, prob):
        # TODO: fix this benchmark (timing out in Aug. 2023)
        # res = milp(c=self.c, constraints=self.constraints, bounds=self.bounds,
        #           integrality=self.integrality)
        # assert res.success
        pass


class MilpMagicSquare(Benchmark):
    # 参数化基准测试用例，参数为size列表中的元素
    params = [[3, 4, 5]]
    param_names = ['size']

    def setup(self, n):
        # 调用magic_square函数获取方阵问题的等式约束和目标系数
        A_eq, b_eq, self.c, self.numbers, self.M = magic_square(n)
        # 设置等式约束为实例属性
        self.constraints = (A_eq, b_eq, b_eq)

    def time_magic_square(self, n):
        # 调用milp函数求解方阵问题的最优解
        res = milp(c=self.c*0, constraints=self.constraints,
                   bounds=(0, 1), integrality=True)
        # 断言求解成功
        assert res.status == 0
        # 对解向量进行四舍五入
        x = np.round(res.x)
        # 计算方阵问题的结果
        s = (self.numbers.flatten() * x).reshape(n**2, n, n)
        square = np.sum(s, axis=0)
        # 断言方阵每行、每列、对角线的和都等于M
        assert_allclose(square.sum(axis=0), self.M)
        assert_allclose(square.sum(axis=1), self.M)
        assert_allclose(np.diag(square).sum(), self.M)
        assert_allclose(np.diag(square[:, ::-1]).sum(), self.M)
```