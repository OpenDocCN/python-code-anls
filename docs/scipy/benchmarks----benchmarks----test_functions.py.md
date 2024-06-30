# `D:\src\scipysrc\scipy\benchmarks\benchmarks\test_functions.py`

```
# 导入时间模块
import time

# 导入 NumPy 库，并从中导入所需函数和常数
import numpy as np
from numpy import sin, cos, pi, exp, sqrt, abs

# 从 SciPy 库中导入 rosen 函数
from scipy.optimize import rosen

# 定义一个简单的二次函数类 SimpleQuadratic
class SimpleQuadratic:

    # 定义函数 fun，计算向量 x 的平方和
    def fun(self, x):
        return np.dot(x, x)

    # 定义函数 der，计算向量 x 的导数，即 2 * x
    def der(self, x):
        return 2. * x

    # 定义函数 hess，返回单位矩阵乘以 2，即 2 * np.eye(x.size)
    def hess(self, x):
        return 2. * np.eye(x.size)


# 定义一个非对称二次函数类 AsymmetricQuadratic
class AsymmetricQuadratic:

    # 定义函数 fun，计算向量 x 的平方和并加上 x[0]
    def fun(self, x):
        return np.dot(x, x) + x[0]

    # 定义函数 der，计算向量 x 的导数，其中第一个元素加上了 1
    def der(self, x):
        d = 2. * x
        d[0] += 1
        return d

    # 定义函数 hess，返回单位矩阵乘以 2，即 2 * np.eye(x.size)
    def hess(self, x):
        return 2. * np.eye(x.size)


# 定义一个较慢的 Rosenbrock 函数类 SlowRosen
class SlowRosen:

    # 定义函数 fun，计算 Rosenbrock 函数在 x 处的值，并在计算过程中休眠 40 微秒
    def fun(self, x):
        time.sleep(40e-6)
        return rosen(x)


# 定义 Lennard-Jones 势能类 LJ
class LJ:
    """
    Lennard-Jones 势能

    一个简单的模型，用于近似描述中性原子或分子之间的相互作用。
    https://en.wikipedia.org/wiki/Lennard-Jones_potential

    E = sum_ij V(r_ij)

    其中 r_ij 是原子 i 和原子 j 之间的笛卡尔距离，势能 V(r) 的形式为

    V(r) = 4 * eps * ( (sigma / r)**12 - (sigma / r)**6

    注意
    -----
    在 Python 中，双重循环处理多个原子使得计算非常缓慢。如果使用编译语言实现，则速度会显著提升。
    """

    # 初始化函数，设定 epsilon (eps) 和 sigma (sig) 的初始值
    def __init__(self, eps=1.0, sig=1.0):
        self.sig = sig
        self.eps = eps

    # 计算 V(r) 的函数 vij
    def vij(self, r):
        return 4. * self.eps * ((self.sig / r)**12 - (self.sig / r)**6)

    # 计算 V(r) 对 r 的导数的函数 dvij
    def dvij(self, r):
        p7 = 6. / self.sig * (self.sig / r)**7
        p13 = -12. / self.sig * (self.sig / r)**13
        return 4. * self.eps * (p7 + p13)

    # 计算势能函数 E 在给定坐标 coords 处的值
    def fun(self, coords):
        natoms = coords.size // 3  # 计算原子数量
        coords = np.reshape(coords, [natoms, 3])  # 将坐标重新整形为原子数乘以3的数组
        energy = 0.  # 初始化能量
        # 双重循环计算每对原子之间的相互作用能量
        for i in range(natoms):
            for j in range(i + 1, natoms):
                dr = coords[j, :] - coords[i, :]  # 计算原子间的距离向量
                r = np.linalg.norm(dr)  # 计算距离的模长
                energy += self.vij(r)  # 累加势能
        return energy  # 返回总能量

    # 计算势能函数 E 在给定坐标 coords 处的梯度
    def der(self, coords):
        natoms = coords.size // 3  # 计算原子数量
        coords = np.reshape(coords, [natoms, 3])  # 将坐标重新整形为原子数乘以3的数组
        energy = 0.  # 初始化能量
        grad = np.zeros([natoms, 3])  # 初始化梯度数组
        # 双重循环计算每对原子之间的相互作用能量和梯度
        for i in range(natoms):
            for j in range(i + 1, natoms):
                dr = coords[j, :] - coords[i, :]  # 计算原子间的距离向量
                r = np.linalg.norm(dr)  # 计算距离的模长
                energy += self.vij(r)  # 累加势能
                g = self.dvij(r)  # 计算梯度分量
                grad[i, :] += -g * dr / r  # 更新第 i 个原子的梯度
                grad[j, :] += g * dr / r  # 更新第 j 个原子的梯度
        grad = grad.reshape([natoms * 3])  # 将梯度数组重新整形为一维数组
        return grad  # 返回梯度数组

    # 获取随机构型的函数
    def get_random_configuration(self):
        rnd = np.random.uniform(-1, 1, [3 * self.natoms])  # 生成均匀分布的随机数数组
        return rnd * float(self.natoms)**(1. / 3)  # 返回随机构型数组，乘以原子数的立方根


# 定义 LJ38 类，继承自 LJ 类，设定原子数和目标能量
class LJ38(LJ):
    natoms = 38
    target_E = -173.928427


# 定义 LJ30 类，继承自 LJ 类，设定原子数和目标能量
class LJ30(LJ):
    natoms = 30
    target_E = -128.286571


# 定义 LJ20 类，继承自 LJ 类，设定原子数和目标能量
class LJ20(LJ):
    natoms = 20
    target_E = -77.177043


# 定义 LJ13 类，继承自 LJ 类，设定原子数和目标能量
class LJ13(LJ):
    natoms = 13
    target_E = -44.326801


# 定义 Booth 类，描述 Booth 函数的目标能量、解和变量范围
class Booth:
    target_E = 0.  # 目标能量为 0
    solution = np.array([1., 3.])  # 解为 [1, 3]
    xmin = np.array([-10., -10.])  # 变量的最小值为 [-10, -10]
    xmax = np.array([10., 10.])  # 变量的最大值为 [10, 10]
    # 定义一个函数 `fun`，计算给定坐标 (x, y) 的目标函数值
    def fun(self, coords):
        # 将坐标拆解为 x 和 y
        x, y = coords
        # 计算目标函数的值并返回
        return (x + 2. * y - 7.)**2 + (2. * x + y - 5.)**2

    # 定义一个函数 `der`，计算给定坐标 (x, y) 的目标函数的梯度
    def der(self, coords):
        # 将坐标拆解为 x 和 y
        x, y = coords
        # 计算目标函数对 x 的偏导数
        dfdx = 2. * (x + 2. * y - 7.) + 4. * (2. * x + y - 5.)
        # 计算目标函数对 y 的偏导数
        dfdy = 4. * (x + 2. * y - 7.) + 2. * (2. * x + y - 5.)
        # 将偏导数组合成一个 NumPy 数组并返回
        return np.array([dfdx, dfdy])
class Beale:
    # 目标能量值
    target_E = 0.
    # 解向量
    solution = np.array([3., 0.5])
    # 变量 x 和 y 的最小值
    xmin = np.array([-4.5, -4.5])
    # 变量 x 和 y 的最大值
    xmax = np.array([4.5, 4.5])

    # 计算 Beale 函数的值
    def fun(self, coords):
        # 提取变量 x 和 y
        x, y = coords
        # 计算 Beale 函数的第一个部分
        p1 = (1.5 - x + x * y)**2
        # 计算 Beale 函数的第二个部分
        p2 = (2.25 - x + x * y**2)**2
        # 计算 Beale 函数的第三个部分
        p3 = (2.625 - x + x * y**3)**2
        # 返回 Beale 函数的值
        return p1 + p2 + p3

    # 计算 Beale 函数关于变量 x 和 y 的导数
    def der(self, coords):
        # 提取变量 x 和 y
        x, y = coords
        # 计算关于 x 的导数
        dfdx = (2. * (1.5 - x + x * y) * (-1. + y) +
                2. * (2.25 - x + x * y**2) * (-1. + y**2) +
                2. * (2.625 - x + x * y**3) * (-1. + y**3))
        # 计算关于 y 的导数
        dfdy = (2. * (1.5 - x + x * y) * (x) +
                2. * (2.25 - x + x * y**2) * (2. * y * x) +
                2. * (2.625 - x + x * y**3) * (3. * x * y**2))
        # 返回导数向量
        return np.array([dfdx, dfdy])


"""
全局测试函数用于最小化器。

HolderTable、Ackey 和 Levi 函数具有许多竞争的局部最小值，适合于
基于全局优化的最小化器，如 basinhopping 或 differential_evolution。
(https://en.wikipedia.org/wiki/Test_functions_for_optimization)

另请参阅 https://mpra.ub.uni-muenchen.de/2718/1/MPRA_paper_2718.pdf
"""


class HolderTable:
    # 目标能量值
    target_E = -19.2085
    # 解向量
    solution = [8.05502, 9.66459]
    # 变量 x 和 y 的最小值
    xmin = np.array([-10, -10])
    # 变量 x 和 y 的最大值
    xmax = np.array([10, 10])
    # 步长
    stepsize = 2.
    # 温度
    temperature = 2.

    # 计算 HolderTable 函数的值
    def fun(self, x):
        return - abs(sin(x[0]) * cos(x[1]) * exp(abs(1. - sqrt(x[0]**2 +
                     x[1]**2) / pi)))

    # 绝对值函数的导数
    def dabs(self, x):
        """绝对值函数的导数"""
        if x < 0:
            return -1.
        elif x > 0:
            return 1.
        else:
            return 0.


class Ackley:
    # 注意：此函数在原点处不光滑。最小化器中梯度永远不会收敛
    target_E = 0.
    solution = [0., 0.]
    xmin = np.array([-5, -5])
    xmax = np.array([5, 5])

    # 计算 Ackley 函数的值
    def fun(self, x):
        E = (-20. * exp(-0.2 * sqrt(0.5 * (x[0]**2 + x[1]**2))) + 20. + np.e -
             exp(0.5 * (cos(2. * pi * x[0]) + cos(2. * pi * x[1]))))
        return E
    `
    # 计算输入向量 x 的梯度
    def der(self, x):
        # 计算点到原点的距离 R
        R = sqrt(x[0]**2 + x[1]**2)
        # 计算第一个项的导数
        term1 = -20. * exp(-0.2 * R)
        # 计算第二个项的导数
        term2 = -exp(0.5 * (cos(2. * pi * x[0]) + cos(2. * pi * x[1])))
    
        # 计算对 x[0] 的偏导数
        deriv1 = term1 * (-0.2 * 0.5 / R)
    
        # 计算对 x[0] 和 x[1] 的总体梯度
        dfdx = 2. * deriv1 * x[0] - term2 * pi * sin(2. * pi * x[0])
        dfdy = 2. * deriv1 * x[1] - term2 * pi * sin(2. * pi * x[1])
    
        # 返回梯度向量
        return np.array([dfdx, dfdy])
class Levi:
    # Levi函数的目标能量值
    target_E = 0.
    # Levi函数的已知解
    solution = [1., 1.]
    # Levi函数的自变量取值范围下界
    xmin = np.array([-10, -10])
    # Levi函数的自变量取值范围上界
    xmax = np.array([10, 10])

    def fun(self, x):
        # 计算Levi函数的能量值
        E = (sin(3. * pi * x[0])**2 + (x[0] - 1.)**2 *
             (1. + sin(3 * pi * x[1])**2) +
             (x[1] - 1.)**2 * (1. + sin(2 * pi * x[1])**2))
        return E

    def der(self, x):
        # 计算Levi函数对自变量x的导数
        dfdx = (2. * 3. * pi *
                cos(3. * pi * x[0]) * sin(3. * pi * x[0]) +
                2. * (x[0] - 1.) * (1. + sin(3 * pi * x[1])**2))

        dfdy = ((x[0] - 1.)**2 * 2. * 3. * pi * cos(3. * pi * x[1]) * sin(3. *
                pi * x[1]) + 2. * (x[1] - 1.) *
                (1. + sin(2 * pi * x[1])**2) + (x[1] - 1.)**2 *
                2. * 2. * pi * cos(2. * pi * x[1]) * sin(2. * pi * x[1]))

        return np.array([dfdx, dfdy])


class EggHolder:
    # EggHolder函数的目标能量值
    target_E = -959.6407
    # EggHolder函数的已知解
    solution = [512, 404.2319]
    # EggHolder函数的自变量取值范围下界
    xmin = np.array([-512., -512])
    # EggHolder函数的自变量取值范围上界
    xmax = np.array([512., 512])

    def fun(self, x):
        # 计算EggHolder函数的能量值
        a = -(x[1] + 47) * np.sin(np.sqrt(abs(x[1] + x[0]/2. + 47)))
        b = -x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47))))
        return a + b


class CrossInTray:
    # CrossInTray函数的目标能量值
    target_E = -2.06261
    # CrossInTray函数的已知解
    solution = [1.34941, -1.34941]
    # CrossInTray函数的自变量取值范围下界
    xmin = np.array([-10., -10])
    # CrossInTray函数的自变量取值范围上界
    xmax = np.array([10., 10])

    def fun(self, x):
        # 计算CrossInTray函数的能量值
        arg = abs(100 - sqrt(x[0]**2 + x[1]**2)/pi)
        val = np.power(abs(sin(x[0]) * sin(x[1]) * exp(arg)) + 1., 0.1)
        return -0.0001 * val


class Schaffer2:
    # Schaffer2函数的目标能量值
    target_E = 0
    # Schaffer2函数的已知解
    solution = [0., 0.]
    # Schaffer2函数的自变量取值范围下界
    xmin = np.array([-100., -100])
    # Schaffer2函数的自变量取值范围上界
    xmax = np.array([100., 100])

    def fun(self, x):
        # 计算Schaffer2函数的能量值
        num = np.power(np.sin(x[0]**2 - x[1]**2), 2) - 0.5
        den = np.power(1 + 0.001 * (x[0]**2 + x[1]**2), 2)
        return 0.5 + num / den


class Schaffer4:
    # Schaffer4函数的目标能量值
    target_E = 0.292579
    # Schaffer4函数的已知解
    solution = [0, 1.253131828927371]
    # Schaffer4函数的自变量取值范围下界
    xmin = np.array([-100., -100])
    # Schaffer4函数的自变量取值范围上界
    xmax = np.array([100., 100])

    def fun(self, x):
        # 计算Schaffer4函数的能量值
        num = cos(sin(abs(x[0]**2 - x[1]**2)))**2 - 0.5
        den = (1+0.001*(x[0]**2 + x[1]**2))**2
        return 0.5 + num / den
```