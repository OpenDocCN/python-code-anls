# `D:\src\scipysrc\scipy\scipy\special\tests\test_digamma.py`

```
# 导入 NumPy 库并使用简称 np
import numpy as np
# 从 NumPy 库中导入常量 pi, log, sqrt
from numpy import pi, log, sqrt
# 从 NumPy 的测试模块中导入 assert_ 和 assert_equal 函数
from numpy.testing import assert_, assert_equal
# 从 SciPy 的特殊函数模块中导入 FuncData 类
from scipy.special._testutils import FuncData
# 导入 SciPy 的特殊函数模块并使用简称 sc
import scipy.special as sc

# Euler-Mascheroni 常数的定义
euler = 0.57721566490153286

# 测试函数：确保对实数参数的 digamma 函数实现与复数参数的 digamma 函数实现一致
def test_consistency():
    # 创建一个包含大量实数和复数参数的数据集
    x = np.r_[-np.logspace(15, -30, 200), np.logspace(-30, 300, 200)]
    dataset = np.vstack((x + 0j, sc.digamma(x))).T
    # 使用 FuncData 类来检查 digamma 函数的实现一致性
    FuncData(sc.digamma, dataset, 0, 1, rtol=5e-14, nan_ok=True).check()

# 测试函数：测试 Gauss 的 digamma 定理中的特殊值
def test_special_values():
    # 定义包含特殊值的数据集，参考 Gauss 的 digamma 定理
    dataset = [
        (1, -euler),
        (0.5, -2*log(2) - euler),
        (1/3, -pi/(2*sqrt(3)) - 3*log(3)/2 - euler),
        (1/4, -pi/2 - 3*log(2) - euler),
        (1/6, -pi*sqrt(3)/2 - 2*log(2) - 3*log(3)/2 - euler),
        (1/8,
         -pi/2 - 4*log(2) - (pi + log(2 + sqrt(2)) - log(2 - sqrt(2)))/sqrt(2) - euler)
    ]
    # 将数据集转换为 NumPy 数组
    dataset = np.asarray(dataset)
    # 使用 FuncData 类来检查 digamma 函数的特殊值
    FuncData(sc.digamma, dataset, 0, 1, rtol=1e-14).check()

# 测试函数：测试 digamma 函数在非有限点上的行为
def test_nonfinite():
    # 定义一些非有限点
    pts = [0.0, -0.0, np.inf]
    std = [-np.inf, np.inf, np.inf]
    # 使用 assert_equal 函数确保 digamma 函数在指定点上的值与预期的标准值相等
    assert_equal(sc.digamma(pts), std)
    # 使用 assert_ 函数确保 digamma 函数在某些非有限点上返回 NaN
    assert_(all(np.isnan(sc.digamma([-np.inf, -1]))))
```