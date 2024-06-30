# `D:\src\scipysrc\scipy\scipy\special\tests\test_sph_harm.py`

```
# 导入NumPy库，并使用np作为别名
import numpy as np
# 导入NumPy的测试模块，用于断言测试结果
from numpy.testing import assert_allclose
# 导入SciPy库中的special模块，并使用sc作为别名
import scipy.special as sc
# 导入SciPy库中special模块的_basic子模块中的_sph_harm_all函数
from scipy.special._basic import _sph_harm_all

# 定义测试函数，用于测试球谐函数的第一类谐波
def test_first_harmonics():
    # 测试与前四个球谐函数的显式表示进行对比
    # 这些函数使用`theta`作为方位角角度，`phi`作为极角角度，并包含Condon-Shortley相位。

    # 表示 Y00 的球谐函数
    def Y00(theta, phi):
        return 0.5 * np.sqrt(1 / np.pi)

    # 表示 Y^-11 的球谐函数
    def Yn11(theta, phi):
        return 0.5 * np.sqrt(3 / (2 * np.pi)) * np.exp(-1j * theta) * np.sin(phi)

    # 表示 Y^01 的球谐函数
    def Y01(theta, phi):
        return 0.5 * np.sqrt(3 / np.pi) * np.cos(phi)

    # 表示 Y^11 的球谐函数
    def Y11(theta, phi):
        return -0.5 * np.sqrt(3 / (2 * np.pi)) * np.exp(1j * theta) * np.sin(phi)

    # 将球谐函数存储在列表中
    harms = [Y00, Yn11, Y01, Y11]
    # 对应的 m 值
    m = [0, -1, 0, 1]
    # 对应的 n 值
    n = [0, 1, 1, 1]

    # 创建角度网格
    theta = np.linspace(0, 2 * np.pi)
    phi = np.linspace(0, np.pi)
    theta, phi = np.meshgrid(theta, phi)

    # 遍历球谐函数和对应的 m、n 值进行测试
    for harm, m, n in zip(harms, m, n):
        # 使用断言检查球谐函数的计算结果与期望值的接近程度
        assert_allclose(sc.sph_harm(m, n, theta, phi),
                        harm(theta, phi),
                        rtol=1e-15, atol=1e-15,
                        err_msg=f"Y^{m}_{n} incorrect")


def test_all_harmonics():
    # 最大 n 值
    n_max = 50

    # 创建角度网格
    theta = np.linspace(0, 2 * np.pi)
    phi = np.linspace(0, np.pi)

    # 计算所有球谐函数的实际值
    y_actual = _sph_harm_all(2 * n_max, n_max, theta, phi)

    # 遍历不同的 n、m 值进行测试
    for n in [0, 1, 2, 5, 10, 20, 50]:
        for m in [0, 1, 2, 5, 10, 20, 50]:
            # 根据条件选择期望的球谐函数值
            if (m <= n):
                y_desired = sc.sph_harm(m, n, theta, phi)
            else:
                y_desired = 0
            # 使用 NumPy 测试模块的断言检查实际计算值与期望值的接近程度
            np.testing.assert_allclose(y_actual[m, n], y_desired, rtol=1e-05)

            # 对于负 m 值的情况，同样进行断言检查
            if (m <= n):
                y_desired = sc.sph_harm(-m, n, theta, phi)
            else:
                y_desired = 0
            np.testing.assert_allclose(y_actual[-m, n], y_desired, rtol=1e-05)
```