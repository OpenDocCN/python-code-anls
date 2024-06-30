# `D:\src\scipysrc\scipy\scipy\special\tests\test_loggamma.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import assert_allclose, assert_  # 导入 NumPy 测试模块中的函数

from scipy.special._testutils import FuncData  # 导入 SciPy 中专门用于函数测试的工具类
from scipy.special import gamma, gammaln, loggamma  # 导入 SciPy 中的 gamma 相关函数


def test_identities1():
    # test the identity exp(loggamma(z)) = gamma(z)
    x = np.array([-99.5, -9.5, -0.5, 0.5, 9.5, 99.5])  # 创建包含特定数值的 NumPy 数组 x
    y = x.copy()  # 复制数组 x 到 y
    x, y = np.meshgrid(x, y)  # 创建 x 和 y 的网格
    z = (x + 1J*y).flatten()  # 生成复数数组 z，并展平为一维
    dataset = np.vstack((z, gamma(z))).T  # 创建数据集，包含 z 和 gamma(z) 的转置

    def f(z):
        return np.exp(loggamma(z))  # 定义函数 f，返回 exp(loggamma(z))

    FuncData(f, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()  # 使用 FuncData 检查函数 f 的表现


def test_identities2():
    # test the identity loggamma(z + 1) = log(z) + loggamma(z)
    x = np.array([-99.5, -9.5, -0.5, 0.5, 9.5, 99.5])  # 创建包含特定数值的 NumPy 数组 x
    y = x.copy()  # 复制数组 x 到 y
    x, y = np.meshgrid(x, y)  # 创建 x 和 y 的网格
    z = (x + 1J*y).flatten()  # 生成复数数组 z，并展平为一维
    dataset = np.vstack((z, np.log(z) + loggamma(z))).T  # 创建数据集，包含 z 和 np.log(z) + loggamma(z) 的转置

    def f(z):
        return loggamma(z + 1)  # 定义函数 f，返回 loggamma(z + 1)

    FuncData(f, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()  # 使用 FuncData 检查函数 f 的表现


def test_complex_dispatch_realpart():
    # Test that the real parts of loggamma and gammaln agree on the
    # real axis.
    x = np.r_[-np.logspace(10, -10), np.logspace(-10, 10)] + 0.5  # 创建包含特定数值的 NumPy 数组 x

    dataset = np.vstack((x, gammaln(x))).T  # 创建数据集，包含 x 和 gammaln(x) 的转置

    def f(z):
        z = np.array(z, dtype='complex128')  # 将 z 转换为复数数组
        return loggamma(z).real  # 返回 loggamma(z) 的实部

    FuncData(f, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()  # 使用 FuncData 检查函数 f 的表现


def test_real_dispatch():
    x = np.logspace(-10, 10) + 0.5  # 创建包含特定数值的 NumPy 数组 x
    dataset = np.vstack((x, gammaln(x))).T  # 创建数据集，包含 x 和 gammaln(x) 的转置

    FuncData(loggamma, dataset, 0, 1, rtol=1e-14, atol=1e-14).check()  # 使用 FuncData 检查 loggamma 函数的表现
    assert_(loggamma(0) == np.inf)  # 断言 loggamma(0) 等于无穷大
    assert_(np.isnan(loggamma(-1)))  # 断言 loggamma(-1) 是 NaN


def test_gh_6536():
    z = loggamma(complex(-3.4, +0.0))  # 计算 loggamma(complex(-3.4, +0.0))
    zbar = loggamma(complex(-3.4, -0.0))  # 计算 loggamma(complex(-3.4, -0.0))
    assert_allclose(z, zbar.conjugate(), rtol=1e-15, atol=0)  # 使用 assert_allclose 检查 z 和 zbar 共轭的近似程度


def test_branch_cut():
    # Make sure negative zero is treated correctly
    x = -np.logspace(300, -30, 100)  # 创建包含特定数值的 NumPy 数组 x
    z = np.asarray([complex(x0, 0.0) for x0 in x])  # 生成复数数组 z，确保负零被正确处理
    zbar = np.asarray([complex(x0, -0.0) for x0 in x])  # 生成复数数组 zbar，确保负零被正确处理
    assert_allclose(z, zbar.conjugate(), rtol=1e-15, atol=0)  # 使用 assert_allclose 检查 z 和 zbar 共轭的近似程度
```