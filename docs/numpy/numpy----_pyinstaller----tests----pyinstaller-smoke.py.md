# `.\numpy\numpy\_pyinstaller\tests\pyinstaller-smoke.py`

```
"""A crude *bit of everything* smoke test to verify PyInstaller compatibility.

PyInstaller typically goes wrong by forgetting to package modules, extension
modules or shared libraries. This script should aim to touch as many of those
as possible in an attempt to trip a ModuleNotFoundError or a DLL load failure
due to an uncollected resource. Missing resources are unlikely to lead to
arithmetic errors so there's generally no need to verify any calculation's
output - merely that it made it to the end OK. This script should not
explicitly import any of numpy's submodules as that gives PyInstaller undue
hints that those submodules exist and should be collected (accessing implicitly
loaded submodules is OK).

"""

# 导入 numpy 库，进行各种数值计算和线性代数操作的测试
import numpy as np

# 创建一个 3x3 的数组并对 5 取模
a = np.arange(1., 10.).reshape((3, 3)) % 5

# 计算数组 a 的行列式
np.linalg.det(a)

# 计算数组 a 与自身的矩阵乘积
a @ a

# 计算数组 a 与其转置的矩阵乘积
a @ a.T

# 计算数组 a 的逆矩阵
np.linalg.inv(a)

# 对数组 a 中的每个元素先计算指数，再求正弦
np.sin(np.exp(a))

# 对数组 a 进行奇异值分解
np.linalg.svd(a)

# 对数组 a 进行特征值分解
np.linalg.eigh(a)

# 生成一个包含 0 到 9 之间随机整数的数组，并返回其中的唯一值
np.unique(np.random.randint(0, 10, 100))

# 生成一个包含 0 到 10 之间均匀分布的随机数的数组，并对其进行排序
np.sort(np.random.uniform(0, 10, 100))

# 对一个复数数组进行傅里叶变换
np.fft.fft(np.exp(2j * np.pi * np.arange(8) / 8))

# 创建一个掩码数组，并计算其中被掩盖的部分的和
np.ma.masked_array(np.arange(10), np.random.rand(10) < .5).sum()

# 计算 Legendre 多项式的根
np.polynomial.Legendre([7, 8, 9]).roots()

# 输出测试通过的消息
print("I made it!")
```