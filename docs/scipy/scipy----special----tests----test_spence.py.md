# `D:\src\scipysrc\scipy\scipy\special\tests\test_spence.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy import sqrt, log, pi  # 从 NumPy 库中导入 sqrt（平方根）、log（自然对数）、pi（圆周率）
from scipy.special._testutils import FuncData  # 导入用于功能测试的辅助工具 FuncData
from scipy.special import spence  # 导入斯邦斯函数（Spence's function）

def test_consistency():
    # 确保斯邦斯函数对于实数参数的实现
    # 与对于虚数参数的实现一致。

    x = np.logspace(-30, 300, 200)  # 生成从 10^-30 到 10^300 的 200 个对数间隔的数组
    dataset = np.vstack((x + 0j, spence(x))).T  # 创建数据集，包含实数参数和对应的斯邦斯函数值
    FuncData(spence, dataset, 0, 1, rtol=1e-14).check()  # 使用 FuncData 进行函数一致性检查


def test_special_points():
    # 根据已知斯邦斯函数值进行检查。

    phi = (1 + sqrt(5))/2  # 黄金比例 φ 的计算
    dataset = [(1, 0),  # 斯邦斯函数在特定点的已知值
               (2, -pi**2/12),
               (0.5, pi**2/12 - log(2)**2/2),
               (0, pi**2/6),
               (-1, pi**2/4 - 1j*pi*log(2)),
               ((-1 + sqrt(5))/2, pi**2/15 - log(phi)**2),
               ((3 - sqrt(5))/2, pi**2/10 - log(phi)**2),
               (phi, -pi**2/15 + log(phi)**2/2),
               # 从 Zagier 的 "The Dilogarithm Function" 中校正得到的值
               ((3 + sqrt(5))/2, -pi**2/10 - log(phi)**2)]

    dataset = np.asarray(dataset)  # 转换数据集为 NumPy 数组
    FuncData(spence, dataset, 0, 1, rtol=1e-14).check()  # 使用 FuncData 进行函数一致性检查
```