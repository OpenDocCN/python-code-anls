# `D:\src\scipysrc\scipy\scipy\special\tests\test_sici.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算

import scipy.special as sc  # 导入 SciPy 库中的特殊函数模块
from scipy.special._testutils import FuncData  # 导入 SciPy 的测试工具 FuncData


def test_sici_consistency():
    # 确保 sici 函数对于实数参数的实现与对复数参数的实现一致

    # 当参数位于负实轴时，Cephes 在 ci 函数中舍弃了虚部
    def sici(x):
        si, ci = sc.sici(x + 0j)  # 调用 SciPy 中的 sici 函数处理复数参数
        return si.real, ci.real  # 返回 si 和 ci 的实部

    # 创建一个包含广泛范围的实数数组 x
    x = np.r_[-np.logspace(8, -30, 200), 0, np.logspace(-30, 8, 200)]
    # 对数组 x 调用 SciPy 中的 sici 函数，获取 si 和 ci 的值
    si, ci = sc.sici(x)
    # 将 x、si、ci 组合成一个数据集
    dataset = np.column_stack((x, si, ci))
    # 使用 FuncData 进行函数一致性检查，指定函数为 sici，数据集为 dataset，第0列为输入，(1, 2)列为si和ci，相对误差容忍度为1e-12
    FuncData(sici, dataset, 0, (1, 2), rtol=1e-12).check()


def test_shichi_consistency():
    # 确保 shichi 函数对于实数参数的实现与对复数参数的实现一致

    # 当参数位于负实轴时，Cephes 在 chi 函数中舍弃了虚部
    def shichi(x):
        shi, chi = sc.shichi(x + 0j)  # 调用 SciPy 中的 shichi 函数处理复数参数
        return shi.real, chi.real  # 返回 shi 和 chi 的实部

    # 创建一个包含广泛范围的实数数组 x，限制在一个较小的范围内以避免溢出
    x = np.r_[-np.logspace(np.log10(700), -30, 200), 0,
              np.logspace(-30, np.log10(700), 200)]
    # 对数组 x 调用 SciPy 中的 shichi 函数，获取 shi 和 chi 的值
    shi, chi = sc.shichi(x)
    # 将 x、shi、chi 组合成一个数据集
    dataset = np.column_stack((x, shi, chi))
    # 使用 FuncData 进行函数一致性检查，指定函数为 shichi，数据集为 dataset，第0列为输入，(1, 2)列为shi和chi，相对误差容忍度为1e-14
    FuncData(shichi, dataset, 0, (1, 2), rtol=1e-14).check()
```