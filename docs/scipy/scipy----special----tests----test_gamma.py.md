# `D:\src\scipysrc\scipy\scipy\special\tests\test_gamma.py`

```
import numpy as np  # 导入 NumPy 库，并使用 np 别名

import scipy.special as sc  # 导入 SciPy 库中的 special 模块，并使用 sc 别名


class TestRgamma:  # 定义一个名为 TestRgamma 的测试类

    def test_gh_11315(self):  # 定义一个名为 test_gh_11315 的测试方法
        assert sc.rgamma(-35) == 0  # 断言调用 SciPy 的 rgamma 函数对参数 -35 的返回值为 0

    def test_rgamma_zeros(self):  # 定义一个名为 test_rgamma_zeros 的测试方法
        x = np.array([0, -10, -100, -1000, -10000])  # 创建一个 NumPy 数组 x，包含整数 0, -10, -100, -1000, -10000
        assert np.all(sc.rgamma(x) == 0)  # 断言调用 SciPy 的 rgamma 函数对数组 x 的每个元素返回值均为 0
```