# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_isolve\tests\test_utils.py`

```
import numpy as np
# 导入 NumPy 库，用于处理数组和数值计算
from pytest import raises as assert_raises
# 导入 pytest 库中的 raises 函数，用于断言抛出特定异常

import scipy.sparse.linalg._isolve.utils as utils
# 导入 SciPy 稀疏矩阵求解模块中的 utils 模块，用于稀疏矩阵求解的工具函数


def test_make_system_bad_shape():
    # 定义测试函数 test_make_system_bad_shape，用于测试 make_system 函数对不良形状输入的处理
    
    assert_raises(ValueError,
                  utils.make_system, np.zeros((5,3)), None, np.zeros(4), np.zeros(4))
    # 断言调用 make_system 函数会引发 ValueError 异常，
    # 传入的参数分别是一个形状为 (5,3) 的零数组，一个 None，一个形状为 (4,) 的零数组，以及一个形状为 (4,) 的零数组
```