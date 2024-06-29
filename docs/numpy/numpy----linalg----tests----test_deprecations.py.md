# `.\numpy\numpy\linalg\tests\test_deprecations.py`

```py
"""Test deprecation and future warnings.

"""
# 导入 numpy 库，简写为 np
import numpy as np
# 从 numpy.testing 模块中导入 assert_warns 函数
from numpy.testing import assert_warns


# 定义测试函数 test_qr_mode_full_future_warning
def test_qr_mode_full_future_warning():
    """Check mode='full' FutureWarning.

    In numpy 1.8 the mode options 'full' and 'economic' in linalg.qr were
    deprecated. The release date will probably be sometime in the summer
    of 2013.

    """
    # 创建一个 2x2 的单位矩阵 a
    a = np.eye(2)
    # 断言调用 np.linalg.qr 函数时会产生 DeprecationWarning，使用 mode='full'
    assert_warns(DeprecationWarning, np.linalg.qr, a, mode='full')
    # 断言调用 np.linalg.qr 函数时会产生 DeprecationWarning，使用 mode='f'
    assert_warns(DeprecationWarning, np.linalg.qr, a, mode='f')
    # 断言调用 np.linalg.qr 函数时会产生 DeprecationWarning，使用 mode='economic'
    assert_warns(DeprecationWarning, np.linalg.qr, a, mode='economic')
    # 断言调用 np.linalg.qr 函数时会产生 DeprecationWarning，使用 mode='e'
    assert_warns(DeprecationWarning, np.linalg.qr, a, mode='e')
```