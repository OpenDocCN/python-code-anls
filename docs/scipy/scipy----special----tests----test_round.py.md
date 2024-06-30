# `D:\src\scipysrc\scipy\scipy\special\tests\test_round.py`

```
import numpy as np
import pytest

from scipy.special import _test_internal

# 为测试函数添加一个“fail_slow”标记，设置其超时时间为20秒
@pytest.mark.fail_slow(20)
# 使用条件跳过装饰器，检查是否没有fenv支持，如果是，则跳过测试
@pytest.mark.skipif(not _test_internal.have_fenv(), reason="no fenv()")
def test_add_round_up():
    # 设置随机种子为1234
    np.random.seed(1234)
    # 调用特殊模块的测试函数，执行加法和舍入操作（向上）
    _test_internal.test_add_round(10**5, 'up')

# 为测试函数添加一个“fail_slow”标记，设置其超时时间为20秒
@pytest.mark.fail_slow(20)
# 使用条件跳过装饰器，检查是否没有fenv支持，如果是，则跳过测试
@pytest.mark.skipif(not _test_internal.have_fenv(), reason="no fenv()")
def test_add_round_down():
    # 设置随机种子为1234
    np.random.seed(1234)
    # 调用特殊模块的测试函数，执行加法和舍入操作（向下）
    _test_internal.test_add_round(10**5, 'down')
```