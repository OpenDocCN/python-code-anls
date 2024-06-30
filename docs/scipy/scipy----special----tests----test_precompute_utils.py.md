# `D:\src\scipysrc\scipy\scipy\special\tests\test_precompute_utils.py`

```
# 导入 pytest 库，用于测试和断言
import pytest

# 从 scipy.special._testutils 中导入 MissingModule 和 check_version 函数
from scipy.special._testutils import MissingModule, check_version
# 从 scipy.special._mptestutils 中导入 mp_assert_allclose 函数
from scipy.special._mptestutils import mp_assert_allclose
# 从 scipy.special._precompute.utils 中导入 lagrange_inversion 函数
from scipy.special._precompute.utils import lagrange_inversion

# 尝试导入 sympy 库，如果失败则使用 MissingModule 创建一个伪造的 sympy 模块
try:
    import sympy
except ImportError:
    sympy = MissingModule('sympy')

# 尝试导入 mpmath 库，如果失败则使用 MissingModule 创建一个伪造的 mpmath 模块
try:
    import mpmath as mp
except ImportError:
    mp = MissingModule('mpmath')

# 使用 pytest 的 mark.slow 装饰器标记测试类 TestInversion 为慢速测试
@pytest.mark.slow
# 使用 check_version 装饰器检查 sympy 模块的版本必须至少为 0.7
@check_version(sympy, '0.7')
# 使用 check_version 装饰器检查 mpmath 模块的版本必须至少为 0.19
@check_version(mp, '0.19')
class TestInversion:
    # 对于 test_log 方法，使用 pytest 的 mark.xfail_on_32bit 装饰器标记为在 32 位系统上可能失败，给出失败的原因
    @pytest.mark.xfail_on_32bit("rtol only 2e-9, see gh-6938")
    def test_log(self):
        # 设置 mpmath 的工作精度为 30
        with mp.workdps(30):
            # 计算对数函数在 0 处的泰勒展开系数
            logcoeffs = mp.taylor(lambda x: mp.log(1 + x), 0, 10)
            # 计算指数函数减一在 0 处的泰勒展开系数
            expcoeffs = mp.taylor(lambda x: mp.exp(x) - 1, 0, 10)
            # 使用 lagrange_inversion 函数计算对数函数的反函数的泰勒展开系数
            invlogcoeffs = lagrange_inversion(logcoeffs)
            # 断言两个泰勒展开系数的近似性，使用自定义的 mp_assert_allclose 函数
            mp_assert_allclose(invlogcoeffs, expcoeffs)

    # 对于 test_sin 方法，使用 pytest 的 mark.xfail_on_32bit 装饰器标记为在 32 位系统上可能失败，给出失败的原因
    @pytest.mark.xfail_on_32bit("rtol only 1e-15, see gh-6938")
    def test_sin(self):
        # 设置 mpmath 的工作精度为 30
        with mp.workdps(30):
            # 计算正弦函数在 0 处的泰勒展开系数
            sincoeffs = mp.taylor(mp.sin, 0, 10)
            # 计算反正弦函数在 0 处的泰勒展开系数
            asincoeffs = mp.taylor(mp.asin, 0, 10)
            # 使用 lagrange_inversion 函数计算正弦函数的反函数的泰勒展开系数
            invsincoeffs = lagrange_inversion(sincoeffs)
            # 断言两个泰勒展开系数的近似性，使用自定义的 mp_assert_allclose 函数，并设置绝对容差为 1e-30
            mp_assert_allclose(invsincoeffs, asincoeffs, atol=1e-30)
```