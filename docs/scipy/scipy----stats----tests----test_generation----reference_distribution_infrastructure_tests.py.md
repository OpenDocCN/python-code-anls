# `D:\src\scipysrc\scipy\scipy\stats\tests\test_generation\reference_distribution_infrastructure_tests.py`

```
# 导入必要的库和模块
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from scipy import stats  # 从 SciPy 中导入 stats 模块，用于统计计算
from numpy.testing import assert_allclose  # 从 NumPy 的测试模块中导入 assert_allclose 函数，用于比较数值是否接近
import scipy.stats.tests.test_generation.reference_distributions as rd  # 导入测试参考分布的模块
import mpmath  # 导入 mpmath 库，用于高精度数学计算
from mpmath import mp  # 从 mpmath 中导入 mp 模块

# 定义一个测试函数，用于测试基本功能
def test_basic():

    # 设置错误信息，用于异常断言
    message = "`mpmath.mp.dps <= 15`. Set a higher precision..."
    # 使用 pytest 的异常断言，检查是否引发 RuntimeError 异常，并匹配错误信息
    with pytest.raises(RuntimeError, match=message):
        rd.Normal()

    # 设置 mpmath 的精度为 20
    mpmath.dps = 20
    # 设置错误信息，用于异常断言
    message = "`mpmath.dps` has been assigned. This is not intended usage..."
    # 使用 pytest 的异常断言，检查是否引发 RuntimeError 异常，并匹配错误信息
    with pytest.raises(RuntimeError, match=message):
        rd.Normal()
    # 删除 mpmath 的精度设置，恢复默认值

    del mpmath.dps

    # 设置 mp 的精度为 20，以确保足够高，同时不至于过慢
    mp.dps = 20  

    # 对 mpmath 分布基础设施进行基本测试，使用 SciPy 分布作为参考
    # 目的是确保实现没有错误，并且广播功能按预期工作。精度是它的准确度。
    
    # 使用默认的随机生成器创建随机数生成器对象 rng
    rng = np.random.default_rng(6716188855217730280)

    # 生成随机数向量 x，长度为 3
    x = rng.random(size=3)
    # 生成随机数矩阵 a，形状为 (2, 1)
    a = rng.random(size=(2, 1))
    # 定义相对误差容忍度
    rtol = 1e-15

    # 创建 SkewNormal 对象 dist，使用参数 a
    dist = rd.SkewNormal(a=a)
    # 创建 stats.skewnorm 对象 dist_ref，使用参数 a
    dist_ref = stats.skewnorm(a)

    # 使用 assert_allclose 函数分别比较 dist 和 dist_ref 的概率密度函数等
    assert_allclose(dist.pdf(x), dist_ref.pdf(x), rtol=rtol)
    assert_allclose(dist.cdf(x), dist_ref.cdf(x), rtol=rtol)
    assert_allclose(dist.sf(x), dist_ref.sf(x), rtol=rtol)
    assert_allclose(dist.ppf(x), dist_ref.ppf(x), rtol=rtol)
    assert_allclose(dist.isf(x), dist_ref.isf(x), rtol=rtol)
    assert_allclose(dist.logpdf(x), dist_ref.logpdf(x), rtol=rtol)
    assert_allclose(dist.logcdf(x), dist_ref.logcdf(x), rtol=rtol)
    assert_allclose(dist.logsf(x), dist_ref.logsf(x), rtol=rtol)
    assert_allclose(dist.support(), dist_ref.support(), rtol=rtol)
    assert_allclose(dist.entropy(), dist_ref.entropy(), rtol=rtol)
    assert_allclose(dist.mean(), dist_ref.mean(), rtol=rtol)
    assert_allclose(dist.var(), dist_ref.var(), rtol=rtol)
    assert_allclose(dist.skew(), dist_ref.stats('s'), rtol=rtol)
    assert_allclose(dist.kurtosis(), dist_ref.stats('k'), rtol=rtol)

# 定义一个测试函数，用于测试补充方法的使用
def test_complementary_method_use():
    # 展示补充方法的预期使用方式。
    # 例如，如果覆盖了 CDF，则使用 1 - CDF 计算 SF。

    # 设置 mp 的精度为 50
    mp.dps = 50  
    # 生成等间隔数列 x，范围从 0 到 1，共 10 个数
    x = np.linspace(0, 1, 10)

    # 定义 MyDist 类，继承自 rd.ReferenceDistribution
    class MyDist(rd.ReferenceDistribution):
        # 定义私有方法 _cdf，覆盖默认的累积分布函数
        def _cdf(self, x):
            return x

    # 创建 MyDist 对象 dist
    dist = MyDist()
    # 使用 assert_allclose 函数检查 dist.sf(x) 是否等于 1 - dist.cdf(x)
    assert_allclose(dist.sf(x), 1 - dist.cdf(x))

    # 定义 MyDist 类，继承自 rd.ReferenceDistribution
    class MyDist(rd.ReferenceDistribution):
        # 定义私有方法 _sf，覆盖默认的生存函数
        def _sf(self, x):
            return 1 - x

    # 创建 MyDist 对象 dist
    dist = MyDist()
    # 使用 assert_allclose 函数检查 dist.cdf(x) 是否等于 1 - dist.sf(x)
    assert_allclose(dist.cdf(x), 1 - dist.sf(x))

    # 定义 MyDist 类，继承自 rd.ReferenceDistribution
    class MyDist(rd.ReferenceDistribution):
        # 定义私有方法 _ppf，覆盖默认的分位点函数
        def _ppf(self, x, guess):
            return x

    # 创建 MyDist 对象 dist
    dist = MyDist()
    # 使用 assert_allclose 函数检查 dist.isf(x) 是否等于 dist.ppf(1-x)
    assert_allclose(dist.isf(x), dist.ppf(1 - x))

    # 定义 MyDist 类，继承自 rd.ReferenceDistribution
    class MyDist(rd.ReferenceDistribution):
        # 定义私有方法 _isf，覆盖默认的逆生存函数
        def _isf(self, x, guess):
            return 1 - x

    # 创建 MyDist 对象 dist
    dist = MyDist()
    # 使用分位点函数和逆分位点函数来验证概率分布的性质
    assert_allclose(dist.ppf(x), dist.isf(1-x))
```