# `D:\src\scipysrc\scipy\scipy\stats\tests\test_resampling.py`

```
# 导入pytest模块，用于测试和断言
import pytest

# 导入numpy模块，并从中导入assert_allclose, assert_equal, suppress_warnings函数
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings

# 从scipy.conftest模块中导入array_api_compatible函数
from scipy.conftest import array_api_compatible

# 从scipy._lib._util模块中导入rng_integers函数
from scipy._lib._util import rng_integers

# 从scipy._lib._array_api模块中导入is_numpy, xp_assert_close,
# xp_assert_equal, array_namespace函数
from scipy._lib._array_api import (is_numpy, xp_assert_close,
                                   xp_assert_equal, array_namespace)

# 从scipy模块中导入stats, special模块
from scipy import stats, special

# 从scipy.optimize模块中导入root函数
from scipy.optimize import root

# 从scipy.stats模块中导入bootstrap, monte_carlo_test, permutation_test, power函数
from scipy.stats import bootstrap, monte_carlo_test, permutation_test, power

# 导入scipy.stats._resampling模块，并重命名为_resampling
import scipy.stats._resampling as _resampling


# 定义一个测试函数test_bootstrap_iv，用于测试bootstrap函数的各种输入情况
def test_bootstrap_iv():

    # 设置错误信息，用于断言检查是否引发特定的ValueError异常，异常信息需匹配给定的字符串
    message = "`data` must be a sequence of samples."
    with pytest.raises(ValueError, match=message):
        bootstrap(1, np.mean)

    message = "`data` must contain at least one sample."
    with pytest.raises(ValueError, match=message):
        bootstrap(tuple(), np.mean)

    message = "each sample in `data` must contain two or more observations..."
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3], [1]), np.mean)

    message = ("When `paired is True`, all samples must have the same length ")
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3], [1, 2, 3, 4]), np.mean, paired=True)

    message = "`vectorized` must be `True`, `False`, or `None`."
    with pytest.raises(ValueError, match=message):
        bootstrap(1, np.mean, vectorized='ekki')

    message = "`axis` must be an integer."
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, axis=1.5)

    message = "could not convert string to float"
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, confidence_level='ni')

    message = "`n_resamples` must be a non-negative integer."
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, n_resamples=-1000)

    message = "`n_resamples` must be a non-negative integer."
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, n_resamples=1000.5)

    message = "`batch` must be a positive integer or None."
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, batch=-1000)

    message = "`batch` must be a positive integer or None."
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, batch=1000.5)

    message = "`method` must be in"
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, method='ekki')

    message = "`bootstrap_result` must have attribute `bootstrap_distribution`"
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, bootstrap_result=10)

    message = "Either `bootstrap_result.bootstrap_distribution.size`"
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, n_resamples=0)

    message = "'herring' cannot be used to seed a"
    # 使用 pytest 模块中的 pytest.raises 上下文管理器来测试函数是否引发指定类型的异常，并匹配异常消息
    with pytest.raises(ValueError, match=message):
        # 调用 bootstrap 函数，传入参数 ([1, 2, 3],)，np.mean，和 random_state='herring'
        bootstrap(([1, 2, 3],), np.mean, random_state='herring')
@pytest.mark.parametrize("method", ['basic', 'percentile', 'BCa'])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_bootstrap_batch(method, axis):
    # 对于单样本统计，批处理大小不应影响结果

    # 设置随机种子，以便结果可重复
    np.random.seed(0)

    # 创建一个形状为 (10, 11, 12) 的随机数组 x
    x = np.random.rand(10, 11, 12)

    # 进行两次自助法统计，分别使用 batch=None 和 batch=10
    res1 = bootstrap((x,), np.mean, batch=None, method=method,
                     random_state=0, axis=axis, n_resamples=100)
    res2 = bootstrap((x,), np.mean, batch=10, method=method,
                     random_state=0, axis=axis, n_resamples=100)

    # 断言两次自助法的置信区间下限和上限应相等
    assert_equal(res2.confidence_interval.low, res1.confidence_interval.low)
    assert_equal(res2.confidence_interval.high, res1.confidence_interval.high)
    # 断言两次自助法的标准误差应相等
    assert_equal(res2.standard_error, res1.standard_error)


@pytest.mark.parametrize("method", ['basic', 'percentile', 'BCa'])
def test_bootstrap_paired(method):
    # 测试 `paired` 参数按预期工作

    # 设置随机种子，以便结果可重复
    np.random.seed(0)
    n = 100
    # 创建两个形状为 (n,) 的随机数组 x 和 y
    x = np.random.rand(n)
    y = np.random.rand(n)

    # 定义一个统计函数，计算差值平方的均值
    def my_statistic(x, y, axis=-1):
        return ((x-y)**2).mean(axis=axis)

    # 定义一个配对统计函数
    def my_paired_statistic(i, axis=-1):
        a = x[i]
        b = y[i]
        res = my_statistic(a, b)
        return res

    # 创建一个包含索引的数组
    i = np.arange(len(x))

    # 使用自助法计算配对统计量的结果 res1
    res1 = bootstrap((i,), my_paired_statistic, random_state=0)
    # 使用自助法计算未配对统计量的结果 res2
    res2 = bootstrap((x, y), my_statistic, paired=True, random_state=0)

    # 断言两次自助法的置信区间应接近
    assert_allclose(res1.confidence_interval, res2.confidence_interval)
    # 断言两次自助法的标准误差应接近
    assert_allclose(res1.standard_error, res2.standard_error)


@pytest.mark.parametrize("method", ['basic', 'percentile', 'BCa'])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("paired", [True, False])
def test_bootstrap_vectorized(method, axis, paired):
    # 测试配对参数在向量化时的预期行为：当样本被铺展时，
    # 每个轴切片的置信区间和标准误差与原始的1维样本相同

    # 设置随机种子，以便结果可重复
    np.random.seed(0)

    # 定义一个统计函数，计算三个数组的均值之和
    def my_statistic(x, y, z, axis=-1):
        return x.mean(axis=axis) + y.mean(axis=axis) + z.mean(axis=axis)

    # 创建一个形状为 (10, 11, 12) 的样本
    shape = 10, 11, 12

    # 根据 axis 参数确定样本的大小
    n_samples = shape[axis]

    # 创建三个形状为 (n_samples,) 的随机数组 x, y, z
    x = np.random.rand(n_samples)
    y = np.random.rand(n_samples)
    z = np.random.rand(n_samples)

    # 使用自助法计算统计量 res1
    res1 = bootstrap((x, y, z), my_statistic, paired=paired, method=method,
                     random_state=0, axis=0, n_resamples=100)

    # 断言 res1 的 bootstrap_distribution 形状应与 standard_error 的形状加上 (100,) 相同
    assert (res1.bootstrap_distribution.shape
            == res1.standard_error.shape + (100,))

    # 将 x, y, z 铺展成与 shape 相同的数组
    reshape = [1, 1, 1]
    reshape[axis] = n_samples
    x = np.broadcast_to(x.reshape(reshape), shape)
    y = np.broadcast_to(y.reshape(reshape), shape)
    z = np.broadcast_to(z.reshape(reshape), shape)

    # 使用自助法计算统计量 res2
    res2 = bootstrap((x, y, z), my_statistic, paired=paired, method=method,
                     random_state=0, axis=axis, n_resamples=100)

    # 断言两次自助法的置信区间下限应接近
    assert_allclose(res2.confidence_interval.low,
                    res1.confidence_interval.low)
    # 断言两次自助法的置信区间上限应接近
    assert_allclose(res2.confidence_interval.high,
                    res1.confidence_interval.high)
    # 断言：验证两个对象的属性 standard_error 的值是否在允许的误差范围内相等
    assert_allclose(res2.standard_error, res1.standard_error)
    
    # 创建结果形状的副本列表，并移除指定的轴
    result_shape = list(shape)
    result_shape.pop(axis)
    
    # 断言：验证 res2 对象的置信区间的低端属性的形状是否与修改后的结果形状匹配
    assert_equal(res2.confidence_interval.low.shape, result_shape)
    # 断言：验证 res2 对象的置信区间的高端属性的形状是否与修改后的结果形状匹配
    assert_equal(res2.confidence_interval.high.shape, result_shape)
    # 断言：验证 res2 对象的标准误差属性的形状是否与修改后的结果形状匹配
    assert_equal(res2.standard_error.shape, result_shape)
@pytest.mark.slow
@pytest.mark.xfail_on_32bit("MemoryError with BCa observed in CI")
# 标记测试为慢速执行，并且在32位系统上运行时预期会出现内存错误
@pytest.mark.parametrize("method", ['basic', 'percentile', 'BCa'])
# 参数化测试方法，使用 'basic', 'percentile', 'BCa' 三种方法分别运行测试

def test_bootstrap_against_theory(method):
    # 基于 https://www.statology.org/confidence-intervals-python/ 修改的测试案例
    rng = np.random.default_rng(2442101192988600726)
    # 使用指定种子创建随机数生成器
    data = stats.norm.rvs(loc=5, scale=2, size=5000, random_state=rng)
    # 从正态分布中生成随机样本数据
    alpha = 0.95
    dist = stats.t(df=len(data)-1, loc=np.mean(data), scale=stats.sem(data))
    # 创建 t 分布对象，用于计算置信区间和标准误差
    expected_interval = dist.interval(confidence=alpha)
    expected_se = dist.std()

    config = dict(data=(data,), statistic=np.mean, n_resamples=5000,
                  method=method, random_state=rng)
    # 构建配置字典，用于调用 bootstrap 函数
    res = bootstrap(**config, confidence_level=alpha)
    # 执行 bootstrap 方法，得到结果对象 res
    assert_allclose(res.confidence_interval, expected_interval, rtol=5e-4)
    assert_allclose(res.standard_error, expected_se, atol=3e-4)

    config.update(dict(n_resamples=0, bootstrap_result=res))
    # 更新配置字典，重置 n_resamples，并使用之前的结果 res
    res = bootstrap(**config, confidence_level=alpha, alternative='less')
    # 再次调用 bootstrap 方法，使用 alternative='less' 参数
    assert_allclose(res.confidence_interval.high, dist.ppf(alpha), rtol=5e-4)

    config.update(dict(n_resamples=0, bootstrap_result=res))
    # 更新配置字典，重置 n_resamples，并使用之前的结果 res
    res = bootstrap(**config, confidence_level=alpha, alternative='greater')
    # 再次调用 bootstrap 方法，使用 alternative='greater' 参数
    assert_allclose(res.confidence_interval.low, dist.ppf(1-alpha), rtol=5e-4)


tests_R = {"basic": (23.77, 79.12),
           "percentile": (28.86, 84.21),
           "BCa": (32.31, 91.43)}

@pytest.mark.parametrize("method, expected", tests_R.items())
# 参数化测试方法和期望结果，依次使用 'basic', 'percentile', 'BCa' 三种方法和对应的期望结果运行测试
def test_bootstrap_against_R(method, expected):
    # 与 R 的 "boot" 库进行比较
    # library(boot)
    #
    # stat <- function (x, a) {
    #     mean(x[a])
    # }
    #
    # x <- c(10, 12, 12.5, 12.5, 13.9, 15, 21, 22,
    #        23, 34, 50, 81, 89, 121, 134, 213)
    #
    # # Use a large value so we get a few significant digits for the CI.
    # n = 1000000
    # bootresult = boot(x, stat, n)
    # result <- boot.ci(bootresult)
    # print(result)

    x = np.array([10, 12, 12.5, 12.5, 13.9, 15, 21, 22,
                  23, 34, 50, 81, 89, 121, 134, 213])
    # 定义输入数据 x
    res = bootstrap((x,), np.mean, n_resamples=1000000, method=method,
                    random_state=0)
    # 使用给定方法调用 bootstrap 方法进行计算
    assert_allclose(res.confidence_interval, expected, rtol=0.005)
    # 断言计算结果的置信区间与期望结果接近


tests_against_itself_1samp = {"basic": 1780,
                              "percentile": 1784,
                              "BCa": 1784}

def test_multisample_BCa_against_R():
    # 因为 bootstrap 是随机的，测试其行为是否与参考行为一致是棘手的。
    # 在这里，我们展示 SciPy 的 BCa 置信区间与 R 中 wboot 的 BCa 置信区间更为接近，
    # 而其他 SciPy 置信区间则不太相似。

    # 使用任意的偏斜数据
    x = [0.75859206, 0.5910282, -0.4419409, -0.36654601,
         0.34955357, -1.38835871, 0.76735821]
    y = [1.41186073, 0.49775975, 0.08275588, 0.24086388,
         0.03567057, 0.52024419, 0.31966611, 1.32067634]
    # 定义多样本统计量，BCa 置信区间通常与其他置信区间不同
    # 定义一个统计函数，计算两个数据集在指定轴上的偏度差
    def statistic(x, y, axis):
        # 计算数据集 x 在指定轴上的偏度
        s1 = stats.skew(x, axis=axis)
        # 计算数据集 y 在指定轴上的偏度
        s2 = stats.skew(y, axis=axis)
        # 返回偏度差
        return s1 - s2

    # 使用指定的随机数生成器创建随机数生成对象
    rng = np.random.default_rng(468865032284792692)

    # 使用基本方法（basic）进行自助法计算置信区间
    res_basic = stats.bootstrap((x, y), statistic, method='basic',
                                batch=100, random_state=rng)
    
    # 使用百分位法（percentile）进行自助法计算置信区间
    res_percent = stats.bootstrap((x, y), statistic, method='percentile',
                                  batch=100, random_state=rng)
    
    # 使用BCa法（bca）进行自助法计算置信区间
    res_bca = stats.bootstrap((x, y), statistic, method='bca',
                              batch=100, random_state=rng)

    # 计算各方法置信区间的中点，以便进行比较
    mid_basic = np.mean(res_basic.confidence_interval)
    mid_percent = np.mean(res_percent.confidence_interval)
    mid_bca = np.mean(res_bca.confidence_interval)

    # 参考使用 R 的 wboot 包计算得到的 BCa 置信区间中点：
    # library(wBoot)
    # library(moments)
    
    # 假设函数 twoskew(x1, y1) {skewness(x1) - skewness(y1)}，计算得到的 wboot BCa 中点
    mid_wboot = -1.5519

    # 计算相对于 wboot BCa 方法的百分比差异
    diff_basic = (mid_basic - mid_wboot)/abs(mid_wboot)
    diff_percent = (mid_percent - mid_wboot)/abs(mid_wboot)
    diff_bca = (mid_bca - mid_wboot)/abs(mid_wboot)

    # SciPy 的 BCa 置信区间中点比其他方法更接近
    # 断言基本方法的差异小于 -0.15
    assert diff_basic < -0.15
    # 断言百分位方法的差异大于 0.15
    assert diff_percent > 0.15
    # 断言 BCa 方法的差异的绝对值小于 0.03
    assert abs(diff_bca) < 0.03
def test_BCa_acceleration_against_reference():
    # 比较多样本问题中的（确定性）加速度参数与参考值的对比。该示例源自[1]，但Efron的值似乎不准确。
    # 可以在以下链接找到计算参考加速度（0.011008228344026734）的直接代码：
    # https://github.com/scipy/scipy/pull/16455#issuecomment-1193400981

    y = np.array([10, 27, 31, 40, 46, 50, 52, 104, 146])
    z = np.array([16, 23, 38, 94, 99, 141, 197])

    def statistic(z, y, axis=0):
        return np.mean(z, axis=axis) - np.mean(y, axis=axis)
    # 定义一个统计函数，计算 z 和 y 的均值之差

    data = [z, y]
    res = stats.bootstrap(data, statistic)
    # 对 data 进行 bootstrap 操作，使用定义的 statistic 函数

    axis = -1
    alpha = 0.95
    theta_hat_b = res.bootstrap_distribution
    batch = 100
    _, _, a_hat = _resampling._bca_interval(data, statistic, axis, alpha,
                                            theta_hat_b, batch)
    # 计算基于 bootstrap 的加速度估计值 a_hat

    assert_allclose(a_hat, 0.011008228344026734)


@pytest.mark.slow
@pytest.mark.parametrize("method, expected",
                         tests_against_itself_1samp.items())
def test_bootstrap_against_itself_1samp(method, expected):
    # 该测试使用 bootstrap 生成的预期值来检查行为上的意外变化。
    # 测试还确保 bootstrap 能够处理多样本统计，并且 axis 参数按预期/函数向量化工作。

    np.random.seed(0)

    n = 100  # 样本大小
    n_resamples = 999  # 用于形成每个置信区间的 bootstrap 重采样次数
    confidence_level = 0.9

    # 真实均值为 5
    dist = stats.norm(loc=5, scale=1)
    stat_true = dist.mean()

    # 对代码进行2000次重复（代码完全向量化）
    n_replications = 2000
    data = dist.rvs(size=(n_replications, n))
    res = bootstrap((data,),
                    statistic=np.mean,
                    confidence_level=confidence_level,
                    n_resamples=n_resamples,
                    batch=50,
                    method=method,
                    axis=-1)
    ci = res.confidence_interval
    # 计算均值的 bootstrap 置信区间

    # ci 包含下界和上界的向量
    ci_contains_true = np.sum((ci[0] < stat_true) & (stat_true < ci[1]))
    assert ci_contains_true == expected
    # 检查 ci 是否包含真实均值的向量

    # ci_contains_true 与 confidence_level 一致性不矛盾
    pvalue = stats.binomtest(ci_contains_true, n_replications,
                             confidence_level).pvalue
    assert pvalue > 0.1


tests_against_itself_2samp = {"basic": 892,
                              "percentile": 890}


@pytest.mark.slow
@pytest.mark.parametrize("method, expected",
                         tests_against_itself_2samp.items())
def test_bootstrap_against_itself_2samp(method, expected):
    # 该测试使用 bootstrap 生成的预期值来检查行为上的意外变化。
    # 测试还确保 bootstrap 能够处理多样本统计，并且
    # 至少在这里，对于双样本检验，它应该运行正确。
    # 设置随机种子以保证结果的可重复性
    np.random.seed(0)

    n1 = 100  # 样本1的大小
    n2 = 120  # 样本2的大小
    n_resamples = 999  # 用于形成每个置信区间的自助法重采样次数
    confidence_level = 0.9  # 置信水平

    # 我们关心的统计量是均值之差
    def my_stat(data1, data2, axis=-1):
        # 计算数据1和数据2的均值
        mean1 = np.mean(data1, axis=axis)
        mean2 = np.mean(data2, axis=axis)
        return mean1 - mean2

    # 真实均值之差为 -0.1
    dist1 = stats.norm(loc=0, scale=1)  # 第一个分布是标准正态分布
    dist2 = stats.norm(loc=0.1, scale=1)  # 第二个分布的均值为0.1的正态分布
    stat_true = dist1.mean() - dist2.mean()  # 真实均值之差

    # 重复进行1000次相同的操作（代码完全向量化）
    n_replications = 1000
    data1 = dist1.rvs(size=(n_replications, n1))  # 从第一个分布中生成数据1
    data2 = dist2.rvs(size=(n_replications, n2))  # 从第二个分布中生成数据2

    # 进行自助法估计
    res = bootstrap((data1, data2),
                    statistic=my_stat,
                    confidence_level=confidence_level,
                    n_resamples=n_resamples,
                    batch=50,
                    method=method,  # 方法参数未指定
                    axis=-1)

    ci = res.confidence_interval  # 置信区间

    # ci 包含了下限和上限置信区间边界的向量
    ci_contains_true = np.sum((ci[0] < stat_true) & (stat_true < ci[1]))

    # 确保置信区间包含真实值
    assert ci_contains_true == expected

    # 置信区间的结果与置信水平不矛盾
    pvalue = stats.binomtest(ci_contains_true, n_replications,
                             confidence_level).pvalue

    # 确保 p 值大于 0.1
    assert pvalue > 0.1
@pytest.mark.parametrize("method", ["basic", "percentile"])
@pytest.mark.parametrize("axis", [0, 1])
def test_bootstrap_vectorized_3samp(method, axis):
    def statistic(*data, axis=0):
        # 定义一个任意的向量化统计量函数
        return sum(sample.mean(axis) for sample in data)

    def statistic_1d(*data):
        # 同样的统计量函数，但不向量化
        for sample in data:
            assert sample.ndim == 1  # 断言每个样本数据是一维的
        return statistic(*data, axis=0)  # 调用向量化统计量函数

    np.random.seed(0)
    x = np.random.rand(4, 5)
    y = np.random.rand(4, 5)
    z = np.random.rand(4, 5)
    res1 = bootstrap((x, y, z), statistic, vectorized=True,
                     axis=axis, n_resamples=100, method=method, random_state=0)
    res2 = bootstrap((x, y, z), statistic_1d, vectorized=False,
                     axis=axis, n_resamples=100, method=method, random_state=0)
    assert_allclose(res1.confidence_interval, res2.confidence_interval)
    assert_allclose(res1.standard_error, res2.standard_error)


@pytest.mark.xfail_on_32bit("Failure is not concerning; see gh-14107")
@pytest.mark.parametrize("method", ["basic", "percentile", "BCa"])
@pytest.mark.parametrize("axis", [0, 1])
def test_bootstrap_vectorized_1samp(method, axis):
    def statistic(x, axis=0):
        # 定义一个任意的向量化统计量函数
        return x.mean(axis=axis)

    def statistic_1d(x):
        # 同样的统计量函数，但不向量化
        assert x.ndim == 1  # 断言输入数据是一维的
        return statistic(x, axis=0)  # 调用向量化统计量函数

    np.random.seed(0)
    x = np.random.rand(4, 5)
    res1 = bootstrap((x,), statistic, vectorized=True, axis=axis,
                     n_resamples=100, batch=None, method=method,
                     random_state=0)
    res2 = bootstrap((x,), statistic_1d, vectorized=False, axis=axis,
                     n_resamples=100, batch=10, method=method,
                     random_state=0)
    assert_allclose(res1.confidence_interval, res2.confidence_interval)
    assert_allclose(res1.standard_error, res2.standard_error)


@pytest.mark.parametrize("method", ["basic", "percentile", "BCa"])
def test_bootstrap_degenerate(method):
    data = 35 * [10000.]
    if method == "BCa":
        with np.errstate(invalid='ignore'):
            msg = "The BCa confidence interval cannot be calculated"
            with pytest.warns(stats.DegenerateDataWarning, match=msg):
                # 对于 BCa 方法，测试处理退化数据的情况
                res = bootstrap([data, ], np.mean, method=method)
                assert_equal(res.confidence_interval, (np.nan, np.nan))
    else:
        # 对于其他方法，测试处理正常数据的情况
        res = bootstrap([data, ], np.mean, method=method)
        assert_equal(res.confidence_interval, (10000., 10000.))
    assert_equal(res.standard_error, 0)


@pytest.mark.parametrize("method", ["basic", "percentile", "BCa"])
def test_bootstrap_gh15678(method):
    # 检查 gh-15678 是否已修复：当统计函数返回 Python 浮点数时，BCa 方法尝试给浮点数增加维度而失败的情况
    rng = np.random.default_rng(354645618886684)
    dist = stats.norm(loc=2, scale=4)
    # 从分布中生成随机样本数据，大小为100，使用给定的随机数生成器rng
    data = dist.rvs(size=100, random_state=rng)
    # 将数据包装成单元素元组
    data = (data,)
    # 使用bootstrap函数对数据进行自助法重采样，计算数据偏度，使用给定的方法和随机种子进行采样
    res = bootstrap(data, stats.skew, method=method, n_resamples=100,
                    random_state=np.random.default_rng(9563))
    # 同上，但此时不使用向量化处理
    ref = bootstrap(data, stats.skew, method=method, n_resamples=100,
                    random_state=np.random.default_rng(9563), vectorized=False)
    # 检查两次重采样结果的置信区间是否接近
    assert_allclose(res.confidence_interval, ref.confidence_interval)
    # 检查两次重采样结果的标准误差是否接近
    assert_allclose(res.standard_error, ref.standard_error)
    # 确认res.standard_error是NumPy的float64数据类型
    assert isinstance(res.standard_error, np.float64)
def test_bootstrap_min():
    # 检查 gh-15883 是否已修复：percentileofscore 应按照 'mean' 行为进行处理，而不应在 BCa 方法中触发 NaN
    rng = np.random.default_rng(1891289180021102)
    # 创建均值为 2，标准差为 4 的正态分布对象
    dist = stats.norm(loc=2, scale=4)
    # 从正态分布中生成包含 100 个随机数的样本数据
    data = dist.rvs(size=100, random_state=rng)
    # 计算样本数据的最小值
    true_min = np.min(data)
    # 将数据作为单元素元组传递给 bootstrap 函数，并指定方法为 BCa，重采样次数为 100
    res = bootstrap(data, np.min, method="BCa", n_resamples=100,
                    random_state=np.random.default_rng(3942))
    # 断言样本数据的真实最小值与 bootstrap 函数计算的置信区间下限相等
    assert true_min == res.confidence_interval.low
    # 对 -data 进行 bootstrap 计算最大值的置信区间，与前一个结果进行比较
    res2 = bootstrap(-np.array(data), np.max, method="BCa", n_resamples=100,
                     random_state=np.random.default_rng(3942))
    # 断言两个置信区间的边界相互接近
    assert_allclose(-res.confidence_interval.low,
                    res2.confidence_interval.high)
    assert_allclose(-res.confidence_interval.high,
                    res2.confidence_interval.low)


@pytest.mark.parametrize("additional_resamples", [0, 1000])
def test_re_bootstrap(additional_resamples):
    # 测试参数 `bootstrap_result` 的行为
    rng = np.random.default_rng(8958153316228384)
    # 生成一个包含 100 个随机数的数组
    x = rng.random(size=100)

    n1 = 1000
    n2 = additional_resamples
    n3 = n1 + additional_resamples

    rng = np.random.default_rng(296689032789913033)
    # 使用 percentile 方法进行 bootstrap，指定重采样次数和置信水平
    res = stats.bootstrap((x,), np.mean, n_resamples=n1, random_state=rng,
                          confidence_level=0.95, method='percentile')
    # 使用 BCa 方法进行 bootstrap，将之前的结果作为参数传入
    res = stats.bootstrap((x,), np.mean, n_resamples=n2, random_state=rng,
                          confidence_level=0.90, method='BCa',
                          bootstrap_result=res)

    rng = np.random.default_rng(296689032789913033)
    # 对相同数据使用 BCa 方法进行 bootstrap，以验证结果
    ref = stats.bootstrap((x,), np.mean, n_resamples=n3, random_state=rng,
                          confidence_level=0.90, method='BCa')

    # 断言两次 bootstrap 的标准误差相等
    assert_allclose(res.standard_error, ref.standard_error, rtol=1e-14)
    # 断言两次 bootstrap 的置信区间相等
    assert_allclose(res.confidence_interval, ref.confidence_interval,
                    rtol=1e-14)


@pytest.mark.xfail_on_32bit("Sensible to machine precision")
@pytest.mark.parametrize("method", ['basic', 'percentile', 'BCa'])
def test_bootstrap_alternative(method):
    rng = np.random.default_rng(5894822712842015040)
    # 创建均值为 2，标准差为 4 的正态分布对象，生成包含 100 个随机数的样本数据
    dist = stats.norm(loc=2, scale=4)
    data = (dist.rvs(size=(100), random_state=rng),)

    config = dict(data=data, statistic=np.std, random_state=rng, axis=-1)
    # 使用默认方法进行 bootstrap 计算
    t = stats.bootstrap(**config, confidence_level=0.9)

    config.update(dict(n_resamples=0, bootstrap_result=t))
    # 使用不同的 alternative 参数进行 bootstrap 计算，验证结果
    l = stats.bootstrap(**config, confidence_level=0.95, alternative='less')
    g = stats.bootstrap(**config, confidence_level=0.95, alternative='greater')

    # 断言不同 alternative 参数下的置信区间边界与默认方法的置信区间边界相接近
    assert_allclose(l.confidence_interval.high, t.confidence_interval.high,
                    rtol=1e-14)
    assert_allclose(g.confidence_interval.low, t.confidence_interval.low,
                    rtol=1e-14)
    # 断言 less 方法的置信区间下限为负无穷，greater 方法的置信区间上限为正无穷
    assert np.isneginf(l.confidence_interval.low)
    assert np.isposinf(g.confidence_interval.high)
    # 使用 pytest 的 `raises` 断言检查是否抛出 ValueError 异常，并验证异常消息中包含特定字符串
    with pytest.raises(ValueError, match='`alternative` must be one of'):
        # 调用 stats 模块中的 bootstrap 函数，传入 config 参数，并设置 alternative 参数为 'ekki-ekki'
        stats.bootstrap(**config, alternative='ekki-ekki')
def test_jackknife_resample():
    # 设置数组形状
    shape = 3, 4, 5, 6
    # 设定随机种子以便可重现性
    np.random.seed(0)
    # 生成指定形状的随机数组
    x = np.random.rand(*shape)
    # 获取下一个杰克纳夫重抽样的结果
    y = next(_resampling._jackknife_resample(x))

    for i in range(shape[-1]):
        # 每个重抽样结果沿倒数第二个轴索引
        # （最后一个轴是将进行统计的轴 / 消费的轴）
        slc = y[..., i, :]
        # 生成期望的删除了第 i 个元素的数组
        expected = np.delete(x, i, axis=-1)

        # 断言每个切片与期望的删除操作结果一致
        assert np.array_equal(slc, expected)

    # 合并所有批次的杰克纳夫重抽样结果
    y2 = np.concatenate(list(_resampling._jackknife_resample(x, batch=2)),
                        axis=-2)
    # 断言合并后的结果与单次抽样的结果一致
    assert np.array_equal(y2, y)


@pytest.mark.parametrize("rng_name", ["RandomState", "default_rng"])
def test_bootstrap_resample(rng_name):
    # 获取指定的随机数生成器
    rng = getattr(np.random, rng_name, None)
    if rng is None:
        pytest.skip(f"{rng_name} not available.")
    # 使用第一个随机数生成器创建两个实例
    rng1 = rng(0)
    rng2 = rng(0)

    # 设置重抽样次数和数组形状
    n_resamples = 10
    shape = 3, 4, 5, 6

    np.random.seed(0)
    # 生成指定形状的随机数组
    x = np.random.rand(*shape)
    # 使用自助法进行重抽样
    y = _resampling._bootstrap_resample(x, n_resamples, random_state=rng1)

    for i in range(n_resamples):
        # 每个重抽样结果沿倒数第二个轴索引
        # （最后一个轴是将进行统计的轴 / 消费的轴）
        slc = y[..., i, :]

        # 生成期望的按照第二个随机数生成器生成的索引进行抽样后的数组
        js = rng_integers(rng2, 0, shape[-1], shape[-1])
        expected = x[..., js]

        # 断言每个切片与期望的抽样结果一致
        assert np.array_equal(slc, expected)


@pytest.mark.parametrize("score", [0, 0.5, 1])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_percentile_of_score(score, axis):
    # 设置数组形状
    shape = 10, 20, 30
    np.random.seed(0)
    # 生成指定形状的随机数组
    x = np.random.rand(*shape)
    # 计算指定得分在指定轴上的百分位数
    p = _resampling._percentile_of_score(x, score, axis=-1)

    def vectorized_pos(a, score, axis):
        # 应用向量化的百分位数计算函数
        return np.apply_along_axis(stats.percentileofscore, axis, a, score)

    # 使用向量化的函数计算期望的百分位数
    p2 = vectorized_pos(x, score, axis=-1)/100

    # 断言计算结果接近（在误差范围内）
    assert_allclose(p, p2, 1e-15)


def test_percentile_along_axis():
    # _percentile_along_axis 与 np.percentile 的区别在于
    # np.percentile 对每个轴切片获取所有百分位数，
    # 而 _percentile_along_axis 获取与每个轴切片相对应的百分位数

    # 设置数组形状
    shape = 10, 20
    np.random.seed(0)
    # 生成指定形状的随机数组
    x = np.random.rand(*shape)
    # 生成随机的百分位数数组
    q = np.random.rand(*shape[:-1]) * 100
    # 计算沿轴上的百分位数
    y = _resampling._percentile_along_axis(x, q)

    for i in range(shape[0]):
        res = y[i]
        expected = np.percentile(x[i], q[i], axis=-1)
        # 断言计算结果接近（在误差范围内）
        assert_allclose(res, expected, 1e-15)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_vectorize_statistic(axis):
    # 测试 _vectorize_statistic 是否能够沿着 `axis` 向量化统计量

    def statistic(*data, axis):
        # 一个任意的向量化统计量
        return sum(sample.mean(axis) for sample in data)

    def statistic_1d(*data):
        # 相同的统计量，未向量化
        for sample in data:
            assert sample.ndim == 1
        return statistic(*data, axis=0)

    # 向量化非向量化的统计量
    statistic2 = _resampling._vectorize_statistic(statistic_1d)

    np.random.seed(0)
    # 创建一个形状为 (4, 5, 6) 的随机数组 x
    x = np.random.rand(4, 5, 6)
    # 创建一个形状为 (4, 1, 6) 的随机数组 y
    y = np.random.rand(4, 1, 6)
    # 创建一个形状为 (1, 5, 6) 的随机数组 z
    z = np.random.rand(1, 5, 6)
    
    # 使用 statistic 函数计算统计量，返回结果给 res1
    res1 = statistic(x, y, z, axis=axis)
    # 使用 statistic2 函数计算统计量，返回结果给 res2
    res2 = statistic2(x, y, z, axis=axis)
    
    # 断言 res1 和 res2 在允许误差范围内相等
    assert_allclose(res1, res2)
@pytest.mark.slow
@pytest.mark.parametrize("method", ["basic", "percentile", "BCa"])
def test_vector_valued_statistic(method):
    # 生成正态分布参数的最大似然估计(MLE)的95%置信区间，重复100次，每次样本大小为100。
    # 检查置信区间大约包含真实参数的情况约95次。
    # 置信区间是估计的并且随机的；测试失败不一定意味着有错误。比 `counts` 的值更重要的是输出的形状是否正确。

    # 使用指定的随机种子创建随机数生成器
    rng = np.random.default_rng(2196847219)
    # 正态分布的参数 mu=1, sigma=0.5
    params = 1, 0.5
    # 生成一个大小为 (100, 100) 的正态分布样本
    sample = stats.norm.rvs(*params, size=(100, 100), random_state=rng)

    # 定义统计函数，计算均值和标准差
    def statistic(data, axis):
        return np.asarray([np.mean(data, axis),
                           np.std(data, axis, ddof=1)])

    # 对样本数据进行自助法(bootstrap)，使用给定的统计函数和方法计算
    res = bootstrap((sample,), statistic, method=method, axis=-1,
                    n_resamples=9999, batch=200)

    # 计算置信区间包含真实参数的次数
    counts = np.sum((res.confidence_interval.low.T < params)
                    & (res.confidence_interval.high.T > params),
                    axis=0)
    # 断言置信区间大约包含真实参数的次数均大于等于90次
    assert np.all(counts >= 90)
    # 断言置信区间大约包含真实参数的次数均小于等于100次
    assert np.all(counts <= 100)
    # 断言置信区间低值的形状为 (2, 100)
    assert res.confidence_interval.low.shape == (2, 100)
    # 断言置信区间高值的形状为 (2, 100)
    assert res.confidence_interval.high.shape == (2, 100)
    # 断言标准误差的形状为 (2, 100)
    assert res.standard_error.shape == (2, 100)
    # 断言自助法分布的形状为 (2, 100, 9999)
    assert res.bootstrap_distribution.shape == (2, 100, 9999)


@pytest.mark.slow
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_vector_valued_statistic_gh17715():
    # gh-17715 报告了在多样本统计的BCa扩展中引入的错误；`len` 应为 `.shape[-1]`。检查这个问题是否解决。

    # 使用指定的随机种子创建随机数生成器
    rng = np.random.default_rng(141921000979291141)

    # 计算 Concordance 统计量的函数
    def concordance(x, y, axis):
        xm = x.mean(axis)
        ym = y.mean(axis)
        cov = ((x - xm[..., None]) * (y - ym[..., None])).mean(axis)
        return (2 * cov) / (x.var(axis) + y.var(axis) + (xm - ym) ** 2)

    # 统计函数，计算给定数据的 Concordance 统计量
    def statistic(tp, tn, fp, fn, axis):
        actual = tp + fp
        expected = tp + fn
        return np.nan_to_num(concordance(actual, expected, axis))

    # 增加维度的统计函数，用于多样本统计
    def statistic_extradim(*args, axis):
        return statistic(*args, axis)[np.newaxis, ...]

    # 原始数据，每列代表 (tp, tn, fp, fn)
    data = [[4, 0, 0, 2],
            [2, 1, 2, 1],
            [0, 6, 0, 0],
            [0, 6, 3, 0],
            [0, 8, 1, 0]]
    data = np.array(data).T

    # 对数据应用自助法，使用 statistic_extradim 函数，paired=True 表示配对样本
    res = bootstrap(data, statistic_extradim, random_state=rng, paired=True)
    # 参考数据，使用原始 statistic 函数，验证 BCa 扩展问题是否解决
    ref = bootstrap(data, statistic, random_state=rng, paired=True)
    # 断言置信区间低值的第一个元素与参考数据一致，允许误差范围 1e-15
    assert_allclose(res.confidence_interval.low[0],
                    ref.confidence_interval.low, atol=1e-15)
    # 断言置信区间高值的第一个元素与参考数据一致，允许误差范围 1e-15
    assert_allclose(res.confidence_interval.high[0],
                    ref.confidence_interval.high, atol=1e-15)


def test_gh_20850():
    # 测试 gh-20850 的问题

    # 使用指定的随机种子创建随机数生成器
    rng = np.random.default_rng(2085020850)
    # 创建大小为 (10, 2) 的随机数组
    x = rng.random((10, 2))
    # 创建大小为 (11, 2) 的随机数组
    y = rng.random((11, 2))
    # 定义一个函数 `statistic`，用于计算在给定轴上的 t 检验统计量
    def statistic(x, y, axis):
        return stats.ttest_ind(x, y, axis=axis).statistic

    # 对 (x, y) 执行自助法（bootstrap），默认统计函数 `statistic`
    stats.bootstrap((x, y), statistic)
    
    # 对 (x.T, y.T) 执行自助法（bootstrap），指定轴为1，使用统计函数 `statistic`
    stats.bootstrap((x.T, y.T), statistic, axis=1)
    
    # 当沿指定轴的形状相同时，警告用户忽略轴指定的维度
    message = "Ignoring the dimension specified by `axis`..."
    
    # 使用 pytest 来检测未来可能的警告，匹配消息字符串 `message`
    with pytest.warns(FutureWarning, match=message):
        # 尝试对 (x, y[:10, 0]) 执行自助法（bootstrap），这在版本 1.16 之后将不起作用
        stats.bootstrap((x, y[:10, 0]), statistic)
    
    with pytest.warns(FutureWarning, match=message):
        # 尝试对 (x, y[:10, 0:1]) 执行自助法（bootstrap），这将起作用
        stats.bootstrap((x, y[:10, 0:1]), statistic)
    
    with pytest.warns(FutureWarning, match=message):
        # 尝试对 (x.T, y.T[0:1, :10]) 执行自助法（bootstrap），指定轴为1，这将起作用
        stats.bootstrap((x.T, y.T[0:1, :10]), statistic, axis=1)
# --- Test Monte Carlo Hypothesis Test --- #

# 定义一个测试类 TestMonteCarloHypothesisTest
class TestMonteCarloHypothesisTest:
    
    # 设置比较 p 值时的容差
    atol = 2.5e-2  # for comparing p-value

    # 定义一个方法，生成随机变量并返回
    def get_rvs(self, rvs_in, rs, dtype=None, xp=np):
        # 返回一个函数，该函数使用指定的随机种子生成随机变量，并转换为指定的数据类型
        return lambda *args, **kwds: xp.asarray(rvs_in(*args, random_state=rs, **kwds),
                                                dtype=dtype)

    # 定义一个方法，计算统计量
    def get_statistic(self, xp):
        # 定义统计函数，计算均值、方差和样本数量，返回统计量
        def statistic(x, axis):
            m = xp.mean(x, axis=axis)
            v = xp.var(x, axis=axis, correction=1)
            n = x.shape[axis]
            return m / (v/n)**0.5
            # 也可以使用下面的注释代码进行 t 检验
            # return stats.ttest_1samp(x, popmean=0., axis=axis).statistic)
        return statistic

    # 声明一个装饰器，使该方法兼容不同的数组 API
    @array_api_compatible
    # 定义一个测试方法，用于验证输入的有效性是否能正确引发相应的错误消息
    def test_input_validation(self, xp):
        # 创建一个NumPy数组，数据为[1., 2., 3.]
        data = xp.asarray([1., 2., 3.])
        
        # 定义一个内部函数 `stat`，用于计算数组的平均值
        def stat(x, axis=None):
            return xp.mean(x, axis=axis)
        
        # 设置错误消息，用于匹配的异常应为 `ValueError`
        message = "Array shapes are incompatible for broadcasting."
        # 创建临时变量 `temp`，包含两个不兼容形状的NumPy数组
        temp = (xp.zeros((2, 5)), xp.zeros((3, 5)))
        # 创建随机变量生成器 `rvs`，包含两个正态分布的随机变量生成函数
        rvs = (stats.norm.rvs, stats.norm.rvs)
        # 使用 `pytest.raises` 验证调用 `monte_carlo_test` 函数时是否会引发 `ValueError` 异常
        with pytest.raises(ValueError, match=message):
            monte_carlo_test(temp, rvs, lambda x, y, axis: 1, axis=-1)
        
        # 设置错误消息，验证 `axis` 是否为整数类型的异常
        message = "`axis` must be an integer."
        with pytest.raises(ValueError, match=message):
            monte_carlo_test(data, stats.norm.rvs, stat, axis=1.5)
        
        # 设置错误消息，验证 `vectorized` 参数是否为合法值的异常
        message = "`vectorized` must be `True`, `False`, or `None`."
        with pytest.raises(ValueError, match=message):
            monte_carlo_test(data, stats.norm.rvs, stat, vectorized=1.5)
        
        # 设置错误消息，验证 `rvs` 参数是否为可调用函数或函数序列的异常
        message = "`rvs` must be callable or sequence of callables."
        with pytest.raises(TypeError, match=message):
            monte_carlo_test(data, None, stat)
        with pytest.raises(TypeError, match=message):
            temp = xp.asarray([[1., 2.], [3., 4.]])
            monte_carlo_test(temp, [lambda x: x, None], stat)
        
        # 设置错误消息，验证如果 `rvs` 参数为函数序列时是否正确的异常
        message = "If `rvs` is a sequence..."
        with pytest.raises(ValueError, match=message):
            temp = xp.asarray([[1., 2., 3.]])
            monte_carlo_test(temp, [lambda x: x, lambda x: x], stat)
        
        # 设置错误消息，验证 `statistic` 是否为可调用函数的异常
        message = "`statistic` must be callable."
        with pytest.raises(TypeError, match=message):
            monte_carlo_test(data, stats.norm.rvs, None)
        
        # 设置错误消息，验证 `n_resamples` 是否为正整数的异常
        message = "`n_resamples` must be a positive integer."
        with pytest.raises(ValueError, match=message):
            monte_carlo_test(data, stats.norm.rvs, stat, n_resamples=-1000)
        with pytest.raises(ValueError, match=message):
            monte_carlo_test(data, stats.norm.rvs, stat, n_resamples=1000.5)
        
        # 设置错误消息，验证 `batch` 是否为正整数或 `None` 的异常
        message = "`batch` must be a positive integer or None."
        with pytest.raises(ValueError, match=message):
            monte_carlo_test(data, stats.norm.rvs, stat, batch=-1000)
        with pytest.raises(ValueError, match=message):
            monte_carlo_test(data, stats.norm.rvs, stat, batch=1000.5)
        
        # 设置错误消息，验证 `alternative` 是否在指定范围内的异常
        message = "`alternative` must be in..."
        with pytest.raises(ValueError, match=message):
            monte_carlo_test(data, stats.norm.rvs, stat, alternative='ekki')
        
        # 设置错误消息，验证 `monte_carlo_test` 函数在引发 `ValueError` 时是否包含预期的消息
        message = "Signature inspection of statistic"
        # 定义一个生成随机变量的函数 `rvs`，确保其返回值是一个NumPy数组
        def rvs(size):
            return xp.asarray(stats.norm.rvs(size=size))
        try:
            # 尝试调用 `monte_carlo_test` 函数，确保其引发的异常消息以 `message` 开头
            monte_carlo_test(data, rvs, xp.mean)
        except ValueError as e:
            assert str(e).startswith(message)
    # 定义测试方法，用于验证输入的统计学函数是否向量化
    def test_input_validation_xp(self, xp):
        # 定义非向量化的统计函数，计算向量的均值
        def non_vectorized_statistic(x):
            return xp.mean(x)

        # 错误信息，当统计函数不是向量化时抛出异常
        message = "`statistic` must be vectorized..."
        # 创建示例数据，将其转换为指定的数组类型
        sample = xp.asarray([1., 2., 3.])
        # 如果使用的是numpy数组，执行蒙特卡洛测试，检验非向量化统计函数是否引发异常
        if is_numpy(xp):
            monte_carlo_test(sample, stats.norm.rvs, non_vectorized_statistic)
            return

        # 当不是numpy数组时，验证非向量化统计函数是否抛出预期的异常
        with pytest.raises(ValueError, match=message):
            monte_carlo_test(sample, stats.norm.rvs, non_vectorized_statistic)
        with pytest.raises(ValueError, match=message):
            monte_carlo_test(sample, stats.norm.rvs, xp.mean, vectorized=False)

    @pytest.mark.xslow
    @array_api_compatible
    # 测试批处理功能，在指定的数组API下执行
    def test_batch(self, xp):
        # 确保批处理参数被正确地应用，通过检查调用`statistic`时提供的最大批处理大小
        rng = np.random.default_rng(23492340193)
        # 创建指定数组API的数组，填充标准正态分布的随机值
        x = xp.asarray(rng.standard_normal(size=10))

        # 根据当前数组API调整测试用的数组库
        xp_test = array_namespace(x)  # numpy.std doesn't have `correction`

        # 定义统计函数，根据轴计算批处理大小，并更新统计次数，最终返回统计结果
        def statistic(x, axis):
            batch_size = 1 if x.ndim == 1 else x.shape[0]
            statistic.batch_size = max(batch_size, statistic.batch_size)
            statistic.counter += 1
            return self.get_statistic(xp_test)(x, axis=axis)
        statistic.counter = 0
        statistic.batch_size = 0

        # 准备测试参数字典，包括样本数据、统计函数、重抽样次数和向量化标志
        kwds = {'sample': x, 'statistic': statistic,
                'n_resamples': 1000, 'vectorized': True}

        # 获取随机变量生成函数，用于蒙特卡洛模拟测试
        kwds['rvs'] = self.get_rvs(stats.norm.rvs, np.random.default_rng(328423), xp=xp)
        # 执行批处理为1的蒙特卡洛测试
        res1 = monte_carlo_test(batch=1, **kwds)
        assert_equal(statistic.counter, 1001)
        assert_equal(statistic.batch_size, 1)

        # 修改批处理为50，再次执行蒙特卡洛测试
        kwds['rvs'] = self.get_rvs(stats.norm.rvs, np.random.default_rng(328423), xp=xp)
        statistic.counter = 0
        res2 = monte_carlo_test(batch=50, **kwds)
        assert_equal(statistic.counter, 21)
        assert_equal(statistic.batch_size, 50)

        # 未指定批处理参数时，执行默认批处理为1000的蒙特卡洛测试
        kwds['rvs'] = self.get_rvs(stats.norm.rvs, np.random.default_rng(328423), xp=xp)
        statistic.counter = 0
        res3 = monte_carlo_test(**kwds)
        assert_equal(statistic.counter, 2)
        assert_equal(statistic.batch_size, 1000)

        # 检查不同批处理条件下的p值是否一致
        xp_assert_equal(res1.pvalue, res3.pvalue)
        xp_assert_equal(res2.pvalue, res3.pvalue)

    @array_api_compatible
    @pytest.mark.parametrize('axis', range(-3, 3))
    # 定义一个测试方法，用于测试在不同 `axis` 参数有效值下，Nd-array 样本的处理是否正确，并确保非默认的 dtype 被保持
    def test_axis_dtype(self, axis, xp):
        # 使用特定种子创建随机数生成器对象
        rng = np.random.default_rng(2389234)
        # 定义数组的维度大小，将指定轴的大小设为100
        size = [2, 3, 4]
        size[axis] = 100

        # 确定非默认的 dtype 类型
        dtype_default = xp.asarray(1.).dtype
        dtype_str = 'float32' if ("64" in str(dtype_default)) else 'float64'
        dtype_np = getattr(np, dtype_str)
        dtype = getattr(xp, dtype_str)

        # ttest_1samp 是 CPU 数组 API 兼容的，但最好在此测试中包括 CuPy。我们将使用 NumPy 数组执行 ttest_1samp，但所有其他操作将完全使用数组 API 兼容的代码。
        # 使用随机数生成器创建指定 dtype 类型和大小的标准正态分布数组
        x = rng.standard_normal(size=size, dtype=dtype_np)
        # 计算标准正态分布数组 x 的单样本 t 检验
        expected = stats.ttest_1samp(x, popmean=0., axis=axis)

        # 将数组 x 转换为 xp 的数组对象，并指定 dtype
        x = xp.asarray(x, dtype=dtype)
        # 获取 xp 数组的命名空间，用于测试
        xp_test = array_namespace(x)  # numpy.std doesn't have `correction`
        # 获取统计量
        statistic = self.get_statistic(xp_test)
        # 使用 stats.norm.rvs 函数获取随机变量样本
        rvs = self.get_rvs(stats.norm.rvs, rng, dtype=dtype, xp=xp)

        # 进行蒙特卡洛模拟检验
        res = monte_carlo_test(x, rvs, statistic, vectorized=True,
                               n_resamples=20000, axis=axis)

        # 将期望的统计量和 p 值转换为 xp 数组的 dtype 类型
        ref_statistic = xp.asarray(expected.statistic, dtype=dtype)
        ref_pvalue = xp.asarray(expected.pvalue, dtype=dtype)
        # 断言蒙特卡洛模拟检验结果的统计量和 p 值与期望的结果非常接近
        xp_assert_close(res.statistic, ref_statistic)
        xp_assert_close(res.pvalue, ref_pvalue, atol=self.atol)

    # 使用数组 API 兼容装饰器，参数化测试不同的 alternative 参数
    @array_api_compatible
    @pytest.mark.parametrize('alternative', ("two-sided", "less", "greater"))
    def test_alternative(self, alternative, xp):
        # 测试 alternative 参数的预期行为
        rng = np.random.default_rng(65723433)

        # 使用随机数生成器创建大小为30的标准正态分布数组
        x = rng.standard_normal(size=30)
        # 对数组 x 执行单样本 t 检验，使用不同的 alternative 参数
        ref = stats.ttest_1samp(x, 0., alternative=alternative)

        # 将数组 x 转换为 xp 的数组对象
        x = xp.asarray(x)
        # 获取 xp 数组的命名空间，用于测试
        xp_test = array_namespace(x)  # numpy.std doesn't have `correction`
        # 获取统计量
        statistic = self.get_statistic(xp_test)
        # 使用 stats.norm.rvs 函数获取随机变量样本
        rvs = self.get_rvs(stats.norm.rvs, rng, xp=xp)

        # 进行蒙特卡洛模拟检验，使用不同的 alternative 参数
        res = monte_carlo_test(x, rvs, statistic, alternative=alternative)

        # 断言蒙特卡洛模拟检验结果的统计量和 p 值与期望的结果非常接近
        xp_assert_close(res.statistic, xp.asarray(ref.statistic))
        xp_assert_close(res.pvalue, xp.asarray(ref.pvalue), atol=self.atol)

    # 下面的测试涉及尚未兼容数组 API 的统计量。
    # 当这些统计量转换为数组 API 兼容时，可以进行转换。
    @pytest.mark.slow
    @pytest.mark.parametrize('alternative', ("less", "greater"))
    @pytest.mark.parametrize('a', np.linspace(-0.5, 0.5, 5))  # skewness
    def test_against_ks_1samp(self, alternative, a):
        # 检验 monte_carlo_test 能否复现 ks_1samp 的 p 值
        rng = np.random.default_rng(65723433)

        # 生成服从 skewnorm 分布的随机数，用于检验
        x = stats.skewnorm.rvs(a=a, size=30, random_state=rng)

        # 计算期望的 ks_1samp 统计量和 p 值
        expected = stats.ks_1samp(x, stats.norm.cdf, alternative=alternative)

        # 定义一维统计函数，使用 stats.ks_1samp 进行计算
        def statistic1d(x):
            return stats.ks_1samp(x, stats.norm.cdf, mode='asymp',
                                  alternative=alternative).statistic

        # 获取服从标准正态分布的随机数，用于 monte_carlo_test
        norm_rvs = self.get_rvs(stats.norm.rvs, rng)

        # 使用 monte_carlo_test 进行蒙特卡洛模拟检验
        res = monte_carlo_test(x, norm_rvs, statistic1d,
                               n_resamples=1000, vectorized=False,
                               alternative=alternative)

        # 断言检验结果的统计量与期望一致
        assert_allclose(res.statistic, expected.statistic)

        # 根据 alternative 不同，断言检验结果的 p 值与期望一致
        if alternative == 'greater':
            assert_allclose(res.pvalue, expected.pvalue, atol=self.atol)
        elif alternative == 'less':
            assert_allclose(1-res.pvalue, expected.pvalue, atol=self.atol)

    @pytest.mark.parametrize('hypotest', (stats.skewtest, stats.kurtosistest))
    @pytest.mark.parametrize('alternative', ("less", "greater", "two-sided"))
    @pytest.mark.parametrize('a', np.linspace(-2, 2, 5))  # skewness
    def test_against_normality_tests(self, hypotest, alternative, a):
        # 检验 monte_carlo_test 能否复现正态性检验的 p 值
        rng = np.random.default_rng(85723405)

        # 生成服从 skewnorm 分布的随机数，用于检验
        x = stats.skewnorm.rvs(a=a, size=150, random_state=rng)

        # 计算期望的正态性检验统计量和 p 值
        expected = hypotest(x, alternative=alternative)

        # 定义统计函数，使用给定的正态性检验函数进行计算
        def statistic(x, axis):
            return hypotest(x, axis=axis).statistic

        # 获取服从标准正态分布的随机数，用于 monte_carlo_test
        norm_rvs = self.get_rvs(stats.norm.rvs, rng)

        # 使用 monte_carlo_test 进行蒙特卡洛模拟检验
        res = monte_carlo_test(x, norm_rvs, statistic, vectorized=True,
                               alternative=alternative)

        # 断言检验结果的统计量与期望一致
        assert_allclose(res.statistic, expected.statistic)

        # 断言检验结果的 p 值与期望一致
        assert_allclose(res.pvalue, expected.pvalue, atol=self.atol)

    @pytest.mark.parametrize('a', np.arange(-2, 3))  # skewness parameter
    def test_against_normaltest(self, a):
        # 检验 monte_carlo_test 能否复现 normaltest 的 p 值
        rng = np.random.default_rng(12340513)

        # 生成服从 skewnorm 分布的随机数，用于检验
        x = stats.skewnorm.rvs(a=a, size=150, random_state=rng)

        # 计算期望的 normaltest 统计量和 p 值
        expected = stats.normaltest(x)

        # 定义统计函数，使用 stats.normaltest 进行计算
        def statistic(x, axis):
            return stats.normaltest(x, axis=axis).statistic

        # 获取服从标准正态分布的随机数，用于 monte_carlo_test
        norm_rvs = self.get_rvs(stats.norm.rvs, rng)

        # 使用 monte_carlo_test 进行蒙特卡洛模拟检验
        res = monte_carlo_test(x, norm_rvs, statistic, vectorized=True,
                               alternative='greater')

        # 断言检验结果的统计量与期望一致
        assert_allclose(res.statistic, expected.statistic)

        # 断言检验结果的 p 值与期望一致
        assert_allclose(res.pvalue, expected.pvalue, atol=self.atol)

    @pytest.mark.xslow
    @pytest.mark.parametrize('a', np.linspace(-0.5, 0.5, 5))  # skewness
    def test_against_cramervonmises(self, a):
        # 测试 monte_carlo_test 能否复现 cramervonmises 的 p 值
        rng = np.random.default_rng(234874135)

        # 生成 skewnorm 分布的随机变量 x
        x = stats.skewnorm.rvs(a=a, size=30, random_state=rng)
        
        # 计算 x 对应的 cramervonmises 统计量和 p 值的期望
        expected = stats.cramervonmises(x, stats.norm.cdf)

        def statistic1d(x):
            return stats.cramervonmises(x, stats.norm.cdf).statistic

        # 生成标准正态分布的随机变量 norm_rvs
        norm_rvs = self.get_rvs(stats.norm.rvs, rng)
        
        # 运行 monte_carlo_test，比较统计量和 p 值
        res = monte_carlo_test(x, norm_rvs, statistic1d,
                               n_resamples=1000, vectorized=False,
                               alternative='greater')

        # 断言检查统计量的一致性
        assert_allclose(res.statistic, expected.statistic)
        # 断言检查 p 值的一致性，使用给定的容差 self.atol
        assert_allclose(res.pvalue, expected.pvalue, atol=self.atol)

    @pytest.mark.slow
    @pytest.mark.parametrize('dist_name', ('norm', 'logistic'))
    @pytest.mark.parametrize('i', range(5))
    def test_against_anderson(self, dist_name, i):
        # 测试 monte_carlo_test 能否复现 anderson 的结果。注意：
        # `anderson` 不提供 p 值，而是提供一组显著性水平和相关的检验统计量临界值。
        # `i` 用于索引这个列表。

        def fun(a):
            rng = np.random.default_rng(394295467)
            # 生成 tukeylambda 分布的随机变量 x
            x = stats.tukeylambda.rvs(a, size=100, random_state=rng)
            # 计算 x 对应的 anderson 测试结果
            expected = stats.anderson(x, dist_name)
            # 返回统计量与给定临界值的差
            return expected.statistic - expected.critical_values[i]
        
        # 使用根据 fun 函数计算出的结果作为初始值求解根
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            sol = root(fun, x0=0)
        # 断言检查求解是否成功
        assert sol.success

        # 获取与求解得到的临界值相关的显著性水平 (p 值)
        a = sol.x[0]
        rng = np.random.default_rng(394295467)
        # 根据求解得到的参数 a 生成 tukeylambda 分布的随机变量 x
        x = stats.tukeylambda.rvs(a, size=100, random_state=rng)
        # 计算 x 对应的 anderson 测试结果
        expected = stats.anderson(x, dist_name)
        # 获取期望的统计量和显著性水平 (p 值)
        expected_stat = expected.statistic
        expected_p = expected.significance_level[i]/100

        # 运行等效的 Monte Carlo 测试并比较结果
        def statistic1d(x):
            return stats.anderson(x, dist_name).statistic

        # 生成指定分布 (norm 或 logistic) 的随机变量 dist_rvs
        dist_rvs = self.get_rvs(getattr(stats, dist_name).rvs, rng)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            res = monte_carlo_test(x, dist_rvs,
                                   statistic1d, n_resamples=1000,
                                   vectorized=False, alternative='greater')

        # 断言检查统计量的一致性
        assert_allclose(res.statistic, expected_stat)
        # 断言检查 p 值的一致性，使用给定的容差 2*self.atol
        assert_allclose(res.pvalue, expected_p, atol=2*self.atol)
    # 测试确保 p 值不会为零，使用偏差估计来保证
    # 参考 monte_carlo_test [1]
    def test_p_never_zero(self):
        # 使用指定种子初始化随机数生成器
        rng = np.random.default_rng(2190176673029737545)
        # 创建一个长度为 100 的全零数组
        x = np.zeros(100)
        # 调用 monte_carlo_test 函数进行测试，期望结果的 p 值为 0.0001
        res = monte_carlo_test(x, rng.random, np.mean,
                               vectorized=True, alternative='less')
        # 断言测试结果的 p 值为 0.0001
        assert res.pvalue == 0.0001

    # 测试 `monte_carlo_test` 是否能够复现 `ttest_ind` 的结果
    def test_against_ttest_ind(self):
        # 使用指定种子初始化随机数生成器
        rng = np.random.default_rng(219017667302737545)
        # 创建不同形状的数据用于测试，支持广播
        data = rng.random(size=(2, 5)), rng.random(size=7)
        rvs = rng.normal, rng.normal
        # 定义统计函数，计算 ttest_ind 的统计量
        def statistic(x, y, axis):
            return stats.ttest_ind(x, y, axis).statistic

        # 调用 monte_carlo_test 函数进行测试
        res = stats.monte_carlo_test(data, rvs, statistic, axis=-1)
        # 调用 ttest_ind 函数获取参考结果
        ref = stats.ttest_ind(data[0], [data[1]], axis=-1)
        # 断言测试结果的统计量与参考结果的统计量接近
        assert_allclose(res.statistic, ref.statistic)
        # 断言测试结果的 p 值与参考结果的 p 值接近，允许相对误差为 2e-2
        assert_allclose(res.pvalue, ref.pvalue, rtol=2e-2)

    # 测试 `monte_carlo_test` 是否能够复现 `f_oneway` 的结果
    def test_against_f_oneway(self):
        # 使用指定种子初始化随机数生成器
        rng = np.random.default_rng(219017667302737545)
        # 创建不同形状的数据用于测试
        data = (rng.random(size=(2, 100)), rng.random(size=(2, 101)),
                rng.random(size=(2, 102)), rng.random(size=(2, 103)))
        rvs = rng.normal, rng.normal, rng.normal, rng.normal

        # 定义统计函数，计算 f_oneway 的统计量
        def statistic(*args, axis):
            return stats.f_oneway(*args, axis=axis).statistic

        # 调用 monte_carlo_test 函数进行测试
        res = stats.monte_carlo_test(data, rvs, statistic, axis=-1,
                                     alternative='greater')
        # 调用 f_oneway 函数获取参考结果
        ref = stats.f_oneway(*data, axis=-1)

        # 断言测试结果的统计量与参考结果的统计量接近
        assert_allclose(res.statistic, ref.statistic)
        # 断言测试结果的 p 值与参考结果的 p 值接近，允许绝对误差为 1e-2
        assert_allclose(res.pvalue, ref.pvalue, atol=1e-2)

    # 标记为 'fail_slow' 的测试用例，设定为 2，表示测试运行较慢
    # 如果在 32 位系统上失败，则预期的统计量可能依赖于样本顺序
    @pytest.mark.fail_slow(2)
    @pytest.mark.xfail_on_32bit("Statistic may not depend on sample order on 32-bit")
    def test_finite_precision_statistic(self):
        # 一些统计量在理论上应该相等时，由于数值精度问题可能返回不同的值。
        # 测试 `monte_carlo_test` 是否在某种程度上考虑了这一点。
        # 使用指定种子初始化随机数生成器
        rng = np.random.default_rng(2549824598234528)
        # 设定重新采样的次数
        n_resamples = 9999
        # 定义随机变量生成函数，使用伯努利分布生成值
        def rvs(size):
            return 1. * stats.bernoulli(p=0.333).rvs(size=size, random_state=rng)

        # 生成一个长度为 100 的随机变量数组
        x = rvs(100)
        # 调用 monte_carlo_test 函数进行测试
        res = stats.monte_carlo_test(x, rvs, np.var, alternative='less',
                                     n_resamples=n_resamples)
        # 展示容忍度的重要性
        c0 = np.sum(res.null_distribution <= res.statistic)
        c1 = np.sum(res.null_distribution <= res.statistic*(1+1e-15))
        # 断言通过容忍度的改变可以得到不同的结果
        assert c0 != c1
        # 断言测试结果的 p 值为 (c1 + 1)/(n_resamples + 1)
        assert res.pvalue == (c1 + 1)/(n_resamples + 1)
class TestPower:
    def test_input_validation(self):
        # test that the appropriate error messages are raised for invalid input

        # 创建一个指定种子的随机数生成器对象
        rng = np.random.default_rng(8519895914314711673)

        # 使用 scipy.stats.ttest_ind 作为测试函数
        test = stats.ttest_ind
        # 生成两个正态分布的随机数生成器
        rvs = (rng.normal, rng.normal)
        # 设定观测值数量
        n_observations = (10, 12)

        # 验证当 vectorized 参数为非法值时，是否引发 ValueError 异常
        message = "`vectorized` must be `True`, `False`, or `None`."
        with pytest.raises(ValueError, match=message):
            power(test, rvs, n_observations, vectorized=1.5)

        # 验证当 rvs 参数为 None 或包含非法值时，是否引发 TypeError 异常
        message = "`rvs` must be callable or sequence of callables."
        with pytest.raises(TypeError, match=message):
            power(test, None, n_observations)
        with pytest.raises(TypeError, match=message):
            power(test, (rng.normal, 'ekki'), n_observations)

        # 验证当 rvs 和 n_observations 的长度不匹配时，是否引发 ValueError 异常
        message = "If `rvs` is a sequence..."
        with pytest.raises(ValueError, match=message):
            power(test, (rng.normal,), n_observations)
        with pytest.raises(ValueError, match=message):
            power(test, rvs, (10,))

        # 验证当 significance 参数为非法值时，是否引发 ValueError 异常
        message = "`significance` must contain floats between 0 and 1."
        with pytest.raises(ValueError, match=message):
            power(test, rvs, n_observations, significance=2)
        with pytest.raises(ValueError, match=message):
            power(test, rvs, n_observations, significance=np.linspace(-1, 1))

        # 验证当 kwargs 参数不是字典时，是否引发 TypeError 异常
        message = "`kwargs` must be a dictionary"
        with pytest.raises(TypeError, match=message):
            power(test, rvs, n_observations, kwargs=(1, 2, 3))

        # 验证当 rvs 的维度不匹配时，是否引发 ValueError 异常
        message = "shape mismatch: objects cannot be broadcast"
        with pytest.raises(ValueError, match=message):
            power(test, rvs, ([10, 11], [12, 13, 14]))
        with pytest.raises(ValueError, match=message):
            power(test, rvs, ([10, 11], [12, 13]), kwargs={'x': [1, 2, 3]})

        # 验证当 test 参数不可调用时，是否引发 TypeError 异常
        message = "`test` must be callable"
        with pytest.raises(TypeError, match=message):
            power(None, rvs, n_observations)

        # 验证当 n_resamples 参数为非正整数时，是否引发 ValueError 异常
        message = "`n_resamples` must be a positive integer"
        with pytest.raises(ValueError, match=message):
            power(test, rvs, n_observations, n_resamples=-10)
        with pytest.raises(ValueError, match=message):
            power(test, rvs, n_observations, n_resamples=10.5)

        # 验证当 batch 参数为非正整数时，是否引发 ValueError 异常
        message = "`batch` must be a positive integer"
        with pytest.raises(ValueError, match=message):
            power(test, rvs, n_observations, batch=-10)
        with pytest.raises(ValueError, match=message):
            power(test, rvs, n_observations, batch=10.5)

    @pytest.mark.slow
    def test_batch(self):
        # 确保 `batch` 参数被尊重，通过检查在调用 `test` 时提供的最大批量大小
        rng = np.random.default_rng(23492340193)

        def test(x, axis):
            # 如果 x 的维度为 1，则批量大小为 1，否则为 x 的长度
            batch_size = 1 if x.ndim == 1 else len(x)
            # 更新 `test` 函数的静态属性 `batch_size`，取当前批量大小和已记录批量大小的最大值
            test.batch_size = max(batch_size, test.batch_size)
            # 计数器加一
            test.counter += 1
            # 执行单样本 t 检验，返回 p 值
            return stats.ttest_1samp(x, 0, axis=axis).pvalue
        # 初始化 `test` 函数的计数器和批量大小属性
        test.counter = 0
        test.batch_size = 0

        kwds = dict(test=test, n_observations=10, n_resamples=1000)

        # 使用相同的随机种子创建新的随机数生成器对象
        rng = np.random.default_rng(23492340193)
        # 调用 `power` 函数，设置 `batch` 参数为 1
        res1 = power(**kwds, rvs=rng.normal, batch=1)
        # 断言 `test` 函数的计数器为 1000
        assert_equal(test.counter, 1000)
        # 断言 `test` 函数的批量大小为 1
        assert_equal(test.batch_size, 1)

        # 使用相同的随机种子创建新的随机数生成器对象
        rng = np.random.default_rng(23492340193)
        # 重置 `test` 函数的计数器
        test.counter = 0
        # 调用 `power` 函数，设置 `batch` 参数为 50
        res2 = power(**kwds, rvs=rng.normal, batch=50)
        # 断言 `test` 函数的计数器为 20
        assert_equal(test.counter, 20)
        # 断言 `test` 函数的批量大小为 50
        assert_equal(test.batch_size, 50)

        # 使用相同的随机种子创建新的随机数生成器对象
        rng = np.random.default_rng(23492340193)
        # 重置 `test` 函数的计数器
        test.counter = 0
        # 调用 `power` 函数，设置 `batch` 参数为 1000
        res3 = power(**kwds, rvs=rng.normal, batch=1000)
        # 断言 `test` 函数的计数器为 1
        assert_equal(test.counter, 1)
        # 断言 `test` 函数的批量大小为 1000
        assert_equal(test.batch_size, 1000)

        # 断言 `res1` 的功效等于 `res3` 的功效
        assert_equal(res1.power, res3.power)
        # 断言 `res2` 的功效等于 `res3` 的功效
        assert_equal(res2.power, res3.power)
    def test_vectorization(self):
        # Test that `power` is vectorized as expected
        # 使用指定种子创建随机数生成器对象
        rng = np.random.default_rng(25495254834552)

        # Single vectorized call
        # 定义一个测试函数，用于执行 t 检验
        popmeans = np.array([0, 0.2])
        def test(x, alternative, axis=-1):
            # 确保 popmeans 轴位于第一维，并且与其他维度正交
            popmeans_expanded = np.expand_dims(popmeans, tuple(range(1, x.ndim + 1)))
            return stats.ttest_1samp(x, popmeans_expanded, alternative=alternative,
                                     axis=axis)

        # nx and kwargs broadcast against one another
        # 创建一个二维数组，nx 在第二维上进行广播
        nx = np.asarray([10, 15, 20, 50, 100])[:, np.newaxis]
        kwargs = {'alternative': ['less', 'greater', 'two-sided']}

        # This dimension is added to the beginning
        # 创建一个一维数组，用于表示显著性水平
        significance = np.asarray([0.01, 0.025, 0.05, 0.1])
        # 使用 stats.power 计算测试结果
        res = stats.power(test, rng.normal, nx, significance=significance,
                          kwargs=kwargs)

        # Looping over all combinations
        # 初始化一个空列表 ref，用于存储每次循环的结果
        ref = []
        for significance_i in significance:
            for nx_i in nx:
                for alternative_i in kwargs['alternative']:
                    for popmean_i in popmeans:
                        # 定义一个新的测试函数 test2，执行 t 检验
                        def test2(x, axis=-1):
                            return stats.ttest_1samp(x, popmean_i, axis=axis,
                                                     alternative=alternative_i)

                        # 使用 stats.power 计算测试结果，并将 power 存入 ref 列表
                        tmp = stats.power(test2, rng.normal, nx_i,
                                          significance=significance_i)
                        ref.append(tmp.power)
        # 将 ref 转换为与 res.power 相同形状的数组
        ref = np.reshape(ref, res.power.shape)

        # Show that results are similar
        # 断言计算结果 res.power 与参考结果 ref 相似
        assert_allclose(res.power, ref, rtol=2e-2, atol=1e-2)

    def test_ttest_ind_null(self):
        # Check that the p-values of `ttest_ind` are uniformly distributed under
        # the null hypothesis
        # 使用指定种子创建随机数生成器对象
        rng = np.random.default_rng(254952548345528)

        # 使用 ttest_ind 进行检验
        test = stats.ttest_ind
        # 随机生成两组样本的观测数目
        n_observations = rng.integers(10, 100, size=(2, 10))
        # 使用正态分布随机数生成数据集 rvs
        rvs = rng.normal, rng.normal
        # 创建一个一维数组，用于表示显著性水平
        significance = np.asarray([0.01, 0.05, 0.1])
        # 使用 stats.power 计算测试结果
        res = stats.power(test, rvs, n_observations, significance=significance)
        # 将显著性水平进行广播，使其与 res.power 的形状相同
        significance = np.broadcast_to(significance[:, np.newaxis], res.power.shape)
        # 断言计算结果 res.power 与显著性水平相似
        assert_allclose(res.power, significance, atol=1e-2)
    # 定义一个测试方法，用于测试 ttest_1samp 的功效计算是否符合参考值
    def test_ttest_1samp_power(self):
        # 使用种子为 254952548345528 的随机数生成器创建一个随机数对象
        rng = np.random.default_rng(254952548345528)

        # 参考值是使用 statmodels 计算得到的 ttest_1samp 功效值
        ref = [[[0.0126515 , 0.10269751, 0.40415802],
                [0.01657775, 0.29734608, 0.86228288]],
               [[0.0592903 , 0.29317561, 0.71718121],
                [0.07094116, 0.56450441, 0.96815163]]]

        # 定义关键字参数字典，设置总体均值 popmean 为 [0.1, 0.5, 0.9]
        kwargs = {'popmean': [0.1, 0.5, 0.9]}
        
        # 定义观测数列表为 [[10], [20]]
        n_observations = [[10], [20]]
        
        # 显著性水平设置为 [0.01, 0.05]
        significance = [0.01, 0.05]
        
        # 使用 stats.power 函数计算 ttest_1samp 的功效值，传入随机数生成函数 rng.normal，
        # 观测数 n_observations，显著性 significance 和关键字参数 kwargs
        res = stats.power(stats.ttest_1samp, rng.normal, n_observations,
                          significance=significance, kwargs=kwargs)
        
        # 断言计算得到的功效值 res.power 与参考值 ref 在容差范围 1e-2 内相等
        assert_allclose(res.power, ref, atol=1e-2)
class TestPermutationTest:
    
    rtol = 1e-14  # 设置相对容差值，用于数值比较时的精度
    
    def setup_method(self):
        self.rng = np.random.default_rng(7170559330470561044)
        # 初始化随机数生成器对象，用于生成随机数序列

    # -- Input validation -- #

    def test_permutation_test_iv(self):

        def stat(x, y, axis):
            return stats.ttest_ind((x, y), axis).statistic
            # 定义统计函数 `stat`，使用 t 检验计算统计量

        message = "each sample in `data` must contain two or more ..."
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1]), stat)
            # 测试当输入数据不符合要求时，是否正确抛出 ValueError 异常

        message = "`data` must be a tuple containing at least two samples"
        with pytest.raises(ValueError, match=message):
            permutation_test((1,), stat)
            # 测试当输入数据不符合要求时，是否正确抛出 ValueError 异常
        with pytest.raises(TypeError, match=message):
            permutation_test(1, stat)
            # 测试当输入数据类型不符合要求时，是否正确抛出 TypeError 异常

        message = "`axis` must be an integer."
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat, axis=1.5)
            # 测试当输入参数 `axis` 类型不符合要求时，是否正确抛出 ValueError 异常

        message = "`permutation_type` must be in..."
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat,
                             permutation_type="ekki")
            # 测试当输入参数 `permutation_type` 不符合要求时，是否正确抛出 ValueError 异常

        message = "`vectorized` must be `True`, `False`, or `None`."
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat, vectorized=1.5)
            # 测试当输入参数 `vectorized` 类型不符合要求时，是否正确抛出 ValueError 异常

        message = "`n_resamples` must be a positive integer."
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat, n_resamples=-1000)
            # 测试当输入参数 `n_resamples` 不符合要求时，是否正确抛出 ValueError 异常

        message = "`n_resamples` must be a positive integer."
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat, n_resamples=1000.5)
            # 测试当输入参数 `n_resamples` 类型不符合要求时，是否正确抛出 ValueError 异常

        message = "`batch` must be a positive integer or None."
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat, batch=-1000)
            # 测试当输入参数 `batch` 类型不符合要求时，是否正确抛出 ValueError 异常

        message = "`batch` must be a positive integer or None."
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat, batch=1000.5)
            # 测试当输入参数 `batch` 类型不符合要求时，是否正确抛出 ValueError 异常

        message = "`alternative` must be in..."
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat, alternative='ekki')
            # 测试当输入参数 `alternative` 不符合要求时，是否正确抛出 ValueError 异常

        message = "'herring' cannot be used to seed a"
        with pytest.raises(ValueError, match=message):
            permutation_test(([1, 2, 3], [1, 2, 3]), stat,
                             random_state='herring')
            # 测试当输入参数 `random_state` 类型不符合要求时，是否正确抛出 ValueError 异常

    # -- Test Parameters -- #
    @pytest.mark.parametrize('random_state', [np.random.RandomState,
                                              np.random.default_rng])
    @pytest.mark.parametrize('permutation_type',
                             ['pairings', 'samples', 'independent'])
    def test_batch(self, permutation_type, random_state):
        # 确保 `batch` 参数被正确处理，通过检查在调用 `statistic` 时提供的最大批处理大小
        x = self.rng.random(10)  # 生成长度为 10 的随机数组 x
        y = self.rng.random(10)  # 生成长度为 10 的随机数组 y

        def statistic(x, y, axis):
            # 根据 x 的维度确定批处理大小，如果 x 是一维的则为 1，否则为 x 的长度
            batch_size = 1 if x.ndim == 1 else len(x)
            # 更新 statistic 函数的静态属性 batch_size，取当前批处理大小与已有大小的最大值
            statistic.batch_size = max(batch_size, statistic.batch_size)
            # 增加统计计数器
            statistic.counter += 1
            # 返回 x 和 y 沿指定轴的均值之差
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)
        # 初始化统计计数器和批处理大小
        statistic.counter = 0
        statistic.batch_size = 0

        kwds = {'n_resamples': 1000, 'permutation_type': permutation_type,
                'vectorized': True}
        # 进行三次置换检验，每次使用不同的批处理大小
        res1 = stats.permutation_test((x, y), statistic, batch=1,
                                      random_state=random_state(0), **kwds)
        assert_equal(statistic.counter, 1001)  # 断言统计计数器的值为 1001
        assert_equal(statistic.batch_size, 1)  # 断言批处理大小的值为 1

        statistic.counter = 0
        # 使用不同的批处理大小再次进行置换检验
        res2 = stats.permutation_test((x, y), statistic, batch=50,
                                      random_state=random_state(0), **kwds)
        assert_equal(statistic.counter, 21)  # 断言统计计数器的值为 21
        assert_equal(statistic.batch_size, 50)  # 断言批处理大小的值为 50

        statistic.counter = 0
        # 再次使用不同的批处理大小进行置换检验
        res3 = stats.permutation_test((x, y), statistic, batch=1000,
                                      random_state=random_state(0), **kwds)
        assert_equal(statistic.counter, 2)  # 断言统计计数器的值为 2
        assert_equal(statistic.batch_size, 1000)  # 断言批处理大小的值为 1000

        # 断言不同批处理大小的置换检验得到的 p 值相等
        assert_equal(res1.pvalue, res3.pvalue)
        assert_equal(res2.pvalue, res3.pvalue)

    @pytest.mark.parametrize('random_state', [np.random.RandomState,
                                              np.random.default_rng])
    @pytest.mark.parametrize('permutation_type, exact_size',
                             [('pairings', special.factorial(3)**2),
                              ('samples', 2**3),
                              ('independent', special.binom(6, 3))])
    def test_permutations(self, permutation_type, exact_size, random_state):
        # 确保 `permutations` 参数被正确处理，通过检查空分布的大小
        x = self.rng.random(3)  # 生成长度为 3 的随机数组 x
        y = self.rng.random(3)  # 生成长度为 3 的随机数组 y

        def statistic(x, y, axis):
            # 返回 x 和 y 沿指定轴的均值之差
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)

        kwds = {'permutation_type': permutation_type,
                'vectorized': True}
        # 进行置换检验，指定 n_resamples=3，即置换次数为 3
        res = stats.permutation_test((x, y), statistic, n_resamples=3,
                                     random_state=random_state(0), **kwds)
        assert_equal(res.null_distribution.size, 3)  # 断言空分布的大小为 3

        # 再次进行置换检验，使用特定的 `permutation_type` 情况下的确切大小
        res = stats.permutation_test((x, y), statistic, **kwds)
        assert_equal(res.null_distribution.size, exact_size)  # 断言空分布的大小为特定的确切大小

    # -- Randomized Permutation Tests -- #

    # 为了获得合理的准确性，下面的三个测试可能会有些慢。
    # 最初，我让它们适用于所有的置换类型组合，
    # 定义一个测试方法，用于测试随机化测试与精确测试在参数'both'情况下的一致性
    def test_randomized_test_against_exact_both(self):
        # 设置变量alternative为'less'，rng为0，用于控制随机数生成器
        alternative, rng = 'less', 0

        # 设定样本数nx和ny，以及排列组合数permutations的值
        nx, ny, permutations = 8, 9, 24000
        # 断言条件，确保二项式系数大于排列组合数permutations
        assert special.binom(nx + ny, nx) > permutations

        # 生成服从正态分布的随机数序列x和y
        x = stats.norm.rvs(size=nx)
        y = stats.norm.rvs(size=ny)
        # 将随机数序列x和y组成数据元组data
        data = x, y

        # 定义统计函数statistic，用于计算均值之差
        def statistic(x, y, axis):
            return np.mean(x, axis=axis) - np.mean(y, axis=axis)

        # 定义关键字参数kwds，包括是否向量化、排列类型、批处理大小、备择假设类型、随机数种子等
        kwds = {'vectorized': True, 'permutation_type': 'independent',
                'batch': 100, 'alternative': alternative, 'random_state': rng}
        # 进行排列检验，返回结果对象res，使用permutations次重采样
        res = permutation_test(data, statistic, n_resamples=permutations,
                               **kwds)
        # 对比使用无限次重采样的结果res2
        res2 = permutation_test(data, statistic, n_resamples=np.inf, **kwds)

        # 断言两次排列检验的统计量相等
        assert res.statistic == res2.statistic
        # 断言两次排列检验的p值在给定的容差范围内相等
        assert_allclose(res.pvalue, res2.pvalue, atol=1e-2)

    # 使用pytest标记为slow，定义测试方法，用于测试随机化测试与精确测试在参数'samples'情况下的一致性
    @pytest.mark.slow()
    def test_randomized_test_against_exact_samples(self):
        # 设置变量alternative为'greater'，rng为None，用于控制随机数生成器
        alternative, rng = 'greater', None

        # 设定样本数nx和ny，以及排列组合数permutations的值
        nx, ny, permutations = 15, 15, 32000
        # 断言条件，确保2的nx次方大于排列组合数permutations
        assert 2**nx > permutations

        # 生成服从正态分布的随机数序列x和y
        x = stats.norm.rvs(size=nx)
        y = stats.norm.rvs(size=ny)
        # 将随机数序列x和y组成数据元组data
        data = x, y

        # 定义统计函数statistic，用于计算均值差异
        def statistic(x, y, axis):
            return np.mean(x - y, axis=axis)

        # 定义关键字参数kwds，包括是否向量化、排列类型、批处理大小、备择假设类型、随机数种子等
        kwds = {'vectorized': True, 'permutation_type': 'samples',
                'batch': 100, 'alternative': alternative, 'random_state': rng}
        # 进行排列检验，返回结果对象res，使用permutations次重采样
        res = permutation_test(data, statistic, n_resamples=permutations,
                               **kwds)
        # 对比使用无限次重采样的结果res2
        res2 = permutation_test(data, statistic, n_resamples=np.inf, **kwds)

        # 断言两次排列检验的统计量相等
        assert res.statistic == res2.statistic
        # 断言两次排列检验的p值在给定的容差范围内相等
        assert_allclose(res.pvalue, res2.pvalue, atol=1e-2)
    def test_randomized_test_against_exact_pairings(self):
        # 检查在 permutation_type='pairings' 下，随机化测试和精确测试的结果是否合理一致

        alternative, rng = 'two-sided', self.rng

        nx, ny, permutations = 8, 8, 40000
        # 确保 nx 的阶乘大于 permutations
        assert special.factorial(nx) > permutations

        x = stats.norm.rvs(size=nx)
        y = stats.norm.rvs(size=ny)
        data = [x]

        def statistic1d(x):
            # 计算 Pearson 相关系数的第一个值
            return stats.pearsonr(x, y)[0]

        statistic = _resampling._vectorize_statistic(statistic1d)

        kwds = {'vectorized': True, 'permutation_type': 'samples',
                'batch': 100, 'alternative': alternative, 'random_state': rng}
        # 进行置换检验
        res = permutation_test(data, statistic, n_resamples=permutations,
                               **kwds)
        # 使用无限置换次数再次进行置换检验
        res2 = permutation_test(data, statistic, n_resamples=np.inf, **kwds)

        # 断言两次置换检验的统计量相等
        assert res.statistic == res2.statistic
        # 断言两次置换检验的 p 值在给定的公差范围内相等
        assert_allclose(res.pvalue, res2.pvalue, atol=1e-2)

    @pytest.mark.parametrize('alternative', ('less', 'greater'))
    # 这里的双侧 p 值与 ttest_ind 的两侧 p 值采用了不同的约定。
    # 最终，我们可以在 permutation_test 中为双侧替代假设添加多个选项。
    @pytest.mark.parametrize('permutations', (30, 1e9))
    @pytest.mark.parametrize('axis', (0, 1, 2))
    def test_against_permutation_ttest(self, alternative, permutations, axis):
        # 检查这个函数与使用置换的 ttest_ind 是否给出基本相同的结果

        x = np.arange(3*4*5).reshape(3, 4, 5)
        y = np.moveaxis(np.arange(4)[:, None, None], 0, axis)

        rng1 = np.random.default_rng(4337234444626115331)
        # 使用置换进行独立样本 t 检验
        res1 = stats.ttest_ind(x, y, permutations=permutations, axis=axis,
                               random_state=rng1, alternative=alternative)

        def statistic(x, y, axis):
            # 计算 ttest_ind 的统计量
            return stats.ttest_ind(x, y, axis=axis).statistic

        rng2 = np.random.default_rng(4337234444626115331)
        # 使用置换检验进行统计
        res2 = permutation_test((x, y), statistic, vectorized=True,
                                n_resamples=permutations,
                                alternative=alternative, axis=axis,
                                random_state=rng2)

        # 断言两次检验的统计量相等
        assert_allclose(res1.statistic, res2.statistic, rtol=self.rtol)
        # 断言两次检验的 p 值在给定的相对公差范围内相等
        assert_allclose(res1.pvalue, res2.pvalue, rtol=self.rtol)

    # -- 独立（非配对）样本检验 -- #

    @pytest.mark.parametrize('alternative', ("less", "greater", "two-sided"))
    def test_against_ks_2samp(self, alternative):
        # 从随机数生成器中生成正态分布样本 x 和 y
        x = self.rng.normal(size=4, scale=1)
        y = self.rng.normal(size=5, loc=3, scale=3)

        # 使用 scipy.stats 中的 ks_2samp 函数计算预期的统计量和 p 值
        expected = stats.ks_2samp(x, y, alternative=alternative, mode='exact')

        # 定义一个一维统计函数 statistic1d，用于计算 ks_2samp 的统计量
        def statistic1d(x, y):
            return stats.ks_2samp(x, y, mode='asymp',
                                  alternative=alternative).statistic

        # 使用 permutation_test 函数进行置换检验，比较统计量和 p 值
        res = permutation_test((x, y), statistic1d, n_resamples=np.inf,
                               alternative='greater', random_state=self.rng)

        # 断言检验结果的统计量与预期相符
        assert_allclose(res.statistic, expected.statistic, rtol=self.rtol)
        # 断言检验结果的 p 值与预期相符
        assert_allclose(res.pvalue, expected.pvalue, rtol=self.rtol)

    @pytest.mark.parametrize('alternative', ("less", "greater", "two-sided"))
    def test_against_ansari(self, alternative):
        # 从随机数生成器中生成正态分布样本 x 和 y
        x = self.rng.normal(size=4, scale=1)
        y = self.rng.normal(size=5, scale=3)

        # ansari 函数的 alternative 参数与 scipy 中的对应关系
        alternative_correspondence = {"less": "greater",
                                      "greater": "less",
                                      "two-sided": "two-sided"}
        alternative_scipy = alternative_correspondence[alternative]
        # 使用 scipy.stats 中的 ansari 函数计算预期的统计量和 p 值
        expected = stats.ansari(x, y, alternative=alternative_scipy)

        # 定义一个一维统计函数 statistic1d，用于计算 ansari 的统计量
        def statistic1d(x, y):
            return stats.ansari(x, y).statistic

        # 使用 permutation_test 函数进行置换检验，比较统计量和 p 值
        res = permutation_test((x, y), statistic1d, n_resamples=np.inf,
                               alternative=alternative, random_state=self.rng)

        # 断言检验结果的统计量与预期相符
        assert_allclose(res.statistic, expected.statistic, rtol=self.rtol)
        # 断言检验结果的 p 值与预期相符
        assert_allclose(res.pvalue, expected.pvalue, rtol=self.rtol)

    @pytest.mark.parametrize('alternative', ("less", "greater", "two-sided"))
    def test_against_mannwhitneyu(self, alternative):
        # 从 scipy.stats 中生成均匀分布样本 x 和 y
        x = stats.uniform.rvs(size=(3, 5, 2), loc=0, random_state=self.rng)
        y = stats.uniform.rvs(size=(3, 5, 2), loc=0.05, random_state=self.rng)

        # 使用 scipy.stats 中的 mannwhitneyu 函数计算预期的统计量和 p 值
        expected = stats.mannwhitneyu(x, y, axis=1, alternative=alternative)

        # 定义一个统计函数 statistic，用于计算 mannwhitneyu 的统计量
        def statistic(x, y, axis):
            return stats.mannwhitneyu(x, y, axis=axis).statistic

        # 使用 permutation_test 函数进行置换检验，比较统计量和 p 值
        res = permutation_test((x, y), statistic, vectorized=True,
                               n_resamples=np.inf, alternative=alternative,
                               axis=1, random_state=self.rng)

        # 断言检验结果的统计量与预期相符
        assert_allclose(res.statistic, expected.statistic, rtol=self.rtol)
        # 断言检验结果的 p 值与预期相符
        assert_allclose(res.pvalue, expected.pvalue, rtol=self.rtol)
    # 定义一个测试方法，用于测试 stats.cramervonmises_2samp 函数的结果是否符合预期
    def test_against_cvm(self):
        
        # 生成两个服从正态分布的随机样本 x 和 y
        x = stats.norm.rvs(size=4, scale=1, random_state=self.rng)
        y = stats.norm.rvs(size=5, loc=3, scale=3, random_state=self.rng)
        
        # 使用 exact 方法计算两个样本 x 和 y 的 Cramér-von Mises 检验
        expected = stats.cramervonmises_2samp(x, y, method='exact')
        
        # 定义一个内部函数 statistic1d，用于计算样本 x 和 y 的 asymptotic 方法的 Cramér-von Mises 统计量
        def statistic1d(x, y):
            return stats.cramervonmises_2samp(x, y,
                                              method='asymptotic').statistic
        
        # 使用 permutation_test 函数进行置换检验，检验样本 x 和 y 是否满足大于的置换检验条件
        res = permutation_test((x, y), statistic1d, n_resamples=np.inf,
                               alternative='greater', random_state=self.rng)
        
        # 断言检验结果的统计量与预期结果的统计量在相对误差范围内一致
        assert_allclose(res.statistic, expected.statistic, rtol=self.rtol)
        
        # 断言检验结果的 p 值与预期结果的 p 值在相对误差范围内一致
        assert_allclose(res.pvalue, expected.pvalue, rtol=self.rtol)

    @pytest.mark.xslow()
    @pytest.mark.parametrize('axis', (-1, 2))
    # 定义一个测试方法，测试在 'independent' 置换类型下的向量化检验函数的正确性
    def test_vectorized_nsamp_ptype_both(self, axis):
        # 测试 permutation_test 函数在 'independent' 置换类型下，对具有不同形状和维度的 nd 数组样本的正确处理
        # 显示确切置换检验和随机置换检验如何逼近 SciPy 的 asymptotic p 值，并且确切和随机置换检验的结果比它们与 asymptotic 结果的差异更小
        
        # 使用指定的随机数生成器生成三个样本，它们具有不同的（但兼容的）形状和维度
        rng = np.random.default_rng(6709265303529651545)
        x = rng.random(size=(3))
        y = rng.random(size=(1, 3, 2))
        z = rng.random(size=(2, 1, 4))
        data = (x, y, z)
        
        # 定义一个统计量函数 statistic1d，用于计算 Kruskal-Wallis 检验的统计量
        def statistic1d(*data):
            return stats.kruskal(*data).statistic
        
        # 定义一个 p 值计算函数 pvalue1d，用于计算 Kruskal-Wallis 检验的 p 值
        def pvalue1d(*data):
            return stats.kruskal(*data).pvalue
        
        # 对 statistic1d 和 pvalue1d 函数进行向量化处理，以便在多维数据上进行操作
        statistic = _resampling._vectorize_statistic(statistic1d)
        pvalue = _resampling._vectorize_statistic(pvalue1d)
        
        # 手动广播数据以匹配 statistic 函数的预期输入形状
        x2 = np.broadcast_to(x, (2, 3, 3))
        y2 = np.broadcast_to(y, (2, 3, 2))
        z2 = np.broadcast_to(z, (2, 3, 4))
        
        # 计算预期的统计量和 p 值
        expected_statistic = statistic(x2, y2, z2, axis=axis)
        expected_pvalue = pvalue(x2, y2, z2, axis=axis)
        
        # 定义置换检验的关键字参数
        kwds = {'vectorized': False, 'axis': axis, 'alternative': 'greater',
                'permutation_type': 'independent', 'random_state': self.rng}
        
        # 使用 permutation_test 函数进行 exact 和 randomized 置换检验
        res = permutation_test(data, statistic1d, n_resamples=np.inf, **kwds)
        res2 = permutation_test(data, statistic1d, n_resamples=1000, **kwds)
        
        # 检查检验结果的统计量是否与预期结果的统计量在相对误差范围内一致
        assert_allclose(res.statistic, expected_statistic, rtol=self.rtol)
        assert_allclose(res.statistic, res2.statistic, rtol=self.rtol)
        
        # 检查检验结果的 p 值是否与预期结果的 p 值在绝对误差范围内一致
        assert_allclose(res.pvalue, expected_pvalue, atol=6e-2)
        assert_allclose(res.pvalue, res2.pvalue, atol=3e-2)
    # -- Paired-Sample Tests -- #

    @pytest.mark.slow
    @pytest.mark.parametrize('alternative', ("less", "greater", "two-sided"))
    # 使用pytest的标记来标识测试为slow，并对alternative参数进行参数化
    def test_against_wilcoxon(self, alternative):

        x = stats.uniform.rvs(size=(3, 6, 2), loc=0, random_state=self.rng)
        y = stats.uniform.rvs(size=(3, 6, 2), loc=0.05, random_state=self.rng)

        # We'll check both 1- and 2-sample versions of the same test;
        # we expect identical results to wilcoxon in all cases.
        # 定义函数，计算一维样本和双样本wilcoxon检验的统计量，确保结果与wilcoxon一致
        def statistic_1samp_1d(z):
            # 'less' ensures we get the same of two statistics every time
            # 使用'less'作为alternative确保每次获得相同的两个统计量
            return stats.wilcoxon(z, alternative='less').statistic

        def statistic_2samp_1d(x, y):
            # 计算双样本wilcoxon检验的统计量
            return stats.wilcoxon(x, y, alternative='less').statistic

        def test_1d(x, y):
            # 执行wilcoxon检验，并根据alternative参数设定假设检验类型
            return stats.wilcoxon(x, y, alternative=alternative)

        # 使用_resampling._vectorize_statistic函数对test_1d进行向量化处理
        test = _resampling._vectorize_statistic(test_1d)

        # 对x和y进行测试，axis=1表示对每行进行检验
        expected = test(x, y, axis=1)
        expected_stat = expected[0]  # 期望的统计量
        expected_p = expected[1]  # 期望的p值

        kwds = {'vectorized': False, 'axis': 1, 'alternative': alternative,
                'permutation_type': 'samples', 'random_state': self.rng,
                'n_resamples': np.inf}
        # 使用permutation_test进行置换检验，分别对单样本和双样本wilcoxon检验进行检验
        res1 = permutation_test((x-y,), statistic_1samp_1d, **kwds)
        res2 = permutation_test((x, y), statistic_2samp_1d, **kwds)

        # 检查'two-sided'时，wilcoxon返回不同的统计量
        assert_allclose(res1.statistic, res2.statistic, rtol=self.rtol)
        if alternative != 'two-sided':
            # 当alternative不是'two-sided'时，检查期望统计量与实际统计量的一致性
            assert_allclose(res2.statistic, expected_stat, rtol=self.rtol)

        # 检查p值的一致性
        assert_allclose(res2.pvalue, expected_p, rtol=self.rtol)
        assert_allclose(res1.pvalue, res2.pvalue, rtol=self.rtol)

    @pytest.mark.parametrize('alternative', ("less", "greater", "two-sided"))
    # 对alternative参数进行参数化
    def test_against_binomtest(self, alternative):

        x = self.rng.integers(0, 2, size=10)
        x[x == 0] = -1
        # 更自然地，测试会在0和1之间翻转元素。
        # 但是，permutation_test会翻转元素的符号，所以我们使用+1/-1而不是1/0。

        def statistic(x, axis=0):
            # 计算统计量，统计大于0的元素个数
            return np.sum(x > 0, axis=axis)

        k, n, p = statistic(x), 10, 0.5
        # 使用binomtest计算期望值
        expected = stats.binomtest(k, n, p, alternative=alternative)

        # 使用permutation_test进行置换检验，对统计量进行向量化处理
        res = stats.permutation_test((x,), statistic, vectorized=True,
                                     permutation_type='samples',
                                     n_resamples=np.inf, random_state=self.rng,
                                     alternative=alternative)
        # 检查置换检验得到的p值与期望值的一致性
        assert_allclose(res.pvalue, expected.pvalue, rtol=self.rtol)

    # -- Exact Association Tests -- #
    # 定义一个测试方法，用于测试 Kendall's Tau 相关性的假设检验
    def test_against_kendalltau(self):

        # 生成一个服从正态分布的随机数组 x
        x = self.rng.normal(size=6)
        # 生成一个与 x 相关的带有噪声的随机数组 y
        y = x + self.rng.normal(size=6)

        # 计算理论上的 Kendall's Tau 相关性指标
        expected = stats.kendalltau(x, y, method='exact')

        # 定义一个函数，用于计算给定样本 x 的 Kendall's Tau 相关性统计量
        def statistic1d(x):
            return stats.kendalltau(x, y, method='asymptotic').statistic

        # 使用置换检验来评估 x 的 Kendall's Tau 相关性，以确认其显著性
        res = permutation_test((x,), statistic1d, permutation_type='pairings',
                               n_resamples=np.inf, random_state=self.rng)

        # 检查置换检验的统计量是否与理论预期的相符
        assert_allclose(res.statistic, expected.statistic, rtol=self.rtol)
        # 检查置换检验的 p 值是否与理论预期的相符
        assert_allclose(res.pvalue, expected.pvalue, rtol=self.rtol)

    # 使用参数化测试装饰器，测试针对 Fisher's Exact Test 的假设检验，支持多个备择假设
    @pytest.mark.parametrize('alternative', ('less', 'greater', 'two-sided'))
    def test_against_fisher_exact(self, alternative):

        # 定义一个统计量函数，计算与二进制随机变量 x 和 y 的依赖关系有关的数量
        def statistic(x):
            return np.sum((x == 1) & (y == 1))

        # 生成随机数生成器 rng，用于创建二进制随机变量 x 和 y
        rng = np.random.default_rng(6235696159000529929)
        # 生成二进制随机变量 x，其值受到随机数生成器和阈值的影响
        x = (rng.random(7) > 0.6).astype(float)
        # 生成二进制随机变量 y，其值受到随机数生成器、阈值和 x 的影响
        y = (rng.random(7) + 0.25*x > 0.6).astype(float)
        # 创建 x 和 y 的列联表
        tab = stats.contingency.crosstab(x, y)[1]

        # 使用置换检验评估 x 和 y 之间的 Fisher's Exact Test，以验证其显著性
        res = permutation_test((x,), statistic, permutation_type='pairings',
                               n_resamples=np.inf, alternative=alternative,
                               random_state=rng)
        # 计算 Fisher's Exact Test 的理论 p 值，支持多个备择假设
        res2 = stats.fisher_exact(tab, alternative=alternative)

        # 检查置换检验得到的 p 值与 Fisher's Exact Test 的 p 值是否一致
        assert_allclose(res.pvalue, res2[1])

    # 使用标记为 'xslow' 的参数化测试装饰器，测试在不同轴上的参数化操作
    @pytest.mark.xslow()
    @pytest.mark.parametrize('axis', (-2, 1))
    def test_vectorized_nsamp_ptype_samples(self, axis):
        # Test that permutation_test with permutation_type='samples' works
        # properly for a 3-sample statistic with nd array samples of different
        # (but compatible) shapes and ndims. Show that exact permutation test
        # reproduces SciPy's exact pvalue and that random permutation test
        # approximates it.

        # 生成随机数据样本 x, y, z，具有不同但兼容的形状和维度
        x = self.rng.random(size=(2, 4, 3))
        y = self.rng.random(size=(1, 4, 3))
        z = self.rng.random(size=(2, 4, 1))
        
        # 对每个样本数据进行秩排名，指定轴方向
        x = stats.rankdata(x, axis=axis)
        y = stats.rankdata(y, axis=axis)
        z = stats.rankdata(z, axis=axis)
        
        # 只保留 y 的第一个元素，用于检查不同维度之间的广播
        y = y[0]
        
        # 将排名后的数据组成元组
        data = (x, y, z)

        # 定义一维统计量函数
        def statistic1d(*data):
            return stats.page_trend_test(data, ranked=True,
                                         method='asymptotic').statistic

        # 定义一维 p 值函数
        def pvalue1d(*data):
            return stats.page_trend_test(data, ranked=True,
                                         method='exact').pvalue

        # 向量化统计量和 p 值函数
        statistic = _resampling._vectorize_statistic(statistic1d)
        pvalue = _resampling._vectorize_statistic(pvalue1d)

        # 计算预期的统计量和 p 值
        expected_statistic = statistic(*np.broadcast_arrays(*data), axis=axis)
        expected_pvalue = pvalue(*np.broadcast_arrays(*data), axis=axis)

        # 使用整数种子进行随机状态，对数据进行置换测试
        kwds = {'vectorized': False, 'axis': axis, 'alternative': 'greater',
                'permutation_type': 'pairings', 'random_state': 0}
        res = permutation_test(data, statistic1d, n_resamples=np.inf, **kwds)
        res2 = permutation_test(data, statistic1d, n_resamples=5000, **kwds)

        # 断言检查计算的统计量和 p 值与预期值的接近程度
        assert_allclose(res.statistic, expected_statistic, rtol=self.rtol)
        assert_allclose(res.statistic, res2.statistic, rtol=self.rtol)
        assert_allclose(res.pvalue, expected_pvalue, rtol=self.rtol)
        assert_allclose(res.pvalue, res2.pvalue, atol=3e-2)

    # -- Test Against External References -- #

    tie_case_1 = {'x': [1, 2, 3, 4], 'y': [1.5, 2, 2.5],
                  'expected_less': 0.2000000000,
                  'expected_2sided': 0.4,  # 2*expected_less
                  'expected_Pr_gte_S_mean': 0.3428571429,  # see note below
                  'expected_statistic': 7.5,
                  'expected_avg': 9.142857, 'expected_std': 1.40698}
    tie_case_2 = {'x': [111, 107, 100, 99, 102, 106, 109, 108],
                  'y': [107, 108, 106, 98, 105, 103, 110, 105, 104],
                  'expected_less': 0.1555738379,
                  'expected_2sided': 0.3111476758,
                  'expected_Pr_gte_S_mean': 0.2969971205,  # see note below
                  'expected_statistic': 32.5,
                  'expected_avg': 38.117647, 'expected_std': 5.172124}

    @pytest.mark.xslow()  # only the second case is slow, really
    @pytest.mark.parametrize('case', (tie_case_1, tie_case_2))
    # 定义测试函数，用于测试带有并列值的情况
    def test_with_ties(self, case):
        """
        Results above from SAS PROC NPAR1WAY, e.g.

        DATA myData;
        INPUT X Y;
        CARDS;
        1 1
        1 2
        1 3
        1 4
        2 1.5
        2 2
        2 2.5
        ods graphics on;
        proc npar1way AB data=myData;
            class X;
            EXACT;
        run;
        ods graphics off;

        Note: SAS provides Pr >= |S-Mean|, which is different from our
        definition of a two-sided p-value.

        """

        # 从 case 参数中提取变量 x 和 y
        x = case['x']
        y = case['y']

        # 从 case 参数中提取预期的统计量、单侧检验 p-value、双侧检验 p-value、
        # 以及预期的 Pr >= |S-Mean|、均值和标准差
        expected_statistic = case['expected_statistic']
        expected_less = case['expected_less']
        expected_2sided = case['expected_2sided']
        expected_Pr_gte_S_mean = case['expected_Pr_gte_S_mean']
        expected_avg = case['expected_avg']
        expected_std = case['expected_std']

        # 定义一个用于计算一维统计量的函数
        def statistic1d(x, y):
            return stats.ansari(x, y).statistic

        # 使用 numpy 的测试工具，抑制特定的警告
        with np.testing.suppress_warnings() as sup:
            sup.filter(UserWarning, "Ties preclude use of exact statistic")
            # 执行置换检验，计算单侧检验的结果
            res = permutation_test((x, y), statistic1d, n_resamples=np.inf,
                                   alternative='less')
            # 执行置换检验，计算双侧检验的结果
            res2 = permutation_test((x, y), statistic1d, n_resamples=np.inf,
                                    alternative='two-sided')

        # 使用 assert_allclose 函数断言单侧检验的统计量与预期统计量的近似相等性
        assert_allclose(res.statistic, expected_statistic, rtol=self.rtol)
        # 使用 assert_allclose 函数断言单侧检验的 p-value 与预期单侧 p-value 的近似相等性
        assert_allclose(res.pvalue, expected_less, atol=1e-10)
        # 使用 assert_allclose 函数断言双侧检验的 p-value 与预期双侧 p-value 的近似相等性
        assert_allclose(res2.pvalue, expected_2sided, atol=1e-10)
        # 使用 assert_allclose 函数断言双侧检验的零分布均值与预期均值的近似相等性
        assert_allclose(res2.null_distribution.mean(), expected_avg, rtol=1e-6)
        # 使用 assert_allclose 函数断言双侧检验的零分布标准差与预期标准差的近似相等性
        assert_allclose(res2.null_distribution.std(), expected_std, rtol=1e-6)

        # 计算 SAS 提供的 Pr >= |S-Mean|，与预期的 Pr >= |S-Mean| 进行比较
        S = res.statistic
        mean = res.null_distribution.mean()
        n = len(res.null_distribution)
        Pr_gte_S_mean = np.sum(np.abs(res.null_distribution-mean)
                               >= np.abs(S-mean))/n
        # 使用 assert_allclose 函数断言 SAS 提供的 Pr >= |S-Mean| 与预期值的近似相等性
        assert_allclose(expected_Pr_gte_S_mean, Pr_gte_S_mean)

    # 使用 pytest 的标记，将当前测试标记为慢速测试
    @pytest.mark.slow
    # 使用 pytest 的参数化功能，为 alternative 和 expected_pvalue 参数化多个测试用例
    @pytest.mark.parametrize('alternative, expected_pvalue',
                             (('less', 0.9708333333333),
                              ('greater', 0.05138888888889),
                              ('two-sided', 0.1027777777778)))
    def test_against_spearmanr_in_R(self, alternative, expected_pvalue):
        """
        Results above from R cor.test, e.g.

        options(digits=16)
        x <- c(1.76405235, 0.40015721, 0.97873798,
               2.2408932, 1.86755799, -0.97727788)
        y <- c(2.71414076, 0.2488, 0.87551913,
               2.6514917, 2.01160156, 0.47699563)
        cor.test(x, y, method = "spearm", alternative = "t")
        """
        # 定义数据 x 和 y
        x = [1.76405235, 0.40015721, 0.97873798,
             2.2408932, 1.86755799, -0.97727788]
        y = [2.71414076, 0.2488, 0.87551913,
             2.6514917, 2.01160156, 0.47699563]
        # 预期的统计量值
        expected_statistic = 0.7714285714285715

        # 定义计算统计量的函数
        def statistic1d(x):
            return stats.spearmanr(x, y).statistic

        # 使用排列测试来计算结果
        res = permutation_test((x,), statistic1d, permutation_type='pairings',
                               n_resamples=np.inf, alternative=alternative)

        # 断言得到的统计量与预期的统计量接近
        assert_allclose(res.statistic, expected_statistic, rtol=self.rtol)
        # 断言得到的 p 值与预期的 p 值接近
        assert_allclose(res.pvalue, expected_pvalue, atol=1e-13)

    @pytest.mark.parametrize("batch", (-1, 0))
    def test_batch_generator_iv(self, batch):
        # 测试批量生成器处理无效输入的情况
        with pytest.raises(ValueError, match="`batch` must be positive."):
            list(_resampling._batch_generator([1, 2, 3], batch))

    batch_generator_cases = [(range(0), 3, []),
                             (range(6), 3, [[0, 1, 2], [3, 4, 5]]),
                             (range(8), 3, [[0, 1, 2], [3, 4, 5], [6, 7]])]

    @pytest.mark.parametrize("iterable, batch, expected",
                             batch_generator_cases)
    def test_batch_generator(self, iterable, batch, expected):
        # 测试批量生成器的不同输入情况
        got = list(_resampling._batch_generator(iterable, batch))
        assert got == expected

    @pytest.mark.fail_slow(2)
    def test_finite_precision_statistic(self):
        # 某些统计量在理论上应该相等时返回数值上不同的值。测试排列测试是否能处理这种情况。
        x = [1, 2, 4, 3]
        y = [2, 4, 6, 8]

        # 定义统计量函数
        def statistic(x, y):
            return stats.pearsonr(x, y)[0]

        # 使用排列测试计算结果
        res = stats.permutation_test((x, y), statistic, vectorized=False,
                                     permutation_type='pairings')
        r, pvalue, null = res.statistic, res.pvalue, res.null_distribution

        # 计算正确的 p 值
        correct_p = 2 * np.sum(null >= r - 1e-14) / len(null)
        # 断言得到的 p 值与预期的 p 值相等
        assert pvalue == correct_p == 1/3
        # 与使用 R 的 corr.test 进行的精确相关性测试进行比较
        # options(digits=16)
        # x = c(1, 2, 4, 3)
        # y = c(2, 4, 6, 8)
        # cor.test(x, y, alternative = "t", method = "spearman")  # 0.333333333
        # cor.test(x, y, alternative = "t", method = "kendall")  # 0.333333333
# 测试函数，验证 _all_partitions_concatenated 函数能正确生成数据分区，并且所有分区都是唯一的
def test_all_partitions_concatenated():
    # 创建一个包含整数的 NumPy 数组，表示每个分区的大小
    n = np.array([3, 2, 4], dtype=int)
    # 计算 n 数组的累积和
    nc = np.cumsum(n)

    # 使用集合存储所有生成的分区，确保分区唯一性
    all_partitions = set()
    # 计数器，记录生成的分区数量
    counter = 0
    # 遍历所有由 _resampling._all_partitions_concatenated(n) 生成的分区
    for partition_concatenated in _resampling._all_partitions_concatenated(n):
        counter += 1
        # 将分区数据 partition_concatenated 拆分成各个子集
        partitioning = np.split(partition_concatenated, nc[:-1])
        # 将每个子集转换为 frozenset 类型，并存入集合 all_partitions
        all_partitions.add(tuple([frozenset(i) for i in partitioning]))

    # 计算预期的分区数量，使用特殊函数库中的二项式系数计算
    expected = np.prod([special.binom(sum(n[i:]), sum(n[i+1:]))
                        for i in range(len(n)-1)])

    # 断言生成的分区数量与预期相符
    assert_equal(counter, expected)
    # 断言集合中存储的分区数量与预期相符
    assert_equal(len(all_partitions), expected)


# 使用参数化测试，验证参数 vectorized 在所有重采样函数中按预期工作。结果不影响，只要不触发断言错误即可。
@pytest.mark.parametrize('fun_name',
                         ['bootstrap', 'permutation_test', 'monte_carlo_test'])
def test_parameter_vectorized(fun_name):
    # 创建一个指定种子的随机数生成器实例
    rng = np.random.default_rng(75245098234592)
    # 生成一个包含随机数的样本数组
    sample = rng.random(size=10)

    # 定义用于 monte_carlo_test 函数的随机变量生成函数
    def rvs(size):
        return stats.norm.rvs(size=size, random_state=rng)

    # 定义包含不同函数及其参数选项的字典
    fun_options = {'bootstrap': {'data': (sample,), 'random_state': rng,
                                 'method': 'percentile'},
                   'permutation_test': {'data': (sample,), 'random_state': rng,
                                        'permutation_type': 'samples'},
                   'monte_carlo_test': {'sample': sample, 'rvs': rvs}}
    # 定义所有函数共用的选项参数
    common_options = {'n_resamples': 100}

    # 获取当前函数名称对应的函数对象
    fun = getattr(stats, fun_name)
    # 获取当前函数的选项参数
    options = fun_options[fun_name]
    # 更新选项参数，添加共同的选项参数
    options.update(common_options)

    # 定义用于统计函数的统计方法
    def statistic(x, axis):
        # 断言输入数组 x 的维度大于 1 或者与 sample 数组相等
        assert x.ndim > 1 or np.array_equal(x, sample)
        return np.mean(x, axis=axis)

    # 调用函数 fun，使用不同的 vectorized 参数进行测试
    fun(statistic=statistic, vectorized=None, **options)
    fun(statistic=statistic, vectorized=True, **options)

    # 定义另一种统计函数
    def statistic(x):
        # 断言输入数组 x 的维度为 1
        assert x.ndim == 1
        return np.mean(x)

    # 调用函数 fun，使用不同的 vectorized 参数进行测试
    fun(statistic=statistic, vectorized=None, **options)
    fun(statistic=statistic, vectorized=False, **options)
```