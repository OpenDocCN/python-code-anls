# `D:\src\scipysrc\scipy\scipy\stats\tests\test_survival.py`

```
# 导入 pytest 库，用于编写和运行测试用例
import pytest
# 导入 numpy 库，并将其命名为 np，用于数值计算和数组操作
import numpy as np
# 从 numpy.testing 模块导入断言函数 assert_equal 和 assert_allclose，用于测试数值计算的正确性
from numpy.testing import assert_equal, assert_allclose
# 导入 scipy 库，用于科学计算和统计分析
from scipy import stats
# 从 scipy.stats 模块导入 _survival，这是 Kaplan-Meier 算法的一个内部实现
from scipy.stats import _survival


def _kaplan_meier_reference(times, censored):
    # 这是 Kaplan-Meier 估计器的一个非常直接的实现，与 stats.ecdf 中的实现几乎完全不同。

    # 首先对原始数据进行排序。注意，在给定时间上的死亡和丢失的顺序很重要：死亡首先发生。
    # 参见 [2] 第461页："这些约定可以通过以下方式进行释义：在一个年龄 t 记录的死亡被视为稍早于 t 发生，
    # 而在一个年龄 t 记录的丢失被视为稍晚于 t 发生。"
    # 我们通过首先按时间，然后按 `censored` （当有死亡时为0，只有丢失时为1）对数据进行排序来实现这一点。
    dtype = [('time', float), ('censored', int)]
    data = np.array([(t, d) for t, d in zip(times, censored)], dtype=dtype)
    data = np.sort(data, order=('time', 'censored'))
    times = data['time']
    died = np.logical_not(data['censored'])

    m = times.size
    n = np.arange(m, 0, -1)  # 在风险中的人数
    sf = np.cumprod((n - died) / n)

    # 找到唯一时间的*最后*出现的索引。`times` 和 `sf` 的相应条目就是我们想要的。
    _, indices = np.unique(times[::-1], return_index=True)
    ref_times = times[-indices - 1]
    ref_sf = sf[-indices - 1]
    return ref_times, ref_sf


class TestSurvival:

    @staticmethod
    def get_random_sample(rng, n_unique):
        # 生成随机样本
        unique_times = rng.random(n_unique)
        # 将重复次数作为 `np.int32` 类型以解决在 32 位 CI 中 `np.repeat` 失败的问题
        repeats = rng.integers(1, 4, n_unique).astype(np.int32)
        times = rng.permuted(np.repeat(unique_times, repeats))
        censored = rng.random(size=times.size) > rng.random()
        # 使用 stats.CensoredData.right_censored 函数创建一个带有右截尾数据的样本
        sample = stats.CensoredData.right_censored(times, censored)
        return sample, times, censored
    # 测试输入验证函数，用于测试 `stats.ecdf` 函数在不同输入情况下的异常处理行为
    def test_input_validation(self):
        # 检查二维数组作为输入时是否抛出值错误，验证消息为 '`sample` must be a one-dimensional sequence.'
        message = '`sample` must be a one-dimensional sequence.'
        with pytest.raises(ValueError, match=message):
            stats.ecdf([[1]])
        # 检查标量作为输入时是否抛出值错误，验证消息同上
        with pytest.raises(ValueError, match=message):
            stats.ecdf(1)

        # 检查包含 NaN 值的输入是否抛出值错误，验证消息为 '`sample` must not contain nan'
        message = '`sample` must not contain nan'
        with pytest.raises(ValueError, match=message):
            stats.ecdf([np.nan])

        # 检查左截尾的数据输入是否抛出未实现错误，验证消息为 'Currently, only uncensored and right-censored data...'
        message = 'Currently, only uncensored and right-censored data...'
        with pytest.raises(NotImplementedError, match=message):
            stats.ecdf(stats.CensoredData.left_censored([1], censored=[True]))

        # 检查无效的方法输入是否抛出值错误，验证消息为 'method` must be one of...'
        message = 'method` must be one of...'
        res = stats.ecdf([1, 2, 3])
        with pytest.raises(ValueError, match=message):
            res.cdf.confidence_interval(method='ekki-ekki')
        with pytest.raises(ValueError, match=message):
            res.sf.confidence_interval(method='shrubbery')

        # 检查非标量置信水平输入是否抛出值错误，验证消息为 'confidence_level` must be a scalar between 0 and 1'
        message = 'confidence_level` must be a scalar between 0 and 1'
        with pytest.raises(ValueError, match=message):
            res.cdf.confidence_interval(-1)
        with pytest.raises(ValueError, match=message):
            res.sf.confidence_interval([0.5, 0.6])

        # 检查在某些观测值上置信区间未定义时是否发出运行时警告，验证消息为 'The confidence interval is undefined at some observations.'
        message = 'The confidence interval is undefined at some observations.'
        with pytest.warns(RuntimeWarning, match=message):
            ci = res.cdf.confidence_interval()

        # 检查置信区间下界未实现置信区间功能时是否抛出未实现错误，验证消息为 'Confidence interval bounds do not implement...'
        message = 'Confidence interval bounds do not implement...'
        with pytest.raises(NotImplementedError, match=message):
            ci.low.confidence_interval()
        with pytest.raises(NotImplementedError, match=message):
            ci.high.confidence_interval()

    # 测试边缘情况函数，用于验证 `stats.ecdf` 在空输入或单个元素输入时的输出
    def test_edge_cases(self):
        # 测试空列表输入时的输出
        res = stats.ecdf([])
        assert_equal(res.cdf.quantiles, [])
        assert_equal(res.cdf.probabilities, [])

        # 测试单个元素列表输入时的输出
        res = stats.ecdf([1])
        assert_equal(res.cdf.quantiles, [1])
        assert_equal(res.cdf.probabilities, [1])

    # 测试非唯一观测值情况函数，验证 `stats.ecdf` 在具有重复元素的输入情况下的输出
    def test_unique(self):
        # 以唯一观测值为例，参考文献 [1] 第 80 页
        sample = [6.23, 5.58, 7.06, 6.42, 5.20]
        res = stats.ecdf(sample)
        ref_x = np.sort(np.unique(sample))
        ref_cdf = np.arange(1, 6) / 5
        ref_sf = 1 - ref_cdf
        assert_equal(res.cdf.quantiles, ref_x)
        assert_equal(res.cdf.probabilities, ref_cdf)
        assert_equal(res.sf.quantiles, ref_x)
        assert_equal(res.sf.probabilities, ref_sf)

    # 测试非唯一观测值情况函数，验证 `stats.ecdf` 在具有非唯一观测值的输入情况下的输出
    def test_nonunique(self):
        # 以非唯一观测值为例，参考文献 [1] 第 82 页
        sample = [0, 2, 1, 2, 3, 4]
        res = stats.ecdf(sample)
        ref_x = np.sort(np.unique(sample))
        ref_cdf = np.array([1/6, 2/6, 4/6, 5/6, 1])
        ref_sf = 1 - ref_cdf
        assert_equal(res.cdf.quantiles, ref_x)
        assert_equal(res.cdf.probabilities, ref_cdf)
        assert_equal(res.sf.quantiles, ref_x)
        assert_equal(res.sf.probabilities, ref_sf)
    # 定义测试方法，用于测试评估方法的正确性
    def test_evaluate_methods(self):
        # 使用固定种子创建随机数生成器对象
        rng = np.random.default_rng(1162729143302572461)
        # 获取随机样本、截尾值和标记值
        sample, _, _ = self.get_random_sample(rng, 15)
        # 计算样本的经验累积分布函数（ECDF）
        res = stats.ecdf(sample)
        # 获取经验累积分布函数的累积分布函数（CDF）的分位数
        x = res.cdf.quantiles
        # 将分位数右移半个步长，生成右移后的点
        xr = x + np.diff(x, append=x[-1]+1)/2  # right shifted points

        # 断言经验累积分布函数对给定分位数的评估结果等于累积概率
        assert_equal(res.cdf.evaluate(x), res.cdf.probabilities)
        # 断言经验累积分布函数对右移后的分位数的评估结果等于累积概率
        assert_equal(res.cdf.evaluate(xr), res.cdf.probabilities)
        # 断言经验累积分布函数对比首个分位数小1的数值的评估结果为0，因为CDF从0开始
        assert_equal(res.cdf.evaluate(x[0]-1), 0)  # CDF starts at 0
        # 断言经验累积分布函数对负无穷和正无穷的评估结果分别为0和1
        assert_equal(res.cdf.evaluate([-np.inf, np.inf]), [0, 1])

        # 断言生存函数对给定分位数的评估结果等于生存概率
        assert_equal(res.sf.evaluate(x), res.sf.probabilities)
        # 断言生存函数对右移后的分位数的评估结果等于生存概率
        assert_equal(res.sf.evaluate(xr), res.sf.probabilities)
        # 断言生存函数对比首个分位数小1的数值的评估结果为1，因为SF从1开始
        assert_equal(res.sf.evaluate(x[0]-1), 1)  # SF starts at 1
        # 断言生存函数对负无穷和正无穷的评估结果分别为1和0
        assert_equal(res.sf.evaluate([-np.inf, np.inf]), [1, 0])

    # 引用 [1]，页面 91
    t1 = [37, 43, 47, 56, 60, 62, 71, 77, 80, 81]  # times
    d1 = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1]  # 1 表示死亡（非截尾）
    r1 = [1, 1, 0.875, 0.75, 0.75, 0.75, 0.75, 0.5, 0.25, 0]  # 参考生存函数

    # https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_survival/BS704_Survival5.html
    t2 = [8, 12, 26, 14, 21, 27, 8, 32, 20, 40]
    d2 = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    r2 = [0.9, 0.788, 0.675, 0.675, 0.54, 0.405, 0.27, 0.27, 0.27]
    t3 = [33, 28, 41, 48, 48, 25, 37, 48, 25, 43]
    d3 = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    r3 = [1, 0.875, 0.75, 0.75, 0.6, 0.6, 0.6]

    # https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_survival/bs704_survival4.html
    t4 = [24, 3, 11, 19, 24, 13, 14, 2, 18, 17,
          24, 21, 12, 1, 10, 23, 6, 5, 9, 17]
    d4 = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1]
    r4 = [0.95, 0.95, 0.897, 0.844, 0.844, 0.844, 0.844, 0.844, 0.844,
          0.844, 0.76, 0.676, 0.676, 0.676, 0.676, 0.507, 0.507]

    # https://www.real-statistics.com/survival-analysis/kaplan-meier-procedure/confidence-interval-for-the-survival-function/
    t5 = [3, 5, 8, 10, 5, 5, 8, 12, 15, 14, 2, 11, 10, 9, 12, 5, 8, 11]
    d5 = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1]
    r5 = [0.944, 0.889, 0.722, 0.542, 0.542, 0.542, 0.361, 0.181, 0.181, 0.181]

    # 使用参数化测试，对多个案例执行右截尾检验
    @pytest.mark.parametrize("case", [(t1, d1, r1), (t2, d2, r2), (t3, d3, r3),
                                      (t4, d4, r4), (t5, d5, r5)])
    def test_right_censored_against_examples(self, case):
        # 测试经验累积分布函数对示例问题的正确性
        times, died, ref = case
        # 创建右截尾数据
        sample = stats.CensoredData.right_censored(times, np.logical_not(died))
        # 计算样本的经验累积分布函数
        res = stats.ecdf(sample)
        # 断言生存函数的生存概率与参考值在指定精度内一致
        assert_allclose(res.sf.probabilities, ref, atol=1e-3)
        # 断言生存函数的分位数与排序后的唯一时间值一致
        assert_equal(res.sf.quantiles, np.sort(np.unique(times)))

        # 测试参考实现与其他实现的一致性
        res = _kaplan_meier_reference(times, np.logical_not(died))
        # 断言第一个元素（时间值的排序）与排序后的唯一时间值一致
        assert_equal(res[0], np.sort(np.unique(times)))
        # 断言生存概率与参考值在指定精度内一致
        assert_allclose(res[1], ref, atol=1e-3)
    # 使用 pytest 的参数化功能，为测试方法指定多个种子进行参数化测试
    @pytest.mark.parametrize('seed', [182746786639392128, 737379171436494115,
                                      576033618403180168, 308115465002673650])
    # 定义测试方法，用于测试右截尾数据与参考实现的一致性
    def test_right_censored_against_reference_implementation(self, seed):
        # 使用 numpy 生成指定种子的随机数生成器
        rng = np.random.default_rng(seed)
        # 生成随机样本及其对应的时间和截尾信息
        n_unique = rng.integers(10, 100)
        sample, times, censored = self.get_random_sample(rng, n_unique)
        # 计算样本的经验累积分布函数（ECDF）
        res = stats.ecdf(sample)
        # 使用参考实现计算 Kaplan-Meier 生存估计
        ref = _kaplan_meier_reference(times, censored)
        # 断言实际结果的存活函数分位数与参考值的一致性
        assert_allclose(res.sf.quantiles, ref[0])
        # 断言实际结果的存活函数概率与参考值的一致性
        assert_allclose(res.sf.probabilities, ref[1])

        # 如果所有观测都未被截尾，Kaplan-Meier 估计应该与无截尾数据的常规估计一致
        # 创建无截尾数据的 CensoredData 对象
        sample = stats.CensoredData(uncensored=times)
        # 强制使用 Kaplan-Meier 方法计算右截尾 ECDF
        res = _survival._ecdf_right_censored(sample)
        # 计算无截尾数据的 ECDF 作为参考
        ref = stats.ecdf(times)
        # 断言实际结果的存活函数分位数与参考值的一致性
        assert_equal(res[0], ref.sf.quantiles)
        # 断言实际结果的累积分布函数概率与参考值的一致性（绝对误差容忍度为 1e-14）
        assert_allclose(res[1], ref.cdf.probabilities, rtol=1e-14)
        # 断言实际结果的存活函数概率与参考值的一致性（绝对误差容忍度为 1e-14）
        assert_allclose(res[2], ref.sf.probabilities, rtol=1e-14)
    def test_right_censored_ci(self):
        # 定义测试方法，用于测试右截尾数据的置信区间计算

        # 获取测试数据中的时间和死亡信息
        times, died = self.t4, self.d4

        # 根据死亡信息创建右截尾数据样本
        sample = stats.CensoredData.right_censored(times, np.logical_not(died))

        # 计算样本的经验累积分布函数（ECDF）
        res = stats.ecdf(sample)

        # 预设的参考允许误差
        ref_allowance = [0.096, 0.096, 0.135, 0.162, 0.162, 0.162, 0.162,
                         0.162, 0.162, 0.162, 0.214, 0.246, 0.246, 0.246,
                         0.246, 0.341, 0.341]

        # 计算 ECDF 的右尾置信区间
        sf_ci = res.sf.confidence_interval()

        # 计算 ECDF 的累积分布函数的置信区间
        cdf_ci = res.cdf.confidence_interval()

        # 计算实际值与置信区间的偏差
        allowance = res.sf.probabilities - sf_ci.low.probabilities

        # 断言实际值与预期值在指定的公差范围内相等
        assert_allclose(allowance, ref_allowance, atol=1e-3)

        # 断言 ECDF 右尾置信区间的下限与修正后的实际值相等
        assert_allclose(sf_ci.low.probabilities,
                        np.clip(res.sf.probabilities - allowance, 0, 1))

        # 断言 ECDF 右尾置信区间的上限与修正后的实际值相等
        assert_allclose(sf_ci.high.probabilities,
                        np.clip(res.sf.probabilities + allowance, 0, 1))

        # 断言 CDF 累积分布函数置信区间的下限与修正后的实际值相等
        assert_allclose(cdf_ci.low.probabilities,
                        np.clip(res.cdf.probabilities - allowance, 0, 1))

        # 断言 CDF 累积分布函数置信区间的上限与修正后的实际值相等
        assert_allclose(cdf_ci.high.probabilities,
                        np.clip(res.cdf.probabilities + allowance, 0, 1))

        # 对数-对数法测试置信区间，与 Mathematica 的结果对比
        ref_low = [0.694743, 0.694743, 0.647529, 0.591142, 0.591142, 0.591142,
                   0.591142, 0.591142, 0.591142, 0.591142, 0.464605, 0.370359,
                   0.370359, 0.370359, 0.370359, 0.160489, 0.160489]

        ref_high = [0.992802, 0.992802, 0.973299, 0.947073, 0.947073, 0.947073,
                    0.947073, 0.947073, 0.947073, 0.947073, 0.906422, 0.856521,
                    0.856521, 0.856521, 0.856521, 0.776724, 0.776724]

        # 计算 ECDF 的右尾对数-对数置信区间
        sf_ci = res.sf.confidence_interval(method='log-log')

        # 断言 ECDF 对数-对数置信区间的下限与预期的参考下限相等
        assert_allclose(sf_ci.low.probabilities, ref_low, atol=1e-6)

        # 断言 ECDF 对数-对数置信区间的上限与预期的参考上限相等
        assert_allclose(sf_ci.high.probabilities, ref_high, atol=1e-6)
    def test_right_censored_ci_example_5(self):
        # 定义单元测试方法：测试“指数格林伍德”置信区间与示例5的比较

        times, died = self.t5, self.d5
        # 从测试对象中获取时间数据和是否死亡的标记

        sample = stats.CensoredData.right_censored(times, np.logical_not(died))
        # 根据时间数据和生存状态创建右截尾数据样本

        res = stats.ecdf(sample)
        # 计算样本的经验累积分布函数

        lower = np.array([0.66639, 0.624174, 0.456179, 0.287822, 0.287822,
                          0.287822, 0.128489, 0.030957, 0.030957, 0.030957])
        upper = np.array([0.991983, 0.970995, 0.87378, 0.739467, 0.739467,
                          0.739467, 0.603133, 0.430365, 0.430365, 0.430365])
        # 定义预期的上下界数组

        sf_ci = res.sf.confidence_interval(method='log-log')
        # 计算存活函数的置信区间，使用对数-对数方法

        cdf_ci = res.cdf.confidence_interval(method='log-log')
        # 计算累积分布函数的置信区间，使用对数-对数方法

        assert_allclose(sf_ci.low.probabilities, lower, atol=1e-5)
        assert_allclose(sf_ci.high.probabilities, upper, atol=1e-5)
        # 断言存活函数的下界和上界概率与预期值接近

        assert_allclose(cdf_ci.low.probabilities, 1-upper, atol=1e-5)
        assert_allclose(cdf_ci.high.probabilities, 1-lower, atol=1e-5)
        # 断言累积分布函数的下界和上界概率与预期值接近

        # 对比 R 的 `survival` 库的 `survfit` 函数，使用90%置信水平

        # low 和 high 是根据 R 的例子计算得出的置信区间边界值
        low = [0.74366748406861172, 0.68582332289196246, 0.50596835651480121,
               0.32913131413336727, 0.32913131413336727, 0.32913131413336727,
               0.15986912028781664, 0.04499539918147757, 0.04499539918147757,
               0.04499539918147757]
        high = [0.9890291867238429, 0.9638835422144144, 0.8560366823086629,
                0.7130167643978450, 0.7130167643978450, 0.7130167643978450,
                0.5678602982997164, 0.3887616766886558, 0.3887616766886558,
                0.3887616766886558]
        # 定义 R 示例中使用的低置信界和高置信界

        sf_ci = res.sf.confidence_interval(method='log-log',
                                           confidence_level=0.9)
        # 计算存活函数的置信区间，使用对数-对数方法和90%置信水平

        assert_allclose(sf_ci.low.probabilities, low)
        assert_allclose(sf_ci.high.probabilities, high)
        # 断言存活函数的下界和上界概率与 R 示例中的值接近

        # 测试使用 "plain" 类型的置信区间

        low = [0.8556383113628162, 0.7670478794850761, 0.5485720663578469,
               0.3441515412527123, 0.3441515412527123, 0.3441515412527123,
               0.1449184105424544, 0., 0., 0.]
        high = [1., 1., 0.8958723780865975, 0.7391817920806210,
                0.7391817920806210, 0.7391817920806210, 0.5773038116797676,
                0.3642270254596720, 0.3642270254596720, 0.3642270254596720]
        # 定义使用 "plain" 类型计算得出的低置信界和高置信界

        sf_ci = res.sf.confidence_interval(confidence_level=0.9)
        # 计算存活函数的置信区间，使用90%置信水平

        assert_allclose(sf_ci.low.probabilities, low)
        assert_allclose(sf_ci.high.probabilities, high)
        # 断言存活函数的下界和上界概率与 "plain" 类型的预期值接近
    # 定义一个测试方法，用于测试右截尾数据与非截尾数据的比较
    def test_right_censored_against_uncensored(self):
        # 创建一个指定种子的随机数生成器
        rng = np.random.default_rng(7463952748044886637)
        # 生成一个大小为1000的整数数组，元素取值范围为10到100
        sample = rng.integers(10, 100, size=1000)
        # 创建一个与sample数组同样大小的全零数组
        censored = np.zeros_like(sample)
        # 将censored数组中sample中最大值对应位置设为True，表示该位置数据为截尾数据
        censored[np.argmax(sample)] = True
        # 计算未截尾数据的经验累积分布函数（ECDF）
        res = stats.ecdf(sample)
        # 计算右截尾数据的经验累积分布函数（ECDF）
        ref = stats.ecdf(stats.CensoredData.right_censored(sample, censored))
        # 断言检查截尾数据与非截尾数据的生存函数（Survival function）的分位数是否相等
        assert_equal(res.sf.quantiles, ref.sf.quantiles)
        # 断言检查截尾数据与非截尾数据的生存函数的观测数是否相等
        assert_equal(res.sf._n, ref.sf._n)
        # 断言检查截尾数据与非截尾数据的生存函数的累积分布函数是否在除最后一个元素外相等
        assert_equal(res.sf._d[:-1], ref.sf._d[:-1])  # difference @ [-1]
        # 断言检查截尾数据与非截尾数据的生存函数的生存函数值是否在除最后一个元素外非常接近
        assert_allclose(res.sf._sf[:-1], ref.sf._sf[:-1], rtol=1e-14)

    # 定义一个测试方法，用于测试绘制IV曲线的功能
    def test_plot_iv(self):
        # 创建一个指定种子的随机数生成器
        rng = np.random.default_rng(1769658657308472721)
        # 生成一个介于10到100之间的随机整数，作为唯一值数量
        n_unique = rng.integers(10, 100)
        # 调用self对象的方法，获取一个随机样本数据
        sample, _, _ = self.get_random_sample(rng, n_unique)
        # 计算样本数据的经验累积分布函数（ECDF）
        res = stats.ecdf(sample)

        try:
            # 尝试导入matplotlib.pyplot库，并忽略未使用警告
            import matplotlib.pyplot as plt  # noqa: F401
            # 使用生存函数对象的plot方法绘制生存函数图
            res.sf.plot()  # no other errors occur
        except (ModuleNotFoundError, ImportError):
            # 捕获未找到模块或导入错误异常，提示用户安装matplotlib以使用plot方法
            # 避免在使用numpy 2.0-dev时调用matplotlib，因为由于ABI不匹配而经常失败
            # 一旦matplotlib发布与numpy 2.0兼容的版本，这个测试将重新正常运行
            if not np.__version__.startswith('2.0.0.dev0'):
                message = r"matplotlib must be installed to use method `plot`."
                # 使用pytest的断言验证异常信息是否匹配指定的消息
                with pytest.raises(ModuleNotFoundError, match=message):
                    res.sf.plot()
    @pytest.mark.parametrize(
        "x, y, statistic, pvalue",
        # 使用 pytest 的参数化装饰器，用于多组输入参数化测试
        # 结果与 R 语言验证一致
        # 参考 https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_survival/BS704_Survival5.html
        # 每个参数组包括两组数据，分别为 x 和 y，统计量和 p 值也提供了多组
        [
            (
                # 第一组参数
                [[8, 12, 26, 14, 21, 27], [8, 32, 20, 40]],
                [[33, 28, 41], [48, 48, 25, 37, 48, 25, 43]],
                6.91598157449,
                [0.008542873404, 0.9957285632979385, 0.004271436702061537]
            ),
            (
                # 第二组参数
                [[19, 6, 5, 4], [20, 19, 17, 14]],
                [[16, 21, 7], [21, 15, 18, 18, 5]],
                0.835004855038,
                [0.3608293039, 0.8195853480676912, 0.1804146519323088]
            ),
            (
                # 第三组参数
                [[6, 13, 21, 30, 37, 38, 49, 50, 63, 79, 86, 98, 202, 219],
                 [31, 47, 80, 82, 82, 149]],
                [[10, 10, 12, 13, 14, 15, 16, 17, 18, 20, 24, 24, 25, 28, 30,
                  33, 35, 37, 40, 40, 46, 48, 76, 81, 82, 91, 112, 181],
                 [34, 40, 70]],
                7.49659416854,
                [0.006181578637, 0.003090789318730882, 0.9969092106812691]
            )
        ]
    )
    def test_log_rank(self, x, y, statistic, pvalue):
        # 创建 CensoredData 对象 x 和 y，用于存放无右侧和有右侧截尾的数据
        x = stats.CensoredData(uncensored=x[0], right=x[1])
        y = stats.CensoredData(uncensored=y[0], right=y[1])

        # 对每种假设（two-sided, less, greater）进行检验
        for i, alternative in enumerate(["two-sided", "less", "greater"]):
            # 进行 logrank 检验，返回结果对象 res
            res = stats.logrank(x=x, y=y, alternative=alternative)

            # 断言检验统计量的平方与预期统计量 statistic 接近
            assert_allclose(res.statistic**2, statistic, atol=1e-10)
            # 断言 p 值与预期 pvalue[i] 接近
            assert_allclose(res.pvalue, pvalue[i], atol=1e-10)
    # 定义一个测试方法，用于测试异常情况
    def test_raises(self):
        # 创建一个 CensoredData 对象，使用样本数据 [1, 2]
        sample = stats.CensoredData([1, 2])

        # 设置错误消息的正则表达式模式
        msg = r"`y` must be"
        # 使用 pytest 的上下文管理器检查是否会抛出 ValueError 异常，并验证异常消息是否符合指定的正则表达式模式
        with pytest.raises(ValueError, match=msg):
            # 调用 logrank 函数，传入参数 x=sample, y=[[1, 2]]，期望抛出 ValueError 异常
            stats.logrank(x=sample, y=[[1, 2]])

        # 设置错误消息的正则表达式模式
        msg = r"`x` must be"
        # 使用 pytest 的上下文管理器检查是否会抛出 ValueError 异常，并验证异常消息是否符合指定的正则表达式模式
        with pytest.raises(ValueError, match=msg):
            # 调用 logrank 函数，传入参数 x=[[1, 2]], y=sample，期望抛出 ValueError 异常
            stats.logrank(x=[[1, 2]], y=sample)
```