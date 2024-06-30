# `D:\src\scipysrc\scipy\scipy\stats\tests\test_axis_nan_policy.py`

```
# 导入必要的模块和库
from itertools import product, combinations_with_replacement, permutations
import os  # 系统操作模块
import re  # 正则表达式模块
import pickle  # 对象序列化和反序列化模块
import pytest  # 测试框架模块
import warnings  # 警告处理模块

import numpy as np  # 数值计算库
from numpy.testing import assert_allclose, assert_equal  # 测试辅助模块
from scipy import stats  # 统计分析库
from scipy.stats import norm  # 正态分布模块（类型提示：忽略属性定义）
from scipy.stats._axis_nan_policy import (_masked_arrays_2_sentinel_arrays,
                                          SmallSampleWarning,
                                          too_small_nd_omit, too_small_nd_not_omit,
                                          too_small_1d_omit, too_small_1d_not_omit)  # 统计分析模块的特定函数
from scipy._lib._util import AxisError  # SciPy库通用工具模块中的轴错误处理
from scipy.conftest import skip_xp_invalid_arg  # SciPy配置测试模块中的无效参数跳过函数

SCIPY_XSLOW = int(os.environ.get('SCIPY_XSLOW', '0'))  # 获取环境变量设置（用于性能调优）

# 解包 ttest 结果的辅助函数
def unpack_ttest_result(res):
    low, high = res.confidence_interval()  # 获取置信区间的上下界
    return (res.statistic, res.pvalue, res.df, res._standard_error,
            res._estimate, low, high)  # 返回统计量、p值、自由度、标准误差、估计值以及置信区间的元组

# 获取 ttest 的置信区间边界的辅助函数
def _get_ttest_ci(ttest):
    def ttest_ci(*args, **kwargs):
        res = ttest(*args, **kwargs)  # 执行 ttest 函数
        return res.confidence_interval()  # 返回置信区间
    return ttest_ci

# 对一样本 t 检验的快速版本函数
def xp_mean_1samp(*args, **kwargs):
    kwargs.pop('_no_deco', None)  # 移除特定关键字参数 '_no_deco'
    return stats._stats_py._xp_mean(*args, **kwargs)  # 调用统计分析库中的一样本均值计算函数

# 对两样本 t 检验的快速版本函数
def xp_mean_2samp(*args, **kwargs):
    kwargs.pop('_no_deco', None)  # 移除特定关键字参数 '_no_deco'
    weights = args[1]  # 获取权重参数
    return stats._stats_py._xp_mean(args[0], *args[2:], weights=weights, **kwargs)  # 调用统计分析库中的两样本均值计算函数

# 方差的快速版本函数
def xp_var(*args, **kwargs):
    kwargs.pop('_no_deco', None)  # 移除特定关键字参数 '_no_deco'
    return stats._stats_py._xp_var(*args, **kwargs)  # 调用统计分析库中的方差计算函数

# 各种轴与 NaN 策略的测试用例
axis_nan_policy_cases = [
    # 函数, 参数, 关键字参数, 样本数, 输出数, 是否配对, 结果解包函数
    # 参数和关键字参数通常不需要；只是展示它们是如何工作的
    (stats.kruskal, tuple(), dict(), 3, 2, False, None),  # 4个样本时速度慢
    (stats.ranksums, ('less',), dict(), 2, 2, False, None),
    (stats.mannwhitneyu, tuple(), {'method': 'asymptotic'}, 2, 2, False, None),
    (stats.wilcoxon, ('pratt',), {'mode': 'auto'}, 2, 2, True,
     lambda res: (res.statistic, res.pvalue)),
    (stats.wilcoxon, tuple(), dict(), 1, 2, True,
     lambda res: (res.statistic, res.pvalue)),
    (stats.wilcoxon, tuple(), {'mode': 'approx'}, 1, 3, True,
     lambda res: (res.statistic, res.pvalue, res.zstatistic)),
    (stats.gmean, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    (stats.hmean, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    (stats.pmean, (1.42,), dict(), 1, 1, False, lambda x: (x,)),
    (stats.sem, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    (stats.iqr, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    # 计算峰度（kurtosis）的统计量，参数包括空元组、空字典、1、1、False，以及一个函数 lambda x: (x,)，返回元组 (x,)
    (stats.kurtosis, tuple(), dict(), 1, 1, False, lambda x: (x,)),

    # 计算偏度（skew）的统计量，参数包括空元组、空字典、1、1、False，以及一个函数 lambda x: (x,)，返回元组 (x,)
    (stats.skew, tuple(), dict(), 1, 1, False, lambda x: (x,)),

    # 计算 k-statistic，参数包括空元组、空字典、1、1、False，以及一个函数 lambda x: (x,)，返回元组 (x,)
    (stats.kstat, tuple(), dict(), 1, 1, False, lambda x: (x,)),

    # 计算 k-statistic variance，参数包括空元组、空字典、1、1、False，以及一个函数 lambda x: (x,)，返回元组 (x,)
    (stats.kstatvar, tuple(), dict(), 1, 1, False, lambda x: (x,)),

    # 计算给定阶数的矩（moment），参数包括空元组、空字典、1、1、False，以及一个函数 lambda x: (x,)，返回元组 (x,)
    (stats.moment, tuple(), dict(), 1, 1, False, lambda x: (x,)),

    # 计算给定多个阶数的矩（moment），参数包括空元组、指定 order=[1, 2] 的字典、1、2、False，没有返回函数
    (stats.moment, tuple(), dict(order=[1, 2]), 1, 2, False, None),

    # 计算 Jarque-Bera 正态性检验，参数包括空元组、空字典、1、2、False，没有返回函数
    (stats.jarque_bera, tuple(), dict(), 1, 2, False, None),

    # 计算一样本 t 检验，参数包括包含一个数组 [0] 的元组、空字典、1、7、False，返回一个解包函数 unpack_ttest_result
    (stats.ttest_1samp, (np.array([0]),), dict(), 1, 7, False, unpack_ttest_result),

    # 计算相关配对样本的 t 检验，参数包括空元组、空字典、2、7、True（有返回 p-value）、解包函数 unpack_ttest_result
    (stats.ttest_rel, tuple(), dict(), 2, 7, True, unpack_ttest_result),

    # 计算独立样本的 t 检验，参数包括空元组、空字典、2、7、False，解包函数 unpack_ttest_result
    (stats.ttest_ind, tuple(), dict(), 2, 7, False, unpack_ttest_result),

    # 获取一样本 t 检验的置信区间，参数包括函数 _get_ttest_ci(stats.ttest_1samp)、元组 (0,)、空字典、1、2、False，没有返回函数
    (_get_ttest_ci(stats.ttest_1samp), (0,), dict(), 1, 2, False, None),

    # 获取相关配对样本 t 检验的置信区间，参数包括函数 _get_ttest_ci(stats.ttest_rel)、空元组、空字典、2、2、True（有返回 p-value），没有返回函数
    (_get_ttest_ci(stats.ttest_rel), tuple(), dict(), 2, 2, True, None),

    # 获取独立样本 t 检验的置信区间，参数包括函数 _get_ttest_ci(stats.ttest_ind)、空元组、空字典、2、2、False，没有返回函数
    (_get_ttest_ci(stats.ttest_ind), tuple(), dict(), 2, 2, False, None),

    # 计算众数（mode），参数包括空元组、空字典、1、2、True（返回众数和频数）、一个函数 lambda x: (x.mode, x.count)
    (stats.mode, tuple(), dict(), 1, 2, True, lambda x: (x.mode, x.count)),

    # 计算差分熵（differential_entropy），参数包括空元组、空字典、1、1、False，以及一个函数 lambda x: (x,)，返回元组 (x,)
    (stats.differential_entropy, tuple(), dict(), 1, 1, False, lambda x: (x,)),

    # 计算变异系数（variation），参数包括空元组、空字典、1、1、False，以及一个函数 lambda x: (x,)，返回元组 (x,)
    (stats.variation, tuple(), dict(), 1, 1, False, lambda x: (x,)),

    # 计算 Friedman 卡方检验，参数包括空元组、空字典、3、2、True（有返回 p-value），没有返回函数
    (stats.friedmanchisquare, tuple(), dict(), 3, 2, True, None),

    # 计算 Brunner-Munzel 检验，参数包括空元组、指定 distribution='normal' 的字典、2、2、False，没有返回函数
    (stats.brunnermunzel, tuple(), dict(distribution='normal'), 2, 2, False, None),

    # 计算 Mood 检验，参数包括空元组、空字典、2、2、False，没有返回函数
    (stats.mood, tuple(), {}, 2, 2, False, None),

    # 计算 Shapiro-Wilk 正态性检验，参数包括空元组、空字典、1、2、False，没有返回函数
    (stats.shapiro, tuple(), {}, 1, 2, False, None),

    # 计算一样本 Kolmogorov-Smirnov 检验，参数包括元组 (norm().cdf,)、空字典、1、4、False，返回元组 (*res, res.statistic_location, res.statistic_sign)
    (stats.ks_1samp, (norm().cdf,), dict(), 1, 4, False, lambda res: (*res, res.statistic_location, res.statistic_sign)),

    # 计算两样本 Kolmogorov-Smirnov 检验，参数包括空元组、空字典、2、4、False，返回元组 (*res, res.statistic_location, res.statistic_sign)
    (stats.ks_2samp, tuple(), dict(), 2, 4, False, lambda res: (*res, res.statistic_location, res.statistic_sign)),

    # 计算 Kolmogorov-Smirnov 正态性检验，参数包括元组 (norm().cdf,)、空字典、1、4、False，返回元组 (*res, res.statistic_location, res.statistic_sign)
    (stats.kstest, (norm().cdf,), dict(), 1, 4, False, lambda res: (*res, res.statistic_location, res.statistic_sign)),

    # 计算 Kolmogorov-Smirnov 两样本检验，参数包括空元组、空字典、2、4、False，返回元组 (*res, res.statistic_location, res.statistic_sign)
    (stats.kstest, tuple(), dict(), 2, 4, False, lambda res: (*res, res.statistic_location, res.statistic_sign)),

    # 计算 Levene 方差齐性检验，参数包括空元组、空字典、2、2、False，没有返回函数
    (stats.levene, tuple(), {}, 2, 2, False, None),

    # 计算 Fligner-Killeen 中位数检验，参数包括空元组、指定 center='trimmed' 和 proportiontocut=0.01 的字典、2、2、False，没有返回函数
    (stats.fligner, tuple(), {'center': 'trimmed', 'proportiontocut': 0.01}, 2, 2, False, None),

    # 计算 Ansari-Bradley 检验，参数包括空元组、空字典、2、2、False，没有返回函数
    (stats.ansari,
    # 使用 scipy.stats 中的函数进行统计分析，每个元组的结构如下：
    # (函数名, 空元组(), 空字典{}, 最小参数个数, 最大参数个数, 是否需要返回 p 值, 处理结果的 lambda 函数)
    
    (stats.tstd, tuple(), {}, 1, 1, False, lambda x: (x,)),
    # 调用 scipy.stats 中的 tstd 函数，计算单样本或多样本标准差，无需返回 p 值，处理结果的 lambda 函数返回输入值本身
    
    (stats.tsem, tuple(), {}, 1, 1, False, lambda x: (x,)),
    # 调用 scipy.stats 中的 tsem 函数，计算 t 分布的标准误，无需返回 p 值，处理结果的 lambda 函数返回输入值本身
    
    (stats.circmean, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    # 调用 scipy.stats 中的 circmean 函数，计算样本的平均角度（适用于循环数据），无需返回 p 值，处理结果的 lambda 函数返回输入值本身
    
    (stats.circvar, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    # 调用 scipy.stats 中的 circvar 函数，计算样本的循环方差，无需返回 p 值，处理结果的 lambda 函数返回输入值本身
    
    (stats.circstd, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    # 调用 scipy.stats 中的 circstd 函数，计算样本的循环标准差，无需返回 p 值，处理结果的 lambda 函数返回输入值本身
    
    (stats.f_oneway, tuple(), {}, 2, 2, False, None),
    # 调用 scipy.stats 中的 f_oneway 函数，执行单因素方差分析，不需要返回 p 值，没有处理结果的 lambda 函数
    
    (stats.alexandergovern, tuple(), {}, 2, 2, False, lambda res: (res.statistic, res.pvalue)),
    # 调用 scipy.stats 中的 alexandergovern 函数，执行 Alexander-Govern 测试，需要返回统计量和 p 值，处理结果的 lambda 函数返回统计量和 p 值
    
    (stats.combine_pvalues, tuple(), {}, 1, 2, False, None),
    # 调用 scipy.stats 中的 combine_pvalues 函数，合并多个 p 值，不需要返回 p 值，没有处理结果的 lambda 函数
    
    (xp_mean_1samp, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    # 调用 xp_mean_1samp 函数，执行单样本均值检验，无需返回 p 值，处理结果的 lambda 函数返回输入值本身
    
    (xp_mean_2samp, tuple(), dict(), 2, 1, True, lambda x: (x,)),
    # 调用 xp_mean_2samp 函数，执行两样本均值检验，需要返回 p 值，处理结果的 lambda 函数返回输入值本身
    
    (xp_var, tuple(), dict(), 1, 1, False, lambda x: (x,)),
    # 调用 xp_var 函数，计算样本方差，无需返回 p 值，处理结果的 lambda 函数返回输入值本身
# 如果消息是预期中的其中之一，将NaN放在`statistics`和`pvalues`的适当位置
too_small_messages = {"Degrees of freedom <= 0 for slice",
                      "x and y should have at least 5 elements",
                      "Data must be at least length 3",
                      "The sample must contain at least two",
                      "x and y must contain at least two",
                      "division by zero",
                      "Mean of empty slice",
                      "Data passed to ks_2samp must not be empty",
                      "Not enough test observations",
                      "Not enough other observations",
                      "Not enough observations.",
                      "At least one observation is required",
                      "zero-size array to reduction operation maximum",
                      "`x` and `y` must be of nonzero size.",
                      "The exact distribution of the Wilcoxon test",
                      "Data input must not be empty",
                      "Window length (0) must be positive and less",
                      "Window length (1) must be positive and less",
                      "Window length (2) must be positive and less",
                      "`skewtest` requires at least",
                      "`kurtosistest` requires at least",
                      "attempt to get argmax of an empty sequence",
                      "No array values within given limits",
                      "Input sample size must be greater than one.",
                      "At least one slice along `axis` has zero length",
                      "One or more sample arguments is too small",
                      "invalid value encountered",
                      "divide by zero encountered",
}

# 如果消息是这些之一，函数的结果可能不准确，但不会放置NaN
inaccuracy_messages = {"Precision loss occurred in moment calculation",
                       "Sample size too small for normal approximation."}

# 对于某些函数，nan_policy='propagate'不应只返回NaN
override_propagate_funcs = {stats.mode}

# 对于某些函数，空数组会产生非NaN结果
empty_special_case_funcs = {stats.entropy}

# 某些函数不遵循通常的“太小”警告规则
too_small_special_case_funcs = {stats.entropy}

def _mixed_data_generator(n_samples, n_repetitions, axis, rng,
                          paired=False):
    # 生成随机样本以检查假设检验对具有不同（但可广播）形状和各种NaN模式（例如所有NaN、一些NaN、没有NaN）的样本的响应，沿轴切片

    data = []
    # 循环处理每个样本
    for i in range(n_samples):
        # 定义 nan 模式的数量
        n_patterns = 6  # number of distinct nan patterns
        # 如果 paired 为 True，则设置观测数为固定值 20；否则根据当前循环次数递增
        n_obs = 20 if paired else 20 + i  # observations per axis-slice
        # 创建一个形状为 (n_repetitions, n_patterns, n_obs) 的数组，全部初始化为 nan
        x = np.ones((n_repetitions, n_patterns, n_obs)) * np.nan

        # 遍历每个重复实验
        for j in range(n_repetitions):
            # 获取当前重复实验下的样本数据
            samples = x[j, :, :]

            # 针对不同的情况设置不同的 nan 模式
            # case 0: 全为 nan 的轴切片 (0 个实数)
            # cases 1-3: 1-3 个实数的轴切片 (其余为 nan)
            # case 4: 大部分为实数的轴切片 (除了两个为 nan)
            # case 5: 全为实数的轴切片
            for k, n_reals in enumerate([0, 1, 2, 3, n_obs-2, n_obs]):
                # 对于 cases 1-3，需要保证 paired 为 True 时 nan 位于相同的位置
                indices = rng.permutation(n_obs)[:n_reals]
                # 在指定位置设置随机生成的实数值
                samples[k, indices] = rng.random(size=n_reals)

            # 随机排列轴切片，以展示顺序无关紧要
            samples[:] = rng.permutation(samples, axis=0)

        # 调整数组形状，用于多样本测试，确保广播和 nan 策略对每个输入的每个 nan 模式都有效
        new_shape = [n_repetitions] + [1]*n_samples + [n_obs]
        new_shape[1 + i] = 6
        x = x.reshape(new_shape)

        # 将轴切片移动到指定位置
        x = np.moveaxis(x, -1, axis)
        # 将处理好的数据添加到结果列表中
        data.append(x)

    # 返回最终的数据列表
    return data
# 定义一个生成具有同质数据的数据集的函数，用于检查假设检验对不同形状数据（但可以广播的）的响应
def _homogeneous_data_generator(n_samples, n_repetitions, axis, rng,
                                paired=False, all_nans=True):
    # 数据集列表，用于存储生成的数据
    data = []
    # 循环生成指定数量的样本数据
    for i in range(n_samples):
        # 如果是配对数据，每个轴切片的观测次数为20
        # 否则，每个轴切片的观测次数为20加上i
        n_obs = 20 if paired else 20 + i  
        # 创建数据的形状，保证广播后的形状，每个轴上的元素数是指定的
        shape = [n_repetitions] + [1]*n_samples + [n_obs]
        # 第二维度上的元素个数设置为2
        shape[1 + i] = 2
        # 生成全为NaN或随机数的数据，取决于all_nans参数
        x = np.ones(shape) * np.nan if all_nans else rng.random(shape)
        # 将数据轴移动到指定的轴位置
        x = np.moveaxis(x, -1, axis)
        # 将生成的数据添加到数据集列表中
        data.append(x)
    # 返回生成的数据集列表
    return data


def nan_policy_1d(hypotest, data1d, unpacker, *args, n_outputs=2,
                  nan_policy='raise', paired=False, _no_deco=True, **kwds):
    # 用于处理一维样本数据的NaN值的策略参考实现

    # 如果策略是'raise'，则检查数据中是否存在NaN值，若存在则抛出ValueError异常
    if nan_policy == 'raise':
        for sample in data1d:
            if np.any(np.isnan(sample)):
                raise ValueError("The input contains nan values")

    # 如果策略是'propagate'且不在覆盖的函数列表中，则返回全为NaN的输出
    elif (nan_policy == 'propagate'
          and hypotest not in override_propagate_funcs):
        # 对于被测试的所有假设检验，返回NaN是正确的行为
        # 但是许多假设检验在处理时不正确地传播NaN值（例如，当涉及到排名时将np.nan和np.inf视为相同）
        # 在此处覆盖这种行为
        for sample in data1d:
            if np.any(np.isnan(sample)):
                return np.full(n_outputs, np.nan)

    # 如果策略是'omit'，则手动省略NaN值（或至少一个元素为NaN的对）
    elif nan_policy == 'omit':
        if not paired:
            # 对于非配对数据，移除每个样本中的NaN值
            data1d = [sample[~np.isnan(sample)] for sample in data1d]
        else:
            # 对于配对数据，创建NaN值掩码并移除至少一个样本中有NaN值的对
            nan_mask = np.isnan(data1d[0])
            for sample in data1d[1:]:
                nan_mask = np.logical_or(nan_mask, np.isnan(sample))
            data1d = [sample[~nan_mask] for sample in data1d]

    # 使用指定的解包器对假设检验函数进行调用，并返回结果
    return unpacker(hypotest(*data1d, *args, _no_deco=_no_deco, **kwds))
# 在这些测试函数中，忽略特定的运行时警告，以及根据参数化测试的配置决定是否跳过或忽略某些测试。

@pytest.mark.filterwarnings('ignore:divide by zero encountered:RuntimeWarning')
# 忽略在运行时可能遇到的除零警告

@pytest.mark.parametrize(("hypotest", "args", "kwds", "n_samples", "n_outputs",
                          "paired", "unpacker"), axis_nan_policy_cases)
# 参数化测试，使用来自 axis_nan_policy_cases 的参数来测试多个假设检验函数和参数

@pytest.mark.parametrize(("nan_policy"), ("propagate", "omit", "raise"))
# 参数化测试，测试不同的 NaN 处理策略：propagate、omit、raise

@pytest.mark.parametrize(("axis"), (1,))
# 参数化测试，测试不同的轴（axis）设置，这里只测试轴为1

@pytest.mark.parametrize(("data_generator"), ("mixed",))
# 参数化测试，测试不同的数据生成器类型，这里使用 mixed（混合类型）

def test_axis_nan_policy_fast(hypotest, args, kwds, n_samples, n_outputs,
                              paired, unpacker, nan_policy, axis,
                              data_generator):
    # 如果假设检验函数是 stats.cramervonmises_2samp 或 stats.kruskal，并且不是在 SCIPY_XSLOW 模式下，跳过测试
    if hypotest in {stats.cramervonmises_2samp, stats.kruskal} and not SCIPY_XSLOW:
        pytest.skip("Too slow.")
    # 执行 _axis_nan_policy_test 函数来进行具体的测试
    _axis_nan_policy_test(hypotest, args, kwds, n_samples, n_outputs, paired,
                          unpacker, nan_policy, axis, data_generator)


if SCIPY_XSLOW:
    # 如果 SCIPY_XSLOW 为真，则以下这些测试函数将被忽略或跳过，因为它们的运行时间较长或计算量较大

    # 下面三个警告是有意为之的
    # 对于 `wilcoxon` 当样本大小 < 50 时的警告
    @pytest.mark.filterwarnings('ignore:Sample size too small for normal:UserWarning')
    # 对于 `kurtosistest` 和 `normaltest` 当样本大小 < 20 时的警告
    @pytest.mark.filterwarnings('ignore:`kurtosistest` p-value may be:UserWarning')
    # 对于 `foneway` 函数的警告
    @pytest.mark.filterwarnings('ignore:all input arrays have length 1.:RuntimeWarning')

    # 其余的警告可能是有意为之的，但需要进一步调查以确定函数的装饰器是否应该定义 `too_small`
    # 对于 `bartlett`, `tvar`, `tstd`, `tsem` 函数的警告
    @pytest.mark.filterwarnings('ignore:Degrees of freedom <= 0 for:RuntimeWarning')
    # 对于多个函数（例如 kstat, kstatvar, ttest_1samp 等）的警告
    @pytest.mark.filterwarnings('ignore:Invalid value encountered in:RuntimeWarning')
    # 对于多个函数（例如 kstatvar, ttest_1samp, ttest_rel 等）的警告
    @pytest.mark.filterwarnings('ignore:divide by zero encountered:RuntimeWarning')

    @pytest.mark.parametrize(("hypotest", "args", "kwds", "n_samples", "n_outputs",
                              "paired", "unpacker"), axis_nan_policy_cases)
    # 参数化测试，使用来自 axis_nan_policy_cases 的参数来测试多个假设检验函数和参数

    @pytest.mark.parametrize(("nan_policy"), ("propagate", "omit", "raise"))
    # 参数化测试，测试不同的 NaN 处理策略：propagate、omit、raise

    @pytest.mark.parametrize(("axis"), range(-3, 3))
    # 参数化测试，测试不同的轴（axis）设置，这里测试从 -3 到 2 的范围

    @pytest.mark.parametrize(("data_generator"),
                             ("all_nans", "all_finite", "mixed"))
    # 参数化测试，测试不同的数据生成器类型：all_nans、all_finite、mixed
    # 定义一个函数test_axis_nan_policy_full，用于执行轴向 NaN 策略的完整性测试
    def test_axis_nan_policy_full(hypotest, args, kwds, n_samples, n_outputs,
                                  paired, unpacker, nan_policy, axis,
                                  data_generator):
        # 调用内部函数 _axis_nan_policy_test，用于执行具体的假设检验
        _axis_nan_policy_test(hypotest, args, kwds, n_samples, n_outputs, paired,
                              unpacker, nan_policy, axis, data_generator)
# 测试假设检验在不同条件下（带有 NaN 值的数据）的一维和向量化行为
def _axis_nan_policy_test(hypotest, args, kwds, n_samples, n_outputs, paired,
                          unpacker, nan_policy, axis, data_generator):
    # 如果未提供解包器（unpacker），则定义一个默认的解包器函数来提取统计量和 p 值
    if not unpacker:
        def unpacker(res):
            return res

    # 使用默认随机数生成器创建 RNG 对象
    rng = np.random.default_rng(0)

    # 生成多维测试数据，以测试在不同的 NaN 值模式下的行为
    n_repetitions = 3  # 每种模式的重复次数
    data_gen_kwds = {'n_samples': n_samples, 'n_repetitions': n_repetitions,
                     'axis': axis, 'rng': rng, 'paired': paired}
    
    # 根据数据生成器类型选择不同的生成数据函数，并设置相应的 inherent_size
    if data_generator == 'mixed':
        inherent_size = 6  # 不同类型模式的数量
        data = _mixed_data_generator(**data_gen_kwds)
    elif data_generator == 'all_nans':
        inherent_size = 2  # 在 _homogeneous_data_generator 中固定为两种模式
        data_gen_kwds['all_nans'] = True
        data = _homogeneous_data_generator(**data_gen_kwds)
    elif data_generator == 'all_finite':
        inherent_size = 2  # 在 _homogeneous_data_generator 中固定为两种模式
        data_gen_kwds['all_nans'] = False
        data = _homogeneous_data_generator(**data_gen_kwds)

    # 输出数据的形状，结构为 [n_repetitions, inherent_size, ..., n_samples]
    output_shape = [n_repetitions] + [inherent_size]*n_samples

    # 将每个样本数据中的 axis 维度移到最后，并广播到相同的形状
    data_b = [np.moveaxis(sample, axis, -1) for sample in data]
    data_b = [np.broadcast_to(sample, output_shape + [sample.shape[-1]])
              for sample in data_b]

    # 初始化一个与输出形状和输出数目相关的零数组
    res_1d = np.zeros(output_shape + [n_outputs])
    # 使用 `np.ndenumerate` 遍历一个形状为 `output_shape` 的全零数组的索引和对应值
    for i, _ in np.ndenumerate(np.zeros(output_shape)):
        # 从 `data_b` 中提取第 `i` 个索引位置的所有样本数据，形成一维数组 `data1d`
        data1d = [sample[i] for sample in data_b]
        # 检查 `data1d` 是否包含任何 NaN 值
        contains_nan = any([np.isnan(sample).any() for sample in data1d])

        # 处理 `nan_policy='raise'` 的情况
        # 在此之后，一维部分的测试已完成
        message = "The input contains nan values"
        if nan_policy == 'raise' and contains_nan:
            # 断言调用 `nan_policy_1d` 函数时会引发 `ValueError` 异常，并匹配特定错误信息 `message`
            with pytest.raises(ValueError, match=message):
                nan_policy_1d(hypotest, data1d, unpacker, *args,
                              n_outputs=n_outputs,
                              nan_policy=nan_policy,
                              paired=paired, _no_deco=True, **kwds)

            # 断言调用 `hypotest` 函数时会引发 `ValueError` 异常，并匹配特定错误信息 `message`
            with pytest.raises(ValueError, match=message):
                hypotest(*data1d, *args, nan_policy=nan_policy, **kwds)

            # 继续下一个循环迭代
            continue

        # 处理 `nan_policy='propagate'` 和 `nan_policy='omit'` 的情况

        # 获取简单参考实现的结果 `res_1da`
        try:
            res_1da = nan_policy_1d(hypotest, data1d, unpacker, *args,
                                    n_outputs=n_outputs,
                                    nan_policy=nan_policy,
                                    paired=paired, _no_deco=True, **kwds)
        except (ValueError, RuntimeWarning, ZeroDivisionError) as ea:
            ea_str = str(ea)
            # 如果异常信息 `ea_str` 以 `too_small_messages` 中的任何消息开头，则用 NaN 填充 `res_1da`
            if any([str(ea_str).startswith(msg) for msg in too_small_messages]):
                res_1da = np.full(n_outputs, np.nan)
            else:
                raise

        # 使用一维切片调用公共函数的结果 `res`，
        # 对所有切片应发出警告
        if (nan_policy == 'omit' and data_generator == "all_nans"
              and hypotest not in too_small_special_case_funcs):
            # 断言使用 `hypotest` 函数时会发出 `SmallSampleWarning` 警告，并匹配 `too_small_1d_omit`
            with pytest.warns(SmallSampleWarning, match=too_small_1d_omit):
                res = hypotest(*data1d, *args, nan_policy=nan_policy, **kwds)
        # 警告依赖于切片
        elif (nan_policy == 'omit' and data_generator == "mixed"
              and hypotest not in too_small_special_case_funcs):
            # 使用 `np.testing.suppress_warnings` 来抑制特定的警告 `SmallSampleWarning: too_small_1d_omit`
            with np.testing.suppress_warnings() as sup:
                sup.filter(SmallSampleWarning, too_small_1d_omit)
                res = hypotest(*data1d, *args, nan_policy=nan_policy, **kwds)
        # 如果没有 NaN，则不应该有警告
        else:
            res = hypotest(*data1d, *args, nan_policy=nan_policy, **kwds)
        
        # 解包 `res` 并存储为 `res_1db`
        res_1db = unpacker(res)

        # 使用 `assert_allclose` 断言 `res_1db` 和 `res_1da` 在相对误差容差 `1e-15` 下相等
        assert_allclose(res_1db, res_1da, rtol=1e-15)
        
        # 将 `res_1db` 存储在 `res_1d` 的第 `i` 列中
        res_1d[i] = res_1db

    # 将 `res_1d` 数组中的轴移动到指定位置，以便进行向量化调用假设检验
    res_1d = np.moveaxis(res_1d, -1, 0)

    # 执行假设检验的向量化调用

    # 如果 `nan_policy == 'raise'`，则检查是否引发适当的错误
    # 测试完成，因此返回
    if nan_policy == 'raise' and not data_generator == "all_finite":
        message = 'The input contains nan values'
        # 断言调用 `hypotest` 函数时会引发 `ValueError` 异常，并匹配特定错误信息 `message`
        with pytest.raises(ValueError, match=message):
            hypotest(*data, axis=axis, nan_policy=nan_policy, *args, **kwds)
        return
    # 如果 `nan_policy == 'omit'`，可能会得到一个较小的样本。
    # 检查是否需要发出相应的警告。
    if (nan_policy == 'omit' and data_generator in {"all_nans", "mixed"}
          and hypotest not in too_small_special_case_funcs):
        # 当符合条件时，使用 pytest.warns 发出 SmallSampleWarning 警告，并匹配特定的提示信息
        with pytest.warns(SmallSampleWarning, match=too_small_nd_omit):
            # 调用假设检验函数 hypotest 进行计算，传入参数包括数据 (*data)，计算轴 (axis)，nan 策略 (nan_policy)，以及其他可变参数和关键字参数
            res = hypotest(*data, axis=axis, nan_policy=nan_policy, *args, **kwds)
    else:  # 否则，不应发出警告
        # 调用假设检验函数 hypotest 进行计算，传入参数包括数据 (*data)，计算轴 (axis)，nan 策略 (nan_policy)，以及其他可变参数和关键字参数
        res = hypotest(*data, axis=axis, nan_policy=nan_policy, *args, **kwds)

    # 将计算结果 res 使用 unpacker 函数解包为 res_nd
    res_nd = unpacker(res)

    # 使用 assert_allclose 函数比较 res_nd 与 res_1d 的输出，设置相对误差容忍度为 1e-14
    assert_allclose(res_nd, res_1d, rtol=1e-14)
@pytest.mark.parametrize(("hypotest", "args", "kwds", "n_samples", "n_outputs",
                          "paired", "unpacker"), axis_nan_policy_cases)
# 使用 pytest 的 parametrize 装饰器，对多个测试参数进行参数化，以覆盖不同情况
@pytest.mark.parametrize(("nan_policy"), ("propagate", "omit", "raise"))
# 再次使用 pytest 的 parametrize 装饰器，对 nan_policy 参数进行参数化，以覆盖不同策略
@pytest.mark.parametrize(("data_generator"),
                         ("all_nans", "all_finite", "mixed", "empty"))
# 使用 pytest 的 parametrize 装饰器，对 data_generator 参数进行参数化，以覆盖不同数据生成方式
def test_axis_nan_policy_axis_is_None(hypotest, args, kwds, n_samples,
                                      n_outputs, paired, unpacker, nan_policy,
                                      data_generator):
    # check for correct behavior when `axis=None`
    # 检查当 `axis=None` 时的正确行为

    if not unpacker:
        # Define an unpacker function if it is not already defined
        # 如果未定义 unpacker 函数，则定义一个 unpacker 函数
        def unpacker(res):
            return res

    rng = np.random.default_rng(0)

    if data_generator == "empty":
        # Generate empty data arrays
        # 生成空的数据数组
        data = [rng.random((2, 0)) for i in range(n_samples)]
    else:
        # Generate data arrays with shape (2, 20)
        # 生成形状为 (2, 20) 的数据数组
        data = [rng.random((2, 20)) for i in range(n_samples)]

    if data_generator == "mixed":
        # Introduce NaN values in the data arrays based on masks
        # 根据 mask 在数据数组中引入 NaN 值
        masks = [rng.random((2, 20)) > 0.9 for i in range(n_samples)]
        for sample, mask in zip(data, masks):
            sample[mask] = np.nan
    elif data_generator == "all_nans":
        # Set all values in the data arrays to NaN
        # 将数据数组中所有值设为 NaN
        data = [sample * np.nan for sample in data]

    data_raveled = [sample.ravel() for sample in data]

    if nan_policy == 'raise' and data_generator not in {"all_finite", "empty"}:
        # If nan_policy is 'raise' and data_generator is not 'all_finite' or 'empty'
        # 如果 nan_policy 为 'raise' 并且 data_generator 不是 'all_finite' 或 'empty'

        message = 'The input contains nan values'

        # check for correct behavior whether or not data is 1d to begin with
        # 检查无论数据是否一维，都应该具有正确的行为

        # Check if ValueError is raised with the specified message
        # 检查是否引发 ValueError，并验证错误消息
        with pytest.raises(ValueError, match=message):
            hypotest(*data, axis=None, nan_policy=nan_policy,
                     *args, **kwds)
        with pytest.raises(ValueError, match=message):
            hypotest(*data_raveled, axis=None, nan_policy=nan_policy,
                     *args, **kwds)

        return

    # behavior of reference implementation with 1d input, public function with 1d
    # input, and public function with Nd input and `axis=None` should be consistent.
    # This means:
    # - If the reference version raises an error or emits a warning, it's because
    #   the sample is too small, so check that the public function emits an
    #   appropriate "too small" warning
    # - Any results returned by the three versions should be the same.
    # 引用实现的行为，对于 1 维输入、具有 1 维输入的公共函数，以及具有 Nd 输入和 `axis=None` 的公共函数应保持一致。
    # 这意味着：
    # - 如果参考版本引发错误或发出警告，则是因为样本太小，因此检查公共函数是否发出适当的 "太小" 警告
    # - 三个版本返回的任何结果应该是相同的。
    with warnings.catch_warnings():  # 捕获警告，将其作为错误处理
        warnings.simplefilter("error")

        ea_str, eb_str, ec_str = None, None, None  # 初始化三个异常消息字符串变量
        try:
            res1da = nan_policy_1d(hypotest, data_raveled, unpacker, *args,
                                   n_outputs=n_outputs, nan_policy=nan_policy,
                                   paired=paired, _no_deco=True, **kwds)
        except (RuntimeWarning, ValueError, ZeroDivisionError) as ea:
            res1da = None  # 如果捕获到异常，将结果置为None
            ea_str = str(ea)  # 记录异常消息字符串

        try:
            res1db = hypotest(*data_raveled, *args, nan_policy=nan_policy, **kwds)
        except SmallSampleWarning as eb:
            eb_str = str(eb)  # 记录警告消息字符串

        try:
            res1dc = hypotest(*data, *args, axis=None, nan_policy=nan_policy, **kwds)
        except SmallSampleWarning as ec:
            ec_str = str(ec)  # 记录警告消息字符串

    if ea_str or eb_str or ec_str:  # 如果有任何异常或警告消息
        # 如果参考实现生成了异常或警告，确保消息是预期的“太小”的消息之一。
        # 注意，有些函数在没有装饰器的情况下根本不会抱怨，这也是可以接受的。
        ok_msg = any([str(ea_str).startswith(msg) for msg in too_small_messages])
        assert (ea_str is None) or ok_msg  # 断言确保异常消息为空或是预期消息之一

        # 确保包装函数发出了预期的警告
        desired_warnings = {too_small_1d_omit, too_small_1d_not_omit}
        assert str(eb_str) in desired_warnings  # 断言确保警告消息在预期的警告集合中
        assert str(ec_str) in desired_warnings  # 断言确保警告消息在预期的警告集合中

        with warnings.catch_warnings():  # 忽略警告以获取返回值
            warnings.simplefilter("ignore")
            res1db = hypotest(*data_raveled, *args, nan_policy=nan_policy, **kwds)
            res1dc = hypotest(*data, *args, axis=None, nan_policy=nan_policy, **kwds)

    # 确保引用/公共函数返回的任何结果是相同的，并且所有属性都是NumPy标量
    res1db, res1dc = unpacker(res1db), unpacker(res1dc)
    assert_equal(res1dc, res1db)  # 断言确保两个结果相等
    all_results = list(res1db) + list(res1dc)

    if res1da is not None:
        assert_allclose(res1db, res1da, rtol=1e-15)  # 断言确保结果非常接近
        all_results += list(res1da)

    for item in all_results:
        assert np.issubdtype(item.dtype, np.number)  # 断言确保所有项的数据类型是NumPy数值类型
        assert np.isscalar(item)  # 断言确保所有项都是标量
# 测试 keepdims 参数用于：
#   - 单输出和多输出函数（gmean 和 mannwhitneyu）
#   - 负轴、正轴、无轴和元组
#   - 1D 数组且没有 NaN
#   - 1D 数组且 NaN 传播
#   - 零尺寸输出
@pytest.mark.filterwarnings('ignore:All axis-slices of one...')
@pytest.mark.filterwarnings('ignore:After omitting NaNs...')
# 这些警告是为 `ttest_1samp` 添加的，在处理后应被处理和移除
@pytest.mark.filterwarnings('ignore:divide by zero encountered...')
@pytest.mark.filterwarnings('ignore:invalid value encountered...')
# 参数化测试，包括不同的假设检验函数（hypotest）、参数（args 和 kwds）、样本数（n_samples）、解包函数（unpacker）
@pytest.mark.parametrize(
    ("hypotest", "args", "kwds", "n_samples", "unpacker"),
    ((stats.gmean, tuple(), dict(), 1, lambda x: (x,)),
     (stats.mannwhitneyu, tuple(), {'method': 'asymptotic'}, 2, None),
     (stats.ttest_1samp, (0,), dict(), 1, unpack_ttest_result),
     (xp_mean_1samp, tuple(), dict(), 1, lambda x: (x,)),
     (xp_mean_2samp, tuple(), dict(), 2, lambda x: (x,))),
)
# 参数化测试，包括样本形状（sample_shape）和轴的情况（axis_cases）
@pytest.mark.parametrize(
    ("sample_shape", "axis_cases"),
    (((2, 3, 3, 4), (None, 0, -1, (0, 2), (1, -1), (3, 1, 2, 0))),
     ((10, ), (0, -1)),
     ((20, 0), (0, 1)))
)
# 测试 keepdims 参数的函数
def test_keepdims(hypotest, args, kwds, n_samples, unpacker,
                  sample_shape, axis_cases, nan_policy):
    # 如果没有指定解包函数，则定义一个默认的解包函数，用于返回结果
    if not unpacker:
        def unpacker(res):
            return res
    # 使用默认随机数生成器创建随机数种子为 0 的实例
    rng = np.random.default_rng(0)
    # 创建包含指定形状数据的列表
    data = [rng.random(sample_shape) for _ in range(n_samples)]
    # 创建包含 NaN 数据的列表，用于测试 NaN 的传播
    nan_data = [sample.copy() for sample in data]
    # 创建掩码，用于将数据中的部分值设为 NaN
    nan_mask = [rng.random(sample_shape) < 0.2 for _ in range(n_samples)]
    # 将数据中的对应位置应用掩码，设为 NaN
    for sample, mask in zip(nan_data, nan_mask):
        sample[mask] = np.nan
    # 对于每一个轴的情况进行迭代处理
    for axis in axis_cases:
        # 复制样本形状作为期望形状
        expected_shape = list(sample_shape)
        # 如果轴为 None，则期望形状中所有维度为 1
        if axis is None:
            expected_shape = np.ones(len(sample_shape))
        else:
            # 如果轴是整数，则将对应位置的维度设为 1
            if isinstance(axis, int):
                expected_shape[axis] = 1
            else:
                # 如果轴是一个列表，则将列表中指定位置的维度设为 1
                for ax in axis:
                    expected_shape[ax] = 1
        # 将期望形状转换为元组
        expected_shape = tuple(expected_shape)
        
        # 执行带有指定参数的假设检验，保留所有维度
        res = unpacker(hypotest(*data, *args, axis=axis, keepdims=True,
                                **kwds))
        # 执行带有指定参数的假设检验，不保留任何维度
        res_base = unpacker(hypotest(*data, *args, axis=axis, keepdims=False,
                                     **kwds))
        # 执行带有指定参数的假设检验，保留所有维度（针对包含 NaN 的数据）
        nan_res = unpacker(hypotest(*nan_data, *args, axis=axis,
                                    keepdims=True, nan_policy=nan_policy,
                                    **kwds))
        # 执行带有指定参数的假设检验，不保留任何维度（针对包含 NaN 的数据）
        nan_res_base = unpacker(hypotest(*nan_data, *args, axis=axis,
                                         keepdims=False,
                                         nan_policy=nan_policy, **kwds))
        
        # 对结果进行迭代处理，确保形状符合期望形状
        for r, r_base, rn, rn_base in zip(res, res_base, nan_res,
                                          nan_res_base):
            # 断言结果 r 的形状与期望形状一致
            assert r.shape == expected_shape
            # 如果指定了轴，则在轴上对结果 r 进行降维处理
            r = np.squeeze(r, axis=axis)
            # 断言处理后的结果与基准结果 r_base 一致
            assert_equal(r, r_base)
            # 断言结果 rn 的形状与期望形状一致
            assert rn.shape == expected_shape
            # 如果指定了轴，则在轴上对结果 rn 进行降维处理
            rn = np.squeeze(rn, axis=axis)
            # 断言处理后的结果与基准结果 rn_base 一致
            assert_equal(rn, rn_base)
# 使用 pytest.mark.parametrize 装饰器定义测试函数，参数包括函数和样本数
@pytest.mark.parametrize(("fun", "nsamp"),
                         [(stats.kstat, 1),
                          (stats.kstatvar, 1)])
def test_hypotest_back_compat_no_axis(fun, nsamp):
    # 定义数组的维度
    m, n = 8, 9

    # 创建随机数生成器对象
    rng = np.random.default_rng(0)
    # 生成指定维度的随机数组 x
    x = rng.random((nsamp, m, n))
    # 调用被测试函数 fun 处理数组 x
    res = fun(*x)
    # 调用带有 _no_deco 参数的 fun 处理数组 x
    res2 = fun(*x, _no_deco=True)
    # 将数组 x 的每个元素展平后作为参数调用 fun
    res3 = fun([xi.ravel() for xi in x])
    # 断言处理结果的一致性
    assert_equal(res, res2)
    assert_equal(res, res3)


# 使用 pytest.mark.parametrize 装饰器定义测试函数，测试不同的 axis 参数
@pytest.mark.parametrize(("axis"), (0, 1, 2))
def test_axis_nan_policy_decorated_positional_axis(axis):
    # 测试函数装饰了 _axis_nan_policy_decorator 的行为，检查当 axis 作为位置参数或关键字参数提供时的行为是否正确

    # 定义数组的形状
    shape = (8, 9, 10)
    # 创建随机数生成器对象
    rng = np.random.default_rng(0)
    # 生成指定形状的随机数组 x 和 y
    x = rng.random(shape)
    y = rng.random(shape)
    # 调用 stats.mannwhitneyu 函数，分别传入 axis 作为位置参数和关键字参数
    res1 = stats.mannwhitneyu(x, y, True, 'two-sided', axis)
    res2 = stats.mannwhitneyu(x, y, True, 'two-sided', axis=axis)
    # 断言两次调用的结果一致
    assert_equal(res1, res2)

    # 测试函数调用时传入了多个同名参数 'axis' 的情况
    message = "mannwhitneyu() got multiple values for argument 'axis'"
    with pytest.raises(TypeError, match=re.escape(message)):
        stats.mannwhitneyu(x, y, True, 'two-sided', axis, axis=axis)


# 测试函数，测试函数装饰了 _axis_nan_policy_decorator 且接受 *args 参数的情况
def test_axis_nan_policy_decorated_positional_args():
    # 测试函数装饰了 _axis_nan_policy_decorator 的行为，检查当函数接受 *args 参数时的行为是否正确

    # 定义数组的形状
    shape = (3, 8, 9, 10)
    # 创建随机数生成器对象
    rng = np.random.default_rng(0)
    # 生成指定形状的随机数组 x
    x = rng.random(shape)
    # 将数组中的一个元素设置为 NaN
    x[0, 0, 0, 0] = np.nan
    # 调用 stats.kruskal 函数，传入 *x 作为参数
    stats.kruskal(*x)

    # 测试函数调用时传入了意外的关键字参数 'samples' 的情况
    message = "kruskal() got an unexpected keyword argument 'samples'"
    with pytest.raises(TypeError, match=re.escape(message)):
        stats.kruskal(samples=x)

    # 测试函数调用时传入了多个同名参数 'samples' 的情况
    with pytest.raises(TypeError, match=re.escape(message)):
        stats.kruskal(*x, samples=x)


# 测试函数，测试函数装饰了 _axis_nan_policy_decorator 且关键字参数为 samples 的情况
def test_axis_nan_policy_decorated_keyword_samples():
    # 测试函数装饰了 _axis_nan_policy_decorator 的行为，检查当 samples 作为位置参数或关键字参数提供时的行为是否正确

    # 定义数组的形状
    shape = (2, 8, 9, 10)
    # 创建随机数生成器对象
    rng = np.random.default_rng(0)
    # 生成指定形状的随机数组 x
    x = rng.random(shape)
    # 将数组中的一个元素设置为 NaN
    x[0, 0, 0, 0] = np.nan
    # 调用 stats.mannwhitneyu 函数，分别传入 *x 作为参数和 x[0], x[1] 作为关键字参数
    res1 = stats.mannwhitneyu(*x)
    res2 = stats.mannwhitneyu(x=x[0], y=x[1])
    # 断言两次调用的结果一致
    assert_equal(res1, res2)

    # 测试函数调用时传入了多个同名参数 'x' 的情况
    message = "mannwhitneyu() got multiple values for argument"
    with pytest.raises(TypeError, match=re.escape(message)):
        stats.mannwhitneyu(*x, x=x[0], y=x[1])


# 使用 pytest.mark.parametrize 装饰器定义测试函数，测试函数装饰了 _axis_nan_policy_decorator 的行为的 pickle 情况
@pytest.mark.parametrize(("hypotest", "args", "kwds", "n_samples", "n_outputs",
                          "paired", "unpacker"), axis_nan_policy_cases)
def test_axis_nan_policy_decorated_pickled(hypotest, args, kwds, n_samples,
                                           n_outputs, paired, unpacker):
    # 当 hypotest 函数名包含 'ttest_ci' 时跳过该测试，因为无法 pickle 在函数内部定义的函数。

    # 创建随机数生成器对象
    rng = np.random.default_rng(0)

    # 有些假设检验返回的是非可迭代对象，需要使用 unpacker 提取统计值和 p 值。对于不需要的情况：

    # 完整的代码已经完成，没有进一步的注释需求。
    # 如果未提供 unpacker 函数，则定义一个简单的 unpacker 函数，直接返回其输入参数
    if not unpacker:
        def unpacker(res):
            return res

    # 使用随机数生成器 rng 创建一个指定大小的三维数组，每个元素均为在[0, 1)范围内的均匀分布随机数
    data = rng.uniform(size=(n_samples, 2, 30))
    
    # 使用 pickle 序列化 hypotest 对象，得到其字节流表示
    pickled_hypotest = pickle.dumps(hypotest)
    
    # 使用 pickle 反序列化 pickled_hypotest 字节流，得到原始的 hypotest 对象
    unpickled_hypotest = pickle.loads(pickled_hypotest)
    
    # 调用 hypotest 函数并传入 data、args 和 kwds 参数，计算结果并通过 unpacker 函数进行处理
    res1 = unpacker(hypotest(*data, *args, axis=-1, **kwds))
    
    # 调用 unpickled_hypotest 函数并传入相同的参数，计算结果并通过 unpacker 函数进行处理
    res2 = unpacker(unpickled_hypotest(*data, *args, axis=-1, **kwds))
    
    # 断言 res1 和 res2 的数值近似相等，如果不相等则会引发 AssertionError
    assert_allclose(res1, res2, rtol=1e-12)
def test_check_empty_inputs():
    # 测试 _check_empty_inputs 是否正常工作，至少对于单样本输入。（多样本功能在下面测试。）
    # 如果输入样本不为空，应返回 None。
    # 如果输入样本为空，应返回具有 NaN 或适当形状的空数组。np.mean 用作输出的参考，因为它像这些函数计算的统计量一样，沿着并"消耗" `axis`，但保留其他轴。
    for i in range(5):
        for combo in combinations_with_replacement([0, 1, 2], i):
            for axis in range(len(combo)):
                samples = (np.zeros(combo),)
                output = stats._axis_nan_policy._check_empty_inputs(samples,
                                                                    axis)
                if output is not None:
                    with np.testing.suppress_warnings() as sup:
                        sup.filter(RuntimeWarning, "Mean of empty slice.")
                        sup.filter(RuntimeWarning, "invalid value encountered")
                        reference = samples[0].mean(axis=axis)
                    np.testing.assert_equal(output, reference)


def _check_arrays_broadcastable(arrays, axis):
    # https://numpy.org/doc/stable/user/basics.broadcasting.html
    # "When operating on two arrays, NumPy compares their shapes element-wise.
    # It starts with the trailing (i.e. rightmost) dimensions and works its
    # way left.
    # Two dimensions are compatible when
    # 1. they are equal, or
    # 2. one of them is 1
    # ...
    # Arrays do not need to have the same number of dimensions."
    # （澄清：如果根据上述标准数组是兼容的，且数组耗尽维度，则它仍然是兼容的。）
    # 在下面，我们遵循上述规则，除了忽略 `axis`

    n_dims = max([arr.ndim for arr in arrays])
    if axis is not None:
        # 转换为负数轴
        axis = (-n_dims + axis) if axis >= 0 else axis

    for dim in range(1, n_dims+1):  # 我们将从 -1 到 -n_dims（包括）进行索引
        if -dim == axis:
            continue  # 忽略沿 `axis` 的长度

        dim_lengths = set()
        for arr in arrays:
            if dim <= arr.ndim and arr.shape[-dim] != 1:
                dim_lengths.add(arr.shape[-dim])

        if len(dim_lengths) > 1:
            return False
    return True


@pytest.mark.slow
@pytest.mark.parametrize(("hypotest", "args", "kwds", "n_samples", "n_outputs",
                          "paired", "unpacker"), axis_nan_policy_cases)
def test_empty(hypotest, args, kwds, n_samples, n_outputs, paired, unpacker):
    # 当至少有一个输入为空时，测试正确的输出形状
    if hypotest in {stats.kruskal, stats.friedmanchisquare} and not SCIPY_XSLOW:
        pytest.skip("Too slow.")
    # 如果 hypotest 函数名在 override_propagate_funcs 列表中
    if hypotest in override_propagate_funcs:
        # 准备跳过测试的原因字符串
        reason = "Doesn't follow the usual pattern. Tested separately."
        # 使用 pytest 跳过测试，并传递跳过的原因
        pytest.skip(reason=reason)

    # 如果 unpacker 函数未定义，使用一个匿名函数作为默认的 unpacker
    if unpacker is None:
        unpacker = lambda res: (res[0], res[1])  # noqa: E731

    # 定义一个生成小数据的生成器函数
    def small_data_generator(n_samples, n_dims):

        # 定义一个生成小样本的生成器函数
        def small_sample_generator(n_dims):
            # 返回所有可能的小数组，最多 n_dims 维度
            for i in n_dims:
                # "small" 表示沿每个维度的大小为 0 或 1
                for combo in combinations_with_replacement([0, 1, 2], i):
                    yield np.zeros(combo)

        # 生成多个小样本生成器的列表
        gens = [small_sample_generator(n_dims) for i in range(n_samples)]
        # 使用 product 函数生成所有小样本的组合
        yield from product(*gens)

    # 定义可能的维度列表
    n_dims = [1, 2, 3]
    # 从 `small_data_generator` 生成器中迭代获取 `n_samples` 个样本
    for samples in small_data_generator(n_samples, n_dims):

        # 检查是否存在大小为零的数组样本，如果有则跳过当前循环
        if not any(sample.size == 0 for sample in samples):
            continue

        # 计算样本中维度最大的数组的维数
        max_axis = max(sample.ndim for sample in samples)

        # 需要测试所有有效的 `axis` 参数值
        for axis in range(-max_axis, max_axis):

            try:
                # 使用 `_broadcast_concatenate` 函数进行广播后的拼接，
                # 所有数组形状相同，因此输出的形状应与单样本统计量的形状相同，以 `np.mean` 作为参考。
                concat = stats._stats_py._broadcast_concatenate(samples, axis)

                # 使用 `np.testing.suppress_warnings` 禁止特定的警告信息
                with np.testing.suppress_warnings() as sup:
                    sup.filter(RuntimeWarning, "Mean of empty slice.")
                    sup.filter(RuntimeWarning, "invalid value encountered")
                    # 计算期望的值，将结果乘以 `np.nan`
                    expected = np.mean(concat, axis=axis) * np.nan

                # 如果 `hypotest` 函数在空特殊情况函数集合中
                if hypotest in empty_special_case_funcs:
                    # 对空值进行特殊处理
                    empty_val = hypotest(*([[]]*len(samples)), *args, **kwds)
                    expected = np.asarray(expected)
                    mask = np.isnan(expected)
                    expected[mask] = empty_val
                    expected = expected[()]

                # 如果期望值的大小不为零，并且 `hypotest` 不在过小特殊情况函数集合中
                if expected.size and hypotest not in too_small_special_case_funcs:
                    # 根据 `axis` 参数进行假设检验，并捕获小样本警告
                    message = (too_small_1d_not_omit if max_axis == 1
                               else too_small_nd_not_omit)
                    with pytest.warns(SmallSampleWarning, match=message):
                        res = hypotest(*samples, *args, axis=axis, **kwds)
                else:
                    # 否则，对 `hypotest` 函数调用进行警告抑制
                    with np.testing.suppress_warnings() as sup:
                        # `f_oneway` 的特殊情况处理
                        sup.filter(SmallSampleWarning, "all input arrays have length 1")
                        res = hypotest(*samples, *args, axis=axis, **kwds)

                # 解包结果 `res`
                res = unpacker(res)

                # 对每个输出结果进行断言，确保与期望值 `expected` 相等
                for i in range(n_outputs):
                    assert_equal(res[i], expected)

            except ValueError:
                # 确认这些数组确实无法进行广播
                assert not _check_arrays_broadcastable(samples,
                                                       None if paired else axis)

                # 确认 `_broadcast_concatenate` 和 `hypotest` 都能提供此信息
                message = "Array shapes are incompatible for broadcasting."
                with pytest.raises(ValueError, match=message):
                    stats._stats_py._broadcast_concatenate(samples, axis, paired)
                with pytest.raises(ValueError, match=message):
                    hypotest(*samples, *args, axis=axis, **kwds)
def test_masked_array_2_sentinel_array():
    # prepare arrays
    np.random.seed(0)
    A = np.random.rand(10, 11, 12)  # 创建一个形状为 (10, 11, 12) 的随机数组 A
    B = np.random.rand(12)  # 创建一个形状为 (12,) 的随机数组 B
    mask = A < 0.5  # 根据 A 中小于 0.5 的元素创建一个掩码数组
    A = np.ma.masked_array(A, mask)  # 使用掩码数组创建一个掩码数组对象 A

    # set arbitrary elements to special values
    # (these values might have been considered for use as sentinel values)
    max_float = np.finfo(np.float64).max  # 获取 np.float64 类型的最大值
    max_float2 = np.nextafter(max_float, -np.inf)  # 获取比 max_float 稍小的浮点数
    max_float3 = np.nextafter(max_float2, -np.inf)  # 进一步获取比 max_float2 更小的浮点数
    A[3, 4, 1] = np.nan  # 将 A 中索引为 (3, 4, 1) 的元素设为 NaN
    A[4, 5, 2] = np.inf  # 将 A 中索引为 (4, 5, 2) 的元素设为正无穷
    A[5, 6, 3] = max_float  # 将 A 中索引为 (5, 6, 3) 的元素设为 max_float
    B[8] = np.nan  # 将 B 中索引为 8 的元素设为 NaN
    B[7] = np.inf  # 将 B 中索引为 7 的元素设为正无穷
    B[6] = max_float2  # 将 B 中索引为 6 的元素设为 max_float2

    # convert masked A to array with sentinel value, don't modify B
    out_arrays, sentinel = _masked_arrays_2_sentinel_arrays([A, B])  # 调用函数将 A 和 B 转换成带有哨兵值的数组
    A_out, B_out = out_arrays  # 分别获取转换后的 A 和 B

    # check that good sentinel value was chosen (according to intended logic)
    assert (sentinel != max_float) and (sentinel != max_float2)  # 检查选择的哨兵值不是 max_float 和 max_float2
    assert sentinel == max_float3  # 检查选择的哨兵值是 max_float3

    # check that output arrays are as intended
    A_reference = A.data  # 获取 A 的数据部分（非掩码部分）
    A_reference[A.mask] = sentinel  # 将 A 中的掩码部分替换为哨兵值
    np.testing.assert_array_equal(A_out, A_reference)  # 断言 A_out 与 A_reference 相等
    assert B_out is B  # 断言 B_out 和原始 B 对象相同


@skip_xp_invalid_arg
def test_masked_dtype():
    # When _masked_arrays_2_sentinel_arrays was first added, it always
    # upcast the arrays to np.float64. After gh16662, check expected promotion
    # and that the expected sentinel is found.

    # these are important because the max of the promoted dtype is the first
    # candidate to be the sentinel value
    max16 = np.iinfo(np.int16).max  # 获取 np.int16 类型的最大值
    max128c = np.finfo(np.complex128).max  # 获取 np.complex128 类型的最大值

    # a is a regular array, b has masked elements, and c has no masked elements
    a = np.array([1, 2, max16], dtype=np.int16)  # 创建一个 np.int16 类型的普通数组 a
    b = np.ma.array([1, 2, 1], dtype=np.int8, mask=[0, 1, 0])  # 创建一个带掩码的 np.int8 类型数组 b
    c = np.ma.array([1, 2, 1], dtype=np.complex128, mask=[0, 0, 0])  # 创建一个带掩码的 np.complex128 类型数组 c

    # check integer masked -> sentinel conversion
    out_arrays, sentinel = _masked_arrays_2_sentinel_arrays([a, b])  # 将 a 和 b 转换为带哨兵值的数组
    a_out, b_out = out_arrays  # 获取转换后的 a 和 b
    assert sentinel == max16 - 1  # 断言选择的哨兵值是 max16-1，因为 max16 已经在数据中
    assert b_out.dtype == np.int16  # 检查 b_out 的类型是否被提升为 np.int16
    assert_allclose(b_out, [b[0], sentinel, b[-1]])  # 检查哨兵值的放置位置是否正确
    assert a_out is a  # 因为 a 不是掩码数组，所以保持不变
    assert not isinstance(b_out, np.ma.MaskedArray)  # 检查 b 是否已经变为普通数组

    # similarly with complex
    out_arrays, sentinel = _masked_arrays_2_sentinel_arrays([b, c])  # 将 b 和 c 转换为带哨兵值的数组
    b_out, c_out = out_arrays  # 获取转换后的 b 和 c
    assert sentinel == max128c  # 断言选择的哨兵值是 max128c，因为它不在数据中
    assert b_out.dtype == np.complex128  # 检查 b 是否被提升为 np.complex128
    assert_allclose(b_out, [b[0], sentinel, b[-1]])  # 检查哨兵值的放置位置是否正确
    assert not isinstance(b_out, np.ma.MaskedArray)  # 检查 b 是否已经变为普通数组
    assert not isinstance(c_out, np.ma.MaskedArray)  # 检查 c 是否已经变为普通数组

    # Also, check edge case when a sentinel value cannot be found in the data
    min8, max8 = np.iinfo(np.int8).min, np.iinfo(np.int8).max  # 获取 np.int8 类型的最小值和最大值
    # 使用 np.arange 创建一个包含指定范围内所有可能值的数组，数据类型为 np.int8
    a = np.arange(min8, max8+1, dtype=np.int8)  # use all possible values

    # 创建一个与数组 a 相同形状的布尔类型的全零数组作为掩码
    mask1 = np.zeros_like(a, dtype=bool)
    mask0 = np.zeros_like(a, dtype=bool)

    # 将掩码数组 mask1 的第二个元素标记为 True，作为哨兵值
    # 创建一个带有掩码的数组 a1，使用 mask1 作为掩码
    mask1[1] = True
    a1 = np.ma.array(a, mask=mask1)

    # 调用函数 _masked_arrays_2_sentinel_arrays 处理带掩码的数组 a1
    # 返回处理后的输出数组列表 out_arrays 和哨兵值 sentinel
    out_arrays, sentinel = _masked_arrays_2_sentinel_arrays([a1])
    # 使用断言检查哨兵值是否等于 min8 + 1
    assert sentinel == min8+1

    # 将掩码数组 mask0 的第一个元素标记为 True，作为最小可能值的哨兵
    # 创建一个带有掩码的数组 a0，使用 mask0 作为掩码
    mask0[0] = True
    a0 = np.ma.array(a, mask=mask0)

    # 准备用于 pytest 的异常消息字符串
    message = "This function replaces masked elements with sentinel..."
    # 使用 pytest.raises 检查调用 _masked_arrays_2_sentinel_arrays 处理带掩码的数组 a0 时是否引发 ValueError 异常，并匹配异常消息
    with pytest.raises(ValueError, match=message):
        _masked_arrays_2_sentinel_arrays([a0])

    # 创建一个带有掩码的数组 a，指定元素和掩码的数据类型为 np.float32
    a = np.ma.array([1, 2, 3], mask=[0, 1, 0], dtype=np.float32)
    # 使用断言检查调用 stats.gmean 函数计算数组 a 的几何平均值后，其数据类型是否仍为 np.float32
    assert stats.gmean(a).dtype == np.float32
def test_masked_stat_1d():
    # basic test of _axis_nan_policy_factory with 1D masked sample

    # 生成两组数据：男性和女性的年龄
    males = [19, 22, 16, 29, 24]
    females = [20, 11, 17, 12]

    # 使用 Mann-Whitney U 检验计算两组数据之间的统计量
    res = stats.mannwhitneyu(males, females)

    # 测试当额外的 nan 被省略时是否得到相同的结果
    females2 = [20, 11, 17, np.nan, 12]
    res2 = stats.mannwhitneyu(males, females2, nan_policy='omit')

    # 使用 NumPy 的测试工具断言两个结果数组相等
    np.testing.assert_array_equal(res2, res)

    # 测试当额外的元素被遮蔽时是否得到相同的结果
    females3 = [20, 11, 17, 1000, 12]
    mask3 = [False, False, False, True, False]
    females3 = np.ma.masked_array(females3, mask=mask3)
    res3 = stats.mannwhitneyu(males, females3)
    np.testing.assert_array_equal(res3, res)

    # 测试当额外的 nan 被省略并且额外的元素被遮蔽时是否得到相同的结果
    females4 = [20, 11, 17, np.nan, 1000, 12]
    mask4 = [False, False, False, False, True, False]
    females4 = np.ma.masked_array(females4, mask=mask4)
    res4 = stats.mannwhitneyu(males, females4, nan_policy='omit')
    np.testing.assert_array_equal(res4, res)

    # 测试当额外的元素（包括 nan）被遮蔽时是否得到相同的结果
    females5 = [20, 11, 17, np.nan, 1000, 12]
    mask5 = [False, False, False, True, True, False]
    females5 = np.ma.masked_array(females5, mask=mask5)

    # 测试不同的 nan_policy 参数下是否得到相同的结果
    res5 = stats.mannwhitneyu(males, females5, nan_policy='propagate')
    res6 = stats.mannwhitneyu(males, females5, nan_policy='raise')
    np.testing.assert_array_equal(res5, res)
    np.testing.assert_array_equal(res6, res)


@pytest.mark.filterwarnings('ignore:After omitting NaNs...')
@pytest.mark.filterwarnings('ignore:One or more axis-slices of one...')
@skip_xp_invalid_arg
@pytest.mark.parametrize(("axis"), range(-3, 3))
def test_masked_stat_3d(axis):
    # basic test of _axis_nan_policy_factory with 3D masked sample

    # 创建随机数种子
    np.random.seed(0)

    # 生成三维数组 a、二维数组 b 和一维数组 c
    a = np.random.rand(3, 4, 5)
    b = np.random.rand(4, 5)
    c = np.random.rand(4, 1)

    # 创建遮蔽数组 mask_a 和 mask_c，遮蔽一些元素
    mask_a = a < 0.1
    mask_c = [False, False, False, True]

    # 使用遮蔽数组创建遮蔽后的数组 a_masked 和 c_masked
    a_masked = np.ma.masked_array(a, mask=mask_a)
    c_masked = np.ma.masked_array(c, mask=mask_c)

    # 创建含有 nan 的数组 a_nans 和 c_nans
    a_nans = a.copy()
    a_nans[mask_a] = np.nan
    c_nans = c.copy()
    c_nans[mask_c] = np.nan

    # 使用 Kruskal-Wallis 检验计算多个数组之间的统计量，比较不同的 nan_policy 参数
    res = stats.kruskal(a_nans, b, c_nans, nan_policy='omit', axis=axis)
    res2 = stats.kruskal(a_masked, b, c_masked, axis=axis)
    np.testing.assert_array_equal(res, res2)


@pytest.mark.filterwarnings('ignore:After omitting NaNs...')
@pytest.mark.filterwarnings('ignore:One or more axis-slices of one...')
@skip_xp_invalid_arg
def test_mixed_mask_nan_1():
    # targeted test of _axis_nan_policy_factory with 2D masked sample:
    # omitting samples with masks and nan_policy='omit' are equivalent
    # also checks paired-sample sentinel value removal

    # 设置数组维度
    m, n = 3, 20
    axis = -1

    # 创建随机数种子
    np.random.seed(0)

    # 生成二维数组 a 和 b，以及对应的遮蔽数组 mask_a1, mask_a2, mask_b1, mask_b2
    a = np.random.rand(m, n)
    b = np.random.rand(m, n)
    mask_a1 = np.random.rand(m, n) < 0.2
    mask_a2 = np.random.rand(m, n) < 0.1
    mask_b1 = np.random.rand(m, n) < 0.15
    mask_b2 = np.random.rand(m, n) < 0.15
    # 将 mask_a1 的第三行全置为 True
    mask_a1[2, :] = True
    
    # 复制数组 a 和 b 到 a_nans 和 b_nans
    a_nans = a.copy()
    b_nans = b.copy()
    
    # 根据 mask_a1 和 mask_a2 将 a_nans 和 b_nans 中相应位置置为 NaN
    a_nans[mask_a1 | mask_a2] = np.nan
    b_nans[mask_b1 | mask_b2] = np.nan
    
    # 使用 mask_a1 创建 a_masked1 和 mask_b1 创建 b_masked1 的掩码数组，将 mask_a2 和 mask_b2 中的位置置为 NaN
    a_masked1 = np.ma.masked_array(a, mask=mask_a1)
    b_masked1 = np.ma.masked_array(b, mask=mask_b1)
    a_masked1[mask_a2] = np.nan
    b_masked1[mask_b2] = np.nan
    
    # 使用 mask_a2 创建 a_masked2 和 mask_b2 创建 b_masked2 的掩码数组，将 mask_a1 和 mask_b1 中的位置置为 NaN
    a_masked2 = np.ma.masked_array(a, mask=mask_a2)
    b_masked2 = np.ma.masked_array(b, mask=mask_b2)
    a_masked2[mask_a1] = np.nan
    b_masked2[mask_b1] = np.nan
    
    # 使用 (mask_a1 | mask_a2) 创建 a_masked3 和 (mask_b1 | mask_b2) 创建 b_masked3 的掩码数组
    
    a_masked3 = np.ma.masked_array(a, mask=(mask_a1 | mask_a2))
    b_masked3 = np.ma.masked_array(b, mask=(mask_b1 | mask_b2))
    
    # 对 a_nans 和 b_nans 进行无 NaN 策略的 Wilcoxon 符号秩检验，结果存储在 res 中
    res = stats.wilcoxon(a_nans, b_nans, nan_policy='omit', axis=axis)
    
    # 对 a_masked1 和 b_masked1 进行无 NaN 策略的 Wilcoxon 符号秩检验，结果存储在 res1 中
    res1 = stats.wilcoxon(a_masked1, b_masked1, nan_policy='omit', axis=axis)
    
    # 对 a_masked2 和 b_masked2 进行无 NaN 策略的 Wilcoxon 符号秩检验，结果存储在 res2 中
    res2 = stats.wilcoxon(a_masked2, b_masked2, nan_policy='omit', axis=axis)
    
    # 对 a_masked3 和 b_masked3 进行 NaN 策略为 'raise' 的 Wilcoxon 符号秩检验，结果存储在 res3 中
    res3 = stats.wilcoxon(a_masked3, b_masked3, nan_policy='raise', axis=axis)
    
    # 对 a_masked3 和 b_masked3 进行 NaN 策略为 'propagate' 的 Wilcoxon 符号秩检验，结果存储在 res4 中
    res4 = stats.wilcoxon(a_masked3, b_masked3, nan_policy='propagate', axis=axis)
    
    # 断言 res1、res2、res3 和 res4 的数组内容与 res 相等
    np.testing.assert_array_equal(res1, res)
    np.testing.assert_array_equal(res2, res)
    np.testing.assert_array_equal(res3, res)
    np.testing.assert_array_equal(res4, res)
@pytest.mark.filterwarnings('ignore:After omitting NaNs...')
@pytest.mark.filterwarnings('ignore:One or more axis-slices of one...')
@skip_xp_invalid_arg
def test_mixed_mask_nan_2():
    # targeted test of _axis_nan_policy_factory with 2D masked sample:
    # check for expected interaction between masks and nans

    # Cases here are
    # [mixed nan/mask, all nans, all masked,
    # unmasked nan, masked nan, unmasked non-nan]
    a = [[1, np.nan, 2], [np.nan, np.nan, np.nan], [1, 2, 3],
         [1, np.nan, 3], [1, np.nan, 3], [1, 2, 3]]
    mask = [[1, 0, 1], [0, 0, 0], [1, 1, 1],
            [0, 0, 0], [0, 1, 0], [0, 0, 0]]
    a_masked = np.ma.masked_array(a, mask=mask)
    b = [[4, 5, 6]]
    ref1 = stats.ranksums([1, 3], [4, 5, 6])
    ref2 = stats.ranksums([1, 2, 3], [4, 5, 6])

    # nan_policy = 'omit'
    # all elements are removed from first three rows
    # middle element is removed from fourth and fifth rows
    # no elements removed from last row
    res = stats.ranksums(a_masked, b, nan_policy='omit', axis=-1)
    stat_ref = [np.nan, np.nan, np.nan,
                ref1.statistic, ref1.statistic, ref2.statistic]
    p_ref = [np.nan, np.nan, np.nan,
             ref1.pvalue, ref1.pvalue, ref2.pvalue]
    np.testing.assert_array_equal(res.statistic, stat_ref)
    np.testing.assert_array_equal(res.pvalue, p_ref)

    # nan_policy = 'propagate'
    # nans propagate in first, second, and fourth row
    # all elements are removed by mask from third row
    # middle element is removed from fifth row
    # no elements removed from last row
    res = stats.ranksums(a_masked, b, nan_policy='propagate', axis=-1)
    stat_ref = [np.nan, np.nan, np.nan,
                np.nan, ref1.statistic, ref2.statistic]
    p_ref = [np.nan, np.nan, np.nan,
             np.nan, ref1.pvalue, ref2.pvalue]
    np.testing.assert_array_equal(res.statistic, stat_ref)
    np.testing.assert_array_equal(res.pvalue, p_ref)


def test_axis_None_vs_tuple():
    # `axis` `None` should be equivalent to tuple with all axes
    shape = (3, 8, 9, 10)
    rng = np.random.default_rng(0)
    x = rng.random(shape)
    res = stats.kruskal(*x, axis=None)
    res2 = stats.kruskal(*x, axis=(0, 1, 2))
    np.testing.assert_array_equal(res, res2)


def test_axis_None_vs_tuple_with_broadcasting():
    # `axis` `None` should be equivalent to tuple with all axes,
    # which should be equivalent to raveling the arrays before passing them
    rng = np.random.default_rng(0)
    x = rng.random((5, 1))
    y = rng.random((1, 5))
    x2, y2 = np.broadcast_arrays(x, y)

    res0 = stats.mannwhitneyu(x.ravel(), y.ravel())
    res1 = stats.mannwhitneyu(x, y, axis=None)
    res2 = stats.mannwhitneyu(x, y, axis=(0, 1))
    res3 = stats.mannwhitneyu(x2.ravel(), y2.ravel())

    assert res1 == res0
    assert res2 == res0
    assert res3 != res0


@pytest.mark.parametrize(("axis"),
                         list(permutations(range(-3, 3), 2)) + [(-4, 1)])
def test_other_axis_tuples(axis):
    # Parameterized test to cover various axis permutations
    # including invalid negative axis values
    # 检查 _axis_nan_policy_factory 是否按预期处理所有 `axis` 元组
    rng = np.random.default_rng(0)
    # 定义数组的形状
    shape_x = (4, 5, 6)
    shape_y = (1, 6)
    # 生成具有给定形状的随机数组
    x = rng.random(shape_x)
    y = rng.random(shape_y)
    # 保存原始的 axis 值
    axis_original = axis

    # 将 axis 的元素转换为非负值
    axis = tuple([(i if i >= 0 else 3 + i) for i in axis])
    # 对 axis 进行排序
    axis = sorted(axis)

    # 检查 axis 是否包含唯一元素
    if len(set(axis)) != len(axis):
        message = "`axis` must contain only distinct elements"
        # 断言抛出 AxisError 异常，并匹配指定的消息
        with pytest.raises(AxisError, match=re.escape(message)):
            stats.mannwhitneyu(x, y, axis=axis_original)
        return

    # 检查 axis 是否超出数组维度的范围
    if axis[0] < 0 or axis[-1] > 2:
        message = "`axis` is out of bounds for array of dimension 3"
        # 断言抛出 AxisError 异常，并匹配指定的消息
        with pytest.raises(AxisError, match=re.escape(message)):
            stats.mannwhitneyu(x, y, axis=axis_original)
        return

    # 执行统计分析，使用原始的 axis 值
    res = stats.mannwhitneyu(x, y, axis=axis_original)

    # 参考行为
    not_axis = {0, 1, 2} - set(axis)  # 找出不在 `axis` 中的轴
    not_axis = next(iter(not_axis))  # 从集合中取出该轴

    # 复制数组 `y`，使其广播到与 `x` 相同的形状
    x2 = x
    shape_y_broadcasted = [1, 1, 6]
    shape_y_broadcasted[not_axis] = shape_x[not_axis]
    y2 = np.broadcast_to(y, shape_y_broadcasted)

    # 获取指定轴上的维度大小
    m = x2.shape[not_axis]
    # 移动数组 `x2` 和 `y2` 的轴，使得 `axis` 对应的轴变为 (1, 2)
    x2 = np.moveaxis(x2, axis, (1, 2))
    y2 = np.moveaxis(y2, axis, (1, 2))
    # 重新调整数组的形状，使其变为二维
    x2 = np.reshape(x2, (m, -1))
    y2 = np.reshape(y2, (m, -1))
    # 执行统计分析，使用 axis=1
    res2 = stats.mannwhitneyu(x2, y2, axis=1)

    # 断言两个结果数组是否完全相等
    np.testing.assert_array_equal(res, res2)
@pytest.mark.filterwarnings('ignore:After omitting NaNs...')
@pytest.mark.filterwarnings('ignore:One or more axis-slices of one...')
@skip_xp_invalid_arg
@pytest.mark.parametrize(
    ("weighted_fun_name, unpacker"),
    [
        ("gmean", lambda x: x),
        ("hmean", lambda x: x),
        ("pmean", lambda x: x),
        ("combine_pvalues", lambda x: (x.pvalue, x.statistic)),
    ],
)
# 定义测试函数，用于测试带有混合掩码和 NaN 权重的平均值函数
def test_mean_mixed_mask_nan_weights(weighted_fun_name, unpacker):
    # targeted test of _axis_nan_policy_factory with 2D masked sample:
    # omitting samples with masks and nan_policy='omit' are equivalent
    # also checks paired-sample sentinel value removal

    # 根据 weighted_fun_name 选择相应的加权平均函数，特别处理 'pmean'
    if weighted_fun_name == 'pmean':
        def weighted_fun(a, **kwargs):
            return stats.pmean(a, p=0.42, **kwargs)
    else:
        weighted_fun = getattr(stats, weighted_fun_name)

    # 定义一个函数 func，用于执行加权平均并解包结果
    def func(*args, **kwargs):
        return unpacker(weighted_fun(*args, **kwargs))

    # 设定数组的维度和大小
    m, n = 3, 20
    axis = -1

    # 使用随机数生成器创建随机数组 a 和 b
    rng = np.random.default_rng(6541968121)
    a = rng.uniform(size=(m, n))
    b = rng.uniform(size=(m, n))

    # 创建用于掩码的随机数组，并设置掩码条件
    mask_a1 = rng.uniform(size=(m, n)) < 0.2
    mask_a2 = rng.uniform(size=(m, n)) < 0.1
    mask_b1 = rng.uniform(size=(m, n)) < 0.15
    mask_b2 = rng.uniform(size=(m, n)) < 0.15
    mask_a1[2, :] = True

    # 复制数组 a 和 b 并根据掩码条件设置 NaN 值
    a_nans = a.copy()
    b_nans = b.copy()
    a_nans[mask_a1 | mask_a2] = np.nan
    b_nans[mask_b1 | mask_b2] = np.nan

    # 创建带有掩码的 masked_array 对象，设置相应的 NaN 值
    a_masked1 = np.ma.masked_array(a, mask=mask_a1)
    b_masked1 = np.ma.masked_array(b, mask=mask_b1)
    a_masked1[mask_a2] = np.nan
    b_masked1[mask_b2] = np.nan

    # 创建另一组带有掩码的 masked_array 对象，设置相应的 NaN 值
    a_masked2 = np.ma.masked_array(a, mask=mask_a2)
    b_masked2 = np.ma.masked_array(b, mask=mask_b2)
    a_masked2[mask_a1] = np.nan
    b_masked2[mask_b1] = np.nan

    # 创建带有组合掩码的 masked_array 对象
    a_masked3 = np.ma.masked_array(a, mask=(mask_a1 | mask_a2))
    b_masked3 = np.ma.masked_array(b, mask=(mask_b1 | mask_b2))

    # 使用 np.testing.suppress_warnings() 上下文管理器来测试警告
    with np.testing.suppress_warnings() as sup:
        message = 'invalid value encountered'
        sup.filter(RuntimeWarning, message)
        # 分别测试四种情况下的加权平均函数，并忽略 NaN 值
        res = func(a_nans, weights=b_nans, nan_policy="omit", axis=axis)
        res1 = func(a_masked1, weights=b_masked1, nan_policy="omit", axis=axis)
        res2 = func(a_masked2, weights=b_masked2, nan_policy="omit", axis=axis)
        res3 = func(a_masked3, weights=b_masked3, nan_policy="raise", axis=axis)
        res4 = func(a_masked3, weights=b_masked3, nan_policy="propagate", axis=axis)

    # 使用 np.testing.assert_array_equal() 断言，确保结果数组相等
    np.testing.assert_array_equal(res1, res)
    np.testing.assert_array_equal(res2, res)
    np.testing.assert_array_equal(res3, res)
    np.testing.assert_array_equal(res4, res)


# 测试特定情况下的异常处理
def test_raise_invalid_args_g17713():
    # other cases are handled in:
    # test_axis_nan_policy_decorated_positional_axis - multiple values for arg
    # test_axis_nan_policy_decorated_positional_args - unexpected kwd arg
    message = "got an unexpected keyword argument"
    # 使用 pytest.raises() 断言，捕获预期的 TypeError 异常
    with pytest.raises(TypeError, match=message):
        stats.gmean([1, 2, 3], invalid_arg=True)
    # 设置错误消息，用于匹配 pytest 抛出的 TypeError 异常
    message = " got multiple values for argument"
    # 使用 pytest 检测是否会抛出 TypeError 异常，并验证异常消息是否符合设定的 message
    with pytest.raises(TypeError, match=message):
        # 调用 stats 模块中的 gmean 函数，传入一个列表和一个命名参数 a=True
        stats.gmean([1, 2, 3], a=True)

    # 设置错误消息，用于匹配 pytest 抛出的 TypeError 异常
    message = "missing 1 required positional argument"
    # 使用 pytest 检测是否会抛出 TypeError 异常，并验证异常消息是否符合设定的 message
    with pytest.raises(TypeError, match=message):
        # 调用 stats 模块中的 gmean 函数，不传入任何参数
        stats.gmean()

    # 设置错误消息，用于匹配 pytest 抛出的 TypeError 异常
    message = "takes from 1 to 4 positional arguments but 5 were given"
    # 使用 pytest 检测是否会抛出 TypeError 异常，并验证异常消息是否符合设定的 message
    with pytest.raises(TypeError, match=message):
        # 调用 stats 模块中的 gmean 函数，传入一个列表，一个整数，一个类 float，一个列表和一个整数
        stats.gmean([1, 2, 3], 0, float, [1, 1, 1], 10)
@pytest.mark.parametrize('dtype', [np.int16, np.float32, np.complex128])
# 使用 pytest 的参数化装饰器，依次测试 np.int16, np.float32, np.complex128 这三种数据类型
def test_array_like_input(dtype):
    # 检查 `_axis_nan_policy` 装饰的函数能否处理可转换为数值数组的自定义容器

    class ArrLike:
        def __init__(self, x, dtype):
            # 初始化函数，保存输入数据和数据类型
            self._x = x
            self._dtype = dtype

        def __array__(self, dtype=None, copy=None):
            # 定义 __array__ 方法，将对象转换为 numpy 数组
            return np.asarray(x, dtype=self._dtype)

    x = [1]*2 + [3, 4, 5]  # 创建一个混合类型的列表 x
    res = stats.mode(ArrLike(x, dtype=dtype))  # 使用 ArrLike 类创建对象并计算众数
    assert res.mode == 1  # 断言众数为 1
    assert res.count == 2  # 断言众数出现次数为 2
```