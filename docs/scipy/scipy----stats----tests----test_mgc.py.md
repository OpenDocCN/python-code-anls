# `D:\src\scipysrc\scipy\scipy\stats\tests\test_mgc.py`

```
import pytest  # 导入 pytest 模块

from pytest import raises as assert_raises, warns as assert_warns  # 从 pytest 模块导入 raises 和 warns 函数别名

import numpy as np  # 导入 NumPy 库
from numpy.testing import assert_approx_equal, assert_allclose, assert_equal  # 从 NumPy.testing 模块导入断言函数

from scipy.spatial.distance import cdist  # 从 SciPy 库导入距离计算函数 cdist
from scipy import stats  # 导入 SciPy 库中的统计模块

class TestMGCErrorWarnings:
    """ Tests errors and warnings derived from MGC.
    """
    def test_error_notndarray(self):
        # 如果 x 或 y 不是 ndarray，则引发 ValueError 错误
        x = np.arange(20)
        y = [5] * 20
        assert_raises(ValueError, stats.multiscale_graphcorr, x, y)
        assert_raises(ValueError, stats.multiscale_graphcorr, y, x)

    def test_error_shape(self):
        # 如果样本数不同（n），则引发 ValueError 错误
        x = np.arange(100).reshape(25, 4)
        y = x.reshape(10, 10)
        assert_raises(ValueError, stats.multiscale_graphcorr, x, y)

    def test_error_lowsamples(self):
        # 如果样本数过少（< 3），则引发 ValueError 错误
        x = np.arange(3)
        y = np.arange(3)
        assert_raises(ValueError, stats.multiscale_graphcorr, x, y)

    def test_error_nans(self):
        # 如果输入包含 NaN，则引发 ValueError 错误
        x = np.arange(20, dtype=float)
        x[0] = np.nan
        assert_raises(ValueError, stats.multiscale_graphcorr, x, x)

        y = np.arange(20)
        assert_raises(ValueError, stats.multiscale_graphcorr, x, y)

    def test_error_wrongdisttype(self):
        # 如果度量标准不是函数，则引发 ValueError 错误
        x = np.arange(20)
        compute_distance = 0
        assert_raises(ValueError, stats.multiscale_graphcorr, x, x,
                      compute_distance=compute_distance)

    @pytest.mark.parametrize("reps", [
        -1,    # reps 为负数
        '1',   # reps 不是整数
    ])
    def test_error_reps(self, reps):
        # 如果 reps 为负数，则引发 ValueError 错误
        x = np.arange(20)
        assert_raises(ValueError, stats.multiscale_graphcorr, x, x, reps=reps)

    def test_warns_reps(self):
        # 当 reps 小于 1000 时，引发 RuntimeWarning 警告
        x = np.arange(20)
        reps = 100
        assert_warns(RuntimeWarning, stats.multiscale_graphcorr, x, x, reps=reps)

    def test_error_infty(self):
        # 如果输入包含无穷大，则引发 ValueError 错误
        x = np.arange(20)
        y = np.ones(20) * np.inf
        assert_raises(ValueError, stats.multiscale_graphcorr, x, y)


class TestMGCStat:
    """ Test validity of MGC test statistic
    """
    def _simulations(self, samps=100, dims=1, sim_type=""):
        # 定义一个函数 `_simulations`，用于生成模拟数据
        # 参数:
        # - samps: 样本数量，默认为100
        # - dims: 数据维度，默认为1
        # - sim_type: 模拟类型，可以是 "linear"、"nonlinear" 或 "independence"
        
        # linear simulation
        if sim_type == "linear":
            # 如果 sim_type 是 "linear"，生成线性模拟数据
            x = np.random.uniform(-1, 1, size=(samps, 1))
            y = x + 0.3 * np.random.random_sample(size=(x.size, 1))

        # spiral simulation
        elif sim_type == "nonlinear":
            # 如果 sim_type 是 "nonlinear"，生成非线性（螺旋形）模拟数据
            unif = np.array(np.random.uniform(0, 5, size=(samps, 1)))
            x = unif * np.cos(np.pi * unif)
            y = (unif * np.sin(np.pi * unif) +
                 0.4*np.random.random_sample(size=(x.size, 1)))

        # independence (tests type I simulation)
        elif sim_type == "independence":
            # 如果 sim_type 是 "independence"，生成独立模拟数据
            u = np.random.normal(0, 1, size=(samps, 1))
            v = np.random.normal(0, 1, size=(samps, 1))
            u_2 = np.random.binomial(1, p=0.5, size=(samps, 1))
            v_2 = np.random.binomial(1, p=0.5, size=(samps, 1))
            x = u/3 + 2*u_2 - 1
            y = v/3 + 2*v_2 - 1

        # raises error if not approved sim_type
        else:
            # 如果 sim_type 不是 "linear"、"nonlinear" 或 "independence"，抛出 ValueError
            raise ValueError("sim_type must be linear, nonlinear, or "
                             "independence")

        # add dimensions of noise for higher dimensions
        # 如果 dims 大于1，为高维数据添加噪声维度
        if dims > 1:
            dims_noise = np.random.normal(0, 1, size=(samps, dims-1))
            x = np.concatenate((x, dims_noise), axis=1)

        # 返回生成的数据 x 和 y
        return x, y

    @pytest.mark.xslow
    @pytest.mark.parametrize("sim_type, obs_stat, obs_pvalue", [
        ("linear", 0.97, 1/1000),           # 测试线性模拟数据
        ("nonlinear", 0.163, 1/1000),       # 测试螺旋模拟数据
        ("independence", -0.0094, 0.78)     # 测试独立模拟数据
    ])
    def test_oned(self, sim_type, obs_stat, obs_pvalue):
        np.random.seed(12345678)

        # generate x and y
        # 生成 x 和 y 数据，调用 _simulations 函数生成对应 sim_type 的模拟数据
        x, y = self._simulations(samps=100, dims=1, sim_type=sim_type)

        # test stat and pvalue
        # 测试统计量和 p 值
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y)
        assert_approx_equal(stat, obs_stat, significant=1)
        assert_approx_equal(pvalue, obs_pvalue, significant=1)

    @pytest.mark.xslow
    @pytest.mark.parametrize("sim_type, obs_stat, obs_pvalue", [
        ("linear", 0.184, 1/1000),           # 测试线性模拟数据
        ("nonlinear", 0.0190, 0.117),        # 测试螺旋模拟数据
    ])
    def test_fived(self, sim_type, obs_stat, obs_pvalue):
        np.random.seed(12345678)

        # generate x and y
        # 生成 x 和 y 数据，调用 _simulations 函数生成对应 sim_type 的模拟数据
        x, y = self._simulations(samps=100, dims=5, sim_type=sim_type)

        # test stat and pvalue
        # 测试统计量和 p 值
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y)
        assert_approx_equal(stat, obs_stat, significant=1)
        assert_approx_equal(pvalue, obs_pvalue, significant=1)

    @pytest.mark.xslow
    def test_twosamp(self):
        # 设定随机种子，确保结果可复现
        np.random.seed(12345678)

        # 生成随机变量 x 和 y
        x = np.random.binomial(100, 0.5, size=(100, 5))
        y = np.random.normal(0, 1, size=(80, 5))

        # 计算统计量 stat 和 p-value
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y)
        # 断言 stat 的近似值为 1.0，精确到一位有效数字
        assert_approx_equal(stat, 1.0, significant=1)
        # 断言 pvalue 的近似值为 0.001，精确到一位有效数字
        assert_approx_equal(pvalue, 0.001, significant=1)

        # 重新生成随机变量 y
        y = np.random.normal(0, 1, size=(100, 5))

        # 再次计算统计量 stat 和 p-value，使用 is_twosamp=True
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y, is_twosamp=True)
        # 断言 stat 的近似值为 1.0，精确到一位有效数字
        assert_approx_equal(stat, 1.0, significant=1)
        # 断言 pvalue 的近似值为 0.001，精确到一位有效数字
        assert_approx_equal(pvalue, 0.001, significant=1)

    @pytest.mark.xslow
    def test_workers(self):
        # 设定随机种子，确保结果可复现
        np.random.seed(12345678)

        # 使用 self._simulations 生成随机变量 x 和 y
        x, y = self._simulations(samps=100, dims=1, sim_type="linear")

        # 计算统计量 stat 和 p-value，设置 workers=2
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y, workers=2)
        # 断言 stat 的近似值为 0.97，精确到一位有效数字
        assert_approx_equal(stat, 0.97, significant=1)
        # 断言 pvalue 的近似值为 0.001，精确到一位有效数字
        assert_approx_equal(pvalue, 0.001, significant=1)

    @pytest.mark.xslow
    def test_random_state(self):
        # 生成随机变量 x 和 y
        x, y = self._simulations(samps=100, dims=1, sim_type="linear")

        # 计算统计量 stat 和 p-value，设置 random_state=1
        stat, pvalue, _ = stats.multiscale_graphcorr(x, y, random_state=1)
        # 断言 stat 的近似值为 0.97，精确到一位有效数字
        assert_approx_equal(stat, 0.97, significant=1)
        # 断言 pvalue 的近似值为 0.001，精确到一位有效数字
        assert_approx_equal(pvalue, 0.001, significant=1)

    @pytest.mark.xslow
    def test_dist_perm(self):
        # 设定随机种子，确保结果可复现
        np.random.seed(12345678)

        # 使用 self._simulations 生成随机变量 x 和 y，sim_type="nonlinear"
        x, y = self._simulations(samps=100, dims=1, sim_type="nonlinear")
        # 计算 x 和 y 的欧氏距离矩阵
        distx = cdist(x, x, metric="euclidean")
        disty = cdist(y, y, metric="euclidean")

        # 计算统计量 stat_dist 和 p-value pvalue_dist，设置 random_state=1
        stat_dist, pvalue_dist, _ = stats.multiscale_graphcorr(distx, disty,
                                                               compute_distance=None,
                                                               random_state=1)
        # 断言 stat_dist 的近似值为 0.163，精确到一位有效数字
        assert_approx_equal(stat_dist, 0.163, significant=1)
        # 断言 pvalue_dist 的近似值为 0.001，精确到一位有效数字
        assert_approx_equal(pvalue_dist, 0.001, significant=1)

    @pytest.mark.fail_slow(20)  # 所有其它测试都是 XSLOW；至少需要运行一个
    @pytest.mark.slow
    def test_pvalue_literature(self):
        # 设定随机种子，确保结果可复现
        np.random.seed(12345678)

        # 使用 self._simulations 生成随机变量 x 和 y，sim_type="linear"
        x, y = self._simulations(samps=100, dims=1, sim_type="linear")

        # 计算统计量和 p-value
        _, pvalue, _ = stats.multiscale_graphcorr(x, y, random_state=1)
        # 断言 pvalue 的值接近于 1/1001
        assert_allclose(pvalue, 1/1001)

    @pytest.mark.xslow
    def test_alias(self):
        # 设定随机种子，确保结果可复现
        np.random.seed(12345678)

        # 使用 self._simulations 生成随机变量 x 和 y，sim_type="linear"
        x, y = self._simulations(samps=100, dims=1, sim_type="linear")

        # 计算统计量和 p-value，设置 random_state=1
        res = stats.multiscale_graphcorr(x, y, random_state=1)
        # 断言 res.stat 等于 res.statistic
        assert_equal(res.stat, res.statistic)
```