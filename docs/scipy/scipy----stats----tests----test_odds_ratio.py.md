# `D:\src\scipysrc\scipy\scipy\stats\tests\test_odds_ratio.py`

```
import pytest  # 导入 pytest 模块，用于编写和运行测试用例
import numpy as np  # 导入 NumPy 库，并使用 np 别名
from numpy.testing import assert_equal, assert_allclose  # 导入 NumPy 测试工具中的断言函数
from .._discrete_distns import nchypergeom_fisher, hypergeom  # 从相对当前模块路径的 _discrete_distns 模块导入 nchypergeom_fisher 和 hypergeom 函数
from scipy.stats._odds_ratio import odds_ratio  # 导入 scipy 库中 _odds_ratio 模块的 odds_ratio 函数
from .data.fisher_exact_results_from_r import data  # 从当前包中的 data 子包中导入 fisher_exact_results_from_r 模块中的 data 变量

# 定义一个测试类 TestOddsRatio，用于测试 odds_ratio 函数
class TestOddsRatio:

    # 使用 pytest.mark.parametrize 装饰器，参数化测试用例，参数为 parameters 和 rresult，使用 data 变量提供的数据
    @pytest.mark.parametrize('parameters, rresult', data)
    def test_results_from_r(self, parameters, rresult):
        # 将参数中的 alternative 字符串中的点替换为破折号，以符合命名规范
        alternative = parameters.alternative.replace('.', '-')
        # 调用 odds_ratio 函数计算结果
        result = odds_ratio(parameters.table)
        
        # 根据计算结果的大小选择不同的容差值
        if result.statistic < 400:
            or_rtol = 5e-4
            ci_rtol = 2e-2
        else:
            or_rtol = 5e-2
            ci_rtol = 1e-1
        
        # 断言计算出的统计量与 R 计算结果的条件比值近似相等
        assert_allclose(result.statistic,
                        rresult.conditional_odds_ratio, rtol=or_rtol)
        
        # 使用结果对象的方法计算置信区间
        ci = result.confidence_interval(parameters.confidence_level,
                                        alternative)
        
        # 断言计算出的置信区间上下界与 R 计算结果的条件比值置信区间近似相等
        assert_allclose((ci.low, ci.high), rresult.conditional_odds_ratio_ci,
                        rtol=ci_rtol)

        # 对条件比值进行自检
        # 使用计算出的条件比值作为非集中超几何分布的非集中参数，
        # 参数为 table.sum(), table[0].sum(), table[:,0].sum() 分别为总数、好样本数和总样本数，
        # 分布的均值应等于 table[0, 0]
        cor = result.statistic
        table = np.array(parameters.table)
        total = table.sum()
        ngood = table[0].sum()
        nsample = table[:, 0].sum()
        
        # 处理非集中参数为 0 或无穷大的边界情况
        if cor == 0:
            nchg_mean = hypergeom.support(total, ngood, nsample)[0]
        elif cor == np.inf:
            nchg_mean = hypergeom.support(total, ngood, nsample)[1]
        else:
            nchg_mean = nchypergeom_fisher.mean(total, ngood, nsample, cor)
        
        # 断言计算出的均值与 table[0, 0] 近似相等
        assert_allclose(nchg_mean, table[0, 0], rtol=1e-13)

        # 检查置信区间的正确性
        alpha = 1 - parameters.confidence_level
        if alternative == 'two-sided':
            if ci.low > 0:
                sf = nchypergeom_fisher.sf(table[0, 0] - 1,
                                           total, ngood, nsample, ci.low)
                # 断言 sf 值与 alpha/2 近似相等
                assert_allclose(sf, alpha/2, rtol=1e-11)
            if np.isfinite(ci.high):
                cdf = nchypergeom_fisher.cdf(table[0, 0],
                                             total, ngood, nsample, ci.high)
                # 断言 cdf 值与 alpha/2 近似相等
                assert_allclose(cdf, alpha/2, rtol=1e-11)
        elif alternative == 'less':
            if np.isfinite(ci.high):
                cdf = nchypergeom_fisher.cdf(table[0, 0],
                                             total, ngood, nsample, ci.high)
                # 断言 cdf 值与 alpha 近似相等
                assert_allclose(cdf, alpha, rtol=1e-11)
        else:
            # alternative == 'greater'
            if ci.low > 0:
                sf = nchypergeom_fisher.sf(table[0, 0] - 1,
                                           total, ngood, nsample, ci.low)
                # 断言 sf 值与 alpha 近似相等
                assert_allclose(sf, alpha, rtol=1e-11)
    @pytest.mark.parametrize('table', [
        [[0, 0], [5, 10]],    # 定义测试参数table，包含一个2x2的列表，用于测试odds_ratio函数
        [[5, 10], [0, 0]],    # 另一个测试参数，不同的2x2列表
        [[0, 5], [0, 10]],    # 第三个测试参数，不同的2x2列表
        [[5, 0], [10, 0]],    # 第四个测试参数，不同的2x2列表
    ])
    def test_row_or_col_zero(self, table):
        result = odds_ratio(table)  # 调用odds_ratio函数计算结果
        assert_equal(result.statistic, np.nan)  # 断言结果的统计量为NaN
        ci = result.confidence_interval()  # 计算结果的置信区间
        assert_equal((ci.low, ci.high), (0, np.inf))  # 断言置信区间的下限为0，上限为正无穷大

    @pytest.mark.parametrize("case",
                             [[0.95, 'two-sided', 0.4879913, 2.635883],
                              [0.90, 'two-sided', 0.5588516, 2.301663]])
    def test_sample_odds_ratio_ci(self, case):
        # 比较样本的比率比置信区间与R包epitools中的oddsratio.wald函数的结果
        # 例如：
        # > library(epitools)
        # > table = matrix(c(10, 20, 41, 93), nrow=2, ncol=2, byrow=TRUE)
        # > result = oddsratio.wald(table)
        # > result$measure
        #           odds ratio with 95% C.I.
        # Predictor  estimate     lower    upper
        #   Exposed1 1.000000        NA       NA
        #   Exposed2 1.134146 0.4879913 2.635883

        confidence_level, alternative, ref_low, ref_high = case  # 解包测试用例中的参数
        table = [[10, 20], [41, 93]]  # 定义一个2x2的表格
        result = odds_ratio(table, kind='sample')  # 调用odds_ratio函数计算样本比率比
        assert_allclose(result.statistic, 1.134146, rtol=1e-6)  # 断言统计量的近似值
        ci = result.confidence_interval(confidence_level, alternative)  # 计算置信区间
        assert_allclose([ci.low, ci.high], [ref_low, ref_high], rtol=1e-6)  # 断言置信区间的近似下限和上限

    @pytest.mark.slow
    @pytest.mark.parametrize('alternative', ['less', 'greater', 'two-sided'])
    def test_sample_odds_ratio_one_sided_ci(self, alternative):
        # 找不到单侧置信区间的良好参考，因此增加样本量，并与条件比率比置信区间进行比较
        table = [[1000, 2000], [4100, 9300]]  # 定义一个大一点的2x2表格
        res = odds_ratio(table, kind='sample')  # 调用odds_ratio函数计算样本比率比
        ref = odds_ratio(table, kind='conditional')  # 调用odds_ratio函数计算条件比率比
        assert_allclose(res.statistic, ref.statistic, atol=1e-5)  # 断言统计量的近似值
        assert_allclose(res.confidence_interval(alternative=alternative),
                        ref.confidence_interval(alternative=alternative),
                        atol=2e-3)  # 断言置信区间的近似值

    @pytest.mark.parametrize('kind', ['sample', 'conditional'])
    @pytest.mark.parametrize('bad_table', [123, "foo", [10, 11, 12]])
    def test_invalid_table_shape(self, kind, bad_table):
        with pytest.raises(ValueError, match="Invalid shape"):  # 断言抛出特定异常和消息
            odds_ratio(bad_table, kind=kind)  # 调用odds_ratio函数

    def test_invalid_table_type(self):
        with pytest.raises(ValueError, match='must be an array of integers'):  # 断言抛出特定异常和消息
            odds_ratio([[1.0, 3.4], [5.0, 9.9]])  # 调用odds_ratio函数

    def test_negative_table_values(self):
        with pytest.raises(ValueError, match='must be nonnegative'):  # 断言抛出特定异常和消息
            odds_ratio([[1, 2], [3, -4]])  # 调用odds_ratio函数

    def test_invalid_kind(self):
        with pytest.raises(ValueError, match='`kind` must be'):  # 断言抛出特定异常和消息
            odds_ratio([[10, 20], [30, 14]], kind='magnetoreluctance')  # 调用odds_ratio函数
    # 测试无效的“alternative”参数时的情况
    def test_invalid_alternative(self):
        # 调用 odds_ratio 函数计算结果
        result = odds_ratio([[5, 10], [2, 32]])
        # 使用 pytest 的断言检查是否引发 ValueError 异常，且异常信息匹配指定的字符串
        with pytest.raises(ValueError, match='`alternative` must be'):
            # 调用 confidence_interval 方法，传入无效的 alternative 参数
            result.confidence_interval(alternative='depleneration')
    
    # 使用参数化测试检查无效的置信水平（confidence_level）时的情况
    @pytest.mark.parametrize('level', [-0.5, 1.5])
    def test_invalid_confidence_level(self, level):
        # 调用 odds_ratio 函数计算结果
        result = odds_ratio([[5, 10], [2, 32]])
        # 使用 pytest 的断言检查是否引发 ValueError 异常，且异常信息匹配指定的字符串
        with pytest.raises(ValueError, match='must be between 0 and 1'):
            # 调用 confidence_interval 方法，传入无效的 confidence_level 参数
            result.confidence_interval(confidence_level=level)
```