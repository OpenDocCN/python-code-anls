# `D:\src\scipysrc\scipy\scipy\stats\tests\test_multicomp.py`

```
import copy  # 导入 copy 模块，用于复制对象

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 模块，用于编写和运行测试用例
from numpy.testing import assert_allclose  # 导入 NumPy 测试模块的函数 assert_allclose

from scipy import stats  # 导入 SciPy 统计模块
from scipy.stats._multicomp import _pvalue_dunnett, DunnettResult  # 导入 Dunnett 相关的统计函数和结果类


class TestDunnett:
    # 对于下面的测试，p 值是使用 Matlab 计算得到的，例如：
    #     sample = [18.  15.  18.  16.  17.  15.  14.  14.  14.  15.  15....
    #               14.  15.  14.  22.  18.  21.  21.  10.  10.  11.  9....
    #               25.  26.  17.5 16.  15.5 14.5 22.  22.  24.  22.5 29....
    #               24.5 20.  18.  18.5 17.5 26.5 13.  16.5 13.  13.  13....
    #               28.  27.  34.  31.  29.  27.  24.  23.  38.  36.  25....
    #               38. 26.  22.  36.  27.  27.  32.  28.  31....
    #               24.  27.  33.  32.  28.  19. 37.  31.  36.  36....
    #               34.  38.  32.  38.  32....
    #               26.  24.  26.  25.  29. 29.5 16.5 36.  44....
    #               25.  27.  19....
    #               25.  20....
    #               28.];
    #     j = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
    #          0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...
    #          0 0 0 0...
    #          1 1 1 1 1 1 1 1 1 1 1 1 1 1 1...
    #          2 2 2 2 2 2 2 2 2...
    #          3 3 3...
    #          4 4...
    #          5];
    #     [~, ~, stats] = anova1(sample, j, "off");
    #     [results, ~, ~, gnames] = multcompare(stats, ...
    #     "CriticalValueType", "dunnett", ...
    #     "Approximate", false);
    #     tbl = array2table(results, "VariableNames", ...
    #     ["Group", "Control Group", "Lower Limit", ...
    #     "Difference", "Upper Limit", "P-value"]);
    #     tbl.("Group") = gnames(tbl.("Group"));
    #     tbl.("Control Group") = gnames(tbl.("Control Group"))

    # Matlab 并未报告统计值，因此统计值是使用 R 的 multcomp 中的 `glht` 计算的，例如：
    #     library(multcomp)
    #     options(digits=16)
    #     control < - c(18.0, 15.0, 18.0, 16.0, 17.0, 15.0, 14.0, 14.0, 14.0,
    #                   15.0, 15.0, 14.0, 15.0, 14.0, 22.0, 18.0, 21.0, 21.0,
    #                   10.0, 10.0, 11.0, 9.0, 25.0, 26.0, 17.5, 16.0, 15.5,
    #                   14.5, 22.0, 22.0, 24.0, 22.5, 29.0, 24.5, 20.0, 18.0,
    #                   18.5, 17.5, 26.5, 13.0, 16.5, 13.0, 13.0, 13.0, 28.0,
    #                   27.0, 34.0, 31.0, 29.0, 27.0, 24.0, 23.0, 38.0, 36.0,
    #                   25.0, 38.0, 26.0, 22.0, 36.0, 27.0, 27.0, 32.0, 28.0,
    #                   31.0)
    #     t < - c(24.0, 27.0, 33.0, 32.0, 28.0, 19.0, 37.0, 31.0, 36.0, 36.0,
    #             34.0, 38.0, 32.0, 38.0, 32.0)
    #     w < - c(26.0, 24.0, 26.0, 25.0, 29.0, 29.5, 16.5, 36.0, 44.0)
    #     x < - c(25.0, 27.0, 19.0)
    #     y < - c(25.0, 20.0)
    #     z < - c(28.0)
    #
    #     groups = factor(rep(c("control", "t", "w", "x", "y", "z"),
    #                         times=c(length(control), length(t), length(w),
    #                                 length(x), length(y), length(z))))
    #
    # 以上代码段展示了使用 Dunnett 方法进行多重比较分析的设置和准备工作
    # 包括样本数据的设定和统计值的计算
    pass  # 空的测试类，用于展示 Dunnett 方法的设置和准备工作
    # 样本数据集 1，包含多个组的数据
    samples_1 = [
        [
            24.0, 27.0, 33.0, 32.0, 28.0, 19.0, 37.0, 31.0, 36.0, 36.0,
            34.0, 38.0, 32.0, 38.0, 32.0
        ],
        [26.0, 24.0, 26.0, 25.0, 29.0, 29.5, 16.5, 36.0, 44.0],
        [25.0, 27.0, 19.0],
        [25.0, 20.0],
        [28.0]
    ]
    # 控制组的数据集 1
    control_1 = [
        18.0, 15.0, 18.0, 16.0, 17.0, 15.0, 14.0, 14.0, 14.0, 15.0, 15.0,
        14.0, 15.0, 14.0, 22.0, 18.0, 21.0, 21.0, 10.0, 10.0, 11.0, 9.0,
        25.0, 26.0, 17.5, 16.0, 15.5, 14.5, 22.0, 22.0, 24.0, 22.5, 29.0,
        24.5, 20.0, 18.0, 18.5, 17.5, 26.5, 13.0, 16.5, 13.0, 13.0, 13.0,
        28.0, 27.0, 34.0, 31.0, 29.0, 27.0, 24.0, 23.0, 38.0, 36.0, 25.0,
        38.0, 26.0, 22.0, 36.0, 27.0, 27.0, 32.0, 28.0, 31.0
    ]
    # P 值数据集 1，与 Matlab 计算结果一致
    pvalue_1 = [4.727e-06, 0.022346, 0.97912, 0.99953, 0.86579]  # Matlab
    # 使用 R 的 multcomp 中 glht 计算得到的双侧 P 值数据集 1
    p_1_twosided = [1e-4, 0.02237, 0.97913, 0.99953, 0.86583]
    # 使用 R 的 multcomp 中 glht 计算得到的大于型 P 值数据集 1
    p_1_greater = [1e-4, 0.011217, 0.768500, 0.896991, 0.577211]
    # 使用 R 的 multcomp 中 glht 计算得到的小于型 P 值数据集 1
    p_1_less = [1, 1, 0.99660, 0.98398, .99953]
    # 使用 R 的 multcomp 中 glht 计算得到的统计量数据集 1
    statistic_1 = [5.27356, 2.91270, 0.60831, 0.27002, 0.96637]
    # 使用 R 的 multcomp 中 glht 计算得到的双侧置信区间数据集 1
    ci_1_twosided = [[5.3633917835622, 0.7296142201217, -8.3879817106607,
                      -11.9090753452911, -11.7655021543469],
                     [15.9709832164378, 13.8936496687672, 13.4556900439941,
                      14.6434503452911, 25.4998771543469]]
    # 使用 R 的 multcomp 中 glht 计算得到的大于型置信区间数据集 1
    ci_1_greater = [5.9036402398526, 1.4000632918725, -7.2754756323636,
                    -10.5567456382391, -9.8675629499576]
    # 使用 R 的 multcomp 中 glht 计算得到的小于型置信区间数据集 1
    ci_1_less = [15.4306165948619, 13.2230539537359, 12.3429406339544,
                 13.2908248513211, 23.6015228251660]
    # 包含样本数据、控制组数据、统计量、多种 P 值及置信区间的数据集 1
    case_1 = dict(samples=samples_1, control=control_1, statistic=statistic_1,
                  pvalues=dict(twosided=p_1_twosided, less=p_1_less, greater=p_1_greater),
                  cis=dict(twosided=ci_1_twosided, less=ci_1_less, greater=ci_1_greater))

    # Dunnett1955 文献中的比较数据集，用于与 R 的 DescTools 中的 DunnettTest 进行比较
    samples_2 = [[9.76, 8.80, 7.68, 9.36], [12.80, 9.68, 12.16, 9.20, 10.55]]
    # 控制组的数据集 2
    control_2 = [7.40, 8.50, 7.20, 8.24, 9.84, 8.32]
    # P 值数据集 2，与 R multcomp 中 glht 计算结果相关
    pvalue_2 = [0.6201, 0.0058]
    # 使用 R 的 multcomp 中 glht 计算得到的双侧 P 值数据集 2
    p_2_twosided = [0.6201020, 0.0058254]
    # 使用 R 的 multcomp 中 glht 计算得到的大于型 P 值数据集 2
    p_2_greater = [0.3249776, 0.0029139]
    # 使用 R 的 multcomp 中 glht 计算得到的小于型 P 值数据集 2
    p_2_less = [0.91676, 0.99984]
    # 使用 R 的 multcomp 中 glht 计算得到的统计量数据集 2
    statistic_2 = [0.85703, 3.69375]
    # 创建一个二维列表，表示第二个案例的双侧置信区间
    ci_2_twosided = [[-1.2564116462124, 0.8396273539789],
                     [2.5564116462124, 4.4163726460211]]
    # 创建一个列表，表示第二个案例的大于置信区间
    ci_2_greater = [-0.9588591188156, 1.1187563667543]
    # 创建一个列表，表示第二个案例的小于置信区间
    ci_2_less = [2.2588591188156, 4.1372436332457]
    # 创建一个字典，包含第二个案例的统计结果的不同假设检验结果
    pvalues_2 = dict(twosided=p_2_twosided, less=p_2_less, greater=p_2_greater)
    # 创建一个字典，包含第二个案例的置信区间的不同假设检验结果
    cis_2 = dict(twosided=ci_2_twosided, less=ci_2_less, greater=ci_2_greater)
    # 创建一个字典，包含第二个案例的各项数据，包括样本数据、对照数据、统计量、假设检验结果和置信区间
    case_2 = dict(samples=samples_2, control=control_2, statistic=statistic_2,
                  pvalues=pvalues_2, cis=cis_2)
    
    # 创建一个二维列表，表示第三个案例的样本数据
    samples_3 = [[55, 64, 64], [55, 49, 52], [50, 44, 41]]
    # 创建一个列表，表示第三个案例的对照数据
    control_3 = [55, 47, 48]
    # 创建一个列表，表示第三个案例的 p 值
    pvalue_3 = [0.0364, 0.8966, 0.4091]
    # 创建一个列表，表示第三个案例的双侧置信区间
    ci_3_twosided = [[0.7529028025053, -8.2470971974947, -15.2470971974947],
                     [21.2470971974947, 12.2470971974947, 5.2470971974947]]
    # 创建一个列表，表示第三个案例的大于置信区间
    ci_3_greater = [2.4023682323149, -6.5976317676851, -13.5976317676851]
    # 创建一个列表，表示第三个案例的小于置信区间
    ci_3_less = [19.5984402363662, 10.5984402363662, 3.5984402363662]
    # 创建一个字典，包含第三个案例的统计结果的不同假设检验结果
    pvalues_3 = dict(twosided=p_3_twosided, less=p_3_less, greater=p_3_greater)
    # 创建一个字典，包含第三个案例的置信区间的不同假设检验结果
    cis_3 = dict(twosided=ci_3_twosided, less=ci_3_less, greater=ci_3_greater)
    # 创建一个字典，包含第三个案例的各项数据，包括样本数据、对照数据、统计量、假设检验结果和置信区间
    case_3 = dict(samples=samples_3, control=control_3, statistic=statistic_3,
                  pvalues=pvalues_3, cis=cis_3)
    
    # 创建一个二维列表，表示第四个案例的样本数据
    samples_4 = [[3.8, 2.7, 4.0, 2.4], [2.8, 3.4, 3.7, 2.2, 2.0]]
    # 创建一个列表，表示第四个案例的对照数据
    control_4 = [2.9, 3.0, 2.5, 2.6, 3.2]
    # 创建一个列表，表示第四个案例的 p 值
    pvalue_4 = [0.5832, 0.9982]
    # 创建一个列表，表示第四个案例的双侧置信区间
    ci_4_twosided = [[-0.6898153448579, -1.0333456251632],
                     [1.4598153448579, 0.9933456251632]]
    # 创建一个列表，表示第四个案例的大于置信区间
    ci_4_greater = [-0.5186459268412, -0.8719655502147 ]
    # 创建一个列表，表示第四个案例的小于置信区间
    ci_4_less = [1.2886459268412, 0.8319655502147]
    # 创建一个字典，包含第四个案例的统计结果的不同假设检验结果
    pvalues_4 = dict(twosided=p_4_twosided, less=p_4_less, greater=p_4_greater)
    # 创建一个字典，包含第四个案例的置信区间的不同假设检验结果
    cis_4 = dict(twosided=ci_4_twosided, less=ci_4_less, greater=ci_4_greater)
    # 创建一个字典，包含第四个案例的各项数据，包括样本数据、对照数据、统计量、假设检验结果和置信区间
    case_4 = dict(samples=samples_4, control=control_4, statistic=statistic_4,
                  pvalues=pvalues_4, cis=cis_4)
    @pytest.mark.parametrize(
        'rho, n_groups, df, statistic, pvalue, alternative',
        [
            # 从 Dunnett1955 表1a和1b页1117-1118获取的数据
            # 参数: 相关系数 rho, 组数 n_groups, 自由度 df, 统计量 statistic, p 值 pvalue, 检验类型 alternative
            (0.5, 1, 10, 1.81, 0.05, "greater"),  # 单边检验，显著性水平为0.05
            (0.5, 3, 10, 2.34, 0.05, "greater"),
            (0.5, 2, 30, 1.99, 0.05, "greater"),
            (0.5, 5, 30, 2.33, 0.05, "greater"),
            (0.5, 4, 12, 3.32, 0.01, "greater"),
            (0.5, 7, 12, 3.56, 0.01, "greater"),
            (0.5, 2, 60, 2.64, 0.01, "greater"),
            (0.5, 4, 60, 2.87, 0.01, "greater"),
            (0.5, 4, 60, [2.87, 2.21], [0.01, 0.05], "greater"),  # 多个检验情况
            # 从 Dunnett1955 表2a和2b页1119-1120获取的数据
            # 参数: 相关系数 rho, 组数 n_groups, 自由度 df, 统计量 statistic, p 值 pvalue, 检验类型 alternative
            (0.5, 1, 10, 2.23, 0.05, "two-sided"),  # 双边检验，显著性水平为0.05
            (0.5, 3, 10, 2.81, 0.05, "two-sided"),
            (0.5, 2, 30, 2.32, 0.05, "two-sided"),
            (0.5, 3, 20, 2.57, 0.05, "two-sided"),
            (0.5, 4, 12, 3.76, 0.01, "two-sided"),
            (0.5, 7, 12, 4.08, 0.01, "two-sided"),
            (0.5, 2, 60, 2.90, 0.01, "two-sided"),
            (0.5, 4, 60, 3.14, 0.01, "two-sided"),
            (0.5, 4, 60, [3.14, 2.55], [0.01, 0.05], "two-sided"),  # 多个检验情况
        ],
    )
    # 测试临界值的方法
    def test_critical_values(
        self, rho, n_groups, df, statistic, pvalue, alternative
    ):
        # 使用固定种子创建随机数生成器
        rng = np.random.default_rng(165250594791731684851746311027739134893)
        # 创建相关系数矩阵，将对角线设置为1
        rho = np.full((n_groups, n_groups), rho)
        np.fill_diagonal(rho, 1)

        # 转换 statistic 为 numpy 数组
        statistic = np.array(statistic)
        # 进行 Dunnett 检验，计算 p 值
        res = _pvalue_dunnett(
            rho=rho, df=df, statistic=statistic,
            alternative=alternative,
            rng=rng
        )
        # 断言检验结果的 p 值接近期望值，允许的误差为 5e-3
        assert_allclose(res, pvalue, atol=5e-3)

    @pytest.mark.parametrize(
        'samples, control, pvalue, statistic',
        [
            # 各种样本集和对应的控制组，以及预期的 p 值和统计量
            (samples_1, control_1, pvalue_1, statistic_1),
            (samples_2, control_2, pvalue_2, statistic_2),
            (samples_3, control_3, pvalue_3, statistic_3),
            (samples_4, control_4, pvalue_4, statistic_4),
        ]
    )
    # 测试基本的 Dunnett 检验
    def test_basic(self, samples, control, pvalue, statistic):
        # 使用固定种子创建随机数生成器
        rng = np.random.default_rng(11681140010308601919115036826969764808)

        # 进行 Dunnett 检验，计算结果
        res = stats.dunnett(*samples, control=control, random_state=rng)

        # 断言返回结果类型为 DunnettResult 类型
        assert isinstance(res, DunnettResult)
        # 断言统计量接近期望值，允许的相对误差为 5e-5
        assert_allclose(res.statistic, statistic, rtol=5e-5)
        # 断言 p 值接近期望值，允许的相对误差为 1e-2，绝对误差为 1e-4
        assert_allclose(res.pvalue, pvalue, rtol=1e-2, atol=1e-4)

    @pytest.mark.parametrize(
        'alternative',
        ['two-sided', 'less', 'greater']
    )
    # 测试不同类型的检验假设
    def test_alternative(self, alternative):
        # 测试代码逻辑未提供
        pass
    def test_ttest_ind(self, alternative):
        # 检查 `dunnett` 在只有两组时与 `ttest_ind` 的一致性
        # 使用指定种子生成随机数生成器
        rng = np.random.default_rng(114184017807316971636137493526995620351)

        # 进行10次测试
        for _ in range(10):
            # 生成范围在[-100, 100]之间的大小为10的整数随机样本和对照组
            sample = rng.integers(-100, 100, size=(10,))
            control = rng.integers(-100, 100, size=(10,))

            # 使用 `dunnett` 函数计算结果
            res = stats.dunnett(
                sample, control=control,
                alternative=alternative, random_state=rng
            )
            # 使用 `ttest_ind` 函数计算结果作为参考
            ref = stats.ttest_ind(
                sample, control,
                alternative=alternative, random_state=rng
            )

            # 断言 `dunnett` 和 `ttest_ind` 的统计量和 p 值接近
            assert_allclose(res.statistic, ref.statistic, rtol=1e-3, atol=1e-5)
            assert_allclose(res.pvalue, ref.pvalue, rtol=1e-3, atol=1e-5)

    @pytest.mark.parametrize(
        'alternative, pvalue',
        [
            ('less', [0, 1]),     # 当备择假设为 'less' 时，预期的 p 值
            ('greater', [1, 0]),  # 当备择假设为 'greater' 时，预期的 p 值
            ('two-sided', [0, 0]),# 当备择假设为 'two-sided' 时，预期的 p 值
        ]
    )
    def test_alternatives(self, alternative, pvalue):
        # 使用指定种子生成随机数生成器
        rng = np.random.default_rng(114184017807316971636137493526995620351)

        # 生成不同备择假设下的随机样本和对照组
        # 'less' 备择假设
        sample_less = rng.integers(0, 20, size=(10,))
        control = rng.integers(80, 100, size=(10,))
        # 'greater' 备择假设
        sample_greater = rng.integers(160, 180, size=(10,))

        # 使用 `dunnett` 函数计算结果
        res = stats.dunnett(
            sample_less, sample_greater, control=control,
            alternative=alternative, random_state=rng
        )
        # 断言 `dunnett` 计算得到的 p 值与预期的 p 值接近
        assert_allclose(res.pvalue, pvalue, atol=1e-7)

        # 计算置信区间
        ci = res.confidence_interval()
        # 根据备择假设不同进行断言置信区间的范围
        if alternative == 'less':
            assert np.isneginf(ci.low).all()  # 置信区间下限为负无穷
            assert -100 < ci.high[0] < -60     # 第一个置信区间上限的范围
            assert 60 < ci.high[1] < 100       # 第二个置信区间上限的范围
        elif alternative == 'greater':
            assert -100 < ci.low[0] < -60      # 第一个置信区间下限的范围
            assert 60 < ci.low[1] < 100        # 第二个置信区间下限的范围
            assert np.isposinf(ci.high).all()  # 置信区间上限为正无穷
        elif alternative == 'two-sided':
            assert -100 < ci.low[0] < -60      # 第一个置信区间下限的范围
            assert 60 < ci.low[1] < 100        # 第二个置信区间下限的范围
            assert -100 < ci.high[0] < -60     # 第一个置信区间上限的范围
            assert 60 < ci.high[1] < 100       # 第二个置信区间上限的范围

    @pytest.mark.parametrize("case", [case_1, case_2, case_3, case_4])
    @pytest.mark.parametrize("alternative", ['less', 'greater', 'two-sided'])
    #`
    # 定义一个测试方法，用于针对多重比较的情况进行测试
    def test_against_R_multicomp_glht(self, case, alternative):
        # 设置随机数生成器的种子，以便复现结果
        rng = np.random.default_rng(189117774084579816190295271136455278291)
        # 从测试案例中获取样本数据和对照组数据
        samples = case['samples']
        control = case['control']
        # 定义备择假设的集合，以备后续检索
        alternatives = {'less': 'less', 'greater': 'greater',
                        'two-sided': 'twosided'}
        # 获取参考 p 值，用于后续断言比较
        p_ref = case['pvalues'][alternative.replace('-', '')]

        # 运行 Dunnett's 测试，计算结果
        res = stats.dunnett(*samples, control=control, alternative=alternative,
                            random_state=rng)
        
        # 断言实际计算得到的 p 值与参考 p 值接近，设置公差为 5e-3 和绝对公差为 1e-4
        assert_allclose(res.pvalue, p_ref, rtol=5e-3, atol=1e-4)

        # 获取参考的置信区间，用于后续断言比较
        ci_ref = case['cis'][alternatives[alternative]]
        # 根据不同的备择假设类型，调整参考的置信区间
        if alternative == "greater":
            ci_ref = [ci_ref, np.inf]
        elif alternative == "less":
            ci_ref = [-np.inf, ci_ref]
        # 断言结果对象中的置信区间信息未定义
        assert res._ci is None
        assert res._ci_cl is None
        # 计算实际的置信区间
        ci = res.confidence_interval(confidence_level=0.95)
        # 断言实际计算得到的置信区间的下界和上界与参考值接近，设置公差为 5e-3 和绝对公差为 1e-5
        assert_allclose(ci.low, ci_ref[0], rtol=5e-3, atol=1e-5)
        assert_allclose(ci.high, ci_ref[1], rtol=5e-3, atol=1e-5)

        # 重新运行以使用缓存的值 "ci" 来检查 id 是否相同
        assert res._ci is ci
        assert res._ci_cl == 0.95
        # 再次计算置信区间，应该返回相同的对象 "ci"
        ci_ = res.confidence_interval(confidence_level=0.95)
        assert ci_ is ci

    # 使用 pytest 的参数化功能，定义多个测试用例来测试不同的备择假设
    @pytest.mark.parametrize('alternative', ["two-sided", "less", "greater"])
    def test_str(self, alternative):
        # 设置随机数生成器的种子，以便复现结果
        rng = np.random.default_rng(189117774084579816190295271136455278291)

        # 运行 Dunnett's 测试，计算结果
        res = stats.dunnett(
            *self.samples_3, control=self.control_3, alternative=alternative,
            random_state=rng
        )

        # 检查一些字符串输出是否包含特定内容
        res_str = str(res)
        assert '(Sample 2 - Control)' in res_str
        assert '95.0%' in res_str

        # 根据不同的备择假设类型，检查字符串输出中特定的值
        if alternative == 'less':
            assert '-inf' in res_str
            assert '19.' in res_str
        elif alternative == 'greater':
            assert 'inf' in res_str
            assert '-13.' in res_str
        else:
            assert 'inf' not in res_str
            assert '21.' in res_str

    # 定义一个测试方法，用于测试警告信息是否正确抛出
    def test_warnings(self):
        # 设置随机数生成器的种子，以便复现结果
        rng = np.random.default_rng(189117774084579816190295271136455278291)

        # 运行 Dunnett's 测试，计算结果
        res = stats.dunnett(
            *self.samples_3, control=self.control_3, random_state=rng
        )
        # 设置预期的警告信息内容
        msg = r"Computation of the confidence interval did not converge"
        # 使用 pytest 的上下文管理器，检查是否抛出了预期的 UserWarning
        with pytest.warns(UserWarning, match=msg):
            res._allowance(tol=1e-5)
    # 定义一个测试方法，用于测试异常情况是否会引发异常

    samples, control = self.samples_3, self.control_3

    # 测试当 alternative 参数为 'bob' 时是否会引发 ValueError 异常
    with pytest.raises(ValueError, match="alternative must be"):
        stats.dunnett(*samples, control=control, alternative='bob')

    # 测试当样本中有二维数组时是否会引发 ValueError 异常
    samples_ = copy.deepcopy(samples)
    samples_[0] = [samples_[0]]
    with pytest.raises(ValueError, match="must be 1D arrays"):
        stats.dunnett(*samples_, control=control)

    # 测试当对照组为二维数组时是否会引发 ValueError 异常
    control_ = copy.deepcopy(control)
    control_ = [control_]
    with pytest.raises(ValueError, match="must be 1D arrays"):
        stats.dunnett(*samples, control=control_)

    # 测试当样本中某一个组没有观测值时是否会引发 ValueError 异常
    samples_ = copy.deepcopy(samples)
    samples_[1] = []
    with pytest.raises(ValueError, match="at least 1 observation"):
        stats.dunnett(*samples_, control=control)

    # 测试当对照组没有观测值时是否会引发 ValueError 异常
    control_ = []
    with pytest.raises(ValueError, match="at least 1 observation"):
        stats.dunnett(*samples, control=control_)

    # 执行统计分析，并测试当置信水平设定错误时是否会引发 ValueError 异常
    res = stats.dunnett(*samples, control=control)
    with pytest.raises(ValueError, match="Confidence level must"):
        res.confidence_interval(confidence_level=3)


@pytest.mark.filterwarnings("ignore:Computation of the confidence")
@pytest.mark.parametrize('n_samples', [1, 2, 3])
def test_shapes(self, n_samples):
    # 定义一个测试方法，用于测试输出结果的形状是否符合预期

    rng = np.random.default_rng(689448934110805334)
    # 生成 n_samples 个样本和一个对照组的正态分布随机数
    samples = rng.normal(size=(n_samples, 10))
    control = rng.normal(size=10)

    # 进行 Dunnett 检验，并获取结果对象 res
    res = stats.dunnett(*samples, control=control, random_state=rng)

    # 断言统计量 statistic 的形状应为 (n_samples,)
    assert res.statistic.shape == (n_samples,)
    
    # 断言 p 值 pvalue 的形状应为 (n_samples,)
    assert res.pvalue.shape == (n_samples,)
    
    # 获取置信区间 ci
    ci = res.confidence_interval()
    
    # 断言置信区间下界 ci.low 的形状应为 (n_samples,)
    assert ci.low.shape == (n_samples,)
    
    # 断言置信区间上界 ci.high 的形状应为 (n_samples,)
    assert ci.high.shape == (n_samples,)
```