# `D:\src\scipysrc\scipy\scipy\stats\_distr_params.py`

```
"""
Sane parameters for stats.distributions.
"""
import numpy as np

# Continuous distributions with their respective parameters
distcont = [
    ['alpha', (3.5704770516650459,)],  # Alpha distribution with shape parameter 3.5704770516650459
    ['anglit', ()],  # Anglit distribution without parameters
    ['arcsine', ()],  # Arcsine distribution without parameters
    ['argus', (1.0,)],  # Argus distribution with shape parameter 1.0
    ['beta', (2.3098496451481823, 0.62687954300963677)],  # Beta distribution with shape parameters (2.3098496451481823, 0.62687954300963677)
    ['betaprime', (5, 6)],  # Beta prime distribution with shape parameters (5, 6)
    ['bradford', (0.29891359763170633,)],  # Bradford distribution with shape parameter 0.29891359763170633
    ['burr', (10.5, 4.3)],  # Burr distribution with shape parameters (10.5, 4.3)
    ['burr12', (10, 4)],  # Burr type XII distribution with shape parameters (10, 4)
    ['cauchy', ()],  # Cauchy distribution without parameters
    ['chi', (78,)],  # Chi distribution with degrees of freedom 78
    ['chi2', (55,)],  # Chi-squared distribution with degrees of freedom 55
    ['cosine', ()],  # Cosine distribution without parameters
    ['crystalball', (2.0, 3.0)],  # Crystalball distribution with shape parameters (2.0, 3.0)
    ['dgamma', (1.1023326088288166,)],  # Double gamma distribution with shape parameter 1.1023326088288166
    ['dweibull', (2.0685080649914673,)],  # Double Weibull distribution with shape parameter 2.0685080649914673
    ['erlang', (10,)],  # Erlang distribution with shape parameter 10
    ['expon', ()],  # Exponential distribution without parameters
    ['exponnorm', (1.5,)],  # Exponentially modified normal distribution with shape parameter 1.5
    ['exponpow', (2.697119160358469,)],  # Exponential power distribution with shape parameter 2.697119160358469
    ['exponweib', (2.8923945291034436, 1.9505288745913174)],  # Exponentiated Weibull distribution with shape parameters (2.8923945291034436, 1.9505288745913174)
    ['f', (29, 18)],  # F distribution with degrees of freedom (29, 18)
    ['fatiguelife', (29,)],  # Fatigue life distribution with shape parameter 29
    ['fisk', (3.0857548622253179,)],  # Fisk distribution with shape parameter 3.0857548622253179
    ['foldcauchy', (4.7164673455831894,)],  # Folded Cauchy distribution with shape parameter 4.7164673455831894
    ['foldnorm', (1.9521253373555869,)],  # Folded normal distribution with shape parameter 1.9521253373555869
    ['gamma', (1.9932305483800778,)],  # Gamma distribution with shape parameter 1.9932305483800778
    ['gausshyper', (13.763771604130699, 3.1189636648681431,
                    2.5145980350183019, 5.1811649903971615)],  # Gauss hypergeometric distribution with parameters (13.763771604130699, 3.1189636648681431, 2.5145980350183019, 5.1811649903971615)
    ['genexpon', (9.1325976465418908, 16.231956600590632, 3.2819552690843983)],  # Generalized exponential distribution with shape parameters (9.1325976465418908, 16.231956600590632, 3.2819552690843983)
    ['genextreme', (-0.1,)],  # Generalized extreme value distribution with shape parameter -0.1
    ['gengamma', (4.4162385429431925, 3.1193091679242761)],  # Generalized gamma distribution with shape parameters (4.4162385429431925, 3.1193091679242761)
    ['gengamma', (4.4162385429431925, -3.1193091679242761)],  # Generalized gamma distribution with shape parameters (4.4162385429431925, -3.1193091679242761)
    ['genhalflogistic', (0.77274727809929322,)],  # Generalized half-logistic distribution with shape parameter 0.77274727809929322
    ['genhyperbolic', (0.5, 1.5, -0.5,)],  # Generalized hyperbolic distribution with shape parameters (0.5, 1.5, -0.5)
    ['geninvgauss', (2.3, 1.5)],  # Generalized inverse Gaussian distribution with shape parameters (2.3, 1.5)
    ['genlogistic', (0.41192440799679475,)],  # Generalized logistic distribution with shape parameter 0.41192440799679475
    ['gennorm', (1.2988442399460265,)],  # Generalized normal distribution with shape parameter 1.2988442399460265
    ['halfgennorm', (0.6748054997000371,)],  # Half generalized normal distribution with shape parameter 0.6748054997000371
    ['genpareto', (0.1,)],  # Generalized Pareto distribution with shape parameter 0.1
    ['gibrat', ()],  # Gibrat distribution without parameters
    ['gompertz', (0.94743713075105251,)],  # Gompertz distribution with shape parameter 0.94743713075105251
    ['gumbel_l', ()],  # Left-skewed Gumbel distribution without parameters
    ['gumbel_r', ()],  # Right-skewed Gumbel distribution without parameters
    ['halfcauchy', ()],  # Half Cauchy distribution without parameters
    ['halflogistic', ()],  # Half logistic distribution without parameters
    ['halfnorm', ()],  # Half normal distribution without parameters
    ['hypsecant', ()],  # Hyperbolic secant distribution without parameters
    ['invgamma', (4.0668996136993067,)],  # Inverse gamma distribution with shape parameter 4.0668996136993067
    ['invgauss', (0.14546264555347513,)],  # Inverse Gaussian distribution with shape parameter 0.14546264555347513
    ['invweibull', (10.58,)],  # Inverse Weibull distribution with shape parameter 10.58
    ['irwinhall', (10,)],  # Irwin-Hall distribution with shape parameter 10
    ['jf_skew_t', (8, 4)],  # Johnson's family skew t distribution with shape parameters (8, 4)
    ['johnsonsb', (4.3172675099141058, 3.1837781130785063)],  # Johnson SB distribution with shape parameters (4.3172675099141058, 3.1837781130785063)
    ['johnsonsu', (2.554395574161155, 2.2482281679651965)],  # Johnson SU distribution with shape parameters (2.554395574161155, 2.2482281679651965)
    ['kappa4', (0.0, 0.0)],  # Kappa 4 distribution with shape parameters (0.0, 0.0)
    ['kappa4', (-0.1, 0.1)],  # Kappa 4 distribution with shape parameters (-0.1, 0.1)
    ['kappa4', (0.0, 0.1)],  # Kappa 4 distribution with shape parameters (0.0, 0.1)
    ['kappa4', (0.1, 0.0)],  # Kappa 4 distribution with shape parameters (0.1, 0.0)
    ['kappa3', (1.0,)],  # Kappa 3 distribution with shape parameter 1.0
    ['ksone', (1000,)],  # Kolmogorov-Smirnov one-sided distribution with shape parameter 1000
    ['kstwo', (10,)],  # Kolmogorov-Smirnov two-sided distribution with shape parameter 10
    ['kstwobign', ()],  # Kolmogorov-S
    # 定义一个包含分布名称和参数元组的列表
    [
        # 正态逆高斯分布，参数 (1.25, 0.5)
        ['norminvgauss', (1.25, 0.5)],
    
        # 帕累托分布，参数 (2.621716532144454,)
        ['pareto', (2.621716532144454,)],
    
        # 皮尔逊类型 III 分布，参数 (0.1,)
        ['pearson3', (0.1,)],
    
        # 皮尔逊类型 III 分布，参数 (-2,)
        ['pearson3', (-2,)],
    
        # 幂律分布，参数 (1.6591133289905851,)
        ['powerlaw', (1.6591133289905851,)],
    
        # 幂律分布，参数 (0.6591133289905851,)
        ['powerlaw', (0.6591133289905851,)],
    
        # 力法正态分布，参数 (2.1413923530064087, 0.44639540782048337)
        ['powerlognorm', (2.1413923530064087, 0.44639540782048337)],
    
        # 力法正态分布，参数 (4.4453652254590779,)
        ['powernorm', (4.4453652254590779,)],
    
        # 雷利分布，无参数
        ['rayleigh', ()],
    
        # R 分布，参数 (1.6,)
        ['rdist', (1.6,)],
    
        # 逆高斯互倒分布，参数 (0.63004267809369119,)
        ['recipinvgauss', (0.63004267809369119,)],
    
        # 倒数分布，参数 (0.01, 1.25)
        ['reciprocal', (0.01, 1.25)],
    
        # 相对布雷特维格纳分布，参数 (36.545206797050334,)
        ['rel_breitwigner', (36.545206797050334,)],
    
        # 莱斯分布，参数 (0.7749725210111873,)
        ['rice', (0.7749725210111873,)],
    
        # 半圆分布，无参数
        ['semicircular', ()],
    
        # 偏斜柯西分布，参数 (0.5,)
        ['skewcauchy', (0.5,)],
    
        # 偏斜正态分布，参数 (4.0,)
        ['skewnorm', (4.0,)],
    
        # 学生化范围分布，参数 (3.0, 10.0)
        ['studentized_range', (3.0, 10.0)],
    
        # 学生 t 分布，参数 (2.7433514990818093,)
        ['t', (2.7433514990818093,)],
    
        # 梯形分布，参数 (0.2, 0.8)
        ['trapezoid', (0.2, 0.8)],
    
        # 三角形分布，参数 (0.15785029824528218,)
        ['triang', (0.15785029824528218,)],
    
        # 截断指数分布，参数 (4.6907725456810478,)
        ['truncexpon', (4.6907725456810478,)],
    
        # 截断正态分布，参数 (-1.0978730080013919, 2.7306754109031979)
        ['truncnorm', (-1.0978730080013919, 2.7306754109031979)],
    
        # 截断正态分布，参数 (0.1, 2.0)
        ['truncnorm', (0.1, 2.0)],
    
        # 截断帕累托分布，参数 (1.8, 5.3)
        ['truncpareto', (1.8, 5.3)],
    
        # 截断帕累托分布，参数 (2, 5)
        ['truncpareto', (2, 5)],
    
        # 截断威布尔最小分布，参数 (2.5, 0.25, 1.75)
        ['truncweibull_min', (2.5, 0.25, 1.75)],
    
        # 图基-λ 分布，参数 (3.1321477856738267,)
        ['tukeylambda', (3.1321477856738267,)],
    
        # 均匀分布，无参数
        ['uniform', ()],
    
        # 冯·米塞斯分布，参数 (3.9939042581071398,)
        ['vonmises', (3.9939042581071398,)],
    
        # 冯·米塞斯线分布，参数 (3.9939042581071398,)
        ['vonmises_line', (3.9939042581071398,)],
    
        # 瓦尔德分布，无参数
        ['wald', ()],
    
        # 威布尔最大分布，参数 (2.8687961709100187,)
        ['weibull_max', (2.8687961709100187,)],
    
        # 威布尔最小分布，参数 (1.7866166930421596,)
        ['weibull_min', (1.7866166930421596,)],
    
        # 包裹柯西分布，参数 (0.031071279018614728,)
        ['wrapcauchy', (0.031071279018614728,)]
    ]
# 离散分布的列表，每个条目包含分布名称和参数元组
distdiscrete = [
    ['bernoulli',(0.3,)],  # 伯努利分布，参数为 (0.3,)
    ['betabinom', (5, 2.3, 0.63)],  # 贝塔二项分布，参数为 (5, 2.3, 0.63)
    ['betanbinom', (5, 9.3, 1)],  # 贝塔负二项分布，参数为 (5, 9.3, 1)
    ['binom', (5, 0.4)],  # 二项分布，参数为 (5, 0.4)
    ['boltzmann',(1.4, 19)],  # 玻尔兹曼分布，参数为 (1.4, 19)
    ['dlaplace', (0.8,)],  # 老拉普拉斯分布，参数为 (0.8,)
    ['geom', (0.5,)],  # 几何分布，参数为 (0.5,)
    ['hypergeom',(30, 12, 6)],  # 超几何分布，参数为 (30, 12, 6)
    ['hypergeom',(21,3,12)],  # 超几何分布，参数为 (21, 3, 12)，参考 numpy.random (3,18,12) numpy ticket:921
    ['hypergeom',(21,18,11)],  # 超几何分布，参数为 (21, 18, 11)，参考 numpy.random (18,3,11) numpy ticket:921
    ['nchypergeom_fisher', (140, 80, 60, 0.5)],  # Fisher型非中心超几何分布，参数为 (140, 80, 60, 0.5)
    ['nchypergeom_wallenius', (140, 80, 60, 0.5)],  # Wallenius型非中心超几何分布，参数为 (140, 80, 60, 0.5)
    ['logser', (0.6,)],  # 对数级数分布，参数为 (0.6,)，重新启用，参考 numpy ticket:921
    ['nbinom', (0.4, 0.4)],  # 负二项分布，参数为 (0.4, 0.4)，来自 tickets: 583
    ['nbinom', (5, 0.5)],  # 负二项分布，参数为 (5, 0.5)
    ['planck', (0.51,)],  # 普朗克分布，参数为 (0.51,)
    ['poisson', (0.6,)],  # 泊松分布，参数为 (0.6,)
    ['randint', (7, 31)],  # 随机整数分布，参数为 (7, 31)
    ['skellam', (15, 8)],  # Skellam分布，参数为 (15, 8)
    ['zipf', (6.6,)],  # Zipf分布，参数为 (6.6,)
    ['zipfian', (0.75, 15)],  # Zipfian分布，参数为 (0.75, 15)
    ['zipfian', (1.25, 10)],  # Zipfian分布，参数为 (1.25, 10)
    ['yulesimon', (11.0,)],  # Yule-Simon分布，参数为 (11.0,)
    ['nhypergeom', (20, 7, 1)]  # 多维超几何分布，参数为 (20, 7, 1)
]


# 无效离散分布的列表，每个条目包含分布名称和参数元组
invdistdiscrete = [
    # 在以下每个条目中，至少有一个形状参数是无效的
    ['hypergeom', (3, 3, 4)],  # 超几何分布，参数为 (3, 3, 4)
    ['nhypergeom', (5, 2, 8)],  # 多维超几何分布，参数为 (5, 2, 8)
    ['nchypergeom_fisher', (3, 3, 4, 1)],  # Fisher型非中心超几何分布，参数为 (3, 3, 4, 1)
    ['nchypergeom_wallenius', (3, 3, 4, 1)],  # Wallenius型非中心超几何分布，参数为 (3, 3, 4, 1)
    ['bernoulli', (1.5, )],  # 伯努利分布，参数为 (1.5,)
    ['binom', (10, 1.5)],  # 二项分布，参数为 (10, 1.5)
    ['betabinom', (10, -0.4, -0.5)],  # 贝塔二项分布，参数为 (10, -0.4, -0.5)
    ['betanbinom', (10, -0.4, -0.5)],  # 贝塔负二项分布，参数为 (10, -0.4, -0.5)
    ['boltzmann', (-1, 4)],  # 玻尔兹曼分布，参数为 (-1, 4)
    ['dlaplace', (-0.5, )],  # 老拉普拉斯分布，参数为 (-0.5,)
    ['geom', (1.5, )],  # 几何分布，参数为 (1.5,)
    ['logser', (1.5, )],  # 对数级数分布，参数为 (1.5,)
    ['nbinom', (10, 1.5)],  # 负二项分布，参数为 (10, 1.5)
    ['planck', (-0.5, )],  # 普朗克分布，参数为 (-0.5,)
    ['poisson', (-0.5, )],  # 泊松分布，参数为 (-0.5,)
    ['randint', (5, 2)],  # 随机整数分布，参数为 (5, 2)
    ['skellam', (-5, -2)],  # Skellam分布，参数为 (-5, -2)
    ['zipf', (-2, )],  # Zipf分布，参数为 (-2,)
    ['yulesimon', (-2, )],  # Yule-Simon分布，参数为 (-2,)
    ['zipfian', (-0.75, 15)]  # Zipfian分布，参数为 (-0.75, 15)
]


# 无效连续分布的列表，每个条目包含分布名称和参数元组
invdistcont = [
    # 在以下每个条目中，至少有一个形状参数是无效的
    ['alpha', (-1, )],  # Alpha分布，参数为 (-1,)
    ['anglit', ()],  # Anglit分布，无参数
    ['arcsine', ()],  # Arcsine分布，无参数
    ['argus', (-1, )],  # Argus分布，参数为 (-1,)
    ['beta', (-2, 2)],  # Beta分布，参数为 (-2, 2)
    ['betaprime', (-2, 2)],  # Beta'分布，参数为 (-2, 2)
    ['bradford', (-1, )],  # Bradford分布，参数为 (-1,)
    ['burr', (-1, 1)],  # Burr分布，参数为 (-1, 1)
    ['burr12', (-1, 1)],  # Burr12分布，参数为 (-1, 1)
    ['cauchy', ()],  # 柯西分布，无参数
    ['chi', (-1, )],  # 卡方分布，参数为 (-1,)
    ['chi2', (-1, )],  # 卡方分布，参数为 (-1,)
    ['cosine', ()],  # 余弦分布，无参数
    ['crystalball', (-1, 2)],  # Crystalball分布，参数为 (-1, 2)
    ['dgamma', (-1, )],  # 双伽玛分布，参数为 (-1,)
    ['
    ['jf_skew_t', (-1, 0)],
    ['johnsonsb', (1, -2)],
    ['johnsonsu', (1, -2)],
    ['kappa4', (np.nan, 0)],
    ['kappa3', (-1, )],
    ['ksone', (-1, )],
    ['kstwo', (-1, )],
    ['kstwobign', ()],
    ['laplace', ()],
    ['laplace_asymmetric', (-1, )],
    ['levy', ()],
    ['levy_l', ()],
    ['levy_stable', (-1, 1)],
    ['logistic', ()],
    ['loggamma', (-1, )],
    ['loglaplace', (-1, )],
    ['lognorm', (-1, )],
    ['loguniform', (10, 5)],
    ['lomax', (-1, )],
    ['maxwell', ()],
    ['mielke', (1, -2)],
    ['moyal', ()],
    ['nakagami', (-1, )],
    ['ncx2', (-1, 2)],
    ['ncf', (10, 20, -1)],
    ['nct', (-1, 2)],
    ['norm', ()],
    ['norminvgauss', (5, -10)],
    ['pareto', (-1, )],
    ['pearson3', (np.nan, )],
    ['powerlaw', (-1, )],
    ['powerlognorm', (1, -2)],
    ['powernorm', (-1, )],
    ['rdist', (-1, )],
    ['rayleigh', ()],
    ['rice', (-1, )],
    ['recipinvgauss', (-1, )],
    ['semicircular', ()],
    ['skewnorm', (np.inf, )],
    ['studentized_range', (-1, 1)],
    ['rel_breitwigner', (-2, )],
    ['t', (-1, )],
    ['trapezoid', (0, 2)],
    ['triang', (2, )],
    ['truncexpon', (-1, )],
    ['truncnorm', (10, 5)],
    ['truncpareto', (-1, 5)],
    ['truncpareto', (1.8, .5)],
    ['truncweibull_min', (-2.5, 0.25, 1.75)],
    ['tukeylambda', (np.nan, )],
    ['uniform', ()],
    ['vonmises', (-1, )],
    ['vonmises_line', (-1, )],
    ['wald', ()],
    ['weibull_min', (-1, )],
    ['weibull_max', (-1, )],
    ['wrapcauchy', (2, )],
    ['reciprocal', (15, 10)],
    ['skewcauchy', (2, )]



    ['jf_skew_t', (-1, 0)],  # 分布名称 'jf_skew_t' 和其参数 (-1, 0)
    ['johnsonsb', (1, -2)],   # 分布名称 'johnsonsb' 和其参数 (1, -2)
    ['johnsonsu', (1, -2)],   # 分布名称 'johnsonsu' 和其参数 (1, -2)
    ['kappa4', (np.nan, 0)],  # 分布名称 'kappa4' 和其参数 (NaN, 0)
    ['kappa3', (-1, )],       # 分布名称 'kappa3' 和其参数 (-1,)
    ['ksone', (-1, )],        # 分布名称 'ksone' 和其参数 (-1,)
    ['kstwo', (-1, )],        # 分布名称 'kstwo' 和其参数 (-1,)
    ['kstwobign', ()],        # 分布名称 'kstwobign' 和其参数 ()
    ['laplace', ()],          # 分布名称 'laplace' 和其参数 ()
    ['laplace_asymmetric', (-1, )],  # 分布名称 'laplace_asymmetric' 和其参数 (-1,)
    ['levy', ()],             # 分布名称 'levy' 和其参数 ()
    ['levy_l', ()],           # 分布名称 'levy_l' 和其参数 ()
    ['levy_stable', (-1, 1)], # 分布名称 'levy_stable' 和其参数 (-1, 1)
    ['logistic', ()],         # 分布名称 'logistic' 和其参数 ()
    ['loggamma', (-1, )],     # 分布名称 'loggamma' 和其参数 (-1,)
    ['loglaplace', (-1, )],   # 分布名称 'loglaplace' 和其参数 (-1,)
    ['lognorm', (-1, )],      # 分布名称 'lognorm' 和其参数 (-1,)
    ['loguniform', (10, 5)],  # 分布名称 'loguniform' 和其参数 (10, 5)
    ['lomax', (-1, )],        # 分布名称 'lomax' 和其参数 (-1,)
    ['maxwell', ()],          # 分布名称 'maxwell' 和其参数 ()
    ['mielke', (1, -2)],      # 分布名称 'mielke' 和其参数 (1, -2)
    ['moyal', ()],            # 分布名称 'moyal' 和其参数 ()
    ['nakagami', (-1, )],     # 分布名称 'nakagami' 和其参数 (-1,)
    ['ncx2', (-1, 2)],        # 分布名称 'ncx2' 和其参数 (-1, 2)
    ['ncf', (10, 20, -1)],    # 分布名称 'ncf' 和其参数 (10, 20, -1)
    ['nct', (-1, 2)],         # 分布名称 'nct' 和其参数 (-1, 2)
    ['norm', ()],             # 分布名称 'norm' 和其参数 ()
    ['norminvgauss', (5, -10)],  # 分布名称 'norminvgauss' 和其参数 (5, -10)
    ['pareto', (-1, )],       # 分布名称 'pareto' 和其参数 (-1,)
    ['pearson3', (np.nan, )], # 分布名称 'pearson3' 和其参数 (NaN,)
    ['powerlaw', (-1, )],     # 分布名称 'powerlaw' 和其参数 (-1,)
    ['powerlognorm', (1, -2)],# 分布名称 'powerlognorm' 和其参数 (1, -2)
    ['powernorm', (-1, )],    # 分布名称 'powernorm' 和其参数 (-1,)
    ['rdist', (-1, )],        # 分布名称 'rdist' 和其参数 (-1,)
    ['rayleigh', ()],         # 分布名称 'rayleigh' 和其参数 ()
    ['rice', (-1, )],         # 分布名称 'rice' 和其参数 (-1,)
    ['recipinvgauss', (-1, )],# 分布名称 'recipinvgauss' 和其参数 (-1,)
    ['semicircular', ()],     # 分布名称 'semicircular' 和其参数 ()
    ['skewnorm', (np.inf, )], # 分布名称 'skewnorm' 和其参数 (无穷,)
    ['studentized_range', (-1, 1)],  # 分布名称 'studentized_range' 和其参数 (-1, 1)
    ['rel_breitwigner', (-2, )],     # 分布名称 'rel_breitwigner' 和其参数 (-2,)
    ['t', (-1, )],            # 分布名称 't' 和其参数 (-1,)
    ['trapezoid', (0, 2)],    # 分布名称 'trapezoid' 和其参数 (0, 2)
    ['triang', (2, )],        # 分布名称 'triang' 和其参数 (2,)
    ['truncexpon', (-1, )],   # 分布名称 'truncexpon' 和其参数 (-1,)
    ['truncnorm', (10, 5)],   # 分布名称 'truncnorm' 和其参数 (10, 5)
    ['truncpareto', (-1, 5)], # 分布名称 'truncpareto' 和其参数 (-1, 5)
    ['truncpareto', (1.8, .5)],# 分布名称 'truncpareto' 和其参数 (1.8, .5)
    ['truncweibull_min', (-2.5, 0.25, 1.75)],  # 分布名称 'truncweibull_min' 和其参数 (-2.5, 0.25, 1.75)
    ['tukeylambda', (np.nan, )],   # 分布名称 'tukeylambda' 和其参数 (NaN,)
    ['uniform
]
```