# `D:\src\scipysrc\scipy\scipy\stats\tests\data\fisher_exact_results_from_r.py`

```
# 从 collections 模块导入 namedtuple 类型
from collections import namedtuple
# 导入 numpy 库并将其重命名为 np
import numpy as np

# 定义 Inf 常量为正无穷
Inf = np.inf

# 定义名为 Parameters 的命名元组，包含 table（二维数组）、confidence_level（置信水平）、alternative（备择假设）
Parameters = namedtuple('Parameters',
                        ['table', 'confidence_level', 'alternative'])
# 定义名为 RResults 的命名元组，包含 pvalue（p 值）、conditional_odds_ratio（条件比值比）、conditional_odds_ratio_ci（条件比值比置信区间）
RResults = namedtuple('RResults',
                      ['pvalue', 'conditional_odds_ratio',
                       'conditional_odds_ratio_ci'])

# 数据列表，包含多个元组，每个元组包含一个 Parameters 对象和一个 RResults 对象
data = [
    # 第一个数据点
    (Parameters(table=[[100, 2], [1000, 5]],  # 2x2 的表格数据
                confidence_level=0.95,       # 置信水平为 95%
                alternative='two.sided'),    # 备择假设为双侧
     RResults(pvalue=0.1300759363430016,                 # p 值
              conditional_odds_ratio=0.25055839934223,   # 条件比值比
              conditional_odds_ratio_ci=(0.04035202926536294, 2.662846672960251))),  # 条件比值比的置信区间

    # 第二个数据点
    (Parameters(table=[[2, 7], [8, 2]],    # 2x2 的表格数据
                confidence_level=0.95,     # 置信水平为 95%
                alternative='two.sided'),  # 备择假设为双侧
     RResults(pvalue=0.02301413756522116,                # p 值
              conditional_odds_ratio=0.0858623513573622,  # 条件比值比
              conditional_odds_ratio_ci=(0.004668988338943325, 0.895792956493601))),  # 条件比值比的置信区间

    # 后续数据点依此类推，每个数据点都是一个 Parameters 对象和一个 RResults 对象的组合
    # 第一个元组：包含参数和结果对象
    (Parameters(table=[[5, 0], [1, 4]],  # 参数对象：包含一个2x2的表格
                confidence_level=0.95,  # 置信水平为95%
                alternative='two.sided'),  # 备择假设为双侧

     RResults(pvalue=0.04761904761904762,  # 结果对象：p值为0.0476
              conditional_odds_ratio=Inf,  # 有条件的比率比为无穷大
              conditional_odds_ratio_ci=(1.024822256141754, Inf))),  # 有条件的比率比的置信区间

    # 第二个元组：包含参数和结果对象
    (Parameters(table=[[0, 5], [1, 4]],  # 参数对象：包含一个2x2的表格
                confidence_level=0.95,  # 置信水平为95%
                alternative='two.sided'),  # 备择假设为双侧

     RResults(pvalue=1,  # 结果对象：p值为1
              conditional_odds_ratio=0,  # 有条件的比率比为0
              conditional_odds_ratio_ci=(0, 39.00054996869288))),  # 有条件的比率比的置信区间

    # 第三个元组：包含参数和结果对象
    (Parameters(table=[[5, 1], [0, 4]],  # 参数对象：包含一个2x2的表格
                confidence_level=0.95,  # 置信水平为95%
                alternative='two.sided'),  # 备择假设为双侧

     RResults(pvalue=0.04761904761904761,  # 结果对象：p值为0.0476
              conditional_odds_ratio=Inf,  # 有条件的比率比为无穷大
              conditional_odds_ratio_ci=(1.024822256141754, Inf))),  # 有条件的比率比的置信区间

    # 第四个元组：包含参数和结果对象
    (Parameters(table=[[0, 1], [3, 2]],  # 参数对象：包含一个2x2的表格
                confidence_level=0.95,  # 置信水平为95%
                alternative='two.sided'),  # 备择假设为双侧

     RResults(pvalue=1,  # 结果对象：p值为1
              conditional_odds_ratio=0,  # 有条件的比率比为0
              conditional_odds_ratio_ci=(0, 39.00054996869287))),  # 有条件的比率比的置信区间

    # 第五个元组：包含参数和结果对象
    (Parameters(table=[[200, 7], [8, 300]],  # 参数对象：包含一个2x2的表格
                confidence_level=0.95,  # 置信水平为95%
                alternative='two.sided'),  # 备择假设为双侧

     RResults(pvalue=2.005657880389071e-122,  # 结果对象：p值为极小的数值
              conditional_odds_ratio=977.7866978606228,  # 有条件的比率比为977.79
              conditional_odds_ratio_ci=(349.2595113327733, 3630.382605689872))),  # 有条件的比率比的置信区间

    # 第六个元组：包含参数和结果对象
    (Parameters(table=[[28, 21], [6, 1957]],  # 参数对象：包含一个2x2的表格
                confidence_level=0.95,  # 置信水平为95%
                alternative='two.sided'),  # 备择假设为双侧

     RResults(pvalue=5.728437460831947e-44,  # 结果对象：p值为极小的数值
              conditional_odds_ratio=425.2403028434684,  # 有条件的比率比为425.24
              conditional_odds_ratio_ci=(152.4166024390096, 1425.700792178893))),  # 有条件的比率比的置信区间

    # 第七个元组：包含参数和结果对象
    (Parameters(table=[[190, 800], [200, 900]],  # 参数对象：包含一个2x2的表格
                confidence_level=0.95,  # 置信水平为95%
                alternative='two.sided'),  # 备择假设为双侧

     RResults(pvalue=0.574111858126088,  # 结果对象：p值为0.5741
              conditional_odds_ratio=1.068697577856801,  # 有条件的比率比为1.0687
              conditional_odds_ratio_ci=(0.8520462587912048, 1.340148950273938))),  # 有条件的比率比的置信区间

    # 第八个元组：包含参数和结果对象
    (Parameters(table=[[100, 2], [1000, 5]],  # 参数对象：包含一个2x2的表格
                confidence_level=0.99,  # 置信水平为99%
                alternative='two.sided'),  # 备择假设为双侧

     RResults(pvalue=0.1300759363430016,  # 结果对象：p值为0.1301
              conditional_odds_ratio=0.25055839934223,  # 有条件的比率比为0.2506
              conditional_odds_ratio_ci=(0.02502345007115455, 6.304424772117853))),  # 有条件的比率比的置信区间

    # 第九个元组：包含参数和结果对象
    (Parameters(table=[[2, 7], [8, 2]],  # 参数对象：包含一个2x2的表格
                confidence_level=0.99,  # 置信水平为99%
                alternative='two.sided'),  # 备择假设为双侧

     RResults(pvalue=0.02301413756522116,  # 结果对象：p值为0.0230
              conditional_odds_ratio=0.0858623513573622,  # 有条件的比率比为0.0859
              conditional_odds_ratio_ci=(0.001923034001462487, 1.53670836950172))),  # 有条件的比率比的置信区间
    # 创建一个包含多个元组的列表，每个元组包含两个对象：
    # - Parameters：包含一个二维数组和其他参数，表示统计分析的输入数据和参数
    # - RResults：包含统计分析的结果，如 p 值、条件比值以及其置信区间
    
    [
        (Parameters(table=[[5, 1], [10, 10]],
                    confidence_level=0.99,
                    alternative='two.sided'),
         RResults(pvalue=0.1973244147157191,
                  conditional_odds_ratio=4.725646047336587,
                  conditional_odds_ratio_ci=(0.2397970951413721,
                                             1291.342011095509))),
        (Parameters(table=[[5, 15], [20, 20]],
                    confidence_level=0.99,
                    alternative='two.sided'),
         RResults(pvalue=0.09580440012477633,
                  conditional_odds_ratio=0.3394396617440851,
                  conditional_odds_ratio_ci=(0.05127576113762925,
                                             1.717176678806983))),
        (Parameters(table=[[5, 16], [16, 25]],
                    confidence_level=0.99,
                    alternative='two.sided'),
         RResults(pvalue=0.2697004098849359,
                  conditional_odds_ratio=0.4937791394540491,
                  conditional_odds_ratio_ci=(0.07498546954483619,
                                             2.506969905199901))),
        (Parameters(table=[[10, 5], [10, 1]],
                    confidence_level=0.99,
                    alternative='two.sided'),
         RResults(pvalue=0.1973244147157192,
                  conditional_odds_ratio=0.2116112781158479,
                  conditional_odds_ratio_ci=(0.0007743881879531337,
                                             4.170192301163831))),
        (Parameters(table=[[10, 5], [10, 0]],
                    confidence_level=0.99,
                    alternative='two.sided'),
         RResults(pvalue=0.06126482213438735,
                  conditional_odds_ratio=0,
                  conditional_odds_ratio_ci=(0,
                                             2.642491011905582))),
        (Parameters(table=[[5, 0], [1, 4]],
                    confidence_level=0.99,
                    alternative='two.sided'),
         RResults(pvalue=0.04761904761904762,
                  conditional_odds_ratio=Inf,
                  conditional_odds_ratio_ci=(0.496935393325443,
                                             Inf))),
        (Parameters(table=[[0, 5], [1, 4]],
                    confidence_level=0.99,
                    alternative='two.sided'),
         RResults(pvalue=1,
                  conditional_odds_ratio=0,
                  conditional_odds_ratio_ci=(0,
                                             198.019801980198))),
        (Parameters(table=[[5, 1], [0, 4]],
                    confidence_level=0.99,
                    alternative='two.sided'),
         RResults(pvalue=0.04761904761904761,
                  conditional_odds_ratio=Inf,
                  conditional_odds_ratio_ci=(0.496935393325443,
                                             Inf))),
        (Parameters(table=[[0, 1], [3, 2]],
                    confidence_level=0.99,
                    alternative='two.sided'),
         RResults(pvalue=1,
                  conditional_odds_ratio=0,
                  conditional_odds_ratio_ci=(0,
                                             198.019801980198))),
    ]
    (Parameters(table=[[200, 7], [8, 300]],
                confidence_level=0.99,
                alternative='two.sided'),
     RResults(pvalue=2.005657880389071e-122,
              conditional_odds_ratio=977.7866978606228,
              conditional_odds_ratio_ci=(270.0334165523604,
                                         5461.333333326708))),


    # 第一组数据参数设置：包含一个2x2的列联表，99%置信水平，双侧检验
    (Parameters(table=[[200, 7], [8, 300]],
                confidence_level=0.99,
                alternative='two.sided'),
     # 第一组数据结果：p 值非常小，条件比率为 977.79，置信区间为 (270.03, 5461.33)
     RResults(pvalue=2.005657880389071e-122,
              conditional_odds_ratio=977.7866978606228,
              conditional_odds_ratio_ci=(270.0334165523604,
                                         5461.333333326708))),



    (Parameters(table=[[28, 21], [6, 1957]],
                confidence_level=0.99,
                alternative='two.sided'),
     RResults(pvalue=5.728437460831947e-44,
              conditional_odds_ratio=425.2403028434684,
              conditional_odds_ratio_ci=(116.7944750275836,
                                         1931.995993191814))),


    # 第二组数据参数设置：包含一个2x2的列联表，99%置信水平，双侧检验
    (Parameters(table=[[28, 21], [6, 1957]],
                confidence_level=0.99,
                alternative='two.sided'),
     # 第二组数据结果：p 值非常小，条件比率为 425.24，置信区间为 (116.79, 1931.99)
     RResults(pvalue=5.728437460831947e-44,
              conditional_odds_ratio=425.2403028434684,
              conditional_odds_ratio_ci=(116.7944750275836,
                                         1931.995993191814))),


（以下同样的方式注释每一组数据）
    (Parameters(table=[[10, 5], [10, 0]],  # 设置参数：包含一个二维列表作为表格，描述了两组数据的交叉统计
                confidence_level=0.95,    # 置信水平为 95%
                alternative='less'),       # 假设检验的备择假设为单侧小于检验
     RResults(pvalue=0.0565217391304348,  # 假设检验的 p 值
              conditional_odds_ratio=0,   # 条件比值（在条件表中，当一个特定条件成立时，比值的比率）
              conditional_odds_ratio_ci=(0, 1.06224603077045))),  # 条件比值的置信区间

    (Parameters(table=[[5, 0], [1, 4]],    # 设置参数：包含一个二维列表作为表格，描述了两组数据的交叉统计
                confidence_level=0.95,    # 置信水平为 95%
                alternative='less'),       # 假设检验的备择假设为单侧小于检验
     RResults(pvalue=1,                    # 假设检验的 p 值
              conditional_odds_ratio=Inf,  # 条件比值，无穷大表示条件下比值的无限大比率
              conditional_odds_ratio_ci=(0, Inf))),  # 条件比值的置信区间

    (Parameters(table=[[0, 5], [1, 4]],    # 设置参数：包含一个二维列表作为表格，描述了两组数据的交叉统计
                confidence_level=0.95,    # 置信水平为 95%
                alternative='less'),       # 假设检验的备择假设为单侧小于检验
     RResults(pvalue=0.5,                  # 假设检验的 p 值
              conditional_odds_ratio=0,    # 条件比值，为零表示条件下比值的零比率
              conditional_odds_ratio_ci=(0, 19.00192394479939))),  # 条件比值的置信区间

    (Parameters(table=[[5, 1], [0, 4]],    # 设置参数：包含一个二维列表作为表格，描述了两组数据的交叉统计
                confidence_level=0.95,    # 置信水平为 95%
                alternative='less'),       # 假设检验的备择假设为单侧小于检验
     RResults(pvalue=1,                    # 假设检验的 p 值
              conditional_odds_ratio=Inf,  # 条件比值，无穷大表示条件下比值的无限大比率
              conditional_odds_ratio_ci=(0, Inf))),  # 条件比值的置信区间

    (Parameters(table=[[0, 1], [3, 2]],    # 设置参数：包含一个二维列表作为表格，描述了两组数据的交叉统计
                confidence_level=0.95,    # 置信水平为 95%
                alternative='less'),       # 假设检验的备择假设为单侧小于检验
     RResults(pvalue=0.4999999999999999,   # 假设检验的 p 值
              conditional_odds_ratio=0,    # 条件比值，为零表示条件下比值的零比率
              conditional_odds_ratio_ci=(0, 19.00192394479939))),  # 条件比值的置信区间

    (Parameters(table=[[200, 7], [8, 300]],  # 设置参数：包含一个二维列表作为表格，描述了两组数据的交叉统计
                confidence_level=0.95,       # 置信水平为 95%
                alternative='less'),          # 假设检验的备择假设为单侧小于检验
     RResults(pvalue=1,                     # 假设检验的 p 值
              conditional_odds_ratio=977.7866978606228,  # 条件比值
              conditional_odds_ratio_ci=(0, 3045.460216525746))),  # 条件比值的置信区间

    (Parameters(table=[[28, 21], [6, 1957]],  # 设置参数：包含一个二维列表作为表格，描述了两组数据的交叉统计
                confidence_level=0.95,         # 置信水平为 95%
                alternative='less'),            # 假设检验的备择假设为单侧小于检验
     RResults(pvalue=1,                       # 假设检验的 p 值
              conditional_odds_ratio=425.2403028434684,  # 条件比值
              conditional_odds_ratio_ci=(0, 1186.440170942579))),  # 条件比值的置信区间

    (Parameters(table=[[190, 800], [200, 900]],  # 设置参数：包含一个二维列表作为表格，描述了两组数据的交叉统计
                confidence_level=0.95,            # 置信水平为 95%
                alternative='less'),               # 假设检验的备择假设为单侧小于检验
     RResults(pvalue=0.7416227010368963,          # 假设检验的 p 值
              conditional_odds_ratio=1.068697577856801,  # 条件比值
              conditional_odds_ratio_ci=(0, 1.293551891610822))),  # 条件比值的置信区间

    (Parameters(table=[[100, 2], [1000, 5]],    # 设置参数：包含一个二维列表作为表格，描述了两组数据的交叉统计
                confidence_level=0.99,           # 置信水平为 99%
                alternative='less'),              # 假设检验的备择假设为单侧小于检验
     RResults(pvalue=0.1300759363430016,         # 假设检验的 p 值
              conditional_odds_ratio=0.25055839934223,  # 条件比值
              conditional_odds_ratio_ci=(0, 4.375946050832565))),  # 条件比值的置信区间
    (Parameters(table=[[2, 7], [8, 2]],
                confidence_level=0.99,
                alternative='less'),
     RResults(pvalue=0.0185217259520665,
              conditional_odds_ratio=0.0858623513573622,
              conditional_odds_ratio_ci=(0,
                                         1.235282118191202))),


    # 定义第一个数据对：参数表格为 [[2, 7], [8, 2]]，置信水平为 0.99，备择假设为“小于”
    (Parameters(table=[[2, 7], [8, 2]],
                confidence_level=0.99,
                alternative='less'),
     # 结果集为 RResults 对象，包含 p 值、条件比值、条件比值置信区间
     RResults(pvalue=0.0185217259520665,
              conditional_odds_ratio=0.0858623513573622,
              conditional_odds_ratio_ci=(0,
                                         1.235282118191202))),



    (Parameters(table=[[5, 1], [10, 10]],
                confidence_level=0.99,
                alternative='less'),
     RResults(pvalue=0.9782608695652173,
              conditional_odds_ratio=4.725646047336587,
              conditional_odds_ratio_ci=(0,
                                         657.2063583945989))),


    # 定义第二个数据对：参数表格为 [[5, 1], [10, 10]]，置信水平为 0.99，备择假设为“小于”
    (Parameters(table=[[5, 1], [10, 10]],
                confidence_level=0.99,
                alternative='less'),
     # 结果集为 RResults 对象，包含 p 值、条件比值、条件比值置信区间
     RResults(pvalue=0.9782608695652173,
              conditional_odds_ratio=4.725646047336587,
              conditional_odds_ratio_ci=(0,
                                         657.2063583945989))),



    (Parameters(table=[[5, 15], [20, 20]],
                confidence_level=0.99,
                alternative='less'),
     RResults(pvalue=0.05625775074399956,
              conditional_odds_ratio=0.3394396617440851,
              conditional_odds_ratio_ci=(0,
                                         1.498867660683128))),


    # 定义第三个数据对：参数表格为 [[5, 15], [20, 20]]，置信水平为 0.99，备择假设为“小于”
    (Parameters(table=[[5, 15], [20, 20]],
                confidence_level=0.99,
                alternative='less'),
     # 结果集为 RResults 对象，包含 p 值、条件比值、条件比值置信区间
     RResults(pvalue=0.05625775074399956,
              conditional_odds_ratio=0.3394396617440851,
              conditional_odds_ratio_ci=(0,
                                         1.498867660683128))),


（以下类似，依次注释每一个数据对和其对应的结果集）
    (Parameters(table=[[0, 1], [3, 2]],
                confidence_level=0.99,
                alternative='less'),
     RResults(pvalue=0.4999999999999999,
              conditional_odds_ratio=0,
              conditional_odds_ratio_ci=(0,
                                         99.00009507969123))),

# 创建第一个元组，包含两个对象：Parameters 和 RResults。Parameters 包含一个2x2的表格，confidence_level 为0.99，alternative 为'less'；RResults 包含 pvalue 为0.5，conditional_odds_ratio 为0，conditional_odds_ratio_ci 的范围为(0, 99.00009507969123)。


    (Parameters(table=[[200, 7], [8, 300]],
                confidence_level=0.99,
                alternative='less'),
     RResults(pvalue=1,
              conditional_odds_ratio=977.7866978606228,
              conditional_odds_ratio_ci=(0,
                                         4503.078257659934))),

# 创建第二个元组，包含两个对象：Parameters 和 RResults。Parameters 包含一个200x7和8x300的表格，confidence_level 为0.99，alternative 为'less'；RResults 包含 pvalue 为1，conditional_odds_ratio 为977.7866978606228，conditional_odds_ratio_ci 的范围为(0, 4503.078257659934)。


    (Parameters(table=[[28, 21], [6, 1957]],
                confidence_level=0.99,
                alternative='less'),
     RResults(pvalue=1,
              conditional_odds_ratio=425.2403028434684,
              conditional_odds_ratio_ci=(0,
                                         1811.766127544222))),

# 创建第三个元组，包含两个对象：Parameters 和 RResults。Parameters 包含一个28x21和6x1957的表格，confidence_level 为0.99，alternative 为'less'；RResults 包含 pvalue 为1，conditional_odds_ratio 为425.2403028434684，conditional_odds_ratio_ci 的范围为(0, 1811.766127544222)。

（以下类似地注释每个元组，描述了它们的结构和各个字段的含义。）
    (Parameters(table=[[10, 5], [10, 1]],
                confidence_level=0.95,
                alternative='greater'),
     RResults(pvalue=0.9782608695652174,
              conditional_odds_ratio=0.2116112781158479,
              conditional_odds_ratio_ci=(0.007821681994077808,
                                         Inf))),

# 定义了一组参数，包括一个2x2的列联表、置信水平和备择假设，并计算了相关的统计结果。


    (Parameters(table=[[10, 5], [10, 0]],
                confidence_level=0.95,
                alternative='greater'),
     RResults(pvalue=1,
              conditional_odds_ratio=0,
              conditional_odds_ratio_ci=(0,
                                         Inf))),

# 定义了另一组参数，列联表包括一个2x2，置信水平和备择假设，并计算了相关的统计结果。


    (Parameters(table=[[5, 0], [1, 4]],
                confidence_level=0.95,
                alternative='greater'),
     RResults(pvalue=0.02380952380952382,
              conditional_odds_ratio=Inf,
              conditional_odds_ratio_ci=(1.487678929918272,
                                         Inf))),

# 定义了另一组参数，列联表包括一个2x2，置信水平和备择假设，并计算了相关的统计结果。


    (Parameters(table=[[0, 5], [1, 4]],
                confidence_level=0.95,
                alternative='greater'),
     RResults(pvalue=1,
              conditional_odds_ratio=0,
              conditional_odds_ratio_ci=(0,
                                         Inf))),

# 定义了另一组参数，列联表包括一个2x2，置信水平和备择假设，并计算了相关的统计结果。


    (Parameters(table=[[5, 1], [0, 4]],
                confidence_level=0.95,
                alternative='greater'),
     RResults(pvalue=0.0238095238095238,
              conditional_odds_ratio=Inf,
              conditional_odds_ratio_ci=(1.487678929918272,
                                         Inf))),

# 定义了另一组参数，列联表包括一个2x2，置信水平和备择假设，并计算了相关的统计结果。


    (Parameters(table=[[0, 1], [3, 2]],
                confidence_level=0.95,
                alternative='greater'),
     RResults(pvalue=1,
              conditional_odds_ratio=0,
              conditional_odds_ratio_ci=(0,
                                         Inf))),

# 定义了另一组参数，列联表包括一个2x2，置信水平和备择假设，并计算了相关的统计结果。


    (Parameters(table=[[200, 7], [8, 300]],
                confidence_level=0.95,
                alternative='greater'),
     RResults(pvalue=2.005657880388915e-122,
              conditional_odds_ratio=977.7866978606228,
              conditional_odds_ratio_ci=(397.784359748113,
                                         Inf))),

# 定义了另一组参数，列联表包括一个2x2，置信水平和备择假设，并计算了相关的统计结果。


    (Parameters(table=[[28, 21], [6, 1957]],
                confidence_level=0.95,
                alternative='greater'),
     RResults(pvalue=5.728437460831983e-44,
              conditional_odds_ratio=425.2403028434684,
              conditional_odds_ratio_ci=(174.7148056880929,
                                         Inf))),

# 定义了另一组参数，列联表包括一个2x2，置信水平和备择假设，并计算了相关的统计结果。


    (Parameters(table=[[190, 800], [200, 900]],
                confidence_level=0.95,
                alternative='greater'),
     RResults(pvalue=0.2959825901308897,
              conditional_odds_ratio=1.068697577856801,
              conditional_odds_ratio_ci=(0.8828406663967776,
                                         Inf))),

# 定义了另一组参数，列联表包括一个2x2，置信水平和备择假设，并计算了相关的统计结果。
    # 定义一个元组，包含两个元素：
    # 第一个元素是一个包含数据表、置信水平和备择假设的 Parameters 对象
    # 第二个元素是一个包含 p 值、条件比值、以及条件比值置信区间的 RResults 对象
    (Parameters(table=[[100, 2], [1000, 5]],  # 数据表包含两行，每行两列的数据
                confidence_level=0.99,       # 置信水平为 99%
                alternative='greater'),       # 备择假设为“大于”
     RResults(pvalue=0.979790445314723,        # p 值为 0.979790445314723
              conditional_odds_ratio=0.25055839934223,  # 条件比值为 0.25055839934223
              conditional_odds_ratio_ci=(0.03045407081240429, Inf))),  # 条件比值置信区间从 0.03045407081240429 到无穷大
    
    (Parameters(table=[[2, 7], [8, 2]],         # 数据表包含两行，每行两列的数据
                confidence_level=0.99,        # 置信水平为 99%
                alternative='greater'),       # 备择假设为“大于”
     RResults(pvalue=0.9990149169715733,      # p 值为 0.9990149169715733
              conditional_odds_ratio=0.0858623513573622,  # 条件比值为 0.0858623513573622
              conditional_odds_ratio_ci=(0.002768053063547901, Inf))),  # 条件比值置信区间从 0.002768053063547901 到无穷大
    
    (Parameters(table=[[5, 1], [10, 10]],      # 数据表包含两行，每行两列的数据
                confidence_level=0.99,        # 置信水平为 99%
                alternative='greater'),       # 备择假设为“大于”
     RResults(pvalue=0.1652173913043478,      # p 值为 0.1652173913043478
              conditional_odds_ratio=4.725646047336587,  # 条件比值为 4.725646047336587
              conditional_odds_ratio_ci=(0.2998184792279909, Inf))),  # 条件比值置信区间从 0.2998184792279909 到无穷大
    
    (Parameters(table=[[5, 15], [20, 20]],     # 数据表包含两行，每行两列的数据
                confidence_level=0.99,        # 置信水平为 99%
                alternative='greater'),       # 备择假设为“大于”
     RResults(pvalue=0.9849086665340765,      # p 值为 0.9849086665340765
              conditional_odds_ratio=0.3394396617440851,  # 条件比值为 0.3394396617440851
              conditional_odds_ratio_ci=(0.06180414342643172, Inf))),  # 条件比值置信区间从 0.06180414342643172 到无穷大
    
    (Parameters(table=[[5, 16], [16, 25]],     # 数据表包含两行，每行两列的数据
                confidence_level=0.99,        # 置信水平为 99%
                alternative='greater'),       # 备择假设为“大于”
     RResults(pvalue=0.9330176609214881,      # p 值为 0.9330176609214881
              conditional_odds_ratio=0.4937791394540491,  # 条件比值为 0.4937791394540491
              conditional_odds_ratio_ci=(0.09037094010066403, Inf))),  # 条件比值置信区间从 0.09037094010066403 到无穷大
    
    (Parameters(table=[[10, 5], [10, 1]],      # 数据表包含两行，每行两列的数据
                confidence_level=0.99,        # 置信水平为 99%
                alternative='greater'),       # 备择假设为“大于”
     RResults(pvalue=0.9782608695652174,      # p 值为 0.9782608695652174
              conditional_odds_ratio=0.2116112781158479,  # 条件比值为 0.2116112781158479
              conditional_odds_ratio_ci=(0.001521592095430679, Inf))),  # 条件比值置信区间从 0.001521592095430679 到无穷大
    
    (Parameters(table=[[10, 5], [10, 0]],      # 数据表包含两行，每行两列的数据
                confidence_level=0.99,        # 置信水平为 99%
                alternative='greater'),       # 备择假设为“大于”
     RResults(pvalue=1,                       # p 值为 1
              conditional_odds_ratio=0,       # 条件比值为 0
              conditional_odds_ratio_ci=(0, Inf))),  # 条件比值置信区间从 0 到无穷大
    
    (Parameters(table=[[5, 0], [1, 4]],        # 数据表包含两行，每行两列的数据
                confidence_level=0.99,        # 置信水平为 99%
                alternative='greater'),       # 备择假设为“大于”
     RResults(pvalue=0.02380952380952382,     # p 值为 0.02380952380952382
              conditional_odds_ratio=Inf,    # 条件比值为无穷大
              conditional_odds_ratio_ci=(0.6661157890359722, Inf))),  # 条件比值置信区间从 0.6661157890359722 到无穷大
    
    (Parameters(table=[[0, 5], [1, 4]],        # 数据表包含两行，每行两列的数据
                confidence_level=0.99,        # 置信水平为 99%
                alternative='greater'),       # 备择假设为“大于”
     RResults(pvalue=1,                       # p 值为 1
              conditional_odds_ratio=0,       # 条件比值为 0
              conditional_odds_ratio_ci=(0, Inf))),  # 条件比值置信区间从 0 到无穷大
    (Parameters(table=[[5, 1], [0, 4]],  # 定义一个二维列表作为参数的表格，表示一个2x2的交叉表
                confidence_level=0.99,  # 置信水平设定为0.99
                alternative='greater'),  # 假设检验的备择假设设定为大于
     RResults(pvalue=0.0238095238095238,  # 假设检验结果的 p 值
              conditional_odds_ratio=Inf,  # 条件比值的估计结果为无穷大
              conditional_odds_ratio_ci=(0.6661157890359725,  # 条件比值的置信区间下界
                                         Inf))),  # 条件比值的置信区间上界为无穷大

    (Parameters(table=[[0, 1], [3, 2]],  # 定义一个二维列表作为参数的表格，表示一个2x2的交叉表
                confidence_level=0.99,  # 置信水平设定为0.99
                alternative='greater'),  # 假设检验的备择假设设定为大于
     RResults(pvalue=1,  # 假设检验结果的 p 值
              conditional_odds_ratio=0,  # 条件比值的估计结果为0
              conditional_odds_ratio_ci=(0,  # 条件比值的置信区间下界
                                         Inf))),  # 条件比值的置信区间上界为无穷大

    (Parameters(table=[[200, 7], [8, 300]],  # 定义一个二维列表作为参数的表格，表示一个2x2的交叉表
                confidence_level=0.99,  # 置信水平设定为0.99
                alternative='greater'),  # 假设检验的备择假设设定为大于
     RResults(pvalue=2.005657880388915e-122,  # 假设检验结果的 p 值
              conditional_odds_ratio=977.7866978606228,  # 条件比值的估计结果
              conditional_odds_ratio_ci=(297.9619252357688,  # 条件比值的置信区间下界
                                         Inf))),  # 条件比值的置信区间上界为无穷大

    (Parameters(table=[[28, 21], [6, 1957]],  # 定义一个二维列表作为参数的表格，表示一个2x2的交叉表
                confidence_level=0.99,  # 置信水平设定为0.99
                alternative='greater'),  # 假设检验的备择假设设定为大于
     RResults(pvalue=5.728437460831983e-44,  # 假设检验结果的 p 值
              conditional_odds_ratio=425.2403028434684,  # 条件比值的估计结果
              conditional_odds_ratio_ci=(130.3213490295859,  # 条件比值的置信区间下界
                                         Inf))),  # 条件比值的置信区间上界为无穷大

    (Parameters(table=[[190, 800], [200, 900]],  # 定义一个二维列表作为参数的表格，表示一个2x2的交叉表
                confidence_level=0.99,  # 置信水平设定为0.99
                alternative='greater'),  # 假设检验的备择假设设定为大于
     RResults(pvalue=0.2959825901308897,  # 假设检验结果的 p 值
              conditional_odds_ratio=1.068697577856801,  # 条件比值的估计结果
              conditional_odds_ratio_ci=(0.8176272148267533,  # 条件比值的置信区间下界
                                         Inf))),  # 条件比值的置信区间上界为无穷大
]
```