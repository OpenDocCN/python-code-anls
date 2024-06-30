# `D:\src\scipysrc\scipy\scipy\stats\mstats_basic.py`

```
# 导入 `_sub_module_deprecation` 函数，用于处理子模块已弃用的警告
# 以后版本中，这个文件将被删除，推荐使用 `scipy.stats` 命名空间导入下面列出的函数

from scipy._lib.deprecation import _sub_module_deprecation

# 声明 `__all__` 变量，定义了模块中公开的函数名列表，用于 `from module import *` 语句
# 禁用 `flake8` 对缺少 `F822` 的警告，这是因为 `__all__` 变量是动态生成的

__all__ = [
    'argstoarray',          # 将参数转换为数组
    'count_tied_groups',    # 计算并返回相同数值组的数量
    'describe',             # 描述性统计分析
    'f_oneway',             # 单因素方差分析
    'find_repeats',         # 查找重复值
    'friedmanchisquare',    # Friedman 秩和检验
    'kendalltau',           # Kendall Tau 相关系数
    'kendalltau_seasonal',  # 季节性 Kendall Tau 相关系数
    'kruskal',              # Kruskal-Wallis 检验
    'kruskalwallis',        # Kruskal-Wallis 检验
    'ks_twosamp',           # 两样本 Kolmogorov-Smirnov 检验
    'ks_2samp',             # 两样本 Kolmogorov-Smirnov 检验
    'kurtosis',             # 峰度
    'kurtosistest',         # 峰度检验
    'ks_1samp',             # 单样本 Kolmogorov-Smirnov 检验
    'kstest',               # Kolmogorov-Smirnov 检验
    'linregress',           # 线性回归
    'mannwhitneyu',         # Mann-Whitney U 检验
    'meppf',                # Meixner-Pollaczek 预测分布函数
    'mode',                 # 众数
    'moment',               # 统计矩
    'mquantiles',           # 分位数
    'msign',                # 符号函数
    'normaltest',           # 正态性检验
    'obrientransform',      # O'Brien 变换
    'pearsonr',             # Pearson 相关系数
    'plotting_positions',   # 绘图位置
    'pointbiserialr',       # 点二列相关系数
    'rankdata',             # 排名数据
    'scoreatpercentile',    # 百分位数
    'sem',                  # 标准误差均值
    'sen_seasonal_slopes',  # 季节性斜率估计
    'skew',                 # 偏度
    'skewtest',             # 偏度检验
    'spearmanr',            # Spearman 秩相关系数
    'siegelslopes',         # Siegel 斜率估计
    'theilslopes',          # Theil 斜率估计
    'tmax',                 # 最大值
    'tmean',                # 平均值
    'tmin',                 # 最小值
    'trim',                 # 修剪数据
    'trimboth',             # 双向修剪数据
    'trimtail',             # 尾部修剪数据
    'trima',                # 前向修剪数据
    'trimr',                # 后向修剪数据
    'trimmed_mean',         # 修剪均值
    'trimmed_std',          # 修剪标准差
    'trimmed_stde',         # 修剪标准误差
    'trimmed_var',          # 修剪方差
    'tsem',                 # 修剪标准误差均值
    'ttest_1samp',          # 单样本 t 检验
    'ttest_onesamp',        # 单样本 t 检验
    'ttest_ind',            # 独立样本 t 检验
    'ttest_rel',            # 相关样本 t 检验
    'tvar',                 # t 分布方差
    'variation',            # 变异系数
    'winsorize',            # 温索赛处理
    'brunnermunzel',        # Brunner-Munzel 检验
]

# 定义 `__dir__()` 函数，返回模块中公开的函数名列表
def __dir__():
    return __all__

# 定义 `__getattr__(name)` 函数，处理对未定义属性的访问，弃用警告处理
def __getattr__(name):
    return _sub_module_deprecation(sub_package="stats", module="mstats_basic",
                                   private_modules=["_mstats_basic"], all=__all__,
                                   attribute=name, correct_module="mstats")
```