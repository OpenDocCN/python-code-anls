# `D:\src\scipysrc\scipy\scipy\stats\morestats.py`

```
# 以下代码不适合公共使用，并将在 SciPy v2.0.0 版本中删除。
# 使用 `scipy.stats` 命名空间来导入以下列出的函数。

# 从 scipy._lib.deprecation 模块导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个列表，包含了将在 `scipy.stats.morestats` 子模块中暴露的函数名
__all__ = [  # noqa: F822
    'mvsdist',                    # 多变量分布函数
    'bayes_mvs', 'kstat', 'kstatvar', 'probplot', 'ppcc_max', 'ppcc_plot',  # 统计函数
    'boxcox_llf', 'boxcox', 'boxcox_normmax', 'boxcox_normplot',             # Box-Cox 变换函数
    'shapiro', 'anderson', 'ansari', 'bartlett', 'levene',                    # 统计检验函数
    'fligner', 'mood', 'wilcoxon', 'median_test',                             # 统计检验函数
    'circmean', 'circvar', 'circstd', 'anderson_ksamp',                      # 环形统计函数
    'yeojohnson_llf', 'yeojohnson', 'yeojohnson_normmax',                    # Yeo-Johnson 变换函数
    'yeojohnson_normplot', 'find_repeats', 'chi2_contingency', 'distributions',  # 统计函数
]

# 定义 __dir__() 函数，返回 __all__ 列表，用于显示可用的属性列表
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，处理属性的访问请求
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，指定参数以显示模块过时警告
    return _sub_module_deprecation(sub_package="stats", module="morestats",
                                   private_modules=["_morestats"], all=__all__,
                                   attribute=name)
```