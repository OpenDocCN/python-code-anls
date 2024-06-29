# `D:\src\scipysrc\numpy\numpy\lib\_nanfunctions_impl.pyi`

```py
# 导入 numpy 中的一系列函数，从 fromnumeric 模块中导入的函数列表
from numpy._core.fromnumeric import (
    amin,      # 导入计算数组元素最小值的函数
    amax,      # 导入计算数组元素最大值的函数
    argmin,    # 导入返回数组中最小元素索引的函数
    argmax,    # 导入返回数组中最大元素索引的函数
    sum,       # 导入计算数组元素和的函数
    prod,      # 导入计算数组元素乘积的函数
    cumsum,    # 导入计算数组累积和的函数
    cumprod,   # 导入计算数组累积乘积的函数
    mean,      # 导入计算数组均值的函数
    var,       # 导入计算数组方差的函数
    std        # 导入计算数组标准差的函数
)

from numpy.lib._function_base_impl import (
    median,     # 导入计算数组中位数的函数
    percentile, # 导入计算数组百分位数的函数
    quantile    # 导入计算数组分位数的函数
)

# 声明 __all__ 变量并赋值为字符串列表，用于模块导入时的限定符
__all__: list[str]

# 注意事项: 实际上这些函数并非别名，而是具有相同签名的独立函数。
# 将 numpy 中的函数赋值给对应的 nan 开头的变量，用于处理包含 NaN 值的数组
nanmin = amin       # 计算数组中 NaN 值忽略后的最小值
nanmax = amax       # 计算数组中 NaN 值忽略后的最大值
nanargmin = argmin  # 返回数组中 NaN 值忽略后的最小值索引
nanargmax = argmax  # 返回数组中 NaN 值忽略后的最大值索引
nansum = sum        # 计算数组中 NaN 值忽略后的元素和
nanprod = prod      # 计算数组中 NaN 值忽略后的元素乘积
nancumsum = cumsum  # 计算数组中 NaN 值忽略后的累积和
nancumprod = cumprod # 计算数组中 NaN 值忽略后的累积乘积
nanmean = mean      # 计算数组中 NaN 值忽略后的均值
nanvar = var        # 计算数组中 NaN 值忽略后的方差
nanstd = std        # 计算数组中 NaN 值忽略后的标准差
nanmedian = median  # 计算数组中 NaN 值忽略后的中位数
nanpercentile = percentile # 计算数组中 NaN 值忽略后的百分位数
nanquantile = quantile     # 计算数组中 NaN 值忽略后的分位数
```