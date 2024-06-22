# `.\pandas-ta\pandas_ta\statistics\__init__.py`

```py
# 设置文件编码为 UTF-8，以支持中文等非 ASCII 字符
# 导入自定义模块中的函数，用于计算数据的不同统计量
from .entropy import entropy  # 导入 entropy 函数，用于计算数据的熵
from .kurtosis import kurtosis  # 导入 kurtosis 函数，用于计算数据的峰度
from .mad import mad  # 导入 mad 函数，用于计算数据的绝对中位差
from .median import median  # 导入 median 函数，用于计算数据的中位数
from .quantile import quantile  # 导入 quantile 函数，用于计算数据的分位数
from .skew import skew  # 导入 skew 函数，用于计算数据的偏度
from .stdev import stdev  # 导入 stdev 函数，用于计算数据的标准差
from .tos_stdevall import tos_stdevall  # 导入 tos_stdevall 函数，用于计算数据的时间序列标准差
from .variance import variance  # 导入 variance 函数，用于计算数据的方差
from .zscore import zscore  # 导入 zscore 函数，用于计算数据的 Z 分数
```