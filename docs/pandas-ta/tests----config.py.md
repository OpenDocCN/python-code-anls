# `.\pandas-ta\tests\config.py`

```py
# 导入 os 模块，用于操作系统相关功能
import os
# 导入 pandas 模块中的 DatetimeIndex 和 read_csv 函数
from pandas import DatetimeIndex, read_csv

# 设定是否显示详细信息的标志
VERBOSE = True

# 设定警报和信息提示的图标
ALERT = f"[!]"
INFO = f"[i]"

# 设定相关性分析方法，这里选择使用 'corr'，也可以选择 'sem'
CORRELATION = "corr"  # "sem"
# 设定相关性阈值，小于 0.99 视为不理想
CORRELATION_THRESHOLD = 0.99  # Less than 0.99 is undesirable

# 读取样本数据集，使用 pandas 的 read_csv 函数
sample_data = read_csv(
    f"data/SPY_D.csv",  # 文件路径
    index_col=0,  # 以第一列作为索引
    parse_dates=True,  # 解析日期
    infer_datetime_format=True,  # 推断日期格式
    keep_date_col=True,  # 保留日期列
)
# 将日期列设置为索引，并丢弃原始日期列
sample_data.set_index(DatetimeIndex(sample_data["date"]), inplace=True, drop=True)
sample_data.drop("date", axis=1, inplace=True)

# 定义错误分析函数，用于输出错误信息
def error_analysis(df, kind, msg, icon=INFO, newline=True):
    if VERBOSE:  # 如果 VERBOSE 为 True，则输出信息
        s = f"{icon} {df.name}['{kind}']: {msg}"  # 构造输出字符串
        if newline:  # 如果需要换行
            s = f"\n{s}"  # 在字符串前添加换行符
        print(s)  # 打印信息

```  
```