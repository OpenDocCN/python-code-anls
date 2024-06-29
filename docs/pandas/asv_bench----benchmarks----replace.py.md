# `D:\src\scipysrc\pandas\asv_bench\benchmarks\replace.py`

```
# 导入NumPy库，并将其重命名为np
import numpy as np

# 导入Pandas库，并将其重命名为pd
import pandas as pd

# 定义FillNa类
class FillNa:
    # 参数设置为布尔类型的列表，表示是否原地填充
    params = [True, False]
    # 参数名称为"inplace"
    param_names = ["inplace"]

    # 初始化方法，生成包含空值的时间序列
    def setup(self, inplace):
        # 创建包含100万个时间点的时间范围
        N = 10**6
        rng = pd.date_range("1/1/2000", periods=N, freq="min")
        # 创建包含随机数据的数组，随机将一半数据设为NaN
        data = np.random.randn(N)
        data[::2] = np.nan
        # 创建时间序列，使用随机数据和时间索引
        self.ts = pd.Series(data, index=rng)

    # 填充NaN值的方法
    def time_fillna(self, inplace):
        # 使用0.0填充NaN值，根据inplace参数决定是否原地填充
        self.ts.fillna(0.0, inplace=inplace)

    # 替换NaN值的方法
    def time_replace(self, inplace):
        # 将NaN值替换为0.0，根据inplace参数决定是否原地替换
        self.ts.replace(np.nan, 0.0, inplace=inplace)


# 定义ReplaceDict类
class ReplaceDict:
    # 参数设置为布尔类型的列表，表示是否原地替换
    params = [True, False]
    # 参数名称为"inplace"
    param_names = ["inplace"]

    # 初始化方法，创建需要替换的字典和随机数据的Series
    def setup(self, inplace):
        # 创建包含10万个键值对的字典，键为递增整数，值为递增整数加上起始值
        N = 10**5
        start_value = 10**5
        self.to_rep = dict(enumerate(np.arange(N) + start_value))
        # 创建包含1000个随机整数的Series
        self.s = pd.Series(np.random.randint(N, size=10**3))

    # 替换Series中值的方法
    def time_replace_series(self, inplace):
        # 根据字典self.to_rep替换Series中的值，根据inplace参数决定是否原地替换
        self.s.replace(self.to_rep, inplace=inplace)


# 定义ReplaceList类
class ReplaceList:
    # 参数设置为包含一个布尔值的元组
    params = [(True, False)]
    # 参数名称为"inplace"
    param_names = ["inplace"]

    # 初始化方法，创建包含大量数据的DataFrame
    def setup(self, inplace):
        # 创建包含1000万行和两列的DataFrame，初始值为0
        self.df = pd.DataFrame({"A": 0, "B": 0}, index=range(10**7))

    # 替换DataFrame中特定值的方法
    def time_replace_list(self, inplace):
        # 将DataFrame中的np.inf和-np.inf替换为NaN，根据inplace参数决定是否原地替换
        self.df.replace([np.inf, -np.inf], np.nan, inplace=inplace)

    # 替换DataFrame中特定值的方法（包含一个可以匹配的值）
    def time_replace_list_one_match(self, inplace):
        # 将DataFrame中的np.inf、-np.inf和1替换为NaN，根据inplace参数决定是否原地替换
        self.df.replace([np.inf, -np.inf, 1], np.nan, inplace=inplace)


# 定义Convert类
class Convert:
    # 参数设置为两个列表，分别表示要操作的数据结构和要替换的数据类型
    params = (["DataFrame", "Series"], ["Timestamp", "Timedelta"])
    # 参数名称为"constructor"和"replace_data"
    param_names = ["constructor", "replace_data"]

    # 初始化方法，根据参数构造数据和要替换的数据类型字典
    def setup(self, constructor, replace_data):
        # 创建包含随机整数的Series和DataFrame
        N = 10**3
        data = {
            "Series": pd.Series(np.random.randint(N, size=N)),
            "DataFrame": pd.DataFrame(
                {"A": np.random.randint(N, size=N), "B": np.random.randint(N, size=N)}
            ),
        }
        # 创建替换字典，将索引映射到Pandas中的Timestamp或Timedelta对象
        self.to_replace = {i: getattr(pd, replace_data) for i in range(N)}
        # 根据constructor参数选择要操作的数据结构
        self.data = data[constructor]

    # 替换数据的方法
    def time_replace(self, constructor, replace_data):
        # 使用self.to_replace字典替换self.data中的值
        self.data.replace(self.to_replace)


# 导入pandas_vb_common模块中的setup函数，跳过isort检查
from .pandas_vb_common import setup  # noqa: F401 isort:skip
```