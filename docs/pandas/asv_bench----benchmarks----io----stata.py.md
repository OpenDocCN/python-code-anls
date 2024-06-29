# `D:\src\scipysrc\pandas\asv_bench\benchmarks\io\stata.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算

from pandas import (  # 从 pandas 库中导入以下模块：
    DataFrame,       # DataFrame：用于创建和操作数据帧
    Index,           # Index：用于处理索引
    date_range,      # date_range：用于生成日期范围
    read_stata,      # read_stata：用于读取 Stata 文件
)

from ..pandas_vb_common import BaseIO  # 从自定义模块中导入 BaseIO 类

class Stata(BaseIO):
    params = ["tc", "td", "tm", "tw", "th", "tq", "ty"]  # 参数列表
    param_names = ["convert_dates"]  # 参数名称列表

    def setup(self, convert_dates):
        self.fname = "__test__.dta"  # 设置文件名为 "__test__.dta"
        N = self.N = 100000  # 设置数据行数 N 为 100000
        C = self.C = 5  # 设置列数 C 为 5
        self.df = DataFrame(  # 创建一个 DataFrame 对象
            np.random.randn(N, C),  # 使用随机数填充 N 行 C 列的数据
            columns=[f"float{i}" for i in range(C)],  # 设置列名为 float0, float1, ...
            index=date_range("20000101", periods=N, freq="h"),  # 设置时间索引，每小时一个时间点
        )
        self.df["object"] = Index([f"i-{i}" for i in range(self.N)], dtype=object)  # 在 DataFrame 中添加一个对象类型的列
        self.df["int8_"] = np.random.randint(  # 添加 int8 类型的随机整数列
            np.iinfo(np.int8).min, np.iinfo(np.int8).max - 27, N
        )
        self.df["int16_"] = np.random.randint(  # 添加 int16 类型的随机整数列
            np.iinfo(np.int16).min, np.iinfo(np.int16).max - 27, N
        )
        self.df["int32_"] = np.random.randint(  # 添加 int32 类型的随机整数列
            np.iinfo(np.int32).min, np.iinfo(np.int32).max - 27, N
        )
        self.df["float32_"] = np.array(np.random.randn(N), dtype=np.float32)  # 添加 float32 类型的随机浮点数列
        self.convert_dates = {"index": convert_dates}  # 设置日期转换的参数字典
        self.df.to_stata(self.fname, convert_dates=self.convert_dates)  # 将 DataFrame 写入 Stata 文件

    def time_read_stata(self, convert_dates):
        read_stata(self.fname)  # 读取指定的 Stata 文件

    def time_write_stata(self, convert_dates):
        self.df.to_stata(self.fname, convert_dates=self.convert_dates)  # 将 DataFrame 写入 Stata 文件

class StataMissing(Stata):
    def setup(self, convert_dates):
        super().setup(convert_dates)  # 调用父类的 setup 方法
        for i in range(10):
            missing_data = np.random.randn(self.N)  # 创建包含随机数据的数组
            missing_data[missing_data < 0] = np.nan  # 将数组中小于零的值设为 NaN
            self.df[f"missing_{i}"] = missing_data  # 将处理后的数据添加到 DataFrame 中
        self.df.to_stata(self.fname, convert_dates=self.convert_dates)  # 将更新后的 DataFrame 写入 Stata 文件

from ..pandas_vb_common import setup  # 导入自定义模块中的 setup 函数，忽略 isort 排序警告
```