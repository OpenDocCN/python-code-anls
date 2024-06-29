# `D:\src\scipysrc\pandas\asv_bench\benchmarks\io\pickle.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算

from pandas import (  # 从 Pandas 库中导入以下模块：
    DataFrame,      # 数据框结构
    Index,          # 索引对象
    date_range,     # 日期范围生成函数
    read_pickle,    # 读取 pickle 文件的函数
)

from ..pandas_vb_common import BaseIO  # 从相对路径的 pandas_vb_common 模块中导入 BaseIO 类


class Pickle(BaseIO):  # 定义 Pickle 类，继承自 BaseIO 类
    def setup(self):
        self.fname = "__test__.pkl"  # 设置文件名为 "__test__.pkl"
        N = 100000  # 设置数据行数为 100000
        C = 5  # 设置数据列数为 5
        self.df = DataFrame(  # 创建 DataFrame 对象 self.df
            np.random.randn(N, C),  # 用随机数填充大小为 N 行 C 列的 DataFrame
            columns=[f"float{i}" for i in range(C)],  # 设置列名为 "float0" 至 "float4"
            index=date_range("20000101", periods=N, freq="h"),  # 设置索引为从 "20000101" 开始的 N 个小时频率的日期范围
        )
        self.df["object"] = Index([f"i-{i}" for i in range(N)], dtype=object)  # 在 DataFrame 中添加名为 "object" 的列，内容为 "i-0" 至 "i-99999" 的对象类型索引
        self.df.to_pickle(self.fname)  # 将 DataFrame 对象保存为 pickle 文件

    def time_read_pickle(self):
        read_pickle(self.fname)  # 读取指定的 pickle 文件

    def time_write_pickle(self):
        self.df.to_pickle(self.fname)  # 将 DataFrame 对象再次保存为 pickle 文件

    def peakmem_read_pickle(self):
        read_pickle(self.fname)  # 读取指定的 pickle 文件（与 time_read_pickle 方法相同）

    def peakmem_write_pickle(self):
        self.df.to_pickle(self.fname)  # 将 DataFrame 对象再次保存为 pickle 文件（与 time_write_pickle 方法相同）


from ..pandas_vb_common import setup  # 从相对路径的 pandas_vb_common 模块中导入 setup 函数（忽略 isort 排序警告）
```