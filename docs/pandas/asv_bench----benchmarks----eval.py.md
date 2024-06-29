# `D:\src\scipysrc\pandas\asv_bench\benchmarks\eval.py`

```
# 导入 numpy 库，并使用 np 作为别名
import numpy as np

# 导入 pandas 库，并使用 pd 作为别名
import pandas as pd

# 尝试导入 pandas 库的表达式模块，根据不同版本尝试不同的路径
try:
    import pandas.core.computation.expressions as expr
except ImportError:
    import pandas.computation.expressions as expr

# 定义一个评估类 Eval
class Eval:
    # 定义类变量 params，表示评估引擎和线程数的组合
    params = [["numexpr", "python"], [1, "all"]]
    # 定义类变量 param_names，表示参数的名称
    param_names = ["engine", "threads"]

    # 设置方法，在每次测试之前初始化数据帧
    def setup(self, engine, threads):
        # 创建四个包含随机数据的数据帧
        self.df = pd.DataFrame(np.random.randn(20000, 100))
        self.df2 = pd.DataFrame(np.random.randn(20000, 100))
        self.df3 = pd.DataFrame(np.random.randn(20000, 100))
        self.df4 = pd.DataFrame(np.random.randn(20000, 100))

        # 如果线程数为 1，则设置 numexpr 的线程数为 1
        if threads == 1:
            expr.set_numexpr_threads(1)

    # 时间加法测试方法，对四个数据帧进行加法运算
    def time_add(self, engine, threads):
        pd.eval("self.df + self.df2 + self.df3 + self.df4", engine=engine)

    # 时间与运算测试方法，对四个数据帧进行逻辑与运算
    def time_and(self, engine, threads):
        pd.eval(
            "(self.df > 0) & (self.df2 > 0) & (self.df3 > 0) & (self.df4 > 0)",
            engine=engine,
        )

    # 时间链式比较测试方法，比较四个数据帧之间的大小关系
    def time_chained_cmp(self, engine, threads):
        pd.eval("self.df < self.df2 < self.df3 < self.df4", engine=engine)

    # 时间乘法测试方法，对四个数据帧进行乘法运算
    def time_mult(self, engine, threads):
        pd.eval("self.df * self.df2 * self.df3 * self.df4", engine=engine)

    # 清理方法，在测试结束后恢复 numexpr 的线程设置
    def teardown(self, engine, threads):
        expr.set_numexpr_threads()


# 查询类 Query
class Query:
    # 设置方法，在每次测试之前初始化时间序列和数据帧
    def setup(self):
        N = 10**6
        halfway = (N // 2) - 1
        index = pd.date_range("20010101", periods=N, freq="min")
        s = pd.Series(index)
        self.ts = s.iloc[halfway]
        self.df = pd.DataFrame({"a": np.random.randn(N), "dates": index}, index=index)
        data = np.random.randn(N)
        self.min_val = data.min()
        self.max_val = data.max()

    # 时间查询测试方法，使用时间索引进行查询
    def time_query_datetime_index(self):
        self.df.query("index < @self.ts")

    # 时间查询测试方法，使用日期列进行查询
    def time_query_datetime_column(self):
        self.df.query("dates < @self.ts")

    # 带布尔选择的时间查询测试方法，使用条件查询数据帧
    def time_query_with_boolean_selection(self):
        self.df.query("(a >= @self.min_val) & (a <= @self.max_val)")


# 导入 .pandas_vb_common 模块中的 setup 函数，跳过 isort 排序，忽略 F401 警告
from .pandas_vb_common import setup  # noqa: F401 isort:skip
```