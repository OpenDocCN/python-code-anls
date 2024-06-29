# `D:\src\scipysrc\pandas\asv_bench\benchmarks\index_cached_properties.py`

```
# 导入 pandas 库，用于处理数据和索引
import pandas as pd

# 定义 IndexCache 类，用于缓存不同类型的索引
class IndexCache:
    # 类级别的变量，用于测试
    number = 1
    repeat = (3, 100, 20)

    # 参数列表，包含不同类型的索引
    params = [
        [
            "CategoricalIndex",
            "DatetimeIndex",
            "Float64Index",
            "IntervalIndex",
            "Int64Index",
            "MultiIndex",
            "PeriodIndex",
            "RangeIndex",
            "TimedeltaIndex",
            "UInt64Index",
        ]
    ]
    # 参数名称列表，用于标识参数
    param_names = ["index_type"]

    # 设置方法，根据不同的索引类型初始化索引对象
    def setup(self, index_type):
        N = 10**5
        # 根据索引类型选择不同的初始化方法
        if index_type == "MultiIndex":
            self.idx = pd.MultiIndex.from_product(
                [pd.date_range("1/1/2000", freq="min", periods=N // 2), ["a", "b"]]
            )
        elif index_type == "DatetimeIndex":
            self.idx = pd.date_range("1/1/2000", freq="min", periods=N)
        elif index_type == "Int64Index":
            self.idx = pd.Index(range(N), dtype="int64")
        elif index_type == "PeriodIndex":
            self.idx = pd.period_range("1/1/2000", freq="min", periods=N)
        elif index_type == "RangeIndex":
            self.idx = pd.RangeIndex(start=0, stop=N)
        elif index_type == "IntervalIndex":
            self.idx = pd.IntervalIndex.from_arrays(range(N), range(1, N + 1))
        elif index_type == "TimedeltaIndex":
            self.idx = pd.TimedeltaIndex(range(N))
        elif index_type == "Float64Index":
            self.idx = pd.Index(range(N), dtype="float64")
        elif index_type == "UInt64Index":
            self.idx = pd.Index(range(N), dtype="uint64")
        elif index_type == "CategoricalIndex":
            self.idx = pd.CategoricalIndex(range(N), range(N))
        else:
            # 如果索引类型不在预期范围内，则引发 ValueError 异常
            raise ValueError
        # 断言索引的长度符合预期长度 N
        assert len(self.idx) == N
        # 初始化索引对象的缓存属性
        self.idx._cache = {}

    # 返回索引值时间
    def time_values(self, index_type):
        self.idx._values

    # 返回索引形状时间
    def time_shape(self, index_type):
        self.idx.shape

    # 返回索引是否单调递减时间
    def time_is_monotonic_decreasing(self, index_type):
        self.idx.is_monotonic_decreasing

    # 返回索引是否单调递增时间
    def time_is_monotonic_increasing(self, index_type):
        self.idx.is_monotonic_increasing

    # 返回索引是否唯一时间
    def time_is_unique(self, index_type):
        self.idx.is_unique

    # 返回索引引擎时间
    def time_engine(self, index_type):
        self.idx._engine

    # 返回推断的索引类型时间
    def time_inferred_type(self, index_type):
        self.idx.inferred_type
```