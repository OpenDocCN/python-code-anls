# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\methods\test_asof.py`

```
# 从 datetime 模块中导入 timedelta 类
from datetime import timedelta

# 从 pandas 库中导入需要使用的模块和函数
from pandas import (
    Index,          # 导入 Index 类
    Timestamp,      # 导入 Timestamp 类
    date_range,     # 导入 date_range 函数，用于生成日期范围
    isna,           # 导入 isna 函数，用于检查是否为缺失值
)


# 定义一个测试类 TestAsOf
class TestAsOf:
    # 定义测试方法 test_asof_partial
    def test_asof_partial(self):
        # 生成一个日期范围对象 index，包含 "2010-01-01" 和 "2010-01-31"
        index = date_range("2010-01-01", periods=2, freq="M")
        # 期望的结果是 Timestamp 类型的 "2010-01-31"
        expected = Timestamp("2010-01-31")
        # 对 index 使用 asof 方法，获取 "2010-02" 对应的日期，结果应与 expected 相等
        result = index.asof("2010-02")
        # 断言结果与期望相等
        assert result == expected
        # 断言 result 不是 Index 类型的对象
        assert not isinstance(result, Index)

    # 定义测试方法 test_asof
    def test_asof(self):
        # 生成一个从 "2020-01-01" 开始的连续 10 天日期范围对象 index
        index = date_range("2020-01-01", periods=10)

        # 获取 index 的第一个日期
        dt = index[0]
        # 断言 index 中最接近 dt 的日期等于 dt 本身
        assert index.asof(dt) == dt
        # 断言 index 中最接近 (dt - 1 天) 的日期是缺失值
        assert isna(index.asof(dt - timedelta(1)))

        # 获取 index 的最后一个日期
        dt = index[-1]
        # 断言 index 中最接近 (dt + 1 天) 的日期等于 dt 本身
        assert index.asof(dt + timedelta(1)) == dt

        # 将 index 的第一个日期转换为 Python 的 datetime 对象
        dt = index[0].to_pydatetime()
        # 断言 index 中最接近 dt 的日期是 Timestamp 类型的对象
        assert isinstance(index.asof(dt), Timestamp)
```