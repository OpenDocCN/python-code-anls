# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_iterrows.py`

```
from pandas import (
    DataFrame,
    Timedelta,
)

# 定义一个测试函数，用于验证数据帧中频率和时间不会溢出
def test_no_overflow_of_freq_and_time_in_dataframe():
    # GH 35665: GitHub issue reference for tracking
    # 创建一个包含列数据的数据帧
    df = DataFrame(
        {
            "some_string": ["2222Y3"],  # 包含字符串的列
            "time": [Timedelta("0 days 00:00:00.990000")],  # 包含时间间隔的列
        }
    )
    # 遍历数据帧的每一行
    for _, row in df.iterrows():
        # 断言每一行的数据类型为 "object"
        assert row.dtype == "object"
```