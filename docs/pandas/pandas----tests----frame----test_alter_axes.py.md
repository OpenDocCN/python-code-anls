# `D:\src\scipysrc\pandas\pandas\tests\frame\test_alter_axes.py`

```
from datetime import (
    datetime,
    timezone,
)

from pandas import DataFrame
import pandas._testing as tm


class TestDataFrameAlterAxes:
    # Tests for setting index/columns attributes directly (i.e. __setattr__)

    def test_set_axis_setattr_index(self):
        # GH 6785
        # set the index manually
        
        # 创建一个包含一个字典元素的DataFrame，包括一个带有时区信息的datetime对象和一个整数值
        df = DataFrame([{"ts": datetime(2014, 4, 1, tzinfo=timezone.utc), "foo": 1}])
        # 期望的结果是将'ts'列设置为索引
        expected = df.set_index("ts")
        # 将索引设置为'ts'列的值，然后删除'ts'列
        df.index = df["ts"]
        df.pop("ts")
        # 断言DataFrame对象df和期望的DataFrame对象expected是否相等
        tm.assert_frame_equal(df, expected)

    # Renaming

    def test_assign_columns(self, float_frame):
        # 给DataFrame对象float_frame新增一个名为'hi'的列，值为字符串"there"
        float_frame["hi"] = "there"

        # 复制float_frame创建一个新的DataFrame对象df
        df = float_frame.copy()
        # 重命名DataFrame对象df的列名
        df.columns = ["foo", "bar", "baz", "quux", "foo2"]
        # 断言float_frame的'C'列和df的'baz'列是否相等，不检查列名
        tm.assert_series_equal(float_frame["C"], df["baz"], check_names=False)
        # 断言float_frame的'hi'列和df的'foo2'列是否相等，不检查列名
        tm.assert_series_equal(float_frame["hi"], df["foo2"], check_names=False)
```