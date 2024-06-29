# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_copy.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数组和数据
import pytest  # 导入 pytest 库，用于编写和运行测试

from pandas import (  # 从 pandas 库中导入 Series 和 Timestamp 类
    Series,
    Timestamp,
)
import pandas._testing as tm  # 导入 pandas 内部测试模块

class TestCopy:
    @pytest.mark.parametrize("deep", ["default", None, False, True])
    def test_copy(self, deep):
        ser = Series(np.arange(10), dtype="float64")  # 创建一个包含浮点数的 Series 对象

        # 默认情况下 deep 参数为 True
        if deep == "default":
            ser2 = ser.copy()  # 进行深拷贝
        else:
            ser2 = ser.copy(deep=deep)  # 根据 deep 参数选择是否进行深拷贝

        # INFO(CoW) 浅拷贝并不复制数据，但父对象不会被修改（CoW）
        if deep is None or deep is False:
            assert np.may_share_memory(ser.values, ser2.values)  # 断言数据是否共享内存
        else:
            assert not np.may_share_memory(ser.values, ser2.values)  # 断言数据不共享内存

        ser2[::2] = np.nan  # 修改拷贝后的 Series 的部分数据为 NaN

        # 没有修改原始 Series 的数据
        assert np.isnan(ser2[0])
        assert not np.isnan(ser[0])

    @pytest.mark.filterwarnings("ignore:Setting a value on a view:FutureWarning")
    @pytest.mark.parametrize("deep", ["default", None, False, True])
    def test_copy_tzaware(self, deep):
        # GH#11794
        # copy of tz-aware
        expected = Series([Timestamp("2012/01/01", tz="UTC")])  # 创建一个具有时区信息的 Series 对象
        expected2 = Series([Timestamp("1999/01/01", tz="UTC")])  # 创建另一个具有不同时间戳的 Series 对象

        ser = Series([Timestamp("2012/01/01", tz="UTC")])  # 创建具有时区信息的 Series 对象

        if deep == "default":
            ser2 = ser.copy()  # 进行深拷贝
        else:
            ser2 = ser.copy(deep=deep)  # 根据 deep 参数选择是否进行深拷贝

        # INFO(CoW) 浅拷贝并不复制数据，但父对象不会被修改（CoW）
        if deep is None or deep is False:
            assert np.may_share_memory(ser.values, ser2.values)  # 断言数据是否共享内存
        else:
            assert not np.may_share_memory(ser.values, ser2.values)  # 断言数据不共享内存

        ser2[0] = Timestamp("1999/01/01", tz="UTC")  # 修改拷贝后的 Series 的第一个时间戳为另一个时间

        # 没有修改原始 Series 的数据
        tm.assert_series_equal(ser2, expected2)
        tm.assert_series_equal(ser, expected)

    def test_copy_name(self, datetime_series):
        result = datetime_series.copy()  # 复制日期时间 Series 对象
        assert result.name == datetime_series.name  # 断言复制后的对象的名称与原对象相同

    def test_copy_index_name_checking(self, datetime_series):
        # 不希望在复制后能够修改存储在别处的索引

        datetime_series.index.name = None  # 将原始 Series 的索引名称设置为 None
        assert datetime_series.index.name is None  # 断言索引名称为 None
        assert datetime_series is datetime_series  # 断言原始 Series 与其自身相同

        cp = datetime_series.copy()  # 复制日期时间 Series 对象
        cp.index.name = "foo"  # 修改复制后对象的索引名称为 "foo"
        assert datetime_series.index.name is None  # 断言原始 Series 的索引名称仍为 None
```