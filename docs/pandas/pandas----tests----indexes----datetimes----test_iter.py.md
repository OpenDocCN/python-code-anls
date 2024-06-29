# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\test_iter.py`

```
import dateutil.tz  # 导入日期时间处理时区的模块
import numpy as np  # 导入数值计算库 numpy
import pytest  # 导入测试框架 pytest

from pandas import (  # 从 pandas 库中导入以下模块
    DatetimeIndex,  # 日期时间索引类
    date_range,  # 生成日期范围的函数
    to_datetime,  # 将输入转换为 datetime 类型的函数
)
from pandas.core.arrays import datetimes  # 导入 pandas 核心模块中的 datetimes

class TestDatetimeIndexIteration:  # 定义日期时间索引迭代的测试类

    @pytest.mark.parametrize(  # 使用 pytest 参数化装饰器标记测试函数
        "tz", [None, "UTC", "US/Central", dateutil.tz.tzoffset(None, -28800)]
    )
    def test_iteration_preserves_nanoseconds(self, tz):  # 测试保留纳秒精度的迭代功能
        # GH#19603
        index = DatetimeIndex(  # 创建日期时间索引对象
            ["2018-02-08 15:00:00.168456358", "2018-02-08 15:00:00.168456359"], tz=tz
        )
        for i, ts in enumerate(index):  # 遍历索引对象
            assert ts == index[i]  # 断言每个时间戳与索引中对应位置的时间戳相等

    def test_iter_readonly(self):  # 测试只读数组下的迭代功能
        # GH#28055 ints_to_pydatetime with readonly array
        arr = np.array([np.datetime64("2012-02-15T12:00:00.000000000")])  # 创建只包含一个日期时间的 numpy 数组
        arr.setflags(write=False)  # 将数组设置为只读
        dti = to_datetime(arr)  # 将数组转换为 pandas 的日期时间索引对象
        list(dti)  # 将日期时间索引对象转换为列表

    def test_iteration_preserves_tz(self):  # 测试保留时区信息的迭代功能
        # see GH#8890
        index = date_range("2012-01-01", periods=3, freq="h", tz="US/Eastern")  # 创建带时区信息的日期时间索引

        for i, ts in enumerate(index):  # 遍历索引对象
            result = ts  # 当前迭代的时间戳
            expected = index[i]  # 期望的时间戳
            assert result == expected  # 断言当前时间戳与期望的时间戳相等

    def test_iteration_preserves_tz2(self):  # 测试保留时区信息的另一种情况下的迭代功能
        index = date_range(  # 创建带特定时区偏移的日期时间索引
            "2012-01-01", periods=3, freq="h", tz=dateutil.tz.tzoffset(None, -28800)
        )

        for i, ts in enumerate(index):  # 遍历索引对象
            result = ts  # 当前迭代的时间戳
            expected = index[i]  # 期望的时间戳
            assert result._repr_base == expected._repr_base  # 断言当前时间戳的基本表示与期望的相同
            assert result == expected  # 断言当前时间戳与期望的时间戳相等

    def test_iteration_preserves_tz3(self):  # 测试保留带时区信息的日期时间索引的迭代功能
        # GH#9100
        index = DatetimeIndex(  # 创建带时区信息的日期时间索引对象
            ["2014-12-01 03:32:39.987000-08:00", "2014-12-01 04:12:34.987000-08:00"]
        )
        for i, ts in enumerate(index):  # 遍历索引对象
            result = ts  # 当前迭代的时间戳
            expected = index[i]  # 期望的时间戳
            assert result._repr_base == expected._repr_base  # 断言当前时间戳的基本表示与期望的相同
            assert result == expected  # 断言当前时间戳与期望的时间戳相等

    @pytest.mark.parametrize("offset", [-5, -1, 0, 1])  # 使用 pytest 参数化装饰器标记测试函数
    def test_iteration_over_chunksize(self, offset, monkeypatch):  # 测试在指定块大小下的迭代功能
        # GH#21012
        chunksize = 5  # 块大小设定为 5
        index = date_range(  # 创建日期时间索引对象
            "2000-01-01 00:00:00", periods=chunksize - offset, freq="min"
        )
        num = 0  # 计数器初始化为 0
        with monkeypatch.context() as m:  # 使用 monkeypatch 上下文管理器
            m.setattr(datetimes, "_ITER_CHUNKSIZE", chunksize)  # 修改块大小属性
            for stamp in index:  # 遍历索引对象
                assert index[num] == stamp  # 断言当前索引位置的时间戳与迭代的时间戳相等
                num += 1  # 计数器加一
        assert num == len(index)  # 最终断言计数器与索引长度相等
```