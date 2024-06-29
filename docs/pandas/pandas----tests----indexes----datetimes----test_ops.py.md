# `D:\src\scipysrc\pandas\pandas\tests\indexes\datetimes\test_ops.py`

```
from datetime import datetime  # 导入 datetime 模块中的 datetime 类

import pytest  # 导入 pytest 测试框架

from pandas import (  # 从 pandas 库中导入以下内容：
    DatetimeIndex,  # DatetimeIndex 类，用于处理日期时间索引
    Index,  # Index 类，用于一般索引操作
    bdate_range,  # bdate_range 函数，生成工作日日期范围
    date_range,  # date_range 函数，生成日期范围
)
import pandas._testing as tm  # 导入 pandas 内部测试模块


class TestDatetimeIndexOps:  # 定义测试类 TestDatetimeIndexOps
    def test_infer_freq(self, freq_sample):  # 定义测试方法 test_infer_freq，使用参数 freq_sample
        # GH 11018
        idx = date_range("2011-01-01 09:00:00", freq=freq_sample, periods=10)
        # 创建日期时间索引 idx，从 "2011-01-01 09:00:00" 开始，使用给定频率 freq_sample，长度为 10
        result = DatetimeIndex(idx.asi8, freq="infer")
        # 使用 idx 的纳秒级整数值创建 DatetimeIndex，推断频率为 "infer"
        tm.assert_index_equal(idx, result)  # 断言 idx 与 result 相等
        assert result.freq == freq_sample  # 断言 result 的频率与 freq_sample 相同


@pytest.mark.parametrize("freq", ["B", "C"])  # 参数化测试用例，freq 可取值 "B" 和 "C"
class TestBusinessDatetimeIndex:  # 定义测试类 TestBusinessDatetimeIndex
    @pytest.fixture  # 使用 pytest 的 fixture 装饰器
    def rng(self, freq):  # 定义 rng 方法，使用参数 freq
        START, END = datetime(2009, 1, 1), datetime(2010, 1, 1)
        # 定义 START 和 END 两个 datetime 对象
        return bdate_range(START, END, freq=freq)
        # 返回从 START 到 END 的工作日日期范围，使用给定频率 freq

    def test_comparison(self, rng):  # 定义测试方法 test_comparison，使用 rng 作为参数
        d = rng[10]  # 获取 rng 的第 11 个日期

        comp = rng > d  # 比较 rng 中每个日期是否大于 d
        assert comp[11]  # 断言第 12 个元素为 True
        assert not comp[9]  # 断言第 10 个元素为 False

    def test_copy(self, rng):  # 定义测试方法 test_copy，使用 rng 作为参数
        cp = rng.copy()  # 复制 rng 生成 cp
        tm.assert_index_equal(cp, rng)  # 断言 cp 与 rng 相等

    def test_identical(self, rng):  # 定义测试方法 test_identical，使用 rng 作为参数
        t1 = rng.copy()  # 复制 rng 生成 t1
        t2 = rng.copy()  # 再次复制 rng 生成 t2
        assert t1.identical(t2)  # 断言 t1 与 t2 是相同的

        # name
        t1 = t1.rename("foo")  # 将 t1 的名称重命名为 "foo"
        assert t1.equals(t2)  # 断言 t1 与 t2 是相等的
        assert not t1.identical(t2)  # 断言 t1 与 t2 不是相同的
        t2 = t2.rename("foo")  # 将 t2 的名称重命名为 "foo"
        assert t1.identical(t2)  # 断言 t1 与 t2 是相同的

        # freq
        t2v = Index(t2.values)  # 创建 t2 值的索引对象 t2v
        assert t1.equals(t2v)  # 断言 t1 与 t2v 是相等的
        assert not t1.identical(t2v)  # 断言 t1 与 t2v 不是相同的
```