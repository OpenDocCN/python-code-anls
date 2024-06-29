# `D:\src\scipysrc\pandas\pandas\tests\scalar\timestamp\methods\test_normalize.py`

```
import pytest  # 导入 pytest 测试框架

from pandas._libs.tslibs import Timestamp  # 导入 pandas 中的 Timestamp 类
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit  # 导入 pandas 中的 NpyDatetimeUnit 枚举类型


class TestTimestampNormalize:  # 定义测试类 TestTimestampNormalize

    @pytest.mark.parametrize("arg", ["2013-11-30", "2013-11-30 12:00:00"])
    def test_normalize(self, tz_naive_fixture, arg, unit):  # 定义测试方法 test_normalize，使用参数化测试
        tz = tz_naive_fixture  # 使用 tz_naive_fixture 设置时区 tz
        ts = Timestamp(arg, tz=tz).as_unit(unit)  # 创建 Timestamp 对象 ts，并转换为指定的时间单位 unit
        result = ts.normalize()  # 对 ts 进行标准化处理
        expected = Timestamp("2013-11-30", tz=tz)  # 创建预期的 Timestamp 对象 expected
        assert result == expected  # 断言结果 result 应与 expected 相等
        assert result._creso == getattr(NpyDatetimeUnit, f"NPY_FR_{unit}").value  # 断言 result 的 _creso 属性与 NpyDatetimeUnit 中对应时间单位的值相等

    def test_normalize_pre_epoch_dates(self):
        # GH: 36294  # 注释：指出这个测试用例是为了解决 GitHub 上的问题编号 36294
        result = Timestamp("1969-01-01 09:00:00").normalize()  # 创建 Timestamp 对象并进行标准化处理
        expected = Timestamp("1969-01-01 00:00:00")  # 创建预期的 Timestamp 对象 expected
        assert result == expected  # 断言结果 result 应与 expected 相等
```