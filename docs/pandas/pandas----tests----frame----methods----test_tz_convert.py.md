# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_tz_convert.py`

```
import zoneinfo  # 导入zoneinfo模块

import numpy as np  # 导入numpy模块
import pytest  # 导入pytest模块

from pandas import (  # 从pandas模块导入以下对象
    DataFrame,
    Index,
    MultiIndex,
    Series,
    date_range,
)
import pandas._testing as tm  # 导入pandas测试模块


class TestTZConvert:
    def test_tz_convert(self, frame_or_series):
        # 创建一个日期范围，包含200天，频率为每日，时区为"US/Eastern"
        rng = date_range(
            "1/1/2011", periods=200, freq="D", tz=zoneinfo.ZoneInfo("US/Eastern")
        )

        # 创建一个DataFrame对象，列名为"a"，索引为rng
        obj = DataFrame({"a": 1}, index=rng)
        # 使用tm.get_obj函数处理obj对象
        obj = tm.get_obj(obj, frame_or_series)

        # 创建"Europe/Berlin"时区对象
        berlin = zoneinfo.ZoneInfo("Europe/Berlin")
        # 对obj对象进行时区转换，结果保存在result中
        result = obj.tz_convert(berlin)
        # 创建一个期望的DataFrame对象，列名为"a"，索引为rng在"Europe/Berlin"时区下的转换
        expected = DataFrame({"a": 1}, rng.tz_convert(berlin))
        # 使用tm.get_obj函数处理expected对象
        expected = tm.get_obj(expected, frame_or_series)

        # 断言结果对象的索引时区关键字为"Europe/Berlin"
        assert result.index.tz.key == "Europe/Berlin"
        # 使用tm.assert_equal函数断言result与expected对象相等

        tm.assert_equal(result, expected)

    def test_tz_convert_axis1(self):
        # 创建一个日期范围，包含200天，频率为每日，时区为"US/Eastern"
        rng = date_range(
            "1/1/2011", periods=200, freq="D", tz=zoneinfo.ZoneInfo("US/Eastern")
        )

        # 创建一个DataFrame对象，列名为"a"，索引为rng
        obj = DataFrame({"a": 1}, index=rng)

        # 对obj对象进行转置
        obj = obj.T
        # 创建"Europe/Berlin"时区对象
        berlin = zoneinfo.ZoneInfo("Europe/Berlin")
        # 对obj对象在axis=1轴上进行时区转换，结果保存在result中
        result = obj.tz_convert(berlin, axis=1)
        # 断言结果对象的列时区关键字为"Europe/Berlin"
        assert result.columns.tz.key == "Europe/Berlin"

        # 创建一个期望的DataFrame对象，列名为"a"，索引为rng在"Europe/Berlin"时区下的转换
        expected = DataFrame({"a": 1}, rng.tz_convert(berlin))
        # 使用tm.assert_equal函数断言result与expected对象的转置相等
        tm.assert_equal(result, expected.T)

    def test_tz_convert_naive(self, frame_or_series):
        # 不能对非时区感知对象进行时区转换
        rng = date_range("1/1/2011", periods=200, freq="D")
        # 创建一个Series对象，所有值为1，索引为rng
        ts = Series(1, index=rng)
        # 使用frame_or_series函数处理ts对象

        with pytest.raises(TypeError, match="Cannot convert tz-naive"):
            # 断言尝试对非时区感知对象进行"US/Eastern"时区转换时会引发TypeError异常
            ts.tz_convert("US/Eastern")

    @pytest.mark.parametrize("fn", ["tz_localize", "tz_convert"])
    # 定义一个测试函数，用于测试时区转换和本地化功能，接受一个函数名称作为参数
    def test_tz_convert_and_localize(self, fn):
        # 创建两个日期范围，每个包含5天，频率为每天一次
        l0 = date_range("20140701", periods=5, freq="D")
        l1 = date_range("20140701", periods=5, freq="D")

        # 创建一个整数索引，范围是0到4
        int_idx = Index(range(5))

        # 根据传入的函数名，如果是"tz_convert"，则将日期范围转换为UTC时区
        if fn == "tz_convert":
            l0 = l0.tz_localize("UTC")
            l1 = l1.tz_localize("UTC")

        # 对于每个日期范围，预期进行时区转换或本地化后的结果
        for idx in [l0, l1]:
            l0_expected = getattr(idx, fn)("US/Pacific")
            l1_expected = getattr(idx, fn)("US/Pacific")

            # 创建一个数据框，包含5个值，索引使用l0
            df1 = DataFrame(np.ones(5), index=l0)
            # 对数据框应用指定的函数（fn），将索引转换为US/Pacific时区
            df1 = getattr(df1, fn)("US/Pacific")
            # 断言数据框的索引与预期的l0_expected相等
            tm.assert_index_equal(df1.index, l0_expected)

            # 创建一个多级索引的数据框，索引由l0和l1数组组成
            df2 = DataFrame(np.ones(5), MultiIndex.from_arrays([l0, l1]))

            # 在多级索引构建时，频率信息不会被保留
            l1_expected = l1_expected._with_freq(None)
            l0_expected = l0_expected._with_freq(None)
            l1 = l1._with_freq(None)
            l0 = l0._with_freq(None)

            # 对多级索引的数据框应用函数（fn），在level=0上进行时区转换或本地化
            df3 = getattr(df2, fn)("US/Pacific", level=0)
            # 断言数据框的第一个级别索引不等于l0
            assert not df3.index.levels[0].equals(l0)
            # 断言数据框的第一个级别索引与预期的l0_expected相等
            tm.assert_index_equal(df3.index.levels[0], l0_expected)
            # 断言数据框的第二个级别索引与l1相等
            tm.assert_index_equal(df3.index.levels[1], l1)
            # 断言数据框的第二个级别索引不等于预期的l1_expected
            assert not df3.index.levels[1].equals(l1_expected)

            # 对多级索引的数据框应用函数（fn），在level=1上进行时区转换或本地化
            df3 = getattr(df2, fn)("US/Pacific", level=1)
            # 断言数据框的第一个级别索引与l0相等
            tm.assert_index_equal(df3.index.levels[0], l0)
            # 断言数据框的第一个级别索引不等于预期的l0_expected
            assert not df3.index.levels[0].equals(l0_expected)
            # 断言数据框的第二个级别索引与预期的l1_expected相等
            tm.assert_index_equal(df3.index.levels[1], l1_expected)
            # 断言数据框的第二个级别索引不等于l1
            assert not df3.index.levels[1].equals(l1)

            # 创建一个多级索引的数据框，第一个级别使用int_idx，第二个级别使用l0
            df4 = DataFrame(np.ones(5), MultiIndex.from_arrays([int_idx, l0]))

            # TODO: untested
            # 对数据框应用函数（fn），在level=1上进行时区转换或本地化（此处代码未测试）

            # 断言数据框的第一个级别索引与l0相等
            tm.assert_index_equal(df3.index.levels[0], l0)
            # 断言数据框的第一个级别索引不等于预期的l0_expected
            assert not df3.index.levels[0].equals(l0_expected)
            # 断言数据框的第二个级别索引与预期的l1_expected相等
            tm.assert_index_equal(df3.index.levels[1], l1_expected)
            # 断言数据框的第二个级别索引不等于l1
            assert not df3.index.levels[1].equals(l1)

    # 使用pytest.mark.parametrize装饰器，参数fn接受"tz_localize"和"tz_convert"两个值
    @pytest.mark.parametrize("fn", ["tz_localize", "tz_convert"])
    # 定义一个测试函数，用于测试不良输入情况下的时区转换和本地化功能，接受一个函数名称作为参数
    def test_tz_convert_and_localize_bad_input(self, fn):
        # 创建一个整数索引，范围是0到4
        int_idx = Index(range(5))
        # 创建一个日期范围，包含5天，频率为每天一次
        l0 = date_range("20140701", periods=5, freq="D")
        
        # 对于不是DatetimeIndex或PeriodIndex的数据框，使用指定的函数（fn），预期会引发TypeError异常
        df = DataFrame(index=int_idx)
        with pytest.raises(TypeError, match="DatetimeIndex"):
            getattr(df, fn)("US/Pacific")

        # 对于不是DatetimeIndex或PeriodIndex的多级索引数据框，使用指定的函数（fn），预期会引发TypeError异常
        df = DataFrame(np.ones(5), MultiIndex.from_arrays([int_idx, l0]))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            getattr(df, fn)("US/Pacific", level=0)

        # 对于无效的级别（level=1），使用指定的函数（fn），预期会引发ValueError异常
        df = DataFrame(index=l0)
        with pytest.raises(ValueError, match="not valid"):
            getattr(df, fn)("US/Pacific", level=1)
    # 定义一个测试方法，用于测试时区转换后是否正确地影响原对象
    def test_tz_convert_copy_inplace_mutate(self, frame_or_series):
        # GH#6326：引用GitHub上的issue编号，说明这段代码是为了解决该问题
        obj = frame_or_series(
            np.arange(0, 5),
            index=date_range("20131027", periods=5, freq="h", tz="Europe/Berlin"),
        )
        # 备份原始对象
        orig = obj.copy()
        # 对象执行时区转换为UTC，并将结果保存在result中
        result = obj.tz_convert("UTC")
        # 创建预期结果，验证转换后对象的索引也应转换为UTC时区
        expected = frame_or_series(np.arange(0, 5), index=obj.index.tz_convert("UTC"))
        # 使用测试框架中的方法验证结果是否符合预期
        tm.assert_equal(result, expected)
        # 验证原始对象在转换后未被修改
        tm.assert_equal(obj, orig)
        # 使用普通断言检查result的索引不应该与obj的索引相同
        assert result.index is not obj.index
        # 使用普通断言检查result对象不应该与obj对象是同一个实例
        assert result is not obj
```