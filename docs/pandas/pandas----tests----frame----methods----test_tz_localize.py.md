# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_tz_localize.py`

```
    # 导入所需的模块和类
    from datetime import timezone
    import numpy as np
    import pytest
    from pandas import (
        DataFrame,
        Series,
        date_range,
    )
    import pandas._testing as tm

    # 定义一个测试类 TestTZLocalize，用于测试时区本地化功能
    class TestTZLocalize:
        
        # 测试时区本地化功能，接受一个参数 frame_or_series 作为测试对象
        # 这个测试方法参考 test_tz_convert.py 文件中的 test_tz_convert_and_localize 方法
        def test_tz_localize(self, frame_or_series):
            # 创建一个时间索引，从 "1/1/2011" 开始，每小时一个时间点，共计 100 个时间点
            rng = date_range("1/1/2011", periods=100, freq="h")
            
            # 创建一个 DataFrame 对象 obj，包含一列名为 'a' 的数据，索引为 rng
            obj = DataFrame({"a": 1}, index=rng)
            # 从测试模块中获取处理后的 obj 对象
            obj = tm.get_obj(obj, frame_or_series)

            # 对 obj 进行时区本地化，设定时区为 "utc"
            result = obj.tz_localize("utc")
            # 创建一个预期的 DataFrame 对象 expected，内容与 result 相同，索引时区为 "UTC"
            expected = DataFrame({"a": 1}, rng.tz_localize("UTC"))
            # 从测试模块中获取处理后的 expected 对象
            expected = tm.get_obj(expected, frame_or_series)

            # 断言结果对象的索引时区为 timezone.utc
            assert result.index.tz is timezone.utc
            # 使用测试模块中的方法验证 result 和 expected 是否相等
            tm.assert_equal(result, expected)

        # 测试沿 axis=1 进行时区本地化功能
        def test_tz_localize_axis1(self):
            # 创建一个时间索引，从 "1/1/2011" 开始，每小时一个时间点，共计 100 个时间点
            rng = date_range("1/1/2011", periods=100, freq="h")

            # 创建一个 DataFrame 对象 df，包含一列名为 'a' 的数据，索引为 rng
            df = DataFrame({"a": 1}, index=rng)

            # 将 df 进行转置操作
            df = df.T
            # 对 df 按 axis=1 进行时区本地化，设定时区为 "utc"
            result = df.tz_localize("utc", axis=1)
            # 断言结果对象的列索引时区为 timezone.utc
            assert result.columns.tz is timezone.utc

            # 创建一个预期的 DataFrame 对象 expected，内容与 df 相同，索引时区为 "UTC"
            expected = DataFrame({"a": 1}, rng.tz_localize("UTC"))
            # 使用测试模块中的方法验证 result 和 expected.T 是否相等
            tm.assert_frame_equal(result, expected.T)

        # 测试在非本地化时间索引上执行时区本地化操作的情况
        def test_tz_localize_naive(self, frame_or_series):
            # 创建一个具有时区信息的时间索引，从 "1/1/2011" 开始，每小时一个时间点，共计 100 个时间点，时区为 "utc"
            rng = date_range("1/1/2011", periods=100, freq="h", tz="utc")
            # 创建一个 Series 对象 ts，包含 100 个时间点，每个时间点的数据为 1
            ts = Series(1, index=rng)
            # 从测试模块中获取处理后的 ts 对象
            ts = frame_or_series(ts)

            # 使用 pytest 模块验证在已经具有时区信息的对象上执行时区本地化操作是否会引发 TypeError 异常
            with pytest.raises(TypeError, match="Already tz-aware"):
                ts.tz_localize("US/Eastern")

        # 测试在原对象上进行时区本地化操作后，原对象是否保持不变的情况
        def test_tz_localize_copy_inplace_mutate(self, frame_or_series):
            # 创建一个 Series 或 DataFrame 对象 obj，索引为从 "20131027" 开始的五个时间点，频率为每小时一次，时区为 None
            obj = frame_or_series(
                np.arange(0, 5), index=date_range("20131027", periods=5, freq="1h", tz=None)
            )
            # 备份原始对象
            orig = obj.copy()
            # 对 obj 进行时区本地化操作，设定时区为 "UTC"
            result = obj.tz_localize("UTC")
            # 创建一个预期的 Series 或 DataFrame 对象 expected，内容与 result 相同，索引时区为 "UTC"
            expected = frame_or_series(
                np.arange(0, 5),
                index=date_range("20131027", periods=5, freq="1h", tz="UTC"),
            )
            # 使用测试模块中的方法验证 result 和 expected 是否相等
            tm.assert_equal(result, expected)
            # 使用测试模块中的方法验证 obj 和 orig 是否相等
            tm.assert_equal(obj, orig)
            # 断言 result 对象的索引与 obj 对象的索引不同
            assert result.index is not obj.index
            # 断言 result 对象与 obj 对象不是同一个对象
            assert result is not obj
```