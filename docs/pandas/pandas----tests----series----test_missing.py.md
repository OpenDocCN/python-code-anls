# `D:\src\scipysrc\pandas\pandas\tests\series\test_missing.py`

```
    from datetime import timedelta  # 导入时间间隔 timedelta 类

    import numpy as np  # 导入 NumPy 库，并简写为 np
    import pytest  # 导入 pytest 测试框架

    from pandas._libs import iNaT  # 从 pandas 库内部导入 iNaT

    import pandas as pd  # 导入 Pandas 库，并简写为 pd
    from pandas import (  # 从 Pandas 导入以下对象
        Categorical,  # 有序分类数据类型 Categorical
        Index,  # 索引对象 Index
        NaT,  # "不是时间" 标记 NaT
        Series,  # 序列对象 Series
        isna,  # 判断是否为缺失值的函数 isna
    )
    import pandas._testing as tm  # 导入 Pandas 测试模块，并简写为 tm


    class TestSeriesMissingData:
        def test_categorical_nan_handling(self):
            # NaN 在分类数据中用 -1 表示
            s = Series(Categorical(["a", "b", np.nan, "a"]))
            tm.assert_index_equal(s.cat.categories, Index(["a", "b"]))  # 断言分类对象的索引
            tm.assert_numpy_array_equal(
                s.values.codes, np.array([0, 1, -1, 0], dtype=np.int8)
            )  # 断言分类对象的代码值数组

        def test_timedelta64_nan(self):
            td = Series([timedelta(days=i) for i in range(10)])  # 创建包含时间间隔的序列对象

            # 对时间间隔进行 NaN 操作
            td1 = td.copy()
            td1[0] = np.nan
            assert isna(td1[0])  # 断言第一个元素是否为 NaN
            assert td1[0]._value == iNaT  # 断言第一个元素的值为 iNaT
            td1[0] = td[0]
            assert not isna(td1[0])  # 断言第一个元素不是 NaN

            # GH#16674 当用户提供 iNaT 时，iNaT 被视为整数
            with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
                td1[1] = iNaT
            assert not isna(td1[1])  # 断言第二个元素不是 NaN
            assert td1.dtype == np.object_  # 断言 td1 的数据类型为 np.object_
            assert td1[1] == iNaT  # 断言第二个元素的值为 iNaT
            td1[1] = td[1]
            assert not isna(td1[1])  # 断言第二个元素不是 NaN

            td1[2] = NaT
            assert isna(td1[2])  # 断言第三个元素是 NaN
            assert td1[2]._value == iNaT  # 断言第三个元素的值为 iNaT
            td1[2] = td[2]
            assert not isna(td1[2])  # 断言第三个元素不是 NaN

            # boolean setting
            # GH#2899 boolean setting
            td3 = np.timedelta64(timedelta(days=3))
            td7 = np.timedelta64(timedelta(days=7))
            td[(td > td3) & (td < td7)] = np.nan  # 根据条件将符合条件的时间间隔设置为 NaN
            assert isna(td).sum() == 3  # 断言 td 中 NaN 的数量为 3

        @pytest.mark.xfail(
            reason="Chained inequality raises when trying to define 'selector'"
        )
        def test_logical_range_select(self, datetime_series):
            # NumPy limitation =(
            # https://github.com/pandas-dev/pandas/commit/9030dc021f07c76809848925cb34828f6c8484f3

            selector = -0.5 <= datetime_series <= 0.5  # 创建选择器，判断是否在范围内
            expected = (datetime_series >= -0.5) & (datetime_series <= 0.5)  # 期望的结果
            tm.assert_series_equal(selector, expected)  # 断言选择器与期望结果相等

        def test_valid(self, datetime_series):
            ts = datetime_series.copy()  # 复制时间序列对象
            ts.index = ts.index._with_freq(None)  # 清除索引的频率信息
            ts[::2] = np.nan  # 每隔两个元素设置为 NaN

            result = ts.dropna()  # 删除 NaN 值后的结果
            assert len(result) == ts.count()  # 断言结果长度与非 NaN 值数量相等
            tm.assert_series_equal(result, ts[1::2])  # 断言结果与每隔一个元素的原始序列相等
            tm.assert_series_equal(result, ts[pd.notna(ts)])  # 断言结果与非 NaN 值的原始序列相等


    def test_hasnans_uncached_for_series():
        # GH#19700
        # set float64 dtype to avoid upcast when setting nan
        idx = Index([0, 1], dtype="float64")  # 创建浮点类型索引对象，避免 NaN 设置时的上升转换
        assert idx.hasnans is False  # 断言索引对象不含 NaN
        assert "hasnans" in idx._cache  # 断言索引对象的缓存中包含 "hasnans"
        ser = idx.to_series()  # 将索引对象转换为系列对象
        assert ser.hasnans is False  # 断言系列对象不含 NaN
        assert not hasattr(ser, "_cache")  # 断言系列对象没有 _cache 属性
        ser.iloc[-1] = np.nan  # 设置最后一个元素为 NaN
        assert ser.hasnans is True  # 断言系列对象含有 NaN
```