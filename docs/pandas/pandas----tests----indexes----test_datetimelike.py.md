# `D:\src\scipysrc\pandas\pandas\tests\indexes\test_datetimelike.py`

```
"""generic datetimelike tests"""

import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库，用于数据分析
import pandas._testing as tm  # 导入 Pandas 内部测试模块

class TestDatetimeLike:
    @pytest.fixture(
        params=[
            pd.period_range("20130101", periods=5, freq="D"),  # 创建一个周期范围的 PeriodIndex 对象
            pd.TimedeltaIndex(
                [
                    "0 days 01:00:00",
                    "1 days 01:00:00",
                    "2 days 01:00:00",
                    "3 days 01:00:00",
                    "4 days 01:00:00",
                ],  # 创建一个 TimedeltaIndex 对象，表示时间间隔
                dtype="timedelta64[ns]",
                freq="D",
            ),
            pd.DatetimeIndex(
                ["2013-01-01", "2013-01-02", "2013-01-03", "2013-01-04", "2013-01-05"],  # 创建一个 DatetimeIndex 对象，表示日期时间
                dtype="datetime64[ns]",
                freq="D",
            ),
        ]
    )
    def simple_index(self, request):
        return request.param  # 返回测试数据集合中的每一个参数对象作为简单索引的值

    def test_isin(self, simple_index):
        index = simple_index[:4]
        result = index.isin(index)  # 判断索引对象是否包含在给定的索引对象中
        assert result.all()

        result = index.isin(list(index))  # 使用列表形式判断索引对象是否包含在给定列表中
        assert result.all()

        result = index.isin([index[2], 5])  # 判断索引对象是否包含在给定的列表中，包含一个索引对象和一个数值
        expected = np.array([False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)  # 断言两个 NumPy 数组是否相等

    def test_argsort_matches_array(self, simple_index):
        idx = simple_index
        idx = idx.insert(1, pd.NaT)  # 在索引对象中插入 pd.NaT (Not a Time)，表示缺失值

        result = idx.argsort()  # 返回排序后的索引对象中元素的位置索引
        expected = idx._data.argsort()  # 返回底层数据结构排序后的位置索引
        tm.assert_numpy_array_equal(result, expected)  # 断言两个 NumPy 数组是否相等

    def test_can_hold_identifiers(self, simple_index):
        idx = simple_index
        key = idx[0]
        assert idx._can_hold_identifiers_and_holds_name(key) is False  # 断言索引对象是否能够容纳标识符，并且保留名称为 key 的标识符名为 False

    def test_shift_identity(self, simple_index):
        idx = simple_index
        tm.assert_index_equal(idx, idx.shift(0))  # 断言索引对象与其移位零位置的索引对象是否相等

    def test_shift_empty(self, simple_index):
        # GH#14811
        idx = simple_index[:0]
        tm.assert_index_equal(idx, idx.shift(1))  # 断言空的索引对象与其移位 1 位置后的索引对象是否相等

    def test_str(self, simple_index):
        # test the string repr
        idx = simple_index.copy()
        idx.name = "foo"
        assert f"length={len(idx)}" not in str(idx)  # 断言字符串表示中不包含长度信息
        assert "'foo'" in str(idx)  # 断言字符串表示中包含名称为 'foo'
        assert type(idx).__name__ in str(idx)  # 断言字符串表示中包含索引对象类型的名称

        if hasattr(idx, "tz"):  # 如果索引对象有时区属性
            if idx.tz is not None:
                assert idx.tz in str(idx)  # 断言字符串表示中包含时区信息
        if isinstance(idx, pd.PeriodIndex):  # 如果索引对象是 PeriodIndex 类型
            assert f"dtype='period[{idx.freqstr}]'" in str(idx)  # 断言字符串表示中包含周期索引的数据类型信息
        else:
            assert f"freq='{idx.freqstr}'" in str(idx)  # 断言字符串表示中包含频率信息

    def test_view(self, simple_index):
        idx = simple_index

        result = type(simple_index)(idx)
        tm.assert_index_equal(result, idx)  # 断言两个索引对象是否相等

        msg = (
            "Cannot change data-type for array of references.|"
            "Cannot change data-type for object array.|"
        )
        with pytest.raises(TypeError, match=msg):
            idx.view(type(simple_index))  # 尝试改变数组视图的数据类型，预期会抛出 TypeError 异常
    # 定义一个测试函数，用于测试索引对象的 map 方法的可调用对象
    def test_map_callable(self, simple_index):
        # 复制传入的简单索引对象
        index = simple_index
        # 计算预期结果，即索引对象中的每个值加上频率值
        expected = index + index.freq
        # 使用 map 方法对索引对象进行映射操作，将每个元素加上索引对象的频率值
        result = index.map(lambda x: x + index.freq)
        # 断言映射后的结果与预期结果相等
        tm.assert_index_equal(result, expected)

        # 将第一个元素映射为 NaT（Not a Time），其余元素保持不变
        result = index.map(lambda x: pd.NaT if x == index[0] else x)
        # 生成预期的索引对象，将第一个元素替换为 NaT，其余元素保持不变
        expected = pd.Index([pd.NaT] + index[1:].tolist())
        # 断言映射后的结果与预期结果相等
        tm.assert_index_equal(result, expected)

    # 使用 pytest 的参数化功能，定义测试函数，测试索引对象的 map 方法与字典类似的映射
    @pytest.mark.parametrize(
        "mapper",
        [
            lambda values, index: {i: e for e, i in zip(values, index)},  # 将数值和索引映射为字典
            lambda values, index: pd.Series(values, index, dtype=object),  # 使用 Series 将数值和索引映射为对象类型的 Series
        ],
    )
    # 标记忽略特定警告，该警告与 PeriodDtype\[B\] 的未来弃用相关
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    # 定义测试函数，测试索引对象的 map 方法对于字典类似的映射操作
    def test_map_dictlike(self, mapper, simple_index):
        # 复制传入的简单索引对象
        index = simple_index
        # 计算预期结果，即索引对象中的每个值加上频率值
        expected = index + index.freq

        # 对于日期时间索引或时间间隔索引，不比较频率信息
        if isinstance(expected, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            expected = expected._with_freq(None)

        # 使用 mapper 函数将预期结果映射到索引对象上
        result = index.map(mapper(expected, index))
        # 断言映射后的结果与预期结果相等
        tm.assert_index_equal(result, expected)

        # 将第一个元素映射为 NaT，其余元素保持不变
        expected = pd.Index([pd.NaT] + index[1:].tolist())
        # 使用 mapper 函数将预期结果映射到索引对象上
        result = index.map(mapper(expected, index))
        # 断言映射后的结果与预期结果相等
        tm.assert_index_equal(result, expected)

        # 空映射；这些映射为 np.nan，因为我们无法知道如何重新推断信息
        expected = pd.Index([np.nan] * len(index))
        # 使用 mapper 函数将空值映射到索引对象上
        result = index.map(mapper([], []))
        # 断言映射后的结果与预期结果相等
        tm.assert_index_equal(result, expected)

    # 定义测试函数，测试获取索引对象的切片时是否保留频率信息
    def test_getitem_preserves_freq(self, simple_index):
        # 复制传入的简单索引对象
        index = simple_index
        # 断言索引对象的频率不为空
        assert index.freq is not None

        # 获取索引对象的全范围切片
        result = index[:]
        # 断言切片后的结果的频率与原索引对象相同
        assert result.freq == index.freq

    # 定义测试函数，测试索引对象的 where 方法是否能正确转换为字符串
    def test_where_cast_str(self, simple_index):
        # 复制传入的简单索引对象
        index = simple_index

        # 创建一个全为 True 的掩码数组
        mask = np.ones(len(index), dtype=bool)
        # 将掩码数组的最后一个元素设为 False
        mask[-1] = False

        # 将索引对象中符合掩码条件的元素映射为字符串表示，不符合条件的保持不变
        result = index.where(mask, str(index[0]))
        # 根据掩码条件，生成预期的索引对象
        expected = index.where(mask, index[0])
        # 断言映射后的结果与预期结果相等
        tm.assert_index_equal(result, expected)

        # 将索引对象中符合掩码条件的元素映射为包含字符串表示的列表，不符合条件的保持不变
        result = index.where(mask, [str(index[0])])
        # 断言映射后的结果与预期结果相等
        tm.assert_index_equal(result, expected)

        # 将索引对象转换为对象类型后，将符合掩码条件的元素映射为字符串 "foo"，不符合条件的保持不变
        expected = index.astype(object).where(mask, "foo")
        result = index.where(mask, "foo")
        # 断言映射后的结果与预期结果相等
        tm.assert_index_equal(result, expected)

        # 将索引对象中符合掩码条件的元素映射为包含字符串 "foo" 的列表，不符合条件的保持不变
        result = index.where(mask, ["foo"])
        # 断言映射后的结果与预期结果相等
        tm.assert_index_equal(result, expected)

    # 定义测试函数，测试索引对象的 diff 方法是否能正确计算时间差
    def test_diff(self, unit):
        # 创建一个日期时间索引对象，指定时间单位
        dti = pd.to_datetime([10, 20, 30], unit=unit).as_unit(unit)
        # 计算时间差，间隔为 1
        result = dti.diff(1)
        # 创建预期的时间差对象
        expected = pd.to_timedelta([pd.NaT, 10, 10], unit=unit).as_unit(unit)
        # 断言计算后的结果与预期结果相等
        tm.assert_index_equal(result, expected)
```