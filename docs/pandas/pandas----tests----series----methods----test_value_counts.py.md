# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_value_counts.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库，并使用别名 pd
from pandas import (  # 导入 Pandas 中的特定模块
    Categorical,
    CategoricalIndex,
    Index,
    Series,
)
import pandas._testing as tm  # 导入 Pandas 内部的测试工具模块


class TestSeriesValueCounts:
    def test_value_counts_datetime(self, unit):
        # 定义日期时间数据列表
        values = [
            pd.Timestamp("2011-01-01 09:00"),
            pd.Timestamp("2011-01-01 10:00"),
            pd.Timestamp("2011-01-01 11:00"),
            pd.Timestamp("2011-01-01 09:00"),
            pd.Timestamp("2011-01-01 09:00"),
            pd.Timestamp("2011-01-01 11:00"),
        ]

        # 预期的日期时间索引，转换为指定单位（unit）
        exp_idx = pd.DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 11:00", "2011-01-01 10:00"],
            name="xxx",
        ).as_unit(unit)
        # 预期的计数 Series 对象
        exp = Series([3, 2, 1], index=exp_idx, name="count")

        # 创建 Series 对象，并将日期时间转换为指定单位（unit）
        ser = Series(values, name="xxx").dt.as_unit(unit)
        # 验证 ser 的值计数是否与预期的 exp 相等
        tm.assert_series_equal(ser.value_counts(), exp)
        # 检查 DatetimeIndex 是否输出相同的结果
        idx = pd.DatetimeIndex(values, name="xxx").as_unit(unit)
        tm.assert_series_equal(idx.value_counts(), exp)

        # 计算归一化后的预期结果
        exp = Series(np.array([3.0, 2.0, 1]) / 6.0, index=exp_idx, name="proportion")
        # 验证归一化后的值计数是否与预期的 exp 相等
        tm.assert_series_equal(ser.value_counts(normalize=True), exp)
        tm.assert_series_equal(idx.value_counts(normalize=True), exp)

    def test_value_counts_datetime_tz(self, unit):
        # 定义带时区信息的日期时间数据列表
        values = [
            pd.Timestamp("2011-01-01 09:00", tz="US/Eastern"),
            pd.Timestamp("2011-01-01 10:00", tz="US/Eastern"),
            pd.Timestamp("2011-01-01 11:00", tz="US/Eastern"),
            pd.Timestamp("2011-01-01 09:00", tz="US/Eastern"),
            pd.Timestamp("2011-01-01 09:00", tz="US/Eastern"),
            pd.Timestamp("2011-01-01 11:00", tz="US/Eastern"),
        ]

        # 预期的带时区信息的日期时间索引，转换为指定单位（unit）
        exp_idx = pd.DatetimeIndex(
            ["2011-01-01 09:00", "2011-01-01 11:00", "2011-01-01 10:00"],
            tz="US/Eastern",
            name="xxx",
        ).as_unit(unit)
        # 预期的计数 Series 对象
        exp = Series([3, 2, 1], index=exp_idx, name="count")

        # 创建带时区信息的 Series 对象，并将日期时间转换为指定单位（unit）
        ser = Series(values, name="xxx").dt.as_unit(unit)
        # 验证 ser 的值计数是否与预期的 exp 相等
        tm.assert_series_equal(ser.value_counts(), exp)
        # 创建带时区信息的 DatetimeIndex 对象，并将日期时间转换为指定单位（unit）
        idx = pd.DatetimeIndex(values, name="xxx").as_unit(unit)
        tm.assert_series_equal(idx.value_counts(), exp)

        # 计算归一化后的预期结果
        exp = Series(np.array([3.0, 2.0, 1]) / 6.0, index=exp_idx, name="proportion")
        # 验证归一化后的值计数是否与预期的 exp 相等
        tm.assert_series_equal(ser.value_counts(normalize=True), exp)
        tm.assert_series_equal(idx.value_counts(normalize=True), exp)
    def test_value_counts_period(self):
        # 创建一个包含 Period 对象的列表
        values = [
            pd.Period("2011-01", freq="M"),  # 创建一个频率为月的 Period 对象
            pd.Period("2011-02", freq="M"),
            pd.Period("2011-03", freq="M"),
            pd.Period("2011-01", freq="M"),
            pd.Period("2011-01", freq="M"),
            pd.Period("2011-03", freq="M"),
        ]

        # 创建一个期间索引对象，包含指定的 Period 对象和频率
        exp_idx = pd.PeriodIndex(
            ["2011-01", "2011-03", "2011-02"], freq="M", name="xxx"
        )
        # 创建一个期望的 Series 对象，表示每个期间出现的次数
        exp = Series([3, 2, 1], index=exp_idx, name="count")

        # 创建一个 Series 对象，用于测试
        ser = Series(values, name="xxx")
        # 断言测试 ser.value_counts() 的结果与期望的 exp 相等
        tm.assert_series_equal(ser.value_counts(), exp)

        # check DatetimeIndex outputs the same result
        # 创建一个日期时间索引对象，包含指定的 Period 对象和名称
        idx = pd.PeriodIndex(values, name="xxx")
        # 断言测试 idx.value_counts() 的结果与期望的 exp 相等
        tm.assert_series_equal(idx.value_counts(), exp)

        # normalize
        # 创建一个期望的归一化后的 Series 对象
        exp = Series(np.array([3.0, 2.0, 1]) / 6.0, index=exp_idx, name="proportion")
        # 断言测试 ser.value_counts(normalize=True) 的结果与期望的 exp 相等
        tm.assert_series_equal(ser.value_counts(normalize=True), exp)
        # 断言测试 idx.value_counts(normalize=True) 的结果与期望的 exp 相等
        tm.assert_series_equal(idx.value_counts(normalize=True), exp)

    def test_value_counts_categorical_ordered(self):
        # 创建一个有序的分类数据对象
        values = Categorical([1, 2, 3, 1, 1, 3], ordered=True)

        # 创建一个有序的分类索引对象，包含指定的分类数据和相关信息
        exp_idx = CategoricalIndex(
            [1, 3, 2], categories=[1, 2, 3], ordered=True, name="xxx"
        )
        # 创建一个期望的 Series 对象，表示每个分类出现的次数
        exp = Series([3, 2, 1], index=exp_idx, name="count")

        # 创建一个 Series 对象，用于测试
        ser = Series(values, name="xxx")
        # 断言测试 ser.value_counts() 的结果与期望的 exp 相等
        tm.assert_series_equal(ser.value_counts(), exp)

        # check CategoricalIndex outputs the same result
        # 创建一个分类索引对象，包含指定的分类数据和名称
        idx = CategoricalIndex(values, name="xxx")
        # 断言测试 idx.value_counts() 的结果与期望的 exp 相等
        tm.assert_series_equal(idx.value_counts(), exp)

        # normalize
        # 创建一个期望的归一化后的 Series 对象
        exp = Series(np.array([3.0, 2.0, 1]) / 6.0, index=exp_idx, name="proportion")
        # 断言测试 ser.value_counts(normalize=True) 的结果与期望的 exp 相等
        tm.assert_series_equal(ser.value_counts(normalize=True), exp)
        # 断言测试 idx.value_counts(normalize=True) 的结果与期望的 exp 相等
        tm.assert_series_equal(idx.value_counts(normalize=True), exp)

    def test_value_counts_categorical_not_ordered(self):
        # 创建一个无序的分类数据对象
        values = Categorical([1, 2, 3, 1, 1, 3], ordered=False)

        # 创建一个无序的分类索引对象，包含指定的分类数据和相关信息
        exp_idx = CategoricalIndex(
            [1, 3, 2], categories=[1, 2, 3], ordered=False, name="xxx"
        )
        # 创建一个期望的 Series 对象，表示每个分类出现的次数
        exp = Series([3, 2, 1], index=exp_idx, name="count")

        # 创建一个 Series 对象，用于测试
        ser = Series(values, name="xxx")
        # 断言测试 ser.value_counts() 的结果与期望的 exp 相等
        tm.assert_series_equal(ser.value_counts(), exp)

        # check CategoricalIndex outputs the same result
        # 创建一个分类索引对象，包含指定的分类数据和名称
        idx = CategoricalIndex(values, name="xxx")
        # 断言测试 idx.value_counts() 的结果与期望的 exp 相等
        tm.assert_series_equal(idx.value_counts(), exp)

        # normalize
        # 创建一个期望的归一化后的 Series 对象
        exp = Series(np.array([3.0, 2.0, 1]) / 6.0, index=exp_idx, name="proportion")
        # 断言测试 ser.value_counts(normalize=True) 的结果与期望的 exp 相等
        tm.assert_series_equal(ser.value_counts(normalize=True), exp)
        # 断言测试 idx.value_counts(normalize=True) 的结果与期望的 exp 相等
        tm.assert_series_equal(idx.value_counts(normalize=True), exp)
    # 定义一个测试函数，用于测试分类数据的值计数功能
    def test_value_counts_categorical(self):
        # GH#12835：GitHub issue编号，指出此测试用例的背景
        # 创建一个包含指定数据和类别的分类对象
        cats = Categorical(list("abcccb"), categories=list("cabd"))
        # 创建一个序列，将分类对象作为数据，并指定名称为"xxx"
        ser = Series(cats, name="xxx")
        # 对序列进行值计数，不排序
        res = ser.value_counts(sort=False)

        # 期望的索引对象，包含指定的类别和名称
        exp_index = CategoricalIndex(
            list("cabd"), categories=cats.categories, name="xxx"
        )
        # 期望的结果序列，包含计数结果和指定的索引
        exp = Series([3, 1, 2, 0], name="count", index=exp_index)
        # 断言序列是否相等
        tm.assert_series_equal(res, exp)

        # 对序列进行值计数，排序
        res = ser.value_counts(sort=True)

        # 更新期望的索引对象，重新指定类别和名称
        exp_index = CategoricalIndex(
            list("cbad"), categories=cats.categories, name="xxx"
        )
        # 更新期望的结果序列，包含计数结果和指定的索引
        exp = Series([3, 2, 1, 0], name="count", index=exp_index)
        # 断言序列是否相等
        tm.assert_series_equal(res, exp)

        # 检查对象类型处理序列名称的方式是否一致
        # 在测试中有详细的测试用例
        ser = Series(["a", "b", "c", "c", "c", "b"], name="xxx")
        # 对序列进行值计数
        res = ser.value_counts()
        # 期望的结果序列，包含计数结果和指定的索引
        exp = Series([3, 2, 1], name="count", index=Index(["c", "b", "a"], name="xxx"))
        # 断言序列是否相等
        tm.assert_series_equal(res, exp)

    # 定义一个测试函数，用于测试包含NaN值的分类数据的值计数功能
    def test_value_counts_categorical_with_nan(self):
        # see GH#9443：GitHub issue编号，指出此测试用例的背景

        # 进行基本检查
        ser = Series(["a", "b", "a"], dtype="category")
        # 期望的结果序列，包含计数结果和指定的索引
        exp = Series([2, 1], index=CategoricalIndex(["a", "b"]), name="count")

        # 对序列进行值计数，排除NaN值
        res = ser.value_counts(dropna=True)
        # 断言序列是否相等
        tm.assert_series_equal(res, exp)

        # 再次对序列进行值计数，排除NaN值
        res = ser.value_counts(dropna=True)
        # 断言序列是否相等
        tm.assert_series_equal(res, exp)

        # 通过两种不同的方式创建相同的序列 --> 应有相同的行为
        series = [
            Series(["a", "b", None, "a", None, None], dtype="category"),
            Series(
                Categorical(["a", "b", None, "a", None, None], categories=["a", "b"])
            ),
        ]

        # 遍历所有的序列
        for ser in series:
            # None被视为NaN值，因此在此处排除其计数
            exp = Series([2, 1], index=CategoricalIndex(["a", "b"]), name="count")
            # 对序列进行值计数，排除NaN值
            res = ser.value_counts(dropna=True)
            # 断言序列是否相等
            tm.assert_series_equal(res, exp)

            # 不排除NaN值，并按计数排序
            exp = Series(
                [3, 2, 1], index=CategoricalIndex([np.nan, "a", "b"]), name="count"
            )
            # 对序列进行值计数，不排除NaN值，按计数排序
            res = ser.value_counts(dropna=False)
            # 断言序列是否相等
            tm.assert_series_equal(res, exp)

            # 当不按计数排序且np.nan不是类别时，它应该排在最后
            exp = Series(
                [2, 1, 3], index=CategoricalIndex(["a", "b", np.nan]), name="count"
            )
            # 对序列进行值计数，不排除NaN值，不按计数排序
            res = ser.value_counts(dropna=False, sort=False)
            # 断言序列是否相等
            tm.assert_series_equal(res, exp)
    @pytest.mark.parametrize(
        "ser, dropna, exp",
        [  # 定义测试参数，包括序列、是否删除缺失值、预期结果
            (
                Series([False, True, True, pd.NA]),  # 第一组测试参数：包含布尔值和缺失值
                False,  # 不删除缺失值
                Series([2, 1, 1], index=[True, False, pd.NA], name="count"),  # 预期的值计数结果
            ),
            (
                Series([False, True, True, pd.NA]),  # 第二组测试参数：包含布尔值和缺失值
                True,  # 删除缺失值
                Series([2, 1], index=Index([True, False], dtype=object), name="count"),  # 预期的值计数结果
            ),
            (
                Series(range(3), index=[True, False, np.nan]).index,  # 第三组测试参数：包含整数和NaN索引
                False,  # 不删除缺失值
                Series([1, 1, 1], index=[True, False, np.nan], name="count"),  # 预期的值计数结果
            ),
        ],
    )
    def test_value_counts_bool_with_nan(self, ser, dropna, exp):
        # GH32146
        # 测试布尔序列中的值计数功能，包括NaN处理
        out = ser.value_counts(dropna=dropna)
        tm.assert_series_equal(out, exp)  # 断言实际结果与预期结果相等

    @pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
    def test_value_counts_complex_numbers(self, dtype):
        # GH 17927
        # 测试复数类型的值计数功能
        input_array = np.array([1 + 1j, 1 + 1j, 1, 3j, 3j, 3j], dtype=dtype)
        result = Series(input_array).value_counts()
        expected = Series(
            [3, 2, 1], index=Index([3j, 1 + 1j, 1], dtype=dtype), name="count"
        )
        tm.assert_series_equal(result, expected)  # 断言实际结果与预期结果相等

    def test_value_counts_masked(self):
        # GH#54984
        # 测试包含掩码值的值计数功能
        dtype = "Int64"
        ser = Series([1, 2, None, 2, None, 3], dtype=dtype)
        result = ser.value_counts(dropna=False)
        expected = Series(
            [2, 2, 1, 1],
            index=Index([2, None, 1, 3], dtype=dtype),
            dtype=dtype,
            name="count",
        )
        tm.assert_series_equal(result, expected)  # 断言实际结果与预期结果相等

        result = ser.value_counts(dropna=True)
        expected = Series(
            [2, 1, 1], index=Index([2, 1, 3], dtype=dtype), dtype=dtype, name="count"
        )
        tm.assert_series_equal(result, expected)  # 断言实际结果与预期结果相等
```