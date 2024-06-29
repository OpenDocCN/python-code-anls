# `D:\src\scipysrc\pandas\pandas\tests\extension\base\missing.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于单元测试

import pandas as pd  # 导入 Pandas 库，用于数据分析
import pandas._testing as tm  # 导入 Pandas 内部测试模块

class BaseMissingTests:  # 定义一个基础的缺失数据测试类
    def test_isna(self, data_missing):  # 定义测试空值的方法，接受一个数据源作为参数
        expected = np.array([True, False])  # 创建一个期望的 NumPy 数组，表示预期的空值情况

        result = pd.isna(data_missing)  # 检查数据源中的空值情况
        tm.assert_numpy_array_equal(result, expected)  # 使用测试模块验证结果是否与期望一致

        result = pd.Series(data_missing).isna()  # 将数据源转换为 Pandas Series 后再次检查空值情况
        expected = pd.Series(expected)  # 将期望的结果转换为 Pandas Series
        tm.assert_series_equal(result, expected)  # 使用测试模块验证 Series 是否一致

        # GH 21189
        result = pd.Series(data_missing).drop([0, 1]).isna()  # 在删除指定索引后再次检查空值情况
        expected = pd.Series([], dtype=bool)  # 创建一个空的 Pandas Series 作为期望结果
        tm.assert_series_equal(result, expected)  # 使用测试模块验证 Series 是否一致

    @pytest.mark.parametrize("na_func", ["isna", "notna"])  # 使用 Pytest 的参数化标记，指定测试方法名参数化
    def test_isna_returns_copy(self, data_missing, na_func):  # 测试空值检查方法是否返回副本
        result = pd.Series(data_missing)  # 将数据源转换为 Pandas Series
        expected = result.copy()  # 创建一个数据源的副本作为期望结果
        mask = getattr(result, na_func)()  # 调用指定的空值检查方法获取掩码

        if isinstance(mask.dtype, pd.SparseDtype):  # 如果掩码的数据类型是稀疏类型
            # TODO: GH 57739
            mask = np.array(mask)  # 将稀疏掩码转换为 NumPy 数组
            mask.flags.writeable = True  # 设置数组可写

        mask[:] = True  # 将掩码数组所有元素设置为 True
        tm.assert_series_equal(result, expected)  # 使用测试模块验证 Series 是否一致

    def test_dropna_array(self, data_missing):  # 测试在数组上应用 dropna 方法
        result = data_missing.dropna()  # 对数据源应用 dropna 方法
        expected = data_missing[[1]]  # 创建一个包含非空值的数组作为期望结果
        tm.assert_extension_array_equal(result, expected)  # 使用测试模块验证 ExtensionArray 是否一致

    def test_dropna_series(self, data_missing):  # 测试在 Series 上应用 dropna 方法
        ser = pd.Series(data_missing)  # 将数据源转换为 Pandas Series
        result = ser.dropna()  # 对 Series 应用 dropna 方法
        expected = ser.iloc[[1]]  # 创建一个包含非空值的 Series 作为期望结果
        tm.assert_series_equal(result, expected)  # 使用测试模块验证 Series 是否一致

    def test_dropna_frame(self, data_missing):  # 测试在 DataFrame 上应用 dropna 方法
        df = pd.DataFrame({"A": data_missing}, columns=pd.Index(["A"], dtype=object))  # 创建一个包含数据源的 DataFrame

        # defaults
        result = df.dropna()  # 对 DataFrame 应用 dropna 方法，默认情况下按行删除缺失值
        expected = df.iloc[[1]]  # 创建一个包含非空值的 DataFrame 作为期望结果
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证 DataFrame 是否一致

        # axis = 1
        result = df.dropna(axis="columns")  # 按列删除缺失值
        expected = pd.DataFrame(index=pd.RangeIndex(2), columns=pd.Index([]))  # 创建一个空的 DataFrame 作为期望结果
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证 DataFrame 是否一致

        # multiple
        df = pd.DataFrame({"A": data_missing, "B": [1, np.nan]})  # 创建一个包含多列数据的 DataFrame
        result = df.dropna()  # 对 DataFrame 应用 dropna 方法
        expected = df.iloc[:0]  # 创建一个空的 DataFrame 作为期望结果
        tm.assert_frame_equal(result, expected)  # 使用测试模块验证 DataFrame 是否一致

    def test_fillna_scalar(self, data_missing):  # 测试使用标量填充缺失值的方法
        valid = data_missing[1]  # 从数据源中获取一个有效的标量值
        result = data_missing.fillna(valid)  # 使用有效值填充缺失值
        expected = data_missing.fillna(valid)  # 创建一个使用有效值填充缺失值的期望结果
        tm.assert_extension_array_equal(result, expected)  # 使用测试模块验证 ExtensionArray 是否一致

    def test_fillna_with_none(self, data_missing):  # 测试使用 None 填充缺失值的方法
        # GH#57723
        result = data_missing.fillna(None)  # 使用 None 填充缺失值
        expected = data_missing  # 创建一个与数据源相同的期望结果
        tm.assert_extension_array_equal(result, expected)  # 使用测试模块验证 ExtensionArray 是否一致

    def test_fillna_limit_pad(self, data_missing):  # 测试使用限制参数填充缺失值的方法
        arr = data_missing.take([1, 0, 0, 0, 1])  # 从数据源中选择部分数据作为填充源
        result = pd.Series(arr).ffill(limit=2)  # 使用向前填充方法填充缺失值，设置填充限制为 2
        expected = pd.Series(data_missing.take([1, 1, 1, 0, 1]))  # 创建一个期望的填充结果
        tm.assert_series_equal(result, expected)  # 使用测试模块验证 Series 是否一致
    @pytest.mark.parametrize(
        "limit_area, input_ilocs, expected_ilocs",
        [   # 使用 pytest 的参数化装饰器，定义多组测试参数
            ("outside", [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]),  # 测试参数组：limit_area为'outside'，input_ilocs为[1, 0, 0, 0, 1]，expected_ilocs为[1, 0, 0, 0, 1]
            ("outside", [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]),  # 测试参数组：limit_area为'outside'，input_ilocs为[1, 0, 1, 0, 1]，expected_ilocs为[1, 0, 1, 0, 1]
            ("outside", [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]),  # 测试参数组：limit_area为'outside'，input_ilocs为[0, 1, 1, 1, 0]，expected_ilocs为[0, 1, 1, 1, 1]
            ("outside", [0, 1, 0, 1, 0], [0, 1, 0, 1, 1]),  # 测试参数组：limit_area为'outside'，input_ilocs为[0, 1, 0, 1, 0]，expected_ilocs为[0, 1, 0, 1, 1]
            ("inside", [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]),    # 测试参数组：limit_area为'inside'，input_ilocs为[1, 0, 0, 0, 1]，expected_ilocs为[1, 1, 1, 1, 1]
            ("inside", [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]),    # 测试参数组：limit_area为'inside'，input_ilocs为[1, 0, 1, 0, 1]，expected_ilocs为[1, 1, 1, 1, 1]
            ("inside", [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]),    # 测试参数组：limit_area为'inside'，input_ilocs为[0, 1, 1, 1, 0]，expected_ilocs为[0, 1, 1, 1, 0]
            ("inside", [0, 1, 0, 1, 0], [0, 1, 1, 1, 0]),    # 测试参数组：limit_area为'inside'，input_ilocs为[0, 1, 0, 1, 0]，expected_ilocs为[0, 1, 1, 1, 0]
        ],
    )
    def test_ffill_limit_area(
        self, data_missing, limit_area, input_ilocs, expected_ilocs
    ):
        # GH#56616
        # 根据 GitHub 问题号 56616 进行测试的目的说明
        arr = data_missing.take(input_ilocs)  # 从 data_missing 中取出 input_ilocs 指定位置的数据，形成数组 arr
        result = pd.Series(arr).ffill(limit_area=limit_area)  # 对 arr 进行向前填充操作，根据 limit_area 参数指定填充范围
        expected = pd.Series(data_missing.take(expected_ilocs))  # 根据 expected_ilocs 从 data_missing 中取出期望的数据，形成期望的结果 expected
        tm.assert_series_equal(result, expected)  # 使用测试框架中的 assert 函数比较 result 和 expected 是否相等

    def test_fillna_limit_backfill(self, data_missing):
        arr = data_missing.take([1, 0, 0, 0, 1])  # 从 data_missing 中取出指定位置的数据，形成数组 arr
        result = pd.Series(arr).bfill(limit=2)  # 对 arr 进行向后填充操作，最多填充 2 个缺失值
        expected = pd.Series(data_missing.take([1, 0, 1, 1, 1]))  # 从 data_missing 中取出期望的数据，形成期望的结果 expected
        tm.assert_series_equal(result, expected)  # 使用测试框架中的 assert 函数比较 result 和 expected 是否相等

    def test_fillna_no_op_returns_copy(self, data):
        data = data[~data.isna()]  # 剔除 data 中的缺失值，形成新的数据对象 data

        valid = data[0]  # 取出 data 的第一个元素，作为 valid
        result = data.fillna(valid)  # 使用 valid 对 data 进行填充操作，生成结果 result
        assert result is not data  # 断言 result 不是 data 的同一对象
        tm.assert_extension_array_equal(result, data)  # 使用测试框架中的 assert 函数比较 result 和 data 是否相等

        result = data._pad_or_backfill(method="backfill")  # 对 data 使用指定的填充或向后填充方法
        assert result is not data  # 断言 result 不是 data 的同一对象
        tm.assert_extension_array_equal(result, data)  # 使用测试框架中的 assert 函数比较 result 和 data 是否相等

    def test_fillna_series(self, data_missing):
        fill_value = data_missing[1]  # 取出 data_missing 的第二个元素，作为填充值 fill_value
        ser = pd.Series(data_missing)  # 将 data_missing 转换为 Pandas 的 Series 对象 ser

        result = ser.fillna(fill_value)  # 使用 fill_value 对 ser 进行填充操作，生成结果 result
        expected = pd.Series(
            data_missing._from_sequence(
                [fill_value, fill_value], dtype=data_missing.dtype
            )
        )  # 构建期望的结果 expected，填充值为 fill_value
        tm.assert_series_equal(result, expected)  # 使用测试框架中的 assert 函数比较 result 和 expected 是否相等

        # Fill with a series
        result = ser.fillna(expected)  # 使用另一个 Series expected 对 ser 进行填充操作
        tm.assert_series_equal(result, expected)  # 使用测试框架中的 assert 函数比较 result 和 expected 是否相等

        # Fill with a series not affecting the missing values
        result = ser.fillna(ser)  # 使用 ser 自身对 ser 进行填充操作，不会影响缺失值
        tm.assert_series_equal(result, ser)  # 使用测试框架中的 assert 函数比较 result 和 ser 是否相等

    def test_fillna_series_method(self, data_missing, fillna_method):
        fill_value = data_missing[1]  # 取出 data_missing 的第二个元素，作为填充值 fill_value

        if fillna_method == "ffill":
            data_missing = data_missing[::-1]  # 如果填充方法为前向填充，则将 data_missing 反向

        result = getattr(pd.Series(data_missing), fillna_method)()  # 根据填充方法名称调用对应的填充方法，生成结果 result
        expected = pd.Series(
            data_missing._from_sequence(
                [fill_value, fill_value], dtype=data_missing.dtype
            )
        )  # 构建期望的结果 expected，填充值为 fill_value
        tm.assert_series_equal(result, expected)  # 使用测试框架中的 assert 函数比较 result 和 expected 是否相等
    # 定义测试方法，用于测试填充缺失值的情况，接受一个数据对象作为输入参数
    def test_fillna_frame(self, data_missing):
        # 从数据对象中获取填充值
        fill_value = data_missing[1]

        # 创建一个包含两列的 Pandas DataFrame 对象，其中一列使用 data_missing 填充缺失值
        result = pd.DataFrame({"A": data_missing, "B": [1, 2]}).fillna(fill_value)

        # 创建一个预期的 Pandas DataFrame 对象，其中 "A" 列使用填充值来填充缺失值，"B" 列填充为 1 和 2
        expected = pd.DataFrame(
            {
                "A": data_missing._from_sequence(
                    [fill_value, fill_value], dtype=data_missing.dtype
                ),
                "B": [1, 2],
            }
        )

        # 使用测试工具比较实际结果和预期结果的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试填充特定列的缺失值的情况，接受一个数据对象作为输入参数
    def test_fillna_fill_other(self, data):
        # 创建一个包含两列的 Pandas DataFrame 对象，其中 "B" 列的缺失值使用 0.0 填充
        result = pd.DataFrame({"A": data, "B": [np.nan] * len(data)}).fillna({"B": 0.0})

        # 创建一个预期的 Pandas DataFrame 对象，其中 "B" 列的所有值填充为 0.0，"A" 列的值与输入数据相同
        expected = pd.DataFrame({"A": data, "B": [0.0] * len(result)})

        # 使用测试工具比较实际结果和预期结果的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
```