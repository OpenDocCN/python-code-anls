# `D:\src\scipysrc\pandas\pandas\tests\frame\indexing\test_mask.py`

```
"""
Tests for DataFrame.mask; tests DataFrame.where as a side-effect.
"""

import numpy as np  # 导入 NumPy 库，用于数值计算

from pandas import (  # 从 Pandas 库中导入以下对象和函数
    NA,  # 缺失值标识符
    DataFrame,  # 数据框对象
    Float64Dtype,  # 浮点类型对象
    Series,  # 系列对象
    StringDtype,  # 字符串类型对象
    Timedelta,  # 时间增量对象
    isna,  # 判断是否为缺失值的函数
)
import pandas._testing as tm  # 导入 Pandas 测试模块


class TestDataFrameMask:
    def test_mask(self):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))  # 创建一个 5x3 的随机数数据框
        cond = df > 0  # 创建一个条件数据框，判断每个元素是否大于 0

        rs = df.where(cond, np.nan)  # 根据条件填充数据框，不满足条件的用 NaN 填充
        tm.assert_frame_equal(rs, df.mask(df <= 0))  # 断言两个数据框相等，其中一个使用 mask 方法处理

        tm.assert_frame_equal(rs, df.mask(~cond))  # 断言两个数据框相等，使用取反条件的 mask 方法处理

        other = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))  # 创建另一个相同形状的随机数数据框
        rs = df.where(cond, other)  # 根据条件填充数据框，不满足条件的用 other 数据框的对应元素填充
        tm.assert_frame_equal(rs, df.mask(df <= 0, other))  # 断言两个数据框相等，其中一个使用 mask 方法处理

        tm.assert_frame_equal(rs, df.mask(~cond, other))  # 断言两个数据框相等，使用取反条件的 mask 方法处理

    def test_mask2(self):
        # see GH#21891
        df = DataFrame([1, 2])  # 创建一个包含两个元素的数据框
        res = df.mask([[True], [False]])  # 根据指定的掩码值创建新的数据框

        exp = DataFrame([np.nan, 2])  # 期望得到的结果数据框
        tm.assert_frame_equal(res, exp)  # 断言两个数据框相等

    def test_mask_inplace(self):
        # GH#8801
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))  # 创建一个随机数数据框
        cond = df > 0  # 创建一个条件数据框，判断每个元素是否大于 0

        rdf = df.copy()  # 复制数据框 df 到 rdf

        return_value = rdf.where(cond, inplace=True)  # 原地修改数据框 rdf，根据条件填充数据
        assert return_value is None  # 断言返回值为 None
        tm.assert_frame_equal(rdf, df.where(cond))  # 断言两个数据框相等，其中一个使用 where 方法处理
        tm.assert_frame_equal(rdf, df.mask(~cond))  # 断言两个数据框相等，使用取反条件的 mask 方法处理

        rdf = df.copy()  # 再次复制数据框 df 到 rdf
        return_value = rdf.where(cond, -df, inplace=True)  # 原地修改数据框 rdf，根据条件填充数据，不满足条件的用相反数填充
        assert return_value is None  # 断言返回值为 None
        tm.assert_frame_equal(rdf, df.where(cond, -df))  # 断言两个数据框相等，其中一个使用 where 方法处理
        tm.assert_frame_equal(rdf, df.mask(~cond, -df))  # 断言两个数据框相等，使用取反条件的 mask 方法处理

    def test_mask_edge_case_1xN_frame(self):
        # GH#4071
        df = DataFrame([[1, 2]])  # 创建一个包含单行两列的数据框
        res = df.mask(DataFrame([[True, False]]))  # 根据指定的掩码值创建新的数据框
        expec = DataFrame([[np.nan, 2]])  # 期望得到的结果数据框
        tm.assert_frame_equal(res, expec)  # 断言两个数据框相等

    def test_mask_callable(self):
        # GH#12533
        df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 创建一个数据框
        result = df.mask(lambda x: x > 4, lambda x: x + 1)  # 根据可调用函数对数据框进行填充
        exp = DataFrame([[1, 2, 3], [4, 6, 7], [8, 9, 10]])  # 期望得到的结果数据框
        tm.assert_frame_equal(result, exp)  # 断言两个数据框相等
        tm.assert_frame_equal(result, df.mask(df > 4, df + 1))  # 断言两个数据框相等，使用 mask 方法处理

        # return ndarray and scalar
        result = df.mask(lambda x: (x % 2 == 0).values, lambda x: 99)  # 根据可调用函数对数据框进行填充
        exp = DataFrame([[1, 99, 3], [99, 5, 99], [7, 99, 9]])  # 期望得到的结果数据框
        tm.assert_frame_equal(result, exp)  # 断言两个数据框相等
        tm.assert_frame_equal(result, df.mask(df % 2 == 0, 99))  # 断言两个数据框相等，使用 mask 方法处理

        # chain
        result = (df + 2).mask(lambda x: x > 8, lambda x: x + 10)  # 根据可调用函数对数据框进行填充
        exp = DataFrame([[3, 4, 5], [6, 7, 8], [19, 20, 21]])  # 期望得到的结果数据框
        tm.assert_frame_equal(result, exp)  # 断言两个数据框相等
        tm.assert_frame_equal(result, (df + 2).mask((df + 2) > 8, (df + 2) + 10))  # 断言两个数据框相等，使用 mask 方法处理
    def test_mask_dtype_bool_conversion(self):
        # 定义一个测试方法，用于验证布尔类型转换和掩码处理的正确性
        # GH#3733 表示这个测试是针对 GitHub 问题编号为 3733 的情况
        # 创建一个包含随机标准正态分布数据的 DataFrame，大小为 (100, 50)
        df = DataFrame(data=np.random.default_rng(2).standard_normal((100, 50)))
        # 使用 where 方法在 DataFrame 中创建 NaN 值，即将所有小于等于 0 的值替换为 NaN
        df = df.where(df > 0)
        # 生成一个布尔型的 DataFrame，标记所有大于 0 的位置为 True，其余为 False
        bools = df > 0
        # 使用 isna 函数生成一个布尔掩码，标记 DataFrame 中的 NaN 值位置为 True，其余为 False
        mask = isna(df)
        # 将 bools 转换为 object 类型，并根据 mask 进行掩码处理，生成期望的 DataFrame
        expected = bools.astype(object).mask(mask)
        # 根据 mask 对 bools 进行掩码处理，生成实际结果的 DataFrame
        result = bools.mask(mask)
        # 使用 tm.assert_frame_equal 方法断言 result 和 expected 是否相等，即验证测试结果
        tm.assert_frame_equal(result, expected)
# 定义函数 test_mask_stringdtype，用于测试 DataFrame 或 Series 对象的 mask 方法
def test_mask_stringdtype(frame_or_series):
    # GH 40824
    # 创建一个 DataFrame 对象 obj，包含一列 "A"，数据为 ["foo", "bar", "baz", NA]，指定 StringDtype 类型
    obj = DataFrame(
        {"A": ["foo", "bar", "baz", NA]},
        index=["id1", "id2", "id3", "id4"],
        dtype=StringDtype(),
    )
    # 创建一个 DataFrame 对象 filtered_obj，包含一列 "A"，数据为 ["this", "that"]，指定 StringDtype 类型
    filtered_obj = DataFrame(
        {"A": ["this", "that"]}, index=["id2", "id3"], dtype=StringDtype()
    )
    # 创建一个预期结果的 DataFrame 对象 expected，包含一列 "A"，数据为 [NA, "this", "that", NA]，指定 StringDtype 类型
    expected = DataFrame(
        {"A": [NA, "this", "that", NA]},
        index=["id1", "id2", "id3", "id4"],
        dtype=StringDtype(),
    )
    
    # 如果传入的 frame_or_series 是 Series 类型，则将 obj、filtered_obj 和 expected 都改为对应的 Series 对象
    if frame_or_series is Series:
        obj = obj["A"]
        filtered_obj = filtered_obj["A"]
        expected = expected["A"]
    
    # 创建一个 Series 对象 filter_ser，包含 [False, True, True, False]，用于过滤 obj 对象
    filter_ser = Series([False, True, True, False])
    # 使用 mask 方法对 obj 应用 filter_ser 进行过滤，filtered_obj 为过滤条件下的替换值，结果赋给 result
    result = obj.mask(filter_ser, filtered_obj)
    
    # 使用测试框架 tm 的 assert_equal 方法断言 result 是否与 expected 相等
    tm.assert_equal(result, expected)


# 定义函数 test_mask_where_dtype_timedelta，用于测试 DataFrame 的 mask 方法在处理 timedelta 类型时的行为
def test_mask_where_dtype_timedelta():
    # https://github.com/pandas-dev/pandas/issues/39548
    # 创建一个 DataFrame 对象 df，包含五个 Timedelta 对象，单位为天
    df = DataFrame([Timedelta(i, unit="D") for i in range(5)])
    
    # 创建一个预期结果的 DataFrame 对象 expected，包含五个空值，类型为 timedelta64[ns]
    expected = DataFrame(np.full(5, np.nan, dtype="timedelta64[ns]"))
    # 使用 mask 方法对 df 应用 df.notna() 的结果进行过滤，结果赋给 df，并与 expected 进行断言比较
    tm.assert_frame_equal(df.mask(df.notna()), expected)
    
    # 创建一个预期结果的 DataFrame 对象 expected，包含 [np.nan, np.nan, np.nan, Timedelta("3 day"), Timedelta("4 day")]
    expected = DataFrame(
        [np.nan, np.nan, np.nan, Timedelta("3 day"), Timedelta("4 day")]
    )
    # 使用 mask 方法对 df 应用 df > Timedelta(2, unit="D") 的条件进行过滤，结果赋给 df，并与 expected 进行断言比较
    tm.assert_frame_equal(df.where(df > Timedelta(2, unit="D")), expected)


# 定义函数 test_mask_return_dtype，用于测试 Series 对象的 mask 方法在不同数据类型下的行为
def test_mask_return_dtype():
    # GH#50488
    # 创建一个 Series 对象 ser，包含 [0.0, 1.0, 2.0, 3.0]，指定 Float64Dtype 类型
    ser = Series([0.0, 1.0, 2.0, 3.0], dtype=Float64Dtype())
    # 创建一个条件 Series 对象 cond，包含 ~ser.isna() 的结果，用于 mask 方法的过滤条件
    cond = ~ser.isna()
    # 创建一个 Series 对象 other，包含 [True, False, True, False]，用于 mask 方法的替换值
    other = Series([True, False, True, False])
    # 创建一个预期结果的 Series 对象 expected，包含 [1.0, 0.0, 1.0, 0.0]，数据类型与 ser 相同
    excepted = Series([1.0, 0.0, 1.0, 0.0], dtype=ser.dtype)
    # 使用 mask 方法对 ser 应用 cond 进行过滤，other 作为替换值，结果赋给 result
    result = ser.mask(cond, other)
    # 使用测试框架 tm 的 assert_series_equal 方法断言 result 是否与 expected 相等
    tm.assert_series_equal(result, excepted)


# 定义函数 test_mask_inplace_no_other，用于测试 DataFrame 对象的 mask 方法在 inplace 模式下的行为
def test_mask_inplace_no_other():
    # GH#51685
    # 创建一个 DataFrame 对象 df，包含两列 "a" 和 "b"，数据分别为 [1.0, 2.0] 和 ["x", "y"]
    df = DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    # 创建一个 DataFrame 对象 cond，包含两列 "a" 和 "b"，数据分别为 [True, False] 和 [False, True]
    cond = DataFrame({"a": [True, False], "b": [False, True]})
    # 使用 mask 方法对 df 应用 cond 进行过滤，并指定 inplace=True，直接修改 df 自身
    df.mask(cond, inplace=True)
    # 创建一个预期结果的 DataFrame 对象 expected，包含两列 "a" 和 "b"，数据分别为 [np.nan, 2] 和 ["x", np.nan]
    expected = DataFrame({"a": [np.nan, 2], "b": ["x", np.nan]})
    # 使用测试框架 tm 的 assert_frame_equal 方法断言 df 是否与 expected 相等
    tm.assert_frame_equal(df, expected)
```