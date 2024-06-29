# `D:\src\scipysrc\pandas\pandas\tests\frame\test_unary.py`

```
# 导入 Decimal 类，用于精确数值计算
from decimal import Decimal

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 导入 pytest 库，用于单元测试
import pytest

# 从 pandas.compat.numpy 模块中导入 np_version_gte1p25 函数
from pandas.compat.numpy import np_version_gte1p25

# 导入 pandas 库，并使用 pd 别名
import pandas as pd

# 导入 pandas._testing 模块，并使用 tm 别名
import pandas._testing as tm

# 定义 TestDataFrameUnaryOperators 类，用于测试 DataFrame 的一元操作符
class TestDataFrameUnaryOperators:
    # 测试 __pos__, __neg__, __invert__ 方法

    # 测试负数运算对数值型数据的影响
    @pytest.mark.parametrize(
        "df_data,expected_data",
        [
            ([-1, 1], [1, -1]),  # 整数列表的负数运算
            ([False, True], [True, False]),  # 布尔值列表的负数运算
            (pd.to_timedelta([-1, 1]), pd.to_timedelta([1, -1])),  # 时间增量的负数运算
        ],
    )
    def test_neg_numeric(self, df_data, expected_data):
        # 创建 DataFrame 对象 df，使用输入的数据 df_data
        df = pd.DataFrame({"a": df_data})
        # 创建期望的 DataFrame 对象 expected，使用期望的数据 expected_data
        expected = pd.DataFrame({"a": expected_data})
        # 断言 df 取负数后与期望结果 expected 相等
        tm.assert_frame_equal(-df, expected)
        # 断言 df 的列 "a" 取负数后与期望结果 expected 的列 "a" 相等
        tm.assert_series_equal(-df["a"], expected["a"])

    # 测试负数运算对对象类型数据的影响
    @pytest.mark.parametrize(
        "df, expected",
        [
            (np.array([1, 2], dtype=object), np.array([-1, -2], dtype=object)),  # 对象数组的负数运算
            ([Decimal("1.0"), Decimal("2.0")], [Decimal("-1.0"), Decimal("-2.0")]),  # Decimal 对象的负数运算
        ],
    )
    def test_neg_object(self, df, expected):
        # 创建 DataFrame 对象 df，使用输入的数据 df
        df = pd.DataFrame({"a": df})
        # 创建期望的 DataFrame 对象 expected，使用期望的数据 expected
        expected = pd.DataFrame({"a": expected})
        # 断言 df 取负数后与期望结果 expected 相等
        tm.assert_frame_equal(-df, expected)
        # 断言 df 的列 "a" 取负数后与期望结果 expected 的列 "a" 相等
        tm.assert_series_equal(-df["a"], expected["a"])

    # 测试负数运算对非数值型数据的错误处理
    @pytest.mark.parametrize(
        "df_data",
        [
            ["a", "b"],  # 字符串列表
            pd.to_datetime(["2017-01-22", "1970-01-01"]),  # 日期时间对象
        ],
    )
    def test_neg_raises(self, df_data, using_infer_string):
        # 创建 DataFrame 对象 df，使用输入的数据 df_data
        df = pd.DataFrame({"a": df_data})
        # 错误信息字符串，根据数据类型不同给出不同错误信息
        msg = (
            "bad operand type for unary -: 'str'|"
            r"bad operand type for unary -: 'DatetimeArray'"
        )
        # 如果 using_infer_string 为 True，并且第一列数据类型为字符串
        if using_infer_string and df.dtypes.iloc[0] == "string":
            # 导入 pyarrow 库，别名为 pa
            import pyarrow as pa
            # 设置错误信息为 "has no kernel"
            msg = "has no kernel"
            # 使用 pytest 的断言检查是否抛出预期的异常
            with pytest.raises(pa.lib.ArrowNotImplementedError, match=msg):
                (-df)
            with pytest.raises(pa.lib.ArrowNotImplementedError, match=msg):
                (-df["a"])
        else:
            # 使用 pytest 的断言检查是否抛出预期的异常
            with pytest.raises(TypeError, match=msg):
                (-df)
            with pytest.raises(TypeError, match=msg):
                (-df["a"])

    # 测试按位取反运算对 DataFrame 的影响
    def test_invert(self, float_frame):
        # 获取测试数据 float_frame
        df = float_frame
        # 断言按位取反后的 DataFrame 与按位非操作后的结果相等
        tm.assert_frame_equal(-(df < 0), ~(df < 0))

    # 测试按位取反运算对混合数据类型的 DataFrame 的影响
    def test_invert_mixed(self):
        # 定义 DataFrame 的形状
        shape = (10, 5)
        # 构造混合数据类型的 DataFrame
        df = pd.concat(
            [
                pd.DataFrame(np.zeros(shape, dtype="bool")),  # 布尔类型的 DataFrame
                pd.DataFrame(np.zeros(shape, dtype=int)),  # 整数类型的 DataFrame
            ],
            axis=1,
            ignore_index=True,
        )
        # 期望的结果 DataFrame
        expected = pd.concat(
            [
                pd.DataFrame(np.ones(shape, dtype="bool")),  # 全为 True 的布尔类型的 DataFrame
                pd.DataFrame(-np.ones(shape, dtype=int)),  # 全为 -1 的整数类型的 DataFrame
            ],
            axis=1,
            ignore_index=True,
        )
        # 断言按位取反后的结果与期望的结果相等
        tm.assert_frame_equal(~df, expected)
    def test_invert_empty_not_input(self):
        # 测试空 DataFrame 取反是否得到空结果，GH#51032
        df = pd.DataFrame()
        result = ~df
        # 断言原始 DataFrame 和取反结果应该相等
        tm.assert_frame_equal(df, result)
        # 断言原始 DataFrame 和取反结果不是同一个对象
        assert df is not result

    @pytest.mark.parametrize(
        "df_data",
        [
            [-1, 1],
            [False, True],
            pd.to_timedelta([-1, 1]),
        ],
    )
    def test_pos_numeric(self, df_data):
        # 测试正数操作对数值类型 DataFrame 的影响，GH#16073
        df = pd.DataFrame({"a": df_data})
        # 断言正数操作后 DataFrame 应该与原始 DataFrame 相等
        tm.assert_frame_equal(+df, df)
        # 断言正数操作后 DataFrame 列 "a" 应该与原始列 "a" 相等
        tm.assert_series_equal(+df["a"], df["a"])

    @pytest.mark.parametrize(
        "df_data",
        [
            np.array([-1, 2], dtype=object),
            [Decimal("-1.0"), Decimal("2.0")],
        ],
    )
    def test_pos_object(self, df_data):
        # 测试正数操作对对象类型 DataFrame 的影响，GH#21380
        df = pd.DataFrame({"a": df_data})
        # 断言正数操作后 DataFrame 应该与原始 DataFrame 相等
        tm.assert_frame_equal(+df, df)
        # 断言正数操作后 DataFrame 列 "a" 应该与原始列 "a" 相等
        tm.assert_series_equal(+df["a"], df["a"])

    @pytest.mark.filterwarnings("ignore:Applying:DeprecationWarning")
    def test_pos_object_raises(self):
        # 测试正数操作对对象类型引发异常的情况，GH#21380
        df = pd.DataFrame({"a": ["a", "b"]})
        if np_version_gte1p25:
            with pytest.raises(
                TypeError, match=r"^bad operand type for unary \+: \'str\'$"
            ):
                # 断言正数操作应该引发 TypeError 异常，匹配特定错误信息
                tm.assert_frame_equal(+df, df)
        else:
            # 断言正数操作后 DataFrame 列 "a" 应该与原始列 "a" 相等
            tm.assert_series_equal(+df["a"], df["a"])

    def test_pos_raises(self):
        # 测试正数操作对特定类型（DatetimeArray）引发异常的情况
        df = pd.DataFrame({"a": pd.to_datetime(["2017-01-22", "1970-01-01"])})
        msg = r"bad operand type for unary \+: 'DatetimeArray'"
        with pytest.raises(TypeError, match=msg):
            # 断言正数操作应该引发 TypeError 异常，匹配特定错误信息
            (+df)
        with pytest.raises(TypeError, match=msg):
            # 断言正数操作后 DataFrame 列 "a" 应该引发 TypeError 异常，匹配特定错误信息
            (+df["a"])
    # 定义一个测试方法，用于测试 DataFrame 的一元运算和空值处理
    def test_unary_nullable(self):
        # 创建一个包含不同数据类型和空值（pd.NA）的 DataFrame
        df = pd.DataFrame(
            {
                "a": pd.array([1, -2, 3, pd.NA], dtype="Int64"),  # 整数数组，包括空值
                "b": pd.array([4.0, -5.0, 6.0, pd.NA], dtype="Float32"),  # 浮点数数组，包括空值
                "c": pd.array([True, False, False, pd.NA], dtype="boolean"),  # 布尔数组，包括空值
                # 在非空值位置加入 numpy 布尔值，确保布尔与 boolean 的行为在非空值位置上一致
                "d": np.array([True, False, False, True]),  # numpy 布尔数组
            }
        )

        # 对 DataFrame 进行正号运算
        result = +df
        res_ufunc = np.positive(df)
        expected = df
        # TODO: assert that we have copies?（待完成：断言是否存在复制？）

        # 断言正号运算后的结果与期望结果一致
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(res_ufunc, expected)

        # 对 DataFrame 进行负号运算
        result = -df
        res_ufunc = np.negative(df)
        expected = pd.DataFrame(
            {
                "a": pd.array([-1, 2, -3, pd.NA], dtype="Int64"),  # 对应位置的数值取反，空值保持不变
                "b": pd.array([-4.0, 5.0, -6.0, pd.NA], dtype="Float32"),  # 对应位置的数值取反，空值保持不变
                "c": pd.array([False, True, True, pd.NA], dtype="boolean"),  # 对应位置的布尔值取反，空值保持不变
                "d": np.array([False, True, True, False]),  # 对应位置的布尔值取反
            }
        )
        # 断言负号运算后的结果与期望结果一致
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(res_ufunc, expected)

        # 对 DataFrame 进行绝对值运算
        result = abs(df)
        res_ufunc = np.abs(df)
        expected = pd.DataFrame(
            {
                "a": pd.array([1, 2, 3, pd.NA], dtype="Int64"),  # 对应位置的数值取绝对值，空值保持不变
                "b": pd.array([4.0, 5.0, 6.0, pd.NA], dtype="Float32"),  # 对应位置的数值取绝对值，空值保持不变
                "c": pd.array([True, False, False, pd.NA], dtype="boolean"),  # 对应位置的布尔值取绝对值，空值保持不变
                "d": np.array([True, False, False, True]),  # 对应位置的布尔值取绝对值
            }
        )
        # 断言绝对值运算后的结果与期望结果一致
        tm.assert_frame_equal(result, expected)
        tm.assert_frame_equal(res_ufunc, expected)
```