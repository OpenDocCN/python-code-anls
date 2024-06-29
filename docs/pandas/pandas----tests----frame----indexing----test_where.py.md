# `D:\src\scipysrc\pandas\pandas\tests\frame\indexing\test_where.py`

```
# 导入 datetime 模块中的 datetime 类
from datetime import datetime

# 导入 hypothesis 模块中的 given 函数
from hypothesis import given

# 导入 numpy 模块并重命名为 np
import numpy as np

# 导入 pytest 模块
import pytest

# 从 pandas.core.dtypes.common 模块中导入 is_scalar 函数
from pandas.core.dtypes.common import is_scalar

# 导入 pandas 模块并重命名为 pd
import pandas as pd

# 从 pandas 模块中导入以下类和函数
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Series,
    StringDtype,
    Timestamp,
    date_range,
    isna,
)

# 导入 pandas._testing 模块并重命名为 tm
import pandas._testing as tm

# 从 pandas._testing._hypothesis 模块中导入 OPTIONAL_ONE_OF_ALL 常量
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL

# 定义一个 pytest fixture 函数，参数化返回不同的 DataFrame 对象
@pytest.fixture(params=["default", "float_string", "mixed_float", "mixed_int"])
def where_frame(request, float_string_frame, mixed_float_frame, mixed_int_frame):
    # 根据参数值返回相应的 DataFrame 对象
    if request.param == "default":
        return DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), columns=["A", "B", "C"]
        )
    if request.param == "float_string":
        return float_string_frame
    if request.param == "mixed_float":
        return mixed_float_frame
    if request.param == "mixed_int":
        return mixed_int_frame

# 定义一个私有函数 _safe_add，用于对 DataFrame 中的数值列进行安全加法操作
def _safe_add(df):
    # 定义内部函数 is_ok，用于检查列是否为数值类型且不是 uint8 类型
    def is_ok(s):
        return (
            issubclass(s.dtype.type, (np.integer, np.floating)) and s.dtype != "uint8"
        )
    
    # 返回一个新的 DataFrame，对数值列进行加一操作，非数值列保持不变
    return DataFrame(dict((c, s + 1) if is_ok(s) else (c, s) for c, s in df.items()))

# 定义 TestDataFrameIndexingWhere 类，用于测试 DataFrame 的条件索引操作
class TestDataFrameIndexingWhere:
    # 定义测试方法 test_where_get，测试 DataFrame 的 where 方法
    def test_where_get(self, where_frame, float_string_frame):
        # 定义内部函数 _check_get，用于验证 where 方法的正确性
        def _check_get(df, cond, check_dtypes=True):
            # 对输入的 DataFrame 进行安全加法操作
            other1 = _safe_add(df)
            # 使用 where 方法进行条件过滤，得到结果 rs 和 rs2
            rs = df.where(cond, other1)
            rs2 = df.where(cond.values, other1)
            # 遍历 rs 的每列
            for k, v in rs.items():
                # 构建预期结果 exp
                exp = Series(np.where(cond[k], df[k], other1[k]), index=v.index)
                # 检查 Series 是否相等，忽略名称检查
                tm.assert_series_equal(v, exp, check_names=False)
            # 检查 rs 和 rs2 是否相等
            tm.assert_frame_equal(rs, rs2)
            
            # 检查数据类型是否保持不变
            if check_dtypes:
                assert (rs.dtypes == df.dtypes).all()

        # 获取测试用的 DataFrame
        df = where_frame
        # 如果 df 是 float_string_frame，则抛出预期的 TypeError 异常
        if df is float_string_frame:
            msg = "'>' not supported between instances of 'str' and 'int'"
            with pytest.raises(TypeError, match=msg):
                df > 0
            return
        # 构建条件 cond，用于测试 where 方法
        cond = df > 0
        # 调用 _check_get 方法进行测试
        _check_get(df, cond)

    # 定义测试方法 test_where_upcasting，测试 DataFrame 的类型转换情况
    def test_where_upcasting(self):
        # 创建一个 DataFrame，包含不同类型的列
        df = DataFrame(
            {
                c: Series([1] * 3, dtype=c)
                for c in ["float32", "float64", "int32", "int64"]
            }
        )
        # 修改第二行数据为 0
        df.iloc[1, :] = 0
        # 执行 dtypes 方法，获取列的数据类型
        result = df.dtypes
        # 构建预期结果 expected，指定每列的预期数据类型
        expected = Series(
            [
                np.dtype("float32"),
                np.dtype("float64"),
                np.dtype("int32"),
                np.dtype("int64"),
            ],
            index=["float32", "float64", "int32", "int64"],
        )

        # 断言 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 当我们不保留布尔转换时的预期结果
        #
        # expected = Series({ 'float32' : 1, 'float64' : 3 })
    def test_where_alignment(self, where_frame, float_string_frame):
        # aligning
        # 定义内部函数 _check_align，用于检查对齐操作
        def _check_align(df, cond, other, check_dtypes=True):
            # 对 DataFrame 进行条件 where 操作，使用指定的 other 替代不满足条件的值
            rs = df.where(cond, other)
            # 遍历结果 DataFrame 的每一列
            for i, k in enumerate(rs.columns):
                # 获取当前列的结果 Series
                result = rs[k]
                # 获取原始 DataFrame 中当前列的数据值
                d = df[k].values
                # 获取条件的布尔值 Series，确保与原始 DataFrame 对齐
                c = cond[k].reindex(df[k].index).fillna(False).values

                # 确定 other 的值，根据其类型进行不同的处理
                if is_scalar(other):
                    o = other
                elif isinstance(other, np.ndarray):
                    o = Series(other[:, i], index=result.index).values
                else:
                    o = other[k].values

                # 根据条件 c，选择要使用的值，构造新的 Series
                new_values = d if c.all() else np.where(c, d, o)
                expected = Series(new_values, index=result.index, name=k)

                # 使用 assert_series_equal 函数比较结果 Series 和期望的 Series
                # 由于 numpy 无法降级转换类型，因此 check_dtype 设为 False
                tm.assert_series_equal(result, expected, check_dtype=False)

            # 检查数据类型是否一致
            # 当 other 是 ndarray 时，无法检查数据类型
            if check_dtypes and not isinstance(other, np.ndarray):
                assert (rs.dtypes == df.dtypes).all()

        # 主测试函数开始
        df = where_frame
        # 如果 DataFrame 等于 float_string_frame，则进行特定的异常测试
        if df is float_string_frame:
            msg = "'>' not supported between instances of 'str' and 'int'"
            # 使用 pytest 的 raises 函数检查是否抛出预期的 TypeError 异常
            with pytest.raises(TypeError, match=msg):
                df > 0
            return

        # 其他情况下，进行不同类型的 where 操作测试

        # 1. other 是 DataFrame
        cond = (df > 0)[1:]
        _check_align(df, cond, _safe_add(df))

        # 2. other 是 ndarray
        cond = df > 0
        _check_align(df, cond, (_safe_add(df).values))

        # 3. 整数类型会升级，因此不检查数据类型
        cond = df > 0
        # 检查是否所有列都不是 np.integer 的子类
        check_dtypes = all(not issubclass(s.type, np.integer) for s in df.dtypes)
        _check_align(df, cond, np.nan, check_dtypes=check_dtypes)

    # 忽略 Python 3.12 中关于反转布尔值的 DeprecationWarning
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_where_invalid(self):
        # invalid conditions
        # 创建一个随机数值的 DataFrame
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 3)), columns=["A", "B", "C"]
        )
        # 设置条件
        cond = df > 0

        # 错误情况 1：other 的形状与自身不匹配
        err1 = (df + 1).values[0:2, :]
        msg = "other must be the same shape as self when an ndarray"
        # 使用 pytest 的 raises 函数检查是否抛出预期的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            df.where(cond, err1)

        # 错误情况 2：条件数组的形状与自身不匹配
        err2 = cond.iloc[:2, :].values
        other1 = _safe_add(df)
        msg = "Array conditional must be same shape as self"
        # 使用 pytest 的 raises 函数检查是否抛出预期的 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            df.where(err2, other1)

        # 使用 mask 函数时，条件数组的形状与自身不匹配
        with pytest.raises(ValueError, match=msg):
            df.mask(True)
        with pytest.raises(ValueError, match=msg):
            df.mask(0)
    def test_where_set(self, where_frame, float_string_frame, mixed_int_frame):
        # where inplace

        def _check_set(df, cond, check_dtypes=True):
            # 复制 DataFrame，以免修改原始数据
            dfi = df.copy()
            # 将条件重新索引并推断对象类型
            econd = cond.reindex_like(df).fillna(True).infer_objects()
            # 根据条件生成预期的 DataFrame
            expected = dfi.mask(~econd)

            # 在原 DataFrame 上使用 where 方法，期望 inplace 修改
            return_value = dfi.where(cond, np.nan, inplace=True)
            assert return_value is None
            # 断言修改后的 DataFrame 与预期的 DataFrame 相等
            tm.assert_frame_equal(dfi, expected)

            # 检查数据类型是否符合预期
            if check_dtypes:
                for k, v in df.dtypes.items():
                    if issubclass(v.type, np.integer) and not cond[k].all():
                        v = np.dtype("float64")
                    assert dfi[k].dtype == v

        # 获取测试 DataFrame
        df = where_frame
        # 如果 DataFrame 是 float_string_frame，则测试不支持的操作
        if df is float_string_frame:
            msg = "'>' not supported between instances of 'str' and 'int'"
            with pytest.raises(TypeError, match=msg):
                df > 0
            return
        # 如果 DataFrame 是 mixed_int_frame，则转换为 float64 类型
        if df is mixed_int_frame:
            df = df.astype("float64")

        # 检查 df 中大于 0 的情况
        cond = df > 0
        _check_set(df, cond)

        # 检查 df 中大于等于 0 的情况
        cond = df >= 0
        _check_set(df, cond)

        # 检查 df 中从第二行开始大于等于 0 的情况
        cond = (df >= 0)[1:]
        _check_set(df, cond)

    def test_where_series_slicing(self):
        # GH 10218
        # test DataFrame.where with Series slicing
        # 创建一个测试用的 DataFrame
        df = DataFrame({"a": range(3), "b": range(4, 7)})
        # 使用 DataFrame 的 where 方法与 Series 切片进行比较
        result = df.where(df["a"] == 1)
        # 预期的结果是对 DataFrame 进行条件切片后的重建索引
        expected = df[df["a"] == 1].reindex(df.index)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("klass", [list, tuple, np.array])
    def test_where_array_like(self, klass):
        # see gh-15414
        # 创建一个测试用的 DataFrame
        df = DataFrame({"a": [1, 2, 3]})
        # 定义一个布尔条件
        cond = [[False], [True], [True]]
        # 创建预期的 DataFrame
        expected = DataFrame({"a": [np.nan, 2, 3]})

        # 使用 DataFrame 的 where 方法与不同类型的数组类进行比较
        result = df.where(klass(cond))
        tm.assert_frame_equal(result, expected)

        # 修改 DataFrame 添加新的列 'b'
        df["b"] = 2
        expected["b"] = [2, np.nan, 2]
        # 定义新的布尔条件
        cond = [[False, True], [True, False], [True, True]]

        # 再次使用 DataFrame 的 where 方法与不同类型的数组类进行比较
        result = df.where(klass(cond))
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "cond",
        [
            [[1], [0], [1]],
            Series([[2], [5], [7]]),
            DataFrame({"a": [2, 5, 7]}),
            [["True"], ["False"], ["True"]],
            [[Timestamp("2017-01-01")], [pd.NaT], [Timestamp("2017-01-02")]],
        ],
    )
    def test_where_invalid_input_single(self, cond):
        # see gh-15414: only boolean arrays accepted
        # 创建一个测试用的 DataFrame
        df = DataFrame({"a": [1, 2, 3]})
        # 预期抛出错误的消息
        msg = "Boolean array expected for the condition"

        # 使用 DataFrame 的 where 方法测试非法输入的情况
        with pytest.raises(TypeError, match=msg):
            df.where(cond)
    @pytest.mark.parametrize(
        "cond",
        [  # 参数化测试用例，测试不同的条件
            [[0, 1], [1, 0], [1, 1]],  # 列表形式的条件：包含嵌套的列表
            Series([[0, 2], [5, 0], [4, 7]]),  # Series 对象作为条件：包含嵌套的列表
            [["False", "True"], ["True", "False"], ["True", "True"]],  # 字符串列表作为条件：包含嵌套的列表
            DataFrame({"a": [2, 5, 7], "b": [4, 8, 9]}),  # DataFrame 对象作为条件：包含两列
            [  # 包含时间戳和 NaT 的列表作为条件
                [pd.NaT, Timestamp("2017-01-01")],
                [Timestamp("2017-01-02"), pd.NaT],
                [Timestamp("2017-01-03"), Timestamp("2017-01-03")],
            ],
        ],
    )
    def test_where_invalid_input_multiple(self, cond):
        # 见 gh-15414：只有布尔数组被接受作为条件
        df = DataFrame({"a": [1, 2, 3], "b": [2, 2, 2]})
        msg = "Boolean array expected for the condition"

        with pytest.raises(TypeError, match=msg):
            df.where(cond)

    def test_where_dataframe_col_match(self):
        df = DataFrame([[1, 2, 3], [4, 5, 6]])
        cond = DataFrame([[True, False, True], [False, False, True]])

        result = df.where(cond)
        expected = DataFrame([[1.0, np.nan, 3], [np.nan, np.nan, 6]])
        tm.assert_frame_equal(result, expected)

        # this *does* align, though has no matching columns
        cond.columns = ["a", "b", "c"]
        result = df.where(cond)
        expected = DataFrame(np.nan, index=df.index, columns=df.columns)
        tm.assert_frame_equal(result, expected)

    def test_where_ndframe_align(self):
        msg = "Array conditional must be same shape as self"
        df = DataFrame([[1, 2, 3], [4, 5, 6]])

        cond = [True]
        with pytest.raises(ValueError, match=msg):
            df.where(cond)

        expected = DataFrame([[1, 2, 3], [np.nan, np.nan, np.nan]])

        out = df.where(Series(cond))
        tm.assert_frame_equal(out, expected)

        cond = np.array([False, True, False, True])
        with pytest.raises(ValueError, match=msg):
            df.where(cond)

        expected = DataFrame([[np.nan, np.nan, np.nan], [4, 5, 6]])

        out = df.where(Series(cond))
        tm.assert_frame_equal(out, expected)

    def test_where_bug(self):
        # 见 gh-2793
        df = DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]}, dtype="float64"
        )
        expected = DataFrame(
            {"a": [np.nan, np.nan, 3.0, 4.0], "b": [4.0, 3.0, np.nan, np.nan]},
            dtype="float64",
        )
        result = df.where(df > 2, np.nan)
        tm.assert_frame_equal(result, expected)

        result = df.copy()
        return_value = result.where(result > 2, np.nan, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(result, expected)
    def test_where_bug_mixed(self, any_signed_int_numpy_dtype):
        # 见问题报告 gh-2793
        # 创建一个包含两列的 DataFrame 对象，其中一列为任意有符号整数类型的 NumPy 数组，另一列为浮点数数组
        df = DataFrame(
            {
                "a": np.array([1, 2, 3, 4], dtype=any_signed_int_numpy_dtype),
                "b": np.array([4.0, 3.0, 2.0, 1.0], dtype="float64"),
            }
        )

        # 创建预期结果的 DataFrame 对象，将 "a" 列的部分数值替换为 -1，并指定列类型
        expected = DataFrame(
            {"a": [-1, -1, 3, 4], "b": [4.0, 3.0, -1, -1]},
        ).astype({"a": any_signed_int_numpy_dtype, "b": "float64"})

        # 使用 where 函数根据条件替换 DataFrame 对象 df 中的元素，小于等于 2 的元素替换为 -1
        result = df.where(df > 2, -1)
        # 断言 result 与预期结果 expected 相等
        tm.assert_frame_equal(result, expected)

        # 复制 DataFrame 对象 df 到 result，并使用 inplace=True 参数替换 df 中小于等于 2 的元素为 -1
        result = df.copy()
        return_value = result.where(result > 2, -1, inplace=True)
        # 确认返回值为 None
        assert return_value is None
        # 断言 result 与预期结果 expected 相等
        tm.assert_frame_equal(result, expected)

    def test_where_bug_transposition(self):
        # 见问题报告 gh-7506
        # 创建两个 DataFrame 对象 a 和 b，每个对象包含两行数据
        a = DataFrame({0: [1, 2], 1: [3, 4], 2: [5, 6]})
        b = DataFrame({0: [np.nan, 8], 1: [9, np.nan], 2: [np.nan, np.nan]})
        # 创建布尔型 DataFrame 对象 do_not_replace，指示哪些元素不应替换
        do_not_replace = b.isna() | (a > b)

        # 复制 DataFrame 对象 a 到 expected，并根据 do_not_replace 替换部分元素为 b 对象的相应元素
        expected = a.copy()
        expected[~do_not_replace] = b
        expected[[0, 1]] = expected[[0, 1]].astype("float64")

        # 使用 where 函数根据条件替换 DataFrame 对象 a 中的元素，小于等于 b 对象相应元素的值替换为 b 中的值
        result = a.where(do_not_replace, b)
        # 断言 result 与预期结果 expected 相等
        tm.assert_frame_equal(result, expected)

        # 重设 DataFrame 对象 a 和 b 的值
        a = DataFrame({0: [4, 6], 1: [1, 0]})
        b = DataFrame({0: [np.nan, 3], 1: [3, np.nan]})
        # 创建布尔型 DataFrame 对象 do_not_replace，指示哪些元素不应替换
        do_not_replace = b.isna() | (a > b)

        # 复制 DataFrame 对象 a 到 expected，并根据 do_not_replace 替换部分元素为 b 对象的相应元素
        expected = a.copy()
        expected[~do_not_replace] = b
        expected[1] = expected[1].astype("float64")

        # 使用 where 函数根据条件替换 DataFrame 对象 a 中的元素，小于等于 b 对象相应元素的值替换为 b 中的值
        result = a.where(do_not_replace, b)
        # 断言 result 与预期结果 expected 相等
        tm.assert_frame_equal(result, expected)

    def test_where_datetime(self):
        # 见问题报告 GH 3311
        # 创建包含日期范围、随机数列的 DataFrame 对象 df
        df = DataFrame(
            {
                "A": date_range("20130102", periods=5),
                "B": date_range("20130104", periods=5),
                "C": np.random.default_rng(2).standard_normal(5),
            }
        )

        # 创建日期时间戳 stamp
        stamp = datetime(2013, 1, 3)
        # 预期引发 TypeError 异常，显示不支持浮点数与日期时间戳之间的比较
        msg = "'>' not supported between instances of 'float' and 'datetime.datetime'"
        with pytest.raises(TypeError, match=msg):
            df > stamp

        # 根据条件选择 DataFrame 对象 df 中的元素，将部分元素置为 NaN
        result = df[df.iloc[:, :-1] > stamp]

        # 复制 DataFrame 对象 df 到 expected，并将部分元素置为 NaN
        expected = df.copy()
        expected.loc[[0, 1], "A"] = np.nan

        expected.loc[:, "C"] = np.nan
        # 断言 result 与预期结果 expected 相等
        tm.assert_frame_equal(result, expected)
    def test_where_none(self):
        # GH 4667
        # setting with None changes dtype
        
        # 创建一个包含单列"series"的DataFrame，并将其转换为浮点类型
        df = DataFrame({"series": Series(range(10))}).astype(float)
        
        # 将大于7的值设置为None
        df[df > 7] = None
        
        # 期望的DataFrame，其中包含一列"series"，前8行为0到7，后两行为NaN
        expected = DataFrame(
            {"series": Series([0, 1, 2, 3, 4, 5, 6, 7, np.nan, np.nan])}
        )
        
        # 断言两个DataFrame相等
        tm.assert_frame_equal(df, expected)

        # GH 7656
        # 创建一个包含字典的DataFrame，字典包含"A", "B", "C"三列，其中包含NaN和字符串
        df = DataFrame(
            [
                {"A": 1, "B": np.nan, "C": "Test"},
                {"A": np.nan, "B": "Test", "C": np.nan},
            ]
        )

        # 备份原始DataFrame
        orig = df.copy()

        # 创建一个布尔掩码，用于标识非NaN值
        mask = ~isna(df)
        
        # 使用布尔掩码将DataFrame中的NaN值替换为None
        df.where(mask, None, inplace=True)
        
        # 期望的DataFrame，根据掩码替换NaN为None
        expected = DataFrame(
            {
                "A": [1.0, np.nan],
                "B": [None, "Test"],
                "C": ["Test", None],
            }
        )
        
        # 断言两个DataFrame相等
        tm.assert_frame_equal(df, expected)

        # 复制原始DataFrame
        df = orig.copy()
        
        # 将掩码之外的值设置为None
        df[~mask] = None
        
        # 断言两个DataFrame相等
        tm.assert_frame_equal(df, expected)

    def test_where_empty_df_and_empty_cond_having_non_bool_dtypes(self):
        # see gh-21947
        # 创建一个空的DataFrame，列名为"a"
        df = DataFrame(columns=["a"])
        
        # 条件为DataFrame本身
        cond = df
        
        # 断言DataFrame中所有列的数据类型都是object
        assert (cond.dtypes == object).all()

        # 使用条件对DataFrame进行过滤
        result = df.where(cond)
        
        # 断言过滤结果与原始DataFrame相等
        tm.assert_frame_equal(result, df)

    def test_where_align(self):
        # 定义一个函数，用于创建一个随机数填充的DataFrame，部分值为NaN
        def create():
            df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)))
            df.iloc[3:5, 0] = np.nan
            df.iloc[4:6, 1] = np.nan
            df.iloc[5:8, 2] = np.nan
            return df

        # series
        # 创建一个DataFrame
        df = create()
        
        # 期望的DataFrame，将NaN值填充为每列的均值
        expected = df.fillna(df.mean())
        
        # 使用where方法将NaN值替换为每列的均值，沿着列方向
        result = df.where(pd.notna(df), df.mean(), axis="columns")
        
        # 断言两个DataFrame相等
        tm.assert_frame_equal(result, expected)

        # inplace=True时，直接在原始DataFrame上进行修改
        return_value = df.where(pd.notna(df), df.mean(), inplace=True, axis="columns")
        
        # 断言返回值为None
        assert return_value is None
        
        # 断言修改后的DataFrame与期望的DataFrame相等
        tm.assert_frame_equal(df, expected)

        # 创建一个DataFrame，并将NaN值填充为0
        df = create().fillna(0)
        
        # 期望的DataFrame，使用apply函数将小于等于0的值替换为第一列的值
        expected = df.apply(lambda x, y: x.where(x > 0, y), y=df[0])
        
        # 使用where方法将小于等于0的值替换为第一列的值，沿着索引方向
        result = df.where(df > 0, df[0], axis="index")
        
        # 断言两个DataFrame相等
        tm.assert_frame_equal(result, expected)
        
        # 使用where方法将小于等于0的值替换为第一列的值，沿着行方向
        result = df.where(df > 0, df[0], axis="rows")
        
        # 断言两个DataFrame相等
        tm.assert_frame_equal(result, expected)

        # frame
        # 创建一个DataFrame
        df = create()
        
        # 期望的DataFrame，将NaN值填充为1
        expected = df.fillna(1)
        
        # 使用where方法将NaN值替换为1
        result = df.where(
            pd.notna(df), DataFrame(1, index=df.index, columns=df.columns)
        )
        
        # 断言两个DataFrame相等
        tm.assert_frame_equal(result, expected)

    def test_where_complex(self):
        # GH 6345
        # 期望的DataFrame，将复数中绝对值大于等于5的值替换为NaN
        expected = DataFrame([[1 + 1j, 2], [np.nan, 4 + 1j]], columns=["a", "b"])
        
        # 创建一个DataFrame，将复数中绝对值大于等于5的值替换为NaN
        df = DataFrame([[1 + 1j, 2], [5 + 1j, 4 + 1j]], columns=["a", "b"])
        df[df.abs() >= 5] = np.nan
        
        # 断言两个DataFrame相等
        tm.assert_frame_equal(df, expected)
    def test_where_axis(self):
        # GH 9736
        # 创建一个 2x2 的随机数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((2, 2)))
        # 创建一个与 df 相同形状的布尔型 DataFrame，所有元素为 False
        mask = DataFrame([[False, False], [False, False]])
        # 创建一个包含两个元素的 Series
        ser = Series([0, 1])

        # 创建预期的 DataFrame，与 df 形状相同，数据类型为 float64
        expected = DataFrame([[0, 0], [1, 1]], dtype="float64")
        # 在指定轴（行）上使用 mask 和 ser 进行条件筛选
        result = df.where(mask, ser, axis="index")
        # 断言 result 与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 复制 df
        result = df.copy()
        # 在指定轴（行）上使用 mask 和 ser 进行条件筛选，将结果直接应用到 result 上
        return_value = result.where(mask, ser, axis="index", inplace=True)
        # 断言 inplace 操作返回 None
        assert return_value is None
        # 断言 result 与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 创建预期的 DataFrame，与 df 形状相同，数据类型为 float64
        expected = DataFrame([[0, 1], [0, 1]], dtype="float64")
        # 在指定轴（列）上使用 mask 和 ser 进行条件筛选
        result = df.where(mask, ser, axis="columns")
        # 断言 result 与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 复制 df
        result = df.copy()
        # 在指定轴（列）上使用 mask 和 ser 进行条件筛选，将结果直接应用到 result 上
        return_value = result.where(mask, ser, axis="columns", inplace=True)
        # 断言 inplace 操作返回 None
        assert return_value is None
        # 断言 result 与预期结果相等
        tm.assert_frame_equal(result, expected)

    def test_where_axis_with_upcast(self):
        # Upcast needed
        # 创建一个 2x2 的整型数据的 DataFrame
        df = DataFrame([[1, 2], [3, 4]], dtype="int64")
        # 创建一个与 df 相同形状的布尔型 DataFrame，所有元素为 False
        mask = DataFrame([[False, False], [False, False]])
        # 创建一个包含两个元素的 Series，其中一个元素为 NaN
        ser = Series([0, np.nan])

        # 创建预期的 DataFrame，与 df 形状相同，数据类型为 float64
        expected = DataFrame([[0, 0], [np.nan, np.nan]], dtype="float64")
        # 在指定轴（行）上使用 mask 和 ser 进行条件筛选
        result = df.where(mask, ser, axis="index")
        # 断言 result 与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 复制 df
        result = df.copy()
        # 在指定轴（行）上使用 mask 和 ser 进行条件筛选，同时验证未来可能的警告
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            return_value = result.where(mask, ser, axis="index", inplace=True)
        # 断言 inplace 操作返回 None
        assert return_value is None
        # 断言 result 与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 创建预期的 DataFrame，与 df 形状相同
        expected = DataFrame([[0, np.nan], [0, np.nan]])
        # 在指定轴（列）上使用 mask 和 ser 进行条件筛选
        result = df.where(mask, ser, axis="columns")
        # 断言 result 与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 创建预期的 DataFrame，具有指定列名和数据类型
        expected = DataFrame(
            {
                0: np.array([0, 0], dtype="int64"),
                1: np.array([np.nan, np.nan], dtype="float64"),
            }
        )
        # 复制 df
        result = df.copy()
        # 在指定轴（列）上使用 mask 和 ser 进行条件筛选，同时验证未来可能的警告
        with tm.assert_produces_warning(FutureWarning, match="incompatible dtype"):
            return_value = result.where(mask, ser, axis="columns", inplace=True)
        # 断言 inplace 操作返回 None
        assert return_value is None
        # 断言 result 与预期结果相等
        tm.assert_frame_equal(result, expected)
    ```python`
        def test_where_callable(self):
            # 测试 DataFrame 的 where 方法，传入 lambda 函数
            # 创建一个 DataFrame 对象 df
            df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            # 使用 where 方法，指定条件函数 lambda x: x > 4 和替代函数 lambda x: x + 1
            result = df.where(lambda x: x > 4, lambda x: x + 1)
            # 期望结果 DataFrame
            exp = DataFrame([[2, 3, 4], [5, 5, 6], [7, 8, 9]])
            # 断言结果与期望值相等
            tm.assert_frame_equal(result, exp)
            # 断言结果与 df 的 where 结果相等，条件是 df > 4，替代是 df + 1
            tm.assert_frame_equal(result, df.where(df > 4, df + 1))
    
            # 测试返回 ndarray 和标量的情况
            # 使用 where 方法，指定条件函数 lambda x: (x % 2 == 0).values 和替代函数 lambda x: 99
            result = df.where(lambda x: (x % 2 == 0).values, lambda x: 99)
            # 期望结果 DataFrame
            exp = DataFrame([[99, 2, 99], [4, 99, 6], [99, 8, 99]])
            # 断言结果与期望值相等
            tm.assert_frame_equal(result, exp)
            # 断言结果与 df 的 where 结果相等，条件是 df % 2 == 0，替代是 99
            tm.assert_frame_equal(result, df.where(df % 2 == 0, 99))
    
            # 测试链式调用
            # 对 df 加 2 后，使用 where 方法，指定条件函数 lambda x: x > 8 和替代函数 lambda x: x + 10
            result = (df + 2).where(lambda x: x > 8, lambda x: x + 10)
            # 期望结果 DataFrame
            exp = DataFrame([[13, 14, 15], [16, 17, 18], [9, 10, 11]])
            # 断言结果与期望值相等
            tm.assert_frame_equal(result, exp)
            # 断言结果与 (df + 2).where((df + 2) > 8, (df + 2) + 10) 相等
            tm.assert_frame_equal(result, (df + 2).where((df + 2) > 8, (df + 2) + 10))
    
        def test_where_tz_values(self, tz_naive_fixture, frame_or_series):
            # 测试在时区值的 DataFrame 上的 where 方法
            # 创建具有时区的 DatetimeIndex
            obj1 = DataFrame(
                DatetimeIndex(["20150101", "20150102", "20150103"], tz=tz_naive_fixture),
                columns=["date"],
            )
            obj2 = DataFrame(
                DatetimeIndex(["20150103", "20150104", "20150105"], tz=tz_naive_fixture),
                columns=["date"],
            )
            # 创建一个 mask DataFrame
            mask = DataFrame([True, True, False], columns=["date"])
            # 期望结果 DataFrame
            exp = DataFrame(
                DatetimeIndex(["20150101", "20150102", "20150105"], tz=tz_naive_fixture),
                columns=["date"],
            )
            # 根据 frame_or_series 类型调整 obj1、obj2、mask 和 exp
            if frame_or_series is Series:
                obj1 = obj1["date"]
                obj2 = obj2["date"]
                mask = mask["date"]
                exp = exp["date"]
    
            # 使用 where 方法，obj1 作为源，mask 作为条件，obj2 作为替代值
            result = obj1.where(mask, obj2)
            # 断言结果与期望值相等
            tm.assert_equal(exp, result)
    
        def test_df_where_change_dtype(self):
            # 测试 DataFrame 的 where 方法，改变数据类型
            # 创建一个 DataFrame，包含整数数据
            df = DataFrame(np.arange(2 * 3).reshape(2, 3), columns=list("ABC"))
            # 创建一个 mask 数组
            mask = np.array([[True, False, False], [False, False, True]])
    
            # 使用 where 方法，mask 作为条件
            result = df.where(mask)
            # 期望结果 DataFrame，其中 False 的位置为 NaN
            expected = DataFrame(
                [[0, np.nan, np.nan], [np.nan, np.nan, 5]], columns=list("ABC")
            )
    
            # 断言结果与期望值相等
            tm.assert_frame_equal(result, expected)
    
        @pytest.mark.parametrize("kwargs", [{}, {"other": None}])
    def test_df_where_with_category(self, kwargs):
        # GH#16979
        # 创建一个 2x3 的 numpy 数组，数据类型为 int64
        data = np.arange(2 * 3, dtype=np.int64).reshape(2, 3)
        # 用数据创建一个 DataFrame，列名为 A, B, C
        df = DataFrame(data, columns=list("ABC"))
        # 创建一个布尔掩码数组
        mask = np.array([[True, False, False], [False, False, True]])

        # 将列 A, B, C 的数据类型改为 category
        df.A = df.A.astype("category")
        df.B = df.B.astype("category")
        df.C = df.C.astype("category")

        # 使用 where 方法根据掩码过滤 DataFrame，并传递额外的参数
        result = df.where(mask, **kwargs)
        # 创建预期的 DataFrame，其中 A, B, C 列为 Categorical 类型，具体数值和分类信息
        A = pd.Categorical([0, np.nan], categories=[0, 3])
        B = pd.Categorical([np.nan, np.nan], categories=[1, 4])
        C = pd.Categorical([np.nan, 5], categories=[2, 5])
        expected = DataFrame({"A": A, "B": B, "C": C})

        # 断言过滤后的结果与预期的结果相等
        tm.assert_frame_equal(result, expected)

        # 在这里同时检查 Series.where 方法
        result = df.A.where(mask[:, 0], **kwargs)
        # 创建预期的 Series，名称为 A，数据为 Categorical 类型
        expected = Series(A, name="A")

        # 断言 Series 过滤后的结果与预期的结果相等
        tm.assert_series_equal(result, expected)

    def test_where_categorical_filtering(self):
        # GH#22609 验证对具有分类 Series 的 DataFrame 进行过滤操作
        df = DataFrame(data=[[0, 0], [1, 1]], columns=["a", "b"])
        # 将列 'b' 的数据类型改为 category
        df["b"] = df["b"].astype("category")

        # 使用 where 方法根据条件 'a > 0' 过滤 DataFrame
        result = df.where(df["a"] > 0)
        # 将预期的 DataFrame 显式转换为 'float' 类型，避免设置 np.nan 时的隐式转换
        expected = df.copy().astype({"a": "float"})
        expected.loc[0, :] = np.nan

        # 断言过滤后的结果与预期的结果相等
        tm.assert_equal(result, expected)

    def test_where_ea_other(self):
        # GH#38729/GH#38742
        # 创建一个包含列 'A', 'B' 的 DataFrame
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        # 创建一个包含 pd.NA 的 Series
        arr = pd.array([7, pd.NA, 9])
        ser = Series(arr)
        # 创建一个全为 True 的布尔掩码数组
        mask = np.ones(df.shape, dtype=bool)
        mask[1, :] = False

        # 使用 where 方法根据掩码在列 'A', 'B' 上填充 Series 'ser' 的值
        result = df.where(mask, ser, axis=0)
        # 创建预期的 DataFrame，使用 Series 'ser' 填充缺失值
        expected = DataFrame({"A": [1, np.nan, 3], "B": [4, np.nan, 6]})
        tm.assert_frame_equal(result, expected)

        # 创建另一个 Series 'ser2'，并使用 where 方法根据掩码在行上填充它的值
        ser2 = Series(arr[:2], index=["A", "B"])
        expected = DataFrame({"A": [1, 7, 3], "B": [4, np.nan, 6]})
        result = df.where(mask, ser2, axis=1)
        tm.assert_frame_equal(result, expected)

    def test_where_interval_noop(self):
        # GH#44181
        # 创建一个包含 pd.Interval(0, 0) 的 DataFrame
        df = DataFrame([pd.Interval(0, 0)])
        # 使用 where 方法根据非空值过滤 DataFrame
        res = df.where(df.notna())
        # 断言过滤后的结果与原始 DataFrame 相等
        tm.assert_frame_equal(res, df)

        # 创建一个 Series 'ser'，包含 DataFrame 的第一列
        ser = df[0]
        # 使用 where 方法根据非空值过滤 Series
        res = ser.where(ser.notna())
        # 断言过滤后的结果与原始 Series 相等
        tm.assert_series_equal(res, ser)

    def test_where_interval_fullop_downcast(self, frame_or_series):
        # GH#45768
        # 创建包含两个 pd.Interval(0, 0) 的对象（DataFrame 或 Series）
        obj = frame_or_series([pd.Interval(0, 0)] * 2)
        # 创建另一个 Series 'other'，数据类型为 object
        other = frame_or_series([1.0, 2.0], dtype=object)
        # 使用 where 方法根据条件在对象 'obj' 上填充 'other' 的值
        res = obj.where(~obj.notna(), other)
        # 断言填充后的结果与 'other' 相等
        tm.assert_equal(res, other)

        # 验证使用 inplace=True 时的警告信息
        with tm.assert_produces_warning(
            FutureWarning, match="Setting an item of incompatible dtype"
        ):
            # 在 'obj' 上使用 mask 方法，根据条件填充 'other' 的值
            obj.mask(obj.notna(), other, inplace=True)
        # 断言填充后的 'obj' 与 'other' 相等
        tm.assert_equal(obj, other.astype(object))
    @pytest.mark.parametrize(
        "dtype",
        [
            "timedelta64[ns]",  # 参数化测试，传入不同的数据类型
            "datetime64[ns]",   # 参数化测试，传入不同的数据类型
            "datetime64[ns, Asia/Tokyo]",  # 参数化测试，传入带时区信息的日期时间数据类型
        ],
    )
    def test_where_datetimelike_noop(self, dtype):
        # GH#45135, 类似于 GH#44181，用于时间段（Period）不引发空操作异常
        # 对于 td64/dt64/dt64tz 类型，我们已经不会引发异常，但同时也在检查避免不必要地向对象类型转换
        ser = Series(np.arange(3) * 10**9, dtype=np.int64).astype(dtype)
        df = ser.to_frame()
        mask = np.array([False, False, False])

        res = ser.where(~mask, "foo")  # 使用 where 方法进行条件筛选
        tm.assert_series_equal(res, ser)  # 断言结果与原始序列相等

        mask2 = mask.reshape(-1, 1)
        res2 = df.where(~mask2, "foo")  # 使用 where 方法进行条件筛选
        tm.assert_frame_equal(res2, df)  # 断言结果与原始数据框相等

        res3 = ser.mask(mask, "foo")  # 使用 mask 方法进行条件筛选
        tm.assert_series_equal(res3, ser)  # 断言结果与原始序列相等

        res4 = df.mask(mask2, "foo")  # 使用 mask 方法进行条件筛选
        tm.assert_frame_equal(res4, df)  # 断言结果与原始数据框相等
        expected = DataFrame(4, index=df.index, columns=df.columns)

        # 与 where 方法不同，Block.putmask 不会降级数据类型
        with tm.assert_produces_warning(
            FutureWarning, match="Setting an item of incompatible dtype"
        ):
            df.mask(~mask2, 4, inplace=True)  # 使用 mask 方法进行条件筛选并就地修改
        tm.assert_frame_equal(df, expected.astype(object))  # 断言结果与预期的对象类型转换后的数据框相等
# 测试函数，验证整数向下转换过时的行为
def test_where_int_downcasting_deprecated():
    # 创建一个包含6个元素的numpy数组，转换为int16类型，然后reshape成3行2列的二维数组
    arr = np.arange(6).astype(np.int16).reshape(3, 2)
    # 用数组创建一个DataFrame对象
    df = DataFrame(arr)

    # 创建一个与arr形状相同的全False布尔掩码数组
    mask = np.zeros(arr.shape, dtype=bool)
    # 将每行的第一个元素设为True
    mask[:, 0] = True

    # 使用df和mask对象调用where方法，将符合条件的元素替换为2^17
    res = df.where(mask, 2**17)

    # 创建一个期望的DataFrame对象，第一列与arr的第一列相同，第二列是3个值为2^17的int32类型元素
    expected = DataFrame({0: arr[:, 0], 1: np.array([2**17] * 3, dtype=np.int32)})
    # 断言res与期望的DataFrame对象相等
    tm.assert_frame_equal(res, expected)


# 测试函数，验证where方法在无操作时是否复制对象
def test_where_copies_with_noop(frame_or_series):
    # 调用frame_or_series函数创建一个包含[1, 2, 3, 4]的对象
    result = frame_or_series([1, 2, 3, 4])
    # 创建一个result对象的拷贝
    expected = result.copy()
    # 获取result对象的第一个列（如果是DataFrame对象）或者整个result对象（如果是Series对象）
    col = result[0] if frame_or_series is DataFrame else result

    # 使用col < 5的条件调用where方法，将符合条件的元素乘以2
    where_res = result.where(col < 5)
    where_res *= 2

    # 断言result对象与其拷贝expected相等
    tm.assert_equal(result, expected)

    # 使用col > 5的条件调用where方法，将不符合条件的元素替换为[1, 2, 3, 4]，再乘以2
    where_res = result.where(col > 5, [1, 2, 3, 4])
    where_res *= 2

    # 再次断言result对象与其拷贝expected相等
    tm.assert_equal(result, expected)


# 测试函数，验证字符串类型的where方法行为
def test_where_string_dtype(frame_or_series):
    # 使用frame_or_series函数创建一个字符串类型的对象obj
    obj = frame_or_series(
        ["a", "b", "c", "d"], index=["id1", "id2", "id3", "id4"], dtype=StringDtype()
    )
    # 使用frame_or_series函数创建一个字符串类型的过滤对象filtered_obj
    filtered_obj = frame_or_series(
        ["b", "c"], index=["id2", "id3"], dtype=StringDtype()
    )
    # 创建一个布尔Series对象，表示哪些元素需要被保留
    filter_ser = Series([False, True, True, False])

    # 使用filter_ser调用obj的where方法，根据条件筛选obj中的元素，并用filtered_obj替换不符合条件的元素
    result = obj.where(filter_ser, filtered_obj)
    # 创建一个期望的字符串类型DataFrame对象，使用pd.NA表示缺失值
    expected = frame_or_series(
        [pd.NA, "b", "c", pd.NA],
        index=["id1", "id2", "id3", "id4"],
        dtype=StringDtype(),
    )
    # 断言result与期望的DataFrame对象相等
    tm.assert_equal(result, expected)

    # 使用filter_ser的相反条件调用obj的mask方法，将不符合条件的元素替换为filtered_obj，并乘以2
    result = obj.mask(~filter_ser, filtered_obj)
    # 再次断言result与期望的DataFrame对象相等
    tm.assert_equal(result, expected)

    # 在原地使用filter_ser的相反条件调用obj的mask方法，将不符合条件的元素替换为filtered_obj
    obj.mask(~filter_ser, filtered_obj, inplace=True)
    # 最后再次断言result与期望的DataFrame对象相等
    tm.assert_equal(result, expected)


# 测试函数，验证布尔类型比较时的where方法行为
def test_where_bool_comparison():
    # 创建一个包含布尔值的DataFrame对象
    df_mask = DataFrame(
        {"AAA": [True] * 4, "BBB": [False] * 4, "CCC": [True, False, True, False]}
    )
    # 使用df_mask == False条件调用where方法，将不符合条件的元素替换为NaN
    result = df_mask.where(df_mask == False)  # noqa: E712
    # 创建一个期望的DataFrame对象，使用np.nan表示NaN
    expected = DataFrame(
        {
            "AAA": np.array([np.nan] * 4, dtype=object),
            "BBB": [False] * 4,
            "CCC": [np.nan, False, np.nan, False],
        }
    )
    # 断言result与期望的DataFrame对象相等
    tm.assert_frame_equal(result, expected)


# 测试函数，验证where方法中None和NaN的转换行为
def test_where_none_nan_coerce():
    # 创建一个包含时间戳和NaN的DataFrame对象
    expected = DataFrame(
        {
            "A": [Timestamp("20130101"), pd.NaT, Timestamp("20130103")],
            "B": [1, 2, np.nan],
        }
    )
    # 使用expected.notnull()条件调用where方法，将不符合条件的元素替换为None
    result = expected.where(expected.notnull(), None)
    # 断言result与期望的DataFrame对象相等
    tm.assert_frame_equal(result, expected)


# 测试函数，验证重复索引和混合数据类型的where方法行为
def test_where_duplicate_axes_mixed_dtypes():
    # 创建一个包含重复列名的DataFrame对象
    result = DataFrame(data=[[0, np.nan]], columns=Index(["A", "A"]))
    # 获取result对象的索引和列名
    index, columns = result.axes
    # 创建一个与result相同形状的布尔DataFrame对象
    mask = DataFrame(data=[[True, True]], columns=columns, index=index)
    # 使用astype(object)方法调用where方法，根据条件筛选元素
    a = result.astype(object).where(mask)
    # 使用astype("f8")方法调用where方法，根据条件筛选元素
    b = result.astype("f8").where(mask)
    # 使用.T转置后调用where方法，根据条件筛选元素再转置
    c = result.T.where(mask.T).T
    # 直接调用where方法，根据条件筛选元素
    d = result.where(mask)  # used to fail with "cannot reindex from a duplicate axis"
    # 断言a和b的f8类型相等
    tm.assert_frame_equal(a.astype("f8"), b.astype("f8"))
    # 断言b和c的f8类型相等
    tm.assert_frame_equal(b.astype("f8"), c.astype("f8"))
    # 使用测试框架中的函数来比较两个数据框架（DataFrame），确保它们在内容上完全相等。
    tm.assert_frame_equal(c.astype("f8"), d.astype("f8"))
def test_where_columns_casting():
    # GH 42295
    # 创建一个包含两列的 DataFrame，其中一列包含浮点数，另一列包含整数和 NaN
    df = DataFrame({"a": [1.0, 2.0], "b": [3, np.nan]})
    # 复制原始 DataFrame
    expected = df.copy()
    # 使用 pd.notnull(df) 条件进行 DataFrame 的条件替换，将 NaN 替换为 None
    result = df.where(pd.notnull(df), None)
    # 确保数据类型不发生变化
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize("as_cat", [True, False])
def test_where_period_invalid_na(frame_or_series, as_cat, request):
    # GH#44697
    # 创建一个包含三个日期周期的索引
    idx = pd.period_range("2016-01-01", periods=3, freq="D")
    if as_cat:
        # 如果 as_cat 为 True，则将索引转换为分类类型
        idx = idx.astype("category")
    # 使用 frame_or_series 函数创建对象，传入上述索引
    obj = frame_or_series(idx)

    # NA 值，不应转换为 Period 类型
    tdnat = pd.NaT.to_numpy("m8[ns]")

    # 创建一个布尔掩码数组
    mask = np.array([True, True, False], ndmin=obj.ndim).T

    if as_cat:
        # 如果 as_cat 为 True，则定义特定错误消息
        msg = (
            r"Cannot setitem on a Categorical with a new category \(NaT\), "
            "set the categories first"
        )
    else:
        # 否则定义另一种错误消息
        msg = "value should be a 'Period'"

    if as_cat:
        # 确保在特定条件下抛出 TypeError 异常，匹配特定错误消息
        with pytest.raises(TypeError, match=msg):
            obj.where(mask, tdnat)

        with pytest.raises(TypeError, match=msg):
            obj.mask(mask, tdnat)

        with pytest.raises(TypeError, match=msg):
            obj.mask(mask, tdnat, inplace=True)

    else:
        # 对于非分类类型，验证条件替换后的期望结果与实际结果是否一致
        expected = obj.astype(object).where(mask, tdnat)
        result = obj.where(mask, tdnat)
        tm.assert_equal(result, expected)

        expected = obj.astype(object).mask(mask, tdnat)
        result = obj.mask(mask, tdnat)
        tm.assert_equal(result, expected)

        # 确保在 inplace=True 时产生 FutureWarning 警告，并验证结果
        with tm.assert_produces_warning(
            FutureWarning, match="Setting an item of incompatible dtype"
        ):
            obj.mask(mask, tdnat, inplace=True)
        tm.assert_equal(obj, expected)


def test_where_nullable_invalid_na(frame_or_series, any_numeric_ea_dtype):
    # GH#44697
    # 创建一个包含整数的数组，使用指定的数据类型
    arr = pd.array([1, 2, 3], dtype=any_numeric_ea_dtype)
    # 使用 frame_or_series 函数创建对象，传入上述数组
    obj = frame_or_series(arr)

    # 创建一个布尔掩码数组
    mask = np.array([True, True, False], ndmin=obj.ndim).T

    # 定义特定的错误消息模式
    msg = r"Invalid value '.*' for dtype (U?Int|Float)\d{1,2}"

    for null in tm.NP_NAT_OBJECTS + [pd.NaT]:
        # NaT 是一个不应转换为 pd.NA 类型的 NA 值
        # 确保在特定条件下抛出 TypeError 异常，匹配特定错误消息
        with pytest.raises(TypeError, match=msg):
            obj.where(mask, null)

        with pytest.raises(TypeError, match=msg):
            obj.mask(mask, null)


@given(data=OPTIONAL_ONE_OF_ALL)
def test_where_inplace_casting(data):
    # GH 22051
    # 创建一个包含数据的 DataFrame
    df = DataFrame({"a": data})
    # 使用 pd.notnull(df) 条件进行 DataFrame 的条件替换，将 NaN 替换为 None，并复制结果
    df_copy = df.where(pd.notnull(df), None).copy()
    # 使用 inplace=True 参数进行原地替换
    df.where(pd.notnull(df), None, inplace=True)
    # 确保替换后的结果与复制的结果相等
    tm.assert_equal(df, df_copy)


def test_where_downcast_to_td64():
    # 创建一个包含整数的序列
    ser = Series([1, 2, 3])

    # 创建一个布尔掩码数组
    mask = np.array([False, False, False])

    # 创建一个时间差序列
    td = pd.Timedelta(days=1)
    expected = Series([td, td, td], dtype="m8[ns]")

    # 使用条件替换，将符合条件的位置替换为 td
    res2 = ser.where(mask, td)
    # 将期望结果转换为 object 类型并进行比较
    expected2 = expected.astype(object)
    tm.assert_series_equal(res2, expected2)
# 定义一个函数用于检查 DataFrame 的 where 和 mask 方法的等效性
def _check_where_equivalences(df, mask, other, expected):
    # 使用 mask 来选择 DataFrame 中的元素，将不符合条件的替换为 other，返回新的 DataFrame
    res = df.where(mask, other)
    # 断言两个 DataFrame 是否相等，如果不相等则抛出异常
    tm.assert_frame_equal(res, expected)

    # 使用 ~mask 来选择 DataFrame 中的元素，将符合条件的替换为 other，返回新的 DataFrame
    res = df.mask(~mask, other)
    # 断言两个 DataFrame 是否相等，如果不相等则抛出异常
    tm.assert_frame_equal(res, expected)

    # 注意：使用 inplace=True 的 mask 操作需要进行额外的处理，因为 Block.putmask 不会降级数据类型。
    # expected 在 test_where_dt64_2d 中的修改是针对 test_where_dt64_2d 中的情况。
    df = df.copy()
    # 使用 inplace=True 的 mask 操作，将符合条件的元素替换为 other
    df.mask(~mask, other, inplace=True)
    # 如果 mask 不全为真，则对 expected 的 "A" 列进行类型转换为 object 类型
    if not mask.all():
        expected = expected.copy()
        expected["A"] = expected["A"].astype(object)
    # 断言两个 DataFrame 是否相等，如果不相等则抛出异常
    tm.assert_frame_equal(df, expected)


# 定义一个测试函数，用于测试 DataFrame 的 where 方法处理 datetime64 类型的二维数组的情况
def test_where_dt64_2d():
    # 创建一个日期范围对象
    dti = date_range("2016-01-01", periods=6)
    # 将日期范围的数据转换成 3x2 的二维数组
    dta = dti._data.reshape(3, 2)
    # 计算 other 数组，与第一个元素的差值
    other = dta - dta[0, 0]

    # 创建一个 DataFrame 对象，列名为 "A" 和 "B"
    df = DataFrame(dta, columns=["A", "B"])

    # 创建一个布尔掩码数组，标识 DataFrame 中的缺失值位置
    mask = np.asarray(df.isna()).copy()
    mask[:, 1] = True

    # 设置部分列的一部分值为 True
    mask[1, 0] = True
    # 创建期望的 DataFrame 对象，"A" 列的期望值替换成了 other 数组的部分值
    expected = DataFrame(
        {
            "A": np.array([other[0, 0], dta[1, 0], other[2, 0]], dtype=object),
            "B": dta[:, 1],
        }
    )
    # 运行时断言，在产生警告时会抛出 FutureWarning 异常
    with tm.assert_produces_warning(
        FutureWarning, match="Setting an item of incompatible dtype"
    ):
        # 调用 _check_where_equivalences 函数进行测试
        _check_where_equivalences(df, mask, other, expected)

    # 将掩码数组中的所有值设为 True，不做任何替换
    mask[:] = True
    expected = df
    # 调用 _check_where_equivalences 函数进行测试
    _check_where_equivalences(df, mask, other, expected)


# 定义一个测试函数，用于测试 DataFrame 的 where 方法产生 numpy 数据类型的条件处理情况
def test_where_producing_ea_cond_for_np_dtype():
    # 创建一个 DataFrame 对象，包含 "a" 列和 "b" 列
    df = DataFrame({"a": Series([1, pd.NA, 2], dtype="Int64"), "b": [1, 2, 3]})
    # 使用 lambda 函数处理 DataFrame 中的每个元素，将大于 1 的元素置为 True
    result = df.where(lambda x: x.apply(lambda y: y > 1, axis=1))
    # 创建期望的 DataFrame 对象，"a" 列中的值大于 1 的元素置为 pd.NA
    expected = DataFrame(
        {"a": Series([pd.NA, pd.NA, 2], dtype="Int64"), "b": [np.nan, 2, 3]}
    )
    # 断言两个 DataFrame 是否相等，如果不相等则抛出异常
    tm.assert_frame_equal(result, expected)


# 使用参数化测试装饰器，测试 DataFrame 的 where 方法处理整数溢出的情况
@pytest.mark.parametrize(
    "replacement", [0.001, True, "snake", None, datetime(2022, 5, 4)]
)
def test_where_int_overflow(replacement, using_infer_string, request):
    # 创建一个包含浮点数、整数和字符串的 DataFrame 对象
    df = DataFrame([[1.0, 2e25, "nine"], [np.nan, 0.1, None]])
    # 如果使用了 infer_string，并且 replacement 不是 None 或 "snake"，则添加标记为 xfail
    if using_infer_string and replacement not in (None, "snake"):
        request.node.add_marker(
            pytest.mark.xfail(reason="Can't set non-string into string column")
        )
    # 使用 replacement 替换 DataFrame 中的缺失值
    result = df.where(pd.notnull(df), replacement)
    # 创建期望的 DataFrame 对象，将缺失值替换为 replacement
    expected = DataFrame([[1.0, 2e25, "nine"], [replacement, 0.1, replacement]])
    # 断言两个 DataFrame 是否相等，如果不相等则抛出异常
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，测试 DataFrame 的 where 方法中 inplace=True 且没有 other 参数的情况
def test_where_inplace_no_other():
    # 创建一个 DataFrame 对象，包含 "a" 列和 "b" 列
    df = DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    # 创建一个布尔掩码 DataFrame 对象
    cond = DataFrame({"a": [True, False], "b": [False, True]})
    # 使用 where 方法处理 DataFrame，根据 cond 的值替换 df 中的元素
    df.where(cond, inplace=True)
    # 创建期望的 DataFrame 对象，"a" 列中值为 False 的元素被替换为 NaN，"b" 列中值为 False 的元素被替换为 NaN
    expected = DataFrame({"a": [1, np.nan], "b": [np.nan, "y"]})
    # 断言两个 DataFrame 是否相等，如果不相等则抛出异常
    tm.assert_frame_equal(df, expected)
```